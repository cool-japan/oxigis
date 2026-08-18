//! Attribute table panel: the selected local vector layer's real features,
//! virtualized.
//!
//! The data model lives in [`crate::attribute_table`]; this module is the egui
//! rendering of it plus the per-frame state (sort, selection, scroll) the
//! renderer needs.
//!
//! # Why not `Table::render_egui`
//!
//! This panel uses `oxiui-table`'s *model* — [`Table`] for the runtime column
//! widths, resize clamping and sort toggle, [`oxiui_table::Cell`] for typed
//! values and its `compare` for numeric-aware ordering, [`oxiui_table::ColumnDef`],
//! [`RowSource`],
//! [`TableEvent`] — but **not** its `egui-table` backend's
//! [`Table::render_egui`], and not [`Table::filtered_sorted_indices`],
//! [`Table::sorted_indices`] or [`Table::materialize_visible`] either.
//!
//! Those entry points are `O(rows)` *per frame*, which defeats the very
//! virtualization they sit behind: `render_egui` materializes a `Vec<Cell>` for
//! every row of the dataset before `ScrollArea::show_rows` ever picks the
//! visible handful out of it (oxiui-table 0.2.1, `egui_table.rs` — the
//! `rows_data` collect), computes a row background per row, and re-runs
//! `sorted_indices` — itself another full materialization, in `sort.rs` — on
//! every frame a sort is active. On the 100k-feature datasets a GIS attribute
//! table is expected to open, that is on the order of a million `String`
//! allocations per frame.
//!
//! So the draw loop here is our own `egui::ScrollArea::show_rows` (no new
//! dependency: it is egui's own row virtualization), reading cells through
//! [`FeatureRowSource::cell_at`], which fetches one cell without building its
//! row. The display order is a `Vec<usize>` recomputed only when the sort or
//! the filter text changes, never per frame. Cost per frame is proportional to
//! the *visible* rows and columns, not to the dataset.
//!
//! # Selection
//!
//! [`AttributeTablePanel::selected_feature`] is the feature's index in the
//! **source** collection, not the visible row index — the two differ the moment
//! the user sorts. That index is what
//! [`crate::local_vector::feature_collection_to_tile`] writes as the drawn
//! feature's `MvtFeature::id`, so it is the handle the later map-highlight work
//! (blueprint §5.2) will address the geometry by. The emitted
//! [`TableEvent::RowSelected`] keeps oxiui-table's own documented meaning
//! (visible row index). An anchor or co-selection index the filter box
//! currently excludes from [`AttributeTablePanel::display_order`] simply is
//! not drawn this frame — it stays selected, ready to reappear the moment the
//! filter no longer excludes it, rather than being dropped.
//!
//! # Filtering and export
//!
//! The filter box is a case-insensitive substring match against every
//! column's [`FeatureRowSource::cell_text`]
//! (`FeatureRowSource::row_contains`), applied *before* the sort in
//! `BoundLayer::sync_order`: a total order restricted to a subset agrees
//! with the same order computed over the subset alone, so filtering first and
//! sorting the (usually smaller) survivors is equivalent to the reverse and
//! cheaper. Like the sort, it is recomputed on the frame the filter text
//! changes and never per frame — see
//! [`AttributeTablePanel::set_filter_text`]. "Copy CSV"
//! ([`egui::Context::copy_text`]) and
//! [`AttributeTablePanel::export_csv_bytes`] (a file-save seam this crate
//! never itself writes to disk) both walk the same filtered-and-sorted
//! [`AttributeTablePanel::display_order`] through
//! [`FeatureRowSource::to_csv`], so what is copied or exported always matches
//! what is currently drawn.
//!
//! The toolbar's **Export CSV** button is the file-writing twin of Copy CSV.
//! It captures the CSV *at click time* into
//! [`AttributeTablePanel::take_export_request`] rather than raising a flag for
//! the app to re-derive from, and that is not a stylistic choice: the app
//! drains the request after this panel has drawn, by which point a sort, a
//! filter edit or a re-bind could have moved the rows underneath it. Capturing
//! at the click is what makes "you export what you were looking at" true
//! rather than nearly true.

use std::collections::HashSet;
use std::sync::Arc;

use egui::{Align2, Rect, Sense, Ui, Vec2, pos2, vec2};
use oxigeo::geojson::types::FeatureCollection;
use oxigis_core::LayerId;
use oxiui_table::{RowSource, SortDirection, SortState, Table, TableEvent};

use crate::attribute_table::{FeatureRowSource, SortKey};
use crate::ui_glyphs::{ELLIPSIS, MIDDLE_DOT, MOVE_DOWN, MOVE_UP};

/// Text shown in place of the table when the selection is not a local vector
/// layer whose features are loaded.
pub const NO_LAYER_PLACEHOLDER: &str = "Select a local layer to see its attributes.";

/// Text shown when a local layer is selected but its features have not arrived
/// yet — a project-load path reference the shell has still to read.
pub const PENDING_PLACEHOLDER: &str = "Loading the layer's features\u{2026}";

/// Width, in logical pixels, of the drag strip at a header cell's right edge.
const RESIZE_HANDLE_WIDTH: f32 = 6.0;

/// Horizontal padding inside a cell, in logical pixels.
const CELL_PAD_X: f32 = 6.0;

/// The dataset an [`AttributeTablePanel`] is currently bound to.
///
/// Keyed by layer *and* by the collection's [`Arc`] identity: re-hydrating a
/// path-referenced layer replaces the features under an unchanged
/// [`LayerId`], and the derived schema and sort order must not survive that.
/// Pointer identity is exact and free; the layer's style `generation` would be
/// wrong here (it tracks style edits, not data).
struct BoundLayer {
    /// Which project layer these rows belong to.
    id: LayerId,
    /// The collection the schema and order were derived from.
    features: Arc<FeatureCollection>,
    /// The rows, and the column layout derived from them.
    table: Table<FeatureRowSource>,
    /// Source row indices in display order — the current sort and filter
    /// applied, filter first (see [`Self::sync_order`]).
    order: Vec<usize>,
    /// The `(sort, filter revision)` `order` was built for. `sort = None`
    /// means document order; the filter revision is the panel's own change
    /// counter (`AttributeTablePanel::filter_revision`), compared as an
    /// opaque `u64` so an unchanged filter costs one integer comparison
    /// rather than re-touching the filter text or the rows.
    order_key: (Option<(usize, SortDirection)>, u64),
}

impl BoundLayer {
    /// Binds a freshly selected dataset, in document order.
    fn new(id: LayerId, features: Arc<FeatureCollection>) -> Self {
        let source = FeatureRowSource::new(Arc::clone(&features));
        let order = (0..source.row_count()).collect();
        Self {
            id,
            features,
            table: Table::new(source),
            order,
            order_key: (None, 0),
        }
    }

    /// Whether this binding still describes `id`'s current features.
    fn matches(&self, id: LayerId, features: &Arc<FeatureCollection>) -> bool {
        self.id == id && Arc::ptr_eq(&self.features, features)
    }

    /// Rebuilds [`Self::order`] if the sort or the filter no longer match
    /// what it was built for. A no-op on the frames — nearly all of them —
    /// where neither changed.
    fn sync_order(&mut self, filter_text: &str, filter_revision: u64) {
        let wanted_sort = self
            .table
            .sort_state()
            .map(|state: SortState| (state.column, state.direction));
        let wanted = (wanted_sort, filter_revision);
        if wanted == self.order_key {
            return;
        }
        let source = self.table.source();
        let count = source.row_count();

        // Filter before sorting: a total order restricted to a subset agrees
        // with the same order computed over the subset alone, so the result
        // is identical either way, but materializing sort keys for only the
        // survivors is cheaper, and it is the filter, not the sort, that is
        // expected to shrink `count`.
        let needle = filter_text.trim().to_lowercase();
        let mut order: Vec<usize> = if needle.is_empty() {
            (0..count).collect()
        } else {
            (0..count)
                .filter(|&row| source.row_contains(row, &needle))
                .collect()
        };

        if let Some((column, direction)) = wanted_sort
            && direction != SortDirection::None
        {
            // Materialize the sort key once per surviving row rather than
            // once per comparison — a comparison sort asks for each key
            // `O(log n)` times — and borrow property text instead of owning
            // it: `cell_at`'s `Cell::Text` would otherwise allocate one
            // `String` per row just to compare them.
            let mut keyed: Vec<(SortKey<'_>, usize)> = order
                .iter()
                .map(|&row| (source.cell_sort_key(row, column), row))
                .collect();
            keyed.sort_by(|(left, _), (right, _)| {
                let ordering = left.compare(right);
                if direction == SortDirection::Descending {
                    ordering.reverse()
                } else {
                    ordering
                }
            });
            order = keyed.into_iter().map(|(_, row)| row).collect();
        }
        self.order = order;
        self.order_key = wanted;
    }

    /// The current display order, rendered as CSV. Shared by the panel's
    /// "Copy CSV" button and [`AttributeTablePanel::export_csv`], so both
    /// always describe the same rows.
    fn csv(&self) -> String {
        self.table.source().to_csv(&self.order)
    }
}

/// What one frame of the panel's toolbar asked for.
///
/// A struct rather than the bare `bool` this used to return, because the
/// Export button's answer is a *document* and not a flag — see the module
/// docs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ToolbarOutcome {
    /// Whether the filter box's text changed this frame.
    filter_changed: bool,
    /// The CSV the Export button captured, if it was pressed.
    export: Option<String>,
}

/// State the attribute-table panel keeps across frames.
///
/// Everything here is caller-persisted for the same reason `oxiui-table`'s own
/// renderer state is: the sort, the column widths, the horizontal scroll
/// position and the derived schema are all user- or data-dependent, and
/// rebuilding them each frame would discard the user's sort and resize on the
/// very next one — and re-derive the schema of a 100k-feature dataset while
/// doing it.
#[derive(Default)]
pub struct AttributeTablePanel {
    /// The dataset currently shown, if any.
    bound: Option<BoundLayer>,
    /// Index of the selected feature in the **source** collection.
    selected_feature: Option<usize>,
    /// Co-selected features (a map multi-selection), source indices, anchor
    /// excluded. A [`HashSet`] rather than a `Vec`: the body draw loop tests
    /// membership once per visible row every frame the selection is live, and
    /// a `Vec` scan there is `O(visible rows × |selected_extra|)` — unbounded
    /// in the second factor, since a marquee select can cover the whole
    /// dataset.
    selected_extra: HashSet<usize>,
    /// Horizontal scroll offset shared between the header and the body, in
    /// logical pixels.
    h_offset: f32,
    /// The filter box's current text; empty disables filtering. Reset with
    /// the rest of the per-binding state in [`Self::bind`].
    filter_text: String,
    /// Bumped by [`Self::set_filter_text`] (and the filter box's own change
    /// handling in [`Self::draw_toolbar`]) whenever [`Self::filter_text`]'s
    /// value actually changes. Part of [`BoundLayer::order_key`], so an
    /// unchanged filter is a `u64` comparison rather than a re-filter.
    filter_revision: u64,
    /// Events emitted during the last drawn frame.
    events: Vec<TableEvent>,
    /// The CSV the toolbar's Export button captured, waiting for the app to
    /// take it — see the module docs on why the *document* is parked here
    /// rather than a flag.
    ///
    /// At most one: a second click before the app drains replaces the first,
    /// which is right because both describe the same table and the newer one
    /// describes it more recently.
    export_request: Option<String>,
}

impl AttributeTablePanel {
    /// Draws `features` (the local vector layer `id`, displayed as `name`) and
    /// returns the [`TableEvent`]s emitted this frame.
    ///
    /// Binding is cached: passing the same layer and the same collection again
    /// reuses the derived schema, the column widths and the sort order.
    pub fn show(
        &mut self,
        ui: &mut Ui,
        id: LayerId,
        name: &str,
        features: &Arc<FeatureCollection>,
    ) -> &[TableEvent] {
        self.events.clear();
        self.bind(id, features);
        let Some(bound) = self.bound.as_mut() else {
            return &self.events;
        };
        bound.sync_order(&self.filter_text, self.filter_revision);

        let feature_count = bound.table.source().row_count();
        let omitted = bound.table.source().schema().omitted_columns();
        ui.horizontal(|ui| {
            ui.strong(name);
            ui.label(match feature_count {
                1 => "1 feature".to_string(),
                count => format!("{count} features"),
            });
            if omitted > 0 {
                ui.weak(format!("+{omitted} more columns not shown"));
            }
            let selected = self.selected_extra.len() + usize::from(self.selected_feature.is_some());
            if selected > 1 {
                ui.weak(format!("{MIDDLE_DOT} {selected} selected on the map"));
            }
        });

        let toolbar = Self::draw_toolbar(ui, bound, &mut self.filter_text);
        if toolbar.filter_changed {
            self.filter_revision = self.filter_revision.wrapping_add(1);
        }
        if let Some(csv) = toolbar.export {
            self.export_request = Some(csv);
        }

        Self::draw_header(ui, bound, &mut self.h_offset, &mut self.events);
        Self::draw_body(
            ui,
            bound,
            &mut self.h_offset,
            &mut self.selected_feature,
            &self.selected_extra,
            &mut self.events,
        );
        &self.events
    }

    /// Draws the panel's placeholder text and unbinds any dataset.
    ///
    /// `pending` distinguishes "the selection is not a local vector layer" from
    /// "it is one, but its features have not been read yet".
    pub fn show_placeholder(&mut self, ui: &mut Ui, pending: bool) {
        self.events.clear();
        self.bound = None;
        self.selected_feature = None;
        ui.weak(if pending {
            PENDING_PLACEHOLDER
        } else {
            NO_LAYER_PLACEHOLDER
        });
    }

    /// The selected feature's index in the source collection, if any.
    #[must_use]
    pub fn selected_feature(&self) -> Option<usize> {
        self.selected_feature
    }

    /// The layer whose features are currently shown, if any.
    #[must_use]
    pub fn bound_layer(&self) -> Option<LayerId> {
        self.bound.as_ref().map(|bound| bound.id)
    }

    /// Total number of rows in the bound dataset (`0` when nothing is bound).
    ///
    /// Unaffected by the filter box: source indices — the ones
    /// [`Self::select_source_feature`] bound-checks against and
    /// [`crate::local_vector::feature_collection_to_tile`] writes as
    /// `MvtFeature::id` — must stay valid against the full dataset regardless
    /// of what the filter currently excludes from view. Was always equal to
    /// [`Self::display_order`]'s length before the filter box existed; it no
    /// longer is whenever the filter excludes any row.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.bound
            .as_ref()
            .map_or(0, |bound| bound.table.source().row_count())
    }

    /// Number of columns currently shown (`0` when nothing is bound).
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.bound
            .as_ref()
            .map_or(0, |bound| bound.table.source().column_defs().len())
    }

    /// Source row indices in display order — the current sort *and* filter
    /// applied. Empty when nothing is bound, and no longer necessarily
    /// [`Self::row_count`] long: shorter whenever the filter box excludes any
    /// row. Exposed for tests, for [`Self::export_csv`], and for the later
    /// map-highlight work, which needs the same visible→source mapping.
    #[must_use]
    pub fn display_order(&self) -> &[usize] {
        self.bound.as_ref().map_or(&[], |bound| &bound.order)
    }

    /// The filter box's current text (empty = no filter).
    #[must_use]
    pub fn filter_text(&self) -> &str {
        &self.filter_text
    }

    /// Sets the filter box's text, re-syncing [`Self::display_order`]
    /// immediately — mirroring [`Self::toggle_sort`], which re-syncs eagerly
    /// for the same reason: both are the direct, `Ui`-free entry point a
    /// caller (or a test) uses without going through [`Self::show`] first. A
    /// no-op if `text` equals the current filter, so setting it repeatedly to
    /// the same value never forces a re-filter. The filter *box*'s own edits
    /// go through a separate path inside `show` that bumps the same revision
    /// counter but, like a header click, takes effect on the next frame.
    pub fn set_filter_text(&mut self, text: impl Into<String>) {
        let text = text.into();
        if self.filter_text != text {
            self.filter_text = text;
            self.filter_revision = self.filter_revision.wrapping_add(1);
            if let Some(bound) = self.bound.as_mut() {
                bound.sync_order(&self.filter_text, self.filter_revision);
            }
        }
    }

    /// How many features are highlighted as a co-selection alongside
    /// [`Self::selected_feature`] — [`Self::select_source_features`]'s
    /// `extra`, anchor and out-of-range indices already excluded,
    /// **duplicates collapsed**: `extra` describes a *set* of co-selected
    /// features, so passing the same source index twice reports one feature
    /// selected, not two.
    #[must_use]
    pub fn co_selected_count(&self) -> usize {
        self.selected_extra.len()
    }

    /// The current display order, rendered as CSV — the header from the
    /// derived schema, then one row per currently filtered-and-sorted
    /// feature. Empty when nothing is bound. See the module-level
    /// "Filtering and export" section.
    #[must_use]
    pub fn export_csv(&self) -> String {
        self.bound
            .as_ref()
            .map_or_else(String::new, BoundLayer::csv)
    }

    /// [`Self::export_csv`], UTF-8 encoded.
    #[must_use]
    pub fn export_csv_bytes(&self) -> Vec<u8> {
        self.export_csv().into_bytes()
    }

    /// Takes the CSV the toolbar's Export button captured, if it was pressed
    /// since the last call — take-once, like every other cross-frame seam in
    /// this crate.
    ///
    /// The document itself, not a flag: it was rendered on the frame of the
    /// click, from the rows that were then on screen. See the module docs.
    pub fn take_export_request(&mut self) -> Option<String> {
        self.export_request.take()
    }

    /// Whether the Export button has been pressed and not yet drained.
    #[must_use]
    pub fn export_requested(&self) -> bool {
        self.export_request.is_some()
    }

    /// Parks `csv` as though the Export button had just been pressed.
    ///
    /// The `Ui`-free entry point the button's own path goes through, so a
    /// caller (or a test) can exercise the drain without simulating a click on
    /// a widget whose position depends on the panel's layout. Replaces any
    /// undrained request, exactly as a second click would.
    pub fn park_export_request(&mut self, csv: impl Into<String>) {
        self.export_request = Some(csv.into());
    }

    /// Sorts by `column`, cycling none → ascending → descending → none, exactly
    /// as [`Table::toggle_sort`] defines it. Ignored when nothing is bound.
    pub fn toggle_sort(&mut self, column: usize) {
        if let Some(bound) = self.bound.as_mut() {
            bound.table.toggle_sort(column);
            bound.sync_order(&self.filter_text, self.filter_revision);
        }
    }

    /// Selects the feature shown in visible row `visible_row`, or clears the
    /// selection when the row does not exist.
    pub fn select_visible_row(&mut self, visible_row: usize) {
        self.selected_feature = self
            .bound
            .as_ref()
            .and_then(|bound| bound.order.get(visible_row).copied());
    }

    /// Selects the feature at `source` in the **source** collection, ignoring
    /// the display order entirely; [`None`], or an index past the end, clears
    /// the selection.
    ///
    /// The counterpart of [`Self::select_visible_row`], which takes a
    /// *visible* row. The two coordinate systems differ the moment a sort is
    /// active, so they get two clearly named entry points and never one:
    /// feeding a source index into `select_visible_row` silently selects the
    /// wrong feature. Map-side code (which addresses features by their source
    /// index, the same number [`Self::selected_feature`] reports and
    /// [`crate::local_vector::feature_collection_to_tile`] writes as the drawn
    /// feature's id) must always come through here.
    pub fn select_source_feature(&mut self, source: Option<usize>) {
        let rows = self.row_count();
        self.selected_feature = source.filter(|row| *row < rows);
        self.selected_extra.clear();
    }

    /// [`Self::select_source_feature`] for a map multi-selection: `anchor`
    /// binds exactly as before, and every other member of `extra` gets the
    /// co-selection highlight. Out-of-range indices are dropped, and
    /// duplicate ones collapse to one — see [`Self::co_selected_count`].
    pub fn select_source_features(&mut self, anchor: Option<usize>, extra: &[usize]) {
        let rows = self.row_count();
        self.selected_feature = anchor.filter(|row| *row < rows);
        self.selected_extra.clear();
        self.selected_extra.extend(
            extra
                .iter()
                .copied()
                .filter(|&row| row < rows && Some(row) != self.selected_feature),
        );
    }

    /// Binds `features` if they are not the bound dataset already, dropping the
    /// previous binding's schema, sort and selection.
    fn bind(&mut self, id: LayerId, features: &Arc<FeatureCollection>) {
        if self
            .bound
            .as_ref()
            .is_some_and(|bound| bound.matches(id, features))
        {
            return;
        }
        self.bound = Some(BoundLayer::new(id, Arc::clone(features)));
        self.selected_feature = None;
        self.h_offset = 0.0;
        // Cleared rather than carried over: a filter typed against the
        // previous layer's columns is not meaningfully applicable to a
        // different dataset, and `BoundLayer::new`'s fresh `order_key`
        // already matches `(None, 0)`, so resetting to exactly that avoids
        // an otherwise-harmless redundant re-filter on the very next sync.
        self.filter_text.clear();
        self.filter_revision = 0;
    }

    /// Total width of every column, in logical pixels.
    fn total_width(bound: &BoundLayer) -> f32 {
        (0..bound.table.source().column_defs().len())
            .map(|column| bound.table.effective_width(column))
            .sum()
    }

    /// Left edge of `column`, in content-space logical pixels.
    fn column_offset(bound: &BoundLayer, column: usize) -> f32 {
        (0..column)
            .map(|index| bound.table.effective_width(index))
            .sum()
    }

    /// Draws the filter box and the "Copy CSV" button below the summary row.
    /// Returns whether the filter text changed this frame; the caller owns
    /// the revision counter (`bound` is rebuilt on every re-bind, so it
    /// cannot).
    ///
    /// Takes `bound` by shared reference only — the caller still needs it
    /// mutably afterwards for [`Self::draw_header`]/[`Self::draw_body`], and
    /// this function only ever reads it (the current display order, for the
    /// match count and the CSV button).
    fn draw_toolbar(ui: &mut Ui, bound: &BoundLayer, filter_text: &mut String) -> ToolbarOutcome {
        let mut outcome = ToolbarOutcome::default();
        ui.horizontal(|ui| {
            ui.label("Filter:");
            let edit = ui.add(
                egui::TextEdit::singleline(filter_text)
                    .hint_text(format!("Search all columns{ELLIPSIS}"))
                    .desired_width(180.0),
            );
            outcome.filter_changed = edit.changed();
            if !filter_text.trim().is_empty() {
                ui.weak(format!(
                    "{MIDDLE_DOT} {} of {} match",
                    bound.order.len(),
                    bound.table.source().row_count()
                ));
            }
            if ui.button("Copy CSV").clicked() {
                ui.ctx().copy_text(bound.csv());
            }
            // The file-writing twin: the CSV is captured HERE, at the click,
            // not re-derived when the app drains it (see the module docs).
            if ui
                .button("Export CSV\u{2026}")
                .on_hover_text("Writes these rows to a file")
                .clicked()
            {
                outcome.export = Some(bound.csv());
            }
        });
        outcome
    }

    /// Draws the clickable, resizable column header, mirroring the body's
    /// horizontal scroll position.
    fn draw_header(
        ui: &mut Ui,
        bound: &mut BoundLayer,
        h_offset: &mut f32,
        events: &mut Vec<TableEvent>,
    ) {
        let row_height = bound.table.row_height();
        let total_width = Self::total_width(bound);
        let sort = bound.table.sort_state();
        let font = egui::TextStyle::Button.resolve(ui.style());
        let text_color = ui.visuals().strong_text_color();

        egui::ScrollArea::horizontal()
            .id_salt("oxigis_attribute_table_header")
            .scroll_offset(vec2(*h_offset, 0.0))
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
            .show(ui, |ui| {
                let (rect, _response) =
                    ui.allocate_exact_size(vec2(total_width, row_height), Sense::hover());
                let painter = ui.painter();
                painter.rect_filled(rect, 0.0, ui.visuals().faint_bg_color);

                let column_count = bound.table.source().column_defs().len();
                let mut left = rect.left();
                for column in 0..column_count {
                    let width = bound.table.effective_width(column);
                    let cell = Rect::from_min_size(pos2(left, rect.top()), vec2(width, row_height));
                    left += width;
                    if !ui.is_rect_visible(cell) {
                        continue;
                    }

                    let name = bound
                        .table
                        .source()
                        .column_defs()
                        .get(column)
                        .map_or(String::new(), |def| def.name.clone());
                    let label = match sort {
                        Some(state) if state.column == column => {
                            let arrow = match state.direction {
                                SortDirection::Ascending => format!(" {MOVE_UP}"),
                                SortDirection::Descending => format!(" {MOVE_DOWN}"),
                                SortDirection::None => String::new(),
                            };
                            format!("{name}{arrow}")
                        }
                        _ => name,
                    };
                    ui.painter().with_clip_rect(cell).text(
                        pos2(cell.left() + CELL_PAD_X, cell.center().y),
                        Align2::LEFT_CENTER,
                        label,
                        font.clone(),
                        text_color,
                    );

                    // Sort toggle: the whole header cell except the drag strip.
                    let click_rect = Rect::from_min_max(
                        cell.min,
                        pos2(
                            (cell.right() - RESIZE_HANDLE_WIDTH).max(cell.left()),
                            cell.bottom(),
                        ),
                    );
                    let click = ui.interact(
                        click_rect,
                        ui.id().with(("oxigis_attr_header", column)),
                        Sense::click(),
                    );
                    if click.hovered() {
                        ui.painter()
                            .rect_filled(cell, 0.0, ui.visuals().widgets.hovered.bg_fill);
                        ui.painter().with_clip_rect(cell).text(
                            pos2(cell.left() + CELL_PAD_X, cell.center().y),
                            Align2::LEFT_CENTER,
                            bound
                                .table
                                .source()
                                .column_defs()
                                .get(column)
                                .map_or(String::new(), |def| def.name.clone()),
                            font.clone(),
                            text_color,
                        );
                    }
                    if click.clicked() {
                        let state = bound.table.toggle_sort(column);
                        events.push(TableEvent::SortChanged {
                            col: column,
                            ascending: state
                                .is_some_and(|state| state.direction == SortDirection::Ascending),
                        });
                    }

                    // Resize handle.
                    let handle = Rect::from_min_max(
                        pos2(
                            (cell.right() - RESIZE_HANDLE_WIDTH).max(cell.left()),
                            cell.top(),
                        ),
                        cell.max,
                    );
                    let drag = ui.interact(
                        handle,
                        ui.id().with(("oxigis_attr_resize", column)),
                        Sense::drag(),
                    );
                    if drag.hovered() || drag.dragged() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                    }
                    if drag.dragged()
                        && let Some(new_width) =
                            bound.table.resize_column(column, drag.drag_delta().x)
                    {
                        events.push(TableEvent::ColumnResized {
                            col: column,
                            new_width,
                        });
                    }
                }
            });
        ui.separator();
    }

    /// Draws the virtualized row body: only the rows and columns inside the
    /// viewport are laid out or painted. `selected_extra` is a [`HashSet`] so
    /// its per-row membership test below stays `O(1)` regardless of how many
    /// features a map marquee co-selected — this loop is otherwise the whole
    /// reason the panel stays proportional to the *visible* rows, not the
    /// dataset.
    fn draw_body(
        ui: &mut Ui,
        bound: &mut BoundLayer,
        h_offset: &mut f32,
        selected_feature: &mut Option<usize>,
        selected_extra: &HashSet<usize>,
        events: &mut Vec<TableEvent>,
    ) {
        let row_height = bound.table.row_height();
        let total_width = Self::total_width(bound);
        let visible_rows = bound.order.len();
        let font = egui::TextStyle::Body.resolve(ui.style());
        let text_color = ui.visuals().text_color();
        let stripe = ui.visuals().faint_bg_color;
        let selected_fill = ui.visuals().selection.bg_fill;

        let columns = bound
            .table
            .visible_column_range(*h_offset, ui.available_width().max(1.0));

        let output = egui::ScrollArea::both()
            .id_salt("oxigis_attribute_table_body")
            .auto_shrink([false, false])
            .show_rows(ui, row_height, visible_rows, |ui, rows| {
                ui.spacing_mut().item_spacing = Vec2::ZERO;
                for visible_row in rows {
                    let Some(source_row) = bound.order.get(visible_row).copied() else {
                        continue;
                    };
                    let (rect, response) =
                        ui.allocate_exact_size(vec2(total_width, row_height), Sense::click());

                    if *selected_feature == Some(source_row) {
                        ui.painter().rect_filled(rect, 0.0, selected_fill);
                    } else if selected_extra.contains(&source_row) {
                        // A co-selected member of the map's multi-selection:
                        // the anchor's fill at half strength.
                        ui.painter()
                            .rect_filled(rect, 0.0, selected_fill.gamma_multiply(0.5));
                    } else if visible_row % 2 == 1 {
                        ui.painter().rect_filled(rect, 0.0, stripe);
                    }
                    if response.clicked() {
                        *selected_feature = Some(source_row);
                        events.push(TableEvent::RowSelected(visible_row));
                    }

                    if !ui.is_rect_visible(rect) {
                        continue;
                    }
                    let mut left = rect.left() + Self::column_offset(bound, columns.start);
                    for column in columns.clone() {
                        let width = bound.table.effective_width(column);
                        let cell =
                            Rect::from_min_size(pos2(left, rect.top()), vec2(width, row_height));
                        left += width;
                        let text = bound.table.source().cell_text(source_row, column);
                        if text.is_empty() {
                            continue;
                        }
                        ui.painter().with_clip_rect(cell).text(
                            pos2(cell.left() + CELL_PAD_X, cell.center().y),
                            Align2::LEFT_CENTER,
                            text,
                            font.clone(),
                            text_color,
                        );
                    }
                }
            });

        // The header is seeded from this on the next frame. egui repaints while
        // a scroll is in flight, so the one-frame follow is not observable.
        *h_offset = output.state.offset.x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigeo::geojson::reader::feature_collection_from_str;
    use oxigis_core::{Layer, LayerKind, LayerStack, VectorSource};

    use crate::attribute_table::SYNTHETIC_COLUMN_COUNT;

    const CITIES: &str = r#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [139.7, 35.7]},
             "properties": {"name": "Tokyo", "pop": 3}},
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [135.5, 34.7]},
             "properties": {"name": "Osaka", "pop": 1}},
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [136.9, 35.2]},
             "properties": {"name": "Nagoya", "pop": 2}}
        ]
    }"#;

    fn cities() -> Arc<FeatureCollection> {
        Arc::new(feature_collection_from_str(CITIES).expect("fixture parses"))
    }

    fn a_layer_id() -> LayerId {
        let mut stack = LayerStack::new();
        stack.add(Layer::new(
            "cities",
            LayerKind::Vector(VectorSource::InlineGeoJson {
                geojson: CITIES.to_string(),
            }),
        ))
    }

    /// Column index of the `pop` property in the derived schema.
    fn pop_column(panel: &AttributeTablePanel) -> usize {
        let bound = panel.bound.as_ref().expect("bound");
        bound
            .table
            .source()
            .column_defs()
            .iter()
            .position(|def| def.name == "pop")
            .expect("pop is a column")
    }

    #[test]
    fn showing_a_layer_binds_its_rows_and_columns() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        assert_eq!(panel.bound_layer(), Some(id));
        assert_eq!(panel.row_count(), 3);
        // `#`, `geometry`, `name`, `pop`.
        assert_eq!(panel.column_count(), 4);
        assert_eq!(panel.display_order(), &[0, 1, 2]);
    }

    #[test]
    fn placeholder_unbinds_and_clears_the_selection() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        panel.select_visible_row(1);
        assert_eq!(panel.selected_feature(), Some(1));
        egui::__run_test_ui(|ui| {
            panel.show_placeholder(ui, false);
        });
        assert_eq!(panel.bound_layer(), None);
        assert_eq!(panel.selected_feature(), None);
        assert_eq!(panel.row_count(), 0);
        assert_eq!(panel.display_order(), &[] as &[usize]);
    }

    #[test]
    fn sorting_is_numeric_for_a_numeric_column_and_cycles_back_to_document_order() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        let column = pop_column(&panel);

        panel.toggle_sort(column);
        // pop 3, 1, 2 → ascending is Osaka(1), Nagoya(2), Tokyo(0).
        assert_eq!(panel.display_order(), &[1, 2, 0]);
        panel.toggle_sort(column);
        assert_eq!(panel.display_order(), &[0, 2, 1]);
        panel.toggle_sort(column);
        assert_eq!(panel.display_order(), &[0, 1, 2]);
    }

    #[test]
    fn selection_is_the_source_index_not_the_visible_row() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        panel.toggle_sort(pop_column(&panel));
        assert_eq!(panel.display_order(), &[1, 2, 0]);
        // Visible row 0 is now Osaka, which is source feature 1.
        panel.select_visible_row(0);
        assert_eq!(panel.selected_feature(), Some(1));
        // A row past the end clears rather than panicking.
        panel.select_visible_row(99);
        assert_eq!(panel.selected_feature(), None);
    }

    #[test]
    fn select_source_feature_ignores_the_display_order_and_bounds_checks() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        panel.toggle_sort(pop_column(&panel));
        assert_eq!(panel.display_order(), &[1, 2, 0]);

        // Source 0 is Tokyo, whatever the sort put in visible row 0 — feeding
        // this number to `select_visible_row` would have picked Osaka instead.
        panel.select_source_feature(Some(0));
        assert_eq!(panel.selected_feature(), Some(0));
        panel.select_source_feature(Some(2));
        assert_eq!(panel.selected_feature(), Some(2));

        // Out of range and `None` both clear rather than panicking.
        panel.select_source_feature(Some(3));
        assert_eq!(panel.selected_feature(), None);
        panel.select_source_feature(Some(0));
        panel.select_source_feature(None);
        assert_eq!(panel.selected_feature(), None);

        // With nothing bound at all, every index is out of range.
        let mut empty = AttributeTablePanel::default();
        empty.select_source_feature(Some(0));
        assert_eq!(empty.selected_feature(), None);
    }

    #[test]
    fn rebinding_the_same_arc_keeps_the_sort_a_different_arc_resets_it() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        let features = cities();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &features);
        });
        panel.toggle_sort(pop_column(&panel));
        assert_eq!(panel.display_order(), &[1, 2, 0]);

        // Same layer, same collection: the binding (and its sort) survives.
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &features);
        });
        assert_eq!(panel.display_order(), &[1, 2, 0]);

        // Same layer id, freshly parsed features (a re-hydrated path layer):
        // the binding is rebuilt, so the sort is gone.
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        assert_eq!(panel.display_order(), &[0, 1, 2]);
    }

    #[test]
    fn a_hundred_thousand_rows_bind_and_sort_without_materializing_them() {
        // A realistic width too, not one column: deriving the schema visits
        // every property of every feature, so a one-key fixture would not
        // exercise the part of that walk that scales with the column count.
        const KEYS: u32 = 12;
        let mut features = Vec::with_capacity(100_000);
        for index in 0..100_000u32 {
            let mut properties = oxigeo::geojson::types::Properties::new();
            properties.insert("n".to_string(), (100_000 - index).into());
            for key in 0..KEYS {
                properties.insert(format!("k{key:02}"), (index % 7).into());
            }
            features.push(oxigeo::geojson::types::Feature::new(None, Some(properties)));
        }
        let collection = Arc::new(FeatureCollection::new(features));
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "big", &collection);
        });
        assert_eq!(panel.row_count(), 100_000);
        let column_count = usize::try_from(KEYS).unwrap_or(0) + 1 + SYNTHETIC_COLUMN_COUNT;
        assert_eq!(panel.column_count(), column_count);

        let n_column = panel
            .bound
            .as_ref()
            .and_then(|bound| {
                bound
                    .table
                    .source()
                    .column_defs()
                    .iter()
                    .position(|def| def.name == "n")
            })
            .expect("n is a column");
        panel.toggle_sort(n_column);
        // Descending by construction, so ascending "n" reverses the rows.
        assert_eq!(panel.display_order().first(), Some(&99_999));
        assert_eq!(panel.display_order().last(), Some(&0));
    }

    #[test]
    fn showing_and_placeholder_emit_no_events_without_interaction() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            assert!(panel.show(ui, id, "cities", &cities()).is_empty());
            panel.show_placeholder(ui, true);
        });
    }

    #[test]
    fn set_filter_text_narrows_display_order_without_changing_row_count() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        assert_eq!(panel.display_order(), &[0, 1, 2]);

        panel.set_filter_text("osaka");
        assert_eq!(panel.filter_text(), "osaka");
        // Osaka is source feature 1 — `row_count` stays the unfiltered total.
        assert_eq!(panel.display_order(), &[1]);
        assert_eq!(panel.row_count(), 3);
    }

    #[test]
    fn set_filter_text_is_case_insensitive_and_a_repeat_call_is_a_no_op() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });

        panel.set_filter_text("TOKYO");
        assert_eq!(panel.display_order(), &[0]);
        // Same value again: still a no-op, not just on the stored text.
        panel.set_filter_text("TOKYO");
        assert_eq!(panel.display_order(), &[0]);
    }

    #[test]
    fn a_non_matching_filter_empties_display_order_not_row_count() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        panel.set_filter_text("no such city");
        assert_eq!(panel.display_order(), &[] as &[usize]);
        assert_eq!(panel.row_count(), 3);
    }

    #[test]
    fn clearing_the_filter_restores_the_full_display_order() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        panel.set_filter_text("osaka");
        assert_eq!(panel.display_order(), &[1]);
        panel.set_filter_text("");
        assert_eq!(panel.display_order(), &[0, 1, 2]);
    }

    #[test]
    fn filter_is_applied_before_sort_so_only_survivors_are_ordered() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        // "a" matches Osaka(1, pop 1) and Nagoya(2, pop 2), not Tokyo(0).
        panel.set_filter_text("a");
        assert_eq!(panel.display_order(), &[1, 2]);

        panel.toggle_sort(pop_column(&panel));
        // Ascending pop over the 2 survivors only: Osaka(1) before Nagoya(2).
        assert_eq!(panel.display_order(), &[1, 2]);
        panel.toggle_sort(pop_column(&panel));
        // Descending: Nagoya(2) before Osaka(1). Tokyo(0) never reappears.
        assert_eq!(panel.display_order(), &[2, 1]);
    }

    #[test]
    fn rebinding_to_a_different_layer_clears_the_filter() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        panel.set_filter_text("osaka");
        assert_eq!(panel.display_order(), &[1]);

        // A different `Arc`, same layer id (a re-hydrated path layer)
        // rebuilds the binding — the stale filter must not silently hide the
        // new data.
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        assert_eq!(panel.filter_text(), "");
        assert_eq!(panel.display_order(), &[0, 1, 2]);
    }

    #[test]
    fn co_selected_count_collapses_duplicate_source_indices() {
        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        panel.select_source_features(Some(0), &[1, 1, 2]);
        assert_eq!(panel.selected_feature(), Some(0));
        // 2 distinct co-selected features, not 3: `extra` is a set.
        assert_eq!(panel.co_selected_count(), 2);
    }

    #[test]
    fn export_csv_reflects_the_current_filter_and_sort_and_is_empty_unbound() {
        let empty = AttributeTablePanel::default();
        assert_eq!(empty.export_csv(), "");
        assert_eq!(empty.export_csv_bytes(), Vec::<u8>::new());

        let mut panel = AttributeTablePanel::default();
        let id = a_layer_id();
        egui::__run_test_ui(|ui| {
            let _events = panel.show(ui, id, "cities", &cities());
        });
        panel.set_filter_text("a"); // Osaka, Nagoya — not Tokyo.
        panel.toggle_sort(pop_column(&panel)); // Ascending pop.

        let csv = panel.export_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3, "header plus the 2 filtered rows, not all 3");
        assert!(lines[1].contains("Osaka"));
        assert!(lines[2].contains("Nagoya"));
        assert!(!csv.contains("Tokyo"), "Tokyo was filtered out");
        assert_eq!(panel.export_csv_bytes(), csv.into_bytes());
    }
}
