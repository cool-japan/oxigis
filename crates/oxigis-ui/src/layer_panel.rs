//! Layer-tree panel: visibility, opacity, reordering, removal, and a
//! demo-layer button.
//!
//! Drawing never mutates [`LayerStack`] directly — [`ui`] returns the
//! user-requested [`LayerAction`]s instead, which the caller (typically
//! [`crate::app::OxigisApp`]) applies after the immediate-mode pass
//! finishes. This sidesteps a borrow conflict (iterating
//! `LayerStack::layers()` while also calling a `&mut self` mutator on the
//! same stack does not borrow-check) and, as a side effect, keeps the
//! state-transition logic testable without an `egui::Ui`.

use egui::Ui;
use oxigis_core::layer::MAX_ZOOM_LEVEL;
use oxigis_core::{
    ArchiveFormat, ArchiveRef, Layer, LayerId, LayerKind, LayerStack, RasterSource, VectorSource,
    VectorTilePaint,
};

use crate::app::providers::TileStack;
use crate::local_input;
use crate::tile_provider::{BASEMAP_PRESETS, BasemapConfig, OSM_ATTRIBUTION, OSM_URL_TEMPLATE};
use crate::ui_glyphs::{BASEMAP_TOGGLE, ELLIPSIS, MOVE_DOWN, MOVE_UP, REMOVE, WARNING};

/// A single user-requested change to the layer stack, collected while
/// drawing [`ui`] and applied by the caller afterwards.
///
/// Not [`Copy`]: [`LayerAction::AddCogLayer`] carries the URL the user typed.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerAction {
    /// The layer became the selected layer (clicked its name).
    Select(LayerId),
    /// The map should move to fit this layer's data (its name was
    /// double-clicked).
    ///
    /// Offered only for a local vector layer, because that is the only kind
    /// whose extent this crate can compute at all — the app-side feature store
    /// holds its collection, while a provider layer's extent lives behind the
    /// network. Whether the store actually *holds* those features is the
    /// gesture's business rather than the panel's: a `LocalGeoJson` path layer
    /// on a build with no filesystem is a local layer whose data never arrived,
    /// and only the app can see that.
    ZoomToLayer(LayerId),
    /// The layer's visibility checkbox was toggled.
    ToggleVisibility(LayerId),
    /// The layer's opacity slider moved to a new value.
    SetOpacity(LayerId, f32),
    /// The layer should move one step toward the front (top of the list).
    MoveUp(LayerId),
    /// The layer should move one step toward the back (bottom of the list).
    MoveDown(LayerId),
    /// The layer should be removed.
    Remove(LayerId),
    /// The "add demo layer" button was clicked.
    AddDemoXyzLayer,
    /// A built-in sample was picked from the basemap preset list; replace the
    /// basemap with it, required credit line included (see
    /// [`crate::tile_provider::BASEMAP_PRESETS`]).
    SetBasemapPreset(BasemapConfig),
    /// An XYZ `{z}/{x}/{y}` URL template was submitted; replace the basemap
    /// with it. WMTS-REST endpoints with `GoogleMapsCompatible` tile matrices
    /// are the same thing spelled longer — placeholder order is free
    /// (`{z}/{y}/{x}` is common there) and `{-y}` flips TMS row numbering.
    SetXyzBasemap(String),
    /// A layer should draw as the basemap instead of the basemap service
    /// ([`None`] demotes whatever was promoted).
    ///
    /// Only an [`RasterSource::Xyz`] layer can be promoted, and the gesture —
    /// not the panel — is what refuses anything else: promotability depends on
    /// the whole project, which this module never sees.
    SetBasemapLayer(Option<LayerId>),
    /// A Cloud-Optimized GeoTIFF URL was submitted; add it as a raster layer.
    AddCogLayer(String),
    /// A `.pmtiles` URL was submitted; read its header and *then* add the
    /// layer, because only the header says whether it is raster or vector.
    AddArchiveUrlLayer(String),
    /// The "Open…" button beside the archive field was clicked: ask the shell
    /// for a file, since this crate has no file dialog and never will (it
    /// compiles to `wasm32`, where there is no filesystem to dialog about).
    OpenArchiveFile,
    /// An MVT `{z}/{x}/{y}.pbf` URL template was submitted; add it as a vector
    /// tile layer.
    AddVectorTileLayer(String),
    /// The "+ GeoJSON" button was clicked: open the paste-GeoJSON dialog. The
    /// panel has no text area of its own for this — a `FeatureCollection` is far
    /// too long for a one-line field — so the app owns the modal.
    AddGeoJsonPaste,
    /// The refusal banner's Retry button was clicked: forget the memoized
    /// install refusals so the outstanding plans are offered to a shell again.
    ///
    /// A refusal names a *plan* (a basemap plus at most one raster layer), not
    /// a layer, which is why this carries no [`LayerId`] and why there is one
    /// banner rather than a per-row badge.
    RetryRefusedInstalls,
}

/// A per-layer change reported by the row's settings section, applied by the
/// caller after the immediate-mode pass — the same contract [`LayerAction`]
/// has, in a second channel.
///
/// # Why a second enum rather than more [`LayerAction`] variants
///
/// [`LayerAction`] is consumed by an exhaustive match in the app's dispatch,
/// and the two families answer different questions: a `LayerAction` is a
/// *panel gesture* (add this, move that, retry), while a `LayerEdit` is a
/// **recorded project edit** — each one maps 1:1 onto a
/// [`crate::edit::project_op::ProjectOp`] and therefore onto exactly one
/// Ctrl+Z. Keeping them apart is what lets the applier be a single choke point
/// that cannot forget to record.
///
/// A caller that does not consume these is not shown the controls that produce
/// them — see [`PanelExtras::edits`] — because a control whose result nothing
/// applies is worse than a missing one.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerEdit {
    /// The layer's name was committed in the row's settings section. Already
    /// trimmed, non-empty, and known to differ from the current name (see
    /// [`rename_edit`]).
    Rename(LayerId, String),
    /// The layer's scale range changed. Both ends always travel together: they
    /// are one user-facing fact, and half a range on the undo stack is exactly
    /// the "an undo that does not restore what the redo set" failure the
    /// project-op family is built to make unrepresentable.
    SetZoomRange {
        /// Which layer.
        layer: LayerId,
        /// Lowest zoom at which it draws (inclusive), or [`None`] for none.
        min_zoom: Option<f32>,
        /// Zoom at which it stops drawing (exclusive), or [`None`] for none.
        max_zoom: Option<f32>,
    },
}

/// The two things the panel learned to say after [`ui`]'s signature froze: what
/// the map's draw budget is doing, and where per-layer edits go.
///
/// Bundled rather than added as parameters for the reason [`PanelFields`] is
/// bundled — and passed to [`ui_with_extras`] rather than to [`ui`] so the
/// older, narrower entry point keeps working unchanged.
pub struct PanelExtras<'a> {
    /// The tiled layers the map actually draws this frame, derived once by the
    /// caller (`OxigisApp::desired_tile_stack`) and shared by every row: it is
    /// what tells a row that it is *listed but buried* under the composite cap,
    /// and it carries the one sentence that explains why.
    ///
    /// [`None`] means the caller does not derive it, and no row is badged.
    pub tiles: Option<&'a TileStack>,
    /// Where [`LayerEdit`]s go, or [`None`] for a caller that does not apply
    /// them — in which case the controls that produce them are not drawn at
    /// all.
    pub edits: Option<&'a mut Vec<LayerEdit>>,
}

/// One-word tag describing where a layer's data comes from, shown greyed next
/// to its name so a local dataset is distinguishable from a tiled one.
#[must_use]
pub fn kind_tag(layer: &Layer) -> &'static str {
    match &layer.kind {
        LayerKind::Raster(RasterSource::Xyz { .. }) => "xyz",
        LayerKind::Raster(RasterSource::Cog { .. }) => "cog",
        LayerKind::Raster(RasterSource::LocalGeoTiff { .. }) => "tiff",
        LayerKind::Raster(RasterSource::TileArchive { format, .. })
        | LayerKind::Vector(VectorSource::TileArchive { format, .. }) => match format {
            ArchiveFormat::PmTiles => "pmtiles",
            ArchiveFormat::MbTiles => "mbtiles",
        },
        LayerKind::Vector(VectorSource::MvtTiles { .. }) => "mvt",
        LayerKind::Vector(VectorSource::LocalGeoJson { .. }) => "geojson",
        LayerKind::Vector(VectorSource::InlineGeoJson { .. }) => "geojson*",
        LayerKind::Vector(VectorSource::LocalShapefile { .. }) => "shp",
        LayerKind::Vector(VectorSource::LocalGpkg { .. }) => "gpkg",
        LayerKind::Vector(VectorSource::LocalGeoParquet { .. }) => "parquet",
    }
}

/// A zoom level as a row chip spells it: `14` rather than `14.0`, `13.5` when
/// the half really is there.
fn zoom_text(value: f32) -> String {
    if (value - value.round()).abs() < 1.0e-3 {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

/// The short chip a row shows when the layer declares a scale range, or
/// [`None`] when it draws at every zoom.
///
/// Deliberately tiny and bounded (`z14+`, `z<18`, `z14-18`): this panel sizes
/// itself to its widest row, so a chip that spelled the whole rule out would
/// narrow the map for every project, permanently, to say something the row's
/// settings section already says in full.
#[must_use]
pub fn zoom_range_tag(layer: &Layer) -> Option<String> {
    match (layer.min_zoom(), layer.max_zoom()) {
        (None, None) => None,
        (Some(min), None) => Some(format!("z{}+", zoom_text(min))),
        (None, Some(max)) => Some(format!("z<{}", zoom_text(max))),
        (Some(min), Some(max)) => Some(format!("z{}-{}", zoom_text(min), zoom_text(max))),
    }
}

/// The rename the row's name field commits, or [`None`] when there is nothing
/// to record.
///
/// The three refusals are all "an undo step that changes nothing is history the
/// user has to press Ctrl+Z through for no reason": a blank name, a
/// whitespace-only name, and a name equal to the current one. Trimming happens
/// here rather than in the applier so the recorded `before`/`after` pair is
/// exactly what the project will hold.
#[must_use]
pub fn rename_edit(layer: &Layer, typed: &str) -> Option<LayerEdit> {
    let trimmed = typed.trim();
    if trimmed.is_empty() || trimmed == layer.name {
        return None;
    }
    Some(LayerEdit::Rename(layer.id, trimmed.to_string()))
}

/// The scale-range change the row's range editor commits, or [`None`] when the
/// requested range is the one the layer already has.
///
/// The comparison is against the layer's *sanitized* bounds — through the same
/// [`oxigis_core::layer::sanitize_zoom_bound`] the model itself uses — so a
/// drag the model would normalize back to where it started records nothing.
#[must_use]
pub fn zoom_range_edit(layer: &Layer, min: Option<f32>, max: Option<f32>) -> Option<LayerEdit> {
    let min_zoom = oxigis_core::layer::sanitize_zoom_bound(min);
    let max_zoom = oxigis_core::layer::sanitize_zoom_bound(max);
    if (min_zoom, max_zoom) == (layer.min_zoom(), layer.max_zoom()) {
        return None;
    }
    Some(LayerEdit::SetZoomRange {
        layer: layer.id,
        min_zoom,
        max_zoom,
    })
}

/// One row's settings section, kept in egui's per-frame temp store rather than
/// in the caller.
///
/// This module holds no state by design — it draws, it reports, the caller
/// applies — and a caller-owned buffer per layer would mean a map from
/// [`LayerId`] in [`crate::app::OxigisApp`] that has to be pruned whenever a
/// layer is removed, hydrated or undone back into existence. egui's temp store
/// is keyed by [`egui::Id`] and evicts on its own, so the panel keeps its "no
/// state" property and a removed layer's buffer simply stops being read.
#[derive(Clone, Default)]
struct RowSettings {
    /// Whether the section is expanded.
    open: bool,
    /// What has been typed into the name field since it was opened.
    name: String,
    /// Whether a lower bound is declared, as the checkbox stands.
    min_on: bool,
    /// The lower bound, live while it is being dragged or typed.
    min: f32,
    /// Whether an upper bound is declared, as the checkbox stands.
    max_on: bool,
    /// The upper bound, live while it is being dragged or typed.
    max: f32,
    /// Whether the four fields above hold a range the project does NOT yet
    /// have, waiting for the gesture to finish.
    ///
    /// This is what makes a drag ONE undo step without any coalescing
    /// machinery: a [`egui::DragValue`] reports `changed` on every frame of a
    /// drag, and recording each of those would put fifty entries on the stack
    /// for one gesture. Nothing is reported until the pointer lifts or the
    /// field loses focus, so the dispatch arm that consumes
    /// [`LayerEdit::SetZoomRange`] needs no coalescing key at all.
    range_pending: bool,
}

impl RowSettings {
    /// Loads the editable buffers from `layer` — done when the section opens,
    /// and on every frame it is open with no gesture in flight, so an undo, a
    /// redo or a project load shows up in the editor rather than being
    /// overwritten by a stale buffer.
    fn seed_from(&mut self, layer: &Layer) {
        self.min_on = layer.min_zoom().is_some();
        self.min = layer.min_zoom().unwrap_or(0.0);
        self.max_on = layer.max_zoom().is_some();
        // An unbounded upper end opens at the ceiling rather than at zero, so
        // ticking the box changes nothing and the user drags DOWN from there —
        // opening at 0 would blank the layer everywhere the instant the box
        // was ticked.
        self.max = layer.max_zoom().unwrap_or(MAX_ZOOM_LEVEL);
    }

    /// The range the buffers currently describe.
    fn range(&self) -> (Option<f32>, Option<f32>) {
        (
            self.min_on.then_some(self.min),
            self.max_on.then_some(self.max),
        )
    }

    /// Reports the buffered range if it is still waiting to be applied, and
    /// clears the flag. Called when a gesture ends and when the section closes,
    /// so a range the user set can never be silently dropped.
    fn commit_range(&mut self, layer: &Layer, edits: &mut Vec<LayerEdit>) {
        if !self.range_pending {
            return;
        }
        self.range_pending = false;
        let (min, max) = self.range();
        if let Some(edit) = zoom_range_edit(layer, min, max) {
            edits.push(edit);
        }
    }
}

/// The temp-store key for one layer's settings section.
///
/// Salted by the layer id rather than by widget position, so scrolling the list
/// or reordering the stack cannot swap two rows' buffers.
fn row_settings_id(layer: LayerId) -> egui::Id {
    egui::Id::new(("oxigis_layer_row_settings", layer.get()))
}

/// The layer promoted to draw as the basemap, as the panel needs to show it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromotedBasemap {
    /// Which layer the project's promotion pointer names. The row toggle is a
    /// control of this pointer, so a promotion that does not currently draw
    /// can still be turned off.
    pub layer: LayerId,
    /// Whether the promotion currently *resolves* — `false` while the layer
    /// is hidden, which the row's own visibility checkbox already shows.
    pub drawn: bool,
}

/// What the panel must say about the map itself, as opposed to about a layer.
///
/// Bundled for the same reason [`PanelFields`] is: [`ui`] would otherwise grow
/// a parameter per sentence and pass clippy's seven-argument cap. Everything
/// here is derived from the project by [`crate::app::OxigisApp`] and read
/// only — the panel holds no state.
pub struct MapStatus<'a> {
    /// The basemap service currently drawn, so the preset picker can show
    /// which entry (if any) is active.
    pub basemap: &'a BasemapConfig,
    /// The layer drawn as the basemap instead of that service, if one is
    /// promoted.
    pub basemap_layer: Option<PromotedBasemap>,
    /// Why the map is not drawing what the project implies, when a shell
    /// refused an install that is still outstanding. Drives the banner under
    /// the Basemap row and its Retry button.
    pub refusal: Option<String>,
}

/// The panel's four caller-owned text buffers, bundled so adding a fifth field
/// does not widen [`ui`]'s signature again.
///
/// They live in [`crate::OxigisApp`] rather than here because this module holds
/// no state at all: it draws, it reports [`LayerAction`]s, and the caller
/// applies them after the immediate-mode pass.
pub struct PanelFields<'a> {
    /// "add COG" — a Cloud-Optimized GeoTIFF URL.
    pub cog_url: &'a mut String,
    /// "add tile archive" — a `.pmtiles` URL.
    pub archive_url: &'a mut String,
    /// "add vector tiles" — an MVT `{z}/{x}/{y}.pbf` template.
    pub vector_url: &'a mut String,
    /// "set XYZ basemap" — a raster `{z}/{x}/{y}` template.
    pub xyz_url: &'a mut String,
}

/// Draws the layer-tree panel and returns every action the user requested
/// this frame (usually zero or one, but a slider drag can repeat
/// [`LayerAction::SetOpacity`] across several frames).
///
/// Layers are listed top-of-stack-first (i.e. the reverse of
/// [`LayerStack::layers`]'s back-to-front storage order), matching how
/// GIS layer panels conventionally read: the layer painted last (on top) is
/// shown first.
/// `map` is what the panel must say about the map itself (see [`MapStatus`]);
/// `cog_url`, `archive_url`, `vector_url` and
/// `xyz_url` are the caller-owned text buffers of the "add COG", "add tile
/// archive", "add vector tiles" and "set XYZ basemap" fields, so the typed URLs
/// survive between frames without this module holding state.
pub fn ui(
    ui: &mut Ui,
    layers: &LayerStack,
    selected: Option<LayerId>,
    map: MapStatus<'_>,
    fields: PanelFields<'_>,
) -> Vec<LayerAction> {
    ui_with_extras(
        ui,
        layers,
        selected,
        map,
        fields,
        PanelExtras {
            tiles: None,
            edits: None,
        },
    )
}

/// [`ui`] plus the two things a caller can opt into: the map's tiled-draw
/// budget, and a sink for per-layer [`LayerEdit`]s.
///
/// The whole panel body lives here; [`ui`] is the narrower, frozen-signature
/// entry point that forwards with both opted out. Both `extras` fields are
/// [`None`]-able rather than required precisely so the two entry points cannot
/// drift: there is one implementation, and opting out is a value, not a
/// separate code path.
///
/// With `extras.edits` at [`None`] the per-row settings control is **not
/// drawn** — a control whose result nothing applies is worse than a missing
/// one, the same rule that keeps the basemap toggle off non-XYZ rows.
pub fn ui_with_extras(
    ui: &mut Ui,
    layers: &LayerStack,
    selected: Option<LayerId>,
    map: MapStatus<'_>,
    fields: PanelFields<'_>,
    extras: PanelExtras<'_>,
) -> Vec<LayerAction> {
    let PanelExtras { tiles, edits } = extras;
    // Collected locally and handed over at the end, rather than threaded as an
    // `Option<&mut Vec<_>>` through every row: the reborrow that would need is
    // pure noise, and this way "the caller consumes edits" is one `bool` the
    // row code reads.
    let editable = edits.is_some();
    let mut collected: Vec<LayerEdit> = Vec::new();
    let MapStatus {
        basemap,
        basemap_layer,
        refusal,
    } = map;
    let PanelFields {
        cog_url,
        archive_url,
        vector_url,
        xyz_url,
    } = fields;
    let mut actions = Vec::new();

    ui.horizontal(|ui| {
        ui.heading("Layers");
        if ui
            .button("+ XYZ demo")
            .on_hover_text("Add a demo OpenStreetMap XYZ raster layer")
            .clicked()
        {
            actions.push(LayerAction::AddDemoXyzLayer);
        }
    });

    // One-click basemap samples with their required credit lines baked in.
    // What makes a sample usable as-is is precisely that picking it installs
    // the attribution its terms demand — EOX's cloudless mosaics, for
    // instance, are free *because* the credit is shown. Hovering an entry
    // shows that credit plus a licence summary. The combo's width is pinned
    // to the row (truncating the text if needed) so it can never overflow the
    // panel — see the ratchet note on the URL rows below.
    ui.horizontal(|ui| {
        ui.label("Basemap");
        // Full-config match, not URL match: the same URL submitted through
        // the "+ XYZ" field carries a generic host credit, and the picker
        // must not imply the preset's required credit is on the map then.
        let current = BASEMAP_PRESETS
            .iter()
            .find(|preset| preset.matches(basemap))
            .map_or("Custom", |preset| preset.name);
        // Honesty: while a promotion DRAWS, no preset is on screen, so the
        // combo names the layer instead of claiming a service it is not
        // showing. Picking any entry demotes the layer (the app records it).
        let selected = basemap_layer
            .filter(|promoted| promoted.drawn)
            .and_then(|promoted| layers.get(promoted.layer))
            .map_or_else(
                || current.to_string(),
                |layer| format!("Layer: {}", layer.name),
            );
        egui::ComboBox::from_id_salt("oxigis_basemap_preset")
            .selected_text(selected)
            .width(ui.available_width())
            .wrap_mode(egui::TextWrapMode::Truncate)
            .show_ui(ui, |ui| {
                for preset in BASEMAP_PRESETS {
                    let entry = ui
                        .selectable_label(preset.name == current, preset.name)
                        .on_hover_text(format!("{}\n{}", preset.attribution, preset.terms));
                    if entry.clicked() {
                        actions.push(LayerAction::SetBasemapPreset(preset.config()));
                    }
                }
            });
    });

    // ONE banner for both slots, under the row that names the map's source: a
    // refusal names a PLAN (a basemap plus at most one raster layer), not a
    // layer, so there is nothing to badge a row with. Right-to-left with a
    // fixed height and a truncating label for the same no-overflow reason as
    // the URL rows below — a shell's error text is arbitrarily long, and a
    // wrapping label would ratchet the panel wider every frame.
    if let Some(reason) = &refusal {
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                if ui
                    .button("Retry")
                    .on_hover_text(
                        "Ask the shell to build the map's tile sources again. \
                         Nothing else changes: this is not an edit and Ctrl+Z \
                         does not undo it.",
                    )
                    .clicked()
                {
                    actions.push(LayerAction::RetryRefusedInstalls);
                }
                ui.add(
                    egui::Label::new(format!("\u{26a0} {reason}"))
                        .wrap_mode(egui::TextWrapMode::Truncate),
                )
                .on_hover_text(reason);
            },
        );
    }

    // Swap the basemap for any XYZ (or WMTS-REST `GoogleMapsCompatible`)
    // service. Placeholders are replaced by name, so `{z}/{y}/{x}` templates —
    // the WMTS REST convention, row before column — work unchanged, and `{-y}`
    // covers TMS row numbering. Right-to-left with a fixed height for the same
    // no-overflow reason as the rows below.
    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| {
            let clicked = ui
                .button("+ XYZ")
                .on_hover_text(
                    "Replace the basemap with an XYZ tile service \
                     ({z}/{x}/{y} in any order; {-y} for TMS rows)",
                )
                .clicked();
            let field = ui.add(
                egui::TextEdit::singleline(xyz_url)
                    .hint_text("Basemap URL template ({z}/{x}/{y})")
                    .desired_width(ui.available_width()),
            );
            let submitted =
                field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            if (submitted || clicked) && !xyz_url.trim().is_empty() {
                actions.push(LayerAction::SetXyzBasemap(xyz_url.trim().to_string()));
            }
        },
    );

    // Minimal "add raster layer" affordance: paste a COG URL, press Enter or the
    // button. Deliberately a plain text field — this crate has no file dialog
    // dependency, and a COG is addressed by URL in every deployment anyway.
    //
    // Right-to-left so the button reserves its width FIRST and the field takes
    // exactly what remains. The previous left-to-right row gave the field
    // `desired_width(f32::INFINITY)` and then appended the button, overflowing
    // the panel by the button's width every frame — egui 0.35 panels persist
    // their content size, so the panel ratcheted wider each repaint until it
    // had swallowed the whole window and the map with it.
    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| {
            let clicked = ui
                .button("+ COG")
                .on_hover_text(
                    "Read a Cloud-Optimized GeoTIFF over HTTP Range requests \
                     (EPSG:3857 or EPSG:4326; the server must allow the Range header)",
                )
                .clicked();
            let field = ui.add(
                egui::TextEdit::singleline(cog_url)
                    .hint_text("Cloud-Optimized GeoTIFF URL")
                    .desired_width(ui.available_width()),
            );
            let submitted =
                field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            if (submitted || clicked) && !cog_url.trim().is_empty() {
                actions.push(LayerAction::AddCogLayer(cog_url.trim().to_string()));
                cog_url.clear();
            }
        },
    );

    // Single-file tile archives. Same right-to-left construction as the COG row
    // above, for the same panel-ratchet reason. The gesture cannot know yet
    // whether the archive holds raster or vector tiles — only its header says —
    // so this reports the URL and the app probes before creating anything.
    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| {
            let open = ui
                .button("Open\u{2026}")
                .on_hover_text(
                    "Open a .pmtiles or .mbtiles file from disk (native builds; \
                     in a browser, drop the file on the map instead)",
                )
                .clicked();
            if open {
                actions.push(LayerAction::OpenArchiveFile);
            }
            let clicked = ui
                .button("+ Archive")
                .on_hover_text(
                    "Read a PMTiles v3 archive over HTTP Range requests (raster or vector; \
                     the server must allow the Range header). Drop a .pmtiles or .mbtiles \
                     file on the map to open one from disk.",
                )
                .clicked();
            let field = ui.add(
                egui::TextEdit::singleline(archive_url)
                    .hint_text("Tile archive URL (.pmtiles)")
                    .desired_width(ui.available_width()),
            );
            let submitted =
                field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            if (submitted || clicked) && !archive_url.trim().is_empty() {
                actions.push(LayerAction::AddArchiveUrlLayer(
                    archive_url.trim().to_string(),
                ));
                archive_url.clear();
            }
        },
    );

    // Same affordance for vector tiles. The buffer is pre-filled by
    // `crate::app::OxigisApp` with the keyless MapLibre demo source, so the
    // button is a one-click demo as well as a URL field. Right-to-left for the
    // same no-overflow reason as the COG row above.
    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| {
            let clicked = ui
                .button("+ Vector")
                .on_hover_text(
                    "Draw Mapbox Vector Tiles (MVT) over the basemap; \
                     defaults to the keyless MapLibre demo tiles",
                )
                .clicked();
            let field = ui.add(
                egui::TextEdit::singleline(vector_url)
                    .hint_text("Vector tile URL template ({z}/{x}/{y}.pbf)")
                    .desired_width(ui.available_width()),
            );
            let submitted =
                field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            if (submitted || clicked) && !vector_url.trim().is_empty() {
                actions.push(LayerAction::AddVectorTileLayer(
                    vector_url.trim().to_string(),
                ));
            }
        },
    );

    // Local vector data. Dropping a `.geojson` file on the map is the primary
    // gesture; this is its keyboard-only twin, for a browser tab with the file
    // already on the clipboard.
    if ui
        .button("+ GeoJSON")
        .on_hover_text(
            "Paste a GeoJSON FeatureCollection as a local layer \
             (or just drop a .geojson file onto the map)",
        )
        .clicked()
    {
        actions.push(LayerAction::AddGeoJsonPaste);
    }
    ui.separator();

    if layers.is_empty() {
        ui.weak("No layers yet. Add one above.");
        return actions;
    }

    // ONE sentence above the list, not a badge per row: the composite cap is a
    // fact about the whole stack ("eight at a time"), and the rows it actually
    // bit are badged individually below. Truncating inside a fixed-height
    // right-to-left strip for the same no-ratchet reason as the refusal banner.
    if let Some(notice) = tiles.and_then(TileStack::notice) {
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                ui.add(
                    egui::Label::new(format!("{WARNING} {notice}"))
                        .wrap_mode(egui::TextWrapMode::Truncate),
                )
                .on_hover_text(&notice);
            },
        );
    }

    let context = RowContext {
        selected,
        promoted: basemap_layer,
        tiles,
        editable,
    };
    egui::ScrollArea::vertical()
        .id_salt("oxigis_layer_tree_scroll")
        .show(ui, |ui| {
            for layer in layers.layers().iter().rev() {
                draw_layer_row(ui, layer, &context, &mut actions, &mut collected);
                ui.separator();
            }
        });

    if let Some(sink) = edits {
        sink.append(&mut collected);
    }
    actions
}

/// Everything a row needs to know beyond its own [`Layer`], bundled so
/// [`draw_layer_row`] keeps a signature a reader can hold in their head.
struct RowContext<'a> {
    /// The selected layer, if any.
    selected: Option<LayerId>,
    /// The project's basemap promotion.
    promoted: Option<PromotedBasemap>,
    /// The map's tiled-draw budget, when the caller derives it.
    tiles: Option<&'a TileStack>,
    /// Whether the caller applies [`LayerEdit`]s — see [`PanelExtras::edits`].
    editable: bool,
}

/// Draws one layer's row (checkbox, name, basemap toggle, reorder/remove
/// buttons, opacity slider), pushing any resulting [`LayerAction`]s onto
/// `actions`.
///
/// `promoted` is the project's basemap promotion, which is why the badge and
/// the suppressed opacity row are built here rather than in [`kind_tag`]:
/// that stays a pure function of the layer, and "is this the basemap" is a
/// fact about the whole project.
fn draw_layer_row(
    ui: &mut Ui,
    layer: &Layer,
    context: &RowContext<'_>,
    actions: &mut Vec<LayerAction>,
    edits: &mut Vec<LayerEdit>,
) {
    let RowContext {
        selected,
        promoted,
        tiles,
        editable,
    } = *context;
    let promotion = promoted.filter(|promoted| promoted.layer == layer.id);
    let settings_id = row_settings_id(layer.id);
    let stored = ui
        .ctx()
        .data_mut(|data| data.get_temp::<RowSettings>(settings_id));
    let mut settings = stored.clone().unwrap_or_default();
    ui.horizontal(|ui| {
        let mut visible = layer.visible;
        if ui.checkbox(&mut visible, "").changed() {
            actions.push(LayerAction::ToggleVisibility(layer.id));
        }

        let is_selected = selected == Some(layer.id);
        // Only a local vector layer has an extent this crate can compute
        // without asking the network, so only its row answers the zoom
        // gesture — the same "a control that is always refused is not an
        // affordance" rule the basemap toggle below follows.
        let zoomable = local_input::is_local_layer(layer);
        let name = ui
            .selectable_label(is_selected, &layer.name)
            .on_hover_text(if zoomable {
                "Click to select this layer; double-click to move the map to its features"
            } else {
                "Click to select this layer"
            });
        if name.clicked() {
            actions.push(LayerAction::Select(layer.id));
        }
        // Deliberately a gesture on the name rather than another row button:
        // this panel sizes itself to its widest row, so one more control per
        // row narrows the map for everyone, permanently, to offer something a
        // double-click already says clearly. A double-click also reports
        // `clicked`, so the layer is selected first and then zoomed to, which
        // is the order the user means.
        if zoomable && name.double_clicked() {
            actions.push(LayerAction::ZoomToLayer(layer.id));
        }
        ui.weak(kind_tag(layer));
        // Provenance, not an instruction: the data was stored in this CRS and
        // reprojected at ingest. The chip is the bare code (bounded width); the
        // hover carries the name the user would recognise.
        if layer.crs.is_some() {
            let crs = layer.source_crs();
            ui.weak(format!("EPSG:{}", crs.epsg()))
                .on_hover_text(format!(
                    "Source data was stored in {} and reprojected to WGS 84 when it was read.",
                    crs.label()
                ));
            // Beside the chip, never instead of it: a historic datum
            // (Tokyo / OSGB36 / ED50 / NAD27) reaches WGS 84 through a
            // published Helmert rather than through the national grid file,
            // so its positions carry a metre-level residual. The chip alone
            // states WHERE the data came from and would let a user read the
            // reprojection as exact; this states how exact it is. A modern
            // datum has no note and therefore no mark, so the common row is
            // unchanged.
            if let Some(note) = crs.accuracy_note() {
                ui.weak(WARNING).on_hover_text(format!(
                    "{note}. Positions are good enough for mapping at this layer's scale, not for \
                     survey-grade work.",
                ));
            }
        }
        if let Some(range) = zoom_range_tag(layer) {
            ui.weak(range).on_hover_text(
                "This layer only draws inside a zoom range. Open its settings to change it.",
            );
        }
        // Listed, visible, and drawing nothing: without this the row's ticked
        // checkbox is a lie the notice above the list cannot pin on anybody.
        if tiles.is_some_and(|tiles| tiles.hides(layer.id)) {
            ui.weak("not drawn").on_hover_text(
                "The map composites a limited number of tiled layers at once and this one is \
                 below the cut. Hide a tiled layer above it to draw this one.",
            );
        }
        if let Some(promotion) = promotion {
            ui.weak(if promotion.drawn {
                "basemap"
            } else {
                "basemap (hidden)"
            });
        }

        // Only an XYZ layer can become the basemap, so only an XYZ row offers
        // the toggle — a control that is always refused is not an affordance.
        if matches!(&layer.kind, LayerKind::Raster(RasterSource::Xyz { .. }))
            && ui
                .selectable_label(promotion.is_some(), BASEMAP_TOGGLE)
                .on_hover_text(
                    "Draw this layer as the basemap, under every other layer. \
                     Rebuilds the tile provider, so the tile cache is dropped.",
                )
                .clicked()
        {
            actions.push(LayerAction::SetBasemapLayer(
                promotion.is_none().then_some(layer.id),
            ));
        }

        // ONE new row control for BOTH per-layer edits, deliberately: this
        // panel sizes itself to its widest row (see the double-click note
        // above), so a rename button and a scale-range button would narrow the
        // map twice over to offer what one disclosure already offers.
        if editable
            && ui
                .selectable_label(settings.open, ELLIPSIS)
                .on_hover_text("Rename this layer and set the zoom range it draws in")
                .clicked()
        {
            settings.open = !settings.open;
            if settings.open {
                // The NAME buffer is seeded on open and never again: re-seeding
                // per frame would overwrite what the user is typing. The range
                // buffers re-seed whenever no gesture is in flight (see
                // `draw_row_settings`), because they have no half-typed state.
                settings.name = layer.name.clone();
                settings.seed_from(layer);
                settings.range_pending = false;
            } else {
                // Closing must not swallow a range the user set with the
                // keyboard and never blurred away from.
                settings.commit_range(layer, edits);
            }
        }

        if ui.small_button(MOVE_UP).on_hover_text("Move up").clicked() {
            actions.push(LayerAction::MoveUp(layer.id));
        }
        if ui
            .small_button(MOVE_DOWN)
            .on_hover_text("Move down")
            .clicked()
        {
            actions.push(LayerAction::MoveDown(layer.id));
        }
        if ui.small_button(REMOVE).on_hover_text("Remove").clicked() {
            actions.push(LayerAction::Remove(layer.id));
        }
    });

    // Drawn BEFORE the promoted-layer early return: a promoted layer has no
    // stack order and no opacity, but it still has a name and a scale range.
    if settings.open {
        draw_row_settings(ui, layer, settings_id, &mut settings, edits);
    }
    // A rename commits by CLOSING the section (see `draw_row_settings`), so a
    // range typed in the same visit would otherwise be stranded in a buffer
    // nothing reads again.
    if !settings.open {
        settings.commit_range(layer, edits);
    }
    // The section is the only thing this row remembers, so it is written back
    // only while it is open — a closed row leaves nothing in egui's store to
    // go stale behind a layer that was removed, hydrated or undone away.
    if settings.open {
        ui.ctx()
            .data_mut(|data| data.insert_temp(settings_id, settings));
    } else if stored.is_some() {
        ui.ctx().data_mut(|data| {
            data.remove_temp::<RowSettings>(settings_id);
        });
    }

    // The basemap is drawn under everything by construction, so a slider that
    // moved neither the layer nor its transparency would be a lie. The line
    // that replaces it says which of the two facts applies.
    if let Some(promotion) = promotion {
        ui.weak(if promotion.drawn {
            "Drawn as the basemap \u{2014} stack order and opacity do not apply."
        } else {
            "Set as the basemap, but hidden \u{2014} stack order and opacity do not apply."
        });
        return;
    }
    let mut opacity = layer.opacity();
    if ui
        .add(egui::Slider::new(&mut opacity, 0.0..=1.0).text("Opacity"))
        .changed()
    {
        actions.push(LayerAction::SetOpacity(layer.id, opacity));
    }
}

/// Draws one row's expanded settings section: the name field and the scale
/// range, each committing through the pure rule that decides whether there is
/// anything to record ([`rename_edit`] / [`zoom_range_edit`]).
///
/// Every strip is a fixed-height right-to-left allocation, exactly like the URL
/// rows at the top of the panel and for the same reason: the control that has a
/// natural width reserves it FIRST and the stretchy one takes what remains, so
/// no frame can overflow the panel and ratchet it wider.
fn draw_row_settings(
    ui: &mut Ui,
    layer: &Layer,
    settings_id: egui::Id,
    settings: &mut RowSettings,
    edits: &mut Vec<LayerEdit>,
) {
    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| {
            let clicked = ui
                .button("Rename")
                .on_hover_text("Rename this layer (Ctrl+Z undoes it)")
                .clicked();
            let field = ui.add(
                egui::TextEdit::singleline(&mut settings.name)
                    // Salted off the row's own key so two rows' fields can
                    // never share focus state.
                    .id(settings_id.with("name"))
                    .hint_text("Layer name")
                    .desired_width(ui.available_width()),
            );
            let submitted =
                field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            if (submitted || clicked)
                && let Some(edit) = rename_edit(layer, &settings.name)
            {
                edits.push(edit);
                // Committing closes the section: the buffer's job is done, and
                // leaving it open would show the OLD name again next frame (the
                // caller applies the edit after this pass).
                settings.open = false;
            }
        },
    );

    // Two independent ends, each a checkbox plus a number, on their own strip.
    // Splitting them is what keeps the strip from overflowing a 230 px panel.
    //
    // While NOTHING is in flight the buffers are re-seeded from the layer every
    // frame, so an undo, a redo or a project load is visible in the editor
    // rather than being clobbered by a stale buffer on the next commit.
    if !settings.range_pending {
        settings.seed_from(layer);
    }
    let RowSettings {
        min_on,
        min,
        max_on,
        max,
        ..
    } = settings;
    // A checkbox is one discrete click, so it commits at once; a `DragValue` is
    // a gesture, so it commits when the gesture ENDS. Recording each frame of a
    // drag would put one undo entry per frame on the stack for one gesture.
    let mut touched = false;
    let mut settled = false;
    for (label, hover, enabled, value) in [
        (
            "Draw from zoom",
            "Lowest zoom at which this layer draws (inclusive)",
            min_on,
            min,
        ),
        (
            "Stop at zoom",
            "Zoom at which this layer stops drawing (exclusive, so a layer that \
             stops at 14 and one that starts at 14 never both draw)",
            max_on,
            max,
        ),
    ] {
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                let number = ui.add_enabled(
                    *enabled,
                    egui::DragValue::new(value)
                        .speed(0.1)
                        .range(0.0..=MAX_ZOOM_LEVEL),
                );
                touched |= number.changed();
                // Both endings, because a `DragValue` is two widgets in one: a
                // drag ends when the pointer lifts, a typed value when the box
                // loses focus. Missing either would strand the edit in the
                // buffer until the section was closed.
                settled |= number.drag_stopped() || number.lost_focus();
                let box_ = ui.checkbox(enabled, label).on_hover_text(hover);
                touched |= box_.changed();
                settled |= box_.changed();
            },
        );
    }
    if touched {
        settings.range_pending = true;
    }
    if settled {
        settings.commit_range(layer, edits);
    }
}

/// Appends a COG raster layer reading `url` and returns its id.
/// Backs [`LayerAction::AddCogLayer`].
pub fn add_cog_layer(layers: &mut LayerStack, url: &str) -> LayerId {
    let name = url
        .rsplit('/')
        .find(|segment| !segment.is_empty())
        .unwrap_or("COG layer");
    let layer = Layer::new(
        name,
        LayerKind::Raster(RasterSource::Cog {
            url: url.to_string(),
        }),
    );
    layers.add(layer)
}

/// Appends a vector-tile layer reading `url_template`, styled by `paints`, and
/// returns its id. Backs [`LayerAction::AddVectorTileLayer`].
///
/// The layer is named after the template's host, which is the only stable,
/// human-meaningful part of a `{z}/{x}/{y}` URL.
pub fn add_vector_tile_layer(
    layers: &mut LayerStack,
    url_template: &str,
    paints: Vec<VectorTilePaint>,
) -> LayerId {
    let name = url_template
        .split_once("://")
        .map_or(url_template, |(_scheme, rest)| rest)
        .split('/')
        .next()
        .filter(|host| !host.is_empty())
        .unwrap_or("Vector tiles");
    let layer = Layer::new(
        format!("{name} (vector)"),
        LayerKind::Vector(VectorSource::MvtTiles {
            url_template: url_template.to_string(),
            paints,
        }),
    );
    layers.add(layer)
}

/// Appends the layer an already-identified tile archive implies, and returns
/// its id.
///
/// **Probe-then-create**: the caller must already hold the archive's
/// [`crate::archive::ArchiveInfo`], because whether this becomes a
/// [`RasterSource::TileArchive`] or a [`VectorSource::TileArchive`] is
/// decided by the archive's own header. Nothing half-decided is ever put in
/// the layer stack, so nothing half-decided can be saved into a project file.
///
/// A vector archive's paint rules are seeded from its declared `vector_layers`
/// by [`crate::archive::archive_paints`]; a raster archive has none.
pub fn add_archive_layer(
    layers: &mut LayerStack,
    archive: &ArchiveRef,
    format: ArchiveFormat,
    info: &crate::archive::ArchiveInfo,
) -> LayerId {
    let name = if info.name.is_empty() {
        archive.file_name().to_owned()
    } else {
        info.name.clone()
    };
    let kind = match info.content {
        crate::archive::ArchiveContent::Raster => LayerKind::Raster(RasterSource::TileArchive {
            archive: archive.clone(),
            format,
            attribution: info.attribution.clone(),
        }),
        crate::archive::ArchiveContent::Vector => LayerKind::Vector(VectorSource::TileArchive {
            archive: archive.clone(),
            format,
            paints: crate::archive::archive_paints(&info.layer_names),
            attribution: info.attribution.clone(),
        }),
    };
    layers.add(Layer::new(name, kind))
}

/// Appends a demo XYZ OpenStreetMap raster layer to `layers` and returns its
/// id. Backs [`LayerAction::AddDemoXyzLayer`].
///
/// The credit is seeded from [`OSM_ATTRIBUTION`], not derived from the host:
/// the OSM tile usage policy requires "© OpenStreetMap contributors", which
/// "© tile.openstreetmap.org" is not — a licence bug, not a cosmetic one.
pub fn add_demo_xyz_layer(layers: &mut LayerStack) -> LayerId {
    let layer = Layer::new(
        "OpenStreetMap (demo)",
        LayerKind::Raster(RasterSource::Xyz {
            url_template: OSM_URL_TEMPLATE.to_string(),
            attribution: OSM_ATTRIBUTION.to_string(),
        }),
    );
    layers.add(layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_demo_xyz_layer_appends_an_osm_raster_layer() {
        let mut layers = LayerStack::new();
        let id = add_demo_xyz_layer(&mut layers);
        let layer = layers.get(id).expect("layer just added must be present");
        assert_eq!(layer.name, "OpenStreetMap (demo)");
        assert!(layer.visible);
        match &layer.kind {
            LayerKind::Raster(RasterSource::Xyz {
                url_template,
                attribution,
            }) => {
                assert!(
                    url_template.contains("{z}")
                        && url_template.contains("{x}")
                        && url_template.contains("{y}")
                );
                assert_eq!(
                    attribution,
                    crate::tile_provider::OSM_ATTRIBUTION,
                    "the OSM tile usage policy requires this exact credit — a \
                     host-derived one would be a licence bug"
                );
            }
            other => panic!("expected a raster XYZ layer, got {other:?}"),
        }
    }

    #[test]
    fn ui_on_an_empty_stack_offers_only_the_add_button() {
        let layers = LayerStack::new();
        egui::__run_test_ui(|ui| {
            let actions = ui_helper(ui, &layers, None);
            assert!(actions.is_empty());
        });
    }

    #[test]
    fn clicking_add_demo_is_reported_as_an_action() {
        let layers = LayerStack::new();
        egui::__run_test_ui(|ui| {
            // Headless test input has no real click; this instead checks
            // that drawing the panel doesn't panic and returns no spurious
            // actions on a frame with no interaction.
            let actions = ui_helper(ui, &layers, None);
            assert!(actions.is_empty());
        });
    }

    #[test]
    fn ui_with_layers_does_not_panic() {
        let mut layers = LayerStack::new();
        let id = add_demo_xyz_layer(&mut layers);
        egui::__run_test_ui(|ui| {
            let actions = ui_helper(ui, &layers, Some(id));
            // No simulated interaction happened, so no actions are expected;
            // the assertion is really "this call completes without panicking".
            assert!(actions.is_empty());
        });
    }

    /// Thin wrapper so tests read `ui_helper(ui, ...)` instead of the
    /// module-qualified `super::ui(ui, ...)` (which would shadow the `ui`
    /// parameter name awkwardly at call sites).
    fn ui_helper(ui: &mut Ui, layers: &LayerStack, selected: Option<LayerId>) -> Vec<LayerAction> {
        let mut cog_url = String::new();
        let mut archive_url = String::new();
        let mut vector_url = String::new();
        let mut xyz_url = String::new();
        let basemap = BasemapConfig::default();
        super::ui(
            ui,
            layers,
            selected,
            MapStatus {
                basemap: &basemap,
                basemap_layer: None,
                refusal: None,
            },
            PanelFields {
                cog_url: &mut cog_url,
                archive_url: &mut archive_url,
                vector_url: &mut vector_url,
                xyz_url: &mut xyz_url,
            },
        )
    }

    /// [`ui_helper`]'s twin for the opt-in entry point, so a test can say what
    /// the tiled-draw budget is doing and collect [`LayerEdit`]s.
    fn extras_helper(
        ui: &mut Ui,
        layers: &LayerStack,
        tiles: Option<&TileStack>,
        edits: &mut Vec<LayerEdit>,
    ) -> Vec<LayerAction> {
        let mut cog_url = String::new();
        let mut archive_url = String::new();
        let mut vector_url = String::new();
        let mut xyz_url = String::new();
        let basemap = BasemapConfig::default();
        super::ui_with_extras(
            ui,
            layers,
            None,
            MapStatus {
                basemap: &basemap,
                basemap_layer: None,
                refusal: None,
            },
            PanelFields {
                cog_url: &mut cog_url,
                archive_url: &mut archive_url,
                vector_url: &mut vector_url,
                xyz_url: &mut xyz_url,
            },
            PanelExtras {
                tiles,
                edits: Some(edits),
            },
        )
    }

    // ---- the scale-range chip and the pure commit rules --------------------

    #[test]
    fn a_scale_range_reads_back_as_a_bounded_chip_or_nothing_at_all() {
        let plain = Layer::new(
            "Roads",
            LayerKind::Raster(RasterSource::xyz("https://x/{z}/{x}/{y}.png")),
        );
        assert_eq!(zoom_range_tag(&plain), None, "no range, no chip");
        assert_eq!(
            zoom_range_tag(&plain.clone().with_zoom_range(Some(14.0), None)).as_deref(),
            Some("z14+")
        );
        assert_eq!(
            zoom_range_tag(&plain.clone().with_zoom_range(None, Some(18.0))).as_deref(),
            Some("z<18")
        );
        assert_eq!(
            zoom_range_tag(&plain.clone().with_zoom_range(Some(14.0), Some(18.0))).as_deref(),
            Some("z14-18")
        );
        // A fractional bound keeps its half rather than being rounded into a
        // lie, and the chip stays short either way.
        let fractional = plain.with_zoom_range(Some(13.5), Some(18.0));
        let tag = zoom_range_tag(&fractional).expect("a range is declared");
        assert_eq!(tag, "z13.5-18");
        assert!(tag.len() <= 12, "the chip must stay row-sized: {tag}");
    }

    #[test]
    fn a_rename_is_trimmed_and_a_no_op_one_is_refused() {
        let layer = Layer::new(
            "Roads",
            LayerKind::Raster(RasterSource::xyz("https://x/{z}/{x}/{y}.png")),
        );
        assert_eq!(
            rename_edit(&layer, "  Roads of Tokyo  "),
            Some(LayerEdit::Rename(layer.id, "Roads of Tokyo".to_string())),
            "the recorded name is what the project will hold, already trimmed"
        );
        // Three shapes of "there is nothing to record", each of which would
        // otherwise cost the user a Ctrl+Z press for no change at all.
        assert_eq!(rename_edit(&layer, "Roads"), None);
        assert_eq!(rename_edit(&layer, "  Roads  "), None);
        assert_eq!(rename_edit(&layer, "   "), None);
        assert_eq!(rename_edit(&layer, ""), None);
    }

    #[test]
    fn a_range_edit_reports_only_a_real_change_and_sanitizes_it_first() {
        let layer = Layer::new(
            "Cadastre",
            LayerKind::Raster(RasterSource::xyz("https://x/{z}/{x}/{y}.png")),
        )
        .with_zoom_range(Some(14.0), Some(18.0));
        assert_eq!(
            zoom_range_edit(&layer, Some(14.0), Some(18.0)),
            None,
            "the range it already has is not an edit"
        );
        assert_eq!(
            zoom_range_edit(&layer, Some(12.0), Some(18.0)),
            Some(LayerEdit::SetZoomRange {
                layer: layer.id,
                min_zoom: Some(12.0),
                max_zoom: Some(18.0),
            })
        );
        // Clearing an end is a real change, and both ends travel together.
        assert_eq!(
            zoom_range_edit(&layer, None, None),
            Some(LayerEdit::SetZoomRange {
                layer: layer.id,
                min_zoom: None,
                max_zoom: None,
            })
        );
        // A hostile value is sanitized by the SAME rule the model applies, so
        // the panel can never propose a range the applier would normalize into
        // something else — which would leave the row showing a value the
        // project does not hold.
        assert_eq!(
            zoom_range_edit(&layer, Some(f32::NAN), Some(18.0)),
            Some(LayerEdit::SetZoomRange {
                layer: layer.id,
                min_zoom: None,
                max_zoom: Some(18.0),
            })
        );
        let rangeless = Layer::new(
            "Roads",
            LayerKind::Raster(RasterSource::xyz("https://x/{z}/{x}/{y}.png")),
        );
        assert_eq!(
            zoom_range_edit(&rangeless, Some(f32::INFINITY), None),
            None,
            "a bound that sanitizes away leaves the layer exactly as it was"
        );
    }

    // ---- the panel's new rows ---------------------------------------------

    #[test]
    fn a_row_that_the_draw_cap_buried_is_badged_and_the_cap_is_explained_once() {
        let mut layers = LayerStack::new();
        let drawn = add_demo_xyz_layer(&mut layers);
        let buried = add_demo_xyz_layer(&mut layers);
        let tiles = TileStack {
            entries: Vec::new(),
            undrawn: vec![buried],
        };
        assert!(tiles.hides(buried) && !tiles.hides(drawn));
        assert!(
            tiles.notice().is_some(),
            "the fixture must actually have the cap biting"
        );
        let mut edits = Vec::new();
        egui::__run_test_ui(|ui| {
            // The assertion is that the badge and the notice draw without
            // reporting anything the user did not do — there is no simulated
            // interaction in a headless pass.
            let actions = extras_helper(ui, &layers, Some(&tiles), &mut edits);
            assert!(actions.is_empty());
        });
        assert!(edits.is_empty());
    }

    #[test]
    fn a_row_draws_its_crs_and_scale_range_without_reporting_anything() {
        let mut layers = LayerStack::new();
        layers.add(
            Layer::new(
                "Tokyo",
                LayerKind::Vector(VectorSource::LocalShapefile {
                    path: "tokyo.shp".to_string(),
                }),
            )
            .with_crs(oxigis_core::crs::Crs::from_epsg(6677))
            .with_zoom_range(Some(14.0), Some(18.0)),
        );
        let mut edits = Vec::new();
        egui::__run_test_ui(|ui| {
            let actions = extras_helper(ui, &layers, None, &mut edits);
            assert!(actions.is_empty());
        });
        assert!(edits.is_empty(), "no simulated interaction, no edits");
    }

    #[test]
    fn a_historic_datum_row_carries_the_accuracy_mark_and_a_modern_one_does_not() {
        // The branch the row draws on: EPSG:4301 is Tokyo Datum, which reaches
        // WGS 84 through a published Helmert and therefore carries a
        // metre-level residual; EPSG:6677 is JGD2011, which does not. The
        // panel must show the mark for the first and nothing for the second —
        // pinned on the model here, because a headless frame cannot be asked
        // which widgets it laid out.
        let historic = oxigis_core::crs::Crs::from_epsg(4301);
        let modern = oxigis_core::crs::Crs::from_epsg(6677);
        let note = historic.accuracy_note().unwrap_or_default();
        assert!(
            note.contains("Tokyo Datum") && note.contains("residual"),
            "the row's hover text has to say WHY the position is approximate: {note}",
        );
        assert_eq!(modern.accuracy_note(), None, "JGD2011 needs no warning");
        assert!(!WARNING.is_empty());

        let mut layers = LayerStack::new();
        for crs in [historic, modern] {
            layers.add(
                Layer::new(
                    "Historic",
                    LayerKind::Vector(VectorSource::LocalShapefile {
                        path: "old.shp".to_string(),
                    }),
                )
                .with_crs(crs),
            );
        }
        let mut edits = Vec::new();
        egui::__run_test_ui(|ui| {
            let actions = extras_helper(ui, &layers, None, &mut edits);
            assert!(actions.is_empty(), "drawing a warning is not an action");
        });
        assert!(edits.is_empty());
    }

    #[test]
    fn the_settings_control_is_not_drawn_for_a_caller_that_applies_no_edits() {
        // A control whose result nothing applies is worse than a missing one,
        // so the narrow entry point must not offer it. Both paths draw the
        // same stack; the frozen one simply opts out.
        let mut layers = LayerStack::new();
        let id = add_demo_xyz_layer(&mut layers);
        egui::__run_test_ui(|ui| {
            let actions = ui_helper(ui, &layers, Some(id));
            assert!(actions.is_empty());
        });
        let mut edits = Vec::new();
        egui::__run_test_ui(|ui| {
            let actions = extras_helper(ui, &layers, None, &mut edits);
            assert!(actions.is_empty());
        });
        assert!(edits.is_empty());
    }

    #[test]
    fn two_rows_never_share_a_settings_buffer() {
        // The buffer is keyed by LAYER id, not by widget position, so
        // scrolling or reordering the list cannot swap two rows' state.
        let mut layers = LayerStack::new();
        let first = add_demo_xyz_layer(&mut layers);
        let second = add_demo_xyz_layer(&mut layers);
        assert_ne!(first, second);
        assert_ne!(
            super::row_settings_id(first),
            super::row_settings_id(second),
        );
    }

    // `the_basemap_toggle_and_banner_glyphs_render_in_egui_s_default_fonts`
    // lived here until editing v1.5. It is SUPERSEDED, not dropped: its three
    // glyphs are rows of `crate::ui_glyphs::ALL`, its explicit-context comment
    // is the header of that module, and its predicate — a positive advance
    // width — is strictly weaker than the replacement's, because the hollow
    // replacement box HAS an advance width and would have passed it. The
    // replacements are `no_ui_glyph_paints_the_replacement_box`,
    // `no_two_ui_glyphs_paint_the_same_slot` and
    // `every_drawn_escape_is_in_the_table`, and they cover every panel rather
    // than this one.

    #[test]
    fn a_promoted_row_draws_its_badge_instead_of_an_opacity_slider() {
        let mut layers = LayerStack::new();
        let id = add_demo_xyz_layer(&mut layers);
        let basemap = BasemapConfig::default();
        let mut cog_url = String::new();
        let mut archive_url = String::new();
        let mut vector_url = String::new();
        let mut xyz_url = String::new();
        for drawn in [true, false] {
            egui::__run_test_ui(|ui| {
                let actions = super::ui(
                    ui,
                    &layers,
                    Some(id),
                    MapStatus {
                        basemap: &basemap,
                        basemap_layer: Some(PromotedBasemap { layer: id, drawn }),
                        refusal: None,
                    },
                    PanelFields {
                        cog_url: &mut cog_url,
                        archive_url: &mut archive_url,
                        vector_url: &mut vector_url,
                        xyz_url: &mut xyz_url,
                    },
                );
                // No simulated interaction: the assertion is that a promoted
                // row draws — badge, toggle and suppressed slider — without
                // reporting anything the user did not do.
                assert!(actions.is_empty());
            });
        }
    }

    #[test]
    fn a_refusal_banner_draws_without_widening_the_panel() {
        let layers = LayerStack::new();
        let mut cog_url = String::new();
        let mut archive_url = String::new();
        let mut vector_url = String::new();
        let mut xyz_url = String::new();
        let basemap = BasemapConfig::default();
        egui::__run_test_ui(|ui| {
            let actions = super::ui(
                ui,
                &layers,
                None,
                MapStatus {
                    basemap: &basemap,
                    basemap_layer: None,
                    // Deliberately far longer than any panel is wide: the
                    // banner truncates rather than ratcheting the panel.
                    refusal: Some("the GPU map is not attached ".repeat(20)),
                },
                PanelFields {
                    cog_url: &mut cog_url,
                    archive_url: &mut archive_url,
                    vector_url: &mut vector_url,
                    xyz_url: &mut xyz_url,
                },
            );
            assert!(actions.is_empty(), "no simulated click, so no Retry");
        });
    }

    #[test]
    fn add_vector_tile_layer_names_the_layer_after_the_host() {
        let mut layers = LayerStack::new();
        let paints = vec![VectorTilePaint::new(
            "countries",
            oxigis_core::LayerStyle::Fill(oxigis_core::FillStyle::new(oxigis_core::Color::WHITE)),
        )];
        let id = add_vector_tile_layer(
            &mut layers,
            "https://demotiles.maplibre.org/tiles/{z}/{x}/{y}.pbf",
            paints.clone(),
        );
        let layer = layers.get(id).expect("layer just added must be present");
        assert_eq!(layer.name, "demotiles.maplibre.org (vector)");
        match &layer.kind {
            LayerKind::Vector(VectorSource::MvtTiles {
                url_template,
                paints: stored,
            }) => {
                assert!(url_template.ends_with(".pbf"));
                assert_eq!(stored, &paints);
            }
            other => panic!("expected an MVT vector layer, got {other:?}"),
        }
    }

    #[test]
    fn add_vector_tile_layer_falls_back_to_a_generic_name() {
        let mut layers = LayerStack::new();
        let id = add_vector_tile_layer(&mut layers, "{z}/{x}/{y}.pbf", Vec::new());
        let layer = layers.get(id).expect("layer just added must be present");
        assert_eq!(layer.name, "{z} (vector)");
    }

    #[test]
    fn add_cog_layer_names_the_layer_after_the_file() {
        let mut layers = LayerStack::new();
        let id = add_cog_layer(&mut layers, "https://example.test/data/scene.tif");
        let layer = layers.get(id).expect("layer just added must be present");
        assert_eq!(layer.name, "scene.tif");
        match &layer.kind {
            LayerKind::Raster(RasterSource::Cog { url }) => {
                assert_eq!(url, "https://example.test/data/scene.tif");
            }
            other => panic!("expected a raster COG layer, got {other:?}"),
        }
    }

    #[test]
    fn add_cog_layer_tolerates_a_trailing_slash() {
        let mut layers = LayerStack::new();
        let id = add_cog_layer(&mut layers, "https://example.test/scene.tif/");
        let layer = layers.get(id).expect("layer just added must be present");
        assert_eq!(layer.name, "scene.tif");
    }

    #[test]
    fn ui_with_a_typed_cog_url_does_not_panic() {
        let layers = LayerStack::new();
        let mut cog_url = "https://example.test/scene.tif".to_string();
        let mut archive_url = String::new();
        let mut vector_url = String::new();
        let mut xyz_url = String::new();
        let basemap = BasemapConfig::default();
        egui::__run_test_ui(|ui| {
            let actions = super::ui(
                ui,
                &layers,
                None,
                MapStatus {
                    basemap: &basemap,
                    basemap_layer: None,
                    refusal: None,
                },
                PanelFields {
                    cog_url: &mut cog_url,
                    archive_url: &mut archive_url,
                    vector_url: &mut vector_url,
                    xyz_url: &mut xyz_url,
                },
            );
            // No simulated click or Enter, so the URL stays in the buffer.
            assert!(actions.is_empty());
            assert_eq!(cog_url, "https://example.test/scene.tif");
        });
    }
}
