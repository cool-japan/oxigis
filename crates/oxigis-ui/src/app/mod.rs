// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! `OxigisApp`: the top-level shell-agnostic UI, tying together the layer
//! tree, style editor, attribute table, and map view panels around a shared
//! [`Project`].
//!
//! The whole UI is driven from a single `ui(&mut self, ui: &mut egui::Ui)`
//! entry point, so a shell is a few lines of glue: hand over the `wgpu`
//! render state once (which attaches the GPU tile map — see
//! [`crate::map_gpu`]) and then call `ui` every frame with the root
//! [`egui::Ui`] that `eframe::App::ui` provides.
//!
//! The type's `impl` spans three files under the 2000-line rule: this one
//! (state, accessors, the frame loop), `data_io` (project load and every
//! add/hydrate/drop seam), and `dispatch` (panel-action application and the
//! Processing window). Unit tests live in the sibling `tests` module
//! (`app/tests.rs`).
//!
//! # Unsaved changes
//!
//! File ▸ New and File ▸ Open are the only two gestures that are not
//! reversible: both replace the project AND call `undo.reset()`, so nothing
//! the user did can be recovered afterwards. They therefore go through
//! `OxigisApp::request_discarding_action`, which asks first whenever
//! [`OxigisApp::has_unsaved_changes`] is true.
//!
//! That answer is a **conservative over-approximation** built from two
//! signals, because there is no single writer to hook: the project-family
//! choke point (`project_edit`) and the seams that bypass it stamp
//! `OxigisApp::mark_project_dirty` directly, and
//! `OxigisApp::observe_recorded_edits` watches the undo log's shape once per
//! frame for everything recorded elsewhere (every feature edit). A change that
//! has since been undone still reports "modified"; the cost of an unnecessary
//! question is one click, the cost of a missed one is the user's work.
//!
//! A shell can also route its **window close** through the same gate with
//! `OxigisApp::request_window_close`, taking the answer back from
//! `OxigisApp::take_confirmed_close`.
//!
//! # Project files
//!
//! This crate compiles to `wasm32` and owns no filesystem, so File ▸ Open /
//! Save / Save As / Open Recent cross to the shell through take-once seams —
//! `take_pending_project_save` and `take_pending_project_open` — exactly as the
//! PDF export and the tile-archive picker already did. The shell resolves the
//! path, moves the bytes, and reports back through one of
//! `confirm_project_saved`, `report_project_save_failed` or
//! `cancel_pending_project_io`; `set_project_path` is what makes the next plain
//! Ctrl+S write back to the same file.
//!
//! All of that is gated on `set_native_project_io`, which is OFF by default.
//! A shell that never turns it on — the browser one — keeps exactly the
//! behaviour it always had: File ▸ Save shows the JSON to copy out, File ▸ Open
//! takes a paste, and no recent list is drawn, because a page cannot re-open a
//! path.

mod archive_io;
mod data_io;
mod dispatch;
mod edit_clipboard;
mod edit_glue;
mod edit_window;
mod print_io;
mod project_edit;
pub mod providers;
mod renderer_ui;

pub use archive_io::{ArchiveProbeRequest, MAX_SESSION_ARCHIVE_BYTES, format_for_url};
pub use data_io::{ProjectOpenRequest, ProjectSaveRequest};
pub use providers::{
    MAX_DRAWN_TILE_LAYERS, RasterWork, Refusal, TileLayerPlan, TileLayerSource, TileStack,
    TileStackWork, VectorWork, draws_as_tile_layer,
};

use crate::edit::EditState;
use crate::edit::stack::EditStack;
use crate::edit::toolbar::EditAction;
use crate::export::ExportRequest;
use crate::layer_panel;
use crate::local_input::{self, LocalInputState, LocalLayerOp, ZOOM_TO_LAYER_MARGIN};
use crate::local_vector::MercatorSquare;
use crate::map_gpu::BoxedTileProvider;
use crate::map_view::MapPanelState;
use crate::measure::{GoToDialog, MeasureSession};
use crate::processing_panel::ProcessingPanelState;
use crate::style_panel::{self, StyleKind};
use crate::table_panel::AttributeTablePanel;
use crate::tile_provider::BasemapConfig;
use crate::vector_provider::VectorTileConfig;
use data_io::{IoDialog, PendingAction};
use eframe::egui_wgpu::RenderState;
use egui::{Context, Panel};
use oxigeo::geojson::types::FeatureCollection;
use oxigis_core::{LayerId, Project, View};
use oxigis_render::{LonLat, MapView};
use oxiui_table::TableEvent;
use std::sync::Arc;
/// Font size of the basemap attribution drawn over the map, in points.
const ATTRIBUTION_FONT_PT: f32 = 11.0;
/// Padding between the attribution text and its backing plate, in points.
const ATTRIBUTION_PAD_PT: f32 = 4.0;
/// Gap between the attribution plate and the map panel's bottom-right corner.
const ATTRIBUTION_MARGIN_PT: f32 = 6.0;
/// Hover text of the Export-PDF vertical-title checkbox.
///
/// A `const` rather than an inline literal so the "no accidental whitespace
/// run" assertion in `app::tests_session` has something to point at: the
/// wrapped form of this sentence shipped once with ~22 spaces in the middle of
/// it, because the line continuations lost their trailing `\`.
const VERTICAL_TITLE_HINT: &str = "Sets the page title top-to-bottom down the right margin. A title \
     containing Latin letters, digits or halfwidth kana prints horizontally \
     instead.";
/// How many projects File ▸ Open Recent lists.
///
/// A menu, not a history: long enough to cover the handful of projects a
/// session moves between, short enough that the submenu stays readable and
/// that a shell persisting the list writes a bounded file.
pub const MAX_RECENT_PROJECTS: usize = 10;
/// Everything about the project that a mutation is guaranteed to move, watched
/// once per frame so edits recorded outside this module still arm the
/// unsaved-changes guard (see the module docs).
///
/// Three independent signals, because none of them alone is enough:
///
/// * the undo log's `epoch` and depths catch every ordinary recorded action —
///   but a push at [`crate::edit::stack::UNDO_MAX_ENTRIES`] also evicts, which
///   leaves both depths where they were;
/// * `bytes` catches most of those evicting pushes — but not the common one,
///   since two consecutive vertex moves of the same feature weigh exactly the
///   same;
/// * `identity` closes it: a feature commit builds its new collection while the
///   old one is still borrowed from the store
///   (`edit_glue`'s `Arc::new(apply_ops(current, …))`), so the replacement can
///   never land on the address it replaced, and the fold provably moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChangeWatermark {
    /// [`EditStack::epoch`] — bumped by the reset a project load performs.
    epoch: u64,
    /// Undoable entries.
    undoable: usize,
    /// Redoable entries.
    redoable: usize,
    /// The log's estimated footprint.
    bytes: usize,
    /// How many layers have their features loaded.
    collections: usize,
    /// Wrapping fold of every loaded collection's `Arc` address.
    identity: usize,
}

/// The top-level OxiGIS UI: current project, selection, panel state, and
/// panel-visibility toggles.
pub struct OxigisApp {
    // The eight fields below carry `pub(crate)` rather than private
    // visibility because two of this type's `impl` blocks live outside the
    // `app` module: `crate::measure` (the map tools) and `crate::export` (the
    // data exports), each of which pairs its `OxigisApp` half with the pure
    // logic it drives so the geodesy and the naming rules can be tested
    // without an app at all. `pub(crate)` is the narrowest visibility that
    // reaches a sibling module of the crate root; none of them is public API.
    /// The project currently open in the editor.
    pub(crate) project: Project,
    /// The currently selected layer, if any (drives the style panel and
    /// highlights the layer-tree row).
    pub(crate) selection: Option<LayerId>,
    /// The central map viewport's camera and painters.
    pub(crate) map_panel: MapPanelState,
    /// The bottom attribute-table panel's persistent widget state.
    pub(crate) table_panel: AttributeTablePanel,
    /// Whether the bottom attribute-table panel is expanded.
    show_table: bool,
    /// Whether the Help ▸ About window is open.
    show_about: bool,
    /// The active File ▸ New/Open/Save modal, if any.
    io_dialog: Option<IoDialog>,
    /// A short status line shown in the menu bar (e.g. "Project loaded.").
    pub(crate) status: Option<String>,
    /// Whether the GPU tile pipeline is attached, i.e. whether the central
    /// panel pushes the `egui_wgpu` map callback (the real renderer) or falls
    /// back to the egui-native tile-placement preview. Set by
    /// [`Self::attach_gpu_map`]; `false` until a shell hands over a `wgpu`
    /// render state, because a paint callback with no `wgpu` renderer behind it
    /// would leave the panel empty. [`Self::ui`] allocates the rect and applies
    /// pan/zoom either way.
    map_gpu: bool,
    /// The central panel's rect from the most recent [`Self::ui`] call, if
    /// any frame has run yet — the rect a GPU shell attaches its
    /// `egui_wgpu::Callback` to.
    map_rect: Option<egui::Rect>,
    /// Why [`Self::attach_gpu_map_with`] failed, once it has, so the failure is
    /// not retried every frame. Without the latch an install error would re-run
    /// the shell's provider factory ~60 times a second, which for the native
    /// shell means starting — and immediately abandoning — a worker pool per
    /// frame.
    ///
    /// The reason is kept rather than only logged: it is folded into
    /// [`Self::provider_refusal`], so the failure reaches the same banner (with
    /// the same Retry button) every refused provider does, and into the
    /// fallback painter's note, which otherwise blames a missing render state
    /// that demonstrably existed. [`Self::retry_refused_installs`] clears it.
    map_gpu_failed: Option<String>,
    /// Which raster basemap the map draws, and the credit line it must show.
    /// Read by a shell when it builds the tile provider, and by [`Self::ui`]
    /// when it paints the attribution.
    basemap: BasemapConfig,
    /// Text buffer of the layer panel's "set XYZ basemap" field, kept here so
    /// the panel itself stays stateless.
    xyz_url_input: String,
    /// Text buffer of the layer panel's "add COG" field, kept here so the
    /// panel itself stays stateless.
    cog_url_input: String,
    /// Text buffer of the layer panel's "add tile archive" field, kept here so
    /// the panel itself stays stateless.
    archive_url_input: String,
    /// Text buffer of the layer panel's "add vector tiles" field, pre-filled
    /// with the keyless MapLibre demo source so the button is a one-click demo.
    vector_url_input: String,
    /// An archive a shell still has to build a range transport for — take-once
    /// through [`Self::take_pending_archive_probe`].
    pending_archive_request: Option<archive_io::ArchiveProbeRequest>,
    /// What the probe currently in flight was asked for, kept so the layer can
    /// be created against the reference the user gave rather than one
    /// reconstructed from the probe.
    probing: Option<archive_io::ArchiveProbeRequest>,
    /// The archive header read a shell has in flight, polled once per frame by
    /// [`Self::poll_archive_probe`].
    archive_probe: Option<crate::archive::ArchiveProbe>,
    /// Whether the user asked for a file dialog a shell still owes them —
    /// take-once through [`Self::take_pending_archive_pick`].
    pending_archive_pick: bool,
    /// Dropped archives whose bytes this session is holding, oldest first.
    ///
    /// A browser drop arrives as bytes and there is no path to re-read, so the
    /// bytes must outlive the drop for as long as the layer is drawn. Bounded
    /// by [`archive_io::MAX_SESSION_ARCHIVE_BYTES`]; a `Vec` rather than a map
    /// because it is at most a handful of entries and eviction wants the order.
    session_archives: Vec<(String, std::sync::Arc<[u8]>)>,
    /// MBTiles archives this session has opened and indexed, keyed the same
    /// way as [`Self::session_archives`] and pruned with it.
    ///
    /// Separate from the bytes because the *index* is what a lookup needs and
    /// it costs one walk of the whole b-tree to build; rebuilding it on every
    /// provider reconciliation would pay that again for nothing.
    session_readers: Vec<(String, std::sync::Arc<crate::mbtiles::MbTilesReader>)>,
    /// The raster work a shell last CONFIRMED installed — GPU state, never
    /// project state, written only by [`Self::settle_raster_work`].
    /// `load_project`/`new_project` must NOT reset it, or a load whose
    /// basemap equals the active one would rebuild the provider and blank +
    /// re-fetch every visible tile for nothing (see `app/providers.rs`).
    raster_installed: Option<providers::RasterWork>,
    /// A raster plan a shell reported it could not build, memoized with the
    /// reason so the same plan is not retried every frame. Cleared by any
    /// successful settle, by the GPU-map attach seam and by
    /// [`Self::retry_refused_installs`]; superseded by any project change
    /// that implies a new plan.
    raster_refused: Option<providers::Refusal<providers::RasterWork>>,
    /// The vector-tile source a shell last confirmed installed ([`None`] =
    /// confirmed detached, which a fresh `MapGpuState` is by construction).
    vector_installed: Option<VectorTileConfig>,
    /// The vector twin of [`Self::raster_refused`].
    vector_refused: Option<providers::Refusal<providers::VectorWork>>,
    /// Whether the shell reconciles through the N-layer tile stack rather than
    /// through the two legacy single-slot seams — declared once at startup with
    /// [`Self::set_tile_stack_shell`], never written by an edit.
    ///
    /// Off by default, so an out-of-tree shell that has not migrated keeps the
    /// exact behaviour it had.
    tile_stack_shell: bool,
    /// Set by the refusal banner's Retry so a stack shell knows to drop the
    /// map's own refused entries too — take-once through
    /// [`Self::take_tile_layer_retry`].
    ///
    /// A flag rather than a direct call because the refused entries live in the
    /// GPU state and only the shell holds the `RenderState`; the app half of the
    /// same click ([`Self::retry_refused_installs`]) runs immediately.
    pending_tile_layer_retry: bool,
    /// What the drawn stack last told the shell it could not build, refreshed
    /// once per frame through [`Self::set_tile_layer_refusals`] — a report
    /// cache read only by the banner, never reconciled against.
    tile_layer_refusals: Vec<(LayerId, String)>,
    /// Per local layer, the `(checkbox, range-resolved)` visibility pair last
    /// pushed into its GPU mirror — see [`Self::sync_local_zoom_visibility`],
    /// which is the only reader and the only writer.
    ///
    /// Bounded by the project's layer count: entries are dropped with their
    /// layers on every pass.
    local_visibility_pushed: std::collections::BTreeMap<LayerId, (bool, bool)>,
    /// Local (dropped / pasted / project-loaded) vector layers — GeoJSON and
    /// Shapefile alike: the GPU work waiting for a shell and the native paths
    /// still to be read. See
    /// [`crate::local_input`] for why this is a queue rather than a direct call.
    pub(crate) local: LocalInputState,
    /// Whether the Processing ▸ Toolbox window is open.
    show_processing: bool,
    /// The built-in Processing tools available this run (`bounds` /
    /// `feature_count` today; see [`oxigis_core::builtin_registry`]).
    processing_registry: oxigis_core::ProcessingRegistry,
    /// The Processing window's persistent form/result state across frames.
    processing: ProcessingPanelState,
    /// Feature editing: the active tool, the picked feature/vertex, and
    /// anything being sketched or dragged. See [`crate::edit`].
    pub(crate) edit: EditState,
    /// The undo/redo log for feature edits. Deliberately **one** stack for the
    /// whole app rather than one per layer: `Ctrl+Z` must undo the last thing
    /// the user did, not the last thing on whichever layer is selected now.
    undo: EditStack,
    /// A PDF export the user just asked for, waiting for a shell to fetch
    /// tiles and assemble the file. Taken by [`Self::take_pending_print`] —
    /// the same shell seam as [`Self::take_pending_basemap`]: `oxigis-ui`
    /// compiles to wasm and owns no transport, no filesystem and no downloads.
    pending_print: Option<crate::print::PrintRequest>,
    /// The Export-PDF options as last set in the dialog — remembered across
    /// exports so a re-export needs no re-configuration.
    print_options: crate::print::PrintOptions,
    /// The style panel's cross-frame state (which slot of the selected
    /// layer's style set the editor is bound to).
    style_panel_state: style_panel::StylePanelState,
    /// The RENDERER editor's cross-frame state (thematic v1.6): the classify
    /// form's field, class count, break rule and last notice.
    ///
    /// Held beside [`Self::style_panel_state`] rather than folded into it
    /// because it carries a `String` and is therefore not `Copy` — see
    /// [`crate::style_panel::StylePanelState`]'s own note on the split.
    renderer_panel_state: crate::renderer_panel::RendererPanelState,
    /// The attribute keys the renderer's field picker offers, cached against
    /// the collection they were derived from — see [`renderer_ui`].
    renderer_fields: Option<renderer_ui::RendererFields>,
    /// Whether the Export-PDF options window is on screen.
    pub(crate) print_dialog_open: bool,
    /// Whether an egui widget held keyboard focus on the **previous** frame.
    ///
    /// The edit shortcuts' focus guard needs one frame of memory because egui
    /// 0.35 clears the focused widget for a bare `Escape` in
    /// `Focus::begin_pass` — before any app code runs — so on exactly the
    /// frame a text field's Escape arrives, `memory.focused()` already reads
    /// [`None`] and the guard would wave the key through to the edit cancel
    /// ladder. See [`Self::edit_shortcuts`].
    edit_focus_last_frame: bool,
    /// Monotone count of project mutations this session has observed — the
    /// left-hand side of "are there unsaved changes?".
    ///
    /// Bumped by [`Self::mark_project_dirty`] from the choke point every
    /// project-family transaction goes through, from the unrecorded writers
    /// that bypass it (a visibility toggle, an add seam), and from
    /// [`Self::observe_recorded_edits`], which watches the undo log's shape so
    /// that feature edits recorded by code outside this module are caught too.
    /// Deliberately over-approximate: undoing back to the saved state still
    /// reports "modified", because asking a question the user can answer beats
    /// discarding work.
    project_revision: u64,
    /// [`Self::project_revision`] as of the last load / File ▸ New / save.
    saved_revision: u64,
    /// The project's observable shape as of the last
    /// [`Self::observe_recorded_edits`] call.
    undo_watermark: ChangeWatermark,
    /// A restyle whose GPU half is waiting for the user to let go of the
    /// pointer, because the layer is too big to re-tessellate every frame of a
    /// drag. See [`crate::app::OxigisApp::sync_local_style`].
    deferred_restyle: Option<LayerId>,
    /// Where the open project lives on disk, when the shell has told us.
    ///
    /// [`None`] for a project that has never been written, one that arrived by
    /// paste or drop, and every project in a shell with no filesystem — which
    /// is exactly the set File ▸ Save has to ask a destination for.
    project_path: Option<std::path::PathBuf>,
    /// Recently opened and saved projects, most recent first, capped at
    /// [`MAX_RECENT_PROJECTS`].
    ///
    /// This is the in-session source of truth: the shell seeds it from its own
    /// store at startup ([`Self::set_recent_projects`]) and reads it back when
    /// it persists ([`Self::recent_projects`]), and nothing else writes it.
    recent_projects: Vec<std::path::PathBuf>,
    /// Whether the shell can read and write project files itself.
    ///
    /// `false` — the default, and what the browser shell keeps — routes
    /// File ▸ Save to the copy-JSON modal and File ▸ Open to the paste box,
    /// which is the only thing a page without a filesystem can offer. A native
    /// shell declares `true` through [`Self::set_native_project_io`] and gets
    /// the real seams below instead.
    native_project_io: bool,
    /// A project write the shell has been asked to perform — take-once through
    /// [`Self::take_pending_project_save`].
    pending_project_save: Option<data_io::ProjectSaveRequest>,
    /// A project read the shell has been asked to perform — take-once through
    /// [`Self::take_pending_project_open`].
    pending_project_open: Option<data_io::ProjectOpenRequest>,
    /// What runs once the pending save is reported successful: the "Save"
    /// answer to the unsaved-changes question parks the gesture here.
    ///
    /// Cleared by every other outcome — a cancelled dialog, a failed write, a
    /// project that will not serialize — because an action that outlived its
    /// save would fire on the *next* unrelated one and destroy the project.
    after_save: Option<PendingAction>,
    /// A window close the user has confirmed, waiting for the shell to act on
    /// it — take-once through [`Self::take_confirmed_close`].
    close_confirmed: bool,
    /// Memoized distinct-property-key count of the selected layer's collection
    /// (layer, the exact collection counted, count), for the attribute form's
    /// column cap when the attribute table is not bound to the selected layer
    /// — most plainly when the table panel is hidden. Keyed on the
    /// collection's [`Arc`] identity, the same staleness scheme the table
    /// panel itself uses. See `edit_glue`'s `form_schema_len`.
    form_schema_memo: Option<(LayerId, Arc<FeatureCollection>, usize)>,
    /// The measuring tape: whether it is armed, and what it has measured. See
    /// [`crate::measure`] for the geodesy and `app::map_tools` for the
    /// gesture.
    pub(crate) measure: MeasureSession,
    /// The Go-to-coordinate window's form state.
    pub(crate) go_to: GoToDialog,
    /// A file the shell has been asked to write — take-once through
    /// [`Self::take_pending_export`], the same seam
    /// [`Self::take_pending_project_save`] is. See [`crate::export`].
    pub(crate) pending_export: Option<ExportRequest>,
}
impl OxigisApp {
    /// Starts a fresh, untitled project with no layers and nothing selected.
    #[must_use]
    pub fn new() -> Self {
        let project = Project::new("Untitled project");
        let mut app = Self {
            map_panel: MapPanelState::default(),
            project,
            selection: None,
            table_panel: AttributeTablePanel::default(),
            show_table: true,
            show_about: false,
            io_dialog: None,
            status: None,
            map_gpu: false,
            map_rect: None,
            map_gpu_failed: None,
            basemap: BasemapConfig::default(),
            xyz_url_input: String::new(),
            cog_url_input: String::new(),
            archive_url_input: String::new(),
            vector_url_input: VectorTileConfig::maplibre_demo().url_template,
            pending_archive_request: None,
            probing: None,
            archive_probe: None,
            pending_archive_pick: false,
            session_archives: Vec::new(),
            session_readers: Vec::new(),
            raster_installed: None,
            raster_refused: None,
            vector_installed: None,
            vector_refused: None,
            tile_stack_shell: false,
            pending_tile_layer_retry: false,
            tile_layer_refusals: Vec::new(),
            local_visibility_pushed: std::collections::BTreeMap::new(),
            local: LocalInputState::new(),
            show_processing: false,
            processing_registry: oxigis_core::builtin_registry(),
            processing: ProcessingPanelState::default(),
            edit: EditState::default(),
            undo: EditStack::new(),
            pending_print: None,
            print_options: crate::print::PrintOptions::default(),
            style_panel_state: style_panel::StylePanelState::default(),
            renderer_panel_state: crate::renderer_panel::RendererPanelState::new(),
            renderer_fields: None,
            print_dialog_open: false,
            edit_focus_last_frame: false,
            project_revision: 0,
            saved_revision: 0,
            undo_watermark: ChangeWatermark {
                epoch: 0,
                undoable: 0,
                redoable: 0,
                bytes: 0,
                collections: 0,
                identity: 0,
            },
            deferred_restyle: None,
            project_path: None,
            recent_projects: Vec::new(),
            native_project_io: false,
            pending_project_save: None,
            pending_project_open: None,
            after_save: None,
            close_confirmed: false,
            form_schema_memo: None,
            measure: MeasureSession::default(),
            go_to: GoToDialog::default(),
            pending_export: None,
        };
        app.apply_project_view();
        app.undo_watermark = app.undo_shape();
        app
    }
    /// Whether the project has changed since it was last loaded, created or
    /// saved — what File ▸ New and File ▸ Open ask before throwing it away,
    /// and what a shell puts the `*` in its window title from.
    ///
    /// Deliberately conservative: it answers `true` for a change that has since
    /// been undone, because the cost of an unnecessary confirmation is one
    /// click and the cost of a missed one is the user's work.
    #[must_use]
    pub fn has_unsaved_changes(&self) -> bool {
        self.project_revision != self.saved_revision
    }

    /// Declares the project saved as it now stands — the seam a shell calls
    /// after its own File ▸ Save wrote the bytes out, and what the in-app
    /// Save… modal calls when the JSON is on screen to be copied.
    ///
    /// Closing the coalescing window is part of saving, not a detail: a slider
    /// drag that continues across the save FOLDS into the entry above it,
    /// changing neither the log's depth nor its byte count, which is exactly
    /// the shape `Self::observe_recorded_edits` cannot see. Closing the
    /// window forces the next edit to push a fresh entry, which it can.
    pub fn mark_saved(&mut self) {
        self.undo.close_coalescing();
        self.undo_watermark = self.undo_shape();
        self.saved_revision = self.project_revision;
    }

    /// Records that the project has been mutated. Called from the
    /// project-family choke point and from the writers that bypass it.
    pub(super) fn mark_project_dirty(&mut self) {
        self.project_revision = self.project_revision.wrapping_add(1);
    }

    /// The project's observable shape — see [`ChangeWatermark`].
    ///
    /// `O(layers)` with one map lookup each, so it is cheap enough to take
    /// every frame; the collections themselves are never walked.
    fn undo_shape(&self) -> ChangeWatermark {
        let (undoable, redoable) = self.undo.depth();
        let mut collections = 0;
        let mut identity = 0_usize;
        for layer in self.project.layers.layers() {
            if let Some(features) = self.local.feature_set(layer.id) {
                collections += 1;
                identity = identity.wrapping_add(Arc::as_ptr(features) as usize);
            }
        }
        ChangeWatermark {
            epoch: self.undo.epoch(),
            undoable,
            redoable,
            bytes: self.undo.bytes(),
            collections,
            identity,
        }
    }

    /// Re-baselines the watermark over a change that is **not** a modification
    /// of the project: a `hydrate_*` completing a path-referenced layer the
    /// loaded file already named.
    ///
    /// Callers must have run [`Self::observe_recorded_edits`] **before** the
    /// change, so anything outstanding from before it is captured rather than
    /// swallowed by this re-baseline — the hydrate seams do exactly that, on
    /// their first line.
    pub(super) fn absorb_project_change(&mut self) {
        self.undo_watermark = self.undo_shape();
    }

    /// Notices edits recorded by code that does not go through the project
    /// choke point — every feature edit, whose applier lives in `edit_glue`
    /// and whose only visible trace here is a new entry on the undo log.
    ///
    /// Called once per frame, before anything can ask
    /// [`Self::has_unsaved_changes`], so the answer is at most one frame behind
    /// the gesture and never behind the click that consumes it.
    fn observe_recorded_edits(&mut self) {
        let shape = self.undo_shape();
        if shape != self.undo_watermark {
            self.undo_watermark = shape;
            self.mark_project_dirty();
        }
    }

    /// Declares that this shell can read and write project files itself.
    ///
    /// Off by default, which is the browser's honest answer: File ▸ Save shows
    /// the JSON to copy out and File ▸ Open takes a paste. A native shell turns
    /// it on once at startup, which switches File ▸ Save/Save As/Open/Open
    /// Recent onto the take-once seams below and makes the window-close
    /// intercept meaningful.
    pub fn set_native_project_io(&mut self, enabled: bool) {
        self.native_project_io = enabled;
    }

    /// Whether the shell declared real project file I/O — see
    /// [`Self::set_native_project_io`].
    #[must_use]
    pub fn native_project_io(&self) -> bool {
        self.native_project_io
    }

    /// The file the open project came from, if it has one.
    #[must_use]
    pub fn project_path(&self) -> Option<&std::path::Path> {
        self.project_path.as_deref()
    }

    /// Tells the app where the open project lives — what a shell calls right
    /// after its own Open dialog read a file, so a later plain Ctrl+S writes
    /// back to the same place.
    ///
    /// Also records the file in the recent list, since a project that was just
    /// opened is by definition the most recent one.
    pub fn set_project_path(&mut self, path: Option<std::path::PathBuf>) {
        if let Some(path) = path.clone() {
            self.note_recent_project(path);
        }
        self.project_path = path;
    }

    /// The recent-project list, most recent first — what a shell persists.
    #[must_use]
    pub fn recent_projects(&self) -> &[std::path::PathBuf] {
        &self.recent_projects
    }

    /// Seeds the recent list from a shell's own store at startup. Bounded and
    /// de-duplicated on the way in, so a hand-edited store cannot grow the
    /// menu without limit.
    pub fn set_recent_projects(&mut self, paths: Vec<std::path::PathBuf>) {
        self.recent_projects.clear();
        for path in paths {
            if path.as_os_str().is_empty() || self.recent_projects.contains(&path) {
                continue;
            }
            if self.recent_projects.len() == MAX_RECENT_PROJECTS {
                break;
            }
            self.recent_projects.push(path);
        }
    }

    /// Moves `path` to the front of the recent list, keeping it capped.
    pub(super) fn note_recent_project(&mut self, path: std::path::PathBuf) {
        if path.as_os_str().is_empty() {
            return;
        }
        self.recent_projects.retain(|existing| existing != &path);
        self.recent_projects.insert(0, path);
        self.recent_projects.truncate(MAX_RECENT_PROJECTS);
    }

    /// A project write the shell is being asked to perform, if one is waiting
    /// — take-once, like [`Self::take_pending_print`].
    ///
    /// The shell writes `content` to `path` (asking the user where when it is
    /// [`None`]) and then reports the outcome through exactly one of
    /// [`Self::confirm_project_saved`], [`Self::report_project_save_failed`] or
    /// [`Self::cancel_pending_project_io`]. Not reporting leaves the project
    /// marked unsaved, which is the safe direction to fail in.
    pub fn take_pending_project_save(&mut self) -> Option<data_io::ProjectSaveRequest> {
        self.pending_project_save.take()
    }

    /// A project read the shell is being asked to perform, if one is waiting.
    ///
    /// The unsaved-changes question has already been asked and answered. The
    /// shell reads the file, hands the document to [`Self::load_project`] (or
    /// [`Self::load_geolibre_project`]) and then calls
    /// [`Self::set_project_path`]; a cancelled dialog goes to
    /// [`Self::cancel_pending_project_io`].
    pub fn take_pending_project_open(&mut self) -> Option<data_io::ProjectOpenRequest> {
        self.pending_project_open.take()
    }

    /// Whether the user has confirmed the window may close — take-once.
    ///
    /// A shell intercepting its own close request calls
    /// [`Self::request_window_close`] and cancels the close; this answers
    /// `true` on the frame the user said Discard (or immediately, for a project
    /// with nothing to lose), which is the shell's cue to let the close through
    /// and stop intercepting.
    pub fn take_confirmed_close(&mut self) -> bool {
        core::mem::take(&mut self.close_confirmed)
    }

    /// Reports that the project's bytes are on disk at `path`.
    ///
    /// Clears the unsaved-changes marker, remembers the path for the next plain
    /// Ctrl+S, files it in the recent list — and runs whatever the
    /// unsaved-changes question parked behind the save (New, Open, quit).
    pub fn confirm_project_saved(&mut self, path: std::path::PathBuf) {
        self.status = Some(format!("Project saved to {}.", path.display()));
        self.note_recent_project(path.clone());
        self.project_path = Some(path);
        self.mark_saved();
        if let Some(pending) = self.after_save.take() {
            self.run_pending_action(pending);
        }
    }

    /// Reports that the write failed. The project stays marked unsaved and
    /// anything parked behind the save is dropped.
    pub fn report_project_save_failed(&mut self, error: &str) {
        self.after_save = None;
        self.status = Some(format!("Save failed: {error}"));
    }

    /// Reports that the user dismissed the shell's own file dialog.
    ///
    /// Distinct from a failure on purpose: cancelling is not an error worth an
    /// alarming status line, but it must still drop the parked action — a
    /// "Save, then quit" whose save the user cancelled has to leave the app
    /// exactly where it was, and must not fire behind the *next* successful
    /// save.
    pub fn cancel_pending_project_io(&mut self) {
        self.after_save = None;
        self.status = Some("Cancelled.".to_string());
    }

    /// Asks the unsaved-changes question in front of File ▸ Open.
    ///
    /// The guarded route: a project with unsaved changes gets the
    /// Save / Discard / Cancel modal first, and only a confirmed discard
    /// reaches [`Self::take_pending_project_open`]. Public for the same reason
    /// [`Self::request_project_save`] is — a shell with its own platform menu
    /// bar must be able to offer the gesture without reimplementing the guard.
    pub fn request_project_open(&mut self) {
        self.request_discarding_action(PendingAction::OpenProject);
    }

    /// Asks the unsaved-changes question in front of File ▸ New.
    pub fn request_new_project(&mut self) {
        self.request_discarding_action(PendingAction::NewProject);
    }

    /// Asks the unsaved-changes question in front of a window close.
    ///
    /// The shell calls this on its close-request intercept and keeps refusing
    /// the close until [`Self::take_confirmed_close`] answers `true`. A clean
    /// project confirms immediately, so a user with nothing to lose sees no
    /// dialog at all.
    pub fn request_window_close(&mut self) {
        self.request_discarding_action(PendingAction::CloseWindow);
    }

    /// Marks the project saved-as-loaded: a fresh load, or File ▸ New. The
    /// revision still moves (the *session* changed), but the saved marker moves
    /// with it, so the new project starts clean.
    pub(super) fn mark_project_loaded(&mut self) {
        self.mark_project_dirty();
        self.undo_watermark = self.undo_shape();
        self.saved_revision = self.project_revision;
    }

    /// The current project (read-only access for shells that want to
    /// display the title, save on their own file-dialog, etc.).
    ///
    /// [`Project::view`] reflects the last time the camera was synced into
    /// the project (construction, [`Self::load_project`], or a File ▸ Save)
    /// — call [`Self::map_view`] for the live, frame-by-frame camera.
    #[must_use]
    pub fn project(&self) -> &Project {
        &self.project
    }
    /// The currently selected layer, if any.
    #[must_use]
    pub fn selection(&self) -> Option<LayerId> {
        self.selection
    }
    /// The map panel's current camera, in the shape
    /// [`oxigis_render::MapRenderer::begin_frame`] expects. A shell driving
    /// its own GPU pipeline around this app's `ui(ctx)` call reads this
    /// after each frame (see `map_view`'s module docs for the full seam).
    #[must_use]
    pub fn map_view(&self) -> MapView {
        self.map_panel.view()
    }
    /// Moves the map panel's camera to [`Project::view`]'s center/zoom,
    /// keeping the panel's current on-screen size (which [`Project::view`]
    /// doesn't record — it's an egui layout detail, not project state).
    fn apply_project_view(&mut self) {
        let view = self.project.view;
        let updated = self
            .map_panel
            .view()
            .with_center(LonLat::new(view.center_lon, view.center_lat))
            .with_zoom(view.zoom);
        self.map_panel.set_view(updated);
    }
    /// Writes the map panel's current camera back into [`Project::view`] —
    /// and the active basemap into [`Project::basemap`] — so a subsequent
    /// [`Project::to_json_string`] captures what the user was actually
    /// looking at rather than the project's last-loaded/default state.
    ///
    /// [`Self::ui`]'s own File ▸ Save… already calls this before
    /// serializing. A shell that saves through its own native file dialog
    /// (reading [`Self::project`] directly, bypassing the in-app Save…
    /// modal) **must** call this first, or the saved file's view will be
    /// stale — this is public for exactly that reason. It must also call
    /// [`Self::mark_saved`] once the bytes are on disk, or File ▸ New will go
    /// on asking about changes the user has already saved.
    pub fn sync_project_view(&mut self) {
        let camera = self.map_panel.view();
        let center = camera.center();
        self.project.view = View {
            center_lon: center.lon,
            center_lat: center.lat,
            zoom: camera.zoom(),
        };
        self.project.basemap = Some((&self.basemap).into());
    }
    /// The central map panel's rect from the most recent [`Self::ui`] call
    /// (`None` before the first frame). A GPU shell attaches its
    /// `egui_wgpu::Callback` to this rect — see `map_view`'s module docs for
    /// the full integration seam.
    #[must_use]
    pub fn map_rect(&self) -> Option<egui::Rect> {
        self.map_rect
    }
    /// Attaches the `wgpu` tile-map pipeline, so the central panel draws the
    /// real [`oxigis_render::MapRenderer`] output instead of the fallback
    /// preview.
    ///
    /// Call this once per frame from the shell with
    /// `eframe::Frame::wgpu_render_state()`: it is idempotent (the first call
    /// installs [`crate::map_gpu::MapGpuState`]; later ones only re-check) and
    /// `None` — a `glow` host, or a headless context in tests — selects the
    /// fallback painter. Tiles come from
    /// [`crate::map_gpu::DebugCheckerboard`] until a shell swaps in a real
    /// source with [`crate::map_gpu::replace_provider`].
    ///
    /// An installation failure is logged and leaves the fallback in place
    /// rather than propagating, so a shell's frame loop needs no error path.
    pub fn attach_gpu_map(&mut self, render_state: Option<&RenderState>) {
        self.attach_gpu_map_with(render_state, || None);
    }
    /// Same as [`Self::attach_gpu_map`], but with the real tile source the
    /// shell can provide.
    ///
    /// `make_provider` is called **at most once per attach attempt**: on the
    /// frame that actually installs the pipeline, and only after it is certain
    /// that the provider will be used — so a native transport does not spin up
    /// worker threads that are immediately dropped, and an install failure does
    /// not re-run the factory on every following frame. A successful attach is
    /// final; a *failed* one latches (see [`Self::provider_refusal`]) until the
    /// user presses the banner's Retry, which clears the latch and lets exactly
    /// one further attempt — and therefore one further `make_provider` call —
    /// through.
    /// Returning [`None`] (or being called on a
    /// frame where nothing needs installing) keeps
    /// [`crate::map_gpu::DebugCheckerboard`] as the fallback, which is also
    /// what happens if the shell's provider construction fails: the map still
    /// paints, just synthetically.
    ///
    /// Typical shell use, once per frame:
    ///
    /// ```text
    /// let ctx = ui.ctx().clone();
    /// let config = self.app.basemap().clone();
    /// self.app.attach_gpu_map_with(frame.wgpu_render_state(), || {
    ///     build_platform_provider(&config, &ctx)
    /// });
    /// ```
    pub fn attach_gpu_map_with<F>(&mut self, render_state: Option<&RenderState>, make_provider: F)
    where
        F: FnOnce() -> Option<BoxedTileProvider>,
    {
        self.attach_gpu_map_using(render_state, |_basemap| make_provider());
    }
    /// [`Self::attach_gpu_map_with`], with the active [`BasemapConfig`] handed
    /// to the factory instead of the shell having to capture a copy of it.
    ///
    /// The distinction is not cosmetic. `attach_gpu_map_with` borrows the app
    /// mutably, so a shell whose factory needs the basemap has to clone it
    /// *before* the call — and, because the call happens every frame while the
    /// factory runs at most once per attach, that clone (a `String` template, a
    /// `Vec<String>` of subdomains and an attribution `String`) is paid sixty
    /// times a second for the lifetime of the process, to be dropped unread.
    /// Passing the borrow through the closure costs nothing and is exact.
    ///
    /// Both entry points are kept: `attach_gpu_map_with` is what the browser
    /// shell calls, and its closure genuinely wants no argument.
    pub fn attach_gpu_map_using<F>(&mut self, render_state: Option<&RenderState>, make_provider: F)
    where
        F: FnOnce(&BasemapConfig) -> Option<BoxedTileProvider>,
    {
        let Some(render_state) = render_state else {
            self.map_gpu = false;
            return;
        };
        if self.map_gpu || self.map_gpu_failed.is_some() {
            return;
        }
        if crate::map_gpu::is_installed(render_state) {
            self.map_gpu = true;
            // Structural, not an error-string classification: both shells
            // settle "the GPU map is not attached" when `replace_provider`
            // finds no map, and the desire is unchanged afterwards — so
            // without this the memo would suppress the SAME plan for ever and
            // leave a blank map until the project changed.
            self.clear_refused_installs();
            return;
        }
        let provider = make_provider(&self.basemap)
            .unwrap_or_else(|| Box::new(crate::map_gpu::DebugCheckerboard::default()));
        match crate::map_gpu::install(render_state, self.map_view(), provider) {
            Ok(installed) => {
                self.map_gpu = true;
                self.clear_refused_installs();
                if installed {
                    // The fresh install IS the bare-basemap plan: seed the
                    // reconciliation mirror so the first frame's
                    // `pending_raster_work` does not rebuild the provider
                    // that was just built (see `app/providers.rs`).
                    self.raster_installed = Some(providers::RasterWork {
                        basemap: self.basemap.clone(),
                        cog: None,
                        archive: None,
                    });
                }
            }
            Err(error) => {
                tracing::error!(% error, "oxigis-ui: could not attach the GPU tile map");
                self.map_gpu = false;
                // Latched WITH its reason, so the failure gets the banner and
                // the Retry button every other refused provider gets instead of
                // living only in the log.
                self.map_gpu_failed = Some(error.to_string());
            }
        }
    }
    /// The basemap the map panel draws (URL template + attribution).
    #[must_use]
    pub fn basemap(&self) -> &BasemapConfig {
        &self.basemap
    }
    /// Points the map at a different basemap — as an EDIT.
    ///
    /// Since editing v1.5 this is exactly what picking a preset does, and it
    /// is documented as such because an unrecorded writer of the service is
    /// the defect the recorded op exists to close: the call is **validated**
    /// (an unusable template is refused, with a status line and no change),
    /// **status-setting**, **promotion-demoting** (a layer drawing as the
    /// basemap steps aside, in the same entry) and **undoable** (one Ctrl+Z
    /// puts the previous service back). It also stamps
    /// [`oxigis_core::Project::basemap`], which this method never did before
    /// and which a save has always needed.
    ///
    /// Only the attribution takes effect immediately; the tiles themselves
    /// come from the provider the shell reconciles through
    /// [`Self::pending_raster_work`] / [`Self::settle_raster_work`] (see
    /// `app/providers.rs`).
    pub fn set_basemap(&mut self, basemap: BasemapConfig) {
        self.apply_basemap(basemap);
    }
    /// The XYZ basemap URL template currently typed into the layer panel.
    #[must_use]
    pub fn xyz_url_input(&self) -> &str {
        &self.xyz_url_input
    }
    /// The COG URL currently typed into the layer panel.
    #[must_use]
    pub fn cog_url_input(&self) -> &str {
        &self.cog_url_input
    }
    /// The vector-tile URL template currently typed into the layer panel.
    #[must_use]
    pub fn vector_url_input(&self) -> &str {
        &self.vector_url_input
    }
    /// Whether the GPU tile map is attached (see [`Self::attach_gpu_map`]).
    #[must_use]
    pub fn map_gpu(&self) -> bool {
        self.map_gpu
    }
    /// The short status line shown in the menu bar, if any.
    #[must_use]
    pub fn status(&self) -> Option<&str> {
        self.status.as_deref()
    }
    /// Replaces the status line — how a shell reports something only it can
    /// know, such as a file it could not read.
    pub fn set_status(&mut self, message: impl Into<String>) {
        self.status = Some(message.into());
    }
    /// Takes every pending local-vector operation, in application order.
    ///
    /// A shell calls this **unconditionally** once per frame and applies the ops
    /// against `map_gpu`; draining only when a `RenderState` happens to exist
    /// would let the queue grow without bound on a host that never has one.
    /// Ops for a map that is not attached yet are simply discarded, matching the
    /// COG and vector-tile handoffs.
    ///
    /// ```text
    /// for op in self.app.take_pending_local_ops() {
    ///     let Some(render_state) = frame.wgpu_render_state() else { continue };
    ///     match op {
    ///         LocalLayerOp::Add(id, layer) => {
    ///             oxigis_ui::map_gpu::add_local_vector_layer(render_state, id, *layer);
    ///         }
    ///         // …one arm per variant; see the shells.
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn take_pending_local_ops(&mut self) -> Vec<LocalLayerOp> {
        self.local.take_ops()
    }
    /// Takes every dropped or project-referenced file waiting to be read.
    ///
    /// `oxigis-ui` never touches the filesystem (it compiles to `wasm32`), so a
    /// native shell drains this, reads each file, and feeds the bytes back —
    /// through [`Self::hydrate_geojson_layer_from_bytes`] when
    /// [`crate::local_input::PendingPath::layer`] is set (a project-load
    /// reference, which must keep its id and style), or
    /// [`Self::add_geojson_layer_from_bytes`] when it is not (a fresh drop). On
    /// the web this is always empty: a browser drop arrives with its bytes
    /// already attached.
    pub fn take_pending_dropped_paths(&mut self) -> Vec<crate::local_input::PendingPath> {
        self.local.take_paths()
    }
    /// Moves the camera so `square` just fits the map panel, with
    /// [`ZOOM_TO_LAYER_MARGIN`] free on each side.
    ///
    /// Both halves are total: [`MercatorSquare::center_lon_lat`] clamps into the
    /// Mercator domain and [`MercatorSquare::fit_zoom`] clamps into
    /// `0..=MAX_ZOOM`, so even a single-point dataset (whose bbox is padded to
    /// [`crate::local_vector::MIN_BBOX_SPAN`]) yields a usable camera.
    fn zoom_to_square(&mut self, square: MercatorSquare) {
        let view = self.map_panel.view();
        let zoom = square.fit_zoom(view.size_px(), ZOOM_TO_LAYER_MARGIN);
        self.map_panel
            .set_view(view.with_center(square.center_lon_lat()).with_zoom(zoom));
    }
    /// Whether `id` names a local GeoJSON layer, i.e. one with a GPU mirror in
    /// the local renderer that must be kept in step with the project.
    fn is_local_layer(&self, id: LayerId) -> bool {
        self.project
            .layers
            .get(id)
            .is_some_and(local_input::is_local_layer)
    }
    /// What the attribute table should show this frame: the selected layer's
    /// id, display name, and features.
    ///
    /// [`None`] means "draw the placeholder" — nothing selected, a selection
    /// that is not a local vector layer (basemap, COG, vector tiles), or a
    /// local layer whose features have not been read yet (a project-load path
    /// reference still queued for the shell).
    ///
    /// The features come from [`LocalInputState::feature_set`], never from the
    /// GPU-side copy: reading that one means holding the render lock, which a
    /// panel must not do while drawing.
    #[must_use]
    pub fn selected_table_source(&self) -> Option<(LayerId, String, Arc<FeatureCollection>)> {
        let id = self.selection?;
        let layer = self.project.layers.get(id)?;
        let features = self.local.feature_set(id)?;
        Some((id, layer.name.clone(), Arc::clone(features)))
    }
    /// Local vector layers eligible as a [`oxigis_core::ParamKind::LayerRef`]
    /// value: loaded (features already in `self.local`'s store — a
    /// project-load path reference still queued for a shell is excluded,
    /// exactly as [`Self::selected_table_source`] excludes it), in the same
    /// top-of-stack-first order the layer panel displays.
    #[must_use]
    pub fn local_vector_layer_options(&self) -> Vec<(LayerId, &str)> {
        dispatch::loaded_local_layer_options(&self.project, &self.local)
    }
    /// Paints the drag-and-drop hint over the map while files hover the window.
    fn paint_drop_hint(&self, ui: &egui::Ui, rect: egui::Rect) {
        if ui.input(|input| input.raw.hovered_files.is_empty()) {
            return;
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(0x80));
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Drop GeoJSON, a Shapefile, a GeoPackage or a tile archive to add a layer",
            egui::FontId::proportional(20.0),
            egui::Color32::from_rgb(0xF0, 0xF4, 0xFA),
        );
    }
    /// Draws the full application UI (menu bar, layer tree, style editor,
    /// attribute table, map view) into the root `ui` for this frame.
    ///
    /// `ui` must be the bare root [`egui::Ui`] handed to
    /// [`eframe::App::ui`](https://docs.rs/eframe) (egui 0.35 removed the
    /// ctx-level `Panel::show`; panels now nest inside a `Ui`). Floating
    /// windows (About, Open/Save) still attach at the [`Context`] level.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        // First, so every guard below (File ▸ New/Open) answers from the state
        // the previous frame's gestures actually left behind.
        self.observe_recorded_edits();
        self.handle_dropped_files(&ctx);
        self.about_window(&ctx);
        self.io_dialog_window(&ctx);
        self.processing_window(&ctx);
        self.print_dialog_window(&ctx);
        self.go_to_window(&ctx);
        // Before anything reads the edit selection: adopts the layer selection
        // as the edit target and clamps the feature selection against the live
        // collection, so no panel can draw against a stale index (invariant I5).
        // The clockless close of the opacity/style coalescing window: a
        // frame with no pointer button down is the end of any slider drag.
        let pointer_down = ctx.input(|input| input.pointer.any_down());
        if !pointer_down {
            self.undo.close_coalescing_for_layer_fields();
        }
        self.sync_edit_state();
        // After the clamp, so the attribute form binds to a feature that exists;
        // before the shortcuts, so a text field focused inside it takes this
        // frame's keys rather than the previous frame's.
        self.edit_window(&ctx);
        // Before `edit_shortcuts`, and only while the tape is out: the edit
        // ladder consumes `Escape` for a retained selection even with its own
        // mode `Off`, and a measurement in progress is the more recent
        // gesture. It peeks before it consumes, so a frame with nothing to
        // cancel leaves the key untouched.
        self.measure_escape(&ctx);
        // Before the menu bar, so a shortcut a menu item also offers is
        // consumed exactly once and by exactly one of them.
        self.edit_shortcuts(&ctx);
        // After the edit pair, which owns Ctrl+Z/Y and the clipboard: these
        // claim no key the edit family wants, and going second means a frame
        // the edit shortcuts returned early from still reaches File.
        self.file_shortcuts(&ctx);
        // Ctrl/Cmd+G — claimed by nothing above it, and a bare `G` (the
        // polygon tool) is a different shortcut entirely.
        self.map_tool_shortcuts(&ctx);
        self.menu_bar(ui);
        self.edit_toolbar(ui);
        // `max_size` on every side panel is load-bearing, not cosmetic: egui
        // 0.35 panels persist their *content* size each frame, so a single
        // frame of overflowing content ratchets the panel wider forever —
        // without a cap, one bad row swallows the window and the map with it
        // (exactly the bug that shipped: see layer_panel's URL rows). The cap
        // also keeps the layout an honest fixed-pane GIS: panels stay panels,
        // the map always keeps the lion's share.
        // Derived before the closure: the panel body borrows
        // `&mut self.cog_url_input` and friends, so no `&self` method can be
        // called from inside it.
        let refusal = self.provider_refusal();
        let basemap_layer = self
            .project
            .basemap_layer
            .map(|layer| layer_panel::PromotedBasemap {
                layer,
                drawn: self.draws_as_basemap(layer),
            });
        // Derived ONCE for the whole panel, for the same borrow reason and the
        // same cost reason as the two above: every row consults it to learn
        // whether it is listed but buried under the composite cap, and it
        // carries the one sentence that explains why.
        let tile_stack = self.desired_tile_stack();
        Panel::left("oxigis_layer_panel")
            .resizable(true)
            .default_size(230.0)
            .max_size(400.0)
            .show(ui, |ui| {
                // The EXTRAS entry point: without it the per-row settings
                // section (rename, scale range) is deliberately not drawn at
                // all — `PanelExtras::edits` is `None` on the plain `ui` path —
                // and the buried-row badges have nothing to consult.
                let mut edits = Vec::new();
                let actions = layer_panel::ui_with_extras(
                    ui,
                    &self.project.layers,
                    self.selection,
                    layer_panel::MapStatus {
                        basemap: &self.basemap,
                        basemap_layer,
                        refusal,
                    },
                    layer_panel::PanelFields {
                        cog_url: &mut self.cog_url_input,
                        archive_url: &mut self.archive_url_input,
                        vector_url: &mut self.vector_url_input,
                        xyz_url: &mut self.xyz_url_input,
                    },
                    layer_panel::PanelExtras {
                        tiles: Some(&tile_stack),
                        edits: Some(&mut edits),
                    },
                );
                for action in actions {
                    self.apply_layer_action(action);
                }
                // After the actions, not before: a row that was removed this
                // frame has no edit to apply, and `apply_layer_edit` reads the
                // *before* side off the project itself.
                for edit in edits {
                    self.apply_layer_edit(edit);
                }
            });
        let style_before = self
            .selection
            .and_then(|id| self.project.styles.get(&id).cloned());
        // The style editor is drawn only for the layers whose drawing it
        // actually decides. A tiled layer (MVT / tile archive / COG) paints
        // from its OWN source rules — `desired_vector` reads
        // `VectorSource::MvtTiles { paints }`, never `project.styles` — so an
        // editor bound to `project.styles` for one is a full set of controls
        // that change nothing on screen, silently rewrite the file that will be
        // saved, and (since `sync_local_style` returns early for a non-local
        // layer) cannot even be undone. Refusing is what makes those three
        // false statements unrepresentable.
        let styleable = self.selection.is_some_and(|id| self.is_local_layer(id));
        let repartitioned = Panel::right("oxigis_style_panel")
            .resizable(true)
            .default_size(260.0)
            .max_size(420.0)
            .show(ui, |ui| self.style_panel_body(ui, styleable))
            .inner;
        // A structural renderer change — Classify, a switched kind, a class
        // added or dropped — is ONE discrete gesture and must be one undo step,
        // so the coalescing window is closed on both sides of it: before, so it
        // cannot fold into a colour drag that was still open; after, so the
        // next drag cannot fold into it. A frame with no repartition (all but a
        // handful) touches neither.
        if repartitioned {
            self.undo.close_coalescing();
        }
        self.sync_local_style_gated(style_before, pointer_down);
        if repartitioned {
            self.undo.close_coalescing();
        }
        if self.show_table {
            let table_source = self.selected_table_source();
            Panel::bottom("oxigis_attribute_table")
                .resizable(true)
                .default_size(220.0)
                .max_size(480.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Attribute table");
                        if ui.small_button("Hide").clicked() {
                            self.show_table = false;
                        }
                    });
                    ui.separator();
                    match table_source {
                        Some((id, name, features)) => {
                            let row_clicked = self
                                .table_panel
                                .show(ui, id, &name, &features)
                                .iter()
                                .any(|event| matches!(event, TableEvent::RowSelected(_)));
                            // After `show`, never before: `bind()` clears the
                            // panel's selection on every new `Arc`.
                            self.sync_table_selection(row_clicked);
                            // The toolbar's Export button parked the CSV it
                            // captured this frame; take it while the table is
                            // still bound to the layer it came from.
                            self.drain_table_export();
                        }
                        None => {
                            let pending = self.selection.is_some_and(|id| self.is_local_layer(id));
                            self.table_panel.show_placeholder(ui, pending);
                        }
                    }
                });
        }
        egui::CentralPanel::default().show(ui, |ui| {
            // Destructuring is required, not stylistic: the gate closure needs
            // `&mut EditState`, `&LocalInputState`, `&Project` and the layer
            // selection while `map_panel` is borrowed mutably.
            let (rect, response) = {
                let Self {
                    map_panel,
                    edit,
                    local,
                    project,
                    selection,
                    ..
                } = self;
                let selection = *selection;
                let view = map_panel.view();
                map_panel.allocate_gated(ui, |rect, response, ppp| {
                    edit.gate_pan(rect, response, ppp, project, local, selection, view)
                })
            };
            self.map_rect = Some(rect);
            if self.map_gpu {
                self.map_panel.paint_gpu(ui, rect);
            } else {
                // The note has to say WHICH of the two situations this is: a
                // host with no `wgpu` at all, or an install that failed on a
                // render state that did exist. The default text asserts the
                // former and would misdirect the diagnosis of the latter.
                // Destructured so the reason is borrowed, not cloned per frame.
                let Self {
                    map_panel,
                    map_gpu_failed,
                    ..
                } = self;
                map_panel.set_fallback_reason(map_gpu_failed.as_deref());
                map_panel.paint_fallback(ui, rect);
            }
            let commands = self.edit_interact(ui, rect, &response);
            self.edit_overlay(ui, rect);
            // After `edit_interact`, which returns early with the edit mode
            // `Off` — the only state the tape can be armed in, so the two
            // never read the same click. See `app::map_tools`.
            self.measure_tool(ui, rect, &response);
            self.paint_attribution(ui, rect);
            self.paint_drop_hint(ui, rect);
            for transaction in commands {
                if self.commit_edit(transaction) {
                    self.mark_project_dirty();
                }
            }
        });
        // LAST, after every gesture this frame has applied: a `SetVisibility`
        // queued here must be the one the queue keeps (`LocalInputState::queue`
        // coalesces last-write-wins), or a checkbox click would push the raw
        // flag over the range-resolved answer.
        self.sync_local_zoom_visibility();
        if self.local.pending_op_count() > 0 || self.local.pending_path_count() > 0 {
            ctx.request_repaint();
        }
    }
    /// Pushes each local layer's **range-resolved** visibility into its GPU
    /// mirror when the answer has changed.
    ///
    /// The local draw path culls on `LocalVectorLayer::visible()` — a mirrored
    /// bool that knows nothing about a scale range — so without this a layer
    /// the tiled derivations have already stopped listing keeps drawing its own
    /// features. Reusing the mirror that already exists is what makes that
    /// impossible to get half-right: there is one predicate
    /// ([`oxigis_core::Layer::visible_at`], the checkbox AND the range
    /// together) and one place it reaches the renderer.
    ///
    /// Camera-dependent, and therefore run every frame — but it *queues*
    /// nothing unless the answer moved, so an idle map at a steady zoom costs
    /// one comparison per local layer and no GPU work at all.
    ///
    /// The remembered value is the PAIR `(checkbox, resolved)`, not the
    /// resolved answer alone. Re-ticking the checkbox of a layer that is out of
    /// range leaves `resolved` at `false` while the applier has just queued the
    /// raw `true`; comparing the pair is what notices that and re-pushes the
    /// `false` over it.
    fn sync_local_zoom_visibility(&mut self) {
        let zoom = self.map_panel.view().zoom();
        // Dropped with their layers, so the map is bounded by the project.
        self.local_visibility_pushed
            .retain(|id, _| self.project.layers.get(*id).is_some());
        for layer in self.project.layers.layers() {
            if !local_input::is_local_layer(layer) {
                continue;
            }
            let resolved = layer.visible_at(zoom);
            let state = (layer.visible, resolved);
            match self.local_visibility_pushed.get(&layer.id) {
                // Nothing moved.
                Some(pushed) if *pushed == state => continue,
                // FIRST sight. `LocalLayerOp::Add` has just put the layer on
                // the GPU fully visible, so an op is owed only when the range
                // (or the checkbox) says otherwise — without this arm every
                // single add would cost a redundant second op.
                None if resolved => {
                    self.local_visibility_pushed.insert(layer.id, state);
                    continue;
                }
                _ => {}
            }
            self.local_visibility_pushed.insert(layer.id, state);
            self.local
                .queue(LocalLayerOp::SetVisibility(layer.id, resolved));
        }
    }
    /// Draws the basemap credit line in the map's bottom-right corner.
    ///
    /// Required rather than decorative: the OpenStreetMap
    /// [tile usage policy](https://operations.osmfoundation.org/policies/tiles/)
    /// makes displaying the attribution a condition of using its tiles, so this
    /// is painted after the map (GPU or fallback) and clipped to the map rect.
    /// An empty [`BasemapConfig::attribution`] draws nothing, which is correct
    /// only for a source that needs no credit.
    fn paint_attribution(&self, ui: &egui::Ui, rect: egui::Rect) {
        // The SAME builder the print snapshot reads, so the painted credit
        // and the exported page cannot drift — and a removed layer's credit
        // dies with it, because the line derives from the project.
        let credit = self.credit_line();
        if credit.is_empty() {
            return;
        }
        let painter = ui.painter_at(rect);
        let galley = painter.layout_no_wrap(
            credit,
            egui::FontId::proportional(ATTRIBUTION_FONT_PT),
            egui::Color32::from_rgb(0xE0, 0xE4, 0xEA),
        );
        let plate_size = galley.size() + egui::vec2(ATTRIBUTION_PAD_PT * 2.0, ATTRIBUTION_PAD_PT);
        let plate = egui::Rect::from_min_size(
            rect.max - plate_size - egui::vec2(ATTRIBUTION_MARGIN_PT, ATTRIBUTION_MARGIN_PT),
            plate_size,
        );
        painter.rect_filled(plate, 3.0, egui::Color32::from_black_alpha(0xB0));
        painter.galley(
            plate.min + egui::vec2(ATTRIBUTION_PAD_PT, ATTRIBUTION_PAD_PT * 0.5),
            galley,
            egui::Color32::PLACEHOLDER,
        );
    }
    /// Draws the top menu bar (File / View / Help).
    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        Panel::top("oxigis_menu_bar").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    // Rendered through `format_shortcut`, so the labels read
                    // `⌘S` on macOS and `Ctrl+S` everywhere else — the same
                    // `Modifiers::COMMAND` the shortcut handler consumes, so
                    // the menu cannot promise a key the app does not take.
                    let shortcut = |ui: &egui::Ui, modifiers, key| {
                        ui.ctx()
                            .format_shortcut(&egui::KeyboardShortcut::new(modifiers, key))
                    };
                    // Both of these DESTROY the open project and reset the undo
                    // log in the same call, so there is no way back afterwards
                    // — they are the only gestures in the app that are not
                    // Ctrl+Z-able, which is exactly why they ask first.
                    let new_keys = shortcut(ui, egui::Modifiers::COMMAND, egui::Key::N);
                    if ui
                        .add(egui::Button::new("New").shortcut_text(new_keys))
                        .clicked()
                    {
                        self.request_discarding_action(PendingAction::NewProject);
                        ui.close();
                    }
                    let open_keys = shortcut(ui, egui::Modifiers::COMMAND, egui::Key::O);
                    if ui
                        .add(egui::Button::new("Open\u{2026}").shortcut_text(open_keys))
                        .clicked()
                    {
                        self.request_discarding_action(PendingAction::OpenProject);
                        ui.close();
                    }
                    self.recent_projects_menu(ui);
                    let save_keys = shortcut(ui, egui::Modifiers::COMMAND, egui::Key::S);
                    if self.native_project_io {
                        // Save writes back to the file this project came from
                        // without a dialog; Save As always asks. A project with
                        // no file yet makes them the same gesture, which is why
                        // Save is never disabled.
                        let save_hint = match self.project_path.as_ref() {
                            Some(path) => format!("Writes back to {}", path.display()),
                            None => "Asks where to write the project".to_string(),
                        };
                        if ui
                            .add(egui::Button::new("Save").shortcut_text(save_keys))
                            .on_hover_text(save_hint)
                            .clicked()
                        {
                            self.request_project_save(false);
                            ui.close();
                        }
                        let save_as_keys = shortcut(
                            ui,
                            egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
                            egui::Key::S,
                        );
                        if ui
                            .add(egui::Button::new("Save As\u{2026}").shortcut_text(save_as_keys))
                            .on_hover_text("Writes the project to a file you choose")
                            .clicked()
                        {
                            self.request_project_save(true);
                            ui.close();
                        }
                    } else if ui
                        .add(egui::Button::new("Save\u{2026}").shortcut_text(save_keys))
                        .on_hover_text(
                            "Shows the project's JSON to copy out — this build has no filesystem",
                        )
                        .clicked()
                    {
                        self.request_project_save(false);
                        ui.close();
                    }
                    ui.separator();
                    // Everything this application can WRITE, under one
                    // heading — the PDF included, which lived at the top level
                    // only while it was the sole answer.
                    self.export_menu(ui);
                });
                ui.menu_button("Edit", |ui| {
                    // Labels are built first: `add_enabled` borrows `ui`
                    // mutably while the enabled flag and the label both read
                    // `self.undo`.
                    let undo_label = self.undo.peek_undo_entry().map_or_else(
                        || "Undo".to_string(),
                        |entry| format!("Undo {}", entry.label()),
                    );
                    let redo_label = self.undo.peek_redo_entry().map_or_else(
                        || "Redo".to_string(),
                        |entry| format!("Redo {}", entry.label()),
                    );
                    let can_undo = self.undo.can_undo();
                    let can_redo = self.undo.can_redo();
                    if ui
                        .add_enabled(can_undo, egui::Button::new(undo_label))
                        .clicked()
                    {
                        self.undo_once();
                        ui.close();
                    }
                    if ui
                        .add_enabled(can_redo, egui::Button::new(redo_label))
                        .clicked()
                    {
                        self.redo_once();
                        ui.close();
                    }
                    ui.separator();
                    // Menu twins of the clipboard events: browser copy events
                    // do not fire in every embed context, and a menu item is
                    // the honest fallback (it goes through the same paths).
                    let can_copy = self.edit.mode() != crate::edit::EditMode::Off
                        && self.edit.selection().is_some();
                    if ui
                        .add_enabled(can_copy, egui::Button::new("Copy features"))
                        .on_hover_text("Copies the selected features as GeoJSON (Ctrl+C)")
                        .clicked()
                    {
                        let ctx = ui.ctx().clone();
                        self.copy_selection(&ctx);
                        ui.close();
                    }
                    let can_cut = can_copy;
                    if ui
                        .add_enabled(can_cut, egui::Button::new("Cut features"))
                        .on_hover_text("Copies the selected features, then deletes them (Ctrl+X)")
                        .clicked()
                    {
                        let ctx = ui.ctx().clone();
                        if self.copy_selection(&ctx) {
                            self.delete_selected_feature();
                        }
                        ui.close();
                    }
                    ui.separator();
                    let window_open = self.edit.show_window();
                    if ui
                        .selectable_label(window_open, "Edit tools…")
                        .on_hover_text("Attributes, validation and snap settings")
                        .clicked()
                    {
                        self.apply_edit_action(EditAction::ToggleWindow);
                        ui.close();
                    }
                    ui.menu_button("New edit layer", |ui| {
                        for (label, kind) in [
                            ("Point", StyleKind::Circle),
                            ("Line", StyleKind::Line),
                            ("Polygon", StyleKind::Fill),
                        ] {
                            if ui.button(label).clicked() {
                                self.apply_edit_action(EditAction::NewLayer(kind));
                                ui.close();
                            }
                        }
                    });
                    ui.separator();
                    let has_feature = self.edit.selection().is_some();
                    if ui
                        .add_enabled(has_feature, egui::Button::new("Delete feature"))
                        .clicked()
                    {
                        self.apply_edit_action(EditAction::DeleteFeature);
                        ui.close();
                    }
                    // Enabled only for a layer whose features are actually in
                    // memory: a project-load path reference still queued for a
                    // shell has nothing to check yet.
                    let can_validate = self
                        .selection
                        .is_some_and(|id| self.local.feature_set(id).is_some());
                    if ui
                        .add_enabled(can_validate, egui::Button::new("Validate layer"))
                        .on_hover_text("Check every feature for topology problems")
                        .on_disabled_hover_text(
                            "Select a loaded local vector layer to validate it.",
                        )
                        .clicked()
                    {
                        self.apply_edit_action(EditAction::ValidateLayer);
                        ui.close();
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_table, "Attribute table");
                    // Enabled only for a layer whose extent is actually known:
                    // a provider layer has none, and a project-load path
                    // reference has not been read yet.
                    let can_zoom = self
                        .selection
                        .is_some_and(|id| self.local.feature_set(id).is_some());
                    if ui
                        .add_enabled(can_zoom, egui::Button::new("Zoom to layer"))
                        .on_hover_text("Fits the selected layer's extent in the map")
                        .on_disabled_hover_text("Select a loaded local vector layer to zoom to it.")
                        .clicked()
                    {
                        self.zoom_to_selected_layer();
                        ui.close();
                    }
                    ui.separator();
                    // The interactive map tools: the tape, the scale bar and
                    // the coordinate box (see `app::map_tools`).
                    self.map_tools_menu(ui);
                });
                ui.menu_button("Processing", |ui| {
                    if ui.button("Toolbox…").clicked() {
                        self.show_processing = true;
                        ui.close();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("About").clicked() {
                        self.show_about = true;
                        ui.close();
                    }
                });
                ui.separator();
                // The one place the app can show the dirty marker itself: the
                // window title belongs to the shell (see
                // [`Self::has_unsaved_changes`], which is what a shell reads
                // to put a `*` there), and this crate owns no window. The clean
                // case borrows the name rather than building a string 60 times
                // a second for it.
                if self.has_unsaved_changes() {
                    ui.weak(self.title_line());
                } else {
                    ui.weak(self.project.name.as_str());
                }
                if let Some(status) = &self.status {
                    ui.separator();
                    ui.weak(status);
                }
            });
        });
    }
    /// Draws File ▸ Open Recent ▸, for a shell that can read files.
    ///
    /// Drawn only when the shell declared real project I/O: a browser cannot
    /// re-open a path, so a list of them there would be a menu of dead
    /// entries. Each item carries the whole path in its tooltip, since several
    /// projects routinely share a file name.
    fn recent_projects_menu(&mut self, ui: &mut egui::Ui) {
        if !self.native_project_io {
            return;
        }
        // Materialized before the closure so it captures no part of `self`:
        // the submenu body only runs while the menu is open, and the list is
        // capped at [`MAX_RECENT_PROJECTS`], so this is a handful of short
        // strings while a menu is on screen and nothing at all otherwise.
        let entries: Vec<(String, String)> = self
            .recent_projects
            .iter()
            .map(|path| {
                let label = path.file_name().map_or_else(
                    || path.display().to_string(),
                    |n| n.to_string_lossy().into(),
                );
                (label, path.display().to_string())
            })
            .collect();
        let mut chosen = None;
        let mut clear = false;
        ui.menu_button("Open Recent", |ui| {
            if entries.is_empty() {
                ui.weak("Nothing yet");
                return;
            }
            for (index, (label, full)) in entries.iter().enumerate() {
                // Several projects routinely share a file name, so the whole
                // path is one hover away.
                if ui.button(label).on_hover_text(full).clicked() {
                    chosen = Some(index);
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Clear list").clicked() {
                clear = true;
                ui.close();
            }
        });
        let chosen = chosen.and_then(|index| self.recent_projects.get(index).cloned());
        if let Some(path) = chosen {
            // The same guarded route File ▸ Open takes: this replaces the
            // project and resets the undo log too.
            self.request_discarding_action(PendingAction::OpenRecent(path));
            ui.close();
        }
        if clear {
            self.recent_projects.clear();
            ui.close();
        }
    }

    /// File-menu keyboard shortcuts: Ctrl/Cmd+N, +O, +S and Shift+S.
    ///
    /// Guarded exactly as [`Self::edit_shortcuts`] is — a focused text field
    /// keeps its keys — and consumed with `consume_shortcut`, so a key a menu
    /// item also offers is taken exactly once. Save-As is tested BEFORE Save,
    /// because `Ctrl+Shift+S` also matches a bare `Ctrl+S` pattern's key.
    fn file_shortcuts(&mut self, ctx: &Context) {
        use egui::{Key, KeyboardShortcut, Modifiers};
        if ctx.memory(|memory| memory.focused()).is_some() {
            return;
        }
        let save_as = ctx.input_mut(|input| {
            input.consume_shortcut(&KeyboardShortcut::new(
                Modifiers::COMMAND.plus(Modifiers::SHIFT),
                Key::S,
            ))
        });
        if save_as {
            self.request_project_save(true);
            return;
        }
        if ctx.input_mut(|input| {
            input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::S))
        }) {
            self.request_project_save(false);
            return;
        }
        if ctx.input_mut(|input| {
            input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::O))
        }) {
            self.request_discarding_action(PendingAction::OpenProject);
            return;
        }
        if ctx.input_mut(|input| {
            input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::N))
        }) {
            self.request_discarding_action(PendingAction::NewProject);
        }
    }

    /// The project's name with a trailing `*` while it has unsaved changes —
    /// the string a shell puts in its window title, and what the menu bar
    /// shows in a crate that owns no window.
    #[must_use]
    pub fn title_line(&self) -> String {
        if self.has_unsaved_changes() {
            format!("{} *", self.project.name)
        } else {
            self.project.name.clone()
        }
    }

    /// The whole window title: `"<project> * — OxiGIS"`.
    ///
    /// The document-first order every desktop application uses, so a taskbar
    /// that truncates keeps the half that tells one window from another. The
    /// `*` is [`Self::has_unsaved_changes`], which is deliberately
    /// conservative — see its docs.
    #[must_use]
    pub fn window_title(&self) -> String {
        format!("{} \u{2014} OxiGIS", self.title_line())
    }

    /// What the style panel draws instead of the editor for a layer whose
    /// drawing it does not decide (see the call site in [`Self::ui`]).
    fn tiled_style_notice(ui: &mut egui::Ui) {
        ui.heading("Style");
        ui.separator();
        ui.weak("Tiled layers are styled by their source's paint rules.")
            .on_hover_text(
                "An MVT, tile-archive or COG layer draws from the paint rules stored on the \
                 layer itself, so there is nothing here for the style editor to change. Add \
                 the data as a local vector layer to style it.",
            );
    }

    /// Fits the selected layer's extent in the map — the missing return trip
    /// for the zoom every *add* seam already performs.
    ///
    /// Answers `false` when nothing is selected or the selection has no known
    /// extent (a provider layer, or a project-load path reference a shell has
    /// not read yet). The extent is derived from the feature store's own
    /// collection, never from the GPU-side copy: reading that means holding the
    /// render lock, which a panel must not do while drawing.
    ///
    /// A camera move, so nothing is recorded: the undo log is for project
    /// state, and the view is not it (`Project::view` is stamped at save time
    /// by [`Self::sync_project_view`]).
    pub fn zoom_to_selected_layer(&mut self) -> bool {
        let Some(id) = self.selection else {
            return false;
        };
        let Some(features) = self.local.feature_set(id) else {
            return false;
        };
        let square = crate::local_vector::collection_square(features);
        self.zoom_to_square(square);
        true
    }

    /// Draws the Help ▸ About window when [`Self::show_about`] is set.
    fn about_window(&mut self, ctx: &Context) {
        if !self.show_about {
            return;
        }
        let mut open = self.show_about;
        egui::Window::new("About OxiGIS")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(format!("oxigis-ui {}", crate::VERSION));
                ui.label(format!("oxigis-core {}", oxigis_core::VERSION));
                ui.label(format!("oxigis-render {}", oxigis_render::VERSION));
                ui.separator();
                ui.weak("Pure Rust full-stack GIS.");
            });
        self.show_about = open;
    }
}

impl Default for OxigisApp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod edit_tests;
#[cfg(test)]
mod edit_tests_marks;
#[cfg(test)]
mod edit_tests_review;
#[cfg(test)]
mod edit_tests_topology;
#[cfg(test)]
mod style_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_archive;
#[cfg(test)]
mod tests_basemap;
#[cfg(test)]
mod tests_basemap_service;
#[cfg(test)]
mod tests_providers;
#[cfg(test)]
mod tests_session;
