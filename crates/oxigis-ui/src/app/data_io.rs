// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Data-ingestion half of [`OxigisApp`]: project load/new, the
//! add/hydrate seams for every local vector format (GeoJSON, Shapefile,
//! GeoPackage, GeoParquet), dropped-file routing, and the File ▸
//! Open/Save/Paste modals. Split from `app/mod.rs` under the 2000-line
//! rule; the frame loop and accessors stay there, the action dispatchers
//! live in `app::dispatch`.

use super::OxigisApp;
use crate::local_input::{self, INLINE_GEOJSON_WARN_BYTES};
use crate::tile_provider::BasemapConfig;
use egui::Context;
use oxigis_core::{LayerId, Project};

/// A gesture that destroys the open project, held while the user is asked
/// whether that is what they meant.
///
/// One variant per destructive entry point rather than a boxed closure: the
/// confirmation has to survive across frames inside [`IoDialog`], and a plain
/// enum keeps it `Clone`, comparable and testable. It is also what "Save, then
/// do this" carries — [`OxigisApp::confirm_project_saved`] runs whatever was
/// parked here once the shell reports the bytes are on disk.
///
/// Deliberately **not** `Copy`: [`Self::OpenRecent`] owns a path, and carrying
/// the path rather than an index into the recent list is what keeps a save
/// that reorders that list between arming and running from opening the wrong
/// project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PendingAction {
    /// File ▸ New: replace the project with an empty one.
    NewProject,
    /// File ▸ Open…: replace the project with one the shell reads (or, with no
    /// shell filesystem, one the user pastes).
    OpenProject,
    /// File ▸ Open Recent ▸: replace the project with this specific file.
    OpenRecent(std::path::PathBuf),
    /// The window's close button: let the shell finish quitting.
    CloseWindow,
}

impl PendingAction {
    /// What the confirmation says is about to happen.
    fn headline(&self) -> &'static str {
        match self {
            Self::NewProject => "Start a new project?",
            Self::OpenProject | Self::OpenRecent(_) => "Open another project?",
            Self::CloseWindow => "Quit OxiGIS?",
        }
    }

    /// The sentence under the headline, which has to name the right
    /// consequence: quitting loses the changes outright, while opening
    /// something else also throws the undo history away.
    fn consequence(&self) -> &'static str {
        match self {
            Self::NewProject | Self::OpenProject | Self::OpenRecent(_) => {
                "This project has changes that have not been saved. Starting or opening another \
                 one discards them, and the undo history goes with them."
            }
            Self::CloseWindow => {
                "This project has changes that have not been saved. Quitting discards them, and \
                 the undo history goes with them."
            }
        }
    }
}

/// A project write the shell has been asked to perform — the take-once seam
/// [`OxigisApp::take_pending_project_save`] hands over.
///
/// `oxigis-ui` owns no filesystem (it compiles to `wasm32`), so every platform
/// capability crosses this way: the crate serializes, the shell writes. The
/// same pattern as [`OxigisApp::take_pending_print`] and
/// [`OxigisApp::take_pending_archive_pick`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectSaveRequest {
    /// Where the bytes go, or [`None`] when the shell must ask first — File ▸
    /// Save As, or a project that has never been written.
    pub path: Option<std::path::PathBuf>,
    /// The serialized project, camera and basemap already stamped in.
    pub content: String,
}

/// A project read the shell has been asked to perform — the take-once seam
/// [`OxigisApp::take_pending_project_open`] hands over.
///
/// The unsaved-changes question has already been asked and answered by the
/// time this appears: the shell may open the file without further ceremony.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectOpenRequest {
    /// The file to read, or [`None`] when the shell must ask which one.
    pub path: Option<std::path::PathBuf>,
}

/// A pending File-menu I/O interaction that needs a modal window because
/// this crate has no native file-dialog dependency (by design — see the
/// task brief). `Open` lets the user paste project JSON in; `Save` shows the
/// current project's serialized JSON for the user to copy out.
pub(super) enum IoDialog {
    /// Asking whether unsaved changes may be discarded, before running
    /// `pending`.
    ConfirmDiscard {
        /// What runs if the user chooses to discard.
        pending: PendingAction,
    },
    /// Pasting JSON to load as the current project.
    Open {
        /// The text area's current contents.
        buffer: String,
        /// The most recent parse error, if any, shown under the text area.
        error: Option<String>,
    },
    /// Showing the current project's serialized JSON for copying.
    Save {
        /// The serialized project JSON.
        content: String,
    },
    /// Showing a Processing result's GeoJSON for copying, instead of adding it
    /// to the project — the `Output ▸ GeoJSON to copy out` destination.
    ///
    /// The read-only twin of [`Self::Save`], and for the same reason: a tool
    /// result has no file to reference, so adding it as a layer embeds the
    /// whole document in the project. This is how a user keeps a large result
    /// without paying for it in every save, and it needs no filesystem — which
    /// this crate does not have on `wasm32` and does not reach for on native.
    ExportGeoJson {
        /// What the result would have been called as a layer, for the title.
        name: String,
        /// The pretty-printed GeoJSON document.
        content: String,
    },
    /// Pasting GeoJSON to add as a local vector layer — the keyboard twin of
    /// dropping a `.geojson` file onto the map.
    PasteGeoJson {
        /// The layer name to give the pasted dataset.
        name: String,
        /// The text area's current contents.
        buffer: String,
        /// The most recent parse error, if any, shown under the text area.
        error: Option<String>,
    },
}

impl OxigisApp {
    /// Runs `pending` — or, when the project has unsaved changes, puts the
    /// confirmation in front of it.
    ///
    /// The ONE gate File ▸ New and File ▸ Open go through. Both replace the
    /// project *and* call `undo.reset()` in the same breath, so the work they
    /// discard cannot be recovered by any means the app offers; a clean
    /// project has nothing to lose and is not interrupted.
    pub(super) fn request_discarding_action(&mut self, pending: PendingAction) {
        if self.has_unsaved_changes() {
            self.io_dialog = Some(IoDialog::ConfirmDiscard { pending });
            return;
        }
        self.run_pending_action(pending);
    }

    /// Performs a confirmed (or unneeded-confirmation) destructive action.
    pub(super) fn run_pending_action(&mut self, pending: PendingAction) {
        match pending {
            PendingAction::NewProject => {
                self.new_project();
                self.project_path = None;
            }
            // With a shell that owns a filesystem this is a request for its
            // Open dialog; without one it is the paste modal, which is the only
            // way a browser can be handed a document at all.
            PendingAction::OpenProject => {
                if self.native_project_io {
                    self.pending_project_open = Some(ProjectOpenRequest { path: None });
                } else {
                    self.io_dialog = Some(IoDialog::Open {
                        buffer: String::new(),
                        error: None,
                    });
                }
            }
            PendingAction::OpenRecent(path) => {
                if self.native_project_io {
                    self.pending_project_open = Some(ProjectOpenRequest { path: Some(path) });
                } else {
                    // Unreachable through the menu (the recent list is only
                    // drawn for a shell that can read files), and still
                    // answered rather than silently dropped.
                    self.status = Some(
                        "This build cannot read files; paste the project instead.".to_string(),
                    );
                }
            }
            PendingAction::CloseWindow => self.close_confirmed = true,
        }
    }

    /// Opens File ▸ Save…: stamps the live camera and basemap into the project
    /// and shows its JSON for copying.
    ///
    /// The browser-shaped path, and the one a shell with no filesystem keeps.
    /// The project counts as saved from here, which is what clears the
    /// unsaved-changes marker — the honest reading of a modal whose whole
    /// content is the document. A shell that reports
    /// [`OxigisApp::set_native_project_io`] never reaches this: it gets
    /// [`Self::request_project_save`] instead, which writes a real file.
    pub(super) fn open_save_dialog(&mut self) {
        self.sync_project_view();
        match self.project.to_json_string() {
            Ok(content) => {
                self.io_dialog = Some(IoDialog::Save { content });
                self.mark_saved();
            }
            Err(error) => {
                self.after_save = None;
                self.status = Some(format!("Save failed: {error:?}"));
            }
        }
    }

    /// Queues a real project write for the shell — File ▸ Save and
    /// File ▸ Save As.
    ///
    /// `save_as` (or a project that has never been written) leaves the
    /// request's path [`None`], which is what tells the shell to ask where it
    /// should go; otherwise the bytes go straight back to the file this
    /// project came from, so a plain Ctrl+S never opens a dialog.
    ///
    /// [`Self::sync_project_view`] first, exactly as the copy-JSON modal does:
    /// the camera and the active basemap live outside [`Project`] until a save
    /// stamps them in, so skipping it would write a file whose map opens
    /// somewhere else.
    ///
    /// Nothing is marked saved here — the bytes are not on disk until the
    /// shell says so through [`OxigisApp::confirm_project_saved`].
    ///
    /// Public so a shell can offer the gesture from its own platform menu bar,
    /// dock menu or accelerator table without going through the in-app File
    /// menu — the same reason [`Self::request_project_open`] and
    /// [`Self::request_new_project`] are.
    pub fn request_project_save(&mut self, save_as: bool) {
        if !self.native_project_io {
            self.open_save_dialog();
            return;
        }
        if self.pending_project_save.is_some() {
            // A second Ctrl+S in the frame before the shell drained the first
            // would silently replace it, and with it any parked action.
            return;
        }
        self.sync_project_view();
        match self.project.to_json_string() {
            Ok(content) => {
                let path = if save_as {
                    None
                } else {
                    self.project_path.clone()
                };
                self.status = Some(match path.as_ref() {
                    Some(path) => format!("Saving to {}\u{2026}", path.display()),
                    None => "Choose where to save the project\u{2026}".to_string(),
                });
                self.pending_project_save = Some(ProjectSaveRequest { path, content });
            }
            Err(error) => {
                // A project that will not serialize cannot be saved, so
                // whatever was waiting on the save must not run either.
                self.after_save = None;
                self.status = Some(format!("Save failed: {error:?}"));
            }
        }
    }

    /// Replaces the current project outright (e.g. a shell's own native
    /// "Open" file dialog finished reading a file). Clears the selection
    /// since layer ids from the old project may not exist in the new one,
    /// and moves the map camera to the loaded project's saved
    /// [`oxigis_core::View`].
    ///
    /// Also rebuilds the local-vector stack: a
    /// [`crate::local_input::LocalLayerOp::Clear`] (the
    /// *previous* project's datasets are still attached to the GPU) followed by
    /// one `Add` per inline-GeoJSON layer, restyled from
    /// [`Project::styles`]. Layers stored as a path reference are queued for a
    /// native shell to read — see [`Self::take_pending_dropped_paths`] — or, on
    /// a build with no filesystem, reported in the status line.
    ///
    /// **Every** path that replaces the project must go through here (or
    /// [`Self::new_project`]); assigning `self.project` directly would leave the
    /// old project's layers on screen.
    pub fn load_project(&mut self, project: Project) {
        self.load_project_with_notices(project, Vec::new());
    }
    /// Loads a project recovered by the GeoLibre compat reader
    /// ([`crate::geolibre_import::import`]), merging its own notices
    /// (dropped basemap/labels/unsupported layers, per §8's exact wordings)
    /// with whatever [`local_input::LocalInputState::rebuild_from_project`]
    /// reports for the layers that *did* map to something we can hold —
    /// using the exact same "N problems" status-line aggregation
    /// [`Self::load_project`] uses for a native project, so the two paths
    /// read identically at the UI seam. Called from the same two places
    /// [`Self::load_project`] is: the File ▸ Open… dialog and a
    /// `*.geolibre.json` drop — see `load_project_text` at the bottom of
    /// this file.
    pub fn load_geolibre_project(&mut self, project: Project, import_notices: Vec<String>) {
        self.load_project_with_notices(project, import_notices);
    }
    /// The shared tail of [`Self::load_project`] and
    /// [`Self::load_geolibre_project`]: replace the project, move the
    /// camera, rebuild the local-vector stack, and aggregate `notices`
    /// (already carrying any GeoLibre-import notices) with whatever that
    /// rebuild itself reports.
    fn load_project_with_notices(&mut self, project: Project, mut notices: Vec<String>) {
        self.project = project;
        self.selection = None;
        // Cleared, not kept: a paste, a `.geolibre.json` drop and a native
        // Open all land here, and only the last of those has a
        // `*.oxigis.json` home — the shell stamps it back with
        // `set_project_path` once its read succeeded. Ctrl+S on anything else
        // must ask where to go rather than overwrite the previous project.
        self.project_path = None;
        // `LayerId`s are reserved per deserialize, so the project just loaded
        // can legitimately hold the same ids the previous one did: an
        // un-cleared undo stack would splice the old project's features into
        // the new project's layers.
        self.undo.reset();
        self.edit.reset();
        self.apply_project_view();
        self.apply_loaded_basemap(&mut notices);
        self.report_loaded_basemap_layer(&mut notices);
        notices.extend(self.local.rebuild_from_project(&self.project));
        // AFTER the rebuild, which installs the loaded file's collections: the
        // loaded document IS the saved state, so the watermark has to be taken
        // over the finished project rather than the half-built one.
        self.mark_project_loaded();
        self.status = Some(match notices.len() {
            0 => "Project loaded.".to_string(),
            1 => format!("Project loaded. {}", notices[0]),
            count => format!("Project loaded, with {count} problems. {}", notices[0]),
        });
        for notice in &notices {
            tracing::warn!(notice, "oxigis-ui: project load notice");
        }
    }
    /// Restores the loaded project's saved basemap, if it recorded one.
    ///
    /// `None` — a file from a build that predates [`Project::basemap`] — leaves
    /// the active basemap alone, which is exactly what those builds did. A
    /// recorded basemap that fails [`BasemapConfig::template`] validation is
    /// reported as a load notice and the active basemap also stays, so a
    /// damaged file still opens instead of taking the map down with it.
    fn apply_loaded_basemap(&mut self, notices: &mut Vec<String>) {
        let Some(saved) = self.project.basemap.as_ref() else {
            return;
        };
        let config = BasemapConfig::from(saved);
        if config == self.basemap {
            // The same service is already on screen: queueing a swap would
            // blank the map and re-fetch every visible tile for nothing.
            return;
        }
        match config.template() {
            Ok(_) => {
                // The swap follows from the derivation next frame; the
                // reconciliation mirror survives the load, which is what
                // keeps an unchanged basemap from re-fetching every tile.
                //
                // This is the project-REPLACEMENT seam, not an edit, so it
                // writes the field directly rather than going through
                // `set_basemap_service`: `load_project_with_notices` calls
                // `self.undo.reset()` before it, so no Ctrl+Z can jump over
                // this write, and stamping `project.basemap` here is
                // redundant — the value came from that very field.
                self.basemap = config;
            }
            Err(error) => {
                notices.push(format!(
                    "The project's saved basemap ({}) is unusable: {error}. Keeping {}.",
                    saved.url_template, self.basemap.url_template
                ));
            }
        }
    }
    /// Reports a loaded [`oxigis_core::Project::basemap_layer`] pointer that
    /// cannot draw — exactly ONE notice, in the same list
    /// [`Self::apply_loaded_basemap`] pushes into.
    ///
    /// The pointer is deliberately **not** cleared. A load must not silently
    /// mutate a file the user may re-save, and scrubbing would break
    /// `remove → save → Ctrl+Z`: the undo restores the layer, and the
    /// promotion has to come back with it.
    ///
    /// A promoted layer that is merely *hidden* is not reported: it is a
    /// legal, recorded state whose affordance is the visibility checkbox.
    fn report_loaded_basemap_layer(&mut self, notices: &mut Vec<String>) {
        let Some(id) = self.project.basemap_layer else {
            return;
        };
        let Some(reason) = self.promotion_refusal(id) else {
            return;
        };
        notices.push(if self.project.layers.get(id).is_none() {
            "The project's basemap layer is not in the project; drawing the saved basemap \
             service instead."
                .to_string()
        } else {
            format!(
                "The project's basemap layer cannot draw as the basemap ({reason}); drawing \
                 the saved basemap service instead."
            )
        });
    }
    /// Starts a fresh, empty project, detaching every local dataset the old one
    /// had attached (File ▸ New).
    pub fn new_project(&mut self) {
        self.project = Project::new("Untitled project");
        self.selection = None;
        // The old project's file is not this project's file: without this, the
        // first Ctrl+S would silently overwrite the document the user still
        // has open in another window.
        self.project_path = None;
        self.undo.reset();
        self.edit.reset();
        self.apply_project_view();
        // A new project is a whole-presentation reset, basemap included. No
        // guard needed: an unchanged basemap derives an unchanged plan, so
        // the reconciliation mirror skips the swap (and its full tile
        // re-fetch) structurally.
        //
        // The project-REPLACEMENT seam again, preceded by `undo.reset()`
        // above. Deliberately NOT `set_basemap_service`: that would stamp
        // `project.basemap = Some(default)` into a brand-new project where
        // `Project::new` leaves it `None`, and a fresh app having no saved
        // basemap is a pinned property.
        self.basemap = BasemapConfig::default();
        let _notices = self.local.rebuild_from_project(&self.project);
        // An empty project has nothing to save, so it starts clean — the same
        // marker reset a load performs, and for the same reason taken after the
        // rebuild has detached the previous project's datasets.
        self.mark_project_loaded();
        self.status = Some("New project created.".to_string());
    }
    /// Restores an existing project layer's GPU copy from the bytes a shell
    /// just read for it.
    ///
    /// Unlike [`Self::add_geojson_layer_from_bytes`] this appends nothing,
    /// changes no selection and does not move the camera: the layer is already
    /// in the project, with its own id, saved style, visibility and opacity, and
    /// the project's saved view has just been restored.
    ///
    /// Returns whether it landed; failures go to the status line.
    pub fn hydrate_geojson_layer_from_bytes(
        &mut self,
        id: LayerId,
        name: &str,
        bytes: &[u8],
    ) -> bool {
        // Before the store moves: a hydrate re-baselines the change
        // watermark, and anything outstanding has to be captured first.
        self.observe_recorded_edits();
        let text = match core::str::from_utf8(bytes) {
            Ok(text) => text,
            Err(error) => {
                self.status = Some(format!("{name} is not UTF-8 text: {error}"));
                return false;
            }
        };
        match self.local.hydrate_geojson(&self.project, id, text) {
            Ok(()) => {
                self.undo.prune_layer(id);
                // The hydrate REPLACED this layer's collection outside the
                // command choke point, so a live index-addressed gesture over
                // it is stale the same way a mid-commit one is (see
                // `apply_transaction`) — drop it before anything can read it.
                if self.selection == Some(id) {
                    self.edit.cancel_live_gestures();
                    // The marquee-marked set survives a cancel by design (an
                    // undo restores the marks of the step it belongs to), but
                    // it addresses the collection that was just replaced — so
                    // here, and only here, it goes with the data it named.
                    self.edit.clear_vertex_marks();
                }
                // A hydrate FINISHES a load: the file already named this layer,
                // so the collection it just installed is not a modification.
                self.absorb_project_change();
                true
            }
            Err(error) => {
                self.status = Some(format!("{name}: {}", error.message()));
                tracing::warn!(name, error = error.message(), "oxigis-ui: rebuild failed");
                false
            }
        }
    }
    /// Adds a local GeoJSON layer from raw file bytes — the seam a native shell
    /// uses after reading a path from [`Self::take_pending_dropped_paths`].
    ///
    /// `path`, when given, is recorded as the layer's source so the project file
    /// stores a reference instead of a copy of the whole document; pass [`None`]
    /// for data that has no file behind it (a browser drop, a paste).
    ///
    /// Returns the new layer's id, or [`None`] when the bytes are not valid
    /// UTF-8 or not a usable GeoJSON `FeatureCollection` — in which case the
    /// reason is put in the status line rather than propagated, since this is
    /// called from a frame loop.
    pub fn add_geojson_layer_from_bytes(
        &mut self,
        name: &str,
        bytes: &[u8],
        path: Option<&str>,
    ) -> Option<LayerId> {
        match core::str::from_utf8(bytes) {
            Ok(text) => {
                let added = self.add_geojson_layer_from_text(name, text, path);
                // The drop gesture records; the `_from_text` primitive stays
                // silent (see `record_layer_add`), which is what keeps test
                // fixtures out of the undo history.
                if let Some(id) = added {
                    self.record_layer_add(&[id]);
                }
                added
            }
            Err(error) => {
                self.status = Some(format!("{name} is not UTF-8 text: {error}"));
                tracing::warn!(name, % error, "oxigis-ui: dropped file is not UTF-8");
                None
            }
        }
    }
    /// Adds a local GeoJSON layer from text, selects it, and zooms the map to
    /// its extent.
    ///
    /// See [`Self::add_geojson_layer_from_bytes`] for `path` and the error
    /// behaviour; this is the shared tail of the drop, paste and shell paths.
    ///
    /// Deliberately records NO undo entry — this is the pure "put this
    /// dataset in the project" primitive. The *gesture* seams (the byte
    /// wrappers, the paste dialog, the Processing result route) call
    /// `record_layer_add` themselves, the same recorder/primitive split
    /// `sync_local_style` / `sync_local_style_for` uses.
    pub fn add_geojson_layer_from_text(
        &mut self,
        name: &str,
        text: &str,
        path: Option<&str>,
    ) -> Option<LayerId> {
        let added = self.local.add_geojson(&mut self.project, name, text, path);
        self.settle_local_add(name, added)
    }
    /// [`Self::add_geojson_layer_from_text`] for a document this process just
    /// produced — a Processing tool's result — which is a
    /// [`serde_json::Value`], not text.
    ///
    /// Identical in every observable way (selects, zooms, warns about an
    /// embedded document at the same threshold, records no undo entry); it
    /// exists so a result never has to be serialised to text purely so that the
    /// text can be parsed straight back. See
    /// [`crate::local_input::LocalInputState::add_geojson_value`] for the
    /// copy-count argument.
    pub(super) fn add_geojson_layer_from_value(
        &mut self,
        name: &str,
        value: serde_json::Value,
    ) -> Option<LayerId> {
        let added = self.local.add_geojson_value(&mut self.project, name, value);
        self.settle_local_add(name, added)
    }
    /// The shared tail of the two GeoJSON add seams: select, stamp, zoom, and
    /// report — or report the refusal.
    ///
    /// Split out rather than duplicated so the embedded-document warning cannot
    /// be present on one route and silently missing on the other: a tool result
    /// is exactly the kind of document that trips
    /// [`INLINE_GEOJSON_WARN_BYTES`].
    fn settle_local_add(
        &mut self,
        name: &str,
        added: Result<local_input::AddedLocalLayer, crate::local_vector::LocalVectorError>,
    ) -> Option<LayerId> {
        match added {
            Ok(added) => {
                self.selection = Some(added.id);
                // Appends to the project outside the transaction choke point,
                // so the unsaved-changes marker is stamped by hand here (and in
                // the three sibling add seams below).
                self.mark_project_dirty();
                self.zoom_to_square(added.square);
                let inline_warning = added
                    .inline_bytes
                    .is_some_and(|bytes| bytes >= INLINE_GEOJSON_WARN_BYTES);
                self.status = Some(if inline_warning {
                    format!(
                        "Added {name} ({} features); its GeoJSON is embedded in the project, \
                         which will make the saved file large.",
                        added.feature_count,
                    )
                } else {
                    format!("Added {name} ({} features).", added.feature_count)
                });
                Some(added.id)
            }
            Err(error) => {
                self.status = Some(format!("{name}: {}", error.message()));
                tracing::warn!(name, error = error.message(), "oxigis-ui: GeoJSON rejected");
                None
            }
        }
    }
    /// Adds a local shapefile layer from the bytes of its `.shp` and whichever
    /// siblings the caller has — the seam a native shell uses after reading a
    /// `.shp` path from [`Self::take_pending_dropped_paths`], and the one the
    /// browser drop path uses directly.
    ///
    /// `dbf` may be [`None`] (geometry-only layer); `prj`/`cpg` are the *text*
    /// of those sidecars. `path`, when given, is recorded as the layer's
    /// [`oxigis_core::VectorSource::LocalShapefile`] source; without one the
    /// dataset is embedded as GeoJSON, since shapefile bytes cannot go into a
    /// project document (see
    /// [`crate::local_input::LocalInputState::add_shapefile`]).
    ///
    /// Returns the new layer's id, or [`None`] with the reason in the status
    /// line — this is called from a frame loop, so nothing is propagated.
    pub fn add_shapefile_layer_from_bytes(
        &mut self,
        name: &str,
        set: crate::shapefile_input::ShapefileBytes<'_>,
        path: Option<&str>,
    ) -> Option<LayerId> {
        match self.local.add_shapefile(&mut self.project, name, set, path) {
            Ok(added) => {
                self.selection = Some(added.id);
                self.mark_project_dirty();
                self.zoom_to_square(added.square);
                let inline_warning = added
                    .inline_bytes
                    .is_some_and(|bytes| bytes >= INLINE_GEOJSON_WARN_BYTES);
                self.status = Some(if inline_warning {
                    format!(
                        "Added {name} ({} features); it is embedded in the project as GeoJSON, \
                         which will make the saved file large.",
                        added.feature_count,
                    )
                } else {
                    format!("Added {name} ({} features).", added.feature_count)
                });
                self.record_layer_add(&[added.id]);
                Some(added.id)
            }
            Err(error) => {
                self.status = Some(format!("{name}: {}", error.message()));
                tracing::warn!(
                    name,
                    error = error.message(),
                    "oxigis-ui: shapefile rejected"
                );
                None
            }
        }
    }
    /// Restores an existing project layer's GPU copy from shapefile bytes a
    /// shell just read for it — [`Self::hydrate_geojson_layer_from_bytes`]'s
    /// shapefile twin, appending nothing and moving no camera.
    ///
    /// Returns whether it landed; failures go to the status line.
    pub fn hydrate_shapefile_layer_from_bytes(
        &mut self,
        id: LayerId,
        name: &str,
        set: crate::shapefile_input::ShapefileBytes<'_>,
    ) -> bool {
        // Before the store moves: a hydrate re-baselines the change
        // watermark, and anything outstanding has to be captured first.
        self.observe_recorded_edits();
        match self.local.hydrate_shapefile(&self.project, id, set) {
            Ok(()) => {
                self.undo.prune_layer(id);
                // The hydrate REPLACED this layer's collection outside the
                // command choke point, so a live index-addressed gesture over
                // it is stale the same way a mid-commit one is (see
                // `apply_transaction`) — drop it before anything can read it.
                if self.selection == Some(id) {
                    self.edit.cancel_live_gestures();
                    // The marquee-marked set survives a cancel by design (an
                    // undo restores the marks of the step it belongs to), but
                    // it addresses the collection that was just replaced — so
                    // here, and only here, it goes with the data it named.
                    self.edit.clear_vertex_marks();
                }
                // A hydrate FINISHES a load: the file already named this layer,
                // so the collection it just installed is not a modification.
                self.absorb_project_change();
                true
            }
            Err(error) => {
                self.status = Some(format!("{name}: {}", error.message()));
                tracing::warn!(name, error = error.message(), "oxigis-ui: rebuild failed");
                false
            }
        }
    }
    /// Adds every feature table of a GeoPackage as its own local layer — the
    /// seam a native shell uses after reading a `.gpkg` path from
    /// [`Self::take_pending_dropped_paths`], and the one the browser drop path
    /// uses directly.
    ///
    /// `path`, when given, is recorded per layer as an
    /// [`oxigis_core::VectorSource::LocalGpkg`] reference naming its table;
    /// without one each table is embedded as GeoJSON (see
    /// [`crate::local_input::LocalInputState::add_gpkg`]).
    ///
    /// Returns the new layers' ids, in file order — the first is selected and
    /// zoomed to, matching the single-layer paths. An empty vector means
    /// nothing was added and the reason is in the status line, which also
    /// carries the per-table refusals when *some* tables loaded and others did
    /// not.
    pub fn add_gpkg_layer_from_bytes(
        &mut self,
        name: &str,
        gpkg: &[u8],
        path: Option<&str>,
    ) -> Vec<LayerId> {
        match self.local.add_gpkg(&mut self.project, name, gpkg, path) {
            Ok(added) => {
                let ids: Vec<LayerId> = added.layers.iter().map(|layer| layer.id).collect();
                if !ids.is_empty() {
                    self.mark_project_dirty();
                }
                if let Some(first) = added.layers.first() {
                    self.selection = Some(first.id);
                    self.zoom_to_square(first.square);
                }
                let features: usize = added.layers.iter().map(|layer| layer.feature_count).sum();
                let inline: usize = added
                    .layers
                    .iter()
                    .filter_map(|layer| layer.inline_bytes)
                    .sum();
                let mut status = if ids.len() == 1 {
                    format!("Added {name} ({features} features).")
                } else {
                    format!(
                        "Added {} layers from {name} ({features} features).",
                        ids.len(),
                    )
                };
                if inline >= INLINE_GEOJSON_WARN_BYTES {
                    status.push_str(
                        " Its tables are embedded in the project as GeoJSON, which will make the \
                         saved file large.",
                    );
                }
                for notice in &added.notices {
                    status.push(' ');
                    status.push_str(notice);
                }
                self.status = Some(status);
                // ONE grouped entry for the whole drop: a 12-table GeoPackage
                // is one gesture, so one Ctrl+Z removes all of it.
                self.record_layer_add(&ids);
                ids
            }
            Err(error) => {
                self.status = Some(format!("{name}: {}", error.message()));
                tracing::warn!(
                    name,
                    error = error.message(),
                    "oxigis-ui: GeoPackage rejected"
                );
                Vec::new()
            }
        }
    }
    /// Restores an existing project layer's GPU copy from the GeoPackage one of
    /// its tables came from — [`Self::hydrate_shapefile_layer_from_bytes`]'s
    /// GeoPackage twin, appending nothing and moving no camera.
    ///
    /// `table` is the name the project file recorded alongside the path; a file
    /// that no longer holds it reports so rather than substituting another.
    ///
    /// Returns whether it landed; failures go to the status line.
    pub fn hydrate_gpkg_layer_from_bytes(
        &mut self,
        id: LayerId,
        name: &str,
        gpkg: &[u8],
        table: &str,
    ) -> bool {
        // Before the store moves: a hydrate re-baselines the change
        // watermark, and anything outstanding has to be captured first.
        self.observe_recorded_edits();
        match self.local.hydrate_gpkg(&self.project, id, gpkg, table) {
            Ok(()) => {
                self.undo.prune_layer(id);
                // The hydrate REPLACED this layer's collection outside the
                // command choke point, so a live index-addressed gesture over
                // it is stale the same way a mid-commit one is (see
                // `apply_transaction`) — drop it before anything can read it.
                if self.selection == Some(id) {
                    self.edit.cancel_live_gestures();
                    // The marquee-marked set survives a cancel by design (an
                    // undo restores the marks of the step it belongs to), but
                    // it addresses the collection that was just replaced — so
                    // here, and only here, it goes with the data it named.
                    self.edit.clear_vertex_marks();
                }
                // A hydrate FINISHES a load: the file already named this layer,
                // so the collection it just installed is not a modification.
                self.absorb_project_change();
                true
            }
            Err(error) => {
                self.status = Some(format!("{name}: {}", error.message()));
                tracing::warn!(name, error = error.message(), "oxigis-ui: rebuild failed");
                false
            }
        }
    }
    /// Adds a local GeoParquet layer from raw file bytes — the seam a native
    /// shell uses after reading a `.parquet`/`.geoparquet` path from
    /// [`Self::take_pending_dropped_paths`].
    ///
    /// Compiled only under the `geoparquet` Cargo feature (native-only — see
    /// `crate::geoparquet_input`'s module docs); a browser build never has
    /// this method, which is why the drop handler that would call it has two
    /// `#[cfg]`-gated bodies rather than calling it unconditionally.
    ///
    /// `path`, when given, is recorded as the layer's
    /// [`oxigis_core::VectorSource::LocalGeoParquet`] source; without one the
    /// dataset is embedded as GeoJSON (see
    /// [`crate::local_input::LocalInputState::add_geoparquet`]).
    ///
    /// Returns the new layer's id, or [`None`] with the reason in the status
    /// line — this is called from a frame loop, so nothing is propagated.
    #[cfg(feature = "geoparquet")]
    pub fn add_geoparquet_layer_from_bytes(
        &mut self,
        name: &str,
        bytes: &[u8],
        path: Option<&str>,
    ) -> Option<LayerId> {
        match self
            .local
            .add_geoparquet(&mut self.project, name, bytes, path)
        {
            Ok(added) => {
                self.selection = Some(added.id);
                self.mark_project_dirty();
                self.zoom_to_square(added.square);
                let inline_warning = added
                    .inline_bytes
                    .is_some_and(|bytes| bytes >= INLINE_GEOJSON_WARN_BYTES);
                self.status = Some(if inline_warning {
                    format!(
                        "Added {name} ({} features); it is embedded in the project as GeoJSON, \
                         which will make the saved file large.",
                        added.feature_count,
                    )
                } else {
                    format!("Added {name} ({} features).", added.feature_count)
                });
                self.record_layer_add(&[added.id]);
                Some(added.id)
            }
            Err(error) => {
                self.status = Some(format!("{name}: {}", error.message()));
                tracing::warn!(
                    name,
                    error = error.message(),
                    "oxigis-ui: GeoParquet rejected"
                );
                None
            }
        }
    }
    /// Restores an existing project layer's GPU copy from GeoParquet bytes a
    /// shell just read for it — [`Self::hydrate_gpkg_layer_from_bytes`]'s
    /// GeoParquet twin, appending nothing and moving no camera.
    ///
    /// Compiled only under the `geoparquet` Cargo feature; see
    /// [`Self::add_geoparquet_layer_from_bytes`].
    ///
    /// Returns whether it landed; failures go to the status line.
    #[cfg(feature = "geoparquet")]
    pub fn hydrate_geoparquet_layer_from_bytes(
        &mut self,
        id: LayerId,
        name: &str,
        bytes: &[u8],
    ) -> bool {
        // Before the store moves: a hydrate re-baselines the change
        // watermark, and anything outstanding has to be captured first.
        self.observe_recorded_edits();
        match self.local.hydrate_geoparquet(&self.project, id, bytes) {
            Ok(()) => {
                self.undo.prune_layer(id);
                // The hydrate REPLACED this layer's collection outside the
                // command choke point, so a live index-addressed gesture over
                // it is stale the same way a mid-commit one is (see
                // `apply_transaction`) — drop it before anything can read it.
                if self.selection == Some(id) {
                    self.edit.cancel_live_gestures();
                    // The marquee-marked set survives a cancel by design (an
                    // undo restores the marks of the step it belongs to), but
                    // it addresses the collection that was just replaced — so
                    // here, and only here, it goes with the data it named.
                    self.edit.clear_vertex_marks();
                }
                // A hydrate FINISHES a load: the file already named this layer,
                // so the collection it just installed is not a modification.
                self.absorb_project_change();
                true
            }
            Err(error) => {
                self.status = Some(format!("{name}: {}", error.message()));
                tracing::warn!(name, error = error.message(), "oxigis-ui: rebuild failed");
                false
            }
        }
    }
    /// Consumes this frame's dropped files.
    ///
    /// The batch is **grouped before anything is loaded**
    /// ([`local_input::group_dropped_files`]): a shapefile arrives as up to five
    /// separate `DroppedFile`s in one drop and only makes sense reassembled, so
    /// per-file routing is not an option. Each resulting dataset then follows
    /// the same two-way split as before — files carrying `bytes` (the browser
    /// path) are parsed straight away, files carrying only a `path` (the
    /// `egui-winit` native path) go to [`Self::take_pending_dropped_paths`] for
    /// the shell to read, because this crate does no I/O.
    pub(super) fn handle_dropped_files(&mut self, ctx: &Context) {
        let dropped = ctx.input(|input| input.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }
        let items: Vec<local_input::DroppedItem> = dropped
            .into_iter()
            .filter_map(|file| {
                let raw_name = if file.name.is_empty() {
                    file.path
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned())
                        .unwrap_or_default()
                } else {
                    file.name.clone()
                };
                let name = local_input::display_name(&raw_name);
                (!name.is_empty()).then(|| local_input::DroppedItem {
                    name,
                    bytes: file.bytes.clone(),
                    path: file.path.clone(),
                })
            })
            .collect();
        let (datasets, notices) = local_input::group_dropped_files(items);
        for notice in notices {
            self.status = Some(notice);
        }
        for dataset in datasets {
            match dataset {
                local_input::DroppedDataset::GeoJson(item) => self.load_dropped_geojson(&item),
                local_input::DroppedDataset::Shapefile(set) => self.load_dropped_shapefile(&set),
                local_input::DroppedDataset::GeoPackage(item) => self.load_dropped_gpkg(&item),
                local_input::DroppedDataset::GeoParquet(item) => {
                    self.load_dropped_geoparquet(&item);
                }
                local_input::DroppedDataset::GeoLibreProject(item) => {
                    self.load_dropped_geolibre_project(&item);
                }
                local_input::DroppedDataset::TileArchive(format, item) => {
                    self.load_dropped_archive(format, &item, ctx);
                }
            }
        }
    }
    /// Loads (or queues) one dropped file whose name ends `.geolibre.json` —
    /// routed here instead of [`Self::load_dropped_geojson`] by
    /// [`local_input::classify_drop`], since an explicit `.geolibre.json`
    /// name is an unambiguous signal to import it as a *project*, replacing
    /// the current one, exactly like File ▸ Open… with GeoLibre-shaped
    /// pasted text (both ultimately funnel through `load_project_text`, so
    /// `.oxigis.json` content saved under a `.geolibre.json` name by
    /// mistake still loads natively).
    ///
    /// The native (path-only) leg queues the path with **no** `layer`
    /// ([`local_input::PendingPath::layer`] stays [`None`]) exactly like a
    /// fresh GeoJSON/Shapefile/GeoPackage/GeoParquet drop does — a shell
    /// reads it back through [`Self::load_geolibre_project_from_bytes`],
    /// which a native shell's `classify_drop`-based path dispatch (the same
    /// mechanism it already uses to tell a `.shp` from a `.gpkg`) can reach
    /// exactly as easily as any other drop kind; there is nothing special
    /// about a whole-project read that needs a wider [`local_input::PendingPath`].
    fn load_dropped_geolibre_project(&mut self, item: &local_input::DroppedItem) {
        match (item.bytes.as_ref(), item.path.as_ref()) {
            (Some(bytes), _) => self.load_geolibre_project_from_bytes(&item.name, bytes),
            (None, Some(path)) if self.local.paths_supported() => {
                self.local.queue_path(None, path.clone());
                self.status = Some(format!("Reading {}…", item.name));
            }
            _ => {
                self.status = Some(format!("{} arrived without any data.", item.name));
            }
        }
    }
    /// Reads a `.geolibre.json` project from raw file bytes — the seam a
    /// native shell uses after reading a path from
    /// [`Self::take_pending_dropped_paths`] whose
    /// [`local_input::PendingPath::layer`] is [`None`] (a fresh
    /// project-import drop; a `Some(id)` for that same file name is instead
    /// a `LocalGeoJson`/etc. layer whose file simply happens to be named
    /// `*.geolibre.json` — a shell must route that one through
    /// [`Self::hydrate_geojson_layer_from_bytes`] as usual, not here).
    ///
    /// Detection order matches [`Self::load_project`]/every other load path:
    /// [`Project::from_json_string`] first, GeoLibre content-sniffing only
    /// on its failure (see `load_project_text`). Failures (non-UTF-8 bytes,
    /// a document that is neither) go to the status line rather than
    /// propagating, since this is called from a frame-driven drain loop.
    pub fn load_geolibre_project_from_bytes(&mut self, name: &str, bytes: &[u8]) {
        let text = match core::str::from_utf8(bytes) {
            Ok(text) => text,
            Err(error) => {
                self.status = Some(format!("{name} is not UTF-8 text: {error}"));
                return;
            }
        };
        match load_project_text(text) {
            Ok(LoadedProject::Native(project)) => self.load_project(project),
            Ok(LoadedProject::GeoLibre(project, notices)) => {
                self.load_geolibre_project(project, notices);
            }
            Err(message) => self.status = Some(format!("{name}: {message}")),
        }
    }
    /// Loads a project document a shell read off disk, answering whether it
    /// parsed — the File ▸ Open seam's other half.
    ///
    /// Same detection ladder as every other load path (native
    /// [`Project::from_json_string`] first, GeoLibre content-sniffing only on
    /// its failure), and the same status-line reporting. The `bool` is what a
    /// shell needs that the drop path does not: a file that did not parse must
    /// not become "the open project's path", or the next Ctrl+S would write the
    /// project over a document it failed to read.
    pub fn load_project_from_text(&mut self, text: &str) -> bool {
        match load_project_text(text) {
            Ok(LoadedProject::Native(project)) => {
                self.load_project(project);
                true
            }
            Ok(LoadedProject::GeoLibre(project, notices)) => {
                self.load_geolibre_project(project, notices);
                true
            }
            Err(message) => {
                self.status = Some(format!("That file is not an OxiGIS project: {message}"));
                false
            }
        }
    }
    /// Loads (or queues) one dropped GeoParquet file.
    ///
    /// Compiled only under the `geoparquet` Cargo feature; the `#[cfg(not(...))]`
    /// twin below reports a notice instead, so a browser build (which never
    /// enables the feature — see `crate::geoparquet_input`'s module docs)
    /// tells the user why the drop did nothing rather than falling through to
    /// [`local_input::DropKind::Unsupported`]'s generic message or, worse,
    /// silently doing nothing at all.
    #[cfg(feature = "geoparquet")]
    fn load_dropped_geoparquet(&mut self, item: &local_input::DroppedItem) {
        match (item.bytes.as_ref(), item.path.as_ref()) {
            (Some(bytes), _) => {
                self.add_geoparquet_layer_from_bytes(&item.name, bytes, None);
            }
            (None, Some(path)) if self.local.paths_supported() => {
                self.local.queue_path(None, path.clone());
                self.status = Some(format!("Reading {}…", item.name));
            }
            _ => {
                self.status = Some(format!("{} arrived without any data.", item.name));
            }
        }
    }
    /// The `geoparquet`-feature-off twin of the method above: this build was
    /// never going to be able to read the file (see
    /// `crate::geoparquet_input`'s module docs — GeoParquet is native-only),
    /// so the drop is reported rather than silently discarded or misparsed.
    #[cfg(not(feature = "geoparquet"))]
    fn load_dropped_geoparquet(&mut self, item: &local_input::DroppedItem) {
        self.status = Some(format!(
            "{} is a GeoParquet file, which this build does not support (native desktop \
             only).",
            item.name,
        ));
    }
    /// Loads (or queues) one dropped GeoPackage.
    ///
    /// The browser leg produces *several* layers from the one file; the native
    /// leg queues the path with no table name, which is what tells the shell's
    /// reader this is a fresh drop importing every table rather than a
    /// project-load reference rebuilding one (see
    /// [`crate::local_input::PendingPath::table`]).
    fn load_dropped_gpkg(&mut self, item: &local_input::DroppedItem) {
        match (item.bytes.as_ref(), item.path.as_ref()) {
            (Some(bytes), _) => {
                self.add_gpkg_layer_from_bytes(&item.name, bytes, None);
            }
            (None, Some(path)) if self.local.paths_supported() => {
                self.local.queue_path(None, path.clone());
                self.status = Some(format!("Reading {}…", item.name));
            }
            _ => {
                self.status = Some(format!("{} arrived without any data.", item.name));
            }
        }
    }
    /// Loads (or queues) one dropped GeoJSON document.
    ///
    /// A bare `.json`/`.geojson` name (as opposed to `.geolibre.json`, which
    /// [`local_input::classify_drop`] already routes to
    /// [`Self::load_dropped_geolibre_project`]) is never auto-imported as a
    /// project even if its content would sniff as one — see
    /// [`Self::hint_if_geolibre_project`] — because the extension alone is
    /// ambiguous: plenty of real GeoJSON datasets are also named plain
    /// `.json`, so silently reinterpreting a failed one as "replace your
    /// whole project" would be surprising.
    fn load_dropped_geojson(&mut self, item: &local_input::DroppedItem) {
        match (item.bytes.as_ref(), item.path.as_ref()) {
            (Some(bytes), _) => {
                if self
                    .add_geojson_layer_from_bytes(&item.name, bytes, None)
                    .is_none()
                {
                    self.hint_if_geolibre_project(bytes);
                }
            }
            (None, Some(path)) if self.local.paths_supported() => {
                self.local.queue_path(None, path.clone());
                self.status = Some(format!("Reading {}…", item.name));
            }
            _ => {
                self.status = Some(format!(
                    "{} arrived without any data; try the layer panel's \
                     \u{201c}+ GeoJSON\u{201d} paste button.",
                    item.name,
                ));
            }
        }
    }
    /// After a `.json`/`.geojson` drop fails to parse as a GeoJSON
    /// `FeatureCollection`, checks whether it is plausibly a GeoLibre
    /// project instead and, if so, replaces the generic parse-failure
    /// status [`Self::add_geojson_layer_from_bytes`] already set with a
    /// pointer to File ▸ Open… — the deliberate ambiguous-extension
    /// safeguard documented on [`Self::load_dropped_geojson`]: a bare
    /// `.json` drop is never auto-imported as a project, only a name ending
    /// `.geolibre.json` is. Does nothing (leaving the original status) when
    /// the bytes are not even valid JSON, or don't sniff as GeoLibre.
    fn hint_if_geolibre_project(&mut self, bytes: &[u8]) {
        let Ok(text) = core::str::from_utf8(bytes) else {
            return;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
            return;
        };
        if crate::geolibre_import::looks_like_geolibre(&value) {
            self.status = Some(
                "This looks like a GeoLibre project — open it via File > Open… or rename it \
                 *.geolibre.json."
                    .to_string(),
            );
        }
    }
    /// Loads (or queues) one dropped shapefile set.
    ///
    /// The native leg queues **only the `.shp` path**: the shell finds the
    /// siblings itself by swapping the extension, which is both simpler than
    /// widening [`crate::local_input::PendingPath`] and the only thing that
    /// works for the project-load path, where no `.dbf` was ever dropped.
    fn load_dropped_shapefile(&mut self, set: &local_input::ShapefileDrop) {
        let name = set.shp.name.clone();
        match (set.shp.bytes.as_ref(), set.shp.path.as_ref()) {
            (Some(shp), _) => {
                let dbf = set.dbf.as_ref().and_then(|item| item.bytes.as_deref());
                let prj = set.prj.as_ref().and_then(sidecar_text);
                let cpg = set.cpg.as_ref().and_then(sidecar_text);
                let bytes = crate::shapefile_input::ShapefileBytes::new(shp)
                    .with_dbf(dbf)
                    .with_sidecars(prj.as_deref(), cpg.as_deref());
                self.add_shapefile_layer_from_bytes(&name, bytes, None);
            }
            (None, Some(path)) if self.local.paths_supported() => {
                self.local.queue_path(None, path.clone());
                self.status = Some(format!("Reading {name}…"));
            }
            _ => {
                self.status = Some(format!("{name} arrived without any data."));
            }
        }
    }
    /// Draws the active File ▸ Open/Save modal, if any.
    pub(super) fn io_dialog_window(&mut self, ctx: &Context) {
        // Read BEFORE the dialog is borrowed mutably: the confirmation's Save
        // button promises different things on the two shells, and `self` is not
        // readable again until the match is done with it.
        let native_project_io = self.native_project_io;
        let Some(dialog) = self.io_dialog.as_mut() else {
            return;
        };
        let mut close = false;
        let mut loaded_project = None;
        let mut geolibre_notices = None;
        let mut pasted_geojson = None;
        let mut discard = None;
        let mut save_then = None;
        match dialog {
            IoDialog::ConfirmDiscard { pending } => {
                let pending = pending.clone();
                egui::Window::new(pending.headline())
                    .collapsible(false)
                    .resizable(false)
                    .show(ctx, |ui| {
                        ui.label(pending.consequence());
                        ui.separator();
                        ui.horizontal(|ui| {
                            // Two different promises, because the two shells
                            // really do behave differently: a shell that can
                            // write files finishes the save and THEN continues,
                            // while the copy-JSON modal is not finished until
                            // the user has selected the document out of it —
                            // so there the pending action is deliberately
                            // dropped rather than run behind their back.
                            let hover = if native_project_io {
                                "Writes the project to disk, then continues"
                            } else {
                                "Shows the project's JSON to copy out; run this command again \
                                 afterwards."
                            };
                            if ui.button("Save\u{2026}").on_hover_text(hover).clicked() {
                                save_then = Some(pending.clone());
                            }
                            if ui
                                .button("Discard changes")
                                .on_hover_text("Throws the unsaved changes away and continues")
                                .clicked()
                            {
                                discard = Some(pending.clone());
                            }
                            if ui.button("Cancel").clicked() {
                                close = true;
                            }
                        });
                    });
            }
            IoDialog::Open { buffer, error } => {
                egui::Window::new("Open project (paste JSON)")
                    .collapsible(false)
                    .show(ctx, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(buffer)
                                .desired_rows(12)
                                .desired_width(420.0),
                        );
                        if let Some(message) = error {
                            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), message);
                        }
                        ui.horizontal(|ui| {
                            if ui.button("Load").clicked() {
                                match load_project_text(buffer) {
                                    Ok(LoadedProject::Native(project)) => {
                                        loaded_project = Some(project);
                                    }
                                    Ok(LoadedProject::GeoLibre(project, notices)) => {
                                        loaded_project = Some(project);
                                        geolibre_notices = Some(notices);
                                    }
                                    Err(message) => *error = Some(message),
                                }
                            }
                            if ui.button("Cancel").clicked() {
                                close = true;
                            }
                        });
                    });
            }
            IoDialog::Save { content } => {
                egui::Window::new("Save project (copy JSON)")
                    .collapsible(false)
                    .show(ctx, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(content)
                                .desired_rows(12)
                                .desired_width(420.0)
                                .interactive(false),
                        );
                        if ui.button("Close").clicked() {
                            close = true;
                        }
                    });
            }
            IoDialog::ExportGeoJson { name, content } => {
                egui::Window::new(format!("{name} (copy GeoJSON)"))
                    .collapsible(false)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.weak(format!("{} bytes", content.len()));
                            if ui.button("Copy").clicked() {
                                ui.ctx().copy_text(content.clone());
                            }
                        });
                        // A `&mut &str` (not `&mut String`) is the documented
                        // egui idiom for "selectable but not editable": `&str`'s
                        // `TextBuffer` impl refuses every mutation, so the field
                        // keeps cursor and selection support without
                        // `.interactive(false)`, which would disable selection
                        // along with editing and leave the document impossible
                        // to drag-select out of the app.
                        let mut display = content.as_str();
                        ui.add(
                            egui::TextEdit::multiline(&mut display)
                                .desired_rows(14)
                                .desired_width(460.0),
                        );
                        if ui.button("Close").clicked() {
                            close = true;
                        }
                    });
            }
            IoDialog::PasteGeoJson {
                name,
                buffer,
                error,
            } => {
                egui::Window::new("Add GeoJSON layer (paste)")
                    .collapsible(false)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Layer name");
                            ui.add(egui::TextEdit::singleline(name).desired_width(260.0));
                        });
                        ui.add(
                            egui::TextEdit::multiline(buffer)
                                .hint_text("A GeoJSON FeatureCollection")
                                .desired_rows(12)
                                .desired_width(420.0),
                        );
                        if let Some(message) = error {
                            ui.colored_label(
                                egui::Color32::from_rgb(220, 80, 80),
                                message.as_str(),
                            );
                        }
                        ui.horizontal(|ui| {
                            if ui.button("Add layer").clicked() {
                                *error = None;
                                pasted_geojson = Some((name.clone(), core::mem::take(buffer)));
                            }
                            if ui.button("Cancel").clicked() {
                                close = true;
                            }
                        });
                    });
            }
        }
        if let Some(pending) = discard {
            // The confirmation is spent either way: `run_pending_action` either
            // replaces the project outright or installs the Open modal in this
            // slot, and neither wants the question still on screen.
            self.io_dialog = None;
            self.run_pending_action(pending);
            return;
        }
        if let Some(pending) = save_then {
            // The question is spent either way; what replaces it is the save.
            self.io_dialog = None;
            if self.native_project_io {
                // Parked, not run: `confirm_project_saved` runs it once the
                // shell reports the bytes are on disk, and every other outcome
                // (a cancelled dialog, a failed write, a project that will not
                // serialize) clears it — so a "Save" that did not save can
                // never quietly discard the project.
                self.after_save = Some(pending);
                self.request_project_save(false);
            } else {
                self.open_save_dialog();
            }
            return;
        }
        if let Some(project) = loaded_project {
            match geolibre_notices.take() {
                Some(notices) => self.load_geolibre_project(project, notices),
                None => self.load_project(project),
            }
            close = true;
        }
        if let Some((name, text)) = pasted_geojson {
            if let Some(id) = self.add_geojson_layer_from_text(&name, &text, None) {
                self.record_layer_add(&[id]);
                close = true;
            } else {
                let message = self.status.clone();
                if let Some(IoDialog::PasteGeoJson { buffer, error, .. }) = self.io_dialog.as_mut()
                {
                    *buffer = text;
                    *error = message;
                }
            }
        }
        if close {
            self.io_dialog = None;
        }
    }
}
/// What [`load_project_text`] recovered from a document — either format,
/// which the caller (the Open dialog, a `.geolibre.json` drop) routes on to
/// [`OxigisApp::load_project`] or [`OxigisApp::load_geolibre_project`]
/// respectively.
#[derive(Debug)]
pub(super) enum LoadedProject {
    /// Parsed natively by [`Project::from_json_string`] — no import notices.
    Native(Project),
    /// Recovered by [`crate::geolibre_import::import`], with its notices.
    GeoLibre(Project, Vec<String>),
}
/// The detection order every "give me a `Project`, trying our own format
/// first" load path shares (GeoLibre compat audit §8):
/// [`Project::from_json_string`] is authoritative when it succeeds — this
/// must never regress `.oxigis.json` loading — and only on its failure does
/// content-sniffing for a GeoLibre `.geolibre.json` document kick in. When
/// `text` is neither, the *original* native parse error is what comes back
/// (as its `Debug` rendering, matching the wording the Open dialog always
/// showed before this module existed), never a fabricated "not GeoLibre
/// either" message.
pub(super) fn load_project_text(text: &str) -> Result<LoadedProject, String> {
    match Project::from_json_string(text) {
        Ok(project) => Ok(LoadedProject::Native(project)),
        Err(parse_error) => {
            let sniffable = serde_json::from_str::<serde_json::Value>(text)
                .ok()
                .filter(crate::geolibre_import::looks_like_geolibre);
            let Some(value) = sniffable else {
                return Err(format!("{parse_error:?}"));
            };
            crate::geolibre_import::import(&value)
                .map(|(project, notices)| LoadedProject::GeoLibre(project, notices))
                .map_err(|import_error| import_error.message().to_string())
        }
    }
}
/// The text of a `.prj` / `.cpg` sidecar that arrived as bytes.
///
/// Both are tiny ASCII files; anything that is not valid UTF-8 is treated as
/// absent, which for a `.prj` means "assume WGS 84" and for a `.cpg` means
/// "fall back to the DBF header's code page" — both better than refusing the
/// whole dataset over a sidecar.
fn sidecar_text(item: &crate::local_input::DroppedItem) -> Option<String> {
    let bytes = item.bytes.as_deref()?;
    core::str::from_utf8(bytes).ok().map(str::to_owned)
}
