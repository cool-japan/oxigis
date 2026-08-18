// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Action-dispatch half of [`OxigisApp`]: applying the layer / style /
//! processing actions the panels report each frame, plus the Processing
//! window itself. Split from `app/mod.rs` under the 2000-line rule; the
//! frame loop and accessors stay there, the data-ingestion seams live in
//! `app::data_io`.

use super::OxigisApp;
use super::data_io::IoDialog;
use crate::edit::project_op::{BasemapServiceChange, ProjectOp, ProjectTransaction};
use crate::layer_panel::{self, LayerAction};
use crate::local_input::{self, LocalLayerOp};
use crate::processing_panel::{
    self, OutputDestination, OutputTarget, ProcessingAction, ProcessingFileRequest, ProcessingJob,
    ProcessingPoll, SelectionSummary,
};
use crate::style_panel::StyleAction;
use crate::tile_provider::BasemapConfig;
use egui::Context;
use oxigeo::geojson::types::FeatureCollection;
use oxigis_core::LayerId;
use std::collections::BTreeMap;
use std::sync::Arc;

/// The clause every basemap-pointer change ends with, so the promise is
/// worded once and cannot drift between the toggle and the preset picker.
const BASEMAP_UNDO_CLAUSE: &str = "Ctrl+Z restores the previous basemap.";

/// Above how many features a live style preview during a colour/width drag
/// costs more than it is worth (see [`OxigisApp::sync_local_style`]).
///
/// Sized so that everything a user draws by hand, and every ordinary
/// administrative-boundary or POI extract, keeps the immediate preview: the
/// deferral is for the imported-a-whole-country case, where one frame of
/// tessellation is measured in hundreds of milliseconds.
const STYLE_PREVIEW_MAX_FEATURES: usize = 20_000;

impl OxigisApp {
    /// Runs a Processing tool descriptor against `params` (already resolved
    /// to JSON, keyed by [`oxigis_core::ParamSpec::name`] — see
    /// [`crate::processing_panel::draw`]), returning its result value or a
    /// human-readable reason it could not run.
    ///
    /// Scoped to descriptors with exactly one
    /// [`oxigis_core::ParamKind::LayerRef`] parameter — the shape every
    /// [`oxigis_core::builtin_registry`] tool has today; see
    /// [`crate::processing_exec::builtin_executor`] for the non-goal this
    /// shares of not yet supporting a multi-layer tool.
    ///
    /// # Errors
    ///
    /// Returns `Err` when `descriptor` has no `LayerRef` parameter, the
    /// parameter is missing or does not name a layer, the named layer's
    /// features are not loaded, or no executor is wired for the descriptor's
    /// id — never panics on a malformed `params` map.
    pub fn run_processing_tool(
        &mut self,
        descriptor: &oxigis_core::ToolDescriptor,
        params: BTreeMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, String> {
        let features = self.tool_features(descriptor, &params, false)?;
        let context = oxigis_core::ToolContext { params };
        crate::processing_exec::builtin_executor(&descriptor.id, features)
            .ok_or_else(|| format!("{} is not implemented yet", descriptor.title))?
            .run(&context)
            .map_err(|error| error.to_string())
    }

    /// The features a run of `descriptor` reads: the layer its
    /// [`oxigis_core::ParamKind::LayerRef`] parameter names, narrowed to the
    /// current selection when `selected_only` is set.
    ///
    /// Shared by the synchronous [`Self::run_processing_tool`] and the
    /// interruptible [`Self::start_processing_run`], so "which features does
    /// this tool see" has exactly one answer however the run is driven.
    ///
    /// # Errors
    ///
    /// Returns `Err` when `descriptor` has no `LayerRef` parameter, the
    /// parameter is missing or does not name a layer, or the named layer's
    /// features are not loaded — never panics on a malformed `params` map.
    fn tool_features(
        &self,
        descriptor: &oxigis_core::ToolDescriptor,
        params: &BTreeMap<String, serde_json::Value>,
        selected_only: bool,
    ) -> Result<Arc<FeatureCollection>, String> {
        let layer_param = descriptor
            .params
            .iter()
            .find(|param| matches!(param.kind, oxigis_core::ParamKind::LayerRef))
            .ok_or_else(|| {
                format!(
                    "{}: multi-/zero-layer tools are not yet supported",
                    descriptor.title
                )
            })?;
        let value = params
            .get(&layer_param.name)
            .ok_or_else(|| format!("{} is required", layer_param.name))?;
        let layer_id: LayerId = serde_json::from_value(value.clone())
            .map_err(|_error| format!("{} must name a layer", layer_param.name))?;
        let features = self
            .local
            .feature_set(layer_id)
            .cloned()
            .ok_or_else(|| "the selected layer's data is not loaded".to_string())?;
        if !selected_only {
            return Ok(features);
        }
        // The selection is only ever applied to the layer it actually
        // addresses: a set of source indices means something else entirely in
        // a different collection, so a mismatch runs over the whole layer
        // (what the unrestricted run would have done) rather than over an
        // arbitrary subset of the wrong features. The panel already refuses to
        // offer the checkbox in that case — this is the same rule enforced at
        // the applier, where it cannot be bypassed.
        let Some(selected) = self.selected_source_features(layer_id) else {
            return Ok(features);
        };
        Ok(Arc::new(FeatureCollection::new(
            selected
                .iter()
                .filter_map(|index| features.features.get(*index).cloned())
                .collect(),
        )))
    }

    /// The selected feature indices, when the selection addresses `layer`.
    ///
    /// Read from the **edit** state, not from the attribute table: the table
    /// mirrors this set every frame (`sync_table_selection`), so the edit
    /// state is the model and the table the view. Indices are into the
    /// layer's *source* collection, ascending and duplicate-free by
    /// [`crate::edit::selection::FeatureSelection`]'s own invariants, and
    /// bounded by its `MAX_MULTI_SELECT`.
    fn selected_source_features(&self, layer: LayerId) -> Option<&[usize]> {
        if self.edit.target() != Some(layer) {
            return None;
        }
        Some(self.edit.multi_selection()?.features())
    }

    /// What the Processing form needs to know about the current selection.
    pub(super) fn selection_summary(&self) -> SelectionSummary {
        match self.edit.multi_selection() {
            Some(multi) => SelectionSummary {
                layer: self.edit.target(),
                count: multi.len(),
            },
            None => SelectionSummary::default(),
        }
    }

    /// Starts an interruptible run of `descriptor` — the path the Run button
    /// takes.
    ///
    /// # Errors
    ///
    /// Returns the same refusals [`Self::tool_features`] does, plus the
    /// tool's own parameter validation and "no executor is wired for this id"
    /// — all of them before any feature is touched.
    fn start_processing_run(
        &mut self,
        descriptor: &oxigis_core::ToolDescriptor,
        params: BTreeMap<String, serde_json::Value>,
        selected_only: bool,
    ) -> Result<crate::processing_exec::ToolRun, String> {
        let features = self.tool_features(descriptor, &params, selected_only)?;
        let context = oxigis_core::ToolContext { params };
        crate::processing_exec::start_builtin_run(&descriptor.id, features, &context)
    }

    /// The Processing run in flight, as its tool's title and how far it has
    /// got — [`None`] when nothing is running.
    ///
    /// The Processing window draws this itself; this is for a shell that
    /// wants to mirror a long run where the window is not (a taskbar progress
    /// indicator, a title bar, a headless harness asserting a run finished).
    #[must_use]
    pub fn processing_progress(&self) -> Option<(String, crate::processing_exec::ToolProgress)> {
        let job = self.processing.job()?;
        Some((job.title().to_string(), job.progress()))
    }

    /// Takes the Processing result the user asked to have written to a file.
    ///
    /// The shell-side half of [`OutputDestination::File`], and the same
    /// take-once shape [`Self::take_pending_print`] has: `oxigis-ui` compiles
    /// to `wasm32` and owns no filesystem, so a native shell writes the
    /// document (through its own file dialog) and a browser shell offers it
    /// as a download. A build that drains neither leaves the request here,
    /// which is exactly why the window's own message says so rather than
    /// claiming the file was written.
    ///
    /// ```text
    /// if let Some(request) = self.app.take_pending_processing_save() {
    ///     // native: rfd::FileDialog::new().set_file_name(format!("{}.geojson", request.name))…
    ///     std::fs::write(path, request.content)?;
    /// }
    /// ```
    pub fn take_pending_processing_save(&mut self) -> Option<ProcessingFileRequest> {
        self.processing.take_file_save()
    }

    /// Mirrors a completed style-panel gesture, i.e. with no pointer held —
    /// the shorthand the sibling test modules drive the seam through.
    #[cfg(test)]
    pub(super) fn sync_local_style(&mut self, before: Option<oxigis_core::LayerStyleSet>) {
        self.sync_local_style_gated(before, false);
    }

    /// Mirrors a style-panel edit of the selected layer into the GPU-side local
    /// layer, if the selected layer is one.
    ///
    /// `before` is the layer's style as it stood *before* this frame's style
    /// panel ran; the comparison keeps an untouched frame from queueing a
    /// (synchronous, mesh-rebuilding) restyle 60 times a second. Removing the
    /// style entry restores the style the layer was created with, so the layer
    /// does not silently keep the deleted one.
    ///
    /// `pointer_down` is what makes a colour or width DRAG affordable on a
    /// large dataset. Colour is baked into the mesh's vertices, so every step
    /// of a slider is a full re-tessellation of the whole layer on the render
    /// thread; past [`STYLE_PREVIEW_MAX_FEATURES`] the GPU half is held back
    /// until the pointer lifts, which turns one stall per frame of the drag
    /// into one stall at the end of it. The undo entry is recorded either way —
    /// deferral is a rendering decision, never a project one.
    pub(super) fn sync_local_style_gated(
        &mut self,
        before: Option<oxigis_core::LayerStyleSet>,
        pointer_down: bool,
    ) {
        self.flush_deferred_restyle(pointer_down);
        let Some(id) = self.selection else { return };
        if !self.is_local_layer(id) {
            return;
        }
        let after = self.project.styles.get(&id).cloned();
        if after == before {
            return;
        }
        self.mark_project_dirty();
        // Recorded coalesced at this seam — the one place a style-panel edit
        // is observed as a before/after pair. The choke point's own style
        // writes go through `sync_local_style_for` instead, so an undo's
        // write cannot re-record itself here.
        self.record_undo(crate::edit::project_op::ProjectTransaction {
            label: "Edit style",
            op: crate::edit::project_op::ProjectOp::SetStyle {
                layer: id,
                before: before.map(Box::new),
                after: after.clone().map(Box::new),
            },
            coalesce: Some(crate::edit::stack::CoalesceKey {
                epoch: self.undo.epoch(),
                layer: id,
                feature: 0,
                // The SLOT discriminates the window: a polygon-fill drag
                // and a line-width drag are structurally two undo steps.
                field: crate::edit::stack::CoalesceField::Style(self.style_panel_state.slot()),
            }),
        });
        if pointer_down && self.style_preview_is_too_expensive(id) {
            // The mesh the map is drawing stays the pre-drag one until the
            // gesture ends; the project is already the new one, so nothing but
            // the preview is behind.
            self.deferred_restyle = Some(id);
            return;
        }
        self.deferred_restyle = None;
        if let Some(style) = after.or_else(|| self.local.default_style(id).cloned()) {
            self.local.queue(LocalLayerOp::SetStyle(id, style));
        }
    }

    /// Whether re-tessellating `layer` on every frame of a drag is too much
    /// work to do live.
    ///
    /// Feature count is the proxy, not vertex count: it is `O(1)` off the
    /// feature store, it is monotone in the real cost, and the threshold only
    /// has to separate "a few thousand shapes, imperceptible" from "a national
    /// dataset, hundreds of milliseconds per frame".
    fn style_preview_is_too_expensive(&self, layer: LayerId) -> bool {
        self.local
            .feature_set(layer)
            .is_some_and(|features| features.features.len() > STYLE_PREVIEW_MAX_FEATURES)
    }

    /// Queues the restyle a drag deferred, once the pointer has lifted.
    ///
    /// The layer may have been removed, replaced by a hydrate or deselected in
    /// the meantime; all three simply drop the deferral, since
    /// `sync_local_style_for` resolves the style the project holds *now*.
    fn flush_deferred_restyle(&mut self, pointer_down: bool) {
        if pointer_down {
            return;
        }
        let Some(id) = self.deferred_restyle.take() else {
            return;
        };
        self.sync_local_style_for(id);
    }
    /// Validates `config` and, when usable, makes it both the basemap the UI
    /// reports and the pending swap a shell consumes next frame. Backs
    /// [`LayerAction::SetBasemapPreset`] and [`LayerAction::SetXyzBasemap`].
    ///
    /// An explicit pick also **demotes** any promoted layer, and the demote
    /// and the service change travel in ONE recorded entry: without the
    /// demote, picking a preset while a layer draws as the basemap would
    /// change nothing on screen; without them being fused, one gesture would
    /// cost two Ctrl+Z presses. Editing v1.5 put the service itself on the
    /// stack (see the writer census in `docs/plans/editing-v15.md` D9), which
    /// is why there is no direct field write left in this method — the
    /// applier owns both of the service's homes.
    pub(super) fn apply_basemap(&mut self, config: BasemapConfig) {
        if config != self.basemap
            && let Err(error) = config.template()
        {
            // A refused config changes nothing at all — a promotion least of
            // all, and nothing is recorded: the user asked for a service that
            // cannot draw.
            self.status = Some(format!("Basemap not changed: {error}"));
            return;
        }
        // The demote rides in the same entry as the service, and the whole
        // step sits ABOVE any "already active" wording, or re-picking the
        // highlighted preset while a layer is promoted would do nothing at
        // all.
        let already_active = config == self.basemap;
        let template = config.url_template.clone();
        let demoted = self.project.basemap_layer.is_some();
        let transaction = self.apply_basemap_change(None, Some(config));
        self.status = Some(if !already_active {
            format!("Basemap set to {template}")
        } else if demoted {
            format!("The basemap service draws again: {template}")
        } else {
            format!("Basemap already {template}")
        });
        // Status first, then the record: `record_undo` APPENDS its eviction
        // sentence, so this order keeps the pick as the headline. The clause
        // rides EVERY recorded pick now, because it is now true of every
        // recorded pick.
        if let Some(transaction) = transaction {
            self.append_status(BASEMAP_UNDO_CLAUSE);
            self.record_undo(transaction);
        }
    }

    /// Applies a change of the promoted-basemap pointer and/or the basemap
    /// SERVICE through the choke point and hands back the transaction to
    /// record, or [`None`] when nothing changed or the choke point refused
    /// (reported on the status line, nothing recorded).
    ///
    /// `after_service` of [`None`] means "leave the service alone", which is
    /// what a plain promote/demote/swap passes and what keeps that shape's
    /// payload byte-identical to editing v1.4's.
    ///
    /// `before` for the service is taken from `self.basemap` — the service the
    /// map is actually DRAWING — and never from `self.project.basemap`. The two
    /// diverge for real: `apply_loaded_basemap` returns early when a loaded
    /// file has no `basemap` field, leaving `self.basemap` at the previous
    /// project's service, so recording the serialized field would record
    /// "nothing" and an undo would install the default instead of what was on
    /// screen.
    ///
    /// Recording is deliberately left to the caller: `record_undo` APPENDS
    /// its eviction sentence to the status line, so the gesture's own message
    /// has to land first — the order the Remove arm already follows.
    fn apply_basemap_change(
        &mut self,
        after_layer: Option<LayerId>,
        after_service: Option<BasemapConfig>,
    ) -> Option<ProjectTransaction> {
        let before = self.project.basemap_layer;
        let service = after_service
            .filter(|config| *config != self.basemap)
            .map(|config| {
                Box::new(BasemapServiceChange {
                    before: self.basemap.clone(),
                    after: config,
                })
            });
        if before == after_layer && service.is_none() {
            // No empty entries: an undo step that changes nothing is history
            // the user has to press Ctrl+Z through for no reason.
            return None;
        }
        let after = after_layer;
        let transaction = ProjectTransaction {
            label: "Set basemap",
            op: ProjectOp::SetBasemap {
                before,
                after,
                service,
            },
            coalesce: None,
        };
        match self.apply_project_transaction(&transaction) {
            Ok(()) => {
                self.undo.close_coalescing();
                Some(transaction)
            }
            Err(error) => {
                self.status = Some(format!("Basemap not changed: {error}"));
                None
            }
        }
    }

    /// What a landed basemap-pointer change says, decided by what the map now
    /// actually draws rather than by what was asked for — so promoting a
    /// hidden layer says the service is still drawing instead of claiming a
    /// change the user cannot see.
    fn basemap_change_status(&self, target: Option<LayerId>) -> String {
        let Some(id) = target else {
            return format!("The basemap service draws again. {BASEMAP_UNDO_CLAUSE}");
        };
        let name = self
            .project
            .layers
            .get(id)
            .map_or("", |layer| layer.name.as_str());
        if self.draws_as_basemap(id) {
            format!(
                "\u{201c}{name}\u{201d} now draws as the basemap \u{2014} stack order and \
                 opacity no longer apply to it. {BASEMAP_UNDO_CLAUSE}"
            )
        } else {
            format!(
                "\u{201c}{name}\u{201d} is set as the basemap, but it is hidden, so the \
                 basemap service still draws \u{2014} show the layer to draw it. \
                 {BASEMAP_UNDO_CLAUSE}"
            )
        }
    }

    /// Applies one [`LayerAction`] reported by the layer panel this frame.
    ///
    /// Local GeoJSON layers keep a second, GPU-side copy of their visibility,
    /// opacity, order and existence (the render thread must not have to walk the
    /// project to draw a frame — see [`crate::local_vector::LocalVectorLayer`]),
    /// so every mutation here that touches one also queues the matching
    /// [`LocalLayerOp`] for the shell.
    pub(super) fn apply_layer_action(&mut self, action: LayerAction) {
        match action {
            LayerAction::Select(id) => {
                self.selection = Some(id);
                // A layer change is a gesture boundary: whatever was being
                // coalesced ended when the user looked somewhere else.
                self.undo.close_coalescing();
                if let Some(notice) = self.edit.retarget(Some(id)) {
                    self.push_edit_notice(notice);
                }
            }
            LayerAction::ZoomToLayer(id) => {
                // Read from the app-side feature store, never from the GPU
                // copy: reading that means holding the render lock, which a
                // panel gesture must not do. Nothing is recorded — the undo log
                // is for project state, and the camera is not it (`Project::view`
                // is stamped at save time by `sync_project_view`).
                //
                // The panel offers this row button for every LOCAL layer,
                // because a layer's kind is all it can see; whether the
                // collection actually arrived is known only here, so the
                // refusal — not the affordance — belongs to the gesture.
                let square = self
                    .local
                    .feature_set(id)
                    .map(|features| crate::local_vector::collection_square(features));
                match square {
                    Some(square) => self.zoom_to_square(square),
                    None => {
                        let name = self
                            .project
                            .layers
                            .get(id)
                            .map_or_else(|| "That layer".to_string(), |layer| layer.name.clone());
                        self.status = Some(format!(
                            "{name}: its features are not loaded, so there is no extent to zoom to."
                        ));
                    }
                }
            }
            LayerAction::ToggleVisibility(id) => {
                // Recorded now: `toggle_layer_visibility` goes through the
                // project choke point, so the flag, the GPU mirror, the dirty
                // stamp and the coalescing boundary are all one code path and
                // Ctrl+Z genuinely re-shows the layer.
                //
                // Doing it here rather than mutating the flag directly is what
                // makes the checkbox undoable: the old arm moved project state
                // WITHOUT recording, so a later Ctrl+Z undid the wrong step.
                self.toggle_layer_visibility(id);
            }
            LayerAction::SetOpacity(id, value) => {
                let before = self.project.layers.get(id).map(oxigis_core::Layer::opacity);
                if self.project.layers.set_opacity(id, value).is_ok()
                    && let Some(after) =
                        self.project.layers.get(id).map(oxigis_core::Layer::opacity)
                {
                    self.mark_project_dirty();
                    if self.is_local_layer(id) {
                        self.local.queue(LocalLayerOp::SetOpacity(id, after));
                    }
                    // Recorded coalesced: the slider emits every frame of a
                    // drag, and one drag must be one undo step.
                    if let Some(before) = before.filter(|before| *before != after) {
                        self.record_undo(crate::edit::project_op::ProjectTransaction {
                            label: "Change opacity",
                            op: crate::edit::project_op::ProjectOp::SetOpacity {
                                layer: id,
                                before,
                                after,
                            },
                            coalesce: Some(crate::edit::stack::CoalesceKey {
                                epoch: self.undo.epoch(),
                                layer: id,
                                feature: 0,
                                field: crate::edit::stack::CoalesceField::Opacity,
                            }),
                        });
                    }
                }
            }
            LayerAction::MoveUp(id) => {
                let before: Vec<LayerId> = self.layer_order();
                if self.project.layers.move_up(id) == Ok(true) {
                    self.queue_local_reorder();
                    self.record_reorder(before);
                }
            }
            LayerAction::MoveDown(id) => {
                let before: Vec<LayerId> = self.layer_order();
                if self.project.layers.move_down(id) == Ok(true) {
                    self.queue_local_reorder();
                    self.record_reorder(before);
                }
            }
            LayerAction::Remove(id) => {
                // The removal is itself an undo entry now: the snapshot —
                // layer value, storage slot, style entry, features Arc — is
                // what one Ctrl+Z restores. Recorder and applier are the same
                // code (`apply_project_transaction`), so the effects are
                // exactly the panel's old ones minus the pruning the recorded
                // family makes unnecessary: under strict LIFO an older feature
                // entry for this layer can only replay after the removal
                // above it has been undone, which puts the exact Arc back.
                let Some(snapshot) = self.layer_snapshot(id) else {
                    return;
                };
                let name = snapshot.layer.name.clone();
                let had_features = snapshot.features.is_some();
                // Captured BEFORE the removal, so the status can say whether
                // the map actually stops drawing something.
                //
                // The STACK is compared too, not only the two single-slot
                // derivations: removing a raster layer that was not the
                // top-most one leaves `desired_raster`/`desired_vector`
                // unchanged while genuinely taking a layer off the map, and the
                // status would then claim the layer was not drawn.
                let drawn_before = (
                    self.desired_raster(),
                    self.desired_vector(),
                    self.desired_tile_stack(),
                );
                let transaction = crate::edit::project_op::ProjectTransaction {
                    label: "Remove layer",
                    op: crate::edit::project_op::ProjectOp::RemoveLayers(vec![snapshot]),
                    coalesce: None,
                };
                match self.apply_project_transaction(&transaction) {
                    Ok(()) => {
                        self.undo.close_coalescing();
                        let was_drawn = drawn_before
                            != (
                                self.desired_raster(),
                                self.desired_vector(),
                                self.desired_tile_stack(),
                            );
                        // Status first: `record_undo` APPENDS its eviction
                        // sentence, so the order keeps both visible. Three
                        // shapes, each telling the truth about THIS layer —
                        // the old line promised "its features" for provider
                        // layers that have none.
                        self.status = Some(if had_features {
                            format!(
                                "Removed \u{201c}{name}\u{201d} \u{2014} Ctrl+Z puts it back \
                                 with its style, its position and its features."
                            )
                        } else if was_drawn {
                            format!(
                                "Removed \u{201c}{name}\u{201d} \u{2014} the map stops drawing \
                                 it; Ctrl+Z puts it back."
                            )
                        } else {
                            format!(
                                "Removed \u{201c}{name}\u{201d} \u{2014} Ctrl+Z puts it back \
                                 with its style and its position."
                            )
                        });
                        self.record_undo(transaction);
                    }
                    Err(error) => {
                        self.status = Some(format!("Remove failed: {error}"));
                    }
                }
            }
            LayerAction::RetryRefusedInstalls => {
                // A command, not an edit: no project state moves, so nothing
                // is recorded and the coalescing window is left open (an
                // interrupted opacity drag still folds). What actually
                // happens is decided by the next settle, which is why the
                // wording promises a retry rather than a result.
                //
                // The stack's refused entries live in the GPU state, which only
                // the shell can reach, so the click ALSO raises a flag a stack
                // shell drains through `take_tile_layer_retry`. Without it the
                // banner would carry a button that cannot dismiss the half of
                // the message the stack contributed.
                //
                // `retry_refused_installs` judges "was anything visible?" by
                // `provider_refusal`, which now includes the stack's half, so a
                // refused stack entry alone still reports a retry rather than
                // "Nothing to retry".
                let cleared = self.retry_refused_installs();
                self.pending_tile_layer_retry = true;
                self.status = Some(if cleared {
                    "Retrying the map's tile sources \u{2014} the result is reported here."
                        .to_string()
                } else {
                    "Nothing to retry.".to_string()
                });
            }
            LayerAction::AddGeoJsonPaste => {
                self.io_dialog = Some(IoDialog::PasteGeoJson {
                    name: "Pasted GeoJSON".to_string(),
                    buffer: String::new(),
                    error: None,
                });
            }
            LayerAction::AddDemoXyzLayer => {
                let id = layer_panel::add_demo_xyz_layer(&mut self.project.layers);
                self.selection = Some(id);
                self.mark_project_dirty();
                // The v1.3 "a list entry only" sentence is retired: an XYZ
                // layer is no longer inert — its row toggle promotes it to
                // the drawn basemap.
                self.status = Some(
                    "Added a demo XYZ layer \u{2014} the globe toggle on its row draws it as \
                     the basemap."
                        .to_string(),
                );
                self.record_layer_add(&[id]);
            }
            LayerAction::SetBasemapLayer(target) => {
                // Promotability is checked HERE and nowhere else: it is
                // derived state, so a refusal belongs to the gesture that
                // asked, not to the applier an undo also goes through.
                if let Some(id) = target
                    && let Some(reason) = self.promotion_refusal(id)
                {
                    self.status = Some(format!("Basemap not changed: {reason}"));
                    return;
                }
                if let Some(transaction) = self.apply_basemap_change(target, None) {
                    // Status first, then the record, so the eviction sentence
                    // `record_undo` appends lands after the headline.
                    let status = self.basemap_change_status(target);
                    self.status = Some(status);
                    self.record_undo(transaction);
                }
            }
            LayerAction::SetBasemapPreset(config) => self.apply_basemap(config),
            LayerAction::SetXyzBasemap(url) => {
                // Credit the serving host: the OSM attribution must not outlive
                // the OSM basemap, and the host is the only honest generic line.
                // (The presets above carry their services' exact required
                // credits instead.)
                let host = url
                    .split_once("://")
                    .map_or(url.as_str(), |(_scheme, rest)| rest)
                    .split(['/', '?'])
                    .next()
                    .unwrap_or_default()
                    .to_string();
                // A template with no derivable host — a relative
                // `/tiles/{z}/{x}/{y}.png`, say — has nobody to credit: an
                // empty attribution hides the overlay (the documented
                // contract), where "© " alone would draw a credit to no one.
                let attribution = if host.is_empty() {
                    String::new()
                } else {
                    format!("© {host}")
                };
                self.apply_basemap(BasemapConfig {
                    url_template: url,
                    subdomains: Vec::new(),
                    attribution,
                });
            }
            LayerAction::AddCogLayer(url) => {
                // The layer entry is the whole gesture: the provider install
                // and the credit line both DERIVE from the project now (see
                // `app/providers.rs`), so there is nothing to queue.
                let id = layer_panel::add_cog_layer(&mut self.project.layers, &url);
                self.selection = Some(id);
                self.mark_project_dirty();
                self.status = Some(format!("Reading COG {url}"));
                self.record_layer_add(&[id]);
            }
            LayerAction::AddArchiveUrlLayer(url) => {
                // Probe-then-create: the layer's KIND is in the archive's
                // header, so nothing is added to the stack until it lands.
                // BOTH formats are probed — a remote `.mbtiles` is surveyed in
                // one 16 KiB read (tiles v1.4) — and its own bytes are what
                // refuse it, at survey time, if anything does.
                let format = super::archive_io::format_for_url(&url);
                let _accepted =
                    self.request_archive_probe(oxigis_core::ArchiveRef::Url { url }, format);
            }
            LayerAction::OpenArchiveFile => {
                // A file dialog is a platform capability, like every other one
                // in this crate: the app only records that the user asked.
                self.request_archive_pick();
            }
            LayerAction::AddVectorTileLayer(url) => {
                // `config_for` is the ONE rule the derivation also uses, so
                // the stored paints and the drawn config cannot drift.
                let config = crate::vector_provider::config_for(
                    &url,
                    crate::vector_provider::maplibre_demo_paints(),
                );
                let id = layer_panel::add_vector_tile_layer(
                    &mut self.project.layers,
                    &config.url_template,
                    config.paints.clone(),
                );
                self.selection = Some(id);
                self.mark_project_dirty();
                self.status = Some(format!("Reading vector tiles {}", config.url_template));
                self.record_layer_add(&[id]);
            }
        }
    }
    /// Queues the local stack's draw order to match the project's, unless there
    /// is no local layer to reorder.
    ///
    /// Storage order, not the panel's reversed display order: the local renderer
    /// paints in list order, exactly as [`oxigis_core::LayerStack`] stores.
    /// The stack's current storage-order ids.
    pub(super) fn layer_order(&self) -> Vec<LayerId> {
        self.project
            .layers
            .layers()
            .iter()
            .map(|layer| layer.id)
            .collect()
    }

    /// Records a completed reorder against the order captured before it.
    fn record_reorder(&mut self, before: Vec<LayerId>) {
        let after = self.layer_order();
        if before == after {
            return;
        }
        self.undo.close_coalescing();
        self.record_undo(crate::edit::project_op::ProjectTransaction {
            label: "Reorder layers",
            op: crate::edit::project_op::ProjectOp::Reorder { before, after },
            coalesce: None,
        });
    }

    pub(super) fn queue_local_reorder(&mut self) {
        let order = local_input::local_layer_order(&self.project);
        if order.len() > 1 {
            self.local.queue(LocalLayerOp::Reorder(order));
        }
    }
    /// Applies one [`StyleAction`] reported by the style panel this frame.
    ///
    /// Refuses outright for a layer whose drawing `project.styles` does not
    /// decide — an MVT, tile-archive or COG layer paints from its own source
    /// rules. The panel already declines to draw the editor for those (see
    /// `OxigisApp::ui`); the guard is repeated here so the no-silent-write
    /// property belongs to the applier rather than to one caller's discipline.
    pub(super) fn apply_style_action(&mut self, action: StyleAction) {
        let Some(id) = self.selection else { return };
        if !self.is_local_layer(id) {
            return;
        }
        match action {
            StyleAction::Create(kind) => {
                self.project.styles.insert(id, kind.default_style().into());
            }
            StyleAction::Remove => {
                self.project.styles.remove(&id);
            }
            StyleAction::CreateFamily(family) => {
                // Seeded from a CLONE of the base: opting in must not
                // change the picture — the user edits from a known state.
                if let Some(set) = self.project.styles.get_mut(&id) {
                    let base = set.base().clone();
                    set.set_override(family, base);
                }
            }
            StyleAction::SetFamilyKind(family, kind) => {
                if let Some(set) = self.project.styles.get_mut(&id) {
                    set.set_override(family, kind.default_style());
                }
            }
            StyleAction::RemoveFamily(family) => {
                if let Some(set) = self.project.styles.get_mut(&id) {
                    set.clear_override(family);
                }
            }
        }
    }
    /// Applies one [`ProcessingAction`] reported by the Processing window
    /// this frame.
    ///
    /// A Run no longer *is* the run: it starts one and returns, so the frame
    /// that clicked the button finishes and paints. Everything that used to
    /// follow inline — the executor, the routing, the new layer — now happens
    /// on whichever later frame [`Self::poll_processing_job`] sees the run
    /// finish. Only a refusal that costs no work (an unresolvable layer, a
    /// malformed parameter) is still reported immediately, because it is
    /// known before any feature is touched.
    fn apply_processing_action(&mut self, action: ProcessingAction) {
        match action {
            ProcessingAction::Run {
                descriptor,
                params,
                output,
                selected_only,
            } => {
                // Captured before the run: a result layer named after the
                // layer it came from is the difference between three legible
                // rows and three rows all called "Simplify
                // (Douglas-Peucker) result".
                let source = self.tool_source_layer(&descriptor, &params);
                match self.start_processing_run(&descriptor, params, selected_only) {
                    Ok(run) => {
                        let total = run.progress().total;
                        let scope = if selected_only {
                            " over the selected features"
                        } else {
                            ""
                        };
                        self.status = Some(format!(
                            "Running {}{scope} \u{2014} {total} features. Cancel in the \
                             Processing window.",
                            descriptor.title
                        ));
                        self.processing
                            .begin_job(ProcessingJob::new(descriptor, source, output, run));
                    }
                    Err(message) => {
                        self.route_processing_result_to(&descriptor, source, &output, Err(message));
                    }
                }
            }
            ProcessingAction::Cancel => {
                if let Some(title) = self.processing.cancel_job() {
                    let message = format!("{title}: cancelled \u{2014} nothing was added.");
                    self.status = Some(message.clone());
                    self.processing.set_result(message);
                }
            }
        }
    }

    /// Drives the Processing run in flight, if there is one, and routes its
    /// result when it lands.
    ///
    /// Called every frame from [`Self::processing_window`] — **before** that
    /// method's own visibility check, deliberately: a run outlives the window
    /// it was started from, and a user who closes the toolbox while a
    /// simplify is working must still get their layer (and must still be able
    /// to reopen the window and find the result there).
    ///
    /// The repaint request is what makes a native run land at all: its worker
    /// thread finishes with no input event behind it, so on an idle desktop
    /// nothing would repaint — and nothing would poll — until the user moved
    /// the mouse.
    fn poll_processing_job(&mut self, ctx: &Context) {
        let frame_seconds = ctx.input(|input| input.stable_dt);
        match self.processing.poll_job(frame_seconds) {
            ProcessingPoll::Idle => {}
            ProcessingPoll::Running => ctx.request_repaint(),
            ProcessingPoll::Finished(finished) => {
                let finished = *finished;
                self.route_processing_result_to(
                    &finished.descriptor,
                    finished.source,
                    &finished.output,
                    finished.result,
                );
                ctx.request_repaint();
            }
            ProcessingPoll::Cancelled(title) => {
                self.status = Some(format!("{title}: cancelled \u{2014} nothing was added."));
                ctx.request_repaint();
            }
        }
    }

    /// The layer a run's [`oxigis_core::ParamKind::LayerRef`] parameter names,
    /// when it names one that is in the project.
    ///
    /// Same lookup [`Self::run_processing_tool`] performs and deliberately not
    /// shared with it: this one is for *naming* the output and must stay total
    /// — a descriptor with no layer parameter, or an id that no longer
    /// resolves, simply yields no source rather than an error.
    fn tool_source_layer(
        &self,
        descriptor: &oxigis_core::ToolDescriptor,
        params: &BTreeMap<String, serde_json::Value>,
    ) -> Option<LayerId> {
        let param = descriptor
            .params
            .iter()
            .find(|param| matches!(param.kind, oxigis_core::ParamKind::LayerRef))?;
        let id: LayerId = serde_json::from_value(params.get(&param.name)?.clone()).ok()?;
        self.project.layers.get(id).map(|layer| layer.id)
    }

    /// A distinct, legible name for a tool run's output layer.
    ///
    /// `"{source} \u{2014} {tool}"` when the run had a source layer, else the
    /// old `"{tool} result"`; either way de-duplicated against the names
    /// already in the stack, because tuning a tolerance means running the same
    /// tool over the same layer several times and the rows have to be tellable
    /// apart. The suffix search is bounded by the stack size, so it always
    /// terminates.
    fn result_layer_name(
        &self,
        descriptor: &oxigis_core::ToolDescriptor,
        source: Option<LayerId>,
    ) -> String {
        let base = match source.and_then(|id| self.project.layers.get(id)) {
            Some(layer) => format!("{} \u{2014} {}", layer.name, descriptor.title),
            None => format!("{} result", descriptor.title),
        };
        self.unique_layer_name(base)
    }

    /// `base`, or the first `"{base} (n)"` no layer in the stack already
    /// answers to.
    ///
    /// Split out of [`Self::result_layer_name`] so a name the user typed in the
    /// Output group is de-duplicated by the same rule the derived one is:
    /// running the same tool over the same layer twice under one chosen name
    /// must still produce two tellable-apart rows, not two identical ones. The
    /// suffix search is bounded by the stack size, so it always terminates.
    fn unique_layer_name(&self, base: String) -> String {
        let taken = |candidate: &str| {
            self.project
                .layers
                .layers()
                .iter()
                .any(|layer| layer.name == candidate)
        };
        if !taken(&base) {
            return base;
        }
        // At most `len + 1` candidates can collide with `len` layers, so one of
        // these is always free.
        for suffix in 2..=self.project.layers.len().saturating_add(2) {
            let candidate = format!("{base} ({suffix})");
            if !taken(&candidate) {
                return candidate;
            }
        }
        base
    }
    /// Routes a run whose source layer is not known, so the output keeps the
    /// historical `"{tool} result"` name — the arity the test modules drive
    /// hand-built results through.
    #[cfg(test)]
    pub(super) fn route_processing_result(
        &mut self,
        descriptor: &oxigis_core::ToolDescriptor,
        result: Result<serde_json::Value, String>,
    ) {
        self.route_processing_result_from(descriptor, None, result);
    }

    /// [`Self::route_processing_result_to`] with the Output group left at its
    /// defaults — a derived name, added as a layer.
    ///
    /// The arity the test modules drive hand-built results through
    /// (`app/tests.rs`, `app/tests_session.rs`, `app/edit_tests_review.rs`), so
    /// they keep asserting the *default* routing rather than a chosen one.
    #[cfg(test)]
    pub(super) fn route_processing_result_from(
        &mut self,
        descriptor: &oxigis_core::ToolDescriptor,
        source: Option<LayerId>,
        result: Result<serde_json::Value, String>,
    ) {
        self.route_processing_result_to(descriptor, source, &OutputTarget::default(), result);
    }

    /// Routes a Processing run's outcome (§1.5's result-routing rule):
    ///
    /// * a `FeatureCollection`-shaped value becomes a new layer through
    ///   [`Self::add_geojson_layer_from_value`] — the same path a pasted
    ///   GeoJSON document takes, so it is selected, zoomed to, and reported
    ///   on the status line for free. The `centroid` / `simplify` /
    ///   `convex_hull` tools take this branch; `bounds` and `feature_count`
    ///   return scalar JSON and take the next one. It is named after `output`
    ///   when the user named it and after `source` otherwise (see
    ///   [`Self::result_layer_name`]), and confirmed in the window whether it
    ///   landed or not — a run that reports nothing at all is
    ///   indistinguishable from a button that did nothing.
    /// * unless `output` asked for [`OutputDestination::GeoJsonText`], which
    ///   hands the document back to copy out instead. That is the answer to a
    ///   result being *embedded* in the project document: a tool result has no
    ///   file to reference, so a large one added as a layer is carried by every
    ///   save from then on. Handing it to [`IoDialog::ExportGeoJson`] is the
    ///   most this crate can do by itself — it compiles to `wasm32` and owns no
    ///   filesystem, exactly as File ▸ Save… does not.
    /// * anything else is pretty-printed into the window's read-only result
    ///   area, plus a short status-line summary.
    /// * an error is shown inline in the window; any prior result is
    ///   cleared either way, since this crate keeps one result/error slot,
    ///   not a history.
    ///
    /// Split out from [`Self::apply_processing_action`] so the routing rule
    /// can be exercised directly against a hand-built result, without a real
    /// [`oxigis_core::ToolExecutor`] wired up for it — there is no seam to
    /// inject a custom one into [`crate::processing_exec::builtin_executor`]
    /// by design (§1.5 non-goals: no user-authored tools).
    fn route_processing_result_to(
        &mut self,
        descriptor: &oxigis_core::ToolDescriptor,
        source: Option<LayerId>,
        output: &OutputTarget,
        result: Result<serde_json::Value, String>,
    ) {
        match result {
            Ok(value) => {
                let is_feature_collection = value.get("type").and_then(serde_json::Value::as_str)
                    == Some("FeatureCollection");
                if is_feature_collection {
                    // An empty collection is a *successful* run with nothing
                    // to draw — `centroid` skips every geometry-less feature
                    // by contract, so an attribute-only layer legitimately
                    // produces zero output. It must not reach
                    // `add_geojson_layer_from_text`: the parser there refuses
                    // an empty collection with "the GeoJSON holds no
                    // features", a message written to distinguish a *failed
                    // file drop*, and dressing this result up as that error
                    // would tell the user their tool run broke when it worked.
                    let counted = value
                        .get("features")
                        .and_then(serde_json::Value::as_array)
                        .map_or(0, Vec::len);
                    if counted == 0 {
                        let message = format!(
                            "{}: the tool ran, but no feature produced a result — no layer was \
                             created.",
                            descriptor.title
                        );
                        self.status = Some(message.clone());
                        self.processing.set_result(message);
                        return;
                    }
                    // A name the user typed wins over the derived one. An empty
                    // field means "not named", which is the historical default.
                    let name = if output.name.is_empty() {
                        self.result_layer_name(descriptor, source)
                    } else {
                        output.name.clone()
                    };
                    match output.destination {
                        OutputDestination::GeoJsonText => {
                            self.export_processing_result(&name, counted, &value);
                            return;
                        }
                        OutputDestination::File => {
                            self.save_processing_result(&name, counted, &value);
                            return;
                        }
                        OutputDestination::Layer => {}
                    }
                    // De-duplicated by the same rule the derived name already
                    // followed, so two runs under one chosen name still make two
                    // tellable-apart rows. Idempotent on a derived name, which
                    // is free of collisions by construction.
                    let name = self.unique_layer_name(name);
                    // By VALUE, not by text: the executor already built a
                    // `FeatureCollection` and serialised it to this `Value`,
                    // and serialising that to a `String` only so the string can
                    // be parsed straight back into a second collection put four
                    // representations of one result in memory at once. This
                    // seam consumes the value into the collection and encodes
                    // the inline copy from it, so at most two exist.
                    match self.add_geojson_layer_from_value(&name, value) {
                        Some(id) => {
                            self.record_layer_add(&[id]);
                            let features = self
                                .local
                                .feature_set(id)
                                .map_or(0, |set| set.features.len());
                            // The window cleared its result area when Run was
                            // clicked, so without this a successful run leaves
                            // it blank and reads as "the button did nothing".
                            self.processing.set_result(format!(
                                "Added \u{201c}{name}\u{201d} ({features} features)."
                            ));
                        }
                        None => {
                            // `add_geojson_layer_from_value` reports its refusal
                            // — a bad document or a result that would not
                            // re-encode — on the shared status line, which the
                            // next action overwrites; the window is where the
                            // user is looking, so the reason is repeated there.
                            let reason = self
                                .status
                                .clone()
                                .unwrap_or_else(|| format!("{name} could not be added as a layer"));
                            self.processing.set_error(reason);
                        }
                    }
                } else {
                    match serde_json::to_string_pretty(&value) {
                        Ok(pretty) => {
                            self.status = Some(scalar_result_status(&descriptor.title, &value));
                            self.processing.set_result(pretty);
                        }
                        Err(error) => self
                            .processing
                            .set_error(format!("could not encode the result: {error}")),
                    }
                }
            }
            Err(message) => self.processing.set_error(message),
        }
    }

    /// Hands a dataset result back as a GeoJSON document instead of adding it
    /// to the project — [`OutputDestination::GeoJsonText`].
    ///
    /// Nothing is added, nothing is recorded and the project is NOT dirtied:
    /// this route exists precisely so a large result does not become part of
    /// the saved document, and a gesture that changes nothing must not claim
    /// otherwise. The window keeps reporting the run, because a Run that
    /// silently opened a modal and left the result area blank reads as a button
    /// that did nothing.
    fn export_processing_result(&mut self, name: &str, features: usize, value: &serde_json::Value) {
        match serde_json::to_string_pretty(value) {
            Ok(content) => {
                let bytes = content.len();
                self.io_dialog = Some(IoDialog::ExportGeoJson {
                    name: name.to_string(),
                    content,
                });
                let message = format!(
                    "\u{201c}{name}\u{201d}: {features} features ({bytes} bytes) ready to copy \
                     out \u{2014} nothing was added to the project."
                );
                self.status = Some(message.clone());
                self.processing.set_result(message);
            }
            Err(error) => self
                .processing
                .set_error(format!("could not encode the result: {error}")),
        }
    }

    /// Hands a dataset result to the shell to write to disk —
    /// [`OutputDestination::File`].
    ///
    /// Nothing is added to the project and nothing is recorded, for the same
    /// reason [`Self::export_processing_result`] adds nothing: the point of
    /// this destination is that the result does *not* become part of the
    /// saved document. What is left behind is a
    /// [`ProcessingFileRequest`] for [`Self::take_pending_processing_save`],
    /// and a message that says exactly that — a build with no file writer
    /// attached must not be told its file was written.
    fn save_processing_result(&mut self, name: &str, features: usize, value: &serde_json::Value) {
        match serde_json::to_string_pretty(value) {
            Ok(content) => {
                let bytes = content.len();
                self.processing.queue_file_save(ProcessingFileRequest {
                    name: name.to_string(),
                    content,
                    features,
                });
                let message = format!(
                    "\u{201c}{name}\u{201d}: {features} features ({bytes} bytes) handed to this \
                     build's shell to write as {name}.geojson \u{2014} nothing was added to the \
                     project. A build with no file writer attached cannot perform it; use \
                     \u{201c}GeoJSON to copy out\u{201d} to save it yourself."
                );
                self.status = Some(message.clone());
                self.processing.set_result(message);
            }
            Err(error) => self
                .processing
                .set_error(format!("could not encode the result: {error}")),
        }
    }
    /// Draws the Processing ▸ Toolbox window when [`Self::show_processing`]
    /// is set — a persistent, closable window (not a one-shot [`IoDialog`]),
    /// since the user keeps it open across multiple tool runs and layer
    /// picks.
    pub(super) fn processing_window(&mut self, ctx: &Context) {
        // Before the visibility check, always: a run outlives the window that
        // started it — closing the toolbox mid-simplify must not strand the
        // job (nor, on the browser build, stop driving it).
        self.poll_processing_job(ctx);
        if !self.show_processing {
            return;
        }
        let selection = self.selection;
        let selected_features = self.selection_summary();
        let mut open = self.show_processing;
        let mut action = None;
        // Destructured, not borrowed through `self`: the layer options borrow
        // `project`, the panel borrows `processing_registry` and `processing`,
        // and those are disjoint fields. Going through `&self` methods instead
        // forced an owned copy of every layer NAME, rebuilt at 60 Hz for as
        // long as the window stayed open.
        let Self {
            project,
            local,
            processing_registry,
            processing,
            ..
        } = self;
        let layer_options = loaded_local_layer_options(project, local);
        egui::Window::new("Processing Toolbox")
            .collapsible(false)
            .open(&mut open)
            .show(ctx, |ui| {
                action = processing_panel::draw(
                    ui,
                    processing_registry,
                    processing,
                    &layer_options,
                    selection,
                    selected_features,
                );
            });
        self.show_processing = open;
        if let Some(action) = action {
            self.apply_processing_action(action);
            // The result/error area this just filled in only shows up in the
            // window on the *next* `.show()` call — unlike the
            // `FeatureCollection` branch, a scalar result queues no
            // `LocalLayerOp` for `Self::ui`'s own pending-work repaint check
            // to catch, so it needs its own nudge.
            ctx.request_repaint();
        }
    }
}
/// Local vector layers whose features are loaded, top-of-stack first — the ONE
/// rule behind both [`OxigisApp::local_vector_layer_options`] and the
/// Processing window's picker, taken over borrowed fields so the caller can
/// hold it beside a `&mut` borrow of an unrelated field.
pub(super) fn loaded_local_layer_options<'a>(
    project: &'a oxigis_core::Project,
    local: &crate::local_input::LocalInputState,
) -> Vec<(LayerId, &'a str)> {
    project
        .layers
        .layers()
        .iter()
        .rev()
        .filter(|layer| local_input::is_local_layer(layer) && local.feature_set(layer.id).is_some())
        .map(|layer| (layer.id, layer.name.as_str()))
        .collect()
}

/// A short status-line summary of a non-`FeatureCollection` Processing
/// result: the value itself for a bare scalar (e.g. `"Feature Count: 2"`),
/// so the status line actually reports the answer rather than just "it
/// worked" — the full value is always shown in the Processing window's
/// read-only result area regardless, so a structured value (object/array)
/// falls back to a generic "finished" here rather than dumping JSON into a
/// one-line status field.
fn scalar_result_status(title: &str, value: &serde_json::Value) -> String {
    let scalar = value
        .as_i64()
        .map(|number| number.to_string())
        .or_else(|| value.as_f64().map(|number| number.to_string()))
        .or_else(|| value.as_bool().map(|flag| flag.to_string()))
        .or_else(|| value.as_str().map(str::to_string));
    match scalar {
        Some(text) => format!("{title}: {text}"),
        None => format!("{title} finished."),
    }
}

#[cfg(test)]
mod tests;
