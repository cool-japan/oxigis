// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The project-family choke point (editing v1.1 stage 5 — see
//! docs/plans/editing-v11.md): applying [`ProjectTransaction`]s, in both
//! directions. The recorder and the applier are the same code, which is
//! what makes undo symmetry provable rather than a coincidence.
//!
//! Check-first: every operation validates before anything applies, so a
//! partial application is unreachable and no rollback machinery exists.
//! A restore needs no serializer at all — the snapshot's `Layer::kind`
//! already carries the inline GeoJSON.

use std::sync::Arc;

use super::OxigisApp;
use crate::edit::project_op::{LayerRename, LayerSnapshot, ProjectOp, ProjectTransaction};
use crate::layer_panel::LayerEdit;
use crate::local_input::LocalLayerOp;
use crate::tile_provider::BasemapConfig;
use crate::ui_glyphs::{EM_DASH, REMOVE};
use oxigis_core::LayerId;

/// What an add too large for the undo budget appends to the status line.
///
/// Built here rather than inline so the glyph constants stay interpolated and
/// the sentence is one line of source: the wrapped literal it replaces shipped
/// with a ~22-space run in the middle, because its line continuations lost
/// their trailing `\`.
pub(super) fn undoable_budget_notice() -> String {
    format!("It is too large to be undoable {EM_DASH} use the layer panel's {REMOVE} to remove it.")
}

impl OxigisApp {
    /// Applies one project-family transaction — layer add/remove, reorder,
    /// opacity, style — in whichever direction the caller built it.
    ///
    /// # Errors
    ///
    /// A human-readable refusal, with nothing applied.
    pub(super) fn apply_project_transaction(
        &mut self,
        transaction: &ProjectTransaction,
    ) -> Result<(), String> {
        let outcome = self.apply_project_transaction_inner(transaction);
        // The ONE place every project-family mutation lands, in both
        // directions — so the unsaved-changes flag is stamped here rather than
        // at each of the a dozen gestures that reach it. A refused transaction
        // changed nothing and must not dirty the project (check-first makes a
        // partial apply unreachable, so `Err` really means "nothing moved").
        if outcome.is_ok() {
            self.mark_project_dirty();
        }
        outcome
    }

    /// [`Self::apply_project_transaction`] minus the dirty stamp — the actual
    /// applier, split out so the stamp cannot be forgotten by a new arm.
    fn apply_project_transaction_inner(
        &mut self,
        transaction: &ProjectTransaction,
    ) -> Result<(), String> {
        match &transaction.op {
            ProjectOp::AddLayers(snapshots) => self.restore_layers(snapshots),
            ProjectOp::RemoveLayers(snapshots) => self.remove_layers_for_undo(snapshots),
            ProjectOp::Reorder { after, .. } => {
                self.apply_layer_order(after);
                Ok(())
            }
            ProjectOp::SetOpacity { layer, after, .. } => {
                self.project
                    .layers
                    .set_opacity(*layer, *after)
                    .map_err(|error| error.to_string())?;
                if self.is_local_layer(*layer) {
                    self.local.queue(LocalLayerOp::SetOpacity(*layer, *after));
                }
                Ok(())
            }
            ProjectOp::SetVisibility { layer, after, .. } => {
                // The ABSOLUTE side is written, never a flip: `set_visibility`
                // is idempotent, so replaying this transaction twice — which a
                // redo after a partially-applied frame really can do — lands on
                // the same state rather than inverting it. Check-first is the
                // stack's own `LayerNotFound`, mapped like `SetOpacity`'s.
                self.project
                    .layers
                    .set_visibility(*layer, *after)
                    .map_err(|error| error.to_string())?;
                // The render thread holds its own copy of the flag (see
                // `crate::local_vector::LocalVectorLayer`), so the mirror is
                // queued HERE — in the applier both directions go through —
                // rather than at the gesture, which is what makes an undo
                // actually re-show the layer on the map.
                if self.is_local_layer(*layer) {
                    self.local
                        .queue(LocalLayerOp::SetVisibility(*layer, *after));
                }
                Ok(())
            }
            ProjectOp::SetZoomRange { layer, after, .. } => {
                let (min, max) = *after;
                self.project
                    .layers
                    .set_zoom_range(*layer, min, max)
                    .map_err(|error| error.to_string())?;
                // Nothing is queued and nothing is rebuilt: what draws at the
                // current zoom is DERIVED from the project every frame (see
                // `app/providers.rs`), so a range change moves the map by
                // itself the way a hide already does.
                Ok(())
            }
            ProjectOp::RenameLayer { layer, names } => {
                self.project
                    .layers
                    .rename(*layer, names.after.clone())
                    .map_err(|error| error.to_string())?;
                // Deliberately no `LocalLayerOp`: no variant carries a name,
                // because the render side never draws one — the layer panel and
                // the attribute table read `project.layers` directly. A rename
                // is a pure project mutation, so an undo of one needs nothing
                // queued.
                Ok(())
            }
            ProjectOp::SetBasemap { after, service, .. } => {
                // EVERY check before ANY write, so a partial apply stays
                // unreachable. Existence only: promotability (XYZ kind, a
                // usable template) is DERIVED state and is deliberately not
                // checked here, or a hand-edited file naming a COG would make
                // an undo refuse. A pointer that cannot draw simply does not
                // resolve — see `app/providers.rs`.
                if let Some(layer) = after
                    && self.project.layers.get(*layer).is_none()
                {
                    return Err(format!("layer {layer} is no longer in the project"));
                }
                // The recorded SERVICE is deliberately not re-validated: the
                // recorder only ever records one `apply_basemap` already
                // accepted, ops are never persisted, and a service a shell
                // refuses raises the existing refusal banner with its Retry
                // button rather than a refusal at apply time.
                if let Some(change) = service {
                    self.set_basemap_service(&change.after);
                }
                self.set_basemap_layer(*after);
                Ok(())
            }
            ProjectOp::SetStyle { layer, after, .. } => {
                if self.project.layers.get(*layer).is_none() {
                    return Err(format!("layer {layer} is no longer in the project"));
                }
                match after {
                    Some(style) => {
                        self.project.styles.insert(*layer, (**style).clone());
                    }
                    None => {
                        self.project.styles.remove(layer);
                    }
                }
                self.sync_local_style_for(*layer);
                Ok(())
            }
        }
    }

    /// Puts a removed group back exactly, check-first: every id is verified
    /// absent before anything is applied, so a group cannot land half-way.
    ///
    /// Restoration walks the snapshots ascending by their recorded storage
    /// position (the order [`ProjectOp::AddLayers`] documents): each restore
    /// then finds a stack exactly one slot shorter than its own recorded slot
    /// needs, so the walk in [`Self::restore_layer`] lands every layer on its
    /// exact slot.
    fn restore_layers(&mut self, snapshots: &[LayerSnapshot]) -> Result<(), String> {
        for snapshot in snapshots {
            let id = snapshot.layer.id;
            if self.project.layers.get(id).is_some() {
                return Err(format!("layer {id} is already in the project"));
            }
        }
        let mut ascending: Vec<&LayerSnapshot> = snapshots.iter().collect();
        ascending.sort_by_key(|snapshot| snapshot.position);
        for snapshot in ascending {
            self.restore_layer(snapshot)?;
        }
        Ok(())
    }

    /// Removes a recorded group, check-first, walking descending by recorded
    /// position — the feature family's descending-`Remove` rule.
    fn remove_layers_for_undo(&mut self, snapshots: &[LayerSnapshot]) -> Result<(), String> {
        for snapshot in snapshots {
            let id = snapshot.layer.id;
            if self.project.layers.get(id).is_none() {
                return Err(format!("layer {id} is no longer in the project"));
            }
        }
        let mut descending: Vec<&LayerSnapshot> = snapshots.iter().collect();
        descending.sort_by_key(|snapshot| core::cmp::Reverse(snapshot.position));
        for snapshot in descending {
            self.remove_layer_for_undo(snapshot.layer.id)?;
        }
        Ok(())
    }

    /// Puts a removed layer back exactly: value, stack position, style
    /// entry, features and the GPU copy.
    fn restore_layer(&mut self, snapshot: &LayerSnapshot) -> Result<(), String> {
        let id = snapshot.layer.id;
        if self.project.layers.get(id).is_some() {
            return Err(format!("layer {id} is already in the project"));
        }
        // Style first: the GPU `Add` the feature restore queues resolves its
        // style from the project.
        if let Some(style) = &snapshot.style {
            self.project.styles.insert(id, style.clone());
        }
        self.project.layers.add(snapshot.layer.clone());
        // Walk down from the top to the recorded storage slot — the same
        // remove→add→walk shape `rewrite_layer_source` uses, because
        // `LayerStack` deliberately exposes no positional insert.
        let top = self.project.layers.len().saturating_sub(1);
        for _ in snapshot.position..top {
            if self.project.layers.move_down(id) != Ok(true) {
                break;
            }
        }
        if let Some(default) = &snapshot.default_style {
            self.local.remember_default_style(id, default.clone());
        }
        if let Some(features) = &snapshot.features {
            self.local
                .replace_features(&self.project, id, Arc::clone(features))
                .map_err(|error| error.message().to_string())?;
            self.queue_local_reorder();
        }
        Ok(())
    }

    /// Removes a layer as the redo of a recorded removal (or the undo of a
    /// recorded add) — the same effects the panel's Remove has, minus the
    /// pruning the recorded family makes unnecessary.
    fn remove_layer_for_undo(&mut self, id: LayerId) -> Result<(), String> {
        if self.project.layers.get(id).is_none() {
            return Err(format!("layer {id} is no longer in the project"));
        }
        let was_local = self.is_local_layer(id);
        let _ = self.project.layers.remove(id);
        self.project.styles.remove(&id);
        if was_local {
            self.local.forget(id);
            self.local.queue(LocalLayerOp::Remove(id));
        }
        self.edit.clear_issues(id);
        if self.selection == Some(id) {
            self.selection = None;
            let _ = self.edit.retarget(None);
        }
        Ok(())
    }

    /// The ONE production writer of [`oxigis_core::Project::basemap_layer`]
    /// (the other writer is serde, on a project load).
    ///
    /// Every user change reaches it through
    /// [`Self::apply_project_transaction`], which is what makes a promotion's
    /// undo symmetry provable rather than a coincidence — recorder and
    /// applier are the same code. Grep for `basemap_layer =` to enforce it.
    fn set_basemap_layer(&mut self, layer: Option<LayerId>) {
        self.project.basemap_layer = layer;
    }

    /// The ONE production writer of the basemap SERVICE — both of its two
    /// homes, always together.
    ///
    /// `self.basemap` is what every consumer derives from
    /// (`app/providers.rs`); `self.project.basemap` is what a save
    /// serializes. A writer that touches one and not the other is the
    /// divergence this method exists to make impossible. The only other
    /// writers are the project-replacement seam (`app/data_io.rs`), which
    /// `undo.reset()` precedes so no Ctrl+Z can jump over it, and serde on a
    /// load. Grep for `self.basemap =` and `project.basemap =` to enforce it.
    fn set_basemap_service(&mut self, service: &BasemapConfig) {
        self.project.basemap = Some(service.into());
        self.basemap = service.clone();
    }

    /// Applies a whole stack order (storage order, bottom-first):
    /// idempotent, unknown ids skipped, unlisted layers keep their relative
    /// order below the listed ones.
    fn apply_layer_order(&mut self, order: &[LayerId]) {
        for id in order {
            while self.project.layers.move_up(*id) == Ok(true) {}
        }
        self.queue_local_reorder();
    }

    /// Everything needed to put layer `id` back exactly as it stands right
    /// now — the ONE snapshot builder both the add recorder and the panel's
    /// Remove use, so the two cannot drift.
    pub(super) fn layer_snapshot(&self, id: LayerId) -> Option<LayerSnapshot> {
        let layer = self.project.layers.get(id).cloned()?;
        Some(LayerSnapshot {
            position: self
                .project
                .layers
                .layers()
                .iter()
                .position(|entry| entry.id == id)
                .unwrap_or_default(),
            style: self.project.styles.get(&id).cloned(),
            features: self.local.feature_set(id).cloned(),
            default_style: self.local.default_style(id).cloned(),
            layer,
        })
    }

    /// Records the addition of `ids` as ONE undo step, so a single Ctrl+Z
    /// removes exactly what one gesture added. Returns whether an entry was
    /// pushed.
    ///
    /// **Every** layer kind is recorded — local vector AND the provider
    /// layers (COG / vector-tile / XYZ). Editing v1.1 decision 9 refused the
    /// provider kinds because "undo would leave the provider drawing"; the
    /// v1.3 reconciliation (`app/providers.rs`) is precisely the removal of
    /// that objection: what is drawn derives from the project, so undoing an
    /// add genuinely un-draws it on the next frame. The seam records what
    /// already happened — the layers exist before the snapshot can — and
    /// exactness is bought by both undo directions going through
    /// [`Self::apply_project_transaction`] (which for a non-local layer is a
    /// pure project mutation; the provider follows from the derivation).
    ///
    /// Deliberately NOT called by the project-load rebuild, the `hydrate_*`
    /// family (they prune) or the choke point itself (an undo must not
    /// re-record).
    pub(super) fn record_layer_add(&mut self, ids: &[LayerId]) -> bool {
        let mut snapshots: Vec<LayerSnapshot> = ids
            .iter()
            .filter_map(|id| self.layer_snapshot(*id))
            .collect();
        if snapshots.is_empty() {
            return false;
        }
        // The group invariant `ProjectOp::AddLayers` documents: ascending by
        // recorded storage position.
        snapshots.sort_by_key(|snapshot| snapshot.position);
        let grouped = snapshots.len() > 1;
        let transaction = ProjectTransaction {
            label: if grouped { "Add layers" } else { "Add layer" },
            op: ProjectOp::AddLayers(snapshots),
            coalesce: None,
        };
        // An add must never destroy history: an entry above half the budget
        // would wipe essentially the whole log to keep itself, for a gesture
        // that is one visible ✖ click to reverse.
        if self.undo.would_dominate(transaction.estimated_bytes()) {
            self.append_status(&undoable_budget_notice());
            return false;
        }
        self.undo.close_coalescing();
        self.record_undo(transaction);
        self.append_status(if grouped {
            "One Ctrl+Z removes them all."
        } else {
            "Ctrl+Z removes it."
        });
        true
    }

    /// Flips a layer's visibility as ONE recorded undo step — the whole of
    /// [`crate::layer_panel::LayerAction::ToggleVisibility`]'s effect.
    ///
    /// Recorder and applier are the same code
    /// (`Self::apply_project_transaction`), which is what makes the toggle's
    /// undo symmetry provable rather than a coincidence: the GPU mirror is
    /// queued by the applier, so an undo genuinely re-shows the layer instead
    /// of only un-ticking a checkbox.
    ///
    /// Returns whether anything was recorded — `false` for a layer a hydrate
    /// removed, which is reported on the status line.
    pub fn toggle_layer_visibility(&mut self, layer: LayerId) -> bool {
        let Some(before) = self.project.layers.get(layer).map(|layer| layer.visible) else {
            return false;
        };
        let transaction = ProjectTransaction {
            label: "Toggle visibility",
            op: ProjectOp::SetVisibility {
                layer,
                before,
                after: !before,
            },
            coalesce: None,
        };
        match self.apply_project_transaction(&transaction) {
            Ok(()) => {
                // A checkbox is one discrete click, so whatever was being
                // coalesced (an opacity or style drag) ended when the user
                // reached for it.
                self.undo.close_coalescing();
                let name = self
                    .project
                    .layers
                    .get(layer)
                    .map_or_else(String::new, |layer| layer.name.clone());
                // Status first, then the record, so `record_undo`'s eviction
                // sentence lands after the headline.
                self.status = Some(format!(
                    "\u{201c}{name}\u{201d} is now {} \u{2014} Ctrl+Z undoes it.",
                    if before { "hidden" } else { "shown" }
                ));
                self.record_undo(transaction);
                true
            }
            Err(error) => {
                self.status = Some(format!("Visibility not changed: {error}"));
                false
            }
        }
    }

    /// Applies one [`crate::layer_panel::LayerEdit`] reported by a layer row's
    /// settings section — record-and-apply, through the choke point, so a
    /// rename and a scale-range change each cost exactly one Ctrl+Z.
    ///
    /// This is the seam a shell drives: the panel reports the edits, the caller
    /// loops them through here, and the *before* side is read here rather than
    /// in the panel because only the app can see the project the instant the
    /// edit lands (the panel drew a frame ago).
    ///
    /// Refusals are reported on the status line and record nothing, the shape
    /// every other project-family gesture follows.
    pub fn apply_layer_edit(&mut self, edit: LayerEdit) {
        let transaction = match edit {
            LayerEdit::Rename(id, after) => {
                let Some(before) = self.project.layers.get(id).map(|layer| layer.name.clone())
                else {
                    return;
                };
                // No empty entries: an undo step that changes nothing is
                // history the user has to press Ctrl+Z through for no reason.
                if before == after {
                    return;
                }
                ProjectTransaction {
                    label: "Rename layer",
                    op: ProjectOp::RenameLayer {
                        layer: id,
                        names: Box::new(LayerRename { before, after }),
                    },
                    coalesce: None,
                }
            }
            LayerEdit::SetZoomRange {
                layer,
                min_zoom,
                max_zoom,
            } => {
                let Some(before) = self
                    .project
                    .layers
                    .get(layer)
                    .map(|layer| (layer.min_zoom(), layer.max_zoom()))
                else {
                    return;
                };
                let after = (min_zoom, max_zoom);
                if before == after {
                    return;
                }
                ProjectTransaction {
                    label: "Set scale range",
                    // Deliberately NOT coalesced: the panel already reports one
                    // edit per finished gesture (a drag commits when the
                    // pointer lifts), so a coalescing key here would fold two
                    // deliberate, separate adjustments into one undo step.
                    op: ProjectOp::SetZoomRange {
                        layer,
                        before,
                        after,
                    },
                    coalesce: None,
                }
            }
        };
        let label = transaction.label;
        match self.apply_project_transaction(&transaction) {
            Ok(()) => {
                // A settings edit is one discrete commit, so whatever was being
                // coalesced (an opacity or style drag) ended when the user
                // reached for it — without this the edit would silently extend
                // that window and a later drag could fold across it.
                self.undo.close_coalescing();
                self.status = Some(self.layer_edit_status(&transaction.op));
                // Status first, then the record: `record_undo` APPENDS its
                // eviction sentence, so this order keeps the edit as the
                // headline — the order the Remove arm already follows.
                self.record_undo(transaction);
            }
            Err(error) => {
                self.status = Some(format!("{label} failed: {error}"));
            }
        }
    }

    /// What a landed [`crate::layer_panel::LayerEdit`] says, named after what
    /// actually changed rather than after what was asked for.
    fn layer_edit_status(&self, op: &ProjectOp) -> String {
        match op {
            ProjectOp::RenameLayer { names, .. } => format!(
                "Renamed \u{201c}{}\u{201d} to \u{201c}{}\u{201d} \u{2014} Ctrl+Z undoes it.",
                names.before, names.after
            ),
            ProjectOp::SetZoomRange { after, .. } => {
                let scale = match *after {
                    (None, None) => "at every zoom".to_string(),
                    (Some(min), None) => format!("from zoom {min}"),
                    (None, Some(max)) => format!("below zoom {max}"),
                    (Some(min), Some(max)) if min >= max => {
                        // An inverted range is stored, not swapped (see
                        // `Layer::set_zoom_range`) — so the line says what the
                        // user will actually see rather than reading the two
                        // bounds back as if they were a range.
                        "at no zoom at all".to_string()
                    }
                    (Some(min), Some(max)) => {
                        format!("from zoom {min} up to (not including) {max}")
                    }
                };
                format!("This layer now draws {scale} \u{2014} Ctrl+Z undoes it.")
            }
            // The method is private and reached only from the two arms above;
            // a future op that wants a sentence adds it there.
            _ => "Layer updated \u{2014} Ctrl+Z undoes it.".to_string(),
        }
    }

    /// Appends one sentence to the status line, keeping whatever the gesture
    /// already reported as the headline.
    pub(super) fn append_status(&mut self, sentence: &str) {
        self.status = Some(match self.status.take() {
            Some(existing) => format!("{existing} {sentence}"),
            None => sentence.to_string(),
        });
    }

    /// Queues the GPU restyle for `layer` from the project's current entry
    /// (or the layer's remembered default when the entry was removed).
    pub(super) fn sync_local_style_for(&mut self, layer: LayerId) {
        if !self.is_local_layer(layer) {
            return;
        }
        let style = self
            .project
            .styles
            .get(&layer)
            .cloned()
            .or_else(|| self.local.default_style(layer).cloned());
        if let Some(style) = style {
            self.local.queue(LocalLayerOp::SetStyle(layer, style));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::project_op::LayerRename;
    use oxigis_core::{Layer, LayerKind, RasterSource};

    /// Two points, so a local layer really has a collection behind it.
    const POINTS: &str = r#"{"type":"FeatureCollection","features":[
        {"type":"Feature","properties":{},
         "geometry":{"type":"Point","coordinates":[139.767,35.681]}},
        {"type":"Feature","properties":{},
         "geometry":{"type":"Point","coordinates":[135.502,34.702]}}]}"#;

    /// A provider layer — no local mirror, so the pure project half of an
    /// applier can be measured without the GPU queue in the way.
    fn xyz_layer(app: &mut OxigisApp, name: &str) -> LayerId {
        app.project.layers.add(Layer::new(
            name,
            LayerKind::Raster(RasterSource::xyz("https://tile.example/{z}/{x}/{y}.png")),
        ))
    }

    fn visibility(app: &OxigisApp, id: LayerId) -> Option<bool> {
        app.project.layers.get(id).map(|layer| layer.visible)
    }

    fn name_of(app: &OxigisApp, id: LayerId) -> Option<String> {
        app.project.layers.get(id).map(|layer| layer.name.clone())
    }

    fn hide(layer: LayerId) -> ProjectTransaction {
        ProjectTransaction {
            label: "Hide layer",
            op: ProjectOp::SetVisibility {
                layer,
                before: true,
                after: false,
            },
            coalesce: None,
        }
    }

    #[test]
    fn a_visibility_transaction_applies_both_ways_and_is_idempotent() {
        let mut app = OxigisApp::new();
        let id = xyz_layer(&mut app, "Roads");
        assert_eq!(visibility(&app, id), Some(true));

        let hidden = hide(id);
        app.apply_project_transaction(&hidden).expect("applies");
        assert_eq!(visibility(&app, id), Some(false));
        // Replaying the SAME side lands on the same state rather than
        // flipping — the property an absolute record buys.
        app.apply_project_transaction(&hidden).expect("applies");
        assert_eq!(visibility(&app, id), Some(false));

        // And the undo direction puts it back exactly.
        app.apply_project_transaction(&hidden.inverted())
            .expect("applies");
        assert_eq!(visibility(&app, id), Some(true));
    }

    #[test]
    fn a_visibility_transaction_dirties_the_project_and_refuses_a_gone_layer() {
        let mut app = OxigisApp::new();
        let id = xyz_layer(&mut app, "Roads");
        app.mark_saved();
        assert!(!app.has_unsaved_changes());
        app.apply_project_transaction(&hide(id)).expect("applies");
        assert!(
            app.has_unsaved_changes(),
            "the choke point stamps the dirty flag for every arm"
        );

        // A layer a hydrate replaced is gone; the transaction must refuse
        // rather than write nothing and report success.
        let _ = app.project.layers.remove(id);
        app.mark_saved();
        let error = app
            .apply_project_transaction(&hide(id))
            .expect_err("a gone layer is refused");
        assert!(error.contains(&id.to_string()), "{error}");
        assert!(
            !app.has_unsaved_changes(),
            "a refused transaction changed nothing and must not dirty the project"
        );
    }

    #[test]
    fn hiding_a_local_layer_reaches_the_render_thread_in_both_directions() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("Points", POINTS, None)
            .expect("valid GeoJSON");
        let _ = app.take_pending_local_ops();

        let hidden = hide(id);
        app.apply_project_transaction(&hidden).expect("applies");
        let ops = app.take_pending_local_ops();
        assert!(
            matches!(ops.as_slice(), [LocalLayerOp::SetVisibility(queued, false)] if *queued == id),
            "the applier must mirror the flag, or an undo leaves the map behind: {ops:?}"
        );

        app.apply_project_transaction(&hidden.inverted())
            .expect("applies");
        let ops = app.take_pending_local_ops();
        assert!(
            matches!(ops.as_slice(), [LocalLayerOp::SetVisibility(queued, true)] if *queued == id),
            "{ops:?}"
        );
    }

    #[test]
    fn a_rename_transaction_applies_both_ways_and_queues_nothing() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("Points", POINTS, None)
            .expect("valid GeoJSON");
        let before = name_of(&app, id).expect("present");
        let _ = app.take_pending_local_ops();

        let transaction = ProjectTransaction {
            label: "Rename layer",
            op: ProjectOp::RenameLayer {
                layer: id,
                names: Box::new(LayerRename {
                    before: before.clone(),
                    after: "Tokyo and Osaka".to_string(),
                }),
            },
            coalesce: None,
        };
        app.apply_project_transaction(&transaction)
            .expect("applies");
        assert_eq!(name_of(&app, id).as_deref(), Some("Tokyo and Osaka"));
        assert!(
            app.take_pending_local_ops().is_empty(),
            "no LocalLayerOp carries a name, so a rename queues nothing"
        );

        app.apply_project_transaction(&transaction.inverted())
            .expect("applies");
        assert_eq!(name_of(&app, id), Some(before));
    }

    #[test]
    fn a_scale_range_transaction_applies_both_ways_and_sanitizes_nothing_twice() {
        let mut app = OxigisApp::new();
        let id = xyz_layer(&mut app, "Cadastre");
        let transaction = ProjectTransaction {
            label: "Set scale range",
            op: ProjectOp::SetZoomRange {
                layer: id,
                before: (None, None),
                after: (Some(14.0), Some(18.0)),
            },
            coalesce: None,
        };
        app.apply_project_transaction(&transaction)
            .expect("applies");
        let layer = app.project.layers.get(id).expect("present");
        assert_eq!(layer.min_zoom(), Some(14.0));
        assert_eq!(layer.max_zoom(), Some(18.0));
        assert!(layer.visible_at(15.0) && !layer.visible_at(10.0));

        app.apply_project_transaction(&transaction.inverted())
            .expect("applies");
        let layer = app.project.layers.get(id).expect("present");
        assert_eq!(layer.min_zoom(), None);
        assert_eq!(layer.max_zoom(), None);
    }

    // ---- the two gesture seams --------------------------------------------

    #[test]
    fn toggling_visibility_is_one_recorded_step_that_ctrl_z_really_reverses() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("Points", POINTS, None)
            .expect("valid GeoJSON");
        let _ = app.take_pending_local_ops();
        let depth = app.undo.depth().0;

        assert!(app.toggle_layer_visibility(id));
        assert_eq!(visibility(&app, id), Some(false));
        assert_eq!(
            app.undo.depth().0,
            depth + 1,
            "a toggle is exactly ONE undo step"
        );
        let status = app.status.clone().unwrap_or_default();
        assert!(
            status.contains("hidden") && status.contains("Ctrl+Z"),
            "{status}"
        );
        assert!(
            matches!(
                app.take_pending_local_ops().as_slice(),
                [LocalLayerOp::SetVisibility(queued, false)] if *queued == id
            ),
            "the flag must reach the render thread"
        );

        // The point of recording it: Ctrl+Z genuinely re-shows the layer, on
        // the map as well as in the checkbox.
        assert!(app.undo_once(), "the entry is undoable");
        assert_eq!(visibility(&app, id), Some(true));
        assert!(
            matches!(
                app.take_pending_local_ops().as_slice(),
                [LocalLayerOp::SetVisibility(queued, true)] if *queued == id
            ),
            "an undo must un-hide it on the map too"
        );
    }

    #[test]
    fn toggling_a_gone_layer_records_nothing_and_says_so() {
        let mut app = OxigisApp::new();
        let id = xyz_layer(&mut app, "Roads");
        let _ = app.project.layers.remove(id);
        let depth = app.undo.depth().0;
        assert!(!app.toggle_layer_visibility(id));
        assert_eq!(app.undo.depth().0, depth, "nothing may be recorded");
    }

    #[test]
    fn a_panel_rename_is_recorded_once_and_a_no_op_one_is_not_recorded_at_all() {
        let mut app = OxigisApp::new();
        let id = xyz_layer(&mut app, "Roads");
        let depth = app.undo.depth().0;

        app.apply_layer_edit(LayerEdit::Rename(id, "Streets".to_string()));
        assert_eq!(name_of(&app, id).as_deref(), Some("Streets"));
        assert_eq!(app.undo.depth().0, depth + 1);
        let status = app.status.clone().unwrap_or_default();
        assert!(
            status.contains("Streets") && status.contains("Roads"),
            "{status}"
        );

        // Re-committing the same name is not history the user must press
        // Ctrl+Z through.
        app.apply_layer_edit(LayerEdit::Rename(id, "Streets".to_string()));
        assert_eq!(app.undo.depth().0, depth + 1, "no empty entries");

        assert!(app.undo_once());
        assert_eq!(name_of(&app, id).as_deref(), Some("Roads"));
    }

    #[test]
    fn a_panel_scale_range_edit_is_recorded_once_and_reverses_exactly() {
        let mut app = OxigisApp::new();
        let id = xyz_layer(&mut app, "Cadastre");
        let depth = app.undo.depth().0;

        app.apply_layer_edit(LayerEdit::SetZoomRange {
            layer: id,
            min_zoom: Some(14.0),
            max_zoom: Some(18.0),
        });
        let layer = app.project.layers.get(id).expect("present");
        assert_eq!(
            (layer.min_zoom(), layer.max_zoom()),
            (Some(14.0), Some(18.0))
        );
        assert_eq!(app.undo.depth().0, depth + 1);
        let status = app.status.clone().unwrap_or_default();
        assert!(status.contains("14") && status.contains("18"), "{status}");

        // Re-proposing the range it already has records nothing.
        app.apply_layer_edit(LayerEdit::SetZoomRange {
            layer: id,
            min_zoom: Some(14.0),
            max_zoom: Some(18.0),
        });
        assert_eq!(app.undo.depth().0, depth + 1, "no empty entries");

        // A second, deliberate adjustment is its OWN step: nothing coalesces
        // here, because the panel already reports one edit per finished
        // gesture.
        app.apply_layer_edit(LayerEdit::SetZoomRange {
            layer: id,
            min_zoom: Some(12.0),
            max_zoom: Some(18.0),
        });
        assert_eq!(app.undo.depth().0, depth + 2);

        assert!(app.undo_once());
        let layer = app.project.layers.get(id).expect("present");
        assert_eq!(
            (layer.min_zoom(), layer.max_zoom()),
            (Some(14.0), Some(18.0))
        );
        assert!(app.undo_once());
        let layer = app.project.layers.get(id).expect("present");
        assert_eq!((layer.min_zoom(), layer.max_zoom()), (None, None));
    }

    #[test]
    fn an_inverted_range_is_reported_honestly_rather_than_read_back_as_a_range() {
        let mut app = OxigisApp::new();
        let id = xyz_layer(&mut app, "Nothing");
        app.apply_layer_edit(LayerEdit::SetZoomRange {
            layer: id,
            min_zoom: Some(18.0),
            max_zoom: Some(14.0),
        });
        let status = app.status.clone().unwrap_or_default();
        assert!(
            status.contains("no zoom at all"),
            "an inverted range draws nowhere and the line must say so: {status}"
        );
        let layer = app.project.layers.get(id).expect("present");
        assert!(!layer.visible_at(16.0) && !layer.visible_at(13.0));
    }

    #[test]
    fn a_panel_edit_naming_a_gone_layer_records_nothing() {
        let mut app = OxigisApp::new();
        let id = xyz_layer(&mut app, "Roads");
        let _ = app.project.layers.remove(id);
        let depth = app.undo.depth().0;
        app.apply_layer_edit(LayerEdit::Rename(id, "Streets".to_string()));
        app.apply_layer_edit(LayerEdit::SetZoomRange {
            layer: id,
            min_zoom: Some(4.0),
            max_zoom: None,
        });
        assert_eq!(app.undo.depth().0, depth, "nothing may be recorded");
    }

    #[test]
    fn a_rename_of_a_gone_layer_is_refused_rather_than_silently_dropped() {
        let mut app = OxigisApp::new();
        let id = xyz_layer(&mut app, "Roads");
        let _ = app.project.layers.remove(id);
        let error = app
            .apply_project_transaction(&ProjectTransaction {
                label: "Rename layer",
                op: ProjectOp::RenameLayer {
                    layer: id,
                    names: Box::new(LayerRename {
                        before: "Roads".to_string(),
                        after: "Streets".to_string(),
                    }),
                },
                coalesce: None,
            })
            .expect_err("a gone layer is refused");
        assert!(error.contains(&id.to_string()), "{error}");
    }
}
