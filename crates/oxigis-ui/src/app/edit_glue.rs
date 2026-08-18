// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Editing half of [`OxigisApp`]: the single choke point through which all
//! feature data changes, the commit/undo/redo path built on it, and the
//! keyboard shortcuts.
//!
//! Split from `app/mod.rs` under the 2000-line rule, alongside `app::data_io`
//! (ingestion) and `app::dispatch` (panel actions). The pure model this drives
//! lives in [`crate::edit`].
//!
//! # The choke point
//!
//! [`OxigisApp::apply_feature_collection`] is the **only** path by which a local
//! layer's feature data changes. Every command, every undo and every redo goes
//! through it, so the four copies of the data — the app's feature store, the
//! GPU's `LocalVectorLayer`, the project's serialized source and the attribute
//! table's bound handle — cannot diverge.
//!
//! Its ordering is load-bearing: **serialize first**. A serialization failure
//! must abort before any state has moved, which is what makes the whole
//! operation atomic.

use super::OxigisApp;
use crate::edit::command::{self, EditError, EditTransaction};
use crate::edit::hit::{self, HitTarget};
use crate::edit::toolbar::{self, EditAction, ToolbarState};
use crate::edit::topology;
use crate::edit::{self as edit, EditCtx, EditMode, EditNotice, EditSelection, overlay, sketch};
use crate::local_input::{self, INLINE_GEOJSON_WARN_BYTES};
use crate::style_panel::StyleKind;
use crate::ui_glyphs::PATH_SEPARATOR;
use egui::{Context, CursorIcon, Key, KeyboardShortcut, Modifiers};
use oxigeo::geojson::types::{FeatureCollection, Geometry};
use oxigis_core::{LayerId, LayerKind, VectorSource};
use std::sync::Arc;

impl OxigisApp {
    /// Replaces a local layer's features with `features`, rewriting its stored
    /// source to hold the edited GeoJSON.
    ///
    /// See the module docs: this is the one place feature data changes.
    ///
    /// The layer's `VectorSource` becomes
    /// [`VectorSource::InlineGeoJson`] on the first edit and stays inline
    /// afterwards. That is the deal, and it is stated once in a status notice:
    /// there is no shapefile, GeoPackage or GeoParquet *writer* anywhere in the
    /// dependency graph and the browser has no filesystem, so writing an edit
    /// back to the original file is not out of scope — it is unimplementable
    /// here. Keeping the path reference and marking the layer dirty would
    /// instead guarantee silent, total data loss on the next project load,
    /// because the layer would simply be re-read from the untouched file.
    ///
    /// # Errors
    ///
    /// [`EditError::LayerGone`] when `id` is not (or is no longer) a local
    /// vector layer of the project, and [`EditError::Serialize`] when the
    /// collection cannot be written or adopted. Nothing has changed in either
    /// case.
    pub(crate) fn apply_feature_collection(
        &mut self,
        id: LayerId,
        features: Arc<FeatureCollection>,
    ) -> Result<(), EditError> {
        self.apply_feature_collection_with(id, features, |collection| {
            oxigeo::geojson::writer::to_string(collection).map_err(|error| error.to_string())
        })
    }

    /// [`Self::apply_feature_collection`] with the serializer supplied.
    ///
    /// The real entry point passes `oxigeo`'s writer. The parameter exists so a
    /// test can exercise the *ordering* contract — that a serialization failure
    /// leaves the project, the feature store, the GPU queue and the undo stack
    /// untouched — which is otherwise unreachable: `serde_json` writes a
    /// non-finite float as `null` rather than failing, and every other input
    /// this crate can build serializes. The same reasoning as
    /// [`crate::local_input::LocalInputState::with_path_support`]: a branch that
    /// cannot be reached cannot be trusted.
    pub(super) fn apply_feature_collection_with<S>(
        &mut self,
        id: LayerId,
        features: Arc<FeatureCollection>,
        serialize: S,
    ) -> Result<(), EditError>
    where
        S: FnOnce(&FeatureCollection) -> Result<String, String>,
    {
        // 1. The layer must still exist and still be local — this is what stops
        //    an undo entry for a removed layer from resurrecting GPU state.
        let layer = self
            .project
            .layers
            .get(id)
            .ok_or(EditError::LayerGone(id))?;
        if !local_input::is_local_layer(layer) {
            return Err(EditError::LayerGone(id));
        }
        let converted_from = match &layer.kind {
            LayerKind::Vector(VectorSource::InlineGeoJson { .. }) => None,
            LayerKind::Vector(source) => Some(source_display_name(source)),
            LayerKind::Raster(_) => None,
        };
        // What the layer's inline text weighed BEFORE this commit, so the size
        // warning below can fire on the commit that CROSSES the threshold and
        // stay quiet afterwards. Without it every single vertex drag on a layer
        // past the threshold allocates the sentence again, grows the notice log
        // and — worse — stomps the status line the gesture itself owns.
        let stored_bytes = match &layer.kind {
            LayerKind::Vector(VectorSource::InlineGeoJson { geojson }) => Some(geojson.len()),
            LayerKind::Vector(_) | LayerKind::Raster(_) => None,
        };

        // 2. Serialize FIRST. Nothing below this line may run if it fails.
        let geojson = serialize(&features).map_err(EditError::Serialize)?;
        let bytes = geojson.len();

        // 3. Adopt the edited data as the layer's own source.
        rewrite_layer_source(
            &mut self.project,
            id,
            VectorSource::InlineGeoJson { geojson },
        );

        // 4. Feature store and GPU queue together, inside `LocalInputState`, so
        //    "the store and the queued Add carry the same Arc" is an invariant
        //    owned by the type that holds both.
        self.local
            .replace_features(&self.project, id, features)
            .map_err(|error| EditError::Serialize(error.message().to_string()))?;

        let mut notice = converted_from.map(|name| {
            format!(
                "{name} is read-only — edits are stored inside the project file; the original \
                 file is not modified."
            )
        });
        // Stated once, on the crossing — see `stored_bytes`. A layer that
        // shrinks back under the threshold and grows past it again is told
        // again, which is the honest reading of "this just became large".
        if bytes >= INLINE_GEOJSON_WARN_BYTES
            && stored_bytes.is_none_or(|before| before < INLINE_GEOJSON_WARN_BYTES)
        {
            let size = format!(
                "This layer's edited GeoJSON is now {} KiB inside the project file.",
                bytes / 1024,
            );
            notice = Some(match notice {
                Some(existing) => format!("{existing} {size}"),
                None => size,
            });
        }
        if let Some(message) = notice {
            self.push_edit_notice(EditNotice::new(message));
        }
        Ok(())
    }

    /// Applies `transaction` to its layer and moves the selection to where the
    /// transaction says it should land.
    ///
    /// Used for both directions: an undo applies
    /// [`EditTransaction::inverted`], through the same choke point, so undo
    /// cannot produce a state a fresh edit could not.
    ///
    /// # Errors
    ///
    /// [`EditError::FeaturesNotLoaded`] when the layer's data has not been read
    /// yet, whatever [`command::apply_ops`] refuses, and whatever
    /// [`Self::apply_feature_collection`] refuses.
    pub(super) fn apply_transaction(
        &mut self,
        transaction: &EditTransaction,
    ) -> Result<(), EditError> {
        // Taken at the TOP: a refused commit must not leak the one-shot marks
        // latch into the next, unrelated commit.
        let armed_marks = self.edit.take_marks_after_commit();
        let current = self
            .local
            .feature_set(transaction.layer)
            .ok_or(EditError::FeaturesNotLoaded(transaction.layer))?;
        let next = Arc::new(command::apply_ops(current, &transaction.ops)?);
        self.apply_feature_collection(transaction.layer, Arc::clone(&next))?;
        // The collection was REPLACED: any live index-addressed gesture now
        // holds an `origin` clone of data that is no longer on screen, and
        // committing it later would write coordinates derived from the
        // pre-change geometry. Drop it here, after the apply succeeded (a
        // refused transaction changed nothing and the gesture stays live) —
        // provably a no-op for the drag's own commit, because
        // `finish_vertex_drag` takes the drag before building its
        // transaction. Silent: the gesture that caused the commit owns the
        // status line, and undo/redo word their own cancellation before
        // arriving here.
        if self.selection == Some(transaction.layer) {
            self.edit.cancel_live_gestures();
        }
        // Index-addressed state follows the renumbering before anything reads
        // it: an add or delete below a feature shifts every later index, and
        // both the recorded topology issues and the attribute form's binding
        // would otherwise keep naming whatever slid into their old slots.
        let form_was_dirty = self.edit.form().is_dirty();
        if !self
            .edit
            .form_mut()
            .remap_bound(transaction.layer, &transaction.ops)
            && form_was_dirty
        {
            self.push_edit_notice(EditNotice::new(
                "The feature the unsaved attribute edits belonged to was deleted — the edits \
                 were discarded.",
            ));
        }
        self.edit
            .remap_issue_indices(transaction.layer, &transaction.ops);
        // The edit selection addresses a feature inside the *selected* layer —
        // `EditSelection` carries no `LayerId` by design — so a transaction
        // replayed against some other layer (a global-stack undo, a stale
        // form's Apply) must not stamp its landing index onto whatever layer
        // is selected now.
        if self.selection == Some(transaction.layer) {
            self.edit.set_selection(transaction.selection_after);
            self.edit.validate_selection(Some(&next));
            // A vertex-set move re-asserts its marks over the collapsed
            // selection — sound because a translation renumbers nothing.
            // Guarded on the anchor still matching, so a transaction that
            // landed somewhere unexpected cannot stamp stale marks.
            if let Some(marks) = armed_marks
                && let Some(multi) = self.edit.multi_selection().cloned()
                && transaction
                    .selection_after
                    .is_some_and(|after| after.feature == multi.anchor())
            {
                self.edit
                    .set_multi_selection(Some(multi.with_vertex_set(marks)));
            }
        }
        // Last, and only after the data has actually landed: validation is a
        // report about committed state, never a gate in front of it.
        // POST-transaction indices: a removed feature has nothing to check
        // and features that merely shifted were remapped above — so a
        // 10 000-feature delete revalidates nothing instead of everything.
        self.revalidate(transaction.layer, &next, &transaction.touched_after());
        Ok(())
    }

    /// Re-checks the topology of the features a transaction touched, and
    /// records the result against `layer`.
    ///
    /// Bounded by construction: one transaction touches a handful of features,
    /// and each of those is capped at
    /// [`topology::MAX_SEGMENTS_FOR_SELF_INTERSECTION`] segments of pairwise
    /// work — so this is imperceptible on the commit path, which is the only
    /// reason it can run there at all. The whole-collection pass is the explicit
    /// **Validate layer** button, and validation never runs on load.
    ///
    /// It also never refuses anything: a half-drawn polygon is
    /// self-intersecting for most of its life, and an editor that blocked the
    /// commit would be unusable.
    pub(super) fn revalidate(
        &mut self,
        layer: LayerId,
        features: &Arc<FeatureCollection>,
        touched: &[usize],
    ) {
        let mut fresh = Vec::new();
        for &index in touched {
            let Some(feature) = features.features.get(index) else {
                continue;
            };
            fresh.extend(topology::validate_feature(index, feature));
        }
        self.edit
            .merge_issues(layer, touched, features.features.len(), fresh);
    }

    /// Runs the topology checks over the whole selected layer.
    ///
    /// Returns whether a run happened. The result **replaces** whatever the
    /// per-commit passes had recorded for that layer: a full pass has seen every
    /// feature, so anything it did not report is not there any more.
    pub(super) fn validate_active_layer(&mut self) -> bool {
        let Some(id) = self.selection else {
            self.status = Some("Select a layer before validating it.".to_string());
            return false;
        };
        let Some(features) = self.local.feature_set(id).map(Arc::clone) else {
            self.status = Some(EditError::FeaturesNotLoaded(id).to_string());
            return false;
        };
        let validation =
            topology::validate_collection(&features, topology::VALIDATE_LAYER_SEGMENT_BUDGET);
        let count = validation.issues.len();
        // Truncation is what the run *reported*, not inferred from a full
        // list: the list is capped by construction, so `count == MAX_NOTICES`
        // alone cannot tell "exactly that many, all shown" from "more exist".
        let capped = validation.truncated;
        self.edit.set_issues(id, validation.issues);
        self.status = Some(match (count, capped) {
            (0, _) => "Validation: no issues.".to_string(),
            (1, _) => "Validation: 1 issue.".to_string(),
            (count, false) => format!("Validation: {count} issues."),
            (count, true) => {
                format!("Validation: {count} issues shown — more exist; the list is capped.")
            }
        });
        true
    }

    /// Applies `transaction` and, if it landed, records it as the newest undo
    /// step. Returns whether it landed.
    ///
    /// The stack is only told about work that actually happened, so it can
    /// never claim a step is undoable when the state it describes was never
    /// reached. Note that the coalescing window is maintained by
    /// [`crate::edit::stack::EditStack::push`] itself, from the transaction's
    /// own key — closing it here would make coalescing structurally impossible.
    ///
    /// Public because it is the entry point every editing gesture — and a
    /// shell driving the app programmatically — reaches the choke point
    /// through; the transaction type itself is public for the same reason.
    pub fn commit_edit(&mut self, transaction: EditTransaction) -> bool {
        if let Err(error) = self.apply_transaction(&transaction) {
            self.status = Some(error.to_string());
            return false;
        }
        self.record_undo(transaction);
        true
    }

    /// Undoes one step, if there is one. Returns whether anything moved.
    ///
    /// On failure the cursor is put back exactly where it was with a matching
    /// `redo`, so the stack never lies about what is applied.
    pub(super) fn undo_once(&mut self) -> bool {
        let Some(entry) = self.undo.undo() else {
            // Nothing yielded, nothing cancelled: an undo that does nothing
            // must change nothing, the live gesture included.
            self.status = Some("Nothing to undo.".to_string());
            return false;
        };
        // Cancel BEFORE applying, unconditionally (whatever layer the entry
        // names): what Ctrl+Z does to a live gesture must not depend on which
        // layer the invisible top of the stack happens to name. The drag has
        // committed nothing, so cancelling destroys nothing. One prefix built
        // once and reused on success AND failure, so the two cannot disagree.
        let prefix = Self::gesture_prefix(self.edit.cancel_live_gestures());
        let label = entry.label();
        let mut restored_marks = 0;
        let applied = match entry.inverted() {
            crate::edit::stack::UndoEntry::Features(inverse) => {
                // The inverse of a set move is a set move (the diff is
                // symmetric), so undoing one lands with the same corners
                // still marked.
                restored_marks = self.arm_marks_for(&inverse);
                self.apply_transaction(&inverse)
                    .map_err(|error| error.to_string())
            }
            crate::edit::stack::UndoEntry::Project(inverse) => {
                self.apply_project_transaction(&inverse)
            }
        };
        match applied {
            Ok(()) => {
                self.status = Some(if restored_marks > 0 {
                    format!(
                        "{prefix}Undo: {label} \u{2014} {restored_marks} vertices still marked."
                    )
                } else {
                    format!("{prefix}Undo: {label}")
                });
                true
            }
            Err(error) => {
                let _restored = self.undo.redo();
                self.status = Some(format!("{prefix}Undo failed: {error}"));
                false
            }
        }
    }

    /// Arms the marks latch when `transaction` is a single `Replace` whose
    /// marked set is recoverable from its own two sides — a vertex-set MOVE
    /// (two or more positions moved) or the undo of a vertex-set DELETE (two
    /// or more positions restored) — so an undo or a redo of one lands with
    /// the same corners still marked. Returns how many marks were armed; `0`
    /// when neither applies, or when the transaction's layer is not the
    /// selected one and the latch's consumer would drop the marks anyway (so
    /// the status can never claim marks the user cannot see).
    ///
    /// The discriminator is the DIFF, never the label: an attribute Apply
    /// moves nothing, and a future rotate/scale set tool restores marks with
    /// no new code. [`command::moved_vertices`] answers every equal-shape
    /// pair, so [`command::restored_vertices`] is consulted only on an arity
    /// change and the two can never both contribute. These derived marks are
    /// valid by construction — they came from the very geometry pair being
    /// applied.
    fn arm_marks_for(&mut self, transaction: &EditTransaction) -> usize {
        if self.selection != Some(transaction.layer) || transaction.selection_after.is_none() {
            return 0;
        }
        let [command::FeatureOp::Replace { before, after, .. }] = transaction.ops.as_slice() else {
            return 0;
        };
        let (Some(before), Some(after)) = (before.geometry.as_ref(), after.geometry.as_ref())
        else {
            return 0;
        };
        let Some(marks) = command::moved_vertices(before, after)
            .or_else(|| command::restored_vertices(before, after))
        else {
            return 0;
        };
        if marks.len() < 2 {
            return 0;
        }
        let count = marks.len();
        self.edit.arm_marks_after_commit(marks);
        count
    }

    /// The status prefix a cancelled live gesture contributes to the undo or
    /// redo that cancelled it — empty when nothing was live.
    fn gesture_prefix(cancelled: Option<crate::edit::LiveGesture>) -> &'static str {
        match cancelled {
            Some(crate::edit::LiveGesture::VertexDrag) => "Drag cancelled \u{2014} ",
            Some(crate::edit::LiveGesture::BoxSelect) => "Box-select cancelled \u{2014} ",
            None => "",
        }
    }

    /// Redoes one step, if there is one. Returns whether anything moved.
    pub(super) fn redo_once(&mut self) -> bool {
        let Some(entry) = self.undo.redo() else {
            self.status = Some("Nothing to redo.".to_string());
            return false;
        };
        // Same rule as `undo_once`: an entry was yielded, so any live
        // index-addressed gesture is about to go stale — cancel it first and
        // say so in the same breath as the redo itself.
        let prefix = Self::gesture_prefix(self.edit.cancel_live_gestures());
        let label = entry.label();
        let mut restored_marks = 0;
        let applied = match &entry {
            crate::edit::stack::UndoEntry::Features(transaction) => {
                restored_marks = self.arm_marks_for(transaction);
                self.apply_transaction(transaction)
                    .map_err(|error| error.to_string())
            }
            crate::edit::stack::UndoEntry::Project(transaction) => {
                self.apply_project_transaction(transaction)
            }
        };
        match applied {
            Ok(()) => {
                self.status = Some(if restored_marks > 0 {
                    format!(
                        "{prefix}Redo: {label} \u{2014} {restored_marks} vertices still marked."
                    )
                } else {
                    format!("{prefix}Redo: {label}")
                });
                true
            }
            Err(error) => {
                let _restored = self.undo.undo();
                self.status = Some(format!("{prefix}Redo failed: {error}"));
                false
            }
        }
    }

    /// Records an edit notice and shows it on the status line.
    pub(super) fn push_edit_notice(&mut self, notice: EditNotice) {
        self.status = Some(notice.message().to_string());
        self.edit.push_notice(notice);
    }

    /// The one production path onto the undo stack: pushes and surfaces any
    /// byte-budget eviction the push cost.
    ///
    /// Entry-cap ageing is deliberately silent — in a long session every push
    /// past [`crate::edit::stack::UNDO_MAX_ENTRIES`] ages one entry out, and a
    /// notice per edit forever would be noise. A byte-budget eviction means
    /// history was traded to keep THIS step undoable, which the user should
    /// hear about — appended to the status line (the gesture's own message is
    /// the headline) and kept in the Edit window's notice log.
    pub(super) fn record_undo(&mut self, entry: impl Into<crate::edit::stack::UndoEntry>) {
        let report = self.undo.push(entry);
        if report.by_bytes == 0 {
            return;
        }
        let sentence = match report.by_bytes {
            1 => "Memory: 1 older undo step was dropped to keep this one undoable.".to_string(),
            n => format!("Memory: {n} older undo steps were dropped to keep this one undoable."),
        };
        self.status = Some(match self.status.take() {
            Some(existing) => format!("{existing} {sentence}"),
            None => sentence.clone(),
        });
        self.edit.push_notice(EditNotice::new(sentence));
    }

    /// Handles this frame's editing shortcuts.
    ///
    /// Two rules make this safe to call every frame:
    ///
    /// * **Focus guard.** egui's `TextEdit` owns `Ctrl+Z`, `Backspace`,
    ///   `Escape` and every character while it has focus, and the attribute form
    ///   is full of them — so a focused widget takes the whole frame's keys.
    ///   The guard also covers the frame *after* focus was held, because egui
    ///   0.35 clears the focused widget for a bare `Escape` in
    ///   `Focus::begin_pass` — before any app code runs — so on exactly the
    ///   frame a text field's Escape arrives, `memory.focused()` is already
    ///   [`None`]. That Escape was spent leaving the field; without the
    ///   last-frame memory it would *also* climb the edit cancel ladder and,
    ///   e.g., silently drop the feature selection out from under the form.
    /// * **`consume_shortcut`, not `key_pressed`.** A handled key must not also
    ///   reach anything else, and it returns `true` exactly *once* per frame, so
    ///   keyboard auto-repeat cannot queue several fully re-tessellated layers
    ///   inside one frame.
    ///
    /// [`Modifiers::COMMAND`] maps to Cmd on macOS and Ctrl everywhere else, so
    /// native and web behave identically with no `cfg`.
    pub(super) fn edit_shortcuts(&mut self, ctx: &Context) {
        let focused_now = ctx.memory(|memory| memory.focused()).is_some();
        let guarded = focused_now || self.edit_focus_last_frame;
        self.edit_focus_last_frame = focused_now;
        if guarded {
            return;
        }
        let redo = ctx.input_mut(|input| {
            input.consume_shortcut(&KeyboardShortcut::new(
                Modifiers::COMMAND.plus(Modifiers::SHIFT),
                Key::Z,
            )) || input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Y))
        });
        let undo = !redo
            && ctx.input_mut(|input| {
                input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Z))
            });
        if redo {
            self.redo_once();
            return;
        }
        if undo {
            self.undo_once();
            return;
        }
        if self.edit_clipboard_events(ctx) {
            return;
        }
        if self.edit_escape(ctx) {
            return;
        }
        if self.edit_sketch_keys(ctx) {
            return;
        }
        if self.edit_mode_keys(ctx) {
            return;
        }
        self.edit_delete_key(ctx);
    }

    /// `Enter` finishes the sketch, `Backspace` takes one vertex back.
    ///
    /// Both are claimed **only** while a drawing tool is active, so neither is
    /// taken away from anything else the app may grow; `Backspace` is claimed
    /// only while there is a vertex to drop, so an empty sketch leaves the key
    /// alone entirely. The focus guard at the top of [`Self::edit_shortcuts`] is
    /// what keeps them out of every text field, `Backspace` most of all.
    fn edit_sketch_keys(&mut self, ctx: &Context) -> bool {
        if !self.edit.mode().is_drawing() {
            return false;
        }
        if ctx.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Enter)) {
            self.finish_sketch();
            return true;
        }
        if self.edit.sketch().is_empty() {
            return false;
        }
        if ctx.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Backspace)) {
            let dropped = self.edit.sketch_mut().pop();
            let left = self.edit.sketch().len();
            self.status = Some(if dropped.is_some() {
                format!("Vertex removed \u{2014} {left} left in the sketch.")
            } else {
                "Nothing to take back.".to_string()
            });
            return true;
        }
        false
    }

    /// `Escape`: one rung of the cancel ladder, and only when there is a rung to
    /// climb down.
    ///
    /// Peeked before it is consumed, deliberately: `Escape` is a universally
    /// meaningful key and swallowing one that this system had no use for would
    /// break whatever else grows to want it.
    fn edit_escape(&mut self, ctx: &Context) -> bool {
        if !ctx.input(|input| input.key_pressed(Key::Escape)) {
            return false;
        }
        if !self.edit.escape() {
            return false;
        }
        ctx.input_mut(|input| {
            let _consumed = input.consume_key(Modifiers::NONE, Key::Escape);
        });
        self.undo.close_coalescing();
        self.status = Some(format!("Cancelled — {} tool.", self.edit.mode().label()));
        true
    }

    /// `B`/`V`/`P`/`L`/`G`: the five tools.
    ///
    /// Bare letters, so the focus guard above is what keeps them out of every
    /// text field in the app; `consume_key` then keeps a handled letter from
    /// also reaching anything else this frame.
    fn edit_mode_keys(&mut self, ctx: &Context) -> bool {
        let bindings = [
            (Key::B, EditMode::Off),
            (Key::V, EditMode::Select),
            (Key::P, EditMode::DrawPoint),
            (Key::L, EditMode::DrawLine),
            (Key::G, EditMode::DrawPolygon),
        ];
        for (key, mode) in bindings {
            if ctx.input_mut(|input| input.consume_key(Modifiers::NONE, key)) {
                self.apply_edit_action(EditAction::SetMode(mode));
                return true;
            }
        }
        false
    }

    /// `Delete`: the picked vertex if there is one, else the selected feature.
    ///
    /// Vertex first because that is the narrower target the user just aimed at;
    /// deleting the whole feature when they meant one of its corners is the more
    /// destructive of the two mistakes, and the only one that is not obvious on
    /// the map immediately.
    ///
    /// Ignored outright while [`EditMode::Off`] — with editing off the key
    /// belongs to whatever else may want it, and destroying data from a mode
    /// that shows no editing affordance at all would be indefensible.
    fn edit_delete_key(&mut self, ctx: &Context) {
        if self.edit.mode() == EditMode::Off {
            return;
        }
        if ctx.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Delete)) {
            if self
                .edit
                .selection()
                .is_some_and(|selection| selection.vertex.is_some())
            {
                self.delete_selected_vertex();
            } else {
                self.delete_selected_feature();
            }
        }
    }

    /// Applies one toolbar, menu or keyboard request.
    ///
    /// Every entry point funnels through here, so a button and its shortcut
    /// cannot drift apart.
    pub(super) fn apply_edit_action(&mut self, action: EditAction) {
        match action {
            EditAction::SetMode(mode) => self.set_edit_mode(mode),
            EditAction::NewLayer(kind) => self.add_edit_layer(kind),
            EditAction::Undo => {
                self.undo_once();
            }
            EditAction::Redo => {
                self.redo_once();
            }
            EditAction::DeleteFeature => {
                self.delete_selected_feature();
            }
            EditAction::ToggleWindow => {
                let open = !self.edit.show_window();
                self.set_edit_window_open(open);
            }
            EditAction::ToggleSnap => {
                let enabled = !self.edit.snap_settings().enabled;
                self.edit.set_snap_enabled(enabled);
                self.status = Some(if enabled {
                    "Snapping on \u{2014} hold Ctrl to suspend it for one gesture.".to_string()
                } else {
                    "Snapping off.".to_string()
                });
            }
            EditAction::DeleteVertex => {
                self.delete_selected_vertex();
            }
            EditAction::ValidateLayer => {
                self.validate_active_layer();
            }
            EditAction::ShowValidation => {
                // Open — never toggle: the badge promises "opens the Edit
                // window's Validation list", and with the window already open
                // (the common case) a toggle would close it instead.
                self.set_edit_window_open(true);
                self.edit.request_reveal_validation();
            }
        }
    }

    /// Opens or closes the Edit window.
    ///
    /// Closing ends the attribute form's coalescing window, which is what makes
    /// "successive Applies fold into one undo step **until the window closes**"
    /// true rather than aspirational.
    pub(super) fn set_edit_window_open(&mut self, open: bool) {
        if self.edit.show_window() == open {
            return;
        }
        self.edit.set_show_window(open);
        if !open {
            self.undo.close_coalescing();
        }
        self.status = Some(if open {
            "Edit window opened.".to_string()
        } else {
            "Edit window closed.".to_string()
        });
    }

    /// Switches tool, closing the coalescing window so the next edit starts its
    /// own undo step.
    fn set_edit_mode(&mut self, mode: EditMode) {
        if self.edit.mode() == mode {
            return;
        }
        self.edit.set_mode(mode);
        self.edit.clear_cycle();
        self.undo.close_coalescing();
        self.status = Some(match mode {
            EditMode::Off => "Editing off — the map is back to browsing.".to_string(),
            _ => format!("{} tool.", mode.label()),
        });
    }

    /// Creates an empty local vector layer to digitize into, selects it and
    /// latches the matching drawing tool.
    ///
    /// Unlike [`Self::add_geojson_layer_from_text`] this does **not** zoom to
    /// the new layer. An empty collection's extent is
    /// [`crate::local_vector::MercatorSquare`]'s whole world, so "zoom to it"
    /// means "throw away wherever the user had navigated to" — precisely when
    /// they are about to draw there.
    fn add_edit_layer(&mut self, kind: StyleKind) {
        let name = match kind {
            StyleKind::Circle => "New points",
            StyleKind::Line => "New lines",
            StyleKind::Fill => "New polygons",
            StyleKind::Symbol => "New labels",
        };
        match self
            .local
            .add_empty_vector_layer(&mut self.project, name, kind)
        {
            Ok(added) => {
                self.selection = Some(added.id);
                self.undo.close_coalescing();
                if let Some(notice) = self.edit.retarget(Some(added.id)) {
                    self.push_edit_notice(notice);
                }
                let mode = toolbar::mode_for_style(kind);
                self.edit.set_mode(mode);
                self.status = Some(format!(
                    "Added {name}; the {} tool draws into it.",
                    mode.label()
                ));
                self.record_layer_add(&[added.id]);
            }
            Err(error) => {
                self.status = Some(format!("{name}: {}", error.message()));
                tracing::warn!(
                    name,
                    error = error.message(),
                    "oxigis-ui: new edit layer refused"
                );
            }
        }
    }

    /// Deletes every selected feature, as ONE undoable transaction —
    /// strictly descending `Remove`s, so each op's index is its
    /// pre-transaction index and one `Ctrl+Z` restores the whole set.
    ///
    /// Returns whether anything was deleted; a refusal always says why on the
    /// status line rather than failing silently.
    pub(super) fn delete_selected_feature(&mut self) -> bool {
        let Some(id) = self.selection else {
            self.status = Some("Select a layer before deleting a feature.".to_string());
            return false;
        };
        let Some(selection) = self.edit.selection() else {
            self.status = Some("Click a feature on the map before deleting it.".to_string());
            return false;
        };
        let selected: Vec<usize> = self
            .edit
            .multi_selection()
            .map(|multi| multi.features().to_vec())
            .unwrap_or_default();
        let Some(features) = self.local.feature_set(id) else {
            self.status = Some(EditError::FeaturesNotLoaded(id).to_string());
            return false;
        };
        let ops = match command::remove_features_ops(features, &selected) {
            Ok(ops) if !ops.is_empty() => ops,
            Ok(_) => {
                self.status = Some("Click a feature on the map before deleting it.".to_string());
                return false;
            }
            Err(error) => {
                self.status = Some(error.to_string());
                return false;
            }
        };
        let count = ops.len();
        self.undo.close_coalescing();
        let deleted = self.commit_edit(EditTransaction {
            layer: id,
            label: if count > 1 {
                "Delete features"
            } else {
                "Delete feature"
            },
            ops,
            selection_before: Some(selection),
            selection_after: None,
            coalesce: None,
        });
        if deleted && count > 1 {
            self.status = Some(format!(
                "{count} features deleted — one Ctrl+Z puts them all back."
            ));
        }
        deleted
    }

    /// Deletes the picked vertex, as one undoable transaction.
    ///
    /// A delete that would leave a line with one position or a ring with two is
    /// **refused** rather than allowed to destroy the geometry: the mutator
    /// answers [`EditError::TooFewVertices`] and that answer is put on the
    /// status line verbatim. Returns whether anything was deleted.
    pub(super) fn delete_selected_vertex(&mut self) -> bool {
        // A marquee-marked set outranks the single pick: that is what the
        // status line promised when the box landed.
        if let Some(multi) = self.edit.multi_selection().cloned()
            && !multi.vertex_set().is_empty()
        {
            return self.delete_marked_vertices(&multi);
        }
        let Some(id) = self.selection else {
            self.status = Some("Select a layer before deleting a vertex.".to_string());
            return false;
        };
        let Some(selection) = self.edit.selection() else {
            self.status = Some("Click a feature on the map before editing it.".to_string());
            return false;
        };
        let Some(at) = selection.vertex else {
            self.status =
                Some("Click a vertex handle to pick it, then Delete removes it.".to_string());
            return false;
        };
        let Some(features) = self.local.feature_set(id).map(Arc::clone) else {
            self.status = Some(EditError::FeaturesNotLoaded(id).to_string());
            return false;
        };
        match edit::remove_vertex_transaction(id, &features, selection.feature, at, Some(selection))
        {
            Ok(transaction) => {
                self.undo.close_coalescing();
                self.commit_edit(transaction)
            }
            Err(error) => {
                self.status = Some(error.to_string());
                false
            }
        }
    }

    /// Adopts the app's layer selection as the edit target and clamps the
    /// feature selection against the layer's live collection.
    ///
    /// Called once per frame, which is what makes invariant I5 — a selection
    /// always addresses a feature that exists — unrepresentable rather than
    /// merely unlikely: any path that shortens or replaces a collection is
    /// caught on the very next frame, whether or not it remembered to say so.
    pub(super) fn sync_edit_state(&mut self) {
        if let Some(notice) = self.edit.retarget(self.selection) {
            self.push_edit_notice(notice);
        }
        let features = self.selection.and_then(|id| self.local.feature_set(id));
        self.edit.validate_selection(features.map(|set| &**set));
    }

    /// This frame's toolbar inputs.
    ///
    /// Separate from [`Self::edit_toolbar`] so what the buttons and their
    /// tooltips promise is a value a test can read, rather than something only
    /// a painted `Ui` knows.
    ///
    /// `undo_label`/`redo_label` come from the family-agnostic
    /// [`crate::edit::stack::EditStack::peek_undo_entry`], **not** from the
    /// feature-only `peek_undo`: `can_undo` counts project operations too, so
    /// reading the feature-only view would leave the enabled `↩` button
    /// tooltipped "Nothing to undo" after every layer add, remove, reorder,
    /// opacity drag, style edit and basemap change.
    pub(super) fn toolbar_state(&self) -> ToolbarState<'static> {
        let can_draw = self
            .selection
            .is_some_and(|id| self.is_local_layer(id) && self.local.feature_set(id).is_some());
        ToolbarState {
            mode: self.edit.mode(),
            can_draw,
            has_selection: self.edit.selection().is_some(),
            has_vertex: self
                .edit
                .selection()
                .is_some_and(|selection| selection.vertex.is_some()),
            snap: self.edit.snap_settings().enabled,
            can_undo: self.undo.can_undo(),
            can_redo: self.undo.can_redo(),
            undo_label: self
                .undo
                .peek_undo_entry()
                .map(crate::edit::stack::UndoEntry::label),
            redo_label: self
                .undo
                .peek_redo_entry()
                .map(crate::edit::stack::UndoEntry::label),
            window_open: self.edit.show_window(),
            issues: self.selection.map_or(0, |id| self.edit.issue_count(id)),
        }
    }

    /// Draws the editing toolbar and applies whatever it reports.
    pub(super) fn edit_toolbar(&mut self, ui: &mut egui::Ui) {
        let state = self.toolbar_state();
        for action in toolbar::panel(ui, &state) {
            self.apply_edit_action(action);
        }
    }

    /// This frame's map interaction: the cursor, the end of a vertex gesture,
    /// and a `Select`-mode click.
    ///
    /// Returns the transactions the gesture produced, for the caller to commit
    /// *after* the frame's painting — so the overlay this frame still shows the
    /// gesture that produced them, and the data moves exactly once.
    pub(super) fn edit_interact(
        &mut self,
        ui: &egui::Ui,
        rect: egui::Rect,
        response: &egui::Response,
    ) -> Vec<EditTransaction> {
        let mode = self.edit.mode();
        if mode == EditMode::Off {
            return Vec::new();
        }
        // A live drag is re-derived against the post-`allocate` camera before
        // anything reads it: the pan gate updated it before this frame's
        // wheel/pinch zoom was applied, so on a zooming frame its position —
        // and its snap marker — would otherwise trail the overlay's camera by
        // one zoom step, and a release on such a frame would commit the stale
        // position the marker never showed.
        if self.edit.drag().is_some() {
            let ppp = ui.ctx().pixels_per_point();
            let view = self.map_panel.view();
            let pointer = response
                .interact_pointer_pos()
                .or_else(|| response.hover_pos());
            let suspend_snap = ui.ctx().input(|input| input.modifiers.ctrl);
            let Self {
                edit,
                local,
                project,
                selection,
                ..
            } = self;
            let target = *selection;
            let ctx = EditCtx {
                project,
                target,
                features: target.and_then(|id| local.feature_set(id)),
                view,
                rect,
                ppp,
            };
            edit.refresh_drag(&ctx, pointer, suspend_snap);
        }
        self.edit_cursor(ui, response, mode);
        if let Some(transaction) = self.finish_vertex_drag(ui.ctx(), response) {
            return vec![transaction];
        }
        if self.edit.drag().is_some() {
            // A live gesture owns the frame: a click resolved now would be the
            // press that started it.
            return Vec::new();
        }
        if mode.is_drawing() {
            return self.draw_interact(ui, rect, response, mode);
        }
        // A finished box-select resolves here, where the app state it marks
        // lives; while the button is down the gate keeps the camera out.
        if self.edit.marquee().is_some() && !ui.ctx().input(|input| input.pointer.primary_down()) {
            let ppp = ui.ctx().pixels_per_point();
            self.finish_marquee(rect, ppp);
        }
        if mode == EditMode::Select
            && response.clicked()
            && let Some(pointer) = response
                .interact_pointer_pos()
                .filter(|pointer| rect.contains(*pointer))
        {
            let ppp = ui.ctx().pixels_per_point();
            let additive = ui.ctx().input(|input| input.modifiers.shift);
            self.select_at(pointer, rect, ppp, additive);
        }
        Vec::new()
    }

    /// Resolves a released box-select: marks the anchor's handles inside the
    /// box, or — for a box too small to mean one — falls back to the
    /// additive click it visually was.
    fn finish_marquee(&mut self, rect: egui::Rect, ppp: f32) {
        let Some(marquee) = self.edit.take_marquee() else {
            return;
        };
        let box_rect = egui::Rect::from_two_pos(marquee.start, marquee.current);
        if box_rect.size().length() < 4.0 {
            self.select_at(marquee.current, rect, ppp, true);
            return;
        }
        let Some(id) = self.selection else {
            return;
        };
        let Some(multi) = self.edit.multi_selection().cloned() else {
            self.status = Some("Click a feature before box-selecting its vertices.".to_string());
            return;
        };
        let Some(features) = self.local.feature_set(id) else {
            return;
        };
        let Some(geometry) = features
            .features
            .get(multi.anchor())
            .and_then(|feature| feature.geometry.as_ref())
        else {
            return;
        };
        // The same rect-adjusted camera the gate hit-tests with.
        let view = self.map_panel.view();
        let view = view
            .with_size_px([rect.width() * ppp, rect.height() * ppp])
            .unwrap_or(view);
        let marked: Vec<crate::edit::VertexRef> =
            hit::visible_vertex_positions(geometry, view, rect, ppp)
                .into_iter()
                .filter(|(_, position)| {
                    box_rect.contains(hit::to_screen(view, rect.min, ppp, *position))
                })
                .map(|(vertex, _)| vertex)
                .collect();
        if marked.is_empty() {
            self.status = Some("No vertex handles inside the box.".to_string());
            return;
        }
        let count = marked.len();
        self.edit
            .set_multi_selection(Some(multi.with_vertex_set(marked)));
        self.status = Some(match count {
            1 => "1 vertex marked — drag it to move it, Delete removes it.".to_string(),
            count => {
                format!("{count} vertices marked — drag one to move them all, Delete removes them.")
            }
        });
    }

    /// Deletes the marquee-marked vertex set of the anchor feature, as ONE
    /// undoable `Replace`. A deletion that would degenerate any path (a line
    /// under two positions, a ring under three) refuses whole.
    fn delete_marked_vertices(&mut self, multi: &crate::edit::selection::FeatureSelection) -> bool {
        let Some(id) = self.selection else {
            return false;
        };
        let index = multi.anchor();
        let Some(features) = self.local.feature_set(id) else {
            self.status = Some(EditError::FeaturesNotLoaded(id).to_string());
            return false;
        };
        let Some(before) = features.features.get(index).cloned() else {
            self.status = Some(
                EditError::IndexOutOfRange {
                    index,
                    len: features.features.len(),
                }
                .to_string(),
            );
            return false;
        };
        let mut after = before.clone();
        let mut sorted: Vec<crate::edit::VertexRef> = multi.vertex_set().to_vec();
        sorted.sort_unstable();
        for vertex in sorted.iter().rev() {
            if let Err(error) = command::remove_vertex(&mut after, *vertex) {
                self.status = Some(format!("{error} — nothing was deleted."));
                return false;
            }
        }
        let count = sorted.len();
        self.undo.close_coalescing();
        let landed = self.commit_edit(EditTransaction {
            layer: id,
            label: "Delete vertices",
            ops: vec![command::FeatureOp::Replace {
                index,
                before: Box::new(before),
                after: Box::new(after),
            }],
            selection_before: self.edit.selection(),
            selection_after: Some(EditSelection::feature(index)),
            coalesce: None,
        });
        if landed {
            self.status = Some(format!(
                "{count} vertices deleted — one Ctrl+Z restores them."
            ));
        }
        landed
    }

    /// One frame of a digitizing tool: the snapped cursor, and whichever gesture
    /// appends or finishes.
    ///
    /// The gesture table, in the order it is tested:
    ///
    /// | Gesture | Effect |
    /// |---|---|
    /// | double click | finish (deduped) — `DrawPoint` ignores it |
    /// | secondary click | finish, **only** while a sketch is in progress |
    /// | click on vertex 0 | close the ring (`DrawPolygon`, from three vertices) |
    /// | click | commit a point / append a vertex |
    ///
    /// The double click is tested **first**, and returns, because egui reports
    /// `clicked()` on the second release of a double click as well — testing the
    /// click first would append a duplicate final vertex to every
    /// double-click-finished feature.
    fn draw_interact(
        &mut self,
        ui: &egui::Ui,
        rect: egui::Rect,
        response: &egui::Response,
        mode: EditMode,
    ) -> Vec<EditTransaction> {
        // No target, no vertices: the toolbar disables the drawing buttons
        // when there is nowhere to draw, but the `P`/`L`/`G` keys latch the
        // tools regardless, and vertices collected with no commit target would
        // be silently discarded the moment the sketch was finished. Refusing
        // the *first* click is the honest version of that refusal.
        let target_ready = match self.selection {
            None => {
                if response.clicked() || response.double_clicked() {
                    self.status = Some(EditError::NoTargetLayer.to_string());
                }
                false
            }
            Some(id) if self.local.feature_set(id).is_none() => {
                if response.clicked() || response.double_clicked() {
                    self.status = Some(EditError::FeaturesNotLoaded(id).to_string());
                }
                false
            }
            Some(_) => true,
        };
        if !target_ready {
            self.edit.sketch_mut().cursor = None;
            return Vec::new();
        }
        let ppp = ui.ctx().pixels_per_point();
        let view = self.map_panel.view();
        // Touch first: on touch the frame the pointer first exists is already
        // the press frame, and there is no hover position at all.
        let pointer = response
            .interact_pointer_pos()
            .or_else(|| response.hover_pos())
            .filter(|pointer| rect.contains(*pointer));
        let Some(pointer) = pointer else {
            // The rubber band's free end has nowhere to be; the committed
            // vertices stay exactly as they are.
            self.edit.sketch_mut().cursor = None;
            return Vec::new();
        };
        // Ctrl suspends snapping for as long as it is held — a per-gesture
        // decision rather than a mode to remember.
        let suspend_snap = ui.ctx().input(|input| input.modifiers.ctrl);
        let position = {
            let Self {
                edit,
                local,
                project,
                selection,
                ..
            } = self;
            let target = *selection;
            let ctx = EditCtx {
                project,
                target,
                features: target.and_then(|id| local.feature_set(id)),
                view,
                rect,
                ppp,
            };
            edit.snap_for_sketch(&ctx, pointer, suspend_snap)
        };
        self.edit.sketch_mut().cursor = Some(position);

        if response.double_clicked() {
            if mode == EditMode::DrawPoint {
                // The first release of the pair already committed a point; a
                // second one on top of it is not what "double click" meant.
                return Vec::new();
            }
            let geometry = self
                .edit
                .sketch_mut()
                .finish_from_double_click(mode, view, ppp);
            return self.sketch_transactions(mode, geometry);
        }
        // Secondary is claimed *only* while a sketch is in progress; elsewhere
        // it stays reserved for the context menu `map_view` names.
        if self.edit.sketch().is_active() && response.secondary_clicked() {
            let geometry = self.edit.sketch_mut().finish(mode);
            return self.sketch_transactions(mode, geometry);
        }
        if !response.clicked() {
            return Vec::new();
        }
        if mode == EditMode::DrawPoint {
            return self.sketch_transactions(mode, sketch::point_geometry(position));
        }
        let tolerance = self.edit.snap_settings().tolerance_pt;
        if mode == EditMode::DrawPolygon
            && self.edit.sketch().len() >= sketch::MIN_POLYGON_VERTICES
            && self
                .edit
                .sketch()
                .closes_at(view, rect.min, pointer, ppp, tolerance)
        {
            let geometry = self.edit.sketch_mut().finish(mode);
            return self.sketch_transactions(mode, geometry);
        }
        self.edit.sketch_mut().append(mode, position);
        let count = self.edit.sketch().len();
        self.status = Some(format!(
            "{} sketch: {count} vertices \u{2014} Enter or double-click to finish.",
            mode.label()
        ));
        Vec::new()
    }

    /// Wraps a finished sketch's geometry into the transaction that adds it.
    ///
    /// [`None`] means the sketch was finished before it was a geometry: the
    /// sketch was left untouched by `finish`, and the refusal says which
    /// gesture is still missing.
    fn sketch_transactions(
        &mut self,
        mode: EditMode,
        geometry: Option<Geometry>,
    ) -> Vec<EditTransaction> {
        let Some(geometry) = geometry else {
            self.status = Some(sketch::too_few_message(mode));
            return Vec::new();
        };
        self.sketch_transaction(mode, geometry)
            .map(|transaction| vec![transaction])
            .unwrap_or_default()
    }

    /// The `Add` transaction one finished sketch asks for.
    ///
    /// The new feature lands at `len`, so no existing index moves and the
    /// `Add`/`Remove` pair stays an exact inverse; it is then selected, so the
    /// vertex handles, the attribute form and `Delete` all address what was just
    /// drawn. The tool stays latched — drawing several features in a row is the
    /// common case.
    fn sketch_transaction(
        &mut self,
        mode: EditMode,
        geometry: Geometry,
    ) -> Option<EditTransaction> {
        let Some(id) = self.selection else {
            self.status = Some(EditError::NoTargetLayer.to_string());
            return None;
        };
        let Some(features) = self.local.feature_set(id) else {
            self.status = Some(EditError::FeaturesNotLoaded(id).to_string());
            return None;
        };
        let index = features.features.len();
        self.undo.close_coalescing();
        Some(sketch::add_feature_transaction(
            id,
            index,
            geometry,
            sketch::draw_label(mode),
            self.edit.selection(),
        ))
    }

    /// Finishes the sketch from the keyboard and commits it straight away.
    ///
    /// Returns whether a feature was added. Unlike the pointer path this does
    /// not defer to the end of the frame: `edit_shortcuts` runs before anything
    /// is painted, so there is no overlay state left to keep consistent.
    pub(super) fn finish_sketch(&mut self) -> bool {
        // The target is checked *before* `Sketch::finish`, which consumes the
        // vertices on success: a finish that then finds nowhere to commit
        // would have already destroyed the sketch, and digitized geometry must
        // never be discarded by a refusal the user can fix and retry.
        let Some(id) = self.selection else {
            self.status = Some(EditError::NoTargetLayer.to_string());
            return false;
        };
        if self.local.feature_set(id).is_none() {
            self.status = Some(EditError::FeaturesNotLoaded(id).to_string());
            return false;
        }
        let mode = self.edit.mode();
        let Some(geometry) = self.edit.sketch_mut().finish(mode) else {
            self.status = Some(sketch::too_few_message(mode));
            return false;
        };
        let Some(transaction) = self.sketch_transaction(mode, geometry) else {
            return false;
        };
        let label = transaction.label;
        if !self.commit_edit(transaction) {
            return false;
        }
        self.status = Some(format!("{label} \u{2014} the tool stays selected."));
        true
    }

    /// Ends a live vertex gesture, if this is the frame it ended on.
    ///
    /// Release is detected by `drag_stopped_by(Primary)` **or** by the primary
    /// button simply not being down any more. The second test is not
    /// redundancy: a release lost to a focus change — an alt-tab, a browser tab
    /// switch, a dialog stealing the pointer — never reaches egui as a drag
    /// stop, and without it the gesture would wedge, the pan gate would stay
    /// shut, and the map would be permanently un-pannable.
    fn finish_vertex_drag(
        &mut self,
        ctx: &Context,
        response: &egui::Response,
    ) -> Option<EditTransaction> {
        self.edit.drag()?;
        let released = response.drag_stopped_by(egui::PointerButton::Primary)
            || !ctx.input(|input| input.pointer.primary_down());
        if !released {
            return None;
        }
        let drag = self.edit.take_drag()?;
        self.undo.close_coalescing();
        if !drag.moved {
            if drag.is_set_move() {
                // A stray click on a MARKED handle must not discard the
                // marquee's work — the set stays, with its affordance.
                self.status = Some(format!(
                    "{} vertices marked \u{2014} drag one of them to move them all.",
                    drag.set.len()
                ));
                return None;
            }
            // A press and release without motion is a *click*: it picks the
            // vertex, and pushes nothing onto the undo stack.
            self.edit
                .set_selection(Some(EditSelection::vertex(drag.feature, drag.vertex)));
            self.status = Some(if drag.inserting {
                "Drag a midpoint ghost to insert a vertex there.".to_string()
            } else {
                format!(
                    "Vertex {} of feature {} picked.",
                    drag.vertex.index, drag.feature
                )
            });
            return None;
        }
        let Some(id) = self.selection else {
            self.status = Some(EditError::NoTargetLayer.to_string());
            return None;
        };
        let Some(features) = self.local.feature_set(id).map(Arc::clone) else {
            self.status = Some(EditError::FeaturesNotLoaded(id).to_string());
            return None;
        };
        match edit::drag_transaction(id, &features, &drag, self.edit.selection()) {
            Ok(transaction) => {
                if drag.is_set_move() {
                    // The marks survive the commit: a translation renumbers
                    // nothing, so every mark still names the same corner.
                    self.edit.arm_marks_after_commit(drag.set.clone());
                    self.status = Some(format!(
                        "{} vertices moved \u{2014} one Ctrl+Z puts them back.",
                        drag.set.len()
                    ));
                }
                Some(transaction)
            }
            Err(error) => {
                self.status = Some(format!("{error} \u{2014} nothing was moved."));
                None
            }
        }
    }

    /// A cursor change is the cheapest possible statement of "this click will do
    /// something different".
    fn edit_cursor(&self, ui: &egui::Ui, response: &egui::Response, mode: EditMode) {
        // Checked before `hovered()`: a drag that has wandered off the panel is
        // still a drag, and the cursor must not revert to an arrow half way
        // through placing a vertex.
        if self.edit.drag().is_some() {
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            return;
        }
        if !response.hovered() {
            return;
        }
        let icon = if mode.is_drawing() {
            CursorIcon::Crosshair
        } else {
            match self.edit.hover_target() {
                Some(HitTarget::Vertex { .. }) => CursorIcon::Grab,
                Some(HitTarget::Midpoint { .. }) => CursorIcon::Copy,
                Some(HitTarget::Feature { .. }) => CursorIcon::PointingHand,
                None => return,
            }
        };
        ui.ctx().set_cursor_icon(icon);
    }

    /// Resolves a `Select`-mode click into a selection: a vertex when the
    /// click lands on a drawn handle, the feature stack otherwise.
    ///
    /// The handle test runs **first**, because a click never reaches the drag
    /// path — egui only calls a gesture a drag after ~6 pt of motion or a long
    /// press, so a plain press-and-release on a handle arrives here as
    /// `clicked()`. Without this test the click would fall through to the
    /// feature pick, which knows nothing about vertices and would happily move
    /// the selection onto an overlapping feature — making "Click a vertex
    /// handle to pick it, then Delete removes it" (the status line's own
    /// instruction) unfollowable by clicking.
    fn select_at(&mut self, pointer: egui::Pos2, rect: egui::Rect, ppp: f32, additive: bool) {
        let Some(id) = self.selection else {
            self.status = Some("Select a layer in the layer panel to edit it.".to_string());
            return;
        };
        let view = self.map_panel.view();
        // A Shift+click is a feature-set gesture: it never picks a vertex
        // handle, so the handle test is skipped outright (the same reasoning
        // that makes a Shift+drag never a handle drag).
        let handle_hit = if additive {
            None
        } else {
            let ctx = EditCtx {
                project: &self.project,
                target: Some(id),
                features: self.local.feature_set(id),
                view,
                rect,
                ppp,
            };
            self.edit
                .hit_test(&ctx, pointer, self.edit.handles().is_active())
        };
        match handle_hit {
            Some(HitTarget::Vertex { feature, at }) => {
                self.undo.close_coalescing();
                self.edit
                    .set_selection(Some(EditSelection::vertex(feature, at)));
                self.status = Some(format!("Vertex {} of feature {feature} picked.", at.index));
                return;
            }
            Some(HitTarget::Midpoint { .. }) => {
                // A midpoint ghost is a drag affordance, not a pick target: a
                // click on it neither inserts (that needs the drag's position)
                // nor should it knock the selection onto the feature stack.
                self.status = Some("Drag a midpoint ghost to insert a vertex there.".to_string());
                return;
            }
            Some(HitTarget::Feature { .. }) | None => {}
        }
        let picked = {
            let ctx = EditCtx {
                project: &self.project,
                target: Some(id),
                features: self.local.feature_set(id),
                view,
                rect,
                ppp,
            };
            self.edit.pick_click(&ctx, pointer)
        };
        self.undo.close_coalescing();
        match picked {
            Some((feature, position, total)) if additive => {
                let was_selected = self
                    .edit
                    .multi_selection()
                    .is_some_and(|multi| multi.contains(feature));
                if self.edit.toggle_feature(feature) {
                    let count = self.edit.multi_selection().map_or(0, |multi| multi.len());
                    self.status = Some(if was_selected {
                        format!("Feature {feature} removed from the selection — {count} selected.")
                    } else if total > 1 {
                        format!(
                            "Feature {feature} added — {count} selected ({position} of {total} \
                             here; Shift+click again to walk the stack)."
                        )
                    } else {
                        format!("Feature {feature} added — {count} selected.")
                    });
                } else {
                    self.edit.clear_cycle();
                    self.status = Some("Selection cleared.".to_string());
                }
            }
            Some((feature, position, total)) => {
                self.edit
                    .set_selection(Some(EditSelection::feature(feature)));
                self.status = Some(if total > 1 {
                    format!(
                        "Feature {feature} selected — {position} of {total} here; click again to \
                         cycle."
                    )
                } else {
                    format!("Feature {feature} selected.")
                });
            }
            None if additive => {
                // An additive gesture that hit nothing must not throw the
                // set away — missing a small feature with Shift held is
                // exactly the moment losing the selection hurts most.
                self.status = Some("Nothing here — the selection is unchanged.".to_string());
            }
            None => {
                self.edit.set_selection(None);
                self.edit.clear_cycle();
                self.status = Some("Nothing here — selection cleared.".to_string());
            }
        }
    }

    /// Paints the edit overlay over the map.
    ///
    /// Order, later on top: selection outline → drag ghost → rubber band →
    /// midpoint ghosts → vertex handles → snap marker → hint plate. Slots
    /// between `paint_gpu` and `paint_attribution` — later shapes draw on top,
    /// and the basemap credit must never be covered.
    pub(super) fn edit_overlay(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        let mode = self.edit.mode();
        if mode == EditMode::Off {
            return;
        }
        let ppp = ui.ctx().pixels_per_point();
        let painter = ui.painter_at(rect);
        // Destructured, so the geometry can stay *borrowed* out of the feature
        // store while the edit state is mutated for its scratch buffers. The
        // alternative is one deep `Geometry` clone every frame a selection is
        // live, which on a 10 000-vertex polygon is exactly the per-frame
        // allocation this overlay exists to avoid.
        let Self {
            selection,
            local,
            edit,
            map_panel,
            project,
            ..
        } = self;
        let view = map_panel.view();
        let target = *selection;
        // Re-plan the handle verdict from *this frame's* selection: the pan
        // gate planned it before `edit_interact` resolved this frame's click,
        // so on the frame a feature is newly picked the stale plan would paint
        // the new selection's outline with no handles to grab.
        let ctx = EditCtx {
            project,
            target,
            features: target.and_then(|id| local.feature_set(id)),
            view,
            rect,
            ppp,
        };
        edit.refresh_handles(&ctx);
        let geometry = target
            .and_then(|id| local.feature_set(id))
            .zip(edit.selection())
            .and_then(|(features, picked)| features.features.get(picked.feature))
            .and_then(|feature| feature.geometry.as_ref());
        // Every co-selected member wears the same selection outline — the
        // anchor is distinguished by being the one with handles, exactly as
        // other GIS tools mark multi-selections.
        if let Some((features, multi)) = target
            .and_then(|id| local.feature_set(id))
            .zip(edit.multi_selection().cloned())
        {
            for &member in multi.features() {
                if member == multi.anchor() {
                    continue;
                }
                if let Some(geometry) = features
                    .features
                    .get(member)
                    .and_then(|feature| feature.geometry.as_ref())
                {
                    edit.paint_selection(&painter, view, rect, ppp, geometry);
                }
            }
        }
        if let Some(geometry) = geometry {
            // While a gesture is live the preview replaces the outline: both are
            // drawn in the same accent from the same coordinates, and the
            // outline on top would simply hide the preview.
            if edit.drag().is_some() {
                edit.paint_drag(&painter, view, rect, ppp);
            } else {
                edit.paint_selection(&painter, view, rect, ppp, geometry);
            }
        }
        edit.paint_sketch(&painter, view, rect, ppp);
        // The live box-select rectangle, over the outlines and under the
        // handles.
        if let Some(marquee) = edit.marquee() {
            let box_rect = egui::Rect::from_two_pos(marquee.start, marquee.current);
            painter.rect_filled(
                box_rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(70, 140, 255, 24),
            );
            painter.rect_stroke(
                box_rect,
                0.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 140, 255)),
                egui::StrokeKind::Inside,
            );
        }
        if let Some(geometry) = geometry.filter(|_| edit.handles().is_active()) {
            let picked = edit.selection().and_then(|picked| picked.vertex);
            let mut handles = hit::visible_vertex_positions(geometry, view, rect, ppp);
            match edit.drag() {
                Some(drag) => handles = overlay::drag_handles(handles, drag),
                None => {
                    // Ghosts during a gesture are noise: nothing may be inserted
                    // until the one in flight lands.
                    let ghosts = hit::visible_midpoint_positions(geometry, view, rect, ppp);
                    overlay::paint_midpoint_ghosts(&painter, view, rect, ppp, &ghosts);
                }
            }
            overlay::paint_handles(&painter, view, rect, ppp, &handles, picked);
            // Marquee-marked vertices wear a ring over their handles — read
            // from the already drag-adjusted handle list, so during a set
            // move the rings follow the preview (and the old per-mark
            // geometry re-walk is gone with them).
            if let Some(multi) = edit.multi_selection() {
                let marks = multi.vertex_set();
                for (reference, position) in &handles {
                    if marks.binary_search(reference).is_err() {
                        continue;
                    }
                    let at = hit::to_screen(view, rect.min, ppp, *position);
                    painter.circle_stroke(
                        at,
                        6.0,
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 170, 40)),
                    );
                }
            }
        }
        if let Some(snap) = edit.hover_snap() {
            overlay::paint_snap_marker(&painter, &snap);
        }
        let base = overlay::hint_text(mode, edit.selection(), edit.cycle_position(), edit.sketch());
        let extra = [
            overlay::handle_hint(edit.handles()),
            edit.snap_degraded()
                .then(|| overlay::SNAP_DEGRADED_HINT.to_string()),
        ];
        if let Some(text) = overlay::plate_text(base, &extra) {
            overlay::paint_hint(&painter, rect, &text);
        }
    }

    /// Keeps the attribute table's selection and the edit selection in step,
    /// with exactly **one** writer per frame.
    ///
    /// Must be called *after* `AttributeTablePanel::show`: `bind()` clears the
    /// panel's selection on every new `Arc`, i.e. after every single edit, so a
    /// re-assert made before the panel draws is thrown away immediately.
    ///
    /// `table_row_clicked` is true on exactly one frame by construction, which
    /// is what makes a ping-pong between the two writers structurally
    /// impossible rather than merely unlikely. The map side always addresses
    /// features by their **source** index, so it goes through
    /// [`crate::table_panel::AttributeTablePanel::select_source_feature`] and
    /// never through `select_visible_row`, which takes a *visible* row and would
    /// silently pick the wrong feature under any active sort.
    pub(super) fn sync_table_selection(&mut self, table_row_clicked: bool) {
        if table_row_clicked {
            let source = self.table_panel.selected_feature();
            self.edit.set_selection(source.map(EditSelection::feature));
            self.edit.clear_cycle();
            self.undo.close_coalescing();
        } else {
            match self.edit.multi_selection() {
                Some(multi) => {
                    self.table_panel
                        .select_source_features(Some(multi.anchor()), multi.features());
                }
                None => self.table_panel.select_source_feature(None),
            }
        }
    }
}

/// Rewrites layer `id`'s stored source, keeping its position in the stack.
///
/// [`oxigis_core::LayerStack`] exposes no mutable accessor — ordering is its
/// whole reason to exist, so it hands out `&[Layer]` and its own move
/// operations and nothing else. The layer is therefore taken out, its `kind`
/// rewritten, put back on top, and walked down to the slot it came from. Its
/// id, name, visibility and opacity all travel inside the `Layer` value, and
/// [`oxigis_core::Project::styles`] is keyed by id, so the only thing that
/// changes is the source. The walk is `Vec::swap`s of a struct, i.e. moves —
/// nothing is deep-copied, which matters because a `kind` can hold a megabyte
/// of inline GeoJSON.
///
/// Returns whether the layer was found.
fn rewrite_layer_source(
    project: &mut oxigis_core::Project,
    id: LayerId,
    source: VectorSource,
) -> bool {
    let Some(index) = project
        .layers
        .layers()
        .iter()
        .position(|layer| layer.id == id)
    else {
        return false;
    };
    let Some(mut layer) = project.layers.remove(id) else {
        return false;
    };
    layer.kind = LayerKind::Vector(source);
    project.layers.add(layer);
    let top = project.layers.len().saturating_sub(1);
    for _ in index..top {
        if project.layers.move_down(id) != Ok(true) {
            break;
        }
    }
    true
}

/// The name to show for the source a layer is being converted away from.
fn source_display_name(source: &VectorSource) -> String {
    match source {
        VectorSource::LocalGeoJson { path }
        | VectorSource::LocalShapefile { path }
        | VectorSource::LocalGeoParquet { path } => local_input::display_name(path),
        VectorSource::LocalGpkg { path, table } => {
            format!(
                "{} {PATH_SEPARATOR} {table}",
                local_input::display_name(path)
            )
        }
        VectorSource::MvtTiles { url_template, .. } => url_template.clone(),
        VectorSource::TileArchive { archive, .. } => archive.location().to_string(),
        VectorSource::InlineGeoJson { .. } => "This layer".to_string(),
    }
}
