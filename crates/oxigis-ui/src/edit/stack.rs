// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The undo stack: a strict-LIFO log of [`EditTransaction`]s with an entry cap,
//! a byte budget, gesture-boundary coalescing and per-layer pruning.
//!
//! # Why deltas and not snapshots
//!
//! An absolute `Arc<FeatureCollection>` per entry retains a whole collection per
//! undo step: on a 100 000-feature layer (tens of megabytes) any honest byte
//! budget yields roughly *one* undo step. A delta retains one [`Feature`] clone
//! per entry — a 10 000-vertex polygon is around 160 KiB — so
//! [`UNDO_MAX_ENTRIES`] entries fit the budget on any dataset.
//!
//! The usual objection, that a delete renumbers the indices later entries name,
//! does not survive strict LIFO: an older entry can only be replayed *after*
//! this one has been undone, which restores the exact vector it was recorded
//! against. The same argument is what makes [`EditStack::prune_layer`] safe —
//! entries for different layers commute, so removing a subsequence leaves every
//! survivor individually valid.
//!
//! # One stack, not one per layer
//!
//! `Ctrl+Z` must undo *the last thing you did*, not the last thing on whichever
//! layer happens to be selected now.
//!
//! # No clock anywhere
//!
//! Coalescing windows are opened and closed by gesture boundaries — a drag
//! release, a mode change, a layer change, a selection change, or simply the
//! next transaction carrying a different key. A time-based window would make
//! every test of this module non-deterministic for no capability.
//!
//! [`Feature`]: oxigeo::geojson::types::Feature

use super::VertexRef;
use super::command::{EditTransaction, FeatureOp};
use super::project_op::{ProjectOp, ProjectTransaction};
use oxigis_core::LayerId;
use std::collections::VecDeque;

/// One undo step of either command family, on the ONE stack — `Ctrl+Z`
/// undoes the last thing the user did, whichever family it belonged to.
/// An enum rather than a trait object: undo/redo clone the entry, the
/// tests compare them, and coalescing matches on the entry's *shape*.
#[derive(Debug, Clone, PartialEq)]
pub enum UndoEntry {
    /// A feature edit ([`super::command`]).
    Features(EditTransaction),
    /// A project operation ([`super::project_op`]).
    Project(ProjectTransaction),
}

impl UndoEntry {
    /// Menu/status wording.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Features(transaction) => transaction.label,
            Self::Project(transaction) => transaction.label,
        }
    }

    /// The entry that exactly undoes this one.
    #[must_use]
    pub fn inverted(&self) -> Self {
        match self {
            Self::Features(transaction) => Self::Features(transaction.inverted()),
            Self::Project(transaction) => Self::Project(transaction.inverted()),
        }
    }

    /// Roughly how much heap this entry pins.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        match self {
            Self::Features(transaction) => transaction.estimated_bytes(),
            Self::Project(transaction) => transaction.estimated_bytes(),
        }
    }

    /// The entry's coalescing key, if it may fold.
    #[must_use]
    pub fn coalesce(&self) -> Option<CoalesceKey> {
        match self {
            Self::Features(transaction) => transaction.coalesce,
            Self::Project(transaction) => transaction.coalesce,
        }
    }

    /// The feature transaction, when this is one.
    #[must_use]
    pub fn features(&self) -> Option<&EditTransaction> {
        match self {
            Self::Features(transaction) => Some(transaction),
            Self::Project(_) => None,
        }
    }

    /// Whether this entry is about `layer` — for a grouped project op, about
    /// ANY of its members: pruning one member must drop the whole group
    /// entry, or a redo would splice a stale snapshot back over a re-read
    /// file.
    fn names_layer(&self, layer: LayerId) -> bool {
        match self {
            Self::Features(transaction) => transaction.layer == layer,
            Self::Project(transaction) => transaction.op.mentions_layer(layer),
        }
    }
}

impl From<EditTransaction> for UndoEntry {
    fn from(transaction: EditTransaction) -> Self {
        Self::Features(transaction)
    }
}

impl From<ProjectTransaction> for UndoEntry {
    fn from(transaction: ProjectTransaction) -> Self {
        Self::Project(transaction)
    }
}

/// What one [`EditStack::push`] had to drop to fit, split by cause.
///
/// The split is load-bearing, not decoration: with [`UNDO_MAX_ENTRIES`]
/// entries, **every** push past the cap ages one entry out — routine, silent.
/// Dropping entries because one push blew the BYTE budget is the surprising
/// case the user is told about: history was traded for this one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EvictionReport {
    /// Entries dropped because the byte budget was exceeded — reported.
    pub by_bytes: usize,
    /// Entries dropped because the entry cap was reached — routine, silent.
    pub by_cap: usize,
}

impl EvictionReport {
    /// Every dropped entry, whatever the cause.
    #[must_use]
    pub fn total(self) -> usize {
        self.by_bytes + self.by_cap
    }
}

/// Most undo entries kept at once.
pub const UNDO_MAX_ENTRIES: usize = 128;

/// Byte budget on a native build.
pub const UNDO_MAX_BYTES_NATIVE: usize = 32 << 20;

/// Byte budget in the browser, where the whole heap is smaller and shared with
/// the tile cache.
pub const UNDO_MAX_BYTES_WASM: usize = 8 << 20;

/// Which field of which feature a coalescing window covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoalesceField {
    /// The feature's whole property map — successive Applies of the attribute
    /// form fold into one entry.
    Properties,
    /// One vertex, so a gesture that emits several moves of the same handle
    /// stays one undo step.
    Vertex(VertexRef),
    /// A layer's opacity slider — the whole drag is one undo step. The key's
    /// `feature` is `0` by convention (the field is layer-scoped).
    Opacity,
    /// A layer's style edits — a slider drag folds the same way.
    Style(oxigis_core::StyleSlot),
}

/// The identity of a coalescing window.
///
/// Carries the stack's [`EditStack::epoch`] so a window can never span a project
/// load: after [`EditStack::reset`] the epoch differs, the key no longer
/// matches, and the next transaction starts a fresh entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoalesceKey {
    /// The stack epoch the window belongs to.
    pub epoch: u64,
    /// The layer being edited.
    pub layer: LayerId,
    /// The feature being edited.
    pub feature: usize,
    /// What about it.
    pub field: CoalesceField,
}

/// A strict-LIFO undo/redo log.
///
/// `entries[..cursor]` are applied and undoable; `entries[cursor..]` are
/// redoable. Every mutation keeps that invariant true, including after an
/// eviction or a prune.
#[derive(Debug)]
pub struct EditStack {
    /// Oldest entry at the front.
    entries: VecDeque<UndoEntry>,
    /// How many entries are currently applied.
    cursor: usize,
    /// Bumped by [`Self::reset`]; scopes [`CoalesceKey`]s to one project.
    epoch: u64,
    /// The coalescing window a further transaction may fold into.
    open_coalesce: Option<CoalesceKey>,
    /// Entry cap.
    max_entries: usize,
    /// Byte budget.
    max_bytes: usize,
    /// Current estimated footprint of [`Self::entries`].
    bytes: usize,
}

impl Default for EditStack {
    fn default() -> Self {
        Self::new()
    }
}

impl EditStack {
    /// A stack with this target's budgets.
    #[must_use]
    pub fn new() -> Self {
        let max_bytes = if cfg!(target_arch = "wasm32") {
            UNDO_MAX_BYTES_WASM
        } else {
            UNDO_MAX_BYTES_NATIVE
        };
        Self::with_budget(UNDO_MAX_ENTRIES, max_bytes)
    }

    /// A stack with explicit budgets.
    ///
    /// Budgets are fields rather than `cfg!` constants precisely so both the
    /// native and the browser figure are reachable — and therefore testable —
    /// in one native test run, the same reasoning
    /// [`crate::local_input::LocalInputState::with_path_support`] follows.
    #[must_use]
    pub fn with_budget(entries: usize, bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            cursor: 0,
            epoch: 0,
            open_coalesce: None,
            max_entries: entries,
            max_bytes: bytes,
            bytes: 0,
        }
    }

    /// The current epoch. Every [`CoalesceKey`] a caller builds must carry it,
    /// or it will simply never fold.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The entry cap in force.
    #[must_use]
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// The byte budget in force.
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Drops everything and bumps the epoch — a project load or File ▸ New.
    ///
    /// Both halves are load-bearing. [`LayerId`]s are reserved per deserialize,
    /// so project A and project B can legitimately both hold layer 3; an
    /// un-cleared stack would splice A's features into B's layer 3. The epoch
    /// bump is defence in depth for the one piece of state that outlives the
    /// entries — a caller still holding a [`CoalesceKey`] from before the load.
    pub fn reset(&mut self) {
        self.entries.clear();
        self.cursor = 0;
        self.bytes = 0;
        self.open_coalesce = None;
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Records `transaction` as the newest undoable step.
    ///
    /// Three things happen, in this order:
    ///
    /// 1. **The redo tail is truncated.** Doing something new after an undo
    ///    discards what was undone — the universal rule.
    /// 2. **Coalescing.** When `transaction.coalesce` matches the open window
    ///    and both it and the top entry are a single `Replace` of the same
    ///    feature, they fold: `Replace{A,B}` + `Replace{B,C}` becomes
    ///    `Replace{A,C}`, keeping the *first*'s `selection_before` and the
    ///    *second*'s `selection_after`. Correct by construction, because the
    ///    intermediate state is exactly what neither selection describes.
    /// 3. **Eviction.** Entries are dropped from the front until both budgets
    ///    hold, decrementing the cursor once per evicted entry. The newest entry
    ///    always survives, so the last thing the user did is always undoable
    ///    even if it alone exceeds the byte budget.
    ///
    /// The window is left open exactly when `transaction.coalesce` is
    /// [`Some`] and carries the current epoch, so a non-coalescing transaction
    /// closes it without a separate call.
    pub fn push(&mut self, entry: impl Into<UndoEntry>) -> EvictionReport {
        let entry = entry.into();
        // Discarding the redo tail is not an eviction: it is what a new
        // action after an undo means, so it is never reported.
        while self.entries.len() > self.cursor {
            if let Some(dropped) = self.entries.pop_back() {
                self.bytes = self.bytes.saturating_sub(dropped.estimated_bytes());
            }
        }

        let key = entry
            .coalesce()
            .filter(|candidate| candidate.epoch == self.epoch);
        let folded = key.is_some_and(|candidate| self.open_coalesce == Some(candidate))
            && self.fold_into_top(&entry);
        if !folded {
            self.bytes = self.bytes.saturating_add(entry.estimated_bytes());
            self.entries.push_back(entry);
            self.cursor = self.entries.len();
        }
        self.open_coalesce = key;
        self.evict()
    }

    /// Folds `entry` into the top entry when both are the same foldable
    /// shape — a single `Replace` of the same feature, or the same layer's
    /// opacity/style op. Returns whether it did.
    fn fold_into_top(&mut self, entry: &UndoEntry) -> bool {
        let Some(top_index) = self.cursor.checked_sub(1) else {
            return false;
        };
        let Some(top) = self.entries.get_mut(top_index) else {
            return false;
        };
        let previous_bytes = top.estimated_bytes();
        let folded = match (top, entry) {
            (UndoEntry::Features(top), UndoEntry::Features(next)) => fold_features(top, next),
            (UndoEntry::Project(top), UndoEntry::Project(next)) => fold_project(top, next),
            _ => false,
        };
        if folded {
            let new_bytes = self
                .entries
                .get(top_index)
                .map_or(0, UndoEntry::estimated_bytes);
            self.bytes = self
                .bytes
                .saturating_sub(previous_bytes)
                .saturating_add(new_bytes);
        }
        folded
    }

    /// Drops entries from the front until both budgets hold, classifying each
    /// drop by its cause — the byte budget wins when both are breached (it is
    /// the more alarming truth).
    fn evict(&mut self) -> EvictionReport {
        let mut report = EvictionReport::default();
        while self.entries.len() > 1
            && (self.entries.len() > self.max_entries || self.bytes > self.max_bytes)
        {
            let over_bytes = self.bytes > self.max_bytes;
            let Some(dropped) = self.entries.pop_front() else {
                break;
            };
            self.bytes = self.bytes.saturating_sub(dropped.estimated_bytes());
            self.cursor = self.cursor.saturating_sub(1);
            if over_bytes {
                report.by_bytes += 1;
            } else {
                report.by_cap += 1;
            }
        }
        report
    }

    /// Whether an entry of `bytes` would dominate the byte budget — the
    /// guard a layer-ADD recording applies before pushing a snapshot: an
    /// entry above half the budget wipes essentially the whole log to keep
    /// itself, and an add is one visible click to reverse anyway. A REMOVE
    /// is never guarded — it is destructive and must always be undoable.
    #[must_use]
    pub fn would_dominate(&self, bytes: usize) -> bool {
        bytes.saturating_mul(2) > self.max_bytes
    }

    /// Ends the coalescing window, so the next transaction starts a fresh
    /// entry.
    ///
    /// Called on drag release, focus loss, mode change, layer change and
    /// selection change — every boundary at which "the same gesture continues"
    /// stops being true.
    pub fn close_coalescing(&mut self) {
        self.open_coalesce = None;
    }

    /// Ends the coalescing window only when it covers a layer-scoped field
    /// (opacity or style) — the clockless "slider drag ended" boundary,
    /// called on every frame with no pointer button down. The attribute
    /// form's Properties window survives, because its Apply *button click*
    /// is itself a pointer release.
    pub fn close_coalescing_for_layer_fields(&mut self) {
        if self.open_coalesce.is_some_and(|key| {
            matches!(key.field, CoalesceField::Opacity | CoalesceField::Style(_))
        }) {
            self.open_coalesce = None;
        }
    }

    /// Drops every entry for `layer` from **both** halves, fixing the cursor and
    /// the byte total.
    ///
    /// Called when a layer is removed, and when its features are re-read from
    /// disk. Keeping the entries as no-ops instead would let an undo re-`Add` a
    /// [`LayerId`] the project no longer holds: the layer panel could not list
    /// it, no `Remove` would ever be queued for it, and the geometry would be
    /// permanently undeletable.
    ///
    /// Returns how many entries were dropped, so a caller can say so.
    pub fn prune_layer(&mut self, layer: LayerId) -> usize {
        let previous = core::mem::take(&mut self.entries);
        let old_cursor = self.cursor;
        let mut cursor = self.cursor;
        let mut bytes = 0_usize;
        let mut dropped = 0_usize;
        let mut kept = VecDeque::with_capacity(previous.len());
        for (index, entry) in previous.into_iter().enumerate() {
            if entry.names_layer(layer) {
                dropped += 1;
                if index < old_cursor {
                    cursor = cursor.saturating_sub(1);
                }
            } else {
                bytes = bytes.saturating_add(entry.estimated_bytes());
                kept.push_back(entry);
            }
        }
        self.entries = kept;
        self.cursor = cursor.min(self.entries.len());
        self.bytes = bytes;
        if dropped > 0 {
            self.open_coalesce = None;
        }
        dropped
    }

    /// Whether there is anything to undo.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    /// Whether there is anything to redo.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        self.cursor < self.entries.len()
    }

    /// The **feature** transaction `undo` would return, when the next step
    /// is one — the historical view most tests read. For a label that
    /// covers both families use [`Self::peek_undo_entry`].
    #[must_use]
    pub fn peek_undo(&self) -> Option<&EditTransaction> {
        self.peek_undo_entry()?.features()
    }

    /// The **feature** transaction `redo` would return, when the next step
    /// is one.
    #[must_use]
    pub fn peek_redo(&self) -> Option<&EditTransaction> {
        self.peek_redo_entry()?.features()
    }

    /// The entry `undo` would return, without moving the cursor — the menu
    /// label and the toolbar tooltip.
    #[must_use]
    pub fn peek_undo_entry(&self) -> Option<&UndoEntry> {
        self.entries.get(self.cursor.checked_sub(1)?)
    }

    /// The entry `redo` would return, without moving the cursor.
    #[must_use]
    pub fn peek_redo_entry(&self) -> Option<&UndoEntry> {
        self.entries.get(self.cursor)
    }

    /// Moves the cursor back and returns the entry to **invert**.
    ///
    /// If the caller cannot apply the inverse it calls [`Self::redo`] and
    /// discards the result — the exact inverse cursor move — so the stack never
    /// lies about what is applied.
    pub fn undo(&mut self) -> Option<UndoEntry> {
        let index = self.cursor.checked_sub(1)?;
        let entry = self.entries.get(index)?.clone();
        self.cursor = index;
        self.open_coalesce = None;
        Some(entry)
    }

    /// Moves the cursor forward and returns the entry to **re-apply**.
    pub fn redo(&mut self) -> Option<UndoEntry> {
        let entry = self.entries.get(self.cursor)?.clone();
        self.cursor += 1;
        self.open_coalesce = None;
        Some(entry)
    }

    /// `(undoable, redoable)` entry counts.
    #[must_use]
    pub fn depth(&self) -> (usize, usize) {
        (self.cursor, self.entries.len() - self.cursor)
    }

    /// The stack's current estimated footprint, in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Folds `next` into `top` when both are a single `Replace` of the same
/// feature of the same layer: `Replace{A,B}` + `Replace{B,C}` becomes
/// `Replace{A,C}`, keeping the first's `selection_before` and the second's
/// `selection_after`.
fn fold_features(top: &mut EditTransaction, next: &EditTransaction) -> bool {
    if top.layer != next.layer {
        return false;
    }
    let (
        [
            FeatureOp::Replace {
                index: top_index_of,
                before,
                ..
            },
        ],
        [
            FeatureOp::Replace {
                index: next_index_of,
                after,
                ..
            },
        ],
    ) = (top.ops.as_slice(), next.ops.as_slice())
    else {
        return false;
    };
    if top_index_of != next_index_of {
        return false;
    }
    let before = before.clone();
    let after = after.clone();
    let index = *top_index_of;
    top.ops = vec![FeatureOp::Replace {
        index,
        before,
        after,
    }];
    top.label = next.label;
    top.selection_after = next.selection_after;
    top.coalesce = next.coalesce;
    true
}

/// Folds `next` into `top` when both set the same layer's opacity or the
/// same layer's style — a slider drag stays one undo step, keeping the
/// first's `before` and the second's `after`.
fn fold_project(top: &mut ProjectTransaction, next: &ProjectTransaction) -> bool {
    let folded = match (&mut top.op, &next.op) {
        (
            ProjectOp::SetOpacity { layer, after, .. },
            ProjectOp::SetOpacity {
                layer: next_layer,
                after: next_after,
                ..
            },
        ) if layer == next_layer => {
            *after = *next_after;
            true
        }
        (
            ProjectOp::SetStyle { layer, after, .. },
            ProjectOp::SetStyle {
                layer: next_layer,
                after: next_after,
                ..
            },
        ) if layer == next_layer => {
            after.clone_from(next_after);
            true
        }
        _ => false,
    };
    if folded {
        top.label = next.label;
        top.coalesce = next.coalesce;
    }
    folded
}
