// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The multi-feature selection model (editing v1.1 stage 2 — see
//! docs/plans/editing-v11.md).
//!
//! [`FeatureSelection`] is the SET the edit state holds; the long-standing
//! [`super::EditSelection`] stays `Copy` and frozen as the **anchor view**
//! every existing consumer keeps reading. Four invariants are enforced by
//! construction, so "the anchor is not selected" is unrepresentable:
//!
//! 1. `features` is ascending and duplicate-free;
//! 2. `features` is never empty;
//! 3. `anchor` is always a member of `features`;
//! 4. a picked vertex always belongs to the anchor.

use super::{EditSelection, VertexRef};

/// Most features one selection may hold — bounds the transaction a
/// multi-delete builds by construction rather than by after-the-fact
/// eviction.
pub const MAX_MULTI_SELECT: usize = 10_000;

/// A non-empty set of selected features with one **anchor** — the feature
/// the handles, the attribute form and the table binding follow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSelection {
    /// Selected feature indices, ascending, duplicate-free, non-empty.
    features: Vec<usize>,
    /// The anchor — always a member of `features`.
    anchor: usize,
    /// The picked vertex of the anchor, when one is picked.
    vertex: Option<VertexRef>,
    /// A marquee-marked set of the ANCHOR's vertices (box-select); empty
    /// when no set is marked. Never survives a feature-set gesture.
    vertex_set: Vec<VertexRef>,
}

impl FeatureSelection {
    /// A single-feature selection — the exact meaning every pre-v1.1
    /// selection had.
    #[must_use]
    pub fn single(feature: usize) -> Self {
        Self {
            features: vec![feature],
            anchor: feature,
            vertex: None,
            vertex_set: Vec::new(),
        }
    }

    /// A single-feature selection with a picked vertex.
    #[must_use]
    pub fn vertex(feature: usize, vertex: VertexRef) -> Self {
        Self {
            features: vec![feature],
            anchor: feature,
            vertex: Some(vertex),
            vertex_set: Vec::new(),
        }
    }

    /// The anchor view — what every single-selection consumer reads.
    #[must_use]
    pub fn anchor_selection(&self) -> EditSelection {
        match self.vertex {
            Some(vertex) => EditSelection::vertex(self.anchor, vertex),
            None => EditSelection::feature(self.anchor),
        }
    }

    /// The selected features, ascending.
    #[must_use]
    pub fn features(&self) -> &[usize] {
        &self.features
    }

    /// The anchor feature.
    #[must_use]
    pub fn anchor(&self) -> usize {
        self.anchor
    }

    /// The anchor's picked vertex, if any.
    #[must_use]
    pub fn picked_vertex(&self) -> Option<VertexRef> {
        self.vertex
    }

    /// Whether `feature` is in the set.
    #[must_use]
    pub fn contains(&self, feature: usize) -> bool {
        self.features.binary_search(&feature).is_ok()
    }

    /// How many features are selected.
    #[must_use]
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Always `false` — the type is non-empty by construction; this exists
    /// so clippy's `len_without_is_empty` contract holds.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Toggles `feature`: adds it (and makes it the anchor) when absent,
    /// removes it when present. Removing the LAST feature — or growing past
    /// [`MAX_MULTI_SELECT`] — returns [`None`] for "the selection is now
    /// empty" / "refused", letting the caller decide what that means.
    ///
    /// Toggling always clears the picked vertex: a set gesture is a
    /// feature-level gesture.
    #[must_use]
    pub fn toggled(&self, feature: usize) -> Option<Self> {
        let mut features = self.features.clone();
        let anchor = match features.binary_search(&feature) {
            Ok(at) => {
                features.remove(at);
                if features.is_empty() {
                    return None;
                }
                if self.anchor == feature {
                    // The anchor left: the nearest remaining member takes
                    // over, preferring the one after it.
                    features
                        .get(at)
                        .or_else(|| features.last())
                        .copied()
                        .unwrap_or(0)
                } else {
                    self.anchor
                }
            }
            Err(at) => {
                if features.len() >= MAX_MULTI_SELECT {
                    return None;
                }
                features.insert(at, feature);
                feature
            }
        };
        Some(Self {
            features,
            anchor,
            vertex: None,
            vertex_set: Vec::new(),
        })
    }

    /// Re-anchors on `feature`, adding it to the set if absent (bounded by
    /// [`MAX_MULTI_SELECT`] — when full, the anchor only moves if the
    /// feature is already a member).
    #[must_use]
    pub fn with_anchor(&self, feature: usize) -> Self {
        let mut features = self.features.clone();
        if let Err(at) = features.binary_search(&feature) {
            if features.len() >= MAX_MULTI_SELECT {
                return self.clone();
            }
            features.insert(at, feature);
        }
        Self {
            features,
            anchor: feature,
            vertex: None,
            vertex_set: Vec::new(),
        }
    }

    /// The same selection with the anchor's picked vertex replaced.
    #[must_use]
    pub fn with_vertex(&self, vertex: Option<VertexRef>) -> Self {
        Self {
            features: self.features.clone(),
            anchor: self.anchor,
            vertex,
            vertex_set: Vec::new(),
        }
    }

    /// Drops every feature index `len` or beyond (the collection shrank
    /// under the selection); [`None`] when nothing survives.
    #[must_use]
    pub fn clamped(&self, len: usize) -> Option<Self> {
        let features: Vec<usize> = self
            .features
            .iter()
            .copied()
            .filter(|&feature| feature < len)
            .collect();
        if features.is_empty() {
            return None;
        }
        let anchor = if features.contains(&self.anchor) {
            self.anchor
        } else {
            *features.last().unwrap_or(&0)
        };
        let vertex = if anchor == self.anchor {
            self.vertex
        } else {
            None
        };
        let vertex_set = if anchor == self.anchor {
            self.vertex_set.clone()
        } else {
            Vec::new()
        };
        Some(Self {
            features,
            anchor,
            vertex,
            vertex_set,
        })
    }

    /// The marquee-marked vertex set of the anchor (empty when none).
    #[must_use]
    pub fn vertex_set(&self) -> &[VertexRef] {
        &self.vertex_set
    }

    /// Drops the picked vertex and every `vertex_set` entry `valid` rejects —
    /// the anchor itself is untouched, and so is feature membership; this is
    /// [`Self::clamped`]'s sibling for the OTHER way a selection goes stale:
    /// the anchor survives a commit, but a delete, an undo or a paste shrank
    /// its geometry under indices the picked vertex or the marked set still
    /// name. Filtering rather than dropping-to-`None` keeps every mark that
    /// still resolves, exactly as [`Self::clamped`] keeps every feature that
    /// still exists.
    #[must_use]
    pub fn clamped_vertices(&self, valid: impl Fn(VertexRef) -> bool) -> Self {
        Self {
            features: self.features.clone(),
            anchor: self.anchor,
            vertex: self.vertex.filter(|&at| valid(at)),
            vertex_set: self
                .vertex_set
                .iter()
                .copied()
                .filter(|&at| valid(at))
                .collect(),
        }
    }

    /// The same selection with the anchor's marked vertex set replaced
    /// (deduplicated; the single picked vertex is cleared — a set and a
    /// pick are different statements).
    #[must_use]
    pub fn with_vertex_set(&self, mut vertex_set: Vec<VertexRef>) -> Self {
        vertex_set.sort_unstable();
        vertex_set.dedup();
        Self {
            features: self.features.clone(),
            anchor: self.anchor,
            vertex: None,
            vertex_set,
        }
    }
}

impl From<EditSelection> for FeatureSelection {
    fn from(selection: EditSelection) -> Self {
        match selection.vertex {
            Some(vertex) => Self::vertex(selection.feature, vertex),
            None => Self::single(selection.feature),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_selection_is_its_own_anchor_view() {
        let selection = FeatureSelection::single(7);
        assert_eq!(selection.features(), &[7]);
        assert_eq!(selection.anchor(), 7);
        assert_eq!(selection.anchor_selection(), EditSelection::feature(7));
        assert!(selection.contains(7));
        assert!(!selection.contains(8));
        assert!(!selection.is_empty());
    }

    #[test]
    fn toggling_adds_ascending_and_moveses_the_anchor_to_the_new_member() {
        let selection = FeatureSelection::single(5)
            .toggled(2)
            .expect("adding keeps the set non-empty");
        assert_eq!(selection.features(), &[2, 5]);
        assert_eq!(selection.anchor(), 2, "the newly added feature anchors");
        let selection = selection.toggled(9).expect("still non-empty");
        assert_eq!(selection.features(), &[2, 5, 9]);
        assert_eq!(selection.anchor(), 9);
    }

    #[test]
    fn toggling_a_member_out_reanchors_and_the_last_one_empties() {
        let selection = FeatureSelection::single(5)
            .toggled(2)
            .and_then(|selection| selection.toggled(9))
            .expect("three members");
        // Remove the anchor (9): the nearest remaining member takes over.
        let selection = selection.toggled(9).expect("two members left");
        assert_eq!(selection.features(), &[2, 5]);
        assert_eq!(selection.anchor(), 5, "prefers the position after");
        // Removing a non-anchor member keeps the anchor.
        let selection = selection.toggled(2).expect("one member left");
        assert_eq!(selection.anchor(), 5);
        assert_eq!(selection.toggled(5), None, "the last member empties it");
    }

    #[test]
    fn a_toggle_clears_the_picked_vertex() {
        let selection = FeatureSelection::vertex(3, VertexRef::new(1));
        assert_eq!(selection.picked_vertex(), Some(VertexRef::new(1)));
        let toggled = selection.toggled(4).expect("non-empty");
        assert_eq!(toggled.picked_vertex(), None);
    }

    #[test]
    fn clamping_drops_stale_indices_and_reanchors() {
        let selection = FeatureSelection::single(1)
            .toggled(4)
            .and_then(|selection| selection.toggled(9))
            .expect("three members");
        let clamped = selection.clamped(5).expect("two survive");
        assert_eq!(clamped.features(), &[1, 4]);
        assert_eq!(clamped.anchor(), 4, "the stale anchor moved to the last");
        assert_eq!(selection.clamped(2).expect("one survives").anchor(), 1);
        assert_eq!(selection.clamped(1), None, "index 1 needs len >= 2");
        assert_eq!(selection.clamped(0), None);
    }

    #[test]
    fn clamped_vertices_drops_only_the_marks_that_stop_resolving() {
        let selection = FeatureSelection::single(3).with_vertex_set(vec![
            VertexRef::new(0),
            VertexRef::new(5),
            VertexRef::new(9),
        ]);
        // A geometry whose only valid index, after some edit, is 0 and 9 —
        // 5 no longer resolves (its path shrank under it).
        let valid = |at: VertexRef| at.index == 0 || at.index == 9;
        let clamped = selection.clamped_vertices(valid);
        assert_eq!(
            clamped.vertex_set(),
            &[VertexRef::new(0), VertexRef::new(9)]
        );
        // Feature membership and the anchor are untouched — this only prunes
        // vertex-level state.
        assert_eq!(clamped.features(), selection.features());
        assert_eq!(clamped.anchor(), selection.anchor());

        // A picked vertex that stops resolving is cleared, not just filtered
        // out of a list.
        let picked = FeatureSelection::vertex(1, VertexRef::new(4));
        assert_eq!(
            picked.clamped_vertices(|at| at.index != 4).picked_vertex(),
            None
        );
        // Nothing to drop is a no-op.
        assert_eq!(
            picked.clamped_vertices(|_| true).picked_vertex(),
            Some(VertexRef::new(4))
        );
    }

    #[test]
    fn the_edit_selection_round_trip_preserves_the_vertex() {
        let single = EditSelection::vertex(2, VertexRef::at(1, 0, 3));
        let set: FeatureSelection = single.into();
        assert_eq!(set.anchor_selection(), single);
    }
}
