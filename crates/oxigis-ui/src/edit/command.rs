// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The reversible command model: three feature operations, the transaction
//! that groups them, and the pure geometry mutators every gesture reduces to.
//!
//! Nothing here touches egui, the GPU or [`crate::OxigisApp`]. [`apply_ops`] is
//! total and all-or-nothing — it either returns the next collection or fails
//! having touched nothing — which is the property the whole undo system rests
//! on.
//!
//! # The three silent data corruptions this module makes unreachable
//!
//! 1. **Ring closure.** A polygon ring is handed to a mutator **open** — its
//!    duplicate closing position popped by the internal `RingCursor`, which
//!    *detects* closure rather than assuming it — and unconditionally re-closed
//!    afterwards, so no mutation can leave a ring open and "dragged handle 0
//!    and the ring came apart" cannot happen.
//! 2. **Altitude.** [`write_lon_lat`] writes elements `0` and `1` of an
//!    existing [`Position`] in place, so a third (altitude) element survives
//!    every edit instead of being dropped by a wholesale replacement.
//! 3. **Stale bounding boxes.** `Feature::bbox`, every nested geometry `bbox`
//!    and `FeatureCollection::bbox` all survive `clone()` and would be
//!    serialized straight into the project file, describing the geometry as it
//!    was *before* the edit. [`scrub_bboxes`] and [`scrub_collection_bbox`] run
//!    on every geometry change.
//!
//! # Geometry coverage
//!
//! Every geometry kind is editable: [`paths_mut`] flattens them all into
//! `(part, ring) -> positions`, so move, insert and delete are one code
//! path. A `GeometryCollection`'s members flatten into `part` with a
//! running counter (identical in [`paths`] and [`paths_mut`], recursing to
//! [`MAX_GEOMETRY_DEPTH`]) — at the top level the six plain kinds number
//! exactly as they did before collections became editable, so no recorded
//! [`crate::edit::VertexRef`] changed meaning. The GPU tile builder splits a
//! *mixed* collection into one drawn feature per geometry family (same
//! depth cap), so edited members draw wherever the layer's style has a rule
//! for their family.

use super::stack::CoalesceKey;
use super::{EditSelection, VertexRef};
use oxigeo::geojson::types::{Feature, FeatureCollection, Geometry, Position, Properties};
use oxigis_core::LayerId;
use oxigis_render::LonLat;

/// One reversible change to one feature of one collection.
///
/// Three variants, no more: every Phase 2 gesture reduces to these, so
/// invertibility is structural rather than per-gesture logic. [`Self::Replace`]
/// covers geometry *and* property edits — the distinction is a UI label, not a
/// data-model one, and a whole-[`Feature`] payload is what makes bbox scrubbing
/// and geometry-kind changes free.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureOp {
    /// Insert `feature` at `index`, shifting later features along.
    Add {
        /// Where the feature lands; `index == len` appends.
        index: usize,
        /// The feature to insert.
        feature: Box<Feature>,
    },
    /// Remove the feature at `index`, shifting later features back.
    Remove {
        /// Which feature to remove.
        index: usize,
        /// The removed feature, kept so the inverse can put it back exactly.
        feature: Box<Feature>,
    },
    /// Swap the feature at `index` for another.
    Replace {
        /// Which feature to replace.
        index: usize,
        /// The feature as it was, kept so the inverse is exact.
        before: Box<Feature>,
        /// The feature as it becomes.
        after: Box<Feature>,
    },
}

impl FeatureOp {
    /// The operation that exactly undoes this one.
    #[must_use]
    pub fn inverted(&self) -> Self {
        match self {
            Self::Add { index, feature } => Self::Remove {
                index: *index,
                feature: feature.clone(),
            },
            Self::Remove { index, feature } => Self::Add {
                index: *index,
                feature: feature.clone(),
            },
            Self::Replace {
                index,
                before,
                after,
            } => Self::Replace {
                index: *index,
                before: after.clone(),
                after: before.clone(),
            },
        }
    }

    /// Roughly how much heap this operation pins, for the undo stack's byte
    /// budget. An estimate, not a measurement: it counts coordinates and
    /// property text, which is where all of the size is.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        let payload = match self {
            Self::Add { feature, .. } | Self::Remove { feature, .. } => feature_bytes(feature),
            Self::Replace { before, after, .. } => feature_bytes(before) + feature_bytes(after),
        };
        size_of::<Self>() + payload
    }

    /// The feature index this operation names.
    #[must_use]
    pub fn index(&self) -> usize {
        match self {
            Self::Add { index, .. } | Self::Remove { index, .. } | Self::Replace { index, .. } => {
                *index
            }
        }
    }
}

/// One undo step: an all-or-nothing group of operations against **one** layer.
///
/// Single-layer by construction, which is what makes per-layer pruning safe:
/// entries for different layers commute, so removing a subsequence leaves every
/// remaining entry individually valid under strict LIFO.
#[derive(Debug, Clone, PartialEq)]
pub struct EditTransaction {
    /// The layer every operation applies to.
    pub layer: LayerId,
    /// Menu/status wording, e.g. `"Move vertex"`.
    pub label: &'static str,
    /// Applied in order; inverted in reverse order.
    pub ops: Vec<FeatureOp>,
    /// Where the selection was before this step.
    pub selection_before: Option<EditSelection>,
    /// Where the selection should land after it. Stored rather than derived, so
    /// undo restores the selection exactly even for a multi-operation step.
    pub selection_after: Option<EditSelection>,
    /// Gestures that may fold into the previous entry share a key; [`None`]
    /// never coalesces and closes any open window.
    pub coalesce: Option<CoalesceKey>,
}

impl EditTransaction {
    /// A single-operation transaction that does not coalesce.
    #[must_use]
    pub fn single(layer: LayerId, label: &'static str, op: FeatureOp) -> Self {
        Self {
            layer,
            label,
            ops: vec![op],
            selection_before: None,
            selection_after: None,
            coalesce: None,
        }
    }

    /// The transaction that exactly undoes this one: the operations reversed
    /// **and** inverted, with the selections swapped.
    #[must_use]
    pub fn inverted(&self) -> Self {
        Self {
            layer: self.layer,
            label: self.label,
            ops: self.ops.iter().rev().map(FeatureOp::inverted).collect(),
            selection_before: self.selection_after,
            selection_after: self.selection_before,
            coalesce: None,
        }
    }

    /// Roughly how much heap this entry pins; see [`FeatureOp::estimated_bytes`].
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        size_of::<Self>()
            + self
                .ops
                .iter()
                .map(FeatureOp::estimated_bytes)
                .sum::<usize>()
    }

    /// The feature indices this transaction touches, ascending and deduplicated
    /// — the set a post-commit validation pass has to re-check.
    #[must_use]
    pub fn touched_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self.ops.iter().map(FeatureOp::index).collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    /// The **post-transaction** indices of features an `Add` or `Replace`
    /// landed — the set a post-commit validation pass has to re-check.
    /// `Remove`s contribute nothing: a removed feature has nothing left to
    /// validate, and features that merely shifted did not change (their
    /// recorded issues travel through the caller's index remap instead).
    /// For a pure-`Remove` transaction — a delete of any size — this is
    /// empty, which is exactly what keeps a 10 000-feature delete from
    /// revalidating 10 000 untouched features.
    #[must_use]
    pub fn touched_after(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .ops
            .iter()
            .enumerate()
            .filter_map(|(at, op)| match op {
                FeatureOp::Remove { .. } => None,
                FeatureOp::Add { index, .. } | FeatureOp::Replace { index, .. } => {
                    remap_index(&self.ops[at + 1..], *index)
                }
            })
            .collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }
}

/// Builds the strictly **descending** `Remove` ops deleting `indices` (any
/// order, duplicates ignored) from `features` — the one op order for which
/// every index is its pre-transaction index and whose inversion re-inserts
/// each feature at exactly its original slot.
///
/// # Errors
///
/// [`EditError::IndexOutOfRange`] naming the first index past the
/// collection; nothing is built partially.
pub fn remove_features_ops(
    features: &FeatureCollection,
    indices: &[usize],
) -> Result<Vec<FeatureOp>, EditError> {
    let len = features.features.len();
    let mut sorted: Vec<usize> = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted
        .into_iter()
        .rev()
        .map(|index| {
            features
                .features
                .get(index)
                .cloned()
                .map(|feature| FeatureOp::Remove {
                    index,
                    feature: Box::new(feature),
                })
                .ok_or(EditError::IndexOutOfRange { index, len })
        })
        .collect()
}

/// Why an edit could not be performed.
///
/// Every variant is a refusal, never a partial application: the operation that
/// produced it left the collection, the project and the undo stack exactly as
/// they were.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditError {
    /// No layer is selected, so there is nothing to edit.
    NoTargetLayer,
    /// The layer is no longer in the project, or is no longer a local vector
    /// layer.
    LayerGone(LayerId),
    /// The layer exists but its features have not been read yet.
    FeaturesNotLoaded(LayerId),
    /// An operation named a feature the collection does not have — the "the
    /// layer changed underneath us" case.
    IndexOutOfRange {
        /// The index the operation named.
        index: usize,
        /// How many features the collection actually holds.
        len: usize,
    },
    /// The feature has no geometry to edit (a null-geometry feature).
    NoGeometry,
    /// There is no such vertex in the feature's geometry.
    BadVertex(VertexRef),
    /// The named geometry kind cannot be edited this way; the payload names it.
    UnsupportedGeometry(&'static str),
    /// A [`Position`] shorter than two elements, or a non-finite coordinate.
    MalformedPosition(VertexRef),
    /// Deleting would leave a geometry with too few positions to be valid;
    /// refusing is the non-destructive answer.
    TooFewVertices {
        /// How many positions would remain.
        have: usize,
        /// How many the geometry needs.
        need: usize,
    },
    /// The edited collection could not be serialized, so nothing was changed.
    Serialize(String),
    /// Adding this property key would push the layer past the attribute table's
    /// column cap; the payload is the cap.
    ColumnCapReached(usize),
    /// The attribute form already holds that key. Overwriting a row the user
    /// cannot see is how attribute data is lost, so the add is refused instead.
    /// An empty payload means the key was blank.
    DuplicateKey(String),
}

impl core::fmt::Display for EditError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoTargetLayer => formatter.write_str("Select a layer to edit first."),
            Self::LayerGone(id) => write!(formatter, "Layer {id} can no longer be edited."),
            Self::FeaturesNotLoaded(id) => {
                write!(formatter, "Layer {id}'s features are not loaded yet.")
            }
            Self::IndexOutOfRange { index, len } => write!(
                formatter,
                "Feature {index} is not in this layer (it holds {len})."
            ),
            Self::NoGeometry => formatter.write_str("That feature has no geometry to edit."),
            Self::BadVertex(at) => write!(
                formatter,
                "There is no vertex at part {}, ring {}, index {}.",
                at.part, at.ring, at.index
            ),
            Self::UnsupportedGeometry(kind) => {
                write!(formatter, "{kind} geometry cannot be edited yet.")
            }
            Self::MalformedPosition(at) => write!(
                formatter,
                "The coordinate at part {}, ring {}, index {} is not a usable position.",
                at.part, at.ring, at.index
            ),
            Self::TooFewVertices { have, need } => write!(
                formatter,
                "That would leave {have} positions where {need} are needed."
            ),
            Self::Serialize(message) => {
                write!(formatter, "The edit could not be stored: {message}")
            }
            Self::ColumnCapReached(cap) => write!(
                formatter,
                "This layer already uses the {cap}-column attribute limit."
            ),
            Self::DuplicateKey(key) if key.is_empty() => {
                formatter.write_str("A property needs a name.")
            }
            Self::DuplicateKey(key) => {
                write!(
                    formatter,
                    "This feature already has a \u{201c}{key}\u{201d}."
                )
            }
        }
    }
}

/// Applies `ops` in order to a copy of `current`.
///
/// Pure, total and all-or-nothing: it builds the next collection or fails
/// without having touched anything. **The** function undo correctness rests on
/// — an operation that fails half-way through a multi-operation transaction
/// would leave a collection no inverse describes.
///
/// The collection-level bounding box is cleared whenever anything is applied:
/// it describes the features as they were, and a stale one would be serialized
/// into the project file. It is not restored by an inverse, because a wrong
/// bbox is worse than an absent one.
///
/// # Errors
///
/// [`EditError::IndexOutOfRange`] when an operation names a feature the
/// collection does not have.
pub fn apply_ops(
    current: &FeatureCollection,
    ops: &[FeatureOp],
) -> Result<FeatureCollection, EditError> {
    // The two bulk shapes the transaction builders mass-produce — a
    // multi-delete's strictly descending `Remove` run, and its inverse, a
    // strictly ascending `Add` run — take one linear pass instead of one
    // `Vec::remove`/`insert` shift per operation. Everything else takes the
    // sequential reference path; the property tests in `tests_bulk` pin the
    // two paths to byte-identical results AND byte-identical errors.
    if descending_remove_run(ops) {
        apply_removes_bulk(current, ops)
    } else if ascending_add_run(ops) {
        apply_adds_bulk(current, ops)
    } else {
        apply_ops_sequential(current, ops)
    }
}

/// Fewest operations for which one linear pass beats per-op shifting. Below
/// this the pass's setup (a mask or a merge) costs more than the shifts it
/// saves — deleting 3 features of a million must not allocate a megabyte.
pub const BULK_MIN_OPS: usize = 8;

/// Whether `ops` is a pure, strictly descending `Remove` run long enough for
/// the retain pass.
///
/// Strictness matters twice: equal indices would make the sequential path
/// remove two *different* features (the second index names the post-shift
/// vector), and strictly descending indices all address the ORIGINAL vector,
/// which is what makes one retain pass equivalent.
fn descending_remove_run(ops: &[FeatureOp]) -> bool {
    if ops.len() < BULK_MIN_OPS {
        return false;
    }
    let mut previous: Option<usize> = None;
    for op in ops {
        let FeatureOp::Remove { index, .. } = op else {
            return false;
        };
        if let Some(previous) = previous
            && *index >= previous
        {
            return false;
        }
        previous = Some(*index);
    }
    true
}

/// Whether `ops` is a pure, strictly ascending `Add` run long enough for the
/// merge pass — the exact inverse shape of a bulk delete, i.e. what undoing
/// one replays.
fn ascending_add_run(ops: &[FeatureOp]) -> bool {
    if ops.len() < BULK_MIN_OPS {
        return false;
    }
    let mut previous: Option<usize> = None;
    for op in ops {
        let FeatureOp::Add { index, .. } = op else {
            return false;
        };
        if let Some(previous) = previous
            && *index <= previous
        {
            return false;
        }
        previous = Some(*index);
    }
    true
}

/// One retain pass over the original vector for a strictly descending
/// `Remove` run.
///
/// Error identity with the sequential path is proved, not hoped: op `k` is
/// checked there against `len − k`, and strict descent gives
/// `index_k ≤ index_0 − k`, so if op 0 passes (`index_0 < len`) every later
/// op passes — the sequential path fails **iff** the first (largest) index is
/// out of range, with exactly the payload written below.
fn apply_removes_bulk(
    current: &FeatureCollection,
    ops: &[FeatureOp],
) -> Result<FeatureCollection, EditError> {
    let len = current.features.len();
    let first = ops.first().map(FeatureOp::index).unwrap_or_default();
    if first >= len {
        return Err(EditError::IndexOutOfRange { index: first, len });
    }
    let mut doomed = vec![false; len];
    for op in ops {
        if let FeatureOp::Remove { index, .. } = op
            && let Some(slot) = doomed.get_mut(*index)
        {
            *slot = true;
        }
    }
    let mut next = current.clone();
    let mut at = 0_usize;
    // `retain` visits in original order, so the survivors keep exactly the
    // relative order repeated `Vec::remove` at descending indices leaves.
    next.features.retain(|_| {
        let keep = !doomed.get(at).copied().unwrap_or(false);
        at += 1;
        keep
    });
    scrub_collection_bbox(&mut next);
    Ok(next)
}

/// One merge pass for a strictly ascending `Add` run.
///
/// Sequentially, add `k` lands at its stated index in the then-current vector
/// and — because every later add targets a strictly higher index — never
/// moves again, so the final vector has added feature `k` at exactly
/// `index_k` with the original features filling the remaining slots in order.
/// The bound check per op is `index_k ≤ len + k`; the first violation is
/// reported with the sequential payload `{ index_k, len: len + k }`.
fn apply_adds_bulk(
    current: &FeatureCollection,
    ops: &[FeatureOp],
) -> Result<FeatureCollection, EditError> {
    let len = current.features.len();
    let mut adds: Vec<(usize, &Feature)> = Vec::with_capacity(ops.len());
    for (applied, op) in ops.iter().enumerate() {
        if let FeatureOp::Add { index, feature } = op {
            if *index > len + applied {
                return Err(EditError::IndexOutOfRange {
                    index: *index,
                    len: len + applied,
                });
            }
            adds.push((*index, feature));
        }
    }
    let mut next = current.clone();
    let originals = core::mem::take(&mut next.features);
    let mut merged: Vec<Feature> = Vec::with_capacity(len + adds.len());
    let mut pending = adds.into_iter().peekable();
    for feature in originals {
        while pending
            .peek()
            .is_some_and(|(index, _)| *index == merged.len())
        {
            if let Some((_, added)) = pending.next() {
                merged.push(added.clone());
            }
        }
        merged.push(feature);
    }
    for (_, added) in pending {
        merged.push(added.clone());
    }
    next.features = merged;
    scrub_collection_bbox(&mut next);
    Ok(next)
}

/// The one-operation-at-a-time reference implementation of [`apply_ops`] —
/// the path every mixed or short transaction takes, and the oracle the bulk
/// paths are property-tested against (`pub(crate)` for exactly that).
pub(crate) fn apply_ops_sequential(
    current: &FeatureCollection,
    ops: &[FeatureOp],
) -> Result<FeatureCollection, EditError> {
    let mut next = current.clone();
    for op in ops {
        let len = next.features.len();
        match op {
            FeatureOp::Add { index, feature } => {
                if *index > len {
                    return Err(EditError::IndexOutOfRange { index: *index, len });
                }
                next.features.insert(*index, (**feature).clone());
            }
            FeatureOp::Remove { index, .. } => {
                if *index >= len {
                    return Err(EditError::IndexOutOfRange { index: *index, len });
                }
                next.features.remove(*index);
            }
            FeatureOp::Replace { index, after, .. } => {
                let Some(slot) = next.features.get_mut(*index) else {
                    return Err(EditError::IndexOutOfRange { index: *index, len });
                };
                *slot = (**after).clone();
            }
        }
    }
    if !ops.is_empty() {
        scrub_collection_bbox(&mut next);
    }
    Ok(next)
}

/// Where the feature at pre-transaction `index` sits after `ops` have been
/// applied, or [`None`] when one of them removed it.
///
/// Walks the operations in application order — the same order
/// [`apply_ops`] uses — so an `Add` below the index shifts it up, a `Remove`
/// below shifts it down, a `Remove` *of* it ends the walk, and a `Replace`
/// moves nothing. This is what lets state that addresses features by index —
/// recorded topology issues, the attribute form's binding — follow a
/// renumbering instead of silently pointing at whatever slid into the slot.
#[must_use]
pub fn remap_index(ops: &[FeatureOp], index: usize) -> Option<usize> {
    let mut current = index;
    for op in ops {
        match op {
            FeatureOp::Add { index, .. } => {
                if *index <= current {
                    current += 1;
                }
            }
            FeatureOp::Remove { index, .. } => {
                if *index == current {
                    return None;
                }
                if *index < current {
                    current -= 1;
                }
            }
            FeatureOp::Replace { .. } => {}
        }
    }
    Some(current)
}

/// Which kind of coordinate sequence a path is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathKind {
    /// A `Point`'s lone position, or a `MultiPoint`'s list of them.
    Points,
    /// A `LineString`, or one member of a `MultiLineString`.
    Line,
    /// One ring of a `Polygon` or `MultiPolygon`, always handed over **open**.
    Ring,
}

/// The fewest positions a path of this kind may hold, measured on the **open**
/// sequence: a `MultiPoint` needs one member, a line two positions, a ring
/// three (four once closed). A delete that would go below this is refused.
#[must_use]
pub fn min_positions(kind: PathKind) -> usize {
    match kind {
        PathKind::Points => 1,
        PathKind::Line => 2,
        PathKind::Ring => 3,
    }
}

/// One editable coordinate sequence of a geometry, as a mutable borrow.
pub struct PathSlot<'a> {
    /// Which `Multi*` member this path belongs to, or `0`.
    pub part: usize,
    /// Which polygon ring this is, or `0`.
    pub ring: usize,
    /// What kind of sequence it is.
    pub kind: PathKind,
    /// The positions themselves.
    coords: PathCoords<'a>,
}

/// How a [`PathSlot`]'s positions are stored.
enum PathCoords<'a> {
    /// A `Point`'s lone position — the one geometry whose coordinates are not a
    /// sequence at all.
    Lone(&'a mut Position),
    /// Everything else.
    Many(&'a mut Vec<Position>),
}

/// Deepest `GeometryCollection` nesting the editor walks — shared with the
/// hit-test, overlay and topology recursions so "what is drawn" and "what is
/// editable" bound out at the same place. Members nested past the cap are
/// simply unaddressable: both [`paths`] and [`paths_mut`] skip them
/// **identically**, which is what keeps the two functions' `(part, ring)`
/// numbering in lock-step.
pub const MAX_GEOMETRY_DEPTH: usize = 8;

/// Takes the next part number off the shared counter.
fn bump_part(next_part: &mut usize) -> usize {
    let part = *next_part;
    *next_part += 1;
    part
}

/// Every editable coordinate sequence of `geometry`, in `(part, ring)` order.
///
/// This is the single mechanism that flattens every geometry kind — a
/// `GeometryCollection`'s members included — into one mutation path. A
/// collection's members are flattened into `part` with a running counter
/// that advances once per part (so at the top level the six plain kinds
/// number exactly as they always have), recursing to
/// [`MAX_GEOMETRY_DEPTH`].
///
/// # Errors
///
/// None today — the `Result` is kept so a future refusal (and the callers'
/// notice paths) need no signature change.
pub fn paths_mut(geometry: &mut Geometry) -> Result<Vec<PathSlot<'_>>, EditError> {
    let mut slots = Vec::new();
    let mut next_part = 0;
    collect_paths_mut(geometry, &mut next_part, 0, &mut slots);
    Ok(slots)
}

/// [`paths_mut`]'s recursive body. `next_part` is the shared per-part
/// counter; `depth` bounds collection nesting.
fn collect_paths_mut<'a>(
    geometry: &'a mut Geometry,
    next_part: &mut usize,
    depth: usize,
    out: &mut Vec<PathSlot<'a>>,
) {
    match geometry {
        Geometry::Point(point) => out.push(PathSlot {
            part: bump_part(next_part),
            ring: 0,
            kind: PathKind::Points,
            coords: PathCoords::Lone(&mut point.coordinates),
        }),
        Geometry::MultiPoint(points) => out.push(PathSlot {
            part: bump_part(next_part),
            ring: 0,
            kind: PathKind::Points,
            coords: PathCoords::Many(&mut points.coordinates),
        }),
        Geometry::LineString(line) => out.push(PathSlot {
            part: bump_part(next_part),
            ring: 0,
            kind: PathKind::Line,
            coords: PathCoords::Many(&mut line.coordinates),
        }),
        Geometry::MultiLineString(lines) => {
            for coords in &mut lines.coordinates {
                out.push(PathSlot {
                    part: bump_part(next_part),
                    ring: 0,
                    kind: PathKind::Line,
                    coords: PathCoords::Many(coords),
                });
            }
        }
        Geometry::Polygon(polygon) => {
            let part = bump_part(next_part);
            for (ring, coords) in polygon.coordinates.iter_mut().enumerate() {
                out.push(PathSlot {
                    part,
                    ring,
                    kind: PathKind::Ring,
                    coords: PathCoords::Many(coords),
                });
            }
        }
        Geometry::MultiPolygon(polygons) => {
            for rings in &mut polygons.coordinates {
                let part = bump_part(next_part);
                for (ring, coords) in rings.iter_mut().enumerate() {
                    out.push(PathSlot {
                        part,
                        ring,
                        kind: PathKind::Ring,
                        coords: PathCoords::Many(coords),
                    });
                }
            }
        }
        Geometry::GeometryCollection(collection) => {
            if depth >= MAX_GEOMETRY_DEPTH {
                return;
            }
            for member in &mut collection.geometries {
                collect_paths_mut(member, next_part, depth + 1, out);
            }
        }
    }
}

/// One coordinate sequence of a geometry, borrowed for reading.
///
/// The read-only twin of [`PathSlot`], with the same `(part, ring)` addressing
/// — and, crucially, the same **open** view of a polygon ring: the duplicate
/// closing position is simply not in [`Self::positions`], so an index into that
/// slice is directly a [`VertexRef::index`] the mutators accept. Hit testing and
/// painting share this one definition of "which coordinates are addressable"
/// with the editing path, which is what keeps "what you can click" and "what you
/// can move" the same set.
#[derive(Debug)]
pub struct PathView<'a> {
    /// Which `Multi*` member this path belongs to, or `0`.
    pub part: usize,
    /// Which polygon ring this is, or `0`.
    pub ring: usize,
    /// What kind of sequence it is.
    pub kind: PathKind,
    /// The positions, with a ring's closing duplicate already excluded.
    pub positions: &'a [Position],
}

/// Every readable coordinate sequence of `geometry`, in `(part, ring)` order.
///
/// The read twin of [`paths_mut`], sharing its flattening — a
/// `GeometryCollection`'s members number into `part` with the same running
/// counter, to the same [`MAX_GEOMETRY_DEPTH`] — so an index a reader (hit
/// testing, painting, the snap index) hands out is exactly the address the
/// mutators accept.
#[must_use]
pub fn paths(geometry: &Geometry) -> Vec<PathView<'_>> {
    let mut views = Vec::new();
    let mut next_part = 0;
    collect_paths(geometry, &mut next_part, 0, &mut views);
    views
}

/// [`paths`]'s recursive body, mirroring [`collect_paths_mut`] exactly.
fn collect_paths<'a>(
    geometry: &'a Geometry,
    next_part: &mut usize,
    depth: usize,
    out: &mut Vec<PathView<'a>>,
) {
    match geometry {
        Geometry::Point(point) => out.push(PathView {
            part: bump_part(next_part),
            ring: 0,
            kind: PathKind::Points,
            positions: core::slice::from_ref(&point.coordinates),
        }),
        Geometry::MultiPoint(points) => out.push(PathView {
            part: bump_part(next_part),
            ring: 0,
            kind: PathKind::Points,
            positions: &points.coordinates,
        }),
        Geometry::LineString(line) => out.push(PathView {
            part: bump_part(next_part),
            ring: 0,
            kind: PathKind::Line,
            positions: &line.coordinates,
        }),
        Geometry::MultiLineString(lines) => {
            for coords in &lines.coordinates {
                out.push(PathView {
                    part: bump_part(next_part),
                    ring: 0,
                    kind: PathKind::Line,
                    positions: coords,
                });
            }
        }
        Geometry::Polygon(polygon) => {
            let part = bump_part(next_part);
            for (ring, coords) in polygon.coordinates.iter().enumerate() {
                out.push(PathView {
                    part,
                    ring,
                    kind: PathKind::Ring,
                    positions: open_ring(coords),
                });
            }
        }
        Geometry::MultiPolygon(polygons) => {
            for rings in &polygons.coordinates {
                let part = bump_part(next_part);
                for (ring, coords) in rings.iter().enumerate() {
                    out.push(PathView {
                        part,
                        ring,
                        kind: PathKind::Ring,
                        positions: open_ring(coords),
                    });
                }
            }
        }
        Geometry::GeometryCollection(collection) => {
            if depth >= MAX_GEOMETRY_DEPTH {
                return;
            }
            for member in &collection.geometries {
                collect_paths(member, next_part, depth + 1, out);
            }
        }
    }
}

/// The vertices whose position differs between two geometries of the SAME
/// shape — how an undo or a redo of a vertex-set move recovers which corners
/// were marked, with nothing stored anywhere.
///
/// [`None`] when the two are not the same shape (a different path count, a
/// different `(part, ring)` address or [`PathKind`], or a different position
/// count in any path): an insert, a delete or a whole-feature replacement is
/// not a translation. What a *set delete's* undo restores instead is
/// [`restored_vertices`]' answer — this function deliberately answers `Some`
/// (possibly empty) for **every** equal-shape pair, which is what makes the
/// two mutually exclusive at the one call site that consults both. [`None`]
/// too when more than `hit::HANDLE_BUDGET` positions moved — a marquee can
/// only mark handles that were *drawn*, so a larger answer is not a marked
/// set and is not worth the allocation.
///
/// A position counts as moved when its first two components are finite in
/// both versions and differ — a non-finite coordinate is never a mark, so a
/// NaN in loaded data cannot invent one (NaN `!=` NaN would otherwise report
/// every untouched NaN position as moved). Altitude is not compared: the
/// vertex mutators never touch it. Ascending by `(part, ring, index)` —
/// `FeatureSelection::vertex_set`'s invariant — because [`paths`] walks in
/// that order.
#[must_use]
pub fn moved_vertices(before: &Geometry, after: &Geometry) -> Option<Vec<VertexRef>> {
    let old = paths(before);
    let new = paths(after);
    if old.len() != new.len() {
        return None;
    }
    let mut moved = Vec::new();
    for (former, latter) in old.iter().zip(&new) {
        if former.part != latter.part
            || former.ring != latter.ring
            || former.kind != latter.kind
            || former.positions.len() != latter.positions.len()
        {
            return None;
        }
        for (index, (was, is)) in former.positions.iter().zip(latter.positions).enumerate() {
            let (Some(&lon_was), Some(&lat_was)) = (was.first(), was.get(1)) else {
                continue;
            };
            let (Some(&lon_is), Some(&lat_is)) = (is.first(), is.get(1)) else {
                continue;
            };
            if !(lon_was.is_finite()
                && lat_was.is_finite()
                && lon_is.is_finite()
                && lat_is.is_finite())
            {
                continue;
            }
            if lon_was != lon_is || lat_was != lat_is {
                if moved.len() >= super::hit::HANDLE_BUDGET {
                    return None;
                }
                moved.push(VertexRef::at(former.part, former.ring, index));
            }
        }
    }
    Some(moved)
}

/// The vertices `after` holds and `before` does not — how an undo of a
/// vertex-set DELETE recovers which corners were marked when they were
/// deleted, with nothing stored anywhere.
///
/// The sibling of [`moved_vertices`], not a replacement: that one answers
/// every equal-shape pair (`Some`, possibly empty), so this is only ever
/// consulted for a pair whose arity changed. What it marks is what the applied
/// step **brings into existence** — undoing a set delete puts the k vertices
/// back with their amber rings on, because those rings are the only affordance
/// saying "this set is live again". A pure deletion (the undo of an insert,
/// or the redo of a delete) therefore answers `Some(vec![])`: marking the
/// surviving neighbours would arm Delete against vertices the user never
/// touched.
///
/// [`None`] — nothing is marked and no marks clause is claimed — when:
///
/// * the path count, a `(part, ring)` address or a [`PathKind`] differs (the
///   forward contract for any future split/join tool: a structural change is
///   not a restoration);
/// * a path kept its length but any position moved (a mixed move-and-insert
///   cannot be aligned, so it must not guess);
/// * a path whose length changed holds a non-finite or malformed coordinate —
///   stricter than [`moved_vertices`]' skip rule, because a skipped position
///   would corrupt the alignment and NaN must never invent a mark;
/// * a path that shrank is not a pure deletion (its survivors are not a
///   subsequence of what was there), or one that grew does not contain the
///   old sequence as a subsequence;
/// * more than [`hit::HANDLE_BUDGET`](super::hit::HANDLE_BUDGET) vertices
///   would be marked — a marquee can only mark handles that were drawn.
///
/// Ascending by `(part, ring, index)` — `FeatureSelection::vertex_set`'s
/// invariant — because [`paths`] walks in that order and each path's extras
/// are recorded in index order. Cost is `O(n + m)` per path: a greedy
/// subsequence walk with a postcondition, never a dynamic-programming
/// alignment, because this runs on the interactive undo path.
///
/// Duplicate coordinates make the *indices* ambiguous but not the answer: the
/// greedy walk resolves to an equivalent set — the same multiset of points is
/// marked either way, so re-deleting the restored set reproduces bit-identical
/// geometry.
#[must_use]
pub fn restored_vertices(before: &Geometry, after: &Geometry) -> Option<Vec<VertexRef>> {
    let old = paths(before);
    let new = paths(after);
    if old.len() != new.len() {
        return None;
    }
    let mut restored: Vec<VertexRef> = Vec::new();
    for (former, latter) in old.iter().zip(&new) {
        if former.part != latter.part || former.ring != latter.ring || former.kind != latter.kind {
            return None;
        }
        if former.positions.len() == latter.positions.len() {
            // Zero delta: nothing was restored here, and nothing may have
            // moved either — a move mixed with an insert has no alignment.
            if !former
                .positions
                .iter()
                .zip(latter.positions)
                .all(|(was, is)| same_lon_lat(was, is))
            {
                return None;
            }
            continue;
        }
        if !all_addressable(former.positions) || !all_addressable(latter.positions) {
            return None;
        }
        if former.positions.len() < latter.positions.len() {
            let extras = extra_positions(former.positions, latter.positions)?;
            if restored.len() + extras.len() > super::hit::HANDLE_BUDGET {
                return None;
            }
            restored.extend(
                extras
                    .into_iter()
                    .map(|index| VertexRef::at(latter.part, latter.ring, index)),
            );
        } else {
            // A pure deletion contributes nothing, but is still checked: a
            // shrink that is not a deletion is a structural change.
            let _deleted = extra_positions(latter.positions, former.positions)?;
        }
    }
    Some(restored)
}

/// The indices of `long` that `short` does not account for, when `short` is a
/// subsequence of `long`.
///
/// A greedy two-pointer walk, which is exact for a subsequence test, plus the
/// postcondition that makes it exact as an *alignment*: `short` must be fully
/// consumed and exactly `long.len() - short.len()` indices recorded, else the
/// two sequences are not one-sided insertions apart and the answer is
/// [`None`].
fn extra_positions(short: &[Position], long: &[Position]) -> Option<Vec<usize>> {
    let delta = long.len().checked_sub(short.len())?;
    let mut extras = Vec::with_capacity(delta);
    let mut taken = 0;
    for (index, position) in long.iter().enumerate() {
        if taken < short.len() && same_lon_lat(&short[taken], position) {
            taken += 1;
        } else {
            extras.push(index);
        }
    }
    (taken == short.len() && extras.len() == delta).then_some(extras)
}

/// Whether both positions name the same point — longitude and latitude only,
/// since the vertex mutators never touch altitude ([`write_lon_lat`]).
///
/// Two malformed positions (fewer than two components) agree with each other
/// and with nothing else, so a damaged coordinate cannot be silently aligned
/// against a well-formed one.
fn same_lon_lat(left: &Position, right: &Position) -> bool {
    match (lon_lat(left), lon_lat(right)) {
        (Some(left), Some(right)) => {
            same_component(left.0, right.0) && same_component(left.1, right.1)
        }
        (None, None) => true,
        _ => false,
    }
}

/// A position's `(lon, lat)`, or [`None`] when it has fewer than two
/// components.
fn lon_lat(position: &Position) -> Option<(f64, f64)> {
    Some((*position.first()?, *position.get(1)?))
}

/// Component equality that survives the two IEEE-754 traps: `-0.0 == 0.0` for
/// finite values, and two non-finite components agree only when they are
/// bit-identical (so NaN never silently equals a different NaN).
fn same_component(left: f64, right: f64) -> bool {
    if left.is_finite() && right.is_finite() {
        left == right
    } else {
        left.to_bits() == right.to_bits()
    }
}

/// Whether every position of a path is a well-formed, finite coordinate — the
/// precondition [`extra_positions`] needs, since a position it cannot compare
/// would shift the whole alignment.
fn all_addressable(positions: &[Position]) -> bool {
    positions.iter().all(|position| {
        lon_lat(position).is_some_and(|(lon, lat)| lon.is_finite() && lat.is_finite())
    })
}

/// A ring without its duplicate closing position, detected rather than assumed
/// — the read-only counterpart of `RingCursor::open`.
#[must_use]
pub fn open_ring(coords: &[Position]) -> &[Position] {
    match coords.split_last() {
        Some((last, head)) if !head.is_empty() && head.first() == Some(last) => head,
        _ => coords,
    }
}

/// A polygon ring, temporarily opened for editing.
///
/// [`Self::open`] **detects** closure rather than assuming it, so an imported
/// unclosed ring is normalized instead of losing its last position;
/// [`Self::finish`] always re-closes, so every mutation leaves the ring closed
/// regardless of how it arrived.
struct RingCursor<'a> {
    /// The ring, with its duplicate closing position popped while open.
    coords: &'a mut Vec<Position>,
    /// Whether the ring arrived closed.
    was_closed: bool,
}

impl<'a> RingCursor<'a> {
    /// Pops the duplicate closing position when `first == last && len >= 2`.
    fn open(coords: &'a mut Vec<Position>) -> Self {
        let was_closed = coords.len() >= 2 && coords.first() == coords.last();
        if was_closed {
            coords.pop();
        }
        Self { coords, was_closed }
    }

    /// The open sequence, for the mutator to work on.
    fn coords_mut(&mut self) -> &mut Vec<Position> {
        self.coords
    }

    /// Re-appends a clone of position `0`. Returns whether the ring had to be
    /// *normalized* — i.e. whether it arrived unclosed, which the topology pass
    /// reports as an issue on the source data rather than as an edit failure.
    fn finish(self) -> bool {
        if let Some(first) = self.coords.first().cloned() {
            self.coords.push(first);
        }
        !self.was_closed
    }
}

/// Runs `body` against the coordinate sequence `at` addresses, bracketing a
/// polygon ring with [`RingCursor`].
///
/// `body` must validate before it mutates: on `Err` the sequence is left exactly
/// as it was, which is what makes each mutator atomic.
fn with_path<R>(
    geometry: &mut Geometry,
    at: VertexRef,
    body: impl FnOnce(PathKind, &mut Vec<Position>) -> Result<R, EditError>,
) -> Result<R, EditError> {
    let mut slots = paths_mut(geometry)?;
    let slot = slots
        .iter_mut()
        .find(|slot| slot.part == at.part && slot.ring == at.ring)
        .ok_or(EditError::BadVertex(at))?;
    let kind = slot.kind;
    match &mut slot.coords {
        PathCoords::Lone(position) => {
            // A `Point` is a one-position sequence for the duration, so every
            // mutator sees the same shape. Working on a clone keeps the
            // original untouched when `body` fails, and a length change is
            // refused outright: a `Point` holds exactly one position.
            let mut buffer = vec![(*position).clone()];
            let result = body(kind, &mut buffer)?;
            if buffer.len() != 1 {
                return Err(EditError::UnsupportedGeometry(
                    "A Point holds exactly one position, so its",
                ));
            }
            **position = buffer.swap_remove(0);
            Ok(result)
        }
        PathCoords::Many(coords) => {
            if kind == PathKind::Ring {
                let mut cursor = RingCursor::open(coords);
                let result = body(kind, cursor.coords_mut());
                // Closed either way — including after a refusal, which restores
                // exactly the ring that arrived.
                let _normalized = cursor.finish();
                result
            } else {
                body(kind, coords)
            }
        }
    }
}

/// Writes `to` into an existing [`Position`] **in place**, preserving any third
/// (altitude) element.
///
/// `at` is carried only so a refusal can name the offending vertex.
///
/// # Errors
///
/// [`EditError::MalformedPosition`] when the position holds fewer than two
/// elements, or when `to` is not finite — a non-finite coordinate would
/// serialize as JSON `null` and make the layer's stored text unreadable.
pub fn write_lon_lat(position: &mut Position, at: VertexRef, to: LonLat) -> Result<(), EditError> {
    if !to.lon.is_finite() || !to.lat.is_finite() {
        return Err(EditError::MalformedPosition(at));
    }
    let mut elements = position.iter_mut();
    match (elements.next(), elements.next()) {
        (Some(lon), Some(lat)) => {
            *lon = to.lon;
            *lat = to.lat;
            Ok(())
        }
        _ => Err(EditError::MalformedPosition(at)),
    }
}

/// Clears `feature`'s own bounding box and every nested geometry's.
pub fn scrub_bboxes(feature: &mut Feature) {
    feature.bbox = None;
    if let Some(geometry) = feature.geometry.as_mut() {
        scrub_geometry_bbox(geometry);
    }
}

/// Clears a collection's bounding box.
pub fn scrub_collection_bbox(collection: &mut FeatureCollection) {
    collection.bbox = None;
}

/// Clears one geometry's bounding box, recursing into a `GeometryCollection`.
pub(crate) fn scrub_geometry_bbox(geometry: &mut Geometry) {
    match geometry {
        Geometry::Point(inner) => inner.bbox = None,
        Geometry::LineString(inner) => inner.bbox = None,
        Geometry::Polygon(inner) => inner.bbox = None,
        Geometry::MultiPoint(inner) => inner.bbox = None,
        Geometry::MultiLineString(inner) => inner.bbox = None,
        Geometry::MultiPolygon(inner) => inner.bbox = None,
        Geometry::GeometryCollection(inner) => {
            inner.bbox = None;
            for member in &mut inner.geometries {
                scrub_geometry_bbox(member);
            }
        }
    }
}

/// Moves the vertex `at` to `to`.
///
/// A ring is opened and re-closed around the write, so moving index `0` moves
/// the closing position with it; the position's altitude survives.
///
/// # Errors
///
/// [`EditError::NoGeometry`], [`EditError::UnsupportedGeometry`],
/// [`EditError::BadVertex`] or [`EditError::MalformedPosition`]; the feature is
/// untouched in every case.
pub fn set_vertex(feature: &mut Feature, at: VertexRef, to: LonLat) -> Result<(), EditError> {
    let geometry = feature.geometry.as_mut().ok_or(EditError::NoGeometry)?;
    with_path(geometry, at, |_kind, coords| {
        let position = coords.get_mut(at.index).ok_or(EditError::BadVertex(at))?;
        write_lon_lat(position, at, to)
    })?;
    scrub_bboxes(feature);
    Ok(())
}

/// Moves every vertex of `moves` in ONE atomic pass — the vertex-set MOVE's
/// mutator.
///
/// Refuse-first: every address is resolved against the read view and every
/// target checked finite **before** a single coordinate is written, so a
/// stale mark (a Ctrl+Z mid-drag, a shape that changed underneath) leaves
/// the feature exactly as it was. One `scrub_bboxes` at the end instead of
/// one per vertex — looping [`set_vertex`] would re-walk and re-scrub the
/// whole geometry per mark, O(n·m) for a marquee that may hold up to the
/// handle budget.
///
/// Degeneration is unreachable by arity: a translation changes no position
/// count, so `TooFewVertices` cannot fire; rings are opened and re-closed
/// per write, so moving index `0` moves the closing duplicate with it.
///
/// # Errors
///
/// [`EditError::NoGeometry`], [`EditError::BadVertex`] or
/// [`EditError::MalformedPosition`]; the feature is untouched in every case.
pub fn set_vertices(feature: &mut Feature, moves: &[(VertexRef, LonLat)]) -> Result<(), EditError> {
    {
        // Pass 1 — validate against the read view (ring-open, the same
        // addressing the mutator uses).
        let geometry = feature.geometry.as_ref().ok_or(EditError::NoGeometry)?;
        let views = paths(geometry);
        for (at, to) in moves {
            if !to.lon.is_finite() || !to.lat.is_finite() {
                return Err(EditError::MalformedPosition(*at));
            }
            let view = views
                .iter()
                .find(|view| view.part == at.part && view.ring == at.ring)
                .ok_or(EditError::BadVertex(*at))?;
            if at.index >= view.positions.len() {
                return Err(EditError::BadVertex(*at));
            }
        }
    }
    let geometry = feature.geometry.as_mut().ok_or(EditError::NoGeometry)?;
    for (at, to) in moves {
        // Pass 1 proved these apply; the `?` stays as defence in depth.
        with_path(geometry, *at, |_kind, coords| {
            let position = coords.get_mut(at.index).ok_or(EditError::BadVertex(*at))?;
            write_lon_lat(position, *at, *to)
        })?;
    }
    scrub_bboxes(feature);
    Ok(())
}

/// Inserts a new position at `after`.
///
/// The new position **takes** index `after.index`, i.e. it lands directly after
/// the vertex at `after.index - 1`; `after.index == len` appends, which is how
/// "insert after the last ring vertex" is expressed. The ring is re-closed
/// afterwards either way.
///
/// # Errors
///
/// [`EditError::NoGeometry`], [`EditError::UnsupportedGeometry`],
/// [`EditError::BadVertex`] (an index past the end of the open sequence) or
/// [`EditError::MalformedPosition`] (a non-finite coordinate).
pub fn insert_vertex(feature: &mut Feature, after: VertexRef, at: LonLat) -> Result<(), EditError> {
    let geometry = feature.geometry.as_mut().ok_or(EditError::NoGeometry)?;
    with_path(geometry, after, |_kind, coords| {
        if !at.lon.is_finite() || !at.lat.is_finite() {
            return Err(EditError::MalformedPosition(after));
        }
        if after.index > coords.len() {
            return Err(EditError::BadVertex(after));
        }
        // A brand new vertex has no altitude to inherit; inventing one from a
        // neighbour would be a guess written into the user's data.
        coords.insert(after.index, vec![at.lon, at.lat]);
        Ok(())
    })?;
    scrub_bboxes(feature);
    Ok(())
}

/// Removes the vertex `at`.
///
/// # Errors
///
/// [`EditError::NoGeometry`], [`EditError::UnsupportedGeometry`],
/// [`EditError::BadVertex`], or [`EditError::TooFewVertices`] when the delete
/// would leave the geometry below [`min_positions`] — refusing is the
/// non-destructive answer, and the feature is untouched.
pub fn remove_vertex(feature: &mut Feature, at: VertexRef) -> Result<(), EditError> {
    let geometry = feature.geometry.as_mut().ok_or(EditError::NoGeometry)?;
    with_path(geometry, at, |kind, coords| {
        if at.index >= coords.len() {
            return Err(EditError::BadVertex(at));
        }
        let need = min_positions(kind);
        let remaining = coords.len().saturating_sub(1);
        if remaining < need {
            return Err(EditError::TooFewVertices {
                have: remaining,
                need,
            });
        }
        coords.remove(at.index);
        Ok(())
    })?;
    scrub_bboxes(feature);
    Ok(())
}

/// Replaces a feature's whole property map.
///
/// Whole-map rather than per-key, because the attribute form edits a buffer and
/// commits it in one go, and because a whole-`Feature` `Replace` is what makes
/// the operation trivially invertible. Bounding boxes are scrubbed for
/// uniformity: every mutator here leaves a feature with no derived state that
/// could describe an older version of it.
pub fn set_properties(feature: &mut Feature, properties: Properties) {
    feature.properties = Some(properties);
    scrub_bboxes(feature);
}

/// Rough heap footprint of a whole collection — the sum of its features'.
pub(crate) fn collection_bytes(collection: &FeatureCollection) -> usize {
    collection.features.iter().map(feature_bytes).sum()
}

/// Rough heap footprint of one feature: coordinates plus property text.
fn feature_bytes(feature: &Feature) -> usize {
    let geometry = feature.geometry.as_ref().map_or(0, geometry_bytes);
    let properties = feature.properties.as_ref().map_or(0, properties_bytes);
    size_of::<Feature>() + geometry + properties
}

/// Rough heap footprint of one geometry.
fn geometry_bytes(geometry: &Geometry) -> usize {
    /// One stored position: its elements plus the `Vec` header.
    fn position_bytes(position: &Position) -> usize {
        size_of::<Position>() + position.len() * size_of::<f64>()
    }
    fn sequence_bytes(coords: &[Position]) -> usize {
        size_of::<Vec<Position>>() + coords.iter().map(position_bytes).sum::<usize>()
    }
    fn rings_bytes(rings: &[Vec<Position>]) -> usize {
        size_of::<Vec<Vec<Position>>>()
            + rings.iter().map(|ring| sequence_bytes(ring)).sum::<usize>()
    }
    match geometry {
        Geometry::Point(point) => position_bytes(&point.coordinates),
        Geometry::MultiPoint(points) => sequence_bytes(&points.coordinates),
        Geometry::LineString(line) => sequence_bytes(&line.coordinates),
        Geometry::MultiLineString(lines) => rings_bytes(&lines.coordinates),
        Geometry::Polygon(polygon) => rings_bytes(&polygon.coordinates),
        Geometry::MultiPolygon(polygons) => polygons
            .coordinates
            .iter()
            .map(|rings| rings_bytes(rings))
            .sum::<usize>(),
        Geometry::GeometryCollection(collection) => {
            collection.geometries.iter().map(geometry_bytes).sum()
        }
    }
}

/// Rough heap footprint of one property map.
fn properties_bytes(properties: &Properties) -> usize {
    properties
        .iter()
        .map(|(key, value)| key.len() + value_bytes(value))
        .sum()
}

/// Rough heap footprint of one JSON value.
fn value_bytes(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            size_of::<serde_json::Value>()
        }
        serde_json::Value::String(text) => size_of::<serde_json::Value>() + text.len(),
        serde_json::Value::Array(items) => {
            size_of::<serde_json::Value>() + items.iter().map(value_bytes).sum::<usize>()
        }
        serde_json::Value::Object(map) => {
            size_of::<serde_json::Value>()
                + map
                    .iter()
                    .map(|(key, value)| key.len() + value_bytes(value))
                    .sum::<usize>()
        }
    }
}
