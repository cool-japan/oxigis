// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Topology validation: the pure geometric predicates, and the
//! [`Geometry`] adapter that gives their answers part/ring provenance.
//!
//! # Two halves, one file
//!
//! The **pure half** works on `&[[f64; 2]]` and knows nothing about `oxigeo`,
//! egui or I/O — [`orient`], [`segments_intersect`], [`signed_area`],
//! [`point_in_ring`], [`validate_line`], [`validate_ring`],
//! [`validate_polygon`]. It is deterministic, allocation-light and directly
//! testable, and a later move into `oxigis-core` would be a file move.
//!
//! The **adapter half** walks a [`Feature`]'s geometry, runs the pure half over
//! every coordinate sequence it finds, and tags each answer with the part, the
//! ring, the ring's role and a representative coordinate to fly to —
//! [`FeatureIssue`], [`validate_feature`], [`validate_collection`],
//! [`describe`], [`severity`].
//!
//! # The self-intersection algorithm, in three passes
//!
//! A naive "test every pair of segments" implementation ships four separate
//! bugs, so the order below is load-bearing:
//!
//! 1. **Closure and compaction.** A ring is checked for `first == last`
//!    ([`TopologyIssue::RingNotClosed`] otherwise) and then worked on *open*.
//!    Consecutive equal positions emit [`TopologyIssue::RepeatedVertex`] and are
//!    dropped, so no zero-length segment ever reaches a later pass and cannot
//!    produce a cascade of degenerate answers. Non-finite positions are dropped
//!    here too, after being reported: nothing downstream should have to reason
//!    about how `NaN` orders.
//! 2. **Spike pass.** Every consecutive triple of compacted positions —
//!    **including the wrap triples `(m-2, m-1, 0)` and `(m-1, 0, 1)` of a closed
//!    ring** — is reported as [`TopologyIssue::Spike`] when it is collinear and
//!    doubles back. This is the shared-endpoint degeneracy a crossing test
//!    structurally cannot see, because it must skip adjacency.
//! 3. **Pairwise pass.** Every `i < j` over compacted segments, **skipping
//!    `j == i + 1` and, for a closed ring, the wrap pair `(0, m-1)`**, goes
//!    through [`segments_intersect`], which handles the all-collinear case
//!    explicitly with a 1-D overlap test on the dominant axis. Without that
//!    branch a segment lying exactly on top of another reports "no
//!    intersection" — the single most common omission.
//!
//! Above [`MAX_SEGMENTS_FOR_SELF_INTERSECTION`] pass 3 is skipped and
//! [`TopologyIssue::Skipped`] reported; passes 1 and 2 are O(n) and always run.
//!
//! # Between rings, not just within one
//!
//! The three passes above are strictly intra-path: a ring is only ever
//! checked against itself. A polygon's rings are additionally checked
//! against **each other** — [`TopologyIssue::HoleOutsideExterior`] (a hole
//! vertex strictly outside the shell), [`TopologyIssue::RingsIntersect`] (a
//! hole's edge crossing the shell, or two holes crossing each other) and
//! [`TopologyIssue::HolesNested`] (one hole's representative point inside
//! another) — drawing from the same pairwise budget the intra-ring passes
//! do, so an adversarial ring count cannot buy unbounded work. A ring or a
//! hole merely **touching** another — a shared vertex, a shared edge — is
//! not reported: real digitizing produces that on purpose, and OGC leaves it
//! valid.
//!
//! # Coordinate space, and the one limitation it carries
//!
//! Stored lon/lat is treated as **planar**. That is the standard GIS convention
//! (`ST_IsValid` on geographic coordinates does the same) and the only
//! camera-independent, deterministic choice available here. The documented
//! consequence: a segment that crosses the antimeridian is nonsense in this
//! frame and may be reported as crossing geometry it does not really cross.
//!
//! # When this runs
//!
//! On commit, over the touched features only, and from the explicit **Validate
//! layer** button with a global segment budget. **Never on load** — real data is
//! full of advisory-grade issues, and an unsolicited "247 problems" on every
//! import is both noise and an O(n²) stall on data the user never touched.
//! Validation **never blocks an edit**: a half-drawn polygon is
//! self-intersecting for most of its life.

use oxigeo::geojson::types::{Feature, FeatureCollection, Geometry, Position};
use oxigis_render::LonLat;

/// Most issues one validation run reports; the rest are dropped.
///
/// Re-exported from [`crate::edit`] rather than defined twice: the notice list
/// and the validation list are the same UI surface with the same reason for a
/// cap, and two constants that must agree is one constant too many.
pub use super::MAX_NOTICES;

/// Segments above which the pairwise pass is skipped for one path.
///
/// `2000² = 4 M` orientation tests is a few milliseconds — the honest ceiling
/// for a naive O(n²) pass at interactive latency.
pub const MAX_SEGMENTS_FOR_SELF_INTERSECTION: usize = 2_000;

/// Total segments the **Validate layer** button will spend on pairwise passes
/// across a whole collection.
///
/// Deliberately far below the sum of the per-path caps: the per-path cap bounds
/// one path at 4 M orientation tests, so this bounds the button at roughly ten
/// such paths — a few hundred milliseconds in the worst case, and untouched by
/// ordinary data, whose rings are two orders of magnitude smaller.
pub const VALIDATE_LAYER_SEGMENT_BUDGET: usize = 20_000;

/// Largest `GeometryCollection` nesting this walks before giving up.
const MAX_GEOMETRY_DEPTH: usize = 8;

/// Takes the next part number off the shared counter — the same flattening
/// [`crate::edit::command::paths`] performs.
fn bump_part(next_part: &mut usize) -> usize {
    let part = *next_part;
    *next_part += 1;
    part
}

/// Fewest positions a `LineString` may hold.
const MIN_LINE_POSITIONS: usize = 2;

/// Fewest positions an **open** ring may hold; four once closed.
const MIN_RING_POSITIONS: usize = 3;

/// Fewest positions a **closed** ring may hold.
const MIN_CLOSED_RING_POSITIONS: usize = 4;

/// Largest longitude magnitude RFC 7946 allows.
const MAX_LONGITUDE_DEG: f64 = 180.0;

/// Largest latitude magnitude RFC 7946 allows.
const MAX_LATITUDE_DEG: f64 = 90.0;

/// What a polygon ring is for.
///
/// `Open` covers everything that is not a ring at all — a `LineString`, a
/// `Point`, a `MultiPoint` — so one provenance type serves every path kind and
/// the winding check has an unambiguous "does not apply" answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RingRole {
    /// Not a ring: a line, a point, or a multi-point.
    #[default]
    Open,
    /// A polygon's first ring.
    Exterior,
    /// One of a polygon's holes.
    Hole,
}

/// How two segments meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingKind {
    /// They cross transversally: each strictly separates the other's endpoints.
    Proper,
    /// They meet at exactly one point, an endpoint of at least one of them.
    Touch,
    /// They are collinear and share an overlap of positive length.
    CollinearOverlap,
}

/// How loudly an issue should be shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Advisory: legal data that is merely unusual, or a check that was not run.
    Info,
    /// Something a consumer of this data is likely to get wrong.
    Warning,
}

/// One thing wrong with one coordinate sequence.
///
/// Every index is an index into the sequence **as the caller passed it** — for a
/// closed ring, into the closed sequence, whose open prefix shares its indices —
/// so a caller never has to undo the compaction this module performs internally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyIssue {
    /// The feature has no geometry, or a geometry with no coordinates.
    EmptyGeometry,
    /// Fewer positions than the geometry kind needs.
    TooFewVertices {
        /// How many positions there are.
        got: usize,
        /// How many there would have to be.
        need: usize,
    },
    /// A ring whose last position is not a copy of its first.
    RingNotClosed,
    /// A position equal to the one before it.
    RepeatedVertex {
        /// Index of the repeat, i.e. of the later of the two.
        index: usize,
    },
    /// Adjacent segments collinear and doubling back — the shared-endpoint
    /// degeneracy a plain crossing test structurally cannot see.
    Spike {
        /// Index of the shared position the path turns around on.
        index: usize,
    },
    /// Two non-adjacent segments of the same path meet.
    SelfIntersection {
        /// Index of the first segment's start position.
        first: usize,
        /// Index of the second segment's start position.
        second: usize,
        /// How they meet.
        kind: CrossingKind,
    },
    /// A ring wound against the RFC 7946 §3.1.6 convention. Advisory only.
    WrongWinding {
        /// The role whose convention was not met.
        role: RingRole,
    },
    /// A hole with a vertex outside its polygon's exterior ring.
    HoleOutsideExterior {
        /// Which ring of the polygon the hole is.
        ring: usize,
        /// The first vertex of that hole found outside.
        index: usize,
    },
    /// A different ring of the same polygon crosses this one transversally —
    /// a hole's edge crossing out of the exterior and back, or two holes
    /// overlapping. A single shared point or a run of shared edge is
    /// deliberately not this: two rings touching or aligning at their
    /// boundary is ordinary, valid data (see `hole_containment`'s doc), so
    /// only a crossing that no amount of intentional edge-sharing produces
    /// counts as a defect here.
    RingsIntersect {
        /// The first ring.
        first_ring: usize,
        /// Index, into the first ring's open sequence, of the segment found
        /// to cross the other ring.
        first_segment: usize,
        /// The second ring.
        second_ring: usize,
        /// Index, into the second ring's open sequence, of the segment found
        /// to cross the first.
        second_segment: usize,
    },
    /// One hole's representative point lies inside another hole of the same
    /// polygon — an interior no renderer can fill unambiguously.
    HolesNested {
        /// The hole found to contain the other.
        outer_ring: usize,
        /// The hole found inside it.
        inner_ring: usize,
    },
    /// A position with a non-finite component, or with fewer than two elements.
    NonFiniteCoordinate {
        /// Index of the position.
        index: usize,
    },
    /// A position outside `|lon| <= 180` / `|lat| <= 90`.
    OutOfRange {
        /// Index of the position.
        index: usize,
    },
    /// The pairwise pass was not run: too many segments.
    Skipped {
        /// How many segments the path has.
        segments: usize,
    },
}

/// How loudly `issue` should be shown.
///
/// [`TopologyIssue::WrongWinding`] is [`Severity::Info`] on purpose: the
/// renderer measures winding on quantized coordinates and never trusts the
/// source, so a ring wound "wrongly" draws correctly here and the report exists
/// for whoever exports the data next. [`TopologyIssue::RepeatedVertex`] and
/// [`TopologyIssue::Skipped`] are informational for the same reason — neither
/// describes geometry that is wrong.
#[must_use]
pub fn severity(issue: &TopologyIssue) -> Severity {
    match issue {
        TopologyIssue::EmptyGeometry
        | TopologyIssue::RepeatedVertex { .. }
        | TopologyIssue::WrongWinding { .. }
        | TopologyIssue::Skipped { .. } => Severity::Info,
        TopologyIssue::TooFewVertices { .. }
        | TopologyIssue::RingNotClosed
        | TopologyIssue::Spike { .. }
        | TopologyIssue::SelfIntersection { .. }
        | TopologyIssue::HoleOutsideExterior { .. }
        | TopologyIssue::RingsIntersect { .. }
        | TopologyIssue::HolesNested { .. }
        | TopologyIssue::NonFiniteCoordinate { .. }
        | TopologyIssue::OutOfRange { .. } => Severity::Warning,
    }
}

// ---------------------------------------------------------------------------
// Pure half
// ---------------------------------------------------------------------------

/// Which side of the directed line `a -> b` the point `c` falls on: `1` to the
/// left (counter-clockwise), `-1` to the right, `0` collinear.
///
/// The epsilon is **extent-relative**, and that is the whole point of this
/// function. The determinant is a product of coordinate *differences* —
/// `(b-a)` and `(c-a)`, each computed exactly for nearby operands (Sterbenz) —
/// so its rounding error grows with how far apart `a`, `b` and `c` actually
/// are, never with how far they sit from the origin. Scaling by absolute
/// coordinate magnitude instead is the classic mistake: a 1 m building corner
/// at longitude 0 and the identical corner at longitude 179 have exactly the
/// same rounding error, but an origin-relative epsilon is ~13 000× bigger at
/// 179 than at 0 for no geometric reason at all, and swallows a real turn as
/// collinear. The rule, stated exactly:
///
/// ```text
/// extent = max(|b.x-a.x|, |b.y-a.y|, |c.x-a.x|, |c.y-a.y|)
/// eps    = max(16 * f64::EPSILON * extent², f64::MIN_POSITIVE)
/// ```
///
/// The floor keeps the epsilon from underflowing to exactly zero for an
/// extent far below anything this application's coordinates produce, where a
/// zero epsilon would turn pure rounding noise into a spurious verdict. The
/// factor 16 is the slack for the four multiplications and two subtractions
/// the determinant costs.
///
/// A non-finite input yields `0`: every comparison against `NaN` is false, and
/// "no opinion" is the only honest answer about a coordinate that is not a
/// number. Callers report such positions separately.
#[must_use]
pub fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> i8 {
    let determinant = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let extent = (b[0] - a[0])
        .abs()
        .max((b[1] - a[1]).abs())
        .max((c[0] - a[0]).abs())
        .max((c[1] - a[1]).abs());
    let epsilon = (16.0 * f64::EPSILON * extent * extent).max(f64::MIN_POSITIVE);
    if determinant > epsilon {
        1
    } else if determinant < -epsilon {
        -1
    } else {
        0
    }
}

/// How segments `a0 -> a1` and `b0 -> b1` meet, or [`None`] when they do not.
///
/// The four orientations decide it. When **all four are zero** the segments are
/// collinear, and no orientation test can distinguish "lying on top of each
/// other" from "disjoint on the same line" — so that case gets its own branch: a
/// 1-D overlap test on whichever axis the first segment extends further along
/// (measuring a vertical segment along `x` would compare two zeros). An overlap
/// of positive length is [`CrossingKind::CollinearOverlap`]; an overlap that is
/// a single shared point is [`CrossingKind::Touch`]; anything else is [`None`].
///
/// Omitting that branch is the classic bug: a segment lying exactly on top of
/// another then reports no intersection at all.
#[must_use]
pub fn segments_intersect(
    a0: [f64; 2],
    a1: [f64; 2],
    b0: [f64; 2],
    b1: [f64; 2],
) -> Option<CrossingKind> {
    let d1 = orient(a0, a1, b0);
    let d2 = orient(a0, a1, b1);
    let d3 = orient(b0, b1, a0);
    let d4 = orient(b0, b1, a1);

    // The explicit all-collinear branch, before anything else can mistake it
    // for "no crossing".
    if d1 == 0 && d2 == 0 && d3 == 0 && d4 == 0 {
        return collinear_overlap(a0, a1, b0, b1);
    }
    // Each segment strictly separates the other's endpoints.
    if d1 * d2 < 0 && d3 * d4 < 0 {
        return Some(CrossingKind::Proper);
    }
    // An endpoint of one lies on the other: a T-junction, or a shared vertex
    // between segments the caller did not exclude as adjacent.
    if (d1 == 0 && within_bounds(a0, a1, b0))
        || (d2 == 0 && within_bounds(a0, a1, b1))
        || (d3 == 0 && within_bounds(b0, b1, a0))
        || (d4 == 0 && within_bounds(b0, b1, a1))
    {
        return Some(CrossingKind::Touch);
    }
    None
}

/// Whether `point`, already known to be collinear with `a -> b`, lies within
/// that segment's bounding box — i.e. on the segment rather than on its
/// extension.
fn within_bounds(a: [f64; 2], b: [f64; 2], point: [f64; 2]) -> bool {
    point[0] >= a[0].min(b[0])
        && point[0] <= a[0].max(b[0])
        && point[1] >= a[1].min(b[1])
        && point[1] <= a[1].max(b[1])
}

/// The collinear branch of [`segments_intersect`]: a 1-D overlap test on the
/// axis along which `a0 -> a1` extends further.
fn collinear_overlap(
    a0: [f64; 2],
    a1: [f64; 2],
    b0: [f64; 2],
    b1: [f64; 2],
) -> Option<CrossingKind> {
    let axis = usize::from((a1[1] - a0[1]).abs() > (a1[0] - a0[0]).abs());
    let a_low = a0[axis].min(a1[axis]);
    let a_high = a0[axis].max(a1[axis]);
    let b_low = b0[axis].min(b1[axis]);
    let b_high = b0[axis].max(b1[axis]);
    let low = a_low.max(b_low);
    let high = a_high.min(b_high);
    if low < high {
        Some(CrossingKind::CollinearOverlap)
    } else if low == high {
        Some(CrossingKind::Touch)
    } else {
        // Also the answer for a non-finite coordinate, where both comparisons
        // are false.
        None
    }
}

/// Twice-the-shoelace-over-two of `ring`: positive counter-clockwise, negative
/// clockwise, zero for a degenerate ring.
///
/// The wrap edge from the last position back to the first is always included, so
/// the result is the same whether `ring` is handed over open or closed — a
/// closed ring's duplicate contributes a zero-area term.
#[must_use]
pub fn signed_area(ring: &[[f64; 2]]) -> f64 {
    let Some(last) = ring.last() else {
        return 0.0;
    };
    let mut previous = *last;
    let mut sum = 0.0;
    for current in ring {
        sum += previous[0] * current[1] - current[0] * previous[1];
        previous = *current;
    }
    sum / 2.0
}

/// Whether `ring` winds counter-clockwise, i.e. whether its
/// [`signed_area`] is positive. A degenerate ring is not counter-clockwise.
#[must_use]
pub fn is_ccw(ring: &[[f64; 2]]) -> bool {
    signed_area(ring) > 0.0
}

/// Whether `point` is inside `ring`, by even-odd ray casting.
///
/// **Convention.** The ray is cast towards `x = -infinity` and each edge is
/// counted with the half-open rule `(current.y > point.y) != (previous.y >
/// point.y)`, so an edge is counted when the ray passes its lower endpoint and
/// not when it passes its upper one. That is what makes the test robust *at
/// vertices*: a ray passing exactly through a shared vertex crosses the two
/// edges that meet there either twice or not at all, never once, so no vertex
/// can flip the answer on its own.
///
/// A point exactly **on** the boundary is deliberately unspecified — it may read
/// as either — because no even-odd rule can be both boundary-exact and
/// vertex-robust in floating point. Callers that must not report a boundary
/// point (hole containment, here) test the boundary separately with the same
/// scale-relative orientation predicate the rest of this module uses.
///
/// The ring may be open or closed: the wrap edge is always included.
#[must_use]
pub fn point_in_ring(point: [f64; 2], ring: &[[f64; 2]]) -> bool {
    let Some(last) = ring.last() else {
        return false;
    };
    let mut previous = *last;
    let mut inside = false;
    for current in ring {
        let current = *current;
        if (current[1] > point[1]) != (previous[1] > point[1]) {
            let span = previous[1] - current[1];
            if span != 0.0 {
                let crossing =
                    current[0] + (point[1] - current[1]) / span * (previous[0] - current[0]);
                if point[0] < crossing {
                    inside = !inside;
                }
            }
        }
        previous = current;
    }
    inside
}

/// Whether `point` lies exactly on one of `ring`'s edges, under the same
/// scale-relative orientation epsilon the rest of this module uses.
fn point_on_ring_boundary(point: [f64; 2], ring: &[[f64; 2]]) -> bool {
    let Some(last) = ring.last() else {
        return false;
    };
    let mut previous = *last;
    for current in ring {
        let current = *current;
        if orient(previous, current, point) == 0 && within_bounds(previous, current, point) {
            return true;
        }
        previous = current;
    }
    false
}

/// The point of segment `a -> b` nearest to `p`, and the clamped parameter
/// `t` in `0..=1` at which it sits (`a` at `0`, `b` at `1`).
///
/// A degenerate segment yields `(a, 0.0)`.
#[must_use]
pub fn nearest_on_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> ([f64; 2], f64) {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let length_sq = dx * dx + dy * dy;
    if length_sq <= 0.0 || !length_sq.is_finite() {
        return (a, 0.0);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq).clamp(0.0, 1.0);
    ([a[0] + t * dx, a[1] + t * dy], t)
}

/// Squared distance from `p` to segment `a -> b`, with both endpoints clamped.
///
/// Squared on purpose: every caller compares it against another distance or a
/// tolerance, and a square root per segment over a whole layer buys nothing.
#[must_use]
pub fn segment_distance_sq(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (nearest, _) = nearest_on_segment(p, a, b);
    let dx = p[0] - nearest[0];
    let dy = p[1] - nearest[1];
    dx * dx + dy * dy
}

/// A coordinate sequence reduced to what the later passes may see, plus the way
/// back to the caller's indices.
#[derive(Debug, Default)]
struct Compacted {
    /// The surviving positions, in order.
    points: Vec<[f64; 2]>,
    /// `origin[k]` is the index `points[k]` had in the input.
    origin: Vec<usize>,
}

/// Records `issue` unless the cap has already been reached.
fn push_capped(out: &mut Vec<TopologyIssue>, issue: TopologyIssue) {
    if out.len() < MAX_NOTICES {
        out.push(issue);
    }
}

/// Reports every non-finite and out-of-range position of `coords`.
fn scan_positions(coords: &[[f64; 2]], out: &mut Vec<TopologyIssue>) {
    for (index, point) in coords.iter().enumerate() {
        if !point[0].is_finite() || !point[1].is_finite() {
            push_capped(out, TopologyIssue::NonFiniteCoordinate { index });
        } else if point[0].abs() > MAX_LONGITUDE_DEG || point[1].abs() > MAX_LATITUDE_DEG {
            push_capped(out, TopologyIssue::OutOfRange { index });
        }
    }
}

/// Pass 1: drops consecutive repeats (reported) and non-finite positions
/// (already reported by [`scan_positions`]).
fn compact(coords: &[[f64; 2]], out: &mut Vec<TopologyIssue>) -> Compacted {
    let mut compacted = Compacted {
        points: Vec::with_capacity(coords.len()),
        origin: Vec::with_capacity(coords.len()),
    };
    for (index, point) in coords.iter().enumerate() {
        if !point[0].is_finite() || !point[1].is_finite() {
            continue;
        }
        if compacted.points.last() == Some(point) {
            push_capped(out, TopologyIssue::RepeatedVertex { index });
            continue;
        }
        compacted.points.push(*point);
        compacted.origin.push(index);
    }
    compacted
}

/// Pass 2: collinear triples that double back.
///
/// `closed` adds the two wrap triples `(m-2, m-1, 0)` and `(m-1, 0, 1)`, which
/// are exactly the ones an open-path loop cannot reach and exactly the ones a
/// ring most often has a spike on — the vertex a user dragged onto its own
/// neighbour while closing the shape.
fn spike_pass(compacted: &Compacted, closed: bool, out: &mut Vec<TopologyIssue>) {
    let count = compacted.points.len();
    if count < 3 {
        return;
    }
    let triples = if closed { count } else { count - 2 };
    for start in 0..triples {
        let a = compacted.points[start];
        let middle = (start + 1) % count;
        let b = compacted.points[middle];
        let c = compacted.points[(start + 2) % count];
        if orient(a, b, c) != 0 {
            continue;
        }
        let incoming = [b[0] - a[0], b[1] - a[1]];
        let outgoing = [c[0] - b[0], c[1] - b[1]];
        if incoming[0] * outgoing[0] + incoming[1] * outgoing[1] < 0.0 {
            push_capped(
                out,
                TopologyIssue::Spike {
                    index: compacted.origin[middle],
                },
            );
        }
    }
}

/// Pass 3: every non-adjacent pair of segments.
///
/// Adjacency is skipped because a shared endpoint is what a path *is*; for a
/// closed ring the pair `(0, m-1)` is adjacent too, across the wrap, and
/// forgetting it makes every legal ring report a self-intersection at vertex 0.
///
/// `wrap_adjacent` says the first and last segments share an endpoint and must
/// be treated as adjacent. It is separate from `closed` because the two facts
/// come apart for a **closed `LineString`** — a loop trail, a circuit, a
/// boundary stored as a line: its segments are the plain open-path ones
/// (`closed == false`), yet segment `0` and segment `m-1` meet at the shared
/// first-equals-last position, and reporting that legal, deliberate touch as a
/// self-intersection would flag every closed line ever imported or snapped
/// shut. JTS `isSimple` makes the same exception.
fn pairwise_pass(
    compacted: &Compacted,
    closed: bool,
    wrap_adjacent: bool,
    out: &mut Vec<TopologyIssue>,
) {
    let count = compacted.points.len();
    let segments = segment_count(count, closed);
    if segments < 3 {
        // With two segments the only pair is adjacent.
        return;
    }
    for i in 0..segments {
        let a0 = compacted.points[i];
        let a1 = compacted.points[(i + 1) % count];
        for j in (i + 2)..segments {
            if wrap_adjacent && i == 0 && j == segments - 1 {
                continue;
            }
            let b0 = compacted.points[j];
            let b1 = compacted.points[(j + 1) % count];
            if let Some(kind) = segments_intersect(a0, a1, b0, b1) {
                push_capped(
                    out,
                    TopologyIssue::SelfIntersection {
                        first: compacted.origin[i],
                        second: compacted.origin[j],
                        kind,
                    },
                );
            }
        }
    }
}

/// How many segments a path of `count` positions has.
fn segment_count(count: usize, closed: bool) -> usize {
    if closed {
        count
    } else {
        count.saturating_sub(1)
    }
}

/// Charges `segments` against the pairwise budget.
///
/// Returns whether the pairwise pass may run; a refusal is always reported as
/// [`TopologyIssue::Skipped`], so a check that did not happen never looks like a
/// check that passed.
fn spend(budget: &mut usize, segments: usize, out: &mut Vec<TopologyIssue>) -> bool {
    if segments < 3 {
        return false;
    }
    if segments > MAX_SEGMENTS_FOR_SELF_INTERSECTION || segments > *budget {
        push_capped(out, TopologyIssue::Skipped { segments });
        return false;
    }
    *budget -= segments;
    true
}

/// [`validate_line`] with the pairwise budget supplied.
fn validate_line_budgeted(coords: &[[f64; 2]], budget: &mut usize, out: &mut Vec<TopologyIssue>) {
    if coords.len() < MIN_LINE_POSITIONS {
        push_capped(
            out,
            TopologyIssue::TooFewVertices {
                got: coords.len(),
                need: MIN_LINE_POSITIONS,
            },
        );
    }
    scan_positions(coords, out);
    let compacted = compact(coords, out);
    spike_pass(&compacted, false, out);
    // A closed line — first and last positions equal — is a legal shape whose
    // end segments share that position; without the wrap-adjacency exception
    // the pairwise pass would report the closure itself as a `Touch`.
    let closes_on_itself =
        compacted.points.len() >= 3 && compacted.points.first() == compacted.points.last();
    let segments = segment_count(compacted.points.len(), false);
    if spend(budget, segments, out) {
        pairwise_pass(&compacted, false, closes_on_itself, out);
    }
}

/// Checks one open coordinate sequence: a `LineString`, or one member of a
/// `MultiLineString`.
///
/// Reports, in order: [`TopologyIssue::TooFewVertices`],
/// [`TopologyIssue::NonFiniteCoordinate`], [`TopologyIssue::OutOfRange`],
/// [`TopologyIssue::RepeatedVertex`], [`TopologyIssue::Spike`], and then either
/// [`TopologyIssue::SelfIntersection`] or [`TopologyIssue::Skipped`].
pub fn validate_line(coords: &[[f64; 2]], out: &mut Vec<TopologyIssue>) {
    let mut budget = usize::MAX;
    validate_line_budgeted(coords, &mut budget, out);
}

/// [`validate_ring`] with the pairwise budget supplied.
fn validate_ring_budgeted(
    coords: &[[f64; 2]],
    role: RingRole,
    budget: &mut usize,
    out: &mut Vec<TopologyIssue>,
) {
    if coords.len() < MIN_LINE_POSITIONS {
        push_capped(
            out,
            TopologyIssue::TooFewVertices {
                got: coords.len(),
                need: MIN_CLOSED_RING_POSITIONS,
            },
        );
        return;
    }
    // Closure is checked first and is only a warning: digitizing auto-closes, so
    // this fires on imported data, and refusing to look any further at a ring
    // that is one position short of legal would hide everything else wrong with
    // it. The rest of the work therefore continues on the open sequence either
    // way.
    let closed = coords.first() == coords.last();
    if !closed {
        push_capped(out, TopologyIssue::RingNotClosed);
    }
    let open = if closed {
        &coords[..coords.len() - 1]
    } else {
        coords
    };
    if open.len() < MIN_RING_POSITIONS {
        push_capped(
            out,
            TopologyIssue::TooFewVertices {
                got: open.len(),
                need: MIN_RING_POSITIONS,
            },
        );
    }
    scan_positions(open, out);
    let mut compacted = compact(open, out);
    // The wrap repeat: an open ring whose first and last positions are equal
    // repeats a vertex across the seam, which the linear scan above cannot see.
    if compacted.points.len() >= 2 && compacted.points.first() == compacted.points.last() {
        if let Some(index) = compacted.origin.last().copied() {
            push_capped(out, TopologyIssue::RepeatedVertex { index });
        }
        compacted.points.pop();
        compacted.origin.pop();
    }
    spike_pass(&compacted, true, out);
    let segments = segment_count(compacted.points.len(), true);
    if spend(budget, segments, out) {
        pairwise_pass(&compacted, true, true, out);
    }
    // Winding last, and measured on the compacted ring: a duplicate position
    // contributes no area, but a non-finite one would make the whole sum `NaN`.
    let area = signed_area(&compacted.points);
    let wrong = compacted.points.len() >= MIN_RING_POSITIONS
        && area != 0.0
        && match role {
            RingRole::Exterior => area < 0.0,
            RingRole::Hole => area > 0.0,
            RingRole::Open => false,
        };
    if wrong {
        push_capped(out, TopologyIssue::WrongWinding { role });
    }
}

/// Checks one polygon ring, closed or not.
///
/// The closure check runs first and produces only a
/// [`TopologyIssue::RingNotClosed`] warning — every later pass then works on the
/// open sequence, so an imported unclosed ring is fully checked rather than
/// dismissed.
///
/// [`TopologyIssue::WrongWinding`] follows RFC 7946 §3.1.6 (exterior rings
/// counter-clockwise, holes clockwise) and is **advisory**: nothing in this
/// application trusts a source's winding.
pub fn validate_ring(coords: &[[f64; 2]], role: RingRole, out: &mut Vec<TopologyIssue>) {
    let mut budget = usize::MAX;
    validate_ring_budgeted(coords, role, &mut budget, out);
}

/// The role of ring `index` of a polygon.
#[must_use]
fn ring_role(index: usize) -> RingRole {
    if index == 0 {
        RingRole::Exterior
    } else {
        RingRole::Hole
    }
}

/// A ring without its duplicate closing position, detected rather than assumed.
fn open_ring(coords: &[[f64; 2]]) -> &[[f64; 2]] {
    match coords.split_last() {
        Some((last, head)) if !head.is_empty() && head.first() == Some(last) => head,
        _ => coords,
    }
}

/// Reports the first vertex of each hole that lies strictly outside the
/// exterior ring.
///
/// One report per hole, not per vertex: a misplaced hole is one fact about one
/// ring, and repeating it for each of its 4 000 vertices would bury every other
/// issue in the list. A vertex exactly *on* the exterior boundary is not
/// outside — holes that share an edge with their exterior are ordinary data.
fn hole_containment(rings: &[Vec<[f64; 2]>], out: &mut Vec<TopologyIssue>) {
    let Some(first) = rings.first() else {
        return;
    };
    let exterior = open_ring(first);
    if exterior.len() < MIN_RING_POSITIONS {
        return;
    }
    for (ring, hole) in rings.iter().enumerate().skip(1) {
        for (index, point) in open_ring(hole).iter().enumerate() {
            if !point[0].is_finite() || !point[1].is_finite() {
                continue;
            }
            if point_in_ring(*point, exterior) || point_on_ring_boundary(*point, exterior) {
                continue;
            }
            push_capped(out, TopologyIssue::HoleOutsideExterior { ring, index });
            break;
        }
    }
}

/// Every ring of a polygon, compacted and closed — the shared input the
/// cross-ring crossing and nesting passes both need, computed once per ring
/// rather than once per pair it takes part in. Index `k` of the result is
/// ring `k` of `rings`: no reordering, no filtering, so a caller may index it
/// with the same ring numbers `rings` itself uses.
fn compact_rings(rings: &[Vec<[f64; 2]>]) -> Vec<Compacted> {
    let mut discarded = Vec::new();
    rings
        .iter()
        .map(|coords| {
            // The per-ring `TooFewVertices` / `RepeatedVertex` / `NonFiniteCoordinate`
            // diagnostics this would produce are already reported by the
            // per-ring pass every caller here also runs; recomputing them a
            // second time would only duplicate the notice list.
            discarded.clear();
            compact(open_ring(coords), &mut discarded)
        })
        .collect()
}

/// The first pair of segments — one from each ring — that cross
/// transversally, as indices into each ring's own compacted sequence, or
/// [`None`] when no such crossing exists.
///
/// A shared point or a collinear run is deliberately not "found" here: two
/// rings sharing an edge or a vertex is ordinary, valid data (see
/// [`hole_containment`]'s doc), and only [`CrossingKind::Proper`] — an edge
/// that actually crosses from one side of the other ring to the other — is a
/// defect two different rings of one polygon can have.
fn find_ring_crossing(a: &Compacted, b: &Compacted) -> Option<(usize, usize)> {
    let count_a = a.points.len();
    let count_b = b.points.len();
    for i in 0..count_a {
        let a0 = a.points[i];
        let a1 = a.points[(i + 1) % count_a];
        for j in 0..count_b {
            let b0 = b.points[j];
            let b1 = b.points[(j + 1) % count_b];
            if segments_intersect(a0, a1, b0, b1) == Some(CrossingKind::Proper) {
                return Some((i, j));
            }
        }
    }
    None
}

/// Reports every pair of different rings of one polygon that cross
/// transversally — the check [`pairwise_pass`] structurally cannot make,
/// since it is called once per ring.
///
/// Charged against `budget`, the same pool the intra-ring pairwise passes
/// draw from, at `first.len() * second.len()` per pair — the true cost of
/// [`find_ring_crossing`]'s worst case. A ring above
/// [`MAX_SEGMENTS_FOR_SELF_INTERSECTION`] is excluded up front, same as the
/// intra-ring pass excludes it, which bounds any one pair's cost at the same
/// 4 M orientations that bounds one path. The pair enumeration itself stops —
/// not merely skips — the moment a pair does not fit: with `budget` capped at
/// a small constant, that keeps a polygon with an adversarial *number* of
/// tiny rings from turning pair enumeration itself into unbounded work, even
/// though each individual pair would be cheap.
fn ring_crossings(compacted: &[Compacted], budget: &mut usize, out: &mut Vec<TopologyIssue>) {
    let usable: Vec<usize> = (0..compacted.len())
        .filter(|&ring| {
            (MIN_RING_POSITIONS..=MAX_SEGMENTS_FOR_SELF_INTERSECTION)
                .contains(&compacted[ring].points.len())
        })
        .collect();
    'pairs: for a in 0..usable.len() {
        for b in (a + 1)..usable.len() {
            let (ring_a, ring_b) = (usable[a], usable[b]);
            let (first, second) = (&compacted[ring_a], &compacted[ring_b]);
            let cost = first.points.len() * second.points.len();
            if cost > *budget {
                push_capped(out, TopologyIssue::Skipped { segments: cost });
                break 'pairs;
            }
            *budget -= cost;
            if let Some((first_index, second_index)) = find_ring_crossing(first, second) {
                push_capped(
                    out,
                    TopologyIssue::RingsIntersect {
                        first_ring: ring_a,
                        first_segment: first.origin[first_index],
                        second_ring: ring_b,
                        second_segment: second.origin[second_index],
                    },
                );
            }
        }
    }
}

/// Reports every hole whose representative point (its own first vertex) lies
/// inside another hole of the same polygon.
///
/// The exterior (ring `0`) never takes part: "inside the exterior" is
/// [`hole_containment`]'s question, not this one's. Charged against `budget`
/// like [`ring_crossings`], at one ring's length per pair tested and with the
/// same "stop enumerating, don't merely skip" behaviour once a pair does not
/// fit — a hole count is exactly as adversary-controlled as a ring count.
fn hole_nesting(compacted: &[Compacted], budget: &mut usize, out: &mut Vec<TopologyIssue>) {
    let holes: Vec<usize> = (1..compacted.len())
        .filter(|&ring| {
            (MIN_RING_POSITIONS..=MAX_SEGMENTS_FOR_SELF_INTERSECTION)
                .contains(&compacted[ring].points.len())
        })
        .collect();
    'pairs: for &outer_ring in &holes {
        for &inner_ring in &holes {
            if inner_ring == outer_ring {
                continue;
            }
            let Some(&representative) = compacted[inner_ring].points.first() else {
                continue;
            };
            let cost = compacted[outer_ring].points.len();
            if cost > *budget {
                push_capped(out, TopologyIssue::Skipped { segments: cost });
                break 'pairs;
            }
            *budget -= cost;
            if point_in_ring(representative, &compacted[outer_ring].points) {
                push_capped(
                    out,
                    TopologyIssue::HolesNested {
                        outer_ring,
                        inner_ring,
                    },
                );
            }
        }
    }
}

/// Checks a whole polygon: every ring in its role, hole containment, and the
/// cross-ring crossing and nesting checks — everything `walk_polygon` does,
/// for callers that want one flat answer about one polygon rather than
/// [`FeatureIssue`] provenance.
///
/// The per-ring issues arrive without ring provenance — [`TopologyIssue`] has
/// nowhere to carry it — so a caller that needs to know *which* ring an issue
/// came from calls [`validate_ring`] per ring itself, exactly as this module's
/// [`validate_feature`] does.
pub fn validate_polygon(rings: &[Vec<[f64; 2]>], out: &mut Vec<TopologyIssue>) {
    if rings.is_empty() {
        push_capped(out, TopologyIssue::EmptyGeometry);
        return;
    }
    let mut budget = usize::MAX;
    for (index, coords) in rings.iter().enumerate() {
        validate_ring_budgeted(coords, ring_role(index), &mut budget, out);
    }
    hole_containment(rings, out);
    if rings.len() >= 2 {
        let compacted = compact_rings(rings);
        ring_crossings(&compacted, &mut budget, out);
        hole_nesting(&compacted, &mut budget, out);
    }
}

// ---------------------------------------------------------------------------
// Adapter half
// ---------------------------------------------------------------------------

/// One [`TopologyIssue`], placed inside a feature.
///
/// `part` is the flattened per-part number (`Multi*` members and
/// `GeometryCollection` members numbering straight through, exactly as
/// [`crate::edit::command::paths`] flattens them), `ring` indexes a polygon
/// ring, and `role` says what that ring is — the same `(part, ring)`
/// addressing [`crate::edit::VertexRef`] uses, so a notice and a vertex
/// handle name the same place.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureIssue {
    /// Index of the feature inside its collection.
    pub feature: usize,
    /// The flattened part number, or `0`.
    pub part: usize,
    /// Which polygon ring, or `0`.
    pub ring: usize,
    /// What that ring is for.
    pub role: RingRole,
    /// What is wrong.
    pub issue: TopologyIssue,
    /// A representative coordinate: where to move the camera when the notice is
    /// clicked. [`None`] when the issue is about the absence of coordinates.
    pub at: Option<LonLat>,
}

/// Records `issue` unless the cap has already been reached.
fn push_feature_issue(out: &mut Vec<FeatureIssue>, issue: FeatureIssue) {
    if out.len() < MAX_NOTICES {
        out.push(issue);
    }
}

/// The first two elements of a GeoJSON position, as a planar point.
///
/// A position shorter than two elements becomes `NaN`, which
/// [`scan_positions`] then reports as [`TopologyIssue::NonFiniteCoordinate`] —
/// the honest answer, and one that keeps every later pass free of a second
/// malformed-input case to reason about.
fn to_xy(position: &Position) -> [f64; 2] {
    [
        position.first().copied().unwrap_or(f64::NAN),
        position.get(1).copied().unwrap_or(f64::NAN),
    ]
}

/// Every position of `coords` as planar points.
fn to_points(coords: &[Position]) -> Vec<[f64; 2]> {
    coords.iter().map(to_xy).collect()
}

/// A finite coordinate as a camera target.
fn fly_to(point: &[f64; 2]) -> Option<LonLat> {
    (point[0].is_finite() && point[1].is_finite()).then(|| LonLat::new(point[0], point[1]))
}

/// Where an issue about `coords` should send the camera.
///
/// Issues that name a vertex point at it; issues about the sequence as a whole
/// point at its first position, which is both cheap and the place a user looking
/// for the shape would start.
fn issue_position(issue: &TopologyIssue, coords: &[[f64; 2]]) -> Option<LonLat> {
    let index = match issue {
        TopologyIssue::RepeatedVertex { index }
        | TopologyIssue::Spike { index }
        | TopologyIssue::NonFiniteCoordinate { index }
        | TopologyIssue::OutOfRange { index }
        | TopologyIssue::HoleOutsideExterior { index, .. } => *index,
        TopologyIssue::SelfIntersection { first, .. } => *first,
        TopologyIssue::TooFewVertices { .. }
        | TopologyIssue::RingNotClosed
        | TopologyIssue::WrongWinding { .. }
        | TopologyIssue::Skipped { .. } => 0,
        // Both name two different rings, so no single `coords` slice can
        // resolve them: `walk_polygon` computes their camera target itself
        // against the whole polygon's converted rings, exactly as it already
        // does for `HoleOutsideExterior` rather than routing it through here.
        TopologyIssue::RingsIntersect { .. } | TopologyIssue::HolesNested { .. } => return None,
        TopologyIssue::EmptyGeometry => return None,
    };
    coords.get(index).and_then(fly_to)
}

/// Where inside a feature a batch of issues came from.
#[derive(Debug, Clone, Copy)]
struct Provenance {
    /// Index of the feature inside its collection.
    feature: usize,
    /// Which `Multi*` or `GeometryCollection` member.
    part: usize,
    /// Which polygon ring.
    ring: usize,
    /// What that ring is for.
    role: RingRole,
}

/// Moves `issues` about `coords` into `out`, tagged with `provenance`.
fn place(
    issues: Vec<TopologyIssue>,
    coords: &[[f64; 2]],
    provenance: Provenance,
    out: &mut Vec<FeatureIssue>,
) {
    for issue in issues {
        let at = issue_position(&issue, coords);
        push_feature_issue(
            out,
            FeatureIssue {
                feature: provenance.feature,
                part: provenance.part,
                ring: provenance.ring,
                role: provenance.role,
                issue,
                at,
            },
        );
    }
}

/// The single-issue case: something is wrong with a whole geometry.
fn place_one(issue: TopologyIssue, provenance: Provenance, out: &mut Vec<FeatureIssue>) {
    push_feature_issue(
        out,
        FeatureIssue {
            feature: provenance.feature,
            part: provenance.part,
            ring: provenance.ring,
            role: provenance.role,
            issue,
            at: None,
        },
    );
}

/// Checks one polygon's rings, keeping each ring's provenance.
fn walk_polygon(
    rings: &[Vec<Position>],
    feature: usize,
    part: usize,
    budget: &mut usize,
    out: &mut Vec<FeatureIssue>,
) {
    let provenance = Provenance {
        feature,
        part,
        ring: 0,
        role: RingRole::Open,
    };
    if rings.is_empty() {
        place_one(TopologyIssue::EmptyGeometry, provenance, out);
        return;
    }
    let converted: Vec<Vec<[f64; 2]>> = rings
        .iter()
        .map(|ring| to_points(ring.as_slice()))
        .collect();
    for (ring, coords) in converted.iter().enumerate() {
        let role = ring_role(ring);
        let mut issues = Vec::new();
        validate_ring_budgeted(coords, role, budget, &mut issues);
        place(
            issues,
            coords,
            Provenance {
                feature,
                part,
                ring,
                role,
            },
            out,
        );
    }
    // Everything below is a whole-polygon question, not a single ring's:
    // `TopologyIssue` has nowhere to carry two rings' worth of provenance, so
    // each answer's `(ring, role, at)` is worked out here instead of through
    // `issue_position`, exactly as it always has been for
    // `HoleOutsideExterior`.
    let mut issues = Vec::new();
    hole_containment(&converted, &mut issues);
    // A lone ring has no partner to cross or nest with, and compacting it
    // again here would be pure repeated work over its full vertex count —
    // `hole_containment` above is the only whole-polygon check a one-ring
    // polygon can ever have anything to say.
    if converted.len() >= 2 {
        let compacted = compact_rings(&converted);
        ring_crossings(&compacted, budget, &mut issues);
        hole_nesting(&compacted, budget, &mut issues);
    }
    for issue in issues {
        let (ring, role, at) = match &issue {
            TopologyIssue::HoleOutsideExterior { ring, index } => (
                *ring,
                RingRole::Hole,
                converted
                    .get(*ring)
                    .and_then(|hole| hole.get(*index))
                    .and_then(fly_to),
            ),
            TopologyIssue::RingsIntersect {
                first_ring,
                first_segment,
                ..
            } => (
                *first_ring,
                ring_role(*first_ring),
                converted
                    .get(*first_ring)
                    .and_then(|ring| ring.get(*first_segment))
                    .and_then(fly_to),
            ),
            TopologyIssue::HolesNested {
                outer_ring,
                inner_ring,
            } => (
                *outer_ring,
                RingRole::Hole,
                converted
                    .get(*inner_ring)
                    .and_then(|hole| hole.first())
                    .and_then(fly_to),
            ),
            // `Skipped`, from a cross-ring pass that ran out of budget: it
            // names a pair, not a single ring, so — like an issue about a
            // whole geometry — it is attributed to no ring in particular.
            _ => (0, RingRole::Open, None),
        };
        push_feature_issue(
            out,
            FeatureIssue {
                feature,
                part,
                ring,
                role,
                issue,
                at,
            },
        );
    }
}

/// Checks one geometry, recursing into `Multi*` members and, best-effort, into
/// `GeometryCollection` members.
///
/// `next_part` is the same running per-part counter
/// [`crate::edit::command::paths`] flattens with — one bump per part,
/// `GeometryCollection` members numbering straight through — so a
/// [`FeatureIssue::part`] and a [`crate::edit::VertexRef::part`] name the
/// same place, nesting included. Nesting deeper than [`MAX_GEOMETRY_DEPTH`]
/// is simply not walked (and consumes no part numbers, exactly like the
/// editor's own skip) — a self-referential collection is not representable
/// in GeoJSON, but a maliciously deep one is.
fn walk_geometry(
    geometry: &Geometry,
    feature: usize,
    next_part: &mut usize,
    depth: usize,
    budget: &mut usize,
    out: &mut Vec<FeatureIssue>,
) {
    // An issue about an *empty* geometry has no addressable part; it is
    // stamped with the current counter without consuming a number, matching
    // the editor's zero-path yield for the same value.
    let empty_provenance = Provenance {
        feature,
        part: *next_part,
        ring: 0,
        role: RingRole::Open,
    };
    match geometry {
        Geometry::Point(point) => {
            let provenance = Provenance {
                feature,
                part: bump_part(next_part),
                ring: 0,
                role: RingRole::Open,
            };
            let coords = [to_xy(&point.coordinates)];
            let mut issues = Vec::new();
            scan_positions(&coords, &mut issues);
            place(issues, &coords, provenance, out);
        }
        Geometry::MultiPoint(points) => {
            if points.coordinates.is_empty() {
                place_one(TopologyIssue::EmptyGeometry, empty_provenance, out);
                return;
            }
            let provenance = Provenance {
                feature,
                part: bump_part(next_part),
                ring: 0,
                role: RingRole::Open,
            };
            let coords = to_points(&points.coordinates);
            let mut issues = Vec::new();
            scan_positions(&coords, &mut issues);
            place(issues, &coords, provenance, out);
        }
        Geometry::LineString(line) => {
            if line.coordinates.is_empty() {
                place_one(TopologyIssue::EmptyGeometry, empty_provenance, out);
                return;
            }
            let provenance = Provenance {
                feature,
                part: bump_part(next_part),
                ring: 0,
                role: RingRole::Open,
            };
            let coords = to_points(&line.coordinates);
            let mut issues = Vec::new();
            validate_line_budgeted(&coords, budget, &mut issues);
            place(issues, &coords, provenance, out);
        }
        Geometry::MultiLineString(lines) => {
            if lines.coordinates.is_empty() {
                place_one(TopologyIssue::EmptyGeometry, empty_provenance, out);
                return;
            }
            for line in &lines.coordinates {
                let part = bump_part(next_part);
                let coords = to_points(line);
                let mut issues = Vec::new();
                validate_line_budgeted(&coords, budget, &mut issues);
                place(
                    issues,
                    &coords,
                    Provenance {
                        feature,
                        part,
                        ring: 0,
                        role: RingRole::Open,
                    },
                    out,
                );
            }
        }
        Geometry::Polygon(polygon) => {
            let part = bump_part(next_part);
            walk_polygon(&polygon.coordinates, feature, part, budget, out);
        }
        Geometry::MultiPolygon(polygons) => {
            if polygons.coordinates.is_empty() {
                place_one(TopologyIssue::EmptyGeometry, empty_provenance, out);
                return;
            }
            for rings in &polygons.coordinates {
                let part = bump_part(next_part);
                walk_polygon(rings, feature, part, budget, out);
            }
        }
        Geometry::GeometryCollection(collection) => {
            if collection.geometries.is_empty() {
                place_one(TopologyIssue::EmptyGeometry, empty_provenance, out);
                return;
            }
            if depth >= MAX_GEOMETRY_DEPTH {
                return;
            }
            for inner in &collection.geometries {
                walk_geometry(inner, feature, next_part, depth + 1, budget, out);
            }
        }
    }
}

/// [`validate_feature`] with the collection-wide pairwise budget supplied.
fn validate_feature_into(
    index: usize,
    feature: &Feature,
    budget: &mut usize,
    out: &mut Vec<FeatureIssue>,
) {
    let Some(geometry) = feature.geometry.as_ref() else {
        push_feature_issue(
            out,
            FeatureIssue {
                feature: index,
                part: 0,
                ring: 0,
                role: RingRole::Open,
                issue: TopologyIssue::EmptyGeometry,
                at: None,
            },
        );
        return;
    };
    let mut next_part = 0;
    walk_geometry(geometry, index, &mut next_part, 0, budget, out);
}

/// Every topology issue of one feature, with part and ring provenance.
///
/// Bounded at [`MAX_SEGMENTS_FOR_SELF_INTERSECTION`] **per feature**, shared
/// across every part and ring it has — not per path. A per-path cap alone
/// bounds one ring's own pairwise work but not the sum over a multi-part
/// feature's rings, and this is the per-commit path: a coastline or an
/// administrative boundary stored as one MultiPolygon with thousands of
/// rings is ordinary data, and moving one of its vertices must cost the same
/// either way. Paths that do not fit the remaining budget report
/// [`TopologyIssue::Skipped`], exactly as the whole-collection pass does.
/// This budget is local to one call, never the collection-wide one: a
/// budget shared with features nobody edited would make one commit's result
/// depend on another's.
#[must_use]
pub fn validate_feature(index: usize, feature: &Feature) -> Vec<FeatureIssue> {
    let mut out = Vec::new();
    let mut budget = MAX_SEGMENTS_FOR_SELF_INTERSECTION;
    validate_feature_into(index, feature, &mut budget, &mut out);
    out
}

/// What one whole-collection validation run produced.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectionValidation {
    /// The issues found, oldest feature first, capped at [`MAX_NOTICES`].
    pub issues: Vec<FeatureIssue>,
    /// Whether issues were actually dropped to stay under the cap. Reported
    /// rather than inferred from `issues.len() == MAX_NOTICES` at the call
    /// site: the list is capped by construction, so a full list alone cannot
    /// distinguish "exactly [`MAX_NOTICES`] issues, all shown" from "more
    /// exist" — and a status line claiming truncation that did not happen is a
    /// claim the code cannot back up.
    pub truncated: bool,
}

/// Every topology issue of a whole collection, capped at [`MAX_NOTICES`].
///
/// `budget` is the total number of segments the pairwise passes may spend across
/// the collection; paths that do not fit report [`TopologyIssue::Skipped`]
/// instead of being silently passed. Passes 1 and 2 always run on everything.
///
/// Each feature validates into its own list, so a drop against the collection
/// cap is *observed* rather than silently swallowed inside the capped
/// accumulator. One residual over-approximation remains: a single
/// feature that alone fills its per-feature list to [`MAX_NOTICES`] reports
/// `truncated` even if it had exactly that many issues — a far narrower window
/// than the whole-collection ambiguity this replaces, and in the honest
/// direction (claiming possible truncation, never denying real truncation).
#[must_use]
pub fn validate_collection(features: &FeatureCollection, budget: usize) -> CollectionValidation {
    let mut out = Vec::new();
    let mut truncated = false;
    let mut remaining = budget;
    for (index, feature) in features.features.iter().enumerate() {
        let mut fresh = Vec::new();
        validate_feature_into(index, feature, &mut remaining, &mut fresh);
        if fresh.len() >= MAX_NOTICES {
            truncated = true;
        }
        for issue in fresh {
            if out.len() < MAX_NOTICES {
                out.push(issue);
            } else {
                truncated = true;
                break;
            }
        }
        if out.len() >= MAX_NOTICES && truncated {
            break;
        }
    }
    CollectionValidation {
        issues: out,
        truncated,
    }
}

/// Where inside the feature `issue` is, as a short prefix — `""`, `"ring 1"`,
/// `"part 2 hole 1"`.
fn location(issue: &FeatureIssue) -> String {
    let ring = match issue.role {
        RingRole::Exterior => Some(format!("ring {}", issue.ring)),
        RingRole::Hole => Some(format!("hole {}", issue.ring)),
        RingRole::Open => None,
    };
    match (issue.part, ring) {
        (0, None) => String::new(),
        (0, Some(ring)) => ring,
        (part, None) => format!("part {part}"),
        (part, Some(ring)) => format!("part {part} {ring}"),
    }
}

/// One short sentence naming what is wrong and where, for the Validation list.
///
/// Deliberately not a [`core::fmt::Display`] implementation: this wording is a
/// UI string with a UI's constraints (short enough to truncate well in a
/// 380-point window), not the canonical rendering of the value.
#[must_use]
pub fn describe(issue: &FeatureIssue) -> String {
    let body = match &issue.issue {
        TopologyIssue::EmptyGeometry => "has no geometry".to_string(),
        TopologyIssue::TooFewVertices { got, need } => {
            format!("has {got} vertices where {need} are needed")
        }
        TopologyIssue::RingNotClosed => "is not closed".to_string(),
        TopologyIssue::RepeatedVertex { index } => format!("repeats vertex {index}"),
        TopologyIssue::Spike { index } => format!("doubles back at vertex {index}"),
        TopologyIssue::SelfIntersection {
            first,
            second,
            kind,
        } => {
            let verb = match kind {
                CrossingKind::Proper => "self-intersects",
                CrossingKind::Touch => "touches itself",
                CrossingKind::CollinearOverlap => "overlaps itself",
            };
            format!("{verb} (segments {first}, {second})")
        }
        TopologyIssue::WrongWinding { role } => match role {
            RingRole::Exterior => {
                "winds clockwise; RFC 7946 wants an exterior ring counter-clockwise".to_string()
            }
            RingRole::Hole => {
                "winds counter-clockwise; RFC 7946 wants a hole clockwise".to_string()
            }
            RingRole::Open => "winds against the RFC 7946 convention".to_string(),
        },
        TopologyIssue::HoleOutsideExterior { index, .. } => {
            format!("vertex {index} lies outside the exterior ring")
        }
        TopologyIssue::RingsIntersect { second_ring, .. } => {
            format!("crosses ring {second_ring}")
        }
        TopologyIssue::HolesNested { inner_ring, .. } => {
            format!("contains hole {inner_ring}")
        }
        TopologyIssue::NonFiniteCoordinate { index } => {
            format!("vertex {index} is not a finite coordinate")
        }
        TopologyIssue::OutOfRange { index } => {
            format!("vertex {index} is outside \u{b1}180\u{b0} / \u{b1}90\u{b0}")
        }
        TopologyIssue::Skipped { segments } => {
            format!("self-intersection check skipped \u{2014} {segments} segments")
        }
    };
    let location = location(issue);
    if location.is_empty() {
        body
    } else {
        format!("{location} {body}")
    }
}
