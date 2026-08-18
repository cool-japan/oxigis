// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The `"simplify"` built-in tool: Douglas-Peucker vertex thinning, plus the
//! ring-legality and ring-simplicity guards that keep a thinned polygon from
//! becoming a shape the tile pipeline silently drops.
//!
//! Split out of [`super`] under the 2 000-line rule.

use std::sync::Arc;

use oxigeo::geojson::types::{
    Feature, FeatureCollection, Geometry, GeometryCollection, LineString, MultiLineString,
    MultiPolygon, Polygon, Position,
};
use oxigis_core::{CoreError, ToolContext, ToolExecutor};

use super::job::FeatureSink;

/// The name of [`SimplifyTool`]'s tolerance parameter, matching the descriptor
/// [`oxigis_core::builtin_registry`] registers for `simplify`.
pub(super) const TOLERANCE_PARAM: &str = "tolerance_deg";

/// The `"simplify"` built-in tool: Douglas-Peucker vertex thinning, applied to
/// every feature's geometry.
pub(super) struct SimplifyTool {
    /// The target layer's parsed features.
    pub(super) features: Arc<FeatureCollection>,
}

impl ToolExecutor for SimplifyTool {
    fn run(&self, context: &ToolContext) -> Result<serde_json::Value, CoreError> {
        super::job::drain(Arc::clone(&self.features), sink(context)?)
    }
}

/// [`SimplifyTool`]'s per-feature accumulator, with the run's tolerance
/// already validated out of `context`.
///
/// # Errors
///
/// Returns [`CoreError::InvalidParameter`] when `tolerance_deg` is missing,
/// not a number, or not a finite non-negative one — see [`tolerance_from`].
pub(super) fn sink(context: &ToolContext) -> Result<Box<dyn FeatureSink>, CoreError> {
    Ok(Box::new(SimplifySink {
        tolerance: tolerance_from(context)?,
        simplified: Vec::new(),
        kept_whole: 0,
    }))
}

/// The accumulator behind [`sink`]: one thinned feature per input feature,
/// plus a count of the ones kept at full detail.
#[derive(Debug)]
struct SimplifySink {
    /// The run's validated tolerance, in degrees.
    tolerance: f64,
    /// The thinned features built so far, in input order.
    simplified: Vec<Feature>,
    /// Features whose geometry could not be thinned safely and came through
    /// unchanged — reported once at the end.
    kept_whole: usize,
}

impl FeatureSink for SimplifySink {
    fn absorb(&mut self, feature: &Feature) {
        // Cloning the whole feature (rather than rebuilding it from
        // geometry + properties + id) also carries `bbox`, `crs` and any
        // foreign members across. The `bbox` stays truthful because
        // Douglas-Peucker only ever *keeps a subset* of the original
        // vertices, so the stored box still contains the simplified
        // geometry — merely less tightly than before.
        let mut next = feature.clone();
        if let Some(geometry) = feature.geometry.as_ref() {
            match simplify_geometry(geometry, self.tolerance) {
                Ok(thinned) if simplified_rings_valid(&thinned) => {
                    next.geometry = Some(thinned);
                }
                Ok(_) => {
                    // Genuinely reachable in production, not merely
                    // defensive: `simplify_polygon_rings` already
                    // guarantees ring *length* at Polygon/MultiPolygon
                    // granularity, but not ring *simplicity* —
                    // Douglas-Peucker can fold two non-adjacent edges
                    // across each other, and `simplified_rings_valid`'s
                    // `ring_is_simple` check is the only place that
                    // catches it. Either failure gets the same
                    // conservative response: keep the feature at full
                    // detail rather than ship an invalid ring.
                    self.kept_whole += 1;
                    tracing::warn!(
                        "oxigis-ui: simplify kept a feature's original geometry \
                         (the simplified ring was too short or self-intersecting)",
                    );
                }
                Err(error) => {
                    // Reachable for a `LineString`/`MultiLineString`
                    // thinned below two positions (`LineString::new`
                    // refuses it) or a `Polygon`/`MultiPolygon` member
                    // whose *exterior* ring collapsed below four
                    // positions (`simplify_polygon_rings` reports that as
                    // a synthetic error rather than shipping an illegal
                    // ring). Kept whole either way — the conservative
                    // choice for a tool the user can simply re-run with a
                    // smaller tolerance.
                    self.kept_whole += 1;
                    tracing::warn!(
                        %error,
                        "oxigis-ui: simplify kept a feature's original geometry",
                    );
                }
            }
        }
        self.simplified.push(next);
    }

    fn finish(self: Box<Self>) -> Result<serde_json::Value, CoreError> {
        if self.kept_whole > 0 {
            // `warn!`, not `debug!`: the FeatureCollection result channel
            // has no room for a "kept n features whole" notice (see
            // `centroid`'s matching comment), so this is the only trace
            // of it that survives a release build's default log filter.
            tracing::warn!(
                kept_whole = self.kept_whole,
                "oxigis-ui: simplify left features at full detail",
            );
        }
        Ok(serde_json::to_value(FeatureCollection::new(
            self.simplified,
        ))?)
    }
}

/// RFC 7946 §3.1.6: a closed linear ring's minimum length.
const MIN_CLOSED_RING_POSITIONS: usize = 4;

/// Replaces `oxigeo::geojson::utils::simplify_geometry`'s recursive,
/// per-level-reallocating implementation: applies [`douglas_peucker`] to
/// every line/ring `geometry` carries, dispatching by variant exactly as the
/// delegate did for `Point`/`MultiPoint` (pass through untouched — there is
/// no line to thin) and `LineString`/`MultiLineString` (rebuild through the
/// matching `oxigeo` constructor, which refuses a result thinned below two
/// positions exactly as before, via the same `Err`).
///
/// `Polygon`/`MultiPolygon` differ from the delegate: see
/// [`simplify_polygon_rings`] for the ring-granularity fix this closes — a
/// hole that collapses is dropped rather than reverting the whole feature,
/// and only a collapsed *exterior* reports the [`oxigeo::geojson::GeoJsonError`]
/// this function's `Err` arm carries. A `MultiPolygon` member whose exterior
/// collapses keeps that one member at full detail instead (an island
/// silently disappearing is a worse surprise than one member staying
/// detailed), so the whole `MultiPolygon` arm never itself returns `Err` for
/// that reason.
fn simplify_geometry(geometry: &Geometry, tolerance: f64) -> oxigeo::geojson::Result<Geometry> {
    match geometry {
        Geometry::Point(point) => Ok(Geometry::Point(point.clone())),
        Geometry::MultiPoint(points) => Ok(Geometry::MultiPoint(points.clone())),
        Geometry::LineString(line) => {
            let coords = douglas_peucker(&line.coordinates, tolerance);
            Ok(Geometry::LineString(LineString::new(coords)?))
        }
        Geometry::MultiLineString(lines) => {
            let coords: Vec<_> = lines
                .coordinates
                .iter()
                .map(|line| douglas_peucker(line, tolerance))
                .collect();
            Ok(Geometry::MultiLineString(MultiLineString::new(coords)?))
        }
        Geometry::Polygon(polygon) => {
            let rings = simplify_polygon_rings(&polygon.coordinates, tolerance)
                .ok_or_else(collapsed_exterior_error)?;
            Ok(Geometry::Polygon(Polygon::new(rings)?))
        }
        Geometry::MultiPolygon(polygons) => {
            let mut members = Vec::with_capacity(polygons.coordinates.len());
            for rings in &polygons.coordinates {
                match simplify_polygon_rings(rings, tolerance) {
                    Some(simplified) => members.push(simplified),
                    None => {
                        tracing::warn!(
                            "oxigis-ui: simplify kept one MultiPolygon member at full \
                             detail (its exterior ring collapsed below four positions)",
                        );
                        members.push(rings.clone());
                    }
                }
            }
            Ok(Geometry::MultiPolygon(MultiPolygon::new(members)?))
        }
        Geometry::GeometryCollection(collection) => {
            let simplified: oxigeo::geojson::Result<Vec<_>> = collection
                .geometries
                .iter()
                .map(|member| simplify_geometry(member, tolerance))
                .collect();
            Ok(Geometry::GeometryCollection(GeometryCollection::new(
                simplified?,
            )?))
        }
    }
}

/// The [`oxigeo::geojson::GeoJsonError`] [`simplify_geometry`]'s `Polygon`
/// arm reports when [`simplify_polygon_rings`] refuses an exterior ring —
/// reusing the same error-propagation channel `oxigeo`'s own constructors
/// use, so [`SimplifySink::absorb`]'s `Err` arm handles this exactly like a
/// constructor refusal.
fn collapsed_exterior_error() -> oxigeo::geojson::GeoJsonError {
    oxigeo::geojson::GeoJsonError::InvalidCoordinates {
        message: "the exterior ring collapsed below four positions".to_string(),
        position: None,
    }
}

/// Simplifies one polygon's ring list at **ring granularity**: the exterior
/// must survive as a legal ring (`>=` [`MIN_CLOSED_RING_POSITIONS`]) or the
/// whole polygon is refused ([`None`]); a hole that collapses below that
/// length is dropped instead of dragging the exterior's simplification down
/// with it. Before this, one over-simplified hole reverted the *whole*
/// feature — exterior included — to full detail, defeating the tool for that
/// polygon even though only the hole was ever a problem.
fn simplify_polygon_rings(rings: &[Vec<Position>], tolerance: f64) -> Option<Vec<Vec<Position>>> {
    let exterior = douglas_peucker(rings.first()?, tolerance);
    if exterior.len() < MIN_CLOSED_RING_POSITIONS {
        return None;
    }
    let mut simplified_rings = Vec::with_capacity(rings.len());
    simplified_rings.push(exterior);
    for hole in rings.iter().skip(1) {
        let hole = douglas_peucker(hole, tolerance);
        if hole.len() >= MIN_CLOSED_RING_POSITIONS {
            simplified_rings.push(hole);
        } else {
            tracing::warn!(
                "oxigis-ui: simplify dropped a hole ring that collapsed below four positions",
            );
        }
    }
    Some(simplified_rings)
}

/// Iterative Douglas-Peucker: an explicit `(start, end)` index-range stack
/// over the original slice plus a keep-mask, materialised into the result
/// once at the end. Replaces `oxigeo`'s recursive `douglas_peucker`
/// (oxigeo-geojson `src/utils/simplify.rs`), which rebuilds and
/// heap-allocates its whole output at *every* recursion level
/// (`coords.to_vec()` at each leaf, `left.extend(right)` at each internal
/// node) — Θ(n log n) individual allocations on a balanced split, worse on
/// the unbalanced ones dense real-world data produces — and whose recursion
/// depth is bounded by nothing in this crate. The `pending` stack here is
/// heap-allocated, not call-stack-allocated, so it grows however deep the
/// data demands without risking the thread's fixed-size stack.
///
/// Produces the identical *set* of kept positions as the recursive
/// algorithm: same tie-breaking, since `distance > max_distance` is strict
/// so the first point at the largest distance wins either way. A caller
/// cannot tell the two apart by their output, only by how much each
/// allocates getting there.
fn douglas_peucker(coords: &[Position], tolerance: f64) -> Vec<Position> {
    let len = coords.len();
    if len <= 2 {
        return coords.to_vec();
    }
    let mut keep = vec![false; len];
    keep[0] = true;
    keep[len - 1] = true;
    // Each entry is a range whose two endpoints are already kept; the
    // vertex furthest from the chord between them either gets kept too
    // (splitting the range into two smaller ones, pushed back) or the whole
    // range collapses to just those two endpoints, adding nothing.
    let mut pending: Vec<(usize, usize)> = vec![(0, len - 1)];
    while let Some((start, end)) = pending.pop() {
        if end <= start + 1 {
            continue;
        }
        let first = &coords[start];
        let last = &coords[end];
        let mut max_distance = 0.0_f64;
        let mut max_index = start;
        for (offset, point) in coords[(start + 1)..end].iter().enumerate() {
            let distance = perpendicular_distance(point, first, last);
            if distance > max_distance {
                max_distance = distance;
                max_index = start + 1 + offset;
            }
        }
        if max_distance > tolerance {
            keep[max_index] = true;
            pending.push((start, max_index));
            pending.push((max_index, end));
        }
    }
    coords
        .iter()
        .zip(keep)
        .filter(|(_, kept)| *kept)
        .map(|(position, _)| position.clone())
        .collect()
}

/// Perpendicular distance from `point` to the infinite line through
/// `line_start`/`line_end`, or the plain point-to-point distance when the
/// two coincide (a zero-length chord has no direction to be perpendicular
/// to). A position with fewer than two components contributes `0.0`.
/// Mirrors the private helper behind the `oxigeo` recursive implementation
/// this file replaces, so a run's kept-vertex set is unaffected by the
/// switch — only its allocation count is.
fn perpendicular_distance(point: &Position, line_start: &Position, line_end: &Position) -> f64 {
    if point.len() < 2 || line_start.len() < 2 || line_end.len() < 2 {
        return 0.0;
    }
    let (x, y) = (point[0], point[1]);
    let (x1, y1) = (line_start[0], line_start[1]);
    let (x2, y2) = (line_end[0], line_end[1]);
    let dx = x2 - x1;
    let dy = y2 - y1;
    if dx == 0.0 && dy == 0.0 {
        let dist_x = x - x1;
        let dist_y = y - y1;
        return (dist_x * dist_x + dist_y * dist_y).sqrt();
    }
    let numerator = (dy * x - dx * y + x2 * y1 - y2 * x1).abs();
    let denominator = (dx * dx + dy * dy).sqrt();
    numerator / denominator
}

/// The largest ring [`ring_is_simple`] sweeps pairwise before giving up and
/// assuming it is simple. The sweep is `O(n^2)`; a **zero** tolerance run
/// leaves a dense coastline's ring at its original size (see
/// [`douglas_peucker`]'s module docs on that scale — hundreds of thousands
/// of positions is realistic), and a full sweep at that size would cost
/// minutes to hours on the frame thread. Above the cap,
/// [`simplified_rings_valid`] falls back to exactly the length-only coverage
/// every ring had before this guard existed — a bound on worst-case cost,
/// not a regression for the common case the guard targets (a simplified
/// ring, which is by construction small).
const MAX_SELF_INTERSECTION_CHECK_VERTICES: usize = 2_000;

/// Whether a closed ring's edges are free of *proper* self-intersection —
/// two non-adjacent edges crossing at a point interior to both.
///
/// Deliberately not a check for any weaker kind of overlap: two edges that
/// merely touch — every ring-adjacent pair, by construction, plus any pair
/// that happens to share a coordinate value through a repeated vertex — are
/// not proper crossings, and flagging them would reject rings
/// Douglas-Peucker legitimately produces (a ring can end up with a repeated
/// vertex without ever crossing itself). [`segments_properly_cross`]'s
/// strict sign test gives exactly that distinction for free: a shared or
/// touching point always makes one of its four orientation products exactly
/// zero, which a strict `< 0.0` never counts as a crossing.
///
/// `O(n^2)` pairwise, skipped (assumed simple) above
/// [`MAX_SELF_INTERSECTION_CHECK_VERTICES`] — see that constant's docs.
fn ring_is_simple(ring: &[Position]) -> bool {
    let n = ring.len();
    if !(MIN_CLOSED_RING_POSITIONS..=MAX_SELF_INTERSECTION_CHECK_VERTICES).contains(&n) {
        return true;
    }
    // `ring` is closed (`ring[0] == ring[n-1]`), so edge `i` runs
    // `ring[i] -> ring[i+1]` for `i` in `0..edge_count`; edge `edge_count-1`
    // is the closing edge back to `ring[0]`.
    let edge_count = n - 1;
    for i in 0..edge_count {
        for j in (i + 1)..edge_count {
            // Ring-adjacent edges share an endpoint by construction
            // (`j == i+1`), as does the wraparound pair (`i == 0`,
            // `j == edge_count-1`, both touching `ring[0]`) — neither is a
            // self-intersection to report.
            if j == i + 1 || (i == 0 && j == edge_count - 1) {
                continue;
            }
            if segments_properly_cross(&ring[i], &ring[i + 1], &ring[j], &ring[j + 1]) {
                return false;
            }
        }
    }
    true
}

/// Whether segment `a1`–`a2` and segment `b1`–`b2` cross at a point interior
/// to both, via the standard four-orientation test: the segments cross iff
/// `a1`/`a2` fall on opposite sides of line `b1`-`b2` *and* `b1`/`b2` fall on
/// opposite sides of line `a1`-`a2`. Strict (`< 0.0`, never `<=`) so that any
/// touching, collinear-overlapping, or shared-endpoint configuration — every
/// one of which drives at least one of the four products to exactly zero —
/// reads as "not crossing" rather than "crossing"; see [`ring_is_simple`]'s
/// docs for why that distinction matters here.
fn segments_properly_cross(a1: &Position, a2: &Position, b1: &Position, b2: &Position) -> bool {
    let d1 = orientation(b1, b2, a1);
    let d2 = orientation(b1, b2, a2);
    let d3 = orientation(a1, a2, b1);
    let d4 = orientation(a1, a2, b2);
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// Twice the signed area of the triangle `origin -> a -> b`: positive for a
/// counter-clockwise turn, negative for clockwise, zero when collinear (which
/// includes `a` or `b` coinciding with `origin`). `0.0` for a position with
/// fewer than two components rather than panicking — the same conservative
/// fallback [`perpendicular_distance`] uses for the same input shape, so a
/// malformed vertex contributes no crossing rather than indexing out of
/// bounds. Not shared with [`super::hull`]'s `cross`: that helper takes
/// `[f64; 2]` arrays (already-validated hull vertices), not raw
/// [`Position`]s that may carry anything a parsed feature handed it.
fn orientation(origin: &Position, a: &Position, b: &Position) -> f64 {
    if origin.len() < 2 || a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let (ox, oy) = (origin[0], origin[1]);
    let (ax, ay) = (a[0], a[1]);
    let (bx, by) = (b[0], b[1]);
    (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
}

/// Whether every polygon ring of a simplified geometry is safe to ship: a
/// legal linear ring (four or more positions, RFC 7946 §3.1.6) *and* free of
/// self-intersection ([`ring_is_simple`]).
///
/// The length half is defense in depth — [`simplify_polygon_rings`] already
/// guarantees it for `Polygon`/`MultiPolygon` at ring granularity (an
/// invalid exterior refuses the whole polygon before this is ever reached;
/// an invalid hole is dropped, never returned) — but the simplicity half is
/// the *only* guard against its own failure mode: Douglas-Peucker gives no
/// non-self-intersection guarantee (dropping a vertex can fold two
/// previously non-adjacent edges across each other), and nothing upstream of
/// this function catches it. Closure needs no re-check: [`douglas_peucker`]
/// always keeps a sequence's first and last positions, so a ring that
/// arrived closed leaves closed.
fn simplified_rings_valid(geometry: &Geometry) -> bool {
    let ring_ok =
        |ring: &Vec<Position>| ring.len() >= MIN_CLOSED_RING_POSITIONS && ring_is_simple(ring);
    match geometry {
        Geometry::Polygon(polygon) => polygon.coordinates.iter().all(ring_ok),
        Geometry::MultiPolygon(polygons) => polygons.coordinates.iter().flatten().all(ring_ok),
        Geometry::GeometryCollection(collection) => {
            collection.geometries.iter().all(simplified_rings_valid)
        }
        Geometry::Point(_)
        | Geometry::LineString(_)
        | Geometry::MultiPoint(_)
        | Geometry::MultiLineString(_) => true,
    }
}

/// Reads [`SimplifyTool`]'s tolerance out of a run's parameters.
///
/// A missing or non-numeric value is a [`CoreError::InvalidParameter`], not a
/// silent fallback to the descriptor's default: the panel always sends the
/// field, so anything else means a caller built the context by hand and should
/// hear about it.
fn tolerance_from(context: &ToolContext) -> Result<f64, CoreError> {
    let value = context
        .params
        .get(TOLERANCE_PARAM)
        .ok_or_else(|| invalid_tolerance("missing required parameter"))?;
    let tolerance = value
        .as_f64()
        .ok_or_else(|| invalid_tolerance("expected a number"))?;
    validate_tolerance(tolerance)
}

/// Rejects a tolerance that is not a finite, non-negative distance.
///
/// Zero is allowed and means "change nothing": Douglas-Peucker only drops a
/// vertex whose perpendicular offset is *strictly greater* than the tolerance,
/// so `0.0` keeps every vertex that is not exactly on its chord.
///
/// The non-finite arm is unreachable through [`tolerance_from`] as long as
/// `serde_json` is built without `arbitrary_precision` (a `Number` cannot hold
/// NaN or an infinity, so `as_f64` has already ruled them out) — it is kept
/// because this function is also the validation seam a hand-built `f64` goes
/// through, and because that feature flag is enabled by workspace
/// dependencies elsewhere in the OxiGeo ecosystem.
fn validate_tolerance(tolerance: f64) -> Result<f64, CoreError> {
    if !tolerance.is_finite() {
        return Err(invalid_tolerance("expected a finite number"));
    }
    if tolerance < 0.0 {
        return Err(invalid_tolerance("must not be negative"));
    }
    Ok(tolerance)
}

/// Builds a [`CoreError::InvalidParameter`] against `tolerance_deg`.
fn invalid_tolerance(reason: &str) -> CoreError {
    CoreError::InvalidParameter {
        name: TOLERANCE_PARAM.to_string(),
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing_exec::fixtures::{
        parse, point_feature, run_to_collection, tolerance_context,
    };
    use oxigeo::geojson::types::{FeatureId, Point};

    /// A line whose interior vertices never stray more than 0.001 from the
    /// straight chord between its endpoints.
    const ZIGZAG: &str = r#"{"type":"FeatureCollection","features":[
        {"type":"Feature","id":42,"properties":{"name":"trail"},
         "geometry":{"type":"LineString","coordinates":
            [[0,0],[1,0.001],[2,-0.001],[3,0.001],[4,-0.001],[5,0]]}}
    ]}"#;

    #[test]
    fn simplify_collapses_a_zig_zag_but_keeps_its_endpoints_properties_and_id() {
        let tool = SimplifyTool {
            features: Arc::new(parse(ZIGZAG)),
        };
        let result = run_to_collection(&tool, &tolerance_context(0.01));

        assert_eq!(result.features.len(), 1);
        let feature = &result.features[0];
        let Some(Geometry::LineString(line)) = feature.geometry.as_ref() else {
            panic!("a simplified LineString must stay a LineString");
        };
        assert!(
            line.coordinates.len() < 6,
            "the tolerance must drop vertices, got {}",
            line.coordinates.len()
        );
        assert_eq!(line.coordinates.first(), Some(&vec![0.0, 0.0]));
        assert_eq!(line.coordinates.last(), Some(&vec![5.0, 0.0]));
        assert_eq!(
            feature.get_property("name"),
            Some(&serde_json::json!("trail"))
        );
        assert_eq!(feature.id, Some(FeatureId::Number(42)));
    }

    #[test]
    fn a_zero_tolerance_keeps_every_vertex() {
        let tool = SimplifyTool {
            features: Arc::new(parse(ZIGZAG)),
        };
        let result = run_to_collection(&tool, &tolerance_context(0.0));
        let Some(Geometry::LineString(line)) = result.features[0].geometry.as_ref() else {
            panic!("a simplified LineString must stay a LineString");
        };
        assert_eq!(
            line.coordinates.len(),
            6,
            "zero tolerance drops only vertices exactly on their chord"
        );
    }

    #[test]
    fn simplify_passes_a_point_through_untouched() {
        let features = FeatureCollection::new(vec![point_feature(139.767, 35.681)]);
        let tool = SimplifyTool {
            features: Arc::new(features),
        };
        // A tolerance far larger than the whole geometry still cannot thin a
        // single position.
        let result = run_to_collection(&tool, &tolerance_context(100.0));
        assert_eq!(
            result.features[0].geometry,
            Some(Geometry::Point(
                Point::new_2d(139.767, 35.681).expect("valid point")
            ))
        );
    }

    #[test]
    fn simplify_passes_through_a_feature_whose_geometry_it_cannot_rebuild() {
        // A one-position LineString is malformed but parses (the derived
        // `Deserialize` runs no validation), and `LineString::new` then refuses
        // to rebuild it — the pass-through-unchanged path.
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{"name":"broken"},
                 "geometry":{"type":"LineString","coordinates":[[1,2]]}}
            ]}"#,
        );
        let tool = SimplifyTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &tolerance_context(0.5));

        assert_eq!(result.features.len(), 1, "the feature must not be dropped");
        let Some(Geometry::LineString(line)) = result.features[0].geometry.as_ref() else {
            panic!("the original geometry must come through unchanged");
        };
        assert_eq!(line.coordinates, vec![vec![1.0, 2.0]]);
        assert_eq!(
            result.features[0].get_property("name"),
            Some(&serde_json::json!("broken"))
        );
    }

    #[test]
    fn simplify_keeps_a_polygon_whole_when_its_ring_would_collapse() {
        // A ~50 m building footprint at the default 0.001-degree tolerance:
        // every vertex sits within tolerance of the degenerate chord, so
        // Douglas-Peucker collapses the ring to `[[0,0],[0,0]]`. `oxigeo`'s
        // `Polygon::new` accepts that (only `validate` checks ring length), so
        // without the ring guard the tool would emit an RFC-invalid two-position
        // ring that the tile pipeline silently drops — a layer whose table
        // lists every feature and whose map shows none.
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{"name":"hut"},
                 "geometry":{"type":"Polygon","coordinates":
                    [[[0,0],[0.00045,0],[0.00045,0.00045],[0,0.00045],[0,0]]]}}
            ]}"#,
        );
        let tool = SimplifyTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &tolerance_context(0.001));

        assert_eq!(result.features.len(), 1, "the feature must not be dropped");
        let Some(Geometry::Polygon(polygon)) = result.features[0].geometry.as_ref() else {
            panic!("the polygon must come through as a polygon");
        };
        assert_eq!(
            polygon.coordinates[0].len(),
            5,
            "the original ring must be kept whole, not collapsed: {polygon:?}"
        );
    }

    #[test]
    fn simplify_keeps_a_thin_strip_whole_rather_than_emit_a_three_position_ring() {
        // A long, thin strip: the far vertex survives the first split, but the
        // ring still thins to three positions — one short of a legal linear
        // ring. The guard must catch this shape too, not only the total
        // collapse.
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},
                 "geometry":{"type":"Polygon","coordinates":
                    [[[0,0],[5,0.0002],[10,0.0005],[5,0.0007],[0,0]]]}}
            ]}"#,
        );
        let tool = SimplifyTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &tolerance_context(0.001));
        let Some(Geometry::Polygon(polygon)) = result.features[0].geometry.as_ref() else {
            panic!("the polygon must come through as a polygon");
        };
        assert!(
            polygon.coordinates[0].len() >= 4,
            "no emitted ring may fall below four positions: {polygon:?}"
        );
    }

    #[test]
    fn simplify_still_thins_a_polygon_that_survives_as_a_legal_ring() {
        // The guard must not turn into "polygons are never simplified": a ring
        // with a redundant near-collinear vertex thins to the square and stays
        // a legal five-position closed ring.
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},
                 "geometry":{"type":"Polygon","coordinates":
                    [[[0,0],[1,0.0001],[2,0],[2,2],[0,2],[0,0]]]}}
            ]}"#,
        );
        let tool = SimplifyTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &tolerance_context(0.01));
        let Some(Geometry::Polygon(polygon)) = result.features[0].geometry.as_ref() else {
            panic!("the polygon must come through as a polygon");
        };
        assert_eq!(
            polygon.coordinates[0].len(),
            5,
            "the redundant vertex must be dropped, the ring kept legal: {polygon:?}"
        );
        assert_eq!(
            polygon.coordinates[0].first(),
            polygon.coordinates[0].last(),
            "the ring must stay closed"
        );
    }

    #[test]
    fn simplify_drops_a_collapsed_hole_but_still_thins_a_fine_exterior() {
        // The exterior has one redundant near-collinear vertex (the same
        // shape the test above uses); the hole is a ~0.0003-degree speck
        // that collapses to two identical positions at the same 0.01
        // tolerance. Before the ring-granularity fix, the collapsed hole
        // reverted the *whole* feature — exterior included — to full,
        // unsimplified detail.
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":
                    [[[0,0],[1,0.0001],[2,0],[2,2],[0,2],[0,0]],
                     [[0.5,0.5],[0.5003,0.5],[0.5003,0.5003],[0.5,0.5003],[0.5,0.5]]]}}
            ]}"#,
        );
        let tool = SimplifyTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &tolerance_context(0.01));
        let Some(Geometry::Polygon(polygon)) = result.features[0].geometry.as_ref() else {
            panic!("the polygon must come through as a polygon");
        };
        assert_eq!(
            polygon.coordinates.len(),
            1,
            "the collapsed hole must be dropped, not kept as an illegal ring: {polygon:?}"
        );
        assert_eq!(
            polygon.coordinates[0].len(),
            5,
            "the exterior must still be thinned, not reverted whole: {polygon:?}"
        );
    }

    #[test]
    fn simplify_keeps_one_multi_polygon_member_whole_while_thinning_its_sibling() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"MultiPolygon","coordinates":[
                    [[[0,0],[1,0.0001],[2,0],[2,2],[0,2],[0,0]]],
                    [[[10,10],[10.00045,10],[10.00045,10.00045],[10,10.00045],[10,10]]]
                ]}}
            ]}"#,
        );
        let tool = SimplifyTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &tolerance_context(0.01));
        let Some(Geometry::MultiPolygon(polygons)) = result.features[0].geometry.as_ref() else {
            panic!("the geometry must stay a MultiPolygon");
        };
        assert_eq!(
            polygons.coordinates[0][0].len(),
            5,
            "the first member must be thinned: {polygons:?}"
        );
        assert_eq!(
            polygons.coordinates[1][0].len(),
            5,
            "the tiny second member must be kept at full detail, not dropped: {polygons:?}"
        );
    }

    #[test]
    fn simplify_walks_into_a_geometry_collection() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"GeometryCollection",
                 "geometries":[
                    {"type":"Point","coordinates":[0,0]},
                    {"type":"LineString","coordinates":[[0,0],[1,0.001],[2,0]]}
                 ]}}
            ]}"#,
        );
        let tool = SimplifyTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &tolerance_context(0.01));
        let Some(Geometry::GeometryCollection(collection)) = result.features[0].geometry.as_ref()
        else {
            panic!("a GeometryCollection must stay one");
        };
        let Some(Geometry::LineString(line)) = collection.geometries.get(1) else {
            panic!("the nested LineString must survive");
        };
        assert_eq!(line.coordinates.len(), 2, "the nested line must be thinned");
    }

    #[test]
    fn a_negative_tolerance_is_refused() {
        let tool = SimplifyTool {
            features: Arc::new(parse(ZIGZAG)),
        };
        let error = tool
            .run(&tolerance_context(-1.0))
            .expect_err("a negative tolerance is meaningless");
        assert_eq!(
            error,
            CoreError::InvalidParameter {
                name: "tolerance_deg".to_string(),
                reason: "must not be negative".to_string(),
            }
        );
    }

    #[test]
    fn a_nan_tolerance_is_refused_rather_than_silently_disabling_simplification() {
        // `serde_json` cannot hold a non-finite number, so a NaN handed to
        // `json!` arrives as `null` — refused by the "expected a number" arm
        // rather than by the finiteness check, but refused either way.
        let tool = SimplifyTool {
            features: Arc::new(parse(ZIGZAG)),
        };
        let error = tool
            .run(&tolerance_context(f64::NAN))
            .expect_err("NaN is not a tolerance");
        assert_eq!(
            error,
            CoreError::InvalidParameter {
                name: "tolerance_deg".to_string(),
                reason: "expected a number".to_string(),
            }
        );
    }

    #[test]
    fn validate_tolerance_refuses_every_non_finite_value() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                validate_tolerance(value),
                Err(CoreError::InvalidParameter {
                    name: "tolerance_deg".to_string(),
                    reason: "expected a finite number".to_string(),
                }),
                "{value} must be refused"
            );
        }
        assert_eq!(validate_tolerance(0.0), Ok(0.0));
        assert_eq!(validate_tolerance(1.5), Ok(1.5));
    }

    #[test]
    fn a_missing_tolerance_is_refused_instead_of_defaulted() {
        let tool = SimplifyTool {
            features: Arc::new(parse(ZIGZAG)),
        };
        let error = tool
            .run(&ToolContext::new())
            .expect_err("the parameter is required");
        assert_eq!(
            error,
            CoreError::InvalidParameter {
                name: "tolerance_deg".to_string(),
                reason: "missing required parameter".to_string(),
            }
        );
    }

    // ---- ring self-intersection guard -------------------------------------

    #[test]
    fn ring_is_simple_accepts_a_plain_square() {
        let square = vec![
            vec![0.0, 0.0],
            vec![2.0, 0.0],
            vec![2.0, 2.0],
            vec![0.0, 2.0],
            vec![0.0, 0.0],
        ];
        assert!(ring_is_simple(&square));
    }

    #[test]
    fn ring_is_simple_rejects_a_bowtie() {
        // The two diagonals of a unit square, connected in bowtie order
        // instead of square order: edge 0 (0,0)-(1,1) and edge 2 (1,0)-(0,1)
        // cross at (0.5, 0.5), a point interior to both.
        let bowtie = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        ];
        assert!(!ring_is_simple(&bowtie));
    }

    #[test]
    fn ring_is_simple_accepts_a_ring_that_only_touches_itself_at_a_repeated_vertex() {
        // Two triangles that meet at a single shared point (1,1), reused as
        // both the ring's 2nd and 4th position, without either triangle's
        // edges crossing the other's. Douglas-Peucker can leave a ring in
        // this shape (two originally-distinct vertices simplifying to the
        // same position); it must not be confused with the bowtie case
        // above, which crosses rather than merely touches.
        let touching = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![0.0, 0.0],
        ];
        assert!(ring_is_simple(&touching));
    }

    #[test]
    fn ring_is_simple_never_panics_on_a_malformed_position() {
        // A one-component position parses as a `Position` fine (no
        // `Deserialize`-time validation) but cannot anchor an orientation
        // test; `orientation`'s length guard must keep this from indexing
        // out of bounds.
        let ring = vec![
            vec![0.0, 0.0],
            vec![1.0],
            vec![2.0, 0.0],
            vec![1.0, 2.0],
            vec![0.0, 0.0],
        ];
        assert!(ring_is_simple(&ring));
    }

    #[test]
    fn ring_is_simple_skips_the_sweep_above_the_vertex_cap() {
        // A ring this large is assumed simple without being swept — the
        // `O(n^2)` sweep is capped to bound worst-case cost (see
        // `MAX_SELF_INTERSECTION_CHECK_VERTICES`'s docs), not run and
        // truncated. Every position is identical, which would make a real
        // sweep degenerate rather than prove anything either way — the
        // point here is only that the cap short-circuits before one starts.
        let oversized = vec![vec![0.0, 0.0]; MAX_SELF_INTERSECTION_CHECK_VERTICES + 2];
        assert!(ring_is_simple(&oversized));
    }

    #[test]
    fn simplified_rings_valid_rejects_a_self_intersecting_polygon() {
        let bowtie = Polygon::from_exterior(vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        ])
        .expect("RFC 7946 does not require a ring to be simple, only closed");
        assert!(!simplified_rings_valid(&Geometry::Polygon(bowtie)));
    }
}
