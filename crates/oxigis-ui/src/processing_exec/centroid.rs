// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The `"centroid"` built-in tool: one point per feature, at that feature's
//! centre of mass.
//!
//! Split out of [`super`] under the 2 000-line rule; the moment accumulators
//! ([`AreaMoment`], [`LengthMoment`]) are the whole reason this is a file of
//! its own — they are the difference between an area-weighted centroid and a
//! vertex mean, which is the difference between agreeing with PostGIS
//! `ST_Centroid` and not.

use std::sync::Arc;

use crate::edit::command::open_ring;
use oxigeo::geojson::types::{Feature, FeatureCollection, Geometry, Point, Position};
use oxigis_core::{CoreError, ToolContext, ToolExecutor};

use super::job::FeatureSink;

/// The `"centroid"` built-in tool: replaces every feature with a point at its
/// centroid, keeping the feature's properties and id.
pub(super) struct CentroidTool {
    /// The target layer's parsed features.
    pub(super) features: Arc<FeatureCollection>,
}

impl ToolExecutor for CentroidTool {
    fn run(&self, _context: &ToolContext) -> Result<serde_json::Value, CoreError> {
        super::job::drain(Arc::clone(&self.features), sink())
    }
}

/// [`CentroidTool`]'s per-feature accumulator — the body of its loop, shared
/// by the one-shot [`ToolExecutor`] path and the sliced
/// [`super::job::ToolPass`] one so the two can never disagree about what a
/// run produces.
#[must_use]
pub(super) fn sink() -> Box<dyn FeatureSink> {
    Box::new(CentroidSink::default())
}

/// The accumulator behind [`sink`]: one centroid point per feature that has
/// one, plus a count of the features that produced none.
#[derive(Debug, Default)]
struct CentroidSink {
    /// The centroids built so far, in input order.
    centroids: Vec<Feature>,
    /// Features with no computable centroid — reported once at the end.
    skipped: usize,
}

impl FeatureSink for CentroidSink {
    fn absorb(&mut self, feature: &Feature) {
        match centroid_feature(feature) {
            Some(centroid) => self.centroids.push(centroid),
            None => self.skipped += 1,
        }
    }

    fn finish(self: Box<Self>) -> Result<serde_json::Value, CoreError> {
        if self.skipped > 0 {
            // The result travels back to the app as a bare `FeatureCollection`
            // (that is what makes it a layer), which leaves no room for a
            // "skipped n features" notice — so `warn!`, not `debug!`, is the
            // only trace of this that survives a release build's default log
            // filter. Surfacing it in the app itself needs a change in
            // `OxigisApp::route_processing_result_from` (`app/dispatch.rs`),
            // outside this module's scope.
            tracing::warn!(
                skipped = self.skipped,
                "oxigis-ui: centroid skipped features with no computable centroid",
            );
        }
        Ok(serde_json::to_value(FeatureCollection::new(
            self.centroids,
        ))?)
    }
}

/// Builds one feature's centroid point, carrying its properties and id over
/// unchanged.
///
/// Returns [`None`] — the caller counts these as skipped — when the feature
/// has no geometry, when [`area_weighted_centroid`] declines (an empty
/// geometry, or one whose every position is malformed), or when the centroid
/// is not a finite 2-D position. That last guard matters: `serde_json`
/// encodes a non-finite `f64` as `null`, so an unguarded NaN would put a
/// malformed coordinate into the result document instead of failing visibly
/// — it is also what makes a poisoned (NaN) moment sum in
/// [`AreaMoment`]/[`LengthMoment`] fail safely rather than shipping a wrong
/// centroid.
///
/// The result is always 2-D — emitting elevation only for point inputs would
/// make the output schema depend on the input's geometry type.
fn centroid_feature(feature: &Feature) -> Option<Feature> {
    let position = area_weighted_centroid(feature.geometry.as_ref()?)?;
    let (&lon, &lat) = (position.first()?, position.get(1)?);
    if !lon.is_finite() || !lat.is_finite() {
        return None;
    }
    let point = Point::new_2d(lon, lat).ok()?;
    let mut centroid = Feature::new(Some(Geometry::Point(point)), feature.properties.clone());
    centroid.id = feature.id.clone();
    Some(centroid)
}

/// The centroid the `centroid` descriptor promises: PostGIS `ST_Centroid` /
/// QGIS `native:centroids` / ArcGIS all return the **area-weighted** centre
/// of mass for a polygon, not the mean of its vertices — a polygon whose
/// boundary is unevenly digitised (a detailed coastline against a handful of
/// straight inland segments) has a vertex mean pulled toward the denser
/// side, while an area centroid does not care how many vertices drew each
/// edge.
///
/// * `Polygon`/`MultiPolygon` → the true area-weighted centroid, holes
///   subtracted (see [`AreaMoment::subtract_hole`] for why a hole's *stored*
///   winding is never trusted), falling back to [`vertex_mean_centroid`]
///   only when the net area is exactly zero (a degenerate/collinear ring).
/// * `LineString`/`MultiLineString` → the segment-**length**-weighted mean
///   of segment midpoints (see [`LengthMoment`]), falling back the same way
///   when the total length is exactly zero.
/// * `Point`/`MultiPoint` → the plain vertex mean — there is no area or
///   length to weight by.
/// * `GeometryCollection` → the plain mean of its members' own (correctly
///   weighted) centroids. Deliberately not PostGIS's "highest dimension
///   wins" rule for a mixed collection: nothing in the audited finding
///   evidenced that case, so it stays a documented non-goal instead of an
///   undocumented gap.
fn area_weighted_centroid(geometry: &Geometry) -> Option<Position> {
    match geometry {
        Geometry::Point(_) | Geometry::MultiPoint(_) => vertex_mean_centroid(geometry),
        Geometry::LineString(line) => line_moment(&line.coordinates)
            .centroid()
            .or_else(|| vertex_mean_centroid(geometry)),
        Geometry::MultiLineString(lines) => {
            let mut total = LengthMoment::default();
            for line in &lines.coordinates {
                total.add(line_moment(line));
            }
            total.centroid().or_else(|| vertex_mean_centroid(geometry))
        }
        Geometry::Polygon(polygon) => polygon_moment(&polygon.coordinates)
            .centroid()
            .or_else(|| vertex_mean_centroid(geometry)),
        Geometry::MultiPolygon(polygons) => {
            let mut total = AreaMoment::default();
            for rings in &polygons.coordinates {
                total.add_member(polygon_moment(rings));
            }
            total.centroid().or_else(|| vertex_mean_centroid(geometry))
        }
        Geometry::GeometryCollection(collection) => {
            let centroids: Vec<Position> = collection
                .geometries
                .iter()
                .filter_map(area_weighted_centroid)
                .collect();
            mean_position(centroids.iter())
        }
    }
}

/// Running, un-normalised first-moment-of-area accumulator: `area2` is twice
/// the signed shoelace area (`Σ x_i·y_{i+1} − x_{i+1}·y_i`), `mx`/`my` the
/// matching `Σ(x_i+x_{i+1})·cross_i` / `Σ(y_i+y_{i+1})·cross_i` sums.
/// [`Self::centroid`] is the only place these three get divided into an
/// actual point, so a `Polygon`'s holes and a `MultiPolygon`'s members can
/// accumulate into one total first.
#[derive(Debug, Default, Clone, Copy)]
struct AreaMoment {
    /// Twice the signed shoelace area accumulated so far.
    area2: f64,
    /// The matching first moment about the y axis.
    mx: f64,
    /// The matching first moment about the x axis.
    my: f64,
}

impl AreaMoment {
    /// One edge's contribution: `cross = ax·by − bx·ay`, added to `area2`,
    /// `mx`, `my`. A position missing its `x` or `y` contributes `f64::NAN`
    /// rather than being skipped: silently dropping a vertex mid-ring would
    /// splice its two neighbours into an edge that was never in the input,
    /// quietly changing the polygon's shape, whereas poisoning the sum fails
    /// loudly through [`centroid_feature`]'s finiteness check instead.
    fn add_edge(&mut self, a: &Position, b: &Position) {
        let ax = a.first().copied().unwrap_or(f64::NAN);
        let ay = a.get(1).copied().unwrap_or(f64::NAN);
        let bx = b.first().copied().unwrap_or(f64::NAN);
        let by = b.get(1).copied().unwrap_or(f64::NAN);
        let cross = ax * by - bx * ay;
        self.area2 += cross;
        self.mx += (ax + bx) * cross;
        self.my += (ay + by) * cross;
    }

    /// This moment with every sum negated — the effect of reversing the
    /// ring's traversal direction: swapping `a`/`b` in every [`Self::add_edge`]
    /// call negates `cross` and, with it, everything derived from `cross`.
    fn negated(self) -> Self {
        Self {
            area2: -self.area2,
            mx: -self.mx,
            my: -self.my,
        }
    }

    /// Adds a hole ring's contribution, flipped so it always **opposes**
    /// `self`'s accumulated sign — i.e. always subtracts area — regardless
    /// of the hole's own stored winding direction. RFC 7946 asks a hole to
    /// wind opposite its exterior, but nothing upstream of this function
    /// enforces it (a hand-built or malformed-but-parseable `Polygon` can
    /// carry either), and trusting it would silently *add* a same-wound
    /// hole's area instead of subtracting it.
    fn subtract_hole(&mut self, hole: &[Position]) {
        let mut moment = ring_moment(hole);
        if moment.area2 != 0.0 && self.area2 != 0.0 && moment.area2.signum() == self.area2.signum()
        {
            moment = moment.negated();
        }
        self.area2 += moment.area2;
        self.mx += moment.mx;
        self.my += moment.my;
    }

    /// Adds one `MultiPolygon` member's own net (already hole-corrected)
    /// moment, after normalising its sign to positive. Two disjoint polygon
    /// members must always **add** area, never cancel one another, so —
    /// unlike [`Self::subtract_hole`], which deliberately wants opposition —
    /// this deliberately wants agreement, and does not trust that every
    /// member was wound the same way the first one was.
    fn add_member(&mut self, mut member: AreaMoment) {
        if member.area2 < 0.0 {
            member = member.negated();
        }
        self.area2 += member.area2;
        self.mx += member.mx;
        self.my += member.my;
    }

    /// The area-weighted centroid, or [`None`] when the net area is exactly
    /// zero — a degenerate polygon (every vertex collinear, or fewer than
    /// three effective vertices) has no well-defined centre of mass. The
    /// caller falls back to [`vertex_mean_centroid`] in that case.
    fn centroid(&self) -> Option<Position> {
        (self.area2 != 0.0)
            .then(|| vec![self.mx / (3.0 * self.area2), self.my / (3.0 * self.area2)])
    }
}

/// The [`AreaMoment`] of one **closed** ring (`ring.first() == ring.last()`,
/// as GeoJSON stores one): `windows(2)` then visits each of the ring's edges
/// exactly once, wraparound included, since the closing duplicate makes the
/// last window `(V_{n-1}, V_0)`. Nothing upstream of this function enforces
/// that a parsed `Polygon`'s ring actually arrived closed (`Polygon`'s own
/// `Deserialize` performs no such check), so a ring that is not already
/// closed gets its missing wraparound edge added explicitly instead of
/// silently shoelace-summing an incomplete boundary.
fn ring_moment(ring: &[Position]) -> AreaMoment {
    let mut moment = AreaMoment::default();
    for pair in ring.windows(2) {
        moment.add_edge(&pair[0], &pair[1]);
    }
    if let (Some(first), Some(last)) = (ring.first(), ring.last())
        && first != last
    {
        moment.add_edge(last, first);
    }
    moment
}

/// The net, hole-corrected [`AreaMoment`] of one polygon's ring list —
/// `rings[0]` is the exterior per RFC 7946 §3.1.6, every ring after it a
/// hole subtracted via [`AreaMoment::subtract_hole`].
fn polygon_moment(rings: &[Vec<Position>]) -> AreaMoment {
    let mut moment = match rings.first() {
        Some(exterior) => ring_moment(exterior),
        None => AreaMoment::default(),
    };
    for hole in rings.iter().skip(1) {
        moment.subtract_hole(hole);
    }
    moment
}

/// Running length-weighted first-moment accumulator for a line's centroid:
/// `length` is the summed segment length, `mx`/`my` the matching
/// `Σ segment_length·midpoint` sums — left un-normalised so a
/// `MultiLineString` can add every member's segments before dividing once.
#[derive(Debug, Default, Clone, Copy)]
struct LengthMoment {
    /// The summed length of every segment seen so far.
    length: f64,
    /// The matching length-weighted sum of segment midpoint `x`s.
    mx: f64,
    /// The matching length-weighted sum of segment midpoint `y`s.
    my: f64,
}

impl LengthMoment {
    /// Folds another line's segment contributions into this total.
    fn add(&mut self, other: LengthMoment) {
        self.length += other.length;
        self.mx += other.mx;
        self.my += other.my;
    }

    /// The length-weighted mean position, or [`None`] when the total length
    /// is exactly zero (a single position, or every segment coincident) —
    /// the caller falls back to [`vertex_mean_centroid`].
    fn centroid(&self) -> Option<Position> {
        (self.length != 0.0).then(|| vec![self.mx / self.length, self.my / self.length])
    }
}

/// The [`LengthMoment`] of one line's own segments (`coords.windows(2)`) — a
/// line is never implicitly closed the way a polygon ring is, so unlike
/// [`ring_moment`] there is no synthetic wraparound edge here. A position
/// missing its `x`/`y` poisons the sum with `f64::NAN` rather than being
/// skipped, for the same reason [`AreaMoment::add_edge`] does.
fn line_moment(coords: &[Position]) -> LengthMoment {
    let mut moment = LengthMoment::default();
    for pair in coords.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        let ax = a.first().copied().unwrap_or(f64::NAN);
        let ay = a.get(1).copied().unwrap_or(f64::NAN);
        let bx = b.first().copied().unwrap_or(f64::NAN);
        let by = b.get(1).copied().unwrap_or(f64::NAN);
        let len = ((bx - ax) * (bx - ax) + (by - ay) * (by - ay)).sqrt();
        moment.length += len;
        moment.mx += len * (ax + bx) / 2.0;
        moment.my += len * (ay + by) / 2.0;
    }
    moment
}

/// The vertex-mean fallback [`area_weighted_centroid`] uses whenever its
/// primary quantity (area or length) is exactly zero, and the whole answer
/// for `Point`/`MultiPoint`, which have neither.
///
/// Deliberately **not** `oxigeo::geojson::utils::geometry_centroid`: that
/// implementation averages a polygon's exterior ring in its **closed** form,
/// so the ring's first vertex is summed twice and the divisor is `n + 1` —
/// even a perfectly regular polygon's centroid lands off-centre, displaced
/// toward whichever corner the file happened to list first by
/// `(v0 − mean)/(n + 1)`, and rotating the ring (a legal, meaning-preserving
/// rewrite) moves the result. Its `MultiPolygon` arm additionally averages
/// hole rings in.
///
/// Here every ring is averaged **open** ([`open_ring`] drops the closing
/// duplicate), and a `MultiPolygon` averages exactly its members' exterior
/// rings. A `GeometryCollection` is the mean of its members' fallback
/// centroids.
fn vertex_mean_centroid(geometry: &Geometry) -> Option<Position> {
    match geometry {
        Geometry::Point(point) => Some(point.coordinates.clone()),
        Geometry::LineString(line) => mean_position(line.coordinates.iter()),
        Geometry::Polygon(polygon) => mean_position(open_ring(polygon.exterior()?).iter()),
        Geometry::MultiPoint(points) => mean_position(points.coordinates.iter()),
        Geometry::MultiLineString(lines) => mean_position(lines.coordinates.iter().flatten()),
        Geometry::MultiPolygon(polygons) => mean_position(
            polygons
                .coordinates
                .iter()
                .filter_map(|rings| rings.first())
                .flat_map(|ring| open_ring(ring).iter()),
        ),
        Geometry::GeometryCollection(collection) => {
            let centroids: Vec<Position> = collection
                .geometries
                .iter()
                .filter_map(vertex_mean_centroid)
                .collect();
            mean_position(centroids.iter())
        }
    }
}

/// The mean of every position that carries at least two elements, or [`None`]
/// when nothing qualifies. Non-finite inputs pass through as non-finite sums;
/// [`centroid_feature`]'s finiteness guard is the single place that refuses
/// them. Unlike [`AreaMoment`]/[`LengthMoment`], a malformed position here is
/// dropped rather than poisoning the sum: this is an unordered mean over a
/// point set (or a fallback for one), not a shoelace/segment traversal, so
/// dropping one entry cannot splice two unrelated neighbours into a phantom
/// edge the way it would mid-ring.
fn mean_position<'a>(positions: impl Iterator<Item = &'a Position>) -> Option<Position> {
    let (mut sum_x, mut sum_y, mut count) = (0.0_f64, 0.0_f64, 0_usize);
    for position in positions {
        if let (Some(&x), Some(&y)) = (position.first(), position.get(1)) {
            sum_x += x;
            sum_y += y;
            count += 1;
        }
    }
    (count > 0).then(|| vec![sum_x / count as f64, sum_y / count as f64])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing_exec::fixtures::{parse, point_feature, run_to_collection};
    use oxigeo::geojson::types::FeatureId;

    #[test]
    fn centroids_preserve_properties_and_skip_null_geometry() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","id":"tokyo","properties":{"name":"Tokyo","pop":13960236},
                 "geometry":{"type":"Point","coordinates":[139.767,35.681]}},
                {"type":"Feature","properties":{"name":"nowhere"},"geometry":null}
            ]}"#,
        );
        let tool = CentroidTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());

        assert_eq!(
            result.features.len(),
            1,
            "the null-geometry feature must be skipped, not emitted without a centroid"
        );
        let centroid = &result.features[0];
        assert_eq!(
            centroid.get_property("name"),
            Some(&serde_json::json!("Tokyo"))
        );
        assert_eq!(
            centroid.get_property("pop"),
            Some(&serde_json::json!(13_960_236))
        );
        assert_eq!(
            centroid.id,
            Some(FeatureId::String("tokyo".to_string())),
            "the source id must survive so the result joins back to the input"
        );
    }

    #[test]
    fn a_points_centroid_is_the_point_itself() {
        let tool = CentroidTool {
            features: Arc::new(FeatureCollection::new(vec![point_feature(139.767, 35.681)])),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        assert_eq!(result.features.len(), 1);
        assert_eq!(
            result.features[0].geometry,
            Some(Geometry::Point(
                Point::new_2d(139.767, 35.681).expect("valid point")
            ))
        );
    }

    #[test]
    fn a_polygons_centroid_is_the_open_ring_vertex_mean_a_squares_centre() {
        // A closed 2x2 square ring: five positions, the first repeated as the
        // last. The mean is taken over the **open** ring — the closing
        // duplicate is a format artifact, not a vertex — so a regular square's
        // centroid is its centre (1, 1), not the (0.8, 0.8) that
        // double-counting corner 0 would produce.
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":
                    [[[0,0],[2,0],[2,2],[0,2],[0,0]]]}}
            ]}"#,
        );
        let tool = CentroidTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());

        let Some(Geometry::Point(point)) = result.features[0].geometry.as_ref() else {
            panic!("a centroid must be a Point");
        };
        let (lon, lat) = (point.coordinates[0], point.coordinates[1]);
        assert!(
            (lon - 1.0).abs() < 1e-12 && (lat - 1.0).abs() < 1e-12,
            "a square's vertex-mean centroid is its centre, got ({lon}, {lat})"
        );
    }

    #[test]
    fn rotating_a_ring_does_not_move_its_centroid() {
        // Listing the same square starting from a different corner is a
        // legal, meaning-preserving rewrite; a centroid that depends on which
        // corner the file happens to open with is measuring the file, not the
        // shape.
        let original = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":
                    [[[0,0],[2,0],[2,2],[0,2],[0,0]]]}}
            ]}"#,
        );
        let rotated = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":
                    [[[2,2],[0,2],[0,0],[2,0],[2,2]]]}}
            ]}"#,
        );
        let centroid_of = |features: FeatureCollection| {
            let tool = CentroidTool {
                features: Arc::new(features),
            };
            let result = run_to_collection(&tool, &ToolContext::new());
            let Some(Geometry::Point(point)) = result.features[0].geometry.as_ref() else {
                panic!("a centroid must be a Point");
            };
            (point.coordinates[0], point.coordinates[1])
        };
        assert_eq!(centroid_of(original), centroid_of(rotated));
    }

    #[test]
    fn a_holes_true_area_is_subtracted_regardless_of_its_stored_winding() {
        // A 2x2 square (area 4, centroid (1,1)) with a small off-centre hole
        // at (0.1,0.1)-(0.3,0.3) (area 0.04, centroid (0.2,0.2)) — wound the
        // *same* direction (CCW) as the exterior, the opposite of RFC 7946's
        // convention, to prove the subtraction never trusts stored winding.
        // Hand-computed: net area 3.96, Cx = (4*1 − 0.04*0.2)/3.96 =
        // 3.992/3.96 = 499/495 (off (1,1): the pre-fix vertex-mean algorithm
        // ignored holes entirely and always landed exactly on (1,1) here).
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"MultiPolygon","coordinates":
                    [[[[0,0],[2,0],[2,2],[0,2],[0,0]],
                      [[0.1,0.1],[0.3,0.1],[0.3,0.3],[0.1,0.3],[0.1,0.1]]]]}}
            ]}"#,
        );
        let tool = CentroidTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        let Some(Geometry::Point(point)) = result.features[0].geometry.as_ref() else {
            panic!("a centroid must be a Point");
        };
        let (lon, lat) = (point.coordinates[0], point.coordinates[1]);
        let expected = 499.0 / 495.0;
        assert!(
            (lon - expected).abs() < 1e-9 && (lat - expected).abs() < 1e-9,
            "expected ({expected}, {expected}), got ({lon}, {lat})"
        );
    }

    #[test]
    fn a_polygons_centroid_is_area_weighted_not_vertex_mean_on_an_l_shape() {
        // The fix's whole point: an L-shape's area centroid, (1.1, 1.1) by
        // decomposing it into rectangles [0,3]x[0,1] (area 3, centre
        // (1.5,0.5)) and [0,1]x[1,3] (area 2, centre (0.5,2.0)), combining to
        // (3*1.5+2*0.5)/5 = 1.1 — visibly disagrees with the vertex mean of
        // its six open-ring corners, (8/6, 8/6) ≈ (1.333, 1.333). The tool
        // must report the former.
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":
                    [[[0,0],[3,0],[3,1],[1,1],[1,3],[0,3],[0,0]]]}}
            ]}"#,
        );
        let tool = CentroidTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        let Some(Geometry::Point(point)) = result.features[0].geometry.as_ref() else {
            panic!("a centroid must be a Point");
        };
        let (lon, lat) = (point.coordinates[0], point.coordinates[1]);
        assert!(
            (lon - 1.1).abs() < 1e-9 && (lat - 1.1).abs() < 1e-9,
            "an L-shape's area centroid is (1.1, 1.1), not its vertex mean ({lon}, {lat})"
        );
    }

    #[test]
    fn multi_polygon_members_add_area_even_when_wound_inconsistently_with_each_other() {
        // Two disjoint unit squares; the second is deliberately wound
        // clockwise (RFC 7946 asks every exterior to wind CCW) to prove one
        // member's "wrong" winding cannot cancel another's area instead of
        // adding to it. Two area-1 squares must combine to the mean of their
        // own centres, (0.5,0.5) and (10.5,10.5): (5.5, 5.5).
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"MultiPolygon","coordinates":
                    [[[[0,0],[1,0],[1,1],[0,1],[0,0]]],
                     [[[10,10],[10,11],[11,11],[11,10],[10,10]]]]}}
            ]}"#,
        );
        let tool = CentroidTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        let Some(Geometry::Point(point)) = result.features[0].geometry.as_ref() else {
            panic!("a centroid must be a Point");
        };
        let (lon, lat) = (point.coordinates[0], point.coordinates[1]);
        assert!(
            (lon - 5.5).abs() < 1e-9 && (lat - 5.5).abs() < 1e-9,
            "two disjoint areas must add, not partly cancel from inconsistent winding: \
             got ({lon}, {lat})"
        );
    }

    #[test]
    fn a_collinear_polygon_ring_falls_back_to_the_vertex_mean() {
        // Zero net area — no centre of mass to weight by — so the
        // documented fallback applies: the open-ring vertex mean of
        // (0,0),(1,0),(2,0), which is (1,0).
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":
                    [[[0,0],[1,0],[2,0],[0,0]]]}}
            ]}"#,
        );
        let tool = CentroidTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        let Some(Geometry::Point(point)) = result.features[0].geometry.as_ref() else {
            panic!("a centroid must be a Point");
        };
        assert_eq!(
            (point.coordinates[0], point.coordinates[1]),
            (1.0, 0.0),
            "a degenerate ring must fall back to the vertex mean, not vanish or NaN"
        );
    }

    #[test]
    fn a_linestrings_centroid_is_length_weighted_not_a_vertex_mean() {
        // Two segments of very different length: (0,0)-(1,0) [length 1] then
        // (1,0)-(1,100) [length 100]. The plain vertex mean of the 3 points
        // is (0.667, 33.33); length-weighted midpoints give
        // Cx=(1*0.5+100*1)/101=100.5/101, Cy=(1*0+100*50)/101=5000/101 — much
        // closer to the long segment's own midpoint, (1, 50).
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"LineString","coordinates":
                    [[0,0],[1,0],[1,100]]}}
            ]}"#,
        );
        let tool = CentroidTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        let Some(Geometry::Point(point)) = result.features[0].geometry.as_ref() else {
            panic!("a centroid must be a Point");
        };
        let (lon, lat) = (point.coordinates[0], point.coordinates[1]);
        assert!(
            (lon - 100.5 / 101.0).abs() < 1e-9 && (lat - 5000.0 / 101.0).abs() < 1e-9,
            "expected the length-weighted mean, got ({lon}, {lat})"
        );
    }

    #[test]
    fn a_multi_linestrings_centroid_is_length_weighted_across_members() {
        // The same two segments as the `LineString` case above — length 1
        // then length 100 — but split across two `MultiLineString` members
        // instead of one line's two segments. The expected centroid is
        // identical, proving `LengthMoment::add` sums a member's
        // contribution the same way `line_moment` sums a member's own
        // segments, not "mean of each member's own centroid" (which would
        // weight the short member and the long one equally).
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"MultiLineString","coordinates":
                    [[[0,0],[1,0]],[[1,0],[1,100]]]}}
            ]}"#,
        );
        let tool = CentroidTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        let Some(Geometry::Point(point)) = result.features[0].geometry.as_ref() else {
            panic!("a centroid must be a Point");
        };
        let (lon, lat) = (point.coordinates[0], point.coordinates[1]);
        assert!(
            (lon - 100.5 / 101.0).abs() < 1e-9 && (lat - 5000.0 / 101.0).abs() < 1e-9,
            "expected the length-weighted mean across members, got ({lon}, {lat})"
        );
    }

    #[test]
    fn centroid_skips_a_geometry_with_no_computable_centre() {
        // An empty MultiPoint parses fine but averages nothing.
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"MultiPoint","coordinates":[]}},
                {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[1,2]}}
            ]}"#,
        );
        let tool = CentroidTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        assert_eq!(result.features.len(), 1);
    }

    #[test]
    fn centroid_of_an_empty_layer_is_an_empty_collection_not_an_error() {
        let tool = CentroidTool {
            features: Arc::new(FeatureCollection::new(vec![])),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        assert!(result.features.is_empty());
    }
}
