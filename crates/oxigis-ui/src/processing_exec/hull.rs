// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The `"convex_hull"` built-in tool: Andrew's monotone chain over a layer's
//! vertices, in either of two modes (one hull for the whole layer, or one per
//! feature).
//!
//! Split out of [`super`] under the 2 000-line rule.

use std::sync::Arc;

use oxigeo::geojson::types::{Feature, FeatureCollection, Geometry, Polygon, Position, Properties};
use oxigis_core::{CoreError, ToolContext, ToolExecutor};

use super::invalid_layer;
use super::job::FeatureSink;

/// The name of [`ConvexHullTool`]'s per-feature/whole-layer switch — see
/// [`per_feature_from`]. Not yet a parameter the registered `convex_hull`
/// descriptor carries (that needs a core-side change; see the module docs),
/// so every run reachable through today's panel omits it and gets the
/// `false` (whole-layer) default.
const PER_FEATURE_PARAM: &str = "per_feature";

/// The `"convex_hull"` built-in tool: the smallest convex polygon containing
/// a set of vertices.
///
/// Two modes, chosen by [`PER_FEATURE_PARAM`]: whole-layer
/// ([`WholeLayerSink`] — one hull polygon over every feature's vertices,
/// the only shape reachable through today's descriptor) and per-feature
/// ([`PerFeatureSink`] — one hull per input feature, matching QGIS
/// `native:convexhull`, whose per-feature behavior is that peer tool's
/// namesake default).
pub(super) struct ConvexHullTool {
    /// The target layer's parsed features.
    pub(super) features: Arc<FeatureCollection>,
}

impl ToolExecutor for ConvexHullTool {
    fn run(&self, context: &ToolContext) -> Result<serde_json::Value, CoreError> {
        super::job::drain(Arc::clone(&self.features), sink(context)?)
    }
}

/// [`ConvexHullTool`]'s per-feature accumulator, in whichever of the two
/// modes `context` asks for.
///
/// # Errors
///
/// Returns [`CoreError::InvalidParameter`] when `per_feature` is present but
/// is not a JSON bool — see [`per_feature_from`].
pub(super) fn sink(context: &ToolContext) -> Result<Box<dyn FeatureSink>, CoreError> {
    if per_feature_from(context)? {
        Ok(Box::new(PerFeatureSink::default()))
    } else {
        Ok(Box::new(WholeLayerSink::default()))
    }
}

/// Reads [`ConvexHullTool`]'s per-feature/whole-layer switch out of a run's
/// parameters.
///
/// Absent or JSON `null` defaults to `false` (whole-layer): every run
/// through today's registered `convex_hull` descriptor omits this key (it
/// does not carry the parameter yet), so absence is the common case, not a
/// caller error, and `false` keeps this build's shipped default behavior
/// bit-for-bit unchanged. Present-but-wrong-shaped is refused rather than
/// silently falling back, mirroring `simplify`'s tolerance: anything other
/// than a JSON bool sending this key is a caller bug, not a value to guess
/// past.
fn per_feature_from(context: &ToolContext) -> Result<bool, CoreError> {
    match context.params.get(PER_FEATURE_PARAM) {
        None | Some(serde_json::Value::Null) => Ok(false),
        Some(value) => value.as_bool().ok_or_else(|| CoreError::InvalidParameter {
            name: PER_FEATURE_PARAM.to_string(),
            reason: "expected a bool".to_string(),
        }),
    }
}

/// [`ConvexHullTool`]'s whole-layer mode: one hull polygon over every
/// feature's vertices, carrying a `hull_of` count of the features that
/// contributed at least one vertex.
#[derive(Debug, Default)]
struct WholeLayerSink {
    /// Every usable 2-D vertex seen so far, across all features.
    vertices: Vec<[f64; 2]>,
    /// How many features contributed at least one of them.
    contributing: usize,
}

impl FeatureSink for WholeLayerSink {
    fn absorb(&mut self, feature: &Feature) {
        let before = self.vertices.len();
        if let Some(geometry) = feature.geometry.as_ref() {
            push_vertices(geometry, &mut self.vertices);
        }
        if self.vertices.len() > before {
            self.contributing += 1;
        }
    }

    fn finish(self: Box<Self>) -> Result<serde_json::Value, CoreError> {
        let polygon = hull_polygon(self.vertices).map_err(|reason| invalid_layer(&reason))?;
        let mut properties = Properties::new();
        properties.insert("hull_of".to_string(), serde_json::json!(self.contributing));
        let feature = Feature::new(Some(Geometry::Polygon(polygon)), Some(properties));
        Ok(serde_json::to_value(FeatureCollection::new(vec![feature]))?)
    }
}

/// [`ConvexHullTool`]'s per-feature mode: one hull polygon per input
/// feature, keeping that feature's own `id`/properties — matching QGIS
/// `native:convexhull`.
///
/// A feature with no geometry, too few distinct vertices, or only collinear
/// ones is skipped (counted, not failed) rather than refusing the whole run:
/// one degenerate feature in an 11,000-feature layer is the common case, not
/// the exception — the same reasoning `centroid` already applies to a
/// geometry with no computable centroid. An input layer with no features, or
/// none that hull successfully, is a legitimate empty result, not an error,
/// for the same precedent.
#[derive(Debug, Default)]
struct PerFeatureSink {
    /// One hull feature per input feature that produced one.
    hulls: Vec<Feature>,
    /// Input features that produced none.
    skipped: usize,
}

impl FeatureSink for PerFeatureSink {
    fn absorb(&mut self, feature: &Feature) {
        let mut vertices = Vec::new();
        if let Some(geometry) = feature.geometry.as_ref() {
            push_vertices(geometry, &mut vertices);
        }
        match hull_polygon(vertices) {
            Ok(polygon) => {
                let mut hull =
                    Feature::new(Some(Geometry::Polygon(polygon)), feature.properties.clone());
                hull.id = feature.id.clone();
                self.hulls.push(hull);
            }
            Err(_reason) => self.skipped += 1,
        }
    }

    fn finish(self: Box<Self>) -> Result<serde_json::Value, CoreError> {
        if self.skipped > 0 {
            tracing::warn!(
                skipped = self.skipped,
                "oxigis-ui: convex_hull (per-feature) skipped features with no geometry or \
                 too few/all-collinear vertices",
            );
        }
        Ok(serde_json::to_value(FeatureCollection::new(self.hulls))?)
    }
}

/// Builds a closed hull [`Polygon`] from a set of 2-D vertices — shared by
/// [`WholeLayerSink`] and [`PerFeatureSink`], which differ only in *which*
/// vertices they hull and what they attach to the result.
fn hull_polygon(vertices: Vec<[f64; 2]>) -> Result<Polygon, String> {
    let hull = convex_hull(vertices)?;
    let mut ring: Vec<Position> = hull.iter().map(|[x, y]| vec![*x, *y]).collect();
    // RFC 7946 §3.1.6: a linear ring's first and last positions must be
    // identical. `convex_hull` returns each hull vertex once, so the ring
    // is closed here — `local_vector::quantize_ring` drops the repeat again
    // on the way to MVT, and `Polygon::validate` would reject the open form.
    if let Some(first) = ring.first().cloned() {
        ring.push(first);
    }
    Polygon::from_exterior(ring)
        .map_err(|error| format!("could not build the hull polygon: {error}"))
}

/// Appends every usable 2-D vertex of `geometry` to `out`, recursing through
/// nested `GeometryCollection`s.
///
/// Written out variant by variant rather than reusing
/// `oxigeo::geojson::utils::extract_coordinates`, which clones every
/// [`Position`] (one `Vec<f64>` allocation per vertex) into an intermediate
/// `Vec` that this caller reads exactly once — and whose exhaustive `match`
/// here means a future `Geometry` variant breaks the build instead of quietly
/// contributing no vertices.
fn push_vertices(geometry: &Geometry, out: &mut Vec<[f64; 2]>) {
    match geometry {
        Geometry::Point(point) => push_position(&point.coordinates, out),
        Geometry::LineString(line) => push_positions(&line.coordinates, out),
        Geometry::Polygon(polygon) => {
            for ring in &polygon.coordinates {
                push_positions(ring, out);
            }
        }
        Geometry::MultiPoint(points) => push_positions(&points.coordinates, out),
        Geometry::MultiLineString(lines) => {
            for line in &lines.coordinates {
                push_positions(line, out);
            }
        }
        Geometry::MultiPolygon(polygons) => {
            for rings in &polygons.coordinates {
                for ring in rings {
                    push_positions(ring, out);
                }
            }
        }
        Geometry::GeometryCollection(collection) => {
            for nested in &collection.geometries {
                push_vertices(nested, out);
            }
        }
    }
}

/// [`push_position`] over a whole coordinate sequence.
fn push_positions(positions: &[Position], out: &mut Vec<[f64; 2]>) {
    for position in positions {
        push_position(position, out);
    }
}

/// Appends one position's first two elements (longitude, latitude) to `out`,
/// skipping a position that has fewer than two elements or is not finite —
/// a malformed coordinate has no business anchoring a hull, and a non-finite
/// one would poison every cross product it takes part in.
fn push_position(position: &Position, out: &mut Vec<[f64; 2]>) {
    if let (Some(&x), Some(&y)) = (position.first(), position.get(1))
        && x.is_finite()
        && y.is_finite()
    {
        out.push([x, y]);
    }
}

/// Computes the 2-D convex hull of `points` with Andrew's monotone chain.
///
/// Returns the hull's vertices in **counter-clockwise** order (RFC 7946 §3.1.6
/// asks an exterior ring to wind that way), each listed once — the caller
/// closes the ring. Vertices lying *on* a hull edge are dropped, so the result
/// is the strictly convex hull: three collinear points never survive as three.
///
/// `Err` carries a user-facing reason for the two inputs that bound no area:
/// fewer than three distinct vertices, and vertices that are all collinear.
/// Duplicates are collapsed first, so a thousand copies of two points is the
/// former, not a degenerate hull.
fn convex_hull(mut points: Vec<[f64; 2]>) -> Result<Vec<[f64; 2]>, &'static str> {
    // Lexicographic by x, then y — the order monotone chain requires.
    // `total_cmp` gives a total order with no `unwrap` on `partial_cmp`;
    // `push_position` has already excluded NaN, so the two agree here anyway.
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    points.dedup();
    if points.len() < 3 {
        return Err("a convex hull needs at least 3 distinct vertices");
    }

    // Lower chain left to right, then upper chain right to left; the upper
    // chain may never pop into the finished lower one, hence the `floor`.
    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(points.len() + 1);
    for &point in &points {
        push_hull_vertex(&mut hull, 1, point);
    }
    let lower_len = hull.len();
    for &point in points.iter().rev().skip(1) {
        push_hull_vertex(&mut hull, lower_len, point);
    }
    // The upper chain closes back onto the lower chain's first vertex.
    hull.pop();

    if hull.len() < 3 {
        return Err("the layer's vertices are all collinear, so they bound no area");
    }
    Ok(hull)
}

/// Pushes `point` onto a monotone chain, first popping every vertex that
/// `point` makes non-convex (a clockwise turn) or redundant (a straight line).
///
/// `floor` is the number of leading vertices that belong to an already-finished
/// chain and must never be popped: `1` while the lower chain is being built
/// (its first vertex is the hull's leftmost point), and the lower chain's whole
/// length while the upper one is.
fn push_hull_vertex(hull: &mut Vec<[f64; 2]>, floor: usize, point: [f64; 2]) {
    while hull.len() > floor {
        // Slice pattern rather than `hull[hull.len() - 2]`: the guard above
        // already implies two elements (`floor >= 1` at both call sites), and
        // this way the compiler enforces it instead of the reader.
        let [.., previous, last] = hull.as_slice() else {
            break;
        };
        if cross(*previous, *last, point) > 0.0 {
            break;
        }
        hull.pop();
    }
    hull.push(point);
}

/// Twice the signed area of the triangle `origin → a → b`: positive for a
/// counter-clockwise (left) turn, negative for a clockwise one, and exactly
/// zero when the three points are collinear.
fn cross(origin: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing_exec::fixtures::{parse, run_to_collection, signed_ring_area};
    use oxigeo::geojson::types::FeatureId;

    /// Extracts the single hull polygon's exterior ring from a `convex_hull`
    /// result, failing the test on any other shape.
    fn hull_ring(result: &FeatureCollection) -> Vec<Position> {
        assert_eq!(result.features.len(), 1, "a hull is exactly one feature");
        let Some(Geometry::Polygon(polygon)) = result.features[0].geometry.as_ref() else {
            panic!("a hull must be a Polygon");
        };
        assert_eq!(polygon.coordinates.len(), 1, "a hull has no holes");
        polygon.exterior().cloned().unwrap_or_default()
    }

    #[test]
    fn a_square_with_interior_points_hulls_to_four_corners_in_a_closed_ccw_ring() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"MultiPoint","coordinates":
                    [[0,0],[10,0],[10,10],[0,10],[5,5],[2,7],[8,3]]}}
            ]}"#,
        );
        let tool = ConvexHullTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        let ring = hull_ring(&result);

        assert_eq!(
            ring.len(),
            5,
            "four corners plus the repeated closing position, got {ring:?}"
        );
        assert_eq!(
            ring.first(),
            ring.last(),
            "RFC 7946 requires a closed linear ring"
        );
        assert!(
            signed_ring_area(&ring) > 0.0,
            "an exterior ring must wind counter-clockwise, got {ring:?}"
        );
        assert!(
            (signed_ring_area(&ring) - 100.0).abs() < 1e-9,
            "the hull of a 10x10 square must have area 100"
        );
        for corner in [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]] {
            assert!(
                ring.iter().any(|position| position[..2] == corner),
                "corner {corner:?} missing from {ring:?}"
            );
        }
    }

    #[test]
    fn a_vertex_lying_on_a_hull_edge_is_dropped() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"MultiPoint","coordinates":
                    [[0,0],[5,0],[10,0],[10,10],[0,10]]}}
            ]}"#,
        );
        let tool = ConvexHullTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        let ring = hull_ring(&result);
        assert_eq!(
            ring.len(),
            5,
            "the mid-edge vertex must not survive, got {ring:?}"
        );
    }

    #[test]
    fn duplicate_vertices_are_collapsed_before_hulling() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"MultiPoint","coordinates":
                    [[0,0],[0,0],[0,0],[10,0],[10,0],[10,10],[10,10],[0,10],[0,10]]}}
            ]}"#,
        );
        let tool = ConvexHullTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        assert_eq!(hull_ring(&result).len(), 5);
    }

    #[test]
    fn collinear_only_input_is_refused_with_a_reason() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"LineString","coordinates":
                    [[0,0],[1,1],[2,2],[3,3]]}}
            ]}"#,
        );
        let tool = ConvexHullTool {
            features: Arc::new(features),
        };
        let error = tool
            .run(&ToolContext::new())
            .expect_err("a line bounds no area");
        assert_eq!(
            error,
            CoreError::InvalidParameter {
                name: "layer".to_string(),
                reason: "the layer's vertices are all collinear, so they bound no area".to_string(),
            }
        );
    }

    #[test]
    fn fewer_than_three_distinct_vertices_are_refused() {
        for coordinates in ["[]", "[[0,0],[0,0]]", "[[0,0],[1,1],[0,0],[1,1]]"] {
            let features = parse(&format!(
                r#"{{"type":"FeatureCollection","features":[
                    {{"type":"Feature","properties":{{}},
                     "geometry":{{"type":"MultiPoint","coordinates":{coordinates}}}}}
                ]}}"#
            ));
            let tool = ConvexHullTool {
                features: Arc::new(features),
            };
            let error = tool
                .run(&ToolContext::new())
                .expect_err("fewer than three distinct vertices bound no area");
            assert_eq!(
                error,
                CoreError::InvalidParameter {
                    name: "layer".to_string(),
                    reason: "a convex hull needs at least 3 distinct vertices".to_string(),
                },
                "for {coordinates}"
            );
        }
    }

    #[test]
    fn an_empty_layer_is_refused_rather_than_hulled_into_nothing() {
        let tool = ConvexHullTool {
            features: Arc::new(FeatureCollection::new(vec![])),
        };
        assert!(tool.run(&ToolContext::new()).is_err());
    }

    #[test]
    fn hull_of_counts_only_the_features_that_contributed_a_vertex() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[0,0]}},
                {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[4,0]}},
                {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[0,4]}},
                {"type":"Feature","properties":{},"geometry":null}
            ]}"#,
        );
        let tool = ConvexHullTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        assert_eq!(
            result.features[0].get_property("hull_of"),
            Some(&serde_json::json!(3))
        );
    }

    #[test]
    fn per_feature_convex_hull_produces_one_hull_per_feature_and_skips_degenerate_ones() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","id":"a","properties":{"name":"square"},
                 "geometry":{"type":"MultiPoint","coordinates":[[0,0],[2,0],[2,2],[0,2],[1,1]]}},
                {"type":"Feature","id":"b","properties":{"name":"triangle"},
                 "geometry":{"type":"MultiPoint","coordinates":[[10,10],[12,10],[10,12]]}},
                {"type":"Feature","properties":{"name":"too few"},
                 "geometry":{"type":"Point","coordinates":[20,20]}}
            ]}"#,
        );
        let tool = ConvexHullTool {
            features: Arc::new(features),
        };
        let mut context = ToolContext::new();
        context
            .params
            .insert(PER_FEATURE_PARAM.to_string(), serde_json::json!(true));
        let result = run_to_collection(&tool, &context);

        assert_eq!(
            result.features.len(),
            2,
            "the single-point feature has too few vertices to hull and must be skipped, \
             not fail the whole run"
        );
        assert_eq!(
            result.features[0].id,
            Some(FeatureId::String("a".to_string()))
        );
        assert_eq!(
            result.features[0].get_property("name"),
            Some(&serde_json::json!("square"))
        );
        assert_eq!(
            result.features[1].id,
            Some(FeatureId::String("b".to_string()))
        );
        for feature in &result.features {
            assert!(
                feature.get_property("hull_of").is_none(),
                "per-feature hulls keep the source properties, not a whole-layer hull_of count"
            );
        }
    }

    #[test]
    fn per_feature_from_defaults_to_whole_layer_and_refuses_a_non_bool() {
        assert_eq!(
            per_feature_from(&ToolContext::new()),
            Ok(false),
            "absent must default to whole-layer — today's only reachable shape"
        );

        let mut null_context = ToolContext::new();
        null_context
            .params
            .insert(PER_FEATURE_PARAM.to_string(), serde_json::Value::Null);
        assert_eq!(per_feature_from(&null_context), Ok(false));

        let mut true_context = ToolContext::new();
        true_context
            .params
            .insert(PER_FEATURE_PARAM.to_string(), serde_json::json!(true));
        assert_eq!(per_feature_from(&true_context), Ok(true));

        let mut bad_context = ToolContext::new();
        bad_context
            .params
            .insert(PER_FEATURE_PARAM.to_string(), serde_json::json!("yes"));
        assert_eq!(
            per_feature_from(&bad_context),
            Err(CoreError::InvalidParameter {
                name: "per_feature".to_string(),
                reason: "expected a bool".to_string(),
            })
        );
    }

    #[test]
    fn the_hull_walks_every_geometry_variant_including_nested_collections() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"GeometryCollection",
                 "geometries":[
                    {"type":"Point","coordinates":[0,0]},
                    {"type":"MultiLineString","coordinates":[[[10,0],[10,5]]]},
                    {"type":"GeometryCollection","geometries":[
                        {"type":"MultiPolygon","coordinates":[[[[0,10],[5,10],[0,12],[0,10]]]]}
                    ]}
                 ]}}
            ]}"#,
        );
        let tool = ConvexHullTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &ToolContext::new());
        let ring = hull_ring(&result);
        assert!(
            signed_ring_area(&ring) > 0.0,
            "the hull must wind counter-clockwise"
        );
        for corner in [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [5.0, 10.0],
            [0.0, 12.0],
        ] {
            assert!(
                ring.iter().any(|position| position[..2] == corner),
                "vertex {corner:?} from a nested geometry missing from {ring:?}"
            );
        }
    }

    #[test]
    fn a_non_finite_coordinate_never_reaches_the_hull() {
        let mut points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        push_position(&vec![f64::NAN, 0.5], &mut points);
        push_position(&vec![0.5, f64::INFINITY], &mut points);
        push_position(&vec![1.0], &mut points);
        assert_eq!(
            points.len(),
            3,
            "only the three finite 2-D positions remain"
        );
        assert_eq!(convex_hull(points).map(|hull| hull.len()), Ok(3));
    }
}
