// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Test-only fixtures shared by every [`super`] submodule's own `tests`
//! module.
//!
//! Lives in its own file rather than being duplicated per submodule: the
//! executors were split out of one 2 400-line file under the 2 000-line rule,
//! and their tests came with them — five copies of `parse`/`run_to_collection`
//! is exactly the drift this crate splits files to avoid.

use oxigeo::geojson::types::{Feature, FeatureCollection, Geometry, Point, Position};
use oxigis_core::{ToolContext, ToolExecutor};

/// Parses a GeoJSON `FeatureCollection` fixture, panicking (test-only)
/// on a malformed literal.
pub(super) fn parse(text: &str) -> FeatureCollection {
    oxigeo::geojson::reader::feature_collection_from_str(text).expect("valid fixture")
}

/// A property-less, id-less point feature at `(lon, lat)`.
pub(super) fn point_feature(lon: f64, lat: f64) -> Feature {
    let point = Point::new_2d(lon, lat).expect("valid point");
    Feature::new(Some(Geometry::Point(point)), None)
}

/// Runs a geometry-producing tool and re-parses its JSON result as a
/// [`FeatureCollection`], asserting the `"type"` member the app's
/// `route_processing_result` keys layer creation off — so every one of
/// these tests also proves its tool's result would actually become a
/// layer, not fall through to the scalar/pretty-print branch.
pub(super) fn run_to_collection(
    tool: &dyn ToolExecutor,
    context: &ToolContext,
) -> FeatureCollection {
    let value = tool.run(context).expect("the tool must succeed");
    assert_eq!(
        value.get("type").and_then(serde_json::Value::as_str),
        Some("FeatureCollection"),
        "a geometry-producing tool must answer with a routable collection",
    );
    serde_json::from_value(value).expect("the result must re-parse as a FeatureCollection")
}

/// A [`ToolContext`] carrying just `simplify`'s tolerance.
pub(super) fn tolerance_context(tolerance: f64) -> ToolContext {
    let mut context = ToolContext::new();
    context.params.insert(
        super::simplify::TOLERANCE_PARAM.to_string(),
        serde_json::json!(tolerance),
    );
    context
}

/// Twice-shoelace signed area of a closed ring: positive when the ring
/// winds counter-clockwise, which RFC 7946 §3.1.6 asks of an exterior ring.
pub(super) fn signed_ring_area(ring: &[Position]) -> f64 {
    let mut sum = 0.0;
    for pair in ring.windows(2) {
        let [a, b] = pair else { continue };
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}
