// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Execution wiring for the built-in Processing tools (`bounds`,
//! `feature_count`, `centroid`, `simplify`, `convex_hull`, `buffer`).
//!
//! [`oxigis_core::processing`] only models tool *descriptors*; this module
//! supplies the Rust code that actually runs them, against the app-side
//! [`Arc<FeatureCollection>`] store ([`crate::local_input::LocalInputState::feature_set`]).
//! It lives in `oxigis-ui`, not `oxigis-core`, because `oxigis-core`'s own
//! docs commit to holding no OxiGeo types (see that crate's module docs), and
//! a [`ToolExecutor`] here needs `Arc<FeatureCollection>` to avoid cloning a
//! layer's whole feature set just to inspect it.
//!
//! Two result shapes come out of here, and the difference is load-bearing:
//! `bounds`/`feature_count` answer with scalar JSON, which the app prints into
//! the Processing window; the geometry tools answer with a serialized
//! [`FeatureCollection`], which `OxigisApp::route_processing_result` detects by
//! its `"type": "FeatureCollection"` member and turns into a new map layer. A
//! tool that means to produce a dataset therefore has to emit a whole
//! collection document — there is no other channel back to the map.
//!
//! # Two ways to run the same tool
//!
//! Every tool here is a fold over the layer's features, and that fold is
//! written **once**, as a `FeatureSink` (`job.rs`). Two drivers consume it:
//!
//! * [`builtin_executor`] — the synchronous [`ToolExecutor`] every existing
//!   caller (and every tool test) uses: build the sink, drain it in one call;
//! * [`start_builtin_run`] — the interruptible one the Processing window
//!   drives: a worker thread on native, one bounded slice per frame on
//!   `wasm32`, with live progress and a cancel that actually stops the work.
//!
//! Because both consume the same sink, a sliced run and a drained one cannot
//! produce different answers.
//!
//! # File layout
//!
//! The geometry lives in one submodule per tool (`centroid`, `simplify`,
//! `hull`, `buffer`) under the 2 000-line rule; this file keeps the two
//! scalar tools (small enough that a file of their own would be noise), the
//! id → executor table, and the shared refusal helper.
//!
//! [`Arc<FeatureCollection>`]: std::sync::Arc

use std::sync::Arc;

use oxigeo::geojson::types::{Feature, FeatureCollection};
use oxigis_core::{CoreError, ToolContext, ToolExecutor};

use job::FeatureSink;

mod buffer;
mod centroid;
#[cfg(test)]
mod fixtures;
mod hull;
mod job;
mod runner;
mod simplify;

pub use buffer::{BUFFER_TOOL_ID, buffer_descriptor};
pub use job::ToolProgress;
pub use runner::{ToolRun, ToolRunState};

/// The parameter name every single-layer tool reports a refusal against —
/// the [`oxigis_core::ParamKind::LayerRef`] field's
/// [`oxigis_core::ParamSpec::name`] in `builtin_registry`.
const LAYER_PARAM: &str = "layer";

/// Builds a [`CoreError::InvalidParameter`] against the `layer` parameter —
/// the only parameter a single-layer tool can meaningfully blame when the
/// data it was handed cannot produce a result.
fn invalid_layer(reason: &str) -> CoreError {
    CoreError::InvalidParameter {
        name: LAYER_PARAM.to_string(),
        reason: reason.to_string(),
    }
}

/// The `"bounds"` built-in tool: computes a vector layer's bounding box.
struct BoundsTool {
    /// The target layer's parsed features, shared (never cloned) with the
    /// app-side store.
    features: Arc<FeatureCollection>,
}

impl ToolExecutor for BoundsTool {
    fn run(&self, _context: &ToolContext) -> Result<serde_json::Value, CoreError> {
        job::drain(Arc::clone(&self.features), Box::new(BoundsSink::default()))
    }
}

/// [`BoundsTool`]'s accumulator: the running envelope of every feature's own
/// bounding box.
///
/// Mirrors `FeatureCollection::compute_bbox`'s aggregation loop (oxigeo-geojson
/// 0.2.2, `feature.rs:334-358`) without requiring `&mut self` or cloning the
/// collection: that method only needs `&mut self` to write its result back
/// into `FeatureCollection::bbox`, a field this module never reads.
#[derive(Debug)]
struct BoundsSink {
    /// `[min_lon, min_lat, max_lon, max_lat]`, still infinite in every
    /// component until some feature contributes a finite box.
    envelope: [f64; 4],
}

impl Default for BoundsSink {
    fn default() -> Self {
        Self {
            envelope: [
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ],
        }
    }
}

impl FeatureSink for BoundsSink {
    fn absorb(&mut self, feature: &Feature) {
        if let Some(geometry) = feature.geometry.as_ref()
            && let Some(bbox) = geometry.compute_bbox()
            && bbox.len() >= 4
        {
            self.envelope[0] = self.envelope[0].min(bbox[0]);
            self.envelope[1] = self.envelope[1].min(bbox[1]);
            self.envelope[2] = self.envelope[2].max(bbox[2]);
            self.envelope[3] = self.envelope[3].max(bbox[3]);
        }
    }

    fn finish(self: Box<Self>) -> Result<serde_json::Value, CoreError> {
        if !self.envelope.iter().all(|value| value.is_finite()) {
            // Every feature's geometry was `None`, or every geometry present
            // was degenerate/empty: there is no box to report.
            return Err(invalid_layer("the layer has no geometry to bound"));
        }
        Ok(serde_json::json!({
            "min_lon": self.envelope[0],
            "min_lat": self.envelope[1],
            "max_lon": self.envelope[2],
            "max_lat": self.envelope[3],
        }))
    }
}

/// The `"feature_count"` built-in tool: counts a vector layer's features.
struct FeatureCountTool {
    /// The target layer's parsed features.
    features: Arc<FeatureCollection>,
}

impl ToolExecutor for FeatureCountTool {
    fn run(&self, _context: &ToolContext) -> Result<serde_json::Value, CoreError> {
        job::drain(Arc::clone(&self.features), Box::new(CountSink::default()))
    }
}

/// [`FeatureCountTool`]'s accumulator.
///
/// Counts through the same per-feature pass every other tool uses rather than
/// reading `features.len()` in one step: the count then reports progress and
/// honours a cancel exactly like its peers, and one code path drives every
/// tool. Walking the collection to add one per entry is arithmetic per
/// feature, not work — even a million-feature layer is a few milliseconds.
#[derive(Debug, Default)]
struct CountSink {
    /// Features seen so far.
    count: usize,
}

impl FeatureSink for CountSink {
    fn absorb(&mut self, _feature: &Feature) {
        self.count += 1;
    }

    fn finish(self: Box<Self>) -> Result<serde_json::Value, CoreError> {
        // Unlike `bounds`, an empty layer is a legitimate, meaningful answer
        // here (`0`), not an error — there is nothing to refuse.
        Ok(serde_json::json!(self.count))
    }
}

/// Resolves a [`oxigis_core::ToolDescriptor::id`] to the [`ToolExecutor`]
/// that runs it, against `features` (the target layer's parsed data).
///
/// Registry-keyed dispatch — the "near-zero marginal cost per tool" seam the
/// blueprint promises for later phases: adding tool #3 means one more match
/// arm here, not new UI plumbing. Returns [`None`] for a descriptor id this
/// build has no executor for; [`crate::app::OxigisApp::run_processing_tool`]
/// turns that into a "not implemented yet" message rather than panicking.
///
/// Scoped to tools with exactly one [`oxigis_core::ParamKind::LayerRef`]
/// parameter, matching every descriptor [`oxigis_core::builtin_registry`] has
/// today; a multi-layer tool (e.g. a future intersect/clip) needs this
/// signature to grow first. Non-layer parameters need nothing here — they
/// arrive per run through [`ToolContext::params`], which is how `simplify`
/// receives its `tolerance_deg`.
#[must_use]
pub fn builtin_executor(
    id: &str,
    features: Arc<FeatureCollection>,
) -> Option<Box<dyn ToolExecutor>> {
    match id {
        "bounds" => Some(Box::new(BoundsTool { features })),
        "feature_count" => Some(Box::new(FeatureCountTool { features })),
        "centroid" => Some(Box::new(centroid::CentroidTool { features })),
        "simplify" => Some(Box::new(simplify::SimplifyTool { features })),
        "convex_hull" => Some(Box::new(hull::ConvexHullTool { features })),
        BUFFER_TOOL_ID => Some(Box::new(buffer::BufferTool { features })),
        _ => None,
    }
}

/// Builds the per-feature accumulator for tool `id`, with `context`'s
/// parameters already validated.
///
/// The one place a tool id becomes a [`FeatureSink`], so the synchronous
/// [`builtin_executor`] path and the interruptible [`start_builtin_run`] one
/// cannot disagree about what a tool does. [`None`] means no executor is
/// wired for that id — the same answer [`builtin_executor`] gives.
///
/// # Errors
///
/// Returns the tool's own [`CoreError::InvalidParameter`] when a parameter it
/// needs is missing or malformed (`simplify`'s `tolerance_deg`, `buffer`'s
/// `distance_deg`), i.e. before any feature has been touched.
fn builtin_sink(
    id: &str,
    context: &ToolContext,
) -> Option<Result<Box<dyn FeatureSink>, CoreError>> {
    match id {
        "bounds" => Some(Ok(Box::new(BoundsSink::default()))),
        "feature_count" => Some(Ok(Box::new(CountSink::default()))),
        "centroid" => Some(Ok(centroid::sink())),
        "simplify" => Some(simplify::sink(context)),
        "convex_hull" => Some(hull::sink(context)),
        BUFFER_TOOL_ID => Some(buffer::sink(context)),
        _ => None,
    }
}

/// Starts an interruptible run of tool `id` over `features`.
///
/// The counterpart of [`builtin_executor`] for a UI that must stay responsive:
/// see [`ToolRun`] for how the run is driven (a worker thread on native, one
/// bounded slice per frame on `wasm32`) and [`ToolProgress`] for what it
/// reports while it works.
///
/// # Errors
///
/// Returns a human-readable reason when no executor is wired for `id`, or
/// when the tool refuses one of `context`'s parameters — both before any
/// feature is touched, so a refused run costs nothing.
pub fn start_builtin_run(
    id: &str,
    features: Arc<FeatureCollection>,
    context: &ToolContext,
) -> Result<ToolRun, String> {
    let sink = builtin_sink(id, context)
        .ok_or_else(|| format!("{id} is not implemented yet"))?
        .map_err(|error| error.to_string())?;
    Ok(ToolRun::start(features, sink))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing_exec::fixtures::{parse, point_feature};
    use oxigeo::geojson::types::{Geometry, GeometryCollection, Point};

    /// The bounding box of a whole collection, through the same sink the
    /// `bounds` tool uses — the shape the old free-standing `compute_bounds`
    /// helper had, kept as a test-side convenience.
    fn compute_bounds(features: &FeatureCollection) -> Option<[f64; 4]> {
        let mut sink = BoundsSink::default();
        for feature in &features.features {
            sink.absorb(feature);
        }
        Box::new(sink).finish().ok().map(|value| {
            [
                value["min_lon"].as_f64().unwrap_or(f64::NAN),
                value["min_lat"].as_f64().unwrap_or(f64::NAN),
                value["max_lon"].as_f64().unwrap_or(f64::NAN),
                value["max_lat"].as_f64().unwrap_or(f64::NAN),
            ]
        })
    }

    #[test]
    fn compute_bounds_of_an_empty_collection_is_none() {
        let features = FeatureCollection::new(vec![]);
        assert_eq!(compute_bounds(&features), None);
    }

    #[test]
    fn compute_bounds_of_a_single_point_is_degenerate() {
        let features = FeatureCollection::new(vec![point_feature(10.0, 20.0)]);
        assert_eq!(compute_bounds(&features), Some([10.0, 20.0, 10.0, 20.0]));
    }

    #[test]
    fn compute_bounds_aggregates_across_features() {
        let features = FeatureCollection::new(vec![
            point_feature(0.0, 0.0),
            point_feature(10.0, -5.0),
            point_feature(-3.0, 8.0),
        ]);
        assert_eq!(compute_bounds(&features), Some([-3.0, -5.0, 10.0, 8.0]));
    }

    #[test]
    fn compute_bounds_skips_a_null_geometry_feature_instead_of_crashing() {
        let features = FeatureCollection::new(vec![
            Feature::new(None, None),
            point_feature(5.0, 6.0),
            Feature::new(None, None),
        ]);
        assert_eq!(compute_bounds(&features), Some([5.0, 6.0, 5.0, 6.0]));
    }

    #[test]
    fn compute_bounds_recurses_into_a_geometry_collection() {
        let collection = GeometryCollection::new(vec![
            Geometry::Point(Point::new_2d(1.0, 2.0).expect("valid point")),
            Geometry::Point(Point::new_2d(3.0, 4.0).expect("valid point")),
        ])
        .expect("valid geometry collection");
        let features = FeatureCollection::new(vec![Feature::new(
            Some(Geometry::GeometryCollection(collection)),
            None,
        )]);
        assert_eq!(compute_bounds(&features), Some([1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn feature_count_tool_reports_zero_one_and_many() {
        let empty = FeatureCountTool {
            features: Arc::new(FeatureCollection::new(vec![])),
        };
        assert_eq!(
            empty.run(&ToolContext::new()).expect("ok"),
            serde_json::json!(0)
        );

        let one = FeatureCountTool {
            features: Arc::new(FeatureCollection::new(vec![point_feature(0.0, 0.0)])),
        };
        assert_eq!(
            one.run(&ToolContext::new()).expect("ok"),
            serde_json::json!(1)
        );

        let many = FeatureCountTool {
            features: Arc::new(FeatureCollection::new(vec![
                point_feature(0.0, 0.0),
                point_feature(1.0, 1.0),
                point_feature(2.0, 2.0),
            ])),
        };
        assert_eq!(
            many.run(&ToolContext::new()).expect("ok"),
            serde_json::json!(3)
        );
    }

    #[test]
    fn bounds_tool_reports_the_exact_key_shape() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[139.767,35.681]}},
                {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[135.502,34.702]}}
            ]}"#,
        );
        let tool = BoundsTool {
            features: Arc::new(features),
        };
        let value = tool.run(&ToolContext::new()).expect("ok");
        assert_eq!(
            value,
            serde_json::json!({
                "min_lon": 135.502,
                "min_lat": 34.702,
                "max_lon": 139.767,
                "max_lat": 35.681,
            })
        );
    }

    #[test]
    fn bounds_tool_refuses_a_collection_with_no_usable_geometry() {
        let features =
            FeatureCollection::new(vec![Feature::new(None, None), Feature::new(None, None)]);
        let tool = BoundsTool {
            features: Arc::new(features),
        };
        let error = tool.run(&ToolContext::new()).expect_err("must refuse");
        assert_eq!(
            error,
            CoreError::InvalidParameter {
                name: "layer".to_string(),
                reason: "the layer has no geometry to bound".to_string(),
            }
        );
    }

    // ---- dispatch ---------------------------------------------------------

    #[test]
    fn builtin_executor_resolves_known_ids_and_refuses_unknown_ones() {
        let features = Arc::new(FeatureCollection::new(vec![]));
        for id in [
            "bounds",
            "feature_count",
            "centroid",
            "simplify",
            "convex_hull",
            BUFFER_TOOL_ID,
        ] {
            assert!(
                builtin_executor(id, Arc::clone(&features)).is_some(),
                "{id} must resolve to an executor"
            );
        }
        assert!(builtin_executor("does_not_exist", features).is_none());
    }

    #[test]
    fn every_builtin_descriptor_has_a_wired_executor() {
        for descriptor in oxigis_core::builtin_registry().iter() {
            let features = Arc::new(FeatureCollection::new(vec![]));
            assert!(
                builtin_executor(&descriptor.id, features).is_some(),
                "{} is registered but has no executor",
                descriptor.id,
            );
        }
    }

    #[test]
    fn every_executor_id_also_resolves_to_a_sink() {
        // The two dispatch tables are what keep a sliced run and a drained
        // one running the same code; an id in one and not the other is a
        // tool that behaves differently depending on which driver ran it.
        let features = Arc::new(FeatureCollection::new(vec![]));
        for id in [
            "bounds",
            "feature_count",
            "centroid",
            "simplify",
            "convex_hull",
            BUFFER_TOOL_ID,
        ] {
            assert!(
                builtin_executor(id, Arc::clone(&features)).is_some()
                    && builtin_sink(id, &ToolContext::new()).is_some(),
                "{id} must resolve in both dispatch tables"
            );
        }
        assert!(builtin_sink("does_not_exist", &ToolContext::new()).is_none());
    }

    #[test]
    fn starting_a_run_refuses_an_unknown_id_and_a_malformed_parameter() {
        let features = Arc::new(FeatureCollection::new(vec![]));
        let error = start_builtin_run("does_not_exist", Arc::clone(&features), &ToolContext::new())
            .err()
            .unwrap_or_default();
        assert!(error.contains("not implemented yet"), "{error}");

        // `simplify` validates its tolerance before the first feature, so a
        // bad parameter costs no work at all.
        let error = start_builtin_run("simplify", features, &ToolContext::new())
            .err()
            .unwrap_or_default();
        assert!(error.contains("tolerance_deg"), "{error}");
    }
}
