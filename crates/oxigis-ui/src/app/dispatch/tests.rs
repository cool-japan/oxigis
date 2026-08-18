// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for the gestures [`super`] owns end to end: the layer panel's
//! zoom-to-layer, the Processing window's Output group, and the whole life of
//! a Processing run — started, driven frame by frame, cancelled, or landed
//! after the window that started it was closed.
//!
//! A child of `app::dispatch` rather than a file under `app`'s own `tests*.rs`
//! set because these exercise routing that is private to that module
//! (`OxigisApp::route_processing_result_to`, `apply_processing_action`), which
//! a sibling file would have to widen to reach. Split out of `dispatch.rs`
//! itself under the 2 000-line rule.

use super::OxigisApp;
use crate::app::data_io::IoDialog;
use crate::layer_panel::LayerAction;
use crate::processing_panel::{OutputDestination, OutputTarget, ProcessingAction};
use oxigis_core::LayerId;
use std::collections::BTreeMap;

/// Two points, far enough apart that fitting them is a visible camera move.
const POINTS: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"Tokyo"},
     "geometry":{"type":"Point","coordinates":[139.767,35.681]}},
    {"type":"Feature","properties":{"name":"Osaka"},
     "geometry":{"type":"Point","coordinates":[135.502,34.702]}}]}"#;

/// Three points, so a two-of-three selection is tellable from the whole
/// layer.
const THREE_POINTS: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"a"},
     "geometry":{"type":"Point","coordinates":[139.767,35.681]}},
    {"type":"Feature","properties":{"name":"b"},
     "geometry":{"type":"Point","coordinates":[135.502,34.702]}},
    {"type":"Feature","properties":{"name":"c"},
     "geometry":{"type":"Point","coordinates":[130.401,33.590]}}]}"#;

fn descriptor() -> oxigis_core::ToolDescriptor {
    oxigis_core::ToolDescriptor {
        id: "synthetic-fc".to_string(),
        title: "Synthetic".to_string(),
        description: String::new(),
        params: Vec::new(),
    }
}

/// A registered built-in descriptor, by id.
fn builtin(id: &str) -> oxigis_core::ToolDescriptor {
    oxigis_core::builtin_registry()
        .get(id)
        .expect("a built-in descriptor")
        .clone()
}

/// The `layer` parameter map every single-layer tool needs.
fn layer_params(id: LayerId) -> BTreeMap<String, serde_json::Value> {
    let mut params = BTreeMap::new();
    params.insert("layer".to_string(), serde_json::json!(id.get()));
    params
}

/// One egui frame of exactly what `OxigisApp::ui` does for Processing.
fn processing_frame(ctx: &egui::Context, app: &mut OxigisApp) {
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        ..Default::default()
    };
    // `begin_pass`/`end_pass` rather than `Context::run_ui`: the window draws
    // itself against the *context*, not into a parent `Ui`, and this pair is
    // egui's own documented alternative for exactly that shape.
    ctx.begin_pass(raw_input);
    app.processing_window(ctx);
    let _output = ctx.end_pass();
}

/// Drives frames until the job in flight lands, as the frame loop would.
///
/// The wait is what makes this deterministic on native: the run is on a
/// worker thread, so a frame loop that never yields could poll a hundred
/// thousand times before that thread is scheduled once.
fn run_to_completion(ctx: &egui::Context, app: &mut OxigisApp) {
    for iteration in 0..100_000 {
        processing_frame(ctx, app);
        if !app.processing.is_running() {
            return;
        }
        if iteration < 100 {
            std::thread::yield_now();
        } else {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    }
    panic!("the processing job never finished");
}

/// Starts `descriptor` over `layer` through the panel's own action, then
/// drives it to completion.
fn run_tool(
    app: &mut OxigisApp,
    descriptor: &oxigis_core::ToolDescriptor,
    params: BTreeMap<String, serde_json::Value>,
    output: OutputTarget,
    selected_only: bool,
) {
    let ctx = egui::Context::default();
    app.apply_processing_action(ProcessingAction::Run {
        descriptor: descriptor.clone(),
        params,
        output,
        selected_only,
    });
    run_to_completion(&ctx, app);
}

fn one_feature() -> serde_json::Value {
    serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {"type":"Feature","properties":{},
             "geometry":{"type":"Point","coordinates":[139.767,35.681]}}
        ]
    })
}

#[test]
fn zoom_to_layer_moves_the_camera_and_records_nothing() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    // The add already zoomed and selected, so move away first — otherwise
    // "the camera is on the layer" would prove nothing.
    app.map_panel.set_view(
        app.map_panel
            .view()
            .with_center(oxigis_render::LonLat::new(0.0, 0.0))
            .with_zoom(2.0),
    );
    app.selection = None;
    // The add itself dirtied the project; the claim under test is that the
    // ZOOM does not.
    app.mark_saved();
    let entries = app.undo.depth();

    app.apply_layer_action(LayerAction::ZoomToLayer(id));

    let center = app.map_view().center();
    assert!(
        (center.lon - 137.6).abs() < 1.0 && (center.lat - 35.2).abs() < 1.0,
        "the camera must sit on the layer, not where it started: {center:?}"
    );
    assert_eq!(
        app.undo.depth(),
        entries,
        "a camera move is not project state and must record nothing"
    );
    // This ACTION alone does not retarget the editor. The user-facing
    // gesture does select, because a double-click reports `clicked` too and
    // the panel emits `Select` from that — see `layer_panel::draw_layer_row`
    // — so the two arrive as a pair and the selection is the click's doing,
    // never the zoom's.
    assert_eq!(app.selection, None);
    assert!(!app.has_unsaved_changes(), "the project did not change");
}

#[test]
fn zoom_to_a_layer_with_no_features_says_so_instead_of_moving() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddDemoXyzLayer);
    let id = app
        .project()
        .layers
        .layers()
        .first()
        .expect("the demo layer")
        .id;
    let before = app.map_view().center();
    app.apply_layer_action(LayerAction::ZoomToLayer(id));
    let after = app.map_view().center();
    assert!((after.lon - before.lon).abs() < f64::EPSILON);
    assert!((after.lat - before.lat).abs() < f64::EPSILON);
    let status = app.status.clone().unwrap_or_default();
    assert!(
        status.contains("no extent"),
        "the refusal must be reported, not silent: {status:?}"
    );
}

#[test]
fn a_named_output_wins_over_the_derived_name_and_is_still_de_duplicated() {
    let mut app = OxigisApp::new();
    let source = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    let output = OutputTarget {
        name: "hulls".to_string(),
        destination: OutputDestination::Layer,
    };
    for _ in 0..2 {
        app.route_processing_result_to(&descriptor(), Some(source), &output, Ok(one_feature()));
    }
    let names: Vec<&str> = app
        .project()
        .layers
        .layers()
        .iter()
        .map(|layer| layer.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["cities", "hulls", "hulls (2)"],
        "a chosen name replaces the derived one but still cannot collide"
    );
}

#[test]
fn a_blank_output_name_keeps_the_historical_derived_name() {
    let mut app = OxigisApp::new();
    let source = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    app.route_processing_result_to(
        &descriptor(),
        Some(source),
        &OutputTarget::default(),
        Ok(one_feature()),
    );
    let names: Vec<&str> = app
        .project()
        .layers
        .layers()
        .iter()
        .map(|layer| layer.name.as_str())
        .collect();
    assert_eq!(names, vec!["cities", "cities \u{2014} Synthetic"]);
}

#[test]
fn the_geojson_destination_hands_the_document_back_and_adds_nothing() {
    let mut app = OxigisApp::new();
    let source = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    app.mark_saved();
    let entries = app.undo.depth();

    app.route_processing_result_to(
        &descriptor(),
        Some(source),
        &OutputTarget {
            name: "centroids".to_string(),
            destination: OutputDestination::GeoJsonText,
        },
        Ok(one_feature()),
    );

    // The whole point: the result never enters the project document, so a
    // large one cannot be carried by every save from here on.
    assert_eq!(app.project().layers.len(), 1, "no layer may be added");
    assert_eq!(app.undo.depth(), entries, "nothing to undo");
    assert!(
        !app.has_unsaved_changes(),
        "a gesture that changed nothing must not claim the project is dirty"
    );
    // `IoDialog` is not `Debug` (it holds whole documents), so the
    // observation is the match itself.
    match &app.io_dialog {
        Some(IoDialog::ExportGeoJson { name, content }) => {
            assert_eq!(name, "centroids");
            // Re-parsable, so what the user copies out is a usable dataset.
            let parsed = crate::local_input::parse_geojson(content)
                .expect("the exported document must be readable GeoJSON");
            assert_eq!(parsed.features.len(), 1);
        }
        _ => panic!("expected the export dialog to be open"),
    }
    // And the run is still confirmed where the user is looking.
    let report = format!("{:?}", app.processing);
    assert!(
        report.contains("last_result: Some"),
        "a Run that opened a modal and left the window blank reads as a \
         button that did nothing: {report}"
    );
}

// ---- The run in flight ------------------------------------------------

#[test]
fn a_run_does_not_finish_inside_the_frame_that_asked_for_it() {
    // The whole point of the job: `apply_processing_action` starts the
    // work and returns, so the frame that clicked Run still paints.
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    let centroid = builtin("centroid");

    app.apply_processing_action(ProcessingAction::Run {
        descriptor: centroid.clone(),
        params: layer_params(id),
        output: OutputTarget::default(),
        selected_only: false,
    });
    assert!(app.processing.is_running(), "the run must be in flight");
    assert_eq!(
        app.project().layers.len(),
        1,
        "nothing may be added before the run has produced anything"
    );
    let status = app.status.clone().unwrap_or_default();
    assert!(
        status.contains("Running") && status.contains("Cancel"),
        "a run in flight must say so, and say how to stop it: {status:?}"
    );

    let ctx = egui::Context::default();
    run_to_completion(&ctx, &mut app);
    let names: Vec<&str> = app
        .project()
        .layers
        .layers()
        .iter()
        .map(|layer| layer.name.as_str())
        .collect();
    assert_eq!(names.len(), 2, "the result must land as a layer: {names:?}");
    assert!(names[1].starts_with("cities \u{2014} "), "{names:?}");
}

#[test]
fn a_shell_can_read_the_run_in_flight_without_opening_the_window() {
    let mut app = OxigisApp::new();
    assert_eq!(app.processing_progress(), None, "nothing is running");
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    let centroid = builtin("centroid");
    app.apply_processing_action(ProcessingAction::Run {
        descriptor: centroid.clone(),
        params: layer_params(id),
        output: OutputTarget::default(),
        selected_only: false,
    });

    let (title, progress) = app
        .processing_progress()
        .expect("a run in flight must be observable");
    assert_eq!(title, centroid.title);
    assert_eq!(progress.total, 2, "both features are the run's denominator");

    let ctx = egui::Context::default();
    run_to_completion(&ctx, &mut app);
    assert_eq!(
        app.processing_progress(),
        None,
        "a finished run is no longer in flight"
    );
}

#[test]
fn a_job_still_lands_after_the_window_is_closed() {
    // A run outlives the toolbox: `processing_window` polls before its own
    // visibility check precisely so closing the window mid-run cannot
    // strand the work (or, on the browser build, stop driving it).
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    app.apply_processing_action(ProcessingAction::Run {
        descriptor: builtin("centroid"),
        params: layer_params(id),
        output: OutputTarget::default(),
        selected_only: false,
    });
    app.show_processing = false;

    let ctx = egui::Context::default();
    run_to_completion(&ctx, &mut app);
    assert_eq!(
        app.project().layers.len(),
        2,
        "the result must land even though nothing was drawing"
    );
}

#[test]
fn cancelling_drops_the_run_and_adds_nothing() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    app.apply_processing_action(ProcessingAction::Run {
        descriptor: builtin("centroid"),
        params: layer_params(id),
        output: OutputTarget::default(),
        selected_only: false,
    });
    app.apply_processing_action(ProcessingAction::Cancel);

    assert!(!app.processing.is_running());
    let status = app.status.clone().unwrap_or_default();
    assert!(status.contains("cancelled"), "{status:?}");

    // Even if the worker had already computed a result, the handle is
    // gone, so no later frame may route one.
    let ctx = egui::Context::default();
    for _ in 0..8 {
        processing_frame(&ctx, &mut app);
    }
    assert_eq!(app.project().layers.len(), 1, "nothing may be added");
}

#[test]
fn a_refusal_known_before_any_work_is_reported_without_starting_a_job() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    app.apply_layer_action(LayerAction::Remove(id));

    app.apply_processing_action(ProcessingAction::Run {
        descriptor: builtin("centroid"),
        params: layer_params(id),
        output: OutputTarget::default(),
        selected_only: false,
    });
    assert!(!app.processing.is_running(), "there is nothing to run");
    let report = format!("{:?}", app.processing);
    assert!(
        report.contains("last_error: Some"),
        "the refusal must show in the window: {report}"
    );
}

#[test]
fn removing_the_source_layer_mid_run_still_routes_the_result() {
    // The run holds an `Arc` of the features, so the work is unaffected;
    // what must stay total is the *routing*, which names the output after
    // a layer that no longer exists.
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    app.apply_processing_action(ProcessingAction::Run {
        descriptor: builtin("centroid"),
        params: layer_params(id),
        output: OutputTarget::default(),
        selected_only: false,
    });
    app.apply_layer_action(LayerAction::Remove(id));

    let ctx = egui::Context::default();
    run_to_completion(&ctx, &mut app);
    let names: Vec<&str> = app
        .project()
        .layers
        .layers()
        .iter()
        .map(|layer| layer.name.as_str())
        .collect();
    assert_eq!(names.len(), 1, "only the result is left: {names:?}");
    assert!(
        names[0].ends_with("result"),
        "with no source layer left, the derived name falls back: {names:?}"
    );
}

// ---- Selected features only ------------------------------------------

/// Selects `features` of `layer` exactly as a map multi-select would.
fn select(app: &mut OxigisApp, layer: LayerId, features: &[usize]) {
    app.edit.retarget(Some(layer));
    let Some((&anchor, rest)) = features.split_first() else {
        return;
    };
    app.edit
        .set_selection(Some(crate::edit::EditSelection::feature(anchor)));
    for &feature in rest {
        app.edit.toggle_feature(feature);
    }
}

#[test]
fn selected_features_only_runs_the_tool_over_just_those_features() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", THREE_POINTS, None)
        .expect("valid GeoJSON");
    select(&mut app, id, &[0, 2]);
    assert!(
        app.selection_summary().covers(Some(id)),
        "the checkbox must be offerable for the layer the selection is on"
    );

    run_tool(
        &mut app,
        &builtin("feature_count"),
        layer_params(id),
        OutputTarget::default(),
        true,
    );
    let report = format!("{:?}", app.processing);
    assert!(
        report.contains("last_result: Some(\"2\")"),
        "two of three features were selected: {report}"
    );
}

#[test]
fn a_selection_on_another_layer_never_filters_this_one() {
    // The silent-wrong-answer case: source indices from layer A mean
    // different features in layer B, so the run must use the whole layer
    // rather than an arbitrary subset of the wrong ones.
    let mut app = OxigisApp::new();
    let cities = app
        .add_geojson_layer_from_text("cities", THREE_POINTS, None)
        .expect("valid GeoJSON");
    let pair = app
        .add_geojson_layer_from_text("pair", POINTS, None)
        .expect("valid GeoJSON");
    select(&mut app, cities, &[0, 1]);
    assert!(
        !app.selection_summary().covers(Some(pair)),
        "a selection on another layer must not be offerable here"
    );

    run_tool(
        &mut app,
        &builtin("feature_count"),
        layer_params(pair),
        OutputTarget::default(),
        // Even asked for directly — bypassing the panel's own guard — the
        // applier must refuse to cross layers.
        true,
    );
    let report = format!("{:?}", app.processing);
    assert!(
        report.contains("last_result: Some(\"2\")"),
        "the whole two-feature layer must be counted: {report}"
    );
}

#[test]
fn a_one_feature_selection_produces_exactly_one_result_feature() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", THREE_POINTS, None)
        .expect("valid GeoJSON");
    select(&mut app, id, &[0]);
    run_tool(
        &mut app,
        &builtin("centroid"),
        layer_params(id),
        OutputTarget::default(),
        true,
    );
    assert_eq!(
        app.project().layers.len(),
        2,
        "one selected feature must still produce one centroid"
    );
    let features = app
        .local
        .feature_set(
            app.project()
                .layers
                .layers()
                .get(1)
                .expect("the result layer")
                .id,
        )
        .map_or(0, |set| set.features.len());
    assert_eq!(features, 1);
}

// ---- Output ▸ Save to file --------------------------------------------

#[test]
fn the_file_destination_records_a_request_and_adds_nothing() {
    let mut app = OxigisApp::new();
    let source = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    app.mark_saved();
    let entries = app.undo.depth();

    app.route_processing_result_to(
        &descriptor(),
        Some(source),
        &OutputTarget {
            name: "centroids".to_string(),
            destination: OutputDestination::File,
        },
        Ok(one_feature()),
    );

    assert_eq!(app.project().layers.len(), 1, "no layer may be added");
    assert_eq!(app.undo.depth(), entries, "nothing to undo");
    assert!(
        !app.has_unsaved_changes(),
        "a gesture that changed nothing must not claim the project is dirty"
    );
    assert!(
        app.io_dialog.is_none(),
        "the file destination is a shell request, not a copy-out modal"
    );

    let request = app
        .take_pending_processing_save()
        .expect("the write must be queued for the shell");
    assert_eq!(request.name, "centroids");
    assert_eq!(request.features, 1);
    let parsed = crate::local_input::parse_geojson(&request.content)
        .expect("the queued document must be readable GeoJSON");
    assert_eq!(parsed.features.len(), 1);
    assert!(
        app.take_pending_processing_save().is_none(),
        "a taken request must not be handed out twice"
    );

    // And the window says what actually happened, including what a build
    // with no file writer should do instead.
    let report = format!("{:?}", app.processing);
    assert!(
        report.contains("GeoJSON to copy out"),
        "the message must name the way out when nothing drains the request: {report}"
    );
}

// ---- The UI-side Buffer tool ------------------------------------------

#[test]
fn the_buffer_descriptor_comes_from_the_core_registry_and_agrees_with_this_crates() {
    // The descriptor MOVED into `oxigis_core::builtin_registry`, so a fresh
    // app carries it with no UI-side registration step at all — the whole
    // point of the move, since a tool the panel could only see after the
    // window had been opened once was a tool that did not exist until then.
    let app = OxigisApp::new();
    let registered = app
        .processing_registry
        .get(crate::processing_exec::BUFFER_TOOL_ID)
        .expect("a fresh app already carries the buffer descriptor");
    // And the copy this crate still exports for its executor's own tests is
    // pinned to it: two descriptions of one tool that could drift is exactly
    // what the move was meant to end, and only an equality assertion makes
    // that impossible rather than merely unlikely.
    assert_eq!(
        registered,
        &crate::processing_exec::buffer_descriptor(),
        "the core registry's descriptor and this crate's must stay identical"
    );
}

#[test]
fn the_buffer_tool_is_registered_by_the_window_and_buffers_a_point_layer() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");

    app.show_processing = true;
    let ctx = egui::Context::default();
    processing_frame(&ctx, &mut app);
    let buffer = app
        .processing_registry
        .get(crate::processing_exec::BUFFER_TOOL_ID)
        .expect("the window must register it")
        .clone();

    let mut params = layer_params(id);
    params.insert("distance_deg".to_string(), serde_json::json!(0.01));
    run_tool(&mut app, &buffer, params, OutputTarget::default(), false);

    let result = app
        .project()
        .layers
        .layers()
        .get(1)
        .expect("the buffer result layer")
        .id;
    let features = app.local.feature_set(result).expect("loaded").clone();
    assert_eq!(features.features.len(), 2, "one buffer per point");
    assert!(
        matches!(
            features.features[0].geometry,
            Some(oxigeo::geojson::types::Geometry::Polygon(_))
        ),
        "a point's buffer is a polygon: {:?}",
        features.features[0].geometry
    );
}

#[test]
fn an_empty_result_is_reported_as_a_success_under_either_destination() {
    // The empty-collection branch runs BEFORE the destination is consulted,
    // so neither destination may dress a successful empty run up as a
    // parse failure.
    let empty = serde_json::json!({"type": "FeatureCollection", "features": []});
    for destination in [
        OutputDestination::Layer,
        OutputDestination::GeoJsonText,
        OutputDestination::File,
    ] {
        let mut app = OxigisApp::new();
        app.route_processing_result_to(
            &descriptor(),
            None,
            &OutputTarget {
                name: String::new(),
                destination,
            },
            Ok(empty.clone()),
        );
        assert!(app.project().layers.is_empty());
        assert!(
            app.io_dialog.is_none(),
            "nothing to copy out of an empty run"
        );
        let status = app.status.clone().unwrap_or_default();
        assert!(
            status.contains("no feature produced a result"),
            "{destination:?}: {status:?}"
        );
    }
}
