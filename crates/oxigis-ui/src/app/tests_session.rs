// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Session-level tests of `OxigisApp`: the unsaved-changes guard in front of
//! File ▸ New / Open, zoom-to-layer, and how a Processing run's output is
//! named and reported.
//!
//! Split out of `app/tests.rs` under the 2000-line rule; the fixtures are
//! deliberately re-declared here rather than shared, so the two files stay
//! independently movable.

use std::sync::Arc;

use egui::Context;
use oxigis_core::Project;
use oxigis_render::LonLat;

use super::OxigisApp;
use super::data_io::{IoDialog, PendingAction};
use crate::layer_panel::LayerAction;

/// Two points, Tokyo and Osaka — a real, non-degenerate bbox.
const POINTS: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"Tokyo"},
     "geometry":{"type":"Point","coordinates":[139.767,35.681]}},
    {"type":"Feature","properties":{"name":"Osaka"},
     "geometry":{"type":"Point","coordinates":[135.502,34.702]}}]}"#;

/// A bare `ToolDescriptor` with no params, for exercising result routing
/// without a real executor.
fn synthetic_descriptor(id: &str) -> oxigis_core::ToolDescriptor {
    oxigis_core::ToolDescriptor {
        id: id.to_string(),
        title: "Synthetic".to_string(),
        description: String::new(),
        params: Vec::new(),
    }
}

// ---- Unsaved-changes guard --------------------------------------------

/// Runs one full frame, which is what advances the unsaved-changes
/// observation and draws whatever modal is open.
fn frame(app: &mut OxigisApp) {
    let ctx = Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
}

#[test]
fn a_fresh_project_has_nothing_to_lose_and_is_never_interrupted() {
    let mut app = OxigisApp::new();
    assert!(!app.has_unsaved_changes());
    assert_eq!(app.title_line(), "Untitled project");
    app.request_discarding_action(PendingAction::NewProject);
    assert!(
        app.io_dialog.is_none(),
        "a clean project must not be confirmed at"
    );
    app.request_discarding_action(PendingAction::OpenProject);
    assert!(
        matches!(app.io_dialog, Some(IoDialog::Open { .. })),
        "Open goes straight to its own modal"
    );
}

#[test]
fn adding_a_layer_makes_new_and_open_ask_first() {
    let mut app = OxigisApp::new();
    let _ = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    assert!(app.has_unsaved_changes());
    assert!(app.title_line().ends_with(" *"));

    app.request_discarding_action(PendingAction::NewProject);
    assert!(matches!(
        app.io_dialog,
        Some(IoDialog::ConfirmDiscard { .. })
    ));
    // Drawing the confirmation must not itself destroy anything.
    frame(&mut app);
    assert_eq!(app.project().layers.len(), 1);
    assert!(matches!(
        app.io_dialog,
        Some(IoDialog::ConfirmDiscard { .. })
    ));
}

#[test]
fn an_entry_recorded_outside_the_choke_point_still_arms_the_guard() {
    // The edit family records through `edit_glue`, which never touches the
    // project choke point — the undo-log watermark is what catches it.
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    app.mark_saved();
    assert!(!app.has_unsaved_changes());

    // Selecting is not an edit and must not arm anything.
    app.apply_layer_action(LayerAction::Select(id));
    app.observe_recorded_edits();
    assert!(!app.has_unsaved_changes());

    // Pushed straight onto the log, i.e. exactly what a recorder outside
    // this module does — none of the hand-placed dirty stamps run.
    let _ = app.undo.push(crate::edit::project_op::ProjectTransaction {
        label: "Change opacity",
        op: crate::edit::project_op::ProjectOp::SetOpacity {
            layer: id,
            before: 1.0,
            after: 0.5,
        },
        coalesce: None,
    });
    app.observe_recorded_edits();
    assert!(
        app.has_unsaved_changes(),
        "a recorded edit that bypasses the choke point still arms the guard"
    );
}

#[test]
fn an_edit_at_the_undo_cap_still_arms_the_guard() {
    // The hole the collection-identity signal exists to close, and the one
    // that matters most: a long digitising session fills the log, after
    // which every push ALSO evicts — leaving both depths unchanged, and
    // (because two vertex moves of one feature weigh exactly the same)
    // the byte total too. Without a third signal, one save would defeat the
    // guard for the rest of the session.
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    let equal_weight = |layer| crate::edit::project_op::ProjectTransaction {
        label: "Change opacity",
        op: crate::edit::project_op::ProjectOp::SetOpacity {
            layer,
            before: 1.0,
            after: 0.5,
        },
        coalesce: None,
    };
    // Past the cap, so the log is already evicting when the save happens.
    for _ in 0..=app.undo.max_entries() {
        let _ = app.undo.push(equal_weight(id));
    }
    app.mark_saved();
    assert!(!app.has_unsaved_changes());
    let saved_depth = app.undo.depth();
    let saved_bytes = app.undo.bytes();

    // One more edit: push + evict, of the same weight.
    let _ = app.undo.push(equal_weight(id));
    assert_eq!(app.undo.depth(), saved_depth, "the depths cannot see it");
    assert_eq!(app.undo.bytes(), saved_bytes, "nor can the byte total");
    // …and the collection it edited is replaced, as every commit does.
    let features = app
        .local
        .feature_set(id)
        .cloned()
        .expect("the layer's features are loaded");
    let replacement = Arc::new((*features).clone());
    app.local
        .replace_features(&app.project, id, replacement)
        .expect("a same-shape replacement is accepted");

    app.observe_recorded_edits();
    assert!(
        app.has_unsaved_changes(),
        "an edit at the undo cap must not be invisible to the guard"
    );
}

#[test]
fn hydrating_a_path_referenced_layer_does_not_count_as_a_modification() {
    // A hydrate FINISHES a load — the file already named the layer — so it
    // moves the collection fingerprint without modifying anything.
    let mut app = OxigisApp::new();
    let mut project = Project::new("Referenced");
    let layer = oxigis_core::Layer::new(
        "cities",
        oxigis_core::LayerKind::Vector(oxigis_core::VectorSource::LocalGeoJson {
            path: "/nowhere/cities.geojson".to_string(),
        }),
    );
    let id = layer.id;
    project.layers.add(layer);
    app.load_project(project);
    assert!(!app.has_unsaved_changes());
    assert!(app.local.feature_set(id).is_none(), "not read yet");

    assert!(app.hydrate_geojson_layer_from_bytes(id, "cities", POINTS.as_bytes()));
    assert!(app.local.feature_set(id).is_some());
    app.observe_recorded_edits();
    assert!(
        !app.has_unsaved_changes(),
        "reading the file the project points at is not an edit"
    );
}

#[test]
fn discarding_runs_the_pending_action_and_cancelling_keeps_the_project() {
    let mut app = OxigisApp::new();
    let _ = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");

    // Cancel: the dialog closes, the project stands.
    app.io_dialog = Some(IoDialog::ConfirmDiscard {
        pending: PendingAction::NewProject,
    });
    app.io_dialog = None;
    assert_eq!(app.project().layers.len(), 1);

    // Discard: the pending New really runs, and the fresh project is clean.
    app.request_discarding_action(PendingAction::NewProject);
    assert!(matches!(
        app.io_dialog,
        Some(IoDialog::ConfirmDiscard { .. })
    ));
    app.io_dialog = None;
    app.new_project();
    assert!(app.project().layers.is_empty());
    assert!(!app.has_unsaved_changes());
}

#[test]
fn saving_clears_the_marker_and_a_later_edit_arms_it_again() {
    let mut app = OxigisApp::new();
    let _ = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    assert!(app.has_unsaved_changes());

    app.open_save_dialog();
    assert!(matches!(app.io_dialog, Some(IoDialog::Save { .. })));
    assert!(!app.has_unsaved_changes(), "the document is on screen");
    app.io_dialog = None;
    app.request_discarding_action(PendingAction::NewProject);
    assert!(
        app.project().layers.is_empty(),
        "a saved project is discarded without a question"
    );

    // And the marker re-arms on the next change.
    let _ = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    assert!(app.has_unsaved_changes());
}

#[test]
fn loading_a_project_starts_it_clean() {
    let mut app = OxigisApp::new();
    let _ = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    assert!(app.has_unsaved_changes());
    app.load_project(Project::new("Loaded"));
    assert!(
        !app.has_unsaved_changes(),
        "the file on disk IS what is now open"
    );
    assert_eq!(app.title_line(), "Loaded");
}

// ---- Native project I/O seams ------------------------------------------

/// A shell that owns a filesystem, with a dirty project in it.
fn native_app() -> OxigisApp {
    let mut app = OxigisApp::new();
    app.set_native_project_io(true);
    let _ = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    app
}

#[test]
fn a_fresh_app_has_no_filesystem_until_a_shell_says_otherwise() {
    // The browser shell's contract: File ▸ Save is the copy-JSON modal, and
    // nothing is ever queued for a shell that cannot write.
    let mut app = OxigisApp::new();
    assert!(!app.native_project_io());
    assert!(app.project_path().is_none());
    assert!(app.recent_projects().is_empty());
    let _ = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    app.request_project_save(false);
    assert!(
        app.take_pending_project_save().is_none(),
        "a shell with no filesystem is never handed bytes to write",
    );
    assert!(matches!(app.io_dialog, Some(IoDialog::Save { .. })));
    assert!(!app.has_unsaved_changes(), "the document is on screen");
}

#[test]
fn save_writes_back_to_the_known_path_and_save_as_always_asks() {
    let mut app = native_app();
    // Never saved: even a plain Save has to ask where.
    app.request_project_save(false);
    let request = app
        .take_pending_project_save()
        .expect("a write must be queued");
    assert!(request.path.is_none(), "there is no file yet");
    assert!(
        request.content.contains("cities"),
        "the queued bytes are the project",
    );
    assert!(
        app.has_unsaved_changes(),
        "nothing is saved until the shell says the bytes landed",
    );

    // The shell reports where it landed.
    let path = std::path::PathBuf::from("/data/cities.oxigis.json");
    app.confirm_project_saved(path.clone());
    assert!(!app.has_unsaved_changes());
    assert_eq!(app.project_path(), Some(path.as_path()));
    assert_eq!(app.recent_projects(), std::slice::from_ref(&path));

    // Now Save writes back without asking, and Save As still asks.
    app.request_project_save(false);
    assert_eq!(
        app.take_pending_project_save().and_then(|r| r.path),
        Some(path.clone()),
    );
    app.request_project_save(true);
    assert_eq!(
        app.take_pending_project_save().map(|r| r.path),
        Some(None),
        "Save As never reuses the remembered path",
    );

    // A stamped camera and basemap are part of what gets written — the
    // difference between a project that reopens where the user was and one
    // that does not.
    assert!(app.project().basemap.is_some());
}

#[test]
fn a_failed_or_cancelled_save_leaves_the_project_unsaved() {
    let mut app = native_app();
    app.request_project_save(true);
    let _ = app.take_pending_project_save();
    app.report_project_save_failed("disk full");
    assert!(app.has_unsaved_changes());
    assert!(app.project_path().is_none());
    let status = app.status().unwrap_or_default();
    assert!(status.contains("disk full"), "{status}");

    app.request_project_save(true);
    let _ = app.take_pending_project_save();
    app.cancel_pending_project_io();
    assert!(app.has_unsaved_changes());
    assert!(app.project_path().is_none());
}

#[test]
fn new_and_open_go_through_the_shell_seam_once_confirmed() {
    let mut app = native_app();
    // Dirty: the question comes first, and nothing is queued behind it.
    app.request_project_open();
    assert!(matches!(
        app.io_dialog,
        Some(IoDialog::ConfirmDiscard { .. })
    ));
    assert!(app.take_pending_project_open().is_none());

    // Confirming runs it — and a shell with a filesystem gets a read request,
    // not the paste modal.
    app.io_dialog = None;
    app.run_pending_action(PendingAction::OpenProject);
    let request = app
        .take_pending_project_open()
        .expect("Open must reach the shell");
    assert!(request.path.is_none(), "the shell asks which file");
    assert!(app.io_dialog.is_none(), "no paste box on a native shell");

    // Open Recent names the file outright.
    let path = std::path::PathBuf::from("/data/osaka.oxigis.json");
    app.run_pending_action(PendingAction::OpenRecent(path.clone()));
    assert_eq!(
        app.take_pending_project_open().and_then(|r| r.path),
        Some(path),
    );
}

#[test]
fn a_project_that_arrived_without_a_file_forgets_the_previous_ones_path() {
    let mut app = native_app();
    app.confirm_project_saved(std::path::PathBuf::from("/data/cities.oxigis.json"));
    assert!(app.project_path().is_some());

    // A paste, a drop or File ▸ New must not leave Ctrl+S pointing at the
    // document that is no longer open.
    app.load_project(Project::new("Pasted"));
    assert!(app.project_path().is_none(), "a pasted project has no file");

    app.set_project_path(Some(std::path::PathBuf::from("/data/other.oxigis.json")));
    app.new_project();
    assert!(app.project_path().is_none(), "File ▸ New starts unattached");
}

#[test]
fn the_window_close_asks_once_and_confirms_exactly_once() {
    // Clean: nothing to ask about, confirmed immediately.
    let mut clean = OxigisApp::new();
    clean.set_native_project_io(true);
    clean.request_window_close();
    assert!(clean.take_confirmed_close());
    assert!(
        !clean.take_confirmed_close(),
        "the confirmation is take-once",
    );

    // Dirty: the question stands, and the app is NOT closing.
    let mut app = native_app();
    app.request_window_close();
    assert!(matches!(
        app.io_dialog,
        Some(IoDialog::ConfirmDiscard { .. })
    ));
    assert!(!app.take_confirmed_close());
    frame(&mut app);
    assert_eq!(app.project().layers.len(), 1, "drawing destroys nothing");
    assert!(!app.take_confirmed_close());

    // Discarding confirms it — and confirming does NOT clean the project,
    // which is exactly why the shell has to latch.
    app.io_dialog = None;
    app.run_pending_action(PendingAction::CloseWindow);
    assert!(app.take_confirmed_close());
    assert!(
        app.has_unsaved_changes(),
        "the shell's latch is load-bearing"
    );
}

/// "Save, then do this" — the chaining the confirmation's Save button arms.
#[test]
fn save_then_continue_runs_the_parked_action_only_after_a_real_save() {
    let mut app = native_app();
    app.after_save = Some(PendingAction::CloseWindow);
    app.request_project_save(true);
    let _ = app
        .take_pending_project_save()
        .expect("the parked quit queued a write");
    assert!(!app.take_confirmed_close(), "nothing is on disk yet");

    app.confirm_project_saved(std::path::PathBuf::from("/data/cities.oxigis.json"));
    assert!(
        app.take_confirmed_close(),
        "the save finished; the quit runs"
    );
    assert!(!app.has_unsaved_changes());

    // A save that FAILED must drop the parked action, and must not let a
    // later unrelated save fire it.
    app.after_save = Some(PendingAction::NewProject);
    app.request_project_save(true);
    let _ = app.take_pending_project_save();
    app.report_project_save_failed("disk full");
    app.confirm_project_saved(std::path::PathBuf::from("/data/cities.oxigis.json"));
    assert_eq!(
        app.project().layers.len(),
        1,
        "a dropped New must not ride the next successful save",
    );

    // Cancelling drops it the same way.
    app.after_save = Some(PendingAction::NewProject);
    app.cancel_pending_project_io();
    app.confirm_project_saved(std::path::PathBuf::from("/data/cities.oxigis.json"));
    assert_eq!(app.project().layers.len(), 1);
}

#[test]
fn a_second_save_in_one_frame_does_not_replace_the_first() {
    let mut app = native_app();
    // Saved once, so the two gestures below produce *distinguishable*
    // requests: a plain Save carries the known path, a Save As carries none.
    let path = std::path::PathBuf::from("/data/cities.oxigis.json");
    app.confirm_project_saved(path.clone());
    let _ = app
        .add_geojson_layer_from_text("more", POINTS, None)
        .expect("valid GeoJSON must be accepted");

    app.after_save = Some(PendingAction::CloseWindow);
    app.request_project_save(false);
    app.request_project_save(true);
    let request = app
        .take_pending_project_save()
        .expect("the first request survives");
    assert_eq!(
        request.path,
        Some(path),
        "the FIRST request is what the shell gets, not the Save As behind it",
    );
    assert!(app.take_pending_project_save().is_none(), "and only one");
    assert_eq!(
        app.after_save,
        Some(PendingAction::CloseWindow),
        "the parked action rides the request that survived",
    );
}

#[test]
fn the_recent_list_is_capped_deduplicated_and_most_recent_first() {
    let mut app = native_app();
    for index in 0..(crate::app::MAX_RECENT_PROJECTS + 4) {
        app.note_recent_project(std::path::PathBuf::from(format!("/data/p{index}.json")));
    }
    assert_eq!(app.recent_projects().len(), crate::app::MAX_RECENT_PROJECTS);
    assert_eq!(
        app.recent_projects()[0],
        std::path::PathBuf::from(format!(
            "/data/p{}.json",
            crate::app::MAX_RECENT_PROJECTS + 3
        )),
    );
    // Re-saving an already-listed project moves it up rather than duplicating.
    let again = app.recent_projects()[4].clone();
    app.confirm_project_saved(again.clone());
    assert_eq!(app.recent_projects()[0], again);
    assert_eq!(
        app.recent_projects()
            .iter()
            .filter(|path| **path == again)
            .count(),
        1,
    );

    // Seeding from a shell's store is bounded and de-duplicated on the way in.
    let mut hostile: Vec<std::path::PathBuf> = (0..100)
        .map(|index| std::path::PathBuf::from(format!("/data/q{index}.json")))
        .collect();
    hostile.push(std::path::PathBuf::from("/data/q0.json"));
    hostile.push(std::path::PathBuf::new());
    app.set_recent_projects(hostile);
    assert_eq!(app.recent_projects().len(), crate::app::MAX_RECENT_PROJECTS);
    assert!(
        app.recent_projects()
            .iter()
            .all(|path| !path.as_os_str().is_empty()),
    );
}

#[test]
fn the_window_title_names_the_project_and_marks_it_dirty() {
    let mut app = OxigisApp::new();
    assert_eq!(app.window_title(), "Untitled project \u{2014} OxiGIS");
    let _ = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    assert_eq!(app.window_title(), "Untitled project * \u{2014} OxiGIS");
    app.mark_saved();
    assert_eq!(app.window_title(), "Untitled project \u{2014} OxiGIS");
}

/// A document a shell read off disk goes through the same detection ladder as
/// every other load path, and says so when it is not a project at all.
#[test]
fn loading_a_document_answers_whether_it_was_one() {
    let mut app = OxigisApp::new();
    let project = Project::new("From disk");
    let text = project.to_json_string().expect("a project serializes");
    assert!(app.load_project_from_text(&text));
    assert_eq!(app.project().name, "From disk");
    assert!(!app.has_unsaved_changes());

    assert!(!app.load_project_from_text("{ not json"));
    assert_eq!(
        app.project().name,
        "From disk",
        "a failed parse changes nothing",
    );
    let status = app.status().unwrap_or_default();
    assert!(status.contains("not an OxiGIS project"), "{status}");
}

// ---- Zoom to layer -----------------------------------------------------

#[test]
fn zoom_to_selected_layer_flies_back_to_a_layer_panned_away_from() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    let fitted = app.map_view();
    // Pan and zoom out, the way a user browsing elsewhere would.
    let elsewhere = app
        .map_view()
        .with_center(LonLat::new(-40.0, 10.0))
        .with_zoom(3.0);
    app.map_panel.set_view(elsewhere);

    assert!(app.zoom_to_selected_layer());
    let back = app.map_view();
    assert!((back.zoom() - fitted.zoom()).abs() < 1e-6);
    assert!((back.center().lon - fitted.center().lon).abs() < 1e-6);
    assert!((back.center().lat - fitted.center().lat).abs() < 1e-6);

    // A layer with no known extent answers `false` rather than moving the
    // camera somewhere arbitrary.
    app.apply_layer_action(LayerAction::AddDemoXyzLayer);
    assert!(!app.zoom_to_selected_layer());
    app.selection = None;
    assert!(!app.zoom_to_selected_layer());
    let _ = id;
}

// ---- Processing result naming and reporting ---------------------------

#[test]
fn repeated_runs_over_one_layer_produce_tellable_apart_layers() {
    let mut app = OxigisApp::new();
    let source = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    let descriptor = synthetic_descriptor("synthetic-fc");
    let value = serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {"type":"Feature","properties":{},
             "geometry":{"type":"Point","coordinates":[139.767,35.681]}}
        ]
    });
    for _ in 0..3 {
        app.route_processing_result_from(&descriptor, Some(source), Ok(value.clone()));
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
        vec![
            "cities",
            "cities \u{2014} Synthetic",
            "cities \u{2014} Synthetic (2)",
            "cities \u{2014} Synthetic (3)",
        ],
        "each run must be tellable from the last"
    );
}

#[test]
fn a_result_that_cannot_become_a_layer_says_so_in_the_processing_window() {
    // `last_error` has no accessor, so the panel state's own `Debug` is the
    // observation — the alternative is widening `processing_panel`'s API,
    // which is not this module's to widen.
    let mut app = OxigisApp::new();
    let descriptor = synthetic_descriptor("synthetic-fc");
    // A FeatureCollection whose single feature names a geometry type that
    // does not exist: the routing accepts it (non-empty) and the parser
    // then refuses it, which is the branch that used to report nothing.
    let value = serde_json::json!({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Hyperboloid", "coordinates": [0.0, 0.0]}
        }]
    });
    app.route_processing_result(&descriptor, Ok(value));
    let report = format!("{:?}", app.processing);
    assert!(
        app.project().layers.is_empty(),
        "nothing was added: {report}"
    );
    assert!(
        report.contains("last_error: Some"),
        "a failed layer creation must not leave the window blank: {report}"
    );
}

#[test]
fn a_result_that_becomes_a_layer_confirms_it_in_the_processing_window() {
    let mut app = OxigisApp::new();
    let descriptor = synthetic_descriptor("synthetic-fc");
    let value = serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {"type":"Feature","properties":{},
             "geometry":{"type":"Point","coordinates":[139.767,35.681]}}
        ]
    });
    app.route_processing_result(&descriptor, Ok(value));
    let report = format!("{:?}", app.processing);
    assert!(
        report.contains("Synthetic result"),
        "a successful run must confirm itself: {report}"
    );
}

#[test]
fn the_drawn_strings_carry_no_dropped_line_continuations() {
    // Both of these shipped as wrapped literals whose continuation lines
    // lost their trailing `\`, so ~22 spaces became part of the sentence.
    for text in [
        super::VERTICAL_TITLE_HINT.to_string(),
        super::project_edit::undoable_budget_notice(),
    ] {
        assert!(
            !text.contains("   "),
            "a user-visible string must not carry an indentation run: {text:?}"
        );
    }
}
