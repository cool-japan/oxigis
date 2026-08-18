// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! App-level integration tests for the editing path: the choke point, the
//! propagation and persistence invariants it carries, and the undo/redo wiring.
//!
//! Separate from `app/tests.rs`, which is already close to the 2000-line limit.

use super::OxigisApp;
use crate::attribute_table::SYNTHETIC_COLUMN_COUNT;
use crate::edit::command::{EditError, EditTransaction, FeatureOp, set_vertex};
use crate::edit::form::FieldKind;
use crate::edit::hit;
use crate::edit::toolbar::EditAction;
use crate::edit::{EditMode, EditSelection, Handles, VertexRef};
use crate::local_input::LocalLayerOp;
use crate::style_panel::StyleKind;
use egui::Context;
use oxigeo::geojson::types::{Feature, FeatureCollection, Geometry, Point, Position};
use oxigis_core::{Color, FillStyle, Layer, LayerId, LayerKind, LayerStyle, Project, VectorSource};
use oxigis_render::LonLat;
use std::sync::Arc;

/// Two point features, the same fixture `app/tests.rs` uses.
const POINTS: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"Tokyo"},
     "geometry":{"type":"Point","coordinates":[139.767,35.681]}},
    {"type":"Feature","properties":{"name":"Osaka"},
     "geometry":{"type":"Point","coordinates":[135.502,34.702]}}]}"#;

/// An empty inline collection — what a freshly created edit layer, or one whose
/// last feature was deleted, stores.
const EMPTY: &str = r#"{"type":"FeatureCollection","features":[]}"#;

/// The layer's stored source, for the assertions about conversion.
fn source_of(app: &OxigisApp, id: LayerId) -> VectorSource {
    match app.project.layers.get(id).map(|layer| &layer.kind) {
        Some(LayerKind::Vector(source)) => source.clone(),
        other => panic!("expected a vector layer, got {other:?}"),
    }
}

/// The inline GeoJSON text a layer currently stores.
fn inline_text(app: &OxigisApp, id: LayerId) -> String {
    match source_of(app, id) {
        VectorSource::InlineGeoJson { geojson } => geojson,
        other => panic!("expected inline GeoJSON, got {other:?}"),
    }
}

/// The layer's features, as currently held by the app.
fn features_of(app: &OxigisApp, id: LayerId) -> Arc<FeatureCollection> {
    Arc::clone(app.local.feature_set(id).expect("the layer is loaded"))
}

/// A transaction that moves feature 0's only vertex to `to`.
fn move_first_vertex(app: &OxigisApp, id: LayerId, to: LonLat) -> EditTransaction {
    let features = features_of(app, id);
    let before = features.features[0].clone();
    let mut after = before.clone();
    set_vertex(&mut after, VertexRef::new(0), to).expect("a point always has vertex 0");
    EditTransaction {
        layer: id,
        label: "Move vertex",
        ops: vec![FeatureOp::Replace {
            index: 0,
            before: Box::new(before),
            after: Box::new(after),
        }],
        selection_before: None,
        selection_after: Some(EditSelection::feature(0)),
        coalesce: None,
    }
}

/// A transaction appending one point feature at `lon`/`lat`.
fn append_point(app: &OxigisApp, id: LayerId, lon: f64, lat: f64) -> EditTransaction {
    let index = app
        .local
        .feature_set(id)
        .map_or(0, |set| set.features.len());
    let point = Point::new(vec![lon, lat]).expect("two elements");
    EditTransaction::single(
        id,
        "Add point",
        FeatureOp::Add {
            index,
            feature: Box::new(Feature::new(Some(Geometry::Point(point)), None)),
        },
    )
}

/// A project holding one layer whose inline source is `geojson`.
fn project_with_inline(name: &str, geojson: &str) -> (Project, LayerId) {
    let mut project = Project::new("test");
    let id = project.layers.add(Layer::new(
        name,
        LayerKind::Vector(VectorSource::InlineGeoJson {
            geojson: geojson.to_string(),
        }),
    ));
    (project, id)
}

/// Drives one whole `ui` frame at a realistic window size, with `events`
/// delivered as this frame's raw input.
fn run_frame(app: &mut OxigisApp, ctx: &Context, events: Vec<egui::Event>) {
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        events,
        ..Default::default()
    };
    let _output = ctx.run_ui(raw_input, |ui| app.ui(ui));
}

#[test]
fn an_edit_replaces_the_arc_and_queues_exactly_one_add_op_for_the_right_id() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    let other = app
        .add_geojson_layer_from_text("more cities", POINTS, None)
        .expect("valid GeoJSON");
    let _drained = app.take_pending_local_ops();

    let before = features_of(&app, id);
    let other_before = features_of(&app, other);
    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(1.0, 2.0))));

    let after = features_of(&app, id);
    assert!(
        !Arc::ptr_eq(&before, &after),
        "an edit must replace the shared collection, never mutate it in place"
    );
    assert!(
        Arc::ptr_eq(&other_before, &features_of(&app, other)),
        "no other layer moves"
    );

    let ops = app.take_pending_local_ops();
    assert_eq!(ops.len(), 1, "one edit, one GPU op: {ops:?}");
    match &ops[0] {
        LocalLayerOp::Add(op_id, layer) => {
            assert_eq!(*op_id, id);
            assert!(
                Arc::ptr_eq(&layer.features_arc(), &after),
                "the feature store and the queued Add must carry the same Arc"
            );
        }
        other => panic!("expected one Add, got {other:?}"),
    }
}

#[test]
fn a_path_referenced_layer_becomes_inline_geojson_on_its_first_edit() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("roads", POINTS, Some("/data/roads.geojson"))
        .expect("valid GeoJSON");
    let filler = app
        .add_geojson_layer_from_text("filler", POINTS, None)
        .expect("valid GeoJSON");
    assert!(matches!(
        source_of(&app, id),
        VectorSource::LocalGeoJson { .. }
    ));
    let stack_before: Vec<LayerId> = app
        .project
        .layers
        .layers()
        .iter()
        .map(|layer| layer.id)
        .collect();

    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(1.0, 2.0))));
    assert!(matches!(
        source_of(&app, id),
        VectorSource::InlineGeoJson { .. }
    ));
    assert!(
        app.status()
            .is_some_and(|line| line.contains("roads.geojson") && line.contains("read-only")),
        "the conversion must be named once: {:?}",
        app.status()
    );
    let stack_after: Vec<LayerId> = app
        .project
        .layers
        .layers()
        .iter()
        .map(|layer| layer.id)
        .collect();
    assert_eq!(
        stack_before, stack_after,
        "rewriting the source must not reorder the layer stack"
    );
    assert_eq!(filler, stack_after[1]);

    // The notice is one-time: the layer is already inline the second time.
    app.set_status("cleared");
    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(3.0, 4.0))));
    assert_eq!(app.status(), Some("cleared"));
}

#[test]
fn the_inline_text_reparses_to_the_feature_set_after_every_edit() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, Some("/data/cities.geojson"))
        .expect("valid GeoJSON");

    for (index, target) in [LonLat::new(1.0, 2.0), LonLat::new(3.0, 4.0)]
        .into_iter()
        .enumerate()
    {
        assert!(app.commit_edit(move_first_vertex(&app, id, target)));
        let text = inline_text(&app, id);
        let reparsed = oxigeo::geojson::reader::feature_collection_from_str(&text)
            .expect("the stored text must be readable");
        assert_eq!(
            &reparsed,
            features_of(&app, id).as_ref(),
            "edit {index}: the stored text and the live collection disagree"
        );
    }
}

#[test]
fn a_serialization_failure_leaves_project_features_and_stack_untouched() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, Some("/data/cities.geojson"))
        .expect("valid GeoJSON");
    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(1.0, 2.0))));
    let _drained = app.take_pending_local_ops();

    let source_before = source_of(&app, id);
    let features_before = features_of(&app, id);
    let depth_before = app.undo.depth();

    // An edited collection that cannot be written. `serde_json` writes even a
    // non-finite float as `null` rather than failing, so the failure is
    // injected instead of contrived — the contract under test is the ordering,
    // not the writer.
    let mut edited = (*features_before).clone();
    let mut moved = edited.features[0].clone();
    set_vertex(&mut moved, VertexRef::new(0), LonLat::new(9.0, 9.0)).expect("in range");
    edited.features[0] = moved;

    let outcome = app.apply_feature_collection_with(id, Arc::new(edited), |_collection| {
        Err("the disk caught fire".to_string())
    });
    assert_eq!(
        outcome,
        Err(EditError::Serialize("the disk caught fire".to_string()))
    );
    assert_eq!(
        source_of(&app, id),
        source_before,
        "the source must not move"
    );
    assert!(
        Arc::ptr_eq(&features_before, &features_of(&app, id)),
        "the feature store must not move"
    );
    assert_eq!(
        app.local.pending_op_count(),
        0,
        "nothing may be queued for the GPU"
    );
    assert_eq!(app.undo.depth(), depth_before);
}

#[test]
fn undo_restores_both_the_arc_and_the_inline_text_and_the_selection() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    app.edit.set_selection(Some(EditSelection::feature(1)));
    let features_before = features_of(&app, id);
    // Compared as data, not as bytes: the text the layer was *added* with is
    // the user's own formatting, while every rewrite is the writer's.
    let stored = |app: &OxigisApp| -> FeatureCollection {
        oxigeo::geojson::reader::feature_collection_from_str(&inline_text(app, id))
            .expect("the stored text must be readable")
    };
    let text_before = stored(&app);

    let mut transaction = move_first_vertex(&app, id, LonLat::new(1.0, 2.0));
    transaction.selection_before = Some(EditSelection::feature(1));
    transaction.selection_after = Some(EditSelection::feature(0));
    assert!(app.commit_edit(transaction));
    assert_eq!(app.edit.selection(), Some(EditSelection::feature(0)));
    assert_ne!(stored(&app), text_before);
    assert_eq!(app.undo.depth(), (1, 0));

    assert!(app.undo_once());
    assert_eq!(app.undo.depth(), (0, 1));
    assert_eq!(
        features_of(&app, id).as_ref(),
        features_before.as_ref(),
        "undo must restore the data exactly"
    );
    assert_eq!(
        stored(&app),
        text_before,
        "undo must rewrite the stored text too"
    );
    assert_eq!(
        app.edit.selection(),
        Some(EditSelection::feature(1)),
        "undo must restore the selection the edit started from"
    );
    assert!(
        app.status()
            .is_some_and(|line| line.contains("Move vertex"))
    );

    assert!(app.redo_once());
    assert_eq!(app.undo.depth(), (1, 0));
    assert_eq!(app.edit.selection(), Some(EditSelection::feature(0)));
    assert_ne!(stored(&app), text_before);
}

#[test]
fn a_user_layer_style_visibility_and_opacity_survive_an_edit() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    let explicit = LayerStyle::Fill(FillStyle::new(Color::from_rgb(1, 2, 3)));
    app.project.styles.insert(id, explicit.clone().into());
    app.project
        .layers
        .toggle_visibility(id)
        .expect("the layer is present");
    app.project
        .layers
        .set_opacity(id, 0.25)
        .expect("the layer is present");
    let _drained = app.take_pending_local_ops();

    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(1.0, 2.0))));

    let layer = app.project.layers.get(id).expect("still present");
    assert!(!layer.visible);
    assert!((layer.opacity() - 0.25).abs() < 1e-6);
    assert_eq!(
        app.project
            .styles
            .get(&id)
            .map(oxigis_core::LayerStyleSet::base),
        Some(&explicit)
    );

    let ops = app.take_pending_local_ops();
    match ops.as_slice() {
        [LocalLayerOp::Add(op_id, local)] => {
            assert_eq!(*op_id, id);
            assert_eq!(
                local.style().base(),
                &explicit,
                "the GPU copy keeps the style"
            );
            assert!(!local.visible());
            assert!((local.opacity() - 0.25).abs() < 1e-6);
        }
        other => panic!("expected one Add, got {other:?}"),
    }
}

#[test]
fn the_default_styles_entry_is_not_overwritten_by_an_edit() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    let derived = app
        .local
        .default_style(id)
        .cloned()
        .expect("the add path records one");
    let explicit = LayerStyle::Fill(FillStyle::new(Color::from_rgb(9, 9, 9)));
    assert_ne!(derived.base(), &explicit);
    app.project.styles.insert(id, explicit.clone().into());

    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(1.0, 2.0))));

    assert_eq!(
        app.local.default_style(id),
        Some(&derived),
        "an edit must not re-derive the layer's remembered default style"
    );
    assert_eq!(
        app.project
            .styles
            .get(&id)
            .map(oxigis_core::LayerStyleSet::base),
        Some(&explicit)
    );

    // Removing the explicit style therefore still restores what the layer was
    // born with, exactly as it would have before the edit.
    app.project.styles.remove(&id);
    let _drained = app.take_pending_local_ops();
    app.sync_local_style(Some(explicit.into()));
    match app.take_pending_local_ops().as_slice() {
        [LocalLayerOp::SetStyle(op_id, style)] => {
            assert_eq!(*op_id, id);
            assert_eq!(style, &derived);
        }
        other => panic!("expected one SetStyle, got {other:?}"),
    }
}

#[test]
fn loading_a_project_and_new_project_each_clear_the_stack() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(1.0, 2.0))));
    app.edit.set_selection(Some(EditSelection::feature(0)));
    assert_eq!(app.undo.depth(), (1, 0));
    let epoch = app.undo.epoch();

    let (project, _other) = project_with_inline("elsewhere", POINTS);
    app.load_project(project);
    assert_eq!(app.undo.depth(), (0, 0));
    assert_ne!(app.undo.epoch(), epoch);
    assert_eq!(app.edit.selection(), None);
    assert_eq!(app.edit.target(), None);

    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(1.0, 2.0))));
    assert_eq!(app.undo.depth(), (1, 0));
    let epoch = app.undo.epoch();
    app.new_project();
    assert_eq!(app.undo.depth(), (0, 0));
    assert_ne!(app.undo.epoch(), epoch);
}

#[test]
fn removing_a_layer_is_itself_undoable_and_keeps_the_history() {
    use crate::layer_panel::LayerAction;

    let mut app = OxigisApp::new();
    let first = app
        .add_geojson_layer_from_text("first", POINTS, None)
        .expect("valid GeoJSON");
    let second = app
        .add_geojson_layer_from_text("second", POINTS, None)
        .expect("valid GeoJSON");
    assert!(app.commit_edit(move_first_vertex(&app, first, LonLat::new(1.0, 2.0))));
    assert!(app.commit_edit(move_first_vertex(&app, second, LonLat::new(3.0, 4.0))));
    assert!(app.commit_edit(move_first_vertex(&app, first, LonLat::new(5.0, 6.0))));
    assert_eq!(app.undo.depth(), (3, 0));
    let features_before = Arc::clone(app.local.feature_set(first).expect("loaded"));
    let position_before = app
        .project
        .layers
        .layers()
        .iter()
        .position(|layer| layer.id == first)
        .expect("in the stack");

    // The removal RECORDS instead of pruning: the edit history survives and
    // the removal itself becomes the newest undo step.
    app.apply_layer_action(LayerAction::Remove(first));
    assert!(app.project.layers.get(first).is_none());
    assert_eq!(
        app.undo.depth(),
        (4, 0),
        "3 edits + the removal, nothing pruned"
    );
    assert!(
        app.status()
            .is_some_and(|line| line.contains("Ctrl+Z puts it back")),
        "the user is told: {:?}",
        app.status()
    );

    // One Ctrl+Z restores the layer — value, position, features.
    assert!(app.undo_once());
    let restored = app.project.layers.get(first).expect("the layer is back");
    assert_eq!(restored.name, "first");
    assert_eq!(
        app.project
            .layers
            .layers()
            .iter()
            .position(|layer| layer.id == first),
        Some(position_before),
        "the stack slot is restored"
    );
    let features_after = app.local.feature_set(first).expect("features restored");
    assert!(
        Arc::ptr_eq(&features_before, features_after),
        "the exact Arc came back"
    );

    // The older edit entries are still applicable after the round trip.
    assert!(app.undo_once(), "the last vertex edit undoes");
    assert!(app.undo_once());
    assert!(app.undo_once());
    assert_eq!(app.undo.depth(), (0, 4));

    // And redoing the removal removes the layer again.
    assert!(app.redo_once());
    assert!(app.redo_once());
    assert!(app.redo_once());
    assert!(app.redo_once());
    assert!(app.project.layers.get(first).is_none());
}

#[test]
fn editing_a_layer_whose_features_are_not_loaded_is_refused() {
    let mut app = OxigisApp::new();
    let mut project = Project::new("test");
    let id = project.layers.add(Layer::new(
        "pending",
        LayerKind::Vector(VectorSource::LocalGeoJson {
            path: "/data/pending.geojson".to_string(),
        }),
    ));
    app.load_project(project);
    assert!(
        app.local.feature_set(id).is_none(),
        "a path reference is not loaded until a shell reads it"
    );

    let transaction = EditTransaction::single(
        id,
        "Add point",
        FeatureOp::Add {
            index: 0,
            feature: Box::new(Feature::new(
                Some(Geometry::Point(Point::new(vec![0.0, 0.0]).expect("valid"))),
                None,
            )),
        },
    );
    assert!(!app.commit_edit(transaction));
    assert_eq!(app.undo.depth(), (0, 0), "nothing is recorded");
    assert!(
        app.status().is_some_and(|line| line.contains("not loaded")),
        "got {:?}",
        app.status()
    );

    // A layer the project does not hold at all is refused just as cleanly.
    let ghost = LayerId::new();
    assert_eq!(
        app.apply_feature_collection(ghost, Arc::new(FeatureCollection::empty())),
        Err(EditError::LayerGone(ghost))
    );

    // As is one that is not a local vector layer at all.
    let raster = app.project.layers.add(Layer::new(
        "basemap",
        LayerKind::Raster(oxigis_core::RasterSource::xyz(
            "https://example.test/{z}/{x}/{y}.png",
        )),
    ));
    assert_eq!(
        app.apply_feature_collection(raster, Arc::new(FeatureCollection::empty())),
        Err(EditError::LayerGone(raster))
    );
}

#[test]
fn an_empty_layer_survives_add_draw_one_point_save_and_reload() {
    let mut app = OxigisApp::new();
    let added = app
        .local
        .add_empty_vector_layer(&mut app.project, "sketch", StyleKind::Circle)
        .expect("an empty layer is a legitimate request");
    let id = added.id;
    assert_eq!(added.feature_count, 0);
    assert!(matches!(
        source_of(&app, id),
        VectorSource::InlineGeoJson { .. }
    ));
    assert_eq!(
        app.local
            .feature_set(id)
            .map(|features| features.features.len()),
        Some(0)
    );

    assert!(app.commit_edit(append_point(&app, id, 139.7, 35.7)));
    assert_eq!(features_of(&app, id).features.len(), 1);

    let json = app
        .project
        .to_json_string()
        .expect("the project serializes");
    let reloaded = Project::from_json_string(&json).expect("and parses back");
    let mut second = OxigisApp::new();
    second.load_project(reloaded);
    assert_eq!(
        second.status(),
        Some("Project loaded."),
        "no layer may be reported as unrebuildable"
    );
    let restored = second
        .local
        .feature_set(id)
        .expect("the edited layer is loaded");
    assert_eq!(restored.features.len(), 1);
    assert_eq!(
        restored.features[0]
            .geometry
            .as_ref()
            .map(oxigeo::geojson::types::Geometry::geometry_type),
        Some(oxigeo::geojson::types::GeometryType::Point)
    );
}

#[test]
fn an_empty_inline_collection_rebuilds_instead_of_becoming_a_pending_stub() {
    let (project, id) = project_with_inline("emptied", EMPTY);
    let mut app = OxigisApp::new();
    app.load_project(project);
    assert_eq!(
        app.status(),
        Some("Project loaded."),
        "an empty layer must not be reported as a problem"
    );
    let features = app
        .local
        .feature_set(id)
        .expect("an empty collection is still a loaded collection");
    assert_eq!(features.features.len(), 0);

    // The attribute table therefore binds it rather than showing the
    // "loading…" stub forever.
    app.apply_layer_action(crate::layer_panel::LayerAction::Select(id));
    assert!(app.selected_table_source().is_some());

    // And it is editable straight away.
    assert!(app.commit_edit(append_point(&app, id, 1.0, 2.0)));
    assert_eq!(features_of(&app, id).features.len(), 1);
}

#[test]
fn at_most_one_undo_is_processed_per_frame() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(1.0, 2.0))));
    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(3.0, 4.0))));
    assert_eq!(app.undo.depth(), (2, 0));

    let ctx = Context::default();
    run_frame(&mut app, &ctx, Vec::new());

    // Keyboard auto-repeat delivering two presses in one frame must still undo
    // exactly once: each one would otherwise re-tessellate the whole layer.
    let undo_key = || egui::Event::Key {
        key: egui::Key::Z,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::COMMAND,
    };
    run_frame(&mut app, &ctx, vec![undo_key(), undo_key()]);
    assert_eq!(app.undo.depth(), (1, 1));

    run_frame(&mut app, &ctx, vec![undo_key()]);
    assert_eq!(app.undo.depth(), (0, 2));

    let redo_key = || egui::Event::Key {
        key: egui::Key::Z,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
    };
    run_frame(&mut app, &ctx, vec![redo_key(), redo_key()]);
    assert_eq!(app.undo.depth(), (1, 1));
}

/// Three points spread far apart, so a click can address one of them without
/// falling inside the next one's tolerance.
const SPREAD: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"c"},
     "geometry":{"type":"Point","coordinates":[0.0,0.0]}},
    {"type":"Feature","properties":{"name":"a"},
     "geometry":{"type":"Point","coordinates":[20.0,0.0]}},
    {"type":"Feature","properties":{"name":"b"},
     "geometry":{"type":"Point","coordinates":[-20.0,0.0]}}]}"#;

/// A primary press at `at` and its release one frame later — what egui calls a
/// click.
fn click_frames(app: &mut OxigisApp, ctx: &Context, at: egui::Pos2) {
    let button = |pressed| egui::Event::PointerButton {
        pos: at,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    run_frame(app, ctx, vec![egui::Event::PointerMoved(at), button(true)]);
    run_frame(app, ctx, vec![button(false)]);
}

/// A primary drag from `from` to `to`, over the frames egui needs before it
/// calls the gesture a drag rather than a click.
fn drag_frames(app: &mut OxigisApp, ctx: &Context, from: egui::Pos2, to: egui::Pos2) {
    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    run_frame(
        app,
        ctx,
        vec![egui::Event::PointerMoved(from), button(from, true)],
    );
    run_frame(app, ctx, vec![egui::Event::PointerMoved(to)]);
    run_frame(app, ctx, vec![button(to, false)]);
}

/// Where the layer's feature `index` lands on screen, as of the last frame.
fn feature_screen_pos(app: &OxigisApp, id: LayerId, index: usize) -> egui::Pos2 {
    let rect = app.map_rect().expect("a frame has run");
    let features = features_of(app, id);
    let point = match features.features[index].geometry.as_ref() {
        Some(Geometry::Point(point)) => point.coordinates.clone(),
        other => panic!("expected a point, got {other:?}"),
    };
    let px = app
        .map_view()
        .lon_lat_to_screen(LonLat::new(point[0], point[1]));
    rect.min + egui::vec2(px[0], px[1])
}

#[test]
fn edit_mode_off_consumes_no_click_and_never_suppresses_pan() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("spread", SPREAD, None)
        .expect("valid GeoJSON");
    let ctx = Context::default();
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.edit.mode(), EditMode::Off);

    // A click straight onto a feature does nothing at all while editing is off.
    let on_feature = feature_screen_pos(&app, id, 0);
    click_frames(&mut app, &ctx, on_feature);
    assert_eq!(
        app.edit.selection(),
        None,
        "with editing off a click must fall through unconsumed"
    );

    // The control, taken FIRST and at the very same screen position: the
    // assertion above cannot be passing merely because nothing was ever
    // clicked. Before the pan, deliberately — a post-pan control click depends
    // on where the camera happened to stop, and on how wide the side panels
    // drew that frame, which is a pixel margin no assertion should rest on.
    app.apply_edit_action(EditAction::SetMode(EditMode::Select));
    run_frame(&mut app, &ctx, Vec::new());
    let control = feature_screen_pos(&app, id, 0);
    click_frames(&mut app, &ctx, control);
    assert_eq!(
        app.edit.selection(),
        Some(EditSelection::feature(0)),
        "the same click in Select mode picks the feature, so the click was real"
    );

    // And a drag pans the camera exactly as it did before this module existed.
    app.apply_edit_action(EditAction::SetMode(EditMode::Off));
    run_frame(&mut app, &ctx, Vec::new());
    let rect = app.map_rect().expect("a frame has run");
    let before = app.map_view().center();
    drag_frames(
        &mut app,
        &ctx,
        rect.center(),
        rect.center() - egui::vec2(120.0, 0.0),
    );
    let after = app.map_view().center();
    assert!(
        (after.lon - before.lon).abs() > 1e-6,
        "EditMode::Off must never suppress a pan: {before:?} -> {after:?}"
    );
}

#[test]
fn map_selection_pushed_to_the_table_uses_the_source_index_under_an_active_sort() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("spread", SPREAD, None)
        .expect("valid GeoJSON");
    let ctx = Context::default();
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.table_panel.bound_layer(), Some(id));

    // Sort by the one property column, so the display order stops matching the
    // source order — the only condition under which the two index spaces differ.
    app.table_panel.toggle_sort(SYNTHETIC_COLUMN_COUNT);
    run_frame(&mut app, &ctx, Vec::new());
    assert_ne!(
        app.table_panel.display_order(),
        [0, 1, 2],
        "the sort must actually reorder, or this test proves nothing"
    );

    app.edit.set_selection(Some(EditSelection::feature(0)));
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(
        app.table_panel.selected_feature(),
        Some(0),
        "the map speaks source indices; feeding one to select_visible_row would \
         have selected a different feature entirely"
    );
}

#[test]
fn a_table_row_click_is_adopted_into_the_edit_selection_and_does_not_ping_pong() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("spread", SPREAD, None)
        .expect("valid GeoJSON");
    let ctx = Context::default();
    run_frame(&mut app, &ctx, Vec::new());
    app.table_panel.toggle_sort(SYNTHETIC_COLUMN_COUNT);
    run_frame(&mut app, &ctx, Vec::new());

    // Exactly what the panel does when a row is clicked: resolve the visible row
    // into a source index and keep that.
    let source = app.table_panel.display_order()[0];
    assert_ne!(source, 0, "the sort put a different feature on top");
    app.table_panel.select_visible_row(0);
    app.sync_table_selection(true);
    assert_eq!(app.edit.selection(), Some(EditSelection::feature(source)));

    // Every later frame is a no-click frame, in which the app re-asserts the
    // same value — so the two writers cannot alternate.
    for _ in 0..3 {
        run_frame(&mut app, &ctx, Vec::new());
        assert_eq!(app.edit.selection(), Some(EditSelection::feature(source)));
        assert_eq!(app.table_panel.selected_feature(), Some(source));
    }
    assert_eq!(app.selection(), Some(id));
}

#[test]
fn delete_feature_clears_the_selection_and_undo_restores_it() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    let ctx = Context::default();
    run_frame(&mut app, &ctx, Vec::new());

    assert!(
        !app.delete_selected_feature(),
        "with nothing picked there is nothing to delete"
    );
    assert!(
        app.status()
            .is_some_and(|status| status.contains("feature")),
        "a refusal always says why: {:?}",
        app.status()
    );

    app.edit.set_selection(Some(EditSelection::feature(0)));
    let before = features_of(&app, id);
    assert!(app.delete_selected_feature());
    assert_eq!(features_of(&app, id).features.len(), 1);
    assert_eq!(
        app.edit.selection(),
        None,
        "the deleted feature's index must not survive it"
    );
    assert!(inline_text(&app, id).contains("135.502"), "Osaka survives");

    assert!(app.undo_once());
    let restored = features_of(&app, id);
    assert_eq!(restored.features.len(), 2);
    assert_eq!(restored.features, before.features);
    assert_eq!(
        app.edit.selection(),
        Some(EditSelection::feature(0)),
        "undo restores exactly the selection the transaction recorded"
    );

    // The toolbar's delete button reaches the same path.
    assert!(app.redo_once());
    assert_eq!(features_of(&app, id).features.len(), 1);
    app.edit.set_selection(Some(EditSelection::feature(0)));
    app.apply_edit_action(EditAction::DeleteFeature);
    assert!(features_of(&app, id).features.is_empty());
}

#[test]
fn a_new_edit_layer_is_selected_latched_and_leaves_the_camera_where_it_was() {
    let mut app = OxigisApp::new();
    let ctx = Context::default();
    run_frame(&mut app, &ctx, Vec::new());
    let before = app.map_view();

    app.apply_edit_action(EditAction::NewLayer(StyleKind::Fill));
    let id = app.selection().expect("the new layer is selected");
    assert_eq!(app.edit.mode(), EditMode::DrawPolygon);
    assert_eq!(features_of(&app, id).features.len(), 0);
    assert_eq!(
        source_of(&app, id),
        VectorSource::InlineGeoJson {
            geojson: r#"{"type":"FeatureCollection","features":[]}"#.to_string(),
        },
        "an edit layer is inline from birth, so it needs no conversion later"
    );
    assert_eq!(
        (
            app.map_view().center().lon,
            app.map_view().center().lat,
            app.map_view().zoom()
        ),
        (before.center().lon, before.center().lat, before.zoom()),
        "an empty layer's extent is the whole world; zooming to it would throw \
         away wherever the user was about to draw"
    );

    // The whole frame draws with an empty target layer selected.
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.edit.selection(), None);
}

#[test]
fn the_mode_keys_and_escape_walk_the_ladder_without_touching_committed_data() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("spread", SPREAD, None)
        .expect("valid GeoJSON");
    let ctx = Context::default();
    run_frame(&mut app, &ctx, Vec::new());

    let key = |key| {
        vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }]
    };
    for (pressed, expected) in [
        (egui::Key::V, EditMode::Select),
        (egui::Key::P, EditMode::DrawPoint),
        (egui::Key::L, EditMode::DrawLine),
        (egui::Key::G, EditMode::DrawPolygon),
        (egui::Key::B, EditMode::Off),
    ] {
        run_frame(&mut app, &ctx, key(pressed));
        assert_eq!(app.edit.mode(), expected);
    }

    run_frame(&mut app, &ctx, key(egui::Key::V));
    app.edit.set_selection(Some(EditSelection::feature(1)));
    run_frame(&mut app, &ctx, key(egui::Key::Escape));
    assert_eq!(app.edit.selection(), None, "first rung: the picked feature");
    run_frame(&mut app, &ctx, key(egui::Key::Escape));
    assert_eq!(app.edit.mode(), EditMode::Off, "second rung: the tool");

    // Delete is inert with editing off, so a stray key cannot destroy data from
    // a mode that shows no editing affordance at all.
    app.edit.set_selection(Some(EditSelection::feature(1)));
    run_frame(&mut app, &ctx, key(egui::Key::Delete));
    assert_eq!(features_of(&app, id).features.len(), 3);
    assert_eq!(app.undo.depth(), (0, 0));

    run_frame(&mut app, &ctx, key(egui::Key::V));
    app.edit.set_selection(Some(EditSelection::feature(1)));
    run_frame(&mut app, &ctx, key(egui::Key::Delete));
    assert_eq!(features_of(&app, id).features.len(), 2);
    assert_eq!(app.undo.depth(), (1, 0));
}

/// One big square, far enough across that every ring segment earns a midpoint
/// ghost and no two handles fall inside one another's grab radius.
const SQUARE: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"plot"},
     "geometry":{"type":"Polygon","coordinates":[[
        [-10.0,-10.0],[10.0,-10.0],[10.0,10.0],[-10.0,10.0],[-10.0,-10.0]]]}}]}"#;

/// A triangle: the smallest legal ring, so one more delete would destroy it.
const TRIANGLE: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"tri"},
     "geometry":{"type":"Polygon","coordinates":[[
        [-10.0,-10.0],[10.0,-10.0],[0.0,10.0],[-10.0,-10.0]]]}}]}"#;

/// The exterior ring of feature `index`, as stored.
fn ring_of(app: &OxigisApp, id: LayerId, index: usize) -> Vec<Position> {
    match features_of(app, id).features[index].geometry.as_ref() {
        Some(Geometry::Polygon(polygon)) => {
            polygon.coordinates.first().cloned().expect("an exterior")
        }
        other => panic!("expected a polygon, got {other:?}"),
    }
}

/// Where a vertex handle — or, with `midpoint`, a segment ghost — is drawn.
fn handle_screen_pos(
    app: &OxigisApp,
    id: LayerId,
    index: usize,
    at: VertexRef,
    midpoint: bool,
) -> egui::Pos2 {
    let rect = app.map_rect().expect("a frame has run");
    let features = features_of(app, id);
    let geometry = features.features[index]
        .geometry
        .as_ref()
        .expect("the feature has geometry");
    let position = hit::handle_position(geometry, at, midpoint, app.map_view(), 1.0)
        .expect("the handle exists");
    hit::to_screen(app.map_view(), rect.min, 1.0, position)
}

/// A camera that frames the fixtures below with room to spare.
///
/// Set explicitly rather than left to `add_geojson_layer_from_text`'s
/// zoom-to-extent: that runs before any frame has sized the panel, so it fits
/// the layer to a viewport the central panel turns out not to have, and the
/// corners — which are exactly what these tests grab — can land off screen.
fn frame_the_fixture(app: &mut OxigisApp) {
    app.map_panel = crate::map_view::MapPanelState::new(0.0, 0.0, 3.0, [1024.0, 768.0]);
}

/// The app in `Select` mode with feature 0 of `geojson` picked, one frame in.
fn picked(geojson: &str, ctx: &Context) -> (OxigisApp, LayerId) {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("plot", geojson, None)
        .expect("valid GeoJSON");
    frame_the_fixture(&mut app);
    run_frame(&mut app, ctx, Vec::new());
    app.apply_edit_action(EditAction::SetMode(EditMode::Select));
    app.edit.set_selection(Some(EditSelection::feature(0)));
    run_frame(&mut app, ctx, Vec::new());
    assert_eq!(
        app.edit.handles(),
        Handles::Active,
        "the fixture must actually show handles, or the drags below grab nothing"
    );
    (app, id)
}

#[test]
fn dragging_ring_handle_zero_moves_the_closing_vertex_and_suppresses_the_pan() {
    let ctx = Context::default();
    let (mut app, id) = picked(SQUARE, &ctx);
    let before = ring_of(&app, id, 0);
    let camera_before = app.map_view().center();

    let from = handle_screen_pos(&app, id, 0, VertexRef::new(0), false);
    drag_frames(&mut app, &ctx, from, from + egui::vec2(40.0, -30.0));

    let after = ring_of(&app, id, 0);
    assert_eq!(after.len(), 5, "a ring keeps its closing position");
    assert_eq!(
        after[4], after[0],
        "moving handle 0 moves the closing vertex with it, or the ring comes apart"
    );
    assert_ne!(after[0], before[0], "the vertex actually moved");
    assert_eq!(
        after[1..4],
        before[1..4],
        "and nothing else in the ring did"
    );

    let camera_after = app.map_view().center();
    assert!(
        (camera_after.lon - camera_before.lon).abs() < 1e-9
            && (camera_after.lat - camera_before.lat).abs() < 1e-9,
        "a drag that starts on a handle must not pan: {camera_before:?} -> \
         {camera_after:?}"
    );

    assert_eq!(app.undo.depth(), (1, 0), "one gesture, one undo step");
    assert_eq!(
        app.undo.peek_undo().map(|entry| entry.label),
        Some("Move vertex")
    );
    assert_eq!(
        app.edit.selection(),
        Some(EditSelection::vertex(0, VertexRef::new(0)))
    );
    assert!(
        app.edit.drag().is_none(),
        "the gesture ended with the button"
    );

    // And undo puts every position back, through the same choke point.
    app.undo_once();
    assert_eq!(ring_of(&app, id, 0), before);
}

#[test]
fn dragging_a_midpoint_ghost_inserts_a_vertex_and_keeps_the_ring_closed() {
    let ctx = Context::default();
    let (mut app, id) = picked(SQUARE, &ctx);
    let before = ring_of(&app, id, 0);

    // The wrap segment's ghost: the one that inserts past the last open
    // position, which is the case a naive implementation gets wrong.
    let from = handle_screen_pos(&app, id, 0, VertexRef::new(4), true);
    drag_frames(&mut app, &ctx, from, from + egui::vec2(-35.0, 0.0));

    let after = ring_of(&app, id, 0);
    assert_eq!(after.len(), 6, "one more stored position: {after:?}");
    assert_eq!(after[5], after[0], "and the ring is still closed");
    assert_eq!(
        after[..4],
        before[..4],
        "the original vertices are untouched"
    );
    assert_eq!(
        app.undo.peek_undo().map(|entry| entry.label),
        Some("Insert vertex")
    );
    assert_eq!(
        app.edit.selection(),
        Some(EditSelection::vertex(0, VertexRef::new(4)))
    );

    app.undo_once();
    assert_eq!(ring_of(&app, id, 0), before);
}

#[test]
fn deleting_a_ring_vertex_recloses_the_ring_and_undo_restores_it() {
    let ctx = Context::default();
    let (mut app, id) = picked(SQUARE, &ctx);
    let before = ring_of(&app, id, 0);
    app.edit
        .set_selection(Some(EditSelection::vertex(0, VertexRef::new(0))));

    // Through the toolbar's button, which is the path the web shell has to rely
    // on when the browser eats the key.
    app.apply_edit_action(EditAction::DeleteVertex);
    let after = ring_of(&app, id, 0);
    assert_eq!(after.len(), 4, "four stored, three open");
    assert_eq!(after[3], after[0]);
    assert_eq!(after[0], before[1], "vertex 0 is the one that went");
    assert_eq!(
        app.undo.peek_undo().map(|entry| entry.label),
        Some("Delete vertex")
    );
    assert_eq!(
        app.edit.selection(),
        Some(EditSelection::feature(0)),
        "the vertex is gone, so the selection may not still name it"
    );

    app.undo_once();
    assert_eq!(ring_of(&app, id, 0), before);
    assert_eq!(
        app.edit.selection(),
        Some(EditSelection::vertex(0, VertexRef::new(0))),
        "undo restores the selection the transaction recorded"
    );

    // And the `Delete` key reaches the same action when a vertex is picked.
    run_frame(&mut app, &ctx, Vec::new());
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::Key {
            key: egui::Key::Delete,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
    assert_eq!(ring_of(&app, id, 0).len(), 4);
    assert_eq!(
        features_of(&app, id).features.len(),
        1,
        "a picked vertex must not take its whole feature with it"
    );
}

#[test]
fn deleting_a_vertex_below_the_minimum_is_refused_and_says_why() {
    let ctx = Context::default();
    let (mut app, id) = picked(TRIANGLE, &ctx);
    let before = ring_of(&app, id, 0);
    app.edit
        .set_selection(Some(EditSelection::vertex(0, VertexRef::new(1))));

    assert!(!app.delete_selected_vertex());
    assert_eq!(ring_of(&app, id, 0), before, "nothing was destroyed");
    assert_eq!(app.undo.depth(), (0, 0), "and nothing was recorded");
    let status = app.status.clone().expect("a refusal always says why");
    assert_eq!(
        status,
        EditError::TooFewVertices { have: 2, need: 3 }.to_string()
    );

    // With no vertex picked at all the button says what to do instead.
    app.edit.set_selection(Some(EditSelection::feature(0)));
    assert!(!app.delete_selected_vertex());
    let status = app.status.clone().expect("a refusal always says why");
    assert!(status.contains("vertex handle"), "{status}");
}

#[test]
fn escape_during_a_drag_restores_the_geometry_and_holds_the_pan_gate_shut() {
    let ctx = Context::default();
    let (mut app, id) = picked(SQUARE, &ctx);
    let before = ring_of(&app, id, 0);
    let camera_before = app.map_view().center();
    let from = handle_screen_pos(&app, id, 0, VertexRef::new(1), false);

    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(from), button(from, true)],
    );
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(from + egui::vec2(50.0, 50.0))],
    );
    assert!(app.edit.drag().is_some(), "the gesture is live");

    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
    assert!(app.edit.drag().is_none(), "Escape drops the gesture");
    assert_eq!(
        ring_of(&app, id, 0),
        before,
        "the pre-drag geometry was never committed, so there is nothing to undo"
    );
    assert_eq!(app.undo.depth(), (0, 0));

    // The button is still down: the camera must not seize it and lurch away
    // under the pointer the user is still holding.
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(from + egui::vec2(160.0, 0.0))],
    );
    let camera_after = app.map_view().center();
    assert!(
        (camera_after.lon - camera_before.lon).abs() < 1e-9,
        "a cancelled gesture must not hand its still-held button to the camera: \
         {camera_before:?} -> {camera_after:?}"
    );

    // Once the button comes up the map is a map again.
    run_frame(
        &mut app,
        &ctx,
        vec![button(from + egui::vec2(160.0, 0.0), false)],
    );
    let resume = app.map_rect().expect("a frame has run").center();
    drag_frames(&mut app, &ctx, resume, resume - egui::vec2(120.0, 0.0));
    assert!(
        (app.map_view().center().lon - camera_before.lon).abs() > 1e-6,
        "the pan gate reopens on release"
    );
}

#[test]
fn a_vertex_drag_snaps_to_another_features_stored_position_unless_ctrl_is_held() {
    let ctx = Context::default();
    let mut app = OxigisApp::new();
    // Two squares whose corners are close but not coincident: dragging one onto
    // the other must land bit-exactly on the stored coordinate, not near it.
    let two = r#"{"type":"FeatureCollection","features":[
        {"type":"Feature","properties":{},
         "geometry":{"type":"Polygon","coordinates":[[
            [-10.0,-10.0],[10.0,-10.0],[10.0,10.0],[-10.0,10.0],[-10.0,-10.0]]]}},
        {"type":"Feature","properties":{},
         "geometry":{"type":"Polygon","coordinates":[[
            [20.0,-10.0],[40.0,-10.0],[40.0,10.0],[20.0,10.0],[20.0,-10.0]]]}}]}"#;
    let id = app
        .add_geojson_layer_from_text("two", two, None)
        .expect("valid GeoJSON");
    // Centred between the two squares, so both corners of interest are on screen.
    app.map_panel = crate::map_view::MapPanelState::new(10.0, 0.0, 3.0, [1024.0, 768.0]);
    run_frame(&mut app, &ctx, Vec::new());
    app.apply_edit_action(EditAction::SetMode(EditMode::Select));
    app.edit.set_selection(Some(EditSelection::feature(0)));
    run_frame(&mut app, &ctx, Vec::new());
    assert!(
        app.edit.snap_settings().enabled,
        "snapping is on by default"
    );

    let from = handle_screen_pos(&app, id, 0, VertexRef::new(1), false);
    let target = handle_screen_pos(&app, id, 1, VertexRef::new(0), false);
    assert!(
        (target - from).length() > 30.0,
        "the two corners must be far enough apart that the drag is a real one"
    );
    // Deliberately a few points short of the target, inside the tolerance.
    drag_frames(&mut app, &ctx, from, target + egui::vec2(4.0, -3.0));

    let moved = ring_of(&app, id, 0);
    assert_eq!(
        (moved[1][0].to_bits(), moved[1][1].to_bits()),
        (20.0_f64.to_bits(), (-10.0_f64).to_bits()),
        "a snapped vertex takes the stored coordinate bit for bit, so repeated \
         snap-and-move cycles accumulate no drift: {:?}",
        moved[1]
    );
    app.undo_once();
    app.edit.set_selection(Some(EditSelection::feature(0)));
    run_frame(&mut app, &ctx, Vec::new());

    // With Ctrl held the pointer wins, and the same gesture lands somewhere else.
    let modifiers = egui::Modifiers {
        ctrl: true,
        command: true,
        ..Default::default()
    };
    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers,
    };
    let raw = |events: Vec<egui::Event>| egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        modifiers,
        events,
        ..Default::default()
    };
    let landing = target + egui::vec2(4.0, -3.0);
    for events in [
        vec![egui::Event::PointerMoved(from), button(from, true)],
        vec![egui::Event::PointerMoved(landing)],
        vec![button(landing, false)],
    ] {
        let _output = ctx.run_ui(raw(events), |ui| app.ui(ui));
    }
    let free = ring_of(&app, id, 0);
    assert_ne!(
        (free[1][0].to_bits(), free[1][1].to_bits()),
        (20.0_f64.to_bits(), (-10.0_f64).to_bits()),
        "Ctrl suspends snapping: {:?}",
        free[1]
    );
    assert!(
        (free[1][0] - 20.0).abs() < 2.0,
        "…but it still landed where the pointer was: {:?}",
        free[1]
    );
}

#[test]
fn the_snap_toggle_reaches_the_settings_and_the_index() {
    let ctx = Context::default();
    let (mut app, _id) = picked(SQUARE, &ctx);
    assert!(app.edit.snap_settings().enabled);
    assert!(
        app.edit.snap_index().segment_count() > 0,
        "the pan gate builds the index from the visible local layers"
    );

    app.apply_edit_action(EditAction::ToggleSnap);
    assert!(!app.edit.snap_settings().enabled);
    assert_eq!(
        app.edit.snap_index().segment_count(),
        0,
        "an index nobody may query is megabytes of held collections for nothing"
    );
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.edit.snap_index().segment_count(), 0);

    app.apply_edit_action(EditAction::ToggleSnap);
    run_frame(&mut app, &ctx, Vec::new());
    assert!(app.edit.snap_settings().enabled);
    assert!(app.edit.snap_index().segment_count() > 0);
}

#[test]
fn a_press_that_misses_every_handle_still_pans() {
    let ctx = Context::default();
    let (mut app, _id) = picked(SQUARE, &ctx);
    let before = app.map_view().center();
    // The square's centre: inside the feature, but nowhere near a handle or a
    // midpoint ghost.
    let rect = app.map_rect().expect("a frame has run");
    drag_frames(
        &mut app,
        &ctx,
        rect.center(),
        rect.center() - egui::vec2(90.0, 0.0),
    );

    assert!(app.edit.drag().is_none(), "no gesture was ever started");
    assert!(
        (app.map_view().center().lon - before.lon).abs() > 1e-6,
        "the gate suppresses a drag on a handle, not every drag: {before:?}"
    );
    assert_eq!(app.undo.depth(), (0, 0));
}

/// One frame carrying a bare key press.
fn key_frame(app: &mut OxigisApp, ctx: &Context, key: egui::Key) {
    run_frame(
        app,
        ctx,
        vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
}

/// A secondary press at `at` and its release one frame later.
fn secondary_click_frames(app: &mut OxigisApp, ctx: &Context, at: egui::Pos2) {
    let button = |pressed| egui::Event::PointerButton {
        pos: at,
        button: egui::PointerButton::Secondary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    run_frame(app, ctx, vec![egui::Event::PointerMoved(at), button(true)]);
    run_frame(app, ctx, vec![button(false)]);
}

/// An empty local layer of `kind`, selected, with the matching drawing tool
/// latched and one frame already run so `map_rect` is known.
fn drawing_into_a_new_layer(kind: StyleKind, ctx: &Context) -> (OxigisApp, LayerId, egui::Rect) {
    let mut app = OxigisApp::new();
    run_frame(&mut app, ctx, Vec::new());
    app.apply_edit_action(EditAction::NewLayer(kind));
    let id = app.selection().expect("the new layer is selected");
    // The fixture's own layer add is an undo step now (v1.2); these tests are
    // about the DRAWING gestures, so their depth numbers count from a clean
    // stack. The add's own contract is pinned in `edit_tests_review`.
    app.undo.reset();
    run_frame(&mut app, ctx, Vec::new());
    let rect = app.map_rect().expect("a frame has run");
    (app, id, rect)
}

#[test]
fn clicking_three_vertices_and_pressing_enter_draws_one_polygon_that_survives_a_reload() {
    let ctx = Context::default();
    let (mut app, id, rect) = drawing_into_a_new_layer(StyleKind::Fill, &ctx);
    assert_eq!(app.edit.mode(), EditMode::DrawPolygon);

    // Well apart, so no click lands inside another's snap radius.
    let corners = [
        rect.center() + egui::vec2(-90.0, -70.0),
        rect.center() + egui::vec2(90.0, -70.0),
        rect.center() + egui::vec2(0.0, 80.0),
    ];
    for (placed, corner) in corners.iter().enumerate() {
        click_frames(&mut app, &ctx, *corner);
        assert_eq!(
            app.edit.sketch().len(),
            placed + 1,
            "each click appends exactly one vertex"
        );
        assert!(
            features_of(&app, id).features.is_empty(),
            "a sketch touches no data until it is finished"
        );
        assert_eq!(app.undo.depth(), (0, 0));
    }

    // Backspace takes one back, and the next click puts it where it belongs.
    key_frame(&mut app, &ctx, egui::Key::Backspace);
    assert_eq!(app.edit.sketch().len(), 2);
    click_frames(&mut app, &ctx, corners[2]);
    assert_eq!(app.edit.sketch().len(), 3);

    key_frame(&mut app, &ctx, egui::Key::Enter);
    assert!(!app.edit.sketch().is_active(), "the sketch is consumed");
    assert_eq!(
        app.edit.mode(),
        EditMode::DrawPolygon,
        "the tool stays latched: drawing several features in a row is the common case"
    );
    assert_eq!(app.undo.depth(), (1, 0));
    assert_eq!(
        app.undo.peek_undo().map(|entry| entry.label),
        Some("Draw polygon")
    );
    assert_eq!(
        app.edit.selection(),
        Some(EditSelection::feature(0)),
        "the new feature is selected"
    );

    let ring = ring_of(&app, id, 0);
    assert_eq!(
        ring.len(),
        4,
        "three clicks store a closed four-position ring"
    );
    assert_eq!(ring.first(), ring.last());

    // One undo takes the whole feature back; one redo brings it back.
    assert!(app.undo_once());
    assert!(features_of(&app, id).features.is_empty());
    assert!(app.redo_once());
    assert_eq!(features_of(&app, id).features.len(), 1);

    // …and it survives save and reload, which is the whole point of drawing
    // into an empty inline layer.
    let json = app
        .project
        .to_json_string()
        .expect("the project serializes");
    let reloaded = Project::from_json_string(&json).expect("and parses back");
    let mut second = OxigisApp::new();
    second.load_project(reloaded);
    assert_eq!(second.status(), Some("Project loaded."));
    let restored = second.local.feature_set(id).expect("the drawn layer loads");
    assert_eq!(restored.features.len(), 1);
    assert_eq!(
        restored.features[0]
            .geometry
            .as_ref()
            .map(oxigeo::geojson::types::Geometry::geometry_type),
        Some(oxigeo::geojson::types::GeometryType::Polygon)
    );
}

#[test]
fn a_point_tool_click_commits_immediately_and_a_secondary_click_finishes_a_line() {
    let ctx = Context::default();
    let (mut app, id, rect) = drawing_into_a_new_layer(StyleKind::Circle, &ctx);
    assert_eq!(app.edit.mode(), EditMode::DrawPoint);

    click_frames(&mut app, &ctx, rect.center() + egui::vec2(-60.0, 0.0));
    assert_eq!(
        features_of(&app, id).features.len(),
        1,
        "a point needs no sketch: one click is the whole gesture"
    );
    assert!(!app.edit.sketch().is_active());
    assert_eq!(app.undo.depth(), (1, 0));
    assert_eq!(
        app.undo.peek_undo().map(|entry| entry.label),
        Some("Draw point")
    );

    // The same layer, drawn into with the line tool: two vertices, finished with
    // a secondary click. Secondary is claimed *only* while a sketch is running.
    app.apply_edit_action(EditAction::SetMode(EditMode::DrawLine));
    secondary_click_frames(&mut app, &ctx, rect.center() + egui::vec2(120.0, 0.0));
    assert_eq!(
        features_of(&app, id).features.len(),
        1,
        "with no sketch in progress the secondary button is left alone"
    );

    click_frames(&mut app, &ctx, rect.center() + egui::vec2(40.0, -60.0));
    click_frames(&mut app, &ctx, rect.center() + egui::vec2(140.0, -60.0));
    assert_eq!(app.edit.sketch().len(), 2);
    secondary_click_frames(&mut app, &ctx, rect.center() + egui::vec2(140.0, -60.0));
    assert_eq!(features_of(&app, id).features.len(), 2);
    assert_eq!(
        app.undo.peek_undo().map(|entry| entry.label),
        Some("Draw line")
    );
    assert!(matches!(
        features_of(&app, id).features[1].geometry,
        Some(Geometry::LineString(_))
    ));
}

#[test]
fn a_polygon_sketch_finished_too_early_is_refused_and_keeps_its_vertices() {
    let ctx = Context::default();
    let (mut app, id, rect) = drawing_into_a_new_layer(StyleKind::Fill, &ctx);
    click_frames(&mut app, &ctx, rect.center() + egui::vec2(-70.0, 0.0));
    click_frames(&mut app, &ctx, rect.center() + egui::vec2(70.0, 0.0));

    key_frame(&mut app, &ctx, egui::Key::Enter);
    assert_eq!(
        app.edit.sketch().len(),
        2,
        "a refused finish costs the user nothing but a status line"
    );
    assert!(features_of(&app, id).features.is_empty());
    assert!(
        app.status().is_some_and(|status| status.contains('3')),
        "the refusal names how many vertices are still needed: {:?}",
        app.status()
    );

    // Escape is the first rung of the ladder: the sketch, and nothing else.
    key_frame(&mut app, &ctx, egui::Key::Escape);
    assert!(!app.edit.sketch().is_active());
    assert_eq!(app.edit.mode(), EditMode::DrawPolygon);
    assert_eq!(app.undo.depth(), (0, 0));
}

#[test]
fn clicking_the_first_vertex_closes_a_ring() {
    let ctx = Context::default();
    let (mut app, id, rect) = drawing_into_a_new_layer(StyleKind::Fill, &ctx);
    let first = rect.center() + egui::vec2(-90.0, -70.0);
    for corner in [
        first,
        rect.center() + egui::vec2(90.0, -70.0),
        rect.center() + egui::vec2(0.0, 80.0),
    ] {
        click_frames(&mut app, &ctx, corner);
    }
    assert_eq!(app.edit.sketch().len(), 3);

    // A fourth click, back on vertex zero: the ring closes rather than gaining a
    // fourth vertex on top of its first.
    click_frames(&mut app, &ctx, first + egui::vec2(2.0, 2.0));
    assert!(!app.edit.sketch().is_active());
    assert_eq!(features_of(&app, id).features.len(), 1);
    assert_eq!(ring_of(&app, id, 0).len(), 4);
}

#[test]
fn applying_the_attribute_form_replaces_the_properties_and_the_table_picks_up_the_new_key() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    let ctx = Context::default();
    app.apply_edit_action(EditAction::ToggleWindow);
    // The first frame adopts the layer as the edit target, which clears any
    // feature selection; picking one is only meaningful after it.
    run_frame(&mut app, &ctx, Vec::new());
    app.edit.set_selection(Some(EditSelection::feature(0)));
    run_frame(&mut app, &ctx, Vec::new());

    // The window bound the form to the picked feature and seeded it from the
    // live properties, without marking anything dirty.
    assert_eq!(app.edit.form().bound(), Some((id, 0)));
    assert_eq!(app.edit.form().rows().len(), 1);
    assert!(!app.edit.form().is_dirty());
    let columns_before = app.table_panel.column_count();
    assert_eq!(columns_before, SYNTHETIC_COLUMN_COUNT + 1);

    app.edit
        .form_mut()
        .add_key("population", FieldKind::Integer, 1)
        .expect("a fresh key is accepted");
    let row = app.edit.form().rows().len() - 1;
    assert!(app.edit.form_mut().set_row_text(row, "13960000"));
    assert!(app.edit.form().is_dirty());

    assert!(app.apply_attribute_form());
    assert_eq!(app.undo.depth(), (1, 0));
    assert_eq!(
        app.undo.peek_undo().map(|entry| entry.label),
        Some("Edit attributes")
    );
    let properties = features_of(&app, id).features[0]
        .properties
        .clone()
        .expect("the feature keeps a property map");
    assert_eq!(properties["name"], serde_json::json!("Tokyo"));
    assert!(
        properties["population"].is_i64(),
        "an Integer draft must not be stored as a float: {:?}",
        properties["population"]
    );
    assert!(!app.edit.form().is_dirty());

    // The table rebinds against the new `Arc` on the next frame and derives a
    // schema that holds the new key.
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.table_panel.column_count(), columns_before + 1);

    // Undo puts the old map back, and the schema shrinks again.
    assert!(app.undo_once());
    let properties = features_of(&app, id).features[0]
        .properties
        .clone()
        .expect("the feature keeps a property map");
    assert!(!properties.contains_key("population"));
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.table_panel.column_count(), columns_before);
}

#[test]
fn two_attribute_applies_to_the_same_feature_fold_into_one_undo_entry() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    let ctx = Context::default();
    app.apply_edit_action(EditAction::ToggleWindow);
    // The first frame adopts the layer as the edit target, which clears any
    // feature selection; picking one is only meaningful after it.
    run_frame(&mut app, &ctx, Vec::new());
    app.edit.set_selection(Some(EditSelection::feature(0)));
    run_frame(&mut app, &ctx, Vec::new());

    assert!(app.edit.form_mut().set_row_text(0, "Edo"));
    assert!(app.apply_attribute_form());
    assert_eq!(app.undo.depth(), (1, 0));

    // The window is still open and the selection has not moved, so the second
    // Apply folds into the first: one Ctrl+Z undoes *the attribute edit*.
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(
        app.edit.form().rows()[0].text,
        "Edo",
        "the form re-seeds from the freshly stored feature"
    );
    assert!(app.edit.form_mut().set_row_text(0, "Yedo"));
    assert!(app.apply_attribute_form());
    assert_eq!(
        app.undo.depth(),
        (1, 0),
        "successive Applies fold until the window closes or the selection moves"
    );

    assert!(app.undo_once());
    let properties = features_of(&app, id).features[0]
        .properties
        .clone()
        .expect("the feature keeps a property map");
    assert_eq!(
        properties["name"],
        serde_json::json!("Tokyo"),
        "the folded entry restores the state before the first Apply"
    );

    // Closing the window ends the coalescing window, so the next Apply is its
    // own undo step.
    assert!(app.redo_once());
    app.apply_edit_action(EditAction::ToggleWindow);
    assert!(!app.edit.show_window());
    run_frame(&mut app, &ctx, Vec::new());
    assert!(app.edit.form_mut().set_row_text(0, "Tokio"));
    assert!(app.apply_attribute_form());
    assert_eq!(app.undo.depth(), (2, 0));
}

#[test]
fn a_dirty_form_survives_a_selection_change_and_a_discard_re_seeds_it() {
    let mut app = OxigisApp::new();
    let _id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    let ctx = Context::default();
    app.apply_edit_action(EditAction::ToggleWindow);
    // The first frame adopts the layer as the edit target, which clears any
    // feature selection; picking one is only meaningful after it.
    run_frame(&mut app, &ctx, Vec::new());
    app.edit.set_selection(Some(EditSelection::feature(0)));
    run_frame(&mut app, &ctx, Vec::new());
    assert!(app.edit.form_mut().set_row_text(0, "Edo"));

    // Selecting another feature must not throw typed data away.
    app.edit.set_selection(Some(EditSelection::feature(1)));
    run_frame(&mut app, &ctx, Vec::new());
    assert!(app.edit.form().is_dirty());
    assert_eq!(app.edit.form().rows()[0].text, "Edo");
    assert_eq!(
        app.edit.form().bound().map(|(_, feature)| feature),
        Some(0),
        "the banner has to be able to name the feature the rows belong to"
    );

    app.edit.form_mut().discard();
    run_frame(&mut app, &ctx, Vec::new());
    assert!(!app.edit.form().is_dirty());
    assert_eq!(app.edit.form().bound().map(|(_, feature)| feature), Some(1));
    assert_eq!(app.edit.form().rows()[0].text, "Osaka");
}

#[test]
fn the_toolbar_labels_name_the_step_the_buttons_would_move_in_either_family() {
    let mut app = OxigisApp::new();
    assert_eq!(
        app.toolbar_state().undo_label,
        None,
        "an empty stack has nothing to name"
    );
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(1.0, 2.0))));
    assert_eq!(app.toolbar_state().undo_label, Some("Move vertex"));

    // A PROJECT step on top. `can_undo` counts both families, so a label read
    // from the feature-only view would leave the enabled button promising
    // "Nothing to undo" about the layer add it is about to reverse.
    app.apply_edit_action(EditAction::NewLayer(StyleKind::Circle));
    let state = app.toolbar_state();
    assert!(state.can_undo, "the layer add is undoable");
    assert_eq!(state.undo_label, Some("Add layer"));
    assert_eq!(state.redo_label, None);

    assert!(app.undo_once());
    let state = app.toolbar_state();
    assert_eq!(state.undo_label, Some("Move vertex"), "one step back");
    assert!(state.can_redo);
    assert_eq!(state.redo_label, Some("Add layer"));
}

#[test]
fn the_inline_size_warning_is_stated_on_the_crossing_not_on_every_commit() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    // One fat property rather than a megabyte of geometry: the stored text
    // only has to WEIGH more than the threshold, and twenty thousand features
    // would make this the slowest test in the suite.
    let mut properties = serde_json::Map::new();
    properties.insert(
        "blob".to_string(),
        serde_json::json!("a".repeat(crate::local_input::INLINE_GEOJSON_WARN_BYTES)),
    );
    let fat = Feature::new(
        Some(Geometry::Point(
            Point::new(vec![5.0, 5.0]).expect("two elements"),
        )),
        Some(properties),
    );
    let index = features_of(&app, id).features.len();
    assert!(app.commit_edit(EditTransaction::single(
        id,
        "Add point",
        FeatureOp::Add {
            index,
            feature: Box::new(fat),
        },
    )));
    assert!(
        app.status()
            .is_some_and(|line| line.contains("inside the project file")),
        "the commit that crosses the threshold must say so: {:?}",
        app.status()
    );
    let notices = app.edit.notices().len();

    // Every later commit on the — still oversized — layer stays quiet: the
    // gesture owns the status line, and one sentence per vertex drag forever
    // is noise rather than information.
    app.set_status("cleared");
    assert!(app.commit_edit(move_first_vertex(&app, id, LonLat::new(1.0, 2.0))));
    assert_eq!(
        app.status(),
        Some("cleared"),
        "the size warning must not stomp every later commit's status line"
    );
    assert_eq!(
        app.edit.notices().len(),
        notices,
        "nor grow the notice log once per commit"
    );
}
