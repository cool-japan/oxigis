// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Gesture-level tests for marks across the three ARITY-changing gestures
//! (editing v1.4 item 2): "Delete vertices" and its undo/redo, "Insert
//! vertex" and its undo, and the hydrate seam that replaces a collection the
//! marks were addressing.
//!
//! Separate from `app/edit_tests_review.rs`, which holds the equal-arity
//! marks family and is close to the 2000-line limit; the two small fixtures
//! are therefore duplicated rather than shared, as that file's own header
//! records.

use super::OxigisApp;
use crate::edit::VertexRef;
use crate::edit::command::{self, EditTransaction, FeatureOp};
use crate::edit::selection::FeatureSelection;
use crate::edit::{EditMode, EditSelection};
use oxigeo::geojson::types::Geometry;
use oxigis_core::LayerId;
use oxigis_render::LonLat;

/// A hexagon: six open corners, so deleting two still leaves a legal ring
/// (`remove_vertex` refuses a ring that would be left with two).
const HEX: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"plot"},
     "geometry":{"type":"Polygon","coordinates":[[
        [-10.0,0.0],[-5.0,-9.0],[5.0,-9.0],[10.0,0.0],[5.0,9.0],[-5.0,9.0],
        [-10.0,0.0]]]}}]}"#;

/// The same hexagon shifted east — what a re-read of the layer's file finds.
const HEX_MOVED: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"plot"},
     "geometry":{"type":"Polygon","coordinates":[[
        [90.0,0.0],[95.0,-9.0],[105.0,-9.0],[110.0,0.0],[105.0,9.0],[95.0,9.0],
        [90.0,0.0]]]}}]}"#;

/// The hexagon layer with feature 0 selected and `marks` marked — the state a
/// released marquee leaves.
fn hex_with_marks(marks: Vec<VertexRef>) -> (OxigisApp, LayerId) {
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("plot", HEX, None);
    let id = app.selection().expect("the added layer is selected");
    app.edit.set_mode(EditMode::Select);
    let _notice = app.edit.retarget(Some(id));
    app.edit
        .set_multi_selection(Some(FeatureSelection::single(0).with_vertex_set(marks)));
    (app, id)
}

/// The anchor's currently marked vertex set.
fn marked_set(app: &OxigisApp) -> Vec<VertexRef> {
    app.edit
        .multi_selection()
        .map(|multi| multi.vertex_set().to_vec())
        .unwrap_or_default()
}

/// The open-ring positions of feature 0, `(lon, lat)` pairs.
fn ring_positions(app: &OxigisApp, id: LayerId) -> Vec<(f64, f64)> {
    let features = app.local.feature_set(id).expect("the layer is loaded");
    let Some(Geometry::Polygon(polygon)) = features.features[0].geometry.as_ref() else {
        panic!("still a polygon");
    };
    command::open_ring(&polygon.coordinates[0])
        .iter()
        .map(|position| (position[0], position[1]))
        .collect()
}

#[test]
fn undoing_a_set_delete_restores_the_vertices_marked() {
    let (mut app, id) = hex_with_marks(vec![VertexRef::new(1), VertexRef::new(2)]);
    let before = ring_positions(&app, id);
    assert!(app.delete_selected_vertex(), "two of six corners go");
    assert_eq!(ring_positions(&app, id).len(), 4);
    assert_eq!(
        marked_set(&app),
        Vec::<VertexRef>::new(),
        "the deleted vertices cannot stay marked — they do not exist"
    );

    assert!(app.undo_once());
    assert_eq!(ring_positions(&app, id), before, "coordinates bit-exact");
    assert_eq!(
        marked_set(&app),
        vec![VertexRef::new(1), VertexRef::new(2)],
        "the amber rings are the only affordance saying this set is live \
         again, so the undo restores them with the geometry"
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.ends_with("2 vertices still marked.")),
        "and the status says how many: {:?}",
        app.status,
    );
}

#[test]
fn a_redo_of_a_set_delete_marks_nothing() {
    let (mut app, id) = hex_with_marks(vec![VertexRef::new(1), VertexRef::new(2)]);
    assert!(app.delete_selected_vertex());
    assert!(app.undo_once());

    assert!(app.redo_once());
    assert_eq!(ring_positions(&app, id).len(), 4);
    assert_eq!(
        marked_set(&app),
        Vec::<VertexRef>::new(),
        "a redo deletes them again: there is nothing left to mark"
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| !status.contains("still marked")),
        "and no marks clause is claimed: {:?}",
        app.status,
    );
}

#[test]
fn undoing_an_insert_marks_nothing() {
    let (mut app, id) = hex_with_marks(Vec::new());
    app.edit.set_selection(Some(EditSelection::feature(0)));
    let features = app.local.feature_set(id).expect("loaded").clone();
    let before = features.features[0].clone();
    let mut after = before.clone();
    command::insert_vertex(&mut after, VertexRef::new(0), LonLat::new(-8.0, -5.0))
        .expect("a ring accepts an inserted vertex");
    assert!(app.commit_edit(EditTransaction {
        layer: id,
        label: "Insert vertex",
        ops: vec![FeatureOp::Replace {
            index: 0,
            before: Box::new(before),
            after: Box::new(after),
        }],
        selection_before: Some(EditSelection::feature(0)),
        selection_after: Some(EditSelection::feature(0)),
        coalesce: None,
    }));
    assert_eq!(ring_positions(&app, id).len(), 7);

    assert!(app.undo_once());
    assert_eq!(ring_positions(&app, id).len(), 6);
    assert_eq!(
        marked_set(&app),
        Vec::<VertexRef>::new(),
        "undoing an insert is a pure deletion: marking the surviving \
         neighbours would arm Delete against vertices the user never touched"
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| !status.contains("still marked")),
        "and no marks clause is claimed: {:?}",
        app.status,
    );
}

#[test]
fn a_hydrate_clears_the_marked_vertex_set() {
    let (mut app, id) = hex_with_marks(vec![VertexRef::new(1), VertexRef::new(2)]);
    assert!(app.delete_selected_vertex(), "there is a step to prune");
    assert_eq!(app.undo.depth(), (1, 0));

    assert!(app.hydrate_geojson_layer_from_bytes(id, "plot", HEX_MOVED.as_bytes()));
    assert_eq!(
        marked_set(&app),
        Vec::<VertexRef>::new(),
        "the marks addressed the collection the re-read just replaced"
    );
    assert_eq!(
        ring_positions(&app, id).first().map(|corner| corner.0),
        Some(90.0),
        "and the re-read data is what is on screen"
    );
    // The stack was pruned with the collection, so there is nothing left to
    // splice the old geometry back in.
    assert_eq!(app.undo.depth(), (0, 0));
    assert!(!app.redo_once());
    assert_eq!(app.status.as_deref(), Some("Nothing to redo."));
}
