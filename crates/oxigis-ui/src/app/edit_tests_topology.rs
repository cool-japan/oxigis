// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! App-level tests for the validation wiring: the per-commit revalidation, the
//! **Validate layer** button, and the lifecycle rules that keep a layer's
//! notices from outliving it.
//!
//! Separate from `app/edit_tests.rs` for the same reason that file is separate
//! from `app/tests.rs`: the 2000-line rule.

use super::OxigisApp;
use crate::edit::command::{EditTransaction, FeatureOp, set_vertex};
use crate::edit::toolbar::{EditAction, issue_badge};
use crate::edit::topology::{CrossingKind, RingRole, TopologyIssue};
use crate::edit::{EditSelection, VertexRef};
use crate::layer_panel::LayerAction;
use egui::Context;
use oxigis_core::LayerId;
use oxigis_render::LonLat;
use std::sync::Arc;

/// Two disjoint, counter-clockwise, perfectly clean squares.
const SQUARES: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"west"},
     "geometry":{"type":"Polygon","coordinates":[[[0,0],[2,0],[2,2],[0,2],[0,0]]]}},
    {"type":"Feature","properties":{"name":"east"},
     "geometry":{"type":"Polygon","coordinates":[[[10,0],[12,0],[12,2],[10,2],[10,0]]]}}]}"#;

/// The same two squares, with the second one already tied into a bowtie.
const ONE_BOWTIE: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"west"},
     "geometry":{"type":"Polygon","coordinates":[[[0,0],[2,0],[2,2],[0,2],[0,0]]]}},
    {"type":"Feature","properties":{"name":"east"},
     "geometry":{"type":"Polygon","coordinates":[[[10,0],[12,2],[12,0],[10,2],[10,0]]]}}]}"#;

/// The transaction that swaps two of feature `index`'s exterior vertices,
/// turning a square into a bowtie in one commit.
fn tie_a_bowtie(app: &OxigisApp, id: LayerId, index: usize) -> EditTransaction {
    let features = app.local.feature_set(id).expect("the layer is loaded");
    let before = features.features[index].clone();
    let mut after = before.clone();
    let corner = |vertex: usize| -> LonLat {
        let ring = match &before.geometry {
            Some(oxigeo::geojson::types::Geometry::Polygon(polygon)) => &polygon.coordinates[0],
            other => panic!("expected a polygon, got {other:?}"),
        };
        LonLat::new(ring[vertex][0], ring[vertex][1])
    };
    let (first, second) = (corner(1), corner(2));
    set_vertex(&mut after, VertexRef::at(0, 0, 1), second).expect("vertex 1 exists");
    set_vertex(&mut after, VertexRef::at(0, 0, 2), first).expect("vertex 2 exists");
    EditTransaction {
        layer: id,
        label: "Move vertex",
        ops: vec![FeatureOp::Replace {
            index,
            before: Box::new(before),
            after: Box::new(after),
        }],
        selection_before: None,
        selection_after: Some(EditSelection::feature(index)),
        coalesce: None,
    }
}

#[test]
fn an_edit_that_self_intersects_a_ring_notices_it_for_the_touched_feature_only() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("blocks", SQUARES, None)
        .expect("valid GeoJSON");
    app.selection = Some(id);
    // Nothing has been edited, so nothing has been validated: validation never
    // runs on load.
    assert!(app.edit.issues(id).is_empty());

    assert!(app.commit_edit(tie_a_bowtie(&app, id, 0)));
    let issues = app.edit.issues(id);
    assert_eq!(issues.len(), 1, "{issues:?}");
    assert_eq!(issues[0].feature, 0);
    assert_eq!(issues[0].ring, 0);
    assert_eq!(issues[0].role, RingRole::Exterior);
    assert!(matches!(
        issues[0].issue,
        TopologyIssue::SelfIntersection {
            kind: CrossingKind::Proper,
            ..
        }
    ));
    assert!(
        issues[0].at.is_some(),
        "a notice row has to be able to move the camera"
    );
    assert_eq!(app.edit.issue_count(id), 1);
    // The untouched feature was never walked, so it never grew a notice.
    assert!(issues.iter().all(|issue| issue.feature != 1));

    // And undo, which goes through exactly the same choke point, takes the
    // notice back with the geometry.
    assert!(app.undo_once());
    assert!(app.edit.issues(id).is_empty(), "{:?}", app.edit.issues(id));
}

#[test]
fn a_commit_replaces_only_the_touched_features_notices() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("blocks", SQUARES, None)
        .expect("valid GeoJSON");
    app.selection = Some(id);
    assert!(app.commit_edit(tie_a_bowtie(&app, id, 0)));
    assert!(app.commit_edit(tie_a_bowtie(&app, id, 1)));
    assert_eq!(app.edit.issue_count(id), 2);
    assert!(app.edit.issues(id).iter().any(|issue| issue.feature == 0));
    assert!(app.edit.issues(id).iter().any(|issue| issue.feature == 1));

    // Fixing feature 0 drops its notice and leaves feature 1's alone.
    assert!(app.commit_edit(tie_a_bowtie(&app, id, 0)));
    let issues = app.edit.issues(id);
    assert_eq!(issues.len(), 1, "{issues:?}");
    assert_eq!(issues[0].feature, 1);
}

#[test]
fn validate_layer_over_a_clean_layer_reports_nothing() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("blocks", SQUARES, None)
        .expect("valid GeoJSON");
    app.selection = Some(id);
    app.apply_edit_action(EditAction::ValidateLayer);
    assert!(app.edit.issues(id).is_empty(), "{:?}", app.edit.issues(id));
    assert_eq!(app.status(), Some("Validation: no issues."));
}

#[test]
fn validate_layer_walks_the_whole_collection_and_replaces_what_it_finds() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("blocks", ONE_BOWTIE, None)
        .expect("valid GeoJSON");
    app.selection = Some(id);
    // Loading is silent; only the button walks a collection nobody edited.
    assert!(app.edit.issues(id).is_empty());

    app.apply_edit_action(EditAction::ValidateLayer);
    let issues = app.edit.issues(id);
    assert_eq!(issues.len(), 1, "{issues:?}");
    assert_eq!(issues[0].feature, 1);
    assert_eq!(app.status(), Some("Validation: 1 issue."));

    // A full run is authoritative: it has seen every feature, so a second run
    // over repaired data replaces rather than accumulates.
    assert!(app.commit_edit(tie_a_bowtie(&app, id, 1)));
    app.apply_edit_action(EditAction::ValidateLayer);
    assert!(app.edit.issues(id).is_empty(), "{:?}", app.edit.issues(id));
    assert_eq!(app.status(), Some("Validation: no issues."));
}

#[test]
fn validate_layer_refuses_without_a_loaded_layer_and_says_why() {
    let mut app = OxigisApp::new();
    app.apply_edit_action(EditAction::ValidateLayer);
    assert_eq!(app.status(), Some("Select a layer before validating it."));

    // A raster layer has no feature collection to walk at all.
    app.apply_layer_action(LayerAction::AddDemoXyzLayer);
    app.apply_edit_action(EditAction::ValidateLayer);
    assert!(
        app.status()
            .is_some_and(|line| line.contains("features are not loaded")),
        "{:?}",
        app.status()
    );
}

#[test]
fn a_deleted_feature_takes_its_notices_with_it() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("blocks", SQUARES, None)
        .expect("valid GeoJSON");
    app.selection = Some(id);
    assert!(app.commit_edit(tie_a_bowtie(&app, id, 1)));
    assert_eq!(app.edit.issue_count(id), 1);

    // Removing feature 1 leaves one feature, so an index-1 notice addresses
    // nothing and must not survive.
    let features = Arc::clone(app.local.feature_set(id).expect("loaded"));
    let removed = features.features[1].clone();
    assert!(app.commit_edit(EditTransaction::single(
        id,
        "Delete feature",
        FeatureOp::Remove {
            index: 1,
            feature: Box::new(removed),
        },
    )));
    assert_eq!(app.edit.issue_count(id), 0);
}

#[test]
fn removing_a_layer_drops_its_validation_notices() {
    let mut app = OxigisApp::new();
    let kept = app
        .add_geojson_layer_from_text("kept", SQUARES, None)
        .expect("valid GeoJSON");
    let doomed = app
        .add_geojson_layer_from_text("doomed", SQUARES, None)
        .expect("valid GeoJSON");
    app.selection = Some(kept);
    assert!(app.commit_edit(tie_a_bowtie(&app, kept, 0)));
    app.selection = Some(doomed);
    assert!(app.commit_edit(tie_a_bowtie(&app, doomed, 0)));
    assert_eq!(app.edit.issue_count(kept), 1);
    assert_eq!(app.edit.issue_count(doomed), 1);

    app.apply_layer_action(LayerAction::Remove(doomed));
    assert_eq!(app.edit.issue_count(doomed), 0);
    assert_eq!(
        app.edit.issue_count(kept),
        1,
        "notices are keyed by layer, so removing one layer must not touch another"
    );
}

#[test]
fn notices_are_kept_per_layer_across_a_retarget_and_dropped_on_a_new_project() {
    let mut app = OxigisApp::new();
    let first = app
        .add_geojson_layer_from_text("first", SQUARES, None)
        .expect("valid GeoJSON");
    let second = app
        .add_geojson_layer_from_text("second", SQUARES, None)
        .expect("valid GeoJSON");
    app.selection = Some(first);
    assert!(app.commit_edit(tie_a_bowtie(&app, first, 0)));
    // Looking at another layer must not throw away work the user asked for.
    app.apply_layer_action(LayerAction::Select(second));
    app.sync_edit_state();
    assert_eq!(app.edit.issue_count(first), 1);
    assert_eq!(app.edit.issue_count(second), 0);

    app.new_project();
    assert_eq!(app.edit.issue_count(first), 0);
}

/// Draws just the Validation section into a real `egui` frame.
fn draw_validation(app: &mut OxigisApp, ctx: &Context) {
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let _output = ctx.run_ui(raw_input, |ui| app.validation_ui(ui));
}

#[test]
fn the_validation_section_draws_its_empty_states_and_its_rows() {
    let mut app = OxigisApp::new();
    let ctx = Context::default();
    // No layer at all.
    draw_validation(&mut app, &ctx);

    let id = app
        .add_geojson_layer_from_text("blocks", SQUARES, None)
        .expect("valid GeoJSON");
    app.selection = Some(id);
    // A clean layer: the buttons and the "nothing recorded" line.
    draw_validation(&mut app, &ctx);

    // One issue of each severity, so both colour branches are painted, plus a
    // row long enough that the truncation matters.
    assert!(app.commit_edit(tie_a_bowtie(&app, id, 0)));
    let mut issues = app.edit.issues(id).to_vec();
    issues.push(crate::edit::topology::FeatureIssue {
        feature: 1,
        part: 0,
        ring: 0,
        role: RingRole::Exterior,
        issue: TopologyIssue::WrongWinding {
            role: RingRole::Exterior,
        },
        at: Some(LonLat::new(11.0, 1.0)),
    });
    app.edit.set_issues(id, issues);
    draw_validation(&mut app, &ctx);
    assert_eq!(app.edit.issue_count(id), 2);
}

#[test]
fn the_toolbar_badge_is_capped_so_the_row_cannot_grow() {
    assert_eq!(issue_badge(0), "0");
    assert_eq!(issue_badge(7), "7");
    assert_eq!(issue_badge(99), "99");
    assert_eq!(issue_badge(100), "99+");
    assert_eq!(issue_badge(usize::MAX), "99+");
}
