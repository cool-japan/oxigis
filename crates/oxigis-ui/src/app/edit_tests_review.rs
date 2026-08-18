// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the Phase 2 adversarial-review fixes: index
//! remapping after a renumbering, the Escape focus guard, vertex picking by
//! click, sketch preservation, the `⚠` badge, the form's column-cap fallback,
//! and empty Processing results.
//!
//! Separate from `app/edit_tests.rs`, which is close to the 2000-line limit;
//! a few small fixtures are therefore duplicated rather than shared.

use super::OxigisApp;
use crate::edit::command::{EditTransaction, FeatureOp, remap_index};
use crate::edit::hit;
use crate::edit::toolbar::EditAction;
use crate::edit::{EditMode, EditSelection, Handles, VertexRef};
use egui::Context;
use oxigeo::geojson::types::{FeatureCollection, Geometry};
use oxigis_core::LayerId;
use oxigis_render::LonLat;
use std::sync::Arc;

/// Three named point features — enough to watch an index shift.
const NAMED: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"a"},
     "geometry":{"type":"Point","coordinates":[0.0,0.0]}},
    {"type":"Feature","properties":{"name":"b"},
     "geometry":{"type":"Point","coordinates":[1.0,1.0]}},
    {"type":"Feature","properties":{"name":"c"},
     "geometry":{"type":"Point","coordinates":[2.0,2.0]}}]}"#;

/// Two clean points, then a self-intersecting bowtie at index 2 — the layer a
/// validation run records an issue against.
const WITH_BOWTIE: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"a"},
     "geometry":{"type":"Point","coordinates":[0.0,0.0]}},
    {"type":"Feature","properties":{"name":"b"},
     "geometry":{"type":"Point","coordinates":[1.0,1.0]}},
    {"type":"Feature","properties":{"name":"bowtie"},
     "geometry":{"type":"Polygon","coordinates":[[
        [0.0,0.0],[2.0,2.0],[2.0,0.0],[0.0,2.0],[0.0,0.0]]]}}]}"#;

/// One big square — the vertex-handle fixture `app/edit_tests.rs` also uses.
const SQUARE: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"plot"},
     "geometry":{"type":"Polygon","coordinates":[[
        [-10.0,-10.0],[10.0,-10.0],[10.0,10.0],[-10.0,10.0],[-10.0,-10.0]]]}}]}"#;

/// The layer's features, as currently held by the app.
fn features_of(app: &OxigisApp, id: LayerId) -> Arc<FeatureCollection> {
    Arc::clone(app.local.feature_set(id).expect("the layer is loaded"))
}

/// Drives one whole `ui` frame at a realistic window size.
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

/// A camera that frames the fixtures with room to spare — see
/// `app/edit_tests.rs`'s twin for why the zoom-to-extent default is not used.
fn frame_the_fixture(app: &mut OxigisApp) {
    app.map_panel = crate::map_view::MapPanelState::new(0.0, 0.0, 3.0, [1024.0, 768.0]);
}

/// A primary press at `at` and its release one frame later — a click.
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

/// One bare `Escape` key-down.
fn escape() -> Vec<egui::Event> {
    vec![egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]
}

/// The `Remove` transaction deleting feature `index` of `id`.
fn remove_feature(app: &OxigisApp, id: LayerId, index: usize) -> EditTransaction {
    let feature = features_of(app, id).features[index].clone();
    EditTransaction::single(
        id,
        "Delete feature",
        FeatureOp::Remove {
            index,
            feature: Box::new(feature),
        },
    )
}

/// Where `at` lands on screen, as of the last frame.
fn screen_pos_of(app: &OxigisApp, at: LonLat) -> egui::Pos2 {
    let rect = app.map_rect().expect("a frame has run");
    let px = app.map_view().lon_lat_to_screen(at);
    rect.min + egui::vec2(px[0], px[1])
}

/// Where a vertex handle of feature `index` is drawn.
fn handle_screen_pos(app: &OxigisApp, id: LayerId, index: usize, at: VertexRef) -> egui::Pos2 {
    let rect = app.map_rect().expect("a frame has run");
    let features = features_of(app, id);
    let geometry = features.features[index]
        .geometry
        .as_ref()
        .expect("the feature has geometry");
    let position =
        hit::handle_position(geometry, at, false, app.map_view(), 1.0).expect("the handle exists");
    hit::to_screen(app.map_view(), rect.min, 1.0, position)
}

#[test]
fn remap_index_follows_adds_and_removes_in_application_order() {
    let features = match oxigeo::geojson::reader::feature_collection_from_str(NAMED) {
        Ok(features) => features,
        Err(error) => panic!("valid fixture: {error}"),
    };
    let filler = Box::new(features.features[0].clone());
    let add_at = |index| FeatureOp::Add {
        index,
        feature: filler.clone(),
    };
    let remove_at = |index| FeatureOp::Remove {
        index,
        feature: filler.clone(),
    };
    let replace_at = |index| FeatureOp::Replace {
        index,
        before: filler.clone(),
        after: filler.clone(),
    };

    // An add at or below the index shifts it up; above leaves it alone.
    assert_eq!(remap_index(&[add_at(0)], 4), Some(5));
    assert_eq!(remap_index(&[add_at(4)], 4), Some(5));
    assert_eq!(remap_index(&[add_at(5)], 4), Some(4));
    // A remove below shifts down; of the index itself, the feature is gone.
    assert_eq!(remap_index(&[remove_at(1)], 4), Some(3));
    assert_eq!(remap_index(&[remove_at(4)], 4), None);
    assert_eq!(remap_index(&[remove_at(6)], 4), Some(4));
    // A replace moves nothing.
    assert_eq!(remap_index(&[replace_at(4)], 4), Some(4));
    // Ops compose in application order, the same order `apply_ops` uses.
    assert_eq!(remap_index(&[remove_at(0), add_at(0)], 4), Some(4));
    assert_eq!(remap_index(&[add_at(0), remove_at(5)], 4), None);
}

#[test]
fn a_delete_renumbers_recorded_issues_and_undo_renumbers_them_back() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("mixed", WITH_BOWTIE, None)
        .expect("valid GeoJSON");
    assert!(app.validate_active_layer());
    assert!(
        !app.edit.issues(id).is_empty()
            && app.edit.issues(id).iter().all(|issue| issue.feature == 2),
        "the fixture's one broken feature is the bowtie at index 2: {:?}",
        app.edit.issues(id)
    );

    // Deleting feature 0 shifts the bowtie to index 1; its recorded issues —
    // and their fly-to coordinates — must follow, not stay glued to index 2
    // (which now names nothing, or worse, a clean neighbour).
    assert!(app.commit_edit(remove_feature(&app, id, 0)));
    assert!(
        !app.edit.issues(id).is_empty()
            && app.edit.issues(id).iter().all(|issue| issue.feature == 1),
        "the issues must follow the feature down to index 1: {:?}",
        app.edit.issues(id)
    );

    // Undo puts feature 0 back and the issues return to index 2.
    assert!(app.undo_once());
    assert!(
        app.edit.issues(id).iter().all(|issue| issue.feature == 2),
        "undo must renumber the issues back up: {:?}",
        app.edit.issues(id)
    );
}

#[test]
fn a_dirty_form_follows_a_renumbering_and_applies_to_the_same_feature() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("named", NAMED, None)
        .expect("valid GeoJSON");
    let features = features_of(&app, id);
    app.edit.form_mut().sync(Some((id, 1)), Some(&features));
    assert!(app.edit.form_mut().set_row_text(0, "renamed"));
    assert!(app.edit.form().is_dirty());

    // Deleting feature 0 renumbers "b" from 1 to 0 while the buffer is dirty.
    // The stale binding would build a `Replace` against whatever now sits at
    // index 1 — destroying "c"'s properties with text typed for "b".
    assert!(app.commit_edit(remove_feature(&app, id, 0)));
    assert_eq!(
        app.edit.form().bound(),
        Some((id, 0)),
        "the binding must follow the feature it was seeded from"
    );

    assert!(app.apply_attribute_form());
    let after = features_of(&app, id);
    assert_eq!(
        after.features[0].get_property("name"),
        Some(&serde_json::json!("renamed")),
        "the typed value lands on the feature it was typed for"
    );
    assert_eq!(
        after.features[1].get_property("name"),
        Some(&serde_json::json!("c")),
        "the neighbour that slid into the old index is untouched"
    );
}

#[test]
fn deleting_the_bound_feature_discards_the_dirty_form_and_says_so() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("named", NAMED, None)
        .expect("valid GeoJSON");
    let features = features_of(&app, id);
    app.edit.form_mut().sync(Some((id, 0)), Some(&features));
    assert!(app.edit.form_mut().set_row_text(0, "doomed"));

    assert!(app.commit_edit(remove_feature(&app, id, 0)));
    assert_eq!(
        app.edit.form().bound(),
        None,
        "there is no feature the rows could be offered back to"
    );
    assert!(!app.edit.form().is_dirty());
    assert!(
        app.edit
            .notices()
            .iter()
            .any(|notice| notice.message().contains("discarded")),
        "silently dropping typed data is never forgiven: {:?}",
        app.edit.notices()
    );
}

#[test]
fn an_undo_for_another_layer_does_not_stamp_the_selection_onto_this_one() {
    let mut app = OxigisApp::new();
    let first = app
        .add_geojson_layer_from_text("first", NAMED, None)
        .expect("valid GeoJSON");
    let second = app
        .add_geojson_layer_from_text("second", NAMED, None)
        .expect("valid GeoJSON");

    // An edit on `first` whose transaction lands the selection on feature 2.
    app.selection = Some(first);
    app.sync_edit_state();
    let feature = features_of(&app, first).features[2].clone();
    let mut moved = feature.clone();
    crate::edit::command::set_vertex(&mut moved, VertexRef::new(0), LonLat::new(5.0, 5.0))
        .expect("a point has vertex 0");
    assert!(app.commit_edit(EditTransaction {
        layer: first,
        label: "Move vertex",
        ops: vec![FeatureOp::Replace {
            index: 2,
            before: Box::new(feature),
            after: Box::new(moved),
        }],
        selection_before: None,
        selection_after: Some(EditSelection::vertex(2, VertexRef::new(0))),
        coalesce: None,
    }));

    // Switch to `second` and pick its feature 0. The undo stack is global by
    // design, so Ctrl+Z now replays a `first` transaction — whose landing
    // index must not be stamped onto `second`'s selection, where it would arm
    // Delete against a feature the user never touched.
    app.selection = Some(second);
    app.sync_edit_state();
    app.edit.set_selection(Some(EditSelection::feature(0)));
    assert!(app.undo_once());
    assert_eq!(
        app.edit.selection(),
        Some(EditSelection::feature(0)),
        "the selection belongs to the selected layer, not to the undone one"
    );
    // The data change itself did land on `first`.
    let restored = features_of(&app, first);
    let Some(Geometry::Point(point)) = restored.features[2].geometry.as_ref() else {
        panic!("feature 2 of `first` is a point");
    };
    assert_eq!(point.coordinates, vec![2.0, 2.0]);
}

#[test]
fn an_escape_spent_leaving_a_text_field_does_not_climb_the_cancel_ladder() {
    let ctx = Context::default();
    let mut app = OxigisApp::new();
    let _id = app
        .add_geojson_layer_from_text("named", NAMED, None)
        .expect("valid GeoJSON");
    run_frame(&mut app, &ctx, Vec::new());
    app.apply_edit_action(EditAction::SetMode(EditMode::Select));
    app.edit.set_selection(Some(EditSelection::feature(0)));

    // A widget owns the keyboard this frame — the attribute form's TextEdit in
    // real use.
    ctx.memory_mut(|memory| memory.request_focus(egui::Id::new("a text field")));
    run_frame(&mut app, &ctx, Vec::new());

    // The field's Escape: egui clears the focused widget in `begin_pass`,
    // before any app code runs, so on this frame `memory.focused()` is
    // already gone — only the one-frame memory knows the key was spent
    // leaving the field.
    run_frame(&mut app, &ctx, escape());
    assert_eq!(
        app.edit.selection(),
        Some(EditSelection::feature(0)),
        "the Escape was spent leaving the field, not on the selection"
    );
    assert_eq!(app.edit.mode(), EditMode::Select);

    // With no field involved, the next Escape belongs to the ladder again.
    run_frame(&mut app, &ctx, escape());
    assert_eq!(app.edit.selection(), None, "the ladder still works");
}

#[test]
fn a_click_on_a_vertex_handle_picks_the_vertex_without_a_drag_or_an_undo_entry() {
    let ctx = Context::default();
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("plot", SQUARE, None)
        .expect("valid GeoJSON");
    frame_the_fixture(&mut app);
    run_frame(&mut app, &ctx, Vec::new());
    app.apply_edit_action(EditAction::SetMode(EditMode::Select));
    app.edit.set_selection(Some(EditSelection::feature(0)));
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.edit.handles(), Handles::Active);

    // A plain click — no motion, no long press — on handle 1. Before the fix
    // this fell through to the feature pick (which knows nothing about
    // vertices), making the status line's own instruction, "Click a vertex
    // handle to pick it, then Delete removes it", unfollowable by clicking.
    let at = handle_screen_pos(&app, id, 0, VertexRef::new(1));
    click_frames(&mut app, &ctx, at);
    assert_eq!(
        app.edit.selection(),
        Some(EditSelection::vertex(0, VertexRef::new(1))),
        "the click picks the vertex, not the feature stack"
    );
    assert_eq!(
        app.undo.depth(),
        (0, 0),
        "a pick is not an edit — no spurious `Move vertex` entry"
    );
    assert!(
        app.status().is_some_and(|line| line.contains("picked")),
        "{:?}",
        app.status()
    );
}

#[test]
fn a_fresh_pick_shows_its_handles_on_the_very_frame_of_the_click() {
    let ctx = Context::default();
    let mut app = OxigisApp::new();
    let _id = app
        .add_geojson_layer_from_text("plot", SQUARE, None)
        .expect("valid GeoJSON");
    frame_the_fixture(&mut app);
    run_frame(&mut app, &ctx, Vec::new());
    app.apply_edit_action(EditAction::SetMode(EditMode::Select));
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.edit.handles(), Handles::None, "nothing picked yet");

    // Click inside the square. The pan gate planned handles *before* the
    // click resolved, so without the overlay-time re-plan the newly picked
    // feature would paint this frame with no handles to grab — a whole missed
    // beat in a reactive event loop that renders nothing until the next input.
    let centre = screen_pos_of(&app, LonLat::new(0.0, 0.0));
    click_frames(&mut app, &ctx, centre);
    assert_eq!(app.edit.selection(), Some(EditSelection::feature(0)));
    assert_eq!(
        app.edit.handles(),
        Handles::Active,
        "the handles are live on the click frame itself, not one frame later"
    );
}

#[test]
fn a_sketch_with_nowhere_to_commit_survives_every_finish_gesture() {
    let ctx = Context::default();
    let mut app = OxigisApp::new(); // no layer at all
    run_frame(&mut app, &ctx, Vec::new());
    app.apply_edit_action(EditAction::SetMode(EditMode::DrawPolygon));

    // A click with no target layer appends nothing — the refusal comes at the
    // first click, not after four of them.
    click_frames(&mut app, &ctx, egui::pos2(400.0, 300.0));
    assert_eq!(app.edit.sketch().len(), 0);
    assert!(
        app.status().is_some_and(|line| line.contains("layer")),
        "the refusal says what is missing: {:?}",
        app.status()
    );

    // And a sketch that somehow exists (here: built directly) is not consumed
    // by a finish that then finds nowhere to commit — `Sketch::finish` eats
    // the vertices on success, so the target check must come first.
    for position in [
        LonLat::new(0.0, 0.0),
        LonLat::new(4.0, 0.0),
        LonLat::new(4.0, 4.0),
    ] {
        app.edit
            .sketch_mut()
            .append(EditMode::DrawPolygon, position);
    }
    assert!(!app.finish_sketch());
    assert_eq!(
        app.edit.sketch().len(),
        3,
        "digitized vertices must never be destroyed by a refusal"
    );
}

#[test]
fn the_warning_badge_only_ever_opens_the_window_and_reveals_validation() {
    let mut app = OxigisApp::new();
    // The failing case before the fix: the window is *already open*, and the
    // badge's `ToggleWindow` closed it instead of showing the list.
    app.set_edit_window_open(true);
    app.apply_edit_action(EditAction::ShowValidation);
    assert!(
        app.edit.show_window(),
        "the badge must never close the window"
    );
    assert!(
        app.edit.take_reveal_validation(),
        "the Validation section is asked to expand"
    );
    assert!(
        !app.edit.take_reveal_validation(),
        "the reveal is one-shot — afterwards the header is the user's again"
    );

    // From closed, it opens.
    app.set_edit_window_open(false);
    app.apply_edit_action(EditAction::ShowValidation);
    assert!(app.edit.show_window());
}

#[test]
fn the_form_column_cap_is_measured_against_the_layer_even_with_the_table_hidden() {
    let ctx = Context::default();
    let mut app = OxigisApp::new();
    let _id = app
        .add_geojson_layer_from_text("named", NAMED, None)
        .expect("valid GeoJSON");
    // Hide the table: `AttributeTablePanel::bind` — the only thing that
    // refreshes its schema — never runs, so before the fix the cap was
    // measured against 0 and per-feature additions could push the layer past
    // the point where the table silently stops showing new keys.
    app.show_table = false;
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(
        app.form_schema_len(),
        1,
        "the fallback derives the layer's real distinct-key count (`name`)"
    );
}

#[test]
fn export_pdf_queues_a_snapshot_of_the_visible_local_layers_take_once() {
    let mut app = OxigisApp::new();
    let visible = app
        .add_geojson_layer_from_text("shown", NAMED, None)
        .expect("valid GeoJSON");
    let hidden = app
        .add_geojson_layer_from_text("hidden", NAMED, None)
        .expect("valid GeoJSON");
    app.project
        .layers
        .toggle_visibility(hidden)
        .expect("the layer exists");

    app.request_print();
    let request = app.take_pending_print().expect("a queued export");
    assert_eq!(request.title, app.project().name);
    assert_eq!(
        request.layers.len(),
        1,
        "only the visible local layer is captured"
    );
    assert!(Arc::ptr_eq(
        &request.layers[0].features,
        &features_of(&app, visible)
    ));
    let live = app.map_view();
    assert_eq!(request.view.center().lon, live.center().lon);
    assert_eq!(request.view.zoom(), live.zoom());
    assert!(
        app.take_pending_print().is_none(),
        "the seam is take-once, like every other shell seam"
    );
}

#[test]
fn an_empty_tool_result_is_reported_as_empty_not_as_a_failed_drop() {
    let mut app = OxigisApp::new();
    let layers_before = app.project.layers.len();
    let descriptor = oxigis_core::builtin_registry()
        .get("centroid")
        .expect("centroid is built in")
        .clone();

    // A successful run with zero output features — e.g. Centroids over an
    // attribute-only layer whose every feature has `geometry: null`.
    app.route_processing_result(
        &descriptor,
        Ok(serde_json::json!({"type": "FeatureCollection", "features": []})),
    );
    assert_eq!(
        app.project.layers.len(),
        layers_before,
        "no layer is created from nothing"
    );
    assert!(
        app.status()
            .is_some_and(|line| line.contains("no layer was created")),
        "the status must describe an empty result, not a failed drop: {:?}",
        app.status()
    );
}

#[test]
fn a_multi_delete_removes_the_whole_set_and_one_undo_restores_it() {
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("named", NAMED, None);
    let id = app.selection().expect("the added layer is selected");
    let before = features_of(&app, id);
    app.edit.set_mode(EditMode::Select);
    let _ = app.edit.retarget(Some(id));
    app.edit.set_multi_selection(Some(
        crate::edit::selection::FeatureSelection::single(0)
            .toggled(2)
            .expect("two features stay selected"),
    ));
    assert!(app.delete_selected_feature(), "the multi-delete must land");
    let after = features_of(&app, id);
    assert_eq!(after.features.len(), 1, "features 0 and 2 both went");
    assert_eq!(
        after.features[0]
            .properties
            .as_ref()
            .and_then(|properties| properties.get("name")),
        Some(&serde_json::Value::from("b")),
        "only the unselected middle feature survives"
    );
    assert!(app.undo_once(), "ONE undo restores the whole set");
    let restored = features_of(&app, id);
    assert_eq!(*restored, *before, "byte-identical restoration");
}

#[test]
fn touched_after_lands_adds_and_ignores_removes() {
    fn tiny() -> Box<oxigeo::geojson::types::Feature> {
        Box::new(oxigeo::geojson::types::Feature::new(None, None))
    }
    let layer = LayerId::new();
    let pure_delete = EditTransaction {
        layer,
        label: "delete",
        ops: vec![
            FeatureOp::Remove {
                index: 2,
                feature: tiny(),
            },
            FeatureOp::Remove {
                index: 0,
                feature: tiny(),
            },
        ],
        selection_before: None,
        selection_after: None,
        coalesce: None,
    };
    assert!(
        pure_delete.touched_after().is_empty(),
        "a delete of any size revalidates nothing"
    );
    let add_then_remove_below = EditTransaction {
        layer,
        label: "mixed",
        ops: vec![
            FeatureOp::Add {
                index: 3,
                feature: tiny(),
            },
            FeatureOp::Remove {
                index: 0,
                feature: tiny(),
            },
        ],
        selection_before: None,
        selection_after: None,
        coalesce: None,
    };
    assert_eq!(
        add_then_remove_below.touched_after(),
        vec![2],
        "the added feature's landing slot shifts down past the removal"
    );
}

/// A primary press-and-release with Shift held — an additive click.
fn shift_click_frames(app: &mut OxigisApp, ctx: &Context, at: egui::Pos2) {
    let button = |pressed| egui::Event::PointerButton {
        pos: at,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::SHIFT,
    };
    let raw = |events| egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        modifiers: egui::Modifiers::SHIFT,
        events,
        ..Default::default()
    };
    let _ = ctx.run_ui(
        raw(vec![egui::Event::PointerMoved(at), button(true)]),
        |ui| app.ui(ui),
    );
    let _ = ctx.run_ui(raw(vec![button(false)]), |ui| app.ui(ui));
}

#[test]
fn shift_clicks_toggle_the_set_and_a_miss_keeps_it() {
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("named", NAMED, None);
    let id = app.selection().expect("the added layer is selected");
    frame_the_fixture(&mut app);
    app.edit.set_mode(EditMode::Select);
    let _ = app.edit.retarget(Some(id));
    let ctx = Context::default();
    run_frame(&mut app, &ctx, Vec::new());

    // Plain click selects feature 0 alone.
    let at_a = screen_pos_of(&app, LonLat::new(0.0, 0.0));
    click_frames(&mut app, &ctx, at_a);
    assert_eq!(app.edit.selection(), Some(EditSelection::feature(0)));

    // Shift+click on feature 2 adds it and anchors there.
    let at_c = screen_pos_of(&app, LonLat::new(2.0, 2.0));
    shift_click_frames(&mut app, &ctx, at_c);
    let multi = app.edit.multi_selection().expect("a set exists").clone();
    assert_eq!(multi.features(), &[0, 2]);
    assert_eq!(multi.anchor(), 2);

    // A missed Shift+click leaves the set alone.
    let at_miss = screen_pos_of(&app, LonLat::new(20.0, -20.0));
    shift_click_frames(&mut app, &ctx, at_miss);
    let kept = app.edit.multi_selection().expect("still selected").clone();
    assert_eq!(kept.features(), &[0, 2], "a miss must not clear the set");

    // Shift+click on feature 2 again removes it; feature 0 remains.
    shift_click_frames(&mut app, &ctx, at_c);
    let multi = app.edit.multi_selection().expect("one member left").clone();
    assert_eq!(multi.features(), &[0]);
}

#[test]
fn copy_and_paste_flow_through_egui_clipboard_events() {
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("named", NAMED, None);
    let id = app.selection().expect("the added layer is selected");
    frame_the_fixture(&mut app);
    app.edit.set_mode(EditMode::Select);
    let _ = app.edit.retarget(Some(id));
    app.edit.set_multi_selection(Some(
        crate::edit::selection::FeatureSelection::single(0)
            .toggled(2)
            .expect("two features selected"),
    ));
    let ctx = Context::default();

    // Copy: the event is consumed by the edit system and reported.
    run_frame(&mut app, &ctx, vec![egui::Event::Copy]);
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.contains("2 features copied")),
        "got {:?}",
        app.status
    );

    // Paste a whole FeatureCollection: one transaction, all selected.
    run_frame(&mut app, &ctx, vec![egui::Event::Paste(NAMED.to_string())]);
    let features = features_of(&app, id);
    assert_eq!(features.features.len(), 6, "three pasted onto three");
    let multi = app.edit.multi_selection().expect("the pasted run").clone();
    assert_eq!(multi.features(), &[3, 4, 5]);
    assert_eq!(multi.anchor(), 3);
    assert!(app.undo_once(), "one undo removes the whole paste");
    assert_eq!(features_of(&app, id).features.len(), 3);
}

#[test]
fn a_paste_past_the_selection_cap_lands_whole_and_says_how_much_is_selected() {
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("named", NAMED, None);
    let id = app.selection().expect("the added layer is selected");
    frame_the_fixture(&mut app);
    app.edit.set_mode(EditMode::Select);
    let _ = app.edit.retarget(Some(id));
    let ctx = Context::default();

    // One past the cap, so the run is truncated by exactly one feature.
    let count = crate::edit::selection::MAX_MULTI_SELECT + 1;
    let mut text = String::from(r#"{"type":"FeatureCollection","features":["#);
    for index in 0..count {
        if index > 0 {
            text.push(',');
        }
        text.push_str(
            r#"{"type":"Feature","properties":null,"geometry":{"type":"Point","coordinates":[0.0,0.0]}}"#,
        );
    }
    text.push_str("]}");

    run_frame(&mut app, &ctx, vec![egui::Event::Paste(text)]);
    assert_eq!(
        features_of(&app, id).features.len(),
        3 + count,
        "every pasted feature lands — the cap is on the SELECTION, not the paste"
    );
    let multi = app.edit.multi_selection().expect("the pasted run").clone();
    assert_eq!(multi.len(), crate::edit::selection::MAX_MULTI_SELECT);
    assert_eq!(multi.anchor(), 3, "anchored on the head of the run");
    assert!(
        app.edit
            .notices()
            .iter()
            .any(|notice| notice.message().contains("are selected")),
        "a truncated run must be reported, not silently trimmed: {:?}",
        app.edit
            .notices()
            .iter()
            .map(crate::edit::EditNotice::message)
            .collect::<Vec<_>>()
    );
}

#[test]
fn clipboard_events_are_left_alone_in_browse_mode() {
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("named", NAMED, None);
    let id = app.selection().expect("selected");
    frame_the_fixture(&mut app);
    // EditMode::Off: the paste event must NOT create features.
    assert_eq!(app.edit.mode(), EditMode::Off);
    let ctx = Context::default();
    run_frame(&mut app, &ctx, vec![egui::Event::Paste(NAMED.to_string())]);
    assert_eq!(
        features_of(&app, id).features.len(),
        3,
        "Browse mode leaves the clipboard to egui"
    );
}

#[test]
fn a_marked_vertex_set_deletes_as_one_replace_and_refuses_degeneration() {
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("plot", SQUARE, None);
    let id = app.selection().expect("the added layer is selected");
    app.edit.set_mode(EditMode::Select);
    let _ = app.edit.retarget(Some(id));
    let ring_len = |app: &OxigisApp| match features_of(app, id).features[0].geometry.as_ref() {
        Some(Geometry::Polygon(polygon)) => polygon.coordinates[0].len(),
        other => panic!("expected the square, got {other:?}"),
    };
    assert_eq!(ring_len(&app), 5, "closed square: 4 open + closing dup");

    // Two of four marked: deleting would leave a 2-vertex ring — refused
    // whole, nothing changed.
    let marked_two = crate::edit::selection::FeatureSelection::single(0)
        .with_vertex_set(vec![VertexRef::new(0), VertexRef::new(1)]);
    app.edit.set_multi_selection(Some(marked_two));
    assert!(!app.delete_selected_vertex(), "degeneration must refuse");
    assert_eq!(ring_len(&app), 5, "a refusal changes nothing");

    // One marked: deletes as one Replace; one undo restores it.
    let marked_one = crate::edit::selection::FeatureSelection::single(0)
        .with_vertex_set(vec![VertexRef::new(1)]);
    app.edit.set_multi_selection(Some(marked_one));
    assert!(app.delete_selected_vertex(), "a legal set deletes");
    assert_eq!(ring_len(&app), 4, "3 open + closing dup");
    assert!(app.undo_once());
    assert_eq!(ring_len(&app), 5);
}

#[test]
fn a_shift_drag_marquee_marks_the_handles_it_encloses() {
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("plot", SQUARE, None);
    let id = app.selection().expect("selected");
    frame_the_fixture(&mut app);
    app.edit.set_mode(EditMode::Select);
    let _ = app.edit.retarget(Some(id));
    app.edit.set_selection(Some(EditSelection::feature(0)));
    let ctx = Context::default();
    run_frame(&mut app, &ctx, Vec::new());
    assert!(app.edit.handles().is_active(), "handles are drawn");

    // A box around the square's south-west corner vertex only.
    let corner = handle_screen_pos(&app, id, 0, VertexRef::new(0));
    let start = corner - egui::vec2(30.0, -30.0);
    let end = corner + egui::vec2(30.0, -30.0);
    let button = |pressed, pos| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::SHIFT,
    };
    let raw = |events| egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        modifiers: egui::Modifiers::SHIFT,
        events,
        ..Default::default()
    };
    let _ = ctx.run_ui(
        raw(vec![egui::Event::PointerMoved(start), button(true, start)]),
        |ui| app.ui(ui),
    );
    let _ = ctx.run_ui(raw(vec![egui::Event::PointerMoved(end)]), |ui| app.ui(ui));
    assert!(
        app.edit.marquee().is_some(),
        "the Shift+drag must open a marquee, not pan"
    );
    let _ = ctx.run_ui(raw(vec![button(false, end)]), |ui| app.ui(ui));
    let _ = ctx.run_ui(raw(Vec::new()), |ui| app.ui(ui));
    assert!(app.edit.marquee().is_none(), "released");
    let multi = app.edit.multi_selection().expect("still selected").clone();
    assert_eq!(
        multi.vertex_set(),
        &[VertexRef::new(0)],
        "exactly the enclosed corner is marked"
    );
}

/// Marks the SQUARE's two southern corners and runs one real drag on the
/// grabbed handle: press, move by `delta_px`, release, settle.
fn marked_square_drag(
    app: &mut OxigisApp,
    ctx: &Context,
    id: LayerId,
    grabbed: VertexRef,
    delta_px: egui::Vec2,
) {
    let multi = crate::edit::selection::FeatureSelection::single(0)
        .with_vertex_set(vec![VertexRef::new(0), VertexRef::new(1)]);
    app.edit.set_multi_selection(Some(multi));
    run_frame(app, ctx, Vec::new());
    let start = handle_screen_pos(app, id, 0, grabbed);
    let end = start + delta_px;
    let button = |pressed, pos| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    run_frame(
        app,
        ctx,
        vec![egui::Event::PointerMoved(start), button(true, start)],
    );
    run_frame(app, ctx, vec![egui::Event::PointerMoved(end)]);
    run_frame(app, ctx, vec![button(false, end)]);
    run_frame(app, ctx, Vec::new());
}

/// The square fixture, framed, in Select mode with handles live.
fn square_ready(ctx: &Context) -> (OxigisApp, LayerId) {
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("plot", SQUARE, None);
    let id = app.selection().expect("selected");
    frame_the_fixture(&mut app);
    app.edit.set_mode(EditMode::Select);
    let _ = app.edit.retarget(Some(id));
    app.edit.set_selection(Some(EditSelection::feature(0)));
    run_frame(&mut app, ctx, Vec::new());
    assert!(app.edit.handles().is_active(), "handles are drawn");
    (app, id)
}

/// The open-ring positions of the square, `(lon, lat)` pairs.
fn ring_positions(app: &OxigisApp, id: LayerId) -> Vec<(f64, f64)> {
    let features = features_of(app, id);
    let Some(Geometry::Polygon(polygon)) = features.features[0].geometry.as_ref() else {
        panic!("still a polygon");
    };
    polygon.coordinates[0]
        .iter()
        .map(|position| (position[0], position[1]))
        .collect()
}

#[test]
fn a_drag_on_a_marked_handle_moves_the_whole_set_as_one_replace_and_keeps_the_marks() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let before = ring_positions(&app, id);
    marked_square_drag(
        &mut app,
        &ctx,
        id,
        VertexRef::new(0),
        egui::vec2(30.0, -20.0),
    );

    assert_eq!(app.undo.depth(), (1, 0), "one gesture, one entry");
    assert_eq!(
        app.undo.peek_undo().map(|entry| entry.label),
        Some(crate::edit::MOVE_VERTICES_LABEL),
    );
    let after = ring_positions(&app, id);
    // Both marked corners moved by the SAME world delta; the others did not.
    let delta_of = |index: usize| {
        let from = LonLat::new(before[index].0, before[index].1).to_world();
        let to = LonLat::new(after[index].0, after[index].1).to_world();
        (to.x - from.x, to.y - from.y)
    };
    let (dx0, dy0) = delta_of(0);
    let (dx1, dy1) = delta_of(1);
    assert!(
        dx0.abs() > 0.0 && dy0.abs() > 0.0,
        "the grabbed corner moved"
    );
    assert!(
        (dx0 - dx1).abs() < 1e-9 && (dy0 - dy1).abs() < 1e-9,
        "the set moved rigidly: {dx0},{dy0} vs {dx1},{dy1}",
    );
    assert_eq!(after[2], before[2], "unmarked corners stay put");
    assert_eq!(after[3], before[3]);
    assert_eq!(after.first(), after.last(), "the ring stays closed");
    // The marks survive the commit, so the set can be nudged again.
    let multi = app.edit.multi_selection().expect("still selected");
    assert_eq!(
        multi.vertex_set(),
        &[VertexRef::new(0), VertexRef::new(1)],
        "the marks survive the move",
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|line| line.contains("2 vertices moved")),
        "{:?}",
        app.status,
    );
    // One Ctrl+Z restores every coordinate exactly.
    assert!(app.undo_once());
    assert_eq!(ring_positions(&app, id), before);
}

#[test]
fn escape_mid_set_drag_restores_the_geometry_and_keeps_the_marks() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let before = ring_positions(&app, id);
    let multi = crate::edit::selection::FeatureSelection::single(0)
        .with_vertex_set(vec![VertexRef::new(0), VertexRef::new(1)]);
    app.edit.set_multi_selection(Some(multi));
    run_frame(&mut app, &ctx, Vec::new());
    let start = handle_screen_pos(&app, id, 0, VertexRef::new(0));
    let end = start + egui::vec2(40.0, -25.0);
    let button = |pressed, pos| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(start), button(true, start)],
    );
    run_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(end)]);
    assert!(app.edit.drag().is_some(), "the set drag is live");
    run_frame(&mut app, &ctx, escape());
    assert!(app.edit.drag().is_none(), "Escape cancels command-free");
    run_frame(&mut app, &ctx, vec![button(false, end)]);
    run_frame(&mut app, &ctx, Vec::new());

    assert_eq!(app.undo.depth(), (0, 0), "nothing was committed");
    assert_eq!(
        ring_positions(&app, id),
        before,
        "the geometry snapped back"
    );
    let multi = app.edit.multi_selection().expect("still selected");
    assert_eq!(
        multi.vertex_set(),
        &[VertexRef::new(0), VertexRef::new(1)],
        "the marks live in the selection, not the drag",
    );
}

#[test]
fn dragging_an_unmarked_handle_stays_a_single_vertex_move() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let before = ring_positions(&app, id);
    // Marks on the southern corners; the drag grabs the NE corner (index 2).
    marked_square_drag(
        &mut app,
        &ctx,
        id,
        VertexRef::new(2),
        egui::vec2(25.0, 25.0),
    );

    assert_eq!(app.undo.depth(), (1, 0));
    assert_eq!(
        app.undo.peek_undo().map(|entry| entry.label),
        Some("Move vertex"),
        "an unmarked handle is the plain v1.1 gesture",
    );
    let after = ring_positions(&app, id);
    assert_eq!(after[0], before[0], "the marked corners did not move");
    assert_eq!(after[1], before[1]);
    assert_ne!(after[2], before[2], "only the grabbed corner moved");
    // A plain single move commits with `selection_after = vertex(..)`, which
    // collapses the multi-selection — the marks do not survive it.
    let vertex_set = app
        .edit
        .multi_selection()
        .map(|multi| multi.vertex_set().to_vec())
        .unwrap_or_default();
    assert!(vertex_set.is_empty(), "the marks die with the collapse");
}

#[test]
fn an_opacity_drag_folds_into_one_undo_step_and_a_second_drag_starts_another() {
    use crate::layer_panel::LayerAction;

    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("named", NAMED, None);
    let id = app.selection().expect("selected");
    let ctx = Context::default();
    // One "drag": three slider frames while the button is down.
    for value in [0.9_f32, 0.7, 0.5] {
        app.apply_layer_action(LayerAction::SetOpacity(id, value));
    }
    assert_eq!(app.undo.depth(), (1, 0), "one drag = one undo step");
    // A frame with no button down closes the window (the clockless
    // boundary); the next drag is a new step.
    run_frame(&mut app, &ctx, Vec::new());
    app.apply_layer_action(LayerAction::SetOpacity(id, 0.2));
    assert_eq!(app.undo.depth(), (2, 0), "a second drag is a second step");
    // Undo restores the value the FIRST drag ended on, then the original.
    assert!(app.undo_once());
    let opacity = app.project().layers.get(id).map(|layer| layer.opacity());
    assert_eq!(opacity, Some(0.5));
    assert!(app.undo_once());
    let opacity = app.project().layers.get(id).map(|layer| layer.opacity());
    assert_eq!(opacity, Some(1.0));
}

#[test]
fn a_reorder_undoes_to_the_exact_previous_order() {
    use crate::layer_panel::LayerAction;

    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("a", NAMED, None);
    let a = app.selection().expect("selected");
    app.add_geojson_layer_from_text("b", NAMED, None);
    let b = app.selection().expect("selected");
    let order = |app: &OxigisApp| -> Vec<_> {
        app.project()
            .layers
            .layers()
            .iter()
            .map(|layer| layer.id)
            .collect()
    };
    assert_eq!(order(&app), vec![a, b]);
    app.apply_layer_action(LayerAction::MoveUp(a));
    assert_eq!(order(&app), vec![b, a], "a moved up over b");
    assert!(app.undo_once());
    assert_eq!(order(&app), vec![a, b], "the previous order is back");
    assert!(app.redo_once());
    assert_eq!(order(&app), vec![b, a]);
}

#[test]
fn adding_a_layer_is_undoable_and_redo_restores_slot_style_and_the_exact_arc() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_bytes("named", NAMED.as_bytes(), None)
        .expect("valid GeoJSON");
    assert_eq!(app.undo.depth(), (1, 0), "the drop gesture records the add");
    assert_eq!(
        app.undo.peek_undo_entry().map(|entry| entry.label()),
        Some("Add layer"),
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|line| line.contains("Ctrl+Z removes it.")),
        "the status says the add is undoable: {:?}",
        app.status,
    );
    let arc_before = features_of(&app, id);
    let style_before = app.project.styles.get(&id).cloned();
    assert!(style_before.is_some(), "a local add always has a style");

    assert!(app.undo_once());
    assert!(
        app.project.layers.get(id).is_none(),
        "undo removes the added layer",
    );
    assert_eq!(app.undo.depth(), (0, 1));

    assert!(app.redo_once());
    let layer = app.project.layers.get(id).expect("redo restores the layer");
    assert_eq!(layer.name, "named");
    assert_eq!(
        app.project
            .layers
            .layers()
            .iter()
            .position(|entry| entry.id == id),
        Some(0),
        "the storage slot is restored",
    );
    assert_eq!(app.project.styles.get(&id).cloned(), style_before);
    assert!(
        Arc::ptr_eq(&features_of(&app, id), &arc_before),
        "redo restores the EXACT features Arc",
    );
    assert!(
        app.local.default_style(id).is_some(),
        "the remembered default style survives the round trip",
    );
}

#[test]
fn a_multi_table_gpkg_drop_is_one_undo_step_that_removes_every_layer() {
    let mut app = OxigisApp::new();
    let ids = app.add_gpkg_layer_from_bytes("basic.gpkg", crate::gpkg_input::fixture::BASIC, None);
    assert_eq!(ids.len(), 3, "the fixture holds three tables");
    assert_eq!(app.undo.depth(), (1, 0), "one gesture, ONE undo step");
    assert_eq!(
        app.undo.peek_undo_entry().map(|entry| entry.label()),
        Some("Add layers"),
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|line| line.contains("One Ctrl+Z removes them all.")),
        "{:?}",
        app.status,
    );
    let positions_before: Vec<Option<usize>> = ids
        .iter()
        .map(|id| {
            app.project
                .layers
                .layers()
                .iter()
                .position(|entry| entry.id == *id)
        })
        .collect();

    assert!(app.undo_once());
    assert!(
        ids.iter().all(|id| app.project.layers.get(*id).is_none()),
        "one Ctrl+Z removes the whole drop",
    );
    assert!(app.redo_once());
    let positions_after: Vec<Option<usize>> = ids
        .iter()
        .map(|id| {
            app.project
                .layers
                .layers()
                .iter()
                .position(|entry| entry.id == *id)
        })
        .collect();
    assert_eq!(
        positions_after, positions_before,
        "redo restores every table at its exact slot",
    );
}

#[test]
fn undoing_an_add_keeps_every_other_layers_history() {
    let mut app = OxigisApp::new();
    let first = app
        .add_geojson_layer_from_bytes("first", NAMED.as_bytes(), None)
        .expect("valid GeoJSON");
    assert!(app.commit_edit(remove_feature(&app, first, 0)));
    let second = app
        .add_geojson_layer_from_bytes("second", NAMED.as_bytes(), None)
        .expect("valid GeoJSON");
    assert_eq!(app.undo.depth(), (3, 0), "add, edit, add");

    // Undo the second add: the first layer's edit history is untouched.
    assert!(app.undo_once());
    assert!(app.project.layers.get(second).is_none());
    assert_eq!(app.undo.depth(), (2, 1));
    assert!(app.undo_once(), "the first layer's edit still undoes");
    assert_eq!(features_of(&app, first).features.len(), 3);
    assert!(app.undo_once(), "and its add still undoes");
    assert!(app.project.layers.get(first).is_none());
}

#[test]
fn a_project_load_records_no_add_entries() {
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_bytes("named", NAMED.as_bytes(), None)
        .expect("valid GeoJSON");
    app.sync_project_view();
    let json = app.project().to_json_string().expect("serialize");
    let loaded = oxigis_core::Project::from_json_string(&json).expect("parse");

    let mut fresh = OxigisApp::new();
    fresh.load_project(loaded);
    assert_eq!(
        fresh.undo.depth(),
        (0, 0),
        "a project load is not an undoable edit",
    );
}

#[test]
fn an_add_too_large_for_the_budget_is_added_without_being_recorded() {
    use crate::edit::stack::EditStack;

    let mut app = OxigisApp::new();
    // A budget so small any snapshot dominates it.
    app.undo = EditStack::with_budget(128, 64);
    let id = app
        .add_geojson_layer_from_bytes("named", NAMED.as_bytes(), None)
        .expect("the add itself must not be blocked");
    assert!(app.project.layers.get(id).is_some());
    assert_eq!(app.undo.depth(), (0, 0), "nothing was recorded");
    assert!(
        app.status
            .as_deref()
            .is_some_and(|line| line.contains("too large to be undoable")),
        "the guard says so: {:?}",
        app.status,
    );
}

#[test]
fn a_new_edit_layer_is_removed_again_by_one_undo() {
    let ctx = Context::default();
    let mut app = OxigisApp::new();
    run_frame(&mut app, &ctx, Vec::new());
    app.apply_edit_action(EditAction::NewLayer(crate::style_panel::StyleKind::Fill));
    let id = app.selection().expect("the new layer is selected");
    assert_eq!(
        app.undo.peek_undo_entry().map(|entry| entry.label()),
        Some("Add layer"),
    );
    assert!(app.undo_once());
    assert!(
        app.project.layers.get(id).is_none(),
        "Ctrl+Z takes the accidental layer back",
    );
}

#[test]
fn a_byte_budget_eviction_is_reported_and_cap_ageing_stays_silent() {
    use crate::edit::stack::EditStack;

    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("named", NAMED, None)
        .expect("valid GeoJSON");
    // A budget with room for roughly one delete, so the second one must trade
    // the first away to stay undoable.
    let one = remove_feature(&app, id, 0).estimated_bytes();
    app.undo = EditStack::with_budget(1_000, one + one / 2);
    assert!(app.commit_edit(remove_feature(&app, id, 0)));
    assert!(
        !app.edit
            .notices()
            .iter()
            .any(|notice| notice.message().starts_with("Memory:")),
        "the first delete fits the budget and reports nothing",
    );
    assert!(app.commit_edit(remove_feature(&app, id, 0)));
    assert!(
        app.status.as_deref().is_some_and(|s| s.contains("Memory:")),
        "the eviction lands on the status line: {:?}",
        app.status,
    );
    assert!(
        app.edit
            .notices()
            .iter()
            .any(|notice| notice.message().starts_with("Memory:")),
        "the eviction survives in the notice log",
    );

    // Entry-cap ageing stays silent: a roomy byte budget, a 2-entry cap.
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("named", NAMED, None)
        .expect("valid GeoJSON");
    app.undo = EditStack::with_budget(2, 1 << 30);
    for _ in 0..3 {
        assert!(app.commit_edit(remove_feature(&app, id, 0)));
    }
    assert!(
        !app.edit
            .notices()
            .iter()
            .any(|notice| notice.message().starts_with("Memory:")),
        "routine cap ageing must not nag",
    );
}

// --- Editing v1.3 E1: a data change mid-gesture cancels the live gesture ---

/// One Ctrl+Z key-down (`Modifiers::COMMAND` is Ctrl off macOS).
fn undo_key() -> egui::Event {
    egui::Event::Key {
        key: egui::Key::Z,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::COMMAND,
    }
}

/// A primary-button event at `pos`.
fn primary(pressed: bool, pos: egui::Pos2) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

/// Feature 0 with vertex 2 nudged — a programmatic single-`Replace` commit.
fn nudge_transaction(app: &OxigisApp, id: LayerId) -> EditTransaction {
    let feature = features_of(app, id).features[0].clone();
    let mut moved = feature.clone();
    crate::edit::command::set_vertex(&mut moved, VertexRef::new(2), LonLat::new(12.0, 12.0))
        .expect("the square has vertex 2");
    EditTransaction {
        layer: id,
        label: "Nudge",
        ops: vec![FeatureOp::Replace {
            index: 0,
            before: Box::new(feature),
            after: Box::new(moved),
        }],
        // Both directions keep feature 0 selected, so the handles stay drawn
        // across an undo — the marquee test re-opens a box-select after one.
        selection_before: Some(EditSelection::feature(0)),
        selection_after: Some(EditSelection::feature(0)),
        coalesce: None,
    }
}

#[test]
fn ctrl_z_mid_set_drag_cancels_the_gesture_and_cannot_resurrect_the_undone_coordinates() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let before = ring_positions(&app, id);
    // One committed set move — the entry Ctrl+Z will undo, and (because the
    // marks survive the commit) the second grab below is a SET drag: the arm
    // that used to silently commit `origin + delta` over the undone state.
    marked_square_drag(
        &mut app,
        &ctx,
        id,
        VertexRef::new(0),
        egui::vec2(30.0, -20.0),
    );
    assert_eq!(app.undo.depth(), (1, 0));
    let moved = ring_positions(&app, id);
    assert_ne!(moved, before);

    let start = handle_screen_pos(&app, id, 0, VertexRef::new(0));
    let pull = start + egui::vec2(25.0, 15.0);
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(start), primary(true, start)],
    );
    run_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(pull)]);
    assert!(app.edit.drag().is_some(), "the second drag is live");

    // Ctrl+Z lands mid-gesture: cancel first, then undo, both said out loud.
    run_frame(&mut app, &ctx, vec![undo_key()]);
    assert!(app.edit.drag().is_none(), "the stale drag was cancelled");
    assert_eq!(app.undo.depth(), (0, 1));
    assert_eq!(ring_positions(&app, id), before, "the undo landed");
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.starts_with("Drag cancelled \u{2014} Undo:")),
        "the user hears both halves: {:?}",
        app.status,
    );

    // The button is still down: travel must not pan the camera...
    let center = app.map_view().center();
    let far = pull + egui::vec2(160.0, 0.0);
    run_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(far)]);
    assert_eq!(app.map_view().center(), center, "the pan gate stays shut");
    // ...and the release commits nothing: the undo stood.
    run_frame(&mut app, &ctx, vec![primary(false, far)]);
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.undo.depth(), (0, 1), "the release committed nothing");
    assert_eq!(
        ring_positions(&app, id),
        before,
        "the undone coordinates stood"
    );
}

#[test]
fn ctrl_z_with_an_empty_stack_leaves_a_live_drag_alone() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let start = handle_screen_pos(&app, id, 0, VertexRef::new(0));
    let pull = start + egui::vec2(30.0, -20.0);
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(start), primary(true, start)],
    );
    run_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(pull)]);
    assert!(app.edit.drag().is_some(), "the drag is live");

    run_frame(&mut app, &ctx, vec![undo_key()]);
    assert!(
        app.edit.drag().is_some(),
        "an undo that does nothing changes nothing, the live gesture included"
    );
    assert_eq!(app.status.as_deref(), Some("Nothing to undo."));

    // The surviving drag still commits.
    run_frame(&mut app, &ctx, vec![primary(false, pull)]);
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.undo.depth(), (1, 0));
}

#[test]
fn a_commit_that_lands_under_a_live_drag_drops_the_drag() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let start = handle_screen_pos(&app, id, 0, VertexRef::new(0));
    let pull = start + egui::vec2(30.0, -20.0);
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(start), primary(true, start)],
    );
    run_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(pull)]);
    assert!(app.edit.drag().is_some(), "the drag is live");

    // A programmatic commit — what a shell or the attribute form does —
    // replaces the collection under the gesture. Not the keyboard path:
    // this pins the `apply_transaction` backstop on its own.
    let nudge = nudge_transaction(&app, id);
    assert!(app.commit_edit(nudge));
    assert!(
        app.edit.drag().is_none(),
        "the commit dropped the stale drag"
    );

    run_frame(&mut app, &ctx, vec![primary(false, pull)]);
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.undo.depth(), (1, 0), "the release committed nothing");
}

#[test]
fn idle_frames_do_not_cancel_a_live_drag() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let start = handle_screen_pos(&app, id, 0, VertexRef::new(0));
    let pull = start + egui::vec2(30.0, -20.0);
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(start), primary(true, start)],
    );
    run_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(pull)]);
    for _ in 0..5 {
        run_frame(&mut app, &ctx, Vec::new());
    }
    assert!(
        app.edit.drag().is_some(),
        "a frame that changes nothing must not cancel a live drag"
    );
    run_frame(&mut app, &ctx, vec![primary(false, pull)]);
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.undo.depth(), (1, 0), "the surviving drag still commits");
}

#[test]
fn a_marquee_is_cancelled_by_an_undo_and_by_a_layer_change() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    // An entry for Ctrl+Z to yield.
    let nudge = nudge_transaction(&app, id);
    assert!(app.commit_edit(nudge));
    run_frame(&mut app, &ctx, Vec::new());
    assert!(app.edit.handles().is_active(), "handles are drawn");

    let corner = handle_screen_pos(&app, id, 0, VertexRef::new(0));
    let start = corner - egui::vec2(30.0, -30.0);
    let end = corner + egui::vec2(30.0, -30.0);
    let shift_button = |pressed, pos| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::SHIFT,
    };
    let shift_raw = |events| egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        modifiers: egui::Modifiers::SHIFT,
        events,
        ..Default::default()
    };
    let _ = ctx.run_ui(
        shift_raw(vec![
            egui::Event::PointerMoved(start),
            shift_button(true, start),
        ]),
        |ui| app.ui(ui),
    );
    let _ = ctx.run_ui(shift_raw(vec![egui::Event::PointerMoved(end)]), |ui| {
        app.ui(ui)
    });
    assert!(app.edit.marquee().is_some(), "the box-select is live");

    // Ctrl+Z: the marquee is cancelled first, then the undo applies.
    run_frame(&mut app, &ctx, vec![undo_key()]);
    assert!(app.edit.marquee().is_none(), "the box-select was cancelled");
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.starts_with("Box-select cancelled \u{2014} Undo:")),
        "the cancellation is said in the same breath as the undo: {:?}",
        app.status,
    );
    run_frame(&mut app, &ctx, vec![primary(false, end)]);

    // And a layer change closes a live box-select too — the pre-existing
    // `retarget` hole: it used to cancel only the drag.
    run_frame(&mut app, &ctx, Vec::new());
    let _ = ctx.run_ui(
        shift_raw(vec![
            egui::Event::PointerMoved(start),
            shift_button(true, start),
        ]),
        |ui| app.ui(ui),
    );
    let _ = ctx.run_ui(shift_raw(vec![egui::Event::PointerMoved(end)]), |ui| {
        app.ui(ui)
    });
    assert!(
        app.edit.marquee().is_some(),
        "the second box-select is live"
    );
    let _ = app.edit.retarget(None);
    assert!(
        app.edit.marquee().is_none(),
        "a layer change closes the live box-select"
    );
}

#[test]
fn file_new_mid_drag_does_not_hand_the_button_to_the_camera() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let start = handle_screen_pos(&app, id, 0, VertexRef::new(0));
    let pull = start + egui::vec2(30.0, -20.0);
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(start), primary(true, start)],
    );
    run_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(pull)]);
    assert!(app.edit.drag().is_some(), "the drag is live");

    // File ▸ New resets the whole edit state — but the button is still down,
    // and the rest of the gesture must not become a camera lurch.
    app.new_project();
    assert!(app.edit.drag().is_none());
    let center = app.map_view().center();
    let far = pull + egui::vec2(160.0, 0.0);
    run_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(far)]);
    assert_eq!(
        app.map_view().center(),
        center,
        "the pan latch survives the reset"
    );
}

#[test]
fn a_sketch_survives_an_undo() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    // An entry for Ctrl+Z to yield.
    let nudge = nudge_transaction(&app, id);
    assert!(app.commit_edit(nudge));

    // A half-drawn line: coordinates, not indices — nothing an undo does can
    // make it stale, so it must survive where the index-addressed gestures
    // are cancelled. The deliberate asymmetry, pinned.
    app.apply_edit_action(EditAction::SetMode(EditMode::DrawLine));
    run_frame(&mut app, &ctx, Vec::new());
    let rect = app.map_rect().expect("a frame has run");
    click_frames(&mut app, &ctx, rect.center());
    click_frames(&mut app, &ctx, rect.center() + egui::vec2(60.0, 0.0));
    assert!(app.edit.sketch().is_active(), "the sketch is in progress");

    assert!(app.undo_once());
    assert!(
        app.edit.sketch().is_active(),
        "digitized geometry survives the undo"
    );
}

// --- Editing v1.3 E2: marks restoration on undo/redo of a set move ---

/// The anchor's marked vertex set, as the overlay would draw it.
fn marked_set(app: &OxigisApp) -> Vec<VertexRef> {
    app.edit
        .multi_selection()
        .map(|multi| multi.vertex_set().to_vec())
        .unwrap_or_default()
}

#[test]
fn an_undo_of_a_set_move_restores_the_coordinates_and_the_marks() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let before = ring_positions(&app, id);
    marked_square_drag(
        &mut app,
        &ctx,
        id,
        VertexRef::new(0),
        egui::vec2(30.0, -20.0),
    );
    assert_eq!(app.undo.depth(), (1, 0));

    assert!(app.undo_once());
    assert_eq!(ring_positions(&app, id), before, "coordinates bit-exact");
    assert_eq!(
        marked_set(&app),
        vec![VertexRef::new(0), VertexRef::new(1)],
        "the amber rings come back with the geometry"
    );
    assert_eq!(
        app.edit.selection().and_then(|selection| selection.vertex),
        None,
        "marks, not a pick"
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.contains("2 vertices still marked")),
        "the status restates what the rings show: {:?}",
        app.status,
    );
}

#[test]
fn a_redo_of_a_set_move_moves_them_again_with_the_marks_still_on() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let before = ring_positions(&app, id);
    marked_square_drag(
        &mut app,
        &ctx,
        id,
        VertexRef::new(0),
        egui::vec2(30.0, -20.0),
    );
    let moved = ring_positions(&app, id);
    let set = vec![VertexRef::new(0), VertexRef::new(1)];

    // The full chain: the set is identical at every stop.
    assert!(app.undo_once());
    assert_eq!(ring_positions(&app, id), before);
    assert_eq!(marked_set(&app), set);
    assert!(app.redo_once());
    assert_eq!(ring_positions(&app, id), moved);
    assert_eq!(marked_set(&app), set);
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.contains("2 vertices still marked")),
        "the redo says so too: {:?}",
        app.status,
    );
    assert!(app.undo_once());
    assert_eq!(ring_positions(&app, id), before);
    assert_eq!(marked_set(&app), set);

    // The restored marks are live, not decorative: a fresh drag on a marked
    // handle moves the whole set again.
    run_frame(&mut app, &ctx, Vec::new());
    marked_square_drag(
        &mut app,
        &ctx,
        id,
        VertexRef::new(0),
        egui::vec2(-25.0, 10.0),
    );
    assert_ne!(ring_positions(&app, id), before, "the second move landed");
    assert_eq!(
        app.undo.peek_undo().map(|entry| entry.label),
        Some(crate::edit::MOVE_VERTICES_LABEL),
        "and it was a set move"
    );
}

#[test]
fn undoing_a_plain_single_vertex_move_marks_nothing() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let start = handle_screen_pos(&app, id, 0, VertexRef::new(0));
    let pull = start + egui::vec2(30.0, -20.0);
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(start), primary(true, start)],
    );
    run_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(pull)]);
    run_frame(&mut app, &ctx, vec![primary(false, pull)]);
    run_frame(&mut app, &ctx, Vec::new());
    assert_eq!(app.undo.depth(), (1, 0), "the single move committed");

    assert!(app.undo_once());
    assert_eq!(
        marked_set(&app),
        Vec::<VertexRef>::new(),
        "one moved vertex is a pick, never a marked set — undoing a v1.1 \
         single-vertex move looks exactly as it always did"
    );
}

#[test]
fn undoing_a_marked_vertex_delete_restores_the_vertices_unmarked() {
    let ctx = Context::default();
    let (mut app, id) = square_ready(&ctx);
    let marked = crate::edit::selection::FeatureSelection::single(0)
        .with_vertex_set(vec![VertexRef::new(1)]);
    app.edit.set_multi_selection(Some(marked));
    assert!(app.delete_selected_vertex(), "a legal set deletes");

    assert!(app.undo_once());
    assert_eq!(
        ring_positions(&app, id).len(),
        5,
        "the vertex is back (4 open corners + the closing duplicate)"
    );
    assert_eq!(
        marked_set(&app),
        Vec::<VertexRef>::new(),
        "the \u{2265} 2 rule, not the arity boundary: ONE restored vertex is a \
         pick, never a marked set — and marking it would change what Delete \
         means one keystroke later"
    );
}

#[test]
fn an_undo_for_another_layer_does_not_stamp_marks_onto_this_one() {
    let ctx = Context::default();
    let (mut app, first) = square_ready(&ctx);
    marked_square_drag(
        &mut app,
        &ctx,
        first,
        VertexRef::new(0),
        egui::vec2(30.0, -20.0),
    );
    assert_eq!(app.undo.depth(), (1, 0));

    // Look at another layer; the global stack still names `first`.
    let second = app
        .add_geojson_layer_from_text("named", NAMED, None)
        .expect("valid GeoJSON");
    app.selection = Some(second);
    app.sync_edit_state();
    app.edit.set_selection(Some(EditSelection::feature(0)));

    assert!(app.undo_once());
    assert_eq!(
        marked_set(&app),
        Vec::<VertexRef>::new(),
        "the undone set move belongs to `first`; its marks must not land \
         on the layer the user is looking at"
    );
    let _ = (first, second);
}
