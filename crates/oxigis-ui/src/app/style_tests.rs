//! Per-family styling gesture tests (tiles v1.3 item C): the undo seam, the
//! slot discriminator and the remove/restore round trip, driven through the
//! same `sync_local_style` seam the panel uses each frame.

use crate::local_input::LocalLayerOp;
use oxigis_core::{GeometryFamily, LayerStyle, StyleSlot};

use super::OxigisApp;

/// A polygon-dominant mixed dataset: the seeded default set carries a line
/// override, so the family machinery is live from the first frame.
const MIXED: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{},
     "geometry":{"type":"Polygon","coordinates":[[[0.0,0.0],[10.0,0.0],
       [10.0,10.0],[0.0,0.0]]]}},
    {"type":"Feature","properties":{},
     "geometry":{"type":"Polygon","coordinates":[[[20.0,0.0],[30.0,0.0],
       [30.0,10.0],[20.0,0.0]]]}},
    {"type":"Feature","properties":{},
     "geometry":{"type":"LineString","coordinates":[[0.0,0.0],[10.0,10.0]]}}]}"#;

/// One "slider frame": mutate the selected layer's line override, then run
/// the frame's style seam exactly as `OxigisApp::ui` does.
fn drag_line_width(app: &mut OxigisApp, width: f32) {
    let id = app.selection.expect("a layer is selected");
    let before = app.project.styles.get(&id).cloned();
    if let Some(set) = app.project.styles.get_mut(&id)
        && let Some(LayerStyle::Line(line)) = set.slot_mut(StyleSlot::Family(GeometryFamily::Line))
    {
        line.set_width(width);
    }
    app.sync_local_style(before);
}

#[test]
fn a_full_per_family_line_drag_is_one_undo_step_that_restores_the_whole_set() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("mixed", MIXED, None)
        .expect("the mixed fixture loads");
    let original = app
        .project
        .styles
        .get(&id)
        .cloned()
        .expect("the add seeded a style set");
    assert!(
        matches!(
            original.override_for(GeometryFamily::Line),
            Some(LayerStyle::Line(_))
        ),
        "a mixed drop seeds the non-dominant family"
    );
    let add_depth = app.undo.depth().0;
    let _ = app.local.take_ops();

    // Three slider frames of ONE drag (the coalesce key is constant:
    // same epoch, same layer, same slot): one undo step.
    for width in [3.0_f32, 4.0, 5.0] {
        drag_line_width(&mut app, width);
    }
    assert_eq!(
        app.undo.depth().0,
        add_depth + 1,
        "three frames of one drag fold into ONE undo step"
    );
    let after = app.project.styles.get(&id).cloned().expect("still styled");
    let Some(LayerStyle::Line(line)) = after.override_for(GeometryFamily::Line) else {
        panic!("the line override survived");
    };
    assert_eq!(line.width(), 5.0, "the final width holds");
    assert!(
        matches!(
            app.local.take_ops().last(),
            Some(LocalLayerOp::SetStyle(other, set))
                if *other == id && set.override_for(GeometryFamily::Line).is_some()
        ),
        "the GPU restyle carries the WHOLE set"
    );

    // One Ctrl+Z restores the whole pre-drag set — overrides included.
    assert!(app.undo_once());
    assert_eq!(app.project.styles.get(&id), Some(&original));
    assert!(
        matches!(
            app.local.take_ops().last(),
            Some(LocalLayerOp::SetStyle(other, set))
                if *other == id && set == &original
        ),
        "the undo queues the restored set to the GPU"
    );

    // And redo brings the drag's result back.
    assert!(app.redo_once());
    assert_eq!(app.project.styles.get(&id), Some(&after));
}

#[test]
fn a_small_layer_keeps_its_live_preview_all_through_a_drag() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("mixed", MIXED, None)
        .expect("the mixed fixture loads");
    let _ = app.local.take_ops();
    let before = app.project.styles.get(&id).cloned();
    if let Some(set) = app.project.styles.get_mut(&id)
        && let Some(LayerStyle::Line(line)) = set.slot_mut(StyleSlot::Family(GeometryFamily::Line))
    {
        line.set_width(9.0);
    }
    // Pointer HELD, i.e. mid-drag. Three features is nothing to re-tessellate,
    // so the preview must stay immediate.
    app.sync_local_style_gated(before, true);
    assert!(app.deferred_restyle.is_none());
    assert!(
        matches!(app.local.take_ops().last(), Some(LocalLayerOp::SetStyle(other, _)) if *other == id),
        "a small layer restyles on the frame the slider moved"
    );
}

#[test]
fn a_deferred_restyle_lands_the_moment_the_pointer_lifts() {
    // The expensive-layer half of the deferral, driven through the state it
    // sets rather than through a 20 000-feature fixture: what matters is that
    // the pending id is honoured exactly once, on the release edge, from the
    // project's CURRENT style.
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("mixed", MIXED, None)
        .expect("the mixed fixture loads");
    let _ = app.local.take_ops();
    app.deferred_restyle = Some(id);

    // Still held: nothing is queued and the deferral survives.
    let unchanged = app.project.styles.get(&id).cloned();
    app.sync_local_style_gated(unchanged.clone(), true);
    assert_eq!(app.deferred_restyle, Some(id));
    assert!(app.local.take_ops().is_empty());

    // Released: exactly one restyle, and the deferral is spent.
    app.sync_local_style_gated(unchanged, false);
    assert_eq!(app.deferred_restyle, None);
    let ops = app.local.take_ops();
    assert_eq!(ops.len(), 1);
    assert!(matches!(&ops[0], LocalLayerOp::SetStyle(other, _) if *other == id));
    // And it does not fire twice.
    let after = app.project.styles.get(&id).cloned();
    app.sync_local_style_gated(after, false);
    assert!(app.local.take_ops().is_empty());
}

#[test]
fn removing_and_undoing_a_mixed_layer_restores_its_overrides() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("mixed", MIXED, None)
        .expect("the mixed fixture loads");
    let styled = app
        .project
        .styles
        .get(&id)
        .cloned()
        .expect("the add seeded a style set");
    assert!(styled.has_overrides(), "a mixed drop seeds an override");

    app.apply_layer_action(crate::layer_panel::LayerAction::Remove(id));
    assert!(!app.project.styles.contains_key(&id));
    assert!(app.undo_once(), "the removal undoes");
    assert_eq!(
        app.project.styles.get(&id),
        Some(&styled),
        "the snapshot carries the whole SET, so the overrides come back"
    );
}
