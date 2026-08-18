// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The basemap **service** on the undo stack (editing v1.5 item 2): the
//! service-carrying `ProjectOp::SetBasemap`, the one writer that owns both of
//! the service's homes, and the recorder that puts every pick on the stack.
//!
//! Separate from `app/tests_basemap.rs`, which holds the v1.4 promoted-layer
//! *pointer* family; these tests are about the service the pointer displaces.
//! Every assertion is a value assertion — no rendering, no network.

use crate::edit::project_op::{BasemapServiceChange, ProjectOp, ProjectTransaction};
use crate::layer_panel::LayerAction;
use crate::tile_provider::BasemapConfig;
use oxigis_core::{Layer, LayerKind, Project, RasterSource};

use super::*;

/// A valid, distinct service with a credit line of its own.
fn service(host: &str) -> BasemapConfig {
    BasemapConfig {
        url_template: format!("https://{host}/{{z}}/{{x}}/{{y}}.png"),
        subdomains: Vec::new(),
        attribution: format!("\u{a9} {host}"),
    }
}

/// A synthetic service-only transaction, as the recorder would build it.
fn service_transaction(before: BasemapConfig, after: BasemapConfig) -> ProjectTransaction {
    ProjectTransaction {
        label: "Set basemap",
        op: ProjectOp::SetBasemap {
            before: None,
            after: None,
            service: Some(Box::new(BasemapServiceChange { before, after })),
        },
        coalesce: None,
    }
}

#[test]
fn the_applier_writes_both_homes_of_the_service() {
    // `self.basemap` is what every consumer derives from; `project.basemap`
    // is what a save serializes. One writer, both homes, always together —
    // a half-write is the divergence the writer exists to prevent.
    let mut app = OxigisApp::new();
    let target = service("tiles.example.test");
    let transaction = service_transaction(app.basemap().clone(), target.clone());

    assert_eq!(app.apply_project_transaction(&transaction), Ok(()));
    assert_eq!(app.basemap(), &target, "the drawn service moved");
    assert_eq!(
        app.project().basemap.as_ref().map(BasemapConfig::from),
        Some(target.clone()),
        "the serialized mirror moved with it"
    );
    assert_eq!(
        app.drawn_basemap(),
        target,
        "nothing is promoted, so the service is what draws"
    );
    assert_eq!(app.desired_raster().basemap, target);
}

#[test]
fn undoing_a_synthetic_service_op_restores_both_homes() {
    let mut app = OxigisApp::new();
    let original = app.basemap().clone();
    let target = service("tiles.example.test");
    let transaction = service_transaction(original.clone(), target.clone());

    assert_eq!(app.apply_project_transaction(&transaction), Ok(()));
    assert_eq!(app.basemap(), &target);

    // The inverse is the same op with its sides swapped — recorder and
    // applier are the same code in both directions.
    assert_eq!(
        app.apply_project_transaction(&transaction.inverted()),
        Ok(())
    );
    assert_eq!(app.basemap(), &original, "the drawn service came back");
    assert_eq!(
        app.project().basemap.as_ref().map(BasemapConfig::from),
        Some(original),
        "and so did the serialized mirror"
    );
}

#[test]
fn a_refused_pointer_leaves_the_service_untouched() {
    // Check-first: EVERY check runs before ANY write, so a fused entry whose
    // pointer half cannot apply must not have moved the service half.
    let mut app = OxigisApp::new();
    let untouched = app.basemap().clone();
    let ghost = ProjectTransaction {
        label: "Set basemap",
        op: ProjectOp::SetBasemap {
            before: None,
            after: Some(LayerId::from_raw(999_999)),
            service: Some(Box::new(BasemapServiceChange {
                before: untouched.clone(),
                after: service("tiles.example.test"),
            })),
        },
        coalesce: None,
    };

    assert!(
        app.apply_project_transaction(&ghost).is_err(),
        "a pointer at a layer that is not there cannot be written"
    );
    assert_eq!(
        app.basemap(),
        &untouched,
        "a partial apply would have moved the service anyway"
    );
    assert_eq!(
        app.project().basemap,
        None,
        "and would have stamped the serialized mirror"
    );
    assert_eq!(app.project().basemap_layer, None);
}

// --- Hostile matrix (docs/plans/editing-v15.md) ---

/// A credited XYZ layer, appended on top and returned.
fn add_xyz(app: &mut OxigisApp, name: &str, url_template: &str) -> LayerId {
    app.project.layers.add(Layer::new(
        name,
        LayerKind::Raster(RasterSource::Xyz {
            url_template: url_template.to_string(),
            attribution: "\u{a9} Example".to_string(),
        }),
    ))
}

/// Picks `config` the way the basemap picker does.
fn pick(app: &mut OxigisApp, config: BasemapConfig) {
    app.apply_layer_action(LayerAction::SetBasemapPreset(config));
}

#[test]
fn b_h1_pick_then_pick_then_undo_restores_the_first_service() {
    let mut app = OxigisApp::new();
    let first = app.basemap().clone();
    let second = service("second.example.test");
    let third = service("third.example.test");

    pick(&mut app, second.clone());
    assert_eq!(app.undo.depth(), (1, 0), "a pick is ONE entry");
    pick(&mut app, third.clone());
    assert_eq!(app.undo.depth(), (2, 0));
    assert_eq!(app.basemap(), &third);

    assert!(app.undo_once());
    assert_eq!(app.basemap(), &second, "back one pick, not all the way");
    assert!(app.undo_once());
    assert_eq!(app.basemap(), &first, "and back to where the session began");
}

#[test]
fn b_h3_redo_re_applies_the_pick() {
    let mut app = OxigisApp::new();
    let picked = service("second.example.test");
    pick(&mut app, picked.clone());
    assert!(app.undo_once());
    assert_ne!(app.basemap(), &picked);
    assert!(app.redo_once());
    assert_eq!(app.basemap(), &picked, "redo puts the pick back");
    assert_eq!(
        app.project().basemap.as_ref().map(BasemapConfig::from),
        Some(picked),
        "both homes, in both directions"
    );
}

#[test]
fn b_h4_pick_then_promote_undoes_in_lifo_order() {
    let mut app = OxigisApp::new();
    let original = app.basemap().clone();
    let picked = service("second.example.test");
    pick(&mut app, picked.clone());
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
    );
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    assert!(app.draws_as_basemap(id));

    assert!(app.undo_once());
    assert!(!app.draws_as_basemap(id), "the promotion is undone first");
    assert_eq!(app.basemap(), &picked, "and the pick is untouched");
    assert!(app.undo_once());
    assert_eq!(app.basemap(), &original, "then the pick");
}

#[test]
fn b_h5_a_refused_template_records_nothing_and_moves_nothing() {
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
    );
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    let depth = app.undo.depth();
    let untouched = app.basemap().clone();

    pick(
        &mut app,
        BasemapConfig {
            url_template: "https://broken.example.test/tiles.png".to_string(),
            subdomains: Vec::new(),
            attribution: String::new(),
        },
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.starts_with("Basemap not changed")),
        "the refusal is loud: {:?}",
        app.status
    );
    assert_eq!(app.undo.depth(), depth, "and records nothing at all");
    assert_eq!(app.basemap(), &untouched);
    assert!(app.draws_as_basemap(id), "the promotion is left alone");
}

#[test]
fn b_h6_re_picking_the_active_service_while_promoted_records_the_demote_only() {
    let mut app = OxigisApp::new();
    let active = app.basemap().clone();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
    );
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    let depth = app.undo.depth();

    pick(&mut app, active.clone());
    assert_eq!(
        app.undo.depth(),
        (depth.0 + 1, 0),
        "ONE entry for the demote"
    );
    assert!(!app.draws_as_basemap(id));
    assert_eq!(app.basemap(), &active, "the service never moved");
    assert_eq!(
        app.project().basemap,
        None,
        "sides that are equal carry no payload, so nothing was stamped"
    );

    assert!(app.undo_once());
    assert!(app.draws_as_basemap(id));
    assert_eq!(app.basemap(), &active);
}

#[test]
fn b_h7_re_picking_the_active_service_with_nothing_promoted_records_nothing() {
    let mut app = OxigisApp::new();
    let active = app.basemap().clone();
    pick(&mut app, active.clone());
    assert_eq!(
        app.undo.depth(),
        (0, 0),
        "no empty entries: a Ctrl+Z through a step that changed nothing is noise"
    );
    assert_eq!(app.basemap(), &active);
}

#[test]
fn b_h8_a_hydrate_prunes_a_fused_entry_and_spares_a_service_only_one() {
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text(
            "points",
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},
                 "geometry":{"type":"Point","coordinates":[0.0,0.0]}}]}"#,
            None,
        )
        .expect("valid GeoJSON");
    // A service-only entry mentions no layer at all.
    app.record_undo(service_transaction(
        app.basemap().clone(),
        service("kept.example.test"),
    ));
    // A fused entry names the layer it demoted, so it goes whole.
    app.record_undo(ProjectTransaction {
        label: "Set basemap",
        op: ProjectOp::SetBasemap {
            before: Some(id),
            after: None,
            service: Some(Box::new(BasemapServiceChange {
                before: app.basemap().clone(),
                after: service("dropped.example.test"),
            })),
        },
        coalesce: None,
    });
    let depth = app.undo.depth().0;
    assert!(depth >= 2);

    assert!(app.hydrate_geojson_layer_from_bytes(
        id,
        "points",
        br#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},
             "geometry":{"type":"Point","coordinates":[1.0,1.0]}}]}"#,
    ));
    assert_eq!(
        app.undo.depth().0,
        depth - 1,
        "the fused entry went with the layer's data; the service-only entry \
         survived, or the user's basemap history would be silently erased"
    );
}

#[test]
fn b_h9_a_refused_service_stays_refused_across_undo_and_redo() {
    // The memo names a PLAN, and it is DERIVED: it suppresses only while the
    // plan it refused is still what the app desires. Undo to A makes it
    // invisible with no new code; redo back to B makes it visible again.
    let mut app = OxigisApp::new();
    let first = app.pending_raster_work().expect("the first plan");
    app.settle_raster_work(first, Ok(()));

    let refused = service("refused.example.test");
    pick(&mut app, refused.clone());
    let work = app.pending_raster_work().expect("the pick replans");
    assert_eq!(work.basemap, refused);
    app.settle_raster_work(work, Err("no worker pool".to_string()));
    assert_eq!(app.raster_refusal(), Some("no worker pool"));
    assert!(
        app.pending_raster_work().is_none(),
        "the memo suppresses the refused plan"
    );

    assert!(app.undo_once());
    assert_eq!(
        app.raster_refusal(),
        None,
        "the memo cannot describe a plan the app no longer wants"
    );
    assert!(
        app.pending_raster_work().is_none(),
        "and the undone-to service is the one already installed, so the undo \
         costs no rebuild either"
    );

    assert!(app.redo_once());
    assert_eq!(app.basemap(), &refused);
    assert_eq!(
        app.raster_refusal(),
        Some("no worker pool"),
        "the same plan is refused again, from the same memo"
    );
    assert!(app.pending_raster_work().is_none());
}

#[test]
fn b_h10_a_pre_field_load_undoes_to_the_service_that_was_drawing() {
    // `apply_loaded_basemap` returns EARLY for a file with no `basemap`
    // field, leaving `self.basemap` at the previous project's service. So
    // `project.basemap` and `self.basemap` really do diverge, and recording
    // `before` from the serialized field would record "nothing" and undo to
    // the default.
    let mut app = OxigisApp::new();
    let a = service("project-a.example.test");
    pick(&mut app, a.clone());
    app.sync_project_view();
    let with_basemap = app.project().to_json_string().expect("serialize");
    app.load_project(Project::from_json_string(&with_basemap).expect("deserialize"));
    assert_eq!(app.basemap(), &a);

    // A file from a build that predates `Project::basemap`.
    let pre_field = Project::new("pre-field");
    assert_eq!(pre_field.basemap, None);
    app.load_project(pre_field);
    assert_eq!(app.project().basemap, None, "the field really is absent");
    assert_eq!(app.basemap(), &a, "but A's service is what is on screen");

    pick(&mut app, service("after.example.test"));
    assert!(app.undo_once());
    assert_eq!(
        app.basemap(),
        &a,
        "the undo restores what was DRAWING, not the default"
    );
    assert_ne!(app.basemap(), &BasemapConfig::default());
}

#[test]
fn b_h11_a_refused_saved_basemap_undoes_to_the_service_that_was_drawing() {
    // The same divergence through the other door: the load keeps the drawn
    // service and leaves the unusable value in `project.basemap`.
    let mut app = OxigisApp::new();
    let drawing = service("drawing.example.test");
    pick(&mut app, drawing.clone());

    let mut damaged = Project::new("damaged");
    damaged.basemap = Some(oxigis_core::ProjectBasemap {
        url_template: "https://no-placeholders.example.test/tiles.png".to_string(),
        subdomains: Vec::new(),
        attribution: String::new(),
    });
    app.load_project(damaged);
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.contains("unusable")),
        "the damaged basemap is reported: {:?}",
        app.status
    );
    assert_eq!(app.basemap(), &drawing, "and the drawn service is kept");

    pick(&mut app, service("after.example.test"));
    assert!(app.undo_once());
    assert_eq!(
        app.basemap(),
        &drawing,
        "the undo restores the kept service, not the unusable saved one"
    );
}

#[test]
fn b_h12_save_pick_undo_save_is_byte_identical() {
    let mut app = OxigisApp::new();
    add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
    );
    app.sync_project_view();
    let before = app.project().to_json_string().expect("serialize");

    pick(&mut app, service("second.example.test"));
    assert!(app.undo_once());
    app.sync_project_view();
    let after = app.project().to_json_string().expect("serialize");
    assert_eq!(
        before, after,
        "an undone pick leaves no trace in the saved bytes"
    );
}

#[test]
fn b_h13_file_new_and_a_load_record_nothing() {
    let mut app = OxigisApp::new();
    pick(&mut app, service("second.example.test"));
    assert_eq!(app.undo.depth(), (1, 0));

    app.new_project();
    assert_eq!(
        app.undo.depth(),
        (0, 0),
        "File — New resets the stack; nothing about the basemap is recorded"
    );
    assert_eq!(app.basemap(), &BasemapConfig::default());
    assert_eq!(
        app.project().basemap,
        None,
        "and a brand-new project has no saved basemap to stamp"
    );

    pick(&mut app, service("third.example.test"));
    app.sync_project_view();
    let json = app.project().to_json_string().expect("serialize");
    app.load_project(Project::from_json_string(&json).expect("deserialize"));
    assert_eq!(app.undo.depth(), (0, 0), "a load resets it too");
}

#[test]
fn b_h14_undoing_a_service_change_under_a_promotion_moves_no_plan() {
    // `drawn_basemap` returns the PROMOTED layer's config, so a service the
    // promotion covers is not part of the desired plan: undoing it must not
    // churn the provider or drop the tile cache.
    //
    // The entry is built by hand with a DEGENERATE pointer pair
    // (`before == after`), because no gesture records that shape: the picker
    // always demotes, so a recorded pick's pointer half moves. This is the
    // one construction that isolates the property — the pointer stays put
    // across the undo, and only the covered service moves.
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
    );
    app.project.basemap_layer = Some(id);
    assert!(app.draws_as_basemap(id));
    let planned = app.desired_raster();
    let covered = app.basemap().clone();

    let transaction = ProjectTransaction {
        label: "Set basemap",
        op: ProjectOp::SetBasemap {
            before: Some(id),
            after: Some(id),
            service: Some(Box::new(BasemapServiceChange {
                before: covered.clone(),
                after: service("under.example.test"),
            })),
        },
        coalesce: None,
    };
    assert_eq!(app.apply_project_transaction(&transaction), Ok(()));
    assert_eq!(
        app.desired_raster(),
        planned,
        "the covered service moved but the plan did not, so no rebuild"
    );

    app.record_undo(transaction);
    assert!(app.undo_once());
    assert_eq!(
        app.desired_raster(),
        planned,
        "and the undo does not churn it back"
    );
    assert!(app.draws_as_basemap(id), "the promotion still draws");
    assert_eq!(
        app.basemap(),
        &covered,
        "while the service underneath is back where it was \u{2014} invisible, \
         which is why the status line is the only honest signal"
    );
}

#[test]
fn b_h16_set_basemap_validates_stamps_and_records_exactly_one_entry() {
    // The public seam is an EDIT since editing v1.5: an unrecorded writer of
    // the service is the defect the recorded op exists to close.
    let mut app = OxigisApp::new();
    let original = app.basemap().clone();
    let target = service("public.example.test");
    app.set_basemap(target.clone());
    assert_eq!(app.basemap(), &target);
    assert_eq!(
        app.project().basemap.as_ref().map(BasemapConfig::from),
        Some(target),
        "it stamps the project, which it never did before"
    );
    assert_eq!(app.undo.depth(), (1, 0), "exactly one entry");
    assert!(app.undo_once());
    assert_eq!(app.basemap(), &original);

    // And it refuses an unusable template, changing and recording nothing.
    let depth = app.undo.depth();
    let kept = app.basemap().clone();
    app.set_basemap(BasemapConfig {
        url_template: "https://broken.example.test/tiles.png".to_string(),
        subdomains: Vec::new(),
        attribution: String::new(),
    });
    assert_eq!(app.basemap(), &kept);
    assert_eq!(app.undo.depth(), depth);
}

#[test]
fn b_h17_undo_and_redo_of_a_service_round_trip_five_times() {
    let mut app = OxigisApp::new();
    add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
    );
    // Stamped first, so the comparison is between two SAVED basemaps rather
    // than across the one-way `None -> Some(...)` step. That step is the
    // carried risk editing v1.4 already named: undoing past the FIRST service
    // change of a session stamps a field a pre-field file left empty. It is
    // strictly better in value than v1.4's (the actual prior service, not a
    // default) and invisible at save, because `sync_project_view` stamps at
    // every File > Save anyway.
    app.sync_project_view();
    let before = app.project().clone();
    let target = service("round.example.test");
    pick(&mut app, target.clone());
    let after = app.project().clone();
    assert_ne!(before.basemap, after.basemap);

    for round in 0..5 {
        assert!(app.undo_once(), "undo {round}");
        assert_eq!(app.project(), &before, "undo {round} restores the project");
        assert!(app.redo_once(), "redo {round}");
        assert_eq!(app.project(), &after, "redo {round} restores the project");
        assert_eq!(app.basemap(), &target);
    }
}

#[test]
fn b_h15_undoing_a_service_pick_mid_drag_cancels_the_drag_and_says_both() {
    // The pick is a service change with nothing promoted, so it is a
    // service-ONLY entry — and a Ctrl+Z through it still cancels the live
    // gesture first, exactly as the promotion case does. Same fixture as
    // `tests_basemap::undoing_a_promotion_mid_drag_cancels_the_drag_and_says_both`,
    // reused rather than re-derived so the two cannot drift.
    let ctx = egui::Context::default();
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("plot", super::tests_basemap::SQUARE, None);
    let plot = app.selection().expect("the square is selected");
    let service_before = app.basemap().clone();
    pick(&mut app, service("mid-drag.example.test"));
    assert_eq!(app.undo.depth(), (1, 0), "the pick is on the stack");

    app.selection = Some(plot);
    app.map_panel = crate::map_view::MapPanelState::new(0.0, 0.0, 3.0, [1024.0, 768.0]);
    app.edit.set_mode(crate::edit::EditMode::Select);
    let _notice = app.edit.retarget(Some(plot));
    app.edit
        .set_selection(Some(crate::edit::EditSelection::feature(0)));
    super::tests_basemap::run_frame(&mut app, &ctx, Vec::new());
    assert!(app.edit.handles().is_active(), "handles are drawn");

    // Grab a corner and hold it: the drag is live and has committed nothing.
    let rect = app.map_rect().expect("a frame has run");
    let features = app.local.feature_set(plot).expect("loaded").clone();
    let geometry = features.features[0]
        .geometry
        .as_ref()
        .expect("the square has geometry");
    let position = crate::edit::hit::handle_position(
        geometry,
        crate::edit::VertexRef::new(0),
        false,
        app.map_view(),
        1.0,
    )
    .expect("the handle exists");
    let start = crate::edit::hit::to_screen(app.map_view(), rect.min, 1.0, position);
    let button = |pressed, pos| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    super::tests_basemap::run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(start), button(true, start)],
    );
    super::tests_basemap::run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(start + egui::vec2(20.0, -15.0))],
    );

    assert!(app.undo_once());
    assert_eq!(
        app.status.as_deref(),
        Some("Drag cancelled \u{2014} Undo: Set basemap"),
        "one press does both, and says both"
    );
    assert_eq!(
        app.basemap(),
        &service_before,
        "and the service really did go back"
    );
}
