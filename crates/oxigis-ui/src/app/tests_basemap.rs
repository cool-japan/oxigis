// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The promoted-basemap family (editing v1.4 item 1): a soft
//! `Project::basemap_layer` pointer, its total resolution, the three
//! consumers that read it, and the recorded `ProjectOp::SetBasemap` that
//! moves it.
//!
//! Separate from `app/tests_providers.rs`, which holds the v1.3
//! reconciliation family; these tests are about which basemap is *chosen*,
//! not about how a chosen one reaches the GPU.

use crate::layer_panel::LayerAction;
use crate::tile_provider::{BasemapConfig, OSM_ATTRIBUTION};
use oxigis_core::{Layer, LayerId, LayerKind, Project, RasterSource};

use super::*;

/// A credited XYZ layer, appended on top and returned.
fn add_xyz(app: &mut OxigisApp, name: &str, url_template: &str, attribution: &str) -> LayerId {
    app.project.layers.add(Layer::new(
        name,
        LayerKind::Raster(RasterSource::Xyz {
            url_template: url_template.to_string(),
            attribution: attribution.to_string(),
        }),
    ))
}

/// The layer's own service, as a basemap configuration.
fn xyz_config(url_template: &str, attribution: &str) -> BasemapConfig {
    BasemapConfig {
        url_template: url_template.to_string(),
        subdomains: Vec::new(),
        attribution: attribution.to_string(),
    }
}

#[test]
fn a_promoted_layer_draws_as_the_basemap() {
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "\u{a9} Example",
    );
    assert!(!app.draws_as_basemap(id), "nothing is promoted yet");
    assert_eq!(app.drawn_basemap(), *app.basemap());

    app.project.basemap_layer = Some(id);
    let promoted = xyz_config(
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "\u{a9} Example",
    );
    assert!(app.draws_as_basemap(id));
    assert_eq!(app.drawn_basemap(), promoted);
    assert_eq!(
        app.desired_raster().basemap,
        promoted,
        "the raster plan draws the promoted layer, not the service"
    );
    assert_eq!(
        app.credit_line(),
        "\u{a9} Example",
        "the promoted layer's credit REPLACES the service's — crediting \
         tiles that are not on screen is as wrong as omitting the ones that \
         are"
    );
}

#[test]
fn a_hidden_promoted_layer_falls_back_to_the_service() {
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "\u{a9} Example",
    );
    app.project.basemap_layer = Some(id);
    assert!(app.draws_as_basemap(id));

    app.apply_layer_action(LayerAction::ToggleVisibility(id));
    assert!(
        !app.draws_as_basemap(id),
        "the toggle reflects RESOLUTION, not the stored pointer"
    );
    assert_eq!(app.drawn_basemap(), *app.basemap());
    assert_eq!(
        app.project().basemap_layer,
        Some(id),
        "the promotion is kept: showing the layer again re-promotes it"
    );

    app.apply_layer_action(LayerAction::ToggleVisibility(id));
    assert!(app.draws_as_basemap(id));
}

/// The three ways a hand-edited `basemap_layer` pointer can fail to resolve.
#[derive(Debug, Clone, Copy)]
enum Broken {
    /// It names no layer the project holds.
    Absent,
    /// It names a layer that is not an XYZ raster.
    WrongKind,
    /// It names an XYZ layer whose template cannot be expanded.
    BadTemplate,
}

impl Broken {
    /// Builds the layer (if any) and returns the id the pointer should carry.
    fn pointed_at(self, app: &mut OxigisApp) -> LayerId {
        match self {
            Self::Absent => LayerId::from_raw(999_999),
            Self::WrongKind => app.project.layers.add(Layer::new(
                "scene.tif",
                LayerKind::Raster(RasterSource::Cog {
                    url: "https://example.org/scene.tif".to_string(),
                }),
            )),
            // `{s}` with no subdomain list: a layer records no host list, so
            // every tile fetch would fail at expansion.
            Self::BadTemplate => add_xyz(
                app,
                "Rotating",
                "https://{s}.example.org/{z}/{x}/{y}.png",
                "",
            ),
        }
    }
}

#[test]
fn an_unresolvable_pointer_falls_back_and_is_never_scrubbed() {
    // Three hand-edited shapes, each of which must fall back to the service,
    // report ONE load notice, and keep the pointer.
    for shape in [Broken::Absent, Broken::WrongKind, Broken::BadTemplate] {
        let mut app = OxigisApp::new();
        let id = shape.pointed_at(&mut app);
        app.project.basemap_layer = Some(id);
        let mut project = app.project().clone();
        project.basemap = Some((&BasemapConfig::default()).into());

        let mut loader = OxigisApp::new();
        loader.load_project(project);
        assert!(
            !loader.draws_as_basemap(id),
            "an unresolvable pointer draws nothing: {shape:?}"
        );
        assert_eq!(loader.drawn_basemap(), BasemapConfig::default());
        assert_eq!(
            loader.project().basemap_layer,
            Some(id),
            "a load must not silently mutate a file the user may re-save"
        );
        assert!(
            loader
                .status
                .as_deref()
                .is_some_and(|status| status.contains("basemap layer")
                    && status.contains("drawing the saved basemap service instead.")),
            "one load notice, naming what happened: {:?}",
            loader.status,
        );
    }
}

#[test]
fn a_promoted_layer_and_its_credit_survive_the_save_load_round_trip() {
    let mut saver = OxigisApp::new();
    saver.apply_layer_action(LayerAction::AddDemoXyzLayer);
    let id = saver.selection().expect("the demo layer is selected");
    saver.project.basemap_layer = Some(id);
    saver.sync_project_view();
    let json = saver.project().to_json_string().expect("serialize");

    let mut loader = OxigisApp::new();
    loader.load_project(Project::from_json_string(&json).expect("deserialize"));
    assert!(loader.draws_as_basemap(id));
    assert_eq!(loader.drawn_basemap().attribution, OSM_ATTRIBUTION);
    assert_eq!(
        loader.credit_line(),
        OSM_ATTRIBUTION,
        "the layer carries the licence's exact credit across the file"
    );
    assert!(
        loader
            .status
            .as_deref()
            .is_some_and(|status| !status.contains("basemap layer")),
        "a pointer that resolves is not a notice: {:?}",
        loader.status,
    );
}

#[test]
fn a_promoted_layer_under_a_cog_is_one_raster_plan() {
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "\u{a9} Example",
    );
    app.project.basemap_layer = Some(id);
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/scene.tif".to_string(),
    ));

    let work = app.desired_raster();
    assert_eq!(
        work.basemap,
        xyz_config(
            "https://tiles.example.org/{z}/{x}/{y}.png",
            "\u{a9} Example"
        ),
    );
    assert_eq!(
        work.cog.as_ref().map(|cog| cog.url.as_str()),
        Some("https://example.org/scene.tif"),
        "one plan: the promoted layer IS the basemap the COG composites over, \
         so a promoted layer can never double-draw"
    );
}

#[test]
fn reordering_the_promoted_layer_changes_nothing() {
    let mut app = OxigisApp::new();
    let promoted = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "\u{a9} Example",
    );
    let _other = add_xyz(
        &mut app,
        "Other",
        "https://other.example/{z}/{x}/{y}.png",
        "",
    );
    app.project.basemap_layer = Some(promoted);
    let before = app.desired_raster();

    app.apply_layer_action(LayerAction::MoveUp(promoted));
    assert!(
        app.draws_as_basemap(promoted),
        "the promotion is an explicit id, not a stack position"
    );
    assert_eq!(app.desired_raster(), before);
}

// --- Editing v1.4 E5: the recorded op, the gesture and the picker demote ---

#[test]
fn promoting_a_layer_is_one_recorded_step() {
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "\u{a9} Example",
    );
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    assert_eq!(app.project().basemap_layer, Some(id));
    assert_eq!(app.undo.depth(), (1, 0), "one gesture, one entry");
    assert_eq!(
        app.undo
            .peek_undo_entry()
            .map(crate::edit::stack::UndoEntry::label),
        Some("Set basemap")
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.contains("now draws as the basemap")
                && status.ends_with("Ctrl+Z restores the previous basemap.")),
        "the status says what happened and how to reverse it: {:?}",
        app.status,
    );

    assert!(app.undo_once());
    assert_eq!(app.project().basemap_layer, None);
    assert_eq!(app.drawn_basemap(), *app.basemap());
    assert!(app.redo_once());
    assert!(app.draws_as_basemap(id));
}

#[test]
fn re_promoting_the_same_layer_records_nothing() {
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "",
    );
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    let depth = app.undo.depth();
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    assert_eq!(
        app.undo.depth(),
        depth,
        "`before == after` is not a step: an undo that changes nothing is \
         history the user has to press through for no reason"
    );
}

#[test]
fn promoting_a_hidden_layer_is_recorded_and_says_the_service_still_draws() {
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "",
    );
    app.apply_layer_action(LayerAction::ToggleVisibility(id));
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    assert_eq!(app.project().basemap_layer, Some(id), "recorded anyway");
    // TWO entries, not one: the hide is itself a recorded step now (the panel's
    // checkbox goes through `toggle_layer_visibility`), so the promotion sits
    // on top of it.
    assert_eq!(app.undo.depth(), (2, 0));
    assert!(!app.draws_as_basemap(id));
    assert_eq!(app.drawn_basemap(), *app.basemap());
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.contains("hidden")
                && status.contains("basemap service still draws")),
        "the status must not claim a change the user cannot see: {:?}",
        app.status,
    );
}

#[test]
fn an_unpromotable_layer_is_refused_and_records_nothing() {
    let mut app = OxigisApp::new();
    let cog = app.project.layers.add(Layer::new(
        "scene.tif",
        LayerKind::Raster(RasterSource::Cog {
            url: "https://example.org/scene.tif".to_string(),
        }),
    ));
    let rotating = add_xyz(
        &mut app,
        "Rotating",
        "https://{s}.example.org/{z}/{x}/{y}.png",
        "",
    );
    for id in [cog, rotating] {
        app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
        assert_eq!(app.project().basemap_layer, None);
        assert_eq!(app.undo.depth(), (0, 0), "nothing was recorded");
        assert!(
            app.status
                .as_deref()
                .is_some_and(|status| status.starts_with("Basemap not changed:")),
            "the refusal names the reason: {:?}",
            app.status,
        );
    }
}

#[test]
fn promoting_over_a_promotion_is_one_entry_and_one_ctrl_z() {
    let mut app = OxigisApp::new();
    let first = add_xyz(&mut app, "A", "https://a.example/{z}/{x}/{y}.png", "");
    let second = add_xyz(&mut app, "B", "https://b.example/{z}/{x}/{y}.png", "");
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(first)));
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(second)));
    assert_eq!(app.undo.depth(), (2, 0), "one entry per promotion");
    assert!(app.draws_as_basemap(second));

    assert!(app.undo_once());
    assert!(
        app.draws_as_basemap(first),
        "ONE Ctrl+Z goes back to A, not to the service"
    );
}

#[test]
fn removing_a_promoted_layer_falls_back_and_one_undo_restores_both() {
    let mut app = OxigisApp::new();
    // Deliberately NOT the demo OSM layer: that one describes the very
    // service the default basemap already draws, so removing it changes no
    // plan at all and the Remove status is right to say so.
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "\u{a9} Example",
    );
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    assert!(app.draws_as_basemap(id));

    app.apply_layer_action(LayerAction::Remove(id));
    assert_eq!(app.drawn_basemap(), *app.basemap(), "the service draws");
    assert_eq!(
        app.project().basemap_layer,
        Some(id),
        "the pointer is kept: removing is not demoting, and the undo has to \
         be able to bring the promotion back"
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.contains("the map stops drawing it")),
        "a promoted layer IS drawn, so its removal says so: {:?}",
        app.status,
    );

    assert!(app.undo_once());
    assert!(
        app.draws_as_basemap(id),
        "ONE Ctrl+Z restores the layer AND the basemap"
    );
}

#[test]
fn picking_a_preset_while_promoted_is_one_entry_that_restores_both() {
    // SUPERSEDED EXPECTATION, quoted so the flip is not silent. Until editing
    // v1.5 this test was `picking_a_preset_while_promoted_records_the_demote`
    // and asserted, after the undo: "Ctrl+Z re-promotes the layer and leaves
    // the newly picked service underneath \u{2014} exactly what was recorded"
    // and "the undo must not revert a service change that was never
    // recorded". Both were sound only while the service stayed off the stack.
    // It is now ON the stack, so ONE click records ONE entry carrying BOTH
    // halves and one Ctrl+Z walks both back. The gesture is one gesture;
    // costing it two presses would break the one-gesture-one-entry rule
    // `AddLayers` already enforces.
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "\u{a9} Example",
    );
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    let promoted_depth = app.undo.depth();
    let service_before = app.basemap().clone();

    let preset = crate::tile_provider::BASEMAP_PRESETS
        .iter()
        .find(|preset| !preset.matches(app.basemap()))
        .expect("a preset that is not the active one");
    app.apply_layer_action(LayerAction::SetBasemapPreset(preset.config()));
    assert_eq!(
        app.undo.depth(),
        (promoted_depth.0 + 1, 0),
        "the demote and the service change are ONE entry, not two"
    );
    assert_eq!(app.drawn_basemap(), preset.config());
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.ends_with("Ctrl+Z restores the previous basemap.")),
        "and the pick says so: {:?}",
        app.status,
    );

    assert!(app.undo_once());
    assert!(app.draws_as_basemap(id), "ONE Ctrl+Z re-promotes the layer");
    assert_eq!(
        app.basemap(),
        &service_before,
        "\u{2014} and puts the service back under it, in the same press"
    );
}

#[test]
fn re_picking_the_active_preset_while_promoted_still_demotes() {
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "",
    );
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    let active = crate::tile_provider::BASEMAP_PRESETS
        .iter()
        .find(|preset| preset.matches(app.basemap()))
        .expect("the default basemap IS a preset");

    app.apply_layer_action(LayerAction::SetBasemapPreset(active.config()));
    assert_eq!(
        app.project().basemap_layer,
        None,
        "the demote sits above the \u{201c}already active\u{201d} early \
         return, or clicking the highlighted entry would do nothing"
    );
    assert_eq!(app.drawn_basemap(), active.config());
}

#[test]
fn a_refused_service_pick_leaves_the_promotion_alone() {
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "",
    );
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    let depth = app.undo.depth();

    app.apply_layer_action(LayerAction::SetXyzBasemap(
        "https://example.org/no/placeholders.png".to_string(),
    ));
    assert!(app.draws_as_basemap(id), "nothing was asked for that works");
    assert_eq!(app.undo.depth(), depth);
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.starts_with("Basemap not changed:")),
        "{:?}",
        app.status,
    );
}

#[test]
fn the_applier_refuses_a_promotion_of_a_layer_that_is_gone() {
    // Check-first, existence only. A hand-edited pointer at a COG must NOT
    // make an undo refuse — promotability is derived — but a pointer at a
    // layer that is not there cannot be written.
    let mut app = OxigisApp::new();
    let cog = app.project.layers.add(Layer::new(
        "scene.tif",
        LayerKind::Raster(RasterSource::Cog {
            url: "https://example.org/scene.tif".to_string(),
        }),
    ));
    let transaction = crate::edit::project_op::ProjectTransaction {
        label: "Set basemap",
        op: crate::edit::project_op::ProjectOp::SetBasemap {
            before: None,
            after: Some(cog),
            service: None,
        },
        coalesce: None,
    };
    assert_eq!(
        app.apply_project_transaction(&transaction),
        Ok(()),
        "a COG pointer applies and simply fails to resolve"
    );
    assert_eq!(app.project().basemap_layer, Some(cog));
    assert_eq!(app.drawn_basemap(), *app.basemap());

    let ghost = crate::edit::project_op::ProjectTransaction {
        label: "Set basemap",
        op: crate::edit::project_op::ProjectOp::SetBasemap {
            before: None,
            after: Some(LayerId::from_raw(999_999)),
            service: None,
        },
        coalesce: None,
    };
    assert!(app.apply_project_transaction(&ghost).is_err());
}

#[test]
fn a_hydrate_prunes_an_entry_that_only_mentions_the_layer_as_a_basemap() {
    // `mentions_layer` checks BOTH sides, so a hydrate of the promoted layer
    // drops the promote entry rather than letting a redo splice a stale
    // pointer back in.
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
    let promote = crate::edit::project_op::ProjectTransaction {
        label: "Set basemap",
        op: crate::edit::project_op::ProjectOp::SetBasemap {
            before: Some(id),
            after: None,
            service: None,
        },
        coalesce: None,
    };
    app.record_undo(promote);
    let before = app.undo.depth().0;
    assert!(before >= 1);

    assert!(app.hydrate_geojson_layer_from_bytes(
        id,
        "points",
        br#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},
             "geometry":{"type":"Point","coordinates":[1.0,1.0]}}]}"#,
    ));
    assert_eq!(
        app.undo.depth(),
        (0, 0),
        "every entry mentioning the layer went with its data"
    );
}

#[test]
fn adding_then_promoting_undoes_in_lifo_order() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddDemoXyzLayer);
    let id = app.selection().expect("the demo layer is selected");
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    assert_eq!(app.undo.depth(), (2, 0));

    assert!(app.undo_once());
    assert_eq!(
        app.project().basemap_layer,
        None,
        "the promotion is undone first"
    );
    assert_eq!(app.project().layers.len(), 1, "the layer is still there");
    assert!(app.undo_once());
    assert!(app.project().layers.is_empty(), "then the add");
}

#[test]
fn file_new_and_a_load_replan_a_promotion_exactly_once() {
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "\u{a9} Example",
    );
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    let work = app.pending_raster_work().expect("the promotion replans");
    app.settle_raster_work(work, Ok(()));
    assert!(app.pending_raster_work().is_none());

    // File ▸ New drops the promotion with the project, so the service is
    // planned again — once.
    app.new_project();
    let work = app
        .pending_raster_work()
        .expect("a fresh project draws the default service");
    assert_eq!(work.basemap, BasemapConfig::default());
    app.settle_raster_work(work, Ok(()));
    assert!(
        app.pending_raster_work().is_none(),
        "the reconciliation mirror survives File \u{25b8} New"
    );

    // And a load of a project WITH a promotion offers that plan, once.
    let mut saver = OxigisApp::new();
    let saved = add_xyz(
        &mut saver,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "\u{a9} Example",
    );
    saver.apply_layer_action(LayerAction::SetBasemapLayer(Some(saved)));
    saver.sync_project_view();
    let json = saver.project().to_json_string().expect("serialize");
    app.load_project(Project::from_json_string(&json).expect("deserialize"));
    let work = app.pending_raster_work().expect("the loaded promotion");
    assert_eq!(
        work.basemap,
        xyz_config(
            "https://tiles.example.org/{z}/{x}/{y}.png",
            "\u{a9} Example"
        ),
    );
    app.settle_raster_work(work, Ok(()));
    assert!(app.pending_raster_work().is_none());
}

/// One big square — the vertex-handle fixture the edit suites also use.
pub(super) const SQUARE: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"plot"},
     "geometry":{"type":"Polygon","coordinates":[[
        [-10.0,-10.0],[10.0,-10.0],[10.0,10.0],[-10.0,10.0],[-10.0,-10.0]]]}}]}"#;

/// Drives one whole `ui` frame at a realistic window size.
pub(super) fn run_frame(app: &mut OxigisApp, ctx: &egui::Context, events: Vec<egui::Event>) {
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
fn undoing_a_promotion_mid_drag_cancels_the_drag_and_says_both() {
    let ctx = egui::Context::default();
    let mut app = OxigisApp::new();
    app.add_geojson_layer_from_text("plot", SQUARE, None);
    let plot = app.selection().expect("the square is selected");
    let promoted = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "",
    );
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(promoted)));
    app.selection = Some(plot);
    app.map_panel = crate::map_view::MapPanelState::new(0.0, 0.0, 3.0, [1024.0, 768.0]);
    app.edit.set_mode(crate::edit::EditMode::Select);
    let _notice = app.edit.retarget(Some(plot));
    app.edit
        .set_selection(Some(crate::edit::EditSelection::feature(0)));
    run_frame(&mut app, &ctx, Vec::new());
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
    run_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(start), button(true, start)],
    );
    run_frame(
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
    assert_eq!(app.project().basemap_layer, None);
}

#[test]
fn the_print_snapshot_draws_the_promoted_basemap() {
    let mut app = OxigisApp::new();
    let id = add_xyz(
        &mut app,
        "Carto",
        "https://tiles.example.org/{z}/{x}/{y}.png",
        "\u{a9} Example",
    );
    app.project.basemap_layer = Some(id);
    app.request_print();
    let request = app.take_pending_print().expect("an export must be queued");
    assert_eq!(
        request.basemap,
        xyz_config(
            "https://tiles.example.org/{z}/{x}/{y}.png",
            "\u{a9} Example"
        ),
        "the page reads the same derivation the screen does"
    );
    assert_eq!(request.attribution, "\u{a9} Example");
}
