//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::layer_panel::LayerAction;
use crate::local_input::LocalLayerOp;
use crate::style_panel::{self, StyleAction};
use crate::tile_provider::BasemapConfig;
use crate::vector_provider::VectorTileConfig;
use egui::Context;
use oxigis_core::{Project, View};
use oxigis_render::LonLat;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::data_io::{LoadedProject, load_project_text};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use oxigis_core::{LayerKind, RasterSource};
    #[test]
    fn new_app_starts_with_no_layers_and_no_selection() {
        let app = OxigisApp::new();
        assert!(app.project().layers.is_empty());
        assert_eq!(app.selection(), None);
    }
    #[test]
    fn new_app_starting_camera_matches_the_default_project_view() {
        let app = OxigisApp::new();
        let default_view = oxigis_core::View::default();
        assert_eq!(app.map_view().center(), LonLat::new(0.0, 0.0));
        assert!((app.map_view().zoom() - default_view.zoom).abs() < 1e-9);
    }
    #[test]
    fn map_rect_is_none_before_the_first_ui_frame() {
        let app = OxigisApp::new();
        assert_eq!(app.map_rect(), None);
    }
    #[test]
    fn ui_frame_records_the_central_panel_rect() {
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        let rect = app
            .map_rect()
            .expect("central rect must be recorded after a frame");
        assert!(rect.width() > 0.0 && rect.height() > 0.0);
    }
    /// Drives one whole `ui` frame at a realistic window size and returns the
    /// app, so a panel's real draw path (inside its `Panel`, with real widths)
    /// is exercised and not just its widgets in isolation.
    fn run_one_frame(app: &mut OxigisApp) {
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
    fn a_full_frame_draws_the_selected_local_layers_attributes() {
        let mut app = OxigisApp::new();
        assert!(app.show_table, "the table panel is open by default");
        run_one_frame(&mut app);
        assert_eq!(app.table_panel.bound_layer(), None);
        let id = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid");
        run_one_frame(&mut app);
        assert_eq!(app.table_panel.bound_layer(), Some(id));
        assert_eq!(app.table_panel.row_count(), 2);
        assert_eq!(app.table_panel.column_count(), 3);
        app.apply_layer_action(LayerAction::AddDemoXyzLayer);
        run_one_frame(&mut app);
        assert_eq!(app.table_panel.bound_layer(), None);
        assert_eq!(app.table_panel.selected_feature(), None);
    }
    #[test]
    fn a_headless_context_selects_the_fallback_painter() {
        let mut app = OxigisApp::new();
        app.attach_gpu_map(None);
        assert!(!app.map_gpu());
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert!(app.map_rect().is_some());
    }
    #[test]
    fn the_default_basemap_is_openstreetmap_and_is_replaceable() {
        let mut app = OxigisApp::new();
        assert_eq!(app.basemap(), &BasemapConfig::openstreetmap());
        assert!(!app.basemap().attribution.is_empty());
        app.set_basemap(BasemapConfig {
            url_template: "https://example.test/{z}/{x}/{y}.png".to_string(),
            subdomains: Vec::new(),
            attribution: "Example".to_string(),
        });
        assert_eq!(app.basemap().attribution, "Example");
    }
    #[test]
    fn attach_gpu_map_with_never_builds_a_provider_without_a_render_state() {
        let mut app = OxigisApp::new();
        let mut called = false;
        app.attach_gpu_map_with(None, || {
            called = true;
            None
        });
        assert!(!called, "no render state means no provider construction");
        assert!(!app.map_gpu());
    }
    #[test]
    fn a_frame_paints_the_attribution_without_panicking() {
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(400.0, 300.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input.clone(), |ui| app.ui(ui));
        app.set_basemap(BasemapConfig {
            url_template: OxigisApp::new().basemap().url_template.clone(),
            subdomains: Vec::new(),
            attribution: String::new(),
        });
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
    }
    #[test]
    fn apply_layer_action_add_demo_selects_the_new_layer() {
        let mut app = OxigisApp::new();
        app.apply_layer_action(LayerAction::AddDemoXyzLayer);
        assert_eq!(app.project().layers.len(), 1);
        let selected = app.selection().expect("demo layer must be selected");
        assert!(app.project().layers.get(selected).is_some());
    }
    #[test]
    fn the_print_snapshot_carries_the_dialog_options() {
        let mut app = OxigisApp::new();
        // Struct-UPDATE form on purpose, not an exhaustive literal: a new
        // `PrintOptions` field must not break this test, which is about the
        // snapshot transport rather than about any one option. Every field is
        // still compared by the assert below — the ones named here carry a
        // deliberately non-default value so a snapshot that quietly shipped
        // `PrintOptions::default()` could not pass.
        app.print_options = crate::print::PrintOptions {
            page: crate::print::PageSize::A3,
            orientation: crate::print::PageOrientation::Portrait,
            vertical_title: true,
            ..crate::print::PrintOptions::default()
        };
        app.print_options.raster_px_per_pt = 3.0;
        app.request_print();
        let request = app.take_pending_print().expect("an export must be queued");
        assert_eq!(request.options, app.print_options);
    }
    #[test]
    fn sync_project_view_stamps_the_active_basemap_into_the_project() {
        let mut app = OxigisApp::new();
        assert!(app.project().basemap.is_none());
        app.sync_project_view();
        assert_eq!(
            app.project().basemap,
            Some((&BasemapConfig::default()).into()),
            "File ▸ Save must capture the basemap the map is actually drawing"
        );
    }
    #[test]
    fn every_layer_credit_is_shown_together() {
        let mut app = OxigisApp::new();
        app.apply_layer_action(LayerAction::AddCogLayer(
            "https://example.test/scene.tif".to_string(),
        ));
        app.apply_layer_action(LayerAction::AddVectorTileLayer(
            VectorTileConfig::maplibre_demo().url_template,
        ));
        assert!(!app.basemap().attribution.is_empty());
        assert!(!app.vector_attribution().is_empty());
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
    }
    #[test]
    fn apply_layer_action_remove_clears_selection_and_style() {
        // A LOCAL layer: `project.styles` is what draws one, so it is the only
        // kind a style action may write (see the tiled-layer test below). Both
        // of these used to use the demo XYZ layer, whose style entry nothing
        // ever read.
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities.geojson", POINTS, None)
            .expect("valid GeoJSON must be accepted");
        app.apply_style_action(StyleAction::Create(style_panel::StyleKind::Fill));
        assert!(app.project().styles.contains_key(&id));
        app.apply_layer_action(LayerAction::Remove(id));
        assert_eq!(app.selection(), None);
        assert!(app.project().layers.get(id).is_none());
        assert!(!app.project().styles.contains_key(&id));
    }
    #[test]
    fn apply_style_action_create_and_remove_round_trips() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities.geojson", POINTS, None)
            .expect("valid GeoJSON must be accepted");
        app.apply_style_action(StyleAction::Create(style_panel::StyleKind::Circle));
        assert!(app.project().styles.contains_key(&id));
        app.apply_style_action(StyleAction::Remove);
        assert!(!app.project().styles.contains_key(&id));
    }
    #[test]
    fn a_style_action_on_a_tiled_layer_writes_nothing_at_all() {
        // The editor is not drawn for these (see `OxigisApp::ui`), and the
        // applier refuses them too: an MVT / XYZ / COG layer draws from its own
        // source rules, so a `project.styles` entry for one changes nothing on
        // screen, is saved into the file anyway, and — since `sync_local_style`
        // returns early for a non-local layer — cannot be undone.
        for action in [
            LayerAction::AddDemoXyzLayer,
            LayerAction::AddCogLayer("https://example.test/scene.tif".to_string()),
            LayerAction::AddVectorTileLayer(VectorTileConfig::maplibre_demo().url_template),
        ] {
            let mut app = OxigisApp::new();
            app.apply_layer_action(action);
            let id = app.selection().expect("the add seam selects its layer");
            for style_action in [
                StyleAction::Create(style_panel::StyleKind::Fill),
                StyleAction::Remove,
            ] {
                app.apply_style_action(style_action);
                assert!(
                    !app.project().styles.contains_key(&id),
                    "a tiled layer must never gain a style entry"
                );
            }
        }
    }
    #[test]
    fn sync_project_view_captures_the_live_camera() {
        let mut app = OxigisApp::new();
        let panned = app
            .map_view()
            .with_center(LonLat::new(139.767_1, 35.681_2))
            .with_zoom(11.0);
        app.map_panel.set_view(panned);
        app.sync_project_view();
        assert!((app.project().view.center_lon - 139.767_1).abs() < 1e-9);
        assert!((app.project().view.center_lat - 35.681_2).abs() < 1e-9);
        assert!((app.project().view.zoom - 11.0).abs() < 1e-9);
    }
    #[test]
    fn save_then_load_round_trips_the_camera_position() {
        let mut app = OxigisApp::new();
        let panned = app
            .map_view()
            .with_center(LonLat::new(2.349_0, 48.864_7))
            .with_zoom(8.5);
        app.map_panel.set_view(panned);
        app.sync_project_view();
        let json = app
            .project()
            .to_json_string()
            .expect("serialization must succeed for a valid project");
        let mut reloaded = OxigisApp::new();
        let loaded = Project::from_json_string(&json).expect("must parse what we just wrote");
        reloaded.load_project(loaded);
        let camera = reloaded.map_view();
        assert!((camera.center().lon - 2.349_0).abs() < 1e-6);
        assert!((camera.center().lat - 48.864_7).abs() < 1e-6);
        assert!((camera.zoom() - 8.5).abs() < 1e-6);
    }
    #[test]
    fn load_project_replaces_project_and_clears_selection() {
        let mut app = OxigisApp::new();
        app.apply_layer_action(LayerAction::AddDemoXyzLayer);
        assert!(app.selection().is_some());
        let mut fresh = Project::new("Reloaded");
        fresh.layers.add(oxigis_core::Layer::new(
            "Reloaded raster",
            LayerKind::Raster(RasterSource::xyz("https://example.test/{z}/{x}/{y}.png")),
        ));
        app.load_project(fresh);
        assert_eq!(app.project().name, "Reloaded");
        assert_eq!(app.project().layers.len(), 1);
        assert_eq!(app.selection(), None);
    }
    #[test]
    fn load_project_text_uses_the_native_format_when_it_parses() {
        let project = Project::new("Native Round Trip");
        let json = project.to_json_string().expect("serialize");
        match load_project_text(&json).expect("must load") {
            LoadedProject::Native(loaded) => assert_eq!(loaded.name, "Native Round Trip"),
            LoadedProject::GeoLibre(..) => {
                panic!("a valid .oxigis.json must never fall through to GeoLibre import")
            }
        }
    }
    #[test]
    fn load_project_text_imports_a_geolibre_document_after_the_native_parse_fails() {
        let text = r#"{
            "version": "0.2.0",
            "name": "GeoLibre Map",
            "mapView": {"center": [10.0, 20.0], "zoom": 3.0},
            "layers": [],
            "styles": {}
        }"#;
        match load_project_text(text).expect("must load") {
            LoadedProject::GeoLibre(project, notices) => {
                assert_eq!(project.name, "GeoLibre Map");
                assert!(notices.is_empty(), "{notices:?}");
            }
            LoadedProject::Native(_) => {
                panic!("a GeoLibre document must not parse as our own format")
            }
        }
    }
    #[test]
    fn load_project_text_surfaces_the_original_error_when_nothing_sniffs_as_geolibre() {
        let broken = "{ this is not json";
        let result = load_project_text(broken);
        let direct = Project::from_json_string(broken).expect_err("still not valid JSON");
        assert_eq!(result.expect_err("must fail"), format!("{direct:?}"));
    }
    #[test]
    fn opening_a_geolibre_document_loads_it_through_the_same_status_line_aggregation() {
        let mut app = OxigisApp::new();
        let text = r#"{
            "version": "0.2.0",
            "name": "Imported",
            "mapView": {"center": [1.0, 2.0], "zoom": 4.0},
            "basemapStyleUrl": "https://tiles.openfreemap.org/styles/liberty",
            "layers": [{"id": "l1", "name": "Terrain", "type": "wms"}],
            "styles": {}
        }"#;
        match load_project_text(text).expect("must load") {
            LoadedProject::GeoLibre(project, notices) => {
                app.load_geolibre_project(project, notices);
            }
            LoadedProject::Native(_) => panic!("unexpected native parse"),
        }
        assert_eq!(app.project().name, "Imported");
        assert!(
            app.status()
                .is_some_and(|line| line.contains("problems") && line.contains("basemap")),
            "status was: {:?}",
            app.status()
        );
    }
    /// Two points, Tokyo and Osaka — a real, non-degenerate bbox.
    const POINTS: &str = r#"{"type":"FeatureCollection","features":[
        {"type":"Feature","properties":{"name":"Tokyo"},
         "geometry":{"type":"Point","coordinates":[139.767,35.681]}},
        {"type":"Feature","properties":{"name":"Osaka"},
         "geometry":{"type":"Point","coordinates":[135.502,34.702]}}]}"#;
    #[test]
    fn adding_geojson_text_populates_the_project_queues_the_layer_and_zooms() {
        let mut app = OxigisApp::new();
        let before = app.map_view();
        let id = app
            .add_geojson_layer_from_text("cities.geojson", POINTS, None)
            .expect("valid GeoJSON must be accepted");
        assert_eq!(app.project().layers.len(), 1);
        assert_eq!(app.selection(), Some(id));
        assert!(app.project().styles.contains_key(&id));
        assert!(app.status().is_some_and(|line| line.contains("2 features")));
        let after = app.map_view();
        assert!(
            after.zoom() > before.zoom(),
            "the map must zoom to the layer"
        );
        assert!(after.zoom().is_finite());
        assert!((after.center().lon - 137.6).abs() < 1.0);
        assert!((after.center().lat - 35.2).abs() < 1.0);
        let ops = app.take_pending_local_ops();
        assert_eq!(ops.len(), 1);
        assert!(matches!(& ops[0], LocalLayerOp::Add(other, _) if * other == id));
        assert!(app.take_pending_local_ops().is_empty(), "drained once");
        app.apply_layer_action(LayerAction::Remove(id));
        assert!(app.project().layers.is_empty());
        assert!(!app.project().styles.contains_key(&id));
        assert_eq!(app.selection(), None);
        let ops = app.take_pending_local_ops();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], LocalLayerOp::Remove(other) if other == id));
    }
    #[test]
    fn a_single_point_layer_still_yields_a_finite_camera() {
        let mut app = OxigisApp::new();
        let point = r#"{"type":"FeatureCollection","features":[{"type":"Feature",
            "properties":{},"geometry":{"type":"Point","coordinates":[2.349,48.864]}}]}"#;
        assert!(
            app.add_geojson_layer_from_text("point.geojson", point, None)
                .is_some()
        );
        let view = app.map_view();
        assert!(view.zoom().is_finite());
        assert!(view.center().lon.is_finite() && view.center().lat.is_finite());
        assert!((view.center().lon - 2.349).abs() < 0.1);
    }
    #[test]
    fn broken_geojson_is_refused_with_a_status_line_and_no_layer() {
        let mut app = OxigisApp::new();
        assert!(
            app.add_geojson_layer_from_text("bad.geojson", "{", None)
                .is_none()
        );
        assert!(app.project().layers.is_empty());
        assert!(app.status().is_some());
        assert!(app.take_pending_local_ops().is_empty());
    }
    #[test]
    fn non_utf8_bytes_are_refused_rather_than_lossily_decoded() {
        let mut app = OxigisApp::new();
        assert!(
            app.add_geojson_layer_from_bytes("blob.geojson", &[0xff, 0xfe, 0x00], None)
                .is_none()
        );
        assert!(app.project().layers.is_empty());
    }
    #[test]
    fn a_web_style_drop_carrying_bytes_is_parsed_during_the_frame() {
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            dropped_files: vec![egui::DroppedFile {
                name: "cities.geojson".to_string(),
                bytes: Some(POINTS.as_bytes().into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert_eq!(app.project().layers.len(), 1);
        assert_eq!(app.take_pending_local_ops().len(), 1);
    }
    #[test]
    fn a_dropped_file_of_the_wrong_kind_is_refused_by_name() {
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            dropped_files: vec![egui::DroppedFile {
                name: "scene.tif".to_string(),
                bytes: Some(vec![0u8; 4].into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert!(app.project().layers.is_empty());
        assert!(
            app.status()
                .is_some_and(|line| line.contains("not a supported file type"))
        );
    }
    #[test]
    fn a_web_style_geolibre_project_drop_replaces_the_current_project() {
        let mut app = OxigisApp::new();
        app.apply_layer_action(LayerAction::AddDemoXyzLayer);
        assert_eq!(app.project().layers.len(), 1);

        let text = br#"{
            "version": "0.2.0", "name": "Dropped",
            "mapView": {"center": [5.0, 6.0], "zoom": 2.0},
            "layers": []
        }"#;
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            dropped_files: vec![egui::DroppedFile {
                name: "map.geolibre.json".to_string(),
                bytes: Some(text.as_slice().into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert_eq!(app.project().name, "Dropped");
        assert!(app.project().layers.is_empty());
    }
    #[test]
    fn an_ambiguous_json_drop_that_is_really_geolibre_hints_at_file_open() {
        let mut app = OxigisApp::new();
        let text = br#"{
            "version": "0.2.0", "name": "X",
            "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": []
        }"#;
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            dropped_files: vec![egui::DroppedFile {
                // Not `.geolibre.json`: an ambiguous plain `.json` name must
                // never be auto-imported as a project, only hinted about.
                name: "map.json".to_string(),
                bytes: Some(text.as_slice().into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert!(app.project().layers.is_empty());
        assert!(
            app.status().is_some_and(
                |line| line.contains("GeoLibre project") && line.contains("File > Open")
            ),
            "status was: {:?}",
            app.status()
        );
    }
    #[test]
    fn a_native_geolibre_project_drop_queues_its_path_with_no_layer() {
        // Mirrors `a_native_geopackage_drop_queues_its_path_with_no_table`:
        // a fresh drop (as opposed to a project-load path reference) is
        // queued with `layer: None`, which is what tells a shell's
        // `classify_drop`-based dispatch this is a whole-project read, not a
        // single layer's bytes.
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            dropped_files: vec![egui::DroppedFile {
                path: Some(std::path::PathBuf::from("/data/map.geolibre.json")),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert!(app.project().layers.is_empty(), "nothing is parsed here");
        let pending = app.take_pending_dropped_paths();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].layer, None);
        assert_eq!(
            pending[0].path,
            std::path::PathBuf::from("/data/map.geolibre.json")
        );
        assert!(app.status().is_some_and(|line| line.contains("Reading")));
    }
    #[test]
    fn load_geolibre_project_from_bytes_is_the_native_drop_read_seam() {
        // What a native shell calls after reading the path queued above.
        let mut app = OxigisApp::new();
        app.apply_layer_action(LayerAction::AddDemoXyzLayer);
        let text = br#"{
            "version": "0.2.0", "name": "Read Natively",
            "mapView": {"center": [7.0, 8.0], "zoom": 3.0},
            "layers": []
        }"#;
        app.load_geolibre_project_from_bytes("map.geolibre.json", text);
        assert_eq!(app.project().name, "Read Natively");
        assert!(app.project().layers.is_empty());
    }
    #[test]
    fn a_web_style_multi_file_shapefile_drop_becomes_one_layer_with_attributes() {
        // The §1.3 headline: five separate `DroppedFile`s in one drop, in the
        // order a browser happens to hand them over, must become exactly one
        // layer — with its .dbf columns readable by the attribute table.
        let (shp, dbf) = crate::shapefile_input::fixture::cities();
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            dropped_files: vec![
                egui::DroppedFile {
                    name: "cities.dbf".to_string(),
                    bytes: Some(dbf.clone().into()),
                    ..Default::default()
                },
                egui::DroppedFile {
                    name: "cities.shp".to_string(),
                    bytes: Some(shp.clone().into()),
                    ..Default::default()
                },
                egui::DroppedFile {
                    name: "cities.prj".to_string(),
                    bytes: Some(br#"GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984"]]"#.to_vec().into()),
                    ..Default::default()
                },
                egui::DroppedFile {
                    name: "cities.shx".to_string(),
                    bytes: Some(vec![0u8; 8].into()),
                    ..Default::default()
                },
                egui::DroppedFile {
                    name: "cities.cpg".to_string(),
                    bytes: Some(b"UTF-8".to_vec().into()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));

        assert_eq!(app.project().layers.len(), 1, "one layer, not five");
        assert_eq!(app.take_pending_local_ops().len(), 1);
        assert!(app.take_pending_dropped_paths().is_empty());
        let (id, name, features) = app.selected_table_source().expect("the table has a source");
        assert_eq!(name, "cities.shp");
        assert_eq!(features.features.len(), 2);
        let row = features.features[0].properties.as_ref().expect("row");
        assert_eq!(row["NAME"], serde_json::json!("Tokyo"));
        // §1.2's attribute table binds the shapefile in that same frame and
        // makes its .dbf columns ordinary columns: `#`, geometry, NAME, POP.
        assert_eq!(app.table_panel.bound_layer(), Some(id));
        assert_eq!(app.table_panel.row_count(), 2);
        assert_eq!(app.table_panel.column_count(), 4);
        // The camera moved to the data (Tokyo/Osaka), as for a GeoJSON drop.
        assert!(app.map_view().center().lon > 130.0);
    }
    #[test]
    fn one_drop_of_a_geojson_and_a_shapefile_set_adds_both_layers() {
        let (shp, dbf) = crate::shapefile_input::fixture::cities();
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            dropped_files: vec![
                egui::DroppedFile {
                    name: "borders.geojson".to_string(),
                    bytes: Some(POINTS.as_bytes().into()),
                    ..Default::default()
                },
                egui::DroppedFile {
                    name: "cities.dbf".to_string(),
                    bytes: Some(dbf.into()),
                    ..Default::default()
                },
                egui::DroppedFile {
                    name: "cities.shp".to_string(),
                    bytes: Some(shp.into()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert_eq!(
            app.project().layers.len(),
            2,
            "one per dataset, not per file"
        );
        let names: Vec<&str> = app
            .project()
            .layers
            .layers()
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, vec!["borders.geojson", "cities.shp"]);
        assert_eq!(app.take_pending_local_ops().len(), 2);
    }
    #[test]
    fn a_native_shapefile_drop_queues_only_the_shp_path() {
        // On the desktop the bytes never arrive; the shell reads the `.shp` and
        // finds the siblings itself, so only one path may be queued.
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            dropped_files: vec![
                egui::DroppedFile {
                    path: Some(std::path::PathBuf::from("/data/cities.shp")),
                    ..Default::default()
                },
                egui::DroppedFile {
                    path: Some(std::path::PathBuf::from("/data/cities.dbf")),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert!(app.project().layers.is_empty(), "nothing is parsed here");
        let pending = app.take_pending_dropped_paths();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].layer, None);
        assert_eq!(
            pending[0].path,
            std::path::PathBuf::from("/data/cities.shp")
        );
    }
    #[test]
    fn a_lone_dbf_drop_is_ignored_with_a_status_line() {
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            dropped_files: vec![egui::DroppedFile {
                name: "cities.dbf".to_string(),
                bytes: Some(vec![0u8; 16].into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert!(app.project().layers.is_empty());
        assert!(
            app.status()
                .is_some_and(|line| line.contains("without its .shp"))
        );
    }
    #[test]
    fn a_shapefile_layer_reloaded_from_a_project_is_rebuilt_in_place() {
        let (shp, dbf) = crate::shapefile_input::fixture::cities();
        let mut app = OxigisApp::new();
        assert!(
            app.add_shapefile_layer_from_bytes(
                "cities.shp",
                crate::shapefile_input::ShapefileBytes::new(&shp).with_dbf(Some(&dbf)),
                Some("/data/cities.shp"),
            )
            .is_some()
        );
        app.sync_project_view();
        let json = app.project().to_json_string().expect("serialize");
        let _ = app.take_pending_local_ops();

        let loaded = Project::from_json_string(&json).expect("parse");
        app.load_project(loaded);
        let pending = app.take_pending_dropped_paths();
        assert_eq!(pending.len(), 1);
        let id = pending[0]
            .layer
            .expect("a project reference names its layer");
        assert_eq!(
            pending[0].path,
            std::path::PathBuf::from("/data/cities.shp")
        );
        let _ = app.take_pending_local_ops();

        assert!(app.hydrate_shapefile_layer_from_bytes(
            id,
            "cities.shp",
            crate::shapefile_input::ShapefileBytes::new(&shp).with_dbf(Some(&dbf)),
        ));
        assert_eq!(app.project().layers.len(), 1, "no duplicate layer");
        let ops = app.take_pending_local_ops();
        assert_eq!(ops.len(), 1);
        assert!(matches!(&ops[0], LocalLayerOp::Add(other, _) if * other == id));
    }
    #[test]
    fn a_shapefile_in_an_unsupported_crs_is_refused_with_the_crs_named() {
        let (shp, dbf) = crate::shapefile_input::fixture::cities();
        let mut app = OxigisApp::new();
        // Krovak: a real, named projection this build has no inverse for, so
        // the reader refuses the set rather than putting the data somewhere
        // wrong. It replaced UTM 54N here when `shapefile_input` gained real
        // reprojection through `oxigis_core::crs::Reprojector` — UTM now
        // *loads* (EPSG:32654 -> WGS 84), which is a feature, not a regression,
        // and this test is about the refusal path that survives it.
        let krovak = r#"PROJCS["S-JTSK_Krovak_East_North",GEOGCS["GCS_S_JTSK",DATUM["D_S_JTSK",SPHEROID["Bessel_1841",6377397.155,299.1528128]]],PROJECTION["Krovak"],UNIT["Meter",1.0]]"#;
        assert!(
            app.add_shapefile_layer_from_bytes(
                "cities.shp",
                crate::shapefile_input::ShapefileBytes::new(&shp)
                    .with_dbf(Some(&dbf))
                    .with_sidecars(Some(krovak), None),
                None,
            )
            .is_none()
        );
        assert!(app.project().layers.is_empty());
        assert!(
            app.status()
                .is_some_and(|line| line.contains("S-JTSK_Krovak_East_North")),
            "the refusal has to name the CRS: {:?}",
            app.status(),
        );
    }
    #[test]
    fn a_hovering_file_paints_the_drop_hint_without_panicking() {
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            hovered_files: vec![egui::HoveredFile::default()],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
    }
    #[test]
    fn local_layer_visibility_opacity_and_order_reach_the_shell() {
        let mut app = OxigisApp::new();
        let bottom = app
            .add_geojson_layer_from_text("bottom", POINTS, None)
            .expect("valid");
        let top = app
            .add_geojson_layer_from_text("top", POINTS, None)
            .expect("valid");
        let _ = app.take_pending_local_ops();
        app.apply_layer_action(LayerAction::ToggleVisibility(top));
        app.apply_layer_action(LayerAction::SetOpacity(top, 0.25));
        app.apply_layer_action(LayerAction::MoveUp(bottom));
        let ops = app.take_pending_local_ops();
        assert_eq!(ops.len(), 3);
        assert!(matches!(ops[0], LocalLayerOp::SetVisibility(id, false) if id == top));
        assert!(
            matches!(ops[1], LocalLayerOp::SetOpacity(id, value) if id == top && (value -
            0.25).abs() < 1e-6)
        );
        match &ops[2] {
            LocalLayerOp::Reorder(order) => assert_eq!(order, &vec![top, bottom]),
            other => panic!("expected a reorder, got {other:?}"),
        }
    }
    #[test]
    fn the_feature_store_tracks_the_project_across_add_save_new_and_load() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities.geojson", POINTS, None)
            .expect("valid");
        assert_eq!(app.local.feature_set_count(), 1);
        let (table_id, name, features) = app
            .selected_table_source()
            .expect("the freshly added layer is selected and stored");
        assert_eq!(table_id, id);
        assert_eq!(name, "cities.geojson");
        assert_eq!(features.features.len(), 2);
        assert!(Arc::ptr_eq(
            &features,
            app.local.feature_set(id).expect("stored")
        ));
        let json = app
            .project()
            .to_json_string()
            .expect("a valid project serializes");
        app.new_project();
        assert_eq!(app.local.feature_set_count(), 0);
        assert!(app.local.feature_set(id).is_none());
        assert!(app.selected_table_source().is_none());
        let loaded = Project::from_json_string(&json).expect("must parse what we wrote");
        app.load_project(loaded);
        assert_eq!(app.local.feature_set_count(), 1);
        let reloaded_id = app
            .project()
            .layers
            .layers()
            .first()
            .map(|layer| layer.id)
            .expect("the loaded project has the layer");
        assert_eq!(
            app.local
                .feature_set(reloaded_id)
                .map(|features| features.features.len()),
            Some(2)
        );
        app.apply_layer_action(LayerAction::Remove(reloaded_id));
        assert_eq!(app.local.feature_set_count(), 0);
    }
    #[test]
    fn the_table_shows_a_local_selection_and_a_placeholder_for_anything_else() {
        let mut app = OxigisApp::new();
        let local = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid");
        assert_eq!(
            app.selected_table_source().map(|(id, ..)| id),
            Some(local),
            "a selected local layer feeds the table"
        );
        app.apply_layer_action(LayerAction::AddDemoXyzLayer);
        assert!(app.selected_table_source().is_none());
        app.apply_layer_action(LayerAction::Select(local));
        assert!(app.selected_table_source().is_some());
        app.new_project();
        assert!(app.selected_table_source().is_none());
    }
    #[test]
    fn a_raster_layer_never_queues_local_work() {
        let mut app = OxigisApp::new();
        app.apply_layer_action(LayerAction::AddDemoXyzLayer);
        let id = app.selection().expect("selected");
        app.apply_layer_action(LayerAction::ToggleVisibility(id));
        app.apply_layer_action(LayerAction::SetOpacity(id, 0.5));
        app.apply_layer_action(LayerAction::Remove(id));
        assert!(app.take_pending_local_ops().is_empty());
    }
    #[test]
    fn editing_a_local_layers_style_queues_exactly_one_restyle_per_change() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid");
        let _ = app.take_pending_local_ops();
        let unchanged = app.project().styles.get(&id).cloned();
        app.sync_local_style(unchanged.clone());
        assert!(app.take_pending_local_ops().is_empty());
        app.apply_style_action(StyleAction::Create(style_panel::StyleKind::Symbol));
        app.sync_local_style(unchanged);
        let ops = app.take_pending_local_ops();
        assert_eq!(ops.len(), 1);
        assert!(matches!(&ops[0], LocalLayerOp::SetStyle(other, set)
            if *other == id
                && matches!(set.base(), oxigis_core::LayerStyle::Symbol(_))));
        let before = app.project().styles.get(&id).cloned();
        app.apply_style_action(StyleAction::Remove);
        app.sync_local_style(before);
        let ops = app.take_pending_local_ops();
        assert_eq!(ops.len(), 1);
        assert!(matches!(& ops[0], LocalLayerOp::SetStyle(other, _) if * other == id));
    }
    #[test]
    fn a_project_round_trip_restores_the_local_layer_and_clears_the_old_one() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid");
        app.sync_project_view();
        let json = app.project().to_json_string().expect("serialize");
        let _ = app.take_pending_local_ops();
        let loaded = Project::from_json_string(&json).expect("parse");
        app.load_project(loaded);
        assert_eq!(app.project().layers.len(), 1);
        let ops = app.take_pending_local_ops();
        assert!(matches!(ops.first(), Some(LocalLayerOp::Clear)));
        assert!(matches!(ops.get(1), Some(LocalLayerOp::Add(other, _)) if * other == id));
        assert!(app.take_pending_dropped_paths().is_empty());
    }
    #[test]
    fn a_path_referenced_layer_is_rehydrated_without_duplicating_or_re_aiming_the_camera() {
        let mut app = OxigisApp::new();
        let mut project = Project::new("saved");
        let id = project.layers.add(oxigis_core::Layer::new(
            "cities",
            oxigis_core::LayerKind::Vector(oxigis_core::VectorSource::LocalGeoJson {
                path: "/data/cities.geojson".to_string(),
            }),
        ));
        let saved_style =
            oxigis_core::LayerStyle::Fill(oxigis_core::FillStyle::new(oxigis_core::Color::WHITE));
        project.styles.insert(id, saved_style.clone().into());
        project.view = View {
            center_lon: 2.349,
            center_lat: 48.864,
            zoom: 9.0,
        };
        app.load_project(project);
        let pending = app.take_pending_dropped_paths();
        let _ = app.take_pending_local_ops();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].layer, Some(id));
        assert!(app.hydrate_geojson_layer_from_bytes(id, "cities.geojson", POINTS.as_bytes()));
        assert_eq!(app.project().layers.len(), 1);
        let ops = app.take_pending_local_ops();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            LocalLayerOp::Add(other, local) => {
                assert_eq!(*other, id);
                assert_eq!(local.style().base(), &saved_style);
            }
            other => panic!("expected the rebuilt layer, got {other:?}"),
        }
        assert!((app.map_view().zoom() - 9.0).abs() < 1e-6);
        assert!((app.map_view().center().lon - 2.349).abs() < 1e-6);
    }
    #[test]
    fn a_new_project_detaches_every_local_layer() {
        let mut app = OxigisApp::new();
        assert!(
            app.add_geojson_layer_from_text("cities", POINTS, None)
                .is_some()
        );
        let _ = app.take_pending_local_ops();
        app.new_project();
        assert!(app.project().layers.is_empty());
        let ops = app.take_pending_local_ops();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], LocalLayerOp::Clear));
    }
    #[test]
    fn the_paste_geojson_action_opens_a_modal_and_adds_on_confirm() {
        let mut app = OxigisApp::new();
        app.apply_layer_action(LayerAction::AddGeoJsonPaste);
        assert!(matches!(app.io_dialog, Some(IoDialog::PasteGeoJson { .. })));
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert!(app.io_dialog.is_some());
    }
    #[test]
    fn a_large_inline_paste_warns_about_the_project_file_size() {
        let mut app = OxigisApp::new();
        let padding = "x".repeat(crate::local_input::INLINE_GEOJSON_WARN_BYTES);
        let text = format!(
            r#"{{"type":"FeatureCollection","features":[{{"type":"Feature",
               "properties":{{"pad":"{padding}"}},
               "geometry":{{"type":"Point","coordinates":[0.0,0.0]}}}}]}}"#
        );
        assert!(
            app.add_geojson_layer_from_text("big", &text, None)
                .is_some()
        );
        assert!(app.status().is_some_and(|line| line.contains("large")));
    }
    #[test]
    fn ui_smoke_test_does_not_panic_with_layers_and_style() {
        let mut app = OxigisApp::new();
        app.apply_layer_action(LayerAction::AddDemoXyzLayer);
        app.apply_style_action(StyleAction::Create(style_panel::StyleKind::Fill));
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

    // ---- §1.5 Processing toolbox ------------------------------------------

    /// A bare `ToolDescriptor` with no params, for exercising result routing
    /// directly ([`OxigisApp::route_processing_result`]) without a real
    /// executor.
    fn synthetic_descriptor(id: &str) -> oxigis_core::ToolDescriptor {
        oxigis_core::ToolDescriptor {
            id: id.to_string(),
            title: "Synthetic".to_string(),
            description: String::new(),
            params: Vec::new(),
        }
    }

    #[test]
    fn local_vector_layer_options_excludes_raster_and_tiled_layers() {
        let mut app = OxigisApp::new();
        app.apply_layer_action(LayerAction::AddDemoXyzLayer);
        app.apply_layer_action(LayerAction::AddVectorTileLayer(
            VectorTileConfig::maplibre_demo().url_template,
        ));
        assert!(
            app.local_vector_layer_options().is_empty(),
            "a raster layer and an MVT layer are never local-vector options"
        );
        let bottom = app
            .add_geojson_layer_from_text("bottom", POINTS, None)
            .expect("valid");
        let top = app
            .add_geojson_layer_from_text("top", POINTS, None)
            .expect("valid");
        assert_eq!(
            app.local_vector_layer_options(),
            vec![(top, "top"), (bottom, "bottom")],
            "top-of-stack-first, like the layer panel"
        );
    }

    #[test]
    fn local_vector_layer_options_excludes_an_unhydrated_path_reference() {
        let mut app = OxigisApp::new();
        let mut project = Project::new("test");
        let id = project.layers.add(oxigis_core::Layer::new(
            "cities",
            oxigis_core::LayerKind::Vector(oxigis_core::VectorSource::LocalGeoJson {
                path: "/data/cities.geojson".to_string(),
            }),
        ));
        app.load_project(project);
        assert!(
            app.local_vector_layer_options().is_empty(),
            "still queued for the shell to read, not yet in the feature store"
        );
        assert!(app.hydrate_geojson_layer_from_bytes(id, "cities.geojson", POINTS.as_bytes()));
        assert_eq!(app.local_vector_layer_options(), vec![(id, "cities")]);
    }

    #[test]
    fn run_processing_tool_computes_bounds_for_a_real_layer() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid");
        let descriptor = app
            .processing_registry
            .get("bounds")
            .expect("registered")
            .clone();
        let mut params = BTreeMap::new();
        params.insert("layer".to_string(), serde_json::json!(id.get()));
        let value = app
            .run_processing_tool(&descriptor, params)
            .expect("must succeed");
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
    fn run_processing_tool_counts_features_for_a_real_layer() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid");
        let descriptor = app
            .processing_registry
            .get("feature_count")
            .expect("registered")
            .clone();
        let mut params = BTreeMap::new();
        params.insert("layer".to_string(), serde_json::json!(id.get()));
        let value = app
            .run_processing_tool(&descriptor, params)
            .expect("must succeed");
        assert_eq!(value, serde_json::json!(2));
    }

    #[test]
    fn run_processing_tool_reports_a_removed_layer_without_panicking() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid");
        app.apply_layer_action(LayerAction::Remove(id));
        let descriptor = app
            .processing_registry
            .get("bounds")
            .expect("registered")
            .clone();
        let mut params = BTreeMap::new();
        params.insert("layer".to_string(), serde_json::json!(id.get()));
        let error = app
            .run_processing_tool(&descriptor, params)
            .expect_err("the layer's data is gone");
        assert!(error.contains("not loaded"), "{error}");
    }

    #[test]
    fn run_processing_tool_reports_a_descriptor_with_no_wired_executor() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid");
        let descriptor = oxigis_core::ToolDescriptor {
            id: "not_a_real_tool".to_string(),
            title: "Not Real".to_string(),
            description: String::new(),
            params: vec![oxigis_core::ParamSpec {
                name: "layer".to_string(),
                kind: oxigis_core::ParamKind::LayerRef,
                required: true,
                default: None,
            }],
        };
        let mut params = BTreeMap::new();
        params.insert("layer".to_string(), serde_json::json!(id.get()));
        let error = app
            .run_processing_tool(&descriptor, params)
            .expect_err("no executor is wired for this id");
        assert!(error.contains("is not implemented yet"), "{error}");
    }

    #[test]
    fn run_processing_tool_refuses_a_descriptor_with_no_layer_ref_param() {
        let mut app = OxigisApp::new();
        let error = app
            .run_processing_tool(&synthetic_descriptor("no_layer"), BTreeMap::new())
            .expect_err("zero-layer tools are not yet supported");
        assert!(error.contains("not yet supported"), "{error}");
    }

    #[test]
    fn a_feature_collection_result_becomes_a_new_layer_and_zooms_to_it() {
        let mut app = OxigisApp::new();
        let before = app.map_view();
        let descriptor = synthetic_descriptor("synthetic-fc");
        let value = serde_json::json!({
            "type": "FeatureCollection",
            "features": [
                {"type":"Feature","properties":{},
                 "geometry":{"type":"Point","coordinates":[139.767,35.681]}}
            ]
        });
        app.route_processing_result(&descriptor, Ok(value));
        assert_eq!(app.project().layers.len(), 1);
        assert_eq!(app.project().layers.layers()[0].name, "Synthetic result");
        assert!(app.selection().is_some());
        let after = app.map_view();
        assert!(
            after.zoom() > before.zoom(),
            "a FeatureCollection result must zoom to its layer, like a pasted GeoJSON does"
        );
    }

    #[test]
    fn a_scalar_result_is_reported_verbatim_on_the_status_line() {
        let mut app = OxigisApp::new();
        let descriptor = synthetic_descriptor("synthetic-scalar");
        app.route_processing_result(&descriptor, Ok(serde_json::json!(42)));
        assert!(app.project().layers.is_empty());
        assert_eq!(app.status(), Some("Synthetic: 42"));
    }

    #[test]
    fn a_structured_non_feature_collection_result_gets_a_generic_status_line() {
        let mut app = OxigisApp::new();
        let descriptor = synthetic_descriptor("synthetic-object");
        app.route_processing_result(&descriptor, Ok(serde_json::json!({"count": 2})));
        assert!(app.project().layers.is_empty());
        assert_eq!(app.status(), Some("Synthetic finished."));
    }

    #[test]
    fn an_error_result_is_reported_without_creating_a_layer() {
        let mut app = OxigisApp::new();
        let descriptor = synthetic_descriptor("synthetic-error");
        app.route_processing_result(&descriptor, Err("boom".to_string()));
        assert!(app.project().layers.is_empty());
    }

    #[test]
    fn a_web_style_geopackage_drop_becomes_one_layer_per_feature_table() {
        // The §1.3 GeoPackage headline: one dropped file, several layers, and
        // the tables it could not place reported rather than dropped in
        // silence.
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            dropped_files: vec![egui::DroppedFile {
                name: "basic.gpkg".to_string(),
                bytes: Some(crate::gpkg_input::fixture::BASIC.into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));

        assert_eq!(app.project().layers.len(), 3, "one layer per feature table");
        assert_eq!(app.take_pending_local_ops().len(), 3);
        assert!(app.take_pending_dropped_paths().is_empty());
        let status = app.status().expect("a status line");
        assert!(status.contains("3 layers"), "{status}");
        assert!(
            status.contains("RGF93 / Lambert-93"),
            "the refused table must be reported: {status}",
        );

        // The first table is selected, so the attribute table binds it in the
        // same frame, and the camera moved to it.
        let (id, name, features) = app.selected_table_source().expect("the table has a source");
        assert_eq!(name, "basic:cities");
        assert_eq!(features.features.len(), 4);
        assert_eq!(
            features.features[0].properties.as_ref().expect("row")["name ja"],
            serde_json::json!("\u{6771}\u{4eac}"),
        );
        assert_eq!(app.table_panel.bound_layer(), Some(id));
        assert!(app.map_view().center().lon > 130.0);
    }

    #[test]
    fn a_native_geopackage_drop_queues_its_path_with_no_table() {
        // On the desktop the bytes never arrive; the shell reads the file and
        // imports every table, which is what an empty `table` means.
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            dropped_files: vec![egui::DroppedFile {
                path: Some(std::path::PathBuf::from("/data/basic.gpkg")),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert!(app.project().layers.is_empty(), "nothing is parsed here");
        let pending = app.take_pending_dropped_paths();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].layer, None);
        assert_eq!(pending[0].table, None);
        assert_eq!(
            pending[0].path,
            std::path::PathBuf::from("/data/basic.gpkg")
        );
    }

    #[test]
    fn a_geopackage_layer_reloaded_from_a_project_is_rebuilt_table_by_table() {
        let bytes = crate::gpkg_input::fixture::BASIC;
        let mut app = OxigisApp::new();
        let added = app.add_gpkg_layer_from_bytes("basic.gpkg", bytes, Some("/data/basic.gpkg"));
        assert_eq!(added.len(), 3);
        app.sync_project_view();
        let json = app.project().to_json_string().expect("serialize");
        let _ = app.take_pending_local_ops();

        let loaded = Project::from_json_string(&json).expect("parse");
        app.load_project(loaded);
        let pending = app.take_pending_dropped_paths();
        assert_eq!(pending.len(), 3, "one read per table-backed layer");
        assert_eq!(pending[0].table.as_deref(), Some("cities"));
        assert_eq!(pending[2].table.as_deref(), Some("roads"));
        let id = pending[0]
            .layer
            .expect("a project reference names its layer");
        let _ = app.take_pending_local_ops();

        assert!(app.hydrate_gpkg_layer_from_bytes(id, "basic.gpkg", bytes, "cities"));
        assert_eq!(app.project().layers.len(), 3, "no duplicate layer");
        let ops = app.take_pending_local_ops();
        assert_eq!(ops.len(), 1);
        assert!(matches!(&ops[0], LocalLayerOp::Add(other, _) if * other == id));

        // A table the file no longer holds is reported, not guessed at.
        assert!(!app.hydrate_gpkg_layer_from_bytes(id, "basic.gpkg", bytes, "gone"));
        assert!(app.status().is_some_and(|line| line.contains("gone")));
    }

    #[test]
    fn a_geopackage_whose_tables_cannot_be_read_adds_nothing_and_says_why() {
        let mut app = OxigisApp::new();
        let added = app.add_gpkg_layer_from_bytes(
            "weird.gpkg",
            crate::gpkg_input::fixture::WITHOUT_ROWID,
            None,
        );
        assert!(added.is_empty());
        assert!(app.project().layers.is_empty());
        assert!(
            app.status()
                .is_some_and(|line| line.contains("WITHOUT ROWID"))
        );
    }

    #[test]
    #[cfg(feature = "geoparquet")]
    fn a_native_geoparquet_drop_queues_its_path() {
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            dropped_files: vec![egui::DroppedFile {
                path: Some(std::path::PathBuf::from("/data/cities.parquet")),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert!(app.project().layers.is_empty(), "nothing is parsed here");
        let pending = app.take_pending_dropped_paths();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].layer, None);
        assert_eq!(pending[0].table, None);
        assert_eq!(
            pending[0].path,
            std::path::PathBuf::from("/data/cities.parquet")
        );
    }

    #[test]
    #[cfg(feature = "geoparquet")]
    fn adding_a_geoparquet_layer_from_bytes_selects_it_and_zooms_to_it() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geoparquet_layer_from_bytes(
                "cities.parquet",
                crate::geoparquet_input::fixture::UNCOMPRESSED,
                Some("/data/cities.parquet"),
            )
            .expect("a readable GeoParquet file");
        assert_eq!(app.selection(), Some(id));
        assert_eq!(app.project().layers.len(), 1);
        assert!(
            app.status().is_some_and(|line| line.contains("4 features")),
            "{:?}",
            app.status(),
        );
        // The fixture's points are all in Japan (Tokyo/Osaka/Sapporo); the
        // camera must have moved there from the default (0, 0) view, exactly
        // as a GeoJSON/shapefile/GeoPackage drop does.
        assert!(app.map_view().center().lon > 100.0);
        match &app.project().layers.layers()[0].kind {
            oxigis_core::LayerKind::Vector(oxigis_core::VectorSource::LocalGeoParquet { path }) => {
                assert_eq!(path, "/data/cities.parquet");
            }
            other => panic!("expected LocalGeoParquet, got {other:?}"),
        }
        // §1.2's attribute table binds a LocalGeoParquet layer just like the
        // other local vector sources — proves `is_local_vector_source`
        // actually delivers the table rather than merely compiling.
        let (table_id, table_name, features) =
            app.selected_table_source().expect("the table has a source");
        assert_eq!(table_id, id);
        assert_eq!(table_name, "cities.parquet");
        assert_eq!(features.features.len(), 4);
        assert_eq!(
            features.features[2].properties.as_ref().expect("row 2")["name"],
            serde_json::json!("\u{672d}\u{5e4c}"),
        );
    }

    #[test]
    #[cfg(feature = "geoparquet")]
    fn a_geoparquet_layer_reloaded_from_a_project_is_rebuilt_in_place() {
        let bytes = crate::geoparquet_input::fixture::UNCOMPRESSED;
        let mut app = OxigisApp::new();
        let id = app
            .add_geoparquet_layer_from_bytes("cities.parquet", bytes, Some("/data/cities.parquet"))
            .expect("valid");
        app.sync_project_view();
        let json = app.project().to_json_string().expect("serialize");
        let _ = app.take_pending_local_ops();

        let loaded = Project::from_json_string(&json).expect("parse");
        app.load_project(loaded);
        let pending = app.take_pending_dropped_paths();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].layer, Some(id));
        let _ = app.take_pending_local_ops();

        assert!(app.hydrate_geoparquet_layer_from_bytes(id, "cities.parquet", bytes));
        assert_eq!(app.project().layers.len(), 1, "no duplicate layer");
        let ops = app.take_pending_local_ops();
        assert_eq!(ops.len(), 1);
        assert!(matches!(&ops[0], LocalLayerOp::Add(other, _) if *other == id));

        // Bytes that no longer parse are reported, not silently ignored.
        assert!(!app.hydrate_geoparquet_layer_from_bytes(id, "cities.parquet", &[0u8; 8]));
        assert!(app.status().is_some_and(|line| !line.is_empty()));
    }

    #[test]
    #[cfg(not(feature = "geoparquet"))]
    fn a_browser_style_geoparquet_drop_is_reported_not_misparsed() {
        // Bytes (not a path) is the shape a real browser drop takes; on a
        // build without the `geoparquet` feature this must be reported
        // clearly rather than silently doing nothing or misparsing the
        // Parquet bytes as something else.
        let mut app = OxigisApp::new();
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            dropped_files: vec![egui::DroppedFile {
                name: "cities.parquet".to_string(),
                bytes: Some(vec![0u8; 16].into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert!(app.project().layers.is_empty());
        assert!(
            app.status()
                .is_some_and(|line| line.contains("does not support")),
            "{:?}",
            app.status(),
        );
    }

    #[test]
    fn the_processing_toolbox_window_draws_without_panicking() {
        let mut app = OxigisApp::new();
        let _ = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid");
        app.show_processing = true;
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_input, |ui| app.ui(ui));
        assert!(app.show_processing, "the window stays open with no click");
    }

    // ---- Scale range: the local draw path and the print snapshot ----------

    /// Runs one frame at `zoom`, which is what the per-frame reconciliations
    /// read.
    fn frame_at(app: &mut OxigisApp, ctx: &Context, zoom: f64) {
        let view = app.map_view().with_zoom(zoom);
        app.map_panel.set_view(view);
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
    fn a_local_layer_out_of_its_scale_range_stops_drawing_and_comes_back() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid GeoJSON");
        let ctx = Context::default();
        frame_at(&mut app, &ctx, 10.0);
        // The add's own ops, drained so what follows is only this test's.
        let _ = app.take_pending_local_ops();

        app.project
            .layers
            .set_zoom_range(id, Some(12.0), None)
            .expect("the layer was just added");
        frame_at(&mut app, &ctx, 10.0);
        let ops = app.take_pending_local_ops();
        assert!(
            ops.iter()
                .any(|op| matches!(op, LocalLayerOp::SetVisibility(other, false) if *other == id)),
            "out of range: the mirror the draw path culls on must be told, {ops:?}"
        );

        // Steady state costs nothing: the value has not moved.
        frame_at(&mut app, &ctx, 10.0);
        assert!(
            app.take_pending_local_ops().is_empty(),
            "an idle map at a steady zoom must queue no GPU work"
        );

        frame_at(&mut app, &ctx, 13.0);
        let ops = app.take_pending_local_ops();
        assert!(
            ops.iter()
                .any(|op| matches!(op, LocalLayerOp::SetVisibility(other, true) if *other == id)),
            "back inside the range, so it draws again: {ops:?}"
        );
    }

    #[test]
    fn re_ticking_the_checkbox_of_an_out_of_range_layer_still_leaves_it_hidden() {
        // The pair-comparison case: the applier queues the RAW flag when the
        // checkbox moves, so the reconciliation has to notice the checkbox
        // moved even though the resolved answer did not, and push its `false`
        // over it. Remembering only the resolved answer would silently draw a
        // layer outside its own scale range.
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid GeoJSON");
        let ctx = Context::default();
        app.project
            .layers
            .set_zoom_range(id, Some(12.0), None)
            .expect("the layer was just added");
        frame_at(&mut app, &ctx, 8.0);
        let _ = app.take_pending_local_ops();

        app.apply_layer_action(LayerAction::ToggleVisibility(id));
        frame_at(&mut app, &ctx, 8.0);
        app.apply_layer_action(LayerAction::ToggleVisibility(id));
        frame_at(&mut app, &ctx, 8.0);
        assert!(
            app.project
                .layers
                .get(id)
                .is_some_and(|layer| layer.visible),
            "the checkbox is back on"
        );
        let ops = app.take_pending_local_ops();
        // Last write wins in the queue, so the surviving op is the answer.
        let last = ops
            .iter()
            .rev()
            .find_map(|op| match op {
                LocalLayerOp::SetVisibility(other, visible) if *other == id => Some(*visible),
                _ => None,
            })
            .expect("the toggles must reach the mirror");
        assert!(
            !last,
            "the checkbox is on but the camera is outside the range: {ops:?}"
        );
    }

    #[test]
    fn the_print_snapshot_names_its_layers_the_way_the_panel_does() {
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid GeoJSON");
        app.project
            .layers
            .rename(id, "Prefecture capitals".to_string())
            .expect("the layer was just added");
        app.request_print();
        let request = app.take_pending_print().expect("a print must be queued");
        assert_eq!(request.layers.len(), 1);
        assert_eq!(
            request.layers[0].name, "Prefecture capitals",
            "the legend names the layer the way the layer panel does"
        );
    }

    /// A three-feature layer whose `zone` column classifies into two values
    /// and whose `pop` column is numeric — enough for both renderer kinds.
    const ZONES: &str = r#"{"type":"FeatureCollection","features":[
        {"type":"Feature","properties":{"zone":"a","pop":10},
         "geometry":{"type":"Point","coordinates":[139.0,35.0]}},
        {"type":"Feature","properties":{"zone":"b","pop":20},
         "geometry":{"type":"Point","coordinates":[140.0,36.0]}},
        {"type":"Feature","properties":{"zone":"a","pop":30},
         "geometry":{"type":"Point","coordinates":[141.0,37.0]}}]}"#;

    #[test]
    fn the_style_panel_reaches_the_renderer_editor_with_the_layers_own_fields() {
        // REACHABILITY, the thing this test exists for: the categorized /
        // graduated editor is built and tested, but a feature nobody can open
        // is not shipped. Drawing the app's own style panel must reach it, and
        // must feed it THIS layer's attribute keys rather than an empty list
        // (which would silently degrade the field picker to free text).
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("zones", ZONES, None)
            .expect("valid GeoJSON");
        assert_eq!(app.selection(), Some(id), "an added layer is selected");
        app.apply_style_action(StyleAction::Create(style_panel::StyleKind::Circle));
        let before = app.project().styles.get(&id).cloned();
        assert!(before.is_some(), "the layer has a style to classify");

        egui::__run_test_ui(|ui| {
            assert!(
                !app.style_panel_body(ui, true),
                "drawing the panel is not editing it",
            );
        });
        let (layer, keys) = app
            .renderer_fields
            .as_ref()
            .expect("the field list is derived on the first draw")
            .offered();
        assert_eq!(layer, id);
        // The schema walks each feature's property map, which is ordered by
        // key, so `pop` precedes `zone` however the file spells them — the
        // picker's order is stable across reloads, which is what matters.
        assert_eq!(keys, ["pop".to_string(), "zone".to_string()]);
        assert_eq!(
            app.project().styles.get(&id).cloned(),
            before,
            "a frame with no interaction rewrites nothing",
        );
    }

    #[test]
    fn the_field_list_is_derived_once_per_collection_and_retargeted_on_selection() {
        // The cache's two obligations: it must not re-walk the features every
        // frame (the panel is drawn at 60 Hz over datasets of any size), and it
        // must never offer one layer's columns while another is selected.
        let mut app = OxigisApp::new();
        let zones = app
            .add_geojson_layer_from_text("zones", ZONES, None)
            .expect("valid GeoJSON");
        app.apply_style_action(StyleAction::Create(style_panel::StyleKind::Circle));
        egui::__run_test_ui(|ui| {
            let _ = app.style_panel_body(ui, true);
        });
        let first = app
            .renderer_fields
            .as_ref()
            .map(|cached| cached.offered().1.as_ptr())
            .expect("derived once");
        egui::__run_test_ui(|ui| {
            let _ = app.style_panel_body(ui, true);
        });
        let second = app
            .renderer_fields
            .as_ref()
            .map(|cached| cached.offered().1.as_ptr())
            .expect("still cached");
        assert_eq!(first, second, "the same Arc must not be re-derived");

        // A second layer: the cache retargets rather than answering for the
        // first one.
        let cities = app
            .add_geojson_layer_from_text("cities", POINTS, None)
            .expect("valid GeoJSON");
        assert_ne!(cities, zones);
        app.apply_style_action(StyleAction::Create(style_panel::StyleKind::Circle));
        egui::__run_test_ui(|ui| {
            let _ = app.style_panel_body(ui, true);
        });
        let (layer, keys) = app
            .renderer_fields
            .as_ref()
            .expect("re-derived for the new selection")
            .offered();
        assert_eq!(layer, cities);
        assert_eq!(keys, ["name".to_string()]);
    }

    #[test]
    fn a_classified_layer_draws_through_the_panel_without_being_rewritten() {
        // The panel has to survive a layer that is ALREADY classified — the
        // state a project reloads into — without touching one byte of it.
        let mut app = OxigisApp::new();
        let id = app
            .add_geojson_layer_from_text("zones", ZONES, None)
            .expect("valid GeoJSON");
        app.apply_style_action(StyleAction::Create(style_panel::StyleKind::Circle));
        let base = app
            .project()
            .styles
            .get(&id)
            .map(|set| set.base().clone())
            .expect("a base style");
        if let Some(set) = app.project.styles.get_mut(&id) {
            set.set_renderer(crate::local_vector::classify::categorized_renderer(
                &base,
                "zone",
                [
                    oxigis_core::AttrValue::text("a"),
                    oxigis_core::AttrValue::text("b"),
                ],
            ));
        }
        let before = app.project().styles.get(&id).cloned();
        assert_eq!(
            before.as_ref().map(oxigis_core::LayerStyleSet::class_count),
            Some(2),
        );
        egui::__run_test_ui(|ui| {
            assert!(!app.style_panel_body(ui, true));
        });
        assert_eq!(app.project().styles.get(&id).cloned(), before);
    }

    #[test]
    fn a_tiled_layer_gets_the_refusal_panel_and_no_style_entry() {
        // The other half of reachability: a layer the renderer provably cannot
        // affect must show the reason, not a working-looking editor, and the
        // draw must not create a style entry for it either.
        let mut app = OxigisApp::new();
        app.apply_layer_action(LayerAction::AddDemoXyzLayer);
        let id = app.project().layers.layers().last().map(|layer| layer.id);
        app.selection = id;
        assert!(id.is_some_and(|id| !app.is_local_layer(id)));
        egui::__run_test_ui(|ui| {
            assert!(
                !app.style_panel_body(ui, false),
                "a refused layer reports no repartition",
            );
        });
        assert!(
            app.project().styles.is_empty(),
            "a tiled layer never gains a style entry from being looked at",
        );
        assert!(!crate::vector_provider::TILED_RENDERER_REFUSAL.is_empty());
    }
}
