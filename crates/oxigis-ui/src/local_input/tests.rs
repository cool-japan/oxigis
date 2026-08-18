//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::local_vector::LocalVectorLayer;
use crate::shapefile_input::ShapefileBytes;
use oxigis_core::{Layer, LayerId, LayerKind, LayerStyle, Project, VectorSource};
use std::path::PathBuf;
use std::sync::Arc;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use oxigis_core::{Color, FillStyle, LineStyle, RasterSource};
    /// Two points in Tokyo and Osaka — enough for a real bbox.
    const POINTS: &str = r#"{"type":"FeatureCollection","features":[
        {"type":"Feature","properties":{"name":"Tokyo"},
         "geometry":{"type":"Point","coordinates":[139.767,35.681]}},
        {"type":"Feature","properties":{"name":"Osaka"},
         "geometry":{"type":"Point","coordinates":[135.502,34.702]}}]}"#;
    fn xyz_layer() -> Layer {
        Layer::new(
            "basemap",
            LayerKind::Raster(RasterSource::xyz("https://tile.example/{z}/{x}/{y}.png")),
        )
    }
    #[test]
    fn adding_pasted_geojson_embeds_it_and_queues_the_dataset() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let added = state
            .add_geojson(&mut project, "pasted.geojson", POINTS, None)
            .expect("valid GeoJSON");
        assert_eq!(added.feature_count, 2);
        assert_eq!(added.inline_bytes, Some(POINTS.len()));
        assert_eq!(project.layers.len(), 1);
        assert!(project.styles.contains_key(&added.id));
        assert!(state.default_style(added.id).is_some());
        let layer = project.layers.get(added.id).expect("just added");
        assert!(is_local_layer(layer));
        match &layer.kind {
            LayerKind::Vector(VectorSource::InlineGeoJson { geojson }) => {
                assert_eq!(geojson, POINTS);
            }
            other => panic!("expected inline GeoJSON, got {other:?}"),
        }
        let ops = state.take_ops();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            LocalLayerOp::Add(id, local) => {
                assert_eq!(*id, added.id);
                assert_eq!(local.feature_count(), 2);
            }
            other => panic!("expected an Add op, got {other:?}"),
        }
        assert!(state.take_ops().is_empty(), "ops are handed over once");
    }
    #[test]
    fn a_known_path_is_referenced_rather_than_embedded() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let added = state
            .add_geojson(
                &mut project,
                "cities.geojson",
                POINTS,
                Some("/data/cities.geojson"),
            )
            .expect("valid GeoJSON");
        assert_eq!(added.inline_bytes, None);
        match &project.layers.get(added.id).expect("present").kind {
            LayerKind::Vector(VectorSource::LocalGeoJson { path }) => {
                assert_eq!(path, "/data/cities.geojson");
            }
            other => panic!("expected a path reference, got {other:?}"),
        }
    }
    #[test]
    fn a_value_result_lands_as_the_same_layer_the_text_path_would_have_built() {
        // The Processing route hands back a `serde_json::Value`, never text.
        // What must hold is that skipping the serialise-and-re-parse round trip
        // changes nothing observable: same features, same style, same queued
        // GPU work, same inline source.
        let value: serde_json::Value = serde_json::from_str(POINTS).expect("the fixture is JSON");

        let mut by_text = Project::new("test");
        let mut text_state = LocalInputState::new();
        let text_added = text_state
            .add_geojson(&mut by_text, "cities", POINTS, None)
            .expect("valid GeoJSON");

        let mut by_value = Project::new("test");
        let mut value_state = LocalInputState::new();
        let value_added = value_state
            .add_geojson_value(&mut by_value, "cities", value)
            .expect("valid GeoJSON");

        assert_eq!(value_added.feature_count, text_added.feature_count);
        assert_eq!(value_added.square, text_added.square);
        assert_eq!(by_value.layers.len(), 1);
        assert!(by_value.styles.contains_key(&value_added.id));
        assert!(value_state.default_style(value_added.id).is_some());
        let layer = by_value.layers.get(value_added.id).expect("just added");
        assert!(is_local_layer(layer));
        // Inline, and re-parsable: the stored text is serialised from the
        // parsed collection, so it is the text a project reload reads back —
        // whitespace, not content, is what differs from the source document.
        match &layer.kind {
            LayerKind::Vector(VectorSource::InlineGeoJson { geojson }) => {
                assert_eq!(value_added.inline_bytes, Some(geojson.len()));
                let reparsed = parse_geojson(geojson).expect("the embedded copy must re-parse");
                assert_eq!(reparsed.features.len(), 2);
            }
            other => panic!("expected inline GeoJSON, got {other:?}"),
        }
        let ops = value_state.take_ops();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            LocalLayerOp::Add(id, local) => {
                assert_eq!(*id, value_added.id);
                assert_eq!(local.feature_count(), 2);
            }
            other => panic!("expected an Add op, got {other:?}"),
        }
    }
    #[test]
    fn a_value_that_is_not_a_usable_collection_is_refused_exactly_as_text_is() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        // Not a FeatureCollection at all.
        let error = state
            .add_geojson_value(&mut project, "bad", serde_json::json!({"type": "Point"}))
            .expect_err("must not parse");
        assert!(!error.message().is_empty());
        // An EMPTY collection: the one refusal both seams must word identically,
        // since a document that is rejected on a drop is rejected on a run.
        let empty = serde_json::json!({"type": "FeatureCollection", "features": []});
        let by_value = state
            .add_geojson_value(&mut project, "empty", empty)
            .expect_err("an empty collection is not a dataset");
        let by_text = state
            .add_geojson(
                &mut project,
                "empty",
                "{\"type\":\"FeatureCollection\",\"features\":[]}",
                None,
            )
            .expect_err("an empty collection is not a dataset");
        assert_eq!(by_value.message(), by_text.message());
        assert!(project.layers.is_empty(), "nothing may be added on refusal");
        assert!(state.take_ops().is_empty());
    }
    #[test]
    fn broken_geojson_is_reported_and_adds_nothing() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let error = state
            .add_geojson(&mut project, "bad.geojson", "{not json", None)
            .expect_err("must not parse");
        assert!(!error.message().is_empty());
        assert!(project.layers.is_empty());
        assert!(state.take_ops().is_empty());
    }
    #[test]
    fn repeated_style_edits_for_one_layer_coalesce() {
        let mut state = LocalInputState::new();
        let id = LayerId::new();
        let other = LayerId::new();
        for width in 1..=5 {
            state.queue(LocalLayerOp::SetStyle(
                id,
                LayerStyle::Line(LineStyle::new(Color::WHITE, width as f32)).into(),
            ));
        }
        state.queue(LocalLayerOp::SetStyle(
            other,
            LayerStyle::Fill(FillStyle::new(Color::WHITE)).into(),
        ));
        let ops = state.take_ops();
        assert_eq!(ops.len(), 2, "one op per layer, not one per frame");
        match &ops[0] {
            LocalLayerOp::SetStyle(_, set) => {
                let oxigis_core::LayerStyle::Line(line) = set.base() else {
                    panic!("a line style");
                };
                assert!((line.width() - 5.0).abs() < 1e-6, "last write wins");
            }
            other => panic!("expected the coalesced line style, got {other:?}"),
        }
    }
    #[test]
    fn adds_are_never_coalesced_away() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let first = state
            .add_geojson(&mut project, "a", POINTS, None)
            .expect("valid");
        let second = state
            .add_geojson(&mut project, "b", POINTS, None)
            .expect("valid");
        let ops = state.take_ops();
        assert_eq!(ops.len(), 2);
        assert!(matches!(& ops[0], LocalLayerOp::Add(id, _) if * id == first.id));
        assert!(matches!(& ops[1], LocalLayerOp::Add(id, _) if * id == second.id));
    }
    #[test]
    fn local_layer_order_follows_storage_order_not_the_panel_order() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        project.layers.add(xyz_layer());
        let bottom = state
            .add_geojson(&mut project, "bottom", POINTS, None)
            .expect("valid")
            .id;
        let top = state
            .add_geojson(&mut project, "top", POINTS, None)
            .expect("valid")
            .id;
        assert_eq!(local_layer_order(&project), vec![bottom, top]);
        project.layers.move_up(bottom).expect("present");
        assert_eq!(local_layer_order(&project), vec![top, bottom]);
    }
    #[test]
    fn a_project_round_trip_rebuilds_inline_layers_with_the_saved_style() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let added = state
            .add_geojson(&mut project, "cities", POINTS, None)
            .expect("valid");
        let saved_style = LayerStyle::Fill(FillStyle::new(Color::from_rgb(1, 2, 3)));
        project.styles.insert(added.id, saved_style.clone().into());
        let json = project.to_json_string().expect("serialize");
        let loaded = Project::from_json_string(&json).expect("parse");
        let mut reloaded = LocalInputState::new();
        let notices = reloaded.rebuild_from_project(&loaded);
        assert!(notices.is_empty(), "{notices:?}");
        let ops = reloaded.take_ops();
        assert!(matches!(ops.first(), Some(LocalLayerOp::Clear)));
        match ops.get(1) {
            Some(LocalLayerOp::Add(id, local)) => {
                assert_eq!(*id, added.id);
                assert_eq!(local.style().base(), &saved_style);
                assert_eq!(local.feature_count(), 2);
            }
            other => panic!("expected the rebuilt layer, got {other:?}"),
        }
        assert_eq!(
            reloaded
                .default_style(added.id)
                .map(oxigis_core::LayerStyleSet::base),
            Some(&saved_style)
        );
    }
    #[test]
    fn a_project_round_trip_rebuilds_the_family_map_alongside_the_style() {
        // `rebuild_from_project`'s `InlineGeoJson` arm must populate
        // `families` exactly like every other "adopt this collection" path
        // (`replace_features`, `add_feature_collection`,
        // `add_empty_vector_layer`) does, or the style panel's family-override
        // row and the print pipeline both see an empty set for a layer that
        // just reloaded with real geometry.
        const POLYGON: &str = r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},"geometry":
             {"type":"Polygon","coordinates":[[[0,0],[0,1],[1,1],[1,0],[0,0]]]}}]}"#;
        let mut project = Project::new("test");
        let id = project.layers.add(Layer::new(
            "square",
            LayerKind::Vector(VectorSource::InlineGeoJson {
                geojson: POLYGON.to_string(),
            }),
        ));
        let mut state = LocalInputState::new();
        assert!(state.rebuild_from_project(&project).is_empty());
        assert!(
            !state.families(id).is_empty(),
            "a freshly rebuilt inline layer must be seen by the family map",
        );
    }

    #[test]
    fn rebuild_from_project_clears_families_left_by_a_previous_project() {
        // Two `.oxigis.json` files saved in different *sessions* can both
        // mint layer id 1 (`LayerId`'s counter restarts at process start —
        // see its docs), so a stale `families` entry left behind by an
        // earlier project must not answer for a same-numbered layer in a
        // newly loaded one. `id` is overwritten directly here (bypassing
        // `LayerId::new`) to force the collision deterministically rather
        // than relying on this test running first in the binary.
        let reused = LayerId::from_raw(1);
        let mut polygon_layer = Layer::new(
            "polygon",
            LayerKind::Vector(VectorSource::InlineGeoJson {
                geojson: r#"{"type":"FeatureCollection","features":[
                    {"type":"Feature","properties":{},"geometry":
                     {"type":"Polygon","coordinates":[[[0,0],[0,1],[1,1],[1,0],[0,0]]]}}]}"#
                    .to_string(),
            }),
        );
        polygon_layer.id = reused;
        let mut project_a = Project::new("a");
        project_a.layers.add(polygon_layer);

        let mut state = LocalInputState::new();
        assert!(state.rebuild_from_project(&project_a).is_empty());
        assert!(
            !state.families(reused).is_empty(),
            "project A's polygon layer must be seen by the family map",
        );

        let mut path_layer = Layer::new(
            "path-ref",
            LayerKind::Vector(VectorSource::LocalGeoJson {
                path: "/data/cities.geojson".to_string(),
            }),
        );
        path_layer.id = reused;
        let mut project_b = Project::new("b");
        project_b.layers.add(path_layer);

        // Same `state`, simulating a second project load reusing the
        // process's `LocalInputState` — exactly what `OxigisApp::load_project`
        // does. Project B's own layer references a path and is never
        // hydrated in this test, so nothing repopulates `families` for it.
        state.rebuild_from_project(&project_b);
        assert!(
            state.families(reused).is_empty(),
            "a stale family set from the previous project leaked across the reload",
        );
    }

    #[test]
    fn a_path_backed_layer_is_queued_natively_and_reported_in_a_browser() {
        let mut project = Project::new("test");
        let id = project.layers.add(Layer::new(
            "cities",
            LayerKind::Vector(VectorSource::LocalGeoJson {
                path: "/data/cities.geojson".to_string(),
            }),
        ));
        let mut native = LocalInputState::with_path_support(true);
        assert!(native.rebuild_from_project(&project).is_empty());
        assert_eq!(
            native.take_paths(),
            vec![PendingPath {
                layer: Some(id),
                path: PathBuf::from("/data/cities.geojson"),
                table: None,
            }],
            "a project reference must name the layer it rebuilds",
        );
        let mut browser = LocalInputState::with_path_support(false);
        let notices = browser.rebuild_from_project(&project);
        assert_eq!(notices.len(), 1);
        assert!(notices[0].contains("not available in the browser"));
        assert!(browser.take_paths().is_empty());
    }
    #[test]
    fn a_path_backed_layer_is_rebuilt_in_place_rather_than_duplicated() {
        let mut project = Project::new("test");
        let id = project.layers.add(Layer::new(
            "cities",
            LayerKind::Vector(VectorSource::LocalGeoJson {
                path: "/data/cities.geojson".to_string(),
            }),
        ));
        let saved_style = LayerStyle::Fill(FillStyle::new(Color::from_rgb(9, 8, 7)));
        project.styles.insert(id, saved_style.clone().into());
        project.layers.set_opacity(id, 0.5).expect("present");
        project.layers.toggle_visibility(id).expect("present");
        let mut state = LocalInputState::with_path_support(true);
        assert!(state.rebuild_from_project(&project).is_empty());
        let paths = state.take_paths();
        assert_eq!(paths.len(), 1);
        let target = paths[0].layer.expect("a project reference names its layer");
        state
            .hydrate_geojson(&project, target, POINTS)
            .expect("valid GeoJSON");
        assert_eq!(project.layers.len(), 1);
        let ops = state.take_ops();
        assert!(matches!(ops.first(), Some(LocalLayerOp::Clear)));
        match ops.get(1) {
            Some(LocalLayerOp::Add(other, local)) => {
                assert_eq!(*other, id, "the original id must be kept");
                assert_eq!(local.style().base(), &saved_style);
                assert!(!local.visible());
                assert!((local.opacity() - 0.5).abs() < 1e-6);
            }
            other => panic!("expected the rebuilt layer, got {other:?}"),
        }
        assert_eq!(
            state
                .default_style(id)
                .map(oxigis_core::LayerStyleSet::base),
            Some(&saved_style)
        );
    }
    #[test]
    fn replace_features_adopts_the_arc_and_leaves_the_default_style_alone() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let added = state
            .add_geojson(&mut project, "cities", POINTS, None)
            .expect("valid");
        let born_with = state
            .default_style(added.id)
            .cloned()
            .expect("the add path records one");
        let explicit = LayerStyle::Line(LineStyle::new(Color::from_rgb(7, 7, 7), 3.0));
        assert_ne!(born_with.base(), &explicit);
        project.styles.insert(added.id, explicit.clone().into());
        project.layers.set_opacity(added.id, 0.5).expect("present");
        project.layers.toggle_visibility(added.id).expect("present");
        let _drained = state.take_ops();

        let edited = Arc::new(parse_geojson(POINTS).expect("valid"));
        state
            .replace_features(&project, added.id, Arc::clone(&edited))
            .expect("the layer is still in the project");

        assert!(
            state
                .feature_set(added.id)
                .is_some_and(|held| Arc::ptr_eq(held, &edited)),
            "the store must adopt the caller's Arc, not a copy of it"
        );
        assert_eq!(
            state.default_style(added.id),
            Some(&born_with),
            "a data change must not re-derive the remembered default style"
        );
        match state.take_ops().as_slice() {
            [LocalLayerOp::Add(id, local)] => {
                assert_eq!(*id, added.id);
                assert!(Arc::ptr_eq(&local.features_arc(), &edited));
                assert_eq!(
                    local.style().base(),
                    &explicit,
                    "the project style still wins"
                );
                assert!(!local.visible());
                assert!((local.opacity() - 0.5).abs() < 1e-6);
            }
            other => panic!("expected one Add, got {other:?}"),
        }

        // Without a project style the remembered default is what gets used —
        // read, never rewritten.
        project.styles.remove(&added.id);
        state
            .replace_features(&project, added.id, edited)
            .expect("still present");
        match state.take_ops().as_slice() {
            [LocalLayerOp::Add(_, local)] => assert_eq!(local.style(), &born_with),
            other => panic!("expected one Add, got {other:?}"),
        }

        // A layer the project no longer holds is refused, with nothing queued.
        let error = state
            .replace_features(
                &project,
                LayerId::new(),
                Arc::new(parse_geojson(POINTS).expect("valid")),
            )
            .expect_err("the layer is gone");
        assert!(error.message().contains("no longer in the project"));
        assert!(state.take_ops().is_empty());
    }
    #[test]
    fn an_empty_vector_layer_is_inline_from_birth_and_carries_the_chosen_style() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let added = state
            .add_empty_vector_layer(&mut project, "sketch", crate::style_panel::StyleKind::Line)
            .expect("an empty layer is a legitimate request, not a failed drop");

        assert_eq!(added.feature_count, 0);
        assert!(added.inline_bytes.is_some_and(|bytes| bytes > 0));
        assert_eq!(added.square, crate::local_vector::MercatorSquare::world());
        match project.layers.get(added.id).map(|layer| &layer.kind) {
            Some(LayerKind::Vector(VectorSource::InlineGeoJson { geojson })) => {
                let parsed = parse_geojson(geojson);
                assert!(
                    parsed.is_err(),
                    "the stored text really is an empty collection, which the \
                     drop path still refuses"
                );
                let direct = oxigeo::geojson::reader::feature_collection_from_str(geojson)
                    .expect("but it is valid GeoJSON");
                assert!(direct.features.is_empty());
            }
            other => panic!("expected inline GeoJSON, got {other:?}"),
        }

        let expected = crate::style_panel::StyleKind::Line.default_style();
        assert_eq!(
            project
                .styles
                .get(&added.id)
                .map(oxigis_core::LayerStyleSet::base),
            Some(&expected)
        );
        assert_eq!(
            state
                .default_style(added.id)
                .map(oxigis_core::LayerStyleSet::base),
            Some(&expected)
        );
        assert_eq!(
            state
                .feature_set(added.id)
                .map(|features| features.features.len()),
            Some(0)
        );
        match state.take_ops().as_slice() {
            [LocalLayerOp::Add(id, local)] => {
                assert_eq!(*id, added.id);
                assert_eq!(local.feature_count(), 0);
                assert_eq!(local.style().base(), &expected);
            }
            other => panic!("expected one Add, got {other:?}"),
        }
    }
    #[test]
    fn an_empty_inline_layer_rebuilds_instead_of_being_reported() {
        let mut project = Project::new("test");
        let id = project.layers.add(Layer::new(
            "emptied",
            LayerKind::Vector(VectorSource::InlineGeoJson {
                geojson: r#"{"type":"FeatureCollection","features":[]}"#.to_string(),
            }),
        ));
        let mut state = LocalInputState::new();
        let notices = state.rebuild_from_project(&project);
        assert!(notices.is_empty(), "{notices:?}");
        assert_eq!(
            state
                .feature_set(id)
                .map(|features| features.features.len()),
            Some(0),
            "an empty collection is still a loaded collection"
        );
        match state.take_ops().as_slice() {
            [LocalLayerOp::Clear, LocalLayerOp::Add(added, local)] => {
                assert_eq!(*added, id);
                assert_eq!(local.feature_count(), 0);
            }
            other => panic!("expected Clear then Add, got {other:?}"),
        }

        // Text that is genuinely unreadable is still reported, with the same
        // wording as before.
        let mut broken = Project::new("test");
        broken.layers.add(Layer::new(
            "broken",
            LayerKind::Vector(VectorSource::InlineGeoJson {
                geojson: "not json at all".to_string(),
            }),
        ));
        let notices = LocalInputState::new().rebuild_from_project(&broken);
        assert_eq!(notices.len(), 1);
        assert!(notices[0].contains("could not be rebuilt"));
        assert!(notices[0].contains("GeoJSON parse failed"));
    }
    #[test]
    fn hydrating_a_layer_the_project_no_longer_has_is_an_error() {
        let project = Project::new("test");
        let mut state = LocalInputState::new();
        let error = state
            .hydrate_geojson(&project, LayerId::new(), POINTS)
            .expect_err("the layer is gone");
        assert!(error.message().contains("no longer in the project"));
        assert!(state.take_ops().is_empty());
    }
    #[test]
    fn an_edit_queued_after_a_remove_and_re_add_is_not_folded_backwards() {
        let mut state = LocalInputState::new();
        let id = LayerId::new();
        state.queue(LocalLayerOp::SetVisibility(id, false));
        state.queue(LocalLayerOp::Remove(id));
        let layer = LocalVectorLayer::from_geojson("cities", POINTS).expect("valid");
        state.queue(LocalLayerOp::Add(id, Box::new(layer)));
        state.queue(LocalLayerOp::SetVisibility(id, true));
        let ops = state.take_ops();
        assert_eq!(ops.len(), 4, "the barrier must stop the backward scan");
        assert!(matches!(ops[0], LocalLayerOp::SetVisibility(_, false)));
        assert!(matches!(ops[3], LocalLayerOp::SetVisibility(_, true)));
    }
    #[test]
    fn a_shapefile_layer_is_a_local_layer_but_a_tiled_one_is_not() {
        let layer = Layer::new(
            "shape",
            LayerKind::Vector(VectorSource::LocalShapefile {
                path: "/data/x.shp".to_string(),
            }),
        );
        assert!(is_local_layer(&layer));
        assert!(is_local_vector_source(&VectorSource::LocalGpkg {
            path: "/data/x.gpkg".to_string(),
            table: "cities".to_string(),
        }));
        assert!(!is_local_vector_source(&VectorSource::MvtTiles {
            url_template: "https://x/{z}/{x}/{y}.pbf".to_string(),
            paints: Vec::new(),
        }));
    }
    #[test]
    fn geojson_file_names_are_recognised_case_insensitively() {
        assert!(looks_like_geojson("Cities.GeoJSON"));
        assert!(looks_like_geojson("cities.json"));
        assert!(!looks_like_geojson("cities.shp"));
        assert!(!looks_like_geojson("geojson"));
        assert_eq!(display_name("/data/sub/cities.geojson"), "cities.geojson");
        assert_eq!(display_name("C:\\data\\cities.json"), "cities.json");
        assert_eq!(display_name("cities.geojson"), "cities.geojson");
    }
    /// A dropped file that carries bytes.
    fn dropped(name: &str, bytes: &[u8]) -> DroppedItem {
        DroppedItem {
            name: name.to_string(),
            bytes: Some(Arc::from(bytes)),
            path: None,
        }
    }
    /// A dropped file that carries only a path (the native shell's shape).
    fn dropped_path(path: &str) -> DroppedItem {
        DroppedItem {
            name: display_name(path),
            bytes: None,
            path: Some(PathBuf::from(path)),
        }
    }
    #[test]
    fn extensions_are_classified_case_insensitively() {
        assert_eq!(classify_drop("Cities.GeoJSON"), DropKind::GeoJson);
        assert_eq!(
            classify_drop("CITIES.SHP"),
            DropKind::Shapefile(ShapefilePart::Shp),
        );
        assert_eq!(
            classify_drop("cities.dbf"),
            DropKind::Shapefile(ShapefilePart::Dbf),
        );
        assert_eq!(classify_drop("scene.tif"), DropKind::Unsupported);
        assert_eq!(file_stem("CITIES.SHP"), "cities");
        assert_eq!(file_stem("no_extension"), "no_extension");
    }
    #[test]
    fn a_geolibre_project_name_is_classified_ahead_of_the_plain_json_rule() {
        // `.geolibre.json` also ends `.json`, so the more specific check
        // must win, or every GeoLibre drop would be misparsed as a
        // GeoJSON `FeatureCollection` and fail.
        assert_eq!(
            classify_drop("map.geolibre.json"),
            DropKind::GeoLibreProject
        );
        assert_eq!(
            classify_drop("MAP.GEOLIBRE.JSON"),
            DropKind::GeoLibreProject
        );
        assert_eq!(classify_drop("map.json"), DropKind::GeoJson);
        assert_eq!(classify_drop("map.geojson"), DropKind::GeoJson);
    }
    #[test]
    fn a_dropped_geolibre_project_becomes_its_own_dataset() {
        let (datasets, notices) = group_dropped_files(vec![dropped(
            "map.geolibre.json",
            br#"{"version":"0.2.0","name":"X","mapView":{"center":[0,0],"zoom":1},"layers":[]}"#,
        )]);
        assert!(notices.is_empty());
        assert_eq!(datasets.len(), 1);
        assert!(matches!(datasets[0], DroppedDataset::GeoLibreProject(_)));
    }
    #[test]
    fn one_drop_holding_geojson_and_a_shapefile_set_is_split_into_two_datasets() {
        let (shp, dbf) = crate::shapefile_input::fixture::cities();
        let items = vec![
            dropped("CITIES.DBF", &dbf),
            dropped("cities.prj", b"GEOGCS[\"GCS_WGS_1984\"]"),
            dropped("borders.geojson", POINTS.as_bytes()),
            dropped("cities.shp", &shp),
            dropped("cities.cpg", b"UTF-8"),
            dropped("cities.shx", &[0u8; 8]),
        ];
        let (datasets, notices) = group_dropped_files(items);
        assert!(notices.is_empty(), "{notices:?}");
        assert_eq!(datasets.len(), 2);
        match &datasets[0] {
            DroppedDataset::GeoJson(item) => assert_eq!(item.name, "borders.geojson"),
            other => panic!("expected the GeoJSON first, got {other:?}"),
        }
        match &datasets[1] {
            DroppedDataset::Shapefile(set) => {
                assert_eq!(set.shp.name, "cities.shp");
                assert!(set.dbf.is_some(), "the upper-cased .DBF must pair");
                assert!(set.prj.is_some());
                assert!(set.cpg.is_some());
            }
            other => panic!("expected the shapefile second, got {other:?}"),
        }
    }
    #[test]
    fn a_companion_without_its_shp_is_ignored_with_a_notice() {
        let (datasets, notices) = group_dropped_files(vec![
            dropped("orphan.dbf", &[0u8; 4]),
            dropped("scene.tif", &[0u8; 4]),
        ]);
        assert!(datasets.is_empty());
        assert_eq!(notices.len(), 2);
        assert!(notices.iter().any(|line| line.contains("without its .shp")));
        assert!(
            notices
                .iter()
                .any(|line| line.contains("not a supported file type"))
        );
    }
    #[test]
    fn two_shapefile_sets_in_one_drop_stay_apart() {
        let (shp, dbf) = crate::shapefile_input::fixture::cities();
        let (datasets, notices) = group_dropped_files(vec![
            dropped("a.shp", &shp),
            dropped("b.shp", &shp),
            dropped("b.dbf", &dbf),
        ]);
        assert!(notices.is_empty(), "{notices:?}");
        assert_eq!(datasets.len(), 2);
        match (&datasets[0], &datasets[1]) {
            (DroppedDataset::Shapefile(first), DroppedDataset::Shapefile(second)) => {
                assert!(first.dbf.is_none(), "a.dbf was never dropped");
                assert!(second.dbf.is_some());
            }
            other => panic!("expected two shapefile sets, got {other:?}"),
        }
    }
    #[test]
    fn a_native_shapefile_drop_arrives_as_paths_and_needs_no_bytes() {
        let (datasets, notices) = group_dropped_files(vec![
            dropped_path("/data/cities.dbf"),
            dropped_path("/data/cities.shp"),
        ]);
        assert!(notices.is_empty(), "{notices:?}");
        assert_eq!(datasets.len(), 1);
        match &datasets[0] {
            DroppedDataset::Shapefile(set) => {
                assert_eq!(set.shp.path, Some(PathBuf::from("/data/cities.shp")));
                assert!(set.shp.bytes.is_none());
            }
            other => panic!("expected a shapefile set, got {other:?}"),
        }
    }
    #[test]
    fn a_native_shapefile_drop_is_referenced_by_path_and_reloads_in_place() {
        let (shp, dbf) = crate::shapefile_input::fixture::cities();
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(true);
        let added = state
            .add_shapefile(
                &mut project,
                "cities.shp",
                ShapefileBytes::new(&shp).with_dbf(Some(&dbf)),
                Some("/data/cities.shp"),
            )
            .expect("valid shapefile");
        assert_eq!(added.feature_count, 2);
        assert_eq!(added.inline_bytes, None, "a path is referenced, not copied");
        match &project.layers.get(added.id).expect("present").kind {
            LayerKind::Vector(VectorSource::LocalShapefile { path }) => {
                assert_eq!(path, "/data/cities.shp");
            }
            other => panic!("expected a shapefile reference, got {other:?}"),
        }
        let features = state.feature_set(added.id).expect("stored");
        let row = features.features[0].properties.as_ref().expect("row");
        assert_eq!(row["NAME"], serde_json::json!("Tokyo"));
        assert!(
            matches!(
                project.styles.get(&added.id).map(|set| set.base()),
                Some(LayerStyle::Circle(_))
            ),
            "a point shapefile must get the circle default, got {:?}",
            project.styles.get(&added.id),
        );
        let saved_style = LayerStyle::Fill(FillStyle::new(Color::from_rgb(4, 5, 6)));
        project.styles.insert(added.id, saved_style.clone().into());
        let json = project.to_json_string().expect("serialize");
        let loaded = Project::from_json_string(&json).expect("parse");
        let mut reloaded = LocalInputState::with_path_support(true);
        assert!(reloaded.rebuild_from_project(&loaded).is_empty());
        assert_eq!(
            reloaded.take_paths(),
            vec![PendingPath {
                layer: Some(added.id),
                path: PathBuf::from("/data/cities.shp"),
                table: None,
            }],
        );
        reloaded
            .hydrate_shapefile(
                &loaded,
                added.id,
                ShapefileBytes::new(&shp).with_dbf(Some(&dbf)),
            )
            .expect("valid shapefile");
        assert_eq!(loaded.layers.len(), 1, "no duplicate layer");
        let ops = reloaded.take_ops();
        assert!(matches!(ops.first(), Some(LocalLayerOp::Clear)));
        match ops.get(1) {
            Some(LocalLayerOp::Add(id, local)) => {
                assert_eq!(*id, added.id);
                assert_eq!(local.style().base(), &saved_style);
                assert_eq!(local.feature_count(), 2);
            }
            other => panic!("expected the rebuilt layer, got {other:?}"),
        }
    }
    #[test]
    fn a_browser_shapefile_drop_is_embedded_as_geojson_and_reloads_inline() {
        let (shp, dbf) = crate::shapefile_input::fixture::cities();
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(false);
        let added = state
            .add_shapefile(
                &mut project,
                "cities.shp",
                ShapefileBytes::new(&shp).with_dbf(Some(&dbf)),
                None,
            )
            .expect("valid shapefile");
        let inline = added.inline_bytes.expect("no path means embedded");
        assert!(inline > 0);
        match &project.layers.get(added.id).expect("present").kind {
            LayerKind::Vector(VectorSource::InlineGeoJson { geojson }) => {
                assert!(geojson.contains("Tokyo"), "the attributes must survive");
            }
            other => panic!("expected inline GeoJSON, got {other:?}"),
        }
        let json = project.to_json_string().expect("serialize");
        let loaded = Project::from_json_string(&json).expect("parse");
        let mut reloaded = LocalInputState::with_path_support(false);
        assert!(reloaded.rebuild_from_project(&loaded).is_empty());
        assert!(reloaded.take_paths().is_empty());
        let ops = reloaded.take_ops();
        match ops.get(1) {
            Some(LocalLayerOp::Add(id, local)) => {
                assert_eq!(*id, added.id);
                assert_eq!(local.feature_count(), 2);
            }
            other => panic!("expected the rebuilt layer, got {other:?}"),
        }
    }
    #[test]
    fn a_polygon_shapefile_gets_the_fill_default_style() {
        let shp = crate::shapefile_input::fixture::shp_bytes(
            oxigeo::shapefile::shp::shapes::ShapeType::Polygon,
            (138.0, 35.0, 140.0, 37.0),
            vec![crate::shapefile_input::fixture::donut()],
        );
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let added = state
            .add_shapefile(&mut project, "donut.shp", ShapefileBytes::new(&shp), None)
            .expect("valid shapefile");
        assert!(
            matches!(
                project.styles.get(&added.id).map(|set| set.base()),
                Some(LayerStyle::Fill(_))
            ),
            "a polygon shapefile must get the fill default, got {:?}",
            project.styles.get(&added.id),
        );
    }
    #[test]
    fn a_shapefile_path_reference_is_reported_in_a_browser() {
        let mut project = Project::new("test");
        project.layers.add(Layer::new(
            "cities",
            LayerKind::Vector(VectorSource::LocalShapefile {
                path: "/data/cities.shp".to_string(),
            }),
        ));
        let mut browser = LocalInputState::with_path_support(false);
        let notices = browser.rebuild_from_project(&project);
        assert_eq!(notices.len(), 1);
        assert!(notices[0].contains("not available in the browser"));
    }
    #[test]
    fn a_broken_shapefile_is_reported_and_adds_nothing() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let error = state
            .add_shapefile(
                &mut project,
                "bad.shp",
                ShapefileBytes::new(&[0u8; 8]),
                None,
            )
            .expect_err("must not parse");
        assert!(!error.message().is_empty());
        assert!(project.layers.is_empty());
        assert!(state.take_ops().is_empty());
    }
    #[test]
    fn a_geopackage_is_its_own_dataset_and_needs_no_grouping() {
        assert_eq!(classify_drop("tokyo.gpkg"), DropKind::GeoPackage);
        assert_eq!(classify_drop("TOKYO.GPKG"), DropKind::GeoPackage);
        assert_eq!(classify_drop("tokyo.gpkg-wal"), DropKind::Unsupported);
        let (datasets, notices) = group_dropped_files(vec![
            dropped("borders.geojson", b"{}"),
            dropped("tokyo.gpkg", &[0u8; 4]),
            dropped("osaka.gpkg", &[0u8; 4]),
        ]);
        assert!(notices.is_empty());
        assert_eq!(datasets.len(), 3, "one dataset per file, in drop order");
        assert!(matches!(datasets[1], DroppedDataset::GeoPackage(_)));
        assert!(matches!(datasets[2], DroppedDataset::GeoPackage(_)));
    }
    #[test]
    fn a_native_geopackage_drop_becomes_one_layer_per_table_and_reloads_in_place() {
        let bytes = crate::gpkg_input::fixture::BASIC;
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(true);
        let added = state
            .add_gpkg(&mut project, "basic.gpkg", bytes, Some("/data/basic.gpkg"))
            .expect("a readable GeoPackage");
        assert_eq!(added.layers.len(), 3, "cities, parks and roads");
        assert_eq!(added.notices.len(), 1, "regions is in EPSG:2154");
        assert!(added.notices[0].contains("RGF93 / Lambert-93"));
        let names: Vec<&str> = project
            .layers
            .layers()
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["basic:cities", "basic:parks", "basic:roads"],
            "several tables from one file are prefixed with its stem",
        );
        let sources: Vec<(String, String)> = project
            .layers
            .layers()
            .iter()
            .filter_map(|layer| match &layer.kind {
                LayerKind::Vector(VectorSource::LocalGpkg { path, table }) => {
                    Some((path.clone(), table.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            sources,
            vec![
                ("/data/basic.gpkg".to_string(), "cities".to_string()),
                ("/data/basic.gpkg".to_string(), "parks".to_string()),
                ("/data/basic.gpkg".to_string(), "roads".to_string()),
            ],
        );
        let cities = added.layers[0].id;
        assert!(
            matches!(
                project.styles.get(&cities).map(|set| set.base()),
                Some(LayerStyle::Circle(_))
            ),
            "a point table must get the circle default",
        );
        assert!(
            matches!(
                project
                    .styles
                    .get(&added.layers[1].id)
                    .map(|set| set.base()),
                Some(LayerStyle::Fill(_))
            ),
            "a polygon table must get the fill default",
        );
        let json = project.to_json_string().expect("serialize");
        let loaded = Project::from_json_string(&json).expect("parse");
        let mut reloaded = LocalInputState::with_path_support(true);
        assert!(reloaded.rebuild_from_project(&loaded).is_empty());
        let paths = reloaded.take_paths();
        assert_eq!(paths.len(), 3);
        assert_eq!(
            paths[0],
            PendingPath {
                layer: Some(cities),
                path: PathBuf::from("/data/basic.gpkg"),
                table: Some("cities".to_string()),
            },
        );
        reloaded
            .hydrate_gpkg(&loaded, cities, bytes, "cities")
            .expect("the table is still there");
        assert_eq!(loaded.layers.len(), 3, "no duplicate layer");
        let ops = reloaded.take_ops();
        match ops.get(1) {
            Some(LocalLayerOp::Add(id, local)) => {
                assert_eq!(*id, cities);
                assert_eq!(local.feature_count(), 4);
            }
            other => panic!("expected the rebuilt layer, got {other:?}"),
        }
        let error = reloaded
            .hydrate_gpkg(&loaded, cities, bytes, "gone")
            .expect_err("no such table");
        assert!(error.message().contains("gone"), "{}", error.message());
        let error = reloaded
            .hydrate_gpkg(&loaded, cities, bytes, "regions")
            .expect_err("an unsupported CRS");
        assert!(
            error.message().contains("RGF93 / Lambert-93"),
            "{}",
            error.message(),
        );
    }
    #[test]
    fn a_browser_geopackage_drop_embeds_every_table_as_geojson() {
        let bytes = crate::gpkg_input::fixture::BASIC;
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(false);
        let added = state
            .add_gpkg(&mut project, "basic.gpkg", bytes, None)
            .expect("a readable GeoPackage");
        assert_eq!(added.layers.len(), 3);
        for layer in &added.layers {
            assert!(layer.inline_bytes.is_some_and(|bytes| bytes > 0));
        }
        match &project
            .layers
            .get(added.layers[0].id)
            .expect("present")
            .kind
        {
            LayerKind::Vector(VectorSource::InlineGeoJson { geojson }) => {
                assert!(geojson.contains("Tokyo"), "the attributes must survive");
                assert!(geojson.contains("\u{6771}\u{4eac}"));
            }
            other => panic!("expected inline GeoJSON, got {other:?}"),
        }
        let json = project.to_json_string().expect("serialize");
        let loaded = Project::from_json_string(&json).expect("parse");
        let mut reloaded = LocalInputState::with_path_support(false);
        assert!(reloaded.rebuild_from_project(&loaded).is_empty());
        assert!(reloaded.take_paths().is_empty());
        assert_eq!(reloaded.feature_set_count(), 3);
    }
    #[test]
    fn a_single_table_geopackage_is_named_after_its_table_alone() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let added = state
            .add_gpkg(
                &mut project,
                "points.gpkg",
                crate::gpkg_input::fixture::PAGED,
                None,
            )
            .expect("a readable GeoPackage");
        assert_eq!(added.layers.len(), 1);
        assert_eq!(added.layers[0].feature_count, 300);
        assert_eq!(project.layers.layers()[0].name, "pts", "no stem prefix");
    }
    #[test]
    fn a_geopackage_whose_every_table_is_refused_adds_nothing_and_says_why() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let error = state
            .add_gpkg(
                &mut project,
                "weird.gpkg",
                crate::gpkg_input::fixture::WITHOUT_ROWID,
                None,
            )
            .expect_err("its only feature table is unreadable");
        assert!(
            error.message().contains("WITHOUT ROWID"),
            "{}",
            error.message()
        );
        assert!(project.layers.is_empty());
        assert!(state.take_ops().is_empty());
    }
    #[test]
    fn a_geopackage_with_nothing_to_refuse_says_so_differently() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let error = state
            .add_gpkg(
                &mut project,
                "notes.gpkg",
                &crate::gpkg_input::fixture::attributes_only_image(),
                None,
            )
            .expect_err("no feature table means no layer");
        assert!(
            error.message().contains("no feature tables"),
            "{}",
            error.message(),
        );
        assert!(project.layers.is_empty());
    }
    #[test]
    fn a_broken_geopackage_is_reported_and_adds_nothing() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let error = state
            .add_gpkg(&mut project, "bad.gpkg", &[0u8; 64], None)
            .expect_err("must not parse");
        assert!(!error.message().is_empty());
        assert!(project.layers.is_empty());
        assert!(state.take_ops().is_empty());
    }
    #[test]
    fn a_geopackage_path_reference_is_reported_in_a_browser() {
        let mut project = Project::new("test");
        project.layers.add(Layer::new(
            "cities",
            LayerKind::Vector(VectorSource::LocalGpkg {
                path: "/data/basic.gpkg".to_string(),
                table: "cities".to_string(),
            }),
        ));
        let mut browser = LocalInputState::with_path_support(false);
        let notices = browser.rebuild_from_project(&project);
        assert_eq!(notices.len(), 1);
        assert!(notices[0].contains("not available in the browser"));
        assert!(notices[0].contains("cities"), "{}", notices[0]);
    }
    // ---- GeoParquet ---------------------------------------------------------

    #[test]
    fn a_geoparquet_file_is_its_own_dataset_and_needs_no_grouping() {
        // Recognition is unconditional — see `DropKind::GeoParquet`'s docs —
        // so this test (unlike the ones below) runs the same whether or not
        // the `geoparquet` feature is on.
        assert_eq!(classify_drop("tokyo.parquet"), DropKind::GeoParquet);
        assert_eq!(classify_drop("TOKYO.PARQUET"), DropKind::GeoParquet);
        assert_eq!(classify_drop("tokyo.geoparquet"), DropKind::GeoParquet);
        assert_eq!(classify_drop("tokyo.parquet.crc"), DropKind::Unsupported);
        let (datasets, notices) = group_dropped_files(vec![
            dropped("borders.geojson", b"{}"),
            dropped("tokyo.parquet", &[0u8; 4]),
            dropped("osaka.geoparquet", &[0u8; 4]),
        ]);
        assert!(notices.is_empty());
        assert_eq!(datasets.len(), 3, "one dataset per file, in drop order");
        assert!(matches!(datasets[1], DroppedDataset::GeoParquet(_)));
        assert!(matches!(datasets[2], DroppedDataset::GeoParquet(_)));
    }

    #[cfg(feature = "geoparquet")]
    #[test]
    fn a_native_geoparquet_drop_is_referenced_by_path_and_reloads_in_place() {
        let bytes = crate::geoparquet_input::fixture::UNCOMPRESSED;
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(true);
        let added = state
            .add_geoparquet(
                &mut project,
                "cities.parquet",
                bytes,
                Some("/data/cities.parquet"),
            )
            .expect("a readable GeoParquet file");
        assert_eq!(added.feature_count, 4);
        assert_eq!(project.layers.len(), 1);
        match &project.layers.layers()[0].kind {
            LayerKind::Vector(VectorSource::LocalGeoParquet { path }) => {
                assert_eq!(path, "/data/cities.parquet");
            }
            other => panic!("expected LocalGeoParquet, got {other:?}"),
        }
        let id = added.id;
        let json = project.to_json_string().expect("serialize");
        let loaded = Project::from_json_string(&json).expect("parse");
        let mut reloaded = LocalInputState::with_path_support(true);
        assert!(reloaded.rebuild_from_project(&loaded).is_empty());
        assert_eq!(
            reloaded.take_paths(),
            vec![PendingPath {
                layer: Some(id),
                path: PathBuf::from("/data/cities.parquet"),
                table: None,
            }],
        );
        reloaded
            .hydrate_geoparquet(&loaded, id, bytes)
            .expect("the file is still there and still readable");
        assert_eq!(loaded.layers.len(), 1, "no duplicate layer");
        match reloaded.take_ops().get(1) {
            Some(LocalLayerOp::Add(rebuilt_id, local)) => {
                assert_eq!(*rebuilt_id, id);
                assert_eq!(local.feature_count(), 4);
            }
            other => panic!("expected the rebuilt layer, got {other:?}"),
        }
        let error = reloaded
            .hydrate_geoparquet(&loaded, id, &[0u8; 8])
            .expect_err("garbage bytes are not a GeoParquet file");
        assert!(!error.message().is_empty());
    }

    #[cfg(feature = "geoparquet")]
    #[test]
    fn a_browser_geoparquet_drop_embeds_it_as_geojson() {
        let bytes = crate::geoparquet_input::fixture::UNCOMPRESSED;
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(false);
        let added = state
            .add_geoparquet(&mut project, "cities.parquet", bytes, None)
            .expect("a readable GeoParquet file");
        assert!(added.inline_bytes.is_some_and(|bytes| bytes > 0));
        match &project.layers.layers()[0].kind {
            LayerKind::Vector(VectorSource::InlineGeoJson { geojson }) => {
                assert!(
                    geojson.contains("\u{672d}\u{5e4c}"),
                    "Japanese text must survive the round trip through GeoJSON text: {geojson}",
                );
            }
            other => panic!("expected inline GeoJSON, got {other:?}"),
        }
        let json = project.to_json_string().expect("serialize");
        let loaded = Project::from_json_string(&json).expect("parse");
        let mut reloaded = LocalInputState::with_path_support(false);
        assert!(reloaded.rebuild_from_project(&loaded).is_empty());
        assert!(reloaded.take_paths().is_empty());
        assert_eq!(reloaded.feature_set_count(), 1);
    }

    #[cfg(feature = "geoparquet")]
    #[test]
    fn a_broken_geoparquet_file_is_reported_and_adds_nothing() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::new();
        let error = state
            .add_geoparquet(&mut project, "bad.parquet", &[0u8; 64], None)
            .expect_err("must not parse");
        assert!(!error.message().is_empty());
        assert!(project.layers.is_empty());
        assert!(state.take_ops().is_empty());
    }

    #[cfg(feature = "geoparquet")]
    #[test]
    fn a_geoparquet_path_reference_is_reported_in_a_browser() {
        let mut project = Project::new("test");
        project.layers.add(Layer::new(
            "cities",
            LayerKind::Vector(VectorSource::LocalGeoParquet {
                path: "/data/cities.parquet".to_string(),
            }),
        ));
        let mut browser = LocalInputState::with_path_support(false);
        let notices = browser.rebuild_from_project(&project);
        assert_eq!(notices.len(), 1);
        assert!(notices[0].contains("not available in the browser"));
        assert!(notices[0].contains("cities"), "{}", notices[0]);
    }

    /// The feature-off twin of the tests above: a build with the
    /// `geoparquet` Cargo feature off (the default — this test only *runs*
    /// under `cargo test -p oxigis-ui` without `--features geoparquet`,
    /// since `cargo nextest run` at the workspace level unifies the feature
    /// on via `oxigis-desktop`) must still recognise a `LocalGeoParquet`
    /// project reference and skip it with a clear notice, rather than
    /// panicking or silently dropping the layer.
    #[cfg(not(feature = "geoparquet"))]
    #[test]
    fn a_geoparquet_reference_on_a_build_without_the_feature_is_skipped_with_a_notice() {
        let mut project = Project::new("test");
        project.layers.add(Layer::new(
            "cities",
            LayerKind::Vector(VectorSource::LocalGeoParquet {
                path: "/data/cities.parquet".to_string(),
            }),
        ));
        // Even with paths nominally supported, the feature being off is what
        // must decide this — the notice's wording must say so, not "not
        // available in the browser" (this is not a browser).
        let mut state = LocalInputState::with_path_support(true);
        let notices = state.rebuild_from_project(&project);
        assert_eq!(notices.len(), 1);
        assert!(notices[0].contains("does not support"), "{}", notices[0]);
        assert!(notices[0].contains("cities"), "{}", notices[0]);
        assert!(
            state.take_paths().is_empty(),
            "nothing should be queued to read without the feature",
        );
    }

    #[test]
    fn a_symbol_style_compiles_into_label_rules() {
        let mut local = LocalVectorLayer::from_geojson("cities", POINTS).expect("valid");
        assert!(local.labels().is_empty());
        local.set_style(crate::local_vector::local_symbol_style("name"));
        assert!(!local.labels().is_empty(), "symbol styles must label");
    }

    #[test]
    fn two_same_named_shapefiles_from_different_folders_keep_their_own_companions() {
        // Every dropped item's name has already been reduced to a bare file
        // name by `display_name`, so grouping on the stem alone collapses
        // `region_a/roads.*` and `region_b/roads.*` into one set: both `.dbf`s
        // land on whichever `.shp` was seen last and the other shapefile loads
        // geometry-only, in WGS 84, with nothing said about it. The folder the
        // native shell *did* hand over is what tells them apart.
        let (datasets, notices) = group_dropped_files(vec![
            dropped_path("/data/region_a/roads.shp"),
            dropped_path("/data/region_b/roads.shp"),
            dropped_path("/data/region_b/roads.prj"),
            dropped_path("/data/region_a/roads.dbf"),
        ]);
        assert!(notices.is_empty(), "{notices:?}");
        assert_eq!(datasets.len(), 2);
        match (&datasets[0], &datasets[1]) {
            (DroppedDataset::Shapefile(first), DroppedDataset::Shapefile(second)) => {
                assert_eq!(
                    first.shp.path,
                    Some(PathBuf::from("/data/region_a/roads.shp"))
                );
                assert_eq!(
                    first.dbf.as_ref().and_then(|item| item.path.clone()),
                    Some(PathBuf::from("/data/region_a/roads.dbf")),
                );
                assert!(first.prj.is_none(), "region_b's .prj is not region_a's");
                assert_eq!(
                    second.shp.path,
                    Some(PathBuf::from("/data/region_b/roads.shp"))
                );
                assert_eq!(
                    second.prj.as_ref().and_then(|item| item.path.clone()),
                    Some(PathBuf::from("/data/region_b/roads.prj")),
                );
                assert!(second.dbf.is_none());
            }
            other => panic!("expected two shapefile sets, got {other:?}"),
        }
        // A companion in a folder no `.shp` was dropped from is still an
        // orphan, not a silent attachment to a same-named set elsewhere.
        let (datasets, notices) = group_dropped_files(vec![
            dropped_path("/data/region_a/roads.shp"),
            dropped_path("/data/region_c/roads.dbf"),
        ]);
        assert_eq!(datasets.len(), 1);
        assert_eq!(notices.len(), 1);
        assert!(notices[0].contains("without its .shp"), "{}", notices[0]);
    }

    #[test]
    fn same_named_shapefiles_in_a_browser_drop_are_reported_rather_than_misgrouped() {
        // A browser hands over bare names and no directory anywhere in the
        // drop, so this collision genuinely cannot be resolved. Attaching the
        // `.dbf` to an arbitrary one of the two would be the misattribution
        // this key exists to prevent, so it is left off both and said out loud.
        let (shp, dbf) = crate::shapefile_input::fixture::cities();
        let (datasets, notices) = group_dropped_files(vec![
            dropped("roads.shp", &shp),
            dropped("roads.shp", &shp),
            dropped("roads.dbf", &dbf),
        ]);
        assert_eq!(datasets.len(), 2, "both .shp files still load");
        assert_eq!(notices.len(), 1, "{notices:?}");
        assert!(notices[0].contains("roads.shp"), "{}", notices[0]);
        assert!(notices[0].contains(".dbf"), "{}", notices[0]);
        for dataset in &datasets {
            match dataset {
                DroppedDataset::Shapefile(set) => assert!(
                    set.dbf.is_none(),
                    "neither set may claim a .dbf that could belong to the other",
                ),
                other => panic!("expected a shapefile set, got {other:?}"),
            }
        }
    }

    #[test]
    fn hydrate_gpkg_reports_the_requested_tables_own_refusal_not_a_similarly_named_ones() {
        // "roads" is a prefix of "roads_old", so a fallback that searched the
        // refusal *text* for the table name would hand the layer whose table
        // is `roads` the reason `roads_old` was left out — a CRS complaint
        // about a table the user never asked for, instead of the truth, which
        // is that `roads` is not in this file any more.
        let image = gpkg_with_one_refused_table("roads_old");
        let mut project = Project::new("test");
        let id = project.layers.add(Layer::new(
            "roads",
            LayerKind::Vector(VectorSource::LocalGpkg {
                path: "/data/x.gpkg".to_string(),
                table: "roads".to_string(),
            }),
        ));
        let mut state = LocalInputState::with_path_support(true);
        let error = state
            .hydrate_gpkg(&project, id, &image, "roads")
            .expect_err("the roads table is gone");
        assert!(
            error.message().contains("no longer in this GeoPackage"),
            "{}",
            error.message(),
        );
        assert!(
            !error.message().contains("roads_old"),
            "another table's refusal must not answer for this one: {}",
            error.message(),
        );
        // …while the table that really was refused still reports its own reason.
        let error = state
            .hydrate_gpkg(&project, id, &image, "roads_old")
            .expect_err("roads_old is in an unsupported CRS");
        assert!(error.message().contains("roads_old"), "{}", error.message());
        assert!(error.message().contains("SRS 2154"), "{}", error.message());
    }

    #[test]
    fn a_browser_geopackage_add_commits_every_table_or_none_of_them() {
        // `add_gpkg` documents "nothing is added to `project`" for every error
        // path, and the browser leg has a fallible conversion *per table*. As
        // long as that conversion runs in the same pass as the mutation, an
        // error on table N leaves tables 1..N-1 appended and queued while the
        // caller reads only the `Err` and reports that nothing was added. The
        // two passes below are what make the contract structural.
        let bytes = crate::gpkg_input::fixture::BASIC;
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(false);
        let added = state
            .add_gpkg(&mut project, "basic.gpkg", bytes, None)
            .expect("a readable GeoPackage");
        assert_eq!(added.layers.len(), 3);
        assert_eq!(project.layers.len(), 3, "all three, or none");
        assert_eq!(state.pending_op_count(), 3);
        assert_eq!(state.feature_set_count(), 3);
        for layer in &added.layers {
            assert!(project.styles.contains_key(&layer.id));
            assert!(state.feature_set(layer.id).is_some());
        }

        // The reachable failure legs, each of which must leave the project and
        // the queue exactly as it found them.
        for (name, image) in [
            (
                "weird.gpkg",
                crate::gpkg_input::fixture::WITHOUT_ROWID.to_vec(),
            ),
            (
                "notes.gpkg",
                crate::gpkg_input::fixture::attributes_only_image(),
            ),
            ("junk.gpkg", vec![0u8; 4096]),
        ] {
            let mut project = Project::new("test");
            let mut state = LocalInputState::with_path_support(false);
            let error = state
                .add_gpkg(&mut project, name, &image, None)
                .expect_err("no layer can come of this");
            assert!(!error.message().is_empty());
            assert!(project.layers.is_empty(), "{name} left a layer behind");
            assert!(project.styles.is_empty(), "{name} left a style behind");
            assert_eq!(state.pending_op_count(), 0, "{name} queued GPU work");
            assert_eq!(state.feature_set_count(), 0, "{name} kept features");
        }
    }

    /// A GeoPackage holding exactly one feature table, named `table`, that this
    /// crate refuses: it is registered in EPSG:2154, which is neither of the
    /// two CRSs the reader loads, and the file has no `gpkg_spatial_ref_sys`
    /// table to name it, so the refusal reads "SRS 2154".
    fn gpkg_with_one_refused_table(table: &str) -> Vec<u8> {
        use crate::gpkg_input::fixture::{Cell, TableSpec, multi_table_image, record};
        let table_sql = format!("CREATE TABLE {table} (fid INTEGER PRIMARY KEY, geom BLOB)");
        multi_table_image(
            4096,
            &[
                TableSpec {
                    name: "gpkg_contents",
                    sql: "CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, \
                          data_type TEXT NOT NULL)",
                    rows: &[(1, record(&[Cell::Text(table), Cell::Text("features")]))],
                },
                TableSpec {
                    name: "gpkg_geometry_columns",
                    sql: "CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, \
                          column_name TEXT NOT NULL, geometry_type_name TEXT NOT NULL, \
                          srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL)",
                    rows: &[(
                        1,
                        record(&[
                            Cell::Text(table),
                            Cell::Text("geom"),
                            Cell::Text("POINT"),
                            Cell::Int(2154),
                            Cell::Zero,
                            Cell::Zero,
                        ]),
                    )],
                },
                TableSpec {
                    name: table,
                    sql: &table_sql,
                    rows: &[(1, record(&[Cell::Null, Cell::Null]))],
                },
            ],
        )
    }

    // ---- the source CRS a layer records ------------------------------------

    #[test]
    fn a_reprojected_shapefile_records_its_source_crs_on_the_layer() {
        // The end of the seam: a JGD2011 plane-rectangular shapefile becomes a
        // layer whose coordinates are WGS 84 and whose `crs` says where they
        // came from — which is what the layer panel shows and what a reload
        // consults.
        let (shp, dbf) = crate::shapefile_input::fixture::jgd2011_zone9_cities();
        let prj = crate::shapefile_input::fixture::JGD2011_ZONE9_PRJ;
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(true);
        let added = state
            .add_shapefile(
                &mut project,
                "tokyo.shp",
                ShapefileBytes::new(&shp)
                    .with_dbf(Some(&dbf))
                    .with_sidecars(Some(prj), None),
                Some("/data/tokyo.shp"),
            )
            .expect("a JGD2011 zone IX shapefile loads");
        assert_eq!(added.feature_count, 3);

        let layer = project.layers.get(added.id).expect("present");
        assert_eq!(
            layer.crs.as_ref().map(oxigis_core::Crs::epsg),
            Some(6677),
            "the layer records the CRS its file was in",
        );
        assert_eq!(layer.source_crs().epsg(), 6677);
        assert_eq!(
            layer.source_crs().name(),
            "JGD2011 / Japan Plane Rectangular CS IX",
        );

        // The data itself is WGS 84 by the time it reaches the GPU-side layer.
        let features = state.feature_set(added.id).expect("stored");
        match features.features[1].geometry.as_ref().expect("geometry") {
            oxigeo::geojson::types::Geometry::Point(point) => {
                assert!((point.coordinates[0] - 139.7671).abs() < 1e-6);
                assert!((point.coordinates[1] - 35.6812).abs() < 1e-6);
            }
            other => panic!("expected a Point, got {other:?}"),
        }

        // And it survives a project save/load round trip.
        let json = project.to_json_string().expect("serialize");
        assert!(json.contains(r#""epsg": 6677"#), "{json}");
        let reloaded = Project::from_json_string(&json).expect("round trip");
        assert_eq!(
            reloaded
                .layers
                .layers()
                .first()
                .and_then(|layer| layer.crs.as_ref())
                .map(oxigis_core::Crs::epsg),
            Some(6677),
        );
    }

    #[test]
    fn a_wgs84_shapefile_records_no_crs_so_the_project_bytes_do_not_move() {
        let (shp, dbf) = crate::shapefile_input::fixture::cities();
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(true);
        let added = state
            .add_shapefile(
                &mut project,
                "cities.shp",
                ShapefileBytes::new(&shp).with_dbf(Some(&dbf)),
                Some("/data/cities.shp"),
            )
            .expect("valid shapefile");
        assert_eq!(project.layers.get(added.id).expect("present").crs, None);
        let json = project.to_json_string().expect("serialize");
        assert!(!json.contains("crs"), "{json}");
    }

    #[test]
    fn a_geojson_layer_records_no_crs_because_rfc_7946_fixes_it_at_wgs84() {
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(false);
        let added = state
            .add_geojson(
                &mut project,
                "a.geojson",
                r#"{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[139.7,35.7]},"properties":{}}]}"#,
                None,
            )
            .expect("valid GeoJSON");
        assert_eq!(project.layers.get(added.id).expect("present").crs, None);
    }

    #[test]
    fn a_shapefile_in_a_crs_this_build_cannot_place_adds_no_layer_at_all() {
        let (shp, dbf) = crate::shapefile_input::fixture::cities();
        let lambert = r#"PROJCS["RGF93 / Lambert-93",PROJECTION["Lambert_Conformal_Conic_2SP"],AUTHORITY["EPSG","2154"]]"#;
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(true);
        let error = state
            .add_shapefile(
                &mut project,
                "roads.shp",
                ShapefileBytes::new(&shp)
                    .with_dbf(Some(&dbf))
                    .with_sidecars(Some(lambert), None),
                Some("/data/roads.shp"),
            )
            .expect_err("Lambert-93 must be refused");
        assert!(
            error.message().contains("Lambert-93"),
            "{}",
            error.message()
        );
        assert!(error.message().contains("EPSG:2154"), "{}", error.message());
        assert!(project.layers.layers().is_empty(), "nothing was added");
    }

    #[test]
    fn each_geopackage_table_records_its_own_crs() {
        // A `.gpkg` regularly mixes CRSs across its tables, so the CRS is
        // per-layer, not per-file.
        use crate::gpkg_input::fixture::{Cell, geopackage_image, gp_blob, wkb_point};
        let blob = gp_blob(0x01, 6677, &[], &wkb_point(0.0, 0.0));
        let bytes = geopackage_image(
            Cell::Int(6677),
            "CREATE TABLE t (fid INTEGER PRIMARY KEY, geom BLOB)",
            &[Cell::Null, Cell::Blob(&blob)],
        );
        let mut project = Project::new("test");
        let mut state = LocalInputState::with_path_support(true);
        let added = state
            .add_gpkg(&mut project, "tokyo.gpkg", &bytes, Some("/data/tokyo.gpkg"))
            .expect("a zone IX GeoPackage");
        let first = added.layers.first().expect("one layer");
        assert_eq!(
            project
                .layers
                .get(first.id)
                .and_then(|layer| layer.crs.as_ref())
                .map(oxigis_core::Crs::epsg),
            Some(6677),
        );
    }
}
