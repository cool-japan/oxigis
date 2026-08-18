// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`super`] — split out of `local_vector` to keep the
//! module under the 2000-line rule; a pure move, no behaviour change.

use super::{
    DEFAULT_CIRCLE_RADIUS_PX, DEFAULT_LINE_WIDTH_PX, EXTERIOR_SHOELACE_IS_NEGATIVE, GeometryKind,
    LOCAL_EXTENT, LOCAL_LINE_LAYER_NAME, LOCAL_POINT_LAYER_NAME, LOCAL_POLYGON_LAYER_NAME,
    LocalVectorLayer, MIN_BBOX_SPAN, MercatorSquare, collection_square, convert_geometry,
    convert_properties, default_style_for, dominant_geometry_kind, family_of_layer_name,
    feature_collection_to_tile, feature_collection_to_tile_with, local_class_layer_name,
    local_layer_name, local_symbol_style, project_position, ring_shoelace,
};
use oxigeo::geojson::types::{FeatureCollection, Geometry};
use oxigis_core::{
    AttrValue, CategoryClass, Color, FillStyle, LayerStyle, LayerStyleSet, MAX_STYLE_CLASSES,
    Renderer,
};
use oxigis_render::{LabelEngine, LabelPlacer, LonLat, MapView, MvtGeometry, MvtValue, VectorMesh};

/// The rings `oxigis-render`'s own tessellator tests declare valid: this is
/// the ground truth the winding convention is anchored to.
fn render_crate_exterior() -> Vec<[i32; 2]> {
    vec![[0, 0], [0, 4096], [4096, 4096], [4096, 0]]
}

/// Likewise, the interior ring of that fixture.
fn render_crate_interior() -> Vec<[i32; 2]> {
    vec![[0, 0], [4096, 0], [4096, 4096], [0, 4096]]
}

fn parse(text: &str) -> FeatureCollection {
    match oxigeo::geojson::reader::feature_collection_from_str(text) {
        Ok(collection) => collection,
        Err(error) => panic!("the fixture must parse: {error}"),
    }
}

/// The square a one-geometry dataset would be quantised against — the
/// tests must not hand-pick a square the fixture falls outside of.
fn square_of(geometry_text: &str) -> MercatorSquare {
    let text = format!(
        r#"{{"type":"FeatureCollection","features":[{{"type":"Feature",
           "properties":{{}},"geometry":{geometry_text}}}]}}"#
    );
    collection_square(&parse(&text))
}

fn geometry(text: &str) -> Geometry {
    match oxigeo::geojson::reader::geometry_from_str(text) {
        Ok(geometry) => geometry,
        Err(error) => panic!("the fixture must parse: {error}"),
    }
}

fn view(zoom: f64) -> MapView {
    match MapView::new(LonLat::new(139.75, 35.68), zoom, [800.0, 600.0]) {
        Ok(view) => view,
        Err(error) => panic!("the view must be valid: {error}"),
    }
}

/// Sum of the (unsigned) areas of a mesh's triangles, in unit-square units.
fn mesh_area(mesh: &VectorMesh) -> f64 {
    mesh.triangles()
        .map(|[a, b, c]| {
            let cross = f64::from(b[0] - a[0]) * f64::from(c[1] - a[1])
                - f64::from(c[0] - a[0]) * f64::from(b[1] - a[1]);
            cross.abs() / 2.0
        })
        .sum()
}

// ---- projection and quantisation ------------------------------------

#[test]
fn the_prime_meridian_and_equator_land_at_the_centre_of_the_world() {
    let world = project_position(&vec![0.0, 0.0]).expect("(0, 0) projects");
    assert!((world[0] - 0.5).abs() < 1e-12);
    assert!((world[1] - 0.5).abs() < 1e-12);
}

#[test]
fn a_known_position_projects_where_the_mercator_helpers_say_it_does() {
    let expected = LonLat::new(139.7, 35.7).to_world();
    let world = project_position(&vec![139.7, 35.7]).expect("Tokyo projects");
    assert!((world[0] - expected.x).abs() < 1e-15);
    assert!((world[1] - expected.y).abs() < 1e-15);
}

#[test]
fn a_short_or_non_finite_position_is_skipped_rather_than_panicking() {
    assert!(project_position(&vec![]).is_none());
    assert!(project_position(&vec![139.7]).is_none());
    assert!(project_position(&vec![f64::NAN, 35.7]).is_none());
    assert!(project_position(&vec![139.7, f64::INFINITY]).is_none());
    // A third (altitude) element is legal and ignored.
    assert!(project_position(&vec![139.7, 35.7, 42.0]).is_some());
}

#[test]
fn a_pole_saturates_instead_of_diverging() {
    let north = project_position(&vec![0.0, 90.0]).expect("the pole must still project");
    assert!(north[1].is_finite());
    assert!(north[1] >= 0.0 && north[1] <= 1.0, "got {north:?}");
}

#[test]
fn a_single_point_gets_a_padded_square_rather_than_a_degenerate_one() {
    let collection = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},
             "geometry":{"type":"Point","coordinates":[139.7,35.7]}}]}"#,
    );
    let square = collection_square(&collection);
    assert!((square.size - MIN_BBOX_SPAN).abs() < 1e-18, "{square:?}");
    let world = LonLat::new(139.7, 35.7).to_world();
    // Centred on the point.
    assert!((square.min_x + square.size / 2.0 - world.x).abs() < 1e-15);
    assert!((square.min_y + square.size / 2.0 - world.y).abs() < 1e-15);
}

#[test]
fn a_rectangular_bbox_is_padded_to_a_square_about_its_centre() {
    // Wide and flat: 4 degrees of longitude, a sliver of latitude.
    let collection = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},"geometry":
             {"type":"LineString","coordinates":[[138.0,35.0],[142.0,35.01]]}}]}"#,
    );
    let square = collection_square(&collection);
    let west = LonLat::new(138.0, 35.0).to_world();
    let east = LonLat::new(142.0, 35.01).to_world();
    assert!(
        (square.size - (east.x - west.x)).abs() < 1e-12,
        "{square:?}"
    );
    // The dataset stays centred inside the padded square.
    let center_y = f64::midpoint(west.y, east.y);
    assert!((square.min_y + square.size / 2.0 - center_y).abs() < 1e-12);
    // And its own extent is inside.
    assert!(square.min_y <= west.y.min(east.y));
    assert!(square.max()[1] >= west.y.max(east.y));
}

#[test]
fn quantisation_round_trips_within_one_grid_step() {
    let square = MercatorSquare::from_world_bounds([0.4, 0.3], [0.6, 0.5]);
    let step = square.size / f64::from(LOCAL_EXTENT);
    for probe in [[0.4, 0.3], [0.5, 0.4], [0.6, 0.5], [0.55, 0.31]] {
        let back = square.dequantize(square.quantize(probe));
        assert!(
            (back[0] - probe[0]).abs() <= step / 2.0 + 1e-15,
            "{probe:?}"
        );
        assert!(
            (back[1] - probe[1]).abs() <= step / 2.0 + 1e-15,
            "{probe:?}"
        );
    }
}

#[test]
fn the_corners_of_the_square_land_on_the_edges_of_the_grid() {
    let square = MercatorSquare::from_world_bounds([0.25, 0.25], [0.75, 0.75]);
    let extent = LOCAL_EXTENT as i32;
    assert_eq!(square.quantize([square.min_x, square.min_y]), [0, 0]);
    assert_eq!(square.quantize(square.max()), [extent, extent]);
}

#[test]
fn a_position_outside_the_square_is_clamped_onto_the_grid() {
    let square = MercatorSquare::from_world_bounds([0.4, 0.4], [0.6, 0.6]);
    let extent = LOCAL_EXTENT as i32;
    assert_eq!(square.quantize([-10.0, -10.0]), [0, 0]);
    assert_eq!(square.quantize([10.0, 10.0]), [extent, extent]);
}

#[test]
fn a_broken_bbox_falls_back_to_the_whole_world() {
    assert_eq!(
        MercatorSquare::from_world_bounds([f64::NAN, 0.0], [1.0, 1.0]),
        MercatorSquare::world()
    );
    assert_eq!(
        MercatorSquare::from_world_bounds([0.9, 0.0], [0.1, 1.0]),
        MercatorSquare::world()
    );
    assert_eq!(
        collection_square(&parse(
            r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},"geometry":null}]}"#,
        )),
        MercatorSquare::world()
    );
}

#[test]
fn the_quantisation_error_is_the_documented_ratio() {
    assert!((MercatorSquare::quantization_error_px(65536.0) - 1.0).abs() < 1e-6);
    assert!(MercatorSquare::quantization_error_px(1024.0) < 0.02);
}

// ---- placement -------------------------------------------------------

#[test]
fn a_placement_agrees_with_the_mercator_helpers_at_several_views() {
    let collection = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},"geometry":
             {"type":"LineString","coordinates":[[139.6,35.6],[139.9,35.8]]}}]}"#,
    );
    let square = collection_square(&collection);
    let north_west = oxigis_render::WorldCoord::new(square.min_x, square.min_y).to_lon_lat();
    let south_east = oxigis_render::WorldCoord::new(square.max()[0], square.max()[1]).to_lon_lat();

    for zoom in [4.0_f64, 8.5, 13.0] {
        for center in [LonLat::new(139.75, 35.7), LonLat::new(120.0, 10.0)] {
            let view = match MapView::new(center, zoom, [800.0, 600.0]) {
                Ok(view) => view,
                Err(error) => panic!("the view must be valid: {error}"),
            };
            let placement = square.place(view);
            let expected_nw = view.lon_lat_to_screen(north_west);
            let expected_se = view.lon_lat_to_screen(south_east);
            assert!(
                (placement.x - expected_nw[0]).abs() < 0.05,
                "x at z{zoom}: {placement:?} vs {expected_nw:?}"
            );
            assert!((placement.y - expected_nw[1]).abs() < 0.05, "y at z{zoom}");
            assert!(
                (placement.x + placement.size - expected_se[0]).abs() < 0.05,
                "east edge at z{zoom}"
            );
            assert!(
                (placement.y + placement.size - expected_se[1]).abs() < 0.05,
                "south edge at z{zoom}"
            );
        }
    }
}

#[test]
fn a_placement_grows_monotonically_with_the_zoom() {
    let square = MercatorSquare::from_world_bounds([0.4, 0.4], [0.5, 0.5]);
    let small = square.place(view(6.0)).size;
    let large = square.place(view(7.0)).size;
    assert!((large / small - 2.0).abs() < 1e-3, "{small} -> {large}");
}

#[test]
fn fit_zoom_makes_the_square_fill_the_viewport() {
    let square = MercatorSquare::from_world_bounds([0.4, 0.4], [0.5, 0.5]);
    let size_px = [800.0_f32, 600.0];
    let zoom = square.fit_zoom(size_px, 0.0);
    let fitted = square.place(view(zoom).with_zoom(zoom));
    assert!(
        (fitted.size - f64::from(size_px[1]) as f32).abs() < 1.0,
        "{fitted:?}"
    );
    // With a margin it is strictly smaller.
    assert!(square.fit_zoom(size_px, 0.1) < zoom);
}

// ---- winding ---------------------------------------------------------

#[test]
fn the_winding_convention_matches_the_render_crate_fixtures() {
    assert!(
        ring_shoelace(&render_crate_exterior()) < 0.0,
        "an MVT exterior must have a negative raw shoelace sum",
    );
    assert!(ring_shoelace(&render_crate_interior()) > 0.0);
    const { assert!(EXTERIOR_SHOELACE_IS_NEGATIVE) };
    // A ring that cannot bound an area has no orientation at all.
    assert_eq!(ring_shoelace(&[[0, 0], [1, 1]]), 0.0);
}

#[test]
fn a_projected_rectangle_comes_out_wound_like_an_exterior_ring() {
    // Deliberately clockwise in lon/lat, i.e. the *wrong* way round per
    // RFC 7946 — normalisation must fix it.
    const CLOCKWISE: &str = r#"{"type":"Polygon","coordinates":[[[138.0,35.0],
        [139.0,35.0],[139.0,36.0],[138.0,36.0],[138.0,35.0]]]}"#;
    const COUNTER: &str = r#"{"type":"Polygon","coordinates":[[[138.0,35.0],
        [138.0,36.0],[139.0,36.0],[139.0,35.0],[138.0,35.0]]]}"#;
    let square = square_of(CLOCKWISE);
    let clockwise = geometry(CLOCKWISE);
    let Some(MvtGeometry::Polygons(polygons)) = convert_geometry(&clockwise, &square) else {
        panic!("a polygon must convert to polygons");
    };
    assert!(ring_shoelace(&polygons[0].exterior) < 0.0);

    // And the counter-clockwise spelling of the same ring agrees.
    let counter = geometry(COUNTER);
    let Some(MvtGeometry::Polygons(other)) = convert_geometry(&counter, &square_of(COUNTER)) else {
        panic!("a polygon must convert to polygons");
    };
    assert!(ring_shoelace(&other[0].exterior) < 0.0);
}

#[test]
fn a_hole_is_wound_against_its_exterior_and_stays_empty_in_the_mesh() {
    let collection = parse(
        r#"{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},
            "geometry":{"type":"Polygon","coordinates":[
              [[138.0,35.0],[140.0,35.0],[140.0,37.0],[138.0,37.0],[138.0,35.0]],
              [[138.5,35.5],[138.5,36.5],[139.5,36.5],[139.5,35.5],[138.5,35.5]]]}}]}"#,
    );
    let square = collection_square(&collection);
    let tile = feature_collection_to_tile(&collection, &square);
    let Some(layer) = tile.layers.first() else {
        panic!("one layer");
    };
    let Some(feature) = layer.features.first() else {
        panic!("one feature");
    };
    let MvtGeometry::Polygons(polygons) = &feature.geometry else {
        panic!("polygons");
    };
    let polygon = &polygons[0];
    assert_eq!(polygon.interiors.len(), 1, "the hole must survive");
    assert!(ring_shoelace(&polygon.exterior) < 0.0, "exterior winding");
    assert!(ring_shoelace(&polygon.interiors[0]) > 0.0, "hole winding");

    // The tessellated area must be the ring's, minus the hole's. Both are
    // measured on the very grid the mesh was built from, then expressed in
    // the unit square the mesh's vertices live in.
    let extent = f64::from(LOCAL_EXTENT);
    let expected = (ring_shoelace(&polygon.exterior).abs()
        - ring_shoelace(&polygon.interiors[0]).abs())
        / 2.0
        / (extent * extent);

    let mut fill_only = LocalVectorLayer::from_feature_collection("holes", collection);
    // A plain fill, so no outline pass adds area of its own.
    fill_only.set_style(LayerStyle::Fill(oxigis_core::FillStyle::new(
        oxigis_core::Color::from_rgb(0xff, 0xff, 0xff),
    )));
    let mesh = match fill_only.tessellate(512.0) {
        Ok(mesh) => mesh,
        Err(error) => panic!("tessellation must succeed: {error}"),
    };
    let area = mesh_area(&mesh);
    assert!(
        (area - expected).abs() < expected * 0.01,
        "mesh area {area} should be {expected}",
    );
    // And the hole is genuinely empty.
    assert!(!mesh.covers([0.5, 0.5]), "the hole was filled in");
}

// ---- geometry conversion --------------------------------------------

#[test]
fn every_geojson_geometry_maps_onto_the_matching_mvt_variant() {
    /// A case: the GeoJSON text, and the shape its conversion must take.
    type Case = (&'static str, fn(&MvtGeometry) -> bool);
    let cases: [Case; 6] = [
        (
            r#"{"type":"Point","coordinates":[139.0,35.0]}"#,
            |g| matches!(g, MvtGeometry::Points(points) if points.len() == 1),
        ),
        (
            r#"{"type":"MultiPoint","coordinates":[[139.0,35.0],[139.5,35.5]]}"#,
            |g| matches!(g, MvtGeometry::Points(points) if points.len() == 2),
        ),
        (
            r#"{"type":"LineString","coordinates":[[139.0,35.0],[139.5,35.5]]}"#,
            |g| matches!(g, MvtGeometry::Lines(lines) if lines.len() == 1),
        ),
        (
            r#"{"type":"MultiLineString","coordinates":[
                [[139.0,35.0],[139.5,35.5]],[[138.0,34.0],[138.5,34.5]]]}"#,
            |g| matches!(g, MvtGeometry::Lines(lines) if lines.len() == 2),
        ),
        (
            r#"{"type":"Polygon","coordinates":[[[138.0,35.0],[139.0,35.0],
                [139.0,36.0],[138.0,35.0]]]}"#,
            |g| matches!(g, MvtGeometry::Polygons(polygons) if polygons.len() == 1),
        ),
        (
            r#"{"type":"MultiPolygon","coordinates":[
                [[[138.0,35.0],[139.0,35.0],[139.0,36.0],[138.0,35.0]]],
                [[[136.0,33.0],[137.0,33.0],[137.0,34.0],[136.0,33.0]]]]}"#,
            // Multipolygons stay one feature: `MvtGeometry` is already multi.
            |g| matches!(g, MvtGeometry::Polygons(polygons) if polygons.len() == 2),
        ),
    ];
    for (text, check) in cases {
        let converted =
            convert_geometry(&geometry(text), &square_of(text)).expect("the geometry must convert");
        assert!(check(&converted), "wrong variant for {text}: {converted:?}");
    }
}

#[test]
fn a_closed_ring_loses_exactly_its_closing_vertex() {
    const CLOSED: &str = r#"{"type":"Polygon","coordinates":[[[138.0,35.0],
        [139.0,35.0],[139.0,36.0],[138.0,36.0],[138.0,35.0]]]}"#;
    let Some(MvtGeometry::Polygons(polygons)) =
        convert_geometry(&geometry(CLOSED), &square_of(CLOSED))
    else {
        panic!("polygons");
    };
    assert_eq!(
        polygons[0].exterior.len(),
        4,
        "five source vertices, four kept"
    );
    assert_ne!(
        polygons[0].exterior.first(),
        polygons[0].exterior.last(),
        "MVT rings are unclosed",
    );
}

#[test]
fn degenerate_geometry_is_skipped_rather_than_drawn() {
    for text in [
        // A line whose two vertices quantise onto the same grid point.
        r#"{"type":"LineString","coordinates":[[139.0,35.0],[139.0,35.0]]}"#,
        // A ring with only two distinct vertices bounds no area.
        r#"{"type":"Polygon","coordinates":[[[138.0,35.0],[139.0,35.0],[138.0,35.0]]]}"#,
        // A multipoint with no usable position at all.
        r#"{"type":"MultiPoint","coordinates":[[139.0]]}"#,
        // An empty line.
        r#"{"type":"LineString","coordinates":[]}"#,
    ] {
        assert!(
            convert_geometry(&geometry(text), &square_of(text)).is_none(),
            "{text} must not be drawn",
        );
    }
}

#[test]
fn a_uniform_geometry_collection_still_flattens_to_one_geometry() {
    const UNIFORM: &str = r#"{"type":"GeometryCollection","geometries":[
        {"type":"Point","coordinates":[139.0,35.0]},
        {"type":"MultiPoint","coordinates":[[139.5,35.5],[139.6,35.6]]}]}"#;
    const MIXED: &str = r#"{"type":"GeometryCollection","geometries":[
        {"type":"Point","coordinates":[139.0,35.0]},
        {"type":"LineString","coordinates":[[139.0,35.0],[139.5,35.5]]}]}"#;
    let Some(MvtGeometry::Points(points)) =
        convert_geometry(&geometry(UNIFORM), &square_of(UNIFORM))
    else {
        panic!("a uniform collection must flatten to points");
    };
    assert_eq!(points.len(), 3);

    // One `MvtGeometry` cannot name two families; the FEATURE converter
    // splits a mixed collection instead (the test below).
    assert!(convert_geometry(&geometry(MIXED), &square_of(MIXED)).is_none());
}

#[test]
fn a_mixed_collection_splits_into_per_family_features_sharing_the_id() {
    let collection = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"name":"gc"},
             "geometry":{"type":"GeometryCollection","geometries":[
               {"type":"Point","coordinates":[139.0,35.0]},
               {"type":"LineString","coordinates":[[139.0,35.0],[139.5,35.5]]},
               {"type":"Polygon","coordinates":[[[138.0,35.0],[139.0,35.0],
                 [139.0,36.0],[138.0,35.0]]]}]}}]}"#,
    );
    let square = collection_square(&collection);
    let tile = feature_collection_to_tile(&collection, &square);
    // Since v1.3 each family is its OWN named layer (the renderer keys
    // styles by layer name), polygons -> lines -> points in painter's
    // order, every split feature keeping the source index as its id.
    assert_eq!(
        tile.layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            LOCAL_POLYGON_LAYER_NAME,
            LOCAL_LINE_LAYER_NAME,
            LOCAL_POINT_LAYER_NAME
        ],
        "one layer per geometry family, so every family can draw"
    );
    assert!(
        tile.layers
            .iter()
            .flat_map(|layer| layer.features.iter())
            .all(|feature| feature.id == Some(0)),
        "every split feature keeps the source index as its id",
    );
    assert!(
        matches!(
            tile.layers[0].features[0].geometry,
            MvtGeometry::Polygons(_)
        ) && matches!(tile.layers[1].features[0].geometry, MvtGeometry::Lines(_))
            && matches!(tile.layers[2].features[0].geometry, MvtGeometry::Points(_)),
    );
    // `geometry_kind` reports the collection's first member's family (the
    // point), so that split feature — and only that one — carries the
    // properties: a Symbol rule anchors ONE label per source feature.
    let carrying: Vec<bool> = tile
        .layers
        .iter()
        .flat_map(|layer| layer.features.iter())
        .map(|feature| !feature.properties.is_empty())
        .collect();
    assert_eq!(carrying, [false, false, true]);
}

#[test]
fn the_properties_fall_back_to_the_first_family_that_survived() {
    // The leading member is a degenerate polygon (two distinct vertices
    // bound no area), so the collection's preferred family quantises away
    // and the properties ride the first emitted feature instead of
    // vanishing with it.
    let collection = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"name":"survivor"},
             "geometry":{"type":"GeometryCollection","geometries":[
               {"type":"Polygon","coordinates":[[[138.0,35.0],[139.0,35.0],
                 [138.0,35.0]]]},
               {"type":"LineString","coordinates":[[139.0,35.0],[139.5,35.5]]}]}}]}"#,
    );
    let square = collection_square(&collection);
    let tile = feature_collection_to_tile(&collection, &square);
    let Some(layer) = tile.layers.first() else {
        panic!("one layer");
    };
    assert_eq!(layer.features.len(), 1, "only the line survives");
    assert!(
        matches!(layer.features[0].geometry, MvtGeometry::Lines(_)),
        "the surviving family is the line",
    );
    assert!(
        !layer.features[0].properties.is_empty(),
        "the properties must not vanish with the degenerate family",
    );
}

#[test]
fn nesting_deeper_than_the_editing_caps_is_dropped_identically() {
    const POINT: &str = r#"{"type":"Point","coordinates":[139.0,35.0]}"#;
    let nested = |levels: usize| {
        let mut text = POINT.to_owned();
        for _ in 0..levels {
            text = format!(r#"{{"type":"GeometryCollection","geometries":[{text}]}}"#);
        }
        text
    };
    let square = square_of(POINT);
    // The editing path's walkers skip a collection ENTERED at depth 8, so
    // eight nested collections still draw and nine do not — what is drawn
    // stays exactly what is editable.
    assert!(convert_geometry(&geometry(&nested(8)), &square).is_some());
    assert!(convert_geometry(&geometry(&nested(9)), &square).is_none());
}

#[test]
fn a_geometry_collection_contributes_its_vertices_to_the_bbox() {
    let collection = parse(
        r#"{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},
            "geometry":{"type":"GeometryCollection","geometries":[
              {"type":"Point","coordinates":[138.0,35.0]},
              {"type":"Point","coordinates":[140.0,37.0]}]}}]}"#,
    );
    let square = collection_square(&collection);
    let west = LonLat::new(138.0, 35.0).to_world();
    let east = LonLat::new(140.0, 37.0).to_world();
    assert!(square.min_x <= west.x + 1e-12);
    assert!(square.max()[0] >= east.x - 1e-12);
}

#[test]
fn features_keep_their_index_as_an_id_and_undrawable_ones_are_dropped() {
    let collection = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},"geometry":null},
            {"type":"Feature","properties":{},
             "geometry":{"type":"Point","coordinates":[139.0,35.0]}},
            {"type":"Feature","properties":{},
             "geometry":{"type":"Point","coordinates":[139.5,35.5]}}]}"#,
    );
    let square = collection_square(&collection);
    let tile = feature_collection_to_tile(&collection, &square);
    let Some(layer) = tile.layers.first() else {
        panic!("one layer");
    };
    // A single-family dataset stays a ONE-layer tile, named for its family
    // (v1.3: the renderer keys styles by layer name).
    assert_eq!(tile.layers.len(), 1);
    assert_eq!(layer.name, LOCAL_POINT_LAYER_NAME);
    assert_eq!(layer.extent, LOCAL_EXTENT);
    assert_eq!(layer.features.len(), 2, "the null geometry is dropped");
    // The ids are source indices, so the geometry-less feature's is missing.
    assert_eq!(layer.features[0].id, Some(1));
    assert_eq!(layer.features[1].id, Some(2));
}

// ---- properties ------------------------------------------------------

#[test]
fn properties_map_onto_the_matching_mvt_values() {
    let collection = parse(
        r#"{"type":"FeatureCollection","features":[{"type":"Feature","geometry":
            {"type":"Point","coordinates":[139.0,35.0]},"properties":{
              "name":"Tokyo","population":13960000,"density":6349.5,
              "capital":true,"nickname":null,
              "tags":["big","busy"],"meta":{"iso":"JP"}}}]}"#,
    );
    let Some(feature) = collection.features.first() else {
        panic!("one feature");
    };
    let properties = convert_properties(feature.properties.as_ref());
    let get = |key: &str| {
        properties
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
    };
    assert_eq!(get("name"), Some(MvtValue::String("Tokyo".to_owned())));
    assert_eq!(get("population"), Some(MvtValue::I64(13_960_000)));
    assert_eq!(get("density"), Some(MvtValue::F64(6349.5)));
    assert_eq!(get("capital"), Some(MvtValue::Bool(true)));
    assert_eq!(get("nickname"), None, "null is dropped, not stored");
    assert_eq!(
        get("tags"),
        Some(MvtValue::String(r#"["big","busy"]"#.to_owned())),
    );
    assert_eq!(
        get("meta"),
        Some(MvtValue::String(r#"{"iso":"JP"}"#.to_owned())),
    );
}

#[test]
fn a_feature_without_properties_yields_an_empty_list() {
    assert!(convert_properties(None).is_empty());
}

// ---- default styles --------------------------------------------------

#[test]
fn the_dominant_geometry_kind_picks_the_default_style() {
    let polygons = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},"geometry":{"type":"Polygon",
             "coordinates":[[[138.0,35.0],[139.0,35.0],[139.0,36.0],[138.0,35.0]]]}},
            {"type":"Feature","properties":{},"geometry":
             {"type":"Point","coordinates":[139.0,35.0]}}]}"#,
    );
    assert_eq!(dominant_geometry_kind(&polygons), GeometryKind::Polygon);
    let LayerStyle::Fill(fill) = default_style_for(&polygons) else {
        panic!("polygons must default to a fill");
    };
    assert!(fill.outline_color.is_some(), "with an outline");
    assert!(fill.opacity() < 1.0, "and semi-transparent");

    let lines = parse(
        r#"{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},
            "geometry":{"type":"LineString","coordinates":[[139.0,35.0],[139.5,35.5]]}}]}"#,
    );
    assert_eq!(dominant_geometry_kind(&lines), GeometryKind::Line);
    let LayerStyle::Line(line) = default_style_for(&lines) else {
        panic!("lines must default to a stroke");
    };
    assert_eq!(line.width(), DEFAULT_LINE_WIDTH_PX);

    let points = parse(
        r#"{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},
            "geometry":{"type":"Point","coordinates":[139.0,35.0]}}]}"#,
    );
    assert_eq!(dominant_geometry_kind(&points), GeometryKind::Point);
    let LayerStyle::Circle(circle) = default_style_for(&points) else {
        panic!("points must default to a circle");
    };
    assert_eq!(circle.radius(), DEFAULT_CIRCLE_RADIUS_PX);
    assert!(circle.stroke_color.is_some());
}

#[test]
fn a_collection_without_geometry_still_gets_a_style() {
    let empty = parse(r#"{"type":"FeatureCollection","features":[]}"#);
    assert_eq!(dominant_geometry_kind(&empty), GeometryKind::Point);
    assert!(matches!(default_style_for(&empty), LayerStyle::Circle(_)));
}

// ---- the layer, end to end ------------------------------------------

#[test]
fn geojson_text_becomes_a_drawable_layer() {
    let layer = match LocalVectorLayer::from_geojson(
        "cities",
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"name":"Tokyo"},
             "geometry":{"type":"Point","coordinates":[139.7,35.7]}},
            {"type":"Feature","properties":{"name":"Osaka"},
             "geometry":{"type":"Point","coordinates":[135.5,34.7]}}]}"#,
    ) {
        Ok(layer) => layer,
        Err(error) => panic!("the fixture must load: {error}"),
    };
    assert_eq!(layer.name(), "cities");
    assert_eq!(layer.feature_count(), 2);
    assert_eq!(layer.drawable_count(), 2);
    assert!(layer.visible());
    assert_eq!(layer.opacity(), 1.0);
    assert!(matches!(layer.style().base(), LayerStyle::Circle(_)));
    assert!(layer.labels().is_empty(), "no symbol rule by default");

    let mesh = match layer.tessellate(512.0) {
        Ok(mesh) => mesh,
        Err(error) => panic!("tessellation must succeed: {error}"),
    };
    assert!(!mesh.is_empty(), "two circles must produce triangles");
}

#[test]
fn a_polygon_layer_tessellates_both_the_fill_and_the_outline() {
    let layer = match LocalVectorLayer::from_geojson(
        "wards",
        r#"{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},
            "geometry":{"type":"Polygon","coordinates":[[[139.6,35.6],[139.8,35.6],
              [139.8,35.8],[139.6,35.8],[139.6,35.6]]]}}]}"#,
    ) {
        Ok(layer) => layer,
        Err(error) => panic!("the fixture must load: {error}"),
    };
    assert_eq!(layer.program().passes().len(), 2, "fill, then outline");
    let mesh = match layer.tessellate(512.0) {
        Ok(mesh) => mesh,
        Err(error) => panic!("tessellation must succeed: {error}"),
    };
    assert!(mesh.covers([0.5, 0.5]), "the polygon must cover its centre");
}

#[test]
fn bad_input_is_reported_rather_than_panicking() {
    assert!(LocalVectorLayer::from_geojson("x", "not json").is_err());
    assert!(LocalVectorLayer::from_geojson("x", r#"{"type":"Feature"}"#).is_err());
    let empty =
        LocalVectorLayer::from_geojson("x", r#"{"type":"FeatureCollection","features":[]}"#);
    assert!(empty.is_err(), "an empty collection is a failed load");
}

#[test]
fn a_non_finite_placement_size_is_a_tessellation_error_not_a_panic() {
    let layer = match LocalVectorLayer::from_geojson(
        "points",
        r#"{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},
            "geometry":{"type":"Point","coordinates":[139.7,35.7]}}]}"#,
    ) {
        Ok(layer) => layer,
        Err(error) => panic!("the fixture must load: {error}"),
    };
    assert!(layer.tessellate(f32::NAN).is_err());
    assert!(layer.tessellate(0.0).is_err());
}

#[test]
fn restyling_recompiles_the_passes_and_bumps_the_generation() {
    let mut layer = match LocalVectorLayer::from_geojson(
        "points",
        r#"{"type":"FeatureCollection","features":[{"type":"Feature",
            "properties":{"name":"Tokyo"},
            "geometry":{"type":"Point","coordinates":[139.7,35.7]}}]}"#,
    ) {
        Ok(layer) => layer,
        Err(error) => panic!("the fixture must load: {error}"),
    };
    let before = layer.generation();
    assert!(!layer.program().is_empty());
    assert!(layer.labels().is_empty());

    layer.set_style(local_symbol_style("name"));
    assert_eq!(layer.generation(), before + 1);
    assert!(
        layer.program().is_empty(),
        "a symbol style draws no geometry",
    );
    // One label spec per FAMILY layer name since v1.3 (the compile emits a
    // rule per family); the properties carrier still anchors exactly one
    // label per source feature.
    assert_eq!(layer.labels().len(), 3);
    // Renaming leaves the style matching, because rules key on the constants.
    layer.set_name("renamed");
    assert_eq!(layer.name(), "renamed");
    assert_eq!(layer.labels().len(), 3);
}

#[test]
fn a_symbol_style_places_a_label_through_the_shared_placer() {
    let mut layer = match LocalVectorLayer::from_geojson(
        "cities",
        r#"{"type":"FeatureCollection","features":[{"type":"Feature",
            "properties":{"name":"Tokyo"},
            "geometry":{"type":"Point","coordinates":[139.7,35.7]}}]}"#,
    ) {
        Ok(layer) => layer,
        Err(error) => panic!("the fixture must load: {error}"),
    };
    layer.set_style(local_symbol_style("name"));

    let mut engine = match LabelEngine::new(oxifont_bundled::NOTO_SANS_REGULAR.to_vec()) {
        Ok(engine) => engine,
        Err(error) => panic!("the bundled Noto Sans must parse: {error}"),
    };
    let view = view(10.0);
    let placement = layer.place(view);
    let mut placer = LabelPlacer::new(view.size_px());
    if let Err(error) = placer.place_tile(&mut engine, layer.tile(), &placement, layer.labels()) {
        panic!("placement must succeed: {error}");
    }
    assert!(!placer.is_stale());
    let placed = placer.finish();
    assert_eq!(placed.len(), 1, "the one city must be labelled");
}

#[test]
fn a_label_lands_where_the_feature_does() {
    let mut layer = match LocalVectorLayer::from_geojson(
        "cities",
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"name":"West"},
             "geometry":{"type":"Point","coordinates":[139.0,35.0]}},
            {"type":"Feature","properties":{"name":"East"},
             "geometry":{"type":"Point","coordinates":[140.0,35.0]}}]}"#,
    ) {
        Ok(layer) => layer,
        Err(error) => panic!("the fixture must load: {error}"),
    };
    layer.set_style(local_symbol_style("name"));
    let mut engine = match LabelEngine::new(oxifont_bundled::NOTO_SANS_REGULAR.to_vec()) {
        Ok(engine) => engine,
        Err(error) => panic!("the bundled Noto Sans must parse: {error}"),
    };
    let view = match MapView::new(LonLat::new(139.5, 35.0), 8.0, [1200.0, 800.0]) {
        Ok(view) => view,
        Err(error) => panic!("the view must be valid: {error}"),
    };
    let placement = layer.place(view);
    let mut placer = LabelPlacer::new(view.size_px());
    if let Err(error) = placer.place_tile(&mut engine, layer.tile(), &placement, layer.labels()) {
        panic!("placement must succeed: {error}");
    }
    let placed = placer.finish();
    assert_eq!(placed.len(), 2);
    // Both labels sit within a few pixels of their feature's screen position.
    for (lon, index) in [(139.0_f64, 0_usize), (140.0, 1)] {
        let expected = view.lon_lat_to_screen(LonLat::new(lon, 35.0));
        let box_center_x = f64::from(
            (placed[index].collision_box.min_x + placed[index].collision_box.max_x) / 2.0,
        );
        assert!(
            (box_center_x - f64::from(expected[0])).abs() < 4.0,
            "label {index} at {box_center_x} should be near {}",
            expected[0],
        );
    }
}

// ---- per-family styling (tiles v1.3 item C) --------------------------

#[test]
fn default_style_set_for_seeds_only_the_non_dominant_families() {
    // Mixed: polygons dominate; lines and points get default overrides so
    // the dataset draws ALL of itself on first sight.
    let mixed = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},
             "geometry":{"type":"Polygon","coordinates":[[[0.0,0.0],[1.0,0.0],
               [1.0,1.0],[0.0,0.0]]]}},
            {"type":"Feature","properties":{},
             "geometry":{"type":"Polygon","coordinates":[[[2.0,0.0],[3.0,0.0],
               [3.0,1.0],[2.0,0.0]]]}},
            {"type":"Feature","properties":{},
             "geometry":{"type":"LineString","coordinates":[[0.0,0.0],[1.0,1.0]]}},
            {"type":"Feature","properties":{},
             "geometry":{"type":"Point","coordinates":[0.5,0.5]}}]}"#,
    );
    let set = super::default_style_set_for(&mixed);
    assert!(
        matches!(set.base(), LayerStyle::Fill(_)),
        "polygons dominate"
    );
    assert!(set.override_for(GeometryKind::Polygon).is_none());
    assert!(matches!(
        set.override_for(GeometryKind::Line),
        Some(LayerStyle::Line(_))
    ));
    assert!(matches!(
        set.override_for(GeometryKind::Point),
        Some(LayerStyle::Circle(_))
    ));

    // Single family: a base-only set, byte-identical to the old default.
    let single = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},
             "geometry":{"type":"Point","coordinates":[0.5,0.5]}}]}"#,
    );
    let set = super::default_style_set_for(&single);
    assert!(!set.has_overrides());
    assert_eq!(set.base(), &default_style_for(&single));
}

#[test]
fn a_point_override_draws_circles_a_fill_base_never_did() {
    // A mixed layer whose base is Fill: pre-v1.3 the points were invisible
    // (the dispatch table had no (Points, Fill) arm). A Circle override on
    // the point family makes them draw.
    let geojson = r#"{"type":"FeatureCollection","features":[
        {"type":"Feature","properties":{},
         "geometry":{"type":"Polygon","coordinates":[[[10.0,10.0],[20.0,10.0],
           [20.0,20.0],[10.0,10.0]]]}},
        {"type":"Feature","properties":{},
         "geometry":{"type":"Point","coordinates":[15.0,14.0]}}]}"#;
    let mut layer = match LocalVectorLayer::from_geojson("mixed", geojson) {
        Ok(layer) => layer,
        Err(error) => panic!("the fixture must load: {error}"),
    };
    // Force the OLD single-style look: a bare Fill base, no overrides.
    let mut fill = oxigis_core::FillStyle::new(oxigis_core::Color::from_rgb(10, 20, 30));
    fill.outline_color = None;
    layer.set_style(LayerStyle::Fill(fill));
    let before = layer
        .tessellate(512.0)
        .expect("the mixed layer tessellates");

    let mut set = oxigis_core::LayerStyleSet::new(LayerStyle::Fill(fill));
    set.set_override(
        GeometryKind::Point,
        LayerStyle::Circle(oxigis_core::CircleStyle::new(
            4.0,
            oxigis_core::Color::from_rgb(200, 0, 0),
        )),
    );
    layer.set_style(set);
    let after = layer
        .tessellate(512.0)
        .expect("the overridden layer tessellates");
    assert!(
        after.vertices.len() > before.vertices.len(),
        "the circle override adds geometry a fill base never drew          ({} -> {} vertices)",
        before.vertices.len(),
        after.vertices.len(),
    );
    assert_eq!(
        layer.families().len(),
        2,
        "the tile reports both families, so the panel offers the row"
    );
}

// ---- categorized / graduated renderers (thematic v1.6) ---------------

/// Four polygons in two categories, one of which is unclassifiable.
fn thematic_geojson() -> &'static str {
    r#"{"type":"FeatureCollection","features":[
        {"type":"Feature","properties":{"zone":"a","pop":10},
         "geometry":{"type":"Polygon","coordinates":[[[0.0,0.0],[1.0,0.0],
           [1.0,1.0],[0.0,0.0]]]}},
        {"type":"Feature","properties":{"zone":"b","pop":20},
         "geometry":{"type":"Polygon","coordinates":[[[2.0,0.0],[3.0,0.0],
           [3.0,1.0],[2.0,0.0]]]}},
        {"type":"Feature","properties":{"zone":"a","pop":30},
         "geometry":{"type":"Polygon","coordinates":[[[4.0,0.0],[5.0,0.0],
           [5.0,1.0],[4.0,0.0]]]}},
        {"type":"Feature","properties":{"pop":40},
         "geometry":{"type":"Polygon","coordinates":[[[6.0,0.0],[7.0,0.0],
           [7.0,1.0],[6.0,0.0]]]}}]}"#
}

fn thematic_layer() -> LocalVectorLayer {
    match LocalVectorLayer::from_geojson("zones", thematic_geojson()) {
        Ok(layer) => layer,
        Err(error) => panic!("the fixture must load: {error}"),
    }
}

/// A categorized renderer over `zone` with `a` and `b` as its two classes.
fn zone_renderer() -> Renderer {
    Renderer::categorized(
        "zone",
        [
            CategoryClass::new(
                AttrValue::text("a"),
                LayerStyle::Fill(FillStyle::new(Color::from_rgb(200, 0, 0))),
            ),
            CategoryClass::new(
                AttrValue::text("b"),
                LayerStyle::Fill(FillStyle::new(Color::from_rgb(0, 0, 200))),
            ),
        ],
        None,
    )
}

fn layer_names(tile: &oxigis_render::VectorTile) -> Vec<String> {
    tile.layers.iter().map(|layer| layer.name.clone()).collect()
}

#[test]
fn a_single_symbol_dataset_builds_exactly_the_tile_it_always_did() {
    // The floor the whole feature stands on: with no renderer, the partition
    // is the v1.3 one — same names, same order, same features — so a
    // single-symbol layer's mesh cannot have moved.
    let features = parse(thematic_geojson());
    let square = collection_square(&features);
    let plain = feature_collection_to_tile(&features, &square);
    let classified = feature_collection_to_tile_with(&features, &square, &Renderer::Single);
    assert_eq!(layer_names(&plain), vec![LOCAL_POLYGON_LAYER_NAME]);
    assert_eq!(layer_names(&classified), layer_names(&plain));
    assert_eq!(classified.layers, plain.layers, "feature for feature");
}

#[test]
fn a_categorized_dataset_partitions_into_one_layer_per_class() {
    let features = parse(thematic_geojson());
    let square = collection_square(&features);
    let tile = feature_collection_to_tile_with(&features, &square, &zone_renderer());
    assert_eq!(
        layer_names(&tile),
        vec![
            LOCAL_POLYGON_LAYER_NAME.to_string(),
            local_class_layer_name(GeometryKind::Polygon, 0),
            local_class_layer_name(GeometryKind::Polygon, 1),
        ],
        "the fallback paints first, then the classes in legend order",
    );
    let counts: Vec<usize> = tile
        .layers
        .iter()
        .map(|layer| layer.features.len())
        .collect();
    assert_eq!(
        counts,
        vec![1, 2, 1],
        "one unclassified feature, two in class a, one in class b",
    );
    // Ids survive the partition, so a table row still addresses its feature.
    let ids: Vec<Option<u64>> = tile
        .layers
        .iter()
        .flat_map(|layer| layer.features.iter().map(|feature| feature.id))
        .collect();
    assert_eq!(ids, vec![Some(3), Some(0), Some(2), Some(1)]);
    for layer in &tile.layers {
        assert_eq!(layer.extent, LOCAL_EXTENT);
        assert_eq!(
            family_of_layer_name(&layer.name),
            Some(GeometryKind::Polygon)
        );
    }
}

#[test]
fn a_class_that_no_feature_lands_in_is_omitted_like_an_empty_family() {
    let features = parse(thematic_geojson());
    let square = collection_square(&features);
    let renderer = Renderer::categorized(
        "zone",
        [
            CategoryClass::new(
                AttrValue::text("nobody"),
                LayerStyle::Fill(FillStyle::new(Color::BLACK)),
            ),
            CategoryClass::new(
                AttrValue::text("b"),
                LayerStyle::Fill(FillStyle::new(Color::WHITE)),
            ),
        ],
        None,
    );
    let tile = feature_collection_to_tile_with(&features, &square, &renderer);
    assert_eq!(
        layer_names(&tile),
        vec![
            LOCAL_POLYGON_LAYER_NAME.to_string(),
            local_class_layer_name(GeometryKind::Polygon, 1),
        ],
        "class 0 drew nothing, so it costs no layer and no mesh",
    );
}

#[test]
fn a_mixed_dataset_keeps_families_apart_inside_every_class() {
    let mixed = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"zone":"a"},
             "geometry":{"type":"Polygon","coordinates":[[[0.0,0.0],[1.0,0.0],
               [1.0,1.0],[0.0,0.0]]]}},
            {"type":"Feature","properties":{"zone":"a"},
             "geometry":{"type":"Point","coordinates":[0.5,0.5]}},
            {"type":"Feature","properties":{"zone":"b"},
             "geometry":{"type":"LineString","coordinates":[[0.0,0.0],[1.0,1.0]]}}]}"#,
    );
    let square = collection_square(&mixed);
    let tile = feature_collection_to_tile_with(&mixed, &square, &zone_renderer());
    assert_eq!(
        layer_names(&tile),
        vec![
            local_class_layer_name(GeometryKind::Polygon, 0),
            local_class_layer_name(GeometryKind::Line, 1),
            local_class_layer_name(GeometryKind::Point, 0),
        ],
        "painter's order across families, class buckets within them",
    );
}

#[test]
fn one_source_feature_is_classified_once_for_every_family_it_draws() {
    // A mixed GeometryCollection must not be half red and half blue: the
    // class is resolved from the source feature, not per emitted geometry.
    let collection = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"zone":"b"},
             "geometry":{"type":"GeometryCollection","geometries":[
               {"type":"Polygon","coordinates":[[[0.0,0.0],[1.0,0.0],
                 [1.0,1.0],[0.0,0.0]]]},
               {"type":"Point","coordinates":[0.5,0.5]}]}}]}"#,
    );
    let square = collection_square(&collection);
    let tile = feature_collection_to_tile_with(&collection, &square, &zone_renderer());
    assert_eq!(
        layer_names(&tile),
        vec![
            local_class_layer_name(GeometryKind::Polygon, 1),
            local_class_layer_name(GeometryKind::Point, 1),
        ],
        "both families in class 1",
    );
}

#[test]
fn the_class_cap_bounds_how_many_meshes_one_layer_can_cost() {
    // A hostile (or hand-edited) project file naming more classes than the
    // model resolves must not turn into more layers than it resolves either.
    // One feature per would-be class, `code` = the class index it names, so
    // the ones past the cap have nowhere to go but the fallback.
    let over = MAX_STYLE_CLASSES + 10;
    let mut text = String::from(r#"{"type":"FeatureCollection","features":["#);
    for code in 0..over {
        if code > 0 {
            text.push(',');
        }
        let x = code as f64;
        text.push_str(&format!(
            r#"{{"type":"Feature","properties":{{"code":{code}}},
               "geometry":{{"type":"Polygon","coordinates":[[[{x}.0,0.0],[{x}.5,0.0],
                 [{x}.5,1.0],[{x}.0,0.0]]]}}}}"#
        ));
    }
    text.push_str("]}");
    let features = parse(&text);
    let square = collection_square(&features);
    let categories = (0..over).map(|index| {
        CategoryClass::new(
            AttrValue::number(index as f64).unwrap_or(AttrValue::Bool(false)),
            LayerStyle::Fill(FillStyle::new(Color::BLACK)),
        )
    });
    let renderer = Renderer::categorized("code", categories, None);
    assert_eq!(renderer.stored_class_count(), over);
    let tile = feature_collection_to_tile_with(&features, &square, &renderer);
    assert_eq!(
        tile.layers.len(),
        MAX_STYLE_CLASSES + 1,
        "one bucket per RESOLVABLE class, plus the one fallback",
    );
    assert_eq!(
        tile.layers.first().map(|layer| layer.features.len()),
        Some(over - MAX_STYLE_CLASSES),
        "the ten features past the cap fell back rather than growing the tile",
    );
    assert_eq!(
        tile.layers.last().map(|layer| layer.name.clone()),
        Some(local_class_layer_name(
            GeometryKind::Polygon,
            MAX_STYLE_CLASSES - 1
        )),
        "and the highest bucket is the last resolvable class",
    );
    // The compiled program is bounded the same way, so the cap holds on the
    // paint side too rather than only on the geometry side.
    let mut set = LayerStyleSet::new(super::default_style_for_kind(GeometryKind::Polygon));
    set.set_renderer(renderer);
    let mut layer = LocalVectorLayer::with_style("capped", features, set);
    assert_eq!(layer.class_count(), MAX_STYLE_CLASSES);
    let Some(pass) = layer.program().passes().first() else {
        panic!("a fill pass");
    };
    assert_eq!(pass.len(), 3 * (MAX_STYLE_CLASSES + 1));
    // And it still tessellates rather than blowing the budget.
    assert!(layer.tessellate(512.0).is_ok());
    layer.set_style(LayerStyleSet::new(super::default_style_for_kind(
        GeometryKind::Polygon,
    )));
    assert_eq!(layer.class_count(), 0);
}

#[test]
fn class_layer_names_round_trip_through_the_family_parser() {
    for family in GeometryKind::ALL {
        assert_eq!(family_of_layer_name(local_layer_name(family)), Some(family));
        for class in [0, 7, MAX_STYLE_CLASSES - 1] {
            let name = local_class_layer_name(family, class);
            assert!(name.starts_with(local_layer_name(family)), "{name}");
            assert_eq!(family_of_layer_name(&name), Some(family));
        }
    }
    assert_eq!(family_of_layer_name("countries"), None);
    assert_eq!(family_of_layer_name("features:polygon#"), None);
    assert_eq!(family_of_layer_name("features:polygon#x"), None);
    assert_eq!(family_of_layer_name("features:hexagon#1"), None);
}

#[test]
fn a_categorized_layer_draws_every_class_with_its_own_paint() {
    let mut layer = thematic_layer();
    let base = super::default_style_for_kind(GeometryKind::Polygon);
    let single = layer
        .tessellate(512.0)
        .expect("the plain layer tessellates");
    let mut set = LayerStyleSet::new(base.clone());
    // Built through the classify helper, so every class inherits the base's
    // opacity and outline and differs from it in colour ALONE — which is what
    // makes the mesh-area comparison below a statement about the partition
    // rather than about two different styles.
    set.set_renderer(super::classify::categorized_renderer(
        &base,
        "zone",
        [AttrValue::text("a"), AttrValue::text("b")],
    ));
    layer.set_style(set);

    // One paint rule per bucket, and they really are different paints.
    let table = layer.program().passes().first().expect("a fill pass");
    use oxigis_render::PaintResolver as _;
    let class_a = table.paint_for(&local_class_layer_name(GeometryKind::Polygon, 0));
    let class_b = table.paint_for(&local_class_layer_name(GeometryKind::Polygon, 1));
    let fallback = table.paint_for(LOCAL_POLYGON_LAYER_NAME);
    assert!(class_a.is_some() && class_b.is_some() && fallback.is_some());
    assert_ne!(class_a, class_b, "each class carries its own colour");
    assert_ne!(class_a, fallback);

    // And the mesh still covers every feature: the same polygons, now split
    // across three layers, must tessellate to the same total area.
    let classified = layer
        .tessellate(512.0)
        .expect("the classified layer tessellates");
    let delta = (mesh_area(&classified) - mesh_area(&single)).abs();
    assert!(
        delta < mesh_area(&single) * 1e-6,
        "partitioning must not lose (or double) geometry: {} vs {}",
        mesh_area(&single),
        mesh_area(&classified),
    );
    assert_eq!(layer.class_count(), 2);
}

#[test]
fn a_colour_edit_repaints_without_repartitioning_and_a_break_edit_does_not() {
    let mut layer = thematic_layer();
    let mut set = LayerStyleSet::new(super::default_style_for_kind(GeometryKind::Polygon));
    set.set_renderer(zone_renderer());
    layer.set_style(set.clone());
    let partitioned = layer_names(layer.tile());
    let classification = layer.classification().clone();
    let generation = layer.generation();

    // A colour edit inside a class: same buckets, new paint, new generation.
    match set.renderer_mut().class_style_mut(0) {
        Some(style) => *style = LayerStyle::Fill(FillStyle::new(Color::WHITE)),
        None => panic!("class 0 exists"),
    }
    layer.set_style(set.clone());
    assert_eq!(layer_names(layer.tile()), partitioned, "same buckets");
    assert_eq!(layer.classification(), &classification);
    assert_ne!(layer.generation(), generation, "but the mesh is stale");

    // A class-list edit: the buckets themselves move.
    set.renderer_mut().remove_class(1);
    layer.set_style(set);
    assert_eq!(
        layer_names(layer.tile()),
        vec![
            LOCAL_POLYGON_LAYER_NAME.to_string(),
            local_class_layer_name(GeometryKind::Polygon, 0),
        ],
        "class b's features fell back to the plain family layer",
    );
    assert_ne!(layer.classification(), &classification);
}

#[test]
fn dropping_the_renderer_restores_the_pre_v16_tile_exactly() {
    let mut layer = thematic_layer();
    let plain = layer_names(layer.tile());
    let mut set = LayerStyleSet::new(super::default_style_for_kind(GeometryKind::Polygon));
    set.set_renderer(zone_renderer());
    layer.set_style(set.clone());
    assert_ne!(layer_names(layer.tile()), plain);

    set.set_renderer(Renderer::Single);
    layer.set_style(set);
    assert_eq!(
        layer_names(layer.tile()),
        plain,
        "back to one layer per family, byte for byte",
    );
    assert!(layer.classification().is_single());
    assert_eq!(layer.class_count(), 0);
}

#[test]
fn a_classified_layer_still_reports_its_families_to_the_panel() {
    let mixed = parse(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"zone":"a"},
             "geometry":{"type":"Polygon","coordinates":[[[0.0,0.0],[1.0,0.0],
               [1.0,1.0],[0.0,0.0]]]}},
            {"type":"Feature","properties":{"zone":"a"},
             "geometry":{"type":"Point","coordinates":[0.5,0.5]}}]}"#,
    );
    let mut layer = LocalVectorLayer::from_feature_collection("mixed", mixed);
    assert_eq!(layer.families().len(), 2);
    let mut set = layer.style().clone();
    set.set_renderer(zone_renderer());
    layer.set_style(set);
    assert_eq!(
        layer.families().len(),
        2,
        "the family probe must see through the class suffix",
    );
}

#[test]
fn a_graduated_layer_partitions_by_range_and_labels_every_class() {
    let features = parse(thematic_geojson());
    let square = collection_square(&features);
    let breaks = super::classify::equal_interval_breaks(10.0, 40.0, 3);
    let renderer = super::classify::graduated_renderer(
        &super::default_style_for_kind(GeometryKind::Polygon),
        "pop",
        &breaks,
    );
    let tile = feature_collection_to_tile_with(&features, &square, &renderer);
    // pop 10 -> class 0 (<= 20), 20 -> class 0, 30 -> class 1 (<= 30), 40 ->
    // class 2. Every feature classifies, so the fallback bucket is empty and
    // is omitted.
    assert_eq!(
        layer_names(&tile),
        vec![
            local_class_layer_name(GeometryKind::Polygon, 0),
            local_class_layer_name(GeometryKind::Polygon, 1),
            local_class_layer_name(GeometryKind::Polygon, 2),
        ],
    );
    let counts: Vec<usize> = tile
        .layers
        .iter()
        .map(|layer| layer.features.len())
        .collect();
    assert_eq!(counts, vec![2, 1, 1]);
    assert_eq!(renderer.class_label(0), "≤ 20");
    assert_eq!(renderer.class_label(2), "> 30");
}

#[test]
fn a_classified_symbol_style_still_labels_every_class() {
    // The label table is built from the same rule list as the paint program,
    // so a labelling style must keep labelling once the layer is classified.
    let mut layer = thematic_layer();
    let mut set = LayerStyleSet::new(local_symbol_style("zone"));
    set.set_renderer(Renderer::categorized(
        "zone",
        [CategoryClass::new(
            AttrValue::text("a"),
            local_symbol_style("zone"),
        )],
        None,
    ));
    layer.set_style(set);
    assert!(
        !layer.labels().is_empty(),
        "a classified symbol layer must still place labels",
    );
    use oxigis_render::LabelResolver as _;
    assert!(
        layer
            .labels()
            .label_for(&local_class_layer_name(GeometryKind::Polygon, 0))
            .is_some(),
        "including for the class layers",
    );
    assert!(
        layer.labels().label_for(LOCAL_POLYGON_LAYER_NAME).is_some(),
        "and for the fallback one",
    );
}

#[test]
fn classifying_a_mixed_layer_never_makes_a_family_disappear() {
    // THE regression this feature could most easily have shipped: a class
    // names ONE style, a mixed layer draws three families, and the
    // tessellator has no (Points, Fill) arm — so a Fill class handed verbatim
    // to the point family would silently erase every point on the map the
    // moment the layer was classified. `style_for_class` composes instead.
    let geojson = r#"{"type":"FeatureCollection","features":[
        {"type":"Feature","properties":{"zone":"a"},
         "geometry":{"type":"Polygon","coordinates":[[[10.0,10.0],[20.0,10.0],
           [20.0,20.0],[10.0,10.0]]]}},
        {"type":"Feature","properties":{"zone":"a"},
         "geometry":{"type":"Point","coordinates":[15.0,14.0]}}]}"#;
    let mut layer = match LocalVectorLayer::from_geojson("mixed", geojson) {
        Ok(layer) => layer,
        Err(error) => panic!("the fixture must load: {error}"),
    };
    assert_eq!(layer.families().len(), 2);
    let before = layer
        .tessellate(512.0)
        .expect("the plain layer tessellates");

    // A FILL class over a layer whose point family draws circles. Built
    // through the classify helper, so the class is the layer's own base
    // recoloured — every difference below is the PARTITION's doing, not a
    // second style's.
    let base = layer.style().base().clone();
    let mut set = layer.style().clone();
    set.set_renderer(super::classify::categorized_renderer(
        &base,
        "zone",
        [AttrValue::text("a")],
    ));
    layer.set_style(set);
    let after = layer
        .tessellate(512.0)
        .expect("the classified layer tessellates");
    assert_eq!(
        after.vertices.len(),
        before.vertices.len(),
        "every vertex the plain layer drew is still drawn",
    );

    // And the paint the point class resolves to really is a circle.
    use oxigis_render::{LayerPaint, PaintResolver as _};
    let Some(pass) = layer.program().passes().first() else {
        panic!("a geometry pass");
    };
    assert!(
        matches!(
            pass.paint_for(&local_class_layer_name(GeometryKind::Point, 0)),
            Some(LayerPaint::Circle(_))
        ),
        "a Fill class on the point family must resolve to a recoloured CIRCLE",
    );
    assert!(
        matches!(
            pass.paint_for(&local_class_layer_name(GeometryKind::Polygon, 0)),
            Some(LayerPaint::Fill(_))
        ),
        "while the polygon family takes the class verbatim",
    );
}
