// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super`]. Fixtures come from [`super::fixture`].

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use oxigeo::geojson::types::Geometry;
use oxigeo::shapefile::dbf::{FieldDescriptor, FieldType, FieldValue};
use oxigeo::shapefile::shp::Shape;
use oxigeo::shapefile::shp::shapes::{MultiPartShape, Point as ShpPoint, ShapeType};

use super::fixture;
use super::fixture::{
    dbf_bytes, dbf_bytes_with_deletions, donut, donut_exterior, donut_web_mercator, ring, shp_bytes,
};
use super::{PrjCrs, assemble_rings, from_bytes, read_dataset, sniff_prj, to_geojson_string};
use crate::local_vector::{LocalVectorLayer, collection_square, feature_collection_to_tile};

/// EPSG:3857 metres for a lon/lat pair, computed independently of
/// `oxigis-render` so the CRS test is a genuine round trip.
fn to_web_mercator(lon: f64, lat: f64) -> (f64, f64) {
    const R: f64 = 6_378_137.0;
    let x = lon.to_radians() * R;
    let y = R * (45.0 + lat / 2.0).to_radians().tan().ln();
    (x, y)
}

/// A minimal WGS 84 geographic `.prj`, as ArcGIS writes it.
const WGS84_PRJ: &str = r#"GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]"#;

/// A Web Mercator `.prj`.
const WEB_MERCATOR_PRJ: &str = r#"PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]]],PROJECTION["Mercator_Auxiliary_Sphere"],UNIT["Meter",1.0]]"#;

/// A UTM zone 54N `.prj`.
///
/// No longer the "refuse it" case: UTM 54N resolves to EPSG:32654 and LOADS,
/// reprojected to WGS 84 — see `a_utm_prj_now_loads_instead_of_being_refused`.
/// The refusal fixture is a CRS the registry genuinely has no answer for
/// (S-JTSK Krovak; see `app::tests`).
const UTM_PRJ: &str = r#"PROJCS["WGS_1984_UTM_Zone_54N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]]],PROJECTION["Transverse_Mercator"],PARAMETER["Central_Meridian",141.0],UNIT["Meter",1.0]]"#;

// ---- geometry + attributes -------------------------------------------------

#[test]
fn point_records_carry_their_dbf_attributes_through_as_properties() {
    let shp = shp_bytes(
        ShapeType::Point,
        (135.502, 34.702, 139.767, 35.681),
        vec![
            Shape::Point(ShpPoint::new(139.767, 35.681)),
            Shape::Point(ShpPoint::new(135.502, 34.702)),
        ],
    );
    let dbf = dbf_bytes(
        vec![
            FieldDescriptor::new("NAME".into(), FieldType::Character, 12, 0).expect("field"),
            FieldDescriptor::new("POP".into(), FieldType::Number, 10, 0).expect("field"),
            FieldDescriptor::new("AREA".into(), FieldType::Number, 12, 2).expect("field"),
            FieldDescriptor::new("FOUNDED".into(), FieldType::Date, 8, 0).expect("field"),
            FieldDescriptor::new("CAPITAL".into(), FieldType::Logical, 1, 0).expect("field"),
        ],
        vec![
            vec![
                FieldValue::String("Tokyo".into()),
                FieldValue::Integer(13_960_000),
                FieldValue::Float(2194.07),
                FieldValue::Date("18681013".into()),
                FieldValue::Boolean(true),
            ],
            vec![
                FieldValue::String("Osaka".into()),
                FieldValue::Integer(2_750_000),
                FieldValue::Float(225.21),
                FieldValue::Date("18890401".into()),
                FieldValue::Boolean(false),
            ],
        ],
    );

    let collection = from_bytes(&shp, Some(&dbf), None, None).expect("valid shapefile");
    assert_eq!(collection.features.len(), 2);

    let first = &collection.features[0];
    match first.geometry.as_ref().expect("geometry") {
        Geometry::Point(point) => {
            assert!((point.coordinates[0] - 139.767).abs() < 1e-9);
            assert!((point.coordinates[1] - 35.681).abs() < 1e-9);
            assert_eq!(point.coordinates.len(), 2, "Z/M are deliberately dropped");
        }
        other => panic!("expected a Point, got {other:?}"),
    }
    let properties = first.properties.as_ref().expect("properties");
    assert_eq!(properties["NAME"], serde_json::json!("Tokyo"));
    assert_eq!(properties["POP"], serde_json::json!(13_960_000));
    assert_eq!(properties["AREA"], serde_json::json!(2194.07));
    assert_eq!(
        properties["FOUNDED"],
        serde_json::json!("1868-10-13"),
        "dates are normalised from YYYYMMDD to ISO-8601",
    );
    assert_eq!(properties["CAPITAL"], serde_json::json!(true));

    let second = collection.features[1]
        .properties
        .as_ref()
        .expect("properties");
    assert_eq!(second["CAPITAL"], serde_json::json!(false));
}

#[test]
fn a_single_part_polyline_is_a_linestring_and_a_multi_part_one_is_not() {
    let single = MultiPartShape::new(
        vec![0],
        ring(&[(139.0, 35.0), (140.0, 36.0), (141.0, 35.5)]),
    )
    .expect("polyline");
    let mut points = ring(&[(139.0, 35.0), (140.0, 36.0)]);
    points.extend(ring(&[(141.0, 34.0), (142.0, 34.5)]));
    let multi = MultiPartShape::new(vec![0, 2], points).expect("polyline");

    let shp = shp_bytes(
        ShapeType::PolyLine,
        (139.0, 34.0, 142.0, 36.0),
        vec![Shape::PolyLine(single), Shape::PolyLine(multi)],
    );
    let collection = from_bytes(&shp, None, None, None).expect("valid shapefile");
    assert_eq!(collection.features.len(), 2);
    match collection.features[0].geometry.as_ref().expect("geometry") {
        Geometry::LineString(line) => assert_eq!(line.coordinates.len(), 3),
        other => panic!("expected a LineString, got {other:?}"),
    }
    match collection.features[1].geometry.as_ref().expect("geometry") {
        Geometry::MultiLineString(lines) => assert_eq!(lines.coordinates.len(), 2),
        other => panic!("expected a MultiLineString, got {other:?}"),
    }
}

#[test]
fn a_multipoint_record_keeps_every_vertex() {
    let shape = MultiPartShape::new(
        vec![0],
        ring(&[(139.0, 35.0), (140.0, 36.0), (141.0, 37.0)]),
    )
    .expect("multipoint");
    let shp = shp_bytes(
        ShapeType::MultiPoint,
        (139.0, 35.0, 141.0, 37.0),
        vec![Shape::MultiPoint(shape)],
    );
    let collection = from_bytes(&shp, None, None, None).expect("valid shapefile");
    match collection.features[0].geometry.as_ref().expect("geometry") {
        Geometry::MultiPoint(points) => assert_eq!(points.coordinates.len(), 3),
        other => panic!("expected a MultiPoint, got {other:?}"),
    }
}

#[test]
fn a_donut_polygon_keeps_its_hole_all_the_way_into_the_mesh() {
    let shp = shp_bytes(
        ShapeType::Polygon,
        (138.0, 35.0, 140.0, 37.0),
        vec![donut()],
    );
    let collection = from_bytes(&shp, None, None, None).expect("valid shapefile");

    // 1. The GeoJSON side: exactly one polygon, exterior + one hole.
    match collection.features[0].geometry.as_ref().expect("geometry") {
        Geometry::Polygon(polygon) => {
            assert_eq!(polygon.coordinates.len(), 2, "exterior plus one hole");
            assert_eq!(polygon.coordinates[1].len(), 5, "the hole ring is closed");
        }
        other => panic!("expected a Polygon, got {other:?}"),
    }

    // 2. §1.1's converter must classify the second ring as an interior — the
    //    winding is measured there, so this is what proves the two agree.
    let square = collection_square(&collection);
    let tile = feature_collection_to_tile(&collection, &square);
    let feature = &tile.layers[0].features[0];
    let oxigis_render::MvtGeometry::Polygons(polygons) = &feature.geometry else {
        panic!("expected polygons");
    };
    assert_eq!(polygons[0].interiors.len(), 1, "the hole must survive");

    // 3. And the tessellated mesh really is empty there.
    let mut layer = LocalVectorLayer::from_feature_collection("donut", collection);
    layer.set_style(oxigis_core::LayerStyle::Fill(oxigis_core::FillStyle::new(
        oxigis_core::Color::from_rgb(0xff, 0xff, 0xff),
    )));
    let mesh = layer.tessellate(512.0).expect("tessellation");
    assert!(!mesh.covers([0.5, 0.5]), "the hole was filled in");

    // 4. The control, so step 3 cannot pass vacuously on an empty mesh: the
    //    *same* exterior ring with no hole must cover that very point. Both
    //    datasets have the same bbox, hence the same unit square.
    let solid_shp = shp_bytes(
        ShapeType::Polygon,
        (138.0, 35.0, 140.0, 37.0),
        vec![donut_exterior()],
    );
    let solid = from_bytes(&solid_shp, None, None, None).expect("valid shapefile");
    let mut solid_layer = LocalVectorLayer::from_feature_collection("solid", solid);
    solid_layer.set_style(oxigis_core::LayerStyle::Fill(oxigis_core::FillStyle::new(
        oxigis_core::Color::from_rgb(0xff, 0xff, 0xff),
    )));
    let solid_mesh = solid_layer.tessellate(512.0).expect("tessellation");
    assert!(
        solid_mesh.covers([0.5, 0.5]),
        "without the hole that point must be filled, or the test above proves nothing",
    );
}

#[test]
fn a_web_mercator_donut_keeps_its_hole_after_the_projection() {
    // `assemble_rings` runs *after* the CRS conversion, so the inverse-Mercator
    // step has to preserve ring winding (it is monotone in both axes). If it
    // ever stopped doing so, the hole would silently become a second polygon.
    let shp = shp_bytes(
        ShapeType::Polygon,
        (-2e7, -2e7, 2e7, 2e7),
        vec![donut_web_mercator()],
    );
    let collection = from_bytes(&shp, None, Some(WEB_MERCATOR_PRJ), None).expect("valid");
    match collection.features[0].geometry.as_ref().expect("geometry") {
        Geometry::Polygon(polygon) => {
            assert_eq!(polygon.coordinates.len(), 2, "exterior plus one hole");
            // …and it really did land back on the WGS 84 rectangle.
            assert!((polygon.coordinates[0][0][0] - 138.0).abs() < 1e-6);
            assert!((polygon.coordinates[0][0][1] - 35.0).abs() < 1e-6);
        }
        other => panic!("expected a Polygon, got {other:?}"),
    }
}

#[test]
fn two_clockwise_rings_in_one_record_become_a_multipolygon() {
    let mut points = ring(&[
        (138.0, 35.0),
        (138.0, 36.0),
        (139.0, 36.0),
        (139.0, 35.0),
        (138.0, 35.0),
    ]);
    points.extend(ring(&[
        (141.0, 35.0),
        (141.0, 36.0),
        (142.0, 36.0),
        (142.0, 35.0),
        (141.0, 35.0),
    ]));
    let shp = shp_bytes(
        ShapeType::Polygon,
        (138.0, 35.0, 142.0, 36.0),
        vec![Shape::Polygon(
            MultiPartShape::new(vec![0, 5], points).expect("islands"),
        )],
    );
    let collection = from_bytes(&shp, None, None, None).expect("valid shapefile");
    match collection.features[0].geometry.as_ref().expect("geometry") {
        Geometry::MultiPolygon(multi) => assert_eq!(multi.coordinates.len(), 2),
        other => panic!("expected a MultiPolygon, got {other:?}"),
    }
}

#[test]
fn a_hole_goes_to_the_smallest_exterior_that_contains_it() {
    // A big ring, a small ring nested inside it, and a hole inside the small
    // one: the hole must land on the small ring, not the big one.
    let cw = |ox: f64, oy: f64, s: f64| {
        vec![
            vec![ox, oy],
            vec![ox, oy + s],
            vec![ox + s, oy + s],
            vec![ox + s, oy],
            vec![ox, oy],
        ]
    };
    let ccw = |ox: f64, oy: f64, s: f64| {
        vec![
            vec![ox, oy],
            vec![ox + s, oy],
            vec![ox + s, oy + s],
            vec![ox, oy + s],
            vec![ox, oy],
        ]
    };
    let geometry = assemble_rings(vec![
        cw(0.0, 0.0, 100.0),
        cw(10.0, 10.0, 50.0),
        ccw(20.0, 20.0, 5.0),
    ])
    .expect("geometry");
    match geometry {
        Geometry::MultiPolygon(multi) => {
            assert_eq!(multi.coordinates.len(), 2);
            let owners: Vec<&Vec<Vec<f64>>> = multi
                .coordinates
                .iter()
                .filter(|rings| rings.len() > 1)
                .map(|rings| &rings[0])
                .collect();
            assert_eq!(owners.len(), 1, "exactly one polygon owns the hole");
            // The owner is the 50-wide ring, whose first vertex is (10, 10).
            assert_eq!(owners[0][0], vec![10.0, 10.0]);
        }
        other => panic!("expected a MultiPolygon, got {other:?}"),
    }
}

#[test]
fn all_counter_clockwise_rings_are_kept_as_separate_exteriors() {
    // Non-conformant producers exist; the geometry must not vanish.
    let ccw = |ox: f64| {
        vec![
            vec![ox, 0.0],
            vec![ox + 1.0, 0.0],
            vec![ox + 1.0, 1.0],
            vec![ox, 1.0],
            vec![ox, 0.0],
        ]
    };
    let geometry = assemble_rings(vec![ccw(0.0), ccw(10.0)]).expect("geometry");
    assert!(matches!(geometry, Geometry::MultiPolygon(_)));
}

// ---- the .dbf half ---------------------------------------------------------

#[test]
fn a_shapefile_without_a_dbf_loads_as_geometry_only() {
    let shp = shp_bytes(
        ShapeType::Point,
        (139.767, 35.681, 139.767, 35.681),
        vec![Shape::Point(ShpPoint::new(139.767, 35.681))],
    );
    let collection = from_bytes(&shp, None, None, None).expect("a lone .shp is loadable");
    assert_eq!(collection.features.len(), 1);
    assert!(
        collection.features[0].properties.is_none(),
        "no .dbf means no properties at all, not empty ones",
    );
}

#[test]
fn a_dbf_that_does_not_match_the_shp_is_refused_naming_both_counts() {
    let shp = shp_bytes(
        ShapeType::Point,
        (139.0, 35.0, 140.0, 36.0),
        vec![
            Shape::Point(ShpPoint::new(139.0, 35.0)),
            Shape::Point(ShpPoint::new(140.0, 36.0)),
        ],
    );
    let dbf = dbf_bytes(
        vec![FieldDescriptor::new("NAME".into(), FieldType::Character, 8, 0).expect("field")],
        vec![vec![FieldValue::String("only one".into())]],
    );
    let error = from_bytes(&shp, Some(&dbf), None, None).expect_err("counts disagree");
    assert!(error.message().contains('2'), "{}", error.message());
    assert!(error.message().contains('1'), "{}", error.message());
}

#[test]
fn a_four_billion_record_count_is_refused_before_anything_is_reserved() {
    // `Vec::with_capacity`'s failure path is `handle_alloc_error`, which
    // aborts the process — not a `Result` this reader's "every failure is a
    // message" architecture can intercept — so an impossible count has to be
    // refused by the byte-count check rather than survived by the allocator.
    let shp = shp_bytes(
        ShapeType::Point,
        (139.0, 35.0, 139.0, 35.0),
        vec![Shape::Point(ShpPoint::new(139.0, 35.0))],
    );
    let mut dbf = dbf_bytes(
        vec![FieldDescriptor::new("NAME".into(), FieldType::Character, 8, 0).expect("field")],
        Vec::new(),
    );
    // Overwrite the on-disk record count (offset 4, little-endian `u32`) with
    // an impossible value while leaving the rest of this tiny file untouched.
    dbf[4..8].copy_from_slice(&u32::MAX.to_le_bytes());

    let started = std::time::Instant::now();
    let error = from_bytes(&shp, Some(&dbf), None, None).expect_err("an impossible count");
    assert!(
        error.message().contains("4294967295"),
        "{}",
        error.message()
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(2),
        "refusing an impossible count must not take real time",
    );
}

#[test]
fn a_spoofed_zero_record_size_does_not_defeat_the_bounds_check() {
    // `record_size` (offset 10) is never consulted by `DbfRecord::read`, so a
    // hostile file could zero it out to make `record_count * record_size`
    // collapse to nothing while `record_count` alone is still enormous — the
    // check has to be built from the field descriptors, not this header field.
    let shp = shp_bytes(
        ShapeType::Point,
        (139.0, 35.0, 139.0, 35.0),
        vec![Shape::Point(ShpPoint::new(139.0, 35.0))],
    );
    let mut dbf = dbf_bytes(
        vec![FieldDescriptor::new("NAME".into(), FieldType::Character, 8, 0).expect("field")],
        Vec::new(),
    );
    dbf[4..8].copy_from_slice(&u32::MAX.to_le_bytes()); // record_count
    dbf[10..12].copy_from_slice(&0u16.to_le_bytes()); // record_size

    let error = from_bytes(&shp, Some(&dbf), None, None).expect_err("still an impossible count");
    assert!(
        error.message().contains("4294967295"),
        "{}",
        error.message()
    );
}

#[test]
fn a_deleted_dbf_record_and_its_paired_shape_are_both_skipped() {
    let shp = shp_bytes(
        ShapeType::Point,
        (139.0, 34.0, 140.0, 36.0),
        vec![
            Shape::Point(ShpPoint::new(139.0, 35.0)),
            Shape::Point(ShpPoint::new(140.0, 36.0)),
            Shape::Point(ShpPoint::new(139.5, 35.5)),
        ],
    );
    let dbf = dbf_bytes_with_deletions(
        vec![FieldDescriptor::new("NAME".into(), FieldType::Character, 8, 0).expect("field")],
        vec![
            (vec![FieldValue::String("alive-1".into())], false),
            (vec![FieldValue::String("gone".into())], true),
            (vec![FieldValue::String("alive-2".into())], false),
        ],
    );

    let collection = from_bytes(&shp, Some(&dbf), None, None).expect("valid shapefile");
    assert_eq!(
        collection.features.len(),
        2,
        "the deleted row must not become a feature",
    );
    let names: Vec<&str> = collection
        .features
        .iter()
        .map(|feature| {
            feature.properties.as_ref().expect("row")["NAME"]
                .as_str()
                .expect("string value")
        })
        .collect();
    assert_eq!(names, vec!["alive-1", "alive-2"]);
}

#[test]
fn a_shapefile_whose_every_dbf_record_is_deleted_is_refused_like_an_empty_one() {
    let shp = shp_bytes(
        ShapeType::Point,
        (139.0, 35.0, 139.0, 35.0),
        vec![Shape::Point(ShpPoint::new(139.0, 35.0))],
    );
    let dbf = dbf_bytes_with_deletions(
        vec![FieldDescriptor::new("NAME".into(), FieldType::Character, 8, 0).expect("field")],
        vec![(vec![FieldValue::String("gone".into())], true)],
    );
    let error = from_bytes(&shp, Some(&dbf), None, None).expect_err("every record is deleted");
    assert!(
        error.message().contains("no records"),
        "{}",
        error.message()
    );
}

#[test]
fn memo_columns_are_skipped_rather_than_showing_their_block_pointer() {
    let shp = shp_bytes(
        ShapeType::Point,
        (139.0, 35.0, 139.0, 35.0),
        vec![Shape::Point(ShpPoint::new(139.0, 35.0))],
    );
    let dbf = dbf_bytes(
        vec![
            FieldDescriptor::new("NAME".into(), FieldType::Character, 8, 0).expect("field"),
            FieldDescriptor::new("NOTES".into(), FieldType::Memo, 10, 0).expect("field"),
        ],
        vec![vec![
            FieldValue::String("Tokyo".into()),
            FieldValue::String("42".into()),
        ]],
    );
    let collection = from_bytes(&shp, Some(&dbf), None, None).expect("valid shapefile");
    let properties = collection.features[0]
        .properties
        .as_ref()
        .expect("properties");
    assert!(properties.contains_key("NAME"));
    assert!(
        !properties.contains_key("NOTES"),
        "a memo column with no .dbt would show a block pointer",
    );
}

#[test]
fn a_cpg_label_picks_the_dbf_text_encoding() {
    // Two bytes of Shift_JIS ("\u{65e5}" = 0x93 0xFA) that are not valid UTF-8:
    // the .cpg is what makes them decode rather than turn into replacements.
    let shp = shp_bytes(
        ShapeType::Point,
        (139.0, 35.0, 139.0, 35.0),
        vec![Shape::Point(ShpPoint::new(139.0, 35.0))],
    );
    let mut dbf = dbf_bytes(
        vec![FieldDescriptor::new("NAME".into(), FieldType::Character, 4, 0).expect("field")],
        vec![vec![FieldValue::String("ab".into())]],
    );
    // Overwrite the record's text cell with the raw Shift_JIS bytes. The record
    // starts after the 32-byte header, one 32-byte descriptor and the 0x0D
    // terminator, then one deletion-flag byte.
    let cell = 32 + 32 + 1 + 1;
    dbf[cell] = 0x93;
    dbf[cell + 1] = 0xFA;

    let decoded = from_bytes(&shp, Some(&dbf), None, Some("Shift_JIS")).expect("valid");
    let properties = decoded.features[0].properties.as_ref().expect("properties");
    assert_eq!(properties["NAME"], serde_json::json!("\u{65e5}"));

    let lossy = from_bytes(&shp, Some(&dbf), None, None).expect("valid");
    let properties = lossy.features[0].properties.as_ref().expect("properties");
    assert_ne!(
        properties["NAME"],
        serde_json::json!("\u{65e5}"),
        "without the .cpg the bytes are not Shift_JIS",
    );
}

#[test]
fn an_empty_shapefile_is_refused_like_an_empty_geojson() {
    let shp = shp_bytes(ShapeType::Point, (0.0, 0.0, 0.0, 0.0), Vec::new());
    let error = from_bytes(&shp, None, None, None).expect_err("no records");
    assert!(error.message().contains("no records"));
}

#[test]
fn a_null_shape_keeps_its_attribute_row_with_a_null_geometry() {
    let shp = shp_bytes(
        ShapeType::Point,
        (139.0, 35.0, 139.0, 35.0),
        vec![Shape::Null, Shape::Point(ShpPoint::new(139.0, 35.0))],
    );
    let dbf = dbf_bytes(
        vec![FieldDescriptor::new("NAME".into(), FieldType::Character, 8, 0).expect("field")],
        vec![
            vec![FieldValue::String("nowhere".into())],
            vec![FieldValue::String("Tokyo".into())],
        ],
    );
    let collection = from_bytes(&shp, Some(&dbf), None, None).expect("valid shapefile");
    assert_eq!(collection.features.len(), 2);
    assert!(collection.features[0].geometry.is_none());
    assert_eq!(
        collection.features[0].properties.as_ref().expect("row")["NAME"],
        serde_json::json!("nowhere"),
    );
}

#[test]
fn broken_bytes_are_reported_rather_than_panicking() {
    let error = from_bytes(&[0u8; 8], None, None, None).expect_err("not a .shp");
    assert!(!error.message().is_empty());
    let shp = shp_bytes(
        ShapeType::Point,
        (139.0, 35.0, 139.0, 35.0),
        vec![Shape::Point(ShpPoint::new(139.0, 35.0))],
    );
    let error = from_bytes(&shp, Some(&[0u8; 4]), None, None).expect_err("not a .dbf");
    assert!(!error.message().is_empty());
}

// ---- CRS -------------------------------------------------------------------

#[test]
fn a_missing_or_wgs84_prj_passes_coordinates_through() {
    for absent in [None, Some("   "), Some(WGS84_PRJ)] {
        let crs = sniff_prj(absent).expect("resolves");
        assert!(crs.is_wgs84(), "{absent:?}");
        assert_eq!(crs.epsg(), 4326);
    }
    assert_eq!(sniff_prj(None).expect("no prj"), PrjCrs::wgs84());

    let shp = shp_bytes(
        ShapeType::Point,
        (139.767, 35.681, 139.767, 35.681),
        vec![Shape::Point(ShpPoint::new(139.767, 35.681))],
    );
    let collection = from_bytes(&shp, None, Some(WGS84_PRJ), None).expect("valid");
    match collection.features[0].geometry.as_ref().expect("geometry") {
        Geometry::Point(point) => assert!((point.coordinates[0] - 139.767).abs() < 1e-9),
        other => panic!("expected a Point, got {other:?}"),
    }

    // The dataset view reports the CRS as WGS 84, which the layer records as
    // "no CRS at all" — see `oxigis_core::Layer::with_crs`.
    let dataset = read_dataset(&shp, None, None, None).expect("valid");
    assert!(dataset.crs.is_wgs84());
}

#[test]
fn a_web_mercator_prj_converts_metres_to_degrees() {
    assert_eq!(
        sniff_prj(Some(WEB_MERCATOR_PRJ)).expect("3857").epsg(),
        3857
    );
    // Every EPSG code spelling must be recognised too.
    for wkt in [
        r#"PROJCS["WGS 84 / Pseudo-Mercator",AUTHORITY["EPSG","3857"]]"#,
        r#"PROJCS["Google Maps Global Mercator",AUTHORITY["EPSG","900913"]]"#,
    ] {
        assert_eq!(sniff_prj(Some(wkt)).expect("3857").epsg(), 3857);
    }

    let (x, y) = to_web_mercator(139.767, 35.681);
    let shp = shp_bytes(
        ShapeType::Point,
        (x, y, x, y),
        vec![Shape::Point(ShpPoint::new(x, y))],
    );
    let collection = from_bytes(&shp, None, Some(WEB_MERCATOR_PRJ), None).expect("valid");
    match collection.features[0].geometry.as_ref().expect("geometry") {
        Geometry::Point(point) => {
            assert!(
                (point.coordinates[0] - 139.767).abs() < 1e-6,
                "lon was {}",
                point.coordinates[0],
            );
            assert!(
                (point.coordinates[1] - 35.681).abs() < 1e-6,
                "lat was {}",
                point.coordinates[1],
            );
        }
        other => panic!("expected a Point, got {other:?}"),
    }
}

// ---- Japan Plane Rectangular: the case the feature exists for --------------

#[test]
fn a_jgd2011_plane_rectangular_shapefile_loads_and_lands_in_tokyo() {
    // Before this feature a Japanese municipal shapefile — which is what this
    // is — was refused outright. Now it loads, in metres, and the three
    // vertices come back as the lon/lat they name.
    let (shp, dbf) = fixture::jgd2011_zone9_cities();
    let dataset =
        read_dataset(&shp, Some(&dbf), Some(fixture::JGD2011_ZONE9_PRJ), None).expect("zone IX");
    assert_eq!(dataset.crs.epsg(), 6677, "the CRS is recorded on the layer");
    assert_eq!(dataset.features.features.len(), 3);

    let point = |index: usize| match dataset.features.features[index]
        .geometry
        .as_ref()
        .expect("geometry")
    {
        Geometry::Point(point) => (point.coordinates[0], point.coordinates[1]),
        other => panic!("expected a Point, got {other:?}"),
    };

    // Vertex 0 is the zone origin: `x_0 = y_0 = 0`, so (0, 0) is exactly
    // 139°50'E / 36°N.
    let (lon, lat) = point(0);
    assert!(
        (lon - fixture::ZONE9_CENTRAL_MERIDIAN).abs() < 1e-7,
        "origin lon {lon}",
    );
    assert!(
        (lat - fixture::ZONE9_LATITUDE_OF_ORIGIN).abs() < 1e-7,
        "origin lat {lat}",
    );

    // Vertex 1 is the pinned control point.
    let (_, (expected_lon, expected_lat)) = fixture::ZONE9_CONTROL_POINT;
    let (lon, lat) = point(1);
    assert!((lon - expected_lon).abs() < 1e-6, "control lon {lon}");
    assert!((lat - expected_lat).abs() < 1e-6, "control lat {lat}");

    // Vertex 2 is 20 km north-east of the origin: north and east of it, and
    // still comfortably inside Japan.
    let (lon, lat) = point(2);
    assert!(
        lon > fixture::ZONE9_CENTRAL_MERIDIAN,
        "north-east lon {lon}"
    );
    assert!(
        lat > fixture::ZONE9_LATITUDE_OF_ORIGIN,
        "north-east lat {lat}"
    );
    assert!((139.0..141.0).contains(&lon) && (35.0..37.0).contains(&lat));

    // The attributes survive the reprojection unchanged.
    assert_eq!(
        dataset.features.features[1]
            .properties
            .as_ref()
            .expect("row")["NAME"],
        serde_json::json!("tokyo"),
    );
}

#[test]
fn the_esri_prj_for_the_same_zone_resolves_without_an_authority_clause() {
    let (shp, dbf) = fixture::jgd2011_zone9_cities();
    let esri = read_dataset(
        &shp,
        Some(&dbf),
        Some(fixture::JGD2011_ZONE9_ESRI_PRJ),
        None,
    )
    .expect("zone IX");
    let gdal =
        read_dataset(&shp, Some(&dbf), Some(fixture::JGD2011_ZONE9_PRJ), None).expect("zone IX");
    assert_eq!(esri.crs.epsg(), 6677);
    assert_eq!(esri.crs.epsg(), gdal.crs.epsg());
    // Same bytes, same CRS, same coordinates — the two `.prj` dialects are not
    // allowed to disagree.
    for (a, b) in esri.features.features.iter().zip(&gdal.features.features) {
        assert_eq!(a.geometry, b.geometry);
    }
}

#[test]
fn a_reprojected_polygon_keeps_its_ring_winding_and_its_hole_logic() {
    // Reprojection must not invert a ring: `assemble_rings` decides
    // exterior-vs-hole from the signed area, and a projection that flipped the
    // y axis would turn every exterior into a hole and lose the whole feature.
    let shp = shp_bytes(
        ShapeType::Polygon,
        (0.0, 0.0, 20_000.0, 20_000.0),
        vec![fixture::jgd2011_zone9_polygon()],
    );
    let collection =
        from_bytes(&shp, None, Some(fixture::JGD2011_ZONE9_PRJ), None).expect("zone IX polygon");
    match collection.features[0].geometry.as_ref().expect("geometry") {
        Geometry::Polygon(polygon) => {
            assert_eq!(polygon.coordinates.len(), 1, "one ring, no hole");
            assert_eq!(polygon.coordinates[0].len(), 5);
            for position in &polygon.coordinates[0] {
                assert!((139.0..141.0).contains(&position[0]), "{position:?}");
                assert!((35.0..37.0).contains(&position[1]), "{position:?}");
            }
        }
        other => panic!("expected a Polygon, got {other:?}"),
    }
}

#[test]
fn every_japanese_plane_rectangular_zone_of_every_datum_is_accepted() {
    // All 57 zone CRSs (19 zones × JGD2011 / JGD2000 / Tokyo Datum), by EPSG
    // authority code, must build a reprojection rather than be refused.
    for base in [6669_u32, 2443, 30161] {
        for zone in 0..19_u32 {
            let epsg = base + zone;
            let wkt = format!(r#"PROJCS["zone",AUTHORITY["EPSG","{epsg}"]]"#);
            let crs = sniff_prj(Some(&wkt))
                .unwrap_or_else(|error| panic!("EPSG:{epsg}: {}", error.message()));
            assert_eq!(crs.epsg(), epsg);
            assert!(!crs.is_wgs84(), "EPSG:{epsg} is projected");
        }
    }
}

#[test]
fn a_tokyo_datum_prj_with_a_towgs84_clause_is_shifted_not_passed_through() {
    // FINDING 73. `TOWGS84[-146.414,…]` contains the literal `WGS84`, so the
    // marker scan this replaced classified the file as WGS 84 and drew it
    // ~450 m from where it belongs, with no notice. Now the datum is
    // recognised and the Helmert shift applied.
    assert!(fixture::TOKYO_DATUM_PRJ.contains("TOWGS84"));
    let crs = sniff_prj(Some(fixture::TOKYO_DATUM_PRJ)).expect("Tokyo Datum loads");
    assert_eq!(crs.epsg(), 4301);
    assert!(!crs.is_wgs84(), "a TOWGS84 clause must not read as WGS 84");

    // Tokyo Datum degrees in, WGS 84 degrees out — moved north and west by a
    // few hundred metres, which is the published magnitude of the shift.
    let shp = shp_bytes(
        ShapeType::Point,
        (139.75, 35.65, 139.75, 35.65),
        vec![Shape::Point(ShpPoint::new(139.75, 35.65))],
    );
    let collection =
        from_bytes(&shp, None, Some(fixture::TOKYO_DATUM_PRJ), None).expect("Tokyo Datum");
    match collection.features[0].geometry.as_ref().expect("geometry") {
        Geometry::Point(point) => {
            let (lon, lat) = (point.coordinates[0], point.coordinates[1]);
            assert!(lat > 35.65, "Tokyo Datum coordinates move north, got {lat}");
            assert!(lon < 139.75, "and west, got {lon}");
            let metres = ((lon - 139.75) * 90_000.0).hypot((lat - 35.65) * 111_000.0);
            assert!(
                (300.0..700.0).contains(&metres),
                "the published shift near Tokyo is ~450 m; got {metres} m",
            );
        }
        other => panic!("expected a Point, got {other:?}"),
    }
}

#[test]
fn a_utm_prj_now_loads_instead_of_being_refused() {
    // The asymmetry finding 203 names: the COG raster path already handled all
    // 120 UTM zones while a shapefile in the same CRS was refused.
    let crs = sniff_prj(Some(UTM_PRJ)).expect("UTM 54N loads");
    assert_eq!(crs.epsg(), 32654);

    let shp = shp_bytes(
        ShapeType::Point,
        (0.0, 0.0, 0.0, 0.0),
        vec![Shape::Point(ShpPoint::new(
            388_433.374_620_895,
            3_949_290.013_641_47,
        ))],
    );
    let collection = from_bytes(&shp, None, Some(UTM_PRJ), None).expect("UTM 54N");
    match collection.features[0].geometry.as_ref().expect("geometry") {
        Geometry::Point(point) => {
            assert!((point.coordinates[0] - 139.7671).abs() < 1e-5, "{point:?}");
            assert!((point.coordinates[1] - 35.6812).abs() < 1e-5, "{point:?}");
        }
        other => panic!("expected a Point, got {other:?}"),
    }
}

#[test]
fn a_crs_this_build_cannot_place_is_still_refused_with_its_name_in_the_message() {
    // Lambert Conformal Conic is a projection family this build does not
    // invert: refused, by name AND by code, so the message is actionable.
    let lambert = r#"PROJCS["RGF93 / Lambert-93",PROJECTION["Lambert_Conformal_Conic_2SP"],AUTHORITY["EPSG","2154"]]"#;
    let error = sniff_prj(Some(lambert)).expect_err("Lambert-93 must be refused");
    assert!(
        error.message().contains("RGF93 / Lambert-93"),
        "the refusal must name the CRS: {}",
        error.message(),
    );
    assert!(error.message().contains("EPSG:2154"), "{}", error.message());
    assert!(error.message().contains("4326"), "{}", error.message());

    // The ESRI Tokyo Datum `.prj` — no TOWGS84, no authority, only a `D_Tokyo`
    // datum name — resolves to nothing and is refused, exactly as before.
    let tokyo_datum =
        r#"GEOGCS["Tokyo",DATUM["D_Tokyo",SPHEROID["Bessel_1841",6377397.155,299.1528128]]]"#;
    let error = sniff_prj(Some(tokyo_datum)).expect_err("an unresolvable datum");
    assert!(error.message().contains("Tokyo"), "{}", error.message());

    // And the whole load fails rather than placing the data wrongly.
    let shp = shp_bytes(
        ShapeType::Point,
        (0.0, 0.0, 0.0, 0.0),
        vec![Shape::Point(ShpPoint::new(381_000.0, 3_949_000.0))],
    );
    assert!(from_bytes(&shp, None, Some(lambert), None).is_err());
}

#[test]
fn a_prj_with_no_quoted_name_still_produces_a_usable_message() {
    let error = sniff_prj(Some("LOCAL_CS")).expect_err("unknown");
    assert!(error.message().contains("LOCAL_CS"), "{}", error.message());
}

#[test]
fn a_no_data_sentinel_vertex_is_dropped_rather_than_reprojected() {
    // `-1e38` is finite, so it survives the "is this a number" gate and reaches
    // the projection; the Transverse Mercator series cannot invert it, and the
    // vertex must be dropped rather than become a NaN position.
    let shp = shp_bytes(
        ShapeType::Point,
        (-1e38, -1e38, 0.0, 0.0),
        vec![
            Shape::Point(ShpPoint::new(-1e38, -1e38)),
            Shape::Point(ShpPoint::new(0.0, 0.0)),
        ],
    );
    let collection =
        from_bytes(&shp, None, Some(fixture::JGD2011_ZONE9_PRJ), None).expect("zone IX");
    assert_eq!(collection.features.len(), 2, "both rows stay");
    for feature in &collection.features {
        if let Some(Geometry::Point(point)) = feature.geometry.as_ref() {
            assert!(point.coordinates[0].is_finite() && point.coordinates[1].is_finite());
        }
    }
}

// ---- the GeoJSON persistence leg ------------------------------------------

#[test]
fn a_collection_survives_the_round_trip_through_geojson_text() {
    let shp = shp_bytes(
        ShapeType::Polygon,
        (138.0, 35.0, 140.0, 37.0),
        vec![donut()],
    );
    let dbf = dbf_bytes(
        vec![FieldDescriptor::new("NAME".into(), FieldType::Character, 8, 0).expect("field")],
        vec![vec![FieldValue::String("donut".into())]],
    );
    let collection = from_bytes(&shp, Some(&dbf), None, None).expect("valid");
    let text = to_geojson_string(&collection).expect("serialise");
    let reparsed = crate::local_input::parse_geojson(&text).expect("reparse");

    assert_eq!(reparsed.features.len(), collection.features.len());
    assert_eq!(
        reparsed.features[0].properties.as_ref().expect("row")["NAME"],
        serde_json::json!("donut"),
    );
    match reparsed.features[0].geometry.as_ref().expect("geometry") {
        Geometry::Polygon(polygon) => assert_eq!(polygon.coordinates.len(), 2),
        other => panic!("expected a Polygon, got {other:?}"),
    }
}
