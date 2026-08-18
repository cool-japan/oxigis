// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! [`super`] against real pyarrow-written GeoParquet files. Every assertion
//! reads its expectation out of `fixture::TRUTH` rather than a number typed
//! here — see [`super::fixture`]. Hand-built hostile cases live in
//! `tests_hostile`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use oxigeo::geojson::types::{FeatureCollection, Geometry};
use oxigis_render::MercatorPoint;
use serde_json::Value;

use super::fixture::{
    BROTLI, COVERING_BBOX, GZIP, LZ4, MIXED_GEOM, PROJJSON_3857, SNAPPY, TRUTH, UNCOMPRESSED,
    UNSUPPORTED_CRS, ZSTD,
};
use super::from_bytes;

/// One fixture's ground-truth object, by file name.
fn truth(name: &str) -> Value {
    let all: Value = serde_json::from_str(TRUTH).expect("truth.json is JSON");
    all.get(name)
        .unwrap_or_else(|| panic!("no truth entry for {name}"))
        .clone()
}

fn load(bytes: &[u8]) -> FeatureCollection {
    from_bytes(bytes).expect("the fixture is a readable GeoParquet file")
}

fn first_point_lonlat(geometry: &Geometry) -> (f64, f64) {
    match geometry {
        Geometry::Point(point) => (point.coordinates[0], point.coordinates[1]),
        other => panic!("expected a Point, got {other:?}"),
    }
}

// ---- codec matrix: uncompressed / snappy / brotli / lz4 --------------------

#[test]
fn every_readable_codec_matches_its_ground_truth() {
    for (bytes, name) in [
        (UNCOMPRESSED, "uncompressed.parquet"),
        (SNAPPY, "snappy.parquet"),
        (BROTLI, "brotli.parquet"),
        (LZ4, "lz4.parquet"),
    ] {
        let expected = truth(name);
        let collection = load(bytes);

        assert_eq!(
            collection.features.len(),
            expected["rows"].as_u64().unwrap() as usize,
            "{name}: row count",
        );

        // Row 2 (0-indexed): UTF-8 Japanese text must survive intact.
        let japanese_row = expected["japanese_name_row"].as_u64().unwrap() as usize;
        let props = collection.features[japanese_row]
            .properties
            .as_ref()
            .unwrap_or_else(|| panic!("{name}: row {japanese_row} has no properties"));
        assert_eq!(
            props["name"],
            Value::String("\u{672d}\u{5e4c}".to_string()),
            "{name}"
        );

        // Last row: null geometry, preserved AT ITS INDEX — the regression
        // test for the upstream `read_geometries`/`decode_native_array`
        // null-desync bug (see the module docs' "Why a direct, low-level
        // read" section). A version of this reader that dropped the null row
        // instead of keeping it would shift every later row up by one and
        // this index would silently point at the wrong feature.
        let last = collection.features.len() - 1;
        assert!(
            collection.features[last].geometry.is_none(),
            "{name}: the last row's geometry must be null, preserved at its index",
        );
        let last_props = collection.features[last]
            .properties
            .as_ref()
            .unwrap_or_else(|| {
                panic!(
                    "{name}: the null-geometry row must still carry a \
                 properties object"
                )
            });
        for column in ["name", "population", "elevation", "active"] {
            assert_eq!(
                last_props[column],
                Value::Null,
                "{name}: row {last} column {column} must be null",
            );
        }

        // First point, within floating-point tolerance.
        let (lon, lat) = first_point_lonlat(
            collection.features[0]
                .geometry
                .as_ref()
                .unwrap_or_else(|| panic!("{name}: row 0 must have a geometry")),
        );
        let expected_lonlat = expected["first_point_lonlat"].as_array().unwrap();
        assert!(
            (lon - expected_lonlat[0].as_f64().unwrap()).abs() < 1e-9,
            "{name}: lon was {lon}",
        );
        assert!(
            (lat - expected_lonlat[1].as_f64().unwrap()).abs() < 1e-9,
            "{name}: lat was {lat}",
        );

        // Exactly the four declared property columns, typed: int64 -> number,
        // float32 -> number, bool -> bool, and the row-0 values are non-null.
        let mut columns: Vec<String> = props.keys().cloned().collect();
        columns.sort();
        let mut wanted: Vec<String> = expected["columns"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        wanted.sort();
        assert_eq!(columns, wanted, "{name}: property columns");

        let row0 = collection.features[0].properties.as_ref().unwrap();
        assert!(row0["population"].is_number(), "{name}: population");
        assert!(
            row0["population"].is_i64(),
            "{name}: population must be an integer"
        );
        assert!(row0["elevation"].is_number(), "{name}: elevation");
        assert!(row0["active"].is_boolean(), "{name}: active");
    }
}

// ---- unsupported codecs: open but fail to read ------------------------------

#[test]
fn zstd_is_refused_naming_the_codec_without_panicking() {
    let error = from_bytes(ZSTD).expect_err("zstd must be refused, not silently mis-read");
    assert!(
        error.message().contains("zstd"),
        "the refusal must name the codec: {}",
        error.message(),
    );
}

#[test]
fn gzip_is_refused_naming_the_codec_without_panicking() {
    let error = from_bytes(GZIP).expect_err("gzip must be refused, not silently mis-read");
    // The raw upstream error names the Cargo *feature* ("flate2"), not the
    // codec; `codec_error` translates it — assert on our own user-facing
    // text, not the upstream string.
    assert!(
        error.message().contains("gzip"),
        "the refusal must name the codec as \u{201c}gzip\u{201d}, not the underlying \
         \u{201c}flate2\u{201d} feature name: {}",
        error.message(),
    );
}

// ---- CRS ---------------------------------------------------------------------

#[test]
fn a_projjson_3857_crs_inverse_projects_every_vertex() {
    let expected = truth("projjson_3857.parquet");
    let collection = load(PROJJSON_3857);
    assert_eq!(
        collection.features.len(),
        expected["rows"].as_u64().unwrap() as usize,
    );

    let mercator = expected["first_point_3857"].as_array().unwrap();
    let want = MercatorPoint::new(mercator[0].as_f64().unwrap(), mercator[1].as_f64().unwrap())
        .to_lon_lat();
    let (want_lon, want_lat) = (want.lon, want.lat);

    let (lon, lat) = first_point_lonlat(
        collection.features[0]
            .geometry
            .as_ref()
            .expect("row 0 has a geometry"),
    );
    assert!(
        (lon - want_lon).abs() < 1e-6,
        "lon was {lon}, wanted {want_lon}"
    );
    assert!(
        (lat - want_lat).abs() < 1e-6,
        "lat was {lat}, wanted {want_lat}"
    );

    // Loose sanity check independent of the reader's own projection call —
    // the fixture's 3857 coordinates are Tokyo, so an x/y swap (which the
    // tautological check above cannot catch, since it uses the same
    // `MercatorPoint` call the reader itself makes) would fail this.
    assert!((lon - 139.6917).abs() < 1e-2, "lon was {lon}");
    assert!((lat - 35.6895).abs() < 1e-2, "lat was {lat}");
}

#[test]
fn an_unsupported_crs_is_refused_naming_it() {
    let expected = truth("unsupported_crs.parquet");
    let error = from_bytes(UNSUPPORTED_CRS).expect_err("Lambert-93 must be refused");
    let crs_name = expected["crs_name"].as_str().unwrap();
    assert!(
        error.message().contains(crs_name),
        "the refusal must name the CRS \u{201c}{crs_name}\u{201d}: {}",
        error.message(),
    );
}

// ---- covering.bbox -----------------------------------------------------------

#[test]
fn a_covering_bbox_struct_column_is_excluded_from_properties() {
    let expected = truth("covering_bbox.parquet");
    let collection = load(COVERING_BBOX);
    assert_eq!(
        collection.features.len(),
        expected["rows"].as_u64().unwrap() as usize,
    );
    for feature in &collection.features {
        let props = feature.properties.as_ref().expect("properties");
        assert!(
            !props.contains_key("bbox"),
            "the bbox struct column leaked through: {props:?}"
        );
    }
    let mut columns: Vec<String> = collection.features[0]
        .properties
        .as_ref()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    columns.sort();
    let mut wanted: Vec<String> = expected["columns"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    wanted.sort();
    assert_eq!(columns, wanted);
}

// ---- mixed geometry (point + donut polygon) -----------------------------------

#[test]
fn a_donut_polygons_hole_survives_conversion() {
    let expected = truth("mixed_geom.parquet");
    let collection = load(MIXED_GEOM);
    assert_eq!(
        collection.features.len(),
        expected["rows"].as_u64().unwrap() as usize,
    );

    let mut saw_point = false;
    let mut saw_donut = false;
    for feature in &collection.features {
        match feature.geometry.as_ref().expect("geometry") {
            Geometry::Point(_) => saw_point = true,
            Geometry::Polygon(polygon) => {
                let holes = expected["donut_holes"].as_u64().unwrap() as usize;
                assert_eq!(
                    polygon.holes().len(),
                    holes,
                    "the donut's hole must survive"
                );
                saw_donut = true;
            }
            other => panic!("unexpected geometry: {other:?}"),
        }
    }
    assert!(
        saw_point && saw_donut,
        "expected one point and one donut polygon"
    );
}

// ---- the dataset view: what CRS the layer records --------------------------

#[test]
fn read_dataset_reports_the_crs_the_file_declared() {
    // An absent `crs` is the GeoParquet spec's OGC:CRS84 default, which the
    // layer records as "no CRS at all".
    let dataset = super::read_dataset(UNCOMPRESSED).expect("a readable file");
    assert!(dataset.crs.is_wgs84());
    assert!(!dataset.features.features.is_empty());

    // A PROJJSON `id` naming EPSG:3857 comes back as 3857, and the features
    // are already lon/lat.
    let dataset = super::read_dataset(PROJJSON_3857).expect("a readable file");
    assert_eq!(dataset.crs.epsg(), 3857);
    let (lon, lat) = first_point_lonlat(
        dataset.features.features[0]
            .geometry
            .as_ref()
            .expect("geometry"),
    );
    assert!((lon - 139.6917).abs() < 1e-2, "lon was {lon}");
    assert!((lat - 35.6895).abs() < 1e-2, "lat was {lat}");

    // And the same bytes through `from_bytes` give the identical features, so
    // the two entry points cannot drift.
    let plain = from_bytes(PROJJSON_3857).expect("a readable file");
    assert_eq!(plain.features.len(), dataset.features.features.len());
    assert_eq!(
        plain.features[0].geometry,
        dataset.features.features[0].geometry
    );
}

#[test]
fn the_refusal_for_an_unplaceable_crs_names_both_its_name_and_its_code() {
    let error = from_bytes(UNSUPPORTED_CRS).expect_err("Lambert-93 must be refused");
    let expected = truth("unsupported_crs.parquet");
    let crs_name = expected["crs_name"].as_str().unwrap();
    assert!(error.message().contains(crs_name), "{}", error.message());
    assert!(error.message().contains("EPSG:2154"), "{}", error.message());
    assert!(
        error.message().contains("GeoParquet"),
        "{}",
        error.message()
    );
    // The dataset view fails the same way — one implementation, two views.
    assert!(super::read_dataset(UNSUPPORTED_CRS).is_err());
}
