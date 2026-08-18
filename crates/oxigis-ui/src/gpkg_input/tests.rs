// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tier 1: [`super`] against GeoPackages written by real SQLite.
//!
//! Every assertion is checked against the ground truth emitted beside the
//! fixture (`fixtures/*_truth.json`) rather than against numbers typed here, so
//! regenerating a fixture cannot silently invalidate a test. See
//! [`super::fixture`] for what each file exercises; the hand-built hostile
//! cases live in `tests_hostile`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use oxigeo::geojson::types::Geometry;
use serde_json::Value;

use super::fixture::{
    BASIC, BASIC_TRUTH, PAGED, PAGED_TRUTH, UNICODE, UNICODE_TRUTH, WITHOUT_ROWID,
};
use super::sqlite::SqliteDb;
use super::{GpkgCrs, GpkgDataset, GpkgTable, from_bytes, resolve_srs};

/// The ground truth of a fixture, parsed.
fn truth(text: &str) -> Value {
    serde_json::from_str(text).expect("the truth file is JSON")
}

/// Loads a fixture, failing the test with the reader's own message.
fn load(bytes: &[u8]) -> GpkgDataset {
    from_bytes(bytes).expect("the fixture is a readable GeoPackage")
}

/// One table of a loaded dataset.
fn table<'a>(dataset: &'a GpkgDataset, name: &str) -> &'a GpkgTable {
    dataset
        .table(name)
        .unwrap_or_else(|| panic!("the {name} table must have loaded"))
}

/// A feature's first `[lon, lat]`, whatever single geometry kind it is.
fn first_vertex(geometry: &Geometry) -> (f64, f64) {
    let position = match geometry {
        Geometry::Point(point) => point.coordinates.clone(),
        Geometry::LineString(line) => line.coordinates[0].clone(),
        Geometry::Polygon(polygon) => polygon.coordinates[0][0].clone(),
        other => panic!("expected a single geometry, got {other:?}"),
    };
    (position[0], position[1])
}

/// The `fid` property of every feature, in order.
fn fids(table: &GpkgTable) -> Vec<i64> {
    table
        .features
        .features
        .iter()
        .map(|feature| {
            feature.properties.as_ref().expect("properties")["fid"]
                .as_i64()
                .expect("a numeric fid")
        })
        .collect()
}

// ---- basic.gpkg ------------------------------------------------------------

#[test]
fn every_supported_table_of_a_multi_table_geopackage_becomes_a_layer() {
    let dataset = load(BASIC);
    let names: Vec<&str> = dataset
        .tables()
        .iter()
        .map(|table| table.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["cities", "parks", "roads"],
        "in gpkg_contents order, and only the loadable ones",
    );
    // regions (EPSG:2154) is the one refusal; notes_attr (an attributes table)
    // and SQLite's own sqlite_sequence must not produce one.
    assert_eq!(dataset.notices().len(), 1, "{:?}", dataset.notices());
    let notice = &dataset.notices()[0];
    assert!(notice.contains("regions"), "{notice}");
    assert!(
        notice.contains("RGF93 / Lambert-93"),
        "the refusal must name the CRS: {notice}",
    );
    assert!(notice.contains("4326"), "{notice}");
}

#[test]
fn the_cities_table_matches_its_ground_truth() {
    let truth = truth(BASIC_TRUTH);
    let expected = &truth["tables"]["cities"];
    let dataset = load(BASIC);
    let cities = table(&dataset, "cities");
    assert_eq!(
        cities.features.features.len(),
        expected["feature_count"].as_u64().unwrap() as usize,
    );

    // The columns come out of the CREATE TABLE statement, geometry excluded —
    // including the quoted one with a space in its name.
    let mut columns: Vec<String> = cities.features.features[0]
        .properties
        .as_ref()
        .expect("properties")
        .keys()
        .cloned()
        .collect();
    let mut wanted: Vec<String> = expected["columns"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .filter(|column| *column != "geom")
        .map(str::to_string)
        .collect();
    columns.sort();
    wanted.sort();
    assert_eq!(columns, wanted);

    // `fid INTEGER PRIMARY KEY AUTOINCREMENT` is the rowid alias: the record
    // stores NULL and the value only exists as the cell's rowid.
    let wanted: Vec<i64> = expected["fids"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_i64)
        .collect();
    assert_eq!(fids(cities), wanted, "rowid substitution failed");

    let tokyo = cities.features.features[0]
        .properties
        .as_ref()
        .expect("properties");
    assert_eq!(tokyo["name"], serde_json::json!("Tokyo"));
    assert_eq!(
        tokyo["name ja"],
        serde_json::json!("\u{6771}\u{4eac}"),
        "a quoted column name and its UTF-8 text must both survive",
    );
    assert_eq!(tokyo["population"], serde_json::json!(13_960_000));
    assert_eq!(tokyo["elevation"], serde_json::json!(40.0));
    assert_eq!(tokyo["notes"], Value::Null);

    let point = cities.features.features[0]
        .geometry
        .as_ref()
        .expect("geometry");
    let (lon, lat) = first_vertex(point);
    let wanted = expected["first_point_lonlat"].as_array().unwrap();
    assert!((lon - wanted[0].as_f64().unwrap()).abs() < 1e-9);
    assert!((lat - wanted[1].as_f64().unwrap()).abs() < 1e-9);

    let nulls = cities
        .features
        .features
        .iter()
        .filter(|feature| feature.geometry.is_none())
        .count();
    assert_eq!(
        nulls,
        expected["null_geometry_rows"].as_u64().unwrap() as usize,
        "a NULL geometry keeps its attribute row",
    );
    let nowhere = cities.features.features[3]
        .properties
        .as_ref()
        .expect("properties");
    assert_eq!(nowhere["name"], serde_json::json!("Nowhere"));
}

#[test]
fn a_row_that_spilled_into_overflow_pages_comes_back_whole() {
    // The reason this reader exists: `oxigeo-gpkg` fails the whole table here.
    // 600 vertices is ~9.6 KB of WKB, well past the 4061-byte inline maximum of
    // a 4096-byte page, so this only passes if the overflow chain was walked
    // and concatenated byte-exactly.
    let truth = truth(BASIC_TRUTH);
    let expected = &truth["tables"]["parks"];
    let dataset = load(BASIC);
    let parks = table(&dataset, "parks");
    assert_eq!(
        parks.features.features.len(),
        expected["feature_count"].as_u64().unwrap() as usize,
    );
    assert_eq!(fids(parks), vec![1, 2]);

    match parks.features.features[1]
        .geometry
        .as_ref()
        .expect("geometry")
    {
        Geometry::Polygon(polygon) => {
            assert_eq!(polygon.coordinates.len(), 1, "one ring");
            let vertices = expected["big_ring_vertices"].as_u64().unwrap() as usize;
            let ring = &polygon.coordinates[0];
            assert_eq!(
                ring.len(),
                vertices,
                "a truncated payload would lose vertices off the end",
            );
            // …and every vertex is the one the generator wrote, not merely the
            // right *number* of vertices. `gen_fixtures.py` lays the ring out
            // as `i` of `n - 1` steps around a circle and then repeats vertex 0
            // to close it, so recomputing that formula here pins the whole
            // sequence: a reassembly that reordered, duplicated or corrupted
            // anything strictly between the endpoints passes a count-and-close
            // check and fails this one. The bytes themselves are exact doubles,
            // so the only slack needed is for Python's and Rust's libm
            // disagreeing in the last ulp of `cos`/`sin`.
            for (index, vertex) in ring.iter().enumerate() {
                let step = if index + 1 == vertices { 0 } else { index };
                let angle = 2.0 * std::f64::consts::PI * (step as f64) / ((vertices - 1) as f64);
                let lon = 140.0 + 0.5 * angle.cos();
                let lat = 36.0 + 0.5 * angle.sin();
                assert!(
                    (vertex[0] - lon).abs() < 1e-9,
                    "vertex {index} lon was {} not {lon}",
                    vertex[0],
                );
                assert!(
                    (vertex[1] - lat).abs() < 1e-9,
                    "vertex {index} lat was {} not {lat}",
                    vertex[1],
                );
            }
            // The last vertex closes the ring onto the first — it lives at the
            // very end of the overflow chain, so a chain that stopped early
            // cannot pass this.
            assert_eq!(ring[0], ring[ring.len() - 1], "the ring must close");
        }
        other => panic!("expected a Polygon, got {other:?}"),
    }

    match parks.features.features[0]
        .geometry
        .as_ref()
        .expect("geometry")
    {
        Geometry::Polygon(polygon) => assert_eq!(
            polygon.coordinates.len(),
            1 + expected["donut_holes"].as_u64().unwrap() as usize,
            "the donut keeps its hole",
        ),
        other => panic!("expected a Polygon, got {other:?}"),
    }
}

#[test]
fn a_web_mercator_table_is_inverse_projected_and_its_envelope_skipped() {
    // `roads` is the only fixture row with a GP envelope (indicator 1, 32
    // bytes). Skipping it by the wrong amount makes the WKB parse fail
    // outright, so a green assertion here is evidence the indicator table is
    // right as much as the projection is.
    let truth = truth(BASIC_TRUTH);
    let expected = &truth["tables"]["roads"];
    let dataset = load(BASIC);
    let roads = table(&dataset, "roads");
    assert_eq!(roads.features.features.len(), 1);

    let stored = expected["first_vertex_3857"].as_array().unwrap();
    let projected =
        oxigis_render::MercatorPoint::new(stored[0].as_f64().unwrap(), stored[1].as_f64().unwrap())
            .to_lon_lat();
    let (lon, lat) = first_vertex(
        roads.features.features[0]
            .geometry
            .as_ref()
            .expect("geometry"),
    );
    assert!((lon - projected.lon).abs() < 1e-9, "lon was {lon}");
    assert!((lat - projected.lat).abs() < 1e-9, "lat was {lat}");

    // …and that really is Tokyo, not a metre value passed through as degrees.
    // The truth's lon/lat is a hand-computed sanity figure rather than a
    // conversion of the stored metres, hence the loose bound; the exact
    // assertion above is the one that pins the projection.
    let approx = expected["first_vertex_lonlat_approx"].as_array().unwrap();
    assert!(
        (lon - approx[0].as_f64().unwrap()).abs() < 0.05,
        "lon was {lon}"
    );
    assert!(
        (lat - approx[1].as_f64().unwrap()).abs() < 0.05,
        "lat was {lat}"
    );
}

#[test]
fn the_fixture_headers_say_what_their_truth_files_say() {
    // Provenance: proves the committed bytes are the ones gen_fixtures.py
    // described, so every other assertion here is about the file it claims.
    for (bytes, text) in [
        (BASIC, BASIC_TRUTH),
        (PAGED, PAGED_TRUTH),
        (UNICODE, UNICODE_TRUTH),
    ] {
        let truth = truth(text);
        let db = SqliteDb::open(bytes).expect("a readable image");
        let page_size = truth["page_size"].as_u64().unwrap() as usize;
        assert_eq!(db.usable_size(), page_size, "no page reserves any bytes");
        let pages = truth["page_count"].as_u64().unwrap() as usize;
        assert_eq!(bytes.len(), pages * page_size);
    }
}

#[test]
fn the_spatial_ref_sys_table_decodes_its_negative_ids() {
    // -1 is stored as the single byte 0xFF: read without sign extension it is
    // 255, and the "undefined SRS" rows silently become unknown ones.
    let db = SqliteDb::open(BASIC).expect("a readable image");
    let master = db.master_entries().expect("a readable catalogue");
    let rows = super::spatial_ref_sys(&db, &master).expect("a readable SRS table");
    let undefined = rows.get(&-1).expect("the undefined cartesian SRS row");
    assert_eq!(undefined.organization, "NONE");
    assert_eq!(undefined.code, -1);
    let wgs84 = rows.get(&4326).expect("the WGS 84 row");
    assert_eq!(wgs84.organization, "EPSG");
    assert_eq!(wgs84.code, 4326);
    assert!(rows.contains_key(&0));
}

#[test]
fn the_srs_policy_resolves_by_authority_and_names_whatever_it_refuses() {
    let db = SqliteDb::open(BASIC).expect("a readable image");
    let master = db.master_entries().expect("a readable catalogue");
    let rows = super::spatial_ref_sys(&db, &master).expect("a readable SRS table");
    assert_eq!(resolve_srs(4326, &rows).epsg(), 4326);
    assert_eq!(resolve_srs(3857, &rows).epsg(), 3857);
    assert!(
        resolve_srs(0, &rows).is_wgs84(),
        "an undefined SRS reads as WGS 84, as a missing .prj does",
    );
    assert!(resolve_srs(-1, &rows).is_wgs84());

    // Lambert-93 is a real CRS this build cannot invert: it is refused, and
    // the refusal quotes the name the FILE gave it.
    let lambert = resolve_srs(2154, &rows);
    assert!(!lambert.is_supported());
    assert!(GpkgCrs::for_crs(&lambert).is_none());
    assert!(
        lambert.name().contains("Lambert-93"),
        "named {:?}",
        lambert.name(),
    );

    // An id the file has no row for falls back to reading the id itself…
    let empty = std::collections::BTreeMap::new();
    assert_eq!(resolve_srs(4326, &empty).epsg(), 4326);
    // …which for UTM 54N now LOADS rather than being refused (finding 203's
    // asymmetry: the COG path already handled UTM while this one did not).
    let utm = resolve_srs(32654, &empty);
    assert_eq!(utm.epsg(), 32654);
    assert!(GpkgCrs::for_crs(&utm).is_some());
    // …and still names itself numerically when even that is unknown.
    let unknown = resolve_srs(999_999, &empty);
    assert!(!unknown.is_supported());
    assert!(GpkgCrs::for_crs(&unknown).is_none());
}

#[test]
fn a_row_whose_organization_is_not_epsg_falls_back_to_its_wkt_definition() {
    // `srs_id` is a file-local primary key: a writer that renumbers its SRS
    // table, or records `organization = 'NONE'`, leaves the WKT `definition`
    // as the only real CRS information in the file.
    let mut rows = std::collections::BTreeMap::new();
    rows.insert(
        7_i64,
        super::SrsRow {
            name: "custom".to_string(),
            organization: "NONE".to_string(),
            code: 7,
            definition:
                r#"PROJCS["JGD2011 / Japan Plane Rectangular CS IX",AUTHORITY["EPSG","6677"]]"#
                    .to_string(),
        },
    );
    assert_eq!(resolve_srs(7, &rows).epsg(), 6677);

    // An EPSG row whose code this build cannot place still tries the WKT.
    rows.insert(
        8,
        super::SrsRow {
            name: "renumbered".to_string(),
            organization: "EPSG".to_string(),
            code: 999_999,
            definition: r#"PROJCS["WGS 84 / UTM zone 54N",AUTHORITY["EPSG","32654"]]"#.to_string(),
        },
    );
    assert_eq!(resolve_srs(8, &rows).epsg(), 32654);

    // The specification's own `definition = 'undefined'` rows are not parsed.
    rows.insert(
        0,
        super::SrsRow {
            name: "Undefined geographic SRS".to_string(),
            organization: "NONE".to_string(),
            code: 0,
            definition: "undefined".to_string(),
        },
    );
    assert!(resolve_srs(0, &rows).is_wgs84());
}

// ---- paged.gpkg ------------------------------------------------------------

#[test]
fn a_table_spanning_interior_pages_yields_every_row_in_rowid_order() {
    let truth = truth(PAGED_TRUTH);
    let expected = &truth["tables"]["pts"];
    let dataset = load(PAGED);
    let points = table(&dataset, "pts");
    let count = expected["feature_count"].as_u64().unwrap() as usize;
    assert_eq!(
        points.features.features.len(),
        count,
        "at page size 512 this table needs interior b-tree pages",
    );
    assert_eq!(
        fids(points),
        (1..=count as i64).collect::<Vec<i64>>(),
        "rows must come back in rowid order, none missing",
    );

    // Every 37th row carries a 600-character tag, which at a 477-byte inline
    // maximum can only arrive whole through an overflow chain.
    let mut long = 0usize;
    for feature in &points.features.features {
        let tag = feature.properties.as_ref().expect("properties")["tag"]
            .as_str()
            .expect("a text tag");
        assert!(
            tag.len() == 17 || tag.len() == 609,
            "a partially reassembled tag would be some other length: {}",
            tag.len(),
        );
        long += usize::from(tag.len() == 609);
    }
    assert_eq!(long, expected["overflow_rows"].as_u64().unwrap() as usize);

    let first = first_vertex(
        points.features.features[0]
            .geometry
            .as_ref()
            .expect("geometry"),
    );
    let wanted = expected["first_point_lonlat"].as_array().unwrap();
    assert!((first.0 - wanted[0].as_f64().unwrap()).abs() < 1e-9);
    assert!((first.1 - wanted[1].as_f64().unwrap()).abs() < 1e-9);
    let last = first_vertex(
        points.features.features[count - 1]
            .geometry
            .as_ref()
            .expect("geometry"),
    );
    let wanted = expected["last_point_lonlat"].as_array().unwrap();
    assert!((last.0 - wanted[0].as_f64().unwrap()).abs() < 1e-9);
    assert!((last.1 - wanted[1].as_f64().unwrap()).abs() < 1e-9);
}

// ---- unicode.gpkg ----------------------------------------------------------

#[test]
fn unquoted_non_ascii_identifiers_keep_every_column_in_its_place() {
    // SQLite's tokenizer treats every byte >= 0x80 as an identifier character,
    // so this table's name and three of its columns are non-ASCII *without*
    // quotes — the truth file carries the statement SQLite actually stored, so
    // the assertion below proves that rather than assuming it.
    //
    // A parser that stops at those bytes drops those columns from the derived
    // schema, and then `normalize`'s resize truncates each record from the
    // *tail* instead of removing the missing slot: every later column reads the
    // value of the one before it, and the geometry BLOB — the last thing in the
    // record — is discarded outright, so every feature in the table silently
    // comes back with no geometry at all.
    let truth = truth(UNICODE_TRUTH);
    let expected = &truth["tables"]["\u{5730}\u{70b9}"];
    let statement = expected["create_sql"].as_str().unwrap();
    assert!(
        !statement.contains('"') && !statement.contains('`') && !statement.contains('['),
        "the fixture only tests what it claims if the identifiers really are unquoted: {statement}",
    );

    let dataset = load(UNICODE);
    assert!(dataset.notices().is_empty(), "{:?}", dataset.notices());
    let table = table(&dataset, "\u{5730}\u{70b9}");
    let count = expected["feature_count"].as_u64().unwrap() as usize;
    assert_eq!(table.features.features.len(), count);

    let mut columns: Vec<String> = table.features.features[0]
        .properties
        .as_ref()
        .expect("properties")
        .keys()
        .cloned()
        .collect();
    let mut wanted: Vec<String> = expected["columns"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .filter(|column| *column != "geom")
        .map(str::to_string)
        .collect();
    columns.sort();
    wanted.sort();
    assert_eq!(columns, wanted, "a dropped column shortens this list");

    // Every attribute must sit under its own name — the check a shifted schema
    // fails even when the column *count* happens to come out right.
    for (index, feature) in table.features.features.iter().enumerate() {
        let row = feature.properties.as_ref().expect("properties");
        assert_eq!(row["\u{540d}\u{524d}"], expected["names"][index]);
        assert_eq!(row["\u{4eba}\u{53e3}"], expected["populations"][index]);
        assert_eq!(row["\u{6a19}\u{9ad8}"], expected["elevations"][index]);
        assert!(
            feature.geometry.is_some(),
            "row {index} lost its geometry, which is what a truncating resize does",
        );
    }

    let (lon, lat) = first_vertex(
        table.features.features[0]
            .geometry
            .as_ref()
            .expect("geometry"),
    );
    let wanted = expected["first_point_lonlat"].as_array().unwrap();
    assert!((lon - wanted[0].as_f64().unwrap()).abs() < 1e-9);
    assert!((lat - wanted[1].as_f64().unwrap()).abs() < 1e-9);
}

#[test]
fn a_table_level_primary_key_with_a_sort_order_is_still_the_rowid_alias() {
    // `unicode.gpkg`'s table is keyed by `PRIMARY KEY (fid DESC)`, which real
    // SQLite treats as a rowid alias exactly like the bare form — verified
    // against SQLite 3.37.2 both by `UPDATE t SET rowid = 999` (fid moves with
    // it) and by dumping the record, whose fid slot is serial type 0, i.e.
    // NULL. Only the *column-level* `INTEGER PRIMARY KEY DESC` spelling breaks
    // the alias. Miss that and every fid in the table reads as JSON null.
    let truth = truth(UNICODE_TRUTH);
    let expected = &truth["tables"]["\u{5730}\u{70b9}"];
    assert!(
        expected["create_sql"]
            .as_str()
            .unwrap()
            .contains("PRIMARY KEY (fid DESC)"),
    );
    let dataset = load(UNICODE);
    let wanted: Vec<i64> = expected["fids"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_i64)
        .collect();
    assert_eq!(
        fids(table(&dataset, "\u{5730}\u{70b9}")),
        wanted,
        "the rowid was not substituted into the aliased column",
    );
}

// ---- without_rowid.gpkg ----------------------------------------------------

#[test]
fn a_without_rowid_feature_table_is_refused_rather_than_guessed_at() {
    // Malformed per the GeoPackage spec (a feature table must have an INTEGER
    // PRIMARY KEY), and its root page is an index b-tree with no rowids to key
    // rows by. The file still parses; the table just does not load.
    let dataset = load(WITHOUT_ROWID);
    assert!(dataset.tables().is_empty());
    assert_eq!(dataset.notices().len(), 1);
    let notice = &dataset.notices()[0];
    assert!(notice.contains("weird"), "{notice}");
    assert!(notice.contains("WITHOUT ROWID"), "{notice}");
}

// ---- the persistence bridge ------------------------------------------------

#[test]
fn a_table_survives_the_round_trip_through_geojson_text() {
    // The browser persistence leg: no path to reference, so the table is stored
    // as GeoJSON text and re-read through the ordinary inline path.
    let dataset = load(BASIC);
    let parks = table(&dataset, "parks");
    let text = super::to_geojson_string(&parks.features).expect("serialise");
    let reparsed = crate::local_input::parse_geojson(&text).expect("reparse");
    assert_eq!(reparsed.features.len(), parks.features.features.len());
    match reparsed.features[1].geometry.as_ref().expect("geometry") {
        Geometry::Polygon(polygon) => assert_eq!(polygon.coordinates[0].len(), 600),
        other => panic!("expected a Polygon, got {other:?}"),
    }
    assert_eq!(
        reparsed.features[0].properties.as_ref().expect("row")["name"],
        serde_json::json!("Donut Park"),
    );
}

// ---- reprojecting CRSs -----------------------------------------------------

/// A one-feature GeoPackage whose geometry column is registered in `srs_id`,
/// holding a single point at `(x, y)` in that CRS's own units.
///
/// The image carries no `gpkg_spatial_ref_sys` table, which is the "the id
/// itself is the authority code" path — the convention every writer follows for
/// an id it did not register, and the shortest way to state "this table is in
/// EPSG:n" as bytes.
fn image_in_srs(srs_id: i64, x: f64, y: f64) -> Vec<u8> {
    use super::fixture::{Cell, geopackage_image, gp_blob, wkb_point};
    let blob = gp_blob(0x01, srs_id as i32, &[], &wkb_point(x, y));
    geopackage_image(
        Cell::Int(srs_id),
        "CREATE TABLE t (fid INTEGER PRIMARY KEY, geom BLOB)",
        &[Cell::Null, Cell::Blob(&blob)],
    )
}

#[test]
fn a_jgd2011_plane_rectangular_table_loads_and_lands_in_tokyo() {
    // The GeoPackage twin of the shapefile case the whole feature exists for.
    // EPSG:6677 used to be refused here; the coordinates are metres from a
    // 139°50'E / 36°00'N origin.
    let bytes = image_in_srs(6677, -5_995.185_165_976_9, -35_367.230_136_018_2);
    let dataset = load(&bytes);
    assert!(dataset.refusals().is_empty(), "{:?}", dataset.notices());
    let table = table(&dataset, "t");
    assert_eq!(table.crs.epsg(), 6677, "the CRS is recorded on the layer");
    let (lon, lat) = first_vertex(
        table.features.features[0]
            .geometry
            .as_ref()
            .expect("geometry"),
    );
    assert!((lon - 139.7671).abs() < 1e-6, "lon {lon}");
    assert!((lat - 35.6812).abs() < 1e-6, "lat {lat}");
}

#[test]
fn the_zone_origin_of_every_japanese_plane_rectangular_srs_lands_on_its_meridian() {
    // `x_0 = y_0 = 0` for all 57 of them, so a point at (0, 0) is exactly the
    // zone's own origin — a cheap, exact check that no zone is mis-registered.
    for base in [6669_i64, 2443] {
        for zone in 0..19_i64 {
            let srs = base + zone;
            let bytes = image_in_srs(srs, 0.0, 0.0);
            let dataset = load(&bytes);
            let table = table(&dataset, "t");
            assert_eq!(table.crs.epsg(), srs as u32);
            let (lon, lat) = first_vertex(
                table.features.features[0]
                    .geometry
                    .as_ref()
                    .expect("geometry"),
            );
            let def = table
                .crs
                .definition()
                .unwrap_or_else(|| panic!("EPSG:{srs}"));
            let oxigis_core::Projection::TransverseMercator {
                latitude_of_origin_deg,
                central_meridian_deg,
                ..
            } = def.projection
            else {
                panic!("EPSG:{srs} must be a Transverse Mercator");
            };
            assert!(
                (lon - central_meridian_deg).abs() < 1e-6,
                "EPSG:{srs} lon {lon}"
            );
            assert!(
                (lat - latitude_of_origin_deg).abs() < 1e-6,
                "EPSG:{srs} lat {lat}"
            );
        }
    }
}

#[test]
fn a_utm_table_now_loads_instead_of_being_refused() {
    let bytes = image_in_srs(32654, 388_433.374_620_895, 3_949_290.013_641_47);
    let dataset = load(&bytes);
    assert!(dataset.refusals().is_empty(), "{:?}", dataset.notices());
    let table = table(&dataset, "t");
    assert_eq!(table.crs.epsg(), 32654);
    let (lon, lat) = first_vertex(
        table.features.features[0]
            .geometry
            .as_ref()
            .expect("geometry"),
    );
    assert!((lon - 139.7671).abs() < 1e-5, "lon {lon}");
    assert!((lat - 35.6812).abs() < 1e-5, "lat {lat}");
}

#[test]
fn a_table_in_a_crs_this_build_cannot_place_is_still_refused_on_its_own() {
    // Lambert-93: a real CRS, a projection family this build does not invert.
    // The table is refused, the refusal names the code, and — the property
    // that matters for a multi-table file — nothing else is affected.
    let bytes = image_in_srs(2154, 650_000.0, 6_860_000.0);
    let dataset = load(&bytes);
    assert!(dataset.tables().is_empty());
    let refusal = dataset.refusals().first().expect("a refusal");
    assert_eq!(refusal.table(), "t");
    assert!(
        refusal.message().contains("EPSG:2154"),
        "{}",
        refusal.message()
    );
    assert!(refusal.message().contains("4326"), "{}", refusal.message());
}

#[test]
fn a_wgs84_table_records_no_crs_on_its_layer() {
    // WGS 84 is the absent form: `Layer::with_crs` drops it, so an ordinary
    // GeoPackage produces a project file with no `crs` keys at all.
    let bytes = image_in_srs(4326, 139.7671, 35.6812);
    let dataset = load(&bytes);
    let table = table(&dataset, "t");
    assert!(table.crs.is_wgs84());
    let layer = oxigis_core::Layer::new(
        "t",
        oxigis_core::LayerKind::Vector(oxigis_core::VectorSource::LocalGpkg {
            path: "a.gpkg".to_string(),
            table: "t".to_string(),
        }),
    )
    .with_crs(table.crs.clone());
    assert_eq!(layer.crs, None);
}
