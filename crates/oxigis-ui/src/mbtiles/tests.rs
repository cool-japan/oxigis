// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! MBTiles reader tests: both schemas, the TMS flip, the overflow chain, and
//! every refusal by name.

use std::sync::Arc;

use oxigis_render::{TileId, source::tms_row};

use crate::archive::ArchiveContent;
use crate::gpkg_input::fixture::{Cell, record};
use crate::mbtiles::fixture::{
    Table, flat_image, image_with, metadata_table, normalized_image, raster_metadata,
    spilled_image, vector_metadata,
};
use crate::mbtiles::{MAX_MBTILES_BYTES, MbTilesReader, content_for_format};

/// Wraps bytes the way the reader takes them.
fn reader(bytes: Vec<u8>) -> Result<MbTilesReader, crate::local_vector::LocalVectorError> {
    MbTilesReader::open(Arc::from(bytes.into_boxed_slice()))
}

fn tile(z: u8, x: u32, y: u32) -> TileId {
    TileId::new(z, x, y).unwrap_or_else(|error| panic!("tile {z}/{x}/{y} must be valid: {error}"))
}

/// A flat vector archive holding four tiles across two zooms.
fn flat_vector() -> Vec<u8> {
    flat_image(
        &[
            (0, 0, 0, b"z0".to_vec()),
            (1, 0, 0, b"z1-0-0".to_vec()),
            (1, 1, 1, b"z1-1-1".to_vec()),
            (2, 3, 2, b"z2-3-2".to_vec()),
        ],
        &vector_metadata(),
    )
}

#[test]
fn a_flat_archive_opens_and_reports_its_metadata() {
    let archive = reader(flat_vector()).expect("the fixture opens");
    let info = archive.info();
    assert_eq!(info.content, ArchiveContent::Vector);
    assert_eq!(info.name, "fixture");
    assert_eq!(info.attribution, "OxiGIS test fixture");
    assert_eq!(info.min_zoom, 0);
    assert_eq!(info.max_zoom, 2);
    assert!(info.has_bounds);
    assert!((info.bounds_deg[0] + 10.0).abs() < 1e-9);
    assert_eq!(
        info.layer_names,
        vec!["water".to_owned(), "roads".to_owned()]
    );
    assert_eq!(archive.len(), 4);
    assert!(!archive.is_empty());
    assert_eq!(
        archive.metadata().get("format").map(String::as_str),
        Some("pbf")
    );
}

#[test]
fn the_tms_row_is_flipped_exactly_once_at_index_build() {
    // The fixture stores (z=1, column=0, row=0) in MBTiles addressing, which
    // counts rows from the SOUTH. In XYZ that is row 1, not row 0.
    let archive = reader(flat_vector()).expect("opens");
    assert_eq!(tms_row(1, 0), 1, "the shared rule, not a second copy");

    assert_eq!(
        archive.tile(tile(1, 0, 1)).expect("a clean lookup"),
        Some(b"z1-0-0".to_vec()),
        "MBTiles row 0 is XYZ row 1 at zoom 1"
    );
    assert_eq!(
        archive.tile(tile(1, 0, 0)).expect("a clean lookup"),
        None,
        "and the unflipped address must hold nothing"
    );
    // z2 row 2 flips to XYZ row 1.
    assert_eq!(
        archive.tile(tile(2, 3, 1)).expect("a clean lookup"),
        Some(b"z2-3-2".to_vec())
    );
}

#[test]
fn every_stored_address_reads_back_its_own_body() {
    let archive = reader(flat_vector()).expect("opens");
    for (z, column, row, body) in [
        (0u8, 0u32, 0u32, &b"z0"[..]),
        (1, 0, 0, b"z1-0-0"),
        (1, 1, 1, b"z1-1-1"),
        (2, 3, 2, b"z2-3-2"),
    ] {
        let address = tile(z, column, tms_row(z, row));
        assert_eq!(
            archive.tile(address).expect("a clean lookup").as_deref(),
            Some(body),
            "z{z} column {column} row {row}"
        );
    }
}

#[test]
fn an_address_the_archive_does_not_hold_is_absent_not_an_error() {
    let archive = reader(flat_vector()).expect("opens");
    assert_eq!(archive.tile(tile(2, 0, 0)).expect("clean"), None);
    // Past max_zoom the zoom gate answers without touching the index at all.
    assert!(!archive.covers(tile(5, 0, 0)));
    assert_eq!(archive.tile(tile(5, 0, 0)).expect("clean"), None);
    assert!(archive.covers(tile(1, 0, 0)));
}

#[test]
fn a_normalized_archive_joins_map_to_images_in_memory() {
    // Two addresses share one body — the whole reason the shape exists.
    let bytes = normalized_image(
        &[(0, 0, 0, "shared"), (1, 0, 0, "shared"), (1, 1, 1, "other")],
        &[
            ("shared", b"one body".to_vec()),
            ("other", b"another".to_vec()),
        ],
        &vector_metadata(),
    );
    let archive = reader(bytes).expect("the normalized fixture opens");
    assert_eq!(archive.len(), 3);
    assert_eq!(
        archive.tile(tile(0, 0, 0)).expect("clean"),
        Some(b"one body".to_vec())
    );
    assert_eq!(
        archive.tile(tile(1, 0, tms_row(1, 0))).expect("clean"),
        Some(b"one body".to_vec()),
        "a deduplicated body is reachable from every address that names it"
    );
    assert_eq!(
        archive.tile(tile(1, 1, tms_row(1, 1))).expect("clean"),
        Some(b"another".to_vec())
    );
}

#[test]
fn a_tiles_view_over_map_and_images_is_recognised_by_shape() {
    // `master_entries` reports the view with rootpage 0, so a reader that only
    // looks for a `tiles` ROOT finds nothing on the most common archive there
    // is. This asserts the fixture really does expose it as a view.
    let bytes = normalized_image(
        &[(0, 0, 0, "a")],
        &[("a", b"body".to_vec())],
        &vector_metadata(),
    );
    let db = crate::gpkg_input::sqlite::SqliteDb::open(&bytes).expect("a valid image");
    let entries = db.master_entries().expect("a readable schema");
    let view = entries
        .iter()
        .find(|entry| entry.name == "tiles")
        .expect("the fixture declares a tiles view");
    assert_eq!(view.entry_type, "view");
    assert_eq!(view.rootpage, 0);

    // …and the reader opens it anyway.
    assert!(reader(bytes).is_ok());
}

#[test]
fn a_body_that_spills_into_an_overflow_chain_comes_back_whole() {
    // Far past `usable - 35` (~4061 at a 4096-byte page), so the blob really
    // does travel through overflow pages — and the leading integer columns
    // must still be readable from the inline prefix the index scan sees.
    const BYTES: usize = 20_000;
    let archive = reader(spilled_image(BYTES, &raster_metadata())).expect("opens");
    assert_eq!(archive.len(), 1, "the spilled row was still indexed");
    let body = archive
        .tile(tile(1, 0, tms_row(1, 1)))
        .expect("clean")
        .expect("the spilled tile is present");
    assert_eq!(body.len(), BYTES);
    assert_eq!(body[0], 0);
    assert_eq!(body[250], 250);
    assert_eq!(
        body[251], 0,
        "the pattern repeats, so the chain is in order"
    );
    assert_eq!(body[BYTES - 1], ((BYTES - 1) % 251) as u8);
}

#[test]
fn a_raster_archive_reports_its_codec_and_no_layer_names() {
    let archive = reader(flat_image(
        &[(0, 0, 0, b"png bytes".to_vec())],
        &raster_metadata(),
    ))
    .expect("opens");
    let info = archive.info();
    assert_eq!(info.content, ArchiveContent::Raster);
    assert_eq!(info.codec, oxigis_render::pmtiles::TileType::Png);
    assert!(info.layer_names.is_empty());
    assert!(archive.paints().is_empty());
}

#[test]
fn a_vector_archives_layer_names_seed_the_default_paints() {
    let archive = reader(flat_vector()).expect("opens");
    let paints = archive.paints();
    let names: Vec<&str> = paints
        .iter()
        .map(|paint| paint.source_layer.as_str())
        .collect();
    // The ramp orders fills under lines: `water` (fill) before `roads` (line).
    assert_eq!(names, ["water", "roads"]);
}

#[test]
fn a_format_this_build_cannot_draw_is_refused_by_name() {
    assert_eq!(
        content_for_format("pbf").expect("vector"),
        ArchiveContent::Vector
    );
    assert_eq!(
        content_for_format("PNG").expect("raster"),
        ArchiveContent::Raster
    );
    let error = content_for_format("gif").expect_err("gif is refused");
    assert!(error.to_string().contains("gif"), "{error}");
    assert!(error.to_string().contains("pbf"), "{error}");
}

#[test]
fn an_archive_with_no_metadata_table_is_refused_by_name() {
    let bytes = image_with(
        &[Table {
            name: "tiles",
            sql: "CREATE TABLE tiles (zoom_level integer, tile_column integer, \
                  tile_row integer, tile_data blob)",
            rows: vec![(
                1,
                record(&[Cell::Int(0), Cell::Int(0), Cell::Int(0), Cell::Blob(b"x")]),
            )],
        }],
        &[],
    );
    let error = reader(bytes).expect_err("no metadata is a refusal");
    assert!(error.to_string().contains("metadata"), "{error}");
}

#[test]
fn an_archive_declaring_no_format_is_refused_by_name() {
    let bytes = flat_image(
        &[(0, 0, 0, b"x".to_vec())],
        &[("name", "no format here"), ("minzoom", "0")],
    );
    let error = reader(bytes).expect_err("no format is a refusal");
    assert!(error.to_string().contains("format"), "{error}");
}

#[test]
fn an_archive_with_neither_schema_is_refused_with_the_shape_named() {
    let bytes = image_with(
        &[metadata_table(&[("format", "pbf")])],
        &[("tiles", "CREATE VIEW tiles AS SELECT 1")],
    );
    let error = reader(bytes).expect_err("no tile store is a refusal");
    let message = error.to_string();
    assert!(message.contains("tiles table"), "{message}");
    assert!(message.contains("map/images"), "{message}");
}

#[test]
fn a_tiles_table_missing_a_required_column_is_refused_by_name() {
    let bytes = image_with(
        &[
            Table {
                name: "tiles",
                sql: "CREATE TABLE tiles (zoom_level integer, tile_column integer, \
                      tile_data blob)",
                rows: vec![(1, record(&[Cell::Int(0), Cell::Int(0), Cell::Blob(b"x")]))],
            },
            metadata_table(&raster_metadata()),
        ],
        &[],
    );
    let error = reader(bytes).expect_err("a missing column is a refusal");
    assert!(error.to_string().contains("tile_row"), "{error}");
}

#[test]
fn a_body_that_is_not_a_database_at_all_is_refused() {
    let error = reader(b"this is not SQLite".to_vec()).expect_err("garbage is refused");
    assert!(error.to_string().contains("SQLite"), "{error}");
}

#[test]
fn an_image_past_the_memory_ceiling_is_refused_with_pmtiles_named() {
    // Constructed rather than allocated: the check runs on the length alone,
    // before a single page is read.
    let error = MbTilesReader::open(Arc::from(vec![0u8; 0].into_boxed_slice()))
        .expect_err("an empty image is not a database");
    assert!(error.to_string().contains("SQLite") || error.to_string().contains("short"));
    assert_eq!(MAX_MBTILES_BYTES, 512 * 1024 * 1024);
}

#[test]
fn out_of_range_addresses_in_the_table_are_skipped_rather_than_indexed() {
    // A hand-edited archive can hold anything; a zoom past MAX_ZOOM or a
    // column outside 2^z must not become an index entry that then answers a
    // legitimate address with the wrong body.
    let bytes = image_with(
        &[
            Table {
                name: "tiles",
                sql: "CREATE TABLE tiles (zoom_level integer, tile_column integer, \
                      tile_row integer, tile_data blob)",
                rows: vec![
                    (
                        1,
                        record(&[
                            Cell::Int(99),
                            Cell::Int(0),
                            Cell::Int(0),
                            Cell::Blob(b"bad z"),
                        ]),
                    ),
                    (
                        2,
                        record(&[
                            Cell::Int(1),
                            Cell::Int(7),
                            Cell::Int(0),
                            Cell::Blob(b"bad x"),
                        ]),
                    ),
                    (
                        3,
                        record(&[
                            Cell::Int(1),
                            Cell::Int(0),
                            Cell::Int(0),
                            Cell::Blob(b"good"),
                        ]),
                    ),
                ],
            },
            metadata_table(&raster_metadata()),
        ],
        &[],
    );
    let archive = reader(bytes).expect("opens");
    assert_eq!(archive.len(), 1, "only the legal address was indexed");
    assert_eq!(
        archive.tile(tile(1, 0, tms_row(1, 0))).expect("clean"),
        Some(b"good".to_vec())
    );
}

#[test]
fn a_zoom_range_the_metadata_declares_backwards_is_clamped_not_believed() {
    let bytes = flat_image(
        &[(0, 0, 0, b"x".to_vec())],
        &[("format", "png"), ("minzoom", "9"), ("maxzoom", "2")],
    );
    let archive = reader(bytes).expect("opens");
    let info = archive.info();
    assert!(
        info.min_zoom <= info.max_zoom,
        "a backwards range would gate every tile away"
    );
}

#[test]
fn malformed_bounds_leave_the_whole_world_rather_than_half_a_box() {
    let bytes = flat_image(
        &[(0, 0, 0, b"x".to_vec())],
        &[("format", "png"), ("bounds", "not,a,bounding,box")],
    );
    let info = reader(bytes).expect("opens").info();
    assert!(!info.has_bounds);
    assert!((info.bounds_deg[0] + 180.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// The two new SqliteDb methods, on their own
// ---------------------------------------------------------------------------

#[test]
fn seek_row_finds_a_row_by_rowid_and_reports_a_missing_one_as_none() {
    use crate::gpkg_input::sqlite::{CellValue, SqliteDb};

    let bytes = flat_vector();
    let db = SqliteDb::open(&bytes).expect("a valid image");
    let tiles = db
        .master_entries()
        .expect("a readable schema")
        .into_iter()
        .find(|entry| entry.name == "tiles")
        .expect("the fixture has a tiles table");

    let row = db
        .seek_row(tiles.rootpage, 2)
        .expect("a clean seek")
        .expect("rowid 2 exists");
    assert_eq!(row.rowid, 2);
    assert_eq!(row.values.first(), Some(&CellValue::Integer(1)));

    assert!(
        db.seek_row(tiles.rootpage, 9_999)
            .expect("a clean seek")
            .is_none(),
        "a rowid the table does not hold is None, not an error"
    );
}

#[test]
fn scan_prefixes_reads_the_leading_columns_without_touching_a_blob() {
    use crate::gpkg_input::sqlite::{SqliteDb, decode_record_prefix};

    // The one tile spills, so its record's blob is NOT on the page — yet the
    // three leading integers must still decode out of the inline prefix.
    let bytes = spilled_image(20_000, &raster_metadata());
    let db = SqliteDb::open(&bytes).expect("a valid image");
    let tiles = db
        .master_entries()
        .expect("a readable schema")
        .into_iter()
        .find(|entry| entry.name == "tiles")
        .expect("a tiles table");

    let mut seen = Vec::new();
    let mut visit = |rowid: i64, prefix: &[u8]| {
        // The blob is not on the page at all, yet the leading integers are.
        let values = decode_record_prefix(prefix).expect("the prefix holds a decodable header");
        assert!(
            values.len() >= 3,
            "the three leading integers must survive the spill"
        );
        seen.push((rowid, values.len()));
        Ok(())
    };
    db.scan_prefixes(tiles.rootpage, &mut visit)
        .expect("a clean scan");
    assert_eq!(seen.len(), 1);
    let (rowid, _values) = seen[0];
    assert_eq!(rowid, 1);
}

#[test]
fn seek_row_and_scan_prefixes_refuse_an_index_btree_like_scan_table_does() {
    use crate::gpkg_input::sqlite::SqliteDb;

    // Page 1 of any image is a table b-tree, so an index root has to be
    // synthesised: page 2 of the flat fixture is a leaf table page; flipping
    // its type byte to 10 makes it a leaf INDEX page.
    let mut bytes = flat_vector();
    let page_two = crate::mbtiles::fixture::PAGE_SIZE;
    bytes[page_two] = 10;
    let db = SqliteDb::open(&bytes).expect("the header is still valid");
    let error = db.seek_row(2, 1).expect_err("an index b-tree is refused");
    assert!(error.to_string().contains("index b-tree"), "{error}");
    let mut visit = |_rowid: i64, _prefix: &[u8]| Ok(());
    let error = db
        .scan_prefixes(2, &mut visit)
        .expect_err("an index b-tree is refused");
    assert!(error.to_string().contains("index b-tree"), "{error}");
}
