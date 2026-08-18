// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tier 2: the cases no writer produces.
//!
//! Every input here is either assembled byte by byte (see [`super::fixture`])
//! or a real fixture with one field damaged. The bar is the same throughout:
//! **an error, never a panic and never a hang** — this parser is fed files the
//! user dropped on a map, so a cyclic overflow chain or a four-billion-element
//! ring has to come back as a message.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use oxigeo::geojson::types::Geometry;

use super::fixture::{
    BASIC, Cell, Image, geopackage_image, gp_blob, one_table_image, record, record_of_size,
    varint as encode_varint, wkb_line, wkb_point, wkb_point_be,
};
use super::geometry::{GpkgCrs, decode};
use super::sqlite::{CellValue, SqliteDb, decode_record, parse_create_table, varint};

/// Page size the hand-built images use: small enough that a few hundred bytes
/// already spill, large enough to be legal (SQLite's minimum is 512).
const PAGE: usize = 512;

/// The largest payload that stays inline at [`PAGE`]: `usable - 35`.
const MAX_INLINE: usize = PAGE - 35;

/// Reads the only table of a hand-built image.
fn scan(image: &[u8]) -> Result<Vec<super::sqlite::Row>, crate::local_vector::LocalVectorError> {
    let db = SqliteDb::open(image)?;
    let master = db.master_entries()?;
    let entry = master.first().expect("one table");
    db.scan_table(entry.rootpage)
}

/// A one-row, one-BLOB-column image whose record payload is `total` bytes.
fn blob_image(total: usize) -> (Vec<u8>, Vec<u8>) {
    let (payload, blob) = record_of_size(total);
    (
        one_table_image(PAGE, "t", "CREATE TABLE t (b BLOB)", &[(1, payload)]),
        blob,
    )
}

/// Overwrites four bytes at the start of `page` (1-based).
fn patch_page_head(image: &mut [u8], page: u32, value: u32) {
    let at = (page as usize - 1) * PAGE;
    image[at..at + 4].copy_from_slice(&value.to_be_bytes());
}

// ---- varints and records ---------------------------------------------------

#[test]
fn a_nine_byte_varint_contributes_all_eight_bits_of_its_last_byte() {
    let encoded = encode_varint(u64::MAX);
    assert_eq!(encoded.len(), 9);
    assert_eq!(varint(&encoded, 0).expect("decodes"), (-1, 9));
    // …and the ordinary widths still round trip.
    for value in [0u64, 1, 127, 128, 16_383, 16_384, 1 << 40] {
        let encoded = encode_varint(value);
        let (decoded, used) = varint(&encoded, 0).expect("decodes");
        assert_eq!(decoded as u64, value);
        assert_eq!(used, encoded.len());
    }
}

#[test]
fn a_varint_that_runs_off_the_end_is_an_error() {
    assert!(varint(&[0x80, 0x80], 0).is_err());
    assert!(varint(&[], 0).is_err());
    assert!(varint(&[0x00], 4).is_err(), "an offset past the end");
    // Nine continuation bytes with no ninth byte behind them.
    assert!(varint(&[0xFF; 8], 0).is_err());
}

#[test]
fn the_bodyless_serial_types_decode_to_the_literals_they_stand_for() {
    let payload = record(&[
        Cell::Zero,
        Cell::One,
        Cell::Null,
        Cell::Int(-1),
        Cell::Real(1.5),
        Cell::Text("t"),
    ]);
    let values = decode_record(&payload).expect("a well-formed record");
    assert_eq!(
        values,
        vec![
            CellValue::Integer(0),
            CellValue::Integer(1),
            CellValue::Null,
            CellValue::Integer(-1),
            CellValue::Float(1.5),
            CellValue::Text("t".to_string()),
        ],
    );
}

#[test]
fn every_integer_width_sign_extends() {
    // Serial types 1..=6 are 1, 2, 3, 4, 6 and 8 bytes; each must come back
    // negative when its top bit is set.
    for (serial, width) in [(1u8, 1usize), (2, 2), (3, 3), (4, 4), (5, 6), (6, 8)] {
        let mut payload = vec![2u8, serial];
        payload.extend(std::iter::repeat_n(0xFFu8, width));
        assert_eq!(
            decode_record(&payload).expect("a well-formed record"),
            vec![CellValue::Integer(-1)],
            "serial type {serial} lost its sign",
        );
    }
}

#[test]
fn the_serial_types_sqlite_reserves_for_itself_are_refused() {
    for serial in [10u8, 11] {
        let payload = vec![2u8, serial];
        assert!(
            decode_record(&payload).is_err(),
            "serial type {serial} must be refused",
        );
    }
}

#[test]
fn a_record_whose_header_or_body_does_not_fit_is_refused() {
    // A header claiming more bytes than the payload holds.
    assert!(decode_record(&[99u8, 1, 0]).is_err());
    // A body that ends mid-value: two 8-byte columns, one byte of body.
    assert!(decode_record(&[3u8, 6, 6, 0]).is_err());
    assert!(decode_record(&[]).is_err());
}

#[test]
fn a_header_varint_that_overruns_the_declared_header_length_is_refused() {
    // header_len = 2, but the one serial-type varint starting at offset 1 is
    // two bytes wide (0x80 continues), so it would otherwise finish at offset
    // 3 — one byte past where the header itself says it ends, with that byte
    // silently reused as the first byte of the body.
    assert!(decode_record(&[2u8, 0x80, 0x01]).is_err());
}

// ---- the inline/overflow boundary ------------------------------------------

#[test]
fn a_payload_of_exactly_the_inline_maximum_stays_on_its_page() {
    let (image, blob) = blob_image(MAX_INLINE);
    assert_eq!(
        image.len(),
        2 * PAGE,
        "at exactly usable-35 bytes no overflow page may be written",
    );
    let rows = scan(&image).expect("a readable image");
    assert_eq!(rows[0].values, vec![CellValue::Blob(blob)]);
}

#[test]
fn one_byte_past_the_inline_maximum_spills_and_still_reads_back_whole() {
    let (image, blob) = blob_image(MAX_INLINE + 1);
    assert!(
        image.len() > 2 * PAGE,
        "an overflow page must have appeared"
    );
    let rows = scan(&image).expect("a readable image");
    assert_eq!(rows[0].values, vec![CellValue::Blob(blob)]);
}

#[test]
fn a_payload_spanning_several_overflow_pages_is_reassembled_in_order() {
    let (image, blob) = blob_image(PAGE * 5);
    let rows = scan(&image).expect("a readable image");
    match &rows[0].values[0] {
        CellValue::Blob(read) => {
            assert_eq!(read.len(), blob.len());
            assert_eq!(*read, blob, "the chain was reassembled out of order");
        }
        other => panic!("expected a blob, got {other:?}"),
    }
}

#[test]
fn an_overflow_chain_that_points_at_itself_is_an_error_rather_than_a_hang() {
    let (mut image, _) = blob_image(PAGE * 5);
    // Page 1 is sqlite_master, page 2 the table's leaf, page 3 the first
    // overflow page — which is made to point at itself.
    patch_page_head(&mut image, 3, 3);
    let error = scan(&image).expect_err("a cycle must be refused");
    assert!(error.message().contains("cycle"), "{}", error.message());
}

#[test]
fn an_overflow_chain_that_ends_early_is_an_error() {
    let (mut image, _) = blob_image(PAGE * 5);
    patch_page_head(&mut image, 3, 0);
    let error = scan(&image).expect_err("a short chain must be refused");
    assert!(error.message().contains("early"), "{}", error.message());
}

#[test]
fn an_overflow_pointer_past_the_end_of_the_file_is_an_error() {
    let (mut image, _) = blob_image(PAGE * 5);
    patch_page_head(&mut image, 3, 9999);
    let error = scan(&image).expect_err("an out-of-range page must be refused");
    assert!(error.message().contains("outside"), "{}", error.message());
}

#[test]
fn a_file_truncated_anywhere_is_reported_rather_than_read_past() {
    let (image, _) = blob_image(PAGE * 5);
    for length in [0, 8, 99, 100, PAGE - 1, PAGE, PAGE + 1, image.len() - 1] {
        let short = &image[..length];
        // Either the header or the walk refuses it; neither may panic, and
        // none may return the whole row.
        match SqliteDb::open(short) {
            Ok(db) => {
                let read = db
                    .master_entries()
                    .and_then(|master| db.scan_table(master.first().map_or(0, |e| e.rootpage)));
                assert!(read.is_err(), "{length} bytes must not read as a full row");
            }
            Err(error) => assert!(!error.message().is_empty()),
        }
    }
}

// ---- pages -----------------------------------------------------------------

#[test]
fn a_cell_pointer_outside_its_page_is_refused() {
    let (mut image, _) = blob_image(64);
    // The leaf page's first cell pointer sits right after its 8-byte header.
    let at = PAGE + 8;
    image[at..at + 2].copy_from_slice(&(PAGE as u16 + 200).to_be_bytes());
    let error = scan(&image).expect_err("an out-of-page cell must be refused");
    assert!(error.message().contains("outside"), "{}", error.message());
}

#[test]
fn a_btree_that_points_back_at_an_ancestor_is_an_error_rather_than_a_hang() {
    // An interior page whose rightmost child is the page itself.
    let mut image = Image::new(PAGE);
    let page1 = image.interior_page(100, 1, &[]);
    image.set_page1(page1);
    let bytes = image.finish();
    let db = SqliteDb::open(&bytes).expect("a readable header");
    let error = db.scan_table(1).expect_err("a cycle must be refused");
    assert!(error.message().contains("cycle"), "{}", error.message());
}

#[test]
fn a_child_pointer_past_the_end_of_the_file_is_an_error() {
    let mut image = Image::new(PAGE);
    let page1 = image.interior_page(100, 4242, &[]);
    image.set_page1(page1);
    let bytes = image.finish();
    let db = SqliteDb::open(&bytes).expect("a readable header");
    let error = db.scan_table(1).expect_err("an out-of-range child");
    assert!(error.message().contains("outside"), "{}", error.message());
}

// ---- the database header ---------------------------------------------------

#[test]
fn a_file_that_is_not_sqlite_is_refused_by_its_magic() {
    let error = SqliteDb::open(&[0u8; 4096]).expect_err("not a database");
    assert!(error.message().contains("magic"), "{}", error.message());
    assert!(SqliteDb::open(b"GeoPackage").is_err(), "too short");
    // …and so is the whole reader, with a message rather than a panic.
    let error = super::from_bytes(&[0u8; 4096]).expect_err("not a database");
    assert!(!error.message().is_empty());
}

#[test]
fn a_utf16_database_is_refused_because_its_text_cells_are_not_utf8() {
    let mut image = BASIC.to_vec();
    // Offset 56: text encoding. 2 is UTF-16le, 3 UTF-16be.
    for encoding in [2u32, 3] {
        image[56..60].copy_from_slice(&encoding.to_be_bytes());
        let error = super::from_bytes(&image).expect_err("UTF-16 must be refused");
        assert!(error.message().contains("UTF-16"), "{}", error.message());
    }
}

#[test]
fn an_illegal_page_size_is_refused_before_any_arithmetic_uses_it() {
    let mut image = BASIC.to_vec();
    // Not a power of two, and below the 512-byte minimum.
    for raw in [0u16, 3, 100, 4095] {
        image[16..18].copy_from_slice(&raw.to_be_bytes());
        let error = SqliteDb::open(&image).expect_err("an illegal page size");
        assert!(error.message().contains("page size"), "{}", error.message());
    }
}

#[test]
fn reserving_most_of_every_page_is_refused() {
    let mut image = BASIC.to_vec();
    // Offset 20: reserved bytes per page. 255 leaves 3841 of 4096 usable, which
    // is legal; what is refused is leaving less than SQLite's own floor.
    image[20] = 255;
    assert!(SqliteDb::open(&image).is_ok());
    image[16..18].copy_from_slice(&512u16.to_be_bytes());
    let error = SqliteDb::open(&image).expect_err("512 - 255 is too little");
    assert!(error.message().contains("usable"), "{}", error.message());
}

#[test]
fn a_geopackage_with_no_feature_table_loads_empty_and_says_nothing() {
    // Tiles and attributes tables are skipped in silence, so a file holding
    // only those is a *successful* read of zero layers — the caller is what
    // decides whether that is worth reporting.
    let dataset =
        super::from_bytes(&super::fixture::attributes_only_image()).expect("a readable GeoPackage");
    assert!(dataset.tables().is_empty());
    assert!(dataset.notices().is_empty(), "{:?}", dataset.notices());
}

// ---- write-ahead-log mode ---------------------------------------------------

#[test]
fn a_header_naming_write_ahead_logging_is_detected() {
    let blob = gp_blob(0x01, 4326, &[], &wkb_point(139.0, 35.0));
    let image = geopackage_image(
        Cell::Int(4326),
        "CREATE TABLE t (geom BLOB)",
        &[Cell::Blob(&blob)],
    );
    assert!(
        !SqliteDb::open(&image).expect("opens").wal_mode(),
        "the control is the legacy rollback journal"
    );
    let mut wal = image;
    wal[18] = 2;
    wal[19] = 2;
    assert!(SqliteDb::open(&wal).expect("opens").wal_mode());
}

#[test]
fn a_wal_geopackage_with_a_feature_table_gets_a_notice_ahead_of_its_tables() {
    let blob = gp_blob(0x01, 4326, &[], &wkb_point(139.0, 35.0));
    let mut image = geopackage_image(
        Cell::Int(4326),
        "CREATE TABLE t (geom BLOB)",
        &[Cell::Blob(&blob)],
    );
    image[18] = 2;
    image[19] = 2;
    let dataset = super::from_bytes(&image).expect("a readable GeoPackage");
    // The table itself still loads: a `-wal` sidecar this reader does not
    // apply means its rows might be stale, not that the file cannot be read.
    assert_eq!(dataset.tables().len(), 1);
    assert_eq!(dataset.refusals().len(), 1);
    let message = dataset.refusals()[0].message();
    assert!(message.contains("write-ahead-log"), "{message}");
    assert_eq!(
        dataset.refusals()[0].table(),
        "",
        "an archive-level notice, not any one table's"
    );
}

#[test]
fn a_wal_geopackage_with_no_feature_table_keeps_its_exact_empty_message() {
    // The WAL caveat is about missing feature *rows*; a file with no feature
    // *tables* at all has nothing for it to apply to, so the message stays
    // exactly what `a_geopackage_with_no_feature_table_loads_empty_and_says_
    // nothing` already asserts, undiluted by an unrelated caveat.
    let mut image = super::fixture::attributes_only_image();
    image[18] = 2;
    image[19] = 2;
    let dataset = super::from_bytes(&image).expect("a readable GeoPackage");
    assert!(dataset.tables().is_empty());
    assert!(dataset.notices().is_empty(), "{:?}", dataset.notices());
}

#[test]
fn a_sqlite_database_that_is_not_a_geopackage_is_refused_by_name() {
    let (image, _) = blob_image(64);
    let error = super::from_bytes(&image).expect_err("no gpkg_contents");
    assert!(
        error.message().contains("gpkg_contents"),
        "{}",
        error.message(),
    );
}

// ---- the GeoPackage geometry header ----------------------------------------

#[test]
fn a_blob_that_is_not_a_geopackage_geometry_is_refused() {
    for blob in [
        b"XX\0\x01".to_vec(),
        b"GP".to_vec(),
        Vec::new(),
        gp_blob(0x01, 4326, &[], &[]),
    ] {
        assert!(decode(&blob, &GpkgCrs::wgs84()).is_err(), "{blob:?}");
    }
}

#[test]
fn a_newer_header_version_is_refused_rather_than_guessed_at() {
    let mut blob = gp_blob(0x01, 4326, &[], &wkb_point(1.0, 2.0));
    blob[2] = 1;
    let error = decode(&blob, &GpkgCrs::wgs84()).expect_err("version 1 is not version 0");
    assert!(error.message().contains("version"), "{}", error.message());
}

#[test]
fn a_vendor_extended_geometry_is_refused_by_its_flag() {
    let blob = gp_blob(0x21, 4326, &[], &wkb_point(1.0, 2.0));
    let error = decode(&blob, &GpkgCrs::wgs84()).expect_err("the extension flag");
    assert!(error.message().contains("extended"), "{}", error.message());
}

#[test]
fn a_reserved_envelope_indicator_is_refused() {
    for indicator in [5u8, 6, 7] {
        let blob = gp_blob(
            0x01 | (indicator << 1),
            4326,
            &[0u8; 64],
            &wkb_point(1.0, 2.0),
        );
        let error = decode(&blob, &GpkgCrs::wgs84()).expect_err("a reserved indicator");
        assert!(error.message().contains("reserved"), "{}", error.message());
    }
}

#[test]
fn every_envelope_size_is_skipped_by_exactly_its_own_length() {
    for (indicator, bytes) in [(0u8, 0usize), (1, 32), (2, 48), (3, 48), (4, 64)] {
        let blob = gp_blob(
            0x01 | (indicator << 1),
            4326,
            &vec![0x7Fu8; bytes],
            &wkb_point(139.0, 35.0),
        );
        match decode(&blob, &GpkgCrs::wgs84()).expect("a decodable geometry") {
            Some(Geometry::Point(point)) => assert_eq!(point.coordinates, vec![139.0, 35.0]),
            other => panic!("indicator {indicator} gave {other:?}"),
        }
    }
}

#[test]
fn the_empty_geometry_flag_yields_a_feature_with_no_geometry() {
    let blob = gp_blob(0x11, 4326, &[], &[]);
    assert!(
        decode(&blob, &GpkgCrs::wgs84())
            .expect("a legal empty geometry")
            .is_none()
    );
}

#[test]
fn a_header_whose_envelope_runs_off_the_end_is_refused() {
    let blob = gp_blob(0x03, 4326, &[0u8; 8], &[]);
    assert!(decode(&blob, &GpkgCrs::wgs84()).is_err());
}

// ---- WKB -------------------------------------------------------------------

#[test]
fn big_endian_wkb_decodes_to_the_same_point_as_little_endian() {
    let little = gp_blob(0x01, 4326, &[], &wkb_point(139.7, 35.7));
    let big = gp_blob(0x01, 4326, &[], &wkb_point_be(139.7, 35.7));
    assert_eq!(
        decode(&little, &GpkgCrs::wgs84()).expect("little endian"),
        decode(&big, &GpkgCrs::wgs84()).expect("big endian"),
    );
}

#[test]
fn an_unknown_wkb_type_code_is_refused() {
    for code in [0u32, 8, 99, 4001] {
        let mut wkb = vec![1u8];
        wkb.extend_from_slice(&code.to_le_bytes());
        wkb.extend_from_slice(&0f64.to_le_bytes());
        wkb.extend_from_slice(&0f64.to_le_bytes());
        let blob = gp_blob(0x01, 4326, &[], &wkb);
        let error = decode(&blob, &GpkgCrs::wgs84()).expect_err("type {code}");
        assert!(error.message().contains("type code"), "{}", error.message());
    }
}

#[test]
fn a_wkb_byte_order_byte_that_is_neither_0_nor_1_is_refused() {
    let mut wkb = wkb_point(1.0, 2.0);
    wkb[0] = 7;
    let blob = gp_blob(0x01, 4326, &[], &wkb);
    assert!(decode(&blob, &GpkgCrs::wgs84()).is_err());
}

#[test]
fn z_and_m_ordinates_are_read_and_dropped() {
    // PointZM (3001) carries four doubles; only the first two survive.
    for (code, extra) in [(1001u32, 1usize), (2001, 1), (3001, 2)] {
        let mut wkb = vec![1u8];
        wkb.extend_from_slice(&code.to_le_bytes());
        wkb.extend_from_slice(&139.5f64.to_le_bytes());
        wkb.extend_from_slice(&35.5f64.to_le_bytes());
        for index in 0..extra {
            wkb.extend_from_slice(&(index as f64).to_le_bytes());
        }
        let blob = gp_blob(0x01, 4326, &[], &wkb);
        match decode(&blob, &GpkgCrs::wgs84()).expect("a decodable geometry") {
            Some(Geometry::Point(point)) => {
                assert_eq!(point.coordinates, vec![139.5, 35.5], "code {code}");
            }
            other => panic!("code {code} gave {other:?}"),
        }
    }
}

#[test]
fn a_point_of_nans_is_the_empty_point_and_yields_no_geometry() {
    let blob = gp_blob(0x01, 4326, &[], &wkb_point(f64::NAN, f64::NAN));
    assert!(
        decode(&blob, &GpkgCrs::wgs84())
            .expect("legal WKB")
            .is_none()
    );
}

#[test]
fn a_count_larger_than_the_bytes_behind_it_is_refused_before_allocating() {
    // A LineString claiming four billion vertices in twenty bytes.
    let mut wkb = vec![1u8];
    wkb.extend_from_slice(&2u32.to_le_bytes());
    wkb.extend_from_slice(&u32::MAX.to_le_bytes());
    wkb.extend_from_slice(&0f64.to_le_bytes());
    let blob = gp_blob(0x01, 4326, &[], &wkb);
    let error = decode(&blob, &GpkgCrs::wgs84()).expect_err("an impossible count");
    assert!(error.message().contains("claims"), "{}", error.message());
}

/// A little-endian WKB polygon from explicit rings, each a list of positions.
///
/// Deliberately does no validation of its own: a ring of zero or three
/// positions is exactly the input these tests exist to feed the reader.
fn wkb_polygon(rings: &[&[(f64, f64)]]) -> Vec<u8> {
    let mut wkb = vec![1u8];
    wkb.extend_from_slice(&3u32.to_le_bytes());
    wkb.extend_from_slice(&(rings.len() as u32).to_le_bytes());
    for ring in rings {
        wkb.extend_from_slice(&(ring.len() as u32).to_le_bytes());
        for (x, y) in *ring {
            wkb.extend_from_slice(&x.to_le_bytes());
            wkb.extend_from_slice(&y.to_le_bytes());
        }
    }
    wkb
}

/// A closed unit square offset to `(x, y)` — a ring that is usable by
/// construction, so a test that loses it lost it to the reader.
fn square(x: f64, y: f64) -> Vec<(f64, f64)> {
    vec![
        (x, y),
        (x + 1.0, y),
        (x + 1.0, y + 1.0),
        (x, y + 1.0),
        (x, y),
    ]
}

#[test]
fn a_polygon_whose_exterior_ring_is_unusable_is_dropped_rather_than_reindexed() {
    // Ring roles in WKB are positional: ring 0 is the exterior, the rest are
    // holes. Dropping a too-short ring 0 while keeping a later one slides the
    // hole into the exterior slot, and nothing downstream catches it —
    // `Polygon::new` only checks that the ring list is non-empty, and
    // `local_vector::convert_polygon` takes `coordinates[0]` as the exterior
    // and forces its winding. The feature would render solid over its hole.
    let hole = vec![(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)];
    let exterior = square(0.0, 0.0);
    let unusable: Vec<Vec<(f64, f64)>> = vec![
        // A ring the blob encoded with no positions at all.
        Vec::new(),
        // Three positions, which cannot close.
        exterior[..3].to_vec(),
        // Five positions, two of which the non-finite filter removes.
        vec![
            (0.0, 0.0),
            (f64::NAN, 0.0),
            (1.0, 1.0),
            (f64::NAN, 1.0),
            (0.0, 0.0),
        ],
    ];
    for (index, ring) in unusable.iter().enumerate() {
        let wkb = wkb_polygon(&[ring.as_slice(), hole.as_slice()]);
        let blob = gp_blob(0x01, 4326, &[], &wkb);
        assert!(
            decode(&blob, &GpkgCrs::wgs84())
                .expect("legal WKB")
                .is_none(),
            "exterior {index} was unusable, so the polygon is",
        );
    }

    // The control: a usable exterior keeps its place even when a *hole* is the
    // ring that goes, which is the case the length filter was written for.
    let wkb = wkb_polygon(&[exterior.as_slice(), &hole[..2]]);
    match decode(&gp_blob(0x01, 4326, &[], &wkb), &GpkgCrs::wgs84()).expect("legal WKB") {
        Some(Geometry::Polygon(polygon)) => {
            assert_eq!(polygon.coordinates.len(), 1, "the short hole is dropped");
            assert_eq!(polygon.coordinates[0][0], vec![0.0, 0.0]);
        }
        other => panic!("expected a Polygon, got {other:?}"),
    }
}

#[test]
fn a_multi_polygon_drops_only_the_member_whose_exterior_is_gone() {
    let hole = vec![(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)];
    let good = square(10.0, 10.0);
    let mut wkb = vec![1u8];
    wkb.extend_from_slice(&6u32.to_le_bytes());
    wkb.extend_from_slice(&2u32.to_le_bytes());
    wkb.extend_from_slice(&wkb_polygon(&[&[], hole.as_slice()]));
    wkb.extend_from_slice(&wkb_polygon(&[good.as_slice()]));
    match decode(&gp_blob(0x01, 4326, &[], &wkb), &GpkgCrs::wgs84()).expect("legal WKB") {
        Some(Geometry::MultiPolygon(multi)) => {
            assert_eq!(multi.coordinates.len(), 1, "one member had no exterior");
            assert_eq!(
                multi.coordinates[0][0][0],
                vec![10.0, 10.0],
                "the surviving member must be the good polygon, not a promoted hole",
            );
        }
        other => panic!("expected a MultiPolygon, got {other:?}"),
    }
}

#[test]
fn a_four_billion_element_count_is_refused_before_anything_is_reserved() {
    // `Vec::with_capacity`'s failure path is `handle_alloc_error`, which aborts
    // the process — it is not a `Result` this reader's "every failure is a
    // message" architecture can intercept, so an impossible count has to be
    // refused by the byte-count check rather than survived by the allocator.
    let started = std::time::Instant::now();
    for code in [3u32, 4, 5, 6, 7] {
        let mut wkb = vec![1u8];
        wkb.extend_from_slice(&code.to_le_bytes());
        wkb.extend_from_slice(&u32::MAX.to_le_bytes());
        wkb.extend_from_slice(&[0u8; 16]);
        let blob = gp_blob(0x01, 4326, &[], &wkb);
        let error = decode(&blob, &GpkgCrs::wgs84()).expect_err("an impossible count");
        assert!(error.message().contains("claims"), "{}", error.message());
    }
    // A polygon whose ring count is honest but whose *ring* claims four
    // billion positions is the same refusal one level down.
    let mut wkb = vec![1u8];
    wkb.extend_from_slice(&3u32.to_le_bytes());
    wkb.extend_from_slice(&1u32.to_le_bytes());
    wkb.extend_from_slice(&u32::MAX.to_le_bytes());
    wkb.extend_from_slice(&[0u8; 16]);
    let blob = gp_blob(0x01, 4326, &[], &wkb);
    assert!(decode(&blob, &GpkgCrs::wgs84()).is_err());
    // Generous by three orders of magnitude: the point is that nothing here
    // walked four billion of anything.
    assert!(
        started.elapsed() < std::time::Duration::from_secs(2),
        "refusing an impossible count must not take real time",
    );
}

#[test]
fn a_count_that_clears_the_byte_check_still_does_not_reserve_by_element_size() {
    // The amplification case the four-billion test above does *not* cover: a
    // count that legitimately passes `counted`, because every element it counts
    // really can be encoded in the bytes behind it, yet whose per-element
    // *slot* is several times larger than its wire form. A ring encoded as a
    // position count of zero is four legal bytes against a `Vec<Position>`
    // slot; a multi-geometry member is five against a whole `Geometry`.
    let rings = 65_536usize;
    let mut wkb = vec![1u8];
    wkb.extend_from_slice(&3u32.to_le_bytes());
    wkb.extend_from_slice(&(rings as u32).to_le_bytes());
    wkb.extend(std::iter::repeat_n(0u8, rings * 4));
    let blob = gp_blob(0x01, 4326, &[], &wkb);
    // Ring 0 has no positions, so there is no exterior and the polygon is
    // nothing — the point is that it got there without reserving megabytes.
    assert!(
        decode(&blob, &GpkgCrs::wgs84())
            .expect("every ring is legal, minimal WKB")
            .is_none()
    );

    // The same shape one level up: a GeometryCollection of five-byte members.
    let members = 65_536usize;
    let mut wkb = vec![1u8];
    wkb.extend_from_slice(&7u32.to_le_bytes());
    wkb.extend_from_slice(&(members as u32).to_le_bytes());
    wkb.extend(std::iter::repeat_n(0u8, members * 5));
    let blob = gp_blob(0x01, 4326, &[], &wkb);
    // Byte order 0 with type code 0 is not an OGC type, so this is refused —
    // but only after the count was accepted and the reserve made.
    assert!(decode(&blob, &GpkgCrs::wgs84()).is_err());
}

#[test]
fn a_multi_geometry_holding_the_wrong_member_type_is_refused() {
    // A MultiPolygon (6) whose only member is a LineString.
    let mut wkb = vec![1u8];
    wkb.extend_from_slice(&6u32.to_le_bytes());
    wkb.extend_from_slice(&1u32.to_le_bytes());
    wkb.extend_from_slice(&wkb_line(&[(0.0, 0.0), (1.0, 1.0)]));
    let blob = gp_blob(0x01, 4326, &[], &wkb);
    assert!(decode(&blob, &GpkgCrs::wgs84()).is_err());
}

#[test]
fn a_multi_geometry_restates_the_byte_order_of_every_member() {
    // A little-endian MultiPoint whose second member is big-endian — legal, and
    // the reason the byte order is per geometry rather than per blob.
    let mut wkb = vec![1u8];
    wkb.extend_from_slice(&4u32.to_le_bytes());
    wkb.extend_from_slice(&2u32.to_le_bytes());
    wkb.extend_from_slice(&wkb_point(1.0, 2.0));
    wkb.extend_from_slice(&wkb_point_be(3.0, 4.0));
    let blob = gp_blob(0x01, 4326, &[], &wkb);
    match decode(&blob, &GpkgCrs::wgs84()).expect("a decodable geometry") {
        Some(Geometry::MultiPoint(points)) => {
            assert_eq!(points.coordinates, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        }
        other => panic!("expected a MultiPoint, got {other:?}"),
    }
}

#[test]
fn truncating_a_geometry_anywhere_is_reported_rather_than_read_past() {
    let mut wkb = vec![1u8];
    wkb.extend_from_slice(&3u32.to_le_bytes());
    wkb.extend_from_slice(&1u32.to_le_bytes());
    wkb.extend_from_slice(&4u32.to_le_bytes());
    for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)] {
        wkb.extend_from_slice(&f64::to_le_bytes(x));
        wkb.extend_from_slice(&f64::to_le_bytes(y));
    }
    let blob = gp_blob(0x01, 4326, &[], &wkb);
    assert!(decode(&blob, &GpkgCrs::wgs84()).expect("whole").is_some());
    for length in 1..blob.len() {
        // Every prefix must be an error or an empty geometry — never a panic.
        let _ = decode(&blob[..length], &GpkgCrs::wgs84());
    }
}

#[test]
fn a_web_mercator_geometry_is_inverse_projected_vertex_by_vertex() {
    let blob = gp_blob(0x01, 3857, &[], &wkb_point(15_550_408.8, 4_257_159.1));
    match decode(
        &blob,
        &GpkgCrs::for_crs(&oxigis_core::Crs::web_mercator()).expect("3857"),
    )
    .expect("a decodable geometry")
    {
        Some(Geometry::Point(point)) => {
            assert!((point.coordinates[0] - 139.69).abs() < 0.01);
            assert!((point.coordinates[1] - 35.68).abs() < 0.01);
        }
        other => panic!("expected a Point, got {other:?}"),
    }
}

// ---- the CREATE TABLE parser -----------------------------------------------

/// The column names a `CREATE TABLE` statement declares.
fn columns(sql: &str) -> Vec<String> {
    parse_create_table(sql)
        .expect("a parsable statement")
        .columns
        .iter()
        .map(|column| column.name.clone())
        .collect()
}

#[test]
fn quoted_identifiers_keep_their_spaces_and_their_case() {
    assert_eq!(
        columns(r#"CREATE TABLE t ("name ja" TEXT, `back tick` INT, [bracket one] REAL, plain)"#),
        vec!["name ja", "back tick", "bracket one", "plain"],
    );
    // A doubled quote inside a quoted identifier is one quote.
    assert_eq!(
        columns(r#"CREATE TABLE t ("say ""hi""" TEXT)"#),
        vec![r#"say "hi""#],
    );
    // A quoted identifier that happens to be a keyword is still a column.
    assert_eq!(
        columns(r#"CREATE TABLE t ("primary" TEXT)"#),
        vec!["primary"]
    );
}

#[test]
fn table_constraints_are_not_columns() {
    assert_eq!(
        columns(
            "CREATE TABLE t (a INT, b TEXT, CONSTRAINT pk PRIMARY KEY (a, b), \
             UNIQUE (b), CHECK (a > 0), FOREIGN KEY (b) REFERENCES u (x))",
        ),
        vec!["a", "b"],
    );
}

#[test]
fn parenthesised_commas_do_not_split_a_definition() {
    assert_eq!(
        columns(
            "CREATE TABLE t (a VARCHAR(10) NOT NULL, b NUMERIC(10, 2), \
             c TEXT DEFAULT (coalesce(1, 2)), d INT)",
        ),
        vec!["a", "b", "c", "d"],
    );
}

#[test]
fn a_newline_wrapped_definition_parses_like_a_one_liner() {
    assert_eq!(
        columns("CREATE TABLE \"t\"\n(\n  fid INTEGER PRIMARY KEY,\n  geom BLOB\n)\n"),
        vec!["fid", "geom"],
    );
}

#[test]
fn unquoted_non_ascii_identifiers_are_columns_like_any_other() {
    // SQLite's tokenizer is byte-oriented and its identifier-character class
    // includes *every* byte >= 0x80, so these need no quotes. Verified against
    // SQLite 3.37.2: `CREATE TABLE t (名前 TEXT, geom BLOB)` is accepted, stored
    // verbatim and unquoted in sqlite_master, and reported by
    // `PRAGMA table_info` as two columns in that order.
    //
    // Reading those bytes as punctuation instead drops the column from the
    // schema entirely — and the parser still returns `Some`, so the caller
    // lines every later record value up against the wrong column.
    assert_eq!(
        columns("CREATE TABLE 地点 (名前 TEXT, geom BLOB)"),
        vec!["名前", "geom"],
    );
    assert_eq!(
        columns("CREATE TABLE t (Привет TEXT, naïve INT, Ω REAL, a名b INT, plain)"),
        vec!["Привет", "naïve", "Ω", "a名b", "plain"],
        "scripts mix freely, inside a word as well as beside one",
    );
    // The type after the name still parses, so affinity still works…
    let schema = parse_create_table("CREATE TABLE t (標高 REAL, 人口 INTEGER)").expect("parsable");
    assert!(schema.columns[0].has_real_affinity());
    assert!(!schema.columns[1].has_real_affinity());
    // …and the column is findable by name, which is how the geometry column is
    // located.
    assert_eq!(schema.column_index("標高"), Some(0));
    // A non-ASCII INTEGER PRIMARY KEY is the rowid alias like any other.
    assert_eq!(
        parse_create_table("CREATE TABLE t (番号 INTEGER PRIMARY KEY, g BLOB)")
            .expect("parsable")
            .rowid_alias,
        Some(0),
    );
}

#[test]
fn the_rowid_alias_is_found_however_it_is_declared() {
    let alias = |sql: &str| parse_create_table(sql).expect("parsable").rowid_alias;
    assert_eq!(
        alias("CREATE TABLE t (fid INTEGER PRIMARY KEY, g BLOB)"),
        Some(0)
    );
    assert_eq!(
        alias("CREATE TABLE t (g BLOB, fid INTEGER PRIMARY KEY AUTOINCREMENT)"),
        Some(1),
    );
    assert_eq!(
        alias("CREATE TABLE t (fid INTEGER NOT NULL PRIMARY KEY, g BLOB)"),
        Some(0),
    );
    assert_eq!(
        alias("CREATE TABLE t (fid INTEGER PRIMARY KEY ASC, g BLOB)"),
        Some(0),
        "ASC is the default order, so the column-level form is still an alias",
    );
    assert_eq!(
        alias("CREATE TABLE t (fid INTEGER, g BLOB, PRIMARY KEY (fid))"),
        Some(0),
        "a table-level key over one INTEGER column is an alias too",
    );
    // The *table-level* clause keeps the alias whatever decorates the column
    // inside it — including DESC, which breaks it at column level. Verified
    // against SQLite 3.37.2 for each spelling below with `UPDATE t SET rowid =
    // 999` (the declared column moves with the rowid) and by dumping the record
    // bytes (the column's slot holds serial type 0, i.e. NULL, so its value
    // exists only as the cell's rowid).
    for clause in [
        "PRIMARY KEY (fid ASC)",
        "PRIMARY KEY (fid DESC)",
        "PRIMARY KEY (\"fid\")",
        "PRIMARY KEY (\"fid\" DESC)",
        "PRIMARY KEY ([fid] ASC)",
        "PRIMARY KEY (`fid`)",
        "PRIMARY KEY (fid COLLATE NOCASE)",
        "PRIMARY KEY (fid) ON CONFLICT REPLACE",
    ] {
        assert_eq!(
            alias(&format!("CREATE TABLE t (fid INTEGER, g BLOB, {clause})")),
            Some(0),
            "{clause} is a rowid alias in real SQLite",
        );
    }
    // …and the cases that are *not* aliases, where the value really is stored.
    assert_eq!(alias("CREATE TABLE t (fid INT PRIMARY KEY, g BLOB)"), None);
    assert_eq!(alias("CREATE TABLE t (fid TEXT PRIMARY KEY, g BLOB)"), None);
    assert_eq!(
        alias("CREATE TABLE t (fid INTEGER PRIMARY KEY DESC, g BLOB)"),
        None,
        "the DESC exception is the column-level clause's alone",
    );
    assert_eq!(
        alias("CREATE TABLE t (a INTEGER, b INTEGER, PRIMARY KEY (a, b))"),
        None,
    );
    assert_eq!(
        alias("CREATE TABLE t (a INTEGER, b INTEGER, PRIMARY KEY (a ASC, b DESC))"),
        None,
        "a composite key stays composite however its columns are sorted",
    );
    assert_eq!(
        alias("CREATE TABLE t (fid INTEGER, g BLOB, PRIMARY KEY (abs(fid)))"),
        None,
        "an expression is not a column this reader may guess at",
    );
    assert_eq!(
        alias("CREATE TABLE t (pk TEXT PRIMARY KEY, g BLOB) WITHOUT ROWID"),
        None,
    );
    assert_eq!(
        alias("CREATE TABLE t (fid INTEGER, g BLOB, PRIMARY KEY (fid DESC)) WITHOUT ROWID"),
        None,
        "a WITHOUT ROWID table has no rowid to alias",
    );
}

// ---- metadata this reader must not guess at --------------------------------

#[test]
fn a_record_holding_more_values_than_its_table_declares_columns_refuses_that_table() {
    // SQLite writes records *shorter* than the column list (a row predating an
    // `ALTER TABLE ADD COLUMN`) and never longer, so this can only mean the
    // CREATE TABLE statement was parsed into too few columns. Resizing down
    // would truncate the record's tail rather than restore the missing slot,
    // sliding every value onto the wrong column and discarding the last one —
    // in a feature table that is the geometry BLOB, so the whole layer would
    // come back with no geometry and no error.
    let blob = gp_blob(0x01, 4326, &[], &wkb_point(139.0, 35.0));
    let image = geopackage_image(
        Cell::Int(4326),
        "CREATE TABLE t (geom BLOB)",
        &[Cell::Blob(&blob), Cell::Text("a value with no column")],
    );
    let dataset = super::from_bytes(&image).expect("a readable GeoPackage");
    assert!(dataset.tables().is_empty());
    assert_eq!(dataset.refusals().len(), 1);
    assert_eq!(dataset.refusals()[0].table(), "t");
    let message = dataset.refusals()[0].message();
    assert!(message.contains("2 values"), "{message}");
    assert!(
        message.contains("more than its table's definition declares"),
        "{message}",
    );

    // The control: the same table with its record and its statement agreeing.
    let image = geopackage_image(
        Cell::Int(4326),
        "CREATE TABLE t (geom BLOB, note TEXT)",
        &[Cell::Blob(&blob), Cell::Text("a value with a column")],
    );
    let dataset = super::from_bytes(&image).expect("a readable GeoPackage");
    assert_eq!(dataset.tables().len(), 1);
    assert!(dataset.refusals().is_empty());
}

// ---- non-geometry cell values -----------------------------------------------

#[test]
fn a_blob_in_a_non_geometry_column_becomes_a_byte_count_not_null() {
    // `null` is indistinguishable from a genuine NULL; the byte count is not,
    // and is the only truthful thing to show without dumping the bytes
    // themselves into an attribute-table cell.
    let geom = gp_blob(0x01, 4326, &[], &wkb_point(139.0, 35.0));
    let photo = [0xDEu8, 0xAD, 0xBE, 0xEF, 0x00];
    let image = geopackage_image(
        Cell::Int(4326),
        "CREATE TABLE t (geom BLOB, photo BLOB, caption TEXT)",
        &[Cell::Blob(&geom), Cell::Blob(&photo), Cell::Null],
    );
    let dataset = super::from_bytes(&image).expect("a readable GeoPackage");
    assert_eq!(dataset.tables().len(), 1);
    let properties = dataset.tables()[0].features.features[0]
        .properties
        .as_ref()
        .expect("properties");
    assert_eq!(properties["photo"].as_str(), Some("<5 bytes>"));
    assert!(properties["caption"].is_null(), "an actual NULL stays null");
}

#[test]
fn a_geometry_column_registration_with_a_non_integer_srs_id_refuses_that_table() {
    // `srs_id` is `INTEGER NOT NULL` per the spec, and SQLite's affinity rules
    // coerce anything storable into an integer on the way in — so a cell of
    // another storage class means the file was not written by SQLite and the
    // table's CRS is simply unknown. Defaulting to 0, which `resolve_srs` reads
    // as "undefined" and therefore as WGS 84, would draw a projected table's
    // metres as degrees with nothing on screen to say why.
    let blob = gp_blob(0x01, 4326, &[], &wkb_point(139.0, 35.0));
    let sql = "CREATE TABLE t (fid INTEGER PRIMARY KEY, geom BLOB)";
    let row = [Cell::Null, Cell::Blob(&blob)];

    let dataset = super::from_bytes(&geopackage_image(Cell::Int(4326), sql, &row))
        .expect("a readable GeoPackage");
    assert_eq!(dataset.tables().len(), 1, "the control must load");

    for cell in [
        Cell::Real(4326.0),
        Cell::Text("4326"),
        Cell::Blob(b"4326".as_slice()),
        Cell::Null,
    ] {
        let dataset =
            super::from_bytes(&geopackage_image(cell, sql, &row)).expect("a readable GeoPackage");
        assert!(dataset.tables().is_empty(), "a guessed CRS is not a load");
        assert_eq!(dataset.refusals().len(), 1);
        assert_eq!(dataset.refusals()[0].table(), "t");
        let message = dataset.refusals()[0].message();
        assert!(message.contains("srs_id"), "{message}");
        assert!(message.contains('t'), "{message}");
    }
}

#[test]
fn a_statement_with_no_column_list_is_not_guessed_at() {
    assert!(parse_create_table("CREATE TABLE t AS SELECT 1").is_none());
    assert!(parse_create_table("").is_none());
}

#[test]
fn real_affinity_follows_sqlites_own_precedence() {
    let real = |sql: &str| {
        parse_create_table(sql)
            .expect("parsable")
            .columns
            .iter()
            .map(super::sqlite::ColumnDef::has_real_affinity)
            .collect::<Vec<bool>>()
    };
    assert_eq!(
        real("CREATE TABLE t (a REAL, b DOUBLE PRECISION, c FLOAT, d INTEGER, e TEXT, f BLOB, g)"),
        vec![true, true, true, false, false, false, false],
    );
    // Rule 1 is "contains INT", not "is an integer type": it wins over a REAL
    // spelling wherever it appears, even inside a word.
    assert_eq!(
        real("CREATE TABLE t (a POINTREAL, b REALINT, c REALLY)"),
        vec![false, false, true],
    );
}
