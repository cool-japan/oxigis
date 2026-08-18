// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Hand-built cases: bytes that are not Parquet at all, a real file
//! truncated mid-footer, a well-formed Parquet file with no GeoParquet `geo`
//! metadata, and unit-level checks of [`super::resolve_crs`] for the CRS
//! shape [`super`]'s module docs explain cannot reach it through a real file
//! today. Mirrors [`crate::gpkg_input::tests_hostile`]'s split from the
//! real-file fixture tests.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use oxigeo_geoparquet::metadata::Crs;
use parquet::arrow::ArrowWriter;
use serde_json::Value;

use super::fixture::UNCOMPRESSED;
use super::{MAX_GEOPARQUET_BYTES, cell_to_json, from_bytes, resolve_crs};

/// A plain Parquet file (one `Int64` column, no `geo` key in its schema
/// metadata) — a legal Parquet file that simply is not a GeoParquet one.
fn plain_parquet_bytes() -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
    )
    .expect("valid batch");
    let mut writer = ArrowWriter::try_new(Vec::new(), schema, None).expect("writer");
    writer.write(&batch).expect("write");
    writer.into_inner().expect("close")
}

#[test]
fn bytes_that_are_not_parquet_at_all_are_reported_rather_than_panicking() {
    let error = from_bytes(&[0u8; 16]).expect_err("16 zero bytes are not a Parquet file");
    assert!(!error.message().is_empty());

    let error = from_bytes(b"not even close to parquet").expect_err("plain text is not parquet");
    assert!(!error.message().is_empty());

    let error = from_bytes(&[]).expect_err("empty input is not parquet");
    assert!(!error.message().is_empty());
}

#[test]
fn a_file_past_the_byte_ceiling_is_refused_before_anything_is_parsed() {
    // The ceiling has to run first: proving that is the whole point of the
    // check, so this uses a zero-filled buffer rather than a real (and far
    // larger still) valid GeoParquet file — a genuine file this size would
    // need to expand well past the ceiling itself.
    let oversized = vec![0u8; MAX_GEOPARQUET_BYTES + 1];
    let started = std::time::Instant::now();
    let error = from_bytes(&oversized).expect_err("past the ceiling");
    assert!(error.message().contains("MiB"), "{}", error.message());
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "refusing an oversized file must not first try to parse it",
    );
}

#[test]
fn a_real_file_truncated_mid_footer_is_reported_rather_than_panicking() {
    // Parquet's footer (row group / schema metadata) sits at the *end* of the
    // file; chopping the last third destroys it while leaving what looks
    // like plausible leading bytes ("PAR1"), which is the sharper hostile
    // case than truncating from the front.
    let truncated = &UNCOMPRESSED[..UNCOMPRESSED.len() * 2 / 3];
    let error = from_bytes(truncated).expect_err("a truncated footer must not parse");
    assert!(!error.message().is_empty());
}

#[test]
fn a_parquet_file_with_no_geo_metadata_names_the_reason() {
    let bytes = plain_parquet_bytes();
    let error = from_bytes(&bytes).expect_err("a plain Parquet file has no geometry column");
    assert!(
        error.message().contains("geo"),
        "the refusal must say the file has no GeoParquet metadata: {}",
        error.message(),
    );
}

// ---- resolve_crs: the Crs::Wkt2 shape unreachable via a real file today ----
//
// See `super`'s module docs' "CRS" section: `Crs` is `#[serde(untagged)]`
// with `ProjJson` declared first, so a WKT2 string parsed from a file always
// becomes `ProjJson(Value::String(..))` — and that shape is itself rejected
// by `Crs::validate` (`"PROJJSON must be an object"`) before `resolve_crs`
// ever sees it. `Crs::from_wkt2` builds a genuine `Crs::Wkt2` directly,
// bypassing JSON parsing and validation entirely, to check the shared
// `oxigis_core::crs` reader is actually wired up correctly rather than merely
// present.

#[test]
fn a_wkt2_crs_built_directly_resolves_through_the_shared_reader() {
    let web_mercator = Crs::from_wkt2(
        r#"PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",AUTHORITY["EPSG","3857"]]"#,
    );
    assert_eq!(resolve_crs(Some(&web_mercator)).epsg(), 3857);

    let wgs84 = Crs::from_wkt2(r#"GEOGCS["GCS_WGS_1984",AUTHORITY["EPSG","4326"]]"#);
    assert!(resolve_crs(Some(&wgs84)).is_wgs84());

    // UTM 54N used to be refused here; it now resolves and reprojects.
    let utm = Crs::from_wkt2(r#"PROJCS["WGS_1984_UTM_Zone_54N"]"#);
    assert_eq!(resolve_crs(Some(&utm)).epsg(), 32654);

    // A projection family this build cannot invert is still refused, by name
    // and by code.
    let lambert = Crs::from_wkt2(
        r#"PROJCS["RGF93 / Lambert-93",PROJECTION["Lambert_Conformal_Conic_2SP"],AUTHORITY["EPSG","2154"]]"#,
    );
    let resolved = resolve_crs(Some(&lambert));
    assert!(!resolved.is_supported());
    let error = crate::crs_sniff::reprojector_or_refuse(&resolved, "GeoParquet")
        .expect_err("Lambert-93 must be refused");
    assert!(
        error.message().contains("Lambert-93") && error.message().contains("EPSG:2154"),
        "the refusal must name the CRS and its code: {}",
        error.message(),
    );
}

#[test]
fn no_crs_at_all_is_wgs84() {
    assert!(
        resolve_crs(None).is_wgs84(),
        "an absent CRS is the GeoParquet-spec default, OGC:CRS84",
    );
}

#[test]
fn a_projjson_object_naming_an_unplaceable_crs_keeps_its_name_for_the_refusal() {
    let projjson: serde_json::Value = serde_json::json!({
        "type": "ProjectedCRS",
        "name": "RGF93 v1 / Lambert-93",
        "id": {"authority": "EPSG", "code": 2154},
    });
    let crs = Crs::ProjJson(projjson);
    let resolved = resolve_crs(Some(&crs));
    assert_eq!(resolved.epsg(), 2154);
    assert!(!resolved.is_supported());
    let error = crate::crs_sniff::reprojector_or_refuse(&resolved, "GeoParquet")
        .expect_err("must be refused");
    assert!(error.message().contains("EPSG:2154"), "{}", error.message());

    // One with no id at all still names itself from the PROJJSON `name`.
    let unnamed = Crs::ProjJson(serde_json::json!({"name": "Site grid"}));
    let resolved = resolve_crs(Some(&unnamed));
    assert!(!resolved.is_supported());
    assert_eq!(resolved.name(), "Site grid");

    // And one with neither still produces a message rather than a panic.
    let nothing = Crs::ProjJson(serde_json::json!({}));
    let resolved = resolve_crs(Some(&nothing));
    assert!(!resolved.is_supported());
    assert!(resolved.name().contains("unnamed"), "{}", resolved.name());
}

#[test]
fn a_projjson_object_naming_a_japanese_plane_rectangular_zone_now_loads() {
    let projjson: serde_json::Value = serde_json::json!({
        "type": "ProjectedCRS",
        "name": "JGD2011 / Japan Plane Rectangular CS IX",
        "id": {"authority": "EPSG", "code": 6677},
    });
    let resolved = resolve_crs(Some(&Crs::ProjJson(projjson)));
    assert_eq!(resolved.epsg(), 6677);
    let reprojector =
        crate::crs_sniff::reprojector_or_refuse(&resolved, "GeoParquet").expect("zone IX loads");
    let (lon, lat) = reprojector
        .to_lon_lat(-5_995.185, -35_367.230)
        .expect("inverts");
    assert!(
        (139.0..140.5).contains(&lon) && (35.0..36.5).contains(&lat),
        "({lon}, {lat})"
    );
}

// ---- cell_to_json: the ArrayFormatter fallback arm and the untyped-in-the-
// ---- fixtures Int16/LargeUtf8 arms ------------------------------------------
//
// None of the ten pyarrow fixtures carries a Date/Timestamp/List/Struct column
// outside the `covering.bbox` struct (which `skip_columns` excludes before
// `cell_to_json` ever sees it) or an Int16/LargeUtf8 column — every typed
// fixture column is Utf8/Int64/Float32/Boolean. Built directly here so the
// module doc's "no column is silently dropped" claim (and the two extra typed
// arms) are actually exercised, not merely present in the match.

#[test]
fn a_non_primitive_column_falls_back_to_the_array_formatter_stringifier() {
    let format_options = arrow::util::display::FormatOptions::default();

    // Date32 has no dedicated match arm, so it must hit the `_ =>` fallback.
    let dates = arrow::array::Date32Array::from(vec![19_723]);
    match cell_to_json(&dates, 0, &format_options) {
        Value::String(text) => assert!(
            !text.is_empty(),
            "the fallback must stringify a Date32 cell, not drop it",
        ),
        other => panic!("expected a stringified date, got {other:?}"),
    }

    // Null still short-circuits ahead of the fallback, same as every other arm.
    let null_dates = arrow::array::Date32Array::from(vec![None]);
    assert_eq!(
        cell_to_json(&null_dates, 0, &format_options),
        Value::Null,
        "a null Date32 cell must stay null even though its type only reaches the fallback arm",
    );
}

#[test]
fn large_utf8_and_int16_use_their_own_typed_arms_not_the_fallback() {
    let format_options = arrow::util::display::FormatOptions::default();

    let large = arrow::array::LargeStringArray::from(vec!["long text column"]);
    assert_eq!(
        cell_to_json(&large, 0, &format_options),
        Value::String("long text column".to_string()),
    );

    let shorts = arrow::array::Int16Array::from(vec![42i16]);
    assert_eq!(cell_to_json(&shorts, 0, &format_options), Value::from(42));
}
