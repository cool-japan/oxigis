// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Test-only GeoParquet fixtures, written by pyarrow 24 (a real writer, not
//! this crate's own — so the reader is checked against an independent
//! implementation of the format).
//!
//! `fixtures/*.parquet` are the ten files; `fixtures/truth.json` is the
//! ground truth emitted alongside them, keyed by file name, so a regenerated
//! fixture cannot silently invalidate a test — every assertion in `tests`
//! reads its expectation out of the JSON rather than a number typed here.
//! `fixtures/gen_fixtures.py` is how they were made, kept as provenance next
//! to them (mirrors `crate::gpkg_input::fixture`'s layout) and not run by the
//! test suite. All of it is `include_bytes!`/`include_str!`-ed under
//! `#[cfg(test)]` only, so none of it reaches a release build even when the
//! `geoparquet` feature is on.

#![allow(clippy::unwrap_used, clippy::expect_used)]

/// Uncompressed row groups — the codec baseline every other file diffs
/// against.
pub(crate) const UNCOMPRESSED: &[u8] = include_bytes!("fixtures/uncompressed.parquet");
/// Snappy-compressed row groups.
pub(crate) const SNAPPY: &[u8] = include_bytes!("fixtures/snappy.parquet");
/// Brotli-compressed row groups.
pub(crate) const BROTLI: &[u8] = include_bytes!("fixtures/brotli.parquet");
/// LZ4-compressed row groups (pyarrow's `LZ4` codec writes Hadoop-style
/// `LZ4_RAW`, which is the arm this crate's `lz4` feature enables).
pub(crate) const LZ4: &[u8] = include_bytes!("fixtures/lz4.parquet");
/// Zstd-compressed row groups — must open (metadata) but fail to read (data);
/// zstd has no path through this crate's codec allow-list at all.
pub(crate) const ZSTD: &[u8] = include_bytes!("fixtures/zstd.parquet");
/// Gzip-compressed row groups — same "opens, data read fails" shape as
/// [`ZSTD`], but the underlying `parquet` crate names the disabled *feature*
/// (`flate2`) rather than the codec (`gzip`) in its own error text; see
/// `super::codec_error`.
pub(crate) const GZIP: &[u8] = include_bytes!("fixtures/gzip.parquet");
/// A PROJJSON `crs` naming EPSG:3857 (Web Mercator); coordinates are stored
/// in metres and must be inverse-projected on the way in.
pub(crate) const PROJJSON_3857: &[u8] = include_bytes!("fixtures/projjson_3857.parquet");
/// A PROJJSON `crs` naming EPSG:2154 (RGF93 / Lambert-93) — refused, naming
/// the CRS.
pub(crate) const UNSUPPORTED_CRS: &[u8] = include_bytes!("fixtures/unsupported_crs.parquet");
/// A GeoParquet 1.1 `covering.bbox` struct column literally named `bbox`,
/// which must not show up in the attribute table.
pub(crate) const COVERING_BBOX: &[u8] = include_bytes!("fixtures/covering_bbox.parquet");
/// One point and one donut polygon (an exterior ring plus one hole), the WKB
/// polygon-with-holes path.
pub(crate) const MIXED_GEOM: &[u8] = include_bytes!("fixtures/mixed_geom.parquet");

/// Ground truth for every fixture above, keyed by file name (see
/// `gen_fixtures.py` for exactly how each value was produced).
pub(crate) const TRUTH: &str = include_str!("fixtures/truth.json");
