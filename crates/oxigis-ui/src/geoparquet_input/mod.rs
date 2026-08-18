// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GeoParquet → GeoJSON [`FeatureCollection`], from **bytes only**.
//!
//! Phase 1 §1.3's third format, and the first that is native-only: this
//! entire module compiles only under the `geoparquet` Cargo feature, enabled
//! by `oxigis-desktop` and deliberately never by `oxigis-web` (arrow/parquet
//! are heavy — see `TODO.md`'s "GeoParquet / GPKG" entry). Recognising a
//! `.parquet`/`.geoparquet` drop is still unconditional
//! ([`crate::local_input::classify_drop`]); only *reading* one needs this
//! feature, so a browser build can say clearly why it did nothing instead of
//! misreporting the file as broken GeoJSON.
//!
//! # Why a direct, low-level read instead of `GeoParquetReader::open`
//!
//! `oxigeo-geoparquet` 0.2.2's only high-level constructor,
//! `GeoParquetReader::open`, is **path-only** (it calls `File::open` twice
//! internally) — there is no bytes/`Cursor` entry point, so it cannot be used
//! from a browser-style byte buffer at all, and this crate never touches the
//! filesystem. Worse, its convenience readers have a real bug: both
//! `GeoParquetReader::read_geometries` and the native-array decoder it calls
//! (`decode_native_array`, via `.flatten().collect()`) **silently drop
//! null-geometry rows** instead of keeping a placeholder, which desyncs
//! `geometries[i]` from row `i` the moment any geometry is null —
//! `GeoParquetBatchReader::extract_geometries` additionally never dispatches
//! on encoding at all, so it fails outright on a GeoArrow-native column.
//!
//! [`from_bytes`] is therefore built on the crate's lower-level, *bytes*
//! capable pieces instead: [`ArrowReaderMetadata::load`] over a [`Bytes`]
//! (`parquet`'s own `ChunkReader` is implemented for `Bytes`, not just
//! `File`) to read the footer, [`oxigeo_geoparquet::arrow_ext::extract_geoparquet_metadata`]
//! plus [`GeoParquetMetadata::from_json`] to find the primary geometry column
//! and its declared encoding, [`oxigeo_geoparquet::pushdown::execute_pushdown`]
//! (with no bbox/filters/limit — a plain full read, reusing the crate's own
//! pushdown machinery as a "read everything" path) to get the row batches,
//! and then a **null-preserving** decode per row: a hand-rolled loop over the
//! `BinaryArray` for the WKB encoding (`Feature::geometry` is already
//! `Option`, so no row need be dropped), or the public, null-preserving
//! [`oxigeo_geoparquet::geometry::decode_native_array_optional`] for a
//! GeoArrow-native column.
//!
//! # What is kept and what is dropped
//!
//! * **2-D only.** See the internal `geometry` submodule's docs.
//! * **Only the primary geometry column.** GeoParquet 1.1 allows several
//!   geometry columns in one file; every column the file's own `geo`
//!   metadata declares (the primary one and any others) is excluded from the
//!   attribute table entirely — showing raw WKB/GeoArrow bytes stringified as
//!   attribute noise would be worse than not showing them, the same call
//!   [`crate::shapefile_input`] makes for DBF memo columns.
//! * **The GeoParquet 1.1 `covering.bbox` columns** (a cached per-row extent
//!   for row-group pruning) are excluded from the attribute table too — a
//!   common writer layout is a single struct column literally named `bbox`,
//!   which would otherwise show up as a confusing extra property on every
//!   feature.
//! * **Null geometry rows are preserved at their index**, becoming a
//!   [`Feature`] with `geometry: None` — this is the specific behaviour the
//!   upstream bug above gets wrong, and the reason it could not be reused.
//!
//! # CRS
//!
//! Absent `crs` is the GeoParquet-spec default, OGC:CRS84 (WGS 84 lon/lat) —
//! see `metadata.rs`'s own doc comment, not an omission here. A PROJJSON
//! object is read by its `id.authority`/`id.code` (only EPSG:4326/3857 pass).
//! WKT2 text is classified the same way [`crate::shapefile_input::sniff_prj`]
//! classifies a `.prj`, via the shared `crate::crs_sniff` module. Anything
//! else is refused, naming the CRS.
//!
//! One CRS shape is handled defensively even though it cannot occur through
//! [`from_bytes`] today: [`oxigeo_geoparquet::metadata::Crs`] is
//! `#[serde(untagged)]` with `ProjJson(serde_json::Value)` declared before
//! `Wkt2(String)`, so a *bare WKT2 string* in a file's `crs` field would, if
//! it ever got past metadata validation, deserialize into
//! `ProjJson(Value::String(..))` rather than `Crs::Wkt2` — untagged enums try
//! variants in declaration order, and `serde_json::Value` accepts any JSON
//! value including a bare string. It does not get past validation today:
//! `Crs::validate` requires a `ProjJson` payload to be a JSON *object*
//! (verified empirically against `oxigeo-geoparquet` 0.2.2 — a bare-string
//! `crs` fails `GeoParquetMetadata::from_json` with `"PROJJSON must be an
//! object"` before `classify_crs` ever runs), so today this is a metadata
//! error, not a silent misclassification. `classify_crs`'s `json.as_str()`
//! branch and the (otherwise unreachable) `Crs::Wkt2` match arm exist so that
//! if a future crate version relaxes that validation, or `Crs::Wkt2` is
//! reached by a `Crs` built in code rather than parsed from a file, this
//! still classifies correctly instead of silently falling through — see the
//! `tests_hostile` unit tests, which construct a `Crs::Wkt2` directly
//! (`Crs::from_wkt2`) to exercise the shared classifier without going through
//! metadata validation at all.
//!
//! # Codecs
//!
//! Only uncompressed, Snappy, Brotli and LZ4 column chunks are compiled in
//! (`Cargo.toml`'s codec allow-list — Pure Rust, deny-clean; see the
//! workspace `Cargo.toml`'s comment on the `oxigeo-geoparquet` dependency
//! line). A file using a disabled codec (Zstd, Gzip) *opens* fine — codecs are
//! a per-column-chunk, per-row-group concern, invisible at the metadata level
//! — and fails only once [`from_bytes`] actually reads that data; the
//! internal `codec_error` helper turns the resulting multiply-wrapped error
//! into a message that names the codec instead of the raw error chain.

#[cfg(test)]
pub(crate) mod fixture;
mod geometry;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_hostile;

use std::collections::HashSet;

use bytes::Bytes;
use oxigeo::geojson::types::{Feature, FeatureCollection, Properties};
use oxigeo_geoparquet::GeoParquetError;
use oxigeo_geoparquet::arrow_ext::extract_geoparquet_metadata;
use oxigeo_geoparquet::geometry::{
    Geometry as GpGeometry, WkbReader, decode_native_array_optional,
};
use oxigeo_geoparquet::metadata::{Crs, EncodingType, GeoParquetMetadata};
use oxigeo_geoparquet::pushdown::execute_pushdown;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use serde_json::Value;

use arrow::array::{
    Array, BinaryArray, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, LargeStringArray, StringArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use arrow::util::display::{ArrayFormatter, FormatOptions};

use crate::local_vector::LocalVectorError;
use oxigis_core::Crs as SourceCrs;

/// Largest `.parquet`/`.geoparquet` **input** this reader accepts.
///
/// [`from_bytes`] holds everything resident and nothing is streamed (see the
/// module docs' closing note): an owned copy of the source bytes, the
/// decompressed Arrow row-group batches, and finally a GeoJSON
/// [`FeatureCollection`] whose every attribute becomes an owned `String` key
/// plus a `serde_json::Value` — several times the on-disk size for a typical
/// file. This bounds only the *input*; a file just under the ceiling can
/// still expand well past it in memory by the time [`from_bytes`] returns.
/// Matches [`crate::mbtiles::MAX_MBTILES_BYTES`]'s figure, the same "resident,
/// no filesystem" ceiling for a single-file browser drop in this crate.
pub const MAX_GEOPARQUET_BYTES: usize = 512 * 1024 * 1024;

/// Builds a GeoJSON [`FeatureCollection`] from the bytes of a `.parquet` /
/// `.geoparquet` file.
///
/// See the module docs for the encoding/CRS/covering-column rules and for why
/// this reads the low-level pushdown machinery rather than
/// `GeoParquetReader::open`.
///
/// # Errors
///
/// Returns a [`LocalVectorError`] when `bytes` is past [`MAX_GEOPARQUET_BYTES`],
/// when the bytes are not a readable Parquet file, when the file has no
/// GeoParquet `geo` metadata, when the metadata is malformed, when the
/// declared CRS is neither WGS 84 nor Web Mercator, when a column chunk uses a
/// codec this build was not compiled with, or when the file holds no rows at
/// all (indistinguishable from a failed read otherwise, matching
/// [`crate::shapefile_input::from_bytes`]'s empty-collection rule).
pub fn from_bytes(bytes: &[u8]) -> Result<FeatureCollection, LocalVectorError> {
    read_dataset(bytes).map(|dataset| dataset.features)
}

/// What one GeoParquet file became: its features, already in WGS 84 lon/lat,
/// and the CRS its geometry column declared.
///
/// The CRS is carried out because it is provenance the layer keeps — see
/// [`oxigis_core::Layer::crs`].
#[derive(Debug, Clone)]
pub struct GeoParquetDataset {
    /// The features, in WGS 84 lon/lat.
    pub features: FeatureCollection,
    /// The CRS the file's `geo` metadata declared, or WGS 84 when it declared
    /// none (which the GeoParquet spec defines as OGC:CRS84).
    pub crs: SourceCrs,
}

/// [`from_bytes`], also reporting the CRS the file was read from.
///
/// # Errors
///
/// As [`from_bytes`].
pub fn read_dataset(bytes: &[u8]) -> Result<GeoParquetDataset, LocalVectorError> {
    if bytes.len() > MAX_GEOPARQUET_BYTES {
        return Err(LocalVectorError::new(format!(
            "the GeoParquet file is {} MiB, past the {} MiB this reader holds in memory at once",
            bytes.len() / (1024 * 1024),
            MAX_GEOPARQUET_BYTES / (1024 * 1024),
        )));
    }
    let data = Bytes::copy_from_slice(bytes);

    // 1. Footer/metadata read — a single pass over the trailing bytes, not
    //    the whole file.
    let arrow_meta = ArrowReaderMetadata::load(&data, ArrowReaderOptions::default())
        .map_err(|error| LocalVectorError::new(format!("not a readable Parquet file: {error}")))?;

    // 2. The GeoParquet "geo" file-metadata key, JSON-decoded, naming the
    //    primary geometry column and its encoding/CRS/covering.
    let geo_json = extract_geoparquet_metadata(arrow_meta.schema())
        .map_err(|error| LocalVectorError::new(format!("{error}")))?
        .ok_or_else(|| {
            LocalVectorError::new(
                "the file has no GeoParquet \u{201c}geo\u{201d} metadata; it may be a plain \
                 Parquet file with no geometry column",
            )
        })?;
    let geo_meta = GeoParquetMetadata::from_json(&geo_json)
        .map_err(|error| LocalVectorError::new(format!("invalid GeoParquet metadata: {error}")))?;
    let geometry_column = geo_meta.primary_column.clone();
    let col_meta = geo_meta
        .primary_column_metadata()
        .map_err(|error| LocalVectorError::new(format!("invalid GeoParquet metadata: {error}")))?;

    // 3. CRS: "absent" is the spec default (OGC:CRS84 = WGS 84); anything
    //    else is resolved to an EPSG code and reprojected, or refused by name.
    let source_crs = resolve_crs(col_meta.crs.as_ref());
    let crs = crate::crs_sniff::reprojector_or_refuse(&source_crs, "GeoParquet")?;
    let encoding = col_meta.encoding;
    let skip = skip_columns(&geo_meta);

    // 4. Read every row group. `execute_pushdown` with `bbox = None` and
    //    `filters = &[]` constructs no `ArrowPredicate` at all, making this a
    //    plain full read — see the module docs for why this reuses the
    //    pushdown machinery instead of `GeoParquetReader::open`.
    let row_groups: Vec<usize> = (0..arrow_meta.metadata().num_row_groups()).collect();
    let batches = execute_pushdown(
        data,
        arrow_meta,
        &geo_meta,
        &geometry_column,
        None,
        &[],
        row_groups,
        None,
        None,
    )
    .map_err(|error| codec_error(&error))?;

    let mut features = Vec::new();
    let format_options = FormatOptions::default();
    for batch in &batches {
        let geom_col = batch.column_by_name(&geometry_column).ok_or_else(|| {
            LocalVectorError::new(format!(
                "the geometry column \u{201c}{geometry_column}\u{201d} named in the file's own \
                 metadata is not in its schema",
            ))
        })?;
        // Null-preserving per encoding — see the module docs for why this
        // does not call `GeoParquetReader::read_geometries`/
        // `extract_geometries`.
        let geometries = decode_geometries(geom_col.as_ref(), encoding)?;

        for row in 0..batch.num_rows() {
            let geometry = geometries
                .get(row)
                .and_then(Option::as_ref)
                .and_then(|geometry| geometry::to_geojson_geometry(geometry, &crs));
            let properties = row_to_properties(batch, row, &skip, &format_options);
            features.push(Feature::new(geometry, Some(properties)));
        }
    }

    if features.is_empty() {
        return Err(LocalVectorError::new("the GeoParquet file holds no rows"));
    }
    Ok(GeoParquetDataset {
        features: FeatureCollection::new(features),
        crs: source_crs,
    })
}

/// Serialises a collection back to compact GeoJSON text.
///
/// The persistence leg for a drop with no filesystem path behind it (a build
/// with the `geoparquet` feature on but reading bytes it did not read from
/// disk itself) — the same bridge [`crate::shapefile_input::to_geojson_string`]
/// and [`crate::gpkg_input::to_geojson_string`] provide for their formats.
///
/// # Errors
///
/// Returns a [`LocalVectorError`] if the collection cannot be serialised.
pub fn to_geojson_string(features: &FeatureCollection) -> Result<String, LocalVectorError> {
    oxigeo::geojson::writer::to_string(features)
        .map_err(|error| LocalVectorError::new(format!("GeoJSON serialization failed: {error}")))
}

/// Decodes one geometry column into null-preserving geometries, dispatching
/// on the encoding the file's own metadata declares — never on the array's
/// runtime shape, which is what the upstream `extract_geometries` gets wrong.
fn decode_geometries(
    geom_col: &dyn Array,
    encoding: EncodingType,
) -> Result<Vec<Option<GpGeometry>>, LocalVectorError> {
    match encoding {
        EncodingType::Wkb => {
            let binary = geom_col
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| {
                    LocalVectorError::new(
                        "the geometry column is declared WKB-encoded but is not a binary column",
                    )
                })?;
            (0..binary.len())
                .map(|index| {
                    if binary.is_null(index) {
                        return Ok(None);
                    }
                    WkbReader::new(binary.value(index))
                        .read_geometry()
                        .map(Some)
                        .map_err(|error| {
                            LocalVectorError::new(format!("invalid WKB geometry: {error}"))
                        })
                })
                .collect()
        }
        native => decode_native_array_optional(geom_col, native)
            .map_err(|error| LocalVectorError::new(format!("invalid geometry column: {error}"))),
    }
}

/// Resolves the geometry column's declared CRS — see the module docs for the
/// PROJJSON-holding-a-bare-WKT2-string case this has to special-case.
///
/// Never fails: an unresolvable CRS becomes one whose EPSG code is
/// [`oxigis_core::EPSG_UNKNOWN`], which the reprojector refuses by name at the
/// call site. Keeping resolution and refusal apart is what lets the refusal
/// message quote the CRS the file actually declared.
fn resolve_crs(crs: Option<&Crs>) -> SourceCrs {
    match crs {
        // The GeoParquet spec's default for an absent `crs` is OGC:CRS84.
        None => SourceCrs::wgs84(),
        Some(Crs::ProjJson(json)) => {
            if let Some(wkt) = json.as_str() {
                return SourceCrs::from_wkt(wkt);
            }
            let authority = json.pointer("/id/authority").and_then(Value::as_str);
            let code = json.pointer("/id/code").and_then(json_as_u64);
            match (authority, code) {
                // PROJJSON states the code in the authority's own namespace;
                // only EPSG's is the one `oxigis-core`'s registry speaks.
                (Some("EPSG"), Some(code)) => match u32::try_from(code) {
                    Ok(code) if oxigis_core::crs::is_supported(code) => SourceCrs::from_epsg(code),
                    // A code this build cannot place: keep it *and* the file's
                    // own name, so the refusal reads
                    // "RGF93 v1 / Lambert-93 (EPSG:2154)" rather than a bare
                    // number the user has to go and look up.
                    Ok(code) => named_projjson(json, code),
                    Err(_) => named_projjson(json, oxigis_core::EPSG_UNKNOWN),
                },
                // OGC:CRS84 is WGS 84 with lon/lat axis order — which is the
                // order every GeoParquet WKB geometry is written in anyway.
                (Some("OGC"), _)
                    if json.get("name").and_then(Value::as_str) == Some("WGS 84 (CRS84)") =>
                {
                    SourceCrs::wgs84()
                }
                _ => named_projjson(json, oxigis_core::EPSG_UNKNOWN),
            }
        }
        // Unreachable from a file read today (see the module docs) but kept
        // so a `Crs` constructed in code, not just deserialized, still works.
        Some(Crs::Wkt2(wkt)) => SourceCrs::from_wkt(wkt),
    }
}

/// A PROJJSON object this reader cannot place, as a [`SourceCrs`] that still
/// names itself — the object's `name` member is what a refusal quotes.
///
/// The name is carried as a minimal `LOCAL_CS["…"]` string rather than as a
/// separate field on the model type: [`SourceCrs::name`] already falls back to
/// a WKT's root name, and `LOCAL_CS` is the one WKT keyword that resolves to no
/// CRS at all, so the value cannot be mistaken for a placeable definition.
fn named_projjson(json: &Value, epsg: u32) -> SourceCrs {
    let name = json
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("<unnamed PROJJSON CRS>");
    SourceCrs::new(
        epsg,
        Some(&format!("LOCAL_CS[\"{}\"]", name.replace('"', "'"))),
    )
}

/// A PROJJSON `id.code` as `u64`, whether the writer spelled it as a JSON
/// number (the common case) or as a string.
fn json_as_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
}

/// Columns excluded from the attribute table: every geometry column the
/// file's own `geo` metadata declares (primary and, if GeoParquet 1.1 ever
/// carries more than one, any others), plus the root of the primary column's
/// `covering.bbox` columns, if it declares one — see the module docs.
fn skip_columns(geo_meta: &GeoParquetMetadata) -> HashSet<String> {
    let mut skip: HashSet<String> = geo_meta.columns.keys().cloned().collect();
    if let Some(col_meta) = geo_meta.columns.get(&geo_meta.primary_column)
        && let Some(covering) = &col_meta.covering
    {
        for path in [
            &covering.bbox.xmin,
            &covering.bbox.ymin,
            &covering.bbox.xmax,
            &covering.bbox.ymax,
        ] {
            if let Some(root) = path.first() {
                skip.insert(root.clone());
            }
        }
    }
    skip
}

/// Maps one row's non-skipped columns onto GeoJSON properties.
///
/// Typed for the JSON-native primitive Arrow types; everything else (Date,
/// Timestamp, Decimal, List, Struct, …) is stringified via `arrow`'s
/// `ArrayFormatter` — reachable without the `prettyprint` feature (verified
/// against `arrow-59.1.0`: `arrow::util::display` has no `cfg` gate, only
/// `arrow::util::pretty` does) — so no column is silently dropped, matching
/// the shapefile driver's "stringify rather than drop" stance for anything it
/// cannot type natively.
fn row_to_properties(
    batch: &RecordBatch,
    row: usize,
    skip: &HashSet<String>,
    format_options: &FormatOptions<'_>,
) -> Properties {
    let mut properties = Properties::new();
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        if skip.contains(field.name().as_str()) {
            continue;
        }
        let value = cell_to_json(column.as_ref(), row, format_options);
        properties.insert(field.name().clone(), value);
    }
    properties
}

/// Maps one Arrow cell onto its JSON counterpart, `null` for both an
/// out-of-band null and a non-finite float (JSON has no encoding for one).
fn cell_to_json(column: &dyn Array, row: usize, format_options: &FormatOptions<'_>) -> Value {
    if column.is_null(row) {
        return Value::Null;
    }
    match column.data_type() {
        DataType::Utf8 => typed(column, |array: &StringArray| {
            Value::String(array.value(row).to_string())
        }),
        DataType::LargeUtf8 => typed(column, |array: &LargeStringArray| {
            Value::String(array.value(row).to_string())
        }),
        DataType::Boolean => typed(column, |array: &BooleanArray| Value::Bool(array.value(row))),
        DataType::Int8 => typed(column, |array: &Int8Array| Value::from(array.value(row))),
        DataType::Int16 => typed(column, |array: &Int16Array| Value::from(array.value(row))),
        DataType::Int32 => typed(column, |array: &Int32Array| Value::from(array.value(row))),
        DataType::Int64 => typed(column, |array: &Int64Array| Value::from(array.value(row))),
        DataType::UInt8 => typed(column, |array: &UInt8Array| Value::from(array.value(row))),
        DataType::UInt16 => typed(column, |array: &UInt16Array| Value::from(array.value(row))),
        DataType::UInt32 => typed(column, |array: &UInt32Array| Value::from(array.value(row))),
        DataType::UInt64 => typed(column, |array: &UInt64Array| Value::from(array.value(row))),
        DataType::Float32 => typed(column, |array: &Float32Array| {
            json_number(f64::from(array.value(row)))
        }),
        DataType::Float64 => typed(column, |array: &Float64Array| json_number(array.value(row))),
        _ => ArrayFormatter::try_new(column, format_options)
            .map(|formatter| Value::String(formatter.value(row).to_string()))
            .unwrap_or(Value::Null),
    }
}

/// Downcasts `column` to `T` and applies `extract`, or `null` if the runtime
/// array does not match the `DataType` arm that chose `T` (never happens for
/// a well-formed Arrow array — defensive rather than an `unwrap`).
fn typed<T: Array + 'static>(column: &dyn Array, extract: impl FnOnce(&T) -> Value) -> Value {
    column
        .as_any()
        .downcast_ref::<T>()
        .map_or(Value::Null, extract)
}

/// A JSON number, or `null` for a value JSON cannot spell (NaN/±infinity).
fn json_number(value: f64) -> Value {
    serde_json::Number::from_f64(value).map_or(Value::Null, Value::Number)
}

/// Translates a codec-disabled read failure into a message that names the
/// codec, instead of the raw multiply-wrapped error chain
/// (`GeoParquetError::Arrow` wrapping an `arrow::error::ArrowError::ParquetError`
/// wrapping the original `parquet::errors::ParquetError::General` — verified
/// against `parquet-59.1.0`'s `compression::create_codec`, whose message is
/// exactly `"Disabled feature at compile time: <feature>"`, where `<feature>`
/// is the *Cargo feature* name, not always the codec's own name — `flate2`
/// for Gzip).
///
/// Reached only from data (row-group) reads: metadata reads succeed even for
/// a file using a disabled codec, because the codec only touches column-chunk
/// decompression.
fn codec_error(error: &GeoParquetError) -> LocalVectorError {
    const MARKER: &str = "Disabled feature at compile time: ";
    let raw = error.to_string();
    let Some(token) = raw.split(MARKER).nth(1) else {
        return LocalVectorError::new(format!("could not read the GeoParquet file: {raw}"));
    };
    let codec = match token.trim() {
        "flate2" => "gzip",
        "snap" => "Snappy",
        other => other,
    };
    LocalVectorError::new(format!(
        "this GeoParquet file uses the {codec} codec, which this build was not compiled to read \
         (supported: uncompressed, Snappy, Brotli, LZ4)",
    ))
}
