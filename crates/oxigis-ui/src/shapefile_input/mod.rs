// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! ESRI Shapefile → GeoJSON [`FeatureCollection`], from **bytes only**.
//!
//! Phase 1 §1.3's first format. Everything downstream of this module — the
//! synthetic vector tile, the tessellator, the label placer, the attribute
//! table, the style panel — already works on any
//! [`oxigeo::geojson::types::FeatureCollection`] (see
//! [`crate::local_vector`]), so a shapefile only has to *become* one. Nothing
//! else in the local-layer pipeline is format-aware.
//!
//! # Why bytes, and only bytes
//!
//! `oxigeo-shapefile` ships a path-based [`ShapefileReader`] that finds the
//! `.shp`'s siblings, resolves the DBF code page and dereferences `.dbt` memos
//! by itself, and the obvious design is to call it from the desktop shell. It
//! was **rejected**, for a reason that only shows up once the types are lined
//! up: `ShapefileFeature::to_oxigeo_feature()` produces an
//! `oxigeo_core::vector::Feature`, whereas this crate's whole local-vector path
//! is written against the *GeoJSON* model (`Position = Vec<f64>`,
//! `Properties = serde_json::Map`). The two are unrelated types. Taking the
//! path-based reader would therefore need a second, hand-written
//! core → GeoJSON bridge *in addition to* the wasm-side byte reader, i.e. two
//! converters to keep in step instead of one.
//!
//! So there is exactly one reader, [`read_dataset`], built on the low-level
//! [`ShpReader`]/[`DbfReader`], which are generic over [`std::io::Read`] and
//! therefore run over a `Cursor<&[u8]>` on `wasm32`. [`from_bytes`] is that
//! same reader with the source CRS dropped, for callers that only want the
//! features. The desktop shell reads the `.shp` and its siblings with
//! `std::fs` and calls the same function the browser drop calls; `oxigis-ui`
//! itself stays filesystem-free.
//!
//! [`ShapefileReader`]: oxigeo::shapefile::ShapefileReader
//!
//! # What is kept and what is dropped
//!
//! * **2-D only.** Z and M ordinates are read (the parser needs them to walk
//!   the record) and then discarded: the map is a Web Mercator plane and
//!   [`crate::local_vector::project_position`] only ever looks at `[0]`/`[1]`.
//!   A `PolyLineZ` therefore arrives as a plain `LineString`.
//! * **`MultiPatch`** becomes a `MultiPoint` of its vertices — the same
//!   lossy-but-visible choice `oxigeo-shapefile`'s own reader makes, rather
//!   than dropping the record.
//! * **Polygon rings** are regrouped by winding + containment, mirroring
//!   `oxigeo-shapefile`'s (crate-private) `assemble_polygons`: a clockwise ring
//!   opens a new exterior, a counter-clockwise ring is a hole assigned to the
//!   smallest exterior that contains it. See [`assemble_rings`].
//! * **Memo (`M`) columns are skipped.** Without a sibling `.dbt` — which a
//!   byte drop never has — the DBF parser hands back the *block pointer* as a
//!   string, so keeping the column would show numbers where text belongs.
//! * **Null shapes** yield a feature with `geometry: null`, keeping the DBF row
//!   visible in the attribute table.
//!
//! # CRS
//!
//! [`sniff_prj`] resolves the `.prj` WKT to an [`oxigis_core::Crs`] with a real
//! parser — the root `AUTHORITY`/`ID` clause first, then the name forms an
//! ESRI `.prj` takes — and hands back the [`oxigis_core::Reprojector`] that
//! places the file's coordinates on WGS 84. Three outcomes, and no fourth:
//!
//! 1. no `.prj` at all → WGS 84, the near-universal convention for `.prj`-less
//!    data, and coordinates pass through as lon/lat;
//! 2. a CRS `oxigis-core` can place — WGS 84, Web Mercator, every UTM zone,
//!    all nineteen Japan Plane Rectangular zones on each of JGD2011, JGD2000
//!    and Tokyo Datum, the British National Grid, and the national geographic
//!    datums — is **reprojected per vertex** on the way in;
//! 3. anything else is refused, with the CRS and its EPSG code named.
//!
//! This module used to recognise exactly two CRSs and refuse the rest, on the
//! grounds that "an honest refusal beats silently drawing a UTM dataset off
//! the coast of Africa". That was the right call while there was no projection
//! engine wired up. There is one now (`oxigis_core::crs::reproject`, over
//! OxiGeo's `proj` feature), so the refusal has moved to where it belongs: the
//! CRSs nothing in this build can invert, rather than every CRS but two.
//!
//! Axis order is `(x, y) = (easting, northing)`, because that is what a `.shp`
//! record holds — the format has no axis-order concept and every writer puts
//! the easting first, including for the Japanese zones, whose EPSG definition
//! declares the opposite. See `oxigis_core::crs::AxisOrder`.

#[cfg(test)]
pub(crate) mod fixture;
#[cfg(test)]
mod tests;

use std::io::Cursor;

use oxigeo::geojson::types::{
    Feature, FeatureCollection, Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon,
    Point, Polygon, Position, Properties,
};
use oxigeo::shapefile::dbf::{
    DbfHeader, DbfReader, FieldDescriptor, FieldType, FieldValue, resolve_cpg,
};
use oxigeo::shapefile::shp::{MultiPartShape, Shape, ShpReader};
use oxigis_core::Crs;
use oxigis_core::crs::Reprojector;
use serde_json::Value;

use crate::local_vector::LocalVectorError;

/// The CRS a `.prj` file was recognised as, together with the reprojection
/// that places its coordinates on WGS 84.
///
/// A [`Copy`] handle over `oxigis_core`'s [`Reprojector`], holding nothing but
/// numbers — no allocation, so the reader's per-vertex loop still allocates
/// exactly once per position. The geometry helpers take it by reference rather
/// than by value (a `Reprojector` is a couple of hundred bytes of projection
/// and datum parameters, against the one byte the two-variant enum this
/// replaces cost), so a million-vertex dataset does not memcpy it a million
/// times.
///
/// See the module docs for the three outcomes and why there is no "unknown but
/// let's try anyway" one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrjCrs(Reprojector);

impl PrjCrs {
    /// WGS 84 geographic — what a shapefile with **no** `.prj` at all is
    /// assumed to be, which is the near-universal convention for `.prj`-less
    /// data.
    #[must_use]
    pub fn wgs84() -> Self {
        Self(Reprojector::wgs84())
    }

    /// The EPSG code the `.prj` resolved to.
    #[must_use]
    pub fn epsg(&self) -> u32 {
        self.0.source_epsg()
    }

    /// The CRS as the model records it on a layer.
    #[must_use]
    pub fn to_crs(self) -> Crs {
        Crs::from_epsg(self.0.source_epsg())
    }

    /// The reprojection itself, for a caller that wants to place points of its
    /// own (a bounding box, say) in the same frame the features landed in.
    #[must_use]
    pub fn reprojector(&self) -> Reprojector {
        self.0
    }

    /// Whether coordinates pass through untouched.
    #[must_use]
    pub fn is_wgs84(&self) -> bool {
        self.0.is_identity()
    }

    /// Maps one `(x, y)` pair from the `.shp` into lon/lat degrees, or
    /// [`None`] for a pair the projection cannot invert.
    #[must_use]
    fn lon_lat(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        self.0.to_lon_lat(x, y)
    }
}

/// Resolves a `.prj` WKT string to the CRS its data is in.
///
/// [`None`] — no `.prj` beside the `.shp` — is [`PrjCrs::wgs84`]: see the
/// module docs. The WKT reading itself, the EPSG registry and the projection
/// are `oxigis-core`'s (`oxigis_core::crs`), shared with the GeoPackage and
/// GeoParquet drivers; only the "no `.prj` at all" default and this format's
/// refusal wording stay here.
///
/// # Errors
///
/// Returns a [`LocalVectorError`] naming the CRS and its EPSG code when it is
/// one this build cannot place, because loading it would put the data
/// somewhere wrong rather than not at all.
pub fn sniff_prj(wkt: Option<&str>) -> Result<PrjCrs, LocalVectorError> {
    Ok(PrjCrs(sniff_prj_crs(wkt)?.1))
}

/// [`sniff_prj`], also handing back the [`Crs`] so the layer can record what
/// its source file was in.
///
/// # Errors
///
/// As [`sniff_prj`].
fn sniff_prj_crs(wkt: Option<&str>) -> Result<(Crs, Reprojector), LocalVectorError> {
    // A UTF-8 BOM is common on `.prj` files written on Windows; `crs_from_wkt`
    // strips it, and answers `None` for an absent or empty sidecar.
    let Some(crs) = crate::crs_sniff::crs_from_wkt(wkt) else {
        return Ok((Crs::wgs84(), Reprojector::wgs84()));
    };
    let reprojector = crate::crs_sniff::reprojector_or_refuse(&crs, "shapefile")?;
    // `compact` past the refusal, not before it: the refusal above quotes the
    // `.prj`'s own CRS name, while what a *loaded* layer records is the code —
    // which, having resolved, names the CRS completely on its own. See
    // `oxigis_core::Crs::compact`.
    Ok((crs.compact(), reprojector))
}

/// Builds a GeoJSON [`FeatureCollection`] from the bytes of a shapefile set.
///
/// `dbf` is optional: without it the features carry geometry and no properties
/// at all, which is what a lone `.shp` drop produces. `prj` and `cpg` are the
/// **text** of those sibling files (they are tiny and always text), used for
/// [`sniff_prj`] and for the DBF code page respectively; without a `.cpg` the
/// encoding falls back to the DBF header's own language-driver byte and then to
/// UTF-8, exactly as `oxigeo-shapefile`'s path reader does.
///
/// # Errors
///
/// Returns a [`LocalVectorError`] when the `.shp` or `.dbf` cannot be parsed,
/// when the two disagree on the record count, when the CRS is one this crate
/// refuses (see [`sniff_prj`]), or when the file holds no records — an empty
/// layer is indistinguishable from a failed drop, so it is reported as one,
/// matching [`crate::local_vector::LocalVectorLayer::from_geojson`].
pub fn from_bytes(
    shp: &[u8],
    dbf: Option<&[u8]>,
    prj: Option<&str>,
    cpg: Option<&str>,
) -> Result<FeatureCollection, LocalVectorError> {
    read_dataset(shp, dbf, prj, cpg).map(|dataset| dataset.features)
}

/// What one shapefile set became: its features, already in WGS 84 lon/lat, and
/// the CRS they were read *from*.
///
/// The CRS is carried out because it is provenance the layer keeps — see
/// [`oxigis_core::Layer::crs`]. Every coordinate in `features` is WGS 84
/// regardless; this says what they were converted from.
#[derive(Debug, Clone)]
pub struct ShapefileDataset {
    /// The features, in WGS 84 lon/lat.
    pub features: FeatureCollection,
    /// The CRS the `.prj` declared, or WGS 84 when there was no `.prj`.
    pub crs: Crs,
}

/// [`from_bytes`], also reporting the CRS the set was read from.
///
/// # Errors
///
/// As [`from_bytes`].
pub fn read_dataset(
    shp: &[u8],
    dbf: Option<&[u8]>,
    prj: Option<&str>,
    cpg: Option<&str>,
) -> Result<ShapefileDataset, LocalVectorError> {
    let (source_crs, reprojector) = sniff_prj_crs(prj)?;
    let crs = PrjCrs(reprojector);

    let mut shp_reader = ShpReader::new(Cursor::new(shp))
        .map_err(|error| LocalVectorError::new(format!(".shp header rejected: {error}")))?;
    let shapes = shp_reader
        .read_all_records()
        .map_err(|error| LocalVectorError::new(format!(".shp record rejected: {error}")))?;

    let features: Vec<Feature> = match dbf {
        Some(bytes) => {
            let rows = read_dbf(bytes, cpg, shapes.len())?;
            shapes
                .iter()
                .zip(rows)
                .filter_map(|(record, row)| match row {
                    DbfRow::Kept(properties) => Some(Feature::new(
                        shape_to_geometry(&record.shape, &crs),
                        Some(properties),
                    )),
                    // The DBF marks a deleted row with its marker byte
                    // (0x2A) without removing it or its paired `.shp`
                    // shape; every reference reader (GDAL/OGR, `shapelib`,
                    // GeoPandas) hides it by default. Keeping the shape with
                    // no properties would still draw it on the map, so the
                    // pair is dropped entirely rather than surfaced as an
                    // attribute-less feature.
                    DbfRow::Deleted => None,
                })
                .collect()
        }
        None => shapes
            .iter()
            .map(|record| Feature::new(shape_to_geometry(&record.shape, &crs), None))
            .collect(),
    };

    if features.is_empty() {
        return Err(LocalVectorError::new("the shapefile holds no records"));
    }
    Ok(ShapefileDataset {
        features: FeatureCollection::new(features),
        crs: source_crs,
    })
}

/// The bytes of one shapefile set, borrowed.
///
/// A carrier so the "load this set" entry points down the stack
/// ([`crate::local_input::LocalInputState::add_shapefile`],
/// [`crate::OxigisApp::add_shapefile_layer_from_bytes`], and the desktop
/// shell's reader) take one argument instead of four, and so adding a sixth
/// member later is not a signature change everywhere.
#[derive(Debug, Clone, Copy)]
pub struct ShapefileBytes<'a> {
    /// The `.shp` geometry file — the only mandatory member.
    pub shp: &'a [u8],
    /// The `.dbf` attribute table, if there is one.
    pub dbf: Option<&'a [u8]>,
    /// The `.prj` WKT text, if there is one.
    pub prj: Option<&'a str>,
    /// The `.cpg` encoding label, if there is one.
    pub cpg: Option<&'a str>,
}

impl<'a> ShapefileBytes<'a> {
    /// A set holding only a `.shp` — a geometry-only, WGS 84 layer.
    #[must_use]
    pub fn new(shp: &'a [u8]) -> Self {
        Self {
            shp,
            dbf: None,
            prj: None,
            cpg: None,
        }
    }

    /// Attaches the `.dbf` attribute table.
    #[must_use]
    pub fn with_dbf(mut self, dbf: Option<&'a [u8]>) -> Self {
        self.dbf = dbf;
        self
    }

    /// Attaches the `.prj` WKT and the `.cpg` label.
    #[must_use]
    pub fn with_sidecars(mut self, prj: Option<&'a str>, cpg: Option<&'a str>) -> Self {
        self.prj = prj;
        self.cpg = cpg;
        self
    }

    /// Reads the set — see [`from_bytes`], which this forwards to.
    ///
    /// # Errors
    ///
    /// Propagates every failure [`from_bytes`] reports.
    pub fn to_feature_collection(self) -> Result<FeatureCollection, LocalVectorError> {
        from_bytes(self.shp, self.dbf, self.prj, self.cpg)
    }

    /// Reads the set, keeping the CRS it was read from — see [`read_dataset`].
    ///
    /// # Errors
    ///
    /// Propagates every failure [`read_dataset`] reports.
    pub fn to_dataset(self) -> Result<ShapefileDataset, LocalVectorError> {
        read_dataset(self.shp, self.dbf, self.prj, self.cpg)
    }
}

/// Serialises a collection back to compact GeoJSON text.
///
/// The web persistence leg: shapefile bytes cannot be embedded in a project
/// document, so a browser-dropped set is stored as
/// [`oxigis_core::VectorSource::InlineGeoJson`] instead. See
/// [`crate::local_input::LocalInputState::add_shapefile`].
///
/// # Errors
///
/// Returns a [`LocalVectorError`] if the collection cannot be serialised.
pub fn to_geojson_string(features: &FeatureCollection) -> Result<String, LocalVectorError> {
    oxigeo::geojson::writer::to_string(features)
        .map_err(|error| LocalVectorError::new(format!("GeoJSON serialization failed: {error}")))
}

/// One DBF row's outcome for [`read_dbf`].
enum DbfRow {
    /// An active record's properties.
    Kept(Properties),
    /// The record's deletion flag (marker byte `0x2A`) was set; the caller
    /// must drop this record's paired `.shp` shape too, not only its
    /// attributes — see [`from_bytes`].
    Deleted,
}

/// Reads every DBF row, one entry per shape record, marking which ones carry
/// the deletion flag rather than dropping them here: the caller needs the
/// shape index alongside the flag to also skip the paired `.shp` record.
fn read_dbf(
    bytes: &[u8],
    cpg: Option<&str>,
    shape_count: usize,
) -> Result<Vec<DbfRow>, LocalVectorError> {
    let mut reader = DbfReader::new(Cursor::new(bytes))
        .map_err(|error| LocalVectorError::new(format!(".dbf header rejected: {error}")))?;
    validate_record_count(reader.header(), reader.field_descriptors(), bytes.len())?;
    if let Some(encoding) = cpg
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .and_then(resolve_cpg)
    {
        reader.set_encoding(encoding);
    }
    let descriptors = reader.field_descriptors().to_vec();
    let records = reader
        .read_all_records()
        .map_err(|error| LocalVectorError::new(format!(".dbf record rejected: {error}")))?;

    if records.len() != shape_count {
        return Err(LocalVectorError::new(format!(
            "the .shp holds {shape_count} records but the .dbf holds {}; the two files do not \
             belong together",
            records.len(),
        )));
    }

    let deleted = records.iter().filter(|record| record.deleted).count();
    if deleted > 0 {
        // `from_bytes`'s return type carries no room for a notice about this
        // (unlike a whole-file refusal, it is not an error), so the count is
        // logged rather than silently dropped.
        tracing::debug!(
            deleted,
            "shapefile_input: skipping records marked deleted in the .dbf",
        );
    }

    Ok(records
        .iter()
        .map(|record| {
            if record.deleted {
                DbfRow::Deleted
            } else {
                DbfRow::Kept(row_to_properties(&descriptors, &record.values))
            }
        })
        .collect())
}

/// Refuses a `.dbf` whose header claims more records than the file could
/// possibly hold.
///
/// [`DbfReader::new`] bounds `header_size` against the field-descriptor count
/// it parses from it, but leaves `record_count` unchecked, and
/// [`DbfReader::read_all_records`] uses it verbatim as a `Vec::with_capacity`
/// argument — whose failure path is `handle_alloc_error`, an abort rather
/// than a `Result` this module's "every failure is a message" architecture
/// can intercept. A file a few dozen bytes long can declare
/// `record_count = 0xFFFF_FFFF`, asking for a reserve of tens of gigabytes.
///
/// The per-record byte cost is computed from `descriptors` — one marker byte
/// plus each field's own `length` — rather than trusted from the header's own
/// `record_size`: `record_size` is a second, independently hostile field that
/// `DbfRecord::read` never actually consults, so a crafted `record_size = 0`
/// alongside a huge `record_count` would pass a check built on the header
/// value alone while still reaching the same unbounded reserve. Every
/// `length` is a `u8` and `DbfReader::new` already bounds the descriptor
/// count (`MAX_DBF_FIELDS`), so this sum cannot itself overflow; the
/// `checked_mul`/`checked_add` below keep the reserve-size arithmetic from
/// wrapping into a false pass on wasm32's 32-bit `usize`.
fn validate_record_count(
    header: &DbfHeader,
    descriptors: &[FieldDescriptor],
    byte_len: usize,
) -> Result<(), LocalVectorError> {
    let record_bytes: usize = 1 + descriptors
        .iter()
        .map(|field| usize::from(field.length))
        .sum::<usize>();
    let record_count = usize::try_from(header.record_count).unwrap_or(usize::MAX);
    let header_size = usize::from(header.header_size);
    let required = record_count
        .checked_mul(record_bytes)
        .and_then(|body| body.checked_add(header_size));
    match required {
        Some(required) if required <= byte_len => Ok(()),
        _ => Err(LocalVectorError::new(format!(
            ".dbf header declares {} records ({record_bytes} bytes each, {header_size} byte \
             header), which does not fit in the {byte_len}-byte file",
            header.record_count,
        ))),
    }
}

/// Maps one DBF row onto GeoJSON properties, skipping memo columns.
fn row_to_properties(descriptors: &[FieldDescriptor], values: &[FieldValue]) -> Properties {
    let mut properties = Properties::new();
    for (descriptor, value) in descriptors.iter().zip(values) {
        if descriptor.field_type == FieldType::Memo {
            continue;
        }
        properties.insert(descriptor.name.clone(), field_value_to_json(value));
    }
    properties
}

/// Maps one DBF cell onto its JSON counterpart.
///
/// Dates are normalised from the on-disk `YYYYMMDD` to ISO-8601 `YYYY-MM-DD`,
/// which is what every other reader in the stack (and every human) expects; a
/// malformed date is kept verbatim rather than dropped. A non-finite numeric
/// becomes `null`, because JSON has no encoding for one.
fn field_value_to_json(value: &FieldValue) -> Value {
    match value {
        FieldValue::String(text) => Value::String(text.clone()),
        FieldValue::Integer(number) => Value::from(*number),
        FieldValue::Float(number) => {
            serde_json::Number::from_f64(*number).map_or(Value::Null, Value::Number)
        }
        FieldValue::Boolean(flag) => Value::Bool(*flag),
        FieldValue::Date(raw) => Value::String(iso_date(raw)),
        FieldValue::Null => Value::Null,
    }
}

/// `YYYYMMDD` → `YYYY-MM-DD`, or the input unchanged when it is not that.
fn iso_date(raw: &str) -> String {
    if raw.len() == 8 && raw.bytes().all(|byte| byte.is_ascii_digit()) {
        format!("{}-{}-{}", &raw[0..4], &raw[4..6], &raw[6..8])
    } else {
        raw.to_string()
    }
}

/// Converts one shape record to a GeoJSON geometry, or [`None`] for a null /
/// degenerate / empty shape (the feature is still emitted, with a null
/// geometry, so its attribute row survives).
fn shape_to_geometry(shape: &Shape, crs: &PrjCrs) -> Option<Geometry> {
    match shape {
        Shape::Null => None,
        Shape::Point(point) => single_point(point.x, point.y, crs),
        Shape::PointZ(point) => single_point(point.x, point.y, crs),
        Shape::PointM(point) => single_point(point.x, point.y, crs),
        Shape::PolyLine(base) => lines(base, crs),
        Shape::PolyLineZ(shape) => lines(&shape.base, crs),
        Shape::PolyLineM(shape) => lines(&shape.base, crs),
        Shape::Polygon(base) => assemble_rings(split_parts(base, crs)),
        Shape::PolygonZ(shape) => assemble_rings(split_parts(&shape.base, crs)),
        Shape::PolygonM(shape) => assemble_rings(split_parts(&shape.base, crs)),
        Shape::MultiPoint(base) => multi_point(base, crs),
        Shape::MultiPointZ(shape) => multi_point(&shape.base, crs),
        Shape::MultiPointM(shape) => multi_point(&shape.base, crs),
        // Mirrors `oxigeo-shapefile`'s own reader: a 3-D patch is surfaced as
        // the point cloud of its vertices rather than dropped.
        Shape::MultiPatch(shape) => multi_point(&shape.base, crs),
    }
}

/// A `[lon, lat]` GeoJSON position, or [`None`] if the pair does not invert —
/// a non-finite ordinate (some producers write `-1e38` as a no-data marker) or
/// a point the projection takes off the ellipsoid.
fn position(x: f64, y: f64, crs: &PrjCrs) -> Option<Position> {
    let (lon, lat) = crs.lon_lat(x, y)?;
    Some(vec![lon, lat])
}

/// A single-point geometry.
fn single_point(x: f64, y: f64, crs: &PrjCrs) -> Option<Geometry> {
    Point::new(position(x, y, crs)?).ok().map(Geometry::Point)
}

/// Splits a multi-part shape into its parts, projected to lon/lat.
///
/// Out-of-range or inverted part offsets are skipped rather than panicking,
/// matching the defensive stance of the upstream reader.
fn split_parts(shape: &MultiPartShape, crs: &PrjCrs) -> Vec<Vec<Position>> {
    let mut parts = Vec::with_capacity(shape.parts.len());
    for index in 0..shape.parts.len() {
        let start = usize::try_from(shape.parts[index]).unwrap_or(usize::MAX);
        let end = match shape.parts.get(index + 1) {
            Some(next) => usize::try_from(*next).unwrap_or(usize::MAX),
            None => shape.points.len(),
        };
        if start >= end || end > shape.points.len() {
            continue;
        }
        let part: Vec<Position> = shape.points[start..end]
            .iter()
            .filter_map(|point| position(point.x, point.y, crs))
            .collect();
        parts.push(part);
    }
    parts
}

/// A `LineString` (one usable part) or `MultiLineString` (several).
fn lines(shape: &MultiPartShape, crs: &PrjCrs) -> Option<Geometry> {
    let mut parts: Vec<Vec<Position>> = split_parts(shape, crs)
        .into_iter()
        .filter(|part| part.len() >= 2)
        .collect();
    match parts.len() {
        0 => None,
        1 => LineString::new(parts.remove(0))
            .ok()
            .map(Geometry::LineString),
        _ => MultiLineString::new(parts)
            .ok()
            .map(Geometry::MultiLineString),
    }
}

/// Every vertex of a multi-part shape as one `MultiPoint`.
fn multi_point(shape: &MultiPartShape, crs: &PrjCrs) -> Option<Geometry> {
    let points: Vec<Position> = shape
        .points
        .iter()
        .filter_map(|point| position(point.x, point.y, crs))
        .collect();
    if points.is_empty() {
        return None;
    }
    MultiPoint::new(points).ok().map(Geometry::MultiPoint)
}

/// Twice the signed area of a ring (shoelace). Positive is counter-clockwise in
/// a y-up frame, which after projection to lon/lat is still y-up.
fn signed_area2(ring: &[Position]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for index in 0..ring.len() {
        let a = &ring[index];
        let b = &ring[(index + 1) % ring.len()];
        match (a.first(), a.get(1), b.first(), b.get(1)) {
            (Some(ax), Some(ay), Some(bx), Some(by)) => sum += ax.mul_add(*by, -(bx * ay)),
            _ => return 0.0,
        }
    }
    sum
}

/// Ray-casting point-in-ring test.
fn point_in_ring(px: f64, py: f64, ring: &[Position]) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = ring.len() - 1;
    for i in 0..ring.len() {
        let (Some(xi), Some(yi)) = (ring[i].first().copied(), ring[i].get(1).copied()) else {
            return inside;
        };
        let (Some(xj), Some(yj)) = (ring[j].first().copied(), ring[j].get(1).copied()) else {
            return inside;
        };
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Regroups a shape record's flat ring list into one or more GeoJSON polygons.
///
/// The ESRI convention distinguishes exteriors from holes by **winding**, not
/// by part order: a clockwise ring opens a polygon, a counter-clockwise ring is
/// a hole. A single `Polygon` record therefore carries every ring of, say, a
/// country and its islands, and the holes must be matched to their exteriors by
/// containment — which is what this does, choosing the *smallest* containing
/// exterior so nested islands work.
///
/// Two defensive cases mirror `oxigeo-shapefile`'s own `assemble_polygons`:
/// a file whose rings are *all* counter-clockwise (non-conformant producers do
/// exist) treats every ring as its own exterior, and a hole enclosed by nothing
/// is surfaced as a standalone polygon rather than silently dropped.
///
/// Ring winding is *not* normalised here: `crate::local_vector` measures the
/// shoelace of every ring it quantises and orients it itself, so both
/// conventions render identically.
#[must_use]
pub fn assemble_rings(rings: Vec<Vec<Position>>) -> Option<Geometry> {
    // A valid shapefile ring is closed, hence at least four vertices.
    let rings: Vec<Vec<Position>> = rings.into_iter().filter(|ring| ring.len() >= 4).collect();
    if rings.is_empty() {
        return None;
    }
    let areas: Vec<f64> = rings.iter().map(|ring| signed_area2(ring)).collect();

    let mut outers: Vec<usize> = Vec::new();
    let mut holes: Vec<usize> = Vec::new();
    for (index, area) in areas.iter().enumerate() {
        if *area < 0.0 {
            outers.push(index);
        } else {
            holes.push(index);
        }
    }
    if outers.is_empty() {
        outers = (0..rings.len()).collect();
        holes.clear();
    }

    let mut owned: Vec<Vec<usize>> = vec![Vec::new(); outers.len()];
    let mut orphans: Vec<usize> = Vec::new();
    for hole in holes {
        // The hole's own first vertex lies inside whichever exterior encloses
        // it, which is enough to match against the outer rings.
        let (Some(px), Some(py)) = (
            rings[hole].first().and_then(|p| p.first().copied()),
            rings[hole].first().and_then(|p| p.get(1).copied()),
        ) else {
            orphans.push(hole);
            continue;
        };
        let mut best: Option<(usize, f64)> = None;
        for (slot, outer) in outers.iter().enumerate() {
            if point_in_ring(px, py, &rings[*outer]) {
                let area = areas[*outer].abs();
                if best.is_none_or(|(_, smallest)| area < smallest) {
                    best = Some((slot, area));
                }
            }
        }
        match best {
            Some((slot, _)) => owned[slot].push(hole),
            None => orphans.push(hole),
        }
    }

    let mut polygons: Vec<Vec<Vec<Position>>> = Vec::with_capacity(outers.len() + orphans.len());
    for (slot, outer) in outers.iter().enumerate() {
        let mut polygon = vec![rings[*outer].clone()];
        for hole in &owned[slot] {
            polygon.push(rings[*hole].clone());
        }
        polygons.push(polygon);
    }
    for orphan in orphans {
        polygons.push(vec![rings[orphan].clone()]);
    }

    match polygons.len() {
        0 => None,
        1 => polygons
            .pop()
            .and_then(|rings| Polygon::new(rings).ok())
            .map(Geometry::Polygon),
        _ => MultiPolygon::new(polygons).ok().map(Geometry::MultiPolygon),
    }
}
