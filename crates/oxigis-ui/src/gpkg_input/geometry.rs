// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The GeoPackage geometry blob: a "GP" header wrapped around standard WKB,
//! decoded straight into the GeoJSON object model the local vector pipeline
//! speaks.
//!
//! # What is dropped, and why
//!
//! * **Z and M ordinates.** They are read — the parser needs them to walk the
//!   coordinate stream — and then discarded, exactly as
//!   [`crate::shapefile_input`] discards a `PolyLineZ`'s: the map is a Web
//!   Mercator plane and `crate::local_vector::project_position` only ever looks
//!   at `[0]`/`[1]`.
//! * **The envelope** in the GP header. It is a cached bounding box, and this
//!   crate computes its own extent from the vertices.
//! * **The per-blob `srs_id`.** The spec requires it to match the one
//!   `gpkg_geometry_columns` registered for the column, and that registration
//!   is what the table's CRS policy was decided from, so honouring a
//!   disagreeing blob would place one feature somewhere the rest of its layer
//!   is not.
//!
//! # What is refused
//!
//! Anything whose meaning is not knowable: a header version other than 0, the
//! "extended geometry type" flag (whose payload is vendor-defined), an envelope
//! indicator in the reserved range 5..=7, and a WKB type code outside the seven
//! OGC types with their Z/M variants. Refusal is per *table*, so a file's other
//! layers still load — see [`super::from_bytes`].

use oxigeo::geojson::types::{
    Geometry, GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point,
    Polygon, Position,
};
use oxigis_core::Crs;
use oxigis_core::crs::Reprojector;

use crate::local_vector::LocalVectorError;

/// Nesting cap for `GeometryCollection`s. Real data nests zero or one deep;
/// this only stops a crafted blob from recursing until the stack ends.
const MAX_DEPTH: u32 = 32;

/// Bytes each GP envelope indicator occupies (indicators 5..=7 are reserved).
const ENVELOPE_BYTES: [usize; 5] = [0, 32, 48, 48, 64];

/// Ceiling on what a count read out of a blob may *eagerly* reserve, in bytes.
///
/// [`counted`] already refuses a count larger than the bytes behind it could
/// possibly encode, but that bound is in *wire* bytes and the reserve is in
/// *element* bytes, and the two are not the same size: a member of a
/// multi-geometry needs only 5 bytes on the wire (byte order + type code) while
/// a `Geometry` slot costs `size_of::<Geometry>()`, and a ring needs only 4
/// (a position count of zero) against a `Vec<Position>` slot. Reserving
/// `count` slots therefore amplifies a crafted blob several-fold *before a
/// single member has been parsed*, and `Vec::with_capacity`'s failure path is
/// `handle_alloc_error`, which aborts — it is not a `Result` this reader's
/// "every failure is a message" architecture can intercept, and on `wasm32`'s
/// hard-capped linear memory the abort is the likely outcome rather than the
/// exotic one.
///
/// 64 KiB is far above any real geometry's first allocation and far below
/// anything worth aborting over; past it the `Vec` simply grows as members are
/// actually read, so the allocation stays proportional to real content.
const MAX_PREALLOC_BYTES: usize = 64 * 1024;

/// A `Vec` pre-sized for `count` elements, but never reserving more than
/// [`MAX_PREALLOC_BYTES`] before any of them exists.
///
/// The cap is computed from `size_of::<T>()` rather than written as an element
/// count so it stays right on both targets this crate ships to, whose pointer
/// widths (and therefore element sizes) differ.
fn prealloc<T>(count: usize) -> Vec<T> {
    let per_element = core::mem::size_of::<T>().max(1);
    Vec::with_capacity(count.min(MAX_PREALLOC_BYTES / per_element))
}

/// Builds a geometry error, prefixed so a status line says which layer of the
/// stack refused the file.
fn err(message: impl AsRef<str>) -> LocalVectorError {
    LocalVectorError::new(format!("GeoPackage geometry: {}", message.as_ref()))
}

/// The CRS a GeoPackage feature table was registered in, together with the
/// reprojection that places its coordinates on WGS 84.
///
/// The same policy [`crate::shapefile_input::PrjCrs`] applies to a `.prj`, made
/// from a different input: there the CRS is read out of WKT, here it is a
/// numeric `srs_id` resolved against the file's own `gpkg_spatial_ref_sys`
/// table (by `organization`/`organization_coordsys_id` first, then by the row's
/// WKT `definition`). There is deliberately no "unknown but let us try anyway"
/// variant: a table whose CRS resolves to nothing is refused on its own, and
/// the file's other tables still load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpkgCrs(Reprojector);

impl GpkgCrs {
    /// WGS 84 geographic — also what the "undefined" SRS ids 0 and -1 mean.
    #[must_use]
    pub fn wgs84() -> Self {
        Self(Reprojector::wgs84())
    }

    /// The reprojection for `crs`, or [`None`] when this build cannot place it.
    #[must_use]
    pub fn for_crs(crs: &Crs) -> Option<Self> {
        crs.reprojector().ok().map(Self)
    }

    /// The EPSG code the table's `srs_id` resolved to.
    #[must_use]
    pub fn epsg(&self) -> u32 {
        self.0.source_epsg()
    }

    /// Whether coordinates pass through untouched.
    #[must_use]
    pub fn is_wgs84(&self) -> bool {
        self.0.is_identity()
    }

    /// Maps one stored `(x, y)` pair into lon/lat degrees, or [`None`] when the
    /// pair does not invert.
    fn lon_lat(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        self.0.to_lon_lat(x, y)
    }
}

/// Decodes one GeoPackage geometry blob.
///
/// [`None`] is a legitimately *empty* geometry — the header's empty flag, or a
/// point whose ordinates are `NaN`, which is how OGC WKB spells "no point".
/// Such a feature is still emitted, with a null geometry, so its attribute row
/// stays in the table.
///
/// # Errors
///
/// Returns a [`LocalVectorError`] for a blob that is not a GeoPackage geometry,
/// uses a header version or flag this crate cannot interpret, or holds WKB that
/// is truncated or of an unknown type.
pub fn decode(blob: &[u8], crs: &GpkgCrs) -> Result<Option<Geometry>, LocalVectorError> {
    let header = blob
        .get(..8)
        .ok_or_else(|| err("the blob is too short to hold a header"))?;
    if header[0] != b'G' || header[1] != b'P' {
        return Err(err("the blob does not start with the GeoPackage magic"));
    }
    if header[2] != 0 {
        return Err(err(format!(
            "header version {} is newer than this reader",
            header[2]
        )));
    }
    let flags = header[3];
    if flags & 0x20 != 0 {
        return Err(err(
            "the geometry uses a vendor-extended type, which has no portable meaning",
        ));
    }
    let indicator = usize::from((flags >> 1) & 0x07);
    let envelope = *ENVELOPE_BYTES
        .get(indicator)
        .ok_or_else(|| err(format!("envelope indicator {indicator} is reserved")))?;
    // Bit 0 chooses the byte order of the header's own integers; the WKB body
    // restates its own, so this only governs the srs_id and the envelope, both
    // of which are skipped.
    let wkb_start = 8 + envelope;
    let body = blob
        .get(wkb_start..)
        .ok_or_else(|| err("the envelope runs past the end of the blob"))?;
    if flags & 0x10 != 0 {
        return Ok(None);
    }
    let mut cursor = Cursor::new(body);
    read_geometry(&mut cursor, crs, 0)
}

/// A byte cursor over a WKB body, with the byte order of the geometry being
/// read.
struct Cursor<'a> {
    /// The remaining bytes of the body.
    bytes: &'a [u8],
    /// Read position within them.
    position: usize,
    /// Whether the current geometry stated little-endian byte order.
    little_endian: bool,
}

impl<'a> Cursor<'a> {
    /// A cursor at the start of `bytes`.
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            little_endian: true,
        }
    }

    /// How many bytes are left to read.
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    /// Reads one byte.
    fn u8(&mut self) -> Result<u8, LocalVectorError> {
        let byte = *self
            .bytes
            .get(self.position)
            .ok_or_else(|| err("the geometry ends mid-value"))?;
        self.position += 1;
        Ok(byte)
    }

    /// Reads a 32-bit unsigned integer in the current byte order.
    fn u32(&mut self) -> Result<u32, LocalVectorError> {
        let slice = self
            .bytes
            .get(self.position..self.position + 4)
            .ok_or_else(|| err("the geometry ends mid-value"))?;
        let raw = [slice[0], slice[1], slice[2], slice[3]];
        self.position += 4;
        Ok(if self.little_endian {
            u32::from_le_bytes(raw)
        } else {
            u32::from_be_bytes(raw)
        })
    }

    /// Reads a double in the current byte order.
    fn f64(&mut self) -> Result<f64, LocalVectorError> {
        let slice = self
            .bytes
            .get(self.position..self.position + 8)
            .ok_or_else(|| err("the geometry ends mid-value"))?;
        let mut raw = [0u8; 8];
        raw.copy_from_slice(slice);
        self.position += 8;
        Ok(if self.little_endian {
            f64::from_le_bytes(raw)
        } else {
            f64::from_be_bytes(raw)
        })
    }
}

/// The seven OGC base types, with the coordinate dimensions the type code asked
/// for.
struct WkbType {
    /// 1..=7: Point, LineString, Polygon, MultiPoint, MultiLineString,
    /// MultiPolygon, GeometryCollection.
    base: u32,
    /// Ordinates per position, 2..=4 — the extra ones are read and dropped.
    dimensions: usize,
}

/// Classifies a WKB type code.
///
/// Both spellings in the wild are accepted: ISO SQL/MM adds 1000 for Z, 2000 for
/// M and 3000 for ZM (what GDAL writes into a GeoPackage, hence the required
/// path), while PostGIS's EWKB sets high bits instead and may append a SRID.
fn classify(raw: u32) -> Result<(WkbType, bool), LocalVectorError> {
    let ewkb_z = raw & 0x8000_0000 != 0;
    let ewkb_m = raw & 0x4000_0000 != 0;
    let ewkb_srid = raw & 0x2000_0000 != 0;
    let code = raw & 0x0FFF_FFFF;
    let (base, iso_z, iso_m) = match code {
        1..=7 => (code, false, false),
        1001..=1007 => (code - 1000, true, false),
        2001..=2007 => (code - 2000, false, true),
        3001..=3007 => (code - 3000, true, true),
        other => return Err(err(format!("WKB type code {other} is not an OGC type"))),
    };
    let dimensions = 2 + usize::from(iso_z || ewkb_z) + usize::from(iso_m || ewkb_m);
    Ok((WkbType { base, dimensions }, ewkb_srid))
}

/// Reads one WKB geometry, including its own byte-order byte.
fn read_geometry(
    cursor: &mut Cursor<'_>,
    crs: &GpkgCrs,
    depth: u32,
) -> Result<Option<Geometry>, LocalVectorError> {
    if depth > MAX_DEPTH {
        return Err(err("the geometry nests deeper than any real dataset"));
    }
    cursor.little_endian = match cursor.u8()? {
        0 => false,
        1 => true,
        other => return Err(err(format!("{other} is not a WKB byte order"))),
    };
    let (kind, has_srid) = classify(cursor.u32()?)?;
    if has_srid {
        let _srid = cursor.u32()?;
    }
    match kind.base {
        1 => Ok(read_position(cursor, crs, kind.dimensions)?
            .and_then(|position| Point::new(position).ok())
            .map(Geometry::Point)),
        2 => {
            let points = read_sequence(cursor, crs, kind.dimensions)?;
            Ok(LineString::new(points).ok().map(Geometry::LineString))
        }
        3 => {
            let rings = read_rings(cursor, crs, kind.dimensions)?;
            Ok(rings
                .and_then(|rings| Polygon::new(rings).ok())
                .map(Geometry::Polygon))
        }
        4 => {
            let points = read_parts(cursor, crs, depth, 1)?
                .into_iter()
                .filter_map(|geometry| match geometry {
                    Geometry::Point(point) => Some(point.coordinates),
                    _ => None,
                })
                .collect::<Vec<Position>>();
            if points.is_empty() {
                return Ok(None);
            }
            Ok(MultiPoint::new(points).ok().map(Geometry::MultiPoint))
        }
        5 => {
            let lines = read_parts(cursor, crs, depth, 2)?
                .into_iter()
                .filter_map(|geometry| match geometry {
                    Geometry::LineString(line) => Some(line.coordinates),
                    _ => None,
                })
                .collect::<Vec<Vec<Position>>>();
            if lines.is_empty() {
                return Ok(None);
            }
            Ok(MultiLineString::new(lines)
                .ok()
                .map(Geometry::MultiLineString))
        }
        6 => {
            let polygons = read_parts(cursor, crs, depth, 3)?
                .into_iter()
                .filter_map(|geometry| match geometry {
                    Geometry::Polygon(polygon) => Some(polygon.coordinates),
                    _ => None,
                })
                .collect::<Vec<Vec<Vec<Position>>>>();
            if polygons.is_empty() {
                return Ok(None);
            }
            Ok(MultiPolygon::new(polygons).ok().map(Geometry::MultiPolygon))
        }
        _ => {
            let members = read_parts(cursor, crs, depth, 0)?;
            if members.is_empty() {
                return Ok(None);
            }
            Ok(GeometryCollection::new(members)
                .ok()
                .map(Geometry::GeometryCollection))
        }
    }
}

/// Reads the members of a `Multi*`/`GeometryCollection`, each of which restates
/// its own byte order and type.
///
/// `expected` is the base type every member must have, or 0 for a collection,
/// which may hold anything. A member of the wrong type is a malformed geometry,
/// not something to silently reinterpret. Empty members are dropped.
fn read_parts(
    cursor: &mut Cursor<'_>,
    crs: &GpkgCrs,
    depth: u32,
    expected: u32,
) -> Result<Vec<Geometry>, LocalVectorError> {
    let count = counted(cursor, 5)?;
    let mut parts = prealloc(count);
    for _ in 0..count {
        let saved = cursor.little_endian;
        let Some(member) = read_geometry(cursor, crs, depth + 1)? else {
            cursor.little_endian = saved;
            continue;
        };
        let actual = match &member {
            Geometry::Point(_) => 1,
            Geometry::LineString(_) => 2,
            Geometry::Polygon(_) => 3,
            Geometry::MultiPoint(_) => 4,
            Geometry::MultiLineString(_) => 5,
            Geometry::MultiPolygon(_) => 6,
            Geometry::GeometryCollection(_) => 7,
        };
        if expected != 0 && actual != expected {
            return Err(err(format!(
                "a multi-geometry of type {expected} holds a member of type {actual}"
            )));
        }
        cursor.little_endian = saved;
        parts.push(member);
    }
    Ok(parts)
}

/// Reads a count and checks it against what is left to read.
///
/// `unit` is the smallest number of bytes one counted element can occupy, so a
/// blob claiming four billion rings is refused before anything is allocated.
fn counted(cursor: &mut Cursor<'_>, unit: usize) -> Result<usize, LocalVectorError> {
    let count = usize::try_from(cursor.u32()?).unwrap_or(usize::MAX);
    if count > cursor.remaining() / unit {
        return Err(err(format!(
            "the geometry claims {count} elements but holds {} bytes",
            cursor.remaining()
        )));
    }
    Ok(count)
}

/// Reads one position, dropping any Z/M ordinates.
///
/// [`None`] means the position is not finite — `NaN` is how WKB spells an empty
/// point, and some producers write `-1e38` as a no-data marker.
fn read_position(
    cursor: &mut Cursor<'_>,
    crs: &GpkgCrs,
    dimensions: usize,
) -> Result<Option<Position>, LocalVectorError> {
    let x = cursor.f64()?;
    let y = cursor.f64()?;
    for _ in 2..dimensions {
        let _dropped = cursor.f64()?;
    }
    Ok(crs.lon_lat(x, y).map(|(lon, lat)| vec![lon, lat]))
}

/// Reads a counted run of positions.
fn read_sequence(
    cursor: &mut Cursor<'_>,
    crs: &GpkgCrs,
    dimensions: usize,
) -> Result<Vec<Position>, LocalVectorError> {
    let count = counted(cursor, dimensions * 8)?;
    let mut positions = prealloc(count);
    for _ in 0..count {
        if let Some(position) = read_position(cursor, crs, dimensions)? {
            positions.push(position);
        }
    }
    Ok(positions)
}

/// Reads a polygon's counted rings, or [`None`] when it has no usable exterior.
///
/// Ring roles in WKB are **positional**: ring 0 is the exterior and every later
/// ring is a hole. A ring with fewer than four positions cannot be closed —
/// either the blob encoded it short, or [`read_position`] dropped a non-finite
/// vertex out of it — and dropping such a ring is only safe when it is a hole.
/// Dropping ring 0 would slide the first surviving hole into the exterior slot,
/// and nothing downstream would notice: `Polygon::new` checks only that the
/// ring list is non-empty (ring order and closure live in a separate
/// `validate()` this reader does not call), and
/// `crate::local_vector::convert_polygon` takes `coordinates[0]` as the
/// exterior positionally and forces its winding to match. The feature would
/// render as a solid polygon covering what should have been its hole.
///
/// So an unusable exterior makes the whole polygon unusable. Every ring is
/// still *read*, whatever the verdict, because the cursor has to end up past
/// this polygon for the next member of a `MultiPolygon` to parse.
fn read_rings(
    cursor: &mut Cursor<'_>,
    crs: &GpkgCrs,
    dimensions: usize,
) -> Result<Option<Vec<Vec<Position>>>, LocalVectorError> {
    let count = counted(cursor, 4)?;
    let mut rings = prealloc(count);
    let mut exterior_usable = count > 0;
    for index in 0..count {
        let ring = read_sequence(cursor, crs, dimensions)?;
        let usable = ring.len() >= 4;
        if index == 0 && !usable {
            exterior_usable = false;
        }
        if usable {
            rings.push(ring);
        }
    }
    Ok(exterior_usable.then_some(rings))
}
