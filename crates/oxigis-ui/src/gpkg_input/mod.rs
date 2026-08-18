// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! OGC GeoPackage (`.gpkg`) → GeoJSON [`FeatureCollection`]s, from **bytes
//! only**.
//!
//! Phase 1 §1.3's second format, and the first that is *several* layers in one
//! file: a GeoPackage is a SQLite database, and every table `gpkg_contents`
//! lists as `features` becomes its own OxiGIS layer. Everything downstream of
//! this module already works on any
//! [`oxigeo::geojson::types::FeatureCollection`] (see [`crate::local_vector`]),
//! so a GeoPackage only has to *become* some.
//!
//! # Why the SQLite reader is ours
//!
//! `oxigeo-gpkg` 0.2.2 was evaluated as the dependency for this and rejected on
//! three counts, in descending order of severity:
//!
//! 1. **Its b-tree reader does not follow overflow-page chains.** Any row whose
//!    record exceeds `usable_size - 35` bytes — ≈ 4061 B at the 4096-byte page
//!    size GDAL writes by default, i.e. a polygon of a few hundred vertices —
//!    fails, and the failure takes the whole table with it. This was reproduced
//!    against the crate, not inferred.
//! 2. Its GeoJSON conversion targets `oxigeo-geojson-stream`'s types, a
//!    different object model from the `oxigeo-geojson` one this crate speaks,
//!    so the conversion would have to be hand-written regardless — the same
//!    conclusion [`crate::shapefile_input`] reached about its own upstream.
//! 3. It pulls `oxigeo-geojson-stream` and the `regex` family into the wasm
//!    bundle for that unusable conversion.
//!
//! So the SQLite file format is implemented here, in this module's `sqlite`
//! submodule (crate-private — nothing outside the reader has any business
//! walking b-trees), with the
//! overflow chain as its first-class case, and this module is only the
//! GeoPackage layer on top of it: three metadata tables, a CRS policy, and the
//! row → [`Feature`] mapping. Zero new dependencies.
//!
//! # One file, many layers, partial success
//!
//! A `.gpkg` regularly mixes CRSs across its tables, so "refuse the file"
//! would be the wrong granularity: a table this crate cannot place is refused
//! **on its own**, with a notice naming the reason, and the file's other tables
//! still load. [`GpkgDataset`] therefore carries both halves — the tables that
//! became layers and, as [`GpkgRefusal`]s, the table-name-plus-reason pairs for
//! the ones that did not.
//!
//! Tiles and attributes tables are skipped in silence: they are not vector
//! layers, and a raster tile pyramid inside a `.gpkg` is a separate surface
//! this module does not touch.
//!
//! # CRS
//!
//! The same three outcomes [`crate::shapefile_input::sniff_prj`] has, decided
//! from `gpkg_spatial_ref_sys` instead of a `.prj`: a table's `srs_id` is
//! resolved to an [`oxigis_core::Crs`] — by `organization` +
//! `organization_coordsys_id` first, then by the row's WKT `definition`, and
//! only for an id the file does not register at all is the id itself read as
//! an EPSG code. The "undefined" ids 0 and -1 are taken as WGS 84, matching the
//! shapefile rule that a missing `.prj` passes. Everything `oxigis-core` can
//! place is **reprojected per vertex**; anything else is refused *per table*,
//! with the CRS named, so the file's other tables still load. Drawing a
//! Lambert-93 dataset as if it were degrees would put France in the Gulf of
//! Guinea; an honest refusal beats that.
//!
//! Reading `definition` matters more here than it looks: `srs_id` is a
//! file-local primary key, not an EPSG code, and a GeoPackage written by a tool
//! that renumbered its SRS table (or that records `organization = 'NONE'`)
//! carries its only real CRS information in that WKT column.

pub mod geometry;
pub(crate) mod sqlite;

// Follows `mbtiles::fixture` across the `fixtures` feature seam because that
// module builds its images on this one's `{Cell, Image, record, varint}`. The
// `.gpkg` blobs it `include_bytes!`es keep their own `#[cfg(test)]`, so the
// feature build carries the builders without the binaries.
#[cfg(any(test, feature = "fixtures"))]
pub(crate) mod fixture;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_hostile;
#[cfg(test)]
mod tests_payload;

use std::collections::BTreeMap;

use oxigeo::geojson::types::{Feature, FeatureCollection, Properties};
use oxigis_core::Crs;
use serde_json::Value;

use crate::local_vector::LocalVectorError;
use sqlite::{CellValue, MasterEntry, Row, SqliteDb, TableSchema};

pub use geometry::GpkgCrs;

/// One feature table of a GeoPackage, already converted.
#[derive(Debug, Clone)]
pub struct GpkgTable {
    /// The table's name inside the file — what a layer is named after and what
    /// [`oxigis_core::VectorSource::LocalGpkg`] records so a reload finds it
    /// again.
    pub name: String,
    /// Its rows as GeoJSON features, in rowid order, with coordinates already
    /// in WGS 84 lon/lat.
    pub features: FeatureCollection,
    /// The CRS the table was registered in — provenance the layer keeps (see
    /// [`oxigis_core::Layer::crs`]); `features` is WGS 84 regardless.
    pub crs: Crs,
}

/// One feature table that was left out, and why.
///
/// The table name is kept **beside** the message rather than only inside it.
/// A caller reloading one named layer has to ask "was *this* table refused?",
/// and answering that by searching the message text is a substring test: a
/// project whose layer reads the table `roads` would happily pick up the
/// refusal of a table called `roads_old` and report its reason as its own.
#[derive(Debug, Clone)]
pub struct GpkgRefusal {
    /// The table's name, exactly as `gpkg_contents` spells it — the same
    /// string [`GpkgTable::name`] would have carried had it loaded.
    table: String,
    /// The human-readable line for the status bar.
    message: String,
}

impl GpkgRefusal {
    /// The name of the table that was left out.
    #[must_use]
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Why it was left out, phrased for the status line.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Every feature table of one GeoPackage, plus why the others were left out.
#[derive(Debug, Clone)]
pub struct GpkgDataset {
    /// The tables that became layers, in `gpkg_contents` order.
    tables: Vec<GpkgTable>,
    /// One entry per feature table that could not be imported.
    refusals: Vec<GpkgRefusal>,
}

impl GpkgDataset {
    /// The tables that became layers.
    #[must_use]
    pub fn tables(&self) -> &[GpkgTable] {
        &self.tables
    }

    /// Why the feature tables that are *not* in [`Self::tables`] were left out
    /// — an unsupported CRS, an unreadable b-tree, an empty table. An empty
    /// slice means everything loaded.
    #[must_use]
    pub fn refusals(&self) -> &[GpkgRefusal] {
        &self.refusals
    }

    /// The same refusals as bare status-line text, in order.
    #[must_use]
    pub fn notices(&self) -> Vec<String> {
        self.refusals
            .iter()
            .map(|refusal| refusal.message.clone())
            .collect()
    }

    /// The table named `name`, if it was imported.
    #[must_use]
    pub fn table(&self, name: &str) -> Option<&GpkgTable> {
        self.tables.iter().find(|table| table.name == name)
    }

    /// Consumes the dataset, yielding its tables and its refusals.
    #[must_use]
    pub fn into_parts(self) -> (Vec<GpkgTable>, Vec<GpkgRefusal>) {
        (self.tables, self.refusals)
    }
}

/// Reads every feature table of a GeoPackage image.
///
/// The bytes are the whole `.gpkg` file. A `-wal` sidecar is **not** applied in
/// this version: a database with uncommitted rows in its write-ahead log reads
/// as it stood at the last checkpoint, which for a file being dropped onto a
/// map is what is on disk. (Deferred, not overlooked — and not silent: a file
/// whose header names write-ahead-log mode gets a notice ahead of its tables',
/// see the `saw_feature_table` gate below.)
///
/// Succeeding here means "this is a GeoPackage and its catalogue was read" —
/// not that any table became a layer. A file whose every feature table was
/// refused returns [`Ok`] with no tables and one notice each; deciding what to
/// do about that belongs to the caller, which is the only place that knows
/// whether "nothing was added" is worth an error (see
/// [`crate::local_input::LocalInputState::add_gpkg`]).
///
/// # Errors
///
/// Returns a [`LocalVectorError`] when the bytes are not a readable SQLite
/// image (bad magic, an illegal page size, UTF-16 text, a truncated or cyclic
/// b-tree) or when they hold no `gpkg_contents` table, which is what makes a
/// SQLite database a GeoPackage.
pub fn from_bytes(gpkg: &[u8]) -> Result<GpkgDataset, LocalVectorError> {
    let db = SqliteDb::open(gpkg)?;
    let master = db.master_entries()?;
    let (schema, contents) =
        read_metadata_table(&db, &master, "gpkg_contents")?.ok_or_else(|| {
            LocalVectorError::new(
                "the file is a SQLite database but not a GeoPackage: it has no gpkg_contents table",
            )
        })?;
    let geometry_columns = geometry_columns(&db, &master)?;
    let spatial_ref_sys = spatial_ref_sys(&db, &master)?;

    let table_name = schema.column_index("table_name").unwrap_or(0);
    let data_type = schema.column_index("data_type").unwrap_or(1);
    let mut tables = Vec::new();
    let mut refusals = Vec::new();
    // Whether `gpkg_contents` names at least one feature table, regardless of
    // whether it ends up loading: gates the WAL notice below so a tiles-only
    // GeoPackage keeps its exact "holds no feature tables" message instead of
    // a caveat about rows that, this file having none, cannot be missing.
    let mut saw_feature_table = false;
    for row in &contents {
        let (Some(name), Some(kind)) = (text_at(row, table_name), text_at(row, data_type)) else {
            continue;
        };
        // Tiles and attributes are not vector layers; skipping them is not a
        // failure and must not produce a notice.
        if !kind.eq_ignore_ascii_case("features") {
            continue;
        }
        saw_feature_table = true;
        match import_table(&db, &master, &name, &geometry_columns, &spatial_ref_sys) {
            Ok(table) => tables.push(table),
            Err(message) => refusals.push(GpkgRefusal {
                table: name,
                message,
            }),
        }
    }
    if saw_feature_table && db.wal_mode() {
        refusals.insert(
            0,
            GpkgRefusal {
                table: String::new(),
                message: "this GeoPackage is in write-ahead-log mode; rows written since its \
                          last checkpoint are in a -wal sidecar that OxiGIS does not read, so a \
                          layer above may be incomplete"
                    .to_string(),
            },
        );
    }
    Ok(GpkgDataset { tables, refusals })
}

/// Serialises a collection back to compact GeoJSON text.
///
/// The web persistence leg: GeoPackage bytes cannot be embedded in a project
/// document, so a browser-dropped table is stored as
/// [`oxigis_core::VectorSource::InlineGeoJson`] instead — the same bridge
/// [`crate::shapefile_input::to_geojson_string`] provides for a shapefile.
///
/// # Errors
///
/// Returns a [`LocalVectorError`] if the collection cannot be serialised.
pub fn to_geojson_string(features: &FeatureCollection) -> Result<String, LocalVectorError> {
    oxigeo::geojson::writer::to_string(features)
        .map_err(|error| LocalVectorError::new(format!("GeoJSON serialization failed: {error}")))
}

/// What `gpkg_geometry_columns` registered for one feature table.
#[derive(Debug, Clone)]
struct GeometryColumn {
    /// Name of the BLOB column holding the geometries.
    column: String,
    /// The SRS every geometry in it is stored in, or [`None`] when the
    /// registration's `srs_id` cell did not decode as an integer.
    ///
    /// [`None`] is a *refusal*, never a default. The column is declared
    /// `INTEGER NOT NULL` by the GeoPackage spec, and SQLite's own affinity
    /// rules coerce anything storable into an integer on the way in, so a cell
    /// of some other storage class means the file was assembled by something
    /// that is not SQLite and the table's CRS is simply unknown. Falling back
    /// to id 0 — which [`resolve_srs`] reads as "undefined", i.e. WGS 84 —
    /// would draw a Web Mercator table's metres as degrees with nothing on
    /// screen to say why the layer is in the wrong hemisphere.
    srs_id: Option<i64>,
}

/// One `gpkg_spatial_ref_sys` row, reduced to what the CRS policy reads.
#[derive(Debug, Clone)]
struct SrsRow {
    /// Human-readable name, used to name the CRS in a refusal.
    name: String,
    /// Authority that defined it — `EPSG` for anything resolved by code.
    organization: String,
    /// That authority's code for it, which for EPSG is the number everyone
    /// means when they say "4326".
    code: i64,
    /// The row's WKT `definition`, the fallback when `organization` names an
    /// authority this crate does not speak (or names none at all).
    ///
    /// A GeoPackage written by a tool that renumbered its SRS table, or one
    /// that records `organization = 'NONE'` while still carrying a perfectly
    /// good WKT string, is common enough that reading the definition is worth
    /// the one extra column: `srs_id` is a *file-local key*, not an EPSG code.
    definition: String,
}

/// Reads one metadata table whole, with the schema its `CREATE TABLE` declares.
///
/// [`None`] means the table is not in the file at all — legal for
/// `gpkg_geometry_columns` and `gpkg_spatial_ref_sys` in a tiles-only
/// GeoPackage, and handled by the callers rather than here.
fn read_metadata_table(
    db: &SqliteDb<'_>,
    master: &[MasterEntry],
    name: &str,
) -> Result<Option<(TableSchema, Vec<Row>)>, LocalVectorError> {
    let Some(entry) = master
        .iter()
        .find(|entry| entry.entry_type == "table" && entry.name.eq_ignore_ascii_case(name))
    else {
        return Ok(None);
    };
    let schema = sqlite::parse_create_table(&entry.sql).ok_or_else(|| {
        LocalVectorError::new(format!("the {name} table's definition could not be read"))
    })?;
    let mut rows = db.scan_table(entry.rootpage)?;
    for row in &mut rows {
        normalize(&schema, row).map_err(|reason| {
            LocalVectorError::new(format!("the {name} table could not be read: {reason}"))
        })?;
    }
    Ok(Some((schema, rows)))
}

/// Builds the table name → geometry column map from `gpkg_geometry_columns`.
fn geometry_columns(
    db: &SqliteDb<'_>,
    master: &[MasterEntry],
) -> Result<BTreeMap<String, GeometryColumn>, LocalVectorError> {
    let mut columns = BTreeMap::new();
    let Some((schema, rows)) = read_metadata_table(db, master, "gpkg_geometry_columns")? else {
        return Ok(columns);
    };
    // Resolved by name, falling back to the spec's column order, so a file
    // written against a different GeoPackage edition still reads.
    let table_name = schema.column_index("table_name").unwrap_or(0);
    let column_name = schema.column_index("column_name").unwrap_or(1);
    let srs_id = schema.column_index("srs_id").unwrap_or(3);
    for row in &rows {
        let (Some(table), Some(column)) = (text_at(row, table_name), text_at(row, column_name))
        else {
            continue;
        };
        columns.insert(
            table.to_ascii_lowercase(),
            GeometryColumn {
                column,
                srs_id: int_at(row, srs_id),
            },
        );
    }
    Ok(columns)
}

/// Builds the srs_id → definition map from `gpkg_spatial_ref_sys`.
fn spatial_ref_sys(
    db: &SqliteDb<'_>,
    master: &[MasterEntry],
) -> Result<BTreeMap<i64, SrsRow>, LocalVectorError> {
    let mut rows_by_id = BTreeMap::new();
    let Some((schema, rows)) = read_metadata_table(db, master, "gpkg_spatial_ref_sys")? else {
        return Ok(rows_by_id);
    };
    // GeoPackage 1.3 appends a seventh column (`definition_12_063`); resolving
    // by name means it is simply never read, rather than shifting anything.
    let srs_name = schema.column_index("srs_name").unwrap_or(0);
    let srs_id = schema.column_index("srs_id").unwrap_or(1);
    let organization = schema.column_index("organization").unwrap_or(2);
    let coordsys_id = schema.column_index("organization_coordsys_id").unwrap_or(3);
    // Resolved by name only — there is no safe positional fallback for it (the
    // index a positional guess would use is already claimed above), so a file
    // whose schema does not name the column simply has no WKT to fall back on.
    let definition = schema.column_index("definition");
    for row in &rows {
        let Some(id) = int_at(row, srs_id) else {
            continue;
        };
        // `unwrap_or(id)` here is deliberate and stays, unlike
        // [`GeometryColumn::srs_id`]'s: it degrades to the same "the id *is*
        // the authority code" convention [`resolve_srs`]'s `None` arm already
        // applies to an unregistered id, rather than to a sentinel that means
        // "assume WGS 84". It is also the one index above that may have come
        // from the *positional* fallback (a GeoPackage edition that names the
        // column differently would put `definition`, a TEXT column, at index
        // 3), so treating a non-integer cell as fatal here would refuse files
        // that load correctly today. The same reasoning covers `gpkg_contents`:
        // its only cells this module reads are `table_name` and `data_type`,
        // both TEXT, and a row whose `data_type` does not decode is skipped
        // rather than assumed to be `features` — no CRS is inferred anywhere in
        // that path.
        rows_by_id.insert(
            id,
            SrsRow {
                name: text_at(row, srs_name).unwrap_or_default(),
                organization: text_at(row, organization).unwrap_or_default(),
                code: int_at(row, coordsys_id).unwrap_or(id),
                definition: definition
                    .and_then(|index| text_at(row, index))
                    .unwrap_or_default(),
            },
        );
    }
    Ok(rows_by_id)
}

/// Decides what a table's `srs_id` means.
///
/// The authority pair is what is checked first, not the bare id: `srs_id` is a
/// **file-local key**, and a GeoPackage may legitimately number a custom CRS
/// 4326. Resolution order, strongest first:
///
/// 1. `organization = 'EPSG'` → `organization_coordsys_id` is the EPSG code;
/// 2. the row's WKT `definition`, read by [`oxigis_core::Crs::from_wkt`] — this
///    is what rescues a file whose `organization` is `NONE` or a vendor string
///    but whose definition is a perfectly ordinary `PROJCS[…]`;
/// 3. only when the file has no row for the id at all does the id itself get
///    read as an EPSG code, which is the convention every writer follows.
///
/// The "undefined" ids 0 and -1 are the GeoPackage spelling of "no CRS was
/// recorded", which the shapefile path already treats as WGS 84.
///
/// Never fails: an unresolvable CRS comes back with
/// [`oxigis_core::EPSG_UNKNOWN`] and whatever name the file gave, so the caller
/// refuses it by name.
fn resolve_srs(srs_id: i64, rows: &BTreeMap<i64, SrsRow>) -> Crs {
    // A synthetic one-line WKT carries the file's own `srs_name` into
    // `Crs::name()` without claiming the row held WKT: `LOCAL_CS` is the one
    // keyword that resolves to no CRS at all, so the value cannot be mistaken
    // for a placeable definition — it exists so the refusal quotes the name the
    // FILE gave, next to whatever code it declared.
    let named = |row: Option<&SrsRow>, epsg: u32| {
        let label = srs_label(srs_id, row).replace('"', "'");
        Crs::new(epsg, Some(&format!("LOCAL_CS[\"{label}\"]")))
    };
    let undefined = matches!(srs_id, 0 | -1);
    match rows.get(&srs_id) {
        Some(row) if row.organization.eq_ignore_ascii_case("EPSG") => {
            match u32::try_from(row.code) {
                Ok(code) if oxigis_core::crs::is_supported(code) => Crs::from_epsg(code),
                // The row named an EPSG code this build cannot place: try its
                // WKT, and failing that keep BOTH the code and the file's name
                // so the refusal is actionable.
                Ok(code) => from_definition(row).unwrap_or_else(|| named(Some(row), code)),
                Err(_) => from_definition(row)
                    .unwrap_or_else(|| named(Some(row), oxigis_core::EPSG_UNKNOWN)),
            }
        }
        Some(row) => from_definition(row).unwrap_or_else(|| {
            if undefined {
                Crs::wgs84()
            } else {
                named(Some(row), oxigis_core::EPSG_UNKNOWN)
            }
        }),
        None if undefined => Crs::wgs84(),
        // No row for the id at all: the id *is* the authority code, the
        // convention every writer follows. When that code is one this build
        // cannot place, the only name available is the numeric one — which is
        // still what the refusal has to quote, so it is carried along.
        None => match u32::try_from(srs_id) {
            Ok(code) if oxigis_core::crs::is_supported(code) => Crs::from_epsg(code),
            Ok(code) => named(None, code),
            Err(_) => named(None, oxigis_core::EPSG_UNKNOWN),
        },
    }
}

/// The CRS a row's WKT `definition` declares, when it declares one this build
/// can place.
///
/// `"undefined"` is what the GeoPackage specification's own two mandatory rows
/// put in the column for srs_id 0 and -1, so it is skipped rather than parsed.
fn from_definition(row: &SrsRow) -> Option<Crs> {
    let text = row.definition.trim();
    if text.is_empty() || text.eq_ignore_ascii_case("undefined") {
        return None;
    }
    let crs = Crs::from_wkt(text);
    crs.is_supported().then_some(crs)
}

/// How a CRS is named in a refusal: its `srs_name` when the file gave one,
/// otherwise its numeric id.
fn srs_label(srs_id: i64, row: Option<&SrsRow>) -> String {
    match row
        .map(|row| row.name.trim())
        .filter(|name| !name.is_empty())
    {
        Some(name) => name.to_string(),
        None => format!("SRS {srs_id}"),
    }
}

/// Imports one feature table, or explains why it could not be.
///
/// Every failure is a *notice*, never an error: one unreadable table in a
/// ten-table file must not cost the other nine.
fn import_table(
    db: &SqliteDb<'_>,
    master: &[MasterEntry],
    name: &str,
    geometry_columns: &BTreeMap<String, GeometryColumn>,
    spatial_ref_sys: &BTreeMap<i64, SrsRow>,
) -> Result<GpkgTable, String> {
    let refused = |reason: &str| format!("The GeoPackage table \u{201c}{name}\u{201d} {reason}");
    let geometry_column = geometry_columns
        .get(&name.to_ascii_lowercase())
        .ok_or_else(|| refused("has no gpkg_geometry_columns entry, so it holds no geometry."))?;
    let srs_id = geometry_column.srs_id.ok_or_else(|| {
        refused(
            "has a gpkg_geometry_columns entry whose srs_id is not an integer, so the CRS its \
             coordinates are in is unknown.",
        )
    })?;
    let source_crs = resolve_srs(srs_id, spatial_ref_sys);
    let crs = GpkgCrs::for_crs(&source_crs).ok_or_else(|| {
        format!(
            "The GeoPackage table \u{201c}{name}\u{201d} is in \u{201c}{}\u{201d}, which OxiGIS \
             cannot place; reproject it to WGS 84 (EPSG:4326) first.",
            source_crs.label(),
        )
    })?;
    // Case-insensitively, like every other identifier lookup here: SQLite
    // identifiers are, so `gpkg_contents` may legally name a table in a case
    // its `CREATE TABLE` did not use.
    let entry = master
        .iter()
        .find(|entry| entry.entry_type == "table" && entry.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| refused("is listed in gpkg_contents but does not exist."))?;
    let schema = sqlite::parse_create_table(&entry.sql)
        .ok_or_else(|| refused("has a definition this reader cannot parse."))?;
    let geometry_index = schema
        .column_index(&geometry_column.column)
        .ok_or_else(|| {
            refused(&format!(
                "has no column named \u{201c}{}\u{201d}.",
                geometry_column.column
            ))
        })?;
    let mut rows = db
        .scan_table(entry.rootpage)
        .map_err(|error| refused(&format!("could not be read: {}.", error.message())))?;

    let mut features = Vec::with_capacity(rows.len());
    for row in &mut rows {
        normalize(&schema, row)
            .map_err(|reason| refused(&format!("could not be read: {reason}.")))?;
        let geometry = match row.values.get(geometry_index) {
            Some(CellValue::Blob(blob)) => geometry::decode(blob, &crs).map_err(|error| {
                refused(&format!(
                    "has a geometry in row {} this reader cannot decode: {}.",
                    row.rowid,
                    error.message(),
                ))
            })?,
            // A NULL geometry is legal, and anything else in the column is
            // treated as one so the attribute row still shows up.
            _ => None,
        };
        features.push(Feature::new(
            geometry,
            Some(properties(&schema, row, geometry_index)),
        ));
    }
    if features.is_empty() {
        return Err(refused("holds no rows."));
    }
    Ok(GpkgTable {
        name: name.to_string(),
        features: FeatureCollection::new(features),
        crs: source_crs,
    })
}

/// Makes a row's values line up with its table's columns.
///
/// Two fixes, both invisible in the record itself: a row written before an
/// `ALTER TABLE ADD COLUMN` is short and its missing tail reads as `NULL`, and
/// an `INTEGER PRIMARY KEY` column stores `NULL` because its real value is the
/// cell's rowid.
///
/// # Errors
///
/// Refuses a row that decodes to **more** values than the table declares
/// columns. Padding a short record is a documented SQLite behaviour; the
/// opposite direction is not producible by SQLite at all, so it can only mean
/// the `CREATE TABLE` statement was parsed into fewer columns than it really
/// has. Resizing down would truncate the record's *tail* rather than remove the
/// column that went missing, silently sliding every value onto the wrong column
/// — the geometry BLOB included, which then reads as no geometry at all. A
/// refusal costs one table; the resize costs a whole layer of wrong data with
/// nothing on screen to say so.
fn normalize(schema: &TableSchema, row: &mut Row) -> Result<(), String> {
    if row.values.len() > schema.columns.len() {
        return Err(format!(
            "row {} holds {} values, more than its table's definition declares ({})",
            row.rowid,
            row.values.len(),
            schema.columns.len(),
        ));
    }
    row.values.resize(schema.columns.len(), CellValue::Null);
    if let Some(alias) = schema.rowid_alias
        && matches!(row.values.get(alias), Some(CellValue::Null))
    {
        row.values[alias] = CellValue::Integer(row.rowid);
    }
    Ok(())
}

/// Maps one row's non-geometry columns onto GeoJSON properties, in column
/// order.
fn properties(schema: &TableSchema, row: &Row, geometry_index: usize) -> Properties {
    let mut properties = Properties::new();
    for (index, column) in schema.columns.iter().enumerate() {
        if index == geometry_index {
            continue;
        }
        let value = row.values.get(index).map_or(Value::Null, |cell| {
            cell_to_json(cell, column.has_real_affinity())
        });
        properties.insert(column.name.clone(), value);
    }
    properties
}

/// Maps one cell onto its JSON counterpart.
///
/// `real` is the column's REAL affinity, which is what restores the type of a
/// `40.0` SQLite stored as an integer — see
/// [`sqlite::ColumnDef::has_real_affinity`].
///
/// A non-finite float becomes `null`, because JSON has no encoding for one. A
/// BLOB in a non-geometry column does **not**: rendering the raw bytes as hex
/// would fill an attribute-table cell with noise, but the byte count is a
/// truthful, compact stand-in that tells the cell apart from a real `NULL` —
/// GeoPackage extensions and ordinary FME/ArcGIS exports both put photos and
/// signature blobs in feature tables often enough that dropping the column
/// silently is the wrong default.
fn cell_to_json(value: &CellValue, real: bool) -> Value {
    match value {
        CellValue::Null => Value::Null,
        CellValue::Integer(number) if real => json_number(*number as f64),
        CellValue::Integer(number) => Value::from(*number),
        CellValue::Float(number) => json_number(*number),
        CellValue::Text(text) => Value::String(text.clone()),
        CellValue::Blob(bytes) => Value::String(format!("<{} bytes>", bytes.len())),
    }
}

/// A JSON number, or `null` for a value JSON cannot spell.
fn json_number(value: f64) -> Value {
    serde_json::Number::from_f64(value).map_or(Value::Null, Value::Number)
}

/// The text of one column of a row, if it holds text.
fn text_at(row: &Row, index: usize) -> Option<String> {
    match row.values.get(index) {
        Some(CellValue::Text(text)) => Some(text.clone()),
        _ => None,
    }
}

/// The integer of one column of a row, if it holds one.
fn int_at(row: &Row, index: usize) -> Option<i64> {
    match row.values.get(index) {
        Some(CellValue::Integer(number)) => Some(*number),
        _ => None,
    }
}
