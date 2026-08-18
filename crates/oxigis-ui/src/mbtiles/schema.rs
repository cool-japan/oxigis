// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Recognising an MBTiles archive's schema, and reading its `metadata` table.
//!
//! Two shapes are in circulation and both must work, because between them they
//! are every `.mbtiles` a user will ever have:
//!
//! * **flat** — a `tiles` *table* with `(zoom_level, tile_column, tile_row,
//!   tile_data)`. What the specification describes, and what a hand-rolled
//!   writer produces.
//! * **normalized** — `tiles` is a *view* over `map` ⋈ `images`, so identical
//!   tiles (an ocean, a blank hillshade) are stored once. What tippecanoe and
//!   mbutil write, i.e. most archives in the wild.
//!
//! [`crate::gpkg_input::sqlite::SqliteDb::master_entries`] reports a view with
//! `rootpage == 0`, so a reader that only looks for a `tiles` root silently
//! finds nothing on the *common* case. That is why the shape is detected rather
//! than assumed, and why an unrecognised one is refused with its own sentence.
//!
//! Column positions come from
//! [`crate::gpkg_input::sqlite::parse_create_table`], never from the
//! specification's column order: a writer is free to declare them in any order,
//! and reading `tile_row` out of the `tile_column` slot draws a plausible map of
//! the wrong place.

use std::collections::BTreeMap;

use crate::archive::{ArchiveContent, ArchiveInfo};
use crate::gpkg_input::sqlite::{
    CellValue, SqliteDb, TableSchema, decode_record_prefix, parse_create_table,
};
use crate::local_vector::LocalVectorError;

use super::{MAX_MBTILES_TILES, content_for_format, index_key, xyz_row};

/// Which container an MBTiles archive's tiles are stored in.
///
/// Public because a status line naming the schema is the difference between
/// "this file did not work" and "this file is a shape OxiGIS does not read".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbTilesFormat {
    /// A `tiles` table holding the blobs directly.
    Flat,
    /// A `map` table of addresses joined to an `images` table of blobs.
    Normalized,
}

impl MbTilesFormat {
    /// A name for a status line.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Flat => "flat (a tiles table)",
            Self::Normalized => "normalized (map joined to images)",
        }
    }
}

/// Where the columns an MBTiles reader needs live, in one table.
///
/// Visible to the whole `mbtiles` module because **both** readers resolve
/// columns the same way — by name, out of the `CREATE TABLE` statement, never by
/// the specification's column order. The paged reader in
/// [`crate::mbtiles::paged`] reuses this type rather than growing a second copy
/// that could drift.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Columns {
    /// Index of `zoom_level`.
    pub(crate) zoom: usize,
    /// Index of `tile_column`.
    pub(crate) column: usize,
    /// Index of `tile_row`.
    pub(crate) row: usize,
    /// Index of the payload column: `tile_data` (flat) or `tile_id`
    /// (normalized).
    pub(crate) payload: usize,
}

impl Columns {
    /// Resolves the four columns by name in `schema`.
    pub(crate) fn resolve(
        schema: &TableSchema,
        table: &str,
        payload_name: &str,
    ) -> Result<Self, LocalVectorError> {
        let find = |name: &str| {
            schema.column_index(name).ok_or_else(|| {
                LocalVectorError::new(format!(
                    "the archive's {table} table has no {name} column, so it is not MBTiles"
                ))
            })
        };
        Ok(Self {
            zoom: find("zoom_level")?,
            column: find("tile_column")?,
            row: find("tile_row")?,
            payload: find(payload_name)?,
        })
    }

    /// The highest of the three address column indices.
    ///
    /// What a *flat* archive needs out of a record's inline prefix: the blob
    /// itself is reached by rowid, never read from the prefix, and requiring it
    /// there would silently drop every spilled tile from the index.
    const fn address_widest(self) -> usize {
        let mut widest = self.zoom;
        if self.column > widest {
            widest = self.column;
        }
        if self.row > widest {
            widest = self.row;
        }
        widest
    }

    /// The highest column index a *normalized* archive needs from the prefix —
    /// the three addresses plus the `tile_id` that names the body.
    const fn widest(self) -> usize {
        let address = self.address_widest();
        if self.payload > address {
            self.payload
        } else {
            address
        }
    }
}

/// How this archive stores its tiles, with the roots and columns to read them.
#[derive(Debug)]
pub(super) struct Layout {
    /// Which of the two shapes it is.
    format: MbTilesFormat,
    /// Root page of `tiles` (flat) or `map` (normalized).
    address_root: u32,
    /// Columns of that table.
    address_columns: Columns,
    /// Root page of `images`, for the normalized shape.
    images_root: u32,
    /// Index of `images.tile_id`, for the normalized shape.
    images_id: usize,
    /// Index of `images.tile_data`, for the normalized shape.
    images_data: usize,
    /// `tile_id` → `images` rowid, for the normalized shape.
    image_rowids: BTreeMap<String, i64>,
}

impl Layout {
    /// Which shape `db` is, refusing anything else by name.
    ///
    /// # Errors
    ///
    /// Refuses an archive with neither a `tiles` table nor a `map`/`images`
    /// pair, and one whose `CREATE TABLE` statements do not declare the columns
    /// MBTiles requires.
    pub(super) fn detect(db: &SqliteDb<'_>) -> Result<Self, LocalVectorError> {
        // The auto-index-preserving catalogue, so BOTH readers see the same
        // `sqlite_master` — this reader ignores indices, but a catalogue that
        // silently drops rows is the kind of asymmetry that hides bugs.
        let entries = db.master_entries_with_autoindex()?;
        let table = |name: &str| {
            entries.iter().find(|entry| {
                entry.entry_type == "table"
                    && entry.name.eq_ignore_ascii_case(name)
                    && entry.rootpage > 0
            })
        };
        if let Some(tiles) = table("tiles") {
            let schema = parse_create_table(&tiles.sql).ok_or_else(|| {
                LocalVectorError::new(
                    "the archive's tiles table does not declare its columns, so it cannot be read",
                )
            })?;
            return Ok(Self {
                format: MbTilesFormat::Flat,
                address_root: tiles.rootpage,
                address_columns: Columns::resolve(&schema, "tiles", "tile_data")?,
                images_root: 0,
                images_id: 0,
                images_data: 0,
                image_rowids: BTreeMap::new(),
            });
        }
        let (Some(map), Some(images)) = (table("map"), table("images")) else {
            return Err(LocalVectorError::new(
                "the archive has neither a tiles table nor the map/images pair MBTiles uses, \
                 so it is not an MBTiles archive OxiGIS can read",
            ));
        };
        let map_schema = parse_create_table(&map.sql).ok_or_else(|| {
            LocalVectorError::new(
                "the archive's map table does not declare its columns, so it cannot be read",
            )
        })?;
        let images_schema = parse_create_table(&images.sql).ok_or_else(|| {
            LocalVectorError::new(
                "the archive's images table does not declare its columns, so it cannot be read",
            )
        })?;
        let images_id = images_schema.column_index("tile_id").ok_or_else(|| {
            LocalVectorError::new("the archive's images table has no tile_id column")
        })?;
        let images_data = images_schema.column_index("tile_data").ok_or_else(|| {
            LocalVectorError::new("the archive's images table has no tile_data column")
        })?;
        let mut layout = Self {
            format: MbTilesFormat::Normalized,
            address_root: map.rootpage,
            address_columns: Columns::resolve(&map_schema, "map", "tile_id")?,
            images_root: images.rootpage,
            images_id,
            images_data,
            image_rowids: BTreeMap::new(),
        };
        layout.index_images(db)?;
        Ok(layout)
    }

    /// Which shape this is.
    pub(super) const fn format(&self) -> MbTilesFormat {
        self.format
    }

    /// Builds `tile_id` → rowid for the normalized shape.
    ///
    /// A prefix scan is enough *only* when `tile_id` sits before `tile_data` in
    /// the record, which is not guaranteed; when it does not, [`SqliteDb::scan_column`]
    /// reads just that one column — never the tile bodies between it and the
    /// header — so this is one bounded-memory pass either way. Every writer in
    /// circulation declares `tile_id` last, so the slow path is what actually
    /// runs.
    ///
    /// # Errors
    ///
    /// Refuses a pyramid past [`MAX_MBTILES_TILES`], the same named refusal
    /// [`Self::build_index`] gives — `image_rowids` is built from the same
    /// archive and cannot legitimately hold more entries than that.
    fn index_images(&mut self, db: &SqliteDb<'_>) -> Result<(), LocalVectorError> {
        let id_column = self.images_id;
        let data_column = self.images_data;
        let prefix_is_enough = id_column < data_column;
        let mut rowids = BTreeMap::new();
        let too_many = || {
            LocalVectorError::new(format!(
                "the archive holds more than {MAX_MBTILES_TILES} tiles, past what OxiGIS \
                 indexes in memory; open it from disk or from a URL instead, which \
                 streams it a page at a time"
            ))
        };
        if prefix_is_enough {
            let mut visit = |rowid: i64, prefix: &[u8]| -> Result<(), LocalVectorError> {
                if let Ok(values) = decode_record_prefix(prefix)
                    && let Some(id) = text_of(values.get(id_column))
                {
                    if rowids.len() >= MAX_MBTILES_TILES {
                        return Err(too_many());
                    }
                    rowids.insert(id, rowid);
                }
                Ok(())
            };
            db.scan_prefixes(self.images_root, &mut visit)?;
        } else {
            let mut visit = |rowid: i64, value: &CellValue| -> Result<(), LocalVectorError> {
                if let Some(id) = text_of(Some(value)) {
                    if rowids.len() >= MAX_MBTILES_TILES {
                        return Err(too_many());
                    }
                    rowids.insert(id, rowid);
                }
                Ok(())
            };
            db.scan_column(self.images_root, id_column, &mut visit)?;
        }
        self.image_rowids = rowids;
        Ok(())
    }

    /// Builds the whole archive's `(key, rowid)` index, sorted by key.
    ///
    /// One [`SqliteDb::scan_prefixes`] pass: the three leading integer columns
    /// of a tile row are a couple of dozen bytes, far inside the ~489 a spilled
    /// leaf cell always keeps on its page, so this never touches a tile body.
    ///
    /// # Errors
    ///
    /// Refuses a pyramid past [`MAX_MBTILES_TILES`], and propagates whatever the
    /// b-tree walk refuses.
    pub(super) fn build_index(
        &self,
        db: &SqliteDb<'_>,
    ) -> Result<Vec<(u64, i64)>, LocalVectorError> {
        let columns = self.address_columns;
        let normalized = self.format == MbTilesFormat::Normalized;
        let widest = if normalized {
            columns.widest()
        } else {
            columns.address_widest()
        };
        let image_rowids = &self.image_rowids;
        let mut index: Vec<(u64, i64)> = Vec::new();
        {
            let mut visit = |rowid: i64, prefix: &[u8]| -> Result<(), LocalVectorError> {
                if index.len() >= MAX_MBTILES_TILES {
                    return Err(LocalVectorError::new(format!(
                        "the archive holds more than {MAX_MBTILES_TILES} tiles, past what OxiGIS \
                         indexes in memory; open it from disk or from a URL instead, which \
                         streams it a page at a time"
                    )));
                }
                // A record whose leading columns spilled is not a tile row this
                // reader can index cheaply; skipping it is honest (the tile
                // simply reads as absent) and cannot happen for three integers.
                let Ok(values) = decode_record_prefix(prefix) else {
                    return Ok(());
                };
                if values.len() <= widest {
                    return Ok(());
                }
                let (Some(zoom), Some(column), Some(row)) = (
                    integer_of(values.get(columns.zoom)),
                    integer_of(values.get(columns.column)),
                    integer_of(values.get(columns.row)),
                ) else {
                    return Ok(());
                };
                let (Ok(zoom), Ok(column), Ok(row)) = (
                    u8::try_from(zoom),
                    u32::try_from(column),
                    u32::try_from(row),
                ) else {
                    return Ok(());
                };
                if zoom > oxigis_render::MAX_ZOOM {
                    return Ok(());
                }
                let side = 1u32 << zoom;
                if column >= side || row >= side {
                    return Ok(());
                }
                // The ONE flip, here and nowhere else.
                let key = index_key(zoom, column, xyz_row(zoom, row));
                let target = if normalized {
                    let Some(id) = text_of(values.get(columns.payload)) else {
                        return Ok(());
                    };
                    let Some(found) = image_rowids.get(&id) else {
                        return Ok(());
                    };
                    *found
                } else {
                    rowid
                };
                index.push((key, target));
                Ok(())
            };
            db.scan_prefixes(self.address_root, &mut visit)?;
        }
        index.sort_unstable_by_key(|(key, _)| *key);
        index.dedup_by_key(|(key, _)| *key);
        Ok(index)
    }

    /// Reads one tile body by the rowid the index recorded.
    ///
    /// # Errors
    ///
    /// Propagates a b-tree failure; a rowid the table no longer holds is
    /// `Ok(None)`.
    pub(super) fn read_blob(
        &self,
        db: &SqliteDb<'_>,
        rowid: i64,
    ) -> Result<Option<Vec<u8>>, LocalVectorError> {
        let (root, column) = match self.format {
            MbTilesFormat::Flat => (self.address_root, self.address_columns.payload),
            MbTilesFormat::Normalized => (self.images_root, self.images_data),
        };
        let Some(row) = db.seek_row(root, rowid)? else {
            return Ok(None);
        };
        Ok(match row.values.get(column) {
            Some(CellValue::Blob(bytes)) => Some(bytes.clone()),
            Some(CellValue::Text(text)) => Some(text.clone().into_bytes()),
            _ => None,
        })
    }
}

/// The `metadata` table, read and interpreted.
#[derive(Debug)]
pub(crate) struct Metadata {
    /// Every key/value the table declared.
    entries: BTreeMap<String, String>,
    /// Whether the tiles are images or vector tiles.
    pub(crate) content: ArchiveContent,
    /// Lowest zoom the archive claims.
    pub(crate) min_zoom: u8,
    /// Highest zoom the archive claims.
    pub(crate) max_zoom: u8,
    /// `[min_lon, min_lat, max_lon, max_lat]` in degrees.
    bounds_deg: [f64; 4],
    /// Whether `bounds` was declared at all.
    has_bounds: bool,
    /// Source-layer names from the `json` key's `vector_layers`.
    pub(crate) vector_layers: Vec<String>,
}

impl Metadata {
    /// Reads and interprets the `metadata` table.
    ///
    /// A plain [`SqliteDb::scan_table`] is right here and nowhere else in this
    /// module: `metadata` is a handful of short rows, so materialising it costs
    /// nothing, while the tile table is the whole archive.
    ///
    /// # Errors
    ///
    /// Refuses an archive with no `metadata` table, one that declares no
    /// `format`, and one whose `format` this build cannot draw.
    pub(super) fn read(db: &SqliteDb<'_>, layout: &Layout) -> Result<Self, LocalVectorError> {
        let entries = db.master_entries_with_autoindex()?;
        let table = entries
            .iter()
            .find(|entry| {
                entry.entry_type == "table"
                    && entry.name.eq_ignore_ascii_case("metadata")
                    && entry.rootpage > 0
            })
            .ok_or_else(|| {
                LocalVectorError::new(format!(
                    "the archive has a {} tile store but no metadata table, so what its tiles \
                     are cannot be known",
                    layout.format().name()
                ))
            })?;
        let schema = parse_create_table(&table.sql).ok_or_else(|| {
            LocalVectorError::new("the archive's metadata table does not declare its columns")
        })?;
        let name_column = schema.column_index("name").unwrap_or(0);
        let value_column = schema.column_index("value").unwrap_or(1);
        let mut map = BTreeMap::new();
        for row in db.scan_table(table.rootpage)? {
            let (Some(key), Some(value)) = (
                text_of(row.values.get(name_column)),
                text_of(row.values.get(value_column)),
            ) else {
                continue;
            };
            map.insert(key.to_ascii_lowercase(), value);
        }
        Self::from_entries(map)
    }

    /// Interprets an already-read `metadata` table.
    ///
    /// Split out of [`Self::read`] so the **paged** reader, which walks the same
    /// table over a byte-range transport and never has a
    /// [`SqliteDb`] to hand, interprets those key/value pairs
    /// through exactly this code. Two readers, one interpretation.
    ///
    /// # Errors
    ///
    /// Refuses a table that declares no `format`, and one whose `format` this
    /// build cannot draw.
    pub(crate) fn from_entries(map: BTreeMap<String, String>) -> Result<Self, LocalVectorError> {
        let format = map.get("format").ok_or_else(|| {
            LocalVectorError::new(
                "the archive's metadata declares no format, so whether its tiles are images or \
                 vector tiles cannot be known",
            )
        })?;
        let content = content_for_format(format)?;
        let min_zoom = map
            .get("minzoom")
            .and_then(|text| text.trim().parse::<u8>().ok())
            .unwrap_or(0)
            .min(oxigis_render::MAX_ZOOM);
        let max_zoom = map
            .get("maxzoom")
            .and_then(|text| text.trim().parse::<u8>().ok())
            .unwrap_or(oxigis_render::MAX_ZOOM)
            .min(oxigis_render::MAX_ZOOM)
            .max(min_zoom);
        let (bounds_deg, has_bounds) = parse_bounds(map.get("bounds").map(String::as_str));
        let vector_layers = parse_vector_layers(map.get("json").map(String::as_str));
        Ok(Self {
            entries: map,
            content,
            min_zoom,
            max_zoom,
            bounds_deg,
            has_bounds,
            vector_layers,
        })
    }

    /// Everything the table declared, keys lower-cased.
    pub(super) const fn entries(&self) -> &BTreeMap<String, String> {
        &self.entries
    }

    /// The archive-level facts, in the shape the archive layer speaks.
    pub(crate) fn info(&self) -> ArchiveInfo {
        ArchiveInfo {
            content: self.content,
            codec: match self.content {
                ArchiveContent::Vector => oxigis_render::pmtiles::TileType::Mvt,
                ArchiveContent::Raster => codec_of(self.entries.get("format")),
            },
            min_zoom: self.min_zoom,
            max_zoom: self.max_zoom,
            bounds_deg: self.bounds_deg,
            has_bounds: self.has_bounds,
            center_deg: (
                (self.bounds_deg[0] + self.bounds_deg[2]) / 2.0,
                (self.bounds_deg[1] + self.bounds_deg[3]) / 2.0,
            ),
            center_zoom: self.min_zoom,
            name: self.entries.get("name").cloned().unwrap_or_default(),
            attribution: self.entries.get("attribution").cloned().unwrap_or_default(),
            layer_names: self.vector_layers.clone(),
            tile_size_px: None,
        }
    }
}

/// The `TileType` a raster `format` string names.
fn codec_of(format: Option<&String>) -> oxigis_render::pmtiles::TileType {
    use oxigis_render::pmtiles::TileType;
    match format
        .map(|text| text.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => TileType::Png,
        Some("jpg" | "jpeg") => TileType::Jpeg,
        Some("webp") => TileType::Webp,
        _ => TileType::Unknown,
    }
}

/// The text of a cell, if it is text.
fn text_of(value: Option<&CellValue>) -> Option<String> {
    match value {
        Some(CellValue::Text(text)) => Some(text.clone()),
        Some(CellValue::Integer(number)) => Some(number.to_string()),
        _ => None,
    }
}

/// The integer of a cell, if it is one.
fn integer_of(value: Option<&CellValue>) -> Option<i64> {
    match value {
        Some(CellValue::Integer(number)) => Some(*number),
        _ => None,
    }
}

/// Parses `bounds` — `"minlon,minlat,maxlon,maxlat"` — leaving the whole world
/// when it is absent or malformed.
///
/// A wrong bounding box gates real tiles away, so a value that does not parse
/// cleanly is treated as no value at all rather than as a partial one.
fn parse_bounds(text: Option<&str>) -> ([f64; 4], bool) {
    const WORLD: [f64; 4] = [-180.0, -85.051_128_7, 180.0, 85.051_128_7];
    let Some(text) = text else {
        return (WORLD, false);
    };
    let parts: Vec<f64> = text
        .split(',')
        .filter_map(|part| part.trim().parse::<f64>().ok())
        .collect();
    match parts.as_slice() {
        [min_lon, min_lat, max_lon, max_lat]
            if min_lon < max_lon && min_lat < max_lat && min_lon.is_finite() =>
        {
            ([*min_lon, *min_lat, *max_lon, *max_lat], true)
        }
        _ => (WORLD, false),
    }
}

/// The `id` of each `vector_layers` entry of the `json` metadata key.
fn parse_vector_layers(json: Option<&str>) -> Vec<String> {
    let Some(json) = json else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(json) else {
        return Vec::new();
    };
    value
        .get("vector_layers")
        .and_then(serde_json::Value::as_array)
        .map(|layers| {
            layers
                .iter()
                .filter_map(|layer| layer.get("id").and_then(serde_json::Value::as_str))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}
