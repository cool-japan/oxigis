// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! [`SqliteSurvey`]: what one 16 KiB read has to tell you before a tile can be
//! looked up, and the refusals that belong to *open* rather than to a tile.
//!
//! ```text
//! bytes 0..16384   ->  100-byte header: page size, reserved area, encoding,
//!                      and how far the page count can be trusted
//!                  ->  page 1 = sqlite_master: the tables, the indices, and
//!                      the auto-indices `master_entries` drops
//!                  ->  the metadata table: format, zooms, bounds, json
//!                  ->  PagedLayout
//! ```
//!
//! Measured on real archives: header, catalogue **and** metadata all land inside
//! that one prefetch, so a cold open costs 1–2 reads — the same as PMTiles.
//!
//! # Every refusal lands here, once, before a layer exists
//!
//! An MBTiles archive read over a byte range is refused at survey time or not at
//! all:
//!
//! * **no usable index** on `(zoom_level, tile_column, tile_row)` — the one case
//!   the old blanket "MBTiles cannot be read over HTTP" refusal was right about,
//!   because without an index a lookup is a full-table scan of the whole
//!   archive. The message says so and names the two ways out.
//! * **a normalized archive with no index on `images.tile_id`** — same reason,
//!   one level down.
//! * **a non-`BINARY` collation** on a key column: a descent that compares bytes
//!   down a `NOCASE` index silently returns the *wrong* tile.
//! * **`WITHOUT ROWID`** (a table rooted at an index b-tree): the machinery could
//!   read it, but the rest of the reader is keyed by rowid throughout, so it is
//!   named rather than half-supported.
//! * **an untrustworthy header**: an illegal page size, a reserved area that
//!   leaves the payload arithmetic meaningless, or UTF-16 text.
//!
//! # How far the page count is trusted
//!
//! Header offset 28 holds a page count that is only meaningful when the
//! file-change counter (24) equals the version-valid-for number (92) — otherwise
//! it was written by a library that never maintained it. When it cannot be
//! trusted the caller may still bound it with the file length it pinned from
//! `Content-Range`; when even that is unknown the count is [`None`] and the only
//! defence left is that a page past the end comes back **short** and is refused
//! by name. It is never anything but an *upper* bound.

use std::collections::{BTreeMap, BTreeSet};

use oxigis_render::ByteRange;

use crate::gpkg_input::sqlite::{
    CellValue, DB_HEADER_LEN, IndexColumn, MAGIC, MAX_DEPTH, MIN_USABLE_SIZE, PAGE_INTERIOR_INDEX,
    PAGE_INTERIOR_TABLE, PAGE_LEAF_INDEX, PAGE_LEAF_TABLE, TableSchema, decode_record,
    parse_create_index, parse_create_table, unique_key_columns,
};
use crate::local_vector::LocalVectorError;
use crate::mbtiles::schema::{Columns, MbTilesFormat, Metadata};

use super::source::{
    ChainStep, OverflowChain, PageRun, PageSource, PageView, err, table_leaf_cell,
};

/// Bytes read speculatively at open.
///
/// The same 16 KiB PMTiles mandates for its header and root directory, and for
/// the same reason: measured over real MBTiles archives, the database header,
/// the whole of `sqlite_master` and the whole `metadata` table all fit inside
/// it, so an open is one round trip.
pub(crate) const SURVEY_PREFETCH_BYTES: u64 = 16 * 1024;

/// How many page runs a survey may issue before it is refused.
///
/// A real archive needs one; sixty-four allows for a large catalogue at a
/// 512-byte page size while stopping a crafted file from making the open walk
/// for ever.
pub(crate) const MAX_SURVEY_PAGES: u32 = 64;

/// The column names an MBTiles address index must be keyed by, in order.
const ADDRESS_KEY: [&str; 3] = ["zoom_level", "tile_column", "tile_row"];

/// What a survey wants next.
#[derive(Debug)]
pub(crate) enum SurveyStep {
    /// The speculative opening read, in bytes: the page size is not known yet,
    /// so this cannot be expressed as pages.
    NeedPrefetch(ByteRange),
    /// These pages are needed.
    NeedPages(PageRun),
    /// The archive is surveyed.
    Ready(Box<PagedLayout>),
}

/// A key column of an index, resolved against the table it indexes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KeyColumn {
    /// Position of the column in the *table's* record.
    pub(crate) position: usize,
    /// Whether the index stores it descending, which flips every comparison at
    /// this position.
    pub(crate) descending: bool,
}

/// One index this reader will descend, and how its keys compare.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndexPlan {
    /// The index's name, for diagnostics and refusals.
    pub(crate) name: String,
    /// Root page of its b-tree.
    pub(crate) root: u32,
    /// Its key columns, in key order.
    pub(crate) columns: Vec<KeyColumn>,
}

/// Everything a paged lookup needs, learned once at open.
#[derive(Debug)]
pub(crate) struct PagedLayout {
    /// Which of the two MBTiles shapes this is.
    pub(crate) format: MbTilesFormat,
    /// Root of `tiles` (flat) or `map` (normalized).
    pub(crate) address_root: u32,
    /// Where the address columns live in that table's records.
    pub(crate) address_columns: Columns,
    /// The index on `(zoom_level, tile_column, tile_row)`.
    pub(crate) address_index: IndexPlan,
    /// Root of `images`, for the normalized shape.
    pub(crate) images_root: u32,
    /// Position of `images.tile_data`, for the normalized shape.
    pub(crate) images_data: usize,
    /// The index on `images.tile_id`, for the normalized shape.
    pub(crate) images_index: Option<IndexPlan>,
    /// What the `metadata` table declared.
    pub(crate) metadata: Metadata,
}

/// How far along a survey is.
#[derive(Debug)]
enum Phase {
    /// The 16 KiB prefetch has not been asked for yet.
    Start,
    /// The catalogue on page 1 is being scanned.
    Catalogue(Box<TableScan>),
    /// The `metadata` table is being scanned.
    Metadata {
        /// The catalogue, already read.
        catalogue: Vec<CatalogueEntry>,
        /// The scan in progress.
        scan: Box<TableScan>,
    },
}

/// One `sqlite_master` row, as the survey needs it.
#[derive(Debug, Clone)]
struct CatalogueEntry {
    /// `"table"`, `"index"`, `"view"` or `"trigger"`.
    kind: String,
    /// The object's own name.
    name: String,
    /// The table it belongs to.
    table: String,
    /// Its b-tree root, or 0.
    rootpage: u32,
    /// Its `CREATE …` statement, empty for an auto-index.
    sql: String,
}

/// Reads an archive's shape out of its own first pages.
#[derive(Debug)]
pub(crate) struct SqliteSurvey {
    /// How far along it is.
    phase: Phase,
    /// The pages, once the page size is known.
    source: Option<PageSource>,
    /// What the caller pinned as the file's total length, if anything.
    declared_total: Option<u64>,
    /// How many page runs have been asked for.
    runs: u32,
}

impl SqliteSurvey {
    /// A survey of an archive whose length the caller may already know.
    ///
    /// `declared_total` is the `Content-Range` total the transport pinned, when
    /// there is one. It is used **only** to bound a page count the header itself
    /// could not vouch for — never to widen one.
    pub(crate) const fn new(declared_total: Option<u64>) -> Self {
        Self {
            phase: Phase::Start,
            source: None,
            declared_total,
            runs: 0,
        }
    }

    /// The pages read so far, once the page size is known.
    pub(crate) const fn source(&self) -> Option<&PageSource> {
        self.source.as_ref()
    }

    /// Takes the page source, so an opened archive keeps the pages its own
    /// survey already paid for.
    pub(crate) fn take_source(&mut self) -> Option<PageSource> {
        self.source.take()
    }

    /// Feeds the opening prefetch in.
    ///
    /// # Errors
    ///
    /// Refuses everything [`parse_db_header`] refuses, each by name.
    pub(crate) fn supply_prefetch(&mut self, bytes: &[u8]) -> Result<(), LocalVectorError> {
        let header = parse_db_header(bytes)?;
        let page_count = header.page_count(self.declared_total);
        let mut source = PageSource::new(header.page_size, header.usable_size, page_count);
        source.supply(1, bytes);
        self.source = Some(source);
        self.phase = Phase::Catalogue(Box::new(TableScan::new(1)));
        Ok(())
    }

    /// Feeds one page run in.
    pub(crate) fn supply_pages(&mut self, first: u32, bytes: &[u8]) {
        if let Some(source) = self.source.as_mut() {
            source.supply(first, bytes);
        }
    }

    /// Drives the survey one step.
    ///
    /// # Errors
    ///
    /// Every named refusal in this module's docs, plus whatever the b-tree walk
    /// itself refuses.
    pub(crate) fn step(&mut self) -> Result<SurveyStep, LocalVectorError> {
        let Some(source) = self.source.as_mut() else {
            let range =
                ByteRange::new(0, SURVEY_PREFETCH_BYTES).map_err(|error| err(error.to_string()))?;
            return Ok(SurveyStep::NeedPrefetch(range));
        };
        loop {
            match &mut self.phase {
                Phase::Start => {
                    return Err(err("the survey was driven before its header arrived"));
                }
                Phase::Catalogue(scan) => {
                    if let Some(run) = scan.step(source)? {
                        self.runs = self.runs.saturating_add(1);
                        if self.runs > MAX_SURVEY_PAGES {
                            return Err(err(format!(
                                "this archive's catalogue is still not read after \
                                 {MAX_SURVEY_PAGES} page reads"
                            )));
                        }
                        return Ok(SurveyStep::NeedPages(run));
                    }
                    let catalogue = catalogue_of(scan.take_rows());
                    let metadata_root = catalogue
                        .iter()
                        .find(|entry| {
                            entry.kind == "table"
                                && entry.name.eq_ignore_ascii_case("metadata")
                                && entry.rootpage > 0
                        })
                        .map(|entry| entry.rootpage)
                        .ok_or_else(|| {
                            err("this archive has no metadata table, so what its tiles are \
                                 cannot be known")
                        })?;
                    self.phase = Phase::Metadata {
                        catalogue,
                        scan: Box::new(TableScan::new(metadata_root)),
                    };
                }
                Phase::Metadata { catalogue, scan } => {
                    if let Some(run) = scan.step(source)? {
                        self.runs = self.runs.saturating_add(1);
                        if self.runs > MAX_SURVEY_PAGES {
                            return Err(err(format!(
                                "this archive's metadata is still not read after \
                                 {MAX_SURVEY_PAGES} page reads"
                            )));
                        }
                        return Ok(SurveyStep::NeedPages(run));
                    }
                    let layout = build_layout(catalogue, scan.take_rows())?;
                    return Ok(SurveyStep::Ready(Box::new(layout)));
                }
            }
        }
    }
}

/// The 100-byte database header, as far as a paged reader cares.
#[derive(Debug, Clone, Copy)]
struct DbHeader {
    /// Bytes per page.
    page_size: usize,
    /// Bytes of each page a b-tree may use.
    usable_size: usize,
    /// The count at offset 28.
    declared_pages: u32,
    /// Whether offsets 24 and 92 agree, which is what makes that count mean
    /// anything.
    trustworthy: bool,
}

impl DbHeader {
    /// The page count, as far as it can be known.
    fn page_count(self, declared_total: Option<u64>) -> Option<u32> {
        if self.trustworthy && self.declared_pages > 0 {
            return Some(self.declared_pages);
        }
        // The header could not vouch for itself. The file's own length, when the
        // transport pinned one, is a real bound; otherwise there is none, and a
        // page past the end is caught by its short delivery instead.
        let total = declared_total?;
        let size = u64::try_from(self.page_size).unwrap_or(1).max(1);
        u32::try_from(total / size).ok().filter(|pages| *pages > 0)
    }
}

/// Parses and validates the 100-byte database header.
///
/// # Errors
///
/// Refuses a file that is not SQLite 3, one whose page size is not a power of
/// two in `512..=65536`, one that reserves so much of each page that the payload
/// arithmetic is meaningless, and one stored in UTF-16. The same refusals
/// [`crate::gpkg_input::sqlite::SqliteDb::open`] makes, restated over a prefix
/// rather than a whole image.
fn parse_db_header(bytes: &[u8]) -> Result<DbHeader, LocalVectorError> {
    let header = bytes
        .get(..DB_HEADER_LEN)
        .ok_or_else(|| err("the first read is too short to hold a database header"))?;
    if header.get(..16) != Some(&MAGIC[..]) {
        return Err(err(
            "this file does not start with the SQLite 3 magic, so it is not an MBTiles archive",
        ));
    }
    let raw = u16::from_be_bytes([header[16], header[17]]);
    let page_size = if raw == 1 { 65536 } else { usize::from(raw) };
    if !(512..=65536).contains(&page_size) || !page_size.is_power_of_two() {
        return Err(err(format!(
            "this archive's header declares a page size of {page_size}, which is not a legal one"
        )));
    }
    let reserved = usize::from(header[20]);
    let usable_size = page_size
        .checked_sub(reserved)
        .filter(|usable| *usable >= MIN_USABLE_SIZE)
        .ok_or_else(|| {
            err(format!(
                "this archive's header reserves {reserved} bytes of every {page_size}-byte page, \
                 which leaves too little of it usable"
            ))
        })?;
    let encoding = u32::from_be_bytes([header[56], header[57], header[58], header[59]]);
    if encoding > 1 {
        return Err(err(
            "this archive is stored in UTF-16; only UTF-8 databases can be read",
        ));
    }
    let change_counter = u32::from_be_bytes([header[24], header[25], header[26], header[27]]);
    let valid_for = u32::from_be_bytes([header[92], header[93], header[94], header[95]]);
    Ok(DbHeader {
        page_size,
        usable_size,
        declared_pages: u32::from_be_bytes([header[28], header[29], header[30], header[31]]),
        trustworthy: change_counter == valid_for,
    })
}

/// Turns `sqlite_master` rows into catalogue entries, **keeping** the
/// auto-indices whose `sql` is NULL.
fn catalogue_of(rows: Vec<(i64, Vec<CellValue>)>) -> Vec<CatalogueEntry> {
    let mut entries = Vec::new();
    for (_, values) in rows {
        let text = |index: usize| match values.get(index) {
            Some(CellValue::Text(value)) => value.clone(),
            _ => String::new(),
        };
        let kind = text(0);
        let name = text(1);
        if kind.is_empty() || name.is_empty() {
            continue;
        }
        let rootpage = match values.get(3) {
            Some(CellValue::Integer(page)) => u32::try_from(*page).unwrap_or(0),
            _ => 0,
        };
        entries.push(CatalogueEntry {
            kind,
            name,
            table: text(2),
            rootpage,
            sql: text(4),
        });
    }
    entries
}

/// Assembles the layout out of the catalogue and the metadata rows.
fn build_layout(
    catalogue: &[CatalogueEntry],
    metadata_rows: Vec<(i64, Vec<CellValue>)>,
) -> Result<PagedLayout, LocalVectorError> {
    let table = |name: &str| {
        catalogue.iter().find(|entry| {
            entry.kind == "table" && entry.name.eq_ignore_ascii_case(name) && entry.rootpage > 0
        })
    };
    let (format, address, address_schema, payload_name) = if let Some(tiles) = table("tiles") {
        let schema = parse_create_table(&tiles.sql).ok_or_else(|| {
            err("this archive's tiles table does not declare its columns, so it cannot be read")
        })?;
        (MbTilesFormat::Flat, tiles, schema, "tile_data")
    } else if let Some(map) = table("map") {
        let schema = parse_create_table(&map.sql).ok_or_else(|| {
            err("this archive's map table does not declare its columns, so it cannot be read")
        })?;
        (MbTilesFormat::Normalized, map, schema, "tile_id")
    } else {
        return Err(err(
            "this archive has neither a tiles table nor the map/images pair MBTiles uses, so it \
             is not an MBTiles archive OxiGIS can read",
        ));
    };
    let address_columns = Columns::resolve(
        &address_schema,
        if format == MbTilesFormat::Flat {
            "tiles"
        } else {
            "map"
        },
        payload_name,
    )?;
    refuse_without_rowid(&address.sql, &address.name)?;

    let address_index = find_index(
        catalogue,
        &address.name,
        &address_schema,
        &ADDRESS_KEY.map(str::to_owned),
    )?
    .ok_or_else(|| {
        err(format!(
            "this archive's {} table has no index on (zoom_level, tile_column, tile_row), so \
             finding one tile would mean reading the whole archive; download it and open the \
             file, or use PMTiles for a remote archive",
            address.name
        ))
    })?;

    let (images_root, images_data, images_index) =
        if format == MbTilesFormat::Normalized {
            let images = table("images").ok_or_else(|| {
                err(
                    "this archive has a map table but no images table, so its tile bodies cannot \
                 be found",
                )
            })?;
            let images_schema = parse_create_table(&images.sql).ok_or_else(|| {
            err("this archive's images table does not declare its columns, so it cannot be read")
        })?;
            refuse_without_rowid(&images.sql, &images.name)?;
            let data = images_schema.column_index("tile_data").ok_or_else(|| {
                err("this archive's images table has no tile_data column, so it is not MBTiles")
            })?;
            let index = find_index(
            catalogue,
            &images.name,
            &images_schema,
            &["tile_id".to_owned()],
        )?
        .ok_or_else(|| {
            err("this archive's images table has no index on tile_id (the images_id index every \
                 MBTiles writer creates), so finding one tile body would mean reading the whole \
                 archive; download it and open the file, or use PMTiles for a remote archive")
        })?;
            (images.rootpage, data, Some(index))
        } else {
            (0, 0, None)
        };

    let mut entries = BTreeMap::new();
    let metadata_schema = table("metadata")
        .and_then(|entry| parse_create_table(&entry.sql))
        .unwrap_or_default();
    let name_column = metadata_schema.column_index("name").unwrap_or(0);
    let value_column = metadata_schema.column_index("value").unwrap_or(1);
    for (_, values) in metadata_rows {
        let (Some(key), Some(value)) = (
            text_of(values.get(name_column)),
            text_of(values.get(value_column)),
        ) else {
            continue;
        };
        entries.insert(key.to_ascii_lowercase(), value);
    }
    let metadata = Metadata::from_entries(entries)?;

    Ok(PagedLayout {
        format,
        address_root: address.rootpage,
        address_columns,
        address_index,
        images_root,
        images_data,
        images_index,
        metadata,
    })
}

/// The text of a cell, if it is text (or an integer, which metadata routinely
/// is).
fn text_of(value: Option<&CellValue>) -> Option<String> {
    match value {
        Some(CellValue::Text(text)) => Some(text.clone()),
        Some(CellValue::Integer(number)) => Some(number.to_string()),
        _ => None,
    }
}

/// Refuses a `WITHOUT ROWID` table by name.
///
/// The descent machinery could read one — its rows live in the index b-tree
/// itself — but every seam above here is keyed by rowid, so half-supporting it
/// would mean a reader that finds the address and then cannot fetch the body.
fn refuse_without_rowid(sql: &str, name: &str) -> Result<(), LocalVectorError> {
    let upper = sql.to_ascii_uppercase().replace(['\n', '\r'], " ");
    if upper.contains("WITHOUT ROWID") {
        return Err(err(format!(
            "this archive's {name} table is declared WITHOUT ROWID, which stores its rows in an \
             index b-tree with no rowids to key them by; download it and open the file, or use \
             PMTiles for a remote archive"
        )));
    }
    Ok(())
}

/// The index on `table` keyed by exactly `wanted`, if there is a usable one.
///
/// Both kinds are considered: a real `CREATE INDEX` statement, and one of
/// SQLite's own auto-indices, whose `sql` is NULL and whose key columns can only
/// be recovered from the **table's** `UNIQUE` constraint (see
/// [`unique_key_columns`]) — the case a `master_entries`-shaped catalogue drops
/// entirely, which would make such an archive read as empty.
///
/// # Errors
///
/// Refuses an otherwise-matching index whose key columns carry a non-`BINARY`
/// collation: comparing bytes down a `NOCASE` index returns the wrong row, and
/// a wrong tile is worse than no tile.
fn find_index(
    catalogue: &[CatalogueEntry],
    table: &str,
    schema: &TableSchema,
    wanted: &[String],
) -> Result<Option<IndexPlan>, LocalVectorError> {
    // Auto-indices are numbered `sqlite_autoindex_<table>_<n>`, `n` counting the
    // implicit indices in declaration order. The numbering is spec-derived, so
    // every record `IndexWalk::rowid_of` decodes validates the shape (key
    // columns plus a trailing integer rowid) before trusting it — on every
    // lookup, not just the first, which is strictly stronger than a one-time
    // probe.
    let table_sql = catalogue
        .iter()
        .find(|entry| entry.kind == "table" && entry.name.eq_ignore_ascii_case(table))
        .map(|entry| entry.sql.clone())
        .unwrap_or_default();
    let implicit = unique_key_columns(&table_sql);

    for entry in catalogue {
        if entry.kind != "index" || entry.rootpage == 0 {
            continue;
        }
        if !entry.table.eq_ignore_ascii_case(table) && !entry.sql.is_empty() {
            continue;
        }
        if entry.sql.is_empty() {
            if !entry.table.eq_ignore_ascii_case(table) {
                continue;
            }
            let Some(position) = autoindex_number(&entry.name) else {
                continue;
            };
            let Some(columns) = implicit.get(position.saturating_sub(1)) else {
                continue;
            };
            let names: Vec<String> = columns.iter().map(|column| column.name.clone()).collect();
            if !same_key(&names, wanted) {
                continue;
            }
            refuse_collation(columns, &entry.name)?;
            let plan = plan_for(
                entry,
                schema,
                columns
                    .iter()
                    .map(|column| (column.name.clone(), column.descending)),
            )?;
            return Ok(Some(plan));
        }
        let Some(parsed) = parse_create_index(&entry.sql) else {
            // A statement this parser cannot read is skipped, never guessed at:
            // descending an index whose key columns are not what the reader
            // thinks they are returns a plausible, wrong row.
            continue;
        };
        if parsed.partial || !parsed.table.eq_ignore_ascii_case(table) {
            continue;
        }
        let names: Vec<String> = parsed
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        if !same_key(&names, wanted) {
            continue;
        }
        refuse_collation(&parsed.columns, &entry.name)?;
        let plan = plan_for(
            entry,
            schema,
            parsed
                .columns
                .iter()
                .map(|column| (column.name.clone(), column.descending)),
        )?;
        return Ok(Some(plan));
    }
    Ok(None)
}

/// Whether an index's key columns are exactly `wanted`, in order.
fn same_key(columns: &[String], wanted: &[String]) -> bool {
    columns.len() == wanted.len()
        && columns
            .iter()
            .zip(wanted.iter())
            .all(|(have, want)| have.eq_ignore_ascii_case(want))
}

/// The `n` of `sqlite_autoindex_<table>_<n>`.
fn autoindex_number(name: &str) -> Option<usize> {
    if !name.to_ascii_lowercase().starts_with("sqlite_autoindex_") {
        return None;
    }
    name.rsplit('_').next()?.parse::<usize>().ok()
}

/// Refuses an index whose keys do not compare byte-for-byte.
fn refuse_collation(columns: &[IndexColumn], name: &str) -> Result<(), LocalVectorError> {
    for column in columns {
        if !column.collation.is_empty() && column.collation != "BINARY" {
            return Err(err(format!(
                "this archive's {name} index sorts {} with the {} collation, which OxiGIS cannot \
                 reproduce; a descent that compared bytes would silently return the wrong tile",
                column.name, column.collation
            )));
        }
    }
    Ok(())
}

/// Resolves an index's key columns against the table's own column order.
fn plan_for(
    entry: &CatalogueEntry,
    schema: &TableSchema,
    columns: impl Iterator<Item = (String, bool)>,
) -> Result<IndexPlan, LocalVectorError> {
    let mut resolved = Vec::new();
    for (name, descending) in columns {
        let position = schema.column_index(&name).ok_or_else(|| {
            err(format!(
                "this archive's {} index is keyed by a column ({name}) its own table does not \
                 declare",
                entry.name
            ))
        })?;
        resolved.push(KeyColumn {
            position,
            descending,
        });
    }
    Ok(IndexPlan {
        name: entry.name.clone(),
        root: entry.rootpage,
        columns: resolved,
    })
}

/// Where a [`TableScan`] is inside one leaf page.
#[derive(Debug)]
struct LeafCursor {
    /// The page being read.
    page: u32,
    /// Its cell offsets.
    cells: Vec<usize>,
    /// How many of them have been consumed.
    index: usize,
}

/// A resumable full scan of one table b-tree, decoding whole records.
///
/// Used **only** by the survey: the catalogue and the metadata table are a
/// handful of short rows each, while the tile table is the whole archive and is
/// never scanned at all.
#[derive(Debug)]
pub(crate) struct TableScan {
    /// Pages still to visit, with their depth.
    stack: Vec<(u32, u32)>,
    /// Pages already visited, so a cycle is an error rather than a hang.
    visited: BTreeSet<u32>,
    /// Where the scan is inside the current leaf.
    cursor: Option<LeafCursor>,
    /// A record being assembled out of an overflow chain.
    pending: Option<(i64, OverflowChain)>,
    /// The rows read so far.
    rows: Vec<(i64, Vec<CellValue>)>,
}

impl TableScan {
    /// A scan of the b-tree rooted at `root`.
    pub(crate) fn new(root: u32) -> Self {
        Self {
            stack: vec![(root, 0)],
            visited: BTreeSet::new(),
            cursor: None,
            pending: None,
            rows: Vec::new(),
        }
    }

    /// The rows read, leaving the scan empty.
    pub(crate) fn take_rows(&mut self) -> Vec<(i64, Vec<CellValue>)> {
        core::mem::take(&mut self.rows)
    }

    /// Drives the scan until it needs a page or is finished.
    ///
    /// # Errors
    ///
    /// Refuses an index b-tree (a `WITHOUT ROWID` table), a truncated page, an
    /// out-of-range pointer, a cycle and a tree past [`MAX_DEPTH`].
    pub(crate) fn step(
        &mut self,
        source: &mut PageSource,
    ) -> Result<Option<PageRun>, LocalVectorError> {
        loop {
            if let Some((rowid, chain)) = &mut self.pending {
                let rowid = *rowid;
                match chain.step(source)? {
                    ChainStep::Need(run) => return Ok(Some(run)),
                    ChainStep::Done(payload) => {
                        self.pending = None;
                        self.rows.push((rowid, decode_record(&payload)?));
                    }
                }
                continue;
            }
            if let Some(cursor) = &mut self.cursor {
                let Some(at) = cursor.cells.get(cursor.index).copied() else {
                    self.cursor = None;
                    continue;
                };
                cursor.index += 1;
                let number = cursor.page;
                let Some(bytes) = source.get(number) else {
                    return Ok(Some(PageRun::one(number)));
                };
                let page = PageView::new(number, bytes);
                let (rowid, payload) = table_leaf_cell(&page, at, source.usable_size())?;
                if payload.is_complete() {
                    self.rows.push((rowid, decode_record(&payload.inline)?));
                } else {
                    self.pending = Some((rowid, OverflowChain::start(payload)?));
                }
                continue;
            }
            let Some((number, depth)) = self.stack.pop() else {
                return Ok(None);
            };
            if depth > MAX_DEPTH {
                return Err(err("the b-tree is deeper than any real database"));
            }
            if !source.is_addressable(number) {
                return Err(err(format!("page {number} is outside the file")));
            }
            if !self.visited.insert(number) {
                return Err(err(format!("page {number} is reachable twice (a cycle)")));
            }
            let Some(bytes) = source.get(number) else {
                // Put it back and ask for it.
                self.visited.remove(&number);
                self.stack.push((number, depth));
                return Ok(Some(PageRun::one(number)));
            };
            let page = PageView::new(number, bytes);
            match page.kind()? {
                PAGE_LEAF_TABLE => {
                    self.cursor = Some(LeafCursor {
                        page: number,
                        cells: page.cells()?,
                        index: 0,
                    });
                }
                PAGE_INTERIOR_TABLE => {
                    self.stack.push((page.right_child()?, depth + 1));
                    for at in page.cells()?.into_iter().rev() {
                        self.stack.push((page.pointer_at(at)?, depth + 1));
                    }
                }
                PAGE_INTERIOR_INDEX | PAGE_LEAF_INDEX => {
                    return Err(err(
                        "this table is stored as an index b-tree (a WITHOUT ROWID table), which \
                         has no rowids",
                    ));
                }
                other => return Err(err(format!("page {number} has b-tree type {other}"))),
            }
        }
    }
}
