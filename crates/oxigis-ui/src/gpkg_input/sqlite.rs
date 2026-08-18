// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! A read-only SQLite b-tree walker over an in-memory database image.
//!
//! Everything a GeoPackage reader needs from SQLite and nothing else: open an
//! image, list `sqlite_master`, walk one table b-tree, hand back decoded rows.
//! No writing, no journal/WAL, no indices, no query planner, no `PRAGMA`.
//!
//! # Why this exists at all
//!
//! `oxigeo-gpkg` 0.2.2 ships a b-tree reader, and it was measured before this
//! module was written: it does not follow **overflow-page chains**, so any row
//! whose record exceeds `usable_size - 35` bytes (≈ 4061 B at the 4096-byte page
//! size GDAL writes by default) makes the *entire table* fail. A polygon of a
//! few hundred vertices is already over that line, so the gap is not exotic —
//! it is most of the polygon layers in the world. Following the chain is the
//! one thing this file must get right; see [`SqliteDb::cell_payload`].
//!
//! # Untrusted input
//!
//! Every byte here comes from a file the user dropped on the map. Page numbers,
//! cell pointers, payload lengths and serial types are all attacker-controlled,
//! so: every slice is bounds-checked against the real buffer length (never
//! against the header's own page count), every allocation is capped by the
//! file's own size, and every traversal is bounded — one shared visited-page set
//! covers the b-tree descent *and* the overflow chains, so a page pointing at
//! its own ancestor is an error rather than a hang.

use std::collections::BTreeSet;

use crate::local_vector::LocalVectorError;

/// Length of the database header, and the offset page 1's b-tree header sits
/// at (every other page's is at 0).
pub(crate) const DB_HEADER_LEN: usize = 100;

/// The 16-byte magic every SQLite 3 image starts with.
pub(crate) const MAGIC: &[u8; 16] = b"SQLite format 3\0";

/// Smallest usable page area SQLite itself will produce. Enforcing it keeps the
/// spilled-payload arithmetic below (which divides by `usable - 4` and by 255)
/// away from degenerate header values.
pub(crate) const MIN_USABLE_SIZE: usize = 480;

/// Depth cap for the b-tree descent. Real trees are a handful of levels deep
/// even at billions of rows; this only stops a crafted cycle-free but very deep
/// pointer chain.
pub(crate) const MAX_DEPTH: u32 = 64;

/// B-tree page type: interior index.
pub(crate) const PAGE_INTERIOR_INDEX: u8 = 2;
/// B-tree page type: interior table.
pub(crate) const PAGE_INTERIOR_TABLE: u8 = 5;
/// B-tree page type: leaf index.
pub(crate) const PAGE_LEAF_INDEX: u8 = 10;
/// B-tree page type: leaf table.
pub(crate) const PAGE_LEAF_TABLE: u8 = 13;

/// Builds a reader error, prefixed so a status line says which layer of the
/// stack refused the file.
fn err(message: impl AsRef<str>) -> LocalVectorError {
    LocalVectorError::new(format!("SQLite: {}", message.as_ref()))
}

/// One decoded column value — SQLite's five storage classes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CellValue {
    /// `NULL`.
    Null,
    /// An integer, however many bytes it was stored in.
    Integer(i64),
    /// An IEEE-754 double.
    Float(f64),
    /// UTF-8 text. Invalid sequences are replaced rather than rejected: one
    /// mojibake cell should not cost a whole layer.
    Text(String),
    /// Raw bytes — how a GeoPackage geometry arrives.
    Blob(Vec<u8>),
}

/// One row of a table b-tree: its rowid and its record's column values.
#[derive(Debug, Clone)]
pub(crate) struct Row {
    /// The cell's rowid — also the value of an `INTEGER PRIMARY KEY` column,
    /// which the record itself stores as `NULL` (see
    /// [`TableSchema::rowid_alias`]).
    pub(crate) rowid: i64,
    /// Column values in schema order. A record may hold *fewer* values than the
    /// table has columns (a column added by `ALTER TABLE` after the row was
    /// written); the caller pads with `NULL`.
    pub(crate) values: Vec<CellValue>,
}

/// One `sqlite_master` row: the schema catalogue.
#[derive(Debug, Clone)]
pub(crate) struct MasterEntry {
    /// `"table"`, `"index"`, `"view"` or `"trigger"`.
    pub(crate) entry_type: String,
    /// Name of the object.
    pub(crate) name: String,
    /// Page number of its b-tree root, or 0 for objects with no storage.
    pub(crate) rootpage: u32,
    /// The `CREATE …` statement it was declared with, or empty for one of
    /// SQLite's own auto-indices, which has none.
    pub(crate) sql: String,
}

/// A read-only SQLite image.
#[derive(Debug)]
pub(crate) struct SqliteDb<'a> {
    /// The whole file.
    bytes: &'a [u8],
    /// Bytes per page, from the header (512..=65536, a power of two).
    page_size: usize,
    /// Bytes of each page a b-tree may use — `page_size` minus the per-page
    /// reserved region. Every payload-size rule below is stated in terms of it.
    usable_size: usize,
    /// Number of pages actually addressable in `bytes`.
    page_count: u32,
    /// Whether the header's file-format version bytes (offsets 18/19) name
    /// write-ahead logging rather than the legacy rollback journal.
    wal_mode: bool,
}

impl<'a> SqliteDb<'a> {
    /// Validates the 100-byte database header and takes the image.
    ///
    /// # Errors
    ///
    /// Refuses a file that is not SQLite 3, that declares a page size which is
    /// not a power of two in `512..=65536`, that reserves so much of each page
    /// that the payload arithmetic would be meaningless, that is not stored in
    /// UTF-8, or that is shorter than a single page.
    pub(crate) fn open(bytes: &'a [u8]) -> Result<Self, LocalVectorError> {
        let header = bytes
            .get(..DB_HEADER_LEN)
            .ok_or_else(|| err("the file is too short to be a database"))?;
        if &header[..16] != MAGIC {
            return Err(err("the file does not start with the SQLite 3 magic"));
        }
        // Offset 16: page size. The value 1 means 65536, which does not fit the
        // u16 the header stores it in.
        let raw = u16::from_be_bytes([header[16], header[17]]);
        let page_size = if raw == 1 { 65536 } else { usize::from(raw) };
        if !(512..=65536).contains(&page_size) || !page_size.is_power_of_two() {
            return Err(err(format!("page size {page_size} is not a legal one")));
        }
        // Offset 20: bytes reserved at the end of every page (an extension's
        // scratch area). The b-tree may not touch them.
        let reserved = usize::from(header[20]);
        let usable_size = page_size
            .checked_sub(reserved)
            .filter(|usable| *usable >= MIN_USABLE_SIZE)
            .ok_or_else(|| {
                err(format!(
                    "{reserved} reserved bytes leave too little of a {page_size}-byte page usable"
                ))
            })?;
        // Offset 56: text encoding. 0 is "not set yet" (an empty database);
        // 1 is UTF-8. UTF-16 would need every TEXT cell transcoded, and no
        // GeoPackage writer emits it.
        let encoding = u32::from_be_bytes([header[56], header[57], header[58], header[59]]);
        if encoding > 1 {
            return Err(err(
                "the database is stored in UTF-16; only UTF-8 databases can be read",
            ));
        }
        // Offset 28 holds a page count, but it is only meaningful when the
        // file-change counter (24) and the version-valid-for number (92) agree;
        // otherwise it was written by a pre-3.7 library that never maintained
        // it. Either way it is only ever an *upper* bound here — what is
        // addressable is decided by the buffer length alone.
        let change_counter = u32::from_be_bytes([header[24], header[25], header[26], header[27]]);
        let valid_for = u32::from_be_bytes([header[92], header[93], header[94], header[95]]);
        let declared = u32::from_be_bytes([header[28], header[29], header[30], header[31]]);
        let present = u32::try_from(bytes.len() / page_size).unwrap_or(u32::MAX);
        let page_count = if change_counter == valid_for && declared > 0 {
            declared.min(present)
        } else {
            present
        };
        if page_count == 0 {
            return Err(err("the file holds less than one whole page"));
        }
        // Offsets 18/19: file-format write/read version. 1 is the legacy
        // rollback journal; 2 is write-ahead logging, whose uncommitted rows
        // live in a `-wal` sidecar this reader never applies (see
        // `gpkg_input::from_bytes`). Either byte naming it is enough — a real
        // writer sets both together, but only one is needed to answer the
        // question this reader actually asks.
        let wal_mode = header[18] == 2 || header[19] == 2;
        Ok(Self {
            bytes,
            page_size,
            usable_size,
            page_count,
            wal_mode,
        })
    }

    /// Whether the header names write-ahead-log mode rather than the legacy
    /// rollback journal — see [`Self::open`]'s `wal_mode` note.
    pub(crate) const fn wal_mode(&self) -> bool {
        self.wal_mode
    }

    /// Bytes of each page a b-tree may use — how a test checks that the
    /// reserved-bytes byte of the header was honoured.
    #[cfg(test)]
    pub(crate) fn usable_size(&self) -> usize {
        self.usable_size
    }

    /// The usable area of page `number` (1-based).
    ///
    /// # Errors
    ///
    /// Refuses a page number outside the file — the only defence against a
    /// crafted child or overflow pointer.
    fn page(&self, number: u32) -> Result<&'a [u8], LocalVectorError> {
        if number == 0 || number > self.page_count {
            return Err(err(format!("page {number} is outside the file")));
        }
        let start = (number as usize - 1) * self.page_size;
        self.bytes
            .get(start..start + self.usable_size)
            .ok_or_else(|| err(format!("page {number} is truncated")))
    }

    /// Every `sqlite_master` entry, in rowid order.
    ///
    /// The schema table's root is always page 1 and its five columns are
    /// `(type, name, tbl_name, rootpage, sql)`. Entries with no `CREATE`
    /// statement (SQLite's own auto-indices) are dropped: nothing here can use
    /// one.
    ///
    /// # Errors
    ///
    /// Propagates any failure of the page-1 b-tree walk.
    pub(crate) fn master_entries(&self) -> Result<Vec<MasterEntry>, LocalVectorError> {
        let mut entries = Vec::new();
        for row in self.scan_table(1)? {
            let text = |index: usize| match row.values.get(index) {
                Some(CellValue::Text(value)) => Some(value.clone()),
                _ => None,
            };
            let (Some(entry_type), Some(name), Some(sql)) = (text(0), text(1), text(4)) else {
                continue;
            };
            let rootpage = match row.values.get(3) {
                Some(CellValue::Integer(page)) => u32::try_from(*page).unwrap_or(0),
                _ => 0,
            };
            entries.push(MasterEntry {
                entry_type,
                name,
                rootpage,
                sql,
            });
        }
        Ok(entries)
    }

    /// Every `sqlite_master` entry, **including SQLite's own auto-indices**.
    ///
    /// [`Self::master_entries`] drops rows with no `CREATE` statement, which is
    /// right for a GeoPackage reader — it can do nothing with an index — and
    /// silently wrong for one that reads *through* an index. A `UNIQUE (z, x,
    /// y)` written as a **table constraint** produces exactly such a row:
    ///
    /// ```text
    /// type=index  name=sqlite_autoindex_tiles_1  tbl_name=tiles  rootpage=4  sql=NULL
    /// ```
    ///
    /// It is a real b-tree with a real root page, and it is the *only* index on
    /// a great many MBTiles archives — so a reader that drops it concludes the
    /// archive has no index and reports it as empty. Such an entry comes back
    /// here with `sql` empty; its key columns are recovered from the **table's**
    /// statement with [`unique_key_columns`].
    ///
    /// # Errors
    ///
    /// Propagates any failure of the page-1 b-tree walk.
    pub(crate) fn master_entries_with_autoindex(
        &self,
    ) -> Result<Vec<MasterEntry>, LocalVectorError> {
        let mut entries = Vec::new();
        for row in self.scan_table(1)? {
            let text = |index: usize| match row.values.get(index) {
                Some(CellValue::Text(value)) => Some(value.clone()),
                _ => None,
            };
            let (Some(entry_type), Some(name)) = (text(0), text(1)) else {
                continue;
            };
            let rootpage = match row.values.get(3) {
                Some(CellValue::Integer(page)) => u32::try_from(*page).unwrap_or(0),
                _ => 0,
            };
            entries.push(MasterEntry {
                entry_type,
                name,
                rootpage,
                sql: text(4).unwrap_or_default(),
            });
        }
        Ok(entries)
    }

    /// Walks the table b-tree rooted at `root` and decodes every row, in rowid
    /// order.
    ///
    /// # Errors
    ///
    /// Refuses an *index* b-tree — which is what the root of a `WITHOUT ROWID`
    /// table is, and which holds no rowids to key rows by — as well as any
    /// truncated page, out-of-range pointer, or cycle.
    pub(crate) fn scan_table(&self, root: u32) -> Result<Vec<Row>, LocalVectorError> {
        let mut visited = BTreeSet::new();
        let mut rows = Vec::new();
        // Children are pushed in reverse so popping yields them left to right,
        // i.e. rows come out in rowid order without a sort.
        let mut stack = vec![(root, 0u32)];
        while let Some((number, depth)) = stack.pop() {
            if depth > MAX_DEPTH {
                return Err(err("the b-tree is deeper than any real database"));
            }
            if !visited.insert(number) {
                return Err(err(format!("page {number} is reachable twice (a cycle)")));
            }
            let page = self.page(number)?;
            let base = if number == 1 { DB_HEADER_LEN } else { 0 };
            let header = page
                .get(base..base + 8)
                .ok_or_else(|| err(format!("page {number} has no b-tree header")))?;
            let kind = header[0];
            let cell_count = usize::from(u16::from_be_bytes([header[3], header[4]]));
            match kind {
                PAGE_LEAF_TABLE => {
                    for cell in self.cell_offsets(page, base, 8, cell_count, number)? {
                        rows.push(self.leaf_row(page, cell, &mut visited)?);
                    }
                }
                PAGE_INTERIOR_TABLE => {
                    let right = u32::from_be_bytes([
                        *page.get(base + 8).ok_or_else(|| err("truncated header"))?,
                        *page.get(base + 9).ok_or_else(|| err("truncated header"))?,
                        *page.get(base + 10).ok_or_else(|| err("truncated header"))?,
                        *page.get(base + 11).ok_or_else(|| err("truncated header"))?,
                    ]);
                    stack.push((right, depth + 1));
                    // An interior cell is a 4-byte left-child page number
                    // followed by a rowid varint that only a search would need.
                    for cell in self
                        .cell_offsets(page, base, 12, cell_count, number)?
                        .into_iter()
                        .rev()
                    {
                        let child = page
                            .get(cell..cell + 4)
                            .ok_or_else(|| err(format!("page {number} has a truncated cell")))?;
                        let child = u32::from_be_bytes([child[0], child[1], child[2], child[3]]);
                        stack.push((child, depth + 1));
                    }
                }
                PAGE_INTERIOR_INDEX | PAGE_LEAF_INDEX => {
                    return Err(err(
                        "the table is stored as an index b-tree (a WITHOUT ROWID table), which \
                         has no rowids",
                    ));
                }
                other => return Err(err(format!("page {number} has b-tree type {other}"))),
            }
        }
        Ok(rows)
    }

    /// The row with `rowid` in the table b-tree rooted at `root`, found by
    /// descending the interior pages' rowid keys.
    ///
    /// One page read per level instead of a whole-table scan, which is what
    /// makes reading a single tile out of an MBTiles archive affordable:
    /// [`Self::scan_table`] materialises **every blob in the file**, and an
    /// archive is nothing but blobs.
    ///
    /// This is the search [`Self::scan_table`]'s own comment anticipates when it
    /// skips "a rowid varint that only a search would need": an interior table
    /// cell is a 4-byte left-child page number followed by that varint, which is
    /// the **largest rowid in the left subtree**. So the first cell whose key is
    /// at least the target owns it, and a target past every key lives under the
    /// page's right-most pointer.
    ///
    /// # Errors
    ///
    /// The same refusals as [`Self::scan_table`]: an index b-tree, a truncated
    /// page, an out-of-range pointer, a cycle, or a tree past [`MAX_DEPTH`].
    /// A rowid the table simply does not hold is `Ok(None)`, not an error.
    pub(crate) fn seek_row(&self, root: u32, rowid: i64) -> Result<Option<Row>, LocalVectorError> {
        let mut visited = BTreeSet::new();
        let mut number = root;
        for _depth in 0..MAX_DEPTH {
            if !visited.insert(number) {
                return Err(err(format!("page {number} is reachable twice (a cycle)")));
            }
            let page = self.page(number)?;
            let base = if number == 1 { DB_HEADER_LEN } else { 0 };
            let header = page
                .get(base..base + 8)
                .ok_or_else(|| err(format!("page {number} has no b-tree header")))?;
            let kind = header[0];
            let cell_count = usize::from(u16::from_be_bytes([header[3], header[4]]));
            match kind {
                PAGE_LEAF_TABLE => {
                    for cell in self.cell_offsets(page, base, 8, cell_count, number)? {
                        let (_payload_len, consumed) = varint(page, cell)?;
                        let (found, _used) = varint(page, cell + consumed)?;
                        if found == rowid {
                            return Ok(Some(self.leaf_row(page, cell, &mut visited)?));
                        }
                    }
                    return Ok(None);
                }
                PAGE_INTERIOR_TABLE => {
                    let mut child = None;
                    for cell in self.cell_offsets(page, base, 12, cell_count, number)? {
                        let pointer = page
                            .get(cell..cell + 4)
                            .ok_or_else(|| err(format!("page {number} has a truncated cell")))?;
                        let (key, _used) = varint(page, cell + 4)?;
                        if rowid <= key {
                            child = Some(u32::from_be_bytes([
                                pointer[0], pointer[1], pointer[2], pointer[3],
                            ]));
                            break;
                        }
                    }
                    number = match child {
                        Some(child) => child,
                        None => {
                            let right = page.get(base + 8..base + 12).ok_or_else(|| {
                                err(format!("page {number} has a truncated header"))
                            })?;
                            u32::from_be_bytes([right[0], right[1], right[2], right[3]])
                        }
                    };
                }
                PAGE_INTERIOR_INDEX | PAGE_LEAF_INDEX => {
                    return Err(err(
                        "the table is stored as an index b-tree (a WITHOUT ROWID table), which \
                         has no rowids",
                    ));
                }
                other => return Err(err(format!("page {number} has b-tree type {other}"))),
            }
        }
        Err(err("the b-tree is deeper than any real database"))
    }

    /// Walks the table b-tree rooted at `root` and hands `visit` each row's
    /// rowid plus the **inline prefix** of its record, in rowid order.
    ///
    /// Deliberately never follows an overflow chain, which is what makes it
    /// cheap enough to run over a whole tile archive: a spilled leaf cell always
    /// keeps at least `M = ((U - 12) * 32 / 255) - 23` bytes on the page —
    /// about 489 at the 4096-byte page size every MBTiles writer uses — and a
    /// record header plus three small integers is a couple of dozen. Indexing
    /// the leading columns of a blob table therefore costs zero overflow reads
    /// and never materialises a single tile body.
    ///
    /// A visitor that needs a column past the inline prefix simply will not find
    /// it in the record; [`Self::seek_row`] is how it then reads that one row in
    /// full.
    ///
    /// # Errors
    ///
    /// The same refusals as [`Self::scan_table`], plus whatever `visit` returns.
    pub(crate) fn scan_prefixes(
        &self,
        root: u32,
        visit: &mut PrefixVisitor<'_>,
    ) -> Result<(), LocalVectorError> {
        let mut visited = BTreeSet::new();
        let mut stack = vec![(root, 0u32)];
        while let Some((number, depth)) = stack.pop() {
            if depth > MAX_DEPTH {
                return Err(err("the b-tree is deeper than any real database"));
            }
            if !visited.insert(number) {
                return Err(err(format!("page {number} is reachable twice (a cycle)")));
            }
            let page = self.page(number)?;
            let base = if number == 1 { DB_HEADER_LEN } else { 0 };
            let header = page
                .get(base..base + 8)
                .ok_or_else(|| err(format!("page {number} has no b-tree header")))?;
            let kind = header[0];
            let cell_count = usize::from(u16::from_be_bytes([header[3], header[4]]));
            match kind {
                PAGE_LEAF_TABLE => {
                    for cell in self.cell_offsets(page, base, 8, cell_count, number)? {
                        let (payload_len, consumed) = varint(page, cell)?;
                        let (rowid, rowid_len) = varint(page, cell + consumed)?;
                        let start = cell + consumed + rowid_len;
                        let inline = self.inline_len(payload_len)?;
                        let prefix = page
                            .get(start..start + inline)
                            .ok_or_else(|| err("a cell's inline payload runs past its page"))?;
                        visit(rowid, prefix)?;
                    }
                }
                PAGE_INTERIOR_TABLE => {
                    let right = page
                        .get(base + 8..base + 12)
                        .ok_or_else(|| err(format!("page {number} has a truncated header")))?;
                    stack.push((
                        u32::from_be_bytes([right[0], right[1], right[2], right[3]]),
                        depth + 1,
                    ));
                    for cell in self
                        .cell_offsets(page, base, 12, cell_count, number)?
                        .into_iter()
                        .rev()
                    {
                        let child = page
                            .get(cell..cell + 4)
                            .ok_or_else(|| err(format!("page {number} has a truncated cell")))?;
                        stack.push((
                            u32::from_be_bytes([child[0], child[1], child[2], child[3]]),
                            depth + 1,
                        ));
                    }
                }
                PAGE_INTERIOR_INDEX | PAGE_LEAF_INDEX => {
                    return Err(err(
                        "the table is stored as an index b-tree (a WITHOUT ROWID table), which \
                         has no rowids",
                    ));
                }
                other => return Err(err(format!("page {number} has b-tree type {other}"))),
            }
        }
        Ok(())
    }

    /// Walks the table b-tree rooted at `root` and hands `visit` each row's
    /// rowid and the value of column `index` alone, in rowid order.
    ///
    /// [`Self::scan_table`] decodes every column of every row, which is right
    /// for a handful of short metadata rows and wrong for a blob table: asking
    /// it for one column out of `(tile_data, tile_id)` still assembles and
    /// keeps every `tile_data` in memory, only to throw each one away. This
    /// instead reads a column's *size* off its record's header — which never
    /// requires touching its bytes — and follows the overflow chain only far
    /// enough to reach `index`'s own bytes, so a normalized archive's
    /// `images.tile_id` (declared last, after the blob) costs the id plus the
    /// overflow pages between it and the header, never the blob itself.
    ///
    /// # Errors
    ///
    /// The same refusals as [`Self::scan_table`], plus whatever `visit`
    /// returns. A row whose record does not reach `index` at all — an
    /// `ALTER TABLE ADD COLUMN` row, or one whose header does not fit the
    /// guaranteed-inline region (see [`Self::column_value`]) — is simply not
    /// visited, the same honesty [`Self::scan_prefixes`] applies to a column
    /// past its own inline prefix.
    pub(crate) fn scan_column(
        &self,
        root: u32,
        index: usize,
        visit: &mut ColumnVisitor<'_>,
    ) -> Result<(), LocalVectorError> {
        let mut visited = BTreeSet::new();
        let mut stack = vec![(root, 0u32)];
        while let Some((number, depth)) = stack.pop() {
            if depth > MAX_DEPTH {
                return Err(err("the b-tree is deeper than any real database"));
            }
            if !visited.insert(number) {
                return Err(err(format!("page {number} is reachable twice (a cycle)")));
            }
            let page = self.page(number)?;
            let base = if number == 1 { DB_HEADER_LEN } else { 0 };
            let header = page
                .get(base..base + 8)
                .ok_or_else(|| err(format!("page {number} has no b-tree header")))?;
            let kind = header[0];
            let cell_count = usize::from(u16::from_be_bytes([header[3], header[4]]));
            match kind {
                PAGE_LEAF_TABLE => {
                    for cell in self.cell_offsets(page, base, 8, cell_count, number)? {
                        let (rowid, value) = self.leaf_column(page, cell, index, &mut visited)?;
                        if let Some(value) = value {
                            visit(rowid, &value)?;
                        }
                    }
                }
                PAGE_INTERIOR_TABLE => {
                    let right = u32::from_be_bytes([
                        *page.get(base + 8).ok_or_else(|| err("truncated header"))?,
                        *page.get(base + 9).ok_or_else(|| err("truncated header"))?,
                        *page.get(base + 10).ok_or_else(|| err("truncated header"))?,
                        *page.get(base + 11).ok_or_else(|| err("truncated header"))?,
                    ]);
                    stack.push((right, depth + 1));
                    for cell in self
                        .cell_offsets(page, base, 12, cell_count, number)?
                        .into_iter()
                        .rev()
                    {
                        let child = page
                            .get(cell..cell + 4)
                            .ok_or_else(|| err(format!("page {number} has a truncated cell")))?;
                        let child = u32::from_be_bytes([child[0], child[1], child[2], child[3]]);
                        stack.push((child, depth + 1));
                    }
                }
                PAGE_INTERIOR_INDEX | PAGE_LEAF_INDEX => {
                    return Err(err(
                        "the table is stored as an index b-tree (a WITHOUT ROWID table), which \
                         has no rowids",
                    ));
                }
                other => return Err(err(format!("page {number} has b-tree type {other}"))),
            }
        }
        Ok(())
    }

    /// How many bytes of a `payload_len`-byte record stay on its leaf page.
    ///
    /// The inline half of [`Self::cell_payload`]'s rule, extracted so
    /// [`Self::scan_prefixes`] can stop exactly where the overflow chain would
    /// start. Keeping one copy of the arithmetic is the point: the two answers
    /// must agree or the prefix would be cut in the wrong place — which is why
    /// the arithmetic itself lives in the free [`inline_len_for`], shared with
    /// every other reader of this format in the crate.
    fn inline_len(&self, payload_len: i64) -> Result<usize, LocalVectorError> {
        let total = usize::try_from(payload_len)
            .ok()
            .filter(|len| *len <= self.bytes.len())
            .ok_or_else(|| err(format!("a cell claims a {payload_len}-byte payload")))?;
        Ok(inline_len_for(self.usable_size, total))
    }

    /// The cell-content offsets of one page, validated against it.
    ///
    /// A one-line delegation to the free [`cell_offsets_in`]; see there for the
    /// pointer-array layout and the bounds rule.
    fn cell_offsets(
        &self,
        page: &[u8],
        base: usize,
        header_len: usize,
        cell_count: usize,
        number: u32,
    ) -> Result<Vec<usize>, LocalVectorError> {
        cell_offsets_in(page, base, header_len, cell_count, number)
    }

    /// Decodes one leaf-table cell into a row.
    fn leaf_row(
        &self,
        page: &[u8],
        cell: usize,
        visited: &mut BTreeSet<u32>,
    ) -> Result<Row, LocalVectorError> {
        let (payload_len, consumed) = varint(page, cell)?;
        let (rowid, rowid_len) = varint(page, cell + consumed)?;
        let payload = self.cell_payload(page, cell + consumed + rowid_len, payload_len, visited)?;
        Ok(Row {
            rowid,
            values: decode_record(&payload)?,
        })
    }

    /// Assembles a cell's record payload, following its overflow chain.
    ///
    /// The inline/overflow split is the part of the file format that is easy to
    /// get subtly wrong, so it is spelled out. With `U` the usable page size and
    /// `P` the total payload length, a leaf-table cell keeps everything inline
    /// while `P <= X` where `X = U - 35`. Above that it keeps `K` bytes inline —
    /// where `M = ((U - 12) * 32 / 255) - 23` is the *minimum* SQLite will ever
    /// leave on the page and `K = M + ((P - M) mod (U - 4))` is chosen so the
    /// overflow pages come out exactly full — unless `K` itself would exceed
    /// `X`, in which case only `M` stays. A 4-byte big-endian page number
    /// follows the inline bytes, and each overflow page is a 4-byte "next"
    /// pointer plus up to `U - 4` bytes of content.
    fn cell_payload(
        &self,
        page: &[u8],
        start: usize,
        payload_len: i64,
        visited: &mut BTreeSet<u32>,
    ) -> Result<Vec<u8>, LocalVectorError> {
        // A record can never be longer than the file that holds it, so this
        // both rejects nonsense and bounds the allocation below.
        let total = usize::try_from(payload_len)
            .ok()
            .filter(|len| *len <= self.bytes.len())
            .ok_or_else(|| err(format!("a cell claims a {payload_len}-byte payload")))?;
        let usable = self.usable_size;
        // One copy of the rule, shared with `scan_prefixes`, which needs to
        // stop exactly where the overflow chain starts.
        let inline = self.inline_len(payload_len)?;
        let mut payload = Vec::with_capacity(total);
        payload.extend_from_slice(
            page.get(start..start + inline)
                .ok_or_else(|| err("a cell's inline payload runs past its page"))?,
        );
        if inline == total {
            return Ok(payload);
        }
        let pointer = page
            .get(start + inline..start + inline + 4)
            .ok_or_else(|| err("a spilled cell has no overflow page number"))?;
        let mut next = u32::from_be_bytes([pointer[0], pointer[1], pointer[2], pointer[3]]);
        while payload.len() < total {
            if next == 0 {
                return Err(err(format!(
                    "an overflow chain ended {} bytes early",
                    total - payload.len()
                )));
            }
            if !visited.insert(next) {
                return Err(err(format!(
                    "overflow page {next} is reachable twice (a cycle)"
                )));
            }
            let overflow = self.page(next)?;
            let (head, content) = overflow.split_at(4);
            let wanted = (total - payload.len()).min(usable - 4);
            payload.extend_from_slice(
                content
                    .get(..wanted)
                    .ok_or_else(|| err(format!("overflow page {next} is truncated")))?,
            );
            next = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
        }
        Ok(payload)
    }

    /// Decodes one leaf-table cell's rowid and, when its record reaches that
    /// far, the value of column `index` alone — [`Self::leaf_row`]'s
    /// counterpart for [`Self::scan_column`].
    fn leaf_column(
        &self,
        page: &[u8],
        cell: usize,
        index: usize,
        visited: &mut BTreeSet<u32>,
    ) -> Result<(i64, Option<CellValue>), LocalVectorError> {
        let (payload_len, consumed) = varint(page, cell)?;
        let (rowid, rowid_len) = varint(page, cell + consumed)?;
        let start = cell + consumed + rowid_len;
        let value = self.column_value(page, start, payload_len, index, visited)?;
        Ok((rowid, value))
    }

    /// Reads one column of a leaf-table cell's record without assembling the
    /// payload before it — [`Self::cell_payload`]'s counterpart for
    /// [`Self::scan_column`].
    ///
    /// The record's header (small: a handful of columns' serial types) must
    /// fit the guaranteed-inline region for the fast path below to apply —
    /// every leaf cell keeps at least [`min_inline`] bytes on its page (35 at
    /// the legal minimum page size, ~489 at the 4096-byte size every MBTiles
    /// writer uses), and no schema this reader targets comes close to that
    /// many columns. When it does not fit, this falls back to
    /// [`Self::cell_payload`] for **this one row** — not the whole table — so
    /// every row still decodes correctly; only the memory-bounded shortcut is
    /// skipped for it. Columns before `index` are walked, not read: their
    /// *sizes* come from the header alone, so the bytes between the header
    /// and `index`'s own column are skipped by following the overflow chain's
    /// `next` pointers without copying their content — which is what keeps
    /// this bounded when `index` names the *last* column of a record whose
    /// *first* is a multi-megabyte blob.
    ///
    /// `Ok(None)` covers both a record that does not reach `index` at all (an
    /// `ALTER TABLE ADD COLUMN` row) and a column whose declared size runs
    /// past the payload — unindexable, not an error, the same rule
    /// [`Self::scan_prefixes`] applies to a column past its inline prefix.
    fn column_value(
        &self,
        page: &[u8],
        start: usize,
        payload_len: i64,
        index: usize,
        visited: &mut BTreeSet<u32>,
    ) -> Result<Option<CellValue>, LocalVectorError> {
        let inline_len = self.inline_len(payload_len)?;
        let inline = page
            .get(start..start + inline_len)
            .ok_or_else(|| err("a cell's inline payload runs past its page"))?;
        let Some((serials, header_len)) = header_from_inline(inline) else {
            let payload = self.cell_payload(page, start, payload_len, visited)?;
            return Ok(decode_record(&payload)?.into_iter().nth(index));
        };
        let Some(&target_serial) = serials.get(index) else {
            return Ok(None);
        };
        let total = usize::try_from(payload_len)
            .ok()
            .filter(|len| *len <= self.bytes.len())
            .ok_or_else(|| err(format!("a cell claims a {payload_len}-byte payload")))?;
        let mut body_pos = header_len;
        for serial in &serials[..index] {
            body_pos = body_pos
                .checked_add(serial_size(*serial)?)
                .ok_or_else(|| err("a record declares a column longer than memory"))?;
        }
        let size = serial_size(target_serial)?;
        let Some(end_pos) = body_pos.checked_add(size) else {
            return Ok(None);
        };
        if end_pos > total {
            return Ok(None);
        }
        let bytes = self.record_range(page, start, inline_len, body_pos, size, visited)?;
        Ok(Some(decode_value(target_serial, &bytes)))
    }

    /// The `size` payload bytes starting at `pos` (0-based from the record's
    /// own first byte), fetched from `inline` when the range sits entirely on
    /// the leaf page, or by walking the overflow chain otherwise. The caller
    /// has already checked `pos + size` against the record's total length.
    ///
    /// Pages the range does not touch are walked for their four-byte "next"
    /// pointer alone — never for their content — so a range near the end of a
    /// long chain costs one small copy, not the whole chain.
    fn record_range(
        &self,
        page: &[u8],
        start: usize,
        inline_len: usize,
        pos: usize,
        size: usize,
        visited: &mut BTreeSet<u32>,
    ) -> Result<Vec<u8>, LocalVectorError> {
        let end = pos.saturating_add(size);
        if end <= inline_len {
            return page
                .get(start + pos..start + end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| err("a cell's inline payload runs past its page"));
        }
        let mut out = Vec::with_capacity(size.min(self.bytes.len()));
        if pos < inline_len {
            out.extend_from_slice(
                page.get(start + pos..start + inline_len)
                    .ok_or_else(|| err("a cell's inline payload runs past its page"))?,
            );
        }
        let pointer = page
            .get(start + inline_len..start + inline_len + 4)
            .ok_or_else(|| err("a spilled cell has no overflow page number"))?;
        let mut next = u32::from_be_bytes([pointer[0], pointer[1], pointer[2], pointer[3]]);
        let content_len = self.usable_size - OVERFLOW_POINTER_LEN;
        let mut page_pos = inline_len;
        while page_pos < end {
            if next == 0 {
                return Err(err(
                    "an overflow chain ended before reaching a record's column",
                ));
            }
            if !visited.insert(next) {
                return Err(err(format!(
                    "overflow page {next} is reachable twice (a cycle)"
                )));
            }
            let overflow = self.page(next)?;
            let (head, content) = overflow.split_at(OVERFLOW_POINTER_LEN);
            let page_end = page_pos + content_len;
            if end > page_pos && pos < page_end {
                let local_start = pos.max(page_pos) - page_pos;
                let local_end = end.min(page_end) - page_pos;
                out.extend_from_slice(
                    content
                        .get(local_start..local_end)
                        .ok_or_else(|| err(format!("overflow page {next} is truncated")))?,
                );
            }
            page_pos = page_end;
            next = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
        }
        Ok(out)
    }
}

/// Bytes of a leaf-table cell's header that are not payload: the two varints
/// plus the four-byte overflow pointer, at their maximum widths.
///
/// SQLite's `X = U - 35` for a table leaf is exactly this: the largest cell that
/// still lets four of them share a page.
const TABLE_LEAF_OVERHEAD: usize = 35;

/// Bytes subtracted from `U` before the index fractions are applied.
const INDEX_HEADER_OVERHEAD: usize = 12;

/// Bytes subtracted after the index fractions are applied.
const INDEX_FRACTION_OVERHEAD: usize = 23;

/// Bytes of every overflow page taken by its "next" pointer.
const OVERFLOW_POINTER_LEN: usize = 4;

/// The largest payload an **index** cell keeps entirely inline: SQLite's
/// `X = ((U - 12) * 64 / 255) - 23`.
///
/// The half of the spill rule the v1.3 reader never needed, because it only ever
/// walked table b-trees. An index descent needs it for the opposite reason a
/// table scan needs `X`: not to find where the overflow chain starts, but to
/// know whether a key can be **compared at all**. A key longer than what is
/// guaranteed inline would be compared short — silently returning the wrong
/// row — so an index whose keys can exceed it is refused by name instead.
pub(crate) const fn index_max_inline(usable: usize) -> usize {
    (usable - INDEX_HEADER_OVERHEAD) * 64 / 255 - INDEX_FRACTION_OVERHEAD
}

/// The *minimum* number of payload bytes SQLite leaves on any page, table or
/// index: `M = ((U - 12) * 32 / 255) - 23`.
///
/// Shared by both cell kinds, and the number that makes a prefix scan sound: at
/// the 4096-byte page size every real writer uses it is 489 bytes, far more than
/// a record header plus a few leading integers.
pub(crate) const fn min_inline(usable: usize) -> usize {
    (usable - INDEX_HEADER_OVERHEAD) * 32 / 255 - INDEX_FRACTION_OVERHEAD
}

/// How many bytes of a `total`-byte payload stay inline given the maximum `X`
/// for that cell kind.
///
/// `K = M + ((P - M) mod (U - 4))`, chosen so the overflow pages come out
/// exactly full — unless `K` itself would exceed `X`, in which case only `M`
/// stays. Both spill rules are this function with a different `X`.
const fn fitted_inline(usable: usize, total: usize, max_inline: usize) -> usize {
    if total <= max_inline {
        return total;
    }
    let floor = min_inline(usable);
    let fitted = floor + (total - floor) % (usable - OVERFLOW_POINTER_LEN);
    if fitted <= max_inline { fitted } else { floor }
}

/// How many bytes of a `total`-byte record stay on a **table leaf** page.
///
/// With `U` the usable page size and `P` the payload length, a leaf-table cell
/// keeps everything inline while `P <= X` where `X = U - 35`. Above that it
/// keeps `K = M + ((P - M) mod (U - 4))` — see [`fitted_inline`] — unless that
/// would exceed `X`, in which case only `M` stays.
///
/// Free rather than a method because three readers need the same answer: the
/// resident b-tree walker here, its prefix scan, and the paged MBTiles reader in
/// `crate::mbtiles::paged`, which never has the whole image in hand. One copy of
/// the arithmetic is the only way the two readers can agree byte for byte.
pub(crate) const fn inline_len_for(usable: usize, total: usize) -> usize {
    fitted_inline(usable, total, usable - TABLE_LEAF_OVERHEAD)
}

/// How many bytes of a `total`-byte record stay on an **index** page.
///
/// The twin of [`inline_len_for`] with `X = ((U - 12) * 64 / 255) - 23`
/// ([`index_max_inline`]). Index cells — interior *and* leaf — carry real keys,
/// so this is what an index descent must respect to know where a cell's key
/// ends and its overflow pointer begins.
pub(crate) const fn index_inline_len_for(usable: usize, total: usize) -> usize {
    fitted_inline(usable, total, index_max_inline(usable))
}

/// The cell-content offsets of one page, validated against it.
///
/// The pointer array follows the page header (8 bytes on a leaf, 12 on an
/// interior page) and holds one big-endian `u16` per cell, each an offset
/// **from the start of the page** — page 1 included, whose header sits at
/// 100 but whose offsets are still absolute.
///
/// Free rather than a method for the same reason as [`inline_len_for`]: the
/// paged reader holds one page at a time and has no [`SqliteDb`] to ask.
pub(crate) fn cell_offsets_in(
    page: &[u8],
    base: usize,
    header_len: usize,
    cell_count: usize,
    number: u32,
) -> Result<Vec<usize>, LocalVectorError> {
    let start = base + header_len;
    let array = page
        .get(start..start + cell_count * 2)
        .ok_or_else(|| err(format!("page {number} has a truncated cell pointer array")))?;
    let mut offsets = Vec::with_capacity(cell_count);
    for pointer in array.chunks_exact(2) {
        let offset = usize::from(u16::from_be_bytes([pointer[0], pointer[1]]));
        if offset < start + cell_count * 2 || offset >= page.len() {
            return Err(err(format!(
                "page {number} points a cell at offset {offset}, which is outside it"
            )));
        }
        offsets.push(offset);
    }
    Ok(offsets)
}

/// What [`SqliteDb::scan_prefixes`] hands each row to: its rowid and the inline
/// prefix of its record.
///
/// A named alias rather than the bare closure type because the signature
/// appears in a public-ish method and a `dyn FnMut(i64, &[u8]) -> Result<…>`
/// spelled inline reads as noise at the call site.
pub(crate) type PrefixVisitor<'a> = dyn FnMut(i64, &[u8]) -> Result<(), LocalVectorError> + 'a;

/// What [`SqliteDb::scan_column`] hands each row to: its rowid and the
/// decoded value of the one column requested.
pub(crate) type ColumnVisitor<'a> = dyn FnMut(i64, &CellValue) -> Result<(), LocalVectorError> + 'a;

/// Reads a SQLite variable-length integer at `offset`, returning it and how
/// many bytes it took.
///
/// One to nine bytes, most significant first, each carrying seven bits with the
/// high bit marking "another byte follows" — except the ninth, which
/// contributes all eight of its bits. The result is reinterpreted as a signed
/// 64-bit integer, which is what SQLite stores in it.
///
/// # Errors
///
/// Refuses a varint that runs off the end of `bytes`.
pub(crate) fn varint(bytes: &[u8], offset: usize) -> Result<(i64, usize), LocalVectorError> {
    let mut value: u64 = 0;
    for index in 0..8 {
        let byte = *bytes
            .get(offset + index)
            .ok_or_else(|| err("a varint runs past the end of its page"))?;
        value = (value << 7) | u64::from(byte & 0x7f);
        if byte & 0x80 == 0 {
            return Ok((value as i64, index + 1));
        }
    }
    let byte = *bytes
        .get(offset + 8)
        .ok_or_else(|| err("a varint runs past the end of its page"))?;
    value = (value << 8) | u64::from(byte);
    Ok((value as i64, 9))
}

/// Decodes one record (SQLite's row serialisation) into its column values.
///
/// A record is a varint header length, that many bytes of one varint "serial
/// type" per column, then the column bodies back to back in the same order.
///
/// # Errors
///
/// Refuses a header that does not fit the payload, a body that runs past it,
/// and serial types 10 and 11, which SQLite reserves for its own internal use
/// and which no table row may contain.
pub(crate) fn decode_record(payload: &[u8]) -> Result<Vec<CellValue>, LocalVectorError> {
    decode_record_inner(payload, false)
}

/// Decodes as many of a record's leading columns as `prefix` actually holds.
///
/// The [`decode_record`] twin for
/// [`SqliteDb::scan_prefixes`]'s inline prefix: a spilled cell keeps only its
/// first few hundred bytes on the page, so the *last* column it names is
/// necessarily cut off. Stopping there — rather than refusing the whole
/// record — is what makes indexing the leading integer columns of a blob table
/// free, which is the entire reason the prefix scan exists.
///
/// The header itself must still be complete: it is written before any body and
/// is a couple of dozen bytes, so a truncated one means the prefix is not a
/// record at all.
///
/// # Errors
///
/// The same header refusals as [`decode_record`]. A body that runs past the
/// prefix is *not* an error — it simply ends the returned list.
pub(crate) fn decode_record_prefix(prefix: &[u8]) -> Result<Vec<CellValue>, LocalVectorError> {
    decode_record_inner(prefix, true)
}

/// Shared body of [`decode_record`] and [`decode_record_prefix`].
///
/// `stop_at_end` decides what a body running past the buffer means: the end of
/// what is readable (a prefix) or a malformed record (a whole payload).
fn decode_record_inner(
    payload: &[u8],
    stop_at_end: bool,
) -> Result<Vec<CellValue>, LocalVectorError> {
    let (header_len, consumed) = varint(payload, 0)?;
    let header_len = usize::try_from(header_len)
        .ok()
        .filter(|len| *len >= consumed && *len <= payload.len())
        .ok_or_else(|| err("a record header does not fit its payload"))?;
    let mut serials = Vec::new();
    let mut cursor = consumed;
    while cursor < header_len {
        let (serial, used) = varint(payload, cursor)?;
        serials.push(serial);
        cursor += used;
    }
    if cursor != header_len {
        // The last serial-type varint overran the header's own declared
        // length — real SQLite treats this as corruption (`zIdx > zEndHdr`)
        // rather than reading the overrun bytes as part of both the header
        // and the body that starts right after it.
        return Err(err("a record's header does not end where it declares"));
    }
    let mut values = Vec::with_capacity(serials.len());
    let mut body = header_len;
    for serial in serials {
        // A serial type may claim a body of nearly `usize::MAX` bytes, so the
        // end of the slice is computed with a checked add: on a 32-bit target
        // (which is what `wasm32` is) the plain one would wrap.
        let size = serial_size(serial)?;
        let end = body
            .checked_add(size)
            .ok_or_else(|| err("a record declares a column longer than memory"))?;
        let bytes = match payload.get(body..end) {
            Some(bytes) => bytes,
            None if stop_at_end => break,
            None => return Err(err("a record body runs past its payload")),
        };
        values.push(decode_value(serial, bytes));
        body = end;
    }
    Ok(values)
}

/// Parses a record's header entirely out of `inline`, returning its declared
/// length and each column's serial type — or [`None`] when the header does
/// not end within `inline`.
///
/// That is inconclusive on its own: it may mean the record is merely spilled
/// somewhere the guaranteed-inline region does not reach, or that it is
/// corrupt. [`SqliteDb::column_value`] resolves the ambiguity for the one row
/// in question — by materialising it in full — rather than guessing here.
fn header_from_inline(inline: &[u8]) -> Option<(Vec<i64>, usize)> {
    let (header_len, consumed) = varint(inline, 0).ok()?;
    let header_len = usize::try_from(header_len)
        .ok()
        .filter(|len| *len >= consumed && *len <= inline.len())?;
    let mut serials = Vec::new();
    let mut cursor = consumed;
    while cursor < header_len {
        let (serial, used) = varint(inline, cursor).ok()?;
        serials.push(serial);
        cursor += used;
    }
    (cursor == header_len).then_some((serials, header_len))
}

/// How many body bytes a serial type occupies.
fn serial_size(serial: i64) -> Result<usize, LocalVectorError> {
    match serial {
        0 | 8 | 9 => Ok(0),
        1 => Ok(1),
        2 => Ok(2),
        3 => Ok(3),
        4 => Ok(4),
        5 => Ok(6),
        6 | 7 => Ok(8),
        10 | 11 => Err(err(
            "a record uses a serial type SQLite reserves for its own internal records",
        )),
        other if other >= 12 => usize::try_from((other - 12) / 2)
            .map_err(|_error| err(format!("a record declares a {other}-byte column"))),
        other => Err(err(format!("a record declares serial type {other}"))),
    }
}

/// Turns one serial type and its body bytes into a value.
///
/// `bytes` is exactly [`serial_size`] long, so nothing here can be short.
fn decode_value(serial: i64, bytes: &[u8]) -> CellValue {
    match serial {
        0 => CellValue::Null,
        1..=6 => CellValue::Integer(signed_be(bytes)),
        7 => CellValue::Float(f64::from_bits(signed_be(bytes) as u64)),
        8 => CellValue::Integer(0),
        9 => CellValue::Integer(1),
        other if other % 2 == 0 => CellValue::Blob(bytes.to_vec()),
        _ => CellValue::Text(String::from_utf8_lossy(bytes).into_owned()),
    }
}

/// A big-endian two's-complement integer of 1..=8 bytes, sign-extended.
///
/// The sign extension is the whole point: SQLite stores `-1` as the single byte
/// `0xFF`, and reading it as 255 quietly corrupts every negative id in the file
/// (`gpkg_spatial_ref_sys` has two of them).
fn signed_be(bytes: &[u8]) -> i64 {
    let negative = bytes.first().is_some_and(|byte| byte & 0x80 != 0);
    let mut value: i64 = if negative { -1 } else { 0 };
    for byte in bytes {
        value = (value << 8) | i64::from(*byte);
    }
    value
}

/// One column of a table, as declared.
#[derive(Debug, Clone)]
pub(crate) struct ColumnDef {
    /// The column's name, unquoted.
    pub(crate) name: String,
    /// Its declared type, verbatim and possibly empty (SQLite allows a column
    /// with no type at all).
    pub(crate) declared_type: String,
}

impl ColumnDef {
    /// Whether the column has REAL *affinity* — which decides how its values
    /// read back, not how they were stored.
    ///
    /// SQLite writes a `REAL` value that happens to be integral (`40.0`) using
    /// an *integer* serial type to save space, and restores its type from the
    /// column's affinity on the way out. Skipping that step turns a column of
    /// elevations into a mix of `40` and `40.5`.
    ///
    /// The tests are SQLite's own, in their own precedence order: `INT` wins
    /// over everything, then the text types, then `BLOB`, then the real ones.
    pub(crate) fn has_real_affinity(&self) -> bool {
        let declared = self.declared_type.to_ascii_uppercase();
        if declared.contains("INT")
            || declared.contains("CHAR")
            || declared.contains("CLOB")
            || declared.contains("TEXT")
            || declared.contains("BLOB")
        {
            return false;
        }
        declared.contains("REAL") || declared.contains("FLOA") || declared.contains("DOUB")
    }
}

/// A table's column list as declared by its `CREATE TABLE` statement.
#[derive(Debug, Clone, Default)]
pub(crate) struct TableSchema {
    /// Columns in declaration order — the order a record's values are in.
    pub(crate) columns: Vec<ColumnDef>,
    /// Index of the column that is an alias for the rowid, if any.
    ///
    /// A single-column `INTEGER PRIMARY KEY` *is* the rowid: the record stores
    /// `NULL` in its slot and the real value only exists as the cell's rowid.
    /// Without this substitution every `fid` in a GeoPackage reads as null.
    pub(crate) rowid_alias: Option<usize>,
}

impl TableSchema {
    /// Index of the column named `name`, case-insensitively (SQLite identifiers
    /// are).
    pub(crate) fn column_index(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|column| column.name.eq_ignore_ascii_case(name))
    }
}

/// One lexical token of a column definition.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// A bare word: a keyword, a type name, or an unquoted identifier.
    Word(String),
    /// A quoted identifier or string literal, already unquoted.
    Quoted(String),
    /// A parenthesised group, kept whole so `NUMERIC(10, 2)` and
    /// `DEFAULT (a, b)` cannot be mistaken for two definitions.
    Group,
    /// Anything else (`=`, `-`, …).
    Other,
}

/// Keywords that, as the first word of a comma-separated item, mean the item is
/// a *table* constraint rather than a column.
const TABLE_CONSTRAINTS: [&str; 5] = ["CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN"];

/// Keywords that end a column's type name and start its constraints.
const COLUMN_CONSTRAINTS: [&str; 12] = [
    "CONSTRAINT",
    "PRIMARY",
    "NOT",
    "NULL",
    "UNIQUE",
    "CHECK",
    "DEFAULT",
    "COLLATE",
    "REFERENCES",
    "GENERATED",
    "AS",
    "AUTOINCREMENT",
];

/// Derives a table's schema from its `CREATE TABLE` statement.
///
/// SQLite keeps no machine-readable column list — `sqlite_master` stores the
/// statement text and nothing else — so reading a table by column name means
/// parsing SQL. Only as much of it as that needs: find the parenthesised body,
/// split it on the commas that are at paren depth zero and outside quotes, and
/// take the first token of each item as a column name unless it is a table
/// constraint. Quoted identifiers (`"name ja"`, `` `x` ``, `[x]`) keep their
/// spaces, `VARCHAR(10)` keeps its type, and `DEFAULT (1, 2)` does not become
/// two columns.
///
/// Returns [`None`] when `sql` has no parenthesised body at all — a view, a
/// `CREATE TABLE … AS SELECT`, or something this parser should not guess at.
pub(crate) fn parse_create_table(sql: &str) -> Option<TableSchema> {
    let body = create_table_body(sql)?;
    let without_rowid = sql.get(body.1 + 1..).is_some_and(|tail| {
        tail.to_ascii_uppercase()
            .replace('\n', " ")
            .contains("WITHOUT")
    });
    let mut schema = TableSchema::default();
    // Table-level `PRIMARY KEY (x)` names its column separately.
    let mut table_primary_key: Option<String> = None;
    let mut integer_columns: Vec<String> = Vec::new();
    for item in split_top_level(sql.get(body.0 + 1..body.1)?) {
        let tokens = tokenize(&item);
        let Some(first) = tokens.first() else {
            continue;
        };
        let (name, quoted) = match first {
            Token::Word(word) => (word.clone(), false),
            Token::Quoted(text) => (text.clone(), true),
            _ => continue,
        };
        if !quoted
            && TABLE_CONSTRAINTS
                .iter()
                .any(|key| key == &name.to_ascii_uppercase())
        {
            if name.eq_ignore_ascii_case("PRIMARY") {
                table_primary_key = single_key_column(&item);
            }
            continue;
        }
        let declared_type = declared_type(&tokens);
        if declared_type.eq_ignore_ascii_case("INTEGER") {
            integer_columns.push(name.clone());
            if column_is_primary_key(&tokens) {
                schema.rowid_alias = Some(schema.columns.len());
            }
        }
        schema.columns.push(ColumnDef {
            name,
            declared_type,
        });
    }
    if schema.rowid_alias.is_none()
        && let Some(key) = table_primary_key
        && integer_columns
            .iter()
            .any(|column| column.eq_ignore_ascii_case(&key))
    {
        schema.rowid_alias = schema.column_index(&key);
    }
    if without_rowid {
        // A WITHOUT ROWID table has no rowid to alias; its rows live in an
        // index b-tree, which `scan_table` refuses anyway.
        schema.rowid_alias = None;
    }
    Some(schema)
}

/// The key columns of every index SQLite creates **implicitly** for `sql`, in
/// `sqlite_autoindex_<table>_<n>` order (`n` is 1-based).
///
/// A `UNIQUE` constraint — table-level `UNIQUE (z, x, y)` or a column-level
/// `… UNIQUE` — has no `CREATE INDEX` statement anywhere in the catalogue, so
/// the only description of what its b-tree is keyed by is the `CREATE TABLE`
/// statement itself. Without this, an archive whose one index is a table
/// constraint reads as an archive with no index at all.
///
/// A table-level `PRIMARY KEY (…)` is included too, because on a rowid table it
/// also gets an auto-index; a *column-level* `INTEGER PRIMARY KEY` is not,
/// because it is the rowid and has no index of its own (the same asymmetry
/// [`single_key_column`] documents).
///
/// An entry this parser cannot read cleanly — an expression, a composite it
/// cannot split, a function call — yields **no** columns for that position
/// rather than a guess, so a caller matching an auto-index by number sees an
/// empty list and can refuse by name.
///
/// Each column carries its `ASC`/`DESC`/`COLLATE` exactly as
/// [`parse_create_index`] would for an explicit index — `UNIQUE (z, x, y
/// DESC)` and `UNIQUE (tile_id COLLATE NOCASE)` order their auto-index's
/// leaves exactly as the equivalent `CREATE INDEX` would, so a caller that
/// dropped the decoration would compare a descending or case-folding b-tree
/// as if it were plain ascending `BINARY`.
pub(crate) fn unique_key_columns(sql: &str) -> Vec<Vec<IndexColumn>> {
    let Some(body) = create_table_body(sql) else {
        return Vec::new();
    };
    let Some(inner) = sql.get(body.0 + 1..body.1) else {
        return Vec::new();
    };
    let mut keys = Vec::new();
    for item in split_top_level(inner) {
        let tokens = tokenize(&item);
        let Some(first) = tokens.first() else {
            continue;
        };
        match first {
            // A table constraint: `UNIQUE (…)` or `PRIMARY KEY (…)`.
            Token::Word(word)
                if word.eq_ignore_ascii_case("UNIQUE") || word.eq_ignore_ascii_case("PRIMARY") =>
            {
                keys.push(group_columns(&item));
            }
            // `CONSTRAINT <name> UNIQUE (…)` — the same thing, named.
            Token::Word(word) if word.eq_ignore_ascii_case("CONSTRAINT") => {
                if tokens.iter().any(|token| {
                    matches!(token, Token::Word(word)
                        if word.eq_ignore_ascii_case("UNIQUE")
                            || word.eq_ignore_ascii_case("PRIMARY"))
                }) {
                    keys.push(group_columns(&item));
                }
            }
            // A column definition carrying its own `UNIQUE`. No group to read
            // decoration out of, so this is always plain ascending `BINARY` —
            // which is what such a column always compares as.
            Token::Word(_) | Token::Quoted(_) => {
                let name = match first {
                    Token::Word(word) => word.clone(),
                    Token::Quoted(text) => text.clone(),
                    Token::Group | Token::Other => continue,
                };
                if tokens.iter().skip(1).any(|token| {
                    matches!(token, Token::Word(word) if word.eq_ignore_ascii_case("UNIQUE"))
                }) {
                    keys.push(vec![IndexColumn {
                        name,
                        collation: String::new(),
                        descending: false,
                    }]);
                }
            }
            Token::Group | Token::Other => {}
        }
    }
    keys
}

/// The key columns inside the first parenthesised group of `item`, with each
/// one's `ASC`/`DESC`/`COLLATE` — read by [`parse_index_column`], the same
/// function [`parse_create_index`] uses, so a table-level `UNIQUE (z, x, y
/// DESC)` and an explicit `CREATE INDEX … (z, x, y DESC)` can never disagree
/// about what that auto-index's leaves compare as.
///
/// Empty when there is no group, or when any entry is not a name
/// `parse_index_column` can read cleanly — an expression index is something
/// this reader must refuse rather than guess at.
fn group_columns(item: &str) -> Vec<IndexColumn> {
    let bytes = item.as_bytes();
    let Some(open) = bytes.iter().position(|byte| *byte == b'(') else {
        return Vec::new();
    };
    let end = skip_group(bytes, open);
    let Some(inner) = item.get(open + 1..end.saturating_sub(1)) else {
        return Vec::new();
    };
    let mut columns = Vec::new();
    for part in split_top_level(inner) {
        let Some(column) = parse_index_column(&part) else {
            return Vec::new();
        };
        columns.push(column);
    }
    columns
}

/// One key column of an index, with the two things that change how it compares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndexColumn {
    /// The column's name, unquoted.
    pub(crate) name: String,
    /// The collation named by a `COLLATE` clause, upper-cased; empty means the
    /// column's own default (which for a `TEXT` column with no `COLLATE` in its
    /// definition is `BINARY`).
    ///
    /// Anything other than `BINARY` must be **refused by name**: a descent that
    /// compares bytes down a `NOCASE` index silently returns the wrong row.
    pub(crate) collation: String,
    /// Whether the column is stored descending, which flips the sign of every
    /// comparison at this position. Verified against real SQLite: a `DESC`
    /// index genuinely reorders the leaves.
    pub(crate) descending: bool,
}

/// A `CREATE INDEX` statement, as much of it as a descent needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndexSchema {
    /// The index's own name.
    pub(crate) name: String,
    /// The table it indexes.
    pub(crate) table: String,
    /// Its key columns, in key order.
    pub(crate) columns: Vec<IndexColumn>,
    /// Whether it was declared `UNIQUE`.
    pub(crate) unique: bool,
    /// Whether it is a **partial** index (`… WHERE …`), which covers only some
    /// rows and therefore cannot answer "is this tile present".
    pub(crate) partial: bool,
}

/// Derives an index's shape from its `CREATE INDEX` statement.
///
/// `CREATE [UNIQUE] INDEX [IF NOT EXISTS] [schema.]name ON table (col [COLLATE
/// c] [ASC|DESC], …) [WHERE …]`, with quoted (`"x"`, `` `x` ``, `[x]`) and
/// non-ASCII identifiers handled by the same [`tokenize`] the table parser uses.
///
/// Returns [`None`] the moment anything is not that shape — an expression key, a
/// missing `ON`, a body this parser cannot split. **A parse miss is a refusal,
/// never a guess**: descending an index whose key columns are not what the
/// reader thinks they are returns a plausible, wrong row.
pub(crate) fn parse_create_index(sql: &str) -> Option<IndexSchema> {
    let (open, close) = create_table_body(sql)?;
    // `Token::Other` — the `.` of a schema qualifier, mostly — is dropped, so a
    // qualified `main.i` arrives as two words and the name is simply the last
    // one before `ON`.
    let words: Vec<(String, bool)> = tokenize(sql.get(..open)?)
        .into_iter()
        .filter_map(|token| match token {
            Token::Word(word) => Some((word, false)),
            Token::Quoted(text) => Some((text, true)),
            Token::Group | Token::Other => None,
        })
        .collect();
    let bare = |index: usize, keyword: &str| matches!(words.get(index), Some((word, false)) if word.eq_ignore_ascii_case(keyword));
    if !bare(0, "CREATE") {
        return None;
    }
    let mut cursor = 1;
    let unique = bare(cursor, "UNIQUE");
    if unique {
        cursor += 1;
    }
    if !bare(cursor, "INDEX") {
        return None;
    }
    cursor += 1;
    if bare(cursor, "IF") {
        if !bare(cursor + 1, "NOT") || !bare(cursor + 2, "EXISTS") {
            return None;
        }
        cursor += 3;
    }
    // `ON` separates the (possibly schema-qualified) index name from the table,
    // and is the last thing before the key list.
    let on = words
        .iter()
        .enumerate()
        .skip(cursor)
        .find(|(_, (word, quoted))| !*quoted && word.eq_ignore_ascii_case("ON"))
        .map(|(index, _)| index)?;
    let name_words = on.checked_sub(cursor)?;
    if !(1..=2).contains(&name_words) || on + 2 != words.len() {
        // Zero or three name words, or something after the table name: a shape
        // this parser does not know, and mis-reading it descends the wrong tree.
        return None;
    }
    let name = words.get(on.checked_sub(1)?)?.0.clone();
    let table = words.get(on + 1)?.0.clone();
    let mut columns = Vec::new();
    for item in split_top_level(sql.get(open + 1..close)?) {
        columns.push(parse_index_column(&item)?);
    }
    if name.is_empty() || table.is_empty() || columns.is_empty() {
        return None;
    }
    let partial = sql
        .get(close + 1..)
        .unwrap_or_default()
        .split(|byte: char| !byte.is_ascii_alphanumeric() && byte != '_')
        .any(|word| word.eq_ignore_ascii_case("WHERE"));
    Some(IndexSchema {
        name,
        table,
        columns,
        unique,
        partial,
    })
}

/// One entry of a `CREATE INDEX` key list.
fn parse_index_column(item: &str) -> Option<IndexColumn> {
    let tokens = tokenize(item);
    let name = match tokens.first()? {
        Token::Word(word) => word.clone(),
        Token::Quoted(text) => text.clone(),
        // A group or an operator means an expression key.
        Token::Group | Token::Other => return None,
    };
    let mut collation = String::new();
    let mut descending = false;
    let mut rest = tokens.iter().skip(1);
    while let Some(token) = rest.next() {
        let Token::Word(word) = token else {
            return None;
        };
        if word.eq_ignore_ascii_case("COLLATE") {
            let Some(Token::Word(named)) = rest.next() else {
                return None;
            };
            collation = named.to_ascii_uppercase();
        } else if word.eq_ignore_ascii_case("DESC") {
            descending = true;
        } else if !word.eq_ignore_ascii_case("ASC") {
            return None;
        }
    }
    Some(IndexColumn {
        name,
        collation,
        descending,
    })
}

/// Byte offsets of the outermost `(` and `)` of a `CREATE TABLE` body.
fn create_table_body(sql: &str) -> Option<(usize, usize)> {
    let bytes = sql.as_bytes();
    let mut open = None;
    let mut depth = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => index = skip_quoted(bytes, index, bytes[index]),
            b'[' => index = skip_quoted(bytes, index, b']'),
            b'(' => {
                if depth == 0 {
                    open = Some(index);
                }
                depth += 1;
                index += 1;
            }
            b')' => {
                depth = depth.saturating_sub(1);
                if depth == 0
                    && let Some(start) = open
                {
                    return Some((start, index));
                }
                index += 1;
            }
            _ => index += 1,
        }
    }
    None
}

/// Index just past the quoted run starting at `start`, whose closing delimiter
/// is `close` (doubled delimiters escape themselves, as in SQL).
fn skip_quoted(bytes: &[u8], start: usize, close: u8) -> usize {
    let mut index = start + 1;
    while index < bytes.len() {
        if bytes[index] == close {
            if bytes.get(index + 1) == Some(&close) {
                index += 2;
                continue;
            }
            return index + 1;
        }
        index += 1;
    }
    bytes.len()
}

/// Splits a `CREATE TABLE` body on the commas at paren depth zero, outside
/// quotes.
fn split_top_level(body: &str) -> Vec<String> {
    let bytes = body.as_bytes();
    let mut items = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => index = skip_quoted(bytes, index, bytes[index]),
            b'[' => index = skip_quoted(bytes, index, b']'),
            b'(' => {
                depth += 1;
                index += 1;
            }
            b')' => {
                depth = depth.saturating_sub(1);
                index += 1;
            }
            b',' if depth == 0 => {
                if let Some(item) = body.get(start..index) {
                    items.push(item.trim().to_string());
                }
                index += 1;
                start = index;
            }
            _ => index += 1,
        }
    }
    if let Some(item) = body.get(start..) {
        let item = item.trim();
        if !item.is_empty() {
            items.push(item.to_string());
        }
    }
    items
}

/// Splits one column/constraint definition into tokens.
fn tokenize(item: &str) -> Vec<Token> {
    let bytes = item.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => index += 1,
            b'"' | b'\'' | b'`' | b'[' => {
                let close = if byte == b'[' { b']' } else { byte };
                let end = skip_quoted(bytes, index, close);
                let inner = item
                    .get(index + 1..end.saturating_sub(1))
                    .unwrap_or_default()
                    .replace(
                        core::str::from_utf8(&[close, close]).unwrap_or_default(),
                        core::str::from_utf8(&[close]).unwrap_or_default(),
                    );
                tokens.push(Token::Quoted(inner));
                index = end;
            }
            b'(' => {
                index = skip_group(bytes, index);
                tokens.push(Token::Group);
            }
            // A run of identifier bytes is one word. Every byte of a non-ASCII
            // UTF-8 character is ≥ 0x80 and therefore an identifier byte, so a
            // whole `名前` is accumulated here rather than falling to the
            // catch-all arm below one byte at a time. The run also always
            // starts and ends on a character boundary, since no identifier
            // byte is a UTF-8 continuation byte of a *non*-identifier one.
            _ if is_identifier_byte(byte) => {
                let start = index;
                while index < bytes.len() && is_identifier_byte(bytes[index]) {
                    index += 1;
                }
                tokens.push(Token::Word(
                    item.get(start..index).unwrap_or_default().to_string(),
                ));
            }
            _ => {
                tokens.push(Token::Other);
                index += 1;
            }
        }
    }
    tokens
}

/// Whether `byte` may appear inside an *unquoted* SQLite identifier.
///
/// SQLite's tokenizer is byte-oriented, and its identifier-character class is
/// exactly "ASCII alphanumeric, `_`, `$`, or **any byte ≥ 0x80**". The last
/// clause is not an accident: it is what makes `CREATE TABLE t (名前 TEXT)`
/// legal without quotes. Verified against SQLite 3.37.2 — the statement is
/// accepted verbatim, `sqlite_master.sql` stores it unquoted, and
/// `PRAGMA table_info` reports two columns, `名前` then `geom`.
///
/// Treating those bytes as punctuation instead drops the column from the
/// parsed schema entirely, which slides every later column — the geometry one
/// included — onto the wrong value of each record.
fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'$' || byte >= 0x80
}

/// Index just past the parenthesised group starting at `start`.
fn skip_group(bytes: &[u8], start: usize) -> usize {
    let mut depth = 0usize;
    let mut index = start;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' | b'`' => index = skip_quoted(bytes, index, bytes[index]),
            b'[' => index = skip_quoted(bytes, index, b']'),
            b'(' => {
                depth += 1;
                index += 1;
            }
            b')' => {
                depth -= 1;
                index += 1;
                if depth == 0 {
                    return index;
                }
            }
            _ => index += 1,
        }
    }
    bytes.len()
}

/// The declared type of a column definition: every bare word after the name up
/// to the first constraint keyword, joined with single spaces (`UNSIGNED BIG
/// INT` is one type).
fn declared_type(tokens: &[Token]) -> String {
    let mut words = Vec::new();
    for token in tokens.iter().skip(1) {
        match token {
            Token::Word(word) => {
                if COLUMN_CONSTRAINTS
                    .iter()
                    .any(|key| key.eq_ignore_ascii_case(word))
                {
                    break;
                }
                words.push(word.as_str());
            }
            Token::Group => {}
            _ => break,
        }
    }
    words.join(" ")
}

/// Whether a column definition carries a `PRIMARY KEY` clause that makes it the
/// rowid alias. `PRIMARY KEY DESC` deliberately does not: SQLite stores such a
/// column in the record like any other.
fn column_is_primary_key(tokens: &[Token]) -> bool {
    for (index, token) in tokens.iter().enumerate() {
        let Token::Word(word) = token else { continue };
        if !word.eq_ignore_ascii_case("PRIMARY") {
            continue;
        }
        if !matches!(tokens.get(index + 1), Some(Token::Word(next)) if next.eq_ignore_ascii_case("KEY"))
        {
            continue;
        }
        return !matches!(tokens.get(index + 2), Some(Token::Word(next)) if next.eq_ignore_ascii_case("DESC"));
    }
    false
}

/// The single column named by a table-level `PRIMARY KEY (x …)`, or [`None`]
/// when the key is composite or an expression.
///
/// Sort order and collation do **not** disqualify it. Real SQLite's rowid-alias
/// rule is asymmetric between the two places a primary key can be declared, and
/// the asymmetry was verified against SQLite 3.37.2 with the definitive probe
/// (`UPDATE t SET rowid = 999`, then check whether the declared column moved
/// with it, plus a byte-level dump showing the record stores `NULL` in the
/// aliased slot):
///
/// * *column-level* `fid INTEGER PRIMARY KEY` and `… PRIMARY KEY ASC` alias the
///   rowid; `… PRIMARY KEY DESC` does **not** (see [`column_is_primary_key`]);
/// * *table-level* `PRIMARY KEY (fid)` over an `INTEGER` column aliases the
///   rowid — and so do `(fid ASC)`, `(fid DESC)`, `("fid")`, `("fid" DESC)` and
///   `(fid COLLATE NOCASE)`. The `DESC` exception is column-level only.
///
/// So the shape accepted here is "one name, then only bare words": a comma or
/// any other operator means a composite key or an expression, and a nested
/// group means a function call — none of which this reader should guess at.
fn single_key_column(item: &str) -> Option<String> {
    let bytes = item.as_bytes();
    let open = bytes.iter().position(|byte| *byte == b'(')?;
    let end = skip_group(bytes, open);
    let inner = item.get(open + 1..end.saturating_sub(1))?;
    let tokens = tokenize(inner);
    // Everything after the name may only be a decoration keyword (`ASC`,
    // `DESC`, `COLLATE <name>`); a `Token::Other` is the comma of a composite
    // key or an operator, and a `Token::Group` is a call.
    if tokens
        .iter()
        .skip(1)
        .any(|token| !matches!(token, Token::Word(_)))
    {
        return None;
    }
    match tokens.first()? {
        Token::Word(word) => Some(word.clone()),
        Token::Quoted(text) => Some(text.clone()),
        Token::Group | Token::Other => None,
    }
}
