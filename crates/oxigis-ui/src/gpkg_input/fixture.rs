// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Test-only GeoPackage fixtures, in two tiers.
//!
//! **Tier 1 — real files.** `fixtures/*.gpkg` were written by Python's
//! `sqlite3`, which *is* SQLite, so they exercise the genuine on-disk format:
//! overflow-page chains, interior b-tree pages, `INTEGER PRIMARY KEY` rowid
//! aliasing, quoted identifiers with spaces, mixed SRSs. `fixtures/*.json` is
//! the ground truth emitted alongside them and `fixtures/gen_fixtures.py` is
//! how they were made — kept as provenance, not run by the test suite. They are
//! `include_bytes!`-ed under `#[cfg(test)]` only, so none of it reaches the
//! shipped wasm bundle.
//!
//! **Tier 2 — hand-built images.** No writer produces a cyclic overflow chain
//! or a cell pointer aimed past its page, so those cases are assembled here,
//! byte by byte. The builders deliberately re-implement the inline/overflow
//! split rather than sharing the reader's copy of it: tier 1 is what proves the
//! rule itself is right, and tier 2 only has to produce *a* well-formed image
//! to then damage.

#![allow(clippy::unwrap_used, clippy::expect_used)]
// Under the `fixtures` feature this module is compiled with `cfg(test)` OFF, so
// every builder below that only the test suite calls is unused. That is the
// point of the feature — it exists to export ONE of them to another crate —
// and 24 `never used` warnings would otherwise fail `clippy -D warnings` under
// the `--all-features` gate.
#![cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the `fixtures` feature exports one builder; the rest are the test suite's"
    )
)]

/// `basic.gpkg` — five tables at page size 4096: points with Japanese text and
/// a quoted column name, polygons including one 600-vertex ring that spills
/// into a real overflow chain, a Web Mercator line table, an unsupported-SRS
/// table, and an attributes-only table.
#[cfg(test)]
pub(crate) const BASIC: &[u8] = include_bytes!("fixtures/basic.gpkg");

/// Ground truth for [`BASIC`].
#[cfg(test)]
pub(crate) const BASIC_TRUTH: &str = include_str!("fixtures/basic_truth.json");

/// `paged.gpkg` — 300 rows at page size 512, so the table needs interior b-tree
/// pages and every 37th row overflows on a 600-character attribute.
#[cfg(test)]
pub(crate) const PAGED: &[u8] = include_bytes!("fixtures/paged.gpkg");

/// Ground truth for [`PAGED`].
#[cfg(test)]
pub(crate) const PAGED_TRUTH: &str = include_str!("fixtures/paged_truth.json");

/// `without_rowid.gpkg` — a `WITHOUT ROWID` feature table, which is malformed
/// per the GeoPackage spec and whose root is an index b-tree.
#[cfg(test)]
pub(crate) const WITHOUT_ROWID: &[u8] = include_bytes!("fixtures/without_rowid.gpkg");

/// `unicode.gpkg` — a feature table whose name and three of whose column names
/// are **unquoted** non-ASCII identifiers, keyed by a table-level
/// `PRIMARY KEY (fid DESC)`. Both are legal SQLite that a hand-rolled parser
/// is likely to mis-tokenise; the truth file carries the statement real SQLite
/// stored, so the test can prove the identifiers really are unquoted.
#[cfg(test)]
pub(crate) const UNICODE: &[u8] = include_bytes!("fixtures/unicode.gpkg");

/// Ground truth for [`UNICODE`].
#[cfg(test)]
pub(crate) const UNICODE_TRUTH: &str = include_str!("fixtures/unicode_truth.json");

/// One value to serialise into a record.
#[derive(Debug, Clone)]
pub(crate) enum Cell<'a> {
    /// `NULL` — serial type 0.
    Null,
    /// An integer, written in the narrowest serial type that holds it.
    Int(i64),
    /// A double — serial type 7.
    Real(f64),
    /// UTF-8 text.
    Text(&'a str),
    /// Raw bytes.
    Blob(&'a [u8]),
    /// The literal 0 — serial type 8, which has no body at all.
    Zero,
    /// The literal 1 — serial type 9.
    One,
}

/// Encodes a SQLite varint.
pub(crate) fn varint(value: u64) -> Vec<u8> {
    if value > 0x00ff_ffff_ffff_ffff {
        let mut out = Vec::with_capacity(9);
        let mut shifted = value >> 8;
        let mut seven = [0u8; 8];
        for slot in seven.iter_mut().rev() {
            *slot = (shifted & 0x7f) as u8;
            shifted >>= 7;
        }
        for byte in seven {
            out.push(byte | 0x80);
        }
        out.push((value & 0xff) as u8);
        return out;
    }
    let mut groups = Vec::new();
    let mut rest = value;
    loop {
        groups.push((rest & 0x7f) as u8);
        rest >>= 7;
        if rest == 0 {
            break;
        }
    }
    groups.reverse();
    let last = groups.len() - 1;
    for (index, byte) in groups.iter_mut().enumerate() {
        if index != last {
            *byte |= 0x80;
        }
    }
    groups
}

/// Serialises one row's values as a SQLite record.
pub(crate) fn record(values: &[Cell<'_>]) -> Vec<u8> {
    let mut serials = Vec::new();
    let mut bodies = Vec::new();
    for value in values {
        match value {
            Cell::Null => serials.push(0u64),
            Cell::Zero => serials.push(8),
            Cell::One => serials.push(9),
            Cell::Int(number) => {
                serials.push(6);
                bodies.extend_from_slice(&number.to_be_bytes());
            }
            Cell::Real(number) => {
                serials.push(7);
                bodies.extend_from_slice(&number.to_be_bytes());
            }
            Cell::Text(text) => {
                serials.push(text.len() as u64 * 2 + 13);
                bodies.extend_from_slice(text.as_bytes());
            }
            Cell::Blob(blob) => {
                serials.push(blob.len() as u64 * 2 + 12);
                bodies.extend_from_slice(blob);
            }
        }
    }
    let types: Vec<u8> = serials.iter().flat_map(|serial| varint(*serial)).collect();
    // The header length counts itself, and its own varint may grow by a byte
    // when it does — so try the short form first and widen once if needed.
    let mut header_len = types.len() + 1;
    if varint(header_len as u64).len() > 1 {
        header_len = types.len() + varint((types.len() + 2) as u64).len();
    }
    let mut out = varint(header_len as u64);
    out.extend_from_slice(&types);
    out.extend_from_slice(&bodies);
    out
}

/// A one-BLOB record whose payload is **exactly** `total` bytes, with the blob
/// it holds — for pinning the inline/overflow boundary at a chosen size.
///
/// The blob's contents vary along its length, so a payload reassembled out of
/// order or from the wrong offset does not compare equal by accident.
pub(crate) fn record_of_size(total: usize) -> (Vec<u8>, Vec<u8>) {
    for overhead in 1..8 {
        let blob: Vec<u8> = (0..total - overhead)
            .map(|index| (index % 251) as u8)
            .collect();
        let payload = record(&[Cell::Blob(&blob)]);
        if payload.len() == total {
            return (payload, blob);
        }
    }
    panic!("no single-blob record is exactly {total} bytes long");
}

/// A database image being assembled page by page.
#[derive(Debug)]
pub(crate) struct Image {
    /// Bytes per page.
    page_size: usize,
    /// Bytes reserved at the end of each page.
    reserved: u8,
    /// Every page, page 1 first.
    pages: Vec<Vec<u8>>,
}

impl Image {
    /// An image with one empty page 1.
    pub(crate) fn new(page_size: usize) -> Self {
        Self {
            page_size,
            reserved: 0,
            pages: vec![vec![0u8; page_size]],
        }
    }

    /// Bytes of each page a b-tree may use.
    pub(crate) fn usable(&self) -> usize {
        self.page_size - usize::from(self.reserved)
    }

    /// Appends a page, returning its 1-based number.
    pub(crate) fn add_page(&mut self, page: Vec<u8>) -> u32 {
        assert_eq!(
            page.len(),
            self.page_size,
            "a page must be exactly one page"
        );
        self.pages.push(page);
        self.pages.len() as u32
    }

    /// Replaces page 1 (whose first 100 bytes [`Self::finish`] overwrites with
    /// the database header).
    pub(crate) fn set_page1(&mut self, page: Vec<u8>) {
        assert_eq!(
            page.len(),
            self.page_size,
            "a page must be exactly one page"
        );
        self.pages[0] = page;
    }

    /// Writes `data` as an overflow chain, returning its first page number.
    ///
    /// Pages are allocated in chain order, so a caller that knows how many
    /// pages preceded the call knows exactly which page the chain starts at —
    /// which is how the cycle and truncation tests find a pointer to damage.
    pub(crate) fn add_overflow(&mut self, data: &[u8]) -> u32 {
        let capacity = self.usable() - 4;
        let chunks: Vec<&[u8]> = data.chunks(capacity).collect();
        let first = self.pages.len() as u32 + 1;
        for (index, chunk) in chunks.iter().enumerate() {
            let next = if index + 1 == chunks.len() {
                0
            } else {
                first + index as u32 + 1
            };
            let mut page = vec![0u8; self.page_size];
            page[..4].copy_from_slice(&next.to_be_bytes());
            page[4..4 + chunk.len()].copy_from_slice(chunk);
            self.add_page(page);
        }
        first
    }

    /// Lays out an interior-table page: `children` are its left-child pointers
    /// (each with a rowid varint after it) and `right` its rightmost pointer.
    pub(crate) fn interior_page(&self, base: usize, right: u32, children: &[u32]) -> Vec<u8> {
        let mut page = vec![0u8; self.page_size];
        page[base] = 5;
        page[base + 8..base + 12].copy_from_slice(&right.to_be_bytes());
        let mut content = self.usable();
        let mut pointers = Vec::new();
        for (index, child) in children.iter().enumerate() {
            let mut cell = child.to_be_bytes().to_vec();
            cell.extend_from_slice(&varint(index as u64 + 1));
            content -= cell.len();
            page[content..content + cell.len()].copy_from_slice(&cell);
            pointers.push(content as u16);
        }
        page[base + 3..base + 5].copy_from_slice(&(children.len() as u16).to_be_bytes());
        page[base + 5..base + 7].copy_from_slice(&(content as u16).to_be_bytes());
        for (index, pointer) in pointers.iter().enumerate() {
            let at = base + 12 + index * 2;
            page[at..at + 2].copy_from_slice(&pointer.to_be_bytes());
        }
        page
    }

    /// Builds one leaf-table cell for `payload`, spilling into overflow pages
    /// by the same rule SQLite applies.
    pub(crate) fn table_cell(&mut self, rowid: i64, payload: &[u8]) -> Vec<u8> {
        let usable = self.usable();
        let max_inline = usable - 35;
        let inline = if payload.len() <= max_inline {
            payload.len()
        } else {
            let min_inline = ((usable - 12) * 32 / 255) - 23;
            let fitted = min_inline + ((payload.len() - min_inline) % (usable - 4));
            if fitted <= max_inline {
                fitted
            } else {
                min_inline
            }
        };
        let mut cell = varint(payload.len() as u64);
        cell.extend_from_slice(&varint(rowid as u64));
        cell.extend_from_slice(&payload[..inline]);
        if inline < payload.len() {
            let first = self.add_overflow(&payload[inline..]);
            cell.extend_from_slice(&first.to_be_bytes());
        }
        cell
    }

    /// Lays out a leaf-table page holding `cells`, whose b-tree header starts
    /// at `base` (100 on page 1, 0 elsewhere).
    pub(crate) fn leaf_page(&self, base: usize, cells: &[Vec<u8>]) -> Vec<u8> {
        let mut page = vec![0u8; self.page_size];
        page[base] = 13;
        let mut content = self.usable();
        let mut pointers = Vec::new();
        for cell in cells {
            content -= cell.len();
            page[content..content + cell.len()].copy_from_slice(cell);
            pointers.push(content as u16);
        }
        page[base + 3..base + 5].copy_from_slice(&(cells.len() as u16).to_be_bytes());
        page[base + 5..base + 7].copy_from_slice(&(content as u16).to_be_bytes());
        for (index, pointer) in pointers.iter().enumerate() {
            let at = base + 8 + index * 2;
            page[at..at + 2].copy_from_slice(&pointer.to_be_bytes());
        }
        page
    }

    /// The whole image, with a valid database header written over the start of
    /// page 1.
    pub(crate) fn finish(self) -> Vec<u8> {
        let page_count = self.pages.len() as u32;
        let mut bytes: Vec<u8> = self.pages.into_iter().flatten().collect();
        bytes[..16].copy_from_slice(b"SQLite format 3\0");
        let stored = if self.page_size == 65536 {
            1u16
        } else {
            self.page_size as u16
        };
        bytes[16..18].copy_from_slice(&stored.to_be_bytes());
        bytes[18] = 1;
        bytes[19] = 1;
        bytes[20] = self.reserved;
        bytes[21] = 64;
        bytes[22] = 32;
        bytes[23] = 32;
        bytes[24..28].copy_from_slice(&1u32.to_be_bytes());
        bytes[28..32].copy_from_slice(&page_count.to_be_bytes());
        bytes[44..48].copy_from_slice(&4u32.to_be_bytes());
        bytes[56..60].copy_from_slice(&1u32.to_be_bytes());
        bytes[92..96].copy_from_slice(&1u32.to_be_bytes());
        bytes[96..100].copy_from_slice(&3_045_000u32.to_be_bytes());
        bytes
    }
}

/// One table of a hand-built image.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TableSpec<'a> {
    /// The name `sqlite_master` records it under.
    pub(crate) name: &'a str,
    /// Its `CREATE TABLE` statement, which is all the reader has to derive a
    /// column list from.
    pub(crate) sql: &'a str,
    /// Its rows as `(rowid, record payload)` pairs — build the payloads with
    /// [`record`].
    pub(crate) rows: &'a [(i64, Vec<u8>)],
}

/// A database holding several tables: page 1 is `sqlite_master`, then one leaf
/// page per table in the order given, then any overflow pages.
///
/// Every root is allocated before any table's cells are built, so no overflow
/// page can take a number a later root was going to use.
pub(crate) fn multi_table_image(page_size: usize, tables: &[TableSpec<'_>]) -> Vec<u8> {
    let mut image = Image::new(page_size);
    let roots: Vec<u32> = tables
        .iter()
        .map(|_| image.add_page(vec![0u8; page_size]))
        .collect();
    let mut master_rows: Vec<(i64, Vec<u8>)> = Vec::new();
    for (index, table) in tables.iter().enumerate() {
        let cells: Vec<Vec<u8>> = table
            .rows
            .iter()
            .map(|(rowid, payload)| image.table_cell(*rowid, payload))
            .collect();
        let leaf = image.leaf_page(0, &cells);
        let root = roots[index];
        image.pages[root as usize - 1] = leaf;
        master_rows.push((
            index as i64 + 1,
            record(&[
                Cell::Text("table"),
                Cell::Text(table.name),
                Cell::Text(table.name),
                Cell::Int(i64::from(root)),
                Cell::Text(table.sql),
            ]),
        ));
    }
    let master_cells: Vec<Vec<u8>> = master_rows
        .iter()
        .map(|(rowid, payload)| image.table_cell(*rowid, payload))
        .collect();
    let page1 = image.leaf_page(100, &master_cells);
    image.set_page1(page1);
    image.finish()
}

/// A one-table database: page 1 is `sqlite_master`, page 2 the table's leaf,
/// and any overflow follows.
///
/// `rows` are `(rowid, record payload)` pairs — build the payloads with
/// [`record`].
pub(crate) fn one_table_image(
    page_size: usize,
    name: &str,
    sql: &str,
    rows: &[(i64, Vec<u8>)],
) -> Vec<u8> {
    multi_table_image(page_size, &[TableSpec { name, sql, rows }])
}

/// A minimal but complete GeoPackage: `gpkg_contents`,
/// `gpkg_geometry_columns` and one feature table `t`, with the three things a
/// hostile file would tamper with left to the caller.
///
/// `srs_id` is the cell of the geometry-column registration the CRS policy is
/// decided from — declared `INTEGER NOT NULL` by the spec, but a file
/// assembled as bytes never went through SQLite's affinity coercion and can
/// put any storage class there. `table_sql` and `row` are `t`'s declaration
/// and its single record, which are deliberately *not* forced to agree: a
/// record holding more values than the statement declares columns is the
/// shape a mis-parsed `CREATE TABLE` produces.
pub(crate) fn geopackage_image(srs_id: Cell<'_>, table_sql: &str, row: &[Cell<'_>]) -> Vec<u8> {
    multi_table_image(
        4096,
        &[
            TableSpec {
                name: "gpkg_contents",
                sql: "CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, \
                      data_type TEXT NOT NULL)",
                rows: &[(1, record(&[Cell::Text("t"), Cell::Text("features")]))],
            },
            TableSpec {
                name: "gpkg_geometry_columns",
                sql: "CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, \
                      column_name TEXT NOT NULL, geometry_type_name TEXT NOT NULL, \
                      srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL)",
                rows: &[(
                    1,
                    record(&[
                        Cell::Text("t"),
                        Cell::Text("geom"),
                        Cell::Text("POINT"),
                        srs_id,
                        Cell::Zero,
                        Cell::Zero,
                    ]),
                )],
            },
            TableSpec {
                name: "t",
                sql: table_sql,
                rows: &[(1, record(row))],
            },
        ],
    )
}

/// A GeoPackage whose catalogue lists one *attributes* table and no feature
/// table at all.
///
/// The "nothing to refuse" case, as distinct from "everything was refused" —
/// the two produce the same empty layer list and must not produce the same
/// message.
pub(crate) fn attributes_only_image() -> Vec<u8> {
    one_table_image(
        512,
        "gpkg_contents",
        "CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL)",
        &[(1, record(&[Cell::Text("notes"), Cell::Text("attributes")]))],
    )
}

/// A GeoPackage geometry blob: magic, version 0, `flags`, `srs_id`, envelope,
/// WKB.
pub(crate) fn gp_blob(flags: u8, srs_id: i32, envelope: &[u8], wkb: &[u8]) -> Vec<u8> {
    let mut blob = vec![b'G', b'P', 0, flags];
    if flags & 0x01 == 0 {
        blob.extend_from_slice(&srs_id.to_be_bytes());
    } else {
        blob.extend_from_slice(&srs_id.to_le_bytes());
    }
    blob.extend_from_slice(envelope);
    blob.extend_from_slice(wkb);
    blob
}

/// A little-endian WKB point.
pub(crate) fn wkb_point(x: f64, y: f64) -> Vec<u8> {
    let mut wkb = vec![1u8];
    wkb.extend_from_slice(&1u32.to_le_bytes());
    wkb.extend_from_slice(&x.to_le_bytes());
    wkb.extend_from_slice(&y.to_le_bytes());
    wkb
}

/// A big-endian WKB point — the byte order half the world's WKB is in.
pub(crate) fn wkb_point_be(x: f64, y: f64) -> Vec<u8> {
    let mut wkb = vec![0u8];
    wkb.extend_from_slice(&1u32.to_be_bytes());
    wkb.extend_from_slice(&x.to_be_bytes());
    wkb.extend_from_slice(&y.to_be_bytes());
    wkb
}

/// A little-endian WKB line string.
pub(crate) fn wkb_line(points: &[(f64, f64)]) -> Vec<u8> {
    let mut wkb = vec![1u8];
    wkb.extend_from_slice(&2u32.to_le_bytes());
    wkb.extend_from_slice(&(points.len() as u32).to_le_bytes());
    for (x, y) in points {
        wkb.extend_from_slice(&x.to_le_bytes());
        wkb.extend_from_slice(&y.to_le_bytes());
    }
    wkb
}
