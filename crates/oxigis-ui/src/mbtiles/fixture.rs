// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Hand-built MBTiles images for this module's tests.
//!
//! Assembled byte by byte on top of [`crate::gpkg_input::fixture`]'s SQLite
//! image builder — the same machinery, and the same review precedent, that
//! proved the b-tree walker against GeoPackages. Reusing it rather than
//! shipping `.mbtiles` binaries keeps the fixtures readable, keeps them out of
//! the wasm bundle, and lets a test produce shapes no writer would emit (a
//! `tiles` view over a *missing* `images` table, a tile whose blob deliberately
//! spills into an overflow chain).
//!
//! Three shapes are built, because between them they are every MBTiles a user
//! will hand OxiGIS:
//!
//! * [`flat_image`] — the specification's `tiles` table;
//! * [`normalized_image`] — `map` ⋈ `images` with `tiles` as a **view**, which
//!   is what tippecanoe and mbutil write and therefore what most real archives
//!   are;
//! * [`spilled_image`] — a flat archive whose one tile is far larger than a
//!   page, so the blob genuinely travels through an overflow chain.
//!
//! # The `fixtures` feature
//!
//! This module is also built under `oxigis-ui`'s default-off `fixtures`
//! feature, whose sole export is [`crate::sample_mbtiles_raster`] — a crate
//! that is **bin-only**, as `oxigis-desktop` is, has its tests inside the
//! binary and cannot reach another crate's `#[cfg(test)]` items, so a feature
//! is the only seam that crosses. The feature is enabled through a
//! *dev*-dependency, which does not reach `cargo build`; nothing here is
//! compiled into `oxigis.exe` or `oxigis_web_bg.wasm`.

#![allow(clippy::unwrap_used, clippy::expect_used)]
// Under `fixtures` this module is compiled with `cfg(test)` OFF, where every
// builder but the one `sample_mbtiles_raster` calls is unused. See
// `gpkg_input::fixture` for the same note.
#![cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the `fixtures` feature exports one builder; the rest are the test suite's"
    )
)]

use crate::gpkg_input::fixture::{Cell, Image, record, varint};

/// Page size every fixture is written at — what GDAL, tippecanoe and mbutil
/// all use, and the size the inline/overflow arithmetic was measured against.
pub(crate) const PAGE_SIZE: usize = 4096;

/// One table of a hand-built archive.
pub(crate) struct Table<'a> {
    /// The name `sqlite_master` records it under.
    pub(crate) name: &'a str,
    /// Its `CREATE TABLE` statement — all the reader has to derive columns
    /// from.
    pub(crate) sql: &'a str,
    /// Its rows as `(rowid, record payload)`.
    pub(crate) rows: Vec<(i64, Vec<u8>)>,
}

/// Assembles an image out of `tables` plus any `views`.
///
/// A view is a `sqlite_master` row with `rootpage = 0` and no storage at all,
/// which is exactly how the normalized schema exposes `tiles` — and exactly the
/// thing a reader that only looks for a `tiles` *root* misses.
pub(crate) fn image_with(tables: &[Table<'_>], views: &[(&str, &str)]) -> Vec<u8> {
    let mut image = Image::new(PAGE_SIZE);
    let mut roots = Vec::with_capacity(tables.len());
    for table in tables {
        // Cells first: a spilling cell allocates its overflow pages here, so
        // the leaf lands on a page number nothing else was going to take.
        let cells: Vec<Vec<u8>> = table
            .rows
            .iter()
            .map(|(rowid, payload)| image.table_cell(*rowid, payload))
            .collect();
        let leaf = image.leaf_page(0, &cells);
        roots.push(image.add_page(leaf));
    }
    let mut master_rows: Vec<(i64, Vec<u8>)> = Vec::new();
    for (index, table) in tables.iter().enumerate() {
        master_rows.push((
            master_rows.len() as i64 + 1,
            record(&[
                Cell::Text("table"),
                Cell::Text(table.name),
                Cell::Text(table.name),
                Cell::Int(i64::from(roots[index])),
                Cell::Text(table.sql),
            ]),
        ));
    }
    for (name, sql) in views {
        master_rows.push((
            master_rows.len() as i64 + 1,
            record(&[
                Cell::Text("view"),
                Cell::Text(name),
                Cell::Text(name),
                Cell::Int(0),
                Cell::Text(sql),
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

/// The `metadata` table for a set of key/value pairs.
pub(crate) fn metadata_table<'a>(entries: &'a [(&'a str, &'a str)]) -> Table<'a> {
    Table {
        name: "metadata",
        sql: "CREATE TABLE metadata (name text, value text)",
        rows: entries
            .iter()
            .enumerate()
            .map(|(index, (key, value))| {
                (
                    index as i64 + 1,
                    record(&[Cell::Text(key), Cell::Text(value)]),
                )
            })
            .collect(),
    }
}

/// The specification's flat archive: one `tiles` table holding the blobs.
///
/// `tiles` are `(zoom_level, tile_column, tile_row, body)` in **MBTiles**
/// addressing, i.e. `tile_row` counted from the south — so a test that asserts
/// an XYZ lookup is genuinely asserting the flip.
pub(crate) fn flat_image(tiles: &[(u8, u32, u32, Vec<u8>)], metadata: &[(&str, &str)]) -> Vec<u8> {
    let rows: Vec<(i64, Vec<u8>)> = tiles
        .iter()
        .enumerate()
        .map(|(index, (z, x, y, body))| {
            (
                index as i64 + 1,
                record(&[
                    Cell::Int(i64::from(*z)),
                    Cell::Int(i64::from(*x)),
                    Cell::Int(i64::from(*y)),
                    Cell::Blob(body),
                ]),
            )
        })
        .collect();
    image_with(
        &[
            Table {
                name: "tiles",
                sql: "CREATE TABLE tiles (zoom_level integer, tile_column integer, \
                      tile_row integer, tile_data blob)",
                rows,
            },
            metadata_table(metadata),
        ],
        &[],
    )
}

/// The deduplicated archive every tippecanoe/mbutil file is: `map` ⋈ `images`,
/// with `tiles` exposed as a view.
///
/// `tiles` are `(zoom_level, tile_column, tile_row, tile_id)`; `images` are
/// `(tile_id, body)`. Two addresses may name one `tile_id`, which is the whole
/// point of the shape and is what the join has to get right.
pub(crate) fn normalized_image(
    map: &[(u8, u32, u32, &str)],
    images: &[(&str, Vec<u8>)],
    metadata: &[(&str, &str)],
) -> Vec<u8> {
    let map_rows: Vec<(i64, Vec<u8>)> = map
        .iter()
        .enumerate()
        .map(|(index, (z, x, y, id))| {
            (
                index as i64 + 1,
                record(&[
                    Cell::Int(i64::from(*z)),
                    Cell::Int(i64::from(*x)),
                    Cell::Int(i64::from(*y)),
                    Cell::Text(id),
                ]),
            )
        })
        .collect();
    // `tile_data` first, `tile_id` second: the order every real writer declares,
    // and the one that forces the reader's full-row path rather than its
    // prefix-scan shortcut.
    let image_rows: Vec<(i64, Vec<u8>)> = images
        .iter()
        .enumerate()
        .map(|(index, (id, body))| {
            (
                index as i64 + 1,
                record(&[Cell::Blob(body), Cell::Text(id)]),
            )
        })
        .collect();
    image_with(
        &[
            Table {
                name: "map",
                sql: "CREATE TABLE map (zoom_level integer, tile_column integer, \
                      tile_row integer, tile_id text)",
                rows: map_rows,
            },
            Table {
                name: "images",
                sql: "CREATE TABLE images (tile_data blob, tile_id text)",
                rows: image_rows,
            },
            metadata_table(metadata),
        ],
        &[(
            "tiles",
            "CREATE VIEW tiles AS SELECT map.zoom_level AS zoom_level, \
             map.tile_column AS tile_column, map.tile_row AS tile_row, \
             images.tile_data AS tile_data FROM map JOIN images ON \
             images.tile_id = map.tile_id",
        )],
    )
}

/// A flat archive whose single tile is far too large to stay on its page.
///
/// `bytes` is chosen well past `usable - 35` (≈ 4061 at [`PAGE_SIZE`]) so the
/// blob genuinely travels through an overflow chain: the leading integer
/// columns must still be readable from the inline prefix, and the body must
/// still come back whole.
pub(crate) fn spilled_image(bytes: usize, metadata: &[(&str, &str)]) -> Vec<u8> {
    let body: Vec<u8> = (0..bytes).map(|index| (index % 251) as u8).collect();
    flat_image(&[(1, 0, 1, body)], metadata)
}

// ---------------------------------------------------------------------------
// Indexed images, for the paged reader (tiles v1.4)
//
// The resident reader never looks at an index — it builds its own — so nothing
// above this line has one. The paged reader reads THROUGH the archive's own
// index, so its fixtures must carry real index b-trees: leaf pages of type 10,
// interior pages of type 2 whose cells hold real separator keys, and the
// `sqlite_autoindex_<table>_<n>` row with `sql = NULL` that a `UNIQUE` table
// constraint produces.
// ---------------------------------------------------------------------------

/// One index of a hand-built archive.
pub(crate) struct IndexSpec<'a> {
    /// The name `sqlite_master` records it under.
    pub(crate) name: &'a str,
    /// The table it indexes.
    pub(crate) table: &'a str,
    /// Its `CREATE INDEX` statement, or [`None`] for one of SQLite's own
    /// auto-indices, which has **no statement at all**.
    pub(crate) sql: Option<&'a str>,
    /// Its key records — `(key columns…, rowid)` — already in key order.
    pub(crate) records: Vec<Vec<u8>>,
}

/// An index record for a tile address: `(zoom, column, row, rowid)`.
pub(crate) fn address_key(z: u8, x: u32, y: u32, rowid: i64) -> Vec<u8> {
    record(&[
        Cell::Int(i64::from(z)),
        Cell::Int(i64::from(x)),
        Cell::Int(i64::from(y)),
        Cell::Int(rowid),
    ])
}

/// An index record for an image id: `(tile_id, rowid)`.
pub(crate) fn image_key(id: &str, rowid: i64) -> Vec<u8> {
    record(&[Cell::Text(id), Cell::Int(rowid)])
}

/// Splits `cells` into groups that each fit one page, largest-first.
///
/// A leaf page holds an 8-byte header, two bytes of pointer per cell and the
/// cells themselves; an interior page's header is 12.
fn chunk_cells(page_size: usize, header_len: usize, cells: &[Vec<u8>]) -> Vec<Vec<Vec<u8>>> {
    let mut chunks: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut current: Vec<Vec<u8>> = Vec::new();
    let mut used = header_len;
    for cell in cells {
        let cost = cell.len() + 2;
        if !current.is_empty() && used + cost > page_size {
            chunks.push(core::mem::take(&mut current));
            used = header_len;
        }
        used += cost;
        current.push(cell.clone());
    }
    if !current.is_empty() || chunks.is_empty() {
        chunks.push(current);
    }
    chunks
}

impl Image {
    /// One leaf-index cell: a payload-length varint then the record.
    ///
    /// The fixtures deliberately keep every index key inline — a spilled key is
    /// a *refusal* in the reader, and the test that pins it builds the spill
    /// explicitly rather than stumbling into one.
    pub(crate) fn index_cell(&self, payload: &[u8]) -> Vec<u8> {
        let mut cell = varint(payload.len() as u64);
        cell.extend_from_slice(payload);
        cell
    }

    /// One interior-index cell: a left-child pointer, then a real separator key.
    pub(crate) fn index_interior_cell(&self, left: u32, payload: &[u8]) -> Vec<u8> {
        let mut cell = left.to_be_bytes().to_vec();
        cell.extend_from_slice(&varint(payload.len() as u64));
        cell.extend_from_slice(payload);
        cell
    }

    /// Lays out a page of `kind` holding `cells`, with `right` for an interior
    /// page.
    pub(crate) fn typed_page(&self, kind: u8, right: Option<u32>, cells: &[Vec<u8>]) -> Vec<u8> {
        let header_len = if right.is_some() { 12 } else { 8 };
        // Every fixture reserves nothing, so the usable area IS the page.
        let mut page = vec![0u8; self.usable()];
        page[0] = kind;
        if let Some(right) = right {
            page[8..12].copy_from_slice(&right.to_be_bytes());
        }
        let mut content = self.usable();
        let mut pointers = Vec::new();
        for cell in cells {
            content -= cell.len();
            page[content..content + cell.len()].copy_from_slice(cell);
            pointers.push(content as u16);
        }
        page[3..5].copy_from_slice(&(cells.len() as u16).to_be_bytes());
        page[5..7].copy_from_slice(&(content as u16).to_be_bytes());
        for (index, pointer) in pointers.iter().enumerate() {
            let at = header_len + index * 2;
            page[at..at + 2].copy_from_slice(&pointer.to_be_bytes());
        }
        page
    }

    /// Writes an index b-tree for `records`, returning its root page.
    ///
    /// One leaf when they fit; otherwise real interior pages whose cells carry
    /// the separator keys, exactly as SQLite lays them out — which is what makes
    /// "an exact match at an interior level is a hit" testable.
    pub(crate) fn add_index(&mut self, records: &[Vec<u8>]) -> u32 {
        let cells: Vec<Vec<u8>> = records
            .iter()
            .map(|record| self.index_cell(record))
            .collect();
        let chunks = chunk_cells(self.usable(), 8, &cells);
        if chunks.len() == 1 {
            let page = self.typed_page(10, None, &chunks[0]);
            return self.add_page(page);
        }
        let mut leaves = Vec::new();
        let mut separators = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let last = chunks.len() - 1;
            if index == last {
                let page = self.typed_page(10, None, chunk);
                leaves.push(self.add_page(page));
            } else {
                let (body, tail) = chunk.split_at(chunk.len().saturating_sub(1));
                let page = self.typed_page(10, None, body);
                leaves.push(self.add_page(page));
                // The separator is the record itself, without the leaf cell's
                // length varint in front — the interior cell writes its own.
                let cell = tail.first().cloned().unwrap_or_default();
                let (length, consumed) = crate::gpkg_input::sqlite::varint(&cell, 0)
                    .expect("the fixture's own cell must parse");
                separators.push(cell[consumed..consumed + length as usize].to_vec());
            }
        }
        let right = *leaves.last().expect("at least one leaf");
        let interior: Vec<Vec<u8>> = leaves
            .iter()
            .take(separators.len())
            .zip(separators.iter())
            .map(|(left, key)| self.index_interior_cell(*left, key))
            .collect();
        let page = self.typed_page(2, Some(right), &interior);
        self.add_page(page)
    }

    /// Writes a table b-tree for `rows`, returning its root page.
    ///
    /// Splits into interior pages the same way [`Self::add_index`] does, so an
    /// archive with more rows than one page holds is a real multi-level tree.
    pub(crate) fn add_table(&mut self, rows: &[(i64, Vec<u8>)]) -> u32 {
        let cells: Vec<(i64, Vec<u8>)> = rows
            .iter()
            .map(|(rowid, payload)| (*rowid, self.table_cell(*rowid, payload)))
            .collect();
        let bare: Vec<Vec<u8>> = cells.iter().map(|(_, cell)| cell.clone()).collect();
        let chunks = chunk_cells(self.usable(), 8, &bare);
        if chunks.len() == 1 {
            let page = self.leaf_page(0, &chunks[0]);
            return self.add_page(page);
        }
        // Recover each chunk's largest rowid, which is the interior separator.
        let mut cursor = 0usize;
        let mut leaves = Vec::new();
        let mut keys = Vec::new();
        for chunk in &chunks {
            let page = self.leaf_page(0, chunk);
            leaves.push(self.add_page(page));
            cursor += chunk.len();
            keys.push(cells.get(cursor.saturating_sub(1)).map_or(0, |(id, _)| *id));
        }
        let right = *leaves.last().expect("at least one leaf");
        let interior: Vec<Vec<u8>> = leaves
            .iter()
            .take(leaves.len() - 1)
            .zip(keys.iter())
            .map(|(left, key)| {
                let mut cell = left.to_be_bytes().to_vec();
                cell.extend_from_slice(&varint(*key as u64));
                cell
            })
            .collect();
        let page = self.typed_page(5, Some(right), &interior);
        self.add_page(page)
    }
}

/// Assembles an image out of `tables`, `views` **and real index b-trees**.
pub(crate) fn indexed_image(
    page_size: usize,
    tables: &[Table<'_>],
    views: &[(&str, &str)],
    indices: &[IndexSpec<'_>],
) -> Vec<u8> {
    let mut image = Image::new(page_size);
    let mut roots = Vec::with_capacity(tables.len());
    for table in tables {
        roots.push(image.add_table(&table.rows));
    }
    let mut index_roots = Vec::with_capacity(indices.len());
    for index in indices {
        index_roots.push(image.add_index(&index.records));
    }
    let mut master_rows: Vec<(i64, Vec<u8>)> = Vec::new();
    for (position, table) in tables.iter().enumerate() {
        master_rows.push((
            master_rows.len() as i64 + 1,
            record(&[
                Cell::Text("table"),
                Cell::Text(table.name),
                Cell::Text(table.name),
                Cell::Int(i64::from(roots[position])),
                Cell::Text(table.sql),
            ]),
        ));
    }
    for (name, sql) in views {
        master_rows.push((
            master_rows.len() as i64 + 1,
            record(&[
                Cell::Text("view"),
                Cell::Text(name),
                Cell::Text(name),
                Cell::Int(0),
                Cell::Text(sql),
            ]),
        ));
    }
    for (position, index) in indices.iter().enumerate() {
        master_rows.push((
            master_rows.len() as i64 + 1,
            record(&[
                Cell::Text("index"),
                Cell::Text(index.name),
                Cell::Text(index.table),
                Cell::Int(i64::from(index_roots[position])),
                index.sql.map_or(Cell::Null, Cell::Text),
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

/// The `CREATE TABLE` a flat archive declares, with no constraint of its own.
pub(crate) const FLAT_TILES_SQL: &str = "CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, \
     tile_data blob)";

/// The same table with the `UNIQUE` **table constraint** whose auto-index has
/// `sql = NULL`.
pub(crate) const FLAT_TILES_UNIQUE_SQL: &str = "CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, \
     tile_data blob, UNIQUE (zoom_level, tile_column, tile_row))";

/// A flat archive with a real `CREATE INDEX` over its addresses.
///
/// `tiles` are `(z, x, y, body)` in **MBTiles** addressing, i.e. `tile_row`
/// counted from the south, so an XYZ lookup really is asserting the flip.
pub(crate) fn indexed_flat_image(
    page_size: usize,
    tiles: &[(u8, u32, u32, Vec<u8>)],
    metadata: &[(&str, &str)],
    autoindex: bool,
) -> Vec<u8> {
    let mut rows = Vec::new();
    let mut keys: Vec<(u8, u32, u32, i64)> = Vec::new();
    for (position, (z, x, y, body)) in tiles.iter().enumerate() {
        let rowid = position as i64 + 1;
        rows.push((
            rowid,
            record(&[
                Cell::Int(i64::from(*z)),
                Cell::Int(i64::from(*x)),
                Cell::Int(i64::from(*y)),
                Cell::Blob(body),
            ]),
        ));
        keys.push((*z, *x, *y, rowid));
    }
    keys.sort_by_key(|(z, x, y, _)| (*z, *x, *y));
    let records: Vec<Vec<u8>> = keys
        .iter()
        .map(|(z, x, y, rowid)| address_key(*z, *x, *y, *rowid))
        .collect();
    let (name, sql) = if autoindex {
        ("sqlite_autoindex_tiles_1", None)
    } else {
        (
            "tile_index",
            Some("CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column, tile_row)"),
        )
    };
    indexed_image(
        page_size,
        &[
            Table {
                name: "tiles",
                sql: if autoindex {
                    FLAT_TILES_UNIQUE_SQL
                } else {
                    FLAT_TILES_SQL
                },
                rows,
            },
            metadata_table(metadata),
        ],
        &[],
        &[IndexSpec {
            name,
            table: "tiles",
            sql,
            records,
        }],
    )
}

/// A normalized archive with both of the indices tippecanoe writes.
pub(crate) fn indexed_normalized_image(
    page_size: usize,
    map: &[(u8, u32, u32, &str)],
    images: &[(&str, Vec<u8>)],
    metadata: &[(&str, &str)],
) -> Vec<u8> {
    let mut map_rows = Vec::new();
    let mut map_keys: Vec<(u8, u32, u32, i64)> = Vec::new();
    for (position, (z, x, y, id)) in map.iter().enumerate() {
        let rowid = position as i64 + 1;
        map_rows.push((
            rowid,
            record(&[
                Cell::Int(i64::from(*z)),
                Cell::Int(i64::from(*x)),
                Cell::Int(i64::from(*y)),
                Cell::Text(id),
            ]),
        ));
        map_keys.push((*z, *x, *y, rowid));
    }
    map_keys.sort_by_key(|(z, x, y, _)| (*z, *x, *y));
    let map_records: Vec<Vec<u8>> = map_keys
        .iter()
        .map(|(z, x, y, rowid)| address_key(*z, *x, *y, *rowid))
        .collect();

    let mut image_rows = Vec::new();
    let mut image_keys: Vec<(String, i64)> = Vec::new();
    for (position, (id, body)) in images.iter().enumerate() {
        let rowid = position as i64 + 1;
        image_rows.push((rowid, record(&[Cell::Blob(body), Cell::Text(id)])));
        image_keys.push(((*id).to_owned(), rowid));
    }
    image_keys.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    let image_records: Vec<Vec<u8>> = image_keys
        .iter()
        .map(|(id, rowid)| image_key(id, *rowid))
        .collect();

    indexed_image(
        page_size,
        &[
            Table {
                name: "map",
                sql: "CREATE TABLE map (zoom_level integer, tile_column integer, \
                      tile_row integer, tile_id text)",
                rows: map_rows,
            },
            Table {
                name: "images",
                sql: "CREATE TABLE images (tile_data blob, tile_id text)",
                rows: image_rows,
            },
            metadata_table(metadata),
        ],
        &[(
            "tiles",
            "CREATE VIEW tiles AS SELECT map.zoom_level AS zoom_level, \
             map.tile_column AS tile_column, map.tile_row AS tile_row, \
             images.tile_data AS tile_data FROM map JOIN images ON \
             images.tile_id = map.tile_id",
        )],
        &[
            IndexSpec {
                name: "map_index",
                table: "map",
                sql: Some(
                    "CREATE UNIQUE INDEX map_index on map (zoom_level, tile_column, tile_row)",
                ),
                records: map_records,
            },
            IndexSpec {
                name: "images_id",
                table: "images",
                sql: Some("CREATE UNIQUE INDEX images_id on images (tile_id)"),
                records: image_records,
            },
        ],
    )
}

/// The metadata a vector archive declares.
pub(crate) fn vector_metadata() -> Vec<(&'static str, &'static str)> {
    vec![
        ("name", "fixture"),
        ("format", "pbf"),
        ("minzoom", "0"),
        ("maxzoom", "2"),
        ("bounds", "-10.0,-5.0,10.0,5.0"),
        ("attribution", "OxiGIS test fixture"),
        (
            "json",
            r#"{"vector_layers":[{"id":"water"},{"id":"roads"}]}"#,
        ),
    ]
}

/// The metadata a raster archive declares.
pub(crate) fn raster_metadata() -> Vec<(&'static str, &'static str)> {
    vec![
        ("name", "raster fixture"),
        ("format", "png"),
        ("minzoom", "0"),
        ("maxzoom", "2"),
    ]
}

/// A genuine 2x2 RGB PNG, hand-assembled the same way
/// `oxigis-render`'s raster fixture does — so the MBTiles decode assertions
/// are about a real image rather than a byte string that happens to be stored.
///
/// The colour deliberately differs from the PMTiles fixture's, so a test that
/// somehow read the wrong archive would say so.
pub(crate) fn tiny_png() -> Vec<u8> {
    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&2u32.to_be_bytes());
    ihdr.extend_from_slice(&2u32.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    push_chunk(&mut png, b"IHDR", &ihdr);
    let mut raw = Vec::new();
    for _row in 0..2 {
        raw.push(0);
        for _pixel in 0..2 {
            raw.extend_from_slice(&[10, 120, 200]);
        }
    }
    let deflated = oxiarc_deflate::zlib_compress(&raw, 6).expect("the fixture must compress");
    push_chunk(&mut png, b"IDAT", &deflated);
    push_chunk(&mut png, b"IEND", &[]);
    png
}

/// Appends one PNG chunk with its CRC.
pub(crate) fn push_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    let length = u32::try_from(data.len()).expect("a tiny chunk");
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc = 0xffff_ffffu32;
    for byte in kind.iter().chain(data.iter()) {
        crc ^= u32::from(*byte);
        for _bit in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    out.extend_from_slice(&(!crc).to_be_bytes());
}
