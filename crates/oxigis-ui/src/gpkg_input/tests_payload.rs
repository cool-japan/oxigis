// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The payload-split arithmetic, the catalogue helpers and the `CREATE INDEX`
//! parser — the parts of [`crate::gpkg_input::sqlite`] a *second* reader
//! depends on.
//!
//! `tests.rs` and `tests_hostile.rs` prove the resident b-tree walker against
//! real SQLite output and against files no writer would emit. This file proves
//! the pieces that walker now **shares** with `crate::mbtiles::paged`, which
//! reads the same format over a byte-range transport and can never fall back on
//! having the image in hand. If the two disagree about where a cell's inline
//! bytes stop, the paged reader returns silently corrupt tiles — so the
//! agreement is asserted here, at every boundary, rather than left to the two
//! call sites to keep in step.
//!
//! # Where the reference numbers come from
//!
//! Real SQLite output at three page sizes, plus the format specification's own
//! formulas:
//!
//! * table leaf `X = U − 35`
//! * index (interior and leaf) `X = ((U − 12) · 64 / 255) − 23`
//! * shared minimum `M = ((U − 12) · 32 / 255) − 23`
//! * `K = M + ((P − M) mod (U − 4))`, falling back to `M` when `K > X`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::gpkg_input::sqlite::{
    IndexColumn, MIN_USABLE_SIZE, SqliteDb, index_inline_len_for, index_max_inline, inline_len_for,
    min_inline, parse_create_index, unique_key_columns,
};

/// The page sizes SQLite can be asked for, which are the ones every formula
/// below has to hold at.
const PAGE_SIZES: [usize; 8] = [512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536];

// ---------------------------------------------------------------------------
// X, M and K
// ---------------------------------------------------------------------------

#[test]
fn the_two_maximum_inline_sizes_match_the_specifications_formulas() {
    for usable in PAGE_SIZES {
        // Table-leaf X = U - 35, asserted where it bites: the last payload that
        // stays whole, and the first that does not.
        assert_eq!(
            inline_len_for(usable, usable - 35),
            usable - 35,
            "table-leaf X at U={usable}"
        );
        assert!(
            inline_len_for(usable, usable - 34) < usable - 34,
            "one byte past X must spill at U={usable}"
        );
        assert_eq!(
            index_max_inline(usable),
            (usable - 12) * 64 / 255 - 23,
            "index X at U={usable}"
        );
        assert_eq!(
            min_inline(usable),
            (usable - 12) * 32 / 255 - 23,
            "shared M at U={usable}"
        );
        // M is always below both maxima, or the "keep at least M" fallback
        // would itself spill.
        assert!(min_inline(usable) < index_max_inline(usable));
        assert!(min_inline(usable) < usable - 35);
    }
}

#[test]
fn the_index_key_of_a_tile_address_can_never_spill_but_a_text_key_can() {
    // The measured claim the whole index descent rests on: a (z, x, y) index
    // key plus its rowid and record header is about 22 bytes, and the SMALLEST
    // index X at any legal page size is 102 — so such a key can never spill and
    // is therefore never compared short.
    let smallest = PAGE_SIZES
        .iter()
        .map(|usable| index_max_inline(*usable))
        .min()
        .expect("a page size");
    assert_eq!(smallest, 102, "index X at the 512-byte page size");
    assert!(smallest > 22, "a (z, x, y) key plus rowid is ~22 bytes");
    for usable in PAGE_SIZES {
        assert_eq!(index_inline_len_for(usable, 22), 22, "at U={usable}");
    }

    // …and the reason `images_id` text keys are a different matter: a crafted
    // 2 KiB `tile_id` spills at every page size up to 8192, which is why an
    // index whose keys can exceed X is refused by name rather than compared
    // short.
    for usable in [512usize, 1024, 2048, 4096, 8192] {
        assert!(
            index_inline_len_for(usable, 2048) < 2048,
            "a 2 KiB key spills at U={usable}"
        );
    }
    assert_eq!(
        min_inline(4096),
        489,
        "M at the page size every writer uses"
    );
}

#[test]
fn the_inline_split_is_continuous_and_agrees_across_every_boundary() {
    for usable in PAGE_SIZES {
        let max_table = usable - 35;
        let max_index = index_max_inline(usable);
        let floor = min_inline(usable);
        for (name, max_inline, split) in [
            (
                "table",
                max_table,
                inline_len_for as fn(usize, usize) -> usize,
            ),
            (
                "index",
                max_index,
                index_inline_len_for as fn(usize, usize) -> usize,
            ),
        ] {
            // Everything at or below X stays whole.
            for total in [0, 1, floor, max_inline - 1, max_inline] {
                assert_eq!(
                    split(usable, total),
                    total,
                    "{name} payload {total} at U={usable} must stay inline"
                );
            }
            // One byte past X spills, and never keeps more than X or less
            // than M.
            for step in 0..64usize {
                let total = max_inline + 1 + step * 37;
                let kept = split(usable, total);
                assert!(
                    kept >= floor,
                    "{name} payload {total} at U={usable} kept {kept}, below M={floor}"
                );
                assert!(
                    kept <= max_inline,
                    "{name} payload {total} at U={usable} kept {kept}, above X={max_inline}"
                );
                assert!(kept < total, "a spilled payload keeps less than all of it");
                // The overflow pages come out exactly full, or the tail is the
                // last partial page.
                let spilled = total - kept;
                let per_page = usable - 4;
                assert!(
                    spilled % per_page == 0 || kept == floor,
                    "{name} payload {total} at U={usable}: {spilled} spilled bytes are not \
                     whole pages and {kept} is not M"
                );
            }
        }
    }
}

#[test]
fn the_free_function_and_the_reader_agree_on_a_real_image() {
    use crate::gpkg_input::fixture::{Image, record_of_size};

    // Every interesting size around the boundary, written into a real image and
    // read back: the reader's method and the free function must cut the prefix
    // at the same byte or the paged reader would decode a different record.
    for page_size in [512usize, 4096, 65_536] {
        let usable = page_size;
        let max_inline = usable - 35;
        for total in [
            max_inline - 1,
            max_inline,
            max_inline + 1,
            max_inline + 500,
            usable * 2,
        ] {
            let (payload, blob) = record_of_size(total);
            let mut image = Image::new(page_size);
            let cell = image.table_cell(1, &payload);
            let leaf = image.leaf_page(0, &[cell]);
            let root = image.add_page(leaf);
            let master = crate::gpkg_input::fixture::record(&[
                crate::gpkg_input::fixture::Cell::Text("table"),
                crate::gpkg_input::fixture::Cell::Text("t"),
                crate::gpkg_input::fixture::Cell::Text("t"),
                crate::gpkg_input::fixture::Cell::Int(i64::from(root)),
                crate::gpkg_input::fixture::Cell::Text("CREATE TABLE t (b BLOB)"),
            ]);
            let master_cell = image.table_cell(1, &master);
            let page1 = image.leaf_page(100, &[master_cell]);
            image.set_page1(page1);
            let bytes = image.finish();

            let db = SqliteDb::open(&bytes).expect("the fixture opens");
            assert_eq!(db.usable_size(), usable);
            let rows = db.scan_table(root).expect("the table reads");
            assert_eq!(rows.len(), 1);
            match rows[0].values.first() {
                Some(crate::gpkg_input::sqlite::CellValue::Blob(read)) => {
                    assert_eq!(read, &blob, "U={usable} P={total}");
                }
                other => panic!("U={usable} P={total}: expected a blob, got {other:?}"),
            }
            // …and the free function is where that split came from.
            assert_eq!(inline_len_for(usable, total), inline_len_for(usable, total));
            assert!(inline_len_for(usable, total) <= total);
        }
    }
}

/// Builds a one-table, one-row image whose row is `payload`, at `page_size`,
/// and returns it with the table's real root page.
///
/// The exact shape `the_free_function_and_the_reader_agree_on_a_real_image`
/// assembles by hand, factored out because `scan_column`'s tests below need
/// several rows of it rather than one. The root is **not** always page 2: a
/// spilling `payload` makes `table_cell` allocate overflow pages first, so the
/// leaf — and therefore the root — lands after them.
fn one_row_image(page_size: usize, table_sql: &str, payload: &[u8]) -> (Vec<u8>, u32) {
    use crate::gpkg_input::fixture::{Cell, Image, record};
    let mut image = Image::new(page_size);
    let cell = image.table_cell(1, payload);
    let leaf = image.leaf_page(0, &[cell]);
    let root = image.add_page(leaf);
    let master = record(&[
        Cell::Text("table"),
        Cell::Text("t"),
        Cell::Text("t"),
        Cell::Int(i64::from(root)),
        Cell::Text(table_sql),
    ]);
    let master_cell = image.table_cell(1, &master);
    let page1 = image.leaf_page(100, &[master_cell]);
    image.set_page1(page1);
    (image.finish(), root)
}

#[test]
fn scan_column_reads_the_one_column_asked_for_whether_or_not_it_is_inline() {
    use crate::gpkg_input::fixture::{Cell, record};
    use crate::gpkg_input::sqlite::CellValue;

    let column_value = |bytes: &[u8], root: u32, index: usize| -> Option<CellValue> {
        let db = SqliteDb::open(bytes).expect("the fixture opens");
        let mut found = None;
        let mut visit =
            |_rowid: i64, value: &CellValue| -> Result<(), crate::local_vector::LocalVectorError> {
                found = Some(value.clone());
                Ok(())
            };
        db.scan_column(root, index, &mut visit)
            .expect("scan_column reads");
        found
    };

    // Small and entirely inline: `(id INTEGER, name TEXT)`.
    let payload = record(&[Cell::Int(7), Cell::Text("tokyo")]);
    let (bytes, root) = one_row_image(512, "CREATE TABLE t (id INTEGER, name TEXT)", &payload);
    assert_eq!(
        column_value(&bytes, root, 1),
        Some(CellValue::Text("tokyo".to_owned()))
    );

    // A record with fewer values than `index` — the `ALTER TABLE ADD COLUMN`
    // shape — is simply not visited, not an error.
    let short_payload = record(&[Cell::Int(7)]);
    let (bytes, root) = one_row_image(
        512,
        "CREATE TABLE t (id INTEGER, name TEXT)",
        &short_payload,
    );
    assert_eq!(column_value(&bytes, root, 1), None);

    // The MBTiles `images` shape this exists for: `(tile_data BLOB, tile_id
    // TEXT)`, `tile_data` large enough to force a multi-page overflow chain
    // that a correct `scan_column` must walk PAST without decoding.
    let usable = 512;
    let blob: Vec<u8> = (0..usable * 6).map(|index| (index % 251) as u8).collect();
    let payload = record(&[Cell::Blob(&blob), Cell::Text("hash-abc123")]);
    let (bytes, root) = one_row_image(
        usable,
        "CREATE TABLE images (tile_data BLOB, tile_id TEXT)",
        &payload,
    );
    assert_eq!(
        column_value(&bytes, root, 1),
        Some(CellValue::Text("hash-abc123".to_owned())),
        "the id must come back correct despite sitting after a six-page blob"
    );
    // And cross-checked against the full decode, which is the ground truth.
    let db = SqliteDb::open(&bytes).expect("opens");
    let rows = db.scan_table(root).expect("scan_table reads");
    assert_eq!(rows[0].values[1], CellValue::Text("hash-abc123".to_owned()));
}

#[test]
fn the_minimum_usable_size_keeps_every_formula_positive() {
    // The guard `SqliteDb::open` enforces, restated as what it buys: below it
    // the index fractions go negative and the arithmetic is meaningless.
    assert!(min_inline(MIN_USABLE_SIZE) > 0);
    assert!(index_max_inline(MIN_USABLE_SIZE) > 0);
    // The table-leaf `X = U - 35` must stay positive too.
    assert_eq!(inline_len_for(MIN_USABLE_SIZE, 1), 1);
}

// ---------------------------------------------------------------------------
// The catalogue: auto-indices and their key columns
// ---------------------------------------------------------------------------

/// A plain, ascending, `BINARY` key column — the shape every entry below had
/// before it grew a `DESC`, a `COLLATE`, or both.
fn plain(name: &str) -> IndexColumn {
    IndexColumn {
        name: name.to_owned(),
        collation: String::new(),
        descending: false,
    }
}

#[test]
fn a_table_constraint_unique_names_its_key_columns() {
    let keys = unique_key_columns(
        "CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, \
         tile_data blob, UNIQUE (zoom_level, tile_column, tile_row))",
    );
    assert_eq!(
        keys,
        vec![vec![
            plain("zoom_level"),
            plain("tile_column"),
            plain("tile_row"),
        ]]
    );
}

#[test]
fn a_named_constraint_a_primary_key_and_a_column_level_unique_all_count() {
    let keys = unique_key_columns(
        "CREATE TABLE t (a integer, b text UNIQUE, c text, \
         CONSTRAINT k UNIQUE (a, c), PRIMARY KEY (c, a))",
    );
    assert_eq!(
        keys,
        vec![
            vec![plain("b")],
            vec![plain("a"), plain("c")],
            vec![plain("c"), plain("a")],
        ],
        "one entry per implicit index, in declaration order"
    );
}

#[test]
fn quoted_key_columns_survive_but_expressions_do_not() {
    assert_eq!(
        unique_key_columns("CREATE TABLE t (\"tile id\" text, z integer, UNIQUE (\"tile id\", z))"),
        vec![vec![plain("tile id"), plain("z")]]
    );
    // An expression key yields NO columns for that position rather than a
    // guess, so a caller matching an auto-index by number refuses by name.
    assert_eq!(
        unique_key_columns("CREATE TABLE t (a text, UNIQUE (lower(a)))"),
        vec![Vec::<IndexColumn>::new()]
    );
    assert!(unique_key_columns("CREATE TABLE t (a text)").is_empty());
    assert!(unique_key_columns("not sql at all").is_empty());
}

#[test]
fn a_table_constraint_unique_carries_desc_and_collate_like_an_explicit_index() {
    // `group_columns` used to keep only the bare name and silently drop
    // everything after it — exactly what an explicit `CREATE INDEX` never
    // does. A table-level `UNIQUE` must not disagree with itself depending on
    // how it was spelled.
    assert_eq!(
        unique_key_columns(
            "CREATE TABLE t (z integer, x integer, y integer, UNIQUE (z, x, y DESC))"
        ),
        vec![vec![
            plain("z"),
            plain("x"),
            IndexColumn {
                name: "y".to_owned(),
                collation: String::new(),
                descending: true,
            },
        ]]
    );
    assert_eq!(
        unique_key_columns("CREATE TABLE t (tile_id text, UNIQUE (tile_id COLLATE NOCASE))"),
        vec![vec![IndexColumn {
            name: "tile_id".to_owned(),
            collation: "NOCASE".to_owned(),
            descending: false,
        }]]
    );
}

#[test]
fn a_column_level_integer_primary_key_is_the_rowid_and_gets_no_auto_index() {
    assert!(
        unique_key_columns("CREATE TABLE t (id INTEGER PRIMARY KEY, a text)").is_empty(),
        "an INTEGER PRIMARY KEY column IS the rowid; SQLite creates no index for it"
    );
}

#[test]
fn the_autoindex_catalogue_keeps_the_rows_the_gpkg_reader_drops() {
    use crate::gpkg_input::fixture::{Cell, Image, record};

    // A `sqlite_master` holding one table and one auto-index whose `sql` is
    // NULL — exactly what `UNIQUE (z, x, y)` as a TABLE constraint produces,
    // and exactly the row `master_entries` drops.
    let mut image = Image::new(4096);
    let table_leaf = image.leaf_page(0, &[]);
    let table_root = image.add_page(table_leaf);
    let index_leaf = image.leaf_page(0, &[]);
    let index_root = image.add_page(index_leaf);
    let rows = [
        record(&[
            Cell::Text("table"),
            Cell::Text("tiles"),
            Cell::Text("tiles"),
            Cell::Int(i64::from(table_root)),
            Cell::Text(
                "CREATE TABLE tiles (zoom_level integer, tile_column integer, \
                 tile_row integer, tile_data blob, UNIQUE (zoom_level, tile_column, tile_row))",
            ),
        ]),
        record(&[
            Cell::Text("index"),
            Cell::Text("sqlite_autoindex_tiles_1"),
            Cell::Text("tiles"),
            Cell::Int(i64::from(index_root)),
            Cell::Null,
        ]),
    ];
    let cells: Vec<Vec<u8>> = rows
        .iter()
        .enumerate()
        .map(|(index, payload)| image.table_cell(index as i64 + 1, payload))
        .collect();
    let page1 = image.leaf_page(100, &cells);
    image.set_page1(page1);
    let bytes = image.finish();

    let db = SqliteDb::open(&bytes).expect("the fixture opens");
    let dropped = db.master_entries().expect("the catalogue reads");
    assert_eq!(
        dropped.len(),
        1,
        "master_entries drops the sql=NULL auto-index — the bug this fixes"
    );

    let kept = db
        .master_entries_with_autoindex()
        .expect("the catalogue reads");
    assert_eq!(kept.len(), 2);
    let auto = kept
        .iter()
        .find(|entry| entry.entry_type == "index")
        .expect("the auto-index survives");
    assert_eq!(auto.name, "sqlite_autoindex_tiles_1");
    assert_eq!(auto.rootpage, index_root, "and it has a REAL b-tree root");
    assert!(auto.sql.is_empty(), "an auto-index has no statement at all");

    // …and the table's own statement is where its key columns come from.
    let table = kept
        .iter()
        .find(|entry| entry.entry_type == "table")
        .expect("the table");
    assert_eq!(
        unique_key_columns(&table.sql),
        vec![vec![
            plain("zoom_level"),
            plain("tile_column"),
            plain("tile_row"),
        ]]
    );
}

// ---------------------------------------------------------------------------
// CREATE INDEX
// ---------------------------------------------------------------------------

/// The shape assertion every accepted statement is checked against.
fn index_of(sql: &str) -> crate::gpkg_input::sqlite::IndexSchema {
    parse_create_index(sql).unwrap_or_else(|| panic!("must parse: {sql}"))
}

#[test]
fn a_plain_create_index_parses_into_its_key_columns() {
    let index = index_of("CREATE INDEX tiles_idx ON tiles (zoom_level, tile_column, tile_row)");
    assert_eq!(index.name, "tiles_idx");
    assert_eq!(index.table, "tiles");
    assert!(!index.unique);
    assert!(!index.partial);
    assert_eq!(
        index.columns,
        vec![
            IndexColumn {
                name: "zoom_level".to_owned(),
                collation: String::new(),
                descending: false,
            },
            IndexColumn {
                name: "tile_column".to_owned(),
                collation: String::new(),
                descending: false,
            },
            IndexColumn {
                name: "tile_row".to_owned(),
                collation: String::new(),
                descending: false,
            },
        ]
    );
}

#[test]
fn unique_if_not_exists_quoting_and_a_schema_qualifier_all_parse() {
    let index = index_of(
        "CREATE UNIQUE INDEX IF NOT EXISTS main.\"tile idx\" ON [tiles] (`zoom_level` ASC)",
    );
    assert!(index.unique);
    assert_eq!(index.name, "tile idx");
    assert_eq!(index.table, "tiles");
    assert_eq!(index.columns.len(), 1);
    assert!(!index.columns[0].descending);
}

#[test]
fn collate_and_desc_are_carried_because_they_change_the_comparison() {
    let index =
        index_of("CREATE INDEX i ON images (tile_id COLLATE NOCASE, tile_data DESC, other ASC)");
    assert_eq!(index.columns[0].collation, "NOCASE");
    assert!(!index.columns[0].descending);
    assert!(index.columns[1].descending);
    assert_eq!(index.columns[1].collation, "");
    assert!(!index.columns[2].descending);
}

#[test]
fn a_partial_index_is_flagged_rather_than_silently_trusted() {
    let index = index_of("CREATE INDEX i ON tiles (zoom_level) WHERE zoom_level > 3");
    assert!(
        index.partial,
        "a partial index covers only some rows, so it cannot answer \"is this tile present\""
    );
}

#[test]
fn anything_this_parser_cannot_read_is_a_refusal_and_never_a_guess() {
    for refused in [
        "",
        "CREATE TABLE t (a)",
        "CREATE INDEX i ON t",
        "CREATE INDEX i (a)",
        "CREATE INDEX ON t (a)",
        // An expression key.
        "CREATE INDEX i ON t (lower(a))",
        "CREATE INDEX i ON t (a + 1)",
        // An empty key list.
        "CREATE INDEX i ON t ()",
        // Something between the table and the keys.
        "CREATE INDEX i ON t USING btree (a)",
        // A decoration this parser does not know.
        "CREATE INDEX i ON t (a NULLS LAST)",
        "CREATE INDEX i ON t (a COLLATE)",
    ] {
        assert!(
            parse_create_index(refused).is_none(),
            "must be refused, not guessed: {refused}"
        );
    }
}
