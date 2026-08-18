// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The paged reader, offline, over hand-built SQLite images.
//!
//! Everything here is driven the way the production path is — a walk answers
//! "these pages, please" and is stepped again when they arrive — with the bytes
//! coming out of an in-memory image instead of a transport. That is exactly the
//! shape [`crate::archive::MemoryRangeTransport`] presents, so what is proven
//! here is the real state machine and not a synchronous stand-in.
//!
//! # The decisive test
//!
//! [`the_paged_reader_and_the_resident_reader_agree_on_every_tile`] reads every
//! address of every fixture shape **both ways** and compares the bytes. Two
//! readers for one format is a divergence risk for as long as both exist; that
//! test is the fence, and weakening it ships the divergence.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use oxigis_render::TileId;

use crate::gpkg_input::fixture::{Cell, record};
use crate::local_vector::LocalVectorError;
use crate::mbtiles::MbTilesReader;
use crate::mbtiles::fixture::{
    FLAT_TILES_SQL, IndexSpec, Table, address_key, indexed_flat_image, indexed_image,
    indexed_normalized_image, metadata_table, raster_metadata, vector_metadata,
};
use crate::mbtiles::paged::descend::MAX_PAGE_READS_PER_TILE;
use crate::mbtiles::paged::source::PageRun;
use crate::mbtiles::paged::{DescentStep, PagedArchive, PagedNeed, PagedOpen, PagedOpenStep};

/// The page size every real MBTiles writer uses.
const PAGE: usize = 4096;

/// A tile address, built the way every caller in this crate builds one.
fn tile(z: u8, x: u32, y: u32) -> TileId {
    TileId::new(z, x, y).unwrap_or_else(|error| panic!("tile {z}/{x}/{y} must be valid: {error}"))
}

/// Drives the paged reader over an in-memory image, counting range reads.
///
/// The whole point of the paged reader is that it reads a *few* pages, so the
/// count is asserted as hard as the bytes are.
struct Harness {
    /// The archive.
    image: Vec<u8>,
    /// How many range reads have been answered.
    reads: usize,
    /// Every page run a *descent* has asked for, in order.
    ///
    /// The run's **width** is the stable observable of a speculative chain read:
    /// a run wider than one page means the reader guessed the chain was
    /// contiguous, and a run of exactly one means it stopped guessing. Run
    /// *counts* are not stable — an earlier speculative over-read leaves pages
    /// cached and silently removes a later round trip — so nothing asserts them.
    runs: Vec<PageRun>,
    /// Whether the harness pins the file's length, as a `Content-Range` total
    /// would.
    declare_total: bool,
}

impl Harness {
    fn new(image: Vec<u8>) -> Self {
        Self {
            image,
            reads: 0,
            runs: Vec::new(),
            declare_total: true,
        }
    }

    /// The same archive with the file length withheld — the shape a host that
    /// declares no `Content-Range` total produces.
    fn without_total(mut self) -> Self {
        self.declare_total = false;
        self
    }

    /// The bytes of one range, clamped at the end of the image.
    fn read(&mut self, start: u64, end: u64) -> Vec<u8> {
        self.reads += 1;
        let start = usize::try_from(start)
            .unwrap_or(usize::MAX)
            .min(self.image.len());
        let end = usize::try_from(end)
            .unwrap_or(usize::MAX)
            .min(self.image.len());
        self.image.get(start..end).unwrap_or_default().to_vec()
    }

    /// Opens the archive, driving the survey to completion.
    fn open(&mut self) -> Result<PagedArchive, LocalVectorError> {
        let total = if self.declare_total {
            Some(self.image.len() as u64)
        } else {
            None
        };
        let mut open = PagedOpen::new(total);
        for _round in 0..64 {
            match open.step()? {
                PagedOpenStep::Ready(archive) => return Ok(*archive),
                PagedOpenStep::Need(PagedNeed::Prefetch(range)) => {
                    let bytes = self.read(range.start, range.end);
                    open.supply_prefetch(&bytes)?;
                }
                PagedOpenStep::Need(PagedNeed::Pages(run)) => {
                    let range = open.range_for(run)?;
                    let bytes = self.read(range.start, range.end);
                    if bytes.is_empty() {
                        return Err(LocalVectorError::new(format!(
                            "SQLite: page {} is past the end of the file",
                            run.first
                        )));
                    }
                    open.supply_pages(run.first, &bytes);
                }
            }
        }
        Err(LocalVectorError::new("the survey never finished"))
    }

    /// Reads one tile, driving the descent to completion.
    fn tile(
        &mut self,
        archive: &mut PagedArchive,
        address: TileId,
    ) -> Result<Option<Vec<u8>>, LocalVectorError> {
        if !archive.covers(address) {
            return Ok(None);
        }
        let mut descent = archive.begin(address);
        for _round in 0..(MAX_PAGE_READS_PER_TILE + 4) {
            match archive.step(&mut descent)? {
                DescentStep::Body(bytes) => return Ok(Some(bytes)),
                DescentStep::Absent => return Ok(None),
                DescentStep::Need(run) => {
                    self.runs.push(run);
                    let range = archive.range_for(run)?;
                    let bytes = self.read(range.start, range.end);
                    if bytes.is_empty() {
                        return Err(LocalVectorError::new(format!(
                            "SQLite: page {} is past the end of the file",
                            run.first
                        )));
                    }
                    archive.supply(run.first, &bytes);
                }
            }
        }
        Err(LocalVectorError::new("the descent never finished"))
    }
}

/// A tile body of `len` bytes whose contents vary along it, so a body
/// reassembled out of order does not compare equal by accident.
fn body(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|index| ((index + usize::from(seed)) % 251) as u8)
        .collect()
}

/// The addresses every fixture below holds, in **MBTiles** row order.
fn sample_tiles() -> Vec<(u8, u32, u32, Vec<u8>)> {
    vec![
        (0, 0, 0, body(1, 24)),
        (1, 0, 0, body(2, 30)),
        (1, 0, 1, body(3, 36)),
        (1, 1, 0, body(4, 42)),
        (1, 1, 1, body(5, 48)),
        (2, 1, 2, body(6, 60)),
        (2, 2, 1, body(7, 66)),
    ]
}

/// One flat archive's `tiles` rows, in insertion order.
fn flat_rows(tiles: &[(u8, u32, u32, Vec<u8>)]) -> Vec<(i64, Vec<u8>)> {
    tiles
        .iter()
        .enumerate()
        .map(|(position, (z, x, y, blob))| {
            (
                position as i64 + 1,
                record(&[
                    Cell::Int(i64::from(*z)),
                    Cell::Int(i64::from(*x)),
                    Cell::Int(i64::from(*y)),
                    Cell::Blob(blob),
                ]),
            )
        })
        .collect()
}

/// Every XYZ address a reader could be asked for over [`sample_tiles`], hits and
/// misses alike.
fn every_address() -> Vec<TileId> {
    let mut addresses = vec![tile(0, 0, 0)];
    for x in 0..2 {
        for y in 0..2 {
            addresses.push(tile(1, x, y));
        }
    }
    for x in 0..4 {
        for y in 0..4 {
            addresses.push(tile(2, x, y));
        }
    }
    addresses.push(tile(3, 0, 0));
    addresses
}

// ---------------------------------------------------------------------------
// The decisive test
// ---------------------------------------------------------------------------

/// Every fixture shape, named, with the image each produces.
fn agreement_fixtures() -> Vec<(&'static str, Vec<u8>)> {
    let tiles = sample_tiles();
    let mut spilled = sample_tiles();
    // A body far past `usable - 35`, so it genuinely travels an overflow chain.
    spilled.push((2, 3, 3, body(9, PAGE * 5 + 17)));
    vec![
        (
            "flat, explicit CREATE INDEX",
            indexed_flat_image(PAGE, &tiles, &raster_metadata(), false),
        ),
        (
            "flat, sqlite_autoindex (sql = NULL)",
            indexed_flat_image(PAGE, &tiles, &raster_metadata(), true),
        ),
        (
            "normalized, map join images",
            indexed_normalized_image(
                PAGE,
                &[
                    (0, 0, 0, "a"),
                    (1, 0, 0, "b"),
                    (1, 0, 1, "b"),
                    (1, 1, 0, "c"),
                    (1, 1, 1, "d"),
                    (2, 1, 2, "e"),
                    (2, 2, 1, "f"),
                ],
                &[
                    ("a", body(1, 24)),
                    ("b", body(2, 30)),
                    ("c", body(4, 42)),
                    ("d", body(5, 48)),
                    ("e", body(6, 60)),
                    ("f", body(7, 66)),
                ],
                &vector_metadata(),
            ),
        ),
        (
            "flat with a spilled body, contiguous chain",
            indexed_flat_image(PAGE, &spilled, &raster_metadata(), false),
        ),
        (
            "flat with a spilled body, SCRAMBLED chain",
            scramble_overflow(indexed_flat_image(
                PAGE,
                &spilled,
                &raster_metadata(),
                false,
            )),
        ),
        (
            "512-byte pages",
            indexed_flat_image(512, &tiles, &raster_metadata(), false),
        ),
        (
            "65 536-byte pages",
            indexed_flat_image(65_536, &tiles, &raster_metadata(), false),
        ),
    ]
}

#[test]
fn the_paged_reader_and_the_resident_reader_agree_on_every_tile() {
    for (name, image) in agreement_fixtures() {
        let resident = MbTilesReader::open(std::sync::Arc::from(image.clone().into_boxed_slice()))
            .unwrap_or_else(|error| panic!("{name}: the resident reader must open it: {error}"));
        let mut harness = Harness::new(image);
        let mut paged = harness
            .open()
            .unwrap_or_else(|error| panic!("{name}: the paged reader must open it: {error}"));

        assert_eq!(
            paged.info().content,
            resident.info().content,
            "{name}: the two readers must agree on what the tiles ARE"
        );
        assert_eq!(paged.info().min_zoom, resident.info().min_zoom, "{name}");
        assert_eq!(paged.info().max_zoom, resident.info().max_zoom, "{name}");

        let mut hits = 0usize;
        for address in every_address() {
            let want = resident
                .tile(address)
                .unwrap_or_else(|error| panic!("{name}: resident {address:?}: {error}"));
            let before = harness.reads;
            let got = harness
                .tile(&mut paged, address)
                .unwrap_or_else(|error| panic!("{name}: paged {address:?}: {error}"));
            assert_eq!(
                got, want,
                "{name}: {}/{}/{} must read the same bytes both ways",
                address.z, address.x, address.y
            );
            if got.is_some() {
                hits += 1;
            }
            let spent = harness.reads - before;
            assert!(
                spent <= MAX_PAGE_READS_PER_TILE as usize,
                "{name}: {}/{}/{} cost {spent} reads",
                address.z,
                address.x,
                address.y
            );
        }
        assert!(
            hits >= 6,
            "{name}: the fixture must actually hold tiles; only {hits} were found"
        );
    }
}

/// Where a spilled cell is, and which pages its chain visits.
#[derive(Debug)]
struct SpilledCell {
    /// Absolute byte offset of the cell's payload-length varint.
    length_at: usize,
    /// How many bytes that varint takes.
    length_len: usize,
    /// Absolute byte offset of the cell's 4-byte overflow pointer.
    pointer_at: usize,
    /// The chain's pages, in order.
    chain: Vec<u32>,
}

/// Finds every spilled cell in a hand-built image and follows each chain.
///
/// The fixtures allocate overflow pages *while* building the cell that spills,
/// so the chain is somewhere in the middle of the file rather than at its end —
/// which is exactly why a test that damages "the last four pages" damages the
/// wrong thing. This locates them honestly, out of the bytes.
///
/// Cells come back in page-then-cell order, so [`Iterator::next`] over the
/// result is the single-cell answer this helper used to give. A malformed cell
/// stops the scan rather than being skipped, for the same reason: the first
/// element must not change because the scan learned to keep going.
fn find_spilled_cells(image: &[u8], page_size: usize) -> Vec<SpilledCell> {
    let mut found: Vec<SpilledCell> = Vec::new();
    let pages = image.len() / page_size;
    for number in 2..=pages {
        let start = (number - 1) * page_size;
        let Some(page) = image.get(start..start + page_size) else {
            return found;
        };
        if page.first() != Some(&13) {
            continue;
        }
        let cell_count = usize::from(u16::from_be_bytes([page[3], page[4]]));
        for index in 0..cell_count {
            let at = 8 + index * 2;
            let Some(pointer) = page.get(at..at + 2) else {
                continue;
            };
            let offset = usize::from(u16::from_be_bytes([pointer[0], pointer[1]]));
            let Ok((payload_len, consumed)) = crate::gpkg_input::sqlite::varint(page, offset)
            else {
                continue;
            };
            let Ok((_rowid, rowid_len)) =
                crate::gpkg_input::sqlite::varint(page, offset + consumed)
            else {
                continue;
            };
            let Ok(total) = usize::try_from(payload_len) else {
                continue;
            };
            let inline = crate::gpkg_input::sqlite::inline_len_for(page_size, total);
            if inline == total {
                continue;
            }
            let pointer_at = start + offset + consumed + rowid_len + inline;
            let Some(head) = image.get(pointer_at..pointer_at + 4) else {
                return found;
            };
            let mut next = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
            let mut chain = Vec::new();
            while next != 0 && chain.len() < 4096 {
                chain.push(next);
                let Some(head) = image
                    .get((next as usize - 1) * page_size..)
                    .and_then(|rest| rest.get(..4))
                else {
                    return found;
                };
                next = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
            }
            found.push(SpilledCell {
                length_at: start + offset,
                length_len: consumed,
                pointer_at,
                chain,
            });
        }
    }
    found
}

/// Reverses the order in which one overflow chain visits its own pages.
///
/// The set of pages is unchanged and every "next" pointer is rewritten, so the
/// chain is still perfectly valid — just not contiguous, and now running
/// *backwards* through the file. A speculative read that trusted contiguity
/// would hand the pages back in the wrong order and the body would decode to the
/// wrong bytes, which is precisely why every pointer is verified.
///
/// Damaging several chains means collecting **every** cell first and only then
/// calling this for each: reversal keeps a chain's page set and leaves the other
/// chains' `length_at`/`pointer_at` untouched, but it rewrites *this* chain's
/// head pointer, so a re-scan in between would report a different chain.
fn scramble_one(image: &mut [u8], cell: &SpilledCell) {
    if cell.chain.len() < 2 {
        return;
    }
    let contents: Vec<Vec<u8>> = cell
        .chain
        .iter()
        .map(|number| {
            let start = (*number as usize - 1) * PAGE;
            image
                .get(start + 4..start + PAGE)
                .unwrap_or_default()
                .to_vec()
        })
        .collect();
    // Chain position i now lives on the physical page that used to hold the
    // LAST position, and so on.
    let places: Vec<u32> = cell.chain.iter().rev().copied().collect();
    for (position, content) in contents.iter().enumerate() {
        let page = places[position];
        let next = places.get(position + 1).copied().unwrap_or(0);
        let start = (page as usize - 1) * PAGE;
        if let Some(slot) = image.get_mut(start..start + 4) {
            slot.copy_from_slice(&next.to_be_bytes());
        }
        if let Some(slot) = image.get_mut(start + 4..start + PAGE) {
            slot.copy_from_slice(content);
        }
    }
    let head = places[0];
    if let Some(slot) = image.get_mut(cell.pointer_at..cell.pointer_at + 4) {
        slot.copy_from_slice(&head.to_be_bytes());
    }
}

/// Reverses the first spilled chain in an image — the single-cell damage the
/// SCRAMBLED agreement fixture has always applied.
fn scramble_overflow(mut image: Vec<u8>) -> Vec<u8> {
    let Some(cell) = find_spilled_cells(&image, PAGE).into_iter().next() else {
        return image;
    };
    scramble_one(&mut image, &cell);
    image
}

// ---------------------------------------------------------------------------
// Speculation, and the archive-lifetime flip when it keeps failing
// ---------------------------------------------------------------------------

/// Five bodies, each far past `usable - 35`, at five distinct **MBTiles**
/// addresses.
///
/// `PAGE * 5 + 17` spills a five-page overflow chain per body — five chains, no
/// shared pages — which is one more damaged chain than
/// [`SPECULATION_FAILURE_LIMIT`] needs. Three failures exhaust speculation, the
/// fourth and fifth tiles then witness the flip, and the spare means a fixture
/// whose damage did not accumulate fails the test rather than passing by luck.
///
/// [`SPECULATION_FAILURE_LIMIT`]: crate::mbtiles::paged::source::SPECULATION_FAILURE_LIMIT
fn five_spilled_tiles() -> Vec<(u8, u32, u32, Vec<u8>)> {
    vec![
        (2, 0, 0, body(11, PAGE * 5 + 17)),
        (2, 1, 0, body(12, PAGE * 5 + 17)),
        (2, 2, 0, body(13, PAGE * 5 + 17)),
        (2, 3, 0, body(14, PAGE * 5 + 17)),
        (2, 0, 1, body(15, PAGE * 5 + 17)),
    ]
}

#[test]
fn speculation_that_keeps_failing_flips_the_archive_to_hop_by_hop() {
    let mut image = indexed_flat_image(PAGE, &five_spilled_tiles(), &raster_metadata(), false);

    // Collect every chain BEFORE damaging any of them: reversal rewrites the
    // head pointer it damages, so a re-scan in between would find a different
    // set of chains and the damage would not accumulate.
    let cells = find_spilled_cells(&image, PAGE);
    assert_eq!(
        cells.len(),
        5,
        "the fixture must spill five separate bodies"
    );
    for (position, cell) in cells.iter().enumerate() {
        assert_eq!(
            cell.chain.len(),
            5,
            "body #{position} must travel a five-page chain"
        );
    }
    for cell in &cells {
        scramble_one(&mut image, cell);
    }

    let resident = MbTilesReader::open(std::sync::Arc::from(image.clone().into_boxed_slice()))
        .expect("the resident reader must open the scrambled archive");
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("the paged reader must open it");

    // XYZ addresses: the MBTiles rows above, flipped.
    let addresses = [
        tile(2, 0, 3),
        tile(2, 1, 3),
        tile(2, 2, 3),
        tile(2, 3, 3),
        tile(2, 0, 2),
    ];
    let mut exhausted_before = Vec::new();
    let mut exhausted_after = Vec::new();
    let mut widest = Vec::new();

    for (position, address) in addresses.into_iter().enumerate() {
        exhausted_before.push(archive.source.speculation_exhausted());
        harness.runs.clear();
        let before = harness.reads;
        let got = harness
            .tile(&mut archive, address)
            .unwrap_or_else(|error| panic!("paged #{position} {address:?}: {error}"));
        let want = resident
            .tile(address)
            .unwrap_or_else(|error| panic!("resident #{position} {address:?}: {error}"));

        // The fence FIRST: a fixture that corrupted the bytes rather than only
        // the chain's order would fail here instead of "proving" a flip.
        assert_eq!(
            got, want,
            "#{position} {}/{}/{} must read the same bytes both ways",
            address.z, address.x, address.y
        );
        assert!(
            got.is_some(),
            "#{position} {}/{}/{} must actually be present",
            address.z,
            address.x,
            address.y
        );

        let spent = harness.reads - before;
        assert!(
            spent <= MAX_PAGE_READS_PER_TILE as usize,
            "#{position} cost {spent} reads, past the {MAX_PAGE_READS_PER_TILE} cap"
        );
        widest.push(harness.runs.iter().map(|run| run.count).max().unwrap_or(0));
        exhausted_after.push(archive.source.speculation_exhausted());
    }

    // The flag: alive going into the third damaged chain, spent coming out of
    // it. Asserting both sides is what stops a fixture that damaged nothing
    // from passing vacuously.
    assert!(
        !exhausted_before[0],
        "speculation must be alive before the first failure"
    );
    assert!(
        !exhausted_before[2],
        "two failures must NOT be enough to abandon speculation"
    );
    assert!(
        exhausted_after[2],
        "the third failed speculation must exhaust it for the archive's lifetime"
    );
    assert!(
        exhausted_after[4],
        "every chain here is damaged, so nothing in this trace ever re-verifies a contiguous \
         read to un-exhaust it (see `note_speculation_success`, which does, given one)"
    );

    // The flip, proven from the SHAPE of the range requests rather than from the
    // flag: a run wider than one page is a speculative read, a run of exactly
    // one page is hop-by-hop. Never assert run *counts* — tiles #1 and #2 need
    // only one round trip each purely because an earlier speculative over-read
    // already cached their pages.
    for (position, width) in widest.iter().enumerate().take(3) {
        assert!(
            *width > 1,
            "#{position} must still have SPECULATED; widest run was {width}"
        );
    }
    for (position, width) in widest.iter().enumerate().skip(3) {
        assert_eq!(
            *width, 1,
            "#{position} must ask one page at a time once speculation is spent"
        );
    }
}

#[test]
fn a_large_tile_still_reads_once_speculation_is_exhausted() {
    // The bug: once speculation is exhausted, every subsequent chain read
    // goes hop-by-hop, one page per round trip. If those round trips were
    // still charged against `MAX_PAGE_READS_PER_TILE` (the b-tree descent's
    // cap), a tile whose body legitimately spans more overflow pages than
    // that budget allows would be refused outright — exactly the failure
    // `MAX_PAGE_READS_PER_TILE`'s own doc disclaims ("punishes malice, not
    // size"). Chain reads now have their own budget, sized to the record.
    let body_len = PAGE * 30 + 17;
    let big = body(20, body_len);
    let image = indexed_flat_image(PAGE, &[(2, 0, 0, big.clone())], &raster_metadata(), false);
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("opens");

    // Force the archive past `SPECULATION_FAILURE_LIMIT` directly — this
    // test is about what happens to a large tile AFTER exhaustion, not how
    // exhaustion is reached (`speculation_that_keeps_failing_flips_the_
    // archive_to_hop_by_hop` already covers that).
    archive.source.note_speculation_failure();
    archive.source.note_speculation_failure();
    archive.source.note_speculation_failure();
    assert!(archive.source.speculation_exhausted());

    let address = tile(2, 0, 3);
    let mut descent = archive.begin(address);
    let mut got = None;
    for _round in 0..64 {
        match archive
            .step(&mut descent)
            .expect("no read may be refused for costing too much")
        {
            DescentStep::Body(bytes) => {
                got = Some(bytes);
                break;
            }
            DescentStep::Absent => panic!("the tile must be present"),
            DescentStep::Need(run) => {
                let range = archive.range_for(run).expect("a real run");
                let bytes = harness.read(range.start, range.end);
                archive.supply(run.first, &bytes);
            }
        }
    }
    assert_eq!(
        got.as_deref(),
        Some(big.as_slice()),
        "the body must still come back whole once every hop-by-hop page is read"
    );
}

#[test]
fn a_successful_speculation_resets_the_failure_count() {
    // Two failures, a success, then one more failure: under a monotonic
    // counter this is three failures total and would exhaust speculation for
    // the archive's whole remaining lifetime. The counter must instead
    // reflect only what happened since the last verified success.
    let image = indexed_flat_image(PAGE, &sample_tiles(), &raster_metadata(), false);
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("opens");

    archive.source.note_speculation_failure();
    archive.source.note_speculation_failure();
    assert!(
        !archive.source.speculation_exhausted(),
        "two is not the limit"
    );
    archive.source.note_speculation_success();
    archive.source.note_speculation_failure();
    assert!(
        !archive.source.speculation_exhausted(),
        "a verified success must clear the failures that came before it"
    );
}

// ---------------------------------------------------------------------------
// The open, and what it costs
// ---------------------------------------------------------------------------

#[test]
fn a_cold_open_costs_one_read_and_a_warm_lookup_costs_nothing() {
    let image = indexed_flat_image(PAGE, &sample_tiles(), &raster_metadata(), false);
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("opens");
    assert_eq!(
        harness.reads, 1,
        "header, catalogue AND metadata all fit one 16 KiB prefetch"
    );

    let before = harness.reads;
    assert!(harness.tile(&mut archive, tile(1, 0, 1)).unwrap().is_some());
    let cold = harness.reads - before;
    assert!(cold <= 4, "a cold lookup cost {cold} reads");

    let before = harness.reads;
    assert!(harness.tile(&mut archive, tile(1, 1, 1)).unwrap().is_some());
    assert_eq!(
        harness.reads, before,
        "a warm lookup reuses the pages already held"
    );
}

#[test]
fn the_zoom_gate_answers_before_a_single_page_is_read() {
    let image = indexed_flat_image(PAGE, &sample_tiles(), &raster_metadata(), false);
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("opens");
    let before = harness.reads;
    // The fixture's metadata declares maxzoom 2.
    assert_eq!(harness.tile(&mut archive, tile(5, 0, 0)).unwrap(), None);
    assert_eq!(harness.reads, before, "a zoom gate costs no reads");
}

// ---------------------------------------------------------------------------
// Lying headers
// ---------------------------------------------------------------------------

#[test]
fn a_lying_page_size_is_refused_by_name() {
    let good = indexed_flat_image(PAGE, &sample_tiles(), &raster_metadata(), false);
    for raw in [0u16, 1u16.wrapping_add(512), 3] {
        let mut image = good.clone();
        image[16..18].copy_from_slice(&raw.to_be_bytes());
        let error = Harness::new(image)
            .open()
            .expect_err("an illegal page size must be refused");
        assert!(error.to_string().contains("page size"), "{raw}: {error}");
    }
    // A reserved area that leaves too little of the page usable. 255 bytes off
    // a 4096-byte page is legal; off a 512-byte one it is not.
    let mut small = indexed_flat_image(512, &sample_tiles(), &raster_metadata(), false);
    small[20] = 255;
    let error = Harness::new(small)
        .open()
        .expect_err("255 reserved bytes of a 512-byte page must be refused");
    assert!(error.to_string().contains("reserves"), "{error}");
    let _ = good;
}

#[test]
fn a_file_that_is_not_sqlite_is_refused_before_anything_else() {
    // PMTiles bytes offered as an MBTiles archive: the magic is the whole
    // defence, and it must fire before any arithmetic runs.
    let pmtiles = oxigis_render::pmtiles::sample_pmtiles_raster();
    let error = Harness::new(pmtiles).open().expect_err("not SQLite");
    assert!(error.to_string().contains("SQLite 3 magic"), "{error}");

    let error = Harness::new(vec![0u8; 8]).open().expect_err("too short");
    assert!(error.to_string().contains("too short"), "{error}");
}

#[test]
fn a_bogus_page_count_is_bounded_rather_than_believed() {
    let mut image = indexed_flat_image(PAGE, &sample_tiles(), &raster_metadata(), false);
    // 0xFFFFFFFF pages, "vouched for" by agreeing change counters.
    image[28..32].copy_from_slice(&u32::MAX.to_be_bytes());
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("a wrong count is not fatal at open");
    // Reading still works: the count is only ever an UPPER bound, and the real
    // defence is that a page past the end comes back short.
    assert!(harness.tile(&mut archive, tile(0, 0, 0)).unwrap().is_some());
}

#[test]
fn disagreeing_change_counters_make_the_page_count_untrusted() {
    let mut image = indexed_flat_image(PAGE, &sample_tiles(), &raster_metadata(), false);
    // Offsets 24 and 92 disagree, so offset 28 means nothing — and here it lies.
    image[92..96].copy_from_slice(&99u32.to_be_bytes());
    image[28..32].copy_from_slice(&2u32.to_be_bytes());
    let mut harness = Harness::new(image);
    let mut archive = harness
        .open()
        .expect("an untrusted count falls back to the pinned file length");
    assert!(
        harness.tile(&mut archive, tile(0, 0, 0)).unwrap().is_some(),
        "the archive is still readable: the bogus count was NOT believed"
    );
}

#[test]
fn a_truncated_file_with_no_pinned_length_is_refused_by_name() {
    let image = indexed_flat_image(PAGE, &sample_tiles(), &raster_metadata(), false);
    let truncated = image.get(..PAGE * 2).unwrap_or_default().to_vec();
    let mut harness = Harness::new(truncated).without_total();
    match harness.open() {
        Err(error) => assert!(!error.to_string().is_empty(), "a named refusal: {error}"),
        Ok(mut archive) => {
            let mut refused = false;
            for address in every_address() {
                if harness.tile(&mut archive, address).is_err() {
                    refused = true;
                    break;
                }
            }
            assert!(refused, "a page past the end must be refused, not invented");
        }
    }
}

// ---------------------------------------------------------------------------
// Schema refusals
// ---------------------------------------------------------------------------

#[test]
fn an_archive_with_no_index_is_refused_and_says_what_to_do() {
    let image = indexed_image(
        PAGE,
        &[
            Table {
                name: "tiles",
                sql: FLAT_TILES_SQL,
                rows: flat_rows(&sample_tiles()),
            },
            metadata_table(&raster_metadata()),
        ],
        &[],
        &[],
    );
    let error = Harness::new(image).open().expect_err("no index, no lookup");
    let message = error.to_string();
    assert!(message.contains("no index"), "{message}");
    assert!(message.contains("download"), "{message}");
    assert!(message.contains("PMTiles"), "{message}");
}

/// A flat archive whose index is declared by `sql`, keyed in ASC order.
fn flat_with_index_sql(sql: &'static str) -> Vec<u8> {
    let tiles = sample_tiles();
    let mut keys: Vec<(u8, u32, u32, i64)> = tiles
        .iter()
        .enumerate()
        .map(|(position, (z, x, y, _))| (*z, *x, *y, position as i64 + 1))
        .collect();
    keys.sort_by_key(|(z, x, y, _)| (*z, *x, *y));
    let records: Vec<Vec<u8>> = keys
        .iter()
        .map(|(z, x, y, rowid)| address_key(*z, *x, *y, *rowid))
        .collect();
    indexed_image(
        PAGE,
        &[
            Table {
                name: "tiles",
                sql: FLAT_TILES_SQL,
                rows: flat_rows(&tiles),
            },
            metadata_table(&raster_metadata()),
        ],
        &[],
        &[IndexSpec {
            name: "tile_index",
            table: "tiles",
            sql: Some(sql),
            records,
        }],
    )
}

#[test]
fn a_nocase_index_is_refused_because_a_byte_descent_would_return_the_wrong_tile() {
    let image = flat_with_index_sql(
        "CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column COLLATE NOCASE, \
         tile_row)",
    );
    let error = Harness::new(image).open().expect_err("NOCASE is refused");
    let message = error.to_string();
    assert!(message.contains("NOCASE"), "{message}");
    assert!(message.contains("wrong tile"), "{message}");
}

#[test]
fn a_partial_index_is_not_used_and_the_archive_is_refused_by_name() {
    let image = flat_with_index_sql(
        "CREATE INDEX tile_index on tiles (zoom_level, tile_column, tile_row) \
         WHERE zoom_level > 0",
    );
    let error = Harness::new(image)
        .open()
        .expect_err("a partial index covers only some rows");
    assert!(error.to_string().contains("no index"), "{error}");
}

#[test]
fn a_desc_index_is_accepted_and_read_correctly() {
    // The bug an ASC-assuming descent would have: a DESC index genuinely
    // reorders the leaves, so every comparison at that position must flip.
    let tiles = sample_tiles();
    let mut keys: Vec<(u8, u32, u32, i64)> = tiles
        .iter()
        .enumerate()
        .map(|(position, (z, x, y, _))| (*z, *x, *y, position as i64 + 1))
        .collect();
    keys.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then(left.1.cmp(&right.1))
            .then(right.2.cmp(&left.2))
    });
    let records: Vec<Vec<u8>> = keys
        .iter()
        .map(|(z, x, y, rowid)| address_key(*z, *x, *y, *rowid))
        .collect();
    let image = indexed_image(
        PAGE,
        &[
            Table {
                name: "tiles",
                sql: FLAT_TILES_SQL,
                rows: flat_rows(&tiles),
            },
            metadata_table(&raster_metadata()),
        ],
        &[],
        &[IndexSpec {
            name: "tile_index",
            table: "tiles",
            sql: Some(
                "CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column, \
                 tile_row DESC)",
            ),
            records,
        }],
    );
    let resident =
        MbTilesReader::open(std::sync::Arc::from(image.clone().into_boxed_slice())).expect("opens");
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("a DESC index is readable");
    for address in every_address() {
        let want = resident.tile(address).expect("resident");
        let got = harness.tile(&mut archive, address).expect("paged");
        assert_eq!(
            got, want,
            "{}/{}/{} through a DESC index",
            address.z, address.x, address.y
        );
    }
}

#[test]
fn a_table_constraint_unique_with_desc_is_read_correctly_as_an_autoindex() {
    // The auto-index twin of `a_desc_index_is_accepted_and_read_correctly`:
    // the same DESC bug, but reached through a table-level `UNIQUE` whose
    // `sqlite_master` row carries no `sql` of its own — `group_columns` used
    // to drop the `DESC` silently instead of reading or refusing it, which
    // pointed this descent at the wrong subtree.
    let tiles = sample_tiles();
    let mut keys: Vec<(u8, u32, u32, i64)> = tiles
        .iter()
        .enumerate()
        .map(|(position, (z, x, y, _))| (*z, *x, *y, position as i64 + 1))
        .collect();
    keys.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then(left.1.cmp(&right.1))
            .then(right.2.cmp(&left.2))
    });
    let records: Vec<Vec<u8>> = keys
        .iter()
        .map(|(z, x, y, rowid)| address_key(*z, *x, *y, *rowid))
        .collect();
    let image = indexed_image(
        PAGE,
        &[
            Table {
                name: "tiles",
                sql: "CREATE TABLE tiles (zoom_level integer, tile_column integer, \
                      tile_row integer, tile_data blob, \
                      UNIQUE (zoom_level, tile_column, tile_row DESC))",
                rows: flat_rows(&tiles),
            },
            metadata_table(&raster_metadata()),
        ],
        &[],
        &[IndexSpec {
            name: "sqlite_autoindex_tiles_1",
            table: "tiles",
            sql: None,
            records,
        }],
    );
    let resident =
        MbTilesReader::open(std::sync::Arc::from(image.clone().into_boxed_slice())).expect("opens");
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("a DESC auto-index is readable");
    for address in every_address() {
        let want = resident.tile(address).expect("resident");
        let got = harness.tile(&mut archive, address).expect("paged");
        assert_eq!(
            got, want,
            "{}/{}/{} through a DESC auto-index",
            address.z, address.x, address.y
        );
    }
}

#[test]
fn a_table_constraint_unique_with_nocase_is_refused_by_name_as_an_autoindex() {
    // The auto-index twin of
    // `a_nocase_index_is_refused_because_a_byte_descent_would_return_the_wrong_tile`:
    // `refuse_collation` used to run only on the explicit `CREATE INDEX`
    // branch, so this shape sailed through and would have compared bytes
    // down a `NOCASE` b-tree.
    let image = indexed_image(
        PAGE,
        &[
            Table {
                name: "tiles",
                sql: "CREATE TABLE tiles (zoom_level integer, tile_column integer, \
                      tile_row integer, tile_data blob, \
                      UNIQUE (zoom_level, tile_column COLLATE NOCASE, tile_row))",
                rows: flat_rows(&sample_tiles()),
            },
            metadata_table(&raster_metadata()),
        ],
        &[],
        &[IndexSpec {
            name: "sqlite_autoindex_tiles_1",
            table: "tiles",
            sql: None,
            records: {
                let mut keys: Vec<(u8, u32, u32, i64)> = sample_tiles()
                    .iter()
                    .enumerate()
                    .map(|(position, (z, x, y, _))| (*z, *x, *y, position as i64 + 1))
                    .collect();
                keys.sort_by_key(|(z, x, y, _)| (*z, *x, *y));
                keys.iter()
                    .map(|(z, x, y, rowid)| address_key(*z, *x, *y, *rowid))
                    .collect()
            },
        }],
    );
    let error = Harness::new(image).open().expect_err("NOCASE is refused");
    let message = error.to_string();
    assert!(message.contains("NOCASE"), "{message}");
    assert!(message.contains("wrong tile"), "{message}");
}

#[test]
fn a_without_rowid_table_is_refused_by_name() {
    let image = indexed_image(
        PAGE,
        &[
            Table {
                name: "tiles",
                sql: "CREATE TABLE tiles (zoom_level integer, tile_column integer, \
                      tile_row integer, tile_data blob, \
                      PRIMARY KEY (zoom_level, tile_column, tile_row)) WITHOUT ROWID",
                rows: flat_rows(&sample_tiles()),
            },
            metadata_table(&raster_metadata()),
        ],
        &[],
        &[IndexSpec {
            name: "tile_index",
            table: "tiles",
            sql: Some(
                "CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column, tile_row)",
            ),
            records: vec![address_key(0, 0, 0, 1)],
        }],
    );
    let error = Harness::new(image).open().expect_err("WITHOUT ROWID");
    let message = error.to_string();
    assert!(message.contains("WITHOUT ROWID"), "{message}");
    assert!(message.contains("PMTiles"), "{message}");
}

#[test]
fn a_normalized_archive_without_an_images_index_is_refused_by_name() {
    let image = indexed_image(
        PAGE,
        &[
            Table {
                name: "map",
                sql: "CREATE TABLE map (zoom_level integer, tile_column integer, \
                      tile_row integer, tile_id text)",
                rows: vec![(
                    1,
                    record(&[Cell::Int(0), Cell::Int(0), Cell::Int(0), Cell::Text("a")]),
                )],
            },
            Table {
                name: "images",
                sql: "CREATE TABLE images (tile_data blob, tile_id text)",
                rows: vec![(1, record(&[Cell::Blob(&[1, 2, 3]), Cell::Text("a")]))],
            },
            metadata_table(&vector_metadata()),
        ],
        &[],
        &[IndexSpec {
            name: "map_index",
            table: "map",
            sql: Some("CREATE UNIQUE INDEX map_index on map (zoom_level, tile_column, tile_row)"),
            records: vec![address_key(0, 0, 0, 1)],
        }],
    );
    let error = Harness::new(image).open().expect_err("no images index");
    let message = error.to_string();
    assert!(message.contains("images"), "{message}");
    assert!(message.contains("tile_id"), "{message}");
    assert!(message.contains("PMTiles"), "{message}");
}

#[test]
fn an_archive_with_no_metadata_table_is_refused_by_name() {
    let image = indexed_image(
        PAGE,
        &[Table {
            name: "tiles",
            sql: FLAT_TILES_SQL,
            rows: flat_rows(&sample_tiles()),
        }],
        &[],
        &[IndexSpec {
            name: "tile_index",
            table: "tiles",
            sql: Some(
                "CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column, tile_row)",
            ),
            records: vec![address_key(0, 0, 0, 1)],
        }],
    );
    let error = Harness::new(image).open().expect_err("no metadata");
    assert!(error.to_string().contains("metadata"), "{error}");
}

#[test]
fn an_index_key_that_spilled_is_refused_rather_than_compared_short() {
    // A 2 KiB `tile_id`: past the index's guaranteed-inline prefix at the
    // 4096-byte page size, so its key cannot be compared without reading the
    // overflow chain — and a short comparison returns the WRONG tile.
    let long_id = "x".repeat(2048);
    let image = indexed_normalized_image(
        PAGE,
        &[(0, 0, 0, long_id.as_str())],
        &[(long_id.as_str(), body(1, 24))],
        &vector_metadata(),
    );
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("the catalogue is fine");
    let error = harness
        .tile(&mut archive, tile(0, 0, 0))
        .expect_err("a spilled index key must be refused");
    let message = error.to_string();
    assert!(message.contains("spilled"), "{message}");
    assert!(message.contains("PMTiles"), "{message}");
}

// ---------------------------------------------------------------------------
// Hostile b-trees
// ---------------------------------------------------------------------------

/// A flat archive with a spilled body, whose chain is the tail of the file.
fn spilled_image() -> Vec<u8> {
    let mut tiles = sample_tiles();
    tiles.push((2, 3, 3, body(9, PAGE * 5 + 17)));
    indexed_flat_image(PAGE, &tiles, &raster_metadata(), false)
}

/// The XYZ address of the spilled tile: MBTiles row 3 at zoom 2 is XYZ row 0.
fn spilled_address() -> TileId {
    tile(2, 3, 0)
}

#[test]
fn a_spilled_body_reads_back_whole_through_a_contiguous_chain() {
    let image = spilled_image();
    let resident =
        MbTilesReader::open(std::sync::Arc::from(image.clone().into_boxed_slice())).expect("opens");
    let want = resident.tile(spilled_address()).expect("resident");
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("opens");
    let got = harness
        .tile(&mut archive, spilled_address())
        .expect("paged");
    assert_eq!(got, want);
    assert_eq!(got.map(|bytes| bytes.len()), Some(PAGE * 5 + 17));
}

#[test]
fn a_non_contiguous_chain_falls_back_and_still_reads_the_right_bytes() {
    let image = scramble_overflow(spilled_image());
    let resident =
        MbTilesReader::open(std::sync::Arc::from(image.clone().into_boxed_slice())).expect("opens");
    let want = resident
        .tile(spilled_address())
        .expect("the resident reader follows the chain hop by hop");
    assert!(want.is_some(), "the scrambled chain is still a valid chain");
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("opens");
    let got = harness
        .tile(&mut archive, spilled_address())
        .expect("the paged reader must recover");
    assert_eq!(
        got, want,
        "byte-equality against the resident reader is the whole point"
    );
}

/// Rewrites the "next" pointer of chain page `position` to `value`.
fn set_chain_pointer(image: &mut [u8], cell: &SpilledCell, position: usize, value: u32) {
    let Some(page) = cell.chain.get(position) else {
        return;
    };
    let start = (*page as usize - 1) * PAGE;
    if let Some(slot) = image.get_mut(start..start + 4) {
        slot.copy_from_slice(&value.to_be_bytes());
    }
}

#[test]
fn an_overflow_chain_that_ends_early_is_refused_by_name() {
    let mut image = spilled_image();
    let cell = find_spilled_cells(&image, PAGE)
        .into_iter()
        .next()
        .expect("the fixture spills");
    assert!(cell.chain.len() >= 3, "a real chain to damage");
    set_chain_pointer(&mut image, &cell, 0, 0);
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("the catalogue is undamaged");
    let error = harness
        .tile(&mut archive, spilled_address())
        .expect_err("a short chain must be refused");
    assert!(error.to_string().contains("early"), "{error}");
}

#[test]
fn an_overflow_chain_that_loops_is_refused_by_name() {
    let mut image = spilled_image();
    let cell = find_spilled_cells(&image, PAGE)
        .into_iter()
        .next()
        .expect("the fixture spills");
    let head = cell.chain.first().copied().unwrap_or(0);
    for position in 0..cell.chain.len() {
        set_chain_pointer(&mut image, &cell, position, head);
    }
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("the catalogue is undamaged");
    let error = harness
        .tile(&mut archive, spilled_address())
        .expect_err("a cyclic chain must be refused");
    assert!(error.to_string().contains("cycle"), "{error}");
}

#[test]
fn a_chain_pointing_outside_the_file_is_refused_by_name() {
    let mut image = spilled_image();
    let cell = find_spilled_cells(&image, PAGE)
        .into_iter()
        .next()
        .expect("the fixture spills");
    set_chain_pointer(&mut image, &cell, 0, 0xFFFF_FFFF);
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("the catalogue is undamaged");
    let error = harness
        .tile(&mut archive, spilled_address())
        .expect_err("a pointer past the end must be refused");
    assert!(error.to_string().contains("outside"), "{error}");
}

#[test]
fn a_cyclic_interior_index_page_is_refused_rather_than_walked_for_ever() {
    // 512-byte pages force a real interior index level over the same addresses.
    let mut image = indexed_flat_image(512, &sample_tiles(), &raster_metadata(), false);
    let pages = image.len() / 512;
    let mut damaged = None;
    for number in 2..=pages {
        let start = (number - 1) * 512;
        if image.get(start) == Some(&2) {
            let at = start + 8;
            image
                .get_mut(at..at + 4)
                .expect("a header")
                .copy_from_slice(&(number as u32).to_be_bytes());
            damaged = Some(number);
            break;
        }
    }
    let Some(_page) = damaged else {
        // Everything fitted in one leaf, so there is no interior page to damage
        // and nothing to assert. The cyclic *table* case is covered above.
        return;
    };
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("the catalogue is undamaged");
    // Some address must reach the damaged page; the right-most one does.
    let mut refused = false;
    for address in every_address() {
        if let Err(error) = harness.tile(&mut archive, address) {
            assert!(
                error.to_string().contains("cycle") || error.to_string().contains("outside"),
                "{error}"
            );
            refused = true;
            break;
        }
    }
    assert!(refused, "a cyclic interior page must be refused");
}

#[test]
fn a_table_root_outside_the_file_is_refused_by_name() {
    let mut image = indexed_flat_image(PAGE, &sample_tiles(), &raster_metadata(), false);
    // Ask the resident reader where `tiles` is rooted, then aim the catalogue's
    // record for it past the end of the file. Finding the byte by re-deriving
    // the layout would mean re-implementing the reader inside its own test.
    let root = {
        let db = crate::gpkg_input::sqlite::SqliteDb::open(&image).expect("opens");
        db.master_entries()
            .expect("a catalogue")
            .into_iter()
            .find(|entry| entry.name == "tiles")
            .expect("the tiles table")
            .rootpage
    };
    let stored = i64::from(root).to_be_bytes();
    let mut done = false;
    for at in 100..image.len().saturating_sub(8) {
        if image.get(at..at + 8) == Some(&stored[..]) {
            image
                .get_mut(at..at + 8)
                .expect("eight bytes")
                .copy_from_slice(&9_999_999i64.to_be_bytes());
            done = true;
            break;
        }
    }
    assert!(done, "the catalogue records the root as an 8-byte integer");
    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("the catalogue itself still reads");
    let error = harness
        .tile(&mut archive, tile(0, 0, 0))
        .expect_err("a root past the end must be refused");
    assert!(error.to_string().contains("outside"), "{error}");
}

#[test]
fn a_record_claiming_hundreds_of_megabytes_is_refused_before_it_is_allocated() {
    // A body just past 2 MiB makes its payload-length varint four bytes wide,
    // which is the only width that can be rewritten in place to claim something
    // enormous without moving a single other byte.
    let mut tiles = sample_tiles();
    tiles.push((2, 3, 3, body(9, 2 * 1024 * 1024 + 64)));
    let mut image = indexed_flat_image(PAGE, &tiles, &raster_metadata(), false);
    let cell = find_spilled_cells(&image, PAGE)
        .into_iter()
        .next()
        .expect("the fixture spills");
    assert_eq!(cell.length_len, 4, "a four-byte payload-length varint");
    // 0x0FFF_FFFF bytes = 256 MiB, sixteen times MAX_RECORD_BYTES.
    let claim: [u8; 4] = [0xFF, 0xFF, 0xFF, 0x7F];
    image
        .get_mut(cell.length_at..cell.length_at + 4)
        .expect("four bytes")
        .copy_from_slice(&claim);

    let mut harness = Harness::new(image);
    let mut archive = harness.open().expect("the catalogue is undamaged");
    let error = harness
        .tile(&mut archive, spilled_address())
        .expect_err("a 256 MiB claim must be refused");
    let message = error.to_string();
    assert!(message.contains("past the"), "{message}");
    assert!(
        harness.reads < 64,
        "and refused BEFORE the pages were fetched: {} reads",
        harness.reads
    );
}

#[test]
fn a_table_rooted_at_page_one_is_read_with_the_hundred_byte_offset() {
    // A `sqlite_master` walk IS a table rooted at page 1, and every fixture
    // exercises it — but the assertion is worth making explicitly, because a
    // reader that forgets the 100-byte database header reads the magic as a
    // b-tree header.
    let image = indexed_flat_image(PAGE, &sample_tiles(), &raster_metadata(), false);
    let mut harness = Harness::new(image);
    let archive = harness.open().expect("page 1's offset is honoured");
    assert_eq!(archive.info().max_zoom, 2);
}

#[test]
fn the_page_cache_is_bounded_and_keyed_by_page_number() {
    use crate::mbtiles::paged::source::{PAGE_CACHE_ENTRIES, PageSource};

    let mut source = PageSource::new(512, 512, Some(u32::MAX));
    let page = vec![7u8; 512];
    for number in 1..=(PAGE_CACHE_ENTRIES as u32 + 32) {
        source.supply(number, &page);
    }
    assert!(source.get(1).is_none(), "the oldest pages were evicted");
    let newest = PAGE_CACHE_ENTRIES as u32 + 32;
    assert!(source.get(newest).is_some());
    assert!(source.holds_run(PageRun::one(newest)));
    assert!(!source.holds_run(PageRun::one(1)));
}

#[test]
fn a_touched_page_survives_eviction_over_one_that_was_only_ever_inserted() {
    // The property `the_page_cache_is_bounded_and_keyed_by_page_number` does
    // not: a purely sequential insert (oldest evicted first) would pass under
    // plain FIFO too. This proves the cache is actually LRU — `get` moves a
    // page to the front of the eviction order, not just insertion does.
    use crate::mbtiles::paged::source::{PAGE_CACHE_ENTRIES, PageSource};

    // Tiny pages, so the entry-count cap trips (the inverse of the case
    // `PAGE_CACHE_ENTRIES`'s own doc names, where 512-byte pages are what
    // hits it first) rather than the 4 MiB byte budget.
    let mut source = PageSource::new(16, 16, None);
    for number in 1..=PAGE_CACHE_ENTRIES as u32 {
        source.supply(number, &[number as u8; 16]);
    }
    // Touch page 1 so it becomes the most, not least, recently used page.
    assert!(source.get(1).is_some());
    source.supply(PAGE_CACHE_ENTRIES as u32 + 1, &[0xFFu8; 16]);
    assert!(
        source.get(1).is_some(),
        "a just-touched page must survive an eviction that follows it"
    );
    assert!(
        source.get(2).is_none(),
        "the now-least-recently-used page must be the one evicted, not page 1"
    );
    assert!(source.get(PAGE_CACHE_ENTRIES as u32 + 1).is_some());
}

#[test]
fn a_page_source_refuses_a_degenerate_run() {
    use crate::mbtiles::paged::source::PageSource;

    let source = PageSource::new(4096, 4096, None);
    assert!(source.range_for(PageRun { first: 0, count: 1 }).is_err());
    assert!(source.range_for(PageRun { first: 1, count: 0 }).is_err());
    let range = source
        .range_for(PageRun { first: 2, count: 3 })
        .expect("a real run");
    assert_eq!(range.start, 4096);
    assert_eq!(range.end, 4096 * 4);
}
