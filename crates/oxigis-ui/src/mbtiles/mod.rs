// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! [`MbTilesReader`]: an MBTiles archive read out of an in-memory SQLite image.
//!
//! MBTiles is a tile pyramid in a SQLite database. This module reads one with
//! the b-tree walker `gpkg_input` already owns — no SQL engine, no query
//! planner, no new dependency, and the same file compiles for
//! `wasm32-unknown-unknown` today.
//!
//! # Why not a SQLite crate
//!
//! `oxisql-sqlite-compat` 0.4.0 was measured before this module was written,
//! the way [`crate::gpkg_input`] measured `oxigeo-gpkg` before it:
//!
//! | | the in-repo reader | `oxisql-sqlite-compat` 0.4.0 |
//! |---|---|---|
//! | new crates in the graph | **0** | ~55 (tokio, futures, regex, chrono, miette, uuid, rand…) |
//! | `wasm32-unknown-unknown` | **builds today** | fails (`getrandom` needs `wasm_js`; fixing that, `uuid` fails too) |
//! | a read-only file | works (bytes in) | `open()` fails, os error 5 |
//! | side effects | none | writes a `-wal` sidecar at open |
//!
//! `oxigis-ui` must keep building for the browser, so the second column is
//! disqualifying on its own. The consequence is a good one: because the reader
//! takes **bytes**, MBTiles is not desktop-only — a `.mbtiles` dropped in a
//! browser tab works.
//!
//! # Shape of the read
//!
//! One `SqliteDb::scan_prefixes` pass builds a
//! sorted index of `(key, rowid)` where `key = z<<48 | x<<24 | y`, **already
//! flipped to XYZ row order** so every later lookup is flip-free. That pass
//! never follows an overflow chain, so it never materialises a tile body: an
//! archive of a hundred thousand tiles costs one walk and 16 bytes per tile.
//! A lookup is then a binary search plus one `SqliteDb::seek_row` — one page
//! per b-tree level — for the blob.
//!
//! # Both schemas in circulation
//!
//! * **flat**: a `tiles` table of `(zoom_level, tile_column, tile_row,
//!   tile_data)`.
//! * **normalized**: `tiles` is a *view* over `map` ⋈ `images`, which is what
//!   every tippecanoe and mbutil archive looks like. `map` holds
//!   `(zoom_level, tile_column, tile_row, tile_id)` and `images` holds
//!   `(tile_data, tile_id)`; they are joined in memory here.
//!
//! Anything else is refused with the schema named, because guessing would draw
//! the wrong tiles rather than none.
//!
//! # Two readers, and which one you get
//!
//! This one is the **resident** reader: bytes in hand, whole image in memory. It
//! is what a browser drop gets, because a browser has no filesystem and the
//! bytes arrived as bytes anyway.
//!
//! Everything else — a local path, a URL — gets the **paged** reader in this
//! module's `paged` submodule, which opens in one 16 KiB read and then costs a
//! handful of page reads per tile whatever the archive's size. That is what
//! retired the v1.3 claim that a `.mbtiles` cannot be read over HTTP `Range`
//! requests: measured, a flat archive costs 2.33 requests per tile cold and
//! 0.33 warm, within ~2.3x of PMTiles.
//!
//! # Bounded, because *this* reader holds the whole file in memory
//!
//! The workspace forbids `unsafe`, so memory-mapping is out and the resident
//! reader's image is read whole. [`MAX_MBTILES_BYTES`] and
//! [`MAX_MBTILES_TILES`] are therefore **resident-mode** ceilings: they bound
//! what is held in memory and what is indexed up front. Neither applies to the
//! paged reader, which holds a few megabytes of pages and builds no index at
//! all — the archive's own b-trees are the index.

pub(crate) mod paged;
pub(crate) mod schema;

// Also built under the `fixtures` feature, which is how a crate that cannot
// reach a `#[cfg(test)]` module — `oxigis-desktop` is bin-only — gets
// `sample_mbtiles_raster` below. Stays `pub(crate)`: the feature adds exactly
// one public item to this crate, not thirty test-scaffolding builders.
#[cfg(any(test, feature = "fixtures"))]
pub(crate) mod fixture;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use oxigis_core::VectorTilePaint;
use oxigis_render::TileId;
use oxigis_render::source::tms_row;

use crate::archive::{ArchiveContent, ArchiveInfo, archive_paints};
use crate::gpkg_input::sqlite::SqliteDb;
use crate::local_vector::LocalVectorError;

pub use crate::mbtiles::schema::MbTilesFormat;
use crate::mbtiles::schema::{Layout, Metadata};

/// Largest MBTiles image the **resident** reader holds in memory.
///
/// Scoped to this reader, and only this reader: `SqliteDb` walks an image, and
/// `unsafe_code = "forbid"` at workspace level rules out memory-mapping a file
/// another process may truncate. 512 MiB covers a city, a prefecture or a small
/// country.
///
/// It is no longer a ceiling on *reading a `.mbtiles`*. A path or a URL goes
/// through this module's `paged` reader, which opens an archive of any size in one
/// 16 KiB read; this bounds only the case where the bytes are already in memory
/// because that is how they arrived — a file dropped into a browser tab, which
/// has no filesystem to stream from.
pub const MAX_MBTILES_BYTES: usize = 512 * 1024 * 1024;

/// Largest tile index the **resident** reader builds: 4 Mi entries × 16 B ≈ 64 MiB.
///
/// Reached only by an archive whose whole pyramid is dense past zoom 11, which
/// at [`MAX_MBTILES_BYTES`] would mean sub-kilobyte tiles throughout. Refusing
/// past this bounds the *index*, which unlike the image is built rather than
/// read.
///
/// Does not apply to this module's `paged` reader, which builds no index at all.
pub const MAX_MBTILES_TILES: usize = 4 * 1024 * 1024;

/// A small indexed raster `.mbtiles`, for tests in crates that cannot reach
/// this module's `#[cfg(test)]` fixtures — the MBTiles twin of
/// `oxigis_render::pmtiles::sample_pmtiles_raster`.
///
/// Three real 2×2 PNGs at MBTiles rows `0/0/0`, `1/0/0` and `1/0/1`, so both
/// readers have something to decode and the TMS flip has two zoom-1 rows to get
/// wrong. **Indexed**, not flat: the paged survey refuses an archive carrying no
/// `(zoom_level, tile_column, tile_row)` index by name, so a flat image would be
/// refused rather than read. The declared zoom range is `raster_metadata`'s,
/// minzoom 0 to maxzoom 2.
///
/// Deterministic — hand-assembled bytes, no clock and no RNG — so a caller may
/// write it to a temp file and compare against it.
#[cfg(any(test, feature = "fixtures"))]
#[must_use]
pub fn sample_mbtiles_raster() -> Vec<u8> {
    fixture::indexed_flat_image(
        fixture::PAGE_SIZE,
        &[
            (0, 0, 0, fixture::tiny_png()),
            (1, 0, 0, fixture::tiny_png()),
            (1, 0, 1, fixture::tiny_png()),
        ],
        &fixture::raster_metadata(),
        false,
    )
}

/// Bits each of `x` and `y` gets in an index key.
///
/// Must be at least `oxigis_render::MAX_ZOOM`: a column or row needs `z` bits
/// at zoom `z`, and one bit short would let a deep tile's `x` alias into the
/// zoom field — a lookup answering with a different zoom's body, silently. The
/// assertion below is what makes that a build failure rather than a wrong map,
/// so `MAX_ZOOM` may be raised without auditing this file by hand.
///
/// The key still fits a `u64` with room to spare: `MAX_ZOOM` needs 5 bits above
/// the two coordinate fields, leaving 11 spare at 24 bits each.
const COORD_BITS: u32 = 24;

const _: () = assert!(
    COORD_BITS >= oxigis_render::MAX_ZOOM as u32,
    "an index key must hold every column and row `MAX_ZOOM` allows"
);
const _: () = assert!(
    COORD_BITS * 2 + 8 <= u64::BITS,
    "an index key must hold the zoom above both coordinate fields"
);

/// The key an index entry is sorted and searched by.
///
/// `z << 48 | x << 24 | y`, with `y` **already** in XYZ order: the TMS flip
/// happens once, at index build, so a lookup never has to remember it.
const fn index_key(z: u8, x: u32, y: u32) -> u64 {
    ((z as u64) << (COORD_BITS * 2)) | ((x as u64) << COORD_BITS) | (y as u64)
}

/// The packing invariant on its own, next to the constant it constrains; the
/// archive-level tests live in [`tests`].
#[cfg(test)]
mod index_key_tests {
    use super::{COORD_BITS, index_key};

    #[test]
    fn the_deepest_address_cannot_alias_into_another_zoom() {
        let max_zoom = oxigis_render::MAX_ZOOM;
        let field = (1u64 << COORD_BITS) - 1;
        let last = (1u32 << max_zoom) - 1;

        // The three fields stay separable at the deepest addressable tile.
        let key = index_key(max_zoom, last, last);
        assert_eq!(key >> (COORD_BITS * 2), u64::from(max_zoom));
        assert_eq!((key >> COORD_BITS) & field, u64::from(last));
        assert_eq!(key & field, u64::from(last));

        // The bug a too-narrow coordinate field would cause: one zoom's widest
        // legal address reaching into the next zoom's range. Every level's keys
        // must stay wholly below the level beneath it.
        for zoom in 0..=max_zoom {
            let widest = (1u32 << zoom) - 1;
            assert!(
                index_key(zoom, widest, widest) < index_key(zoom + 1, 0, 0),
                "zoom {zoom} must stay below zoom {}",
                zoom + 1,
            );
        }
    }

    #[test]
    fn keys_sort_by_zoom_then_column_then_row() {
        // The binary search in `MbTilesReader` depends on this order, not just
        // on the fields being recoverable.
        let mut keys = [
            index_key(1, 1, 1),
            index_key(0, 0, 0),
            index_key(1, 0, 3),
            index_key(1, 0, 2),
        ];
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                index_key(0, 0, 0),
                index_key(1, 0, 2),
                index_key(1, 0, 3),
                index_key(1, 1, 1),
            ]
        );
    }
}

/// An MBTiles archive, opened and indexed.
///
/// Cheap to clone the [`Arc`] behind it; the image and the index are shared.
#[derive(Debug)]
pub struct MbTilesReader {
    /// The whole SQLite image.
    image: Arc<[u8]>,
    /// How the archive stores its tiles.
    layout: Layout,
    /// `(key, rowid)` sorted by key — the whole pyramid's addresses.
    index: Vec<(u64, i64)>,
    /// What the `metadata` table declared.
    metadata: Metadata,
}

impl MbTilesReader {
    /// Opens and indexes the archive in `image`.
    ///
    /// # Errors
    ///
    /// Refuses, each by name: an image past [`MAX_MBTILES_BYTES`]; anything
    /// `SqliteDb::open` refuses; a schema that is neither the flat nor the
    /// normalized shape; a `metadata.format` this build cannot draw; and a
    /// pyramid past [`MAX_MBTILES_TILES`].
    pub fn open(image: Arc<[u8]>) -> Result<Self, LocalVectorError> {
        if image.len() > MAX_MBTILES_BYTES {
            return Err(LocalVectorError::new(format!(
                "the archive is {} MiB, past the {} MiB OxiGIS holds in memory; \
                 open it from disk or from a URL instead, which streams it a page at a time",
                image.len() / (1024 * 1024),
                MAX_MBTILES_BYTES / (1024 * 1024),
            )));
        }
        let (layout, metadata, index) = {
            let db = SqliteDb::open(&image)?;
            let layout = Layout::detect(&db)?;
            let metadata = Metadata::read(&db, &layout)?;
            let index = layout.build_index(&db)?;
            (layout, metadata, index)
        };
        Ok(Self {
            image,
            layout,
            index,
            metadata,
        })
    }

    /// Archive-level facts, in the shape the archive layer everywhere else
    /// speaks.
    #[must_use]
    pub fn info(&self) -> ArchiveInfo {
        self.metadata.info()
    }

    /// What the `metadata` table declared, verbatim.
    #[must_use]
    pub fn metadata(&self) -> &std::collections::BTreeMap<String, String> {
        self.metadata.entries()
    }

    /// How many addresses the archive answers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the archive holds no tiles at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Default paint rules for this archive's declared source layers.
    #[must_use]
    pub fn paints(&self) -> Vec<VectorTilePaint> {
        archive_paints(&self.metadata.vector_layers)
    }

    /// Whether the archive could hold `tile` at all — the zoom gate.
    #[must_use]
    pub fn covers(&self, tile: TileId) -> bool {
        tile.z >= self.metadata.min_zoom && tile.z <= self.metadata.max_zoom
    }

    /// The tile stored at `tile`, in **XYZ** addressing.
    ///
    /// `Ok(None)` means the archive holds no tile there: a final, non-error
    /// answer, exactly like [`oxigis_render::TileLookup::Absent`].
    ///
    /// # Errors
    ///
    /// Propagates a b-tree failure — a truncated page, a cycle, a broken
    /// overflow chain — which is a property of the file and never gets better
    /// on a retry.
    pub fn tile(&self, tile: TileId) -> Result<Option<Vec<u8>>, LocalVectorError> {
        if !self.covers(tile) {
            return Ok(None);
        }
        let key = index_key(tile.z, tile.x, tile.y);
        let Ok(position) = self.index.binary_search_by_key(&key, |(key, _)| *key) else {
            return Ok(None);
        };
        let Some((_, rowid)) = self.index.get(position) else {
            return Ok(None);
        };
        let db = SqliteDb::open(&self.image)?;
        self.layout.read_blob(&db, *rowid)
    }
}

/// The XYZ row an MBTiles `tile_row` names at zoom `z`.
///
/// MBTiles counts rows from the **south**, the map does not, and the flip is
/// the single most-repeated bug in every MBTiles reader ever written. It is
/// applied exactly once, at index build, through the same
/// [`oxigis_render::source::tms_row`] the `{-y}` URL placeholder expands with —
/// one rule, never re-derived.
fn xyz_row(z: u8, tile_row: u32) -> u32 {
    tms_row(z, tile_row)
}

/// Whether `format` names a raster codec, a vector one, or neither.
///
/// # Errors
///
/// Refuses an unrecognised `format` by name rather than guessing: the value
/// decides which provider draws the archive, and drawing MVT through the raster
/// path produces a blank map with no explanation.
pub fn content_for_format(format: &str) -> Result<ArchiveContent, LocalVectorError> {
    match format.trim().to_ascii_lowercase().as_str() {
        "pbf" | "mvt" | "vector" => Ok(ArchiveContent::Vector),
        "png" | "jpg" | "jpeg" | "webp" => Ok(ArchiveContent::Raster),
        other => Err(LocalVectorError::new(format!(
            "the archive's metadata declares format \"{other}\", which OxiGIS does not draw \
             (pbf, mvt, png, jpg, jpeg or webp)"
        ))),
    }
}
