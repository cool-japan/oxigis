// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Single-file tile archives (PMTiles v3, MBTiles): the shell-agnostic half of
//! the read loop.
//!
//! An archive is a *layer*, not a basemap — the [`crate::cog_provider`]
//! precedent — and it holds either raster tiles or vector ones, which is only
//! known once its header has been read. That single fact shapes this whole
//! module:
//!
//! ```text
//! + Archive URL  ->  ArchiveProbe  ->  ArchiveInfo (raster or vector?)
//!                                        |
//!                    +-------------------+-------------------+
//!                    v                                       v
//!            ArchiveTileProvider                   ArchiveTileTransport
//!            (impl TileProvider)                   (impl TileTransport)
//!            raster, composited over               feeds the EXISTING
//!            the basemap like a COG                VectorTileProvider
//! ```
//!
//! # Two consumers, one state machine
//!
//! Raster archives get their own [`crate::map_gpu::TileProvider`], built to the
//! [`crate::cog_provider::CogTileProvider`] pattern verbatim: a
//! `TileCache<Option<DecodedTile>>` where [`None`] is a **final** answer, an
//! optional base provider composited underneath, and a frame budget for waiting
//! on the base tile.
//!
//! Vector archives instead implement [`crate::TileTransport`] — "given a tile,
//! eventually produce bytes" — and hand those bytes to the **unchanged**
//! [`crate::vector_provider::VectorTileProvider`]. One trait impl therefore
//! buys MVT decode, the gzip sniff, `lyon` tessellation, the label table, the
//! bounded caches, the retry policy *and* PDF-export parity, none of it
//! re-implemented.
//!
//! Both sit on one lookup state machine (`archive::state::ArchiveState`,
//! private), so
//! the leaf-directory walk, the depth bound and the concurrency cap exist once.
//!
//! # A missing tile is not a failure
//!
//! Archives are legitimately sparse: an ocean tile simply is not stored. The
//! reader answers such an address [`oxigis_render::TileLookup::Absent`], which
//! travels to the raster provider as a cached [`None`] and to the vector side
//! as [`crate::TileSink::deliver_absent`] — never as an error, never retried,
//! never logged.
//!
//! # Bounded resources
//!
//! | Resource | Bound | Constant |
//! |---|---|---|
//! | Tile lookups in flight at once | 8 | [`MAX_INFLIGHT_ARCHIVE_TILES`] |
//! | Decoded leaf directories held | 16 MiB / 64 entries | [`LEAF_CACHE_BYTES`], [`LEAF_CACHE_ENTRIES`] |
//! | Directories consulted per lookup | 2 | [`oxigis_render::pmtiles::MAX_DIRECTORY_DEPTH`] |
//! | Composed raster tiles held | 64 (LRU) | [`crate::tile_provider::READY_CACHE_TILES`] |
//! | Remembered failures | 1024 (LRU) | [`crate::tile_provider::FAILURE_MEMORY_TILES`] |
//!
//! # Refusals, by name, once, at open
//!
//! Everything an archive can declare that this build will not read is refused
//! when the header lands rather than once per tile — brotli/zstd coding (both
//! banned by the workspace's `deny.toml`), AVIF tile bodies (no pure-Rust
//! decoder is in the graph), an undeclared tile type or tile codec, and an
//! archive whose content is the *other* kind from the layer that asked for it.
//! Each carries its own sentence, because "the map is blank" is not a
//! diagnosis.
//!
//! # Where the bytes come from
//!
//! Nothing here performs I/O. A PMTiles archive is read through the same
//! [`crate::RangeTransport`] the COG reader uses — desktop `ureq`, browser
//! `fetch()`, [`MemoryRangeTransport`] for a dropped file and for every test in
//! this crate — so remote and local archives share one code path and differ
//! only in the transport handed in.

mod config;
mod leaf;
mod memory;
mod open;
mod paged_mbtiles_state;
pub mod paints;
mod pmtiles_state;
mod probe;
mod provider;
mod state;
mod transport;

#[cfg(test)]
mod tests;

pub use crate::archive::config::ArchiveLayerConfig;
pub use crate::archive::leaf::{LEAF_CACHE_BYTES, LEAF_CACHE_ENTRIES};
pub use crate::archive::memory::MemoryRangeTransport;
pub use crate::archive::open::ArchiveContent;
pub use crate::archive::paints::{archive_paints, default_archive_paints};
pub use crate::archive::probe::{ArchiveInfo, ArchiveProbe, OpenedArchive};
pub use crate::archive::provider::{ArchiveTileProvider, MAX_INFLIGHT_ARCHIVE_TILES};
pub use crate::archive::transport::ArchiveTileTransport;
