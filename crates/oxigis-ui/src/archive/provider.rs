// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! [`ArchiveTileProvider`]: a raster tile archive as a
//! [`crate::map_gpu::TileProvider`].
//!
//! Built to the [`crate::cog_provider::CogTileProvider`] pattern, because the
//! problem is the same one: a synchronous, non-blocking seam called from egui's
//! `wgpu` prepare hook, sitting on an asynchronous byte-range capability.
//!
//! ```text
//! frame N     provider.tile(t) -> None, drives the archive open / looks t up,
//!                                 asks the transport for the ranges it needs
//! (off-frame) transport fetches, deliver_range feeds the lookup, decodes the
//!             tile body and repaints
//! frame N+k   provider.tile(t) -> Some(pixels)
//! ```
//!
//! # Every answer is final
//!
//! [`oxigis_render::MapRenderer`] asks for a tile exactly once and uploads a
//! texture for whatever it gets, so this provider never answers provisionally.
//! For a given [`TileId`] it returns [`None`] until it knows which of three
//! things is true:
//!
//! * the archive holds the tile → its pixels, composited over the basemap tile
//!   when they are not fully opaque;
//! * the archive holds **no** tile there, or the archive failed → the basemap
//!   tile alone;
//! * there is no basemap → the archive's pixels, or nothing.
//!
//! A missing tile is stored as a cached [`None`], never as a failure: an
//! archive covering one city is *mostly* missing tiles, and treating that as an
//! error would burn the failure LRU and log a warning per ocean tile.

use std::sync::Arc;

use oxigis_render::{DecodedTile, RenderError, TileCache, TileId, decode_tile};
use parking_lot::Mutex;

use crate::archive::open::ArchiveContent;
use crate::archive::state::{ArchiveOutcome, ArchiveState, ArchiveStep};
use crate::cog_provider::{RangeDelivery, RangeJob, RangeSink, RangeTransport};
use crate::map_gpu::{BoxedTileProvider, TileProvider};
use crate::tile_provider::{
    FAILURE_MEMORY_TILES, MAX_ATTEMPTS, READY_CACHE_TILES, TileError, TileProviderStats,
};

pub use crate::archive::state::MAX_INFLIGHT_ARCHIVE_TILES;

/// Frames a finished, partly transparent archive tile waits for its basemap
/// tile before being drawn without one.
///
/// The same two seconds at 60 fps [`crate::cog_provider::MAX_BASE_WAIT_FRAMES`]
/// allows, and for the same reason: the [`TileProvider`] seam cannot tell "the
/// basemap tile is still loading" from "it will never arrive".
pub use crate::cog_provider::MAX_BASE_WAIT_FRAMES;

/// The provider's shared, interior-mutable state.
struct ArchiveStore {
    /// The archive open, the leaf cache and the live lookups.
    state: ArchiveState,
    /// Final answers, LRU-bounded. [`None`] means "the archive holds no tile
    /// there", which is a final answer and not a failure.
    ready: TileCache<Option<DecodedTile>>,
    /// Attempts spent per failed tile, LRU-bounded.
    failures: TileCache<u32>,
    /// Frames a completed tile has waited for its basemap counterpart.
    base_waits: TileCache<u32>,
}

/// Shared state, transport and repaint handle.
struct ArchiveInner {
    /// What the transport is pointed at: a URL for an HTTP transport, a path
    /// for a file one, ignored entirely by the in-memory one.
    location: String,
    /// Caches and lookup bookkeeping.
    store: Mutex<ArchiveStore>,
    /// Context to wake when a tile completes.
    ctx: egui::Context,
    /// The platform's range-read capability, or [`None`] for an archive that
    /// is already in memory and never asks for a range (MBTiles).
    transport: Option<Box<dyn RangeTransport>>,
}

impl ArchiveInner {
    /// Performs an [`ArchiveStep`]'s work with the store lock released.
    fn perform(self: &Arc<Self>, step: ArchiveStep) {
        if step.is_empty() {
            return;
        }
        if let Some(reason) = &step.failure {
            tracing::warn!(
                archive = self.location,
                "oxigis-ui: the raster archive could not be read: {reason}"
            );
        }
        for (tile, outcome) in step.outcomes {
            self.settle(tile, outcome);
        }
        for (range, job) in step.requests {
            let Some(transport) = self.transport.as_ref() else {
                // Unreachable: only the PMTiles backing emits ranges, and it is
                // the only one built with no transport.
                continue;
            };
            transport.request_range(self.location.clone(), range, job, self.sink());
        }
    }

    /// Records one tile's final answer.
    fn settle(&self, tile: TileId, outcome: ArchiveOutcome) {
        match outcome {
            ArchiveOutcome::Absent => {
                let mut store = self.store.lock();
                store.failures.remove(&tile);
                store.ready.insert(tile, None);
            }
            ArchiveOutcome::Body(bytes) => match decode_tile(&bytes) {
                Ok(decoded) => {
                    let mut store = self.store.lock();
                    store.failures.remove(&tile);
                    store.ready.insert(tile, Some(decoded));
                }
                Err(error) => self.record_failure(
                    tile,
                    &TileError::permanent(format!("tile decode failed: {error}")),
                ),
            },
            ArchiveOutcome::Failed(error) => self.record_failure(tile, &error),
        }
    }

    /// Records a failed tile, so it is not retried for ever.
    fn record_failure(&self, tile: TileId, error: &TileError) {
        let attempts = {
            let mut store = self.store.lock();
            let spent = store.failures.peek(&tile).copied().unwrap_or(0);
            let attempts = if error.retryable() {
                spent.saturating_add(1)
            } else {
                MAX_ATTEMPTS
            };
            store.failures.insert(tile, attempts);
            attempts
        };
        tracing::warn!(
            z = tile.z,
            x = tile.x,
            y = tile.y,
            attempts,
            "oxigis-ui: archive tile failed: {error}"
        );
    }

    /// A sink pointed at this reader.
    fn sink(self: &Arc<Self>) -> RangeSink {
        RangeSink::from_delivery(Arc::clone(self) as Arc<dyn RangeDelivery>)
    }
}

impl RangeDelivery for ArchiveInner {
    fn deliver_range(self: Arc<Self>, job: RangeJob, result: Result<Vec<u8>, TileError>) {
        let step = self.store.lock().state.deliver(job, result);
        self.perform(step);
        self.ctx.request_repaint();
    }
}

/// What [`ArchiveTileProvider::tile`] decided to return.
enum Outcome {
    /// Nothing final yet.
    Wait,
    /// The archive holds no tile here (or could not be read).
    BaseOnly,
    /// The archive's pixels for this tile.
    Archive(DecodedTile),
}

/// A [`TileProvider`] that draws a raster tile archive, optionally over another
/// provider.
///
/// Construct one per archive layer and install it with
/// [`crate::map_gpu::replace_provider`]. See the [module docs](crate::archive)
/// for the protocol and the resource bounds.
pub struct ArchiveTileProvider {
    /// Shared state, transport and repaint handle.
    inner: Arc<ArchiveInner>,
    /// The provider drawn underneath, usually an [`crate::XyzTileProvider`].
    base: Option<BoxedTileProvider>,
}

impl core::fmt::Debug for ArchiveTileProvider {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ArchiveTileProvider")
            .field("location", &self.inner.location)
            .field("has_base", &self.base.is_some())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl ArchiveTileProvider {
    /// Builds a provider reading the PMTiles archive at `location`, waking
    /// `ctx` whenever a tile completes.
    ///
    /// `location` is whatever the transport addresses: a URL for the HTTP
    /// transports, a filesystem path for a file transport, and anything at all
    /// for [`crate::archive::MemoryRangeTransport`], which ignores it.
    ///
    /// The archive is **not** read here — the open is kicked from the frame
    /// loop — so a provider that is built and never drawn issues no request.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if the compile-time cache
    /// bounds are degenerate (unreachable with the constants in this crate).
    pub fn pmtiles(
        location: impl Into<String>,
        ctx: &egui::Context,
        transport: Box<dyn RangeTransport>,
    ) -> Result<Self, RenderError> {
        let store = ArchiveStore {
            state: ArchiveState::new(Some(ArchiveContent::Raster)),
            ready: TileCache::new(READY_CACHE_TILES)?,
            failures: TileCache::new(FAILURE_MEMORY_TILES)?,
            base_waits: TileCache::new(READY_CACHE_TILES)?,
        };
        Ok(Self {
            inner: Arc::new(ArchiveInner {
                location: location.into(),
                store: Mutex::new(store),
                ctx: ctx.clone(),
                transport: Some(transport),
            }),
            base: None,
        })
    }

    /// Builds a provider drawing an MBTiles archive already open in memory.
    ///
    /// No transport: a SQLite image is walked in place, so every answer is
    /// synchronous and there is no range to ask anyone for. Everything else —
    /// the final-answer contract, the caches, the compositing over a basemap —
    /// is the same code the PMTiles path runs.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if the compile-time cache
    /// bounds are degenerate (unreachable with the constants in this crate). An
    /// archive whose tiles turn out to be *vector* is not refused here but on
    /// the first tile, with the same named message the PMTiles path gives.
    pub fn mbtiles(
        location: impl Into<String>,
        reader: Arc<crate::mbtiles::MbTilesReader>,
        ctx: &egui::Context,
    ) -> Result<Self, RenderError> {
        let store = ArchiveStore {
            state: ArchiveState::mbtiles(reader, Some(ArchiveContent::Raster)),
            ready: TileCache::new(READY_CACHE_TILES)?,
            failures: TileCache::new(FAILURE_MEMORY_TILES)?,
            base_waits: TileCache::new(READY_CACHE_TILES)?,
        };
        Ok(Self {
            inner: Arc::new(ArchiveInner {
                location: location.into(),
                store: Mutex::new(store),
                ctx: ctx.clone(),
                transport: None,
            }),
            base: None,
        })
    }

    /// Builds a provider reading the MBTiles archive at `location` **a page at
    /// a time**, through a range transport.
    ///
    /// The twin of [`Self::pmtiles`], and deliberately identical in shape: a
    /// multi-gigabyte `.mbtiles` opens in one 16 KiB read and costs a handful of
    /// page reads per tile, whatever its size. [`Self::mbtiles`] is for bytes
    /// already in hand (a browser drop); this is for everything else.
    ///
    /// `declared_total` is the file length the caller pinned, when it knows one;
    /// it only ever *bounds* a page count the archive's own header could not
    /// vouch for.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if the compile-time cache
    /// bounds are degenerate (unreachable with the constants in this crate).
    pub fn paged_mbtiles(
        location: impl Into<String>,
        ctx: &egui::Context,
        transport: Box<dyn RangeTransport>,
        declared_total: Option<u64>,
    ) -> Result<Self, RenderError> {
        let store = ArchiveStore {
            state: ArchiveState::paged_mbtiles(Some(ArchiveContent::Raster), declared_total),
            ready: TileCache::new(READY_CACHE_TILES)?,
            failures: TileCache::new(FAILURE_MEMORY_TILES)?,
            base_waits: TileCache::new(READY_CACHE_TILES)?,
        };
        Ok(Self {
            inner: Arc::new(ArchiveInner {
                location: location.into(),
                store: Mutex::new(store),
                ctx: ctx.clone(),
                transport: Some(transport),
            }),
            base: None,
        })
    }

    /// Draws `base` underneath the archive.
    #[must_use]
    pub fn with_base(mut self, base: BoxedTileProvider) -> Self {
        self.base = Some(base);
        self
    }

    /// What the transport is pointed at.
    #[must_use]
    pub fn location(&self) -> &str {
        &self.inner.location
    }

    /// Why the archive could not be read, if it could not be.
    #[must_use]
    pub fn failure(&self) -> Option<String> {
        self.inner.store.lock().state.failure().map(str::to_owned)
    }

    /// Whether the archive's header has landed and been accepted.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.inner.store.lock().state.is_open()
    }

    /// Snapshot of what the provider is holding.
    #[must_use]
    pub fn stats(&self) -> TileProviderStats {
        let store = self.inner.store.lock();
        TileProviderStats {
            ready: store.ready.len(),
            inflight: store.state.active_len() + store.state.queued_len(),
            failed: store.failures.len(),
        }
    }

    /// How many decoded leaf directories are held, and what they cost in bytes.
    #[must_use]
    pub fn leaf_stats(&self) -> (usize, usize) {
        self.inner.store.lock().state.leaf_stats()
    }

    /// Decides what `tile` should resolve to, returning the archive work to
    /// perform *after* the lock is released — a transport may answer
    /// synchronously, which would otherwise deadlock.
    fn decide(&self, tile: TileId) -> (Outcome, ArchiveStep) {
        let mut store = self.inner.store.lock();
        if store.state.failure().is_some() {
            return (Outcome::BaseOnly, ArchiveStep::default());
        }
        if let Some(entry) = store.ready.get(&tile) {
            return match entry {
                Some(decoded) => (Outcome::Archive(decoded.clone()), ArchiveStep::default()),
                None => (Outcome::BaseOnly, ArchiveStep::default()),
            };
        }
        if store
            .failures
            .peek(&tile)
            .is_some_and(|attempts| *attempts >= MAX_ATTEMPTS)
        {
            return (Outcome::BaseOnly, ArchiveStep::default());
        }
        if store.state.is_opening() {
            // Kick the open, but do not queue the tile: unlike the vector
            // transport this provider is asked again every frame, so queueing
            // every visible tile against a header that has not landed would
            // fill the lookup queue with addresses the archive may not even
            // cover.
            return (Outcome::Wait, store.state.kick());
        }
        let step = store.state.request_tile(tile);
        (Outcome::Wait, step)
    }

    /// Counts a frame spent waiting for the basemap tile under a finished
    /// archive tile, returning whether the wait has run out.
    fn base_wait_expired(&self, tile: TileId) -> bool {
        let mut store = self.inner.store.lock();
        let waited = store.base_waits.peek(&tile).copied().unwrap_or(0);
        store.base_waits.insert(tile, waited.saturating_add(1));
        waited >= MAX_BASE_WAIT_FRAMES
    }
}

impl TileProvider for ArchiveTileProvider {
    fn tile(&self, tile: TileId) -> Option<DecodedTile> {
        // Asked for first and unconditionally: the basemap provider starts its
        // own fetch when asked, so skipping this while the archive opens would
        // leave the basemap idle underneath.
        let base = self.base.as_ref().and_then(|provider| provider.tile(tile));

        let (outcome, step) = self.decide(tile);
        self.inner.perform(step);

        match outcome {
            Outcome::Wait => None,
            Outcome::BaseOnly => base,
            Outcome::Archive(decoded) => {
                let opaque = decoded
                    .rgba()
                    .chunks_exact(4)
                    .all(|pixel| pixel[3] == u8::MAX);
                if opaque || self.base.is_none() {
                    return Some(decoded);
                }
                match base {
                    Some(base) => match crate::cog_provider::blend_over(&decoded, &base) {
                        Ok(blended) => Some(blended),
                        Err(error) => {
                            tracing::warn!("oxigis-ui: archive/basemap blend failed: {error}");
                            Some(decoded)
                        }
                    },
                    None if self.base_wait_expired(tile) => Some(decoded),
                    None => None,
                }
            }
        }
    }
}
