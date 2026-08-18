// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! [`ArchiveTileTransport`]: a vector tile archive as a
//! [`crate::TileTransport`].
//!
//! This is the highest-leverage twenty lines in the archive work. The
//! [`crate::TileTransport`] seam is "given a tile, eventually produce bytes",
//! and that is *exactly* what an archive does — so implementing it hands a
//! PMTiles archive to the **unchanged**
//! [`crate::vector_provider::VectorTileProvider`], which brings with it:
//!
//! * the gzip sniff (an archive's MVT bodies are routinely gzipped, and a
//!   `tile_compression = Gzip` archive is decoded here before the sniff even
//!   sees it — belt and braces, both correct);
//! * [`oxigis_render::decode_mvt`] and `lyon` tessellation;
//! * the label table and the per-frame re-tessellation budget;
//! * the bounded decoded/mesh/failure caches and the retry policy;
//! * **PDF export parity** — `print/mvt.rs` draws from
//!   [`crate::vector_provider::VectorTileSource::decoded`], which is the same
//!   seam, so an archive-backed vector layer prints with no print-path change
//!   at all.
//!
//! None of it is re-implemented, and none of it can drift.
//!
//! # The URL is a sentinel
//!
//! An archive addresses tiles by [`TileId`] inside one already-named file, so
//! there is no per-tile URL. The provider hands
//! [`crate::tile_provider::ARCHIVE_TILE_URL`] instead of an empty string, so a
//! URL transport that ever receives one can *say so* rather than fetch nothing;
//! this transport ignores it entirely.
//!
//! # A missing tile is `deliver_absent`
//!
//! An archive that covers one country answers most of the world
//! [`oxigis_render::TileLookup::Absent`]. That reaches
//! [`crate::TileSink::deliver_absent`], which the vector sink overrides to
//! insert an **empty** tile: nothing is drawn there, nothing is logged, and the
//! failure LRU is untouched.

use std::collections::HashMap;
use std::sync::Arc;

use oxigis_render::TileId;
use parking_lot::Mutex;

use crate::archive::open::ArchiveContent;
use crate::archive::state::{ArchiveOutcome, ArchiveState, ArchiveStep};
use crate::cog_provider::{RangeDelivery, RangeJob, RangeSink, RangeTransport};
use crate::tile_provider::{TileError, TileSink, TileTransport};

/// Shared state, transport and the sinks waiting for answers.
struct TransportInner {
    /// What the range transport is pointed at.
    location: String,
    /// The archive open, the leaf cache and the live lookups.
    state: Mutex<ArchiveState>,
    /// Where each in-flight tile's answer goes.
    ///
    /// Separate from `state`'s lock because the sink is handed *back* into the
    /// vector provider, whose delivery takes its own lock — nesting the two
    /// would order this crate's two mutexes in both directions.
    sinks: Mutex<HashMap<TileId, TileSink>>,
    /// The platform's range-read capability, or [`None`] for an archive that
    /// is already in memory and never asks for a range (MBTiles).
    transport: Option<Box<dyn RangeTransport>>,
}

impl TransportInner {
    /// Performs an [`ArchiveStep`]'s work with both locks released.
    fn perform(self: &Arc<Self>, step: ArchiveStep) {
        if step.is_empty() {
            return;
        }
        if let Some(reason) = &step.failure {
            tracing::warn!(
                archive = self.location,
                "oxigis-ui: the vector archive could not be read: {reason}"
            );
        }
        for (tile, outcome) in step.outcomes {
            let Some(sink) = self.sinks.lock().remove(&tile) else {
                continue;
            };
            match outcome {
                // The gzip sniff in the vector sink still runs on these bytes,
                // which is harmless: an archive that declares
                // `tile_compression = Gzip` has already been inflated here, and
                // the inflated MVT does not start with the gzip magic.
                ArchiveOutcome::Body(bytes) => sink.deliver(tile, Ok(bytes)),
                ArchiveOutcome::Absent => sink.deliver_absent(tile),
                ArchiveOutcome::Failed(error) => sink.deliver(tile, Err(error)),
            }
        }
        for (range, job) in step.requests {
            let Some(transport) = self.transport.as_ref() else {
                // Unreachable: only the PMTiles backing emits ranges.
                continue;
            };
            transport.request_range(self.location.clone(), range, job, self.range_sink());
        }
    }

    /// A range sink pointed at this reader.
    fn range_sink(self: &Arc<Self>) -> RangeSink {
        RangeSink::from_delivery(Arc::clone(self) as Arc<dyn RangeDelivery>)
    }
}

impl RangeDelivery for TransportInner {
    fn deliver_range(self: Arc<Self>, job: RangeJob, result: Result<Vec<u8>, TileError>) {
        let step = self.state.lock().deliver(job, result);
        self.perform(step);
    }
}

/// A [`TileTransport`] that resolves each tile inside one open archive.
///
/// Hand it to [`crate::vector_provider::VectorTileProvider::new`] in place of a
/// platform HTTP transport and the whole vector path — decode, tessellation,
/// labels, caches, retry, print — works over a PMTiles archive unchanged.
///
/// # Cloning is a second handle, not a second reader
///
/// Every field lives behind one [`Arc`], so a clone shares the archive's open,
/// its leaf cache and its in-flight lookups — exactly like
/// [`crate::archive::MemoryRangeTransport`]. That is what lets a caller keep an
/// observation handle ([`Self::leaf_stats`], [`Self::failure`]) after handing
/// the transport itself to a provider, which takes it by [`Box`].
#[derive(Clone)]
pub struct ArchiveTileTransport {
    /// Shared state, transport and pending sinks.
    inner: Arc<TransportInner>,
}

impl core::fmt::Debug for ArchiveTileTransport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ArchiveTileTransport")
            .field("location", &self.inner.location)
            .finish_non_exhaustive()
    }
}

impl ArchiveTileTransport {
    /// Serves tiles out of the PMTiles archive at `location`.
    ///
    /// `location` is whatever the range transport addresses — a URL, a
    /// filesystem path, or nothing at all for
    /// [`crate::archive::MemoryRangeTransport`]. The archive is opened lazily,
    /// on the first tile the vector provider asks for, so building a transport
    /// costs no I/O.
    #[must_use]
    pub fn pmtiles(location: impl Into<String>, transport: Box<dyn RangeTransport>) -> Self {
        Self {
            inner: Arc::new(TransportInner {
                location: location.into(),
                state: Mutex::new(ArchiveState::new(Some(ArchiveContent::Vector))),
                sinks: Mutex::new(HashMap::new()),
                transport: Some(transport),
            }),
        }
    }

    /// Serves tiles out of an MBTiles archive already open in memory.
    ///
    /// The synchronous twin of [`Self::pmtiles`], and the reason the vector
    /// path needs no MBTiles-specific code at all: the provider above still
    /// only ever sees "given a tile, eventually bytes".
    #[must_use]
    pub fn mbtiles(
        location: impl Into<String>,
        reader: Arc<crate::mbtiles::MbTilesReader>,
    ) -> Self {
        Self {
            inner: Arc::new(TransportInner {
                location: location.into(),
                state: Mutex::new(ArchiveState::mbtiles(reader, Some(ArchiveContent::Vector))),
                sinks: Mutex::new(HashMap::new()),
                transport: None,
            }),
        }
    }

    /// Serves tiles out of an MBTiles archive read **a page at a time**.
    ///
    /// The twin of [`Self::pmtiles`], and the reason the vector path needs no
    /// paged-MBTiles-specific code at all: the provider above still only ever
    /// sees "given a tile, eventually bytes".
    ///
    /// `declared_total` is the file length the caller pinned, when it knows one.
    #[must_use]
    pub fn paged_mbtiles(
        location: impl Into<String>,
        transport: Box<dyn RangeTransport>,
        declared_total: Option<u64>,
    ) -> Self {
        Self {
            inner: Arc::new(TransportInner {
                location: location.into(),
                state: Mutex::new(ArchiveState::paged_mbtiles(
                    Some(ArchiveContent::Vector),
                    declared_total,
                )),
                sinks: Mutex::new(HashMap::new()),
                transport: Some(transport),
            }),
        }
    }

    /// What the range transport is pointed at.
    #[must_use]
    pub fn location(&self) -> &str {
        &self.inner.location
    }

    /// Why the archive could not be read, if it could not be.
    #[must_use]
    pub fn failure(&self) -> Option<String> {
        self.inner.state.lock().failure().map(str::to_owned)
    }

    /// Whether the archive's header has landed and been accepted.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.inner.state.lock().is_open()
    }

    /// How many leaf directories are held, and what they cost in bytes.
    ///
    /// The twin of [`crate::archive::ArchiveTileProvider::leaf_stats`], and
    /// additive for the same reason: a leaf hop is invisible from above — the
    /// vector provider only ever sees "eventually, bytes" — so nothing else can
    /// say whether a pan was served out of the cache or out of the network.
    /// Always `(0, 0)` for an MBTiles backing, which has no leaf directories at
    /// all.
    #[must_use]
    pub fn leaf_stats(&self) -> (usize, usize) {
        self.inner.state.lock().leaf_stats()
    }
}

impl TileTransport for ArchiveTileTransport {
    fn request(&self, tile: TileId, _url: String, sink: TileSink) {
        // Registered BEFORE the lookup starts: an in-memory transport answers
        // synchronously, so the outcome can come back before `request_tile`
        // returns and would otherwise find no sink to deliver through.
        self.inner.sinks.lock().insert(tile, sink);
        let step = self.inner.state.lock().request_tile(tile);
        self.inner.perform(step);
    }
}
