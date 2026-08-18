// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The one archive lookup state machine both consumers drive.
//!
//! [`ArchiveState`] owns everything that is true of *reading an archive* and
//! nothing that is true of drawing one: the open, the leaf-directory walk, the
//! depth bound, the concurrency cap and the tile-body decode. The raster
//! provider and the vector transport each keep one inside their own mutex and
//! differ only in what they do with the answers.
//!
//! # Why it returns work instead of doing it
//!
//! Every method answers with an [`ArchiveStep`]: the range reads to issue and
//! the tiles that now have a final answer. Neither may be performed while the
//! lock is held — a [`crate::RangeTransport`] may answer *synchronously* (the
//! in-memory one always does), and delivering into the consumer's own store
//! takes the same lock. Computing the work under the lock and performing it
//! after is the same discipline [`crate::cog_provider::CogTileProvider::decide`]
//! follows, for the same deadlock.
//!
//! # The queue is the caller's cap, not this one's
//!
//! [`crate::TileTransport::request`] promises exactly one delivery per call, so
//! a tile asked for while the archive is still opening cannot simply be
//! dropped and re-asked: the vector provider has already marked it in flight
//! and will never ask again. Requested tiles are therefore *accepted* into a
//! queue and started as slots free up. The queue is bounded by the caller's own
//! in-flight cap ([`crate::tile_provider::MAX_INFLIGHT_TILES`] = 16 for the
//! vector provider, [`super::MAX_INFLIGHT_ARCHIVE_TILES`] for the raster one),
//! so it cannot grow without limit.

use std::sync::Arc;

use oxigis_render::{ByteRange, TileId};

use crate::archive::open::ArchiveContent;
use crate::archive::paged_mbtiles_state::PagedMbTilesState;
use crate::archive::pmtiles_state::PmtilesState;
use crate::cog_provider::RangeJob;
use crate::mbtiles::MbTilesReader;
use crate::tile_provider::TileError;

/// How many tile lookups may have a range read outstanding at once.
///
/// Between the XYZ path's 16 and the COG path's 4: an archive tile is one small
/// range read rather than a URL fetch or a multi-megabyte COG block, but a leaf
/// hop can double the reads a single tile costs.
pub const MAX_INFLIGHT_ARCHIVE_TILES: usize = 8;

/// A final answer for one tile address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ArchiveOutcome {
    /// The tile's bytes, already decoded per the header's `tile_compression`.
    Body(Vec<u8>),
    /// The archive holds no tile at that address.
    ///
    /// **Not a failure.** A sparse archive missing an ocean tile is normal;
    /// this is cached, never retried and never logged.
    Absent,
    /// The lookup or the fetch failed.
    Failed(TileError),
}

/// What an [`ArchiveState`] call wants done once the lock is released.
#[derive(Debug, Default)]
pub(crate) struct ArchiveStep {
    /// Range reads to issue through the transport.
    pub(crate) requests: Vec<(ByteRange, RangeJob)>,
    /// Tiles that now have a final answer.
    pub(crate) outcomes: Vec<(TileId, ArchiveOutcome)>,
    /// The archive just opened, and holds this kind of tile.
    pub(crate) opened: Option<ArchiveContent>,
    /// The archive just failed to open, for this named reason.
    pub(crate) failure: Option<String>,
}

impl ArchiveStep {
    /// Whether anything at all needs doing.
    pub(crate) fn is_empty(&self) -> bool {
        self.requests.is_empty()
            && self.outcomes.is_empty()
            && self.opened.is_none()
            && self.failure.is_none()
    }
}

/// One archive being read, whichever container it is.
///
/// Three shapes, one seam. PMTiles is asynchronous and directory-shaped — every
/// answer costs a range read; a **resident** MBTiles archive is synchronous,
/// because a SQLite image is already in memory; a **paged** MBTiles archive is
/// asynchronous and page-shaped. Rather than three provider families, the
/// *difference* is confined here, so the raster provider and the vector
/// transport above it are byte-for-byte the same code for all three.
pub(crate) enum ArchiveState {
    /// A PMTiles archive read through a range transport.
    Pmtiles(Box<PmtilesState>),
    /// An MBTiles archive already open in memory.
    MbTiles(Box<MbTilesState>),
    /// An MBTiles archive read a page at a time through a range transport.
    PagedMbTiles(Box<PagedMbTilesState>),
}

/// An MBTiles archive being served.
pub(crate) struct MbTilesState {
    /// The opened, indexed archive.
    reader: Arc<MbTilesReader>,
    /// Set when the archive's content is not what the caller draws.
    refusal: Option<String>,
    /// Whether the one-time `opened` report has been made.
    announced: bool,
    /// What the archive holds.
    content: ArchiveContent,
}

impl MbTilesState {
    /// Wraps an opened reader for a caller that draws `expected`.
    fn new(reader: Arc<MbTilesReader>, expected: Option<ArchiveContent>) -> Self {
        let content = reader.info().content;
        let refusal = expected.filter(|wanted| *wanted != content).map(|wanted| {
            format!(
                "this archive holds {} tiles, so it cannot be drawn as a {} layer",
                content.name(),
                wanted.name()
            )
        });
        Self {
            reader,
            refusal,
            announced: false,
            content,
        }
    }

    /// Answers one tile synchronously.
    fn request_tile(&mut self, tile: TileId) -> ArchiveStep {
        let mut step = ArchiveStep::default();
        if let Some(reason) = &self.refusal {
            if !self.announced {
                self.announced = true;
                step.failure = Some(reason.clone());
            }
            step.outcomes
                .push((tile, ArchiveOutcome::Failed(TileError::permanent(reason))));
            return step;
        }
        if !self.announced {
            self.announced = true;
            step.opened = Some(self.content);
        }
        // Synchronous, and deliberately so: the image is already in memory, so
        // a lookup is a binary search plus one b-tree descent — microseconds,
        // not a round trip. The cost is real but bounded, and it is the same on
        // both targets, which is what keeps a `.mbtiles` working in a browser
        // tab where no worker thread could read one anyway.
        let outcome = match self.reader.tile(tile) {
            Ok(Some(body)) => ArchiveOutcome::Body(body),
            Ok(None) => ArchiveOutcome::Absent,
            Err(error) => ArchiveOutcome::Failed(TileError::permanent(error.to_string())),
        };
        step.outcomes.push((tile, outcome));
        step
    }
}

impl ArchiveState {
    /// A fresh PMTiles state for an archive that has not been read at all.
    pub(crate) fn new(expected: Option<ArchiveContent>) -> Self {
        Self::Pmtiles(Box::new(PmtilesState::new(expected)))
    }

    /// A state serving an already-opened MBTiles archive.
    pub(crate) fn mbtiles(reader: Arc<MbTilesReader>, expected: Option<ArchiveContent>) -> Self {
        Self::MbTiles(Box::new(MbTilesState::new(reader, expected)))
    }

    /// A fresh state for an MBTiles archive read a page at a time.
    ///
    /// `declared_total` is the file length the transport pinned, when there is
    /// one; it only ever *bounds* a page count the archive's own header could
    /// not vouch for.
    pub(crate) fn paged_mbtiles(
        expected: Option<ArchiveContent>,
        declared_total: Option<u64>,
    ) -> Self {
        Self::PagedMbTiles(Box::new(PagedMbTilesState::new(expected, declared_total)))
    }

    /// Why the archive could not be read, if it could not be.
    pub(crate) fn failure(&self) -> Option<&str> {
        match self {
            Self::Pmtiles(state) => state.failure(),
            Self::MbTiles(state) => state.refusal.as_deref(),
            Self::PagedMbTiles(state) => state.failure(),
        }
    }

    /// Whether the opening read is still outstanding. Always `false` for a
    /// resident MBTiles archive, which is open before it is ever handed over.
    pub(crate) const fn is_opening(&self) -> bool {
        match self {
            Self::Pmtiles(state) => state.is_opening(),
            Self::MbTiles(_) => false,
            Self::PagedMbTiles(state) => state.is_opening(),
        }
    }

    /// Whether the archive is open and usable.
    pub(crate) fn is_open(&self) -> bool {
        match self {
            Self::Pmtiles(state) => state.archive().is_some(),
            Self::MbTiles(state) => state.refusal.is_none(),
            Self::PagedMbTiles(state) => state.is_open(),
        }
    }

    /// How many lookups are started but unanswered.
    pub(crate) fn active_len(&self) -> usize {
        match self {
            Self::Pmtiles(state) => state.active_len(),
            Self::MbTiles(_) => 0,
            Self::PagedMbTiles(state) => state.active_len(),
        }
    }

    /// How many tiles are accepted but not started.
    pub(crate) fn queued_len(&self) -> usize {
        match self {
            Self::Pmtiles(state) => state.queued_len(),
            Self::MbTiles(_) => 0,
            Self::PagedMbTiles(state) => state.queued_len(),
        }
    }

    /// How many leaf directories are held, and what their **stored** bytes
    /// cost.
    ///
    /// Stored rather than decoded because that is what the cache holds and what
    /// [`crate::archive::LEAF_CACHE_BYTES`] bounds — see
    /// [`crate::archive::leaf`] for the measurements. Always `(0, 0)` for an
    /// MBTiles archive, which has no leaf directories at all.
    pub(crate) fn leaf_stats(&self) -> (usize, usize) {
        match self {
            Self::Pmtiles(state) => state.leaf_stats(),
            Self::MbTiles(_) | Self::PagedMbTiles(_) => (0, 0),
        }
    }

    /// Drives the open one step; a no-op for an already-open archive.
    pub(crate) fn kick(&mut self) -> ArchiveStep {
        match self {
            Self::Pmtiles(state) => state.kick(),
            Self::MbTiles(_) => ArchiveStep::default(),
            Self::PagedMbTiles(state) => state.kick(),
        }
    }

    /// Accepts `tile` for resolution.
    pub(crate) fn request_tile(&mut self, tile: TileId) -> ArchiveStep {
        match self {
            Self::Pmtiles(state) => state.request_tile(tile),
            Self::MbTiles(state) => state.request_tile(tile),
            Self::PagedMbTiles(state) => state.request_tile(tile),
        }
    }

    /// Feeds one delivered range back in.
    pub(crate) fn deliver(
        &mut self,
        job: RangeJob,
        result: Result<Vec<u8>, TileError>,
    ) -> ArchiveStep {
        match self {
            Self::Pmtiles(state) => state.deliver(job, result),
            Self::MbTiles(_) => {
                tracing::debug!("oxigis-ui: a range job reached an in-memory archive; dropped");
                ArchiveStep::default()
            }
            Self::PagedMbTiles(state) => state.deliver(job, result),
        }
    }
}
