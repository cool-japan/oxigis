// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! One MBTiles archive being read **a page at a time**, over a range transport.
//!
//! The third backing of [`crate::archive::state::ArchiveState`], and the reason
//! there is a third: PMTiles is asynchronous and directory-shaped, the resident
//! MBTiles reader is synchronous and already in memory, and this one is
//! asynchronous and *page*-shaped. Everything above — the raster provider, the
//! vector transport, the print path — is byte-for-byte the same code for all
//! three.
//!
//! ```text
//! kick()          -> ArchiveSurvey { start: 0 }        one 16 KiB read
//!                 -> ArchivePage { first, count }      the survey's own pages
//! request_tile(t) -> ArchivePage { first, count } ...   a handful per tile
//!                 -> Body / Absent
//! ```
//!
//! # `page_waiters` is `leaf_waiters` one format over
//!
//! Sixteen tiles of a viewport descend the *same* index root and the same
//! interior level. Without a waiter map that is sixteen identical range requests
//! for one page; with it, the first tile issues the read and the other fifteen
//! ride it — the measured 2.33 → 0.33 requests-per-tile warm improvement.
//!
//! # The take-then-poll dance
//!
//! A descent needs `&mut PageSource`, and the source lives inside the archive
//! that also owns the map of live descents. Holding both at once is not a
//! borrow this language permits, so a descent is **taken out** of the map,
//! stepped, and put back — exactly the shape
//! [`crate::archive::pmtiles_state::PmtilesState`] uses for its leaf hops.

use std::collections::{HashMap, VecDeque};

use oxigis_render::TileId;

use crate::archive::open::ArchiveContent;
use crate::archive::state::{ArchiveOutcome, ArchiveStep, MAX_INFLIGHT_ARCHIVE_TILES};
use crate::cog_provider::RangeJob;
use crate::mbtiles::paged::source::PageRun;
use crate::mbtiles::paged::{
    DescentStep, PagedArchive, PagedNeed, PagedOpen, PagedOpenStep, TileDescent,
};
use crate::tile_provider::TileError;

/// How far along the archive's own survey is.
enum Stage {
    /// The 16 KiB prefetch (and possibly a page or two more) is still being
    /// read.
    Opening(Box<PagedOpen>),
    /// The archive is surveyed and usable.
    Ready(Box<PagedArchive>),
    /// The archive will never open; the reason is kept for the status line.
    Failed(String),
}

/// What one pumped descent decided, computed while the archive is borrowed and
/// acted on after that borrow ends.
enum Pumped {
    /// The lookup wants this run; the range is what to ask the transport for.
    Need(PageRun, Result<oxigis_render::ByteRange, String>),
    /// The tile's bytes.
    Body(Vec<u8>),
    /// The archive holds no tile there.
    Absent,
    /// The lookup was refused, by name.
    Failed(String),
}

/// One MBTiles archive being read over a range transport.
pub(crate) struct PagedMbTilesState {
    /// How far along the survey is.
    stage: Stage,
    /// What the layer asking for this archive was built to draw, so a
    /// raster-vs-vector mismatch is refused at open by name.
    expected: Option<ArchiveContent>,
    /// Whether a survey read is outstanding.
    open_inflight: bool,
    /// Tiles accepted but not started, in arrival order.
    queued: VecDeque<TileId>,
    /// Started lookups, keyed by address.
    active: HashMap<TileId, TileDescent>,
    /// Tiles waiting on a page run already in flight, keyed by its first page.
    page_waiters: HashMap<u32, Vec<TileId>>,
}

impl PagedMbTilesState {
    /// A fresh state for an archive that has not been read at all.
    ///
    /// `declared_total` is the file length the transport pinned, when there is
    /// one; it only ever **bounds** a page count the header could not vouch for.
    pub(crate) fn new(expected: Option<ArchiveContent>, declared_total: Option<u64>) -> Self {
        Self {
            stage: Stage::Opening(Box::new(PagedOpen::new(declared_total))),
            expected,
            open_inflight: false,
            queued: VecDeque::new(),
            active: HashMap::new(),
            page_waiters: HashMap::new(),
        }
    }

    /// Why the archive could not be read, if it could not be.
    pub(crate) fn failure(&self) -> Option<&str> {
        match &self.stage {
            Stage::Failed(message) => Some(message),
            Stage::Opening(_) | Stage::Ready(_) => None,
        }
    }

    /// Whether the survey is still outstanding.
    pub(crate) const fn is_opening(&self) -> bool {
        matches!(self.stage, Stage::Opening(_))
    }

    /// Whether the archive is surveyed and usable.
    pub(crate) const fn is_open(&self) -> bool {
        matches!(self.stage, Stage::Ready(_))
    }

    /// How many lookups are started but unanswered.
    pub(crate) fn active_len(&self) -> usize {
        self.active.len()
    }

    /// How many tiles are accepted but not started.
    pub(crate) fn queued_len(&self) -> usize {
        self.queued.len()
    }

    /// Drives the survey one step; a no-op for an already-open archive.
    pub(crate) fn kick(&mut self) -> ArchiveStep {
        let mut step = ArchiveStep::default();
        self.advance(&mut step);
        step
    }

    /// Accepts `tile` for resolution, starting it if a slot is free.
    pub(crate) fn request_tile(&mut self, tile: TileId) -> ArchiveStep {
        let mut step = ArchiveStep::default();
        if let Stage::Failed(message) = &self.stage {
            step.outcomes
                .push((tile, ArchiveOutcome::Failed(TileError::permanent(message))));
            return step;
        }
        if self.active.contains_key(&tile) || self.queued.contains(&tile) {
            return step;
        }
        self.queued.push_back(tile);
        self.advance(&mut step);
        step
    }

    /// Feeds one delivered range back in.
    pub(crate) fn deliver(
        &mut self,
        job: RangeJob,
        result: Result<Vec<u8>, TileError>,
    ) -> ArchiveStep {
        let mut step = ArchiveStep::default();
        match job {
            RangeJob::ArchiveSurvey { .. } => self.deliver_survey(result, &mut step),
            RangeJob::ArchivePage { first, .. } => self.deliver_pages(first, result, &mut step),
            other => {
                tracing::debug!("oxigis-ui: {other:?} reached the paged MBTiles sink; dropped");
            }
        }
        self.advance(&mut step);
        step
    }

    /// Drives the survey when it is still opening, then starts as many queued
    /// lookups as there are free slots.
    fn advance(&mut self, step: &mut ArchiveStep) {
        self.drive_open(step);
        if let Stage::Failed(message) = &self.stage {
            let reason = TileError::permanent(message.clone());
            for tile in self
                .queued
                .drain(..)
                .chain(self.active.drain().map(|entry| entry.0))
            {
                step.outcomes
                    .push((tile, ArchiveOutcome::Failed(reason.clone())));
            }
            self.page_waiters.clear();
            return;
        }
        if !self.is_open() {
            return;
        }
        while self.active.len() < MAX_INFLIGHT_ARCHIVE_TILES {
            let Some(tile) = self.queued.pop_front() else {
                return;
            };
            let started = {
                let Stage::Ready(archive) = &self.stage else {
                    return;
                };
                if archive.covers(tile) {
                    Some(archive.begin(tile))
                } else {
                    // The zoom gate, answered with ZERO page reads.
                    None
                }
            };
            match started {
                Some(descent) => {
                    self.active.insert(tile, descent);
                    self.pump(tile, step);
                }
                None => step.outcomes.push((tile, ArchiveOutcome::Absent)),
            }
        }
    }

    /// Polls the survey, issuing the read it asks for.
    fn drive_open(&mut self, step: &mut ArchiveStep) {
        if self.open_inflight {
            return;
        }
        let Stage::Opening(open) = &mut self.stage else {
            return;
        };
        match open.step() {
            Ok(PagedOpenStep::Need(PagedNeed::Prefetch(range))) => {
                self.open_inflight = true;
                step.requests
                    .push((range, RangeJob::ArchiveSurvey { start: range.start }));
            }
            Ok(PagedOpenStep::Need(PagedNeed::Pages(run))) => match open.range_for(run) {
                Ok(range) => {
                    self.open_inflight = true;
                    step.requests.push((
                        range,
                        RangeJob::ArchivePage {
                            first: run.first,
                            count: run.count,
                        },
                    ));
                }
                Err(error) => self.fail(error.to_string(), step),
            },
            Ok(PagedOpenStep::Ready(archive)) => {
                let content = archive.info().content;
                if let Some(expected) = self.expected
                    && expected != content
                {
                    let message = format!(
                        "this archive holds {} tiles, so it cannot be drawn as a {} layer",
                        content.name(),
                        expected.name()
                    );
                    self.fail(message, step);
                    return;
                }
                self.stage = Stage::Ready(archive);
                step.opened = Some(content);
            }
            Err(error) => self.fail(error.to_string(), step),
        }
    }

    /// Records a named refusal, once.
    fn fail(&mut self, message: String, step: &mut ArchiveStep) {
        self.stage = Stage::Failed(message.clone());
        step.failure = Some(message);
    }

    /// Drives one live descent as far as the pages in hand allow.
    ///
    /// The take-then-poll dance: the descent is removed from `active` so the
    /// archive (which owns the page cache) can be borrowed mutably, and every
    /// decision that needs the archive is made *before* the map is touched
    /// again.
    fn pump(&mut self, tile: TileId, step: &mut ArchiveStep) {
        let Some(mut descent) = self.active.remove(&tile) else {
            return;
        };
        let pumped = {
            let Stage::Ready(archive) = &mut self.stage else {
                return;
            };
            match archive.step(&mut descent) {
                Ok(DescentStep::Need(run)) => Pumped::Need(
                    run,
                    archive.range_for(run).map_err(|error| error.to_string()),
                ),
                Ok(DescentStep::Body(bytes)) => Pumped::Body(bytes),
                Ok(DescentStep::Absent) => Pumped::Absent,
                Err(error) => Pumped::Failed(error.to_string()),
            }
        };
        match pumped {
            Pumped::Body(bytes) => step.outcomes.push((tile, ArchiveOutcome::Body(bytes))),
            Pumped::Absent => step.outcomes.push((tile, ArchiveOutcome::Absent)),
            Pumped::Failed(message) => step
                .outcomes
                .push((tile, ArchiveOutcome::Failed(TileError::permanent(message)))),
            Pumped::Need(_run, Err(message)) => step
                .outcomes
                .push((tile, ArchiveOutcome::Failed(TileError::permanent(message)))),
            Pumped::Need(run, Ok(range)) => {
                self.active.insert(tile, descent);
                match self.page_waiters.entry(run.first) {
                    std::collections::hash_map::Entry::Occupied(mut waiting) => {
                        // Another tile already asked for this run; ride its read
                        // rather than issuing a second one.
                        waiting.get_mut().push(tile);
                    }
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(vec![tile]);
                        step.requests.push((
                            range,
                            RangeJob::ArchivePage {
                                first: run.first,
                                count: run.count,
                            },
                        ));
                    }
                }
            }
        }
    }

    /// Feeds the opening prefetch into the survey.
    fn deliver_survey(&mut self, result: Result<Vec<u8>, TileError>, step: &mut ArchiveStep) {
        self.open_inflight = false;
        match result {
            Ok(bytes) => {
                let outcome = match &mut self.stage {
                    Stage::Opening(open) => open.supply_prefetch(&bytes).err(),
                    Stage::Ready(_) | Stage::Failed(_) => None,
                };
                if let Some(error) = outcome {
                    self.fail(error.to_string(), step);
                }
            }
            Err(error) => {
                self.fail(format!("the archive could not be read: {error}"), step);
            }
        }
    }

    /// Feeds one delivered page run in and resumes everything waiting on it.
    fn deliver_pages(
        &mut self,
        first: u32,
        result: Result<Vec<u8>, TileError>,
        step: &mut ArchiveStep,
    ) {
        let waiting = self.page_waiters.remove(&first).unwrap_or_default();
        match result {
            Ok(bytes) => {
                match &mut self.stage {
                    Stage::Opening(open) => {
                        self.open_inflight = false;
                        open.supply_pages(first, &bytes);
                    }
                    Stage::Ready(archive) => archive.supply(first, &bytes),
                    Stage::Failed(_) => return,
                }
                for tile in waiting {
                    self.pump(tile, step);
                }
            }
            Err(error) => {
                self.open_inflight = false;
                for tile in waiting {
                    self.active.remove(&tile);
                    step.outcomes
                        .push((tile, ArchiveOutcome::Failed(error.clone())));
                }
            }
        }
    }
}
