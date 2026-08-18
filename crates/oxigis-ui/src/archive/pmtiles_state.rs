// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! One PMTiles archive being read: its open, its leaf cache and its live
//! lookups.
//!
//! Split out of [`crate::archive::state`] verbatim in tiles v1.4 so a third
//! backing (the paged MBTiles reader) could join [`crate::archive::state`]'s
//! enum without any of the three files growing past what a reviewer holds in
//! their head. Nothing about the state machine changed in the move.
//!
//! # Why a lookup is a state machine and not a function
//!
//! Every answer costs a range read, and there is no thread to block on one — the
//! browser has none and the render thread must not be one. So a lookup is
//! *accepted*, then resumed as bytes arrive: [`PmtilesState::request_tile`] and
//! [`PmtilesState::deliver`] both answer with an
//! [`ArchiveStep`] of work to perform once the caller's lock is released.
//!
//! # The directory walk, and where the reads go
//!
//! ```text
//! find(tile)  ->  Absent                      (the zoom/bbox gate: zero reads)
//!             ->  Tile(range)                 (one read: the body)
//!             ->  Leaf { at, range }          -> decoded in front?   no read
//!                                             -> stored in cache?    no read
//!                                             -> otherwise           one read
//! ```
//!
//! The two cache tiers are [`crate::archive::leaf::LeafCache`]'s; the
//! measurements that chose them are in that module's docs. What lives here is
//! the *bounding*: a leaf that points at another leaf past
//! [`MAX_DIRECTORY_DEPTH`] is refused rather than followed, and tiles waiting on
//! a leaf already in flight ride that one read instead of issuing their own
//! (`leaf_waiters`).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use oxigis_render::TileId;
use oxigis_render::pmtiles::{
    DirEntry, MAX_DIRECTORY_DEPTH, PmtilesArchive, PmtilesError, PmtilesOpen, TileLookup,
    deserialize_directory,
};

use crate::archive::leaf::LeafCache;
use crate::archive::open::{ArchiveContent, OpenStep, advance_open, check_archive, plain_tile};
use crate::archive::state::{ArchiveOutcome, ArchiveStep, MAX_INFLIGHT_ARCHIVE_TILES};
use crate::cog_provider::RangeJob;
use crate::tile_provider::TileError;

/// How far along the archive's own header is.
enum Stage {
    /// The speculative prefetch (and possibly a far metadata block) is still
    /// being read.
    Opening(Box<PmtilesOpen>),
    /// The archive is open and usable.
    Ready(Arc<PmtilesArchive>),
    /// The archive will never open; the reason is kept for the status line.
    Failed(String),
}

/// One PMTiles archive being read: its open, its leaf cache and its live
/// lookups.
pub(crate) struct PmtilesState {
    /// How far along the open is.
    stage: Stage,
    /// What the layer asking for this archive was built to draw, so a
    /// raster-vs-vector mismatch is refused at open by name.
    expected: Option<ArchiveContent>,
    /// Whether a header or metadata range read is outstanding.
    header_inflight: bool,
    /// Leaf directories, byte-budgeted over their stored size.
    leaves: LeafCache,
    /// Tiles accepted but not started, in arrival order.
    queued: VecDeque<TileId>,
    /// Started lookups: tile → how many directories have been consulted.
    active: HashMap<TileId, u32>,
    /// Tiles waiting on a leaf directory already in flight, keyed by its
    /// absolute file offset.
    leaf_waiters: HashMap<u64, Vec<TileId>>,
}

impl PmtilesState {
    /// A fresh state for an archive that has not been read at all.
    ///
    /// `expected` is the content kind the caller draws; [`None`] accepts
    /// either, which is what the probe wants.
    pub(crate) fn new(expected: Option<ArchiveContent>) -> Self {
        Self {
            stage: Stage::Opening(Box::new(PmtilesOpen::new())),
            expected,
            header_inflight: false,
            leaves: LeafCache::new(),
            queued: VecDeque::new(),
            active: HashMap::new(),
            leaf_waiters: HashMap::new(),
        }
    }

    /// The opened archive, once the header has landed.
    pub(crate) fn archive(&self) -> Option<Arc<PmtilesArchive>> {
        match &self.stage {
            Stage::Ready(archive) => Some(Arc::clone(archive)),
            Stage::Opening(_) | Stage::Failed(_) => None,
        }
    }

    /// Why the archive could not be read, if it could not be.
    pub(crate) fn failure(&self) -> Option<&str> {
        match &self.stage {
            Stage::Failed(message) => Some(message),
            Stage::Opening(_) | Stage::Ready(_) => None,
        }
    }

    /// Whether the header read is still outstanding.
    pub(crate) const fn is_opening(&self) -> bool {
        matches!(self.stage, Stage::Opening(_))
    }

    /// How many lookups are started but unanswered.
    pub(crate) fn active_len(&self) -> usize {
        self.active.len()
    }

    /// How many tiles are accepted but not started.
    pub(crate) fn queued_len(&self) -> usize {
        self.queued.len()
    }

    /// How many leaf directories are held, and what their **stored** bytes cost.
    pub(crate) fn leaf_stats(&self) -> (usize, usize) {
        (self.leaves.len(), self.leaves.bytes())
    }

    /// Drives the open one step, issuing the next header read when one is
    /// needed and nothing is already outstanding.
    ///
    /// Called from the frame loop rather than from the constructor, so an
    /// archive layer that is built and never drawn issues no request at all —
    /// the [`crate::cog_provider::CogTileProvider`] rule.
    pub(crate) fn kick(&mut self) -> ArchiveStep {
        let mut step = ArchiveStep::default();
        self.advance(&mut step);
        step
    }

    /// Accepts `tile` for resolution, starting it if a slot is free.
    ///
    /// Answering is *always* eventual: the tile is queued when the archive is
    /// still opening or every slot is busy, and started later. A tile already
    /// queued or in flight is ignored, so repeated frames cost nothing.
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
            RangeJob::ArchiveHeader { start } => self.deliver_header(start, result, &mut step),
            RangeJob::ArchiveLeaf { at, .. } => self.deliver_leaf(at, result, &mut step),
            RangeJob::ArchiveTile { tile } => self.deliver_tile(tile, result, &mut step),
            // Every other variant belongs to another reader. A job always
            // travels WITH its own sink, so this is unreachable short of a
            // broken transport; it is reported rather than silently swallowed.
            other => {
                tracing::debug!("oxigis-ui: {other:?} reached the PMTiles sink; dropped");
            }
        }
        self.advance(&mut step);
        step
    }

    /// Drives the open when it is still opening, then starts as many queued
    /// lookups as there are free slots.
    fn advance(&mut self, step: &mut ArchiveStep) {
        self.drive_open(step);
        let Some(archive) = self.archive() else {
            // Still opening, or failed. A failed archive answers everything it
            // was holding, so nothing is left waiting on a header that will
            // never land.
            if let Stage::Failed(message) = &self.stage {
                let reason = TileError::permanent(message.clone());
                for tile in self
                    .queued
                    .drain(..)
                    .chain(self.active.drain().map(|kv| kv.0))
                {
                    step.outcomes
                        .push((tile, ArchiveOutcome::Failed(reason.clone())));
                }
                self.leaf_waiters.clear();
            }
            return;
        };
        while self.active.len() < MAX_INFLIGHT_ARCHIVE_TILES {
            let Some(tile) = self.queued.pop_front() else {
                return;
            };
            match archive.find(tile) {
                Ok(lookup) => self.follow(&archive, tile, 1, lookup, step),
                Err(error) => step.outcomes.push((
                    tile,
                    ArchiveOutcome::Failed(crate::archive::open::classify(&error)),
                )),
            }
        }
    }

    /// Polls the open state machine, issuing the read it asks for.
    fn drive_open(&mut self, step: &mut ArchiveStep) {
        if self.header_inflight {
            return;
        }
        let Stage::Opening(open) = &mut self.stage else {
            return;
        };
        match advance_open(open) {
            OpenStep::Need(range) => {
                self.header_inflight = true;
                step.requests
                    .push((range, RangeJob::ArchiveHeader { start: range.start }));
            }
            OpenStep::Ready(archive) => match check_archive(&archive, self.expected) {
                Ok(content) => {
                    self.stage = Stage::Ready(Arc::from(archive));
                    step.opened = Some(content);
                }
                Err(message) => {
                    self.stage = Stage::Failed(message.clone());
                    step.failure = Some(message);
                }
            },
            OpenStep::Failed(message) => {
                self.stage = Stage::Failed(message.clone());
                step.failure = Some(message);
            }
        }
    }

    /// Acts on one directory answer for `tile`, hopping through leaf
    /// directories that are already cached and bounding the chain.
    ///
    /// `depth` is how many directories have been consulted so far: the root is
    /// 1, its leaf is 2. A leaf that points at another leaf past
    /// [`MAX_DIRECTORY_DEPTH`] is refused rather than followed, so a crafted
    /// archive whose leaf points at itself cannot loop.
    fn follow(
        &mut self,
        archive: &Arc<PmtilesArchive>,
        tile: TileId,
        depth: u32,
        lookup: TileLookup,
        step: &mut ArchiveStep,
    ) {
        let mut depth = depth;
        let mut lookup = lookup;
        loop {
            match lookup {
                TileLookup::Absent => {
                    self.active.remove(&tile);
                    step.outcomes.push((tile, ArchiveOutcome::Absent));
                    return;
                }
                TileLookup::Tile(range) => {
                    self.active.insert(tile, depth);
                    step.requests.push((range, RangeJob::ArchiveTile { tile }));
                    return;
                }
                TileLookup::Leaf { range, .. } => {
                    let next = depth.saturating_add(1);
                    if next > MAX_DIRECTORY_DEPTH {
                        self.active.remove(&tile);
                        step.outcomes.push((
                            tile,
                            ArchiveOutcome::Failed(crate::archive::open::classify(
                                &PmtilesError::LeafChainTooDeep {
                                    depth: next,
                                    limit: MAX_DIRECTORY_DEPTH,
                                },
                            )),
                        ));
                        return;
                    }
                    // The absolute offset is the cache key AND what the job
                    // carries, so one number identifies the blob everywhere.
                    let at = range.start;
                    match self.cached_leaf(archive, at) {
                        Some(Ok(entries)) => {
                            depth = next;
                            lookup = match archive.find_in_leaf(&entries, tile) {
                                Ok(lookup) => lookup,
                                Err(error) => {
                                    self.active.remove(&tile);
                                    step.outcomes.push((
                                        tile,
                                        ArchiveOutcome::Failed(crate::archive::open::classify(
                                            &error,
                                        )),
                                    ));
                                    return;
                                }
                            };
                            continue;
                        }
                        Some(Err(error)) => {
                            self.active.remove(&tile);
                            step.outcomes.push((tile, ArchiveOutcome::Failed(error)));
                            return;
                        }
                        None => {}
                    }
                    self.active.insert(tile, next);
                    match self.leaf_waiters.entry(at) {
                        std::collections::hash_map::Entry::Occupied(mut waiting) => {
                            // Another tile already asked for this leaf; ride
                            // its read rather than issuing a second one.
                            waiting.get_mut().push(tile);
                        }
                        std::collections::hash_map::Entry::Vacant(slot) => {
                            slot.insert(vec![tile]);
                            step.requests
                                .push((range, RangeJob::ArchiveLeaf { tile, at }));
                        }
                    }
                    return;
                }
            }
        }
    }

    /// The decoded leaf at `at` **without a range read**, or [`None`] when one
    /// is needed.
    ///
    /// Two tiers, in cost order: the one decoded directory kept in front (free),
    /// then the stored blob (a 1.8–2.5 ms re-decode, against the 257–1027 ms
    /// refetch it replaces). A decode here also becomes the new front entry, so
    /// the rest of the viewport's tiles on that leaf pay nothing.
    fn cached_leaf(
        &mut self,
        archive: &PmtilesArchive,
        at: u64,
    ) -> Option<Result<Arc<Vec<DirEntry>>, TileError>> {
        if let Some(entries) = self.leaves.decoded(at) {
            return Some(Ok(entries));
        }
        let stored = self.leaves.stored(at)?;
        Some(match decode_leaf(archive, &stored) {
            Ok(entries) => {
                let entries = Arc::new(entries);
                self.leaves.set_decoded(at, Arc::clone(&entries));
                Ok(entries)
            }
            Err(reason) => Err(TileError::permanent(format!(
                "a leaf directory at byte {at} could not be decoded: {reason}"
            ))),
        })
    }

    /// Feeds header (or far-metadata) bytes into the open state machine.
    fn deliver_header(
        &mut self,
        start: u64,
        result: Result<Vec<u8>, TileError>,
        step: &mut ArchiveStep,
    ) {
        self.header_inflight = false;
        match result {
            Ok(bytes) => {
                if let Stage::Opening(open) = &mut self.stage
                    && let Err(error) = open.supply(start, bytes)
                {
                    let message = error.to_string();
                    self.stage = Stage::Failed(message.clone());
                    step.failure = Some(message);
                }
            }
            Err(error) => {
                let message = format!("the archive's header could not be read: {error}");
                self.stage = Stage::Failed(message.clone());
                step.failure = Some(message);
            }
        }
    }

    /// Caches a delivered leaf directory and resumes everything waiting on it.
    ///
    /// The blob is kept **as delivered** — still coded per the header's
    /// `internal_compression` — and decoded once into the front cache; see
    /// [`crate::archive::leaf`] for the measurements behind that split.
    fn deliver_leaf(
        &mut self,
        at: u64,
        result: Result<Vec<u8>, TileError>,
        step: &mut ArchiveStep,
    ) {
        let waiting = self.leaf_waiters.remove(&at).unwrap_or_default();
        let Some(archive) = self.archive() else {
            // The archive failed while the leaf was in flight; `advance` will
            // answer every tile it was holding.
            return;
        };
        let decoded = match result {
            Ok(raw) => {
                let stored: Arc<[u8]> = Arc::from(raw.into_boxed_slice());
                match decode_leaf(&archive, &stored) {
                    Ok(entries) => Ok((stored, entries)),
                    Err(reason) => Err(TileError::permanent(format!(
                        "a leaf directory at byte {at} could not be decoded: {reason}"
                    ))),
                }
            }
            Err(error) => Err(error),
        };
        let entries = match decoded {
            Ok((stored, entries)) => {
                let entries = Arc::new(entries);
                self.leaves.insert(at, stored);
                self.leaves.set_decoded(at, Arc::clone(&entries));
                entries
            }
            Err(error) => {
                for tile in waiting {
                    self.active.remove(&tile);
                    step.outcomes
                        .push((tile, ArchiveOutcome::Failed(error.clone())));
                }
                return;
            }
        };
        for tile in waiting {
            let depth = self
                .active
                .get(&tile)
                .copied()
                .unwrap_or(MAX_DIRECTORY_DEPTH);
            match archive.find_in_leaf(&entries, tile) {
                Ok(lookup) => self.follow(&archive, tile, depth, lookup, step),
                Err(error) => {
                    self.active.remove(&tile);
                    step.outcomes.push((
                        tile,
                        ArchiveOutcome::Failed(crate::archive::open::classify(&error)),
                    ));
                }
            }
        }
    }

    /// Decodes a delivered tile body per the header's `tile_compression`.
    fn deliver_tile(
        &mut self,
        tile: TileId,
        result: Result<Vec<u8>, TileError>,
        step: &mut ArchiveStep,
    ) {
        self.active.remove(&tile);
        let Some(archive) = self.archive() else {
            return;
        };
        let outcome = match result {
            Ok(raw) => match plain_tile(archive.header().tile_compression, raw) {
                Ok(plain) => ArchiveOutcome::Body(plain),
                Err(error) => ArchiveOutcome::Failed(error),
            },
            Err(error) => ArchiveOutcome::Failed(error),
        };
        step.outcomes.push((tile, outcome));
    }
}

/// One leaf directory's stored bytes turned into entries.
///
/// Honours `internal_compression`, **not** `tile_compression`: they are
/// independent header fields and a measured archive sets them differently.
fn decode_leaf(archive: &PmtilesArchive, stored: &[u8]) -> Result<Vec<DirEntry>, String> {
    let plain = crate::archive::open::plain_directory(
        archive.header().internal_compression,
        stored.to_vec(),
    )?;
    deserialize_directory(&plain).map_err(|error| error.to_string())
}
