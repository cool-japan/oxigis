// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Reading an MBTiles archive **a page at a time**, over a byte-range
//! transport.
//!
//! The resident [`crate::mbtiles::MbTilesReader`] reads a SQLite image that is
//! already in memory. That is right for bytes in hand — a browser drop — and
//! wrong for everything else: it caps at
//! [`crate::mbtiles::MAX_MBTILES_BYTES`], it builds a whole-pyramid index up
//! front, and over HTTP it would mean downloading the archive to draw one tile.
//! This module is the other half: open in one 16 KiB read, then pay a handful of
//! page reads per tile, forever, whatever the archive's size.
//!
//! ```text
//! SqliteSurvey  16 KiB  ->  header + sqlite_master + metadata  ->  PagedLayout
//! TileDescent   per tile ->  index -> row [-> images index -> images row]
//!                            -> overflow chain when the body spilled
//! PageSource             ->  every page, byte-budgeted, keyed by page number
//! ```
//!
//! # Measured
//!
//! | | cold | warm |
//! |---|---|---|
//! | flat archive | 2.33 requests/tile | **0.33** |
//! | normalized archive | 2.96 | **0.42** |
//!
//! Within ~2.3× of PMTiles, and moving *less* data per tile than a single planet
//! leaf-directory miss does. That measurement is what retires the v1.3 claim
//! that a `.mbtiles` "cannot be read over HTTP Range requests".
//!
//! # Two readers for one format is a divergence risk, and this is the fence
//!
//! [`crate::mbtiles::schema`]'s `Layout`, `Columns`, `Metadata` and `xyz_row`
//! are **reused verbatim**, and
//! [`crate::gpkg_input::sqlite`]'s payload-split arithmetic
//! (`inline_len_for`/`index_inline_len_for`/`cell_offsets_in`) is shared rather
//! than re-derived. The decisive test —
//! `the_paged_reader_and_the_resident_reader_agree_on_every_tile` — reads every
//! address of every fixture shape both ways and compares the bytes. Weakening it
//! ships the divergence.

pub(crate) mod descend;
pub(crate) mod source;
pub(crate) mod survey;

#[cfg(test)]
mod tests;

use oxigis_render::{ByteRange, TileId};

use crate::archive::ArchiveInfo;
use crate::local_vector::LocalVectorError;

pub(crate) use crate::mbtiles::paged::descend::{DescentStep, TileDescent};
pub(crate) use crate::mbtiles::paged::source::{PageRun, PageSource};
pub(crate) use crate::mbtiles::paged::survey::{PagedLayout, SqliteSurvey, SurveyStep};

/// What an open still needs.
#[derive(Debug)]
pub(crate) enum PagedNeed {
    /// The speculative opening read, in bytes — the page size is not known yet.
    Prefetch(ByteRange),
    /// A run of pages.
    Pages(PageRun),
}

/// One opened paged archive: its layout and its pages.
///
/// Deliberately holds **no** index of its own. The archive's own b-trees are the
/// index, and building a second one would be the whole-pyramid scan this reader
/// exists to avoid.
#[derive(Debug)]
pub(crate) struct PagedArchive {
    /// What the survey learned.
    layout: PagedLayout,
    /// The pages, byte-budgeted.
    source: PageSource,
}

impl PagedArchive {
    /// Wraps a surveyed layout around the pages its survey already paid for.
    pub(crate) const fn new(layout: PagedLayout, source: PageSource) -> Self {
        Self { layout, source }
    }

    /// Archive-level facts, in the shape the archive layer everywhere else
    /// speaks.
    pub(crate) fn info(&self) -> ArchiveInfo {
        self.layout.metadata.info()
    }

    /// Whether the archive could hold `tile` at all — the zoom gate, answered
    /// with **zero** page reads.
    pub(crate) fn covers(&self, tile: TileId) -> bool {
        tile.z >= self.layout.metadata.min_zoom && tile.z <= self.layout.metadata.max_zoom
    }

    /// How many pages a run occupies, as a byte range.
    ///
    /// # Errors
    ///
    /// Refuses a degenerate run.
    pub(crate) fn range_for(&self, run: PageRun) -> Result<ByteRange, LocalVectorError> {
        self.source.range_for(run)
    }

    /// Feeds one delivered page run in.
    pub(crate) fn supply(&mut self, first: u32, bytes: &[u8]) {
        self.source.supply(first, bytes);
    }

    /// Starts a lookup for `tile`.
    pub(crate) fn begin(&self, tile: TileId) -> TileDescent {
        TileDescent::new(tile, &self.layout)
    }

    /// Drives `descent` as far as the pages in hand allow.
    ///
    /// # Errors
    ///
    /// Every refusal [`TileDescent::step`] makes.
    pub(crate) fn step(
        &mut self,
        descent: &mut TileDescent,
    ) -> Result<DescentStep, LocalVectorError> {
        descent.step(&self.layout, &mut self.source)
    }
}

/// An archive being surveyed, before any tile can be asked for.
#[derive(Debug)]
pub(crate) struct PagedOpen {
    /// The survey in progress.
    survey: SqliteSurvey,
}

/// What a [`PagedOpen`] wants next.
#[derive(Debug)]
pub(crate) enum PagedOpenStep {
    /// These bytes (or pages) are needed.
    Need(PagedNeed),
    /// The archive is open.
    Ready(Box<PagedArchive>),
}

impl PagedOpen {
    /// A survey of an archive whose length the caller may already know.
    ///
    /// `declared_total` is the `Content-Range` total the transport pinned, when
    /// there is one; it only ever **bounds** a page count the header could not
    /// vouch for.
    pub(crate) const fn new(declared_total: Option<u64>) -> Self {
        Self {
            survey: SqliteSurvey::new(declared_total),
        }
    }

    /// Feeds the opening prefetch in.
    ///
    /// # Errors
    ///
    /// Refuses everything the 100-byte header can declare that this build will
    /// not read, each by name.
    pub(crate) fn supply_prefetch(&mut self, bytes: &[u8]) -> Result<(), LocalVectorError> {
        self.survey.supply_prefetch(bytes)
    }

    /// Feeds one delivered page run in.
    pub(crate) fn supply_pages(&mut self, first: u32, bytes: &[u8]) {
        self.survey.supply_pages(first, bytes);
    }

    /// The byte range a page run occupies, once the page size is known.
    ///
    /// # Errors
    ///
    /// Refuses a run asked for before the header landed.
    pub(crate) fn range_for(&self, run: PageRun) -> Result<ByteRange, LocalVectorError> {
        self.survey
            .source()
            .ok_or_else(|| source::err("a page was asked for before the header arrived"))?
            .range_for(run)
    }

    /// Drives the survey one step.
    ///
    /// # Errors
    ///
    /// Every named refusal in [`survey`]'s module docs.
    pub(crate) fn step(&mut self) -> Result<PagedOpenStep, LocalVectorError> {
        match self.survey.step()? {
            SurveyStep::NeedPrefetch(range) => Ok(PagedOpenStep::Need(PagedNeed::Prefetch(range))),
            SurveyStep::NeedPages(run) => Ok(PagedOpenStep::Need(PagedNeed::Pages(run))),
            SurveyStep::Ready(layout) => {
                let source = self
                    .survey
                    .take_source()
                    .ok_or_else(|| source::err("the survey finished with no pages"))?;
                Ok(PagedOpenStep::Ready(Box::new(PagedArchive::new(
                    *layout, source,
                ))))
            }
        }
    }
}
