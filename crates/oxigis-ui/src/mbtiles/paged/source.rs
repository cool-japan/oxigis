// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! [`PageSource`]: SQLite pages held a few thousand at a time, and the
//! overflow-chain reader that reassembles a record out of them.
//!
//! # Why pages and not bytes
//!
//! A SQLite b-tree addresses everything by *page number*, and the same page is
//! visited again and again — the root of the index on every single lookup, the
//! interior level on most of them. Caching by page number rather than by byte
//! range is what turns a cold 2.33 requests per tile into a warm **0.33**
//! (measured over a flat archive; 2.96 → 0.42 normalized). Keying by range
//! instead would cache the same page under a dozen different keys.
//!
//! # The budget, measured
//!
//! [`PAGE_CACHE_BYTES`] is 4 MiB and [`PAGE_CACHE_ENTRIES`] 4096. The knee is
//! much earlier — 64 pages already reaches the 0.33 req/tile warm figure —
//! because a viewport's whole working set (index root, index interior, table
//! interior, the leaves under them) fits several times over.
//!
//! 4 MiB is **not** the point past which more memory buys nothing, and tiles
//! v1.5 corrected this paragraph for saying so. On a working set *larger* than
//! the cache — an antagonistic 60-spot tour over a 402 MB normalized archive,
//! whose `images_id` leaves alone are 7.12 MiB — raising the budget measured
//! −10.9 % range reads at 8 MiB and **−66.9 % at 16 MiB**. *(A direction and a
//! bound, measured through a re-implementation of these cache rules over real
//! SQLite archives, not a production reading.)*
//!
//! 4 MiB stays anyway, and for a stated reason rather than a mistaken one: 16
//! MiB is 12 MiB more **per provider**, which is unpayable on wasm and is paid
//! **twice** whenever one archive is open in a raster provider and a vector
//! transport at the same time. A warm pan — what a user actually does — saves
//! 0 % from the larger budget, so the cost is certain and the benefit is not.
//!
//! # Every walk is resumable
//!
//! Nothing here blocks: a walk that needs a page it does not have answers
//! [`PageRun`] — "these pages, please" — and is stepped again when they arrive.
//! That is what lets the same code run on `wasm32`, where there is no thread to
//! block, and on the render thread, where blocking is forbidden. The tests drive
//! it over an in-memory image; production drives it over a
//! [`crate::RangeTransport`].

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use oxigis_render::ByteRange;

use crate::gpkg_input::sqlite::{
    DB_HEADER_LEN, PAGE_INTERIOR_INDEX, PAGE_INTERIOR_TABLE, PAGE_LEAF_INDEX, PAGE_LEAF_TABLE,
    cell_offsets_in, inline_len_for, varint,
};
use crate::local_vector::LocalVectorError;

/// Memory budget for cached pages, in bytes.
///
/// Measured: 64 pages already gives 0.33 requests per tile on a warm pan, and
/// improvement stops well before this. 4 MiB is the point past which more memory
/// buys nothing, chosen there rather than at the knee so an unusually deep
/// archive or an unusually large page size still fits its working set.
pub(crate) const PAGE_CACHE_BYTES: usize = 4 * 1024 * 1024;

/// Secondary cap on how many pages are held at once.
///
/// Binds only for 512-byte pages, where 4 MiB would otherwise mean 8192
/// entries — [`PageCache`]'s own per-entry bookkeeping (a hash-map slot plus a
/// recency-order entry) is worth bounding independently of the raw bytes it
/// holds.
pub(crate) const PAGE_CACHE_ENTRIES: usize = 4096;

/// Largest overflow chain this reader will follow.
///
/// At the 4096-byte page size that is a 16 MiB record, which is
/// [`MAX_RECORD_BYTES`] — so the two caps agree and a chain that claims more is
/// refused before a single page of it is read.
pub(crate) const MAX_OVERFLOW_PAGES: u32 = 4096;

/// Largest record this reader will assemble.
///
/// Checked **before** the chain length is computed, so a cell claiming a 2 GB
/// payload is a named refusal rather than an allocation.
pub(crate) const MAX_RECORD_BYTES: usize = 16 * 1024 * 1024;

/// How many rounds of chain reading one record may take.
///
/// Speculate, span-refetch, then hop-by-hop; four leaves one spare. A chain that
/// still is not resolved is refused rather than walked for ever.
pub(crate) const MAX_CHAIN_ROUNDS: u32 = 4;

/// How many failed speculations flip an archive to hop-by-hop, until the next
/// verified success un-flips it.
///
/// Measured: overflow chains are 100 % contiguous on a freshly written archive
/// and 80.3 % after churn, so speculation is right far more often than not — but
/// an archive that has failed three times *in a row* is very likely mid-churn,
/// and every further speculation there is a wasted round trip. Churn is a
/// property of a *region*, not the whole archive, which is why a success
/// resets the count (see [`PageSource::note_speculation_success`]) rather than
/// letting three failures anywhere in a session condemn every chain read for
/// the rest of it.
pub(crate) const SPECULATION_FAILURE_LIMIT: u32 = 3;

/// Extra chain-page requests [`TileDescent`](super::descend::TileDescent)
/// allows beyond what one record's own declared length justifies.
///
/// Covers a failed speculation's one wasted round trip plus a span
/// escalation's one more (see `OverflowChain::step`'s mode transitions) —
/// never a multiple of the record's own size, which would defeat the point of
/// sizing the budget to it at all.
pub(crate) const CHAIN_READ_SLACK: u32 = 8;

/// One page's usable bytes.
pub(crate) type Page = Arc<[u8]>;

/// A run of consecutive pages a walk is waiting for.
///
/// Consecutive because that is what makes it **one** range request: a b-tree
/// descent asks for one page at a time, but an overflow chain — the case that
/// actually moves bytes — is contiguous on 80–100 % of real archives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PageRun {
    /// First page of the run, 1-based.
    pub(crate) first: u32,
    /// How many pages it covers.
    pub(crate) count: u32,
}

impl PageRun {
    /// A run of one page.
    pub(crate) const fn one(first: u32) -> Self {
        Self { first, count: 1 }
    }
}

/// Builds a reader error, prefixed the way `gpkg_input`'s is so a status line
/// says which layer of the stack refused the archive.
pub(crate) fn err(message: impl AsRef<str>) -> LocalVectorError {
    LocalVectorError::new(format!("SQLite: {}", message.as_ref()))
}

/// Pages held for one archive, keyed by page number, with recency tracked
/// separately for O(1) lookup and O(log n) eviction.
///
/// A `Vec<(u32, Page)>` scanned and shifted front-to-back — this cache's
/// original shape — makes every one of `get`/`has`/`insert` an O(n) walk, and
/// `get`/`insert` a further O(n) memmove to move an entry to the front. On the
/// per-tile hot path, inside the provider lock, per viewport, that cost grows
/// with exactly the size that makes the cache worth having. A page number
/// deterministically maps to one hash-map slot instead.
#[derive(Debug, Default)]
struct PageCache {
    /// Page number → its bytes and the recency tick it currently owns.
    entries: HashMap<u32, (Page, u64)>,
    /// Recency tick → page number, oldest (smallest tick) first — the
    /// eviction order. A page's tick here always matches the one recorded
    /// beside it in `entries`; the two are kept in lock step by every method
    /// below.
    recency: BTreeMap<u64, u32>,
    /// The next tick to hand out. Never reused, so two entries can never tie.
    clock: u64,
    /// Sum of every entry's length.
    bytes: usize,
}

impl PageCache {
    /// Records `number` as just used, retiring `previous` if it held an
    /// earlier tick, and returns the new one.
    fn touch(&mut self, number: u32, previous: Option<u64>) -> u64 {
        if let Some(previous) = previous {
            self.recency.remove(&previous);
        }
        let tick = self.clock;
        self.clock = self.clock.wrapping_add(1);
        self.recency.insert(tick, number);
        tick
    }

    /// The page numbered `number`, refreshing its recency.
    fn get(&mut self, number: u32) -> Option<Page> {
        let (page, tick) = self.entries.remove(&number)?;
        let tick = self.touch(number, Some(tick));
        self.entries.insert(number, (Arc::clone(&page), tick));
        Some(page)
    }

    /// Whether the page is held, without touching its recency.
    fn has(&self, number: u32) -> bool {
        self.entries.contains_key(&number)
    }

    /// Stores `page`, evicting the least recently used entries until both
    /// bounds hold again.
    fn insert(&mut self, number: u32, page: Page) {
        let previous_tick = self.entries.remove(&number).map(|(evicted, tick)| {
            self.bytes = self.bytes.saturating_sub(evicted.len());
            tick
        });
        self.bytes = self.bytes.saturating_add(page.len());
        let tick = self.touch(number, previous_tick);
        self.entries.insert(number, (page, tick));
        while self.entries.len() > 1
            && (self.bytes > PAGE_CACHE_BYTES || self.entries.len() > PAGE_CACHE_ENTRIES)
        {
            let Some((&oldest_tick, &oldest_number)) = self.recency.first_key_value() else {
                break;
            };
            self.recency.remove(&oldest_tick);
            if let Some((evicted, _)) = self.entries.remove(&oldest_number) {
                self.bytes = self.bytes.saturating_sub(evicted.len());
            }
        }
    }
}

/// The pages of one SQLite image, fetched a run at a time and cached by number.
#[derive(Debug)]
pub(crate) struct PageSource {
    /// Bytes per page, from the 100-byte header.
    page_size: usize,
    /// Bytes of each page a b-tree may use.
    usable_size: usize,
    /// How many pages the file holds, when that can be known or bounded.
    ///
    /// [`None`] means "not knowable": the header's count could not be trusted
    /// and nothing bounded it, so the only defence left is that a page past the
    /// end comes back short and is refused by name.
    page_count: Option<u32>,
    /// Held pages.
    cache: PageCache,
    /// How many page runs have been supplied — the number the read caps are
    /// stated in.
    supplied_runs: u64,
    /// How many speculative chain reads have come back wrong.
    speculation_failures: u32,
}

impl PageSource {
    /// A source over pages of `page_size` bytes, `usable_size` of them usable.
    pub(crate) fn new(page_size: usize, usable_size: usize, page_count: Option<u32>) -> Self {
        Self {
            page_size,
            usable_size,
            page_count,
            cache: PageCache::default(),
            supplied_runs: 0,
            speculation_failures: 0,
        }
    }

    /// Bytes of each page a b-tree may use.
    pub(crate) const fn usable_size(&self) -> usize {
        self.usable_size
    }

    /// How many pages the file holds, when that is known.
    pub(crate) const fn page_count(&self) -> Option<u32> {
        self.page_count
    }

    /// Whether speculation has failed often enough to be abandoned.
    pub(crate) const fn speculation_exhausted(&self) -> bool {
        self.speculation_failures >= SPECULATION_FAILURE_LIMIT
    }

    /// Records one failed speculative chain read.
    pub(crate) fn note_speculation_failure(&mut self) {
        self.speculation_failures = self.speculation_failures.saturating_add(1);
    }

    /// Records one **verified** contiguous chain read, clearing whatever
    /// failures came before it.
    ///
    /// Non-contiguity clusters in churned regions rather than spreading
    /// evenly (see [`SPECULATION_FAILURE_LIMIT`]'s doc), so an archive that
    /// has just proven a chain genuinely was contiguous is not the archive
    /// three stale failures described — the next chain deserves to be
    /// speculated on again rather than inheriting a count from a region this
    /// read has left behind.
    pub(crate) fn note_speculation_success(&mut self) {
        self.speculation_failures = 0;
    }

    /// Whether `number` is a page this file could hold at all.
    ///
    /// The declared count is only ever an **upper** bound: a header that cannot
    /// be trusted leaves [`None`] here and every page is accepted until a short
    /// delivery refuses it by name.
    pub(crate) fn is_addressable(&self, number: u32) -> bool {
        number != 0 && self.page_count.is_none_or(|count| number <= count)
    }

    /// The page numbered `number`, if it is held.
    pub(crate) fn get(&mut self, number: u32) -> Option<Page> {
        self.cache.get(number)
    }

    /// Whether every page of `run` is held.
    pub(crate) fn holds_run(&self, run: PageRun) -> bool {
        (0..run.count).all(|offset| self.cache.has(run.first.saturating_add(offset)))
    }

    /// The byte range `run` occupies.
    ///
    /// # Errors
    ///
    /// Refuses a run that starts at page 0 or whose extent does not fit a `u64`.
    pub(crate) fn range_for(&self, run: PageRun) -> Result<ByteRange, LocalVectorError> {
        if run.first == 0 || run.count == 0 {
            return Err(err(format!(
                "page run {}+{} is not a run of real pages",
                run.first, run.count
            )));
        }
        let size = u64::try_from(self.page_size).unwrap_or(u64::MAX);
        let start = u64::from(run.first - 1) * size;
        let end = start.saturating_add(u64::from(run.count) * size);
        ByteRange::new(start, end).map_err(|error| err(error.to_string()))
    }

    /// Splits a delivered span starting at page `first` into pages.
    ///
    /// A trailing partial page is dropped rather than stored: a short delivery
    /// at the end of the file is legitimate (the survey over-asks), and half a
    /// page is not a page. Whatever asked for it discovers the gap when
    /// [`Self::get`] answers [`None`] and refuses by name.
    pub(crate) fn supply(&mut self, first: u32, bytes: &[u8]) {
        self.supplied_runs = self.supplied_runs.saturating_add(1);
        for (index, chunk) in bytes.chunks(self.page_size).enumerate() {
            let Some(usable) = chunk.get(..self.usable_size) else {
                break;
            };
            let Ok(offset) = u32::try_from(index) else {
                break;
            };
            let Some(number) = first.checked_add(offset) else {
                break;
            };
            self.cache
                .insert(number, Arc::from(usable.to_vec().into_boxed_slice()));
        }
    }
}

/// One b-tree page, ready to be walked.
#[derive(Debug, Clone)]
pub(crate) struct PageView {
    /// Its 1-based number, which decides where the b-tree header sits.
    number: u32,
    /// Its usable bytes.
    bytes: Page,
}

impl PageView {
    /// Wraps a delivered page.
    pub(crate) const fn new(number: u32, bytes: Page) -> Self {
        Self { number, bytes }
    }

    /// Where the b-tree header starts: 100 on page 1, 0 everywhere else.
    const fn base(&self) -> usize {
        if self.number == 1 { DB_HEADER_LEN } else { 0 }
    }

    /// The page's b-tree type byte.
    ///
    /// # Errors
    ///
    /// Refuses a page with no b-tree header at all, and one whose type byte is
    /// not one of the four the format defines.
    pub(crate) fn kind(&self) -> Result<u8, LocalVectorError> {
        let base = self.base();
        let kind = *self
            .bytes
            .get(base)
            .ok_or_else(|| err(format!("page {} has no b-tree header", self.number)))?;
        match kind {
            PAGE_INTERIOR_INDEX | PAGE_INTERIOR_TABLE | PAGE_LEAF_INDEX | PAGE_LEAF_TABLE => {
                Ok(kind)
            }
            other => Err(err(format!("page {} has b-tree type {other}", self.number))),
        }
    }

    /// How many cells the page declares.
    fn cell_count(&self) -> Result<usize, LocalVectorError> {
        let base = self.base();
        let header = self
            .bytes
            .get(base..base + 8)
            .ok_or_else(|| err(format!("page {} has no b-tree header", self.number)))?;
        Ok(usize::from(u16::from_be_bytes([header[3], header[4]])))
    }

    /// The cell-content offsets, validated against the page.
    pub(crate) fn cells(&self) -> Result<Vec<usize>, LocalVectorError> {
        let kind = self.kind()?;
        let header_len = if matches!(kind, PAGE_INTERIOR_INDEX | PAGE_INTERIOR_TABLE) {
            12
        } else {
            8
        };
        cell_offsets_in(
            &self.bytes,
            self.base(),
            header_len,
            self.cell_count()?,
            self.number,
        )
    }

    /// The right-most child pointer of an interior page.
    pub(crate) fn right_child(&self) -> Result<u32, LocalVectorError> {
        let base = self.base();
        let slice = self
            .bytes
            .get(base + 8..base + 12)
            .ok_or_else(|| err(format!("page {} has a truncated header", self.number)))?;
        Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    /// The 4-byte big-endian page number at `at`.
    pub(crate) fn pointer_at(&self, at: usize) -> Result<u32, LocalVectorError> {
        let slice = self
            .bytes
            .get(at..at + 4)
            .ok_or_else(|| err(format!("page {} has a truncated cell", self.number)))?;
        Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    /// A varint at `at`, and how many bytes it took.
    pub(crate) fn varint_at(&self, at: usize) -> Result<(i64, usize), LocalVectorError> {
        varint(&self.bytes, at)
    }

    /// `len` bytes at `at`.
    pub(crate) fn slice(&self, at: usize, len: usize) -> Result<&[u8], LocalVectorError> {
        self.bytes
            .get(at..at + len)
            .ok_or_else(|| err(format!("page {} has a truncated cell", self.number)))
    }
}

/// A cell's payload, as far as it is known: what is inline and where the rest
/// is.
#[derive(Debug, Clone)]
pub(crate) struct CellPayload {
    /// The whole record's declared length.
    pub(crate) total: usize,
    /// The bytes that live on the page itself.
    pub(crate) inline: Vec<u8>,
    /// First page of the overflow chain, when the record spilled.
    pub(crate) overflow: Option<u32>,
}

impl CellPayload {
    /// Reads a payload of `total` bytes starting at `at` on `page`, using
    /// `split` to decide where the inline half stops.
    ///
    /// `split` is [`inline_len_for`] for a table leaf and
    /// [`crate::gpkg_input::sqlite::index_inline_len_for`] for an index cell —
    /// the two halves of the same rule, kept in `gpkg_input::sqlite` so the
    /// resident reader and this one can never disagree about where a record
    /// stops.
    ///
    /// # Errors
    ///
    /// Refuses a payload past [`MAX_RECORD_BYTES`] **before** computing
    /// anything from it, and a cell whose inline half runs off its page.
    pub(crate) fn read(
        page: &PageView,
        at: usize,
        payload_len: i64,
        usable: usize,
        split: fn(usize, usize) -> usize,
    ) -> Result<Self, LocalVectorError> {
        let total = usize::try_from(payload_len)
            .ok()
            .filter(|len| *len <= MAX_RECORD_BYTES)
            .ok_or_else(|| {
                err(format!(
                    "a cell claims a {payload_len}-byte record, past the {MAX_RECORD_BYTES}-byte \
                     limit"
                ))
            })?;
        let inline_len = split(usable, total);
        let inline = page.slice(at, inline_len)?.to_vec();
        let overflow = if inline_len == total {
            None
        } else {
            Some(page.pointer_at(at + inline_len)?)
        };
        Ok(Self {
            total,
            inline,
            overflow,
        })
    }

    /// Whether the record is complete without reading anything more.
    pub(crate) fn is_complete(&self) -> bool {
        self.overflow.is_none()
    }
}

/// How the overflow chain is being read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChainMode {
    /// One range read for the whole chain, assumed contiguous, then verified.
    Speculate,
    /// One range read spanning the pointers the failed speculation revealed.
    Span,
    /// One page at a time, following each page's own "next" pointer.
    HopByHop,
}

/// What an [`OverflowChain`] wants next.
#[derive(Debug)]
pub(crate) enum ChainStep {
    /// These pages are needed.
    Need(PageRun),
    /// The record is complete.
    Done(Vec<u8>),
}

/// A record's overflow chain, read speculatively and **verified**.
///
/// # Why verification is not optional
///
/// The chain is fetched as one range on the assumption that it is contiguous,
/// which is true of 100 % of freshly written archives and 80.3 % after churn.
/// When it is not, the bytes that come back are some *other* record's — and they
/// decode. Without checking every page's own four-byte "next" pointer against
/// the page that actually followed it in the run, this reader would return
/// silently corrupt tiles, which is the worst failure mode in the whole design.
///
/// A mismatch costs one extra range read: the pointers the wrong run revealed
/// bound a span that, on the measured churn cases, covers the real chain
/// (failures land inside ~10-page windows). Only if that fails too does it fall
/// back to a page per round trip.
#[derive(Debug)]
pub(crate) struct OverflowChain {
    /// The whole record's length.
    total: usize,
    /// What has been collected so far, inline bytes first.
    collected: Vec<u8>,
    /// The next page of the chain.
    next: u32,
    /// How it is being read.
    mode: ChainMode,
    /// How many rounds have been spent.
    rounds: u32,
    /// Pages already consumed, so a chain that points at itself cannot loop.
    visited: BTreeSet<u32>,
    /// The span a failed speculation revealed.
    span: Option<PageRun>,
    /// The run already asked for, so a short delivery cannot become a loop.
    asked: Option<PageRun>,
}

impl OverflowChain {
    /// Starts a chain for a spilled `payload`.
    ///
    /// # Errors
    ///
    /// Refuses a payload that did not spill (there is no chain to read) and one
    /// whose arithmetic does not fit.
    pub(crate) fn start(payload: CellPayload) -> Result<Self, LocalVectorError> {
        let next = payload
            .overflow
            .ok_or_else(|| err("a complete record has no overflow chain"))?;
        Ok(Self {
            total: payload.total,
            collected: payload.inline,
            next,
            mode: ChainMode::Speculate,
            rounds: 0,
            visited: BTreeSet::new(),
            span: None,
            asked: None,
        })
    }

    /// How many pages are still outstanding, given the usable page size.
    fn remaining_pages(&self, usable: usize) -> u32 {
        let per_page = usable.saturating_sub(4).max(1);
        let left = self.total.saturating_sub(self.collected.len());
        u32::try_from(left.div_ceil(per_page)).unwrap_or(u32::MAX)
    }

    /// The page-request budget this chain's own declared length justifies:
    /// what [`TileDescent`](super::descend::TileDescent) charges its page
    /// requests against instead of the b-tree descent's read cap, so a
    /// churned archive that forces a large but legitimate record through
    /// hop-by-hop reading costs pages, never an outright refusal — while a
    /// chain that keeps asking for more than its own length ever justified
    /// still is one, exactly as [`MAX_OVERFLOW_PAGES`] and
    /// [`MAX_RECORD_BYTES`] already bound it independently of this.
    pub(crate) fn budget(&self, usable: usize) -> u32 {
        self.remaining_pages(usable)
            .saturating_add(CHAIN_READ_SLACK)
    }

    /// Drives the chain one step.
    ///
    /// # Errors
    ///
    /// Refuses a chain longer than [`MAX_OVERFLOW_PAGES`], one that ends early,
    /// one that revisits a page, one that leaves the file, and one that is still
    /// unresolved after [`MAX_CHAIN_ROUNDS`].
    pub(crate) fn step(&mut self, source: &mut PageSource) -> Result<ChainStep, LocalVectorError> {
        if self.collected.len() >= self.total {
            self.collected.truncate(self.total);
            return Ok(ChainStep::Done(core::mem::take(&mut self.collected)));
        }
        if self.rounds >= MAX_CHAIN_ROUNDS {
            return Err(err(format!(
                "an overflow chain was still unresolved after {MAX_CHAIN_ROUNDS} rounds"
            )));
        }
        let usable = source.usable_size();
        let remaining = self.remaining_pages(usable);
        if remaining > MAX_OVERFLOW_PAGES {
            return Err(err(format!(
                "an overflow chain of {remaining} pages is past the limit of {MAX_OVERFLOW_PAGES}"
            )));
        }
        if !source.is_addressable(self.next) {
            return Err(err(format!(
                "overflow page {} is outside the file",
                self.next
            )));
        }
        if source.speculation_exhausted() && self.mode == ChainMode::Speculate {
            self.mode = ChainMode::HopByHop;
        }
        let run = match self.mode {
            ChainMode::Speculate => PageRun {
                first: self.next,
                // Never speculate past the end of the file: the run would come
                // back short for ever and the walk would never progress.
                count: source.page_count().map_or(remaining, |pages| {
                    remaining.min(pages.saturating_sub(self.next).saturating_add(1))
                }),
            },
            ChainMode::Span => self.span.unwrap_or(PageRun::one(self.next)),
            ChainMode::HopByHop => PageRun::one(self.next),
        };
        if !source.holds_run(run) {
            if self.asked != Some(run) {
                self.asked = Some(run);
                return Ok(ChainStep::Need(run));
            }
            // Asked for once and it did not all arrive — a short delivery at the
            // end of the file. Speculation copes (its verification finds the
            // gap); a linked read cannot, and says so.
            if self.mode != ChainMode::Speculate {
                return Err(err(format!(
                    "overflow page {} could not be read: it is past the end of the file",
                    run.first
                )));
            }
        }
        self.asked = None;
        // Only the *strategy* rounds are counted. Hop-by-hop is bounded by the
        // visited set, by `MAX_OVERFLOW_PAGES` and by the descent's own read
        // cap; counting each hop here would refuse a perfectly valid long chain.
        if self.mode != ChainMode::HopByHop {
            self.rounds = self.rounds.saturating_add(1);
        }
        match self.mode {
            ChainMode::Speculate => self.consume_speculated(source, run),
            ChainMode::Span | ChainMode::HopByHop => self.consume_linked(source),
        }
    }

    /// Consumes a run assumed contiguous, verifying every "next" pointer.
    fn consume_speculated(
        &mut self,
        source: &mut PageSource,
        run: PageRun,
    ) -> Result<ChainStep, LocalVectorError> {
        let usable = source.usable_size();
        let mut collected = Vec::new();
        let mut verified = true;
        for offset in 0..run.count {
            let number = run.first.saturating_add(offset);
            let Some(page) = source.get(number) else {
                verified = false;
                break;
            };
            let Some(head) = page.get(..4) else {
                verified = false;
                break;
            };
            let next = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
            let expected_last = offset + 1 == run.count;
            let follows = if expected_last {
                next == 0
            } else {
                next == number.saturating_add(1)
            };
            if !follows {
                // The chain is NOT contiguous here. Everything read so far is
                // still real (it was verified page by page up to this point),
                // but this page's successor is elsewhere.
                verified = false;
                self.span = Some(span_over(number, next, source));
                break;
            }
            let wanted = (self.total - self.collected.len() - collected.len()).min(usable - 4);
            let Some(content) = page.get(4..4 + wanted) else {
                verified = false;
                break;
            };
            collected.extend_from_slice(content);
        }
        if verified {
            source.note_speculation_success();
            self.collected.extend_from_slice(&collected);
            self.collected.truncate(self.total);
            return Ok(ChainStep::Done(core::mem::take(&mut self.collected)));
        }
        source.note_speculation_failure();
        self.mode = if self.span.is_some() {
            ChainMode::Span
        } else {
            ChainMode::HopByHop
        };
        self.step(source)
    }

    /// Consumes pages by following each one's own "next" pointer.
    fn consume_linked(&mut self, source: &mut PageSource) -> Result<ChainStep, LocalVectorError> {
        let usable = source.usable_size();
        while self.collected.len() < self.total {
            if self.next == 0 {
                return Err(err(format!(
                    "an overflow chain ended {} bytes early",
                    self.total - self.collected.len()
                )));
            }
            if !source.is_addressable(self.next) {
                return Err(err(format!(
                    "overflow page {} is outside the file",
                    self.next
                )));
            }
            if !self.visited.insert(self.next) {
                return Err(err(format!(
                    "overflow page {} is reachable twice (a cycle)",
                    self.next
                )));
            }
            let Some(page) = source.get(self.next) else {
                // The span did not reach far enough; ask for this page alone.
                self.visited.remove(&self.next);
                self.mode = ChainMode::HopByHop;
                let run = PageRun::one(self.next);
                self.asked = Some(run);
                return Ok(ChainStep::Need(run));
            };
            let Some(head) = page.get(..4) else {
                return Err(err(format!("overflow page {} is truncated", self.next)));
            };
            let wanted = (self.total - self.collected.len()).min(usable - 4);
            let content = page
                .get(4..4 + wanted)
                .ok_or_else(|| err(format!("overflow page {} is truncated", self.next)))?;
            self.collected.extend_from_slice(content);
            self.next = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
        }
        self.collected.truncate(self.total);
        Ok(ChainStep::Done(core::mem::take(&mut self.collected)))
    }
}

/// The page run spanning `from` and the pointer it revealed.
///
/// Measured: when a chain is not contiguous, the break lands inside a ~10-page
/// window, so one refetch over `min..=max` of what is known resolves it.
fn span_over(from: u32, to: u32, source: &PageSource) -> PageRun {
    let low = from.min(to).max(1);
    let high = from.max(to);
    let count = high.saturating_sub(low).saturating_add(1);
    let count = count.min(MAX_OVERFLOW_PAGES);
    let count = source
        .page_count()
        .map_or(count, |pages| count.min(pages.saturating_sub(low) + 1));
    PageRun { first: low, count }
}

/// Where a table-leaf cell's record begins, and how long it is.
///
/// # Errors
///
/// Refuses a cell whose varints run off the page.
pub(crate) fn table_leaf_cell(
    page: &PageView,
    at: usize,
    usable: usize,
) -> Result<(i64, CellPayload), LocalVectorError> {
    let (payload_len, consumed) = page.varint_at(at)?;
    let (rowid, rowid_len) = page.varint_at(at + consumed)?;
    let payload = CellPayload::read(
        page,
        at + consumed + rowid_len,
        payload_len,
        usable,
        inline_len_for,
    )?;
    Ok((rowid, payload))
}
