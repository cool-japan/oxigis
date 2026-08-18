// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! [`TileDescent`]: one tile address turned into one tile body, a page at a
//! time.
//!
//! ```text
//! flat        Index -> Row                          (2 + 3 pages measured)
//! normalized  Index -> Row -> ImageIndex -> ImageRow (2 + 2 + 3 + 3 = 9-11)
//! either      -> Overflow, when the body spilled
//! ```
//!
//! Measured fanouts on real archives — 207–334 keys per index interior page,
//! 265–450 per table interior page — put the index at depth 4 even at 10⁷ tiles,
//! which is where [`MAX_PAGE_READS_PER_TILE`] comes from.
//!
//! # The three ways an index descent silently returns the wrong tile
//!
//! All three are refused by name rather than risked:
//!
//! * **comparing a key short.** An index cell whose record spilled keeps only
//!   its first few hundred bytes on the page, so its key is *truncated*. A
//!   comparison against the truncated form orders it wrongly. Such a cell is
//!   refused, never compared.
//! * **comparing with the wrong collation.** Keys compare byte-for-byte
//!   (`BINARY`) here; an index declared `NOCASE` orders its leaves differently
//!   and a byte-comparing descent walks into the wrong subtree. The survey
//!   refuses those archives before a descent starts.
//! * **assuming ascending order.** A per-column `DESC` genuinely reorders the
//!   leaves (verified against real SQLite), so the sign of every comparison at
//!   that position is flipped from [`super::survey::KeyColumn::descending`].
//!
//! # Interior index cells carry real keys
//!
//! Unlike a table b-tree — whose interior cells hold only a *separator* rowid —
//! an index b-tree's interior cells hold complete key records. An exact match at
//! an interior level is therefore a **hit**, not a reason to keep descending,
//! which is worth one page read on a good fraction of lookups.
//!
//! # The TMS flip happens once, on the target
//!
//! MBTiles counts `tile_row` from the south. Rather than flipping every key read
//! out of the index, the flip is applied to the address being looked *for*,
//! through the same [`oxigis_render::source::tms_row`] the `{-y}` URL
//! placeholder expands with — one rule, never re-derived.
//!
//! # Two optimisations that were measured and ruled OUT (tiles v1.5)
//!
//! Recorded here, with their numbers, so neither is re-derived. Both were
//! measured twice and independently; neither is re-opened without a new
//! measurement of the kind named at the end of each.
//!
//! ## A `tile_id` → images-rowid cache, to skip the `ImageIndex` hop
//!
//! v1.4 deferred this with "would cut 9–11 reads to ~7 — measure first". It
//! was measured, and it saves **nothing**.
//!
//! Driving the production state machine over a counting harness with an
//! **unbounded, never-evicting** id map — the ceiling any LRU could reach —
//! saved **0 range reads on every trace**, while the map itself was hit 736 to
//! 1008 times:
//!
//! ```text
//! trace                          addresses  distinct ids  reads  ideal  hits
//! z6 32x32 sweep, no dedup            1024          1024    112    112     0
//! z6 32x32 sweep, 4x dedup            1024           256     52     52   768
//! z6 32x32 sweep, 16x dedup           1024            64     36     36   960
//! z6 32x32 sweep, 64x dedup           1024            16     33     33  1008
//! 6-city x 8-round churn tour           768           64     29     29   736
//! ```
//!
//! The reason is structural: the `images_id` leaf answering the *second*
//! occurrence of a `tile_id` is the page that answered the first, and
//! [`super::source::PageCache`] still holds it. A cache in front of a cache that
//! never misses buys nothing. A repeated tile *address* never even re-descends —
//! the raster provider returns from its ready cache and the vector provider from
//! its decoded cache, above this module entirely.
//!
//! On a real 402 MB archive whose `images_id` leaves (7.12 MiB) exceed
//! [`super::source::PAGE_CACHE_BYTES`], a 4096-entry LRU measured **0 %** on a
//! warm pan and **−0.5 %** (1.1 % hit rate) once address repetition was removed
//! from the trace. The −25.6 % seen on an intermediate trace was address
//! repetition, not `tile_id` de-duplication. *(That figure is a bound measured
//! through a re-implementation of the cache rules, not a production reading.)*
//!
//! Density runs the wrong way too: the archive's own index leaf remembers **1.21
//! to 1.81× more** mappings per byte than a `Vec<(String, i64)>` would, across
//! 8- to 64-character ids — and it answers the neighbouring ids for free. The
//! entire prize is CPU: 22.09 µs/lookup today against 11.81 with the ideal map,
//! i.e. **10.28 µs per tile and zero reads**, against the 250–1000 ms RTT this
//! design exists to save — one part in 25 000.
//!
//! Re-opens only on a trace over the real archive with a per-round working set
//! larger than the provider's ready cache, reporting warm-vs-cold *range reads*.
//!
//! ## Binary-searching the leaf arms
//!
//! `PAGE_LEAF_INDEX` and `PAGE_LEAF_TABLE` below decode every cell's full record
//! until they match, where ~log n probes would do. Measured, release, over
//! 20 480 warm lookups: 22.09 µs/lookup today, 7.17 with both leaves binary
//! searched (3.08×), 7.56 with a refusal-preserving varint pre-pass (2.92×).
//!
//! Ruled **OUT anyway, by the ruling above.** The saving is 14.5 µs of warm CPU
//! and **zero range reads** — the same axis on which 10.28 µs was called noise,
//! and a change cannot be noise when it argues against a cache and material when
//! it argues for itself. Against that it costs the only hot-path edit in the
//! workstream, in the two files v1.4 named the paged/resident divergence fence,
//! and it narrows **refusal reach**: a leaf cell malformed in a way the varint
//! pre-pass does not catch is no longer necessarily probed, degrading a *named*
//! refusal into `Absent`. Refusal behaviour in this crate is ruled on, not
//! drifted into.
//!
//! Re-opens only on a measurement showing leaf-scan time is material *where it
//! lands* — a 65 536-byte-page archive with hundreds of cells per leaf, against
//! the render thread's frame budget rather than a network RTT. If it is ever
//! taken it must arrive with the varint pre-pass and with hostile tests for an
//! out-of-order table leaf, an undecodable cell off the probe path, and a
//! spilled index key on a ≥ 32-cell leaf — never without them.

use core::cmp::Ordering;
use std::collections::BTreeSet;

use oxigis_render::TileId;
use oxigis_render::source::tms_row;

use crate::gpkg_input::sqlite::{
    CellValue, PAGE_INTERIOR_INDEX, PAGE_INTERIOR_TABLE, PAGE_LEAF_INDEX, PAGE_LEAF_TABLE,
    decode_record, index_inline_len_for,
};
use crate::local_vector::LocalVectorError;
use crate::mbtiles::schema::MbTilesFormat;

use super::source::{
    CellPayload, ChainStep, OverflowChain, PageRun, PageSource, PageView, err, table_leaf_cell,
};
use super::survey::{IndexPlan, PagedLayout};

/// How many page reads one tile's **b-tree descent** may cost before it is
/// refused: the index and row walks over the address table, and again over
/// `images` on a normalized archive.
///
/// Measured worst realistic case is 11–15 (a normalized archive at depth 4);
/// 24 is that times ~1.6. Deliberately excludes the pages an
/// [`OverflowChain`] reads while reassembling a spilled record — those are
/// charged against [`OverflowChain::budget`] instead (see
/// [`TileDescent::charge_chain`]), which scales with the *record's* own
/// declared length rather than with index depth, so a large but legitimate
/// tile forced through a hop-by-hop read by a churned archive never competes
/// with the b-tree descent for the same budget. The refusal **logs the
/// observed count**, so a real archive that trips it is diagnosable rather
/// than a mystery — this cap punishes malice, not size.
pub(crate) const MAX_PAGE_READS_PER_TILE: u32 = 24;

/// How deep any single b-tree walk may go.
///
/// Real trees are 2–4 levels even at 10⁷ tiles; twelve only stops a crafted
/// cycle-free but very deep pointer chain.
pub(crate) const MAX_DESCENT_DEPTH: u32 = 12;

/// One value of a lookup key.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum KeyValue {
    /// An integer key column — the three tile-address ones.
    Integer(i64),
    /// A text key column — `images.tile_id`.
    Text(String),
}

impl KeyValue {
    /// SQLite's storage-class rank: NULL < numeric < text < blob.
    const fn class(&self) -> u8 {
        match self {
            Self::Integer(_) => 1,
            Self::Text(_) => 2,
        }
    }
}

/// The storage-class rank of a decoded value.
const fn class_of(value: &CellValue) -> u8 {
    match value {
        CellValue::Null => 0,
        CellValue::Integer(_) | CellValue::Float(_) => 1,
        CellValue::Text(_) => 2,
        CellValue::Blob(_) => 3,
    }
}

/// Compares one key column against the value stored at that position.
///
/// Integers compare numerically, text compares **byte for byte** (`BINARY`, the
/// only collation the survey admits), and values of different storage classes
/// compare by SQLite's own class order.
fn compare_column(target: &KeyValue, stored: &CellValue) -> Ordering {
    match (target, stored) {
        (KeyValue::Integer(left), CellValue::Integer(right)) => left.cmp(right),
        #[expect(
            clippy::cast_precision_loss,
            reason = "SQLite itself compares an integer against a real by widening it"
        )]
        (KeyValue::Integer(left), CellValue::Float(right)) => (*left as f64)
            .partial_cmp(right)
            .unwrap_or(Ordering::Greater),
        (KeyValue::Text(left), CellValue::Text(right)) => left.as_bytes().cmp(right.as_bytes()),
        _ => target.class().cmp(&class_of(stored)),
    }
}

/// Compares a whole key against an index record's leading columns.
///
/// A record with fewer columns than the key has is *shorter*, which orders it
/// before the key — the same rule SQLite applies, and the reason a truncated
/// record could never be safely compared.
fn compare_key(key: &[KeyValue], plan: &IndexPlan, record: &[CellValue]) -> Ordering {
    for (position, target) in key.iter().enumerate() {
        let Some(stored) = record.get(position) else {
            return Ordering::Greater;
        };
        let mut ordering = compare_column(target, stored);
        if plan
            .columns
            .get(position)
            .is_some_and(|column| column.descending)
        {
            ordering = ordering.reverse();
        }
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

/// What one walk wants next.
#[derive(Debug)]
enum WalkStep<T> {
    /// These pages are needed.
    Need(PageRun),
    /// The walk found this.
    Found(T),
    /// There is nothing at that key.
    Absent,
}

/// A descent of one index b-tree, looking for one key.
#[derive(Debug)]
struct IndexWalk {
    /// The index being descended.
    plan: IndexPlan,
    /// The key being looked for.
    key: Vec<KeyValue>,
    /// The page the walk is on.
    number: u32,
    /// How deep it has gone.
    depth: u32,
    /// Pages already visited on this path.
    visited: BTreeSet<u32>,
}

impl IndexWalk {
    /// A walk of `plan` for `key`.
    fn new(plan: IndexPlan, key: Vec<KeyValue>) -> Self {
        let number = plan.root;
        Self {
            plan,
            key,
            number,
            depth: 0,
            visited: BTreeSet::new(),
        }
    }

    /// The rowid an index record's **last** column holds.
    ///
    /// Also the guard on the auto-index numbering rule, which is derived from
    /// the specification rather than read out of the file: the first record this
    /// walk actually decodes must be `(key…, rowid)` with an integer rowid, or
    /// the layout is refused by name. A silently wrong tile is worse than none.
    fn rowid_of(&mut self, record: &[CellValue]) -> Result<i64, LocalVectorError> {
        if record.len() != self.key.len() + 1 {
            return Err(err(format!(
                "this archive's {} index holds {}-column records where {} were expected \
                 (key columns plus a rowid), so it is not the index it is named as",
                self.plan.name,
                record.len(),
                self.key.len() + 1
            )));
        }
        match record.last() {
            Some(CellValue::Integer(rowid)) => Ok(*rowid),
            other => Err(err(format!(
                "this archive's {} index ends its records with {other:?} rather than a rowid",
                self.plan.name
            ))),
        }
    }

    /// Decodes one index cell's record, refusing a key that does not fit its
    /// page.
    fn record_at(
        &self,
        page: &PageView,
        at: usize,
        usable: usize,
    ) -> Result<Vec<CellValue>, LocalVectorError> {
        let (payload_len, consumed) = page.varint_at(at)?;
        let payload = CellPayload::read(
            page,
            at + consumed,
            payload_len,
            usable,
            index_inline_len_for,
        )?;
        if !payload.is_complete() {
            // The key is longer than the prefix SQLite guarantees to keep on the
            // page. Comparing the truncated form would order it wrongly and
            // return a plausible, wrong row — so the archive is refused instead.
            return Err(err(format!(
                "this archive's {} index holds a key that spilled onto an overflow page, which \
                 cannot be compared without reading it; download the archive and open the file, \
                 or use PMTiles for a remote archive",
                self.plan.name
            )));
        }
        decode_record(&payload.inline)
    }

    /// Drives the descent one page.
    fn step(&mut self, source: &mut PageSource) -> Result<WalkStep<i64>, LocalVectorError> {
        loop {
            if self.depth > MAX_DESCENT_DEPTH {
                return Err(err(format!(
                    "this archive's {} index is deeper than {MAX_DESCENT_DEPTH} levels",
                    self.plan.name
                )));
            }
            if !source.is_addressable(self.number) {
                return Err(err(format!("page {} is outside the file", self.number)));
            }
            if !self.visited.insert(self.number) {
                return Err(err(format!(
                    "page {} is reachable twice (a cycle)",
                    self.number
                )));
            }
            let Some(bytes) = source.get(self.number) else {
                self.visited.remove(&self.number);
                return Ok(WalkStep::Need(PageRun::one(self.number)));
            };
            let page = PageView::new(self.number, bytes);
            let usable = source.usable_size();
            match page.kind()? {
                PAGE_LEAF_INDEX => {
                    for at in page.cells()? {
                        let record = self.record_at(&page, at, usable)?;
                        if compare_key(&self.key, &self.plan, &record) == Ordering::Equal {
                            return Ok(WalkStep::Found(self.rowid_of(&record)?));
                        }
                    }
                    return Ok(WalkStep::Absent);
                }
                PAGE_INTERIOR_INDEX => {
                    let mut child = None;
                    for at in page.cells()? {
                        let left = page.pointer_at(at)?;
                        let record = self.record_at(&page, at + 4, usable)?;
                        match compare_key(&self.key, &self.plan, &record) {
                            // An interior index cell carries a REAL key, so an
                            // exact match here is the answer.
                            Ordering::Equal => {
                                return Ok(WalkStep::Found(self.rowid_of(&record)?));
                            }
                            Ordering::Less => {
                                child = Some(left);
                                break;
                            }
                            Ordering::Greater => {}
                        }
                    }
                    self.number = match child {
                        Some(child) => child,
                        None => page.right_child()?,
                    };
                    self.depth = self.depth.saturating_add(1);
                }
                PAGE_INTERIOR_TABLE | PAGE_LEAF_TABLE => {
                    return Err(err(format!(
                        "this archive's {} index is rooted at a table b-tree, so it is not an \
                         index at all",
                        self.plan.name
                    )));
                }
                other => {
                    return Err(err(format!("page {} has b-tree type {other}", self.number)));
                }
            }
        }
    }
}

/// A descent of one table b-tree, looking for one rowid.
#[derive(Debug)]
struct RowWalk {
    /// The rowid being looked for.
    rowid: i64,
    /// The page the walk is on.
    number: u32,
    /// How deep it has gone.
    depth: u32,
    /// Pages already visited on this path.
    visited: BTreeSet<u32>,
}

impl RowWalk {
    /// A walk of the table rooted at `root` for `rowid`.
    const fn new(root: u32, rowid: i64) -> Self {
        Self {
            rowid,
            number: root,
            depth: 0,
            visited: BTreeSet::new(),
        }
    }

    /// Drives the descent one page.
    fn step(&mut self, source: &mut PageSource) -> Result<WalkStep<CellPayload>, LocalVectorError> {
        loop {
            if self.depth > MAX_DESCENT_DEPTH {
                return Err(err(format!(
                    "this archive's table b-tree is deeper than {MAX_DESCENT_DEPTH} levels"
                )));
            }
            if !source.is_addressable(self.number) {
                return Err(err(format!("page {} is outside the file", self.number)));
            }
            if !self.visited.insert(self.number) {
                return Err(err(format!(
                    "page {} is reachable twice (a cycle)",
                    self.number
                )));
            }
            let Some(bytes) = source.get(self.number) else {
                self.visited.remove(&self.number);
                return Ok(WalkStep::Need(PageRun::one(self.number)));
            };
            let page = PageView::new(self.number, bytes);
            let usable = source.usable_size();
            match page.kind()? {
                PAGE_LEAF_TABLE => {
                    for at in page.cells()? {
                        let (found, payload) = table_leaf_cell(&page, at, usable)?;
                        if found == self.rowid {
                            return Ok(WalkStep::Found(payload));
                        }
                    }
                    return Ok(WalkStep::Absent);
                }
                PAGE_INTERIOR_TABLE => {
                    let mut child = None;
                    for at in page.cells()? {
                        // An interior table cell is a 4-byte left-child pointer
                        // followed by the LARGEST rowid in that subtree.
                        let (key, _used) = page.varint_at(at + 4)?;
                        if self.rowid <= key {
                            child = Some(page.pointer_at(at)?);
                            break;
                        }
                    }
                    self.number = match child {
                        Some(child) => child,
                        None => page.right_child()?,
                    };
                    self.depth = self.depth.saturating_add(1);
                }
                PAGE_INTERIOR_INDEX | PAGE_LEAF_INDEX => {
                    return Err(err(
                        "this archive's tile table is stored as an index b-tree (a WITHOUT ROWID \
                         table), which has no rowids",
                    ));
                }
                other => {
                    return Err(err(format!("page {} has b-tree type {other}", self.number)));
                }
            }
        }
    }
}

/// Which record an assembled overflow chain belongs to.
#[derive(Debug, Clone, Copy)]
enum ChainFor {
    /// The address table's record (`tiles` or `map`).
    Address,
    /// The `images` table's record.
    Image,
}

/// How far along one tile lookup is.
#[derive(Debug)]
enum Phase {
    /// Descending the `(zoom_level, tile_column, tile_row)` index.
    Index(Box<IndexWalk>),
    /// Fetching the address table's row.
    Row(Box<RowWalk>),
    /// Descending the `images.tile_id` index (normalized archives only).
    ImageIndex(Box<IndexWalk>),
    /// Fetching the `images` row.
    ImageRow(Box<RowWalk>),
    /// Reassembling a spilled record.
    Chain(Box<OverflowChain>, ChainFor),
    /// Nothing more will happen.
    Done,
}

/// What a [`TileDescent`] wants next.
#[derive(Debug)]
pub(crate) enum DescentStep {
    /// These pages are needed.
    Need(PageRun),
    /// The archive holds no tile at that address — a final, non-error answer.
    Absent,
    /// The tile's bytes.
    Body(Vec<u8>),
}

/// One tile address being resolved inside one paged archive.
#[derive(Debug)]
pub(crate) struct TileDescent {
    /// The address being looked for, in XYZ.
    tile: TileId,
    /// How far along it is.
    phase: Phase,
    /// How many b-tree descent page runs it has asked for.
    reads: u32,
    /// How many page runs the **current** overflow chain has asked for.
    chain_reads: u32,
    /// How many the current chain may ask for before it is refused — see
    /// [`OverflowChain::budget`]. Meaningless (and unused) outside
    /// [`Phase::Chain`].
    chain_budget: u32,
}

impl TileDescent {
    /// Starts a lookup for `tile`.
    ///
    /// The TMS flip is applied **here**, to the target, so nothing further down
    /// has to remember it.
    pub(crate) fn new(tile: TileId, layout: &PagedLayout) -> Self {
        let key = vec![
            KeyValue::Integer(i64::from(tile.z)),
            KeyValue::Integer(i64::from(tile.x)),
            KeyValue::Integer(i64::from(tms_row(tile.z, tile.y))),
        ];
        Self {
            tile,
            phase: Phase::Index(Box::new(IndexWalk::new(layout.address_index.clone(), key))),
            reads: 0,
            chain_reads: 0,
            chain_budget: 0,
        }
    }

    /// Counts one b-tree descent page run, refusing a lookup that has cost
    /// too many.
    fn charge(&mut self, run: PageRun) -> Result<DescentStep, LocalVectorError> {
        self.reads = self.reads.saturating_add(1);
        if self.reads > MAX_PAGE_READS_PER_TILE {
            self.phase = Phase::Done;
            return Err(err(format!(
                "reading tile {}/{}/{} has cost {} page reads, past the limit of \
                 {MAX_PAGE_READS_PER_TILE}",
                self.tile.z, self.tile.x, self.tile.y, self.reads
            )));
        }
        Ok(DescentStep::Need(run))
    }

    /// Counts one page run against the **current chain's own** budget,
    /// refusing a chain that has cost more page requests than its own
    /// declared length justifies — see [`OverflowChain::budget`].
    fn charge_chain(&mut self, run: PageRun) -> Result<DescentStep, LocalVectorError> {
        self.chain_reads = self.chain_reads.saturating_add(1);
        if self.chain_reads > self.chain_budget {
            self.phase = Phase::Done;
            return Err(err(format!(
                "reading tile {}/{}/{}'s spilled record has cost {} chain page reads, past the \
                 {}-page budget its own declared length allows",
                self.tile.z, self.tile.x, self.tile.y, self.chain_reads, self.chain_budget
            )));
        }
        Ok(DescentStep::Need(run))
    }

    /// Drives the lookup as far as the pages in `source` allow.
    ///
    /// # Errors
    ///
    /// Every refusal in this module's docs, plus whatever the b-tree walk and
    /// the overflow chain refuse.
    pub(crate) fn step(
        &mut self,
        layout: &PagedLayout,
        source: &mut PageSource,
    ) -> Result<DescentStep, LocalVectorError> {
        loop {
            match &mut self.phase {
                Phase::Done => return Ok(DescentStep::Absent),
                Phase::Index(walk) => match walk.step(source)? {
                    WalkStep::Need(run) => return self.charge(run),
                    WalkStep::Absent => {
                        self.phase = Phase::Done;
                        return Ok(DescentStep::Absent);
                    }
                    WalkStep::Found(rowid) => {
                        self.phase = Phase::Row(Box::new(RowWalk::new(layout.address_root, rowid)));
                    }
                },
                Phase::Row(walk) => match walk.step(source)? {
                    WalkStep::Need(run) => return self.charge(run),
                    WalkStep::Absent => {
                        self.phase = Phase::Done;
                        return Ok(DescentStep::Absent);
                    }
                    WalkStep::Found(payload) => {
                        if payload.is_complete() {
                            let record = decode_record(&payload.inline)?;
                            if let Some(next) = self.after_address(layout, &record)? {
                                return Ok(next);
                            }
                        } else {
                            let chain = OverflowChain::start(payload)?;
                            // Sized to THIS record, not to the b-tree cap:
                            // see `MAX_PAGE_READS_PER_TILE`'s doc.
                            self.chain_reads = 0;
                            self.chain_budget = chain.budget(source.usable_size());
                            self.phase = Phase::Chain(Box::new(chain), ChainFor::Address);
                        }
                    }
                },
                Phase::ImageIndex(walk) => match walk.step(source)? {
                    WalkStep::Need(run) => return self.charge(run),
                    WalkStep::Absent => {
                        self.phase = Phase::Done;
                        return Ok(DescentStep::Absent);
                    }
                    WalkStep::Found(rowid) => {
                        self.phase =
                            Phase::ImageRow(Box::new(RowWalk::new(layout.images_root, rowid)));
                    }
                },
                Phase::ImageRow(walk) => match walk.step(source)? {
                    WalkStep::Need(run) => return self.charge(run),
                    WalkStep::Absent => {
                        self.phase = Phase::Done;
                        return Ok(DescentStep::Absent);
                    }
                    WalkStep::Found(payload) => {
                        if payload.is_complete() {
                            let record = decode_record(&payload.inline)?;
                            self.phase = Phase::Done;
                            return Ok(body_of(&record, layout.images_data));
                        }
                        let chain = OverflowChain::start(payload)?;
                        self.chain_reads = 0;
                        self.chain_budget = chain.budget(source.usable_size());
                        self.phase = Phase::Chain(Box::new(chain), ChainFor::Image);
                    }
                },
                Phase::Chain(chain, which) => {
                    let which = *which;
                    match chain.step(source)? {
                        ChainStep::Need(run) => return self.charge_chain(run),
                        ChainStep::Done(payload) => {
                            let record = decode_record(&payload)?;
                            match which {
                                ChainFor::Address => {
                                    if let Some(next) = self.after_address(layout, &record)? {
                                        return Ok(next);
                                    }
                                }
                                ChainFor::Image => {
                                    self.phase = Phase::Done;
                                    return Ok(body_of(&record, layout.images_data));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Acts on the address table's decoded record.
    ///
    /// Flat: the record already holds the body. Normalized: it holds a
    /// `tile_id`, and the body is one more index descent away.
    fn after_address(
        &mut self,
        layout: &PagedLayout,
        record: &[CellValue],
    ) -> Result<Option<DescentStep>, LocalVectorError> {
        if layout.format == MbTilesFormat::Flat {
            self.phase = Phase::Done;
            return Ok(Some(body_of(record, layout.address_columns.payload)));
        }
        let Some(CellValue::Text(id)) = record.get(layout.address_columns.payload) else {
            // A map row whose `tile_id` is not text names no image: absent, not
            // an error — the same answer a missing row gives.
            self.phase = Phase::Done;
            return Ok(Some(DescentStep::Absent));
        };
        let index = layout
            .images_index
            .clone()
            .ok_or_else(|| err("this archive is normalized but has no index on images.tile_id"))?;
        self.phase = Phase::ImageIndex(Box::new(IndexWalk::new(
            index,
            vec![KeyValue::Text(id.clone())],
        )));
        Ok(None)
    }
}

/// The tile body held at `column` of a decoded record.
///
/// A row whose payload column is neither a blob nor text holds no tile, which is
/// [`DescentStep::Absent`] rather than a failure: an archive may legitimately
/// store a NULL there.
fn body_of(record: &[CellValue], column: usize) -> DescentStep {
    match record.get(column) {
        Some(CellValue::Blob(bytes)) => DescentStep::Body(bytes.clone()),
        Some(CellValue::Text(text)) => DescentStep::Body(text.clone().into_bytes()),
        _ => DescentStep::Absent,
    }
}
