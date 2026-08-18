// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The leaf-directory cache: **stored** blobs, byte-budgeted, offset-keyed,
//! move-to-front, with a one-entry decoded front cache.
//!
//! # Why the cache holds stored bytes rather than decoded entries
//!
//! Measured, on real archives:
//!
//! | | decoded [`DirEntry`] list | stored (still-coded) blob |
//! |---|---|---|
//! | one planet leaf | 60 740 × 24 B = **1 457 760 B** | **22 912 – 162 935 B** |
//! | leaves inside [`LEAF_CACHE_BYTES`] | ~11 | **100 – 700** |
//! | cost of a hit | free | 1.78 – 2.48 ms to re-decode |
//! | cost of a miss | — | **257 – 1027 ms** to refetch |
//!
//! A re-decode is therefore two orders of magnitude cheaper than the refetch it
//! replaces, and holding stored blobs buys ten to sixty times as many leaves for
//! the same memory. The pattern that punishes the decoded cache is a real one: a
//! six-city zoom-15 bookmark tour misses **66.3 %** of its leaf lookups and
//! moves 14.7 MiB of refetches; under this design it refetches **nothing**
//! (pinned by `an_antagonistic_trace_refetches_no_leaf_twice`). An ordinary pan
//! misses 0.1 % either way.
//!
//! # Why there is a one-entry decoded front cache
//!
//! A viewport is about sixteen tiles, and at any real zoom they land in one or
//! two leaves — so without a front cache a pan would pay the 1.8–2.5 ms decode
//! sixteen times for the same blob. Keyed by the leaf's own offset and holding
//! exactly one entry: a second entry would double the decoded memory to serve a
//! pattern (two leaves interleaved tile-by-tile) that a viewport does not
//! produce, and the *stored* tier behind it already makes the second leaf a
//! decode rather than a fetch.
//!
//! # Why bytes rather than entries
//!
//! A leaf directory is not a small object, and how big it is depends entirely on
//! which archive the user opened: the largest measured in the wild holds 60 740
//! entries while a small archive's holds a handful. A cache of "32 leaves" is
//! therefore not a bound at all. The budget is [`LEAF_CACHE_BYTES`]; an entry
//! count ([`LEAF_CACHE_ENTRIES`]) is kept only as a secondary guard so a
//! pathological archive of thousands of one-entry leaves cannot grow the
//! bookkeeping without limit.
//!
//! # Why not `TileCache`
//!
//! [`oxigis_render::TileCache`] is keyed by [`oxigis_render::TileId`] and is
//! used in six other places whose behaviour must not churn. A leaf is keyed by
//! its byte offset, so it gets its own tiny structure instead: a `Vec` with
//! move-to-front, linear-scanned. At [`LEAF_CACHE_ENTRIES`] entries the scan is
//! free next to the decode — let alone the range read — it is avoiding.

use std::sync::Arc;

use oxigis_render::pmtiles::DirEntry;

/// Memory budget for cached leaf directories, in bytes.
///
/// Counted over the **stored** blobs, which is what the cache holds. 16 MiB is
/// 100–700 real leaves (22 912–162 935 B each, measured) — an interactive pan's
/// working set across three zoom levels many times over, and enough for the
/// antagonistic bookmark-tour trace to refetch nothing at all. It is isolated
/// here so it can be re-tuned without touching the reader.
pub const LEAF_CACHE_BYTES: usize = 16 * 1024 * 1024;

/// Secondary cap on how many leaf directories are held at once.
///
/// [`LEAF_CACHE_BYTES`] alone admits ~700 real leaves, so a count of a few dozen
/// would silently become the binding limit and throw the whole point of storing
/// blobs away. 1024 leaves is past what any measured budget reaches and still
/// leaves the move-to-front scan free next to a 1.8 ms decode; it only binds for
/// archives whose leaves are far below the average size.
pub const LEAF_CACHE_ENTRIES: usize = 1024;

/// One cached leaf directory, as stored in the archive.
#[derive(Debug)]
struct Entry {
    /// Absolute file offset the leaf blob starts at — its identity.
    at: u64,
    /// The blob exactly as the archive stores it, still coded per the header's
    /// `internal_compression`. Shared so a decode can drop the state lock.
    stored: Arc<[u8]>,
}

/// Leaf directories, most recently used first, plus the one decoded in front.
#[derive(Debug, Default)]
pub(crate) struct LeafCache {
    /// Cached stored blobs, front = most recently used.
    entries: Vec<Entry>,
    /// Sum of every entry's stored length.
    total_bytes: usize,
    /// The single decoded directory kept in front, keyed by its offset.
    front: Option<(u64, Arc<Vec<DirEntry>>)>,
}

impl LeafCache {
    /// An empty cache.
    pub(crate) const fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_bytes: 0,
            front: None,
        }
    }

    /// The **decoded** directory at `at`, if it is the one in front.
    ///
    /// The fast path a viewport takes: sixteen tiles on one leaf decode it once.
    pub(crate) fn decoded(&self, at: u64) -> Option<Arc<Vec<DirEntry>>> {
        self.front
            .as_ref()
            .filter(|(held, _)| *held == at)
            .map(|(_, entries)| Arc::clone(entries))
    }

    /// The **stored** blob at `at`, refreshing its recency.
    ///
    /// [`Some`] means "no range read is needed, only a decode".
    pub(crate) fn stored(&mut self, at: u64) -> Option<Arc<[u8]>> {
        let index = self.entries.iter().position(|entry| entry.at == at)?;
        let entry = self.entries.remove(index);
        let shared = Arc::clone(&entry.stored);
        self.entries.insert(0, entry);
        Some(shared)
    }

    /// Stores `blob` under `at`, evicting from the back until both bounds hold
    /// again.
    ///
    /// A single blob larger than the whole budget is stored anyway and then
    /// immediately becomes the only entry: refusing it would mean the archive
    /// could not be read at all, and one oversized leaf is still bounded by
    /// [`oxigis_render::pmtiles::MAX_DIRECTORY_BYTES`].
    pub(crate) fn insert(&mut self, at: u64, stored: Arc<[u8]>) {
        self.remove(at);
        let bytes = stored.len();
        self.total_bytes = self.total_bytes.saturating_add(bytes);
        self.entries.insert(0, Entry { at, stored });
        while self.entries.len() > 1
            && (self.total_bytes > LEAF_CACHE_BYTES || self.entries.len() > LEAF_CACHE_ENTRIES)
        {
            if let Some(evicted) = self.entries.pop() {
                self.total_bytes = self.total_bytes.saturating_sub(evicted.stored.len());
                if self.front.as_ref().is_some_and(|(at, _)| *at == evicted.at) {
                    // The decoded copy must not outlive the blob it came from:
                    // holding it would make `leaf_stats` under-report the memory
                    // actually in use, which is the number the budget is about.
                    self.front = None;
                }
            }
        }
    }

    /// Keeps `entries` as the one decoded directory in front, for `at`.
    pub(crate) fn set_decoded(&mut self, at: u64, entries: Arc<Vec<DirEntry>>) {
        self.front = Some((at, entries));
    }

    /// Drops the leaf stored at `at`, if there is one.
    fn remove(&mut self, at: u64) {
        if let Some(index) = self.entries.iter().position(|entry| entry.at == at) {
            let evicted = self.entries.remove(index);
            self.total_bytes = self.total_bytes.saturating_sub(evicted.stored.len());
        }
    }

    /// How many leaves are held.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// How many **stored** bytes the held leaves account for.
    ///
    /// Deliberately not the decoded size: the decoded copy is one entry deep and
    /// transient, while this is the number [`LEAF_CACHE_BYTES`] bounds.
    pub(crate) const fn bytes(&self) -> usize {
        self.total_bytes
    }
}
