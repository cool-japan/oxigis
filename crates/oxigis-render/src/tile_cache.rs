//! Capacity-bounded least-recently-used cache keyed by [`TileId`].
//!
//! # Eviction bookkeeping
//!
//! The cache is hand-rolled from two `std` collections; no intrusive linked
//! list and therefore no `unsafe` (COOLJAPAN Policy #4):
//!
//! * `entries: HashMap<TileId, Entry<V>>` owns the values. Each entry records
//!   the monotonic *use stamp* it was last touched at.
//! * `order: BTreeMap<u64, TileId>` maps use stamps back to keys. Because
//!   stamps come from a strictly increasing counter, the map's first key is
//!   always the least-recently-used tile and its last key the most recent one.
//!
//! Every touch (a hit in [`TileCache::get`], or an [`TileCache::insert`])
//! allocates a fresh stamp, removes the entry's previous stamp from `order`,
//! and inserts the new one — so `order` holds exactly one stamp per live entry
//! and `order.len() == entries.len()` is an invariant. Eviction is then
//! `order.pop_first()` followed by `entries.remove(..)`, both `O(log n)`.
//!
//! The counter is `u64` and advances once per touch: at a sustained one billion
//! touches per second it would take over five centuries to wrap, so wrap-around
//! is handled with `saturating_add` rather than with a renumbering pass.
//!
//! # Two bounds
//!
//! Entry count alone does not bound memory when the values differ wildly in
//! size — a raster tile may be anywhere from 1x1 to 8192x8192 texels, a factor
//! of 67 million. A cache built with [`TileCache::with_byte_budget`] therefore
//! also carries a weigher and a byte ceiling, and eviction runs until *both*
//! bounds hold. The weigher is a plain `fn` pointer rather than a trait bound
//! so that `TileCache<V>` stays usable for any `V` (the shells cache decoded
//! pixels, meshes and failure counters with the same type).
//!
//! The byte ceiling is a *soft* bound checked at insert: one value larger than
//! the whole budget is still stored, because refusing it would leave the caller
//! with a tile it can neither draw nor re-request. [`CacheStats::bytes`] is the
//! observable, so a frame that overruns its budget shows up as a rising
//! eviction count rather than as invisible thrash.

use std::collections::{BTreeMap, HashMap};

use crate::error::RenderError;
use crate::mercator::TileId;

/// Hit/miss/eviction counters for a [`TileCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheStats {
    /// Lookups that found a live entry.
    pub hits: u64,
    /// Lookups that found nothing.
    pub misses: u64,
    /// Entries dropped to stay within capacity.
    pub evictions: u64,
    /// Entries currently held.
    pub len: usize,
    /// Maximum number of entries the cache will hold.
    pub capacity: usize,
    /// Summed weight of the live entries, `0` without a weigher.
    pub bytes: usize,
    /// Byte ceiling eviction also honours, [`None`] when only the entry count
    /// is bounded.
    pub max_bytes: Option<usize>,
}

#[derive(Debug)]
struct Entry<V> {
    value: V,
    stamp: u64,
    bytes: usize,
}

/// A least-recently-used cache of per-tile values.
///
/// `V` is deliberately generic: the renderer caches GPU textures, while tests
/// and tools can cache decoded pixels or raw bytes with the same policy.
#[derive(Debug)]
pub struct TileCache<V> {
    entries: HashMap<TileId, Entry<V>>,
    order: BTreeMap<u64, TileId>,
    capacity: usize,
    max_bytes: Option<usize>,
    weigh: Option<fn(&V) -> usize>,
    bytes: usize,
    next_stamp: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl<V> TileCache<V> {
    /// Creates a cache holding at most `capacity` entries.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if `capacity` is zero, which
    /// would make every insert an immediate eviction.
    pub fn new(capacity: usize) -> Result<Self, RenderError> {
        if capacity == 0 {
            return Err(RenderError::InvalidCapacity(capacity));
        }
        Ok(Self {
            entries: HashMap::with_capacity(capacity.min(1024)),
            order: BTreeMap::new(),
            capacity,
            max_bytes: None,
            weigh: None,
            bytes: 0,
            next_stamp: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        })
    }

    /// Creates a cache bounded by entry count *and* by summed weight.
    ///
    /// `weigh` is charged once per stored value, at insert; it must therefore
    /// be a pure function of the value, or the accounting drifts from what the
    /// cache actually holds.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if `capacity` or `max_bytes` is
    /// zero.
    pub fn with_byte_budget(
        capacity: usize,
        max_bytes: usize,
        weigh: fn(&V) -> usize,
    ) -> Result<Self, RenderError> {
        if max_bytes == 0 {
            return Err(RenderError::InvalidCapacity(max_bytes));
        }
        let mut cache = Self::new(capacity)?;
        cache.max_bytes = Some(max_bytes);
        cache.weigh = Some(weigh);
        Ok(cache)
    }

    /// Maximum number of entries this cache will hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Raises or lowers the entry ceiling, evicting down to it immediately.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if `capacity` is zero.
    pub fn set_capacity(&mut self, capacity: usize) -> Result<(), RenderError> {
        if capacity == 0 {
            return Err(RenderError::InvalidCapacity(capacity));
        }
        self.capacity = capacity;
        while self.entries.len() > self.capacity {
            if self.evict_lru().is_none() {
                break;
            }
        }
        Ok(())
    }

    /// Summed weight of the live entries; `0` unless the cache was built with
    /// [`TileCache::with_byte_budget`].
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The byte ceiling, if any.
    #[must_use]
    pub fn max_bytes(&self) -> Option<usize> {
        self.max_bytes
    }

    /// Replaces the byte ceiling, evicting down to it immediately.
    ///
    /// Has no effect on a cache built without a weigher — nothing is being
    /// measured, so nothing can be bounded.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if `max_bytes` is `Some(0)`.
    pub fn set_max_bytes(&mut self, max_bytes: Option<usize>) -> Result<(), RenderError> {
        if max_bytes == Some(0) {
            return Err(RenderError::InvalidCapacity(0));
        }
        self.max_bytes = max_bytes;
        self.evict_to_byte_budget();
        Ok(())
    }

    /// Number of entries currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `tile` is cached. Does **not** count as a use.
    #[must_use]
    pub fn contains(&self, tile: &TileId) -> bool {
        self.entries.contains_key(tile)
    }

    /// Looks a tile up, marking it as most-recently-used and updating the
    /// hit/miss counters.
    pub fn get(&mut self, tile: &TileId) -> Option<&V> {
        let stamp = self.take_stamp();
        match self.entries.get_mut(tile) {
            Some(entry) => {
                self.order.remove(&entry.stamp);
                entry.stamp = stamp;
                self.order.insert(stamp, *tile);
                self.hits += 1;
                Some(&entry.value)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Looks a tile up without touching the recency order or the counters.
    #[must_use]
    pub fn peek(&self, tile: &TileId) -> Option<&V> {
        self.entries.get(tile).map(|entry| &entry.value)
    }

    /// Inserts a value, marking it as most-recently-used.
    ///
    /// If `tile` was already cached its previous value is returned and nothing
    /// is evicted. Otherwise the least-recently-used entry is dropped whenever
    /// the cache is at capacity.
    pub fn insert(&mut self, tile: TileId, value: V) -> Option<V> {
        let stamp = self.take_stamp();
        let bytes = self.weigh.map_or(0, |weigh| weigh(&value));
        if let Some(entry) = self.entries.get_mut(&tile) {
            self.order.remove(&entry.stamp);
            entry.stamp = stamp;
            self.order.insert(stamp, tile);
            self.bytes = self.bytes.saturating_sub(entry.bytes).saturating_add(bytes);
            entry.bytes = bytes;
            let previous = std::mem::replace(&mut entry.value, value);
            self.evict_to_byte_budget();
            return Some(previous);
        }
        while self.entries.len() >= self.capacity {
            if self.evict_lru().is_none() {
                break;
            }
        }
        self.entries.insert(
            tile,
            Entry {
                value,
                stamp,
                bytes,
            },
        );
        self.order.insert(stamp, tile);
        self.bytes = self.bytes.saturating_add(bytes);
        self.evict_to_byte_budget();
        None
    }

    /// Removes a tile, returning its value if it was cached.
    pub fn remove(&mut self, tile: &TileId) -> Option<V> {
        let entry = self.entries.remove(tile)?;
        self.order.remove(&entry.stamp);
        self.bytes = self.bytes.saturating_sub(entry.bytes);
        Some(entry.value)
    }

    /// Drops the least-recently-used entry, returning it.
    ///
    /// Counts towards [`CacheStats::evictions`].
    pub fn evict_lru(&mut self) -> Option<(TileId, V)> {
        let (_, tile) = self.order.pop_first()?;
        let entry = self.entries.remove(&tile)?;
        self.bytes = self.bytes.saturating_sub(entry.bytes);
        self.evictions += 1;
        Some((tile, entry.value))
    }

    /// Removes every entry, keeping the capacity and the statistics.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.bytes = 0;
    }

    /// Evicts until the byte ceiling holds, keeping the last entry whatever it
    /// weighs — a value bigger than the whole budget is stored rather than
    /// dropped on the spot, which would leave the caller unable to draw it and
    /// unable to tell it had arrived.
    fn evict_to_byte_budget(&mut self) {
        let Some(max_bytes) = self.max_bytes else {
            return;
        };
        while self.bytes > max_bytes && self.entries.len() > 1 {
            if self.evict_lru().is_none() {
                break;
            }
        }
    }

    /// Cached tiles from least- to most-recently-used.
    ///
    /// Primarily a debugging and testing aid; the order is exactly the one
    /// eviction follows.
    #[must_use]
    pub fn lru_order(&self) -> Vec<TileId> {
        self.order.values().copied().collect()
    }

    /// Current counters.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            len: self.entries.len(),
            capacity: self.capacity,
            bytes: self.bytes,
            max_bytes: self.max_bytes,
        }
    }

    fn take_stamp(&mut self) -> u64 {
        let stamp = self.next_stamp;
        self.next_stamp = self.next_stamp.saturating_add(1);
        stamp
    }
}

#[cfg(test)]
mod tests {
    use super::TileCache;
    use crate::error::RenderError;
    use crate::mercator::TileId;

    fn tile(x: u32) -> TileId {
        match TileId::new(4, x, 0) {
            Ok(tile) => tile,
            Err(err) => panic!("tile construction failed: {err}"),
        }
    }

    fn cache(capacity: usize) -> TileCache<u32> {
        match TileCache::new(capacity) {
            Ok(cache) => cache,
            Err(err) => panic!("cache construction failed: {err}"),
        }
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(matches!(
            TileCache::<u32>::new(0),
            Err(RenderError::InvalidCapacity(0))
        ));
    }

    #[test]
    fn insert_and_get() {
        let mut cache = cache(4);
        assert!(cache.is_empty());
        assert_eq!(cache.insert(tile(1), 11), None);
        assert_eq!(cache.insert(tile(2), 22), None);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&tile(1)), Some(&11));
        assert_eq!(cache.get(&tile(9)), None);
        assert!(cache.contains(&tile(2)));
        assert_eq!(cache.peek(&tile(2)), Some(&22));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.len, 2);
        assert_eq!(stats.capacity, 4);
    }

    #[test]
    fn reinsert_replaces_without_evicting() {
        let mut cache = cache(2);
        cache.insert(tile(1), 11);
        cache.insert(tile(2), 22);
        assert_eq!(cache.insert(tile(1), 111), Some(11));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.peek(&tile(1)), Some(&111));
        // Re-inserting also refreshes recency, so tile 2 is now the LRU.
        assert_eq!(cache.lru_order(), vec![tile(2), tile(1)]);
    }

    #[test]
    fn eviction_follows_insertion_order() {
        let mut cache = cache(3);
        for x in 1..=3 {
            cache.insert(tile(x), x * 10);
        }
        assert_eq!(cache.lru_order(), vec![tile(1), tile(2), tile(3)]);

        cache.insert(tile(4), 40);
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains(&tile(1)), "oldest entry must be evicted");
        assert_eq!(cache.lru_order(), vec![tile(2), tile(3), tile(4)]);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn re_touch_changes_eviction_victim() {
        let mut cache = cache(3);
        for x in 1..=3 {
            cache.insert(tile(x), x * 10);
        }
        // Touch the oldest entry: it must survive the next insert, and the
        // now-oldest tile 2 must go instead.
        assert_eq!(cache.get(&tile(1)), Some(&10));
        assert_eq!(cache.lru_order(), vec![tile(2), tile(3), tile(1)]);

        cache.insert(tile(4), 40);
        assert!(cache.contains(&tile(1)), "touched entry must survive");
        assert!(!cache.contains(&tile(2)), "new LRU victim must be evicted");
        assert_eq!(cache.lru_order(), vec![tile(3), tile(1), tile(4)]);
    }

    #[test]
    fn peek_does_not_touch() {
        let mut cache = cache(2);
        cache.insert(tile(1), 10);
        cache.insert(tile(2), 20);
        assert_eq!(cache.peek(&tile(1)), Some(&10));
        assert_eq!(cache.lru_order(), vec![tile(1), tile(2)]);
        assert_eq!(cache.stats().hits, 0, "peek must not count as a hit");

        cache.insert(tile(3), 30);
        assert!(!cache.contains(&tile(1)), "peek must not rescue the LRU");
    }

    #[test]
    fn explicit_removal_and_eviction() {
        let mut cache = cache(4);
        for x in 1..=3 {
            cache.insert(tile(x), x);
        }
        assert_eq!(cache.remove(&tile(2)), Some(2));
        assert_eq!(cache.remove(&tile(2)), None);
        assert_eq!(cache.lru_order(), vec![tile(1), tile(3)]);
        assert_eq!(cache.stats().evictions, 0, "remove is not an eviction");

        assert_eq!(cache.evict_lru(), Some((tile(1), 1)));
        assert_eq!(cache.stats().evictions, 1);
        cache.clear();
        assert!(cache.is_empty());
        assert!(cache.lru_order().is_empty());
        assert_eq!(cache.evict_lru(), None);
    }

    #[test]
    fn capacity_one_keeps_only_the_newest() {
        let mut cache = cache(1);
        cache.insert(tile(1), 1);
        cache.insert(tile(2), 2);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.peek(&tile(2)), Some(&2));
        assert!(!cache.contains(&tile(1)));
    }

    #[test]
    fn the_byte_budget_evicts_before_the_entry_count_does() {
        // Weight is the value itself, so the arithmetic is readable.
        let Ok(mut cache) = TileCache::<usize>::with_byte_budget(64, 100, |value| *value) else {
            panic!("cache construction failed");
        };
        assert_eq!(cache.max_bytes(), Some(100));

        cache.insert(tile(1), 40);
        cache.insert(tile(2), 40);
        assert_eq!(cache.bytes(), 80);
        assert_eq!(cache.stats().evictions, 0, "entry count is nowhere near 64");

        // 120 > 100: the LRU goes even though only three of 64 slots are used.
        cache.insert(tile(3), 40);
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains(&tile(1)));
        assert_eq!(cache.bytes(), 80);
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.stats().bytes, 80);
        assert_eq!(cache.stats().max_bytes, Some(100));

        // Re-inserting the same key re-weighs it rather than double-counting.
        cache.insert(tile(2), 10);
        assert_eq!(cache.bytes(), 50);
        assert_eq!(cache.len(), 2);

        // Removal and clearing give the bytes back.
        assert_eq!(cache.remove(&tile(2)), Some(10));
        assert_eq!(cache.bytes(), 40);
        cache.clear();
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn one_oversized_value_is_kept_rather_than_dropped_on_arrival() {
        let Ok(mut cache) = TileCache::<usize>::with_byte_budget(8, 100, |value| *value) else {
            panic!("cache construction failed");
        };
        cache.insert(tile(1), 10);
        cache.insert(tile(2), 500);
        // Everything else went, but the value the caller just handed over is
        // still there — dropping it would leave a tile that can be neither
        // drawn nor re-requested.
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.peek(&tile(2)), Some(&500));
        assert_eq!(cache.bytes(), 500);
    }

    #[test]
    fn a_cache_without_a_weigher_counts_no_bytes() {
        let mut cache = cache(4);
        cache.insert(tile(1), 11);
        assert_eq!(cache.bytes(), 0);
        assert_eq!(cache.max_bytes(), None);
        // Setting a ceiling on an unweighed cache is inert, not a trap.
        assert!(cache.set_max_bytes(Some(1)).is_ok());
        assert_eq!(cache.len(), 1);
        assert!(matches!(
            cache.set_max_bytes(Some(0)),
            Err(RenderError::InvalidCapacity(0))
        ));
    }

    #[test]
    fn budgets_and_capacities_are_validated_and_adjustable() {
        assert!(matches!(
            TileCache::<usize>::with_byte_budget(4, 0, |value| *value),
            Err(RenderError::InvalidCapacity(0))
        ));
        assert!(matches!(
            TileCache::<usize>::with_byte_budget(0, 4, |value| *value),
            Err(RenderError::InvalidCapacity(0))
        ));

        let mut cache = cache(2);
        for x in 1..=2 {
            cache.insert(tile(x), x);
        }
        // Growing keeps everything and stops the next insert evicting.
        let Ok(()) = cache.set_capacity(4) else {
            panic!("growing the cache failed");
        };
        cache.insert(tile(3), 3);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 4);

        // Shrinking evicts down to the new ceiling immediately, oldest first.
        let Ok(()) = cache.set_capacity(1) else {
            panic!("shrinking the cache failed");
        };
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.lru_order(), vec![tile(3)]);
        assert!(matches!(
            cache.set_capacity(0),
            Err(RenderError::InvalidCapacity(0))
        ));
    }

    #[test]
    fn lowering_the_budget_evicts_immediately() {
        let Ok(mut cache) = TileCache::<usize>::with_byte_budget(8, 1_000, |value| *value) else {
            panic!("cache construction failed");
        };
        for x in 1..=4 {
            cache.insert(tile(x), 100);
        }
        assert_eq!(cache.bytes(), 400);
        let Ok(()) = cache.set_max_bytes(Some(150)) else {
            panic!("lowering the budget failed");
        };
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes(), 100);
        assert_eq!(cache.lru_order(), vec![tile(4)], "the newest survives");

        let Ok(()) = cache.set_max_bytes(None) else {
            panic!("clearing the budget failed");
        };
        for x in 5..=8 {
            cache.insert(tile(x), 1_000_000);
        }
        assert_eq!(
            cache.len(),
            5,
            "without a ceiling only the entry count binds"
        );
    }

    #[test]
    fn order_map_never_outgrows_the_entry_map() {
        let mut cache = cache(8);
        for round in 0..4 {
            for x in 0..16 {
                cache.insert(tile(x), round * 100 + x);
                let _ = cache.get(&tile(x / 2));
            }
            assert_eq!(cache.len(), 8);
            assert_eq!(cache.lru_order().len(), cache.len());
        }
    }
}
