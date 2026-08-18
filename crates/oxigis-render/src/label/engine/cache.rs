//! The shaped-label LRU, split from the engine because three separate
//! constraints meet in it: a cache hit must allocate nothing, the capacity
//! must follow the frame's real demand, and every label the engine ever handed
//! out has to stay reachable for as long as a caller can still draw it.
//!
//! # No allocation on a hit
//!
//! The key is `(size, weight, orientation)` plus the exact text. Hashing the
//! text down to an integer is NOT an option — a collision in a glyph cache
//! serves the wrong ink — so the exactness is kept and the allocation is
//! removed structurally instead: entries are bucketed by the small `Copy`
//! [`StyleKey`] and the inner map is keyed by `Arc<str>`, which
//! [`std::borrow::Borrow`] lets a `&str` look up for free. Only a miss
//! allocates. The LRU order carries the same `Arc<str>` handle, and a hit
//! *moves* that pair from one stamp to another rather than cloning it, so a
//! hit is two `BTreeMap` operations and no heap traffic at all.
//!
//! # Adaptive capacity
//!
//! A frame shapes every label CANDIDATE, not just the ones it places, and a
//! pass over the same tile list every frame is a cyclic access pattern: with a
//! working set larger than the cache, strict LRU degenerates to a 100 % miss
//! rate — the next frame asks first for the entry the last frame evicted
//! first. The cache therefore watches its own eviction rate over a window of
//! inserts and doubles its capacity, up to [`MAX_LABEL_CACHE`], when it is
//! evicting on essentially every insert. A working set that fits evicts
//! nothing and never grows.
//!
//! # Held labels
//!
//! Eviction may not orphan a label a caller is still drawing: the glyph atlas
//! evicts per glyph, and it decides what is live by walking this cache. An
//! entry evicted while its [`Arc`] is still shared therefore moves to a side
//! list instead of being dropped, and the list is swept — dropping whatever
//! has fallen back to one reference — before every compaction and whenever it
//! outgrows the cache itself.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::label::engine::{LabelOrientation, LabelWeight, ShapedLabel};

/// Hard ceiling on the adaptive capacity, in cached labels.
///
/// A cached label is its glyph vector plus its key: roughly 600 bytes for a
/// long name, so the ceiling bounds the cache at a few megabytes no matter
/// what a style throws at it.
pub const MAX_LABEL_CACHE: usize = 8192;

/// Fewest inserts a capacity review looks at, so a tiny cache is not resized
/// on a handful of samples.
const MIN_REVIEW_WINDOW: usize = 64;

/// Everything a cache key carries except the text: small, `Copy`, and the
/// bucket the text is looked up in.
///
/// Both the weight and the orientation are the EFFECTIVE ones (a bold request
/// with no bold chain is stored as [`LabelWeight::Regular`]; a vertical
/// request the font-free half of the ladder refuses is stored as
/// [`LabelOrientation::Horizontal`]), so neither splits an entry when the two
/// would draw the same glyphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct StyleKey {
    /// `f32::to_bits` of the pixels-per-em the label was shaped at.
    pub size_bits: u32,
    pub weight: LabelWeight,
    pub orientation: LabelOrientation,
}

/// Cache entry plus its use stamp, as in [`crate::tile_cache::TileCache`].
#[derive(Debug)]
struct CacheEntry {
    label: Arc<ShapedLabel>,
    stamp: u64,
}

/// The shaped-label LRU.
#[derive(Debug)]
pub(super) struct LabelCache {
    entries: HashMap<StyleKey, HashMap<Arc<str>, CacheEntry>>,
    order: BTreeMap<u64, (StyleKey, Arc<str>)>,
    /// Labels evicted while a caller still held one. They are no longer
    /// reachable by key, but their glyphs are still on screen.
    held: Vec<Arc<ShapedLabel>>,
    /// Length at which the held list is swept — doubled after each sweep so
    /// sweeping stays amortised constant per insert.
    held_sweep_at: usize,
    /// Entry count across every bucket, kept rather than summed.
    len: usize,
    capacity: usize,
    ceiling: usize,
    inserts_in_window: usize,
    evictions_in_window: usize,
    next_stamp: u64,
}

impl LabelCache {
    /// A cache holding `capacity` labels, allowed to grow to `ceiling`.
    ///
    /// `ceiling == capacity` pins the size: an explicitly sized cache is a
    /// request, not a hint.
    pub(super) fn new(capacity: usize, ceiling: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            entries: HashMap::with_capacity(capacity.min(1024)),
            order: BTreeMap::new(),
            held: Vec::new(),
            held_sweep_at: MIN_REVIEW_WINDOW,
            len: 0,
            capacity,
            ceiling: ceiling.max(capacity),
            inserts_in_window: 0,
            evictions_in_window: 0,
            next_stamp: 0,
        }
    }

    /// Number of labels currently cached.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Labels the cache holds before it evicts.
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Raises the capacity so a pass over `labels` distinct strings fits with
    /// room for the ring of tiles a pan is about to reveal, bounded by
    /// [`MAX_LABEL_CACHE`].
    pub(super) fn reserve(&mut self, labels: usize) {
        let wanted = labels.saturating_mul(2).min(MAX_LABEL_CACHE);
        self.ceiling = self.ceiling.max(wanted);
        self.capacity = self.capacity.max(wanted);
    }

    /// Looks `text` up in `style`'s bucket, marking it most-recently-used.
    ///
    /// Neither the lookup nor the re-filing allocates.
    pub(super) fn get(&mut self, style: StyleKey, text: &str) -> Option<Arc<ShapedLabel>> {
        let stamp = self.take_stamp();
        let entry = self.entries.get_mut(&style)?.get_mut(text)?;
        let previous = entry.stamp;
        entry.stamp = stamp;
        let label = Arc::clone(&entry.label);
        // Moved, not cloned: the `Arc<str>` handle the order map already owns
        // is exactly the one the new stamp needs.
        if let Some(pair) = self.order.remove(&previous) {
            self.order.insert(stamp, pair);
        }
        Some(label)
    }

    /// Files a freshly shaped label, evicting the least recently used entries
    /// if that puts the cache over capacity.
    pub(super) fn insert(&mut self, style: StyleKey, text: &str, label: &Arc<ShapedLabel>) {
        let stamp = self.take_stamp();
        let text: Arc<str> = Arc::from(text);
        let entry = CacheEntry {
            label: Arc::clone(label),
            stamp,
        };
        let bucket = self.entries.entry(style).or_default();
        match bucket.insert(Arc::clone(&text), entry) {
            // Replacing an entry: its stamp must leave the order map or it
            // would evict a live key later.
            Some(previous) => {
                self.order.remove(&previous.stamp);
            }
            None => self.len += 1,
        }
        self.order.insert(stamp, (style, text));
        self.inserts_in_window += 1;
        self.evict_to_capacity();
        self.review_capacity();
    }

    /// Tracks a label that was handed out WITHOUT being cached, so the atlas
    /// still counts its glyphs as live.
    pub(super) fn hold(&mut self, label: &Arc<ShapedLabel>) {
        self.hold_owned(Arc::clone(label));
    }

    fn hold_owned(&mut self, label: Arc<ShapedLabel>) {
        self.held.push(label);
        if self.held.len() >= self.held_sweep_at {
            self.sweep_held();
        }
    }

    /// Drops every held label no caller references any more.
    pub(super) fn sweep_held(&mut self) {
        self.held.retain(|label| Arc::strong_count(label) > 1);
        self.held_sweep_at = self
            .held
            .len()
            .saturating_mul(2)
            .clamp(MIN_REVIEW_WINDOW, MAX_LABEL_CACHE);
    }

    /// Every label that may still be drawn: the cached ones and the held ones.
    pub(super) fn live_labels(&self) -> impl Iterator<Item = &Arc<ShapedLabel>> {
        self.entries
            .values()
            .flat_map(HashMap::values)
            .map(|entry| &entry.label)
            .chain(self.held.iter())
    }

    /// Drops every cached entry `keep` rejects. Held labels are untouched:
    /// they are exactly the ones a caller can still draw.
    pub(super) fn retain(&mut self, mut keep: impl FnMut(&ShapedLabel) -> bool) {
        let mut dropped: Vec<u64> = Vec::new();
        for bucket in self.entries.values_mut() {
            bucket.retain(|_, entry| {
                if keep(&entry.label) {
                    return true;
                }
                dropped.push(entry.stamp);
                false
            });
        }
        if dropped.is_empty() {
            return;
        }
        self.entries.retain(|_, bucket| !bucket.is_empty());
        for stamp in &dropped {
            self.order.remove(stamp);
        }
        self.len = self.len.saturating_sub(dropped.len());
    }

    /// Drops every entry and every held label. The learned capacity survives:
    /// the next frame's demand is the same frame's demand.
    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.held.clear();
        self.held_sweep_at = MIN_REVIEW_WINDOW;
        self.len = 0;
        self.inserts_in_window = 0;
        self.evictions_in_window = 0;
    }

    /// A monotonically increasing use stamp.
    fn take_stamp(&mut self) -> u64 {
        let stamp = self.next_stamp;
        self.next_stamp = self.next_stamp.saturating_add(1);
        stamp
    }

    fn evict_to_capacity(&mut self) {
        while self.len > self.capacity {
            let Some((_, (style, text))) = self.order.pop_first() else {
                break;
            };
            let Some(bucket) = self.entries.get_mut(&style) else {
                continue;
            };
            let evicted = bucket.remove(&text);
            if bucket.is_empty() {
                self.entries.remove(&style);
            }
            let Some(evicted) = evicted else {
                continue;
            };
            self.len = self.len.saturating_sub(1);
            self.evictions_in_window += 1;
            // Still shared: a caller is drawing it, so the glyphs it points at
            // must stay packed even though the key is gone.
            if Arc::strong_count(&evicted.label) > 1 {
                self.hold_owned(evicted.label);
            }
        }
    }

    /// Doubles the capacity when the last window of inserts was evicting on
    /// essentially every insert — the signature of a working set larger than
    /// the cache, which strict LRU turns into a 100 % miss rate.
    fn review_capacity(&mut self) {
        let window = self.capacity.max(MIN_REVIEW_WINDOW);
        if self.inserts_in_window < window {
            return;
        }
        if self.evictions_in_window * 2 >= self.inserts_in_window {
            self.capacity = self.capacity.saturating_mul(2).min(self.ceiling);
        }
        self.inserts_in_window = 0;
        self.evictions_in_window = 0;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{LabelCache, MAX_LABEL_CACHE, StyleKey};
    use crate::label::engine::{LabelOrientation, LabelWeight, ShapedLabel};

    fn style() -> StyleKey {
        StyleKey {
            size_bits: 14.0_f32.to_bits(),
            weight: LabelWeight::Regular,
            orientation: LabelOrientation::Horizontal,
        }
    }

    fn label(width: f32) -> Arc<ShapedLabel> {
        Arc::new(ShapedLabel {
            size_px: [width, 10.0],
            font_size_px: 14.0,
            generation: 0,
            glyphs: Vec::new(),
        })
    }

    #[test]
    fn a_hit_returns_the_same_arc_and_a_miss_is_none() {
        let mut cache = LabelCache::new(4, 4);
        assert!(cache.get(style(), "Kyoto").is_none());
        let stored = label(1.0);
        cache.insert(style(), "Kyoto", &stored);
        let hit = cache.get(style(), "Kyoto").expect("just inserted");
        assert!(Arc::ptr_eq(&stored, &hit));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn the_style_is_part_of_the_key() {
        let mut cache = LabelCache::new(4, 4);
        cache.insert(style(), "Kyoto", &label(1.0));
        let bold = StyleKey {
            weight: LabelWeight::Bold,
            ..style()
        };
        assert!(cache.get(bold, "Kyoto").is_none(), "weight splits the key");
        let vertical = StyleKey {
            orientation: LabelOrientation::Vertical,
            ..style()
        };
        assert!(cache.get(vertical, "Kyoto").is_none());
        let bigger = StyleKey {
            size_bits: 15.0_f32.to_bits(),
            ..style()
        };
        assert!(cache.get(bigger, "Kyoto").is_none());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn the_least_recently_used_entry_is_the_one_that_goes() {
        let mut cache = LabelCache::new(3, 3);
        for text in ["a", "b", "c"] {
            cache.insert(style(), text, &label(1.0));
        }
        // Re-touch "a" so "b" becomes the victim.
        assert!(cache.get(style(), "a").is_some());
        cache.insert(style(), "d", &label(1.0));
        assert_eq!(cache.len(), 3);
        assert!(cache.get(style(), "b").is_none(), "b was oldest");
        assert!(cache.get(style(), "a").is_some());
        assert!(cache.get(style(), "c").is_some());
        assert!(cache.get(style(), "d").is_some());
    }

    #[test]
    fn re_inserting_a_key_replaces_it_without_leaking_an_order_entry() {
        let mut cache = LabelCache::new(2, 2);
        cache.insert(style(), "a", &label(1.0));
        let second = label(2.0);
        cache.insert(style(), "a", &second);
        assert_eq!(cache.len(), 1, "one key, one entry");
        cache.insert(style(), "b", &label(3.0));
        cache.insert(style(), "c", &label(4.0));
        // "a" is the oldest and must be the only casualty; a leaked order
        // entry would have evicted "b" as well.
        assert!(cache.get(style(), "a").is_none());
        assert!(cache.get(style(), "b").is_some());
        assert!(cache.get(style(), "c").is_some());
    }

    #[test]
    fn a_cyclic_working_set_grows_the_capacity_up_to_the_ceiling() {
        let mut cache = LabelCache::new(64, 512);
        // Two passes over 200 distinct strings: exactly the placer's per-frame
        // pattern over a tile list that does not fit.
        for _ in 0..4 {
            for index in 0..200 {
                let text = format!("label {index}");
                if cache.get(style(), &text).is_none() {
                    cache.insert(style(), &text, &label(1.0));
                }
            }
        }
        assert!(
            cache.capacity() > 64,
            "a thrashing cache must grow, got {}",
            cache.capacity()
        );
        assert!(cache.capacity() <= 512, "and never past its ceiling");
    }

    #[test]
    fn a_working_set_that_fits_never_grows() {
        let mut cache = LabelCache::new(64, 512);
        for _ in 0..20 {
            for index in 0..32 {
                let text = format!("label {index}");
                if cache.get(style(), &text).is_none() {
                    cache.insert(style(), &text, &label(1.0));
                }
            }
        }
        assert_eq!(cache.capacity(), 64, "no evictions, no growth");
        assert_eq!(cache.len(), 32);
    }

    #[test]
    fn a_pinned_ceiling_is_honoured() {
        let mut cache = LabelCache::new(4, 4);
        for index in 0..64 {
            cache.insert(style(), &format!("label {index}"), &label(1.0));
        }
        assert_eq!(cache.capacity(), 4, "an explicit capacity is a request");
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn reserving_raises_the_capacity_and_stays_bounded() {
        let mut cache = LabelCache::new(4, 4);
        cache.reserve(100);
        assert_eq!(cache.capacity(), 200);
        cache.reserve(usize::MAX);
        assert_eq!(cache.capacity(), MAX_LABEL_CACHE, "bounded, not overflowed");
    }

    #[test]
    fn an_evicted_label_a_caller_still_holds_stays_live() {
        let mut cache = LabelCache::new(1, 1);
        let drawn = label(1.0);
        cache.insert(style(), "drawn", &drawn);
        let dropped = label(2.0);
        cache.insert(style(), "dropped", &dropped);
        drop(dropped);
        cache.insert(style(), "third", &label(3.0));

        // "drawn" left the keyed cache but is still referenced, so it must
        // still be reported live — the atlas frees glyphs by exactly this set.
        assert!(cache.get(style(), "drawn").is_none());
        let live: Vec<*const ShapedLabel> = cache.live_labels().map(Arc::as_ptr).collect();
        assert!(live.contains(&Arc::as_ptr(&drawn)), "held labels are live");
        assert_eq!(live.len(), 2, "the unreferenced one is gone: {live:?}");
    }

    #[test]
    fn sweeping_drops_held_labels_no_one_draws_any_more() {
        let mut cache = LabelCache::new(1, 1);
        let drawn = label(1.0);
        cache.insert(style(), "drawn", &drawn);
        cache.insert(style(), "next", &label(2.0));
        assert_eq!(cache.live_labels().count(), 2);
        drop(drawn);
        cache.sweep_held();
        assert_eq!(cache.live_labels().count(), 1);
    }

    #[test]
    fn retain_drops_entries_and_their_order_stamps() {
        let mut cache = LabelCache::new(4, 4);
        cache.insert(style(), "keep", &label(1.0));
        cache.insert(style(), "drop", &label(2.0));
        cache.retain(|label| label.width_px() < 1.5);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(style(), "keep").is_some());
        assert!(cache.get(style(), "drop").is_none());
        // The dropped entry's stamp must be gone too, or it would evict the
        // survivor when the cache next fills.
        for index in 0..3 {
            cache.insert(style(), &format!("extra {index}"), &label(1.0));
        }
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn clearing_empties_everything_but_the_learned_capacity() {
        let mut cache = LabelCache::new(64, 512);
        cache.reserve(200);
        let held = label(1.0);
        cache.insert(style(), "a", &held);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.live_labels().count(), 0);
        assert_eq!(cache.capacity(), 400);
        assert!(cache.get(style(), "a").is_none());
    }
}
