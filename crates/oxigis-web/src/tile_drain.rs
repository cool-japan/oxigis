//! The rule the PDF export collects its tiles by: take each one **out** of the
//! provider the moment it appears, rather than waiting for all of them to be
//! resident at once.
//!
//! # Why waiting cannot work
//!
//! A tile provider's ready cache is an LRU sized for a *screen*:
//! `oxigis_ui::tile_provider::READY_CACHE_TILES` is 64 decoded raster tiles,
//! `oxigis_ui::vector_provider::DECODED_CACHE_TILES` is 128 vector ones. A
//! printed page is not a screen — it is the map re-composed at 144–288 dpi, so
//! A4 at the default resolution needs ~42 tiles but A3 at 288 dpi needs ~250.
//!
//! The export used to poll `required.iter().all(|t| provider.tile(t).is_some())`.
//! Past the cache size that predicate can never become true: fetching tile 65
//! evicts tile 1, and asking about an evicted tile both answers `None` *and*
//! re-enqueues its fetch. So the loop spent its entire budget re-requesting
//! tiles it had already been handed — hundreds of redundant GETs at up to
//! `MAX_INFLIGHT_TILES` concurrency, which is a tile-policy violation as well
//! as a bug — and then composed the page from whichever ~64 happened to be
//! resident at that instant, leaving the rest neutral gray. Every non-default
//! choice in the export dialog produced a visibly wrong document, silently.
//!
//! Draining fixes it structurally rather than by raising a limit: a tile that
//! has been collected is never asked for again, so the collection only grows
//! and the loop terminates as soon as the last tile lands. It is also what
//! makes the shortfall *countable* — the caller knows exactly how many tiles
//! never arrived and can say so.
//!
//! Not `cfg`-gated to `wasm32`: this is pure computation over a lookup
//! closure, and gating it would put its tests where nothing compiles them.
//! The `setTimeout` polling around it stays in the shell.

use std::collections::HashMap;

use oxigis_render::TileId;

/// Tiles collected out of a provider, keyed by address.
///
/// Generic over the payload because both export phases need the same rule with
/// different cargo: decoded raster pixels, and `Arc`-shared vector tiles.
#[derive(Debug)]
pub struct TileDrain<T> {
    /// What has been taken out of the provider so far.
    collected: HashMap<TileId, T>,
}

impl<T> Default for TileDrain<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TileDrain<T> {
    /// An empty drain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            collected: HashMap::new(),
        }
    }

    /// Runs one pass over `required`, collecting whatever `lookup` answers for
    /// tiles not already held, and returns how many are **still** missing.
    ///
    /// `lookup` is the provider's `tile()`/`decoded()`, which is also what
    /// enqueues a fetch — so the first pass primes the whole set and later
    /// passes only chase the stragglers. A tile already held is never looked
    /// up again, which is the whole point; a repeated entry in `required` is
    /// therefore harmless, though it is counted per entry (see
    /// [`Self::absent`]).
    pub fn pass(
        &mut self,
        required: &[TileId],
        lookup: &mut dyn FnMut(TileId) -> Option<T>,
    ) -> usize {
        for tile in required {
            if self.collected.contains_key(tile) {
                continue;
            }
            if let Some(payload) = lookup(*tile) {
                self.collected.insert(*tile, payload);
            }
        }
        self.absent(required)
    }

    /// How many tiles have been collected.
    #[must_use]
    pub fn len(&self) -> usize {
        self.collected.len()
    }

    /// Whether nothing has been collected yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.collected.is_empty()
    }

    /// How many entries of `required` never arrived.
    ///
    /// Per entry rather than per distinct address — which is the same number
    /// for every real page, because `MapView::visible_tiles` sorts and
    /// de-duplicates before it returns.
    #[must_use]
    pub fn absent(&self, required: &[TileId]) -> usize {
        required
            .iter()
            .filter(|tile| !self.collected.contains_key(tile))
            .count()
    }

    /// Hands one tile over, removing it.
    ///
    /// Removing rather than cloning is deliberate: the composition step asks
    /// for each tile exactly once, so ownership frees each tile's pixels as
    /// they are pasted instead of holding the whole page twice. "Exactly
    /// once" is load-bearing and it holds because `compose_map_rgb` walks
    /// `MapView::visible_tiles`, which de-duplicates.
    pub fn take(&mut self, tile: TileId) -> Option<T> {
        self.collected.remove(&tile)
    }

    /// Everything collected, in `required`'s order.
    ///
    /// The page's own tile order rather than a hash map's, so an overlay draws
    /// deterministically across runs.
    #[must_use]
    pub fn into_ordered(mut self, required: &[TileId]) -> Vec<(TileId, T)> {
        required
            .iter()
            .filter_map(|tile| self.collected.remove(tile).map(|payload| (*tile, payload)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::TileDrain;
    use oxigis_render::TileId;

    /// The ready cache the raster provider actually has
    /// (`READY_CACHE_TILES`), and the reason this module exists.
    const CACHE: usize = 64;

    /// The provider's concurrency gate (`MAX_INFLIGHT_TILES`).
    const INFLIGHT: usize = 16;

    fn tile(index: u32) -> TileId {
        TileId {
            z: 10,
            x: index,
            y: 7,
        }
    }

    /// `XyzTileProvider`'s shape, reduced to what the drain rule interacts
    /// with: a fixed-size LRU of ready tiles, a bounded in-flight set that is
    /// also the concurrency gate, and fetches that land **between** polls
    /// rather than inside one.
    struct LruProvider {
        /// Decoded tiles, oldest first.
        ready: Vec<u32>,
        /// Fetches started and not yet landed.
        inflight: Vec<u32>,
        /// Every fetch ever started — the number of GETs a real host would
        /// see.
        requests: usize,
    }

    impl LruProvider {
        fn new() -> Self {
            Self {
                ready: Vec::new(),
                inflight: Vec::new(),
                requests: 0,
            }
        }

        /// One `tile()` call: the pixels if ready (refreshing recency, as the
        /// real cache's `get` does), otherwise a fetch — unless the gate is
        /// full, in which case the tile is simply not asked for yet.
        fn lookup(&mut self, tile: TileId) -> Option<u32> {
            if let Some(at) = self.ready.iter().position(|held| *held == tile.x) {
                let held = self.ready.remove(at);
                self.ready.push(held);
                return Some(held);
            }
            if self.inflight.contains(&tile.x) || self.inflight.len() >= INFLIGHT {
                return None;
            }
            self.inflight.push(tile.x);
            self.requests += 1;
            None
        }

        /// The 100 ms between two polls: everything in flight arrives, and the
        /// cache evicts down to its capacity.
        fn settle(&mut self) {
            let landed: Vec<u32> = self.inflight.drain(..).collect();
            self.ready.extend(landed);
            while self.ready.len() > CACHE {
                self.ready.remove(0);
            }
        }
    }

    #[test]
    fn a_page_larger_than_the_cache_never_becomes_all_resident() {
        // The bug, stated as a test: this is the predicate the export used to
        // poll on, and past the cache size it cannot ever answer yes.
        let required: Vec<TileId> = (0..(CACHE as u32 * 2)).map(tile).collect();
        let mut provider = LruProvider::new();
        for _poll in 0..64 {
            let missing = required
                .iter()
                .filter(|tile| provider.lookup(**tile).is_none())
                .count();
            assert!(
                missing > 0,
                "an LRU smaller than the page can never hold all of it at once",
            );
            provider.settle();
        }
        assert!(
            provider.requests > required.len(),
            "and the price of waiting is re-fetching evicted tiles: {} GETs for {} tiles",
            provider.requests,
            required.len(),
        );
    }

    #[test]
    fn draining_converges_on_a_page_larger_than_the_cache() {
        let required: Vec<TileId> = (0..(CACHE as u32 * 4)).map(tile).collect();
        let mut provider = LruProvider::new();
        let mut drain = TileDrain::new();
        let mut polls = 0usize;
        loop {
            if drain.pass(&required, &mut |tile| provider.lookup(tile)) == 0 {
                break;
            }
            polls += 1;
            assert!(polls < 150, "the drain must converge, not spend its budget");
            provider.settle();
        }
        assert_eq!(drain.len(), required.len());
        assert_eq!(drain.absent(&required), 0);
        // Every tile fetched exactly once: the old loop re-requested each
        // evicted tile on every poll instead.
        assert_eq!(provider.requests, required.len());
    }

    #[test]
    fn a_tile_that_never_arrives_is_counted_not_waited_for() {
        let required: Vec<TileId> = (0..8).map(tile).collect();
        let mut drain = TileDrain::new();
        // A provider answering only even tiles — a permanent failure on the
        // rest, which is what a 404 from a tile host looks like here.
        let missing = drain.pass(&required, &mut |tile| (tile.x % 2 == 0).then_some(tile.x));
        assert_eq!(missing, 4);
        assert_eq!(drain.absent(&required), 4);
        assert_eq!(drain.len(), 4);
    }

    #[test]
    fn take_hands_over_ownership_once() {
        let required: Vec<TileId> = (0..3).map(tile).collect();
        let mut drain = TileDrain::new();
        assert_eq!(drain.pass(&required, &mut |tile| Some(tile.x)), 0);
        assert!(!drain.is_empty());
        assert_eq!(drain.take(tile(1)), Some(1));
        assert_eq!(drain.take(tile(1)), None, "a pasted tile is not held twice");
        assert_eq!(drain.len(), 2);
    }

    #[test]
    fn a_repeated_tile_is_collected_once_and_looked_up_once() {
        // `visible_tiles` de-duplicates, so this cannot arise from a real
        // page; the rule still has to be well defined, and "collect once" is
        // what keeps `take` safe for the composition walk.
        let required = vec![tile(0), tile(1), tile(0), tile(1)];
        let mut drain = TileDrain::new();
        let mut lookups = 0usize;
        assert_eq!(
            drain.pass(&required, &mut |tile| {
                lookups += 1;
                Some(tile.x)
            }),
            0
        );
        assert_eq!(lookups, 2, "a collected tile is never asked for again");
        assert_eq!(drain.len(), 2);
        assert_eq!(drain.absent(&required), 0);
    }

    #[test]
    fn into_ordered_follows_the_pages_tile_order() {
        let required = vec![tile(5), tile(2), tile(9)];
        let mut drain = TileDrain::new();
        assert_eq!(drain.pass(&required, &mut |tile| Some(tile.x)), 0);
        let ordered: Vec<u32> = drain
            .into_ordered(&required)
            .into_iter()
            .map(|(_, payload)| payload)
            .collect();
        assert_eq!(ordered, vec![5, 2, 9]);
    }
}
