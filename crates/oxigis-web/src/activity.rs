//! How many network fetches this tab has outstanding right now.
//!
//! Not sourced from `oxigis-ui`'s tile providers: `XyzTileProvider::stats`
//! and `CogTileProvider`'s equivalent exist, but by the time a provider is
//! installed it is stored as a `Box<dyn map_gpu::TileProvider>` inside
//! `egui_wgpu`'s callback resources, and the `TileProvider` trait itself
//! exposes only `fn tile(&self, TileId) -> Option<DecodedTile>` — there is no
//! `stats()`/`health()` on the trait object, and no downcast seam back to the
//! concrete type. So this crate counts at its own transports instead, which
//! is where every fetch this shell makes — basemap tiles, COG/archive
//! ranges, and the CJK label fonts — already funnels through one of two
//! functions (`crate::tile_fetch::fetch_bytes_within`,
//! `crate::range_fetch::fetch_range`). One [`track`] call at the top of each
//! is the whole integration; see those modules.
//!
//! # Split from the glue
//!
//! Counting is pure — an integer and a guard — and deliberately **not**
//! `#[cfg(target_arch = "wasm32")]`-gated, so `cargo nextest run -p
//! oxigis-web` exercises it on a host with no browser at all. Turning the
//! count into a repaint request and a visible DOM element is the wasm-only
//! half, and lives with the transports that call [`track`] instead — this
//! module must not depend on anything that only exists on `wasm32` (that
//! would make it uncompilable, and therefore untested, on a host), so it
//! does not so much as know that a page exists.
//!
//! # Why a `thread_local`
//!
//! wasm is single-threaded, so a `thread_local!` here is simply a
//! module-level counter with interior mutability — the same reasoning
//! `crate::export_status` and `crate::font_fetch` already document for their
//! own slots.

use std::cell::Cell;

thread_local! {
    /// Fetches started but not yet settled (delivered or failed).
    static COUNT: Cell<u32> = const { Cell::new(0) };
}

/// One fetch in flight.
///
/// Dropping the guard — on success, on failure, or (should a future ever be
/// cancelled before it completes) on early drop — is the only way the count
/// decrements, so it can never be left incremented by a codepath that forgot
/// to say it was done. `#[must_use]`: a [`track`] call whose guard is
/// immediately dropped (rather than held across the fetch) counts a fetch
/// that finished before it started, which is never the intent.
#[must_use = "the count only reflects an in-flight fetch while this guard is held"]
pub struct Activity(());

impl Drop for Activity {
    fn drop(&mut self) {
        COUNT.with(|count| count.set(count.get().saturating_sub(1)));
    }
}

/// Marks one fetch as starting. Decrement happens when the returned guard
/// drops — hold it across the whole request, including the `.await`.
pub fn track() -> Activity {
    COUNT.with(|count| count.set(count.get().saturating_add(1)));
    Activity(())
}

/// How many fetches are outstanding right now.
///
/// `0` is both "never fetched anything" and "everything has settled" — the
/// same "nothing to report" a loading indicator wants for both.
#[must_use]
pub fn count() -> u32 {
    COUNT.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    // `cargo nextest` runs each test in its own process (unlike
    // `cargo test`'s shared-process thread pool), so each of these gets its
    // own copy of the `thread_local` regardless of run order. That isolation
    // is not what these tests lean on, though: each still asserts a fresh
    // `count() == 0` at its own start rather than assuming one, so a future
    // switch to a thread-sharing harness would fail loudly here instead of
    // making these tests silently order-dependent.

    #[test]
    fn starts_at_zero() {
        assert_eq!(count(), 0);
    }

    #[test]
    fn tracking_increments_and_dropping_decrements() {
        assert_eq!(count(), 0);
        let guard = track();
        assert_eq!(count(), 1);
        drop(guard);
        assert_eq!(count(), 0);
    }

    #[test]
    fn concurrent_fetches_are_counted_independently() {
        assert_eq!(count(), 0);
        let first = track();
        let second = track();
        let third = track();
        assert_eq!(count(), 3);
        drop(second);
        assert_eq!(count(), 2, "dropping one of three leaves two in flight");
        drop(first);
        drop(third);
        assert_eq!(count(), 0);
    }

    #[test]
    fn never_underflows_below_zero() {
        // Not reachable through the public API (there is no way to drop an
        // `Activity` that was never created), but `saturating_sub` is the
        // guard against a future refactor introducing exactly that, so it is
        // asserted directly here rather than only trusted.
        assert_eq!(count(), 0);
        COUNT.with(|count| count.set(0));
        let guard = Activity(());
        drop(guard);
        assert_eq!(count(), 0);
    }
}
