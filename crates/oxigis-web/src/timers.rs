//! Browser timers: the promise-backed sleep the async paths need, and the
//! deadline every `fetch()` in this shell is held to.
//!
//! # Why a deadline is not optional here
//!
//! The `fetch` spec puts no timeout on a server that accepts a connection and
//! then stalls, so such a request simply never settles. Both browser transports
//! call `sink.deliver` only when their promise settles, and the shared provider
//! removes a tile from its in-flight set only in `deliver` — there is no sweep.
//! A stalled request therefore parks that `TileId` in `inflight` for the
//! session, and since `inflight` is also the concurrency gate
//! (`MAX_INFLIGHT_TILES`), sixteen of them stop the map requesting anything ever
//! again: a captive portal or a flaky mobile link freezes the basemap with no
//! error anywhere. The native shell has no such exposure — its agent is built
//! with a request timeout.
//!
//! [`settle_within`] closes it without new browser capability: the request's
//! promise is raced against a `setTimeout` promise, and the loser is reported
//! as a *transient* failure so the provider's ordinary retry budget applies.
//!
//! What this deliberately does **not** do is abort the underlying request: an
//! `AbortController` would additionally stop paying for the bytes, and needs
//! `AbortController`/`AbortSignal` in the crate's `web-sys` feature list. The
//! wedge — the tile slot held forever — is fixed either way; the wasted
//! bandwidth is not.

use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

/// How long one tile or range GET may take before it is failed as transient.
///
/// Generous next to a tile CDN's p99 (tens of milliseconds) and next to a cold
/// PMTiles leaf read over a slow link, so a merely slow host keeps working; the
/// point is to bound a request that will never answer at all.
pub const TILE_REQUEST_TIMEOUT_MS: i32 = 20_000;

/// How long a font body may take — the same guard, with room for the megabytes
/// a CJK face weighs on a slow link. Sharing the tile deadline here would start
/// failing 16 MB downloads that were merely slow.
pub const FONT_REQUEST_TIMEOUT_MS: i32 = 120_000;

/// Resolves after `ms` milliseconds, through a `setTimeout`-backed promise
/// — the browser has no blocking sleep, and the tile fetches need the event
/// loop this hands back.
pub async fn sleep_ms(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        if let Some(window) = web_sys::window() {
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
        }
    });
    let _ = JsFuture::from(promise).await;
}

/// Awaits `promise`, giving up after `timeout_ms`.
///
/// `Ok(None)` is the timeout. The trick that makes it unambiguous without a
/// sentinel object: `setTimeout(resolve, ms)` calls `resolve()` with no
/// argument, so the timer arm resolves to `undefined` — which neither `fetch`
/// (a `Response`) nor `Response.arrayBuffer` (an `ArrayBuffer`) ever does. A
/// rejection from `promise` still arrives as [`Err`], unchanged.
///
/// The timer is cleared once the race settles, so a page doing thousands of
/// tile requests does not accumulate pending callbacks.
///
/// # Errors
///
/// The value `promise` rejected with.
pub async fn settle_within(
    promise: js_sys::Promise,
    timeout_ms: i32,
) -> Result<Option<JsValue>, JsValue> {
    let window = web_sys::window();
    let mut handle: Option<i32> = None;
    let timer = js_sys::Promise::new(&mut |resolve, _reject| {
        // Executed synchronously by `Promise::new`, which is what lets the
        // handle escape into the outer scope for the clear below.
        if let Some(window) = window.as_ref()
            && let Ok(id) =
                window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, timeout_ms)
        {
            handle = Some(id);
        }
    });
    let raced = js_sys::Promise::race(&js_sys::Array::of2(&promise, &timer));
    let settled = JsFuture::from(raced).await;
    if let (Some(window), Some(id)) = (window.as_ref(), handle) {
        window.clear_timeout_with_handle(id);
    }
    match settled {
        // A window-less context never armed the timer, so `undefined` there
        // cannot be a timeout; it is not a browser and there is nothing to
        // race anyway.
        Ok(value) if value.is_undefined() && handle.is_some() => Ok(None),
        Ok(value) => Ok(Some(value)),
        Err(error) => Err(error),
    }
}
