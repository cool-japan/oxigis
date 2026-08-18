//! Browser tile transport: the `fetch()` API behind `oxigis-ui`'s
//! [`oxigis_ui::TileTransport`] seam.
//!
//! The shared provider ([`oxigis_ui::XyzTileProvider`]) owns the cache, the
//! in-flight set and the retry policy; this module only turns a URL into bytes.
//! Compiled for `wasm32` only — the native shell uses `oxigis-desktop`'s
//! blocking HTTP pool instead.
//!
//! # Why raw `web-sys` and not an HTTP crate
//!
//! In a browser there is no socket to open: the only transport is `fetch()`,
//! which `web-sys` already exposes and which `eframe` has pulled into the graph
//! regardless. Four extra `web-sys` features (`Request`, `RequestInit`,
//! `RequestMode`, `Response`) plus `js_sys`' typed arrays are the whole cost;
//! a wrapper crate such as `gloo-net` would add a dependency to hide about
//! fifteen lines.
//!
//! # Threading and the `Send + Sync` bound
//!
//! [`oxigis_ui::TileTransport`] requires `Send + Sync` on every target, because
//! the provider holding it lives in `egui_wgpu`'s concurrent callback-resource
//! map (see `oxigis_ui::map_gpu`). [`FetchTileTransport`] is a unit struct that
//! holds no JS value, so the bound is satisfied trivially: every `JsValue` is
//! created *inside* the `spawn_local` future and never crosses a thread. The
//! future itself only needs `'static`, which is why
//! [`oxigis_ui::TileTransport::request`] hands over an owned `String` and an
//! owned [`oxigis_ui::TileSink`].
//!
//! # Headers
//!
//! Deliberately none. Browsers set `User-Agent` and `Referer` themselves and
//! forbid overriding them, and adding any non-safelisted header would turn a
//! simple cross-origin GET into a CORS preflight that most tile CDNs answer
//! with a 403.
//!
//! # Bounds
//!
//! Every GET carries a deadline ([`crate::timers::settle_within`], for the
//! in-flight slot a stalled server would otherwise hold forever) and a body
//! ceiling. The ceiling is checked against `Content-Length` *before* the body is
//! read where the header is there to check — it is CORS-safelisted, so a
//! cross-origin host exposes it without configuring anything — and against the
//! real length afterwards, since a chunked response declares no length at all.

use oxigis_render::TileId;
use oxigis_ui::{TileError, TileSink, TileTransport};
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::spawn_local;

use crate::timers::{TILE_REQUEST_TIMEOUT_MS, settle_within};

/// HTTP status at and above which a failure is treated as retryable.
const FIRST_SERVER_ERROR_STATUS: u16 = 500;

/// Largest tile body accepted, in bytes.
///
/// A 512x512 PNG tile is a few hundred KB and a dense JPEG orthophoto tile a
/// couple of MB, so this is far above anything a tile service serves; it exists
/// so an untrusted or misconfigured host cannot make the tab allocate an
/// arbitrary buffer for something the decoder would reject anyway.
pub const MAX_TILE_BYTES: usize = 16 * 1024 * 1024;

/// [`TileTransport`] backed by the browser's `fetch()` API.
///
/// Stateless: the browser owns the connection pool, the HTTP cache and the
/// per-host request scheduling, so there is nothing here to configure and
/// nothing to keep alive between requests.
#[derive(Debug, Clone, Copy, Default)]
pub struct FetchTileTransport;

impl FetchTileTransport {
    /// Creates the transport.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl TileTransport for FetchTileTransport {
    fn request(&self, tile: TileId, url: String, sink: TileSink) {
        // Returns immediately: the future is queued on the browser's microtask
        // queue, so the `egui_wgpu` prepare hook that called us is never blocked.
        spawn_local(async move {
            let result = fetch_bytes(&url).await;
            sink.deliver(tile, result);
        });
    }
}

/// Renders a JS exception as a message for [`TileError`].
fn js_message(value: &JsValue) -> String {
    value
        .as_string()
        .or_else(|| {
            value
                .dyn_ref::<js_sys::Error>()
                .map(|e| e.to_string().into())
        })
        .unwrap_or_else(|| format!("{value:?}"))
}

/// `GET`s `url` as a tile: the tile deadline and [`MAX_TILE_BYTES`].
pub(crate) async fn fetch_bytes(url: &str) -> Result<Vec<u8>, TileError> {
    fetch_bytes_within(url, TILE_REQUEST_TIMEOUT_MS, MAX_TILE_BYTES).await
}

/// `GET`s `url` and returns the response body, giving up after `timeout_ms` and
/// refusing a body over `max_bytes`.
///
/// Classification matches the native transport: HTTP 5xx and network-level
/// failures (which is also what a CORS rejection looks like from JS) are
/// retryable; every 4xx is permanent for the session. A timeout is retryable
/// too, so the provider's ordinary backoff applies to a host that stalls.
///
/// Shared with [`crate::font_fetch`], which needs exactly this one-shot GET with
/// its own (much longer) deadline and its own ceiling, and would otherwise
/// duplicate the `RequestInit`/`Response`/`ArrayBuffer` dance.
pub(crate) async fn fetch_bytes_within(
    url: &str,
    timeout_ms: i32,
    max_bytes: usize,
) -> Result<Vec<u8>, TileError> {
    // In-flight accounting for the page's loading indicator (`crate::activity`)
    // plus the repaint that makes a count change actually visible while the
    // map is otherwise idle (`crate::font_fetch::RepaintOnDrop`). Both are
    // guards held to the end of this function's scope, across every `?`
    // below, so every return path — success, a bad-URL refusal, a timeout, an
    // oversized body — accounts for itself with no separate cleanup.
    let _activity = crate::activity::track();
    let _repaint = crate::font_fetch::RepaintOnDrop::new();

    let window = web_sys::window()
        .ok_or_else(|| TileError::permanent("no `window`: not a browser context"))?;

    let options = web_sys::RequestInit::new();
    options.set_method("GET");
    // Tile CDNs serve `Access-Control-Allow-Origin: *`; an opaque `no-cors`
    // response would not let us read the bytes at all.
    options.set_mode(web_sys::RequestMode::Cors);

    let request = web_sys::Request::new_with_str_and_init(url, &options)
        .map_err(|error| TileError::permanent(format!("bad tile URL: {}", js_message(&error))))?;

    let response = settle_within(window.fetch_with_request(&request), timeout_ms)
        .await
        .map_err(|error| TileError::transient(format!("fetch failed: {}", js_message(&error))))?
        .ok_or_else(|| TileError::transient(format!("no answer within {timeout_ms} ms")))?
        .dyn_into::<web_sys::Response>()
        .map_err(|_| TileError::permanent("fetch did not resolve to a Response"))?;

    if !response.ok() {
        let status = response.status();
        let message = format!("HTTP {status}");
        return Err(if status >= FIRST_SERVER_ERROR_STATUS {
            TileError::transient(message)
        } else {
            TileError::permanent(message)
        });
    }

    // `Content-Length` is CORS-safelisted, so this is readable cross-origin
    // without the host exposing anything; an absent one (a chunked response)
    // just means the ceiling is enforced after the read instead.
    if let Some(declared) = content_length(&response)
        && declared > max_bytes
    {
        return Err(TileError::permanent(format!(
            "the body declares {declared} bytes, over the {max_bytes} byte ceiling"
        )));
    }

    let buffer = settle_within(
        response
            .array_buffer()
            .map_err(|error| TileError::transient(js_message(&error)))?,
        timeout_ms,
    )
    .await
    .map_err(|error| TileError::transient(format!("body read failed: {}", js_message(&error))))?
    .ok_or_else(|| TileError::transient(format!("the body stalled after {timeout_ms} ms")))?;

    let array = js_sys::Uint8Array::new(&buffer);
    let length = array.length() as usize;
    if length > max_bytes {
        return Err(TileError::permanent(format!(
            "the body holds {length} bytes, over the {max_bytes} byte ceiling"
        )));
    }
    Ok(array.to_vec())
}

/// The response's declared body length, when it declared one this script may
/// read.
fn content_length(response: &web_sys::Response) -> Option<usize> {
    response
        .headers()
        .get("content-length")
        .ok()
        .flatten()?
        .trim()
        .parse::<usize>()
        .ok()
}
