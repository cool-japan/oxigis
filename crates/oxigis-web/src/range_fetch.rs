//! Browser range transport: `fetch()` with a `Range` header.
//!
//! The platform half of `oxigis-ui`'s [`oxigis_ui::RangeTransport`] seam, i.e.
//! the Cloud-Optimized GeoTIFF counterpart of [`crate::tile_fetch`]. The shared
//! provider ([`oxigis_ui::CogTileProvider`]) owns the COG parse, the tile
//! composition and the retry policy; this module only turns
//! *(URL, byte range)* into bytes.
//!
//! The *rules* the answer is checked against — `Content-Range` parsing, the
//! answer-matches-the-question comparison and the validator pins — live in
//! [`crate::range_rules`], which is not `cfg`-gated so its tests actually run.
//! This module is the `fetch()` glue and nothing else.
//!
//! # CORS: `Range` is not safelisted
//!
//! [`crate::tile_fetch`] sends no headers at all, precisely so a cross-origin
//! tile GET stays a *simple* request. A COG read cannot: `Range` is **not** a
//! [CORS-safelisted request header], so the browser sends a preflight
//! `OPTIONS` before the very first byte. The host must therefore answer with:
//!
//! ```text
//! Access-Control-Allow-Origin: *
//! Access-Control-Allow-Headers: Range
//! Access-Control-Expose-Headers: Content-Range, Content-Length, Accept-Ranges
//! ```
//!
//! This is the standard configuration for COG hosting — AWS Open Data buckets,
//! OpenAerialMap, the Microsoft Planetary Computer and Cloudflare R2 all ship
//! it — and a host that does not cannot be read from a browser by *any* COG
//! client, not just this one. A missing preflight surfaces as an opaque
//! `TypeError` from `fetch`, which is reported as a transient failure with the
//! browser's own message; the console shows the CORS diagnostic.
//!
//! [CORS-safelisted request header]: https://fetch.spec.whatwg.org/#cors-safelisted-request-header
//!
//! # Why `If-Match` is never sent
//!
//! The native shell pins the first answer's `ETag` and sends `If-Match`
//! afterwards, letting the server refuse a stale read with `412`. **This
//! transport must not**: `If-Match` is not a [CORS-safelisted request header]
//! either, so adding it widens the preflight, and a host whose
//! `Access-Control-Allow-Headers` is exactly `Range` — which is the documented,
//! ubiquitous configuration — would start failing with an opaque `TypeError`
//! (one real archive host refuses *any* preflight outright). The comparison is
//! therefore **passive**: whatever validator the host chose to expose is pinned
//! and compared, and a host that exposes nothing leaves drift undetectable in a
//! browser. That is an accepted, documented gap, not an oversight — closing it
//! needs `Access-Control-Expose-Headers: ETag, Content-Range` from the host.
//!
//! # Threading
//!
//! As in [`crate::tile_fetch`]: [`FetchRangeTransport`] holds no `JsValue` — only
//! its own [`TransportId`] — so the `Send + Sync` bound the seam requires is
//! satisfied trivially and every JS value is created inside the `spawn_local`
//! future. The pinned validators live in a thread-local, which on `wasm32` is
//! the one and only thread every `spawn_local` future runs on.

use oxigis_render::ByteRange;
use oxigis_ui::{RangeJob, RangeSink, RangeTransport, TileError};
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::spawn_local;

use crate::range_rules::{TransportId, Validator, observe_validator, verify_content_range};
use crate::timers::{TILE_REQUEST_TIMEOUT_MS, settle_within};

pub use crate::range_rules::MAX_PINNED_VALIDATORS;

/// HTTP status that means "here is the range you asked for".
const STATUS_PARTIAL_CONTENT: u16 = 206;

/// HTTP status at and above which a failure is treated as retryable.
const FIRST_SERVER_ERROR_STATUS: u16 = 500;

/// Largest single range body accepted, in bytes.
///
/// A reader over-asks for a speculative header block and then reads leaves and
/// tiles; none of that is anywhere near this. The ceiling exists so a host that
/// ignores `Range` in a way the status check does not catch cannot make the tab
/// allocate a whole remote file.
pub const MAX_RANGE_BYTES: usize = 64 * 1024 * 1024;

/// [`RangeTransport`] backed by the browser's `fetch()` API.
///
/// Carries one piece of state: the [`TransportId`] its validator pins are keyed
/// by, which is what makes "remove and re-add the layer" clear a drift refusal
/// exactly as it does on the desktop (see [`crate::range_rules`]). Everything
/// else — connection pool, HTTP cache, per-host scheduling — belongs to the
/// browser.
#[derive(Debug, Clone, Copy)]
pub struct FetchRangeTransport {
    /// Identity this transport's pins are stored under.
    id: TransportId,
}

impl FetchRangeTransport {
    /// Creates a transport with pins of its own.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: TransportId::next(),
        }
    }

    /// The identity this transport's validator pins are keyed by.
    #[must_use]
    pub fn id(&self) -> TransportId {
        self.id
    }
}

impl Default for FetchRangeTransport {
    /// Hand-written rather than derived: a derived `Default` would hand every
    /// transport the same [`TransportId`], which is the tab-wide pin scope this
    /// type exists to stop.
    fn default() -> Self {
        Self::new()
    }
}

impl RangeTransport for FetchRangeTransport {
    fn request_range(&self, url: String, range: ByteRange, job: RangeJob, sink: RangeSink) {
        // Returns immediately: the future is queued on the browser's microtask
        // queue, so the `egui_wgpu` prepare hook that called us is not blocked.
        let id = self.id;
        spawn_local(async move {
            let result = fetch_range(id, &url, range).await;
            sink.deliver(job, result);
        });
    }
}

/// One response header, when the host exposed it to script.
///
/// A cross-origin `fetch()` only sees the headers named in
/// `Access-Control-Expose-Headers`, so [`None`] here means "the host did not
/// expose it", not "the header was absent".
fn exposed_header(response: &web_sys::Response, name: &str) -> Option<String> {
    response.headers().get(name).ok().flatten()
}

/// Renders a JS exception as a message for [`TileError`].
fn js_message(value: &JsValue) -> String {
    value
        .as_string()
        .or_else(|| {
            value
                .dyn_ref::<js_sys::Error>()
                .map(|error| error.to_string().into())
        })
        .unwrap_or_else(|| format!("{value:?}"))
}

/// `GET`s `range` of `url` and returns the bytes, under `transport`'s pins.
///
/// A `200 OK` means the server ignored `Range` and sent the whole file; that is
/// reported as a permanent failure rather than passed off as the requested
/// range, since the parser would otherwise read a tile directory out of the file
/// header. A response *shorter* than requested is fine — the reader deliberately
/// over-asks for a speculative header block.
///
/// Bounded like [`crate::tile_fetch`]: a stalled host is failed as transient
/// after [`TILE_REQUEST_TIMEOUT_MS`] rather than parking the job forever.
async fn fetch_range(
    transport: TransportId,
    url: &str,
    range: ByteRange,
) -> Result<Vec<u8>, TileError> {
    // Same accounting as `crate::tile_fetch::fetch_bytes_within`: the page's
    // in-flight counter, plus the repaint that makes a count change visible
    // while the map is otherwise idle. Held across every `?` below.
    let _activity = crate::activity::track();
    let _repaint = crate::font_fetch::RepaintOnDrop::new();

    let window = web_sys::window()
        .ok_or_else(|| TileError::permanent("no `window`: not a browser context"))?;

    let headers = web_sys::Headers::new()
        .map_err(|error| TileError::permanent(format!("Headers: {}", js_message(&error))))?;
    headers
        .set("Range", &range.header_value())
        .map_err(|error| TileError::permanent(format!("Range header: {}", js_message(&error))))?;

    let options = web_sys::RequestInit::new();
    options.set_method("GET");
    options.set_mode(web_sys::RequestMode::Cors);
    options.set_headers(&headers);

    let request = web_sys::Request::new_with_str_and_init(url, &options)
        .map_err(|error| TileError::permanent(format!("bad COG URL: {}", js_message(&error))))?;

    let response = settle_within(window.fetch_with_request(&request), TILE_REQUEST_TIMEOUT_MS)
        .await
        .map_err(|error| {
            TileError::transient(format!(
                "fetch failed (a CORS preflight rejection looks like this — the host must send \
             `Access-Control-Allow-Headers: Range`): {}",
                js_message(&error)
            ))
        })?
        .ok_or_else(|| {
            TileError::transient(format!(
                "no answer to {} within {TILE_REQUEST_TIMEOUT_MS} ms",
                range.header_value()
            ))
        })?
        .dyn_into::<web_sys::Response>()
        .map_err(|_| TileError::permanent("fetch did not resolve to a Response"))?;

    let status = response.status();
    if status != STATUS_PARTIAL_CONTENT {
        if !response.ok() {
            let message = format!("HTTP {status}");
            return Err(if status >= FIRST_SERVER_ERROR_STATUS {
                TileError::transient(message)
            } else {
                TileError::permanent(message)
            });
        }
        return Err(TileError::permanent(format!(
            "server answered HTTP {status} instead of 206 for {}: the resource does not support \
             Range requests, which a COG must",
            range.header_value()
        )));
    }

    // Read before the body is consumed; both may be `None` on a host that
    // exposes nothing, which is accepted and documented above.
    let content_range = exposed_header(&response, "content-range");
    let etag = exposed_header(&response, "etag");

    let buffer = settle_within(
        response
            .array_buffer()
            .map_err(|error| TileError::transient(js_message(&error)))?,
        TILE_REQUEST_TIMEOUT_MS,
    )
    .await
    .map_err(|error| TileError::transient(format!("body read failed: {}", js_message(&error))))?
    .ok_or_else(|| {
        TileError::transient(format!(
            "the body of {} stalled after {TILE_REQUEST_TIMEOUT_MS} ms",
            range.header_value()
        ))
    })?;

    let array = js_sys::Uint8Array::new(&buffer);
    let length = array.length() as usize;
    if length > MAX_RANGE_BYTES {
        return Err(TileError::permanent(format!(
            "the answer to {} holds {length} bytes, over the {MAX_RANGE_BYTES} byte ceiling",
            range.header_value()
        )));
    }
    let bytes = array.to_vec();
    let total = verify_content_range(content_range.as_deref(), range, bytes.len())?;
    observe_validator(transport, url, &Validator { etag, total })?;
    Ok(bytes)
}
