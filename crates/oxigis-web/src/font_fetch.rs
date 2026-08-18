//! Runtime font loading for the browser shell: fetch a CJK fallback face and
//! hand it to the label pass when it arrives.
//!
//! # Why the CJK font is not in the bundle
//!
//! The primary label font — `oxifont_bundled::NOTO_SANS_REGULAR`, 431 KB of
//! Latin coverage — is compiled in, because it is the one face every label
//! starts in and the map would be textless without it. A pan-CJK face is
//! **16 MB**, which is larger than the rest of the wasm module by a wide margin
//! and would be downloaded by every visitor whether or not the map ever shows a
//! CJK name. So it is not embedded, and there is no default URL either: a page
//! that wants CJK labels opts in.
//!
//! # Opting in
//!
//! Two equivalent ways, both landing in [`load_fallback_font`]:
//!
//! * **Markup** — put the URL on the canvas element, and the shell picks it up
//!   during startup:
//!
//!   ```html
//!   <canvas id="oxigis_canvas"
//!           data-oxigis-fallback-font="https://cdn.jsdelivr.net/fontsource/fonts/noto-sans-jp@latest/japanese-400-normal.ttf">
//!   </canvas>
//!   ```
//!
//! * **JavaScript** — call the exported function at any time, including long
//!   after the map is running:
//!
//!   ```js
//!   import init, { oxigis_load_fallback_font } from "./pkg/oxigis_web.js";
//!   await init();
//!   oxigis_load_fallback_font(
//!     "https://cdn.jsdelivr.net/fontsource/fonts/noto-sans-jp@latest/japanese-400-normal.ttf",
//!   );
//!   ```
//!
//! The URL must be CORS-readable (`Access-Control-Allow-Origin`), like every
//! tile endpoint the shell fetches from.
//!
//! # Print-only opt-in (v1.2)
//!
//! A page that wants CJK **only in PDF exports** — without paying for a
//! screen label font — sets `data-oxigis-print-font` on the canvas (or calls
//! `oxigis_load_print_font(url)`). Unlike the label fallback, nothing is
//! fetched at startup: the bytes are downloaded on the **first export**,
//! cached for later ones, and never touch the label engine. The label
//! fallback, when installed, is still embedded too — unless it is the same
//! URL as the print font, in which case it is embedded once.
//!
//! # Format
//!
//! **TTF and OTF only.** `oxitext` takes raw SFNT bytes; a `.woff2` would have
//! to be decompressed first, and while `oxifont-webfont` can do exactly that,
//! it is not worth a dependency and a wasm-size audit for a format every CDN
//! also serves uncompressed. A WOFF2 body simply fails to parse, which the
//! label engine reports and the shell logs.
//!
//! # Arrival is asynchronous
//!
//! `fetch()` resolves whenever it resolves — possibly seconds after the map is
//! already drawing Latin labels. The bytes therefore land in a slot rather than
//! anywhere useful, and the shell drains that slot on its next frame, when it
//! has the `RenderState` needed to reach
//! [`oxigis_ui::map_gpu::add_label_fallback_font`].
//!
//! "Its next frame" is the catch: eframe's web runner stops scheduling frames
//! when nothing is happening, and a map whose tiles have all landed is exactly
//! that. So the arriving bytes must *ask* for a frame, the same way
//! `oxigis_ui`'s tile providers ask for one when a tile lands. That needs an
//! [`egui::Context`], which the JS entry point has no way to pass in — hence
//! [`remember_context`], called by the shell once per frame, and the two
//! `request_repaint` calls: one when the bytes arrive, one after the fallback
//! is installed (installing it invalidates every shaped label, so the map has
//! to draw once more to show the new glyphs).

use std::cell::RefCell;
use std::sync::Arc;

use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen_futures::spawn_local;

use crate::timers::FONT_REQUEST_TIMEOUT_MS;

/// Canvas attribute the startup path reads a fallback-font URL from.
pub const FALLBACK_FONT_ATTRIBUTE: &str = "data-oxigis-fallback-font";

/// Canvas attribute naming a face to embed in **PDF exports only** — the
/// print twin of [`FALLBACK_FONT_ATTRIBUTE`]. The URL is recorded at
/// startup but the bytes are fetched **lazily, on the first export** (and
/// cached): a visitor who never exports pays zero bytes, which is the whole
/// reason a pan-CJK face is not bundled. The JS half is
/// [`oxigis_load_print_font`]. Same format rule as the label fallback:
/// TTF/OTF (a `.ttc` collection is fine — the print path takes face 0), no
/// WOFF2.
pub const PRINT_FONT_ATTRIBUTE: &str = "data-oxigis-print-font";

/// Canvas attribute naming the **bold** twin of [`FALLBACK_FONT_ATTRIBUTE`]
/// (print/text v1.4, D-W4): the face a `Bold`-weighted symbol style draws
/// through, on screen and on the page.
///
/// Fetched at startup exactly like the regular fallback, because a bold
/// label has to be able to draw on the first frame it is asked for. The
/// label engine puts the whole regular chain BEHIND it, so a Latin-only bold
/// face costs nothing in coverage. Absent — the default — Bold labels draw
/// Regular with one log; nothing is ever synthetically emboldened. The JS
/// half is [`oxigis_load_bold_font`].
pub const BOLD_FONT_ATTRIBUTE: &str = "data-oxigis-bold-font";

/// Canvas attribute naming a **bold** face to embed in PDF exports only —
/// the bold twin of [`PRINT_FONT_ATTRIBUTE`], with the same lazy fetch (on
/// the first export, then cached) and the same [`FaceRole::PrintOnly`]
/// treatment: the screen never rasterises it, so a variable face may be
/// normalised to its nearest-Bold instance. The JS half is
/// [`oxigis_load_print_bold_font`].
///
/// [`FaceRole::PrintOnly`]: oxigis_ui::print::FaceRole::PrintOnly
pub const PRINT_BOLD_FONT_ATTRIBUTE: &str = "data-oxigis-print-bold-font";

/// A working CJK font URL, for documentation and manual testing.
///
/// Not used by default — see the [module docs][self]; opting in is explicit.
pub const EXAMPLE_CJK_FONT_URL: &str =
    "https://cdn.jsdelivr.net/fontsource/fonts/noto-sans-jp@latest/japanese-400-normal.ttf";

/// Largest font body accepted, in bytes.
///
/// Comfortably above a per-script Noto face (a few MB) and below the pan-CJK
/// collections nobody should be shipping to a browser.
///
/// Enforced by `tile_fetch`'s bounded GET, which checks the declared
/// `Content-Length` **before** reading the body — so a mistyped URL
/// pointing at something enormous is refused without downloading it. A chunked
/// response declares no length at all, and is still buffered whole before the
/// ceiling can be applied; that is the platform's limit, not a policy choice.
pub const MAX_FONT_BYTES: usize = 32 * 1024 * 1024;

thread_local! {
    /// The running app's repaint handle, so an arriving font can wake an idle
    /// map. Set by [`remember_context`] every frame; [`None`] before the first.
    static REPAINT_CONTEXT: RefCell<Option<egui::Context>> = const { RefCell::new(None) };

    /// Fetched font bytes waiting for a frame that can install them.
    ///
    /// A `thread_local` rather than a field of the shell because
    /// [`oxigis_load_fallback_font`] is callable from JS at any moment, with no
    /// handle on the running app. wasm is single-threaded, so this is simply a
    /// module-level slot with a `RefCell`'s borrow discipline.
    static PENDING_FONT: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };

    /// The last fallback font that actually reached the label engine, kept so
    /// the PDF export can embed the same face the map draws with. The label
    /// engine's own copy lives inside `wgpu` callback resources, which the
    /// export path cannot reach.
    ///
    /// Shared rather than owned: a CJK face is megabytes, an export asks for
    /// it once per run, and a `Vec` handed out by value made every one of
    /// those asks a full copy in a 32-bit address space.
    static INSTALLED_FONT: RefCell<Option<Arc<[u8]>>> = const { RefCell::new(None) };

    /// The URL the label fallback was fetched from — kept so the export can
    /// skip embedding the same face twice when the page names it as the
    /// print font too (comparing multi-megabyte byte vectors would be silly).
    static FALLBACK_URL: RefCell<Option<String>> = const { RefCell::new(None) };

    /// The print-only font URL the page configured, if any.
    static PRINT_FONT_URL: RefCell<Option<String>> = const { RefCell::new(None) };

    /// The fetched print face, cached so a second export costs nothing.
    /// Shared for the same reason as [`INSTALLED_FONT`].
    static PRINT_FONT_BYTES: RefCell<Option<Arc<[u8]>>> = const { RefCell::new(None) };

    /// Fetched BOLD label-face bytes waiting for a frame that can install
    /// them — the bold twin of [`PENDING_FONT`].
    static PENDING_BOLD_FONT: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };

    /// The bold face that actually reached the label engine, kept so the PDF
    /// export can embed the same face the map draws bold with.
    static INSTALLED_BOLD_FONT: RefCell<Option<Arc<[u8]>>> = const { RefCell::new(None) };

    /// The URL the bold label face was fetched from, so the export can skip
    /// embedding the same face twice when the page names it for print too.
    static BOLD_FONT_URL: RefCell<Option<String>> = const { RefCell::new(None) };

    /// The print-only BOLD font URL the page configured, if any.
    static PRINT_BOLD_FONT_URL: RefCell<Option<String>> = const { RefCell::new(None) };

    /// The fetched print-only bold face, cached like [`PRINT_FONT_BYTES`].
    static PRINT_BOLD_FONT_BYTES: RefCell<Option<Arc<[u8]>>> = const { RefCell::new(None) };
}

/// Records the print-font URL. Safe before the map exists, more than once,
/// and with an empty string (a no-op — "not configured" is the default).
pub fn set_print_font_url(url: &str) {
    let url = url.trim().to_owned();
    if url.is_empty() {
        return;
    }
    PRINT_FONT_URL.with_borrow_mut(|slot| *slot = Some(url));
}

/// The configured print-font URL, if any.
#[must_use]
pub fn print_font_url() -> Option<String> {
    PRINT_FONT_URL.with_borrow(Clone::clone)
}

/// The URL the label fallback was loaded from, if any.
#[must_use]
pub fn fallback_url() -> Option<String> {
    FALLBACK_URL.with_borrow(Clone::clone)
}

/// Reads [`PRINT_FONT_ATTRIBUTE`] off the canvas element, if the page set
/// one.
#[must_use]
pub fn print_url_from_canvas() -> Option<String> {
    attribute_from_canvas(PRINT_FONT_ATTRIBUTE)
}

/// The print face for THIS export: the cached bytes, else one `fetch()` of
/// the configured URL (validated and cached), else [`None`].
///
/// Async on purpose — `export_pdf` already awaits its tile polls, so one
/// more await is structurally free and race-free: the export cannot start
/// assembling before the face is in hand. wasm rule honoured throughout:
/// every `RefCell` borrow ends BEFORE an await point.
pub async fn print_font() -> Option<Arc<[u8]>> {
    if let Some(bytes) = PRINT_FONT_BYTES.with_borrow(Clone::clone) {
        return Some(bytes);
    }
    let url = PRINT_FONT_URL.with_borrow(Clone::clone)?;
    log::info!("OxiGIS web shell: fetching the print font from {url}");
    let bytes: Arc<[u8]> = Arc::from(fetch_font_body(&url, "print font").await?);
    PRINT_FONT_BYTES.with_borrow_mut(|slot| *slot = Some(Arc::clone(&bytes)));
    Some(bytes)
}

/// Remembers the fallback font the shell just installed into the label
/// engine, so [`installed_font`] can hand the same bytes to the PDF export.
pub fn retain_installed_font(bytes: &[u8]) {
    INSTALLED_FONT.with_borrow_mut(|slot| *slot = Some(Arc::from(bytes)));
}

/// The retained fallback font, if one was ever installed.
#[must_use]
pub fn installed_font() -> Option<Arc<[u8]>> {
    INSTALLED_FONT.with_borrow(Clone::clone)
}

/// Records the print-only BOLD font URL — the bold twin of
/// [`set_print_font_url`], with the same tolerances.
pub fn set_print_bold_font_url(url: &str) {
    let url = url.trim().to_owned();
    if url.is_empty() {
        return;
    }
    PRINT_BOLD_FONT_URL.with_borrow_mut(|slot| *slot = Some(url));
}

/// The configured print-only bold font URL, if any.
#[must_use]
pub fn print_bold_font_url() -> Option<String> {
    PRINT_BOLD_FONT_URL.with_borrow(Clone::clone)
}

/// The URL the bold label face was loaded from, if any.
#[must_use]
pub fn bold_font_url() -> Option<String> {
    BOLD_FONT_URL.with_borrow(Clone::clone)
}

/// Reads [`BOLD_FONT_ATTRIBUTE`] off the canvas element, if the page set one.
#[must_use]
pub fn bold_url_from_canvas() -> Option<String> {
    attribute_from_canvas(BOLD_FONT_ATTRIBUTE)
}

/// Reads [`PRINT_BOLD_FONT_ATTRIBUTE`] off the canvas element, if set.
#[must_use]
pub fn print_bold_url_from_canvas() -> Option<String> {
    attribute_from_canvas(PRINT_BOLD_FONT_ATTRIBUTE)
}

/// The print-only BOLD face for THIS export — the bold twin of
/// [`print_font`], same lazy fetch, same cache, same wasm borrow rule (every
/// `RefCell` borrow ends before an await).
pub async fn print_bold_font() -> Option<Arc<[u8]>> {
    if let Some(bytes) = PRINT_BOLD_FONT_BYTES.with_borrow(Clone::clone) {
        return Some(bytes);
    }
    let url = PRINT_BOLD_FONT_URL.with_borrow(Clone::clone)?;
    log::info!("OxiGIS web shell: fetching the print bold font from {url}");
    let bytes: Arc<[u8]> = Arc::from(fetch_font_body(&url, "print bold font").await?);
    PRINT_BOLD_FONT_BYTES.with_borrow_mut(|slot| *slot = Some(Arc::clone(&bytes)));
    Some(bytes)
}

/// Starts fetching `url` as the BOLD label face — the bold twin of
/// [`load_fallback_font`], with the same tolerances and the same failure
/// policy (a miss means Bold labels draw Regular, never a synthetic bold).
pub fn load_bold_font(url: &str) {
    let url = url.trim().to_owned();
    if url.is_empty() {
        return;
    }
    BOLD_FONT_URL.with_borrow_mut(|slot| *slot = Some(url.clone()));
    log::info!("OxiGIS web shell: fetching the bold label font from {url}");
    spawn_local(async move {
        if let Some(bytes) = fetch_font_body(&url, "bold label font").await {
            log::info!(
                "OxiGIS web shell: bold label font fetched ({} bytes)",
                bytes.len()
            );
            PENDING_BOLD_FONT.with_borrow_mut(|slot| *slot = Some(bytes));
            request_repaint();
        }
    });
}

/// Takes the fetched BOLD font bytes, if any have arrived since the last
/// call — the bold twin of [`take_pending_font`].
#[must_use]
pub fn take_pending_bold_font() -> Option<Vec<u8>> {
    PENDING_BOLD_FONT.with_borrow_mut(Option::take)
}

/// Remembers the bold face the shell just installed, so the export can embed
/// the same bytes the map draws bold with.
pub fn retain_installed_bold_font(bytes: &[u8]) {
    INSTALLED_BOLD_FONT.with_borrow_mut(|slot| *slot = Some(Arc::from(bytes)));
}

/// The retained bold label face, if one was ever installed.
#[must_use]
pub fn installed_bold_font() -> Option<Arc<[u8]>> {
    INSTALLED_BOLD_FONT.with_borrow(Clone::clone)
}

/// One validated font body, or [`None`] with the failure logged: empty,
/// oversized and transport failures all mean "no font", never an error the
/// caller has to handle. `what` names the face in the log.
///
/// The ceiling is enforced by the transport rather than here, which is what
/// lets a declared `Content-Length` over [`MAX_FONT_BYTES`] be refused before
/// the body is downloaded at all.
async fn fetch_font_body(url: &str, what: &str) -> Option<Vec<u8>> {
    // The font deadline and the font ceiling, not the tile ones: a 16 MB CJK
    // face on a slow link is not a stalled server, and `MAX_FONT_BYTES` is
    // twice `MAX_TILE_BYTES` — going through `fetch_bytes` silently capped
    // fonts at the tile ceiling and failed a merely slow one after 20 s.
    let bytes =
        match super::tile_fetch::fetch_bytes_within(url, FONT_REQUEST_TIMEOUT_MS, MAX_FONT_BYTES)
            .await
        {
            Ok(bytes) => bytes,
            Err(error) => {
                log::warn!("OxiGIS web shell: the {what} at {url} failed: {error}");
                return None;
            }
        };
    if bytes.is_empty() {
        log::warn!("OxiGIS web shell: the {what} at {url} is empty");
        return None;
    }
    Some(bytes)
}

/// Starts fetching `url` as a label fallback font.
///
/// Returns immediately; the bytes appear in [`take_pending_font`] once the
/// browser has them. An empty or whitespace-only URL is a no-op, which is what
/// makes "no fallback configured" the default rather than a special case.
///
/// Failures — a CORS rejection, a 404, an oversized body — are logged and
/// otherwise ignored: CJK labels stay `.notdef`, Latin ones are unaffected.
pub fn load_fallback_font(url: &str) {
    let url = url.trim().to_owned();
    if url.is_empty() {
        return;
    }
    FALLBACK_URL.with_borrow_mut(|slot| *slot = Some(url.clone()));
    log::info!("OxiGIS web shell: fetching the label fallback font from {url}");
    spawn_local(async move {
        if let Some(bytes) = fetch_font_body(&url, "fallback font").await {
            log::info!(
                "OxiGIS web shell: fallback font fetched ({} bytes)",
                bytes.len()
            );
            PENDING_FONT.with_borrow_mut(|slot| *slot = Some(bytes));
            // The map is very likely idle by now: without this the bytes
            // would sit in the slot until the user happened to pan.
            request_repaint();
        }
    });
}

/// Records the running app's repaint handle.
///
/// Called once per frame by the shell; cloning an [`egui::Context`] is a
/// reference-count bump, so this is not worth guarding.
pub fn remember_context(ctx: &egui::Context) {
    REPAINT_CONTEXT.with_borrow_mut(|slot| {
        if slot.is_none() {
            *slot = Some(ctx.clone());
        }
    });
}

/// Asks the running app for another frame, if there is one.
///
/// A no-op before the first frame — at which point frames are being scheduled
/// anyway, so nothing is lost.
pub fn request_repaint() {
    REPAINT_CONTEXT.with_borrow(|slot| {
        if let Some(ctx) = slot.as_ref() {
            ctx.request_repaint();
        }
    });
}

/// Asks for a repaint now, and again when the returned guard drops.
///
/// A fetch's *start* and its *end* both need to be visible even when the map
/// is otherwise idle — a tile fetch begins mid-frame (a repaint is already
/// under way) but a font fetch can begin from
/// [`oxigis_load_fallback_font`]/[`oxigis_load_bold_font`], callable from JS
/// at any moment with nothing else scheduling a frame. And on the way out:
/// [`fetch_font_body`]'s success path already asks for a repaint, but its
/// failure path (a 404, a CORS rejection, a timeout) used to ask for
/// nothing, leaving whatever consumed the failure invisible until an
/// unrelated repaint happened to occur. One guard, held across the fetch,
/// closes both gaps at once for every caller that adopts it — currently
/// [`crate::tile_fetch::fetch_bytes_within`] (tiles and fonts both funnel
/// through it) and [`crate::range_fetch`]'s COG/archive range GET.
///
/// `#[must_use]`: a guard dropped immediately after construction asks for
/// two repaints back to back instead of bracketing a fetch, which is never
/// the intent.
#[must_use = "hold this across the fetch it brackets, not just at the call site"]
pub(crate) struct RepaintOnDrop(());

impl RepaintOnDrop {
    /// Requests a repaint immediately, then again on drop.
    pub(crate) fn new() -> Self {
        request_repaint();
        Self(())
    }
}

impl Drop for RepaintOnDrop {
    fn drop(&mut self) {
        request_repaint();
    }
}

/// Takes the fetched font bytes, if any have arrived since the last call.
///
/// Called once per frame by the shell; [`None`] is the overwhelmingly common
/// answer and costs one `RefCell` borrow.
#[must_use]
pub fn take_pending_font() -> Option<Vec<u8>> {
    PENDING_FONT.with_borrow_mut(Option::take)
}

/// Reads the fallback-font URL off the canvas element, if the page set one.
///
/// See [`FALLBACK_FONT_ATTRIBUTE`]. A missing attribute, a missing canvas or a
/// document-less context all mean "no fallback", which is the default.
#[must_use]
pub fn url_from_canvas() -> Option<String> {
    attribute_from_canvas(FALLBACK_FONT_ATTRIBUTE)
}

/// Reads one attribute off the canvas element. A missing attribute, a
/// missing canvas or a document-less context all mean "not configured".
///
/// One helper for every `data-oxigis-*` attribute the shell reads, fonts and
/// backend alike; it lives beside [`crate::CANVAS_ID`]'s other consumers.
fn attribute_from_canvas(name: &str) -> Option<String> {
    crate::wasm::canvas_attribute(name)
}

/// Fetches a label fallback font from `url`, from JavaScript.
///
/// The public opt-in for CJK labels; see the [module docs][self] for the markup
/// alternative and the format restriction (TTF/OTF, not WOFF2). Safe to call
/// before the map exists, more than once, or with an empty string.
#[wasm_bindgen]
pub fn oxigis_load_fallback_font(url: String) {
    load_fallback_font(&url);
}

/// Configures a face to embed in PDF exports only, from JavaScript.
///
/// The print twin of [`oxigis_load_fallback_font`]: nothing is fetched
/// until the first export (then cached), and the screen's labels are never
/// touched. Safe to call before the map exists, more than once, or with an
/// empty string.
#[wasm_bindgen]
pub fn oxigis_load_print_font(url: String) {
    set_print_font_url(&url);
}

/// Fetches the BOLD label face from `url`, from JavaScript — the bold twin
/// of [`oxigis_load_fallback_font`] (print/text v1.4). Same tolerances: safe
/// before the map exists, more than once, or with an empty string.
#[wasm_bindgen]
pub fn oxigis_load_bold_font(url: String) {
    load_bold_font(&url);
}

/// Configures a BOLD face to embed in PDF exports only, from JavaScript —
/// the bold twin of [`oxigis_load_print_font`].
#[wasm_bindgen]
pub fn oxigis_load_print_bold_font(url: String) {
    set_print_bold_font_url(&url);
}
