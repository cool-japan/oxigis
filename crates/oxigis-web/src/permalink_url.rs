//! Raw JS interop for the page's URL fragment: read `window.location.hash`
//! at startup, and write it back through `history.replaceState` afterwards
//! — never by assigning `location.hash` directly, which pushes a new
//! session-history entry on every navigation (per the WHATWG HTML
//! standard's "update the session history" steps for a fragment-only
//! navigation) and would turn a debounced pan into a back button full of
//! hundreds of stops. `history.replaceState` updates the address bar with
//! no such entry.
//!
//! # Typed `web_sys::Location` / `History`
//!
//! Both are reached through `Window::location()` / `Window::history()`, which
//! `web-sys` generates only when the crate's `web-sys` feature list enables
//! `"Location"` / `"History"`. Those two entries are now in
//! `crates/oxigis-web/Cargo.toml` and are **load-bearing**: remove either and
//! this module stops compiling, because the method it names ceases to exist.
//!
//! That is the whole reason the typed calls are worth having. The previous
//! shape reached the same two properties through `js_sys::Reflect::get` and
//! `js_sys::Function::call`, on the untyped surface every `JsValue` already
//! has — which compiles whatever property name is typed into it, so
//! `"replaceStat"` was a silent runtime no-op rather than a build error. A
//! permalink that quietly stops updating the address bar is exactly the class
//! of defect a type checker exists to catch.
//!
//! # Testing
//!
//! No `#[wasm_bindgen_test]` harness in this crate (see `serve.sh`'s `test`
//! subcommand for why), and both functions below touch nothing but
//! `web_sys::window()`, so there is no host-runnable assertion to make about
//! them beyond "this compiles for wasm32" — enforced by
//! `cargo check -p oxigis-web --target wasm32-unknown-unknown`, the same way
//! the rest of this crate's `fetch()` glue is. The parsing and formatting
//! these functions feed is the pure, host-tested half; see
//! `crate::permalink`.

/// Reads `window.location.hash`.
///
/// `""` both outside a browser context and when the page has no fragment at
/// all, which is what `Location::hash()` already answers for a fragmentless
/// URL — so the two "no fragment" cases collapse into the one answer the
/// caller wants.
#[must_use]
pub(crate) fn location_hash() -> String {
    web_sys::window()
        .and_then(|window| window.location().hash().ok())
        .unwrap_or_default()
}

/// Replaces the URL fragment with `new_hash` (expected to start with `#`)
/// through `history.replaceState`, updating the address bar with no new
/// session-history entry — see the module docs for why this, and not
/// `location.hash = `.
///
/// A silent no-op outside a browser context, or when the call itself is
/// refused (a sandboxed frame with no history access throws `SecurityError`):
/// a permalink that fails to update the address bar is a page that still
/// works in every other way, so this has nothing worth surfacing to the
/// status line over.
pub(crate) fn replace_location_hash(new_hash: &str) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Ok(history) = window.history() else {
        return;
    };
    // `replaceState(state, unused, url)`: `state` is `null` (this shell keeps
    // no `popstate` payload), the title argument every browser ignores is an
    // empty string, and `url` is the new fragment alone — resolved against
    // the current document, exactly like assigning `location.hash` would
    // resolve it, but without the extra history entry.
    let _ = history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(new_hash));
}
