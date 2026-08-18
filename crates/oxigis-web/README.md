# oxigis-web

WASM shell that hosts [`oxigis-ui`](../oxigis-ui)'s `OxigisApp` in the browser via `eframe` + `wasm-bindgen`, requesting a WebGPU renderer with an automatic WebGL2 fallback.

A hosted build is live at [gis.cooljapan.tech](https://gis.cooljapan.tech/).

**Status:** Alpha

## Features

- **Renderer selection** — asks for `eframe::Renderer::Wgpu`; `wgpu` itself probes `navigator.gpu` first and falls back to WebGL2 when WebGPU is unavailable or refuses an adapter. WebGPU needs a secure context (`https://` or `http://localhost`), which is why `serve.sh` binds localhost rather than a LAN IP.
- **Tile fetch** (`src/tile_fetch.rs`) — `fetch()`-backed transport for XYZ raster basemaps and MVT vector tiles, sending no custom headers so cross-origin requests stay CORS-simple.
- **Range fetch** (`src/range_fetch.rs`, rules in `src/range_rules.rs`) — `fetch()` with a `Range` header for Cloud-Optimized GeoTIFF tiles; verifies `Content-Range` against the requested range, refuses a body the length rule says was decoded, and pins `ETag`s and lengths — per transport, in a bounded store — so an archive that changes mid-session is caught rather than mixed.
- **Font fetch** (`src/font_fetch.rs`) — the Latin label face (Noto Sans Regular) is bundled at compile time; a CJK fallback and a separate print-only face are fetched at runtime and opted into via `data-oxigis-*` attributes on the canvas, so the ~16 MB CJK face is never downloaded unless a page asks for it.
- **Local data by drag-and-drop** — PMTiles/MBTiles archives and local GeoJSON/shapefile sets open by dropping them onto the map; a browser has no filesystem access, so there is no open-by-path fallback.
- **N-layer tile stack** — the shell reconciles the whole project stack rather than the two legacy single-slot seams (`set_tile_stack_shell(true)`): it reads back what is installed, then applies the `Install` / `Remove` / `Reorder` work `oxigis-ui` asks for, rebuilding each entry's source from the bytes a drop left in memory. Entries the GPU refused are re-offered rather than silently dropped.
- **Permalinks** (`src/permalink.rs`, `src/permalink_url.rs`) — the camera round-trips through the URL hash as `#map=<zoom>/<lat>/<lon>`, the same order OpenStreetMap's own permalink uses, so a link copied out of OxiGIS reads like a link copied out of any other slippy map. The parsing/formatting half is deliberately not `wasm32`-gated, so it is tested on a host with no browser; the basemap is deliberately not encoded (a URL template plus a free-text attribution line would dwarf the three numbers around it).
- **Loading indicator** (`src/activity.rs`) — the two transports every fetch this shell makes funnels through (`tile_fetch::fetch_bytes_within`, `range_fetch::fetch_range`) each `track` their in-flight requests, and the count is written into the page's `#oxigis_loading` element only when it changes.
- **Diagnostics that reach the console** — `oxigis-ui` and `oxigis-render` report through `tracing`, and a browser has no `tracing` subscriber. Rather than install one, this shell takes `tracing`'s `log` feature, which emits a `log` record for every event *when no subscriber is active* — exactly this shell's situation — so every warning from the shared crates lands in the console alongside the shell's own.
- **PDF export** — composes the current view to a raster, embeds the active fonts, and delivers the result as a browser download (bytes → `Blob` → object URL → synthetic `<a>` click). Tiles are collected by `src/tile_drain.rs`, which takes each one **out** of its provider the moment it appears: a printed page at 288 dpi needs several times a screen's worth of tiles, so the old "wait until all of them are resident" predicate could never become true past the provider's LRU size. Archive-backed layers are resolved for every stack entry, not just the two legacy slots, so an N-layer export cannot silently print without them. Progress, success and failure are reported back to the app's status line through `src/export_status.rs` instead of vanishing into a console nobody has open.

## Building

```bash
wasm-pack build crates/oxigis-web --target web   # browser bundle
```

Everything that touches the DOM or `fetch()` — `font_fetch`, `range_fetch`, `tile_fetch`, `permalink_url`, `timers`, `export_status` and the shell itself — is `cfg`-gated to `target_arch = "wasm32"`. Native shells use `oxigis-desktop` instead.

## Tests

**36 tests passing** on a native target. The shell's decisions were deliberately split out of its browser plumbing so they can be tested without a browser at all: `permalink` (hash parse/format and its rounding), `activity` (in-flight counting), `range_rules` (`Content-Range` parsing, `ETag`/length drift pinning, pin-store bounds) and `tile_drain` (the take-once collection rule and its convergence on a page larger than the cache) are not `wasm32`-gated, so `cargo nextest run -p oxigis-web` exercises them on the host. What is left behind the gate is the DOM and `fetch()` glue those rules are called from. The logic this shell wraps — layers, styles, the GPU map, panels — is tested in `oxigis-ui` and `oxigis-render`.

Part of [OxiGIS](https://github.com/cool-japan/oxigis) — Pure Rust full-stack GIS.
See the workspace README for the crate matrix and build instructions.

© 2026 COOLJAPAN OU (Team Kitasan) · Apache-2.0
