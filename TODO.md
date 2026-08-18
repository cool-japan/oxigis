# OxiGIS TODO

Phase plan authority: the OxiGIS blueprint (`oxigis.md`), which lives in git
history only — it was removed from the tree before this release as a working
document. Completed work: see CHANGELOG.md and git history.

Everything below is open work, deliberate and verified. Nothing that has already
shipped is recorded here.

## Milestone 1 close-out (user actions)

- [ ] Browser visual smoke test — partially satisfied. Headless Chrome against a
      local server was verified today: the app boots and renders end-to-end
      through the WebGL2 fallback. What a headless run cannot answer is still
      open — the WebGPU path (that run fell back to WebGL2) and a human-eye pass
      over the map, the panels and a dropped GeoJSON in a real browser
      (`bash crates/oxigis-web/serve.sh`). Every other gate the workspace runs is
      headless, so this pass has no substitute.
- [ ] Deploy the browser build to gis.cooljapan.tech — STAGED, not yet live. A
      complete demo site is staged in the cooljapan site repo: a `public/gis/`
      mount plus a Worker Host-header remap, so it will serve as
      <https://gis.cooljapan.tech/>. It was built from this repo's main@24de6f7
      with the `wasm-release` profile and verified locally under `wrangler dev` —
      routing, MIME types and a sha256-identical wasm payload, plus app boot and
      WebGL2 rendering end-to-end in headless Chrome. Left to the user: commit
      and deploy the cooljapan site repo (`npm run deploy`), add the
      `gis.cooljapan.tech` custom domain in the Cloudflare dashboard, and take
      the WebGPU eyeball above once against a real browser.
- [ ] Report upstream to cool-japan/oxitext: `oxitext-raster` 0.2.3's
      thread-local `fontdue::Font` cache (`src/tl_cache.rs`,
      `get_or_parse_fontdue`) keys each font by an FNV-1a hash of **at most the
      first 64 bytes** of the file, so two sibling faces that agree on the sfnt
      header and early table directory (regular/bold cuts from one pipeline,
      static instances of one VF) silently share one parsed font — the second
      face rasterises with the first face's outlines for as long as the entry
      stays in the 32-slot LRU. Suggested fix: hash the whole file (or
      length + whole-file hash). OxiGIS is mitigated — `oxigis-render`'s label
      engine interns fonts by `Arc` pointer identity and warns on key
      collisions (`report_raster_key_collision`,
      `crates/oxigis-render/src/label/engine.rs`) — but the fix itself has to
      land upstream.
- [ ] Report upstream to cool-japan/oxitext + oxifont: the WOFF decoders that
      default features compile into the wasm build (`oxitext` → `oxifont`
      `discovery` → `oxifont-discovery` WOFF defaults); the full analysis and
      the exact upstream ask live in the workspace `Cargo.toml` comment above
      `oxifont-discovery`.
      (The former `docs/upstream-reports.md` draft file was removed 2026-08-18:
      these two items are all that remained open — every other finding it
      recorded is fixed, locally or upstream. The full drafts stay available in
      git history.)

## Roadmap

- [ ] **Web project save as a download.** ONE flag, `set_native_project_io`,
      governs both File ▸ Save and File ▸ Open: enabling it queues a
      `ProjectOpenRequest` the browser cannot satisfy, replacing the working
      paste box with a dead menu item. Needs an `<input type="file">` +
      `FileReader` Open path plumbed into the web frame loop FIRST; after that,
      flipping the flag and draining `take_pending_project_save` into
      `deliver_download(name, "application/json", …)` is ~15 lines, since the
      download helper is already generalised and in use.
- [ ] **LTR script itemisation in print + screen-side Indic parity.** The two
      must land TOGETHER: both sides shape everything as Latin today and measure
      identically, so adopting a real script tag in print alone would open a
      screen/page width gap of up to 35 % (Oriya worst case, Devanagari −24 %).
      Remaining prerequisites: the old adjacent-equal-gid guard must be
      REPLACED, not ported — it has no true-positive case left, so carrying it
      forward is pure false-refusal risk; and the `nkoo` → `nko ` tag byte needs
      a live golden against a covering face before it is changed (N'Ko is
      cursive-joining and this repo has no corpus to catch a regression).
      Two negative constraints, both measured: `oxitext`'s `icu` feature stays
      OFF (its branch of `shape_and_layout` shapes through the primary face with
      NO notdef fallback, i.e. a CJK regression, not a preference), and
      `Pipeline::shape_with_fallback` is unusable (it `to_vec()`s a 9.5 MB
      `meiryo.ttc` per call and still carries no per-glyph font).
- [ ] **Archive zoom range.** `map_gpu::set_tile_layer_zoom_range` is written,
      exported and called by nothing. It is blocked on its own stated
      precondition — "once the archive header lands" — because nothing exposes an
      archive's min/max zoom to a shell. Until then a detail-limited archive
      draws nothing above its top zoom instead of magnified coarse tiles.
- [ ] **Data-driven MVT paint** — deliberately REFUSED rather than half-built.
      An MVT paint is matched by source-layer name and never sees a feature's
      attributes, so a tiled layer's renderer combo is drawn DISABLED with
      `TILED_RENDERER_REFUSAL` as its reason, visible in the app. Closing it
      means widening `oxigis-render`'s `PaintResolver` seam to
      `paint_for(layer, properties)` and adding a `feature_ranges` side table to
      `VectorMesh` for picking. The core model is already ready:
      `Renderer::class_of` takes any `impl Attributes` and
      `local_vector::classify::MvtAttributes` is the adapter for the MVT case.
- [ ] **Print z-order: one arrangement still flattens.** Rasters are composited
      into the ONE image the page embeds and vector-tile layers are paths drawn
      over it, so a raster entry sitting ABOVE a vector-tile entry on screen
      still prints below it. This is the page's own documented order
      (raster → MVT → local → labels); every other arrangement — any number of
      rasters, any number of tilesets, in any order among themselves — composes
      exactly as the screen does. Closing it means one image XObject per
      contiguous raster run, each with an `/SMask` soft mask, because
      `compose_map_rgb` bakes onto `PAPER_WHITE` and has no alpha channel, so a
      second opaque image would erase what is under it. Stated in
      `print/mod.rs`'s module docs and in `page_content_planned_with`'s own docs,
      not only here.
- [ ] **Raster opacity on the legacy print path** (pre-existing, deliberately not
      flipped). A project with a single translucent COG or tile archive satisfies
      `stack_fits_legacy_slots`, and on that path the provider composites the
      basemap in at full alpha and `PrintTileLayer::opacity` is not applied — so
      it prints opaque while the screen shows it faded. Widening the gate to
      catch it would change the bytes of exports that work today, which is the
      exact risk the composed path was gated to avoid. The composed path does
      honour opacity, so this is a one-condition gate change whenever that trade
      is wanted.

## Deferred by design

- [ ] **Plugin mechanism** — postponed by the blueprint itself; the intended
      shape is wasmtime-hosted WASM plugins.
- [ ] **The `oxigis` facade crate** on crates.io holds a 0.0.1 placeholder.
      Decide whether it becomes a meta-crate or is re-published from this
      workspace.

## Housekeeping

- [ ] **Packaging assets** (no owner, all outside the crates): a `.desktop` file
      with `MimeType=application/x-oxigis-project` for app id
      `io.cooljapan.oxigis`, an `Info.plist` with `CFBundleDocumentTypes` for
      `.oxigis.json`, and `.ico`/`.icns` artwork plus a
      `ViewportBuilder::with_icon(include_bytes!(…))` call.
- [ ] **Small optional polish**, none of it blocking: `app/data_io.rs` appending
      ", reprojected from {}." to the add status line; `table_panel.rs` gaining
      `selected_source_rows()`; a print-dialog slider for
      `PrintOptions::jpeg_quality` (the field exists and is honoured, only the
      control is missing); `OxigisApp::export_pending()` used to disable a
      shell's own export UI; surfacing Ctrl/Cmd+G in a platform menu bar (would
      need `open_go_to_dialog`, currently `pub(crate)`, made `pub`).
- [ ] **File sizes to watch** (the workspace rule is 2000 lines per file).
      `map_gpu.rs` is at 1958 — compliant, but the least headroom in
      `oxigis-ui`; the recorded split (`map_gpu/mod.rs` + `map_gpu/tile_stack.rs`,
      public API preserved) remains the right move for whatever grows it next.
      Desktop `main.rs` is at 1905; `export_stack.rs` already carried the bulk of
      the print-stack work out of it, and the next addition should start by
      splitting the export half (`export_pdf_to` + `ExportReport` + the tile
      budget constants) out beside it.
