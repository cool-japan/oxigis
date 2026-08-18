//! OxiGIS web shell — the WASM-first distribution (blueprint §2).
//!
//! Targets `wasm32-unknown-unknown` and hosts [`oxigis_ui::OxigisApp`] inside
//! an [`eframe`] web runner attached to the `<canvas id="oxigis_canvas">`
//! element of `crates/oxigis-web/index.html`. This is the Phase 0 week-1
//! mandatory milestone: the whole UI stack compiling and painting in a
//! browser.
//!
//! # Renderer selection and WebGL fallback
//!
//! The runner asks for `eframe::Renderer::Wgpu`. `eframe` 0.35.0's default
//! feature set does not include `glow` (verified against the registry
//! manifest), and the workspace only adds `wgpu` on top of the defaults, so
//! `Renderer::Glow` is `cfg`-ed out entirely: `Renderer::Wgpu` is the only
//! variant that exists and is also the [`Default`]. The request is spelled out
//! anyway so the intent survives a future feature change.
//!
//! Fallback is handled one level down, inside `wgpu` itself: the dependency
//! graph enables **both** the `webgpu` and the `webgl` features of
//! `wgpu` 29.0.4 (verify with
//! `cargo tree -p oxigis-web --target wasm32-unknown-unknown -e features`).
//! `wgpu`'s browser instance therefore probes `navigator.gpu` first and, when
//! WebGPU is unavailable or refuses an adapter, falls back to a WebGL2
//! backend.
//!
//! Two things the shell *does* configure, because the defaults do not survive
//! contact with a real WebGL2 device:
//!
//! * **The device request is clamped to what the adapter has.** `egui-wgpu`'s
//!   default descriptor asks for `max_texture_dimension_2d: 8192`
//!   unconditionally, *including* on the `Gl` backend, where the guaranteed
//!   floor is 2048. A device reporting 4096 fails `request_device` outright
//!   and the whole app shows "OxiGIS failed to start" instead of degrading.
//!   [`CANVAS_ID`]'s runner therefore asks for
//!   `min(adapter.limits(), 8192)` — see `configure_wgpu`.
//! * **The fallback can be forced,** so it is testable on a machine that has
//!   WebGPU: `data-oxigis-backend="webgl"` on the canvas restricts the
//!   instance to `Backends::GL`, `"webgpu"` to `Backends::BROWSER_WEBGPU`.
//!   Absent — the default — both are offered and `wgpu` chooses.
//!
//! Practical consequences:
//!
//! * A WebGPU-capable browser (Chrome/Edge 113+, Safari 26+, Firefox 141+ on
//!   supported platforms) gets the WebGPU backend, which is what Phase 0's
//!   `oxigis-render` tile pipeline is written against.
//! * An older browser still runs, on WebGL2, with reduced capability
//!   (no compute shaders, tighter binding limits).
//! * A browser with neither leaves the canvas blank and the adapter request
//!   fails; the error is logged to the console and mirrored into the page's
//!   `#oxigis_status` banner, with the real message rather than a guess.
//! * WebGPU requires a **secure context**: `https://` or `http://localhost`.
//!   Serving the page from a LAN IP over plain HTTP silently drops you to the
//!   WebGL2 path. `serve.sh` binds localhost for exactly this reason.
//!
//! # Diagnostics
//!
//! `eframe::WebLogger` bridges the `log` crate to the devtools console, and
//! this shell's own diagnostics go through `log`. `oxigis-ui` and
//! `oxigis-render` instead diagnose through `tracing`, and the browser has no
//! `tracing` subscriber — installing one would need `tracing-subscriber`'s
//! `fmt` layer, which writes to `std::io` and is useless in a tab. The bridge
//! is a **manifest** line instead:
//!
//! ```toml
//! [target.'cfg(target_arch = "wasm32")'.dependencies]
//! tracing = { workspace = true, features = ["log"] }
//! ```
//!
//! `tracing`'s `log` feature emits a `log` record for every event **when no
//! `tracing` subscriber is active**, which is exactly this shell's situation,
//! so every `tracing::warn!` in `oxigis-ui` reaches the console through
//! `WebLogger`. It is declared under the `wasm32` table so native builds keep
//! `oxigis-desktop`'s real subscriber and gain no duplicate records.
//!
//! # Non-WASM builds
//!
//! Everything except [`VERSION`], [`CANVAS_ID`], [`range_rules`],
//! [`tile_drain`], [`permalink`] and [`activity`] is `cfg`-gated to
//! `target_arch = "wasm32"`, so `cargo check -p oxigis-web` on a host target
//! builds a small library exposing those. Native shells should use
//! `oxigis-desktop`.
//!
//! # Testing
//!
//! Host `cargo nextest run -p oxigis-web` runs [`range_rules`]' suite (the
//! `Content-Range` verification and validator-pinning rules), [`tile_drain`]'s,
//! [`permalink`]'s (the `#map=` fragment parser/formatter) and
//! [`activity`]'s (the in-flight fetch counter) — all pure computation and
//! therefore deliberately *not* `cfg`-gated. The `fetch()` and DOM glue around
//! them cannot run without a browser and is covered by
//! `cargo check -p oxigis-web --target wasm32-unknown-unknown`. `serve.sh
//! test` runs both; see it for why `wasm-pack test` executes nothing today.

#![forbid(unsafe_code)]

/// Crate version, re-exported so the page footer can display it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// DOM id of the canvas the web runner attaches to.
///
/// Must match the `id` attribute in `crates/oxigis-web/index.html`.
pub const CANVAS_ID: &str = "oxigis_canvas";

// `range_rules`, `tile_drain`, `permalink` and `activity` are deliberately
// NOT gated to `wasm32`: they are pure computation, and gating them along
// with the `fetch()`/DOM glue they serve is what left `range_rules`' and
// `tile_drain`'s tests compiled by nothing at all, before either existed.
// All four carry their own module docs.
pub mod activity;
pub mod permalink;
pub mod range_rules;
pub mod tile_drain;

#[cfg(target_arch = "wasm32")]
pub mod export_status;
#[cfg(target_arch = "wasm32")]
pub mod font_fetch;
#[cfg(target_arch = "wasm32")]
pub mod permalink_url;
#[cfg(target_arch = "wasm32")]
pub mod range_fetch;
#[cfg(target_arch = "wasm32")]
pub mod tile_fetch;
#[cfg(target_arch = "wasm32")]
pub mod timers;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use eframe::wasm_bindgen::{JsCast as _, JsValue, prelude::wasm_bindgen};
    use oxigis_core::{ArchiveFormat, Project, View};
    use oxigis_ui::{
        ArchiveLayerConfig, ArchiveProbe, ArchiveTileProvider, ArchiveTileTransport, BasemapConfig,
        BoxedTileProvider, BoxedVectorTileSource, CogLayerConfig, CogTileProvider, LocalLayerOp,
        MemoryRangeTransport, RangeTransport, VectorTileConfig, VectorTileProvider,
        XyzTileProvider, mbtiles::MbTilesReader,
    };

    /// Thin [`eframe::App`] adapter over [`oxigis_ui::OxigisApp`].
    ///
    /// The map is drawn by `oxigis-ui`'s `map_gpu` `egui_wgpu` callback on the
    /// browser's WebGPU (or WebGL2) device; this shell only forwards
    /// `eframe`'s render state into `OxigisApp::attach_gpu_map` each frame.
    struct WebShell {
        app: oxigis_ui::OxigisApp,
        /// Whether the label pass's primary font has been installed.
        ///
        /// Set only when [`oxigis_ui::map_gpu::set_label_fonts`] succeeds — it
        /// answers `false` on every frame before the `MapGpuState` exists,
        /// including the first — so the shell keeps trying until it lands.
        fonts_installed: bool,
        /// How many times the primary face has been offered; see
        /// [`MAX_LABEL_FONT_ATTEMPTS`].
        font_attempts: u32,
        /// Debounced write-back of the camera into the page's `#map=`
        /// fragment; see [`PermalinkSync`].
        permalink: PermalinkSync,
        /// The in-flight fetch count last written into `#oxigis_loading`, so
        /// [`update_loading_indicator`] is called only when it would change
        /// the DOM rather than on every frame that happens to paint.
        loading_shown: u32,
    }

    /// How many times the shell offers the primary label face before giving
    /// up.
    ///
    /// The offer costs a 431 KB copy of the bundled face, because the
    /// `map_gpu` seam takes owned bytes. One attempt is the normal case (the
    /// first frame that has a `MapGpuState`), and a face that fails to parse
    /// fails the same way every time — so retrying forever would allocate and
    /// discard 431 KB *per frame*, which is ~26 MB/s at 60 fps in a 32-bit
    /// linear heap, on the one path where the tab is already in trouble.
    const MAX_LABEL_FONT_ATTEMPTS: u32 = 4;

    /// How long the camera must sit still before the permalink commits it to
    /// the URL.
    ///
    /// Long enough that a drag in progress, or a scroll-zoom's burst of
    /// wheel events, writes nothing — `history.replaceState` on every one of
    /// those frames would be its own kind of jank, and some browsers have
    /// historically rate-limited how often it may be called from one tab.
    /// Short enough that "pan somewhere, let go, copy the address bar" reads
    /// as immediate to whoever is doing it.
    const PERMALINK_DEBOUNCE_MS: f64 = 600.0;

    /// Debounced write-back of the map camera into the page's `#map=`
    /// fragment.
    ///
    /// Every field holds the ROUNDED triple
    /// ([`super::permalink::round_view`]), never a raw camera reading:
    /// comparing a raw `f64` against a rounded one would read as "changed"
    /// on every frame from float jitter alone, and a debounce that never
    /// sees two equal readings in a row never fires. See `round_view`'s own
    /// docs for the same reasoning from the pure side.
    struct PermalinkSync {
        /// Rounded camera as of the last frame this was observed, for change
        /// detection.
        last_seen: (f64, f64, f64),
        /// Rounded camera last written to the URL. Compared against on every
        /// commit so an unchanged camera — the common case once the debounce
        /// has already fired once — is never written twice.
        last_committed: (f64, f64, f64),
        /// When [`Self::last_seen`] last changed; [`None`] once it has
        /// settled (and been committed, or matched what was already there).
        dirty_since: Option<f64>,
    }

    impl PermalinkSync {
        /// Starts already settled at `view`.
        ///
        /// Critical at startup, in both directions: a session that just
        /// restored a permalink must not immediately re-write the very
        /// fragment it was read from, and a session with no hash at all must
        /// not stamp a `#map=` fragment onto a clean URL [`PERMALINK_DEBOUNCE_MS`]
        /// after load. Starting with `last_seen == last_committed` and
        /// `dirty_since: None` makes the very first [`Self::observe`] call a
        /// no-op unless the camera has genuinely moved since — which a
        /// restore never has, by construction.
        fn settled_at(view: oxigis_render::MapView) -> Self {
            let center = view.center();
            let rounded = super::permalink::round_view(view.zoom(), center.lat, center.lon);
            Self {
                last_seen: rounded,
                last_committed: rounded,
                dirty_since: None,
            }
        }

        /// Call once per frame with the current camera. Commits to the URL
        /// [`PERMALINK_DEBOUNCE_MS`] after the camera stops changing.
        ///
        /// Uses [`egui::Context::request_repaint_after`] to guarantee a
        /// frame lands at the debounce deadline even if nothing else would
        /// repaint one — the map going idle right after the user lets go of
        /// a drag is the ordinary case, not an edge case, and without this
        /// the write-back would only happen whenever some unrelated repaint
        /// next occurred (a tile arriving, a click elsewhere), which on a
        /// fully-cached view might be "eventually" rather than "soon".
        fn observe(&mut self, view: oxigis_render::MapView, ctx: &egui::Context) {
            let center = view.center();
            let current = super::permalink::round_view(view.zoom(), center.lat, center.lon);
            if current != self.last_seen {
                self.last_seen = current;
                self.dirty_since = Some(now_ms());
                ctx.request_repaint_after(std::time::Duration::from_millis(
                    PERMALINK_DEBOUNCE_MS as u64,
                ));
                return;
            }
            let Some(since) = self.dirty_since else {
                return;
            };
            if now_ms() - since < PERMALINK_DEBOUNCE_MS {
                return;
            }
            self.dirty_since = None;
            if current == self.last_committed {
                return;
            }
            super::permalink_url::replace_location_hash(&super::permalink::format_hash(
                current.0, current.1, current.2,
            ));
            self.last_committed = current;
        }
    }

    impl WebShell {
        /// Builds the shell, restoring the camera from the page's `#map=`
        /// fragment when it has one.
        ///
        /// Read and applied here — before [`eframe::WebRunner::start`] has
        /// driven a single frame — because the permalink contract is
        /// "parsed on startup before first frame": a shell that only fixed
        /// the camera up on frame one would flash the default view first,
        /// visibly, on every restore.
        fn new() -> Self {
            let mut app = oxigis_ui::OxigisApp::new();
            // This shell reconciles through the N-layer tile stack, so the two
            // legacy single-slot seams must stop offering the layers the stack
            // now owns: `pending_raster_work` becomes basemap-only and
            // `pending_vector_work` goes quiet. Declaring it (rather than
            // merely not calling them) is what makes the two rules impossible
            // to get half-right — see `OxigisApp::set_tile_stack_shell`.
            app.set_tile_stack_shell(true);
            // `load_project`, not a bespoke camera setter: it is the one
            // existing seam that moves the map panel's camera outside of
            // live user input (File ▸ Open already exercises it end to end),
            // and it resets the undo stack — so a restored permalink does
            // not leave a phantom "undo" back to the pre-restore default
            // that File ▸ Open's own load never leaves either.
            if let Some(view) = super::permalink::parse_hash(&super::permalink_url::location_hash())
            {
                let mut project = Project::new("Untitled project");
                project.view = View {
                    center_lon: view.lon,
                    center_lat: view.lat,
                    zoom: view.zoom,
                };
                app.load_project(project);
                // `load_project` always reports "Project loaded." — accurate
                // for File ▸ Open, misleading here where nothing was opened;
                // this overwrites it with the truth.
                app.set_status("Restored the view from the page URL.");
            }
            // Seeded from the (possibly just-restored) camera, AFTER the
            // block above: a fresh session with no hash must not schedule an
            // immediate rewrite of a URL that never had one, and a restored
            // session must not immediately re-write the very fragment it was
            // just read from. See `PermalinkSync::settled_at`.
            let permalink = PermalinkSync::settled_at(app.map_view());
            Self {
                app,
                fonts_installed: false,
                font_attempts: 0,
                permalink,
                loading_shown: 0,
            }
        }

        /// Installs the primary label font, then drains any fallback the page
        /// asked for and the browser has since delivered.
        ///
        /// The primary face is the bundled Noto Sans Regular (Latin, OFL-1.1),
        /// compiled in because the map would be textless without it. The
        /// fallback is fetched at runtime and never embedded: see
        /// [`super::font_fetch`] for the 16 MB reason and how a page opts in.
        fn drive_label_fonts(
            &mut self,
            render_state: &eframe::egui_wgpu::RenderState,
            ctx: &egui::Context,
        ) {
            // So an asynchronously arriving font can wake an idle map; see
            // `font_fetch`'s module docs.
            super::font_fetch::remember_context(ctx);
            // `is_installed` FIRST: the 431 KB `to_vec()` below is the price of
            // asking, and `set_label_fonts` answers `false` without a
            // `MapGpuState` — which `attach_gpu_map_with` latches permanently
            // on an install failure. Asking only when there is something to
            // answer turns a per-frame allocation storm into nothing at all.
            if !self.fonts_installed
                && self.font_attempts < MAX_LABEL_FONT_ATTEMPTS
                && oxigis_ui::map_gpu::is_installed(render_state)
            {
                self.font_attempts = self.font_attempts.saturating_add(1);
                self.fonts_installed = oxigis_ui::map_gpu::set_label_fonts(
                    render_state,
                    oxifont_bundled::NOTO_SANS_REGULAR.to_vec(),
                    Vec::new(),
                );
                if self.fonts_installed {
                    log::info!("OxiGIS web shell: label font installed (Noto Sans, Latin)");
                } else if self.font_attempts >= MAX_LABEL_FONT_ATTEMPTS {
                    log::error!(
                        "OxiGIS web shell: the label font was refused {MAX_LABEL_FONT_ATTEMPTS} \
                         times; the map will draw without labels",
                    );
                }
            }
            if let Some(font) = super::font_fetch::take_pending_bold_font() {
                // The BOLD chain (print/text v1.4, D-W4): one face, since the
                // label engine appends the whole regular chain behind it.
                let bytes = font.len();
                super::font_fetch::retain_installed_bold_font(&font);
                if oxigis_ui::map_gpu::set_label_bold_fonts(render_state, vec![font]) {
                    log::info!(
                        "OxiGIS web shell: bold label font installed ({bytes} bytes); Bold \
                         labels draw bold",
                    );
                    ctx.request_repaint();
                } else {
                    log::warn!("OxiGIS web shell: the bold font arrived before the map; discarded");
                }
            }
            if let Some(font) = super::font_fetch::take_pending_font() {
                let bytes = font.len();
                // Retained so the PDF export can embed the same face the map
                // draws with — the label engine's copy is unreachable from
                // the export path.
                super::font_fetch::retain_installed_font(&font);
                if oxigis_ui::map_gpu::add_label_fallback_font(render_state, font) {
                    log::info!(
                        "OxiGIS web shell: label fallback font installed ({bytes} bytes); \
                         labels will re-shape",
                    );
                    // Every shaped label was just invalidated, so one more
                    // frame is needed to draw the new glyphs.
                    ctx.request_repaint();
                } else {
                    log::warn!(
                        "OxiGIS web shell: the fallback font arrived before the map; discarded",
                    );
                }
            }
        }

        /// The session data an archive-backed layer needs to be rebuilt.
        ///
        /// A browser's only local archive is the **bytes** a drop left behind
        /// (see `oxigis_ui::app::archive_io`), and an MBTiles image is indexed
        /// once and shared. Neither is reachable from a `spawn_local` future,
        /// which holds no `&OxigisApp`, so an export must resolve them on the
        /// frame that queues it.
        fn archive_data(&self, config: Option<&ArchiveLayerConfig>) -> ArchiveData {
            match config {
                Some(config) => ArchiveData {
                    bytes: self.app.archive_bytes(config.location()),
                    reader: self.app.mbtiles_reader(config.location()),
                },
                None => ArchiveData::default(),
            }
        }

        /// The session data one tile-stack plan needs, or `(None, None)` for a
        /// plan that names no archive.
        ///
        /// Reuses [`Self::archive_data`], which is the only place a browser's
        /// two archive sources — the bytes a drop left behind and a shared
        /// MBTiles index — are resolved.
        fn stack_archive_data(
            &self,
            plan: &oxigis_ui::TileLayerPlan,
        ) -> (
            Option<std::sync::Arc<[u8]>>,
            Option<std::sync::Arc<MbTilesReader>>,
        ) {
            let config = match &plan.source {
                oxigis_ui::TileLayerSource::RasterArchive(config) => Some(config),
                oxigis_ui::TileLayerSource::Vector(config) => config.archive.as_ref(),
                oxigis_ui::TileLayerSource::Cog(_) | oxigis_ui::TileLayerSource::Xyz(_) => None,
            };
            let data = self.archive_data(config);
            (data.bytes, data.reader)
        }

        /// Queues one PDF export as a browser task.
        ///
        /// Everything the task will need is captured **here**: the request
        /// snapshot, and the session-held bytes/reader of both the raster
        /// archive and the vector archive, which may name different files.
        /// Handing the task `None` for those is what used to make File ▸
        /// Export PDF a guaranteed no-op on every dropped archive.
        fn start_pdf_export(&self, request: oxigis_ui::print::PrintRequest, ctx: &egui::Context) {
            if !super::export_status::begin() {
                return;
            }
            let raster = self.archive_data(request.archive.as_ref());
            let vector = self.archive_data(
                request
                    .vector
                    .as_ref()
                    .and_then(|config| config.archive.as_ref()),
            );
            // One entry per stack layer, resolved HERE for the same reason the
            // two above are: a `spawn_local` future holds no `&OxigisApp`, so
            // an archive a drop left in memory is unreachable from it. Without
            // this an N-layer export would silently print without every
            // archive-backed layer it names.
            let stack: Vec<ArchiveData> = request
                .stack
                .iter()
                .map(|entry| {
                    self.archive_data(match &entry.source {
                        oxigis_ui::TileLayerSource::RasterArchive(config) => Some(config),
                        oxigis_ui::TileLayerSource::Vector(config) => config.archive.as_ref(),
                        oxigis_ui::TileLayerSource::Cog(_) | oxigis_ui::TileLayerSource::Xyz(_) => {
                            None
                        }
                    })
                })
                .collect();
            super::export_status::report("Exporting PDF\u{2026}");
            let ctx = ctx.clone();
            wasm_bindgen_futures::spawn_local(async move {
                match export_pdf(&request, &ctx, raster, vector, stack).await {
                    Ok(export) => {
                        let size = export.bytes.len();
                        let caveat = export.caveat;
                        let stamp = js_sys::Date::now() as u64;
                        let name = format!("oxigis-export-{stamp}.pdf");
                        match deliver_download(&name, "application/pdf", export.bytes) {
                            Ok(()) => super::export_status::report(match caveat {
                                Some(caveat) => {
                                    format!("PDF exported ({size} bytes), but {caveat}.")
                                }
                                None => {
                                    format!("PDF exported ({size} bytes); check your downloads.")
                                }
                            }),
                            Err(error) => super::export_status::report(format!(
                                "The PDF was built but the download failed: {}",
                                js_message(&error)
                            )),
                        }
                    }
                    Err(error) => {
                        super::export_status::report(format!("PDF export failed: {error}"));
                    }
                }
                super::export_status::finish();
            });
        }
    }

    /// The session-held data one archive-backed layer is rebuilt from.
    ///
    /// Both fields are [`None`] for a layer with no archive, and a `.pmtiles`
    /// carries `bytes` where a `.mbtiles` carries `reader` — they are never
    /// both meaningful at once, but resolving them together keeps the call
    /// sites symmetric with the frame-loop block above.
    #[derive(Default)]
    struct ArchiveData {
        /// A dropped archive's bytes, held for the session.
        bytes: Option<std::sync::Arc<[u8]>>,
        /// A dropped MBTiles image's shared index.
        reader: Option<std::sync::Arc<MbTilesReader>>,
    }

    impl eframe::App for WebShell {
        /// eframe 0.35 drives apps through `App::ui` (`App::update` is
        /// `#[deprecated = "Use Self::ui instead"]` and `App::ui` is the only
        /// required method). The `ui` handed in by
        /// `eframe::web::AppRunner` is the bare root [`egui::Ui`] of
        /// `Context::run_ui` — it is *not* wrapped in a `CentralPanel` — so
        /// `OxigisApp`'s own top/side/central panels, which it adds to the
        /// [`egui::Context`], get the full viewport and `map_rect()` stays
        /// accurate. The root `ui` is only used here to reach that context.
        fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
            // Idempotent; installs the tile renderer, with the `fetch()`-backed
            // XYZ basemap provider, on the first frame.
            let ctx = ui.ctx().clone();
            let config = self.app.basemap().clone();
            self.app.attach_gpu_map_with(frame.wgpu_render_state(), || {
                build_tile_provider(&config, &ctx)
            });
            // Label fonts: the compiled-in Latin face, plus whatever CJK
            // fallback the page opted into and `fetch()` has delivered.
            if let Some(render_state) = frame.wgpu_render_state() {
                self.drive_label_fonts(render_state, &ctx);
            }
            // Providers are DERIVED from the project (editing v1.3): an add,
            // a remove, an undo of either, and a project load all reach the
            // GPU through this one reconciliation block — byte-identical to
            // the desktop shell's apart from the logging macro. The work
            // stays offered until it is settled, so a frame without a render
            // state DEFERS an install instead of losing it.
            if let Some(render_state) = frame.wgpu_render_state() {
                // The BASEMAP, and only the basemap. This shell declared
                // `set_tile_stack_shell(true)`, so `pending_raster_work` offers
                // a basemap-only plan and `pending_vector_work` offers nothing:
                // every COG, archive, XYZ overlay and MVT source is a stack
                // entry now, and driving both seams for one layer would draw it
                // twice.
                if let Some(work) = self.app.pending_raster_work() {
                    let outcome = match build_tile_provider(&work.basemap, &ctx) {
                        Some(provider) => {
                            if oxigis_ui::map_gpu::replace_provider(render_state, provider) {
                                Ok(())
                            } else {
                                Err("the GPU map is not attached".to_string())
                            }
                        }
                        None => Err("the tile provider could not be built".to_string()),
                    };
                    self.app.settle_raster_work(work, outcome);
                }
                // The N-layer stack: what is installed is read back from the
                // map itself, so this is a pure diff with no second mirror to
                // go stale. One unit of work per frame, so a frame does at most
                // one provider build.
                let installed = oxigis_ui::map_gpu::installed_tile_stack(render_state);
                if let Some(work) = self.app.tile_stack_work(&installed) {
                    match work {
                        oxigis_ui::TileStackWork::Install(plan) => {
                            let (bytes, reader) = self.stack_archive_data(&plan);
                            let source = build_stack_source(&plan, &ctx, bytes, reader);
                            // Handed over even on failure: the refusal occupies
                            // the slot, which is what stops the plan being
                            // offered once per frame for ever.
                            if let Err(reason) =
                                oxigis_ui::map_gpu::install_tile_layer(render_state, plan, source)
                            {
                                self.app.set_status(reason);
                            }
                        }
                        oxigis_ui::TileStackWork::Remove(layers) => {
                            let _ = oxigis_ui::map_gpu::remove_tile_layers(render_state, &layers);
                        }
                        // Rebuilds nothing: a drag in the layer panel must not
                        // re-fetch a single tile.
                        oxigis_ui::TileStackWork::Reorder(order) => {
                            let _ = oxigis_ui::map_gpu::reorder_tile_layers(render_state, &order);
                        }
                    }
                }
                // Unconditional and cheap: an opacity is an instance tint, so
                // following a slider costs neither a texture nor a
                // tessellation.
                let app = &self.app;
                oxigis_ui::map_gpu::sync_tile_layer_opacities(render_state, |id| {
                    app.tile_layer_opacity(id)
                });
                // The banner's half of the stack: refusals are GPU state, so
                // the shell is the only place they can be read from. It must
                // stay ABOVE `self.app.ui(ui)` — the panel reads the banner
                // while it draws, so writing this afterwards would show the
                // previous frame's refusals.
                self.app
                    .set_tile_layer_refusals(oxigis_ui::map_gpu::tile_layer_refusals(render_state));
                if self.app.take_tile_layer_retry() {
                    let dropped = oxigis_ui::map_gpu::retry_refused_tile_layers(render_state);
                    if dropped > 0 {
                        log::info!("OxiGIS web shell: {dropped} refused stack entries re-offered",);
                    }
                }
            }
            // A tile archive the user asked for: `oxigis-ui` owns no transport,
            // so the shell builds one and hands the probe back. The layer is
            // created by `poll_archive_probe` once the header lands.
            if self.app.take_pending_archive_pick() {
                // No filesystem, and therefore no dialog: the drop gesture is
                // the browser's whole local-archive story.
                self.app.set_status(
                    "A browser has no file dialog: drop a .pmtiles or .mbtiles file onto the \
                     map to open it.",
                );
            }
            if let Some(request) = self.app.take_pending_archive_probe() {
                let config = ArchiveLayerConfig::new(request.archive.clone(), request.format);
                let bytes = self.app.archive_bytes(config.location());
                match build_archive_transport(&config, bytes) {
                    Some(transport) => {
                        self.app.attach_archive_probe(ArchiveProbe::start(
                            config.location().to_owned(),
                            request.format,
                            &ctx,
                            transport,
                        ));
                    }
                    None => self.app.set_status(
                        "The browser has no filesystem, so a local archive has to be dropped \
                         onto the map rather than opened by path.",
                    ),
                }
            }
            let _created = self.app.poll_archive_probe();
            // A queued PDF export runs as a browser task: the tile fetches are
            // `fetch()` callbacks, so unlike the desktop shell this must not
            // block the frame — the finished bytes arrive as a download.
            if let Some(mut request) = self.app.take_pending_print() {
                // `wasm32-unknown-unknown` has NO clock in `std` —
                // `SystemTime::now()` panics there — so the page stamps the
                // creation instant itself from the browser's clock. Without
                // this the exported /Info dictionary carries /Title, /Creator
                // and /Producer but no /CreationDate at all.
                request.options.creation_epoch_secs = Some((js_sys::Date::now() / 1000.0) as i64);
                self.start_pdf_export(request, &ctx);
            }
            // The three text-file seams. A browser has no filesystem, so a
            // DOWNLOAD is the write: there is no path to report back and no way
            // to learn where the file landed, which is why each confirms as
            // soon as the anchor has been clicked.
            if let Some(request) = self.app.take_pending_processing_save() {
                let name = format!("{}.geojson", request.name);
                match deliver_download(&name, "application/geo+json", request.content.into_bytes())
                {
                    Ok(()) => {
                        let features = request.features;
                        let plural = if features == 1 { "feature" } else { "features" };
                        self.app.set_status(format!(
                            "Downloaded {name} ({features} {plural}); check your downloads."
                        ));
                    }
                    Err(error) => self.app.set_status(format!(
                        "The result was built but the download failed: {}",
                        js_message(&error),
                    )),
                }
            }
            if let Some(request) = self.app.take_pending_export() {
                match deliver_download(
                    &request.suggested_file_name,
                    request.kind.mime_type(),
                    request.content_bytes(),
                ) {
                    Ok(()) => self
                        .app
                        .confirm_export_written(std::path::Path::new(&request.suggested_file_name)),
                    Err(error) => self.app.report_export_failed(&js_message(&error)),
                }
            }
            // NOT drained: `take_pending_project_save` only ever yields after
            // `set_native_project_io(true)`, and that ONE flag governs File ▸
            // Open as well. Turning it on here would queue a
            // `ProjectOpenRequest` this shell cannot satisfy — a browser has no
            // file dialog — and would replace the working paste box with a dead
            // menu item. A real browser Open needs `<input type="file">` plus
            // `FileReader` plumbing into this frame loop; until that exists,
            // File ▸ Save stays the copy-JSON modal, which works today.
            // Whatever that task has to say since the last frame. Drained
            // unconditionally, because the task has no `&mut OxigisApp` and
            // this status line is the only place its progress, its refusals
            // and its success can appear.
            if let Some(message) = super::export_status::take_message() {
                self.app.set_status(message);
            }
            // The page's loading indicator: how many tile/COG/font fetches
            // `crate::activity::track` currently has open. Written only when
            // it changed, so an idle map with the indicator already hidden
            // does not touch the DOM every frame it happens to repaint for
            // an unrelated reason (a drag, a blinking cursor).
            let in_flight = super::activity::count();
            if in_flight != self.loading_shown {
                update_loading_indicator(in_flight);
                self.loading_shown = in_flight;
            }
            // Local vector data dropped onto the canvas (GeoJSON, or a
            // multi-file shapefile set — `oxigis-ui` groups the drop by file
            // stem) or pasted into the layer panel. The browser hands `eframe`
            // the file *bytes*, so unlike the desktop shell there is never a
            // path to read — `oxigis-ui` has already parsed the data and only
            // the GPU work is left.
            apply_local_ops(&mut self.app, frame.wgpu_render_state());
            self.app.ui(ui);
            // AFTER `self.app.ui(ui)`, not before: that call is what applies
            // this frame's drag/scroll input to the camera, so reading
            // `map_view()` any earlier would observe last frame's position
            // and lag the permalink by one frame.
            self.permalink.observe(self.app.map_view(), &ctx);
        }
    }

    /// Performs this frame's local-vector work against the GPU map.
    ///
    /// Drained unconditionally — a frame with no `wgpu` render state logs and
    /// discards, rather than letting the queue grow — and applied in order,
    /// since an `Add` must precede the edits that refer to it.
    fn apply_local_ops(
        app: &mut oxigis_ui::OxigisApp,
        render_state: Option<&eframe::egui_wgpu::RenderState>,
    ) {
        for op in app.take_pending_local_ops() {
            let Some(render_state) = render_state else {
                log::error!(
                    "OxiGIS web shell: the GPU map is not attached, so a local layer op was \
                     discarded",
                );
                continue;
            };
            let applied = match op {
                LocalLayerOp::Add(id, layer) => {
                    oxigis_ui::map_gpu::add_local_vector_layer(render_state, id, *layer)
                }
                LocalLayerOp::Remove(id) => {
                    oxigis_ui::map_gpu::remove_local_vector_layer(render_state, id).is_some()
                }
                LocalLayerOp::SetVisibility(id, visible) => {
                    oxigis_ui::map_gpu::set_local_layer_visibility(render_state, id, visible)
                }
                LocalLayerOp::SetOpacity(id, opacity) => {
                    oxigis_ui::map_gpu::set_local_layer_opacity(render_state, id, opacity)
                }
                LocalLayerOp::SetStyle(id, style) => {
                    oxigis_ui::map_gpu::set_local_layer_style(render_state, id, style)
                }
                LocalLayerOp::Reorder(order) => {
                    oxigis_ui::map_gpu::reorder_local_vector_layers(render_state, &order)
                }
                LocalLayerOp::Clear => oxigis_ui::map_gpu::clear_local_vector_layers(render_state),
            };
            if !applied {
                log::debug!("OxiGIS web shell: a local layer op changed nothing");
            }
        }
    }

    /// Builds the browser-backed XYZ tile provider.
    ///
    /// Returns [`None`] — which makes `attach_gpu_map_with` keep the synthetic
    /// checkerboard, so the map is still usable — if the configured URL
    /// template is not a valid `{z}/{x}/{y}` template.
    fn build_tile_provider(
        config: &BasemapConfig,
        ctx: &egui::Context,
    ) -> Option<BoxedTileProvider> {
        let transport = super::tile_fetch::FetchTileTransport::new();
        match XyzTileProvider::new(config, ctx, Box::new(transport)) {
            Ok(provider) => {
                log::info!("OxiGIS web shell: XYZ basemap {}", config.url_template);
                Some(Box::new(provider))
            }
            Err(error) => {
                log::error!("OxiGIS web shell: invalid basemap URL template: {error}");
                None
            }
        }
    }

    /// Builds a COG provider, optionally drawing over a fresh XYZ basemap.
    ///
    /// `basemap` is [`Some`] only for the **single-slot** path, which in this
    /// shell is the PDF export: there the COG and the basemap under it are
    /// CPU-composited into one image.
    ///
    /// For a **stack** entry it is [`None`], and that is load-bearing: with no
    /// base, `CogTileProvider::tile` returns the layer's own alpha-carrying
    /// pixels, which is exactly what the stack composites. Blending the basemap
    /// in here would put a copy of it in every layer tile, and the entry's
    /// opacity tint would then fade the basemap along with the layer.
    ///
    /// Returns [`None`] only if the provider's own caches cannot be built, which
    /// the compile-time bounds make unreachable; the basemap underneath is
    /// optional, so a bad basemap template just means no basemap.
    fn build_cog_provider(
        cog: &CogLayerConfig,
        basemap: Option<&BasemapConfig>,
        ctx: &egui::Context,
    ) -> Option<BoxedTileProvider> {
        let transport = super::range_fetch::FetchRangeTransport::new();
        let provider = match CogTileProvider::new(cog, ctx, Box::new(transport)) {
            Ok(provider) => provider,
            Err(error) => {
                log::error!("OxiGIS web shell: could not build the COG provider: {error}");
                return None;
            }
        };
        log::info!("OxiGIS web shell: COG layer {}", cog.url);
        match basemap.and_then(|config| build_tile_provider(config, ctx)) {
            Some(base) => Some(Box::new(provider.with_base(base))),
            None => Some(Box::new(provider)),
        }
    }

    /// Builds the range transport a tile archive needs in a browser.
    ///
    /// Two answers, and the second is a refusal worth being explicit about:
    ///
    /// * bytes the app is holding for the session (a dropped `.pmtiles`) —
    ///   [`MemoryRangeTransport`];
    /// * a URL — [`super::range_fetch::FetchRangeTransport`], with the same
    ///   CORS preflight requirement a COG has (`Access-Control-Allow-Headers:
    ///   Range`, and `Content-Range` exposed).
    ///
    /// A **path** with no bytes has no answer at all: a page cannot open a file
    /// by name, so the archive has to be dropped again. Returning [`None`] here
    /// is what turns that into a status line instead of a blank map.
    fn build_archive_transport(
        config: &ArchiveLayerConfig,
        bytes: Option<std::sync::Arc<[u8]>>,
    ) -> Option<Box<dyn RangeTransport>> {
        if let Some(bytes) = bytes {
            return Some(Box::new(MemoryRangeTransport::from_shared(bytes)));
        }
        if config.is_local() {
            log::warn!(
                "OxiGIS web shell: {} is a local archive and the browser has no filesystem; \
                 drop the file again to restore it",
                config.location(),
            );
            return None;
        }
        Some(Box::new(super::range_fetch::FetchRangeTransport::new()))
    }

    /// Builds a raster tile-archive provider, optionally drawing over a fresh
    /// XYZ basemap.
    ///
    /// The archive twin of [`build_cog_provider`], including the `Option` base
    /// and the reason for it.
    fn build_archive_provider(
        config: &ArchiveLayerConfig,
        basemap: Option<&BasemapConfig>,
        ctx: &egui::Context,
        bytes: Option<std::sync::Arc<[u8]>>,
        reader: Option<std::sync::Arc<MbTilesReader>>,
    ) -> Option<BoxedTileProvider> {
        let built = match config.format {
            ArchiveFormat::MbTiles => {
                let Some(reader) = reader else {
                    log::warn!(
                        "OxiGIS web shell: {} is an MBTiles archive whose bytes this session no \
                         longer holds; drop the file again",
                        config.location(),
                    );
                    return None;
                };
                ArchiveTileProvider::mbtiles(config.location().to_owned(), reader, ctx)
            }
            ArchiveFormat::PmTiles => {
                let transport = build_archive_transport(config, bytes)?;
                ArchiveTileProvider::pmtiles(config.location().to_owned(), ctx, transport)
            }
        };
        let provider = match built {
            Ok(provider) => provider,
            Err(error) => {
                log::error!("OxiGIS web shell: could not build the archive provider: {error}");
                return None;
            }
        };
        log::info!(
            "OxiGIS web shell: raster tile archive {}",
            config.location()
        );
        match basemap.and_then(|config| build_tile_provider(config, ctx)) {
            Some(base) => Some(Box::new(provider.with_base(base))),
            None => Some(Box::new(provider)),
        }
    }

    /// Builds the browser-backed MVT vector-tile source.
    ///
    /// Reuses [`super::tile_fetch::FetchTileTransport`] — a `.pbf` endpoint is an
    /// ordinary XYZ service — so no new browser capability is involved. Returns
    /// [`None`], leaving the map raster-only, if the URL template is not a valid
    /// `{z}/{x}/{y}` template.
    fn build_vector_source(
        config: &VectorTileConfig,
        ctx: &egui::Context,
        bytes: Option<std::sync::Arc<[u8]>>,
        reader: Option<std::sync::Arc<MbTilesReader>>,
    ) -> Option<BoxedVectorTileSource> {
        // Two arms, one provider: an archive-backed config swaps the browser's
        // fetch transport for one that resolves tiles inside a single file, and
        // the rest of the vector path is untouched.
        let transport: Box<dyn oxigis_ui::TileTransport> = match config.archive.as_ref() {
            Some(archive) if archive.format == ArchiveFormat::MbTiles => Box::new(
                ArchiveTileTransport::mbtiles(archive.location().to_owned(), reader?),
            ),
            Some(archive) => Box::new(ArchiveTileTransport::pmtiles(
                archive.location().to_owned(),
                build_archive_transport(archive, bytes)?,
            )),
            None => Box::new(super::tile_fetch::FetchTileTransport::new()),
        };
        match VectorTileProvider::new(config, ctx, transport) {
            Ok(provider) => {
                log::info!("OxiGIS web shell: vector tiles {}", config.url_template);
                Some(Box::new(provider))
            }
            Err(error) => {
                log::error!("OxiGIS web shell: invalid vector tile URL template: {error}");
                None
            }
        }
    }

    /// Builds one entry of the N-layer tile stack.
    ///
    /// Every arm is deliberately **base-less** (see [`build_cog_provider`]): a
    /// stack entry contributes its own alpha-carrying pixels and the basemap is
    /// its own pass underneath.
    ///
    /// # Errors
    ///
    /// The reason the entry cannot draw, ready for
    /// `map_gpu::install_tile_layer` to memoize against the entry so it is not
    /// rebuilt once per frame for ever.
    fn build_stack_source(
        plan: &oxigis_ui::TileLayerPlan,
        ctx: &egui::Context,
        bytes: Option<std::sync::Arc<[u8]>>,
        reader: Option<std::sync::Arc<MbTilesReader>>,
    ) -> Result<oxigis_ui::TileLayerGpuSource, String> {
        match &plan.source {
            oxigis_ui::TileLayerSource::Cog(config) => build_cog_provider(config, None, ctx)
                .map(oxigis_ui::TileLayerGpuSource::Raster)
                .ok_or_else(|| "the COG provider could not be built".to_owned()),
            oxigis_ui::TileLayerSource::RasterArchive(config) => {
                build_archive_provider(config, None, ctx, bytes, reader)
                    .map(oxigis_ui::TileLayerGpuSource::Raster)
                    .ok_or_else(|| {
                        "the tile archive could not be opened \u{2014} a browser has no \
                         filesystem, so drop the file again"
                            .to_owned()
                    })
            }
            // An XYZ layer that is not the promoted basemap draws as an
            // ordinary overlay. `build_tile_provider` never had a base to skip.
            oxigis_ui::TileLayerSource::Xyz(config) => build_tile_provider(config, ctx)
                .map(oxigis_ui::TileLayerGpuSource::Raster)
                .ok_or_else(|| "the XYZ tile provider could not be built".to_owned()),
            oxigis_ui::TileLayerSource::Vector(config) => {
                build_vector_source(config, ctx, bytes, reader)
                    .map(oxigis_ui::TileLayerGpuSource::Vector)
                    .ok_or_else(|| "the vector tile source could not be built".to_owned())
            }
        }
    }

    /// How long one export phase waits for its tiles, in milliseconds.
    ///
    /// Wall-clock rather than a poll count: a poll costs one `setTimeout` plus
    /// however long the frame it lands on takes, so "150 polls" is a budget
    /// only on an idle tab.
    ///
    /// A phase ends when the budget has run out **and** the poll that noticed
    /// collected nothing: a poll that is still landing tiles buys one more.
    /// That is a small grace, not a guarantee — with sixteen requests in
    /// flight an empty 100 ms poll is ordinary — but it stops a page that was
    /// one tile from done being truncated on the tick. Termination is not at
    /// the budget's mercy either way: progress is bounded by
    /// [`MAX_PRINT_TILES`].
    const PRINT_TILE_BUDGET_MS: f64 = 15_000.0;

    /// Gap between two tile-drain passes.
    const PRINT_POLL_MS: i32 = 100;

    /// Most tiles one page may require before the export refuses it.
    ///
    /// The export holds every drained tile until it is pasted, and a decoded
    /// 256x256 RGBA tile is 256 KB — so this ceiling is really a memory bound:
    /// 512 tiles is ~134 MB of pixels beside the page's own RGB buffer, which
    /// is about as much as a 32-bit linear heap will take beside the tile
    /// caches, the meshes and the label atlas. Every choice the export dialog
    /// offers is well under it (A3 at 288 dpi, the largest, needs ~250), so
    /// this refuses only a page geometry that arrived from somewhere else.
    const MAX_PRINT_TILES: usize = 512;

    /// Milliseconds since the epoch, as the browser reports them.
    fn now_ms() -> f64 {
        js_sys::Date::now()
    }

    /// One finished export: the document, plus anything the reader has to be
    /// told about what is on it.
    struct ExportedPdf {
        /// The assembled PDF.
        bytes: Vec<u8>,
        /// What the page is missing, when it is missing something. A page with
        /// gray plates or an absent overlay looks finished, so saying so is
        /// the difference between a known-incomplete document and a wrong one.
        caveat: Option<String>,
    }

    /// Collects every required raster tile out of `provider`.
    ///
    /// The polling half of [`super::tile_drain`], which carries the reason
    /// this is a *drain* and not a wait: past the provider's 64-tile ready
    /// cache the "is everything resident?" test can never answer yes, so the
    /// old loop spent its budget re-fetching evicted tiles and then printed a
    /// mostly-gray page. Bounded by wall clock rather than by poll count, so
    /// the budget is a budget on a busy tab too.
    async fn drain_raster_tiles(
        provider: &BoxedTileProvider,
        required: &[oxigis_render::TileId],
    ) -> super::tile_drain::TileDrain<oxigis_render::DecodedTile> {
        let mut drain = super::tile_drain::TileDrain::new();
        let deadline = now_ms() + PRINT_TILE_BUDGET_MS;
        loop {
            let held = drain.len();
            // `tile()` is also what enqueues the fetch, so the first pass
            // primes the whole set and later passes chase the stragglers.
            let missing = drain.pass(required, &mut |tile| provider.tile(tile));
            if missing == 0 || (now_ms() >= deadline && drain.len() == held) {
                return drain;
            }
            super::export_status::report(super::export_status::tile_progress(
                "basemap",
                drain.len(),
                required.len(),
            ));
            super::timers::sleep_ms(PRINT_POLL_MS).await;
        }
    }

    /// Collects every required vector tile out of `source`.
    ///
    /// Same rule and the same reason as [`drain_raster_tiles`]:
    /// `VectorTileProvider`'s decoded cache holds 128 tiles, so a page needing
    /// more never sees them all at once. `mesh()` is what drives the fetch
    /// loop; `decoded()` is what the print overlay draws from.
    async fn drain_vector_tiles(
        source: &BoxedVectorTileSource,
        required: &[oxigis_render::TileId],
        compose: oxigis_render::MapView,
    ) -> super::tile_drain::TileDrain<std::sync::Arc<oxigis_render::mvt::VectorTile>> {
        let mut drain = super::tile_drain::TileDrain::new();
        let deadline = now_ms() + PRINT_TILE_BUDGET_MS;
        loop {
            let _ = source.begin_frame(compose);
            let held = drain.len();
            let missing = drain.pass(required, &mut |tile| {
                let decoded = source.decoded(tile);
                if decoded.is_none() {
                    // `decoded()` only reads; `mesh()` is what starts the
                    // fetch and the decode behind it.
                    let _ = source.mesh(tile);
                }
                decoded
            });
            if missing == 0 || (now_ms() >= deadline && drain.len() == held) {
                break;
            }
            super::export_status::report(super::export_status::tile_progress(
                "vector",
                drain.len(),
                required.len(),
            ));
            super::timers::sleep_ms(PRINT_POLL_MS).await;
        }
        drain
    }

    /// Runs one queued PDF export: fresh provider, tile fetches, composition,
    /// assembly. Async because on the web the fetches only progress while this
    /// task yields back to the browser.
    ///
    /// `raster` and `vector` carry the session-held archive data the frame
    /// loop resolved; without them an archive-backed layer cannot be rebuilt
    /// here at all, because a browser's local archive exists only as the bytes
    /// of a drop.
    async fn export_pdf(
        request: &oxigis_ui::print::PrintRequest,
        ctx: &egui::Context,
        raster: ArchiveData,
        vector: ArchiveData,
        stack: Vec<ArchiveData>,
    ) -> Result<ExportedPdf, String> {
        // The single-slot path stays EXACTLY as it was for every project whose
        // stack the three legacy fields already describe. Only a stack those
        // fields would truncate — two rasters, two tilesets, or an XYZ overlay
        // they cannot name at all — takes the composed path below.
        let legacy = request.stack_fits_legacy_slots();
        // The export composites on the CPU into ONE image, so here the base
        // genuinely belongs underneath the layer — the opposite of the stack's
        // rule, and the reason `build_cog_provider` takes an `Option`.
        let provider: BoxedTileProvider = match (
            request.archive.as_ref().filter(|_| legacy),
            request.cog.as_ref().filter(|_| legacy),
        ) {
            (Some(archive), _) => {
                build_archive_provider(
                    archive,
                    Some(&request.basemap),
                    ctx,
                    raster.bytes,
                    raster.reader,
                )
                // Named, because this is now the *residual* archive failure: the
                // session no longer holds the bytes — evicted by
                // `MAX_SESSION_ARCHIVE_BYTES`, or the tab was reloaded — and the
                // generic refusal below would leave the reader with nothing to act
                // on.
                .ok_or_else(|| {
                    format!(
                        "{} could not be rebuilt for the export; a dropped archive has to be \
                     dropped again after a reload",
                        archive.location(),
                    )
                })?
            }
            (None, Some(cog)) => build_cog_provider(cog, Some(&request.basemap), ctx)
                .ok_or_else(|| "no tile provider could be built for the export".to_string())?,
            (None, None) => build_tile_provider(&request.basemap, ctx)
                .ok_or_else(|| "no tile provider could be built for the export".to_string())?,
        };

        // The embedded-font chain: the compiled-in Latin face, then the
        // print-only face the page configured (fetched now, on the export
        // path — the await is structurally free beside the tile polls), then
        // the label fallback — skipped when it IS the print face, so the
        // same megabytes are not embedded twice. The print face is the ONE
        // `PrintOnly` entry: the screen never rasterises it, so a variable
        // face may be normalised to its nearest-Regular instance (a
        // thin-default VF would otherwise print hairlines); the faces the
        // labels share stay `ScreenShared` and are never instanced.
        use oxigis_ui::print::{FaceRole, PrintFace};
        let mut chain = vec![PrintFace::regular(
            oxifont_bundled::NOTO_SANS_REGULAR.to_vec(),
            FaceRole::ScreenShared,
        )];
        // The faces are held as `Arc<[u8]>` so asking for one costs nothing;
        // `to_vec` here is the one copy the `PrintFace` seam still requires.
        let print_face = super::font_fetch::print_font().await;
        let print_url = super::font_fetch::print_font_url();
        if let Some(face) = print_face {
            chain.push(PrintFace::regular(face.to_vec(), FaceRole::PrintOnly));
        }
        if let Some(fallback) = super::font_fetch::installed_font()
            && (print_url.is_none() || print_url != super::font_fetch::fallback_url())
        {
            chain.push(PrintFace::regular(
                fallback.to_vec(),
                FaceRole::ScreenShared,
            ));
        }
        // The BOLD slots (print/text v1.4, D-W4/D-W5), same two sources and
        // the same de-duplication: the export-only bold face is the ONE
        // entry a variable face may be instanced from (PrintOnly), the
        // screen's bold face stays ScreenShared so page and map keep the
        // same ink.
        let print_bold = super::font_fetch::print_bold_font().await;
        let print_bold_url = super::font_fetch::print_bold_font_url();
        if let Some(face) = print_bold {
            chain.push(PrintFace::bold(face.to_vec(), FaceRole::PrintOnly));
        }
        if let Some(bold) = super::font_fetch::installed_bold_font()
            && (print_bold_url.is_none() || print_bold_url != super::font_fetch::bold_font_url())
        {
            chain.push(PrintFace::bold(bold.to_vec(), FaceRole::ScreenShared));
        }
        let fonts = oxigis_ui::print::PrintFonts::with_roles(chain);

        let map_box = oxigis_ui::print::map_box(&request.options);
        let out_px = oxigis_ui::print::raster_size_px(&map_box, &request.options);
        let compose = oxigis_ui::print::compose_view(request.view, out_px);
        let required = oxigis_ui::print::required_tiles(&compose);
        if required.len() > MAX_PRINT_TILES {
            return Err(format!(
                "this page needs {} basemap tiles, past the {MAX_PRINT_TILES} a browser tab can \
                 hold at once; choose a smaller page or a lower resolution",
                required.len(),
            ));
        }
        // Carried out of here rather than reported in place: every later
        // `report` overwrites the last, so a shortfall announced mid-run would
        // be gone by the time the export says it succeeded — which is the one
        // moment the reader is looking.
        let mut caveats: Vec<String> = Vec::new();
        let mut tiles = drain_raster_tiles(&provider, &required).await;
        let absent = tiles.absent(&required);
        if absent > 0 {
            // Said out loud rather than printed silently: a gray plate in the
            // middle of a map is not something a reader can distinguish from
            // the data.
            caveats.push(format!(
                "{absent} of {} basemap tiles never arrived and print gray",
                required.len(),
            ));
        }
        // Streamed vector tiles, when a vector-tile layer is active: the
        // same provider the screen uses — `mesh()` drives the fetch loop,
        // `decoded()` collects the typed tiles the print overlay draws from.
        let mut vector_tiles = Vec::new();
        if let Some(config) = request.vector.as_ref().filter(|_| legacy) {
            match build_vector_source(config, ctx, vector.bytes, vector.reader) {
                Some(source) => {
                    let drained = drain_vector_tiles(&source, &required, compose).await;
                    let absent = drained.absent(&required);
                    vector_tiles = drained.into_ordered(&required);
                    if absent > 0 {
                        caveats.push(format!(
                            "{absent} of {} vector tiles never arrived and are missing from the \
                             page",
                            required.len(),
                        ));
                    }
                }
                // Refused rather than skipped. Falling through here used to
                // assemble the page with the overlay silently absent — a
                // plausible-looking map, missing a whole layer, that would be
                // printed and distributed. A page that is wrong in a way its
                // reader cannot see is worse than no page.
                None => {
                    let name = config
                        .archive
                        .as_ref()
                        .map_or(config.url_template.as_str(), |archive| archive.location());
                    return Err(format!(
                        "the vector layer {name} could not be rebuilt for the export; a dropped \
                         archive has to be dropped again after a reload",
                    ));
                }
            }
        }

        // `take`, not a lookup: `compose_map_rgb` asks for each tile exactly
        // once, so handing over ownership frees each tile as it is pasted
        // instead of holding the whole page's pixels twice.
        let mut rgb = oxigis_ui::print::compose_map_rgb(&compose, &mut |tile| tiles.take(tile));
        // The composed path (compositing v1.6): one provider per stack entry,
        // bottom-up. Rasters are pasted into the image just built — at their
        // own opacity, base-less, because the basemap is already under them —
        // and vector tilesets travel on as tiles, to be painted as real PDF
        // paths in stack order. `stack_tiles` is index-aligned with
        // `request.stack`, so a raster entry keeps an empty list rather than
        // shifting the ones after it.
        let mut stack_tiles: Vec<
            Vec<(
                oxigis_render::TileId,
                std::sync::Arc<oxigis_render::mvt::VectorTile>,
            )>,
        > = Vec::new();
        if !legacy {
            let mut stack = stack;
            for (index, entry) in request.stack.iter().enumerate() {
                let data = stack
                    .get_mut(index)
                    .map(core::mem::take)
                    .unwrap_or_default();
                match &entry.source {
                    oxigis_ui::TileLayerSource::Vector(config) => {
                        match build_vector_source(config, ctx, data.bytes, data.reader) {
                            Some(source) => {
                                let drained = drain_vector_tiles(&source, &required, compose).await;
                                let absent = drained.absent(&required);
                                stack_tiles.push(drained.into_ordered(&required));
                                if absent > 0 {
                                    caveats.push(format!(
                                        "{absent} of {} vector tiles never arrived and are \
                                         missing from the page",
                                        required.len(),
                                    ));
                                }
                            }
                            // Named, not skipped: a page that is missing a
                            // whole layer its reader cannot see is worse than
                            // no page — the same rule the single-slot path
                            // states below.
                            None => {
                                let name = config
                                    .archive
                                    .as_ref()
                                    .map_or(config.url_template.as_str(), |archive| {
                                        archive.location()
                                    });
                                return Err(format!(
                                    "the vector layer {name} could not be rebuilt for the \
                                     export; a dropped archive has to be dropped again after a \
                                     reload",
                                ));
                            }
                        }
                    }
                    source => {
                        stack_tiles.push(Vec::new());
                        let built = match source {
                            oxigis_ui::TileLayerSource::Cog(config) => {
                                build_cog_provider(config, None, ctx)
                            }
                            oxigis_ui::TileLayerSource::RasterArchive(config) => {
                                build_archive_provider(config, None, ctx, data.bytes, data.reader)
                            }
                            oxigis_ui::TileLayerSource::Xyz(config) => {
                                build_tile_provider(config, ctx)
                            }
                            oxigis_ui::TileLayerSource::Vector(_) => None,
                        };
                        let Some(provider) = built else {
                            caveats.push(
                                "a tiled layer could not be rebuilt for the export and is \
                                 missing from the page"
                                    .to_string(),
                            );
                            continue;
                        };
                        let mut layer_tiles = drain_raster_tiles(&provider, &required).await;
                        let absent = layer_tiles.absent(&required);
                        oxigis_ui::print::overlay_map_rgb(
                            &compose,
                            &mut rgb,
                            entry.opacity,
                            &mut |tile| layer_tiles.take(tile),
                        );
                        if absent > 0 {
                            caveats.push(format!(
                                "{absent} of {} tiles of one layer never arrived",
                                required.len(),
                            ));
                        }
                    }
                }
            }
        }
        let bytes = oxigis_ui::print::pdf_document_with(
            request,
            &compose,
            &rgb,
            out_px,
            &fonts,
            &oxigis_ui::print::PrintVectorTiles {
                single: &vector_tiles,
                stack: &stack_tiles,
            },
        )?;
        Ok(ExportedPdf {
            bytes,
            caveat: (!caveats.is_empty()).then(|| caveats.join("; ")),
        })
    }

    /// How long the object URL of a finished download is kept alive.
    ///
    /// Revoking on the statement after `click()` is a race the spec does not
    /// settle: the click has *initiated* a download, not fetched it, and
    /// browsers have historically differed on whether an already-revoked blob
    /// still resolves. A few seconds settles that with certainty, and no more
    /// — the blob is a full copy of the document, and this function's whole
    /// point is not keeping copies of it alive.
    const DOWNLOAD_URL_LIFETIME_MS: i32 = 5_000;

    /// Hands `bytes` to the browser as a download named `file_name`: Blob →
    /// object URL → a synthetic `<a download>` click, the one file-delivery
    /// path a page gets without a save-dialog permission.
    ///
    /// **A download IS the write**, here: the browser has no filesystem for the
    /// page to write to and no way to learn where the file landed, so the
    /// caller confirms as soon as this returns `Ok`.
    ///
    /// Takes the document **by value** so the Rust copy is dropped before the
    /// `Blob` exists: the typed array and the `Blob` are each a full copy of a
    /// document that scales with page size and dpi, and three of them live at
    /// once in a 32-bit address space otherwise. (Two is the floor without
    /// `Uint8Array::view`, which is `unsafe` and this crate forbids it.)
    ///
    /// # Errors
    ///
    /// Whatever the DOM refused: no `Blob`, no object URL, no `document`, or an
    /// `<a>` that did not build.
    fn deliver_download(file_name: &str, mime_type: &str, bytes: Vec<u8>) -> Result<(), JsValue> {
        let array = js_sys::Uint8Array::from(bytes.as_slice());
        drop(bytes);
        let parts = js_sys::Array::new();
        parts.push(&array);
        let options = web_sys::BlobPropertyBag::new();
        options.set_type(mime_type);
        let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options)?;
        let url = web_sys::Url::create_object_url_with_blob(&blob)?;
        let document = web_sys::window()
            .and_then(|window| window.document())
            .ok_or_else(|| JsValue::from_str("oxigis-web: no document for the download"))?;
        let anchor = document
            .create_element("a")?
            .dyn_into::<web_sys::HtmlAnchorElement>()
            .map_err(|_| JsValue::from_str("oxigis-web: <a> did not build"))?;
        anchor.set_href(&url);
        anchor.set_download(file_name);
        anchor.click();
        wasm_bindgen_futures::spawn_local(async move {
            super::timers::sleep_ms(DOWNLOAD_URL_LIFETIME_MS).await;
            if let Err(error) = web_sys::Url::revoke_object_url(&url) {
                log::warn!(
                    "OxiGIS web shell: the export's object URL could not be revoked: {}",
                    js_message(&error),
                );
            }
        });
        Ok(())
    }

    /// Renders a JS exception as a message a status line can hold.
    ///
    /// The transports have their own copies; theirs build a [`TileError`] and
    /// are private to modules that do not exist on a host target.
    ///
    /// [`TileError`]: oxigis_ui::TileError
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

    /// Reads one attribute off the canvas element.
    ///
    /// A missing attribute, a missing canvas or a document-less context all
    /// mean "not configured", which is every one of these attributes' default.
    pub(crate) fn canvas_attribute(name: &str) -> Option<String> {
        web_sys::window()?
            .document()?
            .get_element_by_id(super::CANVAS_ID)?
            .get_attribute(name)
    }

    /// Canvas attribute that pins the graphics backend.
    ///
    /// `"webgl"` restricts the instance to `Backends::GL` and `"webgpu"` to
    /// `Backends::BROWSER_WEBGPU`; anything else is ignored with a warning.
    /// Absent — the default — both are offered and `wgpu` picks, which is what
    /// every real deployment wants. It exists because the WebGL2 fallback is
    /// otherwise unreachable on a machine that has WebGPU, and an untestable
    /// fallback is an untested one.
    pub const BACKEND_ATTRIBUTE: &str = "data-oxigis-backend";

    /// Largest 2D texture the shell would like a device to allow.
    ///
    /// `egui-wgpu`'s own figure, and the reason it asks: with a depth buffer
    /// the surface-sized texture has to exist, and 4K+ displays are ordinary.
    /// The difference here is that it is a *ceiling*, not a demand.
    const DESIRED_MAX_TEXTURE_DIMENSION: u32 = 8192;

    /// Clamps the device request and applies the backend override.
    ///
    /// `egui-wgpu`'s default descriptor asks for
    /// `max_texture_dimension_2d: 8192` unconditionally — it overrides the
    /// `downlevel_webgl2_defaults()` base it otherwise starts from, and
    /// WebGL2's guaranteed floor is 2048. On a device whose adapter reports
    /// less, `request_device` fails and the whole app reports "failed to
    /// start" rather than running with smaller textures, which it could.
    /// Asking for `min(adapter, 8192)` can never exceed what the adapter has,
    /// so it never turns a working device into a failed one.
    fn configure_wgpu(options: &mut eframe::WebOptions) {
        let forced = match canvas_attribute(BACKEND_ATTRIBUTE).as_deref() {
            None => None,
            Some("webgl") => Some(eframe::wgpu::Backends::GL),
            Some("webgpu") => Some(eframe::wgpu::Backends::BROWSER_WEBGPU),
            Some(other) => {
                log::warn!(
                    "OxiGIS web shell: {BACKEND_ATTRIBUTE}=\"{other}\" is not `webgl` or \
                     `webgpu`; letting wgpu choose",
                );
                None
            }
        };
        // `Existing` means an app supplied its own device, which this shell
        // never does; leaving it untouched is the right answer if it ever
        // does.
        let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup
        else {
            return;
        };
        if let Some(backends) = forced {
            log::info!("OxiGIS web shell: graphics backend pinned to {backends:?}");
            setup.instance_descriptor.backends = backends;
        }
        setup.device_descriptor = std::sync::Arc::new(|adapter: &eframe::wgpu::Adapter| {
            let base = if adapter.get_info().backend == eframe::wgpu::Backend::Gl {
                eframe::wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                eframe::wgpu::Limits::default()
            };
            // Never `max`ed back up to the base: `required_limits` is a floor
            // the device must meet, so asking for *less* than the base is
            // always safe and asking for more than the adapter has is the
            // failure this exists to avoid.
            let ceiling = adapter
                .limits()
                .max_texture_dimension_2d
                .min(DESIRED_MAX_TEXTURE_DIMENSION);
            eframe::wgpu::DeviceDescriptor {
                label: Some("oxigis web device"),
                required_limits: eframe::wgpu::Limits {
                    max_texture_dimension_2d: ceiling,
                    ..base
                },
                ..Default::default()
            }
        });
    }

    /// Looks up the canvas named by [`CANVAS_ID`](super::CANVAS_ID) in the
    /// current document.
    fn find_canvas() -> Result<web_sys::HtmlCanvasElement, JsValue> {
        let window = web_sys::window()
            .ok_or_else(|| JsValue::from_str("oxigis-web: no `window` — not a browser context"))?;
        let document = window
            .document()
            .ok_or_else(|| JsValue::from_str("oxigis-web: `window.document` is unavailable"))?;
        let element = document
            .get_element_by_id(super::CANVAS_ID)
            .ok_or_else(|| {
                JsValue::from_str(&format!(
                    "oxigis-web: no element with id `{}` in the document",
                    super::CANVAS_ID
                ))
            })?;
        element
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| {
                JsValue::from_str(&format!(
                    "oxigis-web: element `{}` is not a <canvas>",
                    super::CANVAS_ID
                ))
            })
    }

    /// Creates the eframe web runner and drives [`oxigis_ui::OxigisApp`] on the
    /// `oxigis_canvas` element until the page goes away.
    ///
    /// Resolves once startup has finished; the runner keeps itself alive
    /// afterwards through `requestAnimationFrame`, so the returned future does
    /// not represent the lifetime of the app.
    ///
    /// # Errors
    ///
    /// Returns the JS exception value when the canvas cannot be found, is not
    /// a `<canvas>`, or when neither WebGPU nor WebGL2 yields a usable adapter.
    pub async fn start() -> Result<(), JsValue> {
        let canvas = find_canvas()?;

        // `data-oxigis-fallback-font` on the canvas is the markup half of the
        // CJK opt-in; the JS half is `oxigis_load_fallback_font`. Both are
        // no-ops when absent, which is the default.
        if let Some(url) = super::font_fetch::url_from_canvas() {
            super::font_fetch::load_fallback_font(&url);
        }
        // `data-oxigis-print-font` is the export-only twin: only the URL is
        // recorded here — the bytes are fetched on the first export, so a
        // page that never prints pays nothing.
        if let Some(url) = super::font_fetch::print_url_from_canvas() {
            super::font_fetch::set_print_font_url(&url);
        }
        // `data-oxigis-bold-font` and `data-oxigis-print-bold-font` are the
        // weight twins of the two above, with the same eager/lazy split: the
        // screen face is fetched now, the export-only one on the first
        // export. Absent — the default — Bold labels draw Regular.
        if let Some(url) = super::font_fetch::bold_url_from_canvas() {
            super::font_fetch::load_bold_font(&url);
        }
        if let Some(url) = super::font_fetch::print_bold_url_from_canvas() {
            super::font_fetch::set_print_bold_font_url(&url);
        }

        let mut web_options = eframe::WebOptions {
            // Only variant compiled in (eframe's `glow` feature is off); see
            // the crate-level docs for how the WebGPU -> WebGL2 fallback is
            // actually chosen, inside wgpu.
            renderer: eframe::Renderer::Wgpu,
            ..Default::default()
        };
        // Mutated in place rather than built fresh, so eframe's own defaults —
        // and the display handle its web painter injects afterwards — survive.
        configure_wgpu(&mut web_options);

        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|_cc| Ok(Box::new(WebShell::new()))),
            )
            .await
    }

    /// Writes into the page's `#oxigis_status` banner.
    ///
    /// The one channel this module has into the page, used for both halves of
    /// the startup handshake: the real error on failure, and an empty string
    /// on success — `#oxigis_status:empty` is styled `display: none`, so
    /// clearing it is what uncovers the canvas. The page must therefore NOT
    /// clear it itself: module instantiation resolves long before the canvas,
    /// the adapter or the first frame exist, so it is not a success signal and
    /// clearing on it erases whatever [`start`] had to report.
    fn show_status(message: &str) {
        if let Some(document) = web_sys::window().and_then(|window| window.document())
            && let Some(banner) = document.get_element_by_id("oxigis_status")
        {
            banner.set_text_content(Some(message));
        }
    }

    /// Writes the in-flight fetch count into the page's `#oxigis_loading`
    /// badge (`crates/oxigis-web/index.html`).
    ///
    /// Called by [`WebShell::ui`] only on the frames where the count
    /// actually changed — this function itself is unconditional so a future
    /// caller cannot forget that half of the contract, but doing the DOM
    /// write every single painted frame regardless would touch the document
    /// far more often than the number on screen ever changes.
    ///
    /// An empty string when `count` is `0`: `#oxigis_loading:empty` is
    /// styled `display: none` in the page, the same convention
    /// `#oxigis_status` uses (see [`show_status`]), so writing the text is
    /// the whole show/hide — there is no separate visibility flag to keep in
    /// sync with it.
    fn update_loading_indicator(count: u32) {
        let text = if count == 0 {
            String::new()
        } else {
            format!("Loading\u{2026} ({count})")
        };
        if let Some(document) = web_sys::window().and_then(|window| window.document())
            && let Some(badge) = document.get_element_by_id("oxigis_loading")
        {
            badge.set_text_content(Some(&text));
        }
    }

    /// wasm-bindgen entry point, invoked automatically when the generated
    /// `oxigis_web.js` module is initialised.
    ///
    /// Installs the console panic hook and the `log` bridge, then spawns
    /// [`start`] on the browser's microtask queue. Startup failures are
    /// reported to the console rather than propagated, so a missing canvas
    /// cannot abort module instantiation.
    #[wasm_bindgen(start)]
    pub fn oxigis_web_main() {
        console_error_panic_hook::set_once();
        // Discarded deliberately: this fails only when a logger is already
        // installed, and the one thing that could report the failure is the
        // logger.
        let _ = eframe::WebLogger::init(log::LevelFilter::Debug);

        log::info!(
            "OxiGIS web shell v{} (core {}, render {}, ui {})",
            super::VERSION,
            oxigis_core::VERSION,
            oxigis_render::VERSION,
            oxigis_ui::VERSION,
        );

        wasm_bindgen_futures::spawn_local(async {
            match start().await {
                // The success half of the handshake; see `show_status`.
                Ok(()) => show_status(""),
                Err(err) => {
                    log::error!("OxiGIS web shell failed to start: {err:?}");
                    // The real message, not a guess: `find_canvas` diagnoses a
                    // missing or mistyped canvas id precisely, and reporting
                    // that as a GPU problem sends the reader to the wrong page.
                    show_status(&format!(
                        "OxiGIS failed to start: {}\nA WebGPU or WebGL2 context is required; \
                         see the browser console for details.",
                        js_message(&err),
                    ));
                }
            }
        });
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::start;
