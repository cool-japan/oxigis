//! OxiGIS desktop shell — native single binary (Linux/macOS/Windows).
//!
//! Derived from the WASM-first build (blueprint §2): same `oxigis-ui`
//! panels and `oxigis-render` map view, hosted in a winit window via
//! eframe/wgpu.
//!
//! # Command line
//!
//! `oxigis [OPTIONS] [PATH]…` — see [`USAGE`]. Positional paths are routed
//! through the same [`oxigis_ui::classify_drop`] dispatch a drag-and-drop
//! takes, which is what makes a file association ("Open With") work.
//!
//! # Project lifecycle
//!
//! `oxigis-ui` owns no filesystem, so the whole of File ▸ Open / Save /
//! Save As / Open Recent crosses to this shell through take-once seams
//! (`take_pending_project_save` / `take_pending_project_open`), exactly as the
//! PDF export and the tile-archive picker already did. This shell resolves the
//! path — [`file_dialog`], native where the platform has a Pure-Rust dialog and
//! an in-app prompt where it does not — moves the bytes ([`project_file`],
//! temp-file-then-rename so a failed write cannot truncate a project) and
//! reports the outcome back. It also intercepts the window's close request
//! until the unsaved-changes question has an answer, and remembers window
//! geometry and the recent-project list between launches ([`session`]).

// A GUI binary must not open a console window behind the map when it is
// double-clicked. Release builds only: a console is exactly what is wanted
// when running the debug binary from a terminal.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
#![forbid(unsafe_code)]

mod cli;
mod dataset_read;
mod export_stack;
mod file_dialog;
mod file_write;
mod font_scan;
mod label_fonts;
#[cfg(test)]
mod lifecycle_tests;
mod project_file;
mod provider_watch;
mod range_file;
mod range_http;
mod session;
mod tile_http;

use cli::{Startup, USAGE};
use dataset_read::{drain_dropped_paths, open_startup_paths};
use file_dialog::{Ask, PathPrompt, PromptOutcome};
use file_write::PendingFileWrite;
use label_fonts::{
    ScannedFont, cjk_bold_paths, cjk_regular_paths, drain_cjk_font, install_label_fonts,
    start_cjk_font_scan,
};
use oxigis_core::ArchiveFormat;
use oxigis_ui::{
    ArchiveLayerConfig, ArchiveProbe, ArchiveTileProvider, ArchiveTileTransport, BasemapConfig,
    BoxedTileProvider, CogLayerConfig, CogTileProvider, LocalLayerOp, MemoryRangeTransport,
    RangeTransport, VectorTileConfig, VectorTileProvider, XyzTileProvider, mbtiles::MbTilesReader,
};
use provider_watch::{
    InstalledProvider, InstalledVectorSource, ProviderWatch, RasterWatch, SharedArchive, SharedCog,
    SharedVector, SharedXyz, VectorWatch,
};

/// Native application wrapper: hosts [`oxigis_ui::OxigisApp`] inside an
/// `eframe`/winit window and drives it once per frame.
///
/// The GPU map lives entirely in `oxigis-ui`'s `map_gpu` module: this shell
/// only forwards `eframe`'s `wgpu` render state into
/// `OxigisApp::attach_gpu_map_with`, which installs the tile renderer into
/// `egui_wgpu`'s callback resources on the first frame. No GPU resource is
/// owned here. The one native-specific piece this shell contributes is the
/// tile transport ([`tile_http`]): blocking HTTP on a worker pool.
struct OxigisDesktopApp {
    /// Shared UI state and panel logic from `oxigis-ui`.
    inner: oxigis_ui::OxigisApp,
    /// Whether the label pass's fonts have been installed into the GPU state.
    ///
    /// Set only when [`oxigis_ui::map_gpu::set_label_fonts`] actually succeeds:
    /// it answers `false` on every frame before the `MapGpuState` exists, which
    /// includes the first one, so retrying is the whole point of the flag.
    fonts_installed: bool,
    /// Where the background CJK font scan streams its fallback chain, one
    /// font per message, until the scan thread hangs up.
    ///
    /// [`None`] once the channel has disconnected (chain fully installed, or
    /// the scan finished empty-handed) or if the scan thread could not be
    /// started at all.
    cjk_font: Option<std::sync::mpsc::Receiver<ScannedFont>>,
    /// The PDF export running on its background thread, when one is: the
    /// receiving end of the one-shot result channel, plus the destination
    /// path for the status line. `None` when no export is in flight.
    print_job: Option<PrintJob>,
    /// Paths named on the command line, still to be opened.
    ///
    /// Drained on the first frame rather than before the window exists: a
    /// dataset becomes a layer through `oxigis-ui`, and the GPU work that
    /// creates queues against a render state the frame loop owns.
    startup_paths: Vec<std::path::PathBuf>,
    /// Handles onto the providers currently drawing, for the failures only
    /// they learn about — see [`ProviderWatch`].
    provider_watch: ProviderWatch,
    /// What this shell remembers between launches — window geometry, the
    /// recent-project list, the directory the dialogs open in. Written once,
    /// by [`eframe::App::on_exit`].
    session: session::SessionState,
    /// The in-app path prompt while it is on screen, for a platform with no
    /// Pure-Rust native dialog. [`None`] everywhere else, and on every
    /// platform while nothing is being asked.
    path_prompt: Option<PathPrompt>,
    /// A PDF export whose destination is still being asked for.
    ///
    /// The native dialog answers within the call, but the in-app prompt spans
    /// frames — so the snapshot has to be parked somewhere while the user
    /// types, and it must keep the "an export is already running" guard honest.
    pending_pdf: Option<Box<oxigis_ui::print::PrintRequest>>,
    /// A project write whose destination is still being asked for — the same
    /// across-frames parking as [`Self::pending_pdf`], for the same reason.
    pending_project_save: Option<oxigis_ui::ProjectSaveRequest>,
    /// A data file — a layer/table export, or a Processing result — whose
    /// destination is still being asked for. Same across-frames parking, same
    /// reason: on a platform with no native dialog the answer arrives frames
    /// after the ask, and a request dropped in between is a file the user was
    /// told they would get.
    pending_file_write: Option<PendingFileWrite>,
    /// The window title as last set, so the shell only issues a
    /// [`egui::ViewportCommand::Title`] when it actually changed.
    window_title: String,
    /// The window rectangle observed on the most recent frame that reported
    /// one, and whether it was maximized: what [`eframe::App::on_exit`]
    /// persists.
    ///
    /// Wayland reports no window position at all, so a frame that answers
    /// [`None`] leaves the last known geometry standing rather than replacing
    /// it with nothing.
    observed_geometry: Option<session::WindowGeometry>,
    /// Whether the window was maximized on the most recent frame.
    observed_maximized: bool,
    /// Whether the restored window has been checked against the monitor it
    /// actually landed on. One-shot: there is no monitor list before the
    /// window exists, so the check cannot happen in [`main`].
    geometry_rescued: bool,
    /// Set once the user has confirmed the close: from here the shell stops
    /// intercepting and lets the close through.
    ///
    /// A one-way latch, and load-bearing. Confirming does not make the project
    /// clean, so without it the [`egui::ViewportCommand::Close`] the shell
    /// sends would arrive back as another close request, be intercepted again,
    /// and reopen the very dialog the user just answered — for ever.
    closing: bool,
}

/// A PDF export in flight on its background thread.
struct PrintJob {
    /// Delivers the thread's one result message; disconnection without a
    /// message means the thread panicked.
    result: std::sync::mpsc::Receiver<Result<ExportReport, String>>,
    /// Where the export is writing, for the status line.
    path: std::path::PathBuf,
}

/// What one PDF export produced, beside the file itself.
///
/// The tile loops print whatever arrived when [`tile_budget`] expires,
/// so "the file was written" and "the file shows the map" are different
/// answers, and only this one distinguishes them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ExportReport {
    /// Raster tiles that never arrived; drawn `MISSING_TILE_GRAY`.
    missing_raster: usize,
    /// Vector tiles that never arrived; their features are simply absent.
    missing_vector: usize,
    /// Tiles the page needed in total.
    total: usize,
}

impl ExportReport {
    /// The status line for a finished export, naming what is missing from it.
    fn status(&self, path: &std::path::Path) -> String {
        let mut message = format!("PDF exported to {}.", path.display());
        if self.missing_raster > 0 {
            message.push_str(&format!(
                " {} of {} basemap tiles were missing (drawn gray).",
                self.missing_raster, self.total,
            ));
        }
        if self.missing_vector > 0 {
            message.push_str(&format!(
                " {} of {} vector tiles were missing.",
                self.missing_vector, self.total,
            ));
        }
        message
    }
}

/// Builds the real XYZ tile provider for the native shell.
///
/// Returns [`None`] — which makes `attach_gpu_map_with` fall back to the
/// synthetic checkerboard rather than leaving the map empty — if the worker
/// pool cannot start or the configured URL template is unusable. Both are
/// logged, since a silently synthetic basemap would be baffling.
fn build_tile_provider(config: &BasemapConfig, ctx: &egui::Context) -> Option<InstalledProvider> {
    let transport = match tile_http::HttpTileTransport::new() {
        Ok(transport) => transport,
        Err(error) => {
            tracing::error!(%error, "OxiGIS desktop: could not start the tile worker pool");
            return None;
        }
    };
    match XyzTileProvider::new(config, ctx, Box::new(transport)) {
        Ok(provider) => {
            tracing::info!(
                template = config.url_template,
                user_agent = tile_http::HttpTileTransport::user_agent(),
                "OxiGIS desktop: XYZ basemap attached",
            );
            let provider = std::sync::Arc::new(provider);
            Some(InstalledProvider {
                provider: Box::new(SharedXyz(std::sync::Arc::clone(&provider))),
                watch: RasterWatch::basemap(provider),
            })
        }
        Err(error) => {
            tracing::error!(%error, "OxiGIS desktop: invalid basemap URL template");
            None
        }
    }
}

/// Builds a COG provider, optionally drawing over a fresh XYZ basemap.
///
/// `basemap` is [`Some`] only for the **single-slot** path (the PDF export,
/// and any shell that has not migrated to the tile stack), where the COG and
/// the basemap under it are CPU-composited into one provider.
///
/// For a **stack** entry it is [`None`], and that is load-bearing rather than
/// an optimisation: with no base, `CogTileProvider::tile` returns the layer's
/// own alpha-carrying pixels, which is exactly what the stack composites. Were
/// the basemap blended in here, every layer tile would carry a copy of it and
/// the entry's opacity tint would fade the basemap along with the layer.
///
/// Returns [`None`] if the range worker pool cannot start, in which case the
/// existing provider is left in place and the failure is logged.
fn build_cog_provider(
    cog: &CogLayerConfig,
    basemap: Option<&BasemapConfig>,
    ctx: &egui::Context,
) -> Option<InstalledProvider> {
    let transport = match range_http::HttpRangeTransport::new() {
        Ok(transport) => transport,
        Err(error) => {
            tracing::error!(%error, "OxiGIS desktop: could not start the COG worker pool");
            return None;
        }
    };
    let provider = match CogTileProvider::new(cog, ctx, Box::new(transport)) {
        Ok(provider) => provider,
        Err(error) => {
            tracing::error!(%error, "OxiGIS desktop: could not build the COG provider");
            return None;
        }
    };
    tracing::info!(url = cog.url, "OxiGIS desktop: COG layer attached");
    // The basemap's handle is carried up rather than dropped with its box: it
    // is the layer the user still sees if the COG never opens, so its own
    // silence has to stay reportable.
    let (base, basemap) = split_base(basemap.and_then(|config| build_tile_provider(config, ctx)));
    let provider = match base {
        Some(base) => provider.with_base(base),
        None => provider,
    };
    let provider = std::sync::Arc::new(provider);
    Some(InstalledProvider {
        provider: Box::new(SharedCog(std::sync::Arc::clone(&provider))),
        watch: RasterWatch::cog(provider, basemap),
    })
}

/// Splits a built basemap into the box that goes underneath another provider
/// and the handle the shell keeps watching.
fn split_base(
    base: Option<InstalledProvider>,
) -> (
    Option<BoxedTileProvider>,
    Option<std::sync::Arc<XyzTileProvider>>,
) {
    match base {
        Some(base) => (Some(base.provider), base.watch.into_basemap()),
        None => (None, None),
    }
}

/// Builds the range transport a tile archive needs on this machine.
///
/// Three answers, in the order that keeps the archive readable at all:
///
/// * bytes the app is already holding for the session (a drop the app read
///   itself) — [`MemoryRangeTransport`];
/// * a local path — [`range_file::FileRangeTransport`], so a 137 GB archive
///   *streams* instead of being read whole;
/// * a URL — the same [`range_http::HttpRangeTransport`] a COG uses.
///
/// Returns [`None`] only when a worker pool cannot start, which is logged.
fn build_archive_transport(
    config: &ArchiveLayerConfig,
    bytes: Option<std::sync::Arc<[u8]>>,
) -> Option<Box<dyn RangeTransport>> {
    if let Some(bytes) = bytes {
        return Some(Box::new(MemoryRangeTransport::from_shared(bytes)));
    }
    if config.is_local() {
        return match range_file::FileRangeTransport::new() {
            Ok(transport) => Some(Box::new(transport)),
            Err(error) => {
                tracing::error!(%error, "OxiGIS desktop: could not start the archive worker pool");
                None
            }
        };
    }
    match range_http::HttpRangeTransport::new() {
        Ok(transport) => Some(Box::new(transport)),
        Err(error) => {
            tracing::error!(%error, "OxiGIS desktop: could not start the archive worker pool");
            None
        }
    }
}

/// How long the archive is, when this machine can know without reading it.
///
/// Only ever a *bound*: the paged MBTiles reader uses it to sanity-check a page
/// count its own header could not vouch for, never to widen one. A URL has no
/// answer here — the transport pins the length from `Content-Range` instead —
/// and a file that has vanished simply reports none.
fn archive_length(
    config: &ArchiveLayerConfig,
    bytes: Option<&std::sync::Arc<[u8]>>,
) -> Option<u64> {
    if let Some(bytes) = bytes {
        return u64::try_from(bytes.len()).ok();
    }
    if !config.is_local() {
        return None;
    }
    std::fs::metadata(config.location())
        .ok()
        .map(|metadata| metadata.len())
}

/// Builds a raster tile-archive provider, optionally drawing over a fresh XYZ
/// basemap.
///
/// The archive twin of [`build_cog_provider`], and deliberately identical in
/// shape — including the `Option` base and the reason for it: an archive layer
/// composites over the basemap exactly as a COG does, and is a base-less stack
/// entry for exactly the same reason.
fn build_archive_provider(
    config: &ArchiveLayerConfig,
    basemap: Option<&BasemapConfig>,
    ctx: &egui::Context,
    bytes: Option<std::sync::Arc<[u8]>>,
    reader: Option<std::sync::Arc<MbTilesReader>>,
) -> Option<InstalledProvider> {
    let length = archive_length(config, bytes.as_ref());
    let built = match config.format {
        // An archive the app has already read and indexed — a drop — stays with
        // the resident reader: its bytes are in memory either way.
        ArchiveFormat::MbTiles => match reader {
            Some(reader) => ArchiveTileProvider::mbtiles(config.location().to_owned(), reader, ctx),
            // Everything else streams: a multi-gigabyte local `.mbtiles` opens
            // in one 16 KiB read instead of being loaded whole.
            None => {
                let transport = build_archive_transport(config, bytes)?;
                ArchiveTileProvider::paged_mbtiles(
                    config.location().to_owned(),
                    ctx,
                    transport,
                    length,
                )
            }
        },
        ArchiveFormat::PmTiles => {
            let transport = build_archive_transport(config, bytes)?;
            ArchiveTileProvider::pmtiles(config.location().to_owned(), ctx, transport)
        }
    };
    let provider = match built {
        Ok(provider) => provider,
        Err(error) => {
            tracing::error!(%error, "OxiGIS desktop: could not build the archive provider");
            return None;
        }
    };
    tracing::info!(
        archive = config.location(),
        "OxiGIS desktop: raster tile archive attached",
    );
    let (base, basemap) = split_base(basemap.and_then(|config| build_tile_provider(config, ctx)));
    let provider = match base {
        Some(base) => provider.with_base(base),
        None => provider,
    };
    let provider = std::sync::Arc::new(provider);
    Some(InstalledProvider {
        provider: Box::new(SharedArchive(std::sync::Arc::clone(&provider))),
        watch: RasterWatch::archive(provider, basemap),
    })
}

/// Builds the MVT vector-tile source for the native shell.
///
/// Reuses [`tile_http::HttpTileTransport`] — a `.pbf` endpoint is an ordinary
/// XYZ service — so this adds no new platform capability, only a second worker
/// pool so vector fetches cannot starve the basemap.
///
/// Returns [`None`], leaving the map raster-only, if the worker pool cannot
/// start or the URL template is unusable. Both are logged.
fn build_vector_source(
    config: &VectorTileConfig,
    ctx: &egui::Context,
    bytes: Option<std::sync::Arc<[u8]>>,
    reader: Option<std::sync::Arc<MbTilesReader>>,
) -> Option<InstalledVectorSource> {
    // Two arms, one provider: an archive-backed config swaps the platform HTTP
    // transport for one that resolves tiles inside a single file, and every
    // other part of the vector path -- decode, tessellation, labels, retry, and
    // the PDF export that reads `decoded()` -- is unchanged.
    let archive_transport = match config.archive.as_ref() {
        Some(archive) if archive.format == ArchiveFormat::MbTiles => Some(match reader {
            Some(reader) => ArchiveTileTransport::mbtiles(archive.location().to_owned(), reader),
            None => {
                let length = archive_length(archive, bytes.as_ref());
                ArchiveTileTransport::paged_mbtiles(
                    archive.location().to_owned(),
                    build_archive_transport(archive, bytes)?,
                    length,
                )
            }
        }),
        Some(archive) => Some(ArchiveTileTransport::pmtiles(
            archive.location().to_owned(),
            build_archive_transport(archive, bytes)?,
        )),
        None => None,
    };
    // The clone is a second handle onto the same archive, not a second reader:
    // the provider takes the transport by `Box`, and an archive that refuses to
    // open reports it nowhere else.
    let transport: Box<dyn oxigis_ui::TileTransport> = match archive_transport.clone() {
        Some(transport) => Box::new(transport),
        None => match tile_http::HttpTileTransport::new() {
            Ok(transport) => Box::new(transport),
            Err(error) => {
                tracing::error!(%error, "OxiGIS desktop: could not start the vector worker pool");
                return None;
            }
        },
    };
    match VectorTileProvider::new(config, ctx, transport) {
        Ok(provider) => {
            tracing::info!(
                template = config.url_template,
                rules = config.paints.len(),
                "OxiGIS desktop: vector tile layer attached",
            );
            let provider = std::sync::Arc::new(provider);
            Some(InstalledVectorSource {
                source: Box::new(SharedVector(std::sync::Arc::clone(&provider))),
                watch: VectorWatch::new(provider, archive_transport),
            })
        }
        Err(error) => {
            tracing::error!(%error, "OxiGIS desktop: invalid vector tile URL template");
            None
        }
    }
}

/// Builds one entry of the N-layer tile stack, plus the watch that keeps its
/// later silence reportable.
///
/// Every arm is deliberately **base-less** (see [`build_cog_provider`]): a
/// stack entry contributes its own alpha-carrying pixels and the basemap is its
/// own pass underneath.
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
) -> Result<(oxigis_ui::TileLayerGpuSource, provider_watch::StackWatch), String> {
    match &plan.source {
        oxigis_ui::TileLayerSource::Cog(config) => build_cog_provider(config, None, ctx)
            .map(|built| {
                (
                    oxigis_ui::TileLayerGpuSource::Raster(built.provider),
                    provider_watch::StackWatch::Raster(built.watch),
                )
            })
            .ok_or_else(|| "the COG provider could not be built".to_owned()),
        oxigis_ui::TileLayerSource::RasterArchive(config) => {
            build_archive_provider(config, None, ctx, bytes, reader)
                .map(|built| {
                    (
                        oxigis_ui::TileLayerGpuSource::Raster(built.provider),
                        provider_watch::StackWatch::Raster(built.watch),
                    )
                })
                .ok_or_else(|| "the tile archive could not be opened".to_owned())
        }
        // An XYZ layer that is not the promoted basemap draws as an ordinary
        // overlay. `build_tile_provider` never had a base to skip.
        oxigis_ui::TileLayerSource::Xyz(config) => build_tile_provider(config, ctx)
            .map(|built| {
                (
                    oxigis_ui::TileLayerGpuSource::Raster(built.provider),
                    provider_watch::StackWatch::Raster(built.watch),
                )
            })
            .ok_or_else(|| "the XYZ tile provider could not be built".to_owned()),
        oxigis_ui::TileLayerSource::Vector(config) => {
            build_vector_source(config, ctx, bytes, reader)
                .map(|built| {
                    (
                        oxigis_ui::TileLayerGpuSource::Vector(built.source),
                        provider_watch::StackWatch::Vector(built.watch),
                    )
                })
                .ok_or_else(|| "the vector tile source could not be built".to_owned())
        }
    }
}

/// The session-held archive bytes and MBTiles reader one stack plan needs, or
/// `(None, None)` for a plan that names no archive.
///
/// A browser's — and a drop's — only local archive is the bytes the app already
/// read, and an MBTiles image is indexed once and shared, so both have to be
/// resolved from the app before the build.
fn stack_archive_data(
    app: &oxigis_ui::OxigisApp,
    plan: &oxigis_ui::TileLayerPlan,
) -> (
    Option<std::sync::Arc<[u8]>>,
    Option<std::sync::Arc<MbTilesReader>>,
) {
    let location = match &plan.source {
        oxigis_ui::TileLayerSource::RasterArchive(config) => Some(config.location()),
        oxigis_ui::TileLayerSource::Vector(config) => {
            config.archive.as_ref().map(ArchiveLayerConfig::location)
        }
        oxigis_ui::TileLayerSource::Cog(_) | oxigis_ui::TileLayerSource::Xyz(_) => None,
    };
    match location {
        Some(location) => (app.archive_bytes(location), app.mbtiles_reader(location)),
        None => (None, None),
    }
}

/// The floor of the PDF export's per-pass tile budget, whatever the page
/// costs: enough for an on-screen-sized map over a working link.
const PRINT_TILE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Added to that floor for each tile the page names.
///
/// A flat budget judges a 12-tile page and a 250-tile one by the same clock,
/// which on a slow link is what turns an A3 export into a sheet of
/// `MISSING_TILE_GRAY` rectangles.
const PRINT_TILE_ALLOWANCE: std::time::Duration = std::time::Duration::from_millis(300);

/// The ceiling on one pass, so a page naming thousands of tiles cannot hold
/// the export thread — and with it the next export — for the session.
const PRINT_TILE_MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(90);

/// How long one tile pass waits before printing whatever arrived (missing
/// raster tiles come out neutral gray; missing vector tiles simply have no
/// features). Reported either way — see [`ExportReport`].
fn tile_budget(tiles: usize) -> std::time::Duration {
    let allowance = u32::try_from(tiles)
        .ok()
        .and_then(|tiles| PRINT_TILE_ALLOWANCE.checked_mul(tiles))
        .unwrap_or(PRINT_TILE_MAX_WAIT);
    PRINT_TILE_TIMEOUT
        .saturating_add(allowance)
        .min(PRINT_TILE_MAX_WAIT)
}

/// The default export file name, unique per export.
///
/// A whole-second stamp used to be the only thing separating two exports, so
/// two in the same second silently overwrote each other on the dialog-less
/// path. The per-process counter closes that regardless of clock resolution,
/// and the timestamp is kept because it is what makes a directory of exports
/// readable.
fn export_file_name() -> String {
    static EXPORT_SEQUENCE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let sequence = EXPORT_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if sequence == 0 {
        format!("oxigis-export-{seconds}.pdf")
    } else {
        format!("oxigis-export-{seconds}-{sequence}.pdf")
    }
}

/// Runs one PDF export to completion and writes it to `path`.
///
/// Runs on a background thread (see `start_pdf_export`): the provider is
/// built fresh from the request's snapshot (never borrowed from the live
/// map), so nothing here races a project edit, and the fetch pool it starts
/// dies with this call.
fn export_pdf_to(
    request: &oxigis_ui::print::PrintRequest,
    ctx: &egui::Context,
    path: &std::path::Path,
) -> Result<ExportReport, String> {
    // The single-slot path stays EXACTLY as it was for every project whose
    // stack the three legacy fields already describe — the same provider, the
    // same composite, the same bytes. Only a stack those fields would truncate
    // (two rasters, two tilesets, or an XYZ overlay they cannot name at all)
    // takes the composed path, so the change is invisible to every export that
    // was already right.
    let legacy = request.stack_fits_legacy_slots();
    let provider: Option<BoxedTileProvider> = if legacy {
        Some(
            match (request.archive.as_ref(), request.cog.as_ref()) {
                // The export composites on the CPU into ONE image, so here the
                // base genuinely belongs underneath the layer — the opposite of
                // the stack's rule, and the reason `build_cog_provider` takes an
                // `Option`.
                (Some(archive), _) => {
                    build_archive_provider(archive, Some(&request.basemap), ctx, None, None)
                }
                (None, Some(cog)) => build_cog_provider(cog, Some(&request.basemap), ctx),
                (None, None) => build_tile_provider(&request.basemap, ctx),
            }
            .ok_or_else(|| "no tile provider could be built for the export".to_string())?
            .provider,
        )
    } else {
        None
    };

    // The embedded-font chain: the bundled Latin face first, then the system
    // CJK fallback chain, re-read from disk for this export (transient on
    // this background thread — the subset that lands in the PDF is a few KB).
    // The chain's *paths* come from the process-wide scan
    // ([`cjk_regular_paths`]), so no directory tree is walked again here.
    use oxigis_ui::print::{FaceRole, PrintFace};
    let mut chain = vec![PrintFace::regular(
        oxifont_bundled::NOTO_SANS_REGULAR.to_vec(),
        FaceRole::ScreenShared,
    )];
    for path in cjk_regular_paths().iter() {
        if let Some(bytes) = font_scan::read_cjk_font(path) {
            chain.push(PrintFace::regular(bytes, FaceRole::ScreenShared));
        }
    }
    // The BOLD slots (print/text v1.4, D-W5): the same faces the map draws
    // bold with, so page and screen agree — hence `ScreenShared`, which also
    // means they are never instanced. Read only for the export; a project
    // with no Bold symbol style never touches them, and `font::plan` skips
    // the whole second pass, leaving the PDF byte-identical.
    for path in cjk_bold_paths().iter() {
        if let Some(bytes) = font_scan::read_cjk_font(path) {
            chain.push(PrintFace::bold(bytes, FaceRole::ScreenShared));
        }
    }
    let fonts = oxigis_ui::print::PrintFonts::with_roles(chain);

    let map_box = oxigis_ui::print::map_box(&request.options);
    let out_px = oxigis_ui::print::raster_size_px(&map_box, &request.options);
    let compose = oxigis_ui::print::compose_view(request.view, out_px);
    let required = oxigis_ui::print::required_tiles(&compose);
    let mut report = ExportReport {
        total: required.len(),
        ..ExportReport::default()
    };

    // The composed path builds one provider per stack entry and pastes them all
    // here; the single-slot path keeps its own loop below.
    let composed = match provider.as_ref() {
        Some(_) => None,
        None => Some(export_stack::compose_stack(
            request, ctx, &compose, &required,
        )?),
    };
    if let Some(composed) = composed.as_ref() {
        report.missing_raster = composed.missing_raster;
        report.missing_vector = composed.missing_vector;
    }

    // Poll until every tile has arrived or the budget is spent; `tile()` is
    // what enqueues each fetch, so the first pass primes the whole set.
    if let Some(provider) = provider.as_ref() {
        let deadline = std::time::Instant::now() + tile_budget(required.len());
        loop {
            let missing = required
                .iter()
                .filter(|tile| provider.tile(**tile).is_none())
                .count();
            if missing == 0 {
                break;
            }
            if std::time::Instant::now() >= deadline {
                tracing::warn!(
                    missing,
                    total = required.len(),
                    "OxiGIS desktop: PDF export proceeding with missing tiles",
                );
                // Carried out to the status line: a page of gray rectangles
                // reported as a plain success is the failure this counts.
                report.missing_raster = missing;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    // Streamed vector tiles, when a vector-tile layer is active: the same
    // provider the screen uses — `mesh()` drives the fetch loop, `decoded()`
    // collects the typed tiles the print overlay draws from.
    let mut vector_tiles: Vec<(
        oxigis_render::TileId,
        std::sync::Arc<oxigis_render::mvt::VectorTile>,
    )> = Vec::new();
    if let Some(config) = request.vector.as_ref().filter(|_| legacy)
        && let Some(installed) = build_vector_source(config, ctx, None, None)
    {
        // The watch handles are for the live map's status line; here a tile
        // that never arrives is already counted into the report below.
        let source = installed.source;
        let deadline = std::time::Instant::now() + tile_budget(required.len());
        loop {
            let _ = source.begin_frame(compose);
            let missing = required
                .iter()
                .filter(|tile| {
                    let absent = source.decoded(**tile).is_none();
                    if absent {
                        let _ = source.mesh(**tile);
                    }
                    absent
                })
                .count();
            if missing == 0 {
                break;
            }
            if std::time::Instant::now() >= deadline {
                tracing::warn!(
                    missing,
                    total = required.len(),
                    "OxiGIS desktop: PDF export proceeding with missing vector tiles",
                );
                report.missing_vector = missing;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        for tile in &required {
            if let Some(decoded) = source.decoded(*tile) {
                vector_tiles.push((*tile, decoded));
            }
        }
    }

    // Moved out of `composed`, never cloned: the raster is `out_px[0] ×
    // out_px[1] × 3` bytes — tens of megabytes at 300 dpi on A3.
    let (rgb, stack_tiles) = match (provider.as_ref(), composed) {
        (Some(provider), _) => (
            oxigis_ui::print::compose_map_rgb(&compose, &mut |tile| provider.tile(tile)),
            Vec::new(),
        ),
        (None, Some(composed)) => (composed.rgb, composed.vector),
        // Unreachable: exactly one of the two paths runs, and the composed one
        // fails loudly rather than returning `None`. A blank page beats a
        // panic on a background export thread.
        (None, None) => (
            vec![oxigis_ui::print::MISSING_TILE_GRAY; out_px[0] as usize * out_px[1] as usize * 3],
            Vec::new(),
        ),
    };
    let pdf = oxigis_ui::print::pdf_document_with(
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
    std::fs::write(path, &pdf)
        .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    tracing::info!(path = %path.display(), bytes = pdf.len(), "OxiGIS desktop: PDF exported");
    Ok(report)
}

/// Performs this frame's local-vector work against the GPU map.
///
/// Drained unconditionally — including on a frame with no `wgpu` render state,
/// where the ops are logged and discarded rather than left to accumulate — and
/// applied in order, since an `Add` must precede the edits that refer to it.
fn apply_local_ops(
    app: &mut oxigis_ui::OxigisApp,
    render_state: Option<&eframe::egui_wgpu::RenderState>,
) {
    for op in app.take_pending_local_ops() {
        let Some(render_state) = render_state else {
            tracing::error!(
                ?op,
                "OxiGIS desktop: the GPU map is not attached, so the local layer op was discarded",
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
            // A no-op `Clear` (nothing was attached) is not a failure.
            LocalLayerOp::Clear => oxigis_ui::map_gpu::clear_local_vector_layers(render_state),
        };
        if !applied {
            tracing::debug!("OxiGIS desktop: a local layer op changed nothing");
        }
    }
}

/// How often a hidden window is woken while background work is outstanding.
///
/// `App::logic` runs unconditionally while the window is *visible*, but behind
/// a minimised one only when a repaint was requested — so the thread hand-offs
/// this shell settles (the PDF export, the font scan, an archive probe) need a
/// heartbeat to be noticed at all. Conditional on purpose: an unconditional
/// one would spin a hidden, idle window forever.
const BACKGROUND_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);

impl OxigisDesktopApp {
    /// Create a fresh desktop app wrapping a new [`oxigis_ui::OxigisApp`],
    /// with the paths from the command line queued for the first frame and the
    /// remembered session already applied to the UI.
    fn new(startup_paths: Vec<std::path::PathBuf>, session: session::SessionState) -> Self {
        let mut inner = oxigis_ui::OxigisApp::new();
        // What turns File ▸ Save from the copy-JSON modal into a real write,
        // File ▸ Open from a paste box into a file read, and the close button
        // into a question. Declared once, here, for every platform: even
        // without a native dialog this shell can ask for a path itself and can
        // certainly write the bytes.
        inner.set_native_project_io(true);
        // This shell reconciles through the N-layer tile stack, so the two
        // legacy single-slot seams must stop offering the layers the stack now
        // owns: `pending_raster_work` becomes basemap-only and
        // `pending_vector_work` goes quiet. Declaring it (rather than merely
        // not calling them) is what makes the two rules impossible to get
        // half-right — see `OxigisApp::set_tile_stack_shell`.
        inner.set_tile_stack_shell(true);
        inner.set_recent_projects(session.recent.clone());
        Self {
            inner,
            fonts_installed: false,
            cjk_font: None,
            print_job: None,
            startup_paths,
            provider_watch: ProviderWatch::default(),
            session,
            path_prompt: None,
            pending_pdf: None,
            pending_project_save: None,
            pending_file_write: None,
            window_title: String::new(),
            observed_geometry: None,
            observed_maximized: false,
            geometry_rescued: false,
            closing: false,
        }
    }

    /// Whether anything off-thread is still owed an answer, and therefore
    /// whether the window must keep waking to collect it.
    fn background_work_pending(&self) -> bool {
        self.print_job.is_some() || self.cjk_font.is_some() || self.inner.archive_probe_running()
    }

    /// Whether a destination is still being asked for, so a second gesture of
    /// the same kind must wait rather than replace the first.
    fn asking_for_a_path(&self) -> bool {
        self.path_prompt.is_some()
    }

    /// Where a file dialog should open: the directory this session last used,
    /// then the open project's own directory, then the working directory.
    fn dialog_directory(&self) -> std::path::PathBuf {
        let last = self.session.last_directory.clone().or_else(|| {
            self.inner
                .project_path()
                .and_then(std::path::Path::parent)
                .map(std::path::Path::to_path_buf)
        });
        file_dialog::default_directory(last.as_deref())
    }

    /// Asks for a path: the platform's own dialog where there is one, and the
    /// in-app prompt where there is not.
    ///
    /// Answers [`None`] when the answer is not available *yet* — either the
    /// user cancelled (the caller has already been told nothing happened) or
    /// the prompt is now on screen and the answer arrives in a later frame
    /// through [`Self::poll_path_prompt`].
    fn ask_for_path(&mut self, ask: Ask, suggested_name: &str) -> Option<std::path::PathBuf> {
        let directory = self.dialog_directory();
        if file_dialog::NATIVE_DIALOGS {
            // Chosen or dismissed — either way the ask is finished within this
            // call, and no prompt is left on screen for the caller to wait on.
            return file_dialog::ask_native(ask, suggested_name, Some(&directory));
        }
        self.path_prompt = Some(PathPrompt::new(ask, suggested_name, directory));
        None
    }

    /// Remembers the directory a path landed in, so the next dialog opens
    /// there.
    fn note_dialog_directory(&mut self, path: &std::path::Path) {
        if let Some(parent) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            self.session.last_directory = Some(parent.to_path_buf());
        }
    }

    /// Starts one PDF export on a background thread: asks where the file
    /// should go, then hands the snapshot to the thread and returns — the frame
    /// loop keeps running while the tiles download, and `poll_print_job`
    /// reports the outcome.
    ///
    /// On a platform with no native dialog the ask spans frames, so the
    /// snapshot is parked in `pending_pdf` and `export_pdf_to_path` picks it up
    /// when the prompt answers.
    fn start_pdf_export(&mut self, request: oxigis_ui::print::PrintRequest, ctx: &egui::Context) {
        if self.print_job.is_some() || self.pending_pdf.is_some() {
            self.inner
                .set_status("A PDF export is already running — wait for it to finish.".to_string());
            return;
        }
        if self.asking_for_a_path() {
            self.inner
                .set_status("Finish the open file prompt first.".to_string());
            return;
        }
        let name = export_file_name();
        match self.ask_for_path(Ask::SavePdf, &name) {
            Some(path) => self.export_pdf_to_path(request, path, ctx),
            None if self.path_prompt.is_some() => {
                // The prompt is on screen; the snapshot waits for its answer.
                self.pending_pdf = Some(Box::new(request));
            }
            None => self.inner.set_status("PDF export cancelled.".to_string()),
        }
    }

    /// Hands one settled export to its background thread.
    fn export_pdf_to_path(
        &mut self,
        request: oxigis_ui::print::PrintRequest,
        path: std::path::PathBuf,
        ctx: &egui::Context,
    ) {
        // The native dialogs append the filtered extension themselves; the
        // in-app prompt hands back exactly what was typed, and a name edited
        // down to `map` still means `map.pdf`.
        let path = project_file::with_pdf_extension(&path);
        self.note_dialog_directory(&path);
        let (tx, rx) = std::sync::mpsc::channel();
        let thread_ctx = ctx.clone();
        let thread_path = path.clone();
        let spawned = std::thread::Builder::new()
            .name("oxigis-pdf-export".to_owned())
            .spawn(move || {
                let result = export_pdf_to(&request, &thread_ctx, &thread_path);
                // A dropped receiver just means the app closed mid-export;
                // nothing useful is left to do with the result either way.
                let _ = tx.send(result);
                // The frame loop only polls during frames; make one happen.
                thread_ctx.request_repaint();
            });
        match spawned {
            Ok(_handle) => {
                self.inner
                    .set_status(format!("Exporting PDF to {}…", path.display()));
                self.print_job = Some(PrintJob { result: rx, path });
            }
            Err(error) => {
                self.inner.set_status(format!(
                    "PDF export failed: could not start a thread: {error}"
                ));
            }
        }
    }

    /// Reports a finished background export on the status line, if the one in
    /// flight completed since the last frame.
    fn poll_print_job(&mut self) {
        let Some(job) = self.print_job.as_ref() else {
            return;
        };
        let outcome = match job.result.try_recv() {
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Ok(Ok(report)) => report.status(&job.path),
            Ok(Err(error)) => format!("PDF export failed: {error}"),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                "PDF export failed: the export thread died.".to_string()
            }
        };
        self.inner.set_status(outcome);
        self.print_job = None;
    }

    /// Draws the in-app path prompt, if one is up, and routes the path it
    /// produces to whatever asked for it.
    ///
    /// The prompt is drawn from `ui` (it is a window), but everything it
    /// *decides* is settled here so the four askers share one dispatch.
    fn poll_path_prompt(&mut self, ctx: &egui::Context) {
        let Some(prompt) = self.path_prompt.as_mut() else {
            return;
        };
        let ask = prompt.ask();
        match prompt.show(ctx) {
            PromptOutcome::Pending => {}
            PromptOutcome::Cancelled => {
                self.path_prompt = None;
                self.pending_pdf = None;
                match ask {
                    Ask::SaveProject | Ask::OpenProject => {
                        // The UI may be holding a "save, then quit" behind this
                        // — cancelling has to drop it, or it would fire behind
                        // the next unrelated save.
                        self.pending_project_save = None;
                        self.inner.cancel_pending_project_io();
                    }
                    Ask::SavePdf => self.inner.set_status("PDF export cancelled."),
                    Ask::OpenArchive => self.inner.set_status("No archive was opened."),
                    Ask::SaveGeoJson | Ask::SaveCsv => {
                        if let Some(write) = self.pending_file_write.take() {
                            self.report_file_write_cancelled(&write, "Nothing was exported.");
                        }
                    }
                }
            }
            PromptOutcome::Accepted(path) => {
                // The prompt deliberately outlives the action: a write that
                // fails, or a file that turns out not to be a project, puts
                // its reason back into the box the user typed in instead of
                // dropping their path on the floor. Only a settled ask closes
                // it.
                if self.deliver_path(ask, path, ctx) {
                    self.path_prompt = None;
                }
            }
        }
    }

    /// Acts on a settled path, whichever way it was asked for.
    ///
    /// Answers whether the ask is finished — `false` means it failed in a way
    /// the user can correct by editing the path, so the prompt stays up with
    /// the reason on it.
    fn deliver_path(&mut self, ask: Ask, path: std::path::PathBuf, ctx: &egui::Context) -> bool {
        match ask {
            Ask::SavePdf => {
                match self.pending_pdf.take() {
                    Some(request) => self.export_pdf_to_path(*request, path, ctx),
                    // Cannot happen through either asker, and dropping the path
                    // silently would be the wrong way to be wrong about it.
                    None => self
                        .inner
                        .set_status("The PDF export was no longer waiting for a destination."),
                }
                true
            }
            Ask::SaveProject => match self.pending_project_save.take() {
                Some(request) => {
                    let written = self.write_project(&request.content, path);
                    if !written {
                        // Re-parked so a corrected path can still save these
                        // exact bytes. Whatever the UI had waiting behind the
                        // save is already gone — `report_project_save_failed`
                        // dropped it, which is the safe direction.
                        self.pending_project_save = Some(request);
                    }
                    written
                }
                None => {
                    self.inner
                        .set_status("The project save was no longer waiting for a destination.");
                    true
                }
            },
            Ask::SaveGeoJson | Ask::SaveCsv => match self.pending_file_write.take() {
                Some(write) => self.write_data_file(write, path),
                None => {
                    self.inner
                        .set_status("The export was no longer waiting for a destination.");
                    true
                }
            },
            Ask::OpenProject => self.read_project(path),
            Ask::OpenArchive => {
                self.open_archive(path);
                true
            }
        }
    }

    /// Settles one `take_pending_project_save`: writes straight to a known
    /// path, or asks where the file should go.
    fn resolve_project_save(&mut self, request: oxigis_ui::ProjectSaveRequest) {
        if let Some(path) = request.path.clone() {
            // A known destination: the outcome is already reported to the UI
            // and there is no prompt to keep open over a failure.
            let _written = self.write_project(&request.content, path);
            return;
        }
        if self.asking_for_a_path() {
            // Order matters: `cancel_pending_project_io` writes its own status
            // line, so the reason has to be set after it, not before.
            self.inner.cancel_pending_project_io();
            self.inner
                .set_status("Finish the file prompt that is already open first.".to_string());
            return;
        }
        let suggested = project_file::suggested_file_name(&self.inner.project().name);
        match self.ask_for_path(Ask::SaveProject, &suggested) {
            Some(path) => {
                let _written = self.write_project(&request.content, path);
            }
            // The prompt is on screen; the bytes wait for its answer.
            None if self.path_prompt.is_some() => self.pending_project_save = Some(request),
            None => self.inner.cancel_pending_project_io(),
        }
    }

    /// Writes the project and reports the outcome back to the UI, which is
    /// what clears the unsaved-changes marker and runs anything parked behind
    /// the save. Answers whether the bytes are on disk.
    fn write_project(&mut self, content: &str, path: std::path::PathBuf) -> bool {
        let path = project_file::with_project_extension(&path);
        match project_file::write_project_atomically(&path, content) {
            Ok(()) => {
                tracing::info!(
                    path = %path.display(),
                    bytes = content.len(),
                    "OxiGIS desktop: project saved",
                );
                self.note_dialog_directory(&path);
                self.session.note_recent(path.clone());
                self.inner.confirm_project_saved(path);
                // The recent list the UI now holds is the one that matters —
                // it just gained this file, and `on_exit` reads it back.
                self.session.recent = self.inner.recent_projects().to_vec();
                true
            }
            Err(error) => {
                tracing::error!(path = %path.display(), %error, "OxiGIS desktop: project save failed");
                self.inner.report_project_save_failed(&error);
                // The typed path is kept on screen with the reason, rather than
                // making the user retype it — but only where the prompt was the
                // asker; a native dialog is reopened by the next gesture.
                if let Some(prompt) = self.path_prompt.as_mut() {
                    prompt.reopen_with_error(error);
                }
                false
            }
        }
    }

    /// Settles one `take_pending_project_open`: reads a named file, or asks
    /// which one.
    fn resolve_project_open(&mut self, request: oxigis_ui::ProjectOpenRequest) {
        if let Some(path) = request.path {
            let _opened = self.read_project(path);
            return;
        }
        if self.asking_for_a_path() {
            // Order matters: `cancel_pending_project_io` writes its own status
            // line, so the reason has to be set after it, not before.
            self.inner.cancel_pending_project_io();
            self.inner
                .set_status("Finish the file prompt that is already open first.".to_string());
            return;
        }
        let suggested = project_file::suggested_file_name(&self.inner.project().name);
        match self.ask_for_path(Ask::OpenProject, &suggested) {
            Some(path) => {
                let _opened = self.read_project(path);
            }
            None if self.path_prompt.is_some() => {}
            None => self.inner.cancel_pending_project_io(),
        }
    }

    /// Reads a project file and installs it, keeping the path so a later plain
    /// Ctrl+S writes back to the same place.
    ///
    /// Both document formats go through `oxigis-ui`'s own detection ladder
    /// (native first, then the GeoLibre compat reader), which is the same one
    /// the paste modal and a `.geolibre.json` drop use — so a file opened here
    /// and a file dropped on the map cannot disagree about what it is.
    /// Answers whether a project was really opened; a file that could not be
    /// read or did not parse leaves the open project — and its path — exactly
    /// as they were, so the next Ctrl+S cannot land on a document this shell
    /// failed to understand.
    fn read_project(&mut self, path: std::path::PathBuf) -> bool {
        let text = match project_file::read_project_text(&path) {
            Ok(text) => text,
            Err(error) => {
                tracing::error!(path = %path.display(), %error, "OxiGIS desktop: project open failed");
                self.inner.set_status(format!("Could not open: {error}"));
                if let Some(prompt) = self.path_prompt.as_mut() {
                    prompt.reopen_with_error(error);
                }
                return false;
            }
        };
        if !self.inner.load_project_from_text(&text) {
            // `load_project_from_text` has already put the parse failure on the
            // status line; the path is not remembered, because nothing was
            // opened.
            if let Some(reason) = self.inner.status().map(str::to_owned)
                && let Some(prompt) = self.path_prompt.as_mut()
            {
                prompt.reopen_with_error(reason);
            }
            return false;
        }
        tracing::info!(path = %path.display(), "OxiGIS desktop: project opened");
        self.note_dialog_directory(&path);
        self.session.note_recent(path.clone());
        self.inner.set_project_path(Some(path));
        self.session.recent = self.inner.recent_projects().to_vec();
        true
    }

    /// Starts the probe for a tile archive the user picked.
    fn open_archive(&mut self, path: std::path::PathBuf) {
        self.note_dialog_directory(&path);
        let path = path.display().to_string();
        let format = oxigis_ui::format_for_url(&path);
        let _accepted = self
            .inner
            .request_archive_probe(oxigis_core::ArchiveRef::Path { path }, format);
    }

    /// Intercepts the window's close request until the unsaved-changes
    /// question has an answer.
    ///
    /// Three states, and the latch matters: before the user has answered, the
    /// close is refused and the question asked; once
    /// [`oxigis_ui::OxigisApp::take_confirmed_close`] answers `true` the shell
    /// latches `closing` and stops intercepting for good — confirming does not
    /// make the project clean, so an un-latched shell would re-ask the question
    /// about the very `Close` command it had just sent, for ever.
    fn intercept_close(&mut self, ctx: &egui::Context) {
        if self.closing {
            return;
        }
        if self.inner.take_confirmed_close() {
            self.closing = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if !ctx.input(|input| input.viewport().close_requested()) {
            return;
        }
        self.inner.request_window_close();
        if self.inner.take_confirmed_close() {
            // A project with nothing to lose answered within the call: let the
            // close request that is already in flight through untouched, rather
            // than cancelling it and issuing a fresh one in the same frame.
            self.closing = true;
            return;
        }
        // Unanswered: refuse this close. The modal is now on screen, and the
        // branch above closes the window on whichever later frame the user
        // says Discard (or the save behind "Save…" lands).
        ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
    }

    /// Keeps the window title showing the project name and its dirty marker.
    ///
    /// Only sent when it actually changed: a `ViewportCommand` per frame would
    /// have the window manager re-reading the title sixty times a second.
    fn sync_window_title(&mut self, ctx: &egui::Context) {
        let title = self.inner.window_title();
        if title == self.window_title {
            return;
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
        self.window_title = title;
    }

    /// Notes where the window is, for [`eframe::App::on_exit`] to persist.
    ///
    /// A frame that reports no rectangle — Wayland never reports one — leaves
    /// the last known geometry standing rather than replacing it with nothing,
    /// and a maximized or minimized window is not recorded as its *restore*
    /// rectangle either, so un-maximizing after a restart lands where the user
    /// left it rather than filling the screen for ever.
    fn observe_geometry(&mut self, ctx: &egui::Context) {
        let (rect, monitor, maximized, minimized) = ctx.input(|input| {
            let viewport = input.viewport();
            (
                viewport.outer_rect.or(viewport.inner_rect),
                viewport.monitor_size,
                viewport.maximized.unwrap_or(false),
                viewport.minimized.unwrap_or(false),
            )
        });
        self.rescue_offscreen_window(ctx, rect, monitor);
        self.observed_maximized = maximized;
        if maximized || minimized {
            return;
        }
        let Some(rect) = rect else {
            return;
        };
        let geometry = session::WindowGeometry {
            x: rect.min.x,
            y: rect.min.y,
            width: rect.width(),
            height: rect.height(),
        };
        if let Some(sane) = geometry.sanitized() {
            self.observed_geometry = Some(sane);
        }
    }

    /// Drags a restored window back onto the screen when the display it was
    /// saved on is gone.
    ///
    /// The one geometry failure a user cannot fix with the mouse: a window
    /// placed at the coordinates of a monitor that has since been unplugged is
    /// invisible and un-grabbable, and every launch repeats it. Checked once,
    /// on the first frame that reports both a window rectangle and a monitor
    /// size — neither exists before the window does.
    fn rescue_offscreen_window(
        &mut self,
        ctx: &egui::Context,
        rect: Option<egui::Rect>,
        monitor: Option<egui::Vec2>,
    ) {
        if self.geometry_rescued {
            return;
        }
        let (Some(rect), Some(monitor)) = (rect, monitor) else {
            return;
        };
        self.geometry_rescued = true;
        let geometry = session::WindowGeometry {
            x: rect.min.x,
            y: rect.min.y,
            width: rect.width(),
            height: rect.height(),
        };
        if geometry.is_reachable_on([monitor.x, monitor.y]) {
            return;
        }
        // Centred on the monitor that IS there, clamped to non-negative so the
        // title bar cannot end up above the top edge on a small display.
        let x = ((monitor.x - geometry.width) / 2.0).max(0.0);
        let y = ((monitor.y - geometry.height) / 2.0).max(0.0);
        tracing::info!(
            from = ?(geometry.x, geometry.y),
            to = ?(x, y),
            "OxiGIS desktop: the remembered window position is off every monitor; recentring",
        );
        ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(x, y)));
    }
}

impl eframe::App for OxigisDesktopApp {
    /// Every poll, drain and settle step the shell owns.
    ///
    /// Deliberately not `ui`: eframe calls `logic` before each `ui` *and*
    /// while the window is hidden (whenever a repaint was requested), while
    /// `ui` is skipped entirely for an invisible viewport. A PDF export, an
    /// archive probe or the font chain that finishes behind a minimised
    /// window is therefore settled when it finishes rather than when the user
    /// restores the window — which is what kept `print_job` occupied and
    /// refused the next export with "already running".
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Idempotent: installs the tile renderer on the first frame that has a
        // `wgpu` render state, and selects the non-GPU fallback if there never
        // is one. The closure runs only on that one frame, so the HTTP worker
        // pool is created once.
        //
        // `attach_gpu_map_using` (not `_with`) hands the active basemap to the
        // factory: the shell used to clone the whole `BasemapConfig` — two
        // `String`s and a `Vec<String>` — on EVERY frame to satisfy the borrow
        // checker, for a closure that runs at most once in the process's life.
        let mut attached = None;
        self.inner
            .attach_gpu_map_using(frame.wgpu_render_state(), |config| {
                let built = build_tile_provider(config, ctx)?;
                attached = Some(built.watch);
                Some(built.provider)
            });
        if let Some(watch) = attached {
            self.provider_watch.install_raster(watch);
        }
        // The command line, on the first frame that can turn a file into a
        // layer: the reads queue local-vector ops, which `apply_local_ops`
        // below applies against the map attached just above.
        if !self.startup_paths.is_empty() {
            let paths = core::mem::take(&mut self.startup_paths);
            open_startup_paths(&mut self.inner, paths);
        }
        // Fonts for the label pass. Independent of any layer: installing them
        // costs nothing until a vector style with symbol rules is attached, and
        // doing it here means the very first vector frame already has text.
        if let Some(render_state) = frame.wgpu_render_state() {
            if !self.fonts_installed {
                self.fonts_installed = install_label_fonts(render_state);
                if self.fonts_installed {
                    self.cjk_font = start_cjk_font_scan(ctx);
                }
            }
            if let Some(rx) = self.cjk_font.as_ref()
                && drain_cjk_font(render_state, rx, ctx)
            {
                self.cjk_font = None;
            }
        }
        // Providers are DERIVED from the project (editing v1.3): an add, a
        // remove, an undo of either, and a project load all reach the GPU
        // through this one reconciliation block. The work stays offered until
        // it is settled, so a frame without a render state DEFERS an install
        // instead of losing it.
        if let Some(render_state) = frame.wgpu_render_state() {
            // The BASEMAP, and only the basemap. This shell declared
            // `set_tile_stack_shell(true)`, so `pending_raster_work` offers a
            // basemap-only plan and `pending_vector_work` offers nothing at
            // all: every COG, archive, XYZ overlay and MVT source is a stack
            // entry now, and driving both seams for one layer would draw it
            // twice.
            if let Some(work) = self.inner.pending_raster_work() {
                let outcome = match build_tile_provider(&work.basemap, ctx) {
                    Some(built) => {
                        if oxigis_ui::map_gpu::replace_provider(render_state, built.provider) {
                            // The handles go with the provider that is now
                            // drawing; whatever the last one had to say is
                            // no longer about the map. The stack's watches are
                            // untouched — they are still drawing over this.
                            self.provider_watch.install_raster(built.watch);
                            Ok(())
                        } else {
                            Err("the GPU map is not attached".to_string())
                        }
                    }
                    None => Err("the tile provider could not be built".to_string()),
                };
                self.inner.settle_raster_work(work, outcome);
            }
            // The N-layer stack: what is installed is read back from the map
            // itself, so this is a pure diff with no second mirror to go
            // stale. One unit of work per frame, so a frame does at most one
            // provider build.
            let installed = oxigis_ui::map_gpu::installed_tile_stack(render_state);
            if let Some(work) = self.inner.tile_stack_work(&installed) {
                match work {
                    oxigis_ui::TileStackWork::Install(plan) => {
                        let (bytes, reader) = stack_archive_data(&self.inner, &plan);
                        let layer = plan.layer;
                        let (source, watch) = match build_stack_source(&plan, ctx, bytes, reader) {
                            Ok((source, watch)) => (Ok(source), Some(watch)),
                            Err(reason) => (Err(reason), None),
                        };
                        // Handed over even on failure: the refusal occupies
                        // the slot, which is what stops the plan being offered
                        // once per frame for ever.
                        let outcome =
                            oxigis_ui::map_gpu::install_tile_layer(render_state, plan, source);
                        match outcome {
                            Ok(()) => {
                                if let Some(watch) = watch {
                                    self.provider_watch.install_stack(layer, watch);
                                }
                            }
                            Err(reason) => {
                                // Whatever was watching this slot is not
                                // drawing any more.
                                self.provider_watch.remove_stack(&[layer]);
                                self.inner.set_status(reason);
                            }
                        }
                    }
                    oxigis_ui::TileStackWork::Remove(layers) => {
                        let _ = oxigis_ui::map_gpu::remove_tile_layers(render_state, &layers);
                        self.provider_watch.remove_stack(&layers);
                    }
                    // Rebuilds nothing, so no watch moves: a drag in the layer
                    // panel must not re-fetch a single tile.
                    oxigis_ui::TileStackWork::Reorder(order) => {
                        let _ = oxigis_ui::map_gpu::reorder_tile_layers(render_state, &order);
                    }
                }
            }
            // Unconditional and cheap: an opacity is an instance tint, so
            // following a slider costs neither a texture nor a tessellation.
            let app = &self.inner;
            oxigis_ui::map_gpu::sync_tile_layer_opacities(render_state, |id| {
                app.tile_layer_opacity(id)
            });
            // The banner's half of the stack: refusals are GPU state, so the
            // shell is the only place they can be read from.
            self.inner
                .set_tile_layer_refusals(oxigis_ui::map_gpu::tile_layer_refusals(render_state));
            if self.inner.take_tile_layer_retry() {
                let dropped = oxigis_ui::map_gpu::retry_refused_tile_layers(render_state);
                if dropped > 0 {
                    // The plans are offered again from the next frame; the
                    // watches went with the entries that held them.
                    tracing::info!(dropped, "OxiGIS desktop: refused stack entries re-offered");
                }
            }
        }
        // A tile archive the user asked for: `oxigis-ui` owns no transport, so
        // the shell builds one and hands the probe back. The layer is created
        // by `poll_archive_probe` when the header lands -- see the app's
        // `archive_io` module for why the two halves are separate.
        if self.inner.take_pending_archive_pick() {
            if self.asking_for_a_path() {
                self.inner
                    .set_status("Finish the open file prompt first.".to_string());
            } else {
                match self.ask_for_path(Ask::OpenArchive, "") {
                    Some(path) => self.open_archive(path),
                    // The prompt is now on screen; its answer arrives in a
                    // later frame through `poll_path_prompt`.
                    None if self.path_prompt.is_some() => {}
                    None => self.inner.set_status("No archive was opened."),
                }
            }
        }
        if let Some(request) = self.inner.take_pending_archive_probe() {
            let config = ArchiveLayerConfig::new(request.archive.clone(), request.format);
            // Both formats are probed the same way: a `.mbtiles` is surveyed in
            // one 16 KiB read exactly as a `.pmtiles` is, so its refusals land
            // BEFORE a layer exists. Bytes the app already holds (a drop it read
            // itself) never reach here — `open_archive_bytes` answers those.
            let bytes = self.inner.archive_bytes(config.location());
            match build_archive_transport(&config, bytes) {
                Some(transport) => {
                    self.inner.attach_archive_probe(ArchiveProbe::start(
                        config.location().to_owned(),
                        request.format,
                        ctx,
                        transport,
                    ));
                }
                None => self
                    .inner
                    .set_status("The archive could not be read: no worker thread started."),
            }
        }
        let _created = self.inner.poll_archive_probe();
        // A queued PDF export runs on its own thread (`start_pdf_export`), so
        // the window stays live while the tiles download; the result lands on
        // the status line whenever the thread finishes.
        self.poll_print_job();
        if let Some(request) = self.inner.take_pending_print() {
            self.start_pdf_export(request, ctx);
        }
        // Local GeoJSON: read whatever the app is waiting on (this queues `Add`
        // ops), then apply every queued op to the GPU map.
        drain_dropped_paths(&mut self.inner);
        apply_local_ops(&mut self.inner, frame.wgpu_render_state());
        // A COG that 404s, an archive the server changed underneath us, a
        // basemap answering only 403: every one of those is discovered by the
        // provider long after it was built, and this is the one place that
        // asks.
        if let Some(message) = self.provider_watch.poll() {
            self.inner.set_status(message);
        }
        // Project file I/O: the two take-once seams `oxigis-ui` offers a shell
        // that declared `set_native_project_io(true)`.
        if let Some(request) = self.inner.take_pending_project_save() {
            self.resolve_project_save(request);
        }
        if let Some(request) = self.inner.take_pending_project_open() {
            self.resolve_project_open(request);
        }
        // The two data-file seams, drained in the same block and through the
        // same resolver: a Processing result the user asked to keep, and a
        // layer/table export. Without these the app records the request and
        // says so instead of writing anything.
        if let Some(request) = self.inner.take_pending_processing_save() {
            self.resolve_file_write(PendingFileWrite::Processing(request));
        }
        if let Some(request) = self.inner.take_pending_export() {
            self.resolve_file_write(PendingFileWrite::Export(request));
        }
        self.sync_window_title(ctx);
        self.observe_geometry(ctx);
        // Last, and after every seam above: the close question must be asked
        // against the state this frame actually leaves behind, and the
        // `CancelClose` has to be sent on the same frame the request arrives.
        self.intercept_close(ctx);
        if self.background_work_pending() {
            ctx.request_repaint_after(BACKGROUND_POLL_INTERVAL);
        }
    }

    /// Draws the panels. Everything that is not drawing lives in
    /// `Self::logic`, which also runs while the window is hidden.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.inner.ui(ui);
        // After the app's own panels, so the prompt is a modal ON TOP of them
        // rather than something the map can be dragged over.
        let ctx = ui.ctx().clone();
        self.poll_path_prompt(&ctx);
    }

    /// Persists the session on the way out.
    ///
    /// `on_exit` is NOT gated on eframe's `persistence` feature (only `save`
    /// is), which is what lets this shell remember a window without pulling
    /// `ron` and `serde` into the binary — see `session`'s module docs. It also
    /// runs on the paths the close-request intercept never sees, macOS's
    /// Cmd+Q among them, which is the other reason the session write lives
    /// here rather than beside the confirmation.
    fn on_exit(&mut self) {
        self.session.window = self.observed_geometry;
        self.session.maximized = self.observed_maximized;
        self.session.recent = self.inner.recent_projects().to_vec();
        match session::store(&self.session) {
            Ok(()) => tracing::info!("OxiGIS desktop: session saved"),
            // A window position that could not be remembered is not worth
            // failing a shutdown over.
            Err(error) => tracing::warn!(%error, "OxiGIS desktop: could not save the session"),
        }
    }
}

/// The application id this shell registers itself under.
///
/// On Wayland this is what matches the window to its `.desktop` file, and
/// therefore what a taskbar groups and icons by; on X11 it becomes the
/// `WM_CLASS`. Reverse-DNS, matching the project's own domain, because that is
/// the only form a desktop file's name can take.
const APP_ID: &str = "io.cooljapan.oxigis";

/// Builds the window this launch opens, restoring the remembered geometry.
///
/// The restore is deliberately conservative. A saved rectangle is sanitized
/// (finite, not degenerate, not absurdly far off the origin) before it is
/// used at all, and the *position* is only applied when a size was restored
/// too — half a remembered window is worse than none. A geometry from a
/// monitor that is no longer plugged in is caught later, on the first frame
/// that can see the monitor: there is no monitor list before the window
/// exists.
fn viewport_builder(session: &session::SessionState) -> egui::ViewportBuilder {
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("OxiGIS")
        .with_app_id(APP_ID);
    match session.window.and_then(session::WindowGeometry::sanitized) {
        Some(window) => {
            viewport = viewport
                .with_inner_size(egui::vec2(window.width, window.height))
                .with_position(egui::pos2(window.x, window.y));
        }
        None => {
            viewport = viewport.with_inner_size(egui::vec2(
                session::DEFAULT_WINDOW_SIZE[0],
                session::DEFAULT_WINDOW_SIZE[1],
            ));
        }
    }
    if session.maximized {
        viewport = viewport.with_maximized(true);
    }
    viewport
}

/// Entry point: reads the command line, initializes logging and runs the
/// native `eframe` window.
///
/// Errors from `eframe::run_native` are logged and turned into a non-zero
/// process exit rather than unwrapped, per COOLJAPAN policy.
fn main() -> std::process::ExitCode {
    let (paths, log_file) = match cli::parse_args(std::env::args_os().skip(1)) {
        Startup::Run { paths, log_file } => (paths, log_file),
        Startup::Help => {
            println!("{USAGE}");
            return std::process::ExitCode::SUCCESS;
        }
        Startup::Version => {
            println!("{}", cli::version_line());
            return std::process::ExitCode::SUCCESS;
        }
        Startup::Refused(complaint) => {
            eprintln!("oxigis: {complaint}\n\n{USAGE}");
            return std::process::ExitCode::FAILURE;
        }
    };
    cli::init_logging(log_file.as_deref());
    tracing::info!(
        core = oxigis_core::VERSION,
        render = oxigis_render::VERSION,
        ui = oxigis_ui::VERSION,
        paths = paths.len(),
        "OxiGIS desktop shell starting",
    );

    let session = session::load();
    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: viewport_builder(&session),
        ..Default::default()
    };

    // Moved into the creation closure, which eframe calls exactly once.
    let mut startup = Some((paths, session));
    let run_result = eframe::run_native(
        "OxiGIS",
        native_options,
        Box::new(move |_creation_context| {
            let (paths, session) = startup.take().unwrap_or_default();
            Ok(Box::new(OxigisDesktopApp::new(paths, session)))
        }),
    );

    match run_result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(%error, "OxiGIS desktop shell exited with an error");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live: the whole PDF export path against real OSM tiles — provider, tile
    /// polling, composition, assembly, and the file on disk. Ignored by
    /// default; run with `cargo nextest run -p oxigis-desktop --run-ignored=all`.
    #[test]
    #[ignore = "network: fetches real tiles from tile.openstreetmap.org and writes a PDF"]
    fn live_pdf_export_round_trip() {
        let view = oxigis_render::MapView::new(
            oxigis_render::LonLat::new(139.767, 35.681),
            10.0,
            [1024.0, 768.0],
        )
        .expect("a valid viewport");
        let request = oxigis_ui::print::PrintRequest {
            // The live-export tests build their page from the fields above;
            // the N-layer snapshot is what the app fills in for a real export.
            stack: Vec::new(),
            // Japanese in the title: the live run also proves the CJK
            // embedded-font path against this machine's real font chain.
            title: "OxiGIS live export — 東京".to_string(),
            attribution: oxigis_ui::OSM_ATTRIBUTION.to_string(),
            view,
            basemap: oxigis_ui::BasemapConfig::openstreetmap(),
            cog: None,
            archive: None,
            vector: None,
            layers: Vec::new(),
            options: oxigis_ui::print::PrintOptions::default(),
        };
        let ctx = egui::Context::default();
        let path = std::env::temp_dir().join(export_file_name());
        export_pdf_to(&request, &ctx, &path).expect("the export must succeed");
        let bytes = std::fs::read(&path).expect("the exported file exists");
        let _removed = std::fs::remove_file(&path);
        assert!(bytes.starts_with(b"%PDF-"), "the magic bytes lead");
        assert!(
            bytes.len() > 50_000,
            "a real Tokyo basemap image is not {} bytes",
            bytes.len()
        );
        // The Japanese title must have engaged the embedded-font path: at
        // least two composite fonts (the bundled Latin Noto plus a system
        // CJK face from the scan chain) with Identity-H encoding.
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.matches("/Subtype /Type0").count() >= 2,
            "expected the Latin face AND a system CJK face to be embedded"
        );
        assert!(text.contains("/Encoding /Identity-H"));
        assert!(text.contains("/FontFile2"), "a subset program is embedded");
    }

    /// Live: the PDF export with the MapLibre demo vector-tile source — the
    /// streamed tiles must be fetched, decoded and drawn as vector paths
    /// under their per-rule alpha states.
    #[test]
    #[ignore = "network: fetches real tiles from demotiles.maplibre.org and OSM, writes a PDF"]
    fn live_pdf_export_with_vector_tiles() {
        let view = oxigis_render::MapView::new(
            oxigis_render::LonLat::new(10.0, 45.0),
            3.0,
            [1024.0, 768.0],
        )
        .expect("a valid viewport");
        let request = oxigis_ui::print::PrintRequest {
            // The live-export tests build their page from the fields above;
            // the N-layer snapshot is what the app fills in for a real export.
            stack: Vec::new(),
            title: "OxiGIS vector live export".to_string(),
            attribution: oxigis_ui::MAPLIBRE_ATTRIBUTION.to_string(),
            view,
            basemap: oxigis_ui::BasemapConfig::openstreetmap(),
            cog: None,
            archive: None,
            vector: Some(oxigis_ui::VectorTileConfig::maplibre_demo()),
            layers: Vec::new(),
            options: oxigis_ui::print::PrintOptions::default(),
        };
        let ctx = egui::Context::default();
        let path = std::env::temp_dir().join(export_file_name());
        export_pdf_to(&request, &ctx, &path).expect("the export must succeed");
        let bytes = std::fs::read(&path).expect("the exported file exists");
        let _removed = std::fs::remove_file(&path);
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.contains("/GV0"),
            "the vector rules' alpha states must be registered — were tiles drawn?"
        );
    }

    /// The OS font tree is walked once per process: the export path and the
    /// startup scan must be handed the very same answer, not a fresh one.
    #[test]
    fn the_cjk_chains_are_resolved_once() {
        assert!(std::sync::Arc::ptr_eq(
            &cjk_regular_paths(),
            &cjk_regular_paths()
        ));
        assert!(std::sync::Arc::ptr_eq(&cjk_bold_paths(), &cjk_bold_paths()));
    }

    /// An export that timed out on tiles is not a plain success: the page has
    /// gray rectangles on it and the status line has to say so.
    #[test]
    fn the_export_status_names_what_is_missing() {
        let path = std::path::Path::new("map.pdf");
        let clean = ExportReport {
            missing_raster: 0,
            missing_vector: 0,
            total: 12,
        };
        assert_eq!(clean.status(path), "PDF exported to map.pdf.");
        let short = ExportReport {
            missing_raster: 3,
            missing_vector: 2,
            total: 12,
        };
        let status = short.status(path);
        assert!(status.contains("3 of 12 basemap tiles"), "{status}");
        assert!(status.contains("2 of 12 vector tiles"), "{status}");
    }

    /// A big page is not judged by a small page's clock — and no page holds
    /// the export thread past the ceiling.
    #[test]
    fn the_tile_budget_grows_with_the_page_and_then_stops() {
        assert_eq!(tile_budget(0), PRINT_TILE_TIMEOUT);
        assert_eq!(
            tile_budget(100),
            PRINT_TILE_TIMEOUT + PRINT_TILE_ALLOWANCE * 100,
        );
        assert!(tile_budget(12) > tile_budget(4));
        assert_eq!(tile_budget(100_000), PRINT_TILE_MAX_WAIT);
        // usize::MAX does not fit the `u32` the multiplication takes; the
        // ceiling is the answer, not an overflow.
        assert_eq!(tile_budget(usize::MAX), PRINT_TILE_MAX_WAIT);
    }
}
