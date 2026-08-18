//! Real XYZ raster basemap tiles: the shell-agnostic half of the fetch loop.
//!
//! [`crate::map_gpu::TileProvider`] is a synchronous, non-blocking seam called
//! from egui's `wgpu` prepare hook, while HTTP is asynchronous on every target
//! OxiGIS ships. [`XyzTileProvider`] bridges the two:
//!
//! ```text
//! frame N     provider.tile(t) -> None, marks `t` in flight, asks the transport
//! (off-frame) transport fetches the URL, TileSink::deliver decodes + stores +
//!             ctx.request_repaint()
//! frame N+k   provider.tile(t) -> Some(pixels)   (renderer uploads a texture)
//! ```
//!
//! # What is platform-specific and what is not
//!
//! Everything here — URL expansion, the in-flight set, the decoded-tile LRU,
//! the failure bookkeeping, the repaint kick — is portable. The single
//! platform-dependent step, "given a URL, eventually produce bytes", is the
//! [`TileTransport`] trait, implemented by `oxigis-desktop` (a blocking
//! pure-Rust HTTP client on a small worker pool) and by `oxigis-web`
//! (the browser `fetch()` API on `spawn_local`). This crate therefore stays
//! free of both an HTTP stack and `web-sys`, and compiles unchanged for
//! `wasm32-unknown-unknown`.
//!
//! # Bounded resources
//!
//! | Resource | Bound | Constant |
//! |---|---|---|
//! | Concurrent requests | 16 | [`MAX_INFLIGHT_TILES`] |
//! | Decoded tiles held for the renderer | 64 (LRU) | [`READY_CACHE_TILES`] |
//! | Remembered failures | 1024 (LRU) | [`FAILURE_MEMORY_TILES`] |
//! | Attempts per tile | 3, transient failures only | [`MAX_ATTEMPTS`] |
//!
//! Every one of those is a hard cap, so a long panning session cannot grow the
//! provider's memory without limit: 64 × 256 × 256 × 4 B ≈ 16 MiB of decoded
//! pixels in the worst case, plus at most 16 in-flight response buffers.
//!
//! # Retry policy
//!
//! A failure is classified by the transport as *permanent* or *transient*
//! ([`TileError`]):
//!
//! * **Permanent** — HTTP 4xx (404 missing tile, 403/429 policy blocks), an
//!   unusable URL, an undecodable body. Recorded once and **never retried for
//!   the rest of the session**: retrying cannot help, and against a tile server
//!   that is rate-limiting us it actively makes things worse.
//! * **Transient** — a transport/IO error or HTTP 5xx. Retried up to
//!   [`MAX_ATTEMPTS`] attempts in total, after which the tile is treated as
//!   permanently failed for the rest of the session (until the failure LRU
//!   evicts it, or [`XyzTileProvider::retry_failed_tiles`] is called).
//!
//! A retry is *wall-clock-gated*, not frame-gated: `FailureState` pairs the
//! attempt count with the earliest time another attempt may be made,
//! backing off exponentially from [`RETRY_BASE_DELAY_SECS`] and doubling per
//! attempt up to [`RETRY_MAX_DELAY_SECS`]. The renderer still asks for the
//! tile again every frame — that part stays simple — but a frame before
//! `retry_at` answers [`None`] without touching the transport, so a
//! one-second network hiccup no longer burns the whole attempt budget in the
//! time it takes to draw three frames. A successful fetch clears the tile's
//! entry outright, so a later failure of the *same* tile backs off from
//! scratch rather than inheriting an old attempt count.

use std::collections::HashSet;
use std::sync::Arc;

use oxigis_render::{DecodedTile, RenderError, TileCache, TileId, XyzTemplate, decode_tile};
use parking_lot::Mutex;

use crate::map_gpu::TileProvider;

/// Maximum number of tile requests allowed to be outstanding at once.
///
/// Reached when the user zooms out far enough that a whole new pyramid level
/// becomes visible in one frame; the remaining tiles are simply requested over
/// the following frames, since the renderer re-reports what it is missing every
/// [`oxigis_render::MapRenderer::begin_frame`].
pub const MAX_INFLIGHT_TILES: usize = 16;

/// Number of decoded tiles kept ready for the renderer to pick up.
///
/// The renderer only asks for tiles whose texture is *not* resident, so this
/// cache is what makes panning back to a recently visited area free rather than
/// a second round of HTTP requests.
pub const READY_CACHE_TILES: usize = 64;

/// Number of tile addresses whose failure is remembered.
///
/// Bounded (and LRU) so a session spent panning across a sparse tile set cannot
/// accumulate an unbounded failure table; the cost of forgetting the oldest
/// entry is one extra request should that tile become visible again.
pub const FAILURE_MEMORY_TILES: usize = 1_024;

/// Total number of attempts a single tile gets before it is abandoned.
///
/// Only transient failures consume attempts — see the [module docs][self].
pub const MAX_ATTEMPTS: u32 = 3;

/// Base delay before a retryable failure's first retry, in seconds.
///
/// Doubled per additional attempt (see `FailureState::after_failure`), so a
/// one-off network hiccup (a DNS blip, a connection reset, a laptop resuming
/// from sleep) is retried within a couple of seconds rather than burning the
/// whole attempt budget within the time it takes to draw three frames.
pub const RETRY_BASE_DELAY_SECS: f64 = 0.5;

/// Ceiling the exponential backoff in `FailureState::after_failure` is
/// clamped to, in seconds.
pub const RETRY_MAX_DELAY_SECS: f64 = 8.0;

/// Longest error message [`TileHealth::last_error`] keeps verbatim.
///
/// A transport or a misconfigured host can hand back an arbitrarily long
/// message (a proxy's HTML error page echoed into an error string); this
/// keeps a provider's memory and a status line that displays it bounded
/// regardless of what the network sends.
pub const MAX_STORED_ERROR_CHARS: usize = 200;

/// The `url` an archive-backed provider hands its transport.
///
/// A single-file tile archive (PMTiles, MBTiles) addresses tiles by [`TileId`]
/// *inside* one already-named file, so there is no per-tile URL to expand. A
/// named sentinel rather than an empty string, so a URL transport that ever
/// receives one can say what happened instead of fetching nothing:
/// [`crate::archive::ArchiveTileTransport`] ignores it, and every other
/// transport should refuse it by name.
pub const ARCHIVE_TILE_URL: &str = "oxigis:archive";

/// URL template of the default basemap: the OpenStreetMap standard style.
pub const OSM_URL_TEMPLATE: &str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

/// Attribution text the OSM tile usage policy requires to be displayed.
pub const OSM_ATTRIBUTION: &str = "© OpenStreetMap contributors";

/// A ready-to-use basemap the layer panel offers as a one-click sample.
///
/// Each entry carries the exact credit line its service's terms require —
/// shipping the credit together with the URL is what makes the sample usable
/// as-is, instead of being a URL the user must research before they may
/// legally display it. Only services whose published terms allow this kind of
/// embedding are listed; `terms` summarises the licence for the picker's
/// hover text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BasemapPreset {
    /// Short human-readable name shown in the basemap picker.
    pub name: &'static str,
    /// URL template, as understood by [`XyzTemplate`]. Placeholder order is
    /// free, so WMTS-REST `{z}/{y}/{x}` endpoints fit unchanged.
    pub url_template: &'static str,
    /// The credit line the service's terms require the map to display.
    pub attribution: &'static str,
    /// One-line licence summary, shown when hovering the picker entry.
    pub terms: &'static str,
}

impl BasemapPreset {
    /// The preset as a live [`BasemapConfig`].
    #[must_use]
    pub fn config(&self) -> BasemapConfig {
        BasemapConfig {
            url_template: self.url_template.to_string(),
            subdomains: Vec::new(),
            attribution: self.attribution.to_string(),
        }
    }

    /// Whether `config` is exactly this preset — URL, credit line and (empty)
    /// subdomain list.
    ///
    /// Matching on the URL alone would be a lie in the picker: a custom
    /// submission of the same URL installs a generic host credit, not the
    /// credit line this preset's terms require, and claiming the preset is
    /// active would claim that required credit is on the map when it is not.
    #[must_use]
    pub fn matches(&self, config: &BasemapConfig) -> bool {
        config.url_template == self.url_template
            && config.attribution == self.attribution
            && config.subdomains.is_empty()
    }
}

/// The built-in basemap samples, default first.
///
/// The EOX entries' layer ids, tile formats and credit wordings come from the
/// service's own WMTS capabilities document
/// (<https://tiles.maps.eox.at/wmts/1.0.0/WMTSCapabilities.xml>). Note the
/// licence split it declares: the yearly Sentinel-2 cloudless mosaics from
/// 2018 onwards are CC BY-NC-SA 4.0 (non-commercial), while the original 2016
/// mosaic is plain CC BY 4.0 — which is why both years are offered.
pub const BASEMAP_PRESETS: &[BasemapPreset] = &[
    BasemapPreset {
        name: "OpenStreetMap",
        url_template: OSM_URL_TEMPLATE,
        attribution: OSM_ATTRIBUTION,
        terms: "ODbL data; OpenStreetMap Foundation tile usage policy applies",
    },
    BasemapPreset {
        name: "Sentinel-2 cloudless 2024 (EOX)",
        url_template: "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/\
                       default/GoogleMapsCompatible/{z}/{y}/{x}.jpg",
        attribution: "EOxCloudless https://cloudless.eox.at by EOX IT Services GmbH \
                      (Contains modified Copernicus Sentinel data 2024)",
        terms: "CC BY-NC-SA 4.0 — non-commercial use only; \
                commercial licences via https://cloudless.eox.at",
    },
    BasemapPreset {
        name: "Sentinel-2 cloudless 2016 (EOX)",
        url_template: "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/\
                       default/GoogleMapsCompatible/{z}/{y}/{x}.jpg",
        attribution: "EOxCloudless https://cloudless.eox.at by EOX IT Services GmbH \
                      (Contains modified Copernicus Sentinel data 2016)",
        terms: "CC BY 4.0 — free with the shown credit, commercial use included",
    },
    BasemapPreset {
        name: "Terrain Light (EOX)",
        url_template: "https://tiles.maps.eox.at/wmts/1.0.0/terrain-light_3857/\
                       default/GoogleMapsCompatible/{z}/{y}/{x}.jpg",
        attribution: "Terrain Light { Data © OpenStreetMap contributors and others, \
                      Rendering © EOX }",
        terms: "Free with the shown credit (EOX::Maps)",
    },
    BasemapPreset {
        name: "OpenTopoMap",
        url_template: "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)",
        terms: "CC BY-SA — free with the shown credit",
    },
];

/// Which raster basemap the map panel draws, and the credit line it must show.
///
/// Defaults to OpenStreetMap's standard tiles. Both fields are plain strings so
/// a shell (or a future preferences dialog) can point the map at any XYZ
/// service without touching the fetch machinery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasemapConfig {
    /// `{z}/{x}/{y}` URL template, as understood by [`XyzTemplate`].
    pub url_template: String,
    /// Hosts `{s}` rotates through; empty when the template has no `{s}`.
    pub subdomains: Vec<String>,
    /// Credit line drawn in the corner of the map panel. Empty hides it.
    ///
    /// Removing it for an OSM-hosted basemap violates the
    /// [tile usage policy](https://operations.osmfoundation.org/policies/tiles/).
    pub attribution: String,
}

impl BasemapConfig {
    /// The OpenStreetMap standard basemap with its required attribution.
    #[must_use]
    pub fn openstreetmap() -> Self {
        Self {
            url_template: OSM_URL_TEMPLATE.to_string(),
            subdomains: Vec::new(),
            attribution: OSM_ATTRIBUTION.to_string(),
        }
    }

    /// Builds the [`XyzTemplate`] this configuration describes.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidTemplate`] if the template is missing a
    /// placeholder, if it uses `{s}` while `subdomains` is empty (every tile
    /// fetch would fail at expansion — better to refuse the config up front
    /// than to install a basemap that can never load), or if `subdomains` and
    /// the template's `{s}` disagree.
    pub fn template(&self) -> Result<XyzTemplate, RenderError> {
        let template = XyzTemplate::new(self.url_template.clone())?;
        if self.subdomains.is_empty() {
            if self.url_template.contains("{s}") {
                return Err(RenderError::InvalidTemplate(
                    "the template uses {s} but no subdomains are configured; \
                     use a concrete host instead"
                        .to_string(),
                ));
            }
            Ok(template)
        } else {
            template.with_subdomains(self.subdomains.clone())
        }
    }
}

impl Default for BasemapConfig {
    fn default() -> Self {
        Self::openstreetmap()
    }
}

// `ProjectBasemap` is the serde twin of this struct in `oxigis-core` (the
// project file format must not depend on UI types); the two convert
// field-for-field so a `.oxigis.json` can restore the whole presentation.

impl From<&BasemapConfig> for oxigis_core::ProjectBasemap {
    fn from(config: &BasemapConfig) -> Self {
        Self {
            url_template: config.url_template.clone(),
            subdomains: config.subdomains.clone(),
            attribution: config.attribution.clone(),
        }
    }
}

impl From<&oxigis_core::ProjectBasemap> for BasemapConfig {
    fn from(saved: &oxigis_core::ProjectBasemap) -> Self {
        Self {
            url_template: saved.url_template.clone(),
            subdomains: saved.subdomains.clone(),
            attribution: saved.attribution.clone(),
        }
    }
}

/// Why one tile request failed, and whether trying again could ever help.
///
/// The distinction is the transport's to make, because only it knows the HTTP
/// status or the IO error; the shared retry policy above acts on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileError {
    /// Human-readable cause, logged once per failed tile.
    message: String,
    /// Whether a later attempt might succeed (transport error, HTTP 5xx).
    retryable: bool,
}

impl TileError {
    /// A failure that no number of retries can fix (HTTP 4xx, bad URL, corrupt
    /// body).
    #[must_use]
    pub fn permanent(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
        }
    }

    /// A failure that may succeed later (connection reset, timeout, HTTP 5xx).
    #[must_use]
    pub fn transient(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
        }
    }

    /// The human-readable cause.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Whether a later attempt might succeed.
    #[must_use]
    pub fn retryable(&self) -> bool {
        self.retryable
    }
}

impl core::fmt::Display for TileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// One tile's retry bookkeeping: attempts spent, and (for a retryable
/// failure) the earliest wall-clock time another may be attempted.
///
/// Replaces a bare attempt count so a retry is gated on elapsed time rather
/// than on frame count — see the [Retry policy](self#retry-policy) docs.
/// `retry_at` is read against [`egui::Context`]'s own clock
/// (`ctx.input(|i| i.time)`), which is monotonic and available on every
/// target this crate compiles for, so no `std::time::Instant` — unusable on
/// `wasm32-unknown-unknown` — is needed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FailureState {
    attempts: u32,
    retry_at: f64,
}

impl FailureState {
    /// A never-retried failure recorded directly, with no [`TileError`] in
    /// hand — a planning or tessellation failure is a property of the input,
    /// not of one fetch attempt, so there is no backoff to compute.
    pub(crate) fn permanent(now: f64) -> Self {
        Self {
            attempts: MAX_ATTEMPTS,
            retry_at: now,
        }
    }

    /// The state after `error`, given the tile's previous state (if any), at
    /// wall-clock time `now`.
    ///
    /// A retryable failure backs off exponentially from
    /// [`RETRY_BASE_DELAY_SECS`], doubling per attempt and clamped to
    /// [`RETRY_MAX_DELAY_SECS`]. A non-retryable failure spends the whole
    /// budget at once (see [`FailureState::permanent`]).
    pub(crate) fn after_failure(previous: Option<Self>, now: f64, error: &TileError) -> Self {
        if !error.retryable() {
            return Self::permanent(now);
        }
        let spent = previous.map_or(0, |state| state.attempts);
        let attempts = spent.saturating_add(1);
        // `f64::from(u32)` is exact (f64 has 52 mantissa bits), unlike an
        // `as` cast, so the doubling count never needs a truncation lint.
        let doublings = f64::from(attempts.saturating_sub(1).min(16));
        let delay = (RETRY_BASE_DELAY_SECS * 2.0_f64.powf(doublings)).min(RETRY_MAX_DELAY_SECS);
        Self {
            attempts,
            retry_at: now + delay,
        }
    }

    /// Attempts spent so far, for logging.
    pub(crate) fn attempts(&self) -> u32 {
        self.attempts
    }

    /// Whether the tile's retry budget for this session is spent. Matches
    /// [`MAX_ATTEMPTS`] — see the [module docs][self] for why the cap exists.
    pub(crate) fn exhausted(&self) -> bool {
        self.attempts >= MAX_ATTEMPTS
    }

    /// Whether a retry may be attempted at wall-clock time `now`.
    pub(crate) fn retry_ready(&self, now: f64) -> bool {
        !self.exhausted() && now >= self.retry_at
    }
}

/// Truncates `message` to at most [`MAX_STORED_ERROR_CHARS`] characters, on a
/// `char` boundary, marking a real cut with an ellipsis.
pub(crate) fn truncate_for_display(message: &str) -> String {
    if message.chars().count() <= MAX_STORED_ERROR_CHARS {
        return message.to_string();
    }
    let mut truncated: String = message.chars().take(MAX_STORED_ERROR_CHARS).collect();
    truncated.push('\u{2026}');
    truncated
}

/// A provider's fetch-failure summary — the per-tile-address counterpart to
/// [`TileProviderStats::failed`], for a status line or a test.
///
/// Kept as its own type rather than added to [`TileProviderStats`]: the
/// archive tile provider builds that struct with a field literal outside
/// this module, so widening it would be a breaking change for a sibling this
/// crate does not own end to end. A new type is not.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TileHealth {
    /// Tile addresses currently remembered as failed (retrying or
    /// exhausted). Same count as [`TileProviderStats::failed`], and shrinks
    /// as the failure LRU evicts entries.
    pub failed: usize,
    /// Total fetch failures recorded this session. Unlike `failed` this is
    /// monotonically increasing — it never shrinks when the LRU evicts an
    /// entry — so it is what a "N tiles failed" status line should count.
    pub total_failures: u64,
    /// The most recent failure's message, truncated to
    /// [`MAX_STORED_ERROR_CHARS`]. [`None`] once nothing has failed yet.
    ///
    /// Not cleared by a success or by a manual retry — a status line
    /// reporting "N failures this session, most recently: …" should still
    /// have something to say after the map recovers.
    pub last_error: Option<String>,
}

/// Turns "given a URL, eventually produce bytes" into a platform capability.
///
/// The call must **not** block: it is issued from egui's `wgpu` prepare hook on
/// the render thread. Hand the work to a thread pool (native) or to the
/// microtask queue (browser) and return immediately; the result is reported by
/// calling [`TileSink::deliver`] exactly once, from whichever context finished
/// the transfer.
///
/// `Send + Sync` is required because the provider holding the transport is
/// stored in `egui_wgpu`'s callback resources, which is a concurrent type map
/// on every target (see [`crate::map_gpu::BoxedTileProvider`]).
pub trait TileTransport: Send + Sync + 'static {
    /// Starts fetching `url` for `tile`, reporting the outcome through `sink`.
    fn request(&self, tile: TileId, url: String, sink: TileSink);
}

/// The provider's shared, interior-mutable state.
#[derive(Debug)]
struct TileStore {
    /// Decoded tiles waiting to be handed to the renderer, LRU-bounded.
    ///
    /// `Arc`-wrapped so a ready-cache hit in [`XyzTileProvider::tile`] only
    /// bumps a refcount while `store` is locked; the ~256 KiB pixel clone the
    /// trait's return type requires happens after the lock is released,
    /// where it can no longer stall a worker thread's `deliver_bytes`.
    ready: TileCache<Arc<DecodedTile>>,
    /// Tiles a transport is currently working on.
    inflight: HashSet<TileId>,
    /// Retry state per failed tile, LRU-bounded. See [`FailureState`].
    failures: TileCache<FailureState>,
    /// Total fetch failures recorded this session; see
    /// [`TileHealth::total_failures`].
    total_failures: u64,
    /// The most recent failure's message; see [`TileHealth::last_error`].
    last_error: Option<String>,
}

/// Shared, thread-safe half of the provider: state plus the repaint handle.
#[derive(Debug)]
struct SinkInner {
    /// Ready tiles, in-flight set and failure table.
    store: Mutex<TileStore>,
    /// Context to wake when a tile lands, so the map fills in without the user
    /// having to move the mouse.
    ctx: egui::Context,
}

/// What a [`TileSink`] hands the transport's bytes to.
///
/// The transport seam only ever moves bytes, so *what those bytes are* is the
/// receiver's business: [`SinkInner`] decodes them as a raster image, while
/// [`crate::vector_provider::VectorTileProvider`] gunzips, decodes MVT and
/// tessellates. Keeping the distinction behind this trait is what lets a single
/// [`TileTransport`] implementation per platform serve both layer types.
///
/// `Send + Sync` for the same reason as [`TileTransport`]: the sink travels to a
/// worker thread or into a `spawn_local` future.
pub(crate) trait TileDelivery: Send + Sync + 'static {
    /// Consumes the outcome of one tile request.
    fn deliver_bytes(&self, tile: TileId, result: Result<Vec<u8>, TileError>);

    /// The source has no tile at this address — a FINAL answer, not a
    /// failure (a tile archive is legitimately sparse: an empty ocean tile
    /// is normal). The default records it as a permanent, never-retried
    /// miss; a receiver that can do better (the vector sink inserts an
    /// EMPTY tile so nothing warns and nothing burns the failure LRU)
    /// overrides it.
    fn deliver_absent(&self, tile: TileId) {
        self.deliver_bytes(
            tile,
            Err(TileError::permanent(
                "the source has no tile at this address",
            )),
        );
    }
}

/// The handle a [`TileTransport`] reports results through.
///
/// Cheap to clone (one [`Arc`] bump) and `Send + Sync`, so it can travel to a
/// worker thread or into a `spawn_local` future. Decoding happens here rather
/// than in the transport, which keeps every platform on the same
/// [`oxigis_render::decode_tile`] path (and, for vector layers, on the same
/// [`oxigis_render::decode_mvt`] path).
#[derive(Clone)]
pub struct TileSink {
    /// Whoever consumes the bytes: a raster provider or a vector one.
    inner: Arc<dyn TileDelivery>,
}

impl core::fmt::Debug for TileSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TileSink").finish_non_exhaustive()
    }
}

impl TileSink {
    /// Wraps a receiver of tile bytes.
    pub(crate) fn from_delivery(inner: Arc<dyn TileDelivery>) -> Self {
        Self { inner }
    }

    /// Reports the outcome of the request for `tile`.
    ///
    /// Call this exactly once per [`TileTransport::request`], on success *or*
    /// failure — the tile stays marked in flight (and is therefore never
    /// re-requested) until it arrives.
    pub fn deliver(&self, tile: TileId, result: Result<Vec<u8>, TileError>) {
        self.inner.deliver_bytes(tile, result);
    }

    /// Reports that the source holds no tile at this address — final,
    /// cached, never retried, and (for a receiver that overrides it) never
    /// logged: a sparse archive's misses are its normal shape.
    pub fn deliver_absent(&self, tile: TileId) {
        self.inner.deliver_absent(tile);
    }
}

impl TileDelivery for SinkInner {
    fn deliver_bytes(&self, tile: TileId, result: Result<Vec<u8>, TileError>) {
        let decoded = match result {
            Ok(bytes) => decode_tile(&bytes)
                .map_err(|error| TileError::permanent(format!("decode failed: {error}"))),
            Err(error) => Err(error),
        };
        // Read before locking `store`: egui's own lock (inside `input`) must
        // never nest inside this one, or a worker thread here and the render
        // thread in `XyzTileProvider::tile` could each hold one lock while
        // waiting on the other.
        let now = self.ctx.input(|input| input.time);
        {
            let mut store = self.store.lock();
            store.inflight.remove(&tile);
            match decoded {
                Ok(decoded) => {
                    store.failures.remove(&tile);
                    store.ready.insert(tile, Arc::new(decoded));
                }
                Err(error) => {
                    let previous = store.failures.get(&tile).copied();
                    let state = FailureState::after_failure(previous, now, &error);
                    store.failures.insert(tile, state);
                    store.total_failures = store.total_failures.saturating_add(1);
                    store.last_error = Some(truncate_for_display(error.message()));
                    tracing::warn!(
                        z = tile.z,
                        x = tile.x,
                        y = tile.y,
                        attempts = state.attempts(),
                        retryable = error.retryable(),
                        "oxigis-ui: tile fetch failed: {error}",
                    );
                }
            }
        }
        // Outside the lock: `request_repaint` reaches into egui's own state.
        self.ctx.request_repaint();
    }
}

/// A [`crate::map_gpu::TileProvider`] backed by a real XYZ tile service.
///
/// Construct one per map, hand it to
/// [`crate::OxigisApp::attach_gpu_map_with`], and the map draws real basemap
/// imagery. See the [module docs][self] for the frame protocol, the resource
/// bounds and the retry policy.
pub struct XyzTileProvider {
    /// Expands a [`TileId`] into the URL to fetch.
    template: XyzTemplate,
    /// Shared state plus the repaint handle, also handed to the transport
    /// (wrapped in a [`TileSink`]).
    inner: Arc<SinkInner>,
    /// The platform's fetch capability.
    transport: Box<dyn TileTransport>,
}

impl core::fmt::Debug for XyzTileProvider {
    /// The transport is an opaque platform capability, so it is elided rather
    /// than forcing every implementation to be [`Debug`].
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("XyzTileProvider")
            .field("template", &self.template.template())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl XyzTileProvider {
    /// Builds a provider for `config`, waking `ctx` whenever a tile lands.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidTemplate`] if `config`'s URL template is
    /// not a usable `{z}/{x}/{y}` template, or [`RenderError::InvalidCapacity`]
    /// if the compile-time cache bounds are degenerate (unreachable with the
    /// constants in this module).
    pub fn new(
        config: &BasemapConfig,
        ctx: &egui::Context,
        transport: Box<dyn TileTransport>,
    ) -> Result<Self, RenderError> {
        let template = config.template()?;
        let store = TileStore {
            ready: TileCache::new(READY_CACHE_TILES)?,
            inflight: HashSet::new(),
            failures: TileCache::new(FAILURE_MEMORY_TILES)?,
            total_failures: 0,
            last_error: None,
        };
        Ok(Self {
            template,
            inner: Arc::new(SinkInner {
                store: Mutex::new(store),
                ctx: ctx.clone(),
            }),
            transport,
        })
    }

    /// The URL template tiles are fetched from.
    #[must_use]
    pub fn template(&self) -> &XyzTemplate {
        &self.template
    }

    /// A clone of the sink, for a transport that wants to report out of band.
    #[must_use]
    pub fn sink(&self) -> TileSink {
        TileSink::from_delivery(Arc::clone(&self.inner) as Arc<dyn TileDelivery>)
    }

    /// Snapshot of what the provider is holding: ready tiles, in-flight
    /// requests and remembered failures.
    #[must_use]
    pub fn stats(&self) -> TileProviderStats {
        let store = self.inner.store.lock();
        TileProviderStats {
            ready: store.ready.len(),
            inflight: store.inflight.len(),
            failed: store.failures.len(),
        }
    }

    /// Snapshot of the provider's fetch-failure history — see [`TileHealth`].
    #[must_use]
    pub fn health(&self) -> TileHealth {
        let store = self.inner.store.lock();
        TileHealth {
            failed: store.failures.len(),
            total_failures: store.total_failures,
            last_error: store.last_error.clone(),
        }
    }

    /// Clears every remembered failure, so a tile that gave up for this
    /// session gets a fresh attempt budget immediately — the "reload tiles"
    /// command a Retry control can call once whatever broke the fetches (a
    /// VPN, a typo'd URL, a since-resolved outage) is fixed.
    ///
    /// Does not touch [`TileHealth::total_failures`] or `last_error`: those
    /// are the session's history, and a manual retry does not erase what
    /// already happened.
    pub fn retry_failed_tiles(&self) {
        self.inner.store.lock().failures.clear();
    }
}

/// What an [`XyzTileProvider`] currently holds — for a status bar or a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TileProviderStats {
    /// Decoded tiles waiting for the renderer to collect them.
    pub ready: usize,
    /// Requests a transport has not reported on yet.
    pub inflight: usize,
    /// Tile addresses with a remembered failure (retryable or abandoned).
    pub failed: usize,
}

/// What [`XyzTileProvider::tile`] decided to do, once the lock is released.
enum Decision {
    /// Nothing to do yet; the answer is [`None`].
    Wait,
    /// The tile was ready. Still an [`Arc`]: cloning the pixels out of it
    /// happens after the lock is dropped (see [`TileStore::ready`]).
    Ready(Arc<DecodedTile>),
    /// Start a fetch.
    Start,
}

impl TileProvider for XyzTileProvider {
    fn tile(&self, tile: TileId) -> Option<DecodedTile> {
        // Read before locking `store`, for the same reason as in
        // `SinkInner::deliver_bytes`.
        let now = self.inner.ctx.input(|input| input.time);
        // One short critical section, then the lock is released *before* the
        // transport is touched, or a ready tile's pixels are cloned out of
        // the `Arc`: `request` may hand the job to a worker that immediately
        // calls `deliver`, which takes this same lock.
        let decision = {
            let mut store = self.inner.store.lock();
            if let Some(decoded) = store.ready.get(&tile) {
                Decision::Ready(Arc::clone(decoded))
            } else if store.inflight.contains(&tile) {
                Decision::Wait
            } else if store
                .failures
                .get(&tile)
                .is_some_and(|state| !state.retry_ready(now))
            {
                // `get`, not `peek`: a tile that is backing off is asked
                // about every frame it stays on screen, and only `get`
                // refreshes its place in the LRU. `peek` would let an
                // actively-relevant failure age out of
                // `FAILURE_MEMORY_TILES` purely from OTHER tiles' traffic,
                // which hands it a fresh attempt budget for free — silently
                // defeating the backoff.
                Decision::Wait
            } else if store.inflight.len() >= MAX_INFLIGHT_TILES {
                Decision::Wait
            } else {
                store.inflight.insert(tile);
                Decision::Start
            }
        };

        match decision {
            Decision::Wait => None,
            // Cloned rather than taken: the renderer only asks for tiles
            // whose texture is *not* resident, so this path runs at most
            // once per texture eviction — cheaper (and far kinder to the
            // tile server) than a second HTTP request when the user pans
            // back. Bounded by `READY_CACHE_TILES`.
            Decision::Ready(decoded) => Some((*decoded).clone()),
            Decision::Start => {
                match self.template.expand(tile) {
                    Ok(url) => self.transport.request(tile, url, self.sink()),
                    Err(error) => self
                        .sink()
                        .deliver(tile, Err(TileError::permanent(error.to_string()))),
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BasemapConfig, MAX_ATTEMPTS, MAX_INFLIGHT_TILES, OSM_ATTRIBUTION, RETRY_BASE_DELAY_SECS,
        TileError, TileSink, TileTransport, XyzTileProvider,
    };
    use crate::map_gpu::TileProvider as _;
    use oxigis_render::TileId;
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// A 1×1 RGBA PNG, hand-assembled so the tests need no image encoder.
    const PIXEL_PNG: [u8; 70] = [
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
        0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x30,
        0x4e, 0x9b, 0xf9, 0x1f, 0x00, 0x04, 0x34, 0x02, 0x32, 0x84, 0x73, 0x18, 0xf0, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ];

    /// What a [`ScriptedTransport`] does with a request.
    #[derive(Clone, Copy)]
    enum Reply {
        /// Answer immediately with [`PIXEL_PNG`].
        Png,
        /// Answer immediately with a retryable failure.
        Transient,
        /// Answer immediately with a permanent failure.
        Permanent,
        /// Never answer, leaving the tile in flight.
        Hang,
    }

    /// Test transport: answers synchronously and records every URL asked for.
    struct ScriptedTransport {
        reply: Reply,
        urls: Arc<Mutex<Vec<String>>>,
    }

    impl ScriptedTransport {
        fn new(reply: Reply) -> (Self, Arc<Mutex<Vec<String>>>) {
            let urls = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    reply,
                    urls: Arc::clone(&urls),
                },
                urls,
            )
        }
    }

    impl TileTransport for ScriptedTransport {
        fn request(&self, tile: TileId, url: String, sink: TileSink) {
            self.urls.lock().push(url);
            match self.reply {
                Reply::Png => sink.deliver(tile, Ok(PIXEL_PNG.to_vec())),
                Reply::Transient => sink.deliver(tile, Err(TileError::transient("boom"))),
                Reply::Permanent => sink.deliver(tile, Err(TileError::permanent("404"))),
                Reply::Hang => {}
            }
        }
    }

    fn tile(z: u8, x: u32, y: u32) -> TileId {
        match TileId::new(z, x, y) {
            Ok(tile) => tile,
            Err(err) => panic!("tile {z}/{x}/{y} must be valid: {err}"),
        }
    }

    fn provider(reply: Reply) -> (XyzTileProvider, Arc<Mutex<Vec<String>>>) {
        let (transport, urls) = ScriptedTransport::new(reply);
        let provider = XyzTileProvider::new(
            &BasemapConfig::openstreetmap(),
            &egui::Context::default(),
            Box::new(transport),
        )
        .expect("the default basemap must build a provider");
        (provider, urls)
    }

    #[test]
    fn default_basemap_is_openstreetmap_with_attribution() {
        let config = BasemapConfig::default();
        assert!(config.url_template.contains("tile.openstreetmap.org"));
        assert_eq!(config.attribution, OSM_ATTRIBUTION);
        assert!(config.template().is_ok());
    }

    #[test]
    fn every_basemap_preset_builds_a_valid_template_with_its_credit() {
        for preset in super::BASEMAP_PRESETS {
            let config = preset.config();
            assert!(
                config.template().is_ok(),
                "preset '{}' must have a usable URL template",
                preset.name
            );
            assert!(
                !config.attribution.is_empty(),
                "preset '{}' must ship the credit line its terms require",
                preset.name
            );
            assert!(!preset.name.is_empty());
            assert!(!preset.terms.is_empty());
        }
    }

    #[test]
    fn the_first_basemap_preset_is_the_default_basemap() {
        let first = super::BASEMAP_PRESETS
            .first()
            .expect("the preset list must not be empty");
        assert_eq!(first.config(), BasemapConfig::default());
    }

    #[test]
    fn basemap_preset_names_are_unique() {
        let mut names: Vec<&str> = super::BASEMAP_PRESETS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            super::BASEMAP_PRESETS.len(),
            "duplicate preset names would make the picker ambiguous"
        );
    }

    #[test]
    fn basemap_config_rejects_a_template_without_placeholders() {
        let config = BasemapConfig {
            url_template: "https://example.test/tiles.png".to_string(),
            subdomains: Vec::new(),
            attribution: String::new(),
        };
        assert!(config.template().is_err());
    }

    #[test]
    fn basemap_config_rejects_a_subdomain_template_without_subdomains() {
        // `{s}` with nothing to rotate through would pass construction but
        // fail the expansion of every single tile — refuse it up front.
        let config = BasemapConfig {
            url_template: "https://{s}.example.test/{z}/{x}/{y}.png".to_string(),
            subdomains: Vec::new(),
            attribution: String::new(),
        };
        assert!(config.template().is_err());
    }

    #[test]
    fn a_preset_matches_its_exact_config_but_not_a_host_credited_twin() {
        let preset = super::BASEMAP_PRESETS
            .iter()
            .find(|preset| preset.name.contains("2024"))
            .expect("the 2024 mosaic preset must exist");
        assert!(preset.matches(&preset.config()));
        // The same URL submitted through the free-text field gets a generic
        // host credit; the picker must NOT report the preset as active then.
        let mut host_credited = preset.config();
        host_credited.attribution = "© tiles.maps.eox.at".to_string();
        assert!(!preset.matches(&host_credited));
    }

    #[test]
    fn basemap_config_attaches_subdomains() {
        let config = BasemapConfig {
            url_template: "https://{s}.example.test/{z}/{x}/{y}.png".to_string(),
            subdomains: vec!["a".to_string(), "b".to_string()],
            attribution: String::new(),
        };
        let template = config.template().expect("template must build");
        assert_eq!(template.subdomains().len(), 2);
    }

    #[test]
    fn first_call_misses_and_requests_the_expanded_url() {
        let (provider, urls) = provider(Reply::Hang);
        assert!(provider.tile(tile(3, 4, 5)).is_none());
        assert_eq!(
            urls.lock().as_slice(),
            ["https://tile.openstreetmap.org/3/4/5.png".to_string()]
        );
        assert_eq!(provider.stats().inflight, 1);
    }

    #[test]
    fn an_inflight_tile_is_not_requested_twice() {
        let (provider, urls) = provider(Reply::Hang);
        for _ in 0..5 {
            assert!(provider.tile(tile(1, 0, 0)).is_none());
        }
        assert_eq!(urls.lock().len(), 1);
    }

    #[test]
    fn a_delivered_tile_is_returned_on_the_next_call() {
        // The scripted transport answers inside `request`, so the very first
        // call already leaves the pixels in the store.
        let (provider, _urls) = provider(Reply::Png);
        assert!(provider.tile(tile(2, 1, 1)).is_none());
        let decoded = provider
            .tile(tile(2, 1, 1))
            .expect("the delivered tile must be available");
        assert_eq!(decoded.width(), 1);
        assert_eq!(decoded.height(), 1);
        assert_eq!(provider.stats().inflight, 0);
    }

    #[test]
    fn a_ready_tile_is_served_from_the_cache_without_refetching() {
        let (provider, urls) = provider(Reply::Png);
        assert!(provider.tile(tile(2, 1, 1)).is_none());
        for _ in 0..4 {
            assert!(provider.tile(tile(2, 1, 1)).is_some());
        }
        assert_eq!(urls.lock().len(), 1, "a cached tile must not be refetched");
    }

    #[test]
    fn a_ready_tile_is_shared_via_the_cache_not_deep_cloned_per_call() {
        let (provider, _urls) = provider(Reply::Png);
        let target = tile(2, 1, 1);
        assert!(provider.tile(target).is_none());
        let before = {
            let store = provider.inner.store.lock();
            let decoded = store.ready.peek(&target).expect("the tile must be cached");
            Arc::strong_count(decoded)
        };
        // Two more reads must not change how many handles the cache's own
        // entry holds: each call clones the `Arc` (a refcount bump under the
        // lock), turns it into the `DecodedTile` the trait promises AFTER
        // the lock is released, then drops the temporary — it never mutates
        // or replaces what the cache is holding.
        assert!(provider.tile(target).is_some());
        assert!(provider.tile(target).is_some());
        let after = {
            let store = provider.inner.store.lock();
            let decoded = store.ready.peek(&target).expect("the tile must be cached");
            Arc::strong_count(decoded)
        };
        assert_eq!(
            before, after,
            "the cache's own handle count must be stable across ready-tile reads"
        );
    }

    #[test]
    fn a_permanent_failure_is_never_retried() {
        let (provider, urls) = provider(Reply::Permanent);
        for _ in 0..10 {
            assert!(provider.tile(tile(4, 3, 2)).is_none());
        }
        assert_eq!(urls.lock().len(), 1, "404 must not be retried");
        assert_eq!(provider.stats().failed, 1);
    }

    #[test]
    fn a_transient_failure_backs_off_on_the_wall_clock_not_the_frame_count() {
        // Needs its own `Context` (not the `provider()` helper's throwaway
        // one) so the test can advance the clock `FailureState` reads.
        let ctx = egui::Context::default();
        let (transport, urls) = ScriptedTransport::new(Reply::Transient);
        let provider =
            XyzTileProvider::new(&BasemapConfig::openstreetmap(), &ctx, Box::new(transport))
                .expect("the default basemap must build a provider");
        let target = tile(5, 6, 7);

        // The first attempt is immediate.
        assert!(provider.tile(target).is_none());
        assert_eq!(urls.lock().len(), 1);

        // Many more frames at the SAME instant must not retry: a transient
        // failure now backs off on the clock, not the frame count — this is
        // the fix for a one-second hiccup burning the whole attempt budget
        // within three frames (~50 ms at 60 fps).
        for _ in 0..50 {
            assert!(provider.tile(target).is_none());
        }
        assert_eq!(
            urls.lock().len(),
            1,
            "a retry before its backoff delay must not touch the transport"
        );

        // Advancing the clock past the first backoff releases exactly one
        // more attempt.
        ctx.input_mut(|input| input.time = RETRY_BASE_DELAY_SECS);
        assert!(provider.tile(target).is_none());
        assert_eq!(urls.lock().len(), 2);

        // Jumping far into the future never grants more than `MAX_ATTEMPTS`
        // total, however long the wait.
        ctx.input_mut(|input| input.time = 1_000.0);
        for _ in 0..10 {
            let _ = provider.tile(target);
        }
        assert_eq!(urls.lock().len(), MAX_ATTEMPTS as usize);
    }

    #[test]
    fn a_success_clears_the_entry_so_a_later_failure_backs_off_from_scratch() {
        let ctx = egui::Context::default();
        let (transport, _urls) = ScriptedTransport::new(Reply::Hang);
        let provider =
            XyzTileProvider::new(&BasemapConfig::openstreetmap(), &ctx, Box::new(transport))
                .expect("the default basemap must build a provider");
        let target = tile(6, 1, 1);
        let sink = provider.sink();

        // Spend two attempts, then let the tile succeed.
        sink.deliver(target, Err(TileError::transient("boom")));
        ctx.input_mut(|input| input.time = RETRY_BASE_DELAY_SECS);
        sink.deliver(target, Err(TileError::transient("boom again")));
        sink.deliver(target, Ok(PIXEL_PNG.to_vec()));
        assert!(
            provider.inner.store.lock().failures.peek(&target).is_none(),
            "a success must remove the tile's failure entry outright"
        );

        // A later failure of the SAME tile must look exactly like a tile's
        // first-ever failure — one attempt spent, backing off from the base
        // delay — not pick up where the earlier streak left off.
        ctx.input_mut(|input| input.time = 500.0);
        sink.deliver(target, Err(TileError::transient("boom once more")));
        let state = provider
            .inner
            .store
            .lock()
            .failures
            .peek(&target)
            .copied()
            .expect("the new failure must be recorded");
        assert_eq!(state.attempts, 1, "the attempt count must restart at one");
        assert!(
            (state.retry_at - (500.0 + RETRY_BASE_DELAY_SECS)).abs() < 1e-9,
            "the backoff must restart from the base delay, not continue doubling: {state:?}"
        );
    }

    #[test]
    fn concurrency_is_capped() {
        let (provider, urls) = provider(Reply::Hang);
        // Ask for far more tiles than the cap; z=8 has 256 columns.
        for x in 0..(MAX_INFLIGHT_TILES as u32 * 4) {
            assert!(provider.tile(tile(8, x, 0)).is_none());
        }
        assert_eq!(urls.lock().len(), MAX_INFLIGHT_TILES);
        assert_eq!(provider.stats().inflight, MAX_INFLIGHT_TILES);
    }

    #[test]
    fn an_undecodable_body_is_a_permanent_failure() {
        struct GarbageTransport;
        impl TileTransport for GarbageTransport {
            fn request(&self, tile: TileId, _url: String, sink: TileSink) {
                sink.deliver(tile, Ok(vec![0xDE, 0xAD, 0xBE, 0xEF]));
            }
        }
        let provider = XyzTileProvider::new(
            &BasemapConfig::openstreetmap(),
            &egui::Context::default(),
            Box::new(GarbageTransport),
        )
        .expect("provider must build");
        for _ in 0..5 {
            assert!(provider.tile(tile(0, 0, 0)).is_none());
        }
        assert_eq!(provider.stats().failed, 1);
    }

    #[test]
    fn tile_error_reports_its_own_retryability() {
        assert!(TileError::transient("x").retryable());
        assert!(!TileError::permanent("x").retryable());
        assert_eq!(TileError::permanent("nope").message(), "nope");
        assert_eq!(TileError::transient("nope").to_string(), "nope");
    }

    #[test]
    fn health_reports_a_monotonic_total_and_the_latest_message() {
        let (provider, _urls) = provider(Reply::Permanent);
        assert_eq!(provider.health(), super::TileHealth::default());

        assert!(provider.tile(tile(4, 3, 2)).is_none());
        let health = provider.health();
        assert_eq!(health.failed, 1);
        assert_eq!(health.total_failures, 1);
        assert_eq!(health.last_error.as_deref(), Some("404"));

        // A second, distinct tile failing must add to the total without
        // resetting it — unlike `failed`, `total_failures` never shrinks.
        assert!(provider.tile(tile(4, 5, 6)).is_none());
        let health = provider.health();
        assert_eq!(health.failed, 2);
        assert_eq!(health.total_failures, 2);
    }

    #[test]
    fn retry_failed_tiles_clears_the_retry_gate_but_not_the_history() {
        let (provider, urls) = provider(Reply::Permanent);
        let target = tile(4, 3, 2);
        assert!(provider.tile(target).is_none());
        assert_eq!(urls.lock().len(), 1);
        assert_eq!(provider.health().failed, 1);

        provider.retry_failed_tiles();
        assert_eq!(
            provider.health().failed,
            0,
            "the retry gate must be empty right after a manual retry"
        );
        assert_eq!(
            provider.health().total_failures,
            1,
            "the session history must survive a manual retry"
        );

        // The tile immediately gets a fresh attempt, exactly as if it had
        // never failed.
        assert!(provider.tile(target).is_none());
        assert_eq!(urls.lock().len(), 2);
    }

    #[test]
    fn error_messages_longer_than_the_cap_are_truncated_on_a_char_boundary() {
        let long = "e".repeat(super::MAX_STORED_ERROR_CHARS + 50);
        let truncated = super::truncate_for_display(&long);
        assert_eq!(truncated.chars().count(), super::MAX_STORED_ERROR_CHARS + 1);
        assert!(truncated.ends_with('\u{2026}'));

        // A multi-byte character sitting right at the cut point must not
        // split a codepoint in two.
        let unicode = "\u{1F5FA}".repeat(super::MAX_STORED_ERROR_CHARS + 5);
        let truncated = super::truncate_for_display(&unicode);
        assert!(truncated.is_char_boundary(truncated.len()));

        let short = "short and sweet";
        assert_eq!(super::truncate_for_display(short), short);
    }

    #[test]
    fn the_provider_is_boxable_as_a_send_sync_seam() {
        let (provider, _urls) = provider(Reply::Png);
        let boxed: crate::map_gpu::BoxedTileProvider = Box::new(provider);
        assert!(boxed.tile(tile(0, 0, 0)).is_none());
    }
}
