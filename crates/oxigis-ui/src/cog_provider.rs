//! Cloud-Optimized GeoTIFF layers: the shell-agnostic half of the range loop.
//!
//! [`CogTileProvider`] is to a COG what [`crate::XyzTileProvider`] is to a tile
//! service: it implements the synchronous [`crate::map_gpu::TileProvider`] seam
//! on top of an asynchronous platform capability. The capability is different,
//! though — a COG is *one* URL read with HTTP `Range` requests, not one URL per
//! tile — so it gets its own transport trait, [`RangeTransport`].
//!
//! ```text
//! frame N     provider.tile(t) -> None, drives the COG open / plans t's
//!                                 source tiles, asks the transport for ranges
//! (off-frame) transport fetches, RangeSink::deliver feeds bytes back into
//!             oxigis_render::CogOpen or decodes a source tile, and repaints
//! frame N+k   provider.tile(t) -> Some(pixels)
//! ```
//!
//! # Every answer is final
//!
//! `oxigis_render::MapRenderer` asks for a tile exactly once and uploads a
//! texture for whatever it gets back — there is no way to say "here are better
//! pixels for a tile you already have". So this provider never returns a
//! provisional answer. For a given [`TileId`] it returns [`None`] until it knows
//! which of three things is true, and then returns the final pixels:
//!
//! * the tile is covered by the COG → the COG pixels, composited over the
//!   basemap tile when they are not fully opaque;
//! * the tile is outside the COG's extent, or the COG failed → the basemap tile
//!   alone;
//! * there is no basemap → the COG pixels, transparent where uncovered.
//!
//! # Layer composition
//!
//! The renderer draws one texture per tile, so "COG over basemap" is done in
//! pixels here rather than with two draw calls. That keeps the GPU side
//! unchanged and is exact for the alpha-over case; it costs one 256×256 blend
//! per tile, once, on the frame the tile completes.
//!
//! # CRS coverage
//!
//! EPSG:3857, EPSG:4326 and the WGS 84 UTM zones (EPSG:32601–32660 and
//! EPSG:32701–32760, reprojected per pixel) — see [`oxigis_render::CogCrs`]. A
//! COG in any other CRS is reported as unsupported (and the basemap shows
//! through) rather than drawn in the wrong place.
//!
//! # Bounded resources
//!
//! | Resource | Bound | Constant |
//! |---|---|---|
//! | Map tiles being assembled at once | 4 | [`MAX_INFLIGHT_COG_TILES`] |
//! | Composed tiles held for the renderer | 64 (LRU) | [`crate::tile_provider::READY_CACHE_TILES`] |
//! | Remembered failures | 1024 (LRU) | [`crate::tile_provider::FAILURE_MEMORY_TILES`] |
//! | Frames a completed COG tile waits for its basemap tile | 120 | [`MAX_BASE_WAIT_FRAMES`] |

use std::collections::HashMap;
use std::sync::Arc;

use oxigis_render::{
    ByteRange, CogMetadata, CogOpen, CogOpenProgress, CogSourceTile, CogTilePlan, DecodedTile,
    RasterStretch, RenderError, TileCache, TileId,
};
// Not re-exported at the crate root, unlike the whole-tile `decode_cog_tile`
// this provider used to call; `oxigis_render::cog` is public, so the block form
// and its options are reached through the module path rather than by widening a
// crate this change does not own.
use oxigis_render::cog::{CogDecodeOptions, decode_cog_block};
use parking_lot::Mutex;

use crate::map_gpu::{BoxedTileProvider, TileProvider};
use crate::tile_provider::{
    FAILURE_MEMORY_TILES, FailureState, READY_CACHE_TILES, TileError, TileHealth,
    TileProviderStats, truncate_for_display,
};

/// Number of map tiles whose source ranges may be in flight at once.
///
/// Each map tile fans out into up to four source-tile range requests, so four
/// map tiles is up to sixteen concurrent requests — the same order as
/// [`crate::tile_provider::MAX_INFLIGHT_TILES`] for the XYZ path.
pub const MAX_INFLIGHT_COG_TILES: usize = 4;

/// Frames a finished, partly transparent COG tile waits for its basemap tile
/// before being drawn without one.
///
/// Needed because the [`TileProvider`] seam cannot distinguish "the basemap tile
/// is still loading" from "the basemap tile will never arrive": both are
/// [`None`]. Two seconds at 60 fps is long enough for any tile that is coming,
/// and short enough that a 404 basemap does not hide the COG for ever.
pub const MAX_BASE_WAIT_FRAMES: u32 = 120;

/// A COG raster layer: where to read it from and how to display its samples.
#[derive(Debug, Clone, PartialEq)]
pub struct CogLayerConfig {
    /// URL of the `.tif`/`.tiff` file, served with `Range` support.
    pub url: String,
    /// How 16-bit samples are mapped onto the display range.
    pub stretch: RasterStretch,
    /// Credit line for the imagery; empty when none is required.
    pub attribution: String,
}

impl CogLayerConfig {
    /// A layer reading `url` with the default stretch and no attribution.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            stretch: RasterStretch::default(),
            attribution: String::new(),
        }
    }

    /// Sets the credit line.
    #[must_use]
    pub fn with_attribution(mut self, attribution: impl Into<String>) -> Self {
        self.attribution = attribution.into();
        self
    }

    /// Sets the 16-bit sample stretch.
    #[must_use]
    pub fn with_stretch(mut self, stretch: RasterStretch) -> Self {
        self.stretch = stretch;
        self
    }
}

/// What a delivered byte range belongs to.
///
/// The COG variants came first; the `Archive*` variants (tiles v1.3) let a
/// PMTiles reader ride the SAME platform transports — a transport only
/// moves the job through untouched, so neither shell knows the difference. The
/// `ArchiveSurvey`/`ArchivePage` pair (tiles v1.4) does the same for the paged
/// MBTiles reader, whose unit of work is a *page run* rather than a directory.
///
/// # Why `#[non_exhaustive]`
///
/// Every reader this crate grows adds a variant, and a platform shell should
/// never have to be recompiled for one: a transport only moves the job through
/// untouched. Marking it here — in the same change that adds the pair — takes
/// the semver break once rather than at every future addition. A shell that
/// *does* inspect jobs needs a `_ =>` arm, which is what it should have had all
/// along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RangeJob {
    /// Bytes of the file header / IFD chain, starting at this file offset.
    Header {
        /// File offset the delivered bytes begin at.
        start: u64,
    },
    /// One source tile of one map tile.
    SourceTile {
        /// Map tile being assembled.
        tile: TileId,
        /// Index into the plan's source list.
        index: usize,
    },
    /// A tile archive's speculative header prefetch, or a later metadata
    /// read, starting at this file offset.
    ArchiveHeader {
        /// File offset the delivered bytes begin at.
        start: u64,
    },
    /// One leaf directory a tile lookup needs, keyed by its absolute file
    /// offset (the leaf cache's key).
    ArchiveLeaf {
        /// Map tile whose lookup is waiting on the leaf.
        tile: TileId,
        /// Absolute file offset of the leaf blob.
        at: u64,
    },
    /// One archive tile body.
    ArchiveTile {
        /// The map tile the bytes decode into.
        tile: TileId,
    },
    /// A paged archive's speculative opening read, starting at this file
    /// offset.
    ///
    /// Distinct from [`RangeJob::ArchiveHeader`] because it is *not* expressible
    /// as pages: a SQLite archive's page size is one of the things this very
    /// read discovers.
    ArchiveSurvey {
        /// File offset the delivered bytes begin at.
        start: u64,
    },
    /// A run of consecutive SQLite pages a paged lookup needs.
    ///
    /// One job rather than one per page because the case that actually moves
    /// bytes — an overflow chain — is contiguous on 80–100 % of real archives,
    /// and asking for it as a single range is what makes reading a spilled tile
    /// one round trip instead of a dozen.
    ArchivePage {
        /// First page of the run, 1-based.
        first: u32,
        /// How many pages it covers.
        count: u32,
    },
}

/// Turns "given a URL and a byte range, eventually produce those bytes" into a
/// platform capability.
///
/// The call must **not** block: it is issued from egui's `wgpu` prepare hook.
/// Hand the work to a thread pool (native) or to the microtask queue (browser)
/// and return immediately; report the outcome by calling [`RangeSink::deliver`]
/// exactly once with the [`RangeJob`] that was handed in.
///
/// # Implementation requirements
///
/// * Send `Range: bytes=<start>-<end-1>` (see [`ByteRange::header_value`]).
/// * Treat HTTP 206 as success. A 200 with the **whole** file means the server
///   ignored the header: that is a failure, not a short read to paper over.
/// * A response shorter than the requested range is fine — the reader asks for a
///   speculative 64 KiB header block that may run past the end of a small file —
///   but the bytes must start exactly at the requested offset.
///
/// # CORS
///
/// `Range` is **not** a CORS-safelisted request header, so a cross-origin COG
/// makes the browser send a preflight `OPTIONS`. The host must answer with
/// `Access-Control-Allow-Headers: Range` and expose `Content-Range`
/// (`Access-Control-Expose-Headers`). Every COG-hosting bucket configured for
/// browser clients does this already (AWS Open Data, OpenAerialMap, Planetary
/// Computer); one that does not cannot be read from a browser at all.
///
/// `Send + Sync` is required because the provider holding the transport lives in
/// `egui_wgpu`'s concurrent callback-resource map.
pub trait RangeTransport: Send + Sync + 'static {
    /// Starts fetching `range` of `url`, reporting the outcome through `sink`.
    fn request_range(&self, url: String, range: ByteRange, job: RangeJob, sink: RangeSink);
}

/// How far along the COG's own metadata is.
enum Stage {
    /// The header/IFD chain is still being read.
    Opening(Box<CogOpen>),
    /// The file is fully described.
    ///
    /// `Arc`-wrapped so [`CogTileProvider::decide`] can hand a plan-time
    /// borrow of the metadata to [`CogMetadata::plan_tile`] with a refcount
    /// bump, instead of deep-cloning a `Vec<CogLevel>` — whose
    /// `tile_offsets`/`tile_byte_counts` are one `u64` per tile of that level
    /// — on every frame, for every tile, until it resolves.
    Ready(Arc<CogMetadata>),
    /// The file could not be read; the reason is logged once and kept for
    /// [`CogTileProvider::failure`].
    Failed(String),
}

/// One map tile being assembled out of source tiles.
struct TileJob {
    /// Which source tiles it needs.
    plan: CogTilePlan,
    /// Decoded RGBA of each source tile, parallel to `plan.sources`.
    decoded: Vec<Option<Vec<u8>>>,
    /// Range requests not yet reported on.
    outstanding: usize,
}

/// The provider's shared, interior-mutable state.
struct CogStore {
    /// How far along the metadata read is.
    stage: Stage,
    /// The per-file decode settings, replaced by [`CogOpen::decode_options`]
    /// the moment the open completes.
    ///
    /// [`CogMetadata`] has no field for `GDAL_NODATA` (42113) or `JPEGTables`
    /// (347), so a caller that decodes blocks itself — as this provider does —
    /// has to carry them beside the metadata. Until the open finishes it holds
    /// the configured stretch alone, which is all a pre-`Ready` store could
    /// answer with anyway; no tile decodes before then.
    decode_options: CogDecodeOptions,
    /// Whether a header range request is outstanding.
    header_inflight: bool,
    /// Final answers, LRU-bounded. `None` means "outside the COG's extent".
    ready: TileCache<Option<DecodedTile>>,
    /// Map tiles currently being assembled.
    jobs: HashMap<TileId, TileJob>,
    /// Retry state per failed map tile, LRU-bounded. See [`FailureState`].
    failures: TileCache<FailureState>,
    /// Frames a completed tile has waited for its basemap counterpart.
    base_waits: TileCache<u32>,
    /// Total fetch failures recorded this session; see
    /// [`TileHealth::total_failures`].
    total_failures: u64,
    /// The most recent failure's message; see [`TileHealth::last_error`].
    last_error: Option<String>,
}

/// Shared state, transport and repaint handle.
struct CogInner {
    /// URL every range request is issued against.
    url: String,
    /// How 16-bit samples are mapped onto the display range.
    stretch: RasterStretch,
    /// Metadata, caches and in-flight bookkeeping.
    store: Mutex<CogStore>,
    /// Context to wake when a tile completes.
    ctx: egui::Context,
    /// The platform's range-fetch capability.
    transport: Box<dyn RangeTransport>,
}

impl CogInner {
    /// Drives the COG open as far as the supplied bytes allow, issuing the next
    /// header range request if one is needed.
    fn advance_open(self: &Arc<Self>) {
        let mut request = None;
        let mut failure = None;
        {
            let mut store = self.store.lock();
            let mut next_stage = None;
            let mut next_options = None;
            if let Stage::Opening(open) = &mut store.stage {
                let mut opened = None;
                match open.poll() {
                    Ok(CogOpenProgress::Need(range)) => request = Some(range),
                    // Cloned out of the borrow rather than used in place:
                    // `poll` returns a `CogOpenProgress<'_>` borrowing `open`,
                    // and `decode_options()` below needs `open` again.
                    Ok(CogOpenProgress::Ready(metadata)) => opened = Some(metadata.clone()),
                    Err(error) => {
                        let message = error.to_string();
                        failure = Some(message.clone());
                        next_stage = Some(Stage::Failed(message));
                    }
                }
                if let Some(metadata) = opened {
                    // The IFD chain's `GDAL_NODATA` and `JPEGTables`, which the
                    // metadata does not carry, kept for every later block
                    // decode. The stretch is this layer's configured one, not
                    // the default `decode_options()` fills in.
                    next_options = Some(CogDecodeOptions {
                        stretch: self.stretch,
                        ..open.decode_options()
                    });
                    next_stage = Some(Stage::Ready(Arc::new(metadata)));
                }
            }
            if let Some(options) = next_options {
                store.decode_options = options;
            }
            if let Some(stage) = next_stage {
                store.stage = stage;
            }
            store.header_inflight = request.is_some();
        }
        if let Some(message) = failure {
            tracing::warn!("oxigis-ui: COG open failed: {message}");
        }
        if let Some(range) = request {
            self.transport.request_range(
                self.url.clone(),
                range,
                RangeJob::Header { start: range.start },
                RangeSink::from_delivery(Arc::clone(self) as Arc<dyn RangeDelivery>),
            );
        }
    }

    /// Records a failed map tile, so it is retried on a wall-clock backoff
    /// rather than for ever.
    fn record_failure(&self, tile: TileId, error: &TileError) {
        // Read before locking `store`, for the same reason as
        // `crate::tile_provider::SinkInner::deliver_bytes`.
        let now = self.ctx.input(|input| input.time);
        let mut store = self.store.lock();
        store.jobs.remove(&tile);
        let previous = store.failures.get(&tile).copied();
        let state = FailureState::after_failure(previous, now, error);
        store.failures.insert(tile, state);
        store.total_failures = store.total_failures.saturating_add(1);
        store.last_error = Some(truncate_for_display(error.message()));
        tracing::warn!(
            z = tile.z,
            x = tile.x,
            y = tile.y,
            attempts = state.attempts(),
            "oxigis-ui: COG tile failed: {error}"
        );
    }
}

/// What a [`RangeSink`] hands the transport's bytes to.
///
/// The exact generalisation [`crate::tile_provider::TileSink`] received when
/// vector tiles arrived: the transport seam only moves bytes plus an opaque
/// [`RangeJob`], so *whose* bytes they are is the receiver's business —
/// [`CogInner`] decodes TIFF strips, the archive reader (tiles v1.3) drives
/// a PMTiles directory walk. One transport per platform serves both.
pub(crate) trait RangeDelivery: Send + Sync + 'static {
    /// Consumes the outcome of one range request.
    fn deliver_range(self: Arc<Self>, job: RangeJob, result: Result<Vec<u8>, TileError>);
}

/// The handle a [`RangeTransport`] reports results through.
///
/// Cheap to clone (one [`Arc`] bump) and `Send + Sync`, so it can travel to a
/// worker thread or into a `spawn_local` future.
#[derive(Clone)]
pub struct RangeSink {
    /// Whoever consumes the bytes: the COG reader or an archive reader.
    inner: Arc<dyn RangeDelivery>,
}

impl core::fmt::Debug for RangeSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RangeSink").finish_non_exhaustive()
    }
}

impl RangeSink {
    /// Wraps a receiver of ranged bytes.
    pub(crate) fn from_delivery(inner: Arc<dyn RangeDelivery>) -> Self {
        Self { inner }
    }

    /// Reports the outcome of one range request.
    ///
    /// Call exactly once per [`RangeTransport::request_range`], with the same
    /// [`RangeJob`], on success *or* failure.
    pub fn deliver(&self, job: RangeJob, result: Result<Vec<u8>, TileError>) {
        Arc::clone(&self.inner).deliver_range(job, result);
    }
}

impl RangeDelivery for CogInner {
    fn deliver_range(self: Arc<Self>, job: RangeJob, result: Result<Vec<u8>, TileError>) {
        match job {
            RangeJob::Header { start } => self.deliver_header(start, result),
            RangeJob::SourceTile { tile, index } => self.deliver_source_tile(tile, index, result),
            // The archive variants belong to the archive readers; a transport
            // cannot mix sinks up (the job travels WITH its sink), so this
            // is unreachable short of a broken transport. Logged and dropped
            // rather than silently swallowed.
            RangeJob::ArchiveHeader { .. }
            | RangeJob::ArchiveLeaf { .. }
            | RangeJob::ArchiveTile { .. }
            | RangeJob::ArchiveSurvey { .. }
            | RangeJob::ArchivePage { .. } => {
                tracing::debug!("oxigis-ui: an archive range job reached the COG sink; dropped");
            }
        }
        self.ctx.request_repaint();
    }
}

impl CogInner {
    /// Feeds header bytes into the open state machine.
    fn deliver_header(self: &Arc<Self>, start: u64, result: Result<Vec<u8>, TileError>) {
        match result {
            Ok(bytes) => {
                {
                    let mut store = self.store.lock();
                    store.header_inflight = false;
                    if let Stage::Opening(open) = &mut store.stage {
                        open.supply(start, bytes);
                    }
                }
                self.advance_open();
            }
            Err(error) => {
                let message = format!("COG header fetch failed: {error}");
                {
                    let mut store = self.store.lock();
                    store.header_inflight = false;
                    store.stage = Stage::Failed(message.clone());
                }
                tracing::warn!("oxigis-ui: {message}");
            }
        }
    }

    /// Decodes one source tile and, when the last one lands, composes the map
    /// tile.
    fn deliver_source_tile(&self, tile: TileId, index: usize, result: Result<Vec<u8>, TileError>) {
        let payload = match result {
            Ok(payload) => payload,
            Err(error) => {
                self.record_failure(tile, &error);
                return;
            }
        };

        let composed = {
            let mut store = self.store.lock();
            let CogStore {
                stage,
                jobs,
                decode_options,
                ..
            } = &mut *store;
            let Stage::Ready(metadata) = stage else {
                jobs.remove(&tile);
                return;
            };
            let Some(job) = jobs.get_mut(&tile) else {
                // The job was dropped (a sibling range failed); nothing to do.
                return;
            };
            let level = match metadata.level(job.plan.level) {
                Some(level) => level,
                None => {
                    jobs.remove(&tile);
                    return;
                }
            };
            // The block's row in the level's tile grid, which is what makes a
            // STRIPED TIFF's last strip decodable: a strip holds only the rows
            // that are left, so decoding it as a full block failed the whole
            // map tile. `index` indexes `plan.sources` by construction (see
            // `TileJob::decoded`), and a miss means a corrupt job — dropping it
            // is right, where decoding as row 0 would re-introduce exactly the
            // mis-slice this fixes.
            let Some(tile_y) = job.plan.sources.get(index).map(|source| source.tile_y) else {
                jobs.remove(&tile);
                return;
            };
            let decoded = decode_cog_block(
                level,
                tile_y,
                metadata.little_endian,
                &payload,
                decode_options,
            );
            match decoded {
                Ok(rgba) => {
                    if let Some(slot) = job.decoded.get_mut(index) {
                        *slot = Some(rgba);
                    }
                    job.outstanding = job.outstanding.saturating_sub(1);
                    if job.outstanding > 0 {
                        return;
                    }
                    let sources = collect_sources(job);
                    Some(metadata.compose_tile(&job.plan, &sources))
                }
                Err(error) => Some(Err(error)),
            }
        };

        match composed {
            Some(Ok(decoded)) => {
                let mut store = self.store.lock();
                store.jobs.remove(&tile);
                store.failures.remove(&tile);
                store.ready.insert(tile, Some(decoded));
            }
            Some(Err(error)) => {
                self.record_failure(tile, &classify_render_error(&error));
            }
            None => {}
        }
    }
}

/// Gathers a job's decoded source tiles for composition.
fn collect_sources(job: &TileJob) -> Vec<CogSourceTile> {
    job.plan
        .sources
        .iter()
        .zip(job.decoded.iter())
        .filter_map(|(reference, rgba)| {
            rgba.as_ref().map(|rgba| CogSourceTile {
                tile_x: reference.tile_x,
                tile_y: reference.tile_y,
                rgba: rgba.clone(),
            })
        })
        .collect()
}

/// Classifies a renderer error for the shared retry policy.
///
/// Nothing the parser or the codec reports gets better on a retry, so every
/// [`RenderError`] is permanent; only transport failures are retryable, and those
/// are classified by the transport itself.
fn classify_render_error(error: &RenderError) -> TileError {
    TileError::permanent(error.to_string())
}

/// Alpha-composites `over` on top of `under`, both RGBA8.
///
/// `under` is sampled with nearest neighbour when its size differs from
/// `over`'s, so a 512 px basemap tile still works under a 256 px COG tile — and
/// under a 512 px tile-archive one, which is why `crate::archive`'s raster
/// provider composites through this same function rather than a second copy of
/// the arithmetic.
///
/// # Errors
///
/// Propagates [`RenderError::InvalidTileImage`] from [`DecodedTile::new`] for a
/// degenerate `over`.
pub(crate) fn blend_over(
    over: &DecodedTile,
    under: &DecodedTile,
) -> Result<DecodedTile, RenderError> {
    let width = over.width();
    let height = over.height();
    let mut out = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for y in 0..height {
        for x in 0..width {
            let index = ((y as usize) * (width as usize) + x as usize) * 4;
            let top = over.rgba().get(index..index + 4).unwrap_or(&[0, 0, 0, 0]);
            let alpha = f32::from(top[3]) / 255.0;
            if alpha >= 1.0 {
                out.extend_from_slice(top);
                continue;
            }
            let source_x = (x as u64 * u64::from(under.width()) / u64::from(width.max(1))) as u32;
            let source_y = (y as u64 * u64::from(under.height()) / u64::from(height.max(1))) as u32;
            let base_index =
                ((source_y as usize) * (under.width() as usize) + source_x as usize) * 4;
            let bottom = under
                .rgba()
                .get(base_index..base_index + 4)
                .unwrap_or(&[0, 0, 0, 255]);
            for channel in 0..3 {
                let mixed =
                    f32::from(top[channel]) * alpha + f32::from(bottom[channel]) * (1.0 - alpha);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a convex combination of two bytes is in 0..=255"
                )]
                let byte = mixed.round().clamp(0.0, 255.0) as u8;
                out.push(byte);
            }
            out.push(top[3].max(bottom[3]));
        }
    }
    DecodedTile::new(width, height, out)
}

/// What [`CogTileProvider::tile`] decided to return.
enum Outcome {
    /// Nothing final yet.
    Wait,
    /// The COG does not cover this tile (or could not be read).
    BaseOnly,
    /// The COG's pixels for this tile.
    Cog(DecodedTile),
}

/// A [`TileProvider`] that draws a COG, optionally over another provider.
///
/// Construct one per COG layer and install it with
/// [`crate::map_gpu::replace_provider`]. See the [module docs][self] for the
/// protocol, the resource bounds and the CRS assumption.
pub struct CogTileProvider {
    /// Shared state, transport and repaint handle.
    inner: Arc<CogInner>,
    /// The provider drawn underneath, usually an [`crate::XyzTileProvider`].
    base: Option<BoxedTileProvider>,
}

impl core::fmt::Debug for CogTileProvider {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CogTileProvider")
            .field("url", &self.inner.url)
            .field("has_base", &self.base.is_some())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl CogTileProvider {
    /// Builds a provider for `config`, waking `ctx` whenever a tile completes.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if the compile-time cache bounds
    /// are degenerate (unreachable with the constants in this crate).
    pub fn new(
        config: &CogLayerConfig,
        ctx: &egui::Context,
        transport: Box<dyn RangeTransport>,
    ) -> Result<Self, RenderError> {
        let store = CogStore {
            stage: Stage::Opening(Box::new(CogOpen::new())),
            decode_options: CogDecodeOptions::with_stretch(config.stretch),
            header_inflight: false,
            ready: TileCache::new(READY_CACHE_TILES)?,
            jobs: HashMap::new(),
            failures: TileCache::new(FAILURE_MEMORY_TILES)?,
            base_waits: TileCache::new(READY_CACHE_TILES)?,
            total_failures: 0,
            last_error: None,
        };
        Ok(Self {
            inner: Arc::new(CogInner {
                url: config.url.clone(),
                stretch: config.stretch,
                store: Mutex::new(store),
                ctx: ctx.clone(),
                transport,
            }),
            base: None,
        })
    }

    /// Draws `base` underneath the COG.
    #[must_use]
    pub fn with_base(mut self, base: BoxedTileProvider) -> Self {
        self.base = Some(base);
        self
    }

    /// The URL the COG is read from.
    #[must_use]
    pub fn url(&self) -> &str {
        &self.inner.url
    }

    /// A clone of the sink, for a transport that reports out of band.
    #[must_use]
    pub fn sink(&self) -> RangeSink {
        RangeSink::from_delivery(Arc::clone(&self.inner) as Arc<dyn RangeDelivery>)
    }

    /// The COG's metadata, once the header has been read.
    #[must_use]
    pub fn metadata(&self) -> Option<CogMetadata> {
        match &self.inner.store.lock().stage {
            Stage::Ready(metadata) => Some((**metadata).clone()),
            _ => None,
        }
    }

    /// Why the COG could not be read, if it could not be.
    #[must_use]
    pub fn failure(&self) -> Option<String> {
        match &self.inner.store.lock().stage {
            Stage::Failed(message) => Some(message.clone()),
            _ => None,
        }
    }

    /// Snapshot of what the provider is holding.
    #[must_use]
    pub fn stats(&self) -> TileProviderStats {
        let store = self.inner.store.lock();
        TileProviderStats {
            ready: store.ready.len(),
            inflight: store.jobs.len(),
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

    /// Clears every remembered failure, so a map tile that gave up for this
    /// session gets a fresh attempt budget immediately. See
    /// [`crate::XyzTileProvider::retry_failed_tiles`] for the full rationale;
    /// this is its COG twin.
    pub fn retry_failed_tiles(&self) {
        self.inner.store.lock().failures.clear();
    }

    /// Decides what `tile` should resolve to, queueing any work it needs.
    ///
    /// `now` is `ctx.input(|i| i.time)`, read by the caller *before* the
    /// store is locked (see `SinkInner::deliver_bytes` in `tile_provider.rs`
    /// for why egui's own lock must never nest inside it).
    ///
    /// Returns the outcome plus the range requests to issue *after* the lock is
    /// released — a transport may answer synchronously, which would deadlock.
    fn decide(&self, tile: TileId, now: f64) -> (Outcome, Vec<(ByteRange, RangeJob)>) {
        let mut requests = Vec::new();
        let mut store = self.inner.store.lock();

        // Checked before the metadata match below (finding #75): once a
        // tile's answer is cached, `stage` can only be `Ready` — nothing
        // ever inserts into `ready` from any other stage, and `stage` never
        // leaves `Ready` once entered — so a cache hit needs no metadata
        // access at all, cloned or not.
        if let Some(entry) = store.ready.get(&tile) {
            return match entry {
                Some(decoded) => (Outcome::Cog(decoded.clone()), requests),
                None => (Outcome::BaseOnly, requests),
            };
        }

        let metadata = match &store.stage {
            Stage::Failed(_) => return (Outcome::BaseOnly, requests),
            Stage::Opening(_) => return (Outcome::Wait, requests),
            Stage::Ready(metadata) => Arc::clone(metadata),
        };

        if store.jobs.contains_key(&tile) {
            return (Outcome::Wait, requests);
        }
        // `get`, not `peek`: see the identical comment in
        // `XyzTileProvider::tile` (`tile_provider.rs`) — an on-screen tile
        // that is backing off must not age out of `FAILURE_MEMORY_TILES` for
        // free, or eviction becomes a way around the backoff.
        if let Some(state) = store.failures.get(&tile).copied() {
            if state.exhausted() {
                return (Outcome::BaseOnly, requests);
            }
            if !state.retry_ready(now) {
                // Not yet due: distinct from `BaseOnly` (which the module
                // docs promise is *final*) — the COG may still answer once
                // the backoff elapses, so `Wait` and try again next frame.
                return (Outcome::Wait, requests);
            }
        }
        if store.jobs.len() >= MAX_INFLIGHT_COG_TILES {
            return (Outcome::Wait, requests);
        }

        let plan = match metadata.plan_tile(tile) {
            Ok(Some(plan)) => plan,
            Ok(None) => {
                store.ready.insert(tile, None);
                return (Outcome::BaseOnly, requests);
            }
            Err(error) => {
                // Planning failures are properties of the file (unsupported CRS,
                // too many source tiles), not of this attempt.
                store.failures.insert(tile, FailureState::permanent(now));
                store.total_failures = store.total_failures.saturating_add(1);
                store.last_error = Some(truncate_for_display(&error.to_string()));
                tracing::warn!(
                    z = tile.z,
                    x = tile.x,
                    y = tile.y,
                    "oxigis-ui: COG tile cannot be planned: {error}"
                );
                return (Outcome::BaseOnly, requests);
            }
        };

        for (index, reference) in plan.sources.iter().enumerate() {
            if let Some(range) = reference.range {
                requests.push((range, RangeJob::SourceTile { tile, index }));
            }
        }
        if requests.is_empty() {
            // Every source tile is sparse: the answer is a transparent tile, so
            // the basemap alone is the right thing to draw.
            store.ready.insert(tile, None);
            return (Outcome::BaseOnly, requests);
        }
        store.jobs.insert(
            tile,
            TileJob {
                decoded: vec![None; plan.sources.len()],
                outstanding: requests.len(),
                plan,
            },
        );
        (Outcome::Wait, requests)
    }

    /// Counts a frame spent waiting for the basemap tile under a finished COG
    /// tile, returning whether the wait has run out.
    fn base_wait_expired(&self, tile: TileId) -> bool {
        let mut store = self.inner.store.lock();
        let waited = store.base_waits.peek(&tile).copied().unwrap_or(0);
        store.base_waits.insert(tile, waited.saturating_add(1));
        waited >= MAX_BASE_WAIT_FRAMES
    }
}

impl TileProvider for CogTileProvider {
    fn tile(&self, tile: TileId) -> Option<DecodedTile> {
        // Asked for first and unconditionally: the basemap provider starts its
        // own fetch when asked, so skipping this while the COG loads would leave
        // the basemap idle underneath.
        let base = self.base.as_ref().and_then(|provider| provider.tile(tile));

        // Read before locking `store` inside `decide` (see its docs).
        let now = self.inner.ctx.input(|input| input.time);
        let (outcome, requests) = self.decide(tile, now);
        for (range, job) in requests {
            self.inner
                .transport
                .request_range(self.inner.url.clone(), range, job, self.sink());
        }
        // The open is kicked from the frame loop rather than from `new`, so a
        // provider that is built and never drawn issues no requests at all.
        {
            let needs_kick = {
                let store = self.inner.store.lock();
                matches!(store.stage, Stage::Opening(_)) && !store.header_inflight
            };
            if needs_kick {
                self.inner.advance_open();
            }
        }

        match outcome {
            Outcome::Wait => None,
            Outcome::BaseOnly => base,
            Outcome::Cog(decoded) => {
                let opaque = decoded
                    .rgba()
                    .chunks_exact(4)
                    .all(|pixel| pixel[3] == u8::MAX);
                if opaque || self.base.is_none() {
                    return Some(decoded);
                }
                match base {
                    Some(base) => match blend_over(&decoded, &base) {
                        Ok(blended) => Some(blended),
                        Err(error) => {
                            tracing::warn!("oxigis-ui: COG/basemap blend failed: {error}");
                            Some(decoded)
                        }
                    },
                    None if self.base_wait_expired(tile) => Some(decoded),
                    None => None,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CogLayerConfig, CogTileProvider, MAX_BASE_WAIT_FRAMES, RangeJob, RangeSink, RangeTransport,
        blend_over,
    };
    use crate::map_gpu::{BoxedTileProvider, TileProvider};
    use oxigis_render::{ByteRange, DecodedTile, LonLat, MemoryRangeFetch, RasterStretch, TileId};
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// The COG fixture every test reads: a small tiled EPSG:4326 GeoTIFF, built
    /// by `oxigis-render`'s own generator so the two crates cannot drift.
    fn fixture_bytes() -> Vec<u8> {
        // 8x8 px, 4x4 tiles, EPSG:4326, origin 10 °E / 50 °N at 0.5 °/px.
        oxigis_render::sample_cog_bytes()
    }

    /// Log of `(range, job)` pairs a test transport was asked for.
    type RequestLog = Arc<Mutex<Vec<(ByteRange, RangeJob)>>>;

    /// A transport that answers out of an in-memory buffer, synchronously.
    struct MemoryTransport {
        bytes: Vec<u8>,
        requests: Arc<Mutex<Vec<(ByteRange, RangeJob)>>>,
        answer: bool,
    }

    impl MemoryTransport {
        fn new(bytes: Vec<u8>) -> (Self, RequestLog) {
            let requests = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    bytes,
                    requests: Arc::clone(&requests),
                    answer: true,
                },
                requests,
            )
        }

        fn silent(bytes: Vec<u8>) -> (Self, RequestLog) {
            let (mut transport, requests) = Self::new(bytes);
            transport.answer = false;
            (transport, requests)
        }
    }

    impl RangeTransport for MemoryTransport {
        fn request_range(&self, _url: String, range: ByteRange, job: RangeJob, sink: RangeSink) {
            self.requests.lock().push((range, job));
            if !self.answer {
                return;
            }
            let start = range.start as usize;
            let end = (range.end as usize).min(self.bytes.len());
            if start >= self.bytes.len() {
                sink.deliver(
                    job,
                    Err(crate::TileError::permanent("range past end of file")),
                );
                return;
            }
            sink.deliver(job, Ok(self.bytes[start..end].to_vec()));
        }
    }

    /// A transport that always fails.
    struct BrokenTransport;

    impl RangeTransport for BrokenTransport {
        fn request_range(&self, _url: String, _range: ByteRange, job: RangeJob, sink: RangeSink) {
            sink.deliver(job, Err(crate::TileError::permanent("no such host")));
        }
    }

    /// A basemap stand-in that always has an opaque red tile ready.
    struct RedBase;

    impl TileProvider for RedBase {
        fn tile(&self, _tile: TileId) -> Option<DecodedTile> {
            let rgba = [255, 0, 0, 255].repeat(256 * 256);
            DecodedTile::new(256, 256, rgba).ok()
        }
    }

    /// A basemap stand-in that never has anything.
    struct EmptyBase;

    impl TileProvider for EmptyBase {
        fn tile(&self, _tile: TileId) -> Option<DecodedTile> {
            None
        }
    }

    fn provider(transport: Box<dyn RangeTransport>) -> CogTileProvider {
        CogTileProvider::new(
            &CogLayerConfig::new("https://example.test/cog.tif"),
            &egui::Context::default(),
            transport,
        )
        .expect("the provider must build")
    }

    /// A tile well inside the fixture's extent.
    fn covered_tile() -> TileId {
        LonLat::new(11.0, 49.0)
            .tile(8)
            .expect("a tile inside the fixture")
    }

    /// A tile on the other side of the planet.
    fn uncovered_tile() -> TileId {
        LonLat::new(-150.0, -40.0)
            .tile(8)
            .expect("a tile outside the fixture")
    }

    #[test]
    fn the_config_carries_url_stretch_and_attribution() {
        let config = CogLayerConfig::new("https://example.test/a.tif")
            .with_attribution("Example")
            .with_stretch(RasterStretch::Fixed { min: 0.0, max: 1.0 });
        assert_eq!(config.url, "https://example.test/a.tif");
        assert_eq!(config.attribution, "Example");
        assert!(matches!(config.stretch, RasterStretch::Fixed { .. }));
        assert_eq!(CogLayerConfig::new("x").stretch, RasterStretch::default());
    }

    #[test]
    fn the_header_is_read_on_the_first_frame() {
        let (transport, requests) = MemoryTransport::new(fixture_bytes());
        let provider = provider(Box::new(transport));
        assert!(provider.metadata().is_none());
        assert!(provider.tile(covered_tile()).is_none());
        assert!(
            requests
                .lock()
                .iter()
                .any(|(_, job)| matches!(job, RangeJob::Header { .. })),
            "the first frame must ask for the header"
        );
        let metadata = provider.metadata().expect("the header must have been read");
        assert_eq!(metadata.level_count(), 2);
        assert_eq!(metadata.epsg, Some(4326));
        assert!(provider.failure().is_none());
        assert_eq!(provider.url(), "https://example.test/cog.tif");
    }

    #[test]
    fn a_covered_tile_completes_and_is_cached() {
        let (transport, requests) = MemoryTransport::new(fixture_bytes());
        let provider = provider(Box::new(transport));
        let tile = covered_tile();
        // Frame 1 reads the header, frame 2 plans and reads the source tiles,
        // and because the transport answers synchronously the tile is ready then.
        assert!(provider.tile(tile).is_none());
        let mut decoded = None;
        for _ in 0..4 {
            if let Some(pixels) = provider.tile(tile) {
                decoded = Some(pixels);
                break;
            }
        }
        let decoded = decoded.expect("a covered tile must complete");
        assert_eq!(decoded.width(), 256);
        assert_eq!(decoded.height(), 256);
        assert!(
            decoded.rgba().chunks_exact(4).any(|pixel| pixel[3] == 255),
            "the tile is inside the image"
        );
        assert_eq!(provider.stats().ready, 1);

        let before = requests.lock().len();
        assert!(provider.tile(tile).is_some());
        assert_eq!(
            requests.lock().len(),
            before,
            "a cached tile must not be refetched"
        );
    }

    #[test]
    fn decide_serves_a_ready_cache_hit_without_touching_the_metadata_arc() {
        let (transport, _requests) = MemoryTransport::new(fixture_bytes());
        let provider = provider(Box::new(transport));
        let tile = covered_tile();
        let mut done = false;
        for _ in 0..4 {
            if provider.tile(tile).is_some() {
                done = true;
                break;
            }
        }
        assert!(done, "the fixture tile must complete");

        let strong_count = |tile_provider: &super::CogTileProvider| {
            let store = tile_provider.inner.store.lock();
            let super::Stage::Ready(metadata) = &store.stage else {
                panic!("the header must have been read by now");
            };
            Arc::strong_count(metadata)
        };
        let before = strong_count(&provider);
        // `decide` is called directly (accessible: `tests` is a submodule of
        // `cog_provider`) so the ready-cache hit can be exercised without the
        // `tile()` wrapper's basemap/compositing noise around it.
        let (outcome, requests) = provider.decide(tile, 0.0);
        assert!(requests.is_empty(), "a cache hit issues no range requests");
        assert!(matches!(outcome, super::Outcome::Cog(_)));
        assert_eq!(
            strong_count(&provider),
            before,
            "a ready-cache hit must not touch the metadata Arc at all — no clone, \
             not even a cheap one"
        );
    }

    #[test]
    fn the_open_hands_its_per_file_decode_settings_to_every_block_decode() {
        // `CogMetadata` has no field for `GDAL_NODATA` (42113) or `JPEGTables`
        // (347), so the provider keeps `CogOpen::decode_options()` beside it:
        // without the tables a JPEG COG cannot be decoded at all, and without
        // the nodata value a collar renders as opaque black. What must NOT
        // happen is the refresh clobbering the layer's own stretch with the
        // default `decode_options()` fills in.
        let stretch = RasterStretch::Fixed {
            min: 10.0,
            max: 200.0,
        };
        let (transport, _requests) = MemoryTransport::new(fixture_bytes());
        let provider = CogTileProvider::new(
            &CogLayerConfig::new("https://example.test/cog.tif").with_stretch(stretch),
            &egui::Context::default(),
            Box::new(transport),
        )
        .expect("the provider must build");
        assert_eq!(
            provider.inner.store.lock().decode_options.stretch,
            stretch,
            "before the header lands there is nothing else to answer with"
        );

        for _ in 0..4 {
            if provider.tile(covered_tile()).is_some() {
                break;
            }
        }
        let store = provider.inner.store.lock();
        assert!(
            matches!(store.stage, super::Stage::Ready(_)),
            "the header must have been read by now"
        );
        assert_eq!(store.decode_options.stretch, stretch);
        // The fixture declares neither tag, so both stay empty; what is pinned
        // is that they are read off the FILE rather than hard-coded away, which
        // the old `decode_cog_tile` call could not do at all.
        assert!(store.decode_options.jpeg_tables.is_empty());
        assert_eq!(store.decode_options.nodata, None);
    }

    #[test]
    fn a_tile_from_a_later_block_row_still_decodes() {
        // The provider now decodes with `decode_cog_block(level, tile_y, ..)`
        // so a striped TIFF's short last strip stops failing the whole map
        // tile. The other half of that contract is that a TILED file, whose
        // last block row IS padded to the full tile height, is unaffected: the
        // fixture is 8 px tall in 4 px blocks, so this tile's sources come from
        // block row 1 and must decode exactly as row 0's do.
        let (transport, _requests) = MemoryTransport::new(fixture_bytes());
        let provider = provider(Box::new(transport));
        let southern = LonLat::new(11.0, 47.0)
            .tile(8)
            .expect("a tile in the fixture's lower half");
        let mut decoded = None;
        for _ in 0..8 {
            if let Some(pixels) = provider.tile(southern) {
                decoded = Some(pixels);
                break;
            }
        }
        let decoded = decoded.expect("a covered tile must complete");
        assert!(
            decoded.rgba().chunks_exact(4).any(|pixel| pixel[3] == 255),
            "the lower half of the image is opaque, not the transparent tail \
             of a mis-sliced block"
        );
        assert!(provider.failure().is_none());
    }

    #[test]
    fn an_uncovered_tile_falls_through_to_the_basemap() {
        let (transport, _requests) = MemoryTransport::new(fixture_bytes());
        let provider = provider(Box::new(transport)).with_base(Box::new(RedBase));
        let tile = uncovered_tile();
        let mut base = None;
        for _ in 0..4 {
            if let Some(pixels) = provider.tile(tile) {
                base = Some(pixels);
                break;
            }
        }
        let base = base.expect("the basemap must show through");
        assert_eq!(&base.rgba()[..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn a_failed_open_leaves_the_basemap_visible() {
        let provider = provider(Box::new(BrokenTransport)).with_base(Box::new(RedBase));
        let tile = covered_tile();
        let mut base = None;
        for _ in 0..4 {
            if let Some(pixels) = provider.tile(tile) {
                base = Some(pixels);
                break;
            }
        }
        assert!(base.is_some(), "a broken COG must not hide the basemap");
        assert!(provider.failure().is_some());
    }

    #[test]
    fn a_pending_transport_leaves_the_tile_unanswered_once() {
        let (transport, requests) = MemoryTransport::silent(fixture_bytes());
        let provider = provider(Box::new(transport));
        for _ in 0..5 {
            assert!(provider.tile(covered_tile()).is_none());
        }
        assert_eq!(
            requests.lock().len(),
            1,
            "an in-flight header must not be requested again"
        );
    }

    #[test]
    fn a_transient_source_tile_failure_backs_off_on_the_wall_clock() {
        /// Opens the COG normally but fails every source-tile range with a
        /// retryable error, so the per-tile backoff can be exercised
        /// without also touching the header path (which has no retry
        /// policy of its own — any header failure is permanent).
        struct FlakySourceTransport {
            bytes: Vec<u8>,
        }

        impl RangeTransport for FlakySourceTransport {
            fn request_range(
                &self,
                _url: String,
                range: ByteRange,
                job: RangeJob,
                sink: RangeSink,
            ) {
                match job {
                    RangeJob::SourceTile { .. } => {
                        sink.deliver(job, Err(crate::TileError::transient("connection reset")));
                    }
                    _ => {
                        let start = range.start as usize;
                        let end = (range.end as usize).min(self.bytes.len());
                        sink.deliver(job, Ok(self.bytes[start..end].to_vec()));
                    }
                }
            }
        }

        let ctx = egui::Context::default();
        let provider = CogTileProvider::new(
            &CogLayerConfig::new("https://example.test/cog.tif"),
            &ctx,
            Box::new(FlakySourceTransport {
                bytes: fixture_bytes(),
            }),
        )
        .expect("the provider must build");
        let tile = covered_tile();

        // Frame 1 opens the header; frame 2 plans the tile and spends its
        // first, failing, attempt (the fixture's `covered_tile()` needs
        // exactly one source tile, confirmed empirically).
        for _ in 0..2 {
            let _ = provider.tile(tile);
        }
        let after_first = {
            let store = provider.inner.store.lock();
            store
                .failures
                .peek(&tile)
                .copied()
                .expect("the failure must be recorded")
        };
        assert!(
            !after_first.exhausted(),
            "one failure must not exhaust the budget"
        );

        // Many more frames at the SAME instant must not spend another
        // attempt — the clock, not the frame count, gates the retry.
        for _ in 0..20 {
            let _ = provider.tile(tile);
        }
        let after_many_frames = {
            let store = provider.inner.store.lock();
            store.failures.peek(&tile).copied().expect("still failed")
        };
        assert_eq!(
            after_many_frames, after_first,
            "no retry may happen before its wall-clock delay, however many frames pass"
        );

        // Advancing the clock past the backoff releases the next attempt.
        ctx.input_mut(|input| input.time = 1_000.0);
        let _ = provider.tile(tile);
        let after_clock_advance = {
            let store = provider.inner.store.lock();
            store.failures.peek(&tile).copied().expect("still failed")
        };
        assert_eq!(after_clock_advance.attempts(), after_first.attempts() + 1);
    }

    #[test]
    fn health_reports_failures_and_retry_failed_tiles_clears_the_gate() {
        /// Opens the COG normally but permanently refuses every source
        /// tile — a COG-open failure (`BrokenTransport`) never touches the
        /// per-tile failure LRU at all (see the identical guard in
        /// `a_transient_source_tile_failure_backs_off_on_the_wall_clock`),
        /// so `health`/`retry_failed_tiles` need a per-tile failure to
        /// observe anything.
        struct BrokenSourceTransport {
            bytes: Vec<u8>,
        }

        impl RangeTransport for BrokenSourceTransport {
            fn request_range(
                &self,
                _url: String,
                range: ByteRange,
                job: RangeJob,
                sink: RangeSink,
            ) {
                match job {
                    RangeJob::SourceTile { .. } => {
                        sink.deliver(job, Err(crate::TileError::permanent("access denied")));
                    }
                    _ => {
                        let start = range.start as usize;
                        let end = (range.end as usize).min(self.bytes.len());
                        sink.deliver(job, Ok(self.bytes[start..end].to_vec()));
                    }
                }
            }
        }

        let provider = provider(Box::new(BrokenSourceTransport {
            bytes: fixture_bytes(),
        }));
        assert_eq!(provider.health(), super::TileHealth::default());

        let tile = covered_tile();
        for _ in 0..4 {
            let _ = provider.tile(tile);
        }
        let health = provider.health();
        assert_eq!(health.failed, 1);
        assert_eq!(health.total_failures, 1);
        assert!(health.last_error.is_some());

        provider.retry_failed_tiles();
        assert_eq!(
            provider.health().failed,
            0,
            "the retry gate must be cleared"
        );
        assert_eq!(
            provider.health().total_failures,
            1,
            "the session history must survive a manual retry"
        );
    }

    #[test]
    fn a_transparent_tile_waits_a_bounded_number_of_frames_for_its_basemap() {
        let (transport, _requests) = MemoryTransport::new(fixture_bytes());
        let provider = provider(Box::new(transport)).with_base(Box::new(EmptyBase));
        // A tile that straddles the edge of the fixture has transparent pixels,
        // so it wants a basemap tile that `EmptyBase` never produces.
        let edge = LonLat::new(13.9, 46.1)
            .tile(9)
            .expect("a tile on the fixture's edge");
        let mut answered = None;
        for frame in 0..(MAX_BASE_WAIT_FRAMES as usize + 8) {
            if let Some(pixels) = provider.tile(edge) {
                answered = Some((frame, pixels));
                break;
            }
        }
        let (frame, pixels) = answered.expect("the wait must expire");
        assert!(frame > 1, "the COG tile must wait for the basemap first");
        assert_eq!(pixels.width(), 256);
    }

    #[test]
    fn blending_composites_alpha_over_the_base() {
        let over = DecodedTile::new(1, 1, vec![0, 0, 255, 128]).expect("a translucent pixel");
        let under = DecodedTile::new(1, 1, vec![255, 0, 0, 255]).expect("an opaque pixel");
        let blended = blend_over(&over, &under).expect("blending must succeed");
        let pixel = blended.rgba();
        assert!(pixel[0] > 100 && pixel[0] < 140, "red mixes halfway");
        assert!(pixel[2] > 100 && pixel[2] < 140, "blue mixes halfway");
        assert_eq!(pixel[3], 255);

        // A fully opaque top pixel passes through untouched.
        let opaque = DecodedTile::new(1, 1, vec![1, 2, 3, 255]).expect("an opaque pixel");
        let blended = blend_over(&opaque, &under).expect("blending must succeed");
        assert_eq!(blended.rgba(), &[1, 2, 3, 255]);
    }

    #[test]
    fn blending_resamples_a_differently_sized_base() {
        let over = DecodedTile::new(2, 2, [0, 0, 0, 0].repeat(4)).expect("a transparent tile");
        let under = DecodedTile::new(4, 4, [9, 9, 9, 255].repeat(16)).expect("a bigger tile");
        let blended = blend_over(&over, &under).expect("blending must succeed");
        assert_eq!(blended.width(), 2);
        assert_eq!(&blended.rgba()[..3], &[9, 9, 9]);
    }

    #[test]
    fn the_provider_is_boxable_as_a_send_sync_seam() {
        let (transport, _requests) = MemoryTransport::new(fixture_bytes());
        let boxed: BoxedTileProvider = Box::new(provider(Box::new(transport)));
        assert!(boxed.tile(covered_tile()).is_none());
    }

    #[test]
    fn the_memory_range_fetcher_and_the_transport_agree() {
        // Guards against the fixture drifting: the async reader in
        // `oxigis-render` and this provider must see the same file.
        let bytes = fixture_bytes();
        let source = oxigis_render::CogSource::new(MemoryRangeFetch::new(bytes.clone()));
        let metadata = futures::executor::block_on(source.open()).expect("the fixture must open");
        let (transport, _requests) = MemoryTransport::new(bytes);
        let provider = provider(Box::new(transport));
        assert!(provider.tile(covered_tile()).is_none());
        assert_eq!(
            provider.metadata().map(|meta| meta.level_count()),
            Some(metadata.level_count())
        );
    }
}
