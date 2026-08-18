//! [`MapRenderer`] — the composition point: viewport, tile cache and GPU
//! pipeline behind a four-call frame protocol.
//!
//! ```text
//! begin_frame(view) -> &[TilePlacement]   what the frame needs
//! accept_tile(id, DecodedTile)            feed decoded pixels back in
//! prepare(&Device, &Queue)                upload textures + instances
//! paint(&mut RenderPass)                  issue the draw calls
//! ```
//!
//! # Why this shape
//!
//! It is exactly what an `egui_wgpu::CallbackTrait` implementation needs, one
//! method at a time, *without* this crate depending on `egui` (the callback
//! type itself lives in `oxigis-ui`, per `survey-oxiui.md` §2):
//!
//! * `CallbackTrait::prepare` receives `&wgpu::Device`, `&wgpu::Queue` and a
//!   `&mut CallbackResources`, so a `MapRenderer` stored in those resources is
//!   reachable as `&mut` there — hence [`MapRenderer::prepare`] taking
//!   `&mut self`.
//! * `CallbackTrait::paint` receives `&mut wgpu::RenderPass<'static>` and only
//!   `&CallbackResources`, so painting must not mutate — hence
//!   [`MapRenderer::paint`] taking `&self`. A `&mut RenderPass<'static>`
//!   coerces into the `&mut wgpu::RenderPass<'_>` parameter unchanged.
//!
//! # I/O
//!
//! The renderer never fetches anything. [`MapRenderer::missing_tiles`] reports
//! what the current frame is waiting for; the shell drives its own
//! [`TileFetch`] and hands results back through
//! [`MapRenderer::accept_tile`].
//!
//! [`TileFetch`]: crate::source::TileFetch

use std::collections::{HashMap, HashSet};

use crate::error::RenderError;
use crate::gpu::{
    FULL_TILE_UV, MAX_TILE_TEXTURE_SIZE, TileDraw, TileInstance, TilePipeline, TileTexture,
};
use crate::mercator::TileId;
use crate::tile_cache::{CacheStats, TileCache};
use crate::viewport::{MAX_VISIBLE_TILES, MapView, TilePlacement};

/// Default number of tile textures kept resident on the GPU.
///
/// Sized for a 4K viewport (~150 tiles at native scale) with room left over for
/// the tiles a pan or zoom just moved off screen, which is what the cache is
/// really for.
///
/// # Capacity contract
///
/// The capacity **must exceed the number of tiles a single frame shows**,
/// otherwise [`MapRenderer::prepare`] would evict tiles it uploaded earlier in
/// the same call and those tiles would be silently skipped by
/// [`MapRenderer::paint`]. `prepare` enforces that itself: it raises the
/// cache's ceiling to fit the frame (up to [`MAX_TEXTURE_CAPACITY`]) before
/// uploading anything, so this constant is a starting size rather than a limit
/// a caller has to reason about. What it does *not* raise is the byte budget —
/// see [`DEFAULT_TEXTURE_BYTES`].
pub const DEFAULT_TEXTURE_CAPACITY: usize = 512;

/// Ceiling [`MapRenderer::prepare`] will grow the texture cache to.
///
/// Twice [`MAX_VISIBLE_TILES`], so the largest frame the viewport will ever
/// produce still leaves room for the tiles a pan just moved off screen.
pub const MAX_TEXTURE_CAPACITY: usize = MAX_VISIBLE_TILES * 2;

/// Default device-memory budget for resident tile textures, in bytes.
///
/// Entry count alone does not bound VRAM: [`MAX_TILE_TEXTURE_SIZE`] permits a
/// 268 MB tile, so 512 entries could mean gigabytes. 768 MiB holds ~3000 tiles
/// of the usual 256 px (or ~750 of 512 px) and only ever binds when a source
/// serves unusually large ones. Change it with
/// [`MapRenderer::set_texture_byte_budget`].
///
/// It is a bound, not a reservation, and it is not raised to fit a frame the
/// way the entry ceiling is: a source serving tiles so large that one frame
/// exceeds the budget will evict and re-request within the frame.
/// [`crate::tile_cache::CacheStats::bytes`] is what makes that visible instead
/// of mysterious.
pub const DEFAULT_TEXTURE_BYTES: usize = 768 * 1024 * 1024;

/// How many zoom levels [`MapRenderer::prepare`] walks up looking for a
/// resident ancestor to stand in for a tile that has not arrived.
///
/// Four levels means a stand-in is at worst magnified 16x, which is blurry but
/// still recognisably the right place — past that the pixels say less than the
/// background does.
pub const DEFAULT_OVERZOOM_LEVELS: u8 = 4;

/// How many times a tile whose GPU upload failed is retried before it is
/// quarantined for good.
const MAX_UPLOAD_ATTEMPTS: u32 = 3;

/// Frames a rejected tile waits before its first retry; each further attempt
/// doubles it.
const FIRST_RETRY_FRAMES: u64 = 120;

/// Upper bound on the quarantine, so a source failing every tile cannot grow
/// the renderer without limit. Least-recently-rejected entries fall out first,
/// which means a source failing more than this many tiles at once degrades from
/// "give up after three attempts" to "retry on the backoff forever" — still
/// bounded, and still far from the per-frame refetch the quarantine exists to
/// stop.
const QUARANTINE_CAPACITY: usize = 1024;

/// Decoded, GPU-ready tile pixels: tightly packed RGBA8, top row first.
///
/// The renderer core does not decode PNG/JPEG itself — a shell may already
/// have decoded pixels for free (e.g. the browser's `<img>`/`ImageBitmap`
/// path) and can build a [`DecodedTile`] directly. When that is not
/// available, the optional `decode` feature (on by default) provides
/// [`crate::decode::decode_tile`], a pure-computation PNG/JPEG decoder that
/// stays within this crate's no-I/O contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedTile {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl DecodedTile {
    /// Wraps decoded pixels.
    ///
    /// The dimension limit is [`MAX_TILE_TEXTURE_SIZE`], the same one
    /// [`crate::gpu::TilePipeline::upload_tile`] enforces, so a tile that
    /// cannot possibly be drawn is refused where it enters the renderer rather
    /// than a frame later. It applies to hand-built tiles too — the browser's
    /// `ImageBitmap` path bypasses [`crate::decode::decode_tile`] entirely.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidTileImage`] if the dimensions are zero,
    /// exceed [`MAX_TILE_TEXTURE_SIZE`], or `rgba` is not exactly
    /// `width * height * 4` bytes long.
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidTileImage(format!(
                "tile dimensions must be positive, got {width}x{height}"
            )));
        }
        if width > MAX_TILE_TEXTURE_SIZE || height > MAX_TILE_TEXTURE_SIZE {
            return Err(RenderError::InvalidTileImage(format!(
                "tile {width}x{height} exceeds the {MAX_TILE_TEXTURE_SIZE} texel limit"
            )));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|texels| texels.checked_mul(4));
        if expected != Some(rgba.len()) {
            return Err(RenderError::InvalidTileImage(format!(
                "tile {width}x{height} needs {expected:?} rgba bytes, got {}",
                rgba.len()
            )));
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    /// Tile width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Tile height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The RGBA8 pixels.
    #[must_use]
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Consumes the tile, returning its pixels.
    #[must_use]
    pub fn into_rgba(self) -> Vec<u8> {
        self.rgba
    }
}

/// A tile the GPU refused, and what the renderer intends to do about it.
#[derive(Debug, Clone, Copy)]
struct Quarantined {
    /// Upload attempts so far, capped at [`MAX_UPLOAD_ATTEMPTS`].
    attempts: u32,
    /// Frame number the tile may be requested again at.
    retry_at: u64,
}

/// One tile the last [`MapRenderer::prepare`] could not upload.
///
/// Non-fatal by construction: the rest of the frame uploaded, drew and painted
/// normally. A shell showing "3 tiles rejected" reads this; a shell that does
/// not is still correct, because the renderer stops re-requesting the tile on
/// its own.
#[derive(Debug, Clone)]
pub struct TileUploadFailure {
    /// The tile that could not be uploaded.
    pub tile: TileId,
    /// Why, as rendered by [`RenderError`].
    pub reason: String,
    /// How many times this tile has now failed.
    pub attempts: u32,
    /// Whether the renderer has given up on it entirely.
    pub quarantined: bool,
}

/// Composes a [`MapView`], a texture [`TileCache`] and a [`TilePipeline`] into
/// a per-frame raster map renderer.
#[derive(Debug)]
pub struct MapRenderer {
    view: MapView,
    target_format: wgpu::TextureFormat,
    pipeline: Option<TilePipeline>,
    textures: TileCache<TileTexture>,
    pending: HashMap<TileId, DecodedTile>,
    quarantine: TileCache<Quarantined>,
    placements: Vec<TilePlacement>,
    missing: Vec<TileId>,
    /// Scratch set backing `missing`'s de-duplication; kept as a field so a
    /// frame reuses one allocation. Repeats only arise from world copies.
    queued: HashSet<TileId>,
    draw_order: Vec<TileId>,
    instances: Vec<TileInstance>,
    uv_rects: Vec<[f32; 4]>,
    rejected: Vec<TileUploadFailure>,
    tint: [f32; 4],
    source_zoom_range: Option<(u8, u8)>,
    overzoom_levels: u8,
    frame: u64,
    /// Whether `placements` was computed for the current `view` and zoom
    /// clamp. Cleared by [`MapRenderer::set_source_zoom_range`] and
    /// [`MapRenderer::clear_textures`]; a tint change does not touch it,
    /// because `prepare` rebuilds every instance anyway.
    placements_valid: bool,
}

impl MapRenderer {
    /// Creates a renderer for a colour target of `target_format`.
    ///
    /// No GPU resource is created here: `target_format` is remembered so that
    /// the first [`MapRenderer::prepare`] can build the pipeline with the
    /// device it is handed. That matches `eframe`, which exposes
    /// `RenderState::target_format` at app-construction time but only lends out
    /// the device inside the frame callbacks.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if `texture_capacity` is zero.
    pub fn new(
        view: MapView,
        texture_capacity: usize,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        Ok(Self {
            view,
            target_format,
            pipeline: None,
            textures: TileCache::with_byte_budget(
                texture_capacity,
                DEFAULT_TEXTURE_BYTES,
                TileTexture::byte_size,
            )?,
            pending: HashMap::new(),
            quarantine: TileCache::new(QUARANTINE_CAPACITY)?,
            placements: Vec::new(),
            missing: Vec::new(),
            queued: HashSet::new(),
            draw_order: Vec::new(),
            instances: Vec::new(),
            uv_rects: Vec::new(),
            rejected: Vec::new(),
            tint: TileInstance::OPAQUE,
            source_zoom_range: None,
            overzoom_levels: DEFAULT_OVERZOOM_LEVELS,
            frame: 0,
            placements_valid: false,
        })
    }

    /// The viewport the last [`MapRenderer::begin_frame`] used.
    #[must_use]
    pub fn view(&self) -> MapView {
        self.view
    }

    /// Colour target format the pipeline is (or will be) built for.
    #[must_use]
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    /// Sets the RGBA multiplier applied to every tile; its alpha channel is the
    /// layer opacity. Takes effect on the next [`MapRenderer::prepare`].
    pub fn set_tint(&mut self, tint: [f32; 4]) {
        self.tint = tint;
    }

    /// The RGBA multiplier applied to every tile.
    #[must_use]
    pub fn tint(&self) -> [f32; 4] {
        self.tint
    }

    /// Fades the whole layer to `opacity` (`0..=1`, clamped; a non-finite value
    /// is fully opaque) **without touching its colour**.
    ///
    /// The per-layer half of [`MapRenderer::set_tint`], and the call a stack of
    /// raster layers is composited with: a renderer per layer, each faded on its
    /// own. Cheap by construction — the value is an instance attribute the next
    /// [`MapRenderer::prepare`] rewrites anyway, so no texture is dropped and
    /// nothing is re-fetched. That is the whole reason a slider can drive it
    /// every frame of a drag.
    pub fn set_opacity(&mut self, opacity: f32) {
        self.tint[3] = crate::gpu::opacity_tint(opacity)[3];
    }

    /// The layer opacity — the alpha channel of [`MapRenderer::tint`].
    #[must_use]
    pub fn opacity(&self) -> f32 {
        self.tint[3]
    }

    /// Restricts the tile zooms this renderer asks for to what a source can
    /// actually serve, as `(min_zoom, max_zoom)`.
    ///
    /// Past `max_zoom` the frame requests `max_zoom` tiles and draws them
    /// magnified, which is what makes zooming into a detail-limited archive
    /// show coarse imagery instead of nothing. `min_zoom` is recorded but not
    /// clamped against: requesting deeper tiles than the camera needs would
    /// multiply the tile count by four per level.
    ///
    /// [`None`] (the default) requests [`MapView::tile_zoom`] unchanged.
    pub fn set_source_zoom_range(&mut self, range: Option<(u8, u8)>) {
        let normalized = range.map(|(min, max)| (min.min(max), max.max(min)));
        if self.source_zoom_range != normalized {
            self.source_zoom_range = normalized;
            self.placements_valid = false;
        }
    }

    /// The source zoom range set by [`MapRenderer::set_source_zoom_range`].
    #[must_use]
    pub fn source_zoom_range(&self) -> Option<(u8, u8)> {
        self.source_zoom_range
    }

    /// Zoom level the current frame requests tiles at.
    #[must_use]
    pub fn request_zoom(&self) -> u8 {
        match self.source_zoom_range {
            Some((_, max)) => self.view.tile_zoom().min(max),
            None => self.view.tile_zoom(),
        }
    }

    /// How many zoom levels [`MapRenderer::prepare`] may walk up looking for a
    /// stand-in for a tile that has not arrived. Zero disables the fallback.
    pub fn set_overzoom_levels(&mut self, levels: u8) {
        self.overzoom_levels = levels;
    }

    /// The parent-fallback depth, [`DEFAULT_OVERZOOM_LEVELS`] by default.
    #[must_use]
    pub fn overzoom_levels(&self) -> u8 {
        self.overzoom_levels
    }

    /// Replaces the device-memory budget of the texture cache, evicting down to
    /// it immediately.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if the budget is `Some(0)`.
    pub fn set_texture_byte_budget(&mut self, bytes: Option<usize>) -> Result<(), RenderError> {
        self.textures.set_max_bytes(bytes)
    }

    /// Starts a frame for `view` and returns the tiles it needs, placed on
    /// screen and ordered centre-outward.
    ///
    /// The list has one entry per on-screen *copy* of a tile, so a viewport
    /// straddling the antimeridian is filled on both sides; the same
    /// [`TileId`] may therefore appear more than once with different
    /// coordinates. [`MapRenderer::missing_tiles`] is de-duplicated.
    ///
    /// The fetch queue is rebuilt on every call — that is what lets an
    /// accepted or a retried tile change it — while the placement list is
    /// recomputed only when the camera or the source zoom range moved. So
    /// calling this repeatedly with the same view is idempotent and costs one
    /// pass over the placements, not a re-plan.
    pub fn begin_frame(&mut self, view: MapView) -> &[TilePlacement] {
        self.frame = self.frame.saturating_add(1);
        if !self.placements_valid || self.view != view {
            self.view = view;
            let zoom = self.request_zoom();
            let mut placements = core::mem::take(&mut self.placements);
            view.visible_placements_into(zoom, &mut placements);
            self.placements = placements;
            self.placements_valid = true;
        }
        self.refresh_missing();
        &self.placements
    }

    /// Rebuilds the fetch queue from the current placements and cache contents.
    fn refresh_missing(&mut self) {
        let mut missing = core::mem::take(&mut self.missing);
        missing.clear();
        self.queued.clear();
        for index in 0..self.placements.len() {
            let tile = self.placements[index].tile;
            if self.textures.contains(&tile) || self.pending.contains_key(&tile) {
                continue;
            }
            if self.is_quarantined(tile) {
                continue;
            }
            if self.queued.insert(tile) {
                missing.push(tile);
            }
        }
        self.missing = missing;
    }

    /// Whether `tile` is being held back from the fetch queue: either its
    /// retry backoff has not elapsed, or it has failed too often to be worth
    /// asking for again.
    fn is_quarantined(&self, tile: TileId) -> bool {
        match self.quarantine.peek(&tile) {
            Some(entry) => entry.attempts >= MAX_UPLOAD_ATTEMPTS || self.frame < entry.retry_at,
            None => false,
        }
    }

    /// Placements computed by the last [`MapRenderer::begin_frame`].
    #[must_use]
    pub fn placements(&self) -> &[TilePlacement] {
        &self.placements
    }

    /// Tiles the current frame needs that are neither GPU-resident nor already
    /// accepted and waiting for upload.
    ///
    /// This is the fetch queue: de-duplicated (one entry however many world
    /// copies of a tile are on screen) and ordered centre-outward, so fetching
    /// in order fills the middle of the screen first. Tiles whose upload has
    /// failed are held back until their retry backoff elapses — see
    /// [`MapRenderer::rejected_uploads`].
    #[must_use]
    pub fn missing_tiles(&self) -> &[TileId] {
        &self.missing
    }

    /// Whether `tile` is GPU-resident or waiting for upload.
    #[must_use]
    pub fn has_tile(&self, tile: &TileId) -> bool {
        self.textures.contains(tile) || self.pending.contains_key(tile)
    }

    /// Hands decoded pixels to the renderer.
    ///
    /// The tile is queued and uploaded by the next [`MapRenderer::prepare`];
    /// accepting a tile that is no longer visible is harmless, it simply enters
    /// the LRU cache. Accepting the same tile twice before a `prepare` replaces
    /// the queued pixels.
    pub fn accept_tile(&mut self, tile: TileId, decoded: DecodedTile) {
        self.pending.insert(tile, decoded);
        self.missing.retain(|queued| queued != &tile);
    }

    /// Number of accepted tiles still waiting to be uploaded.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Counters of the GPU texture cache.
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        self.textures.stats()
    }

    /// Tiles the last [`MapRenderer::prepare`] could not upload.
    ///
    /// Cleared at the start of every `prepare`, so this is the current frame's
    /// news and not a running log. A tile appearing here is already out of
    /// [`MapRenderer::missing_tiles`]; one with `quarantined` set will not be
    /// asked for again until [`MapRenderer::retry_rejected_tiles`] or
    /// [`MapRenderer::clear_textures`].
    #[must_use]
    pub fn rejected_uploads(&self) -> &[TileUploadFailure] {
        &self.rejected
    }

    /// Number of tiles currently held out of the fetch queue after a failed
    /// upload.
    #[must_use]
    pub fn quarantined_len(&self) -> usize {
        self.quarantine.len()
    }

    /// Forgets every upload failure, so quarantined tiles are requested again
    /// on the next [`MapRenderer::begin_frame`].
    pub fn retry_rejected_tiles(&mut self) {
        self.quarantine.clear();
        self.rejected.clear();
    }

    /// Drops every GPU-resident tile (e.g. after a device loss or a style
    /// change). Pending pixels are kept — accepted pixels stay valid across a
    /// device change — and so are the placements.
    ///
    /// [`MapRenderer::missing_tiles`] is recomputed here rather than at the
    /// next [`MapRenderer::begin_frame`]: it was filtered against the cache
    /// contents this call just dropped, so leaving it alone would under-report
    /// to a shell that swaps providers and reads the queue before the next
    /// frame. Upload failures are forgotten too, since a new provider deserves
    /// a fresh attempt at the tiles the old one served badly.
    pub fn clear_textures(&mut self) {
        self.textures.clear();
        self.draw_order.clear();
        self.quarantine.clear();
        self.rejected.clear();
        self.placements_valid = false;
        self.refresh_missing();
    }

    /// Builds the pipeline if needed, uploads accepted tiles, and rewrites the
    /// instance buffer for the current frame.
    ///
    /// Maps one-to-one onto `egui_wgpu::CallbackTrait::prepare`.
    ///
    /// # A rejected tile does not fail the frame
    ///
    /// One tile the GPU refuses (an oversized image is the only reachable
    /// case) must not take the frame down with it: the remaining tiles would
    /// stay un-uploaded, the instance buffer would keep the *previous* frame's
    /// rectangles while [`MapRenderer::paint`] drew this frame's still-resident
    /// textures through them, and the offending tile — in neither the cache nor
    /// the pending queue — would be re-requested every single frame. Each
    /// upload is therefore handled on its own: failures land in
    /// [`MapRenderer::rejected_uploads`] and in a bounded quarantine with an
    /// exponential retry backoff, and the frame goes on to place and upload
    /// everything else.
    ///
    /// # Missing tiles fall back to their parent
    ///
    /// A visible tile that is not resident is drawn from the deepest resident
    /// ancestor within [`MapRenderer::overzoom_levels`], sampled through the
    /// sub-rectangle it occupies inside it. That is what keeps a pan or a zoom
    /// showing stretched imagery instead of background while the new tiles are
    /// in flight.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidViewport`] from placement conversion and
    /// [`RenderError::Gpu`] from the instance buffer. Per-tile upload failures
    /// are *not* errors; see above.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), RenderError> {
        if self.pipeline.is_none() {
            self.pipeline = Some(TilePipeline::new(device, self.target_format)?);
        }
        // The capacity contract, enforced rather than documented: a cache that
        // cannot hold the whole frame would evict tiles this very call uploaded.
        let needed = self.placements.len().saturating_add(self.pending.len());
        if needed >= self.textures.capacity() {
            let grown = needed
                .saturating_add(1)
                .checked_next_power_of_two()
                .unwrap_or(usize::MAX)
                .min(MAX_TEXTURE_CAPACITY)
                .max(self.textures.capacity());
            self.textures.set_capacity(grown)?;
        }

        let Some(pipeline) = self.pipeline.as_ref() else {
            return Err(RenderError::NotPrepared);
        };

        self.rejected.clear();
        // Collected rather than handled in place: recording a failure needs
        // `&mut self`, which the borrowed pipeline rules out until the loop is
        // over. Stays empty — and unallocated — on a healthy frame.
        let mut failures: Vec<(TileId, String)> = Vec::new();
        for (tile, decoded) in self.pending.drain().collect::<Vec<_>>() {
            let upload = pipeline.upload_tile(
                device,
                queue,
                decoded.width(),
                decoded.height(),
                decoded.rgba(),
            );
            match upload {
                Ok(texture) => {
                    self.quarantine.remove(&tile);
                    self.textures.insert(tile, texture);
                }
                Err(error) => failures.push((tile, error.to_string())),
            }
        }
        for (tile, reason) in failures {
            self.quarantine_tile(tile, reason);
        }

        let size_px = self.view.size_px();
        let mut instances = core::mem::take(&mut self.instances);
        let mut uv_rects = core::mem::take(&mut self.uv_rects);
        let mut draw_order = core::mem::take(&mut self.draw_order);
        instances.clear();
        uv_rects.clear();
        draw_order.clear();
        let build = self.build_draw_list(size_px, &mut instances, &mut uv_rects, &mut draw_order);
        self.instances = instances;
        self.uv_rects = uv_rects;
        self.draw_order = draw_order;
        if build.is_err() {
            // A frame that cannot be placed must not leave the GPU holding the
            // previous one's rectangles: `paint` would draw this frame's
            // textures through them, i.e. at last frame's screen positions.
            self.instances.clear();
            self.uv_rects.clear();
            self.draw_order.clear();
        }

        // `paint` indexes the instance buffer by position in `draw_order`, so
        // the three lists are one invariant, not three.
        let consistent = self.instances.len() == self.draw_order.len()
            && self.uv_rects.len() == self.instances.len();
        if !consistent {
            return Err(RenderError::Gpu(format!(
                "draw list is inconsistent: {} instances, {} uv rects, {} textures",
                self.instances.len(),
                self.uv_rects.len(),
                self.draw_order.len()
            )));
        }

        let Some(pipeline) = self.pipeline.as_mut() else {
            return Err(RenderError::NotPrepared);
        };
        pipeline.upload_instances_uv(device, queue, &self.instances, &self.uv_rects)?;
        build
    }

    /// Turns this frame's placements into parallel instance / UV / texture
    /// lists, substituting a resident ancestor wherever the tile itself has
    /// not arrived.
    fn build_draw_list(
        &mut self,
        size_px: [f32; 2],
        instances: &mut Vec<TileInstance>,
        uv_rects: &mut Vec<[f32; 4]>,
        draw_order: &mut Vec<TileId>,
    ) -> Result<(), RenderError> {
        for index in 0..self.placements.len() {
            let placement = self.placements[index];
            // `get` rather than `contains`: drawing a tile is a use, and that is
            // what keeps visible tiles at the hot end of the LRU.
            let drawn = if self.textures.get(&placement.tile).is_some() {
                Some((placement.tile, FULL_TILE_UV))
            } else {
                self.resident_ancestor(placement.tile)
            };
            let Some((tile, uv)) = drawn else {
                continue;
            };
            instances.push(TileInstance::from_placement(
                &placement, size_px, self.tint,
            )?);
            uv_rects.push(uv);
            draw_order.push(tile);
        }
        Ok(())
    }

    /// The deepest resident ancestor of `tile` within
    /// [`MapRenderer::overzoom_levels`], with the sub-rectangle `tile` occupies
    /// inside it.
    fn resident_ancestor(&mut self, tile: TileId) -> Option<(TileId, [f32; 4])> {
        let levels = self.overzoom_levels;
        // The search probes without touching the counters — a screen of
        // missing tiles would otherwise report four misses each and drown the
        // statistic that measures the fetch path.
        let found = {
            let textures = &self.textures;
            nearest_ancestor(tile, levels, |ancestor| textures.contains(&ancestor))
        };
        let (ancestor, uv) = found?;
        // The winner *is* used, though: an ancestor propping up a whole screen
        // of missing tiles must not be the next eviction victim.
        let _ = self.textures.get(&ancestor);
        Some((ancestor, uv))
    }

    /// Records a failed upload: the tile leaves the fetch queue until its
    /// backoff elapses, and the frame's caller can see why.
    fn quarantine_tile(&mut self, tile: TileId, reason: String) {
        let attempts = self
            .quarantine
            .peek(&tile)
            .map_or(1, |entry| entry.attempts.saturating_add(1));
        // 120 frames, then 240, then 480 — long enough that a permanently bad
        // tile costs one request every couple of seconds rather than one per
        // frame, short enough that a transient device failure recovers.
        let backoff =
            FIRST_RETRY_FRAMES.saturating_mul(1u64 << attempts.clamp(1, 8).saturating_sub(1));
        self.quarantine.insert(
            tile,
            Quarantined {
                attempts,
                retry_at: self.frame.saturating_add(backoff),
            },
        );
        self.missing.retain(|queued| queued != &tile);
        self.rejected.push(TileUploadFailure {
            tile,
            reason,
            attempts,
            quarantined: attempts >= MAX_UPLOAD_ATTEMPTS,
        });
    }

    /// Draws every resident tile of the current frame.
    ///
    /// Maps one-to-one onto `egui_wgpu::CallbackTrait::paint`.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::NotPrepared`] if [`MapRenderer::prepare`] has
    /// never run, or [`RenderError::Gpu`] if the instance buffer and the draw
    /// list have gone out of sync.
    pub fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>) -> Result<(), RenderError> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Err(RenderError::NotPrepared);
        };
        let mut draws = Vec::with_capacity(self.draw_order.len());
        for (index, tile) in self.draw_order.iter().enumerate() {
            let Some(texture) = self.textures.peek(tile) else {
                return Err(RenderError::Gpu(format!(
                    "tile {tile:?} was dropped between prepare and paint"
                )));
            };
            let Ok(instance) = u32::try_from(index) else {
                return Err(RenderError::Gpu(
                    "draw list exceeds the addressable instance range".to_owned(),
                ));
            };
            draws.push(TileDraw { texture, instance });
        }
        pipeline.draw(render_pass, &draws)
    }
}

/// The nearest ancestor of `tile` — at most `levels` up — that `resident`
/// accepts, with the sub-rectangle `tile` occupies inside it.
///
/// Nearest rather than any: the shallower the substitution the less it is
/// magnified, so the search walks upwards and stops at the first hit. `levels`
/// of zero disables the fallback entirely.
fn nearest_ancestor(
    tile: TileId,
    levels: u8,
    mut resident: impl FnMut(TileId) -> bool,
) -> Option<(TileId, [f32; 4])> {
    let depth = levels.min(tile.z);
    for step in 1..=depth {
        let ancestor = tile.ancestor(step)?;
        if resident(ancestor) {
            return tile.sub_rect_in(&ancestor).map(|uv| (ancestor, uv));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    // GPU-touching paths (`prepare`/`paint`) need a live device and are
    // exercised by the shells; everything below is the device-free half of the
    // frame protocol.
    use super::{
        DEFAULT_OVERZOOM_LEVELS, DEFAULT_TEXTURE_BYTES, DEFAULT_TEXTURE_CAPACITY, DecodedTile,
        FIRST_RETRY_FRAMES, MAX_UPLOAD_ATTEMPTS, MapRenderer, nearest_ancestor,
    };
    use crate::error::RenderError;
    use crate::gpu::MAX_TILE_TEXTURE_SIZE;
    use crate::mercator::{LonLat, TileId};
    use crate::viewport::MapView;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

    fn view(zoom: f64, size: [f32; 2]) -> MapView {
        match MapView::new(LonLat::new(0.0, 0.0), zoom, size) {
            Ok(view) => view,
            Err(err) => panic!("view construction failed: {err}"),
        }
    }

    fn renderer(zoom: f64) -> MapRenderer {
        renderer_for(view(zoom, [256.0, 256.0]))
    }

    fn renderer_for(view: MapView) -> MapRenderer {
        match MapRenderer::new(view, DEFAULT_TEXTURE_CAPACITY, FORMAT) {
            Ok(renderer) => renderer,
            Err(err) => panic!("renderer construction failed: {err}"),
        }
    }

    fn tile(z: u8, x: u32, y: u32) -> TileId {
        match TileId::new(z, x, y) {
            Ok(tile) => tile,
            Err(err) => panic!("tile construction failed: {err}"),
        }
    }

    fn tile_pixels() -> DecodedTile {
        match DecodedTile::new(2, 2, vec![255u8; 16]) {
            Ok(tile) => tile,
            Err(err) => panic!("tile construction failed: {err}"),
        }
    }

    #[test]
    fn decoded_tiles_are_validated() {
        let tile = tile_pixels();
        assert_eq!(tile.width(), 2);
        assert_eq!(tile.height(), 2);
        assert_eq!(tile.rgba().len(), 16);
        assert_eq!(tile.into_rgba().len(), 16);

        assert!(matches!(
            DecodedTile::new(2, 2, vec![0u8; 15]),
            Err(RenderError::InvalidTileImage(_))
        ));
        assert!(matches!(
            DecodedTile::new(0, 2, Vec::new()),
            Err(RenderError::InvalidTileImage(_))
        ));
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(matches!(
            MapRenderer::new(view(0.0, [256.0, 256.0]), 0, FORMAT),
            Err(RenderError::InvalidCapacity(0))
        ));
    }

    #[test]
    fn begin_frame_reports_what_is_needed() {
        let mut renderer = renderer(0.0);
        let placements = renderer.begin_frame(view(0.0, [256.0, 256.0])).to_vec();
        assert_eq!(placements.len(), 1);
        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        assert_eq!(placements[0].tile, root);
        assert_eq!(renderer.missing_tiles(), [root]);
        assert!(!renderer.has_tile(&root));
        assert_eq!(renderer.placements().len(), 1);
        assert_eq!(renderer.target_format(), FORMAT);
    }

    #[test]
    fn accepting_a_tile_clears_it_from_the_queue() {
        let mut renderer = renderer(0.0);
        let _ = renderer.begin_frame(view(0.0, [256.0, 256.0]));
        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        assert_eq!(renderer.missing_tiles().len(), 1);

        renderer.accept_tile(root, tile_pixels());
        assert!(renderer.missing_tiles().is_empty());
        assert!(renderer.has_tile(&root));
        assert_eq!(renderer.pending_len(), 1);

        // Re-accepting replaces rather than duplicating.
        renderer.accept_tile(root, tile_pixels());
        assert_eq!(renderer.pending_len(), 1);

        // A new frame must not re-request an already accepted tile.
        let _ = renderer.begin_frame(view(0.0, [256.0, 256.0]));
        assert!(renderer.missing_tiles().is_empty());
    }

    #[test]
    fn missing_list_follows_the_viewport() {
        let mut renderer = renderer(1.0);
        let placements = renderer.begin_frame(view(1.0, [512.0, 512.0])).to_vec();
        assert_eq!(placements.len(), 4);
        assert_eq!(renderer.missing_tiles().len(), 4);

        // Centre-outward order is preserved from the viewport.
        let expected: Vec<TileId> = placements.iter().map(|p| p.tile).collect();
        assert_eq!(renderer.missing_tiles(), expected.as_slice());

        // Zooming back out changes both lists.
        let _ = renderer.begin_frame(view(0.0, [256.0, 256.0]));
        assert_eq!(renderer.missing_tiles().len(), 1);
        assert_eq!(renderer.view().zoom(), 0.0);
    }

    #[test]
    fn a_fresh_renderer_holds_nothing() {
        let renderer = renderer(0.0);
        let stats = renderer.cache_stats();
        assert_eq!(stats.len, 0);
        assert_eq!(stats.capacity, DEFAULT_TEXTURE_CAPACITY);
        assert_eq!(renderer.pending_len(), 0);
        assert!(renderer.placements().is_empty());
        assert!(renderer.missing_tiles().is_empty());
    }

    #[test]
    fn tint_is_settable() {
        let mut renderer = renderer(0.0);
        assert_eq!(renderer.tint(), [1.0, 1.0, 1.0, 1.0]);
        renderer.set_tint([1.0, 1.0, 1.0, 0.5]);
        assert_eq!(renderer.tint(), [1.0, 1.0, 1.0, 0.5]);
    }

    #[test]
    fn opacity_fades_the_layer_without_recolouring_or_dropping_anything() {
        let mut renderer = renderer(0.0);
        assert_eq!(renderer.opacity(), 1.0);

        // A colour tint set first must survive a later opacity drag: the two
        // are different controls on the same multiplier, and a slider that
        // silently reset a layer's colourisation would be a second lie.
        renderer.set_tint([0.5, 0.25, 0.75, 1.0]);
        renderer.set_opacity(0.25);
        assert_eq!(renderer.tint(), [0.5, 0.25, 0.75, 0.25]);
        assert_eq!(renderer.opacity(), 0.25);

        // Clamped and NaN-proof, exactly as `opacity_tint` promises.
        renderer.set_opacity(-1.0);
        assert_eq!(renderer.opacity(), 0.0);
        renderer.set_opacity(9.0);
        assert_eq!(renderer.opacity(), 1.0);
        renderer.set_opacity(f32::NAN);
        assert_eq!(renderer.opacity(), 1.0);
    }

    #[test]
    fn an_opacity_change_costs_no_tile() {
        // THE property that lets a stacked layer's slider be honoured live: a
        // fade must not invalidate placements, evict a texture or re-open the
        // fetch queue — otherwise every frame of a drag would blank the map.
        let camera = view(1.0, [512.0, 512.0]);
        let mut renderer = renderer_for(camera);
        let _ = renderer.begin_frame(camera);
        for placement in renderer.placements().to_vec() {
            renderer.accept_tile(placement.tile, tile_pixels());
        }
        let pending = renderer.pending_len();
        let placements = renderer.placements().to_vec();
        assert!(renderer.missing_tiles().is_empty());

        renderer.set_opacity(0.3);
        assert_eq!(renderer.pending_len(), pending, "no tile is dropped");
        assert!(
            renderer.missing_tiles().is_empty(),
            "a fade must not re-open the fetch queue"
        );
        // And the very next frame is the same frame, not a re-plan.
        let _ = renderer.begin_frame(camera);
        assert_eq!(renderer.placements(), placements.as_slice());
        assert!(renderer.missing_tiles().is_empty());
    }

    #[test]
    fn clearing_textures_keeps_pending_work() {
        let mut renderer = renderer(0.0);
        let _ = renderer.begin_frame(view(0.0, [256.0, 256.0]));
        let root = tile(0, 0, 0);
        renderer.accept_tile(root, tile_pixels());
        renderer.clear_textures();
        assert_eq!(renderer.pending_len(), 1);
        assert_eq!(renderer.cache_stats().len, 0);
        // Accepted pixels are still valid, so the tile is not re-requested.
        assert!(renderer.missing_tiles().is_empty());
    }

    #[test]
    fn clearing_textures_refreshes_the_fetch_queue() {
        // A shell swapping providers calls `clear_textures` and then reads the
        // queue, before the next frame. `missing` was filtered against the
        // cache contents this call just dropped, so it has to be recomputed
        // here rather than one frame later.
        let mut renderer = renderer(1.0);
        let frame = view(1.0, [512.0, 512.0]);
        let _ = renderer.begin_frame(frame);
        assert_eq!(renderer.missing_tiles().len(), 4);

        // Simulate the four tiles having arrived and been uploaded by
        // accepting them, then verify the queue empties.
        for placement in renderer.placements().to_vec() {
            renderer.accept_tile(placement.tile, tile_pixels());
        }
        assert!(renderer.missing_tiles().is_empty());

        // Pending pixels survive a clear, so they stay out of the queue …
        renderer.clear_textures();
        assert!(renderer.missing_tiles().is_empty());
        assert_eq!(renderer.pending_len(), 4);

        // … but a tile held back by a past upload failure comes back, because
        // a new provider deserves a fresh attempt at it.
        let mut fresh = renderer_for(frame);
        let _ = fresh.begin_frame(frame);
        fresh.quarantine_tile(tile(1, 0, 0), "device lost".to_owned());
        assert_eq!(fresh.missing_tiles().len(), 3);
        fresh.clear_textures();
        assert_eq!(
            fresh.missing_tiles().len(),
            4,
            "a clear forgives past upload failures"
        );
    }

    #[test]
    fn a_tile_too_large_to_upload_is_refused_at_the_door() {
        // `upload_tile` rejects anything past 8192 texels; catching it here
        // keeps a tile that can never be drawn out of the pending queue, where
        // it would fail every `prepare` forever.
        let edge = MAX_TILE_TEXTURE_SIZE + 1;
        assert!(matches!(
            DecodedTile::new(edge, 1, vec![0u8; (edge as usize) * 4]),
            Err(RenderError::InvalidTileImage(_))
        ));
        assert!(matches!(
            DecodedTile::new(1, edge, vec![0u8; (edge as usize) * 4]),
            Err(RenderError::InvalidTileImage(_))
        ));
        // The limit itself is still accepted — the check is `>`, not `>=`.
        assert!(DecodedTile::new(MAX_TILE_TEXTURE_SIZE, 1, vec![0u8; 8192 * 4]).is_ok());
    }

    #[test]
    fn a_failed_upload_leaves_the_queue_and_returns_after_a_backoff() {
        let mut renderer = renderer(0.0);
        let frame = view(0.0, [256.0, 256.0]);
        let root = tile(0, 0, 0);
        let _ = renderer.begin_frame(frame);
        assert_eq!(renderer.missing_tiles(), [root]);

        renderer.quarantine_tile(root, "tile 9000x9000 exceeds the limit".to_owned());
        assert!(
            renderer.missing_tiles().is_empty(),
            "a rejected tile must not be re-requested every frame"
        );
        assert_eq!(renderer.quarantined_len(), 1);
        let [failure] = renderer.rejected_uploads() else {
            panic!("the failure must be reported once");
        };
        assert_eq!(failure.tile, root);
        assert_eq!(failure.attempts, 1);
        assert!(!failure.quarantined, "one failure is not a verdict");
        assert!(failure.reason.contains("9000x9000"));

        // It stays out for the whole backoff …
        for _ in 1..FIRST_RETRY_FRAMES {
            let _ = renderer.begin_frame(frame);
            assert!(renderer.missing_tiles().is_empty());
        }
        // … and comes back exactly once it elapses.
        let _ = renderer.begin_frame(frame);
        assert_eq!(renderer.missing_tiles(), [root]);
    }

    #[test]
    fn a_tile_that_keeps_failing_is_given_up_on_and_can_be_retried_on_demand() {
        let mut renderer = renderer(0.0);
        let frame = view(0.0, [256.0, 256.0]);
        let root = tile(0, 0, 0);
        let _ = renderer.begin_frame(frame);

        for attempt in 1..=MAX_UPLOAD_ATTEMPTS {
            renderer.quarantine_tile(root, "still too large".to_owned());
            let Some(failure) = renderer.rejected_uploads().last() else {
                panic!("every attempt is reported");
            };
            assert_eq!(failure.attempts, attempt);
            assert_eq!(failure.quarantined, attempt >= MAX_UPLOAD_ATTEMPTS);
        }

        // Past the attempt limit no backoff brings it back.
        for _ in 0..(FIRST_RETRY_FRAMES * 8) {
            let _ = renderer.begin_frame(frame);
        }
        assert!(renderer.missing_tiles().is_empty());

        // Only an explicit retry (or a `clear_textures`) does.
        renderer.retry_rejected_tiles();
        assert_eq!(renderer.quarantined_len(), 0);
        assert!(renderer.rejected_uploads().is_empty());
        let _ = renderer.begin_frame(frame);
        assert_eq!(renderer.missing_tiles(), [root]);
    }

    #[test]
    fn the_source_zoom_range_caps_what_the_frame_asks_for() {
        let camera = view(5.0, [1024.0, 1024.0]);
        let mut renderer = renderer_for(camera);
        assert_eq!(renderer.source_zoom_range(), None);
        assert_eq!(renderer.request_zoom(), 5);

        // An archive that stops at zoom 2: the frame asks for z2 tiles and has
        // them placed magnified, instead of asking for z5 tiles that do not
        // exist and rendering nothing.
        renderer.set_source_zoom_range(Some((0, 2)));
        assert_eq!(renderer.request_zoom(), 2);
        let placements = renderer.begin_frame(camera).to_vec();
        assert!(!placements.is_empty());
        assert!(placements.iter().all(|placement| placement.tile.z == 2));
        assert!(placements.iter().all(|placement| placement.size > 256.0));

        // The range is normalised, and changing it re-plans the same view.
        renderer.set_source_zoom_range(Some((7, 3)));
        assert_eq!(renderer.source_zoom_range(), Some((3, 7)));
        let placements = renderer.begin_frame(camera).to_vec();
        assert!(placements.iter().all(|placement| placement.tile.z == 5));

        renderer.set_source_zoom_range(None);
        let placements = renderer.begin_frame(camera).to_vec();
        assert!(placements.iter().all(|placement| placement.tile.z == 5));
    }

    #[test]
    fn repeated_world_copies_are_fetched_once() {
        // A 1024 px surface at zoom 0 shows five copies of the single world
        // tile; it must be placed five times and requested once.
        let wide = view(0.0, [1024.0, 256.0]);
        let mut renderer = renderer_for(wide);
        let placements = renderer.begin_frame(wide).to_vec();
        assert_eq!(placements.len(), 5);
        assert_eq!(renderer.missing_tiles(), [tile(0, 0, 0)]);
    }

    #[test]
    fn repeating_a_frame_does_not_change_it() {
        let camera = view(3.0, [800.0, 600.0]);
        let mut renderer = renderer_for(camera);
        let first = renderer.begin_frame(camera).to_vec();
        let missing = renderer.missing_tiles().to_vec();
        for _ in 0..8 {
            assert_eq!(renderer.begin_frame(camera), first.as_slice());
            assert_eq!(renderer.missing_tiles(), missing.as_slice());
        }
        // A different camera really does re-plan.
        let moved = camera.with_zoom(4.0);
        assert_ne!(renderer.begin_frame(moved).to_vec(), first);
    }

    #[test]
    fn a_stand_in_is_the_shallowest_resident_ancestor() {
        let deep = tile(6, 37, 22);

        assert_eq!(
            nearest_ancestor(deep, DEFAULT_OVERZOOM_LEVELS, |_| false),
            None
        );
        assert_eq!(
            nearest_ancestor(deep, 0, |_| true),
            None,
            "zero levels disables the fallback"
        );
        assert_eq!(
            nearest_ancestor(tile(0, 0, 0), DEFAULT_OVERZOOM_LEVELS, |_| true),
            None,
            "the world tile has no ancestor to fall back to"
        );
        assert_eq!(
            nearest_ancestor(deep, DEFAULT_OVERZOOM_LEVELS, |ancestor| ancestor.z == 0),
            None,
            "the search stops at the configured depth"
        );

        // With everything resident the parent wins: the shallower the
        // substitution, the less it is magnified.
        let Some((ancestor, uv)) = nearest_ancestor(deep, DEFAULT_OVERZOOM_LEVELS, |_| true) else {
            panic!("an ancestor is resident");
        };
        assert_eq!(Some(ancestor), deep.parent());
        assert_eq!(uv, [0.5, 0.0, 0.5, 0.5]);

        // With only a distant one resident, that one is used and the UV rect
        // shrinks to match the extra magnification.
        let Some((ancestor, uv)) =
            nearest_ancestor(deep, DEFAULT_OVERZOOM_LEVELS, |candidate| candidate.z == 3)
        else {
            panic!("the z3 ancestor is resident");
        };
        assert_eq!(ancestor, tile(3, 4, 2));
        assert_eq!(uv, [0.625, 0.75, 0.125, 0.125]);
    }

    #[test]
    fn the_texture_budget_is_reported_and_adjustable() {
        let mut renderer = renderer(0.0);
        let stats = renderer.cache_stats();
        assert_eq!(stats.bytes, 0);
        assert_eq!(stats.max_bytes, Some(DEFAULT_TEXTURE_BYTES));

        let Ok(()) = renderer.set_texture_byte_budget(Some(1_024)) else {
            panic!("lowering the budget failed");
        };
        assert_eq!(renderer.cache_stats().max_bytes, Some(1_024));
        assert!(matches!(
            renderer.set_texture_byte_budget(Some(0)),
            Err(RenderError::InvalidCapacity(0))
        ));

        assert_eq!(renderer.overzoom_levels(), DEFAULT_OVERZOOM_LEVELS);
        renderer.set_overzoom_levels(0);
        assert_eq!(renderer.overzoom_levels(), 0);
    }
}
