//! [`VectorLayerRenderer`] — the vector counterpart of [`crate::MapRenderer`],
//! with the same four-call frame protocol.
//!
//! ```text
//! begin_frame(view) -> &[TilePlacement]   what the frame needs
//! accept_mesh(id, VectorMesh)             feed a tessellated tile back in
//! prepare(&Device, &Queue)                upload buffers + instances
//! paint(&mut RenderPass, clip_origin_px)  issue the draw calls
//! ```
//!
//! # Why a separate renderer
//!
//! [`crate::MapRenderer`] is raster-specific all the way down — it caches
//! [`crate::gpu::TileTexture`]s, accepts [`crate::DecodedTile`] pixels and
//! drives [`crate::gpu::TilePipeline`]. Vector tiles share none of those, only
//! the viewport maths, so this is a parallel struct rather than a second mode
//! inside the raster one. Both can be driven from a single `egui_wgpu`
//! callback; the raster renderer paints first and the vector renderer on top.
//!
//! # Intended call sequence from `oxigis-ui` (the follow-up wiring task)
//!
//! Inside `CallbackTrait::prepare`, holding `&Device`/`&Queue` and the state
//! stored in `callback_resources`:
//!
//! 1. `let placements = vectors.begin_frame(view);`
//! 2. for every [`VectorLayerRenderer::missing_tiles`] entry, fetch and decode
//!    the MVT bytes with [`crate::decode_mvt`], build parameters with
//!    [`crate::vector::TessParams::for_tile`]`(view.tile_size_px(), extent, 0.35)`,
//!    tessellate with [`crate::vector::tessellate_tile`] and hand the mesh over
//!    with [`VectorLayerRenderer::accept_mesh`]. Tessellation is pure CPU work
//!    and may equally happen on a worker thread between frames.
//! 3. `vectors.prepare(device, queue)?;`
//!
//! Inside `CallbackTrait::paint`, holding `&mut RenderPass<'static>` and
//! `PaintCallbackInfo`:
//!
//! 4. `let viewport = info.viewport_in_pixels();`
//! 5. `vectors.paint(render_pass, [viewport.left_px as f32, viewport.top_px as f32])?;`
//!
//! The clip origin is what turns per-tile scissoring into framebuffer
//! coordinates (see [`crate::vector::tile_scissor`]); the clip *size* is the
//! view's own [`crate::MapView::size_px`], which the geometry contract already
//! requires to equal the callback rect.
//!
//! # Zoom changes
//!
//! Pixel widths are no longer frozen into a mesh — the vertices carry their
//! expansion and [`VectorLayerRenderer::prepare`] hands the pipeline the factor
//! that re-derives them for the current tile size (see [`crate::vector::tess`]),
//! so a stale mesh keeps its stroke widths through a smooth zoom. What a mesh
//! *does* bake is its level of detail, so a shell still calls
//! [`VectorLayerRenderer::clear_meshes`] when the zoom has moved far enough for
//! the simplification to show. Panning needs no such thing.
//!
//! # How much GPU memory
//!
//! A vector mesh is as large as its data, so the cache is bounded twice: by tile
//! count ([`DEFAULT_MESH_CAPACITY`]) and by bytes
//! ([`DEFAULT_MESH_BYTE_BUDGET`]). Retired buffers go to a
//! [`crate::vector::MeshBufferPool`] and are handed to the next upload that
//! fits, which is what keeps a zoom step from creating and destroying the whole
//! visible set's buffers.

use std::collections::HashMap;

use crate::error::RenderError;
use crate::gpu::TileInstance;
use crate::mercator::TileId;
use crate::tile_cache::{CacheStats, TileCache};
use crate::vector::pipeline::{
    MeshBufferPool, ScissorRect, VectorDraw, VectorPipeline, VectorTileGpu, tile_scissor,
};
use crate::vector::tess::VectorMesh;
use crate::viewport::{MapView, TilePlacement};

/// Default number of tile meshes kept resident on the GPU.
///
/// Same reasoning as [`crate::DEFAULT_TEXTURE_CAPACITY`]: it must exceed the
/// number of tiles one frame shows, or [`VectorLayerRenderer::prepare`] evicts
/// meshes it uploaded moments earlier.
pub const DEFAULT_MESH_CAPACITY: usize = 512;

/// Default ceiling on the GPU memory the mesh cache holds, in bytes.
///
/// The count alone is not a bound: a dense tile is megabytes of vertices, and
/// [`DEFAULT_MESH_CAPACITY`] of those would be gigabytes of VRAM. Whichever
/// limit is reached first evicts.
pub const DEFAULT_MESH_BYTE_BUDGET: u64 = 256 * 1024 * 1024;

/// Composes a [`MapView`], an LRU of GPU meshes and a [`VectorPipeline`] into a
/// per-frame vector-tile renderer.
#[derive(Debug)]
pub struct VectorLayerRenderer {
    view: MapView,
    target_format: wgpu::TextureFormat,
    pipeline: Option<VectorPipeline>,
    /// `None` marks a tile that was tessellated but drew nothing — still
    /// resident, so it is never re-requested.
    meshes: TileCache<Option<VectorTileGpu>>,
    /// Bytes the resident meshes occupy; kept in step with `meshes` by
    /// `insert_mesh` and `retire`.
    mesh_bytes: u64,
    mesh_byte_budget: u64,
    pool: MeshBufferPool,
    pending: HashMap<TileId, VectorMesh>,
    placements: Vec<TilePlacement>,
    missing: Vec<TileId>,
    draw_order: Vec<TilePlacement>,
    tint: [f32; 4],
}

impl VectorLayerRenderer {
    /// Creates a renderer for a colour target of `target_format`.
    ///
    /// As with [`crate::MapRenderer::new`], no GPU resource is created here:
    /// the pipeline is built by the first [`VectorLayerRenderer::prepare`].
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if `mesh_capacity` is zero.
    pub fn new(
        view: MapView,
        mesh_capacity: usize,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        Ok(Self {
            view,
            target_format,
            pipeline: None,
            meshes: TileCache::new(mesh_capacity)?,
            mesh_bytes: 0,
            mesh_byte_budget: DEFAULT_MESH_BYTE_BUDGET,
            pool: MeshBufferPool::default(),
            pending: HashMap::new(),
            placements: Vec::new(),
            missing: Vec::new(),
            draw_order: Vec::new(),
            tint: TileInstance::OPAQUE,
        })
    }

    /// GPU memory the resident meshes occupy, in bytes.
    #[must_use]
    pub fn mesh_bytes(&self) -> u64 {
        self.mesh_bytes
    }

    /// Ceiling on [`VectorLayerRenderer::mesh_bytes`]; meshes are evicted
    /// least-recently-used first until the total fits.
    #[must_use]
    pub fn mesh_byte_budget(&self) -> u64 {
        self.mesh_byte_budget
    }

    /// Replaces the byte ceiling, evicting immediately if the cache is already
    /// above the new one. A budget of zero still keeps one mesh: a frame with
    /// nothing resident would re-request everything every frame.
    pub fn set_mesh_byte_budget(&mut self, bytes: u64) {
        self.mesh_byte_budget = bytes;
        self.enforce_byte_budget();
    }

    /// Bytes of retired mesh buffers kept for reuse.
    #[must_use]
    pub fn pooled_bytes(&self) -> u64 {
        self.pool.bytes()
    }

    /// Drops the retired buffers instead of keeping them for the next upload.
    ///
    /// The one case that needs it is a lost device, whose buffers must not be
    /// handed to uploads on the replacement: `clear_meshes()` followed by
    /// `clear_pool()` leaves the renderer holding no GPU object at all.
    pub fn clear_pool(&mut self) {
        self.pool.clear();
    }

    /// The viewport the last [`VectorLayerRenderer::begin_frame`] used.
    #[must_use]
    pub fn view(&self) -> MapView {
        self.view
    }

    /// Colour target format the pipeline is (or will be) built for.
    #[must_use]
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    /// Sets the RGBA multiplier applied to every vector vertex; its alpha
    /// channel is a global layer opacity. Takes effect on the next
    /// [`VectorLayerRenderer::prepare`].
    pub fn set_tint(&mut self, tint: [f32; 4]) {
        self.tint = tint;
    }

    /// The RGBA multiplier applied to every vector vertex.
    #[must_use]
    pub fn tint(&self) -> [f32; 4] {
        self.tint
    }

    /// Fades the whole layer to `opacity` (`0..=1`, clamped; a non-finite value
    /// is fully opaque) **without touching the style's colours**.
    ///
    /// The exact twin of [`crate::MapRenderer::set_opacity`], normalised through
    /// the same [`crate::gpu::opacity_tint`], so a vector-tile layer and a raster
    /// layer at the same slider position fade by the same amount. The vector
    /// shader multiplies the instance tint into each vertex colour
    /// (`color.a * instance.tint.a`), which is what makes one uniform enough for
    /// a whole tiled layer whatever its per-feature paints are.
    ///
    /// Cheap: the tint is an instance attribute the next
    /// [`VectorLayerRenderer::prepare`] rewrites, so **nothing is
    /// re-tessellated** and no mesh is dropped.
    pub fn set_opacity(&mut self, opacity: f32) {
        self.tint[3] = crate::gpu::opacity_tint(opacity)[3];
    }

    /// The layer opacity — the alpha channel of [`VectorLayerRenderer::tint`].
    #[must_use]
    pub fn opacity(&self) -> f32 {
        self.tint[3]
    }

    /// Starts a frame for `view` and returns the tiles it needs, placed on
    /// screen and ordered centre-outward.
    pub fn begin_frame(&mut self, view: MapView) -> &[TilePlacement] {
        self.view = view;
        self.placements = view.visible_placements();
        self.missing = self
            .placements
            .iter()
            .map(|placement| placement.tile)
            .filter(|tile| !self.meshes.contains(tile) && !self.pending.contains_key(tile))
            .collect();
        &self.placements
    }

    /// Placements computed by the last [`VectorLayerRenderer::begin_frame`].
    #[must_use]
    pub fn placements(&self) -> &[TilePlacement] {
        &self.placements
    }

    /// Tiles the current frame needs that are neither GPU-resident nor already
    /// accepted and waiting for upload — the tessellation queue.
    #[must_use]
    pub fn missing_tiles(&self) -> &[TileId] {
        &self.missing
    }

    /// Whether `tile` is GPU-resident or waiting for upload.
    #[must_use]
    pub fn has_tile(&self, tile: &TileId) -> bool {
        self.meshes.contains(tile) || self.pending.contains_key(tile)
    }

    /// Hands a tessellated mesh to the renderer.
    ///
    /// An empty mesh is accepted too: the tile counts as resident and drawn (as
    /// nothing), so it is not requested again every frame.
    pub fn accept_mesh(&mut self, tile: TileId, mesh: VectorMesh) {
        self.pending.insert(tile, mesh);
        self.missing.retain(|queued| queued != &tile);
    }

    /// Number of accepted meshes still waiting to be uploaded.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Counters of the GPU mesh cache.
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        self.meshes.stats()
    }

    /// Drops every mesh whose baked tessellation parameters no longer apply —
    /// after a style change or a zoom step that moved the level of detail.
    ///
    /// Accepted-but-not-yet-uploaded meshes go too: they were tessellated with
    /// the same parameters the resident ones were, so keeping them would upload
    /// stale geometry that [`VectorLayerRenderer::begin_frame`] then refuses to
    /// re-request, leaving the tile visibly wrong until it is evicted. A caller
    /// that wants the queued work kept wants
    /// [`VectorLayerRenderer::clear_gpu_meshes`] instead.
    pub fn clear_meshes(&mut self) {
        self.clear_gpu_meshes();
        self.pending.clear();
    }

    /// Drops every GPU-resident mesh, keeping accepted meshes waiting for
    /// upload. Their buffers are recycled into the pool rather than destroyed.
    ///
    /// **Not** the call for a lost device: the retired buffers belong to that
    /// device and would be handed to the next upload. Follow it with
    /// [`VectorLayerRenderer::clear_pool`] there.
    pub fn clear_gpu_meshes(&mut self) {
        for tile in self.meshes.lru_order() {
            if let Some(retired) = self.meshes.remove(&tile) {
                self.retire(retired);
            }
        }
        self.meshes.clear();
        self.mesh_bytes = 0;
        self.draw_order.clear();
    }

    /// Adds an uploaded mesh, evicting whatever the count or byte budget
    /// requires — explicitly, so the buffers of the evicted meshes are recycled
    /// and the byte total stays exact.
    fn insert_mesh(&mut self, tile: TileId, uploaded: Option<VectorTileGpu>) {
        let bytes = uploaded.as_ref().map_or(0, VectorTileGpu::byte_size);
        if !self.meshes.contains(&tile) {
            while self.meshes.len() >= self.meshes.capacity() {
                let Some((_, retired)) = self.meshes.evict_lru() else {
                    break;
                };
                self.retire(retired);
            }
        }
        if let Some(previous) = self.meshes.insert(tile, uploaded) {
            self.retire(previous);
        }
        self.mesh_bytes = self.mesh_bytes.saturating_add(bytes);
        self.enforce_byte_budget();
    }

    /// Evicts least-recently-used meshes until the byte budget is met, always
    /// leaving at least one resident.
    fn enforce_byte_budget(&mut self) {
        while self.mesh_bytes > self.mesh_byte_budget && self.meshes.len() > 1 {
            let Some((_, retired)) = self.meshes.evict_lru() else {
                break;
            };
            self.retire(retired);
        }
    }

    /// Accounts a mesh out of the cache and offers its buffers to the pool.
    fn retire(&mut self, retired: Option<VectorTileGpu>) {
        let Some(mesh) = retired else {
            return;
        };
        self.mesh_bytes = self.mesh_bytes.saturating_sub(mesh.byte_size());
        self.pool.recycle(mesh);
    }

    /// Builds the pipeline if needed, uploads accepted meshes and rewrites the
    /// instance buffer for the current frame.
    ///
    /// Maps one-to-one onto `egui_wgpu::CallbackTrait::prepare`.
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Gpu`] from mesh validation and the instance
    /// buffer, and [`RenderError::InvalidViewport`] from placement conversion.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), RenderError> {
        if self.pipeline.is_none() {
            self.pipeline = Some(VectorPipeline::new(device, self.target_format)?);
        }

        let drained: Vec<(TileId, VectorMesh)> = self.pending.drain().collect();
        let mut uploaded = Vec::with_capacity(drained.len());
        {
            let Some(pipeline) = self.pipeline.as_ref() else {
                return Err(RenderError::NotPrepared);
            };
            for (tile, mesh) in drained {
                let gpu = if mesh.is_empty() {
                    None
                } else {
                    Some(pipeline.upload_mesh_pooled(device, queue, &mesh, &mut self.pool)?)
                };
                uploaded.push((tile, gpu));
            }
        }
        for (tile, gpu) in uploaded {
            self.insert_mesh(tile, gpu);
        }

        let size_px = self.view.size_px();
        let tile_size_px = self.view.tile_size_px();
        let mut instances = Vec::with_capacity(self.placements.len());
        let mut scales = Vec::with_capacity(self.placements.len());
        self.draw_order.clear();
        for index in 0..self.placements.len() {
            let placement = self.placements[index];
            // `get` rather than `contains`: drawing a tile is a use, and that
            // is what keeps visible tiles at the hot end of the LRU.
            let scale = match self.meshes.get(&placement.tile) {
                // The mesh may have been tessellated for another tile size;
                // this is the factor that gives its strokes the width the style
                // asked for at *this* one.
                Some(Some(mesh)) => mesh.offset_scale_at(tile_size_px),
                // Resident but empty, or not resident at all: nothing to draw.
                Some(None) | None => continue,
            };
            instances.push(TileInstance::from_placement(
                &placement, size_px, self.tint,
            )?);
            scales.push(scale);
            self.draw_order.push(placement);
        }

        let Some(pipeline) = self.pipeline.as_mut() else {
            return Err(RenderError::NotPrepared);
        };
        pipeline.upload_instances_scaled(device, queue, &instances, &scales)
    }

    /// Draws every resident mesh of the current frame, scissored to its tile.
    ///
    /// `clip_origin_px` is the top-left corner of the pass viewport inside the
    /// framebuffer — `[0.0, 0.0]` when the pass covers the whole surface, and
    /// `PaintCallbackInfo::viewport_in_pixels()` inside an `egui_wgpu`
    /// callback. The clip size is the view's own
    /// [`MapView::size_px`], which the geometry contract requires to equal the
    /// callback rect.
    ///
    /// Maps one-to-one onto `egui_wgpu::CallbackTrait::paint`.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::NotPrepared`] if
    /// [`VectorLayerRenderer::prepare`] has never run, or
    /// [`RenderError::Gpu`] if the instance buffer and the draw list have gone
    /// out of sync.
    pub fn paint(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        clip_origin_px: [f32; 2],
    ) -> Result<(), RenderError> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Err(RenderError::NotPrepared);
        };
        let size_px = self.view.size_px();
        let mut draws = Vec::with_capacity(self.draw_order.len());
        for (index, placement) in self.draw_order.iter().enumerate() {
            let Some(Some(mesh)) = self.meshes.peek(&placement.tile) else {
                return Err(RenderError::Gpu(format!(
                    "vector tile {:?} was dropped between prepare and paint",
                    placement.tile
                )));
            };
            let Ok(instance) = u32::try_from(index) else {
                return Err(RenderError::Gpu(
                    "vector draw list exceeds the addressable instance range".to_owned(),
                ));
            };
            let Some(scissor) = tile_scissor(placement, clip_origin_px, size_px) else {
                continue;
            };
            draws.push(VectorDraw {
                mesh,
                instance,
                scissor: Some(scissor),
            });
        }
        pipeline.draw(render_pass, &draws, self.clip_scissor(clip_origin_px))
    }

    /// The whole clip rectangle, restored after the scissored draws.
    fn clip_scissor(&self, clip_origin_px: [f32; 2]) -> Option<ScissorRect> {
        // The raw input is what the guard has to see: `f32::max` returns the
        // non-NaN operand, so clamping first would turn a NaN origin into a
        // finite zero and leave nothing for `is_finite` to catch.
        if !clip_origin_px.iter().all(|value| value.is_finite()) {
            return None;
        }
        let size_px = self.view.size_px();
        let origin_x = clip_origin_px[0].max(0.0);
        let origin_y = clip_origin_px[1].max(0.0);
        ScissorRect::new(
            origin_x as u32,
            origin_y as u32,
            size_px[0].ceil().max(0.0) as u32,
            size_px[1].ceil().max(0.0) as u32,
        )
    }
}

#[cfg(test)]
mod tests {
    // GPU-touching paths (`prepare`/`paint`) need a live device and are
    // exercised by the shells; everything below is the device-free half of the
    // frame protocol, mirroring `renderer.rs`.
    use super::{DEFAULT_MESH_BYTE_BUDGET, DEFAULT_MESH_CAPACITY, VectorLayerRenderer};
    use crate::error::RenderError;
    use crate::mercator::{LonLat, TileId};
    use crate::vector::tess::{VectorMesh, VectorVertex};
    use crate::viewport::MapView;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

    fn view(zoom: f64, size: [f32; 2]) -> MapView {
        match MapView::new(LonLat::new(0.0, 0.0), zoom, size) {
            Ok(view) => view,
            Err(error) => panic!("view construction failed: {error}"),
        }
    }

    fn renderer(zoom: f64) -> VectorLayerRenderer {
        match VectorLayerRenderer::new(view(zoom, [256.0, 256.0]), DEFAULT_MESH_CAPACITY, FORMAT) {
            Ok(renderer) => renderer,
            Err(error) => panic!("renderer construction failed: {error}"),
        }
    }

    fn mesh() -> VectorMesh {
        VectorMesh {
            vertices: vec![
                VectorVertex::new([0.0, 0.0], [0, 0, 0, 255]),
                VectorVertex::new([1.0, 0.0], [0, 0, 0, 255]),
                VectorVertex::new([0.0, 1.0], [0, 0, 0, 255]),
            ],
            indices: vec![0, 1, 2],
            baked_tile_size_px: 256.0,
        }
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(matches!(
            VectorLayerRenderer::new(view(0.0, [256.0, 256.0]), 0, FORMAT),
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
        assert_eq!(renderer.missing_tiles(), [root]);
        assert!(!renderer.has_tile(&root));
        assert_eq!(renderer.placements().len(), 1);
        assert_eq!(renderer.target_format(), FORMAT);
        assert_eq!(renderer.view().zoom(), 0.0);
    }

    #[test]
    fn accepting_a_mesh_clears_it_from_the_queue() {
        let mut renderer = renderer(0.0);
        let _ = renderer.begin_frame(view(0.0, [256.0, 256.0]));
        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        renderer.accept_mesh(root, mesh());
        assert!(renderer.missing_tiles().is_empty());
        assert!(renderer.has_tile(&root));
        assert_eq!(renderer.pending_len(), 1);

        // An empty mesh is still an answer: the tile is not re-requested.
        renderer.accept_mesh(root, VectorMesh::new());
        assert_eq!(renderer.pending_len(), 1);
        let _ = renderer.begin_frame(view(0.0, [256.0, 256.0]));
        assert!(renderer.missing_tiles().is_empty());
    }

    #[test]
    fn painting_before_preparing_is_an_error_free_of_a_device() {
        let renderer = renderer(0.0);
        assert_eq!(renderer.cache_stats().len, 0);
        assert_eq!(renderer.cache_stats().capacity, DEFAULT_MESH_CAPACITY);
        assert!(renderer.placements().is_empty());
        assert!(renderer.missing_tiles().is_empty());
    }

    #[test]
    fn clearing_meshes_drops_the_work_queued_with_the_old_parameters() {
        // Pending meshes were tessellated with the very parameters
        // `clear_meshes` declares invalid; keeping them would upload stale
        // geometry that `begin_frame` then never re-requests.
        let mut renderer = renderer(0.0);
        let _ = renderer.begin_frame(view(0.0, [256.0, 256.0]));
        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        renderer.accept_mesh(root, mesh());
        renderer.clear_meshes();
        assert_eq!(renderer.pending_len(), 0);
        assert_eq!(renderer.cache_stats().len, 0);
        assert!(!renderer.has_tile(&root));

        // And the tile is asked for again on the next frame.
        let _ = renderer.begin_frame(view(0.0, [256.0, 256.0]));
        assert_eq!(renderer.missing_tiles(), [root]);
    }

    #[test]
    fn clearing_only_the_gpu_meshes_keeps_pending_work() {
        let mut renderer = renderer(0.0);
        let _ = renderer.begin_frame(view(0.0, [256.0, 256.0]));
        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        renderer.accept_mesh(root, mesh());
        renderer.clear_gpu_meshes();
        assert_eq!(renderer.pending_len(), 1);
        assert_eq!(renderer.cache_stats().len, 0);
        assert_eq!(renderer.mesh_bytes(), 0);
    }

    #[test]
    fn the_byte_budget_is_settable_and_reported() {
        let mut renderer = renderer(0.0);
        assert_eq!(renderer.mesh_byte_budget(), DEFAULT_MESH_BYTE_BUDGET);
        assert_eq!(renderer.mesh_bytes(), 0);
        assert_eq!(renderer.pooled_bytes(), 0);
        renderer.set_mesh_byte_budget(1024);
        assert_eq!(renderer.mesh_byte_budget(), 1024);
        // Nothing resident, so tightening the budget evicts nothing.
        assert_eq!(renderer.cache_stats().evictions, 0);
        // The device-loss sequence leaves no GPU object behind.
        renderer.clear_meshes();
        renderer.clear_pool();
        assert_eq!(renderer.pooled_bytes(), 0);
        assert_eq!(renderer.mesh_bytes(), 0);
    }

    #[test]
    fn tint_is_settable() {
        let mut renderer = renderer(0.0);
        assert_eq!(renderer.tint(), [1.0, 1.0, 1.0, 1.0]);
        renderer.set_tint([1.0, 1.0, 1.0, 0.5]);
        assert_eq!(renderer.tint(), [1.0, 1.0, 1.0, 0.5]);
    }

    #[test]
    fn opacity_fades_the_layer_without_recolouring_or_re_tessellating() {
        let mut renderer = renderer(0.0);
        assert_eq!(renderer.opacity(), 1.0);
        renderer.set_tint([0.5, 0.25, 0.75, 1.0]);
        renderer.set_opacity(0.25);
        assert_eq!(
            renderer.tint(),
            [0.5, 0.25, 0.75, 0.25],
            "a fade must not discard a colourisation"
        );
        renderer.set_opacity(-1.0);
        assert_eq!(renderer.opacity(), 0.0);
        renderer.set_opacity(9.0);
        assert_eq!(renderer.opacity(), 1.0);
        renderer.set_opacity(f32::NAN);
        assert_eq!(renderer.opacity(), 1.0);
    }

    #[test]
    fn an_opacity_change_costs_no_mesh() {
        // The vector half of the property that lets a slider be honoured live:
        // a drag that re-tessellated would stutter the whole map.
        let mut renderer = renderer(0.0);
        let camera = view(0.0, [256.0, 256.0]);
        let _ = renderer.begin_frame(camera);
        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        renderer.accept_mesh(root, mesh());
        assert_eq!(renderer.pending_len(), 1);

        renderer.set_opacity(0.4);
        assert_eq!(renderer.pending_len(), 1, "no accepted mesh is discarded");
        assert!(renderer.has_tile(&root));
        let _ = renderer.begin_frame(camera);
        assert!(
            renderer.missing_tiles().is_empty(),
            "a fade must not re-open the tessellation queue"
        );
    }

    #[test]
    fn the_restore_scissor_covers_the_whole_clip_rect() {
        let renderer = renderer(0.0);
        let Some(rect) = renderer.clip_scissor([12.0, 34.0]) else {
            panic!("the clip rectangle is not empty");
        };
        assert_eq!(
            (rect.x, rect.y, rect.width, rect.height),
            (12, 34, 256, 256)
        );

        // A non-finite origin has no rectangle. `f32::max` returns the non-NaN
        // operand, so this only holds while the guard sees the raw input.
        assert!(renderer.clip_scissor([f32::NAN, 0.0]).is_none());
        assert!(renderer.clip_scissor([0.0, f32::NAN]).is_none());
        assert!(renderer.clip_scissor([f32::INFINITY, 0.0]).is_none());
        // Negative origins still clamp rather than vanish.
        let Some(clamped) = renderer.clip_scissor([-5.0, -7.0]) else {
            panic!("a negative origin clamps to the framebuffer corner");
        };
        assert_eq!((clamped.x, clamped.y), (0, 0));
    }

    #[test]
    fn the_missing_list_follows_the_viewport() {
        let mut renderer = renderer(1.0);
        let placements = renderer.begin_frame(view(1.0, [512.0, 512.0])).to_vec();
        assert_eq!(placements.len(), 4);
        let expected: Vec<TileId> = placements.iter().map(|p| p.tile).collect();
        assert_eq!(renderer.missing_tiles(), expected.as_slice());
    }
}
