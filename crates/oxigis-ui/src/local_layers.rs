//! The GPU half of local vector layers: an ordered stack of
//! [`LocalVectorLayer`]s, one mesh each, drawn over the tiled vector layer
//! (blueprint Phase 1 §1.1, render half).
//!
//! [`crate::local_vector`] turns a dataset into a synthetic tile; this module
//! owns what happens to it every frame. It is the local counterpart of
//! [`oxigis_render::VectorLayerRenderer`] and deliberately *not* an instance of
//! it: that type is a tile pyramid with an LRU keyed by
//! [`oxigis_render::TileId`], while a local stack is a short, explicitly ordered
//! list keyed by [`LayerId`] whose members each occupy the whole screen. What
//! the two share — [`oxigis_render::VectorPipeline`], the instance transform,
//! the vertex format — is shared by *using* it, not by subclassing it.
//!
//! # The frame
//!
//! ```text
//! prepare(device, queue, view)
//!   for every visible layer, in stack order:
//!     placement = layer.square().place(view)
//!     note it if the style changed or the on-screen size drifted
//!       by more than RETESSELLATE_RATIO
//!   re-tessellate at most MAX_TESSELLATIONS_PER_FRAME of those, in turn
//!   for every visible layer with a mesh: record an instance
//!   upload the instance buffer
//! paint(pass, clip_origin_px, view)
//!   one indexed draw per layer, scissored to the whole map viewport
//! ```
//!
//! # Draw order, and the scissor
//!
//! Local layers paint **after** the tiled vector layer and **before** labels:
//! data the user brought is what they are looking at, so it is never hidden by
//! basemap geometry. Within the stack, index `0` paints first — the same
//! painter's-algorithm convention the rest of the renderer uses.
//!
//! Unlike a map tile, a local layer is **not** scissored to its own quad. Its
//! quad is the dataset's bbox, and strokes, circle radii and halo padding
//! legitimately extend past it — clipping to the bbox would shave a boundary
//! feature's outline off. The clip rectangle is the map viewport itself, exactly
//! the one [`crate::label_frame`] uses for labels.
//!
//! # Re-tessellation
//!
//! A local layer's on-screen size is `bbox_span · world_pixels`, which grows
//! monotonically with the zoom — it is **not**
//! [`oxigis_render::MapView::tile_size_px`], which resets at every integer zoom
//! and is therefore the wrong cache key here. Each resident mesh records the
//! placement size it was baked at and is rebuilt when the ratio leaves
//! `1/√2 ..= √2` (see [`crate::local_vector::RETESSELLATE_RATIO`]), or when the
//! layer's style generation moves.
//!
//! The rebuild itself runs on the **render thread**, inside egui's `wgpu`
//! prepare hook, over the whole dataset — so it is budgeted
//! ([`MAX_TESSELLATIONS_PER_FRAME`]) and rotated, and what does not fit is
//! reported by [`LocalVectorRenderer::tessellation_backlog`] for the shell to
//! turn into another frame. Layers waiting their turn keep drawing the mesh
//! they have.

use eframe::wgpu;
use oxigis_core::LayerId;
use oxigis_render::{
    LabelTable, MapView, RenderError, TileInstance, TilePlacement, VectorDraw, VectorPipeline,
    VectorTile, VectorTileGpu,
};

use crate::label_frame::clip_scissor;
use crate::local_vector::{LocalVectorLayer, RETESSELLATE_RATIO};

/// How far outside the viewport a layer's quad may sit and still be drawn, in
/// physical pixels.
///
/// Generous on purpose: a stroke, a circle radius or a label halo extends past
/// the bbox the placement describes, and a wrongly culled layer is a visible
/// bug while a wrongly kept one costs one draw call.
pub const CULL_MARGIN_PX: f32 = 256.0;

/// How many local layers may be re-tessellated in a single
/// [`LocalVectorRenderer::prepare`].
///
/// `prepare` runs inside egui's `wgpu` prepare hook — the render thread — and
/// [`LocalVectorLayer::tessellate`] runs lyon over the WHOLE dataset, so an
/// unbudgeted stack of N layers could stall one frame N times over. One per
/// frame is the same shape the tiled path already has
/// (`vector_provider`'s `MAX_TESSELLATIONS_PER_FRAME`), just tighter because a
/// local layer is a whole dataset rather than one tile of one.
///
/// A layer whose turn has not come keeps drawing its previous mesh, which is
/// at worst [`RETESSELLATE_RATIO`] off in stroke width — the drift the
/// constant already declares acceptable — and a layer with no mesh yet simply
/// appears a frame or two later. [`LocalVectorRenderer::tessellation_backlog`]
/// is what tells a shell to ask for those frames.
pub const MAX_TESSELLATIONS_PER_FRAME: usize = 1;

/// How many classes a local layer may draw before this renderer says so in the
/// log (thematic v1.6).
///
/// NOT a cap — [`oxigis_core::MAX_STYLE_CLASSES`] is the cap, and it is
/// enforced in the model where it belongs. This is the threshold past which a
/// classified layer's per-frame cost stops being invisible: one tessellation
/// pass walks every tile layer, so a 64-class layer walks 195 of them
/// (3 families × 65 buckets) where a single-symbol one walks 3. The work is
/// still bounded and still budgeted by [`MAX_TESSELLATIONS_PER_FRAME`]; what
/// changes is that a user wondering why their map got slower now has one log
/// line naming the reason.
pub const CLASS_COUNT_NOTICE_THRESHOLD: usize = 24;

/// Everything the label pass needs to place one local layer's labels.
///
/// The tiled path hands [`crate::label_frame::LabelFrame`] a
/// [`crate::vector_provider::VectorTileSource`] it can look tiles up in; a local
/// layer already *has* its tile, so it hands over a borrow directly.
#[derive(Debug, Clone, Copy)]
pub struct LocalLabelJob<'a> {
    /// The layer's synthetic tile.
    pub tile: &'a VectorTile,
    /// Where it lands on screen this frame.
    pub placement: TilePlacement,
    /// The layer's own symbol rules.
    pub labels: &'a LabelTable,
}

/// One layer's mesh, resident on the GPU.
#[derive(Debug)]
struct Resident {
    /// [`None`] when the mesh tessellated to nothing — still resident, so it is
    /// not rebuilt every frame.
    gpu: Option<VectorTileGpu>,
    /// On-screen size the mesh's pixel widths were baked for.
    baked_size_px: f32,
    /// [`LocalVectorLayer::generation`] the mesh was built from.
    generation: u32,
}

/// One entry of the stack: a layer, its identity and its resident mesh.
#[derive(Debug)]
struct LocalEntry {
    /// Application-side identity; the key every entry point takes.
    id: LayerId,
    /// The dataset itself.
    layer: LocalVectorLayer,
    /// The GPU mesh, if one has been built and is still valid.
    resident: Option<Resident>,
    /// Where the layer landed in the last [`LocalVectorRenderer::prepare`],
    /// [`None`] when it was hidden or culled.
    placement: Option<TilePlacement>,
}

/// An ordered stack of local vector layers and the pipeline that draws them.
///
/// Holds no GPU resource until the first [`LocalVectorRenderer::prepare`] with a
/// non-empty stack, which is what keeps the raster-only and tiled-vector paths
/// untouched when no local layer has been added.
#[derive(Debug)]
pub struct LocalVectorRenderer {
    /// Colour target the pipeline is (or will be) built for.
    target_format: wgpu::TextureFormat,
    /// Built lazily; shared by every layer in the stack.
    pipeline: Option<VectorPipeline>,
    /// The stack, index `0` painting first.
    entries: Vec<LocalEntry>,
    /// Indices into `entries` with a drawable mesh this frame, in draw order.
    /// Position in this list *is* the instance index.
    draws: Vec<usize>,
    /// Indices into `entries` that are on screen this frame, drawable or not —
    /// the label pass runs over these.
    visible: Vec<usize>,
    /// Set once a tessellation has failed, so the log records it one time.
    tessellation_failed: bool,
    /// Set once a heavily classified layer has been reported, so the notice
    /// costs one log line rather than one per frame of a zoom.
    class_count_reported: bool,
    /// Where the next frame's tessellation budget starts looking, so the
    /// budget rotates through the stack instead of being eaten by the same
    /// low-index layer on every frame of a continuous zoom (which would leave
    /// every layer above it permanently stale — or, for a fresh stack,
    /// permanently invisible).
    tessellation_cursor: usize,
    /// Layers the last [`LocalVectorRenderer::prepare`] wanted to tessellate
    /// and could not fit in its budget. Non-zero means the frame is not
    /// finished and another one is owed — see [`Self::tessellation_backlog`].
    tessellation_backlog: usize,
}

impl LocalVectorRenderer {
    /// Creates an empty stack for a colour target of `target_format`.
    #[must_use]
    pub fn new(target_format: wgpu::TextureFormat) -> Self {
        Self {
            target_format,
            pipeline: None,
            entries: Vec::new(),
            draws: Vec::new(),
            visible: Vec::new(),
            tessellation_failed: false,
            class_count_reported: false,
            tessellation_cursor: 0,
            tessellation_backlog: 0,
        }
    }

    /// Number of layers in the stack.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the stack holds no layer at all — the state in which this
    /// renderer costs one branch per frame and nothing else.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `id` is in the stack.
    #[must_use]
    pub fn contains(&self, id: LayerId) -> bool {
        self.index_of(id).is_some()
    }

    /// The layer ids, in draw order (first painted first).
    #[must_use]
    pub fn layer_ids(&self) -> Vec<LayerId> {
        self.entries.iter().map(|entry| entry.id).collect()
    }

    /// The layer registered under `id`.
    #[must_use]
    pub fn get(&self, id: LayerId) -> Option<&LocalVectorLayer> {
        self.index_of(id).map(|index| &self.entries[index].layer)
    }

    /// The layer registered under `id`, mutably.
    ///
    /// Safe to use for restyling: [`LocalVectorLayer::set_style`] bumps the
    /// layer's generation and the next [`LocalVectorRenderer::prepare`] rebuilds
    /// the mesh because of it.
    pub fn get_mut(&mut self, id: LayerId) -> Option<&mut LocalVectorLayer> {
        self.index_of(id)
            .map(|index| &mut self.entries[index].layer)
    }

    /// Adds `layer` on top of the stack, or replaces the one already registered
    /// under `id` in place (keeping its position).
    ///
    /// Returns `true` if the layer is new, `false` if it replaced one.
    pub fn insert(&mut self, id: LayerId, layer: LocalVectorLayer) -> bool {
        match self.index_of(id) {
            Some(index) => {
                self.entries[index].layer = layer;
                self.entries[index].resident = None;
                self.entries[index].placement = None;
                false
            }
            None => {
                self.entries.push(LocalEntry {
                    id,
                    layer,
                    resident: None,
                    placement: None,
                });
                true
            }
        }
    }

    /// Removes and returns the layer registered under `id`.
    pub fn remove(&mut self, id: LayerId) -> Option<LocalVectorLayer> {
        let index = self.index_of(id)?;
        // The frame lists index into `entries`; removing shifts them, so they
        // are dropped rather than patched. `prepare` rebuilds both.
        self.draws.clear();
        self.visible.clear();
        Some(self.entries.remove(index).layer)
    }

    /// Empties the stack, dropping every mesh.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.draws.clear();
        self.visible.clear();
    }

    /// Drops every resident mesh, keeping the layers — after a device loss, or
    /// when a shell wants a forced rebuild.
    pub fn clear_meshes(&mut self) {
        for entry in &mut self.entries {
            entry.resident = None;
        }
        self.draws.clear();
    }

    /// Reorders the stack to match `order`.
    ///
    /// Ids `order` does not mention keep their relative order and are appended
    /// after the ones it does; ids that are not in the stack are ignored. That
    /// makes it safe to pass the application's whole layer list, local layers
    /// and tiled ones alike. Returns `true` if the order changed.
    pub fn reorder(&mut self, order: &[LayerId]) -> bool {
        let before: Vec<LayerId> = self.layer_ids();
        let mut taken: Vec<LocalEntry> = Vec::with_capacity(self.entries.len());
        let mut rest = core::mem::take(&mut self.entries);
        for id in order {
            if let Some(position) = rest.iter().position(|entry| &entry.id == id) {
                taken.push(rest.remove(position));
            }
        }
        taken.append(&mut rest);
        self.entries = taken;
        let changed = before != self.layer_ids();
        if changed {
            self.draws.clear();
            self.visible.clear();
        }
        changed
    }

    /// Position of `id` in the stack.
    fn index_of(&self, id: LayerId) -> Option<usize> {
        self.entries.iter().position(|entry| entry.id == id)
    }

    /// Places, tessellates and uploads every visible layer for `view`.
    ///
    /// A no-op — down to not building a pipeline — while the stack is empty.
    /// A layer whose tessellation fails is skipped for the frame and logged
    /// once; the rest of the map keeps drawing.
    ///
    /// At most [`MAX_TESSELLATIONS_PER_FRAME`] layers are (re-)tessellated per
    /// call, rotating through the stack so every layer gets its turn; the rest
    /// keep their previous mesh and are counted in
    /// [`Self::tessellation_backlog`], which a shell turns into
    /// another frame. Placement, culling and drawing are NOT budgeted — every
    /// layer with a mesh draws every frame regardless of whose turn it is.
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Gpu`] from mesh validation and the instance
    /// buffer, and [`RenderError::InvalidViewport`] from placement conversion.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: MapView,
    ) -> Result<(), RenderError> {
        self.draws.clear();
        self.visible.clear();
        self.tessellation_backlog = 0;
        if self.entries.is_empty() {
            return Ok(());
        }
        if self.pipeline.is_none() {
            self.pipeline = Some(VectorPipeline::new(device, self.target_format)?);
        }
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Err(RenderError::NotPrepared);
        };

        let size_px = view.size_px();
        let mut draws = Vec::with_capacity(self.entries.len());
        let mut visible = Vec::with_capacity(self.entries.len());
        let mut instances = Vec::with_capacity(self.entries.len());
        let mut candidates = Vec::new();
        let mut failed = false;

        // Pass 1: place and cull. The tessellation decision needs the whole
        // frame's candidate list before it can be fair, so it cannot be taken
        // inside this loop.
        for (index, entry) in self.entries.iter_mut().enumerate() {
            entry.placement = None;
            if !entry.layer.visible() {
                continue;
            }
            let placement = entry.layer.place(view);
            if !placement_is_usable(&placement, size_px) {
                continue;
            }
            entry.placement = Some(placement);
            visible.push(index);
            if needs_tessellation(entry, placement.size) {
                candidates.push(index);
            }
        }

        // Pass 2: spend the frame's budget, oldest turn first.
        let (picked, cursor) = schedule_tessellations(
            &candidates,
            self.tessellation_cursor,
            MAX_TESSELLATIONS_PER_FRAME,
        );
        self.tessellation_cursor = cursor;
        self.tessellation_backlog = candidates.len().saturating_sub(picked.len());
        for index in picked {
            let Some(entry) = self.entries.get_mut(index) else {
                continue;
            };
            let Some(placement) = entry.placement else {
                continue;
            };
            // One line, once, when a classified layer is heavy enough for its
            // cost to be felt — see `CLASS_COUNT_NOTICE_THRESHOLD`.
            if !self.class_count_reported
                && entry.layer.class_count() >= CLASS_COUNT_NOTICE_THRESHOLD
            {
                self.class_count_reported = true;
                tracing::info!(
                    layer = entry.layer.name(),
                    classes = entry.layer.class_count(),
                    "oxigis-ui: a local layer draws many style classes; each is a mesh of its own",
                );
            }
            match entry.layer.tessellate(placement.size) {
                Ok(mesh) => {
                    let gpu = if mesh.is_empty() {
                        None
                    } else {
                        Some(pipeline.upload_mesh(device, queue, &mesh)?)
                    };
                    entry.resident = Some(Resident {
                        gpu,
                        baked_size_px: placement.size,
                        generation: entry.layer.generation(),
                    });
                }
                Err(error) => {
                    failed = true;
                    if !self.tessellation_failed {
                        tracing::error!(
                            %error,
                            layer = entry.layer.name(),
                            "oxigis-ui: local layer tessellation failed",
                        );
                    }
                    // Resident as *nothing*, so a failure costs one attempt
                    // rather than a full-dataset tessellation on the render
                    // thread every frame. A restyle or a √2 zoom step is
                    // what retries it — the same policy an empty mesh gets.
                    entry.resident = Some(Resident {
                        gpu: None,
                        baked_size_px: placement.size,
                        generation: entry.layer.generation(),
                    });
                }
            }
        }

        // Pass 3: draw whatever has a mesh, budgeted or not — a layer waiting
        // its turn keeps drawing the mesh it already has.
        for &index in &visible {
            let Some(entry) = self.entries.get(index) else {
                continue;
            };
            let Some(placement) = entry.placement else {
                continue;
            };
            if entry
                .resident
                .as_ref()
                .is_some_and(|resident| resident.gpu.is_some())
            {
                instances.push(TileInstance::from_placement(
                    &placement,
                    size_px,
                    entry.layer.tint(),
                )?);
                draws.push(index);
            }
        }

        self.tessellation_failed = failed;
        self.draws = draws;
        self.visible = visible;

        let Some(pipeline) = self.pipeline.as_mut() else {
            return Err(RenderError::NotPrepared);
        };
        pipeline.upload_instances(device, queue, &instances)
    }

    /// How many layers the last [`Self::prepare`] wanted to tessellate and
    /// deferred to a later frame.
    ///
    /// Non-zero means the map is drawing a stale (or missing) mesh for that
    /// many layers and another frame is owed — a shell must call
    /// [`egui::Context::request_repaint`] rather than let the UI go idle, or
    /// the backlog would only drain on the next unrelated input event.
    #[must_use]
    pub fn tessellation_backlog(&self) -> usize {
        self.tessellation_backlog
    }

    /// Draws every layer the last [`LocalVectorRenderer::prepare`] accepted.
    ///
    /// `clip_origin_px` is the pass viewport's top-left corner in the
    /// framebuffer; the clip *size* is the view's own
    /// [`MapView::size_px`], as everywhere else in the frame.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Gpu`] if a mesh was dropped between `prepare` and
    /// here, which would mean the stack was mutated inside the frame.
    pub fn paint(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        clip_origin_px: [f32; 2],
        view: MapView,
    ) -> Result<(), RenderError> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(());
        };
        if self.draws.is_empty() {
            return Ok(());
        }
        let scissor = clip_scissor(clip_origin_px, view.size_px());
        let mut draws = Vec::with_capacity(self.draws.len());
        for (instance, &index) in self.draws.iter().enumerate() {
            let Some(mesh) = self
                .entries
                .get(index)
                .and_then(|entry| entry.resident.as_ref())
                .and_then(|resident| resident.gpu.as_ref())
            else {
                return Err(RenderError::Gpu(
                    "a local vector mesh was dropped between prepare and paint".to_owned(),
                ));
            };
            let Ok(instance) = u32::try_from(instance) else {
                return Err(RenderError::Gpu(
                    "the local vector draw list exceeds the addressable instance range".to_owned(),
                ));
            };
            draws.push(VectorDraw {
                mesh,
                instance,
                scissor,
            });
        }
        pipeline.draw(render_pass, &draws, scissor)
    }

    /// What the label pass should place this frame, in draw order.
    ///
    /// Only layers that are visible, on screen and carry at least one symbol
    /// rule are reported, so a stack of unlabelled layers costs the label pass
    /// nothing.
    #[must_use]
    pub fn label_jobs(&self) -> Vec<LocalLabelJob<'_>> {
        self.visible
            .iter()
            .filter_map(|&index| {
                let entry = self.entries.get(index)?;
                let placement = entry.placement?;
                if entry.layer.labels().is_empty() {
                    return None;
                }
                Some(LocalLabelJob {
                    tile: entry.layer.tile(),
                    placement,
                    labels: entry.layer.labels(),
                })
            })
            .collect()
    }

    /// Number of layers the last [`LocalVectorRenderer::prepare`] found on
    /// screen — what a status bar would show.
    #[must_use]
    pub fn visible_count(&self) -> usize {
        self.visible.len()
    }

    /// Number of layers the last [`LocalVectorRenderer::prepare`] uploaded a
    /// drawable mesh for.
    #[must_use]
    pub fn draw_count(&self) -> usize {
        self.draws.len()
    }
}

/// Whether a placement is finite, positive and close enough to the viewport to
/// be worth drawing.
#[must_use]
fn placement_is_usable(placement: &TilePlacement, size_px: [f32; 2]) -> bool {
    if !placement.x.is_finite()
        || !placement.y.is_finite()
        || !placement.size.is_finite()
        || placement.size <= 0.0
    {
        return false;
    }
    placement.x < size_px[0] + CULL_MARGIN_PX
        && placement.y < size_px[1] + CULL_MARGIN_PX
        && placement.x + placement.size > -CULL_MARGIN_PX
        && placement.y + placement.size > -CULL_MARGIN_PX
}

/// Which of this frame's tessellation `candidates` (entry indices, ascending)
/// fit in `budget`, and where the next frame must resume.
///
/// Round-robin rather than "the first `budget` of them": a continuous zoom
/// makes every layer a candidate on every frame, so a fixed scan from index `0`
/// would spend the whole budget on the same low layer for ever and leave the
/// ones above it permanently stale. Resuming at the entry after the last one
/// served gives every layer its turn in at most `candidates.len()` frames.
///
/// The returned cursor is an ENTRY index, not a position in `candidates`: the
/// stack can be reordered or shortened between frames, and an entry index
/// degrades into "start somewhere near there" instead of into a wrong pick.
#[must_use]
fn schedule_tessellations(
    candidates: &[usize],
    cursor: usize,
    budget: usize,
) -> (Vec<usize>, usize) {
    if candidates.is_empty() || budget == 0 {
        return (Vec::new(), cursor);
    }
    let start = candidates
        .iter()
        .position(|index| *index >= cursor)
        .unwrap_or(0);
    let take = budget.min(candidates.len());
    let mut picked = Vec::with_capacity(take);
    for step in 0..take {
        let Some(index) = candidates.get((start + step) % candidates.len()) else {
            break;
        };
        picked.push(*index);
    }
    // One past the last served, so the next frame starts with the layer that
    // waited longest. Saturating rather than wrapping: `usize::MAX` as a cursor
    // simply means "start from the front again", which the `unwrap_or(0)` above
    // already does.
    let next = picked
        .last()
        .map_or(cursor, |index| index.saturating_add(1));
    (picked, next)
}

/// Whether the entry's resident mesh is missing, built from an older style, or
/// baked at an on-screen size too far from the current one.
#[must_use]
fn needs_tessellation(entry: &LocalEntry, placement_size_px: f32) -> bool {
    let Some(resident) = entry.resident.as_ref() else {
        return true;
    };
    if resident.generation != entry.layer.generation() {
        return true;
    }
    if !resident.baked_size_px.is_finite() || resident.baked_size_px <= 0.0 {
        return true;
    }
    let ratio = placement_size_px / resident.baked_size_px;
    !ratio.is_finite() || !(1.0 / RETESSELLATE_RATIO..=RETESSELLATE_RATIO).contains(&ratio)
}

#[cfg(test)]
mod tests {
    // Everything that touches a `wgpu::Device` (`prepare`, `paint`) is exercised
    // by the shells; what is testable headlessly is the stack bookkeeping, the
    // cull and the re-tessellation rule.
    use super::{
        CLASS_COUNT_NOTICE_THRESHOLD, CULL_MARGIN_PX, LocalEntry, LocalVectorRenderer,
        MAX_TESSELLATIONS_PER_FRAME, Resident, needs_tessellation, placement_is_usable,
        schedule_tessellations,
    };
    use crate::local_vector::{LocalVectorLayer, local_symbol_style};
    use oxigis_core::{AttrValue, LayerId, MAX_STYLE_CLASSES};
    use oxigis_render::{TileId, TilePlacement};

    fn points_geojson() -> &'static str {
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"name":"a"},
             "geometry":{"type":"Point","coordinates":[139.7,35.7]}},
            {"type":"Feature","properties":{"name":"b"},
             "geometry":{"type":"Point","coordinates":[139.8,35.8]}}]}"#
    }

    fn layer(name: &str) -> LocalVectorLayer {
        match LocalVectorLayer::from_geojson(name, points_geojson()) {
            Ok(layer) => layer,
            Err(error) => panic!("the fixture must parse: {error}"),
        }
    }

    fn renderer() -> LocalVectorRenderer {
        LocalVectorRenderer::new(eframe::wgpu::TextureFormat::Rgba8UnormSrgb)
    }

    fn placement(x: f32, y: f32, size: f32) -> TilePlacement {
        TilePlacement {
            tile: TileId { z: 0, x: 0, y: 0 },
            x,
            y,
            size,
        }
    }

    fn entry(layer: LocalVectorLayer, resident: Option<Resident>) -> LocalEntry {
        LocalEntry {
            id: LayerId::new(),
            layer,
            resident,
            placement: None,
        }
    }

    fn resident(baked_size_px: f32, generation: u32) -> Resident {
        Resident {
            gpu: None,
            baked_size_px,
            generation,
        }
    }

    #[test]
    fn a_fresh_stack_is_empty_and_holds_no_pipeline() {
        let stack = renderer();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
        assert_eq!(stack.draw_count(), 0);
        assert!(stack.label_jobs().is_empty());
    }

    #[test]
    fn add_remove_round_trips_by_id() {
        let mut stack = renderer();
        let id = LayerId::new();
        assert!(stack.insert(id, layer("points")));
        assert!(stack.contains(id));
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.get(id).map(LocalVectorLayer::name), Some("points"));

        let removed = stack.remove(id).map(|layer| layer.name().to_owned());
        assert_eq!(removed.as_deref(), Some("points"));
        assert!(!stack.contains(id));
        assert!(stack.remove(id).is_none());
    }

    #[test]
    fn inserting_a_known_id_replaces_in_place() {
        let mut stack = renderer();
        let first = LayerId::new();
        let second = LayerId::new();
        stack.insert(first, layer("one"));
        stack.insert(second, layer("two"));
        assert!(!stack.insert(first, layer("one-again")));
        assert_eq!(stack.len(), 2);
        // Position kept: a restyled layer must not jump to the top.
        assert_eq!(stack.layer_ids(), vec![first, second]);
        assert_eq!(
            stack.get(first).map(LocalVectorLayer::name),
            Some("one-again")
        );
    }

    #[test]
    fn visibility_and_opacity_round_trip() {
        let mut stack = renderer();
        let id = LayerId::new();
        stack.insert(id, layer("points"));
        let Some(entry) = stack.get_mut(id) else {
            panic!("the layer must be present");
        };
        assert!(entry.visible());
        entry.set_visible(false);
        entry.set_opacity(0.25);
        assert_eq!(stack.get(id).map(LocalVectorLayer::visible), Some(false));
        assert_eq!(stack.get(id).map(LocalVectorLayer::opacity), Some(0.25));
        assert_eq!(
            stack.get(id).map(LocalVectorLayer::tint),
            Some([1.0, 1.0, 1.0, 0.25])
        );
    }

    #[test]
    fn opacity_is_clamped_and_non_finite_input_is_rejected() {
        let mut layer = layer("points");
        layer.set_opacity(2.0);
        assert_eq!(layer.opacity(), 1.0);
        layer.set_opacity(-1.0);
        assert_eq!(layer.opacity(), 0.0);
        layer.set_opacity(f32::NAN);
        assert_eq!(layer.opacity(), 1.0);
    }

    #[test]
    fn reorder_moves_known_ids_and_ignores_the_rest() {
        let mut stack = renderer();
        let (a, b, c) = (LayerId::new(), LayerId::new(), LayerId::new());
        stack.insert(a, layer("a"));
        stack.insert(b, layer("b"));
        stack.insert(c, layer("c"));
        assert_eq!(stack.layer_ids(), vec![a, b, c]);

        // An unrelated (tiled) layer id in the list is harmless.
        assert!(stack.reorder(&[c, LayerId::new(), a, b]));
        assert_eq!(stack.layer_ids(), vec![c, a, b]);
        // Idempotent, and reported as such.
        assert!(!stack.reorder(&[c, a, b]));

        // Ids left out keep their relative order, after the mentioned ones.
        assert!(stack.reorder(&[b]));
        assert_eq!(stack.layer_ids(), vec![b, c, a]);
    }

    #[test]
    fn clear_drops_every_layer() {
        let mut stack = renderer();
        stack.insert(LayerId::new(), layer("a"));
        stack.insert(LayerId::new(), layer("b"));
        stack.clear();
        assert!(stack.is_empty());
    }

    #[test]
    fn only_labelled_layers_become_label_jobs() {
        let mut stack = renderer();
        let id = LayerId::new();
        stack.insert(id, layer("points"));
        // Nothing has been prepared, so nothing is on screen yet.
        assert!(stack.label_jobs().is_empty());

        let Some(entry) = stack.get_mut(id) else {
            panic!("the layer must be present");
        };
        entry.set_style(local_symbol_style("name"));
        assert!(!entry.labels().is_empty(), "a symbol style must label");
    }

    #[test]
    fn a_placement_off_screen_beyond_the_margin_is_culled() {
        let size = [800.0, 600.0];
        assert!(placement_is_usable(&placement(0.0, 0.0, 100.0), size));
        assert!(placement_is_usable(&placement(-50.0, -50.0, 20.0), size));
        assert!(!placement_is_usable(
            &placement(-CULL_MARGIN_PX - 10.0, 0.0, 5.0),
            size
        ));
        assert!(!placement_is_usable(
            &placement(size[0] + CULL_MARGIN_PX + 1.0, 0.0, 5.0),
            size
        ));
    }

    #[test]
    fn a_degenerate_placement_is_never_drawn() {
        let size = [800.0, 600.0];
        assert!(!placement_is_usable(&placement(0.0, 0.0, 0.0), size));
        assert!(!placement_is_usable(&placement(0.0, 0.0, f32::NAN), size));
        assert!(!placement_is_usable(&placement(f32::NAN, 0.0, 10.0), size));
    }

    #[test]
    fn a_missing_mesh_is_always_tessellated() {
        let entry = entry(layer("points"), None);
        assert!(needs_tessellation(&entry, 512.0));
    }

    #[test]
    fn a_mesh_survives_a_small_zoom_change_and_not_a_large_one() {
        let entry = entry(layer("points"), Some(resident(512.0, 0)));
        assert!(!needs_tessellation(&entry, 512.0));
        assert!(!needs_tessellation(&entry, 700.0), "under √2");
        assert!(!needs_tessellation(&entry, 380.0), "over 1/√2");
        assert!(needs_tessellation(&entry, 1024.0), "a whole zoom step up");
        assert!(needs_tessellation(&entry, 256.0), "a whole zoom step down");
    }

    #[test]
    fn the_tessellation_budget_rotates_so_no_layer_starves() {
        // The zoom case: every layer wants a rebuild on every frame. With a
        // fixed scan the budget would go to layer 0 for ever.
        let candidates = [0, 1, 2, 3];
        let mut cursor = 0;
        let mut served = Vec::new();
        for _ in 0..candidates.len() {
            let (picked, next) = schedule_tessellations(&candidates, cursor, 1);
            served.extend(picked);
            cursor = next;
        }
        assert_eq!(served, vec![0, 1, 2, 3], "every layer gets its turn");
        // And it wraps rather than stopping at the top of the stack.
        let (picked, _next) = schedule_tessellations(&candidates, cursor, 1);
        assert_eq!(picked, vec![0]);
    }

    #[test]
    fn the_budget_caps_the_work_and_reports_what_it_deferred() {
        let candidates = [2, 5, 9];
        let (picked, next) = schedule_tessellations(&candidates, 0, MAX_TESSELLATIONS_PER_FRAME);
        assert_eq!(picked.len(), MAX_TESSELLATIONS_PER_FRAME);
        assert_eq!(picked, vec![2]);
        assert_eq!(
            next, 3,
            "the next frame resumes after the layer just served"
        );
        // A budget wider than the candidate list takes everything and no more.
        let (all, _next) = schedule_tessellations(&candidates, 0, 99);
        assert_eq!(all, vec![2, 5, 9]);
        // Degenerate inputs are answered, never panicked on.
        assert_eq!(schedule_tessellations(&[], 7, 4), (Vec::new(), 7));
        assert_eq!(schedule_tessellations(&candidates, 0, 0), (Vec::new(), 0));
        // A cursor past the end of a shortened stack starts over at the front.
        let (wrapped, _next) = schedule_tessellations(&candidates, usize::MAX, 1);
        assert_eq!(wrapped, vec![2]);
    }

    #[test]
    fn a_fresh_stack_reports_no_backlog() {
        let stack = renderer();
        assert_eq!(stack.tessellation_backlog(), 0);
    }

    #[test]
    fn a_restyled_layer_invalidates_its_mesh() {
        let mut layer = layer("points");
        let before = layer.generation();
        let fresh = entry(layer.clone(), Some(resident(512.0, before)));
        assert!(!needs_tessellation(&fresh, 512.0));

        layer.set_style(local_symbol_style("name"));
        assert_ne!(layer.generation(), before);
        let stale = entry(layer, Some(resident(512.0, before)));
        assert!(needs_tessellation(&stale, 512.0));
    }

    /// The points fixture, classified by its `name` attribute.
    fn classified(name: &str) -> LocalVectorLayer {
        let mut layer = layer(name);
        let base = layer.style().base().clone();
        let mut set = layer.style().clone();
        set.set_renderer(crate::local_vector::classify::categorized_renderer(
            &base,
            "name",
            [AttrValue::text("a"), AttrValue::text("b")],
        ));
        layer.set_style(set);
        layer
    }

    #[test]
    fn a_classified_layer_is_one_stack_entry_and_one_draw_call() {
        // The point of partitioning inside the tile rather than into several
        // stack entries: a classified layer is still ONE layer to the stack,
        // one instance, one indexed draw — the class split lives inside the
        // mesh, where it costs no per-class GPU state.
        let mut stack = renderer();
        let id = LayerId::new();
        stack.insert(id, classified("points"));
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.layer_ids(), vec![id]);
        let Some(entry) = stack.get(id) else {
            panic!("the layer must be present");
        };
        assert_eq!(entry.class_count(), 2);
        assert!(entry.tessellate(512.0).is_ok(), "one mesh, all classes");
    }

    #[test]
    fn a_reclassified_layer_invalidates_its_mesh_exactly_once() {
        // A class-list edit and a colour edit both bump the generation (both
        // change the picture), so both rebuild — but only the first moves the
        // partition, which is what `classification()` is for.
        let mut layer = classified("points");
        let before = layer.generation();
        let partition = layer.classification().clone();
        let fresh = entry(layer.clone(), Some(resident(512.0, before)));
        assert!(!needs_tessellation(&fresh, 512.0));

        // A colour edit: same partition, stale mesh.
        let mut set = layer.style().clone();
        match set.renderer_mut().class_style_mut(0) {
            Some(style) => {
                *style = oxigis_core::recolor_style(style, oxigis_core::Color::WHITE);
            }
            None => panic!("class 0 exists"),
        }
        layer.set_style(set);
        assert_eq!(layer.classification(), &partition);
        let stale = entry(layer.clone(), Some(resident(512.0, before)));
        assert!(needs_tessellation(&stale, 512.0), "the colour moved");

        // A class-list edit: the partition moves too.
        let mut set = layer.style().clone();
        set.renderer_mut().remove_class(1);
        layer.set_style(set);
        assert_ne!(layer.classification(), &partition);
        assert_eq!(layer.class_count(), 1);
    }

    #[test]
    fn the_class_notice_threshold_stays_under_the_models_cap() {
        // The threshold is a LOG line, not a limit: it must sit below the cap
        // the model enforces, or it would never fire.
        const { assert!(CLASS_COUNT_NOTICE_THRESHOLD < MAX_STYLE_CLASSES) };
        const { assert!(CLASS_COUNT_NOTICE_THRESHOLD > 0) };
        // A plain layer is nowhere near it, so the ordinary case logs nothing.
        assert!(layer("points").class_count() < CLASS_COUNT_NOTICE_THRESHOLD);
        assert!(classified("points").class_count() < CLASS_COUNT_NOTICE_THRESHOLD);
        // A fresh stack has not reported anything.
        assert!(!renderer().class_count_reported);
    }

    #[test]
    fn a_classified_layer_still_becomes_a_label_job_when_it_labels() {
        let mut stack = renderer();
        let id = LayerId::new();
        stack.insert(id, classified("points"));
        let Some(entry) = stack.get_mut(id) else {
            panic!("the layer must be present");
        };
        assert!(
            entry.labels().is_empty(),
            "a fill/circle style labels nothing"
        );

        // A SYMBOL base with symbol classes: every bucket labels.
        let symbol = local_symbol_style("name");
        let mut set = oxigis_core::LayerStyleSet::new(symbol.clone());
        set.set_renderer(crate::local_vector::classify::categorized_renderer(
            &symbol,
            "name",
            [AttrValue::text("a")],
        ));
        entry.set_style(set);
        assert!(
            !entry.labels().is_empty(),
            "a classified symbol layer must still place labels",
        );

        // And the composition rule holds here too: a symbol CLASS over a
        // circle-drawing family recolours the circle rather than swapping it
        // for a label the tessellator would draw as nothing.
        let circle =
            crate::local_vector::default_style_for_kind(crate::local_vector::GeometryKind::Point);
        let mut mixed = oxigis_core::LayerStyleSet::new(circle);
        mixed.set_renderer(crate::local_vector::classify::categorized_renderer(
            &symbol,
            "name",
            [AttrValue::text("a")],
        ));
        entry.set_style(mixed);
        assert!(
            entry.labels().is_empty(),
            "a symbol class over a circle base draws circles, not labels",
        );
    }
}
