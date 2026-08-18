//! GPU tile painting: bridges [`oxigis_render::MapRenderer`] into egui's own
//! `wgpu` render pass.
//!
//! # Why this lives here
//!
//! `oxigis-render` deliberately knows nothing about `egui` (see its crate
//! docs): it exposes a four-call frame protocol
//! (`begin_frame` / `accept_tile` / `prepare` / `paint`) shaped exactly like
//! [`CallbackTrait`], and *this* module is
//! the adapter that names both sides. The `egui_wgpu`/`wgpu` types are reached
//! through `eframe`'s re-exports (`eframe::egui_wgpu`, `eframe::wgpu`), so the
//! workspace keeps a single pinned copy of each (egui 0.35.0 / wgpu 29.0.4).
//!
//! # Ownership
//!
//! An `egui_wgpu` paint callback is a *value* that egui may keep until the
//! frame is rendered, so it cannot own the renderer. Instead a [`MapGpuState`]
//! (renderer + tile source) lives in `egui_wgpu`'s type-map,
//! `RenderState::renderer.write().callback_resources`, installed once by
//! [`install`]; the per-frame callback carries nothing but the
//! [`MapView`] to draw.
//!
//! ```text
//! startup (shell, first frame):  install(render_state, view, provider)
//! every frame (map panel):       painter.add(Shape::Callback(paint_callback(rect, view)))
//! egui, before its render pass:  CallbackTrait::prepare -> MapGpuState::run_frame
//! egui, inside its render pass:  CallbackTrait::paint   -> MapRenderer::paint
//! ```
//!
//! # Geometry contract
//!
//! `egui_wgpu` sets the render pass viewport to the callback's rect in
//! *physical pixels* before invoking `paint`, so NDC `-1..1` covers exactly
//! that rect. [`oxigis_render::TilePlacement::to_ndc_rect`] is evaluated
//! against [`MapView::size_px`], which means the map draws correctly if and
//! only if the view's `size_px` equals the callback rect's size in physical
//! pixels — which is what [`crate::map_view::MapPanelState::allocate`]
//! guarantees.
//!
//! # Tile source
//!
//! Tiles arrive through the [`TileProvider`] seam. Phase 0 ships
//! [`DebugCheckerboard`], a synthetic generator with no I/O, so the map is
//! visibly pannable/zoomable before any network code exists; a real
//! [`oxigis_render::TileFetch`]-backed provider replaces it with
//! [`replace_provider`] and nothing else changes.
//!
//! # The drawn stack: N tiled layers, interleaved
//!
//! [`replace_provider`] and [`replace_vector_source`] each fill **one** slot,
//! which is why a project with two COGs, or a COG plus a vector tileset plus a
//! hillshade, only ever showed one raster and one vector-tile layer. Beside
//! those slots a [`MapGpuState`] now holds a *stack*: one whole renderer per
//! visible tiled layer, painted bottom-up in the layer panel's own order, each
//! with its own tile cache and its own opacity tint.
//!
//! ```text
//! basemap  ->  stack entries, bottom-up  ->  legacy vector slot
//!          ->  local datasets            ->  labels
//! ```
//!
//! The seam is [`installed_tile_stack`] / [`install_tile_layer`] /
//! [`remove_tile_layer`] / [`reorder_tile_layers`], driven by
//! [`crate::OxigisApp::tile_stack_work`], plus the once-per-frame
//! [`sync_tile_layer_opacities`]. The stack is empty — and allocation-free —
//! until a shell installs an entry, so a shell that has not migrated draws
//! exactly the frame it drew before. Two bounds keep it from becoming a
//! performance cliff: [`crate::app::providers::MAX_DRAWN_TILE_LAYERS`] on the
//! count, and a device-memory budget divided across the renderers rather than
//! handed to each of them whole.

use std::sync::atomic::{AtomicBool, Ordering};

use eframe::egui_wgpu::{CallbackResources, CallbackTrait, RenderState, ScreenDescriptor};
use eframe::wgpu;
use egui::Rect;
use egui::epaint::{PaintCallback, PaintCallbackInfo};
use oxigis_core::{LayerId, LayerStyleSet};
use oxigis_render::{
    DEFAULT_MESH_BYTE_BUDGET, DEFAULT_MESH_CAPACITY, DEFAULT_TEXTURE_BYTES,
    DEFAULT_TEXTURE_CAPACITY, DecodedTile, MapRenderer, MapView, RenderError, TileId,
    VectorLayerRenderer,
};

use crate::app::providers::{MAX_DRAWN_TILE_LAYERS, TileLayerPlan};
use crate::label_frame::LabelFrame;
use crate::local_layers::LocalVectorRenderer;
use crate::local_vector::LocalVectorLayer;
use crate::vector_provider::BoxedVectorTileSource;

/// Supplies decoded pixels for a tile the current frame is missing.
///
/// Called from the `egui_wgpu` prepare hook, i.e. on the render thread with a
/// live GPU device, which is why it is synchronous and non-blocking by
/// contract: return [`None`] for "not available yet" and the renderer will ask
/// again next frame ([`oxigis_render::MapRenderer::missing_tiles`] is
/// recomputed every [`oxigis_render::MapRenderer::begin_frame`]).
///
/// That is exactly the shape a real fetcher needs: an implementation wrapping
/// an [`oxigis_render::TileFetch`] kicks off the request on the first `None`,
/// parks the bytes in its own map, and returns [`Some`] on a later frame.
pub trait TileProvider: 'static {
    /// Pixels for `tile`, or [`None`] if they are not ready (or never will be).
    fn tile(&self, tile: TileId) -> Option<DecodedTile>;
}

/// A [`TileProvider`] stored inside [`MapGpuState`].
///
/// `Send + Sync` is required **on every target**, including wasm:
/// `egui_wgpu::CallbackResources` is a `type_map::concurrent::TypeMap` unless
/// the `fragile-send-sync-non-atomic-wasm` feature is *off*, and that feature is
/// part of `egui-wgpu`'s default set (verified with `cargo tree -e features`),
/// which is what also makes `wgpu`'s browser handles nominally `Send + Sync`.
/// The bound lives on this alias rather than on [`TileProvider`] itself so that
/// an implementor is free to be thread-bound as long as it is not stored here.
pub type BoxedTileProvider = Box<dyn TileProvider + Send + Sync>;

/// Smallest synthetic tile edge [`DebugCheckerboard`] will generate, in pixels.
pub const MIN_TILE_EDGE_PX: u32 = 8;

/// Largest synthetic tile edge [`DebugCheckerboard`] will generate, in pixels.
///
/// Well below `oxigis_render`'s 8192-texel upload limit: these tiles are
/// generated on the render thread, so the cost is paid per frame that reveals
/// new tiles.
pub const MAX_TILE_EDGE_PX: u32 = 1_024;

/// Width of the border drawn around every synthetic tile, in pixels.
const BORDER_PX: u32 = 2;

/// Number of sub-cells per axis inside a synthetic tile.
const SUB_CELLS: u32 = 8;

/// Darker of the two synthetic checkerboard tones, RGBA.
const TONE_A: [u8; 4] = [0x24, 0x2c, 0x38, 0xff];

/// Lighter of the two synthetic checkerboard tones, RGBA.
const TONE_B: [u8; 4] = [0x3a, 0x46, 0x58, 0xff];

/// Border colour of a synthetic tile, RGBA — deliberately unlike either tone
/// so tile seams (and therefore placement bugs) are obvious.
const TONE_BORDER: [u8; 4] = [0xd0, 0x7a, 0x2c, 0xff];

/// Synthetic tile source: a deterministic two-tone checkerboard with a
/// contrasting border, and no I/O whatsoever.
///
/// The tone assignment is `(x + y + z) & 1`, so neighbouring tiles always
/// differ and a tile's appearance depends only on its [`TileId`] — a wrong
/// placement, a stale cache entry or an off-by-one in the pyramid shows up as a
/// visible discontinuity rather than as a plausible-looking map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DebugCheckerboard {
    /// Edge length of the generated tiles, in pixels.
    edge_px: u32,
}

impl DebugCheckerboard {
    /// Edge length used by [`DebugCheckerboard::default`], matching
    /// [`oxigis_render::TILE_SIZE_PX`].
    pub const DEFAULT_EDGE_PX: u32 = 256;

    /// Creates a generator producing `edge_px`-square tiles, clamped to
    /// [`MIN_TILE_EDGE_PX`]`..=`[`MAX_TILE_EDGE_PX`].
    #[must_use]
    pub fn new(edge_px: u32) -> Self {
        Self {
            edge_px: edge_px.clamp(MIN_TILE_EDGE_PX, MAX_TILE_EDGE_PX),
        }
    }

    /// Edge length of the generated tiles, in pixels.
    #[must_use]
    pub fn edge_px(self) -> u32 {
        self.edge_px
    }
}

impl Default for DebugCheckerboard {
    fn default() -> Self {
        Self::new(Self::DEFAULT_EDGE_PX)
    }
}

impl TileProvider for DebugCheckerboard {
    fn tile(&self, tile: TileId) -> Option<DecodedTile> {
        checkerboard_tile(tile, self.edge_px).ok()
    }
}

/// Builds one synthetic checkerboard tile.
///
/// `edge_px` is clamped to [`MIN_TILE_EDGE_PX`]`..=`[`MAX_TILE_EDGE_PX`], so
/// the only way this fails is a [`DecodedTile`] validation change.
///
/// # Errors
///
/// Propagates [`RenderError::InvalidTileImage`] from [`DecodedTile::new`].
pub fn checkerboard_tile(tile: TileId, edge_px: u32) -> Result<DecodedTile, RenderError> {
    let edge = edge_px.clamp(MIN_TILE_EDGE_PX, MAX_TILE_EDGE_PX);
    let parity = u32::from(tile.z).wrapping_add(tile.x).wrapping_add(tile.y) & 1;
    let (first, second) = if parity == 0 {
        (TONE_A, TONE_B)
    } else {
        (TONE_B, TONE_A)
    };
    let cell = (edge / SUB_CELLS).max(1);

    let mut rgba = Vec::with_capacity((edge as usize) * (edge as usize) * 4);
    for y in 0..edge {
        for x in 0..edge {
            let on_border =
                x < BORDER_PX || y < BORDER_PX || x + BORDER_PX >= edge || y + BORDER_PX >= edge;
            let color = if on_border {
                TONE_BORDER
            } else if ((x / cell) + (y / cell)) & 1 == 0 {
                first
            } else {
                second
            };
            rgba.extend_from_slice(&color);
        }
    }
    DecodedTile::new(edge, edge, rgba)
}

/// A source a shell built for one entry of the tiled-layer stack.
///
/// The kind must match the entry's [`crate::app::providers::TileLayerSource`]:
/// a raster plan needs a [`TileProvider`], a vector-tile plan a
/// [`crate::vector_provider::VectorTileSource`]. A mismatch is recorded as a
/// refusal rather than drawn as something else.
pub enum TileLayerGpuSource {
    /// Decoded pixels for a raster layer (COG, tile archive, XYZ overlay).
    Raster(BoxedTileProvider),
    /// Tessellated meshes for a streamed vector-tile layer.
    Vector(BoxedVectorTileSource),
}

impl core::fmt::Debug for TileLayerGpuSource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Raster(_) => f.write_str("TileLayerGpuSource::Raster"),
            Self::Vector(_) => f.write_str("TileLayerGpuSource::Vector"),
        }
    }
}

/// A raster stack entry's renderer and its source.
struct RasterEntry {
    /// The layer's tile renderer — its own cache, its own pipeline, its own
    /// tint, which is exactly what lets several raster layers draw at once.
    renderer: MapRenderer,
    /// Where its missing tiles come from.
    provider: BoxedTileProvider,
}

/// A vector-tile stack entry's renderer and its source.
struct VectorEntry {
    /// The layer's mesh renderer.
    renderer: VectorLayerRenderer,
    /// Where its missing meshes come from.
    source: BoxedVectorTileSource,
}

/// What one stack entry draws with — its own renderer and its own source, so a
/// layer's tiles, cache and fade belong to it and to nothing else.
///
/// Both payloads are boxed: a [`MapRenderer`] is several hundred bytes, and
/// unboxed it would make every vector entry (and every refusal) in the stack's
/// `Vec` as large as the largest raster one.
enum StackRenderer {
    /// A raster layer: a whole [`MapRenderer`] of its own.
    Raster(Box<RasterEntry>),
    /// A vector-tile layer: a whole [`VectorLayerRenderer`] of its own.
    Vector(Box<VectorEntry>),
    /// The shell could not build a source for this entry. The slot is still
    /// *occupied* — that is the point: the plan matches what is installed, so
    /// the reconciliation stops offering it once per frame for ever. The map
    /// draws everything else and [`retry_refused_tile_layers`] is the way back.
    Refused(String),
}

/// One entry of the drawn stack: what it is, what it draws with, how faded.
struct TileLayerEntry {
    /// Which project layer this is, and the source identity it was built from
    /// — compared by [`crate::OxigisApp::tile_stack_work`] to decide whether a
    /// rebuild is owed.
    plan: TileLayerPlan,
    /// The renderer and source, or the refusal that stands in for them.
    renderer: StackRenderer,
    /// Last opacity pushed into the renderer, so
    /// [`MapGpuState::sync_tile_layer_opacities`] can skip the unchanged ones.
    opacity: f32,
    /// Set once this entry's `prepare` has failed, so the log records it one
    /// time instead of once per frame.
    prepare_failed: bool,
    /// Same, for `paint` — which only has `&self`, hence the atomic.
    paint_failed: AtomicBool,
}

impl TileLayerEntry {
    /// Whether this entry draws through the raster pipeline.
    fn is_raster(&self) -> bool {
        matches!(self.renderer, StackRenderer::Raster(_))
    }

    /// Whether this entry draws through the vector-tile pipeline.
    fn is_vector(&self) -> bool {
        matches!(self.renderer, StackRenderer::Vector(_))
    }
}

/// Splits a device-memory budget between `ways` renderers.
///
/// Never zero: `MapRenderer::set_texture_byte_budget(Some(0))` is an error, and
/// a share of zero would evict every tile the moment it was uploaded.
fn split_budget(total: usize, ways: usize) -> usize {
    (total / ways.max(1)).max(1)
}

/// [`split_budget`] for the mesh budget, which counts in `u64`.
fn split_mesh_budget(total: u64, ways: usize) -> u64 {
    let ways = u64::try_from(ways.max(1)).unwrap_or(u64::MAX);
    (total / ways).max(1)
}

/// The permutation `order` implies over the stack currently holding `current`,
/// or [`None`] when the stack is already in that order.
///
/// `order` may name any layer at all: ids that are not in the stack are
/// ignored, and entries `order` leaves out keep their relative position at the
/// top — so a shell can hand over its whole layer list without filtering it
/// first, exactly as [`reorder_local_vector_layers`] allows.
fn reorder_plan(current: &[LayerId], order: &[LayerId]) -> Option<Vec<usize>> {
    let mut next: Vec<usize> = Vec::with_capacity(current.len());
    for id in order {
        if let Some(index) = current.iter().position(|entry| entry == id)
            && !next.contains(&index)
        {
            next.push(index);
        }
    }
    for index in 0..current.len() {
        if !next.contains(&index) {
            next.push(index);
        }
    }
    next.iter()
        .enumerate()
        .any(|(to, from)| to != *from)
        .then_some(next)
}

/// The map's GPU state, held in `egui_wgpu`'s callback resources: the tile
/// renderer plus the source its missing tiles are filled from.
///
/// Not constructed directly by shells — [`install`] puts it where the paint
/// callback can find it.
pub struct MapGpuState {
    /// The tile-map renderer borrowed by both frame hooks.
    renderer: MapRenderer,
    /// Where missing tiles come from.
    provider: BoxedTileProvider,
    /// The tiled layers composited over the basemap, **bottom-up**: index `0`
    /// paints first. Empty — and therefore allocation-free — until a shell
    /// installs one, which is what keeps the single-layer map exactly what it
    /// was.
    stack: Vec<TileLayerEntry>,
    /// The vector-tile renderer, present only once a vector layer has been
    /// configured. Painted *after* the raster pass, in the same callback.
    vectors: Option<VectorLayerRenderer>,
    /// Where missing vector meshes come from; always [`Some`] when `vectors` is.
    vector_source: Option<BoxedVectorTileSource>,
    /// Local (dropped / pasted) vector datasets, painted *over* the tiled
    /// vector layer. Empty until a shell adds one, and inert while it is.
    locals: LocalVectorRenderer,
    /// The label pass, present only once a shell has supplied font bytes
    /// through [`set_label_fonts`]. Painted *after* the vector meshes, so text
    /// is never covered by geometry.
    labels: Option<LabelFrame>,
    /// Set once a `prepare` has failed, so the log records it one time instead
    /// of once per frame.
    prepare_failed: bool,
    /// Same, for the vector renderer's `prepare`.
    vector_prepare_failed: bool,
    /// Same, for the local vector stack's `prepare`.
    local_prepare_failed: bool,
    /// Same, for `paint` — which only has `&self`, hence the atomic.
    paint_failed: AtomicBool,
    /// Same, for the vector renderer's `paint`.
    vector_paint_failed: AtomicBool,
    /// Same, for the local vector stack's `paint`.
    local_paint_failed: AtomicBool,
    /// Same, for the label pass's `draw`.
    label_paint_failed: AtomicBool,
}

impl MapGpuState {
    /// Wraps a renderer and a tile source.
    #[must_use]
    fn new(renderer: MapRenderer, provider: BoxedTileProvider) -> Self {
        let locals = LocalVectorRenderer::new(renderer.target_format());
        Self {
            renderer,
            provider,
            stack: Vec::new(),
            vectors: None,
            vector_source: None,
            locals,
            labels: None,
            prepare_failed: false,
            vector_prepare_failed: false,
            local_prepare_failed: false,
            paint_failed: AtomicBool::new(false),
            vector_paint_failed: AtomicBool::new(false),
            local_paint_failed: AtomicBool::new(false),
            label_paint_failed: AtomicBool::new(false),
        }
    }

    /// The tile renderer (e.g. to read `cache_stats` for a status bar).
    #[must_use]
    pub fn renderer(&self) -> &MapRenderer {
        &self.renderer
    }

    /// The tile renderer, mutably (e.g. `set_tint`, `clear_textures`).
    pub fn renderer_mut(&mut self) -> &mut MapRenderer {
        &mut self.renderer
    }

    /// Swaps the tile source, dropping the previous one.
    ///
    /// Resident textures are deliberately cleared, since they came from the old
    /// source and would otherwise linger for as long as they stay visible.
    pub fn set_provider(&mut self, provider: BoxedTileProvider) {
        self.provider = provider;
        self.renderer.clear_textures();
    }

    /// What the drawn stack currently holds, bottom-up — the mirror
    /// [`crate::OxigisApp::tile_stack_work`] diffs the project against.
    ///
    /// Read back from the map itself rather than kept beside it: a second copy
    /// of "what is installed" is a second thing that can be wrong, and this one
    /// cannot disagree with what is on screen.
    #[must_use]
    pub fn installed_tile_stack(&self) -> Vec<TileLayerPlan> {
        self.stack.iter().map(|entry| entry.plan.clone()).collect()
    }

    /// Number of tiled layers currently composited over the basemap, refused
    /// entries included.
    #[must_use]
    pub fn tile_layer_count(&self) -> usize {
        self.stack.len()
    }

    /// Installs (or replaces, in place) the stack entry for `plan.layer`.
    ///
    /// `source` is what the shell managed to build: `Err(reason)` records a
    /// **refusal** — the slot is occupied by the plan so the reconciliation
    /// stops offering it, the rest of the map keeps drawing, and
    /// [`MapGpuState::tile_layer_refusals`] reports why.
    ///
    /// Replacing keeps the entry's position in the stack: that is the "the
    /// layer's URL changed" path, not a reorder.
    ///
    /// # Errors
    ///
    /// Returns the refusal reason: the shell's own `Err`, a kind mismatch
    /// between `plan` and `source`, a renderer that could not be constructed, or
    /// [`MAX_DRAWN_TILE_LAYERS`] being full. The entry is recorded as refused in
    /// every case, so the caller may simply put the string on a status line.
    pub fn install_tile_layer(
        &mut self,
        plan: TileLayerPlan,
        source: Result<TileLayerGpuSource, String>,
    ) -> Result<(), String> {
        let existing = self
            .stack
            .iter()
            .position(|entry| entry.plan.layer == plan.layer);
        if existing.is_none() && self.stack.len() >= MAX_DRAWN_TILE_LAYERS {
            return Err(format!(
                "the map already composites {MAX_DRAWN_TILE_LAYERS} tiled layers"
            ));
        }
        let built = source.and_then(|source| self.build_stack_renderer(&plan, source));
        let (renderer, outcome) = match built {
            Ok(renderer) => (renderer, Ok(())),
            Err(reason) => (StackRenderer::Refused(reason.clone()), Err(reason)),
        };
        let entry = TileLayerEntry {
            plan,
            renderer,
            // Fully opaque until the shell's per-frame opacity sync says
            // otherwise, which it does on the very next frame.
            opacity: 1.0,
            prepare_failed: false,
            paint_failed: AtomicBool::new(false),
        };
        match existing {
            Some(index) => self.stack[index] = entry,
            None => self.stack.push(entry),
        }
        self.rebalance_budgets();
        outcome
    }

    /// Builds the renderer one stack entry needs, refusing a source whose kind
    /// does not match the plan.
    fn build_stack_renderer(
        &self,
        plan: &TileLayerPlan,
        source: TileLayerGpuSource,
    ) -> Result<StackRenderer, String> {
        let view = self.renderer.view();
        let format = self.renderer.target_format();
        match (plan.source.is_raster(), source) {
            (true, TileLayerGpuSource::Raster(provider)) => {
                let renderer = MapRenderer::new(view, DEFAULT_TEXTURE_CAPACITY, format)
                    .map_err(|error| error.to_string())?;
                Ok(StackRenderer::Raster(Box::new(RasterEntry {
                    renderer,
                    provider,
                })))
            }
            (false, TileLayerGpuSource::Vector(source)) => {
                let renderer = VectorLayerRenderer::new(view, DEFAULT_MESH_CAPACITY, format)
                    .map_err(|error| error.to_string())?;
                Ok(StackRenderer::Vector(Box::new(VectorEntry {
                    renderer,
                    source,
                })))
            }
            (true, TileLayerGpuSource::Vector(_)) => {
                Err("a raster layer was handed a vector-tile source".to_owned())
            }
            (false, TileLayerGpuSource::Raster(_)) => {
                Err("a vector-tile layer was handed a raster provider".to_owned())
            }
        }
    }

    /// Drops the stack entry for `layer`, returning whether one was there.
    pub fn remove_tile_layer(&mut self, layer: LayerId) -> bool {
        let Some(index) = self
            .stack
            .iter()
            .position(|entry| entry.plan.layer == layer)
        else {
            return false;
        };
        self.stack.remove(index);
        self.rebalance_budgets();
        true
    }

    /// Drops every stack entry `layers` names, in one pass — what a
    /// [`crate::app::providers::TileStackWork::Remove`] settles.
    ///
    /// Batched because a closed project's whole stack has to stop drawing in
    /// ONE frame: removing one per frame would show the previous project's
    /// layers vanishing one by one. Returns how many entries went.
    pub fn remove_tile_layers(&mut self, layers: &[LayerId]) -> usize {
        let before = self.stack.len();
        self.stack
            .retain(|entry| !layers.contains(&entry.plan.layer));
        let removed = before - self.stack.len();
        if removed > 0 {
            self.rebalance_budgets();
        }
        removed
    }

    /// Reorders the stack to `order` (bottom-up), returning whether anything
    /// moved.
    ///
    /// Rebuilds nothing and drops no tile: a drag in the layer panel is a
    /// permutation of renderers, not a re-fetch.
    pub fn reorder_tile_layers(&mut self, order: &[LayerId]) -> bool {
        let current: Vec<LayerId> = self.stack.iter().map(|entry| entry.plan.layer).collect();
        let Some(permutation) = reorder_plan(&current, order) else {
            return false;
        };
        let mut taken: Vec<Option<TileLayerEntry>> = self.stack.drain(..).map(Some).collect();
        let mut rebuilt = Vec::with_capacity(taken.len());
        for index in permutation {
            if let Some(entry) = taken.get_mut(index).and_then(Option::take) {
                rebuilt.push(entry);
            }
        }
        self.stack = rebuilt;
        true
    }

    /// Fades one stack entry to `opacity` (`0..=1`, clamped).
    ///
    /// **Cheap by construction**: the value is an instance tint the next
    /// `prepare` rewrites anyway, so no texture, mesh or fetch is touched. That
    /// is what makes the layer panel's slider honest for a COG, an archive and
    /// an MVT source instead of a value written to disk that nothing reads.
    ///
    /// Returns whether `layer` names an entry.
    pub fn set_tile_layer_opacity(&mut self, layer: LayerId, opacity: f32) -> bool {
        let Some(entry) = self
            .stack
            .iter_mut()
            .find(|entry| entry.plan.layer == layer)
        else {
            return false;
        };
        Self::fade_entry(entry, opacity);
        true
    }

    /// Pushes the current opacity of every stack entry, asking `opacity_of` for
    /// each — the per-frame call, made unconditional because a tint is free.
    ///
    /// Allocation-free and lock-free beyond the one the caller already holds:
    /// `opacity_of` is asked per entry rather than being handed a list to
    /// build. Returns whether anything actually moved.
    pub fn sync_tile_layer_opacities(&mut self, opacity_of: impl Fn(LayerId) -> f32) -> bool {
        let mut changed = false;
        for entry in &mut self.stack {
            let wanted = opacity_of(entry.plan.layer);
            changed |= Self::fade_entry(entry, wanted);
        }
        changed
    }

    /// Applies one entry's opacity, reporting whether it moved.
    fn fade_entry(entry: &mut TileLayerEntry, opacity: f32) -> bool {
        let alpha = oxigis_render::opacity_tint(opacity)[3];
        // Compared against the NORMALISED value, so a slider parked at 1.5 (or
        // at NaN) does not report a change every single frame.
        if entry.opacity == alpha {
            return false;
        }
        entry.opacity = alpha;
        match &mut entry.renderer {
            StackRenderer::Raster(raster) => raster.renderer.set_opacity(alpha),
            StackRenderer::Vector(vector) => vector.renderer.set_opacity(alpha),
            StackRenderer::Refused(_) => {}
        }
        true
    }

    /// The opacity one stack entry is currently drawn with, or [`None`] if
    /// `layer` names no entry.
    #[must_use]
    pub fn tile_layer_opacity(&self, layer: LayerId) -> Option<f32> {
        self.stack
            .iter()
            .find(|entry| entry.plan.layer == layer)
            .map(|entry| entry.opacity)
    }

    /// Restricts one raster entry to the zooms its source can serve, so an
    /// archive that stops at zoom 12 draws magnified z12 tiles instead of
    /// nothing (see [`MapRenderer::set_source_zoom_range`]).
    ///
    /// Returns whether `layer` names a raster entry.
    pub fn set_tile_layer_zoom_range(&mut self, layer: LayerId, range: Option<(u8, u8)>) -> bool {
        let Some(entry) = self
            .stack
            .iter_mut()
            .find(|entry| entry.plan.layer == layer)
        else {
            return false;
        };
        match &mut entry.renderer {
            StackRenderer::Raster(raster) => {
                raster.renderer.set_source_zoom_range(range);
                true
            }
            StackRenderer::Vector(_) | StackRenderer::Refused(_) => false,
        }
    }

    /// Every stack entry the shell could not build a source for, with the
    /// reason it gave — ready to put in front of the user.
    #[must_use]
    pub fn tile_layer_refusals(&self) -> Vec<(LayerId, String)> {
        self.stack
            .iter()
            .filter_map(|entry| match &entry.renderer {
                StackRenderer::Refused(reason) => Some((entry.plan.layer, reason.clone())),
                StackRenderer::Raster(_) | StackRenderer::Vector(_) => None,
            })
            .collect()
    }

    /// Forgets every refused entry, so the reconciliation offers those plans
    /// again. Returns how many were dropped.
    ///
    /// A command, not an edit: the entries that are *drawing* are deliberately
    /// left alone, because dropping those would blank and re-fetch tiles that
    /// are working fine.
    pub fn retry_refused_tile_layers(&mut self) -> usize {
        let before = self.stack.len();
        self.stack
            .retain(|entry| !matches!(entry.renderer, StackRenderer::Refused(_)));
        let dropped = before - self.stack.len();
        if dropped > 0 {
            self.rebalance_budgets();
        }
        dropped
    }

    /// Removes every stack entry — the "close project" path.
    pub fn clear_tile_layers(&mut self) {
        if self.stack.is_empty() {
            return;
        }
        self.stack.clear();
        self.rebalance_budgets();
    }

    /// Divides the device-memory budgets across every renderer that holds
    /// tiles or meshes.
    ///
    /// Called on every stack mutation, because the alternative is arithmetic no
    /// one does: each [`MapRenderer`] defaults to
    /// [`DEFAULT_TEXTURE_BYTES`] (768 MiB) on its own, so eight of them plus
    /// the basemap would authorise ~7 GiB of VRAM. The *total* is what is
    /// bounded here, not the per-layer share, so a one-layer map keeps the
    /// whole budget and an eight-layer map gets ~85 MiB each.
    fn rebalance_budgets(&mut self) {
        let rasters = 1 + self.stack.iter().filter(|entry| entry.is_raster()).count();
        let share = split_budget(DEFAULT_TEXTURE_BYTES, rasters);
        // The budget setter only rejects `Some(0)`, which `split_budget` cannot
        // produce; the map keeps its previous budget if it ever did.
        let _ = self.renderer.set_texture_byte_budget(Some(share));
        for entry in &mut self.stack {
            if let StackRenderer::Raster(raster) = &mut entry.renderer {
                let _ = raster.renderer.set_texture_byte_budget(Some(share));
            }
        }

        let vectors = usize::from(self.vectors.is_some())
            + self.stack.iter().filter(|e| e.is_vector()).count();
        let mesh_share = split_mesh_budget(DEFAULT_MESH_BYTE_BUDGET, vectors);
        if let Some(renderer) = self.vectors.as_mut() {
            renderer.set_mesh_byte_budget(mesh_share);
        }
        for entry in &mut self.stack {
            if let StackRenderer::Vector(vector) = &mut entry.renderer {
                vector.renderer.set_mesh_byte_budget(mesh_share);
            }
        }
    }

    /// Attaches (or detaches, with [`None`]) the vector-tile layer drawn over
    /// the raster basemap.
    ///
    /// The [`oxigis_render::VectorLayerRenderer`] is built on the first attach
    /// and then reused, so swapping the source's style or URL only clears the
    /// meshes rather than rebuilding a pipeline. Passing [`None`] drops both.
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::InvalidCapacity`] from
    /// [`oxigis_render::VectorLayerRenderer::new`].
    pub fn set_vector_source(
        &mut self,
        source: Option<BoxedVectorTileSource>,
    ) -> Result<(), RenderError> {
        let Some(source) = source else {
            self.vectors = None;
            self.vector_source = None;
            self.rebalance_budgets();
            return Ok(());
        };
        match self.vectors.as_mut() {
            Some(vectors) => vectors.clear_meshes(),
            None => {
                self.vectors = Some(VectorLayerRenderer::new(
                    self.renderer.view(),
                    DEFAULT_MESH_CAPACITY,
                    self.renderer.target_format(),
                )?);
            }
        }
        self.vector_source = Some(source);
        self.vector_prepare_failed = false;
        self.vector_paint_failed.store(false, Ordering::Relaxed);
        // The legacy slot is one more mesh cache to divide the budget between.
        self.rebalance_budgets();
        Ok(())
    }

    /// Installs (or replaces) the fonts the label pass shapes with.
    ///
    /// `primary` is the face every label starts in — the shells hand over the
    /// bundled Noto Sans — and `fallbacks` are tried in order for clusters it
    /// maps to `.notdef`, which is how CJK text gets glyphs without embedding a
    /// 16 MB face in the binary.
    ///
    /// Nothing is drawn until a vector layer with symbol rules is also
    /// attached, and no GPU resource is created here, so a shell may call this
    /// on its first frame regardless of what else is configured.
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Text`] if `primary` is not a font `oxitext`
    /// can parse; the previous label pass, if any, is left untouched.
    pub fn set_label_fonts(
        &mut self,
        primary: Vec<u8>,
        fallbacks: Vec<Vec<u8>>,
    ) -> Result<(), RenderError> {
        self.labels = Some(LabelFrame::new(primary, fallbacks)?);
        self.label_paint_failed.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Appends one font to the label pass's fallback chain.
    ///
    /// Returns `false` if no fonts have been installed yet, in which case
    /// `font` is dropped: a fallback chain with no primary face shapes nothing.
    pub fn add_label_fallback_font(&mut self, font: Vec<u8>) -> bool {
        match self.labels.as_mut() {
            Some(labels) => {
                labels.add_fallback_font(font);
                true
            }
            None => false,
        }
    }

    /// Appends several fonts to the label pass's fallback chain in one
    /// invalidation.
    ///
    /// Returns `false` if no fonts have been installed yet, in which case the
    /// batch is dropped for the same reason as [`Self::add_label_fallback_font`].
    pub fn add_label_fallback_fonts(&mut self, fonts: Vec<Vec<u8>>) -> bool {
        match self.labels.as_mut() {
            Some(labels) => {
                labels.add_fallback_fonts(fonts);
                true
            }
            None => false,
        }
    }

    /// Replaces the label pass's BOLD face chain — the weight seam's twin of
    /// [`Self::set_label_fonts`] (print/text v1.4, D-W3).
    ///
    /// Returns `false` if no fonts have been installed yet, in which case the
    /// chain is dropped: a bold chain with no regular primary shapes nothing.
    pub fn set_label_bold_fonts(&mut self, fonts: Vec<Vec<u8>>) -> bool {
        match self.labels.as_mut() {
            Some(labels) => {
                labels.set_bold_fonts(fonts);
                true
            }
            None => false,
        }
    }

    /// The label pass, if fonts have been installed.
    #[must_use]
    pub fn labels(&self) -> Option<&LabelFrame> {
        self.labels.as_ref()
    }

    /// The vector-tile renderer, if a vector layer is configured.
    #[must_use]
    pub fn vectors(&self) -> Option<&VectorLayerRenderer> {
        self.vectors.as_ref()
    }

    /// The vector-tile renderer, mutably (e.g. `set_tint`, `clear_meshes`).
    pub fn vectors_mut(&mut self) -> Option<&mut VectorLayerRenderer> {
        self.vectors.as_mut()
    }

    /// The local vector stack, for status reporting.
    #[must_use]
    pub fn locals(&self) -> &LocalVectorRenderer {
        &self.locals
    }

    /// The local vector stack, mutably — the seam every
    /// `*_local_vector_layer*` entry point of this module goes through.
    pub fn locals_mut(&mut self) -> &mut LocalVectorRenderer {
        &mut self.locals
    }

    /// One frame of the local vector stack, run after the tiled one so local
    /// data paints over basemap geometry.
    fn run_local_frame(&mut self, view: MapView, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.locals.is_empty() {
            return;
        }
        match self.locals.prepare(device, queue, view) {
            Ok(()) => self.local_prepare_failed = false,
            Err(error) => {
                if !self.local_prepare_failed {
                    self.local_prepare_failed = true;
                    tracing::error!(%error, "oxigis-ui: local vector prepare failed");
                }
            }
        }
    }

    /// How many local layers the last local `prepare` deferred to a later
    /// frame (see [`LocalVectorRenderer::tessellation_backlog`]).
    ///
    /// The paint callback reads this to decide whether the frame owes another
    /// one; without the nudge the deferred work would only drain on the next
    /// unrelated input event.
    #[must_use]
    pub fn local_tessellation_backlog(&self) -> usize {
        self.locals.tessellation_backlog()
    }

    /// The single label pass of the frame, fed by both vector paths.
    ///
    /// Lifted out of [`MapGpuState::run_vector_frame`] so that a map with local
    /// layers but no vector-tile source still labels, and so that both sets of
    /// labels collide against each other inside one
    /// [`oxigis_render::LabelPlacer`]. With neither path present it returns
    /// before touching the device, which is what keeps the raster-only frame
    /// exactly what it was.
    fn run_label_frame(&mut self, view: MapView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let format = self.renderer.target_format();
        // A tiled frame whose `prepare` failed contributes nothing, exactly as
        // before this pass was lifted out of `run_vector_frame` (which returned
        // early on that error).
        // EVERY tiled vector layer, not just one: the stack can hold several,
        // and they all collide against the single `LabelPlacer` inside
        // `LabelFrame::run`. Two layers each placing against their own grid is
        // exactly how a cadastral label ends up overprinting a POI label.
        //
        // Collected TOP-MOST FIRST, because the placer is greedy: the layer the
        // user sees on top is the one whose label survives a collision — and,
        // for a shell with a single vector layer, that is the very layer that
        // used to be the only one to label at all, so labelling is unchanged.
        let mut tiled: Vec<crate::label_frame::TiledLabelInput<'_>> = Vec::new();
        // A tiled frame whose `prepare` failed contributes nothing, exactly as
        // before this pass was lifted out of `run_vector_frame` (which returned
        // early on that error).
        if let (Some(vectors), Some(source)) = (
            self.vectors
                .as_ref()
                .filter(|_| !self.vector_prepare_failed),
            self.vector_source.as_ref(),
        ) {
            tiled.push(crate::label_frame::TiledLabelInput {
                placements: vectors.placements(),
                source: &**source as &dyn crate::vector_provider::VectorTileSource,
            });
        }
        for entry in self.stack.iter().rev() {
            if let StackRenderer::Vector(vector) = &entry.renderer
                && !entry.prepare_failed
            {
                tiled.push(crate::label_frame::TiledLabelInput {
                    placements: vector.renderer.placements(),
                    source: &*vector.source as &dyn crate::vector_provider::VectorTileSource,
                });
            }
        }
        let jobs = self.locals.label_jobs();
        if tiled.is_empty() && jobs.is_empty() {
            return;
        }
        // `labels`, `vectors`, `vector_source` and `locals` are four distinct
        // fields, so the borrows are disjoint — which is also why the pass is a
        // free-standing `LabelFrame` method taking explicit arguments.
        if let Some(labels) = self.labels.as_mut() {
            labels.run(device, queue, format, &tiled, &jobs, view);
        }
    }

    /// One frame of the vector half of the protocol, run after the raster half.
    ///
    /// Mirrors [`MapGpuState::run_frame`] with one extra step: the source is
    /// asked first whether the camera invalidated the meshes it handed out
    /// (a zoom step re-bakes the tessellation parameters — see
    /// [`oxigis_render::TessParams`]), and if so the resident meshes are dropped
    /// before the frame starts. Labels are *not* placed here; see
    /// [`MapGpuState::run_label_frame`].
    fn run_vector_frame(&mut self, view: MapView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (Some(vectors), Some(source)) = (self.vectors.as_mut(), self.vector_source.as_ref())
        else {
            return;
        };
        if source.begin_frame(view) {
            vectors.clear_meshes();
        }
        let _ = vectors.begin_frame(view);
        // Copied out because accepting a mesh borrows the renderer mutably.
        let missing: Vec<TileId> = vectors.missing_tiles().to_vec();
        for tile in missing {
            if let Some(mesh) = source.mesh(tile) {
                vectors.accept_mesh(tile, mesh);
            }
        }
        if let Err(error) = vectors.prepare(device, queue) {
            if !self.vector_prepare_failed {
                self.vector_prepare_failed = true;
                tracing::error!(%error, "oxigis-ui: vector renderer prepare failed");
            }
            return;
        }
        self.vector_prepare_failed = false;
    }

    /// One frame of every stack entry, run after the basemap's and before the
    /// legacy vector slot's.
    ///
    /// Each entry runs the same four-call protocol against its *own* renderer
    /// and its own source, which is precisely why several raster layers can
    /// draw at once: the tiles of one never enter the cache of another. An
    /// entry whose `prepare` fails is logged once and skipped; the layers above
    /// and below it still draw.
    ///
    /// Returns immediately on an empty stack, so a map with no tiled layer pays
    /// one `is_empty`.
    fn run_stack_frame(&mut self, view: MapView, device: &wgpu::Device, queue: &wgpu::Queue) {
        for entry in &mut self.stack {
            // Destructured so the failure latch and the renderer are separate
            // borrows: they are different fields, and the compiler only knows
            // that if they are named apart.
            let TileLayerEntry {
                plan,
                renderer,
                prepare_failed,
                ..
            } = entry;
            let result = match renderer {
                StackRenderer::Raster(raster) => {
                    let RasterEntry { renderer, provider } = raster.as_mut();
                    let _ = renderer.begin_frame(view);
                    // Copied out because filling a tile borrows the renderer
                    // mutably, exactly as in `run_frame`.
                    let missing: Vec<TileId> = renderer.missing_tiles().to_vec();
                    for tile in missing {
                        if let Some(decoded) = provider.tile(tile) {
                            renderer.accept_tile(tile, decoded);
                        }
                    }
                    renderer.prepare(device, queue)
                }
                StackRenderer::Vector(vector) => {
                    let VectorEntry { renderer, source } = vector.as_mut();
                    if source.begin_frame(view) {
                        renderer.clear_meshes();
                    }
                    let _ = renderer.begin_frame(view);
                    let missing: Vec<TileId> = renderer.missing_tiles().to_vec();
                    for tile in missing {
                        if let Some(mesh) = source.mesh(tile) {
                            renderer.accept_mesh(tile, mesh);
                        }
                    }
                    renderer.prepare(device, queue)
                }
                // A refused entry holds its slot and draws nothing; it must not
                // cost the frame a single call.
                StackRenderer::Refused(_) => continue,
            };
            match result {
                Ok(()) => *prepare_failed = false,
                Err(error) => {
                    if !*prepare_failed {
                        *prepare_failed = true;
                        let layer = plan.layer;
                        tracing::error!(
                            %error,
                            %layer,
                            "oxigis-ui: stacked tile layer prepare failed"
                        );
                    }
                }
            }
        }
    }

    /// Draws every stack entry, bottom-up.
    ///
    /// One entry's failure is latched and logged once and does not stop the
    /// entries above it: a COG whose pipeline is unhappy must not take the
    /// hillshade over it down with it.
    fn paint_stack(&self, render_pass: &mut wgpu::RenderPass<'static>, clip_origin_px: [f32; 2]) {
        for entry in &self.stack {
            let painted = match &entry.renderer {
                StackRenderer::Raster(raster) => raster.renderer.paint(render_pass),
                StackRenderer::Vector(vector) => vector.renderer.paint(render_pass, clip_origin_px),
                StackRenderer::Refused(_) => continue,
            };
            if let Err(error) = painted
                && !entry.paint_failed.swap(true, Ordering::Relaxed)
            {
                let layer = entry.plan.layer;
                tracing::error!(%error, %layer, "oxigis-ui: stacked tile layer paint failed");
            }
        }
    }

    /// One frame of the `oxigis-render` protocol: start the frame, fill what it
    /// is missing from the provider, upload.
    fn run_frame(&mut self, view: MapView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let _ = self.renderer.begin_frame(view);
        // Copied out because filling a tile borrows the renderer mutably.
        let missing: Vec<TileId> = self.renderer.missing_tiles().to_vec();
        for tile in missing {
            if let Some(decoded) = self.provider.tile(tile) {
                self.renderer.accept_tile(tile, decoded);
            }
        }
        if let Err(error) = self.renderer.prepare(device, queue) {
            if !self.prepare_failed {
                self.prepare_failed = true;
                tracing::error!(%error, "oxigis-ui: map renderer prepare failed");
            }
            return;
        }
        self.prepare_failed = false;
    }

    /// Issues the tile draws into egui's render pass, bottom-up:
    ///
    /// ```text
    /// basemap  ->  stack entries, in layer order  ->  legacy vector slot
    ///          ->  local datasets  ->  labels
    /// ```
    ///
    /// The stack is where several raster layers and several vector-tile layers
    /// **interleave**: a vector cadastre under a hillshade raster is a stack
    /// order the layer panel can express, so it is one the map draws. A shell
    /// uses the stack seam *or* the single-slot seams
    /// ([`replace_provider`] / [`replace_vector_source`]), never both for the
    /// same layer — the stack is empty until one is installed, which is what
    /// keeps the pre-stack frame byte-identical.
    ///
    /// `clip_origin_px` is the pass viewport's top-left corner in the
    /// framebuffer, which the vector pipeline needs to turn its per-tile scissor
    /// rectangles into framebuffer coordinates. The raster pipeline needs no
    /// such thing: it never scissors.
    fn paint_into(&self, render_pass: &mut wgpu::RenderPass<'static>, clip_origin_px: [f32; 2]) {
        if let Err(error) = self.renderer.paint(render_pass)
            && !self.paint_failed.swap(true, Ordering::Relaxed)
        {
            tracing::error!(%error, "oxigis-ui: map renderer paint failed");
        }
        self.paint_stack(render_pass, clip_origin_px);
        if let Some(vectors) = self.vectors.as_ref()
            && let Err(error) = vectors.paint(render_pass, clip_origin_px)
            && !self.vector_paint_failed.swap(true, Ordering::Relaxed)
        {
            tracing::error!(%error, "oxigis-ui: vector renderer paint failed");
        }
        // Local datasets over tiled geometry: what the user brought is never
        // hidden by the basemap.
        if let Err(error) = self
            .locals
            .paint(render_pass, clip_origin_px, self.renderer.view())
            && !self.local_paint_failed.swap(true, Ordering::Relaxed)
        {
            tracing::error!(%error, "oxigis-ui: local vector paint failed");
        }
        // Labels last of all, so text is never covered by the geometry it
        // names. The camera is the renderer's own, which `run_frame` set from
        // the very [`MapView`] the label pass placed against.
        if let Some(labels) = self.labels.as_ref()
            && let Err(error) = labels.draw(render_pass, clip_origin_px, self.renderer.view())
            && !self.label_paint_failed.swap(true, Ordering::Relaxed)
        {
            tracing::error!(%error, "oxigis-ui: label pass paint failed");
        }
    }
}

impl core::fmt::Debug for MapGpuState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MapGpuState")
            .field("renderer", &self.renderer)
            .field(
                "stack",
                &self
                    .stack
                    .iter()
                    .map(|entry| entry.plan.layer)
                    .collect::<Vec<_>>(),
            )
            .field("vectors", &self.vectors)
            .field("labels", &self.labels)
            .field("locals", &self.locals.len())
            .field("prepare_failed", &self.prepare_failed)
            .field("vector_prepare_failed", &self.vector_prepare_failed)
            .field("local_prepare_failed", &self.local_prepare_failed)
            .field("paint_failed", &self.paint_failed)
            .field("vector_paint_failed", &self.vector_paint_failed)
            .field("local_paint_failed", &self.local_paint_failed)
            .finish_non_exhaustive()
    }
}

/// Installs a [`MapGpuState`] into `render_state`'s callback resources.
///
/// Idempotent: returns `Ok(false)` (and keeps the existing state, textures and
/// provider) when one is already installed, so a shell may call it every frame
/// rather than tracking startup itself.
///
/// `view` only seeds the renderer; every frame's
/// [`paint_callback`] overrides it.
///
/// # Errors
///
/// Propagates [`RenderError::InvalidCapacity`] from
/// [`oxigis_render::MapRenderer::new`].
pub fn install(
    render_state: &RenderState,
    view: MapView,
    provider: BoxedTileProvider,
) -> Result<bool, RenderError> {
    let mut renderer = render_state.renderer.write();
    if renderer.callback_resources.get::<MapGpuState>().is_some() {
        return Ok(false);
    }
    let map = MapRenderer::new(view, DEFAULT_TEXTURE_CAPACITY, render_state.target_format)?;
    renderer
        .callback_resources
        .insert(MapGpuState::new(map, provider));
    Ok(true)
}

/// Whether a [`MapGpuState`] is installed in `render_state`.
#[must_use]
pub fn is_installed(render_state: &RenderState) -> bool {
    render_state
        .renderer
        .read()
        .callback_resources
        .get::<MapGpuState>()
        .is_some()
}

/// Replaces the installed tile source — the seam a real
/// [`oxigis_render::TileFetch`]-backed provider plugs into.
///
/// Returns `false` (and drops `provider`) if nothing is installed yet.
pub fn replace_provider(render_state: &RenderState, provider: BoxedTileProvider) -> bool {
    let mut renderer = render_state.renderer.write();
    match renderer.callback_resources.get_mut::<MapGpuState>() {
        Some(state) => {
            state.set_provider(provider);
            true
        }
        None => false,
    }
}

/// Attaches the vector-tile layer drawn over the raster basemap — the seam an
/// MVT source plugs into, and the vector counterpart of [`replace_provider`].
///
/// Returns `false` (and drops `source`) if no [`MapGpuState`] is installed, or
/// if the vector renderer could not be built; the failure is logged in the
/// latter case, and the raster map keeps working either way.
pub fn replace_vector_source(render_state: &RenderState, source: BoxedVectorTileSource) -> bool {
    let mut renderer = render_state.renderer.write();
    let Some(state) = renderer.callback_resources.get_mut::<MapGpuState>() else {
        return false;
    };
    match state.set_vector_source(Some(source)) {
        Ok(()) => true,
        Err(error) => {
            tracing::error!(%error, "oxigis-ui: could not attach the vector layer");
            false
        }
    }
}

/// What the drawn tiled-layer stack currently holds, bottom-up — the mirror a
/// shell hands to [`crate::OxigisApp::tile_stack_work`] each frame.
///
/// Empty when nothing is installed, which makes the first frame of a shell
/// offer the whole stack as work rather than silently drawing nothing.
///
/// # The shell's whole loop
///
/// ```text
/// let installed = map_gpu::installed_tile_stack(render_state);
/// if let Some(work) = app.tile_stack_work(&installed) {
///     match work {
///         TileStackWork::Install(plan) => {
///             let built = build_source_for(&plan);   // Result<TileLayerGpuSource, String>
///             if let Err(reason) = map_gpu::install_tile_layer(render_state, plan, built) {
///                 app.set_status(reason);
///             }
///         }
///         TileStackWork::Remove(layers) => { map_gpu::remove_tile_layers(render_state, &layers); }
///         TileStackWork::Reorder(order) => { map_gpu::reorder_tile_layers(render_state, &order); }
///     }
/// }
/// // Opacity is not part of the plan, so it is pushed every frame: it costs a
/// // tint and can therefore follow a slider without rebuilding anything.
/// map_gpu::sync_tile_layer_opacities(render_state, |id| app.tile_layer_opacity(id));
/// ```
#[must_use]
pub fn installed_tile_stack(render_state: &RenderState) -> Vec<TileLayerPlan> {
    with_state_ref(render_state, MapGpuState::installed_tile_stack).unwrap_or_default()
}

/// Installs (or replaces in place) one entry of the tiled-layer stack — the
/// N-layer counterpart of [`replace_provider`] and [`replace_vector_source`].
///
/// `source` is what the shell managed to build; `Err(reason)` records the
/// refusal against the entry so the plan is not offered again every frame.
///
/// # Errors
///
/// Returns the reason the entry is not drawing: the shell's own, a kind
/// mismatch, a renderer that could not be built, the stack being full, or the
/// GPU map not being attached yet (the normal state on a shell's first frame,
/// and the one case where nothing is recorded, so the work is offered again).
pub fn install_tile_layer(
    render_state: &RenderState,
    plan: TileLayerPlan,
    source: Result<TileLayerGpuSource, String>,
) -> Result<(), String> {
    let mut renderer = render_state.renderer.write();
    match renderer.callback_resources.get_mut::<MapGpuState>() {
        Some(state) => state.install_tile_layer(plan, source),
        None => Err("the GPU map is not attached".to_owned()),
    }
}

/// Drops one entry of the tiled-layer stack.
///
/// Returns `false` if nothing is installed or `layer` names no entry — the two
/// are indistinguishable on purpose, because both mean "there is nothing to
/// remove".
pub fn remove_tile_layer(render_state: &RenderState, layer: LayerId) -> bool {
    with_state(render_state, |state| state.remove_tile_layer(layer)).unwrap_or(false)
}

/// Drops every stack entry `layers` names in one pass — what a
/// [`crate::app::providers::TileStackWork::Remove`] settles, and the call that
/// makes a project close stop drawing the old stack in a single frame.
///
/// Returns how many entries went; `0` when nothing is installed.
pub fn remove_tile_layers(render_state: &RenderState, layers: &[LayerId]) -> usize {
    with_state(render_state, |state| state.remove_tile_layers(layers)).unwrap_or(0)
}

/// Reorders the tiled-layer stack to `order`, bottom-up.
///
/// Ids that name no entry are ignored and entries `order` leaves out keep their
/// relative position, so a shell may hand over its whole layer list. Rebuilds
/// nothing: a reorder must not re-fetch a tile.
///
/// Returns `false` if nothing is installed or the stack was already in that
/// order.
pub fn reorder_tile_layers(render_state: &RenderState, order: &[LayerId]) -> bool {
    with_state(render_state, |state| state.reorder_tile_layers(order)).unwrap_or(false)
}

/// Fades one stack entry to `opacity` (`0..=1`, clamped).
///
/// Cheap: opacity is an instance tint, so nothing is re-fetched or
/// re-tessellated. Returns `false` if nothing is installed or `layer` names no
/// entry.
pub fn set_tile_layer_opacity(render_state: &RenderState, layer: LayerId, opacity: f32) -> bool {
    with_state(render_state, |state| {
        state.set_tile_layer_opacity(layer, opacity)
    })
    .unwrap_or(false)
}

/// Pushes every stack entry's current opacity, asking `opacity_of` for each —
/// the once-per-frame call that makes the layer panel's slider real for tiled
/// layers.
///
/// One lock and no allocation, which is what lets it be unconditional.
/// Returns whether anything moved.
pub fn sync_tile_layer_opacities(
    render_state: &RenderState,
    opacity_of: impl Fn(LayerId) -> f32,
) -> bool {
    with_state(render_state, |state| {
        state.sync_tile_layer_opacities(opacity_of)
    })
    .unwrap_or(false)
}

/// Restricts one raster stack entry to the zoom range its source serves, so a
/// detail-limited archive draws magnified coarse tiles instead of nothing.
///
/// Returns `false` if nothing is installed or `layer` names no raster entry.
pub fn set_tile_layer_zoom_range(
    render_state: &RenderState,
    layer: LayerId,
    range: Option<(u8, u8)>,
) -> bool {
    with_state(render_state, |state| {
        state.set_tile_layer_zoom_range(layer, range)
    })
    .unwrap_or(false)
}

/// Every stack entry whose source the shell could not build, with its reason.
#[must_use]
pub fn tile_layer_refusals(render_state: &RenderState) -> Vec<(LayerId, String)> {
    with_state_ref(render_state, MapGpuState::tile_layer_refusals).unwrap_or_default()
}

/// Forgets every refused stack entry so its plan is offered again; returns how
/// many were dropped. The entries that are drawing are left alone.
pub fn retry_refused_tile_layers(render_state: &RenderState) -> usize {
    with_state(render_state, MapGpuState::retry_refused_tile_layers).unwrap_or(0)
}

/// Removes every tiled-layer stack entry — the "close project" path.
///
/// Returns `false` if no [`MapGpuState`] is installed.
pub fn clear_tile_layers(render_state: &RenderState) -> bool {
    with_state(render_state, MapGpuState::clear_tile_layers).is_some()
}

/// Number of tiled layers composited over the basemap; `0` when nothing is
/// installed.
#[must_use]
pub fn tile_layer_count(render_state: &RenderState) -> usize {
    with_state_ref(render_state, MapGpuState::tile_layer_count).unwrap_or(0)
}

/// Installs the fonts the label pass shapes with — the seam a shell's font
/// discovery plugs into, and the label counterpart of [`replace_vector_source`].
///
/// `primary` is the face every label starts in (both shells hand over
/// `oxifont_bundled::NOTO_SANS_REGULAR`); `fallbacks` are tried in order for
/// clusters it cannot map, which is how CJK names get glyphs without embedding
/// a CJK face.
///
/// Returns `false` — and drops the bytes — if no [`MapGpuState`] is installed
/// yet, which is the normal state on a shell's first frame; the failure to
/// *parse* `primary` is logged and also reported as `false`. Call it again next
/// frame rather than treating one `false` as final.
pub fn set_label_fonts(
    render_state: &RenderState,
    primary: Vec<u8>,
    fallbacks: Vec<Vec<u8>>,
) -> bool {
    let mut renderer = render_state.renderer.write();
    let Some(state) = renderer.callback_resources.get_mut::<MapGpuState>() else {
        return false;
    };
    match state.set_label_fonts(primary, fallbacks) {
        Ok(()) => true,
        Err(error) => {
            tracing::error!(%error, "oxigis-ui: the label font could not be parsed");
            false
        }
    }
}

/// Appends one font to the label pass's fallback chain — the seam an
/// asynchronously fetched CJK face arrives through.
///
/// Returns `false` if no [`MapGpuState`] is installed or no primary font has
/// been set yet, in which case `font` is dropped.
pub fn add_label_fallback_font(render_state: &RenderState, font: Vec<u8>) -> bool {
    let mut renderer = render_state.renderer.write();
    match renderer.callback_resources.get_mut::<MapGpuState>() {
        Some(state) => state.add_label_fallback_font(font),
        None => false,
    }
}

/// Appends several fonts to the label pass's fallback chain in one
/// invalidation — the seam a shell uses when more than one chain entry has
/// arrived by the time it drains its font channel.
///
/// Returns `false` if no [`MapGpuState`] is installed or no primary font has
/// been set yet, in which case the batch is dropped.
pub fn add_label_fallback_fonts(render_state: &RenderState, fonts: Vec<Vec<u8>>) -> bool {
    let mut renderer = render_state.renderer.write();
    match renderer.callback_resources.get_mut::<MapGpuState>() {
        Some(state) => state.add_label_fallback_fonts(fonts),
        None => false,
    }
}

/// Installs the BOLD face chain a `Bold`-weighted symbol style draws through
/// — the mirror of [`set_label_fonts`] for print/text v1.4's weight.
///
/// `fonts` are the bold faces, highest priority first; the label engine puts
/// the WHOLE regular chain behind them, so a bold face that lacks a script
/// falls through to the regular face for it instead of drawing `.notdef`.
/// An empty chain removes bold support: bold labels draw Regular with one
/// log, never a synthetic emboldening.
///
/// Returns `false` — and drops the bytes — if no [`MapGpuState`] is installed
/// or no primary font has been set yet, exactly as the fallback seams do.
pub fn set_label_bold_fonts(render_state: &RenderState, fonts: Vec<Vec<u8>>) -> bool {
    let mut renderer = render_state.renderer.write();
    match renderer.callback_resources.get_mut::<MapGpuState>() {
        Some(state) => state.set_label_bold_fonts(fonts),
        None => false,
    }
}

/// Detaches the vector-tile layer, if one is attached.
///
/// Returns `false` if no [`MapGpuState`] is installed.
pub fn clear_vector_source(render_state: &RenderState) -> bool {
    let mut renderer = render_state.renderer.write();
    match renderer.callback_resources.get_mut::<MapGpuState>() {
        // Detaching cannot fail: no renderer is constructed.
        Some(state) => state.set_vector_source(None).is_ok(),
        None => false,
    }
}

/// Adds a local (dropped, pasted or project-loaded) vector dataset to the map,
/// keyed by its application-side [`LayerId`].
///
/// The layer is added on top of the local stack, i.e. painted last of the local
/// ones and over every tiled vector layer. Re-adding a known id replaces the
/// dataset **in place**, keeping its position in the stack — that is the reload
/// path, not a reorder.
///
/// Returns `false` (and drops `layer`) if no [`MapGpuState`] is installed yet,
/// which is the normal state on a shell's first frame.
pub fn add_local_vector_layer(
    render_state: &RenderState,
    id: LayerId,
    layer: LocalVectorLayer,
) -> bool {
    with_locals(render_state, move |locals| {
        locals.insert(id, layer);
    })
    .is_some()
}

/// Removes the local dataset registered under `id`, returning it.
///
/// [`None`] if no [`MapGpuState`] is installed or `id` names no local layer —
/// the two are indistinguishable to the caller on purpose, because both mean
/// "there is nothing to remove".
pub fn remove_local_vector_layer(
    render_state: &RenderState,
    id: LayerId,
) -> Option<LocalVectorLayer> {
    with_locals(render_state, |locals| locals.remove(id)).flatten()
}

/// Shows or hides one local dataset without dropping its mesh.
///
/// Returns `false` if nothing is installed or `id` names no local layer.
pub fn set_local_layer_visibility(render_state: &RenderState, id: LayerId, visible: bool) -> bool {
    with_local_layer(render_state, id, |layer| layer.set_visible(visible))
}

/// Sets one local dataset's opacity (`0..=1`, clamped).
///
/// Cheap: opacity is an instance tint, so nothing is re-tessellated.
///
/// Returns `false` if nothing is installed or `id` names no local layer.
pub fn set_local_layer_opacity(render_state: &RenderState, id: LayerId, opacity: f32) -> bool {
    with_local_layer(render_state, id, |layer| layer.set_opacity(opacity))
}

/// Restyles one local dataset — the live path from the style panel.
///
/// Recompiles the layer's paint passes and label rules and marks its mesh
/// stale, so the next frame rebuilds it at the current zoom.
///
/// Returns `false` if nothing is installed or `id` names no local layer.
pub fn set_local_layer_style(
    render_state: &RenderState,
    id: LayerId,
    style: LayerStyleSet,
) -> bool {
    with_local_layer(render_state, id, move |layer| layer.set_style(style))
}

/// Reorders the local stack to match `order`, which may name any layer at all.
///
/// Ids that are not local layers are ignored and local layers `order` leaves out
/// keep their relative order at the end, so a shell can hand over its complete
/// layer list after a drag-and-drop without filtering it first.
///
/// Returns `false` if nothing is installed or the order was already that one.
pub fn reorder_local_vector_layers(render_state: &RenderState, order: &[LayerId]) -> bool {
    with_locals(render_state, |locals| locals.reorder(order)).unwrap_or(false)
}

/// Removes every local dataset — the "close project" path.
///
/// Returns `false` if no [`MapGpuState`] is installed.
pub fn clear_local_vector_layers(render_state: &RenderState) -> bool {
    with_locals(render_state, LocalVectorRenderer::clear).is_some()
}

/// Number of local datasets currently attached; `0` when nothing is installed.
#[must_use]
pub fn local_vector_layer_count(render_state: &RenderState) -> usize {
    with_locals(render_state, |locals| locals.len()).unwrap_or(0)
}

/// Whether `id` names a local dataset attached to the map.
#[must_use]
pub fn has_local_vector_layer(render_state: &RenderState, id: LayerId) -> bool {
    with_locals(render_state, |locals| locals.contains(id)).unwrap_or(false)
}

/// The attached local datasets' ids, in draw order (first painted first).
///
/// Empty when nothing is installed. What a layer panel diffs its own list
/// against before calling [`reorder_local_vector_layers`].
#[must_use]
pub fn local_vector_layer_ids(render_state: &RenderState) -> Vec<LayerId> {
    with_locals(render_state, |locals| locals.layer_ids()).unwrap_or_default()
}

/// Runs `f` against one attached local dataset, e.g. to read its
/// [`LocalVectorLayer::square`] for "zoom to layer" or its
/// [`LocalVectorLayer::feature_count`] for a status line.
///
/// **`f` runs while `render_state`'s renderer lock is held**, so keep it to
/// reading or cloning data out. In particular, do not draw the attribution
/// table's UI inside it — clone or copy what the panel needs and release the
/// lock first.
///
/// [`None`] if nothing is installed or `id` names no local layer.
pub fn with_local_vector_layer<R>(
    render_state: &RenderState,
    id: LayerId,
    f: impl FnOnce(&LocalVectorLayer) -> R,
) -> Option<R> {
    with_locals(render_state, |locals| locals.get(id).map(f)).flatten()
}

/// Runs `f` against the installed local stack.
///
/// The shared body of every entry point above; see [`with_state`] for why this
/// must not be called from inside a paint callback.
fn with_locals<R>(
    render_state: &RenderState,
    f: impl FnOnce(&mut LocalVectorRenderer) -> R,
) -> Option<R> {
    let mut renderer = render_state.renderer.write();
    renderer
        .callback_resources
        .get_mut::<MapGpuState>()
        .map(|state| f(state.locals_mut()))
}

/// Runs `f` against one attached local dataset, reporting whether it ran.
fn with_local_layer(
    render_state: &RenderState,
    id: LayerId,
    f: impl FnOnce(&mut LocalVectorLayer),
) -> bool {
    with_locals(render_state, |locals| match locals.get_mut(id) {
        Some(layer) => {
            f(layer);
            true
        }
        None => false,
    })
    .unwrap_or(false)
}

/// Runs `f` against the installed [`MapGpuState`], if any.
///
/// Escape hatch for state a shell wants to read or poke between frames
/// (cache statistics, layer tint, a forced texture flush) without this module
/// growing one wrapper per field. Do **not** call it from inside a paint
/// callback: it takes `render_state`'s renderer lock, which egui already holds
/// there.
pub fn with_state<R>(
    render_state: &RenderState,
    f: impl FnOnce(&mut MapGpuState) -> R,
) -> Option<R> {
    let mut renderer = render_state.renderer.write();
    renderer.callback_resources.get_mut::<MapGpuState>().map(f)
}

/// [`with_state`] for a read-only question, taking the *read* lock.
///
/// Used by the per-frame stack queries ([`installed_tile_stack`],
/// [`tile_layer_count`]), which a shell asks every frame and which have no
/// business excluding the renderer from other readers to do it.
///
/// Same rule as [`with_state`]: never from inside a paint callback.
pub fn with_state_ref<R>(
    render_state: &RenderState,
    f: impl FnOnce(&MapGpuState) -> R,
) -> Option<R> {
    let renderer = render_state.renderer.read();
    renderer.callback_resources.get::<MapGpuState>().map(f)
}

/// The per-frame paint callback for the map: draw `view` into `rect`.
///
/// `rect` is in egui points; `egui_wgpu` converts it to the pass viewport.
/// `view`'s [`MapView::size_px`] must equal `rect`'s size in physical pixels
/// (see the module's geometry contract).
#[must_use]
pub fn paint_callback(rect: Rect, view: MapView) -> PaintCallback {
    eframe::egui_wgpu::Callback::new_paint_callback(
        rect,
        MapPaintCallback {
            view,
            repaint: None,
        },
    )
}

/// [`paint_callback`] plus the context the prepare hook asks for another frame
/// on when it deferred work (a budgeted local re-tessellation — see
/// [`crate::local_layers::MAX_TESSELLATIONS_PER_FRAME`]).
///
/// Separate entry point rather than a widened [`paint_callback`], so an
/// out-of-tree shell keeps compiling; without a context the deferred work only
/// drains on the next input event, which for a static view means "never".
#[must_use]
pub fn paint_callback_repainting(rect: Rect, view: MapView, ctx: &egui::Context) -> PaintCallback {
    eframe::egui_wgpu::Callback::new_paint_callback(
        rect,
        MapPaintCallback {
            view,
            repaint: Some(ctx.clone()),
        },
    )
}

/// The `egui_wgpu` callback itself: a [`MapView`] and, optionally, the context
/// to nudge — the renderer itself lives in the callback resources.
#[derive(Debug, Clone)]
struct MapPaintCallback {
    /// Camera to draw this frame.
    view: MapView,
    /// Where to ask for the follow-up frame deferred work needs.
    repaint: Option<egui::Context>,
}

impl CallbackTrait for MapPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(state) = callback_resources.get_mut::<MapGpuState>() {
            state.run_frame(self.view, device, queue);
            state.run_stack_frame(self.view, device, queue);
            state.run_vector_frame(self.view, device, queue);
            state.run_local_frame(self.view, device, queue);
            state.run_label_frame(self.view, device, queue);
            // Local re-tessellation is budgeted, so a frame can finish with
            // work still owed. `Context::request_repaint` is thread-safe, which
            // is what makes it callable from the render thread at all.
            if state.local_tessellation_backlog() > 0
                && let Some(ctx) = self.repaint.as_ref()
            {
                ctx.request_repaint();
            }
        }
        // `MapRenderer` writes through the queue, not the encoder, so there is
        // no command buffer of ours to submit.
        Vec::new()
    }

    fn paint(
        &self,
        info: PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        if let Some(state) = callback_resources.get::<MapGpuState>() {
            // The vector pipeline scissors each tile to its own quad, which is
            // expressed in framebuffer coordinates — hence the callback rect's
            // origin. Its *size* is the view's own `size_px`, which the
            // geometry contract above already requires to match.
            let viewport = info.viewport_in_pixels();
            let clip_origin_px = [viewport.left_px as f32, viewport.top_px as f32];
            state.paint_into(render_pass, clip_origin_px);
        }
    }
}

#[cfg(test)]
mod tests {
    // Everything GPU-touching (`install`, the callback hooks) needs a live
    // `egui_wgpu::RenderState` and is exercised by the shells; what is testable
    // headlessly is the synthetic tile generator and the seam's plumbing.
    use super::{
        BoxedTileProvider, DEFAULT_MESH_BYTE_BUDGET, DEFAULT_TEXTURE_BYTES, DebugCheckerboard,
        MAX_DRAWN_TILE_LAYERS, MAX_TILE_EDGE_PX, MIN_TILE_EDGE_PX, TONE_BORDER, TileProvider,
        checkerboard_tile, reorder_plan, split_budget, split_mesh_budget,
    };
    use oxigis_core::LayerId;
    use oxigis_render::TileId;

    fn layer(raw: u64) -> LayerId {
        LayerId::from_raw(raw)
    }

    fn tile(z: u8, x: u32, y: u32) -> TileId {
        match TileId::new(z, x, y) {
            Ok(tile) => tile,
            Err(err) => panic!("tile {z}/{x}/{y} must be valid: {err}"),
        }
    }

    #[test]
    fn checkerboard_tiles_are_square_and_well_formed() {
        let decoded = checkerboard_tile(tile(3, 1, 2), 64).expect("generation must succeed");
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 64);
        assert_eq!(decoded.rgba().len(), 64 * 64 * 4);
    }

    #[test]
    fn checkerboard_edge_is_clamped() {
        let small = checkerboard_tile(tile(0, 0, 0), 0).expect("generation must succeed");
        assert_eq!(small.width(), MIN_TILE_EDGE_PX);
        let large = checkerboard_tile(tile(0, 0, 0), u32::MAX).expect("generation must succeed");
        assert_eq!(large.width(), MAX_TILE_EDGE_PX);
    }

    #[test]
    fn checkerboard_is_deterministic() {
        let first = checkerboard_tile(tile(5, 9, 4), 32).expect("generation must succeed");
        let second = checkerboard_tile(tile(5, 9, 4), 32).expect("generation must succeed");
        assert_eq!(first, second);
    }

    #[test]
    fn neighbouring_tiles_differ_in_tone() {
        let left = checkerboard_tile(tile(5, 9, 4), 32).expect("generation must succeed");
        let right = checkerboard_tile(tile(5, 10, 4), 32).expect("generation must succeed");
        let below = checkerboard_tile(tile(5, 9, 5), 32).expect("generation must succeed");
        assert_ne!(left, right, "east neighbour must differ");
        assert_ne!(left, below, "south neighbour must differ");
    }

    #[test]
    fn tiles_carry_a_distinct_border() {
        let decoded = checkerboard_tile(tile(2, 1, 1), 32).expect("generation must succeed");
        let rgba = decoded.rgba();
        assert_eq!(
            &rgba[0..4],
            &TONE_BORDER,
            "north-west corner must be border"
        );
        // Centre texel must not be border-coloured.
        let center = ((32 / 2) * 32 + (32 / 2)) * 4;
        assert_ne!(&rgba[center..center + 4], &TONE_BORDER);
    }

    #[test]
    fn debug_checkerboard_defaults_to_the_native_tile_size() {
        let provider = DebugCheckerboard::default();
        assert_eq!(provider.edge_px(), DebugCheckerboard::DEFAULT_EDGE_PX);
        let decoded = provider
            .tile(tile(1, 0, 1))
            .expect("provider must produce a tile");
        assert_eq!(decoded.width(), DebugCheckerboard::DEFAULT_EDGE_PX);
    }

    #[test]
    fn debug_checkerboard_clamps_its_edge() {
        assert_eq!(DebugCheckerboard::new(1).edge_px(), MIN_TILE_EDGE_PX);
        assert_eq!(DebugCheckerboard::new(u32::MAX).edge_px(), MAX_TILE_EDGE_PX);
    }

    #[test]
    fn a_provider_is_boxable_through_the_seam_alias() {
        let boxed: BoxedTileProvider = Box::new(DebugCheckerboard::new(16));
        assert!(boxed.tile(tile(0, 0, 0)).is_some());
    }

    #[test]
    fn the_device_memory_budget_is_split_and_never_reaches_zero() {
        // The whole point: one `MapRenderer` per layer would otherwise each
        // authorise the full default budget, so eight layers plus the basemap
        // would claim ~7 GiB of VRAM.
        assert_eq!(
            split_budget(DEFAULT_TEXTURE_BYTES, 1),
            DEFAULT_TEXTURE_BYTES
        );
        assert_eq!(
            split_budget(DEFAULT_TEXTURE_BYTES, 4) * 4,
            DEFAULT_TEXTURE_BYTES,
            "the TOTAL is what is bounded, not the per-layer share"
        );
        let full = split_budget(DEFAULT_TEXTURE_BYTES, MAX_DRAWN_TILE_LAYERS + 1);
        assert!(full * (MAX_DRAWN_TILE_LAYERS + 1) <= DEFAULT_TEXTURE_BYTES);
        assert!(
            full > 64 * 1024 * 1024,
            "a full stack still holds real tiles"
        );

        // `set_texture_byte_budget(Some(0))` is an error and a zero share would
        // evict every tile the moment it was uploaded, so neither is reachable.
        assert_eq!(split_budget(0, 0), 1);
        assert_eq!(split_budget(1, usize::MAX), 1);
        assert_eq!(
            split_mesh_budget(DEFAULT_MESH_BYTE_BUDGET, 0),
            DEFAULT_MESH_BYTE_BUDGET
        );
        assert_eq!(
            split_mesh_budget(DEFAULT_MESH_BYTE_BUDGET, 2) * 2,
            DEFAULT_MESH_BYTE_BUDGET
        );
        assert_eq!(split_mesh_budget(1, usize::MAX), 1);
    }

    #[test]
    fn a_reorder_is_a_permutation_and_tolerates_an_unfiltered_layer_list() {
        let current = [layer(1), layer(2), layer(3)];
        assert_eq!(
            reorder_plan(&current, &[layer(1), layer(2), layer(3)]),
            None,
            "the order it is already in moves nothing — and must not re-fetch"
        );
        assert_eq!(
            reorder_plan(&current, &[layer(3), layer(1), layer(2)]),
            Some(vec![2, 0, 1])
        );

        // A shell may hand over its WHOLE layer list: ids that name no stack
        // entry (local datasets, the basemap layer) are ignored …
        assert_eq!(
            reorder_plan(
                &current,
                &[layer(9), layer(3), layer(7), layer(1), layer(2)]
            ),
            Some(vec![2, 0, 1])
        );
        // … and entries the list leaves out keep their relative position at the
        // top rather than vanishing from the stack.
        let Some(partial) = reorder_plan(&current, &[layer(3)]) else {
            panic!("naming only the top entry still moves it");
        };
        assert_eq!(partial, vec![2, 0, 1]);
        assert_eq!(partial.len(), current.len(), "no entry may be dropped");

        // A repeated id cannot duplicate an entry.
        assert_eq!(
            reorder_plan(&current, &[layer(3), layer(3), layer(3)]),
            Some(vec![2, 0, 1])
        );
        assert_eq!(reorder_plan(&[], &[layer(1)]), None);
    }
}
