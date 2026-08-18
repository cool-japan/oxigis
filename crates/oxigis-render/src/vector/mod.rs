//! Vector tiles: paint parameters, `lyon` tessellation and the GPU pass that
//! draws the resulting meshes (blueprint §5.2).
//!
//! # Layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`paint`] | render-side fill/line/circle paints and the [`PaintResolver`] style seam |
//! | [`tess`] | [`VectorTile`] → [`VectorMesh`]: `lyon` fills and strokes, circle fans, per-zoom LoD |
//! | `simplify` | the point-dropping pass fills and outlines share, crate-private |
//! | [`pipeline`] | [`VectorPipeline`]: WGSL, per-tile placement, per-tile scissor |
//! | [`renderer`] | [`VectorLayerRenderer`]: the frame protocol, mesh cache and draw list |
//!
//! [`VectorTile`]: crate::mvt::VectorTile
//!
//! # Pipeline of one tile
//!
//! ```no_run
//! use oxigis_render::{
//!     LonLat, MapView, RenderError, VectorLayerRenderer, decode_mvt,
//!     vector::{FillPaint, PaintTable, Rgba8, TessParams, tessellate_tile},
//! };
//!
//! # fn demo(bytes: &[u8], format: wgpu::TextureFormat) -> Result<(), RenderError> {
//! let view = MapView::new(LonLat::new(139.7, 35.7), 12.0, [1280.0, 720.0])?;
//! let mut vectors = VectorLayerRenderer::new(view, 512, format)?;
//!
//! let table = PaintTable::new().with("water", FillPaint::new(Rgba8::opaque(0x3a, 0x6e, 0xa5)));
//! for tile in vectors.begin_frame(view).to_vec() {
//!     let decoded = decode_mvt(bytes)?; // bytes fetched by the shell
//!     let extent = decoded.layers.first().map_or(4096, |layer| layer.extent);
//!     let params = TessParams::for_tile(
//!         view.tile_size_px(),
//!         extent,
//!         TessParams::DEFAULT_TOLERANCE_PX,
//!     )?;
//!     let mesh = tessellate_tile(&decoded, &table, &params)?;
//!     vectors.accept_mesh(tile.tile, mesh);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Conventions, in one place
//!
//! * **Colour** — straight (non-premultiplied) sRGB bytes on the vertices,
//!   [`wgpu::BlendState::ALPHA_BLENDING`] in the pipeline, sRGB→linear
//!   conversion in the shader when the colour target is an sRGB format.
//! * **Winding** — rings keep the orientation the MVT decoder produced
//!   (exteriors positive-area in y-down space, interiors negative) and the fill
//!   uses [`lyon::tessellation::FillRule::NonZero`], so holes cancel out
//!   without this crate second-guessing `lyon`'s axis conventions. Proven by
//!   the hole test in [`tess`].
//! * **Order** — layers in tile order, features in layer order, one index
//!   buffer, no depth test: a painter's algorithm.
//! * **Pixel widths** — stroke widths and circle radii are expanded on the CPU
//!   but carried as [`VectorVertex::offset`] as well, and the pipeline rescales
//!   them per tile, so a mesh keeps its widths through a fractional zoom
//!   instead of growing with the tile.
//! * **Clipping** — MVT buffer geometry is kept and each tile is scissored to
//!   its own quad at draw time ([`tile_scissor`]).
//!
//! # Out of scope
//!
//! Symbol and text layers (glyph atlas, greedy label placement) are §5.3 and
//! deliberately absent: [`LayerPaint`] has no symbol variant yet.

pub mod paint;
pub mod pipeline;
pub mod renderer;
mod simplify;
pub mod tess;

pub use crate::vector::paint::{
    CirclePaint, FillPaint, LayerPaint, LinePaint, PaintResolver, PaintTable, Rgba8,
};
pub use crate::vector::pipeline::{
    MeshBufferPool, ScissorRect, VECTOR_VERTEX_SIZE, VectorDraw, VectorPipeline, VectorTileGpu,
    check_mesh, tile_scissor,
};
pub use crate::vector::renderer::{
    DEFAULT_MESH_BYTE_BUDGET, DEFAULT_MESH_CAPACITY, VectorLayerRenderer,
};
pub use crate::vector::tess::{
    MAX_CIRCLE_SEGMENTS, MAX_MESH_VERTICES, MAX_OFFSET_SCALE, MIN_CIRCLE_SEGMENTS,
    MIN_TOLERANCE_TILE_UNITS, TessParams, TessScratch, VectorMesh, VectorVertex,
    check_mesh_capacity, circle_segments, offset_scale, tessellate_tile, tessellate_tile_into,
    tessellate_tile_with,
};
