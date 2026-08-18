//! OxiGIS renderer — wgpu tile map rendering (the Phase 0 core).
//!
//! Scope (blueprint §5): raster tiles (XYZ + Cloud-Optimized GeoTIFF over
//! HTTP Range — see [`cog`]), vector tiles (MVT subset: fill / line / circle / symbol),
//! text labels and greedy placement (see [`label`]), tile pyramid + LRU cache,
//! Web Mercator math.
//!
//! This crate intentionally does not depend on `oxigis-core`, so it stays
//! reusable from other projects.
//!
//! # Portability contract
//!
//! Everything here builds for `wasm32-unknown-unknown` as well as for native
//! targets, which shapes two design rules:
//!
//! * **No I/O.** There is no HTTP client, no async runtime and no filesystem
//!   access. Tile bytes are acquired through the [`source::TileFetch`] /
//!   [`source::RangeFetch`] traits, which the desktop and web shells implement
//!   with whatever their platform provides. Fonts follow the same rule: they
//!   are injected as bytes ([`label::LabelEngine::new`]).
//! * **No owned GPU context.** Every GPU entry point borrows a
//!   [`wgpu::Device`]/[`wgpu::Queue`], so the renderer slots into `eframe`'s
//!   existing device instead of creating a second one.
//!
//! # Frame protocol
//!
//! ```no_run
//! use oxigis_render::{DecodedTile, LonLat, MapRenderer, MapView};
//!
//! # fn demo(device: &wgpu::Device, queue: &wgpu::Queue,
//! #         format: wgpu::TextureFormat) -> Result<(), oxigis_render::RenderError> {
//! let view = MapView::new(LonLat::new(139.7, 35.7), 9.0, [1280.0, 720.0])?;
//! let mut renderer = MapRenderer::new(view, 512, format)?;
//!
//! // 1. What does this frame need?
//! for placement in renderer.begin_frame(view) {
//!     let _ = placement.tile;
//! }
//! // 2. The shell fetches and decodes, then feeds the pixels back.
//! for tile in renderer.missing_tiles().to_vec() {
//!     let rgba = vec![0u8; 256 * 256 * 4]; // fetched + decoded by the shell
//!     renderer.accept_tile(tile, DecodedTile::new(256, 256, rgba)?);
//! }
//! // 3. Upload, then draw inside someone else's render pass.
//! renderer.prepare(device, queue)?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

pub mod cog;
#[cfg(feature = "decode")]
pub mod decode;
pub mod error;
pub mod gpu;
pub mod label;
pub mod mercator;
pub mod mvt;
pub mod pmtiles;
pub mod renderer;
pub mod source;
pub mod tile_cache;
pub mod vector;
pub mod viewport;

pub use crate::cog::{
    CogCrs, CogGeoTransform, CogLevel, CogMetadata, CogOpen, CogOpenProgress, CogReadStep,
    CogSource, CogSourceTile, CogTilePlan, CogTileRef, MemoryRangeFetch, RasterStretch,
    TMERC_MAX_LAT_DEG, TMERC_MAX_LON_OFFSET_DEG, TransverseMercator, WGS84_INVERSE_FLATTENING,
    WGS84_SEMI_MAJOR_M, decode_cog_tile, utm_central_meridian_deg,
};
#[cfg(any(test, feature = "fixtures"))]
pub use crate::cog::{sample_cog_bytes, sample_utm_cog_bytes};
#[cfg(feature = "decode")]
pub use crate::decode::decode_tile;
pub use crate::error::RenderError;
pub use crate::gpu::{
    FULL_TILE_UV, MAX_TILE_TEXTURE_SIZE, TILE_UV_SIZE, TileDraw, TileInstance, TilePipeline,
    TileTexture, opacity_tint, tile_texture_format,
};
pub use crate::label::{
    AnchorKind, AtlasRect, DEFAULT_ATLAS_SIZE, DEFAULT_LABEL_CACHE, GLYPH_PADDING_PX, GlyphAtlas,
    GlyphKey, HALO_OFFSETS, LABEL_PADDING_PX, LABEL_VERTEX_SIZE, LabelAnchor, LabelBox,
    LabelEngine, LabelGlyph, LabelHalo, LabelOrientation, LabelPipeline, LabelPlacer,
    LabelResolver, LabelSpec, LabelTable, LabelVertex, MAX_ATLAS_SIZE, MAX_LABEL_SIZE_PX,
    OwnedPlacedLabel, PlacedLabel, ShapedLabel, VERTICAL_ORIENTATION_UNICODE_VERSION, VerticalCell,
    VerticalOrientation, VerticalPlan, VerticalRefusal, VerticalRun, build_label_quads,
    feature_anchor, label_text, placed_labels, vertical_orientation_of, vertical_runs,
    vertical_script,
};
pub use crate::mercator::{
    EARTH_CIRCUMFERENCE_M, EARTH_RADIUS_M, LonLat, MAX_LATITUDE_DEG, MAX_ZOOM, MercatorBounds,
    MercatorPoint, TILE_SIZE_PX, TileId, WorldCoord, ground_resolution,
};
pub use crate::mvt::{
    MvtFeature, MvtGeometry, MvtLayer, MvtPolygon, MvtValue, VectorTile, decode_mvt,
};
pub use crate::pmtiles::{
    DirEntry, PmtilesArchive, PmtilesError, PmtilesHeader, PmtilesInfo, PmtilesOpen,
    PmtilesOpenProgress, TileLookup, deserialize_directory, tile_id_to_zxy, zxy_to_tile_id,
};
#[cfg(any(test, feature = "fixtures"))]
pub use crate::pmtiles::{
    PmtilesBuilder, sample_pmtiles_far_metadata, sample_pmtiles_leafed, sample_pmtiles_raster,
    sample_pmtiles_vector,
};
pub use crate::renderer::{
    DEFAULT_OVERZOOM_LEVELS, DEFAULT_TEXTURE_BYTES, DEFAULT_TEXTURE_CAPACITY, DecodedTile,
    MAX_TEXTURE_CAPACITY, MapRenderer, TileUploadFailure,
};
pub use crate::source::{ByteRange, RangeFetch, TileFetch, TileFuture, XyzTemplate};
pub use crate::tile_cache::{CacheStats, TileCache};
pub use crate::vector::{
    CirclePaint, DEFAULT_MESH_BYTE_BUDGET, DEFAULT_MESH_CAPACITY, FillPaint, LayerPaint, LinePaint,
    PaintResolver, PaintTable, Rgba8, ScissorRect, TessParams, VectorDraw, VectorLayerRenderer,
    VectorMesh, VectorPipeline, VectorTileGpu, VectorVertex, tessellate_tile, tile_scissor,
};
pub use crate::viewport::{MAX_VISIBLE_TILES, MapView, TilePlacement};

/// Crate version, re-exported so shells can display it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_is_exposed() {
        assert!(!VERSION.is_empty());
    }
}
