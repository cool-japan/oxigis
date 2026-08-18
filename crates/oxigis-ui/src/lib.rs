//! OxiGIS UI — panel widgets built on OxiUI's egui backend.
//!
//! Panels (blueprint §3/§6): layer tree (visibility / opacity / reorder /
//! delete), style editor (fill / stroke / opacity / radius), attribute
//! table (`oxiui-table`), and Processing dialogs auto-generated from the
//! `oxigis-core` registry.

#![forbid(unsafe_code)]

pub mod app;
pub mod archive;
pub mod attribute_table;
pub mod cog_provider;
mod crs_sniff;
pub mod edit;
pub mod export;
pub mod geolibre_import;
#[cfg(feature = "geoparquet")]
pub mod geoparquet_input;
pub mod gpkg_input;
pub mod label_frame;
pub mod layer_panel;
pub mod layer_source;
pub mod local_input;
pub mod local_layers;
pub mod local_vector;
pub mod map_gpu;
pub mod map_view;
pub mod mbtiles;
pub mod measure;
pub mod print;
pub mod processing_exec;
pub mod processing_panel;
pub mod renderer_panel;
pub mod scalebar;
pub mod shapefile_input;
pub mod style_paint;
pub mod style_panel;
pub mod table_panel;
pub mod tile_provider;
pub mod ui_glyphs;
pub mod vector_provider;

pub use app::{
    ArchiveProbeRequest, MAX_DRAWN_TILE_LAYERS, MAX_RECENT_PROJECTS, MAX_SESSION_ARCHIVE_BYTES,
    OxigisApp, ProjectOpenRequest, ProjectSaveRequest, RasterWork, Refusal, TileLayerPlan,
    TileLayerSource, TileStack, TileStackWork, VectorWork, draws_as_tile_layer, format_for_url,
};
pub use archive::{
    ArchiveContent, ArchiveInfo, ArchiveLayerConfig, ArchiveProbe, ArchiveTileProvider,
    ArchiveTileTransport, MAX_INFLIGHT_ARCHIVE_TILES, MemoryRangeTransport, OpenedArchive,
    archive_paints,
};
pub use attribute_table::{
    AttributeSchema, FeatureRowSource, GEOMETRY_COLUMN_NAME, INDEX_COLUMN_NAME,
    MAX_PROPERTY_COLUMNS, SYNTHETIC_COLUMN_COUNT,
};
pub use cog_provider::{
    CogLayerConfig, CogTileProvider, MAX_INFLIGHT_COG_TILES, RangeJob, RangeSink, RangeTransport,
};
pub use export::{
    ExportKind, ExportRequest, FALLBACK_EXPORT_STEM, MAX_EXPORT_STEM_CHARS, export_file_name,
    feature_collection_geojson,
};
pub use gpkg_input::{GpkgCrs, GpkgDataset, GpkgRefusal, GpkgTable};
pub use label_frame::{LabelFrame, TiledLabelInput};
pub use local_input::{
    AddedGpkg, AddedLocalLayer, DropKind, DroppedDataset, DroppedItem, INLINE_GEOJSON_WARN_BYTES,
    LocalInputState, LocalLayerOp, PendingPath, ShapefileDrop, ShapefilePart, ZOOM_TO_LAYER_MARGIN,
    classify_drop, group_dropped_files, is_local_layer, is_local_vector_source, local_layer_order,
    looks_like_geojson,
};
pub use local_layers::{CULL_MARGIN_PX, LocalLabelJob, LocalVectorRenderer};
pub use local_vector::{
    GeometryKind, LOCAL_EXTENT, LOCAL_LAYER_NAME, LocalVectorError, LocalVectorLayer,
    MercatorSquare, default_style_for, default_style_for_kind, dominant_geometry_kind,
    family_of_layer_name, feature_collection_to_tile, feature_collection_to_tile_with,
    geometry_kind, local_class_layer_name, local_symbol_style,
};
pub use map_gpu::{
    BoxedTileProvider, DebugCheckerboard, MapGpuState, TileLayerGpuSource, TileProvider,
    add_local_vector_layer, clear_local_vector_layers, clear_tile_layers, has_local_vector_layer,
    install_tile_layer, installed_tile_stack, local_vector_layer_count, local_vector_layer_ids,
    remove_local_vector_layer, remove_tile_layer, remove_tile_layers, reorder_local_vector_layers,
    reorder_tile_layers, retry_refused_tile_layers, set_local_layer_opacity, set_local_layer_style,
    set_local_layer_visibility, set_tile_layer_opacity, set_tile_layer_zoom_range,
    sync_tile_layer_opacities, tile_layer_count, tile_layer_refusals, with_local_vector_layer,
    with_state_ref,
};
pub use map_view::{MapPanelState, PanGate};
pub use measure::{
    CoordinateError, GO_TO_HINT, GoToDialog, MAX_MEASURE_VERTICES, MEASURE_INSTRUCTION,
    MeasureSession, authalic_radius_m, format_area, format_distance, geodesic_distance_m,
    initial_bearing_deg, parse_coordinate, path_length_m, ring_area_m2,
};
pub use processing_panel::ProcessingFileRequest;
pub use scalebar::{ScreenScaleBar, metres_per_logical_px, round_125_down, screen_scale_bar};
// Test scaffolding, behind the default-off `fixtures` feature — the same
// arrangement `oxigis-render` uses for its `sample_pmtiles_*` archives, and the
// only public item that feature adds.
#[cfg(any(test, feature = "fixtures"))]
pub use mbtiles::sample_mbtiles_raster;
pub use renderer_panel::{
    BreakRule, DEFAULT_CLASS_COUNT, LayerAttributes, MAX_GRADUATED_CLASSES, RendererEvent,
    RendererPanelState, legend_rows,
};
pub use shapefile_input::{PrjCrs, ShapefileBytes, assemble_rings, sniff_prj};
pub use style_paint::{PaintProgram, label_spec, label_table, layer_paint, outline_paint};
pub use tile_provider::{
    ARCHIVE_TILE_URL, BasemapConfig, OSM_ATTRIBUTION, OSM_URL_TEMPLATE, TileError,
    TileProviderStats, TileSink, TileTransport, XyzTileProvider,
};
pub use vector_provider::{
    BoxedVectorTileSource, DEMO_CENTROID_TEXT_FIELD, DEMO_GEOLINE_TEXT_FIELD, MAPLIBRE_ATTRIBUTION,
    MAPLIBRE_DEMO_URL_TEMPLATE, TILED_RENDERER_REFUSAL, VectorSink, VectorTileConfig,
    VectorTileProvider, VectorTileSource, maplibre_demo_paints,
};

/// Crate version, re-exported so shells can display it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
