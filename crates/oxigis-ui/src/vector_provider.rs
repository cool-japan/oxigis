//! Vector (MVT) tile layers: the shell-agnostic half of the fetch loop.
//!
//! [`VectorTileProvider`] is to a vector basemap what [`crate::XyzTileProvider`]
//! is to a raster one. It reuses the very same platform seam — the
//! [`crate::TileTransport`] trait, "given a URL, eventually produce bytes" —
//! because an MVT endpoint is an ordinary XYZ service whose tiles happen to
//! carry protobuf instead of PNG. Nothing new is asked of a shell: the desktop
//! `HttpTileTransport` and the browser `FetchTileTransport` drive both paths.
//!
//! ```text
//! frame N     source.mesh(t) -> None, marks `t` in flight, asks the transport
//! (off-frame) transport fetches the URL, VectorSink::deliver
//!             gunzips (if needed) -> decode_mvt -> tessellate -> stores + repaints
//! frame N+k   source.mesh(t) -> Some(VectorMesh)  (renderer uploads buffers)
//! ```
//!
//! # Where the CPU work happens
//!
//! Decoding *and* tessellation run inside [`VectorSink::deliver`], i.e. on the
//! transport's worker thread natively and after the `fetch()` await in the
//! browser — never on the render thread. That mirrors the raster path, where
//! [`crate::TileSink::deliver`] is also what calls `decode_tile`, and it is the
//! reason [`VectorTileSource::mesh`] can stay the synchronous, non-blocking call
//! egui's `wgpu` prepare hook needs.
//!
//! The one exception is a zoom step. Tessellated meshes bake the on-screen tile
//! size (see [`oxigis_render::TessParams`]), so every mesh is thrown away when
//! the integer zoom bucket changes. Re-tessellating from the **decoded** tile
//! cache costs no network round trip, and is done lazily on the render thread
//! under a per-frame budget ([`MAX_TESSELLATIONS_PER_FRAME`]) so a zoom step
//! spreads its cost over a handful of frames instead of stalling one.
//!
//! # Bounded resources
//!
//! | Resource | Bound | Constant |
//! |---|---|---|
//! | Concurrent requests | 16 | [`crate::tile_provider::MAX_INFLIGHT_TILES`] |
//! | Decoded tiles (re-tessellation source) | 128 (LRU) | [`DECODED_CACHE_TILES`] |
//! | Meshes waiting for the renderer | 64 (LRU) | [`crate::tile_provider::READY_CACHE_TILES`] |
//! | Remembered failures | 1024 (LRU) | [`crate::tile_provider::FAILURE_MEMORY_TILES`] |
//! | Tessellations per frame | 4 | [`MAX_TESSELLATIONS_PER_FRAME`] |
//!
//! # Transport encoding
//!
//! Vector tiles are frequently stored gzipped (that is how MBTiles and PMTiles
//! archives hold them), so a body starting with the gzip magic `1f 8b` is
//! inflated here with `oxiarc-deflate` before [`oxigis_render::decode_mvt`] sees
//! it. Plain XYZ endpoints are unaffected: the browser transparently decodes
//! `Content-Encoding`, and the native `ureq` configuration never advertises
//! `Accept-Encoding` at all, so both hand over raw protobuf.

use std::collections::HashSet;
use std::sync::Arc;

use oxigis_core::{
    CircleStyle, Color, FillStyle, LayerStyle, LineStyle, SymbolStyle, VectorSource,
    VectorTilePaint,
};
use oxigis_render::{
    LabelTable, MapView, RenderError, TessParams, TileCache, TileId, VectorMesh, VectorTile,
    XyzTemplate, decode_mvt,
};
use parking_lot::Mutex;

use crate::archive::ArchiveLayerConfig;
use crate::style_paint::{PaintProgram, label_table};
use crate::tile_provider::{
    FAILURE_MEMORY_TILES, FailureState, MAX_INFLIGHT_TILES, READY_CACHE_TILES, TileDelivery,
    TileError, TileHealth, TileProviderStats, TileSink, TileTransport, truncate_for_display,
};

/// Number of decoded [`VectorTile`]s kept so a zoom step can re-tessellate
/// without refetching.
///
/// A decoded world-scale MVT tile is tens of kilobytes of geometry; 128 covers
/// several screenfuls across two or three zoom levels, which is the working set
/// of an interactive zoom.
pub const DECODED_CACHE_TILES: usize = 128;

/// Tiles re-tessellated on the render thread in a single frame.
///
/// Only reached right after a zoom step, when every visible tile needs a new
/// mesh at once; the renderer re-reports what it is missing every frame, so the
/// rest simply arrive over the following frames.
pub const MAX_TESSELLATIONS_PER_FRAME: usize = 4;

/// First two bytes of a gzip member (RFC 1952 §2.3.1).
pub const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Coordinate extent assumed when a tile declares no layers at all.
///
/// Every MVT encoder in practice uses 4096; this only affects the level-of-detail
/// budget of a tile that has no geometry to apply it to.
pub const DEFAULT_EXTENT: u32 = 4096;

/// URL template of the keyless MapLibre demo vector tiles (world countries).
pub const MAPLIBRE_DEMO_URL_TEMPLATE: &str = "https://demotiles.maplibre.org/tiles/{z}/{x}/{y}.pbf";

/// Attribution the MapLibre demo tiles are shown with.
pub const MAPLIBRE_ATTRIBUTION: &str = "\u{a9} MapLibre";

/// Fill colour of the demo style's `countries` layer: a light tan landmass.
pub const DEMO_COUNTRY_FILL: Color = Color {
    r: 0xE8,
    g: 0xDF,
    b: 0xC6,
    a: 0xFF,
};

/// Outline colour of the demo style's `countries` layer.
pub const DEMO_COUNTRY_OUTLINE: Color = Color {
    r: 0x8A,
    g: 0x7F,
    b: 0x66,
    a: 0xFF,
};

/// Stroke colour of the demo style's `geolines` layer (graticule / dateline).
pub const DEMO_GEOLINE_COLOR: Color = Color {
    r: 0x88,
    g: 0x8E,
    b: 0x98,
    a: 0xFF,
};

/// Disc colour of the demo style's `centroids` layer.
pub const DEMO_CENTROID_COLOR: Color = Color {
    r: 0x33,
    g: 0x3A,
    b: 0x44,
    a: 0xFF,
};

/// Text colour of the demo style's `geolines` labels: quieter than a country
/// name, since a graticule label is reference furniture rather than content.
pub const DEMO_GEOLINE_LABEL_COLOR: Color = Color {
    r: 0x5A,
    g: 0x60,
    b: 0x6A,
    a: 0xFF,
};

/// Feature property holding a country name in the demo source's `centroids`
/// layer.
///
/// Upper case, unlike `geolines`' `name` — the demo tiles genuinely spell the
/// two differently, which is why the table cannot use one property everywhere.
pub const DEMO_CENTROID_TEXT_FIELD: &str = "NAME";

/// Feature property holding a graticule name in the demo source's `geolines`
/// layer.
pub const DEMO_GEOLINE_TEXT_FIELD: &str = "name";

/// Label size of the demo style's `geolines` layer, in pixels.
pub const DEMO_GEOLINE_LABEL_SIZE_PX: f32 = 10.0;

/// The default paint rules for the MapLibre demo tiles.
///
/// The demo source ships three layers — `countries` (polygons), `geolines`
/// (lines) and `centroids` (points) — so the table covers one of each geometry
/// kind, which also makes it a end-to-end exercise of the tessellator.
///
/// Two of them additionally carry a [`LayerStyle::Symbol`] rule, which draws no
/// geometry and instead feeds the label pass (see
/// [`crate::style_paint::label_table`]):
///
/// * `centroids` → `NAME`, at [`SymbolStyle::new`]'s defaults, which are exactly
///   what the demo wants: 12 px black text on a 1 px white halo.
/// * `geolines` → `name`, smaller and grey, so the dateline and the equator are
///   named without competing with the country labels.
///
/// `countries` deliberately gets **no** symbol rule. Its polygons cover the same
/// countries the `centroids` points do, so labelling both would draw every name
/// twice — and the second copy would lose the collision pass anyway, wasting a
/// shaping round trip per country per frame.
#[must_use]
pub fn maplibre_demo_paints() -> Vec<VectorTilePaint> {
    let mut countries = FillStyle::new(DEMO_COUNTRY_FILL);
    countries.outline_color = Some(DEMO_COUNTRY_OUTLINE);

    let mut geoline_labels = SymbolStyle::new(DEMO_GEOLINE_TEXT_FIELD);
    geoline_labels.text_color = DEMO_GEOLINE_LABEL_COLOR;
    geoline_labels.set_text_size(DEMO_GEOLINE_LABEL_SIZE_PX);

    vec![
        VectorTilePaint::new("countries", LayerStyle::Fill(countries)),
        VectorTilePaint::new(
            "geolines",
            LayerStyle::Line(LineStyle::new(DEMO_GEOLINE_COLOR, 0.75)),
        ),
        VectorTilePaint::new(
            "centroids",
            LayerStyle::Circle(CircleStyle::new(2.0, DEMO_CENTROID_COLOR)),
        ),
        VectorTilePaint::new("geolines", LayerStyle::Symbol(geoline_labels)),
        VectorTilePaint::new(
            "centroids",
            LayerStyle::Symbol(SymbolStyle::new(DEMO_CENTROID_TEXT_FIELD)),
        ),
    ]
}

/// Why a tiled layer's style panel offers no Categorized / Graduated combo
/// (thematic v1.6).
///
/// Shown as the combo's disabled-hover text, so the refusal names its reason
/// where the user looks for it rather than leaving a control that appears to
/// work. See [`VectorTileConfig::supports_renderer`] for the structural cause.
pub const TILED_RENDERER_REFUSAL: &str = "Tiled layers draw from their source's paint rules, which are matched by source-layer name \
     and never see a feature's attributes — so a categorized or graduated renderer cannot take \
     effect here. Add the data as a local vector layer to classify it.";

/// A vector-tile layer: where the `.pbf` tiles come from and how to draw them.
///
/// Two shapes, one type. Ordinarily `url_template` names an XYZ endpoint and
/// [`VectorTileConfig::template`] expands it per tile. When `archive` is set the
/// tiles instead come from one already-named file, `template` answers
/// [`None`], and the provider hands [`crate::tile_provider::ARCHIVE_TILE_URL`]
/// to its transport. Keeping one type is what lets
/// [`crate::print::PrintRequest`] carry a vector layer without knowing which
/// shape it is — the whole print path is unchanged for archives.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorTileConfig {
    /// `{z}/{x}/{y}` URL template, as understood by [`XyzTemplate`].
    ///
    /// For an archive-backed config this is a display string only (the
    /// archive's location); nothing expands it.
    pub url_template: String,
    /// Hosts `{s}` rotates through; empty when the template has no `{s}`.
    pub subdomains: Vec<String>,
    /// Credit line drawn alongside the basemap's. Empty shows nothing.
    pub attribution: String,
    /// Per-source-layer paint rules, in style order.
    pub paints: Vec<VectorTilePaint>,
    /// When set, the tiles come from this single-file archive and
    /// `url_template` only names it for the UI.
    pub archive: Option<ArchiveLayerConfig>,
}

impl VectorTileConfig {
    /// A layer reading `url_template` with the demo paint rules and no
    /// attribution.
    #[must_use]
    pub fn new(url_template: impl Into<String>) -> Self {
        Self {
            url_template: url_template.into(),
            subdomains: Vec::new(),
            attribution: String::new(),
            paints: maplibre_demo_paints(),
            archive: None,
        }
    }

    /// A layer reading `url_template` with exactly `paints` and no
    /// attribution.
    ///
    /// The paints-free sibling of [`Self::new`], and the one every caller that
    /// already knows its style must use: `new` builds [`maplibre_demo_paints`]
    /// eagerly — five [`VectorTilePaint`]s, each a source-layer [`String`] plus
    /// a [`LayerStyle`] (two of them [`SymbolStyle`]s carrying a text field of
    /// their own) — which `new(..).with_paints(..)` then drops unread.
    /// [`config_for`] is on the per-frame path (`OxigisApp::desired_vector`
    /// calls it for every stored MVT layer, every frame, to decide whether the
    /// provider has to be rebuilt), so that discarded work was paid per layer
    /// per frame rather than once.
    #[must_use]
    pub fn styled(url_template: impl Into<String>, paints: Vec<VectorTilePaint>) -> Self {
        Self {
            url_template: url_template.into(),
            subdomains: Vec::new(),
            attribution: String::new(),
            paints,
            archive: None,
        }
    }

    /// The keyless MapLibre demo tiles with their attribution and default
    /// style.
    #[must_use]
    pub fn maplibre_demo() -> Self {
        Self {
            url_template: MAPLIBRE_DEMO_URL_TEMPLATE.to_string(),
            subdomains: Vec::new(),
            attribution: MAPLIBRE_ATTRIBUTION.to_string(),
            paints: maplibre_demo_paints(),
            archive: None,
        }
    }

    /// A layer whose tiles come out of `archive`, styled by `paints`.
    ///
    /// `url_template` is set to the archive's location so every UI surface that
    /// shows "where the tiles come from" keeps working unchanged; nothing ever
    /// expands it, because [`VectorTileConfig::template`] answers [`None`] for
    /// an archive-backed config.
    #[must_use]
    pub fn from_archive(archive: ArchiveLayerConfig, paints: Vec<VectorTilePaint>) -> Self {
        Self {
            url_template: archive.location().to_owned(),
            subdomains: Vec::new(),
            attribution: archive.attribution.clone(),
            paints,
            archive: Some(archive),
        }
    }

    /// Sets the credit line.
    #[must_use]
    pub fn with_attribution(mut self, attribution: impl Into<String>) -> Self {
        self.attribution = attribution.into();
        self
    }

    /// Replaces the paint rules.
    #[must_use]
    pub fn with_paints(mut self, paints: Vec<VectorTilePaint>) -> Self {
        self.paints = paints;
        self
    }

    /// Builds the [`XyzTemplate`] this configuration describes, if it describes
    /// one.
    ///
    /// [`None`] means "the tiles come from an archive": there is no per-tile
    /// URL to expand, and the provider addresses tiles by [`TileId`] inside one
    /// already-open file instead.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidTemplate`] if the template is missing a
    /// placeholder, or if `subdomains` and the template's `{s}` disagree.
    pub fn template(&self) -> Result<Option<XyzTemplate>, RenderError> {
        if self.archive.is_some() {
            return Ok(None);
        }
        let template = XyzTemplate::new(self.url_template.clone())?;
        if self.subdomains.is_empty() {
            Ok(Some(template))
        } else {
            template.with_subdomains(self.subdomains.clone()).map(Some)
        }
    }

    /// The tessellation program the paint rules describe.
    #[must_use]
    pub fn program(&self) -> PaintProgram {
        PaintProgram::from_rules(
            self.paints
                .iter()
                .map(|rule| (rule.source_layer.as_str(), &rule.style)),
        )
    }

    /// The label table the paint rules describe.
    ///
    /// The [`PaintProgram::from_rules`] twin: same rule list, the symbol half
    /// of it. See [`crate::style_paint`] for why the two are built separately.
    #[must_use]
    pub fn label_table(&self) -> LabelTable {
        label_table(
            self.paints
                .iter()
                .map(|rule| (rule.source_layer.as_str(), &rule.style)),
        )
    }

    /// Whether this configuration can carry a thematic
    /// [`oxigis_core::Renderer`] — always `false`, for every tiled layer
    /// (thematic v1.6).
    ///
    /// A categorized or graduated renderer needs to see a feature's
    /// ATTRIBUTES to classify it, and the tiled path never gives it the
    /// chance: [`oxigis_render::PaintResolver::paint_for`] receives a source
    /// LAYER NAME and nothing else, and `MeshBuilder::add_feature` receives
    /// `&feature.geometry` — an `MvtFeature`'s properties are decoded and then
    /// dropped before either is reached. A combo that offered Categorized here
    /// would therefore be a control that silently changes nothing on screen
    /// while rewriting the file that gets saved.
    ///
    /// The local vector path has no such gap (it partitions the synthetic tile
    /// itself, see [`crate::local_vector::feature_collection_to_tile_with`]),
    /// so thematic styling is offered exactly there. Widening the render seam
    /// to per-feature paints is the roadmap item that would change this
    /// answer; until it lands, the honest answer is a refusal with a reason,
    /// which is what [`TILED_RENDERER_REFUSAL`] carries.
    #[must_use]
    pub fn supports_renderer(&self) -> bool {
        false
    }

    /// The core layer source this configuration serializes as.
    #[must_use]
    pub fn to_source(&self) -> VectorSource {
        match &self.archive {
            Some(archive) => VectorSource::TileArchive {
                archive: archive.archive.clone(),
                format: archive.format,
                paints: self.paints.clone(),
                attribution: archive.attribution.clone(),
            },
            None => VectorSource::MvtTiles {
                url_template: self.url_template.clone(),
                paints: self.paints.clone(),
            },
        }
    }
}

/// The config a project-stored MVT layer means: the stored paints, plus the
/// credit the keyless MapLibre demo source requires; any other source
/// carries none (the project format stores no attribution).
///
/// The ONE rule both the add seam and [`crate::OxigisApp::desired_vector`]
/// go through, so the two cannot drift — which is what makes a *loaded*
/// project credit MapLibre exactly as a freshly added layer does.
///
/// Built through [`VectorTileConfig::styled`] rather than
/// `new(..).with_paints(..)`: `desired_vector` calls this once per stored MVT
/// layer per frame, and `new`'s eager [`maplibre_demo_paints`] would be five
/// paint rules allocated and dropped unread on each of those calls.
#[must_use]
pub fn config_for(url_template: &str, paints: Vec<VectorTilePaint>) -> VectorTileConfig {
    let config = VectorTileConfig::styled(url_template, paints);
    if url_template == MAPLIBRE_DEMO_URL_TEMPLATE {
        config.with_attribution(MAPLIBRE_ATTRIBUTION.to_string())
    } else {
        config
    }
}

impl Default for VectorTileConfig {
    fn default() -> Self {
        Self::maplibre_demo()
    }
}

/// Supplies tessellated meshes for the tiles a frame is missing.
///
/// The vector counterpart of [`crate::map_gpu::TileProvider`], and called from
/// the same place: egui's `wgpu` prepare hook. Both methods must therefore
/// return promptly — [`VectorTileSource::mesh`] answers [`None`] for "not ready
/// yet" and the renderer asks again next frame.
pub trait VectorTileSource: 'static {
    /// Starts a frame for `view`, returning whether every mesh handed out so far
    /// has been invalidated.
    ///
    /// `true` means the shell must call
    /// [`oxigis_render::VectorLayerRenderer::clear_meshes`] before
    /// `begin_frame`, because the tessellation parameters baked into the meshes
    /// no longer match the camera (see [`oxigis_render::TessParams`]).
    fn begin_frame(&self, view: MapView) -> bool;

    /// The mesh for `tile`, or [`None`] if it is not ready (or never will be).
    fn mesh(&self, tile: TileId) -> Option<VectorMesh>;

    /// The *decoded* tile, for a pass that needs feature attributes rather than
    /// triangles — which in practice means label placement
    /// ([`oxigis_render::LabelPlacer::place_tile`]).
    ///
    /// Like [`VectorTileSource::mesh`] this is called from the render thread
    /// and must return promptly: [`None`] means "not decoded (yet)", and the
    /// tile is simply left unlabelled this frame. It must **not** start a fetch
    /// — `mesh` is what drives the fetch loop, and a tile whose mesh is on
    /// screen has necessarily been decoded.
    ///
    /// The default implementation labels nothing, so a source that only ever
    /// produces geometry needs no extra code.
    fn decoded(&self, tile: TileId) -> Option<Arc<VectorTile>> {
        let _ = tile;
        None
    }

    /// How to label each source layer this source serves.
    ///
    /// Consulted once per tile per frame by the label pass. The default is the
    /// empty table, i.e. no labels at all.
    fn label_table(&self) -> &LabelTable {
        &EMPTY_LABEL_TABLE
    }
}

/// The table [`VectorTileSource::label_table`] defaults to: no layer is
/// labelled.
static EMPTY_LABEL_TABLE: LabelTable = LabelTable::new();

/// A [`VectorTileSource`] stored inside [`crate::map_gpu::MapGpuState`].
///
/// `Send + Sync` is required on every target for the same reason as
/// [`crate::map_gpu::BoxedTileProvider`]: `egui_wgpu`'s callback resources are a
/// concurrent type map even on wasm.
pub type BoxedVectorTileSource = Box<dyn VectorTileSource + Send + Sync>;

/// The provider's shared, interior-mutable state.
struct VectorStore {
    /// Decoded tiles, kept so a zoom step can re-tessellate without refetching.
    decoded: TileCache<Arc<VectorTile>>,
    /// Meshes waiting for the renderer to collect them.
    meshes: TileCache<VectorMesh>,
    /// Tiles a transport is currently working on.
    inflight: HashSet<TileId>,
    /// Retry state per failed tile, LRU-bounded. See [`FailureState`].
    failures: TileCache<FailureState>,
    /// Total fetch failures recorded this session; see
    /// [`TileHealth::total_failures`].
    total_failures: u64,
    /// The most recent failure's message; see [`TileHealth::last_error`].
    last_error: Option<String>,
    /// On-screen size of one tile, in physical pixels, the meshes were built
    /// for.
    tile_size_px: f32,
    /// Integer zoom the meshes were built for; [`NO_ZOOM_BUCKET`] before the
    /// first frame.
    zoom_bucket: u16,
    /// Whether the caller still has to be told the meshes were invalidated.
    invalidated: bool,
    /// Re-tessellations left in the current frame.
    budget: usize,
}

/// Sentinel zoom bucket meaning "no frame has run yet"; outside the `u8` range
/// [`MapView::tile_zoom`] returns, so the first frame always counts as a change.
const NO_ZOOM_BUCKET: u16 = u16::MAX;

/// Shared, thread-safe half of the provider.
struct VectorInner {
    /// Caches, in-flight set and per-frame tessellation state.
    store: Mutex<VectorStore>,
    /// The paint program every tile is tessellated with.
    program: Arc<PaintProgram>,
    /// How the frame's label pass labels each source layer. Immutable for the
    /// life of the provider, like `program`: changing the style builds a new
    /// provider.
    labels: LabelTable,
    /// Context to wake when a mesh lands.
    ctx: egui::Context,
}

/// The handle a [`TileTransport`] reports vector-tile bytes through.
///
/// Cheap to clone and `Send + Sync`, so it travels to a worker thread or into a
/// `spawn_local` future. It is *not* [`crate::TileSink`]: the raster sink
/// decodes pixels, this one gunzips, decodes MVT and tessellates.
#[derive(Clone)]
pub struct VectorSink {
    /// Shared state and repaint handle.
    inner: Arc<VectorInner>,
}

impl core::fmt::Debug for VectorSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VectorSink").finish_non_exhaustive()
    }
}

impl VectorSink {
    /// Reports the outcome of the request for `tile`.
    ///
    /// Call this exactly once per [`TileTransport::request`], on success *or*
    /// failure — the tile stays marked in flight (and is therefore never
    /// re-requested) until it arrives.
    pub fn deliver(&self, tile: TileId, result: Result<Vec<u8>, TileError>) {
        self.inner.deliver_bytes(tile, result);
    }
}

impl TileDelivery for VectorInner {
    /// A sparse source's miss is not a failure: an EMPTY decoded tile keeps
    /// the renderer quiet (it simply has nothing to draw or label there),
    /// nothing is logged, and the failure LRU is untouched — an archive
    /// with ocean tiles missing would otherwise warn once per tile per
    /// session.
    fn deliver_absent(&self, tile: TileId) {
        {
            let mut store = self.store.lock();
            store.inflight.remove(&tile);
            store.failures.remove(&tile);
            store
                .decoded
                .insert(tile, Arc::new(oxigis_render::mvt::VectorTile::default()));
        }
        self.ctx.request_repaint();
    }

    fn deliver_bytes(&self, tile: TileId, result: Result<Vec<u8>, TileError>) {
        let decoded = match result {
            Ok(bytes) => decode_vector_tile(&bytes),
            Err(error) => Err(error),
        };
        // Read before locking `store`: egui's own lock (inside `input`) must
        // never nest inside this one — see `SinkInner::deliver_bytes` in
        // `tile_provider.rs` for why.
        let now = self.ctx.input(|input| input.time);

        let work = {
            let mut store = self.store.lock();
            store.inflight.remove(&tile);
            match decoded {
                Ok(vector_tile) => {
                    store.failures.remove(&tile);
                    let shared = Arc::new(vector_tile);
                    store.decoded.insert(tile, Arc::clone(&shared));
                    Some((shared, store.tile_size_px))
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
                        "oxigis-ui: vector tile fetch failed: {error}",
                    );
                    None
                }
            }
        };

        // Tessellation happens outside the lock, on whichever context finished
        // the transfer — a worker thread natively, the microtask queue on wasm.
        if let Some((vector_tile, tile_size_px)) = work {
            match self.tessellate(&vector_tile, tile_size_px) {
                Ok(mesh) => {
                    self.store.lock().meshes.insert(tile, mesh);
                }
                Err(error) => {
                    let mut store = self.store.lock();
                    store.failures.insert(tile, FailureState::permanent(now));
                    store.total_failures = store.total_failures.saturating_add(1);
                    store.last_error = Some(truncate_for_display(&error.to_string()));
                    tracing::warn!(
                        z = tile.z,
                        x = tile.x,
                        y = tile.y,
                        "oxigis-ui: vector tile tessellation failed: {error}",
                    );
                }
            }
        }
        // Outside the lock: `request_repaint` reaches into egui's own state.
        self.ctx.request_repaint();
    }
}

impl VectorInner {
    /// Tessellates one decoded tile for the given on-screen tile size.
    fn tessellate(&self, tile: &VectorTile, tile_size_px: f32) -> Result<VectorMesh, RenderError> {
        let params = tess_params(tile, tile_size_px)?;
        self.program.tessellate(tile, &params)
    }
}

/// Derives the tessellation parameters for one decoded tile.
///
/// The extent is read from the tile itself rather than assumed, so a source
/// using something other than 4096 still gets pixel-accurate stroke widths.
///
/// # Errors
///
/// Propagates [`RenderError::Tessellation`] from
/// [`TessParams::for_tile`] when the extent or the tile size is degenerate.
fn tess_params(tile: &VectorTile, tile_size_px: f32) -> Result<TessParams, RenderError> {
    let extent = tile
        .layers
        .first()
        .map_or(DEFAULT_EXTENT, |layer| layer.extent);
    TessParams::for_tile(tile_size_px, extent, TessParams::DEFAULT_TOLERANCE_PX)
}

/// Inflates a gzipped body, then decodes it as MVT.
///
/// # Errors
///
/// Returns a permanent [`TileError`] for a body that is neither valid gzip nor
/// valid MVT: neither gets better on a retry.
fn decode_vector_tile(bytes: &[u8]) -> Result<VectorTile, TileError> {
    let raw = if bytes.starts_with(&GZIP_MAGIC) {
        oxiarc_deflate::gzip_decompress(bytes)
            .map_err(|error| TileError::permanent(format!("gzip decode failed: {error}")))?
    } else {
        bytes.to_vec()
    };
    decode_mvt(&raw).map_err(|error| TileError::permanent(format!("MVT decode failed: {error}")))
}

/// A [`VectorTileSource`] backed by a real MVT tile service.
///
/// Construct one per vector layer and install it with
/// [`crate::map_gpu::replace_vector_source`]. See the [module docs][self] for
/// the frame protocol, the resource bounds and where the CPU work runs.
pub struct VectorTileProvider {
    /// Expands a [`TileId`] into the URL to fetch, or [`None`] when the
    /// transport addresses tiles inside one archive and there is no URL.
    template: Option<XyzTemplate>,
    /// Shared state, also handed to the transport through the sink.
    inner: Arc<VectorInner>,
    /// The platform's fetch capability.
    transport: Box<dyn TileTransport>,
}

impl core::fmt::Debug for VectorTileProvider {
    /// The transport is an opaque platform capability, so it is elided rather
    /// than forcing every implementation to be [`Debug`].
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VectorTileProvider")
            .field(
                "template",
                &self.template.as_ref().map(XyzTemplate::template),
            )
            .field("passes", &self.inner.program.passes().len())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl VectorTileProvider {
    /// Builds a provider for `config`, waking `ctx` whenever a mesh lands.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidTemplate`] if `config`'s URL template is
    /// not a usable `{z}/{x}/{y}` template, or [`RenderError::InvalidCapacity`]
    /// if the compile-time cache bounds are degenerate (unreachable with the
    /// constants in this module).
    pub fn new(
        config: &VectorTileConfig,
        ctx: &egui::Context,
        transport: Box<dyn TileTransport>,
    ) -> Result<Self, RenderError> {
        let template = config.template()?;
        let store = VectorStore {
            decoded: TileCache::new(DECODED_CACHE_TILES)?,
            meshes: TileCache::new(READY_CACHE_TILES)?,
            inflight: HashSet::new(),
            failures: TileCache::new(FAILURE_MEMORY_TILES)?,
            total_failures: 0,
            last_error: None,
            tile_size_px: oxigis_render::TILE_SIZE_PX as f32,
            zoom_bucket: NO_ZOOM_BUCKET,
            invalidated: false,
            budget: MAX_TESSELLATIONS_PER_FRAME,
        };
        Ok(Self {
            template,
            inner: Arc::new(VectorInner {
                store: Mutex::new(store),
                program: Arc::new(config.program()),
                labels: config.label_table(),
                ctx: ctx.clone(),
            }),
            transport,
        })
    }

    /// The URL template tiles are fetched from, or [`None`] for an
    /// archive-backed provider, which has none.
    #[must_use]
    pub fn template(&self) -> Option<&XyzTemplate> {
        self.template.as_ref()
    }

    /// A clone of the sink, for a transport that wants to report out of band.
    #[must_use]
    pub fn sink(&self) -> VectorSink {
        VectorSink {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Snapshot of what the provider is holding: meshes ready for the renderer,
    /// in-flight requests and remembered failures.
    #[must_use]
    pub fn stats(&self) -> TileProviderStats {
        let store = self.inner.store.lock();
        TileProviderStats {
            ready: store.meshes.len(),
            inflight: store.inflight.len(),
            failed: store.failures.len(),
        }
    }

    /// Number of decoded tiles held for re-tessellation.
    #[must_use]
    pub fn decoded_len(&self) -> usize {
        self.inner.store.lock().decoded.len()
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
    /// session gets a fresh attempt budget immediately. See
    /// [`crate::XyzTileProvider::retry_failed_tiles`] for the full rationale;
    /// this is its vector-tile twin.
    pub fn retry_failed_tiles(&self) {
        self.inner.store.lock().failures.clear();
    }

    /// Decides what asking for `tile` should do, holding the lock for exactly
    /// that decision.
    ///
    /// `now` is `ctx.input(|i| i.time)`, read by the caller *before* the
    /// store is locked (see `SinkInner::deliver_bytes` in `tile_provider.rs`
    /// for why egui's own lock must never nest inside it).
    ///
    /// Split out of [`VectorTileSource::mesh`] because the follow-up work —
    /// starting a request, or tessellating — must happen with the lock released:
    /// a transport may answer synchronously (which would deadlock), and
    /// tessellation is far too long to hold a mutex for.
    fn decide(&self, tile: TileId, now: f64) -> Next {
        let mut store = self.inner.store.lock();
        if let Some(mesh) = store.meshes.remove(&tile) {
            return Next::Ready(mesh);
        }
        if store.inflight.contains(&tile) {
            return Next::Wait;
        }
        if store
            .failures
            .get(&tile)
            .is_some_and(|state| !state.retry_ready(now))
        {
            // `get`, not `peek`: see the identical comment in
            // `XyzTileProvider::tile` — an on-screen tile that is backing
            // off must not age out of `FAILURE_MEMORY_TILES` for free.
            return Next::Wait;
        }
        if let Some(decoded) = store.decoded.get(&tile).map(Arc::clone) {
            if store.budget == 0 {
                return Next::Wait;
            }
            store.budget -= 1;
            return Next::Tessellate(decoded, store.tile_size_px);
        }
        if store.inflight.len() >= MAX_INFLIGHT_TILES {
            return Next::Wait;
        }
        store.inflight.insert(tile);
        Next::Fetch
    }

    /// The [`TileSink`] handed to the transport.
    ///
    /// The transport seam only ever moves bytes, so reusing it costs exactly
    /// this adapter and saves every shell a second transport implementation;
    /// which decoder the bytes reach is decided here.
    fn transport_sink(&self) -> TileSink {
        TileSink::from_delivery(Arc::clone(&self.inner) as Arc<dyn TileDelivery>)
    }
}

/// What [`VectorTileProvider::mesh`] decided to do, once the lock is released.
enum Next {
    /// Nothing to do; the answer is [`None`].
    Wait,
    /// The mesh was ready.
    Ready(VectorMesh),
    /// Re-tessellate this decoded tile at this on-screen tile size.
    Tessellate(Arc<VectorTile>, f32),
    /// Fetch the tile.
    Fetch,
}

impl VectorTileSource for VectorTileProvider {
    fn begin_frame(&self, view: MapView) -> bool {
        let mut store = self.inner.store.lock();
        store.budget = MAX_TESSELLATIONS_PER_FRAME;
        let bucket = u16::from(view.tile_zoom());
        if bucket != store.zoom_bucket {
            store.zoom_bucket = bucket;
            store.tile_size_px = view.tile_size_px();
            store.meshes.clear();
            store.invalidated = true;
        }
        // Reported exactly once, so a shell clears the renderer's meshes on the
        // frame the change happened and not on every frame after it.
        core::mem::take(&mut store.invalidated)
    }

    fn mesh(&self, tile: TileId) -> Option<VectorMesh> {
        // Read before locking `store`, for the same reason as in
        // `SinkInner::deliver_bytes` (`tile_provider.rs`).
        let now = self.inner.ctx.input(|input| input.time);
        // One short critical section, then the lock is released *before* the
        // transport or the tessellator is touched: `request` may hand the job to
        // a worker that immediately calls `deliver`, which takes this same lock,
        // and tessellation must not hold it either.
        let next = self.decide(tile, now);

        match next {
            Next::Wait => None,
            Next::Ready(mesh) => Some(mesh),
            Next::Tessellate(decoded, tile_size_px) => {
                match self.inner.tessellate(&decoded, tile_size_px) {
                    Ok(mesh) => Some(mesh),
                    Err(error) => {
                        let mut store = self.inner.store.lock();
                        store.failures.insert(tile, FailureState::permanent(now));
                        store.total_failures = store.total_failures.saturating_add(1);
                        store.last_error = Some(truncate_for_display(&error.to_string()));
                        drop(store);
                        tracing::warn!(
                            z = tile.z,
                            x = tile.x,
                            y = tile.y,
                            "oxigis-ui: vector tile re-tessellation failed: {error}",
                        );
                        None
                    }
                }
            }
            Next::Fetch => {
                // No template means an archive transport, which addresses tiles
                // by `TileId` inside one file: the sentinel URL is what it is
                // handed, and what any other transport would name in a refusal.
                let url = match &self.template {
                    Some(template) => match template.expand(tile) {
                        Ok(url) => url,
                        Err(error) => {
                            self.sink()
                                .deliver(tile, Err(TileError::permanent(error.to_string())));
                            return None;
                        }
                    },
                    None => crate::tile_provider::ARCHIVE_TILE_URL.to_owned(),
                };
                self.transport.request(tile, url, self.transport_sink());
                None
            }
        }
    }

    fn decoded(&self, tile: TileId) -> Option<Arc<VectorTile>> {
        // `get`, not `peek`: a tile the label pass asks for is on screen, so
        // refreshing its LRU recency is exactly right — it keeps the visible
        // working set resident against a zoom step's re-tessellation.
        self.inner.store.lock().decoded.get(&tile).map(Arc::clone)
    }

    fn label_table(&self) -> &LabelTable {
        &self.inner.labels
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BoxedVectorTileSource, DECODED_CACHE_TILES, DEMO_CENTROID_TEXT_FIELD, DEMO_GEOLINE_COLOR,
        DEMO_GEOLINE_LABEL_SIZE_PX, DEMO_GEOLINE_TEXT_FIELD, GZIP_MAGIC, MAPLIBRE_ATTRIBUTION,
        MAPLIBRE_DEMO_URL_TEMPLATE, MAX_TESSELLATIONS_PER_FRAME, VectorTileConfig,
        VectorTileProvider, VectorTileSource, config_for, decode_vector_tile, maplibre_demo_paints,
        tess_params,
    };
    use crate::tile_provider::{
        MAX_ATTEMPTS, MAX_INFLIGHT_TILES, RETRY_BASE_DELAY_SECS, TileError, TileSink, TileTransport,
    };
    use oxigis_core::{LayerStyle, LineStyle, VectorSource, VectorTilePaint};
    use oxigis_render::{LabelResolver as _, LonLat, MapView, TileId, VectorTile};
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// A minimal but real MVT tile: one `countries` layer with one triangular
    /// polygon on the standard 4096 extent, hand-encoded so the test needs no
    /// encoder. Built by `oxigis-render`'s own wire helpers would be circular,
    /// so the bytes are literal protobuf.
    fn sample_tile_bytes() -> Vec<u8> {
        // layer (field 3, message):
        //   version (15, varint) = 2
        //   name (1, string) = "countries"
        //   feature (2, message):
        //     type (3, enum) = 3 (POLYGON)
        //     geometry (4, packed uint32) = MoveTo(1) + (0,0),
        //                                   LineTo(2) + (2048,0),(0,2048),
        //                                   ClosePath
        //   extent (5, varint) = 4096
        let geometry: [u32; 8] = [
            9,    // MoveTo, count 1
            0,    // zigzag(0)
            0,    // zigzag(0)
            18,   // LineTo, count 2
            4096, // zigzag(2048)
            0,    // zigzag(0)
            0,    // zigzag(0)
            4096, // zigzag(2048)
        ];
        let mut geometry_bytes = Vec::new();
        for value in geometry {
            encode_varint(&mut geometry_bytes, u64::from(value));
        }
        encode_varint(&mut geometry_bytes, 15); // ClosePath, count 1

        let mut feature = Vec::new();
        feature.push(3 << 3); // field 3 (type), varint
        encode_varint(&mut feature, 3); // POLYGON
        feature.push((4 << 3) | 2); // field 4 (geometry), length-delimited
        encode_varint(&mut feature, geometry_bytes.len() as u64);
        feature.extend_from_slice(&geometry_bytes);

        let mut layer = Vec::new();
        layer.push((15 << 3) as u8); // field 15 (version), varint -> 120
        encode_varint(&mut layer, 2);
        layer.push((1 << 3) | 2); // field 1 (name), length-delimited
        let name = b"countries";
        encode_varint(&mut layer, name.len() as u64);
        layer.extend_from_slice(name);
        layer.push((2 << 3) | 2); // field 2 (feature), length-delimited
        encode_varint(&mut layer, feature.len() as u64);
        layer.extend_from_slice(&feature);
        layer.push(5 << 3); // field 5 (extent), varint
        encode_varint(&mut layer, 4096);

        let mut tile = Vec::new();
        tile.push((3 << 3) | 2); // field 3 (layers), length-delimited
        encode_varint(&mut tile, layer.len() as u64);
        tile.extend_from_slice(&layer);
        tile
    }

    /// Appends `value` as a protobuf base-128 varint.
    fn encode_varint(out: &mut Vec<u8>, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    /// What a [`ScriptedTransport`] does with a request.
    #[derive(Clone, Copy)]
    enum Reply {
        /// Answer immediately with [`sample_tile_bytes`].
        Tile,
        /// Answer immediately with the same tile, gzipped.
        Gzipped,
        /// Answer immediately with a body that is not MVT at all.
        Garbage,
        /// Answer immediately with a retryable failure.
        Transient,
        /// Never answer, leaving the tile in flight.
        Hang,
    }

    /// Test transport: answers synchronously and records every URL asked for.
    struct ScriptedTransport {
        reply: Reply,
        urls: Arc<Mutex<Vec<String>>>,
    }

    impl TileTransport for ScriptedTransport {
        fn request(&self, tile: TileId, url: String, sink: TileSink) {
            self.urls.lock().push(url);
            match self.reply {
                Reply::Tile => sink.deliver(tile, Ok(sample_tile_bytes())),
                Reply::Gzipped => {
                    let gzipped = oxiarc_deflate::gzip_compress(&sample_tile_bytes(), 6)
                        .expect("the fixture must compress");
                    sink.deliver(tile, Ok(gzipped));
                }
                Reply::Garbage => sink.deliver(tile, Ok(vec![0xDE, 0xAD, 0xBE, 0xEF])),
                Reply::Transient => sink.deliver(tile, Err(TileError::transient("boom"))),
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

    fn view(zoom: f64) -> MapView {
        MapView::new(LonLat::new(0.0, 0.0), zoom, [256.0, 256.0]).expect("a valid view")
    }

    fn provider(reply: Reply) -> (VectorTileProvider, Arc<Mutex<Vec<String>>>) {
        let urls = Arc::new(Mutex::new(Vec::new()));
        let transport = ScriptedTransport {
            reply,
            urls: Arc::clone(&urls),
        };
        let provider = VectorTileProvider::new(
            &VectorTileConfig::maplibre_demo(),
            &egui::Context::default(),
            Box::new(transport),
        )
        .expect("the demo config must build a provider");
        (provider, urls)
    }

    #[test]
    fn the_demo_config_points_at_maplibre_with_geometry_and_label_rules() {
        let config = VectorTileConfig::default();
        assert_eq!(config.url_template, MAPLIBRE_DEMO_URL_TEMPLATE);
        assert_eq!(config.attribution, MAPLIBRE_ATTRIBUTION);
        assert!(config.template().is_ok());

        let names: Vec<&str> = config
            .paints
            .iter()
            .map(|rule| rule.source_layer.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "countries",
                "geolines",
                "centroids",
                "geolines",
                "centroids"
            ]
        );
        assert!(matches!(config.paints[0].style, LayerStyle::Fill(_)));
        assert!(matches!(config.paints[1].style, LayerStyle::Line(_)));
        assert!(matches!(config.paints[2].style, LayerStyle::Circle(_)));
        assert!(matches!(config.paints[3].style, LayerStyle::Symbol(_)));
        assert!(matches!(config.paints[4].style, LayerStyle::Symbol(_)));

        // The outlined country fill means two tessellation passes; the symbol
        // rules contribute to neither.
        assert_eq!(config.program().passes().len(), 2);
        assert_eq!(config.program().passes()[0].len(), 3);
    }

    #[test]
    fn the_demo_config_labels_centroids_and_geolines_but_not_countries() {
        let table = VectorTileConfig::default().label_table();
        assert_eq!(table.len(), 2);
        let centroids = table.label_for("centroids").expect("a centroid spec");
        assert_eq!(centroids.text_property, DEMO_CENTROID_TEXT_FIELD);
        // `SymbolStyle::new`'s defaults: 12 px black on a 1 px white halo.
        assert_eq!(centroids.size_px, 12.0);
        assert_eq!(centroids.color, [0, 0, 0, 255]);
        assert_eq!(
            centroids.halo.map(|halo| (halo.color, halo.width_px)),
            Some(([255, 255, 255, 255], 1.0))
        );

        let geolines = table.label_for("geolines").expect("a geoline spec");
        assert_eq!(geolines.text_property, DEMO_GEOLINE_TEXT_FIELD);
        assert_eq!(geolines.size_px, DEMO_GEOLINE_LABEL_SIZE_PX);
        assert!(geolines.size_px < centroids.size_px);

        // Country polygons are labelled through their centroids, never twice.
        assert!(table.label_for("countries").is_none());
    }

    #[test]
    fn a_provider_exposes_its_label_table_and_decoded_tiles() {
        let (provider, _urls) = provider(Reply::Tile);
        assert_eq!(provider.label_table().len(), 2);

        let root = tile(0, 0, 0);
        let _ = provider.begin_frame(view(0.0));
        assert!(
            provider.decoded(root).is_none(),
            "nothing is decoded before the first fetch answers"
        );
        assert!(provider.mesh(root).is_none(), "the first ask fetches");
        let decoded = provider.decoded(root).expect("the delivered tile");
        assert_eq!(decoded.layers.len(), 1);
        assert_eq!(decoded.layers[0].name, "countries");
    }

    #[test]
    fn a_source_labels_nothing_unless_it_opts_in() {
        // The trait defaults exist so a geometry-only source needs no new code.
        struct MeshOnly;
        impl VectorTileSource for MeshOnly {
            fn begin_frame(&self, _view: MapView) -> bool {
                false
            }
            fn mesh(&self, _tile: TileId) -> Option<oxigis_render::VectorMesh> {
                None
            }
        }
        let source = MeshOnly;
        assert!(source.label_table().is_empty());
        assert!(source.decoded(tile(0, 0, 0)).is_none());
    }

    #[test]
    fn the_config_round_trips_through_the_core_layer_source() {
        let config = VectorTileConfig::maplibre_demo();
        match config.to_source() {
            VectorSource::MvtTiles {
                url_template,
                paints,
            } => {
                assert_eq!(url_template, MAPLIBRE_DEMO_URL_TEMPLATE);
                assert_eq!(paints, maplibre_demo_paints());
            }
            other => panic!("expected an MVT tile source, got {other:?}"),
        }
    }

    #[test]
    fn a_custom_config_keeps_its_url_and_attribution() {
        let config = VectorTileConfig::new("https://example.test/{z}/{x}/{y}.pbf")
            .with_attribution("Example")
            .with_paints(Vec::new());
        assert_eq!(config.attribution, "Example");
        assert!(config.paints.is_empty());
        assert!(config.program().is_empty());
        assert!(config.template().is_ok());
    }

    #[test]
    fn a_config_without_placeholders_is_rejected() {
        let config = VectorTileConfig::new("https://example.test/tile.pbf");
        assert!(config.template().is_err());
    }

    #[test]
    fn config_for_builds_the_paints_it_is_given_and_never_the_demo_table() {
        // The per-frame property: `desired_vector` calls this for every stored
        // MVT layer on every frame, so a config it builds must allocate the
        // caller's paints and nothing else. A bare style is distinguishable
        // from `maplibre_demo_paints()`, which is five rules.
        let one = vec![VectorTilePaint::new(
            "roads",
            LayerStyle::Line(LineStyle::new(DEMO_GEOLINE_COLOR, 1.0)),
        )];
        let custom = config_for("https://example.test/{z}/{x}/{y}.pbf", one.clone());
        assert_eq!(custom.paints, one);
        assert!(
            custom.attribution.is_empty(),
            "only the demo source credits"
        );
        assert!(custom.archive.is_none());
        assert!(custom.subdomains.is_empty());

        // An EMPTY paint list is the sharp case: `new(..).with_paints(vec![])`
        // built five rules and dropped them, and the result is indistinguishable
        // from this one — so only a zero-length answer proves nothing was built.
        let bare = config_for("https://example.test/{z}/{x}/{y}.pbf", Vec::new());
        assert!(bare.paints.is_empty());

        // The one rule the seam exists for still holds: the keyless demo
        // template carries its required credit line.
        let demo = config_for(MAPLIBRE_DEMO_URL_TEMPLATE, one.clone());
        assert_eq!(demo.attribution, MAPLIBRE_ATTRIBUTION);
        assert_eq!(demo.paints, one, "the stored paints win over the defaults");
    }

    #[test]
    fn styled_is_new_minus_the_demo_paints() {
        // Every other field must match `new`, or the two constructors have
        // drifted and `config_for` quietly changed meaning.
        let url = "https://example.test/{z}/{x}/{y}.pbf";
        let eager = VectorTileConfig::new(url);
        let lean = VectorTileConfig::styled(url, Vec::new());
        assert_eq!(lean.url_template, eager.url_template);
        assert_eq!(lean.subdomains, eager.subdomains);
        assert_eq!(lean.attribution, eager.attribution);
        assert_eq!(lean.archive, eager.archive);
        assert_eq!(eager.paints, maplibre_demo_paints());
        assert!(lean.paints.is_empty());
    }

    #[test]
    fn the_fixture_decodes_as_one_named_layer() {
        let decoded = decode_vector_tile(&sample_tile_bytes()).expect("the fixture must decode");
        assert_eq!(decoded.layers.len(), 1);
        assert_eq!(decoded.layers[0].name, "countries");
        assert_eq!(decoded.layers[0].extent, 4096);
        assert_eq!(decoded.layers[0].features.len(), 1);
    }

    #[test]
    fn a_gzipped_body_is_inflated_before_decoding() {
        let raw = sample_tile_bytes();
        let gzipped = oxiarc_deflate::gzip_compress(&raw, 6).expect("compression must succeed");
        assert_eq!(&gzipped[..2], &GZIP_MAGIC);
        let decoded = decode_vector_tile(&gzipped).expect("a gzipped tile must decode");
        assert_eq!(decoded.layers.len(), 1);
    }

    #[test]
    fn a_body_that_is_neither_gzip_nor_mvt_is_a_permanent_failure() {
        let error = decode_vector_tile(&[0xDE, 0xAD, 0xBE, 0xEF]).expect_err("must fail");
        assert!(!error.retryable());
        let error = decode_vector_tile(&[0x1f, 0x8b, 0x00]).expect_err("must fail");
        assert!(!error.retryable());
        assert!(error.message().contains("gzip"));
    }

    #[test]
    fn tess_params_read_the_extent_from_the_tile() {
        let decoded = decode_vector_tile(&sample_tile_bytes()).expect("decode");
        let params = tess_params(&decoded, 512.0).expect("params must build");
        assert!((params.pixels_per_tile_unit - 512.0 / 4096.0).abs() < 1e-9);
        // An empty tile falls back to the conventional extent rather than failing.
        let empty = VectorTile { layers: Vec::new() };
        assert!(tess_params(&empty, 256.0).is_ok());
    }

    #[test]
    fn the_first_frame_reports_an_invalidation_then_stays_quiet() {
        let (provider, _urls) = provider(Reply::Hang);
        assert!(provider.begin_frame(view(3.0)));
        assert!(!provider.begin_frame(view(3.2)), "same zoom bucket");
        assert!(provider.begin_frame(view(4.0)), "new zoom bucket");
        assert!(!provider.begin_frame(view(4.9)));
    }

    #[test]
    fn a_missing_tile_is_requested_once_and_answered_next_frame() {
        let (provider, urls) = provider(Reply::Tile);
        let _ = provider.begin_frame(view(0.0));
        let root = tile(0, 0, 0);
        // The scripted transport answers inside `request`, so the mesh is
        // already waiting when the next frame asks.
        assert!(provider.mesh(root).is_none());
        assert_eq!(
            urls.lock().as_slice(),
            ["https://demotiles.maplibre.org/tiles/0/0/0.pbf".to_string()]
        );
        let mesh = provider.mesh(root).expect("the delivered mesh");
        assert!(!mesh.is_empty(), "the fixture polygon must tessellate");
        assert_eq!(provider.stats().inflight, 0);
        assert_eq!(provider.decoded_len(), 1);
    }

    #[test]
    fn a_gzipped_tile_takes_the_same_path() {
        let (provider, _urls) = provider(Reply::Gzipped);
        let _ = provider.begin_frame(view(0.0));
        let root = tile(0, 0, 0);
        assert!(provider.mesh(root).is_none());
        assert!(provider.mesh(root).is_some());
    }

    #[test]
    fn a_zoom_step_re_tessellates_from_the_decoded_cache_without_refetching() {
        let (provider, urls) = provider(Reply::Tile);
        let _ = provider.begin_frame(view(0.0));
        let root = tile(0, 0, 0);
        assert!(provider.mesh(root).is_none());
        assert!(provider.mesh(root).is_some());
        assert_eq!(urls.lock().len(), 1);

        // A new bucket clears the meshes, but the decoded tile survives.
        assert!(provider.begin_frame(view(1.0)));
        assert!(
            provider.mesh(root).is_some(),
            "the mesh must be rebuilt from the decoded cache"
        );
        assert_eq!(urls.lock().len(), 1, "no second request");
    }

    #[test]
    fn re_tessellation_is_budgeted_per_frame() {
        let (provider, _urls) = provider(Reply::Tile);
        let _ = provider.begin_frame(view(4.0));
        let tiles: Vec<TileId> = (0..(MAX_TESSELLATIONS_PER_FRAME as u32 + 3))
            .map(|x| tile(4, x, 0))
            .collect();
        for id in &tiles {
            assert!(provider.mesh(*id).is_none(), "first ask starts the fetch");
            assert!(provider.mesh(*id).is_some(), "delivered mesh is ready");
        }
        // New bucket: every mesh is gone and must be rebuilt under the budget.
        assert!(provider.begin_frame(view(5.0)));
        let rebuilt = tiles
            .iter()
            .filter(|id| provider.mesh(**id).is_some())
            .count();
        assert_eq!(rebuilt, MAX_TESSELLATIONS_PER_FRAME);
    }

    #[test]
    fn an_inflight_tile_is_not_requested_twice() {
        let (provider, urls) = provider(Reply::Hang);
        let _ = provider.begin_frame(view(1.0));
        for _ in 0..5 {
            assert!(provider.mesh(tile(1, 0, 0)).is_none());
        }
        assert_eq!(urls.lock().len(), 1);
        assert_eq!(provider.stats().inflight, 1);
    }

    #[test]
    fn concurrency_is_capped() {
        let (provider, urls) = provider(Reply::Hang);
        let _ = provider.begin_frame(view(8.0));
        for x in 0..(MAX_INFLIGHT_TILES as u32 * 4) {
            assert!(provider.mesh(tile(8, x, 0)).is_none());
        }
        assert_eq!(urls.lock().len(), MAX_INFLIGHT_TILES);
    }

    #[test]
    fn an_undecodable_body_is_never_retried() {
        let (provider, urls) = provider(Reply::Garbage);
        let _ = provider.begin_frame(view(2.0));
        for _ in 0..10 {
            assert!(provider.mesh(tile(2, 1, 1)).is_none());
        }
        assert_eq!(urls.lock().len(), 1);
        assert_eq!(provider.stats().failed, 1);
    }

    #[test]
    fn a_transient_failure_backs_off_on_the_wall_clock_not_the_frame_count() {
        // Needs its own `Context` (not the `provider()` helper's throwaway
        // one) so the test can advance the clock `FailureState` reads.
        let ctx = egui::Context::default();
        let urls = Arc::new(Mutex::new(Vec::new()));
        let transport = ScriptedTransport {
            reply: Reply::Transient,
            urls: Arc::clone(&urls),
        };
        let provider = VectorTileProvider::new(
            &VectorTileConfig::maplibre_demo(),
            &ctx,
            Box::new(transport),
        )
        .expect("the demo config must build a provider");
        let _ = provider.begin_frame(view(5.0));
        let target = tile(5, 6, 7);

        assert!(provider.mesh(target).is_none());
        assert_eq!(urls.lock().len(), 1);

        // Many more frames at the SAME instant must not retry: the fix for
        // a one-second hiccup burning the whole attempt budget within three
        // frames (~50 ms at 60 fps).
        for _ in 0..50 {
            assert!(provider.mesh(target).is_none());
        }
        assert_eq!(
            urls.lock().len(),
            1,
            "a retry before its backoff delay must not touch the transport"
        );

        ctx.input_mut(|input| input.time = RETRY_BASE_DELAY_SECS);
        assert!(provider.mesh(target).is_none());
        assert_eq!(urls.lock().len(), 2);

        ctx.input_mut(|input| input.time = 1_000.0);
        for _ in 0..10 {
            let _ = provider.mesh(target);
        }
        assert_eq!(urls.lock().len(), MAX_ATTEMPTS as usize);
    }

    #[test]
    fn health_reports_a_monotonic_total_and_retry_failed_tiles_clears_the_gate() {
        let (provider, urls) = provider(Reply::Garbage);
        let _ = provider.begin_frame(view(2.0));
        let target = tile(2, 1, 1);
        assert_eq!(provider.health(), super::TileHealth::default());

        assert!(provider.mesh(target).is_none());
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
        assert!(provider.mesh(target).is_none());
        assert_eq!(urls.lock().len(), 2, "the tile must get a fresh attempt");
    }

    #[test]
    fn the_decoded_cache_is_bounded() {
        let (provider, _urls) = provider(Reply::Tile);
        let _ = provider.begin_frame(view(10.0));
        for x in 0..(DECODED_CACHE_TILES as u32 + 32) {
            let id = tile(10, x, 0);
            let _ = provider.mesh(id);
            let _ = provider.mesh(id);
        }
        assert_eq!(provider.decoded_len(), DECODED_CACHE_TILES);
    }

    #[test]
    fn the_provider_is_boxable_as_a_send_sync_seam() {
        let (provider, _urls) = provider(Reply::Hang);
        let boxed: BoxedVectorTileSource = Box::new(provider);
        let _ = boxed.begin_frame(view(0.0));
        assert!(boxed.mesh(tile(0, 0, 0)).is_none());
        assert!(format!("{:?}", VectorTileConfig::default()).contains("maplibre"));
    }

    #[test]
    fn no_tiled_config_can_carry_a_thematic_renderer() {
        // Thematic v1.6: an MVT paint is matched by source-layer NAME and
        // never sees a feature's attributes, so every shape of tiled config
        // refuses — the URL one, the archive one, and the demo one alike. A
        // combo that appeared to work here would rewrite the saved file and
        // change nothing on screen.
        let url = VectorTileConfig::new(MAPLIBRE_DEMO_URL_TEMPLATE);
        assert!(!url.supports_renderer());
        assert!(!VectorTileConfig::maplibre_demo().supports_renderer());
        assert!(
            !VectorTileConfig::styled("https://x/{z}/{x}/{y}.pbf", Vec::new()).supports_renderer()
        );
        assert!(!VectorTileConfig::default().supports_renderer());
        // And the refusal names its reason rather than being a bare `false`.
        assert!(super::TILED_RENDERER_REFUSAL.contains("attributes"));
        assert!(super::TILED_RENDERER_REFUSAL.contains("local vector layer"));
    }
}
