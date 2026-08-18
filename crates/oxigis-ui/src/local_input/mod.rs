// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Local-vector *input*: turning dropped files, a pasted GeoJSON document or a
//! loaded project into [`oxigis_core::Layer`] entries plus the work a shell has
//! to hand to the GPU.
//!
//! Three formats arrive here — GeoJSON (§1.1), ESRI Shapefile and OGC
//! GeoPackage (§1.3) — and they converge immediately: all become `oxigeo`
//! `FeatureCollection`s, and from `LocalInputState::add_feature_collection`
//! onwards nothing downstream knows which was which. A shapefile is also the
//! reason [`DroppedItem`] and [`group_dropped_files`] exist: it arrives as up
//! to five separate files in one drop and has to be reassembled before anything
//! can be read. A GeoPackage is the opposite shape of problem — one file, many
//! feature tables, hence many layers — which is why
//! [`LocalInputState::add_gpkg`] is the only entry point here that returns more
//! than one layer.
//!
//! # Why a queue at all
//!
//! Every `map_gpu` entry point for local layers needs an
//! `eframe::egui_wgpu::RenderState`, which [`crate::OxigisApp`] deliberately
//! does not hold (see `app`'s module docs: the shell owns `eframe`, the app owns

use crate::local_vector::{LocalVectorError, LocalVectorLayer, MercatorSquare};
use crate::shapefile_input::ShapefileBytes;
use oxigeo::geojson::types::FeatureCollection;
use oxigis_core::{
    ArchiveFormat, Crs, Layer, LayerId, LayerKind, LayerStyleSet, Project, VectorSource,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
/// Inline GeoJSON above this many bytes gets a status-line warning when it is
/// added: the text is embedded verbatim in the saved `.oxigis.json`, so a
/// pasted or browser-dropped dataset this large makes the project file at least
/// this large too. Purely advisory — the layer is added either way, because a
/// browser drop has no path to fall back to.
pub const INLINE_GEOJSON_WARN_BYTES: usize = 1 << 20;
/// Fraction of the shorter viewport dimension left free on each side when
/// zooming to a freshly added layer.
pub const ZOOM_TO_LAYER_MARGIN: f32 = 0.06;
/// One unit of GPU-side work on the local vector stack, queued by
/// [`crate::OxigisApp`] and performed by the shell against `map_gpu`.
///
/// Ops must be applied **in order**; see the module docs for why additions and
/// edits share one queue.
#[derive(Debug, Clone)]
pub enum LocalLayerOp {
    /// Attach (or replace, for a known id) a parsed dataset.
    ///
    /// Boxed because a [`LocalVectorLayer`] owns a whole synthetic vector tile
    /// and would otherwise make every variant of this enum that large.
    Add(LayerId, Box<LocalVectorLayer>),
    /// Detach the dataset registered under this id.
    Remove(LayerId),
    /// Show or hide the dataset without dropping its mesh.
    SetVisibility(LayerId, bool),
    /// Set the dataset's opacity (`0..=1`, clamped by the renderer).
    SetOpacity(LayerId, f32),
    /// Restyle the dataset; its mesh is rebuilt on the next frame.
    SetStyle(LayerId, LayerStyleSet),
    /// Reorder the local stack to match this id list (see
    /// [`crate::map_gpu::reorder_local_vector_layers`] — non-local ids are
    /// ignored, omitted local layers keep their relative order).
    Reorder(Vec<LayerId>),
    /// Detach every dataset — File ▸ New and the project-load path.
    Clear,
}
/// What [`LocalInputState::add_geojson`] produced, so the caller can select the
/// new layer, zoom to it, and report it in the status line.
#[derive(Debug, Clone, Copy)]
pub struct AddedLocalLayer {
    /// Id of the layer just appended to the project.
    pub id: LayerId,
    /// The dataset's Mercator extent — what "zoom to layer" fits.
    pub square: MercatorSquare,
    /// Number of features parsed.
    pub feature_count: usize,
    /// Size of the GeoJSON text if it was embedded in the project
    /// ([`VectorSource::InlineGeoJson`]), or [`None`] when the layer only
    /// references a path.
    pub inline_bytes: Option<usize>,
}
/// What [`LocalInputState::add_gpkg`] produced: one entry per feature table
/// that became a layer, plus the refusals for the ones that did not.
///
/// A GeoPackage is the only drop that yields several layers at once, which is
/// why it needs its own return type: the caller selects and zooms to the
/// *first* layer and reports the rest.
#[derive(Debug, Clone)]
pub struct AddedGpkg {
    /// The layers appended, in the file's own table order. Never empty — a
    /// GeoPackage that produced none is an error instead.
    pub layers: Vec<AddedLocalLayer>,
    /// Why the file's other feature tables were left out (an unsupported CRS,
    /// an unreadable table); empty when everything loaded.
    pub notices: Vec<String>,
}
/// A file a native shell still has to read, and what to do with the bytes.
///
/// The `layer` field is the whole point: a **fresh drop** ([`None`]) becomes a
/// new project layer, selected and zoomed to, while a **project-load
/// reference** ([`Some`]) must rebuild the layer that is *already* in the
/// project — same id, same saved style, same visibility — without appending a
/// duplicate or moving the camera away from the project's saved view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingPath {
    /// The existing layer these bytes belong to, or [`None`] for a fresh drop.
    pub layer: Option<LayerId>,
    /// The file to read.
    pub path: PathBuf,
    /// For a [`VectorSource::LocalGpkg`] reference, the feature table inside
    /// the file this layer was read from; [`None`] for every other source.
    ///
    /// A `.gpkg` is the one format where the path does not identify the data:
    /// the file holds several feature tables and each is its own layer, so a
    /// reload has to be told which one to rebuild. A *fresh* GeoPackage drop
    /// carries [`None`] and imports every table.
    pub table: Option<String>,
}
/// Whether a vector source is one this crate can load locally: GeoJSON held
/// inline or referenced by path, or a shapefile, GeoPackage table or
/// GeoParquet file referenced by path.
///
/// [`VectorSource::LocalShapefile`] and [`VectorSource::LocalGpkg`] joined the
/// set in Phase 1 §1.3; [`VectorSource::LocalGeoParquet`] joined it in the
/// same phase's GeoParquet stage. It is the predicate every "does the
/// GPU-side local stack own this layer?" question in the app funnels through,
/// so a source missing from here silently loses its style edits, its
/// attribute table and its reorder handling — see [`is_local_layer`]'s
/// callers. Only [`VectorSource::MvtTiles`] is excluded, because a tiled
/// source is drawn by [`crate::vector_provider`] instead.
///
/// Unconditional: [`VectorSource::LocalGeoParquet`] is included here even on
/// a build without the `geoparquet` Cargo feature, so a project file that
/// references one is still recognised as "a local layer that failed to
/// load" (see [`LocalInputState::rebuild_from_project`]) rather than treated
/// as some other, unrecognised source.
#[must_use]
pub fn is_local_vector_source(source: &VectorSource) -> bool {
    matches!(
        source,
        VectorSource::InlineGeoJson { .. }
            | VectorSource::LocalGeoJson { .. }
            | VectorSource::LocalShapefile { .. }
            | VectorSource::LocalGpkg { .. }
            | VectorSource::LocalGeoParquet { .. }
    )
}
/// Whether this layer is a local vector layer — the ones whose GPU mirror lives
/// in [`crate::local_layers::LocalVectorRenderer`].
#[must_use]
pub fn is_local_layer(layer: &Layer) -> bool {
    match &layer.kind {
        LayerKind::Vector(source) => is_local_vector_source(source),
        LayerKind::Raster(_) => false,
    }
}
/// The ids of a project's local layers, **in storage order**.
///
/// [`oxigis_core::LayerStack::layers`] stores back-to-front and the local
/// renderer paints in insert order, so this order is directly the one
/// [`LocalLayerOp::Reorder`] wants. (The layer panel lists the reverse; do not
/// take that order.)
#[must_use]
pub fn local_layer_order(project: &Project) -> Vec<LayerId> {
    project
        .layers
        .layers()
        .iter()
        .filter(|layer| is_local_layer(layer))
        .map(|layer| layer.id)
        .collect()
}
/// Whether a file name looks like GeoJSON.
///
/// Case-insensitive, and `.json` is accepted as well as `.geojson` because
/// plenty of real datasets ship as plain `.json`; a `.json` file that turns out
/// not to be a `FeatureCollection` fails at parse time with a message, not here.
/// This alone still answers `true` for a name ending `.geolibre.json` (it
/// does end `.json`); [`classify_drop`], the actual drag-and-drop router,
/// checks that more specific suffix first and returns
/// [`DropKind::GeoLibreProject`] instead — this function is not the one that
/// makes that distinction.
#[must_use]
pub fn looks_like_geojson(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".geojson") || lower.ends_with(".json")
}
/// The five extensions that make up a shapefile set, in the order the grouper
/// reports them.
///
/// `.shx` is accepted so a full five-file drop is not reported as junk, but its
/// bytes are never used: the reader walks the `.shp` record chain directly, and
/// a sequential read needs no index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapefilePart {
    /// `.shp` — the geometry, and the only mandatory member.
    Shp,
    /// `.dbf` — the attribute table; optional (a lone `.shp` loads geometry).
    Dbf,
    /// `.prj` — the CRS as WKT; optional (absent means WGS 84).
    Prj,
    /// `.cpg` — the DBF code page label; optional.
    Cpg,
    /// `.shx` — the record index; accepted and ignored.
    Shx,
}
/// What a dropped file's name says it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropKind {
    /// `.geojson` / `.json` — [`looks_like_geojson`]. A name ending
    /// `.geolibre.json` is classified as [`DropKind::GeoLibreProject`]
    /// instead, even though it also ends `.json`.
    GeoJson,
    /// `.geolibre.json` — a GeoLibre project document (see
    /// [`crate::geolibre_import`]), routed to project import rather than
    /// added as a dataset — see `crate::app::OxigisApp`'s drop handling.
    GeoLibreProject,
    /// One member of a shapefile set.
    Shapefile(ShapefilePart),
    /// `.gpkg` — an OGC GeoPackage, self-contained (see
    /// [`crate::gpkg_input`]).
    GeoPackage,
    /// `.parquet` / `.geoparquet` — a GeoParquet file, self-contained.
    ///
    /// Recognised **unconditionally**, on every build, even though reading
    /// one needs the native-only `geoparquet` Cargo feature: a browser build
    /// (which never enables it) can then tell the user clearly that
    /// GeoParquet is not supported there, instead of falling through to
    /// [`DropKind::Unsupported`]'s generic "not a supported file type"
    /// notice or, worse, misparsing the bytes as something else.
    GeoParquet,
    /// `.pmtiles` / `.mbtiles` — a single-file tile archive.
    ///
    /// Recognised **unconditionally**, like [`DropKind::GeoParquet`]: which
    /// container it is decides how it can be read (a browser can hold a
    /// dropped `.pmtiles`'s bytes in memory; an `.mbtiles` needs the SQLite
    /// reader), and saying so by name beats the generic "not a supported file
    /// type" notice.
    TileArchive(ArchiveFormat),
    /// Something this build cannot read.
    Unsupported,
}
/// Classifies a dropped file by its extension, case-insensitively.
///
/// Extension only, never content: a `.json` that turns out not to be a
/// `FeatureCollection` fails at parse time with a message, and a `.shp` is
/// binary so there is nothing cheap to sniff.
#[must_use]
pub fn classify_drop(name: &str) -> DropKind {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".geolibre.json") {
        return DropKind::GeoLibreProject;
    }
    if looks_like_geojson(name) {
        return DropKind::GeoJson;
    }
    if lower.ends_with(".gpkg") {
        return DropKind::GeoPackage;
    }
    if lower.ends_with(".parquet") || lower.ends_with(".geoparquet") {
        return DropKind::GeoParquet;
    }
    if let Some(format) = ArchiveFormat::from_file_name(&lower) {
        return DropKind::TileArchive(format);
    }
    for (suffix, part) in [
        (".shp", ShapefilePart::Shp),
        (".dbf", ShapefilePart::Dbf),
        (".prj", ShapefilePart::Prj),
        (".cpg", ShapefilePart::Cpg),
        (".shx", ShapefilePart::Shx),
    ] {
        if lower.ends_with(suffix) {
            return DropKind::Shapefile(part);
        }
    }
    DropKind::Unsupported
}
/// A file name with its extension removed, lower-cased — the key a shapefile
/// set is grouped on.
#[must_use]
pub fn file_stem(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    match lower.rfind('.') {
        Some(dot) if dot > 0 => lower[..dot].to_string(),
        _ => lower,
    }
}
/// One file from a drop, normalised out of egui's `DroppedFile`.
///
/// Exactly one of `bytes` (a browser drop) and `path` (an `egui-winit` native
/// drop) is normally set; both being absent is the "arrived without any data"
/// case the app reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedItem {
    /// The file's display name, already reduced by [`display_name`].
    pub name: String,
    /// The file's contents, when the host attached them.
    pub bytes: Option<Arc<[u8]>>,
    /// The file's path, when the host gave one instead.
    pub path: Option<PathBuf>,
}
/// A shapefile set assembled out of one drop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapefileDrop {
    /// The `.shp` — the member the layer is named and sourced from.
    pub shp: DroppedItem,
    /// The `.dbf`, if it came along. Without it the layer is geometry-only.
    pub dbf: Option<DroppedItem>,
    /// The `.prj`, if it came along. Without it WGS 84 is assumed.
    pub prj: Option<DroppedItem>,
    /// The `.cpg`, if it came along.
    pub cpg: Option<DroppedItem>,
}
/// One loadable dataset out of a drop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DroppedDataset {
    /// A single GeoJSON document.
    GeoJson(DroppedItem),
    /// A shapefile set, its members already paired by stem.
    Shapefile(Box<ShapefileDrop>),
    /// A GeoPackage. Unlike a shapefile it needs no grouping — one file holds
    /// the geometry, the attributes, the CRS definitions and the catalogue —
    /// but it may still become *several* layers, one per feature table.
    GeoPackage(DroppedItem),
    /// A GeoParquet file. Self-contained like a GeoPackage, but always
    /// becomes exactly one layer, like a shapefile.
    GeoParquet(DroppedItem),
    /// A GeoLibre project document (`*.geolibre.json`) — replaces the whole
    /// project instead of becoming a layer; see [`crate::geolibre_import`].
    GeoLibreProject(DroppedItem),
    /// A single-file tile archive. Self-contained like a GeoPackage, and — like
    /// a `.shp` — always becomes exactly one layer, but *which kind* of layer
    /// is only known once its header has been read (see `crate::archive`).
    TileArchive(ArchiveFormat, DroppedItem),
}
/// Sorts one drop's files into loadable datasets, plus the notices to show for
/// the files that were not loadable.
///
/// A shapefile arrives as **several** `DroppedFile`s in a single drop, so this
/// has to see the whole batch at once: members are paired by their file stem
/// (case-insensitively, since `CITIES.SHP` + `cities.dbf` is a real thing on
/// Windows-authored data). Rules:
///
/// * every `.geojson`/`.json` is its own dataset, in drop order — *except* a
///   name ending `.geolibre.json`, which is routed to project import instead
///   (see [`DropKind::GeoLibreProject`]);
/// * every `.gpkg` likewise — it is self-contained;
/// * every `.parquet`/`.geoparquet` likewise — also self-contained, and
///   recognised regardless of whether this build can actually read one (see
///   [`DropKind::GeoParquet`]);
/// * a `.shp` becomes a dataset, taking whichever `.dbf`/`.prj`/`.cpg` share
///   its **source folder and** stem;
/// * a companion **without** its `.shp` is ignored with a notice — dropping a
///   lone `.dbf` is a mistake, not a request to load an attribute table;
/// * anything else gets an "unsupported file type" notice.
///
/// The folder is half the key because the stem alone is not enough:
/// `region_a/roads.shp` + siblings and `region_b/roads.shp` + siblings dragged
/// into a single drop is ordinary usage, and every item arriving here has
/// already had its name reduced to a bare file name by [`display_name`]. On a
/// stem-only key both sets collapse into one, every companion of both lands on
/// whichever `.shp` came last, and the other shapefile loads geometry-only in
/// WGS 84 with nothing said about it. A browser drop really does carry bare
/// names and no folder anywhere, so there the collision is unresolvable and
/// gets a notice — the companions are left off *both* sets rather than
/// attached to an arbitrary one.
///
/// Datasets come back in the order their `.shp`/`.geojson` was dropped, so
/// several files land in a predictable stacking order.
#[must_use]
pub fn group_dropped_files(items: Vec<DroppedItem>) -> (Vec<DroppedDataset>, Vec<String>) {
    let mut datasets = Vec::new();
    let mut notices = Vec::new();
    let mut sets: BTreeMap<ShapefileSetKey, ShapefileSlot> = BTreeMap::new();
    let mut orphans: Vec<(ShapefileSetKey, ShapefilePart, DroppedItem)> = Vec::new();
    for item in items {
        match classify_drop(&item.name) {
            DropKind::GeoJson => datasets.push(DroppedDataset::GeoJson(item)),
            DropKind::GeoLibreProject => datasets.push(DroppedDataset::GeoLibreProject(item)),
            DropKind::GeoPackage => datasets.push(DroppedDataset::GeoPackage(item)),
            DropKind::GeoParquet => datasets.push(DroppedDataset::GeoParquet(item)),
            DropKind::TileArchive(format) => {
                datasets.push(DroppedDataset::TileArchive(format, item));
            }
            DropKind::Shapefile(ShapefilePart::Shp) => {
                match sets.entry(shapefile_set_key(&item)) {
                    std::collections::btree_map::Entry::Vacant(slot) => {
                        slot.insert(ShapefileSlot::One(datasets.len()));
                    }
                    std::collections::btree_map::Entry::Occupied(mut slot) => {
                        // Only reachable for a directory-less (browser) drop:
                        // two native paths that agree on both folder and stem
                        // are the same file. Said once, on the transition.
                        if matches!(slot.get(), ShapefileSlot::One(_)) {
                            notices.push(format!(
                                "More than one {} was dropped at once and a browser drop carries \
                                 no folder to tell them apart, so the .dbf/.prj/.cpg files of \
                                 that name were left off all of them.",
                                item.name,
                            ));
                        }
                        slot.insert(ShapefileSlot::Ambiguous);
                    }
                }
                datasets.push(DroppedDataset::Shapefile(Box::new(ShapefileDrop {
                    shp: item,
                    dbf: None,
                    prj: None,
                    cpg: None,
                })));
            }
            DropKind::Shapefile(ShapefilePart::Shx) => {
                orphans.push((shapefile_set_key(&item), ShapefilePart::Shx, item));
            }
            DropKind::Shapefile(part) => orphans.push((shapefile_set_key(&item), part, item)),
            DropKind::Unsupported => notices.push(format!(
                "{} is not a supported file type (.geojson, .json, .shp, .gpkg, .parquet, \
                 .pmtiles or .mbtiles).",
                item.name,
            )),
        }
    }
    for (key, part, item) in orphans {
        let dataset = match sets.get(&key) {
            Some(ShapefileSlot::One(index)) => datasets.get_mut(*index),
            // Attaching this companion to an arbitrary one of the same-named
            // sets is exactly the misattribution the key exists to prevent, and
            // the notice pushed above already told the user it was left off.
            Some(ShapefileSlot::Ambiguous) => continue,
            None => None,
        };
        let slot = dataset.and_then(|dataset| match dataset {
            DroppedDataset::Shapefile(set) => Some(set),
            DroppedDataset::GeoJson(_)
            | DroppedDataset::GeoPackage(_)
            | DroppedDataset::GeoParquet(_)
            | DroppedDataset::TileArchive(_, _)
            | DroppedDataset::GeoLibreProject(_) => None,
        });
        match (slot, part) {
            (Some(set), ShapefilePart::Dbf) => set.dbf = Some(item),
            (Some(set), ShapefilePart::Prj) => set.prj = Some(item),
            (Some(set), ShapefilePart::Cpg) => set.cpg = Some(item),
            (Some(_), _) => {}
            (None, _) => notices.push(format!(
                "{} was dropped without its .shp file, so it was ignored.",
                item.name,
            )),
        }
    }
    (datasets, notices)
}
/// What pairs one drop's shapefile members together: the folder the file came
/// from, when the host gave one, and its case-insensitive stem. See
/// [`group_dropped_files`] for why the folder is half of it.
type ShapefileSetKey = (Option<PathBuf>, String);
/// Which shapefile set a [`ShapefileSetKey`] resolves to while a drop is being
/// grouped.
enum ShapefileSlot {
    /// Exactly one `.shp` claimed the key; its companions attach to it.
    One(usize),
    /// Several did, so no companion of that key can be attributed to one.
    Ambiguous,
}
/// The [`ShapefileSetKey`] of one dropped file.
fn shapefile_set_key(item: &DroppedItem) -> ShapefileSetKey {
    let directory = item
        .path
        .as_ref()
        .map(|path| path.parent().map_or_else(PathBuf::new, Path::to_path_buf));
    (directory, file_stem(&item.name))
}
/// A human-usable layer name for a dropped path or file name: the final path
/// segment, or the whole string when it has none.
#[must_use]
pub fn display_name(raw: &str) -> String {
    raw.rsplit(['/', '\\'])
        .find(|segment| !segment.is_empty())
        .unwrap_or(raw)
        .to_string()
}
/// Parses GeoJSON text into a collection, refusing an empty one.
///
/// The single definition of "is this text a usable dataset?", shared by the
/// add and hydrate paths so a document that is rejected on a drop is also
/// rejected on a project load. An empty `FeatureCollection` is indistinguishable
/// from a failed drop on screen, so it is reported as one — the same rule
/// [`LocalVectorLayer::from_geojson`] applies.
///
/// # Errors
///
/// Returns [`LocalVectorError`] when the text is not a GeoJSON
/// `FeatureCollection`, or when it holds no features.
pub fn parse_geojson(text: &str) -> Result<FeatureCollection, LocalVectorError> {
    let features = oxigeo::geojson::reader::feature_collection_from_str(text)
        .map_err(|error| LocalVectorError::new(format!("GeoJSON parse failed: {error}")))?;
    if features.features.is_empty() {
        return Err(LocalVectorError::new("the GeoJSON holds no features"));
    }
    Ok(features)
}
/// [`parse_geojson`] for a document that is already a [`serde_json::Value`] —
/// what an in-process producer (the Processing executors) hands back.
///
/// Exactly the same rule, reached by the same route: `feature_collection_from_str`
/// **is** `serde_json::from_str::<FeatureCollection>`, so `from_value` is its
/// value-shaped twin rather than a second, drifting parser. The refusal wording
/// is shared verbatim, because a user who pastes a document and a user who runs
/// a tool must be told the same thing about the same defect.
///
/// The point is what it does *not* do: routing a tool result through
/// [`parse_geojson`] means serialising the value to text and parsing that text
/// straight back, so the document exists four times over (the executor's
/// collection, its `Value`, the text, and the re-parsed collection) at the peak
/// of adding one layer. This consumes the `Value` instead.
///
/// # Errors
///
/// Returns [`LocalVectorError`] when the value is not a GeoJSON
/// `FeatureCollection`, or when it holds no features.
pub fn parse_geojson_value(
    value: serde_json::Value,
) -> Result<FeatureCollection, LocalVectorError> {
    let features: FeatureCollection = serde_json::from_value(value)
        .map_err(|error| LocalVectorError::new(format!("GeoJSON parse failed: {error}")))?;
    if features.features.is_empty() {
        return Err(LocalVectorError::new("the GeoJSON holds no features"));
    }
    Ok(features)
}
/// The app-side half of the local-vector input path: the pending GPU work, the
/// native paths still to be read, and the default style each local layer was
/// created with.
#[derive(Debug)]
pub struct LocalInputState {
    /// GPU work waiting for a shell, in application order.
    ops: Vec<LocalLayerOp>,
    /// Dropped/loaded files a native shell still has to read.
    paths: Vec<PendingPath>,
    /// The style each local layer was born with, so removing its project style
    /// entry can restore something sensible rather than freezing the last edit.
    default_styles: BTreeMap<LayerId, LayerStyleSet>,
    /// Which geometry families each stored collection draws — maintained
    /// beside every `feature_sets` write (and dropped by [`Self::forget`]),
    /// so the style panel can offer the family row without walking a
    /// million-feature collection per frame.
    families: BTreeMap<LayerId, oxigis_core::FamilySet>,
    /// The parsed features of each local layer, shared with the copy that went
    /// to the GPU — the attribute table's data source. See this module's
    /// "app-side feature store" section.
    feature_sets: BTreeMap<LayerId, Arc<FeatureCollection>>,
    /// Whether this build can have paths read for it at all (false on wasm).
    /// A field rather than a bare `cfg!` so both branches are reachable — and
    /// therefore testable — in one native test run.
    paths_supported: bool,
}
impl LocalInputState {
    /// Creates the state for the current target: native builds accept file
    /// paths, `wasm32` builds do not (a browser hands over bytes instead).
    #[must_use]
    pub fn new() -> Self {
        Self::with_path_support(!cfg!(target_arch = "wasm32"))
    }
    /// Same, with the path capability chosen explicitly — how a native test
    /// exercises the browser branch.
    #[must_use]
    pub fn with_path_support(paths_supported: bool) -> Self {
        Self {
            ops: Vec::new(),
            paths: Vec::new(),
            default_styles: BTreeMap::new(),
            families: BTreeMap::new(),
            feature_sets: BTreeMap::new(),
            paths_supported,
        }
    }
    /// Whether this build can read layers referenced only by a filesystem path.
    #[must_use]
    pub fn paths_supported(&self) -> bool {
        self.paths_supported
    }
    /// Queues one op, coalescing repeated per-layer edits.
    ///
    /// A style or opacity slider drag emits an edit every frame while the shell
    /// may drain only once, and a `SetStyle` costs a full synchronous
    /// re-tessellation — so a later edit of the same kind for the same layer
    /// *replaces* the undrained earlier one instead of stacking behind it. The
    /// visible result is identical (last write wins) at a bounded cost.
    pub fn queue(&mut self, op: LocalLayerOp) {
        let replaced = match &op {
            LocalLayerOp::SetStyle(id, _) => self.coalesce_target(Some(*id), |existing| {
                matches!(existing, LocalLayerOp::SetStyle(..))
            }),
            LocalLayerOp::SetOpacity(id, _) => self.coalesce_target(Some(*id), |existing| {
                matches!(existing, LocalLayerOp::SetOpacity(..))
            }),
            LocalLayerOp::SetVisibility(id, _) => self.coalesce_target(Some(*id), |existing| {
                matches!(existing, LocalLayerOp::SetVisibility(..))
            }),
            LocalLayerOp::Reorder(_) => self.coalesce_target(None, |existing| {
                matches!(existing, LocalLayerOp::Reorder(_))
            }),
            _ => None,
        };
        match replaced {
            Some(index) => self.ops[index] = op,
            None => self.ops.push(op),
        }
    }
    /// Index of the queued op `matches` should be folded into, scanning
    /// backwards.
    ///
    /// The scan stops at any op that changes what "this layer" *is* — a
    /// [`LocalLayerOp::Clear`], or an `Add`/`Remove` of `id` — because an edit
    /// queued after a remove-and-re-add must not be folded into the slot in
    /// front of it, where it would apply to the dataset that no longer exists.
    /// `id` of [`None`] (the whole-stack ops) stops only at `Clear`.
    fn coalesce_target(
        &self,
        id: Option<LayerId>,
        matches: impl Fn(&LocalLayerOp) -> bool,
    ) -> Option<usize> {
        for (index, existing) in self.ops.iter().enumerate().rev() {
            let barrier = match existing {
                LocalLayerOp::Clear => true,
                LocalLayerOp::Add(other, _) | LocalLayerOp::Remove(other) => id == Some(*other),
                _ => false,
            };
            if barrier {
                return None;
            }
            let same_layer = match (id, existing) {
                (None, _) => true,
                (
                    Some(id),
                    LocalLayerOp::SetStyle(other, _)
                    | LocalLayerOp::SetOpacity(other, _)
                    | LocalLayerOp::SetVisibility(other, _),
                ) => id == *other,
                _ => false,
            };
            if same_layer && matches(existing) {
                return Some(index);
            }
        }
        None
    }
    /// Takes every queued op, leaving the queue empty. Shells call this once
    /// per frame and apply the ops in the order returned.
    pub fn take_ops(&mut self) -> Vec<LocalLayerOp> {
        core::mem::take(&mut self.ops)
    }
    /// Number of ops currently waiting (diagnostics and tests).
    #[must_use]
    pub fn pending_op_count(&self) -> usize {
        self.ops.len()
    }
    /// Queues a filesystem path for a native shell to read.
    ///
    /// `layer` names the *existing* project layer the bytes belong to (the
    /// project-load path); [`None`] means a fresh drop, which becomes a new
    /// layer. See [`PendingPath`] for why the distinction is load-bearing.
    pub fn queue_path(&mut self, layer: Option<LayerId>, path: PathBuf) {
        self.paths.push(PendingPath {
            layer,
            path,
            table: None,
        });
    }
    /// Queues a GeoPackage path whose bytes must be re-imported as one named
    /// feature table — see [`PendingPath::table`].
    pub fn queue_gpkg_path(&mut self, layer: Option<LayerId>, path: PathBuf, table: String) {
        self.paths.push(PendingPath {
            layer,
            path,
            table: Some(table),
        });
    }
    /// Takes every queued path, leaving the queue empty.
    pub fn take_paths(&mut self) -> Vec<PendingPath> {
        core::mem::take(&mut self.paths)
    }
    /// Number of paths currently waiting to be read.
    #[must_use]
    pub fn pending_path_count(&self) -> usize {
        self.paths.len()
    }
    /// Rebuilds an *existing* project layer from the bytes a shell just read
    /// for it, keeping its id, its saved style, and its visibility/opacity.
    ///
    /// The counterpart of [`Self::add_geojson`]: nothing is appended to the
    /// project, because the layer is already in it — this only restores the GPU
    /// copy a [`PendingPath`] with a `layer` was queued for.
    ///
    /// # Errors
    ///
    /// Propagates [`LocalVectorError`] when the text is not a usable GeoJSON
    /// `FeatureCollection`, or when `id` names no layer of `project`.
    pub fn hydrate_geojson(
        &mut self,
        project: &Project,
        id: LayerId,
        text: &str,
    ) -> Result<(), LocalVectorError> {
        self.hydrate_features(project, id, parse_geojson(text)?)
    }
    /// Rebuilds an existing project layer from the bytes of a shapefile set a
    /// shell just read for it — [`Self::hydrate_geojson`]'s shapefile twin.
    ///
    /// # Errors
    ///
    /// Propagates [`LocalVectorError`] when the bytes are not a usable
    /// shapefile (see [`crate::shapefile_input::from_bytes`]), or when `id`
    /// names no layer of `project`.
    pub fn hydrate_shapefile(
        &mut self,
        project: &Project,
        id: LayerId,
        set: ShapefileBytes<'_>,
    ) -> Result<(), LocalVectorError> {
        let features = set.to_feature_collection()?;
        self.hydrate_features(project, id, features)
    }
    /// Rebuilds an existing project layer from the bytes of the GeoPackage one
    /// of its feature tables came from — [`Self::hydrate_shapefile`]'s
    /// GeoPackage twin.
    ///
    /// `table` is the name [`VectorSource::LocalGpkg`] recorded. The whole file
    /// is parsed and the named table picked out of it, which on a project load
    /// means re-reading the file once per layer it contributed; at the sizes a
    /// desktop drop deals in that is cheaper than the cache the alternative
    /// needs, and it keeps every other table's refusal a notice rather than an
    /// error.
    ///
    /// # Errors
    ///
    /// Propagates [`LocalVectorError`] when the bytes are not a readable
    /// GeoPackage, when the named table is no longer in it (or could no longer
    /// be imported, in which case the reason is reported), or when `id` names
    /// no layer of `project`.
    pub fn hydrate_gpkg(
        &mut self,
        project: &Project,
        id: LayerId,
        gpkg: &[u8],
        table: &str,
    ) -> Result<(), LocalVectorError> {
        let (tables, refusals) = crate::gpkg_input::from_bytes(gpkg)?.into_parts();
        let found = tables
            .into_iter()
            .find(|candidate| candidate.name == table)
            .ok_or_else(|| {
                // Matched on the refusal's own table name, never on the text of
                // its message: a message embeds the offending table's name, so
                // a substring test would let the refusal of `roads_old` answer
                // for a layer whose table is `roads` and report the wrong
                // reason. The wording the user sees is unchanged either way.
                LocalVectorError::new(
                    refusals
                        .iter()
                        .find(|refusal| refusal.table() == table)
                        .map_or_else(
                            || {
                                format!(
                                    "the feature table \u{201c}{table}\u{201d} is no longer in \
                                     this GeoPackage"
                                )
                            },
                            |refusal| refusal.message().to_string(),
                        ),
                )
            })?;
        self.hydrate_features(project, id, found.features)
    }
    /// Rebuilds an existing project layer from the bytes of the GeoParquet
    /// file it was read from — [`Self::hydrate_gpkg`]'s GeoParquet twin.
    ///
    /// Compiled only under the `geoparquet` Cargo feature (native-only, see
    /// [`crate::geoparquet_input`]'s module docs). Callers on a build without
    /// it never reach this — see [`Self::rebuild_from_project`]'s
    /// `LocalGeoParquet` arm, which reports a notice instead.
    ///
    /// # Errors
    ///
    /// Propagates [`LocalVectorError`] when the bytes are not a usable
    /// GeoParquet file (see [`crate::geoparquet_input::from_bytes`]), or when
    /// `id` names no layer of `project`.
    #[cfg(feature = "geoparquet")]
    pub fn hydrate_geoparquet(
        &mut self,
        project: &Project,
        id: LayerId,
        bytes: &[u8],
    ) -> Result<(), LocalVectorError> {
        let features = crate::geoparquet_input::from_bytes(bytes)?;
        self.hydrate_features(project, id, features)
    }
    /// The shared tail of the `hydrate_*` entry points: attach an
    /// already-parsed collection to a layer the project already holds.
    ///
    /// The one thing this owns that [`Self::replace_features`] must not is the
    /// [`Self::default_styles`] entry: a *re-read from disk* legitimately
    /// re-derives the style the layer would have been born with, whereas an
    /// edit of the features must leave that memory alone (otherwise removing
    /// an explicit style after an edit restores something different from
    /// before it). So the derivation happens here and the rest is delegated,
    /// leaving exactly one implementation of "adopt this collection".
    fn hydrate_features(
        &mut self,
        project: &Project,
        id: LayerId,
        features: FeatureCollection,
    ) -> Result<(), LocalVectorError> {
        let features = Arc::new(features);
        // The project's own style still wins where there is one; without it,
        // the geometry decides. Recorded *before* delegating, so
        // `replace_features` reads exactly this back and the queued layer and
        // the remembered default cannot disagree.
        let style = project
            .styles
            .get(&id)
            .cloned()
            .unwrap_or_else(|| crate::local_vector::default_style_set_for(&features));
        let previous = self.default_styles.insert(id, style);
        let result = self.replace_features(project, id, features);
        if result.is_err() {
            // The layer is gone; a failed hydrate must leave nothing behind.
            match previous {
                Some(style) => self.default_styles.insert(id, style),
                None => self.default_styles.remove(&id),
            };
        }
        result
    }
    /// Replaces a known local layer's features with an already-shared
    /// collection, keeping its id, style, visibility and opacity, and queueing
    /// the GPU replacement.
    ///
    /// The edit twin of the `hydrate_*` family, differing in two ways that
    /// matter:
    ///
    /// * it **adopts** an [`Arc`] rather than parsing one, so
    ///   [`Self::feature_set`] and the queued [`LocalLayerOp::Add`] provably
    ///   carry the same collection — the invariant that keeps the app's store,
    ///   the GPU copy and the attribute table's bound handle from diverging;
    /// * it **preserves** the remembered [`Self::default_style`] entry instead
    ///   of re-deriving it. Changing a layer's data must not reset the style it
    ///   was born with, or a user who later removes their explicit style gets a
    ///   different look than they would have before they edited.
    ///
    /// The style used is [`Project::styles`]' entry when there is one, else the
    /// remembered default, else one derived from the new geometry — so a layer
    /// that never had either still gets something sensible.
    ///
    /// Nothing is appended to `project` and nothing in it is written: the layer
    /// is already there, and its `kind` (the serialized source) belongs to the
    /// caller.
    ///
    /// # Errors
    ///
    /// Returns [`LocalVectorError`] when `id` names no layer of `project`.
    pub fn replace_features(
        &mut self,
        project: &Project,
        id: LayerId,
        features: Arc<FeatureCollection>,
    ) -> Result<(), LocalVectorError> {
        let layer = project.layers.get(id).ok_or_else(|| {
            LocalVectorError::new(format!("layer {id} is no longer in the project"))
        })?;
        let style = project
            .styles
            .get(&id)
            .cloned()
            .or_else(|| self.default_styles.get(&id).cloned())
            .unwrap_or_else(|| crate::local_vector::default_style_set_for(&features));
        let mut local = LocalVectorLayer::with_style_arc(&layer.name, Arc::clone(&features), style);
        local.set_visible(layer.visible);
        local.set_opacity(layer.opacity());
        self.families.insert(id, local.families());
        self.feature_sets.insert(id, features);
        self.queue(LocalLayerOp::Add(id, Box::new(local)));
        Ok(())
    }
    /// Appends a local vector layer with no features yet — the on-ramp for
    /// digitizing into a fresh dataset.
    ///
    /// Deliberately bypasses [`parse_geojson`]'s non-empty rule, which exists
    /// to reject a *failed drop* (an empty collection on screen is
    /// indistinguishable from one), not a deliberately empty layer the user
    /// just asked for. The source is [`VectorSource::InlineGeoJson`] from
    /// birth, so the layer needs no conversion on its first edit and survives
    /// save/reload on both shells; `kind` picks the style, since an empty
    /// collection has no geometry to derive one from.
    ///
    /// # Errors
    ///
    /// Returns [`LocalVectorError`] if the empty collection cannot be
    /// serialized — unreachable in practice, but reported rather than
    /// swallowed, since the alternative is a layer whose stored source does not
    /// match its data.
    pub fn add_empty_vector_layer(
        &mut self,
        project: &mut Project,
        name: &str,
        kind: crate::style_panel::StyleKind,
    ) -> Result<AddedLocalLayer, LocalVectorError> {
        let features = FeatureCollection::empty();
        let geojson = oxigeo::geojson::writer::to_string(&features).map_err(|error| {
            LocalVectorError::new(format!("the empty layer could not be serialized: {error}"))
        })?;
        let inline_bytes = geojson.len();
        let style = LayerStyleSet::new(kind.default_style());
        let local = LocalVectorLayer::with_style(name, features, style.clone());
        // An empty collection has no projectable vertex, so this is
        // `MercatorSquare::world()` — "zoom to layer" on a fresh layer shows
        // the whole world rather than refusing to move.
        let square = local.square();
        let id = project.layers.add(Layer::new(
            name,
            LayerKind::Vector(VectorSource::InlineGeoJson { geojson }),
        ));
        project.styles.insert(id, style.clone());
        self.default_styles.insert(id, style);
        self.families.insert(id, local.families());
        self.feature_sets.insert(id, local.features_arc());
        self.queue(LocalLayerOp::Add(id, Box::new(local)));
        Ok(AddedLocalLayer {
            id,
            square,
            feature_count: 0,
            inline_bytes: Some(inline_bytes),
        })
    }
    /// The parsed features of a local layer, if one is still known under `id`.
    ///
    /// The attribute table's entry point: an [`Arc`] handle, so the caller can
    /// hold the collection across a frame without copying it and without
    /// touching the render side's lock.
    #[must_use]
    pub fn feature_set(&self, id: LayerId) -> Option<&Arc<FeatureCollection>> {
        self.feature_sets.get(&id)
    }
    /// How many local layers currently have their features in the store.
    #[must_use]
    pub fn feature_set_count(&self) -> usize {
        self.feature_sets.len()
    }
    /// The geometry families a stored local collection draws — empty for a
    /// layer this store does not know (a provider layer never shows the
    /// family row).
    #[must_use]
    pub fn families(&self, id: LayerId) -> oxigis_core::FamilySet {
        self.families.get(&id).copied().unwrap_or_default()
    }

    /// The style a local layer was created with, if it is still known.
    #[must_use]
    pub fn default_style(&self, id: LayerId) -> Option<&LayerStyleSet> {
        self.default_styles.get(&id)
    }
    /// Re-seeds a layer's remembered default style — the undo path's
    /// counterpart of [`Self::forget`], so a restored layer keeps its
    /// Style ▸ Remove fallback.
    pub fn remember_default_style(&mut self, id: LayerId, style: impl Into<LayerStyleSet>) {
        self.default_styles.insert(id, style.into());
    }
    /// Forgets a layer's remembered default style and stored features (it was
    /// removed).
    pub fn forget(&mut self, id: LayerId) {
        self.default_styles.remove(&id);
        self.families.remove(&id);
        self.feature_sets.remove(&id);
    }
    /// Parses `text` as GeoJSON, appends it to `project` as a local vector
    /// layer, and queues the parsed dataset for the shell.
    ///
    /// `path`, when given, both names the layer's source
    /// ([`VectorSource::LocalGeoJson`], so the project file stores a reference
    /// instead of a copy) and is *not* re-read: the caller already has the text.
    /// Without one the text is embedded ([`VectorSource::InlineGeoJson`]) —
    /// there is no third option, since a browser drop and a paste have no path.
    ///
    /// The layer's style is the one [`LocalVectorLayer::from_geojson`] derived
    /// from the geometry, copied into [`Project::styles`], which stays the
    /// source of truth from then on.
    ///
    /// # Errors
    ///
    /// Propagates [`LocalVectorError`] when the text is not a usable GeoJSON
    /// `FeatureCollection`. Nothing is added to `project` in that case.
    pub fn add_geojson(
        &mut self,
        project: &mut Project,
        name: &str,
        text: &str,
        path: Option<&str>,
    ) -> Result<AddedLocalLayer, LocalVectorError> {
        let features = parse_geojson(text)?;
        let (source, inline_bytes) = match path {
            Some(path) => (
                VectorSource::LocalGeoJson {
                    path: path.to_string(),
                },
                None,
            ),
            None => (
                VectorSource::InlineGeoJson {
                    geojson: text.to_string(),
                },
                Some(text.len()),
            ),
        };
        Ok(
            self.add_feature_collection(
                project,
                name,
                features,
                source,
                inline_bytes,
                Crs::wgs84(),
            ),
        )
    }
    /// [`Self::add_geojson`] for a document produced **in this process** —
    /// a Processing tool's result — which arrives as a [`serde_json::Value`]
    /// rather than as text.
    ///
    /// Same layer, same style derivation, same queued GPU work; what differs is
    /// how many copies of the dataset are alive while the layer is built.
    /// Routing a `Value` through `add_geojson` means `to_string` then a full
    /// re-parse of that text, and `add_geojson` then clones the text again for
    /// the inline source — four representations of one document at the peak, on
    /// the path most likely to be handed a large one. Here the value is
    /// *consumed* into the collection first, and the inline text is serialised
    /// from the collection afterwards, so at most two exist at once.
    ///
    /// There is no `path` argument on purpose: a tool result has no file to
    /// reference, so this seam only ever writes [`VectorSource::InlineGeoJson`].
    ///
    /// # Errors
    ///
    /// Propagates [`LocalVectorError`] when the value is not a usable GeoJSON
    /// `FeatureCollection`, or when the parsed collection cannot be serialised
    /// back for the project document. Nothing is added to `project` in either
    /// case.
    pub fn add_geojson_value(
        &mut self,
        project: &mut Project,
        name: &str,
        value: serde_json::Value,
    ) -> Result<AddedLocalLayer, LocalVectorError> {
        // Parse FIRST: it consumes the value, and a refusal then costs no
        // serialisation at all.
        let features = parse_geojson_value(value)?;
        // Re-serialised from the parsed collection rather than from the value,
        // which no longer exists. Lossless — `FeatureCollection` carries its
        // `bbox`, `crs` and flattened foreign members through both directions —
        // and it is exactly the text a project reload will parse back.
        let geojson = serde_json::to_string(&features).map_err(|error| {
            LocalVectorError::new(format!("the result could not be encoded: {error}"))
        })?;
        let inline_bytes = geojson.len();
        Ok(self.add_feature_collection(
            project,
            name,
            features,
            VectorSource::InlineGeoJson { geojson },
            Some(inline_bytes),
            Crs::wgs84(),
        ))
    }
    /// Reads a shapefile set from its bytes, appends it to `project` as a local
    /// vector layer, and queues the parsed dataset for the shell.
    ///
    /// The shapefile twin of [`Self::add_geojson`]; see
    /// [`crate::shapefile_input::from_bytes`] for what the four byte/text
    /// arguments are and what is kept from them.
    ///
    /// **Persistence** differs from the GeoJSON path, and cannot not:
    ///
    /// * with a `path` (a native drop) the layer is a
    ///   [`VectorSource::LocalShapefile`] reference, re-read by the desktop
    ///   shell on the next project load exactly like a `LocalGeoJson` path is;
    /// * without one (a browser drop) the set is **converted to GeoJSON text**
    ///   and embedded as [`VectorSource::InlineGeoJson`]. Shapefile bytes are
    ///   binary and multi-file, so there is nothing sane to put in a JSON
    ///   document; the GeoJSON rendering carries the same features and
    ///   attributes, and re-loads through the ordinary inline path.
    ///
    /// # Errors
    ///
    /// Propagates [`LocalVectorError`] when the bytes are not a usable
    /// shapefile, or when the browser leg cannot serialise the result. Nothing
    /// is added to `project` in that case.
    pub fn add_shapefile(
        &mut self,
        project: &mut Project,
        name: &str,
        set: ShapefileBytes<'_>,
        path: Option<&str>,
    ) -> Result<AddedLocalLayer, LocalVectorError> {
        let dataset = set.to_dataset()?;
        let features = dataset.features;
        let (source, inline_bytes) = match path {
            Some(path) => (
                VectorSource::LocalShapefile {
                    path: path.to_string(),
                },
                None,
            ),
            None => {
                let geojson = crate::shapefile_input::to_geojson_string(&features)?;
                let length = geojson.len();
                (VectorSource::InlineGeoJson { geojson }, Some(length))
            }
        };
        Ok(self.add_feature_collection(project, name, features, source, inline_bytes, dataset.crs))
    }
    /// Reads every feature table of a GeoPackage, appends **one layer per
    /// table** to `project`, and queues them all.
    ///
    /// The GeoPackage twin of [`Self::add_shapefile`], and the only entry point
    /// here that produces more than one layer from one drop. Layers are named
    /// after their table when the file yielded exactly one, and `stem:table`
    /// when it yielded several — the prefix exists to keep the layer panel
    /// readable when two files both hold a `roads` table, so it is decided by
    /// how many layers are actually being added, not by how many tables the
    /// file lists.
    ///
    /// **Persistence** follows the shapefile rule, per table:
    ///
    /// * with a `path` (a native drop) each layer is a
    ///   [`VectorSource::LocalGpkg`] reference carrying the table's name, so a
    ///   project load can rebuild exactly the table it came from;
    /// * without one (a browser drop) each table is **converted to GeoJSON
    ///   text** and embedded as [`VectorSource::InlineGeoJson`], since a binary
    ///   database cannot go into a JSON document.
    ///
    /// # Errors
    ///
    /// Propagates [`LocalVectorError`] when the bytes are not a readable
    /// GeoPackage, when the browser leg cannot serialise a table, and — the
    /// case that is *not* an error inside [`crate::gpkg_input::from_bytes`] —
    /// when no table would become a layer at all, whose message is the per-table
    /// refusals so the status line names the CRS that caused them. Nothing is
    /// added to `project` in that case.
    pub fn add_gpkg(
        &mut self,
        project: &mut Project,
        name: &str,
        gpkg: &[u8],
        path: Option<&str>,
    ) -> Result<AddedGpkg, LocalVectorError> {
        let (tables, refusals) = crate::gpkg_input::from_bytes(gpkg)?.into_parts();
        let notices: Vec<String> = refusals
            .iter()
            .map(|refusal| refusal.message().to_string())
            .collect();
        if tables.is_empty() {
            return Err(LocalVectorError::new(if notices.is_empty() {
                "the GeoPackage holds no feature tables".to_string()
            } else {
                notices.join(" ")
            }));
        }
        let stem = name.rsplit_once('.').map_or(name, |(head, _)| head);
        let single = tables.len() == 1;
        // Two passes, and the split is the whole point: every fallible step for
        // every table runs *before* the first one touches `project`. Converting
        // table N and mutating table N-1 in the same iteration would leave the
        // earlier tables appended and queued to the GPU when a later `?` fires,
        // while the caller — reading only the `Err` — reports "nothing was
        // added" and never selects them. This method's contract says nothing is
        // added on error, and after this it is true by construction rather than
        // by no input happening to fail today.
        let mut prepared = Vec::with_capacity(tables.len());
        for table in tables {
            let layer_name = if single {
                table.name.clone()
            } else {
                format!("{stem}:{}", table.name)
            };
            let (source, inline_bytes) = match path {
                Some(path) => (
                    VectorSource::LocalGpkg {
                        path: path.to_string(),
                        table: table.name.clone(),
                    },
                    None,
                ),
                None => {
                    let geojson = crate::gpkg_input::to_geojson_string(&table.features)?;
                    let length = geojson.len();
                    (VectorSource::InlineGeoJson { geojson }, Some(length))
                }
            };
            prepared.push((layer_name, table.features, source, inline_bytes, table.crs));
        }
        let mut layers = Vec::with_capacity(prepared.len());
        for (layer_name, features, source, inline_bytes, table_crs) in prepared {
            layers.push(self.add_feature_collection(
                project,
                &layer_name,
                features,
                source,
                inline_bytes,
                table_crs,
            ));
        }
        Ok(AddedGpkg { layers, notices })
    }
    /// Reads a GeoParquet file from its bytes, appends it to `project` as a
    /// local vector layer, and queues the parsed dataset for the shell.
    ///
    /// The GeoParquet twin of [`Self::add_shapefile`] — one file, one layer,
    /// unlike [`Self::add_gpkg`]'s "one file, several layers". Compiled only
    /// under the `geoparquet` Cargo feature; see
    /// [`crate::geoparquet_input`]'s module docs for why (native-only,
    /// arrow/parquet are heavy) and [`Self::rebuild_from_project`]'s
    /// `LocalGeoParquet` arm for what a build without the feature does with a
    /// project that references one.
    ///
    /// **Persistence** follows the shapefile/GeoPackage rule:
    ///
    /// * with a `path` (a native drop) the layer is a
    ///   [`VectorSource::LocalGeoParquet`] reference, re-read by the desktop
    ///   shell on the next project load;
    /// * without one the collection is **converted to GeoJSON text** and
    ///   embedded as [`VectorSource::InlineGeoJson`] — a binary Parquet file
    ///   cannot go into a JSON document. In practice this leg is only reached
    ///   by a caller that hands over bytes with no path (e.g. a direct test
    ///   of this method): the native desktop shell, the only build that ever
    ///   compiles this feature, always has a path for a dropped file.
    ///
    /// # Errors
    ///
    /// Propagates [`LocalVectorError`] when the bytes are not a usable
    /// GeoParquet file (see [`crate::geoparquet_input::from_bytes`]), or when
    /// the browser leg cannot serialise the result. Nothing is added to
    /// `project` in that case.
    #[cfg(feature = "geoparquet")]
    pub fn add_geoparquet(
        &mut self,
        project: &mut Project,
        name: &str,
        bytes: &[u8],
        path: Option<&str>,
    ) -> Result<AddedLocalLayer, LocalVectorError> {
        let dataset = crate::geoparquet_input::read_dataset(bytes)?;
        let features = dataset.features;
        let (source, inline_bytes) = match path {
            Some(path) => (
                VectorSource::LocalGeoParquet {
                    path: path.to_string(),
                },
                None,
            ),
            None => {
                let geojson = crate::geoparquet_input::to_geojson_string(&features)?;
                let length = geojson.len();
                (VectorSource::InlineGeoJson { geojson }, Some(length))
            }
        };
        Ok(self.add_feature_collection(project, name, features, source, inline_bytes, dataset.crs))
    }
    /// The shared tail of every "add a local layer" path: register the parsed
    /// collection, give it the default style for its geometry, and queue it.
    ///
    /// `source_crs` is what the file's *own* coordinates were in — the reader
    /// has already converted `features` to WGS 84 lon/lat, so this is
    /// provenance the layer records (see [`oxigis_core::Layer::crs`]) rather
    /// than anything the renderer acts on. WGS 84 is recorded as "no CRS", so a
    /// project of ordinary GeoJSON layers serializes exactly as it always did.
    fn add_feature_collection(
        &mut self,
        project: &mut Project,
        name: &str,
        features: FeatureCollection,
        source: VectorSource,
        inline_bytes: Option<usize>,
        source_crs: Crs,
    ) -> AddedLocalLayer {
        let local = LocalVectorLayer::from_feature_collection(name, features);
        let style = local.style().clone();
        let square = local.square();
        let feature_count = local.feature_count();
        let id = project
            .layers
            .add(Layer::new(name, LayerKind::Vector(source)).with_crs(source_crs));
        project.styles.insert(id, style.clone());
        self.default_styles.insert(id, style);
        self.families.insert(id, local.families());
        self.feature_sets.insert(id, local.features_arc());
        self.queue(LocalLayerOp::Add(id, Box::new(local)));
        AddedLocalLayer {
            id,
            square,
            feature_count,
            inline_bytes,
        }
    }
    /// Rebuilds every local layer of a freshly loaded `project`.
    ///
    /// Queues a [`LocalLayerOp::Clear`] first — the previous project's datasets
    /// are still attached to the GPU — then one [`LocalLayerOp::Add`] per inline
    /// layer, restyled with the project's own stored [`LayerStyleSet`] where it
    /// has
    /// one. Layers that only reference a path are queued for the native shell
    /// to read; on a build without path support they are reported instead, since
    /// a browser cannot open a file by path.
    ///
    /// Returns the notices to show the user (parse failures, unreadable path
    /// references); an empty vector means everything was rebuilt cleanly.
    pub fn rebuild_from_project(&mut self, project: &Project) -> Vec<String> {
        self.default_styles.clear();
        self.families.clear();
        self.feature_sets.clear();
        self.queue(LocalLayerOp::Clear);
        let mut notices = Vec::new();
        for layer in project.layers.layers() {
            let LayerKind::Vector(source) = &layer.kind else {
                continue;
            };
            match source {
                VectorSource::InlineGeoJson { geojson } => {
                    // Deliberately NOT `LocalVectorLayer::from_geojson`: that
                    // rejects an empty collection, which is the right answer
                    // for a *dropped file* (an empty one is indistinguishable
                    // from a failed drop) and exactly the wrong one here — a
                    // freshly created edit layer, or one whose last feature was
                    // deleted, saves fine and would then reload forever as an
                    // un-hydratable stub. `from_geojson`'s own contract is
                    // untouched.
                    match oxigeo::geojson::reader::feature_collection_from_str(geojson) {
                        Ok(features) => {
                            let mut local =
                                LocalVectorLayer::from_feature_collection(&layer.name, features);
                            if let Some(style) = project.styles.get(&layer.id) {
                                local.set_style(style.clone());
                            }
                            self.default_styles.insert(layer.id, local.style().clone());
                            self.families.insert(layer.id, local.families());
                            self.feature_sets.insert(layer.id, local.features_arc());
                            local.set_visible(layer.visible);
                            local.set_opacity(layer.opacity());
                            self.queue(LocalLayerOp::Add(layer.id, Box::new(local)));
                        }
                        Err(error) => notices.push(format!(
                            "Layer \"{}\" could not be rebuilt: GeoJSON parse failed: {error}",
                            layer.name,
                        )),
                    }
                }
                VectorSource::LocalGeoJson { path } | VectorSource::LocalShapefile { path } => {
                    if self.paths_supported {
                        self.queue_path(Some(layer.id), PathBuf::from(path));
                    } else {
                        notices.push(format!(
                            "Layer \"{}\" references the file {path}, which is not available in \
                             the browser; re-drop the file to restore it.",
                            layer.name,
                        ));
                    }
                }
                VectorSource::LocalGpkg { path, table } => {
                    if self.paths_supported {
                        self.queue_gpkg_path(
                            Some(layer.id),
                            PathBuf::from(path),
                            table.to_string(),
                        );
                    } else {
                        notices.push(format!(
                            "Layer \"{}\" references the table {table} of {path}, which is not \
                             available in the browser; re-drop the file to restore it.",
                            layer.name,
                        ));
                    }
                }
                VectorSource::LocalGeoParquet { path } => {
                    // Unlike the two arms above, whether this can be handled
                    // at all depends on the `geoparquet` Cargo feature, not
                    // only on `paths_supported` — see the module docs on
                    // `crate::geoparquet_input` for why the feature exists.
                    // Both `#[cfg]` arms are unit tests directly (see
                    // `local_input::tests`), one per feature state.
                    #[cfg(feature = "geoparquet")]
                    {
                        if self.paths_supported {
                            self.queue_path(Some(layer.id), PathBuf::from(path));
                        } else {
                            notices.push(format!(
                                "Layer \"{}\" references the file {path}, which is not \
                                 available in the browser; re-drop the file to restore it.",
                                layer.name,
                            ));
                        }
                    }
                    #[cfg(not(feature = "geoparquet"))]
                    {
                        notices.push(format!(
                            "Layer \"{}\" references the GeoParquet file {path}, which this \
                             build does not support (native desktop only); the layer was \
                             skipped.",
                            layer.name,
                        ));
                    }
                }
                // Provider-drawn sources have no local mirror to rebuild: the
                // reconciliation in `app/providers.rs` derives their provider
                // straight from the project on the next frame. An archive whose
                // reference cannot be read at all is the one thing worth saying
                // out loud, since nothing else ever will.
                VectorSource::MvtTiles { .. } => {}
                VectorSource::TileArchive {
                    archive, format, ..
                } => {
                    if let Some(reason) = oxigis_core::archive_refusal(archive, *format) {
                        notices.push(format!(
                            "Layer \"{}\" could not be restored: {reason}",
                            layer.name,
                        ));
                    }
                }
            }
        }
        notices
    }
}
impl Default for LocalInputState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
