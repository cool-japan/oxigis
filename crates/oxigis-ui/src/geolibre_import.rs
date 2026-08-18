// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Best-effort **import** of a GeoLibre (`opengeos/GeoLibre`, MIT)
//! `.geolibre.json` project document — TODO §1.4's "GeoLibre .geolibre.json
//! import compat" line.
//!
//! # Why `serde_json::Value`, not `#[derive(Deserialize)]`
//!
//! GeoLibre's own `GeoLibreProject`/`GeoLibreLayer`/`LayerStyle` types
//! (`packages/core/src/types.ts`) are roughly 15x the surface of
//! [`oxigis_core`]'s model — per-feature classification modes, 3D
//! extrusion, diagrams, heatmaps, clustering, a plugin system, storymaps,
//! dashboards — none of which this reader can *do* anything with even if it
//! parsed them. Modeling that whole schema just to ignore 90% of the fields
//! would be actively misleading (a `#[derive(Deserialize)]` struct reads as
//! "we understand this shape"). Instead this reads only the handful of
//! fields the mapping in [`import`] actually consumes, straight out of the
//! parsed [`serde_json::Value`] tree — a `LayerType` we do not recognise, or
//! a style field of the wrong JSON type, is data we've simply never heard
//! of, not a parse error.
//!
//! # What imports vs. what is dropped
//!
//! Reachable end to end: `name`; `mapView.{center,zoom}`; a `geojson`
//! layer's inline `geojson` payload (any original format, since GeoLibre
//! already normalized it client-side before saving); a `geojson` layer's
//! `sourcePath` ending `.geojson`/`.json`/`.shp`/`.parquet`/`.geoparquet`; an
//! `xyz` raster with a non-TMS tile template; a `cog` raster with a public
//! `http(s)` URL; a `vector-tiles` layer with a raw tile template and named
//! source layers.
//!
//! Recognised, but **not turned into a `Layer` by [`import`] itself**: a
//! `pmtiles`/`mbtiles` layer with a public `http(s)` `source.url`. Whether it
//! becomes a [`oxigis_core::VectorSource::TileArchive`] or a
//! [`oxigis_core::RasterSource::TileArchive`] is only knowable by reading the
//! archive's own header, which this module never does (see "Why
//! `serde_json::Value`" above) — [`deferred_archives`] collects these
//! separately, by name/format/URL, for a caller able to run the same
//! [`crate::archive::ArchiveProbe`] a direct `.pmtiles`/`.mbtiles` drop
//! already does; see `import_tile_archive_layer`'s doc comment.
//!
//! Dropped, always, with a notice where the audit specifies exact wording:
//! `basemapStyleUrl`/`Visible`/`Opacity` (wrong shape — a MapLibre
//! `style.json` URL, not our raster XYZ `BasemapConfig`, and
//! [`oxigis_core::Project`] has nowhere to hold one regardless),
//! `layerGroups`, `selectedLayerId`, `preferences`, `plugins`, `legend`,
//! `storymap`, `models`, `processingHistory`, `widgets`, `dashboardColumns`,
//! `mapLayout`/`secondaryMapViews`, `primaryMapLabel`, `styleLibrary`,
//! `metadata`; `mapView.bearing`/`pitch` (the renderer is north-up only); 12
//! of the 20 `LayerType`s (raster/wms/wmts/arcgis/zarr/lidar/gaussian-splat/
//! 3d-tiles/flatgeobuf/geoparquet/duckdb-query/deckgl-viz/video/image — note
//! `geoparquet` here is GeoLibre's *raw-file* layer `type`, a different thing
//! from a `geojson` layer's `sourcePath` ending `.parquet`, which we *do*
//! import); a `sourcePath` ending `.gpkg`
//! (GeoLibre does not record which table inside it the layer reads, and
//! guessing one would silently show the wrong data — an improvement over the
//! audit this module implements, since GPKG/GeoParquet import landed in
//! `oxigis-core` after the audit was written); a `vector-tiles`/`cog` layer
//! with only a TileJSON URL / blob URL / no URL at all; per-layer text
//! labels (`style.labels.enabled`), since [`oxigis_core::LayerStyle`] is a
//! tagged enum that holds a geometry style *or* a label style, never both —
//! the geometry style always wins here.
//!
//! No round-trip: this is read-only, one direction. Saving still only ever
//! writes `.oxigis.json`.

use oxigeo::geojson::types::FeatureCollection;
use oxigis_core::{
    ArchiveFormat, CircleStyle, Color, FillStyle, Layer, LayerKind, LayerStyle, LineStyle, Project,
    RasterSource, VectorSource, VectorTilePaint,
};
use serde_json::{Map, Value};

use crate::local_input::parse_geojson;
use crate::local_vector::{
    DEFAULT_CIRCLE_RADIUS_PX, DEFAULT_LINE_WIDTH_PX, GeometryKind, LocalVectorError,
    default_style_for_kind, dominant_geometry_kind,
};

/// Fallback project name when a GeoLibre document has no (or a non-string)
/// top-level `name`.
const DEFAULT_PROJECT_NAME: &str = "Imported GeoLibre project";
/// Fallback layer name when a GeoLibre layer entry has no (or a non-string)
/// `name`.
const DEFAULT_LAYER_NAME: &str = "Unnamed layer";

/// Whether `json` looks like a GeoLibre `.geolibre.json` project: a JSON
/// object with a string `"version"` and an object `"mapView"`.
///
/// Deliberately loose (GeoLibre's real `PROJECT_VERSION` today is `"0.2.0"`,
/// but this does not check the value — only that a plausible GeoLibre shape
/// is present) and deliberately narrow: an [`oxigis_core::Project`] has
/// neither field under these names (`format_version` is a bare integer, and
/// `view` — not `mapView` — is its sibling top-level field), so this never
/// fires on our own format. The audit's empirical probe found no document
/// that satisfies both this and [`oxigis_core::Project::from_json_string`];
/// every call site here tries the real format first and only reaches this
/// sniff on its failure (see `crate::app`'s `load_project_text`).
#[must_use]
pub fn looks_like_geolibre(json: &Value) -> bool {
    let Some(obj) = json.as_object() else {
        return false;
    };
    matches!(obj.get("version"), Some(Value::String(_)))
        && matches!(obj.get("mapView"), Some(Value::Object(_)))
}

/// Imports as much of a GeoLibre `.geolibre.json` document as
/// [`oxigis_core`]'s model can represent, never failing outright over a
/// single bad layer or field — see the module docs for the reachable
/// subset. Everything that could not be carried over is instead reported as
/// one entry of the returned notice list, in the wording the audited spec
/// (TODO §1.4) calls for, so the caller can show it on the status line the
/// same way [`crate::local_input::LocalInputState::rebuild_from_project`]'s
/// notices already are.
///
/// # Errors
///
/// Only when `json` is not even a JSON object — every other problem (a
/// malformed color, an unsupported layer type, an unreadable path
/// extension) degrades to a notice instead, per the audit's "best-effort,
/// never a hard failure" recommendation.
pub fn import(json: &Value) -> Result<(Project, Vec<String>), LocalVectorError> {
    let obj = json
        .as_object()
        .ok_or_else(|| LocalVectorError::new("not a GeoLibre project: not a JSON object"))?;

    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_PROJECT_NAME);
    let mut project = Project::new(name);
    let mut notices = Vec::new();

    apply_map_view(obj, &mut project, &mut notices);

    let styles_map = obj.get("styles").and_then(Value::as_object);
    let mut unsupported = 0usize;
    if let Some(layers) = obj.get("layers").and_then(Value::as_array) {
        for layer_value in layers {
            if !import_layer(layer_value, styles_map, &mut project, &mut notices) {
                unsupported += 1;
            }
        }
    }

    let has_basemap = obj
        .get("basemapStyleUrl")
        .and_then(Value::as_str)
        .is_some_and(|url| !url.is_empty());
    let has_groups = obj
        .get("layerGroups")
        .and_then(Value::as_array)
        .is_some_and(|groups| !groups.is_empty());
    if has_basemap || has_groups || unsupported > 0 {
        notices.insert(
            0,
            format!(
                "Imported a GeoLibre project; basemap, groups, and {unsupported} unsupported \
                 layers were dropped."
            ),
        );
    }

    Ok((project, notices))
}

/// Maps `mapView.{center,zoom}` into [`Project::view`] (rule 1); a nonzero
/// `bearing`/`pitch` is noted rather than applied, since
/// [`oxigis_render::MapView`] has no rotation/tilt concept at all — the
/// camera stays north-up and level either way, so this only decides whether
/// the user is told.
fn apply_map_view(obj: &Map<String, Value>, project: &mut Project, notices: &mut Vec<String>) {
    let Some(map_view) = obj.get("mapView").and_then(Value::as_object) else {
        return;
    };
    if let Some(center) = map_view.get("center").and_then(Value::as_array)
        && let (Some(lon), Some(lat)) = (
            center.first().and_then(Value::as_f64),
            center.get(1).and_then(Value::as_f64),
        )
    {
        project.view.center_lon = lon;
        project.view.center_lat = lat;
    }
    if let Some(zoom) = map_view.get("zoom").and_then(Value::as_f64) {
        project.view.zoom = zoom;
    }
    let bearing = map_view
        .get("bearing")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let pitch = map_view.get("pitch").and_then(Value::as_f64).unwrap_or(0.0);
    if bearing != 0.0 || pitch != 0.0 {
        notices.push(
            "This project's saved view is rotated or tilted; OxiGIS's map camera is north-up \
             only, so the rotation/tilt was dropped."
                .to_string(),
        );
    }
}

/// One `layers[]` entry, mapped but not yet appended to the project — the
/// shared tail in [`import_layer`] applies the GeoLibre layer's own
/// `visible`/`opacity` flags and inserts the style, so none of the per-type
/// builders below have to repeat that bookkeeping.
struct MappedLayer {
    /// The mapped [`Layer`], with a fresh [`oxigis_core::LayerId`] (minted by
    /// [`Layer::new`] — the GeoLibre uuid `id` is never parsed as a number,
    /// per rule 3).
    layer: Layer,
    /// The layer's collapsed style, if this layer kind carries one in
    /// [`Project::styles`] at all (a raster or `MvtTiles` layer does not —
    /// its paint lives inside [`VectorSource::MvtTiles::paints`] or nowhere).
    style: Option<LayerStyle>,
}

/// Resolves a layer's style object per rule 2's precedence: the
/// project-level `styles` map (keyed by the GeoLibre layer's own uuid `id`)
/// wins over the layer's own embedded `style`, matching GeoLibre's own
/// `project.ts` load order — `layer.style` is a fallback for a layer whose
/// id doesn't appear in `styles` at all, never read unconditionally.
fn resolve_style<'a>(
    layer_obj: &'a Map<String, Value>,
    styles_map: Option<&'a Map<String, Value>>,
) -> Option<&'a Map<String, Value>> {
    let from_map = layer_obj
        .get("id")
        .and_then(Value::as_str)
        .zip(styles_map)
        .and_then(|(id, styles)| styles.get(id))
        .and_then(Value::as_object);
    from_map.or_else(|| layer_obj.get("style").and_then(Value::as_object))
}

/// Maps one `layers[]` entry (rules 2-3), returning whether it became a
/// layer at all — the caller counts the `false`s for the whole-project
/// summary notice.
fn import_layer(
    layer_value: &Value,
    styles_map: Option<&Map<String, Value>>,
    project: &mut Project,
    notices: &mut Vec<String>,
) -> bool {
    let Some(layer_obj) = layer_value.as_object() else {
        notices.push(
            "A layer entry in this GeoLibre project was not a JSON object; skipped.".to_string(),
        );
        return false;
    };
    let name = layer_obj
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_LAYER_NAME)
        .to_string();
    let layer_type = layer_obj.get("type").and_then(Value::as_str).unwrap_or("");
    let source = layer_obj.get("source").and_then(Value::as_object);
    let style = resolve_style(layer_obj, styles_map);

    let mapped = match layer_type {
        "geojson" => import_geojson_layer(&name, layer_obj, style, notices),
        "xyz" => import_xyz_layer(&name, source, notices),
        "cog" => import_cog_layer(&name, source, notices),
        "vector-tiles" => import_vector_tiles_layer(&name, source, style, notices),
        "pmtiles" | "mbtiles" => {
            import_tile_archive_layer(&name, layer_type, source, notices);
            None
        }
        other => {
            notices.push(format!(
                "Layer \"{name}\" is a GeoLibre {other} layer, which OxiGIS does not support; \
                 skipped."
            ));
            None
        }
    };

    let Some(mut mapped) = mapped else {
        return false;
    };
    apply_layer_flags(&mut mapped.layer, layer_obj);
    let id = mapped.layer.id;
    project.layers.add(mapped.layer);
    if let Some(style) = mapped.style {
        project.styles.insert(id, style.into());
    }
    true
}

/// Applies the GeoLibre layer's own `visible`/`opacity`/`minzoom`/`maxzoom`
/// fields, present on every `type`. [`oxigis_core::Layer::opacity`] is a
/// separate multiplier from a style's own opacity field (see
/// [`collapse_style`]'s docs on why a `Line`'s own opacity is always left at
/// its default), so this always runs, not only for the geometry-styled types.
///
/// `minzoom`/`maxzoom` are MapLibre's own per-layer scale range, `f64` in the
/// style JSON. They are passed straight through: [`Layer::set_zoom_range`]
/// sanitizes both ends (non-finite is dropped, anything past
/// [`oxigis_core::layer::MAX_ZOOM_LEVEL`] is clamped), and that ceiling is
/// MapLibre's own 24, which is why the field's domain was pinned to it. A layer
/// that declares neither keeps the open range it had, so a project written
/// before this existed imports byte-identically.
fn apply_layer_flags(layer: &mut Layer, layer_obj: &Map<String, Value>) {
    if let Some(opacity) = layer_obj.get("opacity").and_then(Value::as_f64) {
        layer.set_opacity(opacity as f32);
    }
    if let Some(visible) = layer_obj.get("visible").and_then(Value::as_bool) {
        layer.visible = visible;
    }
    let zoom_bound = |name: &str| {
        layer_obj
            .get(name)
            .and_then(Value::as_f64)
            .map(|bound| bound as f32)
    };
    let (min, max) = (zoom_bound("minzoom"), zoom_bound("maxzoom"));
    // Written only when at least one end is declared: `set_zoom_range` writes
    // BOTH ends, so calling it unconditionally would overwrite a range a layer
    // kind's own importer had already established with a pair of `None`s.
    if min.is_some() || max.is_some() {
        layer.set_zoom_range(min, max);
    }
}

/// `type: "geojson"` (rule 3, §4 table): inline `geojson` wins when present,
/// even if `sourcePath` is *also* set (GeoLibre's own loader treats
/// `geojson` as authoritative — `project.ts`'s `localFileReloadable` comment
/// cited by the audit). A layer added through GeoLibre's own "Add Vector
/// Layer" control instead has its `geojson` stripped **unconditionally** and
/// the same payload saved under `metadata.embeddedGeoJSON` (§3's
/// `layer-sync.ts`/`project.ts:1236-1244` citation) — checked next, before
/// falling back to `sourcePath`, since both are embedded data and only
/// `sourcePath` is a mere reference. Only when none of the three is present
/// does this give up.
fn import_geojson_layer(
    name: &str,
    layer_obj: &Map<String, Value>,
    style: Option<&Map<String, Value>>,
    notices: &mut Vec<String>,
) -> Option<MappedLayer> {
    let inline = non_null(layer_obj.get("geojson")).or_else(|| {
        non_null(
            layer_obj
                .get("metadata")
                .and_then(Value::as_object)
                .and_then(|metadata| metadata.get("embeddedGeoJSON")),
        )
    });
    match inline {
        Some(geojson_value) => import_inline_geojson_layer(name, geojson_value, style, notices),
        None => match layer_obj.get("sourcePath").and_then(Value::as_str) {
            Some(path) => import_path_referenced_layer(name, path, notices),
            None => {
                notices.push(format!(
                    "Layer \"{name}\" is a GeoLibre geojson layer with no embedded data and no \
                     source path; skipped."
                ));
                None
            }
        },
    }
}

/// `Some(value)` unless `value` is absent or JSON `null` — a present-but-null
/// field (e.g. `"geojson": null`) must fall through exactly like an absent
/// one, not be handed to [`serde_json::to_string`] as if it were data.
fn non_null(value: Option<&Value>) -> Option<&Value> {
    value.filter(|value| !value.is_null())
}

/// The inline-`geojson` case: re-serializes the parsed [`Value`] back to
/// text for [`VectorSource::InlineGeoJson`] (this reader never keeps
/// GeoLibre's parsed tree around — [`Project`] only ever stores GeoJSON as
/// text) and reuses [`parse_geojson`] to get a real [`FeatureCollection`] to
/// collapse the style against.
fn import_inline_geojson_layer(
    name: &str,
    geojson_value: &Value,
    style: Option<&Map<String, Value>>,
    notices: &mut Vec<String>,
) -> Option<MappedLayer> {
    let text = match serde_json::to_string(geojson_value) {
        Ok(text) => text,
        Err(_) => {
            notices.push(format!(
                "Layer \"{name}\" has an unreadable inline GeoJSON payload; skipped."
            ));
            return None;
        }
    };
    let features = match parse_geojson(&text) {
        Ok(features) => features,
        Err(error) => {
            notices.push(format!(
                "Layer \"{name}\" could not be imported: {}",
                error.message()
            ));
            return None;
        }
    };
    let style = collapse_style(&features, style, name, notices);
    Some(MappedLayer {
        layer: Layer::new(
            name,
            LayerKind::Vector(VectorSource::InlineGeoJson { geojson: text }),
        ),
        style: Some(style),
    })
}

/// The `sourcePath`-only case (no inline `geojson`): mapped by extension,
/// per the §4 table plus the GPKG/GeoParquet update noted in the module
/// docs.
///
/// Deliberately carries **no** style: unlike the inline case there is no
/// [`FeatureCollection`] to run [`dominant_geometry_kind`] against here
/// (nothing is read from disk during import — this crate does no I/O), so
/// this leaves [`Project::styles`] without an entry for the layer, exactly
/// like any other path-only local layer this crate creates without a style
/// entry is already handled: whichever shell later reads the file falls
/// back to the *real* geometry's default style at hydrate time
/// (`LocalInputState::hydrate_features`'s `None` arm) — strictly better than
/// guessing Fill/Line/Circle blind here.
fn import_path_referenced_layer(
    name: &str,
    path: &str,
    notices: &mut Vec<String>,
) -> Option<MappedLayer> {
    let lower = path.to_ascii_lowercase();
    let source = if lower.ends_with(".geojson") || lower.ends_with(".json") {
        Some(VectorSource::LocalGeoJson {
            path: path.to_string(),
        })
    } else if lower.ends_with(".shp") {
        Some(VectorSource::LocalShapefile {
            path: path.to_string(),
        })
    } else if lower.ends_with(".parquet") || lower.ends_with(".geoparquet") {
        Some(VectorSource::LocalGeoParquet {
            path: path.to_string(),
        })
    } else {
        None
    };

    let Some(source) = source else {
        if lower.ends_with(".gpkg") {
            // Orchestrator-noted improvement over the audit: GPKG import
            // landed in `oxigis-core` afterwards as `VectorSource::LocalGpkg
            // { path, table }`, but GeoLibre's `sourcePath` is just a file
            // path — it never records which feature table the layer reads,
            // and a GeoPackage can hold several. Guessing one (e.g. "the
            // first") would silently show the wrong table instead of
            // honestly refusing, so this is refused instead.
            notices.push(format!(
                "Layer \"{name}\" references the GeoPackage {path}, which GeoLibre does not \
                 record a table name for; skipped."
            ));
        } else {
            notices.push(format!(
                "Layer \"{name}\" references {path}, a format OxiGIS cannot read; skipped."
            ));
        }
        return None;
    };

    Some(MappedLayer {
        layer: Layer::new(name, LayerKind::Vector(source)),
        style: None,
    })
}

/// `type: "xyz"`: `source.tiles[0]` wins over `source.url` (both are written
/// on save; the audit found no case where only `url` is present for this
/// type), and `source.scheme === "tms"` is refused outright — this renderer
/// has no Y-flip, so drawing TMS tiles as XYZ would be silently upside-down,
/// not just imprecise.
///
/// `source.attribution` is carried across when the document declares one: a
/// credit line is a licence condition, and dropping it on import would leave
/// the layer legally undrawable the moment it is promoted to the basemap.
fn import_xyz_layer(
    name: &str,
    source: Option<&Map<String, Value>>,
    notices: &mut Vec<String>,
) -> Option<MappedLayer> {
    let Some(source) = source else {
        notices.push(format!(
            "Layer \"{name}\" is a GeoLibre xyz layer with no tile source; skipped."
        ));
        return None;
    };
    if source.get("scheme").and_then(Value::as_str) == Some("tms") {
        notices.push(format!(
            "Layer \"{name}\" uses TMS tile numbering, which OxiGIS does not support; skipped."
        ));
        return None;
    }
    let url_template = source
        .get("tiles")
        .and_then(Value::as_array)
        .and_then(|tiles| tiles.first())
        .and_then(Value::as_str)
        .or_else(|| source.get("url").and_then(Value::as_str));
    let Some(url_template) = url_template else {
        notices.push(format!(
            "Layer \"{name}\" is a GeoLibre xyz layer with no tile URL; skipped."
        ));
        return None;
    };
    let attribution = source
        .get("attribution")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Some(MappedLayer {
        layer: Layer::new(
            name,
            LayerKind::Raster(RasterSource::Xyz {
                url_template: url_template.to_string(),
                attribution,
            }),
        ),
        style: None,
    })
}

/// `type: "cog"`: only a real `http(s)` `source.url` is usable — a
/// drag-dropped local COG instead carries `metadata.localBytesUrl`, a
/// per-session `blob:` URL that cannot outlive the browser tab it was
/// created in, so it is refused exactly like a missing URL (audit §8's exact
/// wording covers both).
fn import_cog_layer(
    name: &str,
    source: Option<&Map<String, Value>>,
    notices: &mut Vec<String>,
) -> Option<MappedLayer> {
    let url = source
        .and_then(|source| source.get("url"))
        .and_then(Value::as_str);
    match url {
        Some(url) if url.starts_with("http://") || url.starts_with("https://") => {
            Some(MappedLayer {
                layer: Layer::new(
                    name,
                    LayerKind::Raster(RasterSource::Cog {
                        url: url.to_string(),
                    }),
                ),
                style: None,
            })
        }
        _ => {
            notices.push(format!(
                "Layer \"{name}\"'s COG file was not saved with a public URL; skipped."
            ));
            None
        }
    }
}

/// `type: "pmtiles" | "mbtiles"`: recognised, but never turned into a
/// [`Layer`] here. A [`VectorSource::TileArchive`]/
/// [`oxigis_core::RasterSource::TileArchive`] carries whether the archive
/// holds raster or vector tiles, and that is decided by reading the
/// archive's header — the same [`crate::archive::ArchiveProbe`] a
/// drag-and-drop `.pmtiles`/`.mbtiles` runs — which this reader cannot do:
/// no I/O happens anywhere in this module, by design (see the module docs'
/// "Why `serde_json::Value`" section). [`deferred_archives`] collects the
/// same layers this recognises, for a caller that *can* run one.
///
/// So this only validates the one thing decidable without a byte of I/O — a
/// public `http(s)` URL, [`import_cog_layer`]'s exact rule, since a
/// drag-dropped local archive's `metadata.localBytesUrl` is a `blob:` URL
/// that cannot outlive the browser tab it was created in either — and reports
/// an accurate notice: this is a *recognised, not-yet-wired* format, never
/// the generic "OxiGIS does not support" wording the `other` match arm uses
/// for a format this crate genuinely cannot read at all.
fn import_tile_archive_layer(
    name: &str,
    layer_type: &str,
    source: Option<&Map<String, Value>>,
    notices: &mut Vec<String>,
) {
    let url = source
        .and_then(|source| source.get("url"))
        .and_then(Value::as_str);
    match url {
        Some(url) if url.starts_with("http://") || url.starts_with("https://") => {
            notices.push(format!(
                "Layer \"{name}\" is a GeoLibre {layer_type} archive at {url}; OxiGIS can read \
                 PMTiles/MBTiles archives dropped directly onto the map, but GeoLibre project \
                 import does not yet reconstruct one from a URL, so it was skipped."
            ));
        }
        _ => {
            notices.push(format!(
                "Layer \"{name}\"'s {layer_type} archive was not saved with a public URL; \
                 skipped."
            ));
        }
    }
}

/// One GeoLibre `pmtiles`/`mbtiles` layer [`import`] recognised but could not
/// turn into a [`Layer`] by itself — see `import_tile_archive_layer`'s doc
/// comment for why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeferredArchive {
    /// The GeoLibre layer's display name.
    pub name: String,
    /// The archive's container format, decided by the GeoLibre `type` field
    /// (`"pmtiles"` / `"mbtiles"`), not by sniffing `url`.
    pub format: ArchiveFormat,
    /// The archive's `http(s)` URL — never a `blob:`/local reference, refused
    /// at collection time exactly like `import_cog_layer` refuses one for a
    /// local COG.
    pub url: String,
}

/// Collects every `pmtiles`/`mbtiles` layer of a GeoLibre document that names
/// a usable `http(s)` archive URL: the layers `import_layer` itself
/// recognises but cannot turn into a [`Layer`] without reading the archive's
/// header first (see the module docs and `import_tile_archive_layer`).
///
/// Call *alongside* [`import`], not instead of it — this performs no project
/// mutation and emits no notices of its own ([`import`]'s own per-layer
/// notice already explains why the layer did not appear); it only hands back
/// what a caller able to perform I/O (running a [`crate::archive::ArchiveProbe`]
/// per entry, the same probe a direct `.pmtiles`/`.mbtiles` drop already
/// runs) could still recover as a real layer.
///
/// A document that is not even a JSON object, or one with no `layers` array,
/// yields an empty list rather than an error — this is meant to be called
/// unconditionally alongside `import`, on the same `json` whether or not
/// `import` itself succeeded structurally.
#[must_use]
pub fn deferred_archives(json: &Value) -> Vec<DeferredArchive> {
    let Some(layers) = json
        .as_object()
        .and_then(|obj| obj.get("layers"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    layers
        .iter()
        .filter_map(|layer_value| {
            let layer_obj = layer_value.as_object()?;
            let format = match layer_obj.get("type").and_then(Value::as_str)? {
                "pmtiles" => ArchiveFormat::PmTiles,
                "mbtiles" => ArchiveFormat::MbTiles,
                _ => return None,
            };
            let url = layer_obj
                .get("source")
                .and_then(Value::as_object)
                .and_then(|source| source.get("url"))
                .and_then(Value::as_str)
                .filter(|url| url.starts_with("http://") || url.starts_with("https://"))?;
            let name = layer_obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(DEFAULT_LAYER_NAME)
                .to_string();
            Some(DeferredArchive {
                name,
                format,
                url: url.to_string(),
            })
        })
        .collect()
}

/// `type: "vector-tiles"`: only a raw `source.tiles` template is resolvable
/// — a TileJSON-only `source.url` needs a fetch this crate does not perform
/// (no I/O in this module) to discover its tile template, so it is refused.
/// `source.sourceLayers` (or the singular `sourceLayer`) becomes one
/// [`VectorTilePaint`] per name, all sharing the one collapsed style
/// (GeoLibre keeps no per-source-layer style of its own); the style is
/// collapsed against an *empty* [`FeatureCollection`] since no tile has
/// actually been fetched, which [`dominant_geometry_kind`] resolves to
/// [`GeometryKind::Point`] — a deliberate, precedented fallback (see that
/// function's own doc comment on why an empty/unknown collection defaults
/// there rather than to a fill or a line).
fn import_vector_tiles_layer(
    name: &str,
    source: Option<&Map<String, Value>>,
    style: Option<&Map<String, Value>>,
    notices: &mut Vec<String>,
) -> Option<MappedLayer> {
    let Some(source) = source else {
        notices.push(format!(
            "Layer \"{name}\" is a GeoLibre vector-tiles layer, which OxiGIS does not support; \
             skipped."
        ));
        return None;
    };
    let url_template = source
        .get("tiles")
        .and_then(Value::as_array)
        .and_then(|tiles| tiles.first())
        .and_then(Value::as_str);
    let Some(url_template) = url_template else {
        notices.push(format!(
            "Layer \"{name}\" needs a TileJSON URL, which OxiGIS does not resolve; skipped."
        ));
        return None;
    };
    let mut source_layers: Vec<String> = source
        .get("sourceLayers")
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if source_layers.is_empty()
        && let Some(single) = source.get("sourceLayer").and_then(Value::as_str)
    {
        source_layers.push(single.to_string());
    }
    if source_layers.is_empty() {
        notices.push(format!(
            "Layer \"{name}\" has no named source layers; skipped."
        ));
        return None;
    }
    let collapsed = collapse_style(&FeatureCollection::default(), style, name, notices);
    let paints = source_layers
        .into_iter()
        .map(|source_layer| VectorTilePaint::new(source_layer, collapsed.clone()))
        .collect();
    Some(MappedLayer {
        layer: Layer::new(
            name,
            LayerKind::Vector(VectorSource::MvtTiles {
                url_template: url_template.to_string(),
                paints,
            }),
        ),
        style: None,
    })
}

/// Collapses a GeoLibre flat `style` object into one [`LayerStyle`] (rule
/// 4): [`dominant_geometry_kind`] of `features` decides Fill vs. Line vs.
/// Circle (never [`LayerStyle::Symbol`] — this reader never produces one,
/// since a geometry style always wins over labels here, per rule 4), and the
/// §3 paint-field mapping decides each variant's fields:
///
/// * `fill-color = fillColor`, `fill-opacity = fillOpacity` (GeoLibre also
///   multiplies by `layer.opacity`, but that factor is carried separately —
///   see [`apply_layer_flags`] — so it is not baked in twice here);
/// * `line-color = strokeColor`, `line-width = strokeWidth`; GeoLibre has no
///   dedicated line-opacity style field (its `line-opacity` paint property
///   is just `layer.opacity`), so [`LineStyle`]'s own opacity is left at its
///   constructor default of `1.0` and the layer-level factor alone carries
///   it, reproducing the same product;
/// * `circle-color = fillColor` (not a separate field), `circle-radius =
///   circleRadius`, `circle-opacity = fillOpacity`, stroke = `strokeColor`/
///   `strokeWidth`.
///
/// `style` absent falls back to [`default_style_for_kind`], the same
/// default a fresh drop gets.
fn collapse_style(
    features: &FeatureCollection,
    style: Option<&Map<String, Value>>,
    name: &str,
    notices: &mut Vec<String>,
) -> LayerStyle {
    let kind = dominant_geometry_kind(features);
    let Some(style) = style else {
        return default_style_for_kind(kind);
    };

    let fill_color = read_color(style.get("fillColor"), name, notices);
    let stroke_color = read_color(style.get("strokeColor"), name, notices);
    let fill_opacity = style
        .get("fillOpacity")
        .and_then(Value::as_f64)
        .map(|value| value as f32);
    let stroke_width = style
        .get("strokeWidth")
        .and_then(Value::as_f64)
        .map(|value| value as f32);
    let circle_radius = style
        .get("circleRadius")
        .and_then(Value::as_f64)
        .map(|value| value as f32);

    if style
        .get("labels")
        .and_then(Value::as_object)
        .and_then(|labels| labels.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        notices.push(format!(
            "Layer \"{name}\" has both a geometry style and labels; OxiGIS keeps one style per \
             layer, so its labels were dropped."
        ));
    }

    match kind {
        GeometryKind::Polygon => {
            let mut fill = FillStyle::new(fill_color.unwrap_or_default());
            if let Some(opacity) = fill_opacity {
                fill.set_opacity(opacity);
            }
            fill.outline_color = stroke_color;
            LayerStyle::Fill(fill)
        }
        GeometryKind::Line => LayerStyle::Line(LineStyle::new(
            stroke_color.unwrap_or_default(),
            stroke_width.unwrap_or(DEFAULT_LINE_WIDTH_PX),
        )),
        GeometryKind::Point => {
            let mut circle = CircleStyle::new(
                circle_radius.unwrap_or(DEFAULT_CIRCLE_RADIUS_PX),
                fill_color.unwrap_or_default(),
            );
            circle.stroke_color = stroke_color;
            if let Some(width) = stroke_width {
                circle.set_stroke_width(width);
            }
            if let Some(opacity) = fill_opacity {
                circle.set_opacity(opacity);
            }
            LayerStyle::Circle(circle)
        }
    }
}

/// Reads a GeoLibre hex color field (rule 5): a 3-digit `#abc` shorthand is
/// expanded to 6 digits before [`Color::from_hex`] (which only accepts 6 or
/// 8), and anything else malformed — the wrong JSON type, an unparsable hex
/// string — falls back to [`Color::default`] with a notice rather than
/// failing the layer. An absent or `null` field is not malformed: it
/// silently returns [`None`], leaving the caller's own default in place.
fn read_color(value: Option<&Value>, name: &str, notices: &mut Vec<String>) -> Option<Color> {
    let value = value?;
    if value.is_null() {
        return None;
    }
    let Some(text) = value.as_str() else {
        notices.push(format!(
            "Layer \"{name}\" has a color that is not text; a default color was used instead."
        ));
        return Some(Color::default());
    };
    let body = text.strip_prefix('#').unwrap_or(text);
    let candidate = if body.len() == 3 && body.chars().all(|c| c.is_ascii_hexdigit()) {
        body.chars().flat_map(|c| [c, c]).collect::<String>()
    } else {
        text.to_string()
    };
    match Color::from_hex(&candidate) {
        Ok(color) => Some(color),
        Err(_) => {
            notices.push(format!(
                "Layer \"{name}\" has an unreadable color \"{text}\"; a default color was used \
                 instead."
            ));
            Some(Color::default())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DeferredArchive, deferred_archives, import, looks_like_geolibre};
    use oxigis_core::{
        ArchiveFormat, Color, LayerKind, LayerStyle, Project, RasterSource, VectorSource,
    };
    use serde_json::{Value, json};

    fn polygon_geojson() -> Value {
        json!({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                },
            }],
        })
    }

    #[test]
    fn inline_geojson_layer_imports_features_and_collapses_style() {
        let doc = json!({
            "version": "0.2.0",
            "name": "My Map",
            "mapView": {"center": [139.767, 35.681], "zoom": 10.0},
            "layers": [{
                "id": "layer-1",
                "type": "geojson",
                "visible": true,
                "opacity": 0.75,
                "style": {
                    "fillColor": "#ff0000",
                    "fillOpacity": 0.5,
                    "strokeColor": "#000000",
                    "strokeWidth": 2.0,
                },
                "geojson": polygon_geojson(),
            }],
            "styles": {},
        });

        let (project, notices) = import(&doc).expect("import succeeds");
        assert_eq!(project.name, "My Map");
        assert_eq!(project.view.center_lon, 139.767);
        assert_eq!(project.view.center_lat, 35.681);
        assert_eq!(project.view.zoom, 10.0);
        assert!(notices.is_empty(), "no notices expected: {notices:?}");

        assert_eq!(project.layers.len(), 1);
        let layer = &project.layers.layers()[0];
        assert!(layer.id.get() >= 1, "layer id must be a fresh numeric id");
        assert_eq!(layer.opacity(), 0.75);
        let LayerKind::Vector(VectorSource::InlineGeoJson { geojson }) = &layer.kind else {
            panic!("expected an inline GeoJSON layer");
        };
        assert!(geojson.contains("Polygon"));

        let style = project.styles.get(&layer.id).expect("style present");
        match style.base() {
            LayerStyle::Fill(fill) => {
                assert_eq!(fill.color, Color::from_rgb(0xff, 0x00, 0x00));
                assert_eq!(fill.opacity(), 0.5);
                assert_eq!(fill.outline_color, Some(Color::BLACK));
            }
            other => panic!("expected a Fill style, got {other:?}"),
        }
    }

    #[test]
    fn a_layers_minzoom_and_maxzoom_are_carried_and_sanitized() {
        let doc = json!({
            "version": "0.2.0",
            "name": "Scaled",
            "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [
                {
                    "id": "ranged",
                    "type": "geojson",
                    // MapLibre writes these as JSON numbers, so they arrive as
                    // f64 and are narrowed to the field's f32.
                    "minzoom": 6.0,
                    "maxzoom": 14.0,
                    "geojson": polygon_geojson(),
                },
                {
                    "id": "absurd",
                    "type": "geojson",
                    // Past MapLibre's own ceiling of 24: clamped, not refused,
                    // so a hand-edited style cannot produce a layer that draws
                    // nowhere.
                    "minzoom": 99.0,
                    "geojson": polygon_geojson(),
                },
                {
                    "id": "open",
                    "type": "geojson",
                    "geojson": polygon_geojson(),
                },
            ],
            "styles": {},
        });

        let (project, _notices) = import(&doc).expect("import succeeds");
        let layers = project.layers.layers();
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].min_zoom(), Some(6.0));
        assert_eq!(layers[0].max_zoom(), Some(14.0));
        assert_eq!(
            layers[1].min_zoom(),
            Some(oxigis_core::layer::MAX_ZOOM_LEVEL),
            "a bound past the ceiling clamps to it"
        );
        assert_eq!(layers[1].max_zoom(), None, "the undeclared end stays open");
        assert_eq!(
            (layers[2].min_zoom(), layers[2].max_zoom()),
            (None, None),
            "a layer declaring neither keeps the open range, so a project \
             written before this existed imports unchanged"
        );
    }

    #[test]
    fn styles_map_takes_precedence_over_layer_style() {
        let doc = json!({
            "version": "0.2.0",
            "name": "Styled",
            "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "layer-1",
                "type": "geojson",
                "style": {"fillColor": "#00ff00"},
                "geojson": polygon_geojson(),
            }],
            "styles": {"layer-1": {"fillColor": "#0000ff"}},
        });

        let (project, _notices) = import(&doc).expect("import succeeds");
        let layer = &project.layers.layers()[0];
        let style = project.styles.get(&layer.id).expect("style present");
        match style.base() {
            LayerStyle::Fill(fill) => assert_eq!(fill.color, Color::from_rgb(0x00, 0x00, 0xff)),
            other => panic!("expected Fill, got {other:?}"),
        }
    }

    #[test]
    fn xyz_layer_uses_the_tiles_array_url() {
        let doc = json!({
            "version": "0.2.0", "name": "XYZ", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "type": "xyz",
                "source": {
                    "tiles": ["https://tiles.example/{z}/{x}/{y}.png"],
                    "url": "https://fallback.example/tilejson.json",
                },
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(notices.is_empty(), "{notices:?}");
        let layer = &project.layers.layers()[0];
        match &layer.kind {
            LayerKind::Raster(RasterSource::Xyz { url_template, .. }) => {
                assert_eq!(url_template, "https://tiles.example/{z}/{x}/{y}.png");
            }
            other => panic!("expected an Xyz raster layer, got {other:?}"),
        }
    }

    #[test]
    fn xyz_layer_carries_the_documents_own_attribution() {
        let doc = json!({
            "version": "0.2.0", "name": "XYZ", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "type": "xyz",
                "source": {
                    "tiles": ["https://tiles.example/{z}/{x}/{y}.png"],
                    "attribution": "\u{a9} Example contributors",
                },
            }],
            "styles": {},
        });
        let (project, _notices) = import(&doc).expect("import succeeds");
        match &project.layers.layers()[0].kind {
            LayerKind::Raster(RasterSource::Xyz { attribution, .. }) => {
                assert_eq!(attribution, "\u{a9} Example contributors");
            }
            other => panic!("expected an Xyz raster layer, got {other:?}"),
        }
    }

    #[test]
    fn xyz_layer_falls_back_to_source_url_without_a_tiles_array() {
        let doc = json!({
            "version": "0.2.0", "name": "XYZ", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "type": "xyz",
                "source": {"url": "https://fallback.example/{z}/{x}/{y}.png"},
            }],
            "styles": {},
        });
        let (project, _notices) = import(&doc).expect("import succeeds");
        let layer = &project.layers.layers()[0];
        match &layer.kind {
            LayerKind::Raster(RasterSource::Xyz { url_template, .. }) => {
                assert_eq!(url_template, "https://fallback.example/{z}/{x}/{y}.png");
            }
            other => panic!("expected an Xyz raster layer, got {other:?}"),
        }
    }

    #[test]
    fn xyz_layer_with_tms_scheme_is_skipped_with_a_notice() {
        let doc = json!({
            "version": "0.2.0", "name": "XYZ", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "name": "TMS layer", "type": "xyz",
                "source": {
                    "tiles": ["https://tiles.example/{z}/{x}/{y}.png"],
                    "scheme": "tms",
                },
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(project.layers.is_empty());
        assert!(notices.iter().any(|n| n
            == "Layer \"TMS layer\" uses TMS tile numbering, which OxiGIS does not support; \
                skipped."));
    }

    #[test]
    fn cog_layer_with_a_public_url_imports() {
        let doc = json!({
            "version": "0.2.0", "name": "COG", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "type": "cog",
                "source": {"url": "https://example.test/scene.tif"},
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(notices.is_empty(), "{notices:?}");
        let layer = &project.layers.layers()[0];
        match &layer.kind {
            LayerKind::Raster(RasterSource::Cog { url }) => {
                assert_eq!(url, "https://example.test/scene.tif");
            }
            other => panic!("expected a Cog raster layer, got {other:?}"),
        }
    }

    #[test]
    fn cog_layer_with_only_a_blob_url_is_skipped_with_a_notice() {
        let doc = json!({
            "version": "0.2.0", "name": "COG", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "name": "Local scan", "type": "cog",
                "metadata": {"localBytesUrl": "blob:https://web.geolibre.app/abc-def"},
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(project.layers.is_empty());
        assert!(
            notices.iter().any(|n| n
                == "Layer \"Local scan\"'s COG file was not saved with a public URL; skipped.")
        );
    }

    #[test]
    fn vector_tiles_layer_gets_one_paint_per_source_layer() {
        let doc = json!({
            "version": "0.2.0", "name": "VT", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "type": "vector-tiles",
                "source": {
                    "tiles": ["https://tiles.example/{z}/{x}/{y}.pbf"],
                    "sourceLayers": ["roads", "water", "buildings"],
                },
                "style": {"fillColor": "#336699", "fillOpacity": 0.8},
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(notices.is_empty(), "{notices:?}");
        let layer = &project.layers.layers()[0];
        match &layer.kind {
            LayerKind::Vector(VectorSource::MvtTiles {
                url_template,
                paints,
            }) => {
                assert_eq!(url_template, "https://tiles.example/{z}/{x}/{y}.pbf");
                assert_eq!(paints.len(), 3);
                let names: Vec<&str> = paints.iter().map(|p| p.source_layer.as_str()).collect();
                assert_eq!(names, vec!["roads", "water", "buildings"]);
                // No downloaded tile at import time, so the geometry family
                // is unknown; `dominant_geometry_kind` falls back to Point.
                assert!(matches!(paints[0].style, LayerStyle::Circle(_)));
                assert_eq!(paints[0].style, paints[1].style);
            }
            other => panic!("expected an MvtTiles vector layer, got {other:?}"),
        }
    }

    #[test]
    fn vector_tiles_layer_with_only_a_tilejson_url_is_skipped_with_a_notice() {
        let doc = json!({
            "version": "0.2.0", "name": "VT", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "name": "Basemap tiles", "type": "vector-tiles",
                "source": {"url": "https://tiles.example/tilejson.json"},
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(project.layers.is_empty());
        assert!(notices.iter().any(|n| n
            == "Layer \"Basemap tiles\" needs a TileJSON URL, which OxiGIS does not resolve; \
                skipped."));
    }

    #[test]
    fn a_pmtiles_layer_with_a_public_url_is_recognised_but_not_yet_wired() {
        let doc = json!({
            "version": "0.2.0", "name": "PM", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "name": "Basemap tiles", "type": "pmtiles",
                "source": {"url": "https://example.test/basemap.pmtiles"},
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(
            project.layers.is_empty(),
            "not wired to a Layer by import() itself yet"
        );
        assert!(
            notices.iter().any(|n| n.contains("Basemap tiles")
                && n.contains("PMTiles/MBTiles")
                && !n.contains("does not support")),
            "the notice must not claim OxiGIS cannot read the format at all: {notices:?}",
        );
    }

    #[test]
    fn an_mbtiles_layer_without_a_public_url_is_refused_like_a_local_cog() {
        let doc = json!({
            "version": "0.2.0", "name": "MB", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "name": "Local tiles", "type": "mbtiles",
                "metadata": {"localBytesUrl": "blob:https://web.geolibre.app/abc-def"},
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(project.layers.is_empty());
        assert!(
            notices
                .iter()
                .any(|n| n.contains("Local tiles") && n.contains("public URL")),
            "{notices:?}"
        );
    }

    #[test]
    fn deferred_archives_collects_only_pmtiles_and_mbtiles_layers_with_public_urls() {
        let doc = json!({
            "version": "0.2.0", "name": "Mixed", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [
                {
                    "id": "l1", "name": "Basemap", "type": "pmtiles",
                    "source": {"url": "https://tiles.example/basemap.pmtiles"},
                },
                {
                    "id": "l2", "name": "Offline", "type": "mbtiles",
                    "source": {"url": "https://tiles.example/offline.mbtiles"},
                },
                {
                    "id": "l3", "name": "Local", "type": "pmtiles",
                    "metadata": {"localBytesUrl": "blob:https://web.geolibre.app/xyz"},
                },
                {
                    "id": "l4", "type": "geojson", "geojson": polygon_geojson(),
                },
            ],
            "styles": {},
        });
        let archives = deferred_archives(&doc);
        assert_eq!(
            archives,
            vec![
                DeferredArchive {
                    name: "Basemap".to_string(),
                    format: ArchiveFormat::PmTiles,
                    url: "https://tiles.example/basemap.pmtiles".to_string(),
                },
                DeferredArchive {
                    name: "Offline".to_string(),
                    format: ArchiveFormat::MbTiles,
                    url: "https://tiles.example/offline.mbtiles".to_string(),
                },
            ],
            "the blob-url pmtiles layer and the geojson layer must both be excluded",
        );
    }

    #[test]
    fn deferred_archives_on_a_document_with_no_layers_array_is_empty() {
        assert_eq!(deferred_archives(&json!({"version": "0.2.0"})), Vec::new());
        assert_eq!(deferred_archives(&json!("not an object")), Vec::new());
        assert_eq!(deferred_archives(&json!({"layers": []})), Vec::new());
    }

    #[test]
    fn an_unsupported_layer_type_is_skipped_with_the_exact_wording() {
        let doc = json!({
            "version": "0.2.0", "name": "Unsupported", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{"id": "l1", "name": "Terrain", "type": "wms", "source": {}}],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(project.layers.is_empty());
        assert!(notices.iter().any(|n| n
            == "Layer \"Terrain\" is a GeoLibre wms layer, which OxiGIS does not support; \
                skipped."));
    }

    #[test]
    fn the_whole_project_notice_counts_unsupported_layers() {
        let doc = json!({
            "version": "0.2.0", "name": "Counted", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [
                {"id": "l1", "name": "A", "type": "wms"},
                {"id": "l2", "name": "B", "type": "video"},
            ],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(project.layers.is_empty());
        assert_eq!(
            notices[0],
            "Imported a GeoLibre project; basemap, groups, and 2 unsupported layers were \
             dropped."
        );
        assert_eq!(notices.len(), 3, "whole-project + 2 per-layer: {notices:?}");
    }

    #[test]
    fn a_present_basemap_triggers_the_whole_project_notice_even_with_nothing_unsupported() {
        let doc = json!({
            "version": "0.2.0", "name": "Basemap", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "basemapStyleUrl": "https://tiles.openfreemap.org/styles/liberty",
            "layers": [],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(project.layers.is_empty());
        assert_eq!(
            notices,
            vec![
                "Imported a GeoLibre project; basemap, groups, and 0 unsupported layers were \
                 dropped."
                    .to_string()
            ]
        );
    }

    #[test]
    fn three_digit_hex_colors_are_expanded() {
        let doc = json!({
            "version": "0.2.0", "name": "Hex", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "type": "geojson",
                "style": {"fillColor": "#0f0"},
                "geojson": polygon_geojson(),
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        let layer = &project.layers.layers()[0];
        let style = project.styles.get(&layer.id).expect("style present");
        match style.base() {
            LayerStyle::Fill(fill) => assert_eq!(fill.color, Color::from_rgb(0x00, 0xff, 0x00)),
            other => panic!("expected Fill, got {other:?}"),
        }
        assert!(
            notices.is_empty(),
            "a valid 3-digit hex must not produce a notice: {notices:?}"
        );
    }

    #[test]
    fn a_malformed_color_falls_back_to_default_with_a_notice() {
        let doc = json!({
            "version": "0.2.0", "name": "Bad color", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "name": "Odd fill", "type": "geojson",
                "style": {"fillColor": "not-a-color"},
                "geojson": polygon_geojson(),
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        let layer = &project.layers.layers()[0];
        let style = project.styles.get(&layer.id).expect("style present");
        match style.base() {
            LayerStyle::Fill(fill) => assert_eq!(fill.color, Color::default()),
            other => panic!("expected Fill, got {other:?}"),
        }
        assert!(notices.iter().any(|n| n.contains("unreadable color")));
    }

    #[test]
    fn a_rotated_or_tilted_view_is_noted() {
        let doc = json!({
            "version": "0.2.0", "name": "Rotated",
            "mapView": {"center": [0.0, 0.0], "zoom": 1.0, "bearing": 45.0, "pitch": 0.0},
            "layers": [],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert_eq!(project.view.center_lon, 0.0);
        assert!(notices.iter().any(|n| n.contains("rotated or tilted")));
    }

    #[test]
    fn a_layer_added_through_add_vector_layer_uses_metadata_embedded_geojson() {
        // GeoLibre's own "Add Vector Layer" control strips `layer.geojson`
        // unconditionally and saves the same payload under
        // `metadata.embeddedGeoJSON` instead (audit §3) — this must not be
        // mistaken for "no data, only a path" and skipped.
        let doc = json!({
            "version": "0.2.0", "name": "Added", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "type": "geojson",
                "geojson": null,
                "metadata": {"embeddedGeoJSON": polygon_geojson()},
                "style": {"fillColor": "#ff0000"},
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(notices.is_empty(), "{notices:?}");
        let layer = &project.layers.layers()[0];
        let LayerKind::Vector(VectorSource::InlineGeoJson { geojson }) = &layer.kind else {
            panic!("expected an inline GeoJSON layer");
        };
        assert!(geojson.contains("Polygon"));
        match project.styles.get(&layer.id).map(|set| set.base()) {
            Some(LayerStyle::Fill(fill)) => {
                assert_eq!(fill.color, Color::from_rgb(0xff, 0x00, 0x00));
            }
            other => panic!("expected a Fill style, got {other:?}"),
        }
    }

    #[test]
    fn source_path_shp_maps_to_local_shapefile() {
        let doc = json!({
            "version": "0.2.0", "name": "Shapefile", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{"id": "l1", "type": "geojson", "sourcePath": "/data/cities.shp"}],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(notices.is_empty(), "{notices:?}");
        let layer = &project.layers.layers()[0];
        match &layer.kind {
            LayerKind::Vector(VectorSource::LocalShapefile { path }) => {
                assert_eq!(path, "/data/cities.shp");
            }
            other => panic!("expected a LocalShapefile layer, got {other:?}"),
        }
        assert!(
            !project.styles.contains_key(&layer.id),
            "path-only layers get no style at import time"
        );
    }

    #[test]
    fn source_path_parquet_maps_to_local_geoparquet() {
        let doc = json!({
            "version": "0.2.0", "name": "Parquet", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{"id": "l1", "type": "geojson", "sourcePath": "/data/cities.geoparquet"}],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(notices.is_empty(), "{notices:?}");
        let layer = &project.layers.layers()[0];
        match &layer.kind {
            LayerKind::Vector(VectorSource::LocalGeoParquet { path }) => {
                assert_eq!(path, "/data/cities.geoparquet");
            }
            other => panic!("expected a LocalGeoParquet layer, got {other:?}"),
        }
    }

    #[test]
    fn source_path_gpkg_is_skipped_with_a_notice() {
        let doc = json!({
            "version": "0.2.0", "name": "GPKG", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "name": "Cities", "type": "geojson", "sourcePath": "/data/cities.gpkg",
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(project.layers.is_empty());
        assert!(
            notices
                .iter()
                .any(|n| n.contains("GeoPackage") && n.contains("table name"))
        );
    }

    #[test]
    fn source_path_with_an_unreadable_extension_is_skipped_with_a_notice() {
        let doc = json!({
            "version": "0.2.0", "name": "KML", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "name": "Trail", "type": "geojson", "sourcePath": "/data/trail.kml",
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(project.layers.is_empty());
        assert!(notices.iter().any(|n| n
            == "Layer \"Trail\" references /data/trail.kml, a format OxiGIS cannot read; \
                skipped."));
    }

    #[test]
    fn our_own_project_json_does_not_sniff_as_geolibre() {
        let project = Project::new("Native");
        let text = project.to_json_string().expect("serialize");
        let value: Value = serde_json::from_str(&text).expect("parse");
        assert!(!looks_like_geolibre(&value));
        // The real loader must stay authoritative and unaffected by this
        // module existing at all.
        assert!(Project::from_json_string(&text).is_ok());
    }

    #[test]
    fn a_plain_feature_collection_does_not_sniff_as_geolibre() {
        let value = json!({"type": "FeatureCollection", "features": []});
        assert!(!looks_like_geolibre(&value));
    }

    #[test]
    fn a_geolibre_document_sniffs_as_geolibre() {
        let value = json!({
            "version": "0.2.0", "name": "X", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [], "styles": {},
        });
        assert!(looks_like_geolibre(&value));
        // Per the audit's empirical probe: GeoLibre uses uuid layer ids and
        // has no `format_version`, so the real loader must reject it.
        assert!(Project::from_json_string(&value.to_string()).is_err());
    }

    #[test]
    fn empty_layers_array_imports_name_and_view_alone() {
        let doc = json!({
            "version": "0.2.0", "name": "Bare",
            "mapView": {"center": [12.5, -3.25], "zoom": 7.0},
            "layers": [],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert_eq!(project.name, "Bare");
        assert_eq!(project.view.center_lon, 12.5);
        assert_eq!(project.view.center_lat, -3.25);
        assert_eq!(project.view.zoom, 7.0);
        assert!(project.layers.is_empty());
        assert!(project.styles.is_empty());
        assert!(notices.is_empty());
    }

    #[test]
    fn labels_enabled_alongside_a_geometry_style_are_dropped_with_a_notice() {
        let doc = json!({
            "version": "0.2.0", "name": "Labels", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "name": "Cities", "type": "geojson",
                "style": {
                    "fillColor": "#ff0000",
                    "labels": {"enabled": true, "field": "name"},
                },
                "geojson": {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature", "properties": {},
                        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                    }],
                },
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        let layer = &project.layers.layers()[0];
        let style = project.styles.get(&layer.id).expect("style present");
        assert!(
            matches!(style.base(), LayerStyle::Circle(_)),
            "geometry style must win over labels"
        );
        assert!(notices.iter().any(|n| n.contains("labels were dropped")));
    }

    #[test]
    fn an_imported_project_survives_a_native_save_load_round_trip() {
        // TODO §1.4 is "project save/load round trip" — an imported project
        // must be exactly as round-trippable as one built natively: fresh
        // numeric `LayerId`s, `styles` keyed on them, and `MvtTiles::paints`
        // all have to survive `to_json_string` → `from_json_string` intact.
        let doc = json!({
            "version": "0.2.0", "name": "Round Trip",
            "mapView": {"center": [3.0, 4.0], "zoom": 6.0},
            "layers": [
                {
                    "id": "l1", "type": "geojson",
                    "style": {"fillColor": "#ff0000", "fillOpacity": 0.5},
                    "geojson": polygon_geojson(),
                },
                {
                    "id": "l2", "type": "xyz",
                    "source": {"tiles": ["https://tiles.example/{z}/{x}/{y}.png"]},
                },
                {
                    "id": "l3", "type": "vector-tiles",
                    "source": {
                        "tiles": ["https://tiles.example/{z}/{x}/{y}.pbf"],
                        "sourceLayers": ["roads"],
                    },
                    "style": {"strokeColor": "#00ff00", "strokeWidth": 2.0},
                },
            ],
            "styles": {},
        });
        let (project, _notices) = import(&doc).expect("import succeeds");
        let text = project.to_json_string().expect("serialize");
        let restored = Project::from_json_string(&text).expect("deserialize");
        assert_eq!(restored, project);
    }

    #[test]
    fn a_source_path_layer_reports_the_existing_browser_notice_on_rebuild() {
        // The inherited notice from `LocalInputState::rebuild_from_project`'s
        // `LocalGeoJson` arm (audit §8's third bullet) — proving the layer
        // this module emits is recognised as an ordinary local layer, not
        // some unknown kind that arm would silently skip.
        use crate::local_input::LocalInputState;

        let doc = json!({
            "version": "0.2.0", "name": "Path", "mapView": {"center": [0.0, 0.0], "zoom": 1.0},
            "layers": [{
                "id": "l1", "name": "Cities", "type": "geojson",
                "sourcePath": "/data/cities.geojson",
            }],
            "styles": {},
        });
        let (project, notices) = import(&doc).expect("import succeeds");
        assert!(notices.is_empty(), "{notices:?}");

        let mut browser = LocalInputState::with_path_support(false);
        let rebuild_notices = browser.rebuild_from_project(&project);
        assert!(
            rebuild_notices.iter().any(|n| n
                == "Layer \"Cities\" references the file /data/cities.geojson, which is not \
                    available in the browser; re-drop the file to restore it."),
            "{rebuild_notices:?}"
        );
    }
}
