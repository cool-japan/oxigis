# oxigis-core

Platform-independent core of [OxiGIS](https://github.com/cool-japan/oxigis): the layer
model, style and renderer models, the CRS/EPSG registry, the project file format,
and the Processing tool registry that `oxigis-ui`, `oxigis-web`, and
`oxigis-desktop` build on. Pure Rust, `#![forbid(unsafe_code)]`, with no rendering
or windowing dependencies.

**Status:** Alpha — pre-1.0 semver, the API may still move.

## Features

- **Layers** — `Layer`, `LayerKind` (raster/vector), and `LayerStack`, an ordered,
  back-to-front collection with add/remove/reorder/visibility/opacity operations that
  return `CoreError::LayerNotFound` instead of panicking. Stable, process-unique
  `LayerId`s that never collide, even across a reloaded project.
- **Sources** — *where* a layer's data lives (references, not loaders — this crate
  models the pointer, not the parser). Raster: XYZ tiles, Cloud-Optimized GeoTIFF,
  local GeoTIFF, and PMTiles/MBTiles archives. Vector: GeoJSON (file or inline),
  Shapefile, GeoPackage, GeoParquet, Mapbox Vector Tiles, and PMTiles/MBTiles archives.
- **Styles** — `FillStyle`, `LineStyle`, `CircleStyle`, `SymbolStyle` (a
  MapLibre-inspired subset), an RGBA8 `Color` that serializes as a hex string, and
  `LayerStyleSet` for per-geometry-family (polygon/line/point) overrides on a shared
  base style.
- **Renderers** — `Renderer` decides *which* style a feature gets, the way
  `LayerStyle` decides what that style looks like: `RendererKind::{Single,
  Categorized, Graduated}`, `CategoryClass`/`GraduatedClass` class lists (capped at
  `MAX_STYLE_CLASSES`), and a `Classification` summary. Resolution runs through
  `LayerStyleSet::style_for`, which asks `Renderer::class_of` for the class an
  `Attributes` value falls in, takes `Renderer::class_style` for it, and lays that
  over the geometry family's base style with `class_over_family` — so a classified
  layer and its legend, its hit-testing and its printed page all read the same
  answer. A renderer that classifies nothing serializes exactly as before it
  existed, so older `.oxigis.json` files round-trip byte for byte.
- **CRS** — `Crs` plus an EPSG registry (`crs::epsg::definition`) covering WGS 84,
  Web Mercator, the WGS 84 UTM zones, and Japan in full: every one of the 19 Japan
  Plane Rectangular zones on each of JGD2011 (6669–6687), JGD2000 (2443–2461) and
  Tokyo (30161–30179), plus the matching UTM and geographic CRSs. `crs::wkt` reads
  WKT1 and WKT2 (`parse_wkt`, `resolve_epsg`, `crs_label`), reading the root
  authority code first and stripping `TOWGS84[…]` before any name is matched — the
  substring trap that used to make every Tokyo Datum `.prj` look like WGS 84.
  `Datum::to_wgs84_helmert` carries the published Bursa-Wolf parameters for the
  shifted datums (Tokyo, OSGB36, ED50, NAD27), and `Reprojector` places a source's
  coordinates on lon/lat, with `AxisOrder` inference for the CRSs that declare
  northing first.
- **Project format** — `Project` bundles a layer stack, per-layer styles, view, and
  basemap into one serde-JSON document (`.oxigis.json`), tolerant of unknown fields so
  files keep loading across builds.
- **Processing registry** — `ProcessingRegistry` plus declarative `ToolDescriptor`/
  `ParamSpec` types a UI can turn into parameter forms without knowing each tool's
  Rust type. `builtin_registry()` ships descriptors for `bounds`, `feature_count`,
  `centroid`, `simplify`, `convex_hull`, and `buffer`; this crate holds only the
  descriptors and the `ToolExecutor` contract, because it depends on no OxiGeo types
  to run a tool against. Every built-in id is implemented against that contract in `oxigis-ui`'s
  `processing_exec` module, which dispatches `ToolDescriptor::id` to the matching
  executor from the Processing panel.
- **Errors** — every fallible operation returns `CoreError` (via `thiserror`) rather
  than panicking; no `.unwrap()`/`.expect()` on production paths.

## Quick Start

```rust
use oxigis_core::{Layer, LayerKind, Project, RasterSource};

fn main() -> oxigis_core::CoreResult<()> {
    let mut project = Project::new("My Map");
    let id = project.layers.add(Layer::new(
        "OSM",
        LayerKind::Raster(RasterSource::xyz(
            "https://tile.osm.example/{z}/{x}/{y}.png",
        )),
    ));
    project.layers.set_opacity(id, 0.8)?;

    let json = project.to_json_string()?;
    println!("{json}");
    Ok(())
}
```

## Testing

224 tests passing (`cargo nextest run -p oxigis-core`).

---

Part of [OxiGIS](https://github.com/cool-japan/oxigis) — Pure Rust full-stack GIS.
See the workspace README for the crate matrix and build instructions.

© 2026 COOLJAPAN OU (Team Kitasan) · Apache-2.0
