# oxigis-ui

OxiUI (egui) application layer for OxiGIS: layer tree, style and renderer
editors, attribute table, full vector editing, Processing dialogs, GeoJSON /
CSV export and PDF print export, all driven from one shell-agnostic
`OxigisApp`. Built on
`oxigis-core` (data model) and `oxigis-render` (the wgpu tile renderer) —
application logic and UI, not a narrow single-purpose API.

**Status:** Alpha

## Data ingestion

- **GeoJSON, ESRI Shapefile, OGC GeoPackage** (`local_input`,
  `shapefile_input`, `gpkg_input`) — bytes-only readers that converge on one
  GeoJSON `FeatureCollection` model, projected into a synthetic single-tile
  representation (`local_vector`) that feeds the same tessellator and label
  placer as tiled data. GeoPackage is read through an in-house, Pure Rust
  SQLite b-tree reader — no SQL engine, and it keeps building for
  `wasm32-unknown-unknown`.
- **GeoParquet** (`geoparquet_input`) — off by default, behind the
  `geoparquet` Cargo feature; enabled by `oxigis-desktop` (arrow/parquet are
  native-only).
- **MBTiles and PMTiles v3** single-file tile archives (`mbtiles`,
  `archive`) — both whole-file-in-memory and paged, byte-range reading, for
  raster and vector tiles alike. MBTiles reuses the same in-house SQLite
  reader as GeoPackage.
- **Cloud-Optimized GeoTIFF** (`cog_provider`) over HTTP `Range` requests.
- **GeoLibre project import** (`geolibre_import`) — best-effort
  `.geolibre.json` compatibility reader.
- **CRS-aware ingestion** — whatever a local vector format declares about its
  coordinate system (a Shapefile `.prj`'s WKT, a GeoPackage
  `gpkg_spatial_ref_sys` row, a GeoParquet `crs` object) is resolved to an
  `oxigis_core::Crs` and the `Reprojector` that places it, so a Japanese
  plane-rectangular shapefile lands where it belongs instead of near the
  origin. The predecessor was a two-value substring classifier whose WGS 84
  marker list matched the `TOWGS84` clause GDAL writes for every shifted
  datum, which silently mis-placed Tokyo Datum, NAD27 and ED50 data by
  hundreds of metres; the authority code is now read first and the shift
  clause stripped before any name is matched.

## Editing

Full vector feature editing (`edit`), built on a pure, total
command/transaction model that is testable without egui:

- Sketch tools, a world-space snapping grid, and topology validation (ring
  closure, repeated-vertex and spike detection, self-intersection,
  orientation).
- Selection and hit-testing, with explicit ordering rules (point beats line
  beats polygon; vertex beats midpoint beats feature).
- Attribute forms, clipboard (copy/paste), and an editing toolbar reporting
  user actions.
- An on-map overlay (selection outline, drag ghost) painted directly by
  egui, with no GPU re-tessellation.
- A byte-budgeted undo/redo stack with gesture-boundary coalescing and
  per-layer pruning, shared by both feature edits and reversible
  project-level operations (`edit::project_op`): layer add/remove, reorder,
  opacity, style, visibility, rename, and scale range. Every one of them
  records **both sides absolutely** rather than as a flip or a delta, so
  applying either direction is idempotent and a redo replayed after a
  reload cannot invert the wrong way; a multi-table drop records its whole
  gesture as one group, so a single Ctrl+Z cannot bisect it.

## Rendering / UI panels

- `OxigisApp` (`app`) — the top-level shell-agnostic UI, tying the panels
  together around a shared `Project`.
- Map viewport (`map_view`, `map_gpu`) — bridges `oxigis-render`'s wgpu
  renderer into egui's own render pass, with an egui-native fallback for
  hosts without a `wgpu` render state; local vector layers are drawn from
  their own ordered GPU mesh stack. Tiled layers compose as an **N-layer
  stack** — one `MapRenderer` per entry, each with its own opacity, kept in
  project order — instead of the one raster slot (a COG *or* a tile archive)
  plus one vector tileset the map started with. Both shells reconcile
  against it, and so does the printed page. The stack's texture and mesh
  byte budgets are shared out across its entries, so adding a layer costs
  memory rather than multiplying it.
- Layer tree (`layer_panel`) — visibility, opacity, reordering, removal.
- Style editor and style-model bridge (`style_panel`, `style_paint`),
  mapping `oxigis-core`'s `LayerStyle` onto `oxigis-render`'s paint types.
- Renderer editor (`renderer_panel`, reached through `app::renderer_ui`) —
  the Single / Categorized / Graduated selector, the attribute-field picker,
  per-class rows and a Classify helper. Class edits bind straight to the
  caller's `&mut LayerStyleSet`, so they ride the style panel's existing
  undo seam with no separate plumbing, and the map, the legend, hit-testing
  and the printed page all resolve a feature's style the same way.
- Attribute table (`attribute_table`, `table_panel`) — a scroll-virtualized
  view built on `oxiui-table`'s column/cell model.
- Processing toolbox (`processing_panel`, `processing_exec`) — forms
  auto-generated from tool descriptors, plus execution for the built-in
  tools (`bounds`, `feature_count`, `centroid`, `simplify`, `convex_hull`,
  `buffer`). A run no longer happens inside the frame that clicked Run:
  every tool is a fold over the layer's features, hoisted into a
  `FeatureSink` accumulator and a `ToolPass` cursor, so the cursor *is* the
  progress and a caller that stops advancing it has cancelled the run. The
  native shell moves the pass onto a `std::thread` with an atomic progress
  counter and cancel token; `wasm32` — which has no thread to move it to —
  drives one bounded slice per frame so the browser keeps repainting. The
  panel adds a *Selected features only* restriction and an output
  destination (a new layer, a file, or GeoJSON text). `convex_hull` carries
  both shapes QGIS's `native:convexhull` does — one hull over the layer and
  one hull per feature — behind a `per_feature` parameter; the registered
  descriptor does not expose it yet, so a run through today's panel takes the
  whole-layer default.
- Measurement and navigation (`measure`, `scalebar`) — a measuring tape and
  an area tool computed on the **WGS 84 ellipsoid** rather than on the
  Mercator plane the map is drawn in (which over-states ground distance by
  `1/cos φ` — a factor of two at 60°N), a go-to-coordinate box that flies
  the camera, and an on-screen scale bar that *is* measured in the
  projection, since it answers how much ground the picture's centimetre
  covers.
- Raster XYZ and vector (MVT) basemap tile providers (`tile_provider`,
  `vector_provider`).

## Export

Vector and table export (`export`):

- A layer's features as GeoJSON, and the attribute table as CSV — the rows
  *currently shown*, in the order they are shown.
- `oxigis-ui` compiles to `wasm32` and owns no filesystem, so an export
  crosses to the shell the way a project save and a PDF do: this crate
  serializes, parks the bytes in a take-once slot
  (`OxigisApp::take_pending_export`), and the shell writes them wherever its
  platform writes files — a Save dialog on the desktop, a download in the
  browser — then reports back through exactly one of
  `confirm_export_written` / `report_export_failed` / `cancel_pending_export`.

PDF export pipeline (`print`):

- Page layout with a raster basemap, a true vector overlay of local layers,
  and MVT tiles drawn as real, per-tile-clipped PDF paths — one pass per
  entry of the N-layer tile stack, so the page carries the same layers the
  screen does rather than one COG and one tileset.
- Map-sheet furniture: a 1-2-5 × 10ⁿ scale bar corrected for the Mercator
  scale factor at the view centre's latitude (`print::scalebar`), a north
  arrow drawn as PDF paths so it needs no glyph in any subset
  (`print::north`), a legend whose rows are the distinct style slots the map
  actually painted, classified renderers included (`print::legend`), and an
  `/Info` dictionary with `/Title`, `/Creator`, `/Producer` and
  `/CreationDate` (`print::meta`).
- The embedded map raster is raced honestly: a `/DCTDecode` JPEG encoding is
  measured against the `/FlateDecode` zlib stream and the smaller one is
  written, so a photographic basemap embeds at a fraction of the size while
  a flat one still takes the lossless path.
- Full UAX #9 bidirectional text analysis with a complete Unicode 16.0.0
  glyph-mirroring table, and vertical CJK title layout (upright glyphs with
  sideways Latin runs).
- Shaped text with subsetted, embedded `/Type0` composite fonts
  (`Identity-H`, `CIDFontType2`/`CIDFontType0`). Every Noto Sans CJK /
  Source Han Sans / Hiragino build is a **CID-keyed** CFF, which the
  subsetter hands back verbatim rather than rewriting; a viewer then
  resolves each CID through the program's own charset, so `print::cff`
  reads that charset (bounds-checked, `None` on anything malformed) to
  recover the glyph-id → CID mapping instead of assuming identity, which
  would print the wrong charstrings on any Adobe-Japan1 face.
- Assembled with `pdf-writer` plus the Pure Rust `oxiarc-deflate` compressor
  for `/FlateDecode`.

## Entry points

`oxigis-ui` is application logic, not a library meant to be called
piecemeal. `OxigisApp` exposes a single `ui(&mut self, ui: &mut egui::Ui)`
entry point that a shell calls once per frame; see `oxigis-desktop` (native,
winit) and `oxigis-web` (WASM, WebGPU) for the shells that actually embed
it.

Everything this crate cannot do itself — touch a file, open a socket, read a
font off the disk — leaves through a **take-once** seam the shell drains once
per frame and answers exactly once: `take_pending_project_save` /
`take_pending_project_open` (with `confirm_project_saved` /
`report_project_save_failed` coming back), `take_pending_export`,
`take_pending_print`, `take_pending_processing_save`,
`take_pending_archive_probe`, `take_pending_archive_pick` (a native Open
dialog, which a browser answers with a status line pointing at the drop
gesture), `take_pending_local_ops` (the local vector layers'
add/remove/visibility/opacity/style/reorder work, applied to the shell's GPU
mesh stack) and `take_pending_dropped_paths`. Taking a request removes it, so no
seam can be serviced twice or left pending.

**1565 tests passing** (`cargo nextest run -p oxigis-ui --all-features`; 1541
with the optional `geoparquet` feature off) — the most of any crate in the
OxiGIS workspace.

Part of [OxiGIS](https://github.com/cool-japan/oxigis) — Pure Rust full-stack GIS.
See the workspace README for the crate matrix and build instructions.

© 2026 COOLJAPAN OU (Team Kitasan) · Apache-2.0
