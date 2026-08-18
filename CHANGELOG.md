# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - Unreleased

Development branch for the next patch release. Nothing released yet.

## [0.1.0] - 2026-08-18

Initial release. OxiGIS is a QGIS-class, Pure Rust, zero-FFI GIS application
built on the COOLJAPAN ecosystem, distributed as both a WASM-first browser
build and a native desktop binary from a single 5-crate workspace. A hosted
build of the browser shell is live at <https://gis.cooljapan.tech/>.

### Added

- **`oxigis-core`** — platform-independent application model: the layer
  model (raster/vector sources, archive references, layer stack), the style
  model (fill/line/circle/symbol paints, label styles, colors), the
  `.oxigis.json` project file format (serde-based, versioned), and a
  Processing registry that maps OxiGeo functions to auto-generated parameter
  forms.
- **`oxigis-render`** — wgpu tile map renderer: raster tiles (XYZ and
  Cloud-Optimized GeoTIFF over HTTP Range), vector tiles (MVT: fill / line /
  circle / symbol), and PMTiles archive reading. Includes a label engine
  (glyph atlas, greedy placement, vertical/bidi-aware text orientation),
  Web Mercator math, and a tile pyramid with LRU caching. Deliberately
  I/O-free and GPU-context-free so it builds identically for native and
  `wasm32-unknown-unknown`.
- **`oxigis-ui`** — egui panels and application logic: layer tree, style
  editor, attribute table, and Processing dialogs auto-generated from the
  `oxigis-core` registry. Local and archive data ingestion for GeoJSON
  (dropped, pasted, or embedded in the project), Shapefile, GeoPackage
  (SQLite-backed), GeoParquet (optional `geoparquet` feature), and paged
  MBTiles/PMTiles archives. A full vector editing system (sketch,
  snapping, topology, undo/redo stack, selection, hit-testing, attribute
  forms, clipboard). A print/export pipeline producing PDF output with
  bidi text shaping and font subsetting. GeoLibre project-format import
  compatibility.
- **`oxigis-web`** — WASM shell hosting `oxigis-ui`'s `OxigisApp` via
  `eframe`/`wasm-bindgen`, requesting the WebGPU backend with automatic
  WebGL2 fallback. Browser-native font fetch, HTTP range fetch, and tile
  fetch.
- **`oxigis-desktop`** — native single-binary shell (`oxigis`) for Linux,
  macOS, and Windows via `winit`/`eframe`, sharing the same `oxigis-ui`
  panels and `oxigis-render` map view as the web build. Blocking HTTP
  tile/range transport on a worker pool, local file range reads, and
  background CJK font discovery for label fallback.
- **Coordinate reference systems** — a built-in EPSG registry with complete
  Japanese coverage (JGD2011 / JGD2000 / Tokyo geographic, all 19
  plane-rectangular zones of each, and their UTM zones) alongside the common
  global CRSs. A layer in a supported projected CRS is reprojected at ingest,
  so everything downstream stays in WGS84; the layer panel names each layer's
  CRS and warns where a historic datum's shift is only meter-accurate.
- **Project files** — `.oxigis.json` New / Open / Save / Save As on the
  desktop, with native file dialogs on macOS and Windows. A save is written to
  a temporary file in the destination directory and then renamed, so a full
  disk, a permission error or a crash mid-write leaves the previous save
  byte-for-byte intact.
- **Multi-layer tile compositing** — raster and vector-tile layers draw as a
  real stack in both shells, each with its own tile cache and its own opacity.
  Eight are drawn at once; anything beyond that is reported by name rather
  than silently dropped.
- **Categorized and graduated renderers** — classify a vector layer by exact
  attribute value or by numeric ranges, edited in a renderer panel. The
  classes are honored by the map, the legend, the printed page and
  hit-testing alike. Tiled (MVT) layers state plainly that they cannot be
  classified rather than offering a control that would do nothing.
- **Measurement and navigation** — geodesic distance and area measurement with
  a live readout, an on-screen scale bar, and go-to-coordinate.
- **Vector export** — GeoJSON export of a layer and CSV export of the
  attribute table.
- **Processing** — tools run asynchronously with a progress readout and are
  cancellable, optionally over the selection only, writing to a chosen
  destination. New Buffer tool for point and line inputs (polygon inputs are
  refused explicitly, since buffering an area also requires a dissolve).
- **Print furniture** — a segmented scale bar carrying its representative
  fraction, a north arrow, a legend (one row per visible layer, or one per
  class for a classified layer), and PDF `/Info` metadata (`/Title`,
  `/Creator`, `/Producer`, `/CreationDate`). 300 dpi joins the export
  resolutions, and the embedded map raster is encoded as JPEG (`/DCTDecode`)
  whenever that is smaller than the zlib stream, falling back to
  `/FlateDecode` automatically.
- **Web** — `#map=<zoom>/<lat>/<lon>` permalinks in the URL hash, and a
  loading indicator for outstanding network fetches.
- **Desktop** — command-line arguments (file paths to open, `--help`,
  `--version`, `--log-file`, plus a `RUST_LOG` filter) and session
  persistence: window geometry, the recent-project list, and the directory the
  file dialogs open in.
- Workspace-wide: Pure Rust, zero-FFI, `#![forbid(unsafe_code)]` across all
  five crates, Apache-2.0 licensed.
- `NOTICE`, covering Apache-2.0 §4(d) and the license obligations of every
  font program compiled into the shipped artifacts.
- Built against the COOLJAPAN ecosystem at `oxigeo` 0.2.4 (with
  `oxigeo-core` / `-proj` / `-geojson` / `-shapefile` / `-geoparquet`, the
  latter behind the optional `geoparquet` feature, and `oxiproj` 0.1.5
  resolved underneath `oxigeo-proj`), `oxiui-table` 0.2.1, `oxitext` 0.2.3,
  `oxifont-subset` / `-bundled` / `-discovery` 0.2.2, and `oxiarc-deflate` /
  `oxiarc-lzw` 0.4.1.

### Changed

Relative to the pre-release tree only — nothing here changes behavior for any
published version, because there is none.

- Maximum zoom raised to 24, with compile-time assertions guarding every
  constant that depends on it — the tile-index arithmetic in `oxigis-render`,
  and the MBTiles index's coordinate-field width in `oxigis-ui`, where a
  narrower field would alias into the zoom bits and answer a lookup with a
  different zoom's tile.
- Layer visibility, rename and zoom-range changes are undoable project
  operations rather than immediate edits.
- `futures` is a dev-dependency only. The async surface of `oxigis-render` is
  `Pin<Box<dyn Future>>` from `core`, so the crate owns no executor and a
  consumer never pulls one in through it.
- WebP tile decoding is enabled explicitly in both shells (`decode` +
  `decode-webp`) instead of relying on default features.

### Fixed

Found and fixed by the pre-release hardening audit — none of these ever
shipped, and they are listed because they are the failure modes the code is
now tested against.

- A striped GeoTIFF/COG whose height is not a multiple of `RowsPerStrip` has a
  short final strip; decoding it as a full-height block failed the whole map
  tile, so the bottom of such an image did not draw. Short blocks now fill
  their rows and leave the remainder transparent.
- A `.prj` or WKT string carrying a `TOWGS84[…]` Helmert clause is no longer
  read as WGS84: `TOWGS84` contains `WGS84` as a substring, and GDAL, ogr2ogr
  and QGIS emit that clause for every datum that has one, so a naive name match
  loaded Tokyo, ED50 and OSGB36 data unshifted. Every `TOWGS84[…]`,
  `AUTHORITY[…]` and `ID[…]` group is now stripped before a datum name is
  matched.
- A transport hiccup could poison a tile permanently. Tile failures are now
  classified: HTTP 4xx and undecodable bodies are remembered and never
  retried, while transport errors and 5xx are retried a bounded number of
  times behind a wall-clock exponential backoff, and a success clears the
  tile's history outright.
- A full glyph atlas no longer costs a frame its labels. Overflow is a routine
  condition with a routine answer: eviction is per glyph, survivors do not move
  and the generation is not bumped, so labels the frame is mid-draw stay valid.
  Clearing the whole atlas is the last resort, for the case where every glyph
  in it is still live.
- MVT ring winding is classified *relatively* rather than by the version-2
  absolute rule, so polygons from version-1 and non-conforming encoders keep
  their holes; a malformed layer or feature is isolated instead of failing the
  whole tile.
- A custom XYZ basemap template with no derivable host — a relative
  `/tiles/{z}/{x}/{y}.png`, say, which template validation rightly accepts —
  produced a credit line of `© ` with no holder. The attribution is now left
  empty, which hides the overlay, instead of crediting no one.
- Undo/redo correctness: coalesced opacity and style edits fold to one step,
  reorder undoes to the exact previous order, and non-finite class bounds from
  a hand-edited project load as finite extremes instead of poisoning the class
  list.

### Security

- Untrusted inputs are bounded by construction: a COG block is decompressed
  into a buffer of exactly the size its declared geometry allows (DEFLATE
  expands up to 1032:1, so an unbounded decode would turn a small range
  request into arbitrary memory pressure), the PMTiles directory reader caps
  entry counts, and attacker-controlled WKB vertex counts are allocated
  against a byte budget rather than trusted.
- `deny.toml` bans the C/C++ crypto and TLS stacks outright — `ring`,
  `aws-lc-sys`, `aws-lc-rs`, `openssl`, `openssl-sys`, `openssl-src`,
  `native-tls` and `security-framework-sys` — and admits `cc` only as a
  wrapper of `wayland-backend`, the one Linux windowing case with no
  pure-Rust equivalent. `cargo deny check bans` is part of the release gate.

[0.1.0]: https://github.com/cool-japan/oxigis/releases/tag/v0.1.0
