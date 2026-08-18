# oxigis-render

wgpu tile map renderer — raster (COG/XYZ), vector (MVT), labels.

**Status:** Alpha

`oxigis-render` draws slippy-map tiles with `wgpu`: raster basemaps from XYZ
servers or Cloud-Optimized GeoTIFF, PMTiles archives, Mapbox Vector Tiles,
and greedily-placed labels, all behind a four-call frame protocol
(`begin_frame` / `accept_tile` / `prepare` / `paint`) built to slot into
someone else's render pass (`egui_wgpu`, a bare `winit` surface, a headless
test harness).

## Features

- **Raster tiles** — XYZ URL templates, a wgpu textured-quad pipeline, and PNG/JPEG decoding behind the `decode` feature with WebP behind `decode-webp` (both on by default, because raster PMTiles archives in circulation store WebP).
- **Cloud-Optimized GeoTIFF (`cog`)** — a from-scratch TIFF/IFD reader over HTTP `Range` requests (pull-based `CogOpen`, or `async` via `CogSource`): classic TIFF *and* BigTIFF (magic 43, 8-byte offsets, 20-byte IFD entries — what GDAL's COG driver switches to past 4 GB), tiled or striped, LZW/DEFLATE/PackBits compression plus JPEG (with `JPEGTables` splicing) and WebP through the same `decode` / `decode-webp` gates, 8-bit palette rasters, `GDAL_NODATA` pixels decoded as transparent, and EPSG:3857/4326 plus UTM zones reprojected to Web Mercator via a built-in Transverse Mercator. LERC and ZSTD are refused by name rather than mis-decoded.
- **PMTiles (`pmtiles`)** — PMTiles v3 archives: header, varint directory codec, and Hilbert `(z, x, y)` addressing, opened with one speculative read.
- **Vector tiles (`mvt`, `vector`)** — a Mapbox Vector Tile 2.1 decoder that is total over hostile input (malformed tiles yield `RenderError::Mvt`, never a panic), tessellated with `lyon` into fill/line/circle meshes with per-tile scissoring.
- **Labels (`label`)** — `oxitext` shaping and `fontdue`-backed rasterization (`oxitext-raster`) into a glyph atlas, greedy collision-avoiding placement from decoded MVT features, halos, and UAX #50 vertical text orientation. The atlas tracks the row band it dirtied (`GlyphAtlas::dirty_rows`), so a frame that added one glyph uploads a few kilobytes instead of the whole texture.
- **Tile pyramid + cache** — Web Mercator math up to `MAX_ZOOM` 24 (with build-time `const` asserts keeping the shift and the f32 sub-rect exact), viewport tile selection that *wraps* columns modulo `2^zoom` so a view straddling the antimeridian is filled on both sides, over-zoom fallback onto a parent tile (`DEFAULT_OVERZOOM_LEVELS`, `set_overzoom_levels`) while children are still in flight, and a hand-rolled LRU tile cache (`tile_cache`) with no `unsafe`.
- **Bounded GPU memory** — the caches are byte-budgeted, not just entry-counted: `TileCache::with_byte_budget` evicts until both bounds hold and reports `CacheStats::bytes`, `MapRenderer::set_texture_byte_budget` caps decoded tile textures, and vector meshes recycle their buffers through `MeshBufferPool` (`recycle` / `byte_budget` / `stats`) rather than reallocating every frame. That is what makes running one `MapRenderer` per stack layer affordable; `set_opacity` then fades each of them independently.
- **Portable & Pure Rust** — no I/O: tile bytes are injected by the caller through `TileFetch` / `RangeFetch`, and font bytes are injected directly as `Vec<u8>`, so there is no HTTP client, async runtime or filesystem access anywhere in the crate. Compiles identically for native targets and `wasm32-unknown-unknown`; its codecs (PNG/JPEG/WebP, DEFLATE/LZW) carry no C library or `-sys` crate, and the crate is `#![forbid(unsafe_code)]` throughout.

## No `oxigis-core` dependency

`oxigis-render` deliberately does not depend on `oxigis-core`, so it can be
reused standalone by other projects that just need a tile renderer without
pulling in OxiGIS's layer/style/project model.

## Quick start

```rust
use oxigis_render::{DecodedTile, LonLat, MapRenderer, MapView, RenderError};

fn demo(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) -> Result<(), RenderError> {
    let view = MapView::new(LonLat::new(139.7, 35.7), 9.0, [1280.0, 720.0])?;
    let mut renderer = MapRenderer::new(view, 512, format)?;

    // Ask what the frame needs, feed back decoded pixels, then upload.
    for placement in renderer.begin_frame(view) {
        let _ = placement.tile;
    }
    for tile in renderer.missing_tiles().to_vec() {
        let rgba = vec![0u8; 256 * 256 * 4]; // fetched + decoded by the shell
        renderer.accept_tile(tile, DecodedTile::new(256, 256, rgba)?);
    }
    renderer.prepare(device, queue)?;
    Ok(())
}
```

589 tests passing.

Part of [OxiGIS](https://github.com/cool-japan/oxigis) — Pure Rust full-stack GIS.
See the workspace README for the crate matrix and build instructions.

© 2026 COOLJAPAN OU (Team Kitasan) · Apache-2.0
