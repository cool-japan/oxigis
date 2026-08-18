//! Cloud-Optimized GeoTIFF reading over HTTP `Range` requests.
//!
//! A COG is a plain TIFF whose IFDs, tile directories and tiles are laid out so
//! that a client can read *part* of it: fetch the header, learn where every
//! tile of every overview level lives, then fetch only the tiles a view needs.
//! This module implements that, without performing any I/O itself.
//!
//! # Layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`tiff`] | byte-level TIFF primitives: byte order, header, IFD entries |
//! | [`blocks`] | the fetched-bytes store reads go through |
//! | [`open`] | the pull-based state machine that parses header + IFD chain |
//! | [`meta`] | parsed metadata: levels, tile directories, georeference |
//! | [`codec`] | tile payload → RGBA8 (decompress, predictor, sample mapping) |
//! | [`plan`] | placing the image on the Web Mercator tile grid |
//! | [`tmerc`] | Transverse Mercator, for UTM COGs |
//! | [`reader`] | an `async` convenience wrapper over all of the above |
//!
//! # Two ways to drive it
//!
//! * **Pull-based** ([`CogOpen`]) — the caller performs every fetch and hands
//!   the bytes back. This is what `oxigis-ui`'s COG tile provider uses, because
//!   its transport is a non-blocking callback rather than a future.
//! * **`async`** ([`CogSource`]) — the caller supplies a
//!   [`crate::source::RangeFetch`] and awaits. Convenient for tests and tools.
//!
//! # Supported files
//!
//! * Classic TIFF *and* BigTIFF (magic 43, 8-byte offsets, 20-byte IFD
//!   entries), `II` or `MM`. GDAL's COG driver switches to BigTIFF on its own
//!   once the output would pass 4 GB.
//! * Tiled *or* striped. A strip is a full-width block, and the last strip of
//!   an image whose height is not a multiple of `RowsPerStrip` is **short** —
//!   [`codec::decode_cog_block`] fills the rows it carries and leaves the rest
//!   transparent.
//! * Compression: none, LZW, JPEG (`JPEGTables` splicing included), DEFLATE
//!   (codes 8 and 32946), PackBits, WebP. LERC and ZSTD are refused by name.
//! * 8- and 16-bit samples, signed or unsigned, 1–4 bands (plus "first three
//!   of many").
//! * 8- and 16-bit palette colour through the `ColorMap` tag (320) — land
//!   cover and other classified rasters.
//! * `GDAL_NODATA` (tag 42113): matching pixels decode to alpha 0 and are kept
//!   out of the sample stretch.
//! * EPSG:3857, EPSG:4326 and the WGS 84 UTM zones (EPSG:32601–32660 and
//!   EPSG:32701–32760) via `ModelPixelScale` + `ModelTiepoint`. UTM is
//!   reprojected per pixel ([`tmerc`]); see [`meta::CogCrs`] for why anything
//!   else is rejected rather than mis-drawn.
//!
//! # Not covered
//!
//! * **Internal transparency masks.** A mask IFD is detected and kept out of
//!   the pyramid (so it is never drawn as imagery), but its tile directory is
//!   discarded rather than applied as alpha. Per-pixel transparency therefore
//!   comes from an alpha band or from `GDAL_NODATA`, not from a mask.
//! * **Per-file decode settings on [`CogMetadata`].** `GDAL_NODATA` and
//!   `JPEGTables` travel in a separate [`CogDecodeOptions`] — see
//!   [`open::CogOpen::decode_options`] and
//!   [`reader::CogSource::open_with_options`] — because [`meta::CogLevel`] has
//!   no field for them.
//! * CMYK, YCbCr outside JPEG, and CIELab are refused rather than mis-drawn;
//!   floating-point samples and the floating-point predictor likewise.
//!
//! # Provenance
//!
//! The TIFF/IFD parse and the tile decode path are ported from `oxigeo-wasm`'s
//! `src/cog_reader.rs` (cool-japan/oxigeo, Apache-2.0, same author), which reads
//! COGs in the browser through `fetch()`. Three things differ:
//!
//! * The async reader is turned inside out into [`CogOpen`], so the crate keeps
//!   its no-I/O portability contract and the UI can drive it from a callback.
//! * A single speculative header prefetch replaces the per-IFD 4 KiB windows,
//!   which collapses an open from 10–30 round trips to one.
//! * Web Mercator tile placement ([`plan`]) is new: `oxigeo-wasm` draws a COG in
//!   its own pixel space and never has to intersect it with a slippy-map grid.
//!
//! Fixes worth carrying back upstream are listed in `TODO.md` §5.1.

pub mod blocks;
pub mod codec;
pub mod meta;
pub mod open;
pub mod plan;
pub mod reader;
pub mod tiff;
pub mod tmerc;

// Fixture generation uses `expect` while assembling bytes it has just built
// itself; a failure there is a bug in the fixture, not a runtime condition.
// Outside this crate's own tests only `sample_cog_bytes` is reachable, so the
// builder's other knobs read as dead code.
#[cfg(any(test, feature = "fixtures"))]
#[allow(clippy::expect_used)]
#[cfg_attr(not(test), allow(dead_code))]
mod fixture;

/// Bytes of a tiny synthetic COG, for tests in this and other crates.
///
/// An 8×8 pixel, 8-bit greyscale, EPSG:4326 GeoTIFF with 4×4 tiles and one
/// half-resolution overview, georeferenced at 10 °E / 50 °N with 0.5 °/pixel.
/// Available with the `fixtures` cargo feature (and always inside this crate's
/// own tests), so `oxigis-ui`'s COG provider can be tested against exactly the
/// file this crate's parser is tested against.
#[cfg(any(test, feature = "fixtures"))]
#[must_use]
pub fn sample_cog_bytes() -> Vec<u8> {
    fixture::tiled_geo_tiff().bytes
}

/// Bytes of a tiny synthetic COG in a **projected** CRS, for tests.
///
/// [`sample_cog_bytes`]'s layout declared as EPSG:32654 (WGS 84 / UTM zone 54N)
/// and georeferenced at 380 000 E / 3 950 000 N with 10 m pixels — an 80 m
/// square west of Tokyo. Available with the `fixtures` cargo feature, so callers
/// can exercise the reprojecting path ([`tmerc`]) end to end.
#[cfg(any(test, feature = "fixtures"))]
#[must_use]
pub fn sample_utm_cog_bytes() -> Vec<u8> {
    fixture::utm_geo_tiff().bytes
}

pub use crate::cog::blocks::{BlockMiss, ByteBlocks, HEADER_PREFETCH_BYTES};
pub use crate::cog::codec::{
    CogDecodeOptions, RasterStretch, decode_cog_block, decode_cog_tile, decompress_tile,
};
pub use crate::cog::meta::{
    CogCrs, CogGeoTransform, CogLevel, CogMetadata, MAX_TILE_DECOMPRESSED_BYTES,
};
pub use crate::cog::open::{CogOpen, CogOpenProgress, MAX_IFD_CHAIN};
pub use crate::cog::plan::{
    COG_MAX_SOURCE_TILES, COG_OUTPUT_TILE_PX, CogSourceTile, CogTilePlan, CogTileRef,
};
pub use crate::cog::reader::{
    COG_HEADER_PROBE_BYTES, COG_IFD_WINDOW_BYTES, COG_MAX_CONCURRENT_TILE_FETCHES,
    COG_MAX_OVERVIEW_LEVELS, CogReadStep, CogSource, MemoryRangeFetch,
};
pub use crate::cog::tmerc::{
    TMERC_MAX_LAT_DEG, TMERC_MAX_LON_OFFSET_DEG, TransverseMercator, WGS84_INVERSE_FLATTENING,
    WGS84_SEMI_MAJOR_M, utm_central_meridian_deg,
};
