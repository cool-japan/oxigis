//! Raster tile byte decoding: PNG / JPEG (and optionally WebP) → RGBA8.
//!
//! # Why this lives inside a "no I/O" crate
//!
//! `oxigis-render`'s [portability contract][crate#portability-contract] bans
//! network/filesystem access so the crate stays usable on `wasm32-unknown-unknown`
//! without pulling in a platform HTTP stack. Decoding already-in-memory tile
//! bytes into pixels is **pure computation** — no syscalls, no async, no
//! platform capability — so it does not violate that rule. It is kept behind
//! the `decode` cargo feature (default-on) purely so consumers who already
//! have decoded pixels (e.g. a browser shell using `ImageBitmap`) can skip
//! the `image` dependency entirely.
//!
//! # Supported formats
//!
//! * PNG (`png` feature of `image`, pure Rust: `png` → `fdeflate`/`flate2`
//!   with the `miniz_oxide` backend).
//! * JPEG (`jpeg` feature of `image`, pure Rust: `zune-jpeg`).
//! * WebP, lossy and lossless, through the default-on `decode-webp` feature
//!   (pure Rust: `image-webp` → `quick-error`). Default-on because raster
//!   PMTiles archives in circulation store WebP tiles, and a build that could
//!   not read them would fail every tile of such an archive with a generic
//!   sniffing error; a consumer that wants those two crates gone builds with
//!   `default-features = false, features = ["decode"]`.
//!
//! AVIF is deliberately absent — no pure-Rust AVIF decoder is in the graph —
//! and a tile archive declaring it is refused by name rather than failing per
//! tile.
//!
//! No format here is backed by a C library or an `-sys` crate; see
//! `/deny.toml` for the exact accepted dependency chain.

use std::io::Cursor;

use crate::error::RenderError;
use crate::gpu::MAX_TILE_TEXTURE_SIZE;
use crate::renderer::DecodedTile;

/// Decodes raw tile bytes (PNG, JPEG, or — with the `decode-webp` feature —
/// lossless WebP) into a GPU-uploadable [`DecodedTile`].
///
/// The format is sniffed from the leading magic bytes; the caller does not
/// need to know ahead of time what an XYZ/COG tile server returned.
///
/// # Allocation bound
///
/// The header is read on its own first and the declared dimensions are checked
/// against [`MAX_TILE_TEXTURE_SIZE`] — the same limit
/// [`crate::gpu::TilePipeline::upload_tile`] enforces — *before* a single pixel
/// is allocated. `image`'s own default (a 512 MiB allocation cap, i.e. an
/// 11585-px square) is far above what this renderer can upload, so without this
/// check a hostile or merely oversized tile would decode into hundreds of
/// megabytes only to be refused at upload time.
///
/// # Errors
///
/// Returns [`RenderError::Decode`] if `bytes` is empty, truncated, not a
/// recognised/enabled image format, or otherwise fails to decode.
/// Returns [`RenderError::InvalidTileImage`] if the declared dimensions are
/// zero or larger than [`MAX_TILE_TEXTURE_SIZE`] on either axis.
pub fn decode_tile(bytes: &[u8]) -> Result<DecodedTile, RenderError> {
    if bytes.is_empty() {
        return Err(RenderError::Decode("empty tile buffer".to_string()));
    }

    let format = image::guess_format(bytes)
        .map_err(|err| RenderError::Decode(format!("could not sniff image format: {err}")))?;

    let (width, height) = image::ImageReader::with_format(Cursor::new(bytes), format)
        .into_dimensions()
        .map_err(|err| RenderError::Decode(format!("{format:?} header unreadable: {err}")))?;
    if width > MAX_TILE_TEXTURE_SIZE || height > MAX_TILE_TEXTURE_SIZE {
        return Err(RenderError::InvalidTileImage(format!(
            "tile {width}x{height} exceeds the {MAX_TILE_TEXTURE_SIZE} texel limit"
        )));
    }

    let image = image::load_from_memory_with_format(bytes, format)
        .map_err(|err| RenderError::Decode(format!("{format:?} decode failed: {err}")))?;

    let rgba = image.to_rgba8();
    let width = rgba.width();
    let height = rgba.height();
    let pixels = rgba.into_raw();

    DecodedTile::new(width, height, pixels)
}

#[cfg(test)]
mod tests {
    use super::decode_tile;
    use std::io::Cursor;

    /// Hand-encodes a tiny opaque PNG via `image`'s own (pure-Rust) encoder,
    /// then round-trips it through [`decode_tile`].
    fn make_png(width: u32, height: u32) -> Vec<u8> {
        let mut img = image::RgbaImage::new(width, height);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            *px = image::Rgba([r, g, 128, 255]);
        }
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encoding a tiny fixture PNG must succeed in tests");
        out
    }

    /// A hand-made 8x8 baseline JPEG (solid mid-gray), produced once with
    /// `image`'s JPEG encoder and embedded so decode tests don't depend on
    /// PNG round-tripping alone (exercises the `guess_format` JPEG branch).
    fn make_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_pixel(width, height, image::Rgb([200, 100, 50]));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Jpeg)
            .expect("encoding a tiny fixture JPEG must succeed in tests");
        out
    }

    #[test]
    fn decodes_png_dimensions_and_length() {
        let bytes = make_png(4, 6);
        let tile = decode_tile(&bytes).expect("valid png must decode");
        assert_eq!(tile.width(), 4);
        assert_eq!(tile.height(), 6);
        assert_eq!(tile.rgba().len(), 4 * 6 * 4);
    }

    #[test]
    fn png_alpha_is_preserved() {
        let bytes = make_png(2, 2);
        let tile = decode_tile(&bytes).expect("valid png must decode");
        for chunk in tile.rgba().chunks_exact(4) {
            assert_eq!(chunk[3], 255);
        }
    }

    /// A tiny lossless WebP, produced once with `image`'s own WebP encoder —
    /// the same `image::ImageFormat::WebP` round trip `cog::codec`'s
    /// `a_webp_tile_decodes` fixture uses. Gated on `decode-webp`: with the
    /// feature off, `image`'s WebP encoder does not exist to call.
    #[cfg(feature = "decode-webp")]
    fn make_webp(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(width, height, image::Rgba([10, 20, 30, 255]));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::WebP)
            .expect("encoding a tiny fixture WebP must succeed in tests");
        out
    }

    /// The WebP round trip for [`decode_tile`], gated because `image`'s WebP
    /// codec — and therefore [`make_webp`] itself — exists only when
    /// `decode-webp` is on. With the feature off, WebP bytes still sniff
    /// correctly (format detection is not feature-gated) but fail to decode
    /// through `image`'s own feature-gated decoder dispatch, surfacing as the
    /// ordinary `RenderError::Decode` path with no `#[cfg]` needed in this
    /// file; see the module docs.
    #[cfg(feature = "decode-webp")]
    #[test]
    fn decodes_webp_dimensions_and_length() {
        let bytes = make_webp(4, 6);
        let tile = decode_tile(&bytes).expect("valid webp must decode");
        assert_eq!(tile.width(), 4);
        assert_eq!(tile.height(), 6);
        assert_eq!(tile.rgba().len(), 4 * 6 * 4);
    }

    #[test]
    fn decodes_jpeg_dimensions_and_length() {
        let bytes = make_jpeg(8, 8);
        let tile = decode_tile(&bytes).expect("valid jpeg must decode");
        assert_eq!(tile.width(), 8);
        assert_eq!(tile.height(), 8);
        assert_eq!(tile.rgba().len(), 8 * 8 * 4);
    }

    #[test]
    fn jpeg_is_opaque() {
        let bytes = make_jpeg(4, 4);
        let tile = decode_tile(&bytes).expect("valid jpeg must decode");
        for chunk in tile.rgba().chunks_exact(4) {
            assert_eq!(chunk[3], 255);
        }
    }

    /// IEEE CRC-32, the one PNG chunks carry. Hand-rolled so the test can
    /// rewrite an IHDR without pulling a dependency in for four lines of table
    /// arithmetic.
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for byte in bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                let mask = 0u32.wrapping_sub(crc & 1);
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    /// A structurally valid PNG whose IHDR *declares* `width x height` while
    /// the pixel data is a 1x1 image — what a hostile tile server sends to make
    /// a decoder reserve hundreds of megabytes.
    fn png_declaring(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = make_png(1, 1);
        bytes[16..20].copy_from_slice(&width.to_be_bytes());
        bytes[20..24].copy_from_slice(&height.to_be_bytes());
        let checksum = crc32(&bytes[12..29]);
        bytes[29..33].copy_from_slice(&checksum.to_be_bytes());
        bytes
    }

    #[test]
    fn rejects_a_tile_larger_than_the_gpu_limit_before_decoding_it() {
        use crate::gpu::MAX_TILE_TEXTURE_SIZE;

        // 11585x11585 is inside `image`'s own 512 MiB allocation cap and would
        // decode into 536 MB of RGBA before `upload_tile` refused it.
        let huge = png_declaring(11_585, 11_585);
        let err = decode_tile(&huge).expect_err("an unusable tile must not decode");
        assert!(
            err.to_string().contains("texel limit"),
            "expected the texel-limit message, got {err}"
        );

        // The boundary itself is exact on both sides.
        let over = png_declaring(MAX_TILE_TEXTURE_SIZE + 1, 4);
        assert!(decode_tile(&over).is_err());
        let tall = png_declaring(4, MAX_TILE_TEXTURE_SIZE + 1);
        assert!(decode_tile(&tall).is_err());

        // A tile at the limit is refused only by its (missing) pixels, not by
        // the dimension check — i.e. the check is `>`, not `>=`.
        let at_limit = png_declaring(MAX_TILE_TEXTURE_SIZE, 1);
        let err = decode_tile(&at_limit).expect_err("the truncated body must fail");
        assert!(
            !err.to_string().contains("texel limit"),
            "8192 px is allowed, got {err}"
        );
    }

    #[test]
    fn rejects_empty_input() {
        let err = decode_tile(&[]).expect_err("empty buffer must not decode");
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn rejects_garbage_bytes() {
        let garbage = [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03];
        let err = decode_tile(&garbage).expect_err("garbage must not sniff as an image");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn rejects_truncated_png() {
        let full = make_png(4, 4);
        let truncated = &full[..full.len() / 2];
        let err = decode_tile(truncated).expect_err("truncated png must not decode");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn rejects_truncated_jpeg() {
        let full = make_jpeg(8, 8);
        let truncated = &full[..full.len() / 3];
        let err = decode_tile(truncated).expect_err("truncated jpeg must not decode");
        assert!(!err.to_string().is_empty());
    }
}
