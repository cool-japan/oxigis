//! Turning one COG block — a tile, or a strip — into RGBA8 pixels.
//!
//! Four steps, in this order, all of them pure computation:
//!
//! 1. **Decompress** — `None`, LZW, DEFLATE (both TIFF codes), PackBits, or an
//!    image codec (JPEG, WebP) that produces host-native samples itself.
//! 2. **Undo the predictor** — TIFF horizontal differencing (tag 317 = 2),
//!    which is defined over samples in the *file's* byte order.
//! 3. **Normalise byte order** — rewrite 16/32/64-bit samples into the host's
//!    order, so everything above this layer is host-native.
//! 4. **Map samples to RGBA8** — greyscale/RGB(A)/palette with an optional
//!    stretch for 16-bit data, and `GDAL_NODATA` mapped to alpha 0.
//!
//! Steps 2 and 3 are not interchangeable: swapping them silently corrupts
//! predicted big-endian (`MM`) tiles. Steps 2 and 3 also do not apply to the
//! image codecs, whose decoders emit host-native 8-bit samples directly —
//! the decompression step therefore reports which kind of buffer it produced
//! rather than leaving the caller to guess. This ordering, the predictor itself
//! and the byte-order normalisation are ported from `oxigeo-wasm`'s
//! `cog_reader.rs` (`finish_tile_decode`, `apply_horizontal_predictor`,
//! `normalize_samples_to_native`; cool-japan/oxigeo, Apache-2.0, same author).
//!
//! # Blocks are not always full
//!
//! TIFF pads a **tile** to its declared height, but not a **strip**: the last
//! strip of an image whose height is not a multiple of `RowsPerStrip` carries
//! only the rows that are left. [`decode_cog_block`] therefore takes the block's
//! row in the tile grid, decodes [`CogLevel::block_rows`] rows and leaves the
//! rest of the returned buffer transparent, so the result is always the full
//! `tile_width x tile_height x 4` bytes the compositor expects.
//! [`decode_cog_tile`] is the whole-block form of the same call.
//!
//! # Compression support
//!
//! | TIFF `Compression` | Codec | Backend |
//! |---|---|---|
//! | 1 | none | — |
//! | 5 | LZW (MSB-first, early change) | `oxiarc-lzw` |
//! | 7 | JPEG, abbreviated streams included | `image` (`decode` feature) |
//! | 8, 32946 | zlib-wrapped DEFLATE | `oxiarc-deflate` |
//! | 32773 | PackBits | this module |
//! | 50001 | WebP | `image` (`decode-webp` feature) |
//!
//! LERC (34887) and ZSTD (50000) report [`RenderError::Unsupported`]: both need
//! a decoder that is not in the dependency graph. `oxigeo-wasm`'s ranged reader
//! supports only 1 and 8, so everything else here is new.
//!
//! # Bounded by construction
//!
//! Tile width, height, band count and bit depth all come out of the file with
//! no ceiling of their own, and every codec here is asked to produce
//! `tile_width x tile_height x bands x bits/8` bytes. That product is capped at
//! [`MAX_TILE_DECOMPRESSED_BYTES`] before any buffer is reserved, and the
//! DEFLATE path decompresses *into* a buffer of exactly that size rather than
//! growing one, so a compression bomb (DEFLATE expands up to 1032:1) cannot
//! turn a small range request into arbitrary memory pressure.

use crate::cog::meta::{
    COMPRESSION_DEFLATE, COMPRESSION_DEFLATE_OLD, COMPRESSION_JPEG, COMPRESSION_LERC,
    COMPRESSION_LZW, COMPRESSION_NONE, COMPRESSION_PACKBITS, COMPRESSION_WEBP, COMPRESSION_ZSTD,
    CogLevel, MAX_TILE_DECOMPRESSED_BYTES, PHOTOMETRIC_BLACK_IS_ZERO, PHOTOMETRIC_PALETTE,
    PHOTOMETRIC_RGB, PHOTOMETRIC_WHITE_IS_ZERO, PHOTOMETRIC_YCBCR, SAMPLE_FORMAT_FLOAT,
    SAMPLE_FORMAT_INT,
};
use crate::error::RenderError;

/// Predictor value meaning "horizontal differencing".
const PREDICTOR_HORIZONTAL: u16 = 2;

/// Predictor value meaning "floating-point differencing" (unsupported).
const PREDICTOR_FLOATING_POINT: u16 = 3;

/// Largest factor by which PackBits can expand its input.
///
/// A two-byte repeat run emits at most 128 bytes, so the true worst case is
/// 64:1; rounding up keeps the bound obviously safe while still making a
/// reservation derived from a corrupt tile geometry provably wasted.
const PACKBITS_MAX_EXPANSION: usize = 128;

/// Number of histogram bins used by [`RasterStretch::Percentile`].
const PERCENTILE_BINS: usize = 1 << 16;

/// How 16-bit samples are mapped onto the 0..255 display range.
///
/// 8-bit imagery is already display-ready and ignores this setting.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum RasterStretch {
    /// Stretch each block between its own minimum and maximum sample.
    ///
    /// The default, because it makes any 16-bit product (Sentinel-2
    /// reflectance, a DEM, a radiance band) visible without the caller knowing
    /// its value range. The cost is that neighbouring blocks are stretched
    /// independently, so a scene with strongly varying content shows seams at
    /// block boundaries — use [`RasterStretch::Fixed`] once the range is known,
    /// or [`RasterStretch::Percentile`] to at least stop one outlier pixel from
    /// flattening a whole block.
    #[default]
    MinMaxPerTile,
    /// Stretch between two fixed sample values, identically for every block.
    Fixed {
        /// Sample value that maps to 0.
        min: f64,
        /// Sample value that maps to 255.
        max: f64,
    },
    /// Stretch between two percentiles of the block's own sample histogram.
    ///
    /// `Percentile { low_pct: 2.0, high_pct: 98.0 }` is what GDAL and QGIS
    /// reach for by default, and unlike [`RasterStretch::MinMaxPerTile`] a
    /// single saturated or nodata-adjacent pixel does not compress the rest of
    /// the block into a narrow band of greys. Percentiles outside `0..=100`, or
    /// inverted, fall back to the block's full range.
    Percentile {
        /// Percentile mapped to 0, e.g. `2.0`.
        low_pct: f64,
        /// Percentile mapped to 255, e.g. `98.0`.
        high_pct: f64,
    },
}

/// Everything a block decode needs beyond [`CogLevel`] itself.
///
/// These are per-*file* properties that [`CogLevel`] has no field for, so they
/// travel beside the metadata: [`super::CogOpen::decode_options`] collects them
/// while parsing the IFD chain, and [`super::CogSource`] carries them for the
/// `async` path.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CogDecodeOptions {
    /// How 16-bit samples are mapped onto the display range.
    pub stretch: RasterStretch,
    /// `GDAL_NODATA` (tag 42113): pixels whose first band equals it become
    /// fully transparent and are excluded from the stretch.
    ///
    /// A non-finite value never matches a sample, so `Some(f64::NAN)` reads as
    /// "declared, but not expressible as an integer sample".
    pub nodata: Option<f64>,
    /// `JPEGTables` (tag 347): the quantisation and Huffman tables shared by
    /// every abbreviated JPEG tile stream in the file.
    pub jpeg_tables: Vec<u8>,
}

impl CogDecodeOptions {
    /// Options carrying nothing but a stretch.
    #[must_use]
    pub fn with_stretch(stretch: RasterStretch) -> Self {
        Self {
            stretch,
            nodata: None,
            jpeg_tables: Vec::new(),
        }
    }
}

/// How the sample buffer a codec produced still has to be treated.
enum BlockSamples {
    /// Samples in the file's byte order: the predictor and the byte-order
    /// normalisation still apply.
    Raw(Vec<u8>),
    /// Host-native 8-bit samples an image codec produced, already in the layout
    /// it reports. Applying the predictor or a byte swap to these would corrupt
    /// them.
    #[cfg_attr(
        not(any(feature = "decode", feature = "decode-webp")),
        allow(
            dead_code,
            reason = "only JPEG and WebP construct it, and both are feature-gated"
        )
    )]
    Native(Vec<u8>, SampleLayout),
}

/// How to read the sample buffer that reached [`samples_to_rgba`].
///
/// Usually just the level's own tags, but an image codec decides its own output
/// layout: a JPEG tile of a `PhotometricInterpretation` 6 (YCbCr) image arrives
/// from the decoder already converted to RGB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampleLayout {
    /// Bits per sample of the buffer.
    bits_per_sample: u16,
    /// Bands per pixel of the buffer.
    samples_per_pixel: u16,
    /// Colour model of the buffer.
    photometric: u16,
    /// Whether samples are two's-complement signed.
    signed: bool,
}

impl SampleLayout {
    /// The layout a level's tags describe.
    fn of(level: &CogLevel) -> Self {
        Self {
            bits_per_sample: level.bits_per_sample,
            samples_per_pixel: level.samples_per_pixel,
            photometric: level.photometric,
            signed: level.sample_format == SAMPLE_FORMAT_INT,
        }
    }
}

/// Decompresses one block payload according to its level's `Compression` tag.
///
/// The result is the level's raw samples, in the *file's* byte order and with
/// the predictor still applied — [`decode_cog_block`] is what turns them into
/// pixels. JPEG and WebP payloads are the exception: their decoders emit
/// host-native 8-bit samples, so the predictor and byte-order steps must not be
/// applied to what this returns for them.
///
/// # Errors
///
/// * [`RenderError::Unsupported`] for a codec this reader does not implement,
///   or a block geometry past [`MAX_TILE_DECOMPRESSED_BYTES`].
/// * [`RenderError::Decode`] if the payload is malformed, or if the level's
///   geometry does not give a decompressed size to decode against.
pub fn decompress_tile(level: &CogLevel, payload: &[u8]) -> Result<Vec<u8>, RenderError> {
    match decompress_block(level, payload, &[])? {
        BlockSamples::Raw(samples) | BlockSamples::Native(samples, _) => Ok(samples),
    }
}

/// Largest decompressed size this level's block geometry allows, rejecting a
/// geometry that overflows or exceeds the cap before anything is reserved.
fn block_capacity(level: &CogLevel) -> Result<usize, RenderError> {
    let expected = level
        .tile_bytes()
        .ok_or_else(|| RenderError::Decode("COG block geometry overflows an address".to_owned()))?;
    if expected > MAX_TILE_DECOMPRESSED_BYTES {
        return Err(RenderError::Unsupported(format!(
            "COG block geometry needs {expected} bytes, past the \
             {MAX_TILE_DECOMPRESSED_BYTES}-byte limit; re-encode as a tiled COG"
        )));
    }
    Ok(expected)
}

/// Decompresses one block, reporting whether its samples still need the
/// predictor and byte-order steps.
fn decompress_block(
    level: &CogLevel,
    payload: &[u8],
    jpeg_tables: &[u8],
) -> Result<BlockSamples, RenderError> {
    let expected = block_capacity(level)?;
    match level.compression {
        COMPRESSION_NONE => Ok(BlockSamples::Raw(payload.to_vec())),
        COMPRESSION_LZW => oxiarc_lzw::decompress_tiff(payload, expected)
            .map(BlockSamples::Raw)
            .map_err(|error| RenderError::Decode(format!("COG LZW decode failed: {error}"))),
        COMPRESSION_DEFLATE | COMPRESSION_DEFLATE_OLD => {
            inflate_bounded(payload, expected).map(BlockSamples::Raw)
        }
        COMPRESSION_PACKBITS => Ok(BlockSamples::Raw(unpack_bits(payload, expected))),
        COMPRESSION_JPEG => decode_jpeg_block(level, payload, jpeg_tables),
        COMPRESSION_WEBP => decode_webp_block(level, payload),
        COMPRESSION_ZSTD => Err(RenderError::Unsupported(
            "ZSTD-compressed COG tiles (Compression 50000) are not supported: no pure-Rust ZSTD \
             decoder is in this crate's dependency graph"
                .to_owned(),
        )),
        COMPRESSION_LERC => Err(RenderError::Unsupported(
            "LERC-compressed COG tiles (Compression 34887) are not supported: no pure-Rust LERC \
             decoder is in this crate's dependency graph"
                .to_owned(),
        )),
        other => Err(RenderError::Unsupported(format!(
            "COG Compression {other} is not supported (supported: 1, 5, 7, 8, 32773, 32946, 50001)"
        ))),
    }
}

/// Inflates a zlib-wrapped DEFLATE payload into a buffer of exactly `expected`
/// bytes.
///
/// DEFLATE expands up to 1032:1, so growing an output vector to whatever the
/// stream asks for lets a one-megabyte range request cost a gigabyte of memory.
/// Decompressing *into* the declared block size bounds that by the (already
/// capped) geometry instead. A stream that decodes to fewer bytes is kept as-is
/// — that is exactly what a short final strip looks like.
fn inflate_bounded(payload: &[u8], expected: usize) -> Result<Vec<u8>, RenderError> {
    let mut buffer = vec![0u8; expected];
    let written = oxiarc_deflate::zlib_decompress_into(payload, &mut buffer)
        .map_err(|error| RenderError::Decode(format!("COG DEFLATE decode failed: {error}")))?;
    buffer.truncate(written.min(expected));
    Ok(buffer)
}

/// Decodes a PackBits (TIFF Compression 32773) run-length stream.
///
/// The format is Apple's: a signed control byte `n` means "copy the next
/// `n + 1` literal bytes" for `0..=127`, "repeat the next byte `1 - n` times"
/// for `-127..=-1`, and is a no-op for `-128`. A truncated stream simply ends
/// the output early rather than failing, matching how TIFF readers treat a
/// short final run.
///
/// `capacity` only sizes the buffer, and is clamped to what the payload can
/// provably produce: a block geometry read out of the file is free to claim
/// gigabytes that a one-byte payload can never fill.
#[must_use]
fn unpack_bits(payload: &[u8], capacity: usize) -> Vec<u8> {
    let reachable = payload.len().saturating_mul(PACKBITS_MAX_EXPANSION);
    let mut out = Vec::with_capacity(capacity.min(reachable));
    let mut index = 0usize;
    while let Some(&control) = payload.get(index) {
        index += 1;
        let control = control as i8;
        if control >= 0 {
            let run = control as usize + 1;
            let Some(literal) = payload.get(index..index + run) else {
                // Truncated literal run: copy whatever is left and stop.
                if let Some(tail) = payload.get(index..) {
                    out.extend_from_slice(tail);
                }
                break;
            };
            out.extend_from_slice(literal);
            index += run;
        } else if control != i8::MIN {
            let run = 1 - i32::from(control);
            let Some(&value) = payload.get(index) else {
                break;
            };
            index += 1;
            for _ in 0..run {
                out.push(value);
            }
        }
    }
    out
}

/// Splices a shared `JPEGTables` (tag 347) stream into an abbreviated tile
/// stream, producing a self-contained JFIF byte sequence.
///
/// A JPEG-compressed TIFF stores the quantisation and Huffman tables once, in
/// tag 347, as a tables-only stream (`SOI … EOI`); each tile then carries only
/// its own `SOF`/`SOS` and entropy-coded data behind an `SOI`. Concatenating
/// the tables without their `EOI` and the tile without its `SOI` is what
/// libtiff and GDAL do, and what every JPEG decoder expects.
///
/// A tile that already carries its own tables (some writers repeat them) is
/// still correct after splicing: a later `DQT`/`DHT` simply redefines the slot.
#[cfg(any(test, feature = "decode"))]
fn splice_jpeg_tables(tables: &[u8], payload: &[u8]) -> Vec<u8> {
    const SOI: [u8; 2] = [0xFF, 0xD8];
    const EOI: [u8; 2] = [0xFF, 0xD9];

    let tables_are_a_stream = tables.len() > 4
        && tables.get(..2) == Some(&SOI[..])
        && tables.get(tables.len() - 2..) == Some(&EOI[..]);
    let payload_starts_with_soi = payload.get(..2) == Some(&SOI[..]);
    if !tables_are_a_stream || !payload_starts_with_soi {
        return payload.to_vec();
    }
    let head = tables.get(..tables.len() - 2).unwrap_or(&[]);
    let tail = payload.get(2..).unwrap_or(&[]);
    let mut spliced = Vec::with_capacity(head.len() + tail.len());
    spliced.extend_from_slice(head);
    spliced.extend_from_slice(tail);
    spliced
}

/// Decodes an image-codec block against the level's declared block geometry.
///
/// The dimensions in a JPEG `SOF` or a WebP header are the tile's claim about
/// itself, and `image`'s default limits leave width and height unbounded (only
/// `max_alloc`, at 512 MiB, is set). A tile declaring 20000x20000 would
/// therefore materialise hundreds of megabytes before any check of ours ran —
/// and [`super::CogSource::compose`] now has up to
/// [`super::COG_MAX_SOURCE_TILES`] decodes in flight at once. Handing the
/// already-capped block geometry to the decoder as a *limit* moves that
/// rejection ahead of the allocation.
#[cfg(any(feature = "decode", feature = "decode-webp"))]
fn decode_image_block(
    level: &CogLevel,
    stream: &[u8],
    format: image::ImageFormat,
) -> Result<image::DynamicImage, RenderError> {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(level.tile_width);
    limits.max_image_height = Some(level.tile_height);
    limits.max_alloc = Some(MAX_TILE_DECOMPRESSED_BYTES as u64);
    let mut reader = image::ImageReader::with_format(std::io::Cursor::new(stream), format);
    reader.limits(limits);
    reader
        .decode()
        .map_err(|error| RenderError::Decode(format!("COG {format:?} decode failed: {error}")))
}

/// Decodes a JPEG (Compression 7) block through `image`'s pure-Rust decoder.
#[cfg(feature = "decode")]
fn decode_jpeg_block(
    level: &CogLevel,
    payload: &[u8],
    jpeg_tables: &[u8],
) -> Result<BlockSamples, RenderError> {
    let stream = splice_jpeg_tables(jpeg_tables, payload);
    let decoded = decode_image_block(level, &stream, image::ImageFormat::Jpeg)?;
    image_to_samples(level, &decoded, PHOTOMETRIC_RGB)
}

/// Reports JPEG as unsupported when the `decode` feature is off.
#[cfg(not(feature = "decode"))]
fn decode_jpeg_block(
    _level: &CogLevel,
    _payload: &[u8],
    _jpeg_tables: &[u8],
) -> Result<BlockSamples, RenderError> {
    Err(RenderError::Unsupported(
        "JPEG-compressed COG tiles (Compression 7) need the `decode` cargo feature".to_owned(),
    ))
}

/// Decodes a WebP (Compression 50001) block through `image`'s pure-Rust
/// decoder.
#[cfg(feature = "decode-webp")]
fn decode_webp_block(level: &CogLevel, payload: &[u8]) -> Result<BlockSamples, RenderError> {
    let decoded = decode_image_block(level, payload, image::ImageFormat::WebP)?;
    image_to_samples(level, &decoded, PHOTOMETRIC_RGB)
}

/// Reports WebP as unsupported when the `decode-webp` feature is off.
#[cfg(not(feature = "decode-webp"))]
fn decode_webp_block(_level: &CogLevel, _payload: &[u8]) -> Result<BlockSamples, RenderError> {
    Err(RenderError::Unsupported(
        "WebP-compressed COG tiles (Compression 50001) need the `decode-webp` cargo feature"
            .to_owned(),
    ))
}

/// Converts a decoded image into the level's band layout.
///
/// The decoder has already resolved the colour model (a YCbCr JPEG arrives as
/// RGB), so the reported layout says so rather than repeating the file's
/// `PhotometricInterpretation`. Greyscale levels keep theirs, because
/// `WhiteIsZero` still has to invert the ramp.
#[cfg(any(feature = "decode", feature = "decode-webp"))]
fn image_to_samples(
    level: &CogLevel,
    decoded: &image::DynamicImage,
    colour_photometric: u16,
) -> Result<BlockSamples, RenderError> {
    if decoded.width() != level.tile_width {
        return Err(RenderError::Decode(format!(
            "COG block decoded to {} pixels wide, but the level declares {}",
            decoded.width(),
            level.tile_width
        )));
    }
    let (samples, bands, photometric) = match level.samples_per_pixel {
        1 => (
            decoded.to_luma8().into_raw(),
            1,
            greyscale_photometric(level.photometric),
        ),
        2 => (
            decoded.to_luma_alpha8().into_raw(),
            2,
            greyscale_photometric(level.photometric),
        ),
        3 => (decoded.to_rgb8().into_raw(), 3, colour_photometric),
        4 => (decoded.to_rgba8().into_raw(), 4, colour_photometric),
        other => {
            return Err(RenderError::Unsupported(format!(
                "an image-codec COG block with {other} bands is not supported (supported: 1-4)"
            )));
        }
    };
    Ok(BlockSamples::Native(
        samples,
        SampleLayout {
            bits_per_sample: 8,
            samples_per_pixel: bands,
            photometric,
            signed: false,
        },
    ))
}

/// Keeps a greyscale level's own ramp direction, defaulting to `BlackIsZero`.
#[cfg(any(feature = "decode", feature = "decode-webp"))]
const fn greyscale_photometric(declared: u16) -> u16 {
    if declared == PHOTOMETRIC_WHITE_IS_ZERO {
        PHOTOMETRIC_WHITE_IS_ZERO
    } else {
        PHOTOMETRIC_BLACK_IS_ZERO
    }
}

/// Undoes TIFF horizontal differencing over `rows` rows of a block, in place.
///
/// Rows are independent, and within a row each sample references the sample
/// `samples_per_pixel` positions earlier so interleaved bands are reconstructed
/// separately. Wrapping arithmetic mirrors the encoder's wrapping subtraction,
/// making the transform exactly invertible.
///
/// Ported from `oxigeo-wasm`'s `apply_horizontal_predictor`.
fn undo_horizontal_predictor(data: &mut [u8], level: &CogLevel, rows: u32, little_endian: bool) {
    let width = level.tile_width as usize;
    let height = rows as usize;
    let spp = usize::from(level.samples_per_pixel.max(1));

    match level.bits_per_sample {
        8 => {
            let row_bytes = width * spp;
            for row in 0..height {
                let Some(slice) = data.get_mut(row * row_bytes..(row + 1) * row_bytes) else {
                    break;
                };
                for index in spp..slice.len() {
                    let previous = slice[index - spp];
                    slice[index] = slice[index].wrapping_add(previous);
                }
            }
        }
        16 => {
            let row_bytes = width * spp * 2;
            for row in 0..height {
                let Some(slice) = data.get_mut(row * row_bytes..(row + 1) * row_bytes) else {
                    break;
                };
                let mut samples: Vec<u16> = slice
                    .chunks_exact(2)
                    .map(|pair| {
                        let raw = [pair[0], pair[1]];
                        if little_endian {
                            u16::from_le_bytes(raw)
                        } else {
                            u16::from_be_bytes(raw)
                        }
                    })
                    .collect();
                for index in spp..samples.len() {
                    let previous = samples[index - spp];
                    samples[index] = samples[index].wrapping_add(previous);
                }
                for (index, value) in samples.iter().enumerate() {
                    let bytes = if little_endian {
                        value.to_le_bytes()
                    } else {
                        value.to_be_bytes()
                    };
                    let Some(pair) = slice.get_mut(index * 2..index * 2 + 2) else {
                        break;
                    };
                    pair.copy_from_slice(&bytes);
                }
            }
        }
        // Other widths have no predictor support; leaving the data untouched
        // is what `oxigeo-wasm` does, and `samples_to_rgba` rejects them anyway.
        _ => {}
    }
}

/// Rewrites 16/32/64-bit samples from the file's byte order into the host's.
///
/// 8-bit samples have nothing to swap, and an exotic width (sub-byte packing,
/// 24-bit) has no defined sample boundary to swap across, so both pass through.
///
/// Ported from `oxigeo-wasm`'s `normalize_samples_to_native`.
fn normalize_to_native(data: &mut [u8], bits_per_sample: u16, little_endian: bool) {
    if little_endian == cfg!(target_endian = "little") {
        return;
    }
    let width = match bits_per_sample {
        16 => 2usize,
        32 => 4,
        64 => 8,
        _ => return,
    };
    for sample in data.chunks_exact_mut(width) {
        sample.reverse();
    }
}

/// Reads the `index`-th sample of a host-native buffer as an `f64`.
///
/// `SampleFormat` 2 (two's-complement signed) is read as such: a DEM whose
/// nodata fill is `-9999` stores `0xD8F1`, and reading that as 55 537 would put
/// the fill at the top of the display range instead of matching the file's
/// declared nodata value.
///
/// Inlined because the stretch pass calls it once per colour sample; the bounds
/// check is what keeps a short buffer from panicking, so it stays.
#[inline]
fn sample_at(data: &[u8], index: usize, layout: SampleLayout) -> Option<f64> {
    match layout.bits_per_sample {
        8 => data.get(index).map(|value| {
            if layout.signed {
                f64::from(*value as i8)
            } else {
                f64::from(*value)
            }
        }),
        16 => {
            let raw: [u8; 2] = data.get(index * 2..index * 2 + 2)?.try_into().ok()?;
            let native = u16::from_ne_bytes(raw);
            Some(if layout.signed {
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "a two's-complement reinterpretation is the point"
                )]
                let signed = native as i16;
                f64::from(signed)
            } else {
                f64::from(native)
            })
        }
        _ => None,
    }
}

/// Maps host-native samples onto RGBA8, filling `rows` rows and leaving the
/// rest of the block transparent.
///
/// # Band layout
///
/// A non-empty [`CogLevel::color_map`] overrides this table: the first band is a
/// palette index — see [`palette_to_rgba`].
///
/// | `SamplesPerPixel` | Interpretation |
/// |---|---|
/// | 1 | greyscale, opaque |
/// | 2 | greyscale + alpha |
/// | 3 | RGB, opaque |
/// | 4 | RGBA |
/// | >4 | first three bands as RGB, opaque (multispectral) |
///
/// `PhotometricInterpretation` 0 (`WhiteIsZero`) inverts the greyscale ramp.
///
/// # Errors
///
/// Returns [`RenderError::Unsupported`] for floating-point samples, bit depths
/// other than 8 and 16, a colour model this reader would otherwise mis-draw
/// (CMYK, CIELab), and a palette declared without a usable `ColorMap`;
/// [`RenderError::Decode`] if the buffer is shorter than `rows` rows require.
fn samples_to_rgba(
    level: &CogLevel,
    layout: SampleLayout,
    data: &[u8],
    rows: u32,
    options: &CogDecodeOptions,
) -> Result<Vec<u8>, RenderError> {
    let palette = level.has_color_map();
    if layout.photometric == PHOTOMETRIC_PALETTE && !palette {
        return Err(RenderError::Unsupported(
            "COG declares palette colour (PhotometricInterpretation 3) but carries no usable \
             ColorMap (tag 320)"
                .to_owned(),
        ));
    }
    if level.sample_format == SAMPLE_FORMAT_FLOAT {
        return Err(RenderError::Unsupported(
            "floating-point COG samples (SampleFormat 3) are not supported".to_owned(),
        ));
    }
    if !matches!(layout.bits_per_sample, 8 | 16) {
        return Err(RenderError::Unsupported(format!(
            "COG BitsPerSample {} is not supported (supported: 8, 16)",
            layout.bits_per_sample
        )));
    }
    // Dispatching on band count alone would draw a CMYK image's cyan, magenta
    // and yellow as red, green and blue and take its black as alpha, with no
    // diagnostic at all. Refusing beats mis-drawing, the same way an
    // unsupported CRS is refused rather than placed wrongly.
    if !matches!(
        layout.photometric,
        PHOTOMETRIC_WHITE_IS_ZERO
            | PHOTOMETRIC_BLACK_IS_ZERO
            | PHOTOMETRIC_RGB
            | PHOTOMETRIC_PALETTE
    ) {
        let hint = if layout.photometric == PHOTOMETRIC_YCBCR {
            " (YCbCr is converted by the JPEG decoder, so it is only reachable here on an \
             uncompressed or DEFLATE-compressed YCbCr file)"
        } else {
            ""
        };
        return Err(RenderError::Unsupported(format!(
            "COG PhotometricInterpretation {} is not supported{hint} (supported: 0 WhiteIsZero, \
             1 BlackIsZero, 2 RGB, 3 Palette)",
            layout.photometric
        )));
    }
    if layout.samples_per_pixel == 0 {
        return Err(RenderError::Unsupported(
            "COG declares 0 samples per pixel, which describes no pixels".to_owned(),
        ));
    }

    let geometry = BlockGeometry::new(level, layout, rows)?;
    let available_rows = data.len() / geometry.row_bytes;
    if available_rows < geometry.rows {
        return Err(RenderError::Decode(format!(
            "COG block holds {available_rows} rows of samples but its geometry needs {}",
            geometry.rows
        )));
    }

    if palette {
        return palette_to_rgba(level, layout, data, &geometry, options.nodata);
    }

    let (low, span) = stretch_bounds(layout, data, &geometry, options);
    let invert = layout.photometric == PHOTOMETRIC_WHITE_IS_ZERO && geometry.colour_bands == 1;
    let nodata = usable_nodata(options.nodata);

    let mut rgba = vec![0u8; geometry.rgba_bytes];
    for pixel in 0..geometry.filled_pixels {
        let base = pixel * geometry.spp;
        let Some(slot) = rgba.get_mut(pixel * 4..pixel * 4 + 4) else {
            break;
        };
        if is_nodata(data, base, layout, nodata) {
            continue;
        }
        let mut channels = [0u8; 3];
        for (band, channel) in channels
            .iter_mut()
            .enumerate()
            .take(geometry.colour_bands.min(3))
        {
            let raw = sample_at(data, base + band, layout).unwrap_or(0.0);
            let scaled = if layout.bits_per_sample == 8 && !layout.signed {
                raw
            } else {
                (raw - low) / span * 255.0
            };
            let clamped = scaled.clamp(0.0, 255.0);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to 0..=255 immediately above"
            )]
            let byte = clamped as u8;
            *channel = if invert { 255 - byte } else { byte };
        }
        let (red, green, blue) = if geometry.colour_bands == 1 {
            (channels[0], channels[0], channels[0])
        } else {
            (channels[0], channels[1], channels[2])
        };
        let alpha = match geometry.spp {
            2 | 4 => {
                let raw = sample_at(data, base + geometry.spp - 1, layout)
                    .unwrap_or(255.0)
                    .max(0.0);
                if layout.bits_per_sample == 8 {
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "an 8-bit sample is already 0..=255"
                    )]
                    let byte = raw as u8;
                    byte
                } else {
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "a 16-bit sample divided by 257 is 0..=255"
                    )]
                    let byte = (raw / 257.0) as u8;
                    byte
                }
            }
            _ => 255,
        };
        slot.copy_from_slice(&[red, green, blue, alpha]);
    }
    Ok(rgba)
}

/// The pixel counts and strides one block decode works in.
struct BlockGeometry {
    /// Bands per pixel of the sample buffer.
    spp: usize,
    /// Bands that carry colour, i.e. every band but a 2- or 4-band image's
    /// alpha, so the stretch never rescales opacity.
    colour_bands: usize,
    /// Bytes one row of the block occupies.
    row_bytes: usize,
    /// Rows of real image data this block carries.
    rows: usize,
    /// Pixels the decode fills; the rest of the block stays transparent.
    filled_pixels: usize,
    /// Bytes of the full RGBA8 block.
    rgba_bytes: usize,
}

impl BlockGeometry {
    /// Resolves a level's block geometry, rejecting one that has no pixels or
    /// overflows an address.
    fn new(level: &CogLevel, layout: SampleLayout, rows: u32) -> Result<Self, RenderError> {
        let overflow = || RenderError::Decode("COG block geometry overflows an address".to_owned());
        let spp = usize::from(layout.samples_per_pixel);
        let width = level.tile_width as usize;
        let height = level.tile_height as usize;
        if width == 0 || height == 0 {
            return Err(RenderError::Decode(
                "COG block geometry has a zero dimension".to_owned(),
            ));
        }
        let bytes_per_sample = usize::from(layout.bits_per_sample / 8).max(1);
        let row_bytes = width
            .checked_mul(spp)
            .and_then(|samples| samples.checked_mul(bytes_per_sample))
            .ok_or_else(overflow)?;
        let rows = (rows as usize).min(height);
        let rgba_bytes = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(overflow)?;
        let filled_pixels = width.checked_mul(rows).ok_or_else(overflow)?;
        let colour_bands = match spp {
            2 => 1,
            4 => 3,
            other => other.min(3),
        };
        Ok(Self {
            spp,
            colour_bands,
            row_bytes,
            rows,
            filled_pixels,
            rgba_bytes,
        })
    }
}

/// A nodata value only matches samples if it is finite.
const fn usable_nodata(nodata: Option<f64>) -> Option<f64> {
    match nodata {
        Some(value) if value.is_finite() => Some(value),
        _ => None,
    }
}

/// Whether the pixel whose first band is at `base` is nodata.
fn is_nodata(data: &[u8], base: usize, layout: SampleLayout, nodata: Option<f64>) -> bool {
    let Some(nodata) = nodata else {
        return false;
    };
    sample_at(data, base, layout).is_some_and(|value| value == nodata)
}

/// Maps palette-indexed samples through the level's `ColorMap` to RGBA8.
///
/// A sample that falls outside the palette becomes transparent rather than
/// black, which is how an out-of-range class in a land-cover raster should read.
/// Only the first band is an index; further bands (rare) are ignored.
///
/// Both palette widths TIFF defines are handled: 8-bit indices into 256 entries,
/// and 16-bit indices into up to 65 536. The entry count comes from the map's
/// own length, so a partially written palette still resolves the classes it
/// does define.
fn palette_to_rgba(
    level: &CogLevel,
    layout: SampleLayout,
    data: &[u8],
    geometry: &BlockGeometry,
    nodata: Option<f64>,
) -> Result<Vec<u8>, RenderError> {
    let nodata = usable_nodata(nodata);
    let mut rgba = vec![0u8; geometry.rgba_bytes];
    for pixel in 0..geometry.filled_pixels {
        let Some(slot) = rgba.get_mut(pixel * 4..pixel * 4 + 4) else {
            break;
        };
        let base = pixel * geometry.spp;
        if is_nodata(data, base, layout, nodata) {
            continue;
        }
        let raw = sample_at(data, base, layout).unwrap_or(0.0);
        let index = if raw >= 0.0 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a non-negative 8- or 16-bit sample fits usize exactly"
            )]
            let index = raw as usize;
            index
        } else {
            usize::MAX
        };
        if let Some([red, green, blue]) = level.palette_rgb(index) {
            slot.copy_from_slice(&[red, green, blue, 255]);
        }
    }
    Ok(rgba)
}

/// Resolves a [`RasterStretch`] into `(low, span)`, guaranteeing `span != 0`.
///
/// Only the rows the block actually carries are scanned: the transparent
/// padding of a short strip, and the rows a tile is padded with past the image
/// height, would otherwise pull the range towards zero. Nodata pixels are
/// excluded for the same reason — one `-9999` fill pixel in a DEM tile is
/// enough to wash the terrain out to a narrow band of greys.
fn stretch_bounds(
    layout: SampleLayout,
    data: &[u8],
    geometry: &BlockGeometry,
    options: &CogDecodeOptions,
) -> (f64, f64) {
    if layout.bits_per_sample == 8 && !layout.signed {
        return (0.0, 255.0);
    }
    let (low, high) = match options.stretch {
        RasterStretch::Fixed { min, max } => (min, max),
        RasterStretch::MinMaxPerTile => measured_range(layout, data, geometry, options.nodata)
            .unwrap_or((0.0, f64::from(u16::MAX))),
        RasterStretch::Percentile { low_pct, high_pct } => {
            percentile_range(layout, data, geometry, options, low_pct, high_pct)
                .or_else(|| measured_range(layout, data, geometry, options.nodata))
                .unwrap_or((0.0, f64::from(u16::MAX)))
        }
    };
    let span = high - low;
    if span.is_finite() && span > 0.0 {
        (low, span)
    } else {
        (low, 1.0)
    }
}

/// Visits every colour sample of the block's real rows that is not nodata.
fn for_each_colour_sample(
    layout: SampleLayout,
    data: &[u8],
    geometry: &BlockGeometry,
    nodata: Option<f64>,
    mut visit: impl FnMut(f64),
) {
    let nodata = usable_nodata(nodata);
    for pixel in 0..geometry.filled_pixels {
        let base = pixel * geometry.spp;
        if is_nodata(data, base, layout, nodata) {
            continue;
        }
        for band in 0..geometry.colour_bands {
            if let Some(value) = sample_at(data, base + band, layout) {
                visit(value);
            }
        }
    }
}

/// The block's own minimum and maximum colour sample, or `None` when it has
/// none to measure.
fn measured_range(
    layout: SampleLayout,
    data: &[u8],
    geometry: &BlockGeometry,
    nodata: Option<f64>,
) -> Option<(f64, f64)> {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for_each_colour_sample(layout, data, geometry, nodata, |value| {
        low = low.min(value);
        high = high.max(value);
    });
    (low.is_finite() && high.is_finite()).then_some((low, high))
}

/// The sample values at two percentiles of the block's histogram.
///
/// `None` for percentages outside `0..=100`, inverted percentages, or a block
/// with nothing to measure, so the caller can fall back to the full range.
fn percentile_range(
    layout: SampleLayout,
    data: &[u8],
    geometry: &BlockGeometry,
    options: &CogDecodeOptions,
    low_pct: f64,
    high_pct: f64,
) -> Option<(f64, f64)> {
    if !(low_pct.is_finite() && high_pct.is_finite())
        || low_pct < 0.0
        || high_pct > 100.0
        || low_pct >= high_pct
    {
        return None;
    }
    // The histogram is indexed by sample value, so it needs the *unsigned*
    // reinterpretation of a signed sample; the offset is undone on the way out.
    let offset = if layout.signed {
        f64::from(i16::MIN)
    } else {
        0.0
    };
    let mut histogram = vec![0u32; PERCENTILE_BINS];
    let mut total = 0u64;
    for_each_colour_sample(layout, data, geometry, options.nodata, |value| {
        let shifted = value - offset;
        if (0.0..PERCENTILE_BINS as f64).contains(&shifted) {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "bounded to 0..PERCENTILE_BINS immediately above"
            )]
            let bin = shifted as usize;
            if let Some(slot) = histogram.get_mut(bin) {
                *slot = slot.saturating_add(1);
                total += 1;
            }
        }
    });
    if total == 0 {
        return None;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "a block holds far fewer samples than 2^53"
    )]
    let count = total as f64;
    let low_target = count * low_pct / 100.0;
    let high_target = count * high_pct / 100.0;
    let mut cumulative = 0.0f64;
    let mut low_value = None;
    let mut high_value = None;
    let mut last_value = None;
    for (bin, hits) in histogram.iter().enumerate() {
        if *hits == 0 {
            continue;
        }
        cumulative += f64::from(*hits);
        #[expect(clippy::cast_precision_loss, reason = "a bin index is at most 65 535")]
        let value = bin as f64 + offset;
        last_value = Some(value);
        if low_value.is_none() && cumulative >= low_target {
            low_value = Some(value);
        }
        if high_value.is_none() && cumulative >= high_target {
            high_value = Some(value);
        }
    }
    // `high_pct <= 100` means the target is always reached, but a histogram
    // that somehow ends short falls back to its largest sample rather than to
    // its smallest.
    let high_value = high_value.or(last_value);
    match (low_value, high_value) {
        (Some(low), Some(high)) if high > low => Some((low, high)),
        _ => None,
    }
}

/// Decodes one COG block payload all the way to RGBA8.
///
/// The result is always `tile_width x tile_height` pixels, four bytes each, in
/// raster order, so the compositor's length check holds for every block. A
/// block that carries fewer rows than the level's tile height — the last strip
/// of a striped TIFF, whose payload is *not* padded — fills those rows and
/// leaves the rest fully transparent.
///
/// `tile_y` is the block's row in the level's tile grid, and `little_endian` is
/// the *file's* byte order, from its `II`/`MM` mark.
///
/// # Errors
///
/// Propagates [`decompress_tile`] and the sample-mapping errors documented on
/// `samples_to_rgba`, and returns [`RenderError::Unsupported`] for the
/// floating-point predictor (tag 317 = 3).
pub fn decode_cog_block(
    level: &CogLevel,
    tile_y: u32,
    little_endian: bool,
    payload: &[u8],
    options: &CogDecodeOptions,
) -> Result<Vec<u8>, RenderError> {
    if level.predictor == PREDICTOR_FLOATING_POINT {
        return Err(RenderError::Unsupported(
            "the floating-point predictor (tag 317 = 3) is not supported".to_owned(),
        ));
    }
    let rows = level.block_rows(tile_y);
    match decompress_block(level, payload, &options.jpeg_tables)? {
        BlockSamples::Raw(mut samples) => {
            if level.predictor == PREDICTOR_HORIZONTAL {
                undo_horizontal_predictor(&mut samples, level, rows, little_endian);
            }
            normalize_to_native(&mut samples, level.bits_per_sample, little_endian);
            samples_to_rgba(level, SampleLayout::of(level), &samples, rows, options)
        }
        BlockSamples::Native(samples, layout) => {
            samples_to_rgba(level, layout, &samples, rows, options)
        }
    }
}

/// Decodes one full COG tile payload all the way to RGBA8.
///
/// The whole-block form of [`decode_cog_block`]: it decodes the level's first
/// tile row, where every block is full. Prefer [`decode_cog_block`] when the
/// tile's row is known — a striped TIFF's last strip is short, and decoding it
/// as a full block fails the whole map tile.
///
/// # Errors
///
/// See [`decode_cog_block`].
pub fn decode_cog_tile(
    level: &CogLevel,
    little_endian: bool,
    payload: &[u8],
    stretch: RasterStretch,
) -> Result<Vec<u8>, RenderError> {
    decode_cog_block(
        level,
        0,
        little_endian,
        payload,
        &CogDecodeOptions::with_stretch(stretch),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        CogDecodeOptions, RasterStretch, SampleLayout, decode_cog_block, decode_cog_tile,
        decompress_tile, normalize_to_native, samples_to_rgba, splice_jpeg_tables,
        undo_horizontal_predictor, unpack_bits,
    };
    use crate::cog::meta::{
        COMPRESSION_DEFLATE, COMPRESSION_DEFLATE_OLD, COMPRESSION_LERC, COMPRESSION_LZW,
        COMPRESSION_NONE, COMPRESSION_PACKBITS, COMPRESSION_ZSTD, CogLevel,
        MAX_TILE_DECOMPRESSED_BYTES,
    };
    use crate::error::RenderError;

    fn gray_level(edge: u32, bits: u16, compression: u16) -> CogLevel {
        CogLevel {
            width: edge,
            height: edge,
            tile_width: edge,
            tile_height: edge,
            bits_per_sample: bits,
            samples_per_pixel: 1,
            sample_format: 1,
            compression,
            predictor: 1,
            photometric: 1,
            color_map: Vec::new(),
            tile_offsets: vec![0],
            tile_byte_counts: vec![1],
        }
    }

    /// The default options: per-tile stretch, no nodata, no JPEG tables.
    fn options() -> CogDecodeOptions {
        CogDecodeOptions::default()
    }

    #[test]
    fn uncompressed_gray_becomes_opaque_rgba() {
        let level = gray_level(2, 8, COMPRESSION_NONE);
        let rgba = decode_cog_tile(&level, true, &[0, 64, 128, 255], RasterStretch::default())
            .expect("an uncompressed grey tile must decode");
        assert_eq!(rgba.len(), 4 * 4);
        assert_eq!(&rgba[..4], &[0, 0, 0, 255]);
        assert_eq!(&rgba[12..], &[255, 255, 255, 255]);
    }

    #[test]
    fn white_is_zero_inverts_the_ramp() {
        let mut level = gray_level(2, 8, COMPRESSION_NONE);
        level.photometric = 0;
        let rgba = decode_cog_tile(&level, true, &[0, 0, 0, 0], RasterStretch::default())
            .expect("a WhiteIsZero tile must decode");
        assert_eq!(&rgba[..4], &[255, 255, 255, 255]);
    }

    #[test]
    fn rgb_and_rgba_band_layouts_are_honoured() {
        let mut level = gray_level(1, 8, COMPRESSION_NONE);
        level.samples_per_pixel = 3;
        level.photometric = 2;
        let rgba = decode_cog_tile(&level, true, &[10, 20, 30], RasterStretch::default())
            .expect("an RGB tile must decode");
        assert_eq!(rgba, vec![10, 20, 30, 255]);

        level.samples_per_pixel = 4;
        let rgba = decode_cog_tile(&level, true, &[10, 20, 30, 40], RasterStretch::default())
            .expect("an RGBA tile must decode");
        assert_eq!(rgba, vec![10, 20, 30, 40]);

        // Grey + alpha.
        level.samples_per_pixel = 2;
        level.photometric = 1;
        let rgba = decode_cog_tile(&level, true, &[90, 40], RasterStretch::default())
            .expect("a grey+alpha tile must decode");
        assert_eq!(rgba, vec![90, 90, 90, 40]);

        // Multispectral: the first three bands are taken as RGB.
        level.samples_per_pixel = 5;
        level.photometric = 2;
        let rgba = decode_cog_tile(&level, true, &[1, 2, 3, 4, 5], RasterStretch::default())
            .expect("a multispectral tile must decode");
        assert_eq!(rgba, vec![1, 2, 3, 255]);
    }

    #[test]
    fn sixteen_bit_min_max_stretch_spans_the_display_range() {
        let level = gray_level(2, 16, COMPRESSION_NONE);
        let mut payload = Vec::new();
        for value in [1000u16, 2000, 3000, 5000] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let rgba = decode_cog_tile(&level, true, &payload, RasterStretch::MinMaxPerTile)
            .expect("a 16-bit tile must decode");
        assert_eq!(rgba[0], 0, "the tile minimum maps to black");
        assert_eq!(rgba[12], 255, "the tile maximum maps to white");
    }

    #[test]
    fn a_fixed_stretch_clamps_outside_its_range() {
        let level = gray_level(2, 16, COMPRESSION_NONE);
        let mut payload = Vec::new();
        for value in [0u16, 500, 1000, 4000] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let rgba = decode_cog_tile(
            &level,
            true,
            &payload,
            RasterStretch::Fixed {
                min: 0.0,
                max: 1000.0,
            },
        )
        .expect("a 16-bit tile must decode");
        assert_eq!(rgba[0], 0);
        assert_eq!(rgba[8], 255);
        assert_eq!(rgba[12], 255, "4000 is clamped, not wrapped");
    }

    #[test]
    fn a_percentile_stretch_ignores_a_single_outlier() {
        // 63 samples at 100..162 plus one at 60 000: a min/max stretch squeezes
        // the real data into the bottom 1 % of the ramp, a 2-98 % clip does not.
        let level = gray_level(8, 16, COMPRESSION_NONE);
        let mut payload = Vec::new();
        for index in 0..63u16 {
            payload.extend_from_slice(&(100 + index).to_le_bytes());
        }
        payload.extend_from_slice(&60_000u16.to_le_bytes());

        let min_max = decode_cog_tile(&level, true, &payload, RasterStretch::MinMaxPerTile)
            .expect("a 16-bit tile must decode");
        let percentile = decode_cog_block(
            &level,
            0,
            true,
            &payload,
            &CogDecodeOptions {
                stretch: RasterStretch::Percentile {
                    low_pct: 2.0,
                    high_pct: 98.0,
                },
                ..CogDecodeOptions::default()
            },
        )
        .expect("a 16-bit tile must decode");
        assert!(
            min_max[4 * 62] < 8,
            "min/max leaves the real data almost black"
        );
        assert!(
            percentile[4 * 62] > 200,
            "a 2-98 % clip spreads the real data across the ramp"
        );
        // The outlier is clamped, not wrapped.
        assert_eq!(percentile[4 * 63], 255);
    }

    #[test]
    fn nonsensical_percentiles_fall_back_to_the_full_range() {
        let level = gray_level(2, 16, COMPRESSION_NONE);
        let mut payload = Vec::new();
        for value in [1000u16, 2000, 3000, 5000] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        for stretch in [
            RasterStretch::Percentile {
                low_pct: 98.0,
                high_pct: 2.0,
            },
            RasterStretch::Percentile {
                low_pct: -1.0,
                high_pct: 50.0,
            },
            RasterStretch::Percentile {
                low_pct: f64::NAN,
                high_pct: 98.0,
            },
        ] {
            let rgba = decode_cog_block(
                &level,
                0,
                true,
                &payload,
                &CogDecodeOptions {
                    stretch,
                    ..CogDecodeOptions::default()
                },
            )
            .expect("an unusable percentile must not fail the decode");
            assert_eq!(rgba[0], 0);
            assert_eq!(rgba[12], 255);
        }
    }

    #[test]
    fn a_flat_tile_does_not_divide_by_zero() {
        let level = gray_level(2, 16, COMPRESSION_NONE);
        let payload = vec![0u8; 8];
        let rgba = decode_cog_tile(&level, true, &payload, RasterStretch::MinMaxPerTile)
            .expect("a flat tile must decode");
        assert!(rgba.iter().step_by(4).all(|value| *value == 0));
    }

    #[test]
    fn big_endian_sixteen_bit_samples_are_normalised() {
        let level = gray_level(1, 16, COMPRESSION_NONE);
        let payload = 300u16.to_be_bytes().to_vec();
        let rgba = decode_cog_tile(
            &level,
            false,
            &payload,
            RasterStretch::Fixed {
                min: 0.0,
                max: 300.0,
            },
        )
        .expect("an MM tile must decode");
        assert_eq!(rgba[0], 255, "300 read as big-endian is the maximum");
    }

    #[test]
    fn normalisation_is_a_no_op_for_the_host_order() {
        let mut data = vec![1, 2, 3, 4];
        normalize_to_native(&mut data, 16, cfg!(target_endian = "little"));
        assert_eq!(data, vec![1, 2, 3, 4]);
        normalize_to_native(&mut data, 16, !cfg!(target_endian = "little"));
        assert_eq!(data, vec![2, 1, 4, 3]);
        // An unsupported width passes through.
        let mut odd = vec![1, 2, 3];
        normalize_to_native(&mut odd, 24, !cfg!(target_endian = "little"));
        assert_eq!(odd, vec![1, 2, 3]);
    }

    #[test]
    fn the_horizontal_predictor_round_trips_per_row() {
        // 3x2 tile, RGB, 8-bit: rows must be independent.
        let mut level = gray_level(3, 8, COMPRESSION_NONE);
        level.tile_width = 3;
        level.tile_height = 2;
        level.samples_per_pixel = 3;
        let original: Vec<u8> = (0u8..18).map(|value| value * 7).collect();
        let mut diffed = original.clone();
        for row in 0..2usize {
            for index in (3..9).rev() {
                let base = row * 9;
                diffed[base + index] = diffed[base + index].wrapping_sub(diffed[base + index - 3]);
            }
        }
        assert_ne!(diffed, original);
        undo_horizontal_predictor(&mut diffed, &level, 2, true);
        assert_eq!(diffed, original);
    }

    #[test]
    fn the_horizontal_predictor_round_trips_for_sixteen_bit() {
        let mut level = gray_level(4, 16, COMPRESSION_NONE);
        level.tile_width = 4;
        level.tile_height = 1;
        let original: Vec<u16> = vec![65_530, 5, 65_535, 10];
        let mut diffed = original.clone();
        for index in (1..4).rev() {
            diffed[index] = diffed[index].wrapping_sub(diffed[index - 1]);
        }
        let mut bytes: Vec<u8> = diffed
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        undo_horizontal_predictor(&mut bytes, &level, 1, true);
        let decoded: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        assert_eq!(decoded, original);
    }

    #[test]
    fn a_predicted_big_endian_sixteen_bit_tile_decodes() {
        // The module header calls out the predictor/byte-order ordering as
        // load-bearing: undoing the predictor *after* normalising a big-endian
        // buffer differences the wrong halves and silently corrupts the tile.
        // This is the only combination that can tell the two orderings apart.
        let mut level = gray_level(4, 16, COMPRESSION_NONE);
        level.tile_width = 4;
        level.tile_height = 1;
        level.height = 1;
        level.predictor = 2;
        let original: Vec<u16> = vec![1_000, 1_400, 1_200, 2_000];
        let mut diffed = original.clone();
        for index in (1..4).rev() {
            diffed[index] = diffed[index].wrapping_sub(diffed[index - 1]);
        }
        let payload: Vec<u8> = diffed
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect();

        let rgba = decode_cog_tile(
            &level,
            false,
            &payload,
            RasterStretch::Fixed {
                min: 0.0,
                max: 2_000.0,
            },
        )
        .expect("a predicted MM tile must decode");
        for (index, value) in original.iter().enumerate() {
            let expected = u8::try_from(u32::from(*value) * 255 / 2_000).unwrap_or(255);
            assert_eq!(
                rgba[index * 4],
                expected,
                "predicted MM sample {index} must round-trip"
            );
        }
    }

    #[test]
    fn signed_samples_are_read_as_twos_complement() {
        let mut level = gray_level(2, 16, COMPRESSION_NONE);
        level.sample_format = 2;
        let mut payload = Vec::new();
        for value in [-1000i16, 0, 500, 1000] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let rgba = decode_cog_tile(&level, true, &payload, RasterStretch::MinMaxPerTile)
            .expect("a signed tile must decode");
        assert_eq!(rgba[0], 0, "-1000 is the minimum");
        assert_eq!(rgba[12], 255, "1000 is the maximum");
    }

    #[test]
    fn the_floating_point_predictor_is_rejected() {
        let mut level = gray_level(1, 8, COMPRESSION_NONE);
        level.predictor = 3;
        assert!(matches!(
            decode_cog_tile(&level, true, &[0], RasterStretch::default()),
            Err(RenderError::Unsupported(_))
        ));
    }

    #[test]
    fn packbits_decodes_literals_repeats_and_no_ops() {
        // 2 literals, then 3 copies of 0xAA, then a no-op control byte.
        let payload = [1u8, 0x11, 0x22, 0xFE, 0xAA, 0x80];
        assert_eq!(unpack_bits(&payload, 8), vec![0x11, 0x22, 0xAA, 0xAA, 0xAA]);
        // A truncated literal run copies what is there and stops.
        assert_eq!(unpack_bits(&[3, 0x01, 0x02], 4), vec![0x01, 0x02]);
        // A repeat control byte with no value to repeat stops cleanly.
        assert_eq!(unpack_bits(&[0xFE], 4), Vec::<u8>::new());
        assert_eq!(unpack_bits(&[], 0), Vec::<u8>::new());
    }

    #[test]
    fn packbits_reserves_only_what_the_payload_can_produce() {
        // A one-byte payload cannot expand past 128 bytes whatever the block
        // geometry claims, so the reservation must not follow the claim.
        let out = unpack_bits(&[0xFE, 0xAA], 64 * 1024 * 1024);
        assert_eq!(out, vec![0xAA; 3]);
        assert!(
            out.capacity() <= 2 * super::PACKBITS_MAX_EXPANSION,
            "the reservation must be bounded by the payload, not by the header"
        );
    }

    #[test]
    fn packbits_round_trips_through_the_decoder() {
        let level = gray_level(2, 8, COMPRESSION_PACKBITS);
        // 4 pixels: literal 0x01 0x02, then two 0x03.
        let payload = [1u8, 0x01, 0x02, 0xFF, 0x03];
        let rgba = decode_cog_tile(&level, true, &payload, RasterStretch::default())
            .expect("a PackBits tile must decode");
        assert_eq!(rgba.len(), 16);
    }

    #[test]
    fn deflate_round_trips_under_both_tiff_codes() {
        let payload = oxiarc_deflate::zlib_compress(&[1u8, 2, 3, 4], 6)
            .expect("zlib compression must succeed in tests");
        for code in [COMPRESSION_DEFLATE, COMPRESSION_DEFLATE_OLD] {
            let level = gray_level(2, 8, code);
            let decoded = decompress_tile(&level, &payload).expect("DEFLATE must decode");
            assert_eq!(decoded, vec![1, 2, 3, 4]);
        }
    }

    #[test]
    fn a_deflate_bomb_is_bounded_by_the_block_geometry() {
        // 4 MiB of zeros compress to a few kilobytes; a 2x2 block must not
        // inflate them, because the output buffer is the declared block size.
        let bomb = oxiarc_deflate::zlib_compress(&vec![0u8; 4 * 1024 * 1024], 6)
            .expect("zlib compression must succeed in tests");
        assert!(
            bomb.len() < 64 * 1024,
            "the fixture must actually be a bomb"
        );
        let level = gray_level(2, 8, COMPRESSION_DEFLATE);
        assert!(matches!(
            decompress_tile(&level, &bomb),
            Err(RenderError::Decode(_))
        ));
    }

    #[test]
    fn a_block_geometry_past_the_cap_is_refused_before_anything_is_reserved() {
        let mut level = gray_level(2, 8, COMPRESSION_PACKBITS);
        level.tile_width = 65_536;
        level.tile_height = 65_536;
        assert!(
            level
                .tile_bytes()
                .is_some_and(|bytes| bytes > MAX_TILE_DECOMPRESSED_BYTES)
        );
        assert!(matches!(
            decompress_tile(&level, &[0x80]),
            Err(RenderError::Unsupported(_))
        ));
        // …and a geometry that overflows an address at all.
        level.tile_width = u32::MAX;
        level.tile_height = u32::MAX;
        level.bits_per_sample = 16;
        assert!(decompress_tile(&level, &[0x80]).is_err());
    }

    #[test]
    fn lzw_round_trips() {
        let raw = [7u8, 7, 7, 7, 9, 9, 1, 2];
        let payload = oxiarc_lzw::compress_tiff(&raw).expect("LZW compression must succeed");
        let mut level = gray_level(4, 8, COMPRESSION_LZW);
        level.tile_width = 4;
        level.tile_height = 2;
        let decoded = decompress_tile(&level, &payload).expect("LZW must decode");
        assert_eq!(decoded, raw.to_vec());
    }

    #[test]
    fn a_short_final_strip_fills_its_rows_and_pads_the_rest() {
        // A striped TIFF, 4 px wide, 10 rows, 4 rows per strip: the third strip
        // holds 2 rows, and its payload is *not* padded to a full block. The
        // decode must fill those two rows and leave the rest transparent rather
        // than failing — a failure here kills the whole 256x256 map tile.
        let mut level = gray_level(4, 8, COMPRESSION_NONE);
        level.width = 4;
        level.height = 10;
        level.tile_width = 4;
        level.tile_height = 4;
        assert_eq!(level.block_rows(0), 4);
        assert_eq!(level.block_rows(2), 2);

        let payload: Vec<u8> = (0u8..8).map(|index| index * 10).collect();
        let rgba = decode_cog_block(&level, 2, true, &payload, &options())
            .expect("a short final strip must decode");
        assert_eq!(rgba.len(), 4 * 4 * 4, "the block is still full size");
        assert_eq!(&rgba[..4], &[0, 0, 0, 255], "row 0 is real data");
        assert_eq!(&rgba[28..32], &[70, 70, 70, 255], "row 1 is real data");
        assert!(
            rgba[32..].iter().all(|byte| *byte == 0),
            "the rows the strip does not carry stay transparent"
        );

        // Decoding the same payload as a full block is still an error, so a
        // genuinely truncated tile is not silently half-drawn.
        assert!(matches!(
            decode_cog_block(&level, 0, true, &payload, &options()),
            Err(RenderError::Decode(_))
        ));
    }

    #[test]
    fn a_short_strip_stretch_ignores_the_padding_rows() {
        // The padded rows are zeros; letting them into the stretch would drag
        // the minimum to 0 and darken every real pixel.
        let mut level = gray_level(2, 16, COMPRESSION_NONE);
        level.width = 2;
        level.height = 3;
        level.tile_width = 2;
        level.tile_height = 2;
        let mut payload = Vec::new();
        for value in [1_000u16, 1_500] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let rgba = decode_cog_block(&level, 1, true, &payload, &options())
            .expect("a one-row final strip must decode");
        assert_eq!(rgba[0], 0, "1000 is the minimum of the rows that exist");
        assert_eq!(rgba[4], 255, "1500 is the maximum of the rows that exist");
        assert!(rgba[8..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn nodata_pixels_become_transparent_and_leave_the_stretch_alone() {
        let level = gray_level(2, 16, COMPRESSION_NONE);
        let mut payload = Vec::new();
        for value in [1_000u16, 1_500, 2_000, 60_000] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let rgba = decode_cog_block(
            &level,
            0,
            true,
            &payload,
            &CogDecodeOptions {
                nodata: Some(60_000.0),
                ..CogDecodeOptions::default()
            },
        )
        .expect("a tile with nodata must decode");
        assert_eq!(
            &rgba[12..16],
            &[0, 0, 0, 0],
            "the nodata pixel is invisible"
        );
        assert_eq!(rgba[0], 0, "1000 is the minimum of the real samples");
        assert_eq!(rgba[8], 255, "2000 is the maximum of the real samples");

        // A non-finite nodata never matches, so nothing is hidden.
        let rgba = decode_cog_block(
            &level,
            0,
            true,
            &payload,
            &CogDecodeOptions {
                nodata: Some(f64::NAN),
                ..CogDecodeOptions::default()
            },
        )
        .expect("a NaN nodata must not fail the decode");
        assert_eq!(rgba[15], 255);
    }

    #[test]
    fn unsupported_codecs_and_formats_are_reported() {
        for code in [COMPRESSION_ZSTD, COMPRESSION_LERC, 34_927] {
            let level = gray_level(2, 8, code);
            assert!(
                matches!(
                    decompress_tile(&level, &[0; 4]),
                    Err(RenderError::Unsupported(_))
                ),
                "Compression {code} must be refused by name"
            );
        }

        let mut level = gray_level(2, 8, COMPRESSION_NONE);
        level.photometric = 3;
        assert!(
            matches!(
                samples_to_rgba(&level, SampleLayout::of(&level), &[0; 4], 2, &options()),
                Err(RenderError::Unsupported(_))
            ),
            "palette photometric with no ColorMap must be rejected"
        );

        let mut level = gray_level(2, 32, COMPRESSION_NONE);
        level.sample_format = 3;
        assert!(matches!(
            samples_to_rgba(&level, SampleLayout::of(&level), &[0; 16], 2, &options()),
            Err(RenderError::Unsupported(_))
        ));
        level.sample_format = 1;
        assert!(matches!(
            samples_to_rgba(&level, SampleLayout::of(&level), &[0; 16], 2, &options()),
            Err(RenderError::Unsupported(_))
        ));

        let mut zero_bands = gray_level(2, 8, COMPRESSION_NONE);
        zero_bands.samples_per_pixel = 0;
        assert!(matches!(
            samples_to_rgba(
                &zero_bands,
                SampleLayout::of(&zero_bands),
                &[0; 4],
                2,
                &options()
            ),
            Err(RenderError::Unsupported(_))
        ));
    }

    #[test]
    fn colour_models_this_reader_would_mis_draw_are_refused() {
        // CMYK's four bands would otherwise be drawn as RGB plus K-as-alpha,
        // with no diagnostic at all.
        for photometric in [5u16, 6, 8, 10] {
            let mut level = gray_level(1, 8, COMPRESSION_NONE);
            level.samples_per_pixel = 4;
            level.photometric = photometric;
            let error = decode_cog_tile(&level, true, &[10, 20, 30, 40], RasterStretch::default());
            assert!(
                matches!(error, Err(RenderError::Unsupported(_))),
                "PhotometricInterpretation {photometric} must be refused"
            );
        }
    }

    #[test]
    fn a_palette_tile_is_mapped_through_its_color_map() {
        let mut level = gray_level(2, 8, COMPRESSION_NONE);
        level.photometric = 3;
        // Three ramps of 256 entries: reds, greens, blues, scaled to 0..=65535.
        let mut color_map = vec![0u16; 3 * 256];
        color_map[1] = 65_535; // index 1 -> red
        color_map[256 + 2] = 65_535; // index 2 -> green
        color_map[512 + 3] = 65_535; // index 3 -> blue
        level.color_map = color_map;
        assert!(level.has_color_map());
        let rgba = decode_cog_tile(&level, true, &[1, 2, 3, 200], RasterStretch::default())
            .expect("a palette tile must decode");
        assert_eq!(&rgba[0..4], &[255, 0, 0, 255]);
        assert_eq!(&rgba[4..8], &[0, 255, 0, 255]);
        assert_eq!(&rgba[8..12], &[0, 0, 255, 255]);
        // Index 200 maps to an all-zero palette entry, which is opaque black.
        assert_eq!(&rgba[12..16], &[0, 0, 0, 255]);

        // An index past the end of a short palette is transparent, not black.
        level.color_map = vec![0u16; 3 * 4];
        let rgba = decode_cog_tile(&level, true, &[0, 1, 2, 9], RasterStretch::default())
            .expect("a palette tile must decode");
        assert_eq!(&rgba[12..16], &[0, 0, 0, 0]);
    }

    #[test]
    fn a_sixteen_bit_palette_resolves_its_wide_indices() {
        // 3 x 65 536 entries is a legal ColorMap, and the parser now reads it;
        // rejecting it at the codec would make the reader's own truncation look
        // like a missing tag.
        let mut level = gray_level(2, 16, COMPRESSION_NONE);
        level.photometric = 3;
        let entries = 4_096usize;
        let mut color_map = vec![0u16; 3 * entries];
        color_map[4_000] = 65_535;
        color_map[entries + 4_001] = 65_535;
        level.color_map = color_map;

        let mut payload = Vec::new();
        for index in [4_000u16, 4_001, 0, 60_000] {
            payload.extend_from_slice(&index.to_le_bytes());
        }
        let rgba = decode_cog_tile(&level, true, &payload, RasterStretch::default())
            .expect("a 16-bit palette tile must decode");
        assert_eq!(&rgba[0..4], &[255, 0, 0, 255]);
        assert_eq!(&rgba[4..8], &[0, 255, 0, 255]);
        assert_eq!(&rgba[8..12], &[0, 0, 0, 255]);
        assert_eq!(
            &rgba[12..16],
            &[0, 0, 0, 0],
            "an index past the palette is transparent"
        );
    }

    #[test]
    fn a_short_tile_is_a_decode_error_not_a_panic() {
        let level = gray_level(4, 8, COMPRESSION_NONE);
        assert!(matches!(
            decode_cog_tile(&level, true, &[0; 3], RasterStretch::default()),
            Err(RenderError::Decode(_))
        ));
    }

    #[test]
    fn a_zero_width_block_is_a_decode_error_not_a_division_by_zero() {
        let mut level = gray_level(4, 8, COMPRESSION_NONE);
        level.tile_width = 0;
        assert!(matches!(
            decode_cog_tile(&level, true, &[0; 4], RasterStretch::default()),
            Err(RenderError::Decode(_))
        ));
        let mut level = gray_level(4, 8, COMPRESSION_NONE);
        level.tile_height = 0;
        assert!(decode_cog_tile(&level, true, &[0; 4], RasterStretch::default()).is_err());
    }

    #[test]
    fn jpeg_tables_are_spliced_only_into_a_well_formed_pair() {
        let tables = [0xFFu8, 0xD8, 0xFF, 0xDB, 0x00, 0x02, 0xFF, 0xD9];
        let tile = [0xFFu8, 0xD8, 0xFF, 0xDA, 0x01];
        assert_eq!(
            splice_jpeg_tables(&tables, &tile),
            vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x02, 0xFF, 0xDA, 0x01]
        );
        // No tables, or a tile that is not a JPEG stream: pass the tile through
        // rather than building a malformed one.
        assert_eq!(splice_jpeg_tables(&[], &tile), tile.to_vec());
        assert_eq!(splice_jpeg_tables(&tables, &[1, 2, 3]), vec![1, 2, 3]);
        let not_a_stream = [0xFFu8, 0xD8, 0xFF, 0xDB, 0x00, 0x02];
        assert_eq!(splice_jpeg_tables(&not_a_stream, &tile), tile.to_vec());
    }

    /// Splits a self-contained JPEG into `(tables-only stream, abbreviated
    /// stream)`, the way a JPEG-compressed TIFF stores tag 347 and its tiles.
    #[cfg(feature = "decode")]
    fn split_jpeg(full: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let mut tables = vec![0xFFu8, 0xD8];
        let mut body = vec![0xFFu8, 0xD8];
        let mut index = 2usize;
        while let (Some(0xFF), Some(marker)) =
            (full.get(index).copied(), full.get(index + 1).copied())
        {
            if marker == 0xD9 {
                break;
            }
            let Some(raw) = full.get(index + 2..index + 4) else {
                break;
            };
            let length = usize::from(u16::from_be_bytes([raw[0], raw[1]]));
            let Some(segment) = full.get(index..index + 2 + length) else {
                break;
            };
            if matches!(marker, 0xDB | 0xC4 | 0xDD) {
                tables.extend_from_slice(segment);
            } else {
                body.extend_from_slice(segment);
            }
            index += 2 + length;
            if marker == 0xDA {
                // Entropy-coded data runs to the end of the stream.
                body.extend_from_slice(full.get(index..).unwrap_or(&[]));
                break;
            }
        }
        tables.extend_from_slice(&[0xFF, 0xD9]);
        (tables, body)
    }

    #[cfg(feature = "decode")]
    #[test]
    fn an_abbreviated_jpeg_tile_decodes_once_its_tables_are_spliced_in() {
        use crate::cog::meta::COMPRESSION_JPEG;
        use std::io::Cursor;

        let source = image::RgbImage::from_pixel(8, 8, image::Rgb([200, 100, 50]));
        let mut full = Vec::new();
        source
            .write_to(&mut Cursor::new(&mut full), image::ImageFormat::Jpeg)
            .expect("encoding a fixture JPEG must succeed in tests");
        let (tables, abbreviated) = split_jpeg(&full);
        assert!(
            abbreviated.len() < full.len(),
            "the abbreviated stream must actually be missing its tables"
        );

        let mut level = gray_level(8, 8, COMPRESSION_JPEG);
        level.samples_per_pixel = 3;
        // A JPEG-compressed colour TIFF declares YCbCr; the decoder converts.
        level.photometric = 6;

        assert!(
            decode_cog_block(&level, 0, true, &abbreviated, &options()).is_err(),
            "without the tables the stream cannot be decoded"
        );

        let rgba = decode_cog_block(
            &level,
            0,
            true,
            &abbreviated,
            &CogDecodeOptions {
                jpeg_tables: tables,
                ..CogDecodeOptions::default()
            },
        )
        .expect("an abbreviated JPEG tile must decode once its tables are spliced in");
        assert_eq!(rgba.len(), 8 * 8 * 4);
        // Lossy, so compare loosely; the point is that it is orange, not noise.
        assert!(rgba[0] > 170 && rgba[1] > 70 && rgba[1] < 130 && rgba[3] == 255);
    }

    #[cfg(feature = "decode")]
    #[test]
    fn an_oversized_jpeg_tile_is_refused_before_its_pixels_are_allocated() {
        use crate::cog::meta::COMPRESSION_JPEG;
        use std::io::Cursor;

        // `image`'s default limits leave width and height unbounded, so a tile
        // whose SOF claims far more than the level's block geometry would
        // materialise megabytes before any check of ours saw it — times the
        // source tiles `compose` now has in flight at once.
        let source = image::RgbImage::from_pixel(800, 800, image::Rgb([1, 2, 3]));
        let mut payload = Vec::new();
        source
            .write_to(&mut Cursor::new(&mut payload), image::ImageFormat::Jpeg)
            .expect("encoding a fixture JPEG must succeed in tests");

        let mut level = gray_level(8, 8, COMPRESSION_JPEG);
        level.samples_per_pixel = 3;
        level.photometric = 2;
        let error = decode_cog_block(&level, 0, true, &payload, &options())
            .expect_err("an 800x800 tile in an 8x8 block must be refused");
        let message = error.to_string();
        assert!(
            !message.contains("decoded to"),
            "the post-decode dimension check must never be reached: {message}"
        );
        assert!(
            message.contains("exceeds limit"),
            "the decoder must refuse it on the declared block geometry: {message}"
        );
    }

    #[cfg(feature = "decode-webp")]
    #[test]
    fn a_webp_tile_decodes() {
        use crate::cog::meta::COMPRESSION_WEBP;
        use std::io::Cursor;

        let source = image::RgbaImage::from_pixel(8, 8, image::Rgba([10, 120, 240, 255]));
        let mut payload = Vec::new();
        source
            .write_to(&mut Cursor::new(&mut payload), image::ImageFormat::WebP)
            .expect("encoding a fixture WebP must succeed in tests");

        let mut level = gray_level(8, 8, COMPRESSION_WEBP);
        level.samples_per_pixel = 3;
        level.photometric = 2;
        let rgba = decode_cog_block(&level, 0, true, &payload, &options())
            .expect("a WebP tile must decode");
        assert_eq!(rgba.len(), 8 * 8 * 4);
        assert_eq!(&rgba[..4], &[10, 120, 240, 255]);
    }
}
