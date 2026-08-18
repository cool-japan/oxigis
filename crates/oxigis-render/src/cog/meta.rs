//! Parsed COG metadata: one record per pyramid level, plus the georeference.
//!
//! # Provenance
//!
//! The per-level field set and the tile-directory indexing are ported from
//! `oxigeo-wasm`'s `IfdMetadata`/`read_tile_level` (cool-japan/oxigeo,
//! Apache-2.0, same author). The georeference types and the CRS classification
//! are new: `oxigeo-wasm`'s viewer draws a COG in its own pixel space and never
//! needs to place it on a Web Mercator tile grid.

use crate::error::RenderError;
use crate::source::ByteRange;

/// TIFF `Compression` value for uncompressed data.
pub const COMPRESSION_NONE: u16 = 1;
/// TIFF `Compression` value for LZW.
pub const COMPRESSION_LZW: u16 = 5;
/// TIFF `Compression` value for baseline JPEG.
pub const COMPRESSION_JPEG: u16 = 7;
/// TIFF `Compression` value for zlib-wrapped DEFLATE (the Adobe code).
pub const COMPRESSION_DEFLATE: u16 = 8;
/// TIFF `Compression` value for PackBits run-length encoding.
pub const COMPRESSION_PACKBITS: u16 = 32_773;
/// TIFF `Compression` value for zlib-wrapped DEFLATE (the older, unofficial
/// code GDAL still writes for some products).
pub const COMPRESSION_DEFLATE_OLD: u16 = 32_946;
/// TIFF `Compression` value for LERC, GDAL's error-bounded raster codec.
pub const COMPRESSION_LERC: u16 = 34_887;
/// TIFF `Compression` value for ZSTD, as written by GDAL's COG driver.
pub const COMPRESSION_ZSTD: u16 = 50_000;
/// TIFF `Compression` value for WebP, as written by GDAL's COG driver.
pub const COMPRESSION_WEBP: u16 = 50_001;

/// `PhotometricInterpretation` value for greyscale with 0 = white.
pub const PHOTOMETRIC_WHITE_IS_ZERO: u16 = 0;
/// `PhotometricInterpretation` value for greyscale with 0 = black.
pub const PHOTOMETRIC_BLACK_IS_ZERO: u16 = 1;
/// `PhotometricInterpretation` value for RGB.
pub const PHOTOMETRIC_RGB: u16 = 2;
/// `PhotometricInterpretation` value for palette colour.
pub const PHOTOMETRIC_PALETTE: u16 = 3;
/// `PhotometricInterpretation` value for a transparency mask.
pub const PHOTOMETRIC_MASK: u16 = 4;
/// `PhotometricInterpretation` value for separated colour (CMYK).
pub const PHOTOMETRIC_CMYK: u16 = 5;
/// `PhotometricInterpretation` value for YCbCr, which is what a JPEG-compressed
/// colour TIFF declares.
pub const PHOTOMETRIC_YCBCR: u16 = 6;
/// `PhotometricInterpretation` value for CIE L*a*b*.
pub const PHOTOMETRIC_CIELAB: u16 = 8;

/// Largest number of bytes one decompressed block ([`CogLevel::tile_bytes`])
/// may occupy.
///
/// Tile width and height come straight out of tags 322/323, so without a
/// ceiling a header declaring 65536x65536 makes the reader reserve 4 GiB before
/// it has read one byte of payload — and 2^30 x 2^30 reserves 2^60, which is
/// under `isize::MAX` and therefore reaches the allocator and aborts the
/// process. The cap is on *bytes* rather than on a tile edge on purpose: a
/// striped TIFF's block is as wide as the image, routinely tens of thousands of
/// pixels, so an edge limit would reject legitimate files. 64 MiB is two orders
/// of magnitude past a 1024x1024 four-band tile, the largest GDAL writes. The
/// one legitimate shape it turns away is a TIFF written as a single strip
/// covering the whole image, which cannot be read incrementally at all — that
/// is a file to re-encode as a tiled COG, not one to fetch whole per map tile.
pub const MAX_TILE_DECOMPRESSED_BYTES: usize = 64 * 1024 * 1024;

/// `SampleFormat` value for unsigned integers.
pub const SAMPLE_FORMAT_UINT: u16 = 1;
/// `SampleFormat` value for two's-complement signed integers.
pub const SAMPLE_FORMAT_INT: u16 = 2;
/// `SampleFormat` value for IEEE-754 floats.
pub const SAMPLE_FORMAT_FLOAT: u16 = 3;

/// One resolution level of a COG: the full-resolution image, or an overview.
///
/// Every level carries its own tile geometry, sample layout, codec and tile
/// directory, because a COG's overviews are free to differ from the
/// full-resolution image in all of them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CogLevel {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Tile width in pixels.
    pub tile_width: u32,
    /// Tile height in pixels.
    pub tile_height: u32,
    /// Bits per sample (8 and 16 are renderable; see [`super::codec`]).
    pub bits_per_sample: u16,
    /// Samples (bands) per pixel.
    pub samples_per_pixel: u16,
    /// TIFF `SampleFormat` (tag 339).
    pub sample_format: u16,
    /// TIFF `Compression` (tag 259).
    pub compression: u16,
    /// TIFF `Predictor` (tag 317): 1 = none, 2 = horizontal differencing.
    pub predictor: u16,
    /// TIFF `PhotometricInterpretation` (tag 262).
    pub photometric: u16,
    /// The level's `ColorMap` (tag 320), empty when it declares none.
    ///
    /// TIFF stores it as three consecutive ramps — all reds, then all greens,
    /// then all blues — of `2^BitsPerSample` entries each, scaled to
    /// `0..=65535`. A non-empty map means the samples are palette indices; see
    /// [`super::codec`].
    pub color_map: Vec<u16>,
    /// Byte offset of each tile, in row-major tile order.
    pub tile_offsets: Vec<u64>,
    /// Byte length of each tile, parallel to [`CogLevel::tile_offsets`].
    pub tile_byte_counts: Vec<u64>,
}

impl CogLevel {
    /// Number of tile columns.
    #[must_use]
    pub const fn tiles_across(&self) -> u32 {
        if self.tile_width == 0 {
            return 0;
        }
        self.width.div_ceil(self.tile_width)
    }

    /// Number of tile rows.
    #[must_use]
    pub const fn tiles_down(&self) -> u32 {
        if self.tile_height == 0 {
            return 0;
        }
        self.height.div_ceil(self.tile_height)
    }

    /// Index of `(tile_x, tile_y)` in the tile directory, or `None` when the
    /// coordinates fall outside the level's tile grid.
    #[must_use]
    pub fn tile_index(&self, tile_x: u32, tile_y: u32) -> Option<usize> {
        if tile_x >= self.tiles_across() || tile_y >= self.tiles_down() {
            return None;
        }
        let index =
            usize::try_from(u64::from(tile_y) * u64::from(self.tiles_across()) + u64::from(tile_x))
                .ok()?;
        (index < self.tile_offsets.len() && index < self.tile_byte_counts.len()).then_some(index)
    }

    /// Byte range of one tile's compressed payload.
    ///
    /// `Ok(None)` means the tile is *sparse*: COG writers record a zero byte
    /// count for a tile that was never written (an all-nodata tile), and there
    /// is nothing to fetch.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidRange`] if `(tile_x, tile_y)` is outside
    /// the level's tile grid or the directory entry overflows `u64`.
    pub fn tile_range(&self, tile_x: u32, tile_y: u32) -> Result<Option<ByteRange>, RenderError> {
        let index = self.tile_index(tile_x, tile_y).ok_or_else(|| {
            RenderError::InvalidRange(format!(
                "tile ({tile_x}, {tile_y}) is outside a {}x{} tile grid",
                self.tiles_across(),
                self.tiles_down()
            ))
        })?;
        let (Some(offset), Some(length)) = (
            self.tile_offsets.get(index),
            self.tile_byte_counts.get(index),
        ) else {
            return Err(RenderError::InvalidRange(format!(
                "tile directory entry {index} is missing"
            )));
        };
        if *length == 0 {
            return Ok(None);
        }
        ByteRange::with_len(*offset, *length).map(Some)
    }

    /// Whether the level declares a palette (`ColorMap`, tag 320).
    #[must_use]
    pub fn has_color_map(&self) -> bool {
        !self.color_map.is_empty()
    }

    /// Palette entry `index` as an opaque RGB triple, scaled to `0..=255`.
    ///
    /// `None` when there is no palette, or the index falls outside it.
    #[must_use]
    pub fn palette_rgb(&self, index: usize) -> Option<[u8; 3]> {
        let entries = self.color_map.len() / 3;
        if entries == 0 || index >= entries {
            return None;
        }
        let scale = |value: u16| -> u8 {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a 16-bit value divided by 257 is 0..=255"
            )]
            let byte = (u32::from(value) / 257) as u8;
            byte
        };
        Some([
            scale(*self.color_map.get(index)?),
            scale(*self.color_map.get(entries + index)?),
            scale(*self.color_map.get(2 * entries + index)?),
        ])
    }

    /// Number of image rows the block in tile row `tile_y` actually carries.
    ///
    /// TIFF pads a *tile* to its full declared height, but a *strip* holds only
    /// the rows that are left: the final strip of an image whose height is not
    /// a multiple of `RowsPerStrip` is short, and its `StripByteCounts` payload
    /// decompresses to fewer bytes than a full block. Taking the minimum is
    /// right for both, because the rows a tile is padded with lie past
    /// [`CogLevel::height`] and are never sampled.
    #[must_use]
    pub const fn block_rows(&self, tile_y: u32) -> u32 {
        if self.tile_height == 0 {
            return 0;
        }
        let consumed = tile_y.saturating_mul(self.tile_height);
        let remaining = self.height.saturating_sub(consumed);
        if remaining < self.tile_height {
            remaining
        } else {
            self.tile_height
        }
    }

    /// Number of bytes one row of a block occupies once decompressed.
    #[must_use]
    pub fn block_row_bytes(&self) -> Option<usize> {
        let bits = u64::from(self.tile_width)
            .checked_mul(u64::from(self.samples_per_pixel))?
            .checked_mul(u64::from(self.bits_per_sample))?;
        usize::try_from(bits.div_ceil(8)).ok()
    }

    /// Number of bytes one fully decompressed block occupies.
    ///
    /// Rows are the unit of padding: TIFF rounds each row up to a whole byte,
    /// so a 1-bit block is `block_row_bytes * tile_height`, which is larger
    /// than rounding the block's total bit count up once. The two agree exactly
    /// at the 8- and 16-bit depths [`super::codec`] renders, and the row-padded
    /// figure is the one a decoder must be given so a sub-byte file reports its
    /// bit depth as unsupported rather than "the buffer is too small".
    #[must_use]
    pub fn tile_bytes(&self) -> Option<usize> {
        self.block_row_bytes()?
            .checked_mul(self.tile_height as usize)
    }
}

/// A north-up affine mapping from level-0 pixel coordinates to CRS coordinates.
///
/// Derived from GeoTIFF's `ModelPixelScale` (33550) and `ModelTiepoint`
/// (33922). Per the GeoTIFF specification both scale components are stored
/// **positive** and the northing decreases with the raster row:
///
/// ```text
/// crs_x = origin_x + pixel_x * pixel_size_x
/// crs_y = origin_y - pixel_y * pixel_size_y
/// ```
///
/// Note that `oxigeo-wasm`'s `pixel_scale_y()` accessor documents its value as
/// "degrees/pixel in lat direction, negative", which does not match either the
/// specification or what the tag holds in files GDAL writes; this reader takes
/// the magnitude and applies the sign itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CogGeoTransform {
    /// CRS easting/longitude of the level-0 pixel-grid origin.
    pub origin_x: f64,
    /// CRS northing/latitude of the level-0 pixel-grid origin.
    pub origin_y: f64,
    /// Level-0 pixel width in CRS units, always positive.
    pub pixel_size_x: f64,
    /// Level-0 pixel height in CRS units, always positive.
    pub pixel_size_y: f64,
}

impl CogGeoTransform {
    /// Builds a transform from the raw GeoTIFF tag values.
    ///
    /// `tiepoint` is `(raster_x, raster_y, crs_x, crs_y)` — the first tie point
    /// of tag 33922, whose raster part is `(0, 0)` in every COG in practice but
    /// is honoured here anyway.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Decode`] if either pixel size is zero or not
    /// finite, which would make the mapping non-invertible.
    pub fn new(
        tiepoint: (f64, f64, f64, f64),
        scale_x: f64,
        scale_y: f64,
    ) -> Result<Self, RenderError> {
        let pixel_size_x = scale_x.abs();
        let pixel_size_y = scale_y.abs();
        if !pixel_size_x.is_finite() || !pixel_size_y.is_finite() {
            return Err(RenderError::Decode(
                "GeoTIFF ModelPixelScale is not finite".to_owned(),
            ));
        }
        if pixel_size_x == 0.0 || pixel_size_y == 0.0 {
            return Err(RenderError::Decode(
                "GeoTIFF ModelPixelScale has a zero component".to_owned(),
            ));
        }
        let (raster_x, raster_y, crs_x, crs_y) = tiepoint;
        if !raster_x.is_finite()
            || !raster_y.is_finite()
            || !crs_x.is_finite()
            || !crs_y.is_finite()
        {
            return Err(RenderError::Decode(
                "GeoTIFF ModelTiepoint is not finite".to_owned(),
            ));
        }
        Ok(Self {
            origin_x: crs_x - raster_x * pixel_size_x,
            origin_y: crs_y + raster_y * pixel_size_y,
            pixel_size_x,
            pixel_size_y,
        })
    }

    /// The same transform expressed for a level whose pixels are `factor_x` /
    /// `factor_y` times larger than level 0's.
    #[must_use]
    pub fn scaled(self, factor_x: f64, factor_y: f64) -> Self {
        Self {
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            pixel_size_x: self.pixel_size_x * factor_x,
            pixel_size_y: self.pixel_size_y * factor_y,
        }
    }

    /// CRS coordinate of a pixel position (pixel centres are at `+0.5`).
    #[must_use]
    pub fn to_crs(self, pixel_x: f64, pixel_y: f64) -> (f64, f64) {
        (
            self.origin_x + pixel_x * self.pixel_size_x,
            self.origin_y - pixel_y * self.pixel_size_y,
        )
    }

    /// Pixel position of a CRS coordinate — the inverse of
    /// [`CogGeoTransform::to_crs`].
    #[must_use]
    pub fn to_pixel(self, crs_x: f64, crs_y: f64) -> (f64, f64) {
        (
            (crs_x - self.origin_x) / self.pixel_size_x,
            (self.origin_y - crs_y) / self.pixel_size_y,
        )
    }
}

/// The coordinate reference systems this reader can place on a tile grid.
///
/// # Coverage
///
/// * [`CogCrs::WebMercator`] and [`CogCrs::Geographic`] are *native*: their
///   coordinates map to the Web Mercator tile grid one axis at a time, with no
///   projection step.
/// * [`CogCrs::Utm`] — the WGS 84 UTM zones, EPSG:32601–32660 (north) and
///   EPSG:32701–32760 (south) — is reprojected per pixel through
///   [`super::tmerc`]. Between them these cover essentially every real
///   satellite and aerial product: Sentinel-2, NAIP, OpenAerialMap and Planet
///   all ship in a UTM zone.
///
/// Anything else — a national grid, a polar stereographic product, an
/// equal-area projection — is rejected with [`RenderError::Unsupported`] rather
/// than drawn in the wrong place. Adding one means adding its projection, not
/// relaxing a check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CogCrs {
    /// EPSG:3857 and its aliases — coordinates are Web Mercator metres.
    WebMercator,
    /// EPSG:4326 — coordinates are WGS 84 degrees, longitude first.
    Geographic,
    /// A WGS 84 UTM zone — coordinates are Transverse Mercator metres.
    Utm {
        /// Zone number, `1..=60`.
        zone: u8,
        /// Whether this is the northern-hemisphere zone (EPSG:326`zz`) rather
        /// than the southern one (EPSG:327`zz`, false northing 10 000 km).
        north: bool,
    },
}

impl CogCrs {
    /// Classifies an EPSG code.
    ///
    /// 3857 has accumulated aliases over the years (900913 from the original
    /// Google/OSM era, 3785 deprecated, 102100/102113 from ESRI); all denote the
    /// same spherical Mercator coordinates.
    #[must_use]
    pub const fn from_epsg(epsg: u32) -> Option<Self> {
        match epsg {
            3857 | 3785 | 900_913 | 102_100 | 102_113 => Some(Self::WebMercator),
            4326 => Some(Self::Geographic),
            32_601..=32_660 => Some(Self::Utm {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the match arm bounds the remainder to 1..=60"
                )]
                zone: (epsg - 32_600) as u8,
                north: true,
            }),
            32_701..=32_760 => Some(Self::Utm {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the match arm bounds the remainder to 1..=60"
                )]
                zone: (epsg - 32_700) as u8,
                north: false,
            }),
            _ => None,
        }
    }

    /// Whether this CRS places on the tile grid one axis at a time, i.e. needs
    /// no projection step.
    ///
    /// [`CogCrs::Utm`] does not: its easting depends on the latitude and its
    /// northing on the longitude, so the mapping cannot be split into an
    /// x-mapping and a y-mapping the way [`super::plan`]'s fast path does.
    #[must_use]
    pub const fn is_axis_separable(self) -> bool {
        matches!(self, Self::WebMercator | Self::Geographic)
    }

    /// Full extent of the CRS along each axis, in CRS units: `(x_span,
    /// y_span)`. Used to express a pixel size in normalised world units.
    ///
    /// `None` for [`CogCrs::Utm`], which has no such span: a UTM metre is a
    /// ground metre, and how much of the normalised Web Mercator world that
    /// covers depends on the latitude (by `1 / cos φ`, a factor of two by 60°).
    /// Returning `None` rather than a plausible-looking constant is deliberate
    /// — see [`CogMetadata::world_pixel_size`], which handles UTM itself.
    #[must_use]
    pub fn axis_spans(self) -> Option<(f64, f64)> {
        match self {
            Self::WebMercator => Some((
                crate::mercator::EARTH_CIRCUMFERENCE_M,
                crate::mercator::EARTH_CIRCUMFERENCE_M,
            )),
            Self::Geographic => Some((360.0, 360.0)),
            Self::Utm { .. } => None,
        }
    }
}

/// Everything a COG reader needs to turn a map tile request into byte ranges.
#[derive(Debug, Clone, PartialEq)]
pub struct CogMetadata {
    /// Whether the file is little-endian (`II`) rather than big-endian (`MM`).
    pub little_endian: bool,
    /// Number of samples (bands) per pixel of the full-resolution image.
    pub samples_per_pixel: u16,
    /// Resolution levels, `levels[0]` being the full-resolution image and the
    /// rest the overviews in the order the IFD chain lists them.
    pub levels: Vec<CogLevel>,
    /// EPSG code declared by the file's GeoKeys, when it declares one.
    pub epsg: Option<u32>,
    /// Level-0 pixel-to-CRS mapping, when the file is georeferenced.
    pub geo: Option<CogGeoTransform>,
}

impl Default for CogMetadata {
    fn default() -> Self {
        Self {
            little_endian: true,
            samples_per_pixel: 1,
            levels: Vec::new(),
            epsg: None,
            geo: None,
        }
    }
}

impl CogMetadata {
    /// The full-resolution level.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Decode`] when no usable IFD was found.
    pub fn base_level(&self) -> Result<&CogLevel, RenderError> {
        self.levels
            .first()
            .ok_or_else(|| RenderError::Decode("COG has no usable image directory".to_owned()))
    }

    /// One resolution level by index.
    #[must_use]
    pub fn level(&self, index: usize) -> Option<&CogLevel> {
        self.levels.get(index)
    }

    /// Number of resolution levels (full resolution plus overviews).
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// The CRS the georeference is expressed in.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] when the file declares no EPSG code
    /// or one this reader cannot place on a Web Mercator grid — see [`CogCrs`].
    pub fn crs(&self) -> Result<CogCrs, RenderError> {
        let epsg = self.epsg.ok_or_else(|| {
            RenderError::Unsupported(
                "COG declares no EPSG code, so it cannot be placed on the tile grid".to_owned(),
            )
        })?;
        CogCrs::from_epsg(epsg).ok_or_else(|| {
            RenderError::Unsupported(format!(
                "COG is in EPSG:{epsg}; supported CRSs are EPSG:3857, EPSG:4326 and the WGS 84 \
                 UTM zones (EPSG:32601-32660 / 32701-32760)"
            ))
        })
    }

    /// The level-0 georeference.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] when the file carries no
    /// `ModelPixelScale`/`ModelTiepoint` pair, e.g. a plain TIFF or a GeoTIFF
    /// georeferenced by `ModelTransformation` (34264), which this reader does
    /// not read.
    pub fn geo_transform(&self) -> Result<CogGeoTransform, RenderError> {
        self.geo.ok_or_else(|| {
            RenderError::Unsupported(
                "COG has no ModelPixelScale/ModelTiepoint georeference".to_owned(),
            )
        })
    }

    /// Georeference of one level, derived from level 0's by the ratio of their
    /// pixel dimensions.
    ///
    /// Overviews are *nominally* exact halvings, but a level whose size was
    /// rounded up (an odd-sized parent) covers slightly more ground per pixel,
    /// so the ratio is computed rather than assumed.
    ///
    /// # Errors
    ///
    /// Propagates [`CogMetadata::geo_transform`], and returns
    /// [`RenderError::Decode`] for a missing or degenerate level.
    pub fn level_transform(&self, level: usize) -> Result<CogGeoTransform, RenderError> {
        let base = self.base_level()?;
        let target = self
            .level(level)
            .ok_or_else(|| RenderError::Decode(format!("COG has no level {level}")))?;
        if target.width == 0 || target.height == 0 {
            return Err(RenderError::Decode(format!(
                "COG level {level} has a zero dimension"
            )));
        }
        let factor_x = f64::from(base.width) / f64::from(target.width);
        let factor_y = f64::from(base.height) / f64::from(target.height);
        Ok(self.geo_transform()?.scaled(factor_x, factor_y))
    }
}

#[cfg(test)]
mod tests {
    use super::{CogCrs, CogGeoTransform, CogLevel, CogMetadata};
    use crate::error::RenderError;
    use crate::source::ByteRange;

    fn level(width: u32, height: u32, tile: u32, tiles: usize) -> CogLevel {
        CogLevel {
            width,
            height,
            tile_width: tile,
            tile_height: tile,
            bits_per_sample: 8,
            samples_per_pixel: 1,
            sample_format: 1,
            compression: 1,
            predictor: 1,
            photometric: 1,
            color_map: Vec::new(),
            tile_offsets: (0..tiles as u64).map(|i| 1_000 + i * 100).collect(),
            tile_byte_counts: vec![100; tiles],
        }
    }

    #[test]
    fn tile_grid_is_ceiling_divided() {
        let lvl = level(300, 200, 256, 2);
        assert_eq!(lvl.tiles_across(), 2);
        assert_eq!(lvl.tiles_down(), 1);
        assert_eq!(lvl.tile_index(0, 0), Some(0));
        assert_eq!(lvl.tile_index(1, 0), Some(1));
        assert_eq!(lvl.tile_index(2, 0), None);
        assert_eq!(lvl.tile_index(0, 1), None);
        assert_eq!(lvl.tile_bytes(), Some(256 * 256));
    }

    #[test]
    fn block_rows_shortens_the_last_row_of_blocks() {
        // 300x200 at 256-pixel blocks: the second block row holds 200 - 256's
        // worth of rows, i.e. all that is left. TIFF pads a *tile* to a full
        // block, so those rows exist in the payload but lie past the image; it
        // does not pad a *strip*, so for a strip they are simply not there.
        // Taking the minimum is right for both.
        let lvl = level(300, 200, 256, 2);
        assert_eq!(lvl.block_rows(0), 200);
        assert_eq!(lvl.block_rows(1), 0);
        assert_eq!(lvl.block_row_bytes(), Some(256));

        let mut striped = level(8, 10, 8, 3);
        striped.tile_height = 4;
        assert_eq!(striped.block_rows(0), 4);
        assert_eq!(striped.block_rows(1), 4);
        assert_eq!(striped.block_rows(2), 2, "the final strip is short");
        assert_eq!(striped.block_rows(9), 0, "past the image there are none");
        assert_eq!(striped.block_row_bytes(), Some(8));

        // 16-bit RGB: three bands of two bytes each.
        let mut wide = level(4, 4, 4, 1);
        wide.bits_per_sample = 16;
        wide.samples_per_pixel = 3;
        assert_eq!(wide.block_row_bytes(), Some(4 * 3 * 2));
        assert_eq!(wide.tile_bytes(), Some(4 * 4 * 3 * 2));

        // A degenerate block height has no rows rather than dividing by zero.
        let mut degenerate = level(8, 8, 8, 1);
        degenerate.tile_height = 0;
        assert_eq!(degenerate.block_rows(0), 0);
    }

    #[test]
    fn a_sub_byte_block_is_sized_by_its_padded_rows() {
        // TIFF pads every row to a whole byte, so a 12-pixel-wide 1-bit block
        // is 2 bytes per row, not 12 bits. Rounding the block's total bit count
        // up once would say 15 bytes for eight rows instead of 16, and hand the
        // decoder a buffer one byte short of the payload it is about to read.
        let mut bilevel = level(12, 8, 12, 1);
        bilevel.tile_height = 8;
        bilevel.bits_per_sample = 1;
        assert_eq!(bilevel.block_row_bytes(), Some(2));
        assert_eq!(bilevel.tile_bytes(), Some(16));

        // The two agree exactly at the depths the codec renders.
        let byte_aligned = level(256, 256, 256, 1);
        assert_eq!(byte_aligned.tile_bytes(), Some(256 * 256));
    }

    #[test]
    fn a_degenerate_tile_size_has_no_grid() {
        let mut lvl = level(300, 200, 256, 2);
        lvl.tile_width = 0;
        lvl.tile_height = 0;
        assert_eq!(lvl.tiles_across(), 0);
        assert_eq!(lvl.tiles_down(), 0);
    }

    #[test]
    fn tile_ranges_come_from_the_directory() {
        let lvl = level(512, 256, 256, 2);
        assert_eq!(
            lvl.tile_range(1, 0).ok(),
            Some(Some(ByteRange {
                start: 1_100,
                end: 1_200
            }))
        );
        assert!(matches!(
            lvl.tile_range(5, 5),
            Err(RenderError::InvalidRange(_))
        ));
    }

    #[test]
    fn a_zero_byte_count_is_a_sparse_tile() {
        let mut lvl = level(512, 256, 256, 2);
        lvl.tile_byte_counts[0] = 0;
        assert_eq!(lvl.tile_range(0, 0).ok(), Some(None));
    }

    #[test]
    fn a_short_directory_is_rejected_rather_than_indexed() {
        let mut lvl = level(512, 256, 256, 2);
        lvl.tile_offsets.pop();
        assert_eq!(lvl.tile_index(1, 0), None);
        assert!(lvl.tile_range(1, 0).is_err());
    }

    #[test]
    fn the_geotransform_is_north_up_with_positive_scales() {
        // A GDAL-style geographic transform: origin at (10, 50), 0.01 deg/px.
        let geo = CogGeoTransform::new((0.0, 0.0, 10.0, 50.0), 0.01, 0.01)
            .expect("a well-formed transform");
        assert_eq!(geo.origin_x, 10.0);
        assert_eq!(geo.origin_y, 50.0);
        let (x, y) = geo.to_crs(100.0, 200.0);
        assert!((x - 11.0).abs() < 1e-12);
        // Northing *decreases* with the row index.
        assert!((y - 48.0).abs() < 1e-12);
        let (px, py) = geo.to_pixel(x, y);
        assert!((px - 100.0).abs() < 1e-9);
        assert!((py - 200.0).abs() < 1e-9);
    }

    #[test]
    fn a_negative_scale_is_taken_as_a_magnitude() {
        // Some writers (and oxigeo-wasm's accessor docs) treat ScaleY as
        // negative; the sign must not flip the image upside down.
        let geo = CogGeoTransform::new((0.0, 0.0, 10.0, 50.0), 0.01, -0.01)
            .expect("a well-formed transform");
        assert_eq!(geo.pixel_size_y, 0.01);
        let (_, y) = geo.to_crs(0.0, 100.0);
        assert!(y < 50.0);
    }

    #[test]
    fn a_degenerate_geotransform_is_rejected() {
        assert!(CogGeoTransform::new((0.0, 0.0, 0.0, 0.0), 0.0, 1.0).is_err());
        assert!(CogGeoTransform::new((0.0, 0.0, 0.0, 0.0), 1.0, f64::NAN).is_err());
        assert!(CogGeoTransform::new((0.0, f64::INFINITY, 0.0, 0.0), 1.0, 1.0).is_err());
    }

    #[test]
    fn a_tiepoint_with_a_raster_offset_shifts_the_origin() {
        let geo = CogGeoTransform::new((10.0, 20.0, 100.0, 200.0), 2.0, 4.0)
            .expect("a well-formed transform");
        assert_eq!(geo.origin_x, 80.0);
        assert_eq!(geo.origin_y, 280.0);
        let (x, y) = geo.to_crs(10.0, 20.0);
        assert!((x - 100.0).abs() < 1e-12);
        assert!((y - 200.0).abs() < 1e-12);
    }

    #[test]
    fn epsg_codes_are_classified_with_their_aliases() {
        assert_eq!(CogCrs::from_epsg(3857), Some(CogCrs::WebMercator));
        assert_eq!(CogCrs::from_epsg(900_913), Some(CogCrs::WebMercator));
        assert_eq!(CogCrs::from_epsg(102_100), Some(CogCrs::WebMercator));
        assert_eq!(CogCrs::from_epsg(4326), Some(CogCrs::Geographic));
        assert_eq!(
            CogCrs::from_epsg(32_654),
            Some(CogCrs::Utm {
                zone: 54,
                north: true
            })
        );
        assert_eq!(
            CogCrs::from_epsg(32_601),
            Some(CogCrs::Utm {
                zone: 1,
                north: true
            })
        );
        assert_eq!(
            CogCrs::from_epsg(32_660),
            Some(CogCrs::Utm {
                zone: 60,
                north: true
            })
        );
        assert_eq!(
            CogCrs::from_epsg(32_734),
            Some(CogCrs::Utm {
                zone: 34,
                north: false
            })
        );
        // The zone-0 and zone-61 slots (…600/…700 and …661/…761) are not zones.
        assert_eq!(CogCrs::from_epsg(32_600), None);
        assert_eq!(CogCrs::from_epsg(32_661), None);
        assert_eq!(CogCrs::from_epsg(32_700), None);
        assert_eq!(CogCrs::from_epsg(32_761), None);
        // A national grid still needs its own projection.
        assert_eq!(CogCrs::from_epsg(27_700), None);
        assert_eq!(CogCrs::from_epsg(3035), None);

        assert_eq!(
            CogCrs::Geographic.axis_spans().map(|spans| spans.0),
            Some(360.0)
        );
        assert!(
            CogCrs::WebMercator
                .axis_spans()
                .is_some_and(|spans| spans.0 > 40_000_000.0)
        );
        // UTM has no global axis span; `world_pixel_size` handles it directly.
        assert_eq!(
            CogCrs::Utm {
                zone: 54,
                north: true
            }
            .axis_spans(),
            None
        );
        assert!(CogCrs::Geographic.is_axis_separable());
        assert!(CogCrs::WebMercator.is_axis_separable());
        assert!(
            !CogCrs::Utm {
                zone: 54,
                north: true
            }
            .is_axis_separable()
        );
    }

    #[test]
    fn metadata_rejects_an_unsupported_or_absent_crs() {
        let mut meta = CogMetadata::default();
        assert!(matches!(meta.crs(), Err(RenderError::Unsupported(_))));
        meta.epsg = Some(27_700);
        assert!(matches!(meta.crs(), Err(RenderError::Unsupported(_))));
        meta.epsg = Some(4326);
        assert_eq!(meta.crs().ok(), Some(CogCrs::Geographic));
        assert!(matches!(
            meta.geo_transform(),
            Err(RenderError::Unsupported(_))
        ));
        assert!(meta.base_level().is_err());
        assert_eq!(meta.level_count(), 0);
        assert_eq!(meta.level(0), None);
    }

    #[test]
    fn level_transforms_scale_by_the_measured_ratio() {
        let meta = CogMetadata {
            little_endian: true,
            samples_per_pixel: 1,
            levels: vec![level(1024, 1024, 256, 16), level(512, 512, 256, 4)],
            epsg: Some(4326),
            geo: Some(
                CogGeoTransform::new((0.0, 0.0, 0.0, 0.0), 0.001, 0.001)
                    .expect("a well-formed transform"),
            ),
        };
        let base = meta.level_transform(0).expect("level 0 transform");
        assert!((base.pixel_size_x - 0.001).abs() < 1e-15);
        let overview = meta.level_transform(1).expect("level 1 transform");
        assert!((overview.pixel_size_x - 0.002).abs() < 1e-15);
        assert!(meta.level_transform(9).is_err());
    }
}
