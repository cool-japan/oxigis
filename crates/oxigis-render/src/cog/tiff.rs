//! Byte-level TIFF primitives: byte order, header, IFD entries, field values.
//!
//! Everything in this module is a **pure function of a byte slice**. Nothing
//! fetches, so the same code serves the pull-based state machine in
//! [`super::open`], the async [`super::CogSource`] wrapper and the unit tests
//! that hand-craft TIFF bytes.
//!
//! # Provenance
//!
//! The tag set, the "value fits in four bytes or is an offset" rule and the
//! GeoKey lookup are ported from `oxigeo-wasm`'s `src/cog_reader.rs`
//! (cool-japan/oxigeo, Apache-2.0, same author). The async `FetchBackend`
//! calls of the original are replaced by the [`super::blocks::ByteBlocks`]
//! window store so that no step of the parse performs I/O.

use crate::error::RenderError;

/// TIFF `ImageWidth` (256).
pub const TAG_IMAGE_WIDTH: u16 = 256;
/// TIFF `ImageLength` (257).
pub const TAG_IMAGE_LENGTH: u16 = 257;
/// TIFF `BitsPerSample` (258).
pub const TAG_BITS_PER_SAMPLE: u16 = 258;
/// TIFF `Compression` (259).
pub const TAG_COMPRESSION: u16 = 259;
/// TIFF `PhotometricInterpretation` (262).
pub const TAG_PHOTOMETRIC: u16 = 262;
/// TIFF `StripOffsets` (273).
pub const TAG_STRIP_OFFSETS: u16 = 273;
/// TIFF `SamplesPerPixel` (277).
pub const TAG_SAMPLES_PER_PIXEL: u16 = 277;
/// TIFF `RowsPerStrip` (278).
pub const TAG_ROWS_PER_STRIP: u16 = 278;
/// TIFF `StripByteCounts` (279).
pub const TAG_STRIP_BYTE_COUNTS: u16 = 279;
/// TIFF `Predictor` (317): 1 = none, 2 = horizontal differencing.
pub const TAG_PREDICTOR: u16 = 317;
/// TIFF `JPEGTables` (347) — the quantisation and Huffman tables shared by
/// every abbreviated JPEG tile stream of a `Compression` 7 image.
pub const TAG_JPEG_TABLES: u16 = 347;
/// TIFF `ColorMap` (320) — its presence means palette imagery.
pub const TAG_COLOR_MAP: u16 = 320;
/// TIFF `TileWidth` (322).
pub const TAG_TILE_WIDTH: u16 = 322;
/// TIFF `TileLength` (323).
pub const TAG_TILE_LENGTH: u16 = 323;
/// TIFF `TileOffsets` (324).
pub const TAG_TILE_OFFSETS: u16 = 324;
/// TIFF `TileByteCounts` (325).
pub const TAG_TILE_BYTE_COUNTS: u16 = 325;
/// TIFF `SampleFormat` (339): 1 = uint, 2 = int, 3 = IEEE float.
pub const TAG_SAMPLE_FORMAT: u16 = 339;
/// TIFF `NewSubfileType` (254); bit 2 marks a transparency-mask IFD.
pub const TAG_NEW_SUBFILE_TYPE: u16 = 254;
/// GeoTIFF `ModelPixelScale` (33550).
pub const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
/// GeoTIFF `ModelTiepoint` (33922).
pub const TAG_MODEL_TIEPOINT: u16 = 33922;
/// GeoTIFF `GeoKeyDirectory` (34735).
pub const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
/// GDAL `GDAL_NODATA` (42113) — an ASCII field type (2) holding the nodata
/// value as text, e.g. `-9999` or `nan`.
pub const TAG_GDAL_NODATA: u16 = 42113;

/// `NewSubfileType` bit that marks an IFD as a transparency mask.
///
/// GDAL writes internal mask IFDs into the same next-IFD chain as the
/// overviews, so a reader that walks the chain blindly would treat a 1-bit mask
/// as an image pyramid level.
pub const SUBFILE_TYPE_MASK: u32 = 0x4;

/// GeoKey `ProjectedCSTypeGeoKey` (3072).
pub const GEOKEY_PROJECTED_CS_TYPE: u16 = 3072;
/// GeoKey `GeographicTypeGeoKey` (2048).
pub const GEOKEY_GEOGRAPHIC_TYPE: u16 = 2048;
/// GeoKey value meaning "user-defined", i.e. no EPSG code.
pub const GEOKEY_USER_DEFINED: u16 = 32767;

/// Size in bytes of one classic-TIFF IFD entry (tag, type, count,
/// value/offset). BigTIFF widens it to 20 — see [`TiffVariant::entry_bytes`].
pub const IFD_ENTRY_BYTES: usize = 12;

/// Bytes of the value field embedded in a classic-TIFF IFD entry; anything
/// longer is stored elsewhere in the file and the field holds an offset
/// instead. BigTIFF widens it to 8 — see
/// [`TiffVariant::inline_value_bytes`].
pub const IFD_INLINE_VALUE_BYTES: usize = 4;

/// Largest inline value field of any variant, i.e. BigTIFF's.
pub const IFD_MAX_INLINE_VALUE_BYTES: usize = 8;

/// Which TIFF dialect a file is written in.
///
/// BigTIFF is the same format with every file pointer and count widened to 64
/// bits: GDAL's COG driver switches to it automatically once the output would
/// pass 4 GB (`BIGTIFF=IF_NEEDED`), which is routine for a national-scale
/// mosaic. Everything above this module already works in `u64`, so the whole
/// difference is confined to the four widths below.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TiffVariant {
    /// Classic TIFF (magic 42): 4-byte offsets, 12-byte entries.
    #[default]
    Classic,
    /// BigTIFF (magic 43): 8-byte offsets, 20-byte entries.
    Big,
}

impl TiffVariant {
    /// Bytes one IFD entry occupies.
    #[must_use]
    pub const fn entry_bytes(self) -> usize {
        match self {
            Self::Classic => IFD_ENTRY_BYTES,
            Self::Big => 20,
        }
    }

    /// Bytes of an entry's inline value field.
    #[must_use]
    pub const fn inline_value_bytes(self) -> usize {
        match self {
            Self::Classic => IFD_INLINE_VALUE_BYTES,
            Self::Big => IFD_MAX_INLINE_VALUE_BYTES,
        }
    }

    /// Bytes of an IFD's leading entry count.
    #[must_use]
    pub const fn entry_count_bytes(self) -> usize {
        match self {
            Self::Classic => 2,
            Self::Big => 8,
        }
    }

    /// Bytes of a file offset (the first-IFD and next-IFD pointers).
    #[must_use]
    pub const fn offset_bytes(self) -> usize {
        match self {
            Self::Classic => 4,
            Self::Big => 8,
        }
    }

    /// Bytes of the file header, up to and including the first-IFD offset.
    #[must_use]
    pub const fn header_bytes(self) -> usize {
        match self {
            Self::Classic => 8,
            Self::Big => 16,
        }
    }
}

/// Which end of a multi-byte TIFF field comes first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// `II` — least significant byte first.
    LittleEndian,
    /// `MM` — most significant byte first.
    BigEndian,
}

impl ByteOrder {
    /// Whether this is [`ByteOrder::LittleEndian`].
    #[must_use]
    pub const fn is_little_endian(self) -> bool {
        matches!(self, Self::LittleEndian)
    }

    /// Reads a 16-bit field from the first two bytes of `bytes`.
    #[must_use]
    pub fn short(self, bytes: &[u8]) -> Option<u16> {
        let raw: [u8; 2] = bytes.get(..2)?.try_into().ok()?;
        Some(match self {
            Self::LittleEndian => u16::from_le_bytes(raw),
            Self::BigEndian => u16::from_be_bytes(raw),
        })
    }

    /// Reads a 32-bit field from the first four bytes of `bytes`.
    #[must_use]
    pub fn long(self, bytes: &[u8]) -> Option<u32> {
        let raw: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
        Some(match self {
            Self::LittleEndian => u32::from_le_bytes(raw),
            Self::BigEndian => u32::from_be_bytes(raw),
        })
    }

    /// Reads a 64-bit field (BigTIFF `LONG8`, and every BigTIFF offset) from
    /// the first eight bytes of `bytes`.
    #[must_use]
    pub fn long8(self, bytes: &[u8]) -> Option<u64> {
        let raw: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
        Some(match self {
            Self::LittleEndian => u64::from_le_bytes(raw),
            Self::BigEndian => u64::from_be_bytes(raw),
        })
    }

    /// Reads an IEEE-754 double from the first eight bytes of `bytes`.
    #[must_use]
    pub fn double(self, bytes: &[u8]) -> Option<f64> {
        let raw: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
        Some(match self {
            Self::LittleEndian => f64::from_le_bytes(raw),
            Self::BigEndian => f64::from_be_bytes(raw),
        })
    }
}

/// The TIFF file header: eight bytes for classic TIFF, sixteen for BigTIFF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiffHeader {
    /// Byte order every subsequent field is encoded in.
    pub order: ByteOrder,
    /// Which dialect the rest of the file is written in.
    pub variant: TiffVariant,
    /// File offset of the first (full-resolution) IFD.
    pub first_ifd: u64,
}

/// Magic number of a classic TIFF.
const MAGIC_CLASSIC: u16 = 42;

/// Magic number of a BigTIFF.
const MAGIC_BIGTIFF: u16 = 43;

/// The only offset width BigTIFF defines.
const BIGTIFF_OFFSET_SIZE: u16 = 8;

impl TiffHeader {
    /// Parses the header out of the first bytes of a file.
    ///
    /// A BigTIFF header is `MM`/`II`, magic 43, the offset size (always 8), a
    /// zero pad, then an 8-byte first-IFD offset.
    ///
    /// # Errors
    ///
    /// * [`RenderError::Decode`] if `bytes` is too short or the byte-order mark
    ///   is neither `II` nor `MM`.
    /// * [`RenderError::Unsupported`] for a BigTIFF declaring an offset width
    ///   other than 8, which no writer emits and the format does not define.
    pub fn parse(bytes: &[u8]) -> Result<Self, RenderError> {
        let mark = bytes
            .get(..2)
            .ok_or_else(|| RenderError::Decode("truncated TIFF header".to_owned()))?;
        let order = match mark {
            b"II" => ByteOrder::LittleEndian,
            b"MM" => ByteOrder::BigEndian,
            other => {
                return Err(RenderError::Decode(format!(
                    "not a TIFF: byte-order mark {other:?} is neither II nor MM"
                )));
            }
        };
        let magic = bytes
            .get(2..4)
            .and_then(|raw| order.short(raw))
            .ok_or_else(|| RenderError::Decode("truncated TIFF magic number".to_owned()))?;
        match magic {
            MAGIC_CLASSIC => {
                let first_ifd = bytes
                    .get(4..8)
                    .and_then(|raw| order.long(raw))
                    .ok_or_else(|| RenderError::Decode("truncated first-IFD offset".to_owned()))?;
                Ok(Self {
                    order,
                    variant: TiffVariant::Classic,
                    first_ifd: u64::from(first_ifd),
                })
            }
            MAGIC_BIGTIFF => {
                let offset_size = bytes
                    .get(4..6)
                    .and_then(|raw| order.short(raw))
                    .ok_or_else(|| {
                        RenderError::Decode("truncated BigTIFF offset size".to_owned())
                    })?;
                if offset_size != BIGTIFF_OFFSET_SIZE {
                    return Err(RenderError::Unsupported(format!(
                        "BigTIFF declares {offset_size}-byte offsets; only {BIGTIFF_OFFSET_SIZE} \
                         is defined"
                    )));
                }
                let first_ifd = bytes
                    .get(8..16)
                    .and_then(|raw| order.long8(raw))
                    .ok_or_else(|| RenderError::Decode("truncated first-IFD offset".to_owned()))?;
                Ok(Self {
                    order,
                    variant: TiffVariant::Big,
                    first_ifd,
                })
            }
            other => Err(RenderError::Decode(format!(
                "not a TIFF: magic number {other} is neither {MAGIC_CLASSIC} nor {MAGIC_BIGTIFF}"
            ))),
        }
    }
}

/// One raw directory entry: what it is, how many values, and where they live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IfdEntry {
    /// TIFF tag number.
    pub tag: u16,
    /// TIFF field type (3 = SHORT, 4 = LONG, 12 = DOUBLE, 16 = LONG8, …).
    pub field_type: u16,
    /// Number of values of `field_type`.
    pub count: u64,
    /// The entry's inline value bytes: the values themselves when they fit,
    /// otherwise the file offset they are stored at. Only the first
    /// [`TiffVariant::inline_value_bytes`] are meaningful; the rest are zero.
    pub value: [u8; IFD_MAX_INLINE_VALUE_BYTES],
    /// Which dialect the entry was read from, i.e. how wide its fields are.
    pub variant: TiffVariant,
}

/// Byte width of one value of `field_type`, or `None` for types this reader
/// has no use for.
#[must_use]
pub const fn field_type_size(field_type: u16) -> Option<usize> {
    match field_type {
        // BYTE, ASCII, SBYTE, UNDEFINED
        1 | 2 | 6 | 7 => Some(1),
        // SHORT, SSHORT
        3 | 8 => Some(2),
        // LONG, SLONG, FLOAT
        4 | 9 | 11 => Some(4),
        // RATIONAL, SRATIONAL, DOUBLE, and BigTIFF's LONG8, SLONG8, IFD8
        5 | 10 | 12 | 16..=18 => Some(8),
        _ => None,
    }
}

impl IfdEntry {
    /// Parses one entry from exactly [`TiffVariant::entry_bytes`] bytes.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Decode`] if `bytes` is too short.
    pub fn parse(
        bytes: &[u8],
        order: ByteOrder,
        variant: TiffVariant,
    ) -> Result<Self, RenderError> {
        let short_entry = || RenderError::Decode("truncated IFD entry".to_owned());
        let tag = bytes
            .get(0..2)
            .and_then(|b| order.short(b))
            .ok_or_else(short_entry)?;
        let field_type = bytes
            .get(2..4)
            .and_then(|b| order.short(b))
            .ok_or_else(short_entry)?;
        let count = match variant {
            TiffVariant::Classic => bytes
                .get(4..8)
                .and_then(|b| order.long(b))
                .map(u64::from)
                .ok_or_else(short_entry)?,
            TiffVariant::Big => bytes
                .get(4..12)
                .and_then(|b| order.long8(b))
                .ok_or_else(short_entry)?,
        };
        let inline = variant.inline_value_bytes();
        let raw = bytes
            .get(variant.entry_bytes() - inline..variant.entry_bytes())
            .ok_or_else(short_entry)?;
        let mut value = [0u8; IFD_MAX_INLINE_VALUE_BYTES];
        value
            .get_mut(..inline)
            .ok_or_else(short_entry)?
            .copy_from_slice(raw);
        Ok(Self {
            tag,
            field_type,
            count,
            value,
            variant,
        })
    }

    /// Total byte size of this entry's values, or `None` for an unknown field
    /// type or a count that overflows `usize`.
    #[must_use]
    pub fn values_bytes(&self) -> Option<usize> {
        field_type_size(self.field_type)?.checked_mul(usize::try_from(self.count).ok()?)
    }

    /// Whether the values are stored inline in [`IfdEntry::value`] rather than
    /// at the offset it holds.
    #[must_use]
    pub fn is_inline(&self) -> bool {
        self.values_bytes()
            .is_some_and(|bytes| bytes <= self.variant.inline_value_bytes())
    }

    /// File offset the values are stored at, for a non-inline entry.
    #[must_use]
    pub fn value_offset(&self, order: ByteOrder) -> Option<u64> {
        match self.variant {
            TiffVariant::Classic => order.long(&self.value).map(u64::from),
            TiffVariant::Big => order.long8(&self.value),
        }
    }

    /// Decodes the entry's values from `data`, the bytes holding them.
    ///
    /// `data` is either [`IfdEntry::value`] (inline) or the bytes read from
    /// [`IfdEntry::value_offset`]. Integer field types are widened to `u64`;
    /// other types yield an empty vector, which callers treat as "tag absent".
    #[must_use]
    pub fn integers(&self, data: &[u8], order: ByteOrder) -> Vec<u64> {
        let Some(size) = field_type_size(self.field_type) else {
            return Vec::new();
        };
        // `count` is a raw file field; the loop stops at `data`'s end anyway, so
        // reserving past what `data` can yield is a malformed file's allocation.
        let reachable = data.len() / size;
        let count = usize::try_from(self.count).unwrap_or(usize::MAX);
        let mut values = Vec::with_capacity(count.min(reachable));
        for index in 0..count {
            let Some(chunk) = data.get(index * size..index * size + size) else {
                break;
            };
            let value = match self.field_type {
                1 | 2 | 6 | 7 => chunk.first().map(|b| u64::from(*b)),
                3 | 8 => order.short(chunk).map(u64::from),
                4 | 9 => order.long(chunk).map(u64::from),
                16..=18 => order.long8(chunk),
                _ => None,
            };
            let Some(value) = value else { break };
            values.push(value);
        }
        values
    }

    /// Decodes the entry's values as IEEE doubles (TIFF field type 12).
    ///
    /// Any other field type yields an empty vector.
    #[must_use]
    pub fn doubles(&self, data: &[u8], order: ByteOrder) -> Vec<f64> {
        if self.field_type != 12 {
            return Vec::new();
        }
        let count = usize::try_from(self.count).unwrap_or(usize::MAX);
        let mut values = Vec::with_capacity(count.min(data.len() / 8));
        for index in 0..count {
            let Some(chunk) = data.get(index * 8..index * 8 + 8) else {
                break;
            };
            let Some(value) = order.double(chunk) else {
                break;
            };
            values.push(value);
        }
        values
    }
}

/// Finds the EPSG code declared by a `GeoKeyDirectory` value array.
///
/// The directory is `[version, rev_major, rev_minor, key_count]` followed by
/// `key_count` quadruples `[key_id, tiff_tag_location, count, value_offset]`.
/// Only inline keys (`tiff_tag_location == 0`) can carry an EPSG code, and
/// [`GEOKEY_USER_DEFINED`] means the file declines to name one.
///
/// A `key_count` that runs past the end of `directory` — a short read, or an
/// inflated count — stops the scan rather than discarding it: keys already read
/// still name the CRS, and losing them would report a damaged tag as an
/// unsupported file.
///
/// Ported from `oxigeo-wasm`'s `parse_epsg_from_geokeys`.
#[must_use]
pub fn epsg_from_geo_keys(directory: &[u64]) -> Option<u32> {
    let key_count = usize::try_from(*directory.get(3)?).ok()?;
    let mut projected = None;
    let mut geographic = None;
    for index in 0..key_count {
        let base = 4 + index * 4;
        let (Some(raw_key), Some(raw_location), Some(raw_value)) = (
            directory.get(base),
            directory.get(base + 1),
            directory.get(base + 3),
        ) else {
            break;
        };
        let Ok(key_id) = u16::try_from(*raw_key) else {
            continue;
        };
        let location = *raw_location;
        let Ok(value) = u16::try_from(*raw_value) else {
            continue;
        };
        if location != 0 || value == GEOKEY_USER_DEFINED {
            continue;
        }
        match key_id {
            GEOKEY_PROJECTED_CS_TYPE => projected = Some(u32::from(value)),
            GEOKEY_GEOGRAPHIC_TYPE => geographic = Some(u32::from(value)),
            _ => {}
        }
    }
    // A projected file also carries GeographicType (its datum); the projected
    // code is the one that describes the coordinates in the geotransform.
    projected.or(geographic)
}

#[cfg(test)]
mod tests {
    use super::{
        ByteOrder, IFD_ENTRY_BYTES, IfdEntry, TiffHeader, TiffVariant, epsg_from_geo_keys,
        field_type_size,
    };
    use crate::error::RenderError;

    #[test]
    fn byte_order_reads_both_ends() {
        assert_eq!(ByteOrder::LittleEndian.short(&[0x01, 0x02]), Some(0x0201));
        assert_eq!(ByteOrder::BigEndian.short(&[0x01, 0x02]), Some(0x0102));
        assert_eq!(
            ByteOrder::LittleEndian.long(&[0x01, 0x00, 0x00, 0x00]),
            Some(1)
        );
        assert_eq!(
            ByteOrder::BigEndian.long(&[0x00, 0x00, 0x00, 0x01]),
            Some(1)
        );
        assert_eq!(
            ByteOrder::LittleEndian.double(&1.5f64.to_le_bytes()),
            Some(1.5)
        );
        assert_eq!(
            ByteOrder::BigEndian.double(&1.5f64.to_be_bytes()),
            Some(1.5)
        );
        assert!(ByteOrder::LittleEndian.is_little_endian());
        assert!(!ByteOrder::BigEndian.is_little_endian());
    }

    #[test]
    fn byte_order_refuses_short_input() {
        assert_eq!(ByteOrder::LittleEndian.short(&[0x01]), None);
        assert_eq!(ByteOrder::LittleEndian.long(&[0x01, 0x02]), None);
        assert_eq!(ByteOrder::LittleEndian.long8(&[0x01, 0x02]), None);
        assert_eq!(ByteOrder::LittleEndian.double(&[0x01, 0x02]), None);
    }

    #[test]
    fn byte_order_reads_sixty_four_bit_fields() {
        assert_eq!(ByteOrder::LittleEndian.long8(&1u64.to_le_bytes()), Some(1));
        assert_eq!(
            ByteOrder::BigEndian.long8(&0x0102_0304_0506_0708u64.to_be_bytes()),
            Some(0x0102_0304_0506_0708)
        );
    }

    #[test]
    fn header_parses_little_and_big_endian() {
        let little = [b'I', b'I', 42, 0, 0x08, 0, 0, 0];
        let parsed = TiffHeader::parse(&little).expect("valid II header");
        assert_eq!(parsed.order, ByteOrder::LittleEndian);
        assert_eq!(parsed.variant, TiffVariant::Classic);
        assert_eq!(parsed.first_ifd, 8);

        let big = [b'M', b'M', 0, 42, 0, 0, 0, 0x10];
        let parsed = TiffHeader::parse(&big).expect("valid MM header");
        assert_eq!(parsed.order, ByteOrder::BigEndian);
        assert_eq!(parsed.first_ifd, 16);
    }

    #[test]
    fn a_bigtiff_header_carries_a_sixty_four_bit_first_ifd_offset() {
        // II, magic 43, offset size 8, pad 0, then an 8-byte offset past 4 GB.
        let mut header = vec![b'I', b'I', 43, 0, 8, 0, 0, 0];
        header.extend_from_slice(&5_000_000_000u64.to_le_bytes());
        let parsed = TiffHeader::parse(&header).expect("a valid BigTIFF header");
        assert_eq!(parsed.variant, TiffVariant::Big);
        assert_eq!(parsed.first_ifd, 5_000_000_000);
        assert_eq!(parsed.variant.entry_bytes(), 20);
        assert_eq!(parsed.variant.inline_value_bytes(), 8);
        assert_eq!(parsed.variant.entry_count_bytes(), 8);
        assert_eq!(parsed.variant.offset_bytes(), 8);
        assert_eq!(parsed.variant.header_bytes(), 16);

        // MM, and the same offset the other way round.
        let mut header = vec![b'M', b'M', 0, 43, 0, 8, 0, 0];
        header.extend_from_slice(&5_000_000_000u64.to_be_bytes());
        let parsed = TiffHeader::parse(&header).expect("a valid MM BigTIFF header");
        assert_eq!(parsed.order, ByteOrder::BigEndian);
        assert_eq!(parsed.first_ifd, 5_000_000_000);

        // A truncated BigTIFF header is short, not classic.
        assert!(TiffHeader::parse(&[b'I', b'I', 43, 0, 8, 0, 0, 0]).is_err());
    }

    #[test]
    fn header_rejects_an_undefined_offset_width_and_garbage() {
        // BigTIFF defines exactly one offset size; 16 is not a thing.
        let mut wide = vec![b'I', b'I', 43, 0, 16, 0, 0, 0];
        wide.extend_from_slice(&16u64.to_le_bytes());
        assert!(matches!(
            TiffHeader::parse(&wide),
            Err(RenderError::Unsupported(_))
        ));
        let png = [0x89, b'P', b'N', b'G', 0, 0, 0, 0];
        assert!(matches!(
            TiffHeader::parse(&png),
            Err(RenderError::Decode(_))
        ));
        let wrong_magic = [b'I', b'I', 7, 0, 8, 0, 0, 0];
        assert!(matches!(
            TiffHeader::parse(&wrong_magic),
            Err(RenderError::Decode(_))
        ));
        assert!(TiffHeader::parse(b"I").is_err());
        assert!(TiffHeader::parse(&[b'I', b'I', 42]).is_err());
        assert!(TiffHeader::parse(&[b'I', b'I', 42, 0, 8]).is_err());
    }

    #[test]
    fn field_type_sizes_cover_the_tiff_set() {
        assert_eq!(field_type_size(1), Some(1));
        assert_eq!(field_type_size(3), Some(2));
        assert_eq!(field_type_size(4), Some(4));
        assert_eq!(field_type_size(12), Some(8));
        assert_eq!(field_type_size(99), None);
    }

    /// Builds one 12-byte IFD entry, little-endian.
    fn entry(tag: u16, field_type: u16, count: u32, value: [u8; 4]) -> Vec<u8> {
        let mut out = Vec::with_capacity(IFD_ENTRY_BYTES);
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&field_type.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&value);
        out
    }

    #[test]
    fn entry_parses_and_classifies_inline_values() {
        let bytes = entry(256, 3, 1, [0x00, 0x04, 0, 0]);
        let parsed = IfdEntry::parse(&bytes, ByteOrder::LittleEndian, TiffVariant::Classic)
            .expect("valid entry");
        assert_eq!(parsed.tag, 256);
        assert_eq!(parsed.field_type, 3);
        assert_eq!(parsed.count, 1);
        assert!(parsed.is_inline());
        assert_eq!(parsed.values_bytes(), Some(2));
        assert_eq!(
            parsed.integers(&parsed.value, ByteOrder::LittleEndian),
            vec![1024]
        );

        // Three SHORTs are six bytes: too long to be inline.
        let bytes = entry(258, 3, 3, [0x40, 0x00, 0, 0]);
        let parsed = IfdEntry::parse(&bytes, ByteOrder::LittleEndian, TiffVariant::Classic)
            .expect("valid entry");
        assert!(!parsed.is_inline());
        assert_eq!(parsed.values_bytes(), Some(6));
        assert_eq!(parsed.value_offset(ByteOrder::LittleEndian), Some(0x40));
    }

    #[test]
    fn entry_decodes_integer_and_double_arrays() {
        let bytes = entry(324, 4, 2, [0, 0, 0, 0]);
        let parsed = IfdEntry::parse(&bytes, ByteOrder::LittleEndian, TiffVariant::Classic)
            .expect("valid entry");
        let mut data = Vec::new();
        data.extend_from_slice(&1000u32.to_le_bytes());
        data.extend_from_slice(&2000u32.to_le_bytes());
        assert_eq!(
            parsed.integers(&data, ByteOrder::LittleEndian),
            vec![1000, 2000]
        );
        // A short payload stops early rather than panicking.
        assert_eq!(
            parsed.integers(&data[..5], ByteOrder::LittleEndian),
            vec![1000]
        );
        assert!(parsed.doubles(&data, ByteOrder::LittleEndian).is_empty());

        let bytes = entry(33550, 12, 2, [0, 0, 0, 0]);
        let parsed = IfdEntry::parse(&bytes, ByteOrder::LittleEndian, TiffVariant::Classic)
            .expect("valid entry");
        let mut data = Vec::new();
        data.extend_from_slice(&10.0f64.to_le_bytes());
        data.extend_from_slice(&(-20.0f64).to_le_bytes());
        assert_eq!(
            parsed.doubles(&data, ByteOrder::LittleEndian),
            vec![10.0, -20.0]
        );
        assert!(parsed.integers(&data, ByteOrder::LittleEndian).is_empty());
    }

    #[test]
    fn a_bigtiff_entry_has_a_wide_count_and_an_eight_byte_inline_field() {
        // tag 324, type LONG8, two values — sixteen bytes, so out of line.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&324u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&5_000_000_000u64.to_le_bytes());
        let parsed = IfdEntry::parse(&bytes, ByteOrder::LittleEndian, TiffVariant::Big)
            .expect("a valid BigTIFF entry");
        assert_eq!(parsed.tag, 324);
        assert_eq!(parsed.count, 2);
        assert!(!parsed.is_inline());
        assert_eq!(parsed.values_bytes(), Some(16));
        assert_eq!(
            parsed.value_offset(ByteOrder::LittleEndian),
            Some(5_000_000_000)
        );

        // One LONG8 is eight bytes, which *does* fit BigTIFF's inline field —
        // it would not fit classic TIFF's.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&257u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&9_000_000_000u64.to_le_bytes());
        let parsed = IfdEntry::parse(&bytes, ByteOrder::LittleEndian, TiffVariant::Big)
            .expect("a valid BigTIFF entry");
        assert!(parsed.is_inline());
        assert_eq!(
            parsed.integers(&parsed.value, ByteOrder::LittleEndian),
            vec![9_000_000_000]
        );
        assert_eq!(field_type_size(16), Some(8));
        assert_eq!(field_type_size(18), Some(8));

        // A 19-byte entry is truncated for BigTIFF but long enough for classic.
        assert!(IfdEntry::parse(&[0u8; 19], ByteOrder::LittleEndian, TiffVariant::Big).is_err());
        assert!(IfdEntry::parse(&[0u8; 19], ByteOrder::LittleEndian, TiffVariant::Classic).is_ok());
    }

    #[test]
    fn entry_rejects_truncation() {
        assert!(
            IfdEntry::parse(&[0u8; 11], ByteOrder::LittleEndian, TiffVariant::Classic).is_err()
        );
    }

    #[test]
    fn geo_keys_prefer_the_projected_code() {
        // version, rev, rev, key_count=2, then two inline keys.
        let directory = vec![1, 1, 0, 2, 2048, 0, 1, 4326, 3072, 0, 1, 3857];
        assert_eq!(epsg_from_geo_keys(&directory), Some(3857));

        let geographic_only = vec![1, 1, 0, 1, 2048, 0, 1, 4326];
        assert_eq!(epsg_from_geo_keys(&geographic_only), Some(4326));
    }

    #[test]
    fn geo_keys_ignore_user_defined_and_out_of_line_keys() {
        let user_defined = vec![1, 1, 0, 1, 3072, 0, 1, 32767];
        assert_eq!(epsg_from_geo_keys(&user_defined), None);
        // tiff_tag_location != 0 means the value lives in another tag.
        let out_of_line = vec![1, 1, 0, 1, 3072, 34737, 1, 3857];
        assert_eq!(epsg_from_geo_keys(&out_of_line), None);
        assert_eq!(epsg_from_geo_keys(&[]), None);
    }

    #[test]
    fn a_truncated_geo_key_directory_keeps_the_keys_it_read() {
        // key_count claims nine keys; only the first is present, and it names a
        // perfectly usable EPSG code. Discarding it would report a damaged tag
        // as "COG declares no EPSG code".
        assert_eq!(
            epsg_from_geo_keys(&[1, 1, 0, 9, 3072, 0, 1, 3857]),
            Some(3857)
        );
        // A quadruple cut in half is still a clean stop.
        assert_eq!(
            epsg_from_geo_keys(&[1, 1, 0, 9, 2048, 0, 1, 4326, 3072, 0]),
            Some(4326)
        );
    }
}
