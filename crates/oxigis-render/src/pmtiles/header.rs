//! The fixed 127-byte PMTiles v3 header.
//!
//! Every multi-byte field is little-endian. The four bounding-box fields and
//! the two centre fields are `i32` degrees × 10⁷, which is why they are stored
//! as integers here and only converted to `f64` degrees on the way out —
//! `-85.0511287` is exact in the integer form and is not in the float one.
//!
//! The layout was decoded from real archives before it was read in the spec,
//! and the two agree field for field. The independent arithmetic check that
//! the `u64` offsets are right: a planet build's `addressed_tiles` decodes to
//! `1_431_655_765`, which is exactly `(4^16 − 1)/3`, the number of tiles from
//! z0 through z15 — the archive's declared zoom range.

use crate::pmtiles::PmtilesError;
use crate::source::ByteRange;

/// The seven bytes every PMTiles archive begins with.
pub const PMTILES_MAGIC: &[u8; 7] = b"PMTiles";

/// The only spec version this reader implements.
pub const PMTILES_VERSION: u8 = 3;

/// Size of the header, in bytes. Fixed by the spec; never varies.
pub const HEADER_LEN: usize = 127;

/// How many bytes to read speculatively when opening an archive.
///
/// PMTiles v3 requires the header *and* the root directory to be contained in
/// the first 16 384 bytes, so one read of this size opens any conforming
/// archive in a single round trip — the same trick [`crate::cog::CogOpen`]
/// plays with its header block. Measured against a 137 GB planet build, whose
/// root directory ends at byte 15 691.
pub const PREFETCH_LEN: u64 = 16_384;

/// Highest zoom level a header may declare.
///
/// The tile-id curve runs to zoom 31 ([`crate::pmtiles::MAX_TILE_ID_ZOOM`]);
/// 30 is where the spec's own guidance stops and is already four levels past
/// anything this renderer draws, so a larger `max_zoom` is a corrupt header
/// rather than an exotic archive.
pub const MAX_PMTILES_ZOOM: u8 = 30;

/// Hard cap on the byte length of the metadata block this reader will fetch.
///
/// Every other attacker-controlled length in this reader is capped —
/// [`crate::pmtiles::archive::MAX_DIRECTORY_BYTES`],
/// [`crate::pmtiles::archive::MAX_ARCHIVE_TILE_BYTES`],
/// [`crate::pmtiles::directory::MAX_DIRECTORY_INFLATED_BYTES`] — but
/// `metadata_length` was only overflow-checked, so a header declaring it near
/// `u64::MAX` turned [`PmtilesHeader::metadata_range`] into a multi-terabyte
/// request. 16 MiB matches the inflated-directory cap: metadata is
/// comparably sized JSON (a name, an attribution string, a `vector_layers`
/// list), never a payload that should legitimately run past it.
pub const MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;

/// Scale factor of the header's coordinate fields: degrees × 10⁷.
const COORD_SCALE_E7: f64 = 1e7;

/// How the archive's directories, metadata or tile bodies are coded.
///
/// `internal_compression` (directories + metadata) and `tile_compression`
/// (tile bodies) are **independent**: a measured raster archive carries gzip
/// directories and uncompressed PNG tiles. Honour the bytes, never sniff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compression {
    /// `0` — the writer did not say. The caller decides what to do.
    Unknown,
    /// `1` — stored verbatim.
    None,
    /// `2` — gzip (RFC 1952).
    Gzip,
    /// `3` — brotli. Refused: the codec is banned by `deny.toml`.
    Brotli,
    /// `4` — zstd. Refused: the codec is banned by `deny.toml`.
    Zstd,
}

impl Compression {
    /// Decodes the header byte.
    ///
    /// # Errors
    ///
    /// Returns [`PmtilesError::UnknownEnum`] for a value outside `0..=4`.
    pub fn from_byte(field: &'static str, value: u8) -> Result<Self, PmtilesError> {
        match value {
            0 => Ok(Self::Unknown),
            1 => Ok(Self::None),
            2 => Ok(Self::Gzip),
            3 => Ok(Self::Brotli),
            4 => Ok(Self::Zstd),
            _ => Err(PmtilesError::UnknownEnum { field, value }),
        }
    }

    /// The byte this variant is written as.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::None => 1,
            Self::Gzip => 2,
            Self::Brotli => 3,
            Self::Zstd => 4,
        }
    }

    /// A lowercase name for status lines and error messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unknown => "an unknown compression",
            Self::None => "no compression",
            Self::Gzip => "gzip",
            Self::Brotli => "brotli",
            Self::Zstd => "zstd",
        }
    }

    /// Whether this build refuses the codec outright.
    ///
    /// Brotli and zstd are banned by the workspace's `deny.toml`, so an
    /// archive using either is refused by name once, at open.
    #[must_use]
    pub const fn is_refused(self) -> bool {
        matches!(self, Self::Brotli | Self::Zstd)
    }
}

/// What the archive's tile bodies are.
///
/// Routing a tile type to a raster or a vector provider is the *caller's*
/// decision — this crate only reports what the header says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileType {
    /// `0` — the writer did not say.
    Unknown,
    /// `1` — Mapbox Vector Tile.
    Mvt,
    /// `2` — PNG.
    Png,
    /// `3` — JPEG.
    Jpeg,
    /// `4` — WebP.
    Webp,
    /// `5` — AVIF.
    Avif,
}

impl TileType {
    /// Decodes the header byte.
    ///
    /// # Errors
    ///
    /// Returns [`PmtilesError::UnknownEnum`] for a value outside `0..=5`.
    pub fn from_byte(field: &'static str, value: u8) -> Result<Self, PmtilesError> {
        match value {
            0 => Ok(Self::Unknown),
            1 => Ok(Self::Mvt),
            2 => Ok(Self::Png),
            3 => Ok(Self::Jpeg),
            4 => Ok(Self::Webp),
            5 => Ok(Self::Avif),
            _ => Err(PmtilesError::UnknownEnum { field, value }),
        }
    }

    /// The byte this variant is written as.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Mvt => 1,
            Self::Png => 2,
            Self::Jpeg => 3,
            Self::Webp => 4,
            Self::Avif => 5,
        }
    }

    /// A lowercase name for status lines and error messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unknown => "an unknown tile type",
            Self::Mvt => "MVT",
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Webp => "WebP",
            Self::Avif => "AVIF",
        }
    }

    /// Whether the bodies are vector tiles rather than images.
    #[must_use]
    pub const fn is_vector(self) -> bool {
        matches!(self, Self::Mvt)
    }
}

/// The parsed 127-byte header.
///
/// Field comments carry the byte offset each one occupies, because that
/// mapping is the whole content of this struct and getting one offset wrong
/// silently shifts every field after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PmtilesHeader {
    /// Root directory, `8..16` offset + `16..24` length.
    ///
    /// Guaranteed by [`PmtilesHeader::parse`] to be non-empty and to end at or
    /// before [`PREFETCH_LEN`].
    pub root: ByteRange,
    /// Metadata block offset, `24..32`.
    pub metadata_offset: u64,
    /// Metadata block length, `32..40`. `0` means the archive has none.
    pub metadata_length: u64,
    /// Start of the leaf-directory region, `40..48`.
    ///
    /// **This is the base a leaf entry's `offset` is relative to** — not the
    /// start of the file, and not [`PmtilesHeader::tile_data_offset`].
    pub leaf_dirs_offset: u64,
    /// Length of the leaf-directory region, `48..56`. `0` = root-only archive.
    pub leaf_dirs_len: u64,
    /// Start of the tile-data region, `56..64`.
    ///
    /// **This is the base a tile entry's `offset` is relative to.**
    pub tile_data_offset: u64,
    /// Length of the tile-data region, `64..72`.
    pub tile_data_len: u64,
    /// Number of `(z, x, y)` addresses the archive answers, `72..80`.
    /// `0` means the writer did not count.
    pub addressed_tiles: u64,
    /// Number of directory entries across all directories, `80..88`.
    pub tile_entries: u64,
    /// Number of distinct tile bodies, `88..96`.
    pub tile_contents: u64,
    /// Whether tile bodies are stored in tile-id order, `96`.
    pub clustered: bool,
    /// Codec of the directories and the metadata block, `97`.
    pub internal_compression: Compression,
    /// Codec of the tile bodies, `98`.
    pub tile_compression: Compression,
    /// What the tile bodies are, `99`.
    pub tile_type: TileType,
    /// Lowest zoom the archive holds, `100`.
    pub min_zoom: u8,
    /// Highest zoom the archive holds, `101`.
    pub max_zoom: u8,
    /// Western edge, `102..106`, in degrees × 10⁷.
    pub min_lon_e7: i32,
    /// Southern edge, `106..110`, in degrees × 10⁷.
    pub min_lat_e7: i32,
    /// Eastern edge, `110..114`, in degrees × 10⁷.
    pub max_lon_e7: i32,
    /// Northern edge, `114..118`, in degrees × 10⁷.
    pub max_lat_e7: i32,
    /// Suggested opening zoom, `118`.
    pub center_zoom: u8,
    /// Suggested centre longitude, `119..123`, in degrees × 10⁷.
    pub center_lon_e7: i32,
    /// Suggested centre latitude, `123..127`, in degrees × 10⁷.
    pub center_lat_e7: i32,
}

impl PmtilesHeader {
    /// Parses a header from the first [`HEADER_LEN`] bytes of an archive.
    ///
    /// `bytes` may be longer — the speculative prefetch is handed straight in.
    ///
    /// # Errors
    ///
    /// * [`PmtilesError::Truncated`] if fewer than [`HEADER_LEN`] bytes.
    /// * [`PmtilesError::BadMagic`] / [`PmtilesError::UnsupportedVersion`].
    /// * [`PmtilesError::UnknownEnum`] for a compression or tile-type byte
    ///   outside the spec's range.
    /// * [`PmtilesError::RootOutsidePrefetch`] if the root directory is not
    ///   inside the first [`PREFETCH_LEN`] bytes, which the spec mandates and
    ///   which is what makes a one-round-trip open sound.
    /// * [`PmtilesError::InvalidHeader`] for a degenerate root directory, a
    ///   region whose offset plus length overflows `u64`, a metadata block
    ///   past [`MAX_METADATA_BYTES`], or a zoom range that is inverted or past
    ///   [`MAX_PMTILES_ZOOM`].
    pub fn parse(bytes: &[u8]) -> Result<Self, PmtilesError> {
        let available = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        let Some(head) = bytes.get(..HEADER_LEN) else {
            return Err(PmtilesError::Truncated {
                context: "header",
                needed: u64::try_from(HEADER_LEN).unwrap_or(u64::MAX),
                available,
            });
        };
        if head.get(..7) != Some(&PMTILES_MAGIC[..]) {
            return Err(PmtilesError::BadMagic);
        }
        let version = byte(head, 7)?;
        if version != PMTILES_VERSION {
            return Err(PmtilesError::UnsupportedVersion(version));
        }

        let root_offset = le_u64(head, 8)?;
        let root_length = le_u64(head, 16)?;
        let root = region("root directory", root_offset, root_length)?;
        if root.end > PREFETCH_LEN {
            return Err(PmtilesError::RootOutsidePrefetch {
                end: root.end,
                limit: PREFETCH_LEN,
            });
        }
        if root.start < u64::try_from(HEADER_LEN).unwrap_or(u64::MAX) {
            return Err(PmtilesError::InvalidHeader {
                field: "root directory offset",
                reason: "the root directory overlaps the header",
            });
        }

        let metadata_offset = le_u64(head, 24)?;
        let metadata_length = le_u64(head, 32)?;
        if metadata_offset.checked_add(metadata_length).is_none() {
            return Err(PmtilesError::InvalidHeader {
                field: "metadata",
                reason: "offset plus length overflows u64",
            });
        }
        if metadata_length > MAX_METADATA_BYTES {
            return Err(PmtilesError::InvalidHeader {
                field: "metadata",
                reason: "the metadata block is past the size limit",
            });
        }

        let leaf_dirs_offset = le_u64(head, 40)?;
        let leaf_dirs_len = le_u64(head, 48)?;
        if leaf_dirs_offset.checked_add(leaf_dirs_len).is_none() {
            return Err(PmtilesError::InvalidHeader {
                field: "leaf directories",
                reason: "offset plus length overflows u64",
            });
        }

        let tile_data_offset = le_u64(head, 56)?;
        let tile_data_len = le_u64(head, 64)?;
        if tile_data_offset.checked_add(tile_data_len).is_none() {
            return Err(PmtilesError::InvalidHeader {
                field: "tile data",
                reason: "offset plus length overflows u64",
            });
        }

        let min_zoom = byte(head, 100)?;
        let max_zoom = byte(head, 101)?;
        if min_zoom > max_zoom {
            return Err(PmtilesError::InvalidHeader {
                field: "zoom range",
                reason: "min_zoom is greater than max_zoom",
            });
        }
        if max_zoom > MAX_PMTILES_ZOOM {
            return Err(PmtilesError::InvalidHeader {
                field: "max_zoom",
                reason: "max_zoom is past zoom 30",
            });
        }

        Ok(Self {
            root,
            metadata_offset,
            metadata_length,
            leaf_dirs_offset,
            leaf_dirs_len,
            tile_data_offset,
            tile_data_len,
            addressed_tiles: le_u64(head, 72)?,
            tile_entries: le_u64(head, 80)?,
            tile_contents: le_u64(head, 88)?,
            // The spec writes 0 or 1; anything non-zero is read as "clustered"
            // rather than refused, because the flag is only a hint about tile
            // ordering and nothing in this reader depends on it.
            clustered: byte(head, 96)? != 0,
            internal_compression: Compression::from_byte("internal_compression", byte(head, 97)?)?,
            tile_compression: Compression::from_byte("tile_compression", byte(head, 98)?)?,
            tile_type: TileType::from_byte("tile_type", byte(head, 99)?)?,
            min_zoom,
            max_zoom,
            min_lon_e7: le_i32(head, 102)?,
            min_lat_e7: le_i32(head, 106)?,
            max_lon_e7: le_i32(head, 110)?,
            max_lat_e7: le_i32(head, 114)?,
            center_zoom: byte(head, 118)?,
            center_lon_e7: le_i32(head, 119)?,
            center_lat_e7: le_i32(head, 123)?,
        })
    }

    /// The metadata block as a range, or `None` when the archive has none.
    ///
    /// A zero-length metadata block is legal and must not become an empty
    /// [`ByteRange`], which would be un-requestable. `metadata_length` is
    /// already bounded by [`MAX_METADATA_BYTES`] — [`PmtilesHeader::parse`]
    /// refuses a header declaring more — so the emitted range can never turn
    /// a corrupt length into a multi-terabyte request.
    #[must_use]
    pub fn metadata_range(&self) -> Option<ByteRange> {
        if self.metadata_length == 0 {
            return None;
        }
        let end = self.metadata_offset.checked_add(self.metadata_length)?;
        ByteRange::new(self.metadata_offset, end).ok()
    }

    /// Bounding box as `[min_lon, min_lat, max_lon, max_lat]` in degrees.
    #[must_use]
    pub fn bounds_deg(&self) -> [f64; 4] {
        [
            e7_to_degrees(self.min_lon_e7),
            e7_to_degrees(self.min_lat_e7),
            e7_to_degrees(self.max_lon_e7),
            e7_to_degrees(self.max_lat_e7),
        ]
    }

    /// Suggested map centre as `(lon, lat)` in degrees.
    #[must_use]
    pub fn center_deg(&self) -> (f64, f64) {
        (
            e7_to_degrees(self.center_lon_e7),
            e7_to_degrees(self.center_lat_e7),
        )
    }

    /// Whether the header declares a usable bounding box.
    ///
    /// A writer that did not know the extent leaves all four fields zero; that
    /// degenerate box must not be used as a rejection gate, or the archive
    /// answers `Absent` everywhere.
    #[must_use]
    pub const fn has_bounds(&self) -> bool {
        self.max_lon_e7 > self.min_lon_e7 && self.max_lat_e7 > self.min_lat_e7
    }

    /// One past the last byte of the leaf-directory region.
    #[must_use]
    pub const fn leaf_dirs_end(&self) -> u64 {
        self.leaf_dirs_offset.saturating_add(self.leaf_dirs_len)
    }

    /// One past the last byte of the tile-data region.
    #[must_use]
    pub const fn tile_data_end(&self) -> u64 {
        self.tile_data_offset.saturating_add(self.tile_data_len)
    }
}

/// Converts a header coordinate field to degrees.
fn e7_to_degrees(value: i32) -> f64 {
    f64::from(value) / COORD_SCALE_E7
}

/// Reads one byte, refusing rather than panicking if it is not there.
fn byte(head: &[u8], at: usize) -> Result<u8, PmtilesError> {
    head.get(at).copied().ok_or(PmtilesError::Truncated {
        context: "header",
        needed: u64::try_from(at).unwrap_or(u64::MAX).saturating_add(1),
        available: u64::try_from(head.len()).unwrap_or(u64::MAX),
    })
}

/// Reads a little-endian `u64`.
fn le_u64(head: &[u8], at: usize) -> Result<u64, PmtilesError> {
    let slice = head
        .get(at..at.saturating_add(8))
        .ok_or(PmtilesError::Truncated {
            context: "header",
            needed: u64::try_from(at).unwrap_or(u64::MAX).saturating_add(8),
            available: u64::try_from(head.len()).unwrap_or(u64::MAX),
        })?;
    let array = <[u8; 8]>::try_from(slice).map_err(|_| PmtilesError::Truncated {
        context: "header",
        needed: 8,
        available: u64::try_from(slice.len()).unwrap_or(u64::MAX),
    })?;
    Ok(u64::from_le_bytes(array))
}

/// Reads a little-endian `i32`.
fn le_i32(head: &[u8], at: usize) -> Result<i32, PmtilesError> {
    let slice = head
        .get(at..at.saturating_add(4))
        .ok_or(PmtilesError::Truncated {
            context: "header",
            needed: u64::try_from(at).unwrap_or(u64::MAX).saturating_add(4),
            available: u64::try_from(head.len()).unwrap_or(u64::MAX),
        })?;
    let array = <[u8; 4]>::try_from(slice).map_err(|_| PmtilesError::Truncated {
        context: "header",
        needed: 4,
        available: u64::try_from(slice.len()).unwrap_or(u64::MAX),
    })?;
    Ok(i32::from_le_bytes(array))
}

/// Turns an offset/length pair into a non-degenerate range.
fn region(field: &'static str, offset: u64, length: u64) -> Result<ByteRange, PmtilesError> {
    let Some(end) = offset.checked_add(length) else {
        return Err(PmtilesError::InvalidHeader {
            field,
            reason: "offset plus length overflows u64",
        });
    };
    if length == 0 {
        return Err(PmtilesError::InvalidHeader {
            field,
            reason: "the region is empty",
        });
    }
    ByteRange::new(offset, end).map_err(|_| PmtilesError::InvalidRange { start: offset, end })
}

#[cfg(test)]
mod tests {
    use super::{
        Compression, HEADER_LEN, MAX_METADATA_BYTES, MAX_PMTILES_ZOOM, PMTILES_MAGIC, PREFETCH_LEN,
        PmtilesHeader, TileType,
    };
    use crate::pmtiles::PmtilesError;
    use crate::pmtiles::fixture::{PmtilesBuilder, sample_pmtiles_vector};

    /// A header laid out exactly like a real archive's, so the tests below can
    /// poke one field at a time.
    fn header_bytes() -> Vec<u8> {
        let mut bytes = vec![0u8; HEADER_LEN];
        bytes[..7].copy_from_slice(PMTILES_MAGIC);
        bytes[7] = 3;
        bytes[8..16].copy_from_slice(&127u64.to_le_bytes()); // root offset
        bytes[16..24].copy_from_slice(&15_564u64.to_le_bytes()); // root length
        bytes[24..32].copy_from_slice(&342u64.to_le_bytes()); // metadata offset
        bytes[32..40].copy_from_slice(&681u64.to_le_bytes()); // metadata length
        bytes[40..48].copy_from_slice(&1_023u64.to_le_bytes()); // leaf offset
        bytes[48..56].copy_from_slice(&848_087u64.to_le_bytes()); // leaf length
        bytes[56..64].copy_from_slice(&849_110u64.to_le_bytes()); // tile data offset
        bytes[64..72].copy_from_slice(&12_345u64.to_le_bytes()); // tile data length
        bytes[72..80].copy_from_slice(&1_431_655_765u64.to_le_bytes()); // addressed
        bytes[80..88].copy_from_slice(&2_917u64.to_le_bytes()); // entries
        bytes[88..96].copy_from_slice(&2_048u64.to_le_bytes()); // contents
        bytes[96] = 1; // clustered
        bytes[97] = 2; // internal = gzip
        bytes[98] = 1; // tile = none
        bytes[99] = 1; // tile type = MVT
        bytes[100] = 0;
        bytes[101] = 15;
        bytes[102..106].copy_from_slice(&(-1_800_000_000i32).to_le_bytes());
        bytes[106..110].copy_from_slice(&(-850_511_287i32).to_le_bytes());
        bytes[110..114].copy_from_slice(&1_800_000_000i32.to_le_bytes());
        bytes[114..118].copy_from_slice(&850_511_287i32.to_le_bytes());
        bytes[118] = 5;
        bytes[119..123].copy_from_slice(&1_396_000_000i32.to_le_bytes());
        bytes[123..127].copy_from_slice(&357_000_000i32.to_le_bytes());
        bytes
    }

    #[test]
    fn every_field_of_a_real_shaped_header_decodes() {
        let header = PmtilesHeader::parse(&header_bytes()).expect("a well-formed header");
        assert_eq!(header.root.start, 127);
        assert_eq!(header.root.end, 127 + 15_564);
        assert_eq!(header.metadata_offset, 342);
        assert_eq!(header.metadata_length, 681);
        assert_eq!(header.leaf_dirs_offset, 1_023);
        assert_eq!(header.leaf_dirs_len, 848_087);
        assert_eq!(header.tile_data_offset, 849_110);
        assert_eq!(header.tile_data_len, 12_345);
        assert_eq!(header.addressed_tiles, 1_431_655_765);
        assert_eq!(header.tile_entries, 2_917);
        assert_eq!(header.tile_contents, 2_048);
        assert!(header.clustered);
        assert_eq!(header.internal_compression, Compression::Gzip);
        assert_eq!(header.tile_compression, Compression::None);
        assert_eq!(header.tile_type, TileType::Mvt);
        assert_eq!(header.min_zoom, 0);
        assert_eq!(header.max_zoom, 15);
        assert_eq!(header.center_zoom, 5);
        assert_eq!(header.leaf_dirs_end(), 1_023 + 848_087);
        assert_eq!(header.tile_data_end(), 849_110 + 12_345);
    }

    #[test]
    fn addressed_tiles_is_the_z0_to_z15_tile_count() {
        // (4^16 - 1)/3 = every tile from z0 through z15. An independent check
        // that the u64 field offsets are right rather than shifted.
        let header = PmtilesHeader::parse(&header_bytes()).expect("a well-formed header");
        let expected: u64 = (0..=15).map(|z| 4u64.pow(z)).sum();
        assert_eq!(header.addressed_tiles, expected);
        assert_eq!(expected, 1_431_655_765);
    }

    #[test]
    fn bbox_and_centre_convert_from_e7_to_degrees() {
        let header = PmtilesHeader::parse(&header_bytes()).expect("a well-formed header");
        let bounds = header.bounds_deg();
        assert!((bounds[0] - -180.0).abs() < 1e-9);
        assert!((bounds[1] - -85.051_128_7).abs() < 1e-9);
        assert!((bounds[2] - 180.0).abs() < 1e-9);
        assert!((bounds[3] - 85.051_128_7).abs() < 1e-9);
        let (lon, lat) = header.center_deg();
        assert!((lon - 139.6).abs() < 1e-9);
        assert!((lat - 35.7).abs() < 1e-9);
        assert!(header.has_bounds());
    }

    #[test]
    fn an_undeclared_bbox_is_reported_as_absent() {
        let mut bytes = header_bytes();
        bytes[102..118].fill(0);
        let header = PmtilesHeader::parse(&bytes).expect("a zero bbox is legal");
        assert!(!header.has_bounds());
    }

    #[test]
    fn wrong_magic_is_refused() {
        let mut bytes = header_bytes();
        bytes[0] = b'X';
        assert_eq!(PmtilesHeader::parse(&bytes), Err(PmtilesError::BadMagic));
    }

    #[test]
    fn version_two_is_refused_by_name() {
        let mut bytes = header_bytes();
        bytes[7] = 2;
        assert_eq!(
            PmtilesHeader::parse(&bytes),
            Err(PmtilesError::UnsupportedVersion(2))
        );
    }

    #[test]
    fn a_short_header_is_truncated_not_panicking() {
        let bytes = header_bytes();
        for len in [0usize, 1, 7, 8, 126] {
            let err = PmtilesHeader::parse(&bytes[..len]).expect_err("too short");
            assert!(matches!(err, PmtilesError::Truncated { .. }), "{err:?}");
        }
    }

    #[test]
    fn an_out_of_range_compression_byte_is_refused() {
        let mut bytes = header_bytes();
        bytes[97] = 9;
        assert_eq!(
            PmtilesHeader::parse(&bytes),
            Err(PmtilesError::UnknownEnum {
                field: "internal_compression",
                value: 9
            })
        );
        let mut bytes = header_bytes();
        bytes[98] = 200;
        assert_eq!(
            PmtilesHeader::parse(&bytes),
            Err(PmtilesError::UnknownEnum {
                field: "tile_compression",
                value: 200
            })
        );
    }

    #[test]
    fn an_out_of_range_tile_type_byte_is_refused() {
        let mut bytes = header_bytes();
        bytes[99] = 6;
        assert_eq!(
            PmtilesHeader::parse(&bytes),
            Err(PmtilesError::UnknownEnum {
                field: "tile_type",
                value: 6
            })
        );
    }

    #[test]
    fn a_root_past_the_prefetch_is_refused() {
        let mut bytes = header_bytes();
        bytes[16..24].copy_from_slice(&20_000u64.to_le_bytes());
        assert_eq!(
            PmtilesHeader::parse(&bytes),
            Err(PmtilesError::RootOutsidePrefetch {
                end: 127 + 20_000,
                limit: PREFETCH_LEN,
            })
        );
    }

    #[test]
    fn an_empty_root_directory_is_refused() {
        let mut bytes = header_bytes();
        bytes[16..24].copy_from_slice(&0u64.to_le_bytes());
        assert!(matches!(
            PmtilesHeader::parse(&bytes),
            Err(PmtilesError::InvalidHeader { .. })
        ));
    }

    #[test]
    fn a_root_overlapping_the_header_is_refused() {
        let mut bytes = header_bytes();
        bytes[8..16].copy_from_slice(&100u64.to_le_bytes());
        assert!(matches!(
            PmtilesHeader::parse(&bytes),
            Err(PmtilesError::InvalidHeader {
                field: "root directory offset",
                ..
            })
        ));
    }

    #[test]
    fn an_inverted_or_excessive_zoom_range_is_refused() {
        let mut bytes = header_bytes();
        bytes[100] = 9;
        bytes[101] = 3;
        assert!(matches!(
            PmtilesHeader::parse(&bytes),
            Err(PmtilesError::InvalidHeader {
                field: "zoom range",
                ..
            })
        ));

        let mut bytes = header_bytes();
        bytes[101] = MAX_PMTILES_ZOOM + 1;
        assert!(matches!(
            PmtilesHeader::parse(&bytes),
            Err(PmtilesError::InvalidHeader {
                field: "max_zoom",
                ..
            })
        ));
    }

    #[test]
    fn an_overflowing_region_is_refused() {
        let mut bytes = header_bytes();
        bytes[56..64].copy_from_slice(&(u64::MAX - 4).to_le_bytes());
        bytes[64..72].copy_from_slice(&64u64.to_le_bytes());
        assert!(matches!(
            PmtilesHeader::parse(&bytes),
            Err(PmtilesError::InvalidHeader {
                field: "tile data",
                ..
            })
        ));
    }

    #[test]
    fn a_zero_length_metadata_block_has_no_range() {
        let mut bytes = header_bytes();
        bytes[32..40].copy_from_slice(&0u64.to_le_bytes());
        let header = PmtilesHeader::parse(&bytes).expect("no metadata is legal");
        assert_eq!(header.metadata_range(), None);
    }

    #[test]
    fn a_metadata_length_past_the_cap_is_refused() {
        // Otherwise legal (no overflow), just declaring a metadata block far
        // past what this reader will ever fetch.
        let mut bytes = header_bytes();
        bytes[32..40].copy_from_slice(&(MAX_METADATA_BYTES + 1).to_le_bytes());
        assert!(matches!(
            PmtilesHeader::parse(&bytes),
            Err(PmtilesError::InvalidHeader {
                field: "metadata",
                ..
            })
        ));
    }

    #[test]
    fn a_metadata_length_at_the_cap_is_accepted() {
        let mut bytes = header_bytes();
        bytes[32..40].copy_from_slice(&MAX_METADATA_BYTES.to_le_bytes());
        let header = PmtilesHeader::parse(&bytes).expect("exactly at the cap is legal");
        assert_eq!(header.metadata_length, MAX_METADATA_BYTES);
    }

    #[test]
    fn the_fixture_archive_parses_as_a_v3_header() {
        let archive = sample_pmtiles_vector();
        let header = PmtilesHeader::parse(&archive).expect("the fixture is well-formed");
        assert_eq!(header.tile_type, TileType::Mvt);
        assert_eq!(header.internal_compression, Compression::None);
        assert_eq!(header.root.start, 127);
        assert!(header.root.end <= PREFETCH_LEN);
        assert_eq!(header.min_zoom, 0);
        assert_eq!(header.max_zoom, 1);
    }

    #[test]
    fn enum_bytes_round_trip() {
        for value in 0u8..=4 {
            let compression = Compression::from_byte("internal_compression", value)
                .expect("0..=4 are all defined");
            assert_eq!(compression.to_byte(), value);
            assert!(!compression.name().is_empty());
        }
        for value in 0u8..=5 {
            let tile_type = TileType::from_byte("tile_type", value).expect("0..=5 are all defined");
            assert_eq!(tile_type.to_byte(), value);
            assert!(!tile_type.name().is_empty());
        }
        assert!(Compression::Brotli.is_refused());
        assert!(Compression::Zstd.is_refused());
        assert!(!Compression::Gzip.is_refused());
        assert!(TileType::Mvt.is_vector());
        assert!(!TileType::Png.is_vector());
    }

    #[test]
    fn a_gzip_internal_fixture_declares_gzip() {
        let mut builder = PmtilesBuilder::new(TileType::Mvt)
            .with_compression(Compression::Gzip, Compression::Gzip);
        builder.push_tile(0, 0, 0, vec![1, 2, 3]);
        let bytes = builder.build();
        let header = PmtilesHeader::parse(&bytes).expect("well-formed");
        assert_eq!(header.internal_compression, Compression::Gzip);
        assert_eq!(header.tile_compression, Compression::Gzip);
    }
}
