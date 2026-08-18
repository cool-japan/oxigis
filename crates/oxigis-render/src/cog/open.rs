//! The pull-based COG open: a state machine the caller feeds bytes to.
//!
//! `oxigis-render` performs no I/O, and the UI's tile-provider seam is
//! synchronous and non-blocking, so opening a COG cannot be an `async fn` that
//! awaits a fetch. Instead [`CogOpen`] is driven by alternating two calls:
//!
//! ```text
//! let mut open = CogOpen::new();
//! loop {
//!     match open.poll()? {
//!         CogOpenProgress::Need(range) => {
//!             let bytes = /* caller fetches `range` however it likes */;
//!             open.supply(range.start, bytes);
//!         }
//!         CogOpenProgress::Ready(metadata) => break,
//!     }
//! }
//! ```
//!
//! [`CogOpen::poll`] is re-entrant: it re-runs the parse from whatever bytes are
//! present, so a caller may supply ranges in any order, supply more than was
//! asked for, or drop the machine and start over. In practice a GDAL-written COG
//! completes in **one** round trip, because the first request is a speculative
//! [`HEADER_PREFETCH_BYTES`] read that covers the whole IFD chain and every
//! level's tile directory.
//!
//! # Provenance
//!
//! The parse itself — which tags matter, the inline-versus-offset value rule,
//! the strip-as-tile fallback, walking the next-IFD chain for overviews — is
//! ported from `oxigeo-wasm`'s `WasmCogReader::open`/`parse_ifd`
//! (cool-japan/oxigeo, Apache-2.0, same author). Turning it inside out from
//! `async` into this state machine, the header prefetch, and the skipping of
//! internal mask IFDs are new.

use crate::cog::blocks::{BlockMiss, ByteBlocks, HEADER_PREFETCH_BYTES};
use crate::cog::codec::{CogDecodeOptions, RasterStretch};
use crate::cog::meta::{
    CogGeoTransform, CogLevel, CogMetadata, MAX_TILE_DECOMPRESSED_BYTES, PHOTOMETRIC_MASK,
};
use crate::cog::tiff::{self, ByteOrder, IfdEntry, SUBFILE_TYPE_MASK, TiffHeader, TiffVariant};
use crate::error::RenderError;
use crate::source::ByteRange;

/// Hard limit on how far the next-IFD chain is followed, so a malformed or
/// cyclic file cannot loop forever.
pub const MAX_IFD_CHAIN: usize = 100;

/// Hard limit on the number of entries a single tile directory may declare.
///
/// 4 million tiles is ~32 MiB of offsets, far past any real COG (a
/// 500k × 500k image at 256 px tiles has ~3.8 M tiles), and stops a corrupt
/// `count` field from being turned into a huge allocation.
pub const MAX_TILE_DIRECTORY_ENTRIES: usize = 4 * 1024 * 1024;

/// Hard limit on the number of `ColorMap` (tag 320) entries read.
///
/// A palette is three ramps of `2^BitsPerSample` entries, so 768 at 8 bits and
/// 196 608 at 16 — both widths [`super::codec`] renders. 384 KiB is the whole
/// cost of the larger one, and a count past it is not a palette.
pub const MAX_COLOR_MAP_ENTRIES: usize = 3 * 65_536;

/// Hard limit on the number of `GeoKeyDirectory` (tag 34735) values read.
///
/// The header quadruple plus 1024 keys; the GeoTIFF key set is under a hundred
/// entries, and a directory past this is corrupt rather than exotic.
pub const MAX_GEO_KEY_VALUES: usize = 4 + 4 * 1_024;

/// Hard limit on the number of `ModelTiepoint` (tag 33922) values read.
///
/// Six doubles per tie point. A COG carries exactly one (the north-west
/// corner); the allowance covers a GCP-referenced GeoTIFF without letting a
/// corrupt count size an allocation.
pub const MAX_TIEPOINT_VALUES: usize = 6 * 64;

/// Hard limit on the number of per-band values (`BitsPerSample` 258,
/// `SampleFormat` 339) read.
///
/// Hyperspectral products reach a few hundred bands; the codec reads only the
/// first value of each, so this is purely an allocation ceiling.
pub const MAX_PER_BAND_VALUES: usize = 1_024;

/// Hard limit on the number of `JPEGTables` (tag 347) bytes read.
///
/// A full JFIF table set (four quantisation tables, four Huffman tables) runs
/// to a few hundred bytes; 64 KiB is the largest a conforming encoder can emit
/// in one marker segment, and covers every table set in practice.
pub const MAX_JPEG_TABLE_BYTES: usize = 64 * 1_024;

/// Hard limit on the number of entries one IFD may declare.
///
/// Classic TIFF caps the count at 65 535 by field width; BigTIFF's is 64 bits
/// wide, so a corrupt value would otherwise size a read of the whole address
/// space. A real directory holds a few dozen tags.
pub const MAX_IFD_ENTRIES: usize = 4 * 1_024;

/// Hard limit on the number of values of a tag with no cap of its own.
///
/// [`value_bytes`] only ever reads the tags [`parse_ifd`] matches on, so this
/// applies to the short ones (`ModelPixelScale`, `GDAL_NODATA`).
const MAX_OTHER_TAG_VALUES: usize = 4 * 1_024;

/// What [`CogOpen::poll`] wants next.
#[derive(Debug)]
pub enum CogOpenProgress<'a> {
    /// These bytes are needed before the parse can continue.
    ///
    /// The caller may supply exactly this range, or any superset of it.
    Need(ByteRange),
    /// The file is fully described; stop polling.
    Ready(&'a CogMetadata),
}

/// Where the parse currently is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Nothing read yet; the header prefetch has not been requested.
    Start,
    /// Reading the byte-order mark, magic number and first-IFD offset.
    Header,
    /// Parsing the IFD at this file offset (0 terminates the chain).
    Ifd(u64),
    /// Every IFD has been parsed; the metadata is built.
    Done,
}

/// A COG open in progress: fetched bytes plus the parse position.
#[derive(Debug)]
pub struct CogOpen {
    /// Bytes supplied so far.
    blocks: ByteBlocks,
    /// Byte order, once the header has been read.
    order: Option<ByteOrder>,
    /// Which TIFF dialect the file is written in.
    variant: TiffVariant,
    /// Current parse position.
    stage: Stage,
    /// Levels parsed so far, in file order.
    levels: Vec<CogLevel>,
    /// Offsets of the IFDs already visited, mask IFDs included.
    ///
    /// A set rather than a counter, and rather than requiring the chain to walk
    /// *forward*: TIFF only forbids a cycle, not a backward pointer, and a
    /// writer that appends overviews and rewrites the main IFD afterwards
    /// produces exactly that. Requiring monotonic offsets would silently drop
    /// such a file's whole overview pyramid. Capped by [`MAX_IFD_CHAIN`], and
    /// a chain is at most a hundred entries, so a linear scan beats a hash set.
    visited: Vec<u64>,
    /// EPSG code from the first IFD that declares one.
    epsg: Option<u32>,
    /// Georeference from the first IFD that declares one.
    geo: Option<CogGeoTransform>,
    /// `GDAL_NODATA` from the first IFD that declares one.
    nodata: Option<f64>,
    /// `JPEGTables` from the first IFD that declares them.
    jpeg_tables: Vec<u8>,
    /// The finished metadata, once [`Stage::Done`] is reached.
    metadata: Option<CogMetadata>,
}

impl Default for CogOpen {
    fn default() -> Self {
        Self::new()
    }
}

/// What one IFD contributed to the open.
struct ParsedIfd {
    /// The level it describes, unless it is a mask or has no tile directory.
    level: Option<CogLevel>,
    /// Georeference declared by this IFD, if any.
    geo: Option<CogGeoTransform>,
    /// EPSG code declared by this IFD, if any.
    epsg: Option<u32>,
    /// `GDAL_NODATA` declared by this IFD, if any.
    nodata: Option<f64>,
    /// `JPEGTables` declared by this IFD, empty when it declares none.
    jpeg_tables: Vec<u8>,
    /// Offset of the next IFD, or 0 at the end of the chain.
    next: u64,
}

impl CogOpen {
    /// A fresh open with nothing read.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            blocks: ByteBlocks::new(),
            order: None,
            variant: TiffVariant::Classic,
            stage: Stage::Start,
            levels: Vec::new(),
            visited: Vec::new(),
            epsg: None,
            geo: None,
            nodata: None,
            jpeg_tables: Vec::new(),
            metadata: None,
        }
    }

    /// Hands over the bytes at file offset `start`.
    ///
    /// A short response (a range that ran past the end of the file) is fine and
    /// expected for the speculative header read; it is recorded as-is, and a
    /// later read that genuinely falls off the end fails with a decode error
    /// rather than looping.
    pub fn supply(&mut self, start: u64, bytes: Vec<u8>) {
        self.blocks.supply(start, bytes);
    }

    /// Number of byte blocks supplied so far.
    #[must_use]
    pub fn blocks_supplied(&self) -> usize {
        self.blocks.len()
    }

    /// Total number of bytes supplied so far.
    #[must_use]
    pub fn bytes_supplied(&self) -> usize {
        self.blocks.bytes_held()
    }

    /// The finished metadata, if the open has completed.
    #[must_use]
    pub fn metadata(&self) -> Option<&CogMetadata> {
        self.metadata.as_ref()
    }

    /// The per-file decode settings the IFD chain declared.
    ///
    /// [`CogMetadata`] has no field for `GDAL_NODATA` (42113) or `JPEGTables`
    /// (347), so a caller that decodes blocks itself — as `oxigis-ui`'s COG
    /// provider does — carries these beside the metadata and passes them to
    /// [`super::codec::decode_cog_block`]. Without them a JPEG COG cannot be
    /// decoded at all and a nodata collar renders as opaque black.
    ///
    /// Both are taken from the first IFD that declares them, which is the
    /// full-resolution image in every file GDAL writes; an overview that
    /// declared *different* tables would be decoded with the primary set.
    #[must_use]
    pub fn decode_options(&self) -> CogDecodeOptions {
        CogDecodeOptions {
            stretch: RasterStretch::default(),
            nodata: self.nodata,
            jpeg_tables: self.jpeg_tables.clone(),
        }
    }

    /// Consumes the open, returning the finished metadata.
    #[must_use]
    pub fn into_metadata(self) -> Option<CogMetadata> {
        self.metadata
    }

    /// Advances the parse as far as the supplied bytes allow.
    ///
    /// # Errors
    ///
    /// * [`RenderError::Decode`] for a malformed or truncated file.
    /// * [`RenderError::Unsupported`] for BigTIFF or a directory this reader
    ///   cannot represent.
    pub fn poll(&mut self) -> Result<CogOpenProgress<'_>, RenderError> {
        loop {
            match self.stage {
                Stage::Start => {
                    // One speculative read that covers the whole header block of
                    // any conventionally written COG.
                    let range = ByteRange::with_len(0, HEADER_PREFETCH_BYTES)?;
                    self.blocks.note_requested(range);
                    self.stage = Stage::Header;
                    return Ok(CogOpenProgress::Need(range));
                }
                Stage::Header => {
                    // A BigTIFF header is sixteen bytes; reading that many up
                    // front is safe because the classic parse only looks at the
                    // first eight.
                    let bytes = match self.blocks.read(0, TiffVariant::Big.header_bytes()) {
                        Ok(bytes) => bytes,
                        Err(miss) => return self.handle_miss(miss),
                    };
                    let header = TiffHeader::parse(bytes)?;
                    self.order = Some(header.order);
                    self.variant = header.variant;
                    self.stage = Stage::Ifd(header.first_ifd);
                }
                Stage::Ifd(offset) => {
                    if offset == 0
                        || self.visited.len() >= MAX_IFD_CHAIN
                        || self.visited.contains(&offset)
                    {
                        self.stage = Stage::Done;
                        continue;
                    }
                    let Some(order) = self.order else {
                        return Err(RenderError::Decode(
                            "COG parse reached an IFD before its byte order".to_owned(),
                        ));
                    };
                    let parsed = match parse_ifd(&self.blocks, order, self.variant, offset) {
                        Ok(parsed) => parsed,
                        Err(IfdError::Miss(miss)) => return self.handle_miss(miss),
                        Err(IfdError::Fatal(error)) => return Err(error),
                    };
                    self.visited.push(offset);
                    if self.geo.is_none() {
                        self.geo = parsed.geo;
                    }
                    if self.epsg.is_none() {
                        self.epsg = parsed.epsg;
                    }
                    if self.nodata.is_none() {
                        self.nodata = parsed.nodata;
                    }
                    if self.jpeg_tables.is_empty() {
                        self.jpeg_tables = parsed.jpeg_tables;
                    }
                    if let Some(level) = parsed.level {
                        self.levels.push(level);
                    }
                    // Any offset is allowed; the visited set above is what stops
                    // a cyclic chain, including the self-referential case.
                    self.stage = Stage::Ifd(parsed.next);
                }
                Stage::Done => {
                    if self.metadata.is_none() {
                        if self.levels.is_empty() {
                            return Err(RenderError::Decode(
                                "COG has no tiled image directory (a striped TIFF with no \
                                 StripOffsets, or an all-mask file)"
                                    .to_owned(),
                            ));
                        }
                        let samples_per_pixel = self
                            .levels
                            .first()
                            .map_or(1, |level| level.samples_per_pixel);
                        self.metadata = Some(CogMetadata {
                            little_endian: self.order.is_some_and(ByteOrder::is_little_endian),
                            samples_per_pixel,
                            levels: core::mem::take(&mut self.levels),
                            epsg: self.epsg,
                            geo: self.geo,
                        });
                    }
                    let Some(metadata) = self.metadata.as_ref() else {
                        return Err(RenderError::Decode(
                            "COG metadata vanished after being built".to_owned(),
                        ));
                    };
                    return Ok(CogOpenProgress::Ready(metadata));
                }
            }
        }
    }

    /// Turns a block miss into either a fetch request or a fatal error.
    fn handle_miss(&mut self, miss: BlockMiss) -> Result<CogOpenProgress<'_>, RenderError> {
        match miss {
            BlockMiss::Fetch(range) => {
                self.blocks.note_requested(range);
                Ok(CogOpenProgress::Need(range))
            }
            truncated => Err(truncated.into_error()),
        }
    }
}

/// Why parsing one IFD stopped.
enum IfdError {
    /// Bytes are missing.
    Miss(BlockMiss),
    /// The directory is malformed or unsupported.
    Fatal(RenderError),
}

impl From<RenderError> for IfdError {
    fn from(error: RenderError) -> Self {
        Self::Fatal(error)
    }
}

/// Largest value count `tag` may legally declare.
///
/// A single global byte ceiling is not enough: 33 554 432 *bytes* leaves a
/// BYTE-typed tag free to declare 33.5 M values, which sizes a 268 MB `u64`
/// vector and asks a server for a 33 MB range to answer a twelve-byte tag.
const fn max_values_for_tag(tag: u16) -> usize {
    match tag {
        tiff::TAG_TILE_OFFSETS
        | tiff::TAG_TILE_BYTE_COUNTS
        | tiff::TAG_STRIP_OFFSETS
        | tiff::TAG_STRIP_BYTE_COUNTS => MAX_TILE_DIRECTORY_ENTRIES,
        tiff::TAG_COLOR_MAP => MAX_COLOR_MAP_ENTRIES,
        tiff::TAG_GEO_KEY_DIRECTORY => MAX_GEO_KEY_VALUES,
        tiff::TAG_MODEL_TIEPOINT => MAX_TIEPOINT_VALUES,
        tiff::TAG_BITS_PER_SAMPLE | tiff::TAG_SAMPLE_FORMAT => MAX_PER_BAND_VALUES,
        tiff::TAG_JPEG_TABLES => MAX_JPEG_TABLE_BYTES,
        _ => MAX_OTHER_TAG_VALUES,
    }
}

/// Reads one field's value bytes, wherever they live.
fn value_bytes<'a>(
    blocks: &'a ByteBlocks,
    entry: &'a IfdEntry,
    order: ByteOrder,
) -> Result<&'a [u8], IfdError> {
    let limit = max_values_for_tag(entry.tag);
    if usize::try_from(entry.count).unwrap_or(usize::MAX) > limit {
        return Err(IfdError::Fatal(RenderError::Decode(format!(
            "COG tag {} declares {} values, past its {limit} limit",
            entry.tag, entry.count
        ))));
    }
    if entry.is_inline() {
        return Ok(&entry.value);
    }
    let Some(size) = entry.values_bytes() else {
        // An unknown field type has no size, so there is nothing to read; the
        // caller's decoders return empty for it anyway.
        return Ok(&[]);
    };
    let Some(offset) = entry.value_offset(order) else {
        return Err(IfdError::Fatal(RenderError::Decode(format!(
            "COG tag {} has an unreadable value offset",
            entry.tag
        ))));
    };
    blocks.read(offset, size).map_err(IfdError::Miss)
}

/// Parses the IFD at `offset` into one pyramid level plus georeference.
///
/// Ported from `oxigeo-wasm`'s `WasmCogReader::parse_ifd`, with the async
/// external-array reads replaced by [`ByteBlocks`] lookups.
#[expect(
    clippy::too_many_lines,
    reason = "one match arm per TIFF tag reads better than a dispatch table"
)]
fn parse_ifd(
    blocks: &ByteBlocks,
    order: ByteOrder,
    variant: TiffVariant,
    offset: u64,
) -> Result<ParsedIfd, IfdError> {
    let count_width = variant.entry_count_bytes();
    let entry_bytes = variant.entry_bytes();
    let offset_bytes = variant.offset_bytes();
    let count_span = blocks.read(offset, count_width).map_err(IfdError::Miss)?;
    let entry_count = match variant {
        TiffVariant::Classic => order.short(count_span).map(u64::from),
        TiffVariant::Big => order.long8(count_span),
    }
    .ok_or_else(|| {
        IfdError::Fatal(RenderError::Decode(
            "COG IFD entry count is unreadable".to_owned(),
        ))
    })?;
    // BigTIFF's count is 64 bits wide, so a corrupt one would otherwise size a
    // read of the whole address space; a real directory holds a few dozen tags.
    let entry_count = usize::try_from(entry_count).unwrap_or(usize::MAX);
    if entry_count > MAX_IFD_ENTRIES {
        return Err(IfdError::Fatal(RenderError::Decode(format!(
            "COG IFD declares {entry_count} entries, past the {MAX_IFD_ENTRIES} limit"
        ))));
    }
    let directory_bytes = entry_count * entry_bytes;
    // Entries plus the trailing next-IFD pointer, read as one span.
    let directory = blocks
        .read(offset + count_width as u64, directory_bytes + offset_bytes)
        .map_err(IfdError::Miss)?;

    let mut width = 0u64;
    let mut height = 0u64;
    let mut tile_width = 0u32;
    let mut tile_height = 0u32;
    let mut rows_per_strip = 0u32;
    let mut bits_per_sample = 8u16;
    let mut samples_per_pixel = 1u16;
    let mut sample_format = 1u16;
    let mut compression = 1u16;
    let mut predictor = 1u16;
    let mut photometric = 1u16;
    let mut subfile_type = 0u32;
    let mut color_map: Vec<u16> = Vec::new();
    let mut tile_offsets: Vec<u64> = Vec::new();
    let mut tile_byte_counts: Vec<u64> = Vec::new();
    let mut pixel_scale: Vec<f64> = Vec::new();
    let mut tiepoint: Vec<f64> = Vec::new();
    let mut geo_keys: Vec<u64> = Vec::new();
    let mut jpeg_tables: Vec<u8> = Vec::new();
    let mut nodata: Option<f64> = None;

    for index in 0..entry_count {
        let start = index * entry_bytes;
        let Some(raw) = directory.get(start..start + entry_bytes) else {
            break;
        };
        let entry = IfdEntry::parse(raw, order, variant)?;
        match entry.tag {
            tiff::TAG_NEW_SUBFILE_TYPE => {
                subfile_type = read_scalar(&entry, order)
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0);
            }
            tiff::TAG_IMAGE_WIDTH => {
                width = read_scalar(&entry, order).unwrap_or(0);
            }
            tiff::TAG_IMAGE_LENGTH => {
                height = read_scalar(&entry, order).unwrap_or(0);
            }
            tiff::TAG_BITS_PER_SAMPLE => {
                // Multi-band images carry one SHORT per band; three of them do
                // not fit the inline field, so the entry holds an offset.
                // Reading that offset as a scalar yields a garbage bit depth,
                // which is why the array path exists (oxigeo-wasm's comment on
                // the same tag records the bug this fixed there). Bands share
                // one depth in COG practice, so the first value is authoritative.
                let data = value_bytes(blocks, &entry, order)?;
                bits_per_sample =
                    u16::try_from(entry.integers(data, order).first().copied().unwrap_or(8))
                        .unwrap_or(8);
            }
            // These four are SHORT per the specification, but libtiff and GDAL
            // accept LONG; read_scalar dispatches on the field type, where a
            // fixed two-byte read would take the *high* half of a big-endian
            // LONG and turn SamplesPerPixel 3 into 0.
            tiff::TAG_COMPRESSION => {
                compression = read_scalar(&entry, order)
                    .and_then(|v| u16::try_from(v).ok())
                    .unwrap_or(1);
            }
            tiff::TAG_PHOTOMETRIC => {
                photometric = read_scalar(&entry, order)
                    .and_then(|v| u16::try_from(v).ok())
                    .unwrap_or(1);
            }
            tiff::TAG_PREDICTOR => {
                predictor = read_scalar(&entry, order)
                    .and_then(|v| u16::try_from(v).ok())
                    .unwrap_or(1);
            }
            tiff::TAG_SAMPLES_PER_PIXEL => {
                samples_per_pixel = read_scalar(&entry, order)
                    .and_then(|v| u16::try_from(v).ok())
                    .unwrap_or(1);
            }
            tiff::TAG_SAMPLE_FORMAT => {
                let data = value_bytes(blocks, &entry, order)?;
                sample_format =
                    u16::try_from(entry.integers(data, order).first().copied().unwrap_or(1))
                        .unwrap_or(1);
            }
            tiff::TAG_ROWS_PER_STRIP => {
                rows_per_strip = read_scalar(&entry, order)
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0);
            }
            tiff::TAG_TILE_WIDTH => {
                tile_width = read_scalar(&entry, order)
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0);
            }
            tiff::TAG_TILE_LENGTH => {
                tile_height = read_scalar(&entry, order)
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0);
            }
            // Three ramps of 2^BitsPerSample SHORTs, so 768 entries at 8 bits
            // and 196 608 at 16. Reading only the 8-bit width would leave a
            // legal 16-bit palette with an *empty* map, which the codec then
            // reports as "carries no usable ColorMap" — blaming the file for a
            // tag it does contain. The count cap lives in `value_bytes`.
            tiff::TAG_COLOR_MAP => {
                let data = value_bytes(blocks, &entry, order)?;
                color_map = entry
                    .integers(data, order)
                    .into_iter()
                    .map(|value| u16::try_from(value).unwrap_or(0))
                    .collect();
            }
            tiff::TAG_TILE_OFFSETS | tiff::TAG_STRIP_OFFSETS => {
                let data = value_bytes(blocks, &entry, order)?;
                tile_offsets = entry.integers(data, order);
            }
            tiff::TAG_TILE_BYTE_COUNTS | tiff::TAG_STRIP_BYTE_COUNTS => {
                let data = value_bytes(blocks, &entry, order)?;
                tile_byte_counts = entry.integers(data, order);
            }
            tiff::TAG_MODEL_PIXEL_SCALE => {
                let data = value_bytes(blocks, &entry, order)?;
                pixel_scale = entry.doubles(data, order);
            }
            tiff::TAG_MODEL_TIEPOINT => {
                let data = value_bytes(blocks, &entry, order)?;
                tiepoint = entry.doubles(data, order);
            }
            tiff::TAG_GEO_KEY_DIRECTORY => {
                let data = value_bytes(blocks, &entry, order)?;
                geo_keys = entry.integers(data, order);
            }
            tiff::TAG_JPEG_TABLES => {
                jpeg_tables = value_bytes(blocks, &entry, order)?.to_vec();
            }
            tiff::TAG_GDAL_NODATA => {
                nodata = parse_gdal_nodata(value_bytes(blocks, &entry, order)?);
            }
            _ => {}
        }
    }

    let tail = directory
        .get(directory_bytes..directory_bytes + offset_bytes)
        .unwrap_or(&[]);
    let next = match variant {
        TiffVariant::Classic => order.long(tail).map(u64::from),
        TiffVariant::Big => order.long8(tail),
    }
    .unwrap_or(0);

    // A striped TIFF has no TileWidth/TileLength; each strip spans the full
    // image width and `RowsPerStrip` rows, which is exactly a tile grid one
    // column wide. Ported from oxigeo-wasm's strip fallback.
    if tile_width == 0 && rows_per_strip > 0 {
        tile_width = u32::try_from(width).unwrap_or(0);
        tile_height = rows_per_strip;
    }

    let geo = build_geo(&tiepoint, &pixel_scale);
    let epsg = tiff::epsg_from_geo_keys(&geo_keys);

    // Internal transparency masks share the overview chain; treating one as a
    // pyramid level would draw a 1-bit mask over the imagery.
    let is_mask = subfile_type & SUBFILE_TYPE_MASK != 0 || photometric == PHOTOMETRIC_MASK;
    // `samples_per_pixel == 0` is what a mis-read SamplesPerPixel produces, and
    // it makes every derived size zero; refusing the directory here beats
    // papering over it with `.max(1)` at each use in the codec.
    let usable = !is_mask
        && width > 0
        && height > 0
        && tile_width > 0
        && tile_height > 0
        && samples_per_pixel > 0
        && !tile_offsets.is_empty()
        && tile_offsets.len() == tile_byte_counts.len();

    if tile_offsets.len() > MAX_TILE_DIRECTORY_ENTRIES {
        return Err(IfdError::Fatal(RenderError::Decode(format!(
            "COG tile directory declares {} entries, past the {MAX_TILE_DIRECTORY_ENTRIES} limit",
            tile_offsets.len()
        ))));
    }

    let level = usable.then(|| CogLevel {
        width: u32::try_from(width).unwrap_or(u32::MAX),
        height: u32::try_from(height).unwrap_or(u32::MAX),
        tile_width,
        tile_height,
        bits_per_sample,
        samples_per_pixel,
        sample_format,
        compression,
        predictor,
        photometric,
        color_map,
        tile_offsets,
        tile_byte_counts,
    });

    // A tile whose declared geometry decompresses to more than the cap is a
    // corrupt or hostile header, not a product: the reader would otherwise
    // reserve that much before reading a byte of the payload.
    if let Some(level) = level.as_ref() {
        match level.tile_bytes() {
            Some(bytes) if bytes <= MAX_TILE_DECOMPRESSED_BYTES => {}
            Some(bytes) => {
                return Err(IfdError::Fatal(RenderError::Unsupported(format!(
                    "COG block geometry {}x{} at {} bands x {} bits decompresses to {bytes} bytes, \
                     past the {MAX_TILE_DECOMPRESSED_BYTES}-byte limit; re-encode as a tiled COG",
                    level.tile_width,
                    level.tile_height,
                    level.samples_per_pixel,
                    level.bits_per_sample
                ))));
            }
            None => {
                return Err(IfdError::Fatal(RenderError::Unsupported(
                    "COG block geometry overflows an address".to_owned(),
                )));
            }
        }
    }

    Ok(ParsedIfd {
        level,
        geo,
        epsg,
        nodata,
        jpeg_tables,
        next,
    })
}

/// Parses a `GDAL_NODATA` (tag 42113) ASCII value.
///
/// GDAL writes the nodata value as text — `-9999`, `0`, `nan` — NUL-terminated
/// as TIFF ASCII fields are. A value that does not parse is treated as absent
/// rather than fatal: a file is still readable without transparency.
fn parse_gdal_nodata(data: &[u8]) -> Option<f64> {
    let text = core::str::from_utf8(data).ok()?;
    let trimmed = text.trim_matches(|byte: char| byte == '\0' || byte.is_whitespace());
    trimmed.parse::<f64>().ok()
}

/// Reads a scalar SHORT or LONG field from an IFD entry's inline bytes.
fn read_scalar(entry: &IfdEntry, order: ByteOrder) -> Option<u64> {
    match entry.field_type {
        3 | 8 => order.short(&entry.value).map(u64::from),
        4 | 9 => order.long(&entry.value).map(u64::from),
        _ => None,
    }
}

/// Builds a georeference from `ModelTiepoint` and `ModelPixelScale` values.
///
/// Returns `None` when either tag is absent or too short, which is the case for
/// a plain (non-geo) TIFF; a malformed one is also `None` rather than fatal, so
/// such a file still opens and reports "not georeferenced" later.
fn build_geo(tiepoint: &[f64], pixel_scale: &[f64]) -> Option<CogGeoTransform> {
    let raster_x = *tiepoint.first()?;
    let raster_y = *tiepoint.get(1)?;
    let crs_x = *tiepoint.get(3)?;
    let crs_y = *tiepoint.get(4)?;
    let scale_x = *pixel_scale.first()?;
    let scale_y = *pixel_scale.get(1)?;
    CogGeoTransform::new((raster_x, raster_y, crs_x, crs_y), scale_x, scale_y).ok()
}

#[cfg(test)]
mod tests {
    use super::{CogOpen, CogOpenProgress, tiff};
    use crate::cog::fixture::{TiffFixture, tiled_geo_tiff};
    use crate::cog::meta::CogCrs;
    use crate::cog::tiff::ByteOrder;
    use crate::error::RenderError;

    /// Drives an open to completion against an in-memory file.
    fn open_bytes(file: &[u8]) -> Result<crate::cog::meta::CogMetadata, RenderError> {
        let mut open = CogOpen::new();
        for _ in 0..32 {
            match open.poll()? {
                CogOpenProgress::Need(range) => {
                    let start = range.start as usize;
                    let end = (range.end as usize).min(file.len());
                    let slice = file.get(start..end).unwrap_or(&[]).to_vec();
                    open.supply(range.start, slice);
                }
                CogOpenProgress::Ready(_) => break,
            }
        }
        open.into_metadata()
            .ok_or_else(|| RenderError::Decode("open did not complete".to_owned()))
    }

    #[test]
    fn a_minimal_tiled_geotiff_opens_in_one_round_trip() {
        let fixture = tiled_geo_tiff();
        let mut open = CogOpen::new();
        let Ok(CogOpenProgress::Need(range)) = open.poll() else {
            panic!("the first poll must ask for the header prefetch");
        };
        assert_eq!(range.start, 0);
        let end = (range.end as usize).min(fixture.bytes.len());
        open.supply(0, fixture.bytes[..end].to_vec());
        let metadata = match open.poll() {
            Ok(CogOpenProgress::Ready(metadata)) => metadata.clone(),
            _ => panic!("a prefetched COG must open without a second request"),
        };
        assert_eq!(open.blocks_supplied(), 1);
        assert_eq!(metadata.level_count(), 2);
        assert_eq!(metadata.epsg, Some(4326));
        assert_eq!(metadata.crs().ok(), Some(CogCrs::Geographic));
        assert!(metadata.little_endian);
        assert_eq!(metadata.samples_per_pixel, 1);

        let base = metadata.base_level().expect("a base level");
        assert_eq!((base.width, base.height), (8, 8));
        assert_eq!((base.tile_width, base.tile_height), (4, 4));
        assert_eq!(base.tile_offsets.len(), 4);
        assert_eq!(base.tile_byte_counts.len(), 4);

        let overview = metadata.level(1).expect("an overview level");
        assert_eq!((overview.width, overview.height), (4, 4));

        let geo = metadata.geo_transform().expect("a georeference");
        assert!((geo.origin_x - 10.0).abs() < 1e-12);
        assert!((geo.origin_y - 50.0).abs() < 1e-12);
        assert!((geo.pixel_size_x - 0.5).abs() < 1e-12);
        // The overview's pixels are twice as large.
        let overview_geo = metadata.level_transform(1).expect("an overview transform");
        assert!((overview_geo.pixel_size_x - 1.0).abs() < 1e-12);
        assert_eq!(open.bytes_supplied(), end);
    }

    #[test]
    fn out_of_prefetch_arrays_are_requested_separately() {
        // Push the tile directory far past the prefetch window so the state
        // machine has to ask for a second range.
        let fixture = TiffFixture::builder().directory_gap(200_000).build();
        let mut open = CogOpen::new();
        let mut requests = 0;
        let metadata = loop {
            match open.poll().expect("the parse must not fail") {
                CogOpenProgress::Need(range) => {
                    requests += 1;
                    assert!(requests < 16, "the open must converge");
                    let start = range.start as usize;
                    let end = (range.end as usize).min(fixture.bytes.len());
                    let slice = fixture.bytes.get(start..end).unwrap_or(&[]).to_vec();
                    open.supply(range.start, slice);
                }
                CogOpenProgress::Ready(metadata) => break metadata.clone(),
            }
        };
        assert!(requests >= 2, "a far-away directory needs a second fetch");
        assert_eq!(
            metadata
                .base_level()
                .expect("a base level")
                .tile_offsets
                .len(),
            4
        );
    }

    #[test]
    fn a_big_endian_file_parses() {
        let fixture = TiffFixture::builder().big_endian(true).build();
        let metadata = open_bytes(&fixture.bytes).expect("an MM COG must open");
        assert!(!metadata.little_endian);
        assert_eq!(metadata.epsg, Some(4326));
        assert_eq!(metadata.base_level().expect("a base level").width, 8);
    }

    #[test]
    fn mask_ifds_are_not_taken_for_overviews() {
        let fixture = TiffFixture::builder().mask_overview(true).build();
        let metadata = open_bytes(&fixture.bytes).expect("a masked COG must open");
        assert_eq!(
            metadata.level_count(),
            1,
            "the mask IFD must not become a pyramid level"
        );
    }

    #[test]
    fn a_web_mercator_file_reports_its_crs() {
        let fixture = TiffFixture::builder()
            .epsg(3857)
            .origin(-20_037_508.0, 20_037_508.0)
            .pixel_size(10.0)
            .build();
        let metadata = open_bytes(&fixture.bytes).expect("a 3857 COG must open");
        assert_eq!(metadata.crs().ok(), Some(CogCrs::WebMercator));
    }

    #[test]
    fn a_truncated_file_is_a_decode_error() {
        let fixture = tiled_geo_tiff();
        let truncated = &fixture.bytes[..fixture.bytes.len() / 2];
        assert!(matches!(open_bytes(truncated), Err(RenderError::Decode(_))));
    }

    #[test]
    fn a_non_tiff_is_rejected() {
        let png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
        assert!(matches!(open_bytes(&png), Err(RenderError::Decode(_))));
    }

    #[test]
    fn a_classic_file_relabelled_as_bigtiff_is_refused() {
        // Flipping the magic number leaves the classic 4-byte first-IFD offset
        // where BigTIFF expects its offset-size field, so the file declares an
        // offset width the format does not define.
        let mut bigtiff = tiled_geo_tiff().bytes;
        bigtiff[2] = 43;
        assert!(matches!(
            open_bytes(&bigtiff),
            Err(RenderError::Unsupported(_))
        ));
    }

    #[test]
    fn a_file_with_no_tiled_directory_is_rejected() {
        let fixture = TiffFixture::builder().drop_tile_directory(true).build();
        assert!(matches!(
            open_bytes(&fixture.bytes),
            Err(RenderError::Decode(_))
        ));
    }

    #[test]
    fn a_backward_ifd_pointer_is_followed_rather_than_dropped() {
        // TIFF permits a next-IFD pointer that goes *backwards*; only a cycle is
        // forbidden. A parser that required monotonic offsets would return a
        // single-level pyramid here, and the symptom would surface much later as
        // "this file needs overview levels" at low zoom.
        let fixture = TiffFixture::builder().overview_first(true).build();
        let metadata = open_bytes(&fixture.bytes).expect("the fixture must open");
        assert_eq!(metadata.level_count(), 2);
        assert_eq!(metadata.base_level().expect("a base level").width, 8);
        assert_eq!(metadata.level(1).expect("an overview").width, 4);
    }

    #[test]
    fn a_self_referential_ifd_chain_terminates() {
        let fixture = TiffFixture::builder().self_referential_chain(true).build();
        let metadata = open_bytes(&fixture.bytes).expect("the chain walk must terminate");
        assert_eq!(metadata.level_count(), 1);
    }

    #[test]
    fn an_ungeoreferenced_tiff_still_opens() {
        let fixture = TiffFixture::builder().georeference(false).build();
        let metadata = open_bytes(&fixture.bytes).expect("a plain TIFF must still open");
        assert_eq!(metadata.epsg, None);
        assert!(matches!(
            metadata.geo_transform(),
            Err(RenderError::Unsupported(_))
        ));
        assert!(matches!(metadata.crs(), Err(RenderError::Unsupported(_))));
    }

    #[test]
    fn a_default_open_matches_new() {
        let open = CogOpen::default();
        assert_eq!(open.blocks_supplied(), 0);
        assert_eq!(open.bytes_supplied(), 0);
        assert!(open.metadata().is_none());
    }

    #[test]
    fn a_bigtiff_opens_with_its_wide_offsets_and_counts() {
        // GDAL's COG driver switches to BigTIFF once the output would pass
        // 4 GB, which is routine for a national mosaic; every offset, count and
        // IFD entry is wider, so this exercises the whole parse again.
        let fixture = TiffFixture::builder().big_tiff(true).build();
        assert_eq!(fixture.bytes.get(2..4), Some(&[43u8, 0][..]));
        let metadata = open_bytes(&fixture.bytes).expect("a BigTIFF COG must open");
        assert_eq!(metadata.level_count(), 2);
        assert_eq!(metadata.epsg, Some(4326));
        let base = metadata.base_level().expect("a base level");
        assert_eq!((base.width, base.height), (8, 8));
        assert_eq!((base.tile_width, base.tile_height), (4, 4));
        assert_eq!(base.tile_offsets.len(), 4);
        assert_eq!(base.tile_byte_counts.len(), 4);
        let geo = metadata.geo_transform().expect("a georeference");
        assert!((geo.origin_x - 10.0).abs() < 1e-12);
    }

    #[test]
    fn a_big_endian_bigtiff_opens() {
        let fixture = TiffFixture::builder()
            .big_tiff(true)
            .big_endian(true)
            .build();
        let metadata = open_bytes(&fixture.bytes).expect("an MM BigTIFF must open");
        assert!(!metadata.little_endian);
        assert_eq!(metadata.base_level().expect("a base level").width, 8);
    }

    #[test]
    fn a_striped_tiff_becomes_a_one_column_block_grid() {
        // 8x10 at 4 rows per strip: three strips, the last of which is short.
        let fixture = TiffFixture::builder().striped(4, 10).build();
        let metadata = open_bytes(&fixture.bytes).expect("a striped TIFF must open");
        let base = metadata.base_level().expect("a base level");
        assert_eq!((base.tile_width, base.tile_height), (8, 4));
        assert_eq!(base.tiles_across(), 1);
        assert_eq!(base.tiles_down(), 3);
        assert_eq!(base.block_rows(0), 4);
        assert_eq!(base.block_rows(2), 2);
        // The short final strip's payload is two rows, not four.
        assert_eq!(base.tile_byte_counts.get(2).copied(), Some(16));
    }

    #[test]
    fn gdal_nodata_is_parsed_from_its_ascii_text() {
        let fixture = TiffFixture::builder().nodata("-9999").build();
        let mut open = CogOpen::new();
        for _ in 0..32 {
            match open.poll().expect("the parse must not fail") {
                CogOpenProgress::Need(range) => {
                    let start = range.start as usize;
                    let end = (range.end as usize).min(fixture.bytes.len());
                    let slice = fixture.bytes.get(start..end).unwrap_or(&[]).to_vec();
                    open.supply(range.start, slice);
                }
                CogOpenProgress::Ready(_) => break,
            }
        }
        assert_eq!(open.decode_options().nodata, Some(-9999.0));

        // A file that declares none reports none.
        let plain = tiled_geo_tiff();
        let mut open = CogOpen::new();
        for _ in 0..32 {
            match open.poll().expect("the parse must not fail") {
                CogOpenProgress::Need(range) => {
                    let start = range.start as usize;
                    let end = (range.end as usize).min(plain.bytes.len());
                    let slice = plain.bytes.get(start..end).unwrap_or(&[]).to_vec();
                    open.supply(range.start, slice);
                }
                CogOpenProgress::Ready(_) => break,
            }
        }
        assert_eq!(open.decode_options().nodata, None);
        assert!(open.decode_options().jpeg_tables.is_empty());
    }

    #[test]
    fn nodata_text_is_parsed_leniently_and_never_fatally() {
        assert_eq!(super::parse_gdal_nodata(b"-9999\0"), Some(-9999.0));
        assert_eq!(super::parse_gdal_nodata(b"  0 \0\0"), Some(0.0));
        assert_eq!(super::parse_gdal_nodata(b"1e6"), Some(1e6));
        assert!(super::parse_gdal_nodata(b"nan\0").is_some_and(f64::is_nan));
        assert_eq!(super::parse_gdal_nodata(b"not a number"), None);
        assert_eq!(super::parse_gdal_nodata(&[0xFF, 0xFE]), None);
    }

    #[test]
    fn a_long_typed_scalar_is_read_by_its_field_type() {
        use crate::cog::tiff::{IFD_MAX_INLINE_VALUE_BYTES, IfdEntry, TiffVariant};

        // TIFF left-justifies an inline value, so reading a big-endian LONG as
        // if it were a SHORT takes the *high* half: SamplesPerPixel 3 reads
        // back as 0, which zeroes every derived size downstream.
        let mut value = [0u8; IFD_MAX_INLINE_VALUE_BYTES];
        value[..4].copy_from_slice(&3u32.to_be_bytes());
        let entry = IfdEntry {
            tag: tiff::TAG_SAMPLES_PER_PIXEL,
            field_type: 4,
            count: 1,
            value,
            variant: TiffVariant::Classic,
        };
        assert_eq!(super::read_scalar(&entry, ByteOrder::BigEndian), Some(3));
        assert_eq!(ByteOrder::BigEndian.short(&entry.value), Some(0));
    }

    #[test]
    fn a_block_geometry_past_the_cap_is_refused_at_parse_time() {
        // Rewrite TileWidth/TileLength to 65535 in place: 65535 x 65535 bytes
        // is 4 GiB, which the reader must refuse before reserving it.
        let mut bytes = tiled_geo_tiff().bytes;
        for tag in [322u16, 323] {
            let mut needle = Vec::new();
            needle.extend_from_slice(&tag.to_le_bytes());
            needle.extend_from_slice(&3u16.to_le_bytes());
            needle.extend_from_slice(&1u32.to_le_bytes());
            needle.extend_from_slice(&4u32.to_le_bytes());
            let position = bytes
                .windows(needle.len())
                .position(|window| window == needle.as_slice())
                .expect("the fixture must declare the tag inline");
            let value = position + 8;
            bytes[value] = 0xFF;
            bytes[value + 1] = 0xFF;
        }
        assert!(matches!(
            open_bytes(&bytes),
            Err(RenderError::Unsupported(_))
        ));
    }

    #[test]
    fn a_tag_declaring_more_values_than_its_kind_allows_is_refused() {
        // A BYTE-typed ModelPixelScale with a 33 M count passed the old global
        // byte ceiling, reserved 268 MB of u64 and asked the server for a 33 MB
        // range to answer a twelve-byte tag.
        let mut bytes = tiled_geo_tiff().bytes;
        let mut needle = Vec::new();
        needle.extend_from_slice(&33_550u16.to_le_bytes());
        needle.extend_from_slice(&12u16.to_le_bytes());
        needle.extend_from_slice(&3u32.to_le_bytes());
        let position = bytes
            .windows(needle.len())
            .position(|window| window == needle.as_slice())
            .expect("the fixture must declare ModelPixelScale");
        bytes[position + 2..position + 4].copy_from_slice(&1u16.to_le_bytes());
        bytes[position + 4..position + 8].copy_from_slice(&33_554_432u32.to_le_bytes());
        assert!(matches!(open_bytes(&bytes), Err(RenderError::Decode(_))));
    }
}
