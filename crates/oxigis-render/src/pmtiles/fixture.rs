//! Hand-built PMTiles v3 archives for the tests in this module and in
//! `oxigis-ui`.
//!
//! Test-only: there is no archive writer in `oxigis-render`, and depending on
//! one just to exercise the reader would pull a whole writing stack into the
//! graph. This assembles the bytes directly instead — the same arrangement
//! [`crate::cog::sample_cog_bytes`] uses — which also makes it possible to
//! produce archives a real writer would never emit: a metadata block parked
//! past the 16 KiB prefetch, a leaf level forced at two entries, a tile body
//! deliberately shared by four addresses.
//!
//! **The default archive is uncompressed** (`internal_compression = None`,
//! which the spec allows), so the whole parse is testable with no codec
//! anywhere near it. [`PmtilesBuilder::with_compression`] switches either byte
//! to gzip, which is written with `oxiarc-deflate` — already a dependency of
//! this crate for COG's DEFLATE tiles, so nothing new enters the graph.
//!
//! The strategy is proven end to end: a complete, valid 282-byte archive of
//! this shape was hand-built and read back by the same decoder that reads a
//! 137 GB planet build, entries comparing equal, metadata coming back, all
//! five z0/z1 lookups hitting and z2 correctly missing.

use crate::pmtiles::directory::{DirEntry, serialize_directory};
use crate::pmtiles::header::{Compression, HEADER_LEN, PMTILES_MAGIC, PMTILES_VERSION, TileType};
use crate::pmtiles::hilbert::zxy_to_tile_id;

/// The metadata every sample archive carries, unless overridden.
const DEFAULT_METADATA: &str =
    r#"{"name":"fixture","attribution":"OxiGIS test fixture","vector_layers":[{"id":"land"}]}"#;

/// Builds a PMTiles v3 archive byte by byte.
///
/// ```text
/// let mut builder = PmtilesBuilder::new(TileType::Mvt);
/// builder.push_tile(0, 0, 0, body);
/// let archive = builder.build();
/// ```
#[derive(Debug, Clone)]
pub struct PmtilesBuilder {
    /// What the tile bodies are.
    tile_type: TileType,
    /// Codec written into the directories and the metadata block.
    internal: Compression,
    /// Codec byte written for the tile bodies. Bodies are stored verbatim, so
    /// a caller wanting genuinely gzip tiles pushes already-gzip bodies.
    tile_compression: Compression,
    /// Entries per leaf directory; `0` keeps the archive root-only.
    leaf_threshold: usize,
    /// The metadata block's text.
    metadata: String,
    /// When set, the metadata block is written last, at or past this offset.
    metadata_after: Option<u64>,
    /// Pushed tiles, in insertion order.
    tiles: Vec<(u64, Vec<u8>)>,
    /// Lowest zoom pushed so far.
    min_zoom: Option<u8>,
    /// Highest zoom pushed so far.
    max_zoom: u8,
}

impl PmtilesBuilder {
    /// A builder for an uncompressed, root-only archive of `tile_type`.
    #[must_use]
    pub fn new(tile_type: TileType) -> Self {
        Self {
            tile_type,
            internal: Compression::None,
            tile_compression: Compression::None,
            leaf_threshold: 0,
            metadata: DEFAULT_METADATA.to_owned(),
            metadata_after: None,
            tiles: Vec::new(),
            min_zoom: None,
            max_zoom: 0,
        }
    }

    /// Sets the two compression bytes.
    ///
    /// `internal` genuinely gzips the root directory and the metadata block.
    /// `tile` only sets the header byte — tile bodies are stored exactly as
    /// they were pushed, so a test wanting gzip tiles pushes gzip bodies and
    /// declares them here, which is also how a real writer behaves.
    #[must_use]
    pub const fn with_compression(mut self, internal: Compression, tile: Compression) -> Self {
        self.internal = internal;
        self.tile_compression = tile;
        self
    }

    /// Forces a leaf directory every `entries` root entries.
    ///
    /// `0` (the default) keeps the archive root-only. Any smaller number than
    /// a real writer would use makes the leaf-hop path testable without
    /// building a multi-gigabyte file.
    #[must_use]
    pub const fn with_leaf_threshold(mut self, entries: usize) -> Self {
        self.leaf_threshold = entries;
        self
    }

    /// Replaces the metadata block's text; `""` writes no metadata at all.
    #[must_use]
    pub fn with_metadata(mut self, json: &str) -> Self {
        self.metadata = json.to_owned();
        self
    }

    /// Writes the metadata block last, at or past `offset`.
    ///
    /// Reproduces what a planet-scale writer does — a measured 137 GB archive
    /// parks its metadata at byte 136 805 991 502 — so the two-`Need` open
    /// path has a fixture.
    #[must_use]
    pub const fn with_metadata_after(mut self, offset: u64) -> Self {
        self.metadata_after = Some(offset);
        self
    }

    /// Adds one tile.
    ///
    /// Pushing the same address twice keeps the last body. Two addresses that
    /// share a body share one stored copy, which is what produces the run
    /// lengths and the contiguous-offset forms the decoder must handle.
    pub fn push_tile(&mut self, z: u8, x: u32, y: u32, body: Vec<u8>) {
        let id = zxy_to_tile_id(z, x, y).expect("the fixture pushes valid tile addresses");
        self.tiles.retain(|(existing, _)| *existing != id);
        self.tiles.push((id, body));
        self.min_zoom = Some(self.min_zoom.map_or(z, |current| current.min(z)));
        self.max_zoom = self.max_zoom.max(z);
    }

    /// Assembles the archive.
    #[must_use]
    pub fn build(mut self) -> Vec<u8> {
        self.tiles.sort_by_key(|(id, _)| *id);

        // Distinct bodies, in first-use order: identical bodies are stored
        // once, exactly as a real writer deduplicates them.
        let mut bodies: Vec<Vec<u8>> = Vec::new();
        let mut placement: Vec<(u64, u32)> = Vec::new();
        let mut tile_data: Vec<u8> = Vec::new();
        for (_, body) in &self.tiles {
            let existing = bodies.iter().position(|stored| stored == body);
            let index = existing.unwrap_or_else(|| {
                bodies.push(body.clone());
                tile_data.extend_from_slice(body);
                bodies.len() - 1
            });
            let offset: u64 = bodies
                .iter()
                .take(index)
                .map(|stored| u64::try_from(stored.len()).unwrap_or(0))
                .sum();
            let length = u32::try_from(bodies[index].len()).expect("a small fixture body");
            placement.push((offset, length));
        }

        // Entries, merging consecutive ids that share a body into one run.
        let mut entries: Vec<DirEntry> = Vec::new();
        for ((id, _), (offset, length)) in self.tiles.iter().zip(placement.iter()) {
            let merged = entries.last_mut().is_some_and(|last| {
                last.offset == *offset
                    && last.length == *length
                    && last.tile_id + u64::from(last.run_length) == *id
            });
            if merged {
                if let Some(last) = entries.last_mut() {
                    last.run_length += 1;
                }
            } else {
                entries.push(DirEntry {
                    tile_id: *id,
                    offset: *offset,
                    length: *length,
                    run_length: 1,
                });
            }
        }

        // Split into leaves when asked, then encode both levels.
        let (root_entries, leaf_bytes) = self.split_leaves(entries);
        let root_bytes = encode(&serialize_directory(&root_entries), self.internal);
        let metadata_bytes = if self.metadata.is_empty() {
            Vec::new()
        } else {
            encode(self.metadata.as_bytes(), self.internal)
        };

        let header_len = u64::try_from(HEADER_LEN).expect("127 fits u64");
        let root_offset = header_len;
        let root_len = len_of(&root_bytes);
        let mut out = Vec::new();
        let (metadata_offset, metadata_len, leaf_offset, tile_offset, tail);
        if let Some(after) = self.metadata_after {
            leaf_offset = root_offset + root_len;
            tile_offset = leaf_offset + len_of(&leaf_bytes);
            let end = tile_offset + len_of(&tile_data);
            let placed = after.max(end);
            metadata_offset = placed;
            metadata_len = len_of(&metadata_bytes);
            tail = usize::try_from(placed - end).unwrap_or(0);
        } else {
            metadata_offset = root_offset + root_len;
            metadata_len = len_of(&metadata_bytes);
            leaf_offset = metadata_offset + metadata_len;
            tile_offset = leaf_offset + len_of(&leaf_bytes);
            tail = 0;
        }

        out.extend_from_slice(&self.header_bytes(
            root_offset,
            root_len,
            metadata_offset,
            metadata_len,
            leaf_offset,
            len_of(&leaf_bytes),
            tile_offset,
            len_of(&tile_data),
            &root_entries,
            &bodies,
        ));
        out.extend_from_slice(&root_bytes);
        if self.metadata_after.is_none() {
            out.extend_from_slice(&metadata_bytes);
        }
        out.extend_from_slice(&leaf_bytes);
        out.extend_from_slice(&tile_data);
        if self.metadata_after.is_some() {
            out.extend(std::iter::repeat_n(0u8, tail));
            out.extend_from_slice(&metadata_bytes);
        }
        out
    }

    /// Chunks `entries` into leaf directories when a threshold is set.
    ///
    /// Returns the root entries (leaf pointers, `run_length = 0`) and the
    /// concatenated leaf-directory bytes.
    fn split_leaves(&self, entries: Vec<DirEntry>) -> (Vec<DirEntry>, Vec<u8>) {
        if self.leaf_threshold == 0 || entries.len() <= self.leaf_threshold {
            return (entries, Vec::new());
        }
        let mut root = Vec::new();
        let mut leaves = Vec::new();
        for chunk in entries.chunks(self.leaf_threshold) {
            let Some(first) = chunk.first() else { continue };
            let encoded = encode(&serialize_directory(chunk), self.internal);
            root.push(DirEntry {
                tile_id: first.tile_id,
                offset: len_of(&leaves),
                length: u32::try_from(encoded.len()).expect("a small fixture leaf"),
                run_length: 0,
            });
            leaves.extend_from_slice(&encoded);
        }
        (root, leaves)
    }

    /// Writes the 127-byte header.
    #[allow(clippy::too_many_arguments)]
    fn header_bytes(
        &self,
        root_offset: u64,
        root_len: u64,
        metadata_offset: u64,
        metadata_len: u64,
        leaf_offset: u64,
        leaf_len: u64,
        tile_offset: u64,
        tile_len: u64,
        root_entries: &[DirEntry],
        bodies: &[Vec<u8>],
    ) -> Vec<u8> {
        let mut head = vec![0u8; HEADER_LEN];
        head[..7].copy_from_slice(PMTILES_MAGIC);
        head[7] = PMTILES_VERSION;
        head[8..16].copy_from_slice(&root_offset.to_le_bytes());
        head[16..24].copy_from_slice(&root_len.to_le_bytes());
        head[24..32].copy_from_slice(&metadata_offset.to_le_bytes());
        head[32..40].copy_from_slice(&metadata_len.to_le_bytes());
        head[40..48].copy_from_slice(&leaf_offset.to_le_bytes());
        head[48..56].copy_from_slice(&leaf_len.to_le_bytes());
        head[56..64].copy_from_slice(&tile_offset.to_le_bytes());
        head[64..72].copy_from_slice(&tile_len.to_le_bytes());
        let addressed = u64::try_from(self.tiles.len()).unwrap_or(0);
        head[72..80].copy_from_slice(&addressed.to_le_bytes());
        let entries = u64::try_from(root_entries.len()).unwrap_or(0);
        head[80..88].copy_from_slice(&entries.to_le_bytes());
        let contents = u64::try_from(bodies.len()).unwrap_or(0);
        head[88..96].copy_from_slice(&contents.to_le_bytes());
        head[96] = 1; // clustered
        head[97] = self.internal.to_byte();
        head[98] = self.tile_compression.to_byte();
        head[99] = self.tile_type.to_byte();
        head[100] = self.min_zoom.unwrap_or(0);
        head[101] = self.max_zoom;
        head[102..106].copy_from_slice(&(-1_800_000_000i32).to_le_bytes());
        head[106..110].copy_from_slice(&(-850_511_287i32).to_le_bytes());
        head[110..114].copy_from_slice(&1_800_000_000i32.to_le_bytes());
        head[114..118].copy_from_slice(&850_511_287i32.to_le_bytes());
        head[118] = self.min_zoom.unwrap_or(0);
        head[119..123].copy_from_slice(&0i32.to_le_bytes());
        head[123..127].copy_from_slice(&0i32.to_le_bytes());
        head
    }
}

/// Applies the archive's internal compression to a block.
fn encode(data: &[u8], compression: Compression) -> Vec<u8> {
    match compression {
        Compression::Gzip => {
            oxiarc_deflate::gzip_compress(data, 6).expect("the fixture's gzip must compress")
        }
        // The refused codecs are never actually applied: the archives that
        // declare them exist only to be refused at open, before any block is
        // decoded.
        _ => data.to_vec(),
    }
}

/// Byte length of a block as a `u64`.
fn len_of(data: &[u8]) -> u64 {
    u64::try_from(data.len()).unwrap_or(0)
}

/// A tiny well-formed MVT-shaped body: protobuf field 3 (`layers`), a length
/// and a nested message. Enough for the reader; not a full tile.
fn mvt_body(tag: u8) -> Vec<u8> {
    vec![0x1a, 0x04, 0x0a, 0x02, tag, 0x00]
}

/// Bytes of an uncompressed MVT archive with five addresses over two zooms.
///
/// z0 0/0 has its own body; all four z1 tiles share one, so the root holds
/// exactly two entries — the second with `run_length = 4` and written in the
/// contiguous-offset form. Root-only: no leaf directories.
#[must_use]
pub fn sample_pmtiles_vector() -> Vec<u8> {
    let mut builder = PmtilesBuilder::new(TileType::Mvt);
    builder.push_tile(0, 0, 0, vec![0x1a, 0x02, 0x0a, 0x00]);
    let shared = mvt_body(b'a');
    builder.push_tile(1, 0, 0, shared.clone());
    builder.push_tile(1, 0, 1, shared.clone());
    builder.push_tile(1, 1, 1, shared.clone());
    builder.push_tile(1, 1, 0, shared);
    builder.build()
}

/// Bytes of an uncompressed PNG archive with five distinct tiles.
///
/// `tile_compression = None`, which is what real raster archives do — the
/// bodies are already-compressed images. The bodies are genuine 2×2 PNGs, so
/// a consumer's decode path can be exercised against this fixture too.
#[must_use]
pub fn sample_pmtiles_raster() -> Vec<u8> {
    let mut builder = PmtilesBuilder::new(TileType::Png);
    builder.push_tile(0, 0, 0, tiny_png([220, 40, 40]));
    builder.push_tile(1, 0, 0, tiny_png([40, 220, 40]));
    builder.push_tile(1, 0, 1, tiny_png([40, 40, 220]));
    builder.push_tile(1, 1, 1, tiny_png([220, 220, 40]));
    builder.push_tile(1, 1, 0, tiny_png([40, 220, 220]));
    builder.build()
}

/// Bytes of an uncompressed MVT archive whose root holds only leaf pointers.
///
/// Five distinct bodies and a leaf threshold of two, so the root has three
/// leaf entries and every lookup costs a real second hop through
/// [`crate::pmtiles::PmtilesArchive::find_in_leaf`].
#[must_use]
pub fn sample_pmtiles_leafed() -> Vec<u8> {
    let mut builder = PmtilesBuilder::new(TileType::Mvt).with_leaf_threshold(2);
    builder.push_tile(0, 0, 0, mvt_body(b'0'));
    builder.push_tile(1, 0, 0, mvt_body(b'1'));
    builder.push_tile(1, 0, 1, mvt_body(b'2'));
    builder.push_tile(1, 1, 1, mvt_body(b'3'));
    builder.push_tile(1, 1, 0, mvt_body(b'4'));
    builder.build()
}

/// Bytes of an archive whose metadata sits past the 16 KiB prefetch.
///
/// Forces the two-round-trip open path that a planet-scale archive takes.
#[must_use]
pub fn sample_pmtiles_far_metadata() -> Vec<u8> {
    let mut builder = PmtilesBuilder::new(TileType::Mvt)
        .with_metadata_after(crate::pmtiles::header::PREFETCH_LEN + 64);
    builder.push_tile(0, 0, 0, mvt_body(b'z'));
    builder.build()
}

/// A genuine 2×2 RGB PNG of one solid colour.
///
/// Hand-assembled rather than encoded: `image`'s encoder is behind an optional
/// feature this fixture must not depend on, while `oxiarc-deflate`'s zlib is
/// already a dependency of this crate.
fn tiny_png(rgb: [u8; 3]) -> Vec<u8> {
    const WIDTH: u32 = 2;
    const HEIGHT: u32 = 2;

    // One scanline is a `0` filter byte followed by WIDTH RGB triples; every
    // row of a solid-colour image is identical, so it is built once.
    let mut scanline = vec![0u8];
    for _ in 0..WIDTH {
        scanline.extend_from_slice(&rgb);
    }
    let raw = scanline.repeat(usize::try_from(HEIGHT).unwrap_or(0));
    let idat = oxiarc_deflate::zlib_compress(&raw, 6).expect("the fixture's zlib must compress");

    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&WIDTH.to_be_bytes());
    ihdr.extend_from_slice(&HEIGHT.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, truecolour, no interlace

    let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    push_png_chunk(&mut png, b"IHDR", &ihdr);
    push_png_chunk(&mut png, b"IDAT", &idat);
    push_png_chunk(&mut png, b"IEND", &[]);
    png
}

/// Appends one length-type-data-CRC PNG chunk.
fn push_png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    let length = u32::try_from(data.len()).expect("a small fixture chunk");
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc = crc32(kind);
    crc = crc32_continue(crc, data);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// CRC-32 (IEEE) of `data`, starting a fresh checksum.
fn crc32(data: &[u8]) -> u32 {
    crc32_continue(0, data)
}

/// Continues a CRC-32 (IEEE) checksum with more data.
///
/// Bitwise rather than table-driven: the fixture checksums a few dozen bytes.
fn crc32_continue(previous: u32, data: &[u8]) -> u32 {
    let mut register = !previous;
    for &byte in data {
        register ^= u32::from(byte);
        for _ in 0..8 {
            let carry = register & 1 == 1;
            register >>= 1;
            if carry {
                register ^= 0xedb8_8320;
            }
        }
    }
    !register
}

#[cfg(test)]
mod tests {
    use super::{
        PmtilesBuilder, sample_pmtiles_far_metadata, sample_pmtiles_leafed, sample_pmtiles_raster,
        sample_pmtiles_vector, tiny_png,
    };
    use crate::pmtiles::directory::deserialize_directory;
    use crate::pmtiles::header::{Compression, PREFETCH_LEN, PmtilesHeader, TileType};

    /// The bytes of a range inside an archive.
    fn slice(archive: &[u8], offset: u64, length: u64) -> Vec<u8> {
        let start = usize::try_from(offset).expect("a small fixture");
        let end = start + usize::try_from(length).expect("a small fixture");
        archive.get(start..end).unwrap_or_default().to_vec()
    }

    #[test]
    fn the_vector_sample_has_two_entries_one_of_which_is_a_run_of_four() {
        let archive = sample_pmtiles_vector();
        let header = PmtilesHeader::parse(&archive).expect("well-formed");
        let root = deserialize_directory(&slice(&archive, header.root.start, header.root.len()))
            .expect("a well-formed root");
        assert_eq!(root.len(), 2);
        assert_eq!(root[0].tile_id, 0);
        assert_eq!(root[0].run_length, 1);
        assert_eq!(root[1].tile_id, 1);
        assert_eq!(root[1].run_length, 4, "all four z1 tiles share one body");
        assert_eq!(
            root[1].offset,
            root[0].offset + u64::from(root[0].length),
            "the second entry is contiguous, so it is written as 0"
        );
        assert_eq!(header.addressed_tiles, 5);
        assert_eq!(header.tile_contents, 2);
        assert_eq!(header.leaf_dirs_len, 0, "root-only archive");
    }

    #[test]
    fn the_whole_vector_sample_is_a_few_hundred_bytes() {
        let archive = sample_pmtiles_vector();
        assert!(
            (150..400).contains(&archive.len()),
            "unexpected fixture size {}",
            archive.len()
        );
        assert_eq!(
            u64::try_from(archive.len()).expect("small"),
            PmtilesHeader::parse(&archive)
                .expect("well-formed")
                .tile_data_end()
        );
    }

    #[test]
    fn the_leafed_sample_has_a_root_of_leaf_pointers_only() {
        let archive = sample_pmtiles_leafed();
        let header = PmtilesHeader::parse(&archive).expect("well-formed");
        let root = deserialize_directory(&slice(&archive, header.root.start, header.root.len()))
            .expect("a well-formed root");
        assert_eq!(root.len(), 3, "five entries at a threshold of two");
        assert!(root.iter().all(|entry| entry.run_length == 0));
        assert!(header.leaf_dirs_len > 0);

        // Every leaf decodes, and the entries add up to the five tiles.
        let mut total = 0usize;
        for entry in &root {
            let bytes = slice(
                &archive,
                header.leaf_dirs_offset + entry.offset,
                u64::from(entry.length),
            );
            let leaf = deserialize_directory(&bytes).expect("a well-formed leaf");
            total += leaf.len();
        }
        assert_eq!(total, 5);
    }

    #[test]
    fn the_raster_sample_declares_png_with_uncompressed_bodies() {
        let archive = sample_pmtiles_raster();
        let header = PmtilesHeader::parse(&archive).expect("well-formed");
        assert_eq!(header.tile_type, TileType::Png);
        assert_eq!(header.tile_compression, Compression::None);
        assert_eq!(header.tile_contents, 5, "five distinct colours");
    }

    #[test]
    fn the_far_metadata_sample_puts_its_metadata_past_the_prefetch() {
        let archive = sample_pmtiles_far_metadata();
        let header = PmtilesHeader::parse(&archive).expect("well-formed");
        assert!(header.metadata_offset > PREFETCH_LEN);
        assert!(header.root.end <= PREFETCH_LEN, "the root still fits");
        let metadata = slice(&archive, header.metadata_offset, header.metadata_length);
        assert!(
            String::from_utf8(metadata)
                .expect("utf8")
                .contains("fixture")
        );
    }

    #[test]
    fn a_gzip_archive_stores_gzip_directory_bytes() {
        let mut builder = PmtilesBuilder::new(TileType::Mvt)
            .with_compression(Compression::Gzip, Compression::None);
        builder.push_tile(0, 0, 0, vec![1, 2, 3]);
        let archive = builder.build();
        let header = PmtilesHeader::parse(&archive).expect("well-formed");
        let root = slice(&archive, header.root.start, header.root.len());
        assert_eq!(&root[..3], &[0x1f, 0x8b, 0x08], "gzip magic");
        let plain = oxiarc_deflate::gzip_decompress(&root).expect("a gzip block");
        assert_eq!(deserialize_directory(&plain).expect("well-formed").len(), 1);
    }

    #[test]
    fn pushing_the_same_address_twice_keeps_the_last_body() {
        let mut builder = PmtilesBuilder::new(TileType::Mvt);
        builder.push_tile(0, 0, 0, vec![1, 1, 1, 1]);
        builder.push_tile(0, 0, 0, vec![2, 2]);
        let archive = builder.build();
        let header = PmtilesHeader::parse(&archive).expect("well-formed");
        assert_eq!(header.addressed_tiles, 1);
        assert_eq!(header.tile_data_len, 2);
    }

    #[test]
    fn the_tiny_png_is_a_well_formed_png_stream() {
        let png = tiny_png([1, 2, 3]);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
    }

    #[cfg(feature = "decode")]
    #[test]
    fn the_raster_samples_bodies_actually_decode() {
        let decoded = crate::decode::decode_tile(&tiny_png([220, 40, 40]))
            .expect("the fixture writes a real PNG");
        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 2);
    }
}
