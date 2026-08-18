//! PMTiles v3 reading: header, directory codec, Hilbert addressing, open.
//!
//! A PMTiles archive is a single file holding a whole tile pyramid: a
//! 127-byte header, a root directory, a metadata blob, an optional block of
//! leaf directories and the tile bodies. Every tile is addressed by one `u64`
//! — the Hilbert index of `(z, x, y)` — so a lookup is a binary search inside
//! a directory rather than a path join, and the spec *requires* the header and
//! the root directory to live inside the first 16 KiB. One speculative read
//! therefore opens any archive, which is the same trick [`crate::cog`]'s
//! [`CogOpen`] plays with its header block.
//!
//! [`CogOpen`]: crate::cog::CogOpen
//!
//! # Layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`header`] | the fixed 127-byte header, little-endian |
//! | [`directory`] | the five-varint-block directory codec and its lookup |
//! | [`hilbert`] | `(z, x, y)` ⇄ tile id |
//! | [`open`] | the pull-based state machine that opens an archive |
//! | [`archive`] | the open archive: `find`, the offset bases, `info` |
//!
//! # Driving an open
//!
//! ```text
//! let mut open = PmtilesOpen::new();
//! loop {
//!     match open.poll()? {
//!         PmtilesOpenProgress::Need(range) => {
//!             let bytes = /* caller reads `range` however it likes */;
//!             open.supply(range.start, bytes)?;
//!         }
//!         PmtilesOpenProgress::NeedPlain { slot, compression, raw } => {
//!             // The caller owns the codec (see below).
//!             open.supply_plain(slot, inflate(compression, raw)?)?;
//!         }
//!         PmtilesOpenProgress::Ready(archive) => break archive,
//!     }
//! }
//! ```
//!
//! # Why this crate does not inflate
//!
//! A PMTiles header carries **two independent** compression bytes:
//! `internal_compression` for the directories and the metadata, and
//! `tile_compression` for the tile bodies. They genuinely differ in the wild
//! (a measured raster archive is gzip-internal with uncompressed PNG tiles),
//! so the bytes must be honoured and never sniffed.
//!
//! Honouring them is nonetheless the *caller's* job. `oxigis-render` keeps a
//! no-I/O, no-codec contract so it stays a portable rendering crate, and the
//! UI shell already owns a gzip implementation. [`open::PmtilesOpen`]
//! therefore hands the raw root/metadata bytes back with their compression tag
//! ([`open::PmtilesOpenProgress::NeedPlain`]) and waits for the plain bytes.
//! When `internal_compression` is [`header::Compression::None`] the raw bytes
//! *are* the plain bytes and that step is skipped entirely, which is why the
//! offline fixtures (the `fixture` module, behind the `fixtures` feature)
//! default to an uncompressed archive: the whole parse is exercised without a
//! codec anywhere near it.
//!
//! # Refusals
//!
//! Every rejection is a named [`PmtilesError`] rather than a silently wrong
//! answer, following `gpkg_input`'s untrusted-input posture: a PMTiles archive
//! is a remote file whose directories are attacker-controlled offsets, so
//! every emitted [`ByteRange`] is bounds-checked against the header's own
//! regions first. Brotli- and zstd-compressed archives are refused *by name at
//! open* — a policy refusal (both codecs are banned by `deny.toml`), not a
//! gap — instead of failing once per tile.
//!
//! [`ByteRange`]: crate::source::ByteRange

pub mod archive;
pub mod directory;
pub mod header;
pub mod hilbert;
pub mod open;

// Fixture generation uses `expect` while assembling bytes it has just built
// itself; a failure there is a bug in the fixture, not a runtime condition.
// Same arrangement as `cog::fixture`.
#[cfg(any(test, feature = "fixtures"))]
#[allow(clippy::expect_used)]
pub mod fixture;

use thiserror::Error;

pub use crate::pmtiles::archive::{
    MAX_ARCHIVE_TILE_BYTES, MAX_DIRECTORY_BYTES, MAX_DIRECTORY_DEPTH, PmtilesArchive, PmtilesInfo,
    TileLookup,
};
#[cfg(any(test, feature = "fixtures"))]
pub use crate::pmtiles::directory::serialize_directory;
pub use crate::pmtiles::directory::{
    DirEntry, DirLookup, MAX_DIRECTORY_ENTRIES, MAX_DIRECTORY_INFLATED_BYTES,
    deserialize_directory, lookup,
};
#[cfg(any(test, feature = "fixtures"))]
pub use crate::pmtiles::fixture::{
    PmtilesBuilder, sample_pmtiles_far_metadata, sample_pmtiles_leafed, sample_pmtiles_raster,
    sample_pmtiles_vector,
};
pub use crate::pmtiles::header::{
    Compression, HEADER_LEN, MAX_METADATA_BYTES, MAX_PMTILES_ZOOM, PMTILES_MAGIC, PMTILES_VERSION,
    PREFETCH_LEN, PmtilesHeader, TileType,
};
pub use crate::pmtiles::hilbert::{MAX_TILE_ID, MAX_TILE_ID_ZOOM, tile_id_to_zxy, zxy_to_tile_id};
pub use crate::pmtiles::open::{PlainSlot, PmtilesOpen, PmtilesOpenProgress};

/// Everything the PMTiles reader refuses, each by name.
///
/// Deliberately fine-grained where [`crate::error::RenderError`] is coarse:
/// these variants are what a shell turns into an actionable status line ("the
/// root directory is not inside the first 16 KiB, which PMTiles v3 requires"),
/// and what a test asserts on so a regression cannot hide behind a generic
/// decode failure.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PmtilesError {
    /// The first seven bytes were not `PMTiles`.
    #[error("not a PMTiles archive: the file does not begin with the magic \"PMTiles\"")]
    BadMagic,

    /// The archive declares a spec version this reader does not implement.
    #[error("PMTiles v{0} is not supported; only v3")]
    UnsupportedVersion(
        /// The version byte found at offset 7.
        u8,
    ),

    /// A structure ran past the end of the bytes supplied for it.
    #[error("truncated {context}: needed {needed} bytes, have {available}")]
    Truncated {
        /// What was being read, e.g. `"header"` or `"directory offsets"`.
        context: &'static str,
        /// How many bytes the structure required.
        needed: u64,
        /// How many bytes were actually available.
        available: u64,
    },

    /// An enum-valued header byte held a value outside the spec's range.
    #[error("unknown {field} value {value}")]
    UnknownEnum {
        /// Name of the header field, e.g. `"tile_type"`.
        field: &'static str,
        /// The unrecognised byte.
        value: u8,
    },

    /// The root directory does not fit inside the speculative prefetch.
    ///
    /// PMTiles v3 mandates that the header and the root directory live inside
    /// the first [`PREFETCH_LEN`] bytes; that mandate is what makes a
    /// one-round-trip open sound, so it is enforced rather than worked around.
    #[error(
        "the root directory ends at {end}, not inside the first {limit} bytes, \
         which PMTiles v3 requires"
    )]
    RootOutsidePrefetch {
        /// One past the last byte of the declared root directory.
        end: u64,
        /// The limit the spec sets, i.e. [`PREFETCH_LEN`].
        limit: u64,
    },

    /// A header field was self-contradictory or degenerate.
    #[error("invalid PMTiles header field {field}: {reason}")]
    InvalidHeader {
        /// Name of the offending field.
        field: &'static str,
        /// Why it was rejected.
        reason: &'static str,
    },

    /// A byte range computed from the archive's own fields was empty or
    /// inverted, so no request could be built from it.
    #[error("invalid byte range {start}..{end} derived from the archive")]
    InvalidRange {
        /// First byte of the rejected range.
        start: u64,
        /// One past the last byte of the rejected range.
        end: u64,
    },

    /// The archive is coded with a compression this build will never decode.
    ///
    /// Brotli and zstd are banned by the workspace's `deny.toml`, so this is a
    /// policy refusal made once at open, not a per-tile failure.
    #[error("this PMTiles archive uses {} for its {field}, which OxiGIS does not read", .compression.name())]
    UnsupportedCompression {
        /// Which header byte named it: `"directories"` or `"tiles"`.
        field: &'static str,
        /// The refused codec.
        compression: Compression,
    },

    /// A varint ran past ten bytes or did not fit in a `u64`.
    #[error("varint overflow while reading {context}")]
    VarintOverflow {
        /// What was being read when the varint overflowed.
        context: &'static str,
    },

    /// A directory declared zero entries, which the format does not allow.
    #[error("the directory declares no entries")]
    EmptyDirectory,

    /// The first entry used the contiguous-offset form, which has no
    /// predecessor to be contiguous with.
    #[error(
        "the directory's first entry uses the contiguous-offset form, which has no predecessor"
    )]
    LeadingContiguousOffset,

    /// A directory decoded correctly but did not consume its whole buffer.
    ///
    /// Every real archive's directory consumes its buffer exactly, so trailing
    /// bytes mean the buffer is not what it claimed to be.
    #[error("{extra} bytes remain after the directory was decoded")]
    TrailingBytes {
        /// How many bytes were left over.
        extra: u64,
    },

    /// A declared entry count could not possibly be encoded in the buffer, or
    /// exceeded [`MAX_DIRECTORY_ENTRIES`].
    #[error("the directory declares {count} entries, past the budget of {budget}")]
    EntryCountExceedsBudget {
        /// The declared count.
        count: u64,
        /// The largest count that would have been accepted.
        budget: u64,
    },

    /// An inflated directory buffer was larger than
    /// [`MAX_DIRECTORY_INFLATED_BYTES`].
    #[error("the directory is {bytes} bytes, past the limit of {limit}")]
    DirectoryTooLarge {
        /// Size of the offered buffer.
        bytes: u64,
        /// The limit that was exceeded.
        limit: u64,
    },

    /// A directory field did not fit the width the format gives it.
    #[error("{field} value {value} is out of range")]
    FieldOutOfRange {
        /// Name of the field, e.g. `"run_length"`.
        field: &'static str,
        /// The offending value.
        value: u64,
    },

    /// A directory entry pointed outside the region its offsets are relative
    /// to, so following it would read outside the archive.
    #[error(
        "a directory entry addresses {start}..{end}, outside the {region} region \
         {region_start}..{region_end}"
    )]
    RangeOutOfBounds {
        /// Which region was overrun: `"tile data"` or `"leaf directory"`.
        region: &'static str,
        /// First byte the entry addressed.
        start: u64,
        /// One past the last byte the entry addressed.
        end: u64,
        /// First byte of the region.
        region_start: u64,
        /// One past the last byte of the region.
        region_end: u64,
    },

    /// A leaf-directory chain was followed further than
    /// [`MAX_DIRECTORY_DEPTH`] allows.
    #[error("the archive's leaf-directory chain is {depth} deep, past the limit of {limit}")]
    LeafChainTooDeep {
        /// How deep the chain went.
        depth: u32,
        /// The limit that was exceeded.
        limit: u32,
    },

    /// A `(z, x, y)` address was outside `0..2^z`.
    #[error("invalid tile address {z}/{x}/{y}: a coordinate is not below 2^z")]
    BadCoordinate {
        /// Zoom level of the rejected address.
        z: u8,
        /// Column of the rejected address.
        x: u32,
        /// Row of the rejected address.
        y: u32,
    },

    /// A zoom level was past what a PMTiles tile id can express.
    #[error("zoom {z} is out of range 0..={limit}")]
    ZoomOutOfRange {
        /// The rejected zoom level.
        z: u8,
        /// The highest zoom this reader addresses.
        limit: u8,
    },

    /// A tile id was past the last id of zoom [`MAX_TILE_ID_ZOOM`].
    #[error("tile id {id} is past the end of the Hilbert curve")]
    TileIdOutOfRange {
        /// The rejected id.
        id: u64,
    },

    /// Bytes were supplied for a different offset than the one requested.
    #[error("supplied bytes start at {actual} but {expected} was requested")]
    SupplyOffsetMismatch {
        /// The offset the state machine asked for.
        expected: u64,
        /// The offset the caller supplied.
        actual: u64,
    },

    /// The open state machine was driven out of order.
    #[error("PMTiles open driven out of order: {what}")]
    OpenOutOfOrder {
        /// What the caller did that the machine was not waiting for.
        what: &'static str,
    },

    /// The metadata block was not valid UTF-8, so it cannot be JSON.
    #[error("the archive's metadata block is not valid UTF-8")]
    MetadataNotUtf8,
}

#[cfg(test)]
mod tests {
    use super::PmtilesError;
    use crate::pmtiles::header::Compression;

    #[test]
    fn error_is_std_error() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&PmtilesError::BadMagic);
    }

    #[test]
    fn refusal_messages_name_the_problem() {
        assert_eq!(
            PmtilesError::UnsupportedVersion(2).to_string(),
            "PMTiles v2 is not supported; only v3"
        );
        assert_eq!(
            PmtilesError::RootOutsidePrefetch {
                end: 20_000,
                limit: 16_384,
            }
            .to_string(),
            "the root directory ends at 20000, not inside the first 16384 bytes, \
             which PMTiles v3 requires"
        );
        assert_eq!(
            PmtilesError::UnsupportedCompression {
                field: "tiles",
                compression: Compression::Brotli,
            }
            .to_string(),
            "this PMTiles archive uses brotli for its tiles, which OxiGIS does not read"
        );
    }
}
