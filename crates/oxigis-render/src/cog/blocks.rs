//! The fetched-bytes store the COG parser reads through.
//!
//! A Cloud-Optimized GeoTIFF is parsed out of a handful of byte ranges rather
//! than out of a file, and `oxigis-render` performs no I/O, so the parser reads
//! through this store: a read either hits bytes somebody already supplied, or
//! reports the [`ByteRange`] that has to be fetched before the parse can
//! continue.
//!
//! # Why reads may be satisfied by *more* bytes than were asked for
//!
//! Every COG writer in practice (GDAL's `COG` driver, `rio-cogeo`, …) packs the
//! whole IFD chain — including the `TileOffsets`/`TileByteCounts` arrays of
//! every overview level — into a contiguous header block at the start of the
//! file. Requesting [`HEADER_PREFETCH_BYTES`] once therefore usually answers
//! *all* header reads, turning what would be 10–30 sequential round trips into
//! one. Fetching a superset of a needed range is always safe: reads are
//! satisfied from whichever supplied block contains them.
//!
//! # Short reads
//!
//! A range that runs past the end of the file comes back short, which is
//! expected for the prefetch above and must not be treated as an error. The
//! store therefore records the bytes it actually received *and* the ranges that
//! were requested: a read that misses inside an already-requested range is a
//! genuine truncation ([`BlockMiss::Truncated`]) rather than a request to fetch
//! again, which is what stops a malformed file from looping forever.

use crate::error::RenderError;
use crate::source::ByteRange;

/// Size of the speculative header read performed before any parsing.
///
/// 64 KiB comfortably covers the IFD chain and tile directories of a COG with
/// a full overview pyramid; anything beyond it is fetched on demand.
pub const HEADER_PREFETCH_BYTES: u64 = 64 * 1024;

/// Minimum size of a fetch request emitted for a read that missed.
///
/// Rounding a small miss (say the four bytes of a next-IFD pointer) up to this
/// avoids a round trip per field when a header turns out to be larger than
/// [`HEADER_PREFETCH_BYTES`].
pub const MIN_FETCH_BYTES: u64 = 16 * 1024;

/// Why a read could not be satisfied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockMiss {
    /// The bytes have not been requested yet; fetch this range and supply it.
    Fetch(ByteRange),
    /// The bytes were requested but the resource did not return them, so the
    /// file is shorter than its own offsets claim.
    Truncated(ByteRange),
}

impl BlockMiss {
    /// Renders the miss as a fatal decode error.
    ///
    /// Only meaningful for [`BlockMiss::Truncated`]; a [`BlockMiss::Fetch`]
    /// converted this way records that the caller gave up rather than fetching.
    #[must_use]
    pub fn into_error(self) -> RenderError {
        match self {
            Self::Fetch(range) => RenderError::Fetch(format!(
                "COG parse needs bytes {}..{} which were never fetched",
                range.start, range.end
            )),
            Self::Truncated(range) => RenderError::Decode(format!(
                "COG is truncated: bytes {}..{} are past the end of the resource",
                range.start, range.end
            )),
        }
    }
}

/// Byte ranges of a COG that have been fetched, indexed for random reads.
#[derive(Debug, Clone, Default)]
pub struct ByteBlocks {
    /// `(start, bytes)` pairs, in supply order. A COG open supplies one or two
    /// blocks in practice, so a linear scan is cheaper than any index.
    blocks: Vec<(u64, Vec<u8>)>,
    /// Ranges that were asked for, whether or not they came back in full.
    requested: Vec<ByteRange>,
}

impl ByteBlocks {
    /// An empty store.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            blocks: Vec::new(),
            requested: Vec::new(),
        }
    }

    /// Records that `range` has been requested, before its bytes arrive.
    pub fn note_requested(&mut self, range: ByteRange) {
        self.requested.push(range);
    }

    /// Stores `bytes`, which begin at file offset `start`.
    ///
    /// Empty blocks are ignored, so a zero-length response cannot make a read
    /// look satisfiable.
    pub fn supply(&mut self, start: u64, bytes: Vec<u8>) {
        if bytes.is_empty() {
            return;
        }
        self.blocks.push((start, bytes));
    }

    /// Number of supplied blocks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether nothing has been supplied yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Total number of bytes held.
    #[must_use]
    pub fn bytes_held(&self) -> usize {
        self.blocks.iter().map(|(_, bytes)| bytes.len()).sum()
    }

    /// Reads `len` bytes at file offset `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`BlockMiss::Fetch`] with the range to fetch when the bytes are
    /// absent, or [`BlockMiss::Truncated`] when they were requested and did not
    /// arrive. A zero-length read always succeeds with an empty slice.
    pub fn read(&self, offset: u64, len: usize) -> Result<&[u8], BlockMiss> {
        if len == 0 {
            return Ok(&[]);
        }
        for (start, bytes) in &self.blocks {
            if offset < *start {
                continue;
            }
            let Ok(local) = usize::try_from(offset - *start) else {
                continue;
            };
            if let Some(slice) = bytes.get(local..).and_then(|tail| tail.get(..len)) {
                return Ok(slice);
            }
        }
        let end = offset.saturating_add(len as u64);
        let wanted = ByteRange { start: offset, end };
        if self
            .requested
            .iter()
            .any(|range| range.start <= offset && end <= range.end)
        {
            return Err(BlockMiss::Truncated(wanted));
        }
        let padded_end = offset.saturating_add(u64::max(len as u64, MIN_FETCH_BYTES));
        Err(BlockMiss::Fetch(ByteRange {
            start: offset,
            end: padded_end.max(end),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockMiss, ByteBlocks, MIN_FETCH_BYTES};
    use crate::source::ByteRange;

    #[test]
    fn a_read_inside_a_supplied_block_hits() {
        let mut blocks = ByteBlocks::new();
        blocks.supply(100, (0u8..50).collect());
        assert_eq!(blocks.read(100, 4), Ok(&[0, 1, 2, 3][..]));
        assert_eq!(blocks.read(110, 2), Ok(&[10, 11][..]));
        assert_eq!(blocks.read(149, 1), Ok(&[49][..]));
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks.bytes_held(), 50);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn a_zero_length_read_always_succeeds() {
        let blocks = ByteBlocks::new();
        assert_eq!(blocks.read(12_345, 0), Ok(&[][..]));
        assert!(blocks.is_empty());
    }

    #[test]
    fn an_unfetched_read_reports_a_padded_range() {
        let blocks = ByteBlocks::new();
        let Err(BlockMiss::Fetch(range)) = blocks.read(1_000, 4) else {
            panic!("an unfetched read must ask for bytes");
        };
        assert_eq!(range.start, 1_000);
        assert_eq!(range.len(), MIN_FETCH_BYTES);
    }

    #[test]
    fn a_read_larger_than_the_minimum_is_not_shrunk() {
        let blocks = ByteBlocks::new();
        let huge = (MIN_FETCH_BYTES as usize) * 3;
        let Err(BlockMiss::Fetch(range)) = blocks.read(0, huge) else {
            panic!("an unfetched read must ask for bytes");
        };
        assert_eq!(range.len(), huge as u64);
    }

    #[test]
    fn a_read_that_straddles_two_blocks_misses() {
        let mut blocks = ByteBlocks::new();
        blocks.supply(0, vec![0; 16]);
        blocks.supply(16, vec![1; 16]);
        // Contiguous blocks are still separate: the parser only ever reads a
        // field out of one supplied range, and the prefetch keeps that true.
        assert!(blocks.read(8, 16).is_err());
        assert!(blocks.read(8, 8).is_ok());
    }

    #[test]
    fn a_requested_but_undelivered_read_is_truncation() {
        let mut blocks = ByteBlocks::new();
        blocks.note_requested(ByteRange { start: 0, end: 100 });
        blocks.supply(0, vec![0; 40]);
        let Err(BlockMiss::Truncated(range)) = blocks.read(60, 8) else {
            panic!("bytes inside a satisfied request must report truncation");
        };
        assert_eq!(range, ByteRange { start: 60, end: 68 });
        // …and outside it, a fetch is still the right answer.
        assert!(matches!(blocks.read(200, 8), Err(BlockMiss::Fetch(_))));
    }

    #[test]
    fn an_empty_supply_is_ignored() {
        let mut blocks = ByteBlocks::new();
        blocks.supply(0, Vec::new());
        assert!(blocks.is_empty());
        assert!(blocks.read(0, 1).is_err());
    }

    #[test]
    fn misses_render_as_errors() {
        let fetch = BlockMiss::Fetch(ByteRange { start: 0, end: 8 }).into_error();
        assert!(fetch.to_string().contains("never fetched"));
        let truncated = BlockMiss::Truncated(ByteRange { start: 0, end: 8 }).into_error();
        assert!(truncated.to_string().contains("truncated"));
    }
}
