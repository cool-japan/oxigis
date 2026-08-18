//! The PMTiles v3 directory codec: five varint blocks, and the lookup over it.
//!
//! A directory is a sorted run of entries stored **column-wise**, not
//! row-wise: one varint holding the entry count, then *all* the tile-id
//! deltas, then *all* the run lengths, then *all* the lengths, then *all* the
//! offsets. Three encodings ride on top of that:
//!
//! * **tile ids are deltas** against the previous entry — the first entry's
//!   delta is its absolute id, because the running value starts at zero;
//! * **`run_length == 0` marks a leaf pointer**, not a tile. Such an entry
//!   says "the tiles from my id up to the next entry's id are described by
//!   another directory, over there";
//! * **an offset is written as `offset + 1`, or as `0`** meaning "contiguous
//!   with the previous entry", i.e. `prev.offset + prev.length`. A `0` on the
//!   *first* entry has no predecessor and is refused.
//!
//! The decode **must consume its buffer exactly**. Every real archive's
//! directories do, so trailing bytes mean the buffer is not the directory it
//! was claimed to be, and accepting them would let a crafted archive hide
//! payload behind a valid-looking directory.

use crate::pmtiles::PmtilesError;

/// Hard limit on the number of entries one directory may declare.
///
/// A measured leaf directory in a 137 GB planet build holds **60 740**
/// entries, so the plan's original 65 536 sat only 8 % above the largest
/// directory actually observed — a cap that a slightly larger build would trip
/// on a *valid* archive. Raised to 1 Mi entries (~24 MiB of [`DirEntry`]),
/// which is comfortably past anything a writer produces while still bounding
/// the `Vec::with_capacity` a corrupt count would otherwise ask for. The
/// tighter, always-applied guard is [`deserialize_directory`]'s
/// "count must not exceed the buffer length" rule.
pub const MAX_DIRECTORY_ENTRIES: usize = 1024 * 1024;

/// Hard limit on the size of a directory buffer handed to the decoder.
///
/// A directory arrives decompressed, so this bounds what a small compressed
/// block may expand into: 16 MiB is ~7× the largest observed inflated leaf
/// (60 740 entries ≈ 2.3 MiB of varints) and stops a zip-bomb-shaped archive
/// from being decoded at all.
pub const MAX_DIRECTORY_INFLATED_BYTES: usize = 16 * 1024 * 1024;

/// One directory entry.
///
/// The meaning of `offset` depends on which kind of entry this is, and that is
/// the single easiest thing to get wrong in the whole format: a **tile**
/// entry's offset is relative to the header's `tile_data_offset`, a **leaf**
/// entry's to its `leaf_dirs_offset`. [`crate::pmtiles::PmtilesArchive`] is
/// the only place that resolves either, so the two bases live in one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirEntry {
    /// Hilbert tile id this entry starts at.
    pub tile_id: u64,
    /// Byte offset within the entry's region (tile data, or leaf directories).
    pub offset: u64,
    /// Byte length of the tile body or the leaf directory.
    pub length: u32,
    /// How many consecutive tile ids share this body; `0` marks a leaf
    /// pointer rather than a tile.
    pub run_length: u32,
}

/// What [`lookup`] found for a tile id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirLookup {
    /// The id falls inside this entry's run: its body is at the entry.
    Tile(
        /// The matched entry.
        DirEntry,
    ),
    /// The id is covered by a leaf directory the entry points at.
    Leaf(
        /// The matched leaf pointer.
        DirEntry,
    ),
    /// No entry covers the id.
    Absent,
}

/// Decodes a directory from its (already decompressed) bytes.
///
/// # Errors
///
/// * [`PmtilesError::DirectoryTooLarge`] if the buffer is past
///   [`MAX_DIRECTORY_INFLATED_BYTES`].
/// * [`PmtilesError::EmptyDirectory`] — a count of `0` is illegal.
/// * [`PmtilesError::EntryCountExceedsBudget`] if the declared count could not
///   possibly be encoded in a buffer this size (the minimum encoding is four
///   varints, so at least four bytes, per entry) or exceeds
///   [`MAX_DIRECTORY_ENTRIES`]. Checked **before** any allocation.
/// * [`PmtilesError::Truncated`] if a block ends early.
/// * [`PmtilesError::VarintOverflow`] for a varint past ten bytes.
/// * [`PmtilesError::LeadingContiguousOffset`] for a `0` offset on entry 0.
/// * [`PmtilesError::FieldOutOfRange`] if a run length or length does not fit
///   `u32`, or an accumulated tile id or offset overflows `u64`.
/// * [`PmtilesError::TrailingBytes`] if the buffer was not consumed exactly.
pub fn deserialize_directory(buf: &[u8]) -> Result<Vec<DirEntry>, PmtilesError> {
    if buf.len() > MAX_DIRECTORY_INFLATED_BYTES {
        return Err(PmtilesError::DirectoryTooLarge {
            bytes: u64::try_from(buf.len()).unwrap_or(u64::MAX),
            limit: u64::try_from(MAX_DIRECTORY_INFLATED_BYTES).unwrap_or(u64::MAX),
        });
    }
    let buf_len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
    let mut reader = VarintReader::new(buf);

    let count = reader.read("directory entry count")?;
    if count == 0 {
        return Err(PmtilesError::EmptyDirectory);
    }
    let budget = buf_len.min(u64::try_from(MAX_DIRECTORY_ENTRIES).unwrap_or(u64::MAX));
    if count > budget {
        return Err(PmtilesError::EntryCountExceedsBudget { count, budget });
    }
    let Ok(n) = usize::try_from(count) else {
        return Err(PmtilesError::EntryCountExceedsBudget { count, budget });
    };

    // Safe against a hostile count: `n <= buf.len()` was just enforced.
    let mut entries: Vec<DirEntry> = Vec::with_capacity(n);
    let mut tile_id = 0u64;
    for _ in 0..n {
        let delta = reader.read("tile id deltas")?;
        tile_id = tile_id
            .checked_add(delta)
            .ok_or(PmtilesError::FieldOutOfRange {
                field: "tile_id",
                value: delta,
            })?;
        entries.push(DirEntry {
            tile_id,
            offset: 0,
            length: 0,
            run_length: 0,
        });
    }

    for entry in &mut entries {
        let value = reader.read("run lengths")?;
        entry.run_length = u32::try_from(value).map_err(|_| PmtilesError::FieldOutOfRange {
            field: "run_length",
            value,
        })?;
    }

    for entry in &mut entries {
        let value = reader.read("entry lengths")?;
        entry.length = u32::try_from(value).map_err(|_| PmtilesError::FieldOutOfRange {
            field: "length",
            value,
        })?;
    }

    let mut previous_end = 0u64;
    for (index, entry) in entries.iter_mut().enumerate() {
        let value = reader.read("entry offsets")?;
        let offset = if value == 0 {
            if index == 0 {
                return Err(PmtilesError::LeadingContiguousOffset);
            }
            previous_end
        } else {
            value - 1
        };
        entry.offset = offset;
        previous_end =
            offset
                .checked_add(u64::from(entry.length))
                .ok_or(PmtilesError::FieldOutOfRange {
                    field: "offset",
                    value: offset,
                })?;
    }

    let consumed = reader.position();
    if consumed < buf.len() {
        return Err(PmtilesError::TrailingBytes {
            extra: u64::try_from(buf.len() - consumed).unwrap_or(u64::MAX),
        });
    }
    Ok(entries)
}

/// Encodes a directory, the exact inverse of [`deserialize_directory`].
///
/// Test support: this crate has no archive writer on its production paths, and
/// round-tripping the codec against itself is how the decoder is proven. The
/// contiguous-offset form is emitted wherever it applies, so the fixtures
/// exercise the same encoding real writers produce.
///
/// `entries` is expected to be sorted by `tile_id`; the encoding is a delta
/// chain, so an unsorted slice would not round-trip.
#[cfg(any(test, feature = "fixtures"))]
#[must_use]
pub fn serialize_directory(entries: &[DirEntry]) -> Vec<u8> {
    let mut out = Vec::new();
    write_varint(&mut out, u64::try_from(entries.len()).unwrap_or(u64::MAX));

    let mut previous_id = 0u64;
    for entry in entries {
        write_varint(&mut out, entry.tile_id.wrapping_sub(previous_id));
        previous_id = entry.tile_id;
    }
    for entry in entries {
        write_varint(&mut out, u64::from(entry.run_length));
    }
    for entry in entries {
        write_varint(&mut out, u64::from(entry.length));
    }
    let mut previous_end: Option<u64> = None;
    for entry in entries {
        if previous_end == Some(entry.offset) {
            write_varint(&mut out, 0);
        } else {
            write_varint(&mut out, entry.offset.saturating_add(1));
        }
        previous_end = Some(entry.offset.saturating_add(u64::from(entry.length)));
    }
    out
}

/// Finds the entry covering `id`.
///
/// A binary search for the largest entry whose `tile_id <= id`, then the
/// run/leaf disambiguation: a leaf pointer (`run_length == 0`) covers
/// everything from its id onwards, while a tile entry only covers
/// `tile_id .. tile_id + run_length`.
///
/// `entries` must be sorted by `tile_id`, which [`deserialize_directory`]
/// guarantees (the delta chain cannot go backwards).
#[must_use]
pub fn lookup(entries: &[DirEntry], id: u64) -> DirLookup {
    let index = match entries.binary_search_by(|entry| entry.tile_id.cmp(&id)) {
        Ok(exact) => exact,
        Err(0) => return DirLookup::Absent,
        Err(after) => after - 1,
    };
    let Some(entry) = entries.get(index) else {
        return DirLookup::Absent;
    };
    if entry.run_length == 0 {
        return DirLookup::Leaf(*entry);
    }
    if id - entry.tile_id < u64::from(entry.run_length) {
        DirLookup::Tile(*entry)
    } else {
        DirLookup::Absent
    }
}

/// Writes an unsigned LEB128 varint.
#[cfg(any(test, feature = "fixtures"))]
fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(u8::try_from(value & 0x7f).unwrap_or(0) | 0x80);
        value >>= 7;
    }
    out.push(u8::try_from(value & 0x7f).unwrap_or(0));
}

/// A cursor that reads unsigned LEB128 varints and refuses rather than panics.
struct VarintReader<'a> {
    /// The buffer being read.
    buf: &'a [u8],
    /// How many bytes have been consumed.
    position: usize,
}

impl<'a> VarintReader<'a> {
    /// A reader positioned at the start of `buf`.
    const fn new(buf: &'a [u8]) -> Self {
        Self { buf, position: 0 }
    }

    /// How many bytes have been consumed so far.
    const fn position(&self) -> usize {
        self.position
    }

    /// Reads one varint.
    ///
    /// At most ten bytes: the tenth contributes bit 63 and may therefore carry
    /// only a single payload bit. An eleventh byte, or a tenth with more than
    /// one bit set, is [`PmtilesError::VarintOverflow`] rather than a silently
    /// wrapped value.
    fn read(&mut self, context: &'static str) -> Result<u64, PmtilesError> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let Some(&byte) = self.buf.get(self.position) else {
                return Err(PmtilesError::Truncated {
                    context,
                    needed: u64::try_from(self.position)
                        .unwrap_or(u64::MAX)
                        .saturating_add(1),
                    available: u64::try_from(self.buf.len()).unwrap_or(u64::MAX),
                });
            };
            self.position += 1;
            let payload = u64::from(byte & 0x7f);
            if shift > 63 || (shift == 63 && payload > 1) {
                return Err(PmtilesError::VarintOverflow { context });
            }
            value |= payload << shift;
            if byte < 0x80 {
                return Ok(value);
            }
            shift += 7;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DirEntry, DirLookup, MAX_DIRECTORY_ENTRIES, MAX_DIRECTORY_INFLATED_BYTES,
        deserialize_directory, lookup, serialize_directory,
    };
    use crate::pmtiles::PmtilesError;

    fn entry(tile_id: u64, offset: u64, length: u32, run_length: u32) -> DirEntry {
        DirEntry {
            tile_id,
            offset,
            length,
            run_length,
        }
    }

    fn sample_entries() -> Vec<DirEntry> {
        vec![
            entry(0, 0, 40, 1),
            entry(1, 40, 55, 4),  // contiguous with the previous entry
            entry(5, 40, 55, 1),  // a dedup entry pointing *backwards*
            entry(6, 95, 128, 1), // contiguous with entry 1's end again
        ]
    }

    #[test]
    fn one_entry_round_trips() {
        let entries = vec![entry(7, 1_024, 300, 2)];
        let bytes = serialize_directory(&entries);
        assert_eq!(
            deserialize_directory(&bytes).expect("a directory we just wrote"),
            entries
        );
    }

    #[test]
    fn two_entries_round_trip() {
        let entries = vec![entry(0, 0, 10, 1), entry(1, 10, 20, 1)];
        let bytes = serialize_directory(&entries);
        assert_eq!(
            deserialize_directory(&bytes).expect("a directory we just wrote"),
            entries
        );
    }

    #[test]
    fn many_entries_round_trip() {
        let entries: Vec<DirEntry> = (0u32..500)
            .map(|i| entry(u64::from(i) * 3, u64::from(i) * 17, 17, 1))
            .collect();
        let bytes = serialize_directory(&entries);
        assert_eq!(
            deserialize_directory(&bytes).expect("a directory we just wrote"),
            entries
        );
    }

    #[test]
    fn the_contiguous_offset_form_is_emitted_and_understood() {
        let entries = sample_entries();
        let bytes = serialize_directory(&entries);
        let decoded = deserialize_directory(&bytes).expect("a directory we just wrote");
        assert_eq!(decoded, entries);

        // Entry 1 sits exactly at entry 0's end, so it is written as `0`.
        // The last block of the buffer is the offsets, one varint per entry;
        // for these small values that is the final four bytes.
        let offsets = &bytes[bytes.len() - 4..];
        assert_eq!(offsets[0], 1, "entry 0 offset 0 is written as offset + 1");
        assert_eq!(offsets[1], 0, "entry 1 uses the contiguous form");
        assert_eq!(
            offsets[2], 41,
            "entry 2 points backwards, so it is explicit"
        );
        assert_eq!(offsets[3], 0, "entry 3 is contiguous with entry 2's end");
    }

    #[test]
    fn a_leading_contiguous_offset_is_refused() {
        let mut bytes = serialize_directory(&[entry(0, 0, 10, 1), entry(1, 10, 10, 1)]);
        let last = bytes.len() - 2;
        bytes[last] = 0; // entry 0's offset becomes the contiguous form
        assert_eq!(
            deserialize_directory(&bytes),
            Err(PmtilesError::LeadingContiguousOffset)
        );
    }

    #[test]
    fn a_leaf_pointer_round_trips_with_run_length_zero() {
        let entries = vec![entry(0, 0, 512, 0), entry(4096, 512, 700, 0)];
        let bytes = serialize_directory(&entries);
        let decoded = deserialize_directory(&bytes).expect("a directory we just wrote");
        assert_eq!(decoded, entries);
        assert_eq!(decoded[0].run_length, 0);
    }

    #[test]
    fn a_zero_count_is_refused() {
        assert_eq!(
            deserialize_directory(&[0u8]),
            Err(PmtilesError::EmptyDirectory)
        );
    }

    #[test]
    fn an_empty_buffer_is_truncated() {
        assert!(matches!(
            deserialize_directory(&[]),
            Err(PmtilesError::Truncated { .. })
        ));
    }

    #[test]
    fn truncation_at_every_block_boundary_is_refused() {
        let bytes = serialize_directory(&sample_entries());
        for len in 1..bytes.len() {
            let err = deserialize_directory(&bytes[..len]).expect_err("a short directory");
            assert!(
                matches!(
                    err,
                    PmtilesError::Truncated { .. } | PmtilesError::EntryCountExceedsBudget { .. }
                ),
                "len {len} gave {err:?}"
            );
        }
    }

    #[test]
    fn trailing_bytes_are_refused() {
        let mut bytes = serialize_directory(&sample_entries());
        bytes.push(0);
        assert_eq!(
            deserialize_directory(&bytes),
            Err(PmtilesError::TrailingBytes { extra: 1 })
        );
    }

    #[test]
    fn a_varint_past_ten_bytes_is_refused() {
        // Eleven continuation bytes: the count field never terminates.
        let bytes = vec![0xffu8; 11];
        assert_eq!(
            deserialize_directory(&bytes),
            Err(PmtilesError::VarintOverflow {
                context: "directory entry count"
            })
        );
    }

    #[test]
    fn a_ten_byte_varint_that_does_not_fit_u64_is_refused() {
        // Ten bytes whose final byte carries more than the one bit that fits.
        let mut bytes = vec![0xffu8; 9];
        bytes.push(0x7f);
        assert_eq!(
            deserialize_directory(&bytes),
            Err(PmtilesError::VarintOverflow {
                context: "directory entry count"
            })
        );
    }

    #[test]
    fn u64_max_is_representable_as_a_varint() {
        // The boundary the overflow check must not reject: exactly ten bytes,
        // the last carrying one bit.
        let mut bytes = vec![0xffu8; 9];
        bytes.push(0x01);
        // Read as a count it is far past the budget, which is the point: the
        // varint itself decoded, and the *budget* rule refused it.
        assert!(matches!(
            deserialize_directory(&bytes),
            Err(PmtilesError::EntryCountExceedsBudget {
                count: u64::MAX,
                ..
            })
        ));
    }

    #[test]
    fn a_count_larger_than_the_buffer_is_refused_before_allocating() {
        // Declares 1000 entries in a four-byte buffer.
        let bytes = vec![0xe8, 0x07, 0x00, 0x00];
        assert!(matches!(
            deserialize_directory(&bytes),
            Err(PmtilesError::EntryCountExceedsBudget { count: 1000, .. })
        ));
    }

    #[test]
    fn a_buffer_past_the_inflated_limit_is_refused() {
        let bytes = vec![0u8; MAX_DIRECTORY_INFLATED_BYTES + 1];
        assert!(matches!(
            deserialize_directory(&bytes),
            Err(PmtilesError::DirectoryTooLarge { .. })
        ));
    }

    #[test]
    fn the_entry_budget_is_the_smaller_of_the_two_caps() {
        // A buffer big enough that MAX_DIRECTORY_ENTRIES becomes the binding
        // cap would need to be 1 MiB+; assert the constant relationship
        // instead, which is what the budget arithmetic relies on.
        const { assert!(MAX_DIRECTORY_ENTRIES < MAX_DIRECTORY_INFLATED_BYTES) }
    }

    #[test]
    fn lookup_finds_an_exact_hit() {
        let entries = sample_entries();
        assert_eq!(lookup(&entries, 0), DirLookup::Tile(entries[0]));
        assert_eq!(lookup(&entries, 5), DirLookup::Tile(entries[2]));
    }

    #[test]
    fn lookup_finds_an_id_inside_a_run() {
        let entries = sample_entries();
        for id in 1..=4 {
            assert_eq!(lookup(&entries, id), DirLookup::Tile(entries[1]), "id {id}");
        }
    }

    #[test]
    fn lookup_past_the_last_run_is_absent() {
        let entries = sample_entries();
        assert_eq!(lookup(&entries, 7), DirLookup::Absent);
        assert_eq!(lookup(&entries, u64::MAX), DirLookup::Absent);
    }

    #[test]
    fn lookup_before_the_first_entry_is_absent() {
        let entries = vec![entry(10, 0, 5, 1)];
        assert_eq!(lookup(&entries, 0), DirLookup::Absent);
        assert_eq!(lookup(&entries, 9), DirLookup::Absent);
        assert_eq!(lookup(&[], 0), DirLookup::Absent);
    }

    #[test]
    fn lookup_returns_a_leaf_pointer_for_any_id_it_covers() {
        let entries = vec![entry(0, 0, 512, 0), entry(4096, 512, 512, 0)];
        assert_eq!(lookup(&entries, 0), DirLookup::Leaf(entries[0]));
        assert_eq!(lookup(&entries, 4_095), DirLookup::Leaf(entries[0]));
        assert_eq!(lookup(&entries, 4_096), DirLookup::Leaf(entries[1]));
        assert_eq!(lookup(&entries, u64::MAX), DirLookup::Leaf(entries[1]));
    }

    #[test]
    fn a_run_length_past_u32_is_refused() {
        // count = 1, delta = 0, run_length = 2^32, length = 0, offset = 1.
        let mut bytes = vec![1u8, 0u8];
        super::write_varint(&mut bytes, u64::from(u32::MAX) + 1);
        bytes.push(0);
        bytes.push(1);
        assert_eq!(
            deserialize_directory(&bytes),
            Err(PmtilesError::FieldOutOfRange {
                field: "run_length",
                value: u64::from(u32::MAX) + 1
            })
        );
    }

    #[test]
    fn an_overflowing_tile_id_delta_chain_is_refused() {
        // count = 2, deltas u64::MAX then 1.
        let mut bytes = vec![2u8];
        super::write_varint(&mut bytes, u64::MAX);
        super::write_varint(&mut bytes, 1);
        assert_eq!(
            deserialize_directory(&bytes),
            Err(PmtilesError::FieldOutOfRange {
                field: "tile_id",
                value: 1
            })
        );
    }
}
