//! The pull-based PMTiles open: a state machine the caller feeds bytes to.
//!
//! `oxigis-render` performs no I/O, and the UI's transport seam is a
//! non-blocking callback rather than a future, so opening an archive cannot be
//! an `async fn` that awaits a fetch. [`PmtilesOpen`] is driven by alternating
//! [`PmtilesOpen::poll`] with one of two hand-backs — the same shape
//! [`crate::cog::CogOpen`] uses, with one extra step:
//!
//! ```text
//! let mut open = PmtilesOpen::new();
//! let archive = loop {
//!     match open.poll()? {
//!         PmtilesOpenProgress::Need(range) => open.supply(range.start, fetch(range))?,
//!         PmtilesOpenProgress::NeedPlain { slot, compression, raw } =>
//!             open.supply_plain(slot, inflate(compression, raw)?)?,
//!         PmtilesOpenProgress::Ready(archive) => break archive,
//!     }
//! };
//! ```
//!
//! # The measured sequence
//!
//! 1. `Need(0..16_384)` — one speculative read. The spec requires the header
//!    *and* the root directory to be inside it, so this alone opens any
//!    conforming archive (a 137 GB planet build's root ends at 15 691).
//! 2. The metadata block **may or may not** be inside that same buffer. Both
//!    were measured: a small raster archive keeps it at `342..1023`, while a
//!    planet build parks it at byte 136 805 991 502. A second `Need` is
//!    emitted only in the second case.
//!
//! So a small archive opens in **one** round trip and a huge one in two, and
//! nothing else is ever requested at open time.
//!
//! # `NeedPlain`: why the caller inflates
//!
//! The root directory and the metadata are coded with the header's
//! `internal_compression`, and this crate has no codec on its production paths
//! (see the module docs of [`crate::pmtiles`]). When that byte says gzip,
//! `poll` hands the raw bytes back tagged with the codec and waits for
//! [`PmtilesOpen::supply_plain`]. When it says `None` the raw bytes already
//! *are* the plain bytes and the step is skipped entirely — which is what lets
//! the offline fixtures exercise the whole state machine with no codec
//! involved.
//!
//! # Short reads are fine for the prefetch; a wrong offset never is
//!
//! The 16 KiB ask runs past the end of a 282-byte archive, and both shells'
//! range transports document short responses as legitimate, so a short supply
//! is accepted and only fails later if a structure genuinely needed the
//! missing bytes. A response that does not *start* where it was asked to is a
//! transport bug and is refused with
//! [`PmtilesError::SupplyOffsetMismatch`] rather than silently mis-parsed.
//!
//! That tolerance is specific to the speculative prefetch, though. The
//! metadata request is not speculative — its range is computed exactly from
//! the header's own `metadata_offset`/`metadata_length` — so a response
//! shorter than that is a transport bug too, and [`PmtilesOpen::supply`]
//! refuses it with [`PmtilesError::Truncated`] rather than handing a cut-off
//! blob to `String::from_utf8`. A response *longer* than asked is trimmed to
//! the declared length instead of refused: a server that honours a range's
//! start but not its end is unusual but not a reason to fail an otherwise
//! satisfiable open.

use crate::pmtiles::PmtilesError;
use crate::pmtiles::archive::PmtilesArchive;
use crate::pmtiles::directory::{DirEntry, deserialize_directory};
use crate::pmtiles::header::{Compression, PREFETCH_LEN, PmtilesHeader};
use crate::source::ByteRange;

/// Which `internal_compression`-coded block [`PmtilesOpen`] is waiting for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlainSlot {
    /// The root directory.
    Root,
    /// The metadata block.
    Metadata,
}

impl PlainSlot {
    /// A lowercase name for messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Root => "root directory",
            Self::Metadata => "metadata",
        }
    }
}

/// What [`PmtilesOpen::poll`] wants next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PmtilesOpenProgress {
    /// These archive bytes are needed. Hand them over with
    /// [`PmtilesOpen::supply`]; a short response is accepted.
    Need(
        /// The range to read.
        ByteRange,
    ),
    /// These raw bytes must be decoded per `compression` and handed back with
    /// [`PmtilesOpen::supply_plain`].
    ///
    /// Never emitted when `internal_compression` is
    /// [`Compression::None`] — the raw bytes are already plain then.
    NeedPlain {
        /// Which block the bytes are.
        slot: PlainSlot,
        /// The codec the header declares for it.
        compression: Compression,
        /// The block's bytes, exactly as the archive stores them.
        raw: Vec<u8>,
    },
    /// The archive is open; stop polling.
    Ready(
        /// The opened archive. Boxed because it is far larger than the other
        /// variants and this enum is returned by value.
        Box<PmtilesArchive>,
    ),
}

/// Where the open currently is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Nothing requested yet.
    Start,
    /// The speculative prefetch has been asked for.
    AwaitPrefetch,
    /// The header is parsed; the root directory's raw bytes are next.
    HaveHeader,
    /// Waiting for the caller to inflate the root directory.
    AwaitRootPlain,
    /// The root directory is decoded; the metadata block is next.
    HaveRoot,
    /// The metadata block has been asked for.
    AwaitMetadataBytes,
    /// The metadata block's raw bytes are held.
    HaveMetadataRaw,
    /// Waiting for the caller to inflate the metadata block.
    AwaitMetadataPlain,
    /// Everything is decoded; the next poll yields the archive.
    Assembled,
    /// The archive has been handed over.
    Finished,
}

/// An archive open in progress.
#[derive(Debug, Clone)]
pub struct PmtilesOpen {
    /// Where the parse is.
    stage: Stage,
    /// The range the last [`PmtilesOpenProgress::Need`] asked for.
    pending: Option<ByteRange>,
    /// The speculative prefetch, once supplied. May be shorter than asked.
    prefetch: Option<Vec<u8>>,
    /// The header, once parsed.
    header: Option<PmtilesHeader>,
    /// The root directory's bytes as stored, once located.
    root_raw: Vec<u8>,
    /// The root directory, once decoded.
    root: Option<Vec<DirEntry>>,
    /// The metadata block's bytes as stored, once located.
    metadata_raw: Option<Vec<u8>>,
    /// The metadata block as text, once decoded.
    metadata_json: Option<String>,
}

impl Default for PmtilesOpen {
    fn default() -> Self {
        Self::new()
    }
}

impl PmtilesOpen {
    /// A fresh open with nothing read.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            stage: Stage::Start,
            pending: None,
            prefetch: None,
            header: None,
            root_raw: Vec::new(),
            root: None,
            metadata_raw: None,
            metadata_json: None,
        }
    }

    /// The header, once the prefetch has been parsed.
    #[must_use]
    pub const fn header(&self) -> Option<&PmtilesHeader> {
        self.header.as_ref()
    }

    /// Whether the open has produced its archive.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self.stage, Stage::Finished)
    }

    /// Hands over the bytes at archive offset `start`.
    ///
    /// A **short** response to the speculative prefetch is accepted: the
    /// 16 KiB read runs past the end of a small archive, and that is normal.
    /// The metadata request is not speculative — its range is computed
    /// exactly from the header's own `metadata_offset`/`metadata_length` — so
    /// a short response there is refused instead; a long one is trimmed to
    /// the requested length rather than refused.
    ///
    /// # Errors
    ///
    /// * [`PmtilesError::SupplyOffsetMismatch`] if `start` is not the offset
    ///   the last [`PmtilesOpenProgress::Need`] asked for.
    /// * [`PmtilesError::Truncated`] if the metadata request's response is
    ///   shorter than the range that was asked for.
    /// * [`PmtilesError::OpenOutOfOrder`] if nothing was outstanding.
    pub fn supply(&mut self, start: u64, bytes: Vec<u8>) -> Result<(), PmtilesError> {
        let Some(pending) = self.pending else {
            return Err(PmtilesError::OpenOutOfOrder {
                what: "bytes were supplied but none were requested",
            });
        };
        if start != pending.start {
            return Err(PmtilesError::SupplyOffsetMismatch {
                expected: pending.start,
                actual: start,
            });
        }
        match self.stage {
            Stage::AwaitPrefetch => self.prefetch = Some(bytes),
            Stage::AwaitMetadataBytes => {
                let available = as_u64(bytes.len());
                if available < pending.len() {
                    return Err(PmtilesError::Truncated {
                        context: "metadata",
                        needed: pending.len(),
                        available,
                    });
                }
                // Unlike a short response, extra trailing bytes are not a
                // sign the archive is corrupt — only that the transport did
                // not honour the range's end. Trim rather than feed them to
                // `String::from_utf8` as part of the metadata JSON.
                let mut bytes = bytes;
                if let Ok(exact) = usize::try_from(pending.len()) {
                    bytes.truncate(exact);
                }
                self.metadata_raw = Some(bytes);
            }
            _ => {
                return Err(PmtilesError::OpenOutOfOrder {
                    what: "bytes were supplied but none were requested",
                });
            }
        }
        self.pending = None;
        Ok(())
    }

    /// Hands over the decoded bytes of an `internal_compression`-coded block.
    ///
    /// # Errors
    ///
    /// * [`PmtilesError::OpenOutOfOrder`] if that slot was not the one being
    ///   waited for.
    /// * Whatever [`crate::pmtiles::deserialize_directory`] refuses, for the
    ///   root directory.
    /// * [`PmtilesError::MetadataNotUtf8`] for a metadata block that is not
    ///   text.
    pub fn supply_plain(&mut self, slot: PlainSlot, bytes: Vec<u8>) -> Result<(), PmtilesError> {
        match (self.stage, slot) {
            (Stage::AwaitRootPlain, PlainSlot::Root) => {
                self.root = Some(deserialize_directory(&bytes)?);
                self.root_raw = Vec::new();
                self.stage = Stage::HaveRoot;
                Ok(())
            }
            (Stage::AwaitMetadataPlain, PlainSlot::Metadata) => {
                self.metadata_json =
                    Some(String::from_utf8(bytes).map_err(|_| PmtilesError::MetadataNotUtf8)?);
                self.metadata_raw = None;
                self.stage = Stage::Assembled;
                Ok(())
            }
            _ => Err(PmtilesError::OpenOutOfOrder {
                what: "plain bytes were supplied for a block that was not requested",
            }),
        }
    }

    /// Advances the open as far as the bytes supplied so far allow.
    ///
    /// Re-entrant: polling without supplying anything simply re-states the
    /// same request, so a caller may poll in a loop without tracking whether
    /// its fetch has landed.
    ///
    /// # Errors
    ///
    /// Every [`PmtilesError`] the header and directory parsers produce, plus
    /// [`PmtilesError::UnsupportedCompression`] for a brotli- or
    /// zstd-compressed archive (refused once here, by name, rather than per
    /// tile) and [`PmtilesError::OpenOutOfOrder`] if polled after the archive
    /// was handed over.
    pub fn poll(&mut self) -> Result<PmtilesOpenProgress, PmtilesError> {
        loop {
            match self.stage {
                Stage::Start => {
                    let range = ByteRange::new(0, PREFETCH_LEN).map_err(|_| {
                        PmtilesError::InvalidRange {
                            start: 0,
                            end: PREFETCH_LEN,
                        }
                    })?;
                    self.pending = Some(range);
                    self.stage = Stage::AwaitPrefetch;
                    return Ok(PmtilesOpenProgress::Need(range));
                }
                Stage::AwaitPrefetch => {
                    let Some(prefetch) = self.prefetch.as_ref() else {
                        return self.repeat_request();
                    };
                    let header = PmtilesHeader::parse(prefetch)?;
                    refuse_banned_codecs(&header)?;
                    self.header = Some(header);
                    self.stage = Stage::HaveHeader;
                }
                Stage::HaveHeader => {
                    let (header, prefetch) = self.header_and_prefetch()?;
                    let raw = slice_of(prefetch, header.root, "root directory")?.to_vec();
                    if header.internal_compression == Compression::None {
                        self.root = Some(deserialize_directory(&raw)?);
                        self.stage = Stage::HaveRoot;
                    } else {
                        self.root_raw = raw;
                        self.stage = Stage::AwaitRootPlain;
                    }
                }
                Stage::AwaitRootPlain => {
                    let header = self.header_only()?;
                    return Ok(PmtilesOpenProgress::NeedPlain {
                        slot: PlainSlot::Root,
                        compression: header.internal_compression,
                        raw: self.root_raw.clone(),
                    });
                }
                Stage::HaveRoot => {
                    let header = self.header_only()?;
                    let Some(range) = header.metadata_range() else {
                        // A zero-length metadata block is legal.
                        self.metadata_json = Some(String::new());
                        self.stage = Stage::Assembled;
                        continue;
                    };
                    let inside = self
                        .prefetch
                        .as_ref()
                        .is_some_and(|bytes| range.end <= as_u64(bytes.len()));
                    if inside {
                        let raw = {
                            let (_, prefetch) = self.header_and_prefetch()?;
                            slice_of(prefetch, range, "metadata")?.to_vec()
                        };
                        self.metadata_raw = Some(raw);
                        self.stage = Stage::HaveMetadataRaw;
                    } else {
                        self.pending = Some(range);
                        self.stage = Stage::AwaitMetadataBytes;
                        return Ok(PmtilesOpenProgress::Need(range));
                    }
                }
                Stage::AwaitMetadataBytes => {
                    if self.metadata_raw.is_none() {
                        return self.repeat_request();
                    }
                    self.stage = Stage::HaveMetadataRaw;
                }
                Stage::HaveMetadataRaw => {
                    let header = self.header_only()?;
                    if header.internal_compression == Compression::None {
                        let raw = self.metadata_raw.take().unwrap_or_default();
                        self.metadata_json = Some(
                            String::from_utf8(raw).map_err(|_| PmtilesError::MetadataNotUtf8)?,
                        );
                        self.stage = Stage::Assembled;
                    } else {
                        self.stage = Stage::AwaitMetadataPlain;
                    }
                }
                Stage::AwaitMetadataPlain => {
                    let header = self.header_only()?;
                    return Ok(PmtilesOpenProgress::NeedPlain {
                        slot: PlainSlot::Metadata,
                        compression: header.internal_compression,
                        raw: self.metadata_raw.clone().unwrap_or_default(),
                    });
                }
                Stage::Assembled => {
                    let header = self.header_only()?;
                    let root = self.root.take().unwrap_or_default();
                    let metadata = self.metadata_json.take().unwrap_or_default();
                    self.prefetch = None;
                    self.stage = Stage::Finished;
                    return Ok(PmtilesOpenProgress::Ready(Box::new(PmtilesArchive::new(
                        header, root, metadata,
                    ))));
                }
                Stage::Finished => {
                    return Err(PmtilesError::OpenOutOfOrder {
                        what: "polled again after the archive was handed over",
                    });
                }
            }
        }
    }

    /// Re-states the outstanding request.
    fn repeat_request(&self) -> Result<PmtilesOpenProgress, PmtilesError> {
        self.pending
            .map(PmtilesOpenProgress::Need)
            .ok_or(PmtilesError::OpenOutOfOrder {
                what: "the open is waiting for bytes it never requested",
            })
    }

    /// The header, once parsed.
    fn header_only(&self) -> Result<PmtilesHeader, PmtilesError> {
        self.header.ok_or(PmtilesError::OpenOutOfOrder {
            what: "the open advanced past the header without parsing one",
        })
    }

    /// The header plus the prefetch buffer, once both exist.
    fn header_and_prefetch(&self) -> Result<(PmtilesHeader, &[u8]), PmtilesError> {
        let header = self.header_only()?;
        let prefetch = self
            .prefetch
            .as_deref()
            .ok_or(PmtilesError::OpenOutOfOrder {
                what: "the open advanced past the prefetch without holding one",
            })?;
        Ok((header, prefetch))
    }
}

/// Refuses the codecs this workspace bans, once, at open.
fn refuse_banned_codecs(header: &PmtilesHeader) -> Result<(), PmtilesError> {
    if header.internal_compression.is_refused() {
        return Err(PmtilesError::UnsupportedCompression {
            field: "directories",
            compression: header.internal_compression,
        });
    }
    if header.tile_compression.is_refused() {
        return Err(PmtilesError::UnsupportedCompression {
            field: "tiles",
            compression: header.tile_compression,
        });
    }
    Ok(())
}

/// The bytes of `range` inside a buffer that starts at archive offset 0.
fn slice_of<'a>(
    buffer: &'a [u8],
    range: ByteRange,
    context: &'static str,
) -> Result<&'a [u8], PmtilesError> {
    let (Ok(start), Ok(end)) = (usize::try_from(range.start), usize::try_from(range.end)) else {
        return Err(PmtilesError::Truncated {
            context,
            needed: range.end,
            available: as_u64(buffer.len()),
        });
    };
    buffer.get(start..end).ok_or(PmtilesError::Truncated {
        context,
        needed: range.end,
        available: as_u64(buffer.len()),
    })
}

/// `usize` → `u64` without an `as` cast.
fn as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{PlainSlot, PmtilesOpen, PmtilesOpenProgress};
    use crate::pmtiles::PmtilesError;
    use crate::pmtiles::archive::PmtilesArchive;
    use crate::pmtiles::fixture::{
        PmtilesBuilder, sample_pmtiles_far_metadata, sample_pmtiles_vector,
    };
    use crate::pmtiles::header::{Compression, PREFETCH_LEN, TileType};
    use crate::source::ByteRange;

    /// Drives an open against an in-memory archive, counting the round trips.
    ///
    /// `inflate` mirrors what the UI shell does: honour the header's codec.
    fn open_all(archive: &[u8]) -> (PmtilesArchive, usize, usize) {
        let mut open = PmtilesOpen::new();
        let mut byte_requests = 0usize;
        let mut plain_requests = 0usize;
        loop {
            match open.poll().expect("the fixture opens") {
                PmtilesOpenProgress::Need(range) => {
                    byte_requests += 1;
                    open.supply(range.start, read(archive, range))
                        .expect("the right offset");
                }
                PmtilesOpenProgress::NeedPlain {
                    slot,
                    compression,
                    raw,
                } => {
                    plain_requests += 1;
                    let plain = match compression {
                        Compression::Gzip => {
                            oxiarc_deflate::gzip_decompress(&raw).expect("a gzip block")
                        }
                        _ => raw,
                    };
                    open.supply_plain(slot, plain).expect("the right slot");
                }
                PmtilesOpenProgress::Ready(archive) => {
                    return (*archive, byte_requests, plain_requests);
                }
            }
        }
    }

    /// A short-at-EOF read, exactly like a range transport performs.
    fn read(archive: &[u8], range: ByteRange) -> Vec<u8> {
        let start = usize::try_from(range.start).expect("a small fixture");
        let end = usize::try_from(range.end)
            .expect("a small fixture")
            .min(archive.len());
        archive.get(start..end).unwrap_or_default().to_vec()
    }

    #[test]
    fn the_first_poll_asks_for_the_speculative_prefetch() {
        let mut open = PmtilesOpen::new();
        assert_eq!(
            open.poll(),
            Ok(PmtilesOpenProgress::Need(
                ByteRange::new(0, PREFETCH_LEN).expect("non-empty")
            ))
        );
        assert!(open.header().is_none());
        assert!(!open.is_finished());
    }

    #[test]
    fn polling_again_restates_the_same_request() {
        let mut open = PmtilesOpen::new();
        let first = open.poll().expect("a request");
        let second = open.poll().expect("the same request");
        assert_eq!(first, second);
    }

    #[test]
    fn a_small_archive_opens_in_one_round_trip() {
        let bytes = sample_pmtiles_vector();
        let (archive, byte_requests, plain_requests) = open_all(&bytes);
        assert_eq!(byte_requests, 1, "the prefetch covers the whole archive");
        assert_eq!(plain_requests, 0, "an uncompressed archive needs no codec");
        assert_eq!(archive.header().tile_type, TileType::Mvt);
        assert_eq!(archive.root().len(), 2);
        assert!(archive.metadata_json().contains("fixture"));
    }

    #[test]
    fn a_short_response_past_the_end_of_the_file_is_accepted() {
        // The 16 KiB ask runs far past the end of a fixture of a few hundred
        // bytes; `read` truncates exactly as a range transport does.
        let bytes = sample_pmtiles_vector();
        assert!(bytes.len() < 1_000, "the fixture is meant to be tiny");
        let (archive, _, _) = open_all(&bytes);
        assert_eq!(archive.header().max_zoom, 1);
    }

    #[test]
    fn metadata_past_the_prefetch_costs_a_second_round_trip() {
        let bytes = sample_pmtiles_far_metadata();
        let mut open = PmtilesOpen::new();
        let first = open.poll().expect("a request");
        let PmtilesOpenProgress::Need(range) = first else {
            panic!("the first step is always a byte request, got {first:?}");
        };
        assert_eq!(range.start, 0);
        open.supply(0, read(&bytes, range)).expect("offset 0");

        let second = open.poll().expect("a second request");
        let PmtilesOpenProgress::Need(metadata) = second else {
            panic!("the metadata sits past the prefetch, got {second:?}");
        };
        assert!(
            metadata.start > PREFETCH_LEN,
            "metadata at {} should be past the prefetch",
            metadata.start
        );

        let (archive, byte_requests, _) = open_all(&bytes);
        assert_eq!(byte_requests, 2);
        assert!(archive.metadata_json().contains("fixture"));
    }

    #[test]
    fn a_short_response_for_the_metadata_range_is_refused() {
        // Unlike the prefetch, this range is computed exactly from the
        // header's own metadata_offset/metadata_length, so running short is a
        // transport bug, not the normal over-ask a small archive trips.
        let bytes = sample_pmtiles_far_metadata();
        let mut open = PmtilesOpen::new();
        let PmtilesOpenProgress::Need(prefetch_range) = open.poll().expect("a request") else {
            panic!("the first step is a byte request");
        };
        open.supply(0, read(&bytes, prefetch_range))
            .expect("offset 0");

        let PmtilesOpenProgress::Need(metadata_range) = open.poll().expect("a second request")
        else {
            panic!("the metadata sits past the prefetch");
        };
        let mut short = read(&bytes, metadata_range);
        short.pop();
        let available = u64::try_from(short.len()).expect("a small fixture");
        assert_eq!(
            open.supply(metadata_range.start, short),
            Err(PmtilesError::Truncated {
                context: "metadata",
                needed: metadata_range.len(),
                available,
            })
        );
    }

    #[test]
    fn a_response_longer_than_the_metadata_range_is_trimmed_not_refused() {
        // A server that honours a range's start but not its end is unusual
        // but not corrupt; the extra bytes must not leak into the JSON.
        let bytes = sample_pmtiles_far_metadata();
        let mut open = PmtilesOpen::new();
        let PmtilesOpenProgress::Need(prefetch_range) = open.poll().expect("a request") else {
            panic!("the first step is a byte request");
        };
        open.supply(0, read(&bytes, prefetch_range))
            .expect("offset 0");

        let PmtilesOpenProgress::Need(metadata_range) = open.poll().expect("a second request")
        else {
            panic!("the metadata sits past the prefetch");
        };
        let exact = read(&bytes, metadata_range);
        let mut over_long = exact.clone();
        over_long.extend_from_slice(b"trailing bytes the range did not ask for");
        open.supply(metadata_range.start, over_long)
            .expect("a longer-than-asked response is trimmed, not refused");

        let PmtilesOpenProgress::Ready(archive) = open.poll().expect("the open completes") else {
            panic!("nothing else is outstanding");
        };
        assert_eq!(
            archive.metadata_json(),
            String::from_utf8(exact).expect("the fixture's metadata is utf8")
        );
    }

    #[test]
    fn a_gzip_archive_asks_the_caller_to_inflate_both_blocks() {
        let mut builder = PmtilesBuilder::new(TileType::Mvt)
            .with_compression(Compression::Gzip, Compression::Gzip);
        builder.push_tile(0, 0, 0, vec![0x1a, 0x02, 0x0a, 0x00]);
        let bytes = builder.build();
        let (archive, byte_requests, plain_requests) = open_all(&bytes);
        assert_eq!(byte_requests, 1);
        assert_eq!(plain_requests, 2, "root and metadata are both gzip");
        assert_eq!(archive.root().len(), 1);
        assert!(archive.metadata_json().contains("fixture"));
    }

    #[test]
    fn a_wrong_supply_offset_is_refused() {
        let bytes = sample_pmtiles_vector();
        let mut open = PmtilesOpen::new();
        let PmtilesOpenProgress::Need(range) = open.poll().expect("a request") else {
            panic!("the first step is a byte request");
        };
        assert_eq!(
            open.supply(range.start + 1, read(&bytes, range)),
            Err(PmtilesError::SupplyOffsetMismatch {
                expected: 0,
                actual: 1
            })
        );
    }

    #[test]
    fn supplying_bytes_nobody_asked_for_is_refused() {
        let mut open = PmtilesOpen::new();
        assert!(matches!(
            open.supply(0, vec![0u8; 8]),
            Err(PmtilesError::OpenOutOfOrder { .. })
        ));
    }

    #[test]
    fn supplying_plain_bytes_for_an_unrequested_slot_is_refused() {
        let mut open = PmtilesOpen::new();
        assert!(matches!(
            open.supply_plain(PlainSlot::Root, vec![1u8]),
            Err(PmtilesError::OpenOutOfOrder { .. })
        ));
        assert_eq!(PlainSlot::Root.name(), "root directory");
        assert_eq!(PlainSlot::Metadata.name(), "metadata");
    }

    #[test]
    fn polling_after_the_archive_was_handed_over_is_refused() {
        let bytes = sample_pmtiles_vector();
        let mut open = PmtilesOpen::new();
        let PmtilesOpenProgress::Need(range) = open.poll().expect("a request") else {
            panic!("the first step is a byte request");
        };
        open.supply(0, read(&bytes, range)).expect("offset 0");
        // The uncompressed fixture keeps everything inside the prefetch, so
        // this poll runs the whole open to completion.
        assert!(matches!(open.poll(), Ok(PmtilesOpenProgress::Ready(_))));
        assert!(open.is_finished());
        assert!(matches!(
            open.poll(),
            Err(PmtilesError::OpenOutOfOrder { .. })
        ));
    }

    #[test]
    fn a_brotli_archive_is_refused_by_name_at_open() {
        let mut bytes = sample_pmtiles_vector();
        bytes[97] = 3; // internal_compression = brotli
        let mut open = PmtilesOpen::new();
        let PmtilesOpenProgress::Need(range) = open.poll().expect("a request") else {
            panic!("the first step is a byte request");
        };
        open.supply(0, read(&bytes, range)).expect("offset 0");
        assert_eq!(
            open.poll(),
            Err(PmtilesError::UnsupportedCompression {
                field: "directories",
                compression: Compression::Brotli,
            })
        );
    }

    #[test]
    fn a_zstd_tile_codec_is_refused_by_name_at_open() {
        let mut bytes = sample_pmtiles_vector();
        bytes[98] = 4; // tile_compression = zstd
        let mut open = PmtilesOpen::new();
        let PmtilesOpenProgress::Need(range) = open.poll().expect("a request") else {
            panic!("the first step is a byte request");
        };
        open.supply(0, read(&bytes, range)).expect("offset 0");
        assert_eq!(
            open.poll(),
            Err(PmtilesError::UnsupportedCompression {
                field: "tiles",
                compression: Compression::Zstd,
            })
        );
    }

    #[test]
    fn a_truncated_prefetch_that_cuts_the_root_directory_is_refused() {
        let bytes = sample_pmtiles_vector();
        let mut open = PmtilesOpen::new();
        let PmtilesOpenProgress::Need(_) = open.poll().expect("a request") else {
            panic!("the first step is a byte request");
        };
        // Everything but the last byte of the root directory.
        open.supply(0, bytes[..130].to_vec()).expect("offset 0");
        assert!(matches!(
            open.poll(),
            Err(PmtilesError::Truncated {
                context: "root directory",
                ..
            })
        ));
    }

    #[test]
    fn a_garbage_prefetch_is_refused_not_parsed() {
        let mut open = PmtilesOpen::new();
        let PmtilesOpenProgress::Need(_) = open.poll().expect("a request") else {
            panic!("the first step is a byte request");
        };
        open.supply(0, vec![0u8; 200]).expect("offset 0");
        assert_eq!(open.poll(), Err(PmtilesError::BadMagic));
    }

    #[test]
    fn an_archive_with_no_metadata_still_opens() {
        let mut builder = PmtilesBuilder::new(TileType::Png).with_metadata("");
        builder.push_tile(0, 0, 0, vec![1, 2, 3, 4]);
        let bytes = builder.build();
        let (archive, byte_requests, plain_requests) = open_all(&bytes);
        assert_eq!(byte_requests, 1);
        assert_eq!(plain_requests, 0);
        assert_eq!(archive.metadata_json(), "");
        assert_eq!(archive.header().metadata_length, 0);
    }
}
