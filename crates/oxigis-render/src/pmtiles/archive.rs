//! An opened PMTiles archive: what it holds, and where a tile's bytes are.
//!
//! # The offset bases are different, and that is the #1 trap
//!
//! A directory entry's `offset` is relative to a region, and *which* region
//! depends on the kind of entry:
//!
//! * a **tile** entry's offset is relative to `header.tile_data_offset`;
//! * a **leaf** entry's offset is relative to `header.leaf_dirs_offset`.
//!
//! Both bases were confirmed arithmetically against a real server's
//! `Content-Range` on a 137 GB archive: `136 805 992 682 + 73 218 861 =
//! 136 879 211 543` for the leaf, `16 384 + 50 335 970 293 =
//! 50 335 986 677` for the tile. Getting them the wrong way round produces a
//! plausible-looking range that reads the wrong part of the file, which is why
//! resolution lives here, in one place, and why the tests assert the *emitted*
//! [`ByteRange`] rather than only the outcome.
//!
//! # Nothing is emitted without a bounds check
//!
//! Directory offsets are attacker-controlled numbers from a remote file, so
//! every range is checked against the region it claims to be inside before it
//! leaves this module. A corrupt directory therefore cannot make a transport
//! read outside the archive; it produces [`PmtilesError::RangeOutOfBounds`].
//!
//! # The zoom and bbox gate comes first
//!
//! A view zoomed out past an archive's `max_zoom`, or panned off its bounding
//! box, asks for hundreds of tiles the archive cannot hold. Answering those
//! [`TileLookup::Absent`] *before* touching a directory is what stops a
//! world-zoomed-out view from issuing leaf reads for nothing.

use crate::mercator::TileId;
use crate::pmtiles::PmtilesError;
use crate::pmtiles::directory::{DirEntry, DirLookup, lookup};
use crate::pmtiles::header::{Compression, PmtilesHeader, TileType};
use crate::pmtiles::hilbert::zxy_to_tile_id;
use crate::source::ByteRange;

/// How many directories deep a lookup may go: the root plus one leaf level.
///
/// Every v3 archive measured is exactly this deep — the 137 GB planet build
/// has one leaf level — but the format permits nesting, and a crafted archive
/// whose leaf points at itself must not loop. **Counting the hops is the
/// caller's job**, because the caller is the one that fetches and decodes each
/// leaf: [`PmtilesArchive::find_in_leaf`] may legitimately answer
/// [`TileLookup::Leaf`] again, and a caller that has already used its budget
/// refuses with [`PmtilesError::LeafChainTooDeep`].
pub const MAX_DIRECTORY_DEPTH: u32 = 2;

/// Hard cap on the byte length of one leaf directory this reader will fetch.
///
/// The largest leaf measured in the wild is 125 412 compressed bytes; 4 MiB is
/// ~33× that, and bounds what a corrupt `length` field can turn into a range
/// request. The inflated size is bounded separately by
/// [`crate::pmtiles::MAX_DIRECTORY_INFLATED_BYTES`].
pub const MAX_DIRECTORY_BYTES: u32 = 4 * 1024 * 1024;

/// Hard cap on the byte length of one tile body this reader will fetch.
///
/// A vector tile past a megabyte is already pathological and a raster tile far
/// more so; 16 MiB leaves room for an unusually dense MVT while stopping a
/// corrupt `length` from being turned into a huge request.
pub const MAX_ARCHIVE_TILE_BYTES: u32 = 16 * 1024 * 1024;

/// Where a tile's bytes are, or that there are none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileLookup {
    /// Fetch this range and the tile body is in it, still coded per the
    /// header's `tile_compression`.
    Tile(
        /// Absolute byte range of the body in the archive.
        ByteRange,
    ),
    /// The answer is in a leaf directory. Fetch `range`, decode it per the
    /// header's `internal_compression`, run it through
    /// [`crate::pmtiles::deserialize_directory`] and ask
    /// [`PmtilesArchive::find_in_leaf`].
    Leaf {
        /// The leaf's offset *within the leaf-directory region* — its
        /// identity, and a stable cache key.
        at: u64,
        /// Absolute byte range of the leaf directory in the archive.
        range: ByteRange,
    },
    /// The archive holds no tile at that address.
    ///
    /// A **final, non-error** answer: a sparse archive missing an ocean tile
    /// is normal, and the caller must cache it rather than retry.
    Absent,
}

/// Archive-level facts, for a status line or a layer's default styling.
///
/// Deliberately does *not* carry `vector_layers` or `attribution`:
/// `oxigis-render` has no JSON parser (adding one would break the crate's
/// dependency budget for a field only the UI reads), so the metadata block is
/// handed over verbatim by [`PmtilesArchive::metadata_json`] and the shell —
/// which already depends on `serde_json` — parses it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmtilesInfo {
    /// What the tile bodies are.
    pub tile_type: TileType,
    /// Codec of the tile bodies.
    pub tile_compression: Compression,
    /// Codec of the directories and the metadata block.
    pub internal_compression: Compression,
    /// Lowest zoom the archive holds.
    pub min_zoom: u8,
    /// Highest zoom the archive holds.
    pub max_zoom: u8,
    /// `[min_lon, min_lat, max_lon, max_lat]` in degrees.
    pub bounds_deg: [f64; 4],
    /// Whether the bounding box was actually declared, or is all zeroes.
    pub has_bounds: bool,
    /// Suggested map centre as `(lon, lat)` in degrees.
    pub center_deg: (f64, f64),
    /// Suggested opening zoom.
    pub center_zoom: u8,
    /// How many `(z, x, y)` addresses the archive answers; `0` = not counted.
    pub addressed_tiles: u64,
    /// Whether tile bodies are stored in tile-id order.
    pub clustered: bool,
}

/// An opened archive: its header, its root directory and its metadata.
///
/// Leaf directories are deliberately **not** cached here. A single measured
/// leaf holds 60 740 entries (~1.9 MB in memory), so a cache of them has to be
/// byte-budgeted against the whole application's memory picture — a policy
/// decision that belongs to the shell that also owns the transport, not to a
/// parse type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PmtilesArchive {
    /// The parsed header.
    header: PmtilesHeader,
    /// The decoded root directory, sorted by tile id.
    root: Vec<DirEntry>,
    /// The metadata block, verbatim. Empty when the archive carries none.
    metadata_json: String,
}

impl PmtilesArchive {
    /// Assembles an archive from its already-decoded parts.
    ///
    /// [`crate::pmtiles::PmtilesOpen`] is the normal way to get one; this is
    /// public so a caller that already holds the pieces (a test, a second
    /// reader) can build one without re-running the state machine.
    #[must_use]
    pub fn new(header: PmtilesHeader, root: Vec<DirEntry>, metadata_json: String) -> Self {
        Self {
            header,
            root,
            metadata_json,
        }
    }

    /// The parsed header.
    #[must_use]
    pub const fn header(&self) -> &PmtilesHeader {
        &self.header
    }

    /// The decoded root directory.
    #[must_use]
    pub fn root(&self) -> &[DirEntry] {
        &self.root
    }

    /// The metadata block verbatim, as the archive stored it.
    ///
    /// PMTiles v3 writes JSON here, carrying `vector_layers` and usually
    /// `attribution`. This crate does not parse it — see [`PmtilesInfo`].
    /// Empty when the archive declares no metadata.
    #[must_use]
    pub fn metadata_json(&self) -> &str {
        &self.metadata_json
    }

    /// Archive-level facts.
    #[must_use]
    pub fn info(&self) -> PmtilesInfo {
        PmtilesInfo {
            tile_type: self.header.tile_type,
            tile_compression: self.header.tile_compression,
            internal_compression: self.header.internal_compression,
            min_zoom: self.header.min_zoom,
            max_zoom: self.header.max_zoom,
            bounds_deg: self.header.bounds_deg(),
            has_bounds: self.header.has_bounds(),
            center_deg: self.header.center_deg(),
            center_zoom: self.header.center_zoom,
            addressed_tiles: self.header.addressed_tiles,
            clustered: self.header.clustered,
        }
    }

    /// Whether the archive could hold `tile` at all.
    ///
    /// The zoom range always applies; the bounding box only when the header
    /// actually declared one ([`PmtilesHeader::has_bounds`]), because a writer
    /// that left it zeroed would otherwise gate away the entire archive.
    #[must_use]
    pub fn covers(&self, tile: TileId) -> bool {
        if tile.z < self.header.min_zoom || tile.z > self.header.max_zoom {
            return false;
        }
        if !self.header.has_bounds() {
            return true;
        }
        let [min_lon, min_lat, max_lon, max_lat] = self.header.bounds_deg();
        let (north_west, south_east) = tile.bounds_lon_lat();
        // Strictly disjoint only; a tile merely touching the box is kept, so a
        // rounding difference can never hide a tile the archive does hold.
        let disjoint = south_east.lon < min_lon
            || north_west.lon > max_lon
            || north_west.lat < min_lat
            || south_east.lat > max_lat;
        !disjoint
    }

    /// Resolves `tile` against the root directory.
    ///
    /// # Errors
    ///
    /// [`PmtilesError::RangeOutOfBounds`] if the matched entry addresses bytes
    /// outside the region its offsets are relative to,
    /// [`PmtilesError::FieldOutOfRange`] if its length is past this reader's
    /// caps, and [`PmtilesError::ZoomOutOfRange`] /
    /// [`PmtilesError::BadCoordinate`] for an address the tile-id curve cannot
    /// express. A tile the archive simply does not hold is
    /// [`TileLookup::Absent`], never an error.
    pub fn find(&self, tile: TileId) -> Result<TileLookup, PmtilesError> {
        self.find_in(&self.root, tile)
    }

    /// Resolves `tile` against an already-decoded leaf directory.
    ///
    /// The second hop of a two-level lookup: `leaf_entries` is what
    /// [`crate::pmtiles::deserialize_directory`] returned for the leaf that
    /// [`PmtilesArchive::find`] pointed at. The result may itself be
    /// [`TileLookup::Leaf`] — the format allows nesting — and bounding that
    /// chain against [`MAX_DIRECTORY_DEPTH`] is the caller's job.
    ///
    /// # Errors
    ///
    /// The same as [`PmtilesArchive::find`].
    pub fn find_in_leaf(
        &self,
        leaf_entries: &[DirEntry],
        tile: TileId,
    ) -> Result<TileLookup, PmtilesError> {
        self.find_in(leaf_entries, tile)
    }

    /// Shared body of [`PmtilesArchive::find`] and
    /// [`PmtilesArchive::find_in_leaf`].
    fn find_in(&self, entries: &[DirEntry], tile: TileId) -> Result<TileLookup, PmtilesError> {
        if !self.covers(tile) {
            return Ok(TileLookup::Absent);
        }
        let id = zxy_to_tile_id(tile.z, tile.x, tile.y)?;
        match lookup(entries, id) {
            DirLookup::Absent => Ok(TileLookup::Absent),
            DirLookup::Tile(entry) => Ok(TileLookup::Tile(self.tile_range(entry)?)),
            DirLookup::Leaf(entry) => Ok(TileLookup::Leaf {
                at: entry.offset,
                range: self.leaf_range(entry)?,
            }),
        }
    }

    /// The absolute range of a tile entry's body.
    ///
    /// Base: `tile_data_offset`. Bounds: the tile-data region.
    fn tile_range(&self, entry: DirEntry) -> Result<ByteRange, PmtilesError> {
        if entry.length == 0 || entry.length > MAX_ARCHIVE_TILE_BYTES {
            return Err(PmtilesError::FieldOutOfRange {
                field: "tile length",
                value: u64::from(entry.length),
            });
        }
        bounded_range(
            "tile data",
            self.header.tile_data_offset,
            self.header.tile_data_end(),
            entry.offset,
            entry.length,
        )
    }

    /// The absolute range of a leaf entry's directory.
    ///
    /// Base: `leaf_dirs_offset`. Bounds: the leaf-directory region.
    fn leaf_range(&self, entry: DirEntry) -> Result<ByteRange, PmtilesError> {
        if entry.length == 0 || entry.length > MAX_DIRECTORY_BYTES {
            return Err(PmtilesError::FieldOutOfRange {
                field: "leaf directory length",
                value: u64::from(entry.length),
            });
        }
        bounded_range(
            "leaf directory",
            self.header.leaf_dirs_offset,
            self.header.leaf_dirs_end(),
            entry.offset,
            entry.length,
        )
    }
}

/// Turns a region-relative offset/length into an absolute, bounds-checked
/// range.
fn bounded_range(
    region: &'static str,
    region_start: u64,
    region_end: u64,
    offset: u64,
    length: u32,
) -> Result<ByteRange, PmtilesError> {
    let out_of_bounds = |start: u64, end: u64| PmtilesError::RangeOutOfBounds {
        region,
        start,
        end,
        region_start,
        region_end,
    };
    let Some(start) = region_start.checked_add(offset) else {
        return Err(out_of_bounds(region_start, region_end));
    };
    let Some(end) = start.checked_add(u64::from(length)) else {
        return Err(out_of_bounds(start, region_end));
    };
    if end > region_end {
        return Err(out_of_bounds(start, end));
    }
    ByteRange::new(start, end).map_err(|_| PmtilesError::InvalidRange { start, end })
}

#[cfg(test)]
mod tests {
    use super::{MAX_ARCHIVE_TILE_BYTES, MAX_DIRECTORY_DEPTH, PmtilesArchive, TileLookup};
    use crate::mercator::TileId;
    use crate::pmtiles::PmtilesError;
    use crate::pmtiles::directory::DirEntry;
    use crate::pmtiles::header::{Compression, PmtilesHeader, TileType};
    use crate::source::ByteRange;

    /// A header with distinct, easily recognised region bases so an offset
    /// mix-up is visible in the asserted numbers rather than only in an
    /// outcome.
    fn header() -> PmtilesHeader {
        PmtilesHeader {
            root: ByteRange::new(127, 227).expect("a non-empty root"),
            metadata_offset: 227,
            metadata_length: 20,
            leaf_dirs_offset: 1_000,
            leaf_dirs_len: 500,
            tile_data_offset: 90_000,
            tile_data_len: 10_000,
            addressed_tiles: 5,
            tile_entries: 2,
            tile_contents: 2,
            clustered: true,
            internal_compression: Compression::None,
            tile_compression: Compression::None,
            tile_type: TileType::Mvt,
            min_zoom: 0,
            max_zoom: 3,
            min_lon_e7: -1_800_000_000,
            min_lat_e7: -850_511_287,
            max_lon_e7: 1_800_000_000,
            max_lat_e7: 850_511_287,
            center_zoom: 0,
            center_lon_e7: 0,
            center_lat_e7: 0,
        }
    }

    fn entry(tile_id: u64, offset: u64, length: u32, run_length: u32) -> DirEntry {
        DirEntry {
            tile_id,
            offset,
            length,
            run_length,
        }
    }

    fn archive(root: Vec<DirEntry>) -> PmtilesArchive {
        PmtilesArchive::new(header(), root, "{\"name\":\"fixture\"}".to_owned())
    }

    fn tile(z: u8, x: u32, y: u32) -> TileId {
        TileId::new(z, x, y).expect("a valid tile address")
    }

    #[test]
    fn a_tile_entry_is_based_at_tile_data_offset() {
        // Regression guard for the #1 trap: the emitted numbers, not just the
        // variant, are what this asserts.
        let archive = archive(vec![entry(0, 40, 60, 1)]);
        let found = archive.find(tile(0, 0, 0)).expect("in bounds");
        assert_eq!(
            found,
            TileLookup::Tile(ByteRange::new(90_040, 90_100).expect("non-empty"))
        );
    }

    #[test]
    fn a_leaf_entry_is_based_at_leaf_dirs_offset() {
        let archive = archive(vec![entry(0, 40, 60, 0)]);
        let found = archive.find(tile(0, 0, 0)).expect("in bounds");
        assert_eq!(
            found,
            TileLookup::Leaf {
                at: 40,
                range: ByteRange::new(1_040, 1_100).expect("non-empty"),
            }
        );
    }

    #[test]
    fn the_two_bases_are_not_interchangeable() {
        // The same entry offset resolves to two different absolute ranges
        // depending on the entry kind. If the bases were ever conflated this
        // is the assertion that fails.
        let as_tile = archive(vec![entry(0, 7, 11, 1)])
            .find(tile(0, 0, 0))
            .expect("in bounds");
        let as_leaf = archive(vec![entry(0, 7, 11, 0)])
            .find(tile(0, 0, 0))
            .expect("in bounds");
        assert_eq!(
            as_tile,
            TileLookup::Tile(ByteRange::new(90_007, 90_018).expect("non-empty"))
        );
        assert_eq!(
            as_leaf,
            TileLookup::Leaf {
                at: 7,
                range: ByteRange::new(1_007, 1_018).expect("non-empty"),
            }
        );
    }

    #[test]
    fn a_tile_entry_past_the_tile_region_is_refused() {
        let archive = archive(vec![entry(0, 9_990, 20, 1)]);
        assert_eq!(
            archive.find(tile(0, 0, 0)),
            Err(PmtilesError::RangeOutOfBounds {
                region: "tile data",
                start: 99_990,
                end: 100_010,
                region_start: 90_000,
                region_end: 100_000,
            })
        );
    }

    #[test]
    fn a_leaf_entry_past_the_leaf_region_is_refused() {
        let archive = archive(vec![entry(0, 490, 20, 0)]);
        assert!(matches!(
            archive.find(tile(0, 0, 0)),
            Err(PmtilesError::RangeOutOfBounds {
                region: "leaf directory",
                ..
            })
        ));
    }

    #[test]
    fn an_overflowing_entry_offset_is_refused() {
        let archive = archive(vec![entry(0, u64::MAX, 20, 1)]);
        assert!(matches!(
            archive.find(tile(0, 0, 0)),
            Err(PmtilesError::RangeOutOfBounds { .. })
        ));
    }

    #[test]
    fn a_zero_or_oversized_length_is_refused() {
        let empty = archive(vec![entry(0, 0, 0, 1)]);
        assert_eq!(
            empty.find(tile(0, 0, 0)),
            Err(PmtilesError::FieldOutOfRange {
                field: "tile length",
                value: 0
            })
        );
        let oversized = archive(vec![entry(0, 0, MAX_ARCHIVE_TILE_BYTES + 1, 1)]);
        assert!(matches!(
            oversized.find(tile(0, 0, 0)),
            Err(PmtilesError::FieldOutOfRange {
                field: "tile length",
                ..
            })
        ));
    }

    #[test]
    fn the_zoom_gate_answers_absent_without_touching_a_directory() {
        // An empty root: any directory work at all would be visible as a
        // different answer than Absent.
        let archive = archive(Vec::new());
        assert_eq!(archive.find(tile(4, 0, 0)), Ok(TileLookup::Absent));
        assert!(!archive.covers(tile(4, 0, 0)));

        let mut header = header();
        header.min_zoom = 2;
        let archive = PmtilesArchive::new(header, vec![entry(0, 0, 10, 1)], String::new());
        assert_eq!(archive.find(tile(1, 0, 0)), Ok(TileLookup::Absent));
        assert!(archive.covers(tile(2, 0, 0)));
    }

    #[test]
    fn the_bbox_gate_answers_absent_for_a_tile_outside_the_box() {
        let mut header = header();
        // A small box around Tokyo.
        header.min_lon_e7 = 1_390_000_000;
        header.min_lat_e7 = 350_000_000;
        header.max_lon_e7 = 1_400_000_000;
        header.max_lat_e7 = 360_000_000;
        header.max_zoom = 6;
        let archive = PmtilesArchive::new(header, vec![entry(0, 0, 10, 1_000_000)], String::new());
        // z6 tile over the mid-Atlantic.
        assert_eq!(archive.find(tile(6, 28, 28)), Ok(TileLookup::Absent));
        // z6 tile over Tokyo is not gated away.
        assert!(archive.covers(tile(6, 56, 25)));
    }

    #[test]
    fn an_undeclared_bbox_gates_nothing() {
        let mut header = header();
        header.min_lon_e7 = 0;
        header.min_lat_e7 = 0;
        header.max_lon_e7 = 0;
        header.max_lat_e7 = 0;
        let archive = PmtilesArchive::new(header, vec![entry(0, 0, 10, 1)], String::new());
        assert!(archive.covers(tile(0, 0, 0)));
        assert!(archive.covers(tile(3, 7, 7)));
    }

    #[test]
    fn a_tile_the_directory_does_not_hold_is_absent_not_an_error() {
        let archive = archive(vec![entry(0, 0, 10, 1)]);
        assert_eq!(archive.find(tile(1, 1, 1)), Ok(TileLookup::Absent));
    }

    #[test]
    fn find_in_leaf_uses_the_same_bases() {
        let archive = archive(vec![entry(0, 0, 10, 0)]);
        let leaf = vec![entry(1, 100, 30, 4)];
        assert_eq!(
            archive.find_in_leaf(&leaf, tile(1, 0, 0)),
            Ok(TileLookup::Tile(
                ByteRange::new(90_100, 90_130).expect("non-empty")
            ))
        );
    }

    #[test]
    fn a_leaf_pointing_at_a_leaf_is_reported_so_the_caller_can_bound_it() {
        // The depth budget is the caller's; this asserts the archive hands the
        // second leaf back rather than following it itself.
        let archive = archive(vec![entry(0, 0, 10, 0)]);
        let nested = vec![entry(0, 0, 10, 0)];
        assert!(matches!(
            archive.find_in_leaf(&nested, tile(0, 0, 0)),
            Ok(TileLookup::Leaf { .. })
        ));
        assert_eq!(MAX_DIRECTORY_DEPTH, 2);
        // What a caller that has exhausted its budget then produces.
        let refusal = PmtilesError::LeafChainTooDeep {
            depth: MAX_DIRECTORY_DEPTH + 1,
            limit: MAX_DIRECTORY_DEPTH,
        };
        assert_eq!(
            refusal.to_string(),
            "the archive's leaf-directory chain is 3 deep, past the limit of 2"
        );
    }

    /// Opens an in-memory fixture the way a shell would, but synchronously.
    fn open_fixture(bytes: &[u8]) -> PmtilesArchive {
        use crate::pmtiles::open::{PmtilesOpen, PmtilesOpenProgress};

        let mut open = PmtilesOpen::new();
        loop {
            match open.poll().expect("the fixture opens") {
                PmtilesOpenProgress::Need(range) => {
                    let supplied = read(bytes, range);
                    open.supply(range.start, supplied)
                        .expect("the right offset");
                }
                PmtilesOpenProgress::NeedPlain { slot, raw, .. } => {
                    open.supply_plain(slot, raw).expect("the right slot");
                }
                PmtilesOpenProgress::Ready(archive) => return *archive,
            }
        }
    }

    /// A short-at-EOF read, exactly like a range transport performs.
    fn read(bytes: &[u8], range: ByteRange) -> Vec<u8> {
        let start = usize::try_from(range.start).expect("a small fixture");
        let end = usize::try_from(range.end)
            .expect("a small fixture")
            .min(bytes.len());
        bytes.get(start..end).unwrap_or_default().to_vec()
    }

    #[test]
    fn the_vector_fixture_resolves_every_address_it_holds() {
        use crate::pmtiles::fixture::sample_pmtiles_vector;

        let bytes = sample_pmtiles_vector();
        let archive = open_fixture(&bytes);

        let TileLookup::Tile(z0) = archive.find(tile(0, 0, 0)).expect("in bounds") else {
            panic!("z0 0/0 is stored");
        };
        // All four z1 addresses share one body, reached through one
        // run_length = 4 entry: the same range comes back for each.
        let mut z1_ranges = Vec::new();
        for (x, y) in [(0, 0), (0, 1), (1, 1), (1, 0)] {
            let TileLookup::Tile(range) = archive.find(tile(1, x, y)).expect("in bounds") else {
                panic!("z1 {x}/{y} is stored");
            };
            z1_ranges.push(range);
        }
        assert!(z1_ranges.iter().all(|range| *range == z1_ranges[0]));
        assert_ne!(z0, z1_ranges[0]);

        // The emitted ranges address real bytes inside the archive.
        assert!(z1_ranges[0].end <= u64::try_from(bytes.len()).expect("small"));
        assert_eq!(read(&bytes, z0).len(), 4);
        assert_eq!(read(&bytes, z1_ranges[0]), read(&bytes, z1_ranges[3]));

        // z2 is past max_zoom, so the archive answers Absent.
        assert_eq!(archive.find(tile(2, 0, 0)), Ok(TileLookup::Absent));
    }

    #[test]
    fn a_leaf_hop_through_the_leafed_fixture_reaches_every_tile() {
        use crate::pmtiles::deserialize_directory;
        use crate::pmtiles::fixture::sample_pmtiles_leafed;

        let bytes = sample_pmtiles_leafed();
        let archive = open_fixture(&bytes);
        assert!(
            archive.root().iter().all(|entry| entry.run_length == 0),
            "the fixture's root is leaf pointers only"
        );

        let mut bodies = Vec::new();
        for (z, x, y) in [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)] {
            let found = archive.find(tile(z, x, y)).expect("in bounds");
            let TileLookup::Leaf { at, range } = found else {
                panic!("z{z} {x}/{y} must go through a leaf, got {found:?}");
            };
            assert_eq!(
                range.start,
                archive.header().leaf_dirs_offset + at,
                "a leaf range is based at leaf_dirs_offset"
            );
            let leaf = deserialize_directory(&read(&bytes, range)).expect("a well-formed leaf");
            let second = archive
                .find_in_leaf(&leaf, tile(z, x, y))
                .expect("the second hop resolves");
            let TileLookup::Tile(body) = second else {
                panic!("the second hop must land on a tile, got {second:?}");
            };
            assert!(
                body.start >= archive.header().tile_data_offset,
                "a tile range is based at tile_data_offset"
            );
            bodies.push(read(&bytes, body));
        }
        assert_eq!(bodies.len(), 5);
        bodies.sort_unstable();
        bodies.dedup();
        assert_eq!(bodies.len(), 5, "the fixture stores five distinct bodies");
    }

    #[test]
    fn info_reports_the_headers_facts() {
        let archive = archive(vec![entry(0, 0, 10, 1)]);
        let info = archive.info();
        assert_eq!(info.tile_type, TileType::Mvt);
        assert_eq!(info.internal_compression, Compression::None);
        assert_eq!(info.min_zoom, 0);
        assert_eq!(info.max_zoom, 3);
        assert!(info.has_bounds);
        assert!((info.bounds_deg[2] - 180.0).abs() < 1e-9);
        assert_eq!(info.addressed_tiles, 5);
        assert!(info.clustered);
        assert_eq!(archive.metadata_json(), "{\"name\":\"fixture\"}");
        assert_eq!(archive.root().len(), 1);
        assert_eq!(archive.header().min_zoom, 0);
    }
}
