//! The `async` convenience wrapper over the pull-based COG machinery.
//!
//! [`CogSource`] pairs the state machine in [`super::open`] with a
//! [`RangeFetch`] so a caller that *does* have an async runtime — a test, a
//! command-line tool, a future headless renderer — can write
//! `source.open().await` instead of driving [`CogOpen`] by hand.
//!
//! # Not the path the UI takes
//!
//! `oxigis-ui`'s tile-provider seam is synchronous and its transports are
//! callback-based, and [`crate::source::TileFuture`] is deliberately not `Send`
//! on `wasm32`, which makes a `RangeFetch`-based provider impossible to store in
//! `egui_wgpu`'s callback resources. The UI therefore drives [`CogOpen`]
//! directly from its delivery callback; this type exists alongside that, not
//! under it.

use crate::cog::codec::{CogDecodeOptions, RasterStretch, decode_cog_block};
use crate::cog::meta::CogMetadata;
use crate::cog::open::{CogOpen, CogOpenProgress};
use crate::cog::plan::{CogSourceTile, CogTilePlan};
use crate::error::RenderError;
use crate::mercator::TileId;
use crate::renderer::DecodedTile;
use crate::source::{ByteRange, MaybeSync, RangeFetch, TileFuture};

/// Bytes of the first request an open issues.
///
/// Not a 16-byte header probe: the parse identifies the TIFF from a speculative
/// [`super::blocks::HEADER_PREFETCH_BYTES`] read that, in a conventionally
/// written file, also answers every IFD and tile-directory read — which is what
/// collapses an open from ten-odd round trips to one. Named for what it is
/// *for* rather than for its size, and asserted against an observed request in
/// this module's tests.
pub const COG_HEADER_PROBE_BYTES: u64 = super::blocks::HEADER_PREFETCH_BYTES;

/// Window read at an IFD offset whose values fall outside the header prefetch.
///
/// A tile directory parked past the prefetch is fetched in windows of this
/// size rather than field by field; see [`super::blocks::MIN_FETCH_BYTES`].
pub const COG_IFD_WINDOW_BYTES: u64 = super::blocks::MIN_FETCH_BYTES;

/// Upper bound on how far the "next IFD" chain is followed when enumerating
/// overview levels, so a malformed file cannot loop forever.
pub const COG_MAX_OVERVIEW_LEVELS: usize = super::open::MAX_IFD_CHAIN;

/// Upper bound on simultaneous tile range requests, tuned for a browser's
/// per-origin connection pool rather than for raw bandwidth.
pub const COG_MAX_CONCURRENT_TILE_FETCHES: usize = 8;

/// One step of the COG read plan.
///
/// The sequence is the one validated in production by `oxigeo-wasm`'s
/// `cog_reader.rs`, with the per-IFD window widened into a single speculative
/// header prefetch (see [`super::blocks`]). Every number here is the one the
/// code actually uses, and this module's tests assert them against observed
/// requests rather than against the constants they came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CogReadStep {
    /// Read the speculative header prefetch: the `II`/`MM` byte-order mark, the
    /// magic number `42`, the primary IFD offset, and — in a conventionally
    /// written COG — the whole IFD chain and every level's tile directory.
    HeaderProbe {
        /// Number of bytes the first request asks for.
        bytes: u64,
    },
    /// Read a window at any IFD or value offset the prefetch did not cover, and
    /// parse the directory (dimensions, tile size, bits per sample,
    /// compression, predictor, tile offset/byte-count arrays, GeoKeys).
    PrimaryIfd {
        /// Size of the window read at an offset outside the prefetch.
        window_bytes: u64,
    },
    /// Follow the "next IFD" chain, parsing every overview level with the same
    /// routine so each carries its own tile directory.
    OverviewChain {
        /// Hard limit on the number of levels walked.
        max_levels: usize,
    },
    /// Resolve `(level, tile_x, tile_y)` against that level's tile directory
    /// and read exactly that byte range.
    TileRange {
        /// Maximum number of concurrent range requests.
        max_concurrent: usize,
    },
}

/// A Cloud-Optimized GeoTIFF tile source driven by an async range reader.
#[derive(Debug, Clone)]
pub struct CogSource<R> {
    /// Where bytes come from.
    range_fetch: R,
    /// Stretch, nodata and JPEG tables used when decoding a block.
    options: CogDecodeOptions,
}

impl<R> CogSource<R> {
    /// Wraps a range reader, using the default [`CogDecodeOptions`].
    pub fn new(range_fetch: R) -> Self {
        Self {
            range_fetch,
            options: CogDecodeOptions::default(),
        }
    }

    /// Sets how 16-bit samples are mapped onto the display range.
    #[must_use]
    pub fn with_stretch(mut self, stretch: RasterStretch) -> Self {
        self.options.stretch = stretch;
        self
    }

    /// Sets the `GDAL_NODATA` value whose pixels decode as transparent.
    #[must_use]
    pub fn with_nodata(mut self, nodata: Option<f64>) -> Self {
        self.options.nodata = nodata;
        self
    }

    /// Replaces the whole decode option set, keeping the range reader.
    ///
    /// Pair with [`CogSource::open_with_options`], which reports the nodata
    /// value and `JPEGTables` the file declares — neither of which
    /// [`CogMetadata`] has a field for.
    #[must_use]
    pub fn with_options(mut self, options: CogDecodeOptions) -> Self {
        self.options = options;
        self
    }

    /// The decode options in effect.
    #[must_use]
    pub const fn options(&self) -> &CogDecodeOptions {
        &self.options
    }

    /// The sample stretch in effect.
    #[must_use]
    pub const fn stretch(&self) -> RasterStretch {
        self.options.stretch
    }

    /// The underlying range reader.
    pub const fn range_fetch(&self) -> &R {
        &self.range_fetch
    }

    /// Consumes the source, returning the range reader.
    #[must_use]
    pub fn into_range_fetch(self) -> R {
        self.range_fetch
    }

    /// The sequence of range reads a COG open-and-read performs.
    ///
    /// Every figure is the one the reader uses: the first request really is
    /// [`COG_HEADER_PROBE_BYTES`] long, a value outside it really is fetched in
    /// [`COG_IFD_WINDOW_BYTES`] windows, the chain walk really stops at
    /// [`COG_MAX_OVERVIEW_LEVELS`], and [`CogSource::compose`] really keeps
    /// [`COG_MAX_CONCURRENT_TILE_FETCHES`] requests in flight. This module's
    /// tests pin the first two against observed requests and the last against
    /// an observed peak, so the description cannot drift from the behaviour.
    #[must_use]
    pub const fn read_plan() -> [CogReadStep; 4] {
        [
            CogReadStep::HeaderProbe {
                bytes: COG_HEADER_PROBE_BYTES,
            },
            CogReadStep::PrimaryIfd {
                window_bytes: COG_IFD_WINDOW_BYTES,
            },
            CogReadStep::OverviewChain {
                max_levels: COG_MAX_OVERVIEW_LEVELS,
            },
            CogReadStep::TileRange {
                max_concurrent: COG_MAX_CONCURRENT_TILE_FETCHES,
            },
        ]
    }
}

impl<R: RangeFetch + MaybeSync> CogSource<R> {
    /// Reads the header and every IFD, producing the tile directories.
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Fetch`] from the range reader and
    /// [`RenderError::Decode`]/[`RenderError::Unsupported`] from the parser.
    pub fn open(&self) -> TileFuture<'_, Result<CogMetadata, RenderError>> {
        Box::pin(async move { self.open_with_options().await.map(|(metadata, _)| metadata) })
    }

    /// Reads the header and every IFD, reporting the decode options the file
    /// declares alongside the metadata.
    ///
    /// `GDAL_NODATA` (42113) and `JPEGTables` (347) are per-file properties
    /// [`CogMetadata`] carries no field for, and a JPEG COG cannot be decoded
    /// without the latter — so feed the result back with
    /// [`CogSource::with_options`]:
    ///
    /// ```ignore
    /// let (metadata, options) = source.open_with_options().await?;
    /// let source = source.with_options(options);
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Fetch`] from the range reader and
    /// [`RenderError::Decode`]/[`RenderError::Unsupported`] from the parser.
    pub fn open_with_options(
        &self,
    ) -> TileFuture<'_, Result<(CogMetadata, CogDecodeOptions), RenderError>> {
        Box::pin(async move {
            let mut open = CogOpen::new();
            // One iteration per range the parse needs; the bound is the IFD
            // chain limit plus a few external-array reads per level.
            for _ in 0..(COG_MAX_OVERVIEW_LEVELS * 4 + 8) {
                let range = match open.poll()? {
                    CogOpenProgress::Need(range) => range,
                    CogOpenProgress::Ready(_) => break,
                };
                let bytes = self.range_fetch.fetch_range(range).await?;
                open.supply(range.start, bytes);
            }
            let mut options = open.decode_options();
            // The file is authoritative for what it declares; the source's own
            // settings fill in what it does not, so feeding the result back
            // through `with_options` never erases a caller's configuration.
            options.stretch = self.options.stretch;
            options.nodata = options.nodata.or(self.options.nodata);
            if options.jpeg_tables.is_empty() {
                options.jpeg_tables.clone_from(&self.options.jpeg_tables);
            }
            let metadata = open.into_metadata().ok_or_else(|| {
                RenderError::Decode("COG open did not converge on a directory".to_owned())
            })?;
            Ok((metadata, options))
        })
    }

    /// Reads and decodes one tile of one resolution level to RGBA8.
    ///
    /// The result is `tile_width × tile_height` pixels of that level, *not* a
    /// map tile; see [`CogSource::map_tile`] for the latter.
    ///
    /// # Errors
    ///
    /// Propagates the range reader's [`RenderError::Fetch`], and the codec
    /// errors documented on [`decode_cog_block`]. A sparse tile decodes to a
    /// fully transparent buffer rather than failing.
    pub fn read_tile<'a>(
        &'a self,
        metadata: &'a CogMetadata,
        level: usize,
        tile_x: u32,
        tile_y: u32,
    ) -> TileFuture<'a, Result<Vec<u8>, RenderError>> {
        Box::pin(async move {
            let source = metadata
                .level(level)
                .ok_or_else(|| RenderError::Decode(format!("COG has no level {level}")))?;
            let Some(range) = source.tile_range(tile_x, tile_y)? else {
                // Mirrors the compositor's own sizing: a declared 65536x65536
                // tile is 17 GiB of zeros on native and an overflowed `u32` on
                // wasm32, so the product is checked rather than assumed.
                let pixels = (source.tile_width as usize)
                    .checked_mul(source.tile_height as usize)
                    .and_then(|pixels| pixels.checked_mul(4))
                    .ok_or_else(|| RenderError::Decode("COG tile geometry overflows".to_owned()))?;
                return Ok(vec![0u8; pixels]);
            };
            let payload = self.range_fetch.fetch_range(range).await?;
            decode_cog_block(
                source,
                tile_y,
                metadata.little_endian,
                &payload,
                &self.options,
            )
        })
    }

    /// Reads whatever `plan` needs and composes the map tile.
    ///
    /// The source tiles are fetched with up to
    /// [`COG_MAX_CONCURRENT_TILE_FETCHES`] requests in flight: a map tile is
    /// normally built from four source tiles, and reading them one after
    /// another costs four round trips where one would do.
    ///
    /// # Errors
    ///
    /// Propagates [`CogSource::read_tile`] and
    /// [`CogMetadata::compose_tile`].
    pub fn compose<'a>(
        &'a self,
        metadata: &'a CogMetadata,
        plan: &'a CogTilePlan,
    ) -> TileFuture<'a, Result<DecodedTile, RenderError>> {
        Box::pin(async move {
            let mut references = Vec::with_capacity(plan.sources.len());
            let mut reads = Vec::with_capacity(plan.sources.len());
            for reference in &plan.sources {
                if reference.range.is_none() {
                    continue;
                }
                references.push((reference.tile_x, reference.tile_y));
                reads.push(self.read_tile(
                    metadata,
                    plan.level,
                    reference.tile_x,
                    reference.tile_y,
                ));
            }
            let decoded = join_bounded(reads, COG_MAX_CONCURRENT_TILE_FETCHES).await;

            let mut sources = Vec::with_capacity(references.len());
            for ((tile_x, tile_y), result) in references.into_iter().zip(decoded) {
                let rgba = result.ok_or_else(|| {
                    RenderError::Decode(format!(
                        "COG source tile ({tile_x}, {tile_y}) was never driven to completion"
                    ))
                })??;
                sources.push(CogSourceTile {
                    tile_x,
                    tile_y,
                    rgba,
                });
            }
            metadata.compose_tile(plan, &sources)
        })
    }

    /// Plans, reads and composes one Web Mercator map tile.
    ///
    /// `Ok(None)` means the tile does not overlap the image.
    ///
    /// # Errors
    ///
    /// Propagates [`CogMetadata::plan_tile`] and [`CogSource::compose`].
    pub fn map_tile<'a>(
        &'a self,
        metadata: &'a CogMetadata,
        tile: TileId,
    ) -> TileFuture<'a, Result<Option<DecodedTile>, RenderError>> {
        Box::pin(async move {
            let Some(plan) = metadata.plan_tile(tile)? else {
                return Ok(None);
            };
            self.compose(metadata, &plan).await.map(Some)
        })
    }
}

/// Drives `reads` with at most `limit` of them in flight, in input order.
///
/// Hand-rolled over [`TileFuture`] rather than taken from `futures`'
/// `buffer_unordered`, because the two type aliases in [`crate::source`] are
/// the crate's *only* production use of that meta-crate and a combinator here
/// would make it a permanent dependency of the wasm bundle. A boxed future is
/// already `Pin`ned, so polling a set of them needs nothing but
/// [`core::future::poll_fn`].
///
/// The `i`-th slot of the result is `None` only if the `i`-th future was never
/// driven, which the loop below cannot produce; callers still handle it rather
/// than indexing blind.
async fn join_bounded<T>(reads: Vec<TileFuture<'_, T>>, limit: usize) -> Vec<Option<T>> {
    let mut queue = reads.into_iter().enumerate();
    let mut in_flight: Vec<(usize, TileFuture<'_, T>)> = Vec::new();
    let mut results: Vec<Option<T>> = Vec::new();
    let width = limit.max(1);

    core::future::poll_fn(move |cx| {
        loop {
            while in_flight.len() < width {
                let Some((index, read)) = queue.next() else {
                    break;
                };
                if results.len() <= index {
                    results.resize_with(index + 1, || None);
                }
                in_flight.push((index, read));
            }
            if in_flight.is_empty() {
                return core::task::Poll::Ready(core::mem::take(&mut results));
            }
            // Every pending future is polled before returning, so each has
            // registered the current waker; a wakeup from any of them re-enters
            // here and finds the one that is ready.
            let mut progressed = false;
            let mut cursor = 0usize;
            while cursor < in_flight.len() {
                let Some((_, read)) = in_flight.get_mut(cursor) else {
                    break;
                };
                match read.as_mut().poll(cx) {
                    core::task::Poll::Ready(value) => {
                        let (index, _) = in_flight.swap_remove(cursor);
                        if let Some(slot) = results.get_mut(index) {
                            *slot = Some(value);
                        }
                        progressed = true;
                    }
                    core::task::Poll::Pending => cursor += 1,
                }
            }
            if !progressed {
                return core::task::Poll::Pending;
            }
        }
    })
    .await
}

/// A [`RangeFetch`] over an in-memory buffer.
///
/// The natural way to test the reader without an HTTP stack, and useful in its
/// own right for a shell that has already downloaded (or memory-mapped) a
/// GeoTIFF and wants to read it through the same code path as a remote one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRangeFetch {
    /// The whole resource.
    bytes: Vec<u8>,
}

impl MemoryRangeFetch {
    /// Wraps a complete file.
    #[must_use]
    pub const fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// The wrapped bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl RangeFetch for MemoryRangeFetch {
    fn content_length(&self) -> TileFuture<'_, Result<u64, RenderError>> {
        Box::pin(async move { Ok(self.bytes.len() as u64) })
    }

    fn fetch_range(&self, range: ByteRange) -> TileFuture<'_, Result<Vec<u8>, RenderError>> {
        Box::pin(async move {
            let start = usize::try_from(range.start)
                .map_err(|_| RenderError::Fetch("range start overflows usize".to_owned()))?;
            if start >= self.bytes.len() {
                return Err(RenderError::Fetch(format!(
                    "range {}..{} starts past the end of a {}-byte resource",
                    range.start,
                    range.end,
                    self.bytes.len()
                )));
            }
            // A range that runs past the end is clamped, exactly as an HTTP
            // server answers `Range: bytes=0-65535` on a shorter file.
            let end = usize::try_from(range.end)
                .unwrap_or(usize::MAX)
                .min(self.bytes.len());
            Ok(self.bytes.get(start..end).unwrap_or(&[]).to_vec())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        COG_HEADER_PROBE_BYTES, COG_IFD_WINDOW_BYTES, COG_MAX_CONCURRENT_TILE_FETCHES, CogReadStep,
        CogSource, MemoryRangeFetch,
    };
    use crate::cog::codec::RasterStretch;
    use crate::cog::fixture::{TiffFixture, tiled_geo_tiff};
    use crate::cog::meta::{COMPRESSION_DEFLATE, COMPRESSION_LZW, COMPRESSION_PACKBITS};
    use crate::cog::open::{CogOpen, CogOpenProgress};
    use crate::cog::plan::CogSourceTile;
    use crate::error::RenderError;
    use crate::mercator::LonLat;
    use crate::source::{ByteRange, RangeFetch};

    fn source(fixture: &TiffFixture) -> CogSource<MemoryRangeFetch> {
        CogSource::new(MemoryRangeFetch::new(fixture.bytes.clone()))
    }

    #[test]
    fn the_read_plan_describes_the_requests_the_reader_actually_issues() {
        let plan = CogSource::<MemoryRangeFetch>::read_plan();
        assert_eq!(
            plan[0],
            CogReadStep::HeaderProbe {
                bytes: COG_HEADER_PROBE_BYTES
            }
        );
        assert!(matches!(plan[1], CogReadStep::PrimaryIfd { .. }));
        assert!(matches!(plan[2], CogReadStep::OverviewChain { .. }));
        assert!(matches!(plan[3], CogReadStep::TileRange { .. }));

        // The first step is only honest if the first request is that size: the
        // constant used to say 16 bytes while the reader asked for 64 KiB.
        let mut open = CogOpen::new();
        let Ok(CogOpenProgress::Need(first)) = open.poll() else {
            panic!("the first poll must ask for bytes");
        };
        assert_eq!(first.start, 0);
        assert_eq!(first.len(), COG_HEADER_PROBE_BYTES);

        // …and the follow-up window, for a directory parked outside it.
        let fixture = TiffFixture::builder().directory_gap(200_000).build();
        let mut open = CogOpen::new();
        let mut windows = Vec::new();
        for _ in 0..16 {
            match open.poll().expect("the parse must not fail") {
                CogOpenProgress::Need(range) => {
                    windows.push(range.len());
                    let start = range.start as usize;
                    let end = (range.end as usize).min(fixture.bytes.len());
                    let slice = fixture.bytes.get(start..end).unwrap_or(&[]).to_vec();
                    open.supply(range.start, slice);
                }
                CogOpenProgress::Ready(_) => break,
            }
        }
        assert!(
            windows
                .get(1..)
                .is_some_and(|tail| tail.iter().all(|len| *len >= COG_IFD_WINDOW_BYTES)),
            "every follow-up read is at least one IFD window: {windows:?}"
        );
    }

    #[test]
    fn source_tiles_are_fetched_concurrently_rather_than_one_after_another() {
        use core::sync::atomic::{AtomicUsize, Ordering};

        /// A fetcher that records how many requests are in flight at once.
        ///
        /// `fetch_range` yields once before answering, so a sequential caller
        /// never has two outstanding and a concurrent one does. Atomics rather
        /// than cells because a native `RangeFetch` has to be `Sync`.
        struct CountingFetch {
            inner: MemoryRangeFetch,
            live: AtomicUsize,
            peak: AtomicUsize,
            total: AtomicUsize,
        }

        impl RangeFetch for CountingFetch {
            fn content_length(&self) -> crate::source::TileFuture<'_, Result<u64, RenderError>> {
                self.inner.content_length()
            }

            fn fetch_range(
                &self,
                range: ByteRange,
            ) -> crate::source::TileFuture<'_, Result<Vec<u8>, RenderError>> {
                Box::pin(async move {
                    self.total.fetch_add(1, Ordering::Relaxed);
                    let live = self.live.fetch_add(1, Ordering::Relaxed) + 1;
                    self.peak.fetch_max(live, Ordering::Relaxed);
                    yield_once().await;
                    self.live.fetch_sub(1, Ordering::Relaxed);
                    self.inner.fetch_range(range).await
                })
            }
        }

        /// Returns `Pending` exactly once, waking itself immediately.
        async fn yield_once() {
            let mut yielded = false;
            core::future::poll_fn(move |cx| {
                if yielded {
                    core::task::Poll::Ready(())
                } else {
                    yielded = true;
                    cx.waker().wake_by_ref();
                    core::task::Poll::Pending
                }
            })
            .await;
        }

        let fixture = tiled_geo_tiff();
        let source = CogSource::new(CountingFetch {
            inner: MemoryRangeFetch::new(fixture.bytes.clone()),
            live: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
        });
        let metadata = futures::executor::block_on(source.open()).expect("the fixture must open");
        source.range_fetch().peak.store(0, Ordering::Relaxed);
        source.range_fetch().total.store(0, Ordering::Relaxed);

        // Planned against level 0 at a zoom whose map tile spans the whole
        // 8x8 image, so all four 4x4 source tiles are needed at once.
        let plan = (4u8..=8)
            .filter_map(|zoom| {
                let tile = LonLat::new(12.0, 48.0).tile(zoom).ok()?;
                metadata.plan_tile_at(tile, 0).ok().flatten()
            })
            .find(|plan| plan.sources.len() > 1)
            .expect("the test needs a multi-source plan to say anything");
        let composed = futures::executor::block_on(source.compose(&metadata, &plan))
            .expect("composition must succeed");
        assert_eq!(
            source.range_fetch().total.load(Ordering::Relaxed),
            plan.sources.len()
        );

        // Concurrency completes tiles out of order, so the results have to be
        // put back in plan order before composition — a shuffle here would
        // paint each quadrant with a neighbour's pixels and still be the right
        // length, which no size check would catch.
        let sequential: Vec<CogSourceTile> = plan
            .sources
            .iter()
            .map(|reference| CogSourceTile {
                tile_x: reference.tile_x,
                tile_y: reference.tile_y,
                rgba: futures::executor::block_on(source.read_tile(
                    &metadata,
                    plan.level,
                    reference.tile_x,
                    reference.tile_y,
                ))
                .expect("every source tile must decode"),
            })
            .collect();
        let expected = metadata
            .compose_tile(&plan, &sequential)
            .expect("the sequential composition must succeed");
        assert_eq!(composed.rgba(), expected.rgba());
        assert!(
            source.range_fetch().peak.load(Ordering::Relaxed) > 1,
            "source tiles must be read concurrently, not one round trip each"
        );
        assert!(
            source.range_fetch().peak.load(Ordering::Relaxed) <= COG_MAX_CONCURRENT_TILE_FETCHES,
            "…and never more than the declared limit"
        );
    }

    #[test]
    fn a_striped_tiff_with_a_short_final_strip_composes() {
        // The regression this whole block-row machinery exists for: TIFF pads
        // tiles but not strips, so the last strip of an image whose height is
        // not a multiple of RowsPerStrip decompresses short. Failing it took
        // the entire 256x256 map tile down, not just its bottom band.
        let fixture = TiffFixture::builder().striped(4, 10).build();
        let source = source(&fixture);
        let metadata = futures::executor::block_on(source.open()).expect("the fixture must open");
        let base = metadata.base_level().expect("a base level");
        assert_eq!(base.tiles_down(), 3);

        for tile_y in 0..3u32 {
            let rgba = futures::executor::block_on(source.read_tile(&metadata, 0, 0, tile_y))
                .expect("every strip must decode, the short final one included");
            assert_eq!(rgba.len(), 8 * 4 * 4, "every block is a full-size buffer");
            assert_eq!(
                rgba[0],
                fixture.pixel(0, tile_y * 4).unwrap_or(0),
                "strip {tile_y} must start at its own first row"
            );
        }
        // The short strip's missing rows are transparent, not garbage.
        let last = futures::executor::block_on(source.read_tile(&metadata, 0, 0, 2))
            .expect("the short strip must decode");
        assert!(last[8 * 2 * 4..].iter().all(|byte| *byte == 0));

        let target = LonLat::new(11.0, 48.0).tile(8).expect("a tile");
        let composed = futures::executor::block_on(source.map_tile(&metadata, target))
            .expect("a striped COG must compose")
            .expect("the tile overlaps the image");
        assert_eq!(composed.width(), 256);
    }

    #[test]
    fn a_bigtiff_round_trips_through_the_reader() {
        let fixture = TiffFixture::builder().big_tiff(true).build();
        let source = source(&fixture);
        let metadata = futures::executor::block_on(source.open()).expect("a BigTIFF must open");
        assert_eq!(metadata.level_count(), 2);
        let rgba = futures::executor::block_on(source.read_tile(&metadata, 0, 0, 0))
            .expect("a BigTIFF tile must decode");
        assert_eq!(rgba[0], fixture.pixel(0, 0).unwrap_or(0));
    }

    #[test]
    fn declared_decode_options_survive_the_open() {
        let fixture = TiffFixture::builder().nodata("-9999").build();
        let source = source(&fixture).with_stretch(RasterStretch::Fixed {
            min: 0.0,
            max: 10.0,
        });
        let (metadata, options) =
            futures::executor::block_on(source.open_with_options()).expect("the fixture must open");
        assert_eq!(options.nodata, Some(-9999.0));
        assert_eq!(
            options.stretch,
            RasterStretch::Fixed {
                min: 0.0,
                max: 10.0
            },
            "the source's own stretch is carried through, not reset"
        );
        let source = source.with_options(options);
        assert_eq!(source.options().nodata, Some(-9999.0));
        futures::executor::block_on(source.read_tile(&metadata, 0, 0, 0))
            .expect("the tile must still decode");

        assert_eq!(
            CogSource::new(MemoryRangeFetch::new(Vec::new()))
                .with_nodata(Some(0.0))
                .options()
                .nodata,
            Some(0.0)
        );
    }

    #[test]
    fn an_in_memory_cog_opens_and_reads_tiles() {
        let fixture = tiled_geo_tiff();
        let source = source(&fixture);
        let metadata = futures::executor::block_on(source.open()).expect("the fixture must open");
        assert_eq!(metadata.level_count(), 2);

        let rgba = futures::executor::block_on(source.read_tile(&metadata, 0, 0, 0))
            .expect("tile (0, 0) must decode");
        assert_eq!(rgba.len(), 4 * 4 * 4);
        // Grey source, so R == G == B, and the value is the fixture's gradient.
        assert_eq!(rgba[0], fixture.pixel(0, 0).unwrap_or(0));
        assert_eq!(rgba[1], rgba[0]);
        assert_eq!(rgba[3], 255);
        // Tile (1, 0) starts at pixel column 4.
        let rgba = futures::executor::block_on(source.read_tile(&metadata, 0, 1, 0))
            .expect("tile (1, 0) must decode");
        assert_eq!(rgba[0], fixture.pixel(4, 0).unwrap_or(0));
    }

    #[test]
    fn every_supported_codec_round_trips_through_the_reader() {
        for compression in [COMPRESSION_DEFLATE, COMPRESSION_LZW, COMPRESSION_PACKBITS] {
            let fixture = TiffFixture::builder().compression(compression).build();
            let source = source(&fixture);
            let metadata =
                futures::executor::block_on(source.open()).expect("the fixture must open");
            let rgba = futures::executor::block_on(source.read_tile(&metadata, 0, 0, 0))
                .expect("the tile must decode");
            assert_eq!(
                rgba[0],
                fixture.pixel(0, 0).unwrap_or(0),
                "codec {compression} must reproduce the source pixel"
            );
        }
    }

    #[test]
    fn a_predicted_tile_round_trips_through_the_reader() {
        let fixture = TiffFixture::builder()
            .predictor(true)
            .compression(COMPRESSION_DEFLATE)
            .build();
        let source = source(&fixture);
        let metadata = futures::executor::block_on(source.open()).expect("the fixture must open");
        let rgba = futures::executor::block_on(source.read_tile(&metadata, 0, 0, 0))
            .expect("the tile must decode");
        for x in 0..4u32 {
            assert_eq!(
                rgba[(x as usize) * 4],
                fixture.pixel(x, 0).unwrap_or(0),
                "predictor must be undone at column {x}"
            );
        }
    }

    #[test]
    fn a_cog_without_overviews_serves_every_zoom_from_level_zero() {
        let fixture = TiffFixture::builder().overview(false).build();
        let source = source(&fixture);
        let metadata = futures::executor::block_on(source.open()).expect("the fixture must open");
        assert_eq!(metadata.level_count(), 1);
        assert_eq!(metadata.select_level(0).ok(), Some(0));
        assert_eq!(metadata.select_level(20).ok(), Some(0));
    }

    #[test]
    fn a_big_endian_cog_round_trips_through_the_reader() {
        let fixture = TiffFixture::builder().big_endian(true).build();
        let source = source(&fixture);
        let metadata = futures::executor::block_on(source.open()).expect("the fixture must open");
        let rgba = futures::executor::block_on(source.read_tile(&metadata, 0, 0, 0))
            .expect("the tile must decode");
        assert_eq!(rgba[0], fixture.pixel(0, 0).unwrap_or(0));
    }

    #[test]
    fn a_map_tile_is_planned_read_and_composed() {
        let fixture = tiled_geo_tiff();
        let source = source(&fixture);
        let metadata = futures::executor::block_on(source.open()).expect("the fixture must open");
        // The fixture covers 10..14 °E, 46..50 °N at 0.5 °/px.
        let target = LonLat::new(11.0, 49.0).tile(8).expect("a tile");
        let composed = futures::executor::block_on(source.map_tile(&metadata, target))
            .expect("composition must succeed")
            .expect("the tile overlaps the image");
        assert_eq!(composed.width(), 256);
        assert!(
            composed.rgba().chunks_exact(4).any(|pixel| pixel[3] == 255),
            "the tile is inside the image, so it must have opaque pixels"
        );

        // A tile on the other side of the planet overlaps nothing.
        let elsewhere = LonLat::new(-150.0, -40.0).tile(8).expect("a tile");
        assert!(
            futures::executor::block_on(source.map_tile(&metadata, elsewhere))
                .expect("planning must not fail")
                .is_none()
        );
    }

    #[test]
    fn a_sparse_tile_reads_as_transparent() {
        let fixture = tiled_geo_tiff();
        let source = source(&fixture);
        let mut metadata = futures::executor::block_on(source.open()).expect("the fixture opens");
        metadata.levels[0].tile_byte_counts[0] = 0;
        let rgba = futures::executor::block_on(source.read_tile(&metadata, 0, 0, 0))
            .expect("a sparse tile must not fail");
        assert!(rgba.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn the_stretch_is_configurable_and_carried() {
        let fixture = tiled_geo_tiff();
        let stretch = RasterStretch::Fixed {
            min: 0.0,
            max: 100.0,
        };
        let source = source(&fixture).with_stretch(stretch);
        assert_eq!(source.stretch(), stretch);
        assert_eq!(source.range_fetch().bytes().len(), fixture.bytes.len());
        assert_eq!(source.into_range_fetch().bytes().len(), fixture.bytes.len());
    }

    #[test]
    fn an_unreadable_level_or_tile_is_an_error() {
        let fixture = tiled_geo_tiff();
        let source = source(&fixture);
        let metadata = futures::executor::block_on(source.open()).expect("the fixture opens");
        assert!(futures::executor::block_on(source.read_tile(&metadata, 9, 0, 0)).is_err());
        assert!(futures::executor::block_on(source.read_tile(&metadata, 0, 9, 9)).is_err());
    }

    #[test]
    fn a_non_tiff_buffer_fails_to_open() {
        let source = CogSource::new(MemoryRangeFetch::new(vec![0u8; 4_096]));
        assert!(matches!(
            futures::executor::block_on(source.open()),
            Err(RenderError::Decode(_))
        ));
    }

    #[test]
    fn the_memory_fetcher_clamps_and_rejects_out_of_range_reads() {
        let fetch = MemoryRangeFetch::new((0u8..32).collect());
        assert_eq!(
            futures::executor::block_on(fetch.content_length()).ok(),
            Some(32)
        );
        let Ok(range) = ByteRange::new(24, 1_024) else {
            panic!("range rejected");
        };
        assert_eq!(
            futures::executor::block_on(fetch.fetch_range(range)).ok(),
            Some((24u8..32).collect::<Vec<u8>>())
        );
        let Ok(beyond) = ByteRange::new(64, 128) else {
            panic!("range rejected");
        };
        assert!(futures::executor::block_on(fetch.fetch_range(beyond)).is_err());
    }
}
