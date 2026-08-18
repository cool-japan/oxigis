// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Offline end-to-end tests for the archive layer.
//!
//! Everything here runs against [`MemoryRangeTransport`] over the hand-built
//! fixtures `oxigis-render` ships behind its `fixtures` feature, so the whole
//! read path — open, leaf hop, tile-body decode, raster decode, MVT decode,
//! tessellation — is exercised with no network, no filesystem and no `#[ignore]`
//! on any target.

use std::sync::Arc;

use oxigis_core::ArchiveFormat;
use oxigis_render::pmtiles::{
    Compression, PmtilesBuilder, TileType, sample_pmtiles_far_metadata, sample_pmtiles_raster,
    sample_pmtiles_vector,
};
use oxigis_render::{ByteRange, LonLat, MapView, TileId};
use parking_lot::Mutex;

use crate::archive::open::ArchiveContent;
use crate::archive::paints::{ArchivePaintKind, archive_paints, kind_for};
use crate::archive::{
    ArchiveInfo, ArchiveProbe, ArchiveTileProvider, ArchiveTileTransport, MemoryRangeTransport,
};
use crate::cog_provider::{RangeJob, RangeSink, RangeTransport};
use crate::map_gpu::TileProvider;
// The PNG builders live in `mbtiles::fixture` because the `fixtures` feature
// exports one archive built from `tiny_png`; they are the same functions, moved
// rather than copied, so the `[10, 120, 200]` wrong-archive tripwire stays a
// single value.
use crate::mbtiles::fixture::{push_chunk, tiny_png};
use crate::tile_provider::TileError;
use crate::vector_provider::{VectorTileConfig, VectorTileProvider, VectorTileSource as _};

/// A tile address the fixtures all hold something for.
fn tile(z: u8, x: u32, y: u32) -> TileId {
    TileId::new(z, x, y).unwrap_or_else(|error| panic!("tile {z}/{x}/{y} must be valid: {error}"))
}

/// The location string every test uses; the memory transport ignores it.
const WHERE: &str = "memory://fixture.pmtiles";

/// Every `(range, job)` a [`RecordingTransport`] was asked for.
type RangeLog = Arc<Mutex<Vec<(ByteRange, RangeJob)>>>;

/// Every result a [`Collector`] was handed.
type AnswerLog = Arc<Mutex<Vec<Result<Vec<u8>, TileError>>>>;

/// A transport that records every range it is asked for before answering out of
/// memory, so a test can assert what was *read* and not only what came back.
struct RecordingTransport {
    /// The archive bytes.
    inner: MemoryRangeTransport,
    /// Every `(range, job)` asked for, in order.
    log: RangeLog,
}

impl RecordingTransport {
    fn new(bytes: Vec<u8>) -> (Self, RangeLog) {
        let log = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                inner: MemoryRangeTransport::new(bytes),
                log: Arc::clone(&log),
            },
            log,
        )
    }
}

impl RangeTransport for RecordingTransport {
    fn request_range(&self, url: String, range: ByteRange, job: RangeJob, sink: RangeSink) {
        self.log.lock().push((range, job));
        self.inner.request_range(url, range, job, sink);
    }
}

/// A transport that never answers, leaving every read outstanding.
struct SilentTransport;

impl RangeTransport for SilentTransport {
    fn request_range(&self, _url: String, _range: ByteRange, _job: RangeJob, _sink: RangeSink) {}
}

/// A transport whose every answer is a transient failure.
struct BrokenTransport;

impl RangeTransport for BrokenTransport {
    fn request_range(&self, _url: String, _range: ByteRange, job: RangeJob, sink: RangeSink) {
        sink.deliver(job, Err(TileError::transient("connection reset")));
    }
}

/// A minimal but genuinely decodable MVT tile: one named layer holding one
/// triangle on the standard 4096 extent, hand-encoded so the test needs no
/// encoder.
///
/// The `oxigis-render` fixtures' own bodies are deliberately *shaped* like MVT
/// rather than valid — they exist to prove the directory walk, not the decoder —
/// so the tests that drive a real `VectorTileProvider` build their archives out
/// of these instead.
fn mvt_tile(layer_name: &str) -> Vec<u8> {
    let geometry: [u32; 8] = [9, 0, 0, 18, 4096, 0, 0, 4096];
    let mut geometry_bytes = Vec::new();
    for value in geometry {
        varint(&mut geometry_bytes, u64::from(value));
    }
    varint(&mut geometry_bytes, 15); // ClosePath

    let mut feature = Vec::new();
    feature.push(3 << 3); // type
    varint(&mut feature, 3); // POLYGON
    feature.push((4 << 3) | 2); // geometry
    varint(&mut feature, geometry_bytes.len() as u64);
    feature.extend_from_slice(&geometry_bytes);

    let mut layer = Vec::new();
    layer.push(15 << 3); // version
    varint(&mut layer, 2);
    layer.push((1 << 3) | 2); // name
    varint(&mut layer, layer_name.len() as u64);
    layer.extend_from_slice(layer_name.as_bytes());
    layer.push((2 << 3) | 2); // feature
    varint(&mut layer, feature.len() as u64);
    layer.extend_from_slice(&feature);
    layer.push(5 << 3); // extent
    varint(&mut layer, 4096);

    let mut tile = Vec::new();
    tile.push((3 << 3) | 2); // layers
    varint(&mut tile, layer.len() as u64);
    tile.extend_from_slice(&layer);
    tile
}

/// Appends `value` as a protobuf base-128 varint.
fn varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// A root-only vector archive whose five addresses all decode as real MVT.
fn decodable_vector_archive() -> Vec<u8> {
    let mut builder = PmtilesBuilder::new(TileType::Mvt);
    builder.push_tile(0, 0, 0, mvt_tile("land"));
    for (index, (x, y)) in [(0, 0), (0, 1), (1, 1), (1, 0)].into_iter().enumerate() {
        builder.push_tile(1, x, y, mvt_tile(&format!("land{index}")));
    }
    builder.build()
}

/// A leafed vector archive with **many** leaves: 64 distinct zoom-3 bodies at a
/// leaf threshold of four, so the root is a list of sixteen leaf pointers and a
/// trace that hops around the world touches most of them.
fn many_leaf_archive() -> Vec<u8> {
    let mut builder = PmtilesBuilder::new(TileType::Mvt).with_leaf_threshold(4);
    for x in 0..8u32 {
        for y in 0..8u32 {
            // Distinct bodies: identical ones would be deduplicated into runs
            // and the archive would end up with far fewer entries than leaves.
            builder.push_tile(3, x, y, mvt_tile(&format!("land_{x}_{y}")));
        }
    }
    builder.build()
}

/// The same five addresses, forced through a real leaf level.
fn decodable_leafed_archive() -> Vec<u8> {
    let mut builder = PmtilesBuilder::new(TileType::Mvt).with_leaf_threshold(2);
    builder.push_tile(0, 0, 0, mvt_tile("land"));
    for (index, (x, y)) in [(0, 0), (0, 1), (1, 1), (1, 0)].into_iter().enumerate() {
        builder.push_tile(1, x, y, mvt_tile(&format!("land{index}")));
    }
    builder.build()
}

// ---------------------------------------------------------------------------
// MemoryRangeTransport
// ---------------------------------------------------------------------------

#[test]
fn the_memory_transport_answers_synchronously_and_clamps_at_the_end() {
    let transport = MemoryRangeTransport::new(vec![1, 2, 3, 4, 5]);
    assert_eq!(transport.len(), 5);
    assert!(!transport.is_empty());

    let answers: AnswerLog = Arc::new(Mutex::new(Vec::new()));
    let collector = Collector {
        answers: Arc::clone(&answers),
    };
    let sink = RangeSink::from_delivery(Arc::new(collector));
    let range = ByteRange::new(2, 100).expect("a non-empty range");
    transport.request_range(
        WHERE.to_owned(),
        range,
        RangeJob::ArchiveHeader { start: 2 },
        sink,
    );
    let answers = answers.lock();
    assert_eq!(answers.len(), 1, "answered inside request_range");
    assert_eq!(answers[0].as_deref(), Ok(&[3, 4, 5][..]));
}

#[test]
fn a_read_that_starts_past_the_end_is_a_permanent_failure() {
    let transport = MemoryRangeTransport::new(vec![1, 2, 3]);
    let answers: AnswerLog = Arc::new(Mutex::new(Vec::new()));
    let sink = RangeSink::from_delivery(Arc::new(Collector {
        answers: Arc::clone(&answers),
    }));
    transport.request_range(
        WHERE.to_owned(),
        ByteRange::new(9, 12).expect("a non-empty range"),
        RangeJob::ArchiveHeader { start: 9 },
        sink,
    );
    let answers = answers.lock();
    let error = answers[0].as_ref().expect_err("past the end");
    assert!(!error.retryable());
    assert!(error.message().contains("past the end"));
}

/// Collects whatever a [`RangeSink`] is handed.
struct Collector {
    /// Every delivered result, in order.
    answers: AnswerLog,
}

impl crate::cog_provider::RangeDelivery for Collector {
    fn deliver_range(self: Arc<Self>, _job: RangeJob, result: Result<Vec<u8>, TileError>) {
        self.answers.lock().push(result);
    }
}

// ---------------------------------------------------------------------------
// The probe
// ---------------------------------------------------------------------------

/// Runs a probe to completion over the given bytes.
fn probe(bytes: Vec<u8>) -> Result<crate::archive::OpenedArchive, TileError> {
    let ctx = egui::Context::default();
    let probe = ArchiveProbe::start(
        WHERE,
        ArchiveFormat::PmTiles,
        &ctx,
        Box::new(MemoryRangeTransport::new(bytes)),
    );
    assert_eq!(probe.location(), WHERE);
    probe.take().expect("the memory transport answers inline")
}

#[test]
fn the_probe_identifies_a_vector_archive_and_its_metadata() {
    let opened = probe(sample_pmtiles_vector()).expect("the fixture opens");
    assert_eq!(opened.location, WHERE);
    let info = opened.info;
    assert_eq!(info.content, ArchiveContent::Vector);
    assert_eq!(info.codec, TileType::Mvt);
    assert_eq!(info.min_zoom, 0);
    assert_eq!(info.max_zoom, 1);
    assert_eq!(info.name, "fixture");
    assert_eq!(info.attribution, "OxiGIS test fixture");
    assert_eq!(info.layer_names, vec!["land".to_owned()]);
    assert!(info.has_bounds);
    assert!(info.summary().contains("vector tiles"));
}

#[test]
fn the_probe_identifies_a_raster_archive() {
    let opened = probe(sample_pmtiles_raster()).expect("the fixture opens");
    assert_eq!(opened.info.content, ArchiveContent::Raster);
    assert_eq!(opened.info.codec, TileType::Png);
    assert!(opened.info.summary().contains("PNG"));
}

#[test]
fn the_probe_takes_its_answer_exactly_once() {
    let ctx = egui::Context::default();
    let probe = ArchiveProbe::start(
        WHERE,
        ArchiveFormat::PmTiles,
        &ctx,
        Box::new(MemoryRangeTransport::new(sample_pmtiles_vector())),
    );
    assert!(probe.take().is_some());
    assert!(probe.take().is_none(), "the answer is take-once");
}

#[test]
fn a_probe_over_a_pending_transport_answers_nothing_yet() {
    let ctx = egui::Context::default();
    let probe = ArchiveProbe::start(
        WHERE,
        ArchiveFormat::PmTiles,
        &ctx,
        Box::new(SilentTransport),
    );
    assert!(probe.take().is_none());
    assert!(probe.take().is_none());
}

#[test]
fn a_two_need_open_reaches_the_far_metadata_block() {
    let bytes = sample_pmtiles_far_metadata();
    let (transport, log) = RecordingTransport::new(bytes);
    let ctx = egui::Context::default();
    let probe = ArchiveProbe::start(WHERE, ArchiveFormat::PmTiles, &ctx, Box::new(transport));
    let opened = probe
        .take()
        .expect("answered inline")
        .expect("the fixture opens");
    assert_eq!(opened.info.attribution, "OxiGIS test fixture");
    let reads = log.lock();
    assert_eq!(reads.len(), 2, "one prefetch plus one far metadata read");
    assert_eq!(reads[0].0.start, 0);
    assert!(
        reads[1].0.start > oxigis_render::pmtiles::PREFETCH_LEN,
        "the second read is past the prefetch, at {}",
        reads[1].0.start
    );
}

#[test]
fn a_gzip_internal_archive_is_inflated_by_the_ui_side() {
    let mut builder =
        PmtilesBuilder::new(TileType::Mvt).with_compression(Compression::Gzip, Compression::None);
    builder.push_tile(0, 0, 0, vec![0x1a, 0x02, 0x0a, 0x00]);
    let opened = probe(builder.build()).expect("a gzip-internal archive opens");
    assert_eq!(opened.info.content, ArchiveContent::Vector);
    assert_eq!(opened.info.attribution, "OxiGIS test fixture");
}

#[test]
fn an_avif_archive_is_refused_by_name() {
    let mut builder = PmtilesBuilder::new(TileType::Avif);
    builder.push_tile(0, 0, 0, vec![0u8; 8]);
    let error = probe(builder.build()).expect_err("AVIF is refused");
    assert!(!error.retryable());
    assert!(error.message().contains("AVIF"), "{}", error.message());
}

#[test]
fn a_brotli_archive_is_refused_by_name() {
    let mut builder =
        PmtilesBuilder::new(TileType::Mvt).with_compression(Compression::None, Compression::Brotli);
    builder.push_tile(0, 0, 0, vec![0u8; 8]);
    let error = probe(builder.build()).expect_err("brotli is refused");
    assert!(error.message().contains("brotli"), "{}", error.message());
}

#[test]
fn an_undeclared_tile_type_is_refused_by_name() {
    let mut builder = PmtilesBuilder::new(TileType::Unknown);
    builder.push_tile(0, 0, 0, vec![0u8; 8]);
    let error = probe(builder.build()).expect_err("an unknown tile type is refused");
    assert!(
        error
            .message()
            .contains("does not declare what its tiles are"),
        "{}",
        error.message()
    );
}

#[test]
fn a_body_that_is_not_an_archive_at_all_is_refused() {
    let short = probe(b"nope".to_vec()).expect_err("a body too short to hold a header is refused");
    assert!(!short.retryable());
    assert!(short.message().contains("truncated"), "{}", short.message());

    let garbage = probe(vec![0x7Au8; 512]).expect_err("garbage is refused");
    assert!(!garbage.retryable());
    assert!(
        garbage.message().contains("PMTiles"),
        "{}",
        garbage.message()
    );
}

#[test]
fn an_archive_with_no_metadata_still_opens() {
    let mut builder = PmtilesBuilder::new(TileType::Png).with_metadata("");
    builder.push_tile(0, 0, 0, vec![0u8; 8]);
    let opened = probe(builder.build()).expect("no metadata is not a refusal");
    assert!(opened.info.attribution.is_empty());
    assert!(opened.info.layer_names.is_empty());
    assert!(opened.info.tile_size_px.is_none());
}

#[test]
fn a_metadata_tile_size_is_read_as_a_string_or_a_number() {
    let mut builder = PmtilesBuilder::new(TileType::Png).with_metadata(r#"{"tileSize":"512"}"#);
    builder.push_tile(0, 0, 0, vec![0u8; 8]);
    assert_eq!(
        probe(builder.build()).expect("opens").info.tile_size_px,
        Some(512)
    );

    let mut builder = PmtilesBuilder::new(TileType::Png).with_metadata(r#"{"tileSize":256}"#);
    builder.push_tile(0, 0, 0, vec![0u8; 8]);
    assert_eq!(
        probe(builder.build()).expect("opens").info.tile_size_px,
        Some(256)
    );
}

#[test]
fn metadata_that_is_not_json_is_not_a_refusal() {
    let mut builder = PmtilesBuilder::new(TileType::Png).with_metadata("not json at all");
    builder.push_tile(0, 0, 0, vec![0u8; 8]);
    let opened = probe(builder.build()).expect("unreadable metadata is not fatal");
    assert!(opened.info.name.is_empty());
}

// ---------------------------------------------------------------------------
// The raster provider
// ---------------------------------------------------------------------------

/// Builds a raster provider over `bytes` and pumps `tile` until it answers.
fn raster_provider(bytes: Vec<u8>) -> ArchiveTileProvider {
    ArchiveTileProvider::pmtiles(
        WHERE,
        &egui::Context::default(),
        Box::new(MemoryRangeTransport::new(bytes)),
    )
    .expect("the provider must build")
}

/// Asks `provider` for `tile` up to `frames` times, returning the first answer.
fn pump(
    provider: &ArchiveTileProvider,
    tile: TileId,
    frames: usize,
) -> Option<oxigis_render::DecodedTile> {
    for _ in 0..frames {
        if let Some(decoded) = provider.tile(tile) {
            return Some(decoded);
        }
    }
    None
}

#[test]
fn a_raster_archive_tile_decodes_to_pixels() {
    let provider = raster_provider(sample_pmtiles_raster());
    assert!(!provider.is_open(), "nothing is read until the first frame");
    let decoded = pump(&provider, tile(0, 0, 0), 6).expect("the fixture holds z0");
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 2);
    assert_eq!(&decoded.rgba()[..3], &[220, 40, 40]);
    assert!(provider.is_open());
    assert_eq!(provider.stats().ready, 1);
    assert_eq!(provider.location(), WHERE);
    assert!(provider.failure().is_none());
}

#[test]
fn a_cached_raster_tile_is_not_read_twice() {
    let (transport, log) = RecordingTransport::new(sample_pmtiles_raster());
    let provider =
        ArchiveTileProvider::pmtiles(WHERE, &egui::Context::default(), Box::new(transport))
            .expect("builds");
    let address = tile(1, 0, 1);
    assert!(pump(&provider, address, 6).is_some());
    let before = log.lock().len();
    assert!(provider.tile(address).is_some());
    assert_eq!(log.lock().len(), before, "a cached tile is not refetched");
}

#[test]
fn an_address_the_archive_does_not_hold_falls_through_to_the_base() {
    struct RedBase;
    impl TileProvider for RedBase {
        fn tile(&self, _tile: TileId) -> Option<oxigis_render::DecodedTile> {
            oxigis_render::DecodedTile::new(1, 1, vec![255, 0, 0, 255]).ok()
        }
    }
    let provider = raster_provider(sample_pmtiles_raster()).with_base(Box::new(RedBase));
    // z2 is past the fixture's max_zoom, so the archive answers Absent without
    // reading anything at all.
    let base = pump(&provider, tile(2, 0, 0), 6).expect("the basemap shows through");
    assert_eq!(base.rgba(), &[255, 0, 0, 255]);
    assert_eq!(provider.stats().failed, 0, "absent is not a failure");
}

#[test]
fn a_vector_archive_is_refused_by_the_raster_provider_by_name() {
    let provider = raster_provider(sample_pmtiles_vector());
    for _ in 0..4 {
        let _ = provider.tile(tile(0, 0, 0));
    }
    let failure = provider.failure().expect("a mismatch is refused");
    assert!(failure.contains("vector"), "{failure}");
    assert!(failure.contains("raster"), "{failure}");
}

#[test]
fn a_broken_transport_leaves_the_raster_archive_failed_not_looping() {
    let provider =
        ArchiveTileProvider::pmtiles(WHERE, &egui::Context::default(), Box::new(BrokenTransport))
            .expect("builds");
    for _ in 0..8 {
        assert!(provider.tile(tile(0, 0, 0)).is_none());
    }
    assert!(provider.failure().is_some());
}

#[test]
fn a_pending_transport_asks_for_the_header_exactly_once() {
    let provider =
        ArchiveTileProvider::pmtiles(WHERE, &egui::Context::default(), Box::new(SilentTransport))
            .expect("builds");
    for _ in 0..5 {
        assert!(provider.tile(tile(0, 0, 0)).is_none());
    }
    assert!(!provider.is_open());
    assert!(provider.failure().is_none());
}

// ---------------------------------------------------------------------------
// The vector transport, through the real VectorTileProvider
// ---------------------------------------------------------------------------

/// A vector provider fed from an archive, with paints for the fixture's one
/// declared source layer.
fn vector_over(bytes: Vec<u8>) -> VectorTileProvider {
    let config = VectorTileConfig::new("https://example.test/{z}/{x}/{y}.pbf")
        .with_paints(archive_paints(&["land".to_owned()]));
    VectorTileProvider::new(
        &config,
        &egui::Context::default(),
        Box::new(ArchiveTileTransport::pmtiles(
            WHERE,
            Box::new(MemoryRangeTransport::new(bytes)),
        )),
    )
    .expect("the provider must build")
}

fn view(zoom: f64) -> MapView {
    MapView::new(LonLat::new(0.0, 0.0), zoom, [256.0, 256.0]).expect("a valid view")
}

#[test]
fn a_vector_archive_tile_reaches_the_vector_provider_decoded() {
    let provider = vector_over(decodable_vector_archive());
    let _ = provider.begin_frame(view(0.0));
    let root = tile(0, 0, 0);
    // The archive answers synchronously through the memory transport, so the
    // decoded tile is already there when the second ask comes.
    assert!(
        provider.mesh(root).is_none(),
        "the first ask starts the read"
    );
    assert!(
        provider.decoded(root).is_some(),
        "the archive's MVT body reached decode_mvt"
    );
    assert_eq!(provider.stats().inflight, 0);
    assert_eq!(provider.stats().failed, 0);
}

#[test]
fn an_absent_vector_tile_becomes_an_empty_tile_not_a_failure() {
    let provider = vector_over(decodable_vector_archive());
    let _ = provider.begin_frame(view(2.0));
    // Past the fixture's max_zoom: the archive holds nothing here.
    let missing = tile(2, 1, 1);
    assert!(provider.mesh(missing).is_none());
    let decoded = provider
        .decoded(missing)
        .expect("deliver_absent inserts an EMPTY tile");
    assert!(decoded.layers.is_empty(), "an absent tile has no layers");
    assert_eq!(
        provider.stats().failed,
        0,
        "a sparse archive's miss must not burn the failure LRU"
    );
}

#[test]
fn a_leaf_hop_resolves_every_address_the_leafed_fixture_holds() {
    let (transport, log) = RecordingTransport::new(decodable_leafed_archive());
    let config =
        VectorTileConfig::new("https://example.test/{z}/{x}/{y}.pbf").with_paints(Vec::new());
    let provider = VectorTileProvider::new(
        &config,
        &egui::Context::default(),
        Box::new(ArchiveTileTransport::pmtiles(WHERE, Box::new(transport))),
    )
    .expect("builds");
    let _ = provider.begin_frame(view(1.0));
    for (z, x, y) in [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)] {
        let address = tile(z, x, y);
        assert!(provider.mesh(address).is_none());
        assert!(
            provider.decoded(address).is_some(),
            "z{z} {x}/{y} must resolve through its leaf"
        );
    }
    let reads = log.lock();
    assert!(
        reads
            .iter()
            .any(|(_, job)| matches!(job, RangeJob::ArchiveLeaf { .. })),
        "a leaf directory must actually have been read"
    );
    assert!(
        reads
            .iter()
            .any(|(_, job)| matches!(job, RangeJob::ArchiveTile { .. })),
        "and then a tile body"
    );
}

#[test]
fn a_second_tile_in_the_same_leaf_reuses_the_cached_directory() {
    let (transport, log) = RecordingTransport::new(decodable_leafed_archive());
    let config =
        VectorTileConfig::new("https://example.test/{z}/{x}/{y}.pbf").with_paints(Vec::new());
    let provider = VectorTileProvider::new(
        &config,
        &egui::Context::default(),
        Box::new(ArchiveTileTransport::pmtiles(WHERE, Box::new(transport))),
    )
    .expect("builds");
    let _ = provider.begin_frame(view(1.0));
    for (z, x, y) in [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)] {
        let _ = provider.mesh(tile(z, x, y));
    }
    let leaf_reads = log
        .lock()
        .iter()
        .filter(|(_, job)| matches!(job, RangeJob::ArchiveLeaf { .. }))
        .count();
    // The fixture chunks five entries into leaves of two, so three leaves
    // cover five addresses: a per-tile read would be five.
    assert!(
        leaf_reads <= 3,
        "each leaf must be read once, not once per tile (was {leaf_reads})"
    );
}

#[test]
fn a_raster_archive_is_refused_by_the_vector_transport_by_name() {
    let transport = ArchiveTileTransport::pmtiles(
        WHERE,
        Box::new(MemoryRangeTransport::new(sample_pmtiles_raster())),
    );
    let config =
        VectorTileConfig::new("https://example.test/{z}/{x}/{y}.pbf").with_paints(Vec::new());
    let provider = VectorTileProvider::new(&config, &egui::Context::default(), Box::new(transport))
        .expect("builds");
    let _ = provider.begin_frame(view(0.0));
    assert!(provider.mesh(tile(0, 0, 0)).is_none());
    assert!(provider.decoded(tile(0, 0, 0)).is_none());
    assert_eq!(provider.stats().failed, 1, "the mismatch is a tile failure");
}

#[test]
fn the_archive_transport_reports_its_location_and_open_state() {
    let transport = ArchiveTileTransport::pmtiles(
        WHERE,
        Box::new(MemoryRangeTransport::new(sample_pmtiles_vector())),
    );
    assert_eq!(transport.location(), WHERE);
    assert!(!transport.is_open());
    assert!(transport.failure().is_none());
    assert!(format!("{transport:?}").contains("ArchiveTileTransport"));
}

// ---------------------------------------------------------------------------
// The leaf cache
// ---------------------------------------------------------------------------

/// One stored leaf blob of `bytes` bytes.
fn stored_blob(bytes: usize) -> Arc<[u8]> {
    Arc::from(vec![0u8; bytes].into_boxed_slice())
}

#[test]
fn the_leaf_cache_is_byte_budgeted_and_move_to_front() {
    use crate::archive::leaf::LeafCache;

    let mut cache = LeafCache::new();
    for at in 0..8u64 {
        cache.insert(at, stored_blob(64));
    }
    assert_eq!(cache.len(), 8);
    assert_eq!(cache.bytes(), 8 * 64, "the budget counts STORED bytes");
    // Touching the oldest makes it the newest.
    assert!(cache.stored(0).is_some());
    assert!(cache.stored(99).is_none());

    // One leaf larger than the whole budget evicts everything else and is kept,
    // because refusing it would make the archive unreadable.
    cache.insert(1_000, stored_blob(crate::archive::LEAF_CACHE_BYTES + 16));
    assert_eq!(cache.len(), 1);
    assert!(cache.stored(1_000).is_some());
}

#[test]
fn the_decoded_front_cache_holds_exactly_one_leaf_and_dies_with_its_blob() {
    use crate::archive::leaf::LeafCache;
    use oxigis_render::pmtiles::DirEntry;

    let entry = DirEntry {
        tile_id: 0,
        offset: 0,
        length: 1,
        run_length: 1,
    };
    let mut cache = LeafCache::new();
    cache.insert(10, stored_blob(32));
    cache.set_decoded(10, Arc::new(vec![entry; 4]));
    assert!(cache.decoded(10).is_some());

    // A second decoded leaf replaces the first: one entry, deliberately.
    cache.insert(20, stored_blob(32));
    cache.set_decoded(20, Arc::new(vec![entry; 4]));
    assert!(cache.decoded(10).is_none());
    assert!(cache.decoded(20).is_some());
    // …but the first leaf's STORED blob is still there, so re-reaching it is a
    // decode rather than a range read.
    assert!(cache.stored(10).is_some());

    // Evicting a blob drops any decoded copy of it, so `bytes()` never
    // under-reports what is actually held.
    cache.insert(30, stored_blob(crate::archive::LEAF_CACHE_BYTES + 1));
    assert!(cache.decoded(20).is_none());
    assert_eq!(cache.len(), 1);
}

#[test]
fn an_antagonistic_trace_refetches_no_leaf_twice() {
    use oxigis_render::pmtiles::DirEntry;

    // The pattern the stored-bytes cache exists for: a bookmark-hopping tour
    // that visits far-apart places and then comes back to them. Measured on the
    // planet build, the decoded-entry cache missed 66.3 % of its leaf lookups on
    // this shape and moved 14.7 MiB of refetches; nothing here may refetch at
    // all. It is also the fence against the whole class of "the cache key
    // changed" regressions — a leaf read twice is a bug however it happened.
    let (transport, log) = RecordingTransport::new(many_leaf_archive());
    let archive = ArchiveTileTransport::pmtiles(WHERE, Box::new(transport));
    let handle = archive.clone();
    let config =
        VectorTileConfig::new("https://example.test/{z}/{x}/{y}.pbf").with_paints(Vec::new());
    let provider = VectorTileProvider::new(&config, &egui::Context::default(), Box::new(archive))
        .expect("builds");
    let _ = provider.begin_frame(view(3.0));

    // Two passes over the same scattered addresses, the second in a different
    // order so no amount of "the last leaf happened to still be in front" can
    // carry the test.
    let tour: Vec<TileId> = [
        (0, 0),
        (7, 7),
        (0, 7),
        (7, 0),
        (3, 4),
        (4, 3),
        (1, 6),
        (6, 1),
    ]
    .into_iter()
    .map(|(x, y)| tile(3, x, y))
    .collect();
    for pass in 0..2 {
        let order: Vec<TileId> = if pass == 0 {
            tour.clone()
        } else {
            tour.iter().rev().copied().collect()
        };
        for address in order {
            let _ = provider.mesh(address);
            assert!(
                provider.decoded(address).is_some(),
                "pass {pass}: {}/{}/{} must resolve",
                address.z,
                address.x,
                address.y
            );
        }
    }

    let mut read_at: Vec<u64> = log
        .lock()
        .iter()
        .filter_map(|(_, job)| match job {
            RangeJob::ArchiveLeaf { at, .. } => Some(*at),
            _ => None,
        })
        .collect();
    let total_leaf_reads = read_at.len();
    read_at.sort_unstable();
    read_at.dedup();
    assert!(
        read_at.len() >= 4,
        "the trace must span several leaves to mean anything; it spanned {}",
        read_at.len()
    );
    assert_eq!(
        read_at.len(),
        total_leaf_reads,
        "no leaf directory may be read twice: {total_leaf_reads} reads over {} leaves",
        read_at.len()
    );

    let (held, stored_bytes) = handle.leaf_stats();
    assert_eq!(held, read_at.len(), "every leaf read is still held");
    assert!(stored_bytes > 0);
    assert!(
        stored_bytes < held * 4 * size_of::<DirEntry>(),
        "the cache holds STORED blobs ({stored_bytes} B for {held} leaves), not decoded entries"
    );
    assert_eq!(provider.stats().failed, 0);
}

// ---------------------------------------------------------------------------
// The paints ramp
// ---------------------------------------------------------------------------

#[test]
fn the_ramp_matches_case_insensitively_on_fragments() {
    assert_eq!(kind_for("Earth"), ArchivePaintKind::Earth);
    assert_eq!(kind_for("water_areas"), ArchivePaintKind::Water);
    assert_eq!(kind_for("LANDUSE"), ArchivePaintKind::Earth);
    assert_eq!(kind_for("transportation"), ArchivePaintKind::Road);
    assert_eq!(kind_for("boundaries"), ArchivePaintKind::Boundary);
    assert_eq!(kind_for("places"), ArchivePaintKind::Place);
    assert_eq!(kind_for("zzz_unknown"), ArchivePaintKind::Neutral);
}

#[test]
fn the_ramp_orders_fills_under_lines_under_labels() {
    let names = [
        "places".to_owned(),
        "roads".to_owned(),
        "water".to_owned(),
        "earth".to_owned(),
    ];
    let paints = archive_paints(&names);
    let order: Vec<&str> = paints
        .iter()
        .map(|paint| paint.source_layer.as_str())
        .collect();
    assert_eq!(order, ["earth", "water", "roads", "places", "places"]);
    assert!(matches!(paints[0].style, oxigis_core::LayerStyle::Fill(_)));
    assert!(matches!(paints[2].style, oxigis_core::LayerStyle::Line(_)));
    assert!(matches!(
        paints[4].style,
        oxigis_core::LayerStyle::Symbol(_)
    ));
}

#[test]
fn an_unrecognised_layer_gets_a_neutral_hairline_not_nothing() {
    let paints = archive_paints(&["gemeindegrenzen_2024".to_owned()]);
    assert_eq!(paints.len(), 1);
    match &paints[0].style {
        oxigis_core::LayerStyle::Line(line) => {
            assert!(line.width() < 1.0, "a hairline, not a road");
        }
        other => panic!("expected a neutral line rule, got {other:?}"),
    }
}

#[test]
fn an_archive_declaring_no_layers_gets_no_rules() {
    assert!(archive_paints(&[]).is_empty());
    assert!(archive_paints(&[String::new()]).is_empty());
}

#[test]
fn the_fixtures_own_layer_name_seeds_a_land_fill() {
    let info = probe(sample_pmtiles_vector()).expect("opens").info;
    let paints = archive_paints(&info.layer_names);
    assert_eq!(paints.len(), 1);
    assert_eq!(paints[0].source_layer, "land");
    assert!(matches!(paints[0].style, oxigis_core::LayerStyle::Fill(_)));
}

// ---------------------------------------------------------------------------
// ArchiveInfo
// ---------------------------------------------------------------------------

#[test]
fn archive_info_is_derived_from_the_header_and_the_metadata_together() {
    let bytes = sample_pmtiles_raster();
    let ctx = egui::Context::default();
    let probe = ArchiveProbe::start(
        WHERE,
        ArchiveFormat::PmTiles,
        &ctx,
        Box::new(MemoryRangeTransport::new(bytes)),
    );
    let info: ArchiveInfo = probe.take().expect("answered inline").expect("opens").info;
    assert_eq!(info.content, ArchiveContent::Raster);
    assert!((info.bounds_deg[0] + 180.0).abs() < 1e-6);
    assert_eq!(info.center_zoom, 0);
    assert_eq!(ArchiveContent::Raster.name(), "raster");
    assert_eq!(ArchiveContent::Vector.name(), "vector");
}

// ---------------------------------------------------------------------------
// MBTiles through the SAME two seams
// ---------------------------------------------------------------------------

#[test]
fn an_mbtiles_raster_archive_draws_through_the_same_provider() {
    use crate::mbtiles::MbTilesReader;
    use crate::mbtiles::fixture::{flat_image, raster_metadata};

    // A real 2x2 PNG, so the decode path is exercised and not only the lookup.
    let bytes = flat_image(&[(0, 0, 0, tiny_png())], &raster_metadata());
    let reader = Arc::new(
        MbTilesReader::open(Arc::from(bytes.into_boxed_slice())).expect("the fixture opens"),
    );
    let provider = ArchiveTileProvider::mbtiles("tokyo.mbtiles", reader, &egui::Context::default())
        .expect("the provider must build");
    assert!(
        provider.is_open(),
        "an in-memory archive is open on arrival"
    );

    let decoded = pump(&provider, tile(0, 0, 0), 4).expect("z0 is stored");
    assert_eq!(decoded.width(), 2);
    assert_eq!(provider.stats().ready, 1);
    // An address the archive does not hold is a cached final None, not a
    // failure — the same contract the PMTiles path keeps.
    assert!(pump(&provider, tile(1, 0, 0), 4).is_none());
    assert_eq!(provider.stats().failed, 0);
}

#[test]
fn an_mbtiles_vector_archive_feeds_the_same_vector_provider() {
    use crate::mbtiles::MbTilesReader;
    use crate::mbtiles::fixture::{flat_image, vector_metadata};

    // MBTiles stores rows from the south, so the z1 body at MBTiles row 0 is
    // the XYZ row 1 tile — asserted through the whole provider stack.
    let bytes = flat_image(
        &[(0, 0, 0, mvt_tile("water")), (1, 0, 0, mvt_tile("roads"))],
        &vector_metadata(),
    );
    let reader = Arc::new(
        MbTilesReader::open(Arc::from(bytes.into_boxed_slice())).expect("the fixture opens"),
    );
    let config = VectorTileConfig::new("https://example.test/{z}/{x}/{y}.pbf")
        .with_paints(archive_paints(&["water".to_owned(), "roads".to_owned()]));
    let provider = VectorTileProvider::new(
        &config,
        &egui::Context::default(),
        Box::new(ArchiveTileTransport::mbtiles("tokyo.mbtiles", reader)),
    )
    .expect("builds");

    let _ = provider.begin_frame(view(1.0));
    assert!(
        provider.mesh(tile(0, 0, 0)).is_none(),
        "the first ask reads"
    );
    let decoded = provider.decoded(tile(0, 0, 0)).expect("z0 decoded");
    assert_eq!(
        decoded.layers.first().map(|layer| layer.name.as_str()),
        Some("water")
    );

    assert!(provider.mesh(tile(1, 0, 1)).is_none());
    let flipped = provider
        .decoded(tile(1, 0, 1))
        .expect("MBTiles row 0 is XYZ row 1 at zoom 1");
    assert_eq!(
        flipped.layers.first().map(|layer| layer.name.as_str()),
        Some("roads")
    );

    // And the unflipped address holds nothing — an empty tile, not a failure.
    assert!(provider.mesh(tile(1, 0, 0)).is_none());
    let absent = provider.decoded(tile(1, 0, 0)).expect("an empty tile");
    assert!(absent.layers.is_empty());
    assert_eq!(provider.stats().failed, 0);
}

#[test]
fn an_mbtiles_content_mismatch_is_refused_by_name_too() {
    use crate::mbtiles::MbTilesReader;
    use crate::mbtiles::fixture::{flat_image, vector_metadata};

    let bytes = flat_image(&[(0, 0, 0, mvt_tile("water"))], &vector_metadata());
    let reader = Arc::new(MbTilesReader::open(Arc::from(bytes.into_boxed_slice())).expect("opens"));
    // A vector archive asked to draw as a raster layer.
    let provider = ArchiveTileProvider::mbtiles("tokyo.mbtiles", reader, &egui::Context::default())
        .expect("builds");
    let _ = provider.tile(tile(0, 0, 0));
    let failure = provider.failure().expect("the mismatch is refused");
    assert!(failure.contains("vector"), "{failure}");
    assert!(failure.contains("raster"), "{failure}");
}

// ---------------------------------------------------------------------------
// The PAGED MBTiles seams, composed the way the desktop shell composes them
//
// `ArchiveTileProvider::paged_mbtiles` and `ArchiveTileTransport::paged_mbtiles`
// had no CI-visible coverage at all before tiles v1.5: the only callers were
// `oxigis-desktop`'s `main.rs` and one `#[ignore]`d live test that self-skips
// unless `OXIGIS_LIVE_MBTILES_URL` is set. `mbtiles::paged`'s own tests prove
// the *reader*; these two prove the composition around it, offline, over
// `MemoryRangeTransport` with no filesystem and no polling clock.
// ---------------------------------------------------------------------------

#[test]
fn a_paged_mbtiles_archive_decodes_to_pixels_through_the_provider() {
    use crate::mbtiles::fixture::{PAGE_SIZE, indexed_flat_image, raster_metadata};

    // Indexed, because the paged survey refuses an archive with no address
    // index by name; real PNG bodies, because this provider decodes them.
    let image = indexed_flat_image(
        PAGE_SIZE,
        &[
            (0, 0, 0, tiny_png()),
            (1, 0, 0, tiny_png()),
            (1, 0, 1, tiny_png()),
        ],
        &raster_metadata(),
        false,
    );
    let declared_total = Some(image.len() as u64);
    let provider = ArchiveTileProvider::paged_mbtiles(
        WHERE,
        &egui::Context::default(),
        Box::new(MemoryRangeTransport::new(image)),
        declared_total,
    )
    .expect("the provider must build");
    assert!(
        !provider.is_open(),
        "a paged archive reads nothing until the first frame"
    );

    let decoded = pump(&provider, tile(0, 0, 0), 32).expect("the fixture holds z0");
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 2);
    assert_eq!(
        &decoded.rgba()[..3],
        &[10, 120, 200],
        "the MBTiles fixture's own colour, not the PMTiles one"
    );
    assert!(provider.is_open(), "the survey completed");

    // MBTiles row 0 at zoom 1 is XYZ row 1: the flip survives the paged descent
    // and the provider together, which a single-tile assertion would miss.
    let flipped = pump(&provider, tile(1, 0, 1), 32).expect("MBTiles row 0 is XYZ row 1 at zoom 1");
    assert_eq!(flipped.width(), 2);

    assert_eq!(provider.location(), WHERE);
    assert_eq!(provider.stats().failed, 0);
    assert!(provider.failure().is_none());
}

#[test]
fn a_paged_mbtiles_archive_feeds_the_vector_transport() {
    use crate::mbtiles::fixture::{PAGE_SIZE, indexed_normalized_image, vector_metadata};

    // NORMALIZED, so the two-hop `map` → `images.tile_id` → `images` descent is
    // what the transport actually walks. This is the only CI-visible coverage
    // of that path through a real transport.
    let image = indexed_normalized_image(
        PAGE_SIZE,
        &[(0, 0, 0, "a"), (1, 0, 0, "b")],
        &[("a", mvt_tile("water")), ("b", mvt_tile("roads"))],
        &vector_metadata(),
    );
    let declared_total = Some(image.len() as u64);
    let config = VectorTileConfig::new("https://example.test/{z}/{x}/{y}.pbf")
        .with_paints(archive_paints(&["water".to_owned(), "roads".to_owned()]));
    let provider = VectorTileProvider::new(
        &config,
        &egui::Context::default(),
        Box::new(ArchiveTileTransport::paged_mbtiles(
            WHERE,
            Box::new(MemoryRangeTransport::new(image)),
            declared_total,
        )),
    )
    .expect("the provider must build");

    let _ = provider.begin_frame(view(1.0));
    let root = tile(0, 0, 0);
    let mut decoded = None;
    for _frame in 0..32 {
        if let Some(tile) = provider.decoded(root) {
            decoded = Some(tile);
            break;
        }
        let _ = provider.mesh(root);
    }
    let decoded = decoded.expect("the archive's MVT body reached decode_mvt");
    assert_eq!(
        decoded.layers.first().map(|layer| layer.name.as_str()),
        Some("water")
    );

    // Again the flip, through the vector half of the composition.
    let flipped_address = tile(1, 0, 1);
    let mut flipped = None;
    for _frame in 0..32 {
        if let Some(tile) = provider.decoded(flipped_address) {
            flipped = Some(tile);
            break;
        }
        let _ = provider.mesh(flipped_address);
    }
    let flipped = flipped.expect("MBTiles row 0 is XYZ row 1 at zoom 1");
    assert_eq!(
        flipped.layers.first().map(|layer| layer.name.as_str()),
        Some("roads")
    );
    assert_eq!(provider.stats().failed, 0);
}

// ---------------------------------------------------------------------------
// The offline twins of the live PMTiles round trips (tiles v1.4, stage T1)
//
// Each one is the fixture form of an assertion the `#[ignore]`d live tests in
// `oxigis-desktop`'s `range_http` make against a real archive on a real host.
// They are what keeps the coverage when a third-party host disappears.
// ---------------------------------------------------------------------------

#[test]
fn gzip_directories_with_plain_tile_bodies_read_correctly_both_ways() {
    // The offline twin of `live_pmtiles_raster_webp`: `internal_compression`
    // and `tile_compression` are INDEPENDENT header fields, and a real writer
    // (the pmtiles.io USGS sample) genuinely sets them Gzip/None. Getting the
    // rule wrong fails in both directions — the directories would not decode,
    // or the image would be handed to a gunzip that refuses it — so a decoded
    // tile out of this archive is the whole proof.
    let mut builder =
        PmtilesBuilder::new(TileType::Png).with_compression(Compression::Gzip, Compression::None);
    builder.push_tile(0, 0, 0, raster_png(8, 8));
    let provider = raster_provider(builder.build());
    let decoded = pump(&provider, tile(0, 0, 0), 8).expect("a gzip-directory archive opens");
    assert_eq!(decoded.width(), 8);
    assert_eq!(decoded.height(), 8);
    assert!(provider.failure().is_none());
}

#[test]
fn a_leaf_hop_shows_up_in_leaf_stats_and_a_sibling_costs_no_second_leaf() {
    // The offline twin of `live_pmtiles_planet_leaf_hop`'s leaf assertions.
    // The fixture's leaf threshold is two, so tile ids 0..=1 share one leaf and
    // id 2 starts the next: the first ask pays for a leaf, its neighbour in the
    // same leaf pays for none, and an address in the *next* leaf pays again.
    let transport = ArchiveTileTransport::pmtiles(
        WHERE,
        Box::new(MemoryRangeTransport::new(decodable_leafed_archive())),
    );
    // A clone is a second handle onto the same archive, not a second reader.
    let handle = transport.clone();
    let config =
        VectorTileConfig::new("https://example.test/{z}/{x}/{y}.pbf").with_paints(Vec::new());
    let provider = VectorTileProvider::new(&config, &egui::Context::default(), Box::new(transport))
        .expect("builds");
    assert_eq!(
        handle.leaf_stats(),
        (0, 0),
        "a transport that has not been asked for a tile has read nothing"
    );

    let _ = provider.begin_frame(view(1.0));
    assert!(provider.mesh(tile(0, 0, 0)).is_none());
    assert!(provider.decoded(tile(0, 0, 0)).is_some());
    let (leaves, bytes) = handle.leaf_stats();
    assert_eq!(leaves, 1, "the first lookup hopped through one leaf");
    assert!(bytes > 0, "a held leaf accounts for something");

    assert!(provider.mesh(tile(1, 0, 0)).is_none());
    assert!(provider.decoded(tile(1, 0, 0)).is_some());
    assert_eq!(
        handle.leaf_stats().0,
        1,
        "the neighbour rode the leaf already held"
    );

    assert!(provider.mesh(tile(1, 0, 1)).is_none());
    assert!(provider.decoded(tile(1, 0, 1)).is_some());
    assert_eq!(
        handle.leaf_stats().0,
        2,
        "an address in the next leaf reads that leaf"
    );
}

#[test]
fn a_512_pixel_raster_archive_decodes_at_its_own_size() {
    // The offline twin of `live_pmtiles_raster_webp`'s shape assertion: real
    // raster archives are routinely 512 px per tile (the USGS sample says so in
    // its own metadata), and nothing in this path resamples — the renderer
    // uploads whatever the codec produced.
    let mut builder = PmtilesBuilder::new(TileType::Png)
        .with_metadata(r#"{"name":"512 fixture","tileSize":"512"}"#);
    builder.push_tile(0, 0, 0, raster_png(512, 512));
    let bytes = builder.build();

    let opened = probe(bytes.clone()).expect("the fixture opens");
    assert_eq!(opened.info.tile_size_px, Some(512));
    assert_eq!(opened.info.content, ArchiveContent::Raster);

    let provider = raster_provider(bytes);
    let decoded = pump(&provider, tile(0, 0, 0), 8).expect("the 512 px tile");
    assert_eq!(decoded.width(), 512);
    assert_eq!(decoded.height(), 512);
    let first = decoded.rgba().get(..4).unwrap_or_default();
    assert!(
        decoded.rgba().chunks_exact(4).any(|pixel| pixel != first),
        "the fixture is a gradient, not a flat colour"
    );
}

/// A genuine RGB PNG of `width`×`height`, hand-assembled the same way
/// `oxigis-render`'s raster fixture does.
///
/// The body is a gradient rather than a solid colour, so a "these are real
/// pixels" assertion means something.
fn raster_png(width: u32, height: u32) -> Vec<u8> {
    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    push_chunk(&mut png, b"IHDR", &ihdr);
    let mut raw = Vec::new();
    for row in 0..height {
        raw.push(0);
        for column in 0..width {
            raw.extend_from_slice(&[(row % 251) as u8, (column % 241) as u8, 200]);
        }
    }
    let deflated = oxiarc_deflate::zlib_compress(&raw, 6).expect("the fixture must compress");
    push_chunk(&mut png, b"IDAT", &deflated);
    push_chunk(&mut png, b"IEND", &[]);
    png
}
