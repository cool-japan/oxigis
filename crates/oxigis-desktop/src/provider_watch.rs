//! The refusals only a provider itself learns about, and the handles that make
//! them reachable.
//!
//! Split out of `main.rs` for size (COOLJAPAN: files stay under 2000 lines),
//! the same reason [`crate::cli`] is its own module.
//!
//! Every source in this shell is built *before* its bytes are read — a COG's
//! header, an archive's directory and a tile's first fetch all land on a worker
//! pool — so a successful construction says nothing about whether the layer
//! will ever draw. The refusal lands in the source's own state, and
//! `failure()` (or the failure counters) is the only way to it, which makes
//! polling it the shell's job: without this the user gets a layer in the panel,
//! a basemap-only map, and no message.
//!
//! `map_gpu` takes ownership of what it installs, so every watched source is
//! installed through one of the `Shared*` wrappers here: the provider lives in
//! an [`Arc`], and the map draws through a second reference to it.

use std::sync::Arc;

use oxigis_core::LayerId;
use oxigis_ui::{
    ArchiveTileProvider, ArchiveTileTransport, BoxedTileProvider, BoxedVectorTileSource,
    CogTileProvider, TileProvider, VectorTileProvider, VectorTileSource, XyzTileProvider,
};

/// Failed tiles before an otherwise-silent source is called broken.
///
/// A stray 404 at high zoom is routine and must never reach the status line;
/// this many failures at once is a misconfiguration — a typo'd template, a
/// key-gated host answering 403, or an archive the server has replaced under
/// a live layer. A tile an archive simply does not hold is `Absent`, not a
/// failure, so panning off the covered area never counts here.
const DEAD_SOURCE_FAILURES: usize = 4;

/// A tile provider ready to install, plus the handles the shell keeps so it
/// can report what only the provider itself will learn.
pub struct InstalledProvider {
    /// What `map_gpu` takes ownership of.
    pub provider: BoxedTileProvider,
    /// What the shell polls afterwards — see [`RasterWatch`].
    pub watch: RasterWatch,
}

/// A vector-tile source ready to install, in the same two halves as
/// [`InstalledProvider`].
pub struct InstalledVectorSource {
    /// What `map_gpu` takes ownership of.
    pub source: BoxedVectorTileSource,
    /// What the shell polls afterwards — see [`VectorWatch`].
    pub watch: VectorWatch,
}

/// Handles onto everything currently drawing.
///
/// The basemap and the stack are replaced independently, because their
/// reconciliations are: installing a basemap must not forget which stack
/// entries are drawing over it, and removing one of those must not forget that
/// the basemap under it never answered.
#[derive(Default)]
pub struct ProviderWatch {
    /// The basemap.
    ///
    /// Since the shell migrated to the tile stack this is *only* the basemap:
    /// [`RasterWatch::cog`] and [`RasterWatch::archive`] are now reached
    /// through [`StackWatch::Raster`] instead, one per layer, which is what
    /// lets a COG buried three layers down still report that it never opened.
    raster: RasterWatch,
    /// One watch per entry of the N-layer tile stack, keyed by the project
    /// layer the entry draws.
    ///
    /// The stack replaced the two single slots above for everything but the
    /// basemap, and every reason those two exist applies to each entry: a COG
    /// whose header 404s, an archive the server swapped, a `.pbf` endpoint
    /// answering 403 are all discovered *after* the source was built, and this
    /// is the only handle onto them. Without it, migrating the shell would
    /// silently trade a reported failure for a layer that just never draws.
    ///
    /// A `Vec` rather than a map: it is bounded by
    /// [`oxigis_ui::MAX_DRAWN_TILE_LAYERS`] (eight), so a linear scan is
    /// cheaper than hashing.
    stack: Vec<(LayerId, StackWatch)>,
}

/// The watch one tile-stack entry needs — the raster or vector half, chosen by
/// what the entry actually draws.
pub enum StackWatch {
    /// A raster entry: COG, tile archive or XYZ overlay.
    Raster(RasterWatch),
    /// A streamed vector-tile entry.
    Vector(VectorWatch),
}

impl StackWatch {
    /// This entry's next unreported failure.
    fn poll(&mut self) -> Option<String> {
        match self {
            Self::Raster(watch) => watch.poll(),
            Self::Vector(watch) => watch.poll(),
        }
    }
}

impl ProviderWatch {
    /// Watches the raster stack that has just replaced the previous one.
    pub fn install_raster(&mut self, watch: RasterWatch) {
        self.raster = watch;
    }

    /// Watches the tile-stack entry just installed for `layer`, replacing any
    /// previous watch on the same layer.
    ///
    /// Replace-in-place mirrors `map_gpu::install_tile_layer`, which is also
    /// the "the layer's URL changed" path: the old source is no longer drawing,
    /// so whatever it had left to say is no longer about the map.
    pub fn install_stack(&mut self, layer: LayerId, watch: StackWatch) {
        match self.stack.iter_mut().find(|(id, _)| *id == layer) {
            Some(slot) => slot.1 = watch,
            None => self.stack.push((layer, watch)),
        }
    }

    /// Forgets the stack entries `layers` names — the twin of
    /// `map_gpu::remove_tile_layers`.
    pub fn remove_stack(&mut self, layers: &[LayerId]) {
        self.stack.retain(|(id, _)| !layers.contains(id));
    }

    /// The failure to put on the status line this frame, if there is a new
    /// one.
    ///
    /// At most one message per frame, and each source latches its own report,
    /// so a COG that will never open cannot drown out the first word from the
    /// basemap under it or the vector layer over it. The basemap is asked
    /// first, then the stack in install order: the map underneath everything
    /// staying blank is the more fundamental news.
    pub fn poll(&mut self) -> Option<String> {
        if let Some(message) = self.raster.poll() {
            return Some(message);
        }
        self.stack.iter_mut().find_map(|(_, watch)| watch.poll())
    }
}

/// Handles onto the raster stack: the COG or archive on top, and the XYZ
/// basemap under whichever of them is drawing.
#[derive(Default)]
pub struct RasterWatch {
    /// The COG layer drawing, if one is.
    cog: Option<Arc<CogTileProvider>>,
    /// The tile-archive layer drawing, if one is.
    archive: Option<Arc<ArchiveTileProvider>>,
    /// The basemap underneath, whichever of the three it is under.
    basemap: Option<Arc<XyzTileProvider>>,
    /// Whether the COG's open failure has been reported. It is latched inside
    /// the provider — `Stage::Failed` is terminal — so it is offered on every
    /// frame until someone remembers having taken it.
    cog_reported: bool,
    /// The same, for the archive's open failure.
    archive_reported: bool,
    /// The same, for the archive's *tile* failures, which — unlike its open
    /// failure — are counted rather than latched by the provider.
    archive_tiles_reported: bool,
    /// The same, for the basemap, whose `last_error` keeps changing as tiles
    /// keep failing.
    basemap_reported: bool,
}

impl RasterWatch {
    /// Watches a plain XYZ basemap with nothing over it.
    pub fn basemap(basemap: Arc<XyzTileProvider>) -> Self {
        Self {
            basemap: Some(basemap),
            ..Self::default()
        }
    }

    /// Watches a COG compositing over `basemap` (which is [`None`] when the
    /// basemap itself could not be built).
    pub fn cog(cog: Arc<CogTileProvider>, basemap: Option<Arc<XyzTileProvider>>) -> Self {
        Self {
            cog: Some(cog),
            basemap,
            ..Self::default()
        }
    }

    /// Watches a tile archive compositing over `basemap` — [`Self::cog`]'s
    /// twin, and deliberately identical in shape.
    pub fn archive(
        archive: Arc<ArchiveTileProvider>,
        basemap: Option<Arc<XyzTileProvider>>,
    ) -> Self {
        Self {
            archive: Some(archive),
            basemap,
            ..Self::default()
        }
    }

    /// Takes the basemap handle out, for a caller stacking this provider under
    /// another one: the basemap is the layer the user still sees if the COG
    /// over it never opens, so its own silence has to stay reportable.
    pub fn into_basemap(self) -> Option<Arc<XyzTileProvider>> {
        self.basemap
    }

    /// The raster half's next unreported failure.
    fn poll(&mut self) -> Option<String> {
        if !self.cog_reported
            && let Some(message) = self.cog.as_ref().and_then(|cog| cog.failure())
        {
            self.cog_reported = true;
            return Some(format!("The COG could not be read: {message}"));
        }
        if !self.archive_reported
            && let Some(message) = self.archive.as_ref().and_then(|archive| archive.failure())
        {
            self.archive_reported = true;
            // Every tile of an archive that never opened fails too; that count
            // is the same news, so it is spent here rather than reported again.
            self.archive_tiles_reported = true;
            return Some(format!("The tile archive could not be read: {message}"));
        }
        // An archive that opened and then started refusing tiles — the server
        // replaced it under a live layer, which is what `range_http`'s 412/416
        // handling detects. The provider counts those per tile; only the count
        // is reachable from here, so the advice is the general one.
        if !self.archive_tiles_reported
            && let Some(archive) = self.archive.as_ref()
        {
            let failed = archive.stats().failed;
            if failed >= DEAD_SOURCE_FAILURES {
                self.archive_tiles_reported = true;
                return Some(format!(
                    "{failed} tiles could not be read from the archive. If it changed on the \
                     server, remove and re-add the layer.",
                ));
            }
        }
        if !self.basemap_reported
            && let Some(message) = self
                .basemap
                .as_ref()
                .and_then(|basemap| basemap_failure(basemap))
        {
            self.basemap_reported = true;
            return Some(message);
        }
        None
    }
}

/// Handles onto the vector-tile source drawing over the raster stack.
///
/// Two handles, because a vector layer has two ways to stay silent: the
/// archive behind it never opened, or every tile fetch is refused.
#[derive(Default)]
pub struct VectorWatch {
    /// The source drawing, if one is.
    provider: Option<Arc<VectorTileProvider>>,
    /// The archive it reads tiles out of, when it reads from one rather than
    /// from a URL template. A second handle onto the very transport the
    /// provider owns: [`ArchiveTileTransport`] is clonable precisely so a
    /// caller can keep observing an archive it has handed away by [`Box`].
    archive: Option<ArchiveTileTransport>,
    /// Whether the archive's open failure has been reported; it is latched
    /// inside the transport, so it is offered on every frame.
    archive_reported: bool,
    /// Whether the source's tile failures have been reported.
    tiles_reported: bool,
}

impl VectorWatch {
    /// Watches `provider`, reading from `archive` when it is archive-backed.
    pub fn new(provider: Arc<VectorTileProvider>, archive: Option<ArchiveTileTransport>) -> Self {
        Self {
            provider: Some(provider),
            archive,
            ..Self::default()
        }
    }

    /// The vector half's next unreported failure.
    fn poll(&mut self) -> Option<String> {
        if !self.archive_reported
            && let Some(message) = self.archive.as_ref().and_then(|archive| archive.failure())
        {
            self.archive_reported = true;
            // The refused tiles behind it are the same news — see the raster
            // twin.
            self.tiles_reported = true;
            return Some(format!(
                "The vector tile archive could not be read: {message}",
            ));
        }
        if !self.tiles_reported
            && let Some(message) = self
                .provider
                .as_ref()
                .and_then(|provider| vector_failure(provider))
        {
            self.tiles_reported = true;
            return Some(message);
        }
        None
    }
}

/// The basemap's own refusal, reported only when *nothing* has ever arrived
/// through it — see [`DEAD_SOURCE_FAILURES`].
fn basemap_failure(provider: &XyzTileProvider) -> Option<String> {
    let health = provider.health();
    if provider.stats().ready > 0 || health.total_failures < DEAD_SOURCE_FAILURES as u64 {
        return None;
    }
    let reason = health
        .last_error
        .unwrap_or_else(|| "no reason was recorded".to_owned());
    Some(format!(
        "No basemap tile has loaded ({} failed): {reason}",
        health.total_failures,
    ))
}

/// The vector source's own refusal — [`basemap_failure`]'s twin.
///
/// A decoded tile counts as arrival even when it produced no mesh: an empty
/// tile tessellates to nothing, and a source serving those is working.
fn vector_failure(provider: &VectorTileProvider) -> Option<String> {
    let health = provider.health();
    if provider.stats().ready > 0
        || provider.decoded_len() > 0
        || health.total_failures < DEAD_SOURCE_FAILURES as u64
    {
        return None;
    }
    let reason = health
        .last_error
        .unwrap_or_else(|| "no reason was recorded".to_owned());
    Some(format!(
        "No vector tile has loaded ({} failed): {reason}",
        health.total_failures,
    ))
}

/// A [`CogTileProvider`] the shell also holds a handle to.
pub struct SharedCog(pub Arc<CogTileProvider>);

impl TileProvider for SharedCog {
    fn tile(&self, tile: oxigis_render::TileId) -> Option<oxigis_render::DecodedTile> {
        self.0.tile(tile)
    }
}

/// An [`ArchiveTileProvider`] the shell also holds a handle to — see
/// [`SharedCog`].
pub struct SharedArchive(pub Arc<ArchiveTileProvider>);

impl TileProvider for SharedArchive {
    fn tile(&self, tile: oxigis_render::TileId) -> Option<oxigis_render::DecodedTile> {
        self.0.tile(tile)
    }
}

/// An [`XyzTileProvider`] the shell also holds a handle to — see
/// [`SharedCog`].
pub struct SharedXyz(pub Arc<XyzTileProvider>);

impl TileProvider for SharedXyz {
    fn tile(&self, tile: oxigis_render::TileId) -> Option<oxigis_render::DecodedTile> {
        self.0.tile(tile)
    }
}

/// A [`VectorTileProvider`] the shell also holds a handle to — see
/// [`SharedCog`]. Forwards the whole trait, defaults included: `decoded` is
/// what the label pass reads, and dropping it would silently unlabel the map.
pub struct SharedVector(pub Arc<VectorTileProvider>);

impl VectorTileSource for SharedVector {
    fn begin_frame(&self, view: oxigis_render::MapView) -> bool {
        self.0.begin_frame(view)
    }

    fn mesh(&self, tile: oxigis_render::TileId) -> Option<oxigis_render::VectorMesh> {
        self.0.mesh(tile)
    }

    fn decoded(&self, tile: oxigis_render::TileId) -> Option<Arc<oxigis_render::VectorTile>> {
        self.0.decoded(tile)
    }

    fn label_table(&self) -> &oxigis_render::LabelTable {
        self.0.label_table()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigis_core::{ArchiveFormat, ArchiveRef};
    use oxigis_ui::{
        ArchiveLayerConfig, MemoryRangeTransport, TileError, TileSink, TileTransport,
        VectorTileConfig,
    };

    /// Nothing installed is nothing to report — the poll runs every frame.
    #[test]
    fn an_empty_watch_reports_nothing() {
        assert_eq!(ProviderWatch::default().poll(), None);
    }

    /// The hole finding 168 named, end to end: a provider that BUILDS fine and
    /// then finds it cannot read the archive must reach the status line. The
    /// bytes here are not a PMTiles archive, so the open fails on the worker
    /// side, long after `ArchiveTileProvider::pmtiles` answered `Ok`.
    #[test]
    fn an_archive_that_fails_after_it_was_built_is_reported_once() {
        let transport = MemoryRangeTransport::new(vec![0x7f; 512]);
        let provider = ArchiveTileProvider::pmtiles(
            "memory://not-an-archive".to_owned(),
            &egui::Context::default(),
            Box::new(transport),
        )
        .expect("the provider builds before anything is read");
        let provider = Arc::new(provider);
        let tile = oxigis_render::TileId::new(0, 0, 0).expect("0/0/0");
        // `tile()` is what kicks the open; the memory transport answers inside
        // the call, so a couple of turns settle it.
        for _ in 0..8 {
            if provider.tile(tile).is_some() || provider.failure().is_some() {
                break;
            }
        }
        assert!(
            provider.failure().is_some(),
            "512 bytes of filler are not an archive"
        );
        let mut watch = ProviderWatch::default();
        watch.install_raster(RasterWatch::archive(provider, None));
        let message = watch.poll().expect("the refusal must reach the shell");
        assert!(
            message.starts_with("The tile archive could not be read:"),
            "{message}"
        );
        // Latched in the provider, so it is offered every frame — and taken
        // exactly once.
        assert_eq!(watch.poll(), None);
    }

    /// The vector half of the same hole: an archive-backed vector layer whose
    /// archive never opens is otherwise a layer in the panel, an empty map and
    /// no message.
    #[test]
    fn a_vector_archive_that_never_opens_is_reported_once() {
        let archive = ArchiveLayerConfig::new(
            ArchiveRef::Path {
                path: "memory://not-an-archive".to_owned(),
            },
            ArchiveFormat::PmTiles,
        );
        let transport = ArchiveTileTransport::pmtiles(
            archive.location().to_owned(),
            Box::new(MemoryRangeTransport::new(vec![0x7f; 512])),
        );
        let config = VectorTileConfig {
            archive: Some(archive),
            ..VectorTileConfig::new("memory://not-an-archive")
        };
        let provider = VectorTileProvider::new(
            &config,
            &egui::Context::default(),
            Box::new(transport.clone()),
        )
        .expect("the provider builds before anything is read");
        // A distinct tile per turn: a tile that has just failed is backing
        // off, so asking for the same one again starts no second lookup.
        for x in 0..8 {
            let tile = oxigis_render::TileId::new(4, x, 0).expect("a zoom-4 tile");
            let _mesh = provider.mesh(tile);
        }
        assert!(
            transport.failure().is_some(),
            "512 bytes of filler are not an archive"
        );
        let provider = Arc::new(provider);
        let mut watch = ProviderWatch::default();
        // Watched as a STACK entry, which is the only way this shell installs
        // a vector source since the N-layer migration.
        let layer = LayerId::new();
        watch.install_stack(
            layer,
            StackWatch::Vector(VectorWatch::new(
                Arc::clone(&provider),
                Some(transport.clone()),
            )),
        );
        let message = watch.poll().expect("the refusal must reach the shell");
        assert!(
            message.starts_with("The vector tile archive could not be read:"),
            "{message}"
        );
        // The refused tiles behind the dead archive are the same news, however
        // many of them there are by now.
        assert!(provider.health().total_failures >= DEAD_SOURCE_FAILURES as u64);
        assert_eq!(watch.poll(), None);
        // Removing the layer forgets it: a source that is no longer drawing
        // has nothing left to say.
        watch.remove_stack(&[layer]);
        assert_eq!(watch.poll(), None);
    }

    /// A transport that refuses everything, the way a key-gated host answering
    /// 403 to a typo'd template does.
    struct DeadTransport;

    impl TileTransport for DeadTransport {
        fn request(&self, tile: oxigis_render::TileId, _url: String, sink: TileSink) {
            sink.deliver(tile, Err(TileError::permanent("HTTP 403 Forbidden")));
        }
    }

    /// A vector source over a URL template that refuses every tile: no archive
    /// to latch a failure, so the counters are the only evidence there is.
    #[test]
    fn a_vector_source_that_refuses_every_tile_is_reported_once() {
        let provider = VectorTileProvider::new(
            &VectorTileConfig::new("https://tiles.invalid/{z}/{x}/{y}.pbf"),
            &egui::Context::default(),
            Box::new(DeadTransport),
        )
        .expect("the template is well formed");
        // Distinct tiles: a tile that has just failed is backing off, so
        // asking for the same one again starts no second request.
        for x in 0..=DEAD_SOURCE_FAILURES as u32 {
            let tile = oxigis_render::TileId::new(4, x, 0).expect("a zoom-4 tile");
            let _mesh = provider.mesh(tile);
        }
        assert!(provider.health().total_failures >= DEAD_SOURCE_FAILURES as u64);
        let mut watch = ProviderWatch::default();
        watch.install_stack(
            LayerId::new(),
            StackWatch::Vector(VectorWatch::new(Arc::new(provider), None)),
        );
        let message = watch.poll().expect("a source refusing everything reports");
        assert!(
            message.starts_with("No vector tile has loaded"),
            "{message}"
        );
        assert!(message.contains("403"), "{message}");
        assert_eq!(watch.poll(), None);
    }

    /// A stray 404 at high zoom is routine: under the threshold there is
    /// nothing to say, or every session over a sparse source would open with a
    /// complaint.
    #[test]
    fn a_source_under_the_failure_threshold_is_not_called_broken() {
        let provider = VectorTileProvider::new(
            &VectorTileConfig::new("https://tiles.invalid/{z}/{x}/{y}.pbf"),
            &egui::Context::default(),
            Box::new(DeadTransport),
        )
        .expect("the template is well formed");
        // Under the threshold: routine misses are not a misconfiguration.
        let tile = oxigis_render::TileId::new(4, 0, 0).expect("a zoom-4 tile");
        let _mesh = provider.mesh(tile);
        assert_eq!(vector_failure(&provider), None);
    }

    /// The basemap and the stack are replaced independently: reconciling the
    /// basemap must not forget which stack entries are drawing over it.
    #[test]
    fn installing_a_basemap_keeps_the_stack_watches() {
        let provider = VectorTileProvider::new(
            &VectorTileConfig::new("https://tiles.invalid/{z}/{x}/{y}.pbf"),
            &egui::Context::default(),
            Box::new(DeadTransport),
        )
        .expect("the template is well formed");
        for x in 0..=DEAD_SOURCE_FAILURES as u32 {
            let tile = oxigis_render::TileId::new(4, x, 0).expect("a zoom-4 tile");
            let _mesh = provider.mesh(tile);
        }
        let mut watch = ProviderWatch::default();
        watch.install_stack(
            LayerId::new(),
            StackWatch::Vector(VectorWatch::new(Arc::new(provider), None)),
        );
        // A basemap install lands between the failure and the poll.
        watch.install_raster(RasterWatch::default());
        let message = watch.poll().expect("the vector refusal survives");
        assert!(
            message.starts_with("No vector tile has loaded"),
            "{message}"
        );
    }
}
