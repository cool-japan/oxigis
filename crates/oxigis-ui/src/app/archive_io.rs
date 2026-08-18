// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The tile-archive add seam: probe first, create the layer second.
//!
//! Every other add gesture in OxiGIS knows what it is adding. This one does
//! not: a `.pmtiles` may hold PNG tiles or MVT ones, and those become two
//! different [`oxigis_core::LayerKind`] variants drawn by two different
//! providers. So the gesture and the creation are separated by one round trip.
//!
//! ```text
//! + Archive          -> LayerAction::AddArchiveUrlLayer(url)
//! dispatch           -> pending_archive_request = Some(request)     (take-once)
//! shell, next frame  -> take_pending_archive_probe() -> builds a transport
//!                       -> ArchiveProbe::start(...) -> attach_archive_probe()
//! shell, each frame  -> poll_archive_probe()
//! header lands       -> add_archive_layer(): the RIGHT variant, paints from
//!                       the ramp, credit from the metadata, record_layer_add
//! ```
//!
//! `oxigis-ui` builds no transport of its own — it compiles to `wasm32` and
//! owns no HTTP stack, no filesystem and no downloads — which is why the middle
//! step belongs to the shell, exactly as it does for tiles, COGs, printing and
//! dropped paths.
//!
//! # Why the layer is created only at the end
//!
//! A "pending archive" layer would be a layer whose kind is a lie until the
//! network answers, and `Ctrl+S` does not wait for the network. Creating at the
//! end also makes the undo entry exact: `record_layer_add` fires once, on the
//! layer that actually exists, so one Ctrl+Z removes exactly what one gesture
//! added.

use std::sync::Arc;

use oxigis_core::{ArchiveFormat, ArchiveRef, LayerId, archive_refusal};

use crate::archive::{ArchiveInfo, ArchiveProbe, MemoryRangeTransport};
use crate::layer_panel;
use crate::local_input::DroppedItem;

use super::OxigisApp;

/// Total bytes of dropped archives held for the session at once.
///
/// A browser drop arrives as **bytes**, not a path: there is nothing to re-read
/// later, so the bytes have to be kept for as long as the layer is drawn. That
/// is real memory in a tab, hence a budget; the oldest archive is dropped when
/// a new one would exceed it, and its layer then reports that it needs
/// re-dropping rather than silently drawing nothing.
///
/// **Target-aware, because the store's only real consumer is 32-bit.** A
/// native shell takes the path route and streams a dropped archive from disk
/// (`load_dropped_archive`'s second arm); a browser drop is what actually
/// fills this. A `wasm32` module has one 32-bit linear address space, and
/// browsers refuse `memory.grow` well before its 4 GiB ceiling — so 512 MiB of
/// archive bytes, beside the tile caches, the meshes, the label atlas and a
/// print raster, walks a routine session into an allocation failure, which in
/// wasm aborts the module and takes the canvas with it.
#[cfg(target_pointer_width = "32")]
pub const MAX_SESSION_ARCHIVE_BYTES: usize = 128 * 1024 * 1024;

/// Total bytes of dropped archives held for the session at once.
///
/// See the 32-bit definition for why the value differs by target.
#[cfg(not(target_pointer_width = "32"))]
pub const MAX_SESSION_ARCHIVE_BYTES: usize = 512 * 1024 * 1024;

/// What a refused archive is told to do instead.
///
/// Split by target for the same reason as the budget: a native shell can open
/// the file by path and stream it, and a browser — the only place a 32-bit
/// build of this runs — has no filesystem to stream from, so telling its user
/// to "open it from disk" is advice they cannot follow.
#[cfg(target_pointer_width = "32")]
const OVER_BUDGET_ADVICE: &str = "this build has one 32-bit address space for the whole map, so a large archive has to be \
     served as tiles rather than dropped whole.";

/// What a refused archive is told to do instead.
///
/// See the 32-bit definition for why the advice differs by target.
#[cfg(not(target_pointer_width = "32"))]
const OVER_BUDGET_ADVICE: &str = "open it from disk instead, which streams rather than loading \
                                  the file whole.";

/// An archive a shell still has to read the header of.
///
/// Take-once, like every other shell hand-off in this crate: the shell builds
/// the range transport its platform provides (HTTP for a URL, a file reader for
/// a path) and hands the resulting [`ArchiveProbe`] back with
/// [`OxigisApp::attach_archive_probe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveProbeRequest {
    /// Where the archive is.
    pub archive: ArchiveRef,
    /// Which container format it is.
    pub format: ArchiveFormat,
}

impl ArchiveProbeRequest {
    /// The string the shell's transport addresses — a URL or a path.
    #[must_use]
    pub fn location(&self) -> &str {
        self.archive.location()
    }
}

impl OxigisApp {
    /// The archive whose header a shell still has to read, if one is waiting.
    ///
    /// Take-once. Answer it by building a range transport for
    /// [`ArchiveProbeRequest::location`] and handing back
    /// [`OxigisApp::attach_archive_probe`]; a shell that cannot (a browser
    /// handed a path, say) should call [`OxigisApp::set_status`] instead, and
    /// the gesture simply ends there.
    pub fn take_pending_archive_probe(&mut self) -> Option<ArchiveProbeRequest> {
        self.pending_archive_request.take()
    }

    /// Hands back the probe a shell started for
    /// [`OxigisApp::take_pending_archive_probe`].
    ///
    /// Replaces any probe already in flight: the user asked for a different
    /// archive, and the old answer is no longer the one they want.
    pub fn attach_archive_probe(&mut self, probe: ArchiveProbe) {
        self.archive_probe = Some(probe);
    }

    /// Whether an archive is currently being identified.
    #[must_use]
    pub fn archive_probe_running(&self) -> bool {
        self.archive_probe.is_some()
    }

    /// Creates the layer as soon as the probe answers; a no-op otherwise.
    ///
    /// Call once per frame from the shell, the same way the desktop shell polls
    /// its background font scan. Returns the new layer's id on the frame it is
    /// created, so a shell that wants to react (a log line, a zoom) can.
    pub fn poll_archive_probe(&mut self) -> Option<LayerId> {
        let answer = self.archive_probe.as_ref()?.take()?;
        let probe = self.archive_probe.take()?;
        let location = probe.location().to_owned();
        match answer {
            Ok(opened) => {
                let Some(request) = self.probing.take() else {
                    // Nothing recorded what this probe was for; refuse to guess
                    // a reference rather than record a wrong one.
                    self.status = Some(format!(
                        "Read {location}, but the request it belonged to was gone; try again."
                    ));
                    return None;
                };
                Some(self.add_archive_layer(&request.archive, request.format, &opened.info))
            }
            Err(error) => {
                self.probing = None;
                self.status = Some(format!("Could not read {location}: {error}"));
                None
            }
        }
    }

    /// Appends the layer `info` implies, records it for undo, and selects it.
    ///
    /// The ONE creation point: the probe path and the shell's local-archive
    /// path both go through it, so a `.pmtiles` opened over HTTP and one
    /// dropped on the map produce exactly the same layer.
    pub fn add_archive_layer(
        &mut self,
        archive: &ArchiveRef,
        format: ArchiveFormat,
        info: &ArchiveInfo,
    ) -> LayerId {
        let id = layer_panel::add_archive_layer(&mut self.project.layers, archive, format, info);
        self.selection = Some(id);
        let name = self.project.layers.get(id).map_or_else(
            || archive.file_name().to_owned(),
            |layer| layer.name.clone(),
        );
        self.status = Some(format!(
            "Added \u{201c}{name}\u{201d} \u{2014} {}.",
            info.summary()
        ));
        self.record_layer_add(&[id]);
        id
    }

    /// Queues an archive for identification, or refuses the combination by
    /// name. Backs [`crate::layer_panel::LayerAction::AddArchiveUrlLayer`] and
    /// the drop path.
    ///
    /// Returns whether the request was accepted.
    pub fn request_archive_probe(&mut self, archive: ArchiveRef, format: ArchiveFormat) -> bool {
        if let Some(reason) = archive_refusal(&archive, format) {
            self.status = Some(reason);
            return false;
        }
        // Honest about the trade-off rather than silent about it: a remote
        // `.mbtiles` genuinely works now, and is genuinely slower to first tile
        // than the format designed for range reads. Measured: a paged SQLite
        // lookup is depth x RTT (roughly 1.5-3 s cold), a PMTiles lookup is two
        // round trips.
        let request = ArchiveProbeRequest { archive, format };
        if matches!(request.archive, ArchiveRef::Url { .. })
            && request.format == ArchiveFormat::MbTiles
        {
            self.status = Some(format!(
                "Reading {}\u{2026} A remote MBTiles archive is read a page at a time; \
                 PMTiles is the faster format for a remote archive.",
                request.location()
            ));
            self.probing = Some(request.clone());
            self.pending_archive_request = Some(request);
            self.archive_probe = None;
            return true;
        }
        self.status = Some(format!("Reading {}\u{2026}", request.location()));
        self.probing = Some(request.clone());
        self.pending_archive_request = Some(request);
        // A probe already in flight is for a different archive now.
        self.archive_probe = None;
        true
    }

    /// Records that the user asked for a file dialog. Backs
    /// [`crate::layer_panel::LayerAction::OpenArchiveFile`].
    pub fn request_archive_pick(&mut self) {
        self.pending_archive_pick = true;
    }

    /// Whether a shell still owes the user a file dialog — take-once.
    ///
    /// A native shell answers by opening one and, on a chosen file, calling
    /// [`OxigisApp::request_archive_probe`] — for **both** formats, since a
    /// `.mbtiles` is paged over ranges exactly as a `.pmtiles` is, so both are
    /// answered by handing back a transport. A browser has no dialog to open
    /// and answers with a status line pointing at the drop gesture, which is
    /// the honest per-platform capability rather than a `#[cfg]` in the shared
    /// panel.
    pub fn take_pending_archive_pick(&mut self) -> bool {
        core::mem::take(&mut self.pending_archive_pick)
    }

    /// The tile-archive URL currently typed into the layer panel.
    #[must_use]
    pub fn archive_url_input(&self) -> &str {
        &self.archive_url_input
    }

    /// The bytes of a dropped archive this session is holding, if it holds
    /// one for `location`.
    ///
    /// A shell consults this **before** building a platform transport: bytes in
    /// hand mean [`MemoryRangeTransport`], and are the browser's only way to
    /// read a local archive at all. Native shells normally get [`None`] here
    /// and open the path directly.
    #[must_use]
    pub fn archive_bytes(&self, location: &str) -> Option<Arc<[u8]>> {
        self.session_archives
            .iter()
            .find(|(key, _)| key == location)
            .map(|(_, bytes)| Arc::clone(bytes))
    }

    /// Holds a dropped archive's bytes for the session, under `location`.
    ///
    /// Evicts the oldest entries until the total fits
    /// [`MAX_SESSION_ARCHIVE_BYTES`]. An archive larger than the whole budget
    /// is refused rather than stored, because storing it would evict everything
    /// else and still not fit.
    ///
    /// Returns whether the bytes were kept.
    pub fn remember_archive_bytes(&mut self, location: &str, bytes: Arc<[u8]>) -> bool {
        if bytes.len() > MAX_SESSION_ARCHIVE_BYTES {
            self.status = Some(format!(
                "{location} is {} MiB, past the {} MiB an in-memory archive may use; \
                 {OVER_BUDGET_ADVICE}",
                bytes.len() / (1024 * 1024),
                MAX_SESSION_ARCHIVE_BYTES / (1024 * 1024),
            ));
            return false;
        }
        self.session_archives.retain(|(key, _)| key != location);
        self.session_archives.push((location.to_owned(), bytes));
        while self
            .session_archives
            .iter()
            .map(|(_, bytes)| bytes.len())
            .sum::<usize>()
            > MAX_SESSION_ARCHIVE_BYTES
            && self.session_archives.len() > 1
        {
            self.session_archives.remove(0);
        }
        true
    }

    /// Handles one dropped `.pmtiles`/`.mbtiles`.
    ///
    /// Two routes, decided by what the host attached:
    ///
    /// * **bytes** (a browser drop, and a native shell that already read the
    ///   file) — kept for the session and probed right here, because
    ///   [`MemoryRangeTransport`] needs no platform capability at all;
    /// * **a path** (an `egui-winit` native drop) — handed to the shell as an
    ///   [`ArchiveProbeRequest`], which builds a file range transport so the
    ///   archive *streams* instead of being read whole.
    pub(super) fn load_dropped_archive(
        &mut self,
        format: ArchiveFormat,
        item: &DroppedItem,
        ctx: &egui::Context,
    ) {
        match (item.bytes.as_ref(), item.path.as_ref()) {
            (Some(bytes), _) => {
                self.open_archive_bytes(format, &item.name, Arc::clone(bytes), ctx);
            }
            (None, Some(path)) if self.local.paths_supported() => {
                let _accepted = self.request_archive_probe(
                    ArchiveRef::Path {
                        path: path.display().to_string(),
                    },
                    format,
                );
            }
            (None, Some(_)) => {
                self.status = Some(format!(
                    "{} arrived as a path, which this build cannot open; drop the file itself.",
                    item.name
                ));
            }
            (None, None) => {
                self.status = Some(format!("{} arrived without any data.", item.name));
            }
        }
    }

    /// Opens an archive whose bytes are already in hand, under the session key
    /// `name`.
    ///
    /// The browser's whole local-archive story, and the seam a native shell
    /// uses when it has read a file itself. `name` becomes the
    /// [`ArchiveRef::Path`] the project records, so re-opening the project on a
    /// machine that has the file finds it by name.
    pub fn open_archive_bytes(
        &mut self,
        format: ArchiveFormat,
        name: &str,
        bytes: Arc<[u8]>,
        ctx: &egui::Context,
    ) {
        if !self.remember_archive_bytes(name, Arc::clone(&bytes)) {
            return;
        }
        let archive = ArchiveRef::Path {
            path: name.to_owned(),
        };
        match format {
            ArchiveFormat::PmTiles => {
                if !self.request_archive_probe(archive, format) {
                    return;
                }
                // Answering the request here rather than through the shell: the
                // bytes are already in memory, so no platform capability is
                // involved and the round trip would be theatre.
                let _consumed = self.take_pending_archive_probe();
                self.attach_archive_probe(ArchiveProbe::start(
                    name,
                    format,
                    ctx,
                    Box::new(MemoryRangeTransport::from_shared(bytes)),
                ));
            }
            ArchiveFormat::MbTiles => self.open_mbtiles_bytes(archive, &bytes),
        }
    }

    /// Opens an MBTiles image already in memory.
    ///
    /// Split out from [`OxigisApp::open_archive_bytes`] because MBTiles is
    /// answered *synchronously* — a SQLite image is walked in place, there is
    /// no header round trip to wait for — so it produces an
    /// [`ArchiveInfo`] directly rather than through a probe.
    fn open_mbtiles_bytes(&mut self, archive: ArchiveRef, bytes: &Arc<[u8]>) {
        match crate::mbtiles::MbTilesReader::open(Arc::clone(bytes)) {
            Ok(reader) => {
                let info = reader.info();
                // Held for the session: the index cost one pass over the whole
                // b-tree, and rebuilding it on every reconciliation would pay
                // that again for nothing.
                self.remember_mbtiles_reader(archive.location(), Arc::new(reader));
                self.add_archive_layer(&archive, ArchiveFormat::MbTiles, &info);
            }
            Err(reason) => {
                self.status = Some(format!(
                    "{} could not be read: {reason}",
                    archive.file_name()
                ));
            }
        }
    }

    /// The opened MBTiles archive this session holds for `location`, if it
    /// holds one.
    ///
    /// A shell consults this before building anything: an MBTiles archive is
    /// indexed at open, and the index — not the image — is what makes a lookup
    /// cheap, so it is built once and shared.
    #[must_use]
    pub fn mbtiles_reader(&self, location: &str) -> Option<Arc<crate::mbtiles::MbTilesReader>> {
        self.session_readers
            .iter()
            .find(|(key, _)| key == location)
            .map(|(_, reader)| Arc::clone(reader))
    }

    /// Holds an opened MBTiles archive for the session, under `location`.
    ///
    /// Bounded by the same count the byte store is: an archive whose bytes have
    /// been evicted keeps no reader either, so the two cannot disagree about
    /// what is still readable.
    pub fn remember_mbtiles_reader(
        &mut self,
        location: &str,
        reader: Arc<crate::mbtiles::MbTilesReader>,
    ) {
        self.session_readers.retain(|(key, _)| key != location);
        self.session_readers.push((location.to_owned(), reader));
        let live: Vec<String> = self
            .session_archives
            .iter()
            .map(|(key, _)| key.clone())
            .collect();
        self.session_readers
            .retain(|(key, _)| live.iter().any(|held| held == key));
    }
}

/// The format a submitted archive URL announces, defaulting to PMTiles.
///
/// Both formats are read over HTTP `Range` requests now — see
/// the paged MBTiles reader — so this is a guess about what an *extensionless*
/// URL most likely is, not a gate. PMTiles remains the right default: it is the
/// format designed for range reads, it opens in two round trips where a paged
/// SQLite lookup costs depth × RTT, and an archive served without an extension
/// is far more often one of the many PMTiles tile services than a `.mbtiles`
/// someone put behind a static host.
#[must_use]
pub fn format_for_url(url: &str) -> ArchiveFormat {
    ArchiveFormat::from_file_name(url).unwrap_or(ArchiveFormat::PmTiles)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use oxigis_core::ArchiveFormat;

    use super::{MAX_SESSION_ARCHIVE_BYTES, OxigisApp};

    /// A small, real MBTiles image — the same fixture the reader's own tests
    /// use, so `open` here is the production path and not a stub.
    fn image() -> Arc<[u8]> {
        Arc::from(crate::mbtiles::sample_mbtiles_raster().into_boxed_slice())
    }

    /// An opened reader over [`image`].
    fn reader() -> Arc<crate::mbtiles::MbTilesReader> {
        Arc::new(crate::mbtiles::MbTilesReader::open(image()).expect("the fixture opens"))
    }

    #[test]
    fn the_session_budget_is_target_aware() {
        // The store's only real consumer is the 32-bit browser build: a
        // native shell streams a dropped archive from its path instead. One
        // value for both targets let a routine tab hold half a gigabyte of
        // archive bytes beside its tile caches, meshes, label atlas and print
        // raster, in a linear heap browsers refuse to grow anywhere near its
        // 4 GiB ceiling.
        #[cfg(target_pointer_width = "32")]
        assert_eq!(MAX_SESSION_ARCHIVE_BYTES, 128 * 1024 * 1024);
        #[cfg(not(target_pointer_width = "32"))]
        assert_eq!(MAX_SESSION_ARCHIVE_BYTES, 512 * 1024 * 1024);
        // A drop is double-buffered on its way in (the host's buffer, then the
        // `Arc<[u8]>`), so the budget has to leave room for a second copy of
        // itself in the same address space.
        const {
            assert!(MAX_SESSION_ARCHIVE_BYTES < usize::MAX / 2);
        }
    }

    #[test]
    fn re_dropping_the_same_archive_replaces_its_bytes_rather_than_doubling_them() {
        // The budget is a sum over the store, so a second drop of one file
        // that appended instead of replacing would count it twice and start
        // evicting the archives the user still has open.
        let mut app = OxigisApp::new();
        let bytes = image();
        assert!(app.remember_archive_bytes("tokyo.mbtiles", Arc::clone(&bytes)));
        assert!(app.remember_archive_bytes("tokyo.mbtiles", Arc::clone(&bytes)));
        assert_eq!(
            app.archive_bytes("tokyo.mbtiles").map(|held| held.len()),
            Some(bytes.len()),
        );
        assert_eq!(app.session_archives.len(), 1);
    }

    #[test]
    fn a_reader_whose_bytes_the_session_does_not_hold_is_not_kept() {
        // The coherence rule the two stores are tied by: an index is only
        // useful while the image it indexes is still held, and a reader that
        // outlived its bytes would make `mbtiles_reader` answer for an
        // archive `archive_bytes` has already forgotten.
        let mut app = OxigisApp::new();
        app.remember_mbtiles_reader("never-dropped.mbtiles", reader());
        assert!(
            app.mbtiles_reader("never-dropped.mbtiles").is_none(),
            "a reader with no bytes behind it must not be kept",
        );
    }

    #[test]
    fn a_held_archives_reader_survives_another_archives_refusal() {
        let mut app = OxigisApp::new();
        assert!(app.remember_archive_bytes("held.mbtiles", image()));
        app.remember_mbtiles_reader("held.mbtiles", reader());
        assert!(app.mbtiles_reader("held.mbtiles").is_some());
        // Sweeping the store for the second, byte-less archive must not take
        // the first archive's reader with it.
        app.remember_mbtiles_reader("stray.mbtiles", reader());
        assert!(app.mbtiles_reader("stray.mbtiles").is_none());
        assert!(
            app.mbtiles_reader("held.mbtiles").is_some(),
            "the sweep must be keyed by the bytes actually held",
        );
    }

    #[test]
    fn a_dropped_mbtiles_image_keeps_the_index_a_shell_rebuilds_a_provider_from() {
        // The browser's whole local-archive story, end to end: a drop leaves
        // bytes AND an index behind, and a PDF export resolves both off the
        // frame that queues it.
        let mut app = OxigisApp::new();
        app.open_archive_bytes(
            ArchiveFormat::MbTiles,
            "dropped.mbtiles",
            image(),
            &egui::Context::default(),
        );
        assert!(app.archive_bytes("dropped.mbtiles").is_some());
        assert!(
            app.mbtiles_reader("dropped.mbtiles").is_some(),
            "an MBTiles drop is indexed once and shared for the session",
        );
    }
}
