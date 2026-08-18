// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The tile-archive add flow, end to end and offline.
//!
//! Every test here drives the real seams — the layer action, the shell
//! hand-off, the probe over [`MemoryRangeTransport`], the layer creation, the
//! reconciliation and the undo — against the hand-built PMTiles fixtures
//! `oxigis-render` ships. Nothing is stubbed and nothing touches the network.

use std::sync::Arc;

use oxigis_core::{ArchiveFormat, ArchiveRef, LayerKind, RasterSource, VectorSource};
use oxigis_render::pmtiles::{sample_pmtiles_raster, sample_pmtiles_vector};

use crate::archive::{ArchiveProbe, MemoryRangeTransport};
use crate::layer_panel::LayerAction;

use super::*;

/// Confirms whatever raster/vector work is outstanding, as a shell with a live
/// render state would.
fn settle_everything(app: &mut OxigisApp) {
    if let Some(work) = app.pending_raster_work() {
        app.settle_raster_work(work, Ok(()));
    }
    if let Some(work) = app.pending_vector_work() {
        app.settle_vector_work(work, Ok(()));
    }
}

/// Plays the shell's half of the probe hand-off, over the given archive bytes.
///
/// Exactly what `oxigis-desktop` and `oxigis-web` do each frame, minus the
/// platform transport: take the request, build a transport, hand the probe
/// back, poll it.
fn run_probe(app: &mut OxigisApp, bytes: Vec<u8>) -> Option<oxigis_core::LayerId> {
    let request = app.take_pending_archive_probe()?;
    app.attach_archive_probe(ArchiveProbe::start(
        request.location().to_owned(),
        request.format,
        &egui::Context::default(),
        Box::new(MemoryRangeTransport::new(bytes)),
    ));
    app.poll_archive_probe()
}

#[test]
fn the_add_gesture_only_asks_for_a_probe_and_creates_no_layer_yet() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddArchiveUrlLayer(
        "https://example.test/basemap.pmtiles".to_string(),
    ));
    assert_eq!(
        app.project().layers.len(),
        0,
        "nothing half-decided may enter the layer stack"
    );
    assert!(app.archive_probe_running() || app.take_pending_archive_probe().is_some());
}

#[test]
fn a_raster_archive_becomes_a_raster_layer_when_its_header_lands() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    app.apply_layer_action(LayerAction::AddArchiveUrlLayer(
        "https://example.test/basemap.pmtiles".to_string(),
    ));
    let id = run_probe(&mut app, sample_pmtiles_raster()).expect("the layer must be created");

    let layer = app.project().layers.get(id).expect("just added");
    assert_eq!(layer.name, "fixture", "the archive's own metadata names it");
    match &layer.kind {
        LayerKind::Raster(RasterSource::TileArchive {
            archive,
            format,
            attribution,
        }) => {
            assert_eq!(
                archive,
                &ArchiveRef::Url {
                    url: "https://example.test/basemap.pmtiles".to_string()
                }
            );
            assert_eq!(*format, ArchiveFormat::PmTiles);
            assert_eq!(attribution, "OxiGIS test fixture");
        }
        other => panic!("expected a raster tile archive, got {other:?}"),
    }
    assert_eq!(app.selection(), Some(id));
    assert_eq!(crate::layer_panel::kind_tag(layer), "pmtiles");

    // The reconciliation now offers an archive provider, not a COG one.
    let work = app
        .pending_raster_work()
        .expect("an install must be offered");
    let archive = work.archive.expect("the archive alternative");
    assert!(work.cog.is_none());
    assert_eq!(archive.location(), "https://example.test/basemap.pmtiles");
    assert_eq!(archive.attribution, "OxiGIS test fixture");
    // …and its credit reaches the map's one derived line.
    assert!(app.credit_line().contains("OxiGIS test fixture"));
}

#[test]
fn a_vector_archive_becomes_a_vector_layer_with_paints_from_the_ramp() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    app.apply_layer_action(LayerAction::AddArchiveUrlLayer(
        "https://example.test/tiles.pmtiles".to_string(),
    ));
    let id = run_probe(&mut app, sample_pmtiles_vector()).expect("the layer must be created");

    let layer = app.project().layers.get(id).expect("just added");
    match &layer.kind {
        LayerKind::Vector(VectorSource::TileArchive { paints, .. }) => {
            // The fixture declares one `vector_layers` entry named "land",
            // which the ramp matches to a fill.
            assert_eq!(paints.len(), 1);
            assert_eq!(paints[0].source_layer, "land");
            assert!(matches!(paints[0].style, oxigis_core::LayerStyle::Fill(_)));
        }
        other => panic!("expected a vector tile archive, got {other:?}"),
    }

    let work = app
        .pending_vector_work()
        .expect("an install must be offered");
    let VectorWork::Install(config) = work else {
        panic!("a vector archive layer must offer an install");
    };
    let archive = config.archive.as_ref().expect("an archive-backed config");
    assert_eq!(archive.location(), "https://example.test/tiles.pmtiles");
    assert_eq!(
        config.template().expect("no template error"),
        None,
        "an archive-backed config has no URL template to expand"
    );
    assert_eq!(config.attribution, "OxiGIS test fixture");
    // The raster slot is untouched: a vector archive composites, it does not
    // replace the basemap.
    assert!(
        app.pending_raster_work()
            .is_none_or(|work| { work.archive.is_none() && work.cog.is_none() })
    );
}

#[test]
fn removing_an_archive_layer_detaches_it_and_undo_puts_it_back() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    app.apply_layer_action(LayerAction::AddArchiveUrlLayer(
        "https://example.test/tiles.pmtiles".to_string(),
    ));
    let id = run_probe(&mut app, sample_pmtiles_vector()).expect("created");
    settle_everything(&mut app);
    assert!(app.pending_vector_work().is_none());

    app.apply_layer_action(LayerAction::Remove(id));
    assert_eq!(
        app.pending_vector_work(),
        Some(VectorWork::Detach),
        "removing the layer must detach what the map is drawing"
    );
    settle_everything(&mut app);

    // `record_layer_add` fired at creation, so one Ctrl+Z restores exactly the
    // layer the one gesture produced — variant, paints and credit included.
    app.undo_once();
    let restored = app
        .project()
        .layers
        .get(id)
        .expect("undo restores the archive layer");
    assert!(matches!(
        restored.kind,
        LayerKind::Vector(VectorSource::TileArchive { .. })
    ));
    assert!(matches!(
        app.pending_vector_work(),
        Some(VectorWork::Install(_))
    ));
}

#[test]
fn undoing_the_add_removes_exactly_what_the_gesture_added() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    app.apply_layer_action(LayerAction::AddArchiveUrlLayer(
        "https://example.test/basemap.pmtiles".to_string(),
    ));
    let id = run_probe(&mut app, sample_pmtiles_raster()).expect("created");
    settle_everything(&mut app);

    app.undo_once();
    assert!(app.project().layers.get(id).is_none());
    let work = app.pending_raster_work().expect("the map must go back");
    assert!(work.archive.is_none());
}

#[test]
fn the_top_most_visible_raster_layer_wins_across_kinds() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/scene.tif".to_string(),
    ));
    app.apply_layer_action(LayerAction::AddArchiveUrlLayer(
        "https://example.test/basemap.pmtiles".to_string(),
    ));
    let archive_id = run_probe(&mut app, sample_pmtiles_raster()).expect("created");

    // The archive is on top, so it is what the map draws.
    let work = app.desired_raster();
    assert!(work.cog.is_none());
    assert!(work.archive.is_some());

    // Hide it and the COG underneath takes over — one rule, both kinds.
    app.apply_layer_action(LayerAction::ToggleVisibility(archive_id));
    let work = app.desired_raster();
    assert!(work.archive.is_none());
    assert_eq!(
        work.cog.map(|cog| cog.url),
        Some("https://example.test/scene.tif".to_string())
    );
}

#[test]
fn an_mbtiles_url_is_probed_like_any_other_and_says_pmtiles_is_faster() {
    // Tiles v1.4 flipped this: a `.mbtiles` URL is READ, a page at a time. What
    // the add seam owes the user is now honesty about the trade-off, not a
    // refusal — the refusals that matter are the archive's own bytes' and land
    // at survey time (`mbtiles::paged::tests` pins each by name).
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddArchiveUrlLayer(
        "https://example.test/tokyo.mbtiles".to_string(),
    ));
    let request = app
        .take_pending_archive_probe()
        .expect("a remote MBTiles archive is handed to a shell like any other");
    assert_eq!(request.format, oxigis_core::ArchiveFormat::MbTiles);
    assert_eq!(request.location(), "https://example.test/tokyo.mbtiles");
    assert_eq!(app.project().layers.len(), 0, "probe THEN create, as ever");
    let status = app.status().expect("the trade-off is stated");
    assert!(status.contains("page at a time"), "{status}");
    assert!(status.contains("PMTiles"), "{status}");
}

#[test]
fn an_mbtiles_over_http_layer_in_a_project_file_is_drawn_now() {
    // The load-time twin of the flip: a saved project holding a remote MBTiles
    // layer used to be listed but never offered to a shell. It is offered now,
    // and whether it can actually be read is answered by its own bytes at survey
    // time rather than guessed at from the reference.
    let mut app = OxigisApp::new();
    let json = r#"{
        "format_version": 1,
        "name": "hand edited",
        "view": {"center_lon": 0.0, "center_lat": 0.0, "zoom": 2.0},
        "layers": [{
            "id": 7001,
            "name": "remote mbtiles",
            "visible": true,
            "opacity": 1.0,
            "kind": {"kind":"raster","source":{
                "type":"tile_archive",
                "archive":{"at":"url","url":"https://example.test/tokyo.mbtiles"},
                "format":"mb_tiles"
            }}
        }],
        "styles": {}
    }"#;
    let project: oxigis_core::Project =
        oxigis_core::Project::from_json_string(json).expect("the file still parses");
    app.load_project(project);
    assert_eq!(app.project().layers.len(), 1, "the layer is still listed");
    let work = app.desired_raster();
    let archive = work
        .archive
        .expect("a remote MBTiles layer is a real raster source now");
    assert_eq!(archive.format, oxigis_core::ArchiveFormat::MbTiles);
    assert_eq!(archive.location(), "https://example.test/tokyo.mbtiles");
    assert!(
        archive.refusal().is_none(),
        "no reference-and-format pair is refusable any more"
    );
}

#[test]
fn a_probe_failure_reports_the_reason_and_creates_nothing() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddArchiveUrlLayer(
        "https://example.test/not-an-archive.pmtiles".to_string(),
    ));
    assert!(run_probe(&mut app, vec![0x7Au8; 512]).is_none());
    assert_eq!(app.project().layers.len(), 0);
    let status = app.status().expect("a failure must be reported");
    assert!(status.contains("not-an-archive.pmtiles"), "{status}");
    assert!(!app.archive_probe_running());
}

#[test]
fn a_dropped_archives_bytes_are_held_for_the_session_and_keyed_by_name() {
    let mut app = OxigisApp::new();
    let bytes: Arc<[u8]> = Arc::from(sample_pmtiles_raster().into_boxed_slice());
    app.open_archive_bytes(
        ArchiveFormat::PmTiles,
        "dropped.pmtiles",
        Arc::clone(&bytes),
        &egui::Context::default(),
    );
    let id = app
        .poll_archive_probe()
        .expect("a dropped archive opens inline");
    let layer = app.project().layers.get(id).expect("created");
    assert!(matches!(
        layer.kind,
        LayerKind::Raster(RasterSource::TileArchive {
            archive: ArchiveRef::Path { .. },
            ..
        })
    ));
    // The bytes are what a shell will hand the provider, since there is no
    // path to re-read in a browser.
    let held = app
        .archive_bytes("dropped.pmtiles")
        .expect("the session holds the bytes");
    assert_eq!(held.len(), bytes.len());
    assert!(app.archive_bytes("something-else.pmtiles").is_none());
}

#[test]
fn an_archive_larger_than_the_session_budget_is_refused_by_name() {
    let mut app = OxigisApp::new();
    let huge: Arc<[u8]> = Arc::from(vec![0u8; super::archive_io::MAX_SESSION_ARCHIVE_BYTES + 1]);
    assert!(!app.remember_archive_bytes("huge.pmtiles", huge));
    let status = app.status().expect("a refusal must be reported");
    assert!(status.contains("MiB"), "{status}");
    assert!(app.archive_bytes("huge.pmtiles").is_none());
}

#[test]
fn a_dropped_pmtiles_name_routes_through_the_drop_classifier() {
    use crate::local_input::{DropKind, DroppedDataset, DroppedItem, group_dropped_files};

    assert_eq!(
        crate::local_input::classify_drop("Basemap.PMTiles"),
        DropKind::TileArchive(ArchiveFormat::PmTiles)
    );
    assert_eq!(
        crate::local_input::classify_drop("tokyo.mbtiles"),
        DropKind::TileArchive(ArchiveFormat::MbTiles)
    );

    let (datasets, notices) = group_dropped_files(vec![DroppedItem {
        name: "tokyo.pmtiles".to_string(),
        bytes: Some(Arc::from(vec![1u8, 2, 3].into_boxed_slice())),
        path: None,
    }]);
    assert!(notices.is_empty());
    assert!(matches!(
        datasets.as_slice(),
        [DroppedDataset::TileArchive(ArchiveFormat::PmTiles, _)]
    ));
}

#[test]
fn a_url_names_its_format_and_defaults_to_pmtiles() {
    assert_eq!(
        super::archive_io::format_for_url("https://x/a.mbtiles"),
        ArchiveFormat::MbTiles
    );
    assert_eq!(
        super::archive_io::format_for_url("https://x/a.pmtiles"),
        ArchiveFormat::PmTiles
    );
    // A range-served archive with no extension is far likelier to be the
    // format designed for range reads.
    assert_eq!(
        super::archive_io::format_for_url("https://x/tiles"),
        ArchiveFormat::PmTiles
    );
}

#[test]
fn the_print_snapshot_carries_the_archive_the_map_is_drawing() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    app.apply_layer_action(LayerAction::AddArchiveUrlLayer(
        "https://example.test/basemap.pmtiles".to_string(),
    ));
    let _id = run_probe(&mut app, sample_pmtiles_raster()).expect("created");
    app.request_print();
    let request = app.take_pending_print().expect("a print request");
    let archive = request
        .archive
        .expect("the raster archive travels to print");
    assert_eq!(archive.location(), "https://example.test/basemap.pmtiles");
    assert!(request.cog.is_none());
    assert!(request.attribution.contains("OxiGIS test fixture"));
}

#[test]
fn a_dropped_mbtiles_opens_synchronously_and_creates_its_layer() {
    use crate::mbtiles::fixture::{flat_image, vector_metadata};

    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    let bytes: Arc<[u8]> = Arc::from(
        flat_image(&[(0, 0, 0, b"body".to_vec())], &vector_metadata()).into_boxed_slice(),
    );
    // No probe at all: SQLite has no header to prefetch, so the layer exists
    // the moment the bytes are handed over.
    app.open_archive_bytes(
        ArchiveFormat::MbTiles,
        "tokyo.mbtiles",
        bytes,
        &egui::Context::default(),
    );
    assert!(!app.archive_probe_running());
    assert_eq!(app.project().layers.len(), 1);
    let layer = app
        .project()
        .layers
        .layers()
        .first()
        .expect("the layer exists");
    assert_eq!(crate::layer_panel::kind_tag(layer), "mbtiles");
    match &layer.kind {
        LayerKind::Vector(VectorSource::TileArchive { format, paints, .. }) => {
            assert_eq!(*format, ArchiveFormat::MbTiles);
            // The fixture declares `water` and `roads`, which the ramp orders
            // fill-before-line.
            let names: Vec<&str> = paints
                .iter()
                .map(|paint| paint.source_layer.as_str())
                .collect();
            assert_eq!(names, ["water", "roads"]);
        }
        other => panic!("expected a vector tile archive, got {other:?}"),
    }
    // The indexed reader is held for the session, so a shell does not rebuild
    // the index on every reconciliation.
    assert!(app.mbtiles_reader("tokyo.mbtiles").is_some());
    assert!(app.archive_bytes("tokyo.mbtiles").is_some());
}

#[test]
fn an_mbtiles_that_is_not_a_database_reports_the_reason_and_creates_nothing() {
    let mut app = OxigisApp::new();
    app.open_archive_bytes(
        ArchiveFormat::MbTiles,
        "broken.mbtiles",
        Arc::from(b"not a database".to_vec().into_boxed_slice()),
        &egui::Context::default(),
    );
    assert_eq!(app.project().layers.len(), 0);
    let status = app.status().expect("a failure must be reported");
    assert!(status.contains("broken.mbtiles"), "{status}");
    assert!(app.mbtiles_reader("broken.mbtiles").is_none());
}

#[test]
fn the_open_gesture_asks_the_shell_for_a_file_dialog_take_once() {
    let mut app = OxigisApp::new();
    assert!(!app.take_pending_archive_pick());
    app.apply_layer_action(LayerAction::OpenArchiveFile);
    assert!(app.take_pending_archive_pick());
    assert!(
        !app.take_pending_archive_pick(),
        "the request is take-once, like every other shell hand-off"
    );
}

#[test]
fn a_local_pmtiles_path_is_offered_to_the_shell_as_a_probe_request() {
    let mut app = OxigisApp::new();
    assert!(app.request_archive_probe(
        ArchiveRef::Path {
            path: r"C:\dataasemap.pmtiles".to_string(),
        },
        ArchiveFormat::PmTiles,
    ));
    let request = app
        .take_pending_archive_probe()
        .expect("the shell is asked for a transport");
    assert_eq!(request.location(), r"C:\dataasemap.pmtiles");
    assert_eq!(request.format, ArchiveFormat::PmTiles);
}

#[test]
fn opening_an_mbtiles_by_path_holds_its_bytes_under_that_path() {
    use crate::mbtiles::fixture::{flat_image, raster_metadata};

    let mut app = OxigisApp::new();
    let bytes = flat_image(&[(0, 0, 0, b"png".to_vec())], &raster_metadata());
    app.open_archive_bytes(
        ArchiveFormat::MbTiles,
        "/data/tokyo.mbtiles",
        std::sync::Arc::from(bytes.into_boxed_slice()),
        &egui::Context::default(),
    );
    assert_eq!(app.project().layers.len(), 1);
    assert!(app.mbtiles_reader("/data/tokyo.mbtiles").is_some());
    let layer = app.project().layers.layers().first().expect("created");
    assert!(matches!(
        layer.kind,
        LayerKind::Raster(RasterSource::TileArchive {
            format: ArchiveFormat::MbTiles,
            ..
        })
    ));
}
