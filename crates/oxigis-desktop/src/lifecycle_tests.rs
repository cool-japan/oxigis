// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for the shell's project/window lifecycle: the File ▸ Save / Open
//! seams against real files, the recent list, the close-request latch, and the
//! window this shell opens.
//!
//! A sibling module rather than a `mod tests` inside `main.rs`, under the
//! 2000-line rule — a child module of the crate root sees the root's private
//! items, so `OxigisDesktopApp` and its fields are reachable from here exactly
//! as they would be in place.

use std::path::PathBuf;

use crate::{
    OxigisDesktopApp, export_file_name, file_dialog, project_file, session, viewport_builder,
};

/// Two Tokyo/Osaka points — enough to make a project dirty and non-trivial.
const POINTS: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"Tokyo"},
     "geometry":{"type":"Point","coordinates":[139.767,35.681]}},
    {"type":"Feature","properties":{"name":"Osaka"},
     "geometry":{"type":"Point","coordinates":[135.502,34.702]}}]}"#;

/// A directory under the OS temp dir, unique per call.
fn scratch_dir(label: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let path = std::env::temp_dir().join(format!("oxigis-lifecycle-{label}-{stamp}"));
    std::fs::create_dir_all(&path).expect("the scratch directory is creatable");
    path
}

/// A shell with no startup paths and a blank session.
fn shell() -> OxigisDesktopApp {
    OxigisDesktopApp::new(Vec::new(), session::SessionState::default())
}

#[test]
fn the_shell_declares_real_project_io_so_save_is_a_write_and_not_a_textbox() {
    let app = shell();
    assert!(
        app.inner.native_project_io(),
        "the native shell owns a filesystem and must say so",
    );
    assert!(app.inner.project_path().is_none(), "nothing is open yet");
    assert!(!app.closing);
}

/// The whole File ▸ Save As → File ▸ Open round trip against real files, from
/// the seam the UI hands over to the path it hands back.
#[test]
fn a_project_saves_to_a_file_and_opens_back_out_of_it() {
    let dir = scratch_dir("roundtrip");
    let path = dir.join("cities.oxigis.json");
    let mut app = shell();
    let _layer = app
        .inner
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    assert!(app.inner.has_unsaved_changes());

    // The UI's own File ▸ Save queues the request; the shell resolves it.
    app.inner.request_project_save(true);
    let request = app
        .inner
        .take_pending_project_save()
        .expect("File ▸ Save As must queue a write");
    assert!(request.path.is_none(), "Save As always asks");
    assert!(
        app.write_project(&request.content, path.clone()),
        "the write succeeds"
    );

    assert!(path.is_file(), "the bytes landed");
    assert!(
        !app.inner.has_unsaved_changes(),
        "a confirmed write clears the marker",
    );
    assert_eq!(app.inner.project_path(), Some(path.as_path()));
    assert_eq!(app.inner.recent_projects(), std::slice::from_ref(&path));
    assert_eq!(app.session.recent, vec![path.clone()]);
    assert_eq!(app.session.last_directory.as_deref(), Some(dir.as_path()));

    // A plain Ctrl+S now writes back without asking.
    app.inner.request_project_save(false);
    let again = app
        .inner
        .take_pending_project_save()
        .expect("a second save queues too");
    assert_eq!(again.path.as_deref(), Some(path.as_path()));

    // And the document really is loadable, in a fresh shell.
    let mut reopened = shell();
    assert!(reopened.read_project(path.clone()), "the document opens");
    assert_eq!(reopened.inner.project().layers.len(), 1);
    assert_eq!(reopened.inner.project_path(), Some(path.as_path()));
    assert!(
        !reopened.inner.has_unsaved_changes(),
        "the file on disk IS what is open",
    );
    assert_eq!(reopened.session.recent, vec![path]);
    let _removed = std::fs::remove_dir_all(&dir);
}

/// A save stamps the live camera in, which is the difference between a project
/// that reopens where the user was and one that reopens at Null Island.
#[test]
fn a_saved_project_carries_the_camera_and_basemap_the_map_was_showing() {
    let dir = scratch_dir("camera");
    let path = dir.join("view.oxigis.json");
    let mut app = shell();
    let _layer = app
        .inner
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    // A real camera move, through the app's own public gesture.
    assert!(app.inner.zoom_to_selected_layer());
    let camera = app.inner.map_view();
    assert!(
        app.inner.project().basemap.is_none(),
        "the premise: nothing stamped yet",
    );

    app.inner.request_project_save(true);
    let request = app
        .inner
        .take_pending_project_save()
        .expect("a write is queued");
    assert!(app.write_project(&request.content, path.clone()));

    let mut reopened = shell();
    assert!(reopened.read_project(path));
    let view = reopened.inner.project().view;
    assert!(
        (view.center_lon - camera.center().lon).abs() < 1e-6
            && (view.center_lat - camera.center().lat).abs() < 1e-6
            && (view.zoom - camera.zoom()).abs() < 1e-6,
        "the saved document must reopen where the user was: {view:?} vs {camera:?}",
    );
    assert!(
        reopened.inner.project().basemap.is_some(),
        "the basemap the map was drawing is part of the document",
    );
    let _removed = std::fs::remove_dir_all(&dir);
}

/// A file that is not a project must not become "the open project's path", or
/// the next Ctrl+S would overwrite it with something unrelated.
#[test]
fn opening_a_file_that_is_not_a_project_changes_nothing() {
    let dir = scratch_dir("garbage");
    let path = dir.join("notes.oxigis.json");
    std::fs::write(&path, b"this is not JSON at all").expect("the fixture is writable");
    let mut app = shell();
    let _layer = app
        .inner
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");

    assert!(
        !app.read_project(path),
        "a document that is not a project is refused"
    );
    assert_eq!(
        app.inner.project().layers.len(),
        1,
        "the open project survived the failed read",
    );
    assert!(app.inner.project_path().is_none(), "nothing was opened");
    assert!(app.inner.recent_projects().is_empty());
    let status = app.inner.status().unwrap_or_default().to_string();
    assert!(status.contains("not an OxiGIS project"), "{status}");

    // A file that is not there at all fails the same way.
    let mut app = shell();
    assert!(!app.read_project(dir.join("absent.oxigis.json")));
    assert!(app.inner.project_path().is_none());
    let _removed = std::fs::remove_dir_all(&dir);
}

/// A chosen name with no extension gains `.oxigis.json`, and the file that
/// lands is the one the UI is told about.
#[test]
fn a_save_path_without_an_extension_gains_one_and_is_reported_as_written() {
    let dir = scratch_dir("extension");
    let mut app = shell();
    app.inner.request_project_save(true);
    let request = app
        .inner
        .take_pending_project_save()
        .expect("a write is queued");
    assert!(app.write_project(&request.content, dir.join("city")));

    let expected = dir.join(format!("city.{}", project_file::PROJECT_EXTENSION));
    assert!(expected.is_file(), "the extension was added on disk");
    assert_eq!(
        app.inner.project_path(),
        Some(expected.as_path()),
        "and the UI was told about the path that really exists",
    );
    let _removed = std::fs::remove_dir_all(&dir);
}

/// A write that cannot succeed must leave the project marked unsaved, so the
/// user is asked again before anything discards it.
#[test]
fn a_failed_write_leaves_the_project_dirty_and_unattached() {
    let dir = scratch_dir("failed");
    let mut app = shell();
    let _layer = app
        .inner
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    app.inner.request_project_save(true);
    let request = app
        .inner
        .take_pending_project_save()
        .expect("a write is queued");

    // A directory cannot be replaced by a file.
    let occupied = dir.join("occupied.oxigis.json");
    std::fs::create_dir(&occupied).expect("the directory is creatable");
    std::fs::write(occupied.join("inside"), b"x").expect("the child is writable");
    assert!(
        !app.write_project(&request.content, occupied),
        "a directory cannot be replaced by a file",
    );

    assert!(
        app.inner.has_unsaved_changes(),
        "a write that failed did not save anything",
    );
    assert!(app.inner.project_path().is_none());
    assert!(app.inner.recent_projects().is_empty());
    let status = app.inner.status().unwrap_or_default().to_string();
    assert!(status.starts_with("Save failed"), "{status}");
    let _removed = std::fs::remove_dir_all(&dir);
}

/// The close latch. Confirming does NOT make the project clean, so a shell
/// that kept intercepting would re-ask about its own `Close` command for ever.
#[test]
fn the_close_intercept_latches_once_the_user_has_answered() {
    let ctx = egui::Context::default();

    // A clean project has nothing to lose: the question answers itself.
    let mut clean = shell();
    clean.inner.request_window_close();
    clean.intercept_close(&ctx);
    assert!(clean.closing, "a clean project closes without a dialog");
    // And it stays latched: a second pass must not re-ask.
    clean.inner.request_window_close();
    clean.intercept_close(&ctx);
    assert!(clean.closing);

    // A dirty project holds the close until the user answers.
    let mut dirty = shell();
    let _layer = dirty
        .inner
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    dirty.inner.request_window_close();
    dirty.intercept_close(&ctx);
    assert!(
        !dirty.closing,
        "the window must not close over unsaved changes",
    );
    assert_eq!(
        dirty.inner.project().layers.len(),
        1,
        "and nothing was lost"
    );

    // Saving is what makes the answer unnecessary: the very next close request
    // goes straight through, with no dialog and nothing to discard.
    dirty.inner.mark_saved();
    dirty.inner.request_window_close();
    dirty.intercept_close(&ctx);
    assert!(dirty.closing, "a saved project quits without being asked");
}

/// Quitting after a real save goes through; quitting after a failed one does
/// not. The chaining itself is `oxigis-ui`'s (see its `tests_session`); this is
/// the shell's half — that a write which failed leaves the app running.
#[test]
fn a_failed_write_does_not_let_the_window_close() {
    let dir = scratch_dir("savequit");
    let ctx = egui::Context::default();
    let mut app = shell();
    let _layer = app
        .inner
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");

    app.inner.request_project_save(true);
    let request = app
        .inner
        .take_pending_project_save()
        .expect("a write is queued");
    // A directory cannot be replaced by a file, so this write fails.
    let occupied = dir.join("occupied.oxigis.json");
    std::fs::create_dir(&occupied).expect("the directory is creatable");
    std::fs::write(occupied.join("inside"), b"x").expect("the child is writable");
    assert!(!app.write_project(&request.content, occupied));

    app.inner.request_window_close();
    app.intercept_close(&ctx);
    assert!(
        !app.closing,
        "a project whose save failed still has everything to lose",
    );

    // The same gesture once the write succeeds does close.
    app.inner.request_project_save(true);
    let request = app
        .inner
        .take_pending_project_save()
        .expect("a fresh write is queued");
    assert!(app.write_project(&request.content, dir.join("cities.oxigis.json")));
    app.inner.request_window_close();
    app.intercept_close(&ctx);
    assert!(app.closing, "the bytes are on disk; the app may go");
    let _removed = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_recent_list_is_seeded_from_the_session_and_read_back_bounded() {
    let mut session = session::SessionState::default();
    for index in 0..(session::MAX_RECENT + 4) {
        session.note_recent(PathBuf::from(format!("/data/p{index}.oxigis.json")));
    }
    let app = OxigisDesktopApp::new(Vec::new(), session.clone());
    assert_eq!(app.inner.recent_projects().len(), session::MAX_RECENT);
    assert_eq!(
        app.inner.recent_projects()[0],
        PathBuf::from(format!("/data/p{}.oxigis.json", session::MAX_RECENT + 3)),
        "most recent first, on both sides of the seam",
    );
    assert_eq!(oxigis_ui::MAX_RECENT_PROJECTS, session::MAX_RECENT);
}

/// Two exports in the same second must not overwrite each other — the whole
/// reason the name is not just a whole-second timestamp.
#[test]
fn every_export_gets_its_own_file_name() {
    let names: std::collections::BTreeSet<String> = (0..64).map(|_| export_file_name()).collect();
    assert_eq!(names.len(), 64, "an export name is used once");
    assert!(names.iter().all(|name| name.ends_with(".pdf")));
    assert!(names.iter().all(|name| name.starts_with("oxigis-export-")));
}

#[test]
fn the_window_restores_the_remembered_geometry_and_repairs_a_broken_one() {
    // First launch: the default size, no position to force.
    let fresh = viewport_builder(&session::SessionState::default());
    assert_eq!(
        fresh.inner_size,
        Some(egui::vec2(
            session::DEFAULT_WINDOW_SIZE[0],
            session::DEFAULT_WINDOW_SIZE[1]
        )),
    );
    assert_eq!(fresh.position, None, "let the window manager place it");
    assert_eq!(fresh.app_id.as_deref(), Some(crate::APP_ID));
    assert_eq!(fresh.title.as_deref(), Some("OxiGIS"));

    // A remembered window comes back where it was.
    let remembered = session::SessionState {
        window: Some(session::WindowGeometry {
            x: 120.0,
            y: 80.0,
            width: 1440.0,
            height: 900.0,
        }),
        maximized: true,
        ..session::SessionState::default()
    };
    let restored = viewport_builder(&remembered);
    assert_eq!(restored.inner_size, Some(egui::vec2(1440.0, 900.0)));
    assert_eq!(restored.position, Some(egui::pos2(120.0, 80.0)));
    assert_eq!(restored.maximized, Some(true));

    // A corrupt one is refused rather than opening a 2-pixel window.
    let broken = session::SessionState {
        window: Some(session::WindowGeometry {
            x: f32::NAN,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        }),
        ..session::SessionState::default()
    };
    let repaired = viewport_builder(&broken);
    assert_eq!(
        repaired.inner_size,
        Some(egui::vec2(
            session::DEFAULT_WINDOW_SIZE[0],
            session::DEFAULT_WINDOW_SIZE[1]
        )),
    );
    assert_eq!(repaired.position, None);
}

/// A second ask while one is already on screen must not replace it — the
/// parked snapshot behind the first would be lost with it.
#[test]
#[cfg(not(any(windows, target_os = "macos")))]
fn a_second_ask_waits_for_the_prompt_that_is_already_up() {
    let mut app = shell();
    app.inner.request_project_save(true);
    let request = app
        .inner
        .take_pending_project_save()
        .expect("a write is queued");
    app.resolve_project_save(request);
    assert!(app.path_prompt.is_some(), "the in-app prompt stands in");
    assert!(app.pending_project_save.is_some(), "the bytes are parked");

    // A tile-archive pick arriving now must not take the prompt away.
    app.inner.request_archive_pick();
    assert!(app.inner.take_pending_archive_pick());
    assert!(app.path_prompt.is_some());
    assert!(
        app.pending_project_save.is_some(),
        "the parked project save survived",
    );
}

// ---- The in-app prompt's dispatch, on every platform -------------------
//
// `ask_for_path` short-circuits to the native dialog wherever there is one, so
// on macOS and Windows `PathPrompt` is unreachable at RUNTIME — which would
// leave `deliver_path` and the parking either side of it (the Linux fallback,
// a named deliverable) with no executed coverage on the machines this is
// developed on. These tests set the same state the prompt does and call the
// dispatch directly, so the logic is exercised on every host.

/// The prompt's state as the fallback leaves it: an ask on screen with the
/// bytes parked behind it.
fn armed_prompt(app: &mut OxigisDesktopApp, ask: file_dialog::Ask, base: PathBuf) {
    app.path_prompt = Some(file_dialog::PathPrompt::new(
        ask,
        "cities.oxigis.json",
        base,
    ));
}

#[test]
fn a_delivered_path_settles_the_ask_and_a_correctable_failure_keeps_it_open() {
    let dir = scratch_dir("deliver");
    let ctx = egui::Context::default();
    let mut app = shell();
    let _layer = app
        .inner
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    app.inner.request_project_save(true);
    let request = app
        .inner
        .take_pending_project_save()
        .expect("a write is queued");

    // A destination that cannot be written: the ask is NOT settled, the bytes
    // stay parked, and the prompt keeps the reason so the user can correct it.
    let occupied = dir.join("occupied.oxigis.json");
    std::fs::create_dir(&occupied).expect("the directory is creatable");
    std::fs::write(occupied.join("inside"), b"x").expect("the child is writable");
    armed_prompt(&mut app, file_dialog::Ask::SaveProject, dir.clone());
    app.pending_project_save = Some(request);
    assert!(
        !app.deliver_path(file_dialog::Ask::SaveProject, occupied, &ctx),
        "a failed write must not close the prompt",
    );
    assert!(
        app.pending_project_save.is_some(),
        "the bytes stay parked for a corrected path",
    );
    assert!(app.inner.has_unsaved_changes());

    // The corrected path saves those very bytes and settles the ask.
    let good = dir.join("cities.oxigis.json");
    assert!(app.deliver_path(file_dialog::Ask::SaveProject, good.clone(), &ctx));
    assert!(app.pending_project_save.is_none(), "the park is spent");
    assert!(!app.inner.has_unsaved_changes());
    assert_eq!(app.inner.project_path(), Some(good.as_path()));
    let _removed = std::fs::remove_dir_all(&dir);
}

#[test]
fn delivering_a_path_nothing_is_waiting_for_still_closes_the_prompt() {
    let dir = scratch_dir("orphan");
    let ctx = egui::Context::default();
    let mut app = shell();

    // Nothing parked: the ask must settle anyway, or the prompt wedges on
    // screen with no way to dismiss it but Cancel.
    armed_prompt(&mut app, file_dialog::Ask::SaveProject, dir.clone());
    assert!(app.deliver_path(
        file_dialog::Ask::SaveProject,
        dir.join("nothing.oxigis.json"),
        &ctx,
    ));
    let status = app.inner.status().unwrap_or_default().to_string();
    assert!(status.contains("no longer waiting"), "{status}");

    armed_prompt(&mut app, file_dialog::Ask::SavePdf, dir.clone());
    assert!(app.deliver_path(file_dialog::Ask::SavePdf, dir.join("page.pdf"), &ctx));
    assert!(app.print_job.is_none(), "no snapshot, no export thread");
    let _removed = std::fs::remove_dir_all(&dir);
}

#[test]
fn delivering_an_unreadable_project_keeps_the_prompt_and_the_open_project() {
    let dir = scratch_dir("badopen");
    let ctx = egui::Context::default();
    let mut app = shell();
    let _layer = app
        .inner
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    let garbage = dir.join("notes.oxigis.json");
    std::fs::write(&garbage, b"not a project").expect("the fixture is writable");

    armed_prompt(&mut app, file_dialog::Ask::OpenProject, dir.clone());
    assert!(
        !app.deliver_path(file_dialog::Ask::OpenProject, garbage, &ctx),
        "the user can correct the path, so the prompt stays",
    );
    assert!(app.inner.project_path().is_none());
    assert_eq!(app.inner.project().layers.len(), 1, "nothing was replaced");

    // An archive pick always settles: whatever the probe makes of the file is
    // reported on the status line, not back into the prompt.
    armed_prompt(&mut app, file_dialog::Ask::OpenArchive, dir.clone());
    assert!(app.deliver_path(
        file_dialog::Ask::OpenArchive,
        dir.join("tiles.pmtiles"),
        &ctx,
    ));
    let _removed = std::fs::remove_dir_all(&dir);
}

/// The prompt's cancel path drops everything parked behind it — the invariant
/// that keeps a "save, then quit" the user abandoned from firing behind the
/// next unrelated save.
#[test]
fn cancelling_the_prompt_drops_every_parked_ask() {
    let ctx = egui::Context::default();
    let mut app = shell();
    let _layer = app
        .inner
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON must be accepted");
    app.inner.request_project_save(true);
    app.pending_project_save = app.inner.take_pending_project_save();
    armed_prompt(
        &mut app,
        file_dialog::Ask::SaveProject,
        std::env::temp_dir(),
    );

    // Escape is the prompt's dismissal, delivered as a real egui frame.
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        events: vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };
    let _ = ctx.run_ui(raw_input, |ui| {
        let ctx = ui.ctx().clone();
        app.poll_path_prompt(&ctx);
    });

    assert!(app.path_prompt.is_none(), "Escape dismisses the prompt");
    assert!(
        app.pending_project_save.is_none(),
        "and the parked bytes go with it",
    );
    assert!(
        app.inner.has_unsaved_changes(),
        "a cancelled save saved nothing",
    );
}
