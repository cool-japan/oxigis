// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Reading datasets off the local filesystem for the native shell.
//!
//! `oxigis-ui` compiles to `wasm32` and owns no filesystem, so every path it
//! wants read comes back through this module: a drag-and-drop (which
//! `egui-winit` reports as a bare path), a `.shp` whose siblings have to be
//! found by extension, a `.gpkg` table, a `.parquet`, a `.geolibre.json`
//! project, and the positional arguments a file association turns into an
//! "Open With".
//!
//! Split out of `main.rs` under the 2000-line rule; nothing here touches the
//! window, the GPU or a provider, which is why it moves as one piece.

/// Largest dataset this shell reads into memory in one go.
///
/// The vector twin of [`oxigis_ui::MAX_SESSION_ARCHIVE_BYTES`]: a `.gpkg`, a
/// `.parquet` or a GeoJSON is read whole *and parsed whole*, on the thread
/// that paints, so an accidentally-dropped disk image has to be refused by
/// name instead of freezing the window for the length of the read and then
/// running the process out of memory. A dataset past this size belongs in a
/// tile archive, which streams (`range_file`) instead of being read.
const MAX_DATASET_BYTES: u64 = 512 * 1024 * 1024;

/// How many file notices are spelled out before the rest are only counted.
const MAX_LISTED_NOTICES: usize = 3;

/// Reads a whole dataset file, refusing anything past [`MAX_DATASET_BYTES`].
///
/// The error is the message the user sees, so it names the file and the cap.
fn read_dataset(path: &std::path::Path, name: &str) -> Result<Vec<u8>, String> {
    read_capped(path, name, MAX_DATASET_BYTES)
}

/// [`read_dataset`] with the cap named, so the refusal is testable without a
/// half-gigabyte fixture.
pub(crate) fn read_capped(path: &std::path::Path, name: &str, cap: u64) -> Result<Vec<u8>, String> {
    use std::io::Read as _;
    let file =
        std::fs::File::open(path).map_err(|error| format!("Could not read {name}: {error}"))?;
    // An unreadable stat is not a refusal: `take` below bounds the read
    // whatever the metadata said (or did not say, for a pipe or a device).
    let length = file.metadata().map_or(0, |metadata| metadata.len());
    if length > cap {
        return Err(too_large(name, length, cap));
    }
    // One byte past the cap, so "it fits" and "it was cut" stay distinguishable:
    // metadata is advisory (a growing file, a device, a pipe all under-report),
    // and a truncated document handed to a parser fails as a corrupt file
    // instead of as the oversized one it is.
    let mut bytes = Vec::new();
    file.take(cap.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Could not read {name}: {error}"))?;
    let read = bytes.len() as u64;
    if read > cap {
        return Err(too_large(name, read, cap));
    }
    Ok(bytes)
}

/// The refusal a dataset past the cap is reported with, naming both sizes and
/// the format that would not have to be read whole.
fn too_large(name: &str, length: u64, cap: u64) -> String {
    format!(
        "{name} is {} MB; OxiGIS reads at most {} MB into memory. \
         Convert it to a .pmtiles/.mbtiles archive, which streams.",
        length / (1024 * 1024),
        cap / (1024 * 1024),
    )
}

/// The name a file is reported under: its last component, or the whole path
/// when it has none.
fn drop_name(path: &std::path::Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Folds per-file notices into the one status line the app has.
///
/// [`oxigis_ui::OxigisApp::set_status`] *replaces*, so reporting each failure
/// as it happens would leave only the last one visible — which for a command
/// line of six bad paths is a worse answer than saying how many there were.
fn join_notices(notices: &[String]) -> Option<String> {
    let listed = notices.len().min(MAX_LISTED_NOTICES);
    let (head, rest) = notices.split_at(listed);
    if head.is_empty() {
        return None;
    }
    let mut message = head.join(" ");
    if !rest.is_empty() {
        message.push_str(&format!(" (and {} more.)", rest.len()));
    }
    Some(message)
}

/// Reads every file the app is waiting on and hands the bytes back.
///
/// `oxigis-ui` compiles to `wasm32` and does no file I/O at all, so a native
/// drag-and-drop (which `egui-winit` reports as a bare path) and a project that
/// references a `.geojson` by path both land here — and they are *not* the same
/// operation. A fresh drop becomes a new layer, keeping the path so the project
/// stores a reference rather than a copy of the document. A project-load
/// reference must instead rebuild the layer that is already in the project,
/// which is what `PendingPath::layer` distinguishes.
pub(crate) fn drain_dropped_paths(app: &mut oxigis_ui::OxigisApp) {
    let pending = app.take_pending_dropped_paths();
    let notices = read_pending_paths(app, pending);
    if let Some(message) = join_notices(&notices) {
        app.set_status(message);
    }
}

/// Reads `pending` in order, returning one notice per file that could not be
/// used — the caller decides how to report them, because the command-line
/// route reports a whole argument list at once.
fn read_pending_paths(
    app: &mut oxigis_ui::OxigisApp,
    pending: Vec<oxigis_ui::PendingPath>,
) -> Vec<String> {
    let mut notices = Vec::new();
    for pending in pending {
        let path_text = pending.path.display().to_string();
        let name = drop_name(&pending.path);
        // A `.shp` needs its siblings, which `oxigis-ui` cannot go looking for;
        // a `.gpkg` is one file but potentially many layers; everything else is
        // a single GeoJSON document.
        let notice = match oxigis_ui::classify_drop(&name) {
            oxigis_ui::DropKind::Shapefile(_) => {
                read_shapefile_set(app, &pending, &path_text, &name)
            }
            oxigis_ui::DropKind::GeoPackage => read_geopackage(app, &pending, &path_text, &name),
            oxigis_ui::DropKind::GeoParquet => read_geoparquet(app, &pending, &path_text, &name),
            // A fresh `.geolibre.json` drop (`pending.layer` is `None`) is a
            // whole-project import — see `read_geolibre_project`. A
            // `Some(id)` for that same name is instead an ordinary
            // `VectorSource::LocalGeoJson`/etc. project-load reference whose
            // file simply happens to be named `*.geolibre.json`; that case
            // falls through to the plain-GeoJSON read below, unchanged.
            oxigis_ui::DropKind::GeoLibreProject if pending.layer.is_none() => {
                read_geolibre_project(app, &pending, &path_text, &name)
            }
            // A dropped archive is handled by the app's own drop router (it
            // asks this shell for a transport), so nothing for it ever lands
            // in the pending-path queue.
            oxigis_ui::DropKind::TileArchive(_) => continue,
            oxigis_ui::DropKind::GeoJson
            | oxigis_ui::DropKind::GeoLibreProject
            | oxigis_ui::DropKind::Unsupported => read_geojson(app, &pending, &path_text, &name),
        };
        notices.extend(notice);
    }
    notices
}

/// Reads a GeoJSON document and hands its bytes to the app.
fn read_geojson(
    app: &mut oxigis_ui::OxigisApp,
    pending: &oxigis_ui::PendingPath,
    path_text: &str,
    name: &str,
) -> Option<String> {
    match read_dataset(&pending.path, name) {
        Ok(bytes) => {
            tracing::info!(
                file = path_text,
                bytes = bytes.len(),
                rebuild = pending.layer.is_some(),
                "OxiGIS desktop: read GeoJSON",
            );
            match pending.layer {
                Some(id) => {
                    app.hydrate_geojson_layer_from_bytes(id, name, &bytes);
                }
                None => {
                    app.add_geojson_layer_from_bytes(name, &bytes, Some(path_text));
                }
            }
            None
        }
        Err(notice) => {
            tracing::error!(file = path_text, %notice, "OxiGIS desktop: could not read the file");
            Some(notice)
        }
    }
}

/// Reads a `.shp` and whichever of its siblings exist, then hands the bytes to
/// the app.
///
/// The sibling discovery lives here rather than in `oxigis-ui` for the usual
/// reason — that crate compiles to `wasm32` and touches no filesystem — and it
/// is what makes [`oxigis_ui::PendingPath`] able to carry the `.shp` alone: a
/// project-load reference has no other member recorded anywhere.
fn read_shapefile_set(
    app: &mut oxigis_ui::OxigisApp,
    pending: &oxigis_ui::PendingPath,
    path_text: &str,
    name: &str,
) -> Option<String> {
    let shp = match read_dataset(&pending.path, name) {
        Ok(bytes) => bytes,
        Err(notice) => {
            tracing::error!(file = path_text, %notice, "OxiGIS desktop: could not read the .shp");
            return Some(notice);
        }
    };
    let dbf = read_sibling(&pending.path, "dbf");
    let prj = read_sibling(&pending.path, "prj").and_then(|bytes| String::from_utf8(bytes).ok());
    let cpg = read_sibling(&pending.path, "cpg").and_then(|bytes| String::from_utf8(bytes).ok());
    tracing::info!(
        file = path_text,
        bytes = shp.len(),
        dbf = dbf.is_some(),
        prj = prj.is_some(),
        rebuild = pending.layer.is_some(),
        "OxiGIS desktop: read Shapefile",
    );
    let bytes = oxigis_ui::ShapefileBytes::new(&shp)
        .with_dbf(dbf.as_deref())
        .with_sidecars(prj.as_deref(), cpg.as_deref());
    match pending.layer {
        Some(id) => {
            app.hydrate_shapefile_layer_from_bytes(id, name, bytes);
        }
        None => {
            app.add_shapefile_layer_from_bytes(name, bytes, Some(path_text));
        }
    }
    None
}

/// Reads a `.gpkg` and hands its bytes to the app.
///
/// A GeoPackage is self-contained, so unlike a shapefile there is nothing to go
/// looking for beside it — but it is also the one source whose path does not
/// identify the layer: a fresh drop ([`oxigis_ui::PendingPath::table`] of
/// `None`) imports *every* feature table as its own layer, while a project-load
/// reference names the one table to rebuild.
fn read_geopackage(
    app: &mut oxigis_ui::OxigisApp,
    pending: &oxigis_ui::PendingPath,
    path_text: &str,
    name: &str,
) -> Option<String> {
    let bytes = match read_dataset(&pending.path, name) {
        Ok(bytes) => bytes,
        Err(notice) => {
            tracing::error!(file = path_text, %notice, "OxiGIS desktop: could not read the .gpkg");
            return Some(notice);
        }
    };
    tracing::info!(
        file = path_text,
        bytes = bytes.len(),
        table = pending.table.as_deref().unwrap_or("*"),
        rebuild = pending.layer.is_some(),
        "OxiGIS desktop: read GeoPackage",
    );
    match (pending.layer, pending.table.as_deref()) {
        (Some(id), Some(table)) => {
            app.hydrate_gpkg_layer_from_bytes(id, name, &bytes, table);
            None
        }
        // A layer reference with no table recorded cannot be rebuilt without
        // guessing which of the file's tables it was; say so instead.
        (Some(_), None) => Some(format!(
            "{name} does not say which of its tables this layer came from; re-drop the file.",
        )),
        (None, _) => {
            app.add_gpkg_layer_from_bytes(name, &bytes, Some(path_text));
            None
        }
    }
}

/// Reads a `.parquet`/`.geoparquet` file and hands its bytes to the app.
///
/// Self-contained like a `.gpkg`, but — like a `.shp` — always becomes
/// exactly one layer, so there is no per-table branch to resolve here the
/// way [`read_geopackage`] has to.
///
/// `oxigis_ui::OxigisApp::add_geoparquet_layer_from_bytes`/
/// `hydrate_geoparquet_layer_from_bytes` only exist when `oxigis-ui`'s
/// `geoparquet` Cargo feature is on, which this crate's `Cargo.toml` always
/// requests — see `crates/oxigis-ui/src/geoparquet_input`'s module docs for
/// why the feature exists at all.
fn read_geoparquet(
    app: &mut oxigis_ui::OxigisApp,
    pending: &oxigis_ui::PendingPath,
    path_text: &str,
    name: &str,
) -> Option<String> {
    let bytes = match read_dataset(&pending.path, name) {
        Ok(bytes) => bytes,
        Err(notice) => {
            tracing::error!(file = path_text, %notice, "OxiGIS desktop: could not read the .parquet");
            return Some(notice);
        }
    };
    tracing::info!(
        file = path_text,
        bytes = bytes.len(),
        rebuild = pending.layer.is_some(),
        "OxiGIS desktop: read GeoParquet",
    );
    match pending.layer {
        Some(id) => {
            app.hydrate_geoparquet_layer_from_bytes(id, name, &bytes);
        }
        None => {
            app.add_geoparquet_layer_from_bytes(name, &bytes, Some(path_text));
        }
    }
    None
}

/// Reads a native `.geolibre.json` drop and imports it as a whole project,
/// replacing whatever was open.
///
/// Only ever called for `pending.layer.is_none()` — see the guard at this
/// function's one call site in [`drain_dropped_paths`] — so there is no
/// per-layer rebuild branch to resolve here the way every other `read_*`
/// helper in this file has.
fn read_geolibre_project(
    app: &mut oxigis_ui::OxigisApp,
    pending: &oxigis_ui::PendingPath,
    path_text: &str,
    name: &str,
) -> Option<String> {
    match read_dataset(&pending.path, name) {
        Ok(bytes) => {
            tracing::info!(
                file = path_text,
                bytes = bytes.len(),
                "OxiGIS desktop: read GeoLibre project",
            );
            app.load_geolibre_project_from_bytes(name, &bytes);
            None
        }
        Err(notice) => {
            tracing::error!(file = path_text, %notice, "OxiGIS desktop: could not read the file");
            Some(notice)
        }
    }
}

/// Reads `shp_path` with its extension replaced by `extension`, trying the
/// upper-cased variant too (Windows-authored sets are routinely `CITIES.DBF`
/// beside `cities.shp`), and returning [`None`] when neither exists.
///
/// Capped like every other dataset read: a `.dbf` carries the whole attribute
/// table and is routinely the largest member of the set.
fn read_sibling(shp_path: &std::path::Path, extension: &str) -> Option<Vec<u8>> {
    for candidate in [
        shp_path.with_extension(extension),
        shp_path.with_extension(extension.to_ascii_uppercase()),
    ] {
        if !candidate.is_file() {
            continue;
        }
        match read_dataset(&candidate, &drop_name(&candidate)) {
            Ok(bytes) => return Some(bytes),
            // A sibling that exists and still could not be read is not the
            // same as one that is absent: the set loads without it (no
            // attributes, or the WGS 84 default CRS), so say why.
            Err(notice) => {
                tracing::warn!(%notice, "OxiGIS desktop: a shapefile sibling was skipped")
            }
        }
    }
    None
}

/// Opens the paths named on the command line, through the same routes a drop
/// takes.
///
/// This is what makes a file association work: `oxigis city.gpkg`, "Open
/// With", and a double-clicked `.pmtiles` all arrive here, are classified by
/// [`oxigis_ui::classify_drop`] exactly as a dropped file is, and become the
/// same layers. Everything that could not be opened is reported together —
/// see [`join_notices`].
pub(crate) fn open_startup_paths(app: &mut oxigis_ui::OxigisApp, paths: Vec<std::path::PathBuf>) {
    let mut pending = Vec::new();
    let mut notices = Vec::new();
    let mut archive = None;
    for path in paths {
        let name = drop_name(&path);
        match std::fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() => {}
            Ok(_) => {
                notices.push(format!("{name} is not a file."));
                continue;
            }
            Err(error) => {
                notices.push(format!("Could not open {name}: {error}"));
                continue;
            }
        }
        match oxigis_ui::classify_drop(&name) {
            // One archive layer is drawn at a time, and a second probe request
            // silently replaces the first's, so the rest are refused by name.
            oxigis_ui::DropKind::TileArchive(format) if archive.is_none() => {
                archive = Some((path.display().to_string(), format));
            }
            oxigis_ui::DropKind::TileArchive(_) => notices.push(format!(
                "Only one tile archive can be opened at a time; {name} was not opened.",
            )),
            oxigis_ui::DropKind::Unsupported => {
                notices.push(format!("{name} is not a file type OxiGIS can open."));
            }
            oxigis_ui::DropKind::GeoJson
            | oxigis_ui::DropKind::GeoLibreProject
            | oxigis_ui::DropKind::GeoPackage
            | oxigis_ui::DropKind::GeoParquet
            | oxigis_ui::DropKind::Shapefile(_) => pending.push(oxigis_ui::PendingPath {
                layer: None,
                path,
                table: None,
            }),
        }
    }
    notices.extend(read_pending_paths(app, pending));
    if let Some((path, format)) = archive {
        // The probe writes its own "Reading …" status; a notice below replaces
        // it, and `poll_archive_probe` reports the archive's own outcome when
        // the header lands either way.
        if !app.request_archive_probe(oxigis_core::ArchiveRef::Path { path }, format) {
            notices.push(
                app.status()
                    .map_or_else(|| "The archive was refused.".to_owned(), str::to_owned),
            );
        }
    }
    if let Some(message) = join_notices(&notices) {
        app.set_status(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch path under the OS temp directory, unique per call.
    fn scratch(label: &str) -> std::path::PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        std::env::temp_dir().join(format!("oxigis-desktop-{label}-{stamp}"))
    }

    /// The cap is what keeps a dropped disk image from being read whole on
    /// the thread that paints: a file past it is refused BY NAME, and the
    /// message says what the limit is.
    #[test]
    fn a_dataset_past_the_cap_is_refused_and_one_under_it_is_read() {
        let path = scratch("cap");
        std::fs::write(&path, vec![b'x'; 64]).expect("the fixture is writable");
        let read = read_capped(&path, "cities.geojson", 64).expect("64 bytes fit a 64-byte cap");
        assert_eq!(read.len(), 64);
        let refusal = read_capped(&path, "cities.geojson", 32).expect_err("65 bytes do not fit");
        assert!(refusal.contains("cities.geojson"), "{refusal}");
        assert!(refusal.contains("at most"), "{refusal}");
        let _removed = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_dataset_reports_the_name_rather_than_panicking() {
        let error = read_dataset(&scratch("absent"), "absent.gpkg").expect_err("no such file");
        assert!(error.contains("absent.gpkg"), "{error}");
    }

    #[test]
    fn notices_are_folded_into_one_bounded_status_line() {
        assert_eq!(join_notices(&[]), None);
        let two = [String::from("A failed."), String::from("B failed.")];
        assert_eq!(join_notices(&two).as_deref(), Some("A failed. B failed."));
        let many: Vec<String> = (0..7).map(|index| format!("{index} failed.")).collect();
        let message = join_notices(&many).expect("seven notices report");
        assert!(message.contains("0 failed."), "{message}");
        assert!(message.ends_with("(and 4 more.)"), "{message}");
        assert!(!message.contains("6 failed."), "{message}");
    }

    #[test]
    fn a_file_is_reported_under_its_last_component() {
        assert_eq!(
            drop_name(std::path::Path::new("/data/tokyo.gpkg")),
            "tokyo.gpkg"
        );
        assert_eq!(drop_name(std::path::Path::new("tokyo.gpkg")), "tokyo.gpkg");
    }

    /// A file exactly the size of the cap is under it: the read reaches one
    /// byte past the cap to tell "fits" from "was cut", and that extra byte
    /// must not turn an exact fit into a refusal.
    #[test]
    fn a_dataset_exactly_the_size_of_the_cap_is_read() {
        let path = scratch("exact");
        std::fs::write(&path, vec![b'x'; 4096]).expect("the fixture is writable");
        let read = read_capped(&path, "cities.geojson", 4096).expect("an exact fit is not over");
        assert_eq!(read.len(), 4096);
        let _removed = std::fs::remove_file(&path);
    }

    /// Metadata is advisory, so it cannot be the only bound: a stream the OS
    /// reports as zero-length (a device, a pipe, a file that grew since the
    /// stat) must be refused rather than silently cut to the cap and handed to
    /// a parser as a truncated document.
    #[test]
    #[cfg(unix)]
    fn a_stream_whose_length_the_os_will_not_answer_for_is_still_bounded() {
        let zero = std::path::Path::new("/dev/zero");
        // A sandbox without the device has nothing to prove here.
        let Ok(metadata) = std::fs::metadata(zero) else {
            return;
        };
        assert_eq!(
            metadata.len(),
            0,
            "the premise: an endless stream that stats as empty",
        );
        let refusal = read_capped(zero, "cities.geojson", 4096).expect_err("it never ends");
        assert!(refusal.contains("cities.geojson"), "{refusal}");
        assert!(refusal.contains("at most"), "{refusal}");
    }
}
