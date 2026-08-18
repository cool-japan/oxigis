// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Reading and writing `*.oxigis.json` project files.
//!
//! `oxigis-ui` serializes and parses the document — this module only moves the
//! bytes, and does the two things a GIS must not get wrong about a user's
//! project file:
//!
//! * **A failed write never damages what was already there.** The bytes go to
//!   a temp file in the *same directory* (so the rename is within one
//!   filesystem and is therefore atomic), are flushed to the OS, and only then
//!   replace the destination. A full disk, a permission error or a crash
//!   mid-write leaves the previous save byte-for-byte intact — the failure
//!   mode of a plain `File::create` + `write_all`, which truncates first, is
//!   the whole reason this is not one line.
//! * **A read is bounded.** A project can carry inline GeoJSON, so it is not
//!   automatically small; it is still read on the thread that paints, so it is
//!   capped exactly as a dropped dataset is.

use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The extension every project this shell writes carries.
pub(crate) const PROJECT_EXTENSION: &str = "oxigis.json";

/// Largest project document this shell reads.
///
/// Smaller than a dataset's cap ([`crate::dataset_read::read_capped`]'s
/// callers use 512 MB) because a project is *parsed into a `serde_json::Value`
/// and then into a `Project`* on the UI thread, which costs several times the
/// document in peak memory. A project past this is one whose layers should be
/// path references rather than inline GeoJSON.
pub(crate) const MAX_PROJECT_BYTES: u64 = 128 * 1024 * 1024;

/// Longest file stem derived from a project name, in bytes.
///
/// Well under every filesystem's 255-byte component limit, with room for the
/// `.oxigis.json` suffix and a `~`-style backup suffix a user may add.
const MAX_STEM_BYTES: usize = 96;

/// Reads a project document, refusing anything past [`MAX_PROJECT_BYTES`] or
/// anything that is not UTF-8.
///
/// The error is what the user sees, so it names the file.
pub(crate) fn read_project_text(path: &Path) -> Result<String, String> {
    let name = path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into(),
    );
    let bytes = crate::dataset_read::read_capped(path, &name, MAX_PROJECT_BYTES)?;
    String::from_utf8(bytes)
        .map_err(|_| format!("{name} is not a UTF-8 text document, so it is not a project file."))
}

/// Writes `content` to `path` without ever leaving a half-written project
/// behind — see the module docs.
pub(crate) fn write_project_atomically(path: &Path, content: &str) -> Result<(), String> {
    write_atomically(path, content.as_bytes())
}

/// The shared temp-file-then-rename write, used for the project document and
/// for the session file (`crate::session`) — anything this shell must not be
/// able to truncate by failing halfway.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let Some(directory) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Err(format!("{} has no folder to write into.", path.display()));
    };
    let temporary = directory.join(temp_file_name(path));
    let outcome = write_all_synced(&temporary, bytes).and_then(|()| {
        std::fs::rename(&temporary, path)
            .map_err(|error| format!("could not replace {}: {error}", path.display()))
    });
    if outcome.is_err() {
        // The destination is untouched; the scratch file must not be left
        // beside it. A removal that itself fails changes nothing about the
        // error being reported.
        let _removed = std::fs::remove_file(&temporary);
    }
    outcome
}

/// Writes `bytes` to `path` and asks the OS to flush them before returning,
/// so the rename that follows publishes a complete file.
fn write_all_synced(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = std::fs::File::create(path)
        .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("could not flush {}: {error}", path.display()))?;
    Ok(())
}

/// The scratch name the bytes land under before the rename.
///
/// Leading dot (hidden on Unix, and sorted away elsewhere), the destination's
/// own name so a directory listing explains itself, then the process id and a
/// per-process counter: the pid separates two OxiGIS windows saving the same
/// project, the counter separates two saves within one window. A clock stamp
/// would do neither reliably — `SystemTime` has coarser resolution than a
/// save takes on some platforms, and it can go backwards.
fn temp_file_name(destination: &Path) -> String {
    static SAVE_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let stem = destination
        .file_name()
        .map_or_else(|| "project".to_string(), |n| n.to_string_lossy().into());
    let sequence = SAVE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!(".{stem}.oxigis-save-{}-{sequence}", std::process::id())
}

/// Ensures a chosen path ends in something JSON-shaped.
///
/// A name that already ends `.json` (including `.oxigis.json`, and
/// case-insensitively) is left exactly as the user typed it — overriding an
/// explicit choice is worse than an unusual extension. Anything else gains
/// `.oxigis.json`, which is what the native Save dialogs suggest and what
/// File ▸ Open filters for.
///
/// Note that `.geojson` is *not* in the first set: the dot is in the wrong
/// place, and a project saved under that name should gain the project
/// extension rather than pass itself off as a GeoJSON document.
pub(crate) fn with_project_extension(path: &Path) -> PathBuf {
    let Some(name) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        return path.to_path_buf();
    };
    if name.to_ascii_lowercase().ends_with(".json") {
        return path.to_path_buf();
    }
    path.with_file_name(format!("{name}.{PROJECT_EXTENSION}"))
}

/// Ensures a chosen PDF path ends in `.pdf`.
///
/// The native save dialogs append the filtered extension themselves, but the
/// in-app prompt hands back exactly what was typed — and a user who edits the
/// seeded name down to `map` means `map.pdf`, not an extensionless file no
/// viewer will open by double-click.
pub(crate) fn with_pdf_extension(path: &Path) -> PathBuf {
    let Some(name) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        return path.to_path_buf();
    };
    if name.to_ascii_lowercase().ends_with(".pdf") {
        return path.to_path_buf();
    }
    path.with_file_name(format!("{name}.pdf"))
}

/// The file name File ▸ Save As suggests for a project called `project_name`.
///
/// Every character a filesystem or a shell could take badly is replaced rather
/// than dropped, so two differently-named projects cannot collapse onto one
/// suggestion; a name that sanitizes to nothing falls back to `project`.
pub(crate) fn suggested_file_name(project_name: &str) -> String {
    let mut stem = String::new();
    let mut pending_separator = false;
    for character in project_name.chars() {
        let safe = match character {
            // Path separators, the Windows-reserved set, and anything a
            // terminal or a filesystem treats as structure.
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '\0' => '-',
            c if c.is_control() => '-',
            c if c.is_whitespace() => '-',
            c => c,
        };
        if safe == '-' {
            // Runs of replaced characters collapse, so "A / B" is `A-B` and
            // not `A---B`; a leading run is dropped entirely.
            pending_separator = !stem.is_empty();
            continue;
        }
        // The separator counts against the budget too, so the stem can never
        // overrun it by the one byte a trailing `-` would otherwise add.
        let width = character.len_utf8() + usize::from(pending_separator);
        if stem.len() + width > MAX_STEM_BYTES {
            break;
        }
        if pending_separator {
            stem.push('-');
            pending_separator = false;
        }
        stem.push(safe);
    }
    // A name of only dots is a directory entry on every platform, and a
    // trailing dot is silently dropped by Windows.
    let stem = stem.trim_matches('.');
    if stem.is_empty() {
        return format!("project.{PROJECT_EXTENSION}");
    }
    format!("{stem}.{PROJECT_EXTENSION}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory under the OS temp dir, unique per call.
    fn scratch_dir(label: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let path = std::env::temp_dir().join(format!("oxigis-project-{label}-{stamp}"));
        std::fs::create_dir_all(&path).expect("the scratch directory is creatable");
        path
    }

    #[test]
    fn a_project_round_trips_through_the_atomic_write() {
        let dir = scratch_dir("roundtrip");
        let path = dir.join("city.oxigis.json");
        write_project_atomically(&path, "{\"name\":\"city\"}").expect("a fresh write succeeds");
        assert_eq!(
            read_project_text(&path).expect("the file reads back"),
            "{\"name\":\"city\"}",
        );
        // Overwriting replaces rather than appends, and leaves no scratch file.
        write_project_atomically(&path, "{\"name\":\"city2\"}").expect("an overwrite succeeds");
        assert_eq!(
            read_project_text(&path).expect("the file reads back"),
            "{\"name\":\"city2\"}",
        );
        let leftovers: Vec<String> = std::fs::read_dir(&dir)
            .expect("the directory lists")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name != "city.oxigis.json")
            .collect();
        assert!(
            leftovers.is_empty(),
            "scratch files were left: {leftovers:?}"
        );
        let _removed = std::fs::remove_dir_all(&dir);
    }

    /// The whole point of the temp-and-rename: a write that cannot succeed
    /// must not be able to destroy the previous save.
    #[test]
    fn a_failed_write_leaves_the_previous_save_intact() {
        let dir = scratch_dir("failed");
        let path = dir.join("city.oxigis.json");
        write_project_atomically(&path, "{\"keep\":true}").expect("the first write succeeds");

        // A destination that is a DIRECTORY: `File::create` on the scratch
        // name succeeds, the rename over a non-empty directory does not.
        let occupied = dir.join("occupied.oxigis.json");
        std::fs::create_dir(&occupied).expect("the directory is creatable");
        std::fs::write(occupied.join("inside"), b"x").expect("the child is writable");
        let error = write_project_atomically(&occupied, "{\"new\":true}")
            .expect_err("a non-empty directory cannot be replaced by a file");
        assert!(error.contains("occupied.oxigis.json"), "{error}");
        assert!(
            occupied.join("inside").is_file(),
            "the existing entry survived the failure",
        );
        // And the unrelated project is untouched.
        assert_eq!(
            read_project_text(&path).expect("the file reads back"),
            "{\"keep\":true}",
        );
        // No scratch file was left behind by the failure.
        let scratch: Vec<String> = std::fs::read_dir(&dir)
            .expect("the directory lists")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains("oxigis-save-"))
            .collect();
        assert!(scratch.is_empty(), "a failed write left {scratch:?}");
        let _removed = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_read_is_capped_and_a_non_utf8_document_is_refused() {
        let dir = scratch_dir("bounds");
        let big = dir.join("big.oxigis.json");
        std::fs::write(&big, vec![b'x'; 512]).expect("the fixture is writable");
        let refusal = crate::dataset_read::read_capped(&big, "big.oxigis.json", 128)
            .expect_err("512 bytes do not fit a 128-byte cap");
        assert!(refusal.contains("big.oxigis.json"), "{refusal}");

        let binary = dir.join("binary.oxigis.json");
        std::fs::write(&binary, [0xff_u8, 0xfe, 0xfd]).expect("the fixture is writable");
        let error = read_project_text(&binary).expect_err("invalid UTF-8 is not a project");
        assert!(error.contains("UTF-8"), "{error}");
        let _removed = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_extension_is_added_only_when_one_is_not_already_there() {
        assert_eq!(
            with_project_extension(Path::new("/data/city")),
            PathBuf::from("/data/city.oxigis.json"),
        );
        assert_eq!(
            with_project_extension(Path::new("/data/city.oxigis.json")),
            PathBuf::from("/data/city.oxigis.json"),
        );
        // An explicit `.json` is the user's choice and is left alone.
        assert_eq!(
            with_project_extension(Path::new("/data/city.JSON")),
            PathBuf::from("/data/city.JSON"),
        );
        // `.geojson` does NOT end in `.json` (the dot is in the wrong place),
        // and it should not: a project is not a GeoJSON document, so saving one
        // under that name gains the project extension rather than pretending.
        assert_eq!(
            with_project_extension(Path::new("/data/city.geojson")),
            PathBuf::from("/data/city.geojson.oxigis.json"),
        );
        // A version-numbered name is not an extension either.
        assert_eq!(
            with_project_extension(Path::new("/data/tokyo.v2")),
            PathBuf::from("/data/tokyo.v2.oxigis.json"),
        );
        // A path with no file name at all is left exactly as it is, rather
        // than growing a stray `.oxigis.json` component.
        assert_eq!(with_project_extension(Path::new("/")), PathBuf::from("/"),);
    }

    #[test]
    fn a_suggested_name_is_a_usable_file_name_on_every_platform() {
        assert_eq!(
            suggested_file_name("Tokyo wards"),
            "Tokyo-wards.oxigis.json"
        );
        assert_eq!(
            suggested_file_name("a/b:c*d?e\"f<g>h|i"),
            "a-b-c-d-e-f-g-h-i.oxigis.json",
            "every reserved character is replaced, and runs collapse",
        );
        assert_eq!(suggested_file_name("   "), "project.oxigis.json");
        assert_eq!(suggested_file_name(".."), "project.oxigis.json");
        assert_eq!(suggested_file_name(""), "project.oxigis.json");
        // Non-Latin names survive intact — this is a Japanese-first project.
        assert_eq!(suggested_file_name("東京都"), "東京都.oxigis.json");
        // And a hostile name is bounded.
        let long = suggested_file_name(&"x".repeat(4096));
        assert_eq!(long.len(), MAX_STEM_BYTES + PROJECT_EXTENSION.len() + 1);
        // Multi-byte characters are never cut through the middle.
        let wide = suggested_file_name(&"東".repeat(4096));
        assert!(wide.starts_with('東'), "{wide}");
        assert!(wide.ends_with(PROJECT_EXTENSION), "{wide}");
    }

    #[test]
    fn a_scratch_name_is_hidden_and_names_its_destination() {
        let name = temp_file_name(Path::new("/data/city.oxigis.json"));
        assert!(name.starts_with(".city.oxigis.json"), "{name}");
        assert!(name.contains("oxigis-save-"), "{name}");
        assert_ne!(
            name,
            temp_file_name(Path::new("/data/city.oxigis.json")),
            "two concurrent saves must not collide",
        );
    }
}
