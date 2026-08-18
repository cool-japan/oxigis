// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Asking the user for a path: the native dialog where the platform has a
//! Pure-Rust one, and an in-app prompt where it does not.
//!
//! # Why two answers
//!
//! `rfd` is on a `cfg(any(windows, target_os = "macos"))` edge (see this
//! crate's manifest for the audit trail): on Windows it resolves to raw-dylib
//! `windows-sys` bindings and on macOS to the `objc2` family, both Pure Rust
//! with no C in the graph. On Linux its GTK backend would pull `gtk-sys` /
//! `glib-sys` in and its portal backend an async runtime, so neither is
//! available under the Pure Rust policy.
//!
//! The answer there is [`PathPrompt`]: one small egui window that takes a path,
//! shows the absolute form it resolves to, and refuses the obvious mistakes
//! (an empty box, a directory that does not exist, a file to open that does
//! not). It is used for all four asks, so a Linux user has a working
//! File ▸ Open / Save As, tile-archive Open and PDF export rather than a
//! button that cannot succeed.
//!
//! Both halves answer the *same* `Choice` / [`PromptOutcome`] shape, so the
//! shell's call sites do not branch on the platform — they branch on whether a
//! path came back.

use std::path::{Path, PathBuf};

/// What a path is being asked for.
///
/// One enum for every ask so the prompt, its wording, its file-name filters
/// and the shell's dispatch after a path arrives all key off the same value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ask {
    /// File ▸ Open… — an existing `*.oxigis.json` project.
    OpenProject,
    /// File ▸ Save As… — where to write the project.
    SaveProject,
    /// The layer panel's "Open…" — an existing `.pmtiles` / `.mbtiles`.
    OpenArchive,
    /// File ▸ Export PDF… — where to write the page.
    SavePdf,
    /// Layer ▸ Export ▸ GeoJSON…, and Processing ▸ Output ▸ Save to file… —
    /// where to write a feature collection.
    ///
    /// One variant for both because the file they write is the same kind of
    /// file; what differs is only which seam produced it, which the shell
    /// already knows from the request it is holding.
    SaveGeoJson,
    /// The attribute table's Export ▸ CSV… — where to write the rows.
    SaveCsv,
}

impl Ask {
    /// Whether the path names a file that must already exist.
    pub(crate) fn is_open(self) -> bool {
        matches!(self, Self::OpenProject | Self::OpenArchive)
    }

    /// The dialog / prompt title.
    fn title(self) -> &'static str {
        match self {
            Self::OpenProject => "Open project",
            Self::SaveProject => "Save project as",
            Self::OpenArchive => "Open tile archive",
            Self::SavePdf => "Export PDF to",
            Self::SaveGeoJson => "Export GeoJSON to",
            Self::SaveCsv => "Export CSV to",
        }
    }

    /// The confirming button's label.
    fn verb(self) -> &'static str {
        if self.is_open() { "Open" } else { "Save" }
    }

    /// The file-type filter's label and extensions.
    fn filter(self) -> (&'static str, &'static [&'static str]) {
        match self {
            Self::OpenProject | Self::SaveProject => ("OxiGIS project", &["json"]),
            Self::OpenArchive => ("Tile archives", &["pmtiles", "mbtiles"]),
            Self::SavePdf => ("PDF document", &["pdf"]),
            Self::SaveGeoJson => ("GeoJSON", &["geojson", "json"]),
            Self::SaveCsv => ("Comma-separated values", &["csv"]),
        }
    }

    /// What the in-app prompt says the box wants.
    fn hint(self) -> &'static str {
        match self {
            Self::OpenProject => "The .oxigis.json project to open",
            Self::SaveProject => "Where to write the project (.oxigis.json)",
            Self::OpenArchive => "The .pmtiles or .mbtiles archive to open",
            Self::SavePdf => "Where to write the exported page (.pdf)",
            Self::SaveGeoJson => "Where to write the features (.geojson)",
            Self::SaveCsv => "Where to write the rows (.csv)",
        }
    }
}

/// Whether this build has a native file dialog at all.
///
/// A `const` the caller branches on, rather than an "unavailable" variant on
/// [`ask_native`]'s return type: an enum variant that only one platform ever
/// constructs is dead code on the others, and a `const false` keeps the
/// fallback branch honestly compiled everywhere instead.
pub(crate) const NATIVE_DIALOGS: bool = cfg!(any(windows, target_os = "macos"));

/// Runs the platform's own file dialog, blocking the UI thread — which is
/// exactly how a modal file dialog behaves everywhere else.
///
/// [`None`] means the user dismissed it. Only call this when
/// [`NATIVE_DIALOGS`]; where there is none, [`PathPrompt`] is the ask.
#[cfg(any(windows, target_os = "macos"))]
pub(crate) fn ask_native(
    ask: Ask,
    suggested_name: &str,
    directory: Option<&Path>,
) -> Option<PathBuf> {
    let (label, extensions) = ask.filter();
    let mut dialog = rfd::FileDialog::new()
        .set_title(ask.title())
        .add_filter(label, extensions);
    // A directory that has since been deleted would make the dialog open
    // nowhere in particular; the platform's own default is better than that.
    if let Some(directory) = directory.filter(|dir| dir.is_dir()) {
        dialog = dialog.set_directory(directory);
    }
    if ask.is_open() {
        dialog.pick_file()
    } else {
        dialog.set_file_name(suggested_name).save_file()
    }
}

/// No native dialog on this platform — see the module docs and [`PathPrompt`],
/// which is what the shell asks with instead. Never called: its one call site
/// is behind `if NATIVE_DIALOGS`, which is `false` here.
#[cfg(not(any(windows, target_os = "macos")))]
pub(crate) fn ask_native(
    _ask: Ask,
    _suggested_name: &str,
    _directory: Option<&Path>,
) -> Option<PathBuf> {
    None
}

/// The in-app path prompt: one egui window standing in for a platform dialog.
///
/// Held across frames by the shell (it is a modal), and drawn by
/// [`Self::show`], which answers [`PromptOutcome::Pending`] until the user
/// commits or cancels.
pub(crate) struct PathPrompt {
    /// What this prompt is for — decides the wording and what the shell does
    /// with the path.
    ask: Ask,
    /// The text box's contents.
    buffer: String,
    /// Where a relative path is resolved from, and what the box was seeded
    /// with: the last-used directory, else the working directory.
    base: PathBuf,
    /// The most recent refusal, shown under the box.
    error: Option<String>,
}

/// What one frame of [`PathPrompt::show`] produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PromptOutcome {
    /// Still on screen; ask again next frame.
    Pending,
    /// The user committed this (absolute) path.
    Accepted(PathBuf),
    /// The user dismissed the prompt.
    Cancelled,
}

impl PathPrompt {
    /// Opens a prompt for `ask`, seeded with `suggested_name` inside `base`.
    pub(crate) fn new(ask: Ask, suggested_name: &str, base: PathBuf) -> Self {
        let buffer = if ask.is_open() {
            String::new()
        } else {
            base.join(suggested_name).display().to_string()
        };
        Self {
            ask,
            buffer,
            base,
            error: None,
        }
    }

    /// What this prompt is asking for — the shell's dispatch key once a path
    /// arrives.
    pub(crate) fn ask(&self) -> Ask {
        self.ask
    }

    /// Reports a failure that happened *after* the path was accepted (the
    /// write itself), putting the prompt back on screen with the reason rather
    /// than dropping the user's typing on the floor.
    pub(crate) fn reopen_with_error(&mut self, error: impl Into<String>) {
        self.error = Some(error.into());
    }

    /// Draws one frame of the prompt.
    pub(crate) fn show(&mut self, ctx: &egui::Context) -> PromptOutcome {
        let mut outcome = PromptOutcome::Pending;
        egui::Window::new(self.ask.title())
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label(self.ask.hint());
                let entry = ui.add(
                    egui::TextEdit::singleline(&mut self.buffer)
                        .desired_width(480.0)
                        .hint_text(self.ask.hint()),
                );
                // The absolute form, so a relative path is never a mystery —
                // this is the "with the path shown" half of the fallback.
                match resolve(&self.buffer, &self.base) {
                    Some(resolved) => {
                        ui.weak(resolved.display().to_string());
                    }
                    None => {
                        ui.weak("\u{2014}");
                    }
                }
                if let Some(error) = &self.error {
                    ui.colored_label(egui::Color32::from_rgb(220, 80, 80), error);
                }
                ui.weak(
                    "This platform has no Pure-Rust system file dialog, so OxiGIS asks here \
                     instead. Drag-and-drop still works for data files.",
                );
                ui.separator();
                let submitted =
                    entry.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
                ui.horizontal(|ui| {
                    if ui.button(self.ask.verb()).clicked() || submitted {
                        match validate(&self.buffer, self.ask, &self.base) {
                            Ok(path) => outcome = PromptOutcome::Accepted(path),
                            Err(error) => self.error = Some(error),
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        outcome = PromptOutcome::Cancelled;
                    }
                });
            });
        // Escape dismisses, the way every other modal in this app does — but
        // never over a path the user committed in the same frame.
        if outcome == PromptOutcome::Pending
            && ctx.input(|input| input.key_pressed(egui::Key::Escape))
        {
            outcome = PromptOutcome::Cancelled;
        }
        outcome
    }
}

/// Expands a leading `~`, then resolves `input` against `base`.
///
/// [`None`] for an input that is blank once trimmed — there is no path to
/// show for an empty box, and callers say so rather than displaying `base`
/// as though it were the answer.
pub(crate) fn resolve(input: &str, base: &Path) -> Option<PathBuf> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let expanded = expand_home(trimmed);
    if expanded.is_absolute() {
        return Some(expanded);
    }
    Some(base.join(expanded))
}

/// `~` and `~/…` become `$HOME/…`; everything else is returned unchanged.
///
/// Only a LEADING bare `~` component is expanded — `~user` is a shell
/// convention this shell cannot resolve without a password database, and
/// `a/~/b` is a real (if odd) relative path.
fn expand_home(input: &str) -> PathBuf {
    let Some(rest) = input.strip_prefix('~') else {
        return PathBuf::from(input);
    };
    if !(rest.is_empty() || rest.starts_with('/') || rest.starts_with('\\')) {
        return PathBuf::from(input);
    }
    let Some(home) = home_dir() else {
        return PathBuf::from(input);
    };
    let rest = rest.trim_start_matches(['/', '\\']);
    if rest.is_empty() {
        home
    } else {
        home.join(rest)
    }
}

/// Turns what the user typed into a path to act on, or the reason it cannot
/// be one.
///
/// Deliberately a free function taking every input: the whole decision is
/// testable without an `egui::Context` or a window, which is what lets the
/// refusals below be pinned by unit tests.
pub(crate) fn validate(input: &str, ask: Ask, base: &Path) -> Result<PathBuf, String> {
    let Some(path) = resolve(input, base) else {
        return Err("Type a file path first.".to_string());
    };
    if ask.is_open() {
        if !path.exists() {
            return Err(format!("{} does not exist.", path.display()));
        }
        if path.is_dir() {
            return Err(format!("{} is a directory, not a file.", path.display()));
        }
        return Ok(path);
    }
    if path.is_dir() {
        return Err(format!(
            "{} is a directory — give the file a name too.",
            path.display()
        ));
    }
    match path.parent() {
        // A bare file name resolved against `base` always has a parent, so
        // `None` here means the path is a filesystem root.
        None => Err(format!("{} is not a file to write.", path.display())),
        Some(parent) if parent.as_os_str().is_empty() || parent.is_dir() => Ok(path),
        Some(parent) => Err(format!("The folder {} does not exist.", parent.display())),
    }
}

/// The user's home directory, from the environment only.
///
/// Hand-rolled rather than a `dirs`-style dependency: two environment
/// variables is the whole of it, and the shipped graph stays as it is.
/// [`None`] when neither is set (a daemon-like environment), which every
/// caller treats as "no suggestion", never as an error.
pub(crate) fn home_dir() -> Option<PathBuf> {
    for key in ["HOME", "USERPROFILE"] {
        if let Some(value) = std::env::var_os(key)
            && !value.is_empty()
        {
            return Some(PathBuf::from(value));
        }
    }
    None
}

/// Where a dialog opens when nothing better is known: the last directory the
/// user actually used, else the working directory, else the home directory.
///
/// Never fails — a path is always produced, because a dialog with no starting
/// directory is a worse answer than one starting somewhere plausible.
pub(crate) fn default_directory(last_used: Option<&Path>) -> PathBuf {
    if let Some(directory) = last_used.filter(|dir| dir.is_dir()) {
        return directory.to_path_buf();
    }
    if let Ok(directory) = std::env::current_dir() {
        return directory;
    }
    home_dir().unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory under the OS temp dir that exists for the length of a test.
    fn scratch_dir(label: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let path = std::env::temp_dir().join(format!("oxigis-dialog-{label}-{stamp}"));
        std::fs::create_dir_all(&path).expect("the scratch directory is creatable");
        path
    }

    #[test]
    fn a_relative_path_resolves_against_the_base_and_an_absolute_one_does_not() {
        let base = PathBuf::from("/data/projects");
        assert_eq!(
            resolve("city.oxigis.json", &base),
            Some(PathBuf::from("/data/projects/city.oxigis.json")),
        );
        let absolute = if cfg!(windows) {
            r"C:\x.json"
        } else {
            "/x.json"
        };
        assert_eq!(resolve(absolute, &base), Some(PathBuf::from(absolute)));
        assert_eq!(resolve("   ", &base), None, "an empty box has no path");
    }

    #[test]
    fn a_leading_tilde_expands_and_a_bare_one_is_the_home_directory() {
        let Some(home) = home_dir() else {
            return; // No HOME in this environment; nothing to prove.
        };
        let base = PathBuf::from("/elsewhere");
        assert_eq!(resolve("~", &base), Some(home.clone()));
        assert_eq!(
            resolve("~/maps/a.json", &base),
            Some(home.join("maps/a.json"))
        );
        // `~user` is a shell convention, not a path this shell can resolve.
        assert_eq!(
            resolve("~someone/a.json", &base),
            Some(base.join("~someone/a.json")),
        );
    }

    #[test]
    fn opening_refuses_what_is_not_there_and_saving_refuses_a_missing_folder() {
        let dir = scratch_dir("validate");
        let file = dir.join("cities.oxigis.json");
        std::fs::write(&file, b"{}").expect("the fixture is writable");

        assert_eq!(
            validate("cities.oxigis.json", Ask::OpenProject, &dir),
            Ok(file.clone()),
        );
        let missing = validate("absent.oxigis.json", Ask::OpenProject, &dir)
            .expect_err("an absent file cannot be opened");
        assert!(missing.contains("does not exist"), "{missing}");
        let is_dir =
            validate(".", Ask::OpenProject, &dir).expect_err("a directory is not a project");
        assert!(is_dir.contains("directory"), "{is_dir}");

        // Saving: the folder has to exist, the file does not.
        assert_eq!(
            validate("fresh.oxigis.json", Ask::SaveProject, &dir),
            Ok(dir.join("fresh.oxigis.json")),
        );
        let no_folder = validate("nope/fresh.oxigis.json", Ask::SaveProject, &dir)
            .expect_err("the folder is not there");
        assert!(no_folder.contains("does not exist"), "{no_folder}");
        let blank = validate("  ", Ask::SavePdf, &dir).expect_err("an empty box is refused");
        assert!(blank.contains("file path"), "{blank}");
        let onto_dir =
            validate(".", Ask::SaveProject, &dir).expect_err("a directory is not a file to write");
        assert!(onto_dir.contains("directory"), "{onto_dir}");

        let _removed = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_save_prompt_is_seeded_with_the_suggested_name_and_an_open_prompt_is_blank() {
        let base = PathBuf::from("/data");
        let save = PathPrompt::new(Ask::SaveProject, "city.oxigis.json", base.clone());
        assert_eq!(save.ask(), Ask::SaveProject);
        assert_eq!(
            resolve(&save.buffer, &base),
            Some(base.join("city.oxigis.json")),
        );
        let open = PathPrompt::new(Ask::OpenProject, "ignored", base.clone());
        assert!(
            open.buffer.is_empty(),
            "there is nothing to suggest opening"
        );
    }

    #[test]
    fn the_default_directory_falls_back_rather_than_failing() {
        let dir = scratch_dir("default");
        assert_eq!(default_directory(Some(&dir)), dir);
        let gone = dir.join("removed");
        assert_ne!(
            default_directory(Some(&gone)),
            gone,
            "a directory that is not there is not where a dialog should open",
        );
        assert!(!default_directory(None).as_os_str().is_empty());
        let _removed = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_ask_names_itself_and_filters_for_its_own_extensions() {
        for ask in [
            Ask::OpenProject,
            Ask::SaveProject,
            Ask::OpenArchive,
            Ask::SavePdf,
        ] {
            assert!(!ask.title().is_empty());
            assert!(!ask.hint().is_empty());
            let (label, extensions) = ask.filter();
            assert!(!label.is_empty());
            assert!(!extensions.is_empty(), "{label} filters nothing");
            assert!(
                extensions.iter().all(|ext| !ext.starts_with('.')),
                "rfd wants bare extensions",
            );
        }
        assert_eq!(Ask::OpenArchive.verb(), "Open");
        assert_eq!(Ask::SavePdf.verb(), "Save");
        assert!(Ask::OpenProject.is_open());
        assert!(!Ask::SaveProject.is_open());
    }
}
