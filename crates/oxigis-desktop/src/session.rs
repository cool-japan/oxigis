// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! What OxiGIS remembers between launches: window geometry, the recent-project
//! list, and the directory the file dialogs open in.
//!
//! # Why not `eframe`'s storage
//!
//! `eframe` is built here with `features = ["wgpu"]` and no `persistence`, so
//! `App::save` is never called and `Storage` does not exist. Turning the
//! feature on is a workspace-manifest change that also drags `ron` and
//! `serde` into the shipped binary for four values — so this module writes
//! those four values itself, in a line-oriented text file, with no new
//! dependency at all. `App::on_exit` (which is NOT persistence-gated) is where
//! the shell calls [`store`].
//!
//! # Format
//!
//! One `key<TAB>value` record per line; `#` starts a comment; a line that is
//! not understood is SKIPPED rather than failing the load, so an older or
//! newer OxiGIS reading this file degrades to defaults for the parts it does
//! not know instead of losing the parts it does. `\\`, `\t`, `\n` and `\r`
//! are escaped in values, because a path may legally contain any of them on
//! Unix.
//!
//! The separator shown as a run of spaces below is a literal TAB in the file:
//!
//! ```text
//! oxigis-session   1
//! window           120 80 1440 900
//! maximized        true
//! last-directory   /Users/me/maps
//! recent           /Users/me/maps/tokyo.oxigis.json
//! ```
//!
//! A missing, unreadable, truncated or nonsense file is not an error: the
//! session simply starts at its defaults. Losing a window position must never
//! be able to stop the app from opening.

use std::path::{Path, PathBuf};

/// Format version written into the header line.
const FORMAT_VERSION: u32 = 1;

/// Longest session file this shell will read.
///
/// The file it writes is a few hundred bytes; the cap exists so a path that
/// happens to point at something enormous cannot be read into memory at
/// startup.
const MAX_SESSION_BYTES: u64 = 64 * 1024;

/// How many projects the recent list keeps.
pub(crate) const MAX_RECENT: usize = 10;

/// Smallest window this shell will restore to, in points.
///
/// A saved geometry can be nonsense (a crashed session, an edited file, a
/// display that changed underneath) and a 2×2 window is unrecoverable without
/// deleting the file by hand.
const MIN_WINDOW_SIZE: f32 = 480.0;

/// Largest window this shell will restore to, in points — far past any real
/// display wall, but finite.
const MAX_WINDOW_SIZE: f32 = 32_768.0;

/// How far off the origin a restored window may sit, in points.
const MAX_WINDOW_OFFSET: f32 = 32_768.0;

/// The window size a first launch opens at.
pub(crate) const DEFAULT_WINDOW_SIZE: [f32; 2] = [1280.0, 800.0];

/// A remembered window rectangle, in egui points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WindowGeometry {
    /// Left edge, in monitor space.
    pub(crate) x: f32,
    /// Top edge, in monitor space.
    pub(crate) y: f32,
    /// Width in points.
    pub(crate) width: f32,
    /// Height in points.
    pub(crate) height: f32,
}

impl WindowGeometry {
    /// The geometry with every field checked, or [`None`] when it is not a
    /// rectangle a window can be opened at.
    ///
    /// Clamps rather than rejects where clamping is meaningful (a window
    /// slightly off the left edge is dragged back on), and rejects outright
    /// what cannot be repaired: NaN, infinities, and a size at or below zero.
    pub(crate) fn sanitized(self) -> Option<Self> {
        if ![self.x, self.y, self.width, self.height]
            .iter()
            .all(|value| value.is_finite())
        {
            return None;
        }
        if self.width <= 0.0 || self.height <= 0.0 {
            return None;
        }
        Some(Self {
            x: self.x.clamp(-MAX_WINDOW_OFFSET, MAX_WINDOW_OFFSET),
            y: self.y.clamp(-MAX_WINDOW_OFFSET, MAX_WINDOW_OFFSET),
            width: self.width.clamp(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE),
            height: self.height.clamp(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE),
        })
    }

    /// Whether this rectangle would put the window somewhere the user could
    /// still grab it, given the monitor the app is about to open on.
    ///
    /// A saved position from a monitor that has since been unplugged is the
    /// one geometry failure a user cannot recover from with the mouse, so a
    /// window whose title bar would land past the right or bottom edge — or
    /// far enough above the top to hide the bar — is dropped and the window
    /// manager places the window itself.
    pub(crate) fn is_reachable_on(self, monitor: [f32; 2]) -> bool {
        // A monitor size we were not told is not evidence against the saved
        // position.
        if !monitor.iter().all(|size| size.is_finite() && *size > 0.0) {
            return true;
        }
        // A strip of the title bar must remain inside the monitor.
        const GRAB_MARGIN: f32 = 64.0;
        self.x + GRAB_MARGIN <= monitor[0]
            && self.y + GRAB_MARGIN <= monitor[1]
            && self.x + self.width >= GRAB_MARGIN
            && self.y >= -GRAB_MARGIN
    }
}

/// Everything remembered between launches.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct SessionState {
    /// Where and how big the window was.
    pub(crate) window: Option<WindowGeometry>,
    /// Whether it was maximized (in which case the geometry above is the
    /// restore rectangle, which is what every platform's own windowing does).
    pub(crate) maximized: bool,
    /// Recently opened / saved projects, most recent first, capped at
    /// [`MAX_RECENT`].
    pub(crate) recent: Vec<PathBuf>,
    /// Where the file dialogs last landed.
    pub(crate) last_directory: Option<PathBuf>,
}

impl SessionState {
    /// Records `path` as the most recently used project: moved to the front if
    /// it is already listed, and the list stays capped.
    pub(crate) fn note_recent(&mut self, path: PathBuf) {
        self.last_directory = path.parent().map(Path::to_path_buf);
        self.recent.retain(|existing| existing != &path);
        self.recent.insert(0, path);
        self.recent.truncate(MAX_RECENT);
    }
}

/// The per-user configuration directory this shell writes into, created if it
/// is not there yet.
///
/// Hand-rolled from the environment rather than a `dirs`-style dependency —
/// three platform rules is the whole of it:
///
/// * Windows: `%APPDATA%\OxiGIS`
/// * macOS: `~/Library/Application Support/OxiGIS`
/// * everywhere else: `$XDG_CONFIG_HOME/oxigis`, else `~/.config/oxigis`
///
/// [`None`] when the environment names no home at all, which every caller
/// treats as "this session is not remembered" rather than as an error.
pub(crate) fn config_directory() -> Option<PathBuf> {
    let directory = if cfg!(windows) {
        let base = std::env::var_os("APPDATA")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| crate::file_dialog::home_dir().map(|home| home.join("AppData/Roaming")))?;
        base.join("OxiGIS")
    } else if cfg!(target_os = "macos") {
        crate::file_dialog::home_dir()?.join("Library/Application Support/OxiGIS")
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| crate::file_dialog::home_dir().map(|home| home.join(".config")))?
            .join("oxigis")
    };
    Some(directory)
}

/// The session file's full path, or [`None`] when there is nowhere to put it.
pub(crate) fn session_path() -> Option<PathBuf> {
    Some(config_directory()?.join("session.conf"))
}

/// Reads the remembered session, falling back to defaults for anything the
/// file does not say — including the file not existing at all.
pub(crate) fn load() -> SessionState {
    let Some(path) = session_path() else {
        return SessionState::default();
    };
    let Ok(bytes) = crate::dataset_read::read_capped(&path, "session.conf", MAX_SESSION_BYTES)
    else {
        return SessionState::default();
    };
    match core::str::from_utf8(&bytes) {
        Ok(text) => parse(text),
        Err(_) => SessionState::default(),
    }
}

/// Writes the session out, creating the configuration directory if needed.
///
/// The error is for the log, not for the user: a session that could not be
/// remembered is not something to interrupt a shutdown over.
pub(crate) fn store(state: &SessionState) -> Result<(), String> {
    let Some(path) = session_path() else {
        return Err("no configuration directory for this user".to_string());
    };
    if let Some(directory) = path.parent() {
        std::fs::create_dir_all(directory)
            .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    }
    crate::project_file::write_atomically(&path, render(state).as_bytes())
}

/// Renders a session to the on-disk text — see the module docs for the shape.
fn render(state: &SessionState) -> String {
    let mut out = format!("oxigis-session\t{FORMAT_VERSION}\n");
    if let Some(window) = state.window {
        out.push_str(&format!(
            "window\t{} {} {} {}\n",
            window.x, window.y, window.width, window.height
        ));
    }
    out.push_str(&format!("maximized\t{}\n", state.maximized));
    if let Some(directory) = &state.last_directory {
        out.push_str(&format!(
            "last-directory\t{}\n",
            escape(&directory.display().to_string())
        ));
    }
    for path in state.recent.iter().take(MAX_RECENT) {
        out.push_str(&format!(
            "recent\t{}\n",
            escape(&path.display().to_string())
        ));
    }
    out
}

/// Parses the on-disk text. Never fails: unknown keys, malformed records and
/// out-of-range numbers are skipped, and what is left is the session.
fn parse(text: &str) -> SessionState {
    let mut state = SessionState::default();
    for line in text.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('\t') else {
            continue;
        };
        match key {
            "window" => state.window = parse_geometry(value).and_then(WindowGeometry::sanitized),
            "maximized" => state.maximized = value == "true",
            "last-directory" => state.last_directory = Some(PathBuf::from(unescape(value))),
            // A hand-edited (or older, longer) file must not be able to grow
            // the menu without limit, and must not list one project twice.
            "recent" if state.recent.len() < MAX_RECENT => {
                let path = PathBuf::from(unescape(value));
                if !path.as_os_str().is_empty() && !state.recent.contains(&path) {
                    state.recent.push(path);
                }
            }
            // "oxigis-session" (the header) and anything a future version
            // writes: known-unknown, and deliberately not an error.
            _ => {}
        }
    }
    state
}

/// `x y width height`, all four or nothing.
fn parse_geometry(value: &str) -> Option<WindowGeometry> {
    let mut fields = [0.0_f32; 4];
    let mut parts = value.split_whitespace();
    for field in &mut fields {
        *field = parts.next()?.parse::<f32>().ok()?;
    }
    if parts.next().is_some() {
        return None;
    }
    let [x, y, width, height] = fields;
    Some(WindowGeometry {
        x,
        y,
        width,
        height,
    })
}

/// Makes a value safe to store on one line.
fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out
}

/// The inverse of [`escape`]. An unknown escape keeps both characters, so a
/// hand-edited file cannot silently lose a backslash out of a Windows path.
fn unescape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(character) = chars.next() {
        if character != '\\' {
            out.push(character);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> WindowGeometry {
        WindowGeometry {
            x: 120.0,
            y: 80.0,
            width: 1440.0,
            height: 900.0,
        }
    }

    #[test]
    fn a_session_survives_the_round_trip() {
        let state = SessionState {
            window: Some(geometry()),
            maximized: true,
            recent: vec![
                PathBuf::from("/data/tokyo.oxigis.json"),
                PathBuf::from("/data/osaka.oxigis.json"),
            ],
            last_directory: Some(PathBuf::from("/data")),
        };
        assert_eq!(parse(&render(&state)), state);
    }

    #[test]
    fn a_path_with_a_tab_or_a_newline_in_it_still_round_trips() {
        // Legal on Unix, and the reason the format escapes at all.
        let hostile = PathBuf::from("/data/od\td\nname\\here.oxigis.json");
        let state = SessionState {
            window: None,
            maximized: false,
            recent: vec![hostile.clone()],
            last_directory: None,
        };
        let text = render(&state);
        assert_eq!(text.lines().count(), 3, "one record per line: {text:?}");
        assert_eq!(parse(&text).recent, vec![hostile]);
    }

    #[test]
    fn a_corrupt_file_degrades_to_defaults_instead_of_failing() {
        assert_eq!(parse(""), SessionState::default());
        assert_eq!(parse("garbage without a tab"), SessionState::default());
        // A truncated write: the header landed, the rest did not.
        assert_eq!(parse("oxigis-session\t1\nwin"), SessionState::default());
        // Nonsense geometry is dropped, but the rest of the file is kept.
        let partial = parse("window\tNaN 0 100 100\nmaximized\ttrue\n");
        assert_eq!(partial.window, None);
        assert!(partial.maximized);
        assert_eq!(parse("window\t1 2 3\n").window, None, "four fields or none");
        assert_eq!(parse("window\t1 2 3 4 5\n").window, None);
        // A key from a future version is skipped, not fatal.
        let forward = parse("theme\tmidnight\nmaximized\ttrue\n# a comment\n");
        assert!(forward.maximized);
    }

    #[test]
    fn a_degenerate_geometry_is_repaired_or_refused() {
        assert_eq!(geometry().sanitized(), Some(geometry()));
        let tiny = WindowGeometry {
            width: 4.0,
            height: 4.0,
            ..geometry()
        };
        let repaired = tiny
            .sanitized()
            .expect("a small window is clamped, not dropped");
        assert_eq!(repaired.width, MIN_WINDOW_SIZE);
        assert_eq!(repaired.height, MIN_WINDOW_SIZE);
        let far = WindowGeometry {
            x: -1.0e9,
            ..geometry()
        };
        assert_eq!(
            far.sanitized().map(|g| g.x),
            Some(-MAX_WINDOW_OFFSET),
            "a position off in space is dragged back to a finite one",
        );
        for broken in [
            WindowGeometry {
                width: 0.0,
                ..geometry()
            },
            WindowGeometry {
                height: -10.0,
                ..geometry()
            },
            WindowGeometry {
                x: f32::NAN,
                ..geometry()
            },
            WindowGeometry {
                y: f32::INFINITY,
                ..geometry()
            },
        ] {
            assert_eq!(broken.sanitized(), None, "{broken:?}");
        }
    }

    /// The failure a user cannot fix with the mouse: a window restored onto a
    /// monitor that is no longer plugged in.
    #[test]
    fn a_window_off_every_monitor_is_not_restored_there() {
        let monitor = [1920.0, 1080.0];
        assert!(geometry().is_reachable_on(monitor));
        let off_right = WindowGeometry {
            x: 3400.0,
            ..geometry()
        };
        assert!(
            !off_right.is_reachable_on(monitor),
            "the second display is gone"
        );
        let off_bottom = WindowGeometry {
            y: 2000.0,
            ..geometry()
        };
        assert!(!off_bottom.is_reachable_on(monitor));
        let above = WindowGeometry {
            y: -400.0,
            ..geometry()
        };
        assert!(
            !above.is_reachable_on(monitor),
            "the title bar is off the top"
        );
        let mostly_left = WindowGeometry {
            x: -1300.0,
            ..geometry()
        };
        assert!(
            mostly_left.is_reachable_on(monitor),
            "part of the window is still on screen",
        );
        // An unknown monitor size is not evidence against the saved position.
        assert!(off_right.is_reachable_on([0.0, 0.0]));
        assert!(off_right.is_reachable_on([f32::NAN, f32::NAN]));
    }

    #[test]
    fn the_recent_list_is_most_recent_first_deduplicated_and_capped() {
        let mut state = SessionState::default();
        for index in 0..(MAX_RECENT + 5) {
            state.note_recent(PathBuf::from(format!("/data/p{index}.oxigis.json")));
        }
        assert_eq!(state.recent.len(), MAX_RECENT);
        assert_eq!(
            state.recent[0],
            PathBuf::from(format!("/data/p{}.oxigis.json", MAX_RECENT + 4)),
            "the newest is first",
        );
        assert_eq!(state.last_directory, Some(PathBuf::from("/data")));

        // Re-opening an already-listed project moves it up rather than
        // duplicating it.
        let again = state.recent[3].clone();
        state.note_recent(again.clone());
        assert_eq!(state.recent[0], again);
        assert_eq!(
            state.recent.iter().filter(|path| **path == again).count(),
            1,
        );
        assert_eq!(state.recent.len(), MAX_RECENT);
    }

    #[test]
    fn a_file_listing_more_recents_than_the_cap_is_still_bounded() {
        let mut text = String::new();
        for index in 0..(MAX_RECENT * 3) {
            text.push_str(&format!("recent\t/data/p{index}.oxigis.json\n"));
        }
        // And a duplicate in a hand-edited file is not kept twice.
        text.push_str("recent\t/data/p0.oxigis.json\n");
        assert_eq!(parse(&text).recent.len(), MAX_RECENT);
    }

    #[test]
    fn the_configuration_directory_is_under_this_platforms_own_root() {
        let Some(directory) = config_directory() else {
            return; // No HOME in this environment.
        };
        let shown = directory.display().to_string();
        if cfg!(target_os = "macos") {
            assert!(shown.contains("Library/Application Support"), "{shown}");
            assert!(shown.ends_with("OxiGIS"), "{shown}");
        } else if cfg!(windows) {
            assert!(shown.ends_with("OxiGIS"), "{shown}");
        } else {
            assert!(shown.ends_with("oxigis"), "{shown}");
        }
        let path = session_path().expect("a directory implies a file");
        assert!(path.ends_with("session.conf"), "{}", path.display());
    }

    #[test]
    fn escaping_is_reversible_and_an_unknown_escape_is_left_alone() {
        for value in ["", "plain", "C:\\maps\\a.json", "a\tb\nc\rd", "\\", "\\\\"] {
            assert_eq!(unescape(&escape(value)), value, "{value:?}");
        }
        assert_eq!(unescape("C:\\q"), "C:\\q", "an unknown escape keeps both");
        assert_eq!(unescape("trailing\\"), "trailing\\");
    }

    /// The whole load/store path against a real file, in a scratch HOME.
    #[test]
    fn a_stored_session_reads_back_and_a_missing_one_is_defaults() {
        let path = std::env::temp_dir().join(format!(
            "oxigis-session-{}-{}.conf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos()),
        ));
        let state = SessionState {
            window: Some(geometry()),
            maximized: false,
            recent: vec![PathBuf::from("/data/tokyo.oxigis.json")],
            last_directory: Some(PathBuf::from("/data")),
        };
        crate::project_file::write_atomically(&path, render(&state).as_bytes())
            .expect("the scratch session is writable");
        let bytes = crate::dataset_read::read_capped(&path, "session.conf", MAX_SESSION_BYTES)
            .expect("it reads back");
        let text = core::str::from_utf8(&bytes).expect("what was written was UTF-8");
        assert_eq!(parse(text), state);
        let _removed = std::fs::remove_file(&path);

        // A file that is not there is not an error.
        let absent = crate::dataset_read::read_capped(&path, "session.conf", MAX_SESSION_BYTES);
        assert!(absent.is_err(), "the fixture was removed");
        assert_eq!(parse(""), SessionState::default());
    }
}
