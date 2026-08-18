//! The PDF export's report channel: how a browser task tells the running app
//! what it is doing.
//!
//! An export is spawned with `spawn_local` because its tile fetches only
//! progress while the task yields to the event loop. That future has no
//! `&mut OxigisApp`, so every outcome it produced used to be a `log::` call —
//! success, the download failure, the export failure and the tens of seconds
//! in between. In a browser with no console open (and, before the tracing
//! bridge, with `oxigis-ui`'s own diagnostics discarded outright) the honest
//! description of File ▸ Export PDF was: click, wait, nothing happens.
//!
//! So the task writes here and the shell drains it once per frame into
//! [`oxigis_ui::OxigisApp::set_status`] — the same status line every other
//! shell-only failure already reports through. The slot is a `thread_local!`
//! for the same reason [`crate::font_fetch`]'s is: wasm is single-threaded, and
//! the writer has no handle on the app.
//!
//! Every write asks for a repaint, which is what makes progress *visible*: an
//! idle map schedules no frames, so without it the status would sit here until
//! the user happened to pan.

use std::cell::{Cell, RefCell};

thread_local! {
    /// The latest thing the export has to say, drained per frame.
    static MESSAGE: RefCell<Option<String>> = const { RefCell::new(None) };

    /// Whether an export task is in flight, so a second click does not start a
    /// duplicate one — two exports means two providers fetching the same tiles.
    static RUNNING: Cell<bool> = const { Cell::new(false) };
}

/// Claims the export slot for a new task.
///
/// Returns `false` when one is already running, in which case the caller must
/// not start a second: the click is answered with a status line instead.
#[must_use]
pub fn begin() -> bool {
    if RUNNING.get() {
        report("An export is already running; it will finish or report why it could not.");
        return false;
    }
    RUNNING.set(true);
    true
}

/// Releases the export slot. Call exactly once per successful [`begin`],
/// after the final [`report`].
pub fn finish() {
    RUNNING.set(false);
}

/// Whether an export task is in flight.
#[must_use]
pub fn running() -> bool {
    RUNNING.get()
}

/// Records what the export is doing, and asks for a frame to show it in.
pub fn report(message: impl Into<String>) {
    MESSAGE.with_borrow_mut(|slot| *slot = Some(message.into()));
    super::font_fetch::request_repaint();
}

/// Takes the latest message, if the export has said anything since the last
/// frame. Called once per frame by the shell.
#[must_use]
pub fn take_message() -> Option<String> {
    MESSAGE.with_borrow_mut(Option::take)
}

/// The progress line for a phase that is fetching `done` of `total` tiles.
///
/// Formatted here rather than at each call site so the two phases read
/// identically in the status bar.
#[must_use]
pub fn tile_progress(phase: &str, done: usize, total: usize) -> String {
    format!("Exporting PDF: {phase} {done}/{total} tiles\u{2026}")
}
