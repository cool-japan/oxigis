// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The editing toolbar: one fixed-height top panel, and the [`EditAction`]s it
//! reports.
//!
//! # Why a top panel is ratchet-safe
//!
//! egui 0.35 panels persist their *content* size, so one frame of overflowing
//! content widens a panel forever — the bug that shipped in `layer_panel`'s URL
//! rows and the reason every side panel in this app carries a `max_size`. A
//! **top** panel persists its *height*, and the height of a single non-wrapping
//! `ui.horizontal` of fixed-size buttons is a constant of the font. Nothing in
//! the row is a text field, and the one data-derived string it carries — the
//! `⚠ N` issue counter — is capped at [`ISSUE_BADGE_CAP`] characters by
//! [`issue_badge`], so there is no string that can grow. `resizable(false)` plus
//! [`EDIT_TOOLBAR_MAX_PT`] are the belt and braces on top of that.
//!
//! # Why every keyboard action also has a button
//!
//! On the web shell the browser and the canvas fight over modifiers and focus in
//! ways no amount of care fully fixes. A design in which the keyboard is the
//! only path to an action is therefore broken on the web — so the toolbar and
//! the `Edit` menu together reach everything the shortcuts reach.
//!
//! # Statelessness
//!
//! This module never sees [`crate::OxigisApp`]. It is handed a [`ToolbarState`]
//! snapshot and hands back a list of requests, which is what keeps the panel
//! testable and keeps its borrow requirements trivial at the call site.

use super::EditMode;
use crate::style_panel::StyleKind;
use crate::ui_glyphs::{REDO, UNDO};
use egui::{Panel, TextWrapMode, Ui};

/// Height cap of the toolbar panel, in egui points.
pub const EDIT_TOOLBAR_MAX_PT: f32 = 34.0;

/// What the user asked the editing system to do.
///
/// A request rather than a mutation: the panel has no access to the project, the
/// undo stack or the feature store, and routing everything through one enum
/// keeps the toolbar, the `Edit` menu and the keyboard on a single dispatch
/// path — so a shortcut and its button can never drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditAction {
    /// Switch to this tool.
    SetMode(EditMode),
    /// Create an empty local vector layer to digitize into, styled for `kind`.
    NewLayer(StyleKind),
    /// Undo the newest transaction.
    Undo,
    /// Redo the newest undone transaction.
    Redo,
    /// Turn snapping on or off.
    ToggleSnap,
    /// Delete the selected vertex.
    DeleteVertex,
    /// Delete the selected feature.
    DeleteFeature,
    /// Open or close the Edit window.
    ToggleWindow,
    /// Run the topology checks over the whole target layer.
    ValidateLayer,
    /// Open the Edit window — never close it — with its Validation section
    /// expanded. What the `⚠` issue badge asks for: with the window already
    /// open (the common case), a plain [`Self::ToggleWindow`] would close it
    /// instead of showing the list the badge advertises.
    ShowValidation,
}

/// Everything the toolbar needs to know about the app this frame.
#[derive(Debug, Clone, Copy)]
pub struct ToolbarState<'a> {
    /// The active tool.
    pub mode: EditMode,
    /// Whether the app selection is a local vector layer whose features are
    /// loaded — i.e. whether there is anywhere to draw into.
    pub can_draw: bool,
    /// Whether a feature is picked.
    pub has_selection: bool,
    /// Whether a *vertex* of that feature is picked, which is what `✖ Vertex`
    /// would delete.
    pub has_vertex: bool,
    /// Whether snapping is on.
    pub snap: bool,
    /// Whether the undo stack has a step to undo.
    pub can_undo: bool,
    /// Whether it has one to redo.
    pub can_redo: bool,
    /// Label of the step `↩` would undo, for its tooltip.
    pub undo_label: Option<&'a str>,
    /// Label of the step `↪` would redo, for its tooltip.
    pub redo_label: Option<&'a str>,
    /// Whether the Edit window is open.
    pub window_open: bool,
    /// How many topology issues the active layer has recorded. Zero hides the
    /// counter entirely — an editor that always shows a warning badge has taught
    /// the user to ignore it.
    pub issues: usize,
}

/// Glyph shown on each tool's button.
#[must_use]
pub fn mode_glyph(mode: EditMode) -> &'static str {
    match mode {
        EditMode::Off => "\u{25a3} Browse",
        EditMode::Select => "\u{25b6} Select",
        EditMode::DrawPoint => "\u{2022} Point",
        EditMode::DrawLine => "\u{2197} Line",
        EditMode::DrawPolygon => "\u{2b1f} Polygon",
    }
}

/// The layer style a "new edit layer" of each drawing tool wants.
///
/// Deliberately a total function over the three drawing modes rather than a
/// `StyleKind` picker in the UI: the user asks for "a polygon layer", and the
/// style that makes polygons visible is an implementation detail of that answer.
#[must_use]
pub fn style_for_mode(mode: EditMode) -> Option<StyleKind> {
    match mode {
        EditMode::DrawPoint => Some(StyleKind::Circle),
        EditMode::DrawLine => Some(StyleKind::Line),
        EditMode::DrawPolygon => Some(StyleKind::Fill),
        EditMode::Off | EditMode::Select => None,
    }
}

/// The drawing tool a "new edit layer" of `kind` should switch to.
#[must_use]
pub fn mode_for_style(kind: StyleKind) -> EditMode {
    match kind {
        StyleKind::Circle | StyleKind::Symbol => EditMode::DrawPoint,
        StyleKind::Line => EditMode::DrawLine,
        StyleKind::Fill => EditMode::DrawPolygon,
    }
}

/// Draws the toolbar as a fixed-height top panel and returns what was clicked.
pub fn panel(ui: &mut Ui, state: &ToolbarState<'_>) -> Vec<EditAction> {
    let mut actions = Vec::new();
    Panel::top("oxigis_edit_toolbar")
        .resizable(false)
        .max_size(EDIT_TOOLBAR_MAX_PT)
        .show(ui, |ui| {
            // Nothing in this row may wrap: a wrapped row is a taller row, and
            // a taller row is exactly the ratchet the height cap exists to
            // prevent.
            ui.style_mut().wrap_mode = Some(TextWrapMode::Extend);
            ui.horizontal(|ui| {
                row(ui, state, &mut actions);
            });
        });
    actions
}

/// The toolbar's single row of widgets.
fn row(ui: &mut Ui, state: &ToolbarState<'_>, actions: &mut Vec<EditAction>) {
    for mode in EditMode::ALL {
        let latched = state.mode == mode;
        // A drawing tool with nowhere to draw is disabled rather than hidden,
        // and says where to get somewhere to draw.
        let enabled = !mode.is_drawing() || state.can_draw;
        let response = ui
            .add_enabled_ui(enabled, |ui| ui.selectable_label(latched, mode_glyph(mode)))
            .inner;
        let response = if enabled {
            response
        } else {
            response
                .on_disabled_hover_text("Select a loaded local vector layer, or use + New layer.")
        };
        if response.clicked() {
            actions.push(EditAction::SetMode(mode));
        }
    }

    ui.separator();
    ui.menu_button("+ New layer", |ui| {
        for (label, kind) in [
            ("Point", StyleKind::Circle),
            ("Line", StyleKind::Line),
            ("Polygon", StyleKind::Fill),
        ] {
            if ui.button(label).clicked() {
                actions.push(EditAction::NewLayer(kind));
                ui.close();
            }
        }
    });

    ui.separator();
    let undo = ui
        .add_enabled(state.can_undo, egui::Button::new(UNDO))
        .on_hover_text(state.undo_label.unwrap_or("Nothing to undo"));
    if undo.clicked() {
        actions.push(EditAction::Undo);
    }
    let redo = ui
        .add_enabled(state.can_redo, egui::Button::new(REDO))
        .on_hover_text(state.redo_label.unwrap_or("Nothing to redo"));
    if redo.clicked() {
        actions.push(EditAction::Redo);
    }

    ui.separator();
    // A checkbox rather than a plain button: snapping is a persistent mode, and
    // a mode whose state is not visible is a mode the user has to test for.
    if ui
        .selectable_label(state.snap, "\u{2611} Snap")
        .on_hover_text("Snap to nearby vertices and edges (hold Ctrl to suspend)")
        .clicked()
    {
        actions.push(EditAction::ToggleSnap);
    }

    ui.separator();
    let delete_vertex = ui
        .add_enabled(state.has_vertex, egui::Button::new("\u{2716} Vertex"))
        .on_hover_text("Delete the picked vertex (Delete)");
    if delete_vertex.clicked() {
        actions.push(EditAction::DeleteVertex);
    }
    let delete = ui
        .add_enabled(state.has_selection, egui::Button::new("\u{2716} Feature"))
        .on_hover_text("Delete the selected feature (Delete)");
    if delete.clicked() {
        actions.push(EditAction::DeleteFeature);
    }

    ui.separator();
    if state.issues > 0 {
        if ui
            .selectable_label(false, format!("\u{26a0} {}", issue_badge(state.issues)))
            .on_hover_text(
                "Topology issues on this layer — opens the Edit window's Validation list",
            )
            .clicked()
        {
            actions.push(EditAction::ShowValidation);
        }
        ui.separator();
    }
    if ui
        .selectable_label(state.window_open, "\u{270f} Edit\u{2026}")
        .clicked()
    {
        actions.push(EditAction::ToggleWindow);
    }
}

/// Largest issue count the badge spells out; above it the count is elided.
pub const ISSUE_BADGE_CAP: usize = 99;

/// The counter's text, capped so the one data-derived string in the row cannot
/// grow past three characters and take the height cap's job away.
#[must_use]
pub fn issue_badge(count: usize) -> String {
    if count > ISSUE_BADGE_CAP {
        format!("{ISSUE_BADGE_CAP}+")
    } else {
        count.to_string()
    }
}
