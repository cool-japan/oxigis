// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The Edit window: the attribute form, the validation list and the snap
//! settings, plus the schema-cap helper they share.
//!
//! Split from `app/edit_glue.rs` under the 2000-line rule ahead of the
//! editing-v1.1 work (docs/plans/editing-v11.md, stage 0) as a pure move:
//! the choke point, the shortcuts and the interaction glue stay there.

use super::OxigisApp;
use crate::attribute_table::SYNTHETIC_COLUMN_COUNT;
use crate::edit::command::{self, EditError, EditTransaction, FeatureOp};
use crate::edit::form::FormAction;
use crate::edit::stack::{CoalesceField, CoalesceKey};
use crate::edit::topology;
use crate::edit::{self as edit, EditSelection, overlay};
use crate::ui_glyphs::{MIDDLE_DOT, WARNING};
use egui::Context;
use oxigis_render::LonLat;
use std::sync::Arc;

/// Tallest the validation list may grow before it scrolls, in points.
const VALIDATION_LIST_MAX_PT: f32 = 180.0;

impl OxigisApp {
    /// Draws the Edit window when it is open.
    ///
    /// A Context-level [`egui::Window`], not a panel section: property keys and
    /// values are unbounded user strings, and egui 0.35 panels persist their
    /// content size, so one long key inside a panel would ratchet it wider
    /// forever. A window is free to be as wide as its content needs and takes
    /// nothing away from the map when it is not.
    pub(super) fn edit_window(&mut self, ctx: &Context) {
        if !self.edit.show_window() {
            return;
        }
        let mut open = true;
        egui::Window::new("Edit")
            .default_width(380.0)
            .max_width(560.0)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.set_max_width(ui.available_width());
                self.edit_window_body(ui);
            });
        if !open {
            self.set_edit_window_open(false);
        }
    }

    /// The Edit window's three sections.
    fn edit_window_body(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Attributes")
            .default_open(true)
            .show(ui, |ui| self.attribute_form_ui(ui));
        // The `⚠` badge's one-shot reveal: `Some(true)` forces the section
        // open on exactly the frame the badge asked for it; every other frame
        // passes `None`, leaving the header under the user's control.
        let reveal = self.edit.take_reveal_validation();
        egui::CollapsingHeader::new("Validation")
            .default_open(false)
            .open(reveal.then_some(true))
            .show(ui, |ui| self.validation_ui(ui));
        egui::CollapsingHeader::new("Snapping")
            .default_open(false)
            .show(ui, |ui| self.snap_settings_ui(ui));
    }

    /// The Attributes section: the dirty banner, the form, and whatever it
    /// reports.
    fn attribute_form_ui(&mut self, ui: &mut egui::Ui) {
        let bound = self
            .selection
            .zip(self.edit.selection())
            .map(|(layer, selection)| (layer, selection.feature));
        // The layer's live column count, so the cap is measured against the
        // **layer** rather than against this one feature.
        let schema_len = self.form_schema_len();
        let action = {
            let Self {
                edit,
                local,
                selection,
                ..
            } = self;
            let features = selection.and_then(|id| local.feature_set(id));
            edit.form_mut().sync(bound, features);
            if edit.form().is_dirty() {
                let feature = edit.form().bound().map_or(0, |(_, feature)| feature);
                let message = if edit.form().bound() == bound {
                    format!("Unsaved changes for feature {feature} \u{2014} Apply or Discard.")
                } else {
                    format!(
                        "Unsaved changes for feature {feature}, which is not what is selected \
                         now \u{2014} Apply or Discard."
                    )
                };
                ui.colored_label(ui.visuals().warn_fg_color, message);
            }
            edit.form_mut().ui(ui, schema_len)
        };
        match action {
            Some(FormAction::Apply) => {
                self.apply_attribute_form();
            }
            Some(FormAction::Discard) => {
                self.edit.form_mut().discard();
                self.undo.close_coalescing();
                self.status = Some("Attribute edits discarded.".to_string());
            }
            None => {}
        }
    }

    /// The selected layer's distinct property-key count, for the attribute
    /// form's column cap.
    ///
    /// Read from the attribute table's already-derived schema when the table
    /// is bound to the selected layer — the cheap, common case. When it is not
    /// (most plainly: the table panel is hidden, so `bind()` never ran and its
    /// count is absent or stale), the count is derived here from the
    /// collection itself and memoized on the collection's [`Arc`] identity —
    /// deriving walks every feature, and this runs every frame the Edit window
    /// is open. Without the fallback the cap would be measured against `0`,
    /// letting per-feature additions push the layer past
    /// [`crate::attribute_table::MAX_PROPERTY_COLUMNS`], where the table
    /// silently stops showing the new keys — the exact outcome the cap exists
    /// to prevent.
    pub(super) fn form_schema_len(&mut self) -> usize {
        let Some(id) = self.selection else {
            return 0;
        };
        if self.table_panel.bound_layer() == self.selection {
            return self
                .table_panel
                .column_count()
                .saturating_sub(SYNTHETIC_COLUMN_COUNT);
        }
        let Some(features) = self.local.feature_set(id) else {
            return 0;
        };
        if let Some((memo_id, memo_arc, len)) = self.form_schema_memo.as_ref()
            && *memo_id == id
            && Arc::ptr_eq(memo_arc, features)
        {
            return *len;
        }
        let schema = crate::attribute_table::AttributeSchema::derive(features);
        // Distinct keys, not visible columns: the schema truncates its key
        // list at the cap, and the cap check needs the layer's real total.
        let len = schema
            .columns()
            .len()
            .saturating_sub(SYNTHETIC_COLUMN_COUNT)
            .saturating_add(schema.omitted_columns());
        self.form_schema_memo = Some((id, Arc::clone(features), len));
        len
    }

    /// Turns the form buffer into one `Replace` operation.
    ///
    /// The transaction carries a [`CoalesceField::Properties`] key, so
    /// successive Applies to the same feature fold into a single undo step until
    /// the window closes or the selection moves — one `Ctrl+Z` should undo *the
    /// attribute edit*, not the fourth of six Applies that made it up.
    ///
    /// Returns whether anything was stored.
    pub(super) fn apply_attribute_form(&mut self) -> bool {
        let Some((layer, index)) = self.edit.form().bound() else {
            self.status = Some("Select a feature before editing its attributes.".to_string());
            return false;
        };
        let properties = match self.edit.form().build() {
            Ok(properties) => properties,
            Err(message) => {
                self.status = Some(message);
                return false;
            }
        };
        let Some(features) = self.local.feature_set(layer).map(Arc::clone) else {
            self.status = Some(EditError::FeaturesNotLoaded(layer).to_string());
            return false;
        };
        let Some(before) = features.features.get(index).cloned() else {
            self.status = Some(
                EditError::IndexOutOfRange {
                    index,
                    len: features.features.len(),
                }
                .to_string(),
            );
            return false;
        };
        let mut after = before.clone();
        command::set_properties(&mut after, properties);
        let transaction = EditTransaction {
            layer,
            label: "Edit attributes",
            ops: vec![FeatureOp::Replace {
                index,
                before: Box::new(before),
                after: Box::new(after),
            }],
            selection_before: self.edit.selection(),
            selection_after: Some(EditSelection::feature(index)),
            coalesce: Some(CoalesceKey {
                epoch: self.undo.epoch(),
                layer,
                feature: index,
                field: CoalesceField::Properties,
            }),
        };
        if !self.commit_edit(transaction) {
            return false;
        }
        self.edit.form_mut().mark_applied();
        self.status = Some(format!("Attributes stored for feature {index}."));
        true
    }

    /// The Validation section: the two buttons, and the issue list.
    ///
    /// Rows are **newest-first**, because the newest is what the last commit
    /// produced; each is truncated rather than wrapped, so one long row cannot
    /// grow the window; and each is clickable, selecting its feature and moving
    /// the camera onto the issue **with the zoom left alone** — a validation
    /// click means "show me this", not "reframe everything".
    ///
    /// Rows whose feature index the collection no longer holds are dropped here,
    /// at display time. An index that went stale between a validation run and
    /// this frame is one comparison to skip here and a bookkeeping obligation on
    /// every edit path if it were chased anywhere else.
    pub(super) fn validation_ui(&mut self, ui: &mut egui::Ui) {
        let Some(id) = self.selection else {
            ui.weak("Select a layer to validate it.");
            return;
        };
        let loaded = self.local.feature_set(id).is_some();
        let mut run = false;
        let mut clear = false;
        ui.horizontal(|ui| {
            run = ui
                .add_enabled(loaded, egui::Button::new("Validate layer"))
                .on_hover_text("Check every feature of this layer for topology problems")
                .on_disabled_hover_text("This layer's features have not been read yet.")
                .clicked();
            clear = ui
                .add_enabled(self.edit.issue_count(id) > 0, egui::Button::new("Clear"))
                .on_hover_text("Drop this layer's validation list")
                .clicked();
        });
        if run {
            self.validate_active_layer();
        }
        if clear {
            self.edit.clear_issues(id);
            self.status = Some("Validation list cleared.".to_string());
        }
        if self.edit.issues(id).is_empty() {
            ui.weak(
                "No issues recorded. Edits are checked as they are committed; Validate layer \
                 checks the whole collection.",
            );
            return;
        }
        let feature_count = self
            .local
            .feature_set(id)
            .map_or(0, |features| features.features.len());
        let warn_color = ui.visuals().warn_fg_color;
        let weak_color = ui.visuals().weak_text_color();
        let mut clicked: Option<(usize, Option<LonLat>)> = None;
        egui::ScrollArea::vertical()
            .max_height(VALIDATION_LIST_MAX_PT)
            .show(ui, |ui| {
                for issue in self
                    .edit
                    .issues(id)
                    .iter()
                    .rev()
                    .filter(|issue| issue.feature < feature_count)
                    .take(edit::MAX_NOTICES)
                {
                    let color = match topology::severity(&issue.issue) {
                        topology::Severity::Warning => warn_color,
                        topology::Severity::Info => weak_color,
                    };
                    let text = format!(
                        "{WARNING} #{} {MIDDLE_DOT} {}",
                        issue.feature,
                        topology::describe(issue)
                    );
                    let row = ui.add(
                        egui::Label::new(egui::RichText::new(text).color(color))
                            .truncate()
                            .sense(egui::Sense::click()),
                    );
                    if row.clicked() {
                        clicked = Some((issue.feature, issue.at));
                    }
                }
            });
        let Some((feature, at)) = clicked else {
            return;
        };
        self.edit
            .set_selection(Some(EditSelection::feature(feature)));
        self.edit.clear_cycle();
        self.undo.close_coalescing();
        if let Some(at) = at {
            let view = self.map_panel.view();
            self.map_panel.set_view(view.with_center(at));
        }
        self.status = Some(format!(
            "Feature {feature} selected from the validation list."
        ));
    }

    /// The Snapping section: what attracts the pointer, and how close is close
    /// enough.
    fn snap_settings_ui(&mut self, ui: &mut egui::Ui) {
        let mut settings = self.edit.snap_settings();
        let mut changed = ui
            .checkbox(&mut settings.enabled, "Snap while editing")
            .on_hover_text("Hold Ctrl to suspend snapping for one gesture")
            .changed();
        let enabled = settings.enabled;
        changed |= ui
            .add_enabled_ui(enabled, |ui| {
                let mut inner = ui
                    .checkbox(&mut settings.to_vertices, "Snap to vertices")
                    .changed();
                inner |= ui
                    .checkbox(&mut settings.to_edges, "Snap to edges")
                    .changed();
                inner |= ui
                    .add(
                        egui::DragValue::new(&mut settings.tolerance_pt)
                            .speed(0.5)
                            .range(1.0..=64.0)
                            .suffix(" pt"),
                    )
                    .on_hover_text("How near the pointer has to come before snapping fires")
                    .changed();
                inner
            })
            .inner;
        if changed {
            self.edit.set_snap_settings(settings);
        }
        if self.edit.snap_degraded() {
            ui.colored_label(ui.visuals().warn_fg_color, overlay::SNAP_DEGRADED_HINT);
        }
        ui.weak(format!(
            "{} segments indexed",
            self.edit.snap_index().segment_count()
        ));
    }
}
