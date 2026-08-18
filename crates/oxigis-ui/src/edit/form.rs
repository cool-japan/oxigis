// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The attribute form: a deferred-apply buffer of typed drafts, and the widget
//! that edits it.
//!
//! # Why deferred apply
//!
//! The widgets edit *this buffer*, and only an explicit `Apply` turns it into
//! one [`crate::edit::command::FeatureOp::Replace`]. Per-keystroke apply is
//! disqualified outright: it would mean one deep collection clone, one full
//! re-quantization, one full re-tessellation and one undo entry **per
//! character**.
//!
//! # Why `Integer` and `Float` are separate kinds
//!
//! A JSON number is one type; a GIS attribute column is not. Merging them means
//! an `i64` `3` silently becomes `3.0` the first time its row is touched, which
//! changes the attribute table's column type and every downstream consumer's
//! reading of it. The kind is therefore derived from the **live value** once and
//! preserved across edits, and only an explicit kind change converts.
//!
//! # Why a dirty buffer survives a selection change
//!
//! [`FormBuffer::sync`] re-seeds only when the bound feature or the underlying
//! collection changed **and** nothing is dirty. A dirty buffer is kept, and the
//! window shows a banner naming the feature it belongs to: silently discarding
//! typed data is never forgiven, and the two-button answer (`Apply` /
//! `Discard`) is one click either way.
//!
//! # Width discipline
//!
//! Property keys and values are unbounded user strings, which is exactly what
//! ratchets an egui 0.35 panel wider forever — hence a window rather than a
//! panel, and, inside it, `ui.set_max_width(ui.available_width())`, truncated
//! labels for every data-derived string, and
//! `TextEdit::desired_width(f32::INFINITY).clip_text(true)`.
//!
//! # A note the tooltip repeats
//!
//! [`Properties`] is a plain `serde_json::Map` without `preserve_order`, so a
//! newly added key lands in sort order rather than at the end of the form.

use super::command::{self, EditError, FeatureOp};
use crate::attribute_table::MAX_PROPERTY_COLUMNS;
use crate::ui_glyphs::REMOVE;
use egui::Ui;
use oxigeo::geojson::types::{FeatureCollection, Properties};
use oxigis_core::LayerId;
use serde_json::Value;
use std::sync::Arc;

/// Width of the per-row kind picker, in egui points.
///
/// Fixed rather than content-derived: a combo that resizes with its selection
/// makes every row's value column jump as types are changed.
pub const KIND_COMBO_PT: f32 = 84.0;

/// Width of the per-row key label, in egui points. Keys are truncated into it.
pub const KEY_LABEL_PT: f32 = 120.0;

/// What kind of value a row holds.
///
/// Derived from the live value and preserved across edits — see the module docs
/// on why [`Self::Integer`] and [`Self::Float`] are not one kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FieldKind {
    /// A JSON string.
    #[default]
    Text,
    /// A JSON number with no fractional part, kept as an `i64`.
    Integer,
    /// A JSON number kept as an `f64`.
    Float,
    /// A JSON boolean.
    Bool,
    /// JSON `null`.
    Null,
    /// An array or object, edited as raw JSON text.
    Json,
}

impl FieldKind {
    /// Every kind, in picker order.
    pub const ALL: [Self; 6] = [
        Self::Text,
        Self::Integer,
        Self::Float,
        Self::Bool,
        Self::Null,
        Self::Json,
    ];

    /// The kind a live value already is.
    #[must_use]
    pub fn of(value: &Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(_) => Self::Bool,
            Value::Number(number) => {
                if number.is_i64() || number.is_u64() {
                    Self::Integer
                } else {
                    Self::Float
                }
            }
            Value::String(_) => Self::Text,
            Value::Array(_) | Value::Object(_) => Self::Json,
        }
    }

    /// Label shown in the picker.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Text => "Text",
            Self::Integer => "Integer",
            Self::Float => "Float",
            Self::Bool => "Bool",
            Self::Null => "Null",
            Self::Json => "JSON",
        }
    }
}

/// One editable property row.
#[derive(Debug, Clone, Default)]
pub struct FieldDraft {
    /// The property key. Unique within a buffer.
    pub key: String,
    /// What the value is being edited as.
    pub kind: FieldKind,
    /// Buffer for [`FieldKind::Text`], [`FieldKind::Integer`],
    /// [`FieldKind::Float`] and [`FieldKind::Json`].
    pub text: String,
    /// Value for [`FieldKind::Bool`].
    pub flag: bool,
    /// Why this row would not parse, when it would not.
    pub error: Option<String>,
}

impl FieldDraft {
    /// Seeds a row from a live value, taking its kind from the value.
    #[must_use]
    pub fn from_value(key: impl Into<String>, value: &Value) -> Self {
        let kind = FieldKind::of(value);
        let mut draft = Self {
            key: key.into(),
            kind,
            text: String::new(),
            flag: false,
            error: None,
        };
        match value {
            Value::String(text) => draft.text = text.clone(),
            Value::Bool(flag) => draft.flag = *flag,
            Value::Number(number) => draft.text = number.to_string(),
            Value::Null => {}
            Value::Array(_) | Value::Object(_) => draft.text = value.to_string(),
        }
        draft
    }

    /// A fresh row of `kind`, with that kind's neutral value.
    #[must_use]
    pub fn empty(key: impl Into<String>, kind: FieldKind) -> Self {
        let text = match kind {
            FieldKind::Integer => "0".to_string(),
            FieldKind::Float => "0".to_string(),
            FieldKind::Json => "null".to_string(),
            FieldKind::Text | FieldKind::Bool | FieldKind::Null => String::new(),
        };
        Self {
            key: key.into(),
            kind,
            text,
            flag: false,
            error: None,
        }
    }

    /// The value this row would produce.
    ///
    /// # Errors
    ///
    /// A sentence naming what would not parse — a malformed number, invalid
    /// JSON, a float that is not finite (which `serde_json` would write as
    /// `null`, silently changing the value).
    pub fn value(&self) -> Result<Value, String> {
        match self.kind {
            FieldKind::Text => Ok(Value::String(self.text.clone())),
            FieldKind::Integer => self
                .text
                .trim()
                .parse::<i64>()
                .map(Value::from)
                .map_err(|_| "not a whole number".to_string()),
            FieldKind::Float => {
                let number = self
                    .text
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| "not a number".to_string())?;
                serde_json::Number::from_f64(number)
                    .map(Value::Number)
                    .ok_or_else(|| "not a finite number".to_string())
            }
            FieldKind::Bool => Ok(Value::Bool(self.flag)),
            FieldKind::Null => Ok(Value::Null),
            FieldKind::Json => serde_json::from_str::<Value>(&self.text)
                .map_err(|error| format!("not valid JSON: {error}")),
        }
    }

    /// Switches this row's kind, carrying the current value across where it can.
    ///
    /// Best-effort rather than lossless: a conversion the value cannot survive
    /// falls back to the new kind's neutral value, which is visible immediately
    /// and one `Ctrl+Z`-free keystroke to correct, because nothing has been
    /// applied yet.
    pub fn retype(&mut self, kind: FieldKind) {
        if self.kind == kind {
            return;
        }
        let current = self.value().ok();
        self.kind = kind;
        self.error = None;
        match kind {
            FieldKind::Text => {
                self.text = match current {
                    Some(Value::String(text)) => text,
                    Some(Value::Null) | None => String::new(),
                    Some(value) => value.to_string(),
                };
            }
            FieldKind::Integer => {
                let number = current.as_ref().and_then(number_of).unwrap_or(0.0);
                self.text = (number.trunc() as i64).to_string();
            }
            FieldKind::Float => {
                let number = current.as_ref().and_then(number_of).unwrap_or(0.0);
                self.text = number.to_string();
            }
            FieldKind::Bool => {
                self.flag = match current {
                    Some(Value::Bool(flag)) => flag,
                    Some(value) => number_of(&value).is_some_and(|number| number != 0.0),
                    None => self.flag,
                };
            }
            FieldKind::Null => {}
            FieldKind::Json => {
                self.text = current.unwrap_or(Value::Null).to_string();
            }
        }
    }
}

/// The number a value carries, for a best-effort kind conversion.
fn number_of(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::Bool(true) => Some(1.0),
        Value::Bool(false) => Some(0.0),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// What the form asked the app to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormAction {
    /// Turn the buffer into one `Replace` operation.
    Apply,
    /// Throw the buffer away and re-seed from the live feature.
    Discard,
}

/// The deferred-apply attribute buffer.
///
/// See the module docs: nothing here is applied until the caller acts on a
/// [`FormAction::Apply`].
#[derive(Debug, Default)]
pub struct FormBuffer {
    /// Which layer and feature the rows were seeded from.
    bound: Option<(LayerId, usize)>,
    /// The exact collection they were seeded from, **held** so `Arc::ptr_eq`
    /// staleness detection cannot suffer ABA against a freed-and-reallocated
    /// collection — the same reasoning the snap index follows.
    source: Option<Arc<FeatureCollection>>,
    /// One row per property, in the order the live map yielded them.
    rows: Vec<FieldDraft>,
    /// The "add a key" field's contents.
    new_key: String,
    /// The kind a newly added key gets.
    new_kind: FieldKind,
    /// Whether anything has been typed since the last seed or apply.
    dirty: bool,
    /// Why the last add was refused, shown under the add row.
    add_error: Option<String>,
}

impl FormBuffer {
    /// Re-seeds from the bound feature when the binding or the collection
    /// changed and nothing is dirty.
    ///
    /// Cheap enough to call every frame the window is open: with nothing
    /// changed it is two comparisons and returns.
    pub fn sync(
        &mut self,
        bound: Option<(LayerId, usize)>,
        features: Option<&Arc<FeatureCollection>>,
    ) {
        if self.bound == bound && same_source(self.source.as_ref(), features) {
            return;
        }
        if self.dirty {
            // Kept deliberately. The window's banner names the feature these
            // rows belong to, and `Apply`/`Discard` are both one click away.
            return;
        }
        self.reseed(bound, features);
    }

    /// Replaces the rows with the bound feature's live properties.
    fn reseed(
        &mut self,
        bound: Option<(LayerId, usize)>,
        features: Option<&Arc<FeatureCollection>>,
    ) {
        self.bound = bound;
        self.source = features.map(Arc::clone);
        self.rows.clear();
        self.new_key.clear();
        self.add_error = None;
        self.dirty = false;
        let Some(((_, index), features)) = bound.zip(features) else {
            return;
        };
        let Some(properties) = features
            .features
            .get(index)
            .and_then(|feature| feature.properties.as_ref())
        else {
            return;
        };
        self.rows = properties
            .iter()
            .map(|(key, value)| FieldDraft::from_value(key, value))
            .collect();
    }

    /// Which layer and feature the rows describe.
    #[must_use]
    pub fn bound(&self) -> Option<(LayerId, usize)> {
        self.bound
    }

    /// Whether anything has been typed since the last seed or apply.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// The rows, for the caller and for tests.
    #[must_use]
    pub fn rows(&self) -> &[FieldDraft] {
        &self.rows
    }

    /// Throws the buffer away; the next [`Self::sync`] re-seeds it.
    pub fn discard(&mut self) {
        self.bound = None;
        self.source = None;
        self.rows.clear();
        self.new_key.clear();
        self.new_kind = FieldKind::Text;
        self.add_error = None;
        self.dirty = false;
    }

    /// Follows a renumbering of the bound layer's collection, so the binding
    /// keeps naming the same *feature* after an add or delete below it shifted
    /// every later index — the alternative is an `Apply` that overwrites
    /// whatever feature slid into the stale slot.
    ///
    /// A transaction against another layer moves nothing. When one of `ops`
    /// removed the bound feature itself the buffer is discarded — there is no
    /// feature the rows could truthfully be offered back to — and `false` is
    /// returned so the caller can say so if the buffer was dirty.
    pub fn remap_bound(&mut self, layer: LayerId, ops: &[FeatureOp]) -> bool {
        let Some((bound_layer, index)) = self.bound else {
            return true;
        };
        if bound_layer != layer {
            return true;
        }
        match command::remap_index(ops, index) {
            Some(next) => {
                self.bound = Some((bound_layer, next));
                true
            }
            None => {
                self.discard();
                false
            }
        }
    }

    /// Marks the buffer clean without changing it — what an applied `Apply`
    /// leaves behind, so the next `Arc` re-seeds from the freshly stored
    /// feature.
    pub fn mark_applied(&mut self) {
        self.dirty = false;
        // The collection the rows were seeded from is about to be replaced;
        // dropping the handle is what makes the next `sync` re-seed.
        self.source = None;
    }

    /// Replaces a row's text buffer.
    ///
    /// Returns whether there was such a row.
    pub fn set_row_text(&mut self, index: usize, text: impl Into<String>) -> bool {
        let Some(row) = self.rows.get_mut(index) else {
            return false;
        };
        row.text = text.into();
        self.dirty = true;
        true
    }

    /// Replaces a row's boolean value. Returns whether there was such a row.
    pub fn set_row_flag(&mut self, index: usize, flag: bool) -> bool {
        let Some(row) = self.rows.get_mut(index) else {
            return false;
        };
        row.flag = flag;
        self.dirty = true;
        true
    }

    /// Switches a row's kind, converting where it can. Returns whether there was
    /// such a row.
    pub fn set_row_kind(&mut self, index: usize, kind: FieldKind) -> bool {
        let Some(row) = self.rows.get_mut(index) else {
            return false;
        };
        row.retype(kind);
        self.dirty = true;
        true
    }

    /// Drops the row at `index`. Returns whether there was one.
    pub fn remove_row(&mut self, index: usize) -> bool {
        if index >= self.rows.len() {
            return false;
        }
        self.rows.remove(index);
        self.dirty = true;
        true
    }

    /// Adds a new key.
    ///
    /// `schema_len` is how many property columns the layer's attribute schema
    /// already has, so the column cap is measured against the **layer**, not
    /// against this one feature.
    ///
    /// # Errors
    ///
    /// [`EditError::DuplicateKey`] when the buffer already holds that key —
    /// silently overwriting a row the user cannot see is how attribute data is
    /// lost — and [`EditError::ColumnCapReached`] when the key would push the
    /// layer past [`MAX_PROPERTY_COLUMNS`], where the attribute table would stop
    /// showing it at all.
    pub fn add_key(
        &mut self,
        key: &str,
        kind: FieldKind,
        schema_len: usize,
    ) -> Result<(), EditError> {
        let key = key.trim();
        if key.is_empty() {
            return Err(EditError::DuplicateKey(String::new()));
        }
        if self.rows.iter().any(|row| row.key == key) {
            return Err(EditError::DuplicateKey(key.to_string()));
        }
        if self.rows.len().max(schema_len) >= MAX_PROPERTY_COLUMNS {
            return Err(EditError::ColumnCapReached(MAX_PROPERTY_COLUMNS));
        }
        self.rows.push(FieldDraft::empty(key, kind));
        self.dirty = true;
        Ok(())
    }

    /// The rows as a property map.
    ///
    /// # Errors
    ///
    /// A sentence naming the **first** row that would not parse, so the message
    /// points at something the user can find. The same row also carries the
    /// message inline after [`Self::ui`] has run.
    pub fn build(&self) -> Result<Properties, String> {
        let mut properties = Properties::new();
        for row in &self.rows {
            if row.key.trim().is_empty() {
                return Err("A property with no name cannot be stored.".to_string());
            }
            let value = row
                .value()
                .map_err(|reason| format!("{}: {reason}", row.key))?;
            properties.insert(row.key.clone(), value);
        }
        Ok(properties)
    }

    /// Refreshes every row's inline error.
    fn revalidate(&mut self) {
        for row in &mut self.rows {
            row.error = row.value().err();
        }
    }

    /// Draws the form and reports whichever button was pressed.
    ///
    /// `schema_len` is the layer's current property-column count, for the column
    /// cap.
    pub fn ui(&mut self, ui: &mut Ui, schema_len: usize) -> Option<FormAction> {
        // First, and not cosmetic: every string below is user data, and an
        // unbounded one inside a resizable container is exactly the egui 0.35
        // ratchet this app has already been bitten by once.
        ui.set_max_width(ui.available_width());
        self.revalidate();

        if self.bound.is_none() {
            ui.weak(
                "Select a feature on the map or in the attribute table to edit its attributes.",
            );
            return None;
        }

        let mut action = None;
        let mut removed = None;
        let mut retyped = None;
        for (index, row) in self.rows.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.add_sized(
                    [KEY_LABEL_PT, ui.spacing().interact_size.y],
                    egui::Label::new(row.key.clone()).truncate(),
                )
                .on_hover_text(row.key.clone());
                let mut kind = row.kind;
                egui::ComboBox::from_id_salt(("oxigis_edit_form_kind", index))
                    .width(KIND_COMBO_PT)
                    .selected_text(kind.label())
                    .show_ui(ui, |ui| {
                        for candidate in FieldKind::ALL {
                            ui.selectable_value(&mut kind, candidate, candidate.label());
                        }
                    });
                if kind != row.kind {
                    retyped = Some((index, kind));
                }
                if value_widget(ui, row) {
                    self.dirty = true;
                }
                if ui
                    .small_button(REMOVE)
                    .on_hover_text("Remove this property")
                    .clicked()
                {
                    removed = Some(index);
                }
            });
            if let Some(error) = row.error.as_ref() {
                ui.colored_label(ui.visuals().error_fg_color, format!("  {error}"));
            }
        }
        if let Some((index, kind)) = retyped {
            self.set_row_kind(index, kind);
        }
        if let Some(index) = removed {
            self.remove_row(index);
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.new_key)
                    .hint_text("New property")
                    .desired_width(KEY_LABEL_PT)
                    .clip_text(true),
            );
            egui::ComboBox::from_id_salt("oxigis_edit_form_new_kind")
                .width(KIND_COMBO_PT)
                .selected_text(self.new_kind.label())
                .show_ui(ui, |ui| {
                    for candidate in FieldKind::ALL {
                        ui.selectable_value(&mut self.new_kind, candidate, candidate.label());
                    }
                });
            let can_add = !self.new_key.trim().is_empty();
            if ui
                .add_enabled(can_add, egui::Button::new("+ Add"))
                .on_hover_text(
                    "Properties are stored in a sorted map, so a new key lands in \
                     alphabetical order rather than at the end.",
                )
                .clicked()
            {
                let key = core::mem::take(&mut self.new_key);
                let kind = self.new_kind;
                match self.add_key(&key, kind, schema_len) {
                    Ok(()) => self.add_error = None,
                    Err(error) => {
                        self.new_key = key;
                        self.add_error = Some(error.to_string());
                    }
                }
            }
        });
        if let Some(error) = self.add_error.as_ref() {
            ui.colored_label(ui.visuals().error_fg_color, error.clone());
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.dirty, egui::Button::new("Apply"))
                .on_hover_text("Store these attributes as one undoable step")
                .clicked()
            {
                action = Some(FormAction::Apply);
            }
            if ui
                .add_enabled(self.dirty, egui::Button::new("Discard"))
                .on_hover_text("Throw these edits away and re-read the feature")
                .clicked()
            {
                action = Some(FormAction::Discard);
            }
        });
        action
    }
}

/// Draws one row's value widget, reporting whether it changed.
fn value_widget(ui: &mut Ui, row: &mut FieldDraft) -> bool {
    match row.kind {
        FieldKind::Text => ui
            .add(
                egui::TextEdit::singleline(&mut row.text)
                    .desired_width(f32::INFINITY)
                    .clip_text(true),
            )
            .changed(),
        FieldKind::Integer => {
            let mut value = row.text.trim().parse::<i64>().unwrap_or_default();
            let changed = ui.add(egui::DragValue::new(&mut value)).changed();
            if changed {
                row.text = value.to_string();
            }
            changed
        }
        FieldKind::Float => {
            let mut value = row.text.trim().parse::<f64>().unwrap_or_default();
            let changed = ui
                .add(egui::DragValue::new(&mut value).speed(0.1))
                .changed();
            if changed {
                row.text = value.to_string();
            }
            changed
        }
        FieldKind::Bool => ui.checkbox(&mut row.flag, "").changed(),
        FieldKind::Null => {
            // The kind picker to the left *is* the "set a type" control; a
            // second identical combo in the value column would only be a
            // second way to press the same button.
            ui.weak("null \u{2014} pick a type to give it a value");
            false
        }
        FieldKind::Json => ui
            .add(
                egui::TextEdit::multiline(&mut row.text)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            )
            .changed(),
    }
}

/// Whether the buffer's held collection is the one now on offer.
fn same_source(
    held: Option<&Arc<FeatureCollection>>,
    offered: Option<&Arc<FeatureCollection>>,
) -> bool {
    match (held, offered) {
        (Some(held), Some(offered)) => Arc::ptr_eq(held, offered),
        (None, None) => true,
        _ => false,
    }
}
