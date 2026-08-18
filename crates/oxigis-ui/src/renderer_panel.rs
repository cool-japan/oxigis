//! The renderer editor: the Single / Categorized / Graduated combo, the field
//! picker, the per-class rows and the Classify helper (thematic v1.6).
//!
//! Split out of [`crate::style_panel`] rather than added to it: that module is
//! the *style* editor (one style's fields) and this one is the *renderer*
//! editor (which style a feature gets), the same separation
//! [`oxigis_core::renderer`] draws in the model — and keeping them apart is
//! what keeps both files well under the 2000-line rule.
//!
//! # What is edited in place and what is reported
//!
//! Colours, widths and every other field of a class style bind straight to the
//! `&mut LayerStyleSet` the caller passes in, exactly as the base style's
//! fields do, so they ride the panel's existing undo seam (`sync_local_style`
//! observes the set before and after the frame) with no new plumbing.
//!
//! Structural changes — switching renderer kind, running Classify, adding or
//! removing a class — are ALSO applied in place, for the same reason: they are
//! edits to the style set the caller owns, not to the `Project::styles` map
//! entry, which is the only thing [`crate::style_panel::StyleAction`] exists
//! to change. The one thing this module reports back is
//! [`RendererEvent::Repartitioned`], which tells the caller that the change
//! moved the CLASS LIST and not merely a colour — a hint the app can use to
//! decide how much work the restyle is worth doing mid-drag.
//!
//! # Why the field picker is a fixed list
//!
//! The attribute keys come from the layer's own schema
//! ([`crate::attribute_table::AttributeSchema`], already capped at
//! `MAX_PROPERTY_COLUMNS`), so the combo is bounded by construction and costs
//! no scan per frame. Scanning for *values* — which is O(features) — happens
//! only when Classify is pressed.

use egui::Ui;
use oxigis_core::{
    AttrValue, Classification, GeometryFamily, LayerStyle, LayerStyleSet, MAX_STYLE_CLASSES,
    Renderer, RendererKind,
};

use crate::local_vector::classify::{
    self, NumericSummary, UniqueValues, categorized_renderer, graduated_renderer,
};

/// How many classes the Classify button asks for by default.
///
/// Five is the number every cartography text reaches for: enough steps that a
/// choropleth reads as a gradient, few enough that a legend can be told apart
/// at a glance.
pub const DEFAULT_CLASS_COUNT: usize = 5;

/// The largest class count the classify spinner offers.
///
/// The model's own cap ([`MAX_STYLE_CLASSES`]) is the hard limit; this is the
/// soft one, because a graduated ramp of more than a dozen steps is not
/// legible on a page and asking for 64 of them is almost always a mistake. A
/// *categorized* classification is not limited to it — unique values are what
/// they are — only the graduated spinner is.
pub const MAX_GRADUATED_CLASSES: usize = 12;

/// Which break rule the Classify button applies to a numeric field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BreakRule {
    /// Equal steps between the minimum and the maximum — the ramp reads as the
    /// VALUE, so a skewed distribution puts most features in one class.
    #[default]
    EqualInterval,
    /// Equal feature counts per class — the ramp reads as the RANK, so every
    /// class is equally visible whatever the distribution.
    Quantile,
}

impl BreakRule {
    /// Both rules, in panel order.
    pub const ALL: [BreakRule; 2] = [Self::EqualInterval, Self::Quantile];

    /// The panel label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::EqualInterval => "Equal interval",
            Self::Quantile => "Quantile",
        }
    }
}

/// What the renderer editor did this frame that the caller needs to know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererEvent {
    /// The class list itself moved — a new classification, a switched kind, a
    /// class added or dropped. The layer's meshes have to be re-partitioned,
    /// not merely repainted.
    Repartitioned,
}

/// The renderer editor's cross-frame state.
///
/// Deliberately NOT `Copy`, unlike [`crate::style_panel::StylePanelState`]: it
/// carries the classify form's field name, which is a `String`. The app owns
/// one and hands `&mut` in.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RendererPanelState {
    /// Attribute the classify form will read. Empty until a field is picked.
    field: String,
    /// How many classes a graduated classify asks for.
    classes: usize,
    /// Which break rule a graduated classify applies.
    rule: BreakRule,
    /// What the last Classify press found, for the notice line. Cleared
    /// whenever the field or the kind changes, so a stale count can never be
    /// shown against a different classification.
    notice: Option<String>,
}

impl RendererPanelState {
    /// A fresh state, asking for [`DEFAULT_CLASS_COUNT`] equal-interval
    /// classes.
    #[must_use]
    pub fn new() -> Self {
        Self {
            field: String::new(),
            classes: DEFAULT_CLASS_COUNT,
            rule: BreakRule::default(),
            notice: None,
        }
    }

    /// The field the classify form is pointed at.
    #[must_use]
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Points the classify form at `field`, clearing any stale notice.
    pub fn set_field(&mut self, field: impl Into<String>) {
        let field = field.into();
        if field != self.field {
            self.notice = None;
        }
        self.field = field;
    }

    /// How many classes a graduated classify asks for, always within
    /// `1..=MAX_GRADUATED_CLASSES`.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.classes.clamp(1, MAX_GRADUATED_CLASSES)
    }

    /// Sets the requested class count, clamped.
    pub fn set_class_count(&mut self, count: usize) {
        self.classes = count.clamp(1, MAX_GRADUATED_CLASSES);
    }

    /// The break rule a graduated classify applies.
    #[must_use]
    pub fn rule(&self) -> BreakRule {
        self.rule
    }

    /// The notice line the last Classify left, if any.
    #[must_use]
    pub fn notice(&self) -> Option<&str> {
        self.notice.as_deref()
    }

    /// Adopts the renderer's own field when the state has none — so opening
    /// the panel on a layer that is already classified shows THAT field rather
    /// than a blank picker.
    fn adopt(&mut self, renderer: &Renderer) {
        if self.classes == 0 {
            self.classes = DEFAULT_CLASS_COUNT;
        }
        if let Some(field) = renderer.field()
            && self.field.is_empty()
        {
            self.field = field.to_owned();
        }
    }
}

/// Everything the editor needs about the layer's data, gathered by the caller.
///
/// A borrowed view rather than the `FeatureCollection` itself, because the
/// panel needs the collection only when Classify is pressed — every other
/// frame it needs just the key list, which the app already holds.
#[derive(Debug, Clone, Copy)]
pub struct LayerAttributes<'a> {
    /// The attribute keys the layer carries, in table-column order.
    pub keys: &'a [String],
    /// The features, for the Classify scan. [`None`] disables Classify (a
    /// layer whose data the app cannot reach right now) without hiding the
    /// rest of the editor.
    pub features: Option<&'a oxigeo::geojson::types::FeatureCollection>,
    /// Why this layer cannot carry a thematic renderer at all — [`None`] for
    /// every layer that can.
    ///
    /// Set for a TILED layer
    /// ([`crate::vector_provider::TILED_RENDERER_REFUSAL`]): an MVT paint is
    /// matched by source-layer name and never sees a feature's attributes, so
    /// a working-looking combo there would change the saved file and nothing
    /// on screen. The kind combo is then drawn DISABLED with this as its hover
    /// text — a named refusal rather than a missing control.
    pub refusal: Option<&'a str>,
}

impl LayerAttributes<'_> {
    /// An empty view — no keys, no data, no refusal.
    #[must_use]
    pub fn none() -> LayerAttributes<'static> {
        LayerAttributes {
            keys: &[],
            features: None,
            refusal: None,
        }
    }

    /// A view that refuses thematic styling, with `reason` as the hover text.
    #[must_use]
    pub fn refused(reason: &str) -> LayerAttributes<'_> {
        LayerAttributes {
            keys: &[],
            features: None,
            refusal: Some(reason),
        }
    }
}

/// Draws the renderer editor for `set`, applying every edit in place.
///
/// Returns [`RendererEvent::Repartitioned`] when the change moved the class
/// list rather than only a colour.
pub fn ui(
    ui: &mut Ui,
    set: &mut LayerStyleSet,
    attributes: LayerAttributes<'_>,
    state: &mut RendererPanelState,
) -> Option<RendererEvent> {
    state.adopt(set.renderer());
    let before = set.classification();

    ui.horizontal(|ui| {
        ui.label("Renderer");
        // A refused layer gets the combo DISABLED with the reason on hover,
        // never a hidden control: "there is no such feature" and "this layer
        // cannot use it, because …" are different statements, and only the
        // second one is true.
        if let Some(reason) = attributes.refusal {
            ui.add_enabled(
                false,
                egui::Button::new(RendererKind::Single.label()).wrap(),
            )
            .on_disabled_hover_text(reason);
            return;
        }
        egui::ComboBox::from_id_salt("oxigis_renderer_kind")
            .selected_text(set.renderer().kind().label())
            .show_ui(ui, |ui| {
                for kind in RendererKind::ALL {
                    let selected = set.renderer().kind() == kind;
                    if ui.selectable_label(selected, kind.label()).clicked() && !selected {
                        switch_kind(set, kind, state);
                    }
                }
            });
    });

    if attributes.refusal.is_some() || set.renderer().is_single() {
        return report(before, set);
    }

    field_row(ui, set, attributes.keys, state);
    classify_row(ui, set, attributes, state);
    if let Some(notice) = state.notice() {
        ui.weak(notice);
    }
    class_rows(ui, set);
    fallback_row(ui, set);

    report(before, set)
}

/// Whether the class list moved between `before` and the set's state now.
fn report(before: Classification, set: &LayerStyleSet) -> Option<RendererEvent> {
    (set.classification() != before).then_some(RendererEvent::Repartitioned)
}

/// Switches the renderer to `kind`, keeping what carries over.
///
/// The field survives a Categorized ↔ Graduated switch (the user picked it,
/// and re-picking it after every switch would be a papercut); the CLASSES do
/// not, because an exact-value class list and a range list are not
/// translatable into one another — a switch that pretended otherwise would
/// silently produce a classification that matches nothing.
fn switch_kind(set: &mut LayerStyleSet, kind: RendererKind, state: &mut RendererPanelState) {
    let field = set
        .renderer()
        .field()
        .map(str::to_owned)
        .unwrap_or_else(|| state.field.clone());
    state.notice = None;
    set.set_renderer(match kind {
        RendererKind::Single => Renderer::Single,
        RendererKind::Categorized => Renderer::categorized(field, [], None),
        RendererKind::Graduated => Renderer::graduated(field, [], None),
    });
}

/// The attribute picker: a combo over the layer's keys, plus a free-text field
/// for a layer whose keys the caller could not supply.
fn field_row(
    ui: &mut Ui,
    set: &mut LayerStyleSet,
    keys: &[String],
    state: &mut RendererPanelState,
) {
    ui.horizontal(|ui| {
        ui.label("Field");
        if keys.is_empty() {
            let mut text = state.field.clone();
            if ui.text_edit_singleline(&mut text).changed() {
                state.set_field(text.clone());
                set.renderer_mut().set_field(text);
            }
            return;
        }
        let current = if state.field.is_empty() {
            "(pick one)"
        } else {
            state.field.as_str()
        };
        egui::ComboBox::from_id_salt("oxigis_renderer_field")
            .selected_text(current)
            .show_ui(ui, |ui| {
                for key in keys {
                    let selected = key == &state.field;
                    if ui.selectable_label(selected, key).clicked() && !selected {
                        state.set_field(key.clone());
                        set.renderer_mut().set_field(key.clone());
                    }
                }
            });
    });
}

/// The Classify controls: the break rule and class count for a graduated
/// renderer, then the button itself.
fn classify_row(
    ui: &mut Ui,
    set: &mut LayerStyleSet,
    attributes: LayerAttributes<'_>,
    state: &mut RendererPanelState,
) {
    let graduated = set.renderer().kind() == RendererKind::Graduated;
    if graduated {
        ui.horizontal(|ui| {
            ui.label("Breaks");
            for rule in BreakRule::ALL {
                let selected = state.rule == rule;
                if ui.selectable_label(selected, rule.label()).clicked() {
                    state.rule = rule;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Classes");
            let mut count = state.class_count();
            if ui
                .add(egui::Slider::new(&mut count, 1..=MAX_GRADUATED_CLASSES))
                .changed()
            {
                state.set_class_count(count);
            }
        });
    }

    ui.horizontal(|ui| {
        let ready = !state.field.is_empty() && attributes.features.is_some();
        let button = ui.add_enabled(ready, egui::Button::new("Classify"));
        let button = if state.field.is_empty() {
            button.on_disabled_hover_text("Pick a field first.")
        } else if attributes.features.is_none() {
            button.on_disabled_hover_text("This layer's data is not loaded.")
        } else {
            button
        };
        if button.clicked()
            && let Some(features) = attributes.features
        {
            let notice = apply_classify(set, features, state);
            state.notice = notice;
        }
        if !set.renderer().is_single()
            && ui
                .button("Clear classes")
                .on_hover_text("Keep the renderer, drop every class.")
                .clicked()
        {
            clear_classes(set);
            state.notice = None;
        }
    });
}

/// Runs the scan the Classify button asks for and installs the result,
/// returning the notice line to show under it.
fn apply_classify(
    set: &mut LayerStyleSet,
    features: &oxigeo::geojson::types::FeatureCollection,
    state: &RendererPanelState,
) -> Option<String> {
    let field = state.field.clone();
    let base = set.base().clone();
    match set.renderer().kind() {
        RendererKind::Single => None,
        RendererKind::Categorized => {
            let mut unique: UniqueValues =
                classify::unique_values(features, &field, MAX_STYLE_CLASSES);
            let values = core::mem::take(&mut unique.values);
            let found = values.len();
            set.set_renderer(categorized_renderer(&base, &field, values));
            Some(categorized_notice(found, &unique))
        }
        RendererKind::Graduated => {
            let Some(summary) = classify::numeric_summary(features, &field) else {
                return Some(format!("{field} holds no numeric values."));
            };
            let breaks = match state.rule {
                BreakRule::EqualInterval => {
                    classify::equal_interval_breaks(summary.min, summary.max, state.class_count())
                }
                BreakRule::Quantile => {
                    classify::quantile_breaks(&summary.values, state.class_count())
                }
            };
            let got = breaks.len();
            set.set_renderer(graduated_renderer(&base, &field, &breaks));
            Some(graduated_notice(got, state.class_count(), &summary))
        }
    }
}

/// The line shown under a categorized Classify.
fn categorized_notice(found: usize, unique: &UniqueValues) -> String {
    let mut notice = format!("{found} class{}.", if found == 1 { "" } else { "es" });
    if unique.overflow > 0 {
        notice.push_str(&format!(
            " {} more value{} draw with the fallback (the limit is {MAX_STYLE_CLASSES}).",
            unique.overflow,
            if unique.overflow == 1 { "" } else { "s" },
        ));
    }
    if unique.unclassified > 0 {
        notice.push_str(&format!(
            " {} feature{} carry no value.",
            unique.unclassified,
            if unique.unclassified == 1 { "" } else { "s" },
        ));
    }
    notice
}

/// The line shown under a graduated Classify.
fn graduated_notice(got: usize, asked: usize, summary: &NumericSummary) -> String {
    let mut notice = format!(
        "{got} class{} over {} value{}.",
        if got == 1 { "" } else { "es" },
        summary.count,
        if summary.count == 1 { "" } else { "s" },
    );
    if got < asked {
        notice.push_str(" Tied values collapsed the rest.");
    }
    if summary.sampled {
        notice.push_str(" Breaks taken from a sample of a very large dataset.");
    }
    notice
}

/// Empties the class list, keeping the renderer kind and field.
fn clear_classes(set: &mut LayerStyleSet) {
    let field = set.renderer().field().unwrap_or_default().to_owned();
    let kind = set.renderer().kind();
    set.set_renderer(match kind {
        RendererKind::Single => Renderer::Single,
        RendererKind::Categorized => Renderer::categorized(field, [], None),
        RendererKind::Graduated => Renderer::graduated(field, [], None),
    });
}

/// One editable row per class: its legend label, its colour, and a remove
/// button — plus the width/radius the class's own style kind carries.
fn class_rows(ui: &mut Ui, set: &mut LayerStyleSet) {
    let count = set.renderer().class_count();
    if count == 0 {
        ui.weak("No classes yet — press Classify.");
        return;
    }
    let overflow = set.renderer().overflow_class_count();
    let mut remove = None;
    // Bounded by `MAX_STYLE_CLASSES`, so the scroll area's content is bounded
    // too and the rows can be drawn eagerly.
    egui::ScrollArea::vertical()
        .id_salt("oxigis_renderer_classes")
        .max_height(260.0)
        .show(ui, |ui| {
            for index in 0..count {
                let label = set.renderer().class_label(index);
                ui.horizontal(|ui| {
                    if let Some(style) = set.renderer_mut().class_style_mut(index) {
                        crate::style_panel::edit_style_color(ui, style);
                        edit_class_size(ui, style, index);
                    }
                    ui.label(label);
                    if ui.small_button("×").on_hover_text("Remove").clicked() {
                        remove = Some(index);
                    }
                });
            }
        });
    if overflow > 0 {
        ui.weak(format!(
            "{overflow} further class{} stored but not drawn — the limit is {MAX_STYLE_CLASSES}.",
            if overflow == 1 { "" } else { "es" },
        ));
    }
    if let Some(index) = remove {
        set.renderer_mut().remove_class(index);
    }
}

/// The one size knob a class row carries, chosen by the class style's kind: a
/// line's width, a circle's radius, a fill's opacity.
///
/// Deliberately ONE knob rather than the base style's full editor: a class row
/// has to stay one line high for a 12-class legend to be usable, and the
/// remaining fields (outline, halo, text field) are layer-wide decisions that
/// belong in the base style above.
fn edit_class_size(ui: &mut Ui, style: &mut LayerStyle, index: usize) {
    match style {
        LayerStyle::Fill(fill) => {
            let mut opacity = fill.opacity();
            if ui
                .add(
                    egui::DragValue::new(&mut opacity)
                        .speed(0.01)
                        .range(0.0..=1.0)
                        .prefix("α "),
                )
                .on_hover_text("Opacity")
                .changed()
            {
                fill.set_opacity(opacity);
            }
        }
        LayerStyle::Line(line) => {
            let mut width = line.width();
            if ui
                .add(
                    egui::DragValue::new(&mut width)
                        .speed(0.1)
                        .range(0.0..=20.0)
                        .suffix(" px"),
                )
                .on_hover_text("Width")
                .changed()
            {
                line.set_width(width);
            }
        }
        LayerStyle::Circle(circle) => {
            let mut radius = circle.radius();
            if ui
                .add(
                    egui::DragValue::new(&mut radius)
                        .speed(0.1)
                        .range(0.0..=50.0)
                        .suffix(" px"),
                )
                .on_hover_text("Radius")
                .changed()
            {
                circle.set_radius(radius);
            }
        }
        LayerStyle::Symbol(symbol) => {
            let mut size = symbol.text_size();
            if ui
                .add(
                    egui::DragValue::new(&mut size)
                        .speed(0.5)
                        .range(0.0..=64.0)
                        .suffix(" pt"),
                )
                .on_hover_text("Text size")
                .changed()
            {
                symbol.set_text_size(size);
            }
        }
    }
    let _ = index;
}

/// The fallback row: what a feature matching no class draws with.
fn fallback_row(ui: &mut Ui, set: &mut LayerStyleSet) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("Everything else");
        let base = set.base().clone();
        let Some(slot) = set.renderer_mut().fallback_mut() else {
            return;
        };
        let mut explicit = slot.is_some();
        if ui
            .checkbox(&mut explicit, "own colour")
            .on_hover_text("Otherwise unmatched features draw with the layer's own style.")
            .changed()
        {
            *slot = explicit.then(|| base.clone());
        }
        if let Some(style) = slot.as_mut() {
            crate::style_panel::edit_style_color(ui, style);
        }
    });
}

/// Which class of `set` a feature carrying `value` on the renderer's field
/// would land in — a helper for a legend or a tooltip, so a caller does not
/// have to build a property map to ask.
#[must_use]
pub fn class_of_value(set: &LayerStyleSet, value: &AttrValue) -> Option<usize> {
    let field = set.renderer().field()?;
    let mut map = serde_json::Map::new();
    map.insert(field.to_owned(), attr_to_json(value));
    set.renderer().class_of(&map)
}

/// One stored attribute value as the JSON it came from.
fn attr_to_json(value: &AttrValue) -> serde_json::Value {
    match value {
        AttrValue::Bool(flag) => serde_json::Value::Bool(*flag),
        AttrValue::Number(number) => serde_json::Number::from_f64(*number)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        AttrValue::Text(text) => serde_json::Value::String(text.clone()),
    }
}

/// The legend rows a classified layer's `family` draws — the class label and
/// the style it resolves to, fallback last.
///
/// The ONE place a legend gets its rows from, so the printed legend and any
/// on-screen one cannot disagree with the map: every row is resolved through
/// [`LayerStyleSet::style_for_class`], the same function the mesh partition
/// uses.
#[must_use]
pub fn legend_rows(set: &LayerStyleSet, family: GeometryFamily) -> Vec<(String, LayerStyle)> {
    let count = set.renderer().class_count();
    let mut rows = Vec::with_capacity(count + 1);
    for index in 0..count {
        rows.push((
            set.renderer().class_label(index),
            set.style_for_class(family, Some(index)),
        ));
    }
    if count > 0 {
        rows.push((
            "Everything else".to_string(),
            set.style_for_class(family, None),
        ));
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigeo::geojson::types::FeatureCollection;
    use oxigis_core::{CategoryClass, Color, FillStyle, GraduatedClass, LineStyle};

    fn parse(text: &str) -> FeatureCollection {
        match oxigeo::geojson::reader::feature_collection_from_str(text) {
            Ok(features) => features,
            Err(error) => panic!("the fixture must parse: {error}"),
        }
    }

    fn zones() -> FeatureCollection {
        parse(
            r#"{"type":"FeatureCollection","features":[
              {"type":"Feature","properties":{"zone":"a","pop":10},
               "geometry":{"type":"Point","coordinates":[0,0]}},
              {"type":"Feature","properties":{"zone":"b","pop":20},
               "geometry":{"type":"Point","coordinates":[1,0]}},
              {"type":"Feature","properties":{"zone":"a","pop":30},
               "geometry":{"type":"Point","coordinates":[2,0]}},
              {"type":"Feature","properties":{"zone":"c","pop":40},
               "geometry":{"type":"Point","coordinates":[3,0]}}]}"#,
        )
    }

    fn fill_set() -> LayerStyleSet {
        LayerStyleSet::new(LayerStyle::Fill(FillStyle::new(Color::from_rgb(
            80, 140, 200,
        ))))
    }

    fn keys() -> Vec<String> {
        vec!["zone".to_string(), "pop".to_string()]
    }

    fn state_on(field: &str) -> RendererPanelState {
        let mut state = RendererPanelState::new();
        state.set_field(field);
        state
    }

    #[test]
    fn a_fresh_state_asks_for_five_equal_interval_classes() {
        let state = RendererPanelState::new();
        assert_eq!(state.class_count(), DEFAULT_CLASS_COUNT);
        assert_eq!(state.rule(), BreakRule::EqualInterval);
        assert_eq!(state.field(), "");
        assert_eq!(state.notice(), None);
        assert_eq!(RendererPanelState::default().class_count(), 1, "clamped");
        for rule in BreakRule::ALL {
            assert!(!rule.label().is_empty());
        }
    }

    #[test]
    fn the_class_count_is_clamped_to_the_legible_range() {
        let mut state = RendererPanelState::new();
        state.set_class_count(0);
        assert_eq!(state.class_count(), 1);
        state.set_class_count(usize::MAX);
        assert_eq!(state.class_count(), MAX_GRADUATED_CLASSES);
        state.set_class_count(7);
        assert_eq!(state.class_count(), 7);
    }

    #[test]
    fn drawing_a_single_symbol_set_changes_nothing_and_reports_nothing() {
        // The floor: opening the panel on an unclassified layer must not
        // write one byte, or a project would go dirty just by being looked at.
        let mut set = fill_set();
        let before = set.clone();
        let mut state = RendererPanelState::new();
        egui::__run_test_ui(|ui| {
            let event = super::ui(ui, &mut set, LayerAttributes::none(), &mut state);
            assert_eq!(event, None);
        });
        assert_eq!(set, before);
        assert!(set.is_single_symbol());
    }

    #[test]
    fn drawing_a_classified_set_changes_nothing_without_interaction() {
        let mut set = fill_set();
        set.set_renderer(Renderer::categorized(
            "zone",
            [CategoryClass::new(
                AttrValue::text("a"),
                LayerStyle::Fill(FillStyle::new(Color::BLACK)),
            )],
            Some(LayerStyle::Fill(FillStyle::new(Color::WHITE))),
        ));
        let before = set.clone();
        let features = zones();
        let mut state = state_on("zone");
        egui::__run_test_ui(|ui| {
            let event = super::ui(
                ui,
                &mut set,
                LayerAttributes {
                    keys: &keys(),
                    features: Some(&features),
                    refusal: None,
                },
                &mut state,
            );
            assert_eq!(event, None, "drawing is not editing");
        });
        assert_eq!(set, before);
    }

    #[test]
    fn the_state_adopts_the_renderers_own_field_when_it_has_none() {
        let mut set = fill_set();
        set.set_renderer(Renderer::graduated("pop", [], None));
        let mut state = RendererPanelState::new();
        assert_eq!(state.field(), "");
        egui::__run_test_ui(|ui| {
            let _ = super::ui(ui, &mut set, LayerAttributes::none(), &mut state);
        });
        assert_eq!(
            state.field(),
            "pop",
            "opening the panel on a classified layer shows THAT field",
        );
        // And it does not overwrite a field the user already picked.
        let mut chosen = state_on("zone");
        egui::__run_test_ui(|ui| {
            let _ = super::ui(ui, &mut set, LayerAttributes::none(), &mut chosen);
        });
        assert_eq!(chosen.field(), "zone");
    }

    #[test]
    fn switching_kinds_keeps_the_field_and_drops_the_classes() {
        let mut set = fill_set();
        let mut state = state_on("zone");
        switch_kind(&mut set, RendererKind::Categorized, &mut state);
        assert_eq!(set.renderer().kind(), RendererKind::Categorized);
        assert_eq!(set.renderer().field(), Some("zone"));
        assert_eq!(set.class_count(), 0);

        // Classify, then switch: an exact-value list cannot become a range
        // list, so the classes go rather than becoming a wrong classification.
        set.set_renderer(categorized_renderer(
            &set.base().clone(),
            "zone",
            [AttrValue::text("a"), AttrValue::text("b")],
        ));
        assert_eq!(set.class_count(), 2);
        switch_kind(&mut set, RendererKind::Graduated, &mut state);
        assert_eq!(set.renderer().kind(), RendererKind::Graduated);
        assert_eq!(set.renderer().field(), Some("zone"), "the field survives");
        assert_eq!(set.class_count(), 0, "the classes do not");

        switch_kind(&mut set, RendererKind::Single, &mut state);
        assert!(set.is_single_symbol());
        assert_eq!(set.renderer().field(), None);
    }

    #[test]
    fn classify_builds_one_class_per_unique_value_and_says_how_many() {
        let mut set = fill_set();
        set.set_renderer(Renderer::categorized("zone", [], None));
        let state = state_on("zone");
        let notice = apply_classify(&mut set, &zones(), &state).unwrap_or_default();
        assert_eq!(set.class_count(), 3, "a, b, c");
        assert!(notice.starts_with("3 classes."), "{notice}");
        assert_eq!(set.renderer().class_label(0), "a");
        // Every scanned feature now lands in a class rather than the fallback.
        for feature in &zones().features {
            let Some(properties) = feature.properties.as_ref() else {
                continue;
            };
            assert!(set.renderer().class_of(properties).is_some());
        }
        // And each class draws in its own colour, over the base's shape.
        let first = set.style_for_class(GeometryFamily::Polygon, Some(0));
        let second = set.style_for_class(GeometryFamily::Polygon, Some(1));
        assert_ne!(first, second);
        assert!(matches!(first, LayerStyle::Fill(_)));
    }

    #[test]
    fn classify_reports_what_it_could_not_cover() {
        let mut set = fill_set();
        set.set_renderer(Renderer::categorized("zone", [], None));
        let sparse = parse(
            r#"{"type":"FeatureCollection","features":[
              {"type":"Feature","properties":{"zone":"a"},
               "geometry":{"type":"Point","coordinates":[0,0]}},
              {"type":"Feature","properties":{},
               "geometry":{"type":"Point","coordinates":[1,0]}},
              {"type":"Feature","properties":{"zone":null},
               "geometry":{"type":"Point","coordinates":[2,0]}}]}"#,
        );
        let notice = apply_classify(&mut set, &sparse, &state_on("zone")).unwrap_or_default();
        assert!(notice.contains("1 class."), "{notice}");
        assert!(notice.contains("2 features carry no value"), "{notice}");
    }

    #[test]
    fn a_graduated_classify_uses_the_chosen_rule_and_class_count() {
        let mut set = fill_set();
        set.set_renderer(Renderer::graduated("pop", [], None));
        let mut state = state_on("pop");
        state.set_class_count(2);
        state.rule = BreakRule::EqualInterval;
        let notice = apply_classify(&mut set, &zones(), &state).unwrap_or_default();
        assert_eq!(set.class_count(), 2);
        assert!(notice.contains("over 4 values"), "{notice}");
        let equal: Vec<f64> = set
            .renderer()
            .graduated_classes()
            .iter()
            .map(GraduatedClass::upper)
            .collect();
        assert_eq!(equal, vec![25.0, 40.0]);

        state.rule = BreakRule::Quantile;
        let _ = apply_classify(&mut set, &zones(), &state);
        let quantile: Vec<f64> = set
            .renderer()
            .graduated_classes()
            .iter()
            .map(GraduatedClass::upper)
            .collect();
        assert_eq!(quantile, vec![20.0, 40.0], "equal counts, not equal steps");
        assert_ne!(equal, quantile, "the two rules really differ");
    }

    #[test]
    fn a_graduated_classify_on_a_text_field_refuses_rather_than_inventing_breaks() {
        let mut set = fill_set();
        set.set_renderer(Renderer::graduated("zone", [], None));
        let notice = apply_classify(&mut set, &zones(), &state_on("zone")).unwrap_or_default();
        assert_eq!(notice, "zone holds no numeric values.");
        assert_eq!(set.class_count(), 0, "and nothing was installed");
    }

    #[test]
    fn a_tied_quantile_says_it_collapsed_classes() {
        let mut set = fill_set();
        set.set_renderer(Renderer::graduated("pop", [], None));
        let tied = parse(
            r#"{"type":"FeatureCollection","features":[
              {"type":"Feature","properties":{"pop":1},
               "geometry":{"type":"Point","coordinates":[0,0]}},
              {"type":"Feature","properties":{"pop":1},
               "geometry":{"type":"Point","coordinates":[1,0]}},
              {"type":"Feature","properties":{"pop":1},
               "geometry":{"type":"Point","coordinates":[2,0]}},
              {"type":"Feature","properties":{"pop":9},
               "geometry":{"type":"Point","coordinates":[3,0]}}]}"#,
        );
        let mut state = state_on("pop");
        state.set_class_count(4);
        state.rule = BreakRule::Quantile;
        let notice = apply_classify(&mut set, &tied, &state).unwrap_or_default();
        assert_eq!(set.class_count(), 2);
        assert!(notice.contains("Tied values collapsed"), "{notice}");
    }

    #[test]
    fn clearing_the_classes_keeps_the_kind_and_the_field() {
        let mut set = fill_set();
        set.set_renderer(categorized_renderer(
            &set.base().clone(),
            "zone",
            [AttrValue::text("a")],
        ));
        assert_eq!(set.class_count(), 1);
        clear_classes(&mut set);
        assert_eq!(set.class_count(), 0);
        assert_eq!(set.renderer().kind(), RendererKind::Categorized);
        assert_eq!(set.renderer().field(), Some("zone"));
        assert_eq!(set.renderer().fallback(), None, "and the fallback too");
    }

    #[test]
    fn a_repartition_is_reported_and_a_recolour_is_not() {
        // THE distinction the event exists to make: the caller pays a full
        // re-partition for one and a repaint for the other.
        let mut set = fill_set();
        let before = set.classification();
        set.set_renderer(categorized_renderer(
            &set.base().clone(),
            "zone",
            [AttrValue::text("a")],
        ));
        assert_eq!(report(before, &set), Some(RendererEvent::Repartitioned));

        let unchanged = set.classification();
        match set.renderer_mut().class_style_mut(0) {
            Some(style) => *style = LayerStyle::Fill(FillStyle::new(Color::WHITE)),
            None => panic!("class 0 exists"),
        }
        assert_eq!(report(unchanged, &set), None, "a colour edit repaints only");

        let unchanged = set.classification();
        set.renderer_mut().remove_class(0);
        assert_eq!(report(unchanged, &set), Some(RendererEvent::Repartitioned));
    }

    #[test]
    fn every_class_row_edits_the_one_size_knob_its_kind_carries() {
        let color = Color::from_rgb(1, 2, 3);
        for mut style in [
            LayerStyle::Fill(FillStyle::new(color)),
            LayerStyle::Line(LineStyle::new(color, 2.0)),
            LayerStyle::Circle(oxigis_core::CircleStyle::new(4.0, color)),
            LayerStyle::Symbol(oxigis_core::SymbolStyle::new("name")),
        ] {
            let before = style.clone();
            egui::__run_test_ui(|ui| {
                edit_class_size(ui, &mut style, 0);
            });
            assert_eq!(style, before, "drawing a class row is not editing it");
        }
    }

    #[test]
    fn the_whole_editor_draws_for_every_kind_without_panicking() {
        let features = zones();
        let attributes = LayerAttributes {
            keys: &keys(),
            features: Some(&features),
            refusal: None,
        };
        for kind in RendererKind::ALL {
            let mut set = fill_set();
            let mut state = state_on("zone");
            switch_kind(&mut set, kind, &mut state);
            let after_switch = set.clone();
            egui::__run_test_ui(|ui| {
                let _ = super::ui(ui, &mut set, attributes, &mut state);
            });
            assert_eq!(set, after_switch, "{kind:?} draws without editing");
            // And with a full class list too.
            if kind != RendererKind::Single {
                set.set_renderer(categorized_renderer(
                    &set.base().clone(),
                    "zone",
                    (0..MAX_STYLE_CLASSES).filter_map(|index| AttrValue::number(index as f64)),
                ));
                let full = set.clone();
                egui::__run_test_ui(|ui| {
                    let _ = super::ui(ui, &mut set, attributes, &mut state);
                });
                assert_eq!(set, full, "a full class list draws too");
            }
        }
    }

    #[test]
    fn a_refused_layer_shows_the_reason_and_no_combo_at_all() {
        // A tiled layer: the combo is disabled with the reason on hover, and
        // NOTHING below it is drawn — a class list that could never take
        // effect must not be reachable.
        let mut set = fill_set();
        set.set_renderer(Renderer::categorized("zone", [], None));
        let before = set.clone();
        let mut state = state_on("zone");
        let reason = crate::vector_provider::TILED_RENDERER_REFUSAL;
        egui::__run_test_ui(|ui| {
            let event = super::ui(ui, &mut set, LayerAttributes::refused(reason), &mut state);
            assert_eq!(event, None);
        });
        assert_eq!(set, before, "a refused layer's style is never rewritten");
        assert!(!reason.is_empty());
    }

    #[test]
    fn the_editor_still_draws_with_no_keys_and_no_data() {
        // A layer whose schema the caller could not supply falls back to a
        // free-text field and a disabled Classify — never to a hidden editor.
        let mut set = fill_set();
        set.set_renderer(Renderer::categorized("zone", [], None));
        let before = set.clone();
        let mut state = RendererPanelState::new();
        egui::__run_test_ui(|ui| {
            let _ = super::ui(ui, &mut set, LayerAttributes::none(), &mut state);
        });
        assert_eq!(set, before);
    }

    #[test]
    fn a_value_can_be_asked_which_class_it_falls_in() {
        let mut set = fill_set();
        set.set_renderer(categorized_renderer(
            &set.base().clone(),
            "zone",
            [AttrValue::text("a"), AttrValue::text("b")],
        ));
        assert_eq!(class_of_value(&set, &AttrValue::text("a")), Some(0));
        assert_eq!(class_of_value(&set, &AttrValue::text("b")), Some(1));
        assert_eq!(class_of_value(&set, &AttrValue::text("z")), None);
        // A graduated one answers by range, and a Single one has no field.
        set.set_renderer(graduated_renderer(
            &set.base().clone(),
            "pop",
            &[10.0, 20.0],
        ));
        assert_eq!(class_of_value(&set, &AttrValue::Number(5.0)), Some(0));
        assert_eq!(class_of_value(&set, &AttrValue::Number(15.0)), Some(1));
        assert_eq!(class_of_value(&set, &AttrValue::Number(1e9)), Some(1));
        assert_eq!(class_of_value(&set, &AttrValue::Number(f64::NAN)), None);
        set.set_renderer(Renderer::Single);
        assert_eq!(class_of_value(&set, &AttrValue::text("a")), None);
    }

    #[test]
    fn the_legend_rows_resolve_through_the_same_rule_the_map_does() {
        let mut set = fill_set();
        set.set_override(
            GeometryFamily::Point,
            LayerStyle::Circle(oxigis_core::CircleStyle::new(4.0, Color::WHITE)),
        );
        set.set_renderer(categorized_renderer(
            &set.base().clone(),
            "zone",
            [AttrValue::text("a"), AttrValue::text("b")],
        ));
        let rows = legend_rows(&set, GeometryFamily::Polygon);
        assert_eq!(rows.len(), 3, "two classes plus 'everything else'");
        assert_eq!(rows.first().map(|(label, _)| label.as_str()), Some("a"));
        assert_eq!(
            rows.last().map(|(label, _)| label.as_str()),
            Some("Everything else")
        );
        for (index, (_, style)) in rows.iter().enumerate() {
            let class = (index < 2).then_some(index);
            assert_eq!(style, &set.style_for_class(GeometryFamily::Polygon, class));
        }
        // The POINT family's legend keeps circles — the composition rule.
        let points = legend_rows(&set, GeometryFamily::Point);
        for (_, style) in &points {
            assert!(
                matches!(style, LayerStyle::Circle(_)),
                "a class must never erase a family's own symbol: {style:?}",
            );
        }
        // A single-symbol layer has no legend rows at all.
        let plain = fill_set();
        assert!(legend_rows(&plain, GeometryFamily::Polygon).is_empty());
    }

    #[test]
    fn the_notice_is_cleared_when_the_field_moves() {
        let mut state = state_on("zone");
        state.notice = Some("3 classes.".to_string());
        state.set_field("zone");
        assert_eq!(state.notice(), Some("3 classes."), "same field, kept");
        state.set_field("pop");
        assert_eq!(state.notice(), None, "a different field, dropped");
    }
}
