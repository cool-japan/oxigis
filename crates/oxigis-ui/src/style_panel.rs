//! Style editor panel: edits the [`LayerStyleSet`] (a shared base style
//! plus per-family overrides, tiles v1.3) for the selected layer.
//!
//! A layer's style lives in `Project::styles`, keyed by
//! [`oxigis_core::LayerId`] and absent by default, so this panel has two
//! modes: no style yet (offer to create one of the four kinds) and an
//! existing set (edit its slots in place). Creating/removing styles and
//! overrides is reported as a [`StyleAction`] rather than applied directly,
//! since the caller owns the map entry; in-place field edits bind straight
//! to the `&mut LayerStyleSet` the caller passes in.
//!
//! # The family row
//!
//! For the overwhelmingly common single-family layer the panel is exactly
//! its pre-v1.3 self — zero pixels change. Only when the dataset (or the
//! stored overrides) span more than one geometry family does an
//! `Applies to:` row appear, switching the editor between the shared base
//! and one family's override. Creating an override SEEDS IT FROM THE BASE,
//! so opting in never changes the picture — the user edits from a known
//! state. The family kind picker offers Fill/Line/Circle only, never
//! Symbol: labels are a layer-wide concern living in the base slot (a
//! per-family Symbol is provably a no-op or a synonym — the properties
//! carrier puts a source feature's text on exactly one family).

use egui::{Color32, Ui};
use oxigis_core::{
    CircleStyle, Color, FamilySet, FillStyle, GeometryFamily, LabelOrientation, LabelWeight,
    LayerId, LayerStyle, LayerStyleSet, LineStyle, StyleSlot, SymbolStyle,
};

/// Which kind of [`LayerStyle`] to create, for the "no style yet" picker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleKind {
    /// Polygon fill styling.
    Fill,
    /// Line styling.
    Line,
    /// Point-as-circle styling.
    Circle,
    /// Text label styling.
    Symbol,
}

impl StyleKind {
    /// All four kinds, in the order the picker buttons are drawn.
    pub const ALL: [StyleKind; 4] = [
        StyleKind::Fill,
        StyleKind::Line,
        StyleKind::Circle,
        StyleKind::Symbol,
    ];

    /// The kinds a FAMILY override may take — never `Symbol` (see the
    /// module docs).
    pub const GEOMETRY_KINDS: [StyleKind; 3] =
        [StyleKind::Fill, StyleKind::Line, StyleKind::Circle];

    /// Short label for the picker button.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            StyleKind::Fill => "Fill",
            StyleKind::Line => "Line",
            StyleKind::Circle => "Circle",
            StyleKind::Symbol => "Symbol",
        }
    }

    /// A reasonable default style of this kind, matching each style's own
    /// `new`/`Default` starting point.
    #[must_use]
    pub fn default_style(self) -> LayerStyle {
        match self {
            StyleKind::Fill => LayerStyle::Fill(FillStyle::new(Color::from_rgb(80, 140, 200))),
            StyleKind::Line => LayerStyle::Line(LineStyle::new(Color::from_rgb(60, 60, 60), 2.0)),
            StyleKind::Circle => {
                LayerStyle::Circle(CircleStyle::new(5.0, Color::from_rgb(220, 80, 60)))
            }
            StyleKind::Symbol => LayerStyle::Symbol(SymbolStyle::new("name")),
        }
    }
}

/// A user-requested change to whether/which style (or family override)
/// exists for the selected layer. Field edits within an existing style are
/// applied in place by [`ui`] and don't need an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleAction {
    /// Create a base style of the given kind for the selected layer.
    Create(StyleKind),
    /// Remove the selected layer's WHOLE style entry (overrides included).
    Remove,
    /// Give one family its own override, seeded from a clone of the base.
    CreateFamily(GeometryFamily),
    /// Switch an existing override to a different kind (its default
    /// colours — consistent with the base's only kind-change path,
    /// Remove → Create; colour-preserving conversion is a non-goal).
    SetFamilyKind(GeometryFamily, StyleKind),
    /// Drop one family's override — back to the shared base style.
    RemoveFamily(GeometryFamily),
}

/// The panel's cross-frame state: which layer it was last showing and which
/// slot of the set the editor is bound to. `Copy` and tiny, so the app
/// copies it out, hands `&mut` in, and writes it back — no split-borrow
/// gymnastics against `Project::styles`.
///
/// The RENDERER editor's own state ([`crate::renderer_panel::RendererPanelState`])
/// is deliberately NOT folded in here: it carries a `String` field name, and
/// making this type non-`Copy` would push a clone onto every caller of
/// [`StylePanelState::slot`]. The app owns the two side by side.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StylePanelState {
    layer: Option<LayerId>,
    slot: StyleSlot,
}

impl StylePanelState {
    /// The slot the editor is currently bound to — what the undo seam's
    /// coalesce key discriminates on (a polygon-fill drag and a line-width
    /// drag are structurally two undo steps).
    #[must_use]
    pub fn slot(self) -> StyleSlot {
        self.slot
    }

    /// Which layer the panel was last drawn for.
    #[must_use]
    pub fn layer(self) -> Option<LayerId> {
        self.layer
    }

    /// Resets the slot to `Base` when the selection changed, or when the
    /// selected family stopped being editable (its geometry vanished AND it
    /// carries no override).
    fn retarget(&mut self, layer: Option<LayerId>, editable: FamilySet) {
        if self.layer != layer {
            self.layer = layer;
            self.slot = StyleSlot::Base;
        }
        if let StyleSlot::Family(family) = self.slot
            && !editable.contains(family)
        {
            self.slot = StyleSlot::Base;
        }
    }
}

/// Draws the style editor for `style` (`None` when the selected layer has no
/// style entry yet, or no layer is selected at all), returning a
/// [`StyleAction`] if the user asked to change the entry's shape this frame.
///
/// `families` is what the layer's data actually draws
/// ([`crate::local_input::LocalInputState::families`] — empty for provider
/// layers, so they never grow the family row).
///
/// The style-only entry point, kept for callers that have no renderer state
/// to hand: the RENDERER section (thematic v1.6) is drawn by
/// [`ui_with_renderer`]. This one draws exactly the pre-v1.6 panel.
pub fn ui(
    ui: &mut Ui,
    layer: Option<LayerId>,
    style: Option<&mut LayerStyleSet>,
    families: FamilySet,
    state: &mut StylePanelState,
) -> Option<StyleAction> {
    ui_with_renderer(ui, layer, style, families, state, None).0
}

/// [`ui`] plus the renderer section (thematic v1.6).
///
/// `renderer` is the editor's cross-frame state and the layer's attribute
/// view; [`None`] draws the pre-v1.6 panel exactly. The second half of the
/// return value is the renderer editor's own event — see
/// [`crate::renderer_panel::RendererEvent`].
///
/// # Why the renderer section is drawn last
///
/// A renderer decides which style a feature gets; the style editor above it
/// decides what those styles ARE. Reading the panel top to bottom therefore
/// reads the model outward-in — base style, family overrides, then the
/// classification over both — and a user who never scrolls past the first
/// section sees precisely the panel they saw before this feature existed.
pub fn ui_with_renderer(
    ui: &mut Ui,
    layer: Option<LayerId>,
    style: Option<&mut LayerStyleSet>,
    families: FamilySet,
    state: &mut StylePanelState,
    renderer: Option<(
        &mut crate::renderer_panel::RendererPanelState,
        crate::renderer_panel::LayerAttributes<'_>,
    )>,
) -> (
    Option<StyleAction>,
    Option<crate::renderer_panel::RendererEvent>,
) {
    ui.heading("Style");
    ui.separator();

    // Editable = drawn families plus any that still carry an override, so
    // an override whose geometry has since been deleted stays reachable
    // and removable.
    let editable = families.union(
        style
            .as_ref()
            .map(|set| set.overridden())
            .unwrap_or_default(),
    );
    state.retarget(layer, editable);

    let Some(set) = style else {
        ui.weak("No style set for this layer.");
        ui.label("Create one:");
        let mut action = None;
        ui.horizontal(|ui| {
            for kind in StyleKind::ALL {
                if ui.button(kind.label()).clicked() {
                    action = Some(StyleAction::Create(kind));
                }
            }
        });
        return (action, None);
    };

    let mut action = None;
    ui.horizontal(|ui| {
        ui.label(kind_label(set.base()));
        if ui.button("Remove style").clicked() {
            action = Some(StyleAction::Remove);
        }
    });

    // The family row appears only when it is meaningful; a single-family
    // layer's panel is byte-for-byte its pre-v1.3 self.
    if editable.is_mixed() {
        ui.horizontal(|ui| {
            ui.label("Applies to:");
            if ui
                .selectable_label(state.slot == StyleSlot::Base, "All")
                .clicked()
            {
                state.slot = StyleSlot::Base;
            }
            for family in editable.iter() {
                let selected = state.slot == StyleSlot::Family(family);
                if ui.selectable_label(selected, family.label()).clicked() {
                    state.slot = StyleSlot::Family(family);
                }
            }
        });
    }
    ui.separator();

    match state.slot {
        StyleSlot::Base => edit_style(ui, set.base_mut()),
        StyleSlot::Family(family) => match set.slot_mut(StyleSlot::Family(family)) {
            Some(style) => {
                ui.horizontal(|ui| {
                    for kind in StyleKind::GEOMETRY_KINDS {
                        let selected = kind_of(style) == kind;
                        if ui.selectable_label(selected, kind.label()).clicked() && !selected {
                            action = Some(StyleAction::SetFamilyKind(family, kind));
                        }
                    }
                    if ui.button("Use the shared style").clicked() {
                        action = Some(StyleAction::RemoveFamily(family));
                    }
                });
                edit_style(ui, style);
            }
            None => {
                ui.weak(format!(
                    "{} use the shared {}.",
                    family.label(),
                    kind_label(set.base())
                ));
                if ui
                    .button(format!(
                        "Style {} separately",
                        family.label().to_lowercase()
                    ))
                    .clicked()
                {
                    action = Some(StyleAction::CreateFamily(family));
                }
            }
        },
    }

    // The renderer section, last: see this function's docs on the order.
    let event = renderer.and_then(|(state, attributes)| {
        ui.separator();
        crate::renderer_panel::ui(ui, set, attributes, state)
    });

    (action, event)
}

/// Dispatches one style value to its field editor.
fn edit_style(ui: &mut Ui, style: &mut LayerStyle) {
    match style {
        LayerStyle::Fill(fill) => edit_fill(ui, fill),
        LayerStyle::Line(line) => edit_line(ui, line),
        LayerStyle::Circle(circle) => edit_circle(ui, circle),
        LayerStyle::Symbol(symbol) => edit_symbol(ui, symbol),
    }
}

fn kind_of(style: &LayerStyle) -> StyleKind {
    match style {
        LayerStyle::Fill(_) => StyleKind::Fill,
        LayerStyle::Line(_) => StyleKind::Line,
        LayerStyle::Circle(_) => StyleKind::Circle,
        LayerStyle::Symbol(_) => StyleKind::Symbol,
    }
}

fn kind_label(style: &LayerStyle) -> &'static str {
    match style {
        LayerStyle::Fill(_) => "Fill style",
        LayerStyle::Line(_) => "Line style",
        LayerStyle::Circle(_) => "Circle style",
        LayerStyle::Symbol(_) => "Symbol style",
    }
}

fn edit_fill(ui: &mut Ui, fill: &mut FillStyle) {
    edit_color(ui, "Fill color", &mut fill.color);
    let mut opacity = fill.opacity();
    if ui
        .add(egui::Slider::new(&mut opacity, 0.0..=1.0).text("Opacity"))
        .changed()
    {
        fill.set_opacity(opacity);
    }
    edit_optional_color(ui, "Outline color", &mut fill.outline_color);
}

fn edit_line(ui: &mut Ui, line: &mut LineStyle) {
    edit_color(ui, "Color", &mut line.color);
    let mut width = line.width();
    if ui
        .add(egui::Slider::new(&mut width, 0.0..=20.0).text("Width"))
        .changed()
    {
        line.set_width(width);
    }
    let mut opacity = line.opacity();
    if ui
        .add(egui::Slider::new(&mut opacity, 0.0..=1.0).text("Opacity"))
        .changed()
    {
        line.set_opacity(opacity);
    }
}

fn edit_circle(ui: &mut Ui, circle: &mut CircleStyle) {
    edit_color(ui, "Fill color", &mut circle.color);
    let mut radius = circle.radius();
    if ui
        .add(egui::Slider::new(&mut radius, 0.0..=50.0).text("Radius"))
        .changed()
    {
        circle.set_radius(radius);
    }
    let mut stroke_width = circle.stroke_width();
    if ui
        .add(egui::Slider::new(&mut stroke_width, 0.0..=20.0).text("Stroke width"))
        .changed()
    {
        circle.set_stroke_width(stroke_width);
    }
    edit_optional_color(ui, "Stroke color", &mut circle.stroke_color);
    let mut opacity = circle.opacity();
    if ui
        .add(egui::Slider::new(&mut opacity, 0.0..=1.0).text("Opacity"))
        .changed()
    {
        circle.set_opacity(opacity);
    }
}

fn edit_symbol(ui: &mut Ui, symbol: &mut SymbolStyle) {
    let mut text_field = symbol.text_field.clone().unwrap_or_default();
    ui.horizontal(|ui| {
        ui.label("Text field");
        if ui.text_edit_singleline(&mut text_field).changed() {
            symbol.text_field = if text_field.is_empty() {
                None
            } else {
                Some(text_field)
            };
        }
    });
    edit_color(ui, "Text color", &mut symbol.text_color);
    let mut text_size = symbol.text_size();
    if ui
        .add(egui::Slider::new(&mut text_size, 0.0..=64.0).text("Text size"))
        .changed()
    {
        symbol.set_text_size(text_size);
    }
    // A DISCRETE gesture — one click, one undo step. The sliders around it
    // coalesce because a drag is one intent spread over many frames; a
    // checkbox has no such frames, so it deliberately gets no
    // `CoalesceField` of its own (print/text v1.4, D-W2).
    let mut bold = symbol.weight() == LabelWeight::Bold;
    if ui.checkbox(&mut bold, "Bold").changed() {
        symbol.set_weight(if bold {
            LabelWeight::Bold
        } else {
            LabelWeight::Regular
        });
    }
    // Same reasoning, same shape: one click, one undo step, no new
    // `CoalesceField` (print/text v1.5, D-A8). A label the renderer cannot
    // stack simply draws horizontally, so this box can never blank a label.
    let mut vertical = symbol.orientation() == LabelOrientation::Vertical;
    if ui
        .checkbox(&mut vertical, LabelOrientation::Vertical.label())
        .changed()
    {
        symbol.set_orientation(if vertical {
            LabelOrientation::Vertical
        } else {
            LabelOrientation::Horizontal
        });
    }
    edit_optional_color(ui, "Halo color", &mut symbol.halo_color);
    let mut halo_width = symbol.halo_width();
    if ui
        .add(egui::Slider::new(&mut halo_width, 0.0..=10.0).text("Halo width"))
        .changed()
    {
        symbol.set_halo_width(halo_width);
    }
}

/// A bare colour button bound to a style's MAIN colour (a fill's fill, a
/// line's stroke, a circle's disc, a label's text), with no label of its own.
///
/// The renderer editor's class rows use this: a class row is one line and its
/// text is the class LABEL, so the colour has to be a button rather than a
/// labelled row. Shared from here rather than copied there so the two editors
/// write colours through one function.
pub fn edit_style_color(ui: &mut Ui, style: &mut LayerStyle) {
    let mut color = oxigis_core::style_color(style);
    let mut c32 = Color32::from_rgba_unmultiplied(color.r, color.g, color.b, color.a);
    if ui.color_edit_button_srgba(&mut c32).changed() {
        color = Color::from_rgba(c32.r(), c32.g(), c32.b(), c32.a());
        *style = oxigis_core::recolor_style(style, color);
    }
}

/// A required color field, bound to an egui color-edit button.
fn edit_color(ui: &mut Ui, label: &str, color: &mut Color) {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut c32 = Color32::from_rgba_unmultiplied(color.r, color.g, color.b, color.a);
        if ui.color_edit_button_srgba(&mut c32).changed() {
            *color = Color::from_rgba(c32.r(), c32.g(), c32.b(), c32.a());
        }
    });
}

/// An optional color field: a checkbox to enable/disable it, plus a
/// color-edit button shown only while enabled.
fn edit_optional_color(ui: &mut Ui, label: &str, color: &mut Option<Color>) {
    ui.horizontal(|ui| {
        let mut enabled = color.is_some();
        if ui.checkbox(&mut enabled, label).changed() {
            *color = if enabled {
                Some(color.unwrap_or(Color::BLACK))
            } else {
                None
            };
        }
        if let Some(c) = color.as_mut() {
            let mut c32 = Color32::from_rgba_unmultiplied(c.r, c.g, c.b, c.a);
            if ui.color_edit_button_srgba(&mut c32).changed() {
                *c = Color::from_rgba(c32.r(), c32.g(), c32.b(), c32.a());
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_style_kind_default_style_has_the_matching_tag() {
        assert!(matches!(
            StyleKind::Fill.default_style(),
            LayerStyle::Fill(_)
        ));
        assert!(matches!(
            StyleKind::Line.default_style(),
            LayerStyle::Line(_)
        ));
        assert!(matches!(
            StyleKind::Circle.default_style(),
            LayerStyle::Circle(_)
        ));
        assert!(matches!(
            StyleKind::Symbol.default_style(),
            LayerStyle::Symbol(_)
        ));
    }

    #[test]
    fn ui_with_no_style_offers_the_picker_and_no_action_without_interaction() {
        egui::__run_test_ui(|ui| {
            let mut state = StylePanelState::default();
            let action = self::ui(ui, None, None, FamilySet::default(), &mut state);
            assert_eq!(action, None);
        });
    }

    #[test]
    fn ui_with_an_existing_style_does_not_panic_and_does_not_mutate_without_interaction() {
        let mut style = LayerStyleSet::new(StyleKind::Fill.default_style());
        let before = style.clone();
        egui::__run_test_ui(|ui| {
            let mut state = StylePanelState::default();
            let action = self::ui(ui, None, Some(&mut style), FamilySet::default(), &mut state);
            assert_eq!(action, None);
        });
        assert_eq!(style, before);
    }

    #[test]
    fn a_single_family_layer_shows_no_family_row_and_a_mixed_one_does_not_panic() {
        // Single family: the panel is its pre-v1.3 self (no family row is
        // provable only visually; here we pin that it draws and reports no
        // action).
        let mut single = FamilySet::default();
        single.insert(GeometryFamily::Polygon);
        let mut style = LayerStyleSet::new(StyleKind::Fill.default_style());
        egui::__run_test_ui(|ui| {
            let mut state = StylePanelState::default();
            assert_eq!(
                self::ui(ui, None, Some(&mut style), single, &mut state),
                None
            );
        });

        // Mixed: the family row appears; still no action without a click.
        let mut mixed = single;
        mixed.insert(GeometryFamily::Line);
        egui::__run_test_ui(|ui| {
            let mut state = StylePanelState::default();
            assert_eq!(
                self::ui(ui, None, Some(&mut style), mixed, &mut state),
                None
            );
        });
    }

    #[test]
    fn a_family_with_an_override_but_no_geometry_stays_editable() {
        let mut style = LayerStyleSet::new(StyleKind::Fill.default_style());
        style.set_override(GeometryFamily::Point, StyleKind::Circle.default_style());
        // The data lost its points, but the override keeps the slot
        // reachable so it can be removed.
        let mut families = FamilySet::default();
        families.insert(GeometryFamily::Polygon);
        let editable = families.union(style.overridden());
        assert!(editable.contains(GeometryFamily::Point));

        let mut state = StylePanelState {
            slot: StyleSlot::Family(GeometryFamily::Point),
            ..Default::default()
        };
        state.retarget(None, editable);
        assert_eq!(state.slot(), StyleSlot::Family(GeometryFamily::Point));
    }

    #[test]
    fn the_symbol_editor_draws_the_bold_toggle_without_changing_the_weight() {
        // The panel must never move the weight on its own — a style opened
        // and closed re-saves byte-identically (D-W1's floor).
        let mut style = LayerStyleSet::new(StyleKind::Symbol.default_style());
        let before = style.clone();
        egui::__run_test_ui(|ui| {
            let mut state = StylePanelState::default();
            assert_eq!(
                self::ui(ui, None, Some(&mut style), FamilySet::default(), &mut state),
                None
            );
        });
        assert_eq!(style, before);
        let LayerStyle::Symbol(symbol) = style.base() else {
            panic!("a symbol base");
        };
        assert_eq!(symbol.weight(), LabelWeight::Regular);
    }

    #[test]
    fn retarget_resets_the_slot_when_the_selection_changes() {
        let mut editable = FamilySet::default();
        editable.insert(GeometryFamily::Polygon);
        editable.insert(GeometryFamily::Line);
        let mut state = StylePanelState::default();
        state.retarget(None, editable);
        state.slot = StyleSlot::Family(GeometryFamily::Line);
        // Same layer: the slot survives.
        state.retarget(None, editable);
        assert_eq!(state.slot(), StyleSlot::Family(GeometryFamily::Line));
        // A vanished family resets to Base.
        let mut narrower = FamilySet::default();
        narrower.insert(GeometryFamily::Polygon);
        state.slot = StyleSlot::Family(GeometryFamily::Line);
        state.retarget(None, narrower);
        assert_eq!(state.slot(), StyleSlot::Base);
    }

    #[test]
    fn the_symbol_editor_draws_the_vertical_toggle_without_changing_the_orientation() {
        // The panel must never move the orientation on its own — a style
        // opened and closed re-saves byte-identically (D-A8's floor, the twin
        // of the bold one above).
        let mut style = LayerStyleSet::new(StyleKind::Symbol.default_style());
        let before = style.clone();
        egui::__run_test_ui(|ui| {
            let mut state = StylePanelState::default();
            assert_eq!(
                self::ui(ui, None, Some(&mut style), FamilySet::default(), &mut state),
                None
            );
        });
        assert_eq!(style, before);
        let LayerStyle::Symbol(symbol) = style.base() else {
            panic!("a symbol base");
        };
        assert_eq!(symbol.orientation(), LabelOrientation::Horizontal);
        assert_eq!(symbol.weight(), LabelWeight::Regular);
        // And a style that ASKS for vertical keeps it across a draw.
        let LayerStyle::Symbol(mut symbol) = style.base().clone() else {
            panic!("a symbol base");
        };
        symbol.set_orientation(LabelOrientation::Vertical);
        let mut vertical = LayerStyleSet::new(LayerStyle::Symbol(symbol));
        let asked = vertical.clone();
        egui::__run_test_ui(|ui| {
            let mut state = StylePanelState::default();
            let _ = self::ui(
                ui,
                None,
                Some(&mut vertical),
                FamilySet::default(),
                &mut state,
            );
        });
        assert_eq!(vertical, asked, "drawing is not editing");
    }
}
