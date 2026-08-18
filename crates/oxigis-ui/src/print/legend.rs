// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The page legend (print v1.7): one row per visible local vector layer,
//! drawn from the very style structs the map painted with.
//!
//! # What a row is
//!
//! A layer's style is a SET — a base style plus per-family overrides (tiles
//! v1.3) — and the page paints one pass per present family. The legend
//! follows that partition exactly: rows are the DISTINCT style slots a
//! layer's present families resolve to, so a layer with no overrides is one
//! row however many geometry families it holds, and a layer whose lines are
//! overridden green while its polygons stay blue is two rows. A slot whose
//! effective style is [`LayerStyle::Symbol`] gets no row at all: symbols are
//! labels, and a label has no swatch.
//!
//! # Where the name comes from
//!
//! [`super::PrintLayer`] carries no layer name — the print snapshot is
//! built from the feature store, not from the project tree — so the name is
//! read from the GeoJSON collection's own `name` member (what `ogr2ogr`,
//! QGIS and most exporters write) and otherwise synthesised from the layer's
//! position and geometry family, which at least says what the row is. Wiring
//! the project's own layer name through is one line inside [`layer_name`]
//! once [`super::PrintLayer`] carries one.
//!
//! # Layout, and why it is decided without a font plan
//!
//! The legend claims the map box's bottom-right corner, the scale bar keeps
//! the bottom-left and the north arrow the top-right. Which rows exist is
//! settled by [`rows`] from geometry alone — a fixed plate width, the height
//! left between the bar and the arrow — because the font plan is built FROM
//! the row labels and cannot also be an input to them. Only the eliding of a
//! label to the text column needs the plan, and that happens at paint time,
//! exactly as the title and the attribution have always done.

use oxigis_core::{Color, LayerStyle, StyleSlot};
use oxigis_render::MapView;
use pdf_writer::{Content, Name};

use super::{
    MapBox, PrintLayer, PrintRequest, TextPlan, class_alpha_name, elide_to_width, north,
    paint::emit_circle, scalebar, to_rgb,
};

/// The legend plate's width, in points.
///
/// FIXED rather than fitted to the longest label: the collision test against
/// the scale bar has to give the same answer in the plan pass and in the
/// paint pass, and a fitted width would depend on the plan the plan pass is
/// still building.
const LEGEND_WIDTH_PT: f32 = 150.0;

/// Padding between the plate's edge and its rows, in points.
pub(super) const LEGEND_PAD_PT: f32 = 5.0;

/// One row's height, in points.
pub(super) const LEGEND_ROW_PT: f32 = 13.0;

/// The swatch's width and height, in points.
const LEGEND_SWATCH_W_PT: f32 = 18.0;
const LEGEND_SWATCH_H_PT: f32 = 9.0;

/// Gap between a swatch and its text, in points.
const LEGEND_SWATCH_GAP_PT: f32 = 6.0;

/// The row text's font size, in points.
const LEGEND_FONT_PT: f32 = 8.0;

/// Inset of the plate from the map box's bottom-right corner, in points.
pub(super) const LEGEND_INSET_PT: f32 = 10.0;

/// Clearance the plate keeps from the scale bar's plate and from the north
/// arrow's, in points.
const LEGEND_CLEARANCE_PT: f32 = 8.0;

/// Hard ceiling on the rows one page shows, whatever the page size: a legend
/// taller than this stops being a legend and starts being a table, and the
/// remaining layers are summed into one `+N more` row.
const MAX_LEGEND_ROWS: usize = 12;

/// Longest layer name the legend will carry into the font plan, in
/// characters — a hostile or machine-generated GeoJSON `name` is not allowed
/// to grow the embedded subset without bound.
const MAX_NAME_CHARS: usize = 120;

/// The text column's width, in points — what a row label is elided to.
const fn label_column_pt() -> f32 {
    LEGEND_WIDTH_PT - 2.0 * LEGEND_PAD_PT - LEGEND_SWATCH_W_PT - LEGEND_SWATCH_GAP_PT
}

/// What one row draws beside its label: the layer's own style, reduced to
/// the three things a swatch can show.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Swatch {
    /// A filled rectangle, outlined when the style outlines its polygons.
    Fill {
        /// The fill colour.
        color: Color,
        /// The outline colour, when the style draws one.
        outline: Option<Color>,
    },
    /// A stroked horizontal line.
    Line {
        /// The stroke colour.
        color: Color,
        /// The stroke width, in points.
        width: f32,
    },
    /// A filled circle, stroked when the style strokes its markers.
    Circle {
        /// The fill colour.
        color: Color,
        /// The marker radius, in points.
        radius: f32,
        /// The stroke colour and width, when the style draws one.
        stroke: Option<(Color, f32)>,
    },
    /// The `+N more` row: a count, no swatch.
    Overflow,
}

/// One legend row.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LegendRow {
    /// The text beside the swatch — the FINAL string, so the plan pass and
    /// the paint pass show the same characters.
    pub label: String,
    /// What the swatch draws.
    pub swatch: Swatch,
    /// The ExtGState the row paints under, so a half-transparent layer's
    /// swatch is half-transparent too. [`None`] for the overflow row.
    pub alpha: Option<String>,
}

/// A layer's name and whether the page had to invent it.
struct LayerName {
    /// The name itself.
    text: String,
    /// Whether it was synthesised from the layer's position — in which case
    /// the row is qualified with its geometry family, so it says something.
    synthesized: bool,
}

/// The rows this page's legend shows, or an empty list when the legend is
/// off, the corner is too small, or the plate would collide with the scale
/// bar.
///
/// The ONE source of truth: `document` plans these strings, the painter draws
/// them, and neither computes a row the other does not know about — which
/// holds because both are given `map_box(&request.options)`, as
/// [`super::pdf_document`] does. A caller of the `page_content*` test seams
/// may pass a different rectangle; the page then simply shows the rows THAT
/// rectangle allows (fewer, or none), never a row with no CIDs behind it,
/// because an unplanned character is dropped rather than mis-drawn.
#[must_use]
pub(super) fn rows(request: &PrintRequest, compose: &MapView, map_box: &MapBox) -> Vec<LegendRow> {
    if !request.options.legend {
        return Vec::new();
    }
    let Some(capacity) = row_capacity(request, compose, map_box) else {
        return Vec::new();
    };
    // Counted before anything is built: a project with ten thousand layers
    // must not make the page allocate ten thousand labels only to throw all
    // but a dozen away. `drawable_slots` allocates at most three entries per
    // layer and frees them each turn, and a classified slot's row count is a
    // multiplication rather than a second walk.
    let total: usize = request
        .layers
        .iter()
        .map(|layer| drawable_slots(layer).len() * rows_per_slot(layer))
        .sum();
    let visible = if total > capacity {
        // One slot goes to the count of what did not fit, so the sheet never
        // silently claims a layer does not exist.
        capacity.saturating_sub(1)
    } else {
        total
    };
    let mut rows = Vec::with_capacity(visible.saturating_add(1));
    'layers: for (index, layer) in request.layers.iter().enumerate() {
        let slots = drawable_slots(layer);
        if slots.is_empty() {
            // A symbol-only layer draws labels, not swatches: no row, and no
            // name looked up for one either.
            continue;
        }
        let name = layer_name(layer, index);
        // A row is qualified by its family when the layer speaks with more
        // than one voice, or when the name is invented and the family is the
        // only real thing the row can say.
        let qualify = slots.len() > 1 || name.synthesized;
        for (slot, family) in slots {
            let base = if qualify {
                format!("{} ({})", name.text, family.label())
            } else {
                name.text.clone()
            };
            // A CLASSIFIED slot is one row per class, read through the very
            // helper the on-screen legend uses
            // ([`crate::renderer_panel::legend_rows`]), so a printed legend
            // cannot disagree with the map about which colour means what.
            // `legend_rows` is empty for an unclassified layer, which then
            // takes the single-row path it always did.
            let classes = crate::renderer_panel::legend_rows(&layer.style, family);
            if classes.is_empty() {
                if rows.len() >= visible {
                    break 'layers;
                }
                rows.push(LegendRow {
                    label: base,
                    swatch: swatch_for(layer.style.effective(family)),
                    alpha: Some(class_alpha_name(index, slot, None)),
                });
                continue;
            }
            for (position, (class_label, style)) in classes.iter().enumerate() {
                if rows.len() >= visible {
                    break 'layers;
                }
                // `legend_rows` puts the classes first and the fallback last,
                // so the last position is the one with no class index.
                let class = (position + 1 < classes.len()).then_some(position);
                rows.push(LegendRow {
                    label: format!("{base}: {class_label}"),
                    swatch: swatch_for(style),
                    alpha: Some(class_alpha_name(index, slot, class)),
                });
            }
        }
    }
    if total > visible {
        rows.push(LegendRow {
            label: format!("+{} more", total - visible),
            swatch: Swatch::Overflow,
            alpha: None,
        });
    }
    rows
}

/// Every string the legend shows — what `document` adds to the font plan.
#[must_use]
pub(super) fn texts(request: &PrintRequest, compose: &MapView, map_box: &MapBox) -> Vec<String> {
    rows(request, compose, map_box)
        .into_iter()
        .map(|row| row.label)
        .collect()
}

/// How many rows ONE drawable slot of `layer` is worth: one for an
/// unclassified layer, one per class plus the fallback for a classified one.
///
/// Read from the renderer rather than from a second walk of
/// [`crate::renderer_panel::legend_rows`], so the pre-count that decides
/// `+N more` costs nothing and can never disagree with the rows themselves —
/// both are `class_count() + 1` when there are classes at all.
fn rows_per_slot(layer: &PrintLayer) -> usize {
    match layer.style.class_count() {
        0 => 1,
        classes => classes + 1,
    }
}

/// The DISTINCT style slots one layer's present families resolve to, with a
/// representative family each — the rows that layer is worth, before any of
/// them is given a name.
///
/// At most three entries, and never a [`LayerStyle::Symbol`] slot: symbols
/// are labels, and a label has no swatch.
fn drawable_slots(layer: &PrintLayer) -> Vec<(StyleSlot, oxigis_core::GeometryFamily)> {
    let mut slots: Vec<(StyleSlot, oxigis_core::GeometryFamily)> = Vec::new();
    for family in layer.families.iter() {
        if matches!(layer.style.effective(family), LayerStyle::Symbol(_)) {
            continue;
        }
        let slot = layer.style.slot_of(family);
        if slots.iter().any(|(known, _)| *known == slot) {
            continue;
        }
        slots.push((slot, family));
    }
    slots
}

/// A layer's display name: the GeoJSON collection's own `name` member when
/// the file carried one, else a synthesised `Layer N`.
///
/// The `name` member is what `ogr2ogr`, QGIS and most exporters write at the
/// collection level, and `local_vector` deserialises the whole collection
/// (foreign members included), so a GeoJSON layer usually names itself. A
/// Shapefile, GeoPackage or GeoParquet layer is assembled feature by feature
/// and never can — hence the qualified fallback.
fn layer_name(layer: &PrintLayer, index: usize) -> LayerName {
    // The PROJECT's name first: it is what the layer panel shows and what a
    // rename changes, so a page that ignored it would legend a layer under a
    // name the user has not seen since they imported the file.
    if !layer.name.trim().is_empty() {
        return LayerName {
            text: layer.name.chars().take(MAX_NAME_CHARS).collect(),
            synthesized: false,
        };
    }
    let named = layer
        .features
        .foreign_members
        .as_ref()
        .and_then(|members| members.get("name"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty());
    match named {
        Some(name) => LayerName {
            text: name.chars().take(MAX_NAME_CHARS).collect(),
            synthesized: false,
        },
        None => LayerName {
            text: format!("Layer {}", index + 1),
            synthesized: true,
        },
    }
}

/// One resolved style as a swatch.
pub(super) fn swatch_for(style: &LayerStyle) -> Swatch {
    match style {
        LayerStyle::Fill(fill) => Swatch::Fill {
            color: fill.color,
            outline: fill.outline_color,
        },
        LayerStyle::Line(line) => Swatch::Line {
            color: line.color,
            // Clamped for the swatch only: a 12 pt motorway casing would fill
            // the row, and a 0.05 pt hairline would vanish.
            width: line.width().clamp(0.4, 3.0),
        },
        LayerStyle::Circle(circle) => Swatch::Circle {
            color: circle.color,
            radius: circle.radius().clamp(1.0, LEGEND_SWATCH_H_PT / 2.0),
            stroke: circle
                .stroke_color
                .map(|color| (color, circle.stroke_width().clamp(0.3, 1.5))),
        },
        // Unreachable: `drawable_slots` filters symbol slots out before this
        // is ever asked about one.
        LayerStyle::Symbol(_) => Swatch::Overflow,
    }
}

/// How many rows the corner has room for, or [`None`] when the legend cannot
/// be placed at all.
fn row_capacity(request: &PrintRequest, compose: &MapView, map_box: &MapBox) -> Option<usize> {
    let plate_x = map_box.x + map_box.width - LEGEND_INSET_PT - LEGEND_WIDTH_PT;
    if plate_x < map_box.x {
        return None;
    }
    // The scale bar owns the other corner; its plate's width is measured
    // WITHOUT a plan (an over-estimate), so this test answers the same in the
    // plan pass and in the paint pass.
    let bar_right = scalebar::plate_right_estimate_pt(&request.options, compose, map_box);
    if plate_x < bar_right + LEGEND_CLEARANCE_PT {
        return None;
    }
    // The north arrow owns the top of the same column.
    let ceiling = match north::plate_box(&request.options, map_box) {
        Some(arrow) => arrow.y - LEGEND_CLEARANCE_PT,
        None => map_box.y + map_box.height - LEGEND_INSET_PT,
    };
    let room = ceiling - (map_box.y + LEGEND_INSET_PT) - 2.0 * LEGEND_PAD_PT;
    if !room.is_finite() || room < LEGEND_ROW_PT {
        return None;
    }
    let fits = (room / LEGEND_ROW_PT).floor().min(MAX_LEGEND_ROWS as f32) as usize;
    (fits > 0).then_some(fits)
}

/// The plate `rows` occupy, in page points.
pub(super) fn plate_box(rows: usize, map_box: &MapBox) -> MapBox {
    let height = rows as f32 * LEGEND_ROW_PT + 2.0 * LEGEND_PAD_PT;
    MapBox {
        x: map_box.x + map_box.width - LEGEND_INSET_PT - LEGEND_WIDTH_PT,
        y: map_box.y + LEGEND_INSET_PT,
        width: LEGEND_WIDTH_PT,
        height,
    }
}

/// Draws the legend: a white plate, then one swatch-and-name row per
/// [`rows`] entry, top-down.
pub(super) fn paint(
    content: &mut Content,
    request: &PrintRequest,
    compose: &MapView,
    map_box: &MapBox,
    plan: Option<&TextPlan>,
) {
    let rows = rows(request, compose, map_box);
    if rows.is_empty() {
        return;
    }
    let plate = plate_box(rows.len(), map_box);
    content.save_state();
    content.set_fill_rgb(1.0, 1.0, 1.0);
    content.rect(plate.x, plate.y, plate.width, plate.height);
    content.fill_nonzero();
    content.set_stroke_rgb(0.6, 0.6, 0.6);
    content.set_line_width(0.5);
    content.rect(plate.x, plate.y, plate.width, plate.height);
    content.stroke();
    for (index, row) in rows.iter().enumerate() {
        let row_top = plate.y + plate.height - LEGEND_PAD_PT - index as f32 * LEGEND_ROW_PT;
        let centre_y = row_top - LEGEND_ROW_PT / 2.0;
        let swatch_x = plate.x + LEGEND_PAD_PT;
        content.save_state();
        if let Some(alpha) = row.alpha.as_ref() {
            content.set_parameters(Name(alpha.as_bytes()));
        }
        paint_swatch(content, row.swatch, swatch_x, centre_y);
        content.restore_state();
        content.set_fill_rgb(0.0, 0.0, 0.0);
        super::show_line(
            content,
            plan,
            swatch_x + LEGEND_SWATCH_W_PT + LEGEND_SWATCH_GAP_PT,
            centre_y - LEGEND_FONT_PT * 0.36,
            LEGEND_FONT_PT,
            &elide_to_width(plan, &row.label, LEGEND_FONT_PT, label_column_pt()),
        );
    }
    content.restore_state();
}

/// Draws one swatch, vertically centred on `centre_y`.
fn paint_swatch(content: &mut Content, swatch: Swatch, x: f32, centre_y: f32) {
    match swatch {
        Swatch::Fill { color, outline } => {
            let rgb = to_rgb(color);
            content.set_fill_rgb(rgb[0], rgb[1], rgb[2]);
            // A white or very pale fill needs an edge or the swatch is an
            // invisible hole in the plate; the style's own outline is used
            // when it has one, a neutral hairline otherwise.
            let edge = outline.map_or([0.35, 0.35, 0.35], to_rgb);
            content.set_stroke_rgb(edge[0], edge[1], edge[2]);
            content.set_line_width(0.6);
            content.rect(
                x,
                centre_y - LEGEND_SWATCH_H_PT / 2.0,
                LEGEND_SWATCH_W_PT,
                LEGEND_SWATCH_H_PT,
            );
            content.fill_nonzero_and_stroke();
        }
        Swatch::Line { color, width } => {
            let rgb = to_rgb(color);
            content.set_stroke_rgb(rgb[0], rgb[1], rgb[2]);
            content.set_line_width(width);
            content.move_to(x, centre_y);
            content.line_to(x + LEGEND_SWATCH_W_PT, centre_y);
            content.stroke();
        }
        Swatch::Circle {
            color,
            radius,
            stroke,
        } => {
            let rgb = to_rgb(color);
            content.set_fill_rgb(rgb[0], rgb[1], rgb[2]);
            emit_circle(content, x + LEGEND_SWATCH_W_PT / 2.0, centre_y, radius);
            match stroke {
                Some((color, width)) => {
                    let rgb = to_rgb(color);
                    content.set_stroke_rgb(rgb[0], rgb[1], rgb[2]);
                    content.set_line_width(width);
                    content.fill_nonzero_and_stroke();
                }
                None => {
                    content.fill_nonzero();
                }
            }
        }
        Swatch::Overflow => {}
    }
}
