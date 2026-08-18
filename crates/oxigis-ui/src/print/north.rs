// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The north arrow (print v1.7): a small compass needle in the map box's
//! top-right corner.
//!
//! North is straight up on every page this export produces — the raster, the
//! vector overlay and the labels all come from one Web Mercator view with no
//! rotation — so the arrow is a fixed glyph rather than a computed bearing,
//! and it is drawn as PDF paths rather than as a font character so it stays
//! sharp at any print zoom and needs no glyph in any subset.
//!
//! The shape is the conventional half-filled needle: the western half solid,
//! the eastern half outlined, which reads as an arrow at 26 pt and as an
//! arrow at 200 % magnification. Only the `N` is text.

use pdf_writer::Content;

use super::{MapBox, PrintOptions, TextPlan, line_width_pt, show_line};

/// The needle's width at its base, in points.
const ARROW_WIDTH_PT: f32 = 13.0;

/// The needle's height, in points.
const ARROW_HEIGHT_PT: f32 = 24.0;

/// How far up the needle its rear notch cuts, as a fraction of the height.
const ARROW_NOTCH: f32 = 0.26;

/// The `N` label's font size, in points.
const NORTH_FONT_PT: f32 = 9.0;

/// Gap between the `N` and the needle's tip, in points.
const NORTH_LABEL_GAP_PT: f32 = 3.0;

/// Inset of the arrow's plate from the map box's top-right corner, in points.
const NORTH_INSET_PT: f32 = 10.0;

/// Padding between the plate's edge and the glyph, in points.
const NORTH_PLATE_PAD_PT: f32 = 4.0;

/// The letter over the needle — a constant so the plan and the painter cannot
/// spell it differently.
pub(super) const NORTH_LABEL: &str = "N";

/// The whole arrow's height including its label, in points.
pub(super) const NORTH_BLOCK_HEIGHT_PT: f32 =
    ARROW_HEIGHT_PT + NORTH_LABEL_GAP_PT + NORTH_FONT_PT + 2.0 * NORTH_PLATE_PAD_PT;

/// The smallest map box that still gets an arrow: below this the glyph would
/// take a visible share of the map rather than annotate it.
const MIN_MAP_BOX_FOR_ARROW_PT: f32 = 4.0 * NORTH_BLOCK_HEIGHT_PT;

/// The plate the arrow occupies, or [`None`] when it is switched off or the
/// map box is too small to spare the corner.
///
/// The legend reads this to keep out of the arrow's way, so both furniture
/// pieces answer to ONE geometry function rather than to two constants that
/// could drift.
#[must_use]
pub(super) fn plate_box(options: &PrintOptions, map_box: &MapBox) -> Option<MapBox> {
    if !options.north_arrow {
        return None;
    }
    if map_box.height < MIN_MAP_BOX_FOR_ARROW_PT || map_box.width < MIN_MAP_BOX_FOR_ARROW_PT {
        return None;
    }
    let width = ARROW_WIDTH_PT + 2.0 * NORTH_PLATE_PAD_PT;
    Some(MapBox {
        x: map_box.x + map_box.width - NORTH_INSET_PT - width,
        y: map_box.y + map_box.height - NORTH_INSET_PT - NORTH_BLOCK_HEIGHT_PT,
        width,
        height: NORTH_BLOCK_HEIGHT_PT,
    })
}

/// Draws the north arrow: a white plate, the half-filled needle, and the `N`
/// centred over its tip.
pub(super) fn paint(
    content: &mut Content,
    options: &PrintOptions,
    map_box: &MapBox,
    plan: Option<&TextPlan>,
) {
    let Some(plate) = plate_box(options, map_box) else {
        return;
    };
    let centre_x = plate.x + plate.width / 2.0;
    let base_y = plate.y + NORTH_PLATE_PAD_PT;
    let tip_y = base_y + ARROW_HEIGHT_PT;
    let notch_y = base_y + ARROW_HEIGHT_PT * ARROW_NOTCH;
    let half = ARROW_WIDTH_PT / 2.0;
    content.save_state();
    content.set_fill_rgb(1.0, 1.0, 1.0);
    content.rect(plate.x, plate.y, plate.width, plate.height);
    content.fill_nonzero();
    content.set_stroke_rgb(0.0, 0.0, 0.0);
    content.set_line_width(0.6);
    // Western half: solid.
    content.set_fill_rgb(0.0, 0.0, 0.0);
    content.move_to(centre_x, tip_y);
    content.line_to(centre_x - half, base_y);
    content.line_to(centre_x, notch_y);
    content.close_path();
    content.fill_nonzero_and_stroke();
    // Eastern half: outlined, so the needle reads as a needle and not as a
    // triangle.
    content.set_fill_rgb(1.0, 1.0, 1.0);
    content.move_to(centre_x, tip_y);
    content.line_to(centre_x + half, base_y);
    content.line_to(centre_x, notch_y);
    content.close_path();
    content.fill_nonzero_and_stroke();
    // The letter, centred on the needle's axis.
    content.set_fill_rgb(0.0, 0.0, 0.0);
    let label_w = line_width_pt(plan, NORTH_LABEL, NORTH_FONT_PT);
    show_line(
        content,
        plan,
        centre_x - label_w / 2.0,
        tip_y + NORTH_LABEL_GAP_PT,
        NORTH_FONT_PT,
        NORTH_LABEL,
    );
    content.restore_state();
}
