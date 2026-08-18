// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The page's scale bar (print v1.7) — the maths and the plate.
//!
//! Moved here out of `print/mod.rs`, which had reached its line budget, and
//! then grown into what a printed map sheet is actually expected to carry:
//!
//! * the **1-2-5 × 10ⁿ round distance** the bar has always spanned, still
//!   corrected for the Web Mercator scale factor at the view centre's
//!   latitude (`cos φ`) — one honest number for the whole page, because at
//!   page scale the latitude variation across the map box is invisible;
//! * **round-number segmentation**: the bar is divided into segments that
//!   are themselves round (a 1 km bar into four 250 m segments, a 5 km bar
//!   into five 1 km segments), drawn as the classic alternating checker with
//!   a `0` at the left tick and the total at the right one;
//! * the **representative fraction** — `1:25 000` — which is the one number
//!   a surveyor reads first and which the bar's own metres-per-point already
//!   contains;
//! * **imperial** units for the Letter-size world, chosen by
//!   [`ScaleUnits`] and rounded in feet or miles the same 1-2-5 way.
//!
//! The metric arm is deliberately untouched by the unit work: the unit
//! conversion wraps [`round_125_down`] rather than replacing it, so a metric
//! page emits the same distance and the same label text it always did.

use pdf_writer::Content;

use super::{
    MapBox, PrintOptions, SCALE_BAR_FONT_PT, SCALE_BAR_HEIGHT_PT, SCALE_BAR_INSET_PT, TextPlan,
    line_width_pt, show_line,
};
use oxigis_render::MapView;

/// WGS 84 equatorial circumference, in metres — the Web Mercator constant the
/// scale bar's metres-per-pixel calculation rests on.
pub(super) const EARTH_CIRCUMFERENCE_M: f64 = 40_075_016.685_578_49;

/// One international foot, in metres (exact, by the 1959 agreement).
const FOOT_M: f64 = 0.3048;

/// One international mile, in metres (exact: 5280 ft).
const MILE_M: f64 = 1_609.344;

/// One inch, in metres (exact).
const INCH_M: f64 = 0.0254;

/// PostScript points per inch — the paper-side half of the representative
/// fraction.
const POINTS_PER_INCH: f64 = 72.0;

/// The `0` at the bar's left tick, as a constant so the plan and the painter
/// cannot spell it differently.
pub(super) const ZERO_TICK: &str = "0";

/// Vertical gap between the bar and the text above or below it, in points.
const SCALE_BAR_TEXT_GAP_PT: f32 = 3.0;

/// Padding between the scale-bar plate's edge and its content, in points.
const SCALE_BAR_PLATE_PAD_PT: f32 = 4.0;

/// Which units a scale bar counts in.
///
/// A dialog-level choice rather than a locale sniff: the same project is
/// printed for a European ministry and for a US county, and the person at the
/// export dialog is the one who knows which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleUnits {
    /// Metres and kilometres — the default, and byte-for-byte the pre-v1.7
    /// behaviour.
    #[default]
    Metric,
    /// Feet and miles.
    Imperial,
}

impl ScaleUnits {
    /// Both systems, in dialog order.
    pub const ALL: [Self; 2] = [Self::Metric, Self::Imperial];

    /// Dialog label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Metric => "Metric (m / km)",
            Self::Imperial => "Imperial (ft / mi)",
        }
    }
}

/// A computed scale bar: a round distance, the width it spans on the page,
/// how it divides into round segments, and the scale it implies.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleBar {
    /// The distance the bar represents, in metres — whatever unit the label
    /// counts in, this stays SI so the callers' maths needs no unit switch.
    pub metres: f64,
    /// The bar's width on the page, in points.
    pub width_pt: f32,
    /// The label, e.g. `"500 m"`, `"20 km"` or `"2 mi"`.
    pub label: String,
    /// How many equal segments the bar divides into — each one a round
    /// number in the bar's own unit (`4` or `5`).
    pub segments: u32,
    /// The representative fraction's denominator: the page is `1 :
    /// denominator` of the ground.
    pub denominator: f64,
    /// The representative fraction as it prints, e.g. `"1:25 000"`.
    pub rf_label: String,
}

impl ScaleBar {
    /// The width of one segment, in points.
    #[must_use]
    pub fn segment_width_pt(&self) -> f32 {
        self.width_pt / self.segments.max(1) as f32
    }

    /// The ground distance one segment covers, in metres.
    #[must_use]
    pub fn segment_metres(&self) -> f64 {
        self.metres / f64::from(self.segments.max(1))
    }
}

/// Ground metres one page point covers at the view's centre latitude.
///
/// The Web Mercator scale factor is `cos φ`, and the print raster is a
/// resampling of the same projection, so the page's own metres-per-point is
/// the tile pyramid's metres-per-pixel scaled by the raster's pixels per
/// point. [`None`] when the geometry degenerates.
#[must_use]
pub(super) fn metres_per_pt(view: &MapView, map_box: &MapBox) -> Option<f64> {
    let lat = view.center().lat.to_radians();
    let metres_per_px = EARTH_CIRCUMFERENCE_M * lat.cos() / (256.0 * 2.0_f64.powf(view.zoom()));
    let px_per_pt = f64::from(view.size_px()[0]) / f64::from(map_box.width);
    let metres_per_pt = metres_per_px * px_per_pt;
    (metres_per_pt.is_finite() && metres_per_pt > 0.0).then_some(metres_per_pt)
}

/// Computes the map's METRIC scale bar — the historical entry point, and
/// exactly `scale_bar_with(view, map_box, ScaleUnits::Metric)`.
///
/// Returns [`None`] if the geometry degenerates (non-finite zoom maths), which
/// no reachable [`MapView`] produces.
#[must_use]
pub fn scale_bar(view: &MapView, map_box: &MapBox) -> Option<ScaleBar> {
    scale_bar_with(view, map_box, ScaleUnits::Metric)
}

/// Computes the map's scale bar in `units`: the largest 1-2-5 × 10ⁿ distance
/// that fits in roughly a fifth of the map box's width.
///
/// Ground distance follows the Web Mercator scale factor at the view centre's
/// latitude — `cos(lat)` — which is also why one bar can honestly describe the
/// whole page: at page scale the latitude variation across the map box is
/// visually negligible, and a per-row scale bar is not a thing paper maps do.
#[must_use]
pub fn scale_bar_with(view: &MapView, map_box: &MapBox, units: ScaleUnits) -> Option<ScaleBar> {
    let metres_per_pt = metres_per_pt(view, map_box)?;
    let target_m = f64::from(map_box.width) / 5.0 * metres_per_pt;
    let (metres, label, segments) = match units {
        ScaleUnits::Metric => metric_bar(target_m)?,
        ScaleUnits::Imperial => imperial_bar(target_m)?,
    };
    let width_pt = (metres / metres_per_pt) as f32;
    if !width_pt.is_finite() || width_pt <= 0.0 {
        return None;
    }
    let denominator = metres_per_pt * POINTS_PER_INCH / INCH_M;
    Some(ScaleBar {
        metres,
        width_pt,
        label,
        segments,
        denominator,
        rf_label: format_rf(denominator),
    })
}

/// The metric bar for a target ground distance: rounded in METRES exactly as
/// every version since v1 did, then labelled km / m / cm.
fn metric_bar(target_m: f64) -> Option<(f64, String, u32)> {
    let (factor, base) = round_125_down(target_m)?;
    let metres = factor * base;
    let label = if metres >= 1000.0 {
        format!("{} km", format_round(metres / 1000.0))
    } else if metres >= 1.0 {
        format!("{} m", format_round(metres))
    } else {
        format!("{} cm", format_round(metres * 100.0))
    };
    Some((metres, label, segments_for(factor)))
}

/// The imperial bar for a target ground distance: the unit is picked from the
/// distance (miles, feet, then inches), and the 1-2-5 rounding then happens
/// IN that unit, so `2 mi` and `500 ft` are what the bar reads — never a
/// converted metric number with a fractional label.
fn imperial_bar(target_m: f64) -> Option<(f64, String, u32)> {
    let (unit_m, suffix) = if target_m >= MILE_M {
        (MILE_M, "mi")
    } else if target_m >= FOOT_M {
        (FOOT_M, "ft")
    } else {
        (INCH_M, "in")
    };
    let (factor, base) = round_125_down(target_m / unit_m)?;
    let value = factor * base;
    Some((
        value * unit_m,
        format!("{} {suffix}", format_round(value)),
        segments_for(factor),
    ))
}

/// How many segments a bar with mantissa `factor` divides into so that every
/// segment is itself a round number: `1` and `2` into quarters (250 m of a
/// 1 km bar, 500 m of a 2 km bar), `5` into fifths (1 km of a 5 km bar).
fn segments_for(factor: f64) -> u32 {
    if (factor - 5.0).abs() < 1e-9 { 5 } else { 4 }
}

/// The largest `1`, `2` or `5` × 10ⁿ value that is at most `target`, as its
/// `(mantissa, decade)` parts — the mantissa is what the segmentation reads,
/// and returning it beats re-deriving it from the product with a `log10`.
pub(super) fn round_125_down(target: f64) -> Option<(f64, f64)> {
    if !target.is_finite() || target <= 0.0 {
        return None;
    }
    let exponent = target.log10().floor();
    let base = 10.0_f64.powf(exponent);
    for factor in [5.0, 2.0, 1.0] {
        if factor * base <= target {
            return Some((factor, base));
        }
    }
    // `target` in `[base, base)` is impossible, but floats near a power of ten
    // can land here; one decade down always fits.
    Some((5.0, base / 10.0))
}

/// A whole number without a trailing `.0`, or one decimal otherwise.
fn format_round(value: f64) -> String {
    if (value - value.round()).abs() < 1e-9 {
        format!("{}", value.round() as i64)
    } else {
        format!("{value:.1}")
    }
}

/// The representative fraction as text: `1:25 000`, grouped in threes with a
/// plain space.
///
/// The denominator is the EXACT page-to-ground ratio rounded to a whole
/// number rather than a pretty 1:25 000-style round scale, because the page
/// is framed by the map box and the camera, not by a scale the user picked —
/// rounding the number would state a scale the sheet does not have.
fn format_rf(denominator: f64) -> String {
    if !denominator.is_finite() || denominator <= 0.0 {
        return String::new();
    }
    // Beyond i64 the grouping loop has nothing honest to print; a map at
    // 1:10^18 is not a map, and the saturating cast keeps the function total.
    let rounded = denominator.round().clamp(0.0, i64::MAX as f64) as i64;
    let digits = rounded.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3 + 2);
    for (position, digit) in digits.chars().enumerate() {
        if position > 0 && (digits.len() - position).is_multiple_of(3) {
            grouped.push(' ');
        }
        grouped.push(digit);
    }
    format!("1:{grouped}")
}

/// Every string the scale-bar plate shows, for the font plan — the same
/// strings [`paint`] draws, produced by the same call, so a character can
/// never be shown without a CID behind it.
#[must_use]
pub(super) fn plate_texts(bar: &ScaleBar, options: &PrintOptions) -> Vec<String> {
    let mut texts = vec![ZERO_TICK.to_string(), bar.label.clone()];
    if options.representative_fraction {
        texts.push(bar.rf_label.clone());
    }
    texts
}

/// Draws the scale bar inside the map box's bottom-left corner: a white
/// plate, the alternating checker bar, a `0` and the total over the end
/// ticks, and — when asked for — the representative fraction under it.
pub(super) fn paint(
    content: &mut Content,
    options: &PrintOptions,
    compose: &MapView,
    map_box: &MapBox,
    plan: Option<&TextPlan>,
) {
    if !options.scale_bar {
        return;
    }
    let Some(bar) = scale_bar_with(compose, map_box, options.scale_units) else {
        return;
    };
    let plate = plate_box(&bar, options, map_box, plan);
    let x = map_box.x + SCALE_BAR_INSET_PT;
    let y = map_box.y + SCALE_BAR_INSET_PT;
    let rf_row = rf_row_pt(options);
    let bar_y = y + rf_row;
    content.save_state();
    // The plate: white under everything, so the bar stays readable over a
    // dark basemap.
    content.set_fill_rgb(1.0, 1.0, 1.0);
    content.rect(plate.x, plate.y, plate.width, plate.height);
    content.fill_nonzero();
    // The checker: every other segment filled, the whole bar outlined, so
    // even a fully white segment still reads as part of the bar.
    content.set_fill_rgb(0.0, 0.0, 0.0);
    let segment_w = bar.segment_width_pt();
    for segment in 0..bar.segments {
        if segment % 2 == 0 {
            content.rect(
                x + segment as f32 * segment_w,
                bar_y,
                segment_w,
                SCALE_BAR_HEIGHT_PT,
            );
            content.fill_nonzero();
        }
    }
    content.set_stroke_rgb(0.0, 0.0, 0.0);
    content.set_line_width(0.5);
    content.rect(x, bar_y, bar.width_pt, SCALE_BAR_HEIGHT_PT);
    content.stroke();
    // The end ticks' values, each centred on its tick.
    let text_y = bar_y + SCALE_BAR_HEIGHT_PT + SCALE_BAR_TEXT_GAP_PT;
    let zero_w = line_width_pt(plan, ZERO_TICK, SCALE_BAR_FONT_PT);
    let label_w = line_width_pt(plan, &bar.label, SCALE_BAR_FONT_PT);
    show_line(
        content,
        plan,
        x - zero_w / 2.0,
        text_y,
        SCALE_BAR_FONT_PT,
        ZERO_TICK,
    );
    show_line(
        content,
        plan,
        x + bar.width_pt - label_w / 2.0,
        text_y,
        SCALE_BAR_FONT_PT,
        &bar.label,
    );
    if options.representative_fraction {
        content.set_fill_rgb(0.25, 0.25, 0.25);
        show_line(content, plan, x, y, SCALE_BAR_FONT_PT, &bar.rf_label);
    }
    content.restore_state();
}

/// The height the representative-fraction row occupies under the bar, in
/// points — zero when it is switched off.
fn rf_row_pt(options: &PrintOptions) -> f32 {
    if options.representative_fraction {
        SCALE_BAR_FONT_PT + SCALE_BAR_TEXT_GAP_PT
    } else {
        0.0
    }
}

/// The white plate behind the scale bar, in page points.
///
/// Public to the module because the legend has to know where the bar's plate
/// ends before it may claim the other corner — see `legend::rows`.
pub(super) fn plate_box(
    bar: &ScaleBar,
    options: &PrintOptions,
    map_box: &MapBox,
    plan: Option<&TextPlan>,
) -> MapBox {
    let x = map_box.x + SCALE_BAR_INSET_PT;
    let y = map_box.y + SCALE_BAR_INSET_PT;
    let zero_w = line_width_pt(plan, ZERO_TICK, SCALE_BAR_FONT_PT);
    let label_w = line_width_pt(plan, &bar.label, SCALE_BAR_FONT_PT);
    let rf_w = if options.representative_fraction {
        line_width_pt(plan, &bar.rf_label, SCALE_BAR_FONT_PT)
    } else {
        0.0
    };
    // The end labels are centred on the ticks, so each overhangs its end by
    // half its width; the plate has to cover the overhang or the text would
    // sit on the map.
    let left = x - zero_w / 2.0;
    let right = (x + bar.width_pt + label_w / 2.0).max(x + rf_w);
    let height = rf_row_pt(options)
        + SCALE_BAR_HEIGHT_PT
        + SCALE_BAR_TEXT_GAP_PT
        + SCALE_BAR_FONT_PT
        + SCALE_BAR_PLATE_PAD_PT;
    MapBox {
        x: left - SCALE_BAR_PLATE_PAD_PT,
        y: y - SCALE_BAR_PLATE_PAD_PT,
        width: (right - left) + 2.0 * SCALE_BAR_PLATE_PAD_PT,
        height,
    }
}

/// The right edge the scale-bar plate reaches on this page, measured WITHOUT
/// a font plan.
///
/// The legend decides whether it fits beside the bar, and that decision has
/// to be the same in the plan pass (which has no plan yet, by construction)
/// and in the paint pass. The degraded 0.6-em-per-character estimate is
/// wider than the real advance of the digits, spaces and unit letters a bar
/// label is made of — asserted against the bundled face in
/// `furniture_tests` — so a legend that clears this edge clears the real
/// plate too. A hypothetical face whose digits ran wider than 0.6 em would
/// cost a few points of overlap between two plates, never a malformed page.
#[must_use]
pub(super) fn plate_right_estimate_pt(
    options: &PrintOptions,
    compose: &MapView,
    map_box: &MapBox,
) -> f32 {
    if !options.scale_bar {
        return map_box.x;
    }
    match scale_bar_with(compose, map_box, options.scale_units) {
        Some(bar) => {
            let plate = plate_box(&bar, options, map_box, None);
            plate.x + plate.width
        }
        None => map_box.x,
    }
}
