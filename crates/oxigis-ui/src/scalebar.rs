// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The live map's ground-distance bar: the Web Mercator arithmetic behind it,
//! and the geometry constants the panel paints it with.
//!
//! Split out of [`crate::map_view`] under the 2000-line rule. The camera and
//! both paint paths stay there; what is here is the pure derivation
//! ([`screen_scale_bar`]) plus the plate's own dimensions, so the rounding and
//! the latitude correction are testable without allocating a panel rect.
//!
//! # Why this is projection maths and not geodesy
//!
//! [`crate::measure`] measures the **ground** on the WGS 84 ellipsoid, because
//! a tape has to answer "how far is it really". A scale bar answers a
//! different question — "how much ground does this picture's centimetre
//! cover" — and must therefore be measured in the projection the picture is
//! drawn in: Web Mercator, whose scale factor at the view centre's latitude is
//! `cos φ`. Using the ellipsoidal number here would produce a bar that
//! disagreed with the map it sits on, which is the one thing a scale bar may
//! not do.
//!
//! One bar honestly describes the whole panel because the latitude variation
//! across a single viewport is visually negligible at any zoom where a bar is
//! drawn at all — the same argument the printed page's bar makes.
//!
//! # Relationship to the printed bar
//!
//! `print::scalebar` carries the same 1-2-5 × 10ⁿ rounding and the same
//! circumference constant, and [`round_125_down`] here is deliberately a
//! reimplementation rather than an import. The two have different inputs (a
//! page box in PostScript points versus a panel rect in logical pixels) and
//! different neighbours: the print version is bound to `pdf_writer` types and
//! the composer's own `MapBox`, so importing it would tie the live map's
//! furniture to the PDF writer. The rounding itself is six lines; the coupling
//! would not be.

use oxigis_render::MapView;

/// WGS 84 equatorial circumference, in metres — the Web Mercator constant the
/// on-screen scale bar's metres-per-pixel calculation rests on.
///
/// The same number `print::scalebar`'s `EARTH_CIRCUMFERENCE_M` carries, and
/// deliberately restated rather than imported: this module must not depend on
/// the PDF composer to draw the live map's furniture (see the module docs).
const EARTH_CIRCUMFERENCE_M: f64 = 40_075_016.685_578_49;

/// Longest the scale bar may be, as a fraction of the panel's width.
///
/// A fifth is what the printed page uses and what every slippy map settles on:
/// long enough to read a length off, short enough that it never dominates the
/// corner it sits in.
pub(crate) const SCALE_BAR_MAX_WIDTH_FRACTION: f32 = 0.2;

/// Height of the scale bar's own rule, in logical pixels.
pub(crate) const SCALE_BAR_HEIGHT: f32 = 5.0;

/// Gap between the scale-bar plate and the panel's bottom-left corner, in
/// logical pixels — the attribution plate's margin, so the two sit on the same
/// line.
pub(crate) const SCALE_BAR_MARGIN: f32 = 6.0;

/// Padding inside the scale-bar plate, in logical pixels.
pub(crate) const SCALE_BAR_PAD: f32 = 5.0;

/// Font size of the scale bar's label, in points.
pub(crate) const SCALE_BAR_FONT: f32 = 11.0;

/// Shortest bar worth drawing, in logical pixels: below this the label is
/// wider than the rule it describes, and the plate reads as a caption with a
/// dash rather than as a scale.
pub(crate) const SCALE_BAR_MIN_WIDTH: f32 = 24.0;

/// How far up [`MapPanelState::paint_fallback`] lifts the scale bar, in
/// logical pixels, to clear the note it writes in the same corner.
pub(crate) const FALLBACK_NOTE_ROW: f32 = 18.0;

/// A computed on-screen scale bar: a round ground distance, how wide it is on
/// the panel, and what to call it.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenScaleBar {
    /// The distance the bar spans, in metres.
    pub metres: f64,
    /// The bar's width, in **logical** pixels (what `egui` paints in).
    pub width: f32,
    /// The label, e.g. `"500 m"` or `"20 km"`.
    pub label: String,
}

/// Ground metres one **logical** pixel covers at `view`'s centre latitude.
///
/// Web Mercator's scale factor is `cos φ`, and one tile is 256 physical pixels
/// at integer zoom, so the physical metres-per-pixel is
/// `C·cos φ / (256·2^z)`; `ppp` converts that to the logical pixels `egui`
/// lays out in. [`None`] when the geometry degenerates (a non-finite or
/// non-positive result), which no reachable [`MapView`] produces.
#[must_use]
pub fn metres_per_logical_px(view: &MapView, ppp: f32) -> Option<f64> {
    if !ppp.is_finite() || ppp <= 0.0 {
        return None;
    }
    let lat = view.center().lat.to_radians();
    let metres_per_physical_px =
        EARTH_CIRCUMFERENCE_M * lat.cos() / (256.0 * 2.0_f64.powf(view.zoom()));
    let metres = metres_per_physical_px * f64::from(ppp);
    (metres.is_finite() && metres > 0.0).then_some(metres)
}

/// The largest `1`, `2` or `5` × 10ⁿ value that is at most `target`, as its
/// `(mantissa, decade)` parts.
///
/// The same rounding the printed page's bar uses; see the module docs for why
/// it is restated here rather than imported. [`None`] for a non-finite or
/// non-positive target.
#[must_use]
pub fn round_125_down(target: f64) -> Option<(f64, f64)> {
    if !target.is_finite() || target <= 0.0 {
        return None;
    }
    let base = 10.0_f64.powf(target.log10().floor());
    if !base.is_finite() || base <= 0.0 {
        return None;
    }
    for factor in [5.0, 2.0, 1.0] {
        if factor * base <= target {
            return Some((factor, base));
        }
    }
    // `target` below its own decade is impossible, but floats just under a
    // power of ten can land here; one decade down always fits.
    Some((5.0, base / 10.0))
}

/// The scale bar for `view` drawn into a panel `panel_width` logical pixels
/// wide: the largest round distance that fits in
/// `SCALE_BAR_MAX_WIDTH_FRACTION` of it.
///
/// [`None`] when the geometry degenerates or the resulting bar would be
/// shorter than `SCALE_BAR_MIN_WIDTH` — which happens at the top of the zoom
/// range, where a round distance can be a handful of centimetres.
#[must_use]
pub fn screen_scale_bar(view: &MapView, panel_width: f32, ppp: f32) -> Option<ScreenScaleBar> {
    if !panel_width.is_finite() || panel_width <= 0.0 {
        return None;
    }
    let metres_per_px = metres_per_logical_px(view, ppp)?;
    let target_m = f64::from(panel_width * SCALE_BAR_MAX_WIDTH_FRACTION) * metres_per_px;
    let (factor, base) = round_125_down(target_m)?;
    let metres = factor * base;
    let width = (metres / metres_per_px) as f32;
    if !width.is_finite() || width < SCALE_BAR_MIN_WIDTH {
        return None;
    }
    Some(ScreenScaleBar {
        metres,
        width,
        label: scale_label(metres),
    })
}

/// How a round ground distance is written: km at and above a kilometre, m
/// down to a metre, cm below that.
fn scale_label(metres: f64) -> String {
    if metres >= 1_000.0 {
        format!("{} km", trim_scale(metres / 1_000.0))
    } else if metres >= 1.0 {
        format!("{} m", trim_scale(metres))
    } else {
        format!("{} cm", trim_scale(metres * 100.0))
    }
}

/// A 1-2-5 value without a trailing `.0` — every value this is ever handed is
/// a whole number in its own unit except the `0.5`-style ones a decade
/// boundary produces.
fn trim_scale(value: f64) -> String {
    if (value - value.round()).abs() < 1e-9 {
        format!("{}", value.round())
    } else {
        format!("{value:.1}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigis_render::LonLat;

    #[test]
    fn scalebar_rounding_picks_the_largest_one_two_or_five() {
        // The whole contract of a 1-2-5 bar: the chosen value is round, it is
        // at most the target, and nothing rounder fits between them.
        for (target, expected) in [
            (1.0, 1.0),
            (1.9, 1.0),
            (2.0, 2.0),
            (4.99, 2.0),
            (5.0, 5.0),
            (9.99, 5.0),
            (10.0, 10.0),
            (137.0, 100.0),
            (999.0, 500.0),
            (1_000.0, 1_000.0),
            (0.03, 0.02),
        ] {
            let Some((factor, base)) = round_125_down(target) else {
                panic!("no rounding for {target}");
            };
            let value = factor * base;
            assert!(
                (value - expected).abs() < expected * 1e-9,
                "round_125_down({target}) = {value}, expected {expected}"
            );
            assert!(value <= target * (1.0 + 1e-12), "{value} exceeds {target}");
            assert!(
                [1.0, 2.0, 5.0].iter().any(|m| (factor - m).abs() < 1e-9),
                "mantissa {factor} is not 1, 2 or 5"
            );
        }
    }

    #[test]
    fn scalebar_rounding_refuses_degenerate_targets() {
        assert_eq!(round_125_down(0.0), None);
        assert_eq!(round_125_down(-5.0), None);
        assert_eq!(round_125_down(f64::NAN), None);
        assert_eq!(round_125_down(f64::INFINITY), None);
    }

    #[test]
    fn scalebar_metres_per_pixel_follows_the_web_mercator_cosine() {
        // At zoom 0 the equator is one 256-px tile wide, so a pixel is
        // C/256 metres — the anchor every other value scales from.
        let Ok(equator) = MapView::new(LonLat::new(0.0, 0.0), 0.0, [512.0, 512.0]) else {
            panic!("view construction failed");
        };
        let Some(metres) = metres_per_logical_px(&equator, 1.0) else {
            panic!("no metres-per-pixel at the equator");
        };
        assert!(
            (metres - EARTH_CIRCUMFERENCE_M / 256.0).abs() < 1e-6,
            "{metres} m/px at z0"
        );
        // One zoom level in halves it.
        let Ok(zoomed) = MapView::new(LonLat::new(0.0, 0.0), 1.0, [512.0, 512.0]) else {
            panic!("view construction failed");
        };
        let Some(halved) = metres_per_logical_px(&zoomed, 1.0) else {
            panic!("no metres-per-pixel at z1");
        };
        assert!((halved - metres / 2.0).abs() < 1e-6, "{halved} m/px at z1");
        // And 60°N covers cos 60° = half the ground per pixel — the latitude
        // correction this bar exists for.
        let Ok(north) = MapView::new(LonLat::new(0.0, 60.0), 0.0, [512.0, 512.0]) else {
            panic!("view construction failed");
        };
        let Some(cold) = metres_per_logical_px(&north, 1.0) else {
            panic!("no metres-per-pixel at 60N");
        };
        assert!(
            (cold / metres - 0.5).abs() < 1e-3,
            "60°N is {} of the equator",
            cold / metres
        );
        // A hi-dpi surface lays out in fewer logical pixels, so each covers
        // more ground.
        let Some(hidpi) = metres_per_logical_px(&equator, 2.0) else {
            panic!("no metres-per-pixel at ppp 2");
        };
        assert!((hidpi - metres * 2.0).abs() < 1e-6);
        assert_eq!(metres_per_logical_px(&equator, 0.0), None);
        assert_eq!(metres_per_logical_px(&equator, f32::NAN), None);
    }

    #[test]
    fn scalebar_fits_inside_a_fifth_of_the_panel_and_labels_itself() {
        let Ok(view) = MapView::new(LonLat::new(139.7, 35.7), 12.0, [1024.0, 768.0]) else {
            panic!("view construction failed");
        };
        let Some(bar) = screen_scale_bar(&view, 800.0, 1.0) else {
            panic!("no scale bar at a typical city zoom");
        };
        assert!(
            bar.width <= 800.0 * SCALE_BAR_MAX_WIDTH_FRACTION + 1e-3,
            "bar is {} px of 800",
            bar.width
        );
        assert!(bar.width >= SCALE_BAR_MIN_WIDTH);
        // The label's number must be the metres the bar claims.
        assert!(
            bar.label.ends_with(" m") || bar.label.ends_with(" km"),
            "{}",
            bar.label
        );
        let Some(metres_per_px) = metres_per_logical_px(&view, 1.0) else {
            panic!("no metres-per-pixel");
        };
        assert!(
            (f64::from(bar.width) * metres_per_px - bar.metres).abs() < bar.metres * 1e-6,
            "{} px does not span {} m",
            bar.width,
            bar.metres
        );
    }

    #[test]
    fn scalebar_label_switches_units_at_the_decade_boundaries() {
        assert_eq!(scale_label(1_000.0), "1 km");
        assert_eq!(scale_label(20_000.0), "20 km");
        assert_eq!(scale_label(500.0), "500 m");
        assert_eq!(scale_label(1.0), "1 m");
        assert_eq!(scale_label(0.5), "50 cm");
        assert_eq!(scale_label(0.02), "2 cm");
    }

    #[test]
    fn scalebar_is_wider_in_ground_terms_the_further_north_the_camera_is() {
        // The same zoom and the same panel: a bar of the same pixel order
        // spans HALF the ground at 60°N. Without the cos φ correction both
        // would report the same distance, which is the bug this test names.
        let Ok(equator) = MapView::new(LonLat::new(0.0, 0.0), 8.0, [1024.0, 768.0]) else {
            panic!("view construction failed");
        };
        let Ok(north) = MapView::new(LonLat::new(0.0, 60.0), 8.0, [1024.0, 768.0]) else {
            panic!("view construction failed");
        };
        let Some(warm) = screen_scale_bar(&equator, 800.0, 1.0) else {
            panic!("no equatorial bar");
        };
        let Some(cold) = screen_scale_bar(&north, 800.0, 1.0) else {
            panic!("no northern bar");
        };
        assert!(
            cold.metres < warm.metres,
            "{} m at 60°N vs {} m at the equator",
            cold.metres,
            warm.metres
        );
    }

    #[test]
    fn scalebar_declines_to_draw_when_there_is_nothing_worth_drawing() {
        let Ok(view) = MapView::new(LonLat::new(0.0, 0.0), 4.0, [512.0, 512.0]) else {
            panic!("view construction failed");
        };
        assert_eq!(screen_scale_bar(&view, 0.0, 1.0), None);
        assert_eq!(screen_scale_bar(&view, f32::NAN, 1.0), None);
        // A panel too narrow for the minimum bar gets none.
        assert_eq!(screen_scale_bar(&view, 20.0, 1.0), None);
        // At the very top of the zoom range a round distance is millimetres,
        // which is shorter than the minimum bar however it is rounded.
        let Ok(deep) = MapView::new(
            LonLat::new(0.0, 0.0),
            f64::from(oxigis_render::MAX_ZOOM),
            [512.0, 512.0],
        ) else {
            panic!("view construction failed");
        };
        let bar = screen_scale_bar(&deep, 800.0, 1.0);
        assert!(
            bar.as_ref()
                .is_none_or(|bar| bar.width >= SCALE_BAR_MIN_WIDTH),
            "{bar:?}"
        );
    }
}
