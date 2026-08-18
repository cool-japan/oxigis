// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for the cartographic print furniture (print v1.7): the segmented
//! scale bar with its representative fraction, the north arrow, the legend,
//! and the document's `/Info` metadata.
//!
//! Split from `print/tests.rs` — which is already 1 200 lines — under the
//! same 2000-line rule the module has followed since v1.6. The fixtures are
//! shared rather than copied: `tests::request`, `tests::polygon_layer` and
//! friends are the same pages the older suite reasons about, so a furniture
//! assertion and a geometry assertion can never be about different maps.

use super::tests::{line_layer, opts, point_layer, polygon_layer, request};
use super::*;
use oxigeo::geojson::types::{Feature, Geometry, Polygon};
use oxigis_core::{
    Color as CoreColor, FillStyle, GeometryFamily, LayerStyleSet, LineStyle, StyleSlot,
};

/// The page every furniture test measures on.
fn fixture_box() -> MapBox {
    map_box(&opts())
}

/// A view at `lat`/`zoom` framed like the app's own 800×600 panel.
fn view_at(lat: f64, zoom: f64) -> MapView {
    MapView::new(LonLat::new(10.0, lat), zoom, [800.0, 600.0]).expect("a valid viewport")
}

/// The degraded (no font plan) content stream of `request`.
fn content_of(request: &PrintRequest, compose: &MapView) -> String {
    let ops = page_content(request, compose, &map_box(&request.options));
    String::from_utf8_lossy(&ops).into_owned()
}

/// A one-polygon layer whose GeoJSON collection names itself, the way
/// `ogr2ogr` and QGIS write it.
fn named_polygon_layer(name: &str) -> PrintLayer {
    let ring = vec![
        vec![-5.0, -5.0],
        vec![5.0, -5.0],
        vec![5.0, 5.0],
        vec![-5.0, -5.0],
    ];
    let polygon = Polygon::from_exterior(ring).expect("a valid ring");
    let feature = Feature::new(Some(Geometry::Polygon(polygon)), None);
    let mut collection = FeatureCollection::new(vec![feature]);
    let mut members = serde_json::Map::new();
    members.insert(
        "name".to_string(),
        serde_json::Value::String(name.to_string()),
    );
    collection.foreign_members = Some(members);
    let features = Arc::new(collection);
    PrintLayer {
        name: String::new(),
        families: crate::local_vector::collection_families(&features),
        features,
        style: LayerStyle::Fill(FillStyle::new(
            CoreColor::from_hex("3388ff").expect("valid hex"),
        ))
        .into(),
        opacity: 1.0,
    }
}

// ─── the scale bar ────────────────────────────────────────────────────────

#[test]
fn the_scale_bar_divides_into_round_segments() {
    for (lat, zoom) in [(0.0, 3.0), (35.6, 9.0), (60.0, 12.0), (-45.0, 16.0)] {
        let bar = scale_bar(&view_at(lat, zoom), &fixture_box()).expect("a computable bar");
        assert!(
            bar.segments == 4 || bar.segments == 5,
            "a bar divides into quarters or fifths: {} at lat {lat}",
            bar.segments
        );
        let segment = bar.segment_metres();
        assert!(
            (segment * f64::from(bar.segments) - bar.metres).abs() < 1e-6,
            "the segments must add up to the bar: {segment} × {} vs {}",
            bar.segments,
            bar.metres
        );
        // Every SEGMENT is itself a 1-2-5 × 10ⁿ distance — the point of
        // segmenting at all, since a reader measures against a segment.
        let mantissa = segment / 10.0_f64.powf(segment.log10().floor());
        assert!(
            [1.0, 2.0, 2.5, 5.0]
                .iter()
                .any(|factor| (mantissa - factor).abs() < 1e-6),
            "a round segment: {segment} m (mantissa {mantissa})"
        );
        assert!(
            (bar.segment_width_pt() * bar.segments as f32 - bar.width_pt).abs() < 0.01,
            "the drawn segments span the drawn bar"
        );
    }
}

#[test]
fn the_scale_bar_follows_the_mercator_latitude_factor() {
    let map_box = fixture_box();
    let equator = scalebar::metres_per_pt(&view_at(0.0, 10.0), &map_box).expect("a scale");
    let sixty = scalebar::metres_per_pt(&view_at(60.0, 10.0), &map_box).expect("a scale");
    let ratio = sixty / equator;
    assert!(
        (ratio - 60.0_f64.to_radians().cos()).abs() < 1e-9,
        "ground metres per point must scale by cos(latitude): {ratio}"
    );
    // And the bar the user reads shrinks with it: the same page width at 60°
    // covers half the ground it does at the equator.
    let equator_bar = scale_bar(&view_at(0.0, 10.0), &map_box).expect("a bar");
    let sixty_bar = scale_bar(&view_at(60.0, 10.0), &map_box).expect("a bar");
    assert!(
        sixty_bar.metres < equator_bar.metres,
        "{} m at 60° vs {} m at the equator",
        sixty_bar.metres,
        equator_bar.metres
    );
}

#[test]
fn the_representative_fraction_is_the_page_to_ground_ratio() {
    let map_box = fixture_box();
    for (lat, zoom) in [(0.0, 5.0), (48.0, 14.0)] {
        let view = view_at(lat, zoom);
        let bar = scale_bar(&view, &map_box).expect("a bar");
        let metres_per_pt = scalebar::metres_per_pt(&view, &map_box).expect("a scale");
        // One point is 1/72 inch = 0.0254/72 m of paper.
        let expected = metres_per_pt * 72.0 / 0.0254;
        assert!(
            (bar.denominator - expected).abs() / expected < 1e-12,
            "1:{} vs 1:{expected}",
            bar.denominator
        );
        // The label groups the rounded denominator in threes.
        let digits: String = bar.rf_label.trim_start_matches("1:").replace(' ', "");
        assert_eq!(
            digits,
            format!("{}", bar.denominator.round() as i64),
            "the grouped label must carry the same number: {}",
            bar.rf_label
        );
        assert!(bar.rf_label.starts_with("1:"));
    }
    // Grouping is every three digits from the right, and only there.
    let far = scale_bar(&view_at(0.0, 2.0), &map_box).expect("a bar");
    for group in far.rf_label.trim_start_matches("1:").split(' ').skip(1) {
        assert_eq!(group.len(), 3, "a trailing group is three digits: {group}");
    }
}

#[test]
fn the_metric_bar_keeps_its_pre_v1_7_distance_and_label() {
    // The unit work wraps the 1-2-5 rounding rather than replacing it, so the
    // metric arm must reproduce exactly what v1 computed: round the target in
    // METRES, then label km / m / cm.
    let map_box = fixture_box();
    for (lat, zoom) in [(0.0, 1.0), (0.0, 8.0), (35.6, 11.3), (70.0, 17.0)] {
        let view = view_at(lat, zoom);
        let bar = scale_bar(&view, &map_box).expect("a bar");
        let metres_per_pt = scalebar::metres_per_pt(&view, &map_box).expect("a scale");
        let target = f64::from(map_box.width) / 5.0 * metres_per_pt;
        let base = 10.0_f64.powf(target.log10().floor());
        let legacy = [5.0, 2.0, 1.0]
            .into_iter()
            .find(|factor| factor * base <= target)
            .map_or(base / 2.0, |factor| factor * base);
        assert!(
            (bar.metres - legacy).abs() / legacy < 1e-12,
            "at lat {lat} zoom {zoom}: {} vs the v1 {legacy}",
            bar.metres
        );
        let legacy_label = if legacy >= 1000.0 {
            format!("{} km", (legacy / 1000.0).round() as i64)
        } else if legacy >= 1.0 {
            format!("{} m", legacy.round() as i64)
        } else {
            format!("{} cm", (legacy * 100.0).round() as i64)
        };
        assert_eq!(bar.label, legacy_label);
    }
}

#[test]
fn the_imperial_bar_counts_in_feet_and_miles() {
    let map_box = fixture_box();
    let mut seen_feet = false;
    let mut seen_miles = false;
    for zoom in [3.0, 7.0, 11.0, 14.0, 17.0, 19.0] {
        let view = view_at(0.0, zoom);
        let bar = scale_bar_with(&view, &map_box, ScaleUnits::Imperial).expect("a bar");
        let (value, unit_m) = match bar.label.rsplit_once(' ') {
            Some((value, "mi")) => {
                seen_miles = true;
                (value, 1_609.344)
            }
            Some((value, "ft")) => {
                seen_feet = true;
                (value, 0.3048)
            }
            Some((value, "in")) => (value, 0.0254),
            other => panic!("an imperial label ends in a unit: {other:?}"),
        };
        let value: f64 = value.parse().expect("a numeric label");
        assert!(
            (bar.metres - value * unit_m).abs() / bar.metres < 1e-9,
            "the label states the bar's own ground distance: {}",
            bar.label
        );
        let mantissa = value / 10.0_f64.powf(value.log10().floor());
        assert!(
            [1.0, 2.0, 5.0]
                .iter()
                .any(|factor| (mantissa - factor).abs() < 1e-9),
            "a 1-2-5 imperial value: {value}"
        );
        // The bar still fits the fifth of the box it is budgeted.
        assert!(bar.width_pt > 0.0 && bar.width_pt <= map_box.width / 5.0 + 0.01);
    }
    assert!(seen_feet && seen_miles, "both units are reachable by zoom");
    // The metric arm is untouched by the choice.
    let metric = scale_bar_with(&view_at(0.0, 11.0), &map_box, ScaleUnits::Metric).expect("a bar");
    assert_eq!(
        metric,
        scale_bar(&view_at(0.0, 11.0), &map_box).expect("a bar")
    );
}

#[test]
fn the_scale_bar_plate_stays_inside_the_map_box() {
    let map_box = fixture_box();
    let mut options = opts();
    for representative_fraction in [true, false] {
        options.representative_fraction = representative_fraction;
        let bar = scale_bar(&view_at(35.0, 12.0), &map_box).expect("a bar");
        let plate = scalebar::plate_box(&bar, &options, &map_box, None);
        assert!(plate.x >= map_box.x, "the plate keeps off the left margin");
        assert!(plate.y >= map_box.y, "and off the bottom one");
        assert!(plate.x + plate.width <= map_box.x + map_box.width);
        assert!(plate.y + plate.height <= map_box.y + map_box.height);
        assert!(
            plate.height > SCALE_BAR_HEIGHT_PT,
            "the plate covers bar and text"
        );
    }
    // Turning the fraction on grows the plate downward-to-upward, never the
    // other way round.
    let bar = scale_bar(&view_at(35.0, 12.0), &map_box).expect("a bar");
    options.representative_fraction = true;
    let with_rf = scalebar::plate_box(&bar, &options, &map_box, None);
    options.representative_fraction = false;
    let without_rf = scalebar::plate_box(&bar, &options, &map_box, None);
    assert!(with_rf.height > without_rf.height);
}

#[test]
fn the_page_draws_a_segmented_bar_with_its_fraction() {
    let (request, compose) = request(Vec::new());
    let text = content_of(&request, &compose);
    assert!(text.contains("(0) Tj"), "the left tick reads zero");
    let bar = scale_bar(&compose, &map_box(&opts())).expect("a bar");
    assert!(
        text.contains(&format!("({}) Tj", bar.label)),
        "the right tick reads the distance: {}",
        bar.label
    );
    assert!(
        text.contains(&format!("({}) Tj", bar.rf_label)),
        "the representative fraction prints: {}",
        bar.rf_label
    );
    // The checker is filled rectangles plus one stroked outline.
    assert!(text.contains(" re\nf"), "segments are filled rectangles");
    assert!(text.contains(" re\nS"), "the bar carries an outline");
}

#[test]
fn the_page_prints_the_unit_system_the_dialog_chose() {
    let (mut request, compose) = request(Vec::new());
    let map_box = map_box(&opts());
    let metric = scale_bar(&compose, &map_box).expect("a metric bar");
    let imperial =
        scale_bar_with(&compose, &map_box, ScaleUnits::Imperial).expect("an imperial bar");
    assert_ne!(metric.label, imperial.label, "the fixture separates them");

    request.options.scale_units = ScaleUnits::Imperial;
    let text = content_of(&request, &compose);
    assert!(
        text.contains(&format!("({}) Tj", imperial.label)),
        "the imperial distance prints: {}",
        imperial.label
    );
    assert!(
        !text.contains(&format!("({}) Tj", metric.label)),
        "and the metric one does not"
    );
    // The fraction is a ratio of lengths, so it reads the same in either
    // system — the page must not restate it differently.
    assert_eq!(metric.rf_label, imperial.rf_label);
    assert!(text.contains(&format!("({}) Tj", imperial.rf_label)));
}

#[test]
fn every_furniture_piece_can_be_switched_off() {
    let (mut request, compose) = request(vec![polygon_layer()]);
    let bar = scale_bar(&compose, &map_box(&opts())).expect("a bar");
    let full = content_of(&request, &compose);
    assert!(full.contains(&format!("({}) Tj", bar.label)));
    assert!(full.contains("(N) Tj"));
    assert!(full.contains("Layer 1"));

    request.options.scale_bar = false;
    let no_bar = content_of(&request, &compose);
    assert!(!no_bar.contains(&format!("({}) Tj", bar.label)));
    assert!(!no_bar.contains("(0) Tj"), "and no tick either");

    request.options = opts();
    request.options.representative_fraction = false;
    let no_rf = content_of(&request, &compose);
    assert!(!no_rf.contains(&format!("({}) Tj", bar.rf_label)));
    assert!(
        no_rf.contains(&format!("({}) Tj", bar.label)),
        "the bar itself stays"
    );

    request.options = opts();
    request.options.north_arrow = false;
    assert!(!content_of(&request, &compose).contains("(N) Tj"));

    request.options = opts();
    request.options.legend = false;
    assert!(!content_of(&request, &compose).contains("Layer 1"));
}

#[test]
fn the_default_options_print_every_furniture_piece() {
    let options = PrintOptions::default();
    assert!(options.scale_bar);
    assert!(options.representative_fraction);
    assert!(options.north_arrow);
    assert!(options.legend);
    assert!(options.document_metadata);
    assert_eq!(options.scale_units, ScaleUnits::Metric);
    assert_eq!(options.creation_epoch_secs, None);
    // Still the dialog-state shape the app copies out of `&mut self`.
    let copied = options;
    assert_eq!(copied, options);
    assert_eq!(ScaleUnits::ALL.len(), 2);
    assert!(!ScaleUnits::Imperial.label().is_empty());
}

// ─── the north arrow ──────────────────────────────────────────────────────

#[test]
fn the_north_arrow_sits_in_the_map_box_top_right_corner() {
    let map_box = fixture_box();
    let plate = north::plate_box(&opts(), &map_box).expect("an arrow");
    assert!(plate.x + plate.width <= map_box.x + map_box.width);
    assert!(plate.y + plate.height <= map_box.y + map_box.height);
    assert!(
        plate.x > map_box.x + map_box.width / 2.0,
        "the arrow keeps to the right half"
    );
    assert!(
        plate.y > map_box.y + map_box.height / 2.0,
        "and to the top half"
    );
    let mut off = opts();
    off.north_arrow = false;
    assert!(north::plate_box(&off, &map_box).is_none());
}

#[test]
fn a_map_box_too_small_for_an_arrow_gets_none() {
    let tiny = MapBox {
        x: 10.0,
        y: 10.0,
        width: 60.0,
        height: 60.0,
    };
    assert!(
        north::plate_box(&opts(), &tiny).is_none(),
        "an arrow may annotate a map, not occupy it"
    );
}

#[test]
fn the_north_arrow_draws_a_needle_and_its_letter() {
    let (request, compose) = request(Vec::new());
    let text = content_of(&request, &compose);
    assert!(text.contains("(N) Tj"), "the letter prints");
    assert!(
        text.contains(" l\nh\nB"),
        "each half is a closed path, filled and stroked"
    );
}

// ─── the legend ───────────────────────────────────────────────────────────

#[test]
fn the_legend_names_a_layer_from_its_geojson_name() {
    let (request, compose) = request(vec![named_polygon_layer("Roads")]);
    let rows = legend::rows(&request, &compose, &fixture_box());
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].label, "Roads", "a named collection names its row");
    assert_eq!(rows[0].alpha.as_deref(), Some("GS0"));
    assert!(matches!(rows[0].swatch, legend::Swatch::Fill { .. }));
    assert!(content_of(&request, &compose).contains("Roads"));
}

#[test]
fn the_projects_own_layer_name_beats_the_geojson_member() {
    // What the layer panel shows is what the legend must say: a user who
    // renamed a layer has not seen the file's internal `name` member since
    // they imported it, so legending under that would name a stranger.
    let mut layer = named_polygon_layer("Roads");
    layer.name = "Prefecture roads".to_string();
    let (named, named_compose) = request(vec![layer]);
    let rows = legend::rows(&named, &named_compose, &fixture_box());
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].label, "Prefecture roads");
    assert!(content_of(&named, &named_compose).contains("Prefecture roads"));

    // Whitespace is not a name: it falls through to the member, exactly as an
    // empty string does.
    let mut blank_layer = named_polygon_layer("Roads");
    blank_layer.name = "   ".to_string();
    let (blank, blank_compose) = request(vec![blank_layer]);
    let blank_rows = legend::rows(&blank, &blank_compose, &fixture_box());
    assert_eq!(blank_rows[0].label, "Roads");
}

#[test]
fn an_unnamed_layer_is_qualified_by_its_geometry_family() {
    let (request, compose) = request(vec![polygon_layer(), line_layer(), point_layer()]);
    let rows = legend::rows(&request, &compose, &fixture_box());
    assert_eq!(rows.len(), 3, "one row per layer with no overrides");
    assert_eq!(rows[0].label, "Layer 1 (Polygons)");
    assert_eq!(rows[1].label, "Layer 2 (Lines)");
    assert_eq!(rows[2].label, "Layer 3 (Points)");
    assert!(matches!(rows[1].swatch, legend::Swatch::Line { .. }));
    assert!(matches!(rows[2].swatch, legend::Swatch::Circle { .. }));
    // The rows the plan sees are the rows the page draws — one call, one
    // answer.
    let planned = legend::texts(&request, &compose, &fixture_box());
    let drawn: Vec<String> = rows.into_iter().map(|row| row.label).collect();
    assert_eq!(planned, drawn);
}

#[test]
fn a_layer_with_family_overrides_gets_one_row_per_slot() {
    // A mixed collection: a polygon and a line, the line overridden green.
    let ring = vec![
        vec![-1.0, -1.0],
        vec![1.0, -1.0],
        vec![1.0, 1.0],
        vec![-1.0, -1.0],
    ];
    let polygon = Polygon::from_exterior(ring).expect("a valid ring");
    let line = oxigeo::geojson::types::LineString::new(vec![vec![-2.0, 0.0], vec![2.0, 1.0]])
        .expect("a line");
    let features = Arc::new(FeatureCollection::new(vec![
        Feature::new(Some(Geometry::Polygon(polygon)), None),
        Feature::new(Some(Geometry::LineString(line)), None),
    ]));
    let mut style = LayerStyleSet::new(LayerStyle::Fill(FillStyle::new(
        CoreColor::from_hex("3388ff").expect("valid hex"),
    )));
    style.set_override(
        GeometryFamily::Line,
        LayerStyle::Line(LineStyle::new(
            CoreColor::from_hex("00aa44").expect("valid hex"),
            2.0,
        )),
    );
    let layer = PrintLayer {
        name: String::new(),
        families: crate::local_vector::collection_families(&features),
        features,
        style,
        opacity: 1.0,
    };
    let (request, compose) = request(vec![layer]);
    let rows = legend::rows(&request, &compose, &fixture_box());
    assert_eq!(rows.len(), 2, "the base slot and the line override");
    assert_eq!(rows[0].label, "Layer 1 (Polygons)");
    assert_eq!(rows[1].label, "Layer 1 (Lines)");
    assert_eq!(rows[0].alpha.as_deref(), Some("GS0"));
    assert_eq!(
        rows[1].alpha.as_deref(),
        Some(slot_alpha_name(0, StyleSlot::Family(GeometryFamily::Line)).as_str()),
        "an override row paints under its own alpha state"
    );
    assert!(matches!(rows[1].swatch, legend::Swatch::Line { .. }));
}

#[test]
fn a_symbol_layer_gets_no_legend_row() {
    let mut symbol = point_layer();
    symbol.style = crate::local_vector::local_symbol_style("name").into();
    let (request, compose) = request(vec![symbol]);
    assert!(
        legend::rows(&request, &compose, &fixture_box()).is_empty(),
        "a label layer has no swatch to show"
    );
}

#[test]
fn the_legend_caps_its_rows_and_counts_what_it_dropped() {
    let layers: Vec<PrintLayer> = (0..30).map(|_| polygon_layer()).collect();
    let total = layers.len();
    let (request, compose) = request(layers);
    let rows = legend::rows(&request, &compose, &fixture_box());
    assert!(
        rows.len() <= 12,
        "the legend never grows past its ceiling: {}",
        rows.len()
    );
    let last = rows.last().expect("a capped legend still has rows");
    assert_eq!(last.swatch, legend::Swatch::Overflow);
    assert_eq!(
        last.label,
        format!("+{} more", total - (rows.len() - 1)),
        "the last row accounts for every layer that did not fit"
    );
    assert!(last.alpha.is_none(), "the count row paints no swatch");
    assert!(content_of(&request, &compose).contains("more"));
}

#[test]
fn the_legend_yields_the_corner_to_the_scale_bar() {
    let (request, compose) = request(vec![polygon_layer()]);
    // A map box barely wider than the two plates: the legend must stand down
    // rather than print over the bar.
    let narrow = MapBox {
        x: 36.0,
        y: 36.0,
        width: 190.0,
        height: 300.0,
    };
    assert!(
        legend::rows(&request, &compose, &narrow).is_empty(),
        "no room beside the bar means no legend"
    );
    let roomy = MapBox {
        width: 500.0,
        ..narrow
    };
    assert_eq!(legend::rows(&request, &compose, &roomy).len(), 1);
    // A box with no vertical room left under the arrow drops it too.
    let flat = MapBox {
        height: 40.0,
        ..roomy
    };
    assert!(legend::rows(&request, &compose, &flat).is_empty());
}

#[test]
fn the_legend_plate_stays_inside_the_map_box_and_clear_of_the_arrow() {
    let map_box = fixture_box();
    let layers: Vec<PrintLayer> = (0..30).map(|_| polygon_layer()).collect();
    let (request, compose) = request(layers);
    let rows = legend::rows(&request, &compose, &map_box);
    // Measured through the painter's own geometry, not through copied
    // numbers: retuning a constant must move this test, not slip past it.
    let plate = legend::plate_box(rows.len(), &map_box);
    let top = plate.y + plate.height;
    let arrow = north::plate_box(&opts(), &map_box).expect("an arrow");
    assert!(
        top <= arrow.y,
        "the tallest legend still stops below the arrow: {top} vs {}",
        arrow.y
    );
    assert!(top <= map_box.y + map_box.height);
    assert!(plate.x >= map_box.x && plate.x + plate.width <= map_box.x + map_box.width);
    assert_eq!(
        plate.height,
        rows.len() as f32 * legend::LEGEND_ROW_PT + 2.0 * legend::LEGEND_PAD_PT
    );
    assert!(plate.y >= map_box.y + legend::LEGEND_INSET_PT - 0.01);
}

#[test]
fn the_scale_bar_plate_estimate_is_never_narrower_than_the_real_one() {
    // The legend decides whether it fits beside the bar in the PLAN pass,
    // where no font plan exists yet, using the degraded 0.6-em width
    // estimate. That decision is only sound if the estimate is an
    // OVER-estimate of the real face's plate: a real plate wider than the
    // estimate would let the legend claim a corner the bar already owns.
    let map_box = fixture_box();
    let fonts = PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
    let mut options = opts();
    for representative_fraction in [true, false] {
        options.representative_fraction = representative_fraction;
        for (lat, zoom) in [(0.0, 2.0), (35.6, 11.3), (60.0, 16.0)] {
            let view = view_at(lat, zoom);
            let bar = scale_bar(&view, &map_box).expect("a bar");
            let plan = font::plan(
                &fonts,
                &super::tests::regular(&[&bar.label, &bar.rf_label, scalebar::ZERO_TICK]),
                None,
            )
            .expect("a plan");
            let real = scalebar::plate_box(&bar, &options, &map_box, Some(&plan));
            let estimated = scalebar::plate_box(&bar, &options, &map_box, None);
            assert!(
                real.width <= estimated.width + 0.001,
                "the estimate must bound the real plate: {} vs {} at lat {lat}",
                real.width,
                estimated.width
            );
            assert!(
                scalebar::plate_right_estimate_pt(&options, &view, &map_box)
                    >= real.x + real.width - 0.001,
                "and the right edge the legend keeps off is the outer one"
            );
        }
    }
}

#[test]
fn a_hostile_layer_name_cannot_grow_the_font_plan() {
    let long = "名".repeat(4_000);
    let (request, compose) = request(vec![named_polygon_layer(&long)]);
    let rows = legend::rows(&request, &compose, &fixture_box());
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].label.chars().count(),
        120,
        "a name is bounded before it reaches the subsetter"
    );
}

// ─── document metadata ────────────────────────────────────────────────────

/// One assembled document for `request`, over an empty (all-gray) raster.
fn document(request: &PrintRequest, compose: &MapView) -> Vec<u8> {
    let map_box = map_box(&request.options);
    let out_px = raster_size_px(&map_box, &request.options);
    let rgb = compose_map_rgb(compose, &mut |_| None);
    pdf_document(request, compose, &rgb, out_px, &PrintFonts::none(), &[]).expect("a document")
}

#[test]
fn the_document_info_carries_title_producer_and_creation_date() {
    let (mut request, compose) = request(vec![polygon_layer()]);
    request.options.creation_epoch_secs = Some(0);
    let pdf = document(&request, &compose);
    let text = String::from_utf8_lossy(&pdf);
    assert!(
        text.contains("/Title (Print fixture)"),
        "the project names it"
    );
    assert!(
        text.contains(&format!("/Producer (OxiGIS {})", env!("CARGO_PKG_VERSION"))),
        "the producer records this build"
    );
    assert!(text.contains("/Creator (OxiGIS)"));
    assert!(
        text.contains("/CreationDate (D:19700101000000Z)"),
        "a pinned second stamps exactly, in UTC"
    );
    assert!(
        text.contains("/MarkInfo") && text.contains("/Marked true"),
        "the page's /Artifact and /Span marks are declared"
    );
    assert!(text.contains("/Info 7 0 R"), "the trailer points at it");
}

#[test]
fn the_creation_date_is_the_utc_calendar() {
    let (mut request, compose) = request(Vec::new());
    for (epoch, expected) in [
        (0_i64, "D:19700101000000Z"),
        (1_700_000_000, "D:20231114221320Z"),
        (-1, "D:19691231235959Z"),
        (951_782_400, "D:20000229000000Z"),
        (i64::MIN, "D:00000101000000Z"),
        (i64::MAX, "D:99991231235959Z"),
    ] {
        request.options.creation_epoch_secs = Some(epoch);
        let pdf = document(&request, &compose);
        let text = String::from_utf8_lossy(&pdf);
        assert!(
            text.contains(&format!("/CreationDate ({expected})")),
            "epoch {epoch} must stamp {expected}"
        );
    }
}

#[test]
fn a_pinned_creation_second_makes_the_export_reproducible() {
    let (mut request, compose) = request(vec![polygon_layer(), line_layer()]);
    request.options.creation_epoch_secs = Some(1_600_000_000);
    let first = document(&request, &compose);
    let second = document(&request, &compose);
    assert_eq!(
        first, second,
        "the same options must produce the same bytes"
    );
}

#[test]
fn a_non_ascii_title_is_written_as_utf16() {
    let (mut request, compose) = request(Vec::new());
    request.title = "東京".to_string();
    request.options.creation_epoch_secs = Some(0);
    let pdf = document(&request, &compose);
    let text = String::from_utf8_lossy(&pdf);
    assert!(
        text.contains("/Title <FEFF67714EAC>"),
        "a CJK title takes the UTF-16BE form, BOM and all"
    );
}

#[test]
fn switching_the_metadata_off_writes_no_info_dictionary() {
    let (mut request, compose) = request(Vec::new());
    request.options.document_metadata = false;
    let pdf = document(&request, &compose);
    let text = String::from_utf8_lossy(&pdf);
    assert!(!text.contains("/Producer"));
    assert!(!text.contains("/CreationDate"));
    assert!(!text.contains("/Info "), "and the trailer names none");
    // The catalog's /MarkInfo is not metadata about the file; it describes
    // the content stream, which is unchanged by this switch.
    assert!(text.contains("/Marked true"));
    assert!(pdf.starts_with(b"%PDF-") && text.contains("%%EOF"));
}

#[test]
fn an_empty_title_writes_no_title_entry() {
    let (mut request, compose) = request(Vec::new());
    request.title = String::new();
    let pdf = document(&request, &compose);
    let text = String::from_utf8_lossy(&pdf);
    assert!(
        !text.contains("/Title"),
        "an unnamed project claims no title"
    );
    assert!(text.contains("/Producer"));
}

// ─── the plan and the page agree ──────────────────────────────────────────

#[test]
fn every_furniture_string_reaches_the_font_plan() {
    // A planned page draws CIDs, and a CID only exists for a character the
    // plan saw: any furniture string missing from the plan would print as
    // NOTHING (the synthetic run path drops unmapped characters). Assemble a
    // real document with a real face and read the /ToUnicode map back.
    let (mut request, compose) = request(vec![named_polygon_layer("Roads")]);
    request.options.creation_epoch_secs = Some(0);
    let map_box = map_box(&request.options);
    let out_px = raster_size_px(&map_box, &request.options);
    let rgb = compose_map_rgb(&compose, &mut |_| None);
    let fonts = PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
    let pdf = pdf_document(&request, &compose, &rgb, out_px, &fonts, &[]).expect("a document");
    assert!(pdf.starts_with(b"%PDF-"));

    let bar = scale_bar(&compose, &map_box).expect("a bar");
    let mut wanted: Vec<String> = vec![
        scalebar::ZERO_TICK.to_string(),
        bar.label.clone(),
        bar.rf_label.clone(),
        north::NORTH_LABEL.to_string(),
    ];
    wanted.extend(legend::texts(&request, &compose, &map_box));
    let plan = font::plan(
        &fonts,
        &{
            let mut texts: Vec<(oxigis_core::LabelWeight, &str)> = vec![
                (oxigis_core::LabelWeight::Regular, request.title.as_str()),
                (
                    oxigis_core::LabelWeight::Regular,
                    request.attribution.as_str(),
                ),
            ];
            texts.extend(
                wanted
                    .iter()
                    .map(|text| (oxigis_core::LabelWeight::Regular, text.as_str())),
            );
            texts
        },
        None,
    )
    .expect("a plan");
    for text in &wanted {
        for ch in text.chars().filter(|ch| !ch.is_whitespace()) {
            assert!(
                plan.glyph(oxigis_core::LabelWeight::Regular, ch).is_some(),
                "'{ch}' of {text:?} must have a CID or it prints as nothing"
            );
        }
    }
    // And the page really does show them through the embedded resource.
    let ops = page_content_planned(&request, &compose, &map_box, Some(&plan), &[]);
    let drawn = String::from_utf8_lossy(&ops);
    assert!(drawn.contains("/F1"), "furniture text is CID text now");
    assert!(!drawn.contains("(Roads)"), "not literal WinAnsi bytes");
}

/// The page's CONTENT stream, inflated back out of an assembled document —
/// the operators exactly as a viewer decompresses them.
///
/// Streams are sliced by their dictionary's `/Length N` rather than by
/// scanning for `endstream`, because compressed bytes can contain either
/// keyword; the content stream is the one carrying the image placement.
fn content_stream_of(pdf: &[u8]) -> String {
    let mut cursor = 0;
    while let Some(at) = super::tests::find_bytes(&pdf[cursor..], b"/Length ").map(|at| cursor + at)
    {
        let digits_at = at + b"/Length ".len();
        let mut end_digits = digits_at;
        while pdf.get(end_digits).is_some_and(u8::is_ascii_digit) {
            end_digits += 1;
        }
        let length: usize = String::from_utf8_lossy(&pdf[digits_at..end_digits])
            .parse()
            .unwrap_or(0);
        cursor = end_digits;
        let Some(open) =
            super::tests::find_bytes(&pdf[end_digits..], b"stream").map(|found| end_digits + found)
        else {
            break;
        };
        let mut start = open + b"stream".len();
        if pdf.get(start) == Some(&b'\r') {
            start += 1;
        }
        if pdf.get(start) == Some(&b'\n') {
            start += 1;
        }
        let Some(data) = pdf.get(start..start + length) else {
            continue;
        };
        if let Ok(body) = oxiarc_deflate::zlib::zlib_decompress(data) {
            let body = String::from_utf8_lossy(&body).into_owned();
            if body.contains("/Im0 Do") {
                return body;
            }
        }
        cursor = start + length;
    }
    String::new()
}

#[test]
fn the_assembled_document_carries_every_furniture_operator() {
    let (mut request, compose) = request(vec![named_polygon_layer("Roads")]);
    request.options.creation_epoch_secs = Some(0);
    let bar = scale_bar(&compose, &map_box(&request.options)).expect("a bar");
    let body = content_stream_of(&document(&request, &compose));
    assert!(!body.is_empty(), "the content stream must inflate");
    assert!(body.contains("(N) Tj"), "the north arrow's letter");
    assert!(body.contains("(0) Tj"), "the bar's zero tick");
    assert!(
        body.contains(&format!("({}) Tj", bar.label)),
        "its distance"
    );
    assert!(
        body.contains(&format!("({}) Tj", bar.rf_label)),
        "its representative fraction"
    );
    assert!(body.contains("(Roads) Tj"), "the legend's row");
    assert!(
        body.contains("(Print fixture) Tj"),
        "and the title the page always had"
    );
}

#[test]
fn the_furniture_survives_a_page_with_no_font_plan() {
    // The degraded path: Base-14 Helvetica, WinAnsi bytes, no embedded face.
    let (request, compose) = request(vec![polygon_layer(), point_layer()]);
    let text = content_of(&request, &compose);
    assert!(text.contains("(N) Tj"));
    assert!(text.contains("(0) Tj"));
    assert!(text.contains("Layer 1"));
    assert!(text.contains("/F0"), "the Base-14 resource carries it");
}

#[test]
fn a_degenerate_view_prints_no_furniture_rather_than_panicking() {
    let (request, compose) = request(Vec::new());
    let flat = MapBox {
        x: 0.0,
        y: 0.0,
        width: 0.0,
        height: 0.0,
    };
    assert!(scale_bar(&compose, &flat).is_none());
    assert!(scale_bar_with(&compose, &flat, ScaleUnits::Imperial).is_none());
    assert!(north::plate_box(&opts(), &flat).is_none());
    assert!(legend::rows(&request, &compose, &flat).is_empty());
    let ops = page_content(&request, &compose, &flat);
    assert!(!ops.is_empty(), "the page still assembles");
}
