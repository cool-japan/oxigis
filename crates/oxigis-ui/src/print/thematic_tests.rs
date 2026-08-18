// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Classified-layer printing (thematic v1.6): the page has to show the same
//! many-coloured map the screen does, with one legend row per class and one
//! registered alpha state per bucket.
//!
//! Its own file rather than more of `print/tests.rs`, which is already near the
//! 2000-line rule.
//!
//! # What is actually asserted
//!
//! Not "the bytes changed" — a colour swap would satisfy that — but the three
//! statements the divergence was made of:
//!
//! 1. every class paints, in its OWN colour, under its OWN alpha state;
//! 2. every state the painter names is registered in the page's resource
//!    dictionary (an unregistered `/GS0c1 gs` is an invalid PDF that a
//!    `contains("GS0c1")` assertion would happily pass);
//! 3. a layer that is not classified prints exactly the bytes it always did.

use super::tests::{opts, request};
use super::*;
use oxigeo::geojson::types::{Feature, Geometry, Point, Polygon};
use oxigis_core::{
    AttrValue, CategoryClass, CircleStyle, FillStyle, GeometryFamily, LayerStyleSet, Renderer,
};

/// A colour written the way a PDF operator writes it: three unit floats.
fn rgb_op(color: Color) -> String {
    let rgb = to_rgb(color);
    format!("{} {} {}", rgb[0], rgb[1], rgb[2])
}

/// Three squares carrying `zone` = `a`, `b` and `c`.
fn zoned_polygons() -> Arc<FeatureCollection> {
    let mut features = Vec::new();
    for (index, zone) in ["a", "b", "c"].into_iter().enumerate() {
        let x = index as f64 * 5.0 - 10.0;
        let ring = vec![
            vec![x, -4.0],
            vec![x + 4.0, -4.0],
            vec![x + 4.0, 4.0],
            vec![x, 4.0],
            vec![x, -4.0],
        ];
        let polygon = Polygon::from_exterior(ring).expect("a valid ring");
        let mut properties = serde_json::Map::new();
        properties.insert("zone".to_string(), serde_json::Value::String(zone.into()));
        features.push(Feature::new(
            Some(Geometry::Polygon(polygon)),
            Some(properties),
        ));
    }
    Arc::new(FeatureCollection::new(features))
}

/// The class colours the fixtures classify into, and the base they sit over.
const BASE_HEX: &str = "3388ff";
const CLASS_A_HEX: &str = "ff0000";
const CLASS_B_HEX: &str = "00ff00";

fn hex(text: &str) -> Color {
    Color::from_hex(text).expect("a valid fixture colour")
}

/// A blue-based fill set, categorized on `zone` into red `a` and green `b`.
/// `c` falls through to the fallback, which is the base itself.
fn categorized_set() -> LayerStyleSet {
    let mut set = LayerStyleSet::new(LayerStyle::Fill(FillStyle::new(hex(BASE_HEX))));
    set.set_renderer(Renderer::categorized(
        "zone",
        [
            CategoryClass::new(
                AttrValue::text("a"),
                LayerStyle::Fill(FillStyle::new(hex(CLASS_A_HEX))),
            ),
            CategoryClass::new(
                AttrValue::text("b"),
                LayerStyle::Fill(FillStyle::new(hex(CLASS_B_HEX))),
            ),
        ],
        None,
    ));
    set
}

/// The zoned polygons drawn with `style`.
fn zoned_layer(style: LayerStyleSet) -> PrintLayer {
    let features = zoned_polygons();
    PrintLayer {
        name: "Zones".to_string(),
        families: crate::local_vector::collection_families(&features),
        features,
        style,
        opacity: 1.0,
    }
}

/// The page's raw operators for `layer`, as text.
fn ops_for(layer: PrintLayer) -> String {
    let (request, compose) = request(vec![layer]);
    let bytes = page_content(&request, &compose, &map_box(&opts()));
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Every ExtGState name the operators actually apply, in order of first use.
fn applied_states(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in text.lines() {
        let mut tokens = line.split_whitespace();
        let (Some(name), Some("gs")) = (tokens.next(), tokens.next()) else {
            continue;
        };
        let Some(name) = name.strip_prefix('/') else {
            continue;
        };
        if !names.iter().any(|known: &String| known == name) {
            names.push(name.to_string());
        }
    }
    names
}

#[test]
fn a_classified_layer_paints_one_bucket_per_class_in_its_own_colour() {
    // THE divergence this closes: before, all three squares printed in the
    // base's blue while the screen showed red, green and blue.
    let text = ops_for(zoned_layer(categorized_set()));
    for (label, colour) in [
        ("the fallback", BASE_HEX),
        ("class a", CLASS_A_HEX),
        ("class b", CLASS_B_HEX),
    ] {
        assert!(
            text.contains(&rgb_op(hex(colour))),
            "{label} must reach the page in its own colour",
        );
    }
    // One bucket per class plus the fallback, each under its own state and in
    // the order the map's partition paints them: fallback first, then class 0,
    // then class 1.
    assert_eq!(applied_states(&text), ["GS0", "GS0c0", "GS0c1"]);
    // Every square is still drawn exactly once — the buckets partition the
    // features, they do not duplicate them.
    assert_eq!(text.matches("f*").count(), 3, "three even-odd fills");
}

#[test]
fn every_alpha_state_the_painter_names_is_registered_by_the_document() {
    // An unregistered `/GS0c1 gs` is an INVALID PDF that a `contains` test
    // would pass, so the two halves are checked against each other: the names
    // the content stream applies, and the names the resource dictionary and the
    // ExtGState objects carry.
    let mut set = categorized_set();
    // Classes that differ in opacity are exactly the case one state per class
    // exists for.
    for (index, alpha) in [(0_usize, 0.25_f32), (1, 0.75)] {
        match set.renderer_mut().class_style_mut(index) {
            Some(LayerStyle::Fill(fill)) => fill.set_opacity(alpha),
            _ => panic!("the fixture's classes are fills"),
        }
    }
    let layer = zoned_layer(set);
    let (request, compose) = request(vec![layer]);
    let map_box = map_box(&opts());
    let applied = applied_states(&String::from_utf8_lossy(&page_content(
        &request, &compose, &map_box,
    )));
    let registered: Vec<String> = request
        .layers
        .iter()
        .enumerate()
        .flat_map(|(index, layer)| {
            layer_alpha_slots(layer)
                .into_iter()
                .map(move |(slot, class, _)| class_alpha_name(index, slot, class))
        })
        .collect();
    for name in &applied {
        assert!(
            registered.contains(name),
            "the painter applied /{name} gs, which the page never registered: {registered:?}",
        );
    }
    // And the states really do carry the classes' own alphas, so a
    // half-transparent class is half-transparent on paper too.
    let out_px = raster_size_px(&map_box, &opts());
    let rgb = compose_map_rgb(&compose, &mut |_| None);
    let pdf = pdf_document(&request, &compose, &rgb, out_px, &PrintFonts::none(), &[])
        .expect("a document");
    let pdf_text = String::from_utf8_lossy(&pdf);
    for name in &applied {
        assert!(
            pdf_text.contains(&format!("/{name} ")),
            "/{name} is missing from the page's /ExtGState dictionary",
        );
    }
    assert!(pdf_text.contains("/ca 0.25"), "class 0's own alpha");
    assert!(pdf_text.contains("/ca 0.75"), "class 1's own alpha");
}

#[test]
fn an_unclassified_layer_prints_exactly_the_bytes_it_always_did() {
    // The compatibility floor. Three shapes of "not classified" must all print
    // identically: no renderer at all, a renderer with an empty class list, and
    // one whose classes were cleared.
    let plain = ops_for(zoned_layer(LayerStyleSet::new(LayerStyle::Fill(
        FillStyle::new(hex(BASE_HEX)),
    ))));
    let mut empty = LayerStyleSet::new(LayerStyle::Fill(FillStyle::new(hex(BASE_HEX))));
    empty.set_renderer(Renderer::categorized("zone", [], None));
    assert_eq!(
        ops_for(zoned_layer(empty)),
        plain,
        "an empty classification"
    );

    let mut cleared = categorized_set();
    cleared.set_renderer(Renderer::Single);
    assert_eq!(ops_for(zoned_layer(cleared)), plain, "a cleared one");
    assert!(
        !plain.contains("c0 gs"),
        "an unclassified layer names no class state: {plain}",
    );
    assert_eq!(applied_states(&plain), ["GS0"]);
}

#[test]
fn a_class_never_erases_the_family_its_geometry_belongs_to() {
    // The composition rule, on the page: a `Fill` class over a POINT family
    // must print recoloured circles, not a fill that would draw nothing. This
    // is the one case where naive substitution silently loses the layer.
    let point = Point::new_2d(3.0, 4.0).expect("a point");
    let mut properties = serde_json::Map::new();
    properties.insert("zone".to_string(), serde_json::Value::String("a".into()));
    let features = Arc::new(FeatureCollection::new(vec![Feature::new(
        Some(Geometry::Point(point)),
        Some(properties),
    )]));
    let mut set = categorized_set();
    set.set_override(
        GeometryFamily::Point,
        LayerStyle::Circle(CircleStyle::new(6.0, hex(BASE_HEX))),
    );
    let layer = PrintLayer {
        name: "Points".to_string(),
        families: crate::local_vector::collection_families(&features),
        features,
        style: set,
        opacity: 1.0,
    };
    let text = ops_for(layer);
    assert!(
        text.contains(&rgb_op(hex(CLASS_A_HEX))),
        "the class colour reaches the marker",
    );
    assert!(
        text.contains(" c\n"),
        "the marker is still drawn as a circle's Béziers: {text}",
    );
    // The point family carries an override, so its class states are named
    // against THAT slot rather than the base's.
    assert!(
        applied_states(&text)
            .iter()
            .any(|name| name.starts_with("GS0f") && name.contains('c')),
        "an overridden family's class keeps its own slot name",
    );
}

#[test]
fn the_legend_shows_one_row_per_class_resolved_through_the_shared_helper() {
    let (request, compose) = request(vec![zoned_layer(categorized_set())]);
    let map_box = map_box(&opts());
    let rows = legend::rows(&request, &compose, &map_box);
    let labels: Vec<&str> = rows.iter().map(|row| row.label.as_str()).collect();
    assert_eq!(labels, ["Zones: a", "Zones: b", "Zones: Everything else"]);
    // Each row paints under the bucket it names, so a half-transparent class
    // gets a half-transparent swatch.
    let alphas: Vec<Option<&str>> = rows.iter().map(|row| row.alpha.as_deref()).collect();
    assert_eq!(alphas, [Some("GS0c0"), Some("GS0c1"), Some("GS0")]);
    // And the swatches are the SAME styles the shared helper resolves, so the
    // printed legend cannot disagree with an on-screen one.
    let shared =
        crate::renderer_panel::legend_rows(&request.layers[0].style, GeometryFamily::Polygon);
    assert_eq!(shared.len(), rows.len());
    for ((_, style), row) in shared.iter().zip(&rows) {
        assert_eq!(row.swatch, super::legend::swatch_for(style));
    }
}

#[test]
fn the_overflow_row_counts_classes_and_not_only_layers() {
    // The pre-count decides `+N more`, and before this it counted one row per
    // layer — so a 20-class layer would have claimed the sheet was complete
    // while showing two of its classes.
    let mut set = LayerStyleSet::new(LayerStyle::Fill(FillStyle::new(hex(BASE_HEX))));
    let classes: Vec<CategoryClass> = (0..20)
        .map(|index| {
            CategoryClass::new(
                AttrValue::text(format!("zone-{index}")),
                LayerStyle::Fill(FillStyle::new(hex(CLASS_A_HEX))),
            )
        })
        .collect();
    set.set_renderer(Renderer::categorized("zone", classes, None));
    let (request, compose) = request(vec![zoned_layer(set)]);
    let map_box = map_box(&opts());
    let rows = legend::rows(&request, &compose, &map_box);
    let total = 21; // twenty classes plus "everything else"
    assert!(!rows.is_empty(), "the legend still places");
    let shown = rows.len() - 1;
    assert_eq!(
        rows.last().map(|row| row.label.clone()),
        Some(format!("+{} more", total - shown)),
        "the count has to be of ROWS the reader cannot see",
    );
    assert!(shown < total, "the fixture really does overflow");
}
