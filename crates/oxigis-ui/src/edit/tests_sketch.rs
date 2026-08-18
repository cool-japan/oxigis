// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`super::sketch`]: the finish rules for each drawing tool, the
//! double-click dedupe, `Backspace`, and the ring-closing predicate.
//!
//! Screen positions are produced by projecting a known lon/lat with
//! [`super::hit::to_screen`] and offsetting by a pixel count, so a tolerance
//! assertion says what it means with no degrees-per-pixel arithmetic in the test.

use super::hit::to_screen;
use super::overlay;
use super::sketch::{
    DOUBLE_CLICK_DEDUPE_PT, MIN_LINE_VERTICES, MIN_POLYGON_VERTICES, Sketch,
    add_feature_transaction, draw_label, geometry_from, point_geometry, too_few_message,
};
use super::{EditMode, EditSelection};
use crate::edit::command::FeatureOp;
use egui::{Pos2, pos2, vec2};
use oxigeo::geojson::types::Geometry;
use oxigis_core::LayerId;
use oxigis_render::{LonLat, MapView};

/// The map panel's top-left corner: the identity, so a screen position reads
/// directly as a projected pixel.
const ORIGIN: Pos2 = pos2(0.0, 0.0);
/// One physical pixel per egui point, so "points" and "pixels" coincide.
const PPP: f32 = 1.0;

/// A camera on the equator at zoom 6, over an 800×600 panel.
fn view() -> MapView {
    MapView::new(LonLat::new(0.0, 0.0), 6.0, [800.0, 600.0])
        .expect("a 800x600 panel is a valid viewport")
}

/// Where `at` lands on screen.
fn screen(at: LonLat) -> Pos2 {
    to_screen(view(), ORIGIN, PPP, at)
}

/// The lon/lat that lands `offset` points from the map's centre.
fn offset_lon_lat(offset: egui::Vec2) -> LonLat {
    let target = screen(LonLat::new(0.0, 0.0)) + offset;
    view().screen_to_lon_lat([target.x * PPP, target.y * PPP])
}

/// A sketch of `mode` holding `points`.
fn sketch_of(mode: EditMode, points: &[LonLat]) -> Sketch {
    let mut sketch = Sketch::default();
    for point in points {
        sketch.append(mode, *point);
    }
    sketch
}

/// Three well-separated positions, for a ring that is a real triangle.
fn triangle() -> [LonLat; 3] {
    [
        offset_lon_lat(vec2(0.0, 0.0)),
        offset_lon_lat(vec2(120.0, 0.0)),
        offset_lon_lat(vec2(60.0, -90.0)),
    ]
}

/// The exterior ring of a finished polygon.
fn ring_of(geometry: &Geometry) -> Vec<[f64; 2]> {
    match geometry {
        Geometry::Polygon(polygon) => polygon
            .coordinates
            .first()
            .expect("a polygon has an exterior")
            .iter()
            .map(|position| {
                [
                    position.first().copied().unwrap_or_default(),
                    position.get(1).copied().unwrap_or_default(),
                ]
            })
            .collect(),
        other => panic!("expected a polygon, got {other:?}"),
    }
}

#[test]
fn polygon_finish_appends_the_closing_position_and_yields_at_least_four() {
    let corners = triangle();
    let mut sketch = sketch_of(EditMode::DrawPolygon, &corners);
    assert_eq!(sketch.len(), MIN_POLYGON_VERTICES);

    let geometry = sketch
        .finish(EditMode::DrawPolygon)
        .expect("three vertices are a ring");
    let ring = ring_of(&geometry);
    assert_eq!(
        ring.len(),
        4,
        "a three-vertex sketch stores four positions: the ring is closed on disk"
    );
    assert_eq!(
        ring.first(),
        ring.last(),
        "the closing position must be a clone of position zero"
    );
    // The open sequence — the one `VertexRef` addresses — is unchanged.
    for (index, corner) in corners.iter().enumerate() {
        let stored = ring.get(index).copied().expect("the ring holds it");
        assert!((stored[0] - corner.lon).abs() < 1e-12, "{stored:?}");
        assert!((stored[1] - corner.lat).abs() < 1e-12, "{stored:?}");
    }

    // Finishing consumes the sketch: the tool stays latched but the vertices do
    // not carry into the next feature.
    assert!(!sketch.is_active());
    assert_eq!(sketch.len(), 0);
    assert_eq!(sketch.cursor, None);
}

#[test]
fn line_finish_below_two_points_yields_none() {
    let mut empty = Sketch::default();
    assert!(empty.finish(EditMode::DrawLine).is_none());

    let mut lone = sketch_of(EditMode::DrawLine, &[offset_lon_lat(vec2(0.0, 0.0))]);
    assert!(
        lone.finish(EditMode::DrawLine).is_none(),
        "one vertex is not a line"
    );
    assert_eq!(
        lone.len(),
        1,
        "a refused finish must leave the sketch exactly as it was"
    );

    let mut pair = sketch_of(
        EditMode::DrawLine,
        &[
            offset_lon_lat(vec2(0.0, 0.0)),
            offset_lon_lat(vec2(50.0, 0.0)),
        ],
    );
    assert!(matches!(
        pair.finish(EditMode::DrawLine),
        Some(Geometry::LineString(_))
    ));
    assert_eq!(MIN_LINE_VERTICES, 2);

    // The same rule for a polygon, one vertex short.
    let corners = triangle();
    let mut two = sketch_of(EditMode::DrawPolygon, &corners[..2]);
    assert!(two.finish(EditMode::DrawPolygon).is_none());
    assert_eq!(two.len(), 2);
}

#[test]
fn double_click_dedupe_drops_the_coincident_last_vertex() {
    // A double click's *first* release already appended a vertex, so the last
    // two sit on top of each other. Without the dedupe every double-click
    // finished polygon carries a repeated vertex into the stored data.
    let corners = triangle();
    let nearly_the_same = offset_lon_lat(vec2(60.0 + DOUBLE_CLICK_DEDUPE_PT * 0.5, -90.0));
    let mut sketch = sketch_of(EditMode::DrawPolygon, &corners);
    sketch.append(EditMode::DrawPolygon, nearly_the_same);
    assert_eq!(sketch.len(), 4);

    let geometry = sketch
        .finish_from_double_click(EditMode::DrawPolygon, view(), PPP)
        .expect("three vertices remain after the dedupe");
    assert_eq!(
        ring_of(&geometry).len(),
        4,
        "the duplicate must be dropped before the ring is closed"
    );

    // A genuinely distinct final vertex is kept.
    let far = offset_lon_lat(vec2(200.0, -200.0));
    let mut kept = sketch_of(EditMode::DrawPolygon, &corners);
    kept.append(EditMode::DrawPolygon, far);
    let geometry = kept
        .finish_from_double_click(EditMode::DrawPolygon, view(), PPP)
        .expect("four vertices are a ring");
    assert_eq!(
        ring_of(&geometry).len(),
        5,
        "a vertex further than the dedupe radius is a real vertex"
    );
}

#[test]
fn pop_then_finish_produces_the_expected_geometry() {
    let corners = triangle();
    let stray = offset_lon_lat(vec2(400.0, 200.0));
    let mut sketch = sketch_of(EditMode::DrawPolygon, &corners);
    sketch.append(EditMode::DrawPolygon, stray);
    assert_eq!(sketch.len(), 4);

    let dropped = sketch.pop().expect("there was a vertex to take back");
    assert!((dropped.lon - stray.lon).abs() < 1e-12);
    assert_eq!(sketch.len(), 3);

    let geometry = sketch
        .finish(EditMode::DrawPolygon)
        .expect("the three original corners remain");
    let ring = ring_of(&geometry);
    assert_eq!(ring.len(), 4);
    for corner in &corners {
        assert!(
            ring.iter()
                .any(|stored| (stored[0] - corner.lon).abs() < 1e-12),
            "the popped sketch kept every remaining corner"
        );
    }

    // Popping the last vertex forgets the tool that owned the sketch, so the
    // Escape ladder climbs down to the next rung rather than cancelling nothing.
    let mut lone = sketch_of(EditMode::DrawLine, &[corners[0]]);
    assert!(lone.is_active());
    assert!(lone.pop().is_some());
    assert!(!lone.is_active());
    assert!(lone.pop().is_none());
}

#[test]
fn closes_at_triggers_within_tolerance_of_vertex_zero() {
    let corners = triangle();
    let sketch = sketch_of(EditMode::DrawPolygon, &corners);
    let first = screen(corners[0]);
    let tolerance = 12.0_f32;

    assert!(
        sketch.closes_at(view(), ORIGIN, first, PPP, tolerance),
        "the first vertex itself always closes"
    );
    assert!(
        sketch.closes_at(
            view(),
            ORIGIN,
            first + vec2(tolerance - 1.0, 0.0),
            PPP,
            tolerance
        ),
        "just inside the radius closes"
    );
    assert!(
        !sketch.closes_at(
            view(),
            ORIGIN,
            first + vec2(tolerance + 2.0, 0.0),
            PPP,
            tolerance
        ),
        "just outside it does not"
    );
    assert!(
        !sketch.closes_at(view(), ORIGIN, screen(corners[2]), PPP, tolerance),
        "the *last* vertex is not the one a ring closes on to"
    );

    // Nothing to close on to, and a nonsense tolerance, both answer no rather
    // than closing a ring the user never asked for.
    assert!(!Sketch::default().closes_at(view(), ORIGIN, first, PPP, tolerance));
    assert!(!sketch.closes_at(view(), ORIGIN, first, PPP, f32::NAN));
    assert!(!sketch.closes_at(view(), ORIGIN, first, PPP, 0.0));
}

#[test]
fn a_sketch_never_carries_vertices_across_a_tool_change() {
    let corners = triangle();
    let mut sketch = sketch_of(EditMode::DrawLine, &corners[..2]);
    sketch.append(EditMode::DrawPolygon, corners[2]);
    assert_eq!(
        sketch.len(),
        1,
        "vertices belong to the tool that was collecting them"
    );
    assert_eq!(sketch.mode, Some(EditMode::DrawPolygon));
}

#[test]
fn a_finished_sketch_becomes_one_add_at_the_end_of_the_collection() {
    let layer = LayerId::new();
    let geometry = point_geometry(offset_lon_lat(vec2(10.0, 10.0))).expect("a finite position");
    let transaction = add_feature_transaction(
        layer,
        7,
        geometry,
        draw_label(EditMode::DrawPoint),
        Some(EditSelection::feature(2)),
    );
    assert_eq!(transaction.label, "Draw point");
    assert_eq!(
        transaction.selection_before,
        Some(EditSelection::feature(2))
    );
    assert_eq!(
        transaction.selection_after,
        Some(EditSelection::feature(7)),
        "the new feature is selected, so Delete and the form address what was drawn"
    );
    assert_eq!(
        transaction.coalesce, None,
        "one drawn feature is one undo step"
    );
    match transaction.ops.as_slice() {
        [FeatureOp::Add { index, feature }] => {
            assert_eq!(*index, 7);
            assert!(
                feature.properties.is_some(),
                "an empty map, not None: the form needs somewhere to add a key"
            );
            assert!(matches!(feature.geometry, Some(Geometry::Point(_))));
        }
        other => panic!("expected one Add, got {other:?}"),
    }

    assert_eq!(draw_label(EditMode::DrawLine), "Draw line");
    assert_eq!(draw_label(EditMode::DrawPolygon), "Draw polygon");
}

#[test]
fn geometry_from_refuses_the_non_drawing_tools_and_a_non_finite_position() {
    let corners = triangle();
    for mode in [EditMode::Off, EditMode::Select] {
        assert!(geometry_from(mode, &corners).is_none());
    }
    assert!(point_geometry(LonLat::new(f64::NAN, 0.0)).is_none());
    assert!(
        too_few_message(EditMode::DrawPolygon).contains(&MIN_POLYGON_VERTICES.to_string()),
        "the refusal has to name how many vertices are still needed"
    );
}

#[test]
fn the_draw_hint_reports_the_sketch_size_and_every_way_out_of_it() {
    let corners = triangle();
    let empty = Sketch::default();
    let started = overlay::hint_text(EditMode::DrawLine, None, None, &empty).expect("a hint");
    assert!(started.starts_with("Line"), "{started}");

    let one = sketch_of(EditMode::DrawLine, &corners[..1]);
    let text = overlay::hint_text(EditMode::DrawLine, None, None, &one).expect("a hint");
    assert!(text.contains("1 vertex"), "{text}");
    assert!(!text.contains("1 vertices"), "{text}");
    assert!(text.contains("Backspace"), "{text}");
    assert!(text.contains("Esc"), "{text}");

    let ring = sketch_of(EditMode::DrawPolygon, &corners);
    let text = overlay::hint_text(EditMode::DrawPolygon, None, None, &ring).expect("a hint");
    assert!(text.contains("3 vertices"), "{text}");
    assert!(
        text.contains("first vertex to close"),
        "a closable ring must say so: {text}"
    );

    let two = sketch_of(EditMode::DrawPolygon, &corners[..2]);
    let text = overlay::hint_text(EditMode::DrawPolygon, None, None, &two).expect("a hint");
    assert!(
        !text.contains("first vertex to close"),
        "two vertices cannot close: {text}"
    );

    // A point tool has no sketch to report on, and editing off says nothing.
    let point = overlay::hint_text(EditMode::DrawPoint, None, None, &ring).expect("a hint");
    assert!(point.starts_with("Point"), "{point}");
    assert_eq!(overlay::hint_text(EditMode::Off, None, None, &ring), None);
}
