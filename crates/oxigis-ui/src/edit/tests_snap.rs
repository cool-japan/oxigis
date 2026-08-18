// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`super::snap`]: the world-space grid, the ranking rules, the
//! exclusion a live drag needs, the degraded path, and a differential test
//! against a brute-force scan.
//!
//! Geometry is placed by projecting a known lon/lat with [`super::hit::to_screen`]
//! and offsetting by a pixel count, so a tolerance assertion says what it means
//! without any degrees-per-pixel arithmetic in the test itself.

use super::VertexRef;
use super::hit::to_screen;
use super::snap::{SNAP_TOLERANCE_PT, SnapIndex, SnapKind, SnapSettings, snap_to_sketch_start};
use egui::{Pos2, pos2, vec2};
use oxigeo::geojson::types::{
    Feature, FeatureCollection, Geometry, LineString, Point, Polygon, Position, Properties,
};
use oxigis_core::LayerId;
use oxigis_render::{LonLat, MapView};
use std::sync::Arc;

/// The map panel's top-left corner: the identity, so a screen position reads
/// directly as a projected pixel.
const ORIGIN: Pos2 = pos2(0.0, 0.0);
/// One physical pixel per egui point, so "points" and "pixels" coincide.
const PPP: f32 = 1.0;
/// The panel size every test uses, in egui points.
const PANEL: [f32; 2] = [800.0, 600.0];

/// A camera on the equator at `zoom`, for a panel of `PANEL` points at `ppp`.
fn view_at(zoom: f64, ppp: f32) -> MapView {
    MapView::new(
        LonLat::new(0.0, 0.0),
        zoom,
        [PANEL[0] * ppp, PANEL[1] * ppp],
    )
    .expect("a 800x600 panel is a valid viewport")
}

/// The camera most tests share: the equator at zoom 6.
fn view() -> MapView {
    view_at(6.0, PPP)
}

/// Where `lon`/`lat` lands on screen under [`view`].
fn at(lon: f64, lat: f64) -> Pos2 {
    to_screen(view(), ORIGIN, PPP, LonLat::new(lon, lat))
}

/// The lon/lat that lands `offset` points from the map's origin.
fn offset_lon_lat(offset: egui::Vec2) -> LonLat {
    let target = at(0.0, 0.0) + offset;
    view().screen_to_lon_lat([target.x * PPP, target.y * PPP])
}

fn position(values: &[f64]) -> Position {
    values.to_vec()
}

fn point_feature(lon: f64, lat: f64) -> Feature {
    let point = Point::new(position(&[lon, lat])).expect("a two-element position is a Point");
    Feature::new(Some(Geometry::Point(point)), Some(Properties::new()))
}

fn line_feature(coords: &[[f64; 2]]) -> Feature {
    let line = LineString::new(coords.iter().map(|pair| position(pair)).collect())
        .expect("at least two positions");
    Feature::new(Some(Geometry::LineString(line)), Some(Properties::new()))
}

fn polygon_feature(ring: &[[f64; 2]]) -> Feature {
    let polygon = Polygon::new(vec![ring.iter().map(|pair| position(pair)).collect()])
        .expect("at least one ring");
    Feature::new(Some(Geometry::Polygon(polygon)), Some(Properties::new()))
}

/// One layer's collection, shared.
fn collection(features: Vec<Feature>) -> (LayerId, Arc<FeatureCollection>) {
    (LayerId::new(), Arc::new(FeatureCollection::new(features)))
}

/// An index over one layer.
fn indexed(features: Vec<Feature>) -> (LayerId, Arc<FeatureCollection>, SnapIndex) {
    let (id, shared) = collection(features);
    let mut index = SnapIndex::default();
    index.rebuild_if_stale(&[(id, Arc::clone(&shared))]);
    (id, shared, index)
}

/// Everything on, at the default tolerance.
fn settings() -> SnapSettings {
    SnapSettings::default()
}

#[test]
fn vertex_snap_returns_the_bit_exact_stored_position() {
    // Deliberately not round numbers, and guarded against passing for the wrong
    // reason: the one plausible wrong implementation is to unproject the pixel
    // the vertex was drawn at, so the fixture has to be a position that path
    // visibly perturbs. It is — `MapView::lon_lat_to_screen` returns `f32`, so a
    // screen round trip quantises to about 1e-7 degrees at zoom 6, seven orders
    // above an `f64` ulp here. The guard is stated as that magnitude rather than
    // as a bit difference on a `to_world().to_lon_lat()` round trip: the latter
    // is a last-bit question about the host's `tan`/`ln`/`sinh`/`atan`, which a
    // quarter of latitudes answer "unchanged" on glibc 2.35/x86_64 and which
    // every libm answers differently.
    let lon = 12.345_678_901_234_567;
    let lat = 34.567_890_123_456_78;
    let screen = at(lon, lat);
    let round_tripped = view().screen_to_lon_lat([screen.x * PPP, screen.y * PPP]);
    assert!(
        (round_tripped.lon - lon).abs() > 1e-9 && (round_tripped.lat - lat).abs() > 1e-9,
        "the fixture must be a position a screen round trip perturbs, or the \
         assertions below prove nothing: {round_tripped:?}"
    );

    let (id, _shared, index) = indexed(vec![point_feature(lon, lat)]);
    let pointer = screen + vec2(3.0, 2.0);
    let result = index
        .query(view(), ORIGIN, pointer, settings(), None, PPP)
        .expect("a vertex three points away is well inside the tolerance");

    assert_eq!(result.kind, SnapKind::Vertex);
    assert_eq!(result.layer, id);
    assert_eq!(result.feature, 0);
    assert_eq!(result.vertex, Some(VertexRef::new(0)));
    assert_eq!(
        result.position.lon.to_bits(),
        lon.to_bits(),
        "a snapped vertex is the stored coordinate, never a screen round trip"
    );
    assert_eq!(result.position.lat.to_bits(), lat.to_bits());
    assert!(
        (result.distance_pt - 13.0_f32.sqrt()).abs() < 0.05,
        "distance is reported in egui points: {}",
        result.distance_pt
    );
    assert!((result.screen_pt - screen).length() < 0.01);
}

#[test]
fn edge_snap_returns_the_clamped_perpendicular_foot() {
    let end = offset_lon_lat(vec2(400.0, 0.0));
    let (_id, _shared, index) = indexed(vec![line_feature(&[[0.0, 0.0], [end.lon, 0.0]])]);
    // Vertices off, so the endpoint clamp is what is being measured rather than
    // the vertex that sits on it.
    let edges_only = SnapSettings {
        to_vertices: false,
        ..settings()
    };
    let start = at(0.0, 0.0);

    let foot = index
        .query(
            view(),
            ORIGIN,
            start + vec2(200.0, 5.0),
            edges_only,
            None,
            PPP,
        )
        .expect("five points off a segment is inside the tolerance");
    assert_eq!(foot.kind, SnapKind::Edge);
    assert_eq!(foot.vertex, None, "an edge snap names no handle");
    assert!(
        (foot.screen_pt.x - (start.x + 200.0)).abs() < 0.05,
        "the perpendicular foot keeps the pointer's position along the segment: {:?}",
        foot.screen_pt
    );
    assert!((foot.screen_pt.y - start.y).abs() < 0.05);
    assert!((foot.distance_pt - 5.0).abs() < 0.05);

    // Past the end, the foot is clamped to the endpoint …
    let clamped = index
        .query(
            view(),
            ORIGIN,
            start + vec2(405.0, 0.0),
            edges_only,
            None,
            PPP,
        )
        .expect("five points past the end is still within tolerance of the end");
    assert!(
        (clamped.screen_pt.x - (start.x + 400.0)).abs() < 0.05,
        "the foot never leaves the segment: {:?}",
        clamped.screen_pt
    );

    // … and far past it there is nothing, which an unclamped projection onto the
    // infinite line would report as distance zero.
    assert_eq!(
        index.query(
            view(),
            ORIGIN,
            start + vec2(600.0, 0.0),
            edges_only,
            None,
            PPP
        ),
        None
    );
    assert_eq!(
        index.query(
            view(),
            ORIGIN,
            start + vec2(200.0, 5.0),
            SnapSettings {
                enabled: false,
                ..settings()
            },
            None,
            PPP
        ),
        None,
        "snapping off means snapping off"
    );
}

#[test]
fn vertex_snap_wins_over_a_strictly_nearer_edge() {
    let left = offset_lon_lat(vec2(-200.0, 0.0));
    let right = offset_lon_lat(vec2(200.0, 0.0));
    // A long horizontal edge one point below the pointer, and a lone vertex
    // eight points to its right.
    let vertex = offset_lon_lat(vec2(8.0, 1.0));
    let (_id, _shared, index) = indexed(vec![
        line_feature(&[[left.lon, 0.0], [right.lon, 0.0]]),
        point_feature(vertex.lon, vertex.lat),
    ]);

    let pointer = at(0.0, 0.0) + vec2(0.0, 1.0);
    let result = index
        .query(view(), ORIGIN, pointer, settings(), None, PPP)
        .expect("both candidates are inside the tolerance");
    assert_eq!(
        result.kind,
        SnapKind::Vertex,
        "a vertex is something the user placed; an edge is only the space \
         between two of them"
    );
    assert_eq!(result.feature, 1);
    assert!(
        result.distance_pt > 7.0,
        "and it won despite being much further away: {}",
        result.distance_pt
    );

    // With vertices switched off the nearer edge is free to win.
    let edge = index
        .query(
            view(),
            ORIGIN,
            pointer,
            SnapSettings {
                to_vertices: false,
                ..settings()
            },
            None,
            PPP,
        )
        .expect("the edge is one point away");
    assert_eq!(edge.kind, SnapKind::Edge);
    assert!(edge.distance_pt < 1.5);
}

#[test]
fn tolerance_holds_at_two_zoom_levels_and_two_pixels_per_point() {
    for zoom in [3.0_f64, 12.0] {
        for ppp in [1.0_f32, 2.0] {
            let view = view_at(zoom, ppp);
            let centre = to_screen(view, ORIGIN, ppp, LonLat::new(0.0, 0.0));
            let (_id, _shared, index) = indexed(vec![point_feature(0.0, 0.0)]);

            let inside = index.query(
                view,
                ORIGIN,
                centre + vec2(SNAP_TOLERANCE_PT - 1.0, 0.0),
                settings(),
                None,
                ppp,
            );
            let hit = inside.unwrap_or_else(|| {
                panic!("just inside the tolerance must snap at zoom {zoom}, ppp {ppp}")
            });
            assert!(
                (hit.distance_pt - (SNAP_TOLERANCE_PT - 1.0)).abs() < 0.05,
                "the tolerance is in egui points at every zoom and every scale: \
                 got {} at zoom {zoom}, ppp {ppp}",
                hit.distance_pt
            );

            assert_eq!(
                index.query(
                    view,
                    ORIGIN,
                    centre + vec2(SNAP_TOLERANCE_PT + 2.0, 0.0),
                    settings(),
                    None,
                    ppp
                ),
                None,
                "past the tolerance is a miss at zoom {zoom}, ppp {ppp}"
            );
        }
    }
}

#[test]
fn index_rebuilds_when_an_arc_changes_and_not_on_a_camera_change() {
    let (id, shared) = collection(vec![line_feature(&[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])]);
    let mut index = SnapIndex::default();
    index.rebuild_if_stale(&[(id, Arc::clone(&shared))]);
    assert_eq!(index.generation(), 1);
    assert_eq!(index.segment_count(), 2);

    index.rebuild_if_stale(&[(id, Arc::clone(&shared))]);
    assert_eq!(
        index.generation(),
        1,
        "the same Arc, layer for layer, is a no-op"
    );

    // Panning, zooming and flinging are all just different query arguments.
    for zoom in [1.0_f64, 6.0, 18.0] {
        let view = MapView::new(LonLat::new(40.0, 20.0), zoom, PANEL).expect("a valid viewport");
        let _ignored = index.query(view, ORIGIN, pos2(10.0, 10.0), settings(), None, PPP);
    }
    assert_eq!(
        index.generation(),
        1,
        "the index is in world space: a camera move cannot invalidate it"
    );

    // A different collection with identical *contents* is still a different
    // collection: staleness is Arc identity, never a value comparison.
    let same_content = Arc::new(FeatureCollection::new(vec![line_feature(&[
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
    ])]));
    index.rebuild_if_stale(&[(id, same_content)]);
    assert_eq!(index.generation(), 2);

    // So is the same collection under a different layer id.
    index.rebuild_if_stale(&[(LayerId::new(), Arc::clone(&shared))]);
    assert_eq!(index.generation(), 3);

    // And so is a longer source list.
    let (other, other_features) = collection(vec![point_feature(5.0, 5.0)]);
    index.rebuild_if_stale(&[(id, Arc::clone(&shared)), (other, other_features)]);
    assert_eq!(index.generation(), 4);
    assert_eq!(index.segment_count(), 3, "a point is a zero-length segment");
    assert_eq!(index.source_count(), 2);

    index.clear();
    assert_eq!(index.segment_count(), 0);
    assert!(!index.is_degraded());
}

#[test]
fn index_degrades_to_the_active_layer_above_the_segment_cap_and_reports_it() {
    let active = collection(vec![line_feature(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])]);
    let reference = collection(vec![line_feature(&[
        [10.0, 0.0],
        [11.0, 0.0],
        [12.0, 0.0],
        [13.0, 0.0],
        [14.0, 0.0],
        [15.0, 0.0],
    ])]);
    let sources = [
        (active.0, Arc::clone(&active.1)),
        (reference.0, Arc::clone(&reference.1)),
    ];

    // Two segments plus five is seven: under a generous budget everything is in.
    let mut roomy = SnapIndex::with_budget(100);
    roomy.rebuild_if_stale(&sources);
    assert_eq!(roomy.segment_count(), 7);
    assert!(!roomy.is_degraded());

    // Under a tight one only the active layer — which is deliberately first — is
    // kept, and the index says so rather than going silently partial.
    let mut tight = SnapIndex::with_budget(3);
    tight.rebuild_if_stale(&sources);
    assert!(tight.is_degraded());
    assert_eq!(tight.segment_count(), 2, "the active layer, entire");
    assert_eq!(
        tight.source_count(),
        2,
        "both Arcs are still held, or staleness could not be detected"
    );

    let reference_view =
        MapView::new(LonLat::new(12.0, 0.0), 6.0, PANEL).expect("a valid viewport");
    let on_reference = to_screen(reference_view, ORIGIN, PPP, LonLat::new(12.0, 0.0));
    assert_eq!(
        tight.query(reference_view, ORIGIN, on_reference, settings(), None, PPP),
        None,
        "the dropped layer really is dropped"
    );
    assert!(
        roomy
            .query(reference_view, ORIGIN, on_reference, settings(), None, PPP)
            .is_some(),
        "and the same query finds it when the budget allows"
    );
}

/// Regression (finding 100): the budget used to be checked only BETWEEN
/// features, so one oversized feature — a dissolved coastline, a merged
/// parcel layer — was indexed whole regardless of `max_segments`. Now the
/// check sits inside `index_geometry`'s own per-position loop.
#[test]
fn a_single_oversized_feature_is_capped_mid_feature() {
    let huge: Vec<[f64; 2]> = (0..5_000).map(|step| [step as f64 * 0.001, 0.0]).collect();
    let (id, shared) = collection(vec![line_feature(&huge)]);
    let mut index = SnapIndex::with_budget(100);
    index.rebuild_if_stale(&[(id, shared)]);
    assert!(index.is_degraded());
    assert!(
        index.segment_count() <= 100,
        "one feature alone must not be indexed past the budget: {} segments",
        index.segment_count()
    );
}

#[test]
fn a_moving_set_excludes_every_marked_vertex_and_their_segments() {
    let left = offset_lon_lat(vec2(-200.0, 0.0));
    let right = offset_lon_lat(vec2(200.0, 0.0));
    let (id, _shared, index) = indexed(vec![line_feature(&[
        [left.lon, 0.0],
        [0.0, 0.0],
        [right.lon, 0.0],
    ])]);
    let grabbed = VertexRef::new(1);
    let companion = VertexRef::new(0);
    let on_companion = at(left.lon, 0.0);

    // With only the grabbed vertex excluded (the v1.1 shape), the pointer
    // over the OTHER marked vertex still snaps to its stale stored position —
    // a stationary attractor that would yank the set backwards.
    assert!(
        index
            .query(
                view(),
                ORIGIN,
                on_companion,
                settings(),
                Some((id, 0, grabbed)),
                PPP
            )
            .is_some(),
        "the companion is an attractor under single-vertex exclusion",
    );
    // The set exclusion covers every moving vertex and their segments.
    let marked = [companion, grabbed];
    assert_eq!(
        index.query_excluding_set(
            view(),
            ORIGIN,
            on_companion,
            settings(),
            Some((id, 0, &marked)),
            PPP,
        ),
        None,
        "nothing that is moving may attract the grabbed vertex",
    );
}

#[test]
fn the_dragged_vertex_and_its_two_adjacent_segments_are_excluded() {
    let left = offset_lon_lat(vec2(-200.0, 0.0));
    let right = offset_lon_lat(vec2(200.0, 0.0));
    let (id, _shared, index) = indexed(vec![line_feature(&[
        [left.lon, 0.0],
        [0.0, 0.0],
        [right.lon, 0.0],
    ])]);
    let middle = at(0.0, 0.0);
    let dragged = VertexRef::new(1);

    let unexcluded = index
        .query(view(), ORIGIN, middle, settings(), None, PPP)
        .expect("the pointer is exactly on vertex 1");
    assert_eq!(unexcluded.vertex, Some(dragged));

    assert_eq!(
        index.query(
            view(),
            ORIGIN,
            middle,
            settings(),
            Some((id, 0, dragged)),
            PPP
        ),
        None,
        "a vertex that snapped to itself — or to either segment it is an \
         endpoint of — could never be moved at all"
    );

    // Its neighbours are still perfectly good targets: only the two segments
    // that follow the pointer are out.
    let neighbour = index
        .query(
            view(),
            ORIGIN,
            at(left.lon, 0.0),
            settings(),
            Some((id, 0, dragged)),
            PPP,
        )
        .expect("vertex 0 is not the dragged one");
    assert_eq!(neighbour.vertex, Some(VertexRef::new(0)));

    // And so is another feature lying exactly under the dragged vertex — which
    // is the whole reason snapping exists.
    let (other_id, _other, with_neighbour) = indexed(vec![
        line_feature(&[[left.lon, 0.0], [0.0, 0.0], [right.lon, 0.0]]),
        point_feature(0.0, 0.0),
    ]);
    let across = with_neighbour
        .query(
            view(),
            ORIGIN,
            middle,
            settings(),
            Some((other_id, 0, dragged)),
            PPP,
        )
        .expect("feature 1 sits on the same spot and is not excluded");
    assert_eq!(across.feature, 1);

    // A ring excludes across its wrap, too: vertex 0's two segments are the
    // first and the closing one.
    let corner = offset_lon_lat(vec2(200.0, -200.0));
    let (ring_id, _ring, ring_index) = indexed(vec![polygon_feature(&[
        [0.0, 0.0],
        [corner.lon, 0.0],
        [corner.lon, corner.lat],
        [0.0, corner.lat],
        [0.0, 0.0],
    ])]);
    let on_closing_edge = at(0.0, corner.lat / 2.0);
    assert!(
        ring_index
            .query(view(), ORIGIN, on_closing_edge, settings(), None, PPP)
            .is_some()
    );
    assert_eq!(
        ring_index.query(
            view(),
            ORIGIN,
            on_closing_edge,
            SnapSettings {
                to_vertices: false,
                ..settings()
            },
            Some((ring_id, 0, VertexRef::new(0))),
            PPP
        ),
        None,
        "the closing segment is one of vertex 0's two neighbours"
    );

    // An insert names the vertex it will *become*. On a ring's wrap segment that
    // is the append index, which no plain index comparison matches — and without
    // it a vertex pulled off the closing edge snaps straight back onto it.
    assert_eq!(
        ring_index.query(
            view(),
            ORIGIN,
            on_closing_edge,
            settings(),
            Some((ring_id, 0, VertexRef::new(4))),
            PPP
        ),
        None,
        "the segment a midpoint ghost was pulled out of must not pull it back"
    );
}

/// Regression (finding 94): the edge branch of [`SnapIndex::query_excluding_set`]
/// used to scan the whole marked set linearly instead of the binary search
/// its own doc promises. A single-element or two-element marked set (every
/// other test's shape) cannot tell the difference — this exercises the
/// wrap-adjacency case (a segment's address matching an excluded vertex's
/// index plus one, the ring-closing-segment rule) with the matching entry
/// buried behind unrelated ones in a sorted, multi-entry set, so a rewrite
/// that got the binary search wrong (or searched the wrong field order)
/// would miss it.
#[test]
fn set_exclusion_finds_the_ring_wrap_case_among_unrelated_marks() {
    let corner = offset_lon_lat(vec2(200.0, -200.0));
    let (ring_id, _ring, ring_index) = indexed(vec![polygon_feature(&[
        [0.0, 0.0],
        [corner.lon, 0.0],
        [corner.lon, corner.lat],
        [0.0, corner.lat],
        [0.0, 0.0],
    ])]);
    let on_closing_edge = at(0.0, corner.lat / 2.0);
    let edges_only = SnapSettings {
        to_vertices: false,
        ..settings()
    };

    // Vertex 4 is the closing segment's "append" address (see
    // `the_dragged_vertex_and_its_two_adjacent_segments_are_excluded`); 1
    // and 2 are unrelated marks sorted around it.
    let marked = [VertexRef::new(1), VertexRef::new(2), VertexRef::new(4)];
    assert_eq!(
        ring_index.query_excluding_set(
            view(),
            ORIGIN,
            on_closing_edge,
            edges_only,
            Some((ring_id, 0, &marked)),
            PPP,
        ),
        None,
        "the closing segment is excluded even though its address (4) is \
         neither the first nor the last entry of the marked set"
    );

    // Without vertex 4 in the set, the same segment is a valid target again —
    // an unrelated marked set must not exclude it.
    let unrelated = [VertexRef::new(1), VertexRef::new(2)];
    assert!(
        ring_index
            .query_excluding_set(
                view(),
                ORIGIN,
                on_closing_edge,
                edges_only,
                Some((ring_id, 0, &unrelated)),
                PPP,
            )
            .is_some(),
        "an unrelated marked set must not exclude the closing segment"
    );
}

#[test]
fn sketch_start_snap_closes_a_ring() {
    let id = LayerId::new();
    let first = LonLat::new(12.345_678_901_234_567, 34.567_890_123_456_78);
    let on_screen = at(first.lon, first.lat);

    let closing = snap_to_sketch_start(
        view(),
        ORIGIN,
        on_screen + vec2(4.0, 0.0),
        settings(),
        id,
        first,
        PPP,
    )
    .expect("four points from the first vertex closes the ring");
    assert_eq!(closing.kind, SnapKind::SketchStart);
    assert_eq!(closing.layer, id);
    assert_eq!(closing.vertex, Some(VertexRef::new(0)));
    assert_eq!(
        closing.position.lon.to_bits(),
        first.lon.to_bits(),
        "closing a ring lands on the exact position the ring started at"
    );
    assert_eq!(closing.position.lat.to_bits(), first.lat.to_bits());
    assert!((closing.distance_pt - 4.0).abs() < 0.05);

    assert_eq!(
        snap_to_sketch_start(
            view(),
            ORIGIN,
            on_screen + vec2(SNAP_TOLERANCE_PT + 2.0, 0.0),
            settings(),
            id,
            first,
            PPP,
        ),
        None,
        "past the tolerance a sketch does not close by accident"
    );
    assert_eq!(
        snap_to_sketch_start(
            view(),
            ORIGIN,
            on_screen,
            SnapSettings {
                enabled: false,
                ..settings()
            },
            id,
            first,
            PPP,
        ),
        None
    );
}

/// A deterministic linear congruential generator.
///
/// Deliberately not a dependency: a differential test that cannot be replayed
/// byte for byte from its seed is a test that reports failures nobody can
/// reproduce.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// A value in `0.0..1.0`.
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// A value in `low..high`.
    fn next_range(&mut self, low: f64, high: f64) -> f64 {
        low + self.next_unit() * (high - low)
    }
}

#[test]
fn grid_lookup_agrees_with_a_brute_force_scan_on_1000_random_points() {
    let mut rng = Lcg(0x0BAD_5EED_1234_5678);
    // Forty polylines inside a three-degree box, with segments long enough
    // (~60 points on screen) that the middle of one is out of range of both its
    // endpoints — otherwise a vertex would win every single time and the edge
    // half of the ranking would never be exercised at all.
    let mut features = Vec::new();
    for _ in 0..40 {
        let mut coords = Vec::new();
        let mut lon = rng.next_range(0.0, 3.0);
        let mut lat = rng.next_range(0.0, 3.0);
        for _ in 0..12 {
            coords.push([lon, lat]);
            lon = (lon + rng.next_range(-0.35, 0.35)).clamp(0.0, 3.0);
            lat = (lat + rng.next_range(-0.35, 0.35)).clamp(0.0, 3.0);
        }
        features.push(line_feature(&coords));
    }
    let (id, shared, index) = indexed(features);
    assert_eq!(index.segment_count(), 40 * 11);

    let view = MapView::new(LonLat::new(1.5, 1.5), 8.0, PANEL).expect("a valid viewport");
    let settings = settings();
    let (north_west, _) = view.world_bounds();
    let world_pixels = view.world_pixels();
    let radius_world = f64::from(settings.tolerance_pt * PPP) / world_pixels;

    // Every segment, recomputed straight from the collection: the reference must
    // share no code with the thing it is checking.
    let mut segments: Vec<([f64; 2], [f64; 2], LonLat, LonLat)> = Vec::new();
    for feature in &shared.features {
        let Some(Geometry::LineString(line)) = feature.geometry.as_ref() else {
            panic!("the fixture is line strings");
        };
        for pair in line.coordinates.windows(2) {
            let [a, b] = pair else { continue };
            let (a_pos, b_pos) = (LonLat::new(a[0], a[1]), LonLat::new(b[0], b[1]));
            let (a_world, b_world) = (a_pos.to_world(), b_pos.to_world());
            segments.push(([a_world.x, a_world.y], [b_world.x, b_world.y], a_pos, b_pos));
        }
    }

    let mut vertex_hits = 0_usize;
    let mut edge_hits = 0_usize;
    let mut misses = 0_usize;
    for _ in 0..1000 {
        let pointer = pos2(
            rng.next_range(0.0, f64::from(PANEL[0])) as f32,
            rng.next_range(0.0, f64::from(PANEL[1])) as f32,
        );
        let local = pointer - ORIGIN;
        let at_world = [
            north_west.x + f64::from(local.x * PPP) / world_pixels,
            north_west.y + f64::from(local.y * PPP) / world_pixels,
        ];

        let mut best_vertex = f64::INFINITY;
        let mut best_edge = f64::INFINITY;
        for (a, b, _, _) in &segments {
            for end in [a, b] {
                let dx = at_world[0] - end[0];
                let dy = at_world[1] - end[1];
                best_vertex = best_vertex.min(dx.mul_add(dx, dy * dy));
            }
            let along = [b[0] - a[0], b[1] - a[1]];
            let length_sq = along[0].mul_add(along[0], along[1] * along[1]);
            let foot = if length_sq <= 0.0 {
                *a
            } else {
                let t = (((at_world[0] - a[0]) * along[0] + (at_world[1] - a[1]) * along[1])
                    / length_sq)
                    .clamp(0.0, 1.0);
                [along[0].mul_add(t, a[0]), along[1].mul_add(t, a[1])]
            };
            let dx = at_world[0] - foot[0];
            let dy = at_world[1] - foot[1];
            best_edge = best_edge.min(dx.mul_add(dx, dy * dy));
        }

        let expected = if best_vertex <= radius_world * radius_world {
            Some((SnapKind::Vertex, best_vertex.sqrt()))
        } else if best_edge <= radius_world * radius_world {
            Some((SnapKind::Edge, best_edge.sqrt()))
        } else {
            None
        };

        let found = index.query(view, ORIGIN, pointer, settings, None, PPP);
        match (expected, found) {
            (None, None) => misses += 1,
            (Some((kind, distance)), Some(result)) => {
                assert_eq!(
                    result.kind, kind,
                    "kind disagrees at {pointer:?}: the grid must find exactly \
                     what a full scan finds"
                );
                assert_eq!(result.layer, id);
                let expected_pt = (distance * world_pixels / f64::from(PPP)) as f32;
                assert!(
                    (result.distance_pt - expected_pt).abs() <= 1e-3,
                    "distance disagrees at {pointer:?}: grid {} vs scan {expected_pt}",
                    result.distance_pt
                );
                if kind == SnapKind::Vertex {
                    vertex_hits += 1;
                } else {
                    edge_hits += 1;
                }
            }
            (expected, found) => {
                panic!("the grid and the scan disagree at {pointer:?}: {expected:?} vs {found:?}")
            }
        }
    }

    // A differential test that never hit anything would pass trivially.
    assert!(vertex_hits > 20, "too few vertex hits: {vertex_hits}");
    assert!(edge_hits > 20, "too few edge hits: {edge_hits}");
    assert!(misses > 20, "too few misses: {misses}");
}
