// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the interaction shell: hit testing ([`super::hit`]), the
//! overlay's pure half ([`super::overlay`]) and the toolbar's action mapping
//! ([`super::toolbar`]).
//!
//! Separate from `edit/tests.rs`, which holds the command and stack suites and
//! is already past half of the 2000-line limit; keeping the screen-space suite
//! here leaves both files room for the stages still to come.
//!
//! Every geometry is placed by projecting a known lon/lat with
//! [`super::hit::to_screen`] and then offsetting by a pixel count, so a
//! tolerance assertion says what it means without any degrees-per-pixel
//! arithmetic in the test itself.

use super::hit::{
    CYCLE_SLOP_PT, FeatureHit, HANDLE_BUDGET, HitTarget, MIN_MIDPOINT_SEGMENT_PT, PICK_LINE_PT,
    PICK_POINT_PT, PickCycle, WorldBboxIndex, handle_position, midpoint_positions, pick,
    pick_features, to_screen, vertex_positions, visible_handle_count, visible_midpoint_positions,
    visible_vertex_positions, within_handle_budget,
};
use super::overlay;
use super::snap::{SnapKind, SnapResult};
use super::toolbar::{self, EditAction, mode_glyph};
use super::{EditCtx, EditMode, EditSelection, Handles, VertexDrag, VertexRef, plan_handles};
use crate::local_vector::GeometryKind;
use crate::style_panel::StyleKind;
use crate::ui_glyphs;
use egui::{Pos2, Rect, pos2, vec2};
use oxigeo::geojson::types::{
    Feature, FeatureCollection, Geometry, GeometryCollection, LineString, MultiLineString,
    MultiPoint, MultiPolygon, Point, Polygon, Position, Properties,
};
use oxigis_core::{LayerId, Project};
use oxigis_render::{LonLat, MapView};
use std::sync::Arc;

/// The map panel's top-left corner in these tests: the identity, so a screen
/// position reads directly as a projected pixel.
const ORIGIN: Pos2 = pos2(0.0, 0.0);
/// One physical pixel per egui point, so "points" and "pixels" coincide.
const PPP: f32 = 1.0;
/// The panel size every test uses.
const PANEL: [f32; 2] = [800.0, 600.0];

/// The camera every test shares: the equator at zoom 6.
fn view() -> MapView {
    MapView::new(LonLat::new(0.0, 0.0), 6.0, PANEL).expect("a 800x600 panel is a valid viewport")
}

/// The panel rect matching [`view`].
fn rect() -> Rect {
    Rect::from_min_size(ORIGIN, vec2(PANEL[0], PANEL[1]))
}

/// Where `lon`/`lat` lands on screen.
fn at(lon: f64, lat: f64) -> Pos2 {
    to_screen(view(), ORIGIN, PPP, LonLat::new(lon, lat))
}

/// The lon/lat that lands `offset` points away from the origin of the map.
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

/// A polygon feature; every ring is given **closed**.
fn polygon_feature(rings: &[Vec<[f64; 2]>]) -> Feature {
    let polygon = Polygon::new(
        rings
            .iter()
            .map(|ring| ring.iter().map(|pair| position(pair)).collect())
            .collect(),
    )
    .expect("at least one ring");
    Feature::new(Some(Geometry::Polygon(polygon)), Some(Properties::new()))
}

/// An axis-aligned closed ring.
fn square(min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) -> Vec<[f64; 2]> {
    vec![
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]
}

/// A shared collection plus its freshly built broad-phase index.
fn indexed(features: Vec<Feature>) -> (Arc<FeatureCollection>, WorldBboxIndex) {
    let features = Arc::new(FeatureCollection::new(features));
    let mut index = WorldBboxIndex::default();
    index.rebuild_if_stale(LayerId::new(), &features);
    (features, index)
}

/// The features under `at_pt`, best first.
fn hits(features: &FeatureCollection, index: &WorldBboxIndex, at_pt: Pos2) -> Vec<FeatureHit> {
    pick_features(features, index, view(), ORIGIN, at_pt, PPP, None)
}

/// [`hits`] with the layer's style consulted — what a click on a styled layer
/// really measures against.
fn styled_hits(
    features: &FeatureCollection,
    index: &WorldBboxIndex,
    at_pt: Pos2,
    style: &oxigis_core::LayerStyleSet,
) -> Vec<FeatureHit> {
    pick_features(features, index, view(), ORIGIN, at_pt, PPP, Some(style))
}

/// An [`EditCtx`] over `features`, with the shared camera and rect.
fn ctx<'a>(project: &'a Project, features: &'a Arc<FeatureCollection>) -> EditCtx<'a> {
    EditCtx {
        project,
        target: Some(LayerId::new()),
        features: Some(features),
        view: view(),
        rect: rect(),
        ppp: PPP,
    }
}

#[test]
fn point_pick_inside_and_just_outside_tolerance() {
    let (features, index) = indexed(vec![point_feature(0.0, 0.0)]);
    let center = at(0.0, 0.0);

    let inside = hits(&features, &index, center + vec2(PICK_POINT_PT - 1.0, 0.0));
    assert_eq!(inside.len(), 1, "just inside the tolerance ring is a hit");
    assert_eq!(inside[0].kind, GeometryKind::Point);
    assert!(!inside[0].inside, "a point is never an interior hit");

    let outside = hits(&features, &index, center + vec2(PICK_POINT_PT + 2.0, 0.0));
    assert!(
        outside.is_empty(),
        "past the tolerance is a miss, or the tolerance means nothing"
    );
}

#[test]
fn line_pick_uses_the_minimum_segment_distance_including_endpoint_clamps() {
    let end = offset_lon_lat(vec2(200.0, 0.0));
    let (features, index) = indexed(vec![line_feature(&[[0.0, 0.0], [end.lon, 0.0]])]);
    let start = at(0.0, 0.0);

    let near = hits(&features, &index, start + vec2(100.0, PICK_LINE_PT - 1.0));
    assert_eq!(near.len(), 1, "perpendicular distance inside the tolerance");
    assert!(near[0].distance_pt < PICK_LINE_PT);

    let far = hits(&features, &index, start + vec2(100.0, PICK_LINE_PT + 2.0));
    assert!(far.is_empty(), "perpendicular distance past the tolerance");

    // Exactly on the infinite line, but well beyond the segment's end: the
    // distance must be clamped to the endpoint rather than measured to the line.
    let beyond = hits(&features, &index, start + vec2(-40.0, 0.0));
    assert!(
        beyond.is_empty(),
        "an unclamped projection would report distance 0 here"
    );
}

#[test]
fn polygon_pick_inside_on_edge_and_inside_a_hole() {
    let corner = offset_lon_lat(vec2(300.0, -300.0));
    let hole_min = offset_lon_lat(vec2(100.0, -100.0));
    let hole_max = offset_lon_lat(vec2(200.0, -200.0));
    let exterior = square(0.0, corner.lat, corner.lon, 0.0);
    let mut hole = square(hole_min.lon, hole_max.lat, hole_max.lon, hole_min.lat);
    hole.reverse();
    let (features, index) = indexed(vec![polygon_feature(&[exterior, hole])]);
    let origin = at(0.0, 0.0);

    let inside = hits(&features, &index, origin + vec2(50.0, -50.0));
    assert_eq!(inside.len(), 1);
    assert!(inside[0].inside, "between the exterior and the hole is in");
    assert_eq!(inside[0].distance_pt, 0.0);

    let on_edge = hits(&features, &index, origin + vec2(-2.0, -150.0));
    assert_eq!(on_edge.len(), 1, "just outside, but on the boundary");
    assert!(!on_edge[0].inside);

    let in_hole = hits(&features, &index, origin + vec2(150.0, -150.0));
    assert!(
        in_hole.is_empty(),
        "the interior of a hole is outside the polygon — that is what a hole is"
    );

    let near_hole_edge = hits(&features, &index, origin + vec2(150.0, -102.0));
    assert_eq!(
        near_hole_edge.len(),
        1,
        "the hole's boundary is still grabbable, so a hole can be re-shaped"
    );
    assert!(!near_hole_edge[0].inside);
}

#[test]
fn null_geometry_features_are_never_picked() {
    let null = Feature::new(None, Some(Properties::new()));
    let (features, index) = indexed(vec![null, point_feature(0.0, 0.0)]);

    let found = hits(&features, &index, at(0.0, 0.0));
    assert_eq!(found.len(), 1);
    assert_eq!(
        found[0].feature, 1,
        "the null-geometry feature keeps its index but is never a candidate"
    );
    assert!(
        !index.may_contain(0, [0.5, 0.5], f64::INFINITY),
        "a feature with no geometry is culled by the broad phase itself"
    );
}

#[test]
fn pick_order_is_point_then_line_then_polygon_then_higher_index() {
    let corner = offset_lon_lat(vec2(200.0, -200.0));
    let middle = offset_lon_lat(vec2(100.0, -100.0));
    let (features, index) = indexed(vec![
        polygon_feature(&[square(0.0, corner.lat, corner.lon, 0.0)]),
        line_feature(&[[0.0, 0.0], [corner.lon, corner.lat]]),
        point_feature(middle.lon, middle.lat),
        polygon_feature(&[square(0.0, corner.lat, corner.lon, 0.0)]),
    ]);

    let order: Vec<usize> = hits(&features, &index, at(0.0, 0.0) + vec2(100.0, -100.0))
        .iter()
        .map(|hit| hit.feature)
        .collect();
    assert_eq!(
        order,
        vec![2, 1, 3, 0],
        "point, then line, then the topmost of two identical polygons"
    );
}

#[test]
fn vertex_outranks_midpoint_outranks_feature() {
    let end = offset_lon_lat(vec2(400.0, 0.0));
    let (features, index) = indexed(vec![line_feature(&[[0.0, 0.0], [end.lon, 0.0]])]);
    let project = Project::new("hit");
    let ctx = ctx(&project, &features);
    let start = at(0.0, 0.0);

    assert_eq!(
        pick(&ctx, &index, Some(0), start, true),
        Some(HitTarget::Vertex {
            feature: 0,
            at: VertexRef::new(0),
        })
    );
    assert_eq!(
        pick(&ctx, &index, Some(0), start + vec2(200.0, 0.0), true),
        Some(HitTarget::Midpoint {
            feature: 0,
            at: VertexRef::new(1),
        }),
        "the ghost inserts *at* index 1, between vertices 0 and 1"
    );
    assert_eq!(
        pick(&ctx, &index, Some(0), start + vec2(100.0, 0.0), true),
        Some(HitTarget::Feature { feature: 0 }),
        "a quarter of the way along is neither a handle nor a ghost"
    );
    assert_eq!(
        pick(&ctx, &index, Some(0), start, false),
        Some(HitTarget::Feature { feature: 0 }),
        "handles that were never drawn must never be clickable"
    );
    assert_eq!(
        pick(&ctx, &index, None, start, true),
        Some(HitTarget::Feature { feature: 0 }),
        "with nothing selected there are no handles to consider"
    );
    assert_eq!(
        HitTarget::Vertex {
            feature: 4,
            at: VertexRef::new(1)
        }
        .feature(),
        4
    );
}

#[test]
fn repeat_click_cycles_candidates_and_wraps() {
    let candidates = [
        FeatureHit {
            feature: 7,
            kind: GeometryKind::Point,
            distance_pt: 0.0,
            inside: false,
        },
        FeatureHit {
            feature: 3,
            kind: GeometryKind::Line,
            distance_pt: 1.0,
            inside: false,
        },
        FeatureHit {
            feature: 9,
            kind: GeometryKind::Polygon,
            distance_pt: 0.0,
            inside: true,
        },
    ];
    let mut cycle = PickCycle::default();
    let spot = pos2(100.0, 100.0);

    assert_eq!(cycle.next(spot, &candidates), Some(7));
    assert_eq!(cycle.position(), Some((1, 3)));
    assert_eq!(cycle.next(spot, &candidates), Some(3));
    assert_eq!(cycle.position(), Some((2, 3)));
    assert_eq!(cycle.next(spot, &candidates), Some(9));
    assert_eq!(
        cycle.next(spot, &candidates),
        Some(7),
        "the cycle wraps rather than sticking on the last candidate"
    );

    cycle.clear();
    assert_eq!(cycle.position(), None);
    assert_eq!(cycle.next(spot, &[]), None, "nothing under the pointer");
}

#[test]
fn moving_more_than_slop_or_changing_candidates_restarts_the_cycle() {
    let candidates = [
        FeatureHit {
            feature: 1,
            kind: GeometryKind::Point,
            distance_pt: 0.0,
            inside: false,
        },
        FeatureHit {
            feature: 2,
            kind: GeometryKind::Point,
            distance_pt: 1.0,
            inside: false,
        },
    ];
    let spot = pos2(100.0, 100.0);
    let mut cycle = PickCycle::default();

    assert_eq!(cycle.next(spot, &candidates), Some(1));
    let nudged = spot + vec2(CYCLE_SLOP_PT - 0.5, 0.0);
    assert_eq!(
        cycle.next(nudged, &candidates),
        Some(2),
        "a click inside the slop is still the same click"
    );
    let moved = nudged + vec2(CYCLE_SLOP_PT + 1.0, 0.0);
    assert_eq!(
        cycle.next(moved, &candidates),
        Some(1),
        "a click past the slop starts over at the best candidate"
    );

    // Same spot, different stack: the cursor must not carry across it.
    let other = [FeatureHit {
        feature: 5,
        kind: GeometryKind::Point,
        distance_pt: 0.0,
        inside: false,
    }];
    assert_eq!(cycle.next(moved, &candidates), Some(2));
    assert_eq!(cycle.next(moved, &other), Some(5));
    assert_eq!(cycle.position(), Some((1, 1)));
}

#[test]
fn closed_ring_handles_exclude_the_duplicate_last_position() {
    let corner = offset_lon_lat(vec2(200.0, -200.0));
    let feature = polygon_feature(&[square(0.0, corner.lat, corner.lon, 0.0)]);
    let geometry = feature.geometry.as_ref().expect("a polygon");

    let handles = vertex_positions(geometry);
    assert_eq!(
        handles.len(),
        4,
        "five stored positions, four addressable handles"
    );
    let indices: Vec<usize> = handles.iter().map(|(at, _)| at.index).collect();
    assert_eq!(indices, vec![0, 1, 2, 3]);

    // A closed ring's wrap segment still earns a ghost, addressed by the append
    // index — which is what makes "insert a vertex on the closing edge" work.
    let ghosts = midpoint_positions(geometry, view(), PPP);
    assert_eq!(ghosts.len(), 4, "four segments, four ghosts");
    assert_eq!(
        ghosts.last().map(|(at, _)| at.index),
        Some(4),
        "the wrap segment inserts past the last open position"
    );
}

#[test]
fn handle_budget_exceeded_yields_no_handles_and_no_handle_hits() {
    // `HANDLE_BUDGET + 1` vertices packed into 600 points of the 800-point
    // panel: the budget is measured on what is *in view*, so the fixture has to
    // put them all there.
    let first = offset_lon_lat(vec2(-300.0, 0.0)).lon;
    let step = offset_lon_lat(vec2(0.3, 0.0)).lon;
    let coords: Vec<[f64; 2]> = (0..=HANDLE_BUDGET)
        .map(|index| [step.mul_add(index as f64, first), 0.0])
        .collect();
    let (features, index) = indexed(vec![line_feature(&coords)]);
    let geometry = features.features[0]
        .geometry
        .as_ref()
        .expect("a line string");
    assert!(vertex_positions(geometry).len() > HANDLE_BUDGET);
    assert_eq!(
        visible_handle_count(geometry, view(), rect(), PPP),
        HANDLE_BUDGET + 1,
        "every vertex of this fixture is on screen"
    );
    assert!(!within_handle_budget(geometry, view(), rect(), PPP));

    let project = Project::new("budget");
    let ctx = ctx(&project, &features);
    assert_eq!(
        pick(&ctx, &index, Some(0), at(0.0, 0.0), true),
        Some(HitTarget::Feature { feature: 0 }),
        "past the budget nothing is drawn, so nothing may be grabbed either"
    );

    // …and the plan the overlay draws from agrees, and says so on the plate.
    let plan = plan_handles(&ctx, EditMode::Select, Some(EditSelection::feature(0)));
    assert_eq!(plan, Handles::Suppressed(HANDLE_BUDGET + 1));
    assert!(!plan.is_active(), "nothing draggable is drawn");
    let hint = overlay::handle_hint(plan).expect("a suppressed set must say so");
    assert!(hint.contains(&(HANDLE_BUDGET + 1).to_string()), "{hint}");
    assert!(hint.contains("zoom in"), "{hint}");
    assert_eq!(
        overlay::handle_hint(Handles::Active),
        None,
        "a drawn set has nothing to explain"
    );
}

#[test]
fn handles_are_culled_to_the_view_before_the_budget_is_measured() {
    // The same vertex count, spread over eighty panel widths: only the sliver on
    // screen counts, so the feature stays editable. Budgeting the whole feature
    // instead would make the geometry that most needs editing the only geometry
    // that cannot be edited. The 32-point spacing is deliberately above
    // `MIN_MIDPOINT_SEGMENT_PT`, so the ghosts exist to be culled.
    let first = offset_lon_lat(vec2(-400.0, 0.0)).lon;
    let step = offset_lon_lat(vec2(32.0, 0.0)).lon;
    let coords: Vec<[f64; 2]> = (0..=HANDLE_BUDGET)
        .map(|index| [step.mul_add(index as f64, first), 0.0])
        .collect();
    let (features, index) = indexed(vec![line_feature(&coords)]);
    let geometry = features.features[0]
        .geometry
        .as_ref()
        .expect("a line string");

    let in_view = visible_handle_count(geometry, view(), rect(), PPP);
    assert!(
        in_view > 0 && in_view <= HANDLE_BUDGET,
        "a sliver of a huge feature is still a small set: {in_view}"
    );
    assert!(within_handle_budget(geometry, view(), rect(), PPP));
    assert_eq!(
        visible_vertex_positions(geometry, view(), rect(), PPP).len(),
        in_view,
        "the culled list and the counted one are the same set"
    );
    assert!(
        visible_midpoint_positions(geometry, view(), rect(), PPP).len()
            < midpoint_positions(geometry, view(), PPP).len(),
        "ghosts are culled too"
    );

    let project = Project::new("cull");
    let ctx = ctx(&project, &features);
    assert_eq!(
        plan_handles(&ctx, EditMode::Select, Some(EditSelection::feature(0))),
        Handles::Active
    );
    assert_eq!(
        pick(&ctx, &index, Some(0), at(first, 0.0), true),
        Some(HitTarget::Vertex {
            feature: 0,
            at: VertexRef::new(0),
        }),
        "the first vertex is on screen, drawn, and therefore grabbable"
    );
    assert_eq!(
        plan_handles(&ctx, EditMode::DrawPoint, Some(EditSelection::feature(0))),
        Handles::None,
        "a drawing tool shows no handles at all"
    );
    assert_eq!(plan_handles(&ctx, EditMode::Select, None), Handles::None);
}

#[test]
fn handle_position_reads_back_what_was_drawn() {
    let corner = offset_lon_lat(vec2(200.0, -200.0));
    let feature = polygon_feature(&[square(0.0, corner.lat, corner.lon, 0.0)]);
    let geometry = feature.geometry.as_ref().expect("a polygon");

    let handle = handle_position(geometry, VertexRef::new(1), false, view(), PPP)
        .expect("ring vertex 1 exists");
    assert_eq!(handle, LonLat::new(corner.lon, corner.lat));

    let ghost = handle_position(geometry, VertexRef::new(4), true, view(), PPP)
        .expect("the wrap segment has a ghost");
    let expected = midpoint_positions(geometry, view(), PPP)
        .into_iter()
        .find_map(|(at, position)| (at == VertexRef::new(4)).then_some(position))
        .expect("the same ghost");
    assert_eq!(ghost, expected, "a drag starts where the ghost was drawn");

    assert_eq!(
        handle_position(geometry, VertexRef::at(3, 0, 0), false, view(), PPP),
        None,
        "a part that does not exist has no handle"
    );
}

#[test]
fn midpoint_ghosts_are_omitted_below_the_minimum_segment_length() {
    let short = offset_lon_lat(vec2(MIN_MIDPOINT_SEGMENT_PT - 4.0, 0.0));
    let long = offset_lon_lat(vec2(MIN_MIDPOINT_SEGMENT_PT * 6.0, 0.0));
    let feature = line_feature(&[[0.0, 0.0], [short.lon, 0.0], [long.lon, 0.0]]);
    let geometry = feature.geometry.as_ref().expect("a line string");

    let ghosts = midpoint_positions(geometry, view(), PPP);
    assert_eq!(
        ghosts.len(),
        1,
        "the sub-tolerance segment contributes nothing: {ghosts:?}"
    );
    assert_eq!(
        ghosts[0].0,
        VertexRef::new(2),
        "the surviving ghost belongs to the second segment"
    );
}

#[test]
fn world_bbox_broad_phase_excludes_off_screen_features() {
    let (features, index) = indexed(vec![point_feature(0.0, 0.0), point_feature(170.0, -60.0)]);
    assert_eq!(index.len(), 2);
    assert!(!index.is_empty());

    let here = LonLat::new(0.0, 0.0).to_world();
    let here = [here.x, here.y];
    assert!(index.may_contain(0, here, 1e-6));
    assert!(
        !index.may_contain(1, here, 1e-6),
        "a feature on the other side of the world is never projected at all"
    );
    assert!(
        index.may_contain(2, here, 1e-6),
        "a feature the index has no entry for degrades to 'test it', never to 'skip it'"
    );

    // And the whole pick agrees with the broad phase.
    let found = hits(&features, &index, at(0.0, 0.0));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].feature, 0);

    // The index holds its `Arc`, so re-syncing the same one is a no-op while a
    // different one rebuilds.
    let mut index = index;
    let id = index.layer().expect("an indexed layer");
    index.rebuild_if_stale(id, &features);
    assert_eq!(index.len(), 2);
    let (other, _) = indexed(vec![point_feature(1.0, 1.0)]);
    index.rebuild_if_stale(id, &other);
    assert_eq!(index.len(), 1, "a new collection rebuilds the boxes");
    index.clear();
    assert!(index.is_empty());
    assert_eq!(index.layer(), None);
}

#[test]
fn broad_phase_grid_covers_scattered_features_and_the_oversized_fallback_without_duplicates() {
    // Widely-spaced lines, each with real extent: its bounding box files into
    // several grid cells rather than the single cell a point's degenerate
    // box would, exercising the multi-cell path a click near either end of
    // the query span could otherwise double-count.
    let mut features: Vec<Feature> = (0..12)
        .map(|index| {
            let base = f64::from(index) * 8.0 - 48.0;
            line_feature(&[[base, 0.0], [base + 2.0, 0.5]])
        })
        .collect();
    // A near-world-spanning polygon: its box covers far more cells than the
    // index will ever file individually, so it must land in the oversized
    // fallback instead — and still be found, and found exactly once.
    let huge_index = features.len();
    features.push(polygon_feature(&[square(-179.0, -80.0, 179.0, 80.0)]));
    let (features, index) = indexed(features);

    for line_index in 0..huge_index {
        let base = f64::from(u32::try_from(line_index).expect("small index")) * 8.0 - 48.0;
        let found = hits(&features, &index, at(base, 0.0));
        let matches = found.iter().filter(|hit| hit.feature == line_index).count();
        assert_eq!(
            matches, 1,
            "line {line_index} at lon {base} must be found exactly once, not once per \
             grid cell its box spans: {found:?}"
        );
    }

    let far = hits(&features, &index, at(0.0, 40.0));
    let huge_hits = far.iter().filter(|hit| hit.feature == huge_index).count();
    assert_eq!(
        huge_hits, 1,
        "the oversized-fallback box must still be exactly one candidate: {far:?}"
    );
}

#[test]
fn the_toolbar_maps_every_mode_to_its_own_glyph_style_and_action() {
    let glyphs: Vec<&str> = EditMode::ALL.into_iter().map(mode_glyph).collect();
    let mut unique = glyphs.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), glyphs.len(), "every tool is distinguishable");

    // Distinguishable by the MARK, not just by the trailing word. Until
    // editing v1.5 three of the five labels opened with the same hollow
    // replacement box, so the whole-string assertion above passed on
    // "Select"/"Line"/"Polygon" alone — a test that certified the bug. No egui
    // context is needed to close that: `ui_glyphs::ALL` membership is
    // render-gated by that module's two pins, so composing the two is exactly
    // as strong.
    let marks: Vec<char> = glyphs
        .iter()
        .filter_map(|glyph| glyph.chars().next())
        .collect();
    assert_eq!(marks.len(), glyphs.len(), "every label opens with a mark");
    let mut unique_marks = marks.clone();
    unique_marks.sort_unstable();
    unique_marks.dedup();
    assert_eq!(
        unique_marks.len(),
        marks.len(),
        "two tools drawing the same mark are two tools the user cannot tell apart"
    );
    for mark in marks {
        assert!(
            ui_glyphs::ALL
                .iter()
                .any(|(glyph, _)| glyph.chars().any(|ch| ch == mark)),
            "U+{:04X} is drawn by the toolbar but is not in ui_glyphs::ALL, so nothing              proves it is not a hollow replacement box",
            mark as u32
        );
    }

    // The drawing tools and the layer styles round-trip, so "+ New layer ›
    // Polygon" cannot latch the point tool.
    for mode in EditMode::ALL {
        match toolbar::style_for_mode(mode) {
            Some(kind) => assert_eq!(toolbar::mode_for_style(kind), mode),
            None => assert!(!mode.is_drawing()),
        }
    }
    assert_eq!(
        toolbar::mode_for_style(StyleKind::Symbol),
        EditMode::DrawPoint,
        "a label layer is still a point layer to digitize into"
    );
    assert_eq!(
        EditAction::SetMode(EditMode::Select),
        EditAction::SetMode(EditMode::Select)
    );
    assert_ne!(
        EditAction::NewLayer(StyleKind::Fill),
        EditAction::NewLayer(StyleKind::Line)
    );
}

#[test]
fn the_hint_plate_is_silent_with_editing_off_and_names_the_cycle_otherwise() {
    let idle = crate::edit::Sketch::default();
    assert_eq!(
        overlay::hint_text(
            EditMode::Off,
            Some(EditSelection::feature(3)),
            Some((2, 4)),
            &idle
        ),
        None,
        "with editing off nothing at all is painted over the map"
    );

    let empty = overlay::hint_text(EditMode::Select, None, None, &idle).expect("a hint");
    assert!(empty.contains("click a feature"), "{empty}");

    let picked = overlay::hint_text(
        EditMode::Select,
        Some(EditSelection::feature(3)),
        Some((1, 1)),
        &idle,
    )
    .expect("a hint");
    assert!(picked.contains('3'), "{picked}");
    assert!(
        !picked.contains("cycle"),
        "a lone candidate must not advertise cycling: {picked}"
    );

    let stacked = overlay::hint_text(
        EditMode::Select,
        Some(EditSelection::feature(3)),
        Some((2, 4)),
        &idle,
    )
    .expect("a hint");
    assert!(stacked.contains("2 of 4"), "{stacked}");
    assert!(stacked.contains("cycle"), "{stacked}");

    for mode in [
        EditMode::DrawPoint,
        EditMode::DrawLine,
        EditMode::DrawPolygon,
    ] {
        let text = overlay::hint_text(mode, None, None, &idle).expect("a hint");
        assert!(text.starts_with(mode.label()), "{text}");
    }
}

/// Whether `p` falls inside (or exactly on an edge of) the triangle
/// `a`-`b`-`c`, by the sign of the three edge cross-products.
///
/// A [`Mesh`][egui::Mesh]'s triangles are the only ground truth for what a
/// paint call actually filled, so the fill-versus-hole test below samples
/// them directly with this rather than re-deriving the answer from the same
/// scanline logic it exists to check.
fn point_in_triangle(p: Pos2, a: Pos2, b: Pos2, c: Pos2) -> bool {
    fn sign(p1: Pos2, p2: Pos2, p3: Pos2) -> f32 {
        (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
    }
    let d1 = sign(p, a, b);
    let d2 = sign(p, b, c);
    let d3 = sign(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

#[test]
fn selection_outline_shape_count_is_independent_of_vertex_count() {
    // If the outline still pushed one `Shape::LineSegment` per consecutive
    // pair, the paint list would grow with vertex count. `stroke_passes` now
    // emits one `Shape::line` per PASS instead, so a 2-vertex and a
    // 201-vertex line must cost the paint list exactly the same — the
    // assertion that would have caught the bug this fixes.
    let sparse = line_feature(&[[0.0, 0.0], [10.0, 0.0]]);
    let dense_coords: Vec<[f64; 2]> = (0..=200)
        .map(|index| [f64::from(index) * 0.05, 0.0])
        .collect();
    let dense = line_feature(&dense_coords);

    let egui_ctx = egui::Context::default();
    let mut scratch = overlay::OverlayScratch::default();
    let mut shape_counts = Vec::new();
    for feature in [&sparse, &dense] {
        let geometry = feature.geometry.as_ref().expect("a line string");
        let raw_input = egui::RawInput {
            screen_rect: Some(rect()),
            ..Default::default()
        };
        let output = egui_ctx.run_ui(raw_input, |ui| {
            let painter = ui.painter_at(rect());
            overlay::paint_selection(&painter, view(), rect(), PPP, geometry, &mut scratch);
        });
        shape_counts.push(output.shapes.len());
    }
    assert_eq!(
        shape_counts[0], shape_counts[1],
        "a 2-vertex and a 201-vertex line must cost the paint list the same \
         shape count: {shape_counts:?}"
    );
    assert_eq!(
        shape_counts[0], 2,
        "one shape per stroke pass (halo, accent), not one per segment: {shape_counts:?}"
    );
}

#[test]
fn selection_fill_leaves_the_hole_and_the_concave_notch_untinted() {
    // Same fixture as `polygon_pick_inside_on_edge_and_inside_a_hole`, whose
    // hit-test already pins `origin + (50, -50)` as between the exterior and
    // the hole and `origin + (150, -150)` as inside the hole — reused here so
    // the two tests agree on what "inside the hole" means.
    let corner = offset_lon_lat(vec2(300.0, -300.0));
    let hole_min = offset_lon_lat(vec2(100.0, -100.0));
    let hole_max = offset_lon_lat(vec2(200.0, -200.0));
    let ring = |min_lon, min_lat, max_lon, max_lat| -> Vec<Position> {
        square(min_lon, min_lat, max_lon, max_lat)
            .iter()
            .map(|pair| position(pair))
            .collect()
    };
    let outer = ring(0.0, corner.lat, corner.lon, 0.0);
    let mut inner = ring(hole_min.lon, hole_max.lat, hole_max.lon, hole_min.lat);
    inner.reverse();
    let geometry =
        Geometry::Polygon(Polygon::new(vec![outer, inner]).expect("a polygon with a hole"));

    let egui_ctx = egui::Context::default();
    let mut scratch = overlay::OverlayScratch::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(rect()),
        ..Default::default()
    };
    let output = egui_ctx.run_ui(raw_input, |ui| {
        let painter = ui.painter_at(rect());
        overlay::paint_selection(&painter, view(), rect(), PPP, &geometry, &mut scratch);
    });

    let origin = at(0.0, 0.0);
    let inside_fill = origin + vec2(50.0, -50.0);
    let inside_hole = origin + vec2(150.0, -150.0);

    let mut painted_fill = false;
    let mut painted_hole = false;
    for clipped in &output.shapes {
        let egui::Shape::Mesh(mesh) = &clipped.shape else {
            continue;
        };
        for triangle in mesh.indices.chunks_exact(3) {
            let [a, b, c] = triangle else { continue };
            let (Some(a), Some(b), Some(c)) = (
                mesh.vertices.get(*a as usize),
                mesh.vertices.get(*b as usize),
                mesh.vertices.get(*c as usize),
            ) else {
                continue;
            };
            painted_fill |= point_in_triangle(inside_fill, a.pos, b.pos, c.pos);
            painted_hole |= point_in_triangle(inside_hole, a.pos, b.pos, c.pos);
        }
    }
    assert!(
        painted_fill,
        "the active-edge-table sweep must still tint between the exterior and the hole"
    );
    assert!(
        !painted_hole,
        "the active-edge-table sweep must still leave the hole's interior untinted"
    );
}

#[test]
fn the_selection_overlay_paints_every_geometry_kind_without_panicking() {
    let corner = offset_lon_lat(vec2(300.0, -300.0));
    let hole_min = offset_lon_lat(vec2(100.0, -100.0));
    let hole_max = offset_lon_lat(vec2(200.0, -200.0));
    let ring = |min_lon, min_lat, max_lon, max_lat| -> Vec<Position> {
        square(min_lon, min_lat, max_lon, max_lat)
            .iter()
            .map(|pair| position(pair))
            .collect()
    };
    let outer = ring(0.0, corner.lat, corner.lon, 0.0);
    let mut inner = ring(hole_min.lon, hole_max.lat, hole_max.lon, hole_min.lat);
    inner.reverse();

    // A deliberately concave ring: the case `epaint`'s own fan fill gets wrong,
    // and the reason the tint is scanline-filled instead.
    let concave = vec![
        position(&[0.0, 0.0]),
        position(&[corner.lon, 0.0]),
        position(&[corner.lon, corner.lat]),
        position(&[hole_min.lon, hole_max.lat]),
        position(&[hole_max.lon, hole_min.lat]),
        position(&[0.0, corner.lat]),
        position(&[0.0, 0.0]),
    ];

    let geometries = vec![
        (
            "Point",
            Geometry::Point(Point::new(position(&[0.0, 0.0])).expect("a point")),
        ),
        (
            "MultiPoint",
            Geometry::MultiPoint(
                MultiPoint::new(vec![
                    position(&[0.0, 0.0]),
                    position(&[corner.lon, corner.lat]),
                ])
                .expect("a multipoint"),
            ),
        ),
        (
            "LineString",
            Geometry::LineString(
                LineString::new(vec![
                    position(&[0.0, 0.0]),
                    position(&[corner.lon, corner.lat]),
                ])
                .expect("a line"),
            ),
        ),
        (
            "MultiLineString",
            Geometry::MultiLineString(
                MultiLineString::new(vec![vec![
                    position(&[0.0, 0.0]),
                    position(&[corner.lon, 0.0]),
                ]])
                .expect("a multiline"),
            ),
        ),
        (
            "Polygon with a hole",
            Geometry::Polygon(
                Polygon::new(vec![outer.clone(), inner.clone()]).expect("a polygon with a hole"),
            ),
        ),
        (
            "concave Polygon",
            Geometry::Polygon(Polygon::new(vec![concave]).expect("a concave polygon")),
        ),
        (
            "MultiPolygon",
            Geometry::MultiPolygon(
                MultiPolygon::new(vec![vec![outer, inner]]).expect("a multipolygon"),
            ),
        ),
        (
            "GeometryCollection",
            Geometry::GeometryCollection(
                GeometryCollection::new(vec![Geometry::Point(
                    Point::new(position(&[0.0, 0.0])).expect("a point"),
                )])
                .expect("a collection"),
            ),
        ),
    ];

    // Each kind paints into its **own** frame, so the assertion pins that
    // *this* kind emitted shapes — one shared paint list would let any broken
    // match arm (say, point markers becoming a no-op) hide behind the seven
    // kinds that still painted.
    let egui_ctx = egui::Context::default();
    let mut scratch = overlay::OverlayScratch::default();
    for (name, geometry) in &geometries {
        let raw_input = egui::RawInput {
            screen_rect: Some(rect()),
            ..Default::default()
        };
        let output = egui_ctx.run_ui(raw_input, |ui| {
            let painter = ui.painter_at(rect());
            overlay::paint_selection(&painter, view(), rect(), PPP, geometry, &mut scratch);
        });
        assert!(
            !output.shapes.is_empty(),
            "painting the selection of a {name} must put shapes in the paint list"
        );
    }

    let raw_input = egui::RawInput {
        screen_rect: Some(rect()),
        ..Default::default()
    };
    let output = egui_ctx.run_ui(raw_input, |ui| {
        let painter = ui.painter_at(rect());
        overlay::paint_hint(&painter, rect(), "Select — click a feature to pick it");
    });
    assert!(
        !output.shapes.is_empty(),
        "the hint plate must put shapes in the paint list"
    );
}

#[test]
fn the_handle_ghost_drag_and_snap_layers_all_reach_the_paint_list() {
    let corner = offset_lon_lat(vec2(200.0, -200.0));
    let feature = polygon_feature(&[square(0.0, corner.lat, corner.lon, 0.0)]);
    let geometry = feature.geometry.as_ref().expect("a polygon");
    let handles = visible_vertex_positions(geometry, view(), rect(), PPP);
    let ghosts = visible_midpoint_positions(geometry, view(), rect(), PPP);
    assert_eq!(handles.len(), 4);
    assert_eq!(ghosts.len(), 4);

    // A live gesture on handle 0, moved well off its stored position.
    let moved = offset_lon_lat(vec2(60.0, -60.0));
    let drag = VertexDrag {
        moved: true,
        ..VertexDrag::single(0, VertexRef::new(0), false, geometry.clone(), moved)
    };
    let previewed = overlay::drag_handles(handles.clone(), &drag);
    assert_eq!(previewed.len(), handles.len(), "a move adds no handle");
    assert_eq!(
        previewed[0].1, moved,
        "the handle is drawn where the vertex will land, not where the pointer is"
    );

    let inserting = VertexDrag {
        inserting: true,
        vertex: VertexRef::new(4),
        ..drag.clone()
    };
    let with_insert = overlay::drag_handles(handles.clone(), &inserting);
    assert_eq!(
        with_insert.len(),
        handles.len() + 1,
        "an insert gesture shows the vertex it is about to create"
    );

    let snaps = [SnapKind::Vertex, SnapKind::Edge, SnapKind::SketchStart].map(|kind| SnapResult {
        kind,
        layer: LayerId::new(),
        feature: 0,
        vertex: Some(VertexRef::new(0)),
        position: moved,
        screen_pt: at(0.0, 0.0),
        distance_pt: 3.0,
    });

    // Each visual layer paints into its **own** frame, so the assertion pins
    // that *this* layer emitted shapes — one shared paint list would let a
    // silently broken layer (say, a snap marker whose `SnapKind` arm draws
    // nothing) hide behind the ones that still painted.
    let egui_ctx = egui::Context::default();
    let painted = |paint: &dyn Fn(&egui::Painter, &mut overlay::OverlayScratch)| {
        let raw_input = egui::RawInput {
            screen_rect: Some(rect()),
            ..Default::default()
        };
        let mut scratch = overlay::OverlayScratch::default();
        let output = egui_ctx.run_ui(raw_input, |ui| {
            let painter = ui.painter_at(rect());
            paint(&painter, &mut scratch);
        });
        !output.shapes.is_empty()
    };
    assert!(
        painted(&|painter, _| overlay::paint_midpoint_ghosts(
            painter,
            view(),
            rect(),
            PPP,
            &ghosts
        )),
        "midpoint ghosts must reach the paint list"
    );
    assert!(
        painted(&|painter, _| overlay::paint_handles(
            painter,
            view(),
            rect(),
            PPP,
            &handles,
            Some(VertexRef::new(1)),
        )),
        "vertex handles must reach the paint list"
    );
    assert!(
        painted(&|painter, scratch| overlay::paint_drag(
            painter,
            view(),
            rect(),
            PPP,
            &drag,
            scratch
        )),
        "the move-drag preview must reach the paint list"
    );
    assert!(
        painted(&|painter, scratch| overlay::paint_drag(
            painter,
            view(),
            rect(),
            PPP,
            &inserting,
            scratch
        )),
        "the insert-drag preview must reach the paint list"
    );
    for snap in &snaps {
        assert!(
            painted(&|painter, _| overlay::paint_snap_marker(painter, snap)),
            "the {:?} snap marker must reach the paint list",
            snap.kind
        );
    }

    // Every geometry kind survives a drag preview, including the ones a vertex
    // gesture can never actually start on.
    let mut scratch = overlay::OverlayScratch::default();
    let point = Geometry::Point(Point::new(position(&[0.0, 0.0])).expect("a point"));
    let collection = Geometry::GeometryCollection(
        GeometryCollection::new(vec![point.clone()]).expect("a collection"),
    );
    let raw_input = egui::RawInput {
        screen_rect: Some(rect()),
        ..Default::default()
    };
    let _output = egui_ctx.run_ui(raw_input, |ui| {
        let painter = ui.painter_at(rect());
        for origin in [point.clone(), collection.clone()] {
            let drag = VertexDrag {
                origin,
                ..drag.clone()
            };
            overlay::paint_drag(&painter, view(), rect(), PPP, &drag, &mut scratch);
        }
    });
}

#[test]
fn the_plate_composes_the_lines_that_are_true_at_once() {
    assert_eq!(overlay::plate_text(None, &[]), None);
    assert_eq!(
        overlay::plate_text(Some("Select".to_string()), &[None, None]),
        Some("Select".to_string())
    );

    let composed = overlay::plate_text(
        Some("Select".to_string()),
        &[
            overlay::handle_hint(Handles::Suppressed(34_812)),
            Some(overlay::SNAP_DEGRADED_HINT.to_string()),
        ],
    )
    .expect("three true things");
    assert!(composed.starts_with("Select"), "{composed}");
    assert!(composed.contains("34812"), "{composed}");
    assert!(composed.contains("snapping"), "{composed}");

    // A warning with nothing else to say still reaches the plate on its own.
    let alone = overlay::plate_text(None, &[overlay::handle_hint(Handles::Suppressed(3))])
        .expect("a suppressed set is worth saying by itself");
    assert!(alone.starts_with('3'), "{alone}");
}

// ---------------------------------------------------------------------------
// Style-aware pick tolerance (thematic v1.6)
// ---------------------------------------------------------------------------

/// A point feature carrying one string property.
fn zoned_point(lon: f64, lat: f64, zone: &str) -> Feature {
    let point = Point::new(position(&[lon, lat])).expect("a two-element position is a Point");
    let mut properties = Properties::new();
    properties.insert("zone".to_string(), serde_json::Value::String(zone.into()));
    Feature::new(Some(Geometry::Point(point)), Some(properties))
}

/// A single-symbol set drawing markers of `radius`.
fn circle_set(radius: f32) -> oxigis_core::LayerStyleSet {
    oxigis_core::LayerStyleSet::new(oxigis_core::LayerStyle::Circle(
        oxigis_core::CircleStyle::new(radius, oxigis_core::Color::BLACK),
    ))
}

/// The world-space radius `points` egui points is worth under [`view`] —
/// exactly the conversion `pick_features`' broad phase makes.
fn radius_world(points: f32) -> f64 {
    f64::from(points * PPP) / view().world_pixels()
}

/// `at_pt` in world coordinates, the way the broad phase reads it.
fn world_at(at_pt: Pos2) -> [f64; 2] {
    let world = view()
        .screen_to_lon_lat([at_pt.x * PPP, at_pt.y * PPP])
        .to_world();
    [world.x, world.y]
}

#[test]
fn a_marker_is_picked_from_the_radius_it_is_actually_drawn_at() {
    // "What you can click is what you can see" — this module's own rule. A
    // 24 pt marker that only answered within 9 pt of its centre broke it for
    // every layer styled bigger than the shipped constant.
    let (features, index) = indexed(vec![point_feature(0.0, 0.0)]);
    let near = at(0.0, 0.0) + vec2(18.0, 0.0);
    assert!(
        hits(&features, &index, near).is_empty(),
        "18 pt is outside the {PICK_POINT_PT} pt constant",
    );
    assert_eq!(
        styled_hits(&features, &index, near, &circle_set(24.0)).len(),
        1,
        "and inside the marker the user is looking at",
    );
    // Past the marker's own edge it is a miss again — the tolerance follows
    // the drawing, it does not simply grow.
    let far = at(0.0, 0.0) + vec2(30.0, 0.0);
    assert!(styled_hits(&features, &index, far, &circle_set(24.0)).is_empty());
}

#[test]
fn the_broad_phase_reaches_as_far_as_the_widened_tolerance() {
    // The half that fails SILENTLY: `candidate_features` culls by bounding box
    // before anything is measured, so a broad radius left at the constant would
    // drop the very features a large style exists to make clickable — and only
    // on data whose boxes stop overlapping the pointer, never on a small
    // fixture. Asserted directly against the index rather than through a pick,
    // so no grid cell size can make it pass by accident.
    let (features, index) = indexed(vec![point_feature(0.0, 0.0)]);
    assert_eq!(features.features.len(), 1);
    let at_world = world_at(at(0.0, 0.0) + vec2(18.0, 0.0));
    assert!(
        !index.may_contain(0, at_world, radius_world(PICK_POINT_PT)),
        "the constant radius really does cull this feature",
    );
    assert!(
        index.may_contain(0, at_world, radius_world(24.0)),
        "the widened one must not",
    );
}

#[test]
fn a_style_never_tightens_a_tolerance_below_the_shipped_constant() {
    // A hairline is the target that most needs a generous grab area, so a
    // 2 pt marker keeps the full 9 pt and a 0.5 pt line the full 6 pt.
    let (points, point_index) = indexed(vec![point_feature(0.0, 0.0)]);
    let near = at(0.0, 0.0) + vec2(PICK_POINT_PT - 1.0, 0.0);
    assert_eq!(
        styled_hits(&points, &point_index, near, &circle_set(2.0)).len(),
        1,
    );
    let (lines, line_index) = indexed(vec![line_feature(&[[-4.0, 0.0], [4.0, 0.0]])]);
    let hairline = oxigis_core::LayerStyleSet::new(oxigis_core::LayerStyle::Line(
        oxigis_core::LineStyle::new(oxigis_core::Color::BLACK, 0.5),
    ));
    let beside = at(0.0, 0.0) + vec2(0.0, PICK_LINE_PT - 1.0);
    assert_eq!(
        styled_hits(&lines, &line_index, beside, &hairline).len(),
        1,
        "a hairline is still grabbed at the constant",
    );
}

#[test]
fn a_thick_line_is_grabbable_across_its_own_width() {
    let (features, index) = indexed(vec![line_feature(&[[-4.0, 0.0], [4.0, 0.0]])]);
    let beside = at(0.0, 0.0) + vec2(0.0, 15.0);
    assert!(
        hits(&features, &index, beside).is_empty(),
        "15 pt is outside the {PICK_LINE_PT} pt constant",
    );
    let casing = oxigis_core::LayerStyleSet::new(oxigis_core::LayerStyle::Line(
        oxigis_core::LineStyle::new(oxigis_core::Color::BLACK, 40.0),
    ));
    assert_eq!(
        styled_hits(&features, &index, beside, &casing).len(),
        1,
        "a 40 pt casing is 20 pt of ink either side of its centreline",
    );
}

#[test]
fn a_classified_layer_picks_each_feature_with_its_own_classs_size() {
    // THE parity statement: the map, the printed page and hit testing resolve
    // through one rule, so a class drawn large is picked large and its
    // neighbour drawn small is not.
    let big = at(0.0, 0.0);
    let small_lon = 2.0;
    let (features, index) = indexed(vec![
        zoned_point(0.0, 0.0, "big"),
        zoned_point(small_lon, 0.0, "small"),
    ]);
    let mut set = circle_set(4.0);
    set.set_renderer(oxigis_core::Renderer::categorized(
        "zone",
        [
            oxigis_core::CategoryClass::new(
                oxigis_core::AttrValue::text("big"),
                oxigis_core::LayerStyle::Circle(oxigis_core::CircleStyle::new(
                    24.0,
                    oxigis_core::Color::BLACK,
                )),
            ),
            oxigis_core::CategoryClass::new(
                oxigis_core::AttrValue::text("small"),
                oxigis_core::LayerStyle::Circle(oxigis_core::CircleStyle::new(
                    3.0,
                    oxigis_core::Color::BLACK,
                )),
            ),
        ],
        None,
    ));
    let near_big = big + vec2(18.0, 0.0);
    let picked = styled_hits(&features, &index, near_big, &set);
    assert_eq!(picked.len(), 1, "the big class answers from its own radius");
    assert_eq!(picked[0].feature, 0);

    let near_small = at(small_lon, 0.0) + vec2(18.0, 0.0);
    assert!(
        styled_hits(&features, &index, near_small, &set).is_empty(),
        "the small class must not inherit its neighbour's tolerance",
    );
    // The unstyled path is unchanged for both: this is a widening, never a
    // rewrite of what the constants mean.
    assert!(hits(&features, &index, near_big).is_empty());
    assert!(hits(&features, &index, near_small).is_empty());
}

#[test]
fn a_pick_through_the_edit_context_reads_the_projects_own_style() {
    // Reachability: the tolerance has to arrive through the SAME context the
    // app's hover and click paths use, not only through the low-level entry
    // point — otherwise the fix exists and nothing calls it.
    let mut project = Project::new("pick fixture");
    let layer = LayerId::new();
    project.styles.insert(layer, circle_set(24.0));
    let features = Arc::new(FeatureCollection::new(vec![point_feature(0.0, 0.0)]));
    let mut index = WorldBboxIndex::default();
    index.rebuild_if_stale(layer, &features);
    let ctx = EditCtx {
        project: &project,
        target: Some(layer),
        features: Some(&features),
        view: view(),
        rect: rect(),
        ppp: PPP,
    };
    let near = at(0.0, 0.0) + vec2(18.0, 0.0);
    assert_eq!(
        pick(&ctx, &index, None, near, false),
        Some(HitTarget::Feature { feature: 0 }),
    );
    // And a project with no style entry for the layer picks with the
    // constants, exactly as it always did.
    let bare = Project::new("pick fixture");
    let bare_ctx = EditCtx {
        project: &bare,
        target: Some(layer),
        ..ctx
    };
    assert_eq!(pick(&bare_ctx, &index, None, near, false), None);
}
