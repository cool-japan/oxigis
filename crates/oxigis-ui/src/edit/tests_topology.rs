// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`super::topology`]: the three-pass self-intersection
//! algorithm, the degeneracies each pass exists for, and the `Geometry` adapter
//! that gives their answers part/ring provenance.
//!
//! No egui context and no app: every case here is coordinates in, issues out.

use super::topology::{
    CrossingKind, FeatureIssue, MAX_NOTICES, MAX_SEGMENTS_FOR_SELF_INTERSECTION, RingRole,
    Severity, TopologyIssue, describe, is_ccw, nearest_on_segment, orient, point_in_ring,
    segment_distance_sq, segments_intersect, severity, signed_area, validate_collection,
    validate_feature, validate_line, validate_polygon, validate_ring,
};
use oxigeo::geojson::types::{
    Feature, FeatureCollection, Geometry, GeometryCollection, LineString, MultiPolygon, Point,
    Polygon, Position, Properties,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A GeoJSON position from a planar point.
fn position(point: &[f64; 2]) -> Position {
    vec![point[0], point[1]]
}

/// A ring as GeoJSON positions.
fn ring(points: &[[f64; 2]]) -> Vec<Position> {
    points.iter().map(position).collect()
}

/// An axis-aligned closed square, counter-clockwise.
fn square_ccw(min: f64, max: f64) -> Vec<[f64; 2]> {
    vec![[min, min], [max, min], [max, max], [min, max], [min, min]]
}

/// An axis-aligned closed square, clockwise — the RFC 7946 winding for a hole.
fn square_cw(min: f64, max: f64) -> Vec<[f64; 2]> {
    vec![[min, min], [min, max], [max, max], [max, min], [min, min]]
}

/// The issues one open path produces.
fn line_issues(coords: &[[f64; 2]]) -> Vec<TopologyIssue> {
    let mut issues = Vec::new();
    validate_line(coords, &mut issues);
    issues
}

/// The issues one ring produces in `role`.
fn ring_issues(coords: &[[f64; 2]], role: RingRole) -> Vec<TopologyIssue> {
    let mut issues = Vec::new();
    validate_ring(coords, role, &mut issues);
    issues
}

/// The issues one polygon produces.
fn polygon_issues(rings: &[Vec<[f64; 2]>]) -> Vec<TopologyIssue> {
    let mut issues = Vec::new();
    validate_polygon(rings, &mut issues);
    issues
}

/// A `LineString` feature.
fn line_feature(coords: &[[f64; 2]]) -> Feature {
    let line = LineString {
        coordinates: coords.iter().map(position).collect(),
        bbox: None,
    };
    Feature::new(Some(Geometry::LineString(line)), Some(Properties::new()))
}

/// A `Polygon` feature; every ring is given **closed**.
fn polygon_feature(rings: &[Vec<[f64; 2]>]) -> Feature {
    let polygon = Polygon {
        coordinates: rings.iter().map(|points| ring(points)).collect(),
        bbox: None,
    };
    Feature::new(Some(Geometry::Polygon(polygon)), Some(Properties::new()))
}

/// A `MultiPolygon` feature; every ring of every part is given **closed**.
fn multipolygon_feature(parts: &[Vec<Vec<[f64; 2]>>]) -> Feature {
    let polygons = MultiPolygon {
        coordinates: parts
            .iter()
            .map(|rings| rings.iter().map(|points| ring(points)).collect())
            .collect(),
        bbox: None,
    };
    Feature::new(
        Some(Geometry::MultiPolygon(polygons)),
        Some(Properties::new()),
    )
}

/// Whether `issues` holds `wanted`.
fn holds(issues: &[TopologyIssue], wanted: &TopologyIssue) -> bool {
    issues.iter().any(|issue| issue == wanted)
}

/// How many self-intersections `issues` reports.
fn crossings(issues: &[TopologyIssue]) -> usize {
    issues
        .iter()
        .filter(|issue| matches!(issue, TopologyIssue::SelfIntersection { .. }))
        .count()
}

// ---------------------------------------------------------------------------
// T57–T71
// ---------------------------------------------------------------------------

/// T57.
#[test]
fn figure_eight_line_is_self_intersecting() {
    // Right-up, down, and back up across the first leg: segments 0 and 2 cross
    // at (1, 1), and they are the only non-adjacent pair there is.
    let issues = line_issues(&[[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]]);
    assert_eq!(
        issues,
        vec![TopologyIssue::SelfIntersection {
            first: 0,
            second: 2,
            kind: CrossingKind::Proper,
        }],
        "{issues:?}"
    );
}

/// T58.
#[test]
fn consecutive_segments_sharing_an_endpoint_are_not_reported() {
    // Two segments: the only pair is adjacent, so a shared endpoint — which is
    // what a path *is* — must stay silent.
    assert!(line_issues(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]).is_empty());
    // Three segments, one non-adjacent pair, and it does not meet.
    assert!(line_issues(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]).is_empty());
    // A collinear continuation shares an endpoint too, and is still not a
    // crossing — only a *doubling back* one is, and that is the spike pass.
    assert!(line_issues(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]).is_empty());
}

/// T59.
#[test]
fn closed_ring_wrap_pair_adjacency_is_not_reported() {
    // Segment 0 and segment m-1 share vertex 0 across the seam. Forgetting that
    // pair makes every legal ring in existence report a self-intersection.
    let issues = ring_issues(&square_ccw(0.0, 1.0), RingRole::Exterior);
    assert!(issues.is_empty(), "{issues:?}");
    // The same ring measured as a hole is only ever the winding advisory.
    let issues = ring_issues(&square_ccw(0.0, 1.0), RingRole::Hole);
    assert_eq!(
        issues,
        vec![TopologyIssue::WrongWinding {
            role: RingRole::Hole
        }]
    );
}

/// T60.
#[test]
fn collinear_overlapping_non_adjacent_segments_are_reported() {
    // The predicate first: the all-collinear branch is the one a naive
    // implementation omits, and without it these all answer "no intersection".
    assert_eq!(
        segments_intersect([0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [3.0, 0.0]),
        Some(CrossingKind::CollinearOverlap)
    );
    // A vertical pair: the overlap test has to pick the axis the segment
    // actually extends along, or it compares two identical zeros.
    assert_eq!(
        segments_intersect([0.0, 0.0], [0.0, 2.0], [0.0, 1.0], [0.0, 3.0]),
        Some(CrossingKind::CollinearOverlap)
    );
    // Collinear and meeting at exactly one point is a touch, not an overlap.
    assert_eq!(
        segments_intersect([0.0, 0.0], [2.0, 0.0], [2.0, 0.0], [4.0, 0.0]),
        Some(CrossingKind::Touch)
    );
    // Collinear and disjoint is nothing at all.
    assert_eq!(
        segments_intersect([0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]),
        None
    );
    // And through a whole path: segment 4 retraces part of segment 0.
    let issues = line_issues(&[
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 2.0],
        [3.0, 2.0],
        [3.0, 0.0],
        [1.0, 0.0],
    ]);
    assert!(
        holds(
            &issues,
            &TopologyIssue::SelfIntersection {
                first: 0,
                second: 4,
                kind: CrossingKind::CollinearOverlap,
            }
        ),
        "{issues:?}"
    );
}

/// T61.
#[test]
fn spike_doubling_back_over_the_previous_segment_is_reported_including_the_wrap_triples() {
    // Open path: out to x = 2, back to x = 1. Vertex 1 is the turn, and the
    // crossing test cannot see it — both segments there are adjacent, which is
    // exactly the case it has to skip.
    assert_eq!(
        line_issues(&[[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]]),
        vec![TopologyIssue::Spike { index: 1 }]
    );
    // Closed ring, wrap triple (m-1, 0, 1): the spike is on vertex 0, which no
    // linear triple loop can ever reach.
    let issues = ring_issues(
        &[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 0.0], [0.0, 0.0]],
        RingRole::Exterior,
    );
    assert!(
        holds(&issues, &TopologyIssue::Spike { index: 0 }),
        "{issues:?}"
    );
    // Closed ring, wrap triple (m-2, m-1, 0): the spike is on the last vertex.
    let issues = ring_issues(
        &[[0.0, 3.0], [3.0, 3.0], [0.0, 4.0], [0.0, 2.0], [0.0, 3.0]],
        RingRole::Exterior,
    );
    assert!(
        holds(&issues, &TopologyIssue::Spike { index: 3 }),
        "{issues:?}"
    );
}

/// T62.
#[test]
fn repeated_vertices_are_reported_and_excluded_from_the_pairwise_pass() {
    // The repeat is reported once, and the zero-length segment it would have
    // produced never reaches the crossing test — where it would have been
    // collinear with, and overlapping, both of its neighbours.
    let issues = line_issues(&[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
    assert_eq!(issues, vec![TopologyIssue::RepeatedVertex { index: 2 }]);
    // And compaction must not renumber what the later passes report: the
    // crossing here is between the compacted segments 0 and 2, whose original
    // start positions are 0 and 3.
    let issues = line_issues(&[[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]]);
    assert_eq!(
        issues,
        vec![
            TopologyIssue::RepeatedVertex { index: 1 },
            TopologyIssue::SelfIntersection {
                first: 0,
                second: 3,
                kind: CrossingKind::Proper,
            },
        ],
        "{issues:?}"
    );
}

/// T63.
#[test]
fn bowtie_ring_is_reported_as_a_proper_crossing() {
    let issues = ring_issues(
        &[[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
        RingRole::Exterior,
    );
    assert_eq!(
        issues,
        vec![TopologyIssue::SelfIntersection {
            first: 0,
            second: 2,
            kind: CrossingKind::Proper,
        }],
        "{issues:?}"
    );
}

/// T64.
#[test]
fn ccw_exterior_and_cw_hole_pass_the_winding_check() {
    let exterior = square_ccw(0.0, 4.0);
    let hole = square_cw(1.0, 3.0);
    assert!(is_ccw(&exterior));
    assert!(!is_ccw(&hole));
    assert!(signed_area(&exterior) > 0.0);
    assert!(signed_area(&hole) < 0.0);
    assert!(polygon_issues(&[exterior.clone(), hole.clone()]).is_empty());
    // Each flipped ring is an advisory, and only an advisory.
    assert_eq!(
        polygon_issues(&[square_cw(0.0, 4.0), hole]),
        vec![TopologyIssue::WrongWinding {
            role: RingRole::Exterior
        }]
    );
    assert_eq!(
        polygon_issues(&[exterior, square_ccw(1.0, 3.0)]),
        vec![TopologyIssue::WrongWinding {
            role: RingRole::Hole
        }]
    );
    // Advisory means advisory: it never rises above `Info`.
    assert_eq!(
        severity(&TopologyIssue::WrongWinding {
            role: RingRole::Exterior
        }),
        Severity::Info
    );
}

/// T65.
#[test]
fn hole_vertex_outside_the_exterior_is_reported() {
    let exterior = square_ccw(0.0, 4.0);
    // Clockwise, and reaching out past the exterior's northern edge: vertex 0 is
    // inside, vertex 1 is the first one that is not.
    let escaping = vec![[3.0, 3.0], [3.0, 6.0], [5.0, 6.0], [5.0, 3.0], [3.0, 3.0]];
    let issues = polygon_issues(&[exterior.clone(), escaping]);
    // Not the whole list: this hole's first edge also crosses the exterior's
    // own boundary on its way out, which the cross-ring pass reports as its
    // own, separate fact (finding 99's ring-crossing check) — a real second
    // truth about the same bad hole, not a duplicate of the vertex check.
    assert!(
        issues.contains(&TopologyIssue::HoleOutsideExterior { ring: 1, index: 1 }),
        "{issues:?}"
    );
    // A hole entirely inside is silent...
    assert!(polygon_issues(&[exterior.clone(), square_cw(1.0, 3.0)]).is_empty());
    // ...and so is one that shares two edges with its exterior. A vertex *on*
    // the boundary is not outside it, and holes touching their exterior are
    // ordinary data.
    assert!(polygon_issues(&[exterior, square_cw(0.0, 2.0)]).is_empty());
}

/// T66.
#[test]
fn unclosed_ring_and_too_few_vertices_are_reported() {
    // An unclosed ring is a warning and nothing more: every later pass still
    // runs, on the open sequence.
    let issues = ring_issues(
        &[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
        RingRole::Exterior,
    );
    assert_eq!(issues, vec![TopologyIssue::RingNotClosed]);
    assert_eq!(severity(&TopologyIssue::RingNotClosed), Severity::Warning);
    // A closed ring of three positions is two positions of geometry.
    assert!(holds(
        &ring_issues(&[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], RingRole::Exterior),
        &TopologyIssue::TooFewVertices { got: 2, need: 3 }
    ));
    // Below two positions there is not even a ring to open.
    assert_eq!(
        ring_issues(&[[0.0, 0.0]], RingRole::Exterior),
        vec![TopologyIssue::TooFewVertices { got: 1, need: 4 }]
    );
    assert_eq!(
        ring_issues(&[], RingRole::Exterior),
        vec![TopologyIssue::TooFewVertices { got: 0, need: 4 }]
    );
    // And a line needs two.
    assert_eq!(
        line_issues(&[[0.0, 0.0]]),
        vec![TopologyIssue::TooFewVertices { got: 1, need: 2 }]
    );
}

/// T67.
#[test]
fn non_finite_and_out_of_range_coordinates_are_reported() {
    let issues = line_issues(&[
        [0.0, 0.0],
        [f64::NAN, 1.0],
        [200.0, 0.0],
        [0.0, 95.0],
        [2.0, f64::INFINITY],
        [1.0, 1.0],
    ]);
    assert!(holds(
        &issues,
        &TopologyIssue::NonFiniteCoordinate { index: 1 }
    ));
    assert!(holds(&issues, &TopologyIssue::OutOfRange { index: 2 }));
    assert!(holds(&issues, &TopologyIssue::OutOfRange { index: 3 }));
    assert!(holds(
        &issues,
        &TopologyIssue::NonFiniteCoordinate { index: 4 }
    ));
    // The bounds are inclusive: the antimeridian and the poles are legal.
    assert!(line_issues(&[[-180.0, -90.0], [180.0, 90.0]]).is_empty());
    // A non-finite coordinate never becomes a crossing: it is dropped from the
    // compacted sequence, so no later pass has to reason about how `NaN` orders.
    assert_eq!(crossings(&issues), 0, "{issues:?}");
}

/// T68.
#[test]
fn above_the_segment_cap_returns_skipped_and_runs_no_pairwise_work() {
    // A long straight tail that pushes the path past the cap, and then one
    // crossing held well clear of it.
    let mut coords: Vec<[f64; 2]> = Vec::new();
    let mut lon = 0.0;
    while coords.len() < MAX_SEGMENTS_FOR_SELF_INTERSECTION - 2 {
        coords.push([lon, 0.0]);
        lon += 0.05;
    }
    coords.push([lon, 2.0]);
    coords.push([lon + 2.0, 4.0]);
    coords.push([lon + 2.0, 2.0]);
    coords.push([lon, 4.0]);
    let segments = coords.len() - 1;
    assert!(segments > MAX_SEGMENTS_FOR_SELF_INTERSECTION);
    let issues = line_issues(&coords);
    assert!(
        holds(&issues, &TopologyIssue::Skipped { segments }),
        "{issues:?}"
    );
    // The crossing is really there; the point is that the pass that would have
    // found it did not run, and said so rather than reporting a clean path.
    assert_eq!(crossings(&issues), 0, "{issues:?}");
    assert_eq!(
        severity(&TopologyIssue::Skipped { segments }),
        Severity::Info
    );
    // One segment under the cap and the same path is fully checked.
    coords.remove(0);
    assert_eq!(coords.len() - 1, MAX_SEGMENTS_FOR_SELF_INTERSECTION);
    let issues = line_issues(&coords);
    assert_eq!(crossings(&issues), 1, "{issues:?}");
}

/// T69.
#[test]
fn orient_epsilon_is_extent_relative_not_origin_relative() {
    // The same relative configuration — same shape, same size — at two
    // absolute positions must read identically: the epsilon depends on how
    // far `a`, `b` and `c` are from EACH OTHER, never on how far the triple
    // sits from the origin. This is the corrected invariant; the old
    // formula scaled by absolute coordinate magnitude instead, and read a
    // real turn at longitude 179 as collinear purely because 179 is a big
    // number (see the regression cases below).
    let offset = 1e-12;
    assert_eq!(orient([0.0, 0.0], [1.0, 0.0], [0.5, offset]), 1);
    assert_eq!(
        orient([179.0, 0.0], [180.0, 0.0], [179.5, offset]),
        1,
        "translating the identical triple to longitude 179 must not change \
         the verdict"
    );
    // A deviation that *is* resolvable at that scale is still reported, on both
    // sides — the epsilon widens, it does not blind the predicate.
    assert_eq!(orient([179.0, 0.0], [180.0, 0.0], [179.5, 1e-6]), 1);
    assert_eq!(orient([179.0, 0.0], [180.0, 0.0], [179.5, -1e-6]), -1);
    // Three points that are exactly collinear in exact arithmetic, and are not
    // once the interpolation has been rounded, read as collinear.
    let a = [179.999_999, 45.0];
    let b = [180.0, 45.000_001];
    let t = 1.0 / 3.0;
    let c = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    assert_eq!(orient(a, b, c), 0);
    // A non-finite input has no side to be on.
    assert_eq!(orient([0.0, 0.0], [1.0, 0.0], [f64::NAN, 1.0]), 0);
}

/// T72 (regression, finding 98): a real, metre-scale right angle far from the
/// origin must not read as collinear. Longitude 179, an extent of `1e-5`
/// degrees (about a metre): under the old origin-relative epsilon this
/// determinant (`~1e-10`) and the epsilon there (`~1.1e-10`) were the same
/// order, so a genuine corner vanished into "collinear".
#[test]
fn a_metre_scale_right_angle_far_from_the_origin_is_not_collinear() {
    let lon = 179.0;
    let a = [lon, 0.0];
    let b = [lon + 1e-5, 0.0];
    let c = [lon, 1e-5];
    assert_eq!(
        orient(a, b, c),
        1,
        "a genuine right angle, not a straight line"
    );
    // The same right angle at longitude 0 must read the same way — extent,
    // not position, is what decides it.
    assert_eq!(orient([0.0, 0.0], [1e-5, 0.0], [0.0, 1e-5]), 1);
}

/// T73 (regression, finding 98): two near-parallel segments a centimetre
/// apart, far from the origin, must not be read as lying on top of each
/// other. Under the old epsilon every `orient` call here rounded to `0`
/// (all four determinants were below the origin-inflated epsilon), so
/// [`segments_intersect`] took the all-collinear branch and reported an
/// overlap between segments that do not actually meet.
#[test]
fn near_parallel_segments_a_centimetre_apart_far_from_the_origin_do_not_overlap() {
    let lon = 179.0;
    let length = 1e-4;
    let offset = 1e-7; // roughly a centimetre, at this latitude
    let a0 = [lon, 0.0];
    let a1 = [lon + length, 0.0];
    let b0 = [lon, offset];
    let b1 = [lon + length, offset];
    assert_eq!(
        segments_intersect(a0, a1, b0, b1),
        None,
        "offset, same-length segments that never touch"
    );
}

/// T70.
#[test]
fn a_clean_1000_vertex_polygon_reports_nothing() {
    const COUNT: usize = 1_000;
    let mut coords: Vec<[f64; 2]> = (0..COUNT)
        .map(|step| {
            let angle = step as f64 * core::f64::consts::TAU / COUNT as f64;
            [10.0 * angle.cos(), 10.0 * angle.sin()]
        })
        .collect();
    coords.push(coords[0]);
    let issues = ring_issues(&coords, RingRole::Exterior);
    assert!(issues.is_empty(), "{issues:?}");
}

/// T71.
#[test]
fn multipolygon_issues_carry_their_part_and_ring_indices() {
    let clean = vec![square_ccw(0.0, 4.0), square_cw(1.0, 3.0)];
    let bowtie = vec![vec![
        [10.0, 0.0],
        [12.0, 2.0],
        [12.0, 0.0],
        [10.0, 2.0],
        [10.0, 0.0],
    ]];
    let escaping = vec![
        square_ccw(20.0, 24.0),
        vec![
            [23.0, 23.0],
            [23.0, 26.0],
            [25.0, 26.0],
            [25.0, 23.0],
            [23.0, 23.0],
        ],
    ];
    let feature = multipolygon_feature(&[clean, bowtie, escaping]);
    let issues = validate_feature(7, &feature);
    // Three, not two: the escaping hole's first edge does not just have an
    // outside vertex, it genuinely crosses back out through the exterior's
    // own boundary on the way there — a second, distinct fact the cross-ring
    // pass (finding 99) reports that `hole_containment` structurally cannot.
    assert_eq!(issues.len(), 3, "{issues:?}");

    let crossing = &issues[0];
    assert_eq!(crossing.feature, 7);
    assert_eq!(crossing.part, 1);
    assert_eq!(crossing.ring, 0);
    assert_eq!(crossing.role, RingRole::Exterior);
    assert!(matches!(
        crossing.issue,
        TopologyIssue::SelfIntersection {
            kind: CrossingKind::Proper,
            ..
        }
    ));
    // The camera target is the first crossing segment's start position.
    let at = crossing.at.expect("a crossing names a vertex");
    assert!(
        (at.lon - 10.0).abs() < 1e-9 && at.lat.abs() < 1e-9,
        "{at:?}"
    );

    let hole = &issues[1];
    assert_eq!(hole.part, 2);
    assert_eq!(hole.ring, 1);
    assert_eq!(hole.role, RingRole::Hole);
    assert_eq!(
        hole.issue,
        TopologyIssue::HoleOutsideExterior { ring: 1, index: 1 }
    );
    let at = hole.at.expect("a stray hole vertex names itself");
    assert!(
        (at.lon - 23.0).abs() < 1e-9 && (at.lat - 26.0).abs() < 1e-9,
        "{at:?}"
    );

    let ring_cross = &issues[2];
    assert_eq!(ring_cross.feature, 7);
    assert_eq!(ring_cross.part, 2);
    assert_eq!(ring_cross.role, RingRole::Exterior);
    assert!(
        matches!(
            ring_cross.issue,
            TopologyIssue::RingsIntersect {
                first_ring: 0,
                second_ring: 1,
                ..
            }
        ),
        "{ring_cross:?}"
    );
    assert!(
        ring_cross.at.is_some(),
        "a ring crossing names the vertex it was found at"
    );

    // Nothing is ever attributed to the clean part.
    assert!(issues.iter().all(|issue| issue.part != 0), "{issues:?}");
}

/// T77 (regression, finding 89): [`validate_feature`]'s pairwise budget is
/// shared across every part of a feature, not reset per ring — a
/// per-path-only cap lets an N-ring MultiPolygon cost N times the intended
/// per-commit ceiling.
#[test]
fn validate_feature_shares_one_pairwise_budget_across_every_part() {
    // A clean circle, well under the per-path cap on its own, repeated as
    // two separate MultiPolygon parts. Each part alone would pass in full;
    // together their segment counts sum past `MAX_SEGMENTS_FOR_SELF_INTERSECTION`.
    const RING_SEGMENTS: usize = 1_500;
    const { assert!(RING_SEGMENTS < MAX_SEGMENTS_FOR_SELF_INTERSECTION) };
    const { assert!(RING_SEGMENTS * 2 > MAX_SEGMENTS_FOR_SELF_INTERSECTION) };
    let circle = |offset: f64| -> Vec<[f64; 2]> {
        let mut coords: Vec<[f64; 2]> = (0..RING_SEGMENTS)
            .map(|step| {
                let angle = step as f64 * core::f64::consts::TAU / RING_SEGMENTS as f64;
                [offset + 10.0 * angle.cos(), 10.0 * angle.sin()]
            })
            .collect();
        coords.push(coords[0]);
        coords
    };
    let feature = multipolygon_feature(&[vec![circle(0.0)], vec![circle(100.0)]]);
    let issues = validate_feature(0, &feature);
    // The first part had the whole budget to itself and is a clean circle:
    // nothing attributed to it.
    assert!(
        issues.iter().all(|issue| issue.part != 0),
        "the first part had budget and is clean: {issues:?}"
    );
    // The second part has nothing left to spend and must say so, not
    // silently report a clean path it never actually checked.
    assert!(
        issues.iter().any(|issue| issue.part == 1
            && matches!(
                issue.issue,
                TopologyIssue::Skipped { segments } if segments == RING_SEGMENTS
            )),
        "{issues:?}"
    );
}

/// T78 (regression, finding 99): a hole whose every vertex lies inside a
/// concave exterior, but whose edge crosses the exterior's own boundary
/// (through a notch) and back, used to read as completely clean — the
/// pairwise pass is strictly intra-ring, so no ring was ever tested against
/// another.
#[test]
fn a_hole_straddling_a_concave_exterior_boundary_is_reported() {
    // A 10x10 square with a notch cut from the top edge between x=4 and
    // x=6, down to y=7 — concave on purpose: a straight edge between two
    // points on either side of the notch leaves the shell and re-enters it,
    // which no convex exterior can produce.
    let exterior = vec![
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [6.0, 10.0],
        [6.0, 7.0],
        [4.0, 7.0],
        [4.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ];
    // A thin hole spanning the notch at y in [8, 8.5]; every one of its four
    // corners sits left or right of the notch, strictly inside the shell.
    let hole = vec![[3.0, 8.0], [7.0, 8.0], [7.0, 8.5], [3.0, 8.5], [3.0, 8.0]];
    let issues = polygon_issues(&[exterior, hole]);
    assert!(
        !issues
            .iter()
            .any(|issue| matches!(issue, TopologyIssue::HoleOutsideExterior { .. })),
        "every hole vertex really is inside the shell: {issues:?}"
    );
    assert!(
        issues.iter().any(|issue| matches!(
            issue,
            TopologyIssue::RingsIntersect {
                first_ring: 0,
                second_ring: 1,
                ..
            }
        )),
        "the hole's top and bottom edges both cross the notch and back, \
         which only the cross-ring pass can see: {issues:?}"
    );
}

/// T79 (regression, finding 99): two holes of the same polygon overlapping —
/// an interior no renderer can fill unambiguously — used to be invisible,
/// since the pairwise self-intersection pass never compares one ring against
/// another.
#[test]
fn two_overlapping_holes_are_reported() {
    let exterior = square_ccw(0.0, 20.0);
    let hole_a = square_cw(2.0, 8.0);
    let hole_b = square_cw(5.0, 11.0);
    let issues = polygon_issues(&[exterior, hole_a, hole_b]);
    assert!(
        issues.iter().any(|issue| matches!(
            issue,
            TopologyIssue::RingsIntersect {
                first_ring: 1,
                second_ring: 2,
                ..
            }
        )),
        "{issues:?}"
    );
}

/// T80 (regression, finding 99): one hole entirely inside another hole of the
/// same polygon — not really a hole at all — used to be invisible, since
/// [`hole_containment`](super::topology) only ever compares a hole against
/// the exterior.
#[test]
fn a_hole_nested_inside_another_hole_is_reported() {
    let exterior = square_ccw(0.0, 20.0);
    let outer_hole = square_cw(2.0, 16.0);
    let inner_hole = square_cw(6.0, 10.0);
    let issues = polygon_issues(&[exterior, outer_hole, inner_hole]);
    assert!(
        issues.iter().any(|issue| matches!(
            issue,
            TopologyIssue::HolesNested {
                outer_ring: 1,
                inner_ring: 2,
            }
        )),
        "{issues:?}"
    );
    // Two disjoint, non-nested holes stay silent.
    let separate = polygon_issues(&[
        square_ccw(0.0, 20.0),
        square_cw(1.0, 3.0),
        square_cw(10.0, 12.0),
    ]);
    assert!(
        !separate
            .iter()
            .any(|issue| matches!(issue, TopologyIssue::HolesNested { .. })),
        "{separate:?}"
    );
}

/// T81 (finding 99): a hole sharing two full edges with its exterior — an
/// established, tested-elsewhere case of ordinary data (T65) — must not
/// newly trip the cross-ring pass. Two rings running alongside each other
/// for a stretch is a [`CrossingKind::CollinearOverlap`], not a
/// [`CrossingKind::Proper`] crossing, and only the latter is a defect
/// between different rings.
#[test]
fn edge_sharing_rings_are_not_ring_crossings() {
    let exterior = square_ccw(0.0, 4.0);
    let issues = polygon_issues(&[exterior, square_cw(0.0, 2.0)]);
    assert!(issues.is_empty(), "{issues:?}");
}

// ---------------------------------------------------------------------------
// The rest of the pure half
// ---------------------------------------------------------------------------

#[test]
fn a_proper_crossing_a_touch_and_a_miss_are_told_apart() {
    assert_eq!(
        segments_intersect([0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]),
        Some(CrossingKind::Proper)
    );
    // A T-junction: an endpoint of one lands in the middle of the other.
    assert_eq!(
        segments_intersect([0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 2.0]),
        Some(CrossingKind::Touch)
    );
    // Parallel and apart.
    assert_eq!(
        segments_intersect([0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]),
        None
    );
    // Crossing lines whose *segments* stop short of each other.
    assert_eq!(
        segments_intersect([0.0, 0.0], [1.0, 0.0], [2.0, -1.0], [2.0, 1.0]),
        None
    );
}

#[test]
fn point_in_ring_is_vertex_robust_and_hole_aware() {
    let square = square_ccw(0.0, 4.0);
    assert!(point_in_ring([2.0, 2.0], &square));
    assert!(!point_in_ring([5.0, 2.0], &square));
    assert!(!point_in_ring([-1.0, -1.0], &square));
    // A ray cast at exactly a vertex's latitude must not be counted twice: this
    // diamond has vertices at y = 0 on both sides of the test point.
    let diamond = vec![[0.0, 0.0], [2.0, -2.0], [4.0, 0.0], [2.0, 2.0], [0.0, 0.0]];
    assert!(point_in_ring([2.0, 0.0], &diamond));
    assert!(!point_in_ring([5.0, 0.0], &diamond));
    // An empty ring contains nothing rather than panicking.
    assert!(!point_in_ring([0.0, 0.0], &[]));
    assert_eq!(signed_area(&[]), 0.0);
    assert!(!is_ccw(&[[0.0, 0.0], [1.0, 0.0]]));
}

#[test]
fn nearest_on_segment_clamps_to_both_endpoints() {
    let (nearest, t) = nearest_on_segment([1.0, 1.0], [0.0, 0.0], [2.0, 0.0]);
    assert_eq!(nearest, [1.0, 0.0]);
    assert!((t - 0.5).abs() < 1e-12);
    let (nearest, t) = nearest_on_segment([-5.0, 1.0], [0.0, 0.0], [2.0, 0.0]);
    assert_eq!(nearest, [0.0, 0.0]);
    assert_eq!(t, 0.0);
    let (nearest, t) = nearest_on_segment([9.0, 1.0], [0.0, 0.0], [2.0, 0.0]);
    assert_eq!(nearest, [2.0, 0.0]);
    assert_eq!(t, 1.0);
    // A degenerate segment answers with its own position instead of dividing by
    // zero.
    let (nearest, t) = nearest_on_segment([3.0, 3.0], [1.0, 1.0], [1.0, 1.0]);
    assert_eq!(nearest, [1.0, 1.0]);
    assert_eq!(t, 0.0);
    assert!((segment_distance_sq([1.0, 1.0], [0.0, 0.0], [2.0, 0.0]) - 1.0).abs() < 1e-12);
    assert_eq!(segment_distance_sq([0.0, 0.0], [0.0, 0.0], [2.0, 0.0]), 0.0);
}

#[test]
fn a_polygon_with_no_rings_is_empty_geometry() {
    assert_eq!(polygon_issues(&[]), vec![TopologyIssue::EmptyGeometry]);
    assert_eq!(severity(&TopologyIssue::EmptyGeometry), Severity::Info);
}

// ---------------------------------------------------------------------------
// The adapter half
// ---------------------------------------------------------------------------

#[test]
fn a_feature_without_geometry_reports_empty_geometry_and_no_camera_target() {
    let issues = validate_feature(3, &Feature::new(None, Some(Properties::new())));
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].feature, 3);
    assert_eq!(issues[0].issue, TopologyIssue::EmptyGeometry);
    assert!(issues[0].at.is_none());
    assert_eq!(describe(&issues[0]), "has no geometry");
}

#[test]
fn a_point_is_checked_for_range_and_a_short_position_reads_as_non_finite() {
    let point = Point {
        coordinates: vec![200.0, 0.0],
        bbox: None,
    };
    let feature = Feature::new(Some(Geometry::Point(point)), Some(Properties::new()));
    let issues = validate_feature(0, &feature);
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].issue, TopologyIssue::OutOfRange { index: 0 });
    // A position with one element is malformed rather than merely out of range,
    // and reads as the non-finite coordinate it becomes.
    let point = Point {
        coordinates: vec![1.0],
        bbox: None,
    };
    let feature = Feature::new(Some(Geometry::Point(point)), Some(Properties::new()));
    let issues = validate_feature(0, &feature);
    assert_eq!(
        issues[0].issue,
        TopologyIssue::NonFiniteCoordinate { index: 0 }
    );
}

#[test]
fn a_geometry_collection_carries_its_member_index_as_the_part() {
    let collection = GeometryCollection {
        geometries: vec![
            Geometry::LineString(LineString {
                coordinates: ring(&[[0.0, 0.0], [1.0, 0.0]]),
                bbox: None,
            }),
            Geometry::LineString(LineString {
                coordinates: ring(&[[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
                bbox: None,
            }),
        ],
        bbox: None,
    };
    let feature = Feature::new(
        Some(Geometry::GeometryCollection(collection)),
        Some(Properties::new()),
    );
    let issues = validate_feature(1, &feature);
    assert_eq!(issues.len(), 1, "{issues:?}");
    assert_eq!(issues[0].part, 1);
    assert_eq!(issues[0].role, RingRole::Open);
}

#[test]
fn validate_collection_spends_one_budget_across_every_feature() {
    // Two identical figure eights; a budget that only pays for the first.
    let figure_eight = line_feature(&[[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]]);
    let features = FeatureCollection::new(vec![figure_eight.clone(), figure_eight]);
    let generous = validate_collection(&features, 64);
    assert_eq!(generous.issues.len(), 2, "{generous:?}");
    assert!(
        generous
            .issues
            .iter()
            .all(|issue| matches!(issue.issue, TopologyIssue::SelfIntersection { .. }))
    );
    assert_eq!(generous.issues[0].feature, 0);
    assert_eq!(generous.issues[1].feature, 1);
    assert!(!generous.truncated);

    let tight = validate_collection(&features, 3);
    assert_eq!(tight.issues.len(), 2, "{tight:?}");
    assert!(matches!(
        tight.issues[0].issue,
        TopologyIssue::SelfIntersection { .. }
    ));
    // The second feature had nothing left to spend, so it says so instead of
    // reporting a clean path.
    assert_eq!(
        tight.issues[1].issue,
        TopologyIssue::Skipped { segments: 3 }
    );
}

#[test]
fn validate_collection_stops_at_the_notice_cap_and_reports_the_truncation() {
    let broken = polygon_feature(&[vec![
        [0.0, 0.0],
        [2.0, 2.0],
        [2.0, 0.0],
        [0.0, 2.0],
        [0.0, 0.0],
    ]]);
    let features = FeatureCollection::new(vec![broken; MAX_NOTICES + 50]);
    let validation = validate_collection(&features, usize::MAX);
    assert_eq!(validation.issues.len(), MAX_NOTICES);
    assert!(
        validation.truncated,
        "50 features' issues were dropped past the cap"
    );
}

#[test]
fn a_collection_with_exactly_the_cap_in_issues_is_not_reported_truncated() {
    // Exactly MAX_NOTICES one-issue features: the list is full, but nothing
    // was dropped — the status line must not claim otherwise.
    let broken = polygon_feature(&[vec![
        [0.0, 0.0],
        [2.0, 2.0],
        [2.0, 0.0],
        [0.0, 2.0],
        [0.0, 0.0],
    ]]);
    let features = FeatureCollection::new(vec![broken; MAX_NOTICES]);
    let validation = validate_collection(&features, usize::MAX);
    assert_eq!(validation.issues.len(), MAX_NOTICES);
    assert!(
        !validation.truncated,
        "every issue is shown, so the run must not claim a cut: {validation:?}"
    );
}

#[test]
fn a_closed_line_string_is_not_its_own_self_intersection() {
    // A loop trail: a legal closed LineString whose first and last positions
    // coincide. Its end segments share that position, which is closure, not a
    // topology defect — JTS `isSimple` makes the same exception.
    let closed_loop = line_feature(&[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]);
    let issues = validate_feature(0, &closed_loop);
    assert!(issues.is_empty(), "{issues:?}");
}

#[test]
fn a_closed_line_string_that_really_crosses_itself_still_reports() {
    // Closed *and* self-crossing: the closure exception must only forgive the
    // shared endpoint, never a genuine crossing elsewhere.
    let pinched = line_feature(&[[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]);
    let issues = validate_feature(0, &pinched);
    assert!(
        issues
            .iter()
            .any(|issue| matches!(issue.issue, TopologyIssue::SelfIntersection { .. })),
        "{issues:?}"
    );
}

#[test]
fn a_polygon_issue_names_its_ring_and_flies_to_the_offending_vertex() {
    let feature = polygon_feature(&[
        square_ccw(0.0, 4.0),
        vec![
            [1.0, 1.0],
            [1.0, 3.0],
            [1.0, 3.0],
            [3.0, 3.0],
            [3.0, 1.0],
            [1.0, 1.0],
        ],
    ]);
    let issues = validate_feature(12, &feature);
    assert_eq!(issues.len(), 1, "{issues:?}");
    assert_eq!(issues[0].ring, 1);
    assert_eq!(issues[0].role, RingRole::Hole);
    assert_eq!(issues[0].issue, TopologyIssue::RepeatedVertex { index: 2 });
    let at = issues[0].at.expect("a repeated vertex names itself");
    assert_eq!((at.lon, at.lat), (1.0, 3.0));
    assert_eq!(describe(&issues[0]), "hole 1 repeats vertex 2");
}

#[test]
fn describe_names_the_location_then_the_problem() {
    let issue = |part, ring, role, issue| FeatureIssue {
        feature: 0,
        part,
        ring,
        role,
        issue,
        at: None,
    };
    assert_eq!(
        describe(&issue(
            0,
            0,
            RingRole::Exterior,
            TopologyIssue::SelfIntersection {
                first: 3,
                second: 9,
                kind: CrossingKind::Proper,
            }
        )),
        "ring 0 self-intersects (segments 3, 9)"
    );
    assert_eq!(
        describe(&issue(
            2,
            1,
            RingRole::Hole,
            TopologyIssue::HoleOutsideExterior { ring: 1, index: 4 }
        )),
        "part 2 hole 1 vertex 4 lies outside the exterior ring"
    );
    assert_eq!(
        describe(&issue(
            3,
            0,
            RingRole::Open,
            TopologyIssue::Spike { index: 8 }
        )),
        "part 3 doubles back at vertex 8"
    );
    assert_eq!(
        describe(&issue(
            0,
            0,
            RingRole::Open,
            TopologyIssue::TooFewVertices { got: 1, need: 2 }
        )),
        "has 1 vertices where 2 are needed"
    );
    assert_eq!(
        describe(&issue(
            0,
            0,
            RingRole::Open,
            TopologyIssue::SelfIntersection {
                first: 1,
                second: 4,
                kind: CrossingKind::CollinearOverlap,
            }
        )),
        "overlaps itself (segments 1, 4)"
    );
    assert_eq!(
        describe(&issue(
            0,
            0,
            RingRole::Open,
            TopologyIssue::SelfIntersection {
                first: 1,
                second: 4,
                kind: CrossingKind::Touch,
            }
        )),
        "touches itself (segments 1, 4)"
    );
    assert!(
        describe(&issue(
            0,
            2,
            RingRole::Hole,
            TopologyIssue::WrongWinding {
                role: RingRole::Hole
            }
        ))
        .starts_with("hole 2 winds counter-clockwise")
    );
    assert!(
        describe(&issue(
            0,
            0,
            RingRole::Exterior,
            TopologyIssue::WrongWinding {
                role: RingRole::Exterior
            }
        ))
        .starts_with("ring 0 winds clockwise")
    );
    assert_eq!(
        describe(&issue(
            0,
            0,
            RingRole::Open,
            TopologyIssue::NonFiniteCoordinate { index: 2 }
        )),
        "vertex 2 is not a finite coordinate"
    );
    assert!(
        describe(&issue(
            0,
            0,
            RingRole::Open,
            TopologyIssue::OutOfRange { index: 2 }
        ))
        .starts_with("vertex 2 is outside")
    );
    assert!(
        describe(&issue(
            0,
            0,
            RingRole::Exterior,
            TopologyIssue::RingNotClosed
        ))
        .starts_with("ring 0 is not closed")
    );
    assert!(
        describe(&issue(
            0,
            0,
            RingRole::Open,
            TopologyIssue::Skipped { segments: 12_000 }
        ))
        .contains("12000 segments")
    );
}
