// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Pure-model tests for [`command::moved_vertices`] and its sibling
//! [`command::restored_vertices`] — the diffs an undo or a redo recovers a
//! marked set from, for a set MOVE and for a set DELETE respectively. No
//! egui, no app, no frames. Separate from `edit/tests.rs`, which is at the
//! 2000-line ceiling and must not grow.

use super::command;
use super::{VertexRef, hit};
use oxigeo::geojson::types::{
    Geometry, GeometryCollection, LineString, MultiLineString, Point, Polygon,
};

/// A position from a pair or triple.
fn position(coords: &[f64]) -> Vec<f64> {
    coords.to_vec()
}

/// A single-ring polygon; `ring` is given **closed**.
fn polygon(ring: &[[f64; 2]]) -> Geometry {
    Geometry::Polygon(
        Polygon::new(vec![ring.iter().map(|pair| pair.to_vec()).collect()])
            .expect("a closed ring is a polygon"),
    )
}

/// The square's closed ring, in fixture coordinates.
const RING: [[f64; 2]; 5] = [
    [-10.0, -10.0],
    [10.0, -10.0],
    [10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, -10.0],
];

#[test]
fn moved_vertices_names_exactly_the_translated_corners() {
    let before = polygon(&RING);
    // Corners 0 and 1 translated; `set_vertices` also moves the closing
    // duplicate, exactly as a committed set move records it.
    let after = polygon(&[
        [-7.0, -12.0],
        [13.0, -12.0],
        [10.0, 10.0],
        [-10.0, 10.0],
        [-7.0, -12.0],
    ]);
    let moved = command::moved_vertices(&before, &after).expect("same shape");
    assert_eq!(
        moved,
        vec![VertexRef::new(0), VertexRef::new(1)],
        "exactly the translated corners, ascending — the closing duplicate \
         is not addressable and is never reported"
    );
    // The diff is symmetric: the inverse recovers the same set.
    assert_eq!(
        command::moved_vertices(&after, &before).expect("same shape"),
        moved,
    );
}

#[test]
fn moved_vertices_refuses_a_shape_change() {
    let before = polygon(&RING);
    // An inserted vertex changes a path's position count.
    let inserted = polygon(&[
        [-10.0, -10.0],
        [0.0, -10.0],
        [10.0, -10.0],
        [10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, -10.0],
    ]);
    assert_eq!(command::moved_vertices(&before, &inserted), None);
    // A different geometry kind is a different path kind.
    let line = Geometry::LineString(
        LineString::new(vec![position(&[0.0, 0.0]), position(&[1.0, 1.0])]).expect("a line"),
    );
    assert_eq!(command::moved_vertices(&before, &line), None);
}

#[test]
fn moved_vertices_is_empty_for_an_identical_pair() {
    let geometry = polygon(&RING);
    assert_eq!(
        command::moved_vertices(&geometry, &geometry.clone()),
        Some(Vec::new()),
        "an attribute-only Apply moves nothing and restores no marks"
    );
}

#[test]
fn moved_vertices_ignores_non_finite_positions_and_altitude() {
    // A NaN present in both versions is not a mark: NaN != NaN would
    // otherwise report every untouched malformed position as moved.
    let with_nan = |lat| {
        Geometry::LineString(
            LineString::new(vec![
                position(&[0.0, f64::NAN]),
                position(&[1.0, lat]),
                position(&[2.0, 2.0]),
            ])
            .expect("a line"),
        )
    };
    let moved = command::moved_vertices(&with_nan(1.0), &with_nan(5.0)).expect("same shape");
    assert_eq!(
        moved,
        vec![VertexRef::new(1)],
        "only the finite move counts"
    );

    // Altitude is not compared: the vertex mutators never touch it.
    let flat = Geometry::Point(Point::new(position(&[3.0, 4.0])).expect("a point"));
    let tall = Geometry::Point(Point::new(position(&[3.0, 4.0, 333.0])).expect("a point"));
    assert_eq!(
        command::moved_vertices(&flat, &tall),
        Some(Vec::new()),
        "an altitude-only difference is not a move"
    );
}

#[test]
fn moved_vertices_addresses_collection_parts_like_paths() {
    let collection = |lon| {
        Geometry::GeometryCollection(
            GeometryCollection::new(vec![
                Geometry::Point(Point::new(position(&[0.0, 0.0])).expect("a point")),
                Geometry::Point(Point::new(position(&[lon, 7.0])).expect("a point")),
            ])
            .expect("a collection"),
        )
    };
    let moved = command::moved_vertices(&collection(5.0), &collection(6.0)).expect("same shape");
    assert_eq!(
        moved,
        vec![VertexRef::at(1, 0, 0)],
        "collection members number into `part` with the same running \
         counter `paths` uses — the address the marquee hands out"
    );
}

#[test]
fn moved_vertices_refuses_more_than_the_handle_budget() {
    let line = |offset: f64| {
        Geometry::LineString(
            LineString::new(
                (0..=hit::HANDLE_BUDGET)
                    .map(|index| position(&[index as f64 + offset, 0.0]))
                    .collect(),
            )
            .expect("a long line"),
        )
    };
    assert_eq!(
        command::moved_vertices(&line(0.0), &line(1.0)),
        None,
        "a marquee can only mark drawn handles, so more moves than the \
         handle budget is not a marked set"
    );
}

// --- Editing v1.4 E2: the arity-changing sibling ---

/// A LineString from `(lon, 0.0)` pairs.
fn line_of(lons: &[f64]) -> Geometry {
    Geometry::LineString(
        LineString::new(lons.iter().map(|lon| position(&[*lon, 0.0])).collect()).expect("a line"),
    )
}

#[test]
fn restored_vertices_names_exactly_what_a_set_delete_puts_back() {
    // The undo of "Delete vertices" on 1 and 3: the shorter side is what the
    // delete left, the longer side is what the undo restores.
    let deleted = line_of(&[0.0, 2.0, 4.0]);
    let whole = line_of(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        command::restored_vertices(&deleted, &whole),
        Some(vec![VertexRef::new(1), VertexRef::new(3)]),
        "the restored vertices are marked, at their NEW indices"
    );
    // The redo direction is a pure deletion: the vertices no longer exist, so
    // there is nothing to mark.
    assert_eq!(
        command::restored_vertices(&whole, &deleted),
        Some(Vec::new()),
        "a pure deletion restores nothing — marking the survivors would arm \
         Delete against vertices the user never touched"
    );
}

#[test]
fn moved_and_added_are_mutually_exclusive_on_the_same_pair() {
    let before = polygon(&RING);
    let moved = polygon(&[
        [-7.0, -12.0],
        [13.0, -12.0],
        [10.0, 10.0],
        [-10.0, 10.0],
        [-7.0, -12.0],
    ]);
    // `moved_vertices` answers every equal-shape pair, so the sibling is
    // consulted only on an arity change — and refuses this one outright.
    assert_eq!(
        command::moved_vertices(&before, &moved).map(|marks| marks.len()),
        Some(2)
    );
    assert_eq!(command::restored_vertices(&before, &moved), None);

    let grown = polygon(&[
        [-10.0, -10.0],
        [0.0, -10.0],
        [10.0, -10.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, -10.0],
    ]);
    assert_eq!(command::moved_vertices(&before, &grown), None);
    assert_eq!(
        command::restored_vertices(&before, &grown),
        Some(vec![VertexRef::new(1), VertexRef::new(3)]),
    );
}

#[test]
fn restored_vertices_refuses_a_move_mixed_with_an_insert() {
    // One path grew while another moved: there is no alignment that explains
    // both, so nothing may be guessed.
    let before = Geometry::MultiLineString(
        MultiLineString::new(vec![
            vec![position(&[0.0, 0.0]), position(&[2.0, 0.0])],
            vec![position(&[0.0, 5.0]), position(&[1.0, 5.0])],
        ])
        .expect("two lines"),
    );
    let after = Geometry::MultiLineString(
        MultiLineString::new(vec![
            vec![
                position(&[0.0, 0.0]),
                position(&[1.0, 0.0]),
                position(&[2.0, 0.0]),
            ],
            vec![position(&[0.0, 5.0]), position(&[1.0, 9.0])],
        ])
        .expect("two lines"),
    );
    assert_eq!(command::restored_vertices(&before, &after), None);
}

#[test]
fn restored_vertices_refuses_a_structural_change() {
    let line = line_of(&[0.0, 1.0, 2.0]);
    // A path-count change: one line became two.
    let split = Geometry::MultiLineString(
        MultiLineString::new(vec![
            vec![position(&[0.0, 0.0]), position(&[1.0, 0.0])],
            vec![position(&[1.0, 0.0]), position(&[2.0, 0.0])],
        ])
        .expect("two lines"),
    );
    assert_eq!(
        command::restored_vertices(&line, &split),
        None,
        "the forward contract for any future split/join tool"
    );
    // A ring gained: a polygon with a hole is not the same polygon.
    let square = polygon(&RING);
    let donut = Geometry::Polygon(
        Polygon::new(vec![
            RING.iter().map(|pair| pair.to_vec()).collect(),
            vec![
                position(&[-2.0, -2.0]),
                position(&[2.0, -2.0]),
                position(&[2.0, 2.0]),
                position(&[-2.0, -2.0]),
            ],
        ])
        .expect("a donut"),
    );
    assert_eq!(command::restored_vertices(&square, &donut), None);
}

#[test]
fn restored_vertices_unions_two_paths_ascending() {
    let before = Geometry::MultiLineString(
        MultiLineString::new(vec![
            vec![position(&[0.0, 0.0]), position(&[2.0, 0.0])],
            vec![position(&[0.0, 5.0]), position(&[2.0, 5.0])],
        ])
        .expect("two lines"),
    );
    let after = Geometry::MultiLineString(
        MultiLineString::new(vec![
            vec![
                position(&[0.0, 0.0]),
                position(&[1.0, 0.0]),
                position(&[2.0, 0.0]),
            ],
            vec![
                position(&[0.0, 5.0]),
                position(&[1.0, 5.0]),
                position(&[2.0, 5.0]),
            ],
        ])
        .expect("two lines"),
    );
    assert_eq!(
        command::restored_vertices(&before, &after),
        Some(vec![VertexRef::at(0, 0, 1), VertexRef::at(1, 0, 1)]),
        "`paths` order is the ascending order `vertex_set` requires"
    );
}

#[test]
fn restored_vertices_refuses_more_than_the_handle_budget() {
    const FAR: f64 = 1.0e9;
    let short = line_of(&[0.0, FAR]);
    let long = Geometry::LineString(
        LineString::new(
            (0..=hit::HANDLE_BUDGET + 1)
                .map(|index| position(&[index as f64, 0.0]))
                .chain(core::iter::once(position(&[FAR, 0.0])))
                .collect(),
        )
        .expect("a long line"),
    );
    assert_eq!(
        command::restored_vertices(&short, &long),
        None,
        "a marquee can only mark drawn handles, so more restored positions \
         than the handle budget is not a marked set"
    );
}

#[test]
fn restored_vertices_refuses_a_non_finite_coordinate() {
    // Stricter than `moved_vertices`' skip rule: a position that cannot be
    // compared would shift the alignment, and NaN must never invent a mark.
    let before = Geometry::LineString(
        LineString::new(vec![position(&[0.0, f64::NAN]), position(&[2.0, 0.0])]).expect("a line"),
    );
    let after = Geometry::LineString(
        LineString::new(vec![
            position(&[0.0, f64::NAN]),
            position(&[1.0, 0.0]),
            position(&[2.0, 0.0]),
        ])
        .expect("a line"),
    );
    assert_eq!(command::restored_vertices(&before, &after), None);
}

#[test]
fn restored_vertices_marks_the_same_points_among_duplicate_coordinates() {
    // Three identical corners: which INDEX the greedy walk calls "extra" is
    // ambiguous, but the multiset of marked POINTS is not — and that is what
    // makes a re-delete reproduce bit-identical geometry.
    let before = line_of(&[7.0, 7.0, 1.0]);
    let after = line_of(&[7.0, 7.0, 7.0, 7.0, 1.0]);
    let marks = command::restored_vertices(&before, &after).expect("a pure insertion");
    assert_eq!(marks.len(), 2, "exactly the delta is marked");
    let Geometry::LineString(line) = &after else {
        panic!("the fixture is a line");
    };
    let mut marked: Vec<f64> = marks
        .iter()
        .map(|reference| line.coordinates[reference.index][0])
        .collect();
    marked.sort_by(f64::total_cmp);
    assert_eq!(
        marked,
        vec![7.0, 7.0],
        "the same POINTS are marked whichever equal index the walk picked"
    );
}
