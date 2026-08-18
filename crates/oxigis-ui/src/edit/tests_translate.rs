// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Pure-model tests for the vertex-set MOVE: [`VertexDrag::target_of`],
//! [`drag_translation`], [`command::set_vertices`] and the set arm of
//! [`drag_transaction`] — no egui, no GPU, no frames.

use super::command::{self, EditError};
use super::{MOVE_VERTICES_LABEL, VertexDrag, VertexRef, drag_transaction, drag_translation};
use oxigeo::geojson::types::{Feature, FeatureCollection, Geometry, Polygon, Properties};
use oxigis_core::LayerId;
use oxigis_render::LonLat;

/// A single-ring polygon feature; `ring` is given **closed**.
fn polygon_feature(ring: &[[f64; 2]]) -> Feature {
    let polygon = Polygon::new(vec![ring.iter().map(|pair| pair.to_vec()).collect()])
        .expect("a closed ring is a polygon");
    Feature::new(Some(Geometry::Polygon(polygon)), Some(Properties::new()))
}

/// The square's closed ring, in fixture coordinates.
const RING: [[f64; 2]; 5] = [
    [-10.0, -10.0],
    [10.0, -10.0],
    [10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, -10.0],
];

/// A set drag of the square's two southern corners, grabbed at vertex 0 and
/// moved so the world delta is measurably non-zero.
fn set_drag(feature: &Feature) -> VertexDrag {
    let mut drag = VertexDrag::single(
        0,
        VertexRef::new(0),
        false,
        feature.geometry.clone().expect("the fixture has geometry"),
        LonLat::new(-10.0, -10.0),
    );
    drag.set = vec![VertexRef::new(0), VertexRef::new(1)];
    drag.current = LonLat::new(-7.0, -12.0);
    drag.moved = true;
    drag
}

#[test]
fn target_of_moves_the_grabbed_vertex_verbatim_and_translates_the_rest() {
    let feature = polygon_feature(&RING);
    let drag = set_drag(&feature);
    // Grabbed vertex: the snapped/current position, bit for bit.
    assert_eq!(
        drag.target_of(VertexRef::new(0), LonLat::new(-10.0, -10.0)),
        Some(drag.current),
    );
    // A marked companion: translated by the SAME world delta.
    let delta = drag.translation();
    let stored = LonLat::new(10.0, -10.0);
    let landed = drag
        .target_of(VertexRef::new(1), stored)
        .expect("vertex 1 is marked");
    let world_before = stored.to_world();
    let world_after = landed.to_world();
    assert!(
        (world_after.x - world_before.x - delta[0]).abs() < 1e-12
            && (world_after.y - world_before.y - delta[1]).abs() < 1e-12,
        "the set translates rigidly in world space",
    );
    // An unmarked vertex does not move.
    assert_eq!(
        drag.target_of(VertexRef::new(2), LonLat::new(10.0, 10.0)),
        None,
    );
}

#[test]
fn drag_translation_yields_exactly_the_marked_set() {
    let feature = polygon_feature(&RING);
    let drag = set_drag(&feature);
    let moves = drag_translation(&drag);
    assert_eq!(moves.len(), 2, "both marked corners, nothing else");
    assert_eq!(moves[0].0, VertexRef::new(0));
    assert_eq!(moves[0].1, drag.current, "the grabbed one lands verbatim");
    assert_eq!(moves[1].0, VertexRef::new(1));
}

#[test]
fn set_vertices_moves_the_ring_closing_duplicate_and_keeps_altitude() {
    let mut feature = polygon_feature(&RING);
    if let Some(Geometry::Polygon(polygon)) = feature.geometry.as_mut() {
        // Give vertex 1 an altitude that must survive the move.
        polygon.coordinates[0][1].push(333.0);
    }
    let moves = vec![
        (VertexRef::new(0), LonLat::new(-7.0, -12.0)),
        (VertexRef::new(1), LonLat::new(13.0, -12.0)),
    ];
    command::set_vertices(&mut feature, &moves).expect("in-range moves apply");
    let Some(Geometry::Polygon(polygon)) = feature.geometry.as_ref() else {
        panic!("still a polygon");
    };
    let ring = &polygon.coordinates[0];
    assert_eq!(ring[0][0], -7.0);
    assert_eq!(ring[0][1], -12.0);
    assert_eq!(ring[1][0], 13.0);
    assert_eq!(ring[1][2], 333.0, "altitude survives the write");
    assert_eq!(
        ring.first(),
        ring.last(),
        "moving index 0 moves the closing duplicate with it",
    );
}

#[test]
fn set_vertices_refuses_a_stale_mark_and_writes_nothing() {
    let mut feature = polygon_feature(&RING);
    let before = feature.clone();
    let stale = VertexRef::new(99);
    let moves = vec![
        (VertexRef::new(0), LonLat::new(-7.0, -12.0)),
        (stale, LonLat::new(0.0, 0.0)),
    ];
    assert_eq!(
        command::set_vertices(&mut feature, &moves),
        Err(EditError::BadVertex(stale)),
    );
    assert_eq!(feature, before, "refuse-first: nothing was written");

    // A non-finite target refuses the same way.
    let mut feature = polygon_feature(&RING);
    let bad = vec![(VertexRef::new(0), LonLat::new(f64::NAN, 0.0))];
    assert_eq!(
        command::set_vertices(&mut feature, &bad),
        Err(EditError::MalformedPosition(VertexRef::new(0))),
    );
    assert_eq!(feature, before);
}

#[test]
fn the_set_arm_commits_one_replace_labelled_move_vertices() {
    let feature = polygon_feature(&RING);
    let features = FeatureCollection::new(vec![feature.clone()]);
    let drag = set_drag(&feature);
    let transaction =
        drag_transaction(LayerId::new(), &features, &drag, None).expect("a live set move commits");
    assert_eq!(transaction.label, MOVE_VERTICES_LABEL);
    assert_eq!(transaction.ops.len(), 1, "ONE Replace for the whole set");
    assert!(transaction.coalesce.is_none(), "one drag = one undo step");

    // A lone mark moves like a single vertex and keeps the honest label.
    let mut lone = set_drag(&feature);
    lone.set = vec![VertexRef::new(0)];
    let transaction =
        drag_transaction(LayerId::new(), &features, &lone, None).expect("a lone mark commits");
    assert_eq!(transaction.label, "Move vertex");

    // A stale mark refuses the WHOLE gesture.
    let mut stale = set_drag(&feature);
    stale.set = vec![VertexRef::new(0), VertexRef::new(99)];
    assert_eq!(
        drag_transaction(LayerId::new(), &features, &stale, None),
        Err(EditError::BadVertex(VertexRef::new(99))),
    );
}
