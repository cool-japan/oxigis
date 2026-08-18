// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the pure half of the edit system: the command model
//! ([`super::command`]) and the undo stack ([`super::stack`]).
//!
//! Everything here runs with no app, no egui context and no GPU.

use oxigeo::geojson::types::{
    Feature, FeatureCollection, Geometry, GeometryCollection, LineString, MultiLineString,
    MultiPoint, MultiPolygon, Point, Polygon, Position, Properties,
};
use oxigis_core::LayerId;
use oxigis_render::LonLat;

use super::command::{
    EditError, EditTransaction, FeatureOp, PathKind, apply_ops, insert_vertex, min_positions,
    paths, paths_mut, remove_vertex, set_properties, set_vertex,
};
use super::{EditSelection, VertexRef};

/// A position, as GeoJSON stores one.
fn position(values: &[f64]) -> Position {
    values.to_vec()
}

/// A `Point` feature at `lon`/`lat`.
fn point_feature(lon: f64, lat: f64) -> Feature {
    let point = Point::new(position(&[lon, lat])).expect("a two-element position is a Point");
    Feature::new(Some(Geometry::Point(point)), Some(Properties::new()))
}

/// A `LineString` feature through `coords`.
fn line_feature(coords: &[[f64; 2]]) -> Feature {
    let line = LineString::new(coords.iter().map(|pair| position(pair)).collect())
        .expect("at least two positions");
    Feature::new(Some(Geometry::LineString(line)), Some(Properties::new()))
}

/// A single-ring `Polygon` feature; `ring` is given **closed**.
fn polygon_feature(ring: &[[f64; 2]]) -> Feature {
    let polygon =
        Polygon::from_exterior(ring.iter().map(|pair| position(pair)).collect()).expect("one ring");
    Feature::new(Some(Geometry::Polygon(polygon)), Some(Properties::new()))
}

/// The exterior ring of a feature built by [`polygon_feature`].
fn exterior_of(feature: &Feature) -> &Vec<Position> {
    match feature.geometry.as_ref() {
        Some(Geometry::Polygon(polygon)) => polygon
            .coordinates
            .first()
            .expect("the polygon has an exterior"),
        other => panic!("expected a polygon, got {other:?}"),
    }
}

/// The positions of a feature built by [`line_feature`].
fn line_of(feature: &Feature) -> &Vec<Position> {
    match feature.geometry.as_ref() {
        Some(Geometry::LineString(line)) => &line.coordinates,
        other => panic!("expected a line, got {other:?}"),
    }
}

/// A collection of `count` points, one degree apart along the equator.
fn collection(count: usize) -> FeatureCollection {
    FeatureCollection::new(
        (0..count)
            .map(|index| point_feature(index as f64, 0.0))
            .collect(),
    )
}

mod command_tests {
    use super::*;

    #[test]
    fn move_vertex_preserves_third_position_element() {
        let point = Point::new(position(&[10.0, 20.0, 333.0])).expect("three elements");
        let mut feature = Feature::new(Some(Geometry::Point(point)), None);
        set_vertex(&mut feature, VertexRef::new(0), LonLat::new(11.0, 21.0)).expect("in range");
        match feature.geometry.as_ref() {
            Some(Geometry::Point(point)) => {
                assert_eq!(point.coordinates, position(&[11.0, 21.0, 333.0]));
            }
            other => panic!("expected a point, got {other:?}"),
        }
    }

    #[test]
    fn move_vertex_on_ring_index_zero_moves_the_closing_vertex() {
        let mut feature = polygon_feature(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]);
        set_vertex(&mut feature, VertexRef::new(0), LonLat::new(-5.0, -6.0)).expect("in range");
        let ring = exterior_of(&feature);
        assert_eq!(ring.len(), 4, "the ring keeps its closing position");
        assert_eq!(ring[0], position(&[-5.0, -6.0]));
        assert_eq!(
            ring[0], ring[3],
            "moving handle 0 must move the duplicate closing position with it"
        );
    }

    #[test]
    fn geometry_op_clears_feature_geometry_and_collection_bbox() {
        let mut feature = line_feature(&[[0.0, 0.0], [1.0, 1.0]]);
        feature.bbox = Some(vec![0.0, 0.0, 1.0, 1.0]);
        if let Some(Geometry::LineString(line)) = feature.geometry.as_mut() {
            line.bbox = Some(vec![0.0, 0.0, 1.0, 1.0]);
        }
        let before = feature.clone();
        set_vertex(&mut feature, VertexRef::new(1), LonLat::new(9.0, 9.0)).expect("in range");
        assert!(feature.bbox.is_none(), "the feature bbox must be scrubbed");
        match feature.geometry.as_ref() {
            Some(Geometry::LineString(line)) => assert!(line.bbox.is_none()),
            other => panic!("expected a line, got {other:?}"),
        }

        let mut features = FeatureCollection::new(vec![before.clone()]);
        features.bbox = Some(vec![0.0, 0.0, 1.0, 1.0]);
        let next = apply_ops(
            &features,
            &[FeatureOp::Replace {
                index: 0,
                before: Box::new(before),
                after: Box::new(feature),
            }],
        )
        .expect("in range");
        assert!(
            next.bbox.is_none(),
            "a stale collection bbox would be serialized into the project file"
        );
    }

    #[test]
    fn insert_vertex_after_the_last_ring_vertex_keeps_the_ring_closed() {
        let mut feature = polygon_feature(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]);
        // The open ring holds three positions, so index 3 appends.
        insert_vertex(&mut feature, VertexRef::new(3), LonLat::new(0.0, 1.0)).expect("appends");
        let ring = exterior_of(&feature);
        assert_eq!(ring.len(), 5);
        assert_eq!(ring[3], position(&[0.0, 1.0]));
        assert_eq!(ring[0], ring[4], "the ring must still be closed");
    }

    #[test]
    fn delete_vertex_index_zero_recloses_the_ring() {
        let mut feature =
            polygon_feature(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]);
        remove_vertex(&mut feature, VertexRef::new(0)).expect("four open positions is enough");
        let ring = exterior_of(&feature);
        assert_eq!(ring.len(), 4);
        assert_eq!(ring[0], position(&[1.0, 0.0]));
        assert_eq!(ring[0], ring[3], "the new first position closes the ring");
    }

    #[test]
    fn delete_vertex_refused_below_minimum_for_line_and_ring() {
        let mut line = line_feature(&[[0.0, 0.0], [1.0, 1.0]]);
        let untouched = line.clone();
        assert_eq!(
            remove_vertex(&mut line, VertexRef::new(0)),
            Err(EditError::TooFewVertices { have: 1, need: 2 })
        );
        assert_eq!(line, untouched, "a refusal must change nothing");

        let mut polygon = polygon_feature(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]);
        let untouched = polygon.clone();
        assert_eq!(
            remove_vertex(&mut polygon, VertexRef::new(1)),
            Err(EditError::TooFewVertices { have: 2, need: 3 })
        );
        assert_eq!(polygon, untouched);
        assert_eq!(min_positions(PathKind::Points), 1);
        assert_eq!(min_positions(PathKind::Line), 2);
        assert_eq!(min_positions(PathKind::Ring), 3);
    }

    #[test]
    fn remove_op_renumbers_following_indices() {
        let features = collection(4);
        let removed = features.features[1].clone();
        let next = apply_ops(
            &features,
            &[FeatureOp::Remove {
                index: 1,
                feature: Box::new(removed),
            }],
        )
        .expect("in range");
        assert_eq!(next.features.len(), 3);
        // What was feature 2 is now feature 1.
        assert_eq!(next.features[1], features.features[2]);
        assert_eq!(next.features[2], features.features[3]);
    }

    #[test]
    fn add_op_lands_at_index_len() {
        let features = collection(2);
        let next = apply_ops(
            &features,
            &[FeatureOp::Add {
                index: features.features.len(),
                feature: Box::new(point_feature(99.0, 9.0)),
            }],
        )
        .expect("appending is in range");
        assert_eq!(next.features.len(), 3);
        assert_eq!(next.features[2], point_feature(99.0, 9.0));
    }

    #[test]
    fn apply_ops_then_apply_inverted_returns_an_equal_collection() {
        let features = collection(4);
        let layer = LayerId::new();
        let mut moved = features.features[2].clone();
        set_vertex(&mut moved, VertexRef::new(0), LonLat::new(42.0, 43.0)).expect("in range");

        let cases: Vec<Vec<FeatureOp>> = vec![
            vec![FeatureOp::Add {
                index: 1,
                feature: Box::new(point_feature(7.0, 7.0)),
            }],
            vec![FeatureOp::Remove {
                index: 0,
                feature: Box::new(features.features[0].clone()),
            }],
            vec![FeatureOp::Replace {
                index: 2,
                before: Box::new(features.features[2].clone()),
                after: Box::new(moved.clone()),
            }],
            // A six-operation mixed transaction: the order matters, and the
            // inverse has to walk it backwards.
            vec![
                FeatureOp::Add {
                    index: 0,
                    feature: Box::new(point_feature(-1.0, -1.0)),
                },
                FeatureOp::Replace {
                    index: 3,
                    before: Box::new(features.features[2].clone()),
                    after: Box::new(moved.clone()),
                },
                FeatureOp::Remove {
                    index: 1,
                    feature: Box::new(features.features[0].clone()),
                },
                FeatureOp::Add {
                    index: 2,
                    feature: Box::new(point_feature(-2.0, -2.0)),
                },
                FeatureOp::Remove {
                    index: 0,
                    feature: Box::new(point_feature(-1.0, -1.0)),
                },
                FeatureOp::Add {
                    index: 0,
                    feature: Box::new(point_feature(-3.0, -3.0)),
                },
            ],
        ];

        for ops in cases {
            let transaction = EditTransaction {
                layer,
                label: "test",
                ops,
                selection_before: None,
                selection_after: None,
                coalesce: None,
            };
            let forward = apply_ops(&features, &transaction.ops).expect("forward applies");
            let back = apply_ops(&forward, &transaction.inverted().ops).expect("inverse applies");
            assert_eq!(
                back, features,
                "round trip differed for {:?}",
                transaction.ops
            );
        }
    }

    #[test]
    fn apply_ops_out_of_range_errors_and_leaves_the_input_untouched() {
        let features = collection(2);
        let snapshot = features.clone();
        for (ops, expected) in [
            (
                vec![FeatureOp::Add {
                    index: 3,
                    feature: Box::new(point_feature(0.0, 0.0)),
                }],
                EditError::IndexOutOfRange { index: 3, len: 2 },
            ),
            (
                vec![FeatureOp::Remove {
                    index: 2,
                    feature: Box::new(point_feature(0.0, 0.0)),
                }],
                EditError::IndexOutOfRange { index: 2, len: 2 },
            ),
            (
                vec![FeatureOp::Replace {
                    index: 9,
                    before: Box::new(point_feature(0.0, 0.0)),
                    after: Box::new(point_feature(1.0, 1.0)),
                }],
                EditError::IndexOutOfRange { index: 9, len: 2 },
            ),
        ] {
            assert_eq!(apply_ops(&features, &ops), Err(expected));
            assert_eq!(features, snapshot);
        }

        // And a failure part-way through a multi-op transaction yields nothing
        // at all, not a half-applied collection.
        let ops = vec![
            FeatureOp::Add {
                index: 0,
                feature: Box::new(point_feature(5.0, 5.0)),
            },
            FeatureOp::Remove {
                index: 99,
                feature: Box::new(point_feature(5.0, 5.0)),
            },
        ];
        assert!(apply_ops(&features, &ops).is_err());
        assert_eq!(features, snapshot);
    }

    #[test]
    fn transaction_inverted_twice_equals_the_original() {
        // `inverted` deliberately drops the coalescing key — an undo must never
        // join the gesture window the original edit belonged to — so the round
        // trip is stated for a non-coalescing transaction.
        let transaction = EditTransaction {
            layer: LayerId::new(),
            label: "Move vertex",
            ops: vec![
                FeatureOp::Add {
                    index: 0,
                    feature: Box::new(point_feature(1.0, 2.0)),
                },
                FeatureOp::Replace {
                    index: 1,
                    before: Box::new(point_feature(3.0, 4.0)),
                    after: Box::new(point_feature(5.0, 6.0)),
                },
            ],
            selection_before: Some(EditSelection::feature(0)),
            selection_after: Some(EditSelection::vertex(1, VertexRef::at(0, 0, 2))),
            coalesce: None,
        };
        assert_eq!(transaction.inverted().inverted(), transaction);
        assert_eq!(transaction.touched_indices(), vec![0, 1]);
    }

    #[test]
    fn apply_ops_never_mutates_its_input() {
        let features = collection(3);
        let snapshot = features.clone();
        let _next = apply_ops(
            &features,
            &[
                FeatureOp::Remove {
                    index: 0,
                    feature: Box::new(features.features[0].clone()),
                },
                FeatureOp::Add {
                    index: 0,
                    feature: Box::new(point_feature(0.5, 0.5)),
                },
            ],
        )
        .expect("in range");
        assert_eq!(features, snapshot);
    }

    #[test]
    fn multipolygon_vertex_ref_addresses_part_ring_and_index() {
        let square = |offset: f64| -> Vec<Position> {
            vec![
                position(&[offset, 0.0]),
                position(&[offset + 1.0, 0.0]),
                position(&[offset + 1.0, 1.0]),
                position(&[offset, 0.0]),
            ]
        };
        let hole = |offset: f64| -> Vec<Position> {
            vec![
                position(&[offset + 0.2, 0.2]),
                position(&[offset + 0.4, 0.2]),
                position(&[offset + 0.4, 0.4]),
                position(&[offset + 0.2, 0.2]),
            ]
        };
        let multi = MultiPolygon::new(vec![
            vec![square(0.0), hole(0.0)],
            vec![square(10.0), hole(10.0)],
        ])
        .expect("two polygons");
        let mut feature = Feature::new(Some(Geometry::MultiPolygon(multi)), None);
        let before = match feature.geometry.as_ref() {
            Some(Geometry::MultiPolygon(multi)) => multi.coordinates.clone(),
            other => panic!("expected a multipolygon, got {other:?}"),
        };

        set_vertex(
            &mut feature,
            VertexRef::at(1, 1, 1),
            LonLat::new(77.0, 78.0),
        )
        .expect("part 1, ring 1, index 1 exists");

        match feature.geometry.as_ref() {
            Some(Geometry::MultiPolygon(multi)) => {
                assert_eq!(multi.coordinates[1][1][1], position(&[77.0, 78.0]));
                assert_eq!(multi.coordinates[0], before[0], "nothing else moved");
                assert_eq!(multi.coordinates[1][0], before[1][0]);
            }
            other => panic!("expected a multipolygon, got {other:?}"),
        }

        // A part or ring that does not exist is refused, not clamped.
        assert_eq!(
            set_vertex(&mut feature, VertexRef::at(5, 0, 0), LonLat::new(0.0, 0.0)),
            Err(EditError::BadVertex(VertexRef::at(5, 0, 0)))
        );
    }

    #[test]
    fn geometry_collection_members_are_vertex_editable_through_flattened_parts() {
        // GC[Point, LineString]: the point is part 0, the line part 1 —
        // members number straight through, exactly as `paths` flattens.
        let inner = GeometryCollection::new(vec![
            Geometry::Point(Point::new(position(&[1.0, 2.0])).expect("valid")),
            Geometry::LineString(
                LineString::new(vec![position(&[0.0, 0.0]), position(&[10.0, 0.0])])
                    .expect("a line"),
            ),
        ])
        .expect("two members");
        let mut feature = Feature::new(Some(Geometry::GeometryCollection(inner)), None);

        set_vertex(&mut feature, VertexRef::new(0), LonLat::new(3.0, 4.0))
            .expect("the point member is part 0");
        set_vertex(&mut feature, VertexRef::at(1, 0, 1), LonLat::new(20.0, 5.0))
            .expect("the line member is part 1");
        match feature.geometry.as_ref() {
            Some(Geometry::GeometryCollection(collection)) => {
                match (&collection.geometries[0], &collection.geometries[1]) {
                    (Geometry::Point(point), Geometry::LineString(line)) => {
                        assert_eq!(point.coordinates, position(&[3.0, 4.0]));
                        assert_eq!(line.coordinates[1], position(&[20.0, 5.0]));
                        assert_eq!(line.coordinates[0], position(&[0.0, 0.0]));
                    }
                    other => panic!("member kinds changed: {other:?}"),
                }
            }
            other => panic!("expected a collection, got {other:?}"),
        }

        // A part past the last member is refused, not clamped.
        assert_eq!(
            set_vertex(&mut feature, VertexRef::at(2, 0, 0), LonLat::new(0.0, 0.0)),
            Err(EditError::BadVertex(VertexRef::at(2, 0, 0)))
        );

        // And the attributes stay editable as before.
        let mut properties = Properties::new();
        properties.insert("name".to_string(), serde_json::Value::from("mixed"));
        set_properties(&mut feature, properties);
        assert!(feature.properties.is_some());
    }

    #[test]
    fn paths_and_paths_mut_agree_on_every_part_ring_pair_for_a_nested_collection() {
        // GC[MultiLineString(2), GC[Polygon(with hole)], Point]: parts must
        // number 0,1 (lines), 2 (polygon, both rings), 3 (point) in BOTH
        // functions — a single divergence would draw a handle at one address
        // and move another.
        let polygon = Polygon::new(vec![
            vec![
                position(&[0.0, 0.0]),
                position(&[10.0, 0.0]),
                position(&[10.0, 10.0]),
                position(&[0.0, 10.0]),
                position(&[0.0, 0.0]),
            ],
            vec![
                position(&[4.0, 4.0]),
                position(&[6.0, 4.0]),
                position(&[6.0, 6.0]),
                position(&[4.0, 6.0]),
                position(&[4.0, 4.0]),
            ],
        ])
        .expect("a donut");
        let nested = GeometryCollection::new(vec![Geometry::Polygon(polygon)]).expect("inner");
        let outer = GeometryCollection::new(vec![
            Geometry::MultiLineString(
                MultiLineString::new(vec![
                    vec![position(&[0.0, 0.0]), position(&[1.0, 1.0])],
                    vec![position(&[2.0, 2.0]), position(&[3.0, 3.0])],
                ])
                .expect("two lines"),
            ),
            Geometry::GeometryCollection(nested),
            Geometry::Point(Point::new(position(&[9.0, 9.0])).expect("valid")),
        ])
        .expect("outer");
        let mut geometry = Geometry::GeometryCollection(outer);

        let read: Vec<(usize, usize, PathKind, usize)> = paths(&geometry)
            .iter()
            .map(|path| (path.part, path.ring, path.kind, path.positions.len()))
            .collect();
        assert_eq!(
            read,
            vec![
                (0, 0, PathKind::Line, 2),
                (1, 0, PathKind::Line, 2),
                (2, 0, PathKind::Ring, 4),
                (2, 1, PathKind::Ring, 4),
                (3, 0, PathKind::Points, 1),
            ],
        );
        let written: Vec<(usize, usize, PathKind)> = paths_mut(&mut geometry)
            .expect("collections are editable now")
            .iter()
            .map(|slot| (slot.part, slot.ring, slot.kind))
            .collect();
        let read_addresses: Vec<(usize, usize, PathKind)> = read
            .iter()
            .map(|&(part, ring, kind, _)| (part, ring, kind))
            .collect();
        assert_eq!(written, read_addresses, "the two flattenings must agree");
    }

    #[test]
    fn malformed_position_shorter_than_two_is_rejected_not_panicked() {
        // A hand-built one-element position: `Point::new` refuses it, but a
        // hostile document read straight off disk does not go through `new`.
        let mut feature = Feature::new(
            Some(Geometry::Point(Point {
                coordinates: position(&[1.0]),
                bbox: None,
            })),
            None,
        );
        let untouched = feature.clone();
        assert_eq!(
            set_vertex(&mut feature, VertexRef::new(0), LonLat::new(3.0, 4.0)),
            Err(EditError::MalformedPosition(VertexRef::new(0)))
        );
        assert_eq!(feature, untouched);

        // A non-finite target is refused too: it would serialize as JSON
        // `null` and make the layer's stored text unreadable.
        let mut line = line_feature(&[[0.0, 0.0], [1.0, 1.0]]);
        let untouched = line.clone();
        assert_eq!(
            set_vertex(&mut line, VertexRef::new(0), LonLat::new(f64::NAN, 0.0)),
            Err(EditError::MalformedPosition(VertexRef::new(0)))
        );
        assert_eq!(
            insert_vertex(
                &mut line,
                VertexRef::new(1),
                LonLat::new(0.0, f64::INFINITY)
            ),
            Err(EditError::MalformedPosition(VertexRef::new(1)))
        );
        assert_eq!(line, untouched);

        // And a feature with no geometry at all refuses rather than panics.
        let mut null_geometry = Feature::new(None, None);
        assert_eq!(
            set_vertex(&mut null_geometry, VertexRef::new(0), LonLat::new(0.0, 0.0)),
            Err(EditError::NoGeometry)
        );
    }

    #[test]
    fn unclosed_imported_ring_is_normalized_not_corrupted() {
        // Three positions, first != last: a ring that arrived unclosed. Popping
        // blindly would eat a real vertex.
        let polygon = Polygon::from_exterior(vec![
            position(&[0.0, 0.0]),
            position(&[1.0, 0.0]),
            position(&[1.0, 1.0]),
        ])
        .expect("one ring");
        let mut feature = Feature::new(Some(Geometry::Polygon(polygon)), None);
        set_vertex(&mut feature, VertexRef::new(2), LonLat::new(2.0, 2.0)).expect("in range");
        let ring = exterior_of(&feature);
        assert_eq!(
            ring.len(),
            4,
            "the three original positions survive and the ring is closed"
        );
        assert_eq!(ring[0], position(&[0.0, 0.0]));
        assert_eq!(ring[1], position(&[1.0, 0.0]));
        assert_eq!(ring[2], position(&[2.0, 2.0]));
        assert_eq!(ring[3], ring[0]);
    }

    #[test]
    fn set_properties_replaces_the_whole_map_and_scrubs_bboxes() {
        let mut feature = point_feature(1.0, 2.0);
        feature.add_property("old", "value");
        feature.bbox = Some(vec![1.0, 2.0, 1.0, 2.0]);
        if let Some(Geometry::Point(point)) = feature.geometry.as_mut() {
            point.bbox = Some(vec![1.0, 2.0, 1.0, 2.0]);
        }

        let mut replacement = Properties::new();
        replacement.insert("name".to_string(), serde_json::Value::from("new"));
        set_properties(&mut feature, replacement);

        let properties = feature.properties.as_ref().expect("set");
        assert_eq!(properties.len(), 1, "the old key is gone, not merged");
        assert_eq!(
            properties.get("name").and_then(serde_json::Value::as_str),
            Some("new")
        );
        assert!(feature.bbox.is_none());
        match feature.geometry.as_ref() {
            Some(Geometry::Point(point)) => assert!(point.bbox.is_none()),
            other => panic!("expected a point, got {other:?}"),
        }
    }

    #[test]
    fn multipoint_and_multilinestring_are_editable_through_the_same_paths() {
        let multi = MultiPoint::new(vec![
            position(&[0.0, 0.0]),
            position(&[1.0, 1.0]),
            position(&[2.0, 2.0]),
        ])
        .expect("three members");
        let mut feature = Feature::new(Some(Geometry::MultiPoint(multi)), None);
        // A MultiPoint is one path whose index selects the member.
        set_vertex(&mut feature, VertexRef::new(2), LonLat::new(5.0, 6.0)).expect("in range");
        remove_vertex(&mut feature, VertexRef::new(0)).expect("two members remain");
        match feature.geometry.as_ref() {
            Some(Geometry::MultiPoint(multi)) => {
                assert_eq!(multi.coordinates.len(), 2);
                assert_eq!(multi.coordinates[1], position(&[5.0, 6.0]));
            }
            other => panic!("expected a multipoint, got {other:?}"),
        }

        let lines = MultiLineString::new(vec![
            vec![position(&[0.0, 0.0]), position(&[1.0, 0.0])],
            vec![position(&[0.0, 5.0]), position(&[1.0, 5.0])],
        ])
        .expect("two lines");
        let mut feature = Feature::new(Some(Geometry::MultiLineString(lines)), None);
        set_vertex(&mut feature, VertexRef::at(1, 0, 1), LonLat::new(9.0, 9.0))
            .expect("part 1 exists");
        match feature.geometry.as_ref() {
            Some(Geometry::MultiLineString(lines)) => {
                assert_eq!(lines.coordinates[1][1], position(&[9.0, 9.0]));
                assert_eq!(lines.coordinates[0][1], position(&[1.0, 0.0]));
            }
            other => panic!("expected a multilinestring, got {other:?}"),
        }
    }

    #[test]
    fn a_point_can_be_moved_but_never_grown_or_emptied() {
        let mut feature = point_feature(1.0, 2.0);
        let grown = insert_vertex(&mut feature, VertexRef::new(0), LonLat::new(3.0, 4.0));
        assert!(
            matches!(grown, Err(EditError::UnsupportedGeometry(_))),
            "got {grown:?}"
        );
        assert_eq!(
            remove_vertex(&mut feature, VertexRef::new(0)),
            Err(EditError::TooFewVertices { have: 0, need: 1 })
        );
        assert_eq!(feature, point_feature(1.0, 2.0));
    }

    #[test]
    fn line_positions_are_addressable_from_zero_with_no_ring_closure() {
        let mut feature = line_feature(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]);
        set_vertex(&mut feature, VertexRef::new(0), LonLat::new(-1.0, -1.0)).expect("in range");
        assert_eq!(line_of(&feature).len(), 3, "a line is never closed");
        assert_eq!(line_of(&feature)[0], position(&[-1.0, -1.0]));
        assert_eq!(
            set_vertex(&mut feature, VertexRef::new(3), LonLat::new(0.0, 0.0)),
            Err(EditError::BadVertex(VertexRef::new(3)))
        );
    }
}

mod state_tests {
    use super::*;
    use crate::edit::{EditMode, EditState, Sketch, VertexDrag};
    use crate::local_input::LocalInputState;
    use crate::map_view::PanGate;
    use oxigis_core::Project;
    use oxigis_render::MapView;

    /// Runs one headless frame and reports what the pan gate answered, with the
    /// rect, response and camera the app's own call site hands it.
    fn gate_verdict(state: &mut EditState) -> PanGate {
        let ctx = egui::Context::default();
        let project = Project::new("gate");
        let local = LocalInputState::new();
        let mut verdict = None;
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(640.0, 480.0),
            )),
            ..Default::default()
        };
        let _output = ctx.run_ui(raw_input, |ui| {
            let rect = ui.available_rect_before_wrap();
            let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
            let ppp = ui.ctx().pixels_per_point();
            let view = MapView::new(
                LonLat::new(0.0, 0.0),
                3.0,
                [rect.width() * ppp, rect.height() * ppp],
            )
            .expect("a 640x480 panel is a valid viewport");
            verdict = Some(state.gate_pan(rect, &response, ppp, &project, &local, None, view));
        });
        verdict.expect("the frame ran")
    }

    /// A state with a sketch and a drag in progress, on `target`.
    fn busy(target: LayerId) -> EditState {
        let mut state = EditState::default();
        state.set_mode(EditMode::DrawLine);
        assert!(state.retarget(Some(target)).is_none());
        state.set_selection(Some(EditSelection::vertex(1, VertexRef::new(2))));
        *state.sketch_mut() = Sketch {
            mode: Some(EditMode::DrawLine),
            points: vec![LonLat::new(1.0, 2.0)],
            cursor: None,
        };
        state.set_drag(Some(VertexDrag {
            moved: true,
            ..VertexDrag::single(
                1,
                VertexRef::new(2),
                false,
                Geometry::Point(Point::new(position(&[1.0, 2.0])).expect("valid")),
                LonLat::new(1.0, 2.0),
            )
        }));
        state
    }

    #[test]
    fn escape_undoes_exactly_one_layer_of_in_progress_ness() {
        let mut state = busy(LayerId::new());
        assert!(state.escape(), "first: the sketch");
        assert!(!state.sketch().is_active());
        assert!(state.drag().is_some(), "the drag survives the first press");

        assert!(state.escape(), "second: the drag");
        assert!(state.drag().is_none());
        assert_eq!(
            state.selection(),
            Some(EditSelection::vertex(1, VertexRef::new(2)))
        );

        assert!(state.escape(), "third: the vertex");
        assert_eq!(state.selection(), Some(EditSelection::feature(1)));

        assert!(state.escape(), "fourth: the feature");
        assert_eq!(state.selection(), None);

        assert!(state.escape(), "fifth: back out of the draw tool");
        assert_eq!(state.mode(), EditMode::Select);

        assert!(state.escape(), "sixth: editing off");
        assert_eq!(state.mode(), EditMode::Off);

        assert!(
            !state.escape(),
            "and then the key belongs to someone else again"
        );
    }

    #[test]
    fn retarget_discards_everything_that_addresses_the_old_layer() {
        let first = LayerId::new();
        let second = LayerId::new();
        let mut state = busy(first);
        assert_eq!(state.target(), Some(first));

        let notice = state
            .retarget(Some(second))
            .expect("the discarded sketch is worth saying");
        assert!(notice.message().contains("sketch"));
        assert_eq!(state.target(), Some(second));
        assert_eq!(state.selection(), None);
        assert!(state.drag().is_none());
        assert!(!state.sketch().is_active());
        assert_eq!(
            state.mode(),
            EditMode::DrawLine,
            "the tool is a user choice, not a property of the layer"
        );

        // Re-targeting the same layer is a no-op, which is what lets it be
        // called every frame.
        state.set_selection(Some(EditSelection::feature(0)));
        assert!(state.retarget(Some(second)).is_none());
        assert_eq!(state.selection(), Some(EditSelection::feature(0)));
    }

    #[test]
    fn validate_selection_clamps_against_the_live_collection() {
        let mut state = EditState::default();
        state.set_selection(Some(EditSelection::vertex(2, VertexRef::new(0))));

        state.validate_selection(Some(&collection(3)));
        assert_eq!(
            state.selection(),
            Some(EditSelection::vertex(2, VertexRef::new(0))),
            "a selection that is still in range survives"
        );

        state.validate_selection(Some(&collection(2)));
        assert_eq!(state.selection(), None, "feature 2 of 2 is gone");

        // A feature that lost its geometry keeps the feature selection but
        // loses the vertex.
        state.set_selection(Some(EditSelection::vertex(0, VertexRef::new(0))));
        let null_geometry = FeatureCollection::new(vec![Feature::new(None, None)]);
        state.validate_selection(Some(&null_geometry));
        assert_eq!(state.selection(), Some(EditSelection::feature(0)));

        // No collection at all clears everything.
        state.validate_selection(None);
        assert_eq!(state.selection(), None);
    }

    /// Regression (finding 97): a stale ANCHOR must re-anchor the set, not
    /// drop it outright — `FeatureSelection::clamped` already did the right
    /// thing and had zero production callers before this fix.
    #[test]
    fn validate_selection_reanchors_a_multi_select_instead_of_dropping_it() {
        let mut state = EditState::default();
        assert!(state.toggle_feature(0), "selects {{0}}, anchored on 0");
        assert!(state.toggle_feature(4), "adds 4, features [0, 4], anchor 4");

        // Only features 0 and 1 exist: the anchor (4) is stale, but member 0
        // is not.
        state.validate_selection(Some(&collection(2)));
        assert_eq!(
            state
                .multi_selection()
                .map(|multi| multi.features().to_vec()),
            Some(vec![0]),
            "member 0 survives even though the anchor did not"
        );
        assert_eq!(
            state.selection(),
            Some(EditSelection::feature(0)),
            "the set re-anchors on the surviving member instead of vanishing"
        );
    }

    /// Regression (finding 97): a marked vertex set is clamped against the
    /// anchor's CURRENT geometry, not just checked for feature membership —
    /// a mark whose `(part, ring, index)` no longer resolves must be dropped,
    /// or a later Delete fires at a mark that names nothing.
    #[test]
    fn validate_selection_drops_vertex_marks_that_no_longer_resolve() {
        use super::super::selection::FeatureSelection;

        let mut state = EditState::default();
        // Feature 0 is a `Point`: its only valid address is vertex 0.
        // Marking vertex 5 too models a mark surviving a geometry that
        // shrank out from under it.
        state.set_multi_selection(Some(
            FeatureSelection::single(0).with_vertex_set(vec![VertexRef::new(0), VertexRef::new(5)]),
        ));
        state.validate_selection(Some(&collection(1)));
        assert_eq!(
            state
                .multi_selection()
                .map(|multi| multi.vertex_set().to_vec()),
            Some(vec![VertexRef::new(0)]),
            "the mark that still addresses a real position survives; the \
             one that does not is dropped"
        );
    }

    #[test]
    fn reset_clears_the_whole_machine_and_an_idle_frame_allows_pan_in_every_mode() {
        let mut state = busy(LayerId::new());
        state.push_notice(crate::edit::EditNotice::new("something happened"));
        state.set_show_window(true);
        state.reset();

        assert_eq!(state.mode(), EditMode::Off);
        assert_eq!(state.target(), None);
        assert_eq!(state.selection(), None);
        assert!(state.drag().is_none());
        assert!(!state.sketch().is_active());
        assert!(state.notices().is_empty());
        assert!(!state.show_window());
        assert_eq!(gate_verdict(&mut state), PanGate::Allow);

        // `gate_verdict` runs a frame with **no pointer events**, so what this
        // loop pins is that a frame with no drag in it answers `Allow` in
        // every mode — no tool may hold the camera hostage while merely
        // selected. It deliberately does not exercise invariant I11 ("`Off`
        // never suppresses a real drag"); that needs a genuine drag gesture
        // and is pinned by the app-level
        // `edit_mode_off_consumes_no_click_and_never_suppresses_pan`.
        for mode in EditMode::ALL {
            state.set_mode(mode);
            assert_eq!(gate_verdict(&mut state), PanGate::Allow);
        }
        assert!(EditMode::DrawPolygon.is_drawing());
        assert!(!EditMode::Select.is_drawing());
        assert_eq!(EditMode::Off.label(), "Browse");
    }

    #[test]
    fn a_cancelled_drag_latch_ticks_in_off_and_never_swallows_the_next_gesture() {
        // Cancelling a drag (here: by switching tools mid-gesture) latches the
        // pan gate shut until the button that drove the gesture comes back up.
        // The latch must keep ticking in `Off` too: serviced only in the mode
        // that set it, it would strand itself across a trip through `Off` and
        // then swallow the user's whole next pan gesture.
        let ctx = egui::Context::default();
        let project = Project::new("gate");
        let local = LocalInputState::new();
        let mut state = busy(LayerId::new());
        state.set_mode(EditMode::Off); // cancel_drag: the latch is now set
        assert!(state.drag().is_none());

        let frame = |state: &mut EditState, events: Vec<egui::Event>| -> PanGate {
            let mut verdict = None;
            let raw_input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(640.0, 480.0),
                )),
                events,
                ..Default::default()
            };
            let _output = ctx.run_ui(raw_input, |ui| {
                let rect = ui.available_rect_before_wrap();
                let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
                let ppp = ui.ctx().pixels_per_point();
                let view = MapView::new(
                    LonLat::new(0.0, 0.0),
                    3.0,
                    [rect.width() * ppp, rect.height() * ppp],
                )
                .expect("a 640x480 panel is a valid viewport");
                verdict = Some(state.gate_pan(rect, &response, ppp, &project, &local, None, view));
            });
            verdict.expect("the frame ran")
        };
        let press = egui::Event::PointerButton {
            pos: egui::pos2(320.0, 240.0),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        };
        let release = egui::Event::PointerButton {
            pos: egui::pos2(320.0, 240.0),
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        };

        // The cancelled gesture's button is still held: its tail stays
        // suppressed, in `Off` and back in `Select` alike — handing it to the
        // camera would be exactly the lurch the latch exists to prevent.
        assert_eq!(frame(&mut state, vec![press.clone()]), PanGate::Suppress);
        state.set_mode(EditMode::Select);
        assert_eq!(frame(&mut state, Vec::new()), PanGate::Suppress);

        // Release clears the latch; the next gesture belongs to the camera.
        assert_eq!(frame(&mut state, vec![release]), PanGate::Allow);
        assert_eq!(frame(&mut state, vec![press]), PanGate::Allow);
    }

    #[test]
    fn refresh_drag_rederives_the_position_from_the_fresh_camera() {
        // The pan gate updates a drag *before* `allocate_gated` applies the
        // frame's wheel zoom; `refresh_drag` runs after it, with the post-zoom
        // camera, so a release on a zooming frame commits the position the
        // overlay actually showed.
        let project = Project::new("zoom");
        let mut state = EditState::default();
        state.set_mode(EditMode::Select);
        state.set_drag(Some(VertexDrag::single(
            0,
            VertexRef::new(0),
            false,
            Geometry::Point(Point::new(position(&[0.0, 0.0])).expect("valid")),
            LonLat::new(0.0, 0.0),
        )));
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(640.0, 480.0));
        let pointer = Some(egui::pos2(200.0, 140.0)); // deliberately off-centre
        let view_at = |zoom| {
            MapView::new(LonLat::new(0.0, 0.0), zoom, [640.0, 480.0])
                .expect("a 640x480 viewport is valid")
        };
        let ctx_at = |zoom| crate::edit::EditCtx {
            project: &project,
            target: None,
            features: None,
            view: view_at(zoom),
            rect,
            ppp: 1.0,
        };

        // `suspend_snap` keeps the query out of it: this pins the projection.
        state.refresh_drag(&ctx_at(3.0), pointer, true);
        let before = state.drag().expect("the drag is live").current;
        state.refresh_drag(&ctx_at(4.0), pointer, true);
        let after = state.drag().expect("the drag is live").current;
        assert_ne!(
            before, after,
            "a zoom step reprojects the same pointer to a different world position"
        );
        let expected = view_at(4.0).screen_to_lon_lat([200.0, 140.0]);
        assert_eq!(
            after, expected,
            "the drag lands exactly where the post-zoom camera puts the pointer"
        );
    }

    #[test]
    fn the_notice_log_is_capped() {
        let mut state = EditState::default();
        for index in 0..(crate::edit::MAX_NOTICES + 5) {
            state.push_notice(crate::edit::EditNotice::new(format!("notice {index}")));
        }
        assert_eq!(state.notices().len(), crate::edit::MAX_NOTICES);
        assert_eq!(
            state.notices()[0].message(),
            "notice 5",
            "the oldest are dropped, not the newest"
        );
        state.clear_notices();
        assert!(state.notices().is_empty());
    }
}

mod stack_tests {
    use super::*;
    use crate::edit::stack::{
        CoalesceField, CoalesceKey, EditStack, UNDO_MAX_BYTES_NATIVE, UNDO_MAX_BYTES_WASM,
        UNDO_MAX_ENTRIES,
    };

    /// A `Replace` transaction turning `before` into `after` at feature `index`.
    fn replace(
        layer: LayerId,
        label: &'static str,
        index: usize,
        before: Feature,
        after: Feature,
    ) -> EditTransaction {
        EditTransaction::single(
            layer,
            label,
            FeatureOp::Replace {
                index,
                before: Box::new(before),
                after: Box::new(after),
            },
        )
    }

    /// A cheap `Add` transaction, for cursor bookkeeping tests.
    fn add(layer: LayerId, label: &'static str, index: usize) -> EditTransaction {
        EditTransaction::single(
            layer,
            label,
            FeatureOp::Add {
                index,
                feature: Box::new(point_feature(index as f64, 0.0)),
            },
        )
    }

    /// A transaction whose payload is `positions` positions long, so byte-budget
    /// tests can control the footprint directly.
    fn fat(layer: LayerId, positions: usize) -> EditTransaction {
        let coords: Vec<[f64; 2]> = (0..positions)
            .map(|index| [index as f64, index as f64])
            .collect();
        EditTransaction::single(
            layer,
            "Fat",
            FeatureOp::Add {
                index: 0,
                feature: Box::new(line_feature(&coords)),
            },
        )
    }

    #[test]
    fn push_truncates_the_redo_tail() {
        let layer = LayerId::new();
        let mut stack = EditStack::with_budget(16, 1 << 20);
        stack.push(add(layer, "one", 0));
        stack.push(add(layer, "two", 1));
        stack.push(add(layer, "three", 2));
        assert_eq!(stack.depth(), (3, 0));
        stack.undo().expect("three");
        stack.undo().expect("two");
        assert_eq!(stack.depth(), (1, 2));

        stack.push(add(layer, "four", 1));
        assert_eq!(stack.depth(), (2, 0), "the redo tail is discarded");
        assert_eq!(stack.peek_undo().map(|entry| entry.label), Some("four"));
        assert!(!stack.can_redo());
    }

    #[test]
    fn prune_layer_removes_only_that_layer_and_fixes_the_cursor() {
        let first = LayerId::new();
        let second = LayerId::new();
        let mut stack = EditStack::with_budget(16, 1 << 20);
        stack.push(add(first, "a1", 0));
        stack.push(add(second, "b1", 0));
        stack.push(add(first, "a2", 1));
        stack.push(add(second, "b2", 1));
        stack.undo().expect("b2");
        assert_eq!(stack.depth(), (3, 1));

        assert_eq!(stack.prune_layer(first), 2);
        assert_eq!(
            stack.depth(),
            (1, 1),
            "both halves lose their entries and the cursor follows"
        );
        assert_eq!(stack.peek_undo().map(|entry| entry.label), Some("b1"));
        assert_eq!(stack.peek_redo().map(|entry| entry.label), Some("b2"));
        assert_eq!(stack.prune_layer(first), 0, "nothing left to prune");
    }

    #[test]
    fn pruning_one_member_of_a_grouped_add_drops_the_whole_group_entry() {
        use crate::edit::project_op::{LayerSnapshot, ProjectOp, ProjectTransaction};

        let snapshot = |name: &str| {
            let layer = oxigis_core::Layer::new(
                name,
                oxigis_core::LayerKind::Vector(oxigis_core::VectorSource::InlineGeoJson {
                    geojson: "{\"type\":\"FeatureCollection\",\"features\":[]}".to_string(),
                }),
            );
            LayerSnapshot {
                position: 0,
                style: None,
                features: None,
                default_style: None,
                layer,
            }
        };
        let first = snapshot("first");
        let second = snapshot("second");
        let member = second.layer.id;
        let mut stack = EditStack::with_budget(16, 1 << 20);
        stack.push(ProjectTransaction {
            label: "Add layers",
            op: ProjectOp::AddLayers(vec![first, second]),
            coalesce: None,
        });
        assert_eq!(stack.depth(), (1, 0));
        // A hydrate of ONE member must drop the WHOLE group: a surviving
        // entry would, on redo, splice a stale snapshot over the re-read
        // file.
        assert_eq!(stack.prune_layer(member), 1);
        assert_eq!(stack.depth(), (0, 0));
    }

    #[test]
    fn reset_bumps_the_epoch_and_clears_everything() {
        let layer = LayerId::new();
        let mut stack = EditStack::with_budget(16, 1 << 20);
        let epoch = stack.epoch();
        stack.push(add(layer, "one", 0));
        assert!(stack.bytes() > 0);
        stack.reset();
        assert_eq!(stack.depth(), (0, 0));
        assert_eq!(stack.bytes(), 0);
        assert!(!stack.can_undo() && !stack.can_redo());
        assert_ne!(stack.epoch(), epoch);
    }

    #[test]
    fn entries_from_a_stale_epoch_are_never_returned() {
        let layer = LayerId::new();
        let mut stack = EditStack::with_budget(16, 1 << 20);
        let stale = CoalesceKey {
            epoch: stack.epoch(),
            layer,
            feature: 0,
            field: CoalesceField::Properties,
        };
        let mut first = add(layer, "before the load", 0);
        first.coalesce = Some(stale);
        stack.push(first);
        assert_eq!(stack.depth(), (1, 0));

        stack.reset();
        assert!(
            stack.undo().is_none(),
            "nothing recorded before the load may come back"
        );
        assert_ne!(stack.epoch(), stale.epoch);

        // A caller still holding the old key cannot re-open the window it
        // belonged to, so two edits carrying it stay two separate entries.
        let mut carried = replace(
            layer,
            "one",
            0,
            point_feature(0.0, 0.0),
            point_feature(1.0, 0.0),
        );
        carried.coalesce = Some(stale);
        stack.push(carried);
        let mut carried_again = replace(
            layer,
            "two",
            0,
            point_feature(1.0, 0.0),
            point_feature(2.0, 0.0),
        );
        carried_again.coalesce = Some(stale);
        stack.push(carried_again);
        assert_eq!(stack.depth(), (2, 0));
    }

    #[test]
    fn coalesced_property_edits_collapse_to_one_entry_keeping_outer_selections() {
        let layer = LayerId::new();
        let mut stack = EditStack::with_budget(16, 1 << 20);
        let key = CoalesceKey {
            epoch: stack.epoch(),
            layer,
            feature: 0,
            field: CoalesceField::Properties,
        };
        let start = point_feature(0.0, 0.0);
        let middle = point_feature(1.0, 0.0);
        let end = point_feature(2.0, 0.0);

        let mut first = replace(layer, "Edit attributes", 0, start.clone(), middle.clone());
        first.selection_before = Some(EditSelection::feature(3));
        first.selection_after = Some(EditSelection::feature(0));
        first.coalesce = Some(key);
        let mut second = replace(layer, "Edit attributes", 0, middle, end.clone());
        second.selection_before = Some(EditSelection::feature(0));
        second.selection_after = Some(EditSelection::vertex(0, VertexRef::new(0)));
        second.coalesce = Some(key);

        stack.push(first);
        stack.push(second);
        assert_eq!(stack.depth(), (1, 0), "two applies, one undo step");
        let folded = stack.peek_undo().expect("one entry");
        assert_eq!(
            folded.ops,
            vec![FeatureOp::Replace {
                index: 0,
                before: Box::new(start),
                after: Box::new(end),
            }],
            "the intermediate state is gone"
        );
        assert_eq!(
            folded.selection_before,
            Some(EditSelection::feature(3)),
            "the first entry's selection_before survives"
        );
        assert_eq!(
            folded.selection_after,
            Some(EditSelection::vertex(0, VertexRef::new(0))),
            "the second entry's selection_after survives"
        );
    }

    #[test]
    fn close_coalescing_prevents_the_next_edit_from_folding() {
        let layer = LayerId::new();
        let mut stack = EditStack::with_budget(16, 1 << 20);
        let key = CoalesceKey {
            epoch: stack.epoch(),
            layer,
            feature: 0,
            field: CoalesceField::Vertex(VertexRef::new(1)),
        };
        let mut first = replace(
            layer,
            "Move vertex",
            0,
            point_feature(0.0, 0.0),
            point_feature(1.0, 0.0),
        );
        first.coalesce = Some(key);
        let mut second = replace(
            layer,
            "Move vertex",
            0,
            point_feature(1.0, 0.0),
            point_feature(2.0, 0.0),
        );
        second.coalesce = Some(key);

        stack.push(first);
        stack.close_coalescing();
        stack.push(second);
        assert_eq!(stack.depth(), (2, 0));

        // A different key does not fold either, without any explicit close.
        let mut third = replace(
            layer,
            "Move vertex",
            0,
            point_feature(2.0, 0.0),
            point_feature(3.0, 0.0),
        );
        third.coalesce = Some(CoalesceKey {
            field: CoalesceField::Vertex(VertexRef::new(2)),
            ..key
        });
        stack.push(third);
        assert_eq!(stack.depth(), (3, 0));

        // And a transaction that does not coalesce at all closes the window, so
        // the next one carrying the previous key still starts its own entry.
        let mut fourth = replace(
            layer,
            "Move vertex",
            0,
            point_feature(3.0, 0.0),
            point_feature(4.0, 0.0),
        );
        fourth.coalesce = Some(CoalesceKey {
            field: CoalesceField::Vertex(VertexRef::new(2)),
            ..key
        });
        stack.push(EditTransaction::single(
            layer,
            "Add feature",
            FeatureOp::Add {
                index: 0,
                feature: Box::new(point_feature(9.0, 9.0)),
            },
        ));
        stack.push(fourth);
        assert_eq!(stack.depth(), (5, 0));
    }

    /// A tiny deterministic generator — no dependency, and the same sequence on
    /// every machine and every run, which is the whole point of a property test
    /// that has to be debuggable when it fails.
    struct Lcg(u64);

    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        fn below(&mut self, limit: usize) -> usize {
            if limit == 0 {
                0
            } else {
                self.next_u32() as usize % limit
            }
        }
    }

    #[test]
    fn a_random_50_op_sequence_fully_undone_reproduces_the_original_collection() {
        let layer = LayerId::new();
        let original = collection(6);
        let mut current = original.clone();
        let mut stack = EditStack::with_budget(256, 64 << 20);
        let mut random = Lcg(0x5EED_1234_ABCD_0001);
        let mut applied = 0_usize;

        for step in 0..50_usize {
            let len = current.features.len();
            let op = match random.below(3) {
                0 => FeatureOp::Add {
                    index: random.below(len + 1),
                    feature: Box::new(point_feature(step as f64, 1.0)),
                },
                1 if len > 1 => {
                    let index = random.below(len);
                    FeatureOp::Remove {
                        index,
                        feature: Box::new(current.features[index].clone()),
                    }
                }
                _ if len > 0 => {
                    let index = random.below(len);
                    let before = current.features[index].clone();
                    let mut after = before.clone();
                    // Every feature in this collection is a point, so index 0
                    // is always addressable.
                    set_vertex(
                        &mut after,
                        VertexRef::new(0),
                        LonLat::new(step as f64, step as f64 / 2.0),
                    )
                    .expect("a point always has vertex 0");
                    FeatureOp::Replace {
                        index,
                        before: Box::new(before),
                        after: Box::new(after),
                    }
                }
                _ => continue,
            };
            let transaction = EditTransaction::single(layer, "step", op);
            current = apply_ops(&current, &transaction.ops).expect("in range by construction");
            stack.push(transaction);
            applied += 1;
        }

        assert!(applied > 30, "the generator produced only {applied} steps");
        assert_eq!(stack.depth(), (applied, 0));

        while let Some(entry) = stack.undo() {
            let inverse = entry
                .features()
                .expect("this suite pushes feature entries only")
                .inverted();
            current = apply_ops(&current, &inverse.ops).expect("inverse applies");
        }

        let expected = oxigeo::geojson::writer::to_string(&original).expect("serializes");
        let actual = oxigeo::geojson::writer::to_string(&current).expect("serializes");
        assert_eq!(actual, expected);
        assert_eq!(stack.depth(), (0, applied));

        // And redoing everything walks forward again without refusing a step.
        while let Some(entry) = stack.redo() {
            let forward = entry
                .features()
                .expect("this suite pushes feature entries only")
                .clone();
            current = apply_ops(&current, &forward.ops).expect("forward applies");
        }
        assert_eq!(stack.depth(), (applied, 0));
    }

    #[test]
    fn byte_budget_evicts_from_the_front_and_keeps_the_cursor_valid() {
        let layer = LayerId::new();
        let one = fat(layer, 50).estimated_bytes();
        // Room for roughly three entries.
        let mut stack = EditStack::with_budget(1_000, one * 3 + one / 2);
        for _ in 0..8 {
            stack.push(fat(layer, 50));
        }
        let (undoable, redoable) = stack.depth();
        assert!(undoable <= 4, "budget not enforced: {undoable} entries");
        assert!(undoable >= 2, "too aggressive: {undoable} entries");
        assert_eq!(redoable, 0);
        assert!(stack.bytes() <= stack.max_bytes());

        // Undo a couple, then push again: eviction must not leave the cursor
        // pointing past the entries it indexes.
        stack.undo().expect("something to undo");
        for _ in 0..5 {
            stack.push(fat(layer, 50));
        }
        let (undoable, redoable) = stack.depth();
        assert_eq!(redoable, 0);
        assert!(undoable <= 4);
        for _ in 0..undoable {
            stack.undo().expect("the cursor matches the entry count");
        }
        assert!(!stack.can_undo());

        // A single entry larger than the whole budget still survives: the last
        // thing the user did must always be undoable.
        let mut tiny = EditStack::with_budget(1_000, 8);
        tiny.push(fat(layer, 200));
        assert_eq!(tiny.depth(), (1, 0));
        assert!(tiny.bytes() > tiny.max_bytes());
    }

    #[test]
    fn entry_cap_evicts_the_oldest() {
        let layer = LayerId::new();
        let mut stack = EditStack::with_budget(3, 1 << 30);
        for label in ["one", "two", "three", "four", "five"] {
            stack.push(add(layer, label, 0));
        }
        assert_eq!(stack.depth(), (3, 0));
        let mut seen = Vec::new();
        while let Some(entry) = stack.undo() {
            seen.push(entry.label());
        }
        assert_eq!(seen, vec!["five", "four", "three"]);
    }

    #[test]
    fn push_reports_byte_budget_evictions_and_stays_silent_about_the_cap() {
        let layer = LayerId::new();
        // Entry-cap ageing: routine, so `by_cap` counts it and `by_bytes`
        // stays zero — the caller reports nothing.
        let mut capped = EditStack::with_budget(2, 1 << 30);
        assert_eq!(capped.push(add(layer, "one", 0)).total(), 0);
        assert_eq!(capped.push(add(layer, "two", 1)).total(), 0);
        let report = capped.push(add(layer, "three", 2));
        assert_eq!(report.by_cap, 1, "the cap aged one entry out");
        assert_eq!(report.by_bytes, 0, "no byte pressure was involved");

        // Byte pressure: the report says how much history the push cost.
        let one = fat(layer, 50).estimated_bytes();
        let mut squeezed = EditStack::with_budget(1_000, one * 3 + one / 2);
        for _ in 0..3 {
            assert_eq!(squeezed.push(fat(layer, 50)).total(), 0);
        }
        let report = squeezed.push(fat(layer, 50));
        assert!(report.by_bytes >= 1, "the budget must evict and say so");
        assert_eq!(report.by_cap, 0);

        // The redo tail is not an eviction: undo once, push, nothing reported
        // for the truncated redo entry.
        let mut stack = EditStack::with_budget(16, 1 << 30);
        stack.push(add(layer, "one", 0));
        stack.push(add(layer, "two", 1));
        stack.undo().expect("something to undo");
        let report = stack.push(add(layer, "other", 1));
        assert_eq!(report.total(), 0, "discarding redo is not an eviction");
    }

    #[test]
    fn undo_at_the_bottom_and_redo_at_the_top_return_none() {
        let layer = LayerId::new();
        let mut stack = EditStack::with_budget(8, 1 << 20);
        assert!(stack.undo().is_none());
        assert!(stack.redo().is_none());
        assert!(stack.peek_undo().is_none());
        assert!(stack.peek_redo().is_none());

        stack.push(add(layer, "one", 0));
        assert!(stack.redo().is_none(), "nothing to redo at the top");
        stack.undo().expect("one");
        assert!(stack.undo().is_none(), "nothing below the bottom");
        stack.redo().expect("one");
        assert!(stack.redo().is_none());
    }

    #[test]
    fn redo_after_a_failed_undo_restores_the_cursor_exactly() {
        let layer = LayerId::new();
        let mut stack = EditStack::with_budget(8, 1 << 20);
        stack.push(add(layer, "one", 0));
        stack.push(add(layer, "two", 1));
        assert_eq!(stack.depth(), (2, 0));

        // The caller took a transaction to invert, could not apply it, and put
        // the cursor back.
        let taken = stack.undo().expect("two");
        assert_eq!(stack.depth(), (1, 1));
        let restored = stack.redo().expect("the same entry comes back");
        assert_eq!(restored, taken);
        assert_eq!(stack.depth(), (2, 0));
        assert_eq!(stack.peek_undo().map(|entry| entry.label), Some("two"));
    }

    #[test]
    fn estimated_bytes_scales_with_positions_and_property_text() {
        let layer = LayerId::new();
        let small = fat(layer, 2).estimated_bytes();
        let large = fat(layer, 200).estimated_bytes();
        assert!(
            large > small * 10,
            "coordinates must dominate: {small} vs {large}"
        );

        let mut plain = point_feature(0.0, 0.0);
        let mut wordy = point_feature(0.0, 0.0);
        wordy.add_property("note", "x".repeat(4096));
        let plain_op = FeatureOp::Add {
            index: 0,
            feature: Box::new(plain.clone()),
        };
        let wordy_op = FeatureOp::Add {
            index: 0,
            feature: Box::new(wordy),
        };
        assert!(wordy_op.estimated_bytes() > plain_op.estimated_bytes() + 4_000);

        // A `Replace` pins both halves.
        plain.add_property("k", 1);
        let replace_op = FeatureOp::Replace {
            index: 0,
            before: Box::new(point_feature(0.0, 0.0)),
            after: Box::new(plain),
        };
        assert!(replace_op.estimated_bytes() > plain_op.estimated_bytes());
    }

    #[test]
    fn wasm_and_native_budgets_are_both_reachable_via_with_budget() {
        const { assert!(UNDO_MAX_BYTES_WASM < UNDO_MAX_BYTES_NATIVE) };
        let native = EditStack::with_budget(UNDO_MAX_ENTRIES, UNDO_MAX_BYTES_NATIVE);
        let browser = EditStack::with_budget(UNDO_MAX_ENTRIES, UNDO_MAX_BYTES_WASM);
        assert_eq!(native.max_bytes(), UNDO_MAX_BYTES_NATIVE);
        assert_eq!(browser.max_bytes(), UNDO_MAX_BYTES_WASM);
        assert_eq!(native.max_entries(), UNDO_MAX_ENTRIES);

        // `new()` picks whichever suits this target, and it is one of the two.
        let chosen = EditStack::new().max_bytes();
        assert!(chosen == UNDO_MAX_BYTES_NATIVE || chosen == UNDO_MAX_BYTES_WASM);

        // Both budgets really do bound a stack, on this one native run.
        let layer = LayerId::new();
        let mut browser = EditStack::with_budget(UNDO_MAX_ENTRIES, UNDO_MAX_BYTES_WASM);
        let mut native = EditStack::with_budget(UNDO_MAX_ENTRIES, UNDO_MAX_BYTES_NATIVE);
        for _ in 0..UNDO_MAX_ENTRIES {
            browser.push(fat(layer, 5_000));
            native.push(fat(layer, 5_000));
        }
        assert!(browser.bytes() <= UNDO_MAX_BYTES_WASM);
        assert!(
            browser.depth().0 < native.depth().0,
            "the smaller budget must hold fewer entries: {:?} vs {:?}",
            browser.depth(),
            native.depth()
        );
    }
}

mod drag_tests {
    use super::*;
    use crate::edit::{
        EditMode, EditState, Handles, VertexDrag, drag_transaction, remove_vertex_transaction,
    };

    /// A synthetic gesture on `feature`, exactly as the drag-release path builds
    /// one before handing it to [`drag_transaction`].
    fn drag(
        feature: usize,
        at: VertexRef,
        inserting: bool,
        to: LonLat,
        origin: &Feature,
    ) -> VertexDrag {
        VertexDrag {
            moved: true,
            ..VertexDrag::single(
                feature,
                at,
                inserting,
                origin
                    .geometry
                    .as_ref()
                    .expect("the fixture has geometry")
                    .clone(),
                to,
            )
        }
    }

    /// The `after` feature of a single-`Replace` transaction.
    fn after_of(transaction: &EditTransaction) -> &Feature {
        match transaction.ops.as_slice() {
            [FeatureOp::Replace { after, .. }] => after,
            other => panic!("expected one Replace, got {other:?}"),
        }
    }

    fn square() -> Feature {
        polygon_feature(&[
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [0.0, 0.0],
        ])
    }

    #[test]
    fn a_move_drag_builds_one_replace_that_keeps_the_ring_closed() {
        let layer = LayerId::new();
        let features = FeatureCollection::new(vec![square()]);
        let gesture = drag(
            0,
            VertexRef::new(0),
            false,
            LonLat::new(-5.0, -5.0),
            &features.features[0],
        );

        let transaction =
            drag_transaction(layer, &features, &gesture, Some(EditSelection::feature(0)))
                .expect("a ring vertex is always movable");
        assert_eq!(transaction.layer, layer);
        assert_eq!(transaction.label, "Move vertex");
        assert_eq!(transaction.ops.len(), 1);
        assert_eq!(
            transaction.selection_before,
            Some(EditSelection::feature(0))
        );
        assert_eq!(
            transaction.selection_after,
            Some(EditSelection::vertex(0, VertexRef::new(0))),
            "the vertex just placed stays picked, so Delete means that one"
        );
        assert_eq!(
            transaction.coalesce, None,
            "one gesture is one undo step, or Ctrl+Z undoes an unpredictable \
             number of moves"
        );

        let ring = exterior_of(after_of(&transaction));
        assert_eq!(ring.len(), 5, "still five stored positions");
        assert_eq!(ring[0], position(&[-5.0, -5.0]));
        assert_eq!(
            ring[4], ring[0],
            "moving handle 0 moves the closing position with it"
        );

        // The collection the transaction was built from is untouched: it holds a
        // clone, and nothing lands until the choke point applies it.
        assert_eq!(exterior_of(&features.features[0])[0], position(&[0.0, 0.0]));
    }

    #[test]
    fn an_insert_drag_builds_an_insert_and_names_itself_so() {
        let layer = LayerId::new();
        let features = FeatureCollection::new(vec![square()]);
        // The ghost on the wrap segment inserts at the append index.
        let gesture = drag(
            0,
            VertexRef::new(4),
            true,
            LonLat::new(-2.0, 5.0),
            &features.features[0],
        );

        let transaction = drag_transaction(layer, &features, &gesture, None)
            .expect("the append index is in range");
        assert_eq!(transaction.label, "Insert vertex");
        let ring = exterior_of(after_of(&transaction));
        assert_eq!(ring.len(), 6, "one more stored position");
        assert_eq!(ring[4], position(&[-2.0, 5.0]));
        assert_eq!(ring[5], ring[0], "and the ring is still closed");
        assert_eq!(
            transaction.selection_after,
            Some(EditSelection::vertex(0, VertexRef::new(4)))
        );
    }

    #[test]
    fn a_drag_of_a_feature_that_has_gone_is_refused_rather_than_panicking() {
        let layer = LayerId::new();
        let features = FeatureCollection::new(vec![square()]);
        let gesture = drag(
            7,
            VertexRef::new(0),
            false,
            LonLat::new(1.0, 1.0),
            &features.features[0],
        );
        assert_eq!(
            drag_transaction(layer, &features, &gesture, None),
            Err(EditError::IndexOutOfRange { index: 7, len: 1 })
        );

        // And so is a vertex that does not exist on a feature that does.
        let gone = drag(
            0,
            VertexRef::new(99),
            false,
            LonLat::new(1.0, 1.0),
            &features.features[0],
        );
        assert_eq!(
            drag_transaction(layer, &features, &gone, None),
            Err(EditError::BadVertex(VertexRef::new(99)))
        );
    }

    #[test]
    fn a_delete_vertex_transaction_recloses_the_ring_and_drops_the_vertex_selection() {
        let layer = LayerId::new();
        let features = FeatureCollection::new(vec![square()]);
        let before = Some(EditSelection::vertex(0, VertexRef::new(0)));

        let transaction = remove_vertex_transaction(layer, &features, 0, VertexRef::new(0), before)
            .expect("four open positions may lose one");
        assert_eq!(transaction.label, "Delete vertex");
        assert_eq!(transaction.selection_before, before);
        assert_eq!(
            transaction.selection_after,
            Some(EditSelection::feature(0)),
            "the vertex is gone, so Delete must not stay armed against whatever \
             slid into its index"
        );
        let ring = exterior_of(after_of(&transaction));
        assert_eq!(ring.len(), 4, "three open positions, closed");
        assert_eq!(ring[0], position(&[10.0, 0.0]));
        assert_eq!(ring[3], ring[0]);
    }

    #[test]
    fn a_delete_below_the_minimum_is_refused_and_names_the_shortfall() {
        let layer = LayerId::new();
        let triangle = FeatureCollection::new(vec![polygon_feature(&[
            [0.0, 0.0],
            [10.0, 0.0],
            [5.0, 10.0],
            [0.0, 0.0],
        ])]);
        assert_eq!(
            remove_vertex_transaction(layer, &triangle, 0, VertexRef::new(1), None),
            Err(EditError::TooFewVertices { have: 2, need: 3 }),
            "refusing is the non-destructive answer"
        );

        let segment = FeatureCollection::new(vec![line_feature(&[[0.0, 0.0], [1.0, 1.0]])]);
        assert_eq!(
            remove_vertex_transaction(layer, &segment, 0, VertexRef::new(0), None),
            Err(EditError::TooFewVertices { have: 1, need: 2 })
        );
        assert_eq!(
            remove_vertex_transaction(layer, &segment, 4, VertexRef::new(0), None),
            Err(EditError::IndexOutOfRange { index: 4, len: 1 })
        );
    }

    #[test]
    fn escape_during_a_drag_restores_without_a_command_and_holds_the_gate_shut() {
        let mut state = EditState::default();
        state.set_mode(EditMode::Select);
        let feature = square();
        state.set_drag(Some(drag(
            0,
            VertexRef::new(0),
            false,
            LonLat::new(-5.0, -5.0),
            &feature,
        )));

        assert!(state.escape(), "the drag is a rung of the ladder");
        assert!(state.drag().is_none());
        assert_eq!(
            exterior_of(&feature)[0],
            position(&[0.0, 0.0]),
            "the pre-drag geometry was never committed, so there is nothing to \
             undo and nothing to restore"
        );
        assert_eq!(
            state.handles(),
            Handles::None,
            "nothing has been planned this frame"
        );
    }
}
