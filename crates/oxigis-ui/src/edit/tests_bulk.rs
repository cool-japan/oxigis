// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Property tests pinning [`command::apply_ops`]'s bulk fast paths to the
//! sequential reference implementation — identical results on `Ok` (feature
//! identity AND order) and identical payloads on `Err`.

use super::command::{self, EditTransaction, FeatureOp, apply_ops, apply_ops_sequential};
use oxigeo::geojson::types::{Feature, FeatureCollection, Geometry, Point, Properties};
use oxigis_core::LayerId;

/// A tiny deterministic generator — no dependency, and the same sequence on
/// every machine and every run, so a failing round is reproducible.
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
            return 0;
        }
        self.next_u32() as usize % limit
    }
}

/// A collection of `count` distinct point features, each recognisable by its
/// x coordinate, so order and identity survive an equality assertion.
fn collection(count: usize) -> FeatureCollection {
    FeatureCollection::new(
        (0..count)
            .map(|index| {
                let point = Point::new(vec![index as f64, -(index as f64)])
                    .expect("a two-element position is a Point");
                Feature::new(Some(Geometry::Point(point)), Some(Properties::new()))
            })
            .collect(),
    )
}

/// Strictly descending `Remove` ops over `indices` (deduped internally), the
/// shape `remove_features_ops` mass-produces.
fn descending_removes(source: &FeatureCollection, mut indices: Vec<usize>) -> Vec<FeatureOp> {
    indices.sort_unstable();
    indices.dedup();
    indices
        .into_iter()
        .rev()
        .filter_map(|index| {
            source.features.get(index).map(|feature| FeatureOp::Remove {
                index,
                feature: Box::new(feature.clone()),
            })
        })
        .collect()
}

#[test]
fn the_bulk_remove_path_agrees_with_the_sequential_oracle_on_random_sets() {
    let mut random = Lcg(0x0b5e55ed);
    for round in 0..200 {
        let count = 8 + random.below(72);
        let source = collection(count);
        let removals = 8 + random.below(count.saturating_sub(7));
        let indices: Vec<usize> = (0..removals).map(|_| random.below(count)).collect();
        let ops = descending_removes(&source, indices);
        if ops.len() < command::BULK_MIN_OPS {
            continue;
        }
        let bulk = apply_ops(&source, &ops);
        let sequential = apply_ops_sequential(&source, &ops);
        assert_eq!(bulk, sequential, "round {round} diverged");
        // The inverse (an ascending Add run) must land back on the original.
        let transaction = EditTransaction {
            layer: LayerId::new(),
            label: "bulk",
            ops,
            selection_before: None,
            selection_after: None,
            coalesce: None,
        };
        let Ok(deleted) = bulk else {
            panic!("round {round}: in-range removes must apply");
        };
        let restored = apply_ops(&deleted, &transaction.inverted().ops)
            .unwrap_or_else(|error| panic!("round {round}: the inverse must apply: {error}"));
        assert_eq!(
            restored, source,
            "round {round}: apply → invert → apply must round-trip",
        );
    }
}

#[test]
fn the_bulk_add_path_agrees_with_the_sequential_oracle() {
    let mut random = Lcg(0xadd5);
    for round in 0..200 {
        let count = 9 + random.below(60);
        let source = collection(count);
        let indices: Vec<usize> = (0..8 + random.below(count - 8))
            .map(|_| random.below(count))
            .collect();
        let removes = descending_removes(&source, indices);
        if removes.len() < command::BULK_MIN_OPS {
            continue;
        }
        let deleted = apply_ops_sequential(&source, &removes).expect("in-range removes apply");
        // The inverse of a descending-Remove run is a strictly ascending Add
        // run — exactly the bulk add shape.
        let adds: Vec<FeatureOp> = removes.iter().rev().map(FeatureOp::inverted).collect();
        let bulk = apply_ops(&deleted, &adds);
        let sequential = apply_ops_sequential(&deleted, &adds);
        assert_eq!(bulk, sequential, "round {round} diverged");
        assert_eq!(
            bulk.expect("in-range adds apply"),
            source,
            "round {round}: the adds must rebuild the original",
        );
    }
}

#[test]
fn an_out_of_range_first_index_refuses_identically_on_both_paths() {
    let source = collection(10);
    // Descending removes whose largest index is out of range for `source`.
    let oversized = collection(20);
    let ops = descending_removes(&oversized, (4..16).collect());
    assert!(ops.len() >= command::BULK_MIN_OPS);
    let bulk = apply_ops(&source, &ops);
    let sequential = apply_ops_sequential(&source, &ops);
    assert_eq!(bulk, sequential, "the error payloads must match");
    assert!(bulk.is_err(), "index 15 cannot name a feature of 10");

    // Ascending adds whose k-th index exceeds len + k.
    let adds: Vec<FeatureOp> = ops.iter().rev().map(FeatureOp::inverted).collect();
    let target = collection(3);
    let bulk = apply_ops(&target, &adds);
    let sequential = apply_ops_sequential(&target, &adds);
    assert_eq!(bulk, sequential, "the add error payloads must match");
    assert!(bulk.is_err());
}

#[test]
fn short_mixed_and_nonmonotonic_runs_take_the_sequential_path_and_agree() {
    let source = collection(24);
    // Below the threshold: 7 descending removes.
    let short = descending_removes(&source, (3..10).collect());
    assert_eq!(short.len(), 7);
    assert_eq!(
        apply_ops(&source, &short),
        apply_ops_sequential(&source, &short),
    );
    // Exactly at the threshold: 8 descending removes take the bulk path and
    // must still agree.
    let boundary = descending_removes(&source, (3..11).collect());
    assert_eq!(boundary.len(), 8);
    assert_eq!(
        apply_ops(&source, &boundary),
        apply_ops_sequential(&source, &boundary),
    );
    // ASCENDING removes: the sequential semantics remove post-shift indices,
    // which a retain over the same numbers would get wrong — the dispatcher
    // must not take the bulk path.
    let ascending: Vec<FeatureOp> = (0..10)
        .filter_map(|index| {
            source.features.get(index).map(|feature| FeatureOp::Remove {
                index,
                feature: Box::new(feature.clone()),
            })
        })
        .collect();
    assert_eq!(
        apply_ops(&source, &ascending),
        apply_ops_sequential(&source, &ascending),
    );
    // A mixed transaction stays sequential.
    let mixed = vec![
        FeatureOp::Remove {
            index: 5,
            feature: Box::new(source.features[5].clone()),
        },
        FeatureOp::Add {
            index: 0,
            feature: Box::new(source.features[0].clone()),
        },
    ];
    assert_eq!(
        apply_ops(&source, &mixed),
        apply_ops_sequential(&source, &mixed),
    );
}
