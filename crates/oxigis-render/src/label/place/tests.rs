//! Tests for [`super`]: anchor derivation, the greedy collision pass and
//! the spec surface the shell fills in.

use super::{
    AnchorKind, LABEL_PADDING_PX, LabelAnchorPoint, LabelBox, LabelPlacer, LabelResolver,
    LabelSpec, LabelTable, VIEWPORT_BUFFER_PX, feature_anchor, label_text, label_text_cow,
    placed_labels, rank_value,
};
use crate::label::engine::{LabelEngine, LabelOrientation};
use crate::label::pipeline::LabelHalo;
use crate::mercator::TileId;
use crate::mvt::decode::{MvtFeature, MvtGeometry, MvtLayer, MvtPolygon, MvtValue, VectorTile};
use crate::viewport::TilePlacement;
use std::borrow::Cow;

fn engine() -> LabelEngine {
    match LabelEngine::new(oxifont_bundled::NOTO_SANS_REGULAR.to_vec()) {
        Ok(engine) => engine,
        Err(error) => panic!("engine construction failed: {error}"),
    }
}

fn placement(x: f32, y: f32, size: f32) -> TilePlacement {
    let Ok(tile) = TileId::new(0, 0, 0) else {
        panic!("root tile is valid");
    };
    TilePlacement { tile, x, y, size }
}

fn feature(name: &str, geometry: MvtGeometry) -> MvtFeature {
    MvtFeature {
        id: None,
        properties: vec![("name".to_owned(), MvtValue::String(name.to_owned()))],
        geometry,
    }
}

fn tile(features: Vec<MvtFeature>) -> VectorTile {
    VectorTile {
        layers: vec![MvtLayer {
            name: "places".to_owned(),
            extent: 4096,
            features,
        }],
    }
}

fn table() -> LabelTable {
    LabelTable::new().with("places", LabelSpec::new("name").with_size_px(12.0))
}

fn square(size: i32) -> MvtPolygon {
    MvtPolygon {
        exterior: vec![[0, 0], [size, 0], [size, size], [0, size]],
        interiors: Vec::new(),
    }
}

#[test]
fn values_render_to_text_or_nothing() {
    assert_eq!(
        label_text(&MvtValue::String("Tokyo".to_owned())),
        Some("Tokyo".to_owned())
    );
    assert_eq!(label_text(&MvtValue::I64(-42)), Some("-42".to_owned()));
    assert_eq!(label_text(&MvtValue::U64(7)), Some("7".to_owned()));
    assert_eq!(label_text(&MvtValue::F32(1.5)), Some("1.5".to_owned()));
    assert_eq!(label_text(&MvtValue::F64(-0.25)), Some("-0.25".to_owned()));
    // Booleans and non-finite numbers are not label text.
    assert_eq!(label_text(&MvtValue::Bool(true)), None);
    assert_eq!(label_text(&MvtValue::F64(f64::NAN)), None);
    assert_eq!(label_text(&MvtValue::F32(f32::INFINITY)), None);
    // Whitespace-only strings are not either.
    assert_eq!(label_text(&MvtValue::String("  \n".to_owned())), None);
    assert_eq!(label_text(&MvtValue::String(String::new())), None);
}

#[test]
fn a_point_feature_anchors_on_its_first_point() {
    let Some(anchor) = feature_anchor(&MvtGeometry::Points(vec![[10, 20], [900, 900]])) else {
        panic!("a point geometry has an anchor");
    };
    assert_eq!(anchor.position, [10.0, 20.0]);
    assert_eq!(anchor.kind, AnchorKind::Point);
    assert_eq!(anchor.priority, 0.0);
    // Multi-point labels once, not once per point.
    assert!(feature_anchor(&MvtGeometry::Points(Vec::new())).is_none());
}

#[test]
fn a_square_anchors_on_its_centre() {
    let Some(anchor) = feature_anchor(&MvtGeometry::Polygons(vec![square(100)])) else {
        panic!("a square has a centroid");
    };
    assert!((anchor.position[0] - 50.0).abs() < 1e-9);
    assert!((anchor.position[1] - 50.0).abs() < 1e-9);
    assert_eq!(anchor.kind, AnchorKind::Polygon);
    assert!((anchor.priority - 10_000.0).abs() < 1e-6);
}

#[test]
fn the_largest_ring_wins_and_holes_do_not_count() {
    let small = MvtPolygon {
        exterior: vec![[0, 0], [10, 0], [10, 10], [0, 10]],
        interiors: Vec::new(),
    };
    let large = MvtPolygon {
        exterior: vec![[100, 100], [200, 100], [200, 200], [100, 200]],
        // A hole covering nearly everything must not shrink the priority.
        interiors: vec![vec![[110, 110], [190, 110], [190, 190], [110, 190]]],
    };
    let Some(anchor) = feature_anchor(&MvtGeometry::Polygons(vec![small, large])) else {
        panic!("a polygon geometry has an anchor");
    };
    assert!((anchor.position[0] - 150.0).abs() < 1e-9);
    assert!((anchor.priority - 10_000.0).abs() < 1e-6);
}

#[test]
fn a_concave_polygon_may_centre_outside_itself() {
    // An L shape: the centroid lands in the notch. Documented v1 behaviour.
    let l_shape = MvtPolygon {
        exterior: vec![[0, 0], [100, 0], [100, 20], [20, 20], [20, 100], [0, 100]],
        interiors: Vec::new(),
    };
    let Some(anchor) = feature_anchor(&MvtGeometry::Polygons(vec![l_shape])) else {
        panic!("an L shape has a centroid");
    };
    assert!(anchor.position[0] > 0.0 && anchor.position[1] > 0.0);
    assert!((anchor.priority - 3_600.0).abs() < 1e-6);
}

#[test]
fn a_zero_area_ring_falls_back_to_the_vertex_average() {
    let sliver = MvtPolygon {
        exterior: vec![[0, 0], [10, 0], [20, 0]],
        interiors: Vec::new(),
    };
    let Some(anchor) = feature_anchor(&MvtGeometry::Polygons(vec![sliver])) else {
        panic!("a degenerate ring still anchors");
    };
    assert!((anchor.position[0] - 10.0).abs() < 1e-9);
    assert_eq!(anchor.position[1], 0.0);
    assert_eq!(anchor.priority, 0.0);
    // An empty ring has nothing to average.
    assert!(
        feature_anchor(&MvtGeometry::Polygons(vec![MvtPolygon {
            exterior: Vec::new(),
            interiors: Vec::new(),
        }]))
        .is_none()
    );
    assert!(feature_anchor(&MvtGeometry::Polygons(Vec::new())).is_none());
}

#[test]
fn a_line_anchors_on_its_arc_length_midpoint() {
    // Uneven segments: 10 across, then 90 down. Half of 100 is 50, i.e.
    // 40 into the second segment.
    let line = vec![[0, 0], [10, 0], [10, 90]];
    let Some(anchor) = feature_anchor(&MvtGeometry::Lines(vec![line])) else {
        panic!("a line has a midpoint");
    };
    assert!((anchor.position[0] - 10.0).abs() < 1e-9);
    assert!((anchor.position[1] - 40.0).abs() < 1e-9);
    assert_eq!(anchor.kind, AnchorKind::Line);
    assert!((anchor.priority - 100.0).abs() < 1e-9);
}

#[test]
fn the_longest_line_of_a_feature_wins() {
    let short = vec![[0, 0], [4, 0]];
    let long = vec![[0, 100], [100, 100]];
    let Some(anchor) = feature_anchor(&MvtGeometry::Lines(vec![short, long])) else {
        panic!("a multi-line has an anchor");
    };
    assert!((anchor.position[0] - 50.0).abs() < 1e-9);
    assert!((anchor.position[1] - 100.0).abs() < 1e-9);
    assert!((anchor.priority - 100.0).abs() < 1e-9);
}

#[test]
fn degenerate_and_empty_lines_are_handled() {
    // Zero length: the first vertex, priority zero.
    let Some(anchor) = feature_anchor(&MvtGeometry::Lines(vec![vec![[7, 8], [7, 8]]])) else {
        panic!("a zero-length line still anchors");
    };
    assert_eq!(anchor.position, [7.0, 8.0]);
    assert_eq!(anchor.priority, 0.0);
    assert!(feature_anchor(&MvtGeometry::Lines(Vec::new())).is_none());
    assert!(feature_anchor(&MvtGeometry::Lines(vec![Vec::new()])).is_none());
}

#[test]
fn a_table_resolves_by_name_first_match_wins() {
    let mut table = LabelTable::new();
    assert!(table.is_empty());
    table.push("places", LabelSpec::new("name"));
    table.push("places", LabelSpec::new("other"));
    assert_eq!(table.len(), 2);
    assert_eq!(table.entries().len(), 2);
    assert_eq!(
        table.label_for("places").map(|spec| spec.text_property),
        Some("name".to_owned())
    );
    assert!(table.label_for("roads").is_none());

    let by_ref: &LabelTable = &table;
    assert!(LabelResolver::label_for(&by_ref, "places").is_some());
    let dynamic: &dyn LabelResolver = &table;
    assert!(dynamic.label_for("places").is_some());

    table.clear();
    assert!(table.label_for("places").is_none());

    let collected: LabelTable = vec![("poi".to_owned(), LabelSpec::new("name"))]
        .into_iter()
        .collect();
    assert_eq!(collected.len(), 1);
}

#[test]
fn a_spec_validates_its_size_and_reports_halo_padding() {
    assert!(LabelSpec::new("name").has_usable_size());
    assert!(!LabelSpec::new("name").with_size_px(0.0).has_usable_size());
    assert!(!LabelSpec::new("name").with_size_px(-1.0).has_usable_size());
    assert!(
        !LabelSpec::new("name")
            .with_size_px(f32::NAN)
            .has_usable_size()
    );
    assert!(
        !LabelSpec::new("name")
            .with_size_px(100_000.0)
            .has_usable_size()
    );
    assert_eq!(LabelSpec::new("name").halo_padding_px(), 0.0);
    assert_eq!(
        LabelSpec::new("name")
            .with_halo(LabelHalo::new([255, 255, 255, 255], 3.0))
            .halo_padding_px(),
        3.0
    );
    assert_eq!(
        LabelSpec::new("name")
            .with_halo(LabelHalo::new([255, 255, 255, 255], f32::NAN))
            .halo_padding_px(),
        0.0
    );
    let spec = LabelSpec::new("name").with_color([1, 2, 3, 4]);
    assert_eq!(spec.color, [1, 2, 3, 4]);
}

#[test]
fn boxes_overlap_only_when_they_share_area() {
    let left = LabelBox {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 10.0,
        max_y: 10.0,
    };
    let touching = LabelBox {
        min_x: 10.0,
        min_y: 0.0,
        max_x: 20.0,
        max_y: 10.0,
    };
    let overlapping = LabelBox {
        min_x: 9.0,
        min_y: 9.0,
        max_x: 20.0,
        max_y: 20.0,
    };
    assert!(!left.intersects(&touching));
    assert!(left.intersects(&overlapping));
    assert!(left.is_inside([10.0, 10.0]));
    assert!(!left.is_inside([9.0, 10.0]));
    assert!(left.is_finite());
}

#[test]
fn a_box_only_has_to_touch_the_viewport_to_be_placeable() {
    let straddling = LabelBox {
        min_x: -8.0,
        min_y: 4.0,
        max_x: 2.0,
        max_y: 12.0,
    };
    // The case `is_inside` refuses and this one accepts: half on screen.
    assert!(!straddling.is_inside([100.0, 100.0]));
    assert!(straddling.is_inside_buffered([100.0, 100.0], 0.0));
    // Wholly off to the left, and touching without overlapping: neither.
    let outside = LabelBox {
        max_x: -0.5,
        ..straddling
    };
    let touching = LabelBox {
        max_x: 0.0,
        ..straddling
    };
    assert!(!outside.is_inside_buffered([100.0, 100.0], 0.0));
    assert!(!touching.is_inside_buffered([100.0, 100.0], 0.0));
    // A buffer brings both back, which is what a persistence phase would use.
    assert!(outside.is_inside_buffered([100.0, 100.0], 8.0));
    // A label larger than the window can be drawn at all, which containment
    // made impossible.
    let huge = LabelBox {
        min_x: -50.0,
        min_y: -50.0,
        max_x: 150.0,
        max_y: 150.0,
    };
    assert!(huge.is_inside_buffered([100.0, 100.0], 0.0));
    assert!(!huge.is_inside([100.0, 100.0]));
    // NaN never places, on either side of the comparison.
    let broken = LabelBox {
        min_x: f32::NAN,
        ..straddling
    };
    assert!(!broken.is_inside_buffered([100.0, 100.0], 0.0));
    assert!(!straddling.is_inside_buffered([f32::NAN, 100.0], 0.0));
    assert!(!straddling.is_inside_buffered([100.0, 100.0], f32::NAN));
}

#[test]
fn an_anchor_point_moves_the_box_off_the_feature() {
    let size = [40.0_f32, 10.0_f32];
    assert_eq!(LabelAnchorPoint::Center.origin_shift(size), [-20.0, -5.0]);
    assert_eq!(LabelAnchorPoint::default(), LabelAnchorPoint::Center);
    assert_eq!(LabelAnchorPoint::TopLeft.origin_shift(size), [0.0, 0.0]);
    assert_eq!(LabelAnchorPoint::Top.origin_shift(size), [-20.0, 0.0]);
    assert_eq!(LabelAnchorPoint::TopRight.origin_shift(size), [-40.0, 0.0]);
    assert_eq!(LabelAnchorPoint::Left.origin_shift(size), [0.0, -5.0]);
    assert_eq!(LabelAnchorPoint::Right.origin_shift(size), [-40.0, -5.0]);
    assert_eq!(
        LabelAnchorPoint::BottomLeft.origin_shift(size),
        [0.0, -10.0]
    );
    assert_eq!(LabelAnchorPoint::Bottom.origin_shift(size), [-20.0, -10.0]);
    assert_eq!(
        LabelAnchorPoint::BottomRight.origin_shift(size),
        [-40.0, -10.0]
    );
}

/// The anchor and offset fields move a placed label by exactly what they
/// say, and leave the default spec byte-identical to the centred one.
#[test]
fn anchoring_and_offsetting_move_the_placed_box() {
    let features = vec![feature("Here", MvtGeometry::Points(vec![[2048, 2048]]))];
    let place = |spec: LabelSpec| {
        let mut engine = engine();
        let mut placer = LabelPlacer::new([512.0, 512.0]);
        let table = LabelTable::new().with("places", spec);
        let Ok(()) = placer.place_tile(
            &mut engine,
            &tile(features.clone()),
            &placement(0.0, 0.0, 512.0),
            &table,
        ) else {
            panic!("placement succeeds");
        };
        let Some(label) = placer.labels().first() else {
            panic!("one label was placed");
        };
        (label.origin_px, label.shaped.size_px())
    };

    let base = LabelSpec::new("name").with_size_px(12.0);
    let (centred, [width, height]) = place(base.clone());
    // The anchor projects to (256, 256); the centred box is the old rule.
    assert_eq!(
        centred,
        [
            (256.0 - width / 2.0).round(),
            (256.0 - height / 2.0).round()
        ]
    );

    let (top_left, _) = place(base.clone().with_anchor(LabelAnchorPoint::TopLeft));
    assert_eq!(top_left, [256.0, 256.0], "the box hangs off the anchor");

    // The map styling case: the label sits below its marker, clear of it.
    let (below, _) = place(
        base.with_anchor(LabelAnchorPoint::Top)
            .with_offset_px([0.0, 6.0]),
    );
    assert_eq!(below, [(256.0 - width / 2.0).round(), 262.0]);
}

#[test]
fn a_rank_property_reorders_a_layer_that_geometry_cannot() {
    // Two points close enough to collide. Every point anchor has priority
    // 0.0, so without a rank the tile's feature order decides — and the
    // hamlet listed first wins.
    let features = vec![
        MvtFeature {
            id: None,
            properties: vec![
                ("name".to_owned(), MvtValue::String("Hamlet".to_owned())),
                ("rank".to_owned(), MvtValue::I64(1)),
            ],
            geometry: MvtGeometry::Points(vec![[2040, 2048]]),
        },
        MvtFeature {
            id: None,
            properties: vec![
                ("name".to_owned(), MvtValue::String("Capital".to_owned())),
                ("rank".to_owned(), MvtValue::I64(9)),
            ],
            geometry: MvtGeometry::Points(vec![[2056, 2048]]),
        },
    ];
    let winner = |spec: LabelSpec| {
        let mut engine = engine();
        let mut placer = LabelPlacer::new([512.0, 512.0]);
        let table = LabelTable::new().with("places", spec);
        let Ok(()) = placer.place_tile(
            &mut engine,
            &tile(features.clone()),
            &placement(0.0, 0.0, 512.0),
            &table,
        ) else {
            panic!("placement succeeds");
        };
        assert_eq!(placer.len(), 1, "the two labels must collide");
        let Some(label) = placer.labels().first() else {
            panic!("one label was placed");
        };
        label.shaped.glyphs().len()
    };
    // Identity by shape: the two names differ in glyph count, so the count
    // says which one survived without the label carrying its text.
    let glyphs_of = |text: &str| {
        let mut engine = engine();
        let Ok(shaped) = engine.shape(text, 12.0) else {
            panic!("shaping succeeds");
        };
        shaped.glyphs().len()
    };
    assert_ne!(glyphs_of("Hamlet"), glyphs_of("Capital"));
    let unranked = LabelSpec::new("name").with_size_px(12.0);
    assert_eq!(winner(unranked.clone()), glyphs_of("Hamlet"));
    assert_eq!(
        winner(unranked.with_rank_property("rank")),
        glyphs_of("Capital"),
        "the ranked feature places first however the tile lists it",
    );
}

#[test]
fn only_finite_numbers_are_ranks() {
    assert_eq!(rank_value(&MvtValue::I64(-3)), Some(-3.0));
    assert_eq!(rank_value(&MvtValue::U64(7)), Some(7.0));
    assert_eq!(rank_value(&MvtValue::F32(1.5)), Some(1.5));
    assert_eq!(rank_value(&MvtValue::F64(-0.25)), Some(-0.25));
    assert_eq!(rank_value(&MvtValue::F64(f64::NAN)), None);
    assert_eq!(rank_value(&MvtValue::F32(f32::INFINITY)), None);
    assert_eq!(rank_value(&MvtValue::Bool(true)), None);
    assert_eq!(rank_value(&MvtValue::String("9".to_owned())), None);
}

#[test]
fn string_label_text_is_borrowed_and_numbers_are_formatted() {
    let string = MvtValue::String("Tokyo".to_owned());
    assert!(matches!(
        label_text_cow(&string),
        Some(Cow::Borrowed("Tokyo"))
    ));
    assert!(matches!(
        label_text_cow(&MvtValue::I64(-42)),
        Some(Cow::Owned(_))
    ));
    // The owning form is exactly the borrowing one, materialised.
    for value in [
        MvtValue::String("Kyoto".to_owned()),
        MvtValue::I64(-42),
        MvtValue::F64(0.5),
        MvtValue::Bool(false),
        MvtValue::String("  ".to_owned()),
    ] {
        assert_eq!(
            label_text(&value),
            label_text_cow(&value).map(Cow::into_owned),
        );
    }
}

#[test]
fn a_table_hands_the_placer_a_borrowed_spec() {
    let table = LabelTable::new().with("places", LabelSpec::new("name"));
    assert!(matches!(table.label_spec("places"), Some(Cow::Borrowed(_))));
    assert!(table.label_spec("roads").is_none());
    // Through a reference and through the trait object, both of which the
    // placer's signature produces.
    let by_ref: &LabelTable = &table;
    assert!(matches!(
        LabelResolver::label_spec(&by_ref, "places"),
        Some(Cow::Borrowed(_)),
    ));
    let dynamic: &dyn LabelResolver = &table;
    assert!(matches!(
        dynamic.label_spec("places"),
        Some(Cow::Borrowed(_))
    ));
    // And the owning form still answers the same thing.
    assert_eq!(
        table.label_for("places").as_ref(),
        table.label_spec("places").as_deref()
    );
}

#[test]
fn a_higher_priority_candidate_wins_a_collision() {
    let mut engine = engine();
    // Two squares centred a few pixels apart: the big one places, the small
    // one collides with it.
    let big = MvtPolygon {
        exterior: vec![[0, 0], [2000, 0], [2000, 2000], [0, 2000]],
        interiors: Vec::new(),
    };
    let small = MvtPolygon {
        exterior: vec![[990, 990], [1010, 990], [1010, 1010], [990, 1010]],
        interiors: Vec::new(),
    };
    let tile = tile(vec![
        // Feature order puts the small one first: priority must reorder.
        feature("Small", MvtGeometry::Polygons(vec![small])),
        feature("Big", MvtGeometry::Polygons(vec![big])),
    ]);
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(&mut engine, &tile, &placement(0.0, 0.0, 512.0), &table())
    else {
        panic!("placement succeeds");
    };
    assert_eq!(placer.len(), 1);
    assert_eq!(placer.considered(), 2);
    assert_eq!(placer.labels().len(), 1);
    assert_eq!(placer.boxes().len(), 1);
    assert!(!placer.is_stale());
    assert!(placer.generation().is_some());
    assert_eq!(placer.viewport_px(), [512.0, 512.0]);
}

#[test]
fn far_apart_candidates_are_all_accepted_and_disjoint() {
    let mut engine = engine();
    let tile = tile(vec![
        feature("A", MvtGeometry::Points(vec![[200, 200]])),
        feature("B", MvtGeometry::Points(vec![[3800, 200]])),
        feature("C", MvtGeometry::Points(vec![[2048, 3800]])),
    ]);
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(&mut engine, &tile, &placement(0.0, 0.0, 512.0), &table())
    else {
        panic!("placement succeeds");
    };
    assert_eq!(placer.len(), 3);
    let boxes = placer.boxes().to_vec();
    for (index, first) in boxes.iter().enumerate() {
        for second in &boxes[index + 1..] {
            assert!(!first.intersects(second));
        }
    }
    // Origins are whole pixels.
    for label in placer.labels() {
        assert_eq!(label.origin_px[0], label.origin_px[0].round());
        assert_eq!(label.origin_px[1], label.origin_px[1].round());
    }
    let owned = placer.finish();
    let borrowed = placed_labels(&owned);
    assert_eq!(borrowed.len(), 3);
    assert_eq!(borrowed[0].origin_px, owned[0].origin_px);
}

/// Two labels far enough apart to coexist without haloes, but not with a
/// halo wide enough to close the gap.
#[test]
fn halo_padding_widens_the_collision_box() {
    let features = vec![
        feature("A", MvtGeometry::Points(vec![[1400, 2048]])),
        feature("B", MvtGeometry::Points(vec![[2700, 2048]])),
    ];
    let mut engine = engine();

    let mut bare = LabelPlacer::new([512.0, 512.0]);
    let bare_table = LabelTable::new().with("places", LabelSpec::new("name").with_size_px(12.0));
    let Ok(()) = bare.place_tile(
        &mut engine,
        &tile(features.clone()),
        &placement(0.0, 0.0, 512.0),
        &bare_table,
    ) else {
        panic!("placement succeeds");
    };
    assert_eq!(bare.len(), 2);
    let gap = bare.boxes()[1].min_x - bare.boxes()[0].max_x;
    assert!(gap > 0.0, "the bare boxes must not already touch");

    // A halo wider than half the gap closes it on both sides.
    let halo_width = gap / 2.0 + 1.0;
    let halo_table = LabelTable::new().with(
        "places",
        LabelSpec::new("name")
            .with_size_px(12.0)
            .with_halo(LabelHalo::new([255, 255, 255, 255], halo_width)),
    );
    let mut haloed = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = haloed.place_tile(
        &mut engine,
        &tile(features),
        &placement(0.0, 0.0, 512.0),
        &halo_table,
    ) else {
        panic!("placement succeeds");
    };
    assert_eq!(haloed.len(), 1);
    assert_eq!(haloed.considered(), 2);
}

/// The pass keeps a label the viewport merely *touches* — the scissor draws
/// its visible half — where it used to require containment and leave the
/// map's whole border unlabelled.
#[test]
fn a_label_clipped_by_the_viewport_edge_is_still_placed() {
    let mut engine = engine();
    // Anchored on the tile's left edge: half the box is off-screen.
    let tile = tile(vec![feature("Edge", MvtGeometry::Points(vec![[0, 2048]]))]);
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(&mut engine, &tile, &placement(0.0, 0.0, 512.0), &table())
    else {
        panic!("placement succeeds");
    };
    assert_eq!(placer.len(), 1);
    assert_eq!(placer.considered(), 1);
    let Some(placed) = placer.boxes().first() else {
        panic!("one box was accepted");
    };
    assert!(placed.min_x < 0.0, "the box really does straddle the edge");
    assert!(!placed.is_inside([512.0, 512.0]));

    // The same anchor on a tile placed further right is wholly visible.
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(&mut engine, &tile, &placement(200.0, 0.0, 200.0), &table())
    else {
        panic!("placement succeeds");
    };
    assert_eq!(placer.len(), 1);
}

/// The pre-shaping cull: a tile far larger than the window — one local
/// dataset is exactly that — must not shape the features it holds
/// off-screen. `considered` counts what reached the shaper, so a zero there
/// is the assertion.
#[test]
fn anchors_that_cannot_reach_the_viewport_are_never_shaped() {
    let mut engine = engine();
    // 100 000 px for 4096 tile units: tile unit 4000 lands ~97 656 px out.
    let far = tile(vec![feature(
        "Far",
        MvtGeometry::Points(vec![[4000, 4000]]),
    )]);
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(&mut engine, &far, &placement(0.0, 0.0, 100_000.0), &table())
    else {
        panic!("placement succeeds");
    };
    assert_eq!(placer.considered(), 0, "nothing off-screen reached shaping");
    assert!(placer.is_empty());

    // A feature of the same oversized tile that IS on screen still places.
    let near = tile(vec![
        feature("Far", MvtGeometry::Points(vec![[4000, 4000]])),
        feature("Near", MvtGeometry::Points(vec![[10, 10]])),
    ]);
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(
        &mut engine,
        &near,
        &placement(0.0, 0.0, 100_000.0),
        &table(),
    ) else {
        panic!("placement succeeds");
    };
    assert_eq!(placer.considered(), 1);
    assert_eq!(placer.len(), 1);
}

/// The cull's safety property, swept rather than argued: across a range of
/// anchor positions running well past both edges, a label is placed at
/// exactly the positions whose box touches the viewport. A cull that was
/// even one pixel too eager would drop one of them.
#[test]
fn the_cull_drops_only_what_the_collision_test_would_have_dropped() {
    let mut engine = engine();
    let Ok(shaped) = engine.shape("Edge", 12.0) else {
        panic!("shaping succeeds");
    };
    let [width, height] = shaped.size_px();
    assert!(width > 0.0 && height > 0.0);
    let features = vec![feature("Edge", MvtGeometry::Points(vec![[2048, 2048]]))];

    for step in -30..30 {
        let offset = step as f32 * 37.0;
        let mut placer = LabelPlacer::new([512.0, 512.0]);
        let Ok(()) = placer.place_tile(
            &mut engine,
            &tile(features.clone()),
            &placement(offset, 0.0, 512.0),
            &table(),
        ) else {
            panic!("placement succeeds");
        };
        // The same arithmetic the pass runs, spelled out.
        let origin_x = (offset + 256.0 - width / 2.0).round();
        let touches = origin_x + width + LABEL_PADDING_PX > -VIEWPORT_BUFFER_PX
            && origin_x - LABEL_PADDING_PX < 512.0 + VIEWPORT_BUFFER_PX;
        assert_eq!(placer.len(), usize::from(touches), "offset {offset}");
    }
}

#[test]
fn buffer_zone_anchors_belong_to_the_neighbouring_tile() {
    let mut engine = engine();
    let tile = tile(vec![
        feature("Inside", MvtGeometry::Points(vec![[2048, 2048]])),
        feature("Spill right", MvtGeometry::Points(vec![[4200, 2048]])),
        feature("Spill left", MvtGeometry::Points(vec![[-100, 2048]])),
        feature("Spill up", MvtGeometry::Points(vec![[2048, -8]])),
        feature("Spill down", MvtGeometry::Points(vec![[2048, 4097]])),
    ]);
    let mut placer = LabelPlacer::new([1024.0, 1024.0]);
    let Ok(()) = placer.place_tile(&mut engine, &tile, &placement(0.0, 0.0, 512.0), &table())
    else {
        panic!("placement succeeds");
    };
    assert_eq!(placer.considered(), 1);
    assert_eq!(placer.len(), 1);
}

#[test]
fn collision_accumulates_across_tiles() {
    let mut engine = engine();
    // Two tiles side by side, each labelling a point right at the seam.
    let left = tile(vec![feature(
        "Seam",
        MvtGeometry::Points(vec![[4000, 2048]]),
    )]);
    let right = tile(vec![feature("Seam", MvtGeometry::Points(vec![[96, 2048]]))]);
    let mut placer = LabelPlacer::new([1024.0, 512.0]);
    let table = table();
    let Ok(()) = placer.place_tile(&mut engine, &left, &placement(0.0, 0.0, 512.0), &table) else {
        panic!("placement succeeds");
    };
    assert_eq!(placer.len(), 1);
    let Ok(()) = placer.place_tile(&mut engine, &right, &placement(512.0, 0.0, 512.0), &table)
    else {
        panic!("placement succeeds");
    };
    assert_eq!(placer.considered(), 2);
    assert_eq!(placer.len(), 1, "the seam-adjacent twin must be rejected");
}

#[test]
fn unlabelled_layers_and_unusable_specs_place_nothing() {
    let mut engine = engine();
    let tile = tile(vec![feature(
        "Here",
        MvtGeometry::Points(vec![[2048, 2048]]),
    )]);
    let view = placement(0.0, 0.0, 512.0);

    // No spec for the layer.
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(&mut engine, &tile, &view, &LabelTable::new()) else {
        panic!("placement succeeds");
    };
    assert!(placer.is_empty());

    // A spec with an impossible size.
    let bad = LabelTable::new().with("places", LabelSpec::new("name").with_size_px(0.0));
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(&mut engine, &tile, &view, &bad) else {
        panic!("placement succeeds");
    };
    assert!(placer.is_empty());

    // A spec pointing at a property no feature has.
    let missing = LabelTable::new().with("places", LabelSpec::new("ref"));
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(&mut engine, &tile, &view, &missing) else {
        panic!("placement succeeds");
    };
    assert!(placer.is_empty());
}

#[test]
fn a_nonsense_placement_or_extent_places_nothing() {
    let mut engine = engine();
    let features = vec![feature("Here", MvtGeometry::Points(vec![[2048, 2048]]))];
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    for broken in [
        placement(0.0, 0.0, 0.0),
        placement(0.0, 0.0, f32::NAN),
        placement(f32::INFINITY, 0.0, 512.0),
    ] {
        let Ok(()) = placer.place_tile(&mut engine, &tile(features.clone()), &broken, &table())
        else {
            panic!("placement succeeds");
        };
    }
    assert!(placer.is_empty());

    // A zero extent cannot be scaled from.
    let zero_extent = VectorTile {
        layers: vec![MvtLayer {
            name: "places".to_owned(),
            extent: 0,
            features,
        }],
    };
    let Ok(()) = placer.place_tile(
        &mut engine,
        &zero_extent,
        &placement(0.0, 0.0, 512.0),
        &table(),
    ) else {
        panic!("placement succeeds");
    };
    assert!(placer.is_empty());
}

#[test]
fn whitespace_labels_reserve_no_space() {
    let mut engine = engine();
    let tile = tile(vec![
        MvtFeature {
            id: None,
            // U+200B is a format character, not `White_Space`: it survives
            // `label_text` and reaches the engine, which shapes it to no
            // glyphs at all. That is the `ShapedLabel::is_empty` guard.
            properties: vec![("name".to_owned(), MvtValue::String("\u{200b}".to_owned()))],
            geometry: MvtGeometry::Points(vec![[2048, 2048]]),
        },
        MvtFeature {
            id: None,
            // The `label_text` half of the same rule: dropped before
            // shaping, so it cannot reserve space either.
            properties: vec![("name".to_owned(), MvtValue::String("\u{a0}".to_owned()))],
            geometry: MvtGeometry::Points(vec![[2048, 2048]]),
        },
        feature("Real", MvtGeometry::Points(vec![[2048, 2048]])),
    ]);
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(&mut engine, &tile, &placement(0.0, 0.0, 512.0), &table())
    else {
        panic!("placement succeeds");
    };
    // The blank label neither draws nor blocks the real one at the same
    // anchor.
    assert_eq!(placer.considered(), 1);
    assert_eq!(placer.len(), 1);
}

#[test]
fn layers_are_processed_in_tile_order() {
    let mut engine = engine();
    // Same anchor in two layers: whichever layer the tile lists first wins,
    // regardless of the resolver's order.
    let tile = VectorTile {
        layers: vec![
            MvtLayer {
                name: "first".to_owned(),
                extent: 4096,
                features: vec![feature(
                    "First layer wins",
                    MvtGeometry::Points(vec![[2048, 2048]]),
                )],
            },
            MvtLayer {
                name: "second".to_owned(),
                extent: 4096,
                features: vec![feature("Second", MvtGeometry::Points(vec![[2048, 2048]]))],
            },
        ],
    };
    let table = LabelTable::new()
        .with("second", LabelSpec::new("name").with_size_px(12.0))
        .with("first", LabelSpec::new("name").with_size_px(12.0));
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(&mut engine, &tile, &placement(0.0, 0.0, 512.0), &table) else {
        panic!("placement succeeds");
    };
    assert_eq!(placer.len(), 1);
    let Some(label) = placer.labels().first() else {
        panic!("one label was placed");
    };
    // Identity by shape: the placed label must be the one the first layer
    // asked for, not the second.
    let Ok(expected) = engine.shape("First layer wins", 12.0) else {
        panic!("shaping succeeds");
    };
    assert_eq!(label.shaped.size_px(), expected.size_px());
    assert_eq!(label.shaped.glyphs().len(), expected.glyphs().len());
}

#[test]
fn the_padding_constant_is_applied_on_every_side() {
    let mut engine = engine();
    let tile = tile(vec![feature(
        "Pad",
        MvtGeometry::Points(vec![[2048, 2048]]),
    )]);
    let mut placer = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = placer.place_tile(&mut engine, &tile, &placement(0.0, 0.0, 512.0), &table())
    else {
        panic!("placement succeeds");
    };
    let Some(label) = placer.labels().first() else {
        panic!("one label was placed");
    };
    let [width, height] = label.shaped.size_px();
    assert_eq!(
        label.collision_box.min_x,
        label.origin_px[0] - LABEL_PADDING_PX
    );
    assert_eq!(
        label.collision_box.max_x,
        label.origin_px[0] + width + LABEL_PADDING_PX
    );
    assert_eq!(
        label.collision_box.min_y,
        label.origin_px[1] - LABEL_PADDING_PX
    );
    assert_eq!(
        label.collision_box.max_y,
        label.origin_px[1] + height + LABEL_PADDING_PX
    );
    // The box is centred on the projected anchor (256, 256).
    assert!((label.origin_px[0] - (256.0 - width / 2.0)).abs() <= 0.5);
    assert!((label.origin_px[1] - (256.0 - height / 2.0)).abs() <= 0.5);
}

// --- print/text v1.5 (D-A0/D-A8): orientation ---

#[test]
fn the_spec_carries_an_orientation_that_defaults_to_horizontal() {
    let spec = LabelSpec::new("name");
    assert_eq!(spec.orientation, LabelOrientation::Horizontal);
    assert!(spec.orientation.is_horizontal());
    let vertical = LabelSpec::new("name").with_orientation(LabelOrientation::Vertical);
    assert_eq!(vertical.orientation, LabelOrientation::Vertical);
    // The builder is orthogonal to every other one.
    let both = LabelSpec::new("name")
        .with_size_px(18.0)
        .with_weight(crate::label::LabelWeight::Bold)
        .with_orientation(LabelOrientation::Vertical);
    assert_eq!(both.size_px, 18.0);
    assert_eq!(both.weight, crate::label::LabelWeight::Bold);
    assert_eq!(both.orientation, LabelOrientation::Vertical);
}

#[test]
fn a_refused_vertical_spec_places_exactly_as_the_horizontal_one_does() {
    // The byte-identity floor for placement: the suite's font cannot
    // stack anything, so a vertical spec must produce the SAME boxes as
    // a horizontal one — the ladder's refusal costs the page nothing.
    let features = vec![
        feature("Tokyo", MvtGeometry::Points(vec![[1400, 2048]])),
        feature("Kyoto", MvtGeometry::Points(vec![[2700, 2048]])),
    ];
    let mut engine = engine();
    let mut horizontal = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = horizontal.place_tile(
        &mut engine,
        &tile(features.clone()),
        &placement(0.0, 0.0, 512.0),
        &table(),
    ) else {
        panic!("placement succeeds");
    };
    let vertical_table = LabelTable::new().with(
        "places",
        LabelSpec::new("name")
            .with_size_px(12.0)
            .with_orientation(LabelOrientation::Vertical),
    );
    let mut vertical = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = vertical.place_tile(
        &mut engine,
        &tile(features),
        &placement(0.0, 0.0, 512.0),
        &vertical_table,
    ) else {
        panic!("placement succeeds");
    };
    assert_eq!(vertical.len(), horizontal.len());
    assert_eq!(vertical.boxes(), horizontal.boxes());
}

/// **The test that proves placement, collision and halo needed no
/// change.** A vertical label is a `ShapedLabel` whose box is tall; the
/// greedy pass, the AABB and the halo padding read that box and nothing
/// else, so all three behave correctly without a line of new code.
#[test]
#[ignore = "reads C:/Windows/Fonts; the suite ships no font with a vmtx table"]
fn live_windows_a_vertical_label_collides_by_its_tall_box() {
    let Some(bytes) = ["meiryo.ttc", "YuGothM.ttc", "msgothic.ttc"]
        .into_iter()
        .find_map(|name| std::fs::read(format!("C:/Windows/Fonts/{name}")).ok())
    else {
        return;
    };
    let mut engine = match LabelEngine::new(bytes) {
        Ok(engine) => engine,
        Err(error) => panic!("a Windows CJK face parses: {error}"),
    };
    let name = "\u{6771}\u{4EAC}\u{90FD}";
    let spec = LabelSpec::new("name")
        .with_size_px(16.0)
        .with_orientation(LabelOrientation::Vertical);
    let vertical_table = LabelTable::new().with("places", spec.clone());

    // The box really is tall and narrow.
    let mut one = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = one.place_tile(
        &mut engine,
        &tile(vec![feature(name, MvtGeometry::Points(vec![[2048, 2048]]))]),
        &placement(0.0, 0.0, 512.0),
        &vertical_table,
    ) else {
        panic!("placement succeeds");
    };
    assert_eq!(one.len(), 1);
    let solo = one.boxes()[0];
    let width = solo.max_x - solo.min_x;
    let height = solo.max_y - solo.min_y;
    assert!(
        height > width,
        "a vertical box must be tall: {width}x{height}"
    );

    // Two labels the SAME distance apart collide down a column and not
    // across a row — the opposite of the horizontal case, and it falls
    // out of the box shape alone. 200 tile units is 25 px at this
    // placement: less than the column's ~48 px height, more than its
    // 16 px width plus padding.
    let column = vec![
        feature(name, MvtGeometry::Points(vec![[2048, 1948]])),
        feature(name, MvtGeometry::Points(vec![[2048, 2148]])),
    ];
    let row = vec![
        feature(name, MvtGeometry::Points(vec![[1948, 2048]])),
        feature(name, MvtGeometry::Points(vec![[2148, 2048]])),
    ];
    let mut stacked = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = stacked.place_tile(
        &mut engine,
        &tile(column),
        &placement(0.0, 0.0, 512.0),
        &vertical_table,
    ) else {
        panic!("placement succeeds");
    };
    let mut abreast = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = abreast.place_tile(
        &mut engine,
        &tile(row),
        &placement(0.0, 0.0, 512.0),
        &vertical_table,
    ) else {
        panic!("placement succeeds");
    };
    assert_eq!(stacked.len(), 1, "one column, one survivor");
    assert_eq!(abreast.len(), 2, "side by side, both survive");
    assert_eq!(stacked.considered(), 2);

    // Halo padding widens BOTH axes, through the same one expression.
    let haloed_table = LabelTable::new().with(
        "places",
        spec.with_halo(LabelHalo::new([255, 255, 255, 255], 4.0)),
    );
    let mut haloed = LabelPlacer::new([512.0, 512.0]);
    let Ok(()) = haloed.place_tile(
        &mut engine,
        &tile(vec![feature(name, MvtGeometry::Points(vec![[2048, 2048]]))]),
        &placement(0.0, 0.0, 512.0),
        &haloed_table,
    ) else {
        panic!("placement succeeds");
    };
    let wide = haloed.boxes()[0];
    assert!((wide.max_x - wide.min_x) - width - 8.0 < 0.001);
    assert!((wide.max_y - wide.min_y) - height - 8.0 < 0.001);
    // Both axes grew by the halo alone; the base padding is unchanged.
    assert!((wide.min_x - solo.min_x + 4.0).abs() < 0.001);
    assert!((wide.min_y - solo.min_y + 4.0).abs() < 0.001);

    // And a HALOED vertical label draws exactly `9 x glyphs` quads, like
    // every horizontal one: eight halo copies plus the fill, unchanged.
    let owned = haloed.finish();
    let placed = placed_labels(&owned);
    let glyphs: usize = placed.iter().map(|label| label.shaped.glyphs().len()).sum();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let atlas_size = engine.atlas().size();
    let Ok(()) = crate::label::build_label_quads(
        &placed,
        atlas_size,
        [512.0, 512.0],
        &mut vertices,
        &mut indices,
    ) else {
        panic!("quads build");
    };
    assert_eq!(vertices.len(), glyphs * 9 * 4, "9 quads of 4 vertices each");
    assert_eq!(indices.len(), glyphs * 9 * 6, "and 6 indices per quad");
}
