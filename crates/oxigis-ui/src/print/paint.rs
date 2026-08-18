// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The local-layer geometry painters — a PURE MOVE out of `print/mod.rs`
//! (print v1.6), which reached its line budget as the raster and label work
//! landed. Nothing here changed in the move: the operator sequence one layer
//! emits is byte for byte what it was.

use oxigeo::geojson::types::{Geometry, Position};
use oxigis_core::LayerStyle;
use oxigis_render::MapView;
use pdf_writer::{Content, Name};

use super::{MapBox, PrintLayer, class_alpha_name, class_buckets, path_family, project, to_rgb};
use crate::edit::command::{self, PathKind};

/// Paints one local layer as vector operators — one pass per PRESENT
/// geometry family **and class bucket**, each with that bucket's resolved
/// style and its own alpha state, so the page paints exactly what the
/// screen's own resolution draws.
///
/// [`LayerStyle::Symbol`] slots are skipped — labels go through the label
/// pass — and a `GeometryCollection`'s members are walked recursively so a
/// mixed feature still prints. For a no-override set the three filtered
/// passes emit the SAME operator sequence as one unfiltered pass, merely
/// regrouped by family (family and `PathKind` are the same partition), and
/// for a single-family layer the regrouping is the identity — the
/// byte-for-byte guarantee. Today's quirk that a `Fill` style stroke-draws
/// line leaves is preserved rather than quietly fixed (the Line-family
/// pass of a fill-based set takes that arm).
///
/// # The class dimension (thematic v1.6)
///
/// An unclassified layer has exactly one bucket — the fallback — whose
/// resolved style *is* `effective(family)` and whose name *is* the slot's
/// own, so its output is unchanged to the byte. A classified layer paints
/// the fallback first and then one bucket per class, which is the order the
/// map's own partition emits its tile layers in
/// ([`crate::local_vector::feature_collection_to_tile_with`]) — the page
/// cannot show a different class on top from the screen.
pub(super) fn paint_layer(
    content: &mut Content,
    layer: &PrintLayer,
    index: usize,
    compose: &MapView,
    map_box: &MapBox,
) {
    for family in layer.families.iter() {
        for class in class_buckets(&layer.style) {
            paint_layer_bucket(content, layer, index, family, class, compose, map_box);
        }
    }
}

/// One (family, class) pass of [`paint_layer`].
fn paint_layer_bucket(
    content: &mut Content,
    layer: &PrintLayer,
    index: usize,
    family: oxigis_core::GeometryFamily,
    class: Option<usize>,
    compose: &MapView,
    map_box: &MapBox,
) {
    // Owned, not borrowed: a class style is COMPOSED over its family's (a
    // `Fill` class on a point family becomes a recoloured circle rather than a
    // fill that would draw nothing), so the resolved value exists nowhere in
    // the model to borrow from.
    let style = layer.style.style_for_class(family, class);
    if matches!(style, LayerStyle::Symbol(_)) {
        return;
    }
    content.save_state();
    content.set_parameters(Name(
        class_alpha_name(index, layer.style.slot_of(family), class).as_bytes(),
    ));
    match &style {
        LayerStyle::Fill(fill) => {
            let rgb = to_rgb(fill.color);
            content.set_fill_rgb(rgb[0], rgb[1], rgb[2]);
            let outline = fill.outline_color.map(to_rgb);
            if let Some(outline) = outline {
                content.set_stroke_rgb(outline[0], outline[1], outline[2]);
                content.set_line_width(0.75);
            }
            for geometry in bucket_geometries(layer, class) {
                paint_fill_geometry(
                    content,
                    geometry,
                    outline.is_some(),
                    family,
                    compose,
                    map_box,
                );
            }
        }
        LayerStyle::Line(line) => {
            let rgb = to_rgb(line.color);
            content.set_stroke_rgb(rgb[0], rgb[1], rgb[2]);
            content.set_line_width(line.width().max(0.1));
            for geometry in bucket_geometries(layer, class) {
                paint_stroke_geometry(content, geometry, family, compose, map_box);
            }
        }
        LayerStyle::Circle(circle) => {
            let rgb = to_rgb(circle.color);
            content.set_fill_rgb(rgb[0], rgb[1], rgb[2]);
            let stroked = match circle.stroke_color {
                Some(stroke) => {
                    let rgb = to_rgb(stroke);
                    content.set_stroke_rgb(rgb[0], rgb[1], rgb[2]);
                    content.set_line_width(circle.stroke_width().max(0.1));
                    true
                }
                None => false,
            };
            let radius = circle.radius().max(0.5);
            for geometry in bucket_geometries(layer, class) {
                paint_point_geometry(content, geometry, radius, stroked, family, compose, map_box);
            }
        }
        LayerStyle::Symbol(_) => {}
    }
    content.restore_state();
}

/// The geometries of `layer` that fall in `class`, in source order.
///
/// A feature is classified ONCE, from its own properties, exactly as the map's
/// partition classifies it — so a mixed `GeometryCollection` cannot be half in
/// one class and half in another, on the page any more than on screen. An
/// unclassified layer takes the `is_single` shortcut and walks the collection
/// without touching a single property map, which is what keeps a
/// million-feature export at its pre-v1.6 cost.
fn bucket_geometries(
    layer: &PrintLayer,
    class: Option<usize>,
) -> impl Iterator<Item = &oxigeo::geojson::types::Geometry> {
    let renderer = layer.style.renderer();
    let single = renderer.is_single();
    layer.features.features.iter().filter_map(move |feature| {
        let geometry = feature.geometry.as_ref()?;
        if single {
            return Some(geometry);
        }
        let found = feature
            .properties
            .as_ref()
            .and_then(|properties| renderer.class_of(properties));
        (found == class).then_some(geometry)
    })
}

/// Walks `geometry` — recursing through collections — and hands every leaf to
/// `paint`.
fn for_each_leaf(geometry: &Geometry, paint: &mut dyn FnMut(&Geometry)) {
    for_each_leaf_bounded(geometry, 0, paint);
}

/// [`for_each_leaf`]'s body, bounded at the editor's shared
/// [`command::MAX_GEOMETRY_DEPTH`] so a maliciously nested collection cannot
/// recurse the printer either.
fn for_each_leaf_bounded(geometry: &Geometry, depth: usize, paint: &mut dyn FnMut(&Geometry)) {
    match geometry {
        Geometry::GeometryCollection(collection) => {
            if depth >= command::MAX_GEOMETRY_DEPTH {
                return;
            }
            for member in &collection.geometries {
                for_each_leaf_bounded(member, depth + 1, paint);
            }
        }
        _ => paint(geometry),
    }
}

/// Fill-styled geometry: rings become even-odd-filled subpaths (holes for
/// free, whatever the winding), lines are stroked, lone points skipped.
fn paint_fill_geometry(
    content: &mut Content,
    geometry: &Geometry,
    outlined: bool,
    family: oxigis_core::GeometryFamily,
    compose: &MapView,
    map_box: &MapBox,
) {
    for_each_leaf(geometry, &mut |leaf| {
        let paths = command::paths(leaf);
        let mut any_ring = false;
        for path in &paths {
            if path_family(path.kind) != family {
                continue;
            }
            match path.kind {
                PathKind::Ring => {
                    if emit_path(content, path.positions, compose, map_box) {
                        content.close_path();
                        any_ring = true;
                    }
                }
                PathKind::Line => {
                    if emit_path(content, path.positions, compose, map_box) {
                        content.stroke();
                    }
                }
                PathKind::Points => {}
            }
        }
        if any_ring {
            if outlined {
                content.fill_even_odd_and_stroke();
            } else {
                content.fill_even_odd();
            }
        }
    });
}

/// Line-styled geometry: every line and ring is stroked, points skipped.
fn paint_stroke_geometry(
    content: &mut Content,
    geometry: &Geometry,
    family: oxigis_core::GeometryFamily,
    compose: &MapView,
    map_box: &MapBox,
) {
    for_each_leaf(geometry, &mut |leaf| {
        for path in command::paths(leaf) {
            if path_family(path.kind) != family {
                continue;
            }
            match path.kind {
                PathKind::Ring => {
                    if emit_path(content, path.positions, compose, map_box) {
                        content.close_path();
                        content.stroke();
                    }
                }
                PathKind::Line => {
                    if emit_path(content, path.positions, compose, map_box) {
                        content.stroke();
                    }
                }
                PathKind::Points => {}
            }
        }
    });
}

/// Circle-styled geometry: every position of every path becomes one circle.
fn paint_point_geometry(
    content: &mut Content,
    geometry: &Geometry,
    radius: f32,
    stroked: bool,
    family: oxigis_core::GeometryFamily,
    compose: &MapView,
    map_box: &MapBox,
) {
    for_each_leaf(geometry, &mut |leaf| {
        for path in command::paths(leaf) {
            if path_family(path.kind) != family {
                continue;
            }
            for position in path.positions {
                let Some((x, y)) = project(compose, map_box, position) else {
                    continue;
                };
                emit_circle(content, x, y, radius);
                if stroked {
                    content.fill_nonzero_and_stroke();
                } else {
                    content.fill_nonzero();
                }
            }
        }
    });
}

/// Emits `positions` as one `m`/`l` subpath. Returns whether at least two
/// positions were drawable.
fn emit_path(
    content: &mut Content,
    positions: &[Position],
    compose: &MapView,
    map_box: &MapBox,
) -> bool {
    let mut projected = positions
        .iter()
        .filter_map(|position| project(compose, map_box, position));
    let Some((x, y)) = projected.next() else {
        return false;
    };
    content.move_to(x, y);
    let mut segments = 0_usize;
    for (x, y) in projected {
        content.line_to(x, y);
        segments += 1;
    }
    segments > 0
}

/// The Bézier circle constant: `4/3 · (√2 − 1)`.
const CIRCLE_KAPPA: f32 = 0.552_284_8;

/// Emits a circle at `(cx, cy)` as four cubic Béziers.
pub(super) fn emit_circle(content: &mut Content, cx: f32, cy: f32, radius: f32) {
    let k = CIRCLE_KAPPA * radius;
    content.move_to(cx + radius, cy);
    content.cubic_to(cx + radius, cy + k, cx + k, cy + radius, cx, cy + radius);
    content.cubic_to(cx - k, cy + radius, cx - radius, cy + k, cx - radius, cy);
    content.cubic_to(cx - radius, cy - k, cx - k, cy - radius, cx, cy - radius);
    content.cubic_to(cx + k, cy - radius, cx + radius, cy - k, cx + radius, cy);
    content.close_path();
}
