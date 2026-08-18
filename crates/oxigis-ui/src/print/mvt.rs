// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Streamed MVT vector-tile layers on the printed page.
//!
//! The shell fetches and decodes the visible tiles of the active
//! vector-tile source (the same [`crate::vector_provider::VectorTileProvider`]
//! machinery the screen uses) and hands the decoded tiles in; this module
//! draws them as **real PDF paths**, one clipped group per tile — the clip
//! is what keeps each tile's legal buffer geometry (coordinates outside
//! `0..extent`) from painting over its neighbours.
//!
//! Styling follows the same first-match-wins rule table the GPU path uses
//! ([`VectorTilePaint`]): fills as even-odd subpaths (exterior + holes in
//! one path), lines stroked, points as circles. Symbol rules are skipped —
//! MVT *labels* on the page are a follow-up, tracked in TODO.md. Each rule
//! draws under its own alpha ExtGState (`GV0`, `GV1`, …), registered by
//! [`super::pdf_document`].

use std::sync::Arc;

use oxigis_core::{LayerStyle, VectorTilePaint};
use oxigis_render::mvt::{MvtGeometry, MvtLayer, VectorTile};
use oxigis_render::{MapView, TileId};
use pdf_writer::{Content, Name};

use super::{MapBox, paint::emit_circle, to_rgb};

/// One vector-tile paint rule's ExtGState resource name: `GV0`, `GV1`, ….
pub(super) fn alpha_name(rule: usize) -> String {
    format!("GV{rule}")
}

/// The resource name one rule of one SOURCE paints under.
///
/// [`None`] is the legacy single-slot source (`PrintRequest::vector`), which
/// keeps `GV0`, `GV1`, … byte for byte; a stack entry gets its position woven
/// in, so two tilesets on one page cannot collide over `GV0`.
pub(super) fn rule_alpha_name(entry: Option<usize>, rule: usize) -> String {
    match entry {
        None => alpha_name(rule),
        Some(entry) => format!("GV{entry}s{rule}"),
    }
}

/// A rule's constant alpha, from its style's opacity.
pub(super) fn rule_alpha(paint: &VectorTilePaint) -> f32 {
    let alpha = match &paint.style {
        LayerStyle::Fill(fill) => fill.opacity(),
        LayerStyle::Line(line) => line.opacity(),
        LayerStyle::Circle(circle) => circle.opacity(),
        LayerStyle::Symbol(_) => 1.0,
    };
    alpha.clamp(0.0, 1.0)
}

/// Paints every decoded tile: one clipped, tile-aligned group per tile,
/// each rule of `paints` in order against the tile's matching layer.
///
/// `px_to_pt` converts the style's logical-pixel widths/radii to page
/// points (the same factor the labels use), so strokes keep the proportion
/// they had on screen.
pub(super) fn paint_vector_tiles(
    content: &mut Content,
    paints: &[VectorTilePaint],
    tiles: &[(TileId, Arc<VectorTile>)],
    compose: &MapView,
    map_box: &MapBox,
    px_to_pt: f32,
    entry: Option<usize>,
) {
    if paints.is_empty() || tiles.is_empty() {
        return;
    }
    // Raster pixels per page point — placement is in the compose view's
    // (raster) pixel space, the page is in points.
    let ppp = compose.size_px()[0] / map_box.width;
    if !(ppp.is_finite() && ppp > 0.0) {
        return;
    }
    for (tile, decoded) in tiles {
        // No drawable rule matches this tile (Symbol rules draw as labels,
        // not paths): skip the whole clip group rather than emit an empty
        // one.
        let draws = paints.iter().any(|paint| {
            !matches!(paint.style, LayerStyle::Symbol(_))
                && decoded
                    .layers
                    .iter()
                    .any(|layer| layer.name == paint.source_layer)
        });
        if !draws {
            continue;
        }
        let Some(frame) = TileFrame::for_tile(compose, map_box, *tile) else {
            continue;
        };
        content.save_state();
        content.rect(
            frame.left_pt,
            frame.top_pt - frame.size_pt,
            frame.size_pt,
            frame.size_pt,
        );
        content.clip_nonzero();
        content.end_path();
        for (rule, paint) in paints.iter().enumerate() {
            let Some(layer) = decoded
                .layers
                .iter()
                .find(|layer| layer.name == paint.source_layer)
            else {
                continue;
            };
            paint_rule(content, paint, entry, rule, layer, &frame, px_to_pt);
        }
        content.restore_state();
    }
}

/// One tile's page-space frame: where tile-local coordinates land. Shared
/// between the path painter and the label pass, so the two cannot drift.
pub(super) struct TileFrame {
    /// Left edge, page points.
    left_pt: f32,
    /// TOP edge, page points (PDF y grows upward; MVT v grows downward).
    top_pt: f32,
    /// Side length, page points.
    size_pt: f32,
}

impl TileFrame {
    /// The page-space frame of `tile`, or [`None`] when it does not place
    /// (off the composed view, or a degenerate page geometry).
    pub(super) fn for_tile(
        compose: &MapView,
        map_box: &super::MapBox,
        tile: oxigis_render::TileId,
    ) -> Option<Self> {
        let placement = compose.place_tile(tile);
        let ppp = compose.size_px()[0] / map_box.width;
        (placement.size > 0.0 && ppp.is_finite() && ppp > 0.0).then(|| Self {
            left_pt: map_box.x + placement.x / ppp,
            top_pt: map_box.y + map_box.height - placement.y / ppp,
            size_pt: placement.size / ppp,
        })
    }

    /// A tile-local `[u, v]` position on the `extent` grid, in page points.
    fn project(&self, position: [i32; 2], extent: u32) -> (f32, f32) {
        self.project_f64([f64::from(position[0]), f64::from(position[1])], extent)
    }

    /// A fractional tile-local position (a label anchor) in page points.
    pub(super) fn project_f64(&self, position: [f64; 2], extent: u32) -> (f32, f32) {
        let extent = extent.max(1) as f32;
        (
            self.left_pt + position[0] as f32 / extent * self.size_pt,
            self.top_pt - position[1] as f32 / extent * self.size_pt,
        )
    }
}

/// Draws one rule against one matching tile layer.
#[allow(clippy::too_many_arguments, reason = "one painter, one flat rule set")]
fn paint_rule(
    content: &mut Content,
    paint: &VectorTilePaint,
    entry: Option<usize>,
    rule: usize,
    layer: &MvtLayer,
    frame: &TileFrame,
    px_to_pt: f32,
) {
    if matches!(paint.style, LayerStyle::Symbol(_)) {
        return;
    }
    content.save_state();
    content.set_parameters(Name(rule_alpha_name(entry, rule).as_bytes()));
    match &paint.style {
        LayerStyle::Fill(fill) => {
            let rgb = to_rgb(fill.color);
            content.set_fill_rgb(rgb[0], rgb[1], rgb[2]);
            let outlined = match fill.outline_color.map(to_rgb) {
                Some(outline) => {
                    content.set_stroke_rgb(outline[0], outline[1], outline[2]);
                    content.set_line_width(0.75);
                    true
                }
                None => false,
            };
            for feature in &layer.features {
                let MvtGeometry::Polygons(polygons) = &feature.geometry else {
                    continue;
                };
                for polygon in polygons {
                    let mut any = emit_ring(content, &polygon.exterior, frame, layer.extent);
                    for interior in &polygon.interiors {
                        any |= emit_ring(content, interior, frame, layer.extent);
                    }
                    if any {
                        if outlined {
                            content.fill_even_odd_and_stroke();
                        } else {
                            content.fill_even_odd();
                        }
                    }
                }
            }
        }
        LayerStyle::Line(line) => {
            let rgb = to_rgb(line.color);
            content.set_stroke_rgb(rgb[0], rgb[1], rgb[2]);
            content.set_line_width((line.width() * px_to_pt).max(0.1));
            for feature in &layer.features {
                let MvtGeometry::Lines(lines) = &feature.geometry else {
                    continue;
                };
                for positions in lines {
                    if emit_open_path(content, positions, frame, layer.extent) {
                        content.stroke();
                    }
                }
            }
        }
        LayerStyle::Circle(circle) => {
            let rgb = to_rgb(circle.color);
            content.set_fill_rgb(rgb[0], rgb[1], rgb[2]);
            let stroked = match circle.stroke_color.map(to_rgb) {
                Some(stroke) => {
                    content.set_stroke_rgb(stroke[0], stroke[1], stroke[2]);
                    content.set_line_width((circle.stroke_width() * px_to_pt).max(0.1));
                    true
                }
                None => false,
            };
            let radius = (circle.radius() * px_to_pt).max(0.3);
            for feature in &layer.features {
                let MvtGeometry::Points(points) = &feature.geometry else {
                    continue;
                };
                for &position in points {
                    let (x, y) = frame.project(position, layer.extent);
                    emit_circle(content, x, y, radius);
                    if stroked {
                        content.fill_nonzero_and_stroke();
                    } else {
                        content.fill_nonzero();
                    }
                }
            }
        }
        LayerStyle::Symbol(_) => {}
    }
    content.restore_state();
}

/// Emits an **unclosed** MVT ring as a closed subpath. Returns whether it
/// was drawable.
fn emit_ring(
    content: &mut Content,
    positions: &[[i32; 2]],
    frame: &TileFrame,
    extent: u32,
) -> bool {
    if positions.len() < 3 {
        return false;
    }
    let mut iter = positions.iter();
    let Some(&first) = iter.next() else {
        return false;
    };
    let (x, y) = frame.project(first, extent);
    content.move_to(x, y);
    for &position in iter {
        let (x, y) = frame.project(position, extent);
        content.line_to(x, y);
    }
    content.close_path();
    true
}

/// Emits a line's positions as one open subpath. Returns whether at least
/// one segment was drawable.
fn emit_open_path(
    content: &mut Content,
    positions: &[[i32; 2]],
    frame: &TileFrame,
    extent: u32,
) -> bool {
    if positions.len() < 2 {
        return false;
    }
    let mut iter = positions.iter();
    let Some(&first) = iter.next() else {
        return false;
    };
    let (x, y) = frame.project(first, extent);
    content.move_to(x, y);
    for &position in iter {
        let (x, y) = frame.project(position, extent);
        content.line_to(x, y);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigis_core::{Color, FillStyle, LineStyle};
    use oxigis_render::LonLat;
    use oxigis_render::mvt::{MvtFeature, MvtPolygon};

    fn square_tile() -> VectorTile {
        // One polygon covering the middle half of the extent grid, plus a
        // two-point line, in two differently named layers.
        let polygon = MvtPolygon {
            exterior: vec![[1024, 1024], [3072, 1024], [3072, 3072], [1024, 3072]],
            interiors: Vec::new(),
        };
        VectorTile {
            layers: vec![
                MvtLayer {
                    name: "countries".to_string(),
                    extent: 4096,
                    features: vec![MvtFeature {
                        id: None,
                        properties: Vec::new(),
                        geometry: MvtGeometry::Polygons(vec![polygon]),
                    }],
                },
                MvtLayer {
                    name: "geolines".to_string(),
                    extent: 4096,
                    features: vec![MvtFeature {
                        id: None,
                        properties: Vec::new(),
                        geometry: MvtGeometry::Lines(vec![vec![[0, 2048], [4096, 2048]]]),
                    }],
                },
            ],
        }
    }

    fn page() -> (MapView, MapBox) {
        let options = super::super::PrintOptions::default();
        let map_box = super::super::map_box(&options);
        let out_px = super::super::raster_size_px(&map_box, &options);
        let view = MapView::new(LonLat::new(0.0, 0.0), 0.0, [512.0, 512.0]).expect("a viewport");
        let compose = super::super::compose_view(view, out_px);
        (compose, map_box)
    }

    fn ops(paints: &[VectorTilePaint], tiles: &[(TileId, Arc<VectorTile>)]) -> String {
        let (compose, map_box) = page();
        let mut content = Content::new();
        paint_vector_tiles(&mut content, paints, tiles, &compose, &map_box, 0.5, None);
        String::from_utf8_lossy(&content.finish().into_vec()).into_owned()
    }

    fn root_tile() -> (TileId, Arc<VectorTile>) {
        (TileId { z: 0, x: 0, y: 0 }, Arc::new(square_tile()))
    }

    #[test]
    fn a_matching_fill_rule_paints_an_even_odd_path_under_its_alpha_state() {
        let mut fill = FillStyle::new(Color::from_hex("3388ff").expect("hex"));
        fill.outline_color = None;
        let paints = vec![VectorTilePaint::new("countries", LayerStyle::Fill(fill))];
        let text = ops(&paints, &[root_tile()]);
        assert!(text.contains("/GV0 gs"), "the rule's alpha state applies");
        assert!(text.contains("f*"), "polygons fill even-odd");
        assert!(text.contains("W\nn"), "the tile clip is in place");
    }

    #[test]
    fn a_line_rule_strokes_and_an_unmatched_rule_stays_silent() {
        let paints = vec![
            VectorTilePaint::new(
                "geolines",
                LayerStyle::Line(LineStyle::new(Color::BLACK, 2.0)),
            ),
            VectorTilePaint::new(
                "no_such_layer",
                LayerStyle::Fill(FillStyle::new(Color::BLACK)),
            ),
        ];
        let text = ops(&paints, &[root_tile()]);
        assert!(text.contains('S'), "the line is stroked");
        assert!(
            !text.contains("/GV1 gs"),
            "no layer matched the second rule"
        );
        assert!(!text.contains("f*"), "nothing filled");
    }

    #[test]
    fn a_symbol_rule_and_an_empty_input_paint_nothing() {
        let symbol = VectorTilePaint::new(
            "countries",
            LayerStyle::Symbol(oxigis_core::SymbolStyle::new("name")),
        );
        assert_eq!(
            ops(&[symbol], &[root_tile()]),
            "",
            "symbol rules are the labels follow-up, not silent path output"
        );
        let fill =
            VectorTilePaint::new("countries", LayerStyle::Fill(FillStyle::new(Color::BLACK)));
        assert_eq!(ops(&[fill], &[]), "", "no tiles, no operators");
        assert_eq!(ops(&[], &[root_tile()]), "", "no rules, no operators");
    }

    #[test]
    fn tile_geometry_lands_inside_the_map_box() {
        let fill =
            VectorTilePaint::new("countries", LayerStyle::Fill(FillStyle::new(Color::BLACK)));
        let (compose, map_box) = page();
        let mut content = Content::new();
        paint_vector_tiles(
            &mut content,
            &[fill],
            &[root_tile()],
            &compose,
            &map_box,
            0.5,
            None,
        );
        let text = String::from_utf8_lossy(&content.finish().into_vec()).into_owned();
        // Every `m`/`l` coordinate pair must sit inside the map box (the
        // polygon is centred in the world at zoom 0 and framed by `page()`).
        let mut checked = 0;
        for line in text.lines() {
            let Some(op) = line.split_whitespace().last() else {
                continue;
            };
            if op != "m" && op != "l" {
                continue;
            }
            let numbers: Vec<f32> = line
                .split_whitespace()
                .take(2)
                .filter_map(|token| token.parse().ok())
                .collect();
            if numbers.len() != 2 {
                continue;
            }
            checked += 1;
            assert!(
                numbers[0] >= map_box.x - 0.01
                    && numbers[0] <= map_box.x + map_box.width + 0.01
                    && numbers[1] >= map_box.y - 0.01
                    && numbers[1] <= map_box.y + map_box.height + 0.01,
                "vertex ({}, {}) escaped the map box",
                numbers[0],
                numbers[1],
            );
        }
        assert!(checked >= 4, "the square's vertices were emitted");
    }
}
