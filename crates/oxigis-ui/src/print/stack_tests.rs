// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Printing an N-layer tiled stack (compositing v1.6).
//!
//! Before this, `PrintRequest::stack` was captured and pinned by a test but
//! nothing composed it: an export of a three-layer project printed the top-most
//! raster and the top-most vector tileset and silently dropped the rest. These
//! tests pin the three statements that closing it is made of:
//!
//! 1. a project the legacy single-slot fields already describe keeps the legacy
//!    path — the same provider, the same composite, the same bytes;
//! 2. every raster entry reaches the page's image, at its own opacity, in stack
//!    order;
//! 3. every vector entry reaches the page as paths AND as font-plan strings —
//!    the second half fails silently, because an unplanned character is dropped
//!    rather than mis-drawn.

use super::tests::{opts, polygon_layer, request};
use super::*;
use crate::layer_source::TileLayerSource;
use oxigis_core::{Color, FillStyle, LayerStyle, LineStyle, SymbolStyle, VectorTilePaint};
use oxigis_render::mvt::{MvtFeature, MvtGeometry, MvtLayer, MvtPolygon, VectorTile};

/// A layer id nothing else in the page reads — the stack entries are matched
/// by position here, never by id.
fn layer_id() -> oxigis_core::LayerId {
    oxigis_core::LayerId::new()
}

/// A COG entry at `opacity`.
fn cog_entry(url: &str, opacity: f32) -> PrintTileLayer {
    PrintTileLayer {
        layer: layer_id(),
        source: TileLayerSource::Cog(CogLayerConfig::new(url)),
        opacity,
    }
}

/// A vector-tile entry painting `source_layer` in `style`.
fn vector_entry(source_layer: &str, style: LayerStyle) -> PrintTileLayer {
    let config = VectorTileConfig::new("https://example.test/{z}/{x}/{y}.pbf")
        .with_paints(vec![VectorTilePaint::new(source_layer, style)]);
    PrintTileLayer {
        layer: layer_id(),
        source: TileLayerSource::Vector(config),
        opacity: 1.0,
    }
}

/// One decoded tile holding a square in layer `name`, carrying `"<name>
/// label"` as its `name` property.
fn tile_with(name: &str) -> Arc<VectorTile> {
    tile_of(name, |_| {
        MvtGeometry::Polygons(vec![MvtPolygon {
            exterior: vec![[1024, 1024], [3072, 1024], [3072, 3072], [1024, 3072]],
            interiors: Vec::new(),
        }])
    })
}

/// [`tile_with`] carrying a labellable POINT — what a Symbol rule anchors on.
fn labelled_tile(name: &str) -> Arc<VectorTile> {
    tile_of(name, |_| MvtGeometry::Points(vec![[2048, 2048]]))
}

/// One decoded tile: layer `name`, one feature with the given geometry and a
/// `name` property the Symbol rules read.
fn tile_of(name: &str, geometry: impl Fn(&str) -> MvtGeometry) -> Arc<VectorTile> {
    Arc::new(VectorTile {
        layers: vec![MvtLayer {
            name: name.to_string(),
            extent: 4096,
            features: vec![MvtFeature {
                id: None,
                properties: vec![(
                    "name".to_string(),
                    oxigis_render::mvt::MvtValue::String(format!("{name} label")),
                )],
                geometry: geometry(name),
            }],
        }],
    })
}

/// The root tile paired with `tile`.
fn root(tile: Arc<VectorTile>) -> Vec<(TileId, Arc<VectorTile>)> {
    vec![(TileId { z: 0, x: 0, y: 0 }, tile)]
}

/// A colour as a PDF fill operator writes it.
fn rgb_op(color: Color) -> String {
    let rgb = to_rgb(color);
    format!("{} {} {}", rgb[0], rgb[1], rgb[2])
}

#[test]
fn a_stack_the_legacy_slots_describe_keeps_the_legacy_path() {
    // The compatibility gate. Everything the map could draw BEFORE compositing
    // v1.6 — one raster, one vector tileset — still exports through the exact
    // path it always did, so no working export changes.
    let (mut request, _) = request(Vec::new());
    assert!(request.stack_fits_legacy_slots(), "an empty stack");

    // The legacy fields have to NAME the entries, not merely be outnumbered by
    // them: what the single-slot path prints is `cog` / `archive` / `vector`,
    // so an entry no field names is an entry the page would lose.
    request.stack = vec![cog_entry("https://example.test/a.tif", 1.0)];
    assert!(
        !request.stack_fits_legacy_slots(),
        "one raster the fields do not name",
    );
    request.cog = Some(CogLayerConfig::new("https://example.test/a.tif"));
    assert!(request.stack_fits_legacy_slots(), "one raster");

    request.stack.push(vector_entry(
        "countries",
        LayerStyle::Fill(FillStyle::new(Color::BLACK)),
    ));
    assert!(
        !request.stack_fits_legacy_slots(),
        "one tileset the fields do not name",
    );
    request.vector = Some(VectorTileConfig::new(
        "https://example.test/{z}/{x}/{y}.pbf",
    ));
    assert!(request.stack_fits_legacy_slots(), "one raster, one vector");

    // And everything past that does not.
    let mut two_rasters = request.stack.clone();
    two_rasters.push(cog_entry("https://example.test/b.tif", 1.0));
    request.stack = two_rasters;
    assert!(!request.stack_fits_legacy_slots(), "two rasters");

    request.stack = vec![
        vector_entry("a", LayerStyle::Fill(FillStyle::new(Color::BLACK))),
        vector_entry("b", LayerStyle::Fill(FillStyle::new(Color::BLACK))),
    ];
    assert!(!request.stack_fits_legacy_slots(), "two tilesets");

    // An XYZ overlay has no legacy field at all: `cog` and `archive` cannot
    // hold one, so the single-slot path would print the map without it.
    request.stack = vec![PrintTileLayer {
        layer: layer_id(),
        source: TileLayerSource::Xyz(BasemapConfig::openstreetmap()),
        opacity: 1.0,
    }];
    assert!(!request.stack_fits_legacy_slots(), "an XYZ overlay");
}

#[test]
fn every_vector_entry_of_a_composed_stack_paints_in_its_own_colour() {
    // The divergence this closes: an export of a three-layer project used to
    // print the TOP tileset and drop the one under it.
    let lower = Color::from_hex("ff0000").expect("hex");
    let upper = Color::from_hex("00ff00").expect("hex");
    let (mut request, compose) = request(Vec::new());
    request.stack = vec![
        vector_entry("lower", LayerStyle::Fill(FillStyle::new(lower))),
        vector_entry("upper", LayerStyle::Line(LineStyle::new(upper, 2.0))),
    ];
    assert!(!request.stack_fits_legacy_slots());
    let lower_tiles = root(tile_with("lower"));
    let upper_tiles = root(tile_with("upper"));
    let stack = vec![lower_tiles, upper_tiles];
    let ops = page_content_planned_with(
        &request,
        &compose,
        &map_box(&opts()),
        None,
        &PrintVectorTiles {
            single: &[],
            stack: &stack,
        },
    );
    let text = String::from_utf8_lossy(&ops).into_owned();
    assert!(text.contains(&rgb_op(lower)), "the lower tileset paints");
    assert!(text.contains(&rgb_op(upper)), "and so does the upper");
    // Each entry's rules get their own alpha namespace, so two tilesets cannot
    // collide over `GV0`.
    assert!(text.contains("/GV0s0 gs"), "entry 0's rule 0: {text}");
    assert!(text.contains("/GV1s0 gs"), "entry 1's rule 0");
    // Bottom-up: the lower entry's operators come first, so the upper one
    // draws over it exactly as on screen.
    let lower_at = text.find(&rgb_op(lower));
    let upper_at = text.find(&rgb_op(upper));
    assert!(lower_at < upper_at, "stack order is paint order");
}

#[test]
fn a_legacy_request_still_names_its_rules_gv0() {
    // The byte-compatibility half of the naming change: a page with one
    // tileset keeps `GV0`, `GV1`, … exactly as every previous export wrote
    // them, so the stack's `GV{entry}s{rule}` scheme costs nothing.
    let (mut request, compose) = request(Vec::new());
    request.vector = Some(
        VectorTileConfig::new("https://example.test/{z}/{x}/{y}.pbf").with_paints(vec![
            VectorTilePaint::new("lower", LayerStyle::Fill(FillStyle::new(Color::BLACK))),
        ]),
    );
    assert!(request.stack_fits_legacy_slots());
    let tiles = root(tile_with("lower"));
    let ops = page_content_planned(&request, &compose, &map_box(&opts()), None, &tiles);
    let text = String::from_utf8_lossy(&ops).into_owned();
    assert!(text.contains("/GV0 gs"), "{text}");
    assert!(
        !text.contains("s0 gs"),
        "no entry component on the legacy slot"
    );
}

#[test]
fn every_stack_rule_the_painter_names_is_registered_by_the_document() {
    // An unregistered `/GV1s0 gs` is an invalid PDF that a `contains` test
    // would pass, so the resource dictionary is checked too.
    let (mut request, compose) = request(vec![polygon_layer()]);
    request.stack = vec![
        vector_entry("lower", LayerStyle::Fill(FillStyle::new(Color::BLACK))),
        vector_entry("upper", LayerStyle::Line(LineStyle::new(Color::BLACK, 2.0))),
    ];
    let stack = vec![root(tile_with("lower")), root(tile_with("upper"))];
    let tiles = PrintVectorTiles {
        single: &[],
        stack: &stack,
    };
    let map_box = map_box(&opts());
    let out_px = raster_size_px(&map_box, &opts());
    let rgb = compose_map_rgb(&compose, &mut |_| None);
    let pdf = pdf_document_with(
        &request,
        &compose,
        &rgb,
        out_px,
        &PrintFonts::none(),
        &tiles,
    )
    .expect("a document");
    let pdf_text = String::from_utf8_lossy(&pdf);
    for name in ["GV0s0", "GV1s0"] {
        assert!(
            pdf_text.contains(&format!("/{name} ")),
            "/{name} is missing from the page's /ExtGState dictionary",
        );
    }
}

#[test]
fn a_lower_entrys_labels_reach_the_font_plan_and_the_page() {
    // The half that fails SILENTLY: an unplanned character is dropped rather
    // than mis-drawn, so a second tileset's labels would simply vanish if the
    // plan were still built from `PrintRequest::vector` alone.
    let (mut request, compose) = request(Vec::new());
    // Layer names chosen so each label carries a character NOTHING else on the
    // page does — not the title, the attribution, the scale bar's `km` or the
    // north arrow's `N`. Without that, a plan built from one source only would
    // still cover the other's letters and the test would pass while the bug
    // stood.
    request.stack = vec![
        vector_entry("zeta", LayerStyle::Symbol(SymbolStyle::new("name"))),
        vector_entry("quay", LayerStyle::Symbol(SymbolStyle::new("name"))),
    ];
    let stack = vec![root(labelled_tile("zeta")), root(labelled_tile("quay"))];
    let tiles = PrintVectorTiles {
        single: &[],
        stack: &stack,
    };
    let map_box = map_box(&opts());
    let out_px = raster_size_px(&map_box, &opts());
    let rgb = compose_map_rgb(&compose, &mut |_| None);
    let fonts = PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
    let pdf =
        pdf_document_with(&request, &compose, &rgb, out_px, &fonts, &tiles).expect("a document");
    // Both labels are in the /ToUnicode map, which is built from the CIDs the
    // plan actually embedded — the direct evidence that BOTH sources' strings
    // reached it.
    let cmap = inflated_to_unicode(&pdf);
    assert!(!cmap.is_empty(), "no /ToUnicode stream was inflated at all");
    for (label, distinctive) in [("zeta label", 'z'), ("quay label", 'q')] {
        assert!(
            cmap.contains(&format!("{:04x}", distinctive as u32)),
            "'{distinctive}', which only \"{label}\" carries, never reached the font plan",
        );
    }
}

/// Every `/ToUnicode` stream of `pdf`, inflated and concatenated as lowercase
/// hex text.
///
/// The plan's own output, read back from the file: a character that never
/// reached it has no CID and therefore no `bfchar` entry, which is exactly the
/// silent failure being tested for.
///
/// Streams are sliced by their dict's `/Length N` rather than scanned for
/// `endstream` — compressed bytes can contain either keyword and derail a naive
/// scanner — which is the same walk `tests::to_unicode_contains` makes.
fn inflated_to_unicode(pdf: &[u8]) -> String {
    let mut found = String::new();
    let mut cursor = 0;
    while let Some(at) = super::tests::find_bytes(&pdf[cursor..], b"/Length ").map(|at| cursor + at)
    {
        let digits_at = at + b"/Length ".len();
        let mut end_digits = digits_at;
        while pdf.get(end_digits).is_some_and(u8::is_ascii_digit) {
            end_digits += 1;
        }
        let length: usize = String::from_utf8_lossy(&pdf[digits_at..end_digits])
            .parse()
            .unwrap_or(0);
        cursor = end_digits;
        let Some(open) =
            super::tests::find_bytes(&pdf[end_digits..], b"stream").map(|found| end_digits + found)
        else {
            break;
        };
        let mut start = open + b"stream".len();
        if pdf.get(start) == Some(&b'\r') {
            start += 1;
        }
        if pdf.get(start) == Some(&b'\n') {
            start += 1;
        }
        let Some(data) = pdf.get(start..start.saturating_add(length)) else {
            continue;
        };
        if let Ok(body) = oxiarc_deflate::zlib::zlib_decompress(data) {
            let body = String::from_utf8_lossy(&body).into_owned();
            if body.contains("begincmap") {
                found.push_str(&body.to_lowercase());
            }
        }
        cursor = start + length;
    }
    found
}

#[test]
fn an_overlay_composites_at_its_own_opacity_over_what_is_already_there() {
    // The raster half. A hillshade at 50 % over an orthophoto must print as
    // the blend the screen shows, not as one or the other.
    let view = MapView::new(LonLat::new(0.0, 0.0), 1.0, [512.0, 512.0]).expect("a viewport");
    let solid = |channels: [u8; 4]| {
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for _ in 0..16 {
            rgba.extend_from_slice(&channels);
        }
        DecodedTile::new(4, 4, rgba).expect("a valid tile")
    };
    let mut rgb = compose_map_rgb(&view, &mut |_| Some(solid([0, 0, 0, 255])));
    let pixel = |rgb: &[u8], x: usize, y: usize| {
        let at = (y * 512 + x) * 3;
        [rgb[at], rgb[at + 1], rgb[at + 2]]
    };
    assert_eq!(pixel(&rgb, 100, 100), [0, 0, 0], "the base pass is black");

    overlay_map_rgb(&view, &mut rgb, 0.5, &mut |_| {
        Some(solid([255, 255, 255, 255]))
    });
    assert_eq!(
        pixel(&rgb, 100, 100),
        [128, 128, 128],
        "an opaque white layer at half alpha is a half blend",
    );

    // A fully transparent layer changes nothing, and neither does one at zero
    // opacity — the two ways "this layer is not visible" can be said.
    let before = rgb.clone();
    overlay_map_rgb(&view, &mut rgb, 1.0, &mut |_| Some(solid([255, 0, 0, 0])));
    assert_eq!(rgb, before, "a transparent tile paints nothing");
    overlay_map_rgb(&view, &mut rgb, 0.0, &mut |_| Some(solid([255, 0, 0, 255])));
    assert_eq!(rgb, before, "an invisible layer paints nothing");

    // A buffer that is not this view's is refused rather than written past.
    let mut wrong = vec![0_u8; 8];
    overlay_map_rgb(&view, &mut wrong, 1.0, &mut |_| {
        Some(solid([255, 0, 0, 255]))
    });
    assert_eq!(wrong, vec![0_u8; 8]);
}

#[test]
fn the_raster_entries_are_the_passes_a_shell_composites() {
    // The list a shell loops over, in the order it loops: rasters bottom-up,
    // vector entries left to the path painter.
    let (mut request, _) = request(Vec::new());
    request.stack = vec![
        cog_entry("https://example.test/a.tif", 1.0),
        vector_entry("mid", LayerStyle::Fill(FillStyle::new(Color::BLACK))),
        cog_entry("https://example.test/b.tif", 0.5),
    ];
    let rasters = request.raster_stack();
    assert_eq!(rasters.len(), 2);
    assert!((rasters[0].opacity - 1.0).abs() < f32::EPSILON);
    assert!((rasters[1].opacity - 0.5).abs() < f32::EPSILON);
    // And the vector sources keep their STACK positions, so a decoded-tile
    // list indexed by stack position lines up with them.
    let sources = request.vector_sources();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].0, Some(1), "the middle entry");
}
