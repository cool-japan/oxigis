// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The print module's own test suite — a PURE MOVE out of `print/mod.rs`
//! (print v1.6): the body and the tests together crossed the 2000-line rule.

use super::*;
use oxigeo::geojson::types::{Feature, Geometry, LineString, Point, Polygon};
use oxigis_core::{CircleStyle, FillStyle, LabelWeight, LineStyle};

/// A Regular-only text set — the shape every pre-v1.4 test means.
pub(super) fn regular<'a>(texts: &[&'a str]) -> Vec<(LabelWeight, &'a str)> {
    texts
        .iter()
        .map(|&text| (LabelWeight::Regular, text))
        .collect()
}

/// The default options — the fixture page (A4 landscape, 144 DPI).
pub(super) fn opts() -> PrintOptions {
    PrintOptions::default()
}

/// A request over `layers` framed on a 800×600 screen at zoom 2.
pub(super) fn request(layers: Vec<PrintLayer>) -> (PrintRequest, MapView) {
    let view = MapView::new(LonLat::new(0.0, 0.0), 2.0, [800.0, 600.0]).expect("a valid viewport");
    let request = PrintRequest {
        // No tiled stack: these fixtures exercise the local-vector and
        // furniture halves of the page, which the snapshot does not reach.
        stack: Vec::new(),
        title: "Print fixture".to_string(),
        attribution: "(c) Fixture contributors".to_string(),
        view,
        basemap: BasemapConfig::openstreetmap(),
        cog: None,
        archive: None,
        vector: None,
        layers,
        options: opts(),
    };
    let out_px = raster_size_px(&map_box(&opts()), &opts());
    let compose = compose_view(view, out_px);
    (request, compose)
}

pub(super) fn polygon_layer() -> PrintLayer {
    let ring = vec![
        vec![-10.0, -10.0],
        vec![10.0, -10.0],
        vec![10.0, 10.0],
        vec![-10.0, 10.0],
        vec![-10.0, -10.0],
    ];
    let polygon = Polygon::from_exterior(ring).expect("a valid ring");
    let feature = Feature::new(Some(Geometry::Polygon(polygon)), None);
    let mut fill = FillStyle::new(Color::from_hex("3388ff").expect("valid hex"));
    fill.outline_color = Some(Color::BLACK);
    let features = Arc::new(FeatureCollection::new(vec![feature]));
    PrintLayer {
        // No project name anywhere in these fixtures: they exercise the
        // GeoJSON-member and `Layer N` fallbacks, which is what the legend
        // tests assert, so they legend exactly as they did before
        // `PrintLayer::name` existed.
        name: String::new(),
        families: crate::local_vector::collection_families(&features),
        features,
        style: LayerStyle::Fill(fill).into(),
        opacity: 0.8,
    }
}

pub(super) fn line_layer() -> PrintLayer {
    let line = LineString::new(vec![vec![-20.0, 0.0], vec![20.0, 5.0]]).expect("a line");
    let feature = Feature::new(Some(Geometry::LineString(line)), None);
    let features = Arc::new(FeatureCollection::new(vec![feature]));
    PrintLayer {
        name: String::new(),
        families: crate::local_vector::collection_families(&features),
        features,
        style: LayerStyle::Line(LineStyle::new(Color::BLACK, 2.0)).into(),
        opacity: 1.0,
    }
}

pub(super) fn point_layer() -> PrintLayer {
    let point = Point::new_2d(3.0, 4.0).expect("a point");
    let feature = Feature::new(Some(Geometry::Point(point)), None);
    let features = Arc::new(FeatureCollection::new(vec![feature]));
    PrintLayer {
        name: String::new(),
        families: crate::local_vector::collection_families(&features),
        features,
        style: LayerStyle::Circle(CircleStyle::new(
            4.0,
            Color::from_hex("cc3333").expect("valid hex"),
        ))
        .into(),
        opacity: 1.0,
    }
}

#[test]
fn the_map_box_fits_inside_the_page_with_all_bands() {
    let map_box = map_box(&opts());
    assert!(map_box.x >= PAGE_MARGIN_PT);
    assert!(map_box.y > PAGE_MARGIN_PT, "the footer band sits below");
    assert!(map_box.x + map_box.width <= A4_LANDSCAPE_PT[0] - PAGE_MARGIN_PT + 0.01);
    assert!(
        map_box.y + map_box.height <= A4_LANDSCAPE_PT[1] - PAGE_MARGIN_PT - TITLE_BAND_PT + 0.01,
        "the title band sits above"
    );
    let [w, h] = raster_size_px(&map_box, &opts());
    assert_eq!(w, (map_box.width * RASTER_PX_PER_PT).round() as u32);
    assert_eq!(h, (map_box.height * RASTER_PX_PER_PT).round() as u32);
}

#[test]
fn compose_view_preserves_the_horizontal_extent_exactly() {
    let view =
        MapView::new(LonLat::new(139.7, 35.6), 11.3, [800.0, 600.0]).expect("a valid viewport");
    let out_px = raster_size_px(&map_box(&opts()), &opts());
    let compose = compose_view(view, out_px);
    let (screen_nw, screen_se) = view.world_bounds();
    let (print_nw, print_se) = compose.world_bounds();
    let screen_span = screen_se.x - screen_nw.x;
    let print_span = print_se.x - print_nw.x;
    assert!(
        (screen_span - print_span).abs() / screen_span < 1e-6,
        "the west-east span must survive the reframe: {screen_span} vs {print_span}"
    );
    assert_eq!(compose.size_px(), [out_px[0] as f32, out_px[1] as f32]);
}

#[test]
fn the_scale_bar_is_a_one_two_five_distance_that_fits_a_fifth_of_the_box() {
    for (lat, zoom) in [(0.0, 4.0), (60.0, 12.0)] {
        let view =
            MapView::new(LonLat::new(10.0, lat), zoom, [1540.0, 923.0]).expect("a valid viewport");
        let map_box = map_box(&opts());
        let bar = scale_bar(&view, &map_box).expect("a computable bar");
        let mantissa = bar.metres / 10.0_f64.powf(bar.metres.log10().floor());
        assert!(
            [1.0, 2.0, 5.0]
                .iter()
                .any(|factor| (mantissa - factor).abs() < 1e-9),
            "a 1-2-5 mantissa at lat {lat}: got {}",
            bar.metres
        );
        assert!(
            bar.width_pt > 0.0 && bar.width_pt <= map_box.width / 5.0 + 0.01,
            "the bar fits a fifth of the box: {} pt",
            bar.width_pt
        );
        assert!(!bar.label.is_empty());
    }
    // Doubling the zoom at the same latitude must shrink the distance.
    let map_box = map_box(&opts());
    let coarse =
        MapView::new(LonLat::new(0.0, 0.0), 4.0, [1540.0, 923.0]).expect("a valid viewport");
    let fine = coarse.with_zoom(8.0);
    let coarse_bar = scale_bar(&coarse, &map_box).expect("bar");
    let fine_bar = scale_bar(&fine, &map_box).expect("bar");
    assert!(fine_bar.metres < coarse_bar.metres);
}

#[test]
fn compose_pastes_an_available_tile_and_leaves_missing_ones_gray() {
    // 512x512 at zoom 1: the world is 2x2 tiles of 256 px, exactly filling
    // the surface. Only tile (1, 0) is available, solid red.
    let view = MapView::new(LonLat::new(0.0, 0.0), 1.0, [512.0, 512.0]).expect("a valid viewport");
    let red_tile = {
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for _ in 0..16 {
            rgba.extend_from_slice(&[255, 0, 0, 255]);
        }
        DecodedTile::new(4, 4, rgba).expect("a valid tile")
    };
    let rgb = compose_map_rgb(&view, &mut |tile| {
        (tile.z == 1 && tile.x == 1 && tile.y == 0).then(|| red_tile.clone())
    });
    assert_eq!(rgb.len(), 512 * 512 * 3);
    let pixel = |x: usize, y: usize| {
        let at = (y * 512 + x) * 3;
        [rgb[at], rgb[at + 1], rgb[at + 2]]
    };
    assert_eq!(pixel(300, 100), [255, 0, 0], "inside the pasted tile");
    assert_eq!(
        pixel(100, 100),
        [MISSING_TILE_GRAY; 3],
        "a missing tile stays neutral gray"
    );
    assert_eq!(
        pixel(300, 400),
        [MISSING_TILE_GRAY; 3],
        "below the pasted tile"
    );
}

#[test]
fn the_page_content_carries_image_overlay_and_text_operators() {
    let (request, compose) = request(vec![polygon_layer(), line_layer(), point_layer()]);
    let ops = page_content(&request, &compose, &map_box(&opts()));
    let text = String::from_utf8_lossy(&ops);
    assert!(text.contains("/Im0 Do"), "the raster map is placed");
    // pdf-writer puts each operator on its own line: `... re`, `W`, `n`.
    assert!(
        text.contains("W\nn"),
        "the overlay is clipped to the map box"
    );
    // The fixture's fill carries an outline, so its rings fill-and-stroke
    // even-odd (`B*`); the circle layer fills plain (`f`).
    assert!(
        text.contains("B*"),
        "the outlined polygon ring is even-odd filled and stroked"
    );
    assert!(text.contains('S'), "lines and outlines are stroked");
    assert!(text.contains(" c\n"), "circles are cubic Béziers");
    assert!(text.contains("/GS0 gs"), "layer 0's alpha state is applied");
    assert!(text.contains("/GS2 gs"), "layer 2's too");
    assert!(text.contains("Print fixture"), "the title is shown");
    assert!(
        text.contains("(c) Fixture contributors"),
        "the attribution is shown"
    );
    assert!(text.contains("Tj"), "text is actually drawn");
    assert!(
        text.contains(" m ") || text.contains(" m\n"),
        "paths move before they draw"
    );
}

#[test]
fn a_symbol_layer_is_skipped_and_says_nothing() {
    let mut symbol = point_layer();
    symbol.style = crate::local_vector::local_symbol_style("name").into();
    let (with_symbol, compose) = request(vec![symbol]);
    let ops = page_content(&with_symbol, &compose, &map_box(&opts()));
    let (empty, _) = request(Vec::new());
    let baseline = page_content(&empty, &compose, &map_box(&opts()));
    assert_eq!(
        ops.len(),
        baseline.len(),
        "a symbol layer must add no operators in v1 (labels are v1.1)"
    );
}

#[test]
fn the_document_is_a_pdf_with_the_image_font_and_alpha_states() {
    let (request, compose) = request(vec![polygon_layer()]);
    let map_box = map_box(&opts());
    let out_px = raster_size_px(&map_box, &opts());
    let rgb = compose_map_rgb(&compose, &mut |_| None);
    let pdf = pdf_document(&request, &compose, &rgb, out_px, &PrintFonts::none(), &[])
        .expect("a document");
    assert!(pdf.starts_with(b"%PDF-"), "the magic bytes lead");
    let text = String::from_utf8_lossy(&pdf);
    assert!(text.contains("/Type /Pages"));
    assert!(text.contains("/Subtype /Image"));
    assert!(text.contains(&format!("/Width {}", out_px[0])));
    assert!(text.contains(&format!("/Height {}", out_px[1])));
    assert!(text.contains("/BaseFont /Helvetica"));
    assert!(text.contains("/Encoding /WinAnsiEncoding"));
    assert!(text.contains("/FlateDecode"));
    assert!(text.contains("/ca "), "the layer alpha is registered");
    assert!(text.contains("%%EOF"));
}

#[test]
fn a_mismatched_raster_buffer_is_refused() {
    let (request, compose) = request(Vec::new());
    let out_px = raster_size_px(&map_box(&opts()), &opts());
    let error = pdf_document(
        &request,
        &compose,
        &[0_u8; 3],
        out_px,
        &PrintFonts::none(),
        &[],
    )
    .expect_err("a wrong-sized buffer must be refused");
    assert!(error.contains("RGB"), "{error}");
}

#[test]
fn the_flate_streams_round_trip_through_oxiarc() {
    let (request, compose) = request(vec![polygon_layer()]);
    let ops = page_content(&request, &compose, &map_box(&opts()));
    let compressed = oxiarc_deflate::zlib::zlib_compress(&ops, 6).expect("compression succeeds");
    let restored = oxiarc_deflate::zlib::zlib_decompress(&compressed).expect("a legal zlib stream");
    assert_eq!(restored, ops, "the stream must survive the round trip");
}

#[test]
fn page_sizes_orient_and_measure_as_paper_does() {
    assert_eq!(opts().page_size_pt(), A4_LANDSCAPE_PT);
    let a3_portrait = PrintOptions {
        page: PageSize::A3,
        orientation: PageOrientation::Portrait,
        ..opts()
    };
    assert_eq!(a3_portrait.page_size_pt(), [841.89, 1190.55]);
    let letter = PrintOptions {
        page: PageSize::Letter,
        ..opts()
    };
    assert_eq!(letter.page_size_pt(), [792.0, 612.0]);
    // A3 landscape's width is exactly A4 portrait's height — ISO paper.
    let a3_landscape = PrintOptions {
        page: PageSize::A3,
        ..opts()
    };
    assert_eq!(a3_landscape.page_size_pt()[1], A4_LANDSCAPE_PT[0]);
}

#[test]
fn a_portrait_map_box_is_taller_than_wide_and_stays_inside_the_page() {
    let portrait = PrintOptions {
        orientation: PageOrientation::Portrait,
        ..opts()
    };
    let map_box = map_box(&portrait);
    assert!(map_box.height > map_box.width);
    let [page_w, page_h] = portrait.page_size_pt();
    assert!(map_box.x + map_box.width <= page_w - PAGE_MARGIN_PT + 0.01);
    assert!(map_box.y + map_box.height <= page_h - PAGE_MARGIN_PT - TITLE_BAND_PT + 0.01);
}

#[test]
fn a_higher_raster_resolution_scales_the_pixel_size() {
    let map_box = map_box(&opts());
    let base = raster_size_px(&map_box, &opts());
    let dense = PrintOptions {
        raster_px_per_pt: 4.0,
        ..opts()
    };
    assert_eq!(dense.dpi(), 288.0);
    let [w, h] = raster_size_px(&map_box, &dense);
    assert_eq!(w, (map_box.width * 4.0).round() as u32);
    assert_eq!(h, (map_box.height * 4.0).round() as u32);
    assert!(w > base[0] && h > base[1]);
    // A hostile resolution (NaN, zero, negative) falls back to the default
    // instead of producing a degenerate raster.
    let broken = PrintOptions {
        raster_px_per_pt: f32::NAN,
        ..opts()
    };
    assert_eq!(raster_size_px(&map_box, &broken), base);
}

#[test]
fn the_media_box_follows_the_chosen_page() {
    let (mut request, compose) = request(Vec::new());
    request.options = PrintOptions {
        page: PageSize::A3,
        orientation: PageOrientation::Portrait,
        ..opts()
    };
    let map_box = map_box(&request.options);
    let out_px = raster_size_px(&map_box, &request.options);
    let compose = compose_view(compose, out_px);
    let rgb = compose_map_rgb(&compose, &mut |_| None);
    let pdf = pdf_document(&request, &compose, &rgb, out_px, &PrintFonts::none(), &[])
        .expect("a document");
    let text = String::from_utf8_lossy(&pdf);
    assert!(
        text.contains("/MediaBox [0 0 841.89 1190.55]"),
        "the page object must carry the A3 portrait size"
    );
}

#[test]
fn an_embedded_font_document_carries_the_full_type0_object_graph() {
    let (request, compose) = request(vec![polygon_layer()]);
    let map_box = map_box(&opts());
    let out_px = raster_size_px(&map_box, &opts());
    let rgb = compose_map_rgb(&compose, &mut |_| None);
    let fonts = PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
    let pdf = pdf_document(&request, &compose, &rgb, out_px, &fonts, &[]).expect("a document");
    let text = String::from_utf8_lossy(&pdf);
    assert!(text.contains("/Subtype /Type0"), "a composite font exists");
    assert!(text.contains("/Encoding /Identity-H"));
    assert!(text.contains("/Subtype /CIDFontType2"));
    assert!(text.contains("/CIDToGIDMap /Identity"));
    assert!(
        text.contains("/FontFile2"),
        "the subset program is embedded"
    );
    assert!(text.contains("/ToUnicode"), "copy-paste stays possible");
    assert!(text.contains("/DW 1000"));
    assert!(text.contains("/W ["), "per-CID widths are written");
    assert!(
        text.contains("/Registry (Adobe)") && text.contains("/Ordering (Identity)"),
        "the CIDSystemInfo names Identity ordering"
    );
    assert!(
        text.contains("/BaseFont /") && text.contains("+NotoSans"),
        "the base font is subset-tagged"
    );
    // The Helvetica degraded resource stays present alongside.
    assert!(text.contains("/BaseFont /Helvetica"));
}

#[test]
fn a_planned_page_shows_text_through_the_embedded_font_resource() {
    let (request, compose) = request(Vec::new());
    let fonts = PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
    let map_box = map_box(&opts());
    let label = scale_bar(&compose, &map_box).map(|bar| bar.label);
    let plan = font::plan(
        &fonts,
        &regular(&[
            request.title.as_str(),
            request.attribution.as_str(),
            label.as_deref().unwrap_or(""),
        ]),
        None,
    )
    .expect("a plan");
    let ops = page_content_planned(&request, &compose, &map_box, Some(&plan), &[]);
    let text = String::from_utf8_lossy(&ops);
    assert!(
        text.contains("/F1"),
        "the embedded font resource is selected"
    );
    assert!(
        !text.contains("Print fixture"),
        "the title is CIDs now, not literal WinAnsi bytes"
    );
    assert!(
        text.contains("Tj") || text.contains("TJ"),
        "text is still drawn (kerned runs are TJ arrays since v1.2)"
    );
    // The fixture title carries Noto's r,e kern pair (-20/1000 em), so
    // shaped output MUST show a TJ array with that adjustment.
    assert!(text.contains("TJ"), "the kerned title emits a TJ array");
    assert!(
        text.contains("20 "),
        "the r/e kern lands as a TJ adjustment: {text}"
    );
    // The degraded render of the same page still uses F0/literal text.
    let degraded = page_content(&request, &compose, &map_box);
    let degraded_text = String::from_utf8_lossy(&degraded);
    assert!(degraded_text.contains("/F0"));
    assert!(degraded_text.contains("Print fixture"));
}

#[test]
fn a_symbol_layer_draws_haloed_labels_when_fonts_are_live() {
    let mut properties = serde_json::Map::new();
    properties.insert(
        "name".to_string(),
        serde_json::Value::String("Tokyo".to_string()),
    );
    let feature = Feature::new(
        Some(Geometry::Point(Point::new_2d(3.0, 4.0).expect("a point"))),
        Some(properties),
    );
    let features = Arc::new(FeatureCollection::new(vec![feature]));
    let symbol = PrintLayer {
        name: String::new(),
        families: crate::local_vector::collection_families(&features),
        features,
        style: LayerStyle::Symbol(oxigis_core::SymbolStyle::new("name")).into(),
        opacity: 1.0,
    };
    let (request, compose) = request(vec![symbol]);
    let fonts = PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
    let plan = font::plan(
        &fonts,
        &regular(&["Print fixture", "(c) Fixture contributors", "Tokyo"]),
        None,
    )
    .expect("a plan");
    let map_box = map_box(&opts());
    let ops = page_content_planned(&request, &compose, &map_box, Some(&plan), &[]);
    let text = String::from_utf8_lossy(&ops);
    // The default SymbolStyle has a white halo: stroke-mode text (1 Tr)
    // must precede fill-mode text (0 Tr), under the layer's alpha state.
    assert!(text.contains("1 Tr"), "the halo pass strokes the text");
    assert!(text.contains("0 Tr"), "the fill pass follows");
    assert!(text.contains("/GS0 gs"), "the label honours layer alpha");
    // The stroke pass repeats the fill pass's show operators, so it is
    // marked as an artifact — otherwise extractors read the label TWICE.
    assert!(
        text.contains("/Artifact BMC"),
        "the halo pass is a marked-content artifact"
    );
    assert!(text.contains("EMC"), "and the marking is closed");
    // Without a plan the symbol layer stays silent (v1 behavior).
    let degraded = page_content(&request, &compose, &map_box);
    let degraded_text = String::from_utf8_lossy(&degraded);
    assert!(!degraded_text.contains("1 Tr"));
}

/// print v1.6: a Symbol style asking for VERTICAL labels really does put a
/// column on the page, through the same emitter the vertical title uses.
///
/// The screen has shipped vertical labels since v1.5, so a page that kept
/// printing them horizontally was a documented divergence from the map the
/// user is looking at — which is the one thing a print feature must not have.
#[test]
fn a_vertical_symbol_layer_draws_a_column_with_its_actual_text() {
    let mut properties = serde_json::Map::new();
    properties.insert(
        "name".to_string(),
        serde_json::Value::String("Tokyo".to_string()),
    );
    let feature = Feature::new(
        Some(Geometry::Point(Point::new_2d(3.0, 4.0).expect("a point"))),
        Some(properties),
    );
    let features = Arc::new(FeatureCollection::new(vec![feature]));
    let mut symbol_style = oxigis_core::SymbolStyle::new("name");
    symbol_style.set_orientation(oxigis_core::LabelOrientation::Vertical);
    let symbol = PrintLayer {
        name: String::new(),
        families: crate::local_vector::collection_families(&features),
        features,
        style: LayerStyle::Symbol(symbol_style).into(),
        opacity: 1.0,
    };
    let (request, compose) = request(vec![symbol]);
    let fonts = PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
    let texts = regular(&["Print fixture", "(c) Fixture contributors", "Tokyo"]);
    let map_box = map_box(&opts());

    // Horizontal baseline: the same page with the same style, unrotated.
    let flat = font::plan(&fonts, &texts, None).expect("a plan");
    let flat_ops = page_content_planned(&request, &compose, &map_box, Some(&flat), &[]);
    let flat_text = String::from_utf8_lossy(&flat_ops);

    let plan =
        font::plan_with_verticals(&fonts, &texts, None, &regular(&["Tokyo"])).expect("a plan");
    assert!(
        plan.vertical_line(LabelWeight::Regular, "Tokyo").is_some(),
        "Latin rotates under UAX #50, so the ladder accepts a sideways column",
    );
    let ops = page_content_planned(&request, &compose, &map_box, Some(&plan), &[]);
    let text = String::from_utf8_lossy(&ops);
    assert_ne!(ops, flat_ops, "the column is not the horizontal line");
    // A sideways column positions every item with an absolute `Tm`; the
    // horizontal label never writes one.
    assert!(
        text.contains(" Tm"),
        "the column sets a text matrix: {text}"
    );
    assert!(!flat_text.contains(" Tm"), "and the line does not");
    // The halo pass repeats the geometry as an artifact, so the column's
    // MANDATORY line-level `/ActualText` is written exactly once.
    assert_eq!(
        text.matches("/ActualText").count(),
        1,
        "one span for the fill pass, none for the halo artifact: {text}",
    );
    assert!(
        text.contains("/Artifact BMC"),
        "the halo is still an artifact"
    );
    assert!(text.contains("1 Tr"), "the halo pass strokes the column");
    assert!(text.contains("0 Tr"), "the fill pass follows");
    assert!(text.contains("/GS0 gs"), "under the layer's alpha state");
}

#[test]
fn the_to_unicode_stream_inflates_to_a_cmap_with_the_titles_characters() {
    // A plain 1:1 title must map a CID to U+0041 'A'.
    assert!(
        to_unicode_contains("Ax", "<0041>"),
        "no inflated stream held a ToUnicode CMap mapping 'A'"
    );
    // A ligature title: the single `fi` CID maps to BOTH characters —
    // one bfchar whose destination is the concatenated UTF-16BE.
    assert!(
        to_unicode_contains("fi", "<00660069>"),
        "the fi ligature CID must extract as both characters"
    );
}

/// Whether the document for a page titled `title` holds an inflated
/// ToUnicode CMap containing `needle`.
fn to_unicode_contains(title: &str, needle: &str) -> bool {
    let (mut request, compose) = request(Vec::new());
    request.title = title.to_string();
    request.attribution = String::new();
    let map_box = map_box(&opts());
    let out_px = raster_size_px(&map_box, &opts());
    let compose = compose_view(compose, out_px);
    let rgb = compose_map_rgb(&compose, &mut |_| None);
    let fonts = PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
    let pdf = pdf_document(&request, &compose, &rgb, out_px, &fonts, &[]).expect("a document");
    // Slice each stream by its dict's `/Length N` rather than scanning
    // for "endstream": compressed bytes can contain either keyword and
    // derail a naive scanner.
    let mut cursor = 0;
    while let Some(at) = find_bytes(&pdf[cursor..], b"/Length ").map(|at| cursor + at) {
        let digits_at = at + b"/Length ".len();
        let mut end_digits = digits_at;
        while pdf.get(end_digits).is_some_and(u8::is_ascii_digit) {
            end_digits += 1;
        }
        let length: usize = String::from_utf8_lossy(&pdf[digits_at..end_digits])
            .parse()
            .unwrap_or(0);
        cursor = end_digits;
        let Some(open) = find_bytes(&pdf[end_digits..], b"stream").map(|found| end_digits + found)
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
        let Some(data) = pdf.get(start..start + length) else {
            continue;
        };
        if let Ok(body) = oxiarc_deflate::zlib::zlib_decompress(data) {
            let body = String::from_utf8_lossy(&body).into_owned();
            if body.contains("begincmap") && body.contains(needle) {
                return true;
            }
        }
        cursor = start + length;
    }
    false
}

/// Naive subslice search (tests only).
pub(super) fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[test]
fn round_125_lands_on_the_largest_step_below_the_target() {
    // Since v1.7 the rounding answers its (mantissa, decade) parts — the
    // segmentation reads the mantissa — so the historical assertions are
    // made about the product, which is the number the bar spans.
    let step = |target: f64| scalebar::round_125_down(target).map(|(factor, base)| factor * base);
    assert_eq!(step(7.0), Some(5.0));
    assert_eq!(step(2.4), Some(2.0));
    assert_eq!(step(1.9), Some(1.0));
    assert_eq!(step(10.0), Some(10.0));
    assert_eq!(step(999.0), Some(500.0));
    assert_eq!(step(0.0), None);
    assert_eq!(step(f64::NAN), None);
    // The mantissa itself is what the bar divides by.
    assert_eq!(
        scalebar::round_125_down(7.0).map(|(factor, _)| factor),
        Some(5.0)
    );
}

#[test]
fn no_marked_content_is_emitted_for_ltr_text() {
    // The byte-identity floor's visible half: an LTR-only page carries
    // no /Span and no Ts, so the v1.2 output shape is untouched.
    let (request, compose) = request(vec![polygon_layer()]);
    let map_box = map_box(&opts());
    let fonts = PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
    let plan = font::plan(
        &fonts,
        &regular(&[&request.title, &request.attribution]),
        None,
    );
    let ops = page_content_planned(&request, &compose, &map_box, plan.as_ref(), &[]);
    let text = String::from_utf8_lossy(&ops);
    assert!(!text.contains("/Span"), "no ActualText for LTR-only text");
    assert!(!text.contains(" Ts"), "no rise for LTR-only text");
    assert!(
        plan.as_ref()
            .and_then(|plan| plan.actual_text(LabelWeight::Regular, &request.title))
            .is_none(),
        "an LTR title earns no line-level span"
    );
}

#[test]
#[ignore = "reads C:/Windows/Fonts/tahoma.ttf; the probe-pinned Arabic golden"]
fn live_windows_arabic_shapes_reorders_and_spans() {
    // Documents the manual verification the design probes pinned:
    // tahoma shapes the five contextual forms, the plan records the
    // line-level span, and the emitter wraps the line in
    // /Span <</ActualText <FEFF...>>>.
    let Ok(tahoma) = std::fs::read("C:/Windows/Fonts/tahoma.ttf") else {
        return;
    };
    let fonts = PrintFonts::new(vec![tahoma]);
    let title = "\u{645}\u{631}\u{62D}\u{628}\u{627}";
    let plan = font::plan(&fonts, &regular(&[title]), None).expect("tahoma plans the title");
    assert_eq!(
        plan.actual_text(LabelWeight::Regular, title),
        Some(title),
        "the line-level span carries the LOGICAL text"
    );
    let runs = plan.runs(LabelWeight::Regular, title);
    let glyphs: usize = runs.iter().map(|run| run.glyphs.len()).sum();
    assert_eq!(glyphs, 5, "five contextual forms");
    let mut content = Content::new();
    show_line(&mut content, Some(&plan), 10.0, 20.0, 16.0, title);
    let text = String::from_utf8_lossy(&content.finish()).into_owned();
    assert!(text.contains("/Span"), "{text}");
    assert!(text.contains("/ActualText <FEFF"), "{text}");
}

/// Every CID a plan emits for `text`, in visual order.
fn plan_cids(plan: &TextPlan, text: &str) -> Vec<u16> {
    plan.runs(LabelWeight::Regular, text)
        .iter()
        .flat_map(|run| run.glyphs.iter().map(|glyph| glyph.cid))
        .collect()
}

#[test]
#[ignore = "reads C:/Windows/Fonts/tahoma.ttf; the v1.4 L4 mirroring golden"]
fn live_windows_rtl_brackets_mirror_through_the_complete_table() {
    // D-M2's positive half: inside an RTL line every bracket takes its
    // mirror. Asserted on an UNBALANCED source, so the mirrored glyph
    // cannot be confused with one the string already contained.
    let Ok(tahoma) = std::fs::read("C:/Windows/Fonts/tahoma.ttf") else {
        return;
    };
    let face = ttf_parser::Face::parse(&tahoma, 0).expect("tahoma parses");
    let fonts = PrintFonts::new(vec![tahoma.clone()]);
    let balanced = "\u{642}\u{64A}\u{645}\u{629} (12) [x]";
    let unbalanced = "\u{642}\u{64A}\u{645}\u{629} (12";
    let plan = font::plan(&fonts, &regular(&[balanced, unbalanced]), None)
        .expect("tahoma plans both lines");
    assert_eq!(
        plan.actual_text(LabelWeight::Regular, balanced),
        Some(balanced),
        "the bracketed line shaped through the bidi path"
    );
    let gids = &plan.fonts[0].gids;
    let cid_of = |ch: char| {
        let gid = face.glyph_index(ch).expect("tahoma covers the bracket").0;
        *gids
            .get(&gid)
            .expect("the bracket glyph reached the subset")
    };
    let cids = plan_cids(&plan, unbalanced);
    assert!(
        cids.contains(&cid_of(')')),
        "the lone '(' must render as ')' — UAX #9 L4",
    );
    assert!(
        !cids.contains(&cid_of('(')),
        "and the unmirrored form must be gone",
    );
    // Extraction still says '(' — the line-level span is what makes the
    // mirrored CID's /ToUnicode harmless.
    assert_eq!(
        plan.actual_text(LabelWeight::Regular, unbalanced),
        Some(unbalanced)
    );
}

#[test]
#[ignore = "reads C:/Windows/Fonts/tahoma.ttf; the v1.4 coverage-guard golden"]
fn live_windows_an_obscure_mirror_partner_does_not_un_shape_the_line() {
    // D-M2's decisive half. Growing 32 -> 428 codepoints puts pairs on
    // the page whose PARTNER no face draws; before the guard that gid 0
    // refused the segment and the whole line fell back to the v1.1
    // per-character walk — a regression caused purely by completeness.
    // The pair is discovered from the face itself, so the golden holds on
    // any machine that has tahoma at all.
    let Ok(tahoma) = std::fs::read("C:/Windows/Fonts/tahoma.ttf") else {
        return;
    };
    let face = ttf_parser::Face::parse(&tahoma, 0).expect("tahoma parses");
    let Some((covered, _partner)) = mirror_table::MIRROR_PAIRS
        .iter()
        .copied()
        .find(|&(from, to)| face.glyph_index(from).is_some() && face.glyph_index(to).is_none())
    else {
        // A face that covers every partner cannot exercise the guard.
        return;
    };
    let text = format!("\u{642}\u{64A}\u{645}\u{629} {covered}");
    let fonts = PrintFonts::new(vec![tahoma.clone()]);
    let plan = font::plan(&fonts, &regular(&[text.as_str()]), None).expect("tahoma plans the line");
    assert_eq!(
        plan.actual_text(LabelWeight::Regular, text.as_str()),
        Some(text.as_str()),
        "the line must still shape: an uncoverable partner is left \
         unmirrored, never escalated into a whole-line refusal",
    );
    let gid = face
        .glyph_index(covered)
        .expect("covered by construction")
        .0;
    let cid = *plan.fonts[0]
        .gids
        .get(&gid)
        .expect("the character's own glyph reached the subset");
    assert!(
        plan_cids(&plan, text.as_str()).contains(&cid),
        "{covered:?} renders as itself — a locally wrong-way bracket, \
         which is what every viewer without L4 already shows",
    );
}

#[test]
fn the_vertical_title_option_defaults_off_and_a_refusal_changes_nothing() {
    // The byte-identity floor: with the option OFF the plan never runs
    // the vertical pass at all, and with it ON but a title the ladder
    // REFUSES the page keeps today's exact operators.
    assert!(!opts().vertical_title, "off by default");
    let (mut request, compose) = request(Vec::new());
    let map_box = map_box(&opts());
    let fonts = PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);

    // An RTL title: `has_rtl` refuses it on the ladder's second rung,
    // whatever the option says.
    request.title = "\u{645}\u{631}\u{62D}\u{628}\u{627}".to_string();
    let texts = regular(&[&request.title, &request.attribution]);
    let off = font::plan(&fonts, &texts, None).expect("a plan");
    assert!(off.vertical_title().is_none());
    let baseline = page_content_planned(&request, &compose, &map_box, Some(&off), &[]);

    request.options.vertical_title = true;
    let asked = font::plan(&fonts, &texts, Some(request.title.as_str())).expect("a plan");
    assert!(
        asked.vertical_title().is_none(),
        "a right-to-left title is never set vertically",
    );
    let ops = page_content_planned(&request, &compose, &map_box, Some(&asked), &[]);
    assert_eq!(
        ops, baseline,
        "a refused vertical title is byte-identical to today's output",
    );
}

/// The v1.5 named shell-visible change, at the page level: a Latin title
/// that refused in v1.4 now hangs as a sideways column — but ONLY when
/// the option is on, so no default export moves one byte.
#[test]
fn a_latin_title_becomes_a_sideways_column_only_when_the_option_is_on() {
    let (mut request, compose) = request(Vec::new());
    let map_box = map_box(&opts());
    let fonts = PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
    let texts = regular(&[&request.title, &request.attribution]);

    let off = font::plan(&fonts, &texts, None).expect("a plan");
    assert!(off.vertical_title().is_none(), "the option gates the pass");

    request.options.vertical_title = true;
    let asked = font::plan(&fonts, &texts, Some(request.title.as_str())).expect("a plan");
    let line = asked
        .vertical_title()
        .expect("a rotated-only title is a legal sideways column");
    assert!(!line.is_all_upright(), "nothing in it stacks");
    assert_eq!(line.items.len(), 1, "one Latin run, one item");
    assert_eq!(line.actual_text, request.title);
    let ops = page_content_planned(&request, &compose, &map_box, Some(&asked), &[]);
    let text = String::from_utf8_lossy(&ops).into_owned();
    assert!(text.contains("0 -1 1 0 "), "the sideways matrix: {text}");
    // An all-ASCII title encodes as a PDFDocEncoded literal rather than
    // UTF-16BE, but the span itself is just as mandatory.
    assert_eq!(
        text.matches("/ActualText").count(),
        1,
        "exactly one line-level span: {text}",
    );
}

#[test]
#[ignore = "reads C:/Windows/Fonts/YuGothM.ttc; the v1.4 vertical-title golden"]
fn live_windows_vertical_title() {
    // The end-to-end pin: a CJK title, the option on, a real Windows CJK
    // face — the page must carry the stacked Td/Tj sequence and the
    // mandatory /ActualText span, and the horizontal title operators
    // must be gone.
    let Ok(yugoth) = std::fs::read("C:/Windows/Fonts/YuGothM.ttc") else {
        return;
    };
    let fonts = PrintFonts::new(vec![yugoth]);
    let title = "\u{6771}\u{4EAC}\u{90FD}\u{5FC3}";
    let (mut request, compose) = request(Vec::new());
    request.title = title.to_string();
    request.attribution = String::new();
    request.options.vertical_title = true;
    let map_box = map_box(&request.options);

    let plan = font::plan(&fonts, &regular(&[title]), Some(title)).expect("a plan");
    let line = plan
        .vertical_title()
        .expect("YuGothic sets a four-kanji title vertically");
    assert_eq!(line.items.len(), 4, "one glyph per character");
    assert!(line.is_all_upright(), "four kanji stack, nothing rotates");
    assert_eq!(line.actual_text, title, "the span carries the LOGICAL text",);
    // Full-width kanji centre by exactly zero and step one em each.
    for item in &line.items {
        let super::vertical::VerticalItem::Upright(glyph) = item else {
            panic!("every item of a four-kanji title is upright");
        };
        assert_eq!(glyph.x_shift_1000, 0.0, "full-width cells do not shift");
        assert!((glyph.pitch_1000 - 1000.0).abs() < 1.0, "one em of pitch");
    }
    assert!((line.advance_1000 - 4000.0).abs() < 4.0);
    assert_eq!(line.box_pt(TITLE_FONT_PT), [TITLE_FONT_PT, 64.0]);

    let ops = page_content_planned(&request, &compose, &map_box, Some(&plan), &[]);
    let text = String::from_utf8_lossy(&ops).into_owned();
    assert!(text.contains("/ActualText <FEFF"), "the mandatory span");
    // Four Td steps for the title, the last three relative and downward.
    assert!(
        text.contains(&format!("-{TITLE_FONT_PT} Td")),
        "the pen steps down one em per glyph: {text}",
    );
    assert!(!text.contains(" Tm"), "an all-upright line never uses Tm");
    // And the same page with the option OFF still prints horizontally.
    request.options.vertical_title = false;
    let flat = font::plan(&fonts, &regular(&[title]), None).expect("a plan");
    assert!(flat.vertical_title().is_none());
}

#[test]
#[ignore = "reads C:/Windows/Fonts/YuGothM.ttc; the v1.5 rotated-run golden"]
fn live_windows_vertical_title_with_a_rotated_latin_run() {
    // The v1.5 end-to-end pin (D-B1..D-B4): 東京 2026 and 東京Tower —
    // titles that REFUSED in v1.4 — now hang as one stacked column with a
    // sideways Latin run under it, positioned by absolute `Tm`s and still
    // carrying exactly ONE line-level /ActualText.
    let Ok(yugoth) = std::fs::read("C:/Windows/Fonts/YuGothM.ttc") else {
        return;
    };
    let fonts = PrintFonts::new(vec![yugoth]);
    for title in ["\u{6771}\u{4EAC} 2026", "\u{6771}\u{4EAC}Tower"] {
        let (mut request, compose) = request(Vec::new());
        request.title = title.to_string();
        request.attribution = String::new();
        request.options.vertical_title = true;
        let map_box = map_box(&request.options);

        let plan = font::plan(&fonts, &regular(&[title]), Some(title)).expect("a plan");
        let line = plan
            .vertical_title()
            .unwrap_or_else(|| panic!("{title:?} must set vertically"));
        assert!(!line.is_all_upright(), "{title:?} carries a sideways run");
        // The item sequence is exactly what the shared itemiser cut.
        let expected: Vec<bool> = oxigis_render::label::vertical_runs(title)
            .iter()
            .flat_map(|run| {
                if run.is_upright() {
                    vec![true; run.text().chars().count()]
                } else {
                    vec![false]
                }
            })
            .collect();
        let got: Vec<bool> = line
            .items
            .iter()
            .map(|item| matches!(item, super::vertical::VerticalItem::Upright(_)))
            .collect();
        assert_eq!(got, expected, "{title:?}: item sequence");
        // The column height is the sum of the item advances, and the box
        // stays one em wide.
        let summed: f32 = line.items.iter().map(|item| item.advance_1000()).sum();
        assert!(
            (summed - line.advance_1000).abs() < 0.01,
            "{title:?}: {summed} vs {}",
            line.advance_1000,
        );
        assert_eq!(line.box_pt(TITLE_FONT_PT)[0], TITLE_FONT_PT);

        let ops = page_content_planned(&request, &compose, &map_box, Some(&plan), &[]);
        let text = String::from_utf8_lossy(&ops).into_owned();
        assert_eq!(
            text.matches("/ActualText <FEFF").count(),
            1,
            "{title:?}: exactly one span, flat: {text}",
        );
        assert!(text.contains("0 -1 1 0 "), "{title:?}: the sideways Tm");
        assert!(text.contains("1 0 0 1 "), "{title:?}: the upright Tm");
        assert_eq!(
            text.matches(" Tm").count(),
            line.items.len(),
            "{title:?}: one Tm per item: {text}",
        );
    }
}

/// A tile of `width × height` texels, each given by `pixel`.
fn tile_of(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 4]) -> DecodedTile {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            rgba.extend_from_slice(&pixel(x, y));
        }
    }
    DecodedTile::new(width, height, rgba).expect("a valid tile")
}

/// One tile pasted into a fresh `edge × edge` RGB buffer, filling it.
fn pasted(edge: usize, pixels: &DecodedTile) -> Vec<u8> {
    let mut out = vec![MISSING_TILE_GRAY; edge * edge * 3];
    let placement = oxigis_render::TilePlacement {
        tile: TileId { z: 0, x: 0, y: 0 },
        x: 0.0,
        y: 0.0,
        size: edge as f32,
    };
    paste_tile(&mut out, edge, edge, &placement, pixels, Over::Paper);
    out
}

#[test]
fn an_upscaled_two_colour_tile_interpolates_instead_of_blocking() {
    // Black left half, white right half, magnified 4x: nearest sampling
    // produced exactly two values on the row, bilinear must produce a ramp.
    let tile = tile_of(2, 2, |x, _| {
        if x == 0 {
            [0, 0, 0, 255]
        } else {
            [255, 255, 255, 255]
        }
    });
    let out = pasted(8, &tile);
    let row: Vec<u8> = (0..8).map(|x| out[x * 3]).collect();
    assert_eq!(row[0], 0, "the left edge clamps to the black texel");
    assert_eq!(row[7], 255, "the right edge clamps to the white texel");
    assert!(
        row.iter().any(|&value| value > 0 && value < 255),
        "an upscaled seam must carry intermediate values: {row:?}",
    );
    // Monotone: interpolation, not noise.
    assert!(row.windows(2).all(|pair| pair[0] <= pair[1]), "{row:?}");
}

#[test]
fn an_opaque_tile_still_reproduces_its_own_bytes() {
    // The byte-identity floor every basemap sits on: a flat opaque tile
    // survives interpolation and compositing unchanged.
    let tile = tile_of(4, 4, |_, _| [17, 128, 240, 255]);
    let out = pasted(16, &tile);
    for pixel in out.chunks_exact(3) {
        assert_eq!(pixel, [17, 128, 240]);
    }
}

#[test]
fn tile_alpha_resolves_against_paper_white_not_against_the_missing_plate() {
    // A fully transparent overlay texel used to print its raw RGB — black
    // plates where the page is white.
    let clear = tile_of(2, 2, |_, _| [0, 0, 0, 0]);
    let out = pasted(4, &clear);
    for pixel in out.chunks_exact(3) {
        assert_eq!(pixel, [255, 255, 255], "transparent prints as paper");
    }
    // And a half-covered texel composites, rather than overwriting.
    let half = tile_of(2, 2, |_, _| [255, 0, 0, 128]);
    let out = pasted(4, &half);
    for pixel in out.chunks_exact(3) {
        assert_eq!(pixel, [255, 127, 127], "50 % red over white");
    }
}

#[test]
fn a_pixel_no_tile_covers_keeps_the_missing_plate() {
    // The alpha work must not turn the never-arrived plate white: a page
    // with a hole in the basemap has to look like one.
    let view = MapView::new(LonLat::new(0.0, 0.0), 1.0, [512.0, 512.0]).expect("a valid viewport");
    let rgb = compose_map_rgb(&view, &mut |_| None);
    assert!(rgb.iter().all(|&value| value == MISSING_TILE_GRAY));
}

#[test]
fn a_hostile_raster_resolution_is_clamped_instead_of_allocated() {
    let map_box = map_box(&opts());
    let hostile = PrintOptions {
        raster_px_per_pt: 1e9,
        ..opts()
    };
    let [width, height] = raster_size_px(&map_box, &hostile);
    let pixels = width as usize * height as usize;
    assert!(
        pixels <= max_raster_pixels(),
        "{width}x{height} = {pixels} exceeds the budget",
    );
    // The aspect ratio survives the clamp — the page is not distorted.
    let asked = f64::from(map_box.width) / f64::from(map_box.height);
    let got = f64::from(width) / f64::from(height);
    assert!((asked - got).abs() < 0.01, "{asked} vs {got}");
    // And every offered resolution is inside the ceiling, so the dialog can
    // never ask for something the gate has to take back.
    for px_per_pt in RASTER_PX_PER_PT_CHOICES {
        assert!(px_per_pt <= MAX_RASTER_PX_PER_PT);
        let clamped = raster_size_px(
            &map_box,
            &PrintOptions {
                raster_px_per_pt: px_per_pt,
                ..opts()
            },
        );
        assert_eq!(
            clamped,
            [
                (map_box.width * px_per_pt).round() as u32,
                (map_box.height * px_per_pt).round() as u32,
            ],
            "{px_per_pt} must pass through untouched",
        );
    }
}

#[test]
fn the_resolution_choices_reach_300_dpi() {
    let labels: Vec<String> = RASTER_PX_PER_PT_CHOICES
        .iter()
        .map(|&px_per_pt| dpi_label(px_per_pt))
        .collect();
    assert_eq!(labels, ["144 dpi", "216 dpi", "288 dpi", "300 dpi"]);
}

#[test]
fn the_dialog_starts_on_a_choice_and_can_reach_every_other() {
    // The Export-PDF dialog binds `PrintOptions::raster_px_per_pt` to these
    // entries by EQUALITY (`egui::Ui::selectable_value`), so the default has
    // to BE one of them: a value outside the list opens the combo with
    // nothing highlighted and the user cannot get back to it once they move.
    assert_eq!(PrintOptions::default().raster_px_per_pt, RASTER_PX_PER_PT);
    assert!(
        RASTER_PX_PER_PT_CHOICES.contains(&RASTER_PX_PER_PT),
        "the dialog's initial resolution must be one of the offered ones",
    );
    // Strictly ascending: the menu reads as a scale, and no two entries can
    // ever be highlighted at once.
    for window in RASTER_PX_PER_PT_CHOICES.windows(2) {
        assert!(window[0] < window[1], "the choices must ascend: {window:?}");
    }
    // Distinct labels too — two rows reading "288 dpi" would be unusable
    // however different the underlying f32s are.
    let mut labels: Vec<String> = RASTER_PX_PER_PT_CHOICES
        .iter()
        .map(|&px_per_pt| dpi_label(px_per_pt))
        .collect();
    labels.sort();
    labels.dedup();
    assert_eq!(labels.len(), RASTER_PX_PER_PT_CHOICES.len());
}

#[test]
fn the_default_page_geometry_is_the_shipped_layout() {
    for page in PageSize::ALL {
        for orientation in PageOrientation::ALL {
            let options = PrintOptions {
                page,
                orientation,
                ..opts()
            };
            assert_eq!(
                map_box_with(&options, PageGeometry::default()),
                map_box(&options),
                "the default geometry must reproduce today's box",
            );
        }
    }
}

#[test]
fn a_hostile_page_geometry_still_leaves_a_map_box_on_the_paper() {
    let options = opts();
    let [page_w, page_h] = options.page_size_pt();
    for geometry in [
        PageGeometry {
            margin_pt: 10_000.0,
            ..PageGeometry::default()
        },
        PageGeometry {
            margin_pt: f32::NAN,
            title_band_pt: -50.0,
            footer_band_pt: f32::INFINITY,
            band_gap_pt: 1e30,
        },
    ] {
        let map_box = map_box_with(&options, geometry);
        assert!(map_box.width >= 0.0 && map_box.height >= 0.0, "{map_box:?}");
        assert!(map_box.x >= 0.0 && map_box.y >= 0.0, "{map_box:?}");
        assert!(map_box.x + map_box.width <= page_w + 0.01, "{map_box:?}");
        assert!(map_box.y + map_box.height <= page_h + 0.01, "{map_box:?}");
    }
    // A wider margin really does shrink the map — the option does something.
    let wide = map_box_with(
        &options,
        PageGeometry {
            margin_pt: 72.0,
            ..PageGeometry::default()
        },
    );
    assert!(wide.width < map_box(&options).width);
    assert_eq!(wide.x, 72.0);
}

#[test]
fn an_over_long_title_and_credit_are_elided_inside_the_page() {
    let (mut request, compose) = request(Vec::new());
    request.title = "Ō".repeat(400);
    request.attribution = "(c) A very long credit line, repeated. ".repeat(20);
    let [page_w, _] = request.options.page_size_pt();
    let room = text_room_pt(page_w);
    // Degraded mode (no plan) measures with the Helvetica estimate; the
    // elided line must fit the room the margins leave.
    let title = elide_to_width(None, &request.title, TITLE_FONT_PT, room);
    assert!(title.len() < request.title.len(), "the title was trimmed");
    assert!(title.ends_with('…'), "and says so: {title}");
    assert!(line_width_pt(None, &title, TITLE_FONT_PT) <= room);
    let credit = elide_to_width(None, &request.attribution, FOOTER_FONT_PT, room);
    assert!(line_width_pt(None, &credit, FOOTER_FONT_PT) <= room);
    // A line that fits is passed through untouched, byte for byte.
    assert_eq!(
        elide_to_width(None, "Print fixture", TITLE_FONT_PT, room),
        "Print fixture",
    );
    // End to end: the page never emits the whole 400-character run.
    let ops = page_content(&request, &compose, &map_box(&opts()));
    let text = String::from_utf8_lossy(&ops);
    assert!(
        !text.contains(&"?".repeat(400)),
        "the title ran off the page"
    );
}

/// A synthetic "photographic" raster: smooth per-channel gradients across
/// the whole image — the shape a real aerial or street basemap tile takes.
/// Almost every pixel differs from its neighbours by a few levels, which
/// defeats zlib's LZ77 back-references (print v1.8's `/FlateDecode` floor)
/// but is exactly the low-frequency content JPEG's block DCT concentrates
/// into a handful of coefficients per block.
fn photographic_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f32, y as f32);
            let r = 128.0 + 100.0 * (fx * 0.13).sin() + 20.0 * (fy * 0.29).cos();
            let g = 128.0 + 100.0 * (fy * 0.11).sin() + 20.0 * (fx * 0.31).cos();
            let b = 128.0 + 90.0 * ((fx + fy) * 0.07).sin();
            out.push(r.clamp(0.0, 255.0) as u8);
            out.push(g.clamp(0.0, 255.0) as u8);
            out.push(b.clamp(0.0, 255.0) as u8);
        }
    }
    out
}

/// A synthetic "line-art" raster: a plain white field crossed by thin black
/// rules at periods coprime with the JPEG block size — the shape a diagram,
/// a scanned line drawing or a UI screenshot takes. Almost entirely one
/// repeated byte value, so zlib crushes it with long back-references; every
/// block a rule crosses is still a hard edge, so JPEG's DCT keeps spending
/// real bits there instead of collapsing to a flat run the way zlib does.
fn line_art_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![255_u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            if x % 17 == 0 || y % 23 == 0 || x == y {
                let index = ((y * width + x) * 3) as usize;
                out[index] = 0;
                out[index + 1] = 0;
                out[index + 2] = 0;
            }
        }
    }
    out
}

/// A synthetic "rendered cartography" raster: the shape of a real OSM
/// standard-style raster tile rather than a photograph or pen-plotter line
/// art — large flat fills (land/water), a sparse grid of antialiased 2-3px
/// roads, and a handful of small dense high-contrast blobs standing in for
/// baked-in place-name glyphs. This is the actual default basemap
/// (`BasemapConfig::default()` is `openstreetmap()`), so it is what
/// `photo_jpeg`'s default-on race most needs to be judged against.
fn osm_like_rgb(width: u32, height: u32) -> Vec<u8> {
    let water = [170.0_f32, 211.0, 223.0];
    let land = [242.0_f32, 239.0, 233.0];
    let road = [255.0_f32, 255.0, 255.0];
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f32, y as f32);
            // A diagonal land/water boundary, antialiased over ~3 px.
            let boundary = fx * 0.4 + fy * 0.6 - f32::from(u16::try_from(width).unwrap_or(0)) * 0.3;
            let t = (boundary / 3.0 + 0.5).clamp(0.0, 1.0);
            let mut rgb = [
                water[0] * (1.0 - t) + land[0] * t,
                water[1] * (1.0 - t) + land[1] * t,
                water[2] * (1.0 - t) + land[2] * t,
            ];
            // A sparse grid of antialiased roads: half-width 1.5 px, ~1 px
            // antialiasing skirt either side.
            for (period, coord) in [(97_u32, x), (113_u32, y)] {
                let offset = (coord % period) as f32 - period as f32 / 2.0;
                let distance = offset.abs();
                let alpha = (1.0 - (distance - 1.5).max(0.0)).clamp(0.0, 1.0);
                if alpha > 0.0 {
                    for channel in 0..3 {
                        rgb[channel] = rgb[channel] * (1.0 - alpha) + road[channel] * alpha;
                    }
                }
            }
            for value in rgb {
                out.push(value.round().clamp(0.0, 255.0) as u8);
            }
        }
    }
    // A dozen small, dense, high-contrast blobs — baked-in label glyphs.
    for i in 0..12_u32 {
        let seed = i * 7919 + 104_729;
        let lx = seed % width.max(1);
        let ly = (seed / 7) % height.max(1);
        for oy in 0..4_u32 {
            for ox in 0..6_u32 {
                let x = (lx + ox).min(width.saturating_sub(1));
                let y = (ly + oy).min(height.saturating_sub(1));
                let index = ((y * width + x) * 3) as usize;
                if let Some(pixel) = out.get_mut(index..index + 3) {
                    pixel.copy_from_slice(&[20, 20, 20]);
                }
            }
        }
    }
    out
}

/// A much busier variant of [`osm_like_rgb`]: a dense street grid (roads
/// every ~24 px instead of ~100) and ~15x the label ink, standing in for a
/// dense city-centre sheet rather than a suburban one — the stress case for
/// the "does JPEG ever win on real cartography" question.
fn busy_osm_like_rgb(width: u32, height: u32) -> Vec<u8> {
    let water = [170.0_f32, 211.0, 223.0];
    let land = [242.0_f32, 239.0, 233.0];
    let road = [255.0_f32, 255.0, 255.0];
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f32, y as f32);
            let boundary = fx * 0.4 + fy * 0.6 - f32::from(u16::try_from(width).unwrap_or(0)) * 0.3;
            let t = (boundary / 3.0 + 0.5).clamp(0.0, 1.0);
            let mut rgb = [
                water[0] * (1.0 - t) + land[0] * t,
                water[1] * (1.0 - t) + land[1] * t,
                water[2] * (1.0 - t) + land[2] * t,
            ];
            for (period, coord) in [(23_u32, x), (29_u32, y)] {
                let offset = (coord % period) as f32 - period as f32 / 2.0;
                let distance = offset.abs();
                let alpha = (1.0 - (distance - 1.5).max(0.0)).clamp(0.0, 1.0);
                if alpha > 0.0 {
                    for channel in 0..3 {
                        rgb[channel] = rgb[channel] * (1.0 - alpha) + road[channel] * alpha;
                    }
                }
            }
            for value in rgb {
                out.push(value.round().clamp(0.0, 255.0) as u8);
            }
        }
    }
    for i in 0..200_u32 {
        let seed = i * 7919 + 104_729;
        let lx = seed % width.max(1);
        let ly = (seed / 7) % height.max(1);
        for oy in 0..5_u32 {
            for ox in 0..8_u32 {
                let x = (lx + ox).min(width.saturating_sub(1));
                let y = (ly + oy).min(height.saturating_sub(1));
                let index = ((y * width + x) * 3) as usize;
                if let Some(pixel) = out.get_mut(index..index + 3) {
                    pixel.copy_from_slice(&[20, 20, 20]);
                }
            }
        }
    }
    out
}

#[test]
fn a_realistic_osm_basemap_at_default_resolution_stays_flatedecode() {
    // The risk this guards: `photo_jpeg`'s default-on race must not switch
    // the DEFAULT basemap (`BasemapConfig::default()` is `openstreetmap()`
    // — rendered cartography, not a photograph) to a lossy encoding that
    // would ring the baked-in place-name glyphs real OSM tiles carry. Two
    // content profiles, both measured at the REAL default export
    // resolution (`raster_size_px` on `PrintOptions::default()`, ~1540x935
    // px — not a scaled-down fixture size that would understate how much
    // flat fill a real page carries): a light suburban tile and a dense
    // city-centre one with 15x the road grid density and label ink.
    // Measured: both land at identical byte counts whether `photo_jpeg` is
    // on or off (JPEG loses the race outright) — large flat land/water
    // fills still dominate a busy rendered map's pixel count even with
    // antialiased roads and dense labels layered on top, which is why zlib
    // wins here while it loses decisively for genuinely photographic
    // content (`a_photographic_raster_embeds_as_a_materially_smaller_dct_stream`).
    let out_px = raster_size_px(&map_box(&opts()), &opts());
    let (request, compose) = request(Vec::new());
    for (label, rgb) in [
        ("suburban", osm_like_rgb(out_px[0], out_px[1])),
        ("busy-city", busy_osm_like_rgb(out_px[0], out_px[1])),
    ] {
        let pdf = pdf_document(&request, &compose, &rgb, out_px, &PrintFonts::none(), &[])
            .unwrap_or_else(|error| panic!("{label}: a document must still assemble: {error}"));
        assert!(
            !String::from_utf8_lossy(&pdf).contains("/Filter /DCTDecode"),
            "{label}: a rendered-cartography raster at the real default export \
             resolution must not lose baked-in label fidelity to the JPEG race",
        );
    }
}

/// A page-sized-enough raster for the JPEG-vs-Flate race to mean something:
/// large enough to carry many DCT blocks, small enough that the whole test
/// suite stays fast.
const RACE_RASTER_PX: [u32; 2] = [256, 192];

#[test]
fn a_photographic_raster_embeds_as_a_materially_smaller_dct_stream() {
    let map_px = RACE_RASTER_PX;
    let rgb = photographic_rgb(map_px[0], map_px[1]);

    let (with_jpeg, compose) = request(Vec::new());
    let jpeg_pdf = pdf_document(&with_jpeg, &compose, &rgb, map_px, &PrintFonts::none(), &[])
        .expect("a document");
    let jpeg_text = String::from_utf8_lossy(&jpeg_pdf);
    // The structural markers every print/tests.rs document test checks —
    // this path must stay just as openable as the pre-v1.8 Flate-only one.
    assert!(jpeg_pdf.starts_with(b"%PDF-"), "the magic bytes lead");
    assert!(jpeg_text.contains("/Type /Pages"));
    assert!(jpeg_text.contains("/Subtype /Image"));
    assert!(jpeg_text.contains(&format!("/Width {}", map_px[0])));
    assert!(jpeg_text.contains(&format!("/Height {}", map_px[1])));
    assert!(jpeg_text.contains("%%EOF"));
    assert!(
        jpeg_text.contains("/Filter /DCTDecode"),
        "a photographic raster must win the size race and embed as DCTDecode: {jpeg_text}",
    );

    // The same pixels, forced through the pre-v1.8 zlib-only path, are the
    // honest baseline this has to beat — not an assumed number.
    let (mut forced_flate, _) = request(Vec::new());
    forced_flate.options.photo_jpeg = false;
    let flate_pdf = pdf_document(
        &forced_flate,
        &compose,
        &rgb,
        map_px,
        &PrintFonts::none(),
        &[],
    )
    .expect("a document");
    assert!(!String::from_utf8_lossy(&flate_pdf).contains("/Filter /DCTDecode"));
    // Measured at 256x192: ~15 KB DCTDecode vs ~142 KB FlateDecode, a ~9.5x
    // reduction. The gate is a fraction of that so a future zlib or JPEG
    // encoder version cannot make this test flaky over a few percent drift.
    assert!(
        jpeg_pdf.len() * 3 < flate_pdf.len(),
        "materially smaller: DCTDecode {} bytes vs FlateDecode {} bytes",
        jpeg_pdf.len(),
        flate_pdf.len(),
    );
}

#[test]
fn a_line_art_raster_keeps_flatedecode_because_jpeg_does_not_win_the_race() {
    // photo_jpeg stays at its default (on): the point of this test is that
    // the race itself rejects JPEG for this content, not that the option
    // was turned off.
    let (request, compose) = request(Vec::new());
    let map_px = RACE_RASTER_PX;
    let rgb = line_art_rgb(map_px[0], map_px[1]);
    let pdf = pdf_document(&request, &compose, &rgb, map_px, &PrintFonts::none(), &[])
        .expect("a document");
    let text = String::from_utf8_lossy(&pdf);
    assert!(pdf.starts_with(b"%PDF-"));
    assert!(text.contains("/Subtype /Image"));
    assert!(text.contains("%%EOF"));
    assert!(
        !text.contains("/Filter /DCTDecode"),
        "a line-art raster must lose the size race and stay FlateDecode: {text}",
    );
    assert!(text.contains("/Filter /FlateDecode"));
}

#[test]
fn photo_jpeg_off_forces_flatedecode_even_for_photographic_content() {
    let (mut request, compose) = request(Vec::new());
    request.options.photo_jpeg = false;
    let map_px = RACE_RASTER_PX;
    let rgb = photographic_rgb(map_px[0], map_px[1]);
    let pdf = pdf_document(&request, &compose, &rgb, map_px, &PrintFonts::none(), &[])
        .expect("a document");
    assert!(
        !String::from_utf8_lossy(&pdf).contains("/Filter /DCTDecode"),
        "photo_jpeg = false must be an unconditional opt-out, content notwithstanding",
    );
}

#[test]
fn jpeg_quality_outside_the_dialogs_range_clamps_instead_of_failing() {
    let (mut request, compose) = request(Vec::new());
    let map_px = RACE_RASTER_PX;
    let rgb = photographic_rgb(map_px[0], map_px[1]);
    // 0 and 255 are outside the documented 1..=100; a corrupt settings file
    // or a hostile CLI flag can still produce them.
    for quality in [0_u8, 1, 85, 100, 255] {
        request.options.jpeg_quality = quality;
        let pdf = pdf_document(&request, &compose, &rgb, map_px, &PrintFonts::none(), &[])
            .unwrap_or_else(|error| panic!("quality {quality} must not fail the export: {error}"));
        assert!(pdf.starts_with(b"%PDF-"), "quality {quality}");
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.contains("%%EOF"), "quality {quality}");
        assert!(
            text.contains("/Filter /DCTDecode"),
            "quality {quality} still wins the race for photographic content",
        );
    }
}
