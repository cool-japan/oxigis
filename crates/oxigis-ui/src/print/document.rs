// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! PDF object-graph assembly: [`pdf_document`] and the embedded-font
//! writer — split out of `print` (a pure move) so the module stays under
//! the 2000-line rule as the v1.2 shaped-text work lands.

use std::sync::Arc;

use image::ExtendedColorType;
use image::codecs::jpeg::JpegEncoder;
use oxigis_core::LabelWeight;
use oxigis_render::mvt::VectorTile as MvtVectorTile;
use oxigis_render::{MapView, TileId};
use pdf_writer::types::{CidFontType, FontFlags, SystemInfo, UnicodeCmap};
use pdf_writer::{Filter, Finish, Name, Pdf, Rect, Ref, Str, TextStr};

use super::{
    PrintFonts, PrintRequest, PrintVectorTiles, class_alpha_name, font, labels, layer_alpha_slots,
    legend, map_box, meta, mvt, north, page_content_planned_with, scalebar,
};

/// Object ids for one embedded font's five PDF objects.
struct EmbeddedFontIds {
    type0: Ref,
    cid: Ref,
    descriptor: Ref,
    file: Ref,
    to_unicode: Ref,
}

/// Groups a CID→width map into the `/W` array's consecutive-run form.
fn width_runs(widths: &std::collections::BTreeMap<u16, f32>) -> Vec<(u16, Vec<f32>)> {
    let mut runs: Vec<(u16, Vec<f32>)> = Vec::new();
    for (&cid, &width) in widths {
        match runs.last_mut() {
            Some((start, values)) if usize::from(*start) + values.len() == usize::from(cid) => {
                values.push(width);
            }
            _ => runs.push((cid, vec![width])),
        }
    }
    runs
}

/// The `/DCTDecode` candidate for `map_rgb` (`map_px[0] × map_px[1]` RGB8):
/// a baseline JPEG at `quality`, clamped to
/// [`super::MIN_JPEG_QUALITY`]..=[`super::MAX_JPEG_QUALITY`] (print v1.8).
///
/// [`None`] on any encoder refusal — in practice only a degenerate `0×0`
/// buffer can trigger one, since [`super::raster_size_px`] keeps every real
/// export's edges far inside JPEG's own 65535-pixel-per-side ceiling.
/// [`pdf_document`] treats [`None`] exactly like a JPEG that lost the size
/// race against `/FlateDecode`: the zlib stream ships instead, so a refusal
/// here can never fail the export.
fn jpeg_candidate(map_rgb: &[u8], map_px: [u32; 2], quality: u8) -> Option<Vec<u8>> {
    let quality = quality.clamp(super::MIN_JPEG_QUALITY, super::MAX_JPEG_QUALITY);
    let mut bytes = Vec::new();
    match JpegEncoder::new_with_quality(&mut bytes, quality).encode(
        map_rgb,
        map_px[0],
        map_px[1],
        ExtendedColorType::Rgb8,
    ) {
        Ok(()) => Some(bytes),
        Err(error) => {
            tracing::warn!(
                %error,
                "oxigis-ui print: JPEG encoding of the map raster failed; keeping /FlateDecode",
            );
            None
        }
    }
}

/// Assembles the finished single-page PDF.
///
/// `map_rgb` is [`super::compose_map_rgb`]'s output for `compose` — `width × height
/// × 3` bytes. The content stream, the font programs and the `/ToUnicode`
/// maps are always `/FlateDecode` zlib streams from `oxiarc-deflate`. The map
/// image is too, UNLESS [`super::PrintOptions::photo_jpeg`] is on (the
/// default) AND encoding `map_rgb` as a baseline JPEG at
/// [`super::PrintOptions::jpeg_quality`] actually comes out smaller than the
/// zlib stream — print v1.8's `/DCTDecode` path, raced honestly against
/// `/FlateDecode` on every export rather than switched on a content guess, so
/// a flat, line-art-like or screenshot-like raster keeps the zlib stream
/// exactly as every pre-v1.8 export did. `fonts` is the shell's font chain
/// ([`PrintFonts`]); when it is empty or unusable the text degrades to the v1
/// Base-14 Helvetica path (non-ASCII prints as `?`).
///
/// # Errors
///
/// A message naming the stream that would not compress — which
/// `oxiarc-deflate` only answers for pathological inputs — or a raster buffer
/// whose size does not match the view.
pub fn pdf_document(
    request: &PrintRequest,
    compose: &MapView,
    map_rgb: &[u8],
    map_px: [u32; 2],
    fonts: &PrintFonts,
    vector_tiles: &[(TileId, Arc<MvtVectorTile>)],
) -> Result<Vec<u8>, String> {
    pdf_document_with(
        request,
        compose,
        map_rgb,
        map_px,
        fonts,
        &PrintVectorTiles {
            single: vector_tiles,
            stack: &[],
        },
    )
}

/// [`pdf_document`] over the whole tiled stack (compositing v1.6).
///
/// Identical in every other way; `tiles` simply carries one decoded-tile list
/// per vector-tile source instead of one for `PrintRequest::vector`. A request
/// whose stack fits the legacy slots
/// ([`PrintRequest::stack_fits_legacy_slots`]) reads only `tiles.single`, so
/// calling this is never a behaviour change for such a page.
///
/// # Errors
///
/// The same two as [`pdf_document`]: a stream `oxiarc-deflate` would not
/// compress, or a raster buffer that does not match the view.
pub fn pdf_document_with(
    request: &PrintRequest,
    compose: &MapView,
    map_rgb: &[u8],
    map_px: [u32; 2],
    fonts: &PrintFonts,
    tiles: &PrintVectorTiles<'_>,
) -> Result<Vec<u8>, String> {
    let expected = map_px[0] as usize * map_px[1] as usize * 3;
    if map_rgb.len() != expected {
        return Err(format!(
            "the raster buffer holds {} bytes where {}x{} RGB needs {expected}",
            map_rgb.len(),
            map_px[0],
            map_px[1],
        ));
    }
    let map_box = map_box(&request.options);
    // Every string the page furniture shows, produced by the SAME calls the
    // painter makes: a character with no CID behind it is dropped silently by
    // the synthetic run path, so the plan and the page have to be built from
    // one source. `scalebar::plate_texts` covers the `0` tick, the distance
    // and the representative fraction; `legend::texts` covers the rows the
    // legend will actually show, overflow row included.
    let furniture: Vec<String> = {
        let mut texts = Vec::new();
        if request.options.scale_bar
            && let Some(bar) =
                scalebar::scale_bar_with(compose, &map_box, request.options.scale_units)
        {
            texts.extend(scalebar::plate_texts(&bar, &request.options));
        }
        if north::plate_box(&request.options, &map_box).is_some() {
            texts.push(north::NORTH_LABEL.to_string());
        }
        texts.extend(legend::texts(request, compose, &map_box));
        texts
    };
    // The plan must cover every string the page shows — the labels included,
    // or their characters would have no CIDs at placement time.
    let label_texts = labels::texts(request, compose, &map_box);
    // Streamed vector-tile labels join the plan BEFORE placement, exactly
    // like the local ones — the same candidate builder feeds both, so a
    // placed label can never miss its CIDs.
    // EVERY source's strings, not only the legacy slot's: a character with no
    // CID behind it is dropped silently at placement time, so a second tileset
    // whose labels never reached the plan would simply lose them.
    let mvt_label_texts: Vec<labels::PlanText> = request
        .vector_sources()
        .into_iter()
        .flat_map(|(entry, config)| labels::mvt_texts(&config.paints, tiles.of(entry)))
        .collect();
    // Page furniture — title, attribution, scale bar, north arrow, legend —
    // is always Regular; only a Symbol style can ask for Bold (print/text
    // v1.4, D-W5).
    let mut plan_texts: Vec<(LabelWeight, &str)> = vec![
        (LabelWeight::Regular, request.title.as_str()),
        (LabelWeight::Regular, request.attribution.as_str()),
    ];
    plan_texts.extend(
        furniture
            .iter()
            .map(|text| (LabelWeight::Regular, text.as_str())),
    );
    plan_texts.extend(
        label_texts
            .iter()
            .map(|planned| (planned.weight, planned.text.as_str())),
    );
    plan_texts.extend(
        mvt_label_texts
            .iter()
            .map(|planned| (planned.weight, planned.text.as_str())),
    );
    // Every string a Symbol style asked to set VERTICALLY (print v1.6). It is
    // in `plan_texts` too: the ladder can refuse a column and the label then
    // prints horizontally, so both forms have to exist in the plan.
    let vertical_labels: Vec<(LabelWeight, &str)> = label_texts
        .iter()
        .chain(mvt_label_texts.iter())
        .filter(|planned| planned.vertical)
        .map(|planned| (planned.weight, planned.text.as_str()))
        .collect();
    // The vertical title is planned only when the dialog asked for it; with
    // the option off `plan` never runs the vertical pass at all, so the
    // document is byte-identical to a pre-v1.4 export.
    let vertical_title = request
        .options
        .vertical_title
        .then_some(request.title.as_str());
    let plan = font::plan_with_verticals(fonts, &plan_texts, vertical_title, &vertical_labels);
    if let Some(plan) = &plan
        && plan.substituted > 0
    {
        tracing::warn!(
            characters = plan.substituted,
            "oxigis-ui print: characters no font in the chain covers print as '?'",
        );
    }
    let ops = page_content_planned_with(request, compose, &map_box, plan.as_ref(), tiles);
    let content_stream = oxiarc_deflate::zlib::zlib_compress(&ops, 6)
        .map_err(|error| format!("could not compress the content stream: {error}"))?;
    let image_stream = oxiarc_deflate::zlib::zlib_compress(map_rgb, 6)
        .map_err(|error| format!("could not compress the map image: {error}"))?;
    // print v1.8: race a `/DCTDecode` JPEG against the `/FlateDecode` zlib
    // stream above and keep whichever is smaller. `photo_jpeg = false` skips
    // the attempt outright (an explicit request for the lossless path); a
    // raster the encoder refuses — in practice only a degenerate `0×0`
    // buffer, since `raster_size_px` keeps every real export far inside
    // JPEG's own 65535-pixel-per-side ceiling — or one JPEG does not shrink
    // both fall straight through to the zlib stream, so turning this option
    // on can never grow a page.
    let jpeg_stream = if request.options.photo_jpeg {
        jpeg_candidate(map_rgb, map_px, request.options.jpeg_quality)
    } else {
        None
    };
    let (image_bytes, image_filter): (&[u8], Filter) = match &jpeg_stream {
        Some(jpeg) if jpeg.len() < image_stream.len() => (jpeg.as_slice(), Filter::DctDecode),
        _ => (image_stream.as_slice(), Filter::FlateDecode),
    };

    let catalog_id = Ref::new(1);
    let pages_id = Ref::new(2);
    let page_id = Ref::new(3);
    let content_id = Ref::new(4);
    let image_id = Ref::new(5);
    let font_id = Ref::new(6);
    // The `/Info` dictionary takes one of the three ids the layout has always
    // left free between the fixed objects and the ExtGState block, so adding
    // it moves nothing else in the file.
    let info_id = Ref::new(7);
    let first_gs_id = 10;
    // Every alpha state a layer names: the base always (the label pass
    // paints under it), plus each present family carrying an override —
    // ONE list read by both the registration and the object writes, and
    // exactly one state per layer when nothing is overridden, so the ids
    // (and the bytes) of a pre-v1.3 document are unchanged.
    let alpha_slots: Vec<(String, f32)> = request
        .layers
        .iter()
        .enumerate()
        .flat_map(|(index, layer)| {
            layer_alpha_slots(layer)
                .into_iter()
                .map(move |(slot, class, alpha)| (class_alpha_name(index, slot, class), alpha))
        })
        .collect();
    // Embedded-font objects start after the ExtGState block; gaps in the id
    // space are legal (the xref writes free-list entries for them).
    let first_font_id = first_gs_id + alpha_slots.len() as i32;
    let embedded_ids: Vec<EmbeddedFontIds> = plan
        .as_ref()
        .map(|plan| {
            (0..plan.fonts.len() as i32)
                .map(|index| {
                    let base = first_font_id + index * 5;
                    EmbeddedFontIds {
                        type0: Ref::new(base),
                        cid: Ref::new(base + 1),
                        descriptor: Ref::new(base + 2),
                        file: Ref::new(base + 3),
                        to_unicode: Ref::new(base + 4),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let resource_names: Vec<String> = (0..embedded_ids.len())
        .map(|index| format!("F{}", index + 1))
        .collect();
    // The streamed vector layers' per-rule alpha states follow the fonts —
    // one block per SOURCE, so a page holding three tilesets registers all
    // three rule tables rather than the first one's.
    let first_vector_gs = first_font_id + embedded_ids.len() as i32 * 5;
    let vector_alphas: Vec<(String, f32)> = request
        .vector_sources()
        .into_iter()
        .flat_map(|(entry, config)| {
            config
                .paints
                .iter()
                .enumerate()
                .map(move |(rule, paint)| {
                    (mvt::rule_alpha_name(entry, rule), mvt::rule_alpha(paint))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut pdf = Pdf::new();
    {
        let mut catalog = pdf.catalog(catalog_id);
        catalog.pages(pages_id);
        // The page's halo passes are already wrapped as `/Artifact` and its
        // bidi lines as `/Span <</ActualText …>>`; `/MarkInfo` is the
        // catalog-level statement that those marks are deliberate, which is
        // what a conformant extractor (and PDF/A-2a) looks for.
        catalog.mark_info().marked(true);
    }
    // `/Info`: the title a viewer shows instead of the file name, the
    // producer an archive records, and when the sheet was drawn.
    if request.options.document_metadata {
        let producer = meta::producer();
        let mut info = pdf.document_info(info_id);
        if !request.title.is_empty() {
            info.title(TextStr(&request.title));
        }
        info.creator(TextStr(meta::CREATOR));
        info.producer(TextStr(&producer));
        if let Some(date) = meta::creation_date(&request.options) {
            info.creation_date(date);
        }
    }
    pdf.pages(pages_id).kids([page_id]).count(1);
    {
        let mut page = pdf.page(page_id);
        let [page_w, page_h] = request.options.page_size_pt();
        page.media_box(Rect::new(0.0, 0.0, page_w, page_h));
        page.parent(pages_id);
        page.contents(content_id);
        let mut resources = page.resources();
        resources.x_objects().pair(Name(b"Im0"), image_id);
        let mut fonts_dict = resources.fonts();
        fonts_dict.pair(Name(b"F0"), font_id);
        for (name, ids) in resource_names.iter().zip(&embedded_ids) {
            fonts_dict.pair(Name(name.as_bytes()), ids.type0);
        }
        fonts_dict.finish();
        let mut states = resources.ext_g_states();
        for (offset, (name, _)) in alpha_slots.iter().enumerate() {
            states.pair(Name(name.as_bytes()), Ref::new(first_gs_id + offset as i32));
        }
        for (offset, (name, _)) in vector_alphas.iter().enumerate() {
            states.pair(
                Name(name.as_bytes()),
                Ref::new(first_vector_gs + offset as i32),
            );
        }
        states.finish();
    }
    {
        let mut stream = pdf.stream(content_id, &content_stream);
        stream.filter(Filter::FlateDecode);
    }
    {
        let mut image = pdf.image_xobject(image_id, image_bytes);
        image.filter(image_filter);
        image.width(map_px[0] as i32);
        image.height(map_px[1] as i32);
        image.color_space().device_rgb();
        image.bits_per_component(8);
    }
    pdf.type1_font(font_id)
        .base_font(Name(b"Helvetica"))
        .encoding_predefined(Name(b"WinAnsiEncoding"));
    if let Some(plan) = &plan {
        for (planned, ids) in plan.fonts.iter().zip(&embedded_ids) {
            write_embedded_font(&mut pdf, planned, ids)?;
        }
    }
    for (offset, (_, alpha)) in alpha_slots.iter().enumerate() {
        pdf.ext_graphics(Ref::new(first_gs_id + offset as i32))
            .non_stroking_alpha(*alpha)
            .stroking_alpha(*alpha);
    }
    for (offset, (_, alpha)) in vector_alphas.iter().enumerate() {
        pdf.ext_graphics(Ref::new(first_vector_gs + offset as i32))
            .non_stroking_alpha(*alpha)
            .stroking_alpha(*alpha);
    }
    Ok(pdf.finish())
}

/// Writes one embedded font's five objects: the `/Type0` wrapper, the CID
/// font, the descriptor, the font program stream, and the `/ToUnicode` map.
fn write_embedded_font(
    pdf: &mut Pdf,
    planned: &font::PlannedFont,
    ids: &EmbeddedFontIds,
) -> Result<(), String> {
    let base = Name(planned.base_font.as_bytes());
    pdf.type0_font(ids.type0)
        .base_font(base)
        .encoding_predefined(Name(b"Identity-H"))
        .descendant_font(ids.cid)
        .to_unicode(ids.to_unicode);
    {
        let mut cid = pdf.cid_font(ids.cid);
        cid.subtype(if planned.cff {
            CidFontType::Type0
        } else {
            CidFontType::Type2
        });
        cid.base_font(base);
        // Adobe-Identity-0, unconditionally — including for a CID-keyed CFF
        // whose own ROS says Adobe-Japan1. The descendant's `/CIDSystemInfo`
        // has to be compatible with the CMap's, the CMap is `/Identity-H`
        // (which IS Adobe-Identity-0), and declaring a different ordering
        // here is the incompatibility rather than the fix. What makes glyph
        // selection right on a CID-keyed program is that the CIDs the content
        // stream emits come from that program's charset — see
        // `super::cff::charset` and `subset::subset_face`.
        cid.system_info(SystemInfo {
            registry: Str(b"Adobe"),
            ordering: Str(b"Identity"),
            supplement: 0,
        });
        cid.font_descriptor(ids.descriptor);
        cid.default_width(1000.0);
        {
            let mut widths = cid.widths();
            for (start, values) in width_runs(&planned.widths) {
                widths.consecutive(start, values.iter().copied());
            }
        }
        if !planned.cff {
            cid.cid_to_gid_map_predefined(Name(b"Identity"));
        }
    }
    {
        let mut descriptor = pdf.font_descriptor(ids.descriptor);
        descriptor.name(base);
        let mut flags = FontFlags::SYMBOLIC;
        if planned.metrics.italic {
            flags |= FontFlags::ITALIC;
        }
        if planned.metrics.monospaced {
            flags |= FontFlags::FIXED_PITCH;
        }
        descriptor.flags(flags);
        descriptor.bbox(Rect::new(
            planned.metrics.bbox[0],
            planned.metrics.bbox[1],
            planned.metrics.bbox[2],
            planned.metrics.bbox[3],
        ));
        descriptor.italic_angle(planned.metrics.italic_angle);
        descriptor.ascent(planned.metrics.ascent);
        descriptor.descent(planned.metrics.descent);
        descriptor.cap_height(planned.metrics.cap_height);
        descriptor.stem_v(planned.metrics.stem_v);
        descriptor.weight(planned.metrics.weight);
        if planned.cff {
            descriptor.font_file3(ids.file);
        } else {
            descriptor.font_file2(ids.file);
        }
    }
    let program = oxiarc_deflate::zlib::zlib_compress(&planned.subset, 6)
        .map_err(|error| format!("could not compress a font program: {error}"))?;
    {
        let mut stream = pdf.stream(ids.file, &program);
        stream.filter(Filter::FlateDecode);
        if planned.cff {
            // A bare CFF program: `/FontFile3` requires the subtype.
            stream.pair(Name(b"Subtype"), Name(b"CIDFontType0C"));
        } else {
            // `/Length1` = the uncompressed sfnt length, per `/FontFile2`.
            stream.pair(
                Name(b"Length1"),
                i32::try_from(planned.subset.len()).unwrap_or(i32::MAX),
            );
        }
    }
    let mut cmap = UnicodeCmap::<u16>::new(
        Name(b"OxiGIS-UCS"),
        SystemInfo {
            registry: Str(b"Adobe"),
            ordering: Str(b"UCS"),
            supplement: 0,
        },
    );
    // One bfchar per CID: a ligature CID maps to its WHOLE source text
    // (`pair_with_multiple` concatenates the UTF-16BE code units), so
    // copy-paste out of the PDF stays exact for shaped output too.
    for (&cid, text) in &planned.to_unicode {
        cmap.pair_with_multiple(cid, text.chars());
    }
    let to_unicode = oxiarc_deflate::zlib::zlib_compress(&cmap.finish().into_vec(), 6)
        .map_err(|error| format!("could not compress a /ToUnicode map: {error}"))?;
    pdf.stream(ids.to_unicode, &to_unicode)
        .filter(Filter::FlateDecode);
    Ok(())
}
