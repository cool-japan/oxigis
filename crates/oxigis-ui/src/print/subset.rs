// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Subsetting and `/FontDescriptor` derivation for the PDF export — a PURE
//! MOVE out of `print/font.rs` (print/text v1.4, D-W5), which crossed its
//! line budget once the weight ladder landed. Nothing here changed behaviour
//! in the move; `subset_face`'s `weight` parameter is the one v1.4 addition
//! and is documented on the function.
//!
//! # The subset is PDF-only
//!
//! [`pdf_subset_options`] drops the layout tables, the non-essential `name`
//! records and every variation table, the export requests no codepoints at
//! all — so the embedded program carries an EMPTY `cmap` — and the glyphs are
//! renumbered. That is exactly right for an `Identity-H` composite font,
//! which selects glyphs through the PDF's own CMap, and it makes the output
//! unusable as a standalone font file. Metrics for the `/FontDescriptor` are
//! therefore read from the *original* face, never the subset.
//!
//! # One call, one CID source
//!
//! Both rungs end in the same
//! [`oxifont_subset::subset_with_gid_set_at_face_mapped`] — the `_at_face`
//! form, because a chain entry may be a whole `.ttc` — and the `SubsetGidMap`
//! that call returns is the ONLY source of CIDs. That entry point does not
//! insert `.notdef`; this module does, before calling it.

use std::collections::{BTreeMap, BTreeSet};

use oxifont_subset::SubsetOptions;
use oxigis_core::LabelWeight;
use ttf_parser::{Face, GlyphId, RawFace, Tag};

use super::font::{FaceRole, usable_face};
use super::instance;

/// `/FontDescriptor` numbers, already scaled to 1000/em text space.
pub struct DescriptorMetrics {
    /// `/Ascent`.
    pub ascent: f32,
    /// `/Descent` (negative).
    pub descent: f32,
    /// `/CapHeight`.
    pub cap_height: f32,
    /// `/FontBBox` as `[x_min, y_min, x_max, y_max]`.
    pub bbox: [f32; 4],
    /// `/ItalicAngle`, degrees counter-clockwise from vertical.
    pub italic_angle: f32,
    /// `/StemV` — an estimate from the weight class; only consulted for
    /// substitution, which cannot happen for an embedded font.
    pub stem_v: f32,
    /// `/FontWeight` — the weight of the instance actually embedded (a
    /// variable face embeds at its DEFAULT instance, so this is the `fvar`
    /// `wght` default, falling back to OS/2), rounded to the 100…900 grid
    /// PDF 32000-1 Table 122 requires.
    pub weight: u16,
    /// Sets the ITALIC flag.
    pub italic: bool,
    /// Sets the FIXED_PITCH flag.
    pub monospaced: bool,
}

/// The weight of the instance the subset embeds when nothing was pinned:
/// `drop_variations` deletes `fvar`/`gvar`, leaving the face's DEFAULT
/// master, whose authoritative weight is the `fvar` `wght` axis default —
/// OS/2 `usWeightClass` is the fallback and the answer for static faces.
pub(super) fn default_weight(face: &Face<'_>) -> f32 {
    face.variation_axes()
        .into_iter()
        .find(|axis| axis.tag == Tag::from_bytes(b"wght"))
        .map(|axis| axis.def_value)
        .unwrap_or_else(|| f32::from(face.weight().to_number()))
        .clamp(1.0, 1000.0)
}

/// Rounds a weight onto the `/FontWeight` grid: multiples of 100 in
/// `100..=900`.
pub(super) fn font_weight_grid(weight: f32) -> u16 {
    let rounded = (weight / 100.0).round() * 100.0;
    (rounded.clamp(100.0, 900.0)) as u16
}

/// The one [`SubsetOptions`] value the export builds a font program with,
/// identical on every rung.
///
/// - Layout tables OFF: shaping happens at plan time and the content stream
///   carries final glyph ids, so `GSUB`/`GPOS`/`GDEF` can match nothing —
///   measured 32 720 B → 1 804 B on a 9-glyph Noto subset.
/// - Names OFF: name ids 0-6 (the copyright and licence records) stay, the
///   rest go.
/// - Variations OFF: the embedded program must describe exactly ONE static
///   instance, or `/FontWeight` and the `/W` array claim a location the
///   viewer is free not to draw. A no-op once the instancer has run, whose
///   output carries no variation table to begin with.
///
/// Hints are deliberately NOT stripped: a non-instanced embed carries the
/// face's own outlines, which its `cvt `/`fpgm`/`prep` were tuned against.
/// The instancer drops the hint tables together with the per-glyph
/// instruction streams, which must move as one.
fn pdf_subset_options() -> SubsetOptions {
    SubsetOptions::default()
        .retain_layout_tables(false)
        .retain_names(false)
        .drop_variations(true)
}

/// One face of the chain, subsetted and ready to embed.
pub struct PlannedFont {
    /// The subset font program: a whole sfnt for glyf faces (`/FontFile2`),
    /// or the bare `CFF ` table for CFF faces (`/FontFile3`,
    /// `/Subtype /CIDFontType0C`).
    pub subset: Vec<u8>,
    /// Whether this is a CFF face (`CIDFontType0`) rather than glyf
    /// (`CIDFontType2`).
    pub cff: bool,
    /// `/BaseFont`: `TAG+PostScriptName`, deterministic per subset.
    pub base_font: String,
    /// Character → CID (== subset glyph id) for every character this face
    /// draws on the page — the unshaped-fallback and substitution path.
    pub cids: BTreeMap<char, u16>,
    /// ORIGINAL glyph id → CID, for every glyph in the subset request —
    /// the shaping pass's translation table (a ligature gid appears here
    /// and never in [`Self::cids`]).
    pub gids: BTreeMap<u16, u16>,
    /// CID → `/W` width in 1000/em units — from the EMBEDDED program's
    /// metric space (the instanced hmtx when the face was instanced).
    pub widths: BTreeMap<u16, f32>,
    /// CID → the DEFAULT instance's width in 1000/em units — the space
    /// swash's shaped advances live in, so the kern delta is computed
    /// against THIS, never against [`Self::widths`] (a naive
    /// `W − shaped` under instancing overlaps glyphs by up to 12.65 % em,
    /// measured). Equal to `widths` for a non-instanced face, which is
    /// what reduces the arithmetic to v1.2's algebraically.
    pub kern_base: BTreeMap<u16, f32>,
    /// CID → the exact source text it stands for — the `/ToUnicode` source.
    /// One char is the ordinary case; several is a ligature. Seeded from
    /// [`Self::cids`] in ascending-char order (a glyph shared by two
    /// characters extracts as the LOWER codepoint — strictly more defined
    /// than the old last-write-wins), then overlaid by shaped ligature
    /// texts where absent.
    pub to_unicode: BTreeMap<u16, String>,
    /// `/FontDescriptor` numbers.
    pub metrics: DescriptorMetrics,
}

/// Subsets one face down to `gids` (plus `.notdef`) and derives everything
/// the PDF needs from it — `gids` is the UNION of the cmap-assigned glyphs
/// (`chars`) and every glyph shaping produced, so ligature glyphs get a
/// CID and a `/W` entry too. [`None`] on any failure — the caller degrades.
///
/// A [`FaceRole::PrintOnly`] variable glyf face is normalised to the fvar
/// instance nearest weight 400 (the shipping defect this exists for:
/// NotoSansJP-VF defaults to wght 100 — 44 % of the Regular ink, measured).
/// Three-rung ladder, so no export ever fails because of instancing:
/// no instance chosen ⇒ the plain subset of the original bytes;
/// [`oxifont_subset::instance`] Ok ⇒ the same subset over the pinned static
/// program; Err ⇒ warn + the plain subset.
pub(super) fn subset_face(
    bytes: &[u8],
    gids: &[u16],
    chars: &[(char, u16)],
    role: FaceRole,
    weight: LabelWeight,
) -> Option<PlannedFont> {
    let face = usable_face(bytes)?;
    let upem = f32::from(face.units_per_em());
    if upem <= 0.0 {
        return None;
    }
    let scale = 1000.0 / upem;
    let cff = face.tables().cff.is_some();

    // CFF is never instanced (`instance()` refuses CFF/CFF2 charstrings);
    // ScreenShared is never instanced (page/screen parity).
    let chosen = if role == FaceRole::PrintOnly && !cff && face.is_variable() {
        instance::raw_fvar(bytes).and_then(|fvar| {
            instance::choose_instance(&face, fvar, instance::target_weight(weight))
        })
    } else {
        None
    };

    // Pin the WHOLE face at the chosen location first. The pinned program
    // keeps every glyph id, so the subset call below is the same one on both
    // rungs — only its input bytes differ, and the two never have to agree
    // about a fused instance-and-subset numbering.
    let instanced_bytes = chosen.as_ref().and_then(|inst| {
        match oxifont_subset::instance(bytes, 0, &inst.coordinates) {
            Ok(pinned) => Some(pinned),
            Err(error) => {
                tracing::warn!(
                    %error,
                    "oxigis-ui print: instancing failed; the DEFAULT instance embeds instead",
                );
                None
            }
        }
    });
    let instanced = instanced_bytes.is_some();
    let source: &[u8] = instanced_bytes.as_deref().unwrap_or(bytes);
    // `.notdef` is NOT inserted by this entry point — the caller owns it.
    // Without it new gid 0 becomes the lowest requested glyph (measured: a
    // leading space) and every printed character shifts onto a CID the PDF
    // reserves for `.notdef`.
    let mut gid_set: BTreeSet<u16> = gids.iter().copied().collect();
    gid_set.insert(0);
    // The `_at_face` form is MANDATORY: a chain entry may be a whole `.ttc`
    // (see `PrintFace::bytes`), which the offset-0 entry points refuse by
    // documented convention. A pinned program is always a single-face sfnt,
    // so face 0 is right on either rung.
    let (subset, gid_map, cff_verbatim) = match oxifont_subset::subset_with_gid_set_at_face_mapped(
        source,
        0,
        &gid_set,
        &BTreeMap::new(),
        &pdf_subset_options(),
    ) {
        Ok((subset, stats, map)) => (subset, map, stats.cff_charstrings_verbatim),
        Err(error) => {
            tracing::warn!(%error, "oxigis-ui print: oxifont-subset refused the face");
            return None;
        }
    };
    // The verbatim-CFF fallback embeds the ORIGINAL program, so its CIDs are
    // the program's own — identity for a name-keyed CFF (PDF 32000-1
    // §9.7.4.2: a CIDFontType0 whose program is not CID-keyed uses the CID
    // directly as the glyph index), the CHARSET for a CID-keyed one. A
    // CID-keyed program whose charset this export cannot read is refused
    // here: its characters degrade to '?', which is honest, where an assumed
    // identity would silently print the wrong charstrings.
    let verbatim_charset = if cff_verbatim && cff {
        match RawFace::parse(bytes, 0)
            .ok()
            .and_then(|raw| raw.table(Tag::from_bytes(b"CFF ")))
            .and_then(super::cff::charset)
        {
            Some(found) => found,
            None => {
                tracing::warn!(
                    "oxigis-ui print: a CID-keyed CFF face has no charset this export can \
                     read, so its glyph selection cannot be described; the face is skipped",
                );
                return None;
            }
        }
    } else {
        super::cff::Charset::NameKeyed
    };
    // ONE CID source: the map that same subset call returned, or — on the
    // verbatim rung — the embedded program's own charset. Deriving the CIDs
    // anywhere else would desynchronise `/W` from the `Tj` strings the moment
    // the composite closure pulls a component in, with nothing in the suite
    // to catch it.
    let cid_of = |old_gid: u16| -> Option<u16> {
        if cff_verbatim {
            return match &verbatim_charset {
                super::cff::Charset::NameKeyed => Some(old_gid),
                super::cff::Charset::CidKeyed(cids) => cids.get(usize::from(old_gid)).copied(),
            };
        }
        gid_map.new_gid(old_gid)
    };
    // The instanced program's OWN metrics feed `/W` and the bbox — its
    // subset gid IS the CID, so the lookup is direct. Extracted into owned
    // values here so the borrow ends before `subset` moves on.
    let mut instanced_widths: BTreeMap<u16, u16> = BTreeMap::new();
    let mut instanced_bbox: Option<ttf_parser::Rect> = None;
    if instanced && let Ok(inst) = Face::parse(&subset, 0) {
        for &old_gid in gids {
            if let Some(cid) = cid_of(old_gid)
                && let Some(advance) = inst.glyph_hor_advance(GlyphId(cid))
            {
                instanced_widths.insert(cid, advance);
            }
        }
        // `head` is copied verbatim and MVAR is never applied, so the
        // default-instance box can be too SMALL for instanced outlines
        // (measured: +40 units of y_max at wght 700). Union the instanced
        // glyphs' boxes in — never smaller than today, never a clip.
        let mut union = inst.global_bounding_box();
        for gid in 0..inst.number_of_glyphs() {
            if let Some(glyph_box) = inst.glyph_bounding_box(GlyphId(gid)) {
                union.x_min = union.x_min.min(glyph_box.x_min);
                union.y_min = union.y_min.min(glyph_box.y_min);
                union.x_max = union.x_max.max(glyph_box.x_max);
                union.y_max = union.y_max.max(glyph_box.y_max);
            }
        }
        instanced_bbox = Some(union);
    }
    let subset = if cff {
        // `/FontFile3` with `/Subtype /CIDFontType0C` takes the bare CFF
        // table, not the `OTTO` sfnt wrapper the subset arrives in.
        let source: &[u8] = if cff_verbatim {
            // The CFF rewriter copied the source charstrings while the rest
            // of the subset was renumbered, so its glyphs would draw the
            // wrong characters. Embed the ORIGINAL face's table and take the
            // CIDs from the program itself (see `cid_of`) — correct, at the
            // cost of the whole font's charstrings.
            tracing::warn!(
                cid_keyed = matches!(verbatim_charset, super::cff::Charset::CidKeyed(_)),
                "oxigis-ui print: the CFF rewriter fell back to verbatim charstrings; \
                 embedding the ORIGINAL CFF table and selecting glyphs through the \
                 program's own CID assignment",
            );
            bytes
        } else {
            &subset
        };
        RawFace::parse(source, 0)
            .ok()?
            .table(Tag::from_bytes(b"CFF "))?
            .to_vec()
    } else {
        subset
    };

    let mut planned_gids = BTreeMap::new();
    let mut widths = BTreeMap::new();
    let mut kern_base = BTreeMap::new();
    // A CID names exactly one glyph. Two requested glyphs claiming one CID
    // means the charset is malformed, and the second would silently overwrite
    // the first's `/W` — refuse the face instead.
    let mut claimed: BTreeMap<u16, u16> = BTreeMap::new();
    for &old_gid in gids {
        let cid = cid_of(old_gid)?;
        if let Some(&other) = claimed.get(&cid) {
            tracing::warn!(
                cid,
                first = other,
                second = old_gid,
                "oxigis-ui print: two glyphs claim one CID; the face is skipped",
            );
            return None;
        }
        claimed.insert(cid, old_gid);
        planned_gids.insert(old_gid, cid);
        let advance = face.glyph_hor_advance(GlyphId(old_gid)).unwrap_or(0);
        kern_base.insert(cid, f32::from(advance) * scale);
        let embedded = instanced_widths.get(&cid).copied().unwrap_or(advance);
        widths.insert(cid, f32::from(embedded) * scale);
    }
    let mut cids = BTreeMap::new();
    let mut to_unicode: BTreeMap<u16, String> = BTreeMap::new();
    for &(ch, old_gid) in chars {
        let cid = planned_gids.get(&old_gid).copied()?;
        cids.insert(ch, cid);
    }
    // Seeded ascending by char, first-wins: a glyph two characters share
    // (space and NBSP, commonly) extracts as the LOWER codepoint.
    for (&ch, &cid) in &cids {
        to_unicode.entry(cid).or_insert_with(|| ch.to_string());
    }

    let mut ps_name = postscript_name(&face);
    let weight = match &chosen {
        Some(inst) if instanced => {
            tracing::info!(
                name = %ps_name,
                instance = inst.name.as_deref().unwrap_or("(unnamed)"),
                weight = inst.weight,
                "oxigis-ui print: variable face embedded at its nearest-Regular instance",
            );
            // The tag/name must differ from the default-instance subset of
            // the same glyph set, or two exports collide.
            match inst.name.as_deref() {
                Some(name) => {
                    let suffix: String = name.chars().filter(char::is_ascii_alphanumeric).collect();
                    ps_name = format!("{ps_name}-{suffix}");
                }
                None => ps_name = format!("{ps_name}-w{}", inst.weight.round()),
            }
            inst.weight
        }
        _ => {
            if face.is_variable() {
                // Nothing was pinned — the nearest instance IS the default,
                // or the face is ScreenShared/CFF — so `drop_variations`
                // leaves the static DEFAULT master. Say so once instead of
                // printing silently.
                tracing::info!(
                    name = %ps_name,
                    axes = face.variation_axes().len(),
                    weight = default_weight(&face),
                    "oxigis-ui print: variable face embedded at its DEFAULT instance",
                );
            }
            default_weight(&face)
        }
    };
    let tag = subset_tag(&ps_name, gids, 0);
    let mut bbox = face.global_bounding_box();
    if let Some(union) = instanced_bbox {
        bbox.x_min = bbox.x_min.min(union.x_min);
        bbox.y_min = bbox.y_min.min(union.y_min);
        bbox.x_max = bbox.x_max.max(union.x_max);
        bbox.y_max = bbox.y_max.max(union.y_max);
    }
    let cap_height = face
        .capital_height()
        .map_or(f32::from(face.ascender()) * scale * 0.7, |height| {
            f32::from(height) * scale
        });
    Some(PlannedFont {
        subset,
        cff,
        base_font: format!("{tag}+{ps_name}"),
        cids,
        gids: planned_gids,
        widths,
        kern_base,
        to_unicode,
        metrics: DescriptorMetrics {
            ascent: f32::from(face.ascender()) * scale,
            descent: f32::from(face.descender()) * scale,
            cap_height,
            bbox: [
                f32::from(bbox.x_min) * scale,
                f32::from(bbox.y_min) * scale,
                f32::from(bbox.x_max) * scale,
                f32::from(bbox.y_max) * scale,
            ],
            italic_angle: face.italic_angle(),
            // The same curve as v1.1, fed the honest input: for a VF, OS/2
            // and the fvar default are allowed to disagree, and the embedded
            // program follows fvar.
            stem_v: weight / 5.0,
            weight: font_weight_grid(weight),
            italic: face.is_italic(),
            monospaced: face.is_monospaced(),
        },
    })
}

/// The face's PostScript name (name id 6), reduced to the characters a PDF
/// name can carry cleanly; `"Embedded"` when the face has none.
fn postscript_name(face: &Face<'_>) -> String {
    let names = face.names();
    let raw = (0..names.len())
        .filter_map(|index| names.get(index))
        .find(|name| name.name_id == ttf_parser::name_id::POST_SCRIPT_NAME && name.is_unicode())
        .and_then(|name| name.to_string());
    let cleaned: String = raw
        .unwrap_or_default()
        .chars()
        .filter(|ch| {
            ch.is_ascii_graphic()
                && !matches!(
                    ch,
                    '(' | ')' | '<' | '>' | '[' | ']' | '{' | '}' | '/' | '%' | '#'
                )
        })
        .take(40)
        .collect();
    if cleaned.is_empty() {
        "Embedded".to_string()
    } else {
        cleaned
    }
}

/// The 6-letter subset tag: FNV-1a 64 over the PostScript name and the
/// sorted used glyph ids — a pure function of the subset, no clock and no
/// randomness, so exports are byte-deterministic. `salt` perturbs a
/// same-document collision.
fn subset_tag(ps_name: &str, old_gids: &[u16], salt: u64) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET ^ salt.wrapping_mul(PRIME);
    let mut eat = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    };
    for byte in ps_name.bytes() {
        eat(byte);
    }
    let mut sorted = old_gids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    for gid in sorted {
        for byte in gid.to_be_bytes() {
            eat(byte);
        }
    }
    let mut tag = String::with_capacity(6);
    for _ in 0..6 {
        tag.push(char::from(b'A' + (hash % 26) as u8));
        hash /= 26;
    }
    tag
}

/// Re-derives colliding `/BaseFont` names with a salt until all are unique
/// (two identical names in one document are a spec violation).
pub(super) fn dedupe_base_font_names(fonts: &mut [PlannedFont]) {
    for index in 1..fonts.len() {
        let mut salt = 0_u64;
        while fonts[..index]
            .iter()
            .any(|other| other.base_font == fonts[index].base_font)
        {
            salt += 1;
            let name = fonts[index].base_font.split_once('+').map_or_else(
                || fonts[index].base_font.clone(),
                |(_, name)| name.to_string(),
            );
            let cids: Vec<u16> = fonts[index].widths.keys().copied().collect();
            fonts[index].base_font = format!("{}+{name}", subset_tag(&name, &cids, salt));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::print::cff::Charset;

    /// The whole CID-keyed path, end to end, on a real pan-CJK face.
    ///
    /// `oxifont-subset` refuses to rewrite a CID-keyed CFF, so the ORIGINAL
    /// program is embedded and the CID the content stream emits has to be
    /// the one that program's charset resolves. Before print v1.6 the export
    /// emitted the glyph id instead — and the goldens in `print::cff` measure
    /// the platform's faces as NON-identity, so that was the wrong
    /// charstring, silently, on the best-ranked CJK face on the machine.
    #[test]
    #[ignore = "reads the platform's CJK fonts; the print-v1.6 CID-keyed golden"]
    fn live_cid_keyed_cff_selects_glyphs_through_the_charset() {
        let candidates = [
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ];
        let mut checked = 0_usize;
        for path in candidates {
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            let Some(face) = usable_face(&bytes) else {
                continue;
            };
            if face.tables().cff.is_none() {
                continue;
            }
            let table = RawFace::parse(&bytes, 0)
                .ok()
                .and_then(|raw| raw.table(Tag::from_bytes(b"CFF ")))
                .expect("a CFF table");
            let Some(Charset::CidKeyed(cids)) = crate::print::cff::charset(table) else {
                continue;
            };
            let gid = face.glyph_index('\u{6771}').expect("the face draws 東").0;
            let expected = cids[usize::from(gid)];
            let planned = subset_face(
                &bytes,
                &[gid],
                &[('\u{6771}', gid)],
                FaceRole::ScreenShared,
                LabelWeight::Regular,
            )
            .expect("a CID-keyed face still plans");
            assert!(planned.cff, "{path}: a CFF face embeds through /FontFile3");
            assert_eq!(
                planned.cids.get(&'\u{6771}').copied(),
                Some(expected),
                "{path}: the CID must come from the charset, not from the glyph id",
            );
            assert_eq!(
                planned.subset, table,
                "{path}: the verbatim rung embeds the ORIGINAL program, so the \
                 charset the viewer reads is the one this CID came from",
            );
            checked += 1;
        }
        assert!(checked > 0, "no platform CID-keyed CFF face was found");
    }
}
