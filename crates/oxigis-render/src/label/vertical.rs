// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Vertical writing for map labels: the run itemiser, the shared script tag
//! and the refusal ladder that turns a CJK string into a stacked column of
//! cells (print/text v1.5, D-A2..D-A6).
//!
//! The UAX #50 property itself lives next door in
//! [`super::vertical_table`] — ONE table for the whole workspace, so the map
//! and the PDF exporter can never disagree about which characters set
//! upright.
//!
//! # Why this crate shapes vertical text itself
//!
//! `oxitext`'s own `FlowDirection::Vertical` is measured unusable for this:
//! it never injects `vert`/`vrt2` (so 「あ、」 comes back with the HORIZONTAL
//! glyph ids), its `pos.1` is a cell-TOP cursor rather than a baseline, and
//! its column width is a hard-coded `font_size * 1.2` with no per-cell
//! centring. What this module calls instead is exactly the request the PDF
//! exporter already ships in production — [`oxitext::SwashShaper`] with
//! [`oxitext::ShapeDirection::Ttb`] and [`vertical_script`] — so the page and
//! the map run one algorithm over one table.
//!
//! # The ladder is all-or-nothing, and a refusal costs nothing
//!
//! Every rung refuses the WHOLE label; the caller then draws the ordinary
//! HORIZONTAL label, byte for byte what it draws today. There is deliberately
//! **no CFF rung** here even though the exporter has one: that refusal is a
//! PDF-instancing concern with no screen meaning, and Noto Sans CJK / Source
//! Han Sans — the best-ranked CJK faces on Linux and macOS — are CFF. A
//! future "unification" that copies the exporter's `CffFace` rung over would
//! silently ban vertical labels on them.
//!
//! # Panic safety, stated rather than hoped
//!
//! swash's Indic code is reachable only through `EngineMode::Complex`, which
//! needs `script.is_complex()`. The vertical path can never ask for a complex
//! script, because the ONLY tag it ever passes to the shaper comes from
//! [`vertical_script`], which is total and returns exactly one of `hang`,
//! `kana` or `hani` — none of which is complex — for every input string,
//! including one made entirely of Devanagari. `vertical_script_is_never_complex`
//! pins that, font-free, over every input the ladder could meet.
//!
//! **This paragraph used to rest on a different and FALSE premise** (corrected
//! 2026-08-11): that "every Brahmic / South-East-Asian character is `R` in
//! UAX #50", so [`VerticalRefusal::RotatedCharacter`] would refuse first.
//! Siddham, Soyombo and Zanabazar Square are Brahmic AND `Upright` in
//! UAX #50, so that rung does not catch them —
//! `upright_brahmic_blocks_are_not_caught_by_the_rotation_rung` in
//! `vertical_table` records the counter-example on purpose. The conclusion
//! survives, the reasoning did not; the `vertical_script` argument above is the
//! one that actually holds. `vertical_table`'s
//! `every_complex_ltr_character_the_ladder_meets_is_rotated` remains true as a
//! claim about the exporter's five listed complex-LTR RANGES — which do not
//! include those three scripts — and is kept as that narrower statement.

use std::sync::Arc;

use oxitext::{ShapeDirection, ShapeRequest, SwashShaper};
use ttf_parser::{Face, GlyphId};

use crate::label::vertical_table::vertical_orientation_of;

/// Sanity bound on a cell's vertical advance, as a fraction of the em.
///
/// malgun's `vmtx` mixes 560 and 2176 design units at upem 2048 (0.27 em and
/// 1.06 em) inside one script; a face whose numbers leave this window is
/// describing something this simple stacker cannot lay out. The same two
/// constants the PDF exporter uses.
const MIN_PITCH_EM: f32 = 0.25;

/// Upper half of the pitch bound — see [`MIN_PITCH_EM`].
const MAX_PITCH_EM: f32 = 2.0;

/// Sanity bound on a glyph's vertical ORIGIN, as a fraction of the em.
///
/// The pitch bound guards `vmtx`'s advance; nothing guarded the origin, and
/// vertical metrics are already recorded as inconsistent across faces (meiryo
/// declares a `vhea` ascender of 955 at an upem of 2048). A face whose origin
/// leaves this window would hang its ink most of a line box away from where
/// the column expects it, so the label refuses to horizontal instead.
const MIN_VERTICAL_ORIGIN_EM: f32 = 0.5;

/// Upper half of the origin bound — see [`MIN_VERTICAL_ORIGIN_EM`].
const MAX_VERTICAL_ORIGIN_EM: f32 = 1.5;

/// One maximal same-orientation slice of a line, as [`vertical_runs`] cuts it.
///
/// The two variants are the two things a vertical writing path can do with a
/// character: stack it upright in its own cell, or turn the whole run 90°
/// clockwise so it advances down the column sideways.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalRun<'a> {
    /// Characters that draw upright — `U`, `Tu` or `Tr` in UAX #50.
    Upright(&'a str),
    /// Characters UAX #50 rotates — `R`: Latin, digits, halfwidth kana, and
    /// the space between two Latin words.
    Rotated(&'a str),
}

impl<'a> VerticalRun<'a> {
    /// The run's text, whichever way it is set.
    #[must_use]
    pub fn text(self) -> &'a str {
        match self {
            Self::Upright(text) | Self::Rotated(text) => text,
        }
    }

    /// Whether this run stacks upright rather than turning sideways.
    #[must_use]
    pub fn is_upright(self) -> bool {
        matches!(self, Self::Upright(_))
    }
}

/// Cuts `text` into maximal runs of one orientation class, in logical order.
///
/// The class is [`super::vertical_table::VerticalOrientation::draws_upright`]
/// and nothing else — no
/// font, no locale, no shaper. The cuts are byte ranges of `text` and every
/// one falls on a `char` boundary by construction (the classifier is fed
/// whole `char`s), so the borrowed slices are always valid UTF-8 and a run
/// boundary inside a multi-byte character is impossible.
///
/// An empty input yields no runs at all, and concatenating every run's
/// [`VerticalRun::text`] reproduces `text` exactly — which is what lets a
/// caller rebuild the line and compare it byte for byte.
#[must_use]
pub fn vertical_runs(text: &str) -> Vec<VerticalRun<'_>> {
    let mut runs = Vec::new();
    let mut start = 0_usize;
    let mut class: Option<bool> = None;
    for (index, ch) in text.char_indices() {
        let upright = vertical_orientation_of(ch).draws_upright();
        match class {
            Some(held) if held == upright => {}
            Some(held) => {
                runs.push(run_of(held, &text[start..index]));
                start = index;
                class = Some(upright);
            }
            None => class = Some(upright),
        }
    }
    if let Some(held) = class {
        runs.push(run_of(held, &text[start..]));
    }
    runs
}

/// Wraps a slice in the variant its orientation class names.
fn run_of(upright: bool, text: &str) -> VerticalRun<'_> {
    if upright {
        VerticalRun::Upright(text)
    } else {
        VerticalRun::Rotated(text)
    }
}

/// The OpenType script tag a vertical CJK line is shaped under.
///
/// Hangul first, then kana, then Han: a line mixing them is dominated by
/// whichever the shaper needs to reach the right `vert` lookups, and the
/// three tags select the same vertical features on every measured face.
///
/// Shared by the PDF exporter's vertical title and the renderer's vertical
/// labels so the two cannot ask one face for different features.
#[must_use]
pub fn vertical_script(text: &str) -> [u8; 4] {
    let mut kana = false;
    for ch in text.chars() {
        match u32::from(ch) {
            0x1100..=0x11FF | 0xA960..=0xA97F | 0xAC00..=0xD7FF => return *b"hang",
            0x3040..=0x30FF | 0x31F0..=0x31FF => kana = true,
            _ => {}
        }
    }
    if kana { *b"kana" } else { *b"hani" }
}

/// Why a label cannot be set vertically — carried so the caller can log the
/// reason once per (engine generation, reason) rather than once per label.
///
/// A screenful is hundreds of labels; a per-label log would be the loudest
/// thing the renderer does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VerticalRefusal {
    /// A character whose UAX #50 orientation is `R` (Latin, digits,
    /// halfwidth kana). v1.5's screen scope is upright-only: rotating a
    /// glyph on screen is a vertex-buffer change this stage does not make.
    RotatedCharacter,
    /// A right-to-left character. **Not dead code**: Meroitic Hieroglyphs
    /// (U+10980..U+1099F) are `Upright` *and* RTL, so the orientation table
    /// alone does not catch them.
    RightToLeft,
    /// The label's characters need more than one face. One pitch, one em box
    /// and one substitution behaviour per column.
    MultipleFaces,
    /// The face has no `vmtx`, so there is no top side bearing and hence no
    /// vertical origin. Accepting would mean an unpinned origin rule, which
    /// is exactly the wrong output all-or-nothing exists to refuse.
    NoVerticalMetrics,
    /// Shaping produced a multi-glyph cluster, a placement offset or a
    /// `.notdef` — none of which a one-glyph-per-cell stacker can express.
    InexpressibleCluster,
    /// A `glyph_ver_advance` outside 0.25..=2 em.
    ImplausiblePitch,
    /// A vertical origin outside 0.5..=1.5 em.
    ImplausibleVerticalOrigin,
    /// Empty text, an unparseable face, `upem == 0`, or a shaping error.
    Unusable,
}

impl VerticalRefusal {
    /// The reason, for the one aggregated log.
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self {
            Self::RotatedCharacter => {
                "a character UAX #50 rotates (Latin, digits, halfwidth kana); \
                 screen glyph rotation is not in v1.5"
            }
            Self::RightToLeft => "a right-to-left character",
            Self::MultipleFaces => "the label needs more than one face",
            Self::NoVerticalMetrics => "the face has no vmtx table",
            Self::InexpressibleCluster => {
                "shaping produced a multi-glyph cluster, a placement offset or a .notdef"
            }
            Self::ImplausiblePitch => "a vertical advance outside 0.25..=2 em",
            Self::ImplausibleVerticalOrigin => "a vertical origin outside 0.5..=1.5 em",
            Self::Unusable => "the label or the face is unusable",
        }
    }
}

/// Whether `text` contains right-to-left script or bidi controls.
///
/// A render-local copy of the PDF exporter's own predicate — the two paths
/// must agree on what "right to left" means, and the renderer deliberately
/// depends on neither `oxigis-core` nor `oxigis-ui`.
#[must_use]
pub fn has_rtl(text: &str) -> bool {
    text.chars().any(|ch| {
        matches!(u32::from(ch),
            0x0590..=0x08FF
                | 0x200E
                | 0x200F
                | 0x202A..=0x202E
                | 0x2066..=0x2069
                | 0xFB1D..=0xFDFF
                | 0xFE70..=0xFEFF
                | 0x1_0800..=0x1_0FFF
                | 0x1_E800..=0x1_EFFF
        )
    })
}

/// The first face of `chain` whose cmap covers `ch`, or [`None`] when no face
/// does.
///
/// First hit wins, in chain order — the same order the bold pipeline is built
/// in, so a vertical label and a horizontal one pick the same face for the
/// same character by construction. `oxitext`'s own per-glyph face identity
/// cannot be used for this: its notdef fallback overwrites the RUN-level
/// `font_data`, so a mixed label reports one face for every glyph.
///
/// One `Face::parse` per chain entry per call: use [`parse_chain`] plus
/// [`face_index_in`] when more than one character is resolved against the same
/// chain, which is what [`plan_vertical`] does.
#[must_use]
pub fn face_for(chain: &[Arc<[u8]>], ch: char) -> Option<usize> {
    face_index_in(&parse_chain(chain), ch)
}

/// Parses every entry of `chain` once, preserving indices.
///
/// An entry that does not parse becomes [`None`] rather than being dropped:
/// the position in the returned slice **is** the chain index every caller and
/// [`VerticalPlan::chain_index`] speak in, so a broken face must not shift its
/// successors.
#[must_use]
pub fn parse_chain(chain: &[Arc<[u8]>]) -> Vec<Option<Face<'_>>> {
    chain
        .iter()
        .map(|bytes| Face::parse(bytes.as_ref(), 0).ok())
        .collect()
}

/// The first face of an already-parsed chain whose cmap covers `ch`.
///
/// The allocation-free half of [`face_for`]; same first-hit-wins rule.
#[must_use]
pub fn face_index_in(faces: &[Option<Face<'_>>], ch: char) -> Option<usize> {
    faces.iter().position(|face| {
        face.as_ref()
            .and_then(|face| face.glyph_index(ch))
            .is_some()
    })
}

/// One planned cell of a vertical label: which glyph, where its baseline
/// sits, and what the pen owes the next cell.
///
/// Pixel numbers, unrounded — the engine rounds once, when it adds the
/// glyph's own ink bearings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerticalCell {
    /// Glyph id in the resolved face, post-`vert`/`vrt2` substitution.
    pub gid: u16,
    /// The cell's advance in pixels — how far the pen top drops after it.
    pub pitch_px: f32,
    /// Baseline of this cell, in pixels down from the label box's top edge.
    pub baseline_px: f32,
    /// Horizontal centring shift inside the one-em column, in pixels.
    pub x_shift_px: f32,
}

/// An accepted vertical label, before rasterisation.
#[derive(Debug, Clone, PartialEq)]
pub struct VerticalPlan {
    /// Index into the chain of the ONE face that draws the whole label.
    pub chain_index: usize,
    /// The cells, top to bottom — one per character.
    pub cells: Vec<VerticalCell>,
    /// The collision box: one em wide, the summed pitch tall.
    pub size_px: [f32; 2],
}

/// The OpenType vertical origin of `gid` in design units, or [`None`] for a
/// glyph with no outline at all.
///
/// `VORG` when the face has it; otherwise the OpenType default
/// `yMax + tsb`, which is every measured CJK face (all have `vmtx`, none has
/// `VORG`). A face's `vhea` ascender is deliberately NOT used — those are
/// measured inconsistent across faces.
fn vertical_origin_units(face: &Face<'_>, gid: GlyphId) -> Option<f32> {
    if let Some(origin) = face.glyph_y_origin(gid) {
        return Some(f32::from(origin));
    }
    let bbox = face.glyph_bounding_box(gid)?;
    let tsb = face.glyph_ver_side_bearing(gid).unwrap_or(0);
    Some(f32::from(bbox.y_max) + f32::from(tsb))
}

/// Plans `text` as a vertical column at `size_px`, or refuses it.
///
/// `chain` is the face chain for the label's ALREADY-effective weight, in
/// consultation order. All-or-nothing: any refusal means the caller draws
/// today's horizontal label unchanged.
///
/// # Errors
///
/// One [`VerticalRefusal`] per rung — see the enum for what each means.
pub fn plan_vertical(
    shaper: &mut SwashShaper,
    chain: &[Arc<[u8]>],
    text: &str,
    size_px: f32,
) -> Result<VerticalPlan, VerticalRefusal> {
    if text.is_empty() || !size_px.is_finite() || size_px <= 0.0 {
        return Err(VerticalRefusal::Unusable);
    }
    if has_rtl(text) {
        return Err(VerticalRefusal::RightToLeft);
    }
    // Parsed once for the whole label: `Face::parse` re-reads the sfnt table
    // directory, and resolving per character against the chain would pay that
    // `characters x chain` times.
    let faces = parse_chain(chain);
    let mut chain_index: Option<usize> = None;
    for ch in text.chars() {
        if !vertical_orientation_of(ch).draws_upright() {
            return Err(VerticalRefusal::RotatedCharacter);
        }
        let face = face_index_in(&faces, ch).ok_or(VerticalRefusal::Unusable)?;
        match chain_index {
            Some(held) if held != face => return Err(VerticalRefusal::MultipleFaces),
            Some(_) => {}
            None => chain_index = Some(face),
        }
    }
    let chain_index = chain_index.ok_or(VerticalRefusal::Unusable)?;
    let bytes = chain.get(chain_index).ok_or(VerticalRefusal::Unusable)?;
    let face = faces
        .get(chain_index)
        .and_then(Option::as_ref)
        .ok_or(VerticalRefusal::Unusable)?;
    if face.tables().vmtx.is_none() {
        return Err(VerticalRefusal::NoVerticalMetrics);
    }
    let upem = f32::from(face.units_per_em());
    if upem <= 0.0 {
        return Err(VerticalRefusal::Unusable);
    }

    // `Ttb` is LTR shaping with `vert`+`vrt2` appended; `px_size 0.0` keeps
    // every number in design units, so one shaping serves every size.
    let request = ShapeRequest::builder()
        .text(text)
        .font_data(bytes.as_ref())
        .px_size(0.0)
        .direction(ShapeDirection::Ttb)
        .script(vertical_script(text))
        .build()
        .map_err(|_| VerticalRefusal::Unusable)?;
    let shaped = shaper
        .shape_request(&request)
        .map_err(|_| VerticalRefusal::Unusable)?;
    if shaped.len() != text.chars().count() {
        return Err(VerticalRefusal::InexpressibleCluster);
    }

    let scale = size_px / upem;
    let mut cells = Vec::with_capacity(shaped.len());
    let mut pen_top_px = 0.0_f32;
    for glyph in &shaped {
        if glyph.gid == 0 || glyph.x_offset != 0.0 || glyph.y_offset != 0.0 {
            return Err(VerticalRefusal::InexpressibleCluster);
        }
        let gid = GlyphId(glyph.gid);
        // A glyph `vmtx` does not describe falls back to one em, which is
        // what every full-width CJK cell is anyway.
        let pitch_units = face.glyph_ver_advance(gid).map_or(upem, f32::from);
        if !pitch_units.is_finite()
            || pitch_units < MIN_PITCH_EM * upem
            || pitch_units > MAX_PITCH_EM * upem
        {
            return Err(VerticalRefusal::ImplausiblePitch);
        }
        // A glyph with no outline has no ink to place; it still costs its
        // pitch, and the rasteriser's empty-bitmap path drops it.
        let origin_units = match vertical_origin_units(face, gid) {
            Some(units) => {
                if !units.is_finite()
                    || units < MIN_VERTICAL_ORIGIN_EM * upem
                    || units > MAX_VERTICAL_ORIGIN_EM * upem
                {
                    return Err(VerticalRefusal::ImplausibleVerticalOrigin);
                }
                units
            }
            None => 0.0,
        };
        let pitch_px = pitch_units * scale;
        let horizontal = face.glyph_hor_advance(gid).map_or(upem, f32::from);
        cells.push(VerticalCell {
            gid: glyph.gid,
            pitch_px,
            baseline_px: pen_top_px + origin_units * scale,
            x_shift_px: (size_px - horizontal * scale) / 2.0,
        });
        pen_top_px += pitch_px;
    }
    Ok(VerticalPlan {
        chain_index,
        cells,
        size_px: [size_px, pen_top_px],
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        MAX_PITCH_EM, MAX_VERTICAL_ORIGIN_EM, MIN_PITCH_EM, MIN_VERTICAL_ORIGIN_EM,
        VerticalRefusal, VerticalRun, face_for, face_index_in, has_rtl, parse_chain, plan_vertical,
        vertical_orientation_of, vertical_runs, vertical_script,
    };
    use oxitext::SwashShaper;
    use ttf_parser::Face;

    /// Noto Sans Regular: Latin-only, static, and — decisively for this
    /// module — **no `vmtx`**, so it exercises three rungs of the ladder
    /// without a Windows font.
    fn noto() -> Arc<[u8]> {
        Arc::from(oxifont_bundled::NOTO_SANS_REGULAR)
    }

    /// Noto Sans Mono: the SECOND face of the forced-split fixture. It covers
    /// U+221E (INFINITY, `Upright`) which Noto Sans Regular does not, so a
    /// two-face chain can be made to disagree through real cmaps rather than
    /// through a stub.
    fn noto_mono() -> Arc<[u8]> {
        Arc::from(oxifont_bundled::NOTO_SANS_MONO_REGULAR)
    }

    fn refuse(chain: &[Arc<[u8]>], text: &str) -> VerticalRefusal {
        let mut shaper = SwashShaper::new();
        match plan_vertical(&mut shaper, chain, text, 14.0) {
            Ok(plan) => panic!("{text:?} was expected to refuse, got {plan:?}"),
            Err(refusal) => refusal,
        }
    }

    /// The load-bearing half of this module's panic-safety argument, made
    /// checkable (added 2026-08-11). `vertical_script` is the ONLY source of
    /// the script tag the vertical path ever hands the shaper, and it is total:
    /// whatever it is given, it returns one of `hang`/`kana`/`hani`. None of
    /// the three is complex, so swash's `EngineMode::Complex` — the Indic code
    /// — is unreachable from here regardless of what the label text contains.
    ///
    /// The Devanagari and Siddham cases are the point: the old argument
    /// assumed such text would be refused by the rotation rung before shaping
    /// (false for Siddham, which is `Upright`). This test does not care, because
    /// even if such a string reaches the shaper it is asked for `hani`.
    #[test]
    fn vertical_script_is_never_complex() {
        let complex_tags = [*b"dev2", *b"bng2", *b"ory2", *b"knd2", *b"sinh", *b"telu"];
        let inputs = [
            "",
            "Tokyo",
            "東京",
            "とうきょう",
            "서울",
            "東京とうきょう서울",
            // Complex-LTR text the ladder was wrongly assumed never to meet.
            "मार्ग",
            "\u{11580}\u{11581}", // Siddham, Brahmic AND Upright
            "\u{11A00}\u{11A50}", // Zanabazar Square, Soyombo
            "मार्ग東京",
            "\u{645}\u{631}", // Arabic
        ];
        for text in inputs {
            let tag = vertical_script(text);
            assert!(
                matches!(&tag, b"hang" | b"kana" | b"hani"),
                "{text:?} produced tag {:?}, outside the three vertical tags",
                std::str::from_utf8(&tag),
            );
            assert!(
                !complex_tags.contains(&tag),
                "{text:?} produced a complex tag — the panic-safety argument \
                 in this module's docs would be void",
            );
        }
        // And the exhaustive form over every scalar value on its own: a single
        // character is the smallest label the ladder can be handed.
        for code in 0_u32..=0x10_FFFF {
            let Some(ch) = char::from_u32(code) else {
                continue;
            };
            let tag = vertical_script(&ch.to_string());
            assert!(
                matches!(&tag, b"hang" | b"kana" | b"hani"),
                "U+{code:04X} produced tag {:?}",
                std::str::from_utf8(&tag),
            );
        }
    }

    #[test]
    fn the_ladder_refuses_every_rung_the_bundled_face_can_reach() {
        let chain = [noto()];
        assert_eq!(refuse(&chain, ""), VerticalRefusal::Unusable);
        assert_eq!(
            refuse(&chain, "Tokyo"),
            VerticalRefusal::RotatedCharacter,
            "v1.5 screen scope is upright-only",
        );
        assert_eq!(
            refuse(&chain, "\u{645}\u{631}"),
            VerticalRefusal::RightToLeft
        );
        // The section sign and the plus-minus sign are Upright AND live in a
        // Latin font, so they get past the first two rungs and land on the
        // one this fixture exists for.
        assert_eq!(
            refuse(&chain, "\u{A7}\u{B1}"),
            VerticalRefusal::NoVerticalMetrics,
        );
        // A character no face covers is Unusable, not a silent drop.
        assert_eq!(refuse(&chain, "\u{6771}"), VerticalRefusal::Unusable);
        // And a nonsense size never reaches the shaper.
        let mut shaper = SwashShaper::new();
        for size in [0.0_f32, -1.0, f32::NAN] {
            assert_eq!(
                plan_vertical(&mut shaper, &chain, "\u{A7}", size),
                Err(VerticalRefusal::Unusable),
                "size {size}",
            );
        }
    }

    #[test]
    fn a_label_whose_characters_need_two_faces_refuses() {
        let chain = [noto(), noto_mono()];
        // The premise, asserted rather than assumed: the section sign is in
        // both faces (so it resolves to the FIRST) and U+221E only in the
        // second.
        assert_eq!(face_for(&chain, '\u{A7}'), Some(0), "in Noto Sans");
        assert_eq!(
            face_for(&chain, '\u{221E}'),
            Some(1),
            "U+221E is only in Noto Sans Mono - the forced split",
        );
        assert!(vertical_orientation_of('\u{221E}').draws_upright());
        assert_eq!(
            refuse(&chain, "\u{A7}\u{221E}"),
            VerticalRefusal::MultipleFaces,
        );
        // Either character alone still reaches the vmtx rung, so the split is
        // what refuses and not the characters.
        assert_eq!(refuse(&chain, "\u{A7}"), VerticalRefusal::NoVerticalMetrics);
        assert_eq!(
            refuse(&chain, "\u{221E}"),
            VerticalRefusal::NoVerticalMetrics,
        );
    }

    #[test]
    fn face_resolution_is_first_hit_wins_and_says_no_rather_than_guessing() {
        let chain = [noto(), noto_mono()];
        assert_eq!(face_for(&chain, 'A'), Some(0));
        assert_eq!(face_for(&chain, '\u{6771}'), None, "no CJK in the chain");
        assert_eq!(face_for(&[], 'A'), None, "an empty chain covers nothing");
        // Junk bytes never parse, so they are skipped rather than claimed.
        let junk: Arc<[u8]> = Arc::from(vec![0_u8; 64].as_slice());
        assert_eq!(face_for(&[junk, noto()], 'A'), Some(1));
    }

    #[test]
    fn parsing_the_chain_once_answers_exactly_as_parsing_it_per_character_did() {
        let junk: Arc<[u8]> = Arc::from(vec![0_u8; 64].as_slice());
        let chain = [junk, noto(), noto_mono()];
        let faces = parse_chain(&chain);
        assert_eq!(faces.len(), chain.len(), "indices stay aligned");
        assert!(faces[0].is_none(), "the unparseable entry holds its slot");
        // Every answer matches the one-parse-per-character function it replaces,
        // including the two characters that force the chain to split.
        for ch in ['A', '\u{A7}', '\u{221E}', '\u{6771}', '\u{1F600}'] {
            assert_eq!(face_index_in(&faces, ch), face_for(&chain, ch), "{ch:?}");
        }
        assert_eq!(face_index_in(&faces, 'A'), Some(1));
        assert_eq!(face_index_in(&faces, '\u{221E}'), Some(2));
        assert_eq!(face_index_in(&[], 'A'), None);
        // And the plan built from that chain names the same face.
        let mut shaper = SwashShaper::new();
        assert_eq!(
            plan_vertical(&mut shaper, &chain, "\u{221E}", 14.0),
            Err(VerticalRefusal::NoVerticalMetrics),
            "resolved to Noto Sans Mono, which has no vmtx",
        );
    }

    #[test]
    fn the_rtl_predicate_agrees_with_the_exporters() {
        assert!(has_rtl("\u{645}"), "Arabic");
        assert!(has_rtl("\u{5D0}"), "Hebrew");
        assert!(has_rtl("\u{202E}"), "an explicit embedding control");
        assert!(has_rtl("\u{10980}"), "Meroitic - Upright AND RTL");
        assert!(!has_rtl("\u{6771}\u{4EAC}"));
        assert!(!has_rtl("Tokyo"));
    }

    #[test]
    fn every_refusal_reason_says_something_distinct() {
        let all = [
            VerticalRefusal::RotatedCharacter,
            VerticalRefusal::RightToLeft,
            VerticalRefusal::MultipleFaces,
            VerticalRefusal::NoVerticalMetrics,
            VerticalRefusal::InexpressibleCluster,
            VerticalRefusal::ImplausiblePitch,
            VerticalRefusal::ImplausibleVerticalOrigin,
            VerticalRefusal::Unusable,
        ];
        let mut seen = std::collections::BTreeSet::new();
        for refusal in all {
            assert!(!refusal.reason().is_empty(), "{refusal:?}");
            assert!(seen.insert(refusal.reason()), "{refusal:?} duplicates");
        }
        assert_eq!(seen.len(), 8, "eight rungs, and no CFF one");
    }

    #[test]
    fn the_sanity_bounds_are_the_designs_numbers() {
        assert_eq!([MIN_PITCH_EM, MAX_PITCH_EM], [0.25, 2.0]);
        assert_eq!([MIN_VERTICAL_ORIGIN_EM, MAX_VERTICAL_ORIGIN_EM], [0.5, 1.5]);
    }

    #[test]
    fn the_script_tag_follows_the_line() {
        assert_eq!(vertical_script("\u{6771}\u{4EAC}"), *b"hani");
        assert_eq!(vertical_script("\u{3042}\u{3044}"), *b"kana");
        assert_eq!(vertical_script("\u{6771}\u{4EAC}\u{3042}"), *b"kana");
        assert_eq!(vertical_script("\u{C11C}\u{C6B8}"), *b"hang");
        assert_eq!(
            vertical_script("\u{3042}\u{C11C}"),
            *b"hang",
            "Hangul wins outright",
        );
    }

    #[test]
    fn vertical_runs_cut_maximal_same_class_slices_and_lose_nothing() {
        assert!(vertical_runs("").is_empty(), "no characters, no runs");
        assert_eq!(
            vertical_runs("\u{6771}\u{4EAC}"),
            vec![VerticalRun::Upright("\u{6771}\u{4EAC}")],
        );
        assert_eq!(vertical_runs("Tokyo"), vec![VerticalRun::Rotated("Tokyo")]);
        assert_eq!(
            vertical_runs("\u{6771}"),
            vec![VerticalRun::Upright("\u{6771}")],
            "one upright character",
        );
        assert_eq!(vertical_runs("A"), vec![VerticalRun::Rotated("A")]);
        // The two named shell-visible titles.
        assert_eq!(
            vertical_runs("\u{6771}\u{4EAC} 2026"),
            vec![
                VerticalRun::Upright("\u{6771}\u{4EAC}"),
                VerticalRun::Rotated(" 2026"),
            ],
            "the space is R and joins the Latin run",
        );
        assert_eq!(
            vertical_runs("\u{6771}\u{4EAC}Tower"),
            vec![
                VerticalRun::Upright("\u{6771}\u{4EAC}"),
                VerticalRun::Rotated("Tower"),
            ],
        );
        // Alternating, and a leading rotated run.
        assert_eq!(
            vertical_runs("A\u{6771}B\u{4EAC}"),
            vec![
                VerticalRun::Rotated("A"),
                VerticalRun::Upright("\u{6771}"),
                VerticalRun::Rotated("B"),
                VerticalRun::Upright("\u{4EAC}"),
            ],
        );
        // Whatever the cuts, the concatenation is the input.
        for text in [
            "",
            "\u{6771}\u{4EAC} 2026",
            "A\u{6771}B\u{4EAC}",
            "\u{300C}\u{3042}\u{3001}\u{300D}Tower",
        ] {
            let joined: String = vertical_runs(text)
                .iter()
                .map(|run| run.text())
                .collect::<Vec<_>>()
                .concat();
            assert_eq!(joined, text, "the runs must rebuild {text:?}");
            for run in vertical_runs(text) {
                assert!(!run.text().is_empty(), "no empty run is ever emitted");
                assert_eq!(
                    run.is_upright(),
                    run.text()
                        .chars()
                        .all(|ch| vertical_orientation_of(ch).draws_upright()),
                    "a run is uniform in its class",
                );
            }
        }
    }

    /// Reads a Windows font, or signals that this machine has none.
    fn windows_font(name: &str) -> Option<Arc<[u8]>> {
        std::fs::read(format!("C:/Windows/Fonts/{name}"))
            .ok()
            .map(Arc::from)
    }

    #[test]
    #[ignore = "reads C:/Windows/Fonts; the D-A5/D-A6 screen goldens"]
    fn live_windows_a_vertical_cjk_label_stacks_and_records_its_origins() {
        // 「あ、」 — the same golden line the exporter pins. On screen the
        // claims are: one cell per character, a box one em wide and the
        // summed pitch tall, the `vert` substitution fired on the bracket,
        // and every vertical ORIGIN inside the new sanity bound.
        let text = "\u{300C}\u{3042}\u{3001}\u{300D}";
        let size = 16.0_f32;
        let mut any = false;
        for name in ["meiryo.ttc", "YuGothM.ttc", "msgothic.ttc", "msmincho.ttc"] {
            let Some(bytes) = windows_font(name) else {
                continue;
            };
            any = true;
            let Ok(face) = Face::parse(&bytes, 0) else {
                continue;
            };
            let upem = f32::from(face.units_per_em());
            let horizontal: Vec<u16> = text
                .chars()
                .map(|ch| face.glyph_index(ch).map_or(0, |gid| gid.0))
                .collect();
            let chain = [bytes];
            let mut shaper = SwashShaper::new();
            let plan = plan_vertical(&mut shaper, &chain, text, size)
                .unwrap_or_else(|refusal| panic!("{name} must stack 「あ、」: {refusal:?}"));
            assert_eq!(plan.chain_index, 0);
            assert_eq!(plan.cells.len(), 4, "{name}: one cell per character");
            assert_ne!(
                plan.cells[0].gid, horizontal[0],
                "{name}: the bracket must substitute its vertical form",
            );
            assert_eq!(
                plan.cells[1].gid, horizontal[1],
                "{name}: an Upright kana must NOT substitute",
            );
            // The box: one em wide, the summed pitch tall, and the cells step
            // down by exactly their own pitches.
            let summed: f32 = plan.cells.iter().map(|cell| cell.pitch_px).sum();
            assert_eq!(plan.size_px[0], size, "{name}: one em wide");
            assert!((plan.size_px[1] - summed).abs() < 0.001, "{name}");
            // Every ORIGIN inside the bound, recorded in em for the design.
            let origins: Vec<f32> = plan
                .cells
                .iter()
                .scan(0.0_f32, |pen, cell| {
                    let origin = (cell.baseline_px - *pen) / size;
                    *pen += cell.pitch_px;
                    Some(origin)
                })
                .collect();
            for origin in &origins {
                assert!(
                    (MIN_VERTICAL_ORIGIN_EM..=MAX_VERTICAL_ORIGIN_EM).contains(origin),
                    "{name}: vertical origin {origin} em is outside the bound \
                     (upem {upem}); widen it and log a DEVIATION",
                );
            }
            eprintln!("{name}: upem {upem}, vertical origins (em) {origins:?}");
        }
        assert!(any, "no Windows CJK face on this machine");
    }

    /// The design names malgun as the hostile-`vmtx` face and expected this
    /// to trip [`VerticalRefusal::ImplausiblePitch`]. Measured on this
    /// machine it does NOT: its Hangul cells are a uniform 1.0625 em, so the
    /// recorded 2176/560/1024/1552 run lives elsewhere in the face. The test
    /// therefore measures rather than asserts a refusal — and would still
    /// catch a face whose numbers left the window.
    #[test]
    #[ignore = "reads C:/Windows/Fonts/malgun.ttf; the hostile-vmtx face"]
    fn live_windows_malgun_is_measured_rather_than_assumed() {
        let Some(bytes) = windows_font("malgun.ttf") else {
            return;
        };
        let chain = [bytes];
        let mut shaper = SwashShaper::new();
        // A Hangul run is what reaches those advances.
        let text = "\u{C11C}\u{C6B8}\u{D2B9}\u{BCC4}\u{C2DC}";
        match plan_vertical(&mut shaper, &chain, text, 16.0) {
            Err(VerticalRefusal::ImplausiblePitch | VerticalRefusal::ImplausibleVerticalOrigin) => {
            }
            Err(other) => panic!("malgun refused for an unexpected reason: {other:?}"),
            Ok(plan) => {
                // Measured on this machine: malgun's HANGUL cells are a
                // uniform one em, so the hostile 2176/560/1024/1552 run the
                // design recorded is elsewhere in the face and this string
                // is accepted. The rung is still the reason the bound exists;
                // the evidence is printed rather than assumed.
                let pitches: Vec<f32> = plan.cells.iter().map(|cell| cell.pitch_px).collect();
                eprintln!("malgun.ttf accepted, pitches (px at 16) {pitches:?}");
                for cell in &plan.cells {
                    assert!(cell.pitch_px > 0.0, "an accepted malgun must be sane");
                    assert!(cell.baseline_px.is_finite());
                }
            }
        }
    }

    #[test]
    #[ignore = "reads C:/Windows/Fonts/simsun.ttc; the substitutes-nothing face"]
    fn live_windows_simsun_substitutes_nothing_and_is_accepted() {
        // simsun declares `vert`/`vrt2` and substitutes NOTHING. That is a
        // known, logged state and not a refusal: the glyphs are still
        // correct, the punctuation merely keeps its horizontal form.
        let Some(bytes) = windows_font("simsun.ttc") else {
            return;
        };
        let Ok(face) = Face::parse(&bytes, 0) else {
            return;
        };
        let text = "\u{4E1C}\u{4EAC}";
        let horizontal: Vec<u16> = text
            .chars()
            .map(|ch| face.glyph_index(ch).map_or(0, |gid| gid.0))
            .collect();
        let chain = [bytes];
        let mut shaper = SwashShaper::new();
        let plan = plan_vertical(&mut shaper, &chain, text, 16.0)
            .unwrap_or_else(|refusal| panic!("simsun must be accepted: {refusal:?}"));
        let vertical: Vec<u16> = plan.cells.iter().map(|cell| cell.gid).collect();
        assert_eq!(vertical, horizontal, "ideographs never substitute anyway");
        assert_eq!(plan.cells.len(), 2);
    }

    #[test]
    #[ignore = "reads C:/Windows/Fonts; determinism across cache misses"]
    fn live_windows_one_shaper_shapes_a_vertical_label_the_same_way_twice() {
        // The engine keeps ONE `SwashShaper` for the session. The reused-
        // shaper corruption the exporter's canary pins needs
        // `EngineMode::Complex`, which the ladder makes unreachable — this is
        // the assertion behind that argument.
        let Some(bytes) = windows_font("meiryo.ttc").or_else(|| windows_font("YuGothM.ttc")) else {
            return;
        };
        let chain = [bytes];
        let mut shaper = SwashShaper::new();
        let text = "\u{300C}\u{3042}\u{3001}\u{300D}\u{6771}\u{4EAC}";
        let first = plan_vertical(&mut shaper, &chain, text, 16.0).expect("stacks");
        let interleaved = plan_vertical(&mut shaper, &chain, "\u{4EAC}\u{90FD}", 12.0);
        assert!(interleaved.is_ok());
        let second = plan_vertical(&mut shaper, &chain, text, 16.0).expect("stacks again");
        assert_eq!(first, second, "one shaper, two passes, one answer");
    }
}
