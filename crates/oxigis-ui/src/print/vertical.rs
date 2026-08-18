// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Vertical writing for the PDF export (print/text v1.4, item 3 /
//! D-V1..D-V4): upright CJK glyphs stacked top to bottom, with sideways Latin
//! runs since v1.5 (D-B1..D-B6).
//!
//! # Scope
//!
//! The page TITLE was the whole of v1.4's scope — page furniture has no
//! on-screen counterpart, so a vertical title carries **zero parity risk**.
//! Since v1.6 the same planner also serves vertical LABELS: `print::labels`
//! measures each accepted column with [`VerticalLine::box_pt`], collides it
//! against the same padded-AABB pool the horizontal labels use, and draws it
//! through the same emitter. The screen has drawn vertical labels since v1.5
//! (`oxigis-render` owns their atlas and placement), so a page that kept
//! printing them horizontally disagreed with the map — the one divergence a
//! print feature must not have. A refusal still costs only that label, which
//! then prints horizontally.
//!
//! # A line is a sequence of ITEMS, not of glyphs
//!
//! [`oxigis_render::label::vertical_runs`] cuts the title into maximal
//! same-orientation runs. An `Upright` run becomes one stacked cell per
//! character; a `Rotated` run — Latin, digits, the space between two Latin
//! words — becomes ONE ordinary horizontal run turned 90° clockwise, shaped
//! by the very same [`super::shape::runs_for`] the horizontal path uses, so
//! it keeps its kerning. The emitter positions each item with an absolute
//! `Tm`; a line whose items are ALL upright takes the v1.4 relative-`Td`
//! path byte for byte.
//!
//! # Measured foundations
//!
//! * swash has no vertical direction of its own; `oxitext`'s
//!   [`ShapeDirection::Ttb`] is LTR shaping with `vert`+`vrt2` auto-appended,
//!   which fires real substitutions on meiryo / msgothic / YuGoth / msmincho
//!   / NotoSansJP-VF. **simsun substitutes nothing** despite declaring the
//!   features — accepted with one log, because the glyphs are still correct
//!   (punctuation keeps its horizontal form).
//! * `y_advance` comes back hard-coded `0.0`, so every vertical advance is
//!   ours to compute — from `vmtx` via
//!   [`ttf_parser::Face::glyph_ver_advance`] on the ORIGINAL face.
//! * All measured CJK faces have `vmtx`; none has `VORG`. `vhea`'s
//!   ascender/descender are inconsistent across faces (meiryo 955 against an
//!   upem of 2048) and are NOT used.
//! * malgun's `vmtx` is hostile — adjacent advances 2176/560/1024/1552 at
//!   upem 2048 — which is what the pitch sanity bound below exists for.
//!
//! # The refusal ladder is all-or-nothing
//!
//! Any refusal returns [`None`] and the title prints HORIZONTALLY, byte for
//! byte as it does today. A half-vertical title would be worse than either.

use std::collections::BTreeMap;

// The UAX #50 table, the run itemiser and the vertical script tag live in
// `oxigis-render` (print/text v1.5, D-A1): ONE table, so the page and the map
// can never disagree about which characters set upright.
use oxigis_render::label::{VerticalRun, vertical_runs, vertical_script};
use oxitext::{ShapeDirection, ShapeRequest, SwashShaper};
use ttf_parser::{Face, GlyphId};

use super::font::{ADJUST_EPSILON_1000, GlyphRun, PlannedFont, RunGlyph, to_thousandths};
use super::shape;

/// Sanity bound on a glyph's vertical advance, as a fraction of the em.
///
/// malgun's `vmtx` mixes 560 and 2176 design units at upem 2048 (0.27 em and
/// 1.06 em) inside one script; a face whose numbers leave this window is
/// describing something this simple stacker cannot lay out, so the line
/// refuses rather than printing overlapping or wildly spaced glyphs.
const MIN_PITCH_EM: f32 = 0.25;

/// Upper half of the pitch bound — see [`MIN_PITCH_EM`].
const MAX_PITCH_EM: f32 = 2.0;

/// Widest typographic box a SIDEWAYS run may have, as a fraction of the em.
///
/// The vertical title hangs in a one-em column inside the right-hand margin
/// strip: `print/mod.rs` puts the cell at `[page_w − 26, page_w − 10]` pt
/// inside a `[page_w − 36, page_w]` margin, so a run centred on the column
/// centre line has 2.25 em before it leaves the paper's margin. A face whose
/// `hhea` ascender minus descender exceeds this bound is describing a run
/// this column cannot hold, and the line refuses to horizontal.
const MAX_ROTATED_RUN_EM: f32 = 2.0;

/// One glyph of an accepted vertical line, ready for the emitter: CIDs and
/// 1000/em numbers, exactly like [`super::font::RunGlyph`].
#[derive(Clone, Debug, PartialEq)]
pub struct VerticalGlyph {
    /// Subset glyph id, which is also the PDF CID.
    pub cid: u16,
    /// How far the pen drops AFTER this glyph, in thousandths of an em.
    pub pitch_1000: f32,
    /// Horizontal centring shift within the em cell, in thousandths.
    /// Exactly `0.0` for a full-width glyph, which writes no operator.
    pub x_shift_1000: f32,
}

/// One SIDEWAYS run of a vertical line: an ordinary horizontal run, turned
/// 90° clockwise so it advances down the column.
///
/// The glyphs are exactly what the horizontal path produces — same shaping,
/// same kerning, same `/W` widths — because the emitter reuses
/// `emit::emit_run` on them verbatim. Only the text matrix differs.
#[derive(Clone, Debug, PartialEq)]
pub struct RotatedRun {
    /// The run, ready for the horizontal emitter. `run.font` is the page
    /// resource index and `run.advance_1000` is the column drop this run
    /// costs.
    pub run: GlyphRun,
    /// Cross-column shift in thousandths of an em: what centres the run's
    /// typographic box (`hhea` ascender to descender) on the column's centre
    /// line. `500 − (ascender + descender) / 2` in 1000/em terms.
    pub tx_1000: f32,
}

/// One item of an accepted vertical line, top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub enum VerticalItem {
    /// One upright cell — a stacked CJK glyph.
    Upright(VerticalGlyph),
    /// One sideways run — Latin, digits, or the space between two words.
    Rotated(RotatedRun),
}

impl VerticalItem {
    /// How far the pen drops after this item, in thousandths of an em.
    #[must_use]
    pub fn advance_1000(&self) -> f32 {
        match self {
            Self::Upright(glyph) => glyph.pitch_1000,
            Self::Rotated(rotated) => rotated.run.advance_1000,
        }
    }

    /// The page resource index this item draws through.
    #[must_use]
    pub fn font(&self, upright_font: usize) -> usize {
        match self {
            Self::Upright(_) => upright_font,
            Self::Rotated(rotated) => rotated.run.font,
        }
    }
}

/// An accepted vertical line: a top-to-bottom sequence of items, plus the
/// logical text its MANDATORY `/ActualText` span carries.
///
/// The span is not optional: an extraction probe showed that a vertical line
/// without one comes out of poppler as garbage (the per-glyph `Td` steps
/// carry no reading order), so the emitter always wraps the line and a test
/// asserts it. It stays LINE level even for a mixed line — a nested span per
/// rotated run is exactly the structure the v1.4 poppler probe validated the
/// absence of, and D-B3's single-glyph-cluster rung is what makes the flat
/// structure sufficient.
#[derive(Clone, Debug, PartialEq)]
pub struct VerticalLine {
    /// Index into `TextPlan::fonts` for the UPRIGHT cells; the page resource
    /// is `F{font + 1}`. A rotated item names its own font.
    pub font: usize,
    /// The items, top to bottom.
    pub items: Vec<VerticalItem>,
    /// The logical source text — what `/ActualText` says.
    pub actual_text: String,
    /// Total descent, in thousandths of an em.
    pub advance_1000: f32,
}

impl VerticalLine {
    /// The line's page box at `size` pt: one em wide, the summed advance
    /// tall.
    ///
    /// One em wide is unchanged by v1.5: a sideways Latin run is at most one
    /// em tall by the [`MAX_ROTATED_RUN_EM`] rung, so it cannot widen the
    /// column beyond what the margin strip already reserves.
    #[must_use]
    pub fn box_pt(&self, size: f32) -> [f32; 2] {
        [size, self.advance_1000 / 1000.0 * size]
    }

    /// Whether every item is an upright cell — the v1.4 shape, which the
    /// emitter renders byte for byte as it did before v1.5.
    #[must_use]
    pub fn is_all_upright(&self) -> bool {
        self.items
            .iter()
            .all(|item| matches!(item, VerticalItem::Upright(_)))
    }
}

/// One upright cell of a vertical line, in the ORIGINAL face's terms.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RawVerticalGlyph {
    /// Glyph id in the original face (post-`vert`/`vrt2` substitution).
    pub old_gid: u16,
    /// How far the pen drops AFTER this glyph, in design units.
    pub pitch_units: f32,
    /// Horizontal centring shift, in design units: half the slack between
    /// the glyph's horizontal advance and the em box. Zero for a full-width
    /// glyph, which is the overwhelmingly common case.
    pub x_shift_units: f32,
}

/// One item of a planned vertical line, before CID translation.
// No derives: `shape::RawRun` has none, and this enum carries one whole.
pub(super) enum RawVerticalItem {
    /// One upright cell.
    Upright(RawVerticalGlyph),
    /// One sideways run, shaped by the ordinary horizontal path.
    Rotated {
        /// Index into the shell's font chain — the run's own face, which
        /// need not be the upright face.
        chain_index: usize,
        /// The shaped run, in that face's design units.
        run: shape::RawRun,
        /// Cross-column centring shift, in thousandths of an em.
        tx_1000: f32,
    },
}

/// A whole accepted vertical line, before CID translation.
// No derives, for the same reason as [`RawVerticalItem`].
pub(super) struct RawVertical {
    /// The ONE chain face every UPRIGHT cell draws through. A line with no
    /// upright cell at all — a legal sideways column — names its first
    /// rotated run's face instead, so the caller always has a face to hang
    /// the line's resource index on.
    pub upright_chain_index: usize,
    /// The items, top to bottom.
    pub items: Vec<RawVerticalItem>,
    /// The UPRIGHT face's units per em, for the caller's 1000/em conversion
    /// of the upright cells. A rotated run converts through its own face.
    pub upem: f32,
}

/// Why a line cannot be set vertically — carried so the caller can log the
/// reason once instead of a bare "it printed horizontally".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VerticalRefusal {
    /// A right-to-left character. Vertical RTL is not a thing this path
    /// models, and the bidi reordering would fight the stacker.
    RightToLeft,
    /// The line's UPRIGHT cells span more than one chain face. One face for
    /// the upright column keeps the pitch, the em box and the substitution
    /// behaviour uniform. Since v1.5 the rule is per item: each rotated run
    /// brings its own face, because keeping the whole-line rule would refuse
    /// every mixed title by construction.
    MultipleFaces,
    /// The upright face has no `vmtx`, so there is no authoritative pitch to
    /// use. A line with NO upright cell never reaches this rung — it needs
    /// no vertical metrics at all.
    NoVerticalMetrics,
    /// A CFF upright face. `oxifont_subset::instance` refuses CFF/CFF2
    /// charstrings, and this path has no measurements behind CFF vertical
    /// metrics.
    ///
    /// **The screen ladder in `oxigis-render` deliberately has no such
    /// rung, and that divergence is intentional** (print/text v1.5, D-A5):
    /// this refusal is a PDF-instancing concern with no screen meaning, and
    /// Noto Sans CJK / Source Han Sans — the best-ranked CJK faces on Linux
    /// and macOS — are CFF. Copying this rung over there would ban vertical
    /// labels on them for a reason that does not apply. A future
    /// "unification" that adds it back would be a regression.
    ///
    /// It is also the rung that keeps v1.6's vertical LABELS off exactly
    /// those faces on the page while the screen stacks them — the one
    /// orientation divergence that survives, named here rather than hidden.
    CffFace,
    /// Shaping produced a multi-glyph cluster, a placement offset, or a
    /// `.notdef` in an UPRIGHT run — none of which a one-glyph-per-cell
    /// stacker can express.
    InexpressibleCluster,
    /// A `glyph_ver_advance` outside [`MIN_PITCH_EM`]..=[`MAX_PITCH_EM`].
    ImplausiblePitch,
    /// A rotated run the flat `/ActualText` structure cannot express: no run
    /// came back, a cluster carried more than one glyph, a glyph carried a
    /// placement offset, or shaping dropped characters. Keeping every
    /// rotated cluster single-glyph and unoffset is what lets the LINE-level
    /// span stay the only marked content on the line.
    RotatedRunUnshapeable,
    /// One rotated run spans more than one chain face. A sideways run is
    /// positioned by ONE text matrix built from ONE face's `hhea`, so a
    /// seam inside it would mis-centre half the run.
    RotatedRunMultipleFaces,
    /// A rotated face whose `hhea` ascender minus descender exceeds
    /// [`MAX_ROTATED_RUN_EM`] — a run too tall for the margin strip's column.
    RotatedRunTooWide,
    /// The face or the text is unusable (empty line, unparseable face, a
    /// shaping error).
    Unusable,
}

impl VerticalRefusal {
    /// The reason, for the one aggregated log.
    pub(super) fn reason(self) -> &'static str {
        match self {
            Self::RightToLeft => "a right-to-left character",
            Self::MultipleFaces => "the upright cells span more than one font-chain face",
            Self::NoVerticalMetrics => "the upright face has no vmtx table",
            Self::CffFace => "the upright face has CFF outlines",
            Self::InexpressibleCluster => {
                "shaping produced a multi-glyph cluster, a \
                                           placement offset or a .notdef"
            }
            Self::ImplausiblePitch => "a vertical advance outside 0.25..=2 em",
            Self::RotatedRunUnshapeable => {
                "a sideways run is not a sequence of plain single-glyph clusters"
            }
            Self::RotatedRunMultipleFaces => "a sideways run spans more than one font-chain face",
            Self::RotatedRunTooWide => "a sideways run's face is taller than 2 em",
            Self::Unusable => "the line or the face is unusable",
        }
    }
}

/// Plans one line as vertical text, or refuses it.
///
/// `coverage` is the plan's character → chain-face map for the line's weight;
/// `faces` are the parsed chain faces. Every rung of the ladder is checked
/// here, so the caller's only job is to log the reason and fall back to the
/// horizontal path. All-or-nothing: one refusal costs the whole line, which
/// then prints horizontally, byte for byte as it does today.
///
/// A line whose items are ALL upright plans exactly as it did in v1.4 — one
/// `Ttb` shaping request over the whole string, under the whole string's
/// script tag — so its gids, pitches and shifts are unchanged.
pub(super) fn plan_vertical(
    shaper: &mut SwashShaper,
    chain: &[Vec<u8>],
    faces: &[Option<Face<'_>>],
    coverage: &BTreeMap<char, usize>,
    text: &str,
) -> Result<RawVertical, VerticalRefusal> {
    if text.is_empty() {
        return Err(VerticalRefusal::Unusable);
    }
    if shape::has_rtl(text) {
        return Err(VerticalRefusal::RightToLeft);
    }
    let runs = vertical_runs(text);
    // One face for the whole upright column (pitch, em box and substitution
    // uniformity); each rotated run brings its own.
    let mut upright_chain: Option<usize> = None;
    for run in &runs {
        if !run.is_upright() {
            continue;
        }
        for ch in run.text().chars() {
            let face = *coverage.get(&ch).ok_or(VerticalRefusal::Unusable)?;
            match upright_chain {
                Some(held) if held != face => return Err(VerticalRefusal::MultipleFaces),
                Some(_) => {}
                None => upright_chain = Some(face),
            }
        }
    }
    // The upright face's rungs — CFF, vmtx and upem — are only asked of a
    // line that actually stacks something. A sideways-only column needs no
    // vertical metrics at all.
    let mut upem = 0.0_f32;
    if let Some(chain_index) = upright_chain {
        let face = faces
            .get(chain_index)
            .and_then(Option::as_ref)
            .ok_or(VerticalRefusal::Unusable)?;
        if face.tables().cff.is_some() {
            return Err(VerticalRefusal::CffFace);
        }
        if face.tables().vmtx.is_none() {
            return Err(VerticalRefusal::NoVerticalMetrics);
        }
        upem = f32::from(face.units_per_em());
        if upem <= 0.0 {
            return Err(VerticalRefusal::Unusable);
        }
    }
    // The whole line's script tag, not the run's: an all-upright line must
    // shape through exactly the request v1.4 built.
    let script = vertical_script(text);

    let mut items: Vec<RawVerticalItem> = Vec::new();
    let mut first_chain: Option<usize> = None;
    for run in &runs {
        match run {
            VerticalRun::Upright(slice) => {
                let chain_index = upright_chain.ok_or(VerticalRefusal::Unusable)?;
                let face = faces
                    .get(chain_index)
                    .and_then(Option::as_ref)
                    .ok_or(VerticalRefusal::Unusable)?;
                let bytes = chain.get(chain_index).ok_or(VerticalRefusal::Unusable)?;
                first_chain.get_or_insert(chain_index);
                for glyph in plan_upright(shaper, bytes, face, upem, script, slice)? {
                    items.push(RawVerticalItem::Upright(glyph));
                }
            }
            VerticalRun::Rotated(slice) => {
                let (chain_index, run, tx_1000) =
                    plan_rotated(shaper, chain, faces, coverage, slice)?;
                first_chain.get_or_insert(chain_index);
                items.push(RawVerticalItem::Rotated {
                    chain_index,
                    run,
                    tx_1000,
                });
            }
        }
    }
    let upright_chain_index = upright_chain
        .or(first_chain)
        .ok_or(VerticalRefusal::Unusable)?;
    if items.is_empty() {
        return Err(VerticalRefusal::Unusable);
    }
    Ok(RawVertical {
        upright_chain_index,
        items,
        upem,
    })
}

/// Stacks one upright run into cells, or refuses the line.
fn plan_upright(
    shaper: &mut SwashShaper,
    bytes: &[u8],
    face: &Face<'_>,
    upem: f32,
    script: [u8; 4],
    text: &str,
) -> Result<Vec<RawVerticalGlyph>, VerticalRefusal> {
    // `Ttb` is LTR shaping with `vert`+`vrt2` appended; `px_size 0.0` keeps
    // every number in design units, exactly as the horizontal path does.
    let request = ShapeRequest::builder()
        .text(text)
        .font_data(bytes)
        .px_size(0.0)
        .direction(ShapeDirection::Ttb)
        .script(script)
        .build()
        .map_err(|_| VerticalRefusal::Unusable)?;
    let shaped = shaper
        .shape_request(&request)
        .map_err(|_| VerticalRefusal::Unusable)?;
    // One glyph per character, in order: anything else (a cluster, a dropped
    // character, a reorder) is not a stack of cells.
    if shaped.len() != text.chars().count() {
        return Err(VerticalRefusal::InexpressibleCluster);
    }
    let mut glyphs = Vec::with_capacity(shaped.len());
    for glyph in &shaped {
        if glyph.gid == 0 {
            return Err(VerticalRefusal::InexpressibleCluster);
        }
        if glyph.x_offset != 0.0 || glyph.y_offset != 0.0 {
            return Err(VerticalRefusal::InexpressibleCluster);
        }
        let gid = GlyphId(glyph.gid);
        // The pitch is the face's OWN vertical advance; a glyph `vmtx` does
        // not describe falls back to one em, which is what every full-width
        // CJK cell is anyway.
        let pitch_units = face.glyph_ver_advance(gid).map_or(upem, f32::from);
        if !pitch_units.is_finite()
            || pitch_units < MIN_PITCH_EM * upem
            || pitch_units > MAX_PITCH_EM * upem
        {
            return Err(VerticalRefusal::ImplausiblePitch);
        }
        // Centre a narrower-than-em glyph in its cell; a full-width glyph
        // shifts by exactly zero, which writes no operator at all.
        let horizontal = face.glyph_hor_advance(gid).map_or(upem, f32::from);
        let x_shift_units = (upem - horizontal) / 2.0;
        glyphs.push(RawVerticalGlyph {
            old_gid: glyph.gid,
            pitch_units,
            x_shift_units,
        });
    }
    Ok(glyphs)
}

/// Shapes one rotated run through the ordinary horizontal path, or refuses
/// the line.
///
/// [`super::shape::runs_for`] is the shipped R1 `shape_slice` path, so the
/// run keeps its kerning, its cluster handling and its `.notdef` refusal for
/// free. Everything this function adds is the flatness contract D-B3 needs:
/// ONE run, every cluster a single unoffset glyph, and nothing dropped.
fn plan_rotated(
    shaper: &mut SwashShaper,
    chain: &[Vec<u8>],
    faces: &[Option<Face<'_>>],
    coverage: &BTreeMap<char, usize>,
    text: &str,
) -> Result<(usize, shape::RawRun, f32), VerticalRefusal> {
    let mut runs = shape::runs_for(shaper, chain, faces, coverage, text);
    if runs.len() > 1 {
        return Err(VerticalRefusal::RotatedRunMultipleFaces);
    }
    let run = runs.pop().ok_or(VerticalRefusal::RotatedRunUnshapeable)?;
    // Flat clusters only: one glyph each, no placement offset. That is what
    // keeps the line-level `/ActualText` span the ONLY marked content on the
    // line and the CID translation a straight walk.
    for cluster in &run.clusters {
        if cluster.glyphs.len() != 1 {
            return Err(VerticalRefusal::RotatedRunUnshapeable);
        }
        if cluster
            .glyphs
            .iter()
            .any(|glyph| glyph.x_offset_units != 0.0 || glyph.y_offset_units != 0.0)
        {
            return Err(VerticalRefusal::RotatedRunUnshapeable);
        }
    }
    // Nothing dropped: `runs_for` silently skips a character no face covers,
    // and a title missing a letter is not a title.
    let drawn: String = run
        .clusters
        .iter()
        .map(|cluster| cluster.text.as_str())
        .collect();
    if drawn != text {
        return Err(VerticalRefusal::RotatedRunUnshapeable);
    }
    let face = faces
        .get(run.chain_index)
        .and_then(Option::as_ref)
        .ok_or(VerticalRefusal::Unusable)?;
    let upem = f32::from(face.units_per_em());
    if upem <= 0.0 {
        return Err(VerticalRefusal::Unusable);
    }
    let ascender = f32::from(face.ascender());
    let descender = f32::from(face.descender());
    if !(ascender - descender).is_finite() || ascender - descender > MAX_ROTATED_RUN_EM * upem {
        return Err(VerticalRefusal::RotatedRunTooWide);
    }
    // Centre the run's typographic box on the column's centre line: the
    // rotation sends text +y to page +x, so the box spans `descender` to
    // `ascender` across the column and its midpoint must land at half an em.
    let tx_1000 = 500.0 - (to_thousandths(ascender, upem) + to_thousandths(descender, upem)) / 2.0;
    Ok((run.chain_index, run, tx_1000))
}

/// Translates a planned vertical line into CIDs and 1000/em numbers, or
/// refuses it.
///
/// Moved here out of `font.rs`'s pass 4 (print/text v1.5, D-B6) — the same
/// pure move `print/subset.rs` set the precedent for — and extended in place
/// for the item enum. A gid the subset dropped refuses the WHOLE line,
/// all-or-nothing: half a vertical title would be worse than a horizontal
/// one.
pub(super) fn to_cid_line(
    raw: RawVertical,
    title: String,
    fonts_out: &mut [PlannedFont],
    font_of_chain: &BTreeMap<usize, usize>,
    faces: &[Option<Face<'_>>],
) -> Option<VerticalLine> {
    let upright_font = font_of_chain.get(&raw.upright_chain_index).copied()?;
    let mut items = Vec::with_capacity(raw.items.len());
    let mut advance_1000 = 0.0_f32;
    for item in &raw.items {
        match item {
            RawVerticalItem::Upright(glyph) => {
                let planned = fonts_out.get(upright_font)?;
                let scale = 1000.0 / raw.upem;
                let cid = planned.gids.get(&glyph.old_gid).copied()?;
                let pitch_1000 = glyph.pitch_units * scale;
                let mut x_shift_1000 = glyph.x_shift_units * scale;
                if x_shift_1000.abs() < ADJUST_EPSILON_1000 {
                    x_shift_1000 = 0.0;
                }
                advance_1000 += pitch_1000;
                items.push(VerticalItem::Upright(VerticalGlyph {
                    cid,
                    pitch_1000,
                    x_shift_1000,
                }));
            }
            RawVerticalItem::Rotated {
                chain_index,
                run,
                tx_1000,
            } => {
                let font_index = font_of_chain.get(chain_index).copied()?;
                let upem = faces
                    .get(*chain_index)
                    .and_then(Option::as_ref)
                    .map_or(1000.0, |face| f32::from(face.units_per_em()));
                let planned = fonts_out.get_mut(font_index)?;
                let mut glyphs: Vec<RunGlyph> = Vec::new();
                let mut run_advance = 0.0_f32;
                for cluster in &run.clusters {
                    // D-B3 guaranteed one glyph per cluster and no offsets,
                    // so the `/ToUnicode` entry is exact and no span is
                    // needed — the line-level one covers the reading order.
                    let glyph = cluster.glyphs.first()?;
                    let cid = planned.gids.get(&glyph.old_gid).copied()?;
                    let width = planned.widths.get(&cid).copied().unwrap_or(0.0);
                    let advance = to_thousandths(glyph.advance_units, upem);
                    let base = planned.kern_base.get(&cid).copied().unwrap_or(width);
                    let mut adjust = base - advance;
                    if adjust.abs() < ADJUST_EPSILON_1000 {
                        adjust = 0.0;
                    }
                    planned
                        .to_unicode
                        .entry(cid)
                        .or_insert_with(|| cluster.text.clone());
                    run_advance += width - adjust;
                    glyphs.push(RunGlyph {
                        cid,
                        width_1000: width,
                        adjust_1000: adjust,
                        x_shift_1000: 0.0,
                        rise_1000: 0.0,
                    });
                }
                if glyphs.is_empty() {
                    return None;
                }
                advance_1000 += run_advance;
                items.push(VerticalItem::Rotated(RotatedRun {
                    run: GlyphRun {
                        font: font_index,
                        glyphs,
                        advance_1000: run_advance,
                        spans: Vec::new(),
                    },
                    tx_1000: *tx_1000,
                }));
            }
        }
    }
    Some(VerticalLine {
        font: upright_font,
        items,
        actual_text: title,
        advance_1000,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_item_sequence_is_exactly_what_the_shared_itemiser_cut() {
        // The ladder no longer refuses a rotated character: it itemises. The
        // sequence of item kinds MUST equal the shared itemiser's, or the
        // page and the map are reading two different tables.
        for text in [
            "\u{6771}\u{4EAC}",
            "\u{6771}\u{4EAC} 2026",
            "\u{6771}\u{4EAC}Tower",
            "Tokyo",
            "\u{300C}\u{3042}\u{3001}\u{300D}",
        ] {
            let runs = vertical_runs(text);
            let kinds: Vec<bool> = runs.iter().map(|run| run.is_upright()).collect();
            let rebuilt: String = runs
                .iter()
                .map(|run| run.text())
                .collect::<Vec<_>>()
                .concat();
            assert_eq!(rebuilt, text, "{text:?} must survive itemisation");
            assert!(!kinds.is_empty(), "{text:?} has at least one run");
        }
    }

    #[test]
    fn every_refusal_reason_says_something() {
        for refusal in [
            VerticalRefusal::RightToLeft,
            VerticalRefusal::MultipleFaces,
            VerticalRefusal::NoVerticalMetrics,
            VerticalRefusal::CffFace,
            VerticalRefusal::InexpressibleCluster,
            VerticalRefusal::ImplausiblePitch,
            VerticalRefusal::RotatedRunUnshapeable,
            VerticalRefusal::RotatedRunMultipleFaces,
            VerticalRefusal::RotatedRunTooWide,
            VerticalRefusal::Unusable,
        ] {
            assert!(!refusal.reason().is_empty());
        }
    }

    /// The bundled Noto is Latin-only, static, and has no `vmtx` — so it
    /// exercises three rungs of the ladder without a Windows font.
    fn noto() -> Vec<u8> {
        oxifont_bundled::NOTO_SANS_REGULAR.to_vec()
    }

    fn refuse(text: &str) -> VerticalRefusal {
        let bytes = noto();
        let chain = vec![bytes.clone()];
        let face = Face::parse(&bytes, 0).expect("the bundled face parses");
        let faces = vec![Some(face)];
        let coverage: BTreeMap<char, usize> = text.chars().map(|ch| (ch, 0)).collect();
        let mut shaper = SwashShaper::new();
        match plan_vertical(&mut shaper, &chain, &faces, &coverage, text) {
            Ok(_) => panic!("{text:?} was expected to refuse"),
            Err(refusal) => refusal,
        }
    }

    #[test]
    fn the_ladder_refuses_before_it_shapes() {
        assert_eq!(refuse("\u{645}\u{631}"), VerticalRefusal::RightToLeft);
        assert_eq!(refuse(""), VerticalRefusal::Unusable);
        // Upright characters, but the bundled face has no vmtx.
        assert_eq!(
            refuse("\u{6771}\u{4EAC}"),
            VerticalRefusal::NoVerticalMetrics,
        );
        // And a MIXED line still refuses on the upright half's missing vmtx,
        // even though its Latin run would shape fine.
        assert_eq!(
            refuse("\u{6771}\u{4EAC} 2026"),
            VerticalRefusal::NoVerticalMetrics,
        );
    }

    /// The design's explicit taste ruling: a title with nothing upright in it
    /// is a legal sideways column, not a refusal — and it never asks the face
    /// for vertical metrics it does not have.
    #[test]
    fn a_rotated_only_title_is_accepted_without_any_vertical_metrics() {
        let bytes = noto();
        let chain = vec![bytes.clone()];
        let face = Face::parse(&bytes, 0).expect("the bundled face parses");
        assert!(
            face.tables().vmtx.is_none(),
            "the bundled Noto is the no-vmtx fixture",
        );
        let faces = vec![Some(face)];
        let text = "Tokyo";
        let coverage: BTreeMap<char, usize> = text.chars().map(|ch| (ch, 0)).collect();
        let mut shaper = SwashShaper::new();
        let planned = plan_vertical(&mut shaper, &chain, &faces, &coverage, text)
            .unwrap_or_else(|refusal| panic!("a sideways column is legal: {refusal:?}"));
        assert_eq!(planned.items.len(), 1, "one run, one item");
        assert_eq!(planned.upem, 0.0, "no upright cell asked for an em box");
        let RawVerticalItem::Rotated { run, tx_1000, .. } = &planned.items[0] else {
            panic!("the only item must be the sideways run");
        };
        assert_eq!(run.clusters.len(), 5, "one cluster per letter");
        assert!(
            tx_1000.is_finite(),
            "the cross-column shift is a real number: {tx_1000}",
        );
    }

    #[test]
    fn a_line_whose_upright_cells_span_two_faces_refuses() {
        let bytes = noto();
        let chain = vec![bytes.clone(), bytes.clone()];
        let face = Face::parse(&bytes, 0).expect("the bundled face parses");
        let second = Face::parse(&bytes, 0).expect("the bundled face parses");
        let faces = vec![Some(face), Some(second)];
        let text = "\u{6771}\u{4EAC}";
        let mut coverage: BTreeMap<char, usize> = text.chars().map(|ch| (ch, 0)).collect();
        coverage.insert('\u{4EAC}', 1);
        let mut shaper = SwashShaper::new();
        assert!(matches!(
            plan_vertical(&mut shaper, &chain, &faces, &coverage, text),
            Err(VerticalRefusal::MultipleFaces),
        ));
    }

    #[test]
    fn a_rotated_run_spanning_two_faces_refuses() {
        // `runs_for` cuts a run at every coverage seam, so a Latin run split
        // across two chain faces comes back as two runs — which one text
        // matrix cannot position.
        let bytes = noto();
        let chain = vec![bytes.clone(), bytes.clone()];
        let face = Face::parse(&bytes, 0).expect("the bundled face parses");
        let second = Face::parse(&bytes, 0).expect("the bundled face parses");
        let faces = vec![Some(face), Some(second)];
        let text = "Tokyo";
        let mut coverage: BTreeMap<char, usize> = text.chars().map(|ch| (ch, 0)).collect();
        coverage.insert('y', 1);
        let mut shaper = SwashShaper::new();
        assert!(matches!(
            plan_vertical(&mut shaper, &chain, &faces, &coverage, text),
            Err(VerticalRefusal::RotatedRunMultipleFaces),
        ));
    }

    #[test]
    fn a_rotated_run_with_an_uncovered_character_refuses_rather_than_dropping_it() {
        // `runs_for` silently skips a character no face covers. A title
        // missing a letter is not a title, so the line refuses whole.
        let bytes = noto();
        let chain = vec![bytes.clone()];
        let face = Face::parse(&bytes, 0).expect("the bundled face parses");
        let faces = vec![Some(face)];
        let text = "Tokyo";
        let mut coverage: BTreeMap<char, usize> = text.chars().map(|ch| (ch, 0)).collect();
        coverage.remove(&'k');
        let mut shaper = SwashShaper::new();
        assert!(matches!(
            plan_vertical(&mut shaper, &chain, &faces, &coverage, text),
            Err(VerticalRefusal::RotatedRunUnshapeable),
        ));
    }

    #[test]
    fn the_cross_column_shift_centres_the_typographic_box() {
        // `tx_1000 = 500 − (asc + desc) / 2`, arithmetic re-verified against
        // the bundled face's own hhea rather than a remembered number.
        let bytes = noto();
        let face = Face::parse(&bytes, 0).expect("the bundled face parses");
        let upem = f32::from(face.units_per_em());
        let asc = to_thousandths(f32::from(face.ascender()), upem);
        let desc = to_thousandths(f32::from(face.descender()), upem);
        let expected = 500.0 - (asc + desc) / 2.0;
        let chain = vec![bytes.clone()];
        let faces = vec![Some(face)];
        let text = "Tokyo";
        let coverage: BTreeMap<char, usize> = text.chars().map(|ch| (ch, 0)).collect();
        let mut shaper = SwashShaper::new();
        let planned = plan_vertical(&mut shaper, &chain, &faces, &coverage, text)
            .unwrap_or_else(|refusal| panic!("a sideways column is legal: {refusal:?}"));
        let RawVerticalItem::Rotated { tx_1000, .. } = &planned.items[0] else {
            panic!("the only item must be the sideways run");
        };
        assert!(
            (tx_1000 - expected).abs() < 0.001,
            "{tx_1000} should centre at {expected}",
        );
        // And the bundled face is well inside the too-wide bound.
        assert!(asc - desc <= MAX_ROTATED_RUN_EM * 1000.0);
    }

    #[test]
    fn the_too_wide_bound_is_the_margin_strips_arithmetic() {
        // The margin strip's own arithmetic: PAGE_MARGIN_PT = 36 and
        // TITLE_FONT_PT = 16 put the em cell inside a 36 pt strip, so a run
        // centred on the column centre line has 36/16 = 2.25 em before it
        // leaves the paper. The bound must stay under that.
        let available_em = super::super::PAGE_MARGIN_PT / super::super::TITLE_FONT_PT;
        assert!(
            MAX_ROTATED_RUN_EM < available_em,
            "{MAX_ROTATED_RUN_EM} em must fit inside {available_em} em of margin",
        );
    }

    #[test]
    #[ignore = "reads C:/Windows/Fonts/YuGothM.ttc and meiryo.ttc; the D-V4 goldens"]
    fn live_windows_vertical_substitution_golden() {
        // 「あ、」 — the design's measured line: two Tr brackets, one Upright
        // kana and one Tu comma. Every measured Windows CJK face substitutes
        // the vertical forms of the punctuation, so the bracket and comma
        // gids MUST differ from their horizontal counterparts while the kana
        // stays put. A face that substitutes nothing is a known state
        // (simsun) — but not for these two.
        let text = "\u{300C}\u{3042}\u{3001}\u{300D}";
        for name in ["YuGothM.ttc", "meiryo.ttc"] {
            let Ok(bytes) = std::fs::read(format!("C:/Windows/Fonts/{name}")) else {
                continue;
            };
            let chain = vec![bytes.clone()];
            let Ok(face) = Face::parse(&bytes, 0) else {
                continue;
            };
            let horizontal: Vec<u16> = text
                .chars()
                .map(|ch| face.glyph_index(ch).map_or(0, |gid| gid.0))
                .collect();
            let faces = vec![Some(face)];
            let coverage: BTreeMap<char, usize> = text.chars().map(|ch| (ch, 0)).collect();
            let mut shaper = SwashShaper::new();
            let planned = plan_vertical(&mut shaper, &chain, &faces, &coverage, text)
                .unwrap_or_else(|refusal| panic!("{name} must set 「あ、」: {refusal:?}"));
            let cells: Vec<&RawVerticalGlyph> = planned
                .items
                .iter()
                .map(|item| match item {
                    RawVerticalItem::Upright(glyph) => glyph,
                    RawVerticalItem::Rotated { .. } => {
                        panic!("{name}: 「あ、」 is upright throughout")
                    }
                })
                .collect();
            assert_eq!(cells.len(), 4, "{name}: one glyph per character");
            let vertical: Vec<u16> = cells.iter().map(|g| g.old_gid).collect();
            assert_ne!(
                vertical[0], horizontal[0],
                "{name}: 「 must substitute its vertical form",
            );
            assert_ne!(
                vertical[3], horizontal[3],
                "{name}: 」 must substitute its vertical form",
            );
            assert_ne!(
                vertical[2], horizontal[2],
                "{name}: 、 must substitute its vertical form",
            );
            assert_eq!(
                vertical[1], horizontal[1],
                "{name}: あ is Upright and must NOT substitute",
            );
            // Every pitch inside the sanity window, and full-width cells
            // centre by exactly zero.
            for glyph in &cells {
                assert!(glyph.pitch_units >= MIN_PITCH_EM * planned.upem);
                assert!(glyph.pitch_units <= MAX_PITCH_EM * planned.upem);
            }
        }
    }
}
