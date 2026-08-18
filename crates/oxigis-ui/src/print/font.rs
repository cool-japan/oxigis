// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Embedded fonts for the PDF export: the font-chain walk, subsetting, and
//! everything [`super::pdf_document`] needs to write `/Type0` composite
//! fonts (`Identity-H`, `CIDFontType2` for glyf faces, `CIDFontType0` +
//! bare-CFF `FontFile3` for CFF faces).
//!
//! # How a character becomes a CID
//!
//! [`plan`] walks the shell-supplied font chain **per character**: the first
//! face whose `cmap` answers gets the character. Each used face is then
//! subsetted with [`oxifont_subset`], and the `SubsetGidMap` that call
//! returns **is** the PDF CID assignment — the subset font is written with
//! `/CIDToGIDMap /Identity`, so CID == subset GID by construction, and `/W`
//! widths come from the embedded program's `hmtx` scaled to the PDF's
//! 1000/em text space.
//!
//! # Shaped since v1.2
//!
//! Each same-face run of every planned string is shaped through
//! [`super::shape`] (swash via `oxitext`, the same engine the on-screen
//! labels use), so kerning and ligatures land on the page: glyph ids come
//! from shaping (a ligature gid is unreachable from any cmap walk), `/W`
//! stays the ORIGINAL `hmtx` (context-free, per CID) and the pair-contextual
//! kern delta rides in the `TJ` numbers. A run the shaper cannot express as
//! a plain `TJ` with exact `/ToUnicode` — marks, n:m clusters, RTL — keeps
//! the v1.1 per-character walk, so nothing ever degrades below what v1.1
//! printed. Width measurement is DEFINED as the pen movement the emitter
//! writes, so label collision boxes cannot disagree with the render.
//!
//! # Substitution rule
//!
//! A character no face covers prints as the first covering face's `?` (and
//! is counted, so the caller can say so); if not even `?` exists anywhere in
//! the chain the character is skipped. An empty or unusable chain disables
//! the whole plan and [`super::page_content`] keeps its Base-14 Helvetica
//! degraded mode.
//!
//! # The subset is PDF-only
//!
//! The embedded program carries an empty `cmap`, no layout tables and no
//! variation tables, and its glyphs are renumbered — correct for an
//! `Identity-H` composite font, but not usable as a standalone font file.
//! Metrics for the `/FontDescriptor` are therefore read from the *original*
//! face, never the subset. See [`super::subset`] for the exact profile.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use oxigis_core::LabelWeight;
use ttf_parser::Face;

use super::vertical::{self, VerticalLine};
use super::{instance, shape, subset};

/// The subset machinery moved to `print::subset` (D-W5's pure move); the
/// planned-font type is re-exported here because [`TextPlan::fonts`] is its
/// only public entry point.
pub use super::subset::PlannedFont;

/// Hard cap on glyphs requested from one face — far above any real page
/// (a title + footer is tens of characters) and far below the `u16` glyph-id
/// space a subset must renumber into.
const MAX_SUBSET_GLYPHS: usize = 4096;

/// TJ adjustments smaller than this (in thousandths of an em) are clamped to
/// exactly `0.0` — invisible on paper (0.00016 pt at 16 pt) and it keeps the
/// writer from emitting near-zero reals. Applied ONCE, at plan time, so the
/// emitter and the measurement read the same stored number.
pub(super) const ADJUST_EPSILON_1000: f32 = 0.01;

/// Design units → 1000/em text space — the ONE conversion `/W`, the shaped
/// advances and the TJ deltas all go through, so an unkerned glyph's delta
/// is exactly `0.0` and no adjustment is written.
pub(super) fn to_thousandths(units: f32, upem: f32) -> f32 {
    units * (1000.0 / upem)
}

/// The character substituted for anything the chain cannot draw.
const SUBSTITUTE: char = '?';

/// Whether a chain face is also what the on-screen labels draw with —
/// which decides whether the export may instance it (print v1.3 item C).
///
/// The screen provably CANNOT render a non-default fvar instance (oxitext's
/// `shape_with_variations` discards its argument; fontdue has no variation
/// support), so instancing a [`FaceRole::ScreenShared`] face would make the
/// PDF disagree with the map the user is looking at. Only a
/// [`FaceRole::PrintOnly`] face — fetched for the export alone, never
/// rasterised on screen — is safe to normalise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceRole {
    /// The label engine draws with this face too — never instanced.
    ScreenShared,
    /// Export-only (the web `data-oxigis-print-font` face) — a variable
    /// face is normalised to the instance nearest weight 400.
    PrintOnly,
}

/// One chain entry with its provenance.
pub struct PrintFace {
    /// The face bytes (sfnt or whole `.ttc`; face 0 is used).
    pub bytes: Vec<u8>,
    /// Whether the screen shares this face.
    pub role: FaceRole,
    /// Which weight slot the face fills (print/text v1.4). Bold-slot entries
    /// are consulted FIRST by a `Bold` request and never by a `Regular` one.
    pub weight: LabelWeight,
}

impl PrintFace {
    /// A regular-slot face.
    #[must_use]
    pub fn regular(bytes: Vec<u8>, role: FaceRole) -> Self {
        Self {
            bytes,
            role,
            weight: LabelWeight::Regular,
        }
    }

    /// A bold-slot face.
    #[must_use]
    pub fn bold(bytes: Vec<u8>, role: FaceRole) -> Self {
        Self {
            bytes,
            role,
            weight: LabelWeight::Bold,
        }
    }
}

/// The font byte chain a shell supplies for one export, highest priority
/// first. Entries may be plain sfnt files **or whole `.ttc` collections** —
/// face 0 is used, matching the label pipeline's convention. An empty chain
/// selects the Base-14 Helvetica degraded mode.
#[derive(Default)]
pub struct PrintFonts {
    /// Candidate faces, in fallback order — regular slots and bold slots in
    /// ONE vector, so the `F` numbering and the subset pass do not have to
    /// know about weight at all (`PrintFonts::walk_order` is the only place
    /// that does).
    pub chain: Vec<Vec<u8>>,
    /// Role per chain entry, index-aligned; a missing entry reads as
    /// [`FaceRole::ScreenShared`] — the never-instanced safe default.
    roles: Vec<FaceRole>,
    /// Weight slot per chain entry, index-aligned; a missing entry reads as
    /// [`LabelWeight::Regular`] — so every pre-v1.4 constructor is unchanged.
    weights: Vec<LabelWeight>,
}

impl PrintFonts {
    /// No fonts: the degraded Helvetica mode.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// A chain of candidate faces, highest priority first — every entry
    /// [`FaceRole::ScreenShared`], i.e. never instanced: every existing
    /// caller keeps today's bytes.
    #[must_use]
    pub fn new(chain: Vec<Vec<u8>>) -> Self {
        let roles = vec![FaceRole::ScreenShared; chain.len()];
        let weights = vec![LabelWeight::Regular; chain.len()];
        Self {
            chain,
            roles,
            weights,
        }
    }

    /// A chain with explicit per-face roles and weight slots.
    #[must_use]
    pub fn with_roles(faces: Vec<PrintFace>) -> Self {
        let roles = faces.iter().map(|face| face.role).collect();
        let weights = faces.iter().map(|face| face.weight).collect();
        let chain = faces.into_iter().map(|face| face.bytes).collect();
        Self {
            chain,
            roles,
            weights,
        }
    }

    /// The role of chain entry `index`.
    fn role(&self, index: usize) -> FaceRole {
        self.roles
            .get(index)
            .copied()
            .unwrap_or(FaceRole::ScreenShared)
    }

    /// The weight slot of chain entry `index`.
    fn weight_of(&self, index: usize) -> LabelWeight {
        self.weights
            .get(index)
            .copied()
            .unwrap_or(LabelWeight::Regular)
    }

    /// Chain indices in the order `weight` consults them.
    ///
    /// `Regular` walks the regular slots only — a bold face must never draw
    /// a regular label. `Bold` walks the bold slots first and then **the
    /// whole regular chain**: never-shrink, so a bold Latin face that has no
    /// CJK block falls through to the regular CJK face for those characters.
    /// The line is then mixed-weight, which is honest; `.notdef` would not
    /// be.
    fn walk_order(&self, weight: LabelWeight) -> Vec<usize> {
        let regular =
            (0..self.chain.len()).filter(|&index| self.weight_of(index) == LabelWeight::Regular);
        match weight {
            LabelWeight::Regular => regular.collect(),
            LabelWeight::Bold => (0..self.chain.len())
                .filter(|&index| self.weight_of(index) == LabelWeight::Bold)
                .chain(regular)
                .collect(),
        }
    }
}

/// One glyph exactly as the content stream emits it.
#[derive(Clone, Debug, PartialEq)]
pub struct RunGlyph {
    /// Subset glyph id, which is also the PDF CID (`/CIDToGIDMap /Identity`).
    pub cid: u16,
    /// The `/W` entry for the CID, in thousandths — what the viewer's pen
    /// advances for the shown glyph.
    pub width_1000: f32,
    /// The `TJ` number written AFTER this glyph, in thousandths, already in
    /// the operator's sign convention (positive moves the pen LEFT, i.e.
    /// tightens). Exactly `0.0` means nothing is written — an unkerned run
    /// emits a plain `Tj`, byte-identical to v1.1.
    pub adjust_1000: f32,
    /// Horizontal placement shift in thousandths (a combining mark's
    /// offset). Rides a PAIRED pair of `TJ` numbers — `−x` before the
    /// glyph, `+x` folded into the number after — so the NET pen movement
    /// is untouched and `advance_1000` needs no offset awareness. `0.0` on
    /// every plain path writes nothing.
    pub x_shift_1000: f32,
    /// Vertical rise in thousandths, emitted as `Ts` (`rise/1000·size`) and
    /// ALWAYS reset to `0 Ts` — `Ts` is state `BT` does not clear. `0.0`
    /// writes nothing.
    pub rise_1000: f32,
}

/// One `/ActualText` span over a glyph range of a run: what a conformant
/// extractor reads instead of the per-CID `/ToUnicode` — the only honest
/// answer for a multi-glyph cluster (a reordering script's per-glyph map
/// would be a lie; the v1.2 `group == chars` zip was measured wrong).
#[derive(Clone, Debug, PartialEq)]
pub struct ActualSpan {
    /// First glyph index of the span (inclusive).
    pub start: usize,
    /// Past-the-end glyph index.
    pub end: usize,
    /// The logical source text the span draws.
    pub text: String,
}

/// A maximal same-face span of one planned string.
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphRun {
    /// Index into [`TextPlan::fonts`]; the page resource is `F{font + 1}`.
    pub font: usize,
    /// Glyphs in VISUAL order (== logical for LTR; an RTL run arrives
    /// already reversed cluster-wise from the shaping pass).
    pub glyphs: Vec<RunGlyph>,
    /// The pen movement this run causes, in thousandths — DEFINED as
    /// `Σ (width_1000 − adjust_1000)` over exactly the stored numbers the
    /// emitter writes, so measurement equals rendering by construction
    /// (`x_shift_1000` nets to zero by the paired emission).
    pub advance_1000: f32,
    /// Per-cluster `/ActualText` spans, glyph-indexed, sorted and disjoint.
    /// Ignored by the emitter when the whole line carries a line-level span
    /// (a bidi line — see [`TextPlan::actual_text`]).
    pub spans: Vec<ActualSpan>,
}

/// The whole export's text plan: which face draws which character, with
/// every subset built. Produced by the module's planning pass, consumed by
/// the content-stream text emitter and the font-object writer.
pub struct TextPlan {
    /// Embedded faces in chain order; resource names are `F1`, `F2`, … by
    /// position (`F0` stays the Helvetica degraded-mode resource).
    pub fonts: Vec<PlannedFont>,
    /// Distinct characters that print as `?` because no face covers them.
    pub substituted: usize,
    /// (weight, character) → (index into [`Self::fonts`], CID). Keyed by
    /// weight since v1.4: the same character at two weights is two different
    /// CIDs in two different subsets.
    assignment: BTreeMap<(LabelWeight, char), (usize, u16)>,
    /// Every planned (weight, string) → its shaped runs, keyed by the
    /// ORIGINAL text. A string absent here (a bidi refusal, a never-planned
    /// caller) takes the synthetic per-character path — correct, merely
    /// unkerned.
    lines: BTreeMap<(LabelWeight, String), Vec<GlyphRun>>,
    /// Bidi lines → their LOGICAL resolved string: the emitter wraps the
    /// whole line in ONE `/Span <</ActualText …>>` (per-run spans would
    /// concatenate in VISUAL order and extract a mixed line wrong —
    /// measured), and it is also what makes the mirrored-bracket
    /// `/ToUnicode` lie harmless. If this line-level rule is ever
    /// narrowed, mirroring must be narrowed with it.
    line_spans: BTreeMap<(LabelWeight, String), String>,
    /// The page title set VERTICALLY (print/text v1.4, D-V1..D-V4), when the
    /// export asked for it AND the refusal ladder accepted the line. [`None`]
    /// — including whenever the option is off — means the title prints
    /// horizontally, byte for byte as it always has.
    vertical: Option<VerticalLine>,
    /// Accepted vertical LABEL columns (print v1.6), keyed exactly like
    /// [`Self::lines`]. A `(weight, text)` absent here either never asked for
    /// a column or the ladder refused it; the label then prints horizontally,
    /// which is why every vertical string is also planned the ordinary way.
    verticals: BTreeMap<(LabelWeight, String), VerticalLine>,
}

impl TextPlan {
    /// The (font index, CID) drawing `ch` at `weight`, after substitution.
    /// [`None`] means the character is skipped entirely (not even `?`
    /// exists).
    #[must_use]
    pub fn glyph(&self, weight: LabelWeight, ch: char) -> Option<(usize, u16)> {
        self.assignment.get(&(weight, ch)).copied()
    }

    /// The runs drawing `text` at `weight`: the shaped ones when the plan
    /// saw the (weight, string) pair, a synthesized per-character fallback
    /// otherwise (no panic, no blank — a fallback glyph is always in the
    /// subset because the remapper took the union of shaped and cmap gids).
    #[must_use]
    pub fn runs(&self, weight: LabelWeight, text: &str) -> Cow<'_, [GlyphRun]> {
        // `BTreeMap<(LabelWeight, String), _>` cannot be probed with a
        // borrowed key, and a page has tens of lines: one owned key per
        // lookup is not worth an indirection layer.
        match self.lines.get(&(weight, text.to_string())) {
            Some(runs) => Cow::Borrowed(runs.as_slice()),
            None => Cow::Owned(synthetic_runs(&self.assignment, &self.fonts, weight, text)),
        }
    }

    /// The logical text of a bidi line — [`Some`] means the emitter wraps
    /// the whole line in one `/Span <</ActualText …>>` marked-content
    /// sequence, so extraction yields the logical string despite the
    /// visual-order CIDs (and despite mirrored brackets).
    #[must_use]
    pub fn actual_text(&self, weight: LabelWeight, text: &str) -> Option<&str> {
        self.line_spans
            .get(&(weight, text.to_string()))
            .map(String::as_str)
    }

    /// The page title as a vertical line, when one was planned and accepted.
    #[must_use]
    pub fn vertical_title(&self) -> Option<&VerticalLine> {
        self.vertical.as_ref()
    }

    /// The accepted vertical column for one LABEL, or [`None`] when the label
    /// never asked for one or the ladder refused it — in which case the
    /// caller measures and draws the ordinary horizontal line.
    #[must_use]
    pub fn vertical_line(&self, weight: LabelWeight, text: &str) -> Option<&VerticalLine> {
        // Same owned-key probe as `runs`: a `BTreeMap` with a tuple key
        // cannot be looked up by a borrowed half, and a page holds hundreds
        // of labels, not millions.
        self.verticals.get(&(weight, text.to_string()))
    }

    /// The width `text` advances at `size` pt and `weight` — DEFINED as the
    /// pen movement the emitter's operators cause (`Σ advance_1000`), so the
    /// label collision boxes use the same numbers the page actually renders.
    /// Bold and Regular therefore measure differently, which is the point.
    #[must_use]
    pub fn width_pt(&self, weight: LabelWeight, text: &str, size: f32) -> f32 {
        self.runs(weight, text)
            .iter()
            .map(|run| run.advance_1000)
            .sum::<f32>()
            / 1000.0
            * size
    }
}

/// The v1.1 per-character runs for `text` — the path a refused or
/// never-planned string takes: every adjust is `0.0`, widths from `/W`.
fn synthetic_runs(
    assignment: &BTreeMap<(LabelWeight, char), (usize, u16)>,
    fonts: &[PlannedFont],
    weight: LabelWeight,
    text: &str,
) -> Vec<GlyphRun> {
    let mut runs: Vec<GlyphRun> = Vec::new();
    for ch in text.chars() {
        let Some(&(font, cid)) = assignment.get(&(weight, ch)) else {
            continue;
        };
        let width = fonts
            .get(font)
            .and_then(|planned| planned.widths.get(&cid))
            .copied()
            .unwrap_or(0.0);
        let glyph = RunGlyph {
            cid,
            width_1000: width,
            adjust_1000: 0.0,
            x_shift_1000: 0.0,
            rise_1000: 0.0,
        };
        match runs.last_mut() {
            Some(run) if run.font == font => {
                run.advance_1000 += width;
                run.glyphs.push(glyph);
            }
            _ => runs.push(GlyphRun {
                font,
                glyphs: vec![glyph],
                advance_1000: width,
                spans: Vec::new(),
            }),
        }
    }
    runs
}

/// How a **Bold** request is served by this chain — ONE exhaustive ladder
/// (print/text v1.4, D-W6), resolved once per export and logged once.
///
/// Every rung below L1 draws Regular ink. That is deliberate and it is the
/// whole reason the ladder is a named type rather than a chain of `if`s:
/// bold silently becoming regular is the failure mode a user would otherwise
/// have no way to diagnose, so each rung carries its own reason and the
/// aggregated log names it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum WeightLadder {
    /// **L0** — a real bold face is in the chain (weight class ≥
    /// [`BOLD_WEIGHT_CLASS`]): bold everywhere, nothing instanced.
    RealBoldFace {
        /// Chain index of the face.
        chain_index: usize,
        /// Its weight class, for the log.
        weight: f32,
    },
    /// **L1** — a variable, [`FaceRole::PrintOnly`] bold-slot face,
    /// instanced at `target_weight(Bold)`. PrintOnly is asserted, not
    /// assumed: the screen provably cannot render a non-default instance, so
    /// instancing a shared face would put the page and the map at different
    /// weights.
    Instanced {
        /// Chain index of the face.
        chain_index: usize,
        /// The `wght` coordinate chosen.
        weight: f32,
    },
    /// **L2** — a `wght` axis that cannot reach bold: it clamps below
    /// [`BOLD_WEIGHT_CLASS`], its nearest instance IS the fvar default, or
    /// the face is [`FaceRole::ScreenShared`] and must not be instanced.
    /// Regular bytes, logged.
    AxisCannotReachBold {
        /// Chain index of the face.
        chain_index: usize,
    },
    /// **L3** — a bold-slot face with no `wght` axis and no bold weight
    /// class. Regular bytes.
    NoWeightAxis {
        /// Chain index of the face.
        chain_index: usize,
    },
    /// **L4** — the chain carries no bold face at all. Regular bytes; the
    /// shell never fetched one, or the platform ships none.
    NoBoldFace,
}

/// The OS/2 `usWeightClass` (and fvar `wght`) at or above which a face is
/// accepted as genuinely bold. 600 rather than 700 so a SemiBold-only family
/// still prints heavier than its regular — which is what the user asked for
/// — while 500 (Medium) does not masquerade as bold.
pub(super) const BOLD_WEIGHT_CLASS: f32 = 600.0;

impl WeightLadder {
    /// Whether bold ink actually reaches the page — the ladder's own
    /// summary, asserted by the per-arm tests (production code branches on
    /// the arm itself, never on this).
    #[cfg(test)]
    #[must_use]
    pub(super) fn draws_bold(self) -> bool {
        matches!(self, Self::RealBoldFace { .. } | Self::Instanced { .. })
    }

    /// The ONE aggregated log line for this export.
    fn log(self) {
        match self {
            Self::RealBoldFace {
                chain_index,
                weight,
            } => tracing::info!(
                chain_index,
                weight,
                "oxigis-ui print: bold text draws through a real bold face",
            ),
            Self::Instanced {
                chain_index,
                weight,
            } => tracing::info!(
                chain_index,
                weight,
                "oxigis-ui print: bold text draws through a print-only variable \
                 face instanced at its nearest-Bold weight",
            ),
            Self::AxisCannotReachBold { chain_index } => tracing::warn!(
                chain_index,
                "oxigis-ui print: the bold face's wght axis cannot reach bold \
                 (it clamps below it, its nearest instance is the default, or \
                 the screen shares the face and instancing would break \
                 page/screen parity); bold text prints at Regular",
            ),
            Self::NoWeightAxis { chain_index } => tracing::warn!(
                chain_index,
                "oxigis-ui print: the bold face carries no wght axis and no \
                 bold weight class; bold text prints at Regular",
            ),
            Self::NoBoldFace => tracing::warn!(
                "oxigis-ui print: bold text was requested but the chain has no \
                 bold face; it prints at Regular (never a synthetic bold)",
            ),
        }
    }
}

/// Resolves how a Bold request is served — the ONE place the ladder is
/// decided, so no caller can invent a sixth rung.
///
/// The chain's bold slot is whatever the shell tagged [`LabelWeight::Bold`];
/// the first such entry decides, because it is also the first face the bold
/// coverage walk consults.
pub(super) fn resolve_weight(fonts: &PrintFonts, faces: &[Option<Face<'_>>]) -> WeightLadder {
    let Some((chain_index, face)) = fonts
        .chain
        .iter()
        .enumerate()
        .find(|&(index, _)| fonts.weight_of(index) == LabelWeight::Bold)
        .and_then(|(index, _)| {
            faces
                .get(index)
                .and_then(Option::as_ref)
                .map(|f| (index, f))
        })
    else {
        return WeightLadder::NoBoldFace;
    };
    let declared = subset::default_weight(face);
    if declared >= BOLD_WEIGHT_CLASS {
        // L0. A real bold face — static or variable-with-a-bold-default —
        // needs no instancing at all, so parity holds for free.
        return WeightLadder::RealBoldFace {
            chain_index,
            weight: declared,
        };
    }
    if !face.is_variable() {
        return WeightLadder::NoWeightAxis { chain_index };
    }
    // L1 REQUIRES PrintOnly. A `ScreenShared` variable face is never
    // instanced: `oxitext` cannot shape a non-default instance, so the map
    // would keep drawing the default while the page went bold.
    if fonts.role(chain_index) != FaceRole::PrintOnly {
        return WeightLadder::AxisCannotReachBold { chain_index };
    }
    let bytes = match fonts.chain.get(chain_index) {
        Some(bytes) => bytes,
        None => return WeightLadder::NoBoldFace,
    };
    let chosen = instance::raw_fvar(bytes).and_then(|fvar| {
        instance::choose_instance(face, fvar, instance::target_weight(LabelWeight::Bold))
    });
    match chosen {
        // `choose_instance` answers `None` when the nearest instance IS the
        // fvar default — the byte-identity gate — which for a bold request
        // means the axis has nothing bolder to offer.
        Some(chosen) if chosen.weight >= BOLD_WEIGHT_CLASS => WeightLadder::Instanced {
            chain_index,
            weight: chosen.weight,
        },
        _ => WeightLadder::AxisCannotReachBold { chain_index },
    }
}

/// [`plan_with_verticals`] with no vertical LABEL columns — the pre-v1.6
/// signature, kept for the tests that have nothing to say about them.
/// `cfg(test)`, because production reaches the planner through
/// [`plan_with_verticals`] and an unused second entry point would be dead
/// code rather than an API.
#[cfg(test)]
#[must_use]
pub fn plan(
    fonts: &PrintFonts,
    texts: &[(LabelWeight, &str)],
    vertical_title: Option<&str>,
) -> Option<TextPlan> {
    plan_with_verticals(fonts, texts, vertical_title, &[])
}

/// Builds the export's text plan from the shell's font chain and every
/// `(weight, string)` the page will show. Returns [`None`] — the degraded
/// Helvetica mode — when the chain is empty or no face in it is usable.
///
/// # Weight (print/text v1.4, D-W5)
///
/// The plan is keyed by `(weight, text)` throughout. The **second pass runs
/// only when something is actually Bold**: with an all-Regular text set no
/// bold coverage walk happens, no bold face is shaped or subsetted, and the
/// resulting PDF is byte-identical to what a chain without bold faces
/// produces — pinned by a test, because that is the floor every existing
/// project sits on.
///
/// # Vertical columns (print v1.6)
///
/// `vertical_labels` are the label strings a Symbol style asked to set
/// VERTICALLY, each planned as its own column against **its own weight's**
/// coverage map. Every one of them is ALSO in `texts`: the ladder can refuse
/// a column and the fallback is the ordinary horizontal line, so both forms
/// have to be planned. An empty `vertical_labels` runs no vertical label
/// pass at all — the byte-identity gate the vertical title already had.
#[must_use]
pub fn plan_with_verticals(
    fonts: &PrintFonts,
    texts: &[(LabelWeight, &str)],
    vertical_title: Option<&str>,
    vertical_labels: &[(LabelWeight, &str)],
) -> Option<TextPlan> {
    if fonts.chain.is_empty() {
        return None;
    }
    // Faces are parsed ONCE and kept: the coverage walk, the shaping pass and
    // the synthetic fallback all need them again.
    let faces: Vec<Option<Face<'_>>> = fonts.chain.iter().map(|bytes| usable_face(bytes)).collect();
    for (chain_index, face) in faces.iter().enumerate() {
        if face.is_none() {
            tracing::warn!(
                chain_index,
                "oxigis-ui print: font chain entry is not a usable face; skipped",
            );
        }
    }
    let bold_requested = texts
        .iter()
        .any(|&(weight, text)| weight == LabelWeight::Bold && !text.is_empty());
    if bold_requested {
        resolve_weight(fonts, &faces).log();
    }
    // THE byte-identity gate: one weight unless the page really has bold.
    let active: &[LabelWeight] = if bold_requested {
        &[LabelWeight::Regular, LabelWeight::Bold]
    } else {
        &[LabelWeight::Regular]
    };

    // Pass 1 — coverage, per active weight: the first face in THAT weight's
    // walk order whose cmap answers gets the character. Regular walks the
    // regular faces; Bold walks the bold faces and then the whole regular
    // chain (never-shrink), so a bold face lacking a CJK block yields a
    // mixed-weight line instead of `.notdef`.
    let mut coverages: BTreeMap<LabelWeight, BTreeMap<char, usize>> = BTreeMap::new();
    // chain index → the page characters it draws, with their ORIGINAL gids.
    // A union across weights: one subset per chain face, one `F` number.
    let mut face_chars: BTreeMap<usize, BTreeMap<char, u16>> = BTreeMap::new();
    for &weight in active {
        let order = fonts.walk_order(weight);
        let mut needed: Vec<char> = Vec::new();
        let mut seen = BTreeSet::new();
        for &(text_weight, text) in texts {
            if text_weight != weight {
                continue;
            }
            for ch in text.chars() {
                if ch >= ' ' && seen.insert(ch) {
                    needed.push(ch);
                }
            }
        }
        if seen.insert(SUBSTITUTE) {
            needed.push(SUBSTITUTE);
        }
        let covering = |ch: char| -> Option<usize> {
            order.iter().copied().find(|&index| {
                faces
                    .get(index)
                    .and_then(Option::as_ref)
                    .is_some_and(|face| face.glyph_index(ch).is_some())
            })
        };
        let mut coverage: BTreeMap<char, usize> = BTreeMap::new();
        for ch in needed {
            let Some(index) = covering(ch) else {
                continue;
            };
            let Some(Some(face)) = faces.get(index) else {
                continue;
            };
            let Some(gid) = face.glyph_index(ch) else {
                continue;
            };
            coverage.insert(ch, index);
            face_chars.entry(index).or_default().insert(ch, gid.0);
        }
        // v1.4 (D-M2): the bidi path mirrors a character only onto a partner
        // the SAME face covers, so the map must be able to answer for
        // partners that are not themselves on the page. They resolve through
        // the very same walk and enter NEITHER `face_chars` NOR any subset
        // request — an LTR page's bytes are untouched, and a partner glyph
        // still reaches the subset the way it always did, via `shaped_gids`.
        let partners: Vec<char> = coverage
            .keys()
            .filter_map(|&ch| super::bidi::mirror(ch))
            .filter(|partner| !coverage.contains_key(partner))
            .collect();
        for partner in partners {
            if let Some(index) = covering(partner) {
                coverage.insert(partner, index);
            }
        }
        coverages.insert(weight, coverage);
    }

    // Pass 2 — shape every planned string at its own weight: substitution is
    // resolved FIRST (the resolved string is what the page really draws, so
    // `/ToUnicode` honestly says '?'), then each maximal same-face run is
    // shaped. Glyph ids come from shaping — a ligature gid is unreachable
    // from any cmap.
    let mut shaper = oxitext::SwashShaper::new();
    let mut raw_lines: Vec<(LabelWeight, &str, Vec<shape::RawRun>, Option<String>)> = Vec::new();
    let mut shaped_gids: BTreeMap<usize, BTreeSet<u16>> = BTreeMap::new();
    let mut refused_bidi = 0_usize;
    let mut complex_ltr = 0_usize;
    let mut planned_texts: BTreeSet<(LabelWeight, &str)> = BTreeSet::new();
    for &(weight, text) in texts {
        if !planned_texts.insert((weight, text)) {
            continue;
        }
        let Some(coverage) = coverages.get(&weight) else {
            continue;
        };
        let resolved = resolve_text(coverage, text);
        if resolved.is_empty() {
            continue;
        }
        if shape::has_complex_ltr(&resolved) {
            complex_ltr += 1;
        }
        let (runs, line_span) = if shape::has_rtl(&resolved) {
            // The v1.3 bidi path, all-or-nothing per string: a refusal
            // keeps the v1.1 per-character path (absent from `lines` ⇒
            // the synthetic fallback serves it). Refused here: explicit
            // bidi controls, any `?`-substituted character (its bidi
            // class would lie), a joined-script stretch split across
            // chain faces, and anything the classifier cannot express.
            let shaped = if text
                .chars()
                .any(|ch| ch >= ' ' && !coverage.contains_key(&ch))
            {
                // A substituted (or dropped) character changes the
                // level-run structure; the whole string refuses (D9.2).
                None
            } else {
                super::bidi::segments(&resolved, coverage).and_then(|segments| {
                    shape::runs_for_bidi(&mut shaper, &fonts.chain, coverage, &segments, &resolved)
                })
            };
            match shaped {
                Some(runs) => (runs, Some(resolved.clone())),
                None => {
                    refused_bidi += 1;
                    continue;
                }
            }
        } else {
            (
                shape::runs_for(&mut shaper, &fonts.chain, &faces, coverage, &resolved),
                None,
            )
        };
        for run in &runs {
            let set = shaped_gids.entry(run.chain_index).or_default();
            for glyph in run
                .clusters
                .iter()
                .flat_map(|cluster| cluster.glyphs.iter())
            {
                set.insert(glyph.old_gid);
            }
        }
        raw_lines.push((weight, text, runs, line_span));
    }
    // The vertical title (D-V2), planned against the REGULAR coverage map —
    // page furniture is never bold. Its glyphs join `shaped_gids` here so
    // the subset pass takes them like any other shaped gid; a refusal is
    // logged once and the title simply prints horizontally.
    let raw_vertical = vertical_title.and_then(|title| {
        let coverage = coverages.get(&LabelWeight::Regular)?;
        match vertical::plan_vertical(&mut shaper, &fonts.chain, &faces, coverage, title) {
            Ok(planned) => {
                collect_vertical_gids(&planned, &mut shaped_gids);
                Some((title.to_string(), planned))
            }
            Err(refusal) => {
                tracing::warn!(
                    reason = refusal.reason(),
                    "oxigis-ui print: the page title cannot be set vertically; \
                     it prints horizontally",
                );
                None
            }
        }
    });
    // The vertical LABEL columns (print v1.6), one per DISTINCT (weight,
    // string): a page repeating a place name shapes it once. Refusals are
    // tallied per reason and logged once each — a per-label log would be one
    // line per feature on a dense page.
    let mut raw_vertical_labels: Vec<(LabelWeight, String, vertical::RawVertical)> = Vec::new();
    let mut vertical_refusals: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut planned_verticals: BTreeSet<(LabelWeight, &str)> = BTreeSet::new();
    for &(weight, text) in vertical_labels {
        if text.is_empty() || !planned_verticals.insert((weight, text)) {
            continue;
        }
        let Some(coverage) = coverages.get(&weight) else {
            continue;
        };
        match vertical::plan_vertical(&mut shaper, &fonts.chain, &faces, coverage, text) {
            Ok(planned) => {
                collect_vertical_gids(&planned, &mut shaped_gids);
                raw_vertical_labels.push((weight, text.to_string(), planned));
            }
            Err(refusal) => *vertical_refusals.entry(refusal.reason()).or_default() += 1,
        }
    }
    for (reason, strings) in &vertical_refusals {
        tracing::warn!(
            reason,
            strings,
            "oxigis-ui print: label strings cannot be set vertically; they print horizontally",
        );
    }
    if refused_bidi > 0 {
        tracing::warn!(
            strings = refused_bidi,
            "oxigis-ui print: bidi strings kept the v1.1 unshaped path \
             (explicit controls, substitution, a face seam inside a joined \
             script, or an inexpressible cluster)",
        );
    }
    if complex_ltr > 0 {
        // v1.4 item 1's honesty half (the ruling itself is still DEFERRED —
        // see docs/plans/print-v14.md and the `_a_canary_not_a_complaint`
        // test in `shape`): Indic and South-East-Asian labels shape under the
        // LATIN script tag, which draws every character but reorders none, so
        // matras, conjuncts and reph come out in logical order. Saying so once
        // per export is what ships for this item.
        //
        // The message deliberately no longer names a shaper defect as the
        // reason. It used to say "swash 0.2.10 garbles and panics under a real
        // Indic tag" — true when written, false since the oxitext 0.2.2 bump
        // of 2026-08-11 fixed both (the canary in `shape` re-measures it). The
        // remaining reason is simply that itemisation has not landed here yet,
        // and a user-visible log must not blame an upstream that is now clean.
        tracing::warn!(
            strings = complex_ltr,
            "oxigis-ui print: strings in complex left-to-right scripts print \
             WITHOUT script-aware shaping (Latin GSUB only — no reordering, \
             no conjuncts); script itemisation has not landed on the page \
             path yet",
        );
    }

    // Pass 3 — subset each used face over the UNION of its cmap gids (the
    // fallback path must stay drawable) and its shaped gids; a failed
    // subset drops its characters to substitution rather than the whole
    // export. ONE subset per chain face however many weights use it, so the
    // `F` numbering is unchanged and a face shared by both weights is
    // embedded once.
    let mut fonts_out: Vec<PlannedFont> = Vec::new();
    let mut font_of_chain: BTreeMap<usize, usize> = BTreeMap::new();
    for (chain_index, chars) in &face_chars {
        let chars: Vec<(char, u16)> = chars.iter().map(|(&ch, &gid)| (ch, gid)).collect();
        let mut gids: BTreeSet<u16> = chars.iter().map(|&(_, gid)| gid).collect();
        if let Some(shaped) = shaped_gids.get(chain_index) {
            gids.extend(shaped.iter().copied());
        }
        if gids.len() > MAX_SUBSET_GLYPHS {
            tracing::warn!(
                chain_index,
                glyphs = gids.len(),
                "oxigis-ui print: glyph budget exceeded; face skipped",
            );
            continue;
        }
        let Some(bytes) = fonts.chain.get(*chain_index) else {
            continue;
        };
        let gids: Vec<u16> = gids.into_iter().collect();
        match subset::subset_face(
            bytes,
            &gids,
            &chars,
            fonts.role(*chain_index),
            fonts.weight_of(*chain_index),
        ) {
            Some(planned) => {
                font_of_chain.insert(*chain_index, fonts_out.len());
                fonts_out.push(planned);
            }
            None => {
                tracing::warn!(
                    chain_index,
                    "oxigis-ui print: face failed to subset; its characters degrade to '?'",
                );
            }
        }
    }
    if fonts_out.is_empty() {
        return None;
    }

    // The (weight, char) → (font, CID) table. A mirror partner is in
    // `coverage` but never in `face_chars`, so it has no `cids` entry and
    // lands here only if the page really drew it.
    let mut assignment: BTreeMap<(LabelWeight, char), (usize, u16)> = BTreeMap::new();
    for (&weight, coverage) in &coverages {
        for (&ch, chain_index) in coverage {
            let Some(&font_index) = font_of_chain.get(chain_index) else {
                continue;
            };
            let Some(&cid) = fonts_out
                .get(font_index)
                .and_then(|planned| planned.cids.get(&ch))
            else {
                continue;
            };
            assignment.insert((weight, ch), (font_index, cid));
        }
    }

    // Substitution: whatever still has no glyph borrows '?' where it landed,
    // per weight (the two weights can land on different faces).
    let mut substituted = 0;
    for &(weight, text) in texts {
        let substitute = assignment.get(&(weight, SUBSTITUTE)).copied();
        for ch in text.chars() {
            if ch >= ' ' && !assignment.contains_key(&(weight, ch)) {
                substituted += 1;
                if let Some(question) = substitute {
                    assignment.insert((weight, ch), question);
                }
            }
        }
    }

    // Pass 4 — translate the raw shaped runs into emitter-ready glyph runs:
    // CIDs through each face's old→new map, advances/offsets into
    // thousandths, the kern delta clamped once and stored (emitter and
    // measurement read the SAME number). A single-glyph cluster feeds
    // `/ToUnicode` (1:1 and n:1 ligatures are exact there); a multi-glyph
    // cluster contributes NOTHING per-glyph — it earns an `/ActualText`
    // span instead, and at most a document-wide-unique orphan overlay
    // below. A run whose face failed to subset falls back to the synthetic
    // per-character path over its own text.
    let mut lines: BTreeMap<(LabelWeight, String), Vec<GlyphRun>> = BTreeMap::new();
    let mut line_spans: BTreeMap<(LabelWeight, String), String> = BTreeMap::new();
    // (font, cid) → the ONE cluster text it appears under, or None once two
    // distinct texts claimed it (ambiguous — no overlay).
    let mut orphan_texts: BTreeMap<(usize, u16), Option<String>> = BTreeMap::new();
    for (weight, text, raw_runs, line_span) in raw_lines {
        let mut runs: Vec<GlyphRun> = Vec::new();
        let mut degraded = false;
        for raw in raw_runs {
            match font_of_chain.get(&raw.chain_index).copied() {
                Some(font_index) => {
                    let upem = faces
                        .get(raw.chain_index)
                        .and_then(Option::as_ref)
                        .map(|face| f32::from(face.units_per_em()))
                        .unwrap_or(1000.0);
                    let Some(planned) = fonts_out.get_mut(font_index) else {
                        continue;
                    };
                    let mut glyphs: Vec<RunGlyph> = Vec::new();
                    let mut spans: Vec<ActualSpan> = Vec::new();
                    let mut advance_total = 0.0_f32;
                    for cluster in &raw.clusters {
                        let span_start = glyphs.len();
                        let single = cluster.glyphs.len() == 1;
                        for glyph in &cluster.glyphs {
                            let Some(&cid) = planned.gids.get(&glyph.old_gid) else {
                                tracing::debug!(
                                    gid = glyph.old_gid,
                                    "oxigis-ui print: shaped gid missing from the subset; skipped",
                                );
                                continue;
                            };
                            let width = planned.widths.get(&cid).copied().unwrap_or(0.0);
                            let advance = to_thousandths(glyph.advance_units, upem);
                            // The kern delta lives in the DEFAULT instance's
                            // metric space (swash shaped the original bytes);
                            // `kern_base == widths` for a non-instanced face,
                            // so this is exactly `width − advance` there.
                            let base = planned.kern_base.get(&cid).copied().unwrap_or(width);
                            let mut adjust = base - advance;
                            if adjust.abs() < ADJUST_EPSILON_1000 {
                                adjust = 0.0;
                            }
                            let mut x_shift = to_thousandths(glyph.x_offset_units, upem);
                            if x_shift.abs() < ADJUST_EPSILON_1000 {
                                x_shift = 0.0;
                            }
                            let mut rise = to_thousandths(glyph.y_offset_units, upem);
                            if rise.abs() < ADJUST_EPSILON_1000 {
                                rise = 0.0;
                            }
                            if single {
                                planned
                                    .to_unicode
                                    .entry(cid)
                                    .or_insert_with(|| cluster.text.clone());
                            } else {
                                orphan_texts
                                    .entry((font_index, cid))
                                    .and_modify(|existing| {
                                        if existing.as_deref() != Some(cluster.text.as_str()) {
                                            *existing = None;
                                        }
                                    })
                                    .or_insert_with(|| Some(cluster.text.clone()));
                            }
                            advance_total += width - adjust;
                            glyphs.push(RunGlyph {
                                cid,
                                width_1000: width,
                                adjust_1000: adjust,
                                x_shift_1000: x_shift,
                                rise_1000: rise,
                            });
                        }
                        let needs_span = glyphs.len() > span_start
                            && (glyphs.len() - span_start > 1
                                || glyphs[span_start..].iter().any(|glyph| {
                                    glyph.x_shift_1000 != 0.0 || glyph.rise_1000 != 0.0
                                }));
                        if needs_span {
                            spans.push(ActualSpan {
                                start: span_start,
                                end: glyphs.len(),
                                text: cluster.text.clone(),
                            });
                        }
                    }
                    if !glyphs.is_empty() {
                        runs.push(GlyphRun {
                            font: font_index,
                            glyphs,
                            advance_1000: advance_total,
                            spans,
                        });
                    }
                }
                None => {
                    // The face never made it into the plan: its characters
                    // were re-routed to '?' above, and the synthetic path
                    // over the run's own text follows them there.
                    let text: String = raw
                        .clusters
                        .iter()
                        .map(|cluster| cluster.text.as_str())
                        .collect();
                    runs.extend(synthetic_runs(&assignment, &fonts_out, weight, &text));
                    degraded = true;
                }
            }
        }
        match line_span {
            // A bidi line that partially degraded lost its visual-order
            // guarantee — the whole line takes the synthetic path instead
            // (all-or-nothing, exactly as a shaping refusal).
            Some(_) if degraded => {}
            Some(span) => {
                lines.insert((weight, text.to_string()), runs);
                line_spans.insert((weight, text.to_string()), span);
            }
            None => {
                lines.insert((weight, text.to_string()), runs);
            }
        }
    }
    // The orphan overlay (rule 3): a conjunct/contextual CID reachable from
    // no cmap and no ligature gets its group text — only when that text is
    // unique document-wide, so the map never guesses.
    for ((font_index, cid), candidate) in orphan_texts {
        if let (Some(planned), Some(text)) = (fonts_out.get_mut(font_index), candidate) {
            planned.to_unicode.entry(cid).or_insert(text);
        }
    }

    // Translate the accepted vertical title into CIDs and 1000/em numbers.
    // The walk itself lives in `vertical::to_cid_line` (print/text v1.5,
    // D-B6) so this file keeps room for the planning passes.
    let vertical = raw_vertical.and_then(|(title, raw)| {
        vertical::to_cid_line(raw, title, &mut fonts_out, &font_of_chain, &faces)
    });
    // The same walk per accepted label column. A gid the subset dropped
    // refuses THAT column only — the label then prints horizontally, which
    // the `lines`/synthetic path already covers.
    let mut verticals: BTreeMap<(LabelWeight, String), VerticalLine> = BTreeMap::new();
    let mut dropped_columns = 0_usize;
    for (weight, text, raw) in raw_vertical_labels {
        match vertical::to_cid_line(raw, text.clone(), &mut fonts_out, &font_of_chain, &faces) {
            Some(line) => {
                verticals.insert((weight, text), line);
            }
            None => dropped_columns += 1,
        }
    }
    if dropped_columns > 0 {
        // The label pass counts these as refusals too, so the reason has to
        // be logged here or its aggregated warning would promise an
        // explanation that was never written.
        tracing::warn!(
            strings = dropped_columns,
            "oxigis-ui print: accepted vertical columns lost a glyph in subsetting; \
             those labels print horizontally",
        );
    }

    // Deterministic-by-construction, but a hash collision between two
    // subsets in one document would be a spec violation some viewers
    // handle badly — perturb until unique.
    subset::dedupe_base_font_names(&mut fonts_out);

    Some(TextPlan {
        fonts: fonts_out,
        substituted,
        assignment,
        lines,
        line_spans,
        vertical,
        verticals,
    })
}

/// Adds every glyph an accepted vertical line draws to the subset request —
/// the upright cells and each sideways run alike, so the subset pass takes
/// them like any other shaped gid.
fn collect_vertical_gids(
    planned: &vertical::RawVertical,
    shaped_gids: &mut BTreeMap<usize, BTreeSet<u16>>,
) {
    for item in &planned.items {
        match item {
            vertical::RawVerticalItem::Upright(glyph) => {
                shaped_gids
                    .entry(planned.upright_chain_index)
                    .or_default()
                    .insert(glyph.old_gid);
            }
            vertical::RawVerticalItem::Rotated {
                chain_index, run, ..
            } => {
                let set = shaped_gids.entry(*chain_index).or_default();
                for glyph in run
                    .clusters
                    .iter()
                    .flat_map(|cluster| cluster.glyphs.iter())
                {
                    set.insert(glyph.old_gid);
                }
            }
        }
    }
}

/// The string the page really draws for `text`: characters no face covers
/// become `?` (or vanish when even `?` is uncovered).
fn resolve_text(coverage: &BTreeMap<char, usize>, text: &str) -> String {
    let substitute_covered = coverage.contains_key(&SUBSTITUTE);
    text.chars()
        .filter(|&ch| ch >= ' ')
        .filter_map(|ch| {
            if coverage.contains_key(&ch) {
                Some(ch)
            } else if substitute_covered {
                Some(SUBSTITUTE)
            } else {
                None
            }
        })
        .collect()
}

/// Parses a chain entry as face 0 and screens out flavours the export
/// cannot embed (CFF2, or anything ttf-parser rejects).
pub(super) fn usable_face(bytes: &[u8]) -> Option<Face<'_>> {
    let face = Face::parse(bytes, 0).ok()?;
    if face.tables().cff2.is_some() {
        return None;
    }
    if face.tables().glyf.is_none() && face.tables().cff.is_none() {
        return None;
    }
    Some(face)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ttf_parser::{GlyphId, RawFace, Tag};

    /// The pre-v1.4 `plan` signature, for every test that has nothing to say
    /// about weight: every string at [`LabelWeight::Regular`].
    fn plan_regular(fonts: &PrintFonts, texts: &[&str]) -> Option<TextPlan> {
        let weighted: Vec<(LabelWeight, &str)> = texts
            .iter()
            .map(|&text| (LabelWeight::Regular, text))
            .collect();
        plan(fonts, &weighted, None)
    }

    fn noto() -> Vec<u8> {
        oxifont_bundled::NOTO_SANS_REGULAR.to_vec()
    }

    /// A real tiny font covering ONLY the characters in `keep`, so the chain
    /// walk's "first face that covers it" logic meets a genuine partial face.
    ///
    /// This goes through `subset_font`, the CODEPOINT entry point, whose
    /// output keeps a populated `cmap` — the export instead passes an empty
    /// codepoint map and embeds a program with an empty one. Same crate, two
    /// arguments, two purposes.
    fn tiny_font(keep: &str) -> Vec<u8> {
        let codepoints: std::collections::BTreeSet<char> = keep.chars().collect();
        oxifont_subset::subset_font(&noto(), &codepoints).expect("fixture subset")
    }

    /// A synthesized two-face `ttcf` collection: the bundled Noto reached
    /// through one shared table directory, twice. Both faces point at the
    /// same directory, which is legal and is what keeps the fixture cheap.
    ///
    /// This is the CI-visible form of the guarantee [`PrintFace::bytes`]
    /// makes — a chain entry may be a whole `.ttc` — which otherwise only the
    /// `#[ignore]`d live tests (real `YuGothM.ttc`/`meiryo.ttc`) exercise.
    fn two_face_ttc() -> Vec<u8> {
        /// `ttcf` tag + version + `numFonts` + one offset per face.
        const HEADER: usize = 12 + 2 * 4;
        let face = noto();
        let mut out = Vec::with_capacity(HEADER + face.len());
        out.extend_from_slice(b"ttcf");
        out.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
        out.extend_from_slice(&2_u32.to_be_bytes());
        out.extend_from_slice(&(HEADER as u32).to_be_bytes());
        out.extend_from_slice(&(HEADER as u32).to_be_bytes());
        out.extend_from_slice(&face);
        // Table records carry FILE offsets, so every one of them shifts by
        // the header a collection prepends.
        let count = usize::from(u16::from_be_bytes([out[HEADER + 4], out[HEADER + 5]]));
        for index in 0..count {
            let at = HEADER + 12 + index * 16 + 8;
            let old = u32::from_be_bytes([out[at], out[at + 1], out[at + 2], out[at + 3]]);
            out[at..at + 4].copy_from_slice(&(old + HEADER as u32).to_be_bytes());
        }
        out
    }

    #[test]
    fn a_two_face_ttc_subsets_through_face_zero() {
        let ttc = two_face_ttc();
        assert_eq!(&ttc[..4], b"ttcf", "the fixture really is a collection");
        let face = Face::parse(&ttc, 0).expect("face 0 of the collection parses");
        let gids: Vec<u16> = "AB"
            .chars()
            .map(|ch| face.glyph_index(ch).expect("Noto covers A and B").0)
            .collect();
        let chars: Vec<(char, u16)> = "AB".chars().zip(gids.iter().copied()).collect();
        let planned = subset::subset_face(
            &ttc,
            &gids,
            &chars,
            FaceRole::ScreenShared,
            LabelWeight::Regular,
        )
        .expect("a `ttcf` collection subsets through face 0");
        assert!(!planned.cff);
        let embedded = Face::parse(&planned.subset, 0).expect("the subset re-parses");
        for (&ch, &cid) in &planned.cids {
            assert!(cid != 0, "{ch:?} must not land on .notdef");
            assert!(
                embedded.glyph_hor_advance(GlyphId(cid)).is_some(),
                "{ch:?} reached the embedded program",
            );
        }
    }

    #[test]
    fn notdef_keeps_cid_zero_so_no_printed_character_can_take_it() {
        // The subsetting entry point does not insert `.notdef` — `subset_face`
        // owns that. Were it dropped, the lowest requested glyph (here the
        // leading space) would silently become glyph 0.
        let fonts = PrintFonts::new(vec![noto()]);
        let plan = plan_regular(&fonts, &[" A"]).expect("a plan");
        let font = &plan.fonts[0];
        assert!(
            font.gids.values().all(|&cid| cid != 0),
            "gids: {:?}",
            font.gids,
        );
        let face = Face::parse(&font.subset, 0).expect("re-parse");
        // Exactly one glyph beyond the mapped set, and it is glyph 0: the
        // text is composite-free Latin, so the closure adds nothing else.
        assert_eq!(
            usize::from(face.number_of_glyphs()),
            font.gids.len() + 1,
            ".notdef is the one glyph no requested gid maps to",
        );
        assert!(
            face.glyph_hor_advance(GlyphId(0)).is_some(),
            ".notdef is present in the embedded program",
        );
    }

    #[test]
    fn a_plain_ascii_title_plans_one_font_with_live_cids() {
        let fonts = PrintFonts::new(vec![noto()]);
        let plan = plan_regular(&fonts, &["Map of Tokyo (2026)"]).expect("a plan");
        assert_eq!(plan.fonts.len(), 1);
        assert_eq!(plan.substituted, 0);
        for ch in "Map of Tokyo (2026)".chars() {
            let (font, cid) = plan
                .glyph(LabelWeight::Regular, ch)
                .expect("every ASCII char is covered");
            assert_eq!(font, 0);
            assert!(cid != 0, "no printed character may be .notdef");
        }
    }

    #[test]
    fn the_subset_reparses_and_is_smaller_than_the_original() {
        let fonts = PrintFonts::new(vec![noto()]);
        let plan = plan_regular(&fonts, &["Subset me"]).expect("a plan");
        let font = &plan.fonts[0];
        assert!(!font.cff);
        let face = Face::parse(&font.subset, 0).expect("the subset must re-parse");
        // .notdef + the distinct characters of "Subset me" (space included).
        let distinct = "Subset me?"
            .chars()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(usize::from(face.number_of_glyphs()), distinct.len() + 1);
        assert!(font.subset.len() < noto().len() / 10);
    }

    #[test]
    fn cids_agree_between_plan_and_subset_widths() {
        let fonts = PrintFonts::new(vec![noto()]);
        let plan = plan_regular(&fonts, &["AVA"]).expect("a plan");
        let font = &plan.fonts[0];
        let face = Face::parse(&font.subset, 0).expect("re-parse");
        for (&ch, &cid) in &font.cids {
            // CID == subset glyph id: the advance the subset font itself
            // reports for that gid must equal the /W entry (scaled).
            let advance = face
                .glyph_hor_advance(GlyphId(cid))
                .unwrap_or_else(|| panic!("subset gid {cid} for {ch:?} has no advance"));
            let scale = 1000.0 / f32::from(face.units_per_em());
            let expected = f32::from(advance) * scale;
            let written = font.widths[&cid];
            assert!(
                (written - expected).abs() < 0.01,
                "width mismatch for {ch:?}: /W {written} vs subset hmtx {expected}"
            );
        }
    }

    #[test]
    fn a_mixed_chain_splits_runs_at_the_face_boundary() {
        // Face 0 covers only "AB"; the full Noto behind it picks up the rest.
        // The CONTRACT is unchanged from v1.1 — two runs, split exactly at
        // the face boundary — only the accessor type moved to `GlyphRun`.
        let fonts = PrintFonts::new(vec![tiny_font("AB"), noto()]);
        let plan = plan_regular(&fonts, &["ABC"]).expect("a plan");
        assert_eq!(plan.fonts.len(), 2);
        let runs = plan.runs(LabelWeight::Regular, "ABC");
        assert_eq!(runs.len(), 2, "AB then C: exactly two runs");
        assert_eq!(runs[0].font, 0);
        assert_eq!(runs[0].glyphs.len(), 2, "two glyphs from the tiny face");
        assert_eq!(runs[1].font, 1);
        assert_eq!(runs[1].glyphs.len(), 1, "one glyph from the full face");
    }

    #[test]
    fn a_kerned_pair_emits_a_tj_adjustment_and_a_tighter_width() {
        // Probe-verified: Noto's GPOS kerns A,V by -40/1000 em.
        let fonts = PrintFonts::new(vec![noto()]);
        let shaped = plan_regular(&fonts, &["AV"]).expect("a plan");
        let runs = shaped.runs(LabelWeight::Regular, "AV");
        assert_eq!(runs.len(), 1);
        assert_eq!(
            runs[0].glyphs[0].adjust_1000, 40.0,
            "the kern delta rides the TJ number (positive tightens)",
        );
        assert_eq!(runs[0].glyphs[1].adjust_1000, 0.0);
        // Kerned width: 599 + 600 = 1199/1000 em, strictly under the sum of
        // the /W entries (639 + 600).
        let kerned = shaped.width_pt(LabelWeight::Regular, "AV", 1000.0);
        assert!((kerned - 1199.0).abs() < 0.5, "got {kerned}");
        let unkerned = shaped.width_pt(LabelWeight::Regular, "A", 1000.0)
            + shaped.width_pt(LabelWeight::Regular, "V", 1000.0);
        assert!(
            kerned < unkerned,
            "{kerned} must be tighter than {unkerned}"
        );
    }

    #[test]
    fn a_ligature_survives_a_document_shaped_text_set() {
        // The exact text set `pdf_document` plans for a page titled "fi":
        // title, empty attribution, a scale label — the ligature entry must
        // survive the company.
        let fonts = PrintFonts::new(vec![noto()]);
        let shaped = plan_regular(&fonts, &["fi", "", "500 km"]).expect("a plan");
        let runs = shaped.runs(LabelWeight::Regular, "fi");
        assert_eq!(runs.len(), 1, "runs: {runs:?}");
        assert_eq!(runs[0].glyphs.len(), 1, "glyphs: {:?}", runs[0].glyphs);
        let cid = runs[0].glyphs[0].cid;
        assert_eq!(
            shaped.fonts[0].to_unicode.get(&cid).map(String::as_str),
            Some("fi"),
            "to_unicode: {:?}",
            shaped.fonts[0].to_unicode,
        );
    }

    #[test]
    fn a_ligature_is_one_cid_that_extracts_as_both_characters() {
        let fonts = PrintFonts::new(vec![noto()]);
        let shaped = plan_regular(&fonts, &["fi"]).expect("a plan");
        let runs = shaped.runs(LabelWeight::Regular, "fi");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].glyphs.len(), 1, "liga fires: one glyph for fi");
        let cid = runs[0].glyphs[0].cid;
        assert_eq!(
            shaped.fonts[0].to_unicode.get(&cid).map(String::as_str),
            Some("fi"),
            "the ligature CID maps back to BOTH characters",
        );
        assert!(
            shaped.fonts[0].widths.contains_key(&cid),
            "the ligature CID has a /W entry",
        );
        // The ligature glyph really made it into the embedded program.
        let face = Face::parse(&shaped.fonts[0].subset, 0).expect("re-parse");
        assert!(
            face.glyph_hor_advance(GlyphId(cid)).is_some(),
            "the shaped gid reached the subset program",
        );
    }

    #[test]
    fn the_measured_width_is_exactly_the_pen_movement() {
        let fonts = PrintFonts::new(vec![noto()]);
        let shaped = plan_regular(&fonts, &["Print fixture AV fi"]).expect("a plan");
        for text in ["Print fixture AV fi", "AV", "fi"] {
            let runs = shaped.runs(LabelWeight::Regular, text);
            let pen: f32 = runs
                .iter()
                .flat_map(|run| run.glyphs.iter())
                .map(|glyph| glyph.width_1000 - glyph.adjust_1000)
                .sum();
            let advance: f32 = runs.iter().map(|run| run.advance_1000).sum();
            assert_eq!(advance, pen, "advance_1000 is DEFINED as the pen sum");
            assert_eq!(shaped.width_pt(LabelWeight::Regular, text, 1000.0), advance);
        }
    }

    #[test]
    fn an_rtl_string_keeps_the_v11_per_character_path() {
        let fonts = PrintFonts::new(vec![noto()]);
        let shaped = plan_regular(&fonts, &["مرحبا", "Latin"]).expect("a plan");
        // The RTL string was refused from shaping: its runs come from the
        // synthetic path (every adjust zero), not from a stored line.
        let runs = shaped.runs(LabelWeight::Regular, "مرحبا");
        assert!(
            runs.iter()
                .flat_map(|run| run.glyphs.iter())
                .all(|glyph| glyph.adjust_1000 == 0.0),
            "refused RTL is the v1.1 unshaped output",
        );
        // The Latin string still shaped.
        assert!(!shaped.runs(LabelWeight::Regular, "Latin").is_empty());
    }

    #[test]
    fn an_uncovered_character_substitutes_the_question_mark_and_is_counted() {
        // The bundled Noto is Latin-only: CJK degrades to '?'.
        let fonts = PrintFonts::new(vec![noto()]);
        let plan = plan_regular(&fonts, &["東京 map"]).expect("a plan");
        assert_eq!(plan.substituted, 2);
        let question = plan
            .glyph(LabelWeight::Regular, '?')
            .expect("the substitute itself");
        assert_eq!(plan.glyph(LabelWeight::Regular, '東'), Some(question));
        assert_eq!(plan.glyph(LabelWeight::Regular, '京'), Some(question));
        assert!(plan.glyph(LabelWeight::Regular, 'm').is_some());
    }

    #[test]
    fn an_empty_or_junk_chain_yields_no_plan() {
        assert!(plan_regular(&PrintFonts::none(), &["title"]).is_none());
        let junk = PrintFonts::new(vec![vec![0_u8; 64]]);
        assert!(plan_regular(&junk, &["title"]).is_none());
    }

    #[test]
    fn a_junk_entry_is_skipped_and_the_next_face_serves() {
        let fonts = PrintFonts::new(vec![vec![0xFF_u8; 128], noto()]);
        let plan = plan_regular(&fonts, &["ok"]).expect("the real face must be found");
        assert_eq!(plan.fonts.len(), 1);
        assert_eq!(plan.substituted, 0);
    }

    #[test]
    fn planning_is_deterministic_and_the_base_font_is_tagged() {
        let fonts = PrintFonts::new(vec![noto()]);
        let first = plan_regular(&fonts, &["Same input"]).expect("a plan");
        let second = plan_regular(&fonts, &["Same input"]).expect("a plan");
        assert_eq!(first.fonts[0].subset, second.fonts[0].subset);
        assert_eq!(first.fonts[0].base_font, second.fonts[0].base_font);
        let (tag, name) = first.fonts[0]
            .base_font
            .split_once('+')
            .expect("TAG+PSName shape");
        assert_eq!(tag.len(), 6);
        assert!(tag.chars().all(|ch| ch.is_ascii_uppercase()));
        assert!(name.starts_with("NotoSans"), "got {name}");
        // A different glyph set must change the tag (it names the subset,
        // not the face).
        let other = plan_regular(&fonts, &["Different text"]).expect("a plan");
        assert_ne!(first.fonts[0].base_font, other.fonts[0].base_font);
    }

    #[test]
    fn widths_measure_text_in_the_same_numbers_the_viewer_uses() {
        let fonts = PrintFonts::new(vec![noto()]);
        let plan = plan_regular(&fonts, &["100 m"]).expect("a plan");
        let width = plan.width_pt(LabelWeight::Regular, "100 m", 8.0);
        // Sanity envelope: five 8-pt characters land between 10 and 40 pt.
        assert!(width > 10.0 && width < 40.0, "got {width}");
        assert!((plan.width_pt(LabelWeight::Regular, "", 8.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn descriptor_metrics_are_in_thousandths_and_sane() {
        let fonts = PrintFonts::new(vec![noto()]);
        let plan = plan_regular(&fonts, &["x"]).expect("a plan");
        let metrics = &plan.fonts[0].metrics;
        assert!(metrics.ascent > 500.0 && metrics.ascent < 1500.0);
        assert!(metrics.descent < 0.0 && metrics.descent > -1000.0);
        assert!(metrics.cap_height > 300.0 && metrics.cap_height < 1200.0);
        assert!(metrics.bbox[0] < metrics.bbox[2]);
        assert!(metrics.bbox[1] < metrics.bbox[3]);
        assert!(!metrics.italic);
        // The bundled Noto is a static Regular: /FontWeight 400, StemV 80.
        assert_eq!(metrics.weight, 400);
        assert!((metrics.stem_v - 80.0).abs() < f32::EPSILON);
    }

    #[test]
    fn the_font_weight_grid_rounds_and_clamps() {
        assert_eq!(subset::font_weight_grid(350.0), 400);
        assert_eq!(subset::font_weight_grid(349.0), 300);
        assert_eq!(subset::font_weight_grid(1200.0), 900);
        assert_eq!(subset::font_weight_grid(0.0), 100);
        assert_eq!(subset::font_weight_grid(400.0), 400);
    }

    #[test]
    fn the_subset_carries_no_variation_tables() {
        // Pins the "embedded at ONE static instance" claim: `drop_variations`
        // lets no variation table survive into the program, so the /W widths
        // and the outlines describe the same instance unambiguously. Vacuous
        // on the static bundled Noto — `live_windows_the_instanced_subset_
        // carries_no_variation_tables` is the half that has something to
        // delete.
        let fonts = PrintFonts::new(vec![noto()]);
        let plan = plan_regular(&fonts, &["x"]).expect("a plan");
        let raw = RawFace::parse(&plan.fonts[0].subset, 0).expect("re-parse");
        for table in [b"fvar", b"gvar", b"HVAR", b"avar"] {
            assert!(
                raw.table(Tag::from_bytes(table)).is_none(),
                "{} must not survive the subset",
                String::from_utf8_lossy(table),
            );
        }
        assert!(raw.table(Tag::from_bytes(b"STAT")).is_none());
    }

    /// The path a `PrintOnly` variable face takes: bold-slot, so a `Bold`
    /// request reaches it, and instanced because nothing on screen shares it.
    /// Returns the planned font that really drew `ch`, plus that CID.
    fn planned_bold_face(vf: Vec<u8>, text: &str, ch: char) -> Option<(PlannedFont, u16)> {
        let fonts = bold_chain(noto(), vf, FaceRole::PrintOnly);
        let mut planned = plan(&fonts, &[(LabelWeight::Bold, text)], None)?;
        let (index, cid) = planned.glyph(LabelWeight::Bold, ch)?;
        Some((planned.fonts.swap_remove(index), cid))
    }

    #[test]
    #[ignore = "reads C:/Windows/Fonts/NotoSansJP-VF.ttf; the wght-700 instancing oracle"]
    fn live_windows_a_variable_face_embeds_at_the_bold_instance() {
        // The first test in this repo that MEASURES instancing. The oracle is
        // `ttf_parser::set_variation` at the same location, which the design
        // probes matched bit-for-bit on 95/95 samples across 5 faces, so the
        // advances are EXACT — a tolerance here would hide a rounding-policy
        // regression, which is the defect class this path exists to prevent.
        let Ok(vf) = std::fs::read("C:/Windows/Fonts/NotoSansJP-VF.ttf") else {
            return;
        };
        // NotoSansJP-VF defaults to wght 100; every advance below is heavier.
        for (ch, expected) in [('M', 853_u16), ('A', 641), ('y', 574)] {
            let Some((font, cid)) = planned_bold_face(vf.clone(), "MAy", ch) else {
                panic!("the variable face must plan {ch:?} at Bold");
            };
            assert_eq!(font.metrics.weight, 700, "the /FontWeight follows fvar");
            let embedded = Face::parse(&font.subset, 0).expect("the subset re-parses");
            assert_eq!(
                embedded.units_per_em(),
                1000,
                "the oracle is in the face's own design units",
            );
            assert_eq!(
                embedded.glyph_hor_advance(GlyphId(cid)),
                Some(expected),
                "{ch:?} must be the wght 700 advance, not the wght 100 default",
            );
        }
    }

    #[test]
    #[ignore = "reads C:/Windows/Fonts/NotoSansJP-VF.ttf; the real-VF half of the static claim"]
    fn live_windows_the_instanced_subset_carries_no_variation_tables() {
        // `the_subset_carries_no_variation_tables` runs on a face that has
        // none to begin with. This one starts from 9.6 MB of fvar/gvar/avar/
        // HVAR/STAT/DSIG and asserts the embedded program kept none of it —
        // otherwise `/FontWeight` and the outlines could disagree.
        let Ok(vf) = std::fs::read("C:/Windows/Fonts/NotoSansJP-VF.ttf") else {
            return;
        };
        let Some((font, _cid)) = planned_bold_face(vf, "MAy", 'M') else {
            panic!("the variable face must plan at Bold");
        };
        let raw = RawFace::parse(&font.subset, 0).expect("re-parse");
        for table in [
            b"fvar", b"gvar", b"avar", b"HVAR", b"VVAR", b"MVAR", b"STAT", b"cvar", b"DSIG",
        ] {
            assert!(
                raw.table(Tag::from_bytes(table)).is_none(),
                "{} must not survive instancing",
                String::from_utf8_lossy(table),
            );
        }
        let face = Face::parse(&font.subset, 0).expect("re-parse");
        assert!(!face.is_variable(), "the embedded program is static");
    }

    // --- print/text v1.4 item 4 (D-W5 / D-W6): weight ---

    /// A synthetic VARIABLE face carrying an `fvar` with the named instances
    /// `instances` describes. The outlines stay the bundled Noto's — the
    /// ladder only ever reads `fvar` and the weight class, and building a
    /// real multi-master face in a unit test would prove nothing the
    /// instancing tests in `print::instance` do not already prove.
    fn variable_face(instances: &[(u16, f32)], default_weight: f32) -> Vec<u8> {
        let mut out = noto();
        let mut fvar = Vec::new();
        fvar.extend_from_slice(&1_u16.to_be_bytes());
        fvar.extend_from_slice(&0_u16.to_be_bytes());
        fvar.extend_from_slice(&16_u16.to_be_bytes());
        fvar.extend_from_slice(&2_u16.to_be_bytes());
        fvar.extend_from_slice(&1_u16.to_be_bytes()); // one axis
        fvar.extend_from_slice(&20_u16.to_be_bytes());
        fvar.extend_from_slice(&(instances.len() as u16).to_be_bytes());
        fvar.extend_from_slice(&8_u16.to_be_bytes()); // 4 + 1*4
        let fixed = |value: f32| ((value * 65536.0) as i32).to_be_bytes();
        fvar.extend_from_slice(b"wght");
        fvar.extend_from_slice(&fixed(100.0));
        fvar.extend_from_slice(&fixed(default_weight));
        fvar.extend_from_slice(&fixed(900.0));
        fvar.extend_from_slice(&0_u16.to_be_bytes());
        fvar.extend_from_slice(&256_u16.to_be_bytes());
        for &(name_id, weight) in instances {
            fvar.extend_from_slice(&name_id.to_be_bytes());
            fvar.extend_from_slice(&0_u16.to_be_bytes());
            fvar.extend_from_slice(&fixed(weight));
        }
        append_table(&mut out, b"fvar", &fvar);
        out
    }

    /// Appends `table` to an sfnt, rewriting the table directory. Enough for
    /// `ttf_parser` and this module's `fvar` reads; no checksum fix-ups
    /// (nothing in the print path verifies them).
    ///
    /// The record is inserted in **tag order**, which is not cosmetic:
    /// `ttf_parser::RawFace::table` binary-searches the directory, so an
    /// out-of-order record is invisible to `instance::raw_fvar` while
    /// `Face::parse` (a linear walk) still sees it — a fixture that would
    /// silently test the wrong ladder arm.
    fn append_table(font: &mut Vec<u8>, tag: &[u8; 4], table: &[u8]) {
        let count = usize::from(u16::from_be_bytes([font[4], font[5]]));
        let offset = font.len();
        let mut record = Vec::new();
        record.extend_from_slice(tag);
        record.extend_from_slice(&0_u32.to_be_bytes());
        record.extend_from_slice(&((offset + 16) as u32).to_be_bytes());
        record.extend_from_slice(&(table.len() as u32).to_be_bytes());
        // Every existing table offset shifts by the 16 bytes the new record
        // takes; rewrite them before inserting.
        let mut at_record = 12 + count * 16;
        for index in 0..count {
            let at = 12 + index * 16;
            if at_record == 12 + count * 16 && font[at..at + 4] > tag[..] {
                at_record = at;
            }
            let old =
                u32::from_be_bytes([font[at + 8], font[at + 9], font[at + 10], font[at + 11]]);
            font[at + 8..at + 12].copy_from_slice(&(old + 16).to_be_bytes());
        }
        font.splice(at_record..at_record, record);
        font[4..6].copy_from_slice(&((count + 1) as u16).to_be_bytes());
        font.extend_from_slice(table);
    }

    /// A face whose OS/2 `usWeightClass` says `weight` — a "real bold face"
    /// for the L0 arm, built by patching the bundled Noto's OS/2.
    fn static_face_of_weight(weight: u16) -> Vec<u8> {
        let mut out = noto();
        let at = {
            let raw = RawFace::parse(&out, 0).expect("the bundled face parses");
            let os2 = raw.table(Tag::from_bytes(b"OS/2")).expect("Noto has OS/2");
            out.windows(os2.len())
                .position(|window| window == os2)
                .expect("the OS/2 bytes are findable")
        };
        out[at + 4..at + 6].copy_from_slice(&weight.to_be_bytes());
        out
    }

    fn bold_chain(regular: Vec<u8>, bold: Vec<u8>, role: FaceRole) -> PrintFonts {
        PrintFonts::with_roles(vec![
            PrintFace::regular(regular, FaceRole::ScreenShared),
            PrintFace::bold(bold, role),
        ])
    }

    fn ladder_for(fonts: &PrintFonts) -> WeightLadder {
        let faces: Vec<Option<Face<'_>>> =
            fonts.chain.iter().map(|bytes| usable_face(bytes)).collect();
        resolve_weight(fonts, &faces)
    }

    #[test]
    fn ladder_l0_a_real_bold_face_draws_bold_without_instancing() {
        let fonts = bold_chain(noto(), static_face_of_weight(700), FaceRole::ScreenShared);
        let ladder = ladder_for(&fonts);
        assert_eq!(
            ladder,
            WeightLadder::RealBoldFace {
                chain_index: 1,
                weight: 700.0,
            },
        );
        assert!(ladder.draws_bold());
    }

    #[test]
    fn ladder_l1_a_print_only_variable_face_is_instanced_at_the_bold_target() {
        // The arm asserts PrintOnly: a ScreenShared twin of the SAME face
        // must NOT be instanced, or the page and the map disagree.
        let variable = variable_face(&[(17, 400.0), (18, 700.0)], 400.0);
        let fonts = bold_chain(noto(), variable.clone(), FaceRole::PrintOnly);
        assert_eq!(
            ladder_for(&fonts),
            WeightLadder::Instanced {
                chain_index: 1,
                weight: 700.0,
            },
        );
        let shared = bold_chain(noto(), variable, FaceRole::ScreenShared);
        assert_eq!(
            ladder_for(&shared),
            WeightLadder::AxisCannotReachBold { chain_index: 1 },
            "a ScreenShared face is NEVER instanced — screen parity",
        );
    }

    #[test]
    fn ladder_l2_an_axis_that_cannot_reach_bold_prints_regular() {
        // The nearest instance IS the fvar default: `choose_instance`
        // answers None (the byte-identity gate), so there is nothing bolder
        // to embed.
        let at_default = variable_face(&[(17, 400.0)], 400.0);
        let fonts = bold_chain(noto(), at_default, FaceRole::PrintOnly);
        let ladder = ladder_for(&fonts);
        assert_eq!(ladder, WeightLadder::AxisCannotReachBold { chain_index: 1 });
        assert!(!ladder.draws_bold());
        // And an axis whose heaviest instance is still light.
        let too_light = variable_face(&[(17, 300.0), (18, 500.0)], 300.0);
        let fonts = bold_chain(noto(), too_light, FaceRole::PrintOnly);
        assert_eq!(
            ladder_for(&fonts),
            WeightLadder::AxisCannotReachBold { chain_index: 1 },
        );
    }

    #[test]
    fn ladder_l3_a_bold_slot_face_with_no_weight_axis_prints_regular() {
        // The bundled Noto is a static Regular: nothing to instance, and no
        // bold weight class to believe.
        let fonts = bold_chain(noto(), noto(), FaceRole::PrintOnly);
        let ladder = ladder_for(&fonts);
        assert_eq!(ladder, WeightLadder::NoWeightAxis { chain_index: 1 });
        assert!(!ladder.draws_bold());
    }

    #[test]
    fn ladder_l4_a_chain_with_no_bold_slot_prints_regular() {
        let fonts = PrintFonts::new(vec![noto()]);
        let ladder = ladder_for(&fonts);
        assert_eq!(ladder, WeightLadder::NoBoldFace);
        assert!(!ladder.draws_bold());
    }

    #[test]
    fn a_no_bold_project_plans_byte_identically_to_a_chain_without_bold_faces() {
        // THE gate D-W5 names: with nothing Bold the second pass never runs,
        // so a chain carrying bold faces produces exactly the plan — and
        // therefore exactly the PDF bytes — a chain without them does.
        let texts = ["Map of Tokyo (2026)", "500 m", ""];
        let plain = plan_regular(&PrintFonts::new(vec![noto()]), &texts).expect("a plan");
        let with_bold = plan_regular(
            &bold_chain(noto(), static_face_of_weight(700), FaceRole::ScreenShared),
            &texts,
        )
        .expect("a plan");
        assert_eq!(
            with_bold.fonts.len(),
            1,
            "the bold face is never subsetted when nothing asks for it",
        );
        assert_eq!(plain.fonts[0].subset, with_bold.fonts[0].subset);
        assert_eq!(plain.fonts[0].base_font, with_bold.fonts[0].base_font);
        for text in texts {
            assert_eq!(
                plain.runs(LabelWeight::Regular, text).to_vec(),
                with_bold.runs(LabelWeight::Regular, text).to_vec(),
            );
        }
    }

    #[test]
    fn the_same_text_at_two_weights_gets_two_plans_and_two_advances() {
        // The hostile pin: one page, one string, both weights. The tiny bold
        // face covers only "AB", so the two weights genuinely differ in
        // which face draws what.
        let fonts = bold_chain(noto(), tiny_font("AB"), FaceRole::ScreenShared);
        let planned = plan(
            &fonts,
            &[(LabelWeight::Regular, "AB"), (LabelWeight::Bold, "AB")],
            None,
        )
        .expect("a plan");
        assert_eq!(planned.fonts.len(), 2, "both faces are embedded");
        let regular = planned.runs(LabelWeight::Regular, "AB").to_vec();
        let bold = planned.runs(LabelWeight::Bold, "AB").to_vec();
        assert_eq!(regular.len(), 1);
        assert_eq!(bold.len(), 1);
        assert_ne!(
            regular[0].font, bold[0].font,
            "the two weights draw through different subsets",
        );
        // Distinct plans, and each measures through its OWN face.
        assert!(planned.width_pt(LabelWeight::Regular, "AB", 1000.0) > 0.0);
        assert!(planned.width_pt(LabelWeight::Bold, "AB", 1000.0) > 0.0);
    }

    #[test]
    fn a_bold_face_lacking_a_block_falls_through_to_the_regular_chain() {
        // Never-shrink: the bold face covers only "AB", so 'C' comes from
        // the regular face at Regular weight — a mixed-weight line, and NOT
        // a '?'.
        let fonts = bold_chain(noto(), tiny_font("AB"), FaceRole::ScreenShared);
        let planned = plan(&fonts, &[(LabelWeight::Bold, "ABC")], None).expect("a plan");
        assert_eq!(planned.substituted, 0, "nothing degrades to '?'");
        let runs = planned.runs(LabelWeight::Bold, "ABC");
        assert_eq!(runs.len(), 2, "AB from the bold face, C from the regular");
        assert_ne!(runs[0].font, runs[1].font);
    }

    #[test]
    fn a_regular_request_never_reaches_a_bold_slot_face() {
        // The bold face covers everything; at Regular weight only the tiny
        // regular face is walked, so a character it lacks substitutes rather
        // than borrowing the bold face.
        let fonts = bold_chain(tiny_font("A"), noto(), FaceRole::ScreenShared);
        let planned = plan(&fonts, &[(LabelWeight::Regular, "AB")], None).expect("a plan");
        assert_eq!(
            planned.fonts.len(),
            1,
            "only the regular chain is walked: {} fonts",
            planned.fonts.len(),
        );
        assert_eq!(planned.substituted, 1);
    }

    #[test]
    fn bold_arabic_shapes_against_the_bold_coverage_map() {
        // The bidi path takes the WEIGHT's own coverage map, so a bold RTL
        // string is segmented against the faces that will really draw it.
        // The bundled Noto has no Arabic, so the honest outcome here is the
        // v1.1 unshaped path at BOTH weights — what matters is that the bold
        // request went through its own map rather than the regular one.
        let fonts = bold_chain(noto(), noto(), FaceRole::ScreenShared);
        let arabic = "\u{645}\u{631}\u{62D}\u{628}\u{627}";
        let planned = plan(
            &fonts,
            &[(LabelWeight::Regular, arabic), (LabelWeight::Bold, arabic)],
            None,
        )
        .expect("a plan");
        for weight in [LabelWeight::Regular, LabelWeight::Bold] {
            assert!(
                planned
                    .runs(weight, arabic)
                    .iter()
                    .flat_map(|run| run.glyphs.iter())
                    .all(|glyph| glyph.adjust_1000 == 0.0),
                "{weight:?} keeps the v1.1 unshaped output",
            );
        }
    }
}
