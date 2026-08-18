// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The text emitter: [`show_line`], its marked-content-aware twin and the
//! WinAnsi degraded mapping — split out of `print/mod.rs` (print v1.3 P0)
//! and grown with the tier emitter (P5/P6): `/Span <</ActualText …>>`
//! marked content for bidi lines and multi-glyph clusters, `Ts` rises and
//! paired-`TJ` x-shifts for mark placement.
//!
//! # Invariants the arithmetic assumes
//!
//! `Tc`/`Tw` are never set in the print path and `Tz` stays 100 — the TJ
//! numbers and the rise operand would both scale wrong otherwise. `Ts` is
//! graphics state `BT` does NOT reset, so every nonzero rise is closed with
//! `0 Ts` before the line ends — unconditionally, page-wide (a leaked rise
//! would poison every later text object; the title/footer paths are not
//! wrapped in `q`/`Q`). A `BDC`/`EMC` never opens or closes inside a `TJ`
//! array: the pending byte run is flushed at every span or rise boundary.

use oxigis_core::LabelWeight;
use pdf_writer::{Content, Name, Str, TextStr};

use super::font::{GlyphRun, TextPlan};
use super::vertical::{VerticalItem, VerticalLine};

/// Whether a line's show operators are page CONTENT or a repeated artifact.
///
/// The halo (stroke) pass repeats the fill pass's operators; it is already
/// wrapped in `/Artifact BMC … EMC` (the v1.2 double-extraction fix), and
/// it must NOT also carry `/ActualText` spans — an artifact is by
/// definition not content, so a span there is at best ignored and at worst
/// read twice by a sloppy extractor.
/// One line of text and the weight it draws at.
///
/// Bundled rather than passed as two arguments so the emitter's signature
/// stays under clippy's argument budget — and because the two are never
/// meaningful apart: measuring a string at one weight and drawing it at
/// another is exactly the bug the weight-keyed plan exists to prevent.
#[derive(Clone, Copy)]
pub(super) struct WeightedText<'a> {
    /// The logical string.
    pub text: &'a str,
    /// Which face chain draws it.
    pub weight: LabelWeight,
}

impl<'a> WeightedText<'a> {
    /// Page furniture — a title, the attribution, the scale-bar label — is
    /// always [`LabelWeight::Regular`]; only a symbol style can ask for Bold.
    pub(super) fn regular(text: &'a str) -> Self {
        Self {
            text,
            weight: LabelWeight::Regular,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TextMark {
    /// Real page text: spans are emitted.
    Content,
    /// A repeated decoration pass: geometry only, no marked content.
    Artifact,
}

/// Reduces `text` to WinAnsi-safe bytes: everything `/WinAnsiEncoding` can
/// name passes through, everything else — CJK most notably — becomes `?`.
///
/// Base-14 Helvetica with `/WinAnsiEncoding` is the one text setup that needs
/// no embedded font program; the honest cost is this mapping, and it is paid
/// visibly rather than by emitting bytes the viewer would render as garbage.
/// A full Unicode title needs an embedded Type0 font — the documented v1.1
/// follow-up.
///
/// The mapping is the encoding's WHOLE repertoire since v1.6, not just ASCII:
/// WinAnsi *is* CP1252, so a Latin-1 name (`München`, `Québec`) and the
/// typographic punctuation of the 0x80–0x9F block print as themselves rather
/// than as `?` for no reason the encoding imposes.
#[must_use]
pub fn win_ansi(text: &str) -> Vec<u8> {
    text.chars().map(win_ansi_byte).collect()
}

/// One character's `/WinAnsiEncoding` code, or `b'?'` when the encoding has
/// no slot for it.
fn win_ansi_byte(character: char) -> u8 {
    let code = character as u32;
    // Printable ASCII and the Latin-1 upper half are their own byte values;
    // 0x80..=0x9F is the ONLY range where CP1252 differs from Latin-1, and
    // those glyphs are reached by codepoint below.
    if (0x20..=0x7E).contains(&code) || (0xA0..=0xFF).contains(&code) {
        return code as u8;
    }
    match character {
        '\u{20AC}' => 0x80, // €
        '\u{201A}' => 0x82, // ‚
        '\u{0192}' => 0x83, // ƒ
        '\u{201E}' => 0x84, // „
        '\u{2026}' => 0x85, // …
        '\u{2020}' => 0x86, // †
        '\u{2021}' => 0x87, // ‡
        '\u{02C6}' => 0x88, // ˆ
        '\u{2030}' => 0x89, // ‰
        '\u{0160}' => 0x8A, // Š
        '\u{2039}' => 0x8B, // ‹
        '\u{0152}' => 0x8C, // Œ
        '\u{017D}' => 0x8E, // Ž
        '\u{2018}' => 0x91, // ‘
        '\u{2019}' => 0x92, // ’
        '\u{201C}' => 0x93, // “
        '\u{201D}' => 0x94, // ”
        '\u{2022}' => 0x95, // •
        '\u{2013}' => 0x96, // –
        '\u{2014}' => 0x97, // —
        '\u{02DC}' => 0x98, // ˜
        '\u{2122}' => 0x99, // ™
        '\u{0161}' => 0x9A, // š
        '\u{203A}' => 0x9B, // ›
        '\u{0153}' => 0x9C, // œ
        '\u{017E}' => 0x9E, // ž
        '\u{0178}' => 0x9F, // Ÿ
        _ => b'?',
    }
}

/// Shows one line of CONTENT text at `(x, y)` — the historical signature,
/// kept for every call site that is not the halo pass.
pub(super) fn show_line(
    content: &mut Content,
    plan: Option<&TextPlan>,
    x: f32,
    y: f32,
    size: f32,
    text: &str,
) {
    show_line_marked(
        content,
        plan,
        x,
        y,
        size,
        WeightedText::regular(text),
        TextMark::Content,
    );
}

/// Shows one line of text at `(x, y)`: embedded-font CID runs when a plan
/// is live, the Base-14 Helvetica + WinAnsi degraded path otherwise.
///
/// A bidi line (the plan says so via [`TextPlan::actual_text`]) is wrapped
/// WHOLE in one `/Span <</ActualText (logical)>>` — per-run spans would
/// concatenate in visual order and extract a mixed line wrong. On an
/// LTR-only line, each multi-glyph or offset cluster carries its own span.
pub(super) fn show_line_marked(
    content: &mut Content,
    plan: Option<&TextPlan>,
    x: f32,
    y: f32,
    size: f32,
    line: WeightedText<'_>,
    mark: TextMark,
) {
    let WeightedText { text, weight } = line;
    content.begin_text();
    content.next_line(x, y);
    match plan {
        Some(plan) => {
            let line_span = match mark {
                TextMark::Content => plan.actual_text(weight, text),
                TextMark::Artifact => None,
            };
            if let Some(actual) = line_span {
                let mut marked = content.begin_marked_content_with_properties(Name(b"Span"));
                marked.properties().actual_text(TextStr(actual));
            }
            // Cluster spans only when the line has no line-level span and
            // the pass is content.
            let cluster_spans = mark == TextMark::Content && line_span.is_none();
            let mut rise = 0.0_f32;
            for run in plan.runs(weight, text).iter() {
                let name = format!("F{}", run.font + 1);
                content.set_font(Name(name.as_bytes()), size);
                emit_run(content, run, size, cluster_spans, &mut rise);
            }
            if rise != 0.0 {
                // Unconditional: `Ts` survives `BT`/`ET`, and nothing
                // downstream would rescue a leaked rise.
                content.set_rise(0.0);
            }
            if line_span.is_some() {
                content.end_marked_content();
            }
        }
        None => {
            content.set_font(Name(b"F0"), size);
            content.show(Str(&win_ansi(text)));
        }
    }
    content.end_text();
}

/// Emits one run's glyphs: the all-plain fast path is a single `Tj`,
/// byte-identical to v1.1; everything else segments on span and rise
/// boundaries so no `BDC`/`EMC` or `Ts` ever lands inside a `TJ` array.
/// `rise` is the LINE's current `Ts` state — shared across runs so the
/// caller can close it exactly once.
fn emit_run(content: &mut Content, run: &GlyphRun, size: f32, cluster_spans: bool, rise: &mut f32) {
    let spans: &[super::font::ActualSpan] = if cluster_spans { &run.spans } else { &[] };
    let plain = spans.is_empty()
        && *rise == 0.0
        && run.glyphs.iter().all(|glyph| {
            glyph.adjust_1000 == 0.0 && glyph.x_shift_1000 == 0.0 && glyph.rise_1000 == 0.0
        });
    if plain {
        // Unkerned, unshifted run (all CJK, plain short Latin): a single
        // `Tj`, byte-identical to the v1.1 output.
        let mut bytes = Vec::with_capacity(run.glyphs.len() * 2);
        for glyph in &run.glyphs {
            bytes.extend_from_slice(&glyph.cid.to_be_bytes());
        }
        content.show(Str(&bytes));
        return;
    }
    let mut span_index = 0_usize;
    let mut open_end: Option<usize> = None;
    let mut index = 0_usize;
    while index < run.glyphs.len() {
        // Skip malformed spans behind the cursor (cannot arise from the
        // plan's construction; total-function hygiene).
        while open_end.is_none() && spans.get(span_index).is_some_and(|span| span.start < index) {
            span_index += 1;
        }
        // Open a span that starts here (always outside any pending TJ).
        if open_end.is_none()
            && let Some(span) = spans.get(span_index).filter(|span| span.start == index)
        {
            let mut marked = content.begin_marked_content_with_properties(Name(b"Span"));
            marked.properties().actual_text(TextStr(&span.text));
            open_end = Some(span.end.min(run.glyphs.len()));
            span_index += 1;
        }
        // The flushable segment: constant rise, never crossing a span edge.
        let limit = match open_end {
            Some(close) => close.max(index + 1),
            None => spans
                .get(span_index)
                .map_or(run.glyphs.len(), |span| span.start.max(index + 1)),
        };
        let glyph_rise = run.glyphs[index].rise_1000;
        let mut end = index + 1;
        while end < limit && run.glyphs[end].rise_1000 == glyph_rise {
            end += 1;
        }
        if glyph_rise != *rise {
            content.set_rise(glyph_rise / 1000.0 * size);
            *rise = glyph_rise;
        }
        emit_segment(content, &run.glyphs[index..end]);
        index = end;
        // Close a span that ends here.
        if open_end == Some(index) {
            content.end_marked_content();
            open_end = None;
        }
    }
    if open_end.is_some() {
        content.end_marked_content();
    }
}

/// Emits one same-rise glyph range as a `Tj` (nothing to adjust) or one
/// `TJ` array with the kern deltas and the PAIRED x-shift numbers: `−x`
/// before the glyph, `x + adjust` after, so the net pen movement is exactly
/// `Σ (width − adjust)` and measurement stays equal to rendering.
fn emit_segment(content: &mut Content, glyphs: &[super::font::RunGlyph]) {
    if glyphs
        .iter()
        .all(|glyph| glyph.adjust_1000 == 0.0 && glyph.x_shift_1000 == 0.0)
    {
        let mut bytes = Vec::with_capacity(glyphs.len() * 2);
        for glyph in glyphs {
            bytes.extend_from_slice(&glyph.cid.to_be_bytes());
        }
        content.show(Str(&bytes));
        return;
    }
    let mut positioned = content.show_positioned();
    let mut items = positioned.items();
    let mut pending: Vec<u8> = Vec::new();
    for glyph in glyphs {
        if glyph.x_shift_1000 != 0.0 {
            if !pending.is_empty() {
                items.show(Str(&pending));
                pending.clear();
            }
            // TJ numbers are SUBTRACTED from the pen: `−x` places the
            // glyph at `pen + x`.
            items.adjust(-glyph.x_shift_1000);
        }
        pending.extend_from_slice(&glyph.cid.to_be_bytes());
        let post = glyph.x_shift_1000 + glyph.adjust_1000;
        if post != 0.0 {
            items.show(Str(&pending));
            pending.clear();
            items.adjust(post);
        }
    }
    if !pending.is_empty() {
        items.show(Str(&pending));
    }
}

/// Shows one VERTICAL line: the items of `line` stacked downward from
/// `(x, y)`, inside ONE `BT`..`ET` (print/text v1.4, D-V3; mixed lines since
/// v1.5, D-B4).
///
/// **An all-upright line takes the v1.4 path byte for byte.** Each glyph is a
/// relative `Td` step — down by its predecessor's pitch, sideways by the
/// change in centring shift — followed by one `Tj`. There is no `TJ` array
/// and no `Ts`: a vertical cell's position is entirely the text-line matrix's
/// business, and `Tz`/`Tc`/`Tw` stay at their defaults as everywhere else in
/// this module. The branch is chosen by [`VerticalLine::is_all_upright`] and
/// by nothing else.
///
/// A line carrying a sideways run cannot use `Td`: `Tm` REPLACES the line
/// matrix, so a relative step cannot straddle a rotation. Every item is
/// therefore positioned by its own absolute `Tm` — identity for an upright
/// cell, `[0, −1, 1, 0, …]` for a rotated run (θ = −90°: text +x → page
/// (0,−1) so the run advances DOWN, text +y → page (1,0) so the glyph tops
/// face right, the conventional sideways-right form). `Tf` is written only
/// when the item's font differs from the current one.
///
/// The whole line is wrapped in a **mandatory** line-level
/// `/Span <</ActualText …>>` on the content pass — for a mixed line too, and
/// only there. That is not a nicety: the per-item steps carry no reading
/// order, so an extractor without the span reads a vertical title as garbage
/// (probe-verified against poppler, which honours `/ActualText` on bare
/// marked content). The halo pass is an artifact and carries no span, exactly
/// as the horizontal one does.
pub(super) fn show_vertical_line(
    content: &mut Content,
    line: &VerticalLine,
    x: f32,
    y: f32,
    size: f32,
    mark: TextMark,
) {
    if line.items.is_empty() {
        return;
    }
    content.begin_text();
    let span = mark == TextMark::Content;
    if span {
        let mut marked = content.begin_marked_content_with_properties(Name(b"Span"));
        marked.properties().actual_text(TextStr(&line.actual_text));
    }
    if line.is_all_upright() {
        show_upright_column(content, line, x, y, size);
    } else {
        show_mixed_column(content, line, x, y, size);
    }
    if span {
        content.end_marked_content();
    }
    content.end_text();
}

/// The v1.4 all-upright emission, unchanged: one `Tf`, relative `Td` steps,
/// one `Tj` per cell.
fn show_upright_column(content: &mut Content, line: &VerticalLine, x: f32, y: f32, size: f32) {
    let name = format!("F{}", line.font + 1);
    content.set_font(Name(name.as_bytes()), size);
    let to_pt = |thousandths: f32| thousandths / 1000.0 * size;
    let mut previous_shift = 0.0_f32;
    let mut previous_pitch = 0.0_f32;
    for (index, item) in line.items.iter().enumerate() {
        let VerticalItem::Upright(glyph) = item else {
            continue;
        };
        let shift = to_pt(glyph.x_shift_1000);
        if index == 0 {
            content.next_line(x + shift, y);
        } else {
            content.next_line(shift - previous_shift, -to_pt(previous_pitch));
        }
        content.show(Str(&glyph.cid.to_be_bytes()));
        previous_shift = shift;
        previous_pitch = glyph.pitch_1000;
    }
}

/// The v1.5 mixed emission: one absolute `Tm` per item, `Tf` only on a font
/// change, and `emit_run` verbatim for a sideways run.
fn show_mixed_column(content: &mut Content, line: &VerticalLine, x: f32, y: f32, size: f32) {
    let to_pt = |thousandths: f32| thousandths / 1000.0 * size;
    let mut pen_1000 = 0.0_f32;
    let mut current_font: Option<usize> = None;
    let mut rise = 0.0_f32;
    for item in &line.items {
        let font = item.font(line.font);
        if current_font != Some(font) {
            let name = format!("F{}", font + 1);
            content.set_font(Name(name.as_bytes()), size);
            current_font = Some(font);
        }
        let pen_pt = to_pt(pen_1000);
        match item {
            VerticalItem::Upright(glyph) => {
                let shift = to_pt(glyph.x_shift_1000);
                content.set_text_matrix([1.0, 0.0, 0.0, 1.0, x + shift, y - pen_pt]);
                content.show(Str(&glyph.cid.to_be_bytes()));
            }
            VerticalItem::Rotated(rotated) => {
                let tx = to_pt(rotated.tx_1000);
                content.set_text_matrix([0.0, -1.0, 1.0, 0.0, x + tx, y - pen_pt]);
                // The line already carries its own `/ActualText`, so the run
                // emits no cluster spans — the flat structure D-B3 buys.
                emit_run(content, &rotated.run, size, false, &mut rise);
            }
        }
        pen_1000 += item.advance_1000();
    }
    if rise != 0.0 {
        // `Ts` is state `BT` does not clear; leave it as every other path
        // in this module leaves it.
        content.set_rise(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::super::font::{ActualSpan, GlyphRun, RunGlyph};
    use super::{emit_run, win_ansi};
    use pdf_writer::Content;

    #[test]
    fn win_ansi_keeps_everything_the_encoding_can_name_and_replaces_the_rest() {
        assert_eq!(win_ansi("Map of Tokyo (2026)"), b"Map of Tokyo (2026)");
        assert_eq!(win_ansi("東京の地図"), b"?????");
        // The Latin-1 upper half is its own byte value — the whole point of
        // WinAnsi over bare ASCII.
        assert_eq!(win_ansi("München"), b"M\xfcnchen");
        assert_eq!(win_ansi("Qu\u{e9}bec"), b"Qu\xe9bec");
        // The CP1252 0x80-0x9F block: an em dash and curly quotes are real
        // codes, not '?'.
        assert_eq!(win_ansi("a\u{2014}b"), b"a\x97b");
        assert_eq!(win_ansi("\u{201c}x\u{201d}"), b"\x93x\x94");
        assert_eq!(win_ansi("\u{20ac}5"), b"\x805");
        assert_eq!(win_ansi("\u{2026}"), b"\x85");
        // Outside the repertoire, the honest '?' stays: Ō has no WinAnsi slot.
        assert_eq!(win_ansi("\u{14c}ita \u{2014} 大分"), b"?ita \x97 ??");
    }

    fn glyph(cid: u16, width: f32, adjust: f32, x_shift: f32, rise: f32) -> RunGlyph {
        RunGlyph {
            cid,
            width_1000: width,
            adjust_1000: adjust,
            x_shift_1000: x_shift,
            rise_1000: rise,
        }
    }

    fn emitted(run: &GlyphRun, cluster_spans: bool) -> (String, f32) {
        let mut content = Content::new();
        let mut rise = 0.0_f32;
        emit_run(&mut content, run, 16.0, cluster_spans, &mut rise);
        (
            String::from_utf8_lossy(&content.finish()).into_owned(),
            rise,
        )
    }

    #[test]
    fn a_plain_run_is_a_single_tj_with_no_marked_content() {
        let run = GlyphRun {
            font: 0,
            glyphs: vec![
                glyph(1, 600.0, 0.0, 0.0, 0.0),
                glyph(2, 600.0, 0.0, 0.0, 0.0),
            ],
            advance_1000: 1200.0,
            spans: Vec::new(),
        };
        let (text, rise) = emitted(&run, true);
        assert!(text.contains("Tj"), "the fast path is a plain Tj: {text}");
        assert!(!text.contains("TJ"), "no array for an unkerned run: {text}");
        assert!(!text.contains("Span"), "no marked content: {text}");
        assert!(!text.contains("Ts"), "no rise: {text}");
        assert_eq!(rise, 0.0);
    }

    #[test]
    fn an_x_shift_emits_the_paired_tj_numbers_with_zero_net_pen() {
        // A mark shifted left by 100: `100` before it (TJ numbers are
        // subtracted, so +100 moves the pen LEFT... the pre number is −x =
        // −(−100) — the mark's x_shift is negative), then the give-back.
        let run = GlyphRun {
            font: 0,
            glyphs: vec![
                glyph(1, 600.0, 0.0, 0.0, 0.0),
                glyph(2, 0.0, 0.0, -100.0, 0.0),
            ],
            advance_1000: 600.0,
            spans: vec![ActualSpan {
                start: 0,
                end: 2,
                text: "x\u{0301}".to_string(),
            }],
        };
        let (text, _) = emitted(&run, true);
        assert!(
            text.contains("TJ"),
            "a shifted glyph needs the array: {text}"
        );
        assert!(
            text.contains("100") && text.contains("-100"),
            "the pre and the give-back are both written: {text}"
        );
        assert!(
            text.contains("/ActualText"),
            "an offset cluster is span-wrapped: {text}"
        );
    }

    #[test]
    fn a_rise_sets_ts_and_reports_the_dirty_state_for_the_line_reset() {
        let run = GlyphRun {
            font: 0,
            glyphs: vec![
                glyph(1, 600.0, 0.0, 0.0, 0.0),
                glyph(2, 0.0, 0.0, 0.0, 49.8),
            ],
            advance_1000: 600.0,
            spans: vec![ActualSpan {
                start: 0,
                end: 2,
                text: "สุ".to_string(),
            }],
        };
        let (text, rise) = emitted(&run, true);
        assert!(text.contains("Ts"), "the rise is emitted: {text}");
        assert_eq!(rise, 49.8, "the caller sees the open rise and resets it");
    }

    #[test]
    fn spans_are_balanced_and_never_open_inside_a_tj_array() {
        let run = GlyphRun {
            font: 0,
            glyphs: vec![
                glyph(1, 600.0, 40.0, 0.0, 0.0),
                glyph(2, 500.0, 0.0, 0.0, 0.0),
                glyph(3, 400.0, 0.0, 0.0, 0.0),
            ],
            advance_1000: 1460.0,
            spans: vec![ActualSpan {
                start: 1,
                end: 3,
                text: "कि".to_string(),
            }],
        };
        let (text, _) = emitted(&run, true);
        let bdc = text.matches("BDC").count();
        let emc = text.matches("EMC").count();
        assert_eq!(bdc, 1, "one span: {text}");
        assert_eq!(bdc, emc, "balanced: {text}");
        // No BDC between a `[` and its `]`.
        let mut depth = 0_i32;
        for (index, byte) in text.bytes().enumerate() {
            match byte {
                b'[' => depth += 1,
                b']' => depth -= 1,
                b'B' if depth > 0 && text.as_bytes().get(index..index + 3) == Some(b"BDC") => {
                    panic!("a BDC opened inside a TJ array: {text}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn artifact_mode_suppresses_cluster_spans() {
        let run = GlyphRun {
            font: 0,
            glyphs: vec![
                glyph(1, 600.0, 0.0, 0.0, 0.0),
                glyph(2, 500.0, 0.0, 0.0, 0.0),
            ],
            advance_1000: 1100.0,
            spans: vec![ActualSpan {
                start: 0,
                end: 2,
                text: "कि".to_string(),
            }],
        };
        // `cluster_spans == false` is what `TextMark::Artifact` (and a
        // line-level span) passes down.
        let (text, _) = emitted(&run, false);
        assert!(
            !text.contains("Span"),
            "an artifact pass draws geometry only: {text}"
        );
    }

    fn upright_cell(cid: u16, pitch: f32, shift: f32) -> super::VerticalItem {
        use super::super::vertical::{VerticalGlyph, VerticalItem};
        VerticalItem::Upright(VerticalGlyph {
            cid,
            pitch_1000: pitch,
            x_shift_1000: shift,
        })
    }

    fn vertical_line() -> super::super::vertical::VerticalLine {
        use super::super::vertical::VerticalLine;
        VerticalLine {
            font: 0,
            items: vec![
                upright_cell(11, 1000.0, 0.0),
                upright_cell(12, 1000.0, 250.0),
                upright_cell(13, 500.0, 0.0),
            ],
            actual_text: "\u{6771}\u{4EAC}\u{3002}".to_string(),
            advance_1000: 2500.0,
        }
    }

    /// The same column with a sideways `2026` hung under it — one rotated
    /// item on a second face, the mixed case D-B4's `Tm` branch exists for.
    fn mixed_vertical_line() -> super::super::vertical::VerticalLine {
        use super::super::vertical::{RotatedRun, VerticalItem, VerticalLine};
        let run = GlyphRun {
            font: 1,
            glyphs: vec![
                glyph(20, 500.0, 0.0, 0.0, 0.0),
                glyph(21, 500.0, 12.0, 0.0, 0.0),
                glyph(22, 500.0, 0.0, 0.0, 0.0),
                glyph(23, 500.0, 0.0, 0.0, 0.0),
            ],
            advance_1000: 1988.0,
            spans: Vec::new(),
        };
        VerticalLine {
            font: 0,
            items: vec![
                upright_cell(11, 1000.0, 0.0),
                upright_cell(12, 1000.0, 250.0),
                VerticalItem::Rotated(RotatedRun {
                    run,
                    tx_1000: 130.0,
                }),
            ],
            actual_text: "\u{6771}\u{4EAC}2026".to_string(),
            advance_1000: 3988.0,
        }
    }

    fn shown(line: &super::super::vertical::VerticalLine, mark: super::TextMark) -> String {
        let mut content = Content::new();
        super::show_vertical_line(&mut content, line, 100.0, 700.0, 16.0, mark);
        String::from_utf8_lossy(&content.finish()).into_owned()
    }

    fn shown_vertical(mark: super::TextMark) -> String {
        shown(&vertical_line(), mark)
    }

    #[test]
    fn a_vertical_line_is_one_text_object_of_paired_td_and_tj() {
        let text = shown_vertical(super::TextMark::Content);
        assert_eq!(text.matches("BT").count(), 1, "ONE text object: {text}");
        assert_eq!(text.matches("ET").count(), 1, "{text}");
        assert_eq!(text.matches(" Td").count(), 3, "one Td per glyph: {text}");
        assert_eq!(text.matches(" Tj").count(), 3, "one Tj per glyph: {text}");
        assert!(!text.contains("TJ"), "no arrays in a vertical line: {text}");
        assert!(!text.contains(" Ts"), "no rise in a vertical line: {text}");
        // The first Td is absolute (the line origin); the rest step DOWN by
        // the previous pitch: 1000/1000 * 16 = 16 pt, then 16 pt again.
        assert!(text.contains("100 700 Td"), "{text}");
        assert!(text.contains("-16 Td"), "the pen steps down: {text}");
        // The middle glyph is narrower than its cell, so it shifts right by
        // 250/1000 * 16 = 4 pt and the next glyph gives it back.
        assert!(text.contains("4 -16 Td"), "the centring shift: {text}");
        assert!(text.contains("-4 -16 Td"), "and its give-back: {text}");
    }

    #[test]
    fn a_vertical_line_always_carries_its_actual_text_span() {
        // MANDATORY: the per-glyph Td steps carry no reading order, so
        // without the span a vertical title extracts as garbage.
        let text = shown_vertical(super::TextMark::Content);
        assert_eq!(text.matches("BDC").count(), 1, "{text}");
        assert_eq!(text.matches("EMC").count(), 1, "balanced: {text}");
        assert!(text.contains("/Span"), "{text}");
        assert!(
            text.contains("/ActualText <FEFF"),
            "UTF-16BE with a BOM: {text}",
        );
        // The halo pass repeats the geometry and must NOT repeat the span.
        let artifact = shown_vertical(super::TextMark::Artifact);
        assert!(!artifact.contains("Span"), "{artifact}");
        assert!(!artifact.contains("BDC"), "{artifact}");
        assert_eq!(
            artifact.matches(" Tj").count(),
            3,
            "the artifact still draws every glyph: {artifact}",
        );
    }

    #[test]
    fn a_vertical_box_is_one_em_wide_and_the_summed_pitch_tall() {
        assert_eq!(vertical_line().box_pt(16.0), [16.0, 40.0]);
    }

    // --- print/text v1.5 (D-B4): the mixed line and its byte-identity gate ---

    /// THE gate. A refusal — and an all-upright title is what every refusal
    /// falls back to — must leave today's output byte for byte, so the v1.4
    /// stream is checked in verbatim rather than described.
    #[test]
    fn an_upright_only_vertical_line_is_byte_identical_to_v1_4() {
        // Written out operator by operator, exactly as `pdf-writer` 0.15
        // serialises it — a literal `( )` string per CID, one `Tf`, the
        // absolute `Td` then two relative ones, and the balanced span.
        let v1_4 = [
            "BT",
            "/Span <<",
            "  /ActualText <FEFF67714EAC3002>",
            ">> BDC",
            "/F1 16 Tf",
            "100 700 Td",
            "(\\000\\013) Tj",
            "4 -16 Td",
            "(\\000\\f) Tj",
            "-4 -16 Td",
            "(\\000\\r) Tj",
            "EMC",
            "ET",
        ]
        .join("\n");
        assert_eq!(shown_vertical(super::TextMark::Content), v1_4);
        assert!(
            vertical_line().is_all_upright(),
            "and the branch is chosen by this predicate and nothing else",
        );
    }

    #[test]
    fn a_mixed_vertical_line_positions_every_item_with_an_absolute_tm() {
        let text = shown(&mixed_vertical_line(), super::TextMark::Content);
        assert_eq!(
            text.matches("BT").count(),
            1,
            "still ONE text object: {text}"
        );
        assert_eq!(text.matches("ET").count(), 1, "{text}");
        assert!(
            !text.contains(" Td"),
            "a mixed line never steps relatively: {text}",
        );
        assert_eq!(text.matches(" Tm").count(), 3, "one Tm per item: {text}");
        // Upright cells: identity matrix, y stepping down by the pitch.
        assert!(text.contains("1 0 0 1 100 700 Tm"), "{text}");
        assert!(
            text.contains("1 0 0 1 104 684 Tm"),
            "the shift rides x: {text}"
        );
        // The rotated run: θ = −90°, x offset by tx = 130/1000 * 16 = 2.08 pt,
        // y already dropped by both pitches (2 * 16 pt).
        assert!(text.contains("0 -1 1 0 102.08 668 Tm"), "{text}");
        // Two faces, two Tf — and the rotated run keeps its kerning.
        assert_eq!(text.matches(" Tf").count(), 2, "one Tf per font: {text}");
        assert!(text.contains("/F1 16 Tf"), "{text}");
        assert!(text.contains("/F2 16 Tf"), "{text}");
        assert!(
            text.contains("TJ"),
            "the kerned run keeps its array: {text}"
        );
        assert!(!text.contains(" Ts"), "no rise leaks: {text}");
    }

    #[test]
    fn a_mixed_vertical_line_still_carries_exactly_one_line_level_actual_text_span() {
        let text = shown(&mixed_vertical_line(), super::TextMark::Content);
        assert_eq!(text.matches("BDC").count(), 1, "ONE span, flat: {text}");
        assert_eq!(text.matches("EMC").count(), 1, "balanced: {text}");
        assert!(text.contains("/ActualText <FEFF"), "{text}");
        // The halo pass repeats the geometry and no marked content.
        let artifact = shown(&mixed_vertical_line(), super::TextMark::Artifact);
        assert!(!artifact.contains("BDC"), "{artifact}");
        assert_eq!(
            artifact.matches(" Tm").count(),
            3,
            "the artifact still positions every item: {artifact}",
        );
    }

    #[test]
    fn an_upright_item_of_a_mixed_line_emits_no_tj_array_and_no_rise() {
        let text = shown(&mixed_vertical_line(), super::TextMark::Content);
        // The two upright cells are plain `Tj`s; only the kerned rotated run
        // reaches `TJ`.
        assert_eq!(text.matches(" Tj").count(), 2, "{text}");
        assert_eq!(text.matches(" TJ").count(), 1, "{text}");
    }

    #[test]
    fn a_rotated_only_line_is_one_matrix_and_one_run() {
        use super::super::vertical::{RotatedRun, VerticalItem, VerticalLine};
        let line = VerticalLine {
            font: 0,
            items: vec![VerticalItem::Rotated(RotatedRun {
                run: GlyphRun {
                    font: 0,
                    glyphs: vec![glyph(20, 500.0, 0.0, 0.0, 0.0)],
                    advance_1000: 500.0,
                    spans: Vec::new(),
                },
                tx_1000: 0.0,
            })],
            actual_text: "A".to_string(),
            advance_1000: 500.0,
        };
        assert!(!line.is_all_upright());
        let text = shown(&line, super::TextMark::Content);
        assert_eq!(text.matches(" Tm").count(), 1, "{text}");
        assert!(text.contains("0 -1 1 0 100 700 Tm"), "{text}");
        assert_eq!(line.box_pt(16.0), [16.0, 8.0]);
    }
}
