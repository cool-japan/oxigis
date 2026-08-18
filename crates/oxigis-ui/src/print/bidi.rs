// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! UAX #9 paragraph analysis, script sub-segmentation and L4 mirroring for
//! the PDF export (print v1.3). Pure computation — no fs, no process, wasm32
//! clean; `oxitext::BidiParagraph` (re-exported unconditionally, backed by
//! `unicode-bidi`, already in the graph) does L1+L2 and hands back runs in
//! VISUAL order, so this module only sub-segments, mirrors and refuses.
//!
//! NOT oxitext's `icu` feature: that would pull the icu crates into the
//! graph AND make `shape_request` silently NFC-normalize its text — a
//! byte-output change the byte-identity floor forbids.

use core::ops::Range;
use std::collections::BTreeMap;

use oxitext::BidiParagraph;

use super::mirror_table;

/// One shapeable segment: uniform bidi level, uniform script, uniform face,
/// already in visual (left-to-right) order across the whole vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BidiSegment {
    /// Byte range in the RESOLVED string, logical order within the segment.
    pub range: Range<usize>,
    /// UAX #9 embedding level; odd == RTL.
    pub level: u8,
    /// OpenType script tag handed to `shape_request` (`b"arab"`, `b"hebr"`, …).
    pub script: [u8; 4],
    /// Index into the shell's font chain.
    pub chain_index: usize,
}

impl BidiSegment {
    /// Whether this segment renders right-to-left.
    pub fn is_rtl(&self) -> bool {
        self.level % 2 == 1
    }
}

/// The script classes the segmenter distinguishes. Neutrals (spaces,
/// punctuation, digits — anything non-alphabetic outside the listed blocks)
/// inherit the surrounding class inside a run, so a split can never land on
/// a space and never cuts an Arabic joining chain (a non-Arabic letter is
/// joining type U and breaks the chain anyway).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScriptClass {
    Arab,
    Hebr,
    Syrc,
    Thaa,
    Nkoo,
    Latn,
    Neutral,
}

impl ScriptClass {
    fn of(ch: char) -> Self {
        match u32::from(ch) {
            0x0600..=0x06FF
            | 0x0750..=0x077F
            | 0x08A0..=0x08FF
            | 0xFB50..=0xFDFF
            | 0xFE70..=0xFEFF => Self::Arab,
            0x0590..=0x05FF | 0xFB1D..=0xFB4F => Self::Hebr,
            0x0700..=0x074F => Self::Syrc,
            0x0780..=0x07BF => Self::Thaa,
            0x07C0..=0x07FF => Self::Nkoo,
            _ if ch.is_alphabetic() => Self::Latn,
            _ => Self::Neutral,
        }
    }

    /// The OpenType script tag handed to `shape_request`.
    ///
    /// **Latent defect, recorded in v1.4 and deliberately unchanged**:
    /// the shaper resolves `b"nkoo"` to `None`, so an N'Ko segment shapes
    /// as Latin under an `Rtl` direction — visually reversed but
    /// contextually unjoined. Every other tag here is measured to resolve.
    ///
    /// v1.4 could not say which tag WOULD resolve; that is now known. The
    /// accepted spelling is `b"nko "` with a TRAILING SPACE — decoded from
    /// `SCRIPTS_BY_TAG` in oxitext-swash 0.2.2
    /// (`src/text/unicode_data/script_tables.rs:139`, `110<<24|107<<16|111<<8|32`),
    /// which is also where the `dev2`/`bng2`/`ory2`/`knd2` four-character
    /// Indic spellings come from. The byte is STILL not changed here: N'Ko is
    /// a cursive joining script and this repo has no N'Ko corpus, so swapping
    /// the tag is a real rendering change that nothing would catch. It lands
    /// with a live golden against a covering face, not on the strength of a
    /// table lookup.
    fn tag(self) -> [u8; 4] {
        match self {
            Self::Arab => *b"arab",
            Self::Hebr => *b"hebr",
            Self::Syrc => *b"syrc",
            Self::Thaa => *b"thaa",
            Self::Nkoo => *b"nkoo",
            // A neutral-only run never reaches `tag` (it inherits or the
            // whole run defaults to Latin below).
            Self::Latn | Self::Neutral => *b"latn",
        }
    }
}

/// True for the explicit embedding/override/isolate controls the bidi path
/// refuses (LRE/RLE/PDF/LRO/RLO, LRI/RLI/FSI/PDI): an override can push
/// Arabic into an even level, where the Latin-scripted `shape_slice` would
/// draw isolated forms — refusal to the v1.1 path is strictly safer.
/// RLM/LRM/ZWJ/ZWNJ are deliberately NOT here — the shaper drops them and
/// the cluster classifier absorbs them.
pub(super) fn has_explicit_bidi_control(text: &str) -> bool {
    text.chars()
        .any(|ch| matches!(u32::from(ch), 0x202A..=0x202E | 0x2066..=0x2069))
}

/// UAX #9 L4 bracket/quote mirroring — the COMPLETE Unicode 16.0.0 table
/// since v1.4, one `binary_search_by_key` over
/// [`MIRROR_PAIRS`](super::mirror_table::MIRROR_PAIRS) (the v1.3
/// hand-curated 32-codepoint match covered 7 % of it and is deleted).
///
/// Every pair has EQUAL UTF-8 length, so a mirrored display string indexes
/// byte-for-byte like its source and swash cluster offsets stay exact with
/// no offset map (pinned by a test). Shaping does not mirror (measured:
/// `'('` keeps its gid inside an `arab` RTL run), so this is the caller's
/// job, applied to the DISPLAY text only — `/ToUnicode` and `/ActualText`
/// keep the logical source, which is what makes the lie-free extraction
/// story hold (the line-level `/ActualText` span overrides the mirrored
/// CID's mapping; if that line-span rule is ever narrowed, mirroring must
/// be narrowed with it).
///
/// This is the RAW property. Callers that are about to SHAPE the result
/// must go through [`mirror_for`] instead — see its contract for why.
pub(super) fn mirror(ch: char) -> Option<char> {
    let mirrored = mirror_table::MIRROR_PAIRS
        .binary_search_by_key(&ch, |&(from, _)| from)
        .ok()
        .map(|index| mirror_table::MIRROR_PAIRS[index].1)?;
    debug_assert_eq!(ch.len_utf8(), mirrored.len_utf8());
    Some(mirrored)
}

/// [`mirror`] guarded by glyph coverage: the mirrored character only
/// replaces `ch` when the SAME chain face covers both.
///
/// Growing the table from 32 to 428 codepoints multiplies the chance that
/// the partner of a mirrored character has no glyph in the face that draws
/// the character itself (the `⋲`/`⋺` classes are exactly what real fonts
/// lack). Such a partner shapes to gid 0, `shape::cluster_runs` refuses the
/// segment, and the ALL-OR-NOTHING rule then degrades the whole line to the
/// v1.1 per-character walk — a strict regression caused by completeness.
/// So the guard is part of the same landing as the table: an uncoverable
/// partner leaves `ch` unmirrored, which is a locally wrong-way bracket —
/// exactly what every viewer without UAX #9 L4 already shows, and strictly
/// better than un-shaping the line around it.
///
/// Requiring the same face (rather than merely "some face covers it") also
/// preserves the one-face-per-segment invariant `segments` establishes: a
/// mirrored character must not silently move its cluster onto another face.
pub(super) fn mirror_for(ch: char, coverage: &BTreeMap<char, usize>) -> char {
    let Some(face) = coverage.get(&ch) else {
        return ch;
    };
    match mirror(ch) {
        Some(mirrored) if coverage.get(&mirrored) == Some(face) => mirrored,
        _ => ch,
    }
}

/// Splits `resolved` into shapeable segments **already in visual order**.
///
/// `coverage` maps every character of `resolved` to its chain face (the
/// v1.1 coverage walk's output). [`None`] means this string must keep the
/// v1.1 per-character path — the all-or-nothing rule: a half-reordered line
/// with a logical-order fallback run spliced in would be worse than uniform
/// v1.1 output. Refused here: explicit bidi controls; a character missing
/// from `coverage` (its bidi class would be a lie after `?` substitution);
/// a script-uniform stretch spanning more than one chain face (measured: a
/// joined word split at a face seam changes both the glyph forms and the
/// total advance — it must be shaped in one piece by one face).
pub(super) fn segments(
    resolved: &str,
    coverage: &BTreeMap<char, usize>,
) -> Option<Vec<BidiSegment>> {
    if has_explicit_bidi_control(resolved) {
        return None;
    }
    let paragraph = BidiParagraph::new(resolved, None);
    let mut out = Vec::new();
    for run in paragraph.runs() {
        let text = resolved.get(run.start..run.end)?;
        // Pass 1 (logical): a script class per character, neutrals
        // inheriting the previous class, then the following one for a
        // neutral head, then Latin for a neutral-only run.
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        if chars.is_empty() {
            continue;
        }
        let mut classes: Vec<ScriptClass> =
            chars.iter().map(|&(_, ch)| ScriptClass::of(ch)).collect();
        let mut previous = None;
        for class in classes.iter_mut() {
            match (*class, previous) {
                (ScriptClass::Neutral, Some(inherited)) => *class = inherited,
                (ScriptClass::Neutral, None) => {}
                (concrete, _) => previous = Some(concrete),
            }
        }
        let mut following = None;
        for class in classes.iter_mut().rev() {
            match (*class, following) {
                (ScriptClass::Neutral, Some(inherited)) => *class = inherited,
                (ScriptClass::Neutral, None) => *class = ScriptClass::Latn,
                (concrete, _) => following = Some(concrete),
            }
        }
        // Pass 2 (logical): maximal same-script stretches; each must sit on
        // ONE chain face or the whole string refuses.
        let mut stretches: Vec<(Range<usize>, ScriptClass, usize)> = Vec::new();
        for (index, &(offset, ch)) in chars.iter().enumerate() {
            let face = *coverage.get(&ch)?;
            let class = classes[index];
            let end = offset + ch.len_utf8();
            match stretches.last_mut() {
                Some((range, last_class, last_face)) if *last_class == class => {
                    if *last_face != face {
                        // A joined script split at a face seam garbles; the
                        // whole line keeps the v1.1 path instead.
                        return None;
                    }
                    range.end = end;
                }
                _ => stretches.push((offset..end, class, face)),
            }
        }
        // Pass 3: an odd-level run renders right-to-left, so its
        // logical-first stretch is the visually RIGHTMOST — push reversed.
        let odd = run.level % 2 == 1;
        let mapped = stretches
            .into_iter()
            .map(|(range, class, face)| BidiSegment {
                range: (run.start + range.start)..(run.start + range.end),
                level: run.level,
                script: class.tag(),
                chain_index: face,
            });
        if odd {
            let mut reversed: Vec<BidiSegment> = mapped.collect();
            reversed.reverse();
            out.extend(reversed);
        } else {
            out.extend(mapped);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{BidiSegment, has_explicit_bidi_control, mirror, mirror_for, segments};
    use std::collections::BTreeMap;

    /// Coverage that puts every character on chain face 0.
    fn all_on_face_zero(text: &str) -> BTreeMap<char, usize> {
        text.chars().map(|ch| (ch, 0)).collect()
    }

    /// The segment's text, for readable assertions.
    fn slice<'a>(text: &'a str, segment: &BidiSegment) -> &'a str {
        &text[segment.range.clone()]
    }

    #[test]
    fn segments_are_visual_order_for_a_mixed_paragraph() {
        // Probe-measured golden: base 0; visual order is the leading Latin,
        // then the number (level 2), then the Arabic (level 1), then the
        // trailing Latin.
        let text = "Rue \u{627}\u{644}\u{645}\u{644}\u{643} 12, Cairo";
        let got = segments(text, &all_on_face_zero(text)).expect("segments");
        let texts: Vec<&str> = got.iter().map(|segment| slice(text, segment)).collect();
        assert_eq!(texts[0], "Rue ");
        assert!(
            texts.iter().position(|t| t.contains("12"))
                < texts.iter().position(|t| t.contains('\u{627}')),
            "the number renders LEFT of the Arabic: {texts:?}"
        );
        assert_eq!(*texts.last().expect("non-empty"), ", Cairo");
        let levels: Vec<u8> = got.iter().map(|segment| segment.level).collect();
        assert!(levels.contains(&1), "the Arabic run is level 1: {levels:?}");
        assert!(levels.contains(&2), "the number is level 2: {levels:?}");
    }

    #[test]
    fn an_rtl_run_pushes_its_sub_segments_reversed() {
        // Hebrew then Arabic in logical order: inside the RTL run the
        // logical-first (Hebrew) stretch is visually RIGHTMOST, so the
        // Arabic segment must be emitted first (leftmost).
        let text = "\u{5E9}\u{5DC}\u{5D5}\u{5DD} \u{645}\u{631}\u{62D}\u{628}\u{627}";
        let got = segments(text, &all_on_face_zero(text)).expect("segments");
        assert_eq!(got.len(), 2, "two script stretches: {got:?}");
        assert_eq!(got[0].script, *b"arab", "Arabic left of Hebrew");
        assert_eq!(got[1].script, *b"hebr");
        assert!(got.iter().all(BidiSegment::is_rtl));
    }

    #[test]
    fn pure_rtl_and_pure_ltr_paragraphs_have_the_expected_levels() {
        let arabic = "\u{645}\u{631}\u{62D}\u{628}\u{627}";
        let got = segments(arabic, &all_on_face_zero(arabic)).expect("segments");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].level, 1, "auto base direction (P2/P3) is RTL");

        let latin = "Hello";
        let got = segments(latin, &all_on_face_zero(latin)).expect("segments");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].level, 0);
        assert_eq!(got[0].script, *b"latn");
    }

    #[test]
    fn neutrals_inherit_the_surrounding_script() {
        // An em-dash and spaces between two Arabic words stay ONE segment.
        let text = "\u{645}\u{631} \u{2014} \u{645}\u{631}";
        let got = segments(text, &all_on_face_zero(text)).expect("segments");
        assert_eq!(
            got.len(),
            1,
            "one arab segment, no split on neutrals: {got:?}"
        );
        assert_eq!(got[0].script, *b"arab");
    }

    #[test]
    fn a_script_stretch_spanning_two_faces_refuses() {
        // A joined word whose middle letter lives on a different face:
        // shaping it in pieces damages the forms AND the advances
        // (measured), so the whole string keeps the v1.1 path. Coverage is
        // per-CHARACTER, so the seam is simulated with a distinct middle
        // character on face 1.
        let seam = "\u{628}\u{62D}\u{628}";
        let mut coverage = all_on_face_zero(seam);
        coverage.insert('\u{62D}', 1);
        assert_eq!(segments(seam, &coverage), None);
    }

    #[test]
    fn a_character_missing_from_coverage_refuses() {
        let text = "\u{645}X";
        let mut coverage = all_on_face_zero(text);
        coverage.remove(&'X');
        assert_eq!(segments(text, &coverage), None);
    }

    #[test]
    fn mirror_is_an_involution_and_utf8_length_preserving() {
        // The v1.3 curated set is the regression fence (the full table's own
        // involution and length gates live in `mirror_table`'s tests).
        let table = "()[]{}<>\u{00AB}\u{00BB}\u{2039}\u{203A}\u{2264}\u{2265}\u{226A}\u{226B}\
                     \u{FF08}\u{FF09}\u{FF3B}\u{FF3D}\u{FF5B}\u{FF5D}\u{3008}\u{3009}\u{300A}\
                     \u{300B}\u{300C}\u{300D}\u{300E}\u{300F}\u{3010}\u{3011}";
        for ch in table.chars() {
            let mirrored = mirror(ch).expect("every table entry mirrors");
            assert_eq!(mirror(mirrored), Some(ch), "involution fails for {ch:?}");
            assert_eq!(
                ch.len_utf8(),
                mirrored.len_utf8(),
                "length changes for {ch:?}"
            );
        }
        assert_eq!(mirror('\u{2018}'), None, "quotes are Bidi_Mirrored=N");
        assert_eq!(mirror('a'), None);
        // v1.4: codepoints the curated table missed now mirror — the 396 the
        // page used to draw the wrong way round inside an RTL line.
        assert_eq!(mirror('\u{2308}'), Some('\u{2309}'), "LEFT CEILING");
        assert_eq!(mirror('\u{27E6}'), Some('\u{27E7}'), "white square bracket");
        assert_eq!(mirror('\u{2208}'), Some('\u{220B}'), "ELEMENT OF");
        assert_eq!(mirror('\u{FF63}'), Some('\u{FF62}'), "the table's maximum");
    }

    #[test]
    fn mirror_for_refuses_a_partner_the_covering_face_lacks() {
        // The completeness regression the guard exists for: '⋲' (U+22F2)
        // mirrors to '⋺' (U+22FA), which few faces carry. With the partner
        // absent from coverage the character stays as it is — a wrong-way
        // bracket beats un-shaping the whole line.
        let mut coverage = BTreeMap::new();
        coverage.insert('\u{22F2}', 0_usize);
        assert_eq!(mirror_for('\u{22F2}', &coverage), '\u{22F2}');
        // Present on the SAME face: mirrored.
        coverage.insert('\u{22FA}', 0);
        assert_eq!(mirror_for('\u{22F2}', &coverage), '\u{22FA}');
    }

    #[test]
    fn mirror_for_refuses_a_partner_that_lives_on_a_different_face() {
        // Mirroring across a face seam would move the cluster off the
        // segment's one face — the invariant `segments` establishes.
        let mut coverage = BTreeMap::new();
        coverage.insert('(', 0_usize);
        coverage.insert(')', 1_usize);
        assert_eq!(mirror_for('(', &coverage), '(', "no mirror, no refusal");
        coverage.insert(')', 0);
        assert_eq!(mirror_for('(', &coverage), ')');
    }

    #[test]
    fn mirror_for_passes_unmirrored_and_uncovered_characters_through() {
        let mut coverage = BTreeMap::new();
        coverage.insert('a', 0_usize);
        assert_eq!(mirror_for('a', &coverage), 'a', "Bidi_Mirrored=N");
        assert_eq!(mirror_for('(', &coverage), '(', "not covered at all");
    }

    #[test]
    fn explicit_bidi_controls_are_detected_and_zero_width_marks_are_not() {
        assert!(has_explicit_bidi_control("a\u{202D}b"));
        assert!(has_explicit_bidi_control("a\u{2066}b"));
        assert!(
            !has_explicit_bidi_control("a\u{200F}b"),
            "RLM is absorbed, not refused"
        );
        assert!(
            !has_explicit_bidi_control("a\u{200D}b"),
            "ZWJ is absorbed, not refused"
        );
        let text = "a\u{202A}b";
        assert_eq!(segments(text, &all_on_face_zero(text)), None);
    }
}
