// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The Unicode 16.0.0 `Vertical_Orientation` property (UAX #50) — generated
//! data, not hand-written (print/text v1.4 D-V2; moved into this crate whole
//! by v1.5 D-A1).
//!
//! The UCD file is checked in beside the source under `data/`, the range
//! table was generated from it ONCE, and the test at the bottom re-derives
//! the whole property from that same file through `include_str!` and asserts
//! equality over the entire codespace. A hand edit, a stale table or a
//! Unicode bump all fail the suite. No `build.rs`, no tool crate, no network,
//! no new dependency. The `data/` file is `#[cfg(test)]`-only, so nothing of
//! it enters a shipped binary. The mirroring table in the PDF exporter is
//! built the same way.
//!
//! # Why ranges and not the raw file
//!
//! `VerticalOrientation.txt` lists 2 427 explicit entries; the property they
//! describe — once the file's own **default-U blocks** and the overall `R`
//! default are folded in — is 175 non-default ranges. Storing the resolved
//! property rather than the file's rows is what makes the lookup one
//! `binary_search_by` and keeps the default correct for unassigned code
//! points, which the raw rows do not describe at all.
//!
//! # One table, two consumers
//!
//! This module is the SINGLE definition for the whole workspace. The PDF
//! exporter's vertical-title path and the renderer's vertical-label path both
//! accept a character only when [`VerticalOrientation::draws_upright`] is
//! true — `U`, `Tu` or `Tr` — so the two ladders cannot drift apart on which
//! characters they can set. `R` (Latin, digits, halfwidth kana) needs a
//! rotated glyph transform: the exporter reaches it through
//! [`super::vertical::vertical_runs`], and the screen refuses the label and
//! draws it horizontally, byte for byte as it does today.
//!
//! `oxitext`'s own `is_upright_in_vertical` is deliberately NOT used: it is a
//! hand-rolled range list that disagrees with this checked-in UCD file on
//! ~248 900 code points in both directions, including every `Tr` bracket
//! 〈〉「」『』【】〔〕. A ladder that must agree with the page cannot rest on it.

/// UAX #50 `Vertical_Orientation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalOrientation {
    /// `U` — upright, the same orientation as in the code charts.
    Upright,
    /// `R` — rotated 90° clockwise. The property's DEFAULT, and the one
    /// value the vertical title path refuses.
    Rotated,
    /// `Tu` — transformed typographically, falling back to Upright.
    TransformedUpright,
    /// `Tr` — transformed typographically, falling back to Rotated.
    TransformedRotated,
}

impl VerticalOrientation {
    /// Whether a character with this orientation can be drawn upright with
    /// no glyph rotation — i.e. everything except [`VerticalOrientation::Rotated`].
    ///
    /// `Tr` is included on purpose: a face with real `vert`/`vrt2` lookups
    /// substitutes the vertical form, and a face without them (simsun, the
    /// measured case) draws the horizontal punctuation form upright, which
    /// is a known, logged, test-pinned state rather than garbage.
    pub fn draws_upright(self) -> bool {
        !matches!(self, Self::Rotated)
    }
}

/// The Unicode version this module's range table was generated from.
///
/// Also the verification test's oracle for the data file's own version line;
/// the table itself is a plain sorted array at runtime.
pub const VERTICAL_ORIENTATION_UNICODE_VERSION: &str = "16.0.0";

/// The code-point blocks whose UNASSIGNED members default to `Upright`,
/// verbatim from the data file's header. Explicit rows override them, and
/// everything outside them defaults to [`VerticalOrientation::Rotated`].
#[cfg(test)]
#[rustfmt::skip]
const DEFAULT_UPRIGHT: [(u32, u32); 36] = [
    (0x018B0, 0x018FF), (0x02065, 0x02065), (0x02150, 0x0218F), (0x02400, 0x0245F),
    (0x02BB8, 0x02BFF), (0x02E80, 0x0A4CF), (0x0A960, 0x0A97F), (0x0AC00, 0x0D7FF),
    (0x0E000, 0x0FAFF), (0x0FE10, 0x0FE1F), (0x0FE50, 0x0FE6F), (0x0FFE7, 0x0FFE7),
    (0x0FFF0, 0x0FFF8), (0x11580, 0x115FF), (0x11A00, 0x11AAF), (0x13000, 0x143FF),
    (0x14400, 0x1467F), (0x16FE0, 0x18AFF), (0x18B00, 0x18D7F), (0x1AFF0, 0x1AFFF),
    (0x1B100, 0x1B16F), (0x1B170, 0x1B2FF), (0x1CF00, 0x1CFCF), (0x1D000, 0x1D1FF),
    (0x1D2E0, 0x1D2FF), (0x1D300, 0x1D37F), (0x1D800, 0x1DAAF), (0x1F000, 0x1F0FF),
    (0x1F100, 0x1F2FF), (0x1F680, 0x1F7FF), (0x1F900, 0x1F9FF), (0x1FA00, 0x1FAFF),
    (0x20000, 0x2FFFD), (0x30000, 0x3FFFD), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD),
];

/// Every code-point range whose `Vertical_Orientation` is NOT the `R`
/// default, sorted and disjoint. Anything not covered here is
/// [`VerticalOrientation::Rotated`].
#[rustfmt::skip]
const VERTICAL_RANGES: [(u32, u32, VerticalOrientation); 175] = [
    (0x000A7, 0x000A7, VerticalOrientation::Upright),
    (0x000A9, 0x000A9, VerticalOrientation::Upright),
    (0x000AE, 0x000AE, VerticalOrientation::Upright),
    (0x000B1, 0x000B1, VerticalOrientation::Upright),
    (0x000BC, 0x000BE, VerticalOrientation::Upright),
    (0x000D7, 0x000D7, VerticalOrientation::Upright),
    (0x000F7, 0x000F7, VerticalOrientation::Upright),
    (0x002EA, 0x002EB, VerticalOrientation::Upright),
    (0x01100, 0x011FF, VerticalOrientation::Upright),
    (0x01401, 0x0167F, VerticalOrientation::Upright),
    (0x018B0, 0x018FF, VerticalOrientation::Upright),
    (0x02016, 0x02016, VerticalOrientation::Upright),
    (0x02020, 0x02021, VerticalOrientation::Upright),
    (0x02030, 0x02031, VerticalOrientation::Upright),
    (0x0203B, 0x0203C, VerticalOrientation::Upright),
    (0x02042, 0x02042, VerticalOrientation::Upright),
    (0x02047, 0x02049, VerticalOrientation::Upright),
    (0x02051, 0x02051, VerticalOrientation::Upright),
    (0x02065, 0x02065, VerticalOrientation::Upright),
    (0x020DD, 0x020E0, VerticalOrientation::Upright),
    (0x020E2, 0x020E4, VerticalOrientation::Upright),
    (0x02100, 0x02101, VerticalOrientation::Upright),
    (0x02103, 0x02109, VerticalOrientation::Upright),
    (0x0210F, 0x0210F, VerticalOrientation::Upright),
    (0x02113, 0x02114, VerticalOrientation::Upright),
    (0x02116, 0x02117, VerticalOrientation::Upright),
    (0x0211E, 0x02123, VerticalOrientation::Upright),
    (0x02125, 0x02125, VerticalOrientation::Upright),
    (0x02127, 0x02127, VerticalOrientation::Upright),
    (0x02129, 0x02129, VerticalOrientation::Upright),
    (0x0212E, 0x0212E, VerticalOrientation::Upright),
    (0x02135, 0x0213F, VerticalOrientation::Upright),
    (0x02145, 0x0214A, VerticalOrientation::Upright),
    (0x0214C, 0x0214D, VerticalOrientation::Upright),
    (0x0214F, 0x02189, VerticalOrientation::Upright),
    (0x0218C, 0x0218F, VerticalOrientation::Upright),
    (0x0221E, 0x0221E, VerticalOrientation::Upright),
    (0x02234, 0x02235, VerticalOrientation::Upright),
    (0x02300, 0x02307, VerticalOrientation::Upright),
    (0x0230C, 0x0231F, VerticalOrientation::Upright),
    (0x02324, 0x02328, VerticalOrientation::Upright),
    (0x02329, 0x0232A, VerticalOrientation::TransformedRotated),
    (0x0232B, 0x0232B, VerticalOrientation::Upright),
    (0x0237D, 0x0239A, VerticalOrientation::Upright),
    (0x023BE, 0x023CD, VerticalOrientation::Upright),
    (0x023CF, 0x023CF, VerticalOrientation::Upright),
    (0x023D1, 0x023DB, VerticalOrientation::Upright),
    (0x023E2, 0x02422, VerticalOrientation::Upright),
    (0x02424, 0x024FF, VerticalOrientation::Upright),
    (0x025A0, 0x02619, VerticalOrientation::Upright),
    (0x02620, 0x02767, VerticalOrientation::Upright),
    (0x02776, 0x02793, VerticalOrientation::Upright),
    (0x02B12, 0x02B2F, VerticalOrientation::Upright),
    (0x02B50, 0x02B59, VerticalOrientation::Upright),
    (0x02B97, 0x02B97, VerticalOrientation::Upright),
    (0x02BB8, 0x02BD1, VerticalOrientation::Upright),
    (0x02BD3, 0x02BEB, VerticalOrientation::Upright),
    (0x02BF0, 0x02BFF, VerticalOrientation::Upright),
    (0x02E50, 0x02E51, VerticalOrientation::Upright),
    (0x02E80, 0x03000, VerticalOrientation::Upright),
    (0x03001, 0x03002, VerticalOrientation::TransformedUpright),
    (0x03003, 0x03007, VerticalOrientation::Upright),
    (0x03008, 0x03011, VerticalOrientation::TransformedRotated),
    (0x03012, 0x03013, VerticalOrientation::Upright),
    (0x03014, 0x0301F, VerticalOrientation::TransformedRotated),
    (0x03020, 0x0302F, VerticalOrientation::Upright),
    (0x03030, 0x03030, VerticalOrientation::TransformedRotated),
    (0x03031, 0x03040, VerticalOrientation::Upright),
    (0x03041, 0x03041, VerticalOrientation::TransformedUpright),
    (0x03042, 0x03042, VerticalOrientation::Upright),
    (0x03043, 0x03043, VerticalOrientation::TransformedUpright),
    (0x03044, 0x03044, VerticalOrientation::Upright),
    (0x03045, 0x03045, VerticalOrientation::TransformedUpright),
    (0x03046, 0x03046, VerticalOrientation::Upright),
    (0x03047, 0x03047, VerticalOrientation::TransformedUpright),
    (0x03048, 0x03048, VerticalOrientation::Upright),
    (0x03049, 0x03049, VerticalOrientation::TransformedUpright),
    (0x0304A, 0x03062, VerticalOrientation::Upright),
    (0x03063, 0x03063, VerticalOrientation::TransformedUpright),
    (0x03064, 0x03082, VerticalOrientation::Upright),
    (0x03083, 0x03083, VerticalOrientation::TransformedUpright),
    (0x03084, 0x03084, VerticalOrientation::Upright),
    (0x03085, 0x03085, VerticalOrientation::TransformedUpright),
    (0x03086, 0x03086, VerticalOrientation::Upright),
    (0x03087, 0x03087, VerticalOrientation::TransformedUpright),
    (0x03088, 0x0308D, VerticalOrientation::Upright),
    (0x0308E, 0x0308E, VerticalOrientation::TransformedUpright),
    (0x0308F, 0x03094, VerticalOrientation::Upright),
    (0x03095, 0x03096, VerticalOrientation::TransformedUpright),
    (0x03097, 0x0309A, VerticalOrientation::Upright),
    (0x0309B, 0x0309C, VerticalOrientation::TransformedUpright),
    (0x0309D, 0x0309F, VerticalOrientation::Upright),
    (0x030A0, 0x030A0, VerticalOrientation::TransformedRotated),
    (0x030A1, 0x030A1, VerticalOrientation::TransformedUpright),
    (0x030A2, 0x030A2, VerticalOrientation::Upright),
    (0x030A3, 0x030A3, VerticalOrientation::TransformedUpright),
    (0x030A4, 0x030A4, VerticalOrientation::Upright),
    (0x030A5, 0x030A5, VerticalOrientation::TransformedUpright),
    (0x030A6, 0x030A6, VerticalOrientation::Upright),
    (0x030A7, 0x030A7, VerticalOrientation::TransformedUpright),
    (0x030A8, 0x030A8, VerticalOrientation::Upright),
    (0x030A9, 0x030A9, VerticalOrientation::TransformedUpright),
    (0x030AA, 0x030C2, VerticalOrientation::Upright),
    (0x030C3, 0x030C3, VerticalOrientation::TransformedUpright),
    (0x030C4, 0x030E2, VerticalOrientation::Upright),
    (0x030E3, 0x030E3, VerticalOrientation::TransformedUpright),
    (0x030E4, 0x030E4, VerticalOrientation::Upright),
    (0x030E5, 0x030E5, VerticalOrientation::TransformedUpright),
    (0x030E6, 0x030E6, VerticalOrientation::Upright),
    (0x030E7, 0x030E7, VerticalOrientation::TransformedUpright),
    (0x030E8, 0x030ED, VerticalOrientation::Upright),
    (0x030EE, 0x030EE, VerticalOrientation::TransformedUpright),
    (0x030EF, 0x030F4, VerticalOrientation::Upright),
    (0x030F5, 0x030F6, VerticalOrientation::TransformedUpright),
    (0x030F7, 0x030FB, VerticalOrientation::Upright),
    (0x030FC, 0x030FC, VerticalOrientation::TransformedRotated),
    (0x030FD, 0x03126, VerticalOrientation::Upright),
    (0x03127, 0x03127, VerticalOrientation::TransformedUpright),
    (0x03128, 0x031EF, VerticalOrientation::Upright),
    (0x031F0, 0x031FF, VerticalOrientation::TransformedUpright),
    (0x03200, 0x032FE, VerticalOrientation::Upright),
    (0x032FF, 0x03357, VerticalOrientation::TransformedUpright),
    (0x03358, 0x0337A, VerticalOrientation::Upright),
    (0x0337B, 0x0337F, VerticalOrientation::TransformedUpright),
    (0x03380, 0x0A4CF, VerticalOrientation::Upright),
    (0x0A960, 0x0A97F, VerticalOrientation::Upright),
    (0x0AC00, 0x0D7FF, VerticalOrientation::Upright),
    (0x0E000, 0x0FAFF, VerticalOrientation::Upright),
    (0x0FE10, 0x0FE1F, VerticalOrientation::Upright),
    (0x0FE30, 0x0FE48, VerticalOrientation::Upright),
    (0x0FE50, 0x0FE52, VerticalOrientation::TransformedUpright),
    (0x0FE53, 0x0FE57, VerticalOrientation::Upright),
    (0x0FE59, 0x0FE5E, VerticalOrientation::TransformedRotated),
    (0x0FE5F, 0x0FE62, VerticalOrientation::Upright),
    (0x0FE67, 0x0FE6F, VerticalOrientation::Upright),
    (0x0FF01, 0x0FF01, VerticalOrientation::TransformedUpright),
    (0x0FF02, 0x0FF07, VerticalOrientation::Upright),
    (0x0FF08, 0x0FF09, VerticalOrientation::TransformedRotated),
    (0x0FF0A, 0x0FF0B, VerticalOrientation::Upright),
    (0x0FF0C, 0x0FF0C, VerticalOrientation::TransformedUpright),
    (0x0FF0E, 0x0FF0E, VerticalOrientation::TransformedUpright),
    (0x0FF0F, 0x0FF19, VerticalOrientation::Upright),
    (0x0FF1A, 0x0FF1B, VerticalOrientation::TransformedRotated),
    (0x0FF1F, 0x0FF1F, VerticalOrientation::TransformedUpright),
    (0x0FF20, 0x0FF3A, VerticalOrientation::Upright),
    (0x0FF3B, 0x0FF3B, VerticalOrientation::TransformedRotated),
    (0x0FF3C, 0x0FF3C, VerticalOrientation::Upright),
    (0x0FF3D, 0x0FF3D, VerticalOrientation::TransformedRotated),
    (0x0FF3E, 0x0FF3E, VerticalOrientation::Upright),
    (0x0FF3F, 0x0FF3F, VerticalOrientation::TransformedRotated),
    (0x0FF40, 0x0FF5A, VerticalOrientation::Upright),
    (0x0FF5B, 0x0FF60, VerticalOrientation::TransformedRotated),
    (0x0FFE0, 0x0FFE2, VerticalOrientation::Upright),
    (0x0FFE3, 0x0FFE3, VerticalOrientation::TransformedRotated),
    (0x0FFE4, 0x0FFE7, VerticalOrientation::Upright),
    (0x0FFF0, 0x0FFF8, VerticalOrientation::Upright),
    (0x0FFFC, 0x0FFFD, VerticalOrientation::Upright),
    (0x10980, 0x1099F, VerticalOrientation::Upright),
    (0x11580, 0x115FF, VerticalOrientation::Upright),
    (0x11A00, 0x11ABF, VerticalOrientation::Upright),
    (0x13000, 0x1467F, VerticalOrientation::Upright),
    (0x16FE0, 0x18D7F, VerticalOrientation::Upright),
    (0x1AFF0, 0x1B2FF, VerticalOrientation::Upright),
    (0x1CF00, 0x1CFCF, VerticalOrientation::Upright),
    (0x1D000, 0x1D1FF, VerticalOrientation::Upright),
    (0x1D2E0, 0x1D37F, VerticalOrientation::Upright),
    (0x1D800, 0x1DAAF, VerticalOrientation::Upright),
    (0x1F000, 0x1F1FF, VerticalOrientation::Upright),
    (0x1F200, 0x1F201, VerticalOrientation::TransformedUpright),
    (0x1F202, 0x1F7FF, VerticalOrientation::Upright),
    (0x1F900, 0x1FAFF, VerticalOrientation::Upright),
    (0x20000, 0x2FFFD, VerticalOrientation::Upright),
    (0x30000, 0x3FFFD, VerticalOrientation::Upright),
    (0xF0000, 0xFFFFD, VerticalOrientation::Upright),
    (0x100000, 0x10FFFD, VerticalOrientation::Upright),
];

/// The UAX #50 orientation of `ch`.
pub fn vertical_orientation_of(ch: char) -> VerticalOrientation {
    let code = u32::from(ch);
    match VERTICAL_RANGES.binary_search_by(|&(low, high, _)| {
        if code < low {
            core::cmp::Ordering::Greater
        } else if code > high {
            core::cmp::Ordering::Less
        } else {
            core::cmp::Ordering::Equal
        }
    }) {
        Ok(index) => VERTICAL_RANGES[index].2,
        Err(_) => VerticalOrientation::Rotated,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_UPRIGHT, VERTICAL_ORIENTATION_UNICODE_VERSION, VERTICAL_RANGES,
        VerticalOrientation, vertical_orientation_of,
    };

    /// The checked-in UCD data file, byte-for-byte as `unicode.org` serves
    /// it (its own licence header included).
    const UCD: &str = include_str!("../../data/VerticalOrientation-16.0.0.txt");

    /// Re-derives the whole property exactly as the one-off generator did:
    /// default `R` everywhere, then the header's default-U blocks, then the
    /// explicit rows on top.
    fn derive_property() -> Vec<VerticalOrientation> {
        let mut table = vec![VerticalOrientation::Rotated; 0x11_0000];
        for &(low, high) in &DEFAULT_UPRIGHT {
            for code in low..=high {
                table[code as usize] = VerticalOrientation::Upright;
            }
        }
        for raw in UCD.lines() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut fields = line.split(';');
            let codes = fields.next().expect("a first field").trim();
            let value = fields.next().expect("a second field").trim();
            assert!(fields.next().is_none(), "exactly two fields: {raw:?}");
            let orientation = match value {
                "U" => VerticalOrientation::Upright,
                "R" => VerticalOrientation::Rotated,
                "Tu" => VerticalOrientation::TransformedUpright,
                "Tr" => VerticalOrientation::TransformedRotated,
                other => panic!("unknown Vertical_Orientation value {other:?}"),
            };
            let (low, high) = match codes.split_once("..") {
                Some((low, high)) => (
                    u32::from_str_radix(low, 16).expect("hex"),
                    u32::from_str_radix(high, 16).expect("hex"),
                ),
                None => {
                    let code = u32::from_str_radix(codes, 16).expect("hex");
                    (code, code)
                }
            };
            for code in low..=high {
                table[code as usize] = orientation;
            }
        }
        table
    }

    #[test]
    fn the_const_ranges_reproduce_the_ucd_property_over_the_whole_codespace() {
        // The verification-test-as-generator, at full width: every one of
        // 1 114 112 code points, not a sample.
        let expected = derive_property();
        for (code, &want) in expected.iter().enumerate() {
            let Some(ch) = char::from_u32(code as u32) else {
                continue;
            };
            let got = vertical_orientation_of(ch);
            assert_eq!(
                got, want,
                "U+{code:04X} is {want:?} in the UCD file but {got:?} in the table",
            );
        }
    }

    #[test]
    fn the_data_file_declares_the_recorded_unicode_version() {
        let expected = format!("# VerticalOrientation-{VERTICAL_ORIENTATION_UNICODE_VERSION}.txt");
        assert!(
            UCD.starts_with(&expected),
            "the checked-in file must be Unicode {VERTICAL_ORIENTATION_UNICODE_VERSION}",
        );
        assert!(
            UCD.contains("https://www.unicode.org/terms_of_use.html"),
            "the UCD licence reference must survive verbatim",
        );
        assert!(
            UCD.contains("@missing: 0000..10FFFF; R"),
            "the R default this table relies on is the file's own",
        );
    }

    #[test]
    fn the_ranges_are_sorted_disjoint_and_never_the_default() {
        for window in VERTICAL_RANGES.windows(2) {
            assert!(
                window[0].1 < window[1].0,
                "sorted and disjoint: {:?} then {:?}",
                window[0],
                window[1],
            );
        }
        for &(low, high, orientation) in &VERTICAL_RANGES {
            assert!(low <= high, "U+{low:04X}..U+{high:04X}");
            assert_ne!(
                orientation,
                VerticalOrientation::Rotated,
                "the R default is never stored",
            );
        }
    }

    #[test]
    fn the_measured_characters_classify_as_the_design_recorded_them() {
        // The spot checks the vertical-title refusal ladder rests on.
        assert_eq!(
            vertical_orientation_of('\u{3042}'),
            VerticalOrientation::Upright,
            "hiragana"
        );
        assert_eq!(
            vertical_orientation_of('\u{4E9C}'),
            VerticalOrientation::Upright,
            "kanji"
        );
        assert_eq!(
            vertical_orientation_of('\u{3001}'),
            VerticalOrientation::TransformedUpright,
            "ideographic comma",
        );
        assert_eq!(
            vertical_orientation_of('\u{300C}'),
            VerticalOrientation::TransformedRotated,
            "corner bracket",
        );
        assert_eq!(
            vertical_orientation_of('\u{30FC}'),
            VerticalOrientation::TransformedRotated,
            "the prolonged sound mark",
        );
        // Everything the ladder refuses.
        for ch in ['A', '0', ' ', '\u{FF71}', '\u{2018}', '\u{2026}'] {
            assert_eq!(
                vertical_orientation_of(ch),
                VerticalOrientation::Rotated,
                "{ch:?} must refuse a vertical line",
            );
            assert!(!vertical_orientation_of(ch).draws_upright());
        }
        assert!(vertical_orientation_of('\u{3042}').draws_upright());
        assert!(vertical_orientation_of('\u{3001}').draws_upright());
        assert!(vertical_orientation_of('\u{300C}').draws_upright());
    }

    #[test]
    fn draws_upright_accepts_the_cjk_repertoire_and_refuses_the_rotated_one() {
        // 東 서 ー are the three the ladders are argued over: a kanji, a
        // Hangul syllable and the prolonged sound mark (`Tr`, which a face
        // with `vert` lookups substitutes and one without draws as-is).
        for ch in ['\u{3042}', '\u{6771}', '\u{C11C}', '\u{3001}', '\u{3002}'] {
            assert!(
                vertical_orientation_of(ch).draws_upright(),
                "{ch:?} must be settable upright",
            );
        }
        for ch in ['\u{300C}', '\u{300D}', '\u{30FC}'] {
            assert_eq!(
                vertical_orientation_of(ch),
                VerticalOrientation::TransformedRotated,
                "{ch:?} is Tr",
            );
            assert!(vertical_orientation_of(ch).draws_upright(), "and accepted");
        }
        for ch in ['A', '0', ' ', '\u{FF71}', '\u{2026}', '\u{645}'] {
            assert!(
                !vertical_orientation_of(ch).draws_upright(),
                "{ch:?} must refuse",
            );
        }
    }

    /// Every character in the exporter's **five listed complex-LTR ranges** is
    /// `R`, so for those ranges the upright-only first rung refuses the label
    /// before a complex script could reach swash's `EngineMode::Complex` path.
    ///
    /// Read this as a claim about those five ranges and nothing wider
    /// (narrowed 2026-08-11). It is NOT the general statement "every Brahmic
    /// character is `R`" — that is false, and
    /// [`upright_brahmic_blocks_are_not_caught_by_the_rotation_rung`] below is
    /// the counter-example. `vertical.rs`'s panic-safety argument no longer
    /// rests on this test; it rests on `vertical_script` being unable to return
    /// a complex tag at all.
    #[test]
    fn every_complex_ltr_character_the_ladder_meets_is_rotated() {
        let complex = [
            (0x0900_u32, 0x0FFF_u32), // Devanagari … Sinhala, Thai, Lao, Tibetan
            (0x1000, 0x109F),         // Myanmar
            (0x1780, 0x17FF),         // Khmer
            (0x1CD0, 0x1CFF),         // Vedic Extensions
            (0xA8E0, 0xA8FF),         // Devanagari Extended
        ];
        for (low, high) in complex {
            for code in low..=high {
                let Some(ch) = char::from_u32(code) else {
                    continue;
                };
                assert_eq!(
                    vertical_orientation_of(ch),
                    VerticalOrientation::Rotated,
                    "U+{code:04X} must be R so the ladder refuses before shaping",
                );
            }
        }
    }

    /// The counter-example that killed a false invariant (recorded
    /// 2026-08-11). `vertical.rs`'s panic-safety paragraph used to argue that
    /// "every Brahmic / South-East-Asian character is `R` in UAX #50", so
    /// `VerticalRefusal::RotatedCharacter` would always refuse before a complex
    /// script reached the shaper. Siddham, Zanabazar Square and Soyombo are all
    /// three Brahmic AND `Upright`, so that rung does **not** catch them and
    /// the argument was unsound. This test pins the counter-example so nobody
    /// re-derives the invariant from the narrower five-range test above.
    ///
    /// Nothing is broken by this: the vertical path is safe for the different
    /// reason `vertical.rs` now states, that `vertical_script` can only ever
    /// return `hang`/`kana`/`hani`.
    #[test]
    fn upright_brahmic_blocks_are_not_caught_by_the_rotation_rung() {
        let upright_brahmic = [
            (0x1_1580_u32, 0x1_15FF_u32, "Siddham"),
            (0x1_1A00, 0x1_1A4F, "Zanabazar Square"),
            (0x1_1A50, 0x1_1AAF, "Soyombo"),
        ];
        for (low, high, name) in upright_brahmic {
            let mut upright = 0_usize;
            for code in low..=high {
                let Some(ch) = char::from_u32(code) else {
                    continue;
                };
                if vertical_orientation_of(ch).draws_upright() {
                    upright += 1;
                }
            }
            assert!(
                upright > 0,
                "{name} must contain Upright characters — it is the whole \
                 counter-example to the old Brahmic invariant",
            );
        }
    }

    /// Meroitic Hieroglyphs are `Upright` AND right-to-left, so the
    /// orientation table alone does not catch them: the RTL rung is load
    /// bearing, not dead code.
    #[test]
    fn meroitic_is_upright_so_the_rtl_rung_is_not_dead_code() {
        for code in 0x1_0980_u32..=0x1_099F {
            let Some(ch) = char::from_u32(code) else {
                continue;
            };
            assert!(
                vertical_orientation_of(ch).draws_upright(),
                "U+{code:04X} is Upright",
            );
        }
    }
}
