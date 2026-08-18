// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Just enough CFF (Compact Font Format 1) to answer ONE question the PDF
//! export cannot guess: **is this program CID-keyed, and if so which CID does
//! each glyph carry?**
//!
//! # Why the export has to ask
//!
//! `oxifont-subset` refuses to rewrite a CID-keyed CFF (an FDSelect makes it
//! bail) and hands back the ORIGINAL table verbatim, so `subset::subset_face`
//! embeds the whole program under the original glyph numbering. For a
//! name-keyed CFF that is complete: PDF 32000-1 §9.7.4.2 says a CIDFontType0
//! whose program is **not** CID-keyed uses the CID directly as the glyph
//! index. For a CID-keyed one it is not: the viewer resolves the CID through
//! the program's **charset**, which maps glyph id → CID. Assuming identity
//! there prints the wrong charstrings on any face whose charset is not the
//! identity — an Adobe-Japan1 build being the obvious class — and every
//! Noto Sans CJK / Source Han Sans / Hiragino build is exactly the CID-keyed
//! case the desktop font scan ranks first.
//!
//! # Scope
//!
//! [`charset`] is the whole surface: the Top DICT is read only for the ROS
//! operator (12 30, which is what *makes* a program CID-keyed), the charset
//! offset (15) and the CharStrings offset (17, for the glyph count). Nothing
//! here parses charstrings, private dicts, subrs or FDSelect, and nothing
//! here rewrites anything.
//!
//! # Total functions over hostile bytes
//!
//! Every read is bounds-checked through `get`, every arithmetic step that
//! could leave `usize` is checked, and the answer to anything malformed is
//! [`None`] — the caller then refuses the face rather than embedding a
//! program it cannot describe. A font file is untrusted input.

/// The CID assignment of one CFF program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Charset {
    /// The program is name-keyed (no ROS): PDF uses the CID as the glyph
    /// index directly, so there is nothing to map.
    NameKeyed,
    /// The program is CID-keyed: `cids[gid]` is the CID the viewer resolves
    /// glyph `gid` through. Always `nGlyphs` long and always starts at CID 0
    /// (`.notdef`).
    CidKeyed(Vec<u16>),
}

/// The first charset offset that is a real table rather than one of the three
/// predefined charsets (ISOAdobe, Expert, ExpertSubset). A CID-keyed program
/// may not use a predefined charset — they name glyphs, not CIDs — so the
/// caller treats one as malformed.
const FIRST_CUSTOM_CHARSET: u32 = 3;

/// Reads a CFF table's charset, or [`None`] when the bytes do not describe
/// one this export can trust.
///
/// `table` is a **bare** `CFF ` table, exactly what `/FontFile3` carries.
pub(super) fn charset(table: &[u8]) -> Option<Charset> {
    let header_size = usize::from(*table.get(2)?);
    if header_size < 4 {
        return None;
    }
    // Name INDEX, then Top DICT INDEX. Only the first Top DICT is read: a
    // `CFF ` table inside an OpenType font holds exactly one font.
    let after_name = header_size.checked_add(index_size(table.get(header_size..)?)?)?;
    let top_dicts = table.get(after_name..)?;
    let top_dict = index_entry(top_dicts, 0)?;

    let mut ros = false;
    let mut charset_offset: Option<u32> = None;
    let mut charstrings_offset: Option<u32> = None;
    for (operator, operands) in DictOperators::new(top_dict) {
        match operator {
            Operator::Ros => ros = true,
            Operator::Charset => charset_offset = operands.last().and_then(non_negative),
            Operator::CharStrings => charstrings_offset = operands.last().and_then(non_negative),
        }
    }
    if !ros {
        return Some(Charset::NameKeyed);
    }
    let charstrings = usize::try_from(charstrings_offset?).ok()?;
    let glyphs = index_count(table.get(charstrings..)?)?;
    if glyphs == 0 || glyphs > usize::from(u16::MAX) + 1 {
        return None;
    }
    let offset = charset_offset?;
    if offset < FIRST_CUSTOM_CHARSET {
        // A predefined charset in a CID-keyed program: the file contradicts
        // itself, and guessing would print the wrong glyphs.
        return None;
    }
    let at = usize::try_from(offset).ok()?;
    // A charset table carries no length of its own — it ends where the next
    // structure begins. Every real font lays it out before the CharStrings
    // INDEX, so that offset is an upper bound, and holding the read to it is
    // what turns "the charset names fewer glyphs than the font has" from a
    // silent read of the neighbouring table into a refusal.
    let limit = if at < charstrings {
        charstrings
    } else {
        table.len()
    };
    let cids = read_charset(table.get(at..limit)?, glyphs)?;
    Some(Charset::CidKeyed(cids))
}

/// A DICT operand that has to be a non-negative offset.
fn non_negative(&operand: &i32) -> Option<u32> {
    u32::try_from(operand).ok()
}

/// Reads the charset table itself: `glyphs` entries, `cids[0] == 0` by
/// definition (`.notdef` is never listed).
///
/// Formats 0, 1 and 2 are the whole repertoire CFF 1 defines. A table that
/// runs out of bytes before it has named every glyph is malformed — the CID
/// of the glyphs past the end would be a guess — so it answers [`None`].
fn read_charset(data: &[u8], glyphs: usize) -> Option<Vec<u16>> {
    let format = *data.first()?;
    let body = data.get(1..)?;
    let mut cids = Vec::with_capacity(glyphs);
    cids.push(0_u16);
    match format {
        0 => {
            for index in 0..glyphs.checked_sub(1)? {
                let at = index.checked_mul(2)?;
                let bytes = body.get(at..at.checked_add(2)?)?;
                cids.push(u16::from_be_bytes([bytes[0], bytes[1]]));
            }
        }
        // Ranges of consecutive CIDs: `first` plus `n_left` more. The two
        // formats differ only in the width of `n_left`.
        1 | 2 => {
            let width = if format == 1 { 1 } else { 2 };
            let stride = 2 + width;
            let mut at = 0_usize;
            while cids.len() < glyphs {
                let entry = body.get(at..at.checked_add(stride)?)?;
                let first = u32::from(u16::from_be_bytes([entry[0], entry[1]]));
                let left = if format == 1 {
                    u32::from(entry[2])
                } else {
                    u32::from(u16::from_be_bytes([entry[2], entry[3]]))
                };
                for step in 0..=left {
                    if cids.len() >= glyphs {
                        break;
                    }
                    cids.push(u16::try_from(first.checked_add(step)?).ok()?);
                }
                at = at.checked_add(stride)?;
            }
        }
        _ => return None,
    }
    (cids.len() == glyphs).then_some(cids)
}

/// The entry count of a CFF INDEX.
fn index_count(data: &[u8]) -> Option<usize> {
    let count = data.get(0..2)?;
    Some(usize::from(u16::from_be_bytes([count[0], count[1]])))
}

/// The total byte length of a CFF INDEX, header included.
///
/// An empty INDEX is two bytes and nothing else — the one case where the
/// offset-size byte is absent.
fn index_size(data: &[u8]) -> Option<usize> {
    let count = index_count(data)?;
    if count == 0 {
        return Some(2);
    }
    let offset_size = usize::from(*data.get(2)?);
    if !(1..=4).contains(&offset_size) {
        return None;
    }
    let offsets_len = count.checked_add(1)?.checked_mul(offset_size)?;
    let last = read_offset(data.get(3..)?, count, offset_size)?;
    // Offsets are 1-based from the byte before the data block.
    3usize
        .checked_add(offsets_len)?
        .checked_add(last.checked_sub(1)?)
}

/// One entry of a CFF INDEX, by position.
fn index_entry(data: &[u8], entry: usize) -> Option<&[u8]> {
    let count = index_count(data)?;
    if entry >= count {
        return None;
    }
    let offset_size = usize::from(*data.get(2)?);
    if !(1..=4).contains(&offset_size) {
        return None;
    }
    let offsets = data.get(3..)?;
    let start = read_offset(offsets, entry, offset_size)?;
    let end = read_offset(offsets, entry + 1, offset_size)?;
    if end < start {
        return None;
    }
    let base = 3usize
        .checked_add(count.checked_add(1)?.checked_mul(offset_size)?)?
        .checked_sub(1)?;
    data.get(base.checked_add(start)?..base.checked_add(end)?)
}

/// One big-endian offset of `offset_size` bytes out of an INDEX's offset
/// array.
fn read_offset(offsets: &[u8], index: usize, offset_size: usize) -> Option<usize> {
    let at = index.checked_mul(offset_size)?;
    let bytes = offsets.get(at..at.checked_add(offset_size)?)?;
    Some(
        bytes
            .iter()
            .fold(0_usize, |value, &byte| (value << 8) | usize::from(byte)),
    )
}

/// The three Top DICT operators this module reads; every other one is
/// skipped with its operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    /// `12 30` — Registry/Ordering/Supplement. Its PRESENCE is what makes a
    /// program CID-keyed.
    Ros,
    /// `15` — charset offset.
    Charset,
    /// `17` — CharStrings INDEX offset, read for the glyph count.
    CharStrings,
}

/// Walks a DICT, yielding the operators above with their operand stack.
struct DictOperators<'a> {
    data: &'a [u8],
    at: usize,
    operands: Vec<i32>,
}

impl<'a> DictOperators<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            at: 0,
            operands: Vec::new(),
        }
    }
}

impl Iterator for DictOperators<'_> {
    type Item = (Operator, Vec<i32>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(&byte) = self.data.get(self.at) {
            self.at += 1;
            match byte {
                // Operators. 12 escapes into the two-byte space.
                12 => {
                    let second = *self.data.get(self.at)?;
                    self.at += 1;
                    let operands = std::mem::take(&mut self.operands);
                    if second == 30 {
                        return Some((Operator::Ros, operands));
                    }
                }
                0..=21 => {
                    let operands = std::mem::take(&mut self.operands);
                    match byte {
                        15 => return Some((Operator::Charset, operands)),
                        17 => return Some((Operator::CharStrings, operands)),
                        _ => {}
                    }
                }
                // Operands.
                28 => {
                    let bytes = self.data.get(self.at..self.at + 2)?;
                    self.operands
                        .push(i32::from(i16::from_be_bytes([bytes[0], bytes[1]])));
                    self.at += 2;
                }
                29 => {
                    let bytes = self.data.get(self.at..self.at + 4)?;
                    self.operands
                        .push(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
                    self.at += 4;
                }
                // Real number: nibbles until the 0xf terminator. The export
                // never reads a real operand, so it is skipped, not decoded.
                30 => {
                    loop {
                        let byte = *self.data.get(self.at)?;
                        self.at += 1;
                        if byte & 0x0f == 0x0f || byte >> 4 == 0x0f {
                            break;
                        }
                    }
                    self.operands.push(0);
                }
                32..=246 => self.operands.push(i32::from(byte) - 139),
                247..=250 => {
                    let low = i32::from(*self.data.get(self.at)?);
                    self.at += 1;
                    self.operands
                        .push((i32::from(byte) - 247) * 256 + low + 108);
                }
                251..=254 => {
                    let low = i32::from(*self.data.get(self.at)?);
                    self.at += 1;
                    self.operands
                        .push(-(i32::from(byte) - 251) * 256 - low - 108);
                }
                // 22..=27, 31 and 255 are reserved: a DICT carrying one is
                // not something this parser can keep its place in.
                _ => return None,
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A CFF INDEX with one-byte offsets over `entries`.
    fn index(entries: &[&[u8]]) -> Vec<u8> {
        let mut out = vec![0, 0];
        let count = u16::try_from(entries.len()).expect("a small fixture");
        out[0..2].copy_from_slice(&count.to_be_bytes());
        if entries.is_empty() {
            return out;
        }
        out.push(1);
        let mut offset = 1_u8;
        out.push(offset);
        for entry in entries {
            offset += u8::try_from(entry.len()).expect("a small fixture");
            out.push(offset);
        }
        for entry in entries {
            out.extend_from_slice(entry);
        }
        out
    }

    /// A DICT operand encoded in the 5-byte long-integer form, so the
    /// fixtures can name real offsets without arithmetic games.
    fn operand(value: i32) -> Vec<u8> {
        let mut out = vec![29];
        out.extend_from_slice(&value.to_be_bytes());
        out
    }

    /// A whole `CFF ` table: header, one-entry Name INDEX, one Top DICT,
    /// then `charset` and a CharStrings INDEX of `glyphs` empty entries at
    /// the offsets the DICT names.
    fn table(cid_keyed: bool, charset_bytes: &[u8], glyphs: usize) -> Vec<u8> {
        let header = vec![1_u8, 0, 4, 1];
        let names = index(&[b"Fixture"]);
        // The Top DICT's own length depends on the offsets it carries, so
        // build it once with placeholders to measure, then again for real.
        let build = |charset_at: i32, charstrings_at: i32| {
            let mut dict = Vec::new();
            if cid_keyed {
                // ROS: three operands (registry SID, ordering SID, supplement).
                dict.extend(operand(391));
                dict.extend(operand(392));
                dict.extend(operand(0));
                dict.extend_from_slice(&[12, 30]);
            }
            dict.extend(operand(charset_at));
            dict.push(15);
            dict.extend(operand(charstrings_at));
            dict.push(17);
            dict
        };
        let measured = index(&[&build(0, 0)]);
        let charset_at = header.len() + names.len() + measured.len();
        let charstrings_at = charset_at + charset_bytes.len();
        let dicts = index(&[&build(
            i32::try_from(charset_at).expect("a small fixture"),
            i32::try_from(charstrings_at).expect("a small fixture"),
        )]);
        assert_eq!(dicts.len(), measured.len(), "the fixture stays measurable");
        let charstrings = index(&vec![b"\x0e".as_slice(); glyphs]);
        let mut out = header;
        out.extend(names);
        out.extend(dicts);
        out.extend_from_slice(charset_bytes);
        out.extend(charstrings);
        out
    }

    #[test]
    fn a_name_keyed_program_reports_no_cid_mapping() {
        // Identity is right for a name-keyed program, so the caller must be
        // told "nothing to map" rather than handed a table.
        let bytes = table(false, &[0, 0, 1, 0, 2], 3);
        assert_eq!(charset(&bytes), Some(Charset::NameKeyed));
    }

    #[test]
    fn format_0_lists_one_cid_per_glyph_after_notdef() {
        // Four glyphs: .notdef plus three listed CIDs, deliberately NOT the
        // identity — the whole point of reading the charset.
        let mut charset_bytes = vec![0_u8];
        for cid in [1200_u16, 42, 65535] {
            charset_bytes.extend_from_slice(&cid.to_be_bytes());
        }
        let bytes = table(true, &charset_bytes, 4);
        assert_eq!(
            charset(&bytes),
            Some(Charset::CidKeyed(vec![0, 1200, 42, 65535])),
        );
    }

    #[test]
    fn format_1_expands_its_ranges() {
        // Two ranges: CIDs 10..=12 then 200..=201, over six glyphs.
        let charset_bytes = [1_u8, 0, 10, 2, 0, 200, 1];
        let bytes = table(true, &charset_bytes, 6);
        assert_eq!(
            charset(&bytes),
            Some(Charset::CidKeyed(vec![0, 10, 11, 12, 200, 201])),
        );
    }

    #[test]
    fn format_2_expands_its_wide_ranges() {
        // One range of 300 consecutive CIDs from 1000, truncated by the
        // glyph count — the common shape of a real CJK charset.
        let charset_bytes = [2_u8, 0x03, 0xE8, 0x01, 0x2B];
        let bytes = table(true, &charset_bytes, 5);
        assert_eq!(
            charset(&bytes),
            Some(Charset::CidKeyed(vec![0, 1000, 1001, 1002, 1003])),
        );
    }

    #[test]
    fn a_truncated_or_predefined_charset_is_refused_rather_than_guessed() {
        // Format 0 with fewer SIDs than glyphs: the tail would be a guess.
        let bytes = table(true, &[0_u8, 0, 5], 4);
        assert_eq!(charset(&bytes), None);
        // A predefined charset (offset 0..2) in a CID-keyed program.
        let mut predefined = table(true, &[], 3);
        // Rewrite the charset operand to 0 — the first long-int operand
        // after the ROS block.
        let at = predefined
            .windows(2)
            .position(|pair| pair == [15, 29])
            .expect("the charset operator precedes the CharStrings operand");
        predefined[at - 4..at].copy_from_slice(&[0, 0, 0, 0]);
        assert_eq!(charset(&predefined), None);
    }

    /// The shipping case, on a real pan-CJK face: Hiragino (macOS), Noto
    /// Sans CJK / Source Han Sans (Linux) are all CID-keyed CFF, and they are
    /// exactly what the desktop font scan ranks FIRST for CJK — so this
    /// module's answer for them is what a Japanese page's glyph selection
    /// rests on.
    #[test]
    #[ignore = "reads the platform's CJK fonts; the print-v1.6 CID-keyed golden"]
    fn live_cid_keyed_face_golden() {
        use ttf_parser::{RawFace, Tag};

        let candidates = [
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/source-han-sans/SourceHanSans.ttc",
        ];
        let mut checked = 0_usize;
        for path in candidates {
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            let Ok(raw) = RawFace::parse(&bytes, 0) else {
                continue;
            };
            let Some(table) = raw.table(Tag::from_bytes(b"CFF ")) else {
                continue;
            };
            let Some(Charset::CidKeyed(cids)) = charset(table) else {
                panic!("{path} is a CID-keyed CFF and must parse as one");
            };
            let face = ttf_parser::Face::parse(&bytes, 0).expect("a parseable face");
            assert_eq!(
                cids.len(),
                usize::from(face.number_of_glyphs()),
                "{path}: one CID per glyph",
            );
            assert_eq!(cids[0], 0, "{path}: .notdef is CID 0");
            // The identity assumption this module exists to replace: if it
            // held, the charset would be redundant. Recorded rather than
            // asserted either way, because a CID-keyed build with an identity
            // charset is legal — and correct under both codes.
            let identity = cids
                .iter()
                .enumerate()
                .all(|(gid, &cid)| u16::try_from(gid).is_ok_and(|gid| gid == cid));
            println!(
                "{path}: {} glyphs, identity charset: {identity}",
                cids.len()
            );
            checked += 1;
        }
        assert!(checked > 0, "no platform CID-keyed CFF face was found");
    }

    #[test]
    fn hostile_bytes_answer_none_instead_of_panicking() {
        assert_eq!(charset(&[]), None);
        assert_eq!(charset(&[1, 0, 4]), None);
        let full = table(true, &[0_u8, 0, 5, 0, 6], 3);
        for cut in 0..full.len() {
            // Every truncation of a real table: no panic, no index out of
            // bounds, just an answer.
            let _ = charset(&full[..cut]);
        }
        for byte in 0..=255_u8 {
            let mut noisy = full.clone();
            if let Some(slot) = noisy.get_mut(6) {
                *slot = byte;
            }
            let _ = charset(&noisy);
        }
    }
}
