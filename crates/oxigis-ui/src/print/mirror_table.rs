// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The COMPLETE Unicode 16.0.0 `Bidi_Mirroring_Glyph` table (print v1.4
//! item 2, D-M1) — generated data, not hand-written.
//!
//! # Why the table is checked in and verified rather than built
//!
//! The v1.3 hand-curated table covered 32 of the 428 mirrored codepoints, so
//! 396 brackets rendered the wrong way round inside an RTL line. Growing it
//! by hand is exactly the mistake this module removes: [`MIRROR_PAIRS`] was
//! generated ONCE from the checked-in UCD file, and the test at the bottom
//! re-parses that same file through `include_str!` and asserts equality — a
//! hand edit, a stale table or a Unicode bump all fail the suite. No
//! `build.rs`, no tool crate, no network, no new dependency.
//!
//! Deliberately NOT the `unicode-bidi-mirroring` crate at runtime: its
//! reverse lookup binary-searches an unsorted column and silently misses 21
//! codepoints (measured: `get_mirrored('\u{2265}') == None`). Here BOTH
//! directions are materialised as rows, so the lookup is one
//! `binary_search_by_key` over a sorted, verified involution.
//!
//! # The load-bearing property
//!
//! Every pair has EQUAL `len_utf8` (histogram: 8 one-byte, 2 two-byte, 418
//! three-byte codepoints; maximum `U+FF63`). That is what lets [`super::bidi`]
//! mirror the DISPLAY string in place while swash's cluster byte offsets keep
//! indexing the logical `source` identically, with no offset map. The test
//! `every_pair_has_equal_utf8_length` is the gate; if a future Unicode version
//! ever breaks it, the in-place mirroring premise must be redesigned before
//! the table is regenerated.

/// The Unicode version [`MIRROR_PAIRS`] was generated from.
///
/// The verification test's oracle for the data file's own repertoire line,
/// hence `cfg(test)`: the table is a plain sorted array at runtime and the
/// version has no shipped consumer — recording it in a place the suite
/// checks is what keeps the two from drifting.
#[cfg(test)]
pub(super) const UNICODE_VERSION: &str = "16.0.0";

/// Every mirrored codepoint and its `Bidi_Mirroring_Glyph`, **sorted by the
/// first element** so [`super::bidi::mirror`] is one `binary_search_by_key`.
///
/// Both directions are present as separate rows (the UCD file already lists
/// them, and the test proves the symmetric closure adds nothing), so the
/// table is a total involution over its own keys.
#[rustfmt::skip]
pub(super) const MIRROR_PAIRS: [(char, char); 428] = [
    ('\u{0028}', '\u{0029}'), ('\u{0029}', '\u{0028}'), ('\u{003C}', '\u{003E}'),
    ('\u{003E}', '\u{003C}'), ('\u{005B}', '\u{005D}'), ('\u{005D}', '\u{005B}'),
    ('\u{007B}', '\u{007D}'), ('\u{007D}', '\u{007B}'), ('\u{00AB}', '\u{00BB}'),
    ('\u{00BB}', '\u{00AB}'), ('\u{0F3A}', '\u{0F3B}'), ('\u{0F3B}', '\u{0F3A}'),
    ('\u{0F3C}', '\u{0F3D}'), ('\u{0F3D}', '\u{0F3C}'), ('\u{169B}', '\u{169C}'),
    ('\u{169C}', '\u{169B}'), ('\u{2039}', '\u{203A}'), ('\u{203A}', '\u{2039}'),
    ('\u{2045}', '\u{2046}'), ('\u{2046}', '\u{2045}'), ('\u{207D}', '\u{207E}'),
    ('\u{207E}', '\u{207D}'), ('\u{208D}', '\u{208E}'), ('\u{208E}', '\u{208D}'),
    ('\u{2208}', '\u{220B}'), ('\u{2209}', '\u{220C}'), ('\u{220A}', '\u{220D}'),
    ('\u{220B}', '\u{2208}'), ('\u{220C}', '\u{2209}'), ('\u{220D}', '\u{220A}'),
    ('\u{2215}', '\u{29F5}'), ('\u{221F}', '\u{2BFE}'), ('\u{2220}', '\u{29A3}'),
    ('\u{2221}', '\u{299B}'), ('\u{2222}', '\u{29A0}'), ('\u{2224}', '\u{2AEE}'),
    ('\u{223C}', '\u{223D}'), ('\u{223D}', '\u{223C}'), ('\u{2243}', '\u{22CD}'),
    ('\u{2245}', '\u{224C}'), ('\u{224C}', '\u{2245}'), ('\u{2252}', '\u{2253}'),
    ('\u{2253}', '\u{2252}'), ('\u{2254}', '\u{2255}'), ('\u{2255}', '\u{2254}'),
    ('\u{2264}', '\u{2265}'), ('\u{2265}', '\u{2264}'), ('\u{2266}', '\u{2267}'),
    ('\u{2267}', '\u{2266}'), ('\u{2268}', '\u{2269}'), ('\u{2269}', '\u{2268}'),
    ('\u{226A}', '\u{226B}'), ('\u{226B}', '\u{226A}'), ('\u{226E}', '\u{226F}'),
    ('\u{226F}', '\u{226E}'), ('\u{2270}', '\u{2271}'), ('\u{2271}', '\u{2270}'),
    ('\u{2272}', '\u{2273}'), ('\u{2273}', '\u{2272}'), ('\u{2274}', '\u{2275}'),
    ('\u{2275}', '\u{2274}'), ('\u{2276}', '\u{2277}'), ('\u{2277}', '\u{2276}'),
    ('\u{2278}', '\u{2279}'), ('\u{2279}', '\u{2278}'), ('\u{227A}', '\u{227B}'),
    ('\u{227B}', '\u{227A}'), ('\u{227C}', '\u{227D}'), ('\u{227D}', '\u{227C}'),
    ('\u{227E}', '\u{227F}'), ('\u{227F}', '\u{227E}'), ('\u{2280}', '\u{2281}'),
    ('\u{2281}', '\u{2280}'), ('\u{2282}', '\u{2283}'), ('\u{2283}', '\u{2282}'),
    ('\u{2284}', '\u{2285}'), ('\u{2285}', '\u{2284}'), ('\u{2286}', '\u{2287}'),
    ('\u{2287}', '\u{2286}'), ('\u{2288}', '\u{2289}'), ('\u{2289}', '\u{2288}'),
    ('\u{228A}', '\u{228B}'), ('\u{228B}', '\u{228A}'), ('\u{228F}', '\u{2290}'),
    ('\u{2290}', '\u{228F}'), ('\u{2291}', '\u{2292}'), ('\u{2292}', '\u{2291}'),
    ('\u{2298}', '\u{29B8}'), ('\u{22A2}', '\u{22A3}'), ('\u{22A3}', '\u{22A2}'),
    ('\u{22A6}', '\u{2ADE}'), ('\u{22A8}', '\u{2AE4}'), ('\u{22A9}', '\u{2AE3}'),
    ('\u{22AB}', '\u{2AE5}'), ('\u{22B0}', '\u{22B1}'), ('\u{22B1}', '\u{22B0}'),
    ('\u{22B2}', '\u{22B3}'), ('\u{22B3}', '\u{22B2}'), ('\u{22B4}', '\u{22B5}'),
    ('\u{22B5}', '\u{22B4}'), ('\u{22B6}', '\u{22B7}'), ('\u{22B7}', '\u{22B6}'),
    ('\u{22B8}', '\u{27DC}'), ('\u{22C9}', '\u{22CA}'), ('\u{22CA}', '\u{22C9}'),
    ('\u{22CB}', '\u{22CC}'), ('\u{22CC}', '\u{22CB}'), ('\u{22CD}', '\u{2243}'),
    ('\u{22D0}', '\u{22D1}'), ('\u{22D1}', '\u{22D0}'), ('\u{22D6}', '\u{22D7}'),
    ('\u{22D7}', '\u{22D6}'), ('\u{22D8}', '\u{22D9}'), ('\u{22D9}', '\u{22D8}'),
    ('\u{22DA}', '\u{22DB}'), ('\u{22DB}', '\u{22DA}'), ('\u{22DC}', '\u{22DD}'),
    ('\u{22DD}', '\u{22DC}'), ('\u{22DE}', '\u{22DF}'), ('\u{22DF}', '\u{22DE}'),
    ('\u{22E0}', '\u{22E1}'), ('\u{22E1}', '\u{22E0}'), ('\u{22E2}', '\u{22E3}'),
    ('\u{22E3}', '\u{22E2}'), ('\u{22E4}', '\u{22E5}'), ('\u{22E5}', '\u{22E4}'),
    ('\u{22E6}', '\u{22E7}'), ('\u{22E7}', '\u{22E6}'), ('\u{22E8}', '\u{22E9}'),
    ('\u{22E9}', '\u{22E8}'), ('\u{22EA}', '\u{22EB}'), ('\u{22EB}', '\u{22EA}'),
    ('\u{22EC}', '\u{22ED}'), ('\u{22ED}', '\u{22EC}'), ('\u{22F0}', '\u{22F1}'),
    ('\u{22F1}', '\u{22F0}'), ('\u{22F2}', '\u{22FA}'), ('\u{22F3}', '\u{22FB}'),
    ('\u{22F4}', '\u{22FC}'), ('\u{22F6}', '\u{22FD}'), ('\u{22F7}', '\u{22FE}'),
    ('\u{22FA}', '\u{22F2}'), ('\u{22FB}', '\u{22F3}'), ('\u{22FC}', '\u{22F4}'),
    ('\u{22FD}', '\u{22F6}'), ('\u{22FE}', '\u{22F7}'), ('\u{2308}', '\u{2309}'),
    ('\u{2309}', '\u{2308}'), ('\u{230A}', '\u{230B}'), ('\u{230B}', '\u{230A}'),
    ('\u{2329}', '\u{232A}'), ('\u{232A}', '\u{2329}'), ('\u{2768}', '\u{2769}'),
    ('\u{2769}', '\u{2768}'), ('\u{276A}', '\u{276B}'), ('\u{276B}', '\u{276A}'),
    ('\u{276C}', '\u{276D}'), ('\u{276D}', '\u{276C}'), ('\u{276E}', '\u{276F}'),
    ('\u{276F}', '\u{276E}'), ('\u{2770}', '\u{2771}'), ('\u{2771}', '\u{2770}'),
    ('\u{2772}', '\u{2773}'), ('\u{2773}', '\u{2772}'), ('\u{2774}', '\u{2775}'),
    ('\u{2775}', '\u{2774}'), ('\u{27C3}', '\u{27C4}'), ('\u{27C4}', '\u{27C3}'),
    ('\u{27C5}', '\u{27C6}'), ('\u{27C6}', '\u{27C5}'), ('\u{27C8}', '\u{27C9}'),
    ('\u{27C9}', '\u{27C8}'), ('\u{27CB}', '\u{27CD}'), ('\u{27CD}', '\u{27CB}'),
    ('\u{27D5}', '\u{27D6}'), ('\u{27D6}', '\u{27D5}'), ('\u{27DC}', '\u{22B8}'),
    ('\u{27DD}', '\u{27DE}'), ('\u{27DE}', '\u{27DD}'), ('\u{27E2}', '\u{27E3}'),
    ('\u{27E3}', '\u{27E2}'), ('\u{27E4}', '\u{27E5}'), ('\u{27E5}', '\u{27E4}'),
    ('\u{27E6}', '\u{27E7}'), ('\u{27E7}', '\u{27E6}'), ('\u{27E8}', '\u{27E9}'),
    ('\u{27E9}', '\u{27E8}'), ('\u{27EA}', '\u{27EB}'), ('\u{27EB}', '\u{27EA}'),
    ('\u{27EC}', '\u{27ED}'), ('\u{27ED}', '\u{27EC}'), ('\u{27EE}', '\u{27EF}'),
    ('\u{27EF}', '\u{27EE}'), ('\u{2983}', '\u{2984}'), ('\u{2984}', '\u{2983}'),
    ('\u{2985}', '\u{2986}'), ('\u{2986}', '\u{2985}'), ('\u{2987}', '\u{2988}'),
    ('\u{2988}', '\u{2987}'), ('\u{2989}', '\u{298A}'), ('\u{298A}', '\u{2989}'),
    ('\u{298B}', '\u{298C}'), ('\u{298C}', '\u{298B}'), ('\u{298D}', '\u{2990}'),
    ('\u{298E}', '\u{298F}'), ('\u{298F}', '\u{298E}'), ('\u{2990}', '\u{298D}'),
    ('\u{2991}', '\u{2992}'), ('\u{2992}', '\u{2991}'), ('\u{2993}', '\u{2994}'),
    ('\u{2994}', '\u{2993}'), ('\u{2995}', '\u{2996}'), ('\u{2996}', '\u{2995}'),
    ('\u{2997}', '\u{2998}'), ('\u{2998}', '\u{2997}'), ('\u{299B}', '\u{2221}'),
    ('\u{29A0}', '\u{2222}'), ('\u{29A3}', '\u{2220}'), ('\u{29A4}', '\u{29A5}'),
    ('\u{29A5}', '\u{29A4}'), ('\u{29A8}', '\u{29A9}'), ('\u{29A9}', '\u{29A8}'),
    ('\u{29AA}', '\u{29AB}'), ('\u{29AB}', '\u{29AA}'), ('\u{29AC}', '\u{29AD}'),
    ('\u{29AD}', '\u{29AC}'), ('\u{29AE}', '\u{29AF}'), ('\u{29AF}', '\u{29AE}'),
    ('\u{29B8}', '\u{2298}'), ('\u{29C0}', '\u{29C1}'), ('\u{29C1}', '\u{29C0}'),
    ('\u{29C4}', '\u{29C5}'), ('\u{29C5}', '\u{29C4}'), ('\u{29CF}', '\u{29D0}'),
    ('\u{29D0}', '\u{29CF}'), ('\u{29D1}', '\u{29D2}'), ('\u{29D2}', '\u{29D1}'),
    ('\u{29D4}', '\u{29D5}'), ('\u{29D5}', '\u{29D4}'), ('\u{29D8}', '\u{29D9}'),
    ('\u{29D9}', '\u{29D8}'), ('\u{29DA}', '\u{29DB}'), ('\u{29DB}', '\u{29DA}'),
    ('\u{29E8}', '\u{29E9}'), ('\u{29E9}', '\u{29E8}'), ('\u{29F5}', '\u{2215}'),
    ('\u{29F8}', '\u{29F9}'), ('\u{29F9}', '\u{29F8}'), ('\u{29FC}', '\u{29FD}'),
    ('\u{29FD}', '\u{29FC}'), ('\u{2A2B}', '\u{2A2C}'), ('\u{2A2C}', '\u{2A2B}'),
    ('\u{2A2D}', '\u{2A2E}'), ('\u{2A2E}', '\u{2A2D}'), ('\u{2A34}', '\u{2A35}'),
    ('\u{2A35}', '\u{2A34}'), ('\u{2A3C}', '\u{2A3D}'), ('\u{2A3D}', '\u{2A3C}'),
    ('\u{2A64}', '\u{2A65}'), ('\u{2A65}', '\u{2A64}'), ('\u{2A79}', '\u{2A7A}'),
    ('\u{2A7A}', '\u{2A79}'), ('\u{2A7B}', '\u{2A7C}'), ('\u{2A7C}', '\u{2A7B}'),
    ('\u{2A7D}', '\u{2A7E}'), ('\u{2A7E}', '\u{2A7D}'), ('\u{2A7F}', '\u{2A80}'),
    ('\u{2A80}', '\u{2A7F}'), ('\u{2A81}', '\u{2A82}'), ('\u{2A82}', '\u{2A81}'),
    ('\u{2A83}', '\u{2A84}'), ('\u{2A84}', '\u{2A83}'), ('\u{2A85}', '\u{2A86}'),
    ('\u{2A86}', '\u{2A85}'), ('\u{2A87}', '\u{2A88}'), ('\u{2A88}', '\u{2A87}'),
    ('\u{2A89}', '\u{2A8A}'), ('\u{2A8A}', '\u{2A89}'), ('\u{2A8B}', '\u{2A8C}'),
    ('\u{2A8C}', '\u{2A8B}'), ('\u{2A8D}', '\u{2A8E}'), ('\u{2A8E}', '\u{2A8D}'),
    ('\u{2A8F}', '\u{2A90}'), ('\u{2A90}', '\u{2A8F}'), ('\u{2A91}', '\u{2A92}'),
    ('\u{2A92}', '\u{2A91}'), ('\u{2A93}', '\u{2A94}'), ('\u{2A94}', '\u{2A93}'),
    ('\u{2A95}', '\u{2A96}'), ('\u{2A96}', '\u{2A95}'), ('\u{2A97}', '\u{2A98}'),
    ('\u{2A98}', '\u{2A97}'), ('\u{2A99}', '\u{2A9A}'), ('\u{2A9A}', '\u{2A99}'),
    ('\u{2A9B}', '\u{2A9C}'), ('\u{2A9C}', '\u{2A9B}'), ('\u{2A9D}', '\u{2A9E}'),
    ('\u{2A9E}', '\u{2A9D}'), ('\u{2A9F}', '\u{2AA0}'), ('\u{2AA0}', '\u{2A9F}'),
    ('\u{2AA1}', '\u{2AA2}'), ('\u{2AA2}', '\u{2AA1}'), ('\u{2AA6}', '\u{2AA7}'),
    ('\u{2AA7}', '\u{2AA6}'), ('\u{2AA8}', '\u{2AA9}'), ('\u{2AA9}', '\u{2AA8}'),
    ('\u{2AAA}', '\u{2AAB}'), ('\u{2AAB}', '\u{2AAA}'), ('\u{2AAC}', '\u{2AAD}'),
    ('\u{2AAD}', '\u{2AAC}'), ('\u{2AAF}', '\u{2AB0}'), ('\u{2AB0}', '\u{2AAF}'),
    ('\u{2AB1}', '\u{2AB2}'), ('\u{2AB2}', '\u{2AB1}'), ('\u{2AB3}', '\u{2AB4}'),
    ('\u{2AB4}', '\u{2AB3}'), ('\u{2AB5}', '\u{2AB6}'), ('\u{2AB6}', '\u{2AB5}'),
    ('\u{2AB7}', '\u{2AB8}'), ('\u{2AB8}', '\u{2AB7}'), ('\u{2AB9}', '\u{2ABA}'),
    ('\u{2ABA}', '\u{2AB9}'), ('\u{2ABB}', '\u{2ABC}'), ('\u{2ABC}', '\u{2ABB}'),
    ('\u{2ABD}', '\u{2ABE}'), ('\u{2ABE}', '\u{2ABD}'), ('\u{2ABF}', '\u{2AC0}'),
    ('\u{2AC0}', '\u{2ABF}'), ('\u{2AC1}', '\u{2AC2}'), ('\u{2AC2}', '\u{2AC1}'),
    ('\u{2AC3}', '\u{2AC4}'), ('\u{2AC4}', '\u{2AC3}'), ('\u{2AC5}', '\u{2AC6}'),
    ('\u{2AC6}', '\u{2AC5}'), ('\u{2AC7}', '\u{2AC8}'), ('\u{2AC8}', '\u{2AC7}'),
    ('\u{2AC9}', '\u{2ACA}'), ('\u{2ACA}', '\u{2AC9}'), ('\u{2ACB}', '\u{2ACC}'),
    ('\u{2ACC}', '\u{2ACB}'), ('\u{2ACD}', '\u{2ACE}'), ('\u{2ACE}', '\u{2ACD}'),
    ('\u{2ACF}', '\u{2AD0}'), ('\u{2AD0}', '\u{2ACF}'), ('\u{2AD1}', '\u{2AD2}'),
    ('\u{2AD2}', '\u{2AD1}'), ('\u{2AD3}', '\u{2AD4}'), ('\u{2AD4}', '\u{2AD3}'),
    ('\u{2AD5}', '\u{2AD6}'), ('\u{2AD6}', '\u{2AD5}'), ('\u{2ADE}', '\u{22A6}'),
    ('\u{2AE3}', '\u{22A9}'), ('\u{2AE4}', '\u{22A8}'), ('\u{2AE5}', '\u{22AB}'),
    ('\u{2AEC}', '\u{2AED}'), ('\u{2AED}', '\u{2AEC}'), ('\u{2AEE}', '\u{2224}'),
    ('\u{2AF7}', '\u{2AF8}'), ('\u{2AF8}', '\u{2AF7}'), ('\u{2AF9}', '\u{2AFA}'),
    ('\u{2AFA}', '\u{2AF9}'), ('\u{2BFE}', '\u{221F}'), ('\u{2E02}', '\u{2E03}'),
    ('\u{2E03}', '\u{2E02}'), ('\u{2E04}', '\u{2E05}'), ('\u{2E05}', '\u{2E04}'),
    ('\u{2E09}', '\u{2E0A}'), ('\u{2E0A}', '\u{2E09}'), ('\u{2E0C}', '\u{2E0D}'),
    ('\u{2E0D}', '\u{2E0C}'), ('\u{2E1C}', '\u{2E1D}'), ('\u{2E1D}', '\u{2E1C}'),
    ('\u{2E20}', '\u{2E21}'), ('\u{2E21}', '\u{2E20}'), ('\u{2E22}', '\u{2E23}'),
    ('\u{2E23}', '\u{2E22}'), ('\u{2E24}', '\u{2E25}'), ('\u{2E25}', '\u{2E24}'),
    ('\u{2E26}', '\u{2E27}'), ('\u{2E27}', '\u{2E26}'), ('\u{2E28}', '\u{2E29}'),
    ('\u{2E29}', '\u{2E28}'), ('\u{2E55}', '\u{2E56}'), ('\u{2E56}', '\u{2E55}'),
    ('\u{2E57}', '\u{2E58}'), ('\u{2E58}', '\u{2E57}'), ('\u{2E59}', '\u{2E5A}'),
    ('\u{2E5A}', '\u{2E59}'), ('\u{2E5B}', '\u{2E5C}'), ('\u{2E5C}', '\u{2E5B}'),
    ('\u{3008}', '\u{3009}'), ('\u{3009}', '\u{3008}'), ('\u{300A}', '\u{300B}'),
    ('\u{300B}', '\u{300A}'), ('\u{300C}', '\u{300D}'), ('\u{300D}', '\u{300C}'),
    ('\u{300E}', '\u{300F}'), ('\u{300F}', '\u{300E}'), ('\u{3010}', '\u{3011}'),
    ('\u{3011}', '\u{3010}'), ('\u{3014}', '\u{3015}'), ('\u{3015}', '\u{3014}'),
    ('\u{3016}', '\u{3017}'), ('\u{3017}', '\u{3016}'), ('\u{3018}', '\u{3019}'),
    ('\u{3019}', '\u{3018}'), ('\u{301A}', '\u{301B}'), ('\u{301B}', '\u{301A}'),
    ('\u{FE59}', '\u{FE5A}'), ('\u{FE5A}', '\u{FE59}'), ('\u{FE5B}', '\u{FE5C}'),
    ('\u{FE5C}', '\u{FE5B}'), ('\u{FE5D}', '\u{FE5E}'), ('\u{FE5E}', '\u{FE5D}'),
    ('\u{FE64}', '\u{FE65}'), ('\u{FE65}', '\u{FE64}'), ('\u{FF08}', '\u{FF09}'),
    ('\u{FF09}', '\u{FF08}'), ('\u{FF1C}', '\u{FF1E}'), ('\u{FF1E}', '\u{FF1C}'),
    ('\u{FF3B}', '\u{FF3D}'), ('\u{FF3D}', '\u{FF3B}'), ('\u{FF5B}', '\u{FF5D}'),
    ('\u{FF5D}', '\u{FF5B}'), ('\u{FF5F}', '\u{FF60}'), ('\u{FF60}', '\u{FF5F}'),
    ('\u{FF62}', '\u{FF63}'), ('\u{FF63}', '\u{FF62}'),
];

#[cfg(test)]
mod tests {
    use super::{MIRROR_PAIRS, UNICODE_VERSION};

    /// The checked-in UCD data file, byte-for-byte as `unicode.org` serves it
    /// (its own licence header included) — the generator's input and this
    /// module's oracle.
    const UCD: &str = include_str!("../../data/BidiMirroring-16.0.0.txt");

    /// Parses `BidiMirroring.txt` exactly as the one-off generator did: strip
    /// comments, split on `;`, hex-decode both fields, take the symmetric
    /// closure, sort.
    fn parse_ucd() -> Vec<(char, char)> {
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for raw in UCD.lines() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut fields = line.split(';');
            let from = fields.next().expect("a first field").trim();
            let to = fields.next().expect("a second field").trim();
            assert!(fields.next().is_none(), "exactly two fields: {raw:?}");
            let from = u32::from_str_radix(from, 16).expect("a hex codepoint");
            let to = u32::from_str_radix(to, 16).expect("a hex codepoint");
            pairs.push((from, to));
        }
        let mut closed: std::collections::BTreeMap<u32, u32> = pairs.iter().copied().collect();
        for (from, to) in pairs {
            closed.entry(to).or_insert(from);
        }
        closed
            .into_iter()
            .map(|(from, to)| {
                (
                    char::from_u32(from).expect("a scalar value"),
                    char::from_u32(to).expect("a scalar value"),
                )
            })
            .collect()
    }

    #[test]
    fn the_const_table_equals_the_checked_in_ucd_file() {
        // The verification-test-as-generator: this is what turns a hand edit
        // or a stale table into a red suite instead of a silent mis-render.
        let parsed = parse_ucd();
        assert_eq!(
            parsed.len(),
            MIRROR_PAIRS.len(),
            "the UCD file yields {} pairs, the const holds {}",
            parsed.len(),
            MIRROR_PAIRS.len(),
        );
        assert_eq!(parsed.as_slice(), MIRROR_PAIRS.as_slice());
    }

    #[test]
    fn the_data_file_declares_the_recorded_unicode_version() {
        let expected = format!("The repertoire covered by the file is Unicode {UNICODE_VERSION}.");
        assert!(
            UCD.contains(&expected),
            "the checked-in file must be Unicode {UNICODE_VERSION}",
        );
        assert!(
            UCD.contains("https://www.unicode.org/terms_of_use.html"),
            "the UCD licence reference must survive verbatim",
        );
    }

    #[test]
    fn the_table_is_sorted_with_unique_keys() {
        for window in MIRROR_PAIRS.windows(2) {
            assert!(
                window[0].0 < window[1].0,
                "sorted and unique: {:?} then {:?}",
                window[0],
                window[1],
            );
        }
    }

    #[test]
    fn the_table_is_a_total_involution() {
        for &(from, to) in &MIRROR_PAIRS {
            let back = MIRROR_PAIRS
                .binary_search_by_key(&to, |&(key, _)| key)
                .map(|index| MIRROR_PAIRS[index].1)
                .unwrap_or_else(|_| panic!("{to:?} is not a key, so {from:?} has no way back"));
            assert_eq!(back, from, "involution fails for {from:?}");
            assert_ne!(from, to, "no codepoint mirrors to itself");
        }
    }

    #[test]
    fn every_pair_has_equal_utf8_length() {
        // THE load-bearing gate: `shape::runs_for_bidi` mirrors the display
        // string in place and relies on the byte length never changing.
        let mut histogram = [0_usize; 5];
        for &(from, to) in &MIRROR_PAIRS {
            assert_eq!(
                from.len_utf8(),
                to.len_utf8(),
                "a cross-length pair would break in-place mirroring: {from:?} -> {to:?}",
            );
            histogram[from.len_utf8()] += 1;
        }
        assert_eq!(
            histogram,
            [0, 8, 2, 418, 0],
            "the measured Unicode 16.0.0 histogram, by UTF-8 length",
        );
    }

    #[test]
    fn the_whole_table_stays_in_the_basic_multilingual_plane() {
        let max = MIRROR_PAIRS
            .iter()
            .map(|&(from, _)| from)
            .max()
            .expect("a non-empty table");
        assert_eq!(max, '\u{FF63}', "the recorded maximum mirrored codepoint");
    }

    #[test]
    fn every_v13_hand_table_pair_survives_unchanged() {
        // The regression fence: the 32 pairs the curated v1.3 table held must
        // mirror exactly as they did, or completeness changed old output.
        const V13: [(char, char); 32] = [
            ('(', ')'),
            (')', '('),
            ('[', ']'),
            (']', '['),
            ('{', '}'),
            ('}', '{'),
            ('<', '>'),
            ('>', '<'),
            ('\u{00AB}', '\u{00BB}'),
            ('\u{00BB}', '\u{00AB}'),
            ('\u{2039}', '\u{203A}'),
            ('\u{203A}', '\u{2039}'),
            ('\u{2264}', '\u{2265}'),
            ('\u{2265}', '\u{2264}'),
            ('\u{226A}', '\u{226B}'),
            ('\u{226B}', '\u{226A}'),
            ('\u{FF08}', '\u{FF09}'),
            ('\u{FF09}', '\u{FF08}'),
            ('\u{FF3B}', '\u{FF3D}'),
            ('\u{FF3D}', '\u{FF3B}'),
            ('\u{FF5B}', '\u{FF5D}'),
            ('\u{FF5D}', '\u{FF5B}'),
            ('\u{3008}', '\u{3009}'),
            ('\u{3009}', '\u{3008}'),
            ('\u{300A}', '\u{300B}'),
            ('\u{300B}', '\u{300A}'),
            ('\u{300C}', '\u{300D}'),
            ('\u{300D}', '\u{300C}'),
            ('\u{300E}', '\u{300F}'),
            ('\u{300F}', '\u{300E}'),
            ('\u{3010}', '\u{3011}'),
            ('\u{3011}', '\u{3010}'),
        ];
        for (from, to) in V13 {
            let found = MIRROR_PAIRS
                .binary_search_by_key(&from, |&(key, _)| key)
                .map(|index| MIRROR_PAIRS[index].1)
                .unwrap_or_else(|_| panic!("the v1.3 pair {from:?} vanished"));
            assert_eq!(found, to, "the v1.3 mapping for {from:?} changed");
        }
    }
}
