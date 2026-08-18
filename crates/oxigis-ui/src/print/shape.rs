// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Shaped text for the PDF export (print v1.2, bidi since v1.3 — see
//! docs/plans/print-v12.md and print-v13.md): per-face runs shaped through
//! [`oxitext::SwashShaper`], classified into TJ-expressible clusters, and
//! degraded per run (LTR) or per string (bidi) to the v1.1 cmap+hmtx walk
//! when the result is not expressible.
//!
//! # Units
//!
//! Every advance and offset here is in **font design units**: shaping is
//! called with `px_size = 0.0`, which swash documents (and the probe runs
//! verified) as "offsets and advances in font units" — integers-as-f32,
//! exactly representable, so shaping is bit-deterministic and one pass
//! serves every text size on the page.
//!
//! # The cluster is the unit of everything
//!
//! swash's own contract: an RTL consumer must reverse **clusters, not
//! glyphs** — intra-cluster glyphs stay in logical order because a mark's
//! `x_offset` is relative to the pen *after* its base's advance. So a run
//! is a vector of [`RawCluster`]s; visual order for an RTL segment is
//! `clusters.reverse()` and nothing else ever reorders.
//!
//! # The classifier is the safety story
//!
//! A shaped run is used only when every glyph is live (`gid != 0`),
//! advances are finite and non-negative, `y_advance` is zero (no vertical
//! writing), offsets are finite and sane, and clusters are non-decreasing
//! char boundaries. A one-glyph cluster is plain (1:1 or n:1 ligature —
//! exact through `/ToUnicode`); a multi-glyph cluster is carried WHOLE and
//! rendered under an `/ActualText` span (per-glyph `/ToUnicode` for a
//! reordering script would be a lie — the v1.2 `group == chars` zip was
//! measurably wrong on Devanagari and is deleted, not extended).
//! Anything else falls back to the v1.1 walk: complex scripts degrade
//! exactly as they printed yesterday, never garble.

use oxitext::{ShapeDirection, ShapeRequest, ShapedGlyph, SwashShaper};
use std::collections::BTreeMap;
use ttf_parser::Face;

use super::bidi::{self, BidiSegment};

/// One glyph before subsetting: the id shaping produced (post-GSUB, so a
/// ligature or contextual-form gid is reachable here and never from a cmap
/// walk), its advance and offsets in the face's own design units.
pub(super) struct RawGlyph {
    /// Glyph id in the ORIGINAL face.
    pub old_gid: u16,
    /// Advance in font design units (post-GPOS: kerning already applied).
    pub advance_units: f32,
    /// Horizontal placement offset in design units (`0.0` on every plain
    /// path; a combining mark's shift otherwise — rides a paired TJ).
    pub x_offset_units: f32,
    /// Vertical placement offset in design units (`0.0` on every plain
    /// path; rides `Ts` otherwise).
    pub y_offset_units: f32,
}

/// One shaping cluster: 1+ glyphs in LOGICAL intra-cluster order, plus the
/// exact SOURCE characters the cluster stands for (pre-mirroring — what
/// `/ToUnicode` and `/ActualText` say). Never empty.
pub(super) struct RawCluster {
    /// The glyphs, logical order (mark placement depends on it).
    pub glyphs: Vec<RawGlyph>,
    /// The source text this cluster draws.
    pub text: String,
}

impl RawCluster {
    /// Whether this cluster needs an `/ActualText` wrapper: more than one
    /// glyph, or any placement offset — the cases a per-CID `/ToUnicode`
    /// cannot express. (The production decision lives in the plan's pass 4,
    /// where sub-epsilon offsets have already been clamped away; this is
    /// the tests' shorthand for the same rule.)
    #[cfg(test)]
    pub fn needs_actual_text(&self) -> bool {
        self.glyphs.len() > 1
            || self
                .glyphs
                .iter()
                .any(|glyph| glyph.x_offset_units != 0.0 || glyph.y_offset_units != 0.0)
    }
}

/// One maximal same-face, same-direction span of a resolved string, in
/// VISUAL order across the containing vector.
pub(super) struct RawRun {
    /// Index into the shell's font chain.
    pub chain_index: usize,
    /// Clusters in visual order (== logical for LTR; reversed for RTL).
    pub clusters: Vec<RawCluster>,
}

/// Whether `text` contains right-to-left script or bidi controls — the gate
/// between the v1.2 LTR path (false ⇒ byte-identical output by
/// construction) and the v1.3 bidi path. The explicit embedding controls
/// are included so they reach the bidi path to be REFUSED there
/// (`bidi::has_explicit_bidi_control`) instead of shaping as Latin.
pub(super) fn has_rtl(text: &str) -> bool {
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

/// Whether `text` contains a **complex left-to-right script** — one whose
/// correct rendering needs script-aware itemisation (reordering matras,
/// conjuncts, reph) that v1.4 deliberately does NOT do.
///
/// Nothing branches on this: it feeds ONE aggregated per-export log, so a
/// user whose Devanagari or Khmer labels print without conjuncts is told
/// rather than left to wonder.
///
/// The original reason `shape_request` with a real Indic tag did not ship —
/// swash 0.2.10 garbled on a reused shaper and PANICKED (index out of bounds)
/// on real Hindi — **no longer holds**: the oxitext 0.2.2 bump of 2026-08-11
/// fixed both, and the `_a_canary_not_a_complaint` test below now pins the
/// fixed state. Itemisation simply has not landed on this path yet; see
/// docs/plans/print-v14.md's item-1 ruling for what it still needs.
///
/// The Brahmic and South-East-Asian blocks are listed by range rather than
/// by script property: no new dependency, and the boundary only has to be
/// right enough to count strings for a log.
pub(super) fn has_complex_ltr(text: &str) -> bool {
    text.chars().any(|ch| {
        matches!(u32::from(ch),
            // Devanagari … Sinhala, then Thai/Lao/Tibetan.
            0x0900..=0x0FFF
                // Myanmar.
                | 0x1000..=0x109F
                // Khmer.
                | 0x1780..=0x17FF
                // Vedic Extensions.
                | 0x1CD0..=0x1CFF
                // Devanagari Extended.
                | 0xA8E0..=0xA8FF
        )
    })
}

/// Splits `resolved` into maximal same-face runs (by the coverage map) and
/// shapes each one — the v1.2 LTR path, byte-identical for every string
/// `has_rtl` answers `false` for. A run the classifier refuses falls back
/// to the v1.1 cmap walk for that run alone.
pub(super) fn runs_for(
    shaper: &mut SwashShaper,
    chain: &[Vec<u8>],
    faces: &[Option<Face<'_>>],
    coverage: &BTreeMap<char, usize>,
    resolved: &str,
) -> Vec<RawRun> {
    let mut runs: Vec<RawRun> = Vec::new();
    let mut current: Option<(usize, String)> = None;
    let mut flush = |runs: &mut Vec<RawRun>, chain_index: usize, text: String| {
        let Some(bytes) = chain.get(chain_index) else {
            return;
        };
        let shaped = shaper
            .shape_slice(bytes, &text, 0.0)
            .ok()
            .and_then(|glyphs| cluster_runs(&glyphs, &text, &text));
        let clusters = match shaped {
            Some(clusters) => clusters,
            None => {
                let Some(Some(face)) = faces.get(chain_index) else {
                    return;
                };
                tracing::debug!(
                    chain_index,
                    "oxigis-ui print: run not TJ-expressible; the v1.1 walk serves it",
                );
                synthesize(face, &text)
            }
        };
        if !clusters.is_empty() {
            runs.push(RawRun {
                chain_index,
                clusters,
            });
        }
    };
    for ch in resolved.chars() {
        let Some(&chain_index) = coverage.get(&ch) else {
            continue;
        };
        match current.as_mut() {
            Some((index, text)) if *index == chain_index => text.push(ch),
            Some((index, text)) => {
                let done = (*index, core::mem::take(text));
                flush(&mut runs, done.0, done.1);
                current = Some((chain_index, ch.to_string()));
            }
            None => current = Some((chain_index, ch.to_string())),
        }
    }
    if let Some((index, text)) = current {
        flush(&mut runs, index, text);
    }
    runs
}

/// Shapes a bidi string through its visual-order [`BidiSegment`]s — the
/// v1.3 path. All-or-nothing: [`None`] sends the WHOLE string to the v1.1
/// per-character walk (a half-reordered line would be worse than uniform
/// v1.1 output).
///
/// Even-level segments shape through `shape_slice` exactly as the LTR path
/// (which also keeps Arabic-Indic digits — class AN, even level — away
/// from `shape_request`'s Ltr→Rtl auto-upgrade trap). Odd-level segments
/// shape through `shape_request` with an explicit `Rtl` direction and the
/// segment's script tag — the ONLY entry point that reaches Arabic GSUB
/// (measured: `shape_slice_rtl` hard-codes the Latin script and returns
/// isolated forms). The shaper hands RTL glyphs back in LOGICAL order;
/// visual order is `clusters.reverse()`, per swash's own contract.
///
/// `coverage` is the plan's character → chain-face map, EXTENDED with the
/// mirror partners of the page's characters (`font::plan` does the
/// extension). It is what makes UAX #9 L4 safe now that the mirror table is
/// complete: [`bidi::mirror_for`] only mirrors onto a partner the same face
/// covers, so a partner no face draws can no longer produce a gid 0 that
/// refuses the segment and un-shapes the whole line.
pub(super) fn runs_for_bidi(
    shaper: &mut SwashShaper,
    chain: &[Vec<u8>],
    coverage: &BTreeMap<char, usize>,
    segments: &[BidiSegment],
    resolved: &str,
) -> Option<Vec<RawRun>> {
    let mut runs: Vec<RawRun> = Vec::new();
    for segment in segments {
        let source = resolved.get(segment.range.clone())?;
        if source.is_empty() {
            continue;
        }
        let bytes = chain.get(segment.chain_index)?;
        let clusters = if segment.is_rtl() {
            // UAX #9 L4 on the DISPLAY text only; every mirror pair has
            // equal UTF-8 length, so swash's cluster byte offsets index
            // `source` identically (no offset map).
            let display: String = source
                .chars()
                .map(|ch| bidi::mirror_for(ch, coverage))
                .collect();
            debug_assert_eq!(display.len(), source.len());
            let request = ShapeRequest::builder()
                .text(&display)
                .font_data(bytes)
                .px_size(0.0)
                .direction(ShapeDirection::Rtl)
                .script(segment.script)
                .build()
                .ok()?;
            let glyphs = shaper.shape_request(&request).ok()?;
            let mut clusters = cluster_runs(&glyphs, &display, source)?;
            clusters.reverse();
            clusters
        } else {
            let glyphs = shaper.shape_slice(bytes, source, 0.0).ok()?;
            cluster_runs(&glyphs, source, source)?
        };
        if clusters.is_empty() {
            continue;
        }
        match runs.last_mut() {
            // Adjacent visual segments on the same face merge into one run
            // (one `Tf`, one span decision for the emitter).
            Some(run) if run.chain_index == segment.chain_index => {
                run.clusters.extend(clusters);
            }
            _ => runs.push(RawRun {
                chain_index: segment.chain_index,
                clusters,
            }),
        }
    }
    Some(runs)
}

/// The v1.1 per-character cmap+hmtx walk — byte-identical degradation for a
/// run the shaper cannot express, as one-glyph clusters.
fn synthesize(face: &Face<'_>, text: &str) -> Vec<RawCluster> {
    text.chars()
        .filter_map(|ch| {
            let gid = face.glyph_index(ch)?;
            Some(RawCluster {
                glyphs: vec![RawGlyph {
                    old_gid: gid.0,
                    advance_units: f32::from(face.glyph_hor_advance(gid).unwrap_or(0)),
                    x_offset_units: 0.0,
                    y_offset_units: 0.0,
                }],
                text: ch.to_string(),
            })
        })
        .collect()
}

/// Whether `ch` is a default-ignorable formatting character the shaper may
/// drop without a glyph (ZWNJ/ZWJ, LRM/RLM, and friends).
fn is_default_ignorable(ch: char) -> bool {
    matches!(
        u32::from(ch),
        0x00AD | 0x200B..=0x200F | 0x2060..=0x2064 | 0xFEFF
    )
}

/// Classifies a shaped run into clusters, or refuses it to the fallback.
///
/// `display` is the string that was actually shaped (post-mirroring);
/// `source` is the logical original the clusters' text is sliced from —
/// equal lengths by the mirror table's construction. Kept rules from v1.2:
/// every `gid != 0`; advances finite and `>= 0`; `y_advance == 0`;
/// clusters non-decreasing on char boundaries; sane glyph count. Changed
/// in v1.3: a leading run of dropped default-ignorables is absorbed into
/// the first cluster (Persian's ZWNJ, RLM-prefixed labels); a multi-glyph
/// cluster is ONE [`RawCluster`] (the `/ActualText` tier) instead of the
/// deleted per-glyph zip; placement offsets are accepted when finite and
/// sane (|y| ≤ 2 em, |x| ≤ 4 em in 1000-unit terms is checked downstream
/// against the face's own upem by the caller's translation — here the
/// guard is finiteness, non-garbage magnitude is the emitter's clamp).
fn cluster_runs(glyphs: &[ShapedGlyph], display: &str, source: &str) -> Option<Vec<RawCluster>> {
    if glyphs.is_empty() || glyphs.len() > display.chars().count().saturating_mul(4) + 8 {
        return None;
    }
    debug_assert_eq!(display.len(), source.len());
    let mut previous_cluster = 0_u32;
    for (index, glyph) in glyphs.iter().enumerate() {
        if glyph.gid == 0 {
            return None;
        }
        if glyph.y_advance != 0.0 {
            return None;
        }
        if !glyph.x_advance.is_finite() || glyph.x_advance < 0.0 {
            return None;
        }
        if !glyph.x_offset.is_finite() || !glyph.y_offset.is_finite() {
            return None;
        }
        let cluster = glyph.cluster;
        if index == 0 && cluster != 0 {
            // The shaper drops default-ignorable formatting characters, so
            // a string starting with ZWJ/RLM shapes with a nonzero first
            // cluster. Absorb exactly that; anything else is a broken map.
            let head = source.get(..cluster as usize)?;
            if head.is_empty() || !head.chars().all(is_default_ignorable) {
                return None;
            }
        }
        if cluster < previous_cluster {
            return None;
        }
        if !display.is_char_boundary(cluster as usize) {
            return None;
        }
        previous_cluster = cluster;
    }
    // Group by cluster value; a group's source range ends where the next
    // distinct cluster begins (or at the end of the run's text). A leading
    // absorbed ignorable prefix rides in the first cluster's text.
    let mut out = Vec::with_capacity(glyphs.len());
    let mut start = 0_usize;
    while start < glyphs.len() {
        let cluster = glyphs[start].cluster as usize;
        let source_start = if start == 0 { 0 } else { cluster };
        let mut end = start + 1;
        while end < glyphs.len() && glyphs[end].cluster as usize == cluster {
            end += 1;
        }
        let source_end = glyphs
            .get(end)
            .map_or(source.len(), |next| next.cluster as usize);
        let text = source.get(source_start..source_end)?;
        if text.is_empty() {
            return None;
        }
        out.push(RawCluster {
            glyphs: glyphs[start..end]
                .iter()
                .map(|glyph| RawGlyph {
                    old_gid: glyph.gid,
                    advance_units: glyph.x_advance,
                    x_offset_units: glyph.x_offset,
                    y_offset_units: glyph.y_offset,
                })
                .collect(),
            text: text.to_string(),
        });
        start = end;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noto() -> Vec<u8> {
        oxifont_bundled::NOTO_SANS_REGULAR.to_vec()
    }

    fn shaped(text: &str) -> Vec<RawCluster> {
        let chain = vec![noto()];
        let bytes = noto();
        let face = Face::parse(&bytes, 0).expect("the bundled face parses");
        let faces = vec![Some(face)];
        let coverage: BTreeMap<char, usize> = text.chars().map(|ch| (ch, 0)).collect();
        let mut shaper = SwashShaper::new();
        let runs = runs_for(&mut shaper, &chain, &faces, &coverage, text);
        assert_eq!(runs.len(), 1, "one face, one run");
        runs.into_iter()
            .next()
            .map(|run| run.clusters)
            .unwrap_or_default()
    }

    /// A hand-built glyph for the font-free classifier tests, with real
    /// probe-measured numbers where the test names them.
    fn glyph(gid: u16, advance: f32, x_offset: f32, y_offset: f32, cluster: u32) -> ShapedGlyph {
        ShapedGlyph {
            gid,
            x_advance: advance,
            x_offset,
            y_offset,
            cluster,
            ..ShapedGlyph::default()
        }
    }

    #[test]
    fn kerning_lands_in_the_shaped_advances() {
        // Probe-verified goldens on the bundled Noto (upem 1000, GPOS
        // XAdvance A,V = -40): "AV" shapes to 599 + 600 design units where
        // hmtx says 639 + 600.
        let clusters = shaped("AV");
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].glyphs.len(), 1);
        assert_eq!(clusters[0].glyphs[0].advance_units, 599.0);
        assert_eq!(clusters[1].glyphs[0].advance_units, 600.0);
    }

    #[test]
    fn a_ligature_is_one_cluster_that_stands_for_both_characters() {
        // GSUB latn/liga: f + i -> gid 1654 — a glyph no cmap walk reaches.
        let clusters = shaped("fi");
        assert_eq!(clusters.len(), 1, "liga fires by default");
        assert_eq!(clusters[0].text, "fi");
        assert_eq!(clusters[0].glyphs.len(), 1);
        assert!(
            !clusters[0].needs_actual_text(),
            "n:1 is exact via /ToUnicode"
        );
    }

    #[test]
    fn an_rtl_string_is_detected_for_the_bidi_path() {
        assert!(has_rtl("مرحبا"));
        assert!(has_rtl("שלום"));
        assert!(
            has_rtl("a\u{202D}b"),
            "explicit controls route to the bidi refusal"
        );
        assert!(has_rtl("a\u{200E}b"), "LRM routes to the bidi path");
        assert!(!has_rtl("Tokyo 東京"));
    }

    #[test]
    fn complex_ltr_scripts_are_detected_for_the_honesty_log() {
        // The log's input, and nothing else — no code path branches on it.
        assert!(has_complex_ltr("नई दिल्ली"), "Devanagari");
        assert!(has_complex_ltr("กรุงเทพ"), "Thai");
        assert!(has_complex_ltr("ភ្នំពេញ"), "Khmer");
        assert!(has_complex_ltr("ಬೆಂಗಳೂರು"), "Kannada");
        assert!(has_complex_ltr("Road to नई"), "one character is enough");
        assert!(!has_complex_ltr("Tokyo 東京"), "CJK needs no itemisation");
        assert!(!has_complex_ltr("مرحبا"), "RTL has its own path");
        assert!(!has_complex_ltr(""));
    }

    #[test]
    fn shaping_is_deterministic_across_shaper_instances() {
        let first = shaped("Tokyo AV fi");
        let second = shaped("Tokyo AV fi");
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.text, b.text);
            assert_eq!(a.glyphs.len(), b.glyphs.len());
            for (x, y) in a.glyphs.iter().zip(&b.glyphs) {
                assert_eq!(x.old_gid, y.old_gid);
                assert_eq!(x.advance_units, y.advance_units);
            }
        }
    }

    #[test]
    fn plain_rtl_clusters_reverse_and_keep_advances() {
        // Probe-measured tahoma مرحبا: logical gids [995,942,931,914,910]
        // with advances 807/854/1085/500/470; classify then reverse as
        // `runs_for_bidi` does for an odd-level segment.
        let text = "\u{645}\u{631}\u{62D}\u{628}\u{627}";
        let advances = [807.0, 854.0, 1085.0, 500.0, 470.0];
        let gids = [995_u16, 942, 931, 914, 910];
        let glyphs: Vec<ShapedGlyph> = text
            .char_indices()
            .zip(gids.iter().zip(&advances))
            .map(|((offset, _), (&gid, &advance))| glyph(gid, advance, 0.0, 0.0, offset as u32))
            .collect();
        let mut clusters = cluster_runs(&glyphs, text, text).expect("plain clusters");
        let total: f32 = clusters
            .iter()
            .flat_map(|cluster| cluster.glyphs.iter())
            .map(|glyph| glyph.advance_units)
            .sum();
        clusters.reverse();
        assert_eq!(
            clusters
                .iter()
                .map(|cluster| cluster.glyphs[0].old_gid)
                .collect::<Vec<_>>(),
            vec![910, 914, 931, 942, 995],
            "visual order is the reverse of logical"
        );
        let after: f32 = clusters
            .iter()
            .flat_map(|cluster| cluster.glyphs.iter())
            .map(|glyph| glyph.advance_units)
            .sum();
        assert_eq!(total, after, "reversal moves no ink");
    }

    #[test]
    fn intra_cluster_glyphs_never_reverse() {
        // Probe-measured tahoma beh+fatha: ONE cluster, two glyphs, the
        // mark placed at x −1395 relative to the pen AFTER the base — so
        // the base must stay first even in an RTL run.
        let glyphs = vec![
            glyph(911, 1653.0, 0.0, 0.0, 0),
            glyph(756, 0.0, -1395.0, 27.0, 0),
        ];
        let clusters = cluster_runs(&glyphs, "\u{628}\u{64E}", "\u{628}\u{64E}")
            .expect("a marked cluster classifies");
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].glyphs.len(), 2);
        assert_eq!(clusters[0].glyphs[0].old_gid, 911, "base first");
        assert_eq!(clusters[0].glyphs[1].old_gid, 756, "mark second");
        assert!(clusters[0].needs_actual_text());
    }

    #[test]
    fn a_leading_zero_width_joiner_is_absorbed() {
        // Probe-measured tahoma ZWJ+beh: one glyph, cluster 3 — the ZWJ is
        // dropped by the shaper. v1.2 refused every such string.
        let text = "\u{200D}\u{628}";
        let glyphs = vec![glyph(912, 1919.0, 0.0, 0.0, 3)];
        let clusters = cluster_runs(&glyphs, text, text).expect("the prefix is absorbed");
        assert_eq!(clusters.len(), 1);
        assert_eq!(
            clusters[0].text, text,
            "the ignorable rides in the cluster text"
        );

        // A nonzero first cluster over REAL characters is still refused.
        let broken = vec![glyph(912, 1919.0, 0.0, 0.0, 2)];
        assert!(cluster_runs(&broken, "ab\u{628}", "ab\u{628}").is_none());
    }

    #[test]
    fn an_n_to_m_cluster_is_one_actual_text_cluster_with_no_per_glyph_zip() {
        // Probe-measured Nirmala कि under dev2: the i-matra glyph comes
        // FIRST (gid 302), the ka second (gid 248) — the v1.2 "glyph i
        // draws char i" zip wrote a scrambled /ToUnicode for both. One
        // cluster, whole text, ActualText tier.
        let text = "\u{915}\u{93F}";
        let glyphs = vec![
            glyph(302, 532.0, 0.0, 0.0, 0),
            glyph(248, 1768.0, 0.0, 0.0, 0),
        ];
        let clusters = cluster_runs(&glyphs, text, text).expect("n:m classifies now");
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].text, text);
        assert_eq!(clusters[0].glyphs.len(), 2);
        assert!(clusters[0].needs_actual_text());
    }

    #[test]
    fn vertical_advance_and_notdef_still_refuse() {
        let text = "ab";
        let vertical = vec![glyph(5, 100.0, 0.0, 0.0, 0), {
            let mut g = glyph(6, 100.0, 0.0, 0.0, 1);
            g.y_advance = 10.0;
            g
        }];
        assert!(cluster_runs(&vertical, text, text).is_none());
        let notdef = vec![glyph(0, 100.0, 0.0, 0.0, 0)];
        assert!(cluster_runs(&notdef, "a", "a").is_none());
    }

    #[test]
    fn the_oxifont_subset_fixture_cannot_kern_a_canary_not_a_complaint() {
        // MEASURED in the design probes: oxifont-subset keeps GSUB/GPOS in
        // its table directory but the rewritten lookups do not fire. The
        // kerning tests therefore use the FULL bundled Noto. Re-measured
        // 2026-08-11 against oxifont-subset 0.2.2 from crates.io (the git
        // dependency on rev 900dd4b is gone, and so is that rev): still 1239.
        // If this canary ever fails,
        // oxifont-subset fixed its rewriters upstream — re-read the fixture
        // strategy rather than deleting the test.
        let codepoints: std::collections::BTreeSet<char> = "AV".chars().collect();
        let tiny = oxifont_subset::subset_font(&noto(), &codepoints).expect("fixture subset");
        let mut shaper = SwashShaper::new();
        let glyphs = shaper
            .shape_slice(&tiny, "AV", 0.0)
            .expect("the fixture shapes");
        let total: f32 = glyphs.iter().map(|glyph| glyph.x_advance).sum();
        assert_eq!(
            total, 1239.0,
            "639 + 600: the fixture's GPOS does not fire (full Noto gives 1199)",
        );
    }

    /// 24 real Hindi words, most carrying a `र्` reph — the corpus the v1.4
    /// design measured `shape_request(Ltr, dev2)` against.
    const HINDI_CORPUS: [&str; 24] = [
        "कर्म",
        "धर्म",
        "वर्ष",
        "मार्ग",
        "सूर्य",
        "पूर्व",
        "कार्य",
        "दर्शन",
        "पर्वत",
        "सर्व",
        "गर्व",
        "अर्थ",
        "स्वर्ग",
        "निर्माण",
        "पूर्ण",
        "वर्तमान",
        "वर्षा",
        "आदर्श",
        "संघर्ष",
        "उत्तर",
        "दिल्ली",
        "हिन्दी",
        "भारत",
        "कि",
    ];

    /// What one `shape_request(Ltr, dev2)` call did.
    #[derive(Debug, PartialEq, Eq)]
    enum IndicOutcome {
        /// The glyph ids it produced.
        Glyphs(Vec<u16>),
        /// It panicked (payload text).
        Panicked(String),
    }

    /// Shapes `text` through `shaper` under an explicit Devanagari tag,
    /// converting a panic into a value so the canary can assert on it.
    ///
    /// The `catch_unwind` is load-bearing even though nothing panics here any
    /// more (oxitext-swash 0.2.2 fixed it — see the canary below). It is what
    /// makes a REGRESSION report as a clean assertion failure naming the word
    /// that broke, instead of aborting the whole suite with a backtrace. The
    /// panic hook is silenced for the same reason: if the shaper ever panics
    /// again, this test's own message is the news, not libstd's.
    fn shape_devanagari(shaper: &mut SwashShaper, font: &[u8], text: &str) -> Option<IndicOutcome> {
        match shape_caught(shaper, font, text, *b"dev2")? {
            Ok(glyphs) => Some(IndicOutcome::Glyphs(
                glyphs.iter().map(|glyph| glyph.gid).collect(),
            )),
            Err(payload) => Some(IndicOutcome::Panicked(payload)),
        }
    }

    /// `shape_request(Ltr, tag)`, with any panic converted into an `Err`
    /// carrying the payload text instead of unwinding out of the test.
    ///
    /// The one place the `catch_unwind` lives, shared by the Indic canary and
    /// the v1.6 measurement sweep — see [`shape_devanagari`] for why it is
    /// kept now that nothing panics. [`None`] means the REQUEST failed to
    /// build, which is a different thing from the shaper refusing or dying.
    fn shape_caught(
        shaper: &mut SwashShaper,
        font: &[u8],
        text: &str,
        tag: [u8; 4],
    ) -> Option<Result<Vec<ShapedGlyph>, String>> {
        let request = ShapeRequest::builder()
            .text(text)
            .font_data(font)
            .px_size(0.0)
            .direction(ShapeDirection::Ltr)
            .script(tag)
            .build()
            .ok()?;
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            shaper.shape_request(&request)
        }));
        std::panic::set_hook(previous);
        match caught {
            Ok(shaped) => Some(Ok(shaped.ok()?)),
            Err(payload) => Some(Err(payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|text| (*text).to_string())
                })
                .unwrap_or_else(|| "(non-string panic payload)".to_string()))),
        }
    }

    #[test]
    #[ignore = "reads C:/Windows/Fonts/Nirmala.ttc; pins the post-0.2.2 Indic shaping state"]
    fn swash_no_longer_garbles_or_panics_on_indic_a_canary_not_a_complaint() {
        // WHAT THIS TEST USED TO SAY, AND WHY IT NOW SAYS THE OPPOSITE
        //
        // Until 2026-08-11 this test was named
        // `swash_still_garbles_and_panics_on_indic_...` and asserted that
        // swash 0.2.10 was BROKEN under a real Indic tag: (1) an inter-call
        // shaping-state defect that replaced मार्ग's reph with a duplicate of
        // the preceding ga on a reused shaper, and (2) an index-out-of-bounds
        // panic (swash-0.2.10/src/shape/buffer.rs:680) on four-word Hindi.
        // Those two defects were v1.4 item 1's whole reason for deferring LTR
        // script itemisation. The canary was written to FAIL the day they were
        // fixed, and on 2026-08-11 it did: the oxitext 0.2.2 bump vendored the
        // shaper as `oxitext-swash`, and both defects are gone.
        //
        // **v1.4 item 1's swash-side blocker is therefore DISCHARGED.** From
        // here a failure of this test means an oxitext-swash REGRESSION, not a
        // discovery — re-read docs/plans/print-v14.md before touching it.
        //
        // The panic is fixed STRUCTURALLY, not just on this corpus: the
        // reorder buffer now ends with a terminal `for i in 0..len { emit!(i) }`
        // sweep that makes `order[..len]` a total permutation (any index still
        // unplaced keeps its source position instead of aliasing a stale entry
        // from an earlier cluster), and the `placed: [bool; 64]` bitmap is
        // matched by a `.min(64)` clamp on `len`. That argument is what makes
        // the fix trustworthy — a 24-word corpus can never prove the absence of
        // a panic, so it is recorded here alongside the numbers.
        //
        // Re-measured 2026-08-11 on Nirmala.ttc face 0 (Windows 11) against
        // oxitext-swash 0.2.2.
        let Ok(nirmala) = std::fs::read("C:/Windows/Fonts/Nirmala.ttc") else {
            return;
        };
        let delhi = "दिल्ली";
        let marg = "मार्ग";

        // Defect 1, the inter-call half. मार्ग is [ma, aa, ga, reph] where gid
        // 330 is the zero-advance reph mark. The baseline gids are UNCHANGED
        // from the broken era — only the reused-shaper result moved — so this
        // first assert still guards against the whole face drifting under us.
        // One `SwashShaper` serves a whole export (`font::plan`), so the reused
        // case below is exactly the production condition.
        let mut fresh = SwashShaper::new();
        let correct = shape_devanagari(&mut fresh, &nirmala, marg).expect("a request builds");
        assert_eq!(
            correct,
            IndicOutcome::Glyphs(vec![273, 301, 250, 330]),
            "the fresh-shaper baseline moved; re-measure before trusting the rest",
        );
        let mut reused = SwashShaper::new();
        let _ = shape_devanagari(&mut reused, &nirmala, delhi).expect("a request builds");
        let after = shape_devanagari(&mut reused, &nirmala, marg).expect("a request builds");
        assert_eq!(
            after, correct,
            "a reused shaper must now agree with a fresh one; it used to \
             return [273, 301, 250, 250], duplicating ga in place of the reph",
        );

        // Defect 1, the intra-call half: one call, two words. The second word
        // used to lose its reph to a duplicate even on a brand-new shaper, so
        // "one shaper per string" was never a workaround. Gid 3 is the space.
        let mut single_call = SwashShaper::new();
        let phrase = shape_devanagari(&mut single_call, &nirmala, &format!("{delhi} {marg}"))
            .expect("a request builds");
        assert_eq!(
            phrase,
            IndicOutcome::Glyphs(vec![302, 265, 671, 305, 3, 273, 301, 250, 330]),
            "both words must keep their own shaping inside ONE call",
        );

        // Defect 2, the panic. This exact string used to abort with an
        // index-out-of-bounds; it now returns four rephs, one per word. The
        // `catch_unwind` inside `shape_devanagari` is kept precisely so that a
        // regression here fails this assertion instead of killing the suite —
        // on wasm32 `panic = "abort"` is set by the target and no `catch_unwind`
        // can help, which is why a panicking shaper could never ship.
        let mut no_longer_panicking = SwashShaper::new();
        let four_words = "सूर्य पूर्व वर्षा मार्ग";
        let outcome = shape_devanagari(&mut no_longer_panicking, &nirmala, four_words)
            .expect("a request builds");
        assert_eq!(
            outcome,
            IndicOutcome::Glyphs(vec![
                285, 307, 274, 330, 3, 269, 307, 281, 330, 3, 281, 284, 301, 330, 3, 273, 301, 250,
                330,
            ]),
            "{four_words:?} used to panic at swash-0.2.10/src/shape/buffer.rs:680",
        );

        // The corpus pass under production conditions: ONE shaper, 24 real
        // words, each cross-checked against a shaper that has seen nothing
        // else. 4 of these used to panic (पूर्ण, वर्तमान, आदर्श, संघर्ष) and
        // स्वर्ग lost its reph even on a fresh shaper. Now every word agrees
        // with its own fresh baseline and none panics — which is the property
        // itemisation would actually depend on.
        let mut corpus_shaper = SwashShaper::new();
        for word in HINDI_CORPUS {
            let reused_word =
                shape_devanagari(&mut corpus_shaper, &nirmala, word).expect("a request builds");
            assert!(
                matches!(reused_word, IndicOutcome::Glyphs(_)),
                "{word:?} panicked on a reused shaper: {reused_word:?}",
            );
            let mut own = SwashShaper::new();
            let fresh_word = shape_devanagari(&mut own, &nirmala, word).expect("a request builds");
            assert_eq!(
                reused_word, fresh_word,
                "{word:?} shapes differently on a reused shaper than on a fresh one",
            );
        }
        let mut svarga_shaper = SwashShaper::new();
        assert_eq!(
            shape_devanagari(&mut svarga_shaper, &nirmala, "स्वर्ग"),
            Some(IndicOutcome::Glyphs(vec![738, 250, 330])),
            "स्वर्ग must keep its reph (gid 330); it used to duplicate ga instead",
        );
    }

    /// One script's row in the v1.6 itemisation measurement sweep.
    struct ScriptProbe {
        /// The OpenType tag `shape_request` would be handed. Spelled exactly
        /// as `SCRIPTS_BY_TAG` spells it in oxitext-swash 0.2.2
        /// (`src/text/unicode_data/script_tables.rs`) — the four-character
        /// `dev2`/`bng2`/`ory2`/`knd2` forms, NOT `deva`/`beng`/`orya`/`knda`.
        tag: [u8; 4],
        /// Human name, for the printed table.
        name: &'static str,
        /// A face on this machine that covers the script. Windows system
        /// fonts, following the Indic canary's existing gating idiom: absent
        /// means skip, never fail.
        font: &'static str,
        /// A real word chosen to carry reordering (a reph, a below-base form,
        /// or a pre-base matra) rather than a bare consonant string.
        word: &'static str,
        /// A second word, shaped first, to put the shaper in a used state.
        primer: &'static str,
    }

    const SWEEP: [ScriptProbe; 7] = [
        ScriptProbe {
            tag: *b"dev2",
            name: "Devanagari",
            font: "C:/Windows/Fonts/Nirmala.ttc",
            word: "मार्ग",
            primer: "दिल्ली",
        },
        ScriptProbe {
            tag: *b"bng2",
            name: "Bengali",
            font: "C:/Windows/Fonts/Nirmala.ttc",
            word: "কর্ম",
            primer: "বাংলা",
        },
        ScriptProbe {
            tag: *b"ory2",
            name: "Oriya",
            font: "C:/Windows/Fonts/Nirmala.ttc",
            word: "କର୍ମ",
            primer: "ଓଡ଼ିଶା",
        },
        ScriptProbe {
            tag: *b"knd2",
            name: "Kannada",
            font: "C:/Windows/Fonts/Nirmala.ttc",
            word: "ಕರ್ಮ",
            primer: "ಕನ್ನಡ",
        },
        ScriptProbe {
            tag: *b"sinh",
            name: "Sinhala",
            font: "C:/Windows/Fonts/Nirmala.ttc",
            word: "ශ්‍රී",
            primer: "සිංහල",
        },
        ScriptProbe {
            tag: *b"telu",
            name: "Telugu",
            font: "C:/Windows/Fonts/Nirmala.ttc",
            word: "కర్మ",
            primer: "తెలుగు",
        },
        ScriptProbe {
            tag: *b"thai",
            name: "Thai",
            font: "C:/Windows/Fonts/LeelawUI.ttf",
            word: "ภาษาไทย",
            primer: "กรุงเทพ",
        },
    ];

    /// The v1.6 measurement sweep (plan stage S3) — R1 versus R2, per script.
    ///
    /// **This test calls no production code path and changes no shipped byte.**
    /// It exists to produce numbers, because print/text v1.4's guards G3 and
    /// G4 were calibrated against the reused-shaper reph duplication that
    /// `oxitext-swash` 0.2.2 fixed. G3's one discriminating case (a `+387`
    /// advance) *was* that defect, so both guards now have plausible
    /// false-refusal modes and neither may be ported forward on its old
    /// numbers. Everything printed here is pasted into `TODO.md`'s v1.6 entry
    /// and becomes the calibration for the itemisation work itself.
    ///
    /// Ladder vocabulary is v1.3's: **R1** is `shape_slice`, the byte-identity
    /// floor that ships today and shapes everything as Latin; **R2** is
    /// `shape_request` under a real script tag, the capability being measured.
    ///
    /// The GSUB check is done with `ttf_parser` **before** the shaper is
    /// called, never by shaping and inspecting the result. That ordering is
    /// the whole point: on wasm32 `panic = "abort"` is set by the target, so
    /// "try it and see" is not available to the shipping code this calibrates.
    /// Expect at least one face to DECLARE a script it cannot actually cover —
    /// presence is not coverage, which is exactly why G1 and G2 are separate
    /// guards.
    #[test]
    #[ignore = "reads Windows system fonts; prints the v1.6 itemisation measurements"]
    fn the_itemisation_measurement_sweep() {
        let mut disagreements: Vec<String> = Vec::new();
        let mut panics: Vec<String> = Vec::new();
        let mut widened: Vec<String> = Vec::new();

        println!("\n=== v1.6 itemisation sweep: R1 (shape_slice) vs R2 (shape_request) ===");
        for probe in &SWEEP {
            let Ok(bytes) = std::fs::read(probe.font) else {
                println!("[{}] SKIP — {} not present", probe.name, probe.font);
                continue;
            };
            let tag = String::from_utf8_lossy(&probe.tag).to_string();

            // G1's pre-shaping form: does the face's GSUB declare the script
            // at all? `index` is a lookup into the script list, not a claim
            // about coverage.
            let declared = Face::parse(&bytes, 0).ok().and_then(|face| {
                face.tables()
                    .gsub
                    .and_then(|gsub| gsub.scripts.index(ttf_parser::Tag::from_bytes(&probe.tag)))
            });

            // R1 — what ships today: no tag, so swash defaults to Latin.
            let mut r1_shaper = SwashShaper::new();
            let Some(Ok(r1)) = r1_shaper
                .shape_slice(&bytes, probe.word, 0.0)
                .ok()
                .map(Ok::<_, ()>)
            else {
                println!("[{}] R1 refused to shape {:?}", probe.name, probe.word);
                continue;
            };
            let r1_gids: Vec<u16> = r1.iter().map(|glyph| glyph.gid).collect();
            let r1_advance: f32 = r1.iter().map(|glyph| glyph.x_advance).sum();

            // R2 — the capability: an explicit script tag on a fresh shaper.
            let mut r2_shaper = SwashShaper::new();
            let Some(r2_outcome) = shape_caught(&mut r2_shaper, &bytes, probe.word, probe.tag)
            else {
                println!("[{}] R2 request failed to build", probe.name);
                continue;
            };
            let r2 = match r2_outcome {
                Ok(glyphs) => glyphs,
                Err(payload) => {
                    println!("[{}] R2 PANICKED: {payload}", probe.name);
                    panics.push(format!("{} ({tag}) fresh: {payload}", probe.name));
                    continue;
                }
            };
            let r2_gids: Vec<u16> = r2.iter().map(|glyph| glyph.gid).collect();
            let r2_advance: f32 = r2.iter().map(|glyph| glyph.x_advance).sum();

            // G4's raw material: adjacent-equal gid pairs R2 introduces and R1
            // does not. The dead defect produced exactly this shape, so the
            // count is recorded rather than turned into a threshold here.
            let adjacent_equal = |gids: &[u16]| -> usize {
                gids.windows(2).filter(|pair| pair[0] == pair[1]).count()
            };
            let r2_pairs = adjacent_equal(&r2_gids);
            let r1_pairs = adjacent_equal(&r1_gids);

            // R2 on a shaper that has already seen an unrelated word — the
            // production condition, since one `SwashShaper` serves a whole
            // export (`font::plan`).
            let mut reused_shaper = SwashShaper::new();
            let _ = shape_caught(&mut reused_shaper, &bytes, probe.primer, probe.tag);
            let reused_gids = match shape_caught(&mut reused_shaper, &bytes, probe.word, probe.tag)
            {
                Some(Ok(glyphs)) => glyphs.iter().map(|glyph| glyph.gid).collect::<Vec<u16>>(),
                Some(Err(payload)) => {
                    println!("[{}] R2 reused PANICKED: {payload}", probe.name);
                    panics.push(format!("{} ({tag}) reused: {payload}", probe.name));
                    continue;
                }
                None => continue,
            };
            if reused_gids != r2_gids {
                disagreements.push(format!(
                    "{} ({tag}): fresh {r2_gids:?} vs reused {reused_gids:?}",
                    probe.name,
                ));
            }
            // G3's DIRECTION, re-derived. v1.4 calibrated this guard on a
            // −548..−2619-good / +387-bad split, but the `+387` case was the
            // reph duplication and no longer exists. What survives measurement
            // across all seven scripts is the inequality itself: itemisation
            // substitutes and reorders, so R2 either matches R1 or comes out
            // NARROWER — never wider. A widening R2 is still the signature of
            // something having gone wrong.
            if r2_advance > r1_advance {
                widened.push(format!(
                    "{} ({tag}): R2 {r2_advance} > R1 {r1_advance}",
                    probe.name,
                ));
            }

            println!(
                "\n[{}] tag={tag} word={:?}\n  GSUB script index : {}\n  \
                 R1 gids           : {r1_gids:?}\n  R1 advance        : {r1_advance:.1}\n  \
                 R2 gids           : {r2_gids:?}\n  R2 advance        : {r2_advance:.1}\n  \
                 SIGNED DELTA      : {:+.1}  (G3 re-calibration input)\n  \
                 adjacent-equal    : R1 {r1_pairs}, R2 {r2_pairs}  (G4 input)\n  \
                 gid 0 in R2       : {}  (G2 input)\n  \
                 reused == fresh   : {}",
                probe.name,
                probe.word,
                declared.map_or_else(|| "ABSENT".to_string(), |index| index.to_string()),
                r2_advance - r1_advance,
                r2_gids.contains(&0),
                reused_gids == r2_gids,
            );
        }

        // Screen versus page width, for the one string v1.4 recorded a "13 %"
        // disagreement on. Both sides are measured through the paths that
        // actually ship, so the number is comparable to what a user sees.
        if let Ok(bytes) = std::fs::read("C:/Windows/Fonts/Nirmala.ttc") {
            let word = "मार्ग";
            use crate::print::font::{PrintFonts, plan};
            use oxigis_core::LabelWeight;

            let fonts = PrintFonts::new(vec![bytes.clone()]);
            let page = plan(&fonts, &[(LabelWeight::Regular, word)], None)
                .map(|plan| plan.width_pt(LabelWeight::Regular, word, 16.0));
            let screen = oxigis_render::LabelEngine::new(bytes)
                .ok()
                .and_then(|mut engine| engine.measure(word, 16.0).ok())
                .map(|size| size[0]);
            println!(
                "\n[screen vs page] {word:?} at 16.0 — page {page:?} pt, screen {screen:?} px",
            );
            if let (Some(page), Some(screen)) = (page, screen)
                && page > 0.0
            {
                println!(
                    "  disagreement    : {:+.1} % of the page width",
                    (screen - page) / page * 100.0,
                );
            }
        }

        println!("\n=== end of sweep ===\n");
        assert!(
            panics.is_empty(),
            "the shaper must not panic under any measured tag: {panics:?}",
        );
        assert!(
            disagreements.is_empty(),
            "a reused shaper must agree with a fresh one — this is the \
             property itemisation depends on, and the defect that used to \
             break it is fixed: {disagreements:?}",
        );
        assert!(
            widened.is_empty(),
            "G3's surviving direction: an itemised run must never advance \
             WIDER than the unitemised one: {widened:?}",
        );
    }
}
