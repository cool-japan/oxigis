//! Unit tests for the shaping engine — split out of `engine.rs` so the
//! production module stays well inside the 2000-line limit.

use std::sync::Arc;

use super::{
    DEFAULT_LABEL_CACHE, FaceCache, LabelEngine, LabelOrientation, LabelWeight, MAX_LABEL_CACHE,
    MAX_LABEL_SIZE_PX, label_style, raster_cache_key,
};
use crate::error::RenderError;

/// Noto Sans Regular (OFL-1.1), the only font the tests need: Latin-only,
/// so it doubles as the "no CJK coverage" case.
fn font() -> Vec<u8> {
    oxifont_bundled::NOTO_SANS_REGULAR.to_vec()
}

fn engine() -> LabelEngine {
    LabelEngine::new(font()).expect("bundled Noto Sans parses")
}

#[test]
fn a_latin_label_shapes_into_packed_glyphs() {
    let mut engine = engine();
    let label = engine.shape("Hello", 14.0).expect("Hello shapes");
    assert_eq!(label.glyphs().len(), 5);
    assert!(label.width_px() > 0.0 && label.height_px() > 0.0);
    assert_eq!(label.font_size_px(), 14.0);
    assert!(!label.is_empty());

    // Glyphs advance left to right and stay inside the label box.
    let mut previous = f32::NEG_INFINITY;
    for glyph in label.glyphs() {
        assert!(
            glyph.offset_px[0] >= previous,
            "glyphs must not run backwards"
        );
        previous = glyph.offset_px[0];
        assert!(glyph.slot.width > 0 && glyph.slot.height > 0);
        assert!(glyph.offset_px[0] + glyph.slot.width as f32 <= label.width_px() + 2.0);
        assert_eq!(glyph.offset_px[0], glyph.offset_px[0].round());
        assert_eq!(glyph.offset_px[1], glyph.offset_px[1].round());
    }
    assert_eq!(engine.atlas().len(), 4, "'l' is packed once for two uses");
    assert!(engine.atlas().is_dirty());
}

#[test]
fn descenders_sit_lower_than_cap_heights() {
    let mut engine = engine();
    let label = engine.shape("Hg", 24.0).expect("Hg shapes");
    let [cap, descender] = [0usize, 1].map(|i| label.glyphs()[i]);
    // 'H' rests on the baseline; 'g' hangs below it, so its box both starts
    // lower and ends lower.
    assert!(descender.offset_px[1] > cap.offset_px[1]);
    let cap_bottom = cap.offset_px[1] + cap.slot.height as f32;
    let descender_bottom = descender.offset_px[1] + descender.slot.height as f32;
    assert!(
        descender_bottom > cap_bottom,
        "descender bottom {descender_bottom} should fall below cap bottom {cap_bottom}"
    );
}

#[test]
fn measuring_agrees_with_shaping() {
    let mut engine = engine();
    let measured = engine.measure("Tokyo", 16.0).expect("measure works");
    let shaped = engine.shape("Tokyo", 16.0).expect("shape works");
    assert!(measured[0] > 0.0 && measured[1] > 0.0);
    assert!((measured[0] - shaped.width_px()).abs() < 0.01);
    assert!((measured[1] - shaped.height_px()).abs() < 0.01);
}

#[test]
fn empty_and_whitespace_labels_have_no_box_and_no_glyphs() {
    let mut engine = engine();
    for text in ["", "   ", "\t "] {
        let label = engine.shape(text, 14.0).expect("whitespace shapes");
        assert!(label.is_empty(), "{text:?} should produce no glyphs");
        assert_eq!(label.size_px(), [0.0, 0.0], "{text:?} should have no box");
    }
    assert_eq!(engine.measure("", 14.0).expect("measure works"), [0.0, 0.0]);
    assert!(engine.atlas().is_empty());
}

#[test]
fn shaping_is_cached_by_text_and_size() {
    let mut engine = engine();
    let first = engine.shape("Kyoto", 14.0).expect("shapes");
    let second = engine.shape("Kyoto", 14.0).expect("shapes");
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "same key, same Arc"
    );
    assert_eq!(engine.cached_labels(), 1);

    let bigger = engine.shape("Kyoto", 15.0).expect("shapes");
    assert!(
        !std::sync::Arc::ptr_eq(&first, &bigger),
        "size is part of the key"
    );
    assert_eq!(engine.cached_labels(), 2);
    assert_eq!(engine.cache_capacity(), DEFAULT_LABEL_CACHE);
}

#[test]
fn the_cache_evicts_least_recently_used_entries() {
    let mut engine = LabelEngine::with_capacity(font(), 4).expect("capacity 4 is valid");
    for index in 0..4 {
        let _ = engine
            .shape(&format!("label {index}"), 12.0)
            .expect("shapes");
    }
    assert_eq!(engine.cached_labels(), 4);
    // Re-touch the oldest so it is no longer the eviction victim.
    let _ = engine.shape("label 0", 12.0).expect("shapes");
    let _ = engine.shape("label 4", 12.0).expect("shapes");
    assert_eq!(engine.cached_labels(), 4);
    // "label 0" is seven characters, but the space has no outline.
    assert_eq!(
        engine
            .shape("label 0", 12.0)
            .expect("shapes")
            .glyphs()
            .len(),
        6
    );
}

#[test]
fn a_zero_capacity_cache_is_rejected() {
    assert!(matches!(
        LabelEngine::with_capacity(font(), 0),
        Err(RenderError::InvalidCapacity(0))
    ));
}

#[test]
fn garbage_font_bytes_are_an_error_not_a_panic() {
    assert!(matches!(
        LabelEngine::new(vec![0u8; 64]),
        Err(RenderError::Text(_))
    ));
}

#[test]
fn cjk_without_a_cjk_font_still_succeeds() {
    // The whole point of the fallback path: a Latin-only primary font must
    // not fail on Japanese, it must return `.notdef` boxes.
    let mut engine = engine();
    let label = engine.shape("東京都", 14.0).expect("CJK must not fail");
    assert!(label.width_px() > 0.0, "notdef boxes still occupy space");
    // Adding a fallback later (what the web shell does once its CJK font
    // has been fetched) invalidates everything shaped before it.
    let generation = engine.generation();
    engine.add_fallback_font(font());
    assert_eq!(engine.fallback_count(), 1);
    assert!(
        engine.generation() > generation,
        "fallbacks must invalidate"
    );
    assert_eq!(engine.cached_labels(), 0);

    let again = engine.shape("東京都", 14.0).expect("CJK still shapes");
    assert_eq!(again.generation(), engine.generation());
}

/// The mixed-script screen baseline (v1.6 plan stage S4) — **retired
/// 2026-08-12**, and now pinning the FIXED behaviour.
///
/// It was written on 2026-08-11 to pin a defect. `oxitext` 0.2.2's
/// `shape_run_with_notdef_fallback` resolved a `.notdef` glyph through a
/// fallback face and then overwrote the **run-level** `font_data` with that
/// fallback (`run.font_data = Arc::clone(fb_data)`). A [`ShapedRun`]
/// carries ONE font for all its glyphs, so a single fallback character
/// re-pointed the WHOLE run — including the glyphs the primary had
/// resolved perfectly well, whose gids were left untouched. Those primary
/// gids were then rasterised out of the fallback face, and `"Tokyo東京"`
/// came back as `[(1,55),(1,82),(1,78),(1,92),(1,82),(1,4138),(1,2652)]`,
/// drawing **"Uplzp東京"**.
///
/// `oxitext` 0.2.3 (published 2026-08-12, workspace bumped in the same
/// change as this retirement) emits one run per font instead of overwriting
/// a shared one, and this test flipped from PASS to FAIL — the intended
/// signal. It is retired in place rather than deleted: the same assertions,
/// read the other way round, are now the regression fence for the fix.
///
/// Measured here with `"Tokyo東京"`, Noto Sans Regular primary and
/// **BIZ-UDGothic** fallback:
///
/// * the primary is interned first, so it is font index 0 and the fallback
///   is index 1 — otherwise "font 0" would just mean "first face seen";
/// * the five Latin glyphs stay on font **0** with gids
///   `[55, 82, 78, 92, 82]`, identical to the Latin-only control;
/// * only 東京 moves to font **1**, as `[4138, 2652]`;
/// * read through the primary those five ids spell **"Tokyo"**. Read
///   through the fallback — which is what 0.2.2 did — they would still
///   spell "Uplzp", and this test asserts that too, so a regression reports
///   as the specific wrong word rather than as an opaque index mismatch.
///
/// The fallback face is chosen deliberately. Meiryo, MS Gothic, MS Mincho
/// and Yu Gothic all number basic Latin exactly as Noto does
/// (`T`=55, `o`=82, `k`=78, `y`=92), so with any of them the defect
/// produced perfectly correct output and could not be seen at all.
/// BIZ-UDGothic is off by one (`T`=54) and turned the same bug into visible
/// garbage. That near-miss is the point: this class of defect survives
/// review because the common fonts hide it — which is why the fixture must
/// not be "simplified" to a more ordinary face.
///
/// Nothing else in the workspace observes this —
/// [`cjk_without_a_cjk_font_still_succeeds`] passes the same bundled Noto
/// as both primary and fallback, so there is no second face to be wrong
/// about.
///
/// **A failure here now means an `oxitext` regression**, not a defect to be
/// recorded. Read the v1.6 entry in `TODO.md` before changing a number — a
/// "corrected" golden written by hand proves nothing.
#[test]
#[ignore = "reads C:/Windows/Fonts/BIZ-UDGothicR.ttc; pins per-font run splitting"]
fn a_mixed_script_label_keeps_primary_glyphs_on_the_primary_face_a_canary_not_a_complaint() {
    let Ok(fallback) = std::fs::read("C:/Windows/Fonts/BIZ-UDGothicR.ttc") else {
        return;
    };
    let primary = ttf_parser::Face::parse(oxifont_bundled::NOTO_SANS_REGULAR, 0)
        .expect("bundled Noto parses");
    let fallback_face = ttf_parser::Face::parse(&fallback, 0).expect("BIZ-UDGothic parses");

    // The premises, read from the faces themselves rather than remembered.
    assert!(
        primary.glyph_index('東').is_none() && primary.glyph_index('京').is_none(),
        "the premise is that the primary cannot draw 東京",
    );
    assert_ne!(
        primary.glyph_index('T').map(|id| id.0),
        fallback_face.glyph_index('T').map(|id| id.0),
        "this fallback was picked BECAUSE its Latin numbering differs from \
         the primary's; if they now agree the test can no longer see the \
         defect and a different face must be chosen",
    );

    let mut engine = engine();
    engine.add_fallback_font(fallback.clone());
    assert_eq!(engine.fallback_count(), 1);

    // Shape Latin alone FIRST — *after* registering the fallback, which
    // matters: `add_fallback_font` bumps the generation and clears the
    // intern table, so a font interned before it loses its index. Doing it
    // in this order both interns the primary as index 0 (making the
    // indices below mean something rather than "first face seen") and
    // records the gids the primary resolves. Latin needs no fallback, so
    // this run keeps the primary's `font_data`.
    let control: Vec<(u16, u16)> = engine
        .shape("Tokyo", 16.0)
        .expect("Latin shapes")
        .glyphs()
        .iter()
        .map(|glyph| (glyph.key.font, glyph.key.gid))
        .collect();
    assert_eq!(
        control,
        vec![(0, 55), (0, 82), (0, 78), (0, 92), (0, 82)],
        "the primary must be font 0 here, with its own Latin gids; \
         re-measure before trusting the rest",
    );
    let seen: Vec<(u16, u16)> = engine
        .shape("Tokyo東京", 16.0)
        .expect("a mixed label shapes")
        .glyphs()
        .iter()
        .map(|glyph| (glyph.key.font, glyph.key.gid))
        .collect();
    println!("S4 mixed-script golden: {seen:?}");

    // THE FIX. Each glyph is attributed to the face that actually resolved
    // it: the five Latin ones to the primary, 東京 to the fallback. Under
    // 0.2.2 every entry here was font 1.
    assert_eq!(
        seen.iter().map(|(font, _)| *font).collect::<Vec<u16>>(),
        vec![0, 0, 0, 0, 0, 1, 1],
        "one CJK character must no longer re-point the Latin glyphs; \
         primary is index 0, fallback is index 1: {seen:?}",
    );
    assert_eq!(
        seen,
        vec![
            (0, 55),
            (0, 82),
            (0, 78),
            (0, 92),
            (0, 82),
            (1, 4138),
            (1, 2652),
        ],
        "the golden moved: 55/82/78/92/82 are the primary's Latin gids on \
         the primary face, 4138/2652 the fallback's own 東京",
    );
    assert_eq!(
        &seen[..control.len()]
            .iter()
            .map(|(_, gid)| *gid)
            .collect::<Vec<u16>>(),
        &control.iter().map(|(_, gid)| *gid).collect::<Vec<u16>>(),
        "the Latin gids must be unchanged by the presence of a fallback",
    );

    // And what that actually draws, spelled out both ways: through the
    // primary, which is where those gids now belong, and through the
    // fallback, which is where 0.2.2 sent them. Naming the wrong word keeps
    // a regression legible instead of reducing it to an index mismatch.
    let spell = |face: &ttf_parser::Face<'_>| -> String {
        seen[..control.len()]
            .iter()
            .map(|(_, gid)| {
                (0x20_u32..0x7F)
                    .filter_map(char::from_u32)
                    .find(|ch| face.glyph_index(*ch).map(|id| id.0) == Some(*gid))
                    .unwrap_or('?')
            })
            .collect()
    };
    assert_eq!(
        spell(&primary),
        "Tokyo",
        "the Latin gids must spell \"Tokyo\" in the face they are drawn from",
    );
    assert_eq!(
        spell(&fallback_face),
        "Uplzp",
        "the fixture must keep discriminating: these gids have to mean \
         something else in the fallback, or the test cannot see the defect \
         come back",
    );
}

#[test]
fn setting_fallbacks_invalidates_the_cache() {
    let mut engine = engine();
    let _ = engine.shape("Osaka", 14.0).expect("shapes");
    assert_eq!(engine.cached_labels(), 1);
    engine.set_fallback_fonts(vec![font()]);
    assert_eq!(engine.cached_labels(), 0);
    assert_eq!(engine.fallback_count(), 1);
    engine.set_fallback_fonts(Vec::new());
    assert_eq!(engine.fallback_count(), 0);
}

#[test]
fn clearing_empties_the_atlas_too() {
    let mut engine = engine();
    let _ = engine.shape("Nagoya", 14.0).expect("shapes");
    assert!(!engine.atlas().is_empty());
    engine.clear();
    assert!(engine.atlas().is_empty());
    assert_eq!(engine.cached_labels(), 0);
    assert_eq!(engine.atlas().generation(), 1);
}

#[test]
fn out_of_range_sizes_are_rejected() {
    let mut engine = engine();
    for size in [0.0, -1.0, f32::NAN, f32::INFINITY, MAX_LABEL_SIZE_PX + 1.0] {
        assert!(
            matches!(engine.shape("x", size), Err(RenderError::Text(_))),
            "size {size} should be rejected"
        );
        assert!(matches!(
            engine.measure("x", size),
            Err(RenderError::Text(_))
        ));
    }
    assert!(label_style(14.0).is_ok());
}

/// A 64² atlas cannot hold many 32 px glyphs, so the labels below exercise
/// the full-atlas path.
fn crowded_engine() -> LabelEngine {
    let mut engine = engine();
    *engine.atlas_mut() = crate::label::atlas::GlyphAtlas::with_size(64, 64).expect("64² is valid");
    engine
}

const CROWD: [&str; 8] = ["Aa", "Bb", "Cc", "Dd", "Ee", "Ff", "Gg", "Hh"];

#[test]
fn labels_stay_correct_when_the_atlas_fills_up() {
    // Was: "the atlas should have been rebuilt at least once". It is not —
    // the labels here are dropped as they are shaped, exactly as the placer
    // drops the candidates it rejects, so a full atlas is answered by freeing
    // the glyphs nothing draws. The invariant the test is named for is
    // unchanged, and the generation now proves the *cheap* rung ran.
    let mut engine = crowded_engine();
    let generation = engine.generation();
    for text in CROWD {
        let label = engine.shape(text, 32.0).expect("shapes");
        assert!(!label.is_empty(), "{text} must still draw");
        for glyph in label.glyphs() {
            assert_eq!(
                engine.atlas().get(&glyph.key),
                Some(glyph.slot),
                "a live label's slots must still be in the atlas"
            );
        }
    }
    assert_eq!(
        engine.generation(),
        generation,
        "freeing cold glyphs must not invalidate anything",
    );
}

#[test]
fn a_full_atlas_of_labels_the_caller_still_holds_rebuilds_and_says_so() {
    // The other rung: with every label still referenced there is nothing to
    // free, so the atlas is cleared and the generation moves — which is
    // exactly what `ShapedLabel::generation` exists to tell the caller.
    let mut engine = crowded_engine();
    let generation = engine.generation();
    let held: Vec<_> = CROWD
        .iter()
        .map(|text| engine.shape(text, 32.0).expect("shapes"))
        .collect();
    assert!(
        engine.generation() > generation,
        "nothing was free to evict, so the atlas must have been rebuilt",
    );
    assert_eq!(
        held.last().map(|label| label.generation()),
        Some(engine.generation()),
        "the label that forced the rebuild belongs to the new generation",
    );
}

#[test]
fn compaction_never_frees_a_glyph_a_caller_is_still_drawing() {
    // The soundness rule the whole eviction design rests on. `drawn` leaves a
    // two-entry cache almost immediately, but the caller still holds it, so
    // its slots may not be handed to another glyph — there is no generation
    // bump to warn anybody, which is what makes silent re-use unacceptable.
    let mut engine = LabelEngine::with_capacity(font(), 2).expect("capacity 2 is valid");
    *engine.atlas_mut() = crate::label::atlas::GlyphAtlas::with_size(64, 64).expect("64² is valid");
    let drawn = engine.shape("Aa", 32.0).expect("shapes");
    let slots: Vec<_> = drawn.glyphs().to_vec();
    assert!(!slots.is_empty());

    for text in CROWD.iter().skip(1) {
        let _ = engine.shape(text, 32.0).expect("shapes");
    }
    assert_eq!(
        engine.generation(),
        drawn.generation(),
        "the cold labels were free to evict, so nothing should have rebuilt",
    );
    for glyph in &slots {
        assert_eq!(
            engine.atlas().get(&glyph.key),
            Some(glyph.slot),
            "a held label's glyph was recycled under it",
        );
    }
}

#[test]
fn a_label_whose_glyphs_cannot_ever_be_packed_keeps_what_fits_and_is_cached() {
    // The documented contract: a glyph too large to ever fit the atlas is
    // dropped from the label rather than failing it. An 8² atlas holds no
    // glyph of a 32 px label at all, so the label draws nothing — but it is a
    // PERMANENT answer, so it is cached rather than re-shaped every frame.
    let mut engine = engine();
    *engine.atlas_mut() = crate::label::atlas::GlyphAtlas::with_size(8, 8).expect("8² is valid");
    let label = engine
        .shape("Hello", 32.0)
        .expect("an unpackable label still shapes");
    assert!(label.is_empty(), "no glyph of that size fits an 8² atlas");
    assert!(
        label.width_px() > 0.0,
        "the collision box is the layout's, not the atlas's",
    );
    assert_eq!(engine.cached_labels(), 1, "a permanent answer is cached");
    let again = engine.shape("Hello", 32.0).expect("shapes");
    assert!(std::sync::Arc::ptr_eq(&label, &again), "and it is a hit");
}

#[test]
fn a_label_that_alone_overfills_the_atlas_draws_what_fits_and_is_not_cached() {
    // THE regression fence for the overflow defect: the label used to be
    // dropped whole AND cached empty, so the feature stayed unlabelled for
    // the rest of the session. Now the glyphs that fit are drawn, and the
    // failure is not keyed — the next frame, against a different label set,
    // gets another go.
    let mut engine = engine();
    *engine.atlas_mut() = crate::label::atlas::GlyphAtlas::with_size(64, 64).expect("64² is valid");
    let text = "abcdefghijklmnopqrstuvwxyz";
    let label = engine.shape(text, 32.0).expect("shapes");
    assert!(
        !label.is_empty(),
        "the glyphs that fit must still be drawn, not thrown away with the label",
    );
    assert!(
        label.glyphs().len() < text.len(),
        "the premise is that this label cannot fit a 64² atlas",
    );
    assert_eq!(
        engine.cached_labels(),
        0,
        "a transient loss must not be frozen into the cache",
    );
    let again = engine.shape(text, 32.0).expect("shapes again");
    assert!(
        !std::sync::Arc::ptr_eq(&label, &again),
        "the next frame re-shapes rather than being served the damaged label",
    );
    // And the first label is still intact while the caller holds it.
    for glyph in label.glyphs() {
        assert_eq!(engine.atlas().get(&glyph.key), Some(glyph.slot));
    }
}

// --- print/text v1.4 item 4 (D-W3): weight ---

/// A stand-in "bold" face. The suite ships exactly one real font, so the
/// bold chain is exercised with a SECOND COPY of it: different bytes,
/// different `Arc`, different interned font index — which is precisely
/// what the cache-key and shared-atlas claims are about. It cannot prove
/// the glyphs look heavier, and does not pretend to; the real faces are
/// asserted in the desktop shell's font scan.
fn pretend_bold() -> Vec<u8> {
    font()
}

#[test]
fn a_bold_request_with_no_bold_chain_draws_regular_and_shares_its_cache_entry() {
    let mut engine = engine();
    assert!(!engine.has_bold());
    assert_eq!(engine.bold_face_count(), 0);
    let regular = engine.shape("Kyoto", 14.0).expect("shapes");
    let bold = engine
        .shape_weighted("Kyoto", 14.0, LabelWeight::Bold)
        .expect("shapes");
    assert!(
        std::sync::Arc::ptr_eq(&regular, &bold),
        "with no bold face the two weights are the same label",
    );
    assert_eq!(engine.cached_labels(), 1, "and share ONE cache entry");
}

#[test]
fn the_cache_key_carries_the_weight_so_bold_is_never_served_the_regular_label() {
    // THE correctness bug D-W3 names: a `(text, size)`-keyed cache hands
    // whichever weight asked first to both.
    let mut engine = engine();
    engine.set_bold_fonts(vec![pretend_bold()]);
    assert!(engine.has_bold());
    assert_eq!(engine.bold_face_count(), 1);
    let regular = engine.shape("Kyoto", 14.0).expect("shapes");
    let bold = engine
        .shape_weighted("Kyoto", 14.0, LabelWeight::Bold)
        .expect("shapes");
    assert!(
        !std::sync::Arc::ptr_eq(&regular, &bold),
        "the weights must not share a cache entry",
    );
    assert_eq!(engine.cached_labels(), 2);
    // Each weight still caches by itself.
    let again = engine
        .shape_weighted("Kyoto", 14.0, LabelWeight::Bold)
        .expect("shapes");
    assert!(std::sync::Arc::ptr_eq(&bold, &again));
    assert_eq!(engine.cached_labels(), 2);
}

#[test]
fn both_weights_pack_into_the_one_shared_atlas() {
    let mut engine = engine();
    engine.set_bold_fonts(vec![pretend_bold()]);
    let regular = engine.shape("Hi", 14.0).expect("shapes");
    let bold = engine
        .shape_weighted("Hi", 14.0, LabelWeight::Bold)
        .expect("shapes");
    for glyph in regular.glyphs().iter().chain(bold.glyphs()) {
        assert_eq!(
            engine.atlas().get(&glyph.key),
            Some(glyph.slot),
            "every glyph of either weight lives in the ONE atlas",
        );
    }
    assert_eq!(
        regular.generation(),
        bold.generation(),
        "one atlas, one generation",
    );
}

#[test]
fn installing_or_removing_a_bold_chain_invalidates_and_an_empty_one_is_a_no_op() {
    let mut engine = engine();
    let _ = engine.shape("Osaka", 14.0).expect("shapes");
    assert_eq!(engine.cached_labels(), 1);
    let generation = engine.generation();
    // Removing a chain that was never there must not cost an
    // invalidation — the no-bold shell calls this every font install.
    engine.set_bold_fonts(Vec::new());
    assert_eq!(engine.generation(), generation, "no-op");
    assert_eq!(engine.cached_labels(), 1);

    engine.set_bold_fonts(vec![pretend_bold()]);
    assert!(engine.generation() > generation);
    assert_eq!(engine.cached_labels(), 0);
    engine.set_bold_fonts(Vec::new());
    assert!(!engine.has_bold(), "bold can be taken away again");
}

#[test]
fn an_unparseable_bold_face_disables_bold_instead_of_failing() {
    let mut engine = engine();
    engine.set_bold_fonts(vec![vec![0u8; 64]]);
    assert!(!engine.has_bold(), "junk bytes must not become a pipeline");
    // And the map still draws.
    let label = engine
        .shape_weighted("Sendai", 14.0, LabelWeight::Bold)
        .expect("still shapes");
    assert!(!label.is_empty());
}

#[test]
fn the_bold_chain_keeps_the_whole_regular_chain_behind_it() {
    // Never-shrink: a fallback added AFTER the bold chain must still be
    // reachable from a bold request, so a bold Latin face cannot cost the
    // map its CJK coverage.
    let mut engine = engine();
    engine.set_bold_fonts(vec![pretend_bold()]);
    engine.add_fallback_font(font());
    assert!(engine.has_bold(), "the bold pipeline survives the rebuild");
    assert_eq!(engine.fallback_count(), 1);
    let label = engine
        .shape_weighted("東京都", 14.0, LabelWeight::Bold)
        .expect("CJK must not fail under bold");
    assert!(label.width_px() > 0.0);
}

#[test]
fn measuring_at_a_weight_agrees_with_shaping_at_that_weight() {
    let mut engine = engine();
    engine.set_bold_fonts(vec![pretend_bold()]);
    let measured = engine
        .measure_weighted("Tokyo", 16.0, LabelWeight::Bold)
        .expect("measure works");
    let shaped = engine
        .shape_weighted("Tokyo", 16.0, LabelWeight::Bold)
        .expect("shape works");
    assert!((measured[0] - shaped.width_px()).abs() < 0.01);
    assert!((measured[1] - shaped.height_px()).abs() < 0.01);
}

#[test]
fn newlines_produce_a_taller_box() {
    let mut engine = engine();
    let one = engine.shape("Sapporo", 14.0).expect("shapes");
    let two = engine.shape("Sap\nporo", 14.0).expect("shapes");
    assert!(two.height_px() > one.height_px());
    assert!(two.width_px() < one.width_px());
}

// --- print/text v1.5 (D-A7): orientation ---

#[test]
fn shape_oriented_horizontal_returns_the_same_arc_as_shape_weighted() {
    // The regression fence for every pre-v1.5 caller: the four historical
    // entry points are `shape_oriented(.., Horizontal)` by delegation.
    let mut engine = engine();
    let weighted = engine
        .shape_weighted("Kyoto", 14.0, LabelWeight::Regular)
        .expect("shapes");
    let oriented = engine
        .shape_oriented(
            "Kyoto",
            14.0,
            LabelWeight::Regular,
            LabelOrientation::Horizontal,
        )
        .expect("shapes");
    assert!(std::sync::Arc::ptr_eq(&weighted, &oriented));
    assert!(std::sync::Arc::ptr_eq(
        &engine.shape("Kyoto", 14.0).expect("shapes"),
        &oriented
    ));
    assert_eq!(engine.cached_labels(), 1, "ONE entry for all three");
    let measured = engine.measure("Kyoto", 14.0).expect("measures");
    let measured_oriented = engine
        .measure_oriented(
            "Kyoto",
            14.0,
            LabelWeight::Regular,
            LabelOrientation::Horizontal,
        )
        .expect("measures");
    assert_eq!(measured, measured_oriented);
    assert!(LabelOrientation::default().is_horizontal());
}

#[test]
fn a_refused_vertical_request_shares_the_horizontal_entry() {
    // "Kyoto" is Latin: every character is UAX #50 Rotated, so the
    // FONT-FREE half of the ladder refuses and the key stores Horizontal.
    let mut engine = engine();
    let horizontal = engine.shape("Kyoto", 14.0).expect("shapes");
    let vertical = engine
        .shape_oriented(
            "Kyoto",
            14.0,
            LabelWeight::Regular,
            LabelOrientation::Vertical,
        )
        .expect("shapes");
    assert!(
        std::sync::Arc::ptr_eq(&horizontal, &vertical),
        "a refused vertical request draws the horizontal label",
    );
    assert_eq!(engine.cached_labels(), 1, "and shares its ONE cache entry");
    // Measuring agrees with it, rather than reporting a column.
    let measured = engine
        .measure_oriented(
            "Kyoto",
            14.0,
            LabelWeight::Regular,
            LabelOrientation::Vertical,
        )
        .expect("measures");
    assert!((measured[0] - horizontal.width_px()).abs() < 0.01);
}

#[test]
fn a_font_dependent_refusal_still_shares_one_label_under_both_keys() {
    // The bundled Noto has no `vmtx`, so a label that clears the
    // font-free half still refuses at build time. It must cost ONE
    // shaping and ONE set of atlas slots, reachable from both keys.
    let mut engine = engine();
    let text = "\u{A7}\u{B1}"; // both Upright, both in a Latin font
    let vertical = engine
        .shape_oriented(text, 14.0, LabelWeight::Regular, LabelOrientation::Vertical)
        .expect("shapes");
    let horizontal = engine.shape(text, 14.0).expect("shapes");
    assert!(
        std::sync::Arc::ptr_eq(&vertical, &horizontal),
        "one label, two keys",
    );
    assert_eq!(engine.cached_labels(), 2, "the two keys, one Arc");
    // And a second vertical request hits rather than re-planning.
    let again = engine
        .shape_oriented(text, 14.0, LabelWeight::Regular, LabelOrientation::Vertical)
        .expect("shapes");
    assert!(std::sync::Arc::ptr_eq(&vertical, &again));
    assert_eq!(engine.cached_labels(), 2);
}

#[test]
fn the_cache_key_carries_the_orientation_so_vertical_is_never_served_the_horizontal_label() {
    // The weight bug's twin (D-W3): a `(text, size, weight)`-keyed cache
    // would hand whichever orientation asked first to both. There is no
    // vmtx-bearing font in the suite, so the proof is structural — the
    // key type carries the field and the two requests resolve through it.
    let mut engine = engine();
    let text = "\u{A7}";
    let horizontal = engine.shape(text, 14.0).expect("shapes");
    assert_eq!(engine.cached_labels(), 1);
    let vertical = engine
        .shape_oriented(text, 14.0, LabelWeight::Regular, LabelOrientation::Vertical)
        .expect("shapes");
    // Refused (no vmtx), so it is served the horizontal label — but under
    // its OWN key, which is what a future accepted request will occupy.
    assert!(std::sync::Arc::ptr_eq(&horizontal, &vertical));
    assert_eq!(
        engine.cached_labels(),
        2,
        "two keys, one for each orientation"
    );
}

#[test]
fn a_horizontal_only_session_never_builds_a_shaper() {
    let mut engine = engine();
    let _ = engine.shape("Osaka", 14.0).expect("shapes");
    assert!(
        engine.shaper.is_none(),
        "the vertical shaper is built lazily, on the first vertical label",
    );
    // A vertical request that the font-free half refuses does not need
    // one either.
    let _ = engine
        .shape_oriented(
            "Osaka",
            14.0,
            LabelWeight::Regular,
            LabelOrientation::Vertical,
        )
        .expect("shapes");
    assert!(
        engine.shaper.is_none(),
        "a font-free refusal shapes nothing"
    );
    // One that reaches the font-dependent rungs does.
    let _ = engine
        .shape_oriented(
            "\u{A7}",
            14.0,
            LabelWeight::Regular,
            LabelOrientation::Vertical,
        )
        .expect("shapes");
    assert!(engine.shaper.is_some());
}

#[test]
fn the_vertical_refusal_log_memo_is_cleared_with_every_other_cache() {
    let mut engine = engine();
    let _ = engine
        .shape_oriented(
            "Osaka",
            14.0,
            LabelWeight::Regular,
            LabelOrientation::Vertical,
        )
        .expect("shapes");
    assert_eq!(engine.vertical_refusals_logged.len(), 1, "one reason, once");
    let _ = engine
        .shape_oriented(
            "Sendai",
            14.0,
            LabelWeight::Regular,
            LabelOrientation::Vertical,
        )
        .expect("shapes");
    assert_eq!(
        engine.vertical_refusals_logged.len(),
        1,
        "not once per label"
    );
    engine.clear();
    assert!(
        engine.vertical_refusals_logged.is_empty(),
        "a new generation may have a new answer",
    );
}

#[test]
fn the_face_chain_is_the_same_order_the_bold_pipeline_is_built_in() {
    let mut engine = engine();
    assert_eq!(
        engine.chain_for(LabelWeight::Regular).len(),
        1,
        "primary only"
    );
    engine.set_fallback_fonts(vec![font()]);
    assert_eq!(engine.chain_for(LabelWeight::Regular).len(), 2);
    engine.set_bold_fonts(vec![pretend_bold()]);
    let bold = engine.chain_for(LabelWeight::Bold);
    assert_eq!(
        bold.len(),
        3,
        "bold ++ primary ++ fallbacks, never shrinking"
    );
    // The regular chain never sees a bold face.
    assert_eq!(engine.chain_for(LabelWeight::Regular).len(), 2);
    // And every entry is a live handle to real font bytes.
    for face in bold {
        assert!(!face.is_empty());
    }
}

/// A Windows CJK face, or [`None`] on a machine without one. The suite
/// ships no font with a `vmtx` table, so every claim about an ACCEPTED
/// vertical label is necessarily live.
fn windows_cjk() -> Option<(&'static str, Vec<u8>)> {
    ["meiryo.ttc", "YuGothM.ttc", "msgothic.ttc"]
        .into_iter()
        .find_map(|name| {
            std::fs::read(format!("C:/Windows/Fonts/{name}"))
                .ok()
                .map(|bytes| (name, bytes))
        })
}

#[test]
#[ignore = "reads C:/Windows/Fonts; the only font with a vmtx table"]
fn live_windows_a_vertical_cjk_label_stacks_and_collides() {
    let Some((name, bytes)) = windows_cjk() else {
        return;
    };
    let mut engine = LabelEngine::new(bytes).expect("a Windows CJK face parses");
    let text = "\u{6771}\u{4EAC}\u{90FD}";
    let size = 16.0_f32;
    let vertical = engine
        .shape_oriented(text, size, LabelWeight::Regular, LabelOrientation::Vertical)
        .expect("stacks");
    let horizontal = engine.shape(text, size).expect("shapes");
    assert!(
        !std::sync::Arc::ptr_eq(&vertical, &horizontal),
        "{name}: the two orientations are two labels",
    );
    assert_eq!(engine.cached_labels(), 2, "{name}");
    assert_eq!(vertical.glyphs().len(), 3, "{name}: one cell per character");
    // The box: one em wide and the summed pitch tall — a TALL box, which
    // is what makes a vertical label collide down a column and not across
    // a line. The placement pass needs no other change.
    assert_eq!(vertical.size_px()[0], size, "{name}: one em wide");
    assert!(
        vertical.height_px() > vertical.width_px(),
        "{name}: {:?} must be tall",
        vertical.size_px(),
    );
    assert!(
        horizontal.width_px() > horizontal.height_px(),
        "{name}: and the horizontal one wide",
    );
    assert!(
        (vertical.height_px() - horizontal.width_px()).abs() < size,
        "{name}: the column is about as tall as the line is wide",
    );
    // Cells stack downward and stay inside the one-em column.
    let mut previous = f32::NEG_INFINITY;
    for glyph in vertical.glyphs() {
        assert!(glyph.offset_px[1] > previous, "{name}: cells must descend");
        previous = glyph.offset_px[1];
        assert_eq!(glyph.offset_px[0], glyph.offset_px[0].round());
        assert_eq!(glyph.offset_px[1], glyph.offset_px[1].round());
    }
    // Measuring agrees with shaping, as it does horizontally.
    let measured = engine
        .measure_oriented(text, size, LabelWeight::Regular, LabelOrientation::Vertical)
        .expect("measures");
    assert_eq!(measured, vertical.size_px(), "{name}");
}

#[test]
#[ignore = "reads C:/Windows/Fonts; the shared-atlas claim for orientation"]
fn live_windows_both_orientations_pack_into_the_one_shared_atlas() {
    let Some((name, bytes)) = windows_cjk() else {
        return;
    };
    let mut engine = LabelEngine::new(bytes).expect("a Windows CJK face parses");
    let text = "\u{300C}\u{6771}\u{300D}";
    let horizontal = engine.shape(text, 16.0).expect("shapes");
    let vertical = engine
        .shape_oriented(text, 16.0, LabelWeight::Regular, LabelOrientation::Vertical)
        .expect("stacks");
    for glyph in horizontal.glyphs().iter().chain(vertical.glyphs()) {
        assert_eq!(
            engine.atlas().get(&glyph.key),
            Some(glyph.slot),
            "{name}: every glyph of either orientation lives in the ONE atlas",
        );
    }
    assert_eq!(
        horizontal.generation(),
        vertical.generation(),
        "{name}: one atlas, one generation",
    );
    // The `vert` substitution gives most vertical cells a different gid,
    // so the two orientations pack side by side with no key change.
    let bracket = vertical.glyphs().first().map(|glyph| glyph.key.gid);
    let flat = horizontal.glyphs().first().map(|glyph| glyph.key.gid);
    assert_ne!(bracket, flat, "{name}: the bracket substitutes");
}

#[test]
#[ignore = "reads C:/Windows/Fonts; one engine, one shaper, two cache misses"]
fn live_windows_one_shaper_shapes_a_vertical_label_the_same_way_twice() {
    let Some((name, bytes)) = windows_cjk() else {
        return;
    };
    let mut engine = LabelEngine::new(bytes).expect("a Windows CJK face parses");
    let text = "\u{300C}\u{3042}\u{3001}\u{300D}";
    let first = engine
        .shape_oriented(text, 16.0, LabelWeight::Regular, LabelOrientation::Vertical)
        .expect("stacks");
    let first_gids: Vec<u16> = first.glyphs().iter().map(|g| g.key.gid).collect();
    // Force a cache MISS through the same shaper: clear, then re-request.
    engine.clear();
    let second = engine
        .shape_oriented(text, 16.0, LabelWeight::Regular, LabelOrientation::Vertical)
        .expect("stacks again");
    let second_gids: Vec<u16> = second.glyphs().iter().map(|g| g.key.gid).collect();
    assert_eq!(first_gids, second_gids, "{name}: one shaper, one answer");
    assert_eq!(first.size_px(), second.size_px(), "{name}");
}

// --- audit fixes: bounded glyphs, an adaptive cache, the rasteriser's key ---

#[test]
fn a_glyph_whose_ink_spans_thousands_of_ems_is_refused_before_it_is_rasterised() {
    // `fontdue` allocates `width * height` coverage bytes from the outline's
    // own bounds, so the refusal has to happen before it is called at all.
    // Noto is a well-behaved face and stands in for the arithmetic; a hostile
    // one differs only in reaching the limit at a sane em.
    let mut engine = engine();
    let bytes: Arc<[u8]> = Arc::from(font());
    let face = ttf_parser::Face::parse(&bytes, 0).expect("bundled Noto parses");
    let cap = face.glyph_index('H').expect("Noto has an H").0;
    let space = face.glyph_index(' ').expect("Noto has a space").0;
    let mut faces = FaceCache::new();

    assert!(
        engine.glyph_ink_is_bounded(&bytes, cap, MAX_LABEL_SIZE_PX, &mut faces),
        "a real face at the largest label size must never be refused",
    );
    assert!(
        !engine.glyph_ink_is_bounded(&bytes, cap, 100_000.0, &mut faces),
        "an ink box past the largest atlas is refused unrasterised",
    );
    assert!(
        engine.glyph_ink_is_bounded(&bytes, space, MAX_LABEL_SIZE_PX, &mut faces),
        "a glyph with no outline allocates nothing and is not refused",
    );
}

#[test]
fn the_largest_allowed_label_still_draws() {
    // The other half of the bound: no false positive at the extreme a caller
    // is actually allowed to ask for.
    let mut engine = engine();
    let label = engine.shape("H", MAX_LABEL_SIZE_PX).expect("shapes");
    assert_eq!(label.glyphs().len(), 1, "a 512 px H is not a hostile glyph");
    assert_eq!(engine.cached_labels(), 1);
}

#[test]
fn a_frame_with_more_labels_than_the_cache_grows_it_instead_of_thrashing() {
    // The placer shapes every candidate, so a pass over the tile list is a
    // cyclic access pattern; strict LRU over a working set larger than the
    // cache turns that into a 100 % miss rate, frame after frame.
    let mut engine = engine();
    assert_eq!(engine.cache_capacity(), DEFAULT_LABEL_CACHE);
    let texts: Vec<String> = (0..DEFAULT_LABEL_CACHE + 200)
        .map(|index| format!("label {index}"))
        .collect();
    for _ in 0..3 {
        for text in &texts {
            let _ = engine.shape(text, 12.0).expect("shapes");
        }
    }
    assert!(
        engine.cache_capacity() > DEFAULT_LABEL_CACHE,
        "the cache must follow the frame's demand, got {}",
        engine.cache_capacity(),
    );
    assert!(
        engine.cached_labels() >= texts.len(),
        "and then hold a whole pass: {} of {}",
        engine.cached_labels(),
        texts.len(),
    );
    // A fourth pass is now all hits, which is the entire point.
    let first = engine.shape(&texts[0], 12.0).expect("shapes");
    let again = engine.shape(&texts[0], 12.0).expect("shapes");
    assert!(Arc::ptr_eq(&first, &again));
}

#[test]
fn reserving_sizes_the_cache_up_front_and_stays_bounded() {
    let mut engine = engine();
    engine.reserve_labels(1000);
    assert_eq!(engine.cache_capacity(), 2000, "room for the pan ring too");
    engine.reserve_labels(usize::MAX);
    assert_eq!(engine.cache_capacity(), MAX_LABEL_CACHE);
    engine.reserve_labels(1);
    assert_eq!(
        engine.cache_capacity(),
        MAX_LABEL_CACHE,
        "capacity only rises"
    );
}

#[test]
fn an_explicit_capacity_is_never_grown_behind_the_callers_back() {
    let mut engine = LabelEngine::with_capacity(font(), 4).expect("capacity 4 is valid");
    for index in 0..64 {
        let _ = engine
            .shape(&format!("label {index}"), 12.0)
            .expect("shapes");
    }
    assert_eq!(engine.cache_capacity(), 4);
    assert_eq!(engine.cached_labels(), 4);
}

#[test]
fn the_rasterisers_font_key_sees_only_the_first_64_bytes() {
    // Pins the upstream hazard in one line: `oxitext-raster`'s thread-local
    // `fontdue::Font` cache keys on FNV-1a of at most the first 64 bytes, so
    // two files that agree there are ONE parsed face to it. The fix belongs
    // upstream; this asserts what the engine is defending against.
    let mut first = font();
    let mut second = font();
    first.extend_from_slice(b"one");
    second.extend_from_slice(b"two");
    assert_ne!(first, second);
    assert_eq!(
        raster_cache_key(&first),
        raster_cache_key(&second),
        "two different font files, one rasteriser cache key",
    );
    assert_eq!(
        raster_cache_key(&[]),
        0xcbf2_9ce4_8422_2325,
        "the FNV-1a offset basis, i.e. the upstream constant",
    );
}

#[test]
fn two_faces_the_rasteriser_cannot_tell_apart_are_reported_rather_than_drawn() {
    let mut aliased = engine();
    let first: Arc<[u8]> = Arc::from(font());
    let mut tail = font();
    tail.extend_from_slice(b"a different tail");
    let second: Arc<[u8]> = Arc::from(tail);
    assert_eq!(aliased.intern_font(&first), 0);
    assert_eq!(aliased.intern_font(&second), 1);
    assert!(
        aliased.raster_key_collision_logged,
        "an aliasing pair must be said out loud, not drawn silently",
    );

    // A third face collides with the FIRST one, which is the face the
    // rasteriser's cache actually holds; comparing it against whichever face
    // collided most recently would report nothing.
    let mut trio = engine();
    let mut third = font();
    third.extend_from_slice(b"a third tail");
    assert_eq!(trio.intern_font(&first), 0);
    assert_eq!(trio.intern_font(&Arc::from(font())), 1);
    assert!(
        !trio.raster_key_collision_logged,
        "a copy is not a collision"
    );
    assert_eq!(trio.intern_font(&Arc::from(third)), 2);
    assert!(
        trio.raster_key_collision_logged,
        "the first claimant of a key stays the one later faces are checked against",
    );

    // Two handles on identical bytes are the same face: same outlines, same
    // ink, nothing to report. The bold chain does exactly this.
    let mut copies = engine();
    let one: Arc<[u8]> = Arc::from(font());
    let other: Arc<[u8]> = Arc::from(font());
    assert_eq!(copies.intern_font(&one), 0);
    assert_eq!(copies.intern_font(&other), 1);
    assert!(!copies.raster_key_collision_logged);
}
