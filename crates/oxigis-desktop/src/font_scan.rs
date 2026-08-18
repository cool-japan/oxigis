//! Native CJK font discovery: find the system faces that can draw the names
//! the bundled Latin font cannot.
//!
//! # Why the desktop shell and not `oxigis-render`
//!
//! [`oxigis_render::label::LabelEngine`] takes font *bytes* and never touches
//! the file system — that is what lets the same engine run in a browser, where
//! there is no file system to touch. Reading fonts off disk is therefore a
//! shell responsibility, and this is the native shell's half of it. The browser
//! shell fetches its fallback over HTTP instead.
//!
//! # What "find CJK faces" means here
//!
//! [`oxifont_discovery`] reports the directories the OS keeps fonts in; this
//! module walks them and matches **file names** against a list of the faces
//! known to carry CJK coverage on Linux, macOS and Windows
//! ([`CJK_FONT_STEMS`]). It does not parse `cmap` tables, which would mean
//! opening every font on the system at startup; the only content check is a
//! four-byte signature sniff on candidate files (see below).
//!
//! No single platform face covers Japanese, Korean, Simplified and
//! Traditional Chinese at once — Meiryo has no Hangul, Malgun Gothic no kana —
//! so each stem is tagged with the [`CjkScript`] repertoire it serves and the
//! scan assembles a **fallback chain**: the best-ranked candidate per script,
//! ordered by rank. A pan-CJK face (Noto Sans CJK, the full Source Han
//! builds) covers everything after it, so the chain is cut right after the
//! first pan face; on a Linux box with Noto Sans CJK that means one file, on
//! stock Windows it means up to four (Meiryo + Malgun + YaHei + JhengHei,
//! roughly 60 MB — a one-time cost a desktop GIS drawing multinational
//! basemap labels can afford).
//!
//! # How candidates are ranked
//!
//! Within one tier of directories the walk keeps, per script, the candidate
//! with the lowest [`CJK_FONT_STEMS`] index; ties on rank go to the fewest
//! characters after the matched stem, so `meiryo.ttc` beats `meiryob.ttc`
//! (Bold) and `msyh.ttc` beats `msyhbd.ttc` no matter which one the directory
//! yields first. A remaining tie goes to the file seen first, and the walk is
//! breadth-first across the whole tier, so a shallower file genuinely wins.
//! Symlinks are followed when classifying entries (link farms à la Nix or a
//! hand-made `~/.fonts` are real font dirs); a directory-link cycle cannot
//! run away because the depth bound applies to every enqueued hop.
//!
//! Both tiers are always scanned and merged **per script slot**: a
//! user-installed face wins its own script over anything the system ships,
//! but scripts the user tier says nothing about still fill from the system
//! tier — installing one Japanese font must not cost the map its Korean
//! labels. User-tier picks also sort ahead of system-tier picks in the final
//! chain, so a user face shapes its clusters even when a system pan face
//! follows as the safety net.
//!
//! # The bytes are checked, not just the name
//!
//! A file called `NotoSansCJK-Regular.otf` is not necessarily a font, and one
//! HTML error page saved under a well-ranked name must not cost the map its
//! CJK labels. Candidates are therefore signature-sniffed (first four bytes,
//! [`is_sfnt`] / [`is_ttc`]) *at selection time*, so junk loses to a valid
//! lower-ranked file instead of poisoning the chain; [`read_cjk_font`]
//! re-checks size (on the open handle) and signature at read time in case the
//! file changed in between.
//!
//! # `.ttc` is passed through whole
//!
//! TrueType Collections hold several faces in one file behind a `ttcf` header.
//! This matters on Windows, which ships its CJK staples almost exclusively as
//! collections (`msgothic.ttc`, `meiryo.ttc`, `YuGothM.ttc`, `msyh.ttc`,
//! `msjh.ttc`, `simsun.ttc`) — skipping them would leave a stock Windows
//! install with no CJK fallback at all.
//!
//! No face-extraction step is needed: every consumer in the label pipeline
//! already resolves **face 0** of a collection from the raw bytes. In
//! `oxitext` 0.2.1's `pure` backend, font validation and metrics go through
//! `oxifont`'s `ParsedFace::parse(bytes, 0)` (which has an explicit `ttcf`
//! branch), shaping — including the per-cluster fallback selection, which
//! re-shapes each `.notdef` cluster through the chain and keeps the first
//! font yielding a non-zero glyph id — is swash's
//! `FontRef::from_index(bytes, 0)`, and rasterisation is fontdue's
//! `Font::from_bytes` with its default `collection_index: 0` (ttf_parser
//! underneath). Face 0 of each Windows CJK collection is the canonical family
//! face (MS Gothic, Meiryo, Yu Gothic Medium, Microsoft YaHei, Microsoft
//! JhengHei, SimSun), so index 0 is not just what the pipeline does but also
//! what we want. Verified end-to-end on Windows 11 (2026-07-31): every CJK
//! cluster shaped out of the collection bytes and rasterised with ink through
//! the exact `shape_and_layout` + `rasterize_full` sequence the label engine
//! uses.
//!
//! macOS ships its stock CJK faces as collections too, and face 0 matters
//! there for a different reason: it decides which *script* a multi-face
//! file counts as. Confirmed with fontTools (macOS 26.6.1): face 0 of every
//! native-named `ヒラギノ角ゴシック W‹n›.ttc` is plain "Hiragino Sans"
//! (never the private `.Hiragino … Interface` face UI text uses), and face 0
//! of `STHeiti Light/Medium.ttc` is "Heiti TC" — Traditional, not the
//! "Heiti SC" face 1 right behind it — exactly why [`CJK_FONT_STEMS`] tags
//! `stheiti` as [`CjkScript::TraditionalChinese`], not Simplified.
//! `macos_fills_the_japanese_and_traditional_chinese_slots_with_real_ink`
//! below is the Windows paragraph's proof run for real, and asserted rather
//! than merely permitted to pass either way, on every Mac these tests run.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};

/// The CJK repertoire a known face serves, used to assemble a fallback chain
/// with one face per script.
///
/// `PanCjk` marks faces covering all four repertoires at once; everything
/// ranked after a pan face in the chain would be redundant and is dropped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CjkScript {
    /// Japanese + Korean + Simplified + Traditional in one face.
    PanCjk,
    /// Japanese (kanji + kana).
    Japanese,
    /// Korean (Hangul, plus Hanja in the platform faces).
    Korean,
    /// Simplified Chinese.
    SimplifiedChinese,
    /// Traditional Chinese.
    TraditionalChinese,
}

impl CjkScript {
    /// Number of distinct script slots the scan tracks.
    const COUNT: usize = 5;

    /// Slot index in the per-script candidate table.
    const fn slot(self) -> usize {
        match self {
            Self::PanCjk => 0,
            Self::Japanese => 1,
            Self::Korean => 2,
            Self::SimplifiedChinese => 3,
            Self::TraditionalChinese => 4,
        }
    }
}

/// File-name stems, lower-cased, of the fonts known to carry CJK coverage,
/// each tagged with the script repertoire it serves.
///
/// Matched as a *prefix* of the file stem with separators stripped, so
/// `NotoSansCJKjp-Regular.otf`, `Noto Sans CJK JP.ttc` and
/// `NotoSansJP-Regular.ttf` are all recognised. Ordered by preference and the
/// order is enforced: the scan keeps the earliest-ranked candidate per
/// script. Because matching is first-entry-wins, a specific stem must appear
/// before any stem that prefixes it (`sourcehansansjp` before
/// `sourcehansans`, `hiraginosansgb` before `hiragino`,
/// `droidsansfallbackfull` before `droidsansfallback`).
///
/// The Windows block prefers Meiryo and Yu Gothic over MS Gothic: all three
/// cover JIS, but MS Gothic carries embedded bitmap strikes that read poorly
/// at map-label sizes. `yugothm`/`yugothr` appear before the bare `yugoth` so
/// the Medium/Regular collections outrank `YuGothB.ttc` (Bold) on rank, not
/// just on the shorter-suffix tie-break. Microsoft JhengHei is listed by its
/// on-disk stem `msjh` (the `microsoftjhenghei` entry only catches renamed
/// copies; Windows never ships a file by that name).
pub const CJK_FONT_STEMS: &[(&str, CjkScript)] = &[
    // Pan-CJK Noto (Linux distributions, and the upstream Google releases).
    ("notosanscjk", CjkScript::PanCjk),
    ("notoserifcjk", CjkScript::PanCjk),
    // Adobe Source Han. The regional SubsetOTF releases
    // (SourceHanSans{JP,KR,CN,TW,HK}) carry ONLY their region's repertoire —
    // the JP subset has no Hangul — so they are per-script and must precede
    // the bare stem, which then catches the language-specific full builds
    // (SourceHanSansSC/TC/HC/K…: regional default glyphs, full pan cmap).
    ("sourcehansansjp", CjkScript::Japanese),
    ("sourcehansanskr", CjkScript::Korean),
    ("sourcehansanscn", CjkScript::SimplifiedChinese),
    ("sourcehansanstw", CjkScript::TraditionalChinese),
    ("sourcehansanshk", CjkScript::TraditionalChinese),
    ("sourcehanserifjp", CjkScript::Japanese),
    ("sourcehanserifkr", CjkScript::Korean),
    ("sourcehanserifcn", CjkScript::SimplifiedChinese),
    ("sourcehanseriftw", CjkScript::TraditionalChinese),
    ("sourcehanserifhk", CjkScript::TraditionalChinese),
    ("sourcehansans", CjkScript::PanCjk),
    ("sourcehanserif", CjkScript::PanCjk),
    // Per-script Noto (Google Fonts packaging, Flatpak runtimes; recent
    // Windows 11 ships NotoSansJP-VF.ttf too).
    ("notosansjp", CjkScript::Japanese),
    ("notosanskr", CjkScript::Korean),
    ("notosanssc", CjkScript::SimplifiedChinese),
    ("notosanstc", CjkScript::TraditionalChinese),
    ("notosanshk", CjkScript::TraditionalChinese),
    // Android / Chrome OS. Only the Full build is genuinely pan-CJK: the
    // space-reduced DroidSansFallback shipped on devices keeps just the
    // KS X 1001 common Hangul (~2,350 of 11,172 syllables) and drops CJK
    // Extension A, so it is tagged with its densest repertoire instead of
    // PanCjk — a pan tag would truncate the chain and cost a coexisting real
    // Korean face its full coverage.
    ("droidsansfallbackfull", CjkScript::PanCjk),
    ("droidsansfallback", CjkScript::SimplifiedChinese),
    ("droidsansjapanese", CjkScript::Japanese),
    // macOS. The native Japanese Hiragino collections lead the whole
    // section, same convention as Windows below (meiryo/yugoth* rank ahead
    // of msyh/msjh): one shared, ordered fallback chain keeps the first
    // font yielding a non-zero glyph id, no per-run script routing, so a
    // kanji/kana ALSO covered by Hiragino Sans GB (confirmed by cmap probe)
    // renders in Simplified Chinese forms unless Japanese is tried first.
    // They ship under native names (`ヒラギノ角ゴシック W3.ttc` and
    // siblings), matched by their real on-disk stem: `normalized_font_stem`
    // keeps Unicode letters and recomposes kana voicing (APFS/HFS+ hand
    // `read_dir` names back canonically decomposed — `ギ`/`ゴ` as base kana
    // plus a combining mark, confirmed empirically — see `recompose_kana`).
    // Face 0 of each is confirmed (fontTools, macOS 26.6.1) to be plain
    // "Hiragino Sans" / "Hiragino Maru Gothic Pro" / "Hiragino Mincho
    // ProN", never the private `.Hiragino … Interface` face. Kaku Gothic
    // ships one file per weight (W0..W9) tying on trailing-character count
    // once normalised, so W3 (Regular-equivalent) is listed ahead of the
    // bare stem, W4 as a fallback, or the winner is whichever file readdir
    // enumerates first; Maru Gothic/Mincho need no such entry (one weight
    // each), and Sans outranks Mincho (serif) since the bundled Latin face
    // is a sans.
    //
    // Hiragino Sans GB is the only stock ASCII-named Hiragino file and is
    // *Simplified* Chinese; listed before the bare `hiragino`, which
    // catches ASCII-renamed user copies of the Japanese faces — a
    // name-matching necessity, independent of the JP-leads ordering above.
    // PingFang, the modern default, is an on-demand "mobile asset" this
    // scan cannot reach and is listed only for a manually installed copy;
    // face 0 is "PingFang HK", so it fills *Traditional*. STHeiti
    // Light/Medium.ttc live in the ordinary directory this scan already
    // walks; face 0 of both is "Heiti TC" — the stock, always-reachable
    // Traditional candidate — with Medium listed ahead of the bare stem so
    // Light (hairline, wrong for labels) does not win the shorter-suffix
    // tie. Osaka is the legacy ASCII Japanese face, also now a mobile
    // asset, kept for older systems.
    //
    // Filling Japanese and Traditional on a stock install adds real weight:
    // Hiragino Sans W3 (~7.8 MB) + STHeiti (~55 MB), on top of the Korean
    // and Simplified faces already found (~55 MB + ~23 MB), puts a stock
    // chain around 140 MB — over double the ~60 MB the module docs above
    // call affordable, and a correct consequence of these scripts no
    // longer resolving to nothing.
    ("ヒラギノ角ゴシックw3", CjkScript::Japanese),
    ("ヒラギノ角ゴシックw4", CjkScript::Japanese),
    ("ヒラギノ角ゴシック", CjkScript::Japanese),
    ("ヒラギノ丸ゴ", CjkScript::Japanese),
    ("ヒラギノ明朝", CjkScript::Japanese),
    ("pingfang", CjkScript::TraditionalChinese),
    ("hiraginosansgb", CjkScript::SimplifiedChinese),
    ("stheitimedium", CjkScript::TraditionalChinese),
    ("stheiti", CjkScript::TraditionalChinese),
    ("hiragino", CjkScript::Japanese),
    ("osaka", CjkScript::Japanese),
    ("applesdgothicneo", CjkScript::Korean),
    // Windows (see the doc comment above for the ordering rationale).
    ("meiryo", CjkScript::Japanese),
    ("yugothm", CjkScript::Japanese),
    ("yugothr", CjkScript::Japanese),
    ("yugoth", CjkScript::Japanese),
    ("msgothic", CjkScript::Japanese),
    ("malgun", CjkScript::Korean),
    ("msyh", CjkScript::SimplifiedChinese),
    ("msjh", CjkScript::TraditionalChinese),
    ("simsun", CjkScript::SimplifiedChinese),
    ("simhei", CjkScript::SimplifiedChinese),
    ("microsoftjhenghei", CjkScript::TraditionalChinese),
];

/// Latin **bold** faces, in preference order — the head of the bold chain
/// (print/text v1.4, D-W4).
///
/// The primary label face is the bundled Noto Sans **Regular** and no bold
/// twin is bundled, so without one of these a bold Latin label has nothing to
/// draw with. This list is deliberately short and platform-obvious; a face
/// that is not here simply means bold falls back to Regular with one log,
/// which is the never-synthetic rule working as designed.
///
/// Matching is the same normalised prefix match [`CJK_FONT_STEMS`] uses, and
/// the same fewest-extra-characters tie-break applies, so `arialbd.ttf` beats
/// `arialbdi.ttf` (Bold Italic) without an explicit italic rule.
pub const LATIN_BOLD_FONT_STEMS: &[&str] = &[
    // Windows.
    "segoeuib",
    "arialbd",
    "tahomabd",
    "verdanab",
    // macOS and Linux name the same weight with the word spelled out.
    "arialbold",
    "notosansbold",
    "dejavusansbold",
    "liberationsansbold",
];

/// CJK **bold** faces, tagged exactly like [`CJK_FONT_STEMS`] and scanned
/// through the same per-script-slot machinery (print/text v1.4, D-W4).
///
/// Notable absences, all deliberate: **MS Gothic ships no bold** (Windows
/// synthesises it, which this pipeline never does); SimSun likewise, and
/// `simsunb` is NOT its bold — it is SimSun-ExtB, kept in
/// [`CJK_STEM_EXCLUSIONS`] because it carries no BMP glyph at all. A script
/// with no entry here keeps its regular face through the label engine's
/// never-shrink bold chain, so the line is mixed-weight rather than
/// `.notdef`.
pub const CJK_BOLD_FONT_STEMS: &[(&str, CjkScript)] = &[
    // Pan-CJK first, same as the regular table: it truncates the chain.
    ("notosanscjkbold", CjkScript::PanCjk),
    ("sourcehansansbold", CjkScript::PanCjk),
    ("sourcehanserifbold", CjkScript::PanCjk),
    // Windows.
    ("meiryob", CjkScript::Japanese),
    ("yugothb", CjkScript::Japanese),
    ("malgunbd", CjkScript::Korean),
    ("msyhbd", CjkScript::SimplifiedChinese),
    ("msjhbd", CjkScript::TraditionalChinese),
];

/// Normalised stems that must never match even though they prefix-match a
/// [`CJK_FONT_STEMS`] entry: supplementary-plane-only fonts with **zero** BMP
/// CJK coverage, which would satisfy the name check and then draw every real
/// label as `.notdef`.
///
/// `simsunb` is SimSun-ExtB (CJK Extension B, plane 2 only) and `simsunext`
/// catches SimsunExtG (Extensions G+, plane 3); both ship on stock Windows
/// right next to the real `simsun.ttc`.
pub const CJK_STEM_EXCLUSIONS: &[&str] = &["simsunb", "simsunext"];

/// File extensions accepted as font data the label pipeline can load.
///
/// `.ttc` rides along since the whole pipeline resolves face 0 of a
/// collection; see the [module docs][self].
pub const FONT_EXTENSIONS: &[&str] = &["ttf", "otf", "ttc"];

/// SFNT signatures a single-face font file may start with.
///
/// `0x00010000` is TrueType outlines, `OTTO` is CFF (PostScript) outlines, and
/// `true` is the older Apple TrueType tag. A collection announces itself with
/// [`TTC_SIGNATURE`] instead.
pub const SFNT_SIGNATURES: &[[u8; 4]] = &[[0x00, 0x01, 0x00, 0x00], *b"OTTO", *b"true"];

/// The four-byte header of a TrueType Collection.
pub const TTC_SIGNATURE: [u8; 4] = *b"ttcf";

/// Whether `bytes` begin with a single-face SFNT signature.
///
/// Together with [`is_ttc`], the one thing standing between a mislabelled
/// `.woff2` (or an HTML error page saved with a font's name) and the shaping
/// engine.
#[must_use]
pub fn is_sfnt(bytes: &[u8]) -> bool {
    let Some(magic) = bytes.get(..4) else {
        return false;
    };
    SFNT_SIGNATURES
        .iter()
        .any(|signature| signature.as_slice() == magic)
}

/// Whether `bytes` begin with the TrueType Collection signature.
#[must_use]
pub fn is_ttc(bytes: &[u8]) -> bool {
    bytes.get(..4) == Some(TTC_SIGNATURE.as_slice())
}

/// How deep the scan descends into a font directory.
///
/// Font directories nest by family or by vendor, rarely more than two levels
/// (`/usr/share/fonts/opentype/noto/`), so three is generous and bounds the
/// walk — including through followed directory symlinks — on a machine with
/// a pathological link farm.
pub const MAX_SCAN_DEPTH: usize = 3;

/// Largest font file the scan will read, in bytes.
///
/// A pan-CJK OTF is around 16 MB and the Windows CJK collections run 9–37 MB;
/// 64 MB leaves room for the largest real candidate while refusing to pull an
/// accidental multi-gigabyte file (or the ~190 MB all-weights Source Han
/// super-collection) into memory. Enforced at selection time and again on the
/// open handle at read time ([`read_cjk_font`]).
pub const MAX_FONT_BYTES: u64 = 64 * 1024 * 1024;

/// The user tier: a candidate here wins its script slot outright.
const TIER_USER: usize = 0;
/// The system tier: fills the script slots the user tier left empty.
const TIER_SYSTEM: usize = 1;

/// One scored candidate: which tier found it, its stem rank, how many stem
/// characters its name has beyond the match (the weight-variant tie-breaker),
/// and where it lives.
#[derive(Debug, Clone)]
struct Candidate {
    tier: usize,
    rank: usize,
    extra: usize,
    path: PathBuf,
}

/// Reads and validates one **selected** font, or [`None`].
///
/// Never panics and never propagates an I/O error: a font that vanished, was
/// truncated, was swapped for something that is not a font, or grew past
/// [`MAX_FONT_BYTES`] since the scan is dropped with a debug log. The size
/// bound is re-checked on the open handle and the read itself is capped, so
/// a file racing the scan cannot balloon memory.
///
/// Meant for a path the scan already picked (a winning [`Candidate`], or one
/// of the paths [`find_cjk_font_paths`]/[`find_cjk_bold_font_paths`]
/// returns) — the "loaded" log below is only true at that point. Nothing
/// during ranking may call this: [`consider`] answers signature, size and
/// thin-default weight through bounded probes ([`sniffs_as_font`],
/// [`is_readable_size`], [`fvar_default_weight`]), so a losing multi-
/// megabyte candidate is never pulled fully into memory, or logged loaded.
#[must_use]
pub fn read_cjk_font(path: &Path) -> Option<Vec<u8>> {
    use std::io::Read as _;
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) => {
            tracing::debug!(
                path = %path.display(),
                %error,
                "OxiGIS desktop: a CJK fallback font could not be opened",
            );
            return None;
        }
    };
    let size = match file.metadata() {
        Ok(meta) => meta.len(),
        Err(error) => {
            tracing::debug!(
                path = %path.display(),
                %error,
                "OxiGIS desktop: a CJK fallback font could not be sized",
            );
            return None;
        }
    };
    if size == 0 || size > MAX_FONT_BYTES {
        tracing::debug!(
            path = %path.display(),
            size,
            "OxiGIS desktop: the CJK candidate's size changed out of bounds; dropped",
        );
        return None;
    }
    let mut bytes = Vec::with_capacity(size as usize);
    // The +1 turns "grew past the cap mid-read" into a detectable length.
    if let Err(error) = file.take(MAX_FONT_BYTES + 1).read_to_end(&mut bytes) {
        tracing::debug!(
            path = %path.display(),
            %error,
            "OxiGIS desktop: a CJK fallback font could not be read",
        );
        return None;
    }
    if bytes.len() as u64 > MAX_FONT_BYTES || !(is_sfnt(&bytes) || is_ttc(&bytes)) {
        tracing::debug!(
            path = %path.display(),
            bytes = bytes.len(),
            "OxiGIS desktop: the CJK candidate changed on disk and is no longer loadable; dropped",
        );
        return None;
    }
    tracing::info!(
        path = %path.display(),
        bytes = bytes.len(),
        container = if is_ttc(&bytes) { "ttc (face 0)" } else { "sfnt" },
        stem = cjk_stem_rank(path).map(|rank| CJK_FONT_STEMS[rank].0),
        "OxiGIS desktop: CJK label fallback font loaded",
    );
    Some(bytes)
}

/// Locates the fallback chain in the OS's font directories: per
/// [`CjkScript`] the best candidate (user tier wins its slot, system tier
/// fills the rest), ordered user-first then by rank, truncated after the
/// first pan-CJK face (which makes the rest redundant).
#[must_use]
pub fn find_cjk_font_paths() -> Vec<PathBuf> {
    let user = oxifont_discovery::user_font_dirs();
    let system = oxifont_discovery::system_font_dirs();
    let chain = chain_for_tiers(&user, &system);
    if chain.is_empty() {
        tracing::info!(
            user_directories = user.len(),
            system_directories = system.len(),
            "OxiGIS desktop: no CJK font found; CJK labels will render as .notdef",
        );
    }
    chain.into_iter().map(|candidate| candidate.path).collect()
}

/// Locates the **bold** chain (print/text v1.4, D-W4): the best Latin bold
/// face, then the per-script CJK bold chain, both through exactly the rules
/// [`find_cjk_font_paths`] uses.
///
/// Latin leads because the primary label face is Latin: a bold Latin face at
/// the head means ordinary bold labels are drawn by a face designed for them,
/// and CJK clusters it cannot map fall through to the bold CJK faces behind
/// it — and past those, to the whole REGULAR chain, which the label engine
/// appends (never-shrink). An empty result is not a failure: bold labels then
/// draw Regular with one log.
#[must_use]
pub fn find_cjk_bold_font_paths() -> Vec<PathBuf> {
    let user = oxifont_discovery::user_font_dirs();
    let system = oxifont_discovery::system_font_dirs();
    let mut chain = latin_bold_for_tiers(&user, &system)
        .into_iter()
        .collect::<Vec<_>>();
    chain.extend(
        chain_for_tiers_of(&user, &system, CJK_BOLD_FONT_STEMS)
            .into_iter()
            .map(|candidate| candidate.path),
    );
    if chain.is_empty() {
        tracing::info!(
            user_directories = user.len(),
            system_directories = system.len(),
            "OxiGIS desktop: no bold font found; Bold labels will draw Regular",
        );
    }
    chain
}

/// The single best Latin bold face across both tiers, user tier first.
///
/// One slot rather than a per-script array: [`LATIN_BOLD_FONT_STEMS`] entries
/// all cover the same repertoire, so a second one would only add megabytes.
fn latin_bold_for_tiers(user_dirs: &[PathBuf], system_dirs: &[PathBuf]) -> Option<PathBuf> {
    let mut best: Option<Candidate> = None;
    for (tier, dirs) in [(TIER_USER, user_dirs), (TIER_SYSTEM, system_dirs)] {
        walk_tier(dirs, &mut |path| {
            let Some((rank, extra)) = latin_bold_stem_match(path) else {
                return;
            };
            let improves = match best.as_ref() {
                None => true,
                Some(held) => (tier, rank, extra) < (held.tier, held.rank, held.extra),
            };
            if !improves || !is_readable_size(path) || !sniffs_as_font(path) {
                return;
            }
            best = Some(Candidate {
                tier,
                rank,
                extra,
                path: path.to_path_buf(),
            });
        });
    }
    best.map(|candidate| candidate.path)
}

/// [`stem_match_in`] against [`LATIN_BOLD_FONT_STEMS`], which carries no
/// script tag.
fn latin_bold_stem_match(path: &Path) -> Option<(usize, usize)> {
    let normalized = normalized_font_stem(path)?;
    LATIN_BOLD_FONT_STEMS
        .iter()
        .position(|known| normalized.starts_with(known))
        .map(|rank| (rank, normalized.len() - LATIN_BOLD_FONT_STEMS[rank].len()))
}

/// Scans both tiers and merges them per script slot: a user candidate keeps
/// its slot no matter what the system tier offers; empty slots fill from the
/// system scan. Installing one Japanese font must not cost the map the
/// Korean coverage the system already ships.
fn chain_for_tiers(user_dirs: &[PathBuf], system_dirs: &[PathBuf]) -> Vec<Candidate> {
    chain_for_tiers_of(user_dirs, system_dirs, CJK_FONT_STEMS)
}

/// [`chain_for_tiers`] over an explicit stem table, so the regular
/// ([`CJK_FONT_STEMS`]) and bold ([`CJK_BOLD_FONT_STEMS`]) scans share every
/// rule — tiers, ranking, the pan truncation and the thin-default demotion —
/// instead of growing a second set that could drift.
fn chain_for_tiers_of(
    user_dirs: &[PathBuf],
    system_dirs: &[PathBuf],
    stems: &[(&str, CjkScript)],
) -> Vec<Candidate> {
    let mut slots = scan_tier(user_dirs, TIER_USER, stems);
    let system = scan_tier(system_dirs, TIER_SYSTEM, stems);
    for (slot, from_system) in slots.iter_mut().zip(system) {
        if slot.is_none() {
            *slot = from_system;
        }
    }
    assemble_chain(slots, stems)
}

/// Orders the per-script picks — user tier first, then rank, then suffix
/// length — and drops everything after the first pan-CJK face, whose
/// repertoire covers it. A user face therefore still precedes a system pan
/// face and shapes its own script's clusters first.
fn assemble_chain(
    slots: [Option<Candidate>; CjkScript::COUNT],
    stems: &[(&str, CjkScript)],
) -> Vec<Candidate> {
    let mut picks: Vec<Candidate> = slots.into_iter().flatten().collect();
    picks.sort_by_key(|candidate| (candidate.tier, candidate.rank, candidate.extra));
    if let Some(position) = picks
        .iter()
        // A demoted rank is out of the table's range; such a candidate is a
        // thin-default variable face, never a pan truncation point.
        .position(|candidate| {
            stems
                .get(candidate.rank)
                .is_some_and(|(_, script)| *script == CjkScript::PanCjk)
        })
    {
        picks.truncate(position + 1);
    }
    picks
}

/// Breadth-first walk over one whole tier of font directories, keeping the
/// best candidate per script slot.
///
/// Level order across the entire tier: every directory's own files are
/// considered before any subdirectory anywhere is opened, so a shallower file
/// wins a full tie against a deeper one. Depth is bounded by
/// [`MAX_SCAN_DEPTH`] for every enqueued hop, which also bounds followed
/// directory symlinks.
fn scan_tier(
    dirs: &[PathBuf],
    tier: usize,
    stems: &[(&str, CjkScript)],
) -> [Option<Candidate>; CjkScript::COUNT] {
    let mut slots: [Option<Candidate>; CjkScript::COUNT] = [None, None, None, None, None];
    walk_tier(dirs, &mut |path| consider(&mut slots, tier, path, stems));
    slots
}

/// The walk itself, handing every FILE it reaches to `visit` — shared by the
/// per-script scan and the Latin bold scan so the two cannot disagree about
/// depth bounds, symlink handling or directory order.
///
/// `visited` dedupes by canonicalised identity, not by the queued path: a
/// directory symlink pointing at an ancestor (or two link-farm aliases of
/// one real directory) would otherwise be re-read, and every file in it
/// re-visited, once per alias reachable within [`MAX_SCAN_DEPTH`]. The
/// depth-bounded BFS already queues the shallowest alias first, so this
/// only discards true duplicates, never the shallow-wins property the rest
/// of the module relies on. A path that fails to canonicalise (permission
/// denied on an ancestor, removed mid-walk) falls back to the as-given one,
/// reaching the `read_dir` error below rather than being dropped here.
fn walk_tier(dirs: &[PathBuf], visit: &mut dyn FnMut(&Path)) {
    let mut queue: VecDeque<(PathBuf, usize)> = dirs.iter().map(|dir| (dir.clone(), 0)).collect();
    let mut visited: HashSet<PathBuf> = HashSet::new();
    while let Some((dir, depth)) = queue.pop_front() {
        let identity = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
        if !visited.insert(identity) {
            continue;
        }
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            // Unreadable or absent directories are the norm, not an error:
            // the discovery crate reports *candidate* locations.
            Err(error) => {
                tracing::debug!(path = %dir.display(), %error, "OxiGIS desktop: font directory skipped");
                continue;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => {
                    if depth < MAX_SCAN_DEPTH {
                        queue.push_back((path, depth + 1));
                    }
                }
                Ok(kind) if kind.is_file() => visit(&path),
                // Symlinks (a hand-made `~/.fonts`, Nix/home-manager link
                // farms): classify by following them. `file_type()` never
                // traverses links, so without this arm a linked font is
                // silently invisible.
                _ => match std::fs::metadata(&path) {
                    Ok(meta) if meta.is_dir() && depth < MAX_SCAN_DEPTH => {
                        queue.push_back((path, depth + 1));
                    }
                    Ok(meta) if meta.is_file() => visit(&path),
                    _ => {}
                },
            }
        }
    }
}

/// Scores `path` and stores it in its script slot if it beats the holder.
///
/// The expensive checks (a `stat` and a four-byte read) run only when the
/// name alone would win, so the walk stays cheap: on a real system fewer than
/// a dozen files ever reach the sniff.
fn consider(
    slots: &mut [Option<Candidate>; CjkScript::COUNT],
    tier: usize,
    path: &Path,
    stems: &[(&str, CjkScript)],
) {
    let Some((rank, extra)) = stem_match_in(path, stems) else {
        return;
    };
    let slot = &mut slots[stems[rank].1.slot()];
    let improves = match slot.as_ref() {
        None => true,
        Some(held) => (rank, extra) < (held.rank, held.extra),
    };
    if !improves || !is_readable_size(path) {
        return;
    }
    if !sniffs_as_font(path) {
        tracing::debug!(
            path = %path.display(),
            "OxiGIS desktop: a CJK-named file is not SFNT/TTC data; ignored",
        );
        return;
    }
    // A variable face whose DEFAULT master is a hairline ranks below every
    // static stem, then re-competes: neither the label rasteriser nor the
    // print path's screen-shared chain can select a heavier instance, so
    // letting it win the slot draws (and prints) hairline CJK. Demoted,
    // not excluded — on a system with no static CJK face it still beats
    // nothing at all. Answered with a bounded probe ([`is_thin_default_variable_file`])
    // rather than [`read_cjk_font`]'s full read: every candidate that would
    // win its slot reaches this line, so a full read here would pull a
    // 64 MB collection into memory — and log it as "loaded" — for a
    // question the table directory plus a few `fvar` bytes already answers.
    let rank = if is_thin_default_variable_file(path) {
        tracing::debug!(
            path = %path.display(),
            "OxiGIS desktop: thin-default variable face demoted below the static candidates",
        );
        rank + stems.len()
    } else {
        rank
    };
    let improves = match slot.as_ref() {
        None => true,
        Some(held) => (rank, extra) < (held.rank, held.extra),
    };
    if !improves {
        return;
    }
    *slot = Some(Candidate {
        tier,
        rank,
        extra,
        path: path.to_path_buf(),
    });
}

/// Rank penalty pushing a thin-default variable face below every listed
/// static stem: stock Windows 11 ships `NotoSansJP-VF.ttf` whose fvar
/// `wght` DEFAULT is 100 (Thin) — 44 % of the Regular instance's ink,
/// measured — and `notosansjp` outranks `meiryo` by name.
///
/// The applied penalty is the SCANNED table's own length (`stems.len()` in
/// [`consider`]), so it is one table wide whichever table is being scanned;
/// this constant is the regular table's value, which the test asserts is
/// large enough there.
#[cfg(test)]
const THIN_DEFAULT_VF_DEMOTION: usize = CJK_FONT_STEMS.len();

/// Largest number of sfnt table-directory records, or `fvar` axis records,
/// [`fvar_default_weight`] steps through before concluding a file offers no
/// evidence either way. Real fonts carry a few dozen tables and a handful of
/// axes; generous headroom against a hostile `numTables`/`axisCount` (each a
/// full `u16`) while keeping the probe bounded no matter what is claimed.
const MAX_PROBED_RECORDS: u16 = 256;

/// Whether `path`'s default instance is a hairline master, established with
/// [`fvar_default_weight`]'s bounded reads instead of a full [`read_cjk_font`]
/// load. `None` from the probe (not variable, no `wght` axis, unreadable,
/// not a font) answers `false`, matching `is_thin_default_variable` on a
/// parse failure: the file simply competes on name rank alone.
fn is_thin_default_variable_file(path: &Path) -> bool {
    fvar_default_weight(path).is_some_and(|weight| weight < 300.0)
}

/// Reads a font's `fvar` `wght` axis default straight off disk: the 12-byte
/// sfnt/ttcf header, up to [`MAX_PROBED_RECORDS`] 16-byte table-directory
/// records, and — only once an `fvar` table is listed — its 10-byte header
/// plus, per axis, the 12 leading bytes of up to [`MAX_PROBED_RECORDS`]
/// 20-byte OpenType VariationAxisRecords (tag and default value; trailing
/// max-value/flags/name-ID bytes are seeked past, never transferred). A
/// multi-megabyte collection is answered in well under a kilobyte of I/O.
///
/// [`None`] on any I/O error, a file that is neither SFNT nor TTC, an `fvar`
/// whose version is not `0x00010000`, or one with no `wght` axis within the
/// probed prefix. The fixed 20-byte-per-axis stride ignores the header's
/// own `axisSize` field, exactly as `ttf_parser` 0.25.1 does; both branches
/// (bare sfnt and `ttcf`) are checked against the OpenType spec by the
/// `#[cfg(test)]` suite below, on synthetic bytes.
fn fvar_default_weight(path: &Path) -> Option<f32> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let mut head = [0u8; 12];
    file.read_exact(&mut head).ok()?;
    let sfnt_offset: u64 = if is_ttc(&head) {
        // The TTC header's first OffsetTable entry, right after the 12
        // bytes already read: face 0's own sfnt offset table, the same face
        // every consumer in the label pipeline resolves (see the module
        // docs on why face 0).
        let mut first_face = [0u8; 4];
        file.read_exact(&mut first_face).ok()?;
        u32::from_be_bytes(first_face).into()
    } else if is_sfnt(&head) {
        0
    } else {
        return None;
    };
    file.seek(SeekFrom::Start(sfnt_offset)).ok()?;
    let mut sfnt_head = [0u8; 12];
    file.read_exact(&mut sfnt_head).ok()?;
    let num_tables = u16::from_be_bytes([sfnt_head[4], sfnt_head[5]]);
    let mut fvar_table: Option<(u32, u32)> = None;
    for _ in 0..num_tables.min(MAX_PROBED_RECORDS) {
        let mut record = [0u8; 16];
        file.read_exact(&mut record).ok()?;
        if record.get(..4) == Some(b"fvar".as_slice()) {
            fvar_table = Some((
                u32::from_be_bytes([record[8], record[9], record[10], record[11]]),
                u32::from_be_bytes([record[12], record[13], record[14], record[15]]),
            ));
            break;
        }
    }
    let (fvar_offset, fvar_length) = fvar_table?;
    // Spec-minimum header: version(4) + axesArrayOffset(2) + reserved(2) +
    // axisCount(2).
    if fvar_length < 10 {
        return None;
    }
    file.seek(SeekFrom::Start(fvar_offset.into())).ok()?;
    let mut fvar_head = [0u8; 10];
    file.read_exact(&mut fvar_head).ok()?;
    let version = u32::from_be_bytes([fvar_head[0], fvar_head[1], fvar_head[2], fvar_head[3]]);
    if version != 0x0001_0000 {
        return None;
    }
    let axes_array_offset = u16::from_be_bytes([fvar_head[4], fvar_head[5]]);
    let axis_count = u16::from_be_bytes([fvar_head[8], fvar_head[9]]);
    let axes_start = u64::from(fvar_offset).saturating_add(axes_array_offset.into());
    file.seek(SeekFrom::Start(axes_start)).ok()?;
    // A VariationAxisRecord is a fixed 20 bytes; only the leading tag (4) and
    // default value (4, at +8) of the 12-byte probe below are ever read, and
    // the remaining 8 (max-value, flags, name ID) are skipped rather than
    // transferred, whatever the file's own `axisSize` field claims.
    const AXIS_RECORD_SIZE: i64 = 20;
    let mut probe = [0u8; 12];
    for _ in 0..axis_count.min(MAX_PROBED_RECORDS) {
        file.read_exact(&mut probe).ok()?;
        if probe.get(..4) == Some(b"wght".as_slice()) {
            let default_fixed = i32::from_be_bytes([probe[8], probe[9], probe[10], probe[11]]);
            return Some(default_fixed as f32 / 65536.0);
        }
        file.seek(SeekFrom::Current(AXIS_RECORD_SIZE - probe.len() as i64))
            .ok()?;
    }
    None
}

/// Whether the face's default instance is a hairline master (fvar `wght`
/// default below 300) — the one shape both the screen and the default print
/// path would render wrong, because neither can pick an instance.
///
/// Full-bytes and `ttf_parser`-based, unlike [`fvar_default_weight`]; no
/// longer on the production path (see that function's docs) but kept as the
/// ground-truth oracle the bounded reader's tests check agreement against,
/// and to confirm the bundled static Noto is never demoted.
#[cfg(test)]
fn is_thin_default_variable(bytes: &[u8]) -> bool {
    let Ok(face) = ttf_parser::Face::parse(bytes, 0) else {
        return false;
    };
    face.is_variable()
        && face
            .variation_axes()
            .into_iter()
            .find(|axis| axis.tag == ttf_parser::Tag::from_bytes(b"wght"))
            .is_some_and(|axis| axis.def_value < 300.0)
}

/// Whether the file begins with a loadable font signature ([`is_sfnt`] /
/// [`is_ttc`]), reading only its first four bytes.
fn sniffs_as_font(path: &Path) -> bool {
    use std::io::Read as _;
    let mut magic = [0u8; 4];
    match std::fs::File::open(path).and_then(|mut file| file.read_exact(&mut magic)) {
        Ok(()) => is_sfnt(&magic) || is_ttc(&magic),
        Err(_) => false,
    }
}

/// The [`CJK_FONT_STEMS`] index the file name matches, or [`None`].
///
/// Lower is better; the scan keeps the minimum per script.
#[must_use]
pub fn cjk_stem_rank(path: &Path) -> Option<usize> {
    stem_match(path).map(|(rank, _)| rank)
}

/// Rank plus the number of normalised characters beyond the matched stem
/// (`meiryob` → 1 past `meiryo`), the weight-variant tie-breaker.
fn stem_match(path: &Path) -> Option<(usize, usize)> {
    stem_match_in(path, CJK_FONT_STEMS)
}

/// [`stem_match`] against an explicit stem table. The exclusion list applies
/// to every table: `simsunb` looks like a bold SimSun and is in fact
/// SimSun-ExtB, with no BMP coverage at all.
fn stem_match_in(path: &Path, stems: &[(&str, CjkScript)]) -> Option<(usize, usize)> {
    let normalized = normalized_font_stem(path)?;
    stems
        .iter()
        .position(|(known, _)| normalized.starts_with(known))
        .map(|rank| (rank, normalized.len() - stems[rank].0.len()))
}

/// A font file's stem, normalised for stem matching, or [`None`] when the
/// extension is not a font one or the name is on [`CJK_STEM_EXCLUSIONS`].
///
/// Every separator is stripped so `Noto Sans CJK JP` and `NotoSansCJKjp`
/// compare equal; distributions disagree about spaces, hyphens and
/// underscores. The keep test is Unicode `is_alphanumeric`, not
/// `is_ascii_alphanumeric`: a stock macOS install names its Japanese
/// Hiragino collections entirely in Japanese (`ヒラギノ角ゴシック W3.ttc`),
/// and stripping every non-ASCII character — as this function used to —
/// left nothing to match against [`CJK_FONT_STEMS`] at all. ASCII behaviour
/// is unchanged: only ASCII punctuation and whitespace are separators.
///
/// [`recompose_kana`] runs first and is not optional decoration: APFS/HFS+
/// hand `read_dir` names back canonically decomposed (confirmed
/// empirically — `ヒラギノ角ゴシック W3.ttc` reads back with `キ` +
/// U+3099, never the single codepoint `ギ`), and U+3099/U+309A are
/// combining marks (Mn), not alphabetic. Without recomposing first, the
/// filter below would drop the voicing and corrupt `ギ` into `キ`, `ゴ`
/// into `コ` — matching nothing, since native entries below are written
/// the ordinary, correctly-voiced way.
fn normalized_font_stem(path: &Path) -> Option<String> {
    let stem = path.file_stem().and_then(|stem| stem.to_str())?;
    let extension = path.extension().and_then(|ext| ext.to_str())?;
    let extension = extension.to_ascii_lowercase();
    if !FONT_EXTENSIONS.contains(&extension.as_str()) {
        return None;
    }
    let normalized: String = recompose_kana(stem)
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    if CJK_STEM_EXCLUSIONS
        .iter()
        .any(|excluded| normalized.starts_with(excluded))
    {
        return None;
    }
    Some(normalized)
}

/// Recomposes a kana base character immediately followed by a combining
/// dakuten (voiced sound mark, U+3099) or handakuten (semi-voiced sound
/// mark, U+309A) into its precomposed single codepoint — the inverse of
/// Unicode canonical decomposition, applied only to the finite, closed set
/// of kana voicing pairs ([`dakuten`], [`handakuten`]); see
/// [`normalized_font_stem`] for why. A base with no recognised mark
/// following it — including an already-precomposed `ギ`, the normal case
/// off macOS/APFS — passes through unchanged. An unpaired combining mark
/// is left for the alphanumeric filter downstream to drop, exactly as
/// before this function existed: total, never a panic.
fn recompose_kana(stem: &str) -> String {
    let mut composed = String::with_capacity(stem.len());
    let mut chars = stem.chars().peekable();
    while let Some(ch) = chars.next() {
        let voiced = match chars.peek() {
            Some('\u{3099}') => dakuten(ch),
            Some('\u{309A}') => handakuten(ch),
            _ => None,
        };
        match voiced {
            Some(fused) => {
                composed.push(fused);
                chars.next(); // the combining mark just fused into `fused`
            }
            None => composed.push(ch),
        }
    }
    composed
}

/// (base, dakuten) pairs for the closed set of hiragana/katakana that have
/// one, alternating: `.chars()` index `2i` is a base, `2i+1` its voiced
/// form. Generated from, and checked against, `unicodedata.normalize('NFD',
/// …)` over the whole Hiragana and Katakana blocks — not hand-transcribed.
const DAKUTEN_PAIRS: &str = "かがきぎくぐけげこごさざしじすずせぜそぞただちぢつづてでとどはばひびふぶへべほぼうゔカガキギクグケゲコゴサザシジスズセゼソゾタダチヂツヅテデトドハバヒビフブヘベホボウヴワヷヰヸヱヹヲヺ";

/// Same layout as [`DAKUTEN_PAIRS`] — only the h-row of each block has a
/// handakuten form.
const HANDAKUTEN_PAIRS: &str = "はぱひぴふぷへぺほぽハパヒピフプヘペホポ";

/// The precomposed dakuten form of `base`, or [`None`] outside [`DAKUTEN_PAIRS`].
fn dakuten(base: char) -> Option<char> {
    voiced_form(DAKUTEN_PAIRS, base)
}

/// The precomposed handakuten form of `base`, or [`None`] outside [`HANDAKUTEN_PAIRS`].
fn handakuten(base: char) -> Option<char> {
    voiced_form(HANDAKUTEN_PAIRS, base)
}

/// Scans `pairs` (see [`DAKUTEN_PAIRS`]'s layout) two characters at a time
/// for one whose first half is `base`, returning its second half. Total
/// even if a table were malformed (odd length; checked in `#[cfg(test)]`
/// below): the trailing unpaired character has no `?` partner, so the scan
/// ends in [`None`] rather than a panic.
fn voiced_form(pairs: &str, base: char) -> Option<char> {
    let mut chars = pairs.chars();
    while let Some(candidate) = chars.next() {
        let voiced = chars.next()?;
        if candidate == base {
            return Some(voiced);
        }
    }
    None
}

/// Whether `path`'s size is within (0, [`MAX_FONT_BYTES`]].
fn is_readable_size(path: &Path) -> bool {
    match std::fs::metadata(path) {
        Ok(meta) => meta.len() > 0 && meta.len() <= MAX_FONT_BYTES,
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CJK_FONT_STEMS, CjkScript, FONT_EXTENSIONS, TIER_USER, assemble_chain, chain_for_tiers,
        cjk_stem_rank, find_cjk_font_paths, is_sfnt, is_ttc, read_cjk_font, scan_tier,
    };
    use std::path::{Path, PathBuf};

    /// Test-side spelling of "the scan would consider this file".
    fn is_cjk_font_file(path: &Path) -> bool {
        cjk_stem_rank(path).is_some()
    }

    /// The whole chain as (path, bytes) pairs — what the shell's streaming
    /// loop produces over its channel, gathered for assertions.
    fn find_cjk_fonts() -> Vec<(PathBuf, Vec<u8>)> {
        find_cjk_font_paths()
            .into_iter()
            .filter_map(|path| read_cjk_font(&path).map(|bytes| (path, bytes)))
            .collect()
    }

    /// The script a file name classifies under.
    fn script_of(name: &str) -> CjkScript {
        CJK_FONT_STEMS[rank(name)].1
    }

    #[test]
    fn the_common_cjk_faces_are_recognised_whatever_the_separators() {
        for name in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
            "/usr/share/fonts/truetype/NotoSansJP-Regular.ttf",
            "/Library/Fonts/Hiragino Sans W3.otf",
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/YuGothM.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msjh.ttc",
            "/system/fonts/DroidSansFallback.ttf",
            "/home/user/.fonts/source_han_sans_jp.otf",
        ] {
            assert!(is_cjk_font_file(Path::new(name)), "must match: {name}");
        }
    }

    #[test]
    fn latin_faces_and_unknown_extensions_are_not_mistaken_for_cjk() {
        for name in [
            "/usr/share/fonts/truetype/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/NotoSans-Regular.ttf",
            "/usr/share/fonts/NotoSansCJK-Regular.woff2",
            "/usr/share/fonts/NotoSansCJK-Regular",
            "/usr/share/fonts/README",
            // Century Gothic: "gothic" alone is not a CJK stem.
            "C:/Windows/Fonts/GOTHIC.TTF",
        ] {
            assert!(!is_cjk_font_file(Path::new(name)), "must not match: {name}");
        }
    }

    #[test]
    fn collections_are_accepted_and_rank_like_any_other_candidate() {
        assert!(is_cjk_font_file(Path::new(
            "/fonts/NotoSansCJK-Regular.ttc"
        )));
        assert!(FONT_EXTENSIONS.contains(&"ttc"));
        assert_eq!(
            cjk_stem_rank(Path::new("/fonts/NotoSansCJK-Regular.ttc")),
            Some(0)
        );
    }

    #[test]
    fn supplementary_plane_only_faces_are_excluded_by_name() {
        // SimSun-ExtB / SimsunExtG carry no BMP CJK at all: matching them
        // would satisfy the name check and then draw every label as .notdef.
        assert_eq!(
            cjk_stem_rank(Path::new("C:/Windows/Fonts/simsunb.ttf")),
            None
        );
        assert_eq!(
            cjk_stem_rank(Path::new("C:/Windows/Fonts/SimsunExtG.ttf")),
            None
        );
        assert!(is_cjk_font_file(Path::new("C:/Windows/Fonts/simsun.ttc")));
    }

    /// Rank as a concrete usize so ordering assertions cannot pass vacuously
    /// through `None < Some(_)`.
    fn rank(name: &str) -> usize {
        cjk_stem_rank(Path::new(name)).unwrap_or_else(|| panic!("{name} must match a CJK stem"))
    }

    #[test]
    fn the_stem_order_prefers_modern_faces_and_regular_weights() {
        // Pan-CJK beats per-script beats platform faces.
        assert!(rank("NotoSansCJK-Regular.otf") < rank("NotoSansJP-Regular.ttf"));
        // By NAME NotoSansJP still outranks Meiryo — the thin-default
        // demotion happens at `consider`, from the file's own fvar (see
        // `a_thin_default_variable_face_is_demoted_below_every_stem`).
        assert!(rank("NotoSansJP-VF.ttf") < rank("meiryo.ttc"));
        // Windows: Meiryo and Yu Gothic beat MS Gothic's bitmap strikes...
        assert!(rank("meiryo.ttc") < rank("msgothic.ttc"));
        assert!(rank("YuGothM.ttc") < rank("msgothic.ttc"));
        // ...and Yu Gothic Medium/Regular outrank Bold/Light, which only the
        // bare `yugoth` stem catches.
        assert!(rank("YuGothM.ttc") < rank("YuGothB.ttc"));
        assert!(rank("YuGothR.ttc") < rank("YuGothL.ttc"));
        assert_eq!(rank("YuGothB.ttc"), rank("YuGothL.ttc"));
        // JhengHei is reachable through its real on-disk stem.
        assert_eq!(script_of("msjh.ttc"), CjkScript::TraditionalChinese);
        // Unknown names rank nowhere.
        assert_eq!(cjk_stem_rank(Path::new("DejaVuSans.ttf")), None);
    }

    #[test]
    fn a_thin_default_variable_face_is_demoted_below_every_stem() {
        // The bundled Noto is static: never demoted.
        assert!(!super::is_thin_default_variable(
            oxifont_bundled::NOTO_SANS_REGULAR
        ));
        // The penalty is large enough that a demoted rank loses to EVERY
        // listed static stem, whatever its own name rank was.
        let best_named = 0;
        let worst_named = CJK_FONT_STEMS.len() - 1;
        assert!(best_named + super::THIN_DEFAULT_VF_DEMOTION > worst_named);
        // Garbage bytes are not a variable face (total, never a panic).
        assert!(!super::is_thin_default_variable(&[0xDE, 0xAD, 0xBE, 0xEF]));
    }

    #[test]
    fn ground_truth_classifications_hold_for_the_tricky_faces() {
        // The only ASCII-named stock macOS "Hiragino" file is the GB18030
        // Simplified-Chinese one; an ASCII "Hiragino Sans …" name is a
        // user-renamed copy of a Japanese face, not a stock file (the real
        // ones ship under native Japanese names — see below). Osaka is the
        // legacy ASCII-named stock JP face.
        assert_eq!(
            script_of("/System/Library/Fonts/Hiragino Sans GB.ttc"),
            CjkScript::SimplifiedChinese
        );
        assert_eq!(script_of("Hiragino Sans W3.otf"), CjkScript::Japanese);
        assert_eq!(script_of("Osaka.ttf"), CjkScript::Japanese);
        // The real stock Hiragino files, native Japanese names and all —
        // ground truth for finding [169]. `normalized_font_stem` keeps
        // Unicode letters now, so these match without any ASCII renaming.
        assert_eq!(
            script_of("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
            CjkScript::Japanese
        );
        assert_eq!(
            script_of("/System/Library/Fonts/ヒラギノ明朝 ProN.ttc"),
            CjkScript::Japanese
        );
        assert_eq!(
            script_of("/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc"),
            CjkScript::Japanese
        );
        // STHeiti is the stock, always-reachable Traditional Chinese face:
        // face 0 of both weights is "Heiti TC" (verified with fontTools,
        // macOS 26.6.1). PingFang's face 0 is "PingFang HK", also
        // Traditional — not Simplified, which the table used to claim.
        assert_eq!(
            script_of("/System/Library/Fonts/STHeiti Medium.ttc"),
            CjkScript::TraditionalChinese
        );
        assert_eq!(
            script_of("/System/Library/Fonts/STHeiti Light.ttc"),
            CjkScript::TraditionalChinese
        );
        assert_eq!(script_of("PingFang.ttc"), CjkScript::TraditionalChinese);
        // Adobe regional subsets carry only their region's repertoire; the
        // language-specific full builds keep the pan tag.
        assert_eq!(
            script_of("SourceHanSansJP-Regular.otf"),
            CjkScript::Japanese
        );
        assert_eq!(script_of("SourceHanSansKR-Bold.otf"), CjkScript::Korean);
        assert_eq!(script_of("SourceHanSansSC-Regular.otf"), CjkScript::PanCjk);
        // Only the Full Droid build is pan-CJK; the reduced one keeps its
        // densest repertoire so it cannot truncate a real Korean face away.
        assert_eq!(script_of("DroidSansFallbackFull.ttf"), CjkScript::PanCjk);
        assert_eq!(
            script_of("DroidSansFallback.ttf"),
            CjkScript::SimplifiedChinese
        );
    }

    #[test]
    fn matching_is_case_insensitive_on_both_halves() {
        assert!(is_cjk_font_file(Path::new("/fonts/NOTOSANSCJK-BOLD.OTF")));
        assert!(is_cjk_font_file(Path::new("/fonts/notosanscjk-bold.otf")));
        assert!(is_cjk_font_file(Path::new("/fonts/MSGOTHIC.TTC")));
    }

    // ---- Unicode normalisation of native kana stems (finding [169]) -------

    /// The real on-disk NFD spelling of `ヒラギノ角ゴシック W3.ttc`: what
    /// `read_dir` actually hands back on macOS/APFS (confirmed
    /// empirically), built from explicit codepoints rather than typed
    /// decomposed text so it cannot be silently undone by source-level
    /// Unicode renormalisation. Shared by the two tests below it.
    fn nfd_hiragino_kaku_gothic_w3() -> String {
        [
            'ヒ', 'ラ', '\u{30AD}', '\u{3099}', 'ノ', '角', '\u{30B3}', '\u{3099}', 'シ', 'ッ',
            'ク', ' ', 'W', '3', '.', 't', 't', 'c',
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn the_voicing_tables_are_well_formed_pairs() {
        // `voiced_form` never panics on an odd-length table either way, but
        // it would silently under-match, so this is what makes the tables
        // trustworthy, not just safe.
        for table in [super::DAKUTEN_PAIRS, super::HANDAKUTEN_PAIRS] {
            assert_eq!(table.chars().count() % 2, 0, "{table:?} must pair up");
        }
    }

    #[test]
    fn recompose_kana_matches_the_verified_unicode_pairs() {
        use super::recompose_kana;
        // Spot-checked against `unicodedata.normalize('NFC', …)`, one pair
        // per row family, both blocks, both mark kinds; every literal is an
        // explicit `\u{...}` escape, on both sides, for the same reason as
        // `nfd_hiragino_kaku_gothic_w3` above.
        assert_eq!(recompose_kana("\u{304B}\u{3099}"), "\u{304C}"); // か+゙ → が
        assert_eq!(recompose_kana("\u{30AD}\u{3099}"), "\u{30AE}"); // キ+゙ → ギ (Hiragino)
        assert_eq!(recompose_kana("\u{30B3}\u{3099}"), "\u{30B4}"); // コ+゙ → ゴ (Gothic)
        assert_eq!(recompose_kana("\u{306F}\u{309A}"), "\u{3071}"); // は+゚ → ぱ
        assert_eq!(recompose_kana("\u{30DB}\u{309A}"), "\u{30DD}"); // ホ+゚ → ポ
        // No pair ('ア' has no dakuten form): base untouched, mark left for
        // the caller's alphanumeric filter to drop, not silently eaten.
        assert_eq!(recompose_kana("\u{30A2}\u{3099}"), "\u{30A2}\u{3099}");
        // Already-precomposed input passes straight through.
        assert_eq!(recompose_kana("\u{30AE}\u{30B4}"), "\u{30AE}\u{30B4}");
        // Total on the edges: empty, a leading mark with no base, ASCII.
        assert_eq!(recompose_kana(""), "");
        assert_eq!(recompose_kana("\u{3099}"), "\u{3099}");
        assert_eq!(recompose_kana("a\u{3099}"), "a\u{3099}");
    }

    #[test]
    fn kana_recomposition_matches_regardless_of_input_normalisation_form() {
        // `CJK_FONT_STEMS`' native entries are written NFC (ordinary,
        // correctly-voiced Japanese); the real file name arrives NFD. Both
        // spellings of the same logical name must classify identically.
        let nfc_name = "ヒラギノ角ゴシック W3.ttc";
        let nfd_name = nfd_hiragino_kaku_gothic_w3();
        assert!(
            nfd_name.chars().count() > nfc_name.chars().count(),
            "the NFD spelling must genuinely add the two combining marks, not collapse to the same bytes",
        );
        assert_eq!(script_of(nfc_name), CjkScript::Japanese);
        assert_eq!(
            cjk_stem_rank(Path::new(&nfd_name)),
            cjk_stem_rank(Path::new(nfc_name)),
            "NFC and NFD spellings of the same name must resolve to the same table entry",
        );
    }

    #[test]
    fn a_canonically_decomposed_file_name_is_still_recognised() {
        // Regression guard through the whole pipeline, not just
        // `normalized_font_stem` in isolation: write the file under its
        // NFD spelling and confirm the scan still wins the Japanese slot
        // with it, on any OS (the bytes are decomposed explicitly, so this
        // does not depend on the test-runner's own filesystem behaviour).
        let nfd_name = nfd_hiragino_kaku_gothic_w3();
        let dir = TempFontDir::new("nfd-roundtrip");
        dir.put(&nfd_name, TTC_HEAD);
        assert_eq!(
            file_names(&dir.chain()),
            [nfd_name],
            "the NFD-named Hiragino file must win the Japanese slot",
        );
    }

    // --- print/text v1.4 item 4 (D-W4): the bold chain ---

    #[test]
    fn the_bold_stems_match_the_real_windows_bold_files() {
        use super::{CJK_BOLD_FONT_STEMS, latin_bold_stem_match, stem_match_in};
        let bold_rank = |name: &str| stem_match_in(Path::new(name), CJK_BOLD_FONT_STEMS);
        for name in [
            "C:/Windows/Fonts/meiryob.ttc",
            "C:/Windows/Fonts/YuGothB.ttc",
            "C:/Windows/Fonts/malgunbd.ttf",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/msjhbd.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        ] {
            assert!(bold_rank(name).is_some(), "bold stem must match: {name}");
        }
        for name in [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/tahomabd.ttf",
            "/usr/share/fonts/truetype/DejaVuSans-Bold.ttf",
        ] {
            assert!(
                latin_bold_stem_match(Path::new(name)).is_some(),
                "Latin bold stem must match: {name}",
            );
        }
        // Regular faces are not bold faces...
        assert_eq!(bold_rank("C:/Windows/Fonts/meiryo.ttc"), None);
        assert_eq!(latin_bold_stem_match(Path::new("arial.ttf")), None);
        // ...MS Gothic and SimSun ship no bold at all...
        assert_eq!(bold_rank("C:/Windows/Fonts/msgothic.ttc"), None);
        assert_eq!(bold_rank("C:/Windows/Fonts/simsun.ttc"), None);
        // ...and `simsunb` is SimSun-ExtB, not a bold SimSun: the exclusion
        // list applies to the bold table too, or the bold chain would carry a
        // face with zero BMP coverage.
        assert_eq!(bold_rank("C:/Windows/Fonts/simsunb.ttf"), None);
        // The italic twin loses the tie-break to the upright bold.
        let upright = latin_bold_stem_match(Path::new("arialbd.ttf")).expect("upright");
        let italic = latin_bold_stem_match(Path::new("arialbdi.ttf")).expect("italic");
        assert!(upright < italic, "{upright:?} must beat {italic:?}");
    }

    #[test]
    fn the_bold_stems_are_normalised_and_pan_faces_lead() {
        use super::{CJK_BOLD_FONT_STEMS, LATIN_BOLD_FONT_STEMS};
        for stem in LATIN_BOLD_FONT_STEMS
            .iter()
            .copied()
            .chain(CJK_BOLD_FONT_STEMS.iter().map(|(stem, _)| *stem))
        {
            assert!(
                stem.chars()
                    .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit()),
                "stem {stem:?} would never match a normalised file name",
            );
        }
        // A pan-CJK bold truncates the chain, so it has to come first — the
        // same rule the regular table follows.
        let first_pan = CJK_BOLD_FONT_STEMS
            .iter()
            .position(|(_, script)| *script == CjkScript::PanCjk)
            .expect("a pan entry exists");
        let last_pan = CJK_BOLD_FONT_STEMS
            .iter()
            .rposition(|(_, script)| *script == CjkScript::PanCjk)
            .expect("a pan entry exists");
        assert_eq!(first_pan, 0);
        assert!(
            CJK_BOLD_FONT_STEMS[..=last_pan]
                .iter()
                .all(|(_, script)| *script == CjkScript::PanCjk),
            "the pan block must be a prefix",
        );
    }

    #[test]
    fn the_bold_scan_is_total_on_this_machine() {
        // Like `find_cjk_font_paths`, this walks the real OS font directories:
        // the assertion is that it is TOTAL, not that any particular face is
        // installed (CI images ship no fonts at all, and an empty bold chain
        // simply means Bold labels draw Regular).
        for path in super::find_cjk_bold_font_paths() {
            assert!(
                super::latin_bold_stem_match(&path).is_some()
                    || super::stem_match_in(&path, super::CJK_BOLD_FONT_STEMS).is_some(),
                "every returned path matched a bold stem: {}",
                path.display(),
            );
            assert!(read_cjk_font(&path).is_some(), "and reads back as a font");
        }
    }

    #[test]
    fn the_known_stems_are_normalised_lower_case_alphanumerics() {
        // Unicode-aware, not ASCII-only: `normalized_font_stem` keeps any
        // alphanumeric character (native Japanese stems included), only
        // ASCII-lowercases, and drops everything else (separators). A stem
        // with an uppercase letter, a digit-lookalike, or a separator could
        // never actually come out of that normalisation, so could never
        // match a real file name.
        for (stem, _) in CJK_FONT_STEMS {
            assert!(
                stem.chars()
                    .all(|ch| ch.is_alphanumeric() && ch.to_ascii_lowercase() == ch),
                "stem {stem:?} would never match a normalised file name",
            );
        }
    }

    #[test]
    fn specific_stems_precede_every_stem_that_prefixes_them() {
        // First-entry-wins matching: if a general stem came first, the
        // specific entry (and its script tag) would be unreachable.
        for (i, (specific, _)) in CJK_FONT_STEMS.iter().enumerate() {
            for (j, (general, _)) in CJK_FONT_STEMS.iter().enumerate() {
                if i != j && specific.starts_with(general) {
                    assert!(
                        j > i,
                        "{general:?} (index {j}) must come after {specific:?} (index {i})",
                    );
                }
            }
        }
    }

    #[test]
    fn signatures_split_single_faces_collections_and_junk_three_ways() {
        assert!(is_sfnt(&[0x00, 0x01, 0x00, 0x00, 0x99]));
        assert!(is_sfnt(b"OTTO...."));
        assert!(is_sfnt(b"true...."));
        assert!(!is_sfnt(b"ttcf...."));
        // A collection is not an SFNT face but is loadable.
        assert!(is_ttc(b"ttcf...."));
        assert!(!is_ttc(b"OTTO...."));
        // A WOFF2 and an HTML error page saved as `.ttf` are neither.
        for junk in [b"wOF2....".as_slice(), b"<!DOCTYPE html>", b"abc", b""] {
            assert!(!is_sfnt(junk));
            assert!(!is_ttc(junk));
        }
    }

    #[test]
    fn the_bundled_latin_face_is_recognised_as_an_sfnt() {
        // Not a CJK font, but it is the one real font every build has.
        assert!(is_sfnt(oxifont_bundled::NOTO_SANS_REGULAR));
    }

    #[test]
    fn the_scan_answers_without_panicking_on_this_machine() {
        // Whether a CJK font exists here is not something a test may assert;
        // that the scan terminates and returns a chain is.
        let _ = find_cjk_font_paths();
    }

    // ---- walk behaviour, on throwaway directories -------------------------

    /// Minimal bytes that pass the four-byte sniff.
    const SFNT_HEAD: &[u8] = &[0x00, 0x01, 0x00, 0x00, 0, 0, 0, 0];
    const TTC_HEAD: &[u8] = b"ttcf\x00\x02\x00\x00";

    /// A self-cleaning directory of fake fonts under the OS temp dir.
    struct TempFontDir {
        root: PathBuf,
    }

    impl TempFontDir {
        fn new(tag: &str) -> Self {
            let root =
                std::env::temp_dir().join(format!("oxigis-font-scan-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).expect("create temp font dir");
            Self { root }
        }

        fn put(&self, relative: &str, contents: &[u8]) -> PathBuf {
            let path = self.root.join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create parent dirs");
            }
            std::fs::write(&path, contents).expect("write fake font");
            path
        }

        fn chain(&self) -> Vec<PathBuf> {
            assemble_chain(
                scan_tier(std::slice::from_ref(&self.root), TIER_USER, CJK_FONT_STEMS),
                CJK_FONT_STEMS,
            )
            .into_iter()
            .map(|candidate| candidate.path)
            .collect()
        }
    }

    impl Drop for TempFontDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn file_names(paths: &[PathBuf]) -> Vec<String> {
        paths
            .iter()
            .map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .expect("test paths are unicode")
                    .to_owned()
            })
            .collect()
    }

    #[test]
    fn rank_beats_directory_order_within_a_script() {
        let dir = TempFontDir::new("rank");
        dir.put("meiryo.ttc", TTC_HEAD);
        dir.put("msgothic.ttc", TTC_HEAD);
        assert_eq!(file_names(&dir.chain()), ["meiryo.ttc"]);
    }

    #[test]
    fn junk_content_never_poisons_the_chain() {
        let dir = TempFontDir::new("junk");
        // Best-ranked name, worthless bytes: must lose to the real face, not
        // erase the whole fallback.
        dir.put("NotoSansCJK-Regular.otf", b"<!DOCTYPE html><html>...");
        dir.put("meiryo.ttc", TTC_HEAD);
        assert_eq!(file_names(&dir.chain()), ["meiryo.ttc"]);
    }

    #[test]
    fn bold_variants_lose_the_rank_tie_to_the_shorter_name() {
        let dir = TempFontDir::new("weights");
        dir.put("meiryob.ttc", TTC_HEAD);
        dir.put("meiryo.ttc", TTC_HEAD);
        dir.put("msyhbd.ttc", TTC_HEAD);
        dir.put("msyhl.ttc", TTC_HEAD);
        dir.put("msyh.ttc", TTC_HEAD);
        assert_eq!(file_names(&dir.chain()), ["meiryo.ttc", "msyh.ttc"]);
    }

    #[test]
    fn macos_native_weight_variants_lose_the_tie_to_the_preferred_weight() {
        // finding [169]: `ヒラギノ角ゴシック W0..W9.ttc` all tie on trailing
        // character count once normalised (`w0`..`w9` are each 2 chars); W3
        // must win by RANK — a dedicated table entry ahead of the bare
        // stem — not by "file seen first". Inserted out of weight order so
        // an order-dependent pick would fail this test.
        let dir = TempFontDir::new("hiragino-weights");
        dir.put("ヒラギノ角ゴシック W6.ttc", TTC_HEAD);
        dir.put("ヒラギノ角ゴシック W0.ttc", TTC_HEAD);
        dir.put("ヒラギノ角ゴシック W3.ttc", TTC_HEAD);
        dir.put("ヒラギノ角ゴシック W9.ttc", TTC_HEAD);
        assert_eq!(file_names(&dir.chain()), ["ヒラギノ角ゴシック W3.ttc"]);

        // STHeiti ships two weights; Medium must win over Light — a
        // hairline weight, wrong for map labels — by rank, even though
        // "stheitilight" has FEWER trailing characters than
        // "stheitimedium" and would win the old fewest-extra-chars
        // tie-break.
        let dir = TempFontDir::new("stheiti-weights");
        dir.put("STHeiti Light.ttc", TTC_HEAD);
        dir.put("STHeiti Medium.ttc", TTC_HEAD);
        assert_eq!(file_names(&dir.chain()), ["STHeiti Medium.ttc"]);
    }

    #[test]
    fn the_chain_collects_one_face_per_script_in_rank_order() {
        let dir = TempFontDir::new("chain");
        dir.put("msjh.ttc", TTC_HEAD);
        dir.put("malgun.ttf", SFNT_HEAD);
        dir.put("msyh.ttc", TTC_HEAD);
        dir.put("meiryo.ttc", TTC_HEAD);
        assert_eq!(
            file_names(&dir.chain()),
            ["meiryo.ttc", "malgun.ttf", "msyh.ttc", "msjh.ttc"]
        );
    }

    #[test]
    fn a_pan_cjk_face_truncates_everything_ranked_after_it() {
        let dir = TempFontDir::new("pan");
        dir.put("NotoSansCJK-Regular.otf", SFNT_HEAD);
        dir.put("malgun.ttf", SFNT_HEAD);
        assert_eq!(file_names(&dir.chain()), ["NotoSansCJK-Regular.otf"]);

        // A better-ranked per-script face still precedes the pan face.
        let dir = TempFontDir::new("pan2");
        dir.put("NotoSansJP-Regular.ttf", SFNT_HEAD);
        dir.put("DroidSansFallbackFull.ttf", SFNT_HEAD);
        dir.put("malgun.ttf", SFNT_HEAD);
        assert_eq!(
            file_names(&dir.chain()),
            ["NotoSansJP-Regular.ttf", "DroidSansFallbackFull.ttf"]
        );

        // The reduced Droid build is NOT pan: it must coexist with a real
        // Korean face instead of truncating it away.
        let dir = TempFontDir::new("pan3");
        dir.put("DroidSansFallback.ttf", SFNT_HEAD);
        dir.put("malgun.ttf", SFNT_HEAD);
        assert_eq!(
            file_names(&dir.chain()),
            ["DroidSansFallback.ttf", "malgun.ttf"]
        );

        // A regional Source Han subset is per-script, not pan; the full
        // language-specific build is pan.
        let dir = TempFontDir::new("pan4");
        dir.put("SourceHanSansJP-Regular.otf", SFNT_HEAD);
        dir.put("malgun.ttf", SFNT_HEAD);
        assert_eq!(
            file_names(&dir.chain()),
            ["SourceHanSansJP-Regular.otf", "malgun.ttf"]
        );
        let dir = TempFontDir::new("pan5");
        dir.put("SourceHanSansSC-Regular.otf", SFNT_HEAD);
        dir.put("malgun.ttf", SFNT_HEAD);
        assert_eq!(file_names(&dir.chain()), ["SourceHanSansSC-Regular.otf"]);
    }

    #[test]
    fn empty_files_are_not_candidates() {
        let dir = TempFontDir::new("empty");
        dir.put("meiryo.ttc", b"");
        assert!(dir.chain().is_empty());
    }

    #[test]
    fn the_walk_stops_at_max_scan_depth() {
        let dir = TempFontDir::new("depth");
        dir.put("a/b/c/meiryo.ttc", TTC_HEAD);
        assert_eq!(file_names(&dir.chain()), ["meiryo.ttc"]);

        let dir = TempFontDir::new("depth2");
        dir.put("a/b/c/d/meiryo.ttc", TTC_HEAD);
        assert!(dir.chain().is_empty());
    }

    #[test]
    fn a_shallow_file_wins_a_full_tie_against_a_deep_one() {
        let dir = TempFontDir::new("bfs");
        // Same name, same rank, same suffix — only depth differs. Breadth-
        // first means the root copy is seen, and therefore kept, first.
        let deep = dir.put("aaa/meiryo.ttc", TTC_HEAD);
        let shallow = dir.put("meiryo.ttc", TTC_HEAD);
        let chain = dir.chain();
        assert_eq!(chain, [shallow]);
        assert_ne!(chain, [deep]);
    }

    #[test]
    fn symlinked_font_files_are_followed() {
        let dir = TempFontDir::new("symlink");
        // The target's own name matches nothing; only the link is a
        // candidate, so this passes solely if the link is classified by
        // following it.
        let target = dir.put("real/target-bytes.dat", TTC_HEAD);
        let link = dir.root.join("meiryo.ttc");
        #[cfg(windows)]
        let made = std::os::windows::fs::symlink_file(&target, &link).is_ok();
        #[cfg(unix)]
        let made = std::os::unix::fs::symlink(&target, &link).is_ok();
        if !made {
            eprintln!("skipped: creating symlinks needs privileges on this machine");
            return;
        }
        assert_eq!(file_names(&dir.chain()), ["meiryo.ttc"]);
    }

    #[test]
    fn a_directory_symlink_cycle_does_not_revisit_files() {
        // print/text v1.4 finding [175]: without a canonicalised visited
        // set, a directory symlink pointing at an ancestor is walked as a
        // second, distinct directory, and every file in it is scored again.
        let dir = TempFontDir::new("cycle");
        dir.put("meiryo.ttc", TTC_HEAD);
        let link = dir.root.join("loop");
        #[cfg(windows)]
        let made = std::os::windows::fs::symlink_dir(&dir.root, &link).is_ok();
        #[cfg(unix)]
        let made = std::os::unix::fs::symlink(&dir.root, &link).is_ok();
        if !made {
            eprintln!("skipped: creating symlinks needs privileges on this machine");
            return;
        }
        let visits = std::cell::Cell::new(0usize);
        super::walk_tier(std::slice::from_ref(&dir.root), &mut |_path| {
            visits.set(visits.get() + 1);
        });
        // `loop` canonicalises to the same directory already walked at
        // depth 0, so it must be skipped outright: `meiryo.ttc` is visited
        // once, not once per depth the cycle would otherwise re-enter
        // (four times, up to `MAX_SCAN_DEPTH`, before this fix).
        assert_eq!(
            visits.get(),
            1,
            "meiryo.ttc must be visited exactly once despite the ancestor-cycling symlink",
        );
    }

    #[test]
    fn a_user_face_wins_its_slot_and_the_system_fills_the_rest() {
        // One user-installed Japanese font must not cost the map its Korean
        // coverage: the user face keeps its slot (and leads the chain), the
        // system fills every other script.
        let user = TempFontDir::new("tier-user");
        let system = TempFontDir::new("tier-system");
        user.put("msgothic.ttc", TTC_HEAD);
        system.put("meiryo.ttc", TTC_HEAD);
        system.put("malgun.ttf", SFNT_HEAD);
        let chain = chain_for_tiers(
            std::slice::from_ref(&user.root),
            std::slice::from_ref(&system.root),
        );
        let paths: Vec<PathBuf> = chain.into_iter().map(|c| c.path).collect();
        // msgothic (user) beats meiryo (system) for the Japanese slot despite
        // meiryo's better rank; malgun still arrives from the system tier.
        assert_eq!(file_names(&paths), ["msgothic.ttc", "malgun.ttf"]);
    }

    #[test]
    fn a_user_face_precedes_a_system_pan_face_instead_of_vanishing() {
        let user = TempFontDir::new("tier-user2");
        let system = TempFontDir::new("tier-system2");
        user.put("malgun.ttf", SFNT_HEAD);
        system.put("NotoSansCJK-Regular.otf", SFNT_HEAD);
        let chain = chain_for_tiers(
            std::slice::from_ref(&user.root),
            std::slice::from_ref(&system.root),
        );
        let paths: Vec<PathBuf> = chain.into_iter().map(|c| c.path).collect();
        // The user's Korean face shapes Korean first; the system pan face
        // backstops every other script.
        assert_eq!(
            file_names(&paths),
            ["malgun.ttf", "NotoSansCJK-Regular.otf"]
        );
    }

    #[test]
    fn read_cjk_font_rejects_what_is_no_longer_a_font() {
        let dir = TempFontDir::new("read");
        let good = dir.put("meiryo.ttc", TTC_HEAD);
        let junk = dir.put("msyh.ttc", b"<!DOCTYPE html>");
        let empty = dir.put("msjh.ttc", b"");
        assert_eq!(read_cjk_font(&good).as_deref(), Some(TTC_HEAD));
        assert_eq!(read_cjk_font(&junk), None);
        assert_eq!(read_cjk_font(&empty), None);
        assert_eq!(read_cjk_font(Path::new("does/not/exist.ttc")), None);
    }

    // ---- the bounded `fvar` reader (finding [175]) -------------------------

    /// Builds a minimal single-face SFNT — sfntVersion, a table directory,
    /// and each listed table's bytes — starting at `base_offset` bytes into
    /// whatever larger buffer it ends up embedded in (0 standalone, 16 —
    /// the synthetic TTC header's length — for a collection's face 0).
    /// Real sfnt table-directory offsets are absolute from the FILE's
    /// start even inside a collection, so building this without knowing
    /// where "start" is would silently mis-point every table. Real enough
    /// for [`fvar_default_weight`], which looks at nothing but the table
    /// directory and `fvar`; not a loadable font otherwise.
    fn synthetic_sfnt(base_offset: u32, tables: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let dir_start = 12_usize;
        let bodies_start = dir_start + tables.len() * 16;
        let mut head_and_dir = vec![0u8; bodies_start];
        head_and_dir[0..4].copy_from_slice(&0x0001_0000u32.to_be_bytes());
        head_and_dir[4..6].copy_from_slice(&(tables.len() as u16).to_be_bytes());
        let mut bodies = Vec::new();
        for (i, (tag, data)) in tables.iter().enumerate() {
            let record = dir_start + i * 16;
            let table_offset = base_offset + bodies_start as u32 + bodies.len() as u32;
            head_and_dir[record..record + 4].copy_from_slice(tag.as_slice());
            head_and_dir[record + 8..record + 12].copy_from_slice(&table_offset.to_be_bytes());
            head_and_dir[record + 12..record + 16]
                .copy_from_slice(&(data.len() as u32).to_be_bytes());
            bodies.extend_from_slice(data);
        }
        head_and_dir.extend_from_slice(&bodies);
        head_and_dir
    }

    /// A minimal `fvar` table body: one `wght` axis whose default (and, for
    /// simplicity, min/max too — the probe never reads those) is `default_wght`.
    fn synthetic_fvar(default_wght: f32) -> Vec<u8> {
        let mut out = vec![0u8; 16];
        out[0..4].copy_from_slice(&0x0001_0000u32.to_be_bytes());
        out[4..6].copy_from_slice(&16u16.to_be_bytes()); // axesArrayOffset
        out[8..10].copy_from_slice(&1u16.to_be_bytes()); // axisCount
        out[10..12].copy_from_slice(&20u16.to_be_bytes()); // axisSize (spec value; the probe ignores it, matching ttf_parser)
        let fixed = (default_wght * 65536.0).round() as i32;
        let mut axis = vec![0u8; 20];
        axis[0..4].copy_from_slice(b"wght");
        axis[4..8].copy_from_slice(&fixed.to_be_bytes()); // minValue, unread by the probe
        axis[8..12].copy_from_slice(&fixed.to_be_bytes()); // defaultValue
        axis[12..16].copy_from_slice(&fixed.to_be_bytes()); // maxValue, unread by the probe
        out.extend_from_slice(&axis);
        out
    }

    /// Wraps a `synthetic_sfnt(16, ..)` face as a one-font TTC, exercising
    /// the branch that matters most: every real macOS/Windows CJK candidate
    /// is a collection, so an offset-base mistake here would silently say
    /// "not variable" for exactly the files the demotion exists to catch.
    fn synthetic_ttc(face0: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; 16];
        out[0..4].copy_from_slice(b"ttcf");
        out[4..6].copy_from_slice(&1u16.to_be_bytes());
        out[8..12].copy_from_slice(&1u32.to_be_bytes()); // numFonts
        out[12..16].copy_from_slice(&16u32.to_be_bytes()); // OffsetTable[0]
        out.extend_from_slice(face0);
        out
    }

    #[test]
    fn fvar_default_weight_reads_the_wght_default_with_bounded_io() {
        use super::{fvar_default_weight, is_thin_default_variable_file};
        let dir = TempFontDir::new("fvar-probe");
        let thin = dir.put(
            "thin.ttf",
            &synthetic_sfnt(0, &[(b"fvar", &synthetic_fvar(100.0))]),
        );
        let regular = dir.put(
            "regular.ttf",
            &synthetic_sfnt(0, &[(b"fvar", &synthetic_fvar(400.0))]),
        );
        let no_fvar = dir.put("static.ttf", &synthetic_sfnt(0, &[(b"head", &[0u8; 4])]));
        let thin_collection = dir.put(
            "thin.ttc",
            &synthetic_ttc(&synthetic_sfnt(16, &[(b"fvar", &synthetic_fvar(100.0))])),
        );

        assert_eq!(fvar_default_weight(&thin).map(f32::round), Some(100.0));
        assert_eq!(fvar_default_weight(&regular).map(f32::round), Some(400.0));
        assert_eq!(fvar_default_weight(&no_fvar), None);
        // The `ttcf` branch: face 0's sfnt offset is found through the
        // collection header, not assumed to start at byte 0.
        assert_eq!(
            fvar_default_weight(&thin_collection).map(f32::round),
            Some(100.0)
        );
        assert_eq!(fvar_default_weight(Path::new("does/not/exist.ttf")), None);
        assert_eq!(fvar_default_weight(&dir.put("junk.ttf", b"nope")), None);

        assert!(is_thin_default_variable_file(&thin));
        assert!(!is_thin_default_variable_file(&regular));
        assert!(!is_thin_default_variable_file(&no_fvar));
        assert!(is_thin_default_variable_file(&thin_collection));
    }

    // `is_thin_default_variable` (full bytes) is no longer on the
    // production path — `is_thin_default_variable_file` above is — but
    // `a_thin_default_variable_face_is_demoted_below_every_stem` already
    // covers it clearing the bundled static Noto and never panicking on
    // garbage, so that is not re-asserted here.

    // ---- end to end through the label engine ------------------------------

    /// Which probe a face must cover, judged by its stem's script tag. A pan
    /// face must cover all four repertoires; per-script faces get their own
    /// two characters.
    fn probe_text_for(path: &Path) -> &'static str {
        let rank = cjk_stem_rank(path).expect("the scan only returns stem matches");
        match CJK_FONT_STEMS[rank].1 {
            CjkScript::Korean => "서울",
            CjkScript::SimplifiedChinese => "北京",
            CjkScript::TraditionalChinese => "台北",
            CjkScript::Japanese => "東京",
            CjkScript::PanCjk => "東京서울北京台北",
        }
    }

    #[test]
    fn the_found_fonts_shape_real_cjk_through_the_label_engine() {
        // End-to-end guard for the TTC pass-through and the chain assembly:
        // whatever the scan hands over must be bytes the label engine can
        // parse, shape and rasterise real (non-.notdef) CJK glyphs from.
        // Passes trivially only on a machine with no CJK font at all.
        let paths = find_cjk_font_paths();
        if paths.is_empty() {
            eprintln!("skipped: no CJK font on this machine");
            return;
        }
        let fonts = find_cjk_fonts();
        // A path the scan already signature-sniffed must load — silently
        // skipping here would hide a byte-gate regression. Only a file that
        // genuinely changed on disk since the scan may drop out.
        for path in &paths {
            let still_valid = std::fs::read(path)
                .map(|bytes| is_sfnt(&bytes) || is_ttc(&bytes))
                .unwrap_or(false);
            assert!(
                !still_valid || fonts.iter().any(|(loaded, _)| loaded == path),
                "{} sniffed valid but read_cjk_font dropped it",
                path.display(),
            );
        }
        if fonts.is_empty() {
            eprintln!("skipped: every selected font vanished mid-test");
            return;
        }
        let mut engine =
            oxigis_render::label::LabelEngine::new(oxifont_bundled::NOTO_SANS_REGULAR.to_vec())
                .expect("bundled Noto Sans parses");
        for (_, bytes) in &fonts {
            engine.add_fallback_font(bytes.clone());
        }
        for (path, _) in &fonts {
            let probe = probe_text_for(path);
            let label = engine
                .shape(probe, 24.0)
                .unwrap_or_else(|e| panic!("{} must shape {probe}: {e}", path.display()));
            assert!(
                !label.is_empty(),
                "{} shaped {probe} to an inkless label",
                path.display(),
            );
            // .notdef boxes rasterise with ink, so a non-empty label alone
            // proves nothing — gid 0 anywhere means the chain failed to cover
            // its own script's probe.
            assert!(
                label.glyphs().iter().all(|glyph| glyph.key.gid != 0),
                "{} shaped {probe} with .notdef glyphs",
                path.display(),
            );
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn macos_fills_the_japanese_and_traditional_chinese_slots_with_real_ink() {
        // Unlike `the_found_fonts_shape_real_cjk_through_the_label_engine`
        // above, which only checks whatever the scan returned (and so would
        // pass just as well with Japanese silently empty), this pins finding
        // [169] itself: Japanese and Traditional Chinese must both resolve
        // to a real face. Stock Hiragino Sans and STHeiti sit under System
        // Integrity Protection, so their absence would be the regression,
        // not an environment quirk to skip past — this loop would have
        // failed on its first iteration against the pre-fix table.
        let fonts = find_cjk_fonts();
        let mut engine =
            oxigis_render::label::LabelEngine::new(oxifont_bundled::NOTO_SANS_REGULAR.to_vec())
                .expect("bundled Noto Sans parses");
        for (_, bytes) in &fonts {
            engine.add_fallback_font(bytes.clone());
        }
        let tagged_index = |script: CjkScript| {
            fonts.iter().position(|(path, _)| {
                cjk_stem_rank(path).is_some_and(|rank| CJK_FONT_STEMS[rank].1 == script)
            })
        };
        for (script, probe) in [
            (CjkScript::Japanese, "東京"),
            (CjkScript::TraditionalChinese, "台北"),
        ] {
            // A pan-CJK face installed alongside the stock ones (Homebrew's
            // `font-noto-sans-cjk`) truncates the chain right after itself
            // and legitimately satisfies every slot alone — accepted here,
            // not just the exact per-script tag.
            assert!(
                fonts
                    .iter()
                    .any(|(path, _)| cjk_stem_rank(path).is_some_and(|rank| {
                        let tagged = CJK_FONT_STEMS[rank].1;
                        tagged == script || tagged == CjkScript::PanCjk
                    })),
                "{script:?} slot is empty on a stock macOS install",
            );
            let label = engine
                .shape(probe, 24.0)
                .unwrap_or_else(|error| panic!("{probe} must shape: {error}"));
            assert!(
                label.glyphs().iter().all(|glyph| glyph.key.gid != 0),
                "{probe} shaped with .notdef glyphs — the {script:?} face the scan picked cannot cover it",
            );
        }
        // finding [169]'s other half: Japanese must SHAPE with Japanese
        // glyph forms, which first-glyph-wins only gives it if the Japanese
        // face precedes Simplified Chinese (Hiragino Sans GB has kana and
        // kanji coverage too). No PanCjk exception needed: a truncated
        // chain leaves at most one position filled, `position` gives `None`.
        if let (Some(japanese), Some(simplified)) = (
            tagged_index(CjkScript::Japanese),
            tagged_index(CjkScript::SimplifiedChinese),
        ) {
            assert!(
                japanese < simplified,
                "the Japanese face must precede the Simplified Chinese face in the fallback chain",
            );
        }
    }
}
