//! Shaping and rasterisation: a string plus a pixel size becomes a
//! [`ShapedLabel`] and a set of glyph slots in the [`GlyphAtlas`].
//!
//! # What `oxitext` does and what this module adds
//!
//! [`oxitext::Pipeline`] owns the hard parts — swash shaping (kerning,
//! ligatures, marks), UAX #14 line breaking, and the per-cluster fallback that
//! re-shapes any cluster whose glyph id came back `0` through the next font in
//! the chain. That last one is what makes CJK work: a Latin-only primary font
//! plus a CJK fallback covers a mixed label without the caller splitting runs.
//!
//! Three things are added here.
//!
//! 1. **Bearings.** `Pipeline::render` hands back bitmaps but not the bearings
//!    that place them: its own compositor blits every glyph's top-left at the
//!    baseline pen, which top-aligns `H` and `g` alike. This module drives
//!    [`oxitext::Pipeline::shape_and_layout`] for positions and
//!    [`oxitext_raster::RasterBackend::rasterize_full`] for coverage *and*
//!    metrics, then places each glyph box at
//!    `(pen.x + bearing_x, baseline − (height + ymin))`. Note that
//!    `RasterOutput::bearing_y` is `fontdue`'s `ymin` — the signed distance
//!    from the baseline to the glyph's **bottom** edge, positive up — despite
//!    the upstream doc comment calling it a top bearing. Verified against Noto
//!    Sans at 14 px: `H` (10 px tall, `ymin` 0) sits on the baseline, `g`
//!    (12 px tall, `ymin` −4) hangs 4 px below it.
//! 2. **Caching.** Map labels are re-placed every frame but their strings
//!    change only when the tile set does, so shaping runs once per
//!    `(text, size, weight, orientation)` and the result is kept in an LRU
//!    that starts at [`DEFAULT_LABEL_CACHE`] entries and grows, up to
//!    [`MAX_LABEL_CACHE`], when a frame's working set does not fit it. The
//!    weight is part of the key by necessity: a `(text, size)`-keyed cache
//!    would hand a bold request whichever weight asked first (print/text
//!    v1.4, D-W3). A cache HIT allocates nothing at all: the entries are
//!    bucketed by the `Copy` part of the key so the text is looked up as a
//!    borrowed `&str`, and the LRU re-files a shared handle rather than a
//!    copied string. The key stays EXACT — hashing a label's text down to an
//!    integer would let a collision serve the wrong ink.
//! 3. **Font identity.** `PositionedGlyph::font_data` is an `Arc<[u8]>` whose
//!    address is the only handle on "which font". Addresses get recycled once a
//!    fallback chain is replaced, so this module interns them: each distinct
//!    pointer is assigned a stable index and the `Arc` is *kept alive* by the
//!    intern table, which makes the address unique for as long as the index is.
//!
//! # No I/O
//!
//! Fonts arrive as bytes ([`LabelEngine::new`],
//! [`LabelEngine::set_fallback_fonts`], [`LabelEngine::add_fallback_font`]) and
//! never as paths. The web shell fetches a CJK font asynchronously *after*
//! startup and calls `add_fallback_font` when it lands; that bumps
//! [`LabelEngine::generation`], which clears the label cache and the atlas so
//! the labels shaped as `.notdef` before the font arrived are re-shaped with
//! it. Without that bump the async font would silently do nothing.
//!
//! # Weight
//!
//! [`LabelWeight::Bold`] draws through a SECOND [`oxitext::Pipeline`] built
//! over a bold face chain the shell supplies ([`LabelEngine::set_bold_fonts`]),
//! sharing the ONE glyph atlas — atlas keys carry an interned font index, and
//! a bold face is a different `Arc`, so the two weights cannot collide.
//! The bold chain is **bold faces ++ the whole regular chain**: a bold face
//! that lacks a CJK block falls through to the regular face for those
//! clusters, so a mixed-weight line is the worst case and `.notdef` never is.
//! With no bold chain a bold request draws Regular and says so once —
//! **never** a synthetic emboldening, on this side or in the PDF exporter, so
//! the page and the map keep showing the same ink.
//!
//! # Generations
//!
//! A [`ShapedLabel`] holds atlas rectangles, so it is only meaningful while the
//! engine's [`LabelEngine::generation`] is the one it was built at. The counter
//! moves when the font set changes and when the atlas is rebuilt wholesale.
//! Callers that keep labels across frames (part B's placement pass) should
//! compare [`ShapedLabel::generation`] against the engine's and re-request on a
//! mismatch; callers that re-request every frame — the intended usage, since a
//! cache hit is a hash lookup — need not care.
//!
//! # A full atlas
//!
//! Packing a label into a full atlas is a routine condition with a three-rung
//! answer, and only the last rung is visible to a caller:
//!
//! 1. **Compact.** Free the slots of every glyph no live label points at and
//!    pack again. Nothing moves and the generation does not change, so the
//!    labels the current frame has already been handed stay valid — an
//!    overflow costs one re-pack of one label, not a re-shape of the screen.
//! 2. **Rebuild.** When everything packed is still live, [`GlyphAtlas::clear`]
//!    the atlas and bump the generation, which is what
//!    [`ShapedLabel::generation`] exists for.
//! 3. **Drop the glyph, not the label.** Against a freshly emptied atlas, a
//!    glyph that still will not fit never will ([`GlyphAtlas::fits`]), so it
//!    is dropped from the label and the rest of the string is still drawn.
//!    A label that lost glyphs only because THIS label overfilled the atlas is
//!    returned but not cached, so the next frame retries it.
//!
//! # The rasteriser's font cache
//!
//! `oxitext-raster`'s thread-local `fontdue::Font` cache is keyed by a 64-bit
//! FNV-1a hash of only the FIRST 64 BYTES of a font file
//! (`oxitext-raster-0.2.3/src/tl_cache.rs`) — the sfnt header plus three table
//! records. Two faces that agree there share ONE parsed font, and the second
//! face's glyph ids are then rasterised out of the first face's outlines: the
//! same class of aliasing the font interning below exists to prevent on this
//! side of the boundary. Nothing here can compensate, because the key is
//! derived from the bytes and even a copy of a face collides with it, so the
//! engine computes the same key when it interns a face and *reports* a
//! collision between two distinct faces instead of drawing it silently. The
//! fix belongs upstream (hash the whole file). The `FontdueRaster` fallback
//! path keyed on `face_data.as_ptr()` is not a second hazard: it is reached
//! only when the thread-local cache declines to parse the bytes at all, and it
//! then fails to parse them too, so it never serves a usable font from a
//! recycled address.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use oxitext::{
    LayoutResult, OxiTextError, Pipeline, RenderOutput, SwashShaper, TextAlignment, TextStyle,
};
use oxitext_raster::{FontdueRaster, RasterBackend};

use crate::error::RenderError;
use crate::label::atlas::{AtlasRect, GlyphAtlas, GlyphKey, MAX_ATLAS_SIZE};
use crate::label::vertical::{self, VerticalPlan, VerticalRefusal};

mod cache;

pub use crate::label::engine::cache::MAX_LABEL_CACHE;

use crate::label::engine::cache::{LabelCache, StyleKey};

/// Number of `(text, size)` pairs the shaped-label cache STARTS with.
///
/// Sized like `oxigis-ui`'s `DECODED_CACHE_TILES`: a screenful of vector tiles
/// carries a few hundred labels, so 512 holds a full viewport plus the ring of
/// tiles a pan is about to reveal. A frame that shapes more distinct strings
/// than this — the placer shapes every candidate, not just the ones it places
/// — grows the cache instead of thrashing against it, up to
/// [`MAX_LABEL_CACHE`]; an engine built with
/// [`LabelEngine::with_capacity`] keeps the size it was asked for.
pub const DEFAULT_LABEL_CACHE: usize = 512;

/// Largest pixels-per-em a label may be shaped at.
///
/// Guards the atlas against a runaway zoom expression: a single glyph at
/// 512 px is already a quarter of a 1024² atlas.
pub const MAX_LABEL_SIZE_PX: f32 = 512.0;

/// Largest ink box, in multiples of the em, a glyph may have before it is
/// dropped from its label unrasterised.
///
/// `fontdue` sizes its coverage buffer from the glyph's own outline, so a
/// broken or hostile face — and fonts arrive from the user's machine and from
/// the web shell's font fetch — can ask for a bitmap of tens of thousands of
/// texels a side at [`MAX_LABEL_SIZE_PX`]. Even a decorative swash stays
/// inside four ems, so eight is a bound no legitimate face notices.
const MAX_GLYPH_EM_EXTENT: f32 = 8.0;

/// Bytes of a font file `oxitext-raster`'s thread-local cache hashes into its
/// key; mirrored here to detect a collision, see the module docs.
const RASTER_KEY_BYTES: usize = 64;

/// One glyph of a [`ShapedLabel`]: where it sits and where its pixels are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelGlyph {
    /// Identity of the rasterised glyph — font, glyph id, size.
    pub key: GlyphKey,
    /// The glyph's rectangle inside the atlas, in texels.
    pub slot: AtlasRect,
    /// Top-left of the glyph box relative to the label box's top-left, in
    /// pixels. Always integral: see [`ShapedLabel`].
    pub offset_px: [f32; 2],
}

/// A shaped, rasterised and packed label, ready to be placed and drawn.
///
/// # Coordinates
///
/// The label box has its origin at the top-left, is [`ShapedLabel::size_px`]
/// big, and every glyph offset is relative to it — so placing a label is just
/// choosing that origin. The box height is the layout engine's line box
/// (ascent + descent + leading), not the ink bounding box, which is what makes
/// two labels of different strings collide consistently.
///
/// # Integral offsets
///
/// `fontdue` rasterises on the pixel grid with no subpixel phase, so a glyph
/// drawn at a fractional offset would be a blurred copy of the same bitmap.
/// Offsets are therefore rounded at shaping time, and the drawing pass rounds
/// the label origin too ([`crate::label::pipeline`]): together they guarantee
/// the 1:1 texel-to-pixel mapping that keeps small text crisp.
#[derive(Debug, Clone)]
pub struct ShapedLabel {
    size_px: [f32; 2],
    font_size_px: f32,
    generation: u32,
    glyphs: Vec<LabelGlyph>,
}

impl ShapedLabel {
    /// Width and height of the collision box, in pixels.
    #[must_use]
    pub fn size_px(&self) -> [f32; 2] {
        self.size_px
    }

    /// Width of the collision box, in pixels.
    #[must_use]
    pub fn width_px(&self) -> f32 {
        self.size_px[0]
    }

    /// Height of the collision box, in pixels.
    #[must_use]
    pub fn height_px(&self) -> f32 {
        self.size_px[1]
    }

    /// The pixels-per-em the label was shaped and rasterised at.
    #[must_use]
    pub fn font_size_px(&self) -> f32 {
        self.font_size_px
    }

    /// The engine generation this label's atlas slots belong to.
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// The drawable glyphs, in layout order. Whitespace and glyphs with no
    /// outline are absent.
    #[must_use]
    pub fn glyphs(&self) -> &[LabelGlyph] {
        &self.glyphs
    }

    /// Whether the label has nothing to draw — an empty or whitespace-only
    /// string, in which case its collision box is zero-sized too.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }
}

/// Which face chain a label is shaped through.
///
/// A render-local mirror of `oxigis_core::LabelWeight` — this crate
/// deliberately does not depend on `oxigis-core` (blueprint §3), and the
/// shell maps one onto the other where it already translates a style into a
/// [`crate::label::LabelSpec`].
///
/// Two values rather than a numeric weight, for the same reason the core
/// type has two: both this engine and the PDF exporter resolve a weight to a
/// REAL face and never to a synthetic emboldening, so the value has to be a
/// request a font chain can actually answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LabelWeight {
    /// The regular chain — what every pre-v1.4 caller means.
    #[default]
    Regular,
    /// The bold chain, when the shell supplied one.
    Bold,
}

/// Which way a label's glyphs run.
///
/// A render-local mirror of `oxigis_core::LabelOrientation`, for exactly the
/// reason [`LabelWeight`] is one: this crate deliberately does not depend on
/// `oxigis-core` (blueprint §3), and the shell maps one onto the other where
/// it already translates a style into a [`crate::label::LabelSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LabelOrientation {
    /// Left to right on one baseline — what every pre-v1.5 caller means.
    #[default]
    Horizontal,
    /// Top to bottom, one upright cell per character. Refused — and served
    /// [`LabelOrientation::Horizontal`] — for anything the ladder in
    /// [`crate::label::vertical`] cannot stack.
    Vertical,
}

impl LabelOrientation {
    /// Whether this is the default, horizontal orientation.
    #[must_use]
    pub fn is_horizontal(self) -> bool {
        matches!(self, Self::Horizontal)
    }
}

/// Ink metrics kept per packed glyph so a cache hit needs no re-rasterisation.
#[derive(Debug, Clone, Copy)]
struct GlyphInk {
    bearing_x: i32,
    /// `fontdue`'s `ymin`: baseline to bottom edge, positive up.
    ymin: i32,
    height: u32,
}

/// What a packing attempt does when the atlas has no room for a glyph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnOverflow {
    /// Stop and report [`PackOutcome::Full`] so the caller can free room.
    Report,
    /// Drop the glyph and carry on — the last rung, where there is no room
    /// left to free.
    DropGlyph,
}

/// A packed label plus what it cost to pack it.
#[derive(Debug)]
struct Packed {
    label: ShapedLabel,
    /// Glyphs no atlas will ever hold: an ink box past [`GlyphAtlas::fits`] or
    /// past [`MAX_GLYPH_EM_EXTENT`]. Deterministic, so a label carrying them
    /// is still worth caching.
    unpackable: usize,
    /// Glyphs dropped because the atlas was full at the time. The next frame
    /// may well have room, so a label carrying any of these is NOT cached.
    crowded_out: usize,
}

/// The result of one packing attempt.
#[derive(Debug)]
enum PackOutcome {
    Done(Packed),
    /// The atlas is full and the caller may still be able to free room.
    Full,
}

/// What making one glyph available in the atlas produced.
#[derive(Debug, Clone, Copy)]
enum GlyphSlot {
    /// Packed, with the ink metrics its offsets are built from.
    Ready(AtlasRect, GlyphInk),
    /// Nothing to draw: whitespace, an outline-less glyph, or a colour glyph
    /// the R8 pipeline does not carry.
    Blank,
    /// No atlas will ever hold it — see [`GlyphAtlas::fits`] and
    /// [`MAX_GLYPH_EM_EXTENT`].
    Unpackable,
    /// The atlas is full right now.
    Full,
}

/// What is being packed — the shaped layout of a horizontal label, or the
/// planned column of a vertical one with the face chain it resolved through.
///
/// Built ONCE, before the overflow ladder starts, so freeing atlas room and
/// packing again costs a re-pack and never a re-shape.
enum Shaped {
    Horizontal(LayoutResult),
    Vertical {
        plan: VerticalPlan,
        chain: Vec<Arc<[u8]>>,
    },
}

/// Faces parsed for the glyphs of ONE packing pass.
///
/// A glyph's ink box has to be read from its face before the rasteriser is
/// allowed to allocate for it, and `ttf_parser::Face::parse` is far too
/// expensive to repeat per glyph. A label draws from one or two faces, so a
/// linear probe over an inline `Vec` beats hashing.
///
/// Face index 0 throughout, as `vertical::face_for` and `fontdue` both use:
/// bounding a collection's face 0 while the rasteriser drew face 1 would bound
/// the wrong outlines.
struct FaceCache<'a> {
    faces: Vec<(usize, Option<ttf_parser::Face<'a>>)>,
}

impl<'a> FaceCache<'a> {
    fn new() -> Self {
        Self { faces: Vec::new() }
    }

    fn face(&mut self, font_data: &'a Arc<[u8]>) -> Option<&ttf_parser::Face<'a>> {
        let address = Arc::as_ptr(font_data).cast::<u8>() as usize;
        let index = match self.faces.iter().position(|(seen, _)| *seen == address) {
            Some(index) => index,
            None => {
                self.faces
                    .push((address, ttf_parser::Face::parse(font_data, 0).ok()));
                self.faces.len() - 1
            }
        };
        self.faces.get(index).and_then(|(_, face)| face.as_ref())
    }
}

/// Shaping engine: fonts in, [`ShapedLabel`]s and a packed [`GlyphAtlas`] out.
///
/// The engine owns its atlas so that a font change or an atlas rebuild can
/// invalidate both halves of the cached state in one step; borrow it with
/// [`LabelEngine::atlas`] / [`LabelEngine::atlas_mut`] to upload it.
///
/// Not [`Debug`]: neither `oxitext::Pipeline` nor `FontdueRaster` implements
/// it, and a manual impl would print a megabyte of font bytes. The observable
/// state is available through the accessors instead.
pub struct LabelEngine {
    pipeline: Pipeline,
    rasterizer: FontdueRaster,
    atlas: GlyphAtlas,
    /// Interned font bytes; the index is [`GlyphKey::font`]. Holding the `Arc`
    /// keeps the address in `font_ids` from being recycled by another font.
    fonts: Vec<Arc<[u8]>>,
    font_ids: HashMap<usize, u16>,
    /// Bearings of every glyph currently packed in the atlas.
    ink: HashMap<GlyphKey, GlyphInk>,
    /// The primary face's bytes, kept because the BOLD chain's never-shrink
    /// tail is `bold faces ++ primary ++ fallbacks`.
    ///
    /// `Arc` since v1.5: the vertical path resolves faces by cmap itself and
    /// interns the very bytes it shaped through, so it needs a handle whose
    /// address is stable for as long as the chain is. The `Vec` handed in by
    /// the public setters is moved into the `Arc` once — no extra copy.
    primary: Arc<[u8]>,
    /// Fallback chain as handed in, kept so `add_fallback_font` can append.
    fallbacks: Vec<Arc<[u8]>>,
    /// The bold face chain as handed in, highest priority first. Empty means
    /// a bold request draws Regular.
    bold_faces: Vec<Arc<[u8]>>,
    /// The bold pipeline, rebuilt whenever either chain changes. [`None`]
    /// when `bold_faces` is empty or its first face does not parse.
    bold: Option<Pipeline>,
    /// Whether the "asked for bold, have none" line has already been logged
    /// for the current chain — once per chain, not once per label.
    bold_missing_logged: bool,
    /// The shaper the vertical path drives, built on the FIRST vertical
    /// label and never for a horizontal-only session.
    ///
    /// One shaper for the whole engine is safe here for a stated reason
    /// rather than a hoped one: the reused-shaper corruption the exporter's
    /// canary pins needs swash's `EngineMode::Complex`, and the ladder's
    /// upright-only first rung makes that unreachable (see
    /// [`crate::label::vertical`]).
    shaper: Option<SwashShaper>,
    /// Vertical refusals already logged for the current generation — one
    /// line per reason, never one per label (a screenful is hundreds).
    vertical_refusals_logged: HashSet<VerticalRefusal>,
    /// Whether the "a label lost glyphs" line has already been said for the
    /// current generation.
    dropped_glyphs_logged: bool,
    /// The same, for the "this glyph's ink box is absurd" line.
    oversized_glyph_logged: bool,
    /// Whether two faces have already been reported as indistinguishable to
    /// the rasteriser's cache. Not per generation: the faces do not change
    /// when the atlas is rebuilt, so neither does the answer.
    raster_key_collision_logged: bool,
    /// The key `oxitext-raster`'s font cache would give each interned face,
    /// for the collision report described in the module docs.
    raster_keys: HashMap<u64, u16>,
    cache: LabelCache,
    generation: u32,
}

impl LabelEngine {
    /// Builds an engine over `primary_font` (raw TTF/OTF bytes).
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Text`] if the bytes are not a font `oxitext` can
    /// parse.
    pub fn new(primary_font: Vec<u8>) -> Result<Self, RenderError> {
        Self::build(
            primary_font,
            LabelCache::new(DEFAULT_LABEL_CACHE, MAX_LABEL_CACHE),
        )
    }

    /// Builds an engine whose shaped-label cache holds `capacity` entries.
    ///
    /// Unlike [`LabelEngine::new`], the size is **fixed**: an explicit
    /// capacity is a request, not a starting point, so the adaptive growth
    /// [`DEFAULT_LABEL_CACHE`] describes does not apply.
    /// [`LabelEngine::reserve_labels`] raises it again if the caller wants it
    /// raised.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidCapacity`] if `capacity` is zero and
    /// [`RenderError::Text`] if the font bytes cannot be parsed.
    pub fn with_capacity(primary_font: Vec<u8>, capacity: usize) -> Result<Self, RenderError> {
        if capacity == 0 {
            return Err(RenderError::InvalidCapacity(capacity));
        }
        Self::build(primary_font, LabelCache::new(capacity, capacity))
    }

    fn build(primary_font: Vec<u8>, cache: LabelCache) -> Result<Self, RenderError> {
        let pipeline = Pipeline::from_bytes(&primary_font).map_err(text_error)?;
        Ok(Self {
            pipeline,
            rasterizer: FontdueRaster::new(),
            atlas: GlyphAtlas::new()?,
            fonts: Vec::new(),
            font_ids: HashMap::new(),
            ink: HashMap::new(),
            primary: Arc::from(primary_font),
            fallbacks: Vec::new(),
            bold_faces: Vec::new(),
            bold: None,
            bold_missing_logged: false,
            shaper: None,
            vertical_refusals_logged: HashSet::new(),
            dropped_glyphs_logged: false,
            oversized_glyph_logged: false,
            raster_key_collision_logged: false,
            raster_keys: HashMap::new(),
            cache,
            generation: 0,
        })
    }

    /// Replaces the fallback chain and invalidates everything shaped so far.
    ///
    /// Each entry is raw font bytes. `oxitext` re-shapes any cluster the
    /// primary font maps to `.notdef` through this chain in order.
    pub fn set_fallback_fonts(&mut self, fonts: Vec<Vec<u8>>) {
        self.pipeline.set_fallback_fonts(fonts.clone());
        self.fallbacks = fonts.into_iter().map(Arc::from).collect();
        self.rebuild_bold();
        self.invalidate();
    }

    /// Appends one font to the fallback chain.
    ///
    /// This is the runtime path: the web shell fetches a CJK font after
    /// startup and calls this when the bytes arrive.
    pub fn add_fallback_font(&mut self, font: Vec<u8>) {
        self.add_fallback_fonts(vec![font]);
    }

    /// Appends several fonts to the fallback chain in one step.
    ///
    /// One re-shape generation and one hand-off to `oxitext` no matter how
    /// many fonts land together — the hand-off clones every accumulated
    /// fallback's bytes (`oxitext`'s `set_fallback_fonts` takes owned
    /// `Vec<Vec<u8>>` and re-wraps them in fresh `Arc`s), so a shell draining
    /// a multi-font chain in one frame should prefer this over N single
    /// `add_fallback_font` calls, which would clone the growing chain N
    /// times. An empty `fonts` is a no-op: nothing changes, nothing
    /// invalidates.
    pub fn add_fallback_fonts(&mut self, fonts: Vec<Vec<u8>>) {
        if fonts.is_empty() {
            return;
        }
        self.fallbacks.extend(fonts.into_iter().map(Arc::from));
        self.pipeline.set_fallback_fonts(owned(&self.fallbacks));
        self.rebuild_bold();
        self.invalidate();
    }

    /// Number of fonts in the fallback chain.
    #[must_use]
    pub fn fallback_count(&self) -> usize {
        self.fallbacks.len()
    }

    /// Replaces the BOLD face chain, highest priority first, and invalidates
    /// everything shaped so far.
    ///
    /// The resulting shaping chain is **never-shrinking**: the bold faces in
    /// the order given, then the regular primary, then the regular fallbacks.
    /// A bold Latin face therefore still gets CJK glyphs from the regular CJK
    /// face — a mixed-weight line, which is honest, rather than `.notdef`
    /// boxes, which are not. Passing an empty chain removes bold support;
    /// bold requests then draw Regular with one log.
    ///
    /// An unparseable FIRST face disables bold with a warning rather than
    /// failing: a font the OS offered and `oxitext` rejects must not cost the
    /// map its labels.
    pub fn set_bold_fonts(&mut self, fonts: Vec<Vec<u8>>) {
        if fonts.is_empty() && self.bold_faces.is_empty() {
            return;
        }
        self.bold_faces = fonts.into_iter().map(Arc::from).collect();
        self.bold_missing_logged = false;
        self.rebuild_bold();
        self.invalidate();
    }

    /// Number of faces in the bold chain, before the regular tail.
    #[must_use]
    pub fn bold_face_count(&self) -> usize {
        self.bold_faces.len()
    }

    /// Whether a [`LabelWeight::Bold`] request will actually draw bold.
    #[must_use]
    pub fn has_bold(&self) -> bool {
        self.bold.is_some()
    }

    /// Rebuilds the bold pipeline from `bold_faces` plus the current regular
    /// chain — called whenever EITHER chain changes, because the bold chain's
    /// tail is the regular chain.
    fn rebuild_bold(&mut self) {
        self.bold = None;
        let Some((first, rest)) = self.bold_faces.split_first() else {
            return;
        };
        let mut pipeline = match Pipeline::from_bytes(first.as_ref()) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                tracing::warn!(
                    %error,
                    "oxigis-render: the bold label face did not parse; labels stay Regular",
                );
                return;
            }
        };
        let mut chain: Vec<Vec<u8>> = owned(rest);
        chain.push(self.primary.to_vec());
        chain.extend(self.fallbacks.iter().map(|face| face.to_vec()));
        pipeline.set_fallback_fonts(chain);
        self.bold = Some(pipeline);
    }

    /// The weight a request will really be served at, logging the demotion
    /// ONCE per bold chain rather than once per label.
    fn effective_weight(&mut self, requested: LabelWeight) -> LabelWeight {
        match requested {
            LabelWeight::Regular => LabelWeight::Regular,
            LabelWeight::Bold if self.bold.is_some() => LabelWeight::Bold,
            LabelWeight::Bold => {
                if !self.bold_missing_logged {
                    self.bold_missing_logged = true;
                    tracing::info!(
                        "oxigis-render: bold labels were requested but no bold face is \
                         available; they draw Regular (never a synthetic emboldening)",
                    );
                }
                LabelWeight::Regular
            }
        }
    }

    /// The chain for an ALREADY-effective weight: bold faces, then the
    /// primary, then the fallbacks — the same order [`Self::rebuild_bold`]
    /// builds the bold pipeline in, so a vertical label and a horizontal one
    /// pick the same face for the same character by construction.
    ///
    /// Owned rather than borrowed because the caller needs `&mut self.shaper`
    /// at the same time; the elements are `Arc` handles, so the copy is a
    /// refcount bump per face and never a font.
    fn chain_for(&self, weight: LabelWeight) -> Vec<Arc<[u8]>> {
        let mut chain: Vec<Arc<[u8]>> = Vec::with_capacity(self.fallbacks.len() + 2);
        if weight == LabelWeight::Bold {
            chain.extend(self.bold_faces.iter().map(Arc::clone));
        }
        chain.push(Arc::clone(&self.primary));
        chain.extend(self.fallbacks.iter().map(Arc::clone));
        chain
    }

    /// The orientation a request will really be served at, decided WITHOUT
    /// shaping so the cache key is deterministic.
    ///
    /// This is the font-free half of the ladder — empty text, right-to-left
    /// script, and any character UAX #50 rotates. The font-dependent rungs
    /// (no `vmtx`, an inexpressible cluster, an implausible pitch or origin)
    /// can only be answered by shaping, so a request that clears this half
    /// keeps [`LabelOrientation::Vertical`] in its key and falls back at
    /// build time, sharing the horizontal label's `Arc` under both keys.
    fn effective_orientation(
        &mut self,
        text: &str,
        requested: LabelOrientation,
    ) -> LabelOrientation {
        if requested.is_horizontal() {
            return LabelOrientation::Horizontal;
        }
        let refusal = if text.is_empty() {
            Some(VerticalRefusal::Unusable)
        } else if vertical::has_rtl(text) {
            Some(VerticalRefusal::RightToLeft)
        } else if !text
            .chars()
            .all(|ch| crate::label::vertical_orientation_of(ch).draws_upright())
        {
            Some(VerticalRefusal::RotatedCharacter)
        } else {
            None
        };
        match refusal {
            Some(refusal) => {
                self.log_vertical_refusal(refusal);
                LabelOrientation::Horizontal
            }
            None => LabelOrientation::Vertical,
        }
    }

    /// Says a vertical refusal ONCE per (generation, reason).
    fn log_vertical_refusal(&mut self, refusal: VerticalRefusal) {
        if self.vertical_refusals_logged.insert(refusal) {
            tracing::info!(
                reason = refusal.reason(),
                "oxigis-render: vertical labels were requested but cannot be stacked; \
                 they draw horizontally",
            );
        }
    }

    /// Plans a vertical column, or [`None`] with one aggregated log.
    fn vertical_plan(
        &mut self,
        text: &str,
        size_px: f32,
        weight: LabelWeight,
    ) -> Option<VerticalPlan> {
        let chain = self.chain_for(weight);
        let shaper = self.shaper.get_or_insert_with(SwashShaper::new);
        match vertical::plan_vertical(shaper, &chain, text, size_px) {
            Ok(plan) => Some(plan),
            Err(refusal) => {
                self.log_vertical_refusal(refusal);
                None
            }
        }
    }

    /// Counter identifying the current font set and atlas contents.
    ///
    /// Bumped whenever the fallback chain changes or the atlas is rebuilt; a
    /// [`ShapedLabel`] from an older generation must be discarded.
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// The glyph atlas, for reading pixels or querying occupancy.
    #[must_use]
    pub fn atlas(&self) -> &GlyphAtlas {
        &self.atlas
    }

    /// The glyph atlas, mutably — for
    /// [`crate::label::LabelPipeline::upload_atlas`], which clears the dirty
    /// flag once the pixels have reached the GPU.
    pub fn atlas_mut(&mut self) -> &mut GlyphAtlas {
        &mut self.atlas
    }

    /// Number of shaped labels currently cached.
    #[must_use]
    pub fn cached_labels(&self) -> usize {
        self.cache.len()
    }

    /// Maximum number of shaped labels the cache holds *right now* — a moving
    /// number for an engine from [`LabelEngine::new`], see
    /// [`DEFAULT_LABEL_CACHE`].
    #[must_use]
    pub fn cache_capacity(&self) -> usize {
        self.cache.capacity()
    }

    /// Sizes the cache for a pass that is about to shape `labels` distinct
    /// strings, bounded by [`MAX_LABEL_CACHE`].
    ///
    /// The cache learns a frame's demand on its own, but only after a frame
    /// has thrashed against it; a placer that knows its candidate count can
    /// say so up front and skip that frame. Capacity only ever goes up, and
    /// asking for less than the cache already holds is a no-op.
    pub fn reserve_labels(&mut self, labels: usize) {
        self.cache.reserve(labels);
    }

    /// Measures `text` at `size_px` without rasterising or packing anything.
    ///
    /// Returns `[width, height]` of the same box [`ShapedLabel::size_px`] would
    /// report, which is what a placement pass needs when it only wants to know
    /// whether a label *could* fit.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Text`] if `size_px` is not a finite value in
    /// `0.0 < size_px <=` [`MAX_LABEL_SIZE_PX`], or if shaping fails.
    pub fn measure(&mut self, text: &str, size_px: f32) -> Result<[f32; 2], RenderError> {
        self.measure_weighted(text, size_px, LabelWeight::Regular)
    }

    /// [`LabelEngine::measure`] at an explicit weight.
    ///
    /// # Errors
    ///
    /// As [`LabelEngine::measure`].
    pub fn measure_weighted(
        &mut self,
        text: &str,
        size_px: f32,
        weight: LabelWeight,
    ) -> Result<[f32; 2], RenderError> {
        self.measure_oriented(text, size_px, weight, LabelOrientation::Horizontal)
    }

    /// [`LabelEngine::measure`] at an explicit weight AND orientation.
    ///
    /// A vertical box is one em wide and the summed cell pitch tall; a
    /// vertical request the ladder refuses measures the HORIZONTAL box, which
    /// is what such a label will be drawn as.
    ///
    /// # Errors
    ///
    /// As [`LabelEngine::measure`].
    pub fn measure_oriented(
        &mut self,
        text: &str,
        size_px: f32,
        weight: LabelWeight,
        orientation: LabelOrientation,
    ) -> Result<[f32; 2], RenderError> {
        let style = label_style(size_px)?;
        let weight = self.effective_weight(weight);
        if self.effective_orientation(text, orientation) == LabelOrientation::Vertical
            && let Some(plan) = self.vertical_plan(text, size_px, weight)
        {
            return Ok(plan.size_px);
        }
        let metrics = self
            .pipeline_for(weight)
            .measure(text, &style)
            .map_err(text_error)?;
        if metrics.total_width <= 0.0 {
            return Ok([0.0, 0.0]);
        }
        Ok([metrics.total_width, metrics.total_height])
    }

    /// The pipeline serving an ALREADY-effective weight — `bold` is
    /// [`Some`] whenever the caller resolved to [`LabelWeight::Bold`], so the
    /// fallback arm is unreachable defensiveness rather than a policy.
    fn pipeline_for(&mut self, weight: LabelWeight) -> &mut Pipeline {
        match weight {
            LabelWeight::Bold => self.bold.as_mut().unwrap_or(&mut self.pipeline),
            LabelWeight::Regular => &mut self.pipeline,
        }
    }

    /// Shapes, rasterises and packs `text` at `size_px`, or returns the cached
    /// result of having done so.
    ///
    /// Text is laid out on a single line unless it contains an explicit
    /// newline — no automatic wrapping, because a map label's width is a
    /// placement input, not a constraint (`TextStyle::max_width` is 0).
    ///
    /// If the atlas fills up mid-label, the glyphs nothing is drawing are
    /// freed and the label is packed again — invisibly, since freeing moves
    /// nothing. Only when every packed glyph is still live is the atlas
    /// cleared and rebuilt, which bumps [`LabelEngine::generation`] and
    /// invalidates every previously returned [`ShapedLabel`]. A glyph too
    /// large to ever fit the atlas is dropped from the label rather than
    /// failing it; see the module docs for the whole ladder.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Text`] for an out-of-range `size_px`, for a
    /// shaping failure, or for a rasteriser whose bitmap dimensions and pixel
    /// count disagree.
    pub fn shape(&mut self, text: &str, size_px: f32) -> Result<Arc<ShapedLabel>, RenderError> {
        self.shape_weighted(text, size_px, LabelWeight::Regular)
    }

    /// [`LabelEngine::shape`] at an explicit weight — the same cache, keyed
    /// by the weight as well, so a bold request can never be served the
    /// regular label that happened to be shaped first.
    ///
    /// A [`LabelWeight::Bold`] request with no bold chain shapes Regular (one
    /// log, see [`LabelEngine::set_bold_fonts`]) and shares the regular
    /// entry.
    ///
    /// # Errors
    ///
    /// As [`LabelEngine::shape`].
    pub fn shape_weighted(
        &mut self,
        text: &str,
        size_px: f32,
        weight: LabelWeight,
    ) -> Result<Arc<ShapedLabel>, RenderError> {
        self.shape_oriented(text, size_px, weight, LabelOrientation::Horizontal)
    }

    /// [`LabelEngine::shape`] at an explicit weight AND orientation — the
    /// same cache, keyed by both, so a vertical request can never be served
    /// the horizontal label that happened to be shaped first.
    ///
    /// A [`LabelOrientation::Vertical`] request the ladder refuses draws the
    /// horizontal label (one log per generation and reason, see
    /// [`crate::label::vertical::VerticalRefusal`]) and shares its `Arc` —
    /// and, when the font-free half of the ladder was what refused, its cache
    /// entry too.
    ///
    /// # Errors
    ///
    /// As [`LabelEngine::shape`].
    pub fn shape_oriented(
        &mut self,
        text: &str,
        size_px: f32,
        weight: LabelWeight,
        orientation: LabelOrientation,
    ) -> Result<Arc<ShapedLabel>, RenderError> {
        let style = label_style(size_px)?;
        let weight = self.effective_weight(weight);
        let orientation = self.effective_orientation(text, orientation);
        let key = StyleKey {
            size_bits: size_px.to_bits(),
            weight,
            orientation,
        };
        if let Some(label) = self.cache.get(key, text) {
            return Ok(label);
        }
        // A vertical request that clears the font-free half still has to
        // survive shaping. A refusal here falls back to the horizontal
        // label — and shares its entry, so the fallback is planned once per
        // (text, size, weight) and not once per frame.
        let plan = match orientation {
            LabelOrientation::Vertical => match self.vertical_plan(text, size_px, weight) {
                Some(plan) => Some(plan),
                None => {
                    let horizontal = StyleKey {
                        orientation: LabelOrientation::Horizontal,
                        ..key
                    };
                    if let Some(label) = self.cache.get(horizontal, text) {
                        self.cache.insert(key, text, &label);
                        return Ok(label);
                    }
                    None
                }
            },
            LabelOrientation::Horizontal => None,
        };
        // Shaped once, packed as many times as the overflow ladder needs.
        let shaped = match plan {
            Some(plan) => Shaped::Vertical {
                plan,
                chain: self.chain_for(weight),
            },
            None => Shaped::Horizontal(self.shape_layout(text, &style, weight)?),
        };
        let refused_vertical =
            orientation == LabelOrientation::Vertical && matches!(shaped, Shaped::Horizontal(_));
        let packed = self.pack_with_recovery(&shaped, size_px)?;
        if packed.unpackable > 0 || packed.crowded_out > 0 {
            self.log_dropped_glyphs(text, &packed);
        }
        let label = Arc::new(packed.label);
        if packed.crowded_out > 0 {
            // Only THIS label's own glyphs overfilled a freshly emptied
            // atlas, which the next frame — with a different label set — may
            // well have room for. Caching it would freeze the loss for the
            // rest of the session, so it is tracked but not keyed.
            self.cache.hold(&label);
            return Ok(label);
        }
        if refused_vertical {
            // The refused case: ONE label, reachable under both keys, so the
            // atlas cost and the shaping cost are paid exactly once.
            self.cache.insert(
                StyleKey {
                    orientation: LabelOrientation::Horizontal,
                    ..key
                },
                text,
                &label,
            );
        }
        self.cache.insert(key, text, &label);
        Ok(label)
    }

    /// Drops every cached label and empties the atlas, bumping the generation.
    ///
    /// Useful when a shell knows its label set changed wholesale (a style
    /// switch, a project reload) and would rather not pay for stale glyphs.
    pub fn clear(&mut self) {
        self.invalidate();
    }

    /// Shapes `text` once, whatever the overflow ladder then does with it.
    ///
    /// Split from packing so that freeing atlas room and packing again costs a
    /// re-pack and never a re-shape — shaping (cmap, GSUB/GPOS) is the
    /// expensive half.
    fn shape_layout(
        &mut self,
        text: &str,
        style: &TextStyle,
        weight: LabelWeight,
    ) -> Result<LayoutResult, RenderError> {
        // The chosen pipeline's borrow ends with the statement; everything
        // below (interning, the atlas) needs `self` again. The atlas is the
        // ONE atlas either weight packs into — a bold face is a different
        // `Arc`, so it interns to a different font index and the keys of the
        // two weights cannot collide.
        self.pipeline_for(weight)
            .shape_and_layout(text, style)
            .map_err(text_error)
    }

    /// A label with nothing to draw and a zero-area collision box.
    fn empty_label(&self, size_px: f32) -> ShapedLabel {
        ShapedLabel {
            size_px: [0.0, 0.0],
            font_size_px: size_px,
            generation: self.generation,
            glyphs: Vec::new(),
        }
    }

    /// Packs a shaped label, freeing atlas room as it has to — the ladder the
    /// module docs describe, and the only place the atlas is ever cleared.
    fn pack_with_recovery(&mut self, shaped: &Shaped, size_px: f32) -> Result<Packed, RenderError> {
        let mut rung = 0u8;
        loop {
            // The last rung runs against a freshly emptied atlas, so a glyph
            // that fails there fails for good and is dropped rather than
            // costing the label another round.
            let on_overflow = if rung == 2 {
                OnOverflow::DropGlyph
            } else {
                OnOverflow::Report
            };
            let outcome = match shaped {
                Shaped::Horizontal(layout) => self.pack_horizontal(layout, size_px, on_overflow)?,
                Shaped::Vertical { plan, chain } => {
                    self.pack_vertical(plan, chain, size_px, on_overflow)?
                }
            };
            if let PackOutcome::Done(packed) = outcome {
                return Ok(packed);
            }
            // Freeing the glyphs nothing is drawing costs no caller anything;
            // a rebuild costs every outstanding label. Only take the second
            // when the first found nothing to free.
            if rung == 0 && self.compact() > 0 {
                rung = 1;
                continue;
            }
            self.invalidate();
            rung = 2;
        }
    }

    /// Rasterises and packs a shaped layout.
    fn pack_horizontal(
        &mut self,
        layout: &LayoutResult,
        size_px: f32,
        on_overflow: OnOverflow,
    ) -> Result<PackOutcome, RenderError> {
        if layout.metrics.total_width <= 0.0 {
            // Empty or whitespace-only: a zero-area collision box, no glyphs.
            return Ok(PackOutcome::Done(Packed {
                label: self.empty_label(size_px),
                unpackable: 0,
                crowded_out: 0,
            }));
        }

        let mut faces = FaceCache::new();
        let mut glyphs = Vec::with_capacity(layout.glyphs.len());
        let mut unpackable = 0usize;
        let mut crowded_out = 0usize;
        for glyph in &layout.glyphs {
            let font = self.intern_font(&glyph.font_data);
            let key = GlyphKey::new(font, glyph.gid, glyph.font_size);
            let (slot, ink) =
                match self.place_glyph(key, &glyph.font_data, glyph.font_size, &mut faces)? {
                    GlyphSlot::Ready(slot, ink) => (slot, ink),
                    GlyphSlot::Blank => continue,
                    GlyphSlot::Unpackable => {
                        unpackable += 1;
                        continue;
                    }
                    GlyphSlot::Full => {
                        if on_overflow == OnOverflow::Report {
                            return Ok(PackOutcome::Full);
                        }
                        crowded_out += 1;
                        continue;
                    }
                };

            // `pos` is the pen origin on the baseline, y down from the label
            // box's top-left. The ink box starts one left bearing to the right
            // of the pen and `height + ymin` above the baseline.
            let left = glyph.pos.0.round() + ink.bearing_x as f32;
            let top = glyph.pos.1.round() - (ink.height as f32 + ink.ymin as f32);
            glyphs.push(LabelGlyph {
                key,
                slot,
                offset_px: [left, top.round()],
            });
        }

        Ok(PackOutcome::Done(Packed {
            label: ShapedLabel {
                size_px: [layout.metrics.total_width, layout.metrics.total_height],
                font_size_px: size_px,
                generation: self.generation,
                glyphs,
            },
            unpackable,
            crowded_out,
        }))
    }

    /// Rasterises and packs an accepted vertical column.
    ///
    /// The offsets are D-A6's geometry with the glyph's own ink folded in:
    /// `x_shift + bearing_x` across the column, `baseline − (height + ymin)`
    /// down it — the SAME expression the horizontal path uses, so a glyph
    /// sits on its vertical origin exactly as it sits on its baseline.
    /// Non-negative offsets are deliberately NOT asserted: a `vmtx` top side
    /// bearing is an `i16` and may be negative, so ink can legitimately sit
    /// above its cell, exactly as a horizontal accent can exceed the line box.
    fn pack_vertical(
        &mut self,
        plan: &VerticalPlan,
        chain: &[Arc<[u8]>],
        size_px: f32,
        on_overflow: OnOverflow,
    ) -> Result<PackOutcome, RenderError> {
        let Some(bytes) = chain.get(plan.chain_index) else {
            return Ok(PackOutcome::Done(Packed {
                label: self.empty_label(size_px),
                unpackable: 0,
                crowded_out: 0,
            }));
        };
        let bytes = Arc::clone(bytes);
        let font = self.intern_font(&bytes);
        let mut faces = FaceCache::new();
        let mut glyphs = Vec::with_capacity(plan.cells.len());
        let mut unpackable = 0usize;
        let mut crowded_out = 0usize;
        for cell in &plan.cells {
            let key = GlyphKey::new(font, cell.gid, size_px);
            let (slot, ink) = match self.place_glyph(key, &bytes, size_px, &mut faces)? {
                GlyphSlot::Ready(slot, ink) => (slot, ink),
                // An inkless cell still cost its pitch.
                GlyphSlot::Blank => continue,
                GlyphSlot::Unpackable => {
                    unpackable += 1;
                    continue;
                }
                GlyphSlot::Full => {
                    if on_overflow == OnOverflow::Report {
                        return Ok(PackOutcome::Full);
                    }
                    crowded_out += 1;
                    continue;
                }
            };
            let left = cell.x_shift_px + ink.bearing_x as f32;
            let top = cell.baseline_px - (ink.height as f32 + ink.ymin as f32);
            glyphs.push(LabelGlyph {
                key,
                slot,
                offset_px: [left.round(), top.round()],
            });
        }
        Ok(PackOutcome::Done(Packed {
            label: ShapedLabel {
                size_px: plan.size_px,
                font_size_px: size_px,
                generation: self.generation,
                glyphs,
            },
            unpackable,
            crowded_out,
        }))
    }

    /// Makes one glyph available in the atlas: its packed slot when it is
    /// already there, otherwise a *bounded* rasterisation and one packing
    /// attempt.
    fn place_glyph<'a>(
        &mut self,
        key: GlyphKey,
        font_data: &'a Arc<[u8]>,
        size_px: f32,
        faces: &mut FaceCache<'a>,
    ) -> Result<GlyphSlot, RenderError> {
        if let (Some(slot), Some(ink)) = (self.atlas.get(&key), self.ink.get(&key).copied()) {
            return Ok(GlyphSlot::Ready(slot, ink));
        }
        if !self.glyph_ink_is_bounded(font_data, key.gid, size_px, faces) {
            return Ok(GlyphSlot::Unpackable);
        }
        let raster = self.rasterizer.rasterize_full(font_data, key.gid, size_px);
        // Colour glyphs (COLR/CBDT) and any SDF variant are not part of the
        // v1 R8 pipeline: skip them, do not fail.
        let RenderOutput::Greyscale(bitmap) = &raster.output else {
            return Ok(GlyphSlot::Blank);
        };
        if bitmap.width == 0 || bitmap.height == 0 {
            // Whitespace, or a glyph with no outline.
            return Ok(GlyphSlot::Blank);
        }
        // The second gate on what the face's own box promised: a face that
        // under-reports its outline still cannot be packed, and the atlas is
        // the authority on that.
        if !self.atlas.fits(bitmap.width, bitmap.height) {
            return Ok(GlyphSlot::Unpackable);
        }
        let Some(slot) = self
            .atlas
            .try_insert(key, bitmap.width, bitmap.height, &bitmap.pixels)?
        else {
            return Ok(GlyphSlot::Full);
        };
        let ink = GlyphInk {
            bearing_x: raster.bearing_x,
            ymin: raster.bearing_y,
            height: bitmap.height,
        };
        self.ink.insert(key, ink);
        Ok(GlyphSlot::Ready(slot, ink))
    }

    /// Whether `gid`'s ink box at `size_px` is worth allocating for.
    ///
    /// Read from the face BEFORE the rasteriser runs: `fontdue` sizes its
    /// coverage buffer from the outline itself, so this is the only point at
    /// which a glyph whose outline spans thousands of ems can be stopped
    /// before it costs hundreds of megabytes. A face that will not parse here
    /// is left to the rasteriser, which will not parse it either.
    fn glyph_ink_is_bounded<'a>(
        &mut self,
        font_data: &'a Arc<[u8]>,
        gid: u16,
        size_px: f32,
        faces: &mut FaceCache<'a>,
    ) -> bool {
        let limit = (size_px * MAX_GLYPH_EM_EXTENT).min(MAX_ATLAS_SIZE as f32);
        let extent = {
            let Some(face) = faces.face(font_data) else {
                return true;
            };
            let units_per_em = f32::from(face.units_per_em());
            if units_per_em <= 0.0 {
                return true;
            }
            let Some(bbox) = face.glyph_bounding_box(ttf_parser::GlyphId(gid)) else {
                // No outline at all: there is nothing to allocate.
                return true;
            };
            let scale = size_px / units_per_em;
            [
                f32::from(bbox.x_max.saturating_sub(bbox.x_min)) * scale,
                f32::from(bbox.y_max.saturating_sub(bbox.y_min)) * scale,
            ]
        };
        // `fontdue` rounds the scaled box outward, so a texel of slack keeps a
        // glyph sitting exactly on the limit from being refused for a rounding
        // step.
        if extent[0] <= limit + 2.0 && extent[1] <= limit + 2.0 {
            return true;
        }
        if !self.oversized_glyph_logged {
            self.oversized_glyph_logged = true;
            tracing::warn!(
                gid,
                width_px = extent[0],
                height_px = extent[1],
                limit_px = limit,
                "oxigis-render: a glyph's ink box is far larger than its em; it is \
                 dropped from the label unrasterised (a broken or hostile face)",
            );
        }
        false
    }

    /// Says ONCE per generation that a label was drawn without some of its
    /// glyphs — a screenful under atlas pressure is hundreds of labels, and
    /// one line per label would be the whole log.
    fn log_dropped_glyphs(&mut self, text: &str, packed: &Packed) {
        if self.dropped_glyphs_logged {
            return;
        }
        self.dropped_glyphs_logged = true;
        tracing::warn!(
            chars = text.chars().count(),
            too_large = packed.unpackable,
            no_room = packed.crowded_out,
            atlas_px = self.atlas.size(),
            "oxigis-render: a label is drawn without some of its glyphs; those too \
             large for the atlas are dropped for good, those that only met a full \
             atlas are re-tried on the next frame",
        );
    }

    /// Frees the atlas slots of every glyph no live label points at, and drops
    /// the cache entries that pointed at them. Returns how many glyphs went.
    ///
    /// Zero means everything packed is still being drawn and only
    /// [`Self::invalidate`] can make room.
    ///
    /// # What "live" means
    ///
    /// A [`ShapedLabel`] the cache is the ONLY holder of is not on screen: no
    /// caller can obtain one without going through `&mut self`. Everything
    /// else — a cached label a caller also holds, and a label the LRU evicted
    /// while a caller held it, which is why the cache keeps those instead of
    /// dropping them — is or may be mid-draw, so its glyphs stay. Nothing
    /// moves and the generation does not change, so those labels stay exactly
    /// as valid as they were.
    fn compact(&mut self) -> usize {
        self.cache.sweep_held();
        let mut live: HashSet<GlyphKey> = HashSet::new();
        for label in self.cache.live_labels() {
            if Arc::strong_count(label) <= 1 {
                continue;
            }
            live.extend(label.glyphs.iter().map(|glyph| glyph.key));
        }
        let evicted = self.atlas.retain(|key| live.contains(key));
        if evicted == 0 {
            return 0;
        }
        self.ink.retain(|key, _| live.contains(key));
        // An entry pointing at a freed slot would serve another glyph's ink on
        // its next cache hit, so it goes with the glyphs it named.
        self.cache
            .retain(|label| label.glyphs.iter().all(|glyph| live.contains(&glyph.key)));
        evicted
    }

    /// Maps an `oxitext` font handle to a stable small index, keeping the
    /// `Arc` alive so the address cannot be reused by a different font.
    fn intern_font(&mut self, font_data: &Arc<[u8]>) -> u16 {
        let address = Arc::as_ptr(font_data).cast::<u8>() as usize;
        if let Some(index) = self.font_ids.get(&address) {
            return *index;
        }
        // Beyond 65 535 distinct fonts in one session the index saturates and
        // two fonts share atlas slots; a map style with that many fonts is not
        // a case worth a fallible signature.
        let index = u16::try_from(self.fonts.len()).unwrap_or(u16::MAX);
        // Before the push, so `fonts` still holds only the faces this one
        // could be colliding with.
        self.report_raster_key_collision(font_data, index);
        self.fonts.push(Arc::clone(font_data));
        self.font_ids.insert(address, index);
        index
    }

    /// Reports two distinct faces that `oxitext-raster`'s thread-local font
    /// cache cannot tell apart, once per engine.
    ///
    /// The key is the upstream one (see the module docs), so this is the exact
    /// condition rather than an approximation of it. Two faces with identical
    /// bytes share that cache entry legitimately — same outlines, same ink —
    /// and are not reported.
    fn report_raster_key_collision(&mut self, font_data: &Arc<[u8]>, index: u16) {
        let key = raster_cache_key(font_data);
        // The FIRST face to claim a key keeps it: that is the one the
        // rasteriser's cache actually parsed and would keep serving, so every
        // later face has to be compared against it and not against whichever
        // collided most recently.
        let other = match self.raster_keys.entry(key) {
            Entry::Vacant(slot) => {
                slot.insert(index);
                return;
            }
            Entry::Occupied(claimed) => *claimed.get(),
        };
        let collides = self
            .fonts
            .get(usize::from(other))
            .is_some_and(|seen| seen.as_ref() != font_data.as_ref());
        if !collides || self.raster_key_collision_logged {
            return;
        }
        self.raster_key_collision_logged = true;
        tracing::warn!(
            key,
            "oxigis-render: two different fonts share the 64-byte key \
             `oxitext-raster` caches parsed faces by, so one face's glyphs may be \
             rasterised from the other's outlines; drop one of the faces or fix \
             the key upstream",
        );
    }

    /// Drops the cached labels, the per-glyph ink metrics, the interned fonts,
    /// the atlas contents and the per-generation log memos, then bumps the
    /// generation.
    ///
    /// The first four have to go together: font indices are what atlas keys
    /// are made of, so releasing the intern table while packed glyphs still
    /// reference its indices is exactly the aliasing the interning exists to
    /// prevent. The memos join them because a new generation is a new font
    /// set, and a reason that no longer applies must be able to stop being
    /// said — and one that newly applies must be able to start.
    fn invalidate(&mut self) {
        self.cache.clear();
        self.ink.clear();
        self.fonts.clear();
        self.font_ids.clear();
        self.raster_keys.clear();
        self.atlas.clear();
        self.vertical_refusals_logged.clear();
        self.dropped_glyphs_logged = false;
        self.oversized_glyph_logged = false;
        self.generation = self.generation.saturating_add(1);
    }
}

/// The key `oxitext-raster`'s thread-local font cache computes for these
/// bytes: FNV-1a over at most the first [`RASTER_KEY_BYTES`] of them.
///
/// Mirrors `oxitext-raster-0.2.3/src/tl_cache.rs` exactly, offset basis and
/// prime included — an approximation would report collisions that are not
/// there and miss the ones that are.
fn raster_cache_key(font_data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for byte in font_data.iter().take(RASTER_KEY_BYTES) {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// The text style every label is shaped with: one line, left aligned, no wrap.
///
/// # Errors
///
/// Returns [`RenderError::Text`] if `size_px` is not finite or falls outside
/// `0.0 < size_px <=` [`MAX_LABEL_SIZE_PX`].
fn label_style(size_px: f32) -> Result<TextStyle, RenderError> {
    if !size_px.is_finite() || size_px <= 0.0 || size_px > MAX_LABEL_SIZE_PX {
        return Err(RenderError::Text(format!(
            "label size {size_px} px is outside 0 < size <= {MAX_LABEL_SIZE_PX}"
        )));
    }
    Ok(TextStyle::default()
        .with_font_size(size_px)
        // 0 = no wrapping. `TextStyle::default()` is 800 px, which would wrap a
        // long label behind the caller's back.
        .with_max_width(0.0)
        .with_alignment(TextAlignment::Left))
}

/// Wraps an `oxitext` failure into the crate's error type.
fn text_error(error: OxiTextError) -> RenderError {
    RenderError::Text(error.to_string())
}

/// Copies a chain of shared face bytes into the owned form `oxitext`'s
/// `set_fallback_fonts` takes.
fn owned(chain: &[Arc<[u8]>]) -> Vec<Vec<u8>> {
    chain.iter().map(|face| face.to_vec()).collect()
}

#[cfg(test)]
mod tests;
