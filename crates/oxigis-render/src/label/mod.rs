//! Labels: text shaping, a glyph atlas and the GPU pass that draws them
//! (blueprint §5.3, part A).
//!
//! # Layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`engine`] | [`LabelEngine`]: `oxitext` shaping + `fontdue` rasterisation → [`ShapedLabel`], with a self-sizing LRU keyed by `(text, size, weight, orientation)` |
//! | [`atlas`] | [`GlyphAtlas`]: CPU-side R8 shelf packer, dirty tracking, growth, per-glyph eviction |
//! | [`pipeline`] | [`LabelPipeline`]: screen-space glyph quads, halo copies, one indexed draw |
//! | [`place`] | [`LabelPlacer`]: anchors from decoded MVT features, greedy collision avoidance |
//! | [`vertical`] | Vertical labels: the run itemiser, the shared script tag and the stacking refusal ladder |
//! | [`vertical_table`] | UAX #50 `Vertical_Orientation` — the ONE table the map and the PDF exporter share |
//!
//! # One label, end to end
//!
//! ```no_run
//! use oxigis_render::{
//!     RenderError,
//!     label::{LabelEngine, LabelHalo, LabelPipeline, PlacedLabel},
//! };
//!
//! # fn demo(
//! #     device: &wgpu::Device,
//! #     queue: &wgpu::Queue,
//! #     format: wgpu::TextureFormat,
//! #     font: Vec<u8>,
//! # ) -> Result<(), RenderError> {
//! // The shell supplies font bytes; this crate never reads a file.
//! let mut engine = LabelEngine::new(font)?;
//! let mut pipeline = LabelPipeline::new(device, format)?;
//!
//! // 1. Shape at the *final* on-screen size (a cache hit after the first frame).
//! let tokyo = engine.shape("東京", 14.0)?;
//!
//! // 2. Place (see [`place`] for the tile-driven pass; here, a fixed point).
//! let placed = [PlacedLabel {
//!     shaped: &tokyo,
//!     origin_px: [120.0, 240.0],
//!     color: [20, 20, 20, 255],
//!     halo: Some(LabelHalo::new([255, 255, 255, 255], 1.0)),
//! }];
//! pipeline.upload_atlas(device, queue, engine.atlas_mut())?;
//! pipeline.upload_labels(device, queue, &placed, engine.atlas(), [1280.0, 720.0])?;
//! # Ok(())
//! # }
//! ```
//!
//! # Conventions, in one place
//!
//! * **Colour** — straight (non-premultiplied) sRGB bytes on the vertices,
//!   [`wgpu::BlendState::ALPHA_BLENDING`], sRGB→linear in the shader when the
//!   colour target is an sRGB format: the same three rules the vector pass
//!   follows, so a label and a fill of the same colour match.
//! * **Pixels, not tiles** — labels live in the pass viewport's pixel space
//!   with `y` down, unlike vector meshes which live in the unit tile square.
//!   The placement pass (part B) works in the same space.
//! * **Integer positions** — glyph offsets and label origins are rounded so
//!   atlas texels map 1:1 onto screen pixels. There is no subpixel phase in the
//!   rasteriser, so a fractional position would only blur.
//! * **Final-size rasterisation** — a label is rasterised at the size it is
//!   drawn at. Zoom does not scale the quads; the caller re-requests the label
//!   at the new size and the old entries age out of the LRU.
//! * **Order** — halos for every label, then fills for every label, then done:
//!   a painter's algorithm with no depth test, drawn after the vector meshes.
//! * **A full atlas is routine** — the engine frees the glyphs no live label
//!   points at and packs again, which moves nothing and leaves every
//!   outstanding [`ShapedLabel`] valid. Only an atlas whose every glyph is
//!   still being drawn is rebuilt, and only that bumps the generation a
//!   caller has to watch. See [`engine`] for the whole ladder.
//!
//! # Placement
//!
//! [`place`] turns decoded MVT tiles into [`PlacedLabel`]s: one anchor per
//! feature, a greedy collision pass over the frame's tiles, and a
//! [`LabelSpec`] per layer supplied by a [`LabelResolver`] the shell
//! implements. Its module docs carry the frame call order and the tile-seam
//! deduplication rule.
//!
//! # Vertical labels
//!
//! Vertical **point** labels are in scope since print/text v1.5: a
//! [`LabelSpec`] may ask for [`LabelOrientation::Vertical`] and the engine
//! stacks one upright cell per character, refusing to horizontal for anything
//! the ladder in [`vertical`] cannot set. Placement, collision and halo
//! needed no change at all — a vertical label is a [`ShapedLabel`] whose box
//! is tall, and all three read that box and nothing else.
//!
//! # Out of scope
//!
//! The style→label mapping (evaluating a style's `text-field` into a
//! [`LabelSpec`]) belongs to the shell. SDF glyphs, curved labels, leader
//! lines and label density scoring are Phase 2+ — `TODO.md` §5.3 says in as
//! many words not to gold-plate this. Vertical **line and polygon** anchors,
//! per-column wrapping, tate-chu-yoko, vertical RTL and SCREEN-side rotated
//! runs (the UAX #50 `R` fallback) are out too: the last one is a pure
//! vertex-buffer change — a swapped quad extent plus a one-corner UV
//! rotation in [`pipeline`] — recorded so it is not re-derived, but not v1.5.

pub mod atlas;
pub mod engine;
pub mod pipeline;
pub mod place;
pub mod vertical;
pub mod vertical_table;

pub use crate::label::atlas::{
    AtlasRect, DEFAULT_ATLAS_SIZE, GLYPH_PADDING_PX, GlyphAtlas, GlyphKey, MAX_ATLAS_SIZE,
};
pub use crate::label::engine::{
    DEFAULT_LABEL_CACHE, LabelEngine, LabelGlyph, LabelOrientation, LabelWeight, MAX_LABEL_CACHE,
    MAX_LABEL_SIZE_PX, ShapedLabel,
};
pub use crate::label::pipeline::{
    HALO_OFFSETS, LABEL_VERTEX_SIZE, LabelHalo, LabelPipeline, LabelVertex, PlacedLabel,
    build_label_quads,
};
pub use crate::label::place::{
    AnchorKind, LABEL_PADDING_PX, LabelAnchor, LabelBox, LabelPlacer, LabelResolver, LabelSpec,
    LabelTable, OwnedPlacedLabel, feature_anchor, label_text, placed_labels,
};
pub use crate::label::vertical::{
    VerticalCell, VerticalPlan, VerticalRefusal, VerticalRun, face_for, has_rtl, plan_vertical,
    vertical_runs, vertical_script,
};
pub use crate::label::vertical_table::{
    VERTICAL_ORIENTATION_UNICODE_VERSION, VerticalOrientation, vertical_orientation_of,
};
