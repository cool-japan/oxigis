//! The label half of a vector frame: shaping, placement and the GPU upload
//! that feeds [`oxigis_render::LabelPipeline`] (blueprint §5.3, part C).
//!
//! [`crate::map_gpu`] owns the frame; this module owns everything about it that
//! is specific to text, so `map_gpu.rs` gains one field and two call sites
//! rather than a second renderer's worth of code.
//!
//! # Where the fonts come from
//!
//! `oxigis-render` never reads a file — [`oxigis_render::LabelEngine`] takes raw
//! font bytes and nothing else — and neither does this crate. The **shell**
//! supplies them: `oxigis-desktop` hands over the bundled Noto Sans plus
//! whatever CJK face it finds on the system, `oxigis-web` hands over the same
//! bundled Noto Sans and, optionally, a CJK face fetched at runtime. Both reach
//! [`LabelFrame`] through [`crate::map_gpu::set_label_fonts`] and
//! [`crate::map_gpu::add_label_fallback_font`].
//!
//! Until a shell does that, `MapGpuState` holds no [`LabelFrame`] at all and the
//! label pass does not run — which is also what keeps the raster-only map
//! byte-for-byte unchanged.
//!
//! # The pass, in order
//!
//! ```text
//! prepare:  vectors.begin_frame(view) -> placements
//!           LabelFrame::run(device, queue, placements, source, view)
//!             place every placed tile the source has decoded
//!             is_stale()?  -> throw the placer away and do it again, once
//!             upload_atlas -> upload_labels
//! paint:    vectors.paint(...)          (meshes first)
//!           LabelFrame::draw(pass, ...) (labels on top)
//! ```
//!
//! The staleness re-run is [`oxigis_render::LabelPlacer`]'s documented frame
//! contract, not a heuristic: shaping a label can overflow the glyph atlas,
//! which clears and repacks it, and every [`oxigis_render::ShapedLabel`] handed
//! out before that points at texels which have since moved. The placer notices
//! (its recorded generation stops matching) and the only correct response is to
//! discard the whole pass and re-place against the now-roomy atlas. One re-run
//! is enough in every case but a label set larger than the atlas itself, which
//! is why the second pass is accepted unconditionally — with a log line if it
//! was stale too.
//!
//! Labels are re-placed from scratch every frame. That is deliberate: panning
//! moves every anchor, so a cached placement would be wrong immediately, and the
//! expensive half — shaping and rasterising — is cached inside the engine and
//! keyed by `(text, size)`, so a pan costs collision tests and a vertex rebuild,
//! not glyph work.
//!
//! # Sizes are zoom-independent
//!
//! A [`oxigis_render::LabelSpec`]'s `size_px` is a screen size, so zooming
//! changes which features are labelled (the tiles change) but never how big the
//! text is. Zoom-dependent text sizing is a style-model feature, and Phase 0's
//! [`oxigis_core::SymbolStyle`] has no stops to interpolate.

use eframe::wgpu;
use oxigis_render::{
    LabelEngine, LabelPipeline, LabelPlacer, MapView, OwnedPlacedLabel, RenderError, ScissorRect,
    TilePlacement, placed_labels,
};

use crate::local_layers::LocalLabelJob;
use crate::vector_provider::VectorTileSource;

/// One tiled vector layer's contribution to a label pass: the frame's
/// placements and the source the decoded tiles come from.
///
/// [`LabelFrame::run`] takes a **slice** of these, not one, because the drawn
/// stack can hold several vector-tile layers at once (compositing v1.6) and
/// all of them must collide against ONE [`LabelPlacer`] — a cadastral layer
/// under a POI layer that each placed labels against its own grid would
/// overprint the other, which is precisely the defect the single shared placer
/// exists to prevent. An empty slice is the normal state of a map showing only
/// local (dropped) datasets.
///
/// Each entry carries its own [`oxigis_render::LabelTable`]: the symbol rules are the
/// *source's*, so two layers styled by different rule sets keep them.
pub struct TiledLabelInput<'a> {
    /// The tiles this layer drew this frame, and where.
    pub placements: &'a [TilePlacement],
    /// Where its decoded tiles and its symbol rules come from.
    pub source: &'a dyn VectorTileSource,
}

/// Everything the label pass owns across frames: the shaping engine (with its
/// glyph atlas and label cache) and the GPU pipeline that draws the quads.
///
/// Constructed only once a shell has supplied font bytes; see the
/// [module docs][self].
pub struct LabelFrame {
    /// Shaping, rasterisation and the CPU-side glyph atlas.
    engine: LabelEngine,
    /// The screen-space quad pipeline, built on the first frame that has a
    /// device — [`LabelFrame::new`] deliberately takes none, so fonts can be
    /// installed before the map is attached.
    pipeline: Option<LabelPipeline>,
    /// Labels accepted by the most recent pass, for status reporting.
    placed: usize,
    /// Set once a pass has failed, so the log records it one time instead of
    /// once per frame.
    failed: bool,
}

impl core::fmt::Debug for LabelFrame {
    /// Neither [`LabelEngine`] nor [`LabelPipeline`] is [`Debug`] (both wrap
    /// `oxitext`/`wgpu` handles), so this reports the state that matters.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LabelFrame")
            .field("fallback_fonts", &self.engine.fallback_count())
            .field("generation", &self.engine.generation())
            .field("atlas_glyphs", &self.engine.atlas().len())
            .field("has_pipeline", &self.pipeline.is_some())
            .field("placed", &self.placed)
            .field("failed", &self.failed)
            .finish()
    }
}

impl LabelFrame {
    /// Builds the label pass over `primary_font` (raw TTF/OTF bytes), with
    /// `fallbacks` tried in order for clusters the primary font cannot map.
    ///
    /// No GPU resource is created here: the pipeline is built lazily on the
    /// first [`LabelFrame::run`], so a shell may install fonts before (or
    /// without) attaching the map.
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Text`] if `primary_font` is not a font
    /// `oxitext` can parse.
    pub fn new(primary_font: Vec<u8>, fallbacks: Vec<Vec<u8>>) -> Result<Self, RenderError> {
        let mut engine = LabelEngine::new(primary_font)?;
        if !fallbacks.is_empty() {
            engine.set_fallback_fonts(fallbacks);
        }
        Ok(Self {
            engine,
            pipeline: None,
            placed: 0,
            failed: false,
        })
    }

    /// Appends one font to the fallback chain — the runtime path a browser shell
    /// takes when an asynchronously fetched CJK face arrives.
    ///
    /// Invalidates every shaped label and the glyph atlas, so the next frame
    /// re-shapes from scratch. Cheap enough to be unremarkable once, ruinous
    /// every frame: call it when bytes actually arrive, not speculatively.
    pub fn add_fallback_font(&mut self, font: Vec<u8>) {
        self.engine.add_fallback_font(font);
    }

    /// Appends several fonts to the fallback chain in one invalidation — what
    /// the native shell uses when more than one chain entry arrives within a
    /// single frame. Empty input is a no-op.
    pub fn add_fallback_fonts(&mut self, fonts: Vec<Vec<u8>>) {
        self.engine.add_fallback_fonts(fonts);
    }

    /// Number of fonts in the fallback chain.
    #[must_use]
    pub fn fallback_count(&self) -> usize {
        self.engine.fallback_count()
    }

    /// Replaces the BOLD face chain a `Bold`-weighted symbol style draws
    /// through (print/text v1.4, D-W3/D-W4). Empty removes bold support;
    /// bold labels then draw Regular with one log — never a synthetic
    /// emboldening. Invalidates every shaped label, exactly as a fallback
    /// change does, so it belongs with font installation rather than in a
    /// frame.
    pub fn set_bold_fonts(&mut self, fonts: Vec<Vec<u8>>) {
        self.engine.set_bold_fonts(fonts);
    }

    /// Number of faces in the bold chain, before its regular tail.
    #[must_use]
    pub fn bold_face_count(&self) -> usize {
        self.engine.bold_face_count()
    }

    /// Labels accepted by the most recent pass — what a status bar would show.
    #[must_use]
    pub fn placed(&self) -> usize {
        self.placed
    }

    /// Runs the whole label pass for one frame.
    ///
    /// `tiled`'s placements come straight from
    /// [`oxigis_render::VectorLayerRenderer::begin_frame`] and its source
    /// supplies the decoded tiles and the [`oxigis_render::LabelTable`]; `locals` carries the
    /// same three things for every on-screen local dataset (see
    /// [`crate::local_layers`]); `view` fixes the pixel space every label lives
    /// in. Tiles the source has not decoded are skipped, so a tile whose mesh is
    /// still in flight simply carries no labels yet.
    ///
    /// Every half feeds **one** [`LabelPlacer`], which is the point: a label
    /// from a dropped GeoJSON collides against a basemap place name instead of
    /// overprinting it, and so does a label from the second vector-tile layer
    /// of the stack against the first.
    ///
    /// `tiled` is expected **top-most first**: the placer is greedy and
    /// first-come-first-served, so the layer the user sees on top is the one
    /// whose label survives a collision — the same layer that was the only one
    /// to label at all before the stack existed.
    ///
    /// Errors are logged once and swallowed: a frame that cannot label is a
    /// frame without text, not a dead map.
    pub fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        tiled: &[TiledLabelInput<'_>],
        locals: &[LocalLabelJob<'_>],
        view: MapView,
    ) {
        match self.try_run(device, queue, format, tiled, locals, view) {
            Ok(placed) => {
                self.placed = placed;
                self.failed = false;
            }
            Err(error) => {
                self.placed = 0;
                if !self.failed {
                    self.failed = true;
                    tracing::error!(%error, "oxigis-ui: label pass failed");
                }
            }
        }
    }

    /// The fallible body of [`LabelFrame::run`], returning the label count.
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Text`] from shaping and [`RenderError::Gpu`] /
    /// [`RenderError::InvalidViewport`] from the two uploads.
    fn try_run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        tiled: &[TiledLabelInput<'_>],
        locals: &[LocalLabelJob<'_>],
        view: MapView,
    ) -> Result<usize, RenderError> {
        // Nothing to do at all, and — importantly — no pipeline to build: a
        // vector style without a single symbol rule costs the label pass one
        // `is_empty` per source per frame. `locals` only ever holds jobs that
        // *do* label (see `LocalVectorRenderer::label_jobs`), so a non-empty
        // slice always means work.
        let any_tiled_labels = tiled
            .iter()
            .any(|entry| !entry.source.label_table().is_empty());
        if !any_tiled_labels && locals.is_empty() {
            if let Some(pipeline) = self.pipeline.as_mut() {
                pipeline.upload_labels(device, queue, &[], self.engine.atlas(), view.size_px())?;
            }
            return Ok(0);
        }

        let viewport_px = view.size_px();
        // The frame contract: `upload_atlas` may not run between the two passes,
        // because a stale pass's labels are exactly the ones whose texels moved.
        let placed = match self.place(tiled, locals, viewport_px)? {
            Some(placed) => placed,
            None => match self.place(tiled, locals, viewport_px)? {
                Some(placed) => placed,
                None => {
                    // A label set that cannot fit the atlas even after a repack.
                    // The second pass's placements are drawn anyway; some glyphs
                    // may show the wrong ink for one frame.
                    tracing::debug!(
                        "oxigis-ui: the glyph atlas repacked twice in one frame; \
                         labels may be briefly wrong",
                    );
                    Vec::new()
                }
            },
        };

        let pipeline = match self.pipeline.as_mut() {
            Some(pipeline) => pipeline,
            None => self.pipeline.insert(LabelPipeline::new(device, format)?),
        };
        pipeline.upload_atlas(device, queue, self.engine.atlas_mut())?;
        let borrowed = placed_labels(&placed);
        pipeline.upload_labels(device, queue, &borrowed, self.engine.atlas(), viewport_px)?;
        Ok(placed.len())
    }

    /// One placement pass over every visible tile and local layer.
    ///
    /// Returns [`None`] — meaning "the atlas repacked underneath us, run me
    /// again" — instead of the labels, so the staleness check provably happens
    /// before [`LabelPlacer::finish`] consumes the placer.
    ///
    /// Tiled tiles are placed first and local layers after, so a basemap label
    /// wins a collision against a dropped dataset's label at the same spot; the
    /// placer is greedy and first-come-first-served. Within the tiled half the
    /// slice's own order decides, which is why `run` documents it as top-most
    /// first.
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Text`] from
    /// [`LabelPlacer::place_tile`].
    fn place(
        &mut self,
        tiled: &[TiledLabelInput<'_>],
        locals: &[LocalLabelJob<'_>],
        viewport_px: [f32; 2],
    ) -> Result<Option<Vec<OwnedPlacedLabel>>, RenderError> {
        let mut placer = LabelPlacer::new(viewport_px);
        for entry in tiled {
            // Each layer's OWN symbol rules: two vector sources styled
            // differently must not be read through one another's table.
            let labels = entry.source.label_table();
            if labels.is_empty() {
                continue;
            }
            for placement in entry.placements {
                let Some(tile) = entry.source.decoded(placement.tile) else {
                    continue;
                };
                placer.place_tile(&mut self.engine, &tile, placement, labels)?;
            }
        }
        for job in locals {
            placer.place_tile(&mut self.engine, job.tile, &job.placement, job.labels)?;
        }
        if placer.is_stale() {
            return Ok(None);
        }
        Ok(Some(placer.finish()))
    }

    /// Draws the frame's labels, on top of whatever the pass already holds.
    ///
    /// `clip_origin_px` is the pass viewport's top-left corner in the
    /// framebuffer and `view` the camera [`LabelFrame::run`] placed against —
    /// together they are the scissor rectangle, because a placed label is in
    /// *viewport* pixels while `wgpu`'s scissor is in *framebuffer* pixels.
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Gpu`] from
    /// [`oxigis_render::LabelPipeline::draw`].
    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        clip_origin_px: [f32; 2],
        view: MapView,
    ) -> Result<(), RenderError> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(());
        };
        pipeline.draw(render_pass, clip_scissor(clip_origin_px, view.size_px()))
    }
}

/// The pass viewport as a scissor rectangle, in framebuffer pixels.
///
/// Deliberately identical to `VectorLayerRenderer`'s own clip rectangle: the
/// labels of a frame cover exactly the area its meshes do, and placement has
/// already rejected anything not fully inside it. Returns [`None`] for a
/// degenerate viewport, which leaves the pass's existing scissor alone.
#[must_use]
pub(crate) fn clip_scissor(clip_origin_px: [f32; 2], size_px: [f32; 2]) -> Option<ScissorRect> {
    // Checked *before* the clamp: `f32::NAN.max(0.0)` is `0.0`, so clamping
    // first would launder a NaN origin into a plausible-looking rectangle.
    if clip_origin_px
        .iter()
        .chain(size_px.iter())
        .any(|value| !value.is_finite())
    {
        return None;
    }
    let origin_x = clip_origin_px[0].max(0.0);
    let origin_y = clip_origin_px[1].max(0.0);
    ScissorRect::new(
        origin_x as u32,
        origin_y as u32,
        size_px[0].ceil().max(0.0) as u32,
        size_px[1].ceil().max(0.0) as u32,
    )
}

#[cfg(test)]
mod tests {
    // Everything that touches a `wgpu::Device` (`run`, `draw`) is exercised by
    // the shells; what is testable headlessly is the font plumbing and the
    // scissor arithmetic.
    use super::{LabelFrame, clip_scissor};

    fn noto() -> Vec<u8> {
        oxifont_bundled::NOTO_SANS_REGULAR.to_vec()
    }

    #[test]
    fn a_frame_starts_with_no_fallbacks_and_no_pipeline() {
        let frame = LabelFrame::new(noto(), Vec::new()).expect("bundled Noto Sans parses");
        assert_eq!(frame.fallback_count(), 0);
        assert_eq!(frame.placed(), 0);
        assert!(format!("{frame:?}").contains("has_pipeline: false"));
    }

    #[test]
    fn fallbacks_can_be_supplied_up_front_or_appended_later() {
        let mut frame =
            LabelFrame::new(noto(), vec![noto()]).expect("bundled Noto Sans parses twice");
        assert_eq!(frame.fallback_count(), 1);
        frame.add_fallback_font(noto());
        assert_eq!(frame.fallback_count(), 2);
    }

    #[test]
    fn bytes_that_are_not_a_font_are_rejected_rather_than_panicking() {
        assert!(LabelFrame::new(vec![0xDE, 0xAD, 0xBE, 0xEF], Vec::new()).is_err());
    }

    #[test]
    fn the_scissor_covers_the_pass_viewport_at_its_framebuffer_origin() {
        let rect = clip_scissor([12.0, 48.0], [800.0, 600.5]).expect("a non-empty rectangle");
        assert_eq!((rect.x, rect.y), (12, 48));
        // Rounded *up*, so a fractional viewport never clips its own last row.
        assert_eq!((rect.width, rect.height), (800, 601));
    }

    #[test]
    fn a_degenerate_viewport_leaves_the_scissor_alone() {
        assert!(clip_scissor([0.0, 0.0], [0.0, 600.0]).is_none());
        assert!(clip_scissor([f32::NAN, 0.0], [800.0, 600.0]).is_none());
        // A negative origin is clamped rather than wrapping into a huge `u32`.
        let rect = clip_scissor([-5.0, -5.0], [10.0, 10.0]).expect("a rectangle");
        assert_eq!((rect.x, rect.y), (0, 0));
    }
}
