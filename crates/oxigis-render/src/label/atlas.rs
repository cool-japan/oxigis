//! CPU-side R8 glyph atlas: a shelf packer plus the pixels the GPU uploads.
//!
//! # Why the atlas lives on the CPU
//!
//! `oxigis-render` owns no GPU context (see the crate docs), so the atlas is a
//! plain `Vec<u8>` of 8-bit coverage with a dirty flag. Whoever holds a
//! [`wgpu::Queue`] — [`crate::label::LabelPipeline`] in practice — copies it
//! into a texture. That also keeps the packer unit-testable without a device.
//!
//! The flag comes with the row band that changed
//! ([`GlyphAtlas::dirty_rows`]), so that copy is a few kilobytes for the usual
//! "one new glyph this frame" rather than the whole 1–16 MiB buffer.
//!
//! # Packing
//!
//! A hand-written shelf packer, ~100 lines, no external dependency: glyphs are
//! appended left-to-right onto horizontal shelves, a new shelf opening below
//! the last one when the current row is full. Shelf packing wastes the vertical
//! slack of a row whose tallest glyph is much taller than the rest, which for a
//! *map label* workload is close to free: every glyph on a shelf comes from a
//! handful of font sizes, so heights within a row barely vary.
//!
//! Each glyph is padded by [`GLYPH_PADDING_PX`] on its right and bottom, so a
//! sampler never picks up a neighbour's coverage and, more importantly, so the
//! halo pass (which offsets quads by whole pixels) cannot smear one glyph into
//! another.
//!
//! # Growth and overflow
//!
//! The atlas starts at [`DEFAULT_ATLAS_SIZE`] and doubles — copying the old
//! rows into the top-left of the new buffer, which leaves **every previously
//! returned [`AtlasRect`] valid** — until [`MAX_ATLAS_SIZE`]. Once a glyph will
//! not fit into the maximum size, [`GlyphAtlas::try_insert`] returns `Ok(None)`
//! rather than an error: "full" is a routine condition with a routine answer.
//!
//! # Eviction
//!
//! The routine answer is **per glyph**, not wholesale.
//! [`GlyphAtlas::retain`] frees the slots of the glyphs a caller no longer
//! counts as live, then recomputes the free list from the slots that
//! survived — the complement of what each shelf still holds, so a run of cold
//! glyphs comes back as ONE region a wide glyph can use rather than as the
//! slivers incremental merging would leave. Eviction **does not move a
//! surviving glyph and does not bump the generation**, which is what lets
//! [`crate::label::LabelEngine`] recover from a full atlas without
//! invalidating the labels a frame is in the middle of drawing.
//!
//! [`GlyphAtlas::clear`] — evict everything, bump the generation, rebuild from
//! the labels the next frame asks for — stays as the last resort, for the
//! atlas that is full of glyphs which are all still live.
//!
//! Because R8 rows are one byte per texel and `wgpu` uploads want 256-byte row
//! alignment, only sizes that are a multiple of 256 reach the GPU — which every
//! size reachable from [`DEFAULT_ATLAS_SIZE`] is. See [`GlyphAtlas::with_size`].
//!
//! Rects are *texel* rectangles, not UVs, precisely because the atlas can grow:
//! UVs are derived at vertex-build time from [`GlyphAtlas::size`], so a resize
//! costs a re-upload and nothing else.

use std::collections::HashMap;

use crate::error::RenderError;

/// Side length, in texels, a fresh atlas is created with.
///
/// 1024² = 1 MiB of R8 — enough for several thousand Latin glyphs at map label
/// sizes, so the common case never resizes.
pub const DEFAULT_ATLAS_SIZE: u32 = 1024;

/// Largest side length the atlas will grow to before reporting "full".
///
/// 4096 is the smallest maximum texture dimension guaranteed by WebGL2 / the
/// `downlevel_webgl2_defaults()` wgpu limits, which is the floor the web shell
/// targets.
pub const MAX_ATLAS_SIZE: u32 = 4096;

/// Transparent gutter kept to the right of and below every packed glyph.
pub const GLYPH_PADDING_PX: u32 = 1;

/// Most freed regions the packer tracks at once.
///
/// Fragmentation is bounded rather than trusted: past this many regions the
/// smallest are dropped from the list, which keeps the free list — and the
/// best-fit scan over it — a fixed cost. Nothing is lost for good: the list is
/// recomputed from the packed slots on the next eviction, so a dropped region
/// comes back as soon as anything around it does.
const MAX_FREE_REGIONS: usize = 1024;

/// Identity of one rasterised glyph: which font, which glyph, which size.
///
/// `font` is an index into [`crate::label::LabelEngine`]'s font table rather
/// than a pointer: `oxitext` hands out `Arc<[u8]>` font handles whose addresses
/// are reused after a fallback chain is replaced, and an aliased address would
/// silently map one font's glyph onto another's atlas slot.
///
/// `size_bits` is `f32::to_bits` of the pixels-per-em the glyph was rasterised
/// at, so two requests for the same nominal size are one atlas entry while a
/// zoom-driven size change is a different one (labels are rasterised at their
/// final pixel size — see [`crate::label::pipeline`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// Index of the font in the engine's font table.
    pub font: u16,
    /// Glyph id inside that font.
    pub gid: u16,
    /// `f32::to_bits` of the rasterisation size in pixels-per-em.
    pub size_bits: u32,
}

impl GlyphKey {
    /// Builds a key for `gid` in font `font` rasterised at `size_px`.
    #[must_use]
    pub fn new(font: u16, gid: u16, size_px: f32) -> Self {
        Self {
            font,
            gid,
            size_bits: size_px.to_bits(),
        }
    }

    /// The rasterisation size this key was built with.
    #[must_use]
    pub fn size_px(&self) -> f32 {
        f32::from_bits(self.size_bits)
    }
}

/// A packed glyph's rectangle inside the atlas, in texels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtlasRect {
    /// Left edge, from the atlas's left.
    pub x: u32,
    /// Top edge, from the atlas's top.
    pub y: u32,
    /// Width in texels; never zero (empty glyphs are not packed).
    pub width: u32,
    /// Height in texels; never zero.
    pub height: u32,
}

impl AtlasRect {
    /// Whether this rectangle shares any texel with `other`.
    ///
    /// Used by the packing tests; also handy for a debug overlay.
    #[must_use]
    pub const fn intersects(&self, other: &Self) -> bool {
        self.x < other.x + other.width
            && other.x < self.x + self.width
            && self.y < other.y + other.height
            && other.y < self.y + self.height
    }

    /// The rectangle's UV bounds — `[u_min, v_min, u_max, v_max]` — inside an
    /// atlas of `atlas_size` texels a side.
    ///
    /// Returns zeros when `atlas_size` is zero rather than dividing by it.
    #[must_use]
    pub fn uv(&self, atlas_size: u32) -> [f32; 4] {
        if atlas_size == 0 {
            return [0.0; 4];
        }
        let scale = 1.0 / atlas_size as f32;
        [
            self.x as f32 * scale,
            self.y as f32 * scale,
            (self.x + self.width) as f32 * scale,
            (self.y + self.height) as f32 * scale,
        ]
    }
}

/// One horizontal row of the shelf packer.
#[derive(Debug, Clone, Copy)]
struct Shelf {
    /// Top edge of the shelf.
    y: u32,
    /// Height reserved by the shelf's first (tallest-so-far) glyph.
    height: u32,
    /// Left edge available for the next glyph.
    cursor_x: u32,
}

/// A square, single-channel (R8) glyph atlas with shelf packing.
///
/// The pixel buffer is `size * size` bytes of coverage, `0` transparent and
/// `255` opaque — exactly what a `Bitmap` from `oxitext` carries, so packing is
/// a row-wise `copy_from_slice`.
#[derive(Debug)]
pub struct GlyphAtlas {
    size: u32,
    max_size: u32,
    pixels: Vec<u8>,
    shelves: Vec<Shelf>,
    slots: HashMap<GlyphKey, AtlasRect>,
    /// Packable regions no slot occupies, in texels, recomputed by
    /// [`Self::rebuild_free`] after every eviction.
    ///
    /// A region is only ever carved out of a shelf's unoccupied span, so it
    /// cannot overlap a live slot however many times it is split.
    free: Vec<AtlasRect>,
    dirty: bool,
    /// The half-open row band `[top, bottom)` the pixels changed in since the
    /// last [`Self::mark_clean`], or `None` when nothing changed.
    ///
    /// A *band*, not a count and not a rectangle: the uploader copies whole
    /// rows (a partial row would break `wgpu`'s 256-byte row alignment), and
    /// one span is what a shelf packer's touches almost always collapse to.
    /// Kept in lockstep with `dirty`: [`Self::mark_dirty`] and
    /// [`Self::mark_clean`] are the only writers of either field, and each
    /// writes both.
    dirty_rows: Option<(u32, u32)>,
    generation: u32,
}

impl GlyphAtlas {
    /// Creates an empty atlas of [`DEFAULT_ATLAS_SIZE`] texels a side.
    ///
    /// # Errors
    ///
    /// Cannot fail with the default constants; see
    /// [`GlyphAtlas::with_size`] for the conditions it checks.
    pub fn new() -> Result<Self, RenderError> {
        Self::with_size(DEFAULT_ATLAS_SIZE, MAX_ATLAS_SIZE)
    }

    /// Creates an empty atlas of `initial_size` texels a side, allowed to grow
    /// (by doubling) up to `max_size`.
    ///
    /// **Only sizes that are a multiple of 256 can be uploaded to the GPU**:
    /// one R8 row is one byte per texel and `wgpu` wants a 256-byte row
    /// alignment, so [`crate::label::LabelPipeline::upload_atlas`] rejects
    /// anything else. [`GlyphAtlas::new`] and every doubling from it satisfy
    /// that; smaller sizes are accepted here on purpose, because the packer's
    /// own tests want a 64² atlas they can fill in a few inserts and no device
    /// is involved in that.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Text`] if either size is zero, if `initial_size`
    /// exceeds `max_size`, or if `max_size` is beyond what a `usize` can
    /// address as a square buffer.
    pub fn with_size(initial_size: u32, max_size: u32) -> Result<Self, RenderError> {
        if initial_size == 0 || max_size == 0 {
            return Err(RenderError::Text(
                "glyph atlas size must be at least one texel".to_owned(),
            ));
        }
        if initial_size > max_size {
            return Err(RenderError::Text(format!(
                "glyph atlas initial size {initial_size} exceeds its maximum {max_size}"
            )));
        }
        let area = usize::try_from(initial_size)
            .ok()
            .and_then(|side| side.checked_mul(side))
            .ok_or_else(|| {
                RenderError::Text(format!(
                    "glyph atlas of {initial_size}² texels is not addressable"
                ))
            })?;
        Ok(Self {
            size: initial_size,
            max_size,
            pixels: vec![0u8; area],
            shelves: Vec::new(),
            slots: HashMap::new(),
            free: Vec::new(),
            dirty: false,
            dirty_rows: None,
            generation: 0,
        })
    }

    /// Current side length in texels.
    #[must_use]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Largest side length this atlas will grow to.
    #[must_use]
    pub fn max_size(&self) -> u32 {
        self.max_size
    }

    /// The coverage buffer, `size() * size()` bytes, row-major.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Whether the pixels changed since the last [`GlyphAtlas::mark_clean`].
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// The half-open row band `[top, bottom)` the pixels changed in since the
    /// last [`GlyphAtlas::mark_clean`], or `None` when they did not.
    ///
    /// `Some` exactly when [`GlyphAtlas::is_dirty`] is `true`, by
    /// construction: this field and the flag are written only by
    /// [`GlyphAtlas::mark_clean`] and `mark_dirty`, and each writes them
    /// together. An uploader may therefore copy
    /// `bottom - top` rows instead of `size()` of them — for the usual frame
    /// that packs one 20-texel glyph into a 1024² atlas that is 20 KiB instead
    /// of 1 MiB — as long as it still copies everything after a resize, which
    /// hands the GPU a texture that has never been written.
    #[must_use]
    pub fn dirty_rows(&self) -> Option<(u32, u32)> {
        self.dirty_rows
    }

    /// Records that the current pixels have reached the GPU.
    pub fn mark_clean(&mut self) {
        self.dirty = false;
        self.dirty_rows = None;
    }

    /// Records that rows `[top, bottom)` changed. **The only writer of
    /// `dirty`/`dirty_rows`**, so the two cannot drift apart.
    ///
    /// Widens any band already pending rather than replacing it: several
    /// inserts and evictions land between two uploads, and the union of their
    /// bands is the smallest span of whole rows that covers them all. An empty
    /// or inverted band is ignored, so a zero-height rect cannot mark the
    /// atlas dirty with nothing to send.
    fn mark_dirty(&mut self, top: u32, bottom: u32) {
        if bottom <= top {
            return;
        }
        // Clamped so a caller's arithmetic can never name a row the buffer
        // does not have; `write_texture` would reject the copy outright.
        let bottom = bottom.min(self.size);
        if bottom <= top {
            return;
        }
        self.dirty = true;
        self.dirty_rows = Some(match self.dirty_rows {
            Some((current_top, current_bottom)) => {
                (current_top.min(top), current_bottom.max(bottom))
            }
            None => (top, bottom),
        });
    }

    /// How many times the atlas has been cleared.
    ///
    /// Every [`AtlasRect`] handed out belongs to a generation; a clear
    /// invalidates all of them at once, which is how
    /// [`crate::label::ShapedLabel`] knows whether its slots are still live.
    /// A *resize* does not bump the generation — growth preserves rects.
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Number of glyphs currently packed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether no glyph is packed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The rectangle `key` occupies, if it is packed.
    #[must_use]
    pub fn get(&self, key: &GlyphKey) -> Option<AtlasRect> {
        self.slots.get(key).copied()
    }

    /// Fraction of the atlas area covered by packed glyphs, `0.0..=1.0`.
    #[must_use]
    pub fn occupancy(&self) -> f32 {
        let area = self.size as f32 * self.size as f32;
        if area <= 0.0 {
            return 0.0;
        }
        let used: f32 = self
            .slots
            .values()
            .map(|rect| rect.width as f32 * rect.height as f32)
            .sum();
        used / area
    }

    /// Number of freed regions the packer is tracking.
    ///
    /// Diagnostic: a number that keeps climbing while [`GlyphAtlas::len`] does
    /// not means the evicted set is fragmenting faster than it is being
    /// reused.
    #[must_use]
    pub fn free_regions(&self) -> usize {
        self.free.len()
    }

    /// Whether a `width * height` glyph could EVER be packed — into this atlas
    /// grown to its maximum and empty.
    ///
    /// The one distinction the engine's overflow ladder rests on: a glyph that
    /// fails this is dropped from its label, because no amount of evicting
    /// will make room for it; a glyph that fails to pack while passing it only
    /// met an atlas that was full at the time.
    #[must_use]
    pub fn fits(&self, width: u32, height: u32) -> bool {
        if width == 0 || height == 0 {
            return false;
        }
        let (Some(slot_width), Some(slot_height)) = (
            width.checked_add(GLYPH_PADDING_PX),
            height.checked_add(GLYPH_PADDING_PX),
        ) else {
            return false;
        };
        slot_width <= self.max_size && slot_height <= self.max_size
    }

    /// Frees `key`'s slot for re-use, returning whether it was packed.
    ///
    /// Surviving glyphs do not move and the generation does not change, so
    /// every [`AtlasRect`] handed out for a glyph that is NOT evicted stays
    /// valid. Evicting a glyph a live [`crate::label::ShapedLabel`] still
    /// points at is therefore the one thing a caller must not do — the slot
    /// can be handed to another glyph and the label would draw its ink.
    pub fn evict(&mut self, key: &GlyphKey) -> bool {
        let Some(rect) = self.slots.remove(key) else {
            return false;
        };
        self.release(rect);
        self.rebuild_free();
        true
    }

    /// Frees every packed glyph `keep` rejects, returning how many went.
    ///
    /// The batch form of [`GlyphAtlas::evict`], and the one worth using: the
    /// free list is recomputed once afterwards, so a shelf whose cold glyphs
    /// all go comes back as one region rather than a dozen slivers.
    pub fn retain(&mut self, mut keep: impl FnMut(&GlyphKey) -> bool) -> usize {
        let mut released: Vec<AtlasRect> = Vec::new();
        self.slots.retain(|key, rect| {
            if keep(key) {
                return true;
            }
            released.push(*rect);
            false
        });
        if released.is_empty() {
            return 0;
        }
        let evicted = released.len();
        for rect in released {
            self.release(rect);
        }
        self.rebuild_free();
        evicted
    }

    /// Drops every packed glyph, zeroes the pixels and bumps the generation.
    ///
    /// The last resort, not the routine answer to a full atlas: it invalidates
    /// every [`AtlasRect`] in existence at once, which costs the caller a
    /// re-shape of everything on screen. Try [`GlyphAtlas::retain`] first.
    ///
    /// The buffer keeps its current size: a workload that needed a 2048² atlas
    /// once will need it again, and re-growing costs a reallocation per frame.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.shelves.clear();
        self.free.clear();
        self.pixels.iter_mut().for_each(|texel| *texel = 0);
        self.generation = self.generation.saturating_add(1);
        self.mark_dirty(0, self.size);
    }

    /// Zeroes the ink one evicted glyph leaves behind.
    ///
    /// Not cosmetic: a smaller glyph packed into part of the freed region
    /// would otherwise leave the evicted glyph's coverage inside its own
    /// gutter, which is exactly the bleed the gutter exists to prevent.
    fn release(&mut self, rect: AtlasRect) {
        let stride = self.size as usize;
        for row in 0..rect.height as usize {
            let start = (rect.y as usize + row) * stride + rect.x as usize;
            let Some(texels) = self.pixels.get_mut(start..start + rect.width as usize) else {
                continue;
            };
            texels.fill(0);
        }
        // The zeroing counts as a change: skipping it would leave the evicted
        // glyph's coverage live on the GPU under whatever gets packed there
        // next, which is the bleed the gutter exists to prevent.
        self.mark_dirty(rect.y, rect.y.saturating_add(rect.height));
    }

    /// Adds a region to the free list, dropping the smallest one when the list
    /// is at [`MAX_FREE_REGIONS`].
    fn push_free(&mut self, region: AtlasRect) {
        if region.width == 0 || region.height == 0 {
            return;
        }
        if self.free.len() >= MAX_FREE_REGIONS {
            let area = |rect: &AtlasRect| u64::from(rect.width) * u64::from(rect.height);
            let Some((index, smallest)) = self
                .free
                .iter()
                .enumerate()
                .map(|(index, rect)| (index, area(rect)))
                .min_by_key(|(_, rect_area)| *rect_area)
            else {
                return;
            };
            if smallest >= area(&region) {
                return;
            }
            self.free.swap_remove(index);
        }
        self.free.push(region);
    }

    /// Recomputes the whole free list from the slots that survived, and resets
    /// the packer outright when none did.
    ///
    /// Rebuilding from ground truth rather than merging regions as they are
    /// freed is what keeps eviction worth doing: a guillotine split leaves
    /// offcuts of mismatched extents that no pairwise merge can put back
    /// together, and a few rounds of that fragment an atlas into slivers that
    /// fit nothing while its occupancy reads near zero.
    ///
    /// Conservative by construction: a shelf's free space is the complement of
    /// the x-intervals its slots reserve, at the shelf's full height. The
    /// slack under a slot shorter than its shelf is left alone — that is the
    /// slack shelf packing has always wasted, and claiming it would mean
    /// tracking which parts of it another slot already sits in.
    fn rebuild_free(&mut self) {
        self.free.clear();
        if self.slots.is_empty() {
            // Nothing is packed at all: the shelf packer expresses an empty
            // atlas better than any list of regions can.
            self.shelves.clear();
            return;
        }
        let mut reserved: Vec<Vec<(u32, u32)>> = vec![Vec::new(); self.shelves.len()];
        for rect in self.slots.values() {
            // Every slot sits inside exactly one shelf band: shelf allocation
            // starts at the band's top, and a freed region never spans two.
            let band = self.shelves.iter().position(|shelf| {
                rect.y >= shelf.y && rect.y < shelf.y.saturating_add(shelf.height)
            });
            let Some(intervals) = band.and_then(|band| reserved.get_mut(band)) else {
                continue;
            };
            intervals.push((
                rect.x,
                rect.x
                    .saturating_add(rect.width)
                    .saturating_add(GLYPH_PADDING_PX),
            ));
        }
        for (band, mut intervals) in reserved.into_iter().enumerate() {
            let Some(shelf) = self.shelves.get(band).copied() else {
                continue;
            };
            intervals.sort_unstable();
            let mut cursor = 0u32;
            for (start, end) in intervals {
                if start > cursor {
                    self.push_free(AtlasRect {
                        x: cursor,
                        y: shelf.y,
                        width: start - cursor,
                        height: shelf.height,
                    });
                }
                cursor = cursor.max(end);
            }
            // Past the shelf's own cursor the shelf packer allocates directly,
            // so the free list stops there.
            if cursor < shelf.cursor_x {
                self.push_free(AtlasRect {
                    x: cursor,
                    y: shelf.y,
                    width: shelf.cursor_x - cursor,
                    height: shelf.height,
                });
            }
        }
    }

    /// Packs one glyph's coverage, growing the atlas if that is what it takes.
    ///
    /// Returns the rectangle the glyph occupies — the existing one when `key`
    /// is already packed, without touching the pixels — or `Ok(None)` when the
    /// glyph does not fit even at [`GlyphAtlas::max_size`]. `Ok(None)` is the
    /// caller's cue to free room ([`GlyphAtlas::retain`], then
    /// [`GlyphAtlas::clear`]) and retry, *not* an error: it happens whenever a
    /// long-lived map session accumulates more glyphs than the atlas holds.
    /// [`GlyphAtlas::fits`] separates the two reasons it can happen.
    ///
    /// Zero-area glyphs (a space, a `.notdef` with no outline) are never
    /// packed; they return `Ok(None)` too, and callers are expected to skip
    /// them before asking — see [`crate::label::LabelEngine`].
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Text`] if `coverage` is not exactly
    /// `width * height` bytes, i.e. if the rasteriser's dimensions and pixels
    /// disagree.
    pub fn try_insert(
        &mut self,
        key: GlyphKey,
        width: u32,
        height: u32,
        coverage: &[u8],
    ) -> Result<Option<AtlasRect>, RenderError> {
        let expected = usize::try_from(width)
            .ok()
            .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
            .ok_or_else(|| {
                RenderError::Text(format!("glyph bitmap {width}x{height} is not addressable"))
            })?;
        if coverage.len() != expected {
            return Err(RenderError::Text(format!(
                "glyph bitmap {width}x{height} carries {} bytes, expected {expected}",
                coverage.len()
            )));
        }
        if width == 0 || height == 0 {
            return Ok(None);
        }
        if let Some(rect) = self.slots.get(&key) {
            return Ok(Some(*rect));
        }

        let Some(rect) = self.allocate(width, height) else {
            return Ok(None);
        };
        self.blit(&rect, coverage);
        self.slots.insert(key, rect);
        self.mark_dirty(rect.y, rect.y.saturating_add(rect.height));
        Ok(Some(rect))
    }

    /// Finds (or makes) room for a `width * height` glyph, growing if needed.
    fn allocate(&mut self, width: u32, height: u32) -> Option<AtlasRect> {
        let slot_width = width.checked_add(GLYPH_PADDING_PX)?;
        let slot_height = height.checked_add(GLYPH_PADDING_PX)?;
        loop {
            if let Some(rect) = self.allocate_in_current(width, height, slot_width, slot_height) {
                return Some(rect);
            }
            if !self.grow() {
                return None;
            }
        }
    }

    /// One packing attempt at the current size; `None` means "grow and retry".
    fn allocate_in_current(
        &mut self,
        width: u32,
        height: u32,
        slot_width: u32,
        slot_height: u32,
    ) -> Option<AtlasRect> {
        if slot_width > self.size || slot_height > self.size {
            return None;
        }

        // A freed region first: recycling a hole keeps the shelf frontier for
        // the glyphs no hole is big enough for.
        if let Some(origin) = self.take_free(slot_width, slot_height) {
            return Some(AtlasRect {
                x: origin[0],
                y: origin[1],
                width,
                height,
            });
        }

        // Existing shelf: tall enough for this glyph and with room to its right.
        for shelf in &mut self.shelves {
            if shelf.height >= slot_height && shelf.cursor_x + slot_width <= self.size {
                let rect = AtlasRect {
                    x: shelf.cursor_x,
                    y: shelf.y,
                    width,
                    height,
                };
                shelf.cursor_x += slot_width;
                return Some(rect);
            }
        }

        // A new shelf below the last one.
        let next_y = self
            .shelves
            .last()
            .map_or(0, |shelf| shelf.y.saturating_add(shelf.height));
        if next_y.checked_add(slot_height)? > self.size {
            return None;
        }
        self.shelves.push(Shelf {
            y: next_y,
            height: slot_height,
            cursor_x: slot_width,
        });
        Some(AtlasRect {
            x: 0,
            y: next_y,
            width,
            height,
        })
    }

    /// Takes the tightest freed region a `slot_width * slot_height` glyph fits
    /// in, returning its top-left corner, and puts the two leftovers of the
    /// split back.
    ///
    /// Best fit rather than first fit because the alternative wastes a whole
    /// evicted CJK slot on an `i`: the free list is short (bounded by
    /// [`MAX_FREE_REGIONS`]) and the scan is only paid on a packing attempt
    /// that follows an eviction.
    fn take_free(&mut self, slot_width: u32, slot_height: u32) -> Option<[u32; 2]> {
        let needed = u64::from(slot_width) * u64::from(slot_height);
        let (index, _) = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, region)| region.width >= slot_width && region.height >= slot_height)
            .map(|(index, region)| {
                (
                    index,
                    u64::from(region.width) * u64::from(region.height) - needed,
                )
            })
            .min_by_key(|(_, waste)| *waste)?;
        let region = self.free.swap_remove(index);
        let leftover_width = region.width - slot_width;
        let leftover_height = region.height - slot_height;
        // Guillotine along the axis that leaves the wider of the two offcuts
        // whole, so the bigger leftover stays usable for another glyph.
        if leftover_width >= leftover_height {
            self.push_free(AtlasRect {
                x: region.x + slot_width,
                y: region.y,
                width: leftover_width,
                height: region.height,
            });
            self.push_free(AtlasRect {
                x: region.x,
                y: region.y + slot_height,
                width: slot_width,
                height: leftover_height,
            });
        } else {
            self.push_free(AtlasRect {
                x: region.x,
                y: region.y + slot_height,
                width: region.width,
                height: leftover_height,
            });
            self.push_free(AtlasRect {
                x: region.x + slot_width,
                y: region.y,
                width: leftover_width,
                height: slot_height,
            });
        }
        Some([region.x, region.y])
    }

    /// Doubles the atlas, preserving every packed rectangle. `false` when the
    /// maximum size has already been reached.
    fn grow(&mut self) -> bool {
        let Some(new_size) = self.size.checked_mul(2) else {
            return false;
        };
        if new_size > self.max_size {
            return false;
        }
        let Some(area) = usize::try_from(new_size)
            .ok()
            .and_then(|side| side.checked_mul(side))
        else {
            return false;
        };

        // Copy row by row into the top-left corner: shelves keep their `y`, so
        // all outstanding rects stay correct.
        let mut grown = vec![0u8; area];
        let old_size = self.size as usize;
        for row in 0..old_size {
            let src = row * old_size;
            let dst = row * new_size as usize;
            grown[dst..dst + old_size].copy_from_slice(&self.pixels[src..src + old_size]);
        }
        self.pixels = grown;
        self.size = new_size;
        // After the assignment, so the band names rows of the grown buffer:
        // every row moved to a new stride, and the uploader is about to be
        // handed a freshly created texture anyway.
        self.mark_dirty(0, self.size);
        true
    }

    /// Copies `coverage` into `rect`. The rect is guaranteed in-bounds by
    /// [`GlyphAtlas::allocate`]; the bounds checks below are belt-and-braces
    /// against a future packer bug, not a live condition.
    fn blit(&mut self, rect: &AtlasRect, coverage: &[u8]) {
        let stride = self.size as usize;
        for row in 0..rect.height as usize {
            let src = row * rect.width as usize;
            let Some(src_row) = coverage.get(src..src + rect.width as usize) else {
                continue;
            };
            let dst = (rect.y as usize + row) * stride + rect.x as usize;
            let Some(dst_row) = self.pixels.get_mut(dst..dst + rect.width as usize) else {
                continue;
            };
            dst_row.copy_from_slice(src_row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AtlasRect, DEFAULT_ATLAS_SIZE, GLYPH_PADDING_PX, GlyphAtlas, GlyphKey, MAX_FREE_REGIONS,
    };
    use crate::error::RenderError;

    fn key(gid: u16) -> GlyphKey {
        GlyphKey::new(0, gid, 14.0)
    }

    #[test]
    fn a_fresh_atlas_is_empty_and_clean() {
        let atlas = GlyphAtlas::new().expect("default atlas is valid");
        assert_eq!(atlas.size(), DEFAULT_ATLAS_SIZE);
        assert_eq!(atlas.len(), 0);
        assert!(atlas.is_empty());
        assert!(!atlas.is_dirty());
        assert_eq!(atlas.generation(), 0);
        assert_eq!(
            atlas.pixels().len(),
            (DEFAULT_ATLAS_SIZE * DEFAULT_ATLAS_SIZE) as usize
        );
    }

    #[test]
    fn invalid_sizes_are_rejected() {
        assert!(matches!(
            GlyphAtlas::with_size(0, 64),
            Err(RenderError::Text(_))
        ));
        assert!(matches!(
            GlyphAtlas::with_size(128, 64),
            Err(RenderError::Text(_))
        ));
    }

    #[test]
    fn a_packed_glyph_lands_where_it_was_told() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        let coverage = vec![200u8; 4 * 3];
        let rect = atlas
            .try_insert(key(1), 4, 3, &coverage)
            .expect("bitmap is well formed")
            .expect("4x3 fits in 64²");
        assert_eq!(
            rect,
            AtlasRect {
                x: 0,
                y: 0,
                width: 4,
                height: 3
            }
        );
        assert!(atlas.is_dirty());
        assert_eq!(atlas.len(), 1);

        // Pixels are where the rect says they are, and the gutter stayed clear.
        let stride = atlas.size() as usize;
        for row in 0..3 {
            for col in 0..4 {
                assert_eq!(atlas.pixels()[row * stride + col], 200);
            }
            assert_eq!(atlas.pixels()[row * stride + 4], 0);
        }
    }

    #[test]
    fn re_inserting_a_key_returns_the_same_slot() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        let first = atlas.try_insert(key(1), 4, 3, &[1u8; 12]).expect("valid");
        let second = atlas.try_insert(key(1), 4, 3, &[1u8; 12]).expect("valid");
        assert_eq!(first, second);
        assert_eq!(atlas.len(), 1);
    }

    #[test]
    fn many_glyphs_pack_without_overlapping() {
        let mut atlas = GlyphAtlas::with_size(128, 128).expect("128² atlas is valid");
        let mut rects = Vec::new();
        for gid in 0..200u16 {
            // Deliberately ragged sizes so shelves of different heights open.
            let width = 3 + u32::from(gid % 7);
            let height = 4 + u32::from(gid % 5);
            let coverage = vec![u8::try_from(gid % 256).unwrap_or(0); (width * height) as usize];
            if let Some(rect) = atlas
                .try_insert(key(gid), width, height, &coverage)
                .expect("bitmap is well formed")
            {
                rects.push(rect);
            }
        }
        assert!(
            rects.len() > 100,
            "expected most glyphs to fit, got {}",
            rects.len()
        );

        for (i, a) in rects.iter().enumerate() {
            assert!(a.x + a.width <= atlas.size() && a.y + a.height <= atlas.size());
            for b in &rects[i + 1..] {
                assert!(!a.intersects(b), "{a:?} overlaps {b:?}");
            }
        }
        assert!(atlas.occupancy() > 0.0 && atlas.occupancy() <= 1.0);
    }

    #[test]
    fn the_atlas_doubles_instead_of_failing_and_keeps_old_rects() {
        let mut atlas = GlyphAtlas::with_size(32, 128).expect("32² atlas is valid");
        let first = atlas
            .try_insert(key(0), 8, 8, &[255u8; 64])
            .expect("valid")
            .expect("fits");
        assert_eq!(atlas.size(), 32);

        // Fill well past 32² so growth is forced.
        for gid in 1..40u16 {
            let _ = atlas
                .try_insert(key(gid), 8, 8, &[255u8; 64])
                .expect("valid");
        }
        assert!(atlas.size() > 32, "atlas should have grown");
        // The first glyph is still where it was, pixels included.
        assert_eq!(atlas.get(&key(0)), Some(first));
        let stride = atlas.size() as usize;
        assert_eq!(
            atlas.pixels()[first.y as usize * stride + first.x as usize],
            255
        );
    }

    #[test]
    fn an_oversized_glyph_reports_full_rather_than_erroring() {
        let mut atlas = GlyphAtlas::with_size(16, 16).expect("16² atlas is valid");
        let huge = vec![255u8; 32 * 32];
        assert_eq!(
            atlas.try_insert(key(1), 32, 32, &huge).expect("valid"),
            None
        );
        assert_eq!(atlas.len(), 0);
    }

    #[test]
    fn empty_glyphs_are_not_packed() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        assert_eq!(atlas.try_insert(key(3), 0, 0, &[]).expect("valid"), None);
        assert!(atlas.is_empty());
        assert!(!atlas.is_dirty());
    }

    #[test]
    fn a_mismatched_bitmap_is_an_error() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        assert!(matches!(
            atlas.try_insert(key(1), 4, 4, &[0u8; 3]),
            Err(RenderError::Text(_))
        ));
    }

    #[test]
    fn clearing_bumps_the_generation_and_zeroes_the_pixels() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        let _ = atlas.try_insert(key(1), 4, 4, &[255u8; 16]).expect("valid");
        atlas.mark_clean();
        atlas.clear();
        assert_eq!(atlas.generation(), 1);
        assert!(atlas.is_empty());
        assert!(atlas.is_dirty());
        assert!(atlas.pixels().iter().all(|texel| *texel == 0));
        assert_eq!(atlas.size(), 64, "clear keeps the buffer size");
    }

    #[test]
    fn an_insert_reports_only_the_rows_it_touched() {
        let mut atlas = GlyphAtlas::new().expect("default atlas is valid");
        assert_eq!(
            atlas.dirty_rows(),
            None,
            "a fresh atlas has nothing to send"
        );

        let rect = atlas
            .try_insert(key(1), 12, 20, &[255u8; 12 * 20])
            .expect("bitmap is well formed")
            .expect("12x20 fits in the default atlas");
        assert!(atlas.is_dirty());
        assert_eq!(
            atlas.dirty_rows(),
            Some((rect.y, rect.y + rect.height)),
            "the band is the glyph's own rows, half-open",
        );

        atlas.mark_clean();
        assert!(!atlas.is_dirty());
        assert_eq!(atlas.dirty_rows(), None, "an upload retires the band");

        let second = atlas
            .try_insert(key(2), 12, 20, &[255u8; 12 * 20])
            .expect("bitmap is well formed")
            .expect("a second glyph fits");
        let Some((top, bottom)) = atlas.dirty_rows() else {
            panic!("an insert must report a band");
        };
        assert_eq!((top, bottom), (second.y, second.y + second.height));
        // The whole point: a 20-texel glyph is a few kilobytes, not a megabyte.
        assert!(
            bottom - top <= 20,
            "a 20-texel glyph dirtied {} of {} rows",
            bottom - top,
            atlas.size(),
        );
    }

    #[test]
    fn touches_between_two_uploads_collapse_into_one_band() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        atlas.mark_clean();
        let mut packed: Vec<AtlasRect> = Vec::new();
        for gid in 0..24u16 {
            let Some(rect) = atlas
                .try_insert(key(gid), 8, 8, &[255u8; 64])
                .expect("bitmap is well formed")
            else {
                break;
            };
            packed.push(rect);
        }
        assert!(
            packed.iter().any(|rect| rect.y > 0),
            "a 64² atlas must open more than one shelf",
        );
        let expected = (
            packed.iter().map(|rect| rect.y).min().unwrap_or(0),
            packed
                .iter()
                .map(|rect| rect.y + rect.height)
                .max()
                .unwrap_or(0),
        );
        assert_eq!(
            atlas.dirty_rows(),
            Some(expected),
            "the union of the touched rows, not the last one alone",
        );
    }

    #[test]
    fn an_eviction_reports_the_rows_it_zeroed() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        let mut packed: Vec<(u16, AtlasRect)> = Vec::new();
        for gid in 0..24u16 {
            let Some(rect) = atlas
                .try_insert(key(gid), 8, 8, &[255u8; 64])
                .expect("bitmap is well formed")
            else {
                break;
            };
            packed.push((gid, rect));
        }
        // A glyph off the first shelf, so the band cannot pass by accident.
        let Some(&(gid, rect)) = packed.iter().find(|(_, rect)| rect.y > 0) else {
            panic!("a 64² atlas must open more than one shelf");
        };

        // Everything so far is on the GPU; the eviction is the only change.
        atlas.mark_clean();
        assert!(atlas.evict(&key(gid)));

        assert!(atlas.is_dirty(), "zeroing evicted ink is a change");
        assert_eq!(
            atlas.dirty_rows(),
            Some((rect.y, rect.y + rect.height)),
            "without this band the evicted coverage stays live on the GPU and \
             bleeds into whatever is packed there next",
        );
    }

    #[test]
    fn a_retain_sweep_reports_the_union_of_every_row_it_zeroed() {
        // What a real frame does: one `retain` drops a whole cold working set
        // at once, across shelves. The band has to span all of them.
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        let mut packed: Vec<(u16, AtlasRect)> = Vec::new();
        for gid in 0..24u16 {
            let Some(rect) = atlas
                .try_insert(key(gid), 8, 8, &[255u8; 64])
                .expect("bitmap is well formed")
            else {
                break;
            };
            packed.push((gid, rect));
        }
        let half = packed.len() / 2;
        assert!(half > 1, "the sweep must drop more than one glyph");

        atlas.mark_clean();
        let evicted = atlas.retain(|key| usize::from(key.gid) >= half);
        assert_eq!(evicted, half);

        let dropped = &packed[..half];
        let expected = (
            dropped.iter().map(|(_, rect)| rect.y).min().unwrap_or(0),
            dropped
                .iter()
                .map(|(_, rect)| rect.y + rect.height)
                .max()
                .unwrap_or(0),
        );
        assert_eq!(
            atlas.dirty_rows(),
            Some(expected),
            "the sweep's band spans every shelf it touched",
        );
    }

    #[test]
    fn a_clear_and_a_doubling_ask_for_every_row() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        let _ = atlas.try_insert(key(1), 4, 4, &[255u8; 16]).expect("valid");
        atlas.mark_clean();
        atlas.clear();
        assert_eq!(atlas.dirty_rows(), Some((0, atlas.size())));

        let mut growing = GlyphAtlas::with_size(32, 128).expect("32² atlas is valid");
        let _ = growing
            .try_insert(key(0), 8, 8, &[255u8; 64])
            .expect("valid");
        growing.mark_clean();
        for gid in 1..40u16 {
            let _ = growing
                .try_insert(key(gid), 8, 8, &[255u8; 64])
                .expect("valid");
        }
        assert!(growing.size() > 32, "the atlas should have grown");
        assert_eq!(
            growing.dirty_rows(),
            Some((0, growing.size())),
            "a doubling re-strides every row, and names them at the new size",
        );
    }

    #[test]
    fn evicting_cold_glyphs_makes_room_without_moving_the_survivors() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        let coverage = vec![255u8; 8 * 8];
        let mut packed: Vec<AtlasRect> = Vec::new();
        while let Some(rect) = atlas
            .try_insert(
                key(u16::try_from(packed.len()).unwrap_or(u16::MAX)),
                8,
                8,
                &coverage,
            )
            .expect("bitmap is well formed")
        {
            packed.push(rect);
            assert!(packed.len() < 200, "a 64² atlas must fill up");
        }
        let full = packed.len();
        assert!(full > 4, "expected a few shelves, got {full}");
        let generation = atlas.generation();

        let half = full / 2;
        let evicted = atlas.retain(|key| usize::from(key.gid) >= half);
        assert_eq!(evicted, half);
        assert_eq!(
            atlas.generation(),
            generation,
            "per-glyph eviction is not a rebuild",
        );
        for (gid, rect) in packed.iter().enumerate().skip(half) {
            assert_eq!(
                atlas.get(&key(u16::try_from(gid).unwrap_or(u16::MAX))),
                Some(*rect),
                "a survivor must not have moved",
            );
        }

        // The freed room is usable again, and what it hands out cannot land on
        // a survivor.
        let mut refilled = 0usize;
        while let Some(rect) = atlas
            .try_insert(key(1000 + refilled as u16), 8, 8, &coverage)
            .expect("bitmap is well formed")
        {
            for live in &packed[half..] {
                assert!(!rect.intersects(live), "{rect:?} overlaps {live:?}");
            }
            refilled += 1;
            assert!(refilled <= full, "eviction cannot create room");
        }
        assert_eq!(refilled, half, "every evicted slot came back");
    }

    #[test]
    fn an_evicted_glyphs_ink_is_zeroed_so_it_cannot_bleed_into_its_successor() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        let first = atlas
            .try_insert(key(1), 8, 8, &[255u8; 64])
            .expect("bitmap is well formed")
            .expect("8x8 fits");
        assert!(atlas.evict(&key(1)));
        assert!(atlas.is_empty());
        assert!(
            atlas.pixels().iter().all(|texel| *texel == 0),
            "an evicted glyph leaves no coverage behind",
        );

        let second = atlas
            .try_insert(key(2), 4, 4, &[128u8; 16])
            .expect("bitmap is well formed")
            .expect("the freed region is re-used");
        assert_eq!([second.x, second.y], [first.x, first.y]);
        let stride = atlas.size() as usize;
        assert_eq!(
            atlas.pixels()[4 * stride + 4],
            0,
            "the successor's gutter must be clear of its predecessor",
        );
    }

    #[test]
    fn the_gap_a_run_of_evicted_glyphs_leaves_is_one_region_again() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        for gid in 0..6u16 {
            let _ = atlas
                .try_insert(key(gid), 8, 8, &[255u8; 64])
                .expect("bitmap is well formed")
                .expect("8x8 fits");
        }
        // Keep the leftmost glyph of the shelf; the five behind it must come
        // back as ONE region, not five slivers a wide glyph cannot use.
        assert_eq!(atlas.retain(|key| key.gid == 0), 5);
        assert_eq!(atlas.free_regions(), 1);
        let wide = atlas
            .try_insert(key(9), 26, 8, &[255u8; 26 * 8])
            .expect("bitmap is well formed")
            .expect("26x8 fits the region the five left behind");
        assert_eq!(
            [wide.x, wide.y],
            [8 + GLYPH_PADDING_PX, 0],
            "a glyph wider than any single freed slot must land in the gap",
        );
    }

    #[test]
    fn evicting_everything_resets_the_packer_rather_than_leaving_a_free_list() {
        let mut atlas = GlyphAtlas::with_size(64, 64).expect("64² atlas is valid");
        for gid in 0..3u16 {
            let _ = atlas
                .try_insert(key(gid), 8, 8, &[255u8; 64])
                .expect("bitmap is well formed")
                .expect("8x8 fits");
        }
        assert_eq!(atlas.retain(|_| false), 3);
        assert_eq!(
            atlas.free_regions(),
            0,
            "an empty atlas is the shelf packer's own state, not a free list",
        );
        assert_eq!(atlas.generation(), 0, "and still not a rebuild");
        let rect = atlas
            .try_insert(key(9), 40, 20, &[255u8; 40 * 20])
            .expect("bitmap is well formed")
            .expect("a glyph too big for any freed slot fits the empty atlas");
        assert_eq!([rect.x, rect.y], [0, 0]);
    }

    #[test]
    fn fits_separates_a_glyph_too_big_to_ever_pack_from_a_full_atlas() {
        let atlas = GlyphAtlas::with_size(16, 64).expect("16² atlas is valid");
        assert!(
            atlas.fits(60, 60),
            "growth up to max_size is what fits means"
        );
        assert!(!atlas.fits(64, 8), "64 plus its gutter is past the maximum");
        assert!(!atlas.fits(8, 0), "a glyph with no ink is not packable");
        assert!(!atlas.fits(u32::MAX, 8));
    }

    #[test]
    fn the_free_list_stays_bounded_under_fragmentation() {
        let mut atlas = GlyphAtlas::with_size(256, 256).expect("256² atlas is valid");
        let coverage = [255u8; 16];
        let mut gid = 0u16;
        while atlas
            .try_insert(key(gid), 4, 4, &coverage)
            .expect("bitmap is well formed")
            .is_some()
        {
            gid = gid.saturating_add(1);
            assert!(gid < u16::MAX, "a 256² atlas must fill up");
        }
        // A checkerboard of survivors: the worst fragmentation the packer can
        // be handed, since no two freed regions touch.
        let evicted = atlas.retain(|key| key.gid % 2 == 0);
        assert!(
            evicted > MAX_FREE_REGIONS,
            "the fragmentation must exceed the cap to test it, got {evicted}",
        );
        assert!(
            atlas.free_regions() <= MAX_FREE_REGIONS,
            "the free list is bounded, got {}",
            atlas.free_regions(),
        );
    }

    #[test]
    fn uv_bounds_follow_the_atlas_size() {
        let rect = AtlasRect {
            x: 16,
            y: 32,
            width: 8,
            height: 4,
        };
        assert_eq!(rect.uv(64), [0.25, 0.5, 0.375, 0.5625]);
        assert_eq!(rect.uv(0), [0.0; 4]);
    }

    #[test]
    fn glyph_keys_round_trip_their_size() {
        let key = GlyphKey::new(2, 77, 13.5);
        assert_eq!(key.size_px(), 13.5);
        assert_ne!(key, GlyphKey::new(2, 77, 14.0));
        assert_eq!(GLYPH_PADDING_PX, 1);
    }
}
