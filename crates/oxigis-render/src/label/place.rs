//! Label placement: anchors from decoded MVT features, and a greedy
//! collision pass over them (blueprint §5.3, part B).
//!
//! [`crate::label::engine`] answers *what a label looks like*; this module
//! answers *where it goes*. The two meet at [`OwnedPlacedLabel`], which is the
//! owned twin of [`PlacedLabel`] that [`crate::label::LabelPipeline`] draws.
//!
//! # The pass, in order
//!
//! ```text
//! LabelPlacer::new(viewport_px)
//!   for every visible tile:  place_tile(&mut engine, &tile, &placement, &labels)
//! finish() -> Vec<OwnedPlacedLabel>
//! ```
//!
//! One placer covers **one frame**: the accepted-box set spans every tile, so
//! labels from neighbouring tiles collide with each other exactly as they would
//! within a tile.
//!
//! # Frame contract for the drawing pass
//!
//! [`LabelEngine::shape`] may fill the glyph atlas, clear it and repack — which
//! bumps [`LabelEngine::generation`] and invalidates every [`ShapedLabel`]
//! handed out before it. The placer therefore records the generation of the
//! first label it shapes and reports a mismatch through
//! [`LabelPlacer::is_stale`]. The contract is:
//!
//! 1. run the whole placement pass,
//! 2. check [`LabelPlacer::is_stale`] — **before** [`LabelPlacer::finish`],
//!    which consumes the placer. If it is `true`, throw the placer away and run
//!    the pass again: a repack empties the shaping cache, so the second pass
//!    re-shapes, but it does so against an atlas that now has room and
//!    therefore (barring a label set larger than the atlas itself) converges,
//! 3. [`LabelPlacer::finish`], then `upload_atlas`, then `upload_labels` with
//!    the placements borrowed through [`placed_labels`].
//!
//! `upload_atlas` must run *after* the last `place_tile`, or the pixels of the
//! last labels shaped will not have reached the GPU.
//!
//! # Conventions
//!
//! * **Tile coordinates in, viewport pixels out** — anchors are computed in the
//!   layer's own `0..=extent` grid (`y` down, as MVT stores it) and projected
//!   with the same maths [`crate::vector::VectorLayerRenderer`] uses for its
//!   meshes: `screen = placement.{x,y} + tile_coord * placement.size / extent`.
//!   MVT's `y`-down grid and the viewport's `y`-down pixels agree, so no flip
//!   is involved.
//! * **Anchored boxes** — the collision box hangs off the anchor at
//!   [`LabelSpec::anchor`], shifted by [`LabelSpec::offset_px`]. The default is
//!   [`LabelAnchorPoint::Center`] with no offset, which is the centred box every
//!   geometry kind got before the field existed.
//! * **Partly visible labels count** — a candidate is kept when its padded box
//!   *intersects* the viewport, not when the viewport contains it: the label
//!   pass is scissored, so the visible half of an edge label draws correctly.
//! * **Integer origins** — the origin is rounded, because the rasteriser has no
//!   subpixel phase (see [`crate::label::engine`]).
//! * **Horizontal text only** — no rotation, no line-following, no leader
//!   lines. `TODO.md` §5.3 names those a rabbit hole; they are Phase 2+.
//!
//! # What is still ordered per tile
//!
//! Candidates are sorted, and therefore compete, **within one layer of one
//! tile**: the accepted-box set spans the frame, but the *order* labels are
//! offered in is the caller's tile order. A capital city in a tile placed late
//! can still lose to a hamlet in a tile placed early. Closing that needs the
//! whole frame's candidates in one vector and one normalised rank across
//! geometry kinds; [`LabelSpec::rank_property`] is the per-feature half of it —
//! it fixes the ordering inside a layer, which is where a `symbolrank` or
//! `population` property lives.
//!
//! # Duplicate labels at tile seams
//!
//! An MVT tile carries features that spill out of its `0..=extent` square so
//! that geometry crossing the seam can be drawn without a gap. Those same
//! features appear in the neighbouring tile too, so labelling them would draw
//! the same name twice. The rule that prevents it: **an anchor outside
//! `0..=extent` on either axis is dropped**, so exactly the tile whose square
//! contains the anchor labels the feature. It costs nothing and needs no
//! cross-tile identity.
//!
//! It is not airtight: a feature the tiler *split* into two pieces has an
//! anchor per piece, and both can be in bounds, so a long river can be labelled
//! on both sides of a seam. Deduplicating that needs feature identity across
//! tiles, which is Phase 2.

use std::borrow::Cow;
use std::sync::Arc;

use crate::error::RenderError;
use crate::label::engine::{
    LabelEngine, LabelOrientation, LabelWeight, MAX_LABEL_SIZE_PX, ShapedLabel,
};
use crate::label::pipeline::{LabelHalo, PlacedLabel};
use crate::mvt::decode::{MvtGeometry, MvtPolygon, MvtValue, VectorTile};
use crate::viewport::TilePlacement;

/// Breathing room added around every label's collision box, in pixels per side.
///
/// Purely aesthetic: two labels whose boxes touch exactly are legible but look
/// cramped, and glyph boxes are tight around the ink.
pub const LABEL_PADDING_PX: f32 = 2.0;

/// How far outside the viewport a label may still be placed, in pixels.
///
/// Zero: a candidate has to touch the viewport to be worth a collision slot.
/// The knob exists because the one reason to grow it — letting a label that is
/// about to slide in reserve its space a frame early, so the labels already on
/// screen do not reshuffle as it arrives — only pays off with the placement
/// persistence this phase does not have. Until then a positive buffer would
/// shape and pack invisible text.
const VIEWPORT_BUFFER_PX: f32 = 0.0;

/// Upper bound on one glyph's advance, in ems, for the pre-shaping cull.
///
/// Deliberately far above any advance a real face declares (full-width CJK is
/// 1 em, the widest Latin ornaments are under 2): multiplied by the text's
/// **byte** length — itself at least its character count — it gives a bound on
/// the label's extent that shaping cannot exceed, so the cull can only ever
/// drop a label that could not have been placed.
const MAX_GLYPH_ADVANCE_EM: f32 = 3.0;

/// How to label one vector-tile layer.
///
/// The render-side twin of [`crate::vector::LayerPaint`]: the shell evaluates
/// its own style for the current zoom and hands the placer a plain spec per
/// layer name. A layer no [`LabelResolver`] has a spec for is not labelled.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelSpec {
    /// Feature property holding the label text, e.g. `"name"`. A feature
    /// without that property, or with a value [`label_text`] renders as
    /// nothing, is skipped.
    pub text_property: String,
    /// Font size in physical pixels; the label is rasterised at exactly this
    /// size. Outside `0.0 < size_px <=` [`MAX_LABEL_SIZE_PX`] the layer is
    /// skipped rather than failing the frame.
    pub size_px: f32,
    /// Straight sRGB fill colour.
    pub color: [u8; 4],
    /// Optional halo. Its width widens the collision box on every side.
    pub halo: Option<LabelHalo>,
    /// Which face chain shapes the label (print/text v1.4). Defaults to
    /// [`LabelWeight::Regular`], so every pre-v1.4 construction — this type
    /// is only ever built through [`LabelSpec::new`] plus the `with_*`
    /// builders — keeps its exact behaviour.
    pub weight: LabelWeight,
    /// Which way the label's glyphs run (print/text v1.5). Defaults to
    /// [`LabelOrientation::Horizontal`], for the same reason `weight`
    /// defaults to Regular. A [`LabelOrientation::Vertical`] spec whose text
    /// the ladder refuses draws horizontally, so this field can never make a
    /// label disappear — only make its collision box tall and narrow instead
    /// of short and wide.
    pub orientation: LabelOrientation,
    /// Feature property carrying the label's rank, **larger placing first**.
    ///
    /// Overrides [`LabelAnchor::priority`] for the features that have it, which
    /// is what stops a hamlet from beating a capital inside a places layer:
    /// every point anchor has priority `0.0`, so without a rank the order is
    /// the tile's feature order.
    ///
    /// The value must be numeric — see [`rank_value`]. MapLibre's
    /// `symbol-sort-key` runs the other way (lower places first), so a shell
    /// mapping one onto the other negates it.
    ///
    /// Holding a [`String`] here means [`LabelResolver::label_for`] clones two
    /// of them; the placer reads specs through [`LabelResolver::label_spec`],
    /// which borrows and clones neither.
    pub rank_property: Option<String>,
    /// Which point of the label's box sits on the feature's anchor.
    pub anchor: LabelAnchorPoint,
    /// Shift applied to the box after [`LabelSpec::anchor`], in pixels, `y`
    /// down — how a point label is moved clear of its own marker.
    pub offset_px: [f32; 2],
}

/// Which point of the label's box sits on the feature's anchor.
///
/// Named the way MapLibre's `text-anchor` is: the variant names the part of the
/// **label** that is pinned to the feature, so [`LabelAnchorPoint::Top`] puts
/// the label's top edge on the anchor and the text hangs below it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum LabelAnchorPoint {
    /// The box's centre — the default, and what every label did before the
    /// field existed.
    #[default]
    Center,
    /// The middle of the box's left edge: the label runs to the right of the
    /// feature.
    Left,
    /// The middle of the box's right edge: the label runs to its left.
    Right,
    /// The middle of the box's top edge: the label hangs below the feature.
    Top,
    /// The middle of the box's bottom edge: the label sits above it.
    Bottom,
    /// The box's top-left corner.
    TopLeft,
    /// The box's top-right corner.
    TopRight,
    /// The box's bottom-left corner.
    BottomLeft,
    /// The box's bottom-right corner.
    BottomRight,
}

impl LabelAnchorPoint {
    /// Shift from the anchor to the box's top-left corner, for a box of
    /// `size_px`.
    ///
    /// [`LabelAnchorPoint::Center`] gives `[-width / 2, -height / 2]`, which is
    /// the centring the pass has always done.
    #[must_use]
    pub fn origin_shift(self, size_px: [f32; 2]) -> [f32; 2] {
        let [width, height] = size_px;
        let x = match self {
            Self::Left | Self::TopLeft | Self::BottomLeft => 0.0,
            Self::Center | Self::Top | Self::Bottom => -width / 2.0,
            Self::Right | Self::TopRight | Self::BottomRight => -width,
        };
        let y = match self {
            Self::Top | Self::TopLeft | Self::TopRight => 0.0,
            Self::Center | Self::Left | Self::Right => -height / 2.0,
            Self::Bottom | Self::BottomLeft | Self::BottomRight => -height,
        };
        [x, y]
    }
}

impl LabelSpec {
    /// Default label size in physical pixels.
    pub const DEFAULT_SIZE_PX: f32 = 12.0;

    /// A black, halo-less spec reading `text_property`.
    #[must_use]
    pub fn new(text_property: impl Into<String>) -> Self {
        Self {
            text_property: text_property.into(),
            size_px: Self::DEFAULT_SIZE_PX,
            color: [0, 0, 0, 255],
            halo: None,
            weight: LabelWeight::Regular,
            orientation: LabelOrientation::Horizontal,
            rank_property: None,
            anchor: LabelAnchorPoint::Center,
            offset_px: [0.0, 0.0],
        }
    }

    /// Returns the spec with a new font size in physical pixels.
    #[must_use]
    pub const fn with_size_px(mut self, size_px: f32) -> Self {
        self.size_px = size_px;
        self
    }

    /// Returns the spec with a new fill colour.
    #[must_use]
    pub const fn with_color(mut self, color: [u8; 4]) -> Self {
        self.color = color;
        self
    }

    /// Returns the spec with a halo.
    #[must_use]
    pub const fn with_halo(mut self, halo: LabelHalo) -> Self {
        self.halo = Some(halo);
        self
    }

    /// Returns the spec drawn at `weight`.
    #[must_use]
    pub const fn with_weight(mut self, weight: LabelWeight) -> Self {
        self.weight = weight;
        self
    }

    /// Returns the spec drawn at `orientation`.
    #[must_use]
    pub const fn with_orientation(mut self, orientation: LabelOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Returns the spec ranked by a feature property, larger placing first.
    #[must_use]
    pub fn with_rank_property(mut self, rank_property: impl Into<String>) -> Self {
        self.rank_property = Some(rank_property.into());
        self
    }

    /// Returns the spec whose box hangs off the anchor at `anchor`.
    #[must_use]
    pub const fn with_anchor(mut self, anchor: LabelAnchorPoint) -> Self {
        self.anchor = anchor;
        self
    }

    /// Returns the spec shifted by `offset_px` after anchoring.
    #[must_use]
    pub const fn with_offset_px(mut self, offset_px: [f32; 2]) -> Self {
        self.offset_px = offset_px;
        self
    }

    /// Whether [`LabelSpec::size_px`] is a size the engine can shape at.
    #[must_use]
    pub fn has_usable_size(&self) -> bool {
        self.size_px.is_finite() && self.size_px > 0.0 && self.size_px <= MAX_LABEL_SIZE_PX
    }

    /// Halo width in pixels, or `0.0` when there is no halo — the amount the
    /// collision box grows on every side beyond [`LABEL_PADDING_PX`].
    #[must_use]
    pub fn halo_padding_px(&self) -> f32 {
        match self.halo {
            Some(halo) if halo.width_px.is_finite() && halo.width_px > 0.0 => halo.width_px,
            _ => 0.0,
        }
    }
}

/// Supplies the label spec for a vector-tile layer, by name.
///
/// The label counterpart of [`crate::vector::PaintResolver`], with the same
/// contract: implemented by the shell against its own style model, [`None`]
/// meaning "do not label this layer". [`LabelTable`] is the trivial
/// implementation this crate ships for tests and simple viewers.
pub trait LabelResolver {
    /// The spec for `layer_name`, or [`None`] to leave the layer unlabelled.
    fn label_for(&self, layer_name: &str) -> Option<LabelSpec>;

    /// [`LabelResolver::label_for`] without the clone, which is what
    /// [`LabelPlacer::place_tile`] calls.
    ///
    /// A spec owns two [`String`]s, and the placer asks for one per layer of
    /// every visible tile every frame — a screenful of tiles is hundreds of
    /// allocations a frame purely to answer "how is this layer labelled". An
    /// implementation that stores its specs (all of them, in practice) should
    /// override this with [`Cow::Borrowed`] and allocate none of them; the
    /// default clones through `label_for` so that implementing the trait stays
    /// a one-method job.
    fn label_spec(&self, layer_name: &str) -> Option<Cow<'_, LabelSpec>> {
        self.label_for(layer_name).map(Cow::Owned)
    }
}

/// A [`LabelResolver`] backed by an ordered `(layer name, spec)` list.
///
/// Lookup is a linear scan and the **first** matching entry wins, exactly as in
/// [`crate::vector::PaintTable`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LabelTable {
    entries: Vec<(String, LabelSpec)>,
}

impl LabelTable {
    /// Creates an empty table, which labels nothing.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds an entry, keeping insertion order.
    pub fn push(&mut self, layer_name: impl Into<String>, spec: LabelSpec) {
        self.entries.push((layer_name.into(), spec));
    }

    /// Builder form of [`LabelTable::push`].
    #[must_use]
    pub fn with(mut self, layer_name: impl Into<String>, spec: LabelSpec) -> Self {
        self.push(layer_name, spec);
        self
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table labels nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries, in insertion order.
    #[must_use]
    pub fn entries(&self) -> &[(String, LabelSpec)] {
        &self.entries
    }

    /// Removes every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl FromIterator<(String, LabelSpec)> for LabelTable {
    fn from_iter<I: IntoIterator<Item = (String, LabelSpec)>>(iter: I) -> Self {
        Self {
            entries: iter.into_iter().collect(),
        }
    }
}

impl LabelResolver for LabelTable {
    fn label_for(&self, layer_name: &str) -> Option<LabelSpec> {
        self.label_spec(layer_name).map(Cow::into_owned)
    }

    fn label_spec(&self, layer_name: &str) -> Option<Cow<'_, LabelSpec>> {
        self.entries
            .iter()
            .find(|(name, _)| name == layer_name)
            .map(|(_, spec)| Cow::Borrowed(spec))
    }
}

impl<T: LabelResolver + ?Sized> LabelResolver for &T {
    fn label_for(&self, layer_name: &str) -> Option<LabelSpec> {
        (**self).label_for(layer_name)
    }

    /// Forwarded, or `&&LabelTable` would silently take the cloning default.
    fn label_spec(&self, layer_name: &str) -> Option<Cow<'_, LabelSpec>> {
        (**self).label_spec(layer_name)
    }
}

/// Renders an MVT property value as label text.
///
/// | Variant | Rendering |
/// |---|---|
/// | [`MvtValue::String`] | used as-is |
/// | [`MvtValue::F32`] / [`MvtValue::F64`] | Rust's shortest round-trip form, [`None`] if not finite |
/// | [`MvtValue::I64`] / [`MvtValue::U64`] | decimal |
/// | [`MvtValue::Bool`] | [`None`] — a map label reading "true" is a styling mistake, not a label |
///
/// [`None`] means "this feature has no label"; so does a value that renders to
/// whitespace only.
#[must_use]
pub fn label_text(value: &MvtValue) -> Option<String> {
    label_text_cow(value).map(Cow::into_owned)
}

/// [`label_text`] without the copy for the case that needs none.
///
/// [`Cow::Borrowed`] for [`MvtValue::String`] — which is every real basemap
/// label — and [`Cow::Owned`] for the numeric variants, which genuinely have to
/// be formatted. The placer holds the borrow only for as long as it holds the
/// decoded tile, so the common path allocates nothing per feature per frame.
#[must_use]
pub fn label_text_cow(value: &MvtValue) -> Option<Cow<'_, str>> {
    let text = match value {
        MvtValue::String(text) => Cow::Borrowed(text.as_str()),
        MvtValue::F32(number) if number.is_finite() => Cow::Owned(number.to_string()),
        MvtValue::F64(number) if number.is_finite() => Cow::Owned(number.to_string()),
        MvtValue::I64(number) => Cow::Owned(number.to_string()),
        MvtValue::U64(number) => Cow::Owned(number.to_string()),
        MvtValue::F32(_) | MvtValue::F64(_) | MvtValue::Bool(_) => return None,
    };
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

/// The rank an [`MvtValue`] carries for [`LabelSpec::rank_property`], larger
/// placing first.
///
/// Finite numbers only: a rank is compared, and neither a string nor a boolean
/// has an order a placement pass could defend. [`None`] means "this feature is
/// not ranked", which falls back to [`LabelAnchor::priority`] — so a partly
/// ranked layer keeps working, with its unranked features ordered among
/// themselves as before.
#[must_use]
pub fn rank_value(value: &MvtValue) -> Option<f64> {
    let rank = match value {
        MvtValue::F32(number) => f64::from(*number),
        MvtValue::F64(number) => *number,
        MvtValue::I64(number) => *number as f64,
        MvtValue::U64(number) => *number as f64,
        MvtValue::String(_) | MvtValue::Bool(_) => return None,
    };
    rank.is_finite().then_some(rank)
}

/// Which geometry an anchor came from — the placement rules are identical, this
/// only explains how [`LabelAnchor::priority`] was measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorKind {
    /// The feature's first point.
    Point,
    /// The arc-length midpoint of the feature's longest line string.
    Line,
    /// The centroid of the largest-area exterior ring of the feature.
    Polygon,
}

/// Where one feature wants its label, in the layer's tile-local grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelAnchor {
    /// Position in tile-local coordinates, `y` down.
    pub position: [f64; 2],
    /// What the position was derived from.
    pub kind: AnchorKind,
    /// Ranking key, larger placing first: polygon `|area|`, line length, and
    /// `0.0` for points.
    ///
    /// The units differ per kind (tile units squared versus tile units), which
    /// is harmless because priorities are only ever compared **within one layer
    /// of one tile**, where the kind and the extent are effectively fixed.
    /// Normalising across kinds is Phase 2.
    pub priority: f64,
}

/// The anchor for one feature's geometry, in tile-local coordinates.
///
/// * `Points` — the **first** point only. A multi-point feature gets one label,
///   not one per point.
/// * `Lines` — the arc-length midpoint of the longest line string: walk the
///   segments until half the total length is consumed, then interpolate inside
///   that segment. The text stays horizontal there; it is not rotated to the
///   line.
/// * `Polygons` — the shoelace centroid of the exterior ring with the largest
///   `|signed area|` among the feature's polygons. Holes are not subtracted.
///   For a ring whose area is zero (a degenerate sliver) the vertex average is
///   used instead.
///
/// Returns [`None`] for an empty geometry, and for any input that produces a
/// non-finite position.
///
/// # Concave shapes
///
/// A centroid can fall outside a C- or L-shaped polygon, putting the label off
/// its own feature. Accepted for v1: a pole-of-inaccessibility solver is
/// explicitly out of scope for this phase.
#[must_use]
pub fn feature_anchor(geometry: &MvtGeometry) -> Option<LabelAnchor> {
    let anchor = match geometry {
        MvtGeometry::Points(points) => {
            let first = points.first()?;
            LabelAnchor {
                position: [f64::from(first[0]), f64::from(first[1])],
                kind: AnchorKind::Point,
                priority: 0.0,
            }
        }
        MvtGeometry::Lines(lines) => {
            let mut best: Option<(f64, &Vec<[i32; 2]>)> = None;
            for line in lines {
                let length = polyline_length(line);
                if length <= 0.0 {
                    continue;
                }
                if best.is_none_or(|(seen, _)| length > seen) {
                    best = Some((length, line));
                }
            }
            let (length, line) = match best {
                Some(found) => found,
                // Every line is degenerate (zero length): fall back to the
                // first vertex there is, so a labelled point-like line is not
                // silently lost.
                None => {
                    let first = lines.iter().flatten().next()?;
                    return finite_anchor(LabelAnchor {
                        position: [f64::from(first[0]), f64::from(first[1])],
                        kind: AnchorKind::Line,
                        priority: 0.0,
                    });
                }
            };
            LabelAnchor {
                position: polyline_midpoint(line, length)?,
                kind: AnchorKind::Line,
                priority: length,
            }
        }
        MvtGeometry::Polygons(polygons) => {
            let mut best: Option<(f64, [f64; 2])> = None;
            for polygon in polygons {
                let Some((area, centroid)) = ring_centroid(polygon) else {
                    continue;
                };
                if best.is_none_or(|(seen, _)| area > seen) {
                    best = Some((area, centroid));
                }
            }
            let (area, position) = best?;
            LabelAnchor {
                position,
                kind: AnchorKind::Polygon,
                priority: area,
            }
        }
    };
    finite_anchor(anchor)
}

/// Passes an anchor through only if its position is finite.
fn finite_anchor(anchor: LabelAnchor) -> Option<LabelAnchor> {
    if anchor.position.iter().all(|value| value.is_finite()) {
        Some(anchor)
    } else {
        None
    }
}

/// Total length of a polyline, in tile units.
fn polyline_length(line: &[[i32; 2]]) -> f64 {
    line.windows(2)
        .map(|pair| segment_length(pair[0], pair[1]))
        .sum()
}

/// Euclidean distance between two tile-local points.
fn segment_length(from: [i32; 2], to: [i32; 2]) -> f64 {
    let dx = f64::from(to[0]) - f64::from(from[0]);
    let dy = f64::from(to[1]) - f64::from(from[1]);
    dx.hypot(dy)
}

/// The point half of `total` along `line`, interpolated inside the segment that
/// straddles it.
fn polyline_midpoint(line: &[[i32; 2]], total: f64) -> Option<[f64; 2]> {
    let half = total / 2.0;
    let mut walked = 0.0;
    for pair in line.windows(2) {
        let length = segment_length(pair[0], pair[1]);
        if length <= 0.0 {
            continue;
        }
        if walked + length >= half {
            let t = ((half - walked) / length).clamp(0.0, 1.0);
            let from = [f64::from(pair[0][0]), f64::from(pair[0][1])];
            let to = [f64::from(pair[1][0]), f64::from(pair[1][1])];
            return Some([
                from[0] + (to[0] - from[0]) * t,
                from[1] + (to[1] - from[1]) * t,
            ]);
        }
        walked += length;
    }
    // Floating-point drift only: the walk consumed everything without reaching
    // the halfway mark. The last vertex is the best answer available.
    line.last()
        .map(|last| [f64::from(last[0]), f64::from(last[1])])
}

/// `(|area|, centroid)` of a polygon's exterior ring.
///
/// The shoelace centroid, or — when the ring encloses no area — the average of
/// its vertices, which keeps degenerate slivers labelable at priority `0.0`.
fn ring_centroid(polygon: &MvtPolygon) -> Option<(f64, [f64; 2])> {
    let ring = &polygon.exterior;
    if ring.is_empty() {
        return None;
    }
    let mut area2 = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for index in 0..ring.len() {
        // Rings are stored unclosed, so the last edge wraps to the first point.
        let current = ring[index];
        let next = ring[(index + 1) % ring.len()];
        let (x0, y0) = (f64::from(current[0]), f64::from(current[1]));
        let (x1, y1) = (f64::from(next[0]), f64::from(next[1]));
        let cross = x0 * y1 - x1 * y0;
        area2 += cross;
        cx += (x0 + x1) * cross;
        cy += (y0 + y1) * cross;
    }
    if area2.abs() > f64::EPSILON {
        return Some((
            (area2 / 2.0).abs(),
            [cx / (3.0 * area2), cy / (3.0 * area2)],
        ));
    }
    let count = ring.len() as f64;
    let sum = ring.iter().fold([0.0, 0.0], |acc, point| {
        [acc[0] + f64::from(point[0]), acc[1] + f64::from(point[1])]
    });
    Some((0.0, [sum[0] / count, sum[1] / count]))
}

/// An axis-aligned collision box in viewport pixels, `y` down.
///
/// Stored padded: it is the label's own box grown by its halo width and
/// [`LABEL_PADDING_PX`] on every side, which is the box both the overlap test
/// and the viewport-containment test use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelBox {
    /// Left edge.
    pub min_x: f32,
    /// Top edge.
    pub min_y: f32,
    /// Right edge.
    pub max_x: f32,
    /// Bottom edge.
    pub max_y: f32,
}

impl LabelBox {
    /// Whether two boxes share any area. Boxes that merely touch do not.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min_x < other.max_x
            && other.min_x < self.max_x
            && self.min_y < other.max_y
            && other.min_y < self.max_y
    }

    /// Whether the box lies wholly inside `0..=viewport_px` on both axes.
    ///
    /// Placement does **not** use this — see [`LabelBox::is_inside_buffered`];
    /// it is kept for callers that want to know whether a label is fully
    /// visible, which is a different question.
    #[must_use]
    pub fn is_inside(&self, viewport_px: [f32; 2]) -> bool {
        self.min_x >= 0.0
            && self.min_y >= 0.0
            && self.max_x <= viewport_px[0]
            && self.max_y <= viewport_px[1]
    }

    /// Whether the box shares any area with `0..=viewport_px` grown by
    /// `buffer_px` on every side. Boxes that merely touch do not, as in
    /// [`LabelBox::intersects`].
    ///
    /// This is the test placement applies, and the difference from
    /// [`LabelBox::is_inside`] is three visible artefacts: requiring
    /// containment leaves the whole border of the map unlabelled — which is
    /// where a user pans towards — makes labels pop in rather than slide in,
    /// and drops any label larger than the window outright. Nothing downstream
    /// needs containment: [`crate::label::LabelPipeline::draw`] scissors the
    /// pass to the viewport, so the visible half of an edge label is what
    /// reaches the screen.
    ///
    /// A non-finite edge or buffer fails, every comparison against NaN being
    /// false.
    #[must_use]
    pub fn is_inside_buffered(&self, viewport_px: [f32; 2], buffer_px: f32) -> bool {
        self.max_x > -buffer_px
            && self.max_y > -buffer_px
            && self.min_x < viewport_px[0] + buffer_px
            && self.min_y < viewport_px[1] + buffer_px
    }

    /// Whether every edge is a finite number.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.min_x.is_finite()
            && self.min_y.is_finite()
            && self.max_x.is_finite()
            && self.max_y.is_finite()
    }
}

/// A placed label that owns its [`ShapedLabel`].
///
/// [`PlacedLabel`] borrows the shaped label, which a pass accumulating across
/// tiles cannot do: the engine stays mutably borrowed for the whole pass.
/// Because [`LabelEngine::shape`] hands out [`Arc`]s, holding one costs a
/// refcount bump. Borrow the whole set for drawing with [`placed_labels`].
#[derive(Debug, Clone)]
pub struct OwnedPlacedLabel {
    /// The shaped label, from [`LabelEngine::shape`].
    pub shaped: Arc<ShapedLabel>,
    /// Top-left of the label box, in viewport pixels, `y` down, whole numbers.
    pub origin_px: [f32; 2],
    /// Straight sRGB fill colour.
    pub color: [u8; 4],
    /// Optional halo drawn underneath the fill.
    pub halo: Option<LabelHalo>,
    /// The padded collision box this label occupied.
    pub collision_box: LabelBox,
}

impl OwnedPlacedLabel {
    /// Borrows the label in the form [`crate::label::LabelPipeline`] draws.
    #[must_use]
    pub fn as_placed(&self) -> PlacedLabel<'_> {
        PlacedLabel {
            shaped: &self.shaped,
            origin_px: self.origin_px,
            color: self.color,
            halo: self.halo,
        }
    }
}

/// Borrows a whole placement set for
/// [`crate::label::LabelPipeline::upload_labels`].
#[must_use]
pub fn placed_labels(labels: &[OwnedPlacedLabel]) -> Vec<PlacedLabel<'_>> {
    let mut borrowed = Vec::new();
    placed_labels_into(labels, &mut borrowed);
    borrowed
}

/// [`placed_labels`] into a caller-owned buffer, which is what a per-frame
/// caller wants: the borrow list is rebuilt every frame and is exactly as long
/// as the last one, so reusing the allocation costs nothing and saves one heap
/// round trip per frame.
///
/// `borrowed` is cleared first.
pub fn placed_labels_into<'a>(labels: &'a [OwnedPlacedLabel], borrowed: &mut Vec<PlacedLabel<'a>>) {
    borrowed.clear();
    borrowed.extend(labels.iter().map(OwnedPlacedLabel::as_placed));
}

/// One candidate before it is shaped: text plus where it wants to be.
///
/// The text is borrowed from the tile whenever the property is a string, which
/// is every real basemap label; the candidate lives no longer than the
/// [`LabelPlacer::place_tile`] call that built it.
#[derive(Debug, Clone)]
struct Candidate<'a> {
    text: Cow<'a, str>,
    anchor_px: [f32; 2],
    priority: f64,
}

/// Greedy label placement for one frame.
///
/// Feed it every visible tile with [`LabelPlacer::place_tile`]; the accepted
/// boxes accumulate across tiles, so a label near a seam competes with its
/// neighbour's labels. [`LabelPlacer::finish`] hands back what survived.
///
/// # Ordering, and what "greedy" costs
///
/// Within one layer of one tile, candidates are sorted descending by
/// [`LabelSpec::rank_property`] where the spec names one and by
/// [`LabelAnchor::priority`] otherwise (a stable sort, so equal priorities —
/// every unranked point layer — keep the tile's feature order). Layers are
/// processed in the order the tile stores them, and tiles in the order the
/// caller supplies them: first come, first served. A label is placed if its
/// padded box touches the viewport and misses every box accepted so far;
/// otherwise it is dropped for this frame. There is no backtracking, no second
/// position to try, and no density scoring — deliberately, per `TODO.md` §5.3.
///
/// The overlap test is a linear scan over accepted boxes. At the few hundred
/// labels a screenful of tiles produces that is cheaper than maintaining a
/// spatial index — and the accepted set is bounded by what fits the viewport,
/// not by how many features the caller offered, because a candidate whose
/// anchor cannot reach the viewport is dropped before it is even shaped. A
/// uniform grid over the viewport is the answer if that stops holding.
#[derive(Debug, Clone)]
pub struct LabelPlacer {
    viewport_px: [f32; 2],
    accepted: Vec<LabelBox>,
    placed: Vec<OwnedPlacedLabel>,
    considered: usize,
    generation: Option<u32>,
    stale: bool,
}

impl LabelPlacer {
    /// Starts a placement pass for a viewport of `viewport_px` physical pixels.
    ///
    /// A non-positive or non-finite viewport is legal but places nothing:
    /// [`LabelPlacer::place_tile`] returns before it looks at a layer.
    #[must_use]
    pub fn new(viewport_px: [f32; 2]) -> Self {
        Self {
            viewport_px,
            accepted: Vec::new(),
            placed: Vec::new(),
            considered: 0,
            generation: None,
            stale: false,
        }
    }

    /// The viewport the pass was started for.
    #[must_use]
    pub fn viewport_px(&self) -> [f32; 2] {
        self.viewport_px
    }

    /// Empties the placer for a new frame at `viewport_px`, keeping the
    /// capacity it has grown to.
    ///
    /// [`LabelPlacer::new`] per frame is correct and costs two allocations that
    /// a viewport's worth of labels immediately re-grows; a caller that keeps
    /// one placer for the session calls this instead. The result is the same
    /// pass either way — nothing here carries over between frames, which is
    /// also why placement is not yet stable across them.
    pub fn reset(&mut self, viewport_px: [f32; 2]) {
        self.viewport_px = viewport_px;
        self.accepted.clear();
        self.placed.clear();
        self.considered = 0;
        self.generation = None;
        self.stale = false;
    }

    /// Number of labels accepted so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.placed.len()
    }

    /// Whether nothing has been placed yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.placed.is_empty()
    }

    /// Number of candidates that reached the collision test — accepted plus
    /// rejected. The difference from [`LabelPlacer::len`] is how much the
    /// frame's label density is costing.
    ///
    /// Counted after shaping, so it excludes what never got that far: features
    /// with no text, anchors in a tile's buffer zone, anchors too far
    /// off-screen to reach the viewport, and labels that shape to no glyphs.
    #[must_use]
    pub fn considered(&self) -> usize {
        self.considered
    }

    /// The labels accepted so far.
    #[must_use]
    pub fn labels(&self) -> &[OwnedPlacedLabel] {
        &self.placed
    }

    /// The padded boxes accepted so far, in acceptance order.
    #[must_use]
    pub fn boxes(&self) -> &[LabelBox] {
        &self.accepted
    }

    /// The [`LabelEngine::generation`] the first shaped label belonged to, or
    /// [`None`] if nothing has been shaped yet.
    #[must_use]
    pub fn generation(&self) -> Option<u32> {
        self.generation
    }

    /// Whether the atlas was repacked mid-pass, invalidating the labels shaped
    /// before it. See the module docs: re-run the pass when this is `true`.
    #[must_use]
    pub fn is_stale(&self) -> bool {
        self.stale
    }

    /// Places the labels of one decoded tile.
    ///
    /// `placement` is the tile's screen rectangle, straight from
    /// [`crate::vector::VectorLayerRenderer::begin_frame`]. Layers with no spec
    /// from `labels`, layers whose spec has an unusable size, features with no
    /// text, anchors outside the tile square and anchors too far off-screen for
    /// any box to reach the viewport are all skipped silently — a style that
    /// asks for the impossible loses labels, it does not fail the frame.
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Text`] from [`LabelEngine::shape`].
    pub fn place_tile(
        &mut self,
        engine: &mut LabelEngine,
        tile: &VectorTile,
        placement: &TilePlacement,
        labels: &dyn LabelResolver,
    ) -> Result<(), RenderError> {
        if !placement.size.is_finite()
            || placement.size <= 0.0
            || !placement.x.is_finite()
            || !placement.y.is_finite()
        {
            return Ok(());
        }
        // A viewport with no area shows nothing, and the intersection test would
        // otherwise accept a box straddling a degenerate rectangle.
        if !self
            .viewport_px
            .iter()
            .all(|edge| edge.is_finite() && *edge > 0.0)
        {
            return Ok(());
        }

        // Reused across layers: the borrowed text ties every candidate to the
        // same tile, so one allocation serves the whole call.
        let mut candidates: Vec<Candidate<'_>> = Vec::new();
        for layer in &tile.layers {
            let Some(spec) = labels.label_spec(&layer.name) else {
                continue;
            };
            let spec = &*spec;
            if !spec.has_usable_size() || layer.extent == 0 {
                continue;
            }
            let scale = placement.size / layer.extent as f32;
            let extent = f64::from(layer.extent);
            let padding = LABEL_PADDING_PX + spec.halo_padding_px();

            candidates.clear();
            for feature in &layer.features {
                let Some(text) = feature
                    .properties
                    .iter()
                    .find(|(key, _)| key == &spec.text_property)
                    .and_then(|(_, value)| label_text_cow(value))
                else {
                    continue;
                };
                let Some(anchor) = feature_anchor(&feature.geometry) else {
                    continue;
                };
                // Buffer-zone spill belongs to the neighbouring tile.
                if anchor.position[0] < 0.0
                    || anchor.position[0] > extent
                    || anchor.position[1] < 0.0
                    || anchor.position[1] > extent
                {
                    continue;
                }
                let anchor_px = [
                    placement.x + anchor.position[0] as f32 * scale,
                    placement.y + anchor.position[1] as f32 * scale,
                ];
                if !anchor_px[0].is_finite() || !anchor_px[1].is_finite() {
                    continue;
                }
                // Shaping is the expensive half of the pass and a tile can cover
                // far more ground than the window — one synthetic tile holds a
                // whole local dataset — so an anchor no box could bring into the
                // viewport is dropped *before* the shaper sees it.
                if !self.anchor_can_reach(anchor_px, &text, spec, padding) {
                    continue;
                }
                let priority = spec
                    .rank_property
                    .as_deref()
                    .and_then(|rank| {
                        feature
                            .properties
                            .iter()
                            .find(|(key, _)| key.as_str() == rank)
                            .and_then(|(_, value)| rank_value(value))
                    })
                    .unwrap_or(anchor.priority);
                candidates.push(Candidate {
                    text,
                    anchor_px,
                    priority,
                });
            }

            // Stable: equal priorities (every unranked point layer) keep feature
            // order.
            candidates.sort_by(|left, right| right.priority.total_cmp(&left.priority));

            for candidate in &candidates {
                let shaped = engine.shape_oriented(
                    &candidate.text,
                    spec.size_px,
                    spec.weight,
                    spec.orientation,
                )?;
                match self.generation {
                    None => self.generation = Some(shaped.generation()),
                    Some(seen) if seen != shaped.generation() => self.stale = true,
                    Some(_) => {}
                }
                // A whitespace-only label has no glyphs and a zero-sized box;
                // accepting it would reserve screen space for nothing.
                if shaped.is_empty() {
                    continue;
                }
                self.considered += 1;

                let size_px = shaped.size_px();
                let [width, height] = size_px;
                let shift = spec.anchor.origin_shift(size_px);
                let origin_px = [
                    (candidate.anchor_px[0] + shift[0] + spec.offset_px[0]).round(),
                    (candidate.anchor_px[1] + shift[1] + spec.offset_px[1]).round(),
                ];
                let collision_box = LabelBox {
                    min_x: origin_px[0] - padding,
                    min_y: origin_px[1] - padding,
                    max_x: origin_px[0] + width + padding,
                    max_y: origin_px[1] + height + padding,
                };
                if !collision_box.is_finite()
                    || !collision_box.is_inside_buffered(self.viewport_px, VIEWPORT_BUFFER_PX)
                {
                    continue;
                }
                if self
                    .accepted
                    .iter()
                    .any(|accepted| accepted.intersects(&collision_box))
                {
                    continue;
                }
                self.accepted.push(collision_box);
                self.placed.push(OwnedPlacedLabel {
                    shaped,
                    origin_px,
                    color: spec.color,
                    halo: spec.halo,
                    collision_box,
                });
            }
        }
        Ok(())
    }

    /// Whether a label anchored at `anchor_px` could reach the viewport at all.
    ///
    /// Over-estimates the label's extent on both axes from
    /// [`MAX_GLYPH_ADVANCE_EM`] and the text's byte length, then asks whether a
    /// box of that size, hung off the anchor at any [`LabelAnchorPoint`] and
    /// shifted by [`LabelSpec::offset_px`], could still intersect the buffered
    /// viewport. Because the bound is never smaller than the box shaping
    /// produces, a candidate this rejects could not have been placed — the cull
    /// moves no label, it only keeps the shaper away from the ones off-screen.
    fn anchor_can_reach(
        &self,
        anchor_px: [f32; 2],
        text: &str,
        spec: &LabelSpec,
        padding: f32,
    ) -> bool {
        let span = text.len() as f32 * spec.size_px * MAX_GLYPH_ADVANCE_EM;
        let reach_x = span + padding + spec.offset_px[0].abs() + VIEWPORT_BUFFER_PX;
        let reach_y = span + padding + spec.offset_px[1].abs() + VIEWPORT_BUFFER_PX;
        anchor_px[0] >= -reach_x
            && anchor_px[0] <= self.viewport_px[0] + reach_x
            && anchor_px[1] >= -reach_y
            && anchor_px[1] <= self.viewport_px[1] + reach_y
    }

    /// Ends the pass and returns the accepted labels, in acceptance order.
    #[must_use]
    pub fn finish(self) -> Vec<OwnedPlacedLabel> {
        self.placed
    }
}

#[cfg(test)]
mod tests;
