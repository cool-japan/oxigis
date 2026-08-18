//! Render-side paint parameters: the small, style-agnostic description of how
//! one vector-tile layer should look.
//!
//! This crate deliberately does not depend on `oxigis-core`, so it cannot see
//! the style model (`LayerStyle`, expressions, zoom stops). The seam is
//! [`PaintResolver`]: the shell evaluates its own style for the current zoom and
//! hands the renderer a plain [`LayerPaint`] per layer name. A layer the
//! resolver has no paint for is skipped entirely — that is how layer visibility
//! and style filtering reach the tessellator.
//!
//! # Colour space and premultiplication
//!
//! [`Rgba8`] holds **straight (non-premultiplied) sRGB** components, the same
//! convention CSS and the Mapbox style spec use. The vertex colours produced by
//! [`crate::vector::tessellate_tile`] keep that convention, and
//! [`crate::vector::VectorPipeline`] pairs it with
//! [`wgpu::BlendState::ALPHA_BLENDING`] (source factor `SrcAlpha`), converting
//! sRGB to linear in the shader when the colour target is an sRGB format.
//!
//! # Out of scope
//!
//! Symbol and text paints (icon/label placement) belong to `TODO.md` §5.3 and
//! are deliberately absent from [`LayerPaint`]: label layout needs glyph
//! metrics and a collision index, neither of which exists yet.

/// Straight (non-premultiplied) sRGB colour with 8 bits per channel.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Rgba8 {
    /// Red component.
    pub r: u8,
    /// Green component.
    pub g: u8,
    /// Blue component.
    pub b: u8,
    /// Alpha component; `255` is fully opaque.
    pub a: u8,
}

impl Rgba8 {
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self::new(0, 0, 0, 0);

    /// Opaque black.
    pub const BLACK: Self = Self::opaque(0, 0, 0);

    /// Opaque white.
    pub const WHITE: Self = Self::opaque(255, 255, 255);

    /// Creates a colour from its four components.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Creates an opaque colour.
    #[must_use]
    pub const fn opaque(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// The components as the array a [`crate::vector::VectorVertex`] stores.
    #[must_use]
    pub const fn to_array(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Returns the colour with its alpha scaled by `opacity`.
    ///
    /// `opacity` is clamped to `0.0..=1.0`; a non-finite value is treated as
    /// `0.0`, so a broken style fades a layer out instead of painting garbage.
    #[must_use]
    pub fn with_opacity(self, opacity: f32) -> Self {
        let factor = if opacity.is_finite() {
            opacity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let alpha = (f32::from(self.a) * factor).round().clamp(0.0, 255.0);
        Self {
            a: alpha as u8,
            ..self
        }
    }

    /// Whether the colour would contribute nothing to the frame.
    #[must_use]
    pub const fn is_invisible(self) -> bool {
        self.a == 0
    }
}

impl Default for Rgba8 {
    fn default() -> Self {
        Self::BLACK
    }
}

/// How to paint the interior of polygons.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillPaint {
    /// Interior colour.
    pub color: Rgba8,
    /// Extra opacity multiplied into [`FillPaint::color`]'s alpha.
    pub opacity: f32,
}

impl FillPaint {
    /// Creates a fully opaque fill paint.
    #[must_use]
    pub const fn new(color: Rgba8) -> Self {
        Self {
            color,
            opacity: 1.0,
        }
    }

    /// Returns the paint with `opacity` applied.
    #[must_use]
    pub const fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// The colour actually written into vertices.
    #[must_use]
    pub fn resolved_color(&self) -> Rgba8 {
        self.color.with_opacity(self.opacity)
    }
}

impl Default for FillPaint {
    fn default() -> Self {
        Self::new(Rgba8::BLACK)
    }
}

/// How to paint line strings (and, when applied to a polygon layer, the
/// polygon's rings).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinePaint {
    /// Stroke colour.
    pub color: Rgba8,
    /// Stroke width in physical pixels, converted to tile units at
    /// tessellation time with [`crate::vector::TessParams`].
    pub width_px: f32,
    /// Extra opacity multiplied into [`LinePaint::color`]'s alpha.
    pub opacity: f32,
}

impl LinePaint {
    /// Default stroke width in physical pixels.
    pub const DEFAULT_WIDTH_PX: f32 = 1.0;

    /// Creates a one-pixel, fully opaque line paint.
    #[must_use]
    pub const fn new(color: Rgba8) -> Self {
        Self {
            color,
            width_px: Self::DEFAULT_WIDTH_PX,
            opacity: 1.0,
        }
    }

    /// Returns the paint with a new width in physical pixels.
    #[must_use]
    pub const fn with_width_px(mut self, width_px: f32) -> Self {
        self.width_px = width_px;
        self
    }

    /// Returns the paint with `opacity` applied.
    #[must_use]
    pub const fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// The colour actually written into vertices.
    #[must_use]
    pub fn resolved_color(&self) -> Rgba8 {
        self.color.with_opacity(self.opacity)
    }
}

impl Default for LinePaint {
    fn default() -> Self {
        Self::new(Rgba8::BLACK)
    }
}

/// How to paint point features: a disc with an optional outline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CirclePaint {
    /// Disc radius in physical pixels.
    pub radius_px: f32,
    /// Disc colour.
    pub color: Rgba8,
    /// Outline colour; [`None`] draws no outline.
    pub stroke_color: Option<Rgba8>,
    /// Outline width in physical pixels, grown outwards from `radius_px`.
    pub stroke_width_px: f32,
    /// Extra opacity multiplied into both colours' alpha.
    pub opacity: f32,
}

impl CirclePaint {
    /// Default disc radius in physical pixels.
    pub const DEFAULT_RADIUS_PX: f32 = 4.0;

    /// Creates an unstroked, fully opaque circle paint.
    #[must_use]
    pub const fn new(color: Rgba8) -> Self {
        Self {
            radius_px: Self::DEFAULT_RADIUS_PX,
            color,
            stroke_color: None,
            stroke_width_px: 0.0,
            opacity: 1.0,
        }
    }

    /// Returns the paint with a new radius in physical pixels.
    #[must_use]
    pub const fn with_radius_px(mut self, radius_px: f32) -> Self {
        self.radius_px = radius_px;
        self
    }

    /// Returns the paint with an outline.
    #[must_use]
    pub const fn with_stroke(mut self, color: Rgba8, width_px: f32) -> Self {
        self.stroke_color = Some(color);
        self.stroke_width_px = width_px;
        self
    }

    /// Returns the paint with `opacity` applied.
    #[must_use]
    pub const fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// The disc colour actually written into vertices.
    #[must_use]
    pub fn resolved_color(&self) -> Rgba8 {
        self.color.with_opacity(self.opacity)
    }

    /// The outline colour actually written into vertices, if any.
    #[must_use]
    pub fn resolved_stroke_color(&self) -> Option<Rgba8> {
        self.stroke_color
            .map(|color| color.with_opacity(self.opacity))
    }
}

impl Default for CirclePaint {
    fn default() -> Self {
        Self::new(Rgba8::BLACK)
    }
}

/// The paint chosen for one vector-tile layer.
///
/// The variant decides which geometries of the layer are drawn:
///
/// | Variant | Draws |
/// |---|---|
/// | [`LayerPaint::Fill`] | polygon interiors |
/// | [`LayerPaint::Line`] | line strings, and polygon rings as closed strokes |
/// | [`LayerPaint::Circle`] | point features |
///
/// Geometry a variant has no rule for is skipped, so a `fill` paint on a
/// point layer simply produces nothing rather than an error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerPaint {
    /// Filled polygons.
    Fill(FillPaint),
    /// Stroked lines.
    Line(LinePaint),
    /// Discs at point features.
    Circle(CirclePaint),
}

impl From<FillPaint> for LayerPaint {
    fn from(paint: FillPaint) -> Self {
        Self::Fill(paint)
    }
}

impl From<LinePaint> for LayerPaint {
    fn from(paint: LinePaint) -> Self {
        Self::Line(paint)
    }
}

impl From<CirclePaint> for LayerPaint {
    fn from(paint: CirclePaint) -> Self {
        Self::Circle(paint)
    }
}

/// Supplies the paint for a vector-tile layer, by name.
///
/// Implemented by the shell against its own style model; [`PaintTable`] is the
/// trivial implementation this crate ships for tests and simple viewers.
/// Returning [`None`] means "do not draw this layer".
pub trait PaintResolver {
    /// The paint for `layer_name`, or [`None`] to skip the layer.
    fn paint_for(&self, layer_name: &str) -> Option<LayerPaint>;
}

/// A [`PaintResolver`] backed by an ordered `(layer name, paint)` list.
///
/// Lookup is a linear scan and the **first** matching entry wins, so a table
/// keeps the order a style file listed its layers in. Tile layers are drawn in
/// the order the *tile* stores them, not the order of this table — see
/// [`crate::vector::tessellate_tile`] for the draw-order contract.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PaintTable {
    entries: Vec<(String, LayerPaint)>,
}

impl PaintTable {
    /// Creates an empty table, which skips every layer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds an entry, keeping insertion order.
    pub fn push(&mut self, layer_name: impl Into<String>, paint: impl Into<LayerPaint>) {
        self.entries.push((layer_name.into(), paint.into()));
    }

    /// Builder form of [`PaintTable::push`].
    #[must_use]
    pub fn with(mut self, layer_name: impl Into<String>, paint: impl Into<LayerPaint>) -> Self {
        self.push(layer_name, paint);
        self
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table skips every layer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries, in insertion order.
    #[must_use]
    pub fn entries(&self) -> &[(String, LayerPaint)] {
        &self.entries
    }

    /// Removes every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl FromIterator<(String, LayerPaint)> for PaintTable {
    fn from_iter<I: IntoIterator<Item = (String, LayerPaint)>>(iter: I) -> Self {
        Self {
            entries: iter.into_iter().collect(),
        }
    }
}

impl PaintResolver for PaintTable {
    fn paint_for(&self, layer_name: &str) -> Option<LayerPaint> {
        self.entries
            .iter()
            .find(|(name, _)| name == layer_name)
            .map(|(_, paint)| *paint)
    }
}

impl<T: PaintResolver + ?Sized> PaintResolver for &T {
    fn paint_for(&self, layer_name: &str) -> Option<LayerPaint> {
        (**self).paint_for(layer_name)
    }
}

#[cfg(test)]
mod tests {
    use super::{CirclePaint, FillPaint, LayerPaint, LinePaint, PaintResolver, PaintTable, Rgba8};

    #[test]
    fn opacity_scales_alpha_and_survives_nonsense() {
        let color = Rgba8::opaque(10, 20, 30);
        assert_eq!(color.with_opacity(1.0), color);
        assert_eq!(color.with_opacity(0.5).a, 128);
        assert_eq!(color.with_opacity(0.0).a, 0);
        assert!(color.with_opacity(0.0).is_invisible());
        assert_eq!(color.with_opacity(4.0), color);
        assert_eq!(color.with_opacity(-1.0).a, 0);
        assert_eq!(color.with_opacity(f32::NAN).a, 0);
        // Only the alpha channel moves.
        assert_eq!(color.with_opacity(0.5).to_array()[..3], [10, 20, 30]);
    }

    #[test]
    fn paints_resolve_their_color() {
        let fill = FillPaint::new(Rgba8::opaque(1, 2, 3)).with_opacity(0.5);
        assert_eq!(fill.resolved_color().a, 128);

        let line = LinePaint::new(Rgba8::WHITE)
            .with_width_px(3.0)
            .with_opacity(0.25);
        assert_eq!(line.width_px, 3.0);
        assert_eq!(line.resolved_color().a, 64);

        let circle = CirclePaint::new(Rgba8::BLACK)
            .with_radius_px(6.0)
            .with_stroke(Rgba8::WHITE, 2.0)
            .with_opacity(0.5);
        assert_eq!(circle.radius_px, 6.0);
        assert_eq!(circle.resolved_color().a, 128);
        assert_eq!(circle.resolved_stroke_color().map(|c| c.a), Some(128));
        assert_eq!(CirclePaint::new(Rgba8::BLACK).resolved_stroke_color(), None);
    }

    #[test]
    fn defaults_are_sane() {
        assert_eq!(Rgba8::default(), Rgba8::BLACK);
        assert_eq!(FillPaint::default().opacity, 1.0);
        assert_eq!(LinePaint::default().width_px, LinePaint::DEFAULT_WIDTH_PX);
        assert_eq!(
            CirclePaint::default().radius_px,
            CirclePaint::DEFAULT_RADIUS_PX
        );
        assert!(Rgba8::TRANSPARENT.is_invisible());
    }

    #[test]
    fn a_table_resolves_by_name_first_match_wins() {
        let mut table = PaintTable::new();
        assert!(table.is_empty());
        table.push("water", FillPaint::new(Rgba8::opaque(0, 0, 255)));
        table.push("water", FillPaint::new(Rgba8::WHITE));
        table.push("roads", LinePaint::new(Rgba8::BLACK));
        assert_eq!(table.len(), 3);
        assert_eq!(table.entries().len(), 3);

        match table.paint_for("water") {
            Some(LayerPaint::Fill(fill)) => assert_eq!(fill.color, Rgba8::opaque(0, 0, 255)),
            other => panic!("expected the first water fill, got {other:?}"),
        }
        assert!(matches!(
            table.paint_for("roads"),
            Some(LayerPaint::Line(_))
        ));
        assert!(table.paint_for("buildings").is_none());

        table.clear();
        assert!(table.paint_for("water").is_none());
    }

    #[test]
    fn references_forward_to_the_resolver() {
        let table = PaintTable::new().with("poi", CirclePaint::new(Rgba8::WHITE));
        let by_ref: &PaintTable = &table;
        assert!(matches!(
            PaintResolver::paint_for(&by_ref, "poi"),
            Some(LayerPaint::Circle(_))
        ));
        let dynamic: &dyn PaintResolver = &table;
        assert!(dynamic.paint_for("poi").is_some());
    }
}
