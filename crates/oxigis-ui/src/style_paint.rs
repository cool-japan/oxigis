//! The style seam: `oxigis-core`'s [`LayerStyle`] → `oxigis-render`'s
//! [`LayerPaint`].
//!
//! `oxigis-render` deliberately does not depend on `oxigis-core` (see its crate
//! docs), so the two style models are structurally similar but nominally
//! unrelated. This module is the one place that names both, exactly as
//! [`crate::map_gpu`] is the one place that names `egui_wgpu` and the renderer.
//!
//! # What maps onto what
//!
//! | [`LayerStyle`] | [`LayerPaint`] | Notes |
//! |---|---|---|
//! | [`LayerStyle::Fill`] | [`oxigis_render::FillPaint`] | `outline_color` becomes a **second** paint — see below |
//! | [`LayerStyle::Line`] | [`oxigis_render::LinePaint`] | `width` is in physical pixels on both sides |
//! | [`LayerStyle::Circle`] | [`oxigis_render::CirclePaint`] | stroke maps natively, `None` stroke stays `None` |
//! | [`LayerStyle::Symbol`] | [`oxigis_render::LabelSpec`] | via [`label_spec`], **not** [`layer_paint`]: a symbol rule draws text, never geometry |
//!
//! # Two products from one rule list
//!
//! A `(source layer, style)` list therefore yields *two* independent things:
//! [`PaintProgram::from_rules`] builds the tessellation passes, and
//! [`label_table`] builds the [`oxigis_render::LabelTable`] the placement pass
//! resolves against. They partition the rules — [`layer_paint`] returns [`None`]
//! for [`LayerStyle::Symbol`], [`label_spec`] returns [`None`] for everything
//! else — so a layer can carry both a circle rule and a symbol rule without the
//! first-match-wins lookups on either side interfering.
//!
//! # Logical versus physical pixels
//!
//! [`oxigis_core::SymbolStyle::text_size`] and
//! [`oxigis_core::SymbolStyle::halo_width`] are documented as *logical* pixels,
//! while [`oxigis_render::LabelSpec::size_px`] is *physical*. They are mapped
//! straight through with no `pixels_per_point` factor, exactly as
//! [`oxigis_core::LineStyle::width`] already maps onto
//! [`oxigis_render::LinePaint::width_px`] and as [`FILL_OUTLINE_WIDTH_PX`] is
//! declared: Phase 0 treats the core style's pixels as physical throughout, and
//! resolving that properly is a single change to this module when a DPI-aware
//! style model lands.
//!
//! # Why fill outlines need a second pass
//!
//! [`oxigis_render::FillPaint`] has no outline field, and a
//! [`oxigis_render::PaintTable`] resolves **one** paint per source-layer name
//! (first match wins), so a filled polygon layer that also wants its rings
//! stroked cannot be expressed in a single table. [`PaintProgram`] therefore
//! holds a *list* of tables and tessellates the same tile once per table,
//! concatenating the meshes. With depth testing off that is a painter's
//! algorithm, so pass 1 (outlines) lands on top of pass 0 (fills), which is the
//! order a stroked polygon needs.
//!
//! A program with a single pass — the common case, and what a table of pure
//! line/circle rules produces — costs exactly one tessellation, so the extra
//! machinery is free unless a fill actually asks for an outline.
//!
//! # Thematic renderers (v1.6) need nothing here
//!
//! A categorized or graduated layer is drawn by exactly this machinery,
//! unextended: [`crate::local_vector`] partitions its synthetic tile into one
//! tile LAYER per (geometry family, style class) and compiles one rule per
//! bucket, so a class is a name in the rule list and its style is an ordinary
//! [`LayerStyle`]. `PaintProgram` never learns what a class is.
//!
//! That is deliberate, and it is what makes the outline pass keep working: a
//! classified layer whose classes are outlined fills produces one fill rule and
//! one outline rule PER CLASS, and the existing two-pass split still puts every
//! outline over every fill — because the split is per TABLE, not per rule. A
//! per-feature paint resolver (the roadmap item behind finding 49) would
//! change this module; a per-class tile partition does not.

use oxigis_core::{Color, LabelOrientation, LabelWeight, LayerStyle};
use oxigis_render::label::{
    LabelOrientation as RenderLabelOrientation, LabelWeight as RenderLabelWeight,
};
use oxigis_render::{
    CirclePaint, FillPaint, LabelHalo, LabelSpec, LabelTable, LayerPaint, LinePaint, PaintTable,
    RenderError, Rgba8, TessParams, VectorMesh, VectorTile, tessellate_tile,
};

/// Maps the core style model's weight onto the renderer's own.
///
/// Two types on purpose: `oxigis-render` deliberately does not depend on
/// `oxigis-core` (blueprint §3), so the crate boundary is crossed exactly
/// here, where a style is already being translated into a render spec.
fn render_weight(weight: LabelWeight) -> RenderLabelWeight {
    match weight {
        LabelWeight::Regular => RenderLabelWeight::Regular,
        LabelWeight::Bold => RenderLabelWeight::Bold,
    }
}

/// Maps the core style model's orientation onto the renderer's own — the twin
/// of [`render_weight`], and the ONE place the two mirrors meet.
fn render_orientation(orientation: LabelOrientation) -> RenderLabelOrientation {
    match orientation {
        LabelOrientation::Horizontal => RenderLabelOrientation::Horizontal,
        LabelOrientation::Vertical => RenderLabelOrientation::Vertical,
    }
}

/// Stroke width, in physical pixels, given to the line paint synthesized from
/// [`oxigis_core::FillStyle::outline_color`].
///
/// The core fill style has no outline *width* (the MapLibre `fill-outline-color`
/// property it mirrors is likewise always hairline), so one pixel is the only
/// faithful choice.
pub const FILL_OUTLINE_WIDTH_PX: f32 = 1.0;

/// Converts a core [`Color`] into the renderer's [`Rgba8`].
///
/// Both are straight (non-premultiplied) sRGB bytes, so this is a field-by-field
/// move rather than a conversion.
#[must_use]
pub fn rgba8(color: Color) -> Rgba8 {
    Rgba8::new(color.r, color.g, color.b, color.a)
}

/// Converts a core [`Color`] into the raw straight-sRGB bytes the label types
/// carry.
///
/// [`oxigis_render::LabelSpec::color`] and [`oxigis_render::LabelHalo::color`]
/// are plain `[u8; 4]` rather than [`Rgba8`], because the label vertex format
/// is written by the pipeline itself and never resolves an opacity.
#[must_use]
pub fn rgba_bytes(color: Color) -> [u8; 4] {
    [color.r, color.g, color.b, color.a]
}

/// The label spec for `style`, or [`None`] if the style draws no text.
///
/// Only [`LayerStyle::Symbol`] produces one, and only when it names a
/// `text_field`: a symbol style without one is MapLibre's way of spelling
/// "icon, no label", which Phase 0 does not draw at all.
///
/// The halo is dropped unless the style carries **both** a colour and a
/// strictly positive width, since either alone describes nothing the
/// [`oxigis_render::LabelPipeline`] can draw — and a zero-width halo would
/// still cost eight offset copies per glyph.
///
/// Sizes map through unchanged; see the [module docs][self] on logical versus
/// physical pixels.
#[must_use]
pub fn label_spec(style: &LayerStyle) -> Option<LabelSpec> {
    let LayerStyle::Symbol(symbol) = style else {
        return None;
    };
    let text_field = symbol.text_field.as_ref()?;
    let mut spec = LabelSpec::new(text_field.clone())
        .with_size_px(symbol.text_size())
        .with_color(rgba_bytes(symbol.text_color))
        .with_weight(render_weight(symbol.weight()))
        .with_orientation(render_orientation(symbol.orientation()));
    if let Some(halo_color) = symbol.halo_color
        && symbol.halo_width() > 0.0
    {
        spec = spec.with_halo(LabelHalo::new(rgba_bytes(halo_color), symbol.halo_width()));
    }
    Some(spec)
}

/// Builds the [`LabelTable`] an ordered `(source layer name, style)` rule list
/// describes.
///
/// The label counterpart of [`PaintProgram::from_rules`], and deliberately a
/// free function rather than a field of it: the two products have different
/// lifetimes in the frame (a program is consumed off the render thread when a
/// tile is tessellated, a table is consulted on the render thread every frame)
/// and nothing is shared between them but the input list.
///
/// Rules whose style is not a labelling [`LayerStyle::Symbol`] are skipped, so
/// the table only ever holds entries [`label_spec`] accepted. Lookup is
/// first-match-wins, exactly as in [`oxigis_render::PaintTable`].
#[must_use]
pub fn label_table<'a, I>(rules: I) -> LabelTable
where
    I: IntoIterator<Item = (&'a str, &'a LayerStyle)>,
{
    let mut table = LabelTable::new();
    for (name, style) in rules {
        if let Some(spec) = label_spec(style) {
            table.push(name, spec);
        }
    }
    table
}

/// The main paint for `style`, or [`None`] if the style draws no geometry.
///
/// [`LayerStyle::Symbol`] returns [`None`] and always will: [`LayerPaint`] has
/// no symbol variant, because a label is not a mesh. Symbol rules are picked up
/// by [`label_spec`] instead.
#[must_use]
pub fn layer_paint(style: &LayerStyle) -> Option<LayerPaint> {
    match style {
        LayerStyle::Fill(fill) => Some(LayerPaint::Fill(
            FillPaint::new(rgba8(fill.color)).with_opacity(fill.opacity()),
        )),
        LayerStyle::Line(line) => Some(LayerPaint::Line(
            LinePaint::new(rgba8(line.color))
                .with_width_px(line.width())
                .with_opacity(line.opacity()),
        )),
        LayerStyle::Circle(circle) => {
            let mut paint = CirclePaint::new(rgba8(circle.color))
                .with_radius_px(circle.radius())
                .with_opacity(circle.opacity());
            if let Some(stroke) = circle.stroke_color {
                paint = paint.with_stroke(rgba8(stroke), circle.stroke_width());
            }
            Some(LayerPaint::Circle(paint))
        }
        LayerStyle::Symbol(_) => None,
    }
}

/// The extra outline paint `style` asks for, if any.
///
/// Only [`LayerStyle::Fill`] with an `outline_color` produces one — a circle's
/// stroke is already part of [`oxigis_render::CirclePaint`], and a line has no
/// second colour. See the [module docs][self] for why this cannot be folded
/// into [`layer_paint`]'s return value.
#[must_use]
pub fn outline_paint(style: &LayerStyle) -> Option<LinePaint> {
    match style {
        LayerStyle::Fill(fill) => fill.outline_color.map(|color| {
            LinePaint::new(rgba8(color))
                .with_width_px(FILL_OUTLINE_WIDTH_PX)
                .with_opacity(fill.opacity())
        }),
        _ => None,
    }
}

/// An ordered set of [`PaintTable`] passes, tessellated back-to-front into one
/// mesh.
///
/// Built from a list of `(source layer name, style)` rules with
/// [`PaintProgram::from_rules`]; see the [module docs][self] for why more than
/// one pass is ever needed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PaintProgram {
    /// Tables in draw order; later passes paint over earlier ones.
    passes: Vec<PaintTable>,
}

impl PaintProgram {
    /// A program that draws nothing.
    #[must_use]
    pub const fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Builds a program from ordered `(source layer name, style)` rules.
    ///
    /// Rules whose style maps to no paint (currently
    /// [`LayerStyle::Symbol`]) are skipped. Fill rules carrying an outline
    /// colour additionally contribute a hairline stroke to a second pass, which
    /// is created only if at least one rule needs it.
    #[must_use]
    pub fn from_rules<'a, I>(rules: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a LayerStyle)>,
    {
        let mut fills = PaintTable::new();
        let mut outlines = PaintTable::new();
        for (name, style) in rules {
            if let Some(paint) = layer_paint(style) {
                fills.push(name, paint);
            }
            if let Some(outline) = outline_paint(style) {
                outlines.push(name, outline);
            }
        }
        let mut passes = Vec::new();
        if !fills.is_empty() {
            passes.push(fills);
        }
        if !outlines.is_empty() {
            passes.push(outlines);
        }
        Self { passes }
    }

    /// Appends a pass, which will paint over every pass already present.
    pub fn push_pass(&mut self, table: PaintTable) {
        self.passes.push(table);
    }

    /// The passes, in draw order.
    #[must_use]
    pub fn passes(&self) -> &[PaintTable] {
        &self.passes
    }

    /// Whether the program would draw nothing at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.passes.iter().all(PaintTable::is_empty)
    }

    /// Tessellates `tile` once per pass and concatenates the meshes.
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Tessellation`] from
    /// [`oxigis_render::tessellate_tile`], and reports the same error if the
    /// combined mesh would overflow the `u32` index space.
    pub fn tessellate(
        &self,
        tile: &VectorTile,
        params: &TessParams,
    ) -> Result<VectorMesh, RenderError> {
        let mut combined = VectorMesh::new();
        for table in &self.passes {
            let mesh = tessellate_tile(tile, table, params)?;
            append_mesh(&mut combined, mesh)?;
        }
        Ok(combined)
    }
}

/// Appends `src` onto `dst`, shifting `src`'s indices past `dst`'s vertices.
///
/// # Errors
///
/// Returns [`RenderError::Tessellation`] if the combined vertex count would
/// leave the `u32` index space the GPU buffers use.
fn append_mesh(dst: &mut VectorMesh, src: VectorMesh) -> Result<(), RenderError> {
    if src.vertices.is_empty() {
        return Ok(());
    }
    if dst.vertices.is_empty() {
        *dst = src;
        return Ok(());
    }
    let Ok(offset) = u32::try_from(dst.vertices.len()) else {
        return Err(RenderError::Tessellation(
            "combined vector mesh exceeds the u32 index space".to_owned(),
        ));
    };
    let total = dst
        .vertices
        .len()
        .checked_add(src.vertices.len())
        .filter(|total| u32::try_from(*total).is_ok());
    if total.is_none() {
        return Err(RenderError::Tessellation(
            "combined vector mesh exceeds the u32 index space".to_owned(),
        ));
    }
    dst.vertices.extend(src.vertices);
    dst.indices
        .extend(src.indices.into_iter().map(|index| index + offset));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        FILL_OUTLINE_WIDTH_PX, PaintProgram, label_spec, label_table, layer_paint, outline_paint,
        rgba8,
    };
    use oxigis_core::{CircleStyle, Color, FillStyle, LayerStyle, LineStyle, SymbolStyle};
    use oxigis_render::{
        LabelResolver as _, LayerPaint, PaintResolver as _, Rgba8, TessParams, VectorTile,
    };

    #[test]
    fn colors_move_channel_for_channel() {
        let color = Color::from_rgba(1, 2, 3, 4);
        assert_eq!(rgba8(color), Rgba8::new(1, 2, 3, 4));
    }

    #[test]
    fn a_fill_style_maps_color_and_opacity() {
        let mut fill = FillStyle::new(Color::from_rgb(0x10, 0x20, 0x30));
        fill.set_opacity(0.5);
        match layer_paint(&LayerStyle::Fill(fill)) {
            Some(LayerPaint::Fill(paint)) => {
                assert_eq!(paint.color, Rgba8::opaque(0x10, 0x20, 0x30));
                assert_eq!(paint.opacity, 0.5);
                // Opacity is applied to alpha at resolve time, not baked here.
                assert_eq!(paint.resolved_color().a, 128);
            }
            other => panic!("expected a fill paint, got {other:?}"),
        }
    }

    #[test]
    fn a_fill_without_an_outline_has_no_second_paint() {
        let fill = FillStyle::new(Color::WHITE);
        assert!(fill.outline_color.is_none());
        assert!(outline_paint(&LayerStyle::Fill(fill)).is_none());
    }

    #[test]
    fn a_fill_outline_becomes_a_hairline_line_paint() {
        let mut fill = FillStyle::new(Color::WHITE);
        fill.outline_color = Some(Color::from_rgb(9, 8, 7));
        fill.set_opacity(0.25);
        let outline = outline_paint(&LayerStyle::Fill(fill)).expect("an outline paint");
        assert_eq!(outline.color, Rgba8::opaque(9, 8, 7));
        assert_eq!(outline.width_px, FILL_OUTLINE_WIDTH_PX);
        assert_eq!(outline.opacity, 0.25);
    }

    #[test]
    fn a_line_style_maps_color_width_and_opacity() {
        let mut line = LineStyle::new(Color::BLACK, 2.5);
        line.set_opacity(0.75);
        match layer_paint(&LayerStyle::Line(line)) {
            Some(LayerPaint::Line(paint)) => {
                assert_eq!(paint.color, Rgba8::BLACK);
                assert_eq!(paint.width_px, 2.5);
                assert_eq!(paint.opacity, 0.75);
            }
            other => panic!("expected a line paint, got {other:?}"),
        }
        assert!(outline_paint(&LayerStyle::Line(line)).is_none());
    }

    #[test]
    fn a_circle_style_maps_radius_and_an_absent_stroke() {
        let circle = CircleStyle::new(6.0, Color::from_rgb(1, 2, 3));
        match layer_paint(&LayerStyle::Circle(circle)) {
            Some(LayerPaint::Circle(paint)) => {
                assert_eq!(paint.radius_px, 6.0);
                assert_eq!(paint.color, Rgba8::opaque(1, 2, 3));
                assert_eq!(paint.stroke_color, None);
                assert_eq!(paint.stroke_width_px, 0.0);
                assert_eq!(paint.opacity, 1.0);
            }
            other => panic!("expected a circle paint, got {other:?}"),
        }
    }

    #[test]
    fn a_circle_stroke_maps_natively_without_a_second_pass() {
        let mut circle = CircleStyle::new(4.0, Color::WHITE);
        circle.stroke_color = Some(Color::BLACK);
        circle.set_stroke_width(1.5);
        circle.set_opacity(0.5);
        match layer_paint(&LayerStyle::Circle(circle)) {
            Some(LayerPaint::Circle(paint)) => {
                assert_eq!(paint.stroke_color, Some(Rgba8::BLACK));
                assert_eq!(paint.stroke_width_px, 1.5);
                assert_eq!(paint.resolved_stroke_color().map(|c| c.a), Some(128));
            }
            other => panic!("expected a circle paint, got {other:?}"),
        }
        assert!(outline_paint(&LayerStyle::Circle(circle)).is_none());
    }

    #[test]
    fn a_symbol_style_draws_no_geometry() {
        // No symbol variant exists in `LayerPaint`: labels are a separate pass.
        let symbol = LayerStyle::Symbol(SymbolStyle::new("name"));
        assert!(layer_paint(&symbol).is_none());
        assert!(outline_paint(&symbol).is_none());
    }

    #[test]
    fn a_symbol_style_maps_text_size_colour_and_halo() {
        let mut symbol = SymbolStyle::new("NAME");
        symbol.text_color = Color::from_rgb(0x21, 0x25, 0x2b);
        symbol.set_text_size(13.5);
        symbol.halo_color = Some(Color::WHITE);
        symbol.set_halo_width(1.5);
        let spec = label_spec(&LayerStyle::Symbol(symbol)).expect("a label spec");
        assert_eq!(spec.text_property, "NAME");
        assert_eq!(spec.size_px, 13.5);
        assert_eq!(spec.color, [0x21, 0x25, 0x2b, 0xff]);
        let halo = spec.halo.expect("a halo");
        assert_eq!(halo.color, [255, 255, 255, 255]);
        assert_eq!(halo.width_px, 1.5);
        assert_eq!(spec.halo_padding_px(), 1.5);
    }

    #[test]
    fn a_symbol_style_without_a_text_field_labels_nothing() {
        let symbol = SymbolStyle::default();
        assert!(symbol.text_field.is_none());
        assert!(label_spec(&LayerStyle::Symbol(symbol)).is_none());
    }

    #[test]
    fn a_halo_needs_both_a_colour_and_a_positive_width() {
        let mut no_width = SymbolStyle::new("name");
        no_width.set_halo_width(0.0);
        assert!(
            label_spec(&LayerStyle::Symbol(no_width))
                .expect("a spec")
                .halo
                .is_none()
        );

        let mut no_color = SymbolStyle::new("name");
        no_color.halo_color = None;
        assert!(
            label_spec(&LayerStyle::Symbol(no_color))
                .expect("a spec")
                .halo
                .is_none()
        );

        // A negative width is clamped to zero by the core style itself.
        let mut negative = SymbolStyle::new("name");
        negative.set_halo_width(-4.0);
        assert_eq!(negative.halo_width(), 0.0);
    }

    #[test]
    fn only_symbol_rules_reach_the_label_table() {
        let rules = [
            (
                "countries".to_string(),
                LayerStyle::Fill(FillStyle::new(Color::WHITE)),
            ),
            (
                "centroids".to_string(),
                LayerStyle::Circle(CircleStyle::new(2.0, Color::BLACK)),
            ),
            (
                "centroids".to_string(),
                LayerStyle::Symbol(SymbolStyle::new("NAME")),
            ),
            (
                "geolines".to_string(),
                LayerStyle::Line(LineStyle::new(Color::BLACK, 1.0)),
            ),
        ];
        let table = label_table(rules.iter().map(|(name, style)| (name.as_str(), style)));
        assert_eq!(table.len(), 1);
        assert_eq!(
            table.label_for("centroids").map(|spec| spec.text_property),
            Some("NAME".to_string())
        );
        assert!(table.label_for("countries").is_none());
        assert!(table.label_for("geolines").is_none());

        // The very same list still tessellates the non-symbol rules, and the
        // symbol rule contributes nothing to the geometry passes.
        let program =
            PaintProgram::from_rules(rules.iter().map(|(name, style)| (name.as_str(), style)));
        assert_eq!(program.passes().len(), 1);
        assert_eq!(program.passes()[0].len(), 3);
    }

    #[test]
    fn an_empty_rule_list_labels_nothing() {
        let table = label_table([]);
        assert!(table.is_empty());
        assert!(table.label_for("anything").is_none());
    }

    #[test]
    fn a_program_of_plain_rules_has_exactly_one_pass() {
        let rules = [
            (
                "water".to_string(),
                LayerStyle::Fill(FillStyle::new(Color::BLACK)),
            ),
            (
                "roads".to_string(),
                LayerStyle::Line(LineStyle::new(Color::WHITE, 1.0)),
            ),
        ];
        let program =
            PaintProgram::from_rules(rules.iter().map(|(name, style)| (name.as_str(), style)));
        assert_eq!(program.passes().len(), 1);
        assert!(!program.is_empty());
        let pass = &program.passes()[0];
        assert_eq!(pass.len(), 2);
        assert!(matches!(pass.paint_for("water"), Some(LayerPaint::Fill(_))));
        assert!(matches!(pass.paint_for("roads"), Some(LayerPaint::Line(_))));
    }

    #[test]
    fn an_outlined_fill_adds_a_second_pass_that_paints_last() {
        let mut fill = FillStyle::new(Color::WHITE);
        fill.outline_color = Some(Color::BLACK);
        let rules = [("countries".to_string(), LayerStyle::Fill(fill))];
        let program =
            PaintProgram::from_rules(rules.iter().map(|(name, style)| (name.as_str(), style)));
        assert_eq!(program.passes().len(), 2);
        assert!(matches!(
            program.passes()[0].paint_for("countries"),
            Some(LayerPaint::Fill(_))
        ));
        assert!(matches!(
            program.passes()[1].paint_for("countries"),
            Some(LayerPaint::Line(_))
        ));
    }

    #[test]
    fn a_program_of_symbol_rules_only_draws_nothing() {
        let rules = [(
            "labels".to_string(),
            LayerStyle::Symbol(SymbolStyle::new("name")),
        )];
        let program =
            PaintProgram::from_rules(rules.iter().map(|(name, style)| (name.as_str(), style)));
        assert!(program.passes().is_empty());
        assert!(program.is_empty());
        assert!(PaintProgram::new().is_empty());
    }

    #[test]
    fn tessellating_an_empty_tile_yields_an_empty_mesh() {
        let program =
            PaintProgram::from_rules([("water", &LayerStyle::Fill(FillStyle::new(Color::BLACK)))]);
        let tile = VectorTile { layers: Vec::new() };
        let mesh = program
            .tessellate(&tile, &TessParams::default())
            .expect("an empty tile must tessellate");
        assert!(mesh.is_empty());
    }

    #[test]
    fn a_symbol_style_carries_its_orientation_into_the_render_spec() {
        use oxigis_core::LabelOrientation;
        use oxigis_render::label::LabelOrientation as RenderLabelOrientation;

        let plain = SymbolStyle::new("name");
        assert_eq!(
            label_spec(&LayerStyle::Symbol(plain))
                .expect("a spec")
                .orientation,
            RenderLabelOrientation::Horizontal,
            "the default crosses the crate boundary unchanged",
        );

        let mut vertical = SymbolStyle::new("name");
        vertical.set_orientation(LabelOrientation::Vertical);
        vertical.set_weight(oxigis_core::LabelWeight::Bold);
        let spec = label_spec(&LayerStyle::Symbol(vertical)).expect("a spec");
        assert_eq!(spec.orientation, RenderLabelOrientation::Vertical);
        assert_eq!(
            spec.weight,
            oxigis_render::label::LabelWeight::Bold,
            "and the two mirrors are independent",
        );

        // A non-symbol style still yields nothing at all.
        assert!(label_spec(&LayerStyle::Fill(FillStyle::new(Color::BLACK))).is_none());
    }

    #[test]
    fn a_classified_layers_rules_keep_the_outline_pass_ordered() {
        // Thematic v1.6: a class is just another rule name, so the two-pass
        // split has to keep putting EVERY outline over EVERY fill — the split
        // is per table, not per rule. A per-class second pass would instead
        // stripe fill/outline/fill/outline and let class 1's fill cover class
        // 0's outline.
        let mut fill = FillStyle::new(Color::WHITE);
        fill.outline_color = Some(Color::BLACK);
        let rules: Vec<(String, LayerStyle)> = (0..4)
            .map(|index| (format!("features:polygon#{index}"), LayerStyle::Fill(fill)))
            .collect();
        let program =
            PaintProgram::from_rules(rules.iter().map(|(name, style)| (name.as_str(), style)));
        assert_eq!(program.passes().len(), 2, "fills, then outlines");
        for (name, _) in &rules {
            assert!(matches!(
                program.passes()[0].paint_for(name),
                Some(LayerPaint::Fill(_))
            ));
            assert!(matches!(
                program.passes()[1].paint_for(name),
                Some(LayerPaint::Line(_))
            ));
        }
    }

    #[test]
    fn the_paint_seam_is_unchanged_by_the_renderer_model() {
        // The floor: a single-symbol rule list compiles to exactly what it
        // compiled to before `Renderer` existed. `PaintProgram` never learns
        // what a class is, so this is a statement about the whole seam.
        use oxigis_core::{GeometryFamily, LayerStyleSet};

        let mut fill = FillStyle::new(Color::from_rgb(80, 140, 200));
        fill.set_opacity(0.35);
        let set = LayerStyleSet::new(LayerStyle::Fill(fill));
        assert!(set.is_single_symbol());
        let resolved = set.style_for_class(GeometryFamily::Polygon, None);
        assert_eq!(&resolved, set.base(), "the fallback IS the base");
        assert_eq!(
            layer_paint(&resolved),
            layer_paint(set.base()),
            "and it maps onto the very same paint",
        );
        assert_eq!(outline_paint(&resolved), outline_paint(set.base()));
    }
}
