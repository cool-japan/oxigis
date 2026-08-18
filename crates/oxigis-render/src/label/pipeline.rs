//! wgpu pipeline for label glyphs: one screen-space textured quad per glyph.
//!
//! Built to the same contract as [`crate::vector::VectorPipeline`] — borrowed
//! [`wgpu::Device`]/[`wgpu::Queue`], no owned GPU state the caller cannot see,
//! a `upload_* (&mut)` phase followed by a `draw (&)` phase, `bytemuck` `Pod`
//! vertices, [`wgpu::BlendState::ALPHA_BLENDING`], and a WGSL `const` that
//! folds away the sRGB conversion when the colour target is not an sRGB format.
//!
//! # Coordinates
//!
//! Glyph quads are built in *screen pixels* and converted to NDC on the CPU,
//! while the vector and raster passes hand the GPU a per-tile rectangle and let
//! the vertex shader do the arithmetic. The difference is deliberate: the
//! vertex buffer is rebuilt from scratch every frame anyway (a viewport holds
//! hundreds of labels, not hundreds of thousands of triangles), so a uniform
//! buffer would buy nothing and cost a bind group slot — and doing the
//! transform in `f32` on the CPU is what lets label origins be *rounded* before
//! they are projected.
//!
//! Rounding is the whole crispness story. `fontdue` rasterises on the pixel
//! grid with no subpixel phase, so a glyph sampled at a fractional offset is
//! the same bitmap, blurred. Glyph offsets are already integral inside a
//! [`ShapedLabel`]; this module rounds [`PlacedLabel::origin_px`] as well, and
//! the two together put every glyph texel on exactly one screen pixel. Labels
//! are rasterised at their final on-screen size, so a zoom change is not a
//! scaled draw — it is the caller re-requesting the label at the new size.
//!
//! # Halo
//!
//! [`crate::label::LabelHalo`] is drawn as displaced copies of the glyph quads,
//! in the halo colour, *before* any fill quad. No signed distance field: SDF is
//! Phase 2+ (`TODO.md` §5.3 says explicitly not to gold-plate this).
//!
//! How many copies is a function of the width, because the halo is the whole
//! per-frame cost of the label pass: it is rebuilt and re-uploaded every frame,
//! so each copy is another multiple of the vertex and index bandwidth. Below
//! [`HALO_DIAGONAL_MIN_WIDTH_PX`] only the four axis-aligned offsets are
//! emitted — the difference from all eight is then at most one corner texel at
//! partial coverage, and the frame's label geometry is 5× the glyph count
//! instead of 9× (see [`halo_offsets`]). At the wider widths, where the
//! diagonals are the difference between a ring and a cross, all eight are.
//!
//! Two consequences worth knowing. Overlapping offset copies accumulate alpha,
//! so a *semi-transparent* halo colour looks blotchy; v1 assumes an opaque
//! halo. And every halo quad in the frame is emitted before every fill quad,
//! not per label, so a neighbouring label's halo can never paint over this
//! label's glyphs.
//!
//! The remaining 5×→1× step is a shader change: one quad per glyph expanded by
//! the halo width, with the taps done in the fragment shader. It does *not*
//! need the atlas gutter widened to the halo width, as one might assume —
//! carrying the glyph's UV bounds on the vertex and treating a tap outside them
//! as zero coverage keeps a neighbour's ink out with a one-texel gutter. It is
//! deferred because this crate has no GPU test path to verify a fragment shader
//! against.
//!
//! # Atlas upload
//!
//! [`LabelPipeline::upload_atlas`] writes the [`GlyphAtlas`] whenever it is
//! dirty. The copy is expressed as a **row band**, and the band is the one
//! [`GlyphAtlas::dirty_rows`] reports — the rows this frame's inserts and
//! evictions actually touched, typically a few kilobytes out of a 1–16 MiB
//! buffer. A resize copies every row instead, because the texture is recreated
//! then and has no previous contents to keep.

use bytemuck::{Pod, Zeroable};

use crate::error::RenderError;
use crate::label::atlas::GlyphAtlas;
use crate::label::engine::ShapedLabel;
use crate::vector::pipeline::ScissorRect;

/// Bytes occupied by one [`LabelVertex`] in a vertex buffer.
pub const LABEL_VERTEX_SIZE: u64 = core::mem::size_of::<LabelVertex>() as u64;

/// Vertices the buffer is created with before it has to grow.
const INITIAL_VERTEX_CAPACITY: u32 = 4 * 256;

/// Indices the buffer is created with before it has to grow.
const INITIAL_INDEX_CAPACITY: u32 = 6 * 256;

/// Row alignment `wgpu` requires of a texture upload, in bytes.
const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// The eight directions a halo copy is displaced along, as unit offsets.
///
/// Four axis-aligned **first**, then four diagonal — the order is load-bearing,
/// because [`halo_offsets`] takes the axis-aligned prefix for thin haloes. The
/// diagonals are *not* scaled by `1/√2`, so the halo is a square-ish ring rather
/// than a circular one. At the 1–2 px widths map styles use, the difference is
/// one corner pixel.
pub const HALO_OFFSETS: [[f32; 2]; 8] = [
    [-1.0, 0.0],
    [1.0, 0.0],
    [0.0, -1.0],
    [0.0, 1.0],
    [-1.0, -1.0],
    [1.0, -1.0],
    [-1.0, 1.0],
    [1.0, 1.0],
];

/// Halo width from which the four diagonal copies are worth their bandwidth.
///
/// Below it the diagonal copy lands one whole pixel from its axis-aligned
/// neighbours in *both* axes, so the union it adds is the outer corner texel of
/// the ring and nothing else — while costing the frame four more copies of
/// every glyph quad in it.
pub const HALO_DIAGONAL_MIN_WIDTH_PX: f32 = 1.5;

/// The offsets a halo of `width_px` is drawn along.
///
/// All eight of [`HALO_OFFSETS`] at [`HALO_DIAGONAL_MIN_WIDTH_PX`] and above,
/// the four axis-aligned ones below it. An empty slice means the halo is too
/// thin to draw at all, which is the same `< 0.5 px` rule
/// [`build_label_quads`] applies.
#[must_use]
pub fn halo_offsets(width_px: f32) -> &'static [[f32; 2]] {
    if !width_px.is_finite() || width_px < 0.5 {
        &[]
    } else if width_px < HALO_DIAGONAL_MIN_WIDTH_PX {
        &HALO_OFFSETS[..4]
    } else {
        &HALO_OFFSETS
    }
}

/// The whole label shader: screen-space quads sampling R8 coverage.
///
/// `{srgb}` is substituted at pipeline creation with `true` or `false`; a WGSL
/// `const` keeps the branch free at runtime.
const LABEL_SHADER_WGSL: &str = r#"
const CONVERT_SRGB: bool = {srgb};

@group(0) @binding(0) var atlas_texture: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    let low = color / 12.92;
    let high = pow((color + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, color <= vec3<f32>(0.04045));
}

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.uv = uv;
    var rgb = color.rgb;
    if (CONVERT_SRGB) {
        rgb = srgb_to_linear(rgb);
    }
    out.color = vec4<f32>(rgb, color.a);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coverage = textureSample(atlas_texture, atlas_sampler, in.uv).r;
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
"#;

/// One vertex of a glyph quad: screen position already in NDC, atlas UV, and a
/// straight (non-premultiplied) sRGB colour whose alpha the coverage scales.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct LabelVertex {
    /// Clip-space position; `x` right, `y` up, both in `-1..=1`.
    pub position: [f32; 2],
    /// Texture coordinate into the atlas, `0..=1`.
    pub uv: [f32; 2],
    /// Straight sRGB colour, RGBA.
    pub color: [u8; 4],
}

impl LabelVertex {
    /// Builds a vertex.
    #[must_use]
    pub const fn new(position: [f32; 2], uv: [f32; 2], color: [u8; 4]) -> Self {
        Self {
            position,
            uv,
            color,
        }
    }
}

/// The outline drawn under a label so it stays legible over busy imagery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelHalo {
    /// Straight sRGB colour of the halo. Assumed opaque — see the module docs.
    pub color: [u8; 4],
    /// Displacement of each halo copy, in pixels. Values below half a pixel
    /// produce no visible halo and are skipped.
    pub width_px: f32,
}

impl LabelHalo {
    /// Builds a halo.
    #[must_use]
    pub const fn new(color: [u8; 4], width_px: f32) -> Self {
        Self { color, width_px }
    }
}

/// One label placed on screen, ready to be turned into quads.
///
/// `origin_px` is the top-left of the label's collision box in the pass
/// viewport's pixel space (`y` down), i.e. the box
/// [`ShapedLabel::size_px`] describes. It is rounded to whole pixels before
/// projection.
#[derive(Debug, Clone, Copy)]
pub struct PlacedLabel<'a> {
    /// The shaped label, from [`crate::label::LabelEngine::shape`].
    pub shaped: &'a ShapedLabel,
    /// Top-left of the label box, in viewport pixels, `y` down.
    pub origin_px: [f32; 2],
    /// Straight sRGB fill colour.
    pub color: [u8; 4],
    /// Optional halo drawn underneath every label's fill.
    pub halo: Option<LabelHalo>,
}

/// Appends the quads for `labels` to `vertices`/`indices`, in NDC.
///
/// Halo copies for *all* labels come first, then every fill, so labels never
/// erase each other's outlines. A haloed label contributes
/// [`halo_offsets`]`(width).len() + 1` copies of its glyph quads — five or nine.
/// Labels with no glyphs contribute nothing, and so do non-finite origins — a
/// placement bug should drop a label, not corrupt the buffer.
///
/// `viewport_px` is the size of the render pass's viewport; `atlas_size` the
/// current [`GlyphAtlas::size`], which is what UVs are relative to.
///
/// # Errors
///
/// Returns [`RenderError::Gpu`] if `viewport_px` is not a positive finite size,
/// or if the resulting vertex count would exceed the `u32` index space.
pub fn build_label_quads(
    labels: &[PlacedLabel<'_>],
    atlas_size: u32,
    viewport_px: [f32; 2],
    vertices: &mut Vec<LabelVertex>,
    indices: &mut Vec<u32>,
) -> Result<(), RenderError> {
    if !viewport_px.iter().all(|value| value.is_finite())
        || viewport_px[0] <= 0.0
        || viewport_px[1] <= 0.0
    {
        return Err(RenderError::Gpu(format!(
            "label viewport {viewport_px:?} is not a positive finite size"
        )));
    }
    if atlas_size == 0 {
        return Ok(());
    }

    for label in labels {
        let Some(halo) = label.halo else {
            continue;
        };
        for offset in halo_offsets(halo.width_px) {
            push_label(
                label,
                [offset[0] * halo.width_px, offset[1] * halo.width_px],
                halo.color,
                atlas_size,
                viewport_px,
                vertices,
                indices,
            )?;
        }
    }
    for label in labels {
        push_label(
            label,
            [0.0, 0.0],
            label.color,
            atlas_size,
            viewport_px,
            vertices,
            indices,
        )?;
    }
    Ok(())
}

/// Emits one copy of a label's glyph quads, displaced by `shift_px`.
fn push_label(
    label: &PlacedLabel<'_>,
    shift_px: [f32; 2],
    color: [u8; 4],
    atlas_size: u32,
    viewport_px: [f32; 2],
    vertices: &mut Vec<LabelVertex>,
    indices: &mut Vec<u32>,
) -> Result<(), RenderError> {
    if label.shaped.is_empty() || !label.origin_px.iter().all(|value| value.is_finite()) {
        return Ok(());
    }
    // Round the origin, then the displacement: both keep texels on pixels.
    let origin = [
        label.origin_px[0].round() + shift_px[0].round(),
        label.origin_px[1].round() + shift_px[1].round(),
    ];

    for glyph in label.shaped.glyphs() {
        let left = origin[0] + glyph.offset_px[0];
        let top = origin[1] + glyph.offset_px[1];
        let right = left + glyph.slot.width as f32;
        let bottom = top + glyph.slot.height as f32;
        let [u0, v0, u1, v1] = glyph.slot.uv(atlas_size);

        let base = u32::try_from(vertices.len()).map_err(|_| {
            RenderError::Gpu(format!(
                "{} label vertices exceed the u32 index space",
                vertices.len()
            ))
        })?;
        if base > u32::MAX - 4 {
            return Err(RenderError::Gpu(
                "label vertex buffer would exceed the u32 index space".to_owned(),
            ));
        }
        vertices.push(LabelVertex::new(
            ndc(left, top, viewport_px),
            [u0, v0],
            color,
        ));
        vertices.push(LabelVertex::new(
            ndc(right, top, viewport_px),
            [u1, v0],
            color,
        ));
        vertices.push(LabelVertex::new(
            ndc(right, bottom, viewport_px),
            [u1, v1],
            color,
        ));
        vertices.push(LabelVertex::new(
            ndc(left, bottom, viewport_px),
            [u0, v1],
            color,
        ));
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Ok(())
}

/// The first pair of disagreeing [`ShapedLabel::generation`]s in `labels`, in
/// the order they were met.
///
/// [`None`] — the only state that draws — when every label was shaped against
/// one atlas generation, the empty set included.
///
/// The labels are compared against **each other** rather than against
/// [`GlyphAtlas::generation`] on purpose: the shaping generation is the
/// engine's counter, the atlas keeps its own, and the two are equal only
/// because today's engine happens to bump them together — a shell holding
/// `atlas_mut()` can clear the atlas without the engine noticing. This test
/// holds whatever either counter does.
///
/// It therefore catches the reachable half of the hazard: a caller that ignores
/// [`crate::label::place::LabelPlacer::is_stale`] and draws a set straddling a
/// repack. A set that is *entirely* pre-repack is caught by the resident
/// generation check in [`LabelPipeline::upload_labels`] unless the caller also
/// re-uploaded the atlas in between, which no order the module documents does.
fn mixed_generation(labels: &[PlacedLabel<'_>]) -> Option<[u32; 2]> {
    let mut first: Option<u32> = None;
    for label in labels {
        let generation = label.shaped.generation();
        match first {
            Some(seen) if seen != generation => return Some([seen, generation]),
            Some(_) => {}
            None => first = Some(generation),
        }
    }
    None
}

/// The `(first_row, rows)` band [`LabelPipeline::upload_atlas`] copies, given
/// what [`GlyphAtlas::dirty_rows`] reports for an atlas of `size` rows.
///
/// Three cases, and `upload_atlas` reaches exactly these three because it has
/// already returned when the atlas is neither dirty nor resized:
///
/// * not resized, band reported — the routine frame. One more glyph costs its
///   own handful of rows rather than the whole 1–16 MiB buffer.
/// * resized — every row, whatever the band says. The texture was created this
///   call, so its rows outside the band have never been written and would
///   sample as undefined. This covers the resize of an atlas that is *not*
///   dirty (a fresh pipeline handed an already-uploaded atlas) as well.
/// * no band at all — every row. Unreachable without a resize, since
///   [`GlyphAtlas::dirty_rows`] is `Some` exactly when the atlas is dirty, and
///   the safe reading of "dirty, extent unknown" regardless.
///
/// Split out so the choice is testable without a device; `write_rows` clamps
/// the band it is handed, so an out-of-range answer here drops a copy rather
/// than tripping the driver.
const fn upload_band(dirty_rows: Option<(u32, u32)>, size: u32, resized: bool) -> (u32, u32) {
    match dirty_rows {
        Some((top, bottom)) if !resized => (top, bottom.saturating_sub(top)),
        _ => (0, size),
    }
}

/// Viewport pixels (`y` down, origin top-left) to NDC (`y` up).
fn ndc(x: f32, y: f32, viewport_px: [f32; 2]) -> [f32; 2] {
    [
        (x / viewport_px[0]) * 2.0 - 1.0,
        1.0 - (y / viewport_px[1]) * 2.0,
    ]
}

/// The atlas as it currently exists on the GPU.
#[derive(Debug)]
struct AtlasTexture {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    size: u32,
    /// [`GlyphAtlas::generation`] of the pixels last written into `texture`.
    ///
    /// The size alone cannot tell a repack apart from a no-op: [`GlyphAtlas::clear`]
    /// keeps the buffer size and invalidates every [`crate::label::AtlasRect`],
    /// so without this a label from before the repack would sample whatever
    /// glyph now occupies its slot — wrong text rather than missing text.
    generation: u32,
}

/// Render pipeline for label glyph quads, plus the vertex, index and atlas
/// resources one frame's worth of labels needs.
///
/// Used in the same fixed order as [`crate::vector::VectorPipeline`]:
///
/// 1. [`LabelPipeline::upload_atlas`] and [`LabelPipeline::upload_labels`]
///    (in `prepare`), then
/// 2. [`LabelPipeline::draw`] (in `paint`), after the vector meshes so that
///    text sits on top of the geometry it names.
#[derive(Debug)]
pub struct LabelPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    atlas: Option<AtlasTexture>,
    vertices: wgpu::Buffer,
    vertex_capacity: u32,
    indices: wgpu::Buffer,
    index_capacity: u32,
    index_count: u32,
    scratch_vertices: Vec<LabelVertex>,
    scratch_indices: Vec<u32>,
}

impl LabelPipeline {
    /// Builds the pipeline for a colour target of `target_format`.
    ///
    /// # Errors
    ///
    /// Currently infallible in practice, but returns `Result` so future
    /// capability checks do not become a breaking change; `wgpu` reports shader
    /// and pipeline validation failures through the device's error scope.
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        let source = LABEL_SHADER_WGSL.replace(
            "{srgb}",
            if target_format.is_srgb() {
                "true"
            } else {
                "false"
            },
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oxigis-render label shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oxigis-render label atlas layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Nearest, not linear: glyphs are rasterised at their final size and
        // drawn at integer positions, so the mapping is 1:1 and interpolation
        // could only soften it.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("oxigis-render label sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("oxigis-render label pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Unorm8x4];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oxigis-render label pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: LABEL_VERTEX_SIZE,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &VERTEX_ATTRIBUTES,
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Quads are emitted in screen order; culling would depend on
                // which way `y` points after the NDC flip.
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            // Painter's algorithm: halos first, then fills.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxigis-render label vertices"),
            size: LABEL_VERTEX_SIZE * u64::from(INITIAL_VERTEX_CAPACITY),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxigis-render label indices"),
            size: 4 * u64::from(INITIAL_INDEX_CAPACITY),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            sampler,
            atlas: None,
            vertices,
            vertex_capacity: INITIAL_VERTEX_CAPACITY,
            indices,
            index_capacity: INITIAL_INDEX_CAPACITY,
            index_count: 0,
            scratch_vertices: Vec::new(),
            scratch_indices: Vec::new(),
        })
    }

    /// Number of indices the last [`LabelPipeline::upload_labels`] produced.
    #[must_use]
    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    /// Side length of the atlas texture currently on the GPU, if any.
    #[must_use]
    pub fn atlas_size(&self) -> Option<u32> {
        self.atlas.as_ref().map(|atlas| atlas.size)
    }

    /// Number of vertices the vertex buffer holds without reallocating.
    #[must_use]
    pub fn vertex_capacity(&self) -> u32 {
        self.vertex_capacity
    }

    /// Number of indices the index buffer holds without reallocating.
    #[must_use]
    pub fn index_capacity(&self) -> u32 {
        self.index_capacity
    }

    /// Copies the atlas into a texture if it changed since the last call.
    ///
    /// Recreates the texture (and its bind group) when the atlas has grown.
    /// Clears [`GlyphAtlas::is_dirty`] on success, which is why the atlas is
    /// taken mutably.
    ///
    /// Only the rows [`GlyphAtlas::dirty_rows`] reports are copied, so a frame
    /// that packs one new glyph moves a few kilobytes rather than the whole
    /// 1–16 MiB buffer. A resize copies everything regardless: the texture is
    /// created here and its rows outside the band have never been written.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Gpu`] if the atlas side length is not a multiple
    /// of [`wgpu`]'s 256-byte row alignment — every size
    /// [`GlyphAtlas`] produces by default is, since it starts at 1024 and
    /// doubles.
    pub fn upload_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: &mut GlyphAtlas,
    ) -> Result<(), RenderError> {
        let size = atlas.size();
        let resized = self
            .atlas
            .as_ref()
            .is_none_or(|current| current.size != size);
        if !resized && !atlas.is_dirty() {
            return Ok(());
        }
        if !size.is_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT) {
            return Err(RenderError::Gpu(format!(
                "glyph atlas of {size}² R8 texels has rows that are not a multiple of \
                 {COPY_BYTES_PER_ROW_ALIGNMENT} bytes"
            )));
        }

        if resized {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("oxigis-render label atlas"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxigis-render label atlas bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.atlas = Some(AtlasTexture {
                texture,
                bind_group,
                size,
                // Nothing has been written into the new texture yet; the copy
                // below is what makes the recorded generation true.
                generation: u32::MAX,
            });
        }

        let Some(resident) = self.atlas.as_mut() else {
            return Err(RenderError::Gpu(
                "label atlas texture was not created".to_owned(),
            ));
        };
        let (first_row, rows) = upload_band(atlas.dirty_rows(), size, resized);
        Self::write_rows(queue, resident, atlas, first_row, rows);
        atlas.mark_clean();
        resident.generation = atlas.generation();
        Ok(())
    }

    /// Copies rows `first_row..first_row + rows` of `atlas` into the texture.
    ///
    /// Silently copies nothing when the band is empty or reaches past the
    /// buffer — the caller derives it from the atlas's own size, so an
    /// out-of-range band is a bug here rather than a condition to report to the
    /// shell, and a dropped copy shows up as an unpainted glyph rather than as
    /// a driver-level validation failure.
    ///
    /// `bytes_per_row` stays the atlas's full side length, which is what makes
    /// the 256-byte row-alignment check on the caller's side cover this copy
    /// unchanged.
    fn write_rows(
        queue: &wgpu::Queue,
        resident: &AtlasTexture,
        atlas: &GlyphAtlas,
        first_row: u32,
        rows: u32,
    ) {
        let size = resident.size;
        if rows == 0 || first_row >= size || size - first_row < rows {
            return;
        }
        let stride = size as usize;
        let Some((offset, length)) = usize::try_from(first_row)
            .ok()
            .zip(usize::try_from(rows).ok())
            .and_then(|(row, rows)| Some((row.checked_mul(stride)?, rows.checked_mul(stride)?)))
        else {
            return;
        };
        let Some(band) = atlas
            .pixels()
            .get(offset..)
            .and_then(|tail| tail.get(..length))
        else {
            return;
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &resident.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: first_row,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            band,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width: size,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Rebuilds the frame's vertex and index buffers from `labels`.
    ///
    /// Returns the index count, which is also available from
    /// [`LabelPipeline::index_count`]. Buffers grow to the next power of two
    /// and are never shrunk, so a busy frame sizes them for the session.
    ///
    /// # Errors
    ///
    /// Propagates [`build_label_quads`]'s errors, and returns
    /// [`RenderError::Gpu`] if [`LabelPipeline::upload_atlas`] has not run yet
    /// (there would be nothing to sample), if the atlas resident on the GPU is
    /// not the one the labels were packed into, or if `labels` mixes
    /// [`crate::label::ShapedLabel::generation`]s — see
    /// [`crate::label::place::LabelPlacer::is_stale`], whose contract this
    /// enforces for callers that ignore it.
    pub fn upload_labels(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        labels: &[PlacedLabel<'_>],
        atlas: &GlyphAtlas,
        viewport_px: [f32; 2],
    ) -> Result<u32, RenderError> {
        let Some(resident) = self.atlas.as_ref() else {
            return Err(RenderError::Gpu(
                "label atlas has not been uploaded: call upload_atlas first".to_owned(),
            ));
        };
        if resident.size != atlas.size() {
            return Err(RenderError::Gpu(format!(
                "label atlas on the GPU is {}² but the labels were packed into {}²: \
                 call upload_atlas first",
                resident.size,
                atlas.size()
            )));
        }
        // A repack keeps the size and invalidates every rect, so the size check
        // above cannot see it. These two can: the texture must hold the current
        // atlas contents, and every label must come from one shaping pass.
        if resident.generation != atlas.generation() {
            return Err(RenderError::Gpu(format!(
                "label atlas on the GPU holds generation {} but the labels were packed into \
                 generation {}: call upload_atlas first",
                resident.generation,
                atlas.generation()
            )));
        }
        if let Some(mixed) = mixed_generation(labels) {
            return Err(RenderError::Gpu(format!(
                "labels mix shaping generations {} and {}: the atlas was repacked mid-pass, \
                 so the older half indexes glyphs that have moved",
                mixed[0], mixed[1]
            )));
        }

        self.scratch_vertices.clear();
        self.scratch_indices.clear();
        build_label_quads(
            labels,
            atlas.size(),
            viewport_px,
            &mut self.scratch_vertices,
            &mut self.scratch_indices,
        )?;

        let Ok(vertex_len) = u32::try_from(self.scratch_vertices.len()) else {
            return Err(RenderError::Gpu(format!(
                "{} label vertices exceed the addressable range",
                self.scratch_vertices.len()
            )));
        };
        let Ok(index_len) = u32::try_from(self.scratch_indices.len()) else {
            return Err(RenderError::Gpu(format!(
                "{} label indices exceed the addressable range",
                self.scratch_indices.len()
            )));
        };

        if vertex_len > self.vertex_capacity {
            let capacity = vertex_len.next_power_of_two().max(INITIAL_VERTEX_CAPACITY);
            self.vertices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oxigis-render label vertices"),
                size: LABEL_VERTEX_SIZE * u64::from(capacity),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = capacity;
        }
        if index_len > self.index_capacity {
            let capacity = index_len.next_power_of_two().max(INITIAL_INDEX_CAPACITY);
            self.indices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oxigis-render label indices"),
                size: 4 * u64::from(capacity),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.index_capacity = capacity;
        }

        if vertex_len > 0 {
            queue.write_buffer(
                &self.vertices,
                0,
                bytemuck::cast_slice(&self.scratch_vertices),
            );
            queue.write_buffer(
                &self.indices,
                0,
                bytemuck::cast_slice(&self.scratch_indices),
            );
        }
        self.index_count = index_len;
        Ok(index_len)
    }

    /// Draws the frame's labels in one indexed call.
    ///
    /// `scissor` clips the pass — normally the caller's own viewport rectangle,
    /// since labels are already in its pixel space; [`None`] leaves whatever
    /// the pass had.
    ///
    /// The pass is left with this pipeline, bind group and buffers bound;
    /// callers sharing a pass (an `egui_wgpu` callback does) must restore their
    /// own state.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Gpu`] if no atlas has been uploaded while there
    /// are labels to draw.
    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        scissor: Option<ScissorRect>,
    ) -> Result<(), RenderError> {
        if self.index_count == 0 {
            return Ok(());
        }
        let Some(atlas) = self.atlas.as_ref() else {
            return Err(RenderError::Gpu(
                "label draw without an uploaded atlas".to_owned(),
            ));
        };
        if let Some(rect) = scissor {
            render_pass.set_scissor_rect(rect.x, rect.y, rect.width, rect.height);
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &atlas.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertices.slice(..));
        render_pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // As in `vector/pipeline.rs`, the device-touching half is exercised by the
    // shells; the tests below cover the pure data paths this module owns.
    use super::{
        HALO_DIAGONAL_MIN_WIDTH_PX, HALO_OFFSETS, LABEL_VERTEX_SIZE, LabelHalo, LabelVertex,
        PlacedLabel, build_label_quads, halo_offsets, mixed_generation, ndc, upload_band,
    };
    use crate::error::RenderError;
    use crate::label::engine::{LabelEngine, ShapedLabel};
    use std::sync::Arc;

    fn label(text: &str) -> Arc<ShapedLabel> {
        let mut engine = LabelEngine::new(oxifont_bundled::NOTO_SANS_REGULAR.to_vec())
            .expect("bundled Noto Sans parses");
        engine.shape(text, 14.0).expect("shapes")
    }

    #[test]
    fn the_vertex_stride_matches_the_attribute_layout() {
        // Float32x2 (8) + Float32x2 (8) + Unorm8x4 (4).
        assert_eq!(LABEL_VERTEX_SIZE, 20);
    }

    #[test]
    fn the_atlas_upload_copies_the_dirty_band_and_nothing_else() {
        // The routine frame: one 20-texel glyph packed into a 1024² atlas is
        // 20 KiB on the wire, not 1 MiB.
        assert_eq!(upload_band(Some((0, 20)), 1024, false), (0, 20));
        assert_eq!(upload_band(Some((512, 540)), 1024, false), (512, 28));
        // The band is half-open, so a one-row change is one row.
        assert_eq!(upload_band(Some((7, 8)), 1024, false), (7, 1));
    }

    #[test]
    fn a_resized_atlas_is_uploaded_whole_however_narrow_its_band() {
        // The texture was created this call; rows outside the band have never
        // been written and would sample as undefined.
        assert_eq!(upload_band(Some((0, 20)), 2048, true), (0, 2048));
        assert_eq!(upload_band(Some((1000, 1004)), 2048, true), (0, 2048));
        // The case a band alone would get wrong: a *clean* atlas whose texture
        // is stale anyway, e.g. a fresh pipeline handed an atlas some earlier
        // one already uploaded. Skipping rows here would leave them undefined.
        assert_eq!(upload_band(None, 2048, true), (0, 2048));
        // "Dirty, extent unknown" reads as every row too.
        assert_eq!(upload_band(None, 1024, false), (0, 1024));
    }

    #[test]
    fn an_inverted_band_asks_for_no_rows_rather_than_wrapping() {
        // `GlyphAtlas` never produces one; `saturating_sub` is what keeps a
        // future bug a dropped copy instead of a 4-billion-row `write_texture`.
        assert_eq!(upload_band(Some((900, 100)), 1024, false), (900, 0));
    }

    #[test]
    fn pixels_project_to_ndc_with_y_flipped() {
        assert_eq!(ndc(0.0, 0.0, [800.0, 600.0]), [-1.0, 1.0]);
        assert_eq!(ndc(800.0, 600.0, [800.0, 600.0]), [1.0, -1.0]);
        assert_eq!(ndc(400.0, 300.0, [800.0, 600.0]), [0.0, 0.0]);
    }

    #[test]
    fn one_label_becomes_one_quad_per_glyph() {
        let shaped = label("Hi");
        let placed = PlacedLabel {
            shaped: &shaped,
            origin_px: [10.0, 20.0],
            color: [255, 255, 255, 255],
            halo: None,
        };
        let (mut vertices, mut indices) = (Vec::new(), Vec::new());
        build_label_quads(&[placed], 1024, [800.0, 600.0], &mut vertices, &mut indices)
            .expect("quads build");

        let glyphs = shaped.glyphs().len();
        assert_eq!(glyphs, 2);
        assert_eq!(vertices.len(), glyphs * 4);
        assert_eq!(indices.len(), glyphs * 6);
        // Every index addresses a vertex, and every quad is two triangles.
        let vertex_count = u32::try_from(vertices.len()).expect("small");
        assert!(indices.iter().all(|index| *index < vertex_count));
        // UVs stay inside the atlas.
        assert!(
            vertices
                .iter()
                .all(|v| v.uv.iter().all(|c| (0.0..=1.0).contains(c)))
        );
    }

    #[test]
    fn a_halo_adds_one_copy_per_offset_and_draws_first() {
        let shaped = label("Hi");
        let placed = PlacedLabel {
            shaped: &shaped,
            origin_px: [10.0, 20.0],
            color: [0, 0, 0, 255],
            halo: Some(LabelHalo::new([255, 255, 255, 255], 2.0)),
        };
        let (mut vertices, mut indices) = (Vec::new(), Vec::new());
        build_label_quads(&[placed], 1024, [800.0, 600.0], &mut vertices, &mut indices)
            .expect("quads build");

        let per_copy = shaped.glyphs().len() * 4;
        assert_eq!(vertices.len(), per_copy * (HALO_OFFSETS.len() + 1));
        // The halo copies come first, in the halo colour; the fill is last.
        assert_eq!(vertices[0].color, [255, 255, 255, 255]);
        assert_eq!(
            vertices[per_copy * HALO_OFFSETS.len()].color,
            [0, 0, 0, 255]
        );
        // Every halo copy is displaced from the fill by the halo width.
        let fill_x = vertices[per_copy * HALO_OFFSETS.len()].position[0];
        const HALO_WIDTH_PX: f32 = 2.0;
        for copy in 0..HALO_OFFSETS.len() {
            let halo_x = vertices[copy * per_copy].position[0];
            // offset · width, in pixels, converted to the NDC span of one pixel
            // across an 800 px viewport (2 units / 800 px).
            let expected = HALO_OFFSETS[copy][0] * HALO_WIDTH_PX * (2.0 / 800.0);
            assert!((halo_x - fill_x - expected).abs() < 1e-6, "copy {copy}");
        }
    }

    #[test]
    fn a_thin_halo_costs_four_copies_instead_of_eight() {
        let shaped = label("Hi");
        let per_copy = shaped.glyphs().len() * 4;
        // Just under the threshold: the diagonals would land a whole pixel out
        // in both axes, so they are not worth four more copies of every quad.
        for (width, copies) in [
            (0.5_f32, 4_usize),
            (1.0, 4),
            (HALO_DIAGONAL_MIN_WIDTH_PX, 8),
        ] {
            let placed = PlacedLabel {
                shaped: &shaped,
                origin_px: [10.0, 20.0],
                color: [0, 0, 0, 255],
                halo: Some(LabelHalo::new([255, 255, 255, 255], width)),
            };
            let (mut vertices, mut indices) = (Vec::new(), Vec::new());
            build_label_quads(&[placed], 1024, [800.0, 600.0], &mut vertices, &mut indices)
                .expect("quads build");
            assert_eq!(halo_offsets(width).len(), copies, "width {width}");
            assert_eq!(vertices.len(), per_copy * (copies + 1), "width {width}");
            assert_eq!(indices.len(), shaped.glyphs().len() * 6 * (copies + 1));
            // Whichever count it is, the copies emitted are a prefix of the
            // documented eight and every one of them is axis-aligned or not
            // exactly as `HALO_OFFSETS` says.
            assert_eq!(halo_offsets(width), &HALO_OFFSETS[..copies]);
        }
        // Nothing to draw is an empty slice, not a panic on the index.
        for width in [0.0_f32, 0.49, f32::NAN, f32::INFINITY, -2.0] {
            assert!(halo_offsets(width).is_empty(), "width {width}");
        }
    }

    #[test]
    fn labels_from_two_shaping_generations_are_reported_rather_than_drawn() {
        let mut engine = LabelEngine::new(oxifont_bundled::NOTO_SANS_REGULAR.to_vec())
            .expect("bundled Noto Sans parses");
        let first = engine.shape("Hi", 14.0).expect("shapes");
        // A fallback font invalidates the cache and the atlas, which is exactly
        // the repack `LabelPlacer::is_stale` exists to report.
        engine.add_fallback_font(oxifont_bundled::NOTO_SANS_MONO_REGULAR.to_vec());
        let second = engine.shape("Hi", 14.0).expect("shapes again");
        assert_ne!(first.generation(), second.generation());

        fn placed(shaped: &ShapedLabel) -> PlacedLabel<'_> {
            PlacedLabel {
                shaped,
                origin_px: [0.0, 0.0],
                color: [255; 4],
                halo: None,
            }
        }
        assert_eq!(mixed_generation(&[]), None, "an empty frame is uniform");
        assert_eq!(mixed_generation(&[placed(&first), placed(&first)]), None);
        assert_eq!(
            mixed_generation(&[placed(&first), placed(&second)]),
            Some([first.generation(), second.generation()]),
        );
    }

    #[test]
    fn a_sub_pixel_halo_is_skipped() {
        let shaped = label("Hi");
        let placed = PlacedLabel {
            shaped: &shaped,
            origin_px: [10.0, 20.0],
            color: [0, 0, 0, 255],
            halo: Some(LabelHalo::new([255, 255, 255, 255], 0.25)),
        };
        let (mut vertices, mut indices) = (Vec::new(), Vec::new());
        build_label_quads(&[placed], 1024, [800.0, 600.0], &mut vertices, &mut indices)
            .expect("quads build");
        assert_eq!(vertices.len(), shaped.glyphs().len() * 4);
    }

    #[test]
    fn empty_labels_and_bad_origins_contribute_nothing() {
        let blank = label("   ");
        let visible = label("Hi");
        let labels = [
            PlacedLabel {
                shaped: &blank,
                origin_px: [0.0, 0.0],
                color: [255; 4],
                halo: Some(LabelHalo::new([0; 4], 2.0)),
            },
            PlacedLabel {
                shaped: &visible,
                origin_px: [f32::NAN, 0.0],
                color: [255; 4],
                halo: None,
            },
        ];
        let (mut vertices, mut indices) = (Vec::new(), Vec::new());
        build_label_quads(&labels, 1024, [800.0, 600.0], &mut vertices, &mut indices)
            .expect("quads build");
        assert!(vertices.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn a_degenerate_viewport_is_an_error() {
        let shaped = label("Hi");
        let placed = PlacedLabel {
            shaped: &shaped,
            origin_px: [0.0, 0.0],
            color: [255; 4],
            halo: None,
        };
        let (mut vertices, mut indices) = (Vec::new(), Vec::new());
        for viewport in [[0.0, 600.0], [800.0, -1.0], [f32::NAN, 600.0]] {
            assert!(matches!(
                build_label_quads(&[placed], 1024, viewport, &mut vertices, &mut indices),
                Err(RenderError::Gpu(_))
            ));
        }
    }

    #[test]
    fn glyph_quads_land_on_whole_pixels() {
        let shaped = label("Hi");
        let placed = PlacedLabel {
            shaped: &shaped,
            // A fractional origin must be rounded away, or the atlas texels
            // stop lining up with screen pixels and the text goes soft.
            origin_px: [10.4, 20.6],
            color: [255; 4],
            halo: None,
        };
        let (mut vertices, mut indices) = (Vec::new(), Vec::new());
        build_label_quads(&[placed], 1024, [800.0, 600.0], &mut vertices, &mut indices)
            .expect("quads build");
        for vertex in &vertices {
            let x_px = (vertex.position[0] + 1.0) / 2.0 * 800.0;
            let y_px = (1.0 - vertex.position[1]) / 2.0 * 600.0;
            assert!(
                (x_px - x_px.round()).abs() < 1e-3,
                "x {x_px} is not integral"
            );
            assert!(
                (y_px - y_px.round()).abs() < 1e-3,
                "y {y_px} is not integral"
            );
        }
    }

    #[test]
    fn vertices_are_plain_old_data() {
        let vertex = LabelVertex::new([0.5, -0.5], [0.25, 0.75], [1, 2, 3, 4]);
        let bytes: &[u8] = bytemuck::bytes_of(&vertex);
        assert_eq!(bytes.len(), LABEL_VERTEX_SIZE as usize);
        assert_eq!(bytes[16..], [1, 2, 3, 4]);
    }
}
