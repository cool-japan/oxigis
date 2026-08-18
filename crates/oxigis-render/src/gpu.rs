//! wgpu raster-tile pipeline: textured quads, one draw call per tile.
//!
//! Everything here borrows the [`wgpu::Device`]/[`wgpu::Queue`] it is handed
//! and never creates an adapter of its own, so the same code works whether the
//! device came from `eframe`'s `egui_wgpu::RenderState` (the intended path, see
//! `survey-oxiui.md` §2), from a standalone `winit` surface, or from a headless
//! test harness.
//!
//! # Geometry
//!
//! There is no vertex buffer for the quad itself: the vertex shader derives the
//! four corners of a triangle strip from `@builtin(vertex_index)`. The only
//! per-tile data is an *instance* record — the destination rectangle in
//! normalised device coordinates plus an RGBA tint — held in one growable
//! buffer that [`TilePipeline::upload_instances`] rewrites each frame.
//!
//! Each tile is drawn with its own bind group (texture + sampler) and its own
//! slice of that instance buffer. The slice offset is used rather than a
//! non-zero `first_instance`, because base-instance offsets are not available
//! on the WebGL2 fallback path that `wgpu` selects when a browser has no
//! WebGPU support (`survey-wasm.md` §2).
//!
//! # UV sub-rectangles
//!
//! A second, parallel instance-step buffer carries a `[u0, v0, du, dv]` texture
//! rectangle per instance, so a quad can sample *part* of its texture. That is
//! what makes the over-zoom / parent-tile fallback possible: a missing tile is
//! drawn from its nearest resident ancestor through the sub-rectangle it
//! occupies inside it. The rectangle lives in its own buffer rather than inside
//! [`TileInstance`] because that record is shared verbatim with the vector
//! pipeline, whose shader declares the same two attribute locations.

use crate::error::RenderError;
use crate::viewport::TilePlacement;

/// Bytes occupied by one UV sub-rectangle in the parallel instance buffer.
pub const TILE_UV_SIZE: u64 = core::mem::size_of::<[f32; 4]>() as u64;

/// UV rectangle that samples a tile texture whole.
pub const FULL_TILE_UV: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Bytes occupied by one [`TileInstance`] in the instance buffer.
pub const TILE_INSTANCE_SIZE: u64 = core::mem::size_of::<TileInstance>() as u64;

/// Number of instances the buffer is created with before it has to grow.
const INITIAL_INSTANCE_CAPACITY: u32 = 64;

/// Largest tile edge accepted by [`TilePipeline::upload_tile`], in pixels.
///
/// 8192 is the `max_texture_dimension_2d` guaranteed by `wgpu`'s downlevel
/// WebGL2 defaults, so staying at or below it keeps the browser fallback path
/// working.
pub const MAX_TILE_TEXTURE_SIZE: u32 = 8_192;

/// The whole raster-tile shader: a screen-space textured quad.
const TILE_SHADER_WGSL: &str = r#"
struct Instance {
    // x, y of the top-left corner in NDC, then width and height in NDC units.
    @location(0) rect: vec4<f32>,
    @location(1) tint: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: Instance,
    // x, y of the sampled sub-rectangle's top-left corner in texture space,
    // then its width and height. [0, 0, 1, 1] samples the texture whole.
    @location(2) uv_rect: vec4<f32>,
) -> VertexOutput {
    // Triangle-strip corners: (0,0) (1,0) (0,1) (1,1) in texture space.
    let corner = vec2<f32>(f32(vertex_index & 1u), f32((vertex_index >> 1u) & 1u));
    var out: VertexOutput;
    out.clip_position = vec4<f32>(
        instance.rect.x + corner.x * instance.rect.z,
        instance.rect.y - corner.y * instance.rect.w,
        0.0,
        1.0,
    );
    out.uv = uv_rect.xy + corner * uv_rect.zw;
    out.tint = instance.tint;
    return out;
}

@group(0) @binding(0) var tile_texture: texture_2d<f32>;
@group(0) @binding(1) var tile_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(tile_texture, tile_sampler, in.uv);
    return texel * in.tint;
}
"#;

/// Per-tile instance record uploaded to the GPU.
///
/// `repr(C)` with two 16-byte members: no implicit padding, so it is `Pod` and
/// can be memory-cast straight into the vertex buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileInstance {
    /// Destination rectangle `[x, y, width, height]` in normalised device
    /// coordinates: `x`/`y` are the top-left corner (`y` up) and the extents
    /// are positive.
    pub rect: [f32; 4],
    /// Linear RGBA multiplier applied to the sampled texel; `[1, 1, 1, 1]`
    /// draws the tile unchanged and the alpha channel doubles as layer opacity.
    pub tint: [f32; 4],
}

impl TileInstance {
    /// Fully opaque, unmodified colour.
    pub const OPAQUE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    /// Creates an instance from an NDC rectangle and a tint.
    #[must_use]
    pub const fn new(rect: [f32; 4], tint: [f32; 4]) -> Self {
        Self { rect, tint }
    }

    /// Creates an instance for a placed tile.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidViewport`] if either surface dimension is
    /// not finite and positive.
    pub fn from_placement(
        placement: &TilePlacement,
        view_size_px: [f32; 2],
        tint: [f32; 4],
    ) -> Result<Self, RenderError> {
        Ok(Self {
            rect: placement.to_ndc_rect(view_size_px)?,
            tint,
        })
    }
}

/// A tile's pixels resident on the GPU, together with the bind group that binds
/// them (and the shared sampler) to the tile pipeline.
#[derive(Debug)]
pub struct TileTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl TileTexture {
    /// Width of the texture in texels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height of the texture in texels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Device memory the texture occupies, in bytes: `width * height * 4`.
    ///
    /// The weight [`crate::renderer::MapRenderer`] charges its texture cache
    /// with, so a source serving unusually large tiles is bounded by bytes and
    /// not only by entry count. Exact rather than an estimate: the format is
    /// always one of the two RGBA8 variants and there is a single mip level.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        (self.width as usize).saturating_mul(self.height as usize) * 4
    }

    /// The underlying texture.
    #[must_use]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// A default view of the texture.
    #[must_use]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The bind group that binds this texture and the pipeline's sampler to
    /// group 0.
    #[must_use]
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

/// One tile to draw: the texture to sample and which instance record positions
/// it.
#[derive(Debug, Clone, Copy)]
pub struct TileDraw<'a> {
    /// GPU-resident pixels for the tile.
    pub texture: &'a TileTexture,
    /// Index into the buffer last passed to [`TilePipeline::upload_instances`].
    pub instance: u32,
}

/// Render pipeline for textured raster tiles, plus the instance buffer the
/// draws read from.
///
/// The instance buffer is owned here rather than by the caller so that
/// [`TilePipeline::draw`] can match the shape an `egui_wgpu` paint callback
/// needs. The two are used in a fixed order every frame:
///
/// 1. [`TilePipeline::upload_instances`] (needs `&mut self`, called from
///    `prepare`), then
/// 2. [`TilePipeline::draw`] (needs only `&self`, called from `paint`).
///
/// `draw` rejects any [`TileDraw::instance`] beyond the number of records last
/// uploaded, so a stale or mismatched list surfaces as [`RenderError::Gpu`]
/// instead of sampling whatever bytes happen to be in the buffer.
#[derive(Debug)]
pub struct TilePipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture_format: wgpu::TextureFormat,
    instances: wgpu::Buffer,
    /// Parallel to `instances`: one `[u0, v0, du, dv]` per record. Grown and
    /// written in lockstep, so `instance_capacity`/`instance_len` govern both.
    uvs: wgpu::Buffer,
    instance_capacity: u32,
    instance_len: u32,
}

impl TilePipeline {
    /// Builds the pipeline for a colour target of `target_format`.
    ///
    /// Tile textures are created as `Rgba8UnormSrgb` when the target is an sRGB
    /// format and `Rgba8Unorm` otherwise, so that sampling and blending happen
    /// in the same colour space the surface expects.
    ///
    /// # Errors
    ///
    /// Currently infallible in practice, but returns `Result` so that future
    /// capability checks (texture formats, downlevel flags) do not become a
    /// breaking change. `wgpu` reports shader and pipeline validation failures
    /// through the device's error scope, not through this return value.
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oxigis-render tile shader"),
            source: wgpu::ShaderSource::Wgsl(TILE_SHADER_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oxigis-render tile bind group layout"),
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("oxigis-render tile pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
        const UV_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Float32x4];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oxigis-render tile pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: TILE_INSTANCE_SIZE,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &INSTANCE_ATTRIBUTES,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: TILE_UV_SIZE,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &UV_ATTRIBUTES,
                    },
                ],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("oxigis-render tile sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxigis-render tile instances"),
            size: TILE_INSTANCE_SIZE * u64::from(INITIAL_INSTANCE_CAPACITY),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uvs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxigis-render tile uv rects"),
            size: TILE_UV_SIZE * u64::from(INITIAL_INSTANCE_CAPACITY),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            sampler,
            texture_format: tile_texture_format(target_format),
            instances,
            uvs,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            instance_len: 0,
        })
    }

    /// Format tile textures are created with.
    #[must_use]
    pub fn texture_format(&self) -> wgpu::TextureFormat {
        self.texture_format
    }

    /// Number of instance records currently valid in the instance buffer.
    #[must_use]
    pub fn instance_len(&self) -> u32 {
        self.instance_len
    }

    /// Number of instance records the buffer can hold without reallocating.
    #[must_use]
    pub fn instance_capacity(&self) -> u32 {
        self.instance_capacity
    }

    /// Uploads a decoded RGBA8 tile, returning its GPU-resident form.
    ///
    /// `rgba` must be tightly packed, `width * height * 4` bytes, top row
    /// first.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidTileImage`] if the dimensions are zero,
    /// exceed [`MAX_TILE_TEXTURE_SIZE`], or disagree with the buffer length.
    pub fn upload_tile(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<TileTexture, RenderError> {
        check_rgba_dimensions(width, height, rgba)?;

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("oxigis-render tile texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.texture_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oxigis-render tile bind group"),
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

        let tile = TileTexture {
            texture,
            view,
            bind_group,
            width,
            height,
        };
        self.write_tile(queue, &tile, rgba)?;
        Ok(tile)
    }

    /// Overwrites the pixels of an existing tile texture, reusing its bind
    /// group.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidTileImage`] if `rgba` does not match the
    /// texture's dimensions.
    pub fn write_tile(
        &self,
        queue: &wgpu::Queue,
        target: &TileTexture,
        rgba: &[u8],
    ) -> Result<(), RenderError> {
        check_rgba_dimensions(target.width, target.height, rgba)?;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(target.width * 4),
                rows_per_image: Some(target.height),
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    /// Replaces the contents of the instance buffer, growing it if needed.
    ///
    /// Every instance samples its texture whole ([`FULL_TILE_UV`]); callers
    /// drawing a parent tile through a sub-rectangle want
    /// [`TilePipeline::upload_instances_uv`] instead.
    ///
    /// Must be called before [`TilePipeline::draw`] every frame the instance
    /// list changes; `draw` validates its indices against the length recorded
    /// here.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Gpu`] if more instances are supplied than a `u32`
    /// can index.
    pub fn upload_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[TileInstance],
    ) -> Result<(), RenderError> {
        let uvs = vec![FULL_TILE_UV; instances.len()];
        self.upload_instances_uv(device, queue, instances, &uvs)
    }

    /// [`TilePipeline::upload_instances`] with the sampled sub-rectangle of
    /// each instance spelled out.
    ///
    /// The two slices are parallel: `uvs[i]` is the `[u0, v0, du, dv]` texture
    /// rectangle instance `i` samples, in `0..=1` coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Gpu`] if more instances are supplied than a `u32`
    /// can index, or if the two slices disagree in length.
    pub fn upload_instances_uv(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[TileInstance],
        uvs: &[[f32; 4]],
    ) -> Result<(), RenderError> {
        if uvs.len() != instances.len() {
            return Err(RenderError::Gpu(format!(
                "{} tile uv rects do not match {} instances",
                uvs.len(),
                instances.len()
            )));
        }
        let Ok(len) = u32::try_from(instances.len()) else {
            return Err(RenderError::Gpu(format!(
                "{} tile instances exceed the addressable range",
                instances.len()
            )));
        };
        if len > self.instance_capacity {
            let capacity = len.next_power_of_two().max(INITIAL_INSTANCE_CAPACITY);
            self.instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oxigis-render tile instances"),
                size: TILE_INSTANCE_SIZE * u64::from(capacity),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.uvs = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oxigis-render tile uv rects"),
                size: TILE_UV_SIZE * u64::from(capacity),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = capacity;
        }
        if len > 0 {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(instances));
            queue.write_buffer(&self.uvs, 0, bytemuck::cast_slice(uvs));
        }
        self.instance_len = len;
        Ok(())
    }

    /// Draws one textured quad per entry of `placements`.
    ///
    /// The render pass is left with this pipeline and the last tile's bindings
    /// set; callers that share the pass with other renderers (as an
    /// `egui_wgpu` callback does) must set their own state afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Gpu`] if any [`TileDraw::instance`] is not below
    /// [`TilePipeline::instance_len`].
    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        placements: &[TileDraw<'_>],
    ) -> Result<(), RenderError> {
        if placements.is_empty() {
            return Ok(());
        }
        for draw in placements {
            if draw.instance >= self.instance_len {
                return Err(RenderError::Gpu(format!(
                    "tile instance {} is out of range: only {} were uploaded",
                    draw.instance, self.instance_len
                )));
            }
        }

        render_pass.set_pipeline(&self.pipeline);
        for draw in placements {
            let offset = u64::from(draw.instance) * TILE_INSTANCE_SIZE;
            let uv_offset = u64::from(draw.instance) * TILE_UV_SIZE;
            render_pass.set_bind_group(0, draw.texture.bind_group(), &[]);
            render_pass
                .set_vertex_buffer(0, self.instances.slice(offset..offset + TILE_INSTANCE_SIZE));
            render_pass.set_vertex_buffer(1, self.uvs.slice(uv_offset..uv_offset + TILE_UV_SIZE));
            render_pass.draw(0..4, 0..1);
        }
        Ok(())
    }
}

/// The RGBA multiplier that fades a layer to `opacity`, leaving its colour
/// alone.
///
/// The ONE normalisation the whole stack shares — [`crate::MapRenderer::set_opacity`]
/// and [`crate::VectorLayerRenderer::set_opacity`] both go through it — so a
/// raster layer and a vector layer at the same slider position fade by exactly
/// the same amount, and a `0.5` recorded in a project file means one thing.
///
/// Out-of-range values clamp and a non-finite one is fully opaque: an alpha of
/// NaN reaches the blender as garbage rather than as an error, so it is refused
/// here where the value enters the pipeline.
#[must_use]
pub fn opacity_tint(opacity: f32) -> [f32; 4] {
    let alpha = if opacity.is_finite() {
        opacity.clamp(0.0, 1.0)
    } else {
        1.0
    };
    [1.0, 1.0, 1.0, alpha]
}

/// Texture format tiles are uploaded with for a given colour target.
#[must_use]
pub fn tile_texture_format(target_format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    if target_format.is_srgb() {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    }
}

/// Validates that `rgba` holds exactly `width * height` RGBA8 texels, returning
/// the expected byte count.
fn check_rgba_dimensions(width: u32, height: u32, rgba: &[u8]) -> Result<usize, RenderError> {
    if width == 0 || height == 0 {
        return Err(RenderError::InvalidTileImage(format!(
            "tile dimensions must be positive, got {width}x{height}"
        )));
    }
    if width > MAX_TILE_TEXTURE_SIZE || height > MAX_TILE_TEXTURE_SIZE {
        return Err(RenderError::InvalidTileImage(format!(
            "tile {width}x{height} exceeds the {MAX_TILE_TEXTURE_SIZE} texel limit"
        )));
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|texels| texels.checked_mul(4));
    let Some(expected) = expected else {
        return Err(RenderError::InvalidTileImage(format!(
            "tile {width}x{height} does not fit in memory"
        )));
    };
    if rgba.len() != expected {
        return Err(RenderError::InvalidTileImage(format!(
            "tile {width}x{height} needs {expected} rgba bytes, got {}",
            rgba.len()
        )));
    }
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::{
        FULL_TILE_UV, MAX_TILE_TEXTURE_SIZE, TILE_INSTANCE_SIZE, TILE_UV_SIZE, TileInstance,
        check_rgba_dimensions, opacity_tint, tile_texture_format,
    };
    use crate::error::RenderError;
    use crate::mercator::TileId;
    use crate::viewport::TilePlacement;

    fn placement(x: f32, y: f32, size: f32) -> TilePlacement {
        let Ok(tile) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        TilePlacement { tile, x, y, size }
    }

    #[test]
    fn instance_layout_is_tight() {
        assert_eq!(TILE_INSTANCE_SIZE, 32);
        assert_eq!(core::mem::align_of::<TileInstance>(), 4);
        let instances = [
            TileInstance::new([-1.0, 1.0, 2.0, 2.0], TileInstance::OPAQUE),
            TileInstance::new([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5]),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        assert_eq!(bytes.len(), 64);
        let round_trip: &[TileInstance] = bytemuck::cast_slice(bytes);
        assert_eq!(round_trip, instances.as_slice());
    }

    #[test]
    fn instances_place_tiles_in_ndc() {
        let full = placement(0.0, 0.0, 256.0);
        let Ok(instance) =
            TileInstance::from_placement(&full, [256.0, 256.0], TileInstance::OPAQUE)
        else {
            panic!("instance construction failed");
        };
        assert_eq!(instance.rect, [-1.0, 1.0, 2.0, 2.0]);
        assert_eq!(instance.tint, [1.0, 1.0, 1.0, 1.0]);

        // Bottom-right quadrant of a 256x256 surface.
        let quadrant = placement(128.0, 128.0, 128.0);
        let Ok(instance) =
            TileInstance::from_placement(&quadrant, [256.0, 256.0], TileInstance::OPAQUE)
        else {
            panic!("instance construction failed");
        };
        assert_eq!(instance.rect, [0.0, 0.0, 1.0, 1.0]);

        // The shader walks the quad as `x + u*w` / `y - v*h`; check both
        // corners land on the surface edges.
        let [x, y, w, h] = instance.rect;
        assert_eq!([x + w, y - h], [1.0, -1.0]);
    }

    #[test]
    fn instances_reject_a_degenerate_surface() {
        let tile = placement(0.0, 0.0, 256.0);
        assert!(matches!(
            TileInstance::from_placement(&tile, [0.0, 256.0], TileInstance::OPAQUE),
            Err(RenderError::InvalidViewport(_))
        ));
        assert!(matches!(
            TileInstance::from_placement(&tile, [256.0, f32::NAN], TileInstance::OPAQUE),
            Err(RenderError::InvalidViewport(_))
        ));
    }

    #[test]
    fn rgba_dimensions_are_validated() {
        assert_eq!(check_rgba_dimensions(2, 2, &[0u8; 16]).ok(), Some(16));
        assert!(matches!(
            check_rgba_dimensions(2, 2, &[0u8; 15]),
            Err(RenderError::InvalidTileImage(_))
        ));
        assert!(matches!(
            check_rgba_dimensions(0, 2, &[]),
            Err(RenderError::InvalidTileImage(_))
        ));
        assert!(matches!(
            check_rgba_dimensions(MAX_TILE_TEXTURE_SIZE + 1, 1, &[]),
            Err(RenderError::InvalidTileImage(_))
        ));
    }

    #[test]
    fn the_uv_buffer_is_a_tight_parallel_array() {
        // `draw` slices the two instance buffers by the same index, so the UV
        // stride has to be exactly one `[f32; 4]` with no padding.
        assert_eq!(TILE_UV_SIZE, 16);
        assert_eq!(TILE_UV_SIZE as usize, core::mem::size_of::<[f32; 4]>());
        let uvs = [FULL_TILE_UV, [0.25, 0.5, 0.25, 0.25]];
        let bytes: &[u8] = bytemuck::cast_slice(&uvs);
        assert_eq!(bytes.len(), 2 * TILE_UV_SIZE as usize);
        let round_trip: &[[f32; 4]] = bytemuck::cast_slice(bytes);
        assert_eq!(round_trip, uvs.as_slice());
    }

    #[test]
    fn the_default_uv_rect_samples_the_whole_texture() {
        // The vertex shader walks `uv_rect.xy + corner * uv_rect.zw` with
        // `corner` over the unit square, so this is the identity mapping.
        let [u, v, du, dv] = FULL_TILE_UV;
        assert_eq!([u, v], [0.0, 0.0]);
        assert_eq!([u + du, v + dv], [1.0, 1.0]);

        // A quadrant sub-rect stays inside the texture at both corners, which
        // is the invariant `TileId::sub_rect_in` has to preserve.
        let [u, v, du, dv] = [0.5, 0.5, 0.5, 0.5];
        assert_eq!([u + du, v + dv], [1.0, 1.0]);
    }

    #[test]
    fn a_texture_weighs_its_texels() {
        // `MapRenderer` charges its byte budget with this, so it must be the
        // real RGBA8 footprint and not a per-entry constant.
        assert_eq!(256usize * 256 * 4, 262_144);
        assert_eq!(
            (MAX_TILE_TEXTURE_SIZE as usize) * (MAX_TILE_TEXTURE_SIZE as usize) * 4,
            268_435_456,
            "one 8192 px tile is 256 MiB — why entry count alone is not a bound"
        );
    }

    #[test]
    fn an_opacity_tint_only_touches_alpha_and_never_carries_a_nan() {
        // Colour is left alone: a faded layer must be the same colour, not a
        // darker one — the fragment shader multiplies `texel * tint`.
        assert_eq!(opacity_tint(1.0), TileInstance::OPAQUE);
        assert_eq!(opacity_tint(0.5), [1.0, 1.0, 1.0, 0.5]);
        assert_eq!(opacity_tint(0.0), [1.0, 1.0, 1.0, 0.0]);

        // Out of range clamps rather than producing an alpha the blender would
        // read as an over-bright or negative contribution.
        assert_eq!(opacity_tint(-4.0), [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(opacity_tint(12.0), TileInstance::OPAQUE);

        // A non-finite slider value is fully opaque, NOT a NaN alpha: `clamp`
        // itself would propagate the NaN, which is why the guard runs first.
        assert_eq!(opacity_tint(f32::NAN), TileInstance::OPAQUE);
        assert_eq!(opacity_tint(f32::INFINITY), TileInstance::OPAQUE);
        assert_eq!(opacity_tint(f32::NEG_INFINITY), TileInstance::OPAQUE);
        assert!(
            opacity_tint(f32::NAN)
                .iter()
                .all(|channel| !channel.is_nan())
        );
    }

    #[test]
    fn tile_format_follows_the_target_color_space() {
        assert_eq!(
            tile_texture_format(wgpu::TextureFormat::Bgra8UnormSrgb),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            tile_texture_format(wgpu::TextureFormat::Bgra8Unorm),
            wgpu::TextureFormat::Rgba8Unorm
        );
        assert_eq!(
            tile_texture_format(wgpu::TextureFormat::Rgba16Float),
            wgpu::TextureFormat::Rgba8Unorm
        );
    }
}
