//! wgpu pipeline for tessellated vector tiles: one indexed draw per tile.
//!
//! The structure mirrors [`crate::gpu::TilePipeline`] deliberately — same
//! borrowed-device contract, same growable instance buffer, same
//! `upload_* (&mut) then draw (&)` split that an `egui_wgpu` paint callback
//! needs — so that the raster and vector passes can share a frame without
//! either owning GPU state the other cannot see.
//!
//! # Geometry and placement
//!
//! A [`crate::vector::VectorMesh`] lives in the *unit tile square*: `0..1`,
//! `y` down, with MVT buffer geometry allowed to spill outside it. Where that
//! square lands on screen is exactly the raster path's per-tile transform, so
//! this pipeline reuses [`crate::gpu::TileInstance`] verbatim: `rect` is
//! [`crate::viewport::TilePlacement::to_ndc_rect`] and `tint` is a global
//! multiplier whose alpha is the layer opacity. The vertex shader evaluates
//! `rect.xy + position * rect.zw` (with `y` negated) — the same expression the
//! raster quad uses for its corners.
//!
//! # Clipping
//!
//! Buffer geometry would otherwise paint over the neighbouring tile, so every
//! draw is scissored to the intersection of its tile quad and the caller's clip
//! rectangle ([`tile_scissor`]). A scissor rectangle is in *framebuffer* pixels,
//! not viewport-relative ones, which is why the draw call is given the clip
//! origin: inside an `egui_wgpu` callback that is
//! `PaintCallbackInfo::viewport_in_pixels()`'s top-left corner.
//!
//! # Colour
//!
//! Vertex colours are straight (non-premultiplied) sRGB bytes, blended with
//! [`wgpu::BlendState::ALPHA_BLENDING`]. When the colour target is an sRGB
//! format the shader converts the RGB components to linear first, so a vector
//! layer and a raster tile of the same colour match.
//!
//! # Pixel widths
//!
//! A mesh is tessellated for one on-screen tile size but drawn at whatever size
//! the camera currently gives a tile, which differs from it by up to a factor of
//! two during a smooth zoom. Every vertex therefore carries the pixel-width
//! expansion that produced it ([`crate::vector::VectorVertex::offset`]) next to
//! its position, and a per-instance scale — [`crate::vector::offset_scale`] of
//! the mesh's baked tile size against the frame's — rescales it in the vertex
//! shader. Fills carry a zero offset and are untouched by any of it.
//!
//! # Buffer reuse
//!
//! Meshes come and go with the viewport, so [`MeshBufferPool`] keeps the buffers
//! of retired tiles in power-of-two size classes and
//! [`VectorPipeline::upload_mesh_pooled`] draws from it before asking the device
//! for a new allocation. A mesh smaller than its backing buffer is not a
//! problem: [`VectorTileGpu`] carries explicit counts and the draw call reads
//! only those.
//!
//! # Batching, later
//!
//! One tile is one vertex buffer, one index buffer and one draw call, which is
//! the right granularity while tiles arrive and expire independently. The next
//! step (tracked in `TODO.md` §5.2) is a per-frame arena: concatenate the
//! frame's meshes into one growable vertex/index buffer pair, push the tile
//! index into a vertex attribute, and issue a single `draw_indexed` per
//! *scissor group* instead of per tile. Nothing in this module's public shape
//! blocks that — [`VectorTileGpu`] would become a range inside the arena and
//! [`VectorDraw`] would keep its meaning.

use crate::error::RenderError;
use crate::gpu::{TILE_INSTANCE_SIZE, TileInstance};
use crate::vector::tess::{VectorMesh, VectorVertex, offset_scale};
use crate::viewport::TilePlacement;

/// Bytes occupied by one [`VectorVertex`] in a vertex buffer.
pub const VECTOR_VERTEX_SIZE: u64 = core::mem::size_of::<VectorVertex>() as u64;

/// Bytes occupied by one per-instance offset scale.
pub const VECTOR_SCALE_SIZE: u64 = core::mem::size_of::<f32>() as u64;

/// Number of tile instances the buffer is created with before it has to grow.
const INITIAL_INSTANCE_CAPACITY: u32 = 64;

/// Smallest buffer [`MeshBufferPool`] hands out, so a tile of a handful of
/// triangles does not take a size class of its own.
const MIN_POOLED_BYTES: u64 = 4096;

/// Default byte budget of a [`MeshBufferPool`]: retired buffers past this are
/// dropped rather than kept for reuse.
pub const DEFAULT_POOL_BYTE_BUDGET: u64 = 64 * 1024 * 1024;

/// Buffers of one kind [`MeshBufferPool`] keeps, whatever the byte budget
/// allows. Reuse scans the list, so its length is the cost of an upload.
const MAX_POOLED_BUFFERS: usize = 256;

/// The whole vector shader: per-tile placement plus interpolated vertex colour.
///
/// `{srgb}` is substituted at pipeline creation with `true` or `false`; a WGSL
/// `const` keeps the branch free at runtime.
const VECTOR_SHADER_WGSL: &str = r#"
const CONVERT_SRGB: bool = {srgb};

struct Instance {
    // x, y of the tile's top-left corner in NDC, then width and height.
    @location(0) rect: vec4<f32>,
    @location(1) tint: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    let low = color / 12.92;
    let high = pow((color + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, color <= vec3<f32>(0.04045));
}

@vertex
fn vs_main(
    @location(2) position: vec2<f32>,
    @location(3) offset: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) offset_scale: f32,
    instance: Instance,
) -> VertexOutput {
    var out: VertexOutput;
    // `offset` is the pixel-width expansion already inside `position`, so
    // rescaling it re-derives the width for this frame's tile size and leaves
    // fill geometry (offset zero) exactly where it was tessellated.
    let placed = position + offset * (offset_scale - 1.0);
    out.clip_position = vec4<f32>(
        instance.rect.x + placed.x * instance.rect.z,
        instance.rect.y - placed.y * instance.rect.w,
        0.0,
        1.0,
    );
    var rgb = color.rgb;
    if (CONVERT_SRGB) {
        rgb = srgb_to_linear(rgb);
    }
    out.color = vec4<f32>(rgb * instance.tint.rgb, color.a * instance.tint.a);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// A scissor rectangle in framebuffer pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScissorRect {
    /// Left edge, from the framebuffer's left.
    pub x: u32,
    /// Top edge, from the framebuffer's top.
    pub y: u32,
    /// Width in pixels; never zero (an empty rectangle is [`None`] instead).
    pub width: u32,
    /// Height in pixels; never zero.
    pub height: u32,
}

impl ScissorRect {
    /// Creates a rectangle, returning [`None`] if it would be empty.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        Some(Self {
            x,
            y,
            width,
            height,
        })
    }
}

/// The scissor rectangle for one placed tile, clipped to the caller's viewport.
///
/// `clip_origin_px` is the top-left of the pass viewport inside the
/// framebuffer and `clip_size_px` its size — together, the rectangle the
/// placement's coordinates are relative to. Returns [`None`] when the tile
/// falls entirely outside it, in which case the draw can be skipped.
///
/// Both edges round **down**. Neighbouring tiles share the same fractional
/// boundary and therefore the same floor, so their rectangles are exactly
/// contiguous: no gap, and — what matters for translucent paint — no column
/// inside two scissors at once, which would blend that column twice and draw a
/// grid of seams over the map at every fractional placement.
#[must_use]
pub fn tile_scissor(
    placement: &TilePlacement,
    clip_origin_px: [f32; 2],
    clip_size_px: [f32; 2],
) -> Option<ScissorRect> {
    if !clip_size_px.iter().all(|value| value.is_finite())
        || !clip_origin_px.iter().all(|value| value.is_finite())
        || clip_size_px[0] <= 0.0
        || clip_size_px[1] <= 0.0
    {
        return None;
    }
    if !placement.x.is_finite() || !placement.y.is_finite() || !placement.size.is_finite() {
        return None;
    }

    let left = (placement.x.max(0.0)).min(clip_size_px[0]);
    let top = (placement.y.max(0.0)).min(clip_size_px[1]);
    let right = ((placement.x + placement.size).max(0.0)).min(clip_size_px[0]);
    let bottom = ((placement.y + placement.size).max(0.0)).min(clip_size_px[1]);
    if right <= left || bottom <= top {
        return None;
    }

    let origin_x = clip_origin_px[0].max(0.0);
    let origin_y = clip_origin_px[1].max(0.0);
    let x = (origin_x + left).floor().max(0.0) as u32;
    let y = (origin_y + top).floor().max(0.0) as u32;
    let far_x = (origin_x + right).floor().max(0.0) as u32;
    let far_y = (origin_y + bottom).floor().max(0.0) as u32;
    ScissorRect::new(x, y, far_x.saturating_sub(x), far_y.saturating_sub(y))
}

/// One tile's mesh resident on the GPU.
///
/// The buffers may be larger than the mesh: they come in power-of-two size
/// classes so [`MeshBufferPool`] can hand them back out. The counts below are
/// what the draw call reads.
#[derive(Debug)]
pub struct VectorTileGpu {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    vertex_count: u32,
    index_count: u32,
    baked_tile_size_px: f32,
}

impl VectorTileGpu {
    /// Number of vertices in the buffer.
    #[must_use]
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Number of indices in the buffer.
    #[must_use]
    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    /// The vertex buffer.
    #[must_use]
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertices
    }

    /// The index buffer.
    #[must_use]
    pub fn index_buffer(&self) -> &wgpu::Buffer {
        &self.indices
    }

    /// GPU memory the two buffers occupy, allocation classes included.
    ///
    /// This is what a byte-bounded mesh cache accounts with: unlike a raster
    /// tile, whose size the texture format fixes, a vector mesh is as large as
    /// its data.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        self.vertices.size().saturating_add(self.indices.size())
    }

    /// On-screen tile size the mesh's pixel widths were tessellated for, or
    /// `0.0` when the mesh did not record one.
    #[must_use]
    pub fn baked_tile_size_px(&self) -> f32 {
        self.baked_tile_size_px
    }

    /// Factor the vertex offsets need when one tile covers `tile_size_px`
    /// pixels — what [`VectorPipeline::upload_instances_scaled`] wants.
    #[must_use]
    pub fn offset_scale_at(&self, tile_size_px: f32) -> f32 {
        offset_scale(self.baked_tile_size_px, tile_size_px)
    }
}

/// Retired mesh buffers, kept for the next upload instead of being destroyed.
///
/// A zoom step retires and re-uploads the whole visible set at once — a few
/// hundred buffer creations, each of which the device defers cleanup for — so
/// buffers are handed back here in power-of-two size classes and reused
/// whenever the incoming mesh fits. Bounded by bytes: once the pool holds
/// [`MeshBufferPool::byte_budget`], further buffers are dropped.
#[derive(Debug)]
pub struct MeshBufferPool {
    vertices: Vec<PooledBuffer>,
    indices: Vec<PooledBuffer>,
    bytes: u64,
    byte_budget: u64,
    reuses: u64,
    allocations: u64,
}

#[derive(Debug)]
struct PooledBuffer {
    size: u64,
    buffer: wgpu::Buffer,
}

impl MeshBufferPool {
    /// Creates a pool that keeps at most `byte_budget` bytes of retired
    /// buffers.
    #[must_use]
    pub fn new(byte_budget: u64) -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            bytes: 0,
            byte_budget,
            reuses: 0,
            allocations: 0,
        }
    }

    /// Bytes of retired buffers currently held.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Bytes of retired buffers this pool will hold.
    #[must_use]
    pub fn byte_budget(&self) -> u64 {
        self.byte_budget
    }

    /// Number of buffers handed back out, and number created because the pool
    /// had nothing that fit.
    #[must_use]
    pub fn stats(&self) -> (u64, u64) {
        (self.reuses, self.allocations)
    }

    /// Hands a retired mesh's buffers back for reuse.
    pub fn recycle(&mut self, mesh: VectorTileGpu) {
        let VectorTileGpu {
            vertices, indices, ..
        } = mesh;
        self.keep(true, vertices);
        self.keep(false, indices);
    }

    /// Drops every pooled buffer — for a device loss, or to give the memory
    /// back.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.bytes = 0;
    }

    fn keep(&mut self, vertex_buffer: bool, buffer: wgpu::Buffer) {
        let size = buffer.size();
        if size == 0 || self.bytes.saturating_add(size) > self.byte_budget {
            return;
        }
        let pool = if vertex_buffer {
            &mut self.vertices
        } else {
            &mut self.indices
        };
        if pool.len() >= MAX_POOLED_BUFFERS {
            return;
        }
        pool.push(PooledBuffer { size, buffer });
        self.bytes = self.bytes.saturating_add(size);
    }

    /// The smallest pooled buffer that holds `size` bytes, if there is one.
    fn take(&mut self, vertex_buffer: bool, size: u64) -> Option<wgpu::Buffer> {
        let pool = if vertex_buffer {
            &mut self.vertices
        } else {
            &mut self.indices
        };
        let mut best: Option<usize> = None;
        for (index, entry) in pool.iter().enumerate() {
            if entry.size < size {
                continue;
            }
            match best {
                Some(current) if pool[current].size <= entry.size => {}
                _ => best = Some(index),
            }
        }
        let entry = pool.swap_remove(best?);
        self.bytes = self.bytes.saturating_sub(entry.size);
        self.reuses += 1;
        Some(entry.buffer)
    }
}

impl Default for MeshBufferPool {
    /// A pool holding [`DEFAULT_POOL_BYTE_BUDGET`].
    fn default() -> Self {
        Self::new(DEFAULT_POOL_BYTE_BUDGET)
    }
}

/// Size class a buffer of `size` bytes is allocated at: the next power of two,
/// never below [`MIN_POOLED_BYTES`], so retired buffers fit the next mesh of
/// roughly the same size.
fn buffer_class(size: u64) -> u64 {
    size.max(MIN_POOLED_BYTES).next_power_of_two()
}

/// One tile to draw: its mesh, its placement instance and its clip rectangle.
#[derive(Debug, Clone, Copy)]
pub struct VectorDraw<'a> {
    /// GPU-resident mesh for the tile.
    pub mesh: &'a VectorTileGpu,
    /// Index into the buffer last passed to
    /// [`VectorPipeline::upload_instances`].
    pub instance: u32,
    /// Clip rectangle, normally [`tile_scissor`]'s result. [`None`] leaves
    /// whatever scissor the pass already had, which lets buffer geometry spill.
    pub scissor: Option<ScissorRect>,
}

/// Render pipeline for tessellated vector tiles, plus the per-tile instance
/// buffer the draws read their placement from.
///
/// Used in the same fixed order as [`crate::gpu::TilePipeline`]:
///
/// 1. [`VectorPipeline::upload_mesh`] / [`VectorPipeline::upload_instances`]
///    (in `prepare`), then
/// 2. [`VectorPipeline::draw`] (in `paint`).
#[derive(Debug)]
pub struct VectorPipeline {
    pipeline: wgpu::RenderPipeline,
    instances: wgpu::Buffer,
    /// One [`crate::vector::offset_scale`] per instance, in the same order.
    scales: wgpu::Buffer,
    instance_capacity: u32,
    instance_len: u32,
}

impl VectorPipeline {
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
        let source = VECTOR_SHADER_WGSL.replace(
            "{srgb}",
            if target_format.is_srgb() {
                "true"
            } else {
                "false"
            },
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oxigis-render vector shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("oxigis-render vector pipeline layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
        // Position, the pixel-width expansion inside it, then the colour.
        const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![2 => Float32x2, 3 => Float32x2, 4 => Unorm8x4];
        const SCALE_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![5 => Float32];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oxigis-render vector pipeline"),
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
                        array_stride: VECTOR_VERTEX_SIZE,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &VERTEX_ATTRIBUTES,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: VECTOR_SCALE_SIZE,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &SCALE_ATTRIBUTES,
                    },
                ],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Tessellated geometry has no consistent winding (lyon fills
                // and hand-built circle fans disagree), and the mesh is flat.
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            // Painter's algorithm: the mesh is already in draw order.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // Straight (non-premultiplied) alpha, matching the colours
                    // the tessellator writes.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxigis-render vector instances"),
            size: TILE_INSTANCE_SIZE * u64::from(INITIAL_INSTANCE_CAPACITY),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scales = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxigis-render vector offset scales"),
            size: VECTOR_SCALE_SIZE * u64::from(INITIAL_INSTANCE_CAPACITY),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            instances,
            scales,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            instance_len: 0,
        })
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

    /// Uploads a tessellated mesh into fresh GPU buffers.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Gpu`] if the mesh is empty, if its index count is
    /// not a multiple of three, or if any index is out of range — all of which
    /// would otherwise become a device-lost error at draw time.
    pub fn upload_mesh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &VectorMesh,
    ) -> Result<VectorTileGpu, RenderError> {
        let (vertex_count, index_count) = check_mesh(mesh)?;

        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxigis-render vector vertices"),
            size: VECTOR_VERTEX_SIZE * u64::from(vertex_count),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxigis-render vector indices"),
            size: 4 * u64::from(index_count),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertices, 0, bytemuck::cast_slice(&mesh.vertices));
        queue.write_buffer(&indices, 0, bytemuck::cast_slice(&mesh.indices));

        Ok(VectorTileGpu {
            vertices,
            indices,
            vertex_count,
            index_count,
            baked_tile_size_px: mesh.baked_tile_size_px,
        })
    }

    /// [`VectorPipeline::upload_mesh`] reusing the buffers of retired meshes.
    ///
    /// Takes a buffer of the mesh's size class out of `pool` when there is one
    /// and asks the device only otherwise, which is what keeps a zoom step from
    /// creating and destroying a few hundred buffers. Writes go through the
    /// queue, so a recycled buffer is ordered behind the commands that still
    /// read it.
    ///
    /// # Errors
    ///
    /// As [`VectorPipeline::upload_mesh`].
    pub fn upload_mesh_pooled(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &VectorMesh,
        pool: &mut MeshBufferPool,
    ) -> Result<VectorTileGpu, RenderError> {
        let (vertex_count, index_count) = check_mesh(mesh)?;
        let vertex_bytes = VECTOR_VERTEX_SIZE * u64::from(vertex_count);
        let index_bytes = 4 * u64::from(index_count);

        let vertices = match pool.take(true, vertex_bytes) {
            Some(buffer) => buffer,
            None => {
                pool.allocations += 1;
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("oxigis-render vector vertices"),
                    size: buffer_class(vertex_bytes),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }
        };
        let indices = match pool.take(false, index_bytes) {
            Some(buffer) => buffer,
            None => {
                pool.allocations += 1;
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("oxigis-render vector indices"),
                    size: buffer_class(index_bytes),
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }
        };
        queue.write_buffer(&vertices, 0, bytemuck::cast_slice(&mesh.vertices));
        queue.write_buffer(&indices, 0, bytemuck::cast_slice(&mesh.indices));

        Ok(VectorTileGpu {
            vertices,
            indices,
            vertex_count,
            index_count,
            baked_tile_size_px: mesh.baked_tile_size_px,
        })
    }

    /// Replaces the contents of the instance buffer, growing it if needed.
    ///
    /// Every instance is given an offset scale of `1.0`, i.e. the meshes are
    /// drawn with the pixel widths they were tessellated for. Callers that keep
    /// meshes across a zoom step want
    /// [`VectorPipeline::upload_instances_scaled`] instead.
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
        let scales = vec![1.0f32; instances.len()];
        self.upload_instances_scaled(device, queue, instances, &scales)
    }

    /// [`VectorPipeline::upload_instances`] with the per-instance offset scale
    /// spelled out — [`VectorTileGpu::offset_scale_at`] of the tile drawn by
    /// that instance.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Gpu`] if more instances are supplied than a `u32`
    /// can index, or if the two slices disagree in length.
    pub fn upload_instances_scaled(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[TileInstance],
        scales: &[f32],
    ) -> Result<(), RenderError> {
        if scales.len() != instances.len() {
            return Err(RenderError::Gpu(format!(
                "{} vector offset scales do not match {} instances",
                scales.len(),
                instances.len()
            )));
        }
        let Ok(len) = u32::try_from(instances.len()) else {
            return Err(RenderError::Gpu(format!(
                "{} vector tile instances exceed the addressable range",
                instances.len()
            )));
        };
        if len > self.instance_capacity {
            let capacity = len.next_power_of_two().max(INITIAL_INSTANCE_CAPACITY);
            self.instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oxigis-render vector instances"),
                size: TILE_INSTANCE_SIZE * u64::from(capacity),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.scales = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oxigis-render vector offset scales"),
                size: VECTOR_SCALE_SIZE * u64::from(capacity),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = capacity;
        }
        if len > 0 {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(instances));
            queue.write_buffer(&self.scales, 0, bytemuck::cast_slice(scales));
        }
        self.instance_len = len;
        Ok(())
    }

    /// Draws one indexed mesh per entry of `draws`, in order.
    ///
    /// `restore_scissor` is set once after the last draw, so the pass is handed
    /// back with the caller's own clip rectangle rather than the last tile's;
    /// pass [`None`] only when the caller sets its own scissor afterwards.
    ///
    /// The pass is left with this pipeline and the last mesh's buffers bound —
    /// callers sharing the pass (an `egui_wgpu` callback does) must restore
    /// their state.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Gpu`] if any [`VectorDraw::instance`] is not
    /// below [`VectorPipeline::instance_len`].
    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        draws: &[VectorDraw<'_>],
        restore_scissor: Option<ScissorRect>,
    ) -> Result<(), RenderError> {
        if draws.is_empty() {
            return Ok(());
        }
        for draw in draws {
            if draw.instance >= self.instance_len {
                return Err(RenderError::Gpu(format!(
                    "vector instance {} is out of range: only {} were uploaded",
                    draw.instance, self.instance_len
                )));
            }
        }

        render_pass.set_pipeline(&self.pipeline);
        let mut scissored = false;
        for draw in draws {
            if draw.mesh.index_count == 0 {
                continue;
            }
            if let Some(scissor) = draw.scissor {
                render_pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                scissored = true;
            }
            let offset = u64::from(draw.instance) * TILE_INSTANCE_SIZE;
            render_pass
                .set_vertex_buffer(0, self.instances.slice(offset..offset + TILE_INSTANCE_SIZE));
            render_pass.set_vertex_buffer(1, draw.mesh.vertices.slice(..));
            let scale_offset = u64::from(draw.instance) * VECTOR_SCALE_SIZE;
            render_pass.set_vertex_buffer(
                2,
                self.scales
                    .slice(scale_offset..scale_offset + VECTOR_SCALE_SIZE),
            );
            render_pass.set_index_buffer(draw.mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..draw.mesh.index_count, 0, 0..1);
        }
        if scissored && let Some(restore) = restore_scissor {
            render_pass.set_scissor_rect(restore.x, restore.y, restore.width, restore.height);
        }
        Ok(())
    }
}

/// Validates a mesh before it reaches the GPU, returning
/// `(vertex_count, index_count)`.
///
/// # Errors
///
/// Returns [`RenderError::Gpu`] for an empty mesh, an index count that is not a
/// multiple of three, a vertex or index count beyond `u32`, or an index that
/// does not address a vertex.
pub fn check_mesh(mesh: &VectorMesh) -> Result<(u32, u32), RenderError> {
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return Err(RenderError::Gpu(
            "vector mesh is empty: skip the tile instead of uploading it".to_owned(),
        ));
    }
    if !mesh.indices.len().is_multiple_of(3) {
        return Err(RenderError::Gpu(format!(
            "vector mesh has {} indices, not a whole number of triangles",
            mesh.indices.len()
        )));
    }
    let Ok(vertex_count) = u32::try_from(mesh.vertices.len()) else {
        return Err(RenderError::Gpu(format!(
            "vector mesh has {} vertices, beyond the u32 index space",
            mesh.vertices.len()
        )));
    };
    let Ok(index_count) = u32::try_from(mesh.indices.len()) else {
        return Err(RenderError::Gpu(format!(
            "vector mesh has {} indices, beyond the addressable range",
            mesh.indices.len()
        )));
    };
    if let Some(bad) = mesh.indices.iter().find(|index| **index >= vertex_count) {
        return Err(RenderError::Gpu(format!(
            "vector mesh index {bad} addresses no vertex (only {vertex_count})"
        )));
    }
    Ok((vertex_count, index_count))
}

#[cfg(test)]
mod tests {
    // As in `gpu.rs`, the device-touching half is exercised by the shells; the
    // tests below cover the pure data paths this module owns.
    use super::{
        DEFAULT_POOL_BYTE_BUDGET, MIN_POOLED_BYTES, MeshBufferPool, ScissorRect, VECTOR_SCALE_SIZE,
        VECTOR_VERTEX_SIZE, buffer_class, check_mesh, tile_scissor,
    };
    use crate::error::RenderError;
    use crate::mercator::TileId;
    use crate::vector::tess::{VectorMesh, VectorVertex};
    use crate::viewport::TilePlacement;

    fn placement(x: f32, y: f32, size: f32) -> TilePlacement {
        let Ok(tile) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        TilePlacement { tile, x, y, size }
    }

    fn mesh() -> VectorMesh {
        VectorMesh {
            vertices: vec![
                VectorVertex::new([0.0, 0.0], [255, 0, 0, 255]),
                VectorVertex::new([1.0, 0.0], [255, 0, 0, 255]),
                VectorVertex::new([0.0, 1.0], [255, 0, 0, 255]),
            ],
            indices: vec![0, 1, 2],
            baked_tile_size_px: 256.0,
        }
    }

    #[test]
    fn the_vertex_stride_matches_the_attribute_layout() {
        // Float32x2 position (8) + Float32x2 offset (8) + Unorm8x4 colour (4).
        assert_eq!(VECTOR_VERTEX_SIZE, 20);
        assert_eq!(VECTOR_SCALE_SIZE, 4);
    }

    #[test]
    fn buffers_are_allocated_in_reusable_size_classes() {
        assert_eq!(buffer_class(1), MIN_POOLED_BYTES);
        assert_eq!(buffer_class(MIN_POOLED_BYTES), MIN_POOLED_BYTES);
        assert_eq!(buffer_class(MIN_POOLED_BYTES + 1), MIN_POOLED_BYTES * 2);
        assert_eq!(buffer_class(3 * 1024 * 1024), 4 * 1024 * 1024);
        // Rounding up is what lets a retired buffer serve the next mesh: any
        // size in a class fits the class it rounds to.
        for size in [1u64, 5000, 100_000, 7_000_000] {
            assert!(buffer_class(size) >= size);
        }
    }

    #[test]
    fn an_empty_pool_hands_nothing_out() {
        let mut pool = MeshBufferPool::default();
        assert_eq!(pool.byte_budget(), DEFAULT_POOL_BYTE_BUDGET);
        assert_eq!(pool.bytes(), 0);
        assert_eq!(pool.stats(), (0, 0));
        assert!(pool.take(true, 1024).is_none());
        assert!(pool.take(false, 1024).is_none());
        pool.clear();
        assert_eq!(pool.bytes(), 0);
        assert_eq!(MeshBufferPool::new(0).byte_budget(), 0);
    }

    #[test]
    fn meshes_are_validated_before_upload() {
        assert_eq!(check_mesh(&mesh()).ok(), Some((3, 3)));

        assert!(matches!(
            check_mesh(&VectorMesh::new()),
            Err(RenderError::Gpu(_))
        ));

        let mut partial = mesh();
        partial.indices.pop();
        assert!(matches!(check_mesh(&partial), Err(RenderError::Gpu(_))));

        let mut out_of_range = mesh();
        out_of_range.indices = vec![0, 1, 9];
        assert!(matches!(
            check_mesh(&out_of_range),
            Err(RenderError::Gpu(_))
        ));

        let mut no_indices = mesh();
        no_indices.indices.clear();
        assert!(matches!(check_mesh(&no_indices), Err(RenderError::Gpu(_))));
    }

    #[test]
    fn a_scissor_clips_the_tile_to_the_viewport() {
        // Tile fully inside a 256x256 clip rect that starts at (10, 20).
        let inside = tile_scissor(&placement(32.0, 64.0, 128.0), [10.0, 20.0], [256.0, 256.0]);
        assert_eq!(inside, ScissorRect::new(42, 84, 128, 128));

        // Tile hanging off the left and top edges: clipped, not shifted.
        let clipped = tile_scissor(&placement(-64.0, -32.0, 128.0), [0.0, 0.0], [256.0, 256.0]);
        assert_eq!(clipped, ScissorRect::new(0, 0, 64, 96));

        // Tile larger than the viewport.
        let covering = tile_scissor(&placement(-10.0, -10.0, 1000.0), [0.0, 0.0], [256.0, 128.0]);
        assert_eq!(covering, ScissorRect::new(0, 0, 256, 128));
    }

    #[test]
    fn offscreen_and_degenerate_tiles_have_no_scissor() {
        assert!(tile_scissor(&placement(300.0, 0.0, 64.0), [0.0, 0.0], [256.0, 256.0]).is_none());
        assert!(tile_scissor(&placement(0.0, -64.0, 64.0), [0.0, 0.0], [256.0, 256.0]).is_none());
        assert!(tile_scissor(&placement(0.0, 0.0, 0.0), [0.0, 0.0], [256.0, 256.0]).is_none());
        assert!(
            tile_scissor(&placement(f32::NAN, 0.0, 64.0), [0.0, 0.0], [256.0, 256.0]).is_none()
        );
        assert!(tile_scissor(&placement(0.0, 0.0, 64.0), [0.0, 0.0], [0.0, 256.0]).is_none());
        assert!(
            tile_scissor(&placement(0.0, 0.0, 64.0), [f32::NAN, 0.0], [256.0, 256.0]).is_none()
        );
    }

    #[test]
    fn adjacent_tiles_never_share_a_pixel_column() {
        // The common case while panning: tiles land on a fractional boundary.
        for offset in [0.0f32, 0.5, 0.25, 0.75, 0.1, 0.999] {
            let left = tile_scissor(
                &placement(100.0 + offset, 0.0, 128.0),
                [0.0, 0.0],
                [512.0, 512.0],
            );
            let right = tile_scissor(
                &placement(228.0 + offset, 0.0, 128.0),
                [0.0, 0.0],
                [512.0, 512.0],
            );
            let (Some(left), Some(right)) = (left, right) else {
                panic!("both tiles are on screen at offset {offset}");
            };
            assert_eq!(
                left.x + left.width,
                right.x,
                "tiles overlap or leave a gap at offset {offset}"
            );
        }

        // Vertically too, and with a fractional clip origin.
        let top = tile_scissor(&placement(0.0, 50.5, 128.0), [4.5, 8.5], [512.0, 512.0]);
        let bottom = tile_scissor(&placement(0.0, 178.5, 128.0), [4.5, 8.5], [512.0, 512.0]);
        let (Some(top), Some(bottom)) = (top, bottom) else {
            panic!("both tiles are on screen");
        };
        assert_eq!(top.y + top.height, bottom.y);
    }

    #[test]
    fn empty_scissor_rectangles_are_rejected() {
        assert!(ScissorRect::new(0, 0, 0, 8).is_none());
        assert!(ScissorRect::new(0, 0, 8, 0).is_none());
        let Some(rect) = ScissorRect::new(1, 2, 3, 4) else {
            panic!("rectangle is not empty");
        };
        assert_eq!((rect.x, rect.y, rect.width, rect.height), (1, 2, 3, 4));
    }
}
