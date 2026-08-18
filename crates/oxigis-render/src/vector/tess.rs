//! CPU tessellation: decoded vector tiles → one triangle mesh per tile.
//!
//! [`tessellate_tile`] walks a [`VectorTile`], asks a [`PaintResolver`] what
//! each layer looks like, and turns the geometry into a single indexed
//! [`VectorMesh`] with `lyon`. The mesh is pure data — no GPU object is touched
//! here — so it can be built on a worker thread (or a web worker) and uploaded
//! later by [`crate::vector::VectorPipeline`].
//!
//! # Coordinate space
//!
//! Input coordinates are tile-local integers on the layer's `extent` grid and
//! may fall outside `0..extent` (MVT buffer geometry, deliberately preserved by
//! the decoder). Tessellation happens in those units — that is what makes
//! [`TessParams::tolerance_tile_units`] meaningful — and only the emitted
//! vertices are divided by `extent`, so [`VectorVertex::position`] is in the
//! unit tile square `0..1` with `y` growing downwards, matching
//! [`crate::viewport::TilePlacement`]. Buffer geometry therefore lands slightly
//! outside `0..1` on purpose; the pipeline scissors it away per tile.
//!
//! # Level of detail, and when to re-tessellate
//!
//! [`TessParams`] carries the two quantities that depend on the camera:
//!
//! * `tolerance_tile_units` — the flattening/simplification budget. Larger at
//!   low zoom, where one tile unit covers a fraction of a pixel, which is what
//!   makes this the LoD knob.
//! * `pixels_per_tile_unit` — converts the pixel widths and radii of
//!   [`crate::vector::LayerPaint`] into tile units at tessellation time.
//!
//! Pixel-denominated quantities — stroke widths, circle radii, circle outlines —
//! are expanded on the CPU but **not** frozen there: every vertex also records
//! the expansion vector that took it off its own centre line in
//! [`VectorVertex::offset`], and the vertex shader rescales that offset by the
//! ratio between [`VectorMesh::baked_tile_size_px`] and the tile size the frame
//! actually draws at (see [`crate::vector::VectorPipeline`]). A mesh therefore
//! keeps constant *pixel* widths at any zoom, fractional zoom included, and
//! `position - offset` is the geometry the width was measured from.
//!
//! The tolerance, by contrast, is genuinely baked: a mesh simplified for z4 is
//! coarse at z14. Re-tessellating on a zoom step is therefore a level-of-detail
//! decision a caller may take at its leisure, no longer a correctness
//! requirement. [`TessParams::for_tile`] derives every camera-dependent field
//! from [`crate::viewport::MapView::tile_size_px`].
//!
//! All of this describes the tile's `extent` grid. MVT tiles use one extent for
//! every layer in practice (4096), but a tile that mixes them is handled
//! per layer: [`TessParams::for_tile`] records the extent it was derived from,
//! and each layer's pixel scale is re-derived from its own extent, so a 512-unit
//! layer next to a 4096-unit one still gets the stroke width its style asked
//! for.
//!
//! # Draw order
//!
//! Layers are emitted in the order the *tile* stores them and features in the
//! order their layer stores them, appending to one index buffer. With depth
//! testing off (see [`crate::vector::VectorPipeline`]) that makes the mesh a
//! painter's-algorithm draw: later features paint over earlier ones. The
//! resolver's own ordering is irrelevant — a shell that needs style order must
//! sort or re-emit its tiles.
//!
//! # Robustness
//!
//! Degenerate input is skipped, never fatal: empty or sub-three-point rings,
//! lines that collapse to a single point, non-finite or non-positive widths and
//! radii, and fully transparent paints all contribute no triangles. Genuine
//! failures — a `lyon` tessellation error, a zero `extent`, a mesh that would
//! overflow the `u32` index space — return [`RenderError::Tessellation`].
//!
//! Tiles arrive from an operator-supplied URL, so hostile geometry is in scope
//! and the decoder's totality is not enough on its own: `lyon`'s fill is
//! `O((n + k) log n)` in the number `k` of self-intersections, and `k` is
//! quadratic for a spiral or a star. Two budgets bound the work, both carried by
//! [`TessParams`]:
//!
//! * `max_feature_points` rejects a feature before a [`Path`] is built for it —
//!   a sanity ceiling on the linear passes.
//! * `max_vertices` is enforced *inside* the geometry builder, so a blow-up
//!   aborts the tessellator on the vertex that crosses the budget instead of
//!   being noticed after `lyon` has already grown the buffers. This is the one
//!   that bounds the quadratic case.
//!
//! Both report [`RenderError::Tessellation`]; the caller drops the tile. The
//! defaults sit well above any legitimate input — a whole local dataset
//! tessellated as a single mesh included — because the point is to bound the
//! unbounded case, not to refuse work.

use lyon::math::Point;
use lyon::path::Path;
use lyon::path::path::Builder as PathBuilder;
use lyon::tessellation::{
    BuffersBuilder, FillGeometryBuilder, FillOptions, FillRule, FillTessellator, FillVertex,
    FillVertexConstructor, GeometryBuilder, GeometryBuilderError, LineCap, LineJoin,
    StrokeGeometryBuilder, StrokeOptions, StrokeTessellator, StrokeVertex, StrokeVertexConstructor,
    TessellationError, VertexBuffers, VertexId,
};

use crate::error::RenderError;
use crate::mvt::{MvtGeometry, MvtPolygon, VectorTile};
use crate::vector::paint::{CirclePaint, FillPaint, LayerPaint, LinePaint, PaintResolver, Rgba8};
use crate::vector::simplify::Simplifier;

/// Fewest segments a circle is approximated with.
pub const MIN_CIRCLE_SEGMENTS: u32 = 8;

/// Most segments a circle is approximated with, whatever its on-screen radius.
///
/// A cap matters because a style can ask for an enormous radius at high zoom
/// and a tile can hold thousands of points; 64 segments is smooth well past any
/// radius a map symbol uses.
pub const MAX_CIRCLE_SEGMENTS: u32 = 64;

/// Largest number of vertices a [`VectorMesh`] can *address*.
///
/// Indices are `u32`, so this is the index-space ceiling and nothing else: at
/// twenty bytes a vertex it is over 80 GB, which no allocator would have
/// reached. The budget that actually bounds memory is
/// [`TessParams::max_vertices`], three orders of magnitude below it. Crossing
/// either is reported as [`RenderError::Tessellation`] instead of silently
/// wrapping.
pub const MAX_MESH_VERTICES: usize = u32::MAX as usize;

/// Smallest tolerance accepted, in tile units — below this `lyon` spends
/// unbounded time flattening.
pub const MIN_TOLERANCE_TILE_UNITS: f32 = 1e-4;

/// One vertex of a tessellated tile.
///
/// `position` is in the unit tile square (`0..1`, `y` down, buffer geometry
/// allowed to exceed it) and `color` is **straight (non-premultiplied) sRGB**
/// with the layer opacity already multiplied into alpha — see
/// [`crate::vector::paint`].
///
/// `offset` is the part of `position` that came from a *pixel* width: half a
/// stroke width, a circle radius, a circle outline. `position - offset` is the
/// centre line or centre point the paint was measured from, and the vertex
/// shader rescales `offset` when the tile is drawn at a size other than the one
/// it was tessellated for. It is `[0.0, 0.0]` for fills, whose geometry is not
/// pixel-denominated at all.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorVertex {
    /// Position in the unit tile square.
    pub position: [f32; 2],
    /// Pixel-width expansion already contained in `position`.
    pub offset: [f32; 2],
    /// Straight sRGB colour, alpha last.
    pub color: [u8; 4],
}

impl VectorVertex {
    /// Creates a vertex whose position owes nothing to a pixel width.
    #[must_use]
    pub const fn new(position: [f32; 2], color: [u8; 4]) -> Self {
        Self {
            position,
            offset: [0.0, 0.0],
            color,
        }
    }

    /// Creates a vertex that a pixel width pushed `offset` away from its centre.
    #[must_use]
    pub const fn expanded(position: [f32; 2], offset: [f32; 2], color: [u8; 4]) -> Self {
        Self {
            position,
            offset,
            color,
        }
    }

    /// The position the pixel-width expansion started from.
    #[must_use]
    pub fn center(&self) -> [f32; 2] {
        [
            self.position[0] - self.offset[0],
            self.position[1] - self.offset[1],
        ]
    }
}

/// An indexed triangle mesh for one tile, ready to upload.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VectorMesh {
    /// Vertices, in emission order.
    pub vertices: Vec<VectorVertex>,
    /// Triangle-list indices into [`VectorMesh::vertices`].
    pub indices: Vec<u32>,
    /// On-screen size of one tile, in physical pixels, that
    /// [`VectorVertex::offset`] was computed for; `0.0` when unknown, which
    /// makes the pipeline draw the offsets exactly as they are.
    ///
    /// [`tessellate_tile`] fills it in from [`TessParams::tile_size_px`], so it
    /// is only zero for meshes built by hand or with parameters that never named
    /// an extent.
    pub baked_tile_size_px: f32,
}

impl VectorMesh {
    /// An empty mesh.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            baked_tile_size_px: 0.0,
        }
    }

    /// Factor [`VectorVertex::offset`] must be multiplied by for the mesh to
    /// keep its pixel widths when one tile covers `tile_size_px` pixels.
    ///
    /// `1.0` — the identity, i.e. draw what was baked — whenever either size is
    /// unusable or the mesh does not know what it was baked for.
    #[must_use]
    pub fn offset_scale_at(&self, tile_size_px: f32) -> f32 {
        offset_scale(self.baked_tile_size_px, tile_size_px)
    }

    /// Whether the mesh would draw nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() || self.vertices.is_empty()
    }

    /// Number of triangles (indices divided by three, rounding down).
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// The triangles as position triples, skipping any out-of-range index.
    ///
    /// Handy for hit-testing and for tests; the tessellator never emits an
    /// out-of-range index.
    pub fn triangles(&self) -> impl Iterator<Item = [[f32; 2]; 3]> + '_ {
        self.indices.chunks_exact(3).filter_map(move |chunk| {
            let a = self.vertices.get(chunk[0] as usize)?;
            let b = self.vertices.get(chunk[1] as usize)?;
            let c = self.vertices.get(chunk[2] as usize)?;
            Some([a.position, b.position, c.position])
        })
    }

    /// Whether any triangle of the mesh covers `position` (unit-square
    /// coordinates), i.e. whether the mesh paints that spot.
    #[must_use]
    pub fn covers(&self, position: [f32; 2]) -> bool {
        self.triangles()
            .any(|triangle| point_in_triangle(position, triangle))
    }
}

/// Camera-dependent tessellation inputs.
///
/// See the module documentation for the re-tessellation contract: a mesh is
/// only valid for the zoom its parameters were derived from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TessParams {
    /// Flattening and simplification budget, in tile units. The LoD knob.
    pub tolerance_tile_units: f32,
    /// Physical pixels covered by one tile unit at the current zoom.
    pub pixels_per_tile_unit: f32,
    /// Layer extent the two fields above were derived from, when one is known.
    ///
    /// [`TessParams::for_tile`] records it so that a layer declaring a
    /// *different* extent can have its own pixel scale derived instead of
    /// inheriting one that belongs to another grid. [`TessParams::new`] leaves
    /// it [`None`], which applies the fields verbatim to every layer.
    pub reference_extent: Option<u32>,
    /// Most vertices one mesh may hold before tessellation is abandoned.
    pub max_vertices: usize,
    /// Most input points one feature may hold before it is rejected.
    pub max_feature_points: usize,
}

impl TessParams {
    /// Default on-screen error budget, in physical pixels, used by
    /// [`TessParams::for_tile`].
    pub const DEFAULT_TOLERANCE_PX: f32 = 0.35;

    /// Default value of [`TessParams::max_vertices`]: four million vertices, or
    /// about 80 MB of [`VectorVertex`].
    ///
    /// Two orders of magnitude above a dense simplified z14 tile, and above
    /// what a whole local dataset tessellated as one mesh reaches, because the
    /// budget exists to turn an unbounded blow-up into a bounded error — not to
    /// refuse work anybody legitimately asked for.
    pub const DEFAULT_MAX_VERTICES: usize = 4_000_000;

    /// Default value of [`TessParams::max_feature_points`].
    ///
    /// The passes this bounds — simplification and path building — are linear,
    /// so the number is a sanity ceiling rather than the work budget:
    /// [`TessParams::max_vertices`] is what bounds `lyon`'s quadratic
    /// intersection search, and it does so while the tessellator runs. A
    /// one-megabyte MVT tile holds at most a quarter of this in a single ring.
    pub const DEFAULT_MAX_FEATURE_POINTS: usize = 4_000_000;

    /// Creates parameters from raw values.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Tessellation`] if either value is not finite or
    /// not positive. `tolerance_tile_units` is raised to
    /// [`MIN_TOLERANCE_TILE_UNITS`] if it is smaller.
    pub fn new(tolerance_tile_units: f32, pixels_per_tile_unit: f32) -> Result<Self, RenderError> {
        if !tolerance_tile_units.is_finite() || tolerance_tile_units <= 0.0 {
            return Err(RenderError::Tessellation(format!(
                "tolerance must be finite and positive, got {tolerance_tile_units}"
            )));
        }
        if !pixels_per_tile_unit.is_finite() || pixels_per_tile_unit <= 0.0 {
            return Err(RenderError::Tessellation(format!(
                "pixels per tile unit must be finite and positive, got {pixels_per_tile_unit}"
            )));
        }
        Ok(Self {
            tolerance_tile_units: tolerance_tile_units.max(MIN_TOLERANCE_TILE_UNITS),
            pixels_per_tile_unit,
            reference_extent: None,
            max_vertices: Self::DEFAULT_MAX_VERTICES,
            max_feature_points: Self::DEFAULT_MAX_FEATURE_POINTS,
        })
    }

    /// Replaces the vertex budget; at least one vertex is always allowed.
    #[must_use]
    pub fn with_max_vertices(mut self, max_vertices: usize) -> Self {
        self.max_vertices = max_vertices.max(1);
        self
    }

    /// Replaces the per-feature input budget; at least one point is always
    /// allowed.
    #[must_use]
    pub fn with_max_feature_points(mut self, max_feature_points: usize) -> Self {
        self.max_feature_points = max_feature_points.max(1);
        self
    }

    /// On-screen size of one whole tile, in physical pixels, or [`None`] when
    /// the parameters were built without naming an extent.
    #[must_use]
    pub fn tile_size_px(&self) -> Option<f32> {
        let extent = self.reference_extent?;
        let size = self.pixels_per_tile_unit * extent as f32;
        size.is_finite().then_some(size)
    }

    /// Derives parameters from the on-screen size of a tile.
    ///
    /// `tile_size_px` is [`crate::viewport::MapView::tile_size_px`], `extent`
    /// the layer extent of the tile (4096 for virtually every encoder) and
    /// `tolerance_px` the acceptable on-screen error, typically
    /// [`TessParams::DEFAULT_TOLERANCE_PX`].
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Tessellation`] if `extent` is zero or any value
    /// is not finite and positive.
    pub fn for_tile(
        tile_size_px: f32,
        extent: u32,
        tolerance_px: f32,
    ) -> Result<Self, RenderError> {
        if extent == 0 {
            return Err(RenderError::Tessellation(
                "layer extent must be positive".to_owned(),
            ));
        }
        if !tile_size_px.is_finite() || tile_size_px <= 0.0 {
            return Err(RenderError::Tessellation(format!(
                "tile size must be finite and positive, got {tile_size_px}"
            )));
        }
        if !tolerance_px.is_finite() || tolerance_px <= 0.0 {
            return Err(RenderError::Tessellation(format!(
                "tolerance must be finite and positive, got {tolerance_px}"
            )));
        }
        let pixels_per_tile_unit = tile_size_px / extent as f32;
        let params = Self::new(tolerance_px / pixels_per_tile_unit, pixels_per_tile_unit)?;
        Ok(Self {
            reference_extent: Some(extent),
            ..params
        })
    }

    /// The tolerance expressed in physical pixels.
    #[must_use]
    pub fn tolerance_px(&self) -> f32 {
        self.tolerance_tile_units * self.pixels_per_tile_unit
    }

    /// Converts a length in physical pixels to tile units.
    #[must_use]
    pub fn px_to_tile_units(&self, px: f32) -> f32 {
        px / self.pixels_per_tile_unit
    }
}

impl Default for TessParams {
    /// One tile unit per pixel with [`TessParams::DEFAULT_TOLERANCE_PX`] of
    /// slack — the identity case, useful for tests.
    fn default() -> Self {
        Self {
            tolerance_tile_units: Self::DEFAULT_TOLERANCE_PX,
            pixels_per_tile_unit: 1.0,
            reference_extent: None,
            max_vertices: Self::DEFAULT_MAX_VERTICES,
            max_feature_points: Self::DEFAULT_MAX_FEATURE_POINTS,
        }
    }
}

/// Tessellates every styled layer of `tile` into one mesh.
///
/// Layers the resolver returns [`None`] for are skipped without being visited
/// further. See the module documentation for the coordinate space, the draw
/// order and the LoD contract.
///
/// Allocates a fresh [`TessScratch`] per call; a caller tessellating more than
/// one tile (or more than one pass over the same tile) should keep one and use
/// [`tessellate_tile_with`] instead.
///
/// # Errors
///
/// Returns [`RenderError::Tessellation`] if a layer declares a zero `extent`,
/// if `lyon` fails on a geometry, or if either budget of [`TessParams`] is
/// exceeded.
pub fn tessellate_tile(
    tile: &VectorTile,
    resolver: &dyn PaintResolver,
    params: &TessParams,
) -> Result<VectorMesh, RenderError> {
    let mut scratch = TessScratch::new();
    tessellate_tile_with(tile, resolver, params, &mut scratch)
}

/// [`tessellate_tile`] with the tessellator state handed in by the caller.
///
/// `lyon`'s tessellators own an event queue and sweep-line buffers that are
/// rebuilt from scratch on every construction; reusing one [`TessScratch`]
/// across tiles and passes keeps those allocations, which is what the two-pass
/// fill/outline programs of the shells do for every tile they draw.
///
/// # Errors
///
/// As [`tessellate_tile`].
pub fn tessellate_tile_with(
    tile: &VectorTile,
    resolver: &dyn PaintResolver,
    params: &TessParams,
    scratch: &mut TessScratch,
) -> Result<VectorMesh, RenderError> {
    let mut mesh = VectorMesh::new();
    tessellate_tile_into(tile, resolver, params, scratch, &mut mesh)?;
    Ok(mesh)
}

/// [`tessellate_tile_with`] appending to a mesh the caller already has.
///
/// This is what a multi-pass program wants — a fill pass and an outline pass
/// over the same tile land in one mesh, in pass order, without the intermediate
/// mesh and the index-shifting copy that concatenating two meshes costs. The
/// vertex budget applies to the combined mesh, as it must.
///
/// On failure `mesh` holds an unspecified prefix of the run: a caller that
/// cannot use a partial tile drops it, which is what every caller does.
///
/// # Errors
///
/// As [`tessellate_tile`].
pub fn tessellate_tile_into(
    tile: &VectorTile,
    resolver: &dyn PaintResolver,
    params: &TessParams,
    scratch: &mut TessScratch,
    mesh: &mut VectorMesh,
) -> Result<(), RenderError> {
    let mut builder = MeshBuilder::adopt(*params, scratch, core::mem::take(mesh));
    let outcome = builder.add_tile(tile, resolver);
    *mesh = builder.finish();
    outcome
}

/// Reusable state for [`tessellate_tile_with`]: `lyon`'s two tessellators and
/// the buffers the simplification passes work in.
///
/// Holds no tile data and no camera state, so one instance serves every tile a
/// provider tessellates, one per thread.
#[derive(Default)]
pub struct TessScratch {
    fill: FillTessellator,
    stroke: StrokeTessellator,
    simplifier: Simplifier,
}

impl TessScratch {
    /// Creates empty tessellator state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fill: FillTessellator::new(),
            stroke: StrokeTessellator::new(),
            simplifier: Simplifier::new(),
        }
    }
}

impl core::fmt::Debug for TessScratch {
    /// `lyon`'s tessellators are opaque, so only the shape of this type is
    /// reported.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("TessScratch")
            .finish_non_exhaustive()
    }
}

/// Per-layer scaling derived from the layer extent and the tessellation
/// parameters.
#[derive(Debug, Clone, Copy)]
struct LayerScale {
    /// Multiplier taking tile units to the unit square.
    inv_extent: f32,
    /// Physical pixels per tile unit.
    pixels_per_tile_unit: f32,
    /// Flattening/simplification budget in tile units.
    tolerance: f32,
}

impl LayerScale {
    fn new(extent: u32, params: TessParams) -> Self {
        // A pixel width is a property of the camera, not of the grid a layer
        // happens to be quantised on: re-derive both camera quantities from
        // this layer's own extent whenever it differs from the one the
        // parameters were built for, or a 512-unit layer beside a 4096-unit one
        // draws its strokes eight times too wide.
        let mut pixels_per_tile_unit = params.pixels_per_tile_unit;
        let mut tolerance = params.tolerance_tile_units;
        if let Some(reference) = params.reference_extent
            && reference != extent
            && reference > 0
        {
            let rescaled = params.pixels_per_tile_unit * reference as f32 / extent as f32;
            if rescaled.is_finite() && rescaled > 0.0 {
                pixels_per_tile_unit = rescaled;
                tolerance = params.tolerance_px() / rescaled;
            }
        }
        Self {
            inv_extent: 1.0 / extent as f32,
            pixels_per_tile_unit,
            tolerance: tolerance.max(MIN_TOLERANCE_TILE_UNITS),
        }
    }

    /// Converts a physical-pixel length to tile units.
    fn px_to_units(&self, px: f32) -> f32 {
        px / self.pixels_per_tile_unit
    }

    /// The flattening budget in physical pixels — extent-independent, so it is
    /// the same number for every layer of a tile.
    fn tolerance_px(&self) -> f32 {
        self.tolerance * self.pixels_per_tile_unit
    }
}

/// Builds [`VectorVertex`] values for `lyon`, applying the layer scale and the
/// feature colour.
#[derive(Debug, Clone, Copy)]
struct VertexCtor {
    inv_extent: f32,
    color: [u8; 4],
}

impl VertexCtor {
    /// A vertex whose position is geometry, not width: no expansion to record.
    fn build(&self, position: Point) -> VectorVertex {
        VectorVertex {
            position: [position.x * self.inv_extent, position.y * self.inv_extent],
            offset: [0.0, 0.0],
            color: self.color,
        }
    }

    /// A vertex a pixel width pushed off `center`, which the shader needs in
    /// order to re-derive the width at the zoom it draws at.
    fn build_expanded(&self, position: Point, center: Point) -> VectorVertex {
        VectorVertex {
            position: [position.x * self.inv_extent, position.y * self.inv_extent],
            offset: [
                (position.x - center.x) * self.inv_extent,
                (position.y - center.y) * self.inv_extent,
            ],
            color: self.color,
        }
    }
}

impl FillVertexConstructor<VectorVertex> for VertexCtor {
    fn new_vertex(&mut self, vertex: FillVertex) -> VectorVertex {
        self.build(vertex.position())
    }
}

impl StrokeVertexConstructor<VectorVertex> for VertexCtor {
    fn new_vertex(&mut self, vertex: StrokeVertex) -> VectorVertex {
        // `position_on_path` is the centre line; the difference is exactly the
        // half-width `lyon` extruded this vertex by.
        self.build_expanded(vertex.position(), vertex.position_on_path())
    }
}

/// [`BuffersBuilder`] with a hard vertex budget, checked as the tessellator
/// emits rather than after it has returned.
///
/// Returning [`GeometryBuilderError::TooManyVertices`] is how `lyon` lets a
/// consumer stop a run: the tessellator abandons the path and propagates the
/// error, so the buffers never grow past the budget in the first place.
struct BudgetedBuilder<'a> {
    inner: BuffersBuilder<'a, VectorVertex, u32, VertexCtor>,
    max_vertices: usize,
}

impl<'a> BudgetedBuilder<'a> {
    fn new(
        buffers: &'a mut VertexBuffers<VectorVertex, u32>,
        ctor: VertexCtor,
        max_vertices: usize,
    ) -> Self {
        Self {
            inner: BuffersBuilder::new(buffers, ctor),
            max_vertices,
        }
    }

    fn check(&self) -> Result<(), GeometryBuilderError> {
        if self.inner.buffers().vertices.len() >= self.max_vertices {
            return Err(GeometryBuilderError::TooManyVertices);
        }
        Ok(())
    }
}

impl GeometryBuilder for BudgetedBuilder<'_> {
    fn begin_geometry(&mut self) {
        self.inner.begin_geometry();
    }

    fn end_geometry(&mut self) {
        self.inner.end_geometry();
    }

    fn add_triangle(&mut self, a: VertexId, b: VertexId, c: VertexId) {
        self.inner.add_triangle(a, b, c);
    }

    fn abort_geometry(&mut self) {
        self.inner.abort_geometry();
    }
}

impl FillGeometryBuilder for BudgetedBuilder<'_> {
    fn add_fill_vertex(&mut self, vertex: FillVertex) -> Result<VertexId, GeometryBuilderError> {
        self.check()?;
        self.inner.add_fill_vertex(vertex)
    }
}

impl StrokeGeometryBuilder for BudgetedBuilder<'_> {
    fn add_stroke_vertex(
        &mut self,
        vertex: StrokeVertex,
    ) -> Result<VertexId, GeometryBuilderError> {
        self.check()?;
        self.inner.add_stroke_vertex(vertex)
    }
}

/// Accumulates one tile's triangles.
struct MeshBuilder<'a> {
    buffers: VertexBuffers<VectorVertex, u32>,
    scratch: &'a mut TessScratch,
    params: TessParams,
}

impl<'a> MeshBuilder<'a> {
    /// Takes over an existing mesh, so a second pass appends to the first.
    fn adopt(params: TessParams, scratch: &'a mut TessScratch, mesh: VectorMesh) -> Self {
        Self {
            buffers: VertexBuffers {
                vertices: mesh.vertices,
                indices: mesh.indices,
            },
            scratch,
            params,
        }
    }

    fn finish(self) -> VectorMesh {
        VectorMesh {
            vertices: self.buffers.vertices,
            indices: self.buffers.indices,
            baked_tile_size_px: self.params.tile_size_px().unwrap_or(0.0),
        }
    }

    /// Emits every styled layer of `tile`, in the order the tile stores them.
    fn add_tile(
        &mut self,
        tile: &VectorTile,
        resolver: &dyn PaintResolver,
    ) -> Result<(), RenderError> {
        for layer in &tile.layers {
            let Some(paint) = resolver.paint_for(&layer.name) else {
                continue;
            };
            if layer.extent == 0 {
                return Err(RenderError::Tessellation(format!(
                    "layer {} declares a zero extent",
                    layer.name
                )));
            }
            let scale = LayerScale::new(layer.extent, self.params);
            for feature in &layer.features {
                self.add_feature(&feature.geometry, &paint, &scale)?;
            }
        }
        Ok(())
    }

    /// Dispatches one feature on the (geometry, paint) pair; combinations with
    /// no rule (a fill paint on a point layer, say) contribute nothing.
    fn add_feature(
        &mut self,
        geometry: &MvtGeometry,
        paint: &LayerPaint,
        scale: &LayerScale,
    ) -> Result<(), RenderError> {
        self.check_budget(geometry)?;
        match (geometry, paint) {
            (MvtGeometry::Polygons(polygons), LayerPaint::Fill(fill)) => {
                self.fill_polygons(polygons, fill, scale)
            }
            (MvtGeometry::Polygons(polygons), LayerPaint::Line(line)) => {
                self.stroke_rings(polygons, line, scale)
            }
            (MvtGeometry::Lines(lines), LayerPaint::Line(line)) => {
                self.stroke_lines(lines, line, scale)
            }
            (MvtGeometry::Points(points), LayerPaint::Circle(circle)) => {
                self.add_circles(points, circle, scale)
            }
            _ => Ok(()),
        }
    }

    /// Rejects a feature before a [`Path`] is built for it, so neither budget
    /// is discovered after the allocation it was meant to prevent.
    fn check_budget(&self, geometry: &MvtGeometry) -> Result<(), RenderError> {
        let points = geometry_points(geometry);
        if points > self.params.max_feature_points {
            return Err(RenderError::Tessellation(format!(
                "feature holds {points} points, more than the {} budgeted",
                self.params.max_feature_points
            )));
        }
        if self.buffers.vertices.len() >= self.params.max_vertices {
            return Err(RenderError::Tessellation(format!(
                "tile mesh reached its budget of {} vertices",
                self.params.max_vertices
            )));
        }
        Ok(())
    }

    /// Reserves room for `count` hand-built vertices, returning the index the
    /// first of them will take.
    fn reserve(&mut self, count: usize) -> Result<u32, RenderError> {
        let base = check_mesh_capacity(self.buffers.vertices.len())?;
        let total = self.buffers.vertices.len().saturating_add(count);
        if total > self.params.max_vertices {
            return Err(RenderError::Tessellation(format!(
                "mesh would need {total} vertices, more than the {} budgeted",
                self.params.max_vertices
            )));
        }
        check_mesh_capacity(total)?;
        Ok(base)
    }

    fn fill_polygons(
        &mut self,
        polygons: &[MvtPolygon],
        paint: &FillPaint,
        scale: &LayerScale,
    ) -> Result<(), RenderError> {
        let color = paint.resolved_color();
        if color.is_invisible() {
            return Ok(());
        }

        let mut path = Path::builder();
        let mut rings = 0usize;
        for polygon in polygons {
            // Rings keep their decoded winding: exteriors are positive-area in
            // this y-down space and interiors negative, so the non-zero fill
            // rule cancels holes out without this code having to know which
            // way lyon considers "clockwise".
            if self.add_ring(&mut path, &polygon.exterior, scale.tolerance) {
                rings += 1;
            }
            for interior in &polygon.interiors {
                if self.add_ring(&mut path, interior, scale.tolerance) {
                    rings += 1;
                }
            }
        }
        let path = path.build();
        if rings == 0 {
            return Ok(());
        }

        let options = FillOptions::tolerance(scale.tolerance).with_fill_rule(FillRule::NonZero);
        let ctor = VertexCtor {
            inv_extent: scale.inv_extent,
            color: color.to_array(),
        };
        let max_vertices = self.params.max_vertices;
        let mut output = BudgetedBuilder::new(&mut self.buffers, ctor, max_vertices);
        self.scratch
            .fill
            .tessellate_path(&path, &options, &mut output)
            .map_err(|error| tessellation_error("fill", &error, max_vertices))?;
        check_mesh_capacity(self.buffers.vertices.len()).map(|_| ())
    }

    /// Adds one MVT ring (stored unclosed) to `path` as a closed sub-path,
    /// simplified with the very pass the outline of the same ring uses.
    ///
    /// Returns whether anything was added: rings left with fewer than three
    /// points cannot bound an area and are skipped.
    fn add_ring(&mut self, path: &mut PathBuilder, ring: &[[i32; 2]], tolerance: f32) -> bool {
        let simplified = self.scratch.simplifier.closed(ring, tolerance);
        if simplified.len() < 3 {
            return false;
        }
        path.begin(simplified[0]);
        for position in &simplified[1..] {
            path.line_to(*position);
        }
        path.end(true);
        true
    }

    fn stroke_lines(
        &mut self,
        lines: &[Vec<[i32; 2]>],
        paint: &LinePaint,
        scale: &LayerScale,
    ) -> Result<(), RenderError> {
        let Some((options, ctor)) = self.stroke_setup(paint, scale) else {
            return Ok(());
        };

        let mut path = Path::builder();
        let mut strokes = 0usize;
        for line in lines {
            let simplified = self.scratch.simplifier.open(line, scale.tolerance);
            if simplified.len() < 2 {
                continue;
            }
            path.begin(simplified[0]);
            for position in &simplified[1..] {
                path.line_to(*position);
            }
            path.end(false);
            strokes += 1;
        }
        let path = path.build();
        if strokes == 0 {
            return Ok(());
        }
        self.run_stroke(&path, &options, ctor)
    }

    fn stroke_rings(
        &mut self,
        polygons: &[MvtPolygon],
        paint: &LinePaint,
        scale: &LayerScale,
    ) -> Result<(), RenderError> {
        let Some((options, ctor)) = self.stroke_setup(paint, scale) else {
            return Ok(());
        };

        let mut path = Path::builder();
        let mut strokes = 0usize;
        for polygon in polygons {
            for ring in core::iter::once(&polygon.exterior).chain(polygon.interiors.iter()) {
                if self.add_ring(&mut path, ring, scale.tolerance) {
                    strokes += 1;
                }
            }
        }
        let path = path.build();
        if strokes == 0 {
            return Ok(());
        }
        self.run_stroke(&path, &options, ctor)
    }

    /// Shared stroke options/colour, or [`None`] when the paint draws nothing.
    fn stroke_setup(
        &self,
        paint: &LinePaint,
        scale: &LayerScale,
    ) -> Option<(StrokeOptions, VertexCtor)> {
        let color = paint.resolved_color();
        if color.is_invisible() {
            return None;
        }
        let width = scale.px_to_units(paint.width_px);
        if !width.is_finite() || width <= 0.0 {
            return None;
        }
        let options = StrokeOptions::tolerance(scale.tolerance)
            .with_line_width(width)
            .with_line_join(LineJoin::Round)
            .with_line_cap(LineCap::Round);
        Some((
            options,
            VertexCtor {
                inv_extent: scale.inv_extent,
                color: color.to_array(),
            },
        ))
    }

    fn run_stroke(
        &mut self,
        path: &Path,
        options: &StrokeOptions,
        ctor: VertexCtor,
    ) -> Result<(), RenderError> {
        let max_vertices = self.params.max_vertices;
        let mut output = BudgetedBuilder::new(&mut self.buffers, ctor, max_vertices);
        self.scratch
            .stroke
            .tessellate_path(path, options, &mut output)
            .map_err(|error| tessellation_error("stroke", &error, max_vertices))?;
        check_mesh_capacity(self.buffers.vertices.len()).map(|_| ())
    }

    fn add_circles(
        &mut self,
        points: &[[i32; 2]],
        paint: &CirclePaint,
        scale: &LayerScale,
    ) -> Result<(), RenderError> {
        let radius = scale.px_to_units(paint.radius_px);
        if !radius.is_finite() || radius <= 0.0 {
            return Ok(());
        }
        let segments = circle_segments(paint.radius_px, scale.tolerance_px());
        let disc = paint.resolved_color();
        let outline = paint.resolved_stroke_color().filter(|color| {
            !color.is_invisible()
                && paint.stroke_width_px.is_finite()
                && paint.stroke_width_px > 0.0
        });
        if disc.is_invisible() && outline.is_none() {
            return Ok(());
        }
        let stroke_width = scale.px_to_units(paint.stroke_width_px);

        for position in points {
            let center = [position[0] as f32, position[1] as f32];
            if !disc.is_invisible() {
                self.push_disc(center, radius, segments, disc, scale)?;
            }
            if let Some(color) = outline {
                self.push_ring(
                    center,
                    radius,
                    radius + stroke_width,
                    segments,
                    color,
                    scale,
                )?;
            }
        }
        Ok(())
    }

    /// Emits a triangle fan approximating a disc.
    fn push_disc(
        &mut self,
        center: [f32; 2],
        radius: f32,
        segments: u32,
        color: Rgba8,
        scale: &LayerScale,
    ) -> Result<(), RenderError> {
        let base = self.reserve(segments as usize + 1)?;
        let color = color.to_array();
        self.buffers.vertices.push(VectorVertex {
            position: [center[0] * scale.inv_extent, center[1] * scale.inv_extent],
            offset: [0.0, 0.0],
            color,
        });
        for index in 0..segments {
            let (x, y) = circle_offset(index, segments);
            // The radius is a pixel quantity, so the rim carries it as an
            // offset the shader can re-derive at another zoom.
            self.buffers.vertices.push(VectorVertex {
                position: [
                    (center[0] + x * radius) * scale.inv_extent,
                    (center[1] + y * radius) * scale.inv_extent,
                ],
                offset: [x * radius * scale.inv_extent, y * radius * scale.inv_extent],
                color,
            });
        }
        for index in 0..segments {
            let next = (index + 1) % segments;
            self.buffers
                .indices
                .extend_from_slice(&[base, base + 1 + index, base + 1 + next]);
        }
        Ok(())
    }

    /// Emits an annulus between `inner` and `outer` radius (the circle
    /// outline).
    fn push_ring(
        &mut self,
        center: [f32; 2],
        inner: f32,
        outer: f32,
        segments: u32,
        color: Rgba8,
        scale: &LayerScale,
    ) -> Result<(), RenderError> {
        let base = self.reserve((segments as usize) * 2)?;
        let color = color.to_array();
        for index in 0..segments {
            let (x, y) = circle_offset(index, segments);
            for radius in [inner, outer] {
                self.buffers.vertices.push(VectorVertex {
                    position: [
                        (center[0] + x * radius) * scale.inv_extent,
                        (center[1] + y * radius) * scale.inv_extent,
                    ],
                    offset: [x * radius * scale.inv_extent, y * radius * scale.inv_extent],
                    color,
                });
            }
        }
        for index in 0..segments {
            let next = (index + 1) % segments;
            let (a, b) = (base + index * 2, base + index * 2 + 1);
            let (c, d) = (base + next * 2, base + next * 2 + 1);
            self.buffers.indices.extend_from_slice(&[a, b, d, a, d, c]);
        }
        Ok(())
    }
}

/// Number of input points a feature's geometry holds, whatever its kind.
fn geometry_points(geometry: &MvtGeometry) -> usize {
    match geometry {
        MvtGeometry::Points(points) => points.len(),
        MvtGeometry::Lines(lines) => lines.iter().map(Vec::len).sum(),
        MvtGeometry::Polygons(polygons) => polygons
            .iter()
            .map(|polygon| {
                polygon.exterior.len() + polygon.interiors.iter().map(Vec::len).sum::<usize>()
            })
            .sum(),
    }
}

/// Turns a `lyon` failure into a [`RenderError`], naming the budget when the
/// budget is what stopped the run.
fn tessellation_error(stage: &str, error: &TessellationError, max_vertices: usize) -> RenderError {
    match error {
        TessellationError::GeometryBuilder(GeometryBuilderError::TooManyVertices) => {
            RenderError::Tessellation(format!(
                "{stage}: geometry needs more than the {max_vertices} vertices budgeted"
            ))
        }
        other => RenderError::Tessellation(format!("{stage}: {other}")),
    }
}

/// Widest correction [`offset_scale`] applies, in either direction.
///
/// Ten zoom levels of drift, which no camera reaches before its tiles have been
/// re-tessellated; the clamp only exists so that a nonsensical pair of sizes
/// cannot inflate a hairline into a screenful.
pub const MAX_OFFSET_SCALE: f32 = 1024.0;

/// Factor a mesh's [`VectorVertex::offset`] must be multiplied by to keep its
/// pixel widths when one tile covers `tile_size_px` pixels on screen.
///
/// Returns `1.0` — draw what was baked — whenever either size is unusable,
/// which is also how a mesh that never recorded its tile size is handled.
#[must_use]
pub fn offset_scale(baked_tile_size_px: f32, tile_size_px: f32) -> f32 {
    if !baked_tile_size_px.is_finite()
        || baked_tile_size_px <= 0.0
        || !tile_size_px.is_finite()
        || tile_size_px <= 0.0
    {
        return 1.0;
    }
    (baked_tile_size_px / tile_size_px).clamp(1.0 / MAX_OFFSET_SCALE, MAX_OFFSET_SCALE)
}

/// Segment count for a circle of `radius_px` drawn within `tolerance_px`.
///
/// Derived from the sagitta of one segment and clamped to
/// [`MIN_CIRCLE_SEGMENTS`]`..=`[`MAX_CIRCLE_SEGMENTS`].
#[must_use]
pub fn circle_segments(radius_px: f32, tolerance_px: f32) -> u32 {
    if !radius_px.is_finite() || radius_px <= 0.0 {
        return MIN_CIRCLE_SEGMENTS;
    }
    if !tolerance_px.is_finite() || tolerance_px <= 0.0 {
        return MAX_CIRCLE_SEGMENTS;
    }
    if tolerance_px >= radius_px {
        return MIN_CIRCLE_SEGMENTS;
    }
    let ratio = (1.0 - tolerance_px / radius_px).clamp(-1.0, 1.0);
    let angle = ratio.acos();
    if angle <= 0.0 {
        return MAX_CIRCLE_SEGMENTS;
    }
    let segments = (core::f32::consts::PI / angle).ceil();
    if !segments.is_finite() || segments >= MAX_CIRCLE_SEGMENTS as f32 {
        return MAX_CIRCLE_SEGMENTS;
    }
    (segments as u32).clamp(MIN_CIRCLE_SEGMENTS, MAX_CIRCLE_SEGMENTS)
}

/// Unit-circle offset of segment `index` out of `segments`.
fn circle_offset(index: u32, segments: u32) -> (f32, f32) {
    let step = core::f32::consts::TAU / segments.max(1) as f32;
    let angle = step * index as f32;
    (angle.cos(), angle.sin())
}

/// Checks that `vertex_count` still fits the `u32` index space, returning it as
/// a `u32`.
///
/// # Errors
///
/// Returns [`RenderError::Tessellation`] above [`MAX_MESH_VERTICES`]; the mesh
/// is never truncated, because a truncated index buffer draws garbage.
pub fn check_mesh_capacity(vertex_count: usize) -> Result<u32, RenderError> {
    u32::try_from(vertex_count).map_err(|_| {
        RenderError::Tessellation(format!(
            "mesh needs {vertex_count} vertices, more than the {MAX_MESH_VERTICES} a u32 index can address"
        ))
    })
}

/// Whether `position` lies inside (or on the edge of) `triangle`.
fn point_in_triangle(position: [f32; 2], triangle: [[f32; 2]; 3]) -> bool {
    let [a, b, c] = triangle;
    let cross = |from: [f32; 2], to: [f32; 2]| {
        (to[0] - from[0]) * (position[1] - from[1]) - (to[1] - from[1]) * (position[0] - from[0])
    };
    let (ab, bc, ca) = (cross(a, b), cross(b, c), cross(c, a));
    let negative = ab < 0.0 || bc < 0.0 || ca < 0.0;
    let positive = ab > 0.0 || bc > 0.0 || ca > 0.0;
    !(negative && positive)
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_CIRCLE_SEGMENTS, MAX_MESH_VERTICES, MAX_OFFSET_SCALE, MIN_CIRCLE_SEGMENTS, TessParams,
        TessScratch, VectorMesh, VectorVertex, check_mesh_capacity, circle_segments, offset_scale,
        tessellate_tile, tessellate_tile_into, tessellate_tile_with,
    };
    use crate::error::RenderError;
    use crate::mvt::{MvtFeature, MvtGeometry, MvtLayer, MvtPolygon, VectorTile};
    use crate::vector::paint::{
        CirclePaint, FillPaint, LinePaint, PaintResolver, PaintTable, Rgba8,
    };

    const EXTENT: u32 = 4096;

    fn params(pixels_per_tile_unit: f32, tolerance: f32) -> TessParams {
        match TessParams::new(tolerance, pixels_per_tile_unit) {
            Ok(params) => params,
            Err(error) => panic!("params rejected: {error}"),
        }
    }

    fn feature(geometry: MvtGeometry) -> MvtFeature {
        MvtFeature {
            id: None,
            properties: Vec::new(),
            geometry,
        }
    }

    fn tile(name: &str, geometry: MvtGeometry) -> VectorTile {
        VectorTile {
            layers: vec![MvtLayer {
                name: name.to_owned(),
                extent: EXTENT,
                features: vec![feature(geometry)],
            }],
        }
    }

    fn square(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<[i32; 2]> {
        // Positive shoelace area in y-down space: the decoder's exterior rule.
        vec![[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
    }

    fn hole(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<[i32; 2]> {
        // Opposite winding: an interior ring.
        vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    }

    /// A convex ring of `points` samples on a circle centred in the tile — the
    /// densely sampled geometry the LoD pass exists for.
    fn dense_circle(points: u32, radius: f32) -> Vec<[i32; 2]> {
        (0..points)
            .map(|index| {
                let angle = core::f32::consts::TAU * index as f32 / points as f32;
                [
                    (2048.0 + radius * angle.cos()) as i32,
                    (2048.0 + radius * angle.sin()) as i32,
                ]
            })
            .collect()
    }

    fn close(left: [f32; 2], right: [f32; 2], epsilon: f32) -> bool {
        (left[0] - right[0]).abs() <= epsilon && (left[1] - right[1]).abs() <= epsilon
    }

    fn mesh_of(tile: &VectorTile, table: &PaintTable, params: &TessParams) -> VectorMesh {
        let resolver: &dyn PaintResolver = table;
        match tessellate_tile(tile, resolver, params) {
            Ok(mesh) => mesh,
            Err(error) => panic!("tessellation failed: {error}"),
        }
    }

    fn bounds(mesh: &VectorMesh) -> ([f32; 2], [f32; 2]) {
        let mut min = [f32::INFINITY; 2];
        let mut max = [f32::NEG_INFINITY; 2];
        for vertex in &mesh.vertices {
            for axis in 0..2 {
                min[axis] = min[axis].min(vertex.position[axis]);
                max[axis] = max[axis].max(vertex.position[axis]);
            }
        }
        (min, max)
    }

    #[test]
    fn vertex_layout_is_tight() {
        assert_eq!(core::mem::size_of::<VectorVertex>(), 20);
        assert_eq!(core::mem::align_of::<VectorVertex>(), 4);
        let vertices = [
            VectorVertex::new([0.0, 1.0], [1, 2, 3, 4]),
            VectorVertex::expanded([0.5, 0.25], [0.0, 0.125], [5, 6, 7, 8]),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&vertices);
        assert_eq!(bytes.len(), 40);
        let round_trip: &[VectorVertex] = bytemuck::cast_slice(bytes);
        assert_eq!(round_trip, vertices.as_slice());
        assert_eq!(vertices[0].offset, [0.0, 0.0]);
        assert_eq!(vertices[0].center(), [0.0, 1.0]);
        assert_eq!(vertices[1].center(), [0.5, 0.125]);
    }

    #[test]
    fn an_empty_tile_makes_an_empty_mesh() {
        let table = PaintTable::new().with("water", FillPaint::new(Rgba8::WHITE));
        let mesh = mesh_of(&VectorTile::default(), &table, &TessParams::default());
        assert!(mesh.is_empty());
        assert_eq!(mesh.triangle_count(), 0);
        assert_eq!(mesh.triangles().count(), 0);
    }

    #[test]
    fn unstyled_layers_are_skipped() {
        let tile = tile(
            "water",
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: square(0, 0, 1024, 1024),
                interiors: Vec::new(),
            }]),
        );
        let empty = PaintTable::new();
        assert!(mesh_of(&tile, &empty, &TessParams::default()).is_empty());

        let styled = PaintTable::new().with("water", FillPaint::new(Rgba8::WHITE));
        assert!(!mesh_of(&tile, &styled, &TessParams::default()).is_empty());
    }

    #[test]
    fn a_square_fill_covers_the_square() {
        let tile = tile(
            "water",
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: square(0, 0, 2048, 2048),
                interiors: Vec::new(),
            }]),
        );
        let table = PaintTable::new().with("water", FillPaint::new(Rgba8::opaque(1, 2, 3)));
        let mesh = mesh_of(&tile, &table, &TessParams::default());

        assert!(mesh.triangle_count() >= 2, "{}", mesh.triangle_count());
        let (min, max) = bounds(&mesh);
        assert!(min[0] >= -1e-6 && min[1] >= -1e-6, "{min:?}");
        assert!(max[0] <= 0.5 + 1e-6 && max[1] <= 0.5 + 1e-6, "{max:?}");
        for triangle in mesh.triangles() {
            for corner in triangle {
                assert!((-1e-6..=0.5 + 1e-6).contains(&corner[0]), "{corner:?}");
                assert!((-1e-6..=0.5 + 1e-6).contains(&corner[1]), "{corner:?}");
            }
        }
        assert!(mesh.covers([0.25, 0.25]));
        assert!(!mesh.covers([0.75, 0.75]));
        assert!(mesh.vertices.iter().all(|v| v.color == [1, 2, 3, 255]));
    }

    #[test]
    fn a_hole_stays_empty() {
        let tile = tile(
            "land",
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: square(0, 0, 4096, 4096),
                interiors: vec![hole(1024, 1024, 3072, 3072)],
            }]),
        );
        let table = PaintTable::new().with("land", FillPaint::new(Rgba8::WHITE));
        let mesh = mesh_of(&tile, &table, &TessParams::default());

        // The centre falls inside the hole, the corner in the solid part.
        assert!(!mesh.covers([0.5, 0.5]), "the hole was filled in");
        assert!(mesh.covers([0.1, 0.1]), "the ring itself was not filled");
        assert!(mesh.covers([0.9, 0.5]));
    }

    #[test]
    fn stroke_width_follows_pixels_per_tile_unit() {
        let tile = tile(
            "roads",
            MvtGeometry::Lines(vec![vec![[0, 2048], [4096, 2048]]]),
        );
        let table =
            PaintTable::new().with("roads", LinePaint::new(Rgba8::BLACK).with_width_px(8.0));

        let thick = mesh_of(&tile, &table, &params(1.0, 0.5));
        let thin = mesh_of(&tile, &table, &params(2.0, 0.5));

        let height = |mesh: &VectorMesh| {
            let (min, max) = bounds(mesh);
            max[1] - min[1]
        };
        // 8 px at 1 px per tile unit is 8 tile units; at 2 px per unit, 4.
        let expected_thick = 8.0 / EXTENT as f32;
        assert!(
            (height(&thick) - expected_thick).abs() < 1e-5,
            "{:?}",
            height(&thick)
        );
        assert!(
            (height(&thick) / height(&thin) - 2.0).abs() < 1e-3,
            "{} vs {}",
            height(&thick),
            height(&thin)
        );
    }

    #[test]
    fn a_coarser_tolerance_sheds_vertices() {
        // A zig-zag of many short segments: the LoD case.
        let mut line = Vec::new();
        for index in 0..400i32 {
            line.push([index * 8, 2048 + (index % 2) * 6]);
        }
        let tile = tile("roads", MvtGeometry::Lines(vec![line]));
        let table =
            PaintTable::new().with("roads", LinePaint::new(Rgba8::BLACK).with_width_px(2.0));

        let fine = mesh_of(&tile, &table, &params(1.0, 0.25));
        let coarse = mesh_of(&tile, &table, &params(1.0, 32.0));
        assert!(
            coarse.vertices.len() * 2 < fine.vertices.len(),
            "coarse {} vs fine {}",
            coarse.vertices.len(),
            fine.vertices.len()
        );
        assert!(!coarse.is_empty());
    }

    #[test]
    fn circle_segment_counts_are_capped() {
        assert_eq!(circle_segments(0.5, 1.0), MIN_CIRCLE_SEGMENTS);
        assert_eq!(circle_segments(1e9, 0.01), MAX_CIRCLE_SEGMENTS);
        assert_eq!(circle_segments(f32::NAN, 0.5), MIN_CIRCLE_SEGMENTS);
        assert_eq!(circle_segments(10.0, f32::NAN), MAX_CIRCLE_SEGMENTS);
        let mid = circle_segments(8.0, 0.35);
        assert!(
            (MIN_CIRCLE_SEGMENTS..=MAX_CIRCLE_SEGMENTS).contains(&mid),
            "{mid}"
        );

        let tile = tile("poi", MvtGeometry::Points(vec![[2048, 2048]]));
        let table = PaintTable::new().with(
            "poi",
            CirclePaint::new(Rgba8::WHITE).with_radius_px(100_000.0),
        );
        let mesh = mesh_of(&tile, &table, &params(1.0, 0.01));
        assert_eq!(mesh.vertices.len(), MAX_CIRCLE_SEGMENTS as usize + 1);
        assert_eq!(mesh.triangle_count(), MAX_CIRCLE_SEGMENTS as usize);
        assert!(mesh.covers([0.5, 0.5]));
    }

    #[test]
    fn a_circle_outline_adds_a_ring() {
        let tile = tile("poi", MvtGeometry::Points(vec![[2048, 2048]]));
        let plain =
            PaintTable::new().with("poi", CirclePaint::new(Rgba8::WHITE).with_radius_px(64.0));
        let stroked = PaintTable::new().with(
            "poi",
            CirclePaint::new(Rgba8::WHITE)
                .with_radius_px(64.0)
                .with_stroke(Rgba8::BLACK, 16.0),
        );
        let params = params(1.0, 0.5);
        let plain = mesh_of(&tile, &plain, &params);
        let stroked = mesh_of(&tile, &stroked, &params);
        assert!(stroked.vertices.len() > plain.vertices.len());
        // The outline sits outside the disc.
        let (_, plain_max) = bounds(&plain);
        let (_, stroked_max) = bounds(&stroked);
        assert!(stroked_max[0] > plain_max[0]);
        assert!(stroked.vertices.iter().any(|v| v.color == [0, 0, 0, 255]));
    }

    #[test]
    fn opacity_lands_in_the_vertex_alpha() {
        let tile = tile(
            "water",
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: square(0, 0, 1024, 1024),
                interiors: Vec::new(),
            }]),
        );
        let table = PaintTable::new().with(
            "water",
            FillPaint::new(Rgba8::opaque(9, 9, 9)).with_opacity(0.5),
        );
        let mesh = mesh_of(&tile, &table, &TessParams::default());
        assert!(!mesh.is_empty());
        assert!(mesh.vertices.iter().all(|v| v.color == [9, 9, 9, 128]));

        // A fully transparent paint draws nothing at all.
        let invisible = PaintTable::new().with(
            "water",
            FillPaint::new(Rgba8::opaque(9, 9, 9)).with_opacity(0.0),
        );
        assert!(mesh_of(&tile, &invisible, &TessParams::default()).is_empty());
    }

    #[test]
    fn degenerate_geometry_is_skipped_not_fatal() {
        let table = PaintTable::new()
            .with("water", FillPaint::new(Rgba8::WHITE))
            .with("roads", LinePaint::new(Rgba8::BLACK))
            .with("poi", CirclePaint::new(Rgba8::WHITE).with_radius_px(0.0));

        let ring = tile(
            "water",
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: vec![[0, 0], [10, 10]],
                interiors: vec![Vec::new()],
            }]),
        );
        assert!(mesh_of(&ring, &table, &TessParams::default()).is_empty());

        let line = tile("roads", MvtGeometry::Lines(vec![vec![[5, 5]], Vec::new()]));
        assert!(mesh_of(&line, &table, &TessParams::default()).is_empty());

        let points = tile("poi", MvtGeometry::Points(vec![[1, 1]]));
        assert!(mesh_of(&points, &table, &TessParams::default()).is_empty());
    }

    #[test]
    fn mismatched_paint_and_geometry_draw_nothing() {
        let tile = tile("poi", MvtGeometry::Points(vec![[100, 100]]));
        let table = PaintTable::new().with("poi", FillPaint::new(Rgba8::WHITE));
        assert!(mesh_of(&tile, &table, &TessParams::default()).is_empty());
    }

    #[test]
    fn a_line_paint_outlines_polygons() {
        let tile = tile(
            "buildings",
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: square(1024, 1024, 3072, 3072),
                interiors: Vec::new(),
            }]),
        );
        let table =
            PaintTable::new().with("buildings", LinePaint::new(Rgba8::BLACK).with_width_px(4.0));
        let mesh = mesh_of(&tile, &table, &params(1.0, 0.5));
        assert!(!mesh.is_empty());
        // Only the ring is painted, not the interior.
        assert!(!mesh.covers([0.5, 0.5]));
        assert!(mesh.covers([0.25, 0.5]));
    }

    #[test]
    fn draw_order_follows_the_tile() {
        let tile = VectorTile {
            layers: vec![
                MvtLayer {
                    name: "under".to_owned(),
                    extent: EXTENT,
                    features: vec![feature(MvtGeometry::Polygons(vec![MvtPolygon {
                        exterior: square(0, 0, 2048, 2048),
                        interiors: Vec::new(),
                    }]))],
                },
                MvtLayer {
                    name: "over".to_owned(),
                    extent: EXTENT,
                    features: vec![feature(MvtGeometry::Polygons(vec![MvtPolygon {
                        exterior: square(0, 0, 1024, 1024),
                        interiors: Vec::new(),
                    }]))],
                },
            ],
        };
        // The table lists them the other way round on purpose.
        let table = PaintTable::new()
            .with("over", FillPaint::new(Rgba8::opaque(2, 2, 2)))
            .with("under", FillPaint::new(Rgba8::opaque(1, 1, 1)));
        let mesh = mesh_of(&tile, &table, &TessParams::default());
        let Some(first) = mesh.vertices.first() else {
            panic!("mesh is empty");
        };
        let Some(last) = mesh.vertices.last() else {
            panic!("mesh is empty");
        };
        assert_eq!(first.color, [1, 1, 1, 255]);
        assert_eq!(last.color, [2, 2, 2, 255]);
    }

    #[test]
    fn a_zero_extent_layer_is_an_error() {
        let tile = VectorTile {
            layers: vec![MvtLayer {
                name: "water".to_owned(),
                extent: 0,
                features: Vec::new(),
            }],
        };
        let table = PaintTable::new().with("water", FillPaint::new(Rgba8::WHITE));
        let resolver: &dyn PaintResolver = &table;
        assert!(matches!(
            tessellate_tile(&tile, resolver, &TessParams::default()),
            Err(RenderError::Tessellation(_))
        ));
    }

    #[test]
    fn the_index_space_guard_rejects_oversized_meshes() {
        assert_eq!(check_mesh_capacity(0).ok(), Some(0));
        assert_eq!(check_mesh_capacity(MAX_MESH_VERTICES).ok(), Some(u32::MAX));
        assert!(matches!(
            check_mesh_capacity(MAX_MESH_VERTICES + 1),
            Err(RenderError::Tessellation(_))
        ));
    }

    #[test]
    fn params_validate_and_convert() {
        assert!(matches!(
            TessParams::new(0.0, 1.0),
            Err(RenderError::Tessellation(_))
        ));
        assert!(matches!(
            TessParams::new(1.0, f32::NAN),
            Err(RenderError::Tessellation(_))
        ));
        assert!(matches!(
            TessParams::for_tile(256.0, 0, 0.5),
            Err(RenderError::Tessellation(_))
        ));
        assert!(matches!(
            TessParams::for_tile(0.0, 4096, 0.5),
            Err(RenderError::Tessellation(_))
        ));
        assert!(matches!(
            TessParams::for_tile(256.0, 4096, -1.0),
            Err(RenderError::Tessellation(_))
        ));

        let Ok(derived) = TessParams::for_tile(512.0, EXTENT, 0.5) else {
            panic!("params rejected");
        };
        assert!((derived.pixels_per_tile_unit - 0.125).abs() < 1e-6);
        assert!((derived.tolerance_px() - 0.5).abs() < 1e-4);
        assert!((derived.px_to_tile_units(1.0) - 8.0).abs() < 1e-4);
        assert_eq!(derived.reference_extent, Some(EXTENT));
        assert_eq!(derived.tile_size_px(), Some(512.0));

        // Raw parameters name no grid, so they apply to every layer verbatim.
        assert_eq!(TessParams::default().reference_extent, None);
        assert_eq!(TessParams::default().tile_size_px(), None);

        let budgeted = TessParams::default()
            .with_max_vertices(0)
            .with_max_feature_points(0);
        assert_eq!(budgeted.max_vertices, 1, "a budget of zero draws nothing");
        assert_eq!(budgeted.max_feature_points, 1);
    }

    #[test]
    fn fills_are_simplified_exactly_as_their_own_outline_is() {
        // A dense convex ring: `lyon` adds no vertex of its own to a convex
        // polygon, so the fill's vertices are the ring points it was given.
        let ring = dense_circle(256, 1500.0);
        let tile = tile(
            "water",
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: ring,
                interiors: Vec::new(),
            }]),
        );
        let coarse = params(1.0, 64.0);
        let fill = mesh_of(
            &tile,
            &PaintTable::new().with("water", FillPaint::new(Rgba8::WHITE)),
            &coarse,
        );
        let outline = mesh_of(
            &tile,
            &PaintTable::new().with("water", LinePaint::new(Rgba8::BLACK).with_width_px(2.0)),
            &coarse,
        );

        let centers: Vec<[f32; 2]> = outline.vertices.iter().map(VectorVertex::center).collect();
        for vertex in &fill.vertices {
            let matched = centers
                .iter()
                .any(|center| close(*center, vertex.position, 1e-5));
            assert!(
                matched,
                "the fill kept {:?}, which its own outline dropped",
                vertex.position
            );
        }
    }

    #[test]
    fn a_coarser_tolerance_sheds_fill_vertices() {
        let tile = tile(
            "water",
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: dense_circle(512, 1800.0),
                interiors: Vec::new(),
            }]),
        );
        let table = PaintTable::new().with("water", FillPaint::new(Rgba8::WHITE));
        let fine = mesh_of(&tile, &table, &params(1.0, 0.25));
        let coarse = mesh_of(&tile, &table, &params(1.0, 128.0));
        assert!(
            coarse.vertices.len() * 4 < fine.vertices.len(),
            "coarse {} vs fine {}",
            coarse.vertices.len(),
            fine.vertices.len()
        );
        assert!(!coarse.is_empty());
        // Still a disc: the centre is painted and the far corner is not.
        assert!(coarse.covers([0.5, 0.5]));
        assert!(!coarse.covers([0.02, 0.02]));
    }

    #[test]
    fn a_layer_with_its_own_extent_keeps_the_stroke_width_it_asked_for() {
        let line = |extent: u32| MvtLayer {
            name: if extent == EXTENT {
                "wide".to_owned()
            } else {
                "narrow".to_owned()
            },
            extent,
            features: vec![feature(MvtGeometry::Lines(vec![vec![
                [0, (extent / 2) as i32],
                [extent as i32, (extent / 2) as i32],
            ]]))],
        };
        let tile = VectorTile {
            layers: vec![line(EXTENT), line(512)],
        };
        let table = PaintTable::new()
            .with(
                "wide",
                LinePaint::new(Rgba8::opaque(1, 1, 1)).with_width_px(8.0),
            )
            .with(
                "narrow",
                LinePaint::new(Rgba8::opaque(2, 2, 2)).with_width_px(8.0),
            );
        // Derived from the first layer's extent, as a tile-wide value must be.
        let Ok(params) = TessParams::for_tile(512.0, EXTENT, TessParams::DEFAULT_TOLERANCE_PX)
        else {
            panic!("params rejected");
        };
        let mesh = mesh_of(&tile, &table, &params);

        let height = |color: [u8; 4]| {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for vertex in mesh.vertices.iter().filter(|v| v.color == color) {
                min = min.min(vertex.position[1]);
                max = max.max(vertex.position[1]);
            }
            max - min
        };
        let wide = height([1, 1, 1, 255]);
        let narrow = height([2, 2, 2, 255]);
        // 8 px of a 512 px tile is 8/512 of the unit square, whatever grid the
        // layer quantises its coordinates on.
        assert!((wide - 8.0 / 512.0).abs() < 1e-5, "{wide}");
        assert!((narrow - wide).abs() < 1e-5, "{narrow} vs {wide}");
    }

    #[test]
    fn the_vertex_budget_stops_the_run_rather_than_reporting_it_afterwards() {
        let tile = tile(
            "water",
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: dense_circle(512, 1800.0),
                interiors: Vec::new(),
            }]),
        );
        let table = PaintTable::new().with("water", FillPaint::new(Rgba8::WHITE));
        let resolver: &dyn PaintResolver = &table;
        let budget = params(1.0, 0.25).with_max_vertices(32);
        let Err(RenderError::Tessellation(message)) = tessellate_tile(&tile, resolver, &budget)
        else {
            panic!("a 32-vertex budget must reject a 512-point ring");
        };
        assert!(message.contains("budget"), "{message}");
    }

    #[test]
    fn the_feature_budget_rejects_a_feature_before_it_is_tessellated() {
        let tile = tile(
            "roads",
            MvtGeometry::Lines(vec![(0..400i32).map(|index| [index * 8, 2048]).collect()]),
        );
        let table = PaintTable::new().with("roads", LinePaint::new(Rgba8::BLACK));
        let resolver: &dyn PaintResolver = &table;
        let budget = params(1.0, 0.25).with_max_feature_points(100);
        let Err(RenderError::Tessellation(message)) = tessellate_tile(&tile, resolver, &budget)
        else {
            panic!("a 100-point budget must reject a 400-point line");
        };
        assert!(message.contains("400 points"), "{message}");

        // The same tile is fine once the budget covers it.
        let generous = params(1.0, 0.25).with_max_feature_points(400);
        assert!(!mesh_of(&tile, &table, &generous).is_empty());
    }

    #[test]
    fn strokes_carry_the_expansion_that_widened_them() {
        let tile = tile(
            "roads",
            MvtGeometry::Lines(vec![vec![[0, 2048], [4096, 2048]]]),
        );
        let table =
            PaintTable::new().with("roads", LinePaint::new(Rgba8::BLACK).with_width_px(8.0));
        let Ok(params) = TessParams::for_tile(512.0, EXTENT, TessParams::DEFAULT_TOLERANCE_PX)
        else {
            panic!("params rejected");
        };
        let mesh = mesh_of(&tile, &table, &params);
        assert!((mesh.baked_tile_size_px - 512.0).abs() < 1e-4);

        for vertex in &mesh.vertices {
            // Every stroke vertex sits half a width off the centre line, and
            // the centre line is what `offset` takes it back to.
            assert!(
                (vertex.center()[1] - 0.5).abs() < 1e-5,
                "{:?} is not on the centre line",
                vertex.center()
            );
        }

        // What the shader does: the same mesh drawn on a 1024 px tile keeps its
        // 8 px width, i.e. half the unit-square height it had at 512 px.
        let scale = mesh.offset_scale_at(1024.0);
        assert!((scale - 0.5).abs() < 1e-6, "{scale}");
        let height = |scale: f32| {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for vertex in &mesh.vertices {
                let y = vertex.position[1] + vertex.offset[1] * (scale - 1.0);
                min = min.min(y);
                max = max.max(y);
            }
            max - min
        };
        assert!((height(1.0) - 8.0 / 512.0).abs() < 1e-5, "{}", height(1.0));
        assert!(
            (height(scale) - 8.0 / 1024.0).abs() < 1e-5,
            "{}",
            height(scale)
        );
    }

    #[test]
    fn circle_radii_are_expansions_too() {
        let tile = tile("poi", MvtGeometry::Points(vec![[2048, 2048]]));
        let table = PaintTable::new().with(
            "poi",
            CirclePaint::new(Rgba8::WHITE)
                .with_radius_px(16.0)
                .with_stroke(Rgba8::BLACK, 4.0),
        );
        let Ok(params) = TessParams::for_tile(512.0, EXTENT, TessParams::DEFAULT_TOLERANCE_PX)
        else {
            panic!("params rejected");
        };
        let mesh = mesh_of(&tile, &table, &params);
        assert!(!mesh.is_empty());
        for vertex in &mesh.vertices {
            assert!(
                close(vertex.center(), [0.5, 0.5], 1e-5),
                "{:?} does not come back to the centre",
                vertex.center()
            );
        }
    }

    #[test]
    fn the_offset_scale_is_bounded_and_falls_back_to_the_identity() {
        assert!((offset_scale(512.0, 256.0) - 2.0).abs() < 1e-6);
        assert!((offset_scale(256.0, 512.0) - 0.5).abs() < 1e-6);
        // An unknown or unusable size draws exactly what was baked.
        assert!((offset_scale(0.0, 256.0) - 1.0).abs() < 1e-6);
        assert!((offset_scale(256.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((offset_scale(f32::NAN, 256.0) - 1.0).abs() < 1e-6);
        assert!((offset_scale(f32::INFINITY, 256.0) - 1.0).abs() < 1e-6);
        assert!((offset_scale(1e30, 1e-30) - MAX_OFFSET_SCALE).abs() < 1e-3);
        assert!((offset_scale(1e-30, 1e30) - 1.0 / MAX_OFFSET_SCALE).abs() < 1e-9);
        assert_eq!(VectorMesh::new().offset_scale_at(256.0), 1.0);
    }

    #[test]
    fn reusing_the_scratch_changes_nothing_about_the_mesh() {
        let water = tile(
            "water",
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: dense_circle(64, 900.0),
                interiors: Vec::new(),
            }]),
        );
        let roads = tile(
            "roads",
            MvtGeometry::Lines(vec![vec![[0, 2048], [4096, 2048]]]),
        );
        let table = PaintTable::new()
            .with("water", FillPaint::new(Rgba8::WHITE))
            .with("roads", LinePaint::new(Rgba8::BLACK).with_width_px(3.0));
        let resolver: &dyn PaintResolver = &table;
        let params = params(1.0, 4.0);

        let mut scratch = TessScratch::new();
        let Ok(first) = tessellate_tile_with(&water, resolver, &params, &mut scratch) else {
            panic!("tessellation failed");
        };
        let Ok(second) = tessellate_tile_with(&roads, resolver, &params, &mut scratch) else {
            panic!("tessellation failed");
        };
        assert_eq!(first, mesh_of(&water, &table, &params));
        assert_eq!(second, mesh_of(&roads, &table, &params));
        assert!(format!("{scratch:?}").contains("TessScratch"));
    }

    #[test]
    fn a_second_pass_appends_to_the_mesh_the_first_one_built() {
        let tile = tile(
            "water",
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: square(0, 0, 2048, 2048),
                interiors: Vec::new(),
            }]),
        );
        let fill = PaintTable::new().with("water", FillPaint::new(Rgba8::opaque(1, 1, 1)));
        let outline = PaintTable::new().with(
            "water",
            LinePaint::new(Rgba8::opaque(2, 2, 2)).with_width_px(2.0),
        );
        let Ok(params) = TessParams::for_tile(512.0, EXTENT, TessParams::DEFAULT_TOLERANCE_PX)
        else {
            panic!("params rejected");
        };

        let mut scratch = TessScratch::new();
        let mut combined = VectorMesh::new();
        let fill_resolver: &dyn PaintResolver = &fill;
        let outline_resolver: &dyn PaintResolver = &outline;
        if tessellate_tile_into(&tile, fill_resolver, &params, &mut scratch, &mut combined).is_err()
        {
            panic!("the fill pass failed");
        }
        let after_fill = combined.vertices.len();
        if tessellate_tile_into(
            &tile,
            outline_resolver,
            &params,
            &mut scratch,
            &mut combined,
        )
        .is_err()
        {
            panic!("the outline pass failed");
        }

        assert!(combined.vertices.len() > after_fill);
        assert!((combined.baked_tile_size_px - 512.0).abs() < 1e-4);
        // Indices of the second pass address its own vertices, not the first
        // pass's: every triangle must still be in range and the fill must still
        // paint its centre.
        let count = combined.vertices.len();
        assert!(
            combined
                .indices
                .iter()
                .all(|index| (*index as usize) < count)
        );
        assert_eq!(combined.triangles().count(), combined.triangle_count());
        assert!(combined.covers([0.25, 0.25]));
        let outline_only = mesh_of(&tile, &outline, &params);
        assert_eq!(
            combined.vertices.len(),
            after_fill + outline_only.vertices.len()
        );
    }
}
