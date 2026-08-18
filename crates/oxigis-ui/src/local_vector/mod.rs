//! Local (in-memory) vector datasets: GeoJSON text → a synthetic
//! [`VectorTile`] the §5.2 tessellator and the §5.3 label placer can consume
//! unchanged (blueprint Phase 1 §1.1, data half).
//!
//! # Why a synthetic tile
//!
//! Everything OxiGIS can already draw for vector data — `lyon` fills, strokes
//! and circle fans ([`oxigis_render::tessellate_tile`]), greedy label placement
//! ([`oxigis_render::LabelPlacer`]), the GPU pipeline and its per-quad instance
//! transform ([`oxigis_render::VectorPipeline`]) — is expressed in terms of *one
//! tile*: integer coordinates on an `extent` grid, placed on screen by a
//! [`TilePlacement`]. A dropped GeoJSON file is not tiled, but it *is* a bounded
//! set of features, so the cheapest way to reuse all of that is to declare the
//! whole dataset to be a single tile:
//!
//! ```text
//! FeatureCollection
//!   → project every vertex to normalised Web Mercator (0..1, y down)
//!   → take the bbox, pad it to a square (`MercatorSquare`)
//!   → quantise every vertex onto that square's LOCAL_EXTENT grid
//!   → VectorTile { layers: [MvtLayer { name: LOCAL_LAYER_NAME, .. }] }
//! ```
//!
//! and to compute one [`TilePlacement`] per frame from the camera. No clipping
//! is needed: the "tile" *is* the dataset, and the map viewport does the
//! clipping.
//!
//! # Why the bbox is squared
//!
//! [`TilePlacement`] carries a single scalar `size` — the renderer's quads are
//! square by construction, because map tiles are. Padding the shorter mercator
//! axis until the bbox is square is therefore not a nicety but the only way a
//! rectangular dataset can be expressed as one placement. The padding is
//! centred, so the dataset stays put; the unused margin holds no geometry.
//!
//! # Precision, and its limit
//!
//! Vertices are quantised to the *dataset's own* square, not to the world, so a
//! city-sized dataset keeps roughly `span / 65536` of resolution rather than
//! `world / 65536`. The on-screen quantisation error is exactly
//!
//! ```text
//! error_px = placement.size / LOCAL_EXTENT
//! ```
//!
//! which is sub-pixel while the dataset is smaller than 65536 physical pixels on
//! screen — i.e. for any dataset zoomed to roughly fit the window, and well
//! beyond. It degrades once a dataset is zoomed far past its own extent (a
//! country-wide dataset at z18 spans about 2·10⁶ px, giving ~30 px steps). That
//! is inherent to "one tile at every zoom" — MVT escapes it by re-tiling per
//! zoom — and is a documented Phase 1 limit, not something a larger extent
//! fixes: [`LOCAL_EXTENT`] is already 2¹⁶ and MVT coordinates are `i32`.
//!
//! # Winding
//!
//! [`MvtPolygon`] rings are **unclosed** and carry the sign convention the MVT
//! decoder produces, which [`oxigis_render::tessellate_tile`] feeds to `lyon`'s
//! non-zero fill rule verbatim: exteriors one way round, interiors the other.
//! GeoJSON in the wild routinely ignores RFC 7946's right-hand rule, so rings
//! are **normalised** here rather than trusted — [`ring_shoelace`] is computed
//! on the quantised grid and the ring reversed when the sign is wrong. See
//! [`EXTERIOR_SHOELACE_IS_NEGATIVE`] for the convention and this module's tests
//! for the cross-check against `oxigis-render`'s own fixtures.
//!
//! # Where the OxiGeo dependency lives
//!
//! Here, and only here. `oxigis-render` must never grow one (it is deliberately
//! `oxigis`-agnostic and wasm-first), and `oxigis-core`'s crate docs commit to
//! holding no OxiGeo types at all, so routing the parse through a core
//! re-export would break a documented invariant. `oxigis-ui` therefore takes
//! `oxigeo` directly, with the `geojson` feature only — a strict subset of what
//! `oxigis-core` already enables, so no new crate enters the dependency graph.

use std::sync::Arc;

use oxigeo::geojson::types::{Feature, FeatureCollection, Geometry, Position, Properties};
use oxigis_core::{
    CircleStyle, Classification, Color, FillStyle, LayerStyle, LayerStyleSet, LineStyle, Renderer,
    SymbolStyle,
};
use oxigis_render::{
    LabelTable, LonLat, MAX_ZOOM, MapView, MvtFeature, MvtGeometry, MvtLayer, MvtPolygon, MvtValue,
    RenderError, TILE_SIZE_PX, TessParams, TileId, TilePlacement, VectorMesh, VectorTile,
    WorldCoord,
};

use crate::style_paint::{PaintProgram, label_table};

/// Name of the single [`MvtLayer`] a local dataset is converted into.
///
/// Style rules for a local layer are matched against this name, which is why it
/// is a constant rather than the dataset's own name: a [`LayerStyle`] edited in
/// the style panel must keep applying after the layer is renamed.
pub const LOCAL_LAYER_NAME: &str = "features";

/// The polygon family's tile layer (tiles v1.3: the renderer keys styles by
/// LAYER NAME, so per-family styling means per-family layers; the colon
/// follows the `stem:table` GeoPackage idiom).
pub const LOCAL_POLYGON_LAYER_NAME: &str = "features:polygon";
/// The line family's tile layer.
pub const LOCAL_LINE_LAYER_NAME: &str = "features:line";
/// The point family's tile layer.
pub const LOCAL_POINT_LAYER_NAME: &str = "features:point";

/// The tile layer one geometry family lands in — the FALLBACK bucket once a
/// renderer classifies the layer (thematic v1.6), and the only bucket before
/// it does.
#[must_use]
pub fn local_layer_name(family: GeometryKind) -> &'static str {
    match family {
        GeometryKind::Polygon => LOCAL_POLYGON_LAYER_NAME,
        GeometryKind::Line => LOCAL_LINE_LAYER_NAME,
        GeometryKind::Point => LOCAL_POINT_LAYER_NAME,
    }
}

/// What separates a family layer's name from its class index (thematic v1.6).
///
/// `'#'` because it cannot occur in a family name and reads as "number":
/// `features:polygon#3` is the polygon geometry of the fourth class.
pub const LOCAL_CLASS_SEPARATOR: char = '#';

/// The tile layer one *class* of one geometry family lands in.
///
/// The renderer keys paints by LAYER NAME, so per-feature styling means
/// per-class layers, exactly as per-family styling meant per-family layers in
/// v1.3. Each class is then one mesh with one paint, drawn by the machinery
/// that already exists — no new render path, and a
/// [`oxigis_core::Renderer::Single`] layer keeps producing precisely the v1.3
/// layer list.
#[must_use]
pub fn local_class_layer_name(family: GeometryKind, class: usize) -> String {
    format!("{}{LOCAL_CLASS_SEPARATOR}{class}", local_layer_name(family))
}

/// Which family a local tile layer name belongs to, class suffix or not.
///
/// The inverse of [`local_layer_name`] / [`local_class_layer_name`], so a
/// caller reading a built tile back (the family probe, a debug dump) does not
/// have to know whether the layer was classified.
#[must_use]
pub fn family_of_layer_name(name: &str) -> Option<GeometryKind> {
    let stem = match name.split_once(LOCAL_CLASS_SEPARATOR) {
        Some((stem, class)) if !class.is_empty() && class.bytes().all(|b| b.is_ascii_digit()) => {
            stem
        }
        Some(_) => return None,
        None => name,
    };
    match stem {
        LOCAL_POLYGON_LAYER_NAME => Some(GeometryKind::Polygon),
        LOCAL_LINE_LAYER_NAME => Some(GeometryKind::Line),
        LOCAL_POINT_LAYER_NAME => Some(GeometryKind::Point),
        _ => None,
    }
}

/// Coordinate grid a local dataset is quantised onto, per axis.
///
/// See the [module docs][self] on precision: `2^16` keeps the error below one
/// physical pixel until the dataset covers 65536 px on screen, and stays far
/// inside the `i32` range MVT coordinates use.
pub const LOCAL_EXTENT: u32 = 1 << 16;

/// Smallest side a dataset's mercator square is padded to, in normalised world
/// units (`1.0` is the whole world).
///
/// `10^-6` of the world is about 40 m, which is what a single-point dataset —
/// whose bbox is degenerate — is given so that it has a placement with a
/// positive size and its circle symbol has somewhere to land.
pub const MIN_BBOX_SPAN: f64 = 1e-6;

/// Sign convention: an MVT exterior ring has a **negative** raw shoelace sum.
///
/// The MVT decoder documents its exteriors as "positive shoelace area in Y-down
/// space", which is the same statement with the y-flip folded in; expressed as
/// the plain `Σ (xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)` [`ring_shoelace`] computes, the sign is
/// negative. Interior rings are the opposite. Both are cross-checked against
/// `oxigis-render`'s own tessellator fixtures in this module's tests.
pub const EXTERIOR_SHOELACE_IS_NEGATIVE: bool = true;

/// Ratio by which the on-screen size of a dataset may drift from the size its
/// mesh was tessellated at before the mesh is rebuilt.
///
/// Stroke widths and circle radii are baked into the mesh in *tile units* (see
/// [`oxigis_render::TessParams`]), so a mesh is only correct at the zoom it was
/// built for. A factor of `√2` is half a zoom step: widths are never more than
/// 41 % off, and a continuous pinch-zoom re-tessellates twice per zoom level
/// rather than every frame.
pub const RETESSELLATE_RATIO: f32 = core::f32::consts::SQRT_2;

/// Accent colour every default style is built from.
pub const DEFAULT_ACCENT: Color = Color {
    r: 0x33,
    g: 0x77,
    b: 0xdd,
    a: 0xff,
};

/// Outline colour of the default polygon style: the accent, darkened.
pub const DEFAULT_OUTLINE: Color = Color {
    r: 0x1c,
    g: 0x4b,
    b: 0x8f,
    a: 0xff,
};

/// Fill opacity of the default polygon style — semi-transparent, so a basemap
/// stays readable underneath.
pub const DEFAULT_FILL_OPACITY: f32 = 0.35;

/// Stroke width of the default line style, in physical pixels.
pub const DEFAULT_LINE_WIDTH_PX: f32 = 2.0;

/// Radius of the default point style, in physical pixels.
pub const DEFAULT_CIRCLE_RADIUS_PX: f32 = 4.0;

/// Why a local dataset could not be loaded.
///
/// Deliberately a small owned type rather than a `thiserror` enum: this crate
/// has no `thiserror` dependency (see [`crate::tile_provider::TileError`] for
/// the same pattern) and the caller only ever shows the message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalVectorError {
    /// Human-readable cause, shown in the status bar.
    message: String,
}

impl LocalVectorError {
    /// Wraps a message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The human-readable cause.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl core::fmt::Display for LocalVectorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl core::error::Error for LocalVectorError {}

/// The square region of normalised Web Mercator space a local dataset is
/// quantised against.
///
/// `min_x`/`min_y` are the north-west corner and `size` the side length, all in
/// normalised world units where `(0, 0)` is the world's north-west corner,
/// `(1, 1)` its south-east corner and `y` grows southwards. The square may
/// legitimately extend outside `0..1` after padding; nothing depends on it
/// being inside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MercatorSquare {
    /// Western edge, in normalised world units.
    pub min_x: f64,
    /// Northern edge, in normalised world units.
    pub min_y: f64,
    /// Side length, in normalised world units; always finite and positive.
    pub size: f64,
}

impl MercatorSquare {
    /// The whole world — the fallback for a dataset with no projectable vertex.
    #[must_use]
    pub const fn world() -> Self {
        Self {
            min_x: 0.0,
            min_y: 0.0,
            size: 1.0,
        }
    }

    /// The square covering `[min, max]`, padded to at least [`MIN_BBOX_SPAN`]
    /// per axis and then to a square, both times about the centre.
    ///
    /// Non-finite or inverted input collapses to [`MercatorSquare::world`], so
    /// the result is always usable.
    #[must_use]
    pub fn from_world_bounds(min: [f64; 2], max: [f64; 2]) -> Self {
        if !min.iter().chain(max.iter()).all(|value| value.is_finite())
            || min[0] > max[0]
            || min[1] > max[1]
        {
            return Self::world();
        }
        // Degenerate axes are padded *before* squaring: squaring a zero-height
        // bbox first would leave nothing to work with.
        let span_x = (max[0] - min[0]).max(MIN_BBOX_SPAN);
        let span_y = (max[1] - min[1]).max(MIN_BBOX_SPAN);
        let size = span_x.max(span_y);
        let center_x = f64::midpoint(min[0], max[0]);
        let center_y = f64::midpoint(min[1], max[1]);
        Self {
            min_x: center_x - size / 2.0,
            min_y: center_y - size / 2.0,
            size,
        }
    }

    /// The south-east corner, in normalised world units.
    #[must_use]
    pub fn max(&self) -> [f64; 2] {
        [self.min_x + self.size, self.min_y + self.size]
    }

    /// The centre, as a geographic position — what "zoom to layer" pans to.
    #[must_use]
    pub fn center_lon_lat(&self) -> LonLat {
        WorldCoord::new(self.min_x + self.size / 2.0, self.min_y + self.size / 2.0)
            .clamped()
            .to_lon_lat()
    }

    /// The zoom at which the square just fits `size_px`, with `margin` of the
    /// shorter viewport dimension left free on each side.
    ///
    /// Clamped to `0..=`[`oxigis_render::MAX_ZOOM`], so the result is always a
    /// zoom [`MapView::with_zoom`] accepts.
    #[must_use]
    pub fn fit_zoom(&self, size_px: [f32; 2], margin: f32) -> f64 {
        let shortest = f64::from(size_px[0].min(size_px[1]));
        if !shortest.is_finite() || shortest <= 0.0 {
            return 0.0;
        }
        let usable = shortest * (1.0 - 2.0 * f64::from(margin.clamp(0.0, 0.45)));
        // world_pixels = 256 · 2^zoom, and the square covers `size` of it.
        let zoom = (usable / (self.size * TILE_SIZE_PX)).log2();
        if zoom.is_finite() {
            zoom.clamp(0.0, f64::from(MAX_ZOOM))
        } else {
            0.0
        }
    }

    /// Quantises a normalised world position onto the square's
    /// [`LOCAL_EXTENT`] grid, clamped to `0..=LOCAL_EXTENT`.
    #[must_use]
    pub fn quantize(&self, world: [f64; 2]) -> [i32; 2] {
        let extent = f64::from(LOCAL_EXTENT);
        let axis = |value: f64, min: f64| -> i32 {
            let scaled = ((value - min) / self.size * extent).round();
            if scaled.is_finite() {
                scaled.clamp(0.0, extent) as i32
            } else {
                0
            }
        };
        [axis(world[0], self.min_x), axis(world[1], self.min_y)]
    }

    /// The inverse of [`MercatorSquare::quantize`], for tests and hit-testing.
    #[must_use]
    pub fn dequantize(&self, grid: [i32; 2]) -> [f64; 2] {
        let extent = f64::from(LOCAL_EXTENT);
        [
            self.min_x + f64::from(grid[0]) / extent * self.size,
            self.min_y + f64::from(grid[1]) / extent * self.size,
        ]
    }

    /// Where the square lands on screen under `view`, in physical pixels.
    ///
    /// The arithmetic is deliberately identical to [`MapView::place_tile`]'s —
    /// the square *is* a tile as far as the renderer is concerned. The returned
    /// [`TilePlacement::tile`] is a placeholder (`0/0/0`): nothing in the local
    /// path is keyed by a [`TileId`], and neither
    /// [`oxigis_render::tile_scissor`] nor
    /// [`oxigis_render::LabelPlacer::place_tile`] reads the field.
    #[must_use]
    pub fn place(&self, view: MapView) -> TilePlacement {
        let scale = view.world_pixels();
        let center = view.center().to_world();
        let size_px = view.size_px();
        TilePlacement {
            tile: TileId { z: 0, x: 0, y: 0 },
            x: ((self.min_x - center.x) * scale + f64::from(size_px[0]) / 2.0) as f32,
            y: ((self.min_y - center.y) * scale + f64::from(size_px[1]) / 2.0) as f32,
            size: (self.size * scale) as f32,
        }
    }

    /// Worst-case on-screen quantisation error at a given placement size, in
    /// physical pixels. See the [module docs][self].
    #[must_use]
    pub fn quantization_error_px(placement_size_px: f32) -> f32 {
        placement_size_px / LOCAL_EXTENT as f32
    }
}

/// Projects a GeoJSON position to normalised Web Mercator, or [`None`] if it is
/// not a usable 2-D position.
///
/// GeoJSON positions are `Vec<f64>` of length 2 or 3 (RFC 7946 allows a third
/// altitude element, which is dropped); anything shorter, or carrying a
/// non-finite ordinate, is skipped rather than silently projected to the null
/// island. Latitudes beyond the Mercator cut-off saturate instead of diverging —
/// that is [`LonLat::to_world`]'s documented behaviour — so a legal `lat: 90`
/// point cannot poison the dataset's bbox with an infinity.
#[must_use]
pub fn project_position(position: &Position) -> Option<[f64; 2]> {
    let lon = *position.first()?;
    let lat = *position.get(1)?;
    if !lon.is_finite() || !lat.is_finite() {
        return None;
    }
    let world = LonLat::new(lon, lat).to_world();
    if world.x.is_finite() && world.y.is_finite() {
        Some([world.x, world.y])
    } else {
        None
    }
}

/// The normalised-mercator bbox of a whole collection, as a padded square.
///
/// Features with no geometry, and vertices [`project_position`] rejects, are
/// skipped; a collection with no usable vertex at all yields
/// [`MercatorSquare::world`].
#[must_use]
pub fn collection_square(features: &FeatureCollection) -> MercatorSquare {
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    let mut seen = false;
    for feature in &features.features {
        if let Some(geometry) = feature.geometry.as_ref() {
            visit_positions(geometry, &mut |world| {
                seen = true;
                for axis in 0..2 {
                    min[axis] = min[axis].min(world[axis]);
                    max[axis] = max[axis].max(world[axis]);
                }
            });
        }
    }
    if seen {
        MercatorSquare::from_world_bounds(min, max)
    } else {
        MercatorSquare::world()
    }
}

/// Calls `visit` with every projectable vertex of `geometry`, recursing into
/// geometry collections.
fn visit_positions(geometry: &Geometry, visit: &mut impl FnMut([f64; 2])) {
    let sequence = |positions: &[Position], visit: &mut dyn FnMut([f64; 2])| {
        for position in positions {
            if let Some(world) = project_position(position) {
                visit(world);
            }
        }
    };
    match geometry {
        Geometry::Point(point) => {
            if let Some(world) = project_position(&point.coordinates) {
                visit(world);
            }
        }
        Geometry::MultiPoint(points) => sequence(&points.coordinates, visit),
        Geometry::LineString(line) => sequence(&line.coordinates, visit),
        Geometry::MultiLineString(lines) => {
            for line in &lines.coordinates {
                sequence(line, visit);
            }
        }
        Geometry::Polygon(polygon) => {
            for ring in &polygon.coordinates {
                sequence(ring, visit);
            }
        }
        Geometry::MultiPolygon(polygons) => {
            for polygon in &polygons.coordinates {
                for ring in polygon {
                    sequence(ring, visit);
                }
            }
        }
        Geometry::GeometryCollection(collection) => {
            for child in &collection.geometries {
                visit_positions(child, visit);
            }
        }
    }
}

/// Twice the signed area of a ring, as the plain shoelace sum on the quantised
/// grid.
///
/// Computed in `f64`: the product of two `i32` grid coordinates overflows
/// `i32`, and `f64` represents every value a 2¹⁶ grid can produce exactly. See
/// [`EXTERIOR_SHOELACE_IS_NEGATIVE`] for what the sign means.
#[must_use]
pub fn ring_shoelace(ring: &[[i32; 2]]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for index in 0..ring.len() {
        let current = ring[index];
        let next = ring[(index + 1) % ring.len()];
        sum +=
            f64::from(current[0]) * f64::from(next[1]) - f64::from(next[0]) * f64::from(current[1]);
    }
    sum
}

/// Reverses `ring` in place unless it already winds the way MVT expects for its
/// role.
///
/// The orientation is *measured*, never inferred from the source document: real
/// GeoJSON breaks RFC 7946's right-hand rule often enough that trusting it
/// would produce silently unfilled polygons.
pub fn normalize_ring_winding(ring: &mut [[i32; 2]], exterior: bool) {
    let shoelace = ring_shoelace(ring);
    if shoelace == 0.0 {
        return;
    }
    let wants_negative = exterior == EXTERIOR_SHOELACE_IS_NEGATIVE;
    if (shoelace < 0.0) != wants_negative {
        ring.reverse();
    }
}

/// Quantises a GeoJSON coordinate sequence, dropping unusable positions and
/// collapsing consecutive duplicates.
fn quantize_sequence(positions: &[Position], square: &MercatorSquare) -> Vec<[i32; 2]> {
    let mut out: Vec<[i32; 2]> = Vec::with_capacity(positions.len());
    for position in positions {
        let Some(world) = project_position(position) else {
            continue;
        };
        let grid = square.quantize(world);
        if out.last() != Some(&grid) {
            out.push(grid);
        }
    }
    out
}

/// Quantises one linear ring: closing vertex dropped, winding normalised.
///
/// Returns [`None`] for a ring that cannot bound an area (fewer than three
/// distinct grid points after quantisation), which the tessellator would skip
/// anyway.
fn quantize_ring(
    positions: &[Position],
    square: &MercatorSquare,
    exterior: bool,
) -> Option<Vec<[i32; 2]>> {
    let mut ring = quantize_sequence(positions, square);
    // GeoJSON rings are closed, MVT rings are not. Quantisation can also fold
    // several source vertices onto the first one, hence the loop.
    while ring.len() > 1 && ring.first() == ring.last() {
        ring.pop();
    }
    if ring.len() < 3 {
        return None;
    }
    normalize_ring_winding(&mut ring, exterior);
    Some(ring)
}

/// Converts one GeoJSON polygon (exterior ring first, holes after) into an
/// [`MvtPolygon`], or [`None`] if its exterior ring is degenerate.
fn convert_polygon(rings: &[Vec<Position>], square: &MercatorSquare) -> Option<MvtPolygon> {
    let mut iter = rings.iter();
    let exterior = quantize_ring(iter.next()?, square, true)?;
    let interiors = iter
        .filter_map(|ring| quantize_ring(ring, square, false))
        .collect();
    Some(MvtPolygon {
        exterior,
        interiors,
    })
}

/// How deep a nested `GeometryCollection` is followed before the rest of the
/// subtree is dropped.
///
/// The same value as `edit::command::MAX_GEOMETRY_DEPTH` (and the private
/// copies in the hit/overlay/topology walkers), with the same check placement
/// — a collection *entered at* this depth contributes nothing — so what the
/// tile builder draws and what the editing path exposes stay the same set,
/// nesting included.
const MAX_GEOMETRY_DEPTH: usize = 8;

/// One feature's drawable geometry, quantised and grouped by family.
///
/// A plain geometry fills exactly one group; a `GeometryCollection` may fill
/// several. Grouping (rather than a single [`MvtGeometry`]) is what lets a
/// mixed collection draw at all: one MVT feature has exactly one geometry, so
/// the feature converter emits one feature per non-empty group instead.
#[derive(Debug, Default)]
struct GeometryGroups {
    polygons: Vec<MvtPolygon>,
    lines: Vec<Vec<[i32; 2]>>,
    points: Vec<[i32; 2]>,
}

impl GeometryGroups {
    /// The single [`MvtGeometry`] these groups describe, or [`None`] when they
    /// are empty **or** mixed — one MVT geometry cannot name two families.
    fn into_single(self) -> Option<MvtGeometry> {
        match (
            self.polygons.is_empty(),
            self.lines.is_empty(),
            self.points.is_empty(),
        ) {
            (false, true, true) => Some(MvtGeometry::Polygons(self.polygons)),
            (true, false, true) => Some(MvtGeometry::Lines(self.lines)),
            (true, true, false) => Some(MvtGeometry::Points(self.points)),
            (true, true, true) => None,
            _ => {
                tracing::debug!(
                    "oxigis-ui: a GeometryCollection that mixes geometry kinds \
                     is one feature per kind, not one geometry",
                );
                None
            }
        }
    }
}

/// Quantises `geometry` into `out`, recursing into collections to
/// [`MAX_GEOMETRY_DEPTH`].
///
/// Degenerate members are dropped exactly as their standalone conversions
/// would drop them: an unprojectable point, a line that collapses to one grid
/// point, a polygon whose exterior ring degenerates.
fn collect_geometry(
    geometry: &Geometry,
    square: &MercatorSquare,
    depth: usize,
    out: &mut GeometryGroups,
) {
    match geometry {
        Geometry::Point(point) => {
            if let Some(world) = project_position(&point.coordinates) {
                out.points.push(square.quantize(world));
            }
        }
        Geometry::MultiPoint(points) => {
            for position in &points.coordinates {
                if let Some(world) = project_position(position) {
                    out.points.push(square.quantize(world));
                }
            }
        }
        Geometry::LineString(line) => {
            let points = quantize_sequence(&line.coordinates, square);
            if points.len() >= 2 {
                out.lines.push(points);
            }
        }
        Geometry::MultiLineString(lines) => {
            for line in &lines.coordinates {
                let points = quantize_sequence(line, square);
                if points.len() >= 2 {
                    out.lines.push(points);
                }
            }
        }
        Geometry::Polygon(polygon) => {
            if let Some(converted) = convert_polygon(&polygon.coordinates, square) {
                out.polygons.push(converted);
            }
        }
        Geometry::MultiPolygon(polygons) => {
            for rings in &polygons.coordinates {
                if let Some(converted) = convert_polygon(rings, square) {
                    out.polygons.push(converted);
                }
            }
        }
        Geometry::GeometryCollection(collection) => {
            if depth >= MAX_GEOMETRY_DEPTH {
                tracing::debug!(
                    "oxigis-ui: dropping a GeometryCollection nested deeper than {}",
                    MAX_GEOMETRY_DEPTH,
                );
                return;
            }
            for member in &collection.geometries {
                collect_geometry(member, square, depth + 1, out);
            }
        }
    }
}

/// Converts a GeoJSON geometry into MVT geometry on `square`'s grid.
///
/// Returns [`None`] when nothing drawable survives — an empty geometry, a line
/// that collapses to one point, a polygon whose exterior ring degenerates —
/// and for a `GeometryCollection` that mixes geometry families, which no
/// single [`MvtGeometry`] can express (the feature converter splits such a
/// collection into one feature per family instead; see
/// [`feature_collection_to_tile`]).
///
/// `MultiPoint`/`MultiLineString`/`MultiPolygon` need no special handling: every
/// [`MvtGeometry`] variant is already inherently "multi", exactly as in the MVT
/// specification, so a multipolygon stays one feature with several
/// [`MvtPolygon`]s. A collection whose members agree on a family flattens into
/// that one variant. Collections nest to `MAX_GEOMETRY_DEPTH`, the editing
/// path's cap.
#[must_use]
pub fn convert_geometry(geometry: &Geometry, square: &MercatorSquare) -> Option<MvtGeometry> {
    let mut groups = GeometryGroups::default();
    collect_geometry(geometry, square, 0, &mut groups);
    groups.into_single()
}

/// Converts GeoJSON properties into the MVT property list the style, the label
/// placer and (later) the attribute table read.
///
/// The mapping, in probe order: `null` is dropped (MVT has no null value, and a
/// missing key reads the same to every consumer), strings, booleans and numbers
/// map onto [`MvtValue::String`] / [`MvtValue::Bool`] /
/// [`MvtValue::I64`]-[`MvtValue::U64`]-[`MvtValue::F64`], and arrays and nested
/// objects are re-serialised to their compact JSON text as a
/// [`MvtValue::String`] — lossy for arithmetic, faithful for display and for a
/// `text_field` that names one.
#[must_use]
pub fn convert_properties(properties: Option<&Properties>) -> Vec<(String, MvtValue)> {
    let Some(properties) = properties else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(properties.len());
    for (key, value) in properties {
        if value.is_null() {
            continue;
        }
        let converted = if let Some(text) = value.as_str() {
            MvtValue::String(text.to_owned())
        } else if let Some(flag) = value.as_bool() {
            MvtValue::Bool(flag)
        } else if let Some(number) = value.as_i64() {
            MvtValue::I64(number)
        } else if let Some(number) = value.as_u64() {
            MvtValue::U64(number)
        } else if let Some(number) = value.as_f64() {
            MvtValue::F64(number)
        } else {
            // Arrays and objects: their own compact JSON rendering.
            MvtValue::String(value.to_string())
        };
        out.push((key.clone(), converted));
    }
    out
}

/// Converts one GeoJSON feature into the MVT features that draw it — usually
/// one, empty if it draws nothing, and one **per geometry family** for a
/// mixed `GeometryCollection`.
///
/// Every emitted feature shares the source index as its [`MvtFeature::id`], so
/// the id → source-feature relationship survives the split. Properties ride on
/// exactly **one** of them — the family [`geometry_kind`] reports for the
/// source geometry (the same rule hit testing and default styling rank by),
/// falling back to the first emitted family when that one quantised away —
/// so a labelling Symbol rule anchors one label per source feature, never one
/// per family. Emission order is polygons → lines → points, the painter's
/// order fills-under-strokes-under-markers.
fn convert_feature(feature: &Feature, index: usize, square: &MercatorSquare) -> Vec<MvtFeature> {
    let Some(geometry) = feature.geometry.as_ref() else {
        return Vec::new();
    };
    let mut groups = GeometryGroups::default();
    collect_geometry(geometry, square, 0, &mut groups);
    let id = u64::try_from(index).ok();
    let mut out = Vec::new();
    if !groups.polygons.is_empty() {
        out.push(MvtFeature {
            id,
            properties: Vec::new(),
            geometry: MvtGeometry::Polygons(groups.polygons),
        });
    }
    if !groups.lines.is_empty() {
        out.push(MvtFeature {
            id,
            properties: Vec::new(),
            geometry: MvtGeometry::Lines(groups.lines),
        });
    }
    if !groups.points.is_empty() {
        out.push(MvtFeature {
            id,
            properties: Vec::new(),
            geometry: MvtGeometry::Points(groups.points),
        });
    }
    if let Some(carrier) = properties_carrier(geometry, &mut out) {
        carrier.properties = convert_properties(feature.properties.as_ref());
    }
    out
}

/// Which of `emitted` carries the source feature's properties.
///
/// The family [`geometry_kind`] names when it survived quantisation, the first
/// emitted feature otherwise.
fn properties_carrier<'a>(
    geometry: &Geometry,
    emitted: &'a mut [MvtFeature],
) -> Option<&'a mut MvtFeature> {
    let preferred = geometry_kind(geometry);
    let position = preferred.and_then(|kind| {
        emitted.iter().position(|feature| {
            matches!(
                (kind, &feature.geometry),
                (GeometryKind::Polygon, MvtGeometry::Polygons(_))
                    | (GeometryKind::Line, MvtGeometry::Lines(_))
                    | (GeometryKind::Point, MvtGeometry::Points(_))
            )
        })
    });
    match position {
        Some(found) => emitted.get_mut(found),
        None => emitted.first_mut(),
    }
}

/// Converts a whole collection into the single-layer [`VectorTile`] the render
/// path consumes.
///
/// Feature order is preserved (it is the painter's-algorithm draw order and the
/// attribute table's row order) and [`MvtFeature::id`] is the feature's index in
/// the source collection, which is what lets a table row address a feature. A
/// mixed `GeometryCollection` contributes one feature per geometry family, all
/// sharing that id — see `convert_feature`.
#[must_use]
pub fn feature_collection_to_tile(
    features: &FeatureCollection,
    square: &MercatorSquare,
) -> VectorTile {
    feature_collection_to_tile_with(features, square, &Renderer::Single)
}

/// [`feature_collection_to_tile`], partitioned by `renderer` (thematic v1.6).
///
/// A [`Renderer::Single`] layer produces exactly the layer list
/// [`feature_collection_to_tile`] always produced — same names, same order,
/// same features — so the single-symbol picture is unchanged by construction
/// rather than by inspection.
///
/// A classified layer additionally gets one tile layer per (family, class)
/// that any feature actually lands in, named by [`local_class_layer_name`].
/// The order is, per family, the FALLBACK bucket first and then the classes in
/// legend order: tessellation walks layers in order and later layers paint
/// over earlier ones, so "everything else" sits under the classified features
/// it is the background of. Families keep their painter's order
/// (polygon → line → point), and empty buckets are omitted exactly as empty
/// families always were.
///
/// A source feature is classified **once**, from its own GeoJSON properties,
/// and every family it contributes to lands in that one class — so a mixed
/// `GeometryCollection` cannot be half red and half blue.
#[must_use]
pub fn feature_collection_to_tile_with(
    features: &FeatureCollection,
    square: &MercatorSquare,
    renderer: &Renderer,
) -> VectorTile {
    // `class_count` is capped at `MAX_STYLE_CLASSES` by the model itself, so
    // the bucket count is bounded no matter what a hand-edited project file
    // asks for: at most 3 families x (64 classes + 1 fallback).
    let classes = renderer.class_count();
    let buckets = classes + 1;
    let mut families: [Vec<Vec<MvtFeature>>; 3] = [
        new_buckets(buckets),
        new_buckets(buckets),
        new_buckets(buckets),
    ];
    for (index, feature) in features.features.iter().enumerate() {
        // Bucket 0 is the fallback and class `c` is bucket `c + 1`, which is
        // the order `bucket_names` names them in and therefore the order they
        // are emitted and painted in.
        let class = feature
            .properties
            .as_ref()
            .and_then(|properties| renderer.class_of(properties))
            .map_or(0, |class| class + 1);
        for converted in convert_feature(feature, index, square) {
            let family = match &converted.geometry {
                MvtGeometry::Polygons(_) => GeometryKind::Polygon,
                MvtGeometry::Lines(_) => GeometryKind::Line,
                MvtGeometry::Points(_) => GeometryKind::Point,
            };
            if let Some(bucket) = families
                .get_mut(family.index())
                .and_then(|buckets| buckets.get_mut(class))
            {
                bucket.push(converted);
            }
        }
    }

    let mut layers = Vec::new();
    for family in GeometryKind::ALL {
        let Some(buckets) = families.get_mut(family.index()) else {
            continue;
        };
        // The fallback bucket keeps the plain family name, which is what makes
        // a single-symbol tile byte-identical to a pre-v1.6 one.
        for (bucket, name) in buckets.iter_mut().zip(bucket_names(family, classes)) {
            if bucket.is_empty() {
                continue;
            }
            layers.push(MvtLayer {
                name,
                extent: LOCAL_EXTENT,
                features: core::mem::take(bucket),
            });
        }
    }
    VectorTile { layers }
}

/// `count` empty feature buckets.
fn new_buckets(count: usize) -> Vec<Vec<MvtFeature>> {
    (0..count).map(|_| Vec::new()).collect()
}

/// One family's bucket names, in bucket order: the fallback first, then one
/// per class. The list `feature_collection_to_tile_with` fills and
/// [`compile_style`] resolves paints for must agree, so they are generated by
/// this one function.
fn bucket_names(family: GeometryKind, classes: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(classes + 1);
    names.push(local_layer_name(family).to_owned());
    for class in 0..classes {
        names.push(local_class_layer_name(family, class));
    }
    names
}

/// Which of the three drawable geometry families a dataset is mostly made
/// of. An ALIAS of the core enum, not a second copy: hit testing ranks
/// candidates "by exactly the same rule the styling and tile paths use",
/// and since tiles v1.3 the styling rule lives in `oxigis-core`
/// ([`oxigis_core::GeometryFamily`]) so the project file can store
/// per-family overrides against it.
pub use oxigis_core::GeometryFamily as GeometryKind;

/// The geometry family most features of `features` belong to.
///
/// Ties go to the more specific style (polygon over line, line over point),
/// which degrades more gracefully on the other kinds; a collection with no
/// geometry at all resolves to [`GeometryKind::Point`], the one symbol that
/// shows *something* for every geometry the tessellator can draw.
#[must_use]
pub fn dominant_geometry_kind(features: &FeatureCollection) -> GeometryKind {
    let mut points = 0_usize;
    let mut lines = 0_usize;
    let mut polygons = 0_usize;
    for feature in &features.features {
        if let Some(geometry) = feature.geometry.as_ref() {
            match geometry_kind(geometry) {
                Some(GeometryKind::Point) => points += 1,
                Some(GeometryKind::Line) => lines += 1,
                Some(GeometryKind::Polygon) => polygons += 1,
                None => {}
            }
        }
    }
    if polygons > 0 && polygons >= lines && polygons >= points {
        GeometryKind::Polygon
    } else if lines > 0 && lines >= points {
        GeometryKind::Line
    } else {
        GeometryKind::Point
    }
}

/// The family one geometry belongs to, recursing into geometry collections and
/// reporting the first member that has one.
///
/// Public because hit testing ranks candidates by this family — a point inside a
/// polygon has to stay clickable — and must rank them by exactly the same rule
/// the styling and tile paths use, rather than by a second private copy of it.
#[must_use]
pub fn geometry_kind(geometry: &Geometry) -> Option<GeometryKind> {
    match geometry {
        Geometry::Point(_) | Geometry::MultiPoint(_) => Some(GeometryKind::Point),
        Geometry::LineString(_) | Geometry::MultiLineString(_) => Some(GeometryKind::Line),
        Geometry::Polygon(_) | Geometry::MultiPolygon(_) => Some(GeometryKind::Polygon),
        Geometry::GeometryCollection(collection) => {
            collection.geometries.iter().find_map(geometry_kind)
        }
    }
}

/// The style a freshly dropped dataset is drawn with, chosen from its dominant
/// geometry kind.
///
/// Polygons get a semi-transparent accent fill with a darker hairline outline,
/// lines a 2 px accent stroke, points a 4 px accent circle with a white stroke.
/// The style panel edits this afterwards; nothing here is remembered.
#[must_use]
pub fn default_style_for(features: &FeatureCollection) -> LayerStyle {
    default_style_for_kind(dominant_geometry_kind(features))
}

/// The style SET a freshly dropped dataset draws with: the dominant
/// family's default as the base, plus a per-family default override for
/// every OTHER family the dataset actually contains — so a mixed
/// collection finally draws all of itself on first sight (the user-visible
/// point of tiles v1.3 item C). For a single-family dataset this is a
/// base-only set: byte-identical to the pre-v1.3 default.
#[must_use]
pub fn default_style_set_for(features: &FeatureCollection) -> LayerStyleSet {
    let dominant = dominant_geometry_kind(features);
    let mut set = LayerStyleSet::new(default_style_for_kind(dominant));
    for feature in &features.features {
        if let Some(geometry) = feature.geometry.as_ref() {
            for family in geometry_families(geometry, 0).iter() {
                if family != dominant && set.override_for(family).is_none() {
                    set.set_override(family, default_style_for_kind(family));
                }
            }
        }
    }
    set
}

/// Every family `features` draws — the union over its geometries.
#[must_use]
pub fn collection_families(features: &FeatureCollection) -> oxigis_core::FamilySet {
    let mut set = oxigis_core::FamilySet::default();
    for feature in &features.features {
        if let Some(geometry) = feature.geometry.as_ref() {
            set = set.union(geometry_families(geometry, 0));
        }
    }
    set
}

/// Every family `geometry` contributes, recursing into collections to the
/// same depth cap the tile conversion walks.
fn geometry_families(geometry: &Geometry, depth: usize) -> oxigis_core::FamilySet {
    let mut set = oxigis_core::FamilySet::default();
    match geometry {
        Geometry::Point(_) | Geometry::MultiPoint(_) => set.insert(GeometryKind::Point),
        Geometry::LineString(_) | Geometry::MultiLineString(_) => {
            set.insert(GeometryKind::Line);
        }
        Geometry::Polygon(_) | Geometry::MultiPolygon(_) => set.insert(GeometryKind::Polygon),
        Geometry::GeometryCollection(collection) => {
            if depth < MAX_GEOMETRY_DEPTH {
                for member in &collection.geometries {
                    set = set.union(geometry_families(member, depth + 1));
                }
            }
        }
    }
    set
}

/// [`default_style_for`], with the geometry kind supplied directly.
#[must_use]
pub fn default_style_for_kind(kind: GeometryKind) -> LayerStyle {
    match kind {
        GeometryKind::Polygon => {
            let mut fill = FillStyle::new(DEFAULT_ACCENT);
            fill.set_opacity(DEFAULT_FILL_OPACITY);
            fill.outline_color = Some(DEFAULT_OUTLINE);
            LayerStyle::Fill(fill)
        }
        GeometryKind::Line => {
            LayerStyle::Line(LineStyle::new(DEFAULT_ACCENT, DEFAULT_LINE_WIDTH_PX))
        }
        GeometryKind::Point => {
            let mut circle = CircleStyle::new(DEFAULT_CIRCLE_RADIUS_PX, DEFAULT_ACCENT);
            circle.stroke_color = Some(Color::from_rgb(0xff, 0xff, 0xff));
            circle.set_stroke_width(1.0);
            LayerStyle::Circle(circle)
        }
    }
}

/// A [`LayerStyle::Symbol`] that labels local features from `property`.
///
/// A convenience for part B's style panel and for tests: every rule of a local
/// layer is keyed by [`LOCAL_LAYER_NAME`], so a symbol style needs nothing else
/// to take effect.
#[must_use]
pub fn local_symbol_style(property: impl Into<String>) -> LayerStyle {
    LayerStyle::Symbol(SymbolStyle::new(property))
}

/// One in-memory vector dataset, ready to draw.
///
/// Holds three things that are derived from each other and must stay in step,
/// which is why they are private behind [`LocalVectorLayer::set_style`]:
///
/// * the parsed [`FeatureCollection`] — kept, not discarded, because §1.2's
///   attribute table reads exactly this and re-parsing the source text for it
///   would double the cost of every dropped file;
/// * the synthetic [`VectorTile`] and its [`MercatorSquare`], the render-side
///   view of the same features;
/// * the [`LayerStyle`] and the [`PaintProgram`] / [`LabelTable`] compiled from
///   it.
///
/// Visibility and opacity live here too rather than being read out of
/// `oxigis_core::Layer`: the render thread touches them every frame and must not
/// have to walk the application's layer stack to do it. Part B keeps the two
/// mirrors in step through [`crate::map_gpu`]'s entry points.
#[derive(Debug, Clone)]
pub struct LocalVectorLayer {
    /// Display name, for logs and the attribute-table title.
    name: String,
    /// The parsed source features, in document order.
    ///
    /// Shared rather than owned outright: the attribute table reads the same
    /// collection from the UI thread while this layer lives on the render side
    /// (see [`crate::local_input::LocalInputState::feature_set`]), and a
    /// multi-megabyte dataset must not be copied to show a table of it. The
    /// collection is immutable once parsed, so a plain [`Arc`] — no lock — is
    /// the whole of the sharing story.
    features: Arc<FeatureCollection>,
    /// The render-side view of `features`.
    tile: VectorTile,
    /// The square `tile`'s coordinates are quantised against.
    square: MercatorSquare,
    /// The style set `program` and `labels` were compiled from.
    style: LayerStyleSet,
    /// How `style`'s renderer partitioned `tile` into class layers — the key
    /// [`LocalVectorLayer::set_style`] compares to decide whether a restyle
    /// needs a re-partition (a class list changed) or only a repaint (a colour
    /// moved). See [`oxigis_core::Renderer::classification`].
    classification: Classification,
    /// Geometry passes, in draw order.
    program: PaintProgram,
    /// Symbol rules; empty unless `style` is a labelling one.
    labels: LabelTable,
    /// Whether the layer is drawn at all.
    visible: bool,
    /// Global multiplier on the layer's alpha, `0..=1`.
    opacity: f32,
    /// Bumped by every [`LocalVectorLayer::set_style`], so a renderer holding a
    /// mesh can tell that the mesh no longer matches the style.
    generation: u32,
}

impl LocalVectorLayer {
    /// Parses GeoJSON text and builds a layer with the default style for its
    /// dominant geometry kind.
    ///
    /// # Errors
    ///
    /// Returns [`LocalVectorError`] if the text is not a GeoJSON
    /// `FeatureCollection`, or if it holds no features at all — an empty layer
    /// is indistinguishable from a failed drop, so it is reported as one.
    pub fn from_geojson(name: impl Into<String>, text: &str) -> Result<Self, LocalVectorError> {
        let features = oxigeo::geojson::reader::feature_collection_from_str(text)
            .map_err(|error| LocalVectorError::new(format!("GeoJSON parse failed: {error}")))?;
        if features.features.is_empty() {
            return Err(LocalVectorError::new("the GeoJSON holds no features"));
        }
        Ok(Self::from_feature_collection(name, features))
    }

    /// Builds a layer from an already-parsed collection, with the default style
    /// for its dominant geometry kind.
    #[must_use]
    pub fn from_feature_collection(name: impl Into<String>, features: FeatureCollection) -> Self {
        let style = default_style_set_for(&features);
        Self::with_style(name, features, style)
    }

    /// Builds a layer with an explicit style — the project-load path, where the
    /// style comes from the saved document rather than from the geometry.
    #[must_use]
    pub fn with_style(
        name: impl Into<String>,
        features: FeatureCollection,
        style: impl Into<LayerStyleSet>,
    ) -> Self {
        Self::with_style_arc(name, Arc::new(features), style)
    }

    /// Same as [`Self::with_style`], but adopting an already-shared collection.
    ///
    /// This is the path that re-styles a dataset the caller already holds an
    /// [`Arc`] of (project load, re-hydration) without deep-copying every
    /// feature to do it.
    #[must_use]
    pub fn with_style_arc(
        name: impl Into<String>,
        features: Arc<FeatureCollection>,
        style: impl Into<LayerStyleSet>,
    ) -> Self {
        let style = style.into();
        let square = collection_square(&features);
        let classification = style.classification();
        let tile = feature_collection_to_tile_with(&features, &square, style.renderer());
        let (program, labels) = compile_style(&style);
        Self {
            name: name.into(),
            features,
            tile,
            square,
            style,
            classification,
            program,
            labels,
            visible: true,
            opacity: 1.0,
            generation: 0,
        }
    }

    /// Display name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Renames the layer. The style keeps matching: rules are keyed by
    /// [`LOCAL_LAYER_NAME`], not by this.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// The parsed source features — the attribute table's data source.
    #[must_use]
    pub fn features(&self) -> &FeatureCollection {
        &self.features
    }

    /// A new handle on the shared source collection — what an attribute-table
    /// store keeps so the panel can read the features without reaching across
    /// the render thread's lock.
    #[must_use]
    pub fn features_arc(&self) -> Arc<FeatureCollection> {
        Arc::clone(&self.features)
    }

    /// Number of features in the source collection, drawable or not.
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.features.features.len()
    }

    /// The synthetic tile, for tessellation and label placement.
    #[must_use]
    pub fn tile(&self) -> &VectorTile {
        &self.tile
    }

    /// Number of MVT features that survived conversion, summed over the
    /// family layers (a mixed source feature counts once per family it
    /// contributes to, exactly as it draws).
    #[must_use]
    pub fn drawable_count(&self) -> usize {
        self.tile
            .layers
            .iter()
            .map(|layer| layer.features.len())
            .sum()
    }

    /// The geometry families this dataset actually draws — read from the
    /// tile the layer already built, so it is exact by construction: the
    /// panel offers a family editor exactly when the renderer would draw
    /// that family.
    #[must_use]
    pub fn families(&self) -> oxigis_core::FamilySet {
        let mut set = oxigis_core::FamilySet::default();
        for layer in &self.tile.layers {
            // Through the name parser, not a three-way match on the constants:
            // a classified layer's polygons live in `features:polygon#0`, and
            // a family probe that missed those would hide the family row for
            // exactly the layers that need it most.
            if let Some(family) = family_of_layer_name(&layer.name) {
                set.insert(family);
            }
        }
        set
    }

    /// The square the tile's coordinates are quantised against.
    #[must_use]
    pub fn square(&self) -> MercatorSquare {
        self.square
    }

    /// The current style set.
    #[must_use]
    pub fn style(&self) -> &LayerStyleSet {
        &self.style
    }

    /// Replaces the style set, recompiling the paint passes and label rules
    /// and bumping [`LocalVectorLayer::generation`].
    ///
    /// Re-partitions the synthetic tile as well — but ONLY when the new
    /// style's [`Classification`] differs from the one the tile was built for.
    /// A colour edit, a width drag, a swapped fallback style all leave the
    /// buckets exactly where they were, so the expensive half of a restyle is
    /// paid once per *classification* change rather than once per frame of a
    /// slider drag.
    pub fn set_style(&mut self, style: impl Into<LayerStyleSet>) {
        let style = style.into();
        let classification = style.classification();
        if classification != self.classification {
            self.tile =
                feature_collection_to_tile_with(&self.features, &self.square, style.renderer());
            self.classification = classification;
        }
        let (program, labels) = compile_style(&style);
        self.style = style;
        self.program = program;
        self.labels = labels;
        self.generation = self.generation.wrapping_add(1);
    }

    /// How the current renderer partitions this layer's features.
    #[must_use]
    pub fn classification(&self) -> &Classification {
        &self.classification
    }

    /// How many class meshes this layer draws, beyond the fallback one.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.style.class_count()
    }

    /// Counts the style edits this layer has seen — a resident mesh built at a
    /// different generation is stale and must be rebuilt.
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// The compiled geometry passes.
    #[must_use]
    pub fn program(&self) -> &PaintProgram {
        &self.program
    }

    /// The compiled symbol rules; empty unless the style is a labelling
    /// [`LayerStyle::Symbol`].
    #[must_use]
    pub fn labels(&self) -> &LabelTable {
        &self.labels
    }

    /// Whether the layer is drawn.
    #[must_use]
    pub fn visible(&self) -> bool {
        self.visible
    }

    /// Shows or hides the layer.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Global alpha multiplier, `0..=1`.
    #[must_use]
    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    /// Sets the global alpha multiplier; non-finite input becomes `1.0` and
    /// everything else is clamped to `0..=1`.
    pub fn set_opacity(&mut self, opacity: f32) {
        self.opacity = if opacity.is_finite() {
            opacity.clamp(0.0, 1.0)
        } else {
            1.0
        };
    }

    /// The RGBA multiplier the vector pipeline applies to every vertex of this
    /// layer — white, with the layer opacity in alpha.
    #[must_use]
    pub fn tint(&self) -> [f32; 4] {
        [1.0, 1.0, 1.0, self.opacity]
    }

    /// Where the layer lands on screen under `view`.
    #[must_use]
    pub fn place(&self, view: MapView) -> TilePlacement {
        self.square.place(view)
    }

    /// Tessellates the layer for an on-screen size of `placement_size_px`.
    ///
    /// Unlike a map tile, a local dataset's on-screen size is *not*
    /// [`MapView::tile_size_px`] — that quantity resets at every integer zoom,
    /// while a dataset grows monotonically with it — so the caller passes the
    /// placement's own size and re-tessellates when it drifts by more than
    /// [`RETESSELLATE_RATIO`].
    ///
    /// # Errors
    ///
    /// Propagates [`RenderError::Tessellation`] from
    /// [`oxigis_render::tessellate_tile`], including for a placement size that
    /// is not finite and positive.
    pub fn tessellate(&self, placement_size_px: f32) -> Result<VectorMesh, RenderError> {
        let params = TessParams::for_tile(
            placement_size_px,
            LOCAL_EXTENT,
            TessParams::DEFAULT_TOLERANCE_PX,
        )?;
        self.program.tessellate(&self.tile, &params)
    }
}

/// Compiles a style set into the geometry passes and symbol rules a local
/// layer draws with — one rule per tile layer the partition can produce, each
/// resolving through [`LayerStyleSet::style_for_class`].
///
/// For a [`Renderer::Single`] set that is exactly the pre-v1.6 list: three
/// rules, one per family, each the family's override or the shared base
/// VERBATIM — which is what makes "defaults = the pre-v1.3 picture" true by
/// construction rather than by inspection. A classified set adds one rule per
/// (family, class), in the same order [`feature_collection_to_tile_with`]
/// emits the buckets.
fn compile_style(set: &LayerStyleSet) -> (PaintProgram, LabelTable) {
    let classes = set.class_count();
    let mut rules: Vec<(String, LayerStyle)> = Vec::with_capacity(3 * (classes + 1));
    for family in GeometryKind::ALL {
        for (bucket, name) in bucket_names(family, classes).into_iter().enumerate() {
            // Bucket 0 is the fallback; class `c` is bucket `c + 1`. Owned
            // rather than borrowed because `style_for_class` COMPOSES a class
            // over its family (a Fill class on a point family becomes a
            // recoloured circle, not a fill that would draw nothing) — the
            // composed value exists nowhere in the model to borrow from.
            let class = bucket.checked_sub(1);
            rules.push((name, set.style_for_class(family, class)));
        }
    }
    (
        PaintProgram::from_rules(rules.iter().map(|(name, style)| (name.as_str(), style))),
        label_table(rules.iter().map(|(name, style)| (name.as_str(), style))),
    )
}

pub mod classify;

#[cfg(test)]
mod tests;
