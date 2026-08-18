//! Placing a COG on the Web Mercator tile grid.
//!
//! This is the part with no counterpart in `oxigeo-wasm`: its viewer draws a
//! COG in the file's own pixel space on a canvas, so it never has to answer
//! "which bytes of this file does slippy-map tile `z/x/y` need, and how do those
//! pixels land in a 256×256 RGBA tile". Three steps:
//!
//! 1. [`CogMetadata::select_level`] — pick the pyramid level whose ground
//!    resolution matches the requested zoom.
//! 2. [`CogMetadata::plan_tile`] — intersect the map tile with the image and
//!    list the source tiles (and byte ranges) that cover it.
//! 3. [`CogMetadata::compose_tile`] — resample the decoded source tiles into
//!    one [`DecodedTile`], leaving anything outside the image transparent.
//!
//! # Resolution-independent level choice
//!
//! Levels are compared in **normalised world units** (the `0..1` Web Mercator
//! square), not in CRS units, so metres (EPSG:3857) and degrees (EPSG:4326) go
//! through one code path: a level's pixel size is divided by the CRS's axis span
//! before being compared with a tile's `1 / (256 · 2^z)`.
//!
//! # Reprojection
//!
//! EPSG:3857 and EPSG:4326 place on the grid one axis at a time — easting
//! depends only on `world.x`, northing only on `world.y` — which is what lets
//! [`CogMetadata::compose_tile`] compute a source column once per output column.
//! A UTM COG does not: the map-tile pixel → source-pixel mapping goes
//!
//! ```text
//! output pixel → normalised world → lon/lat → UTM easting/northing → source pixel
//! ```
//!
//! and both output axes feed both source axes. `Placement` holds whichever of
//! the two mappings applies, and every step that used to be per-axis
//! ([`CogMetadata::world_bounds`], [`CogMetadata::world_pixel_size`],
//! [`CogMetadata::plan_tile_at`], [`CogMetadata::compose_tile`]) has a projected
//! branch beside its native one.
//!
//! Two properties are worth stating because they are what make the projected
//! branch safe rather than merely plausible:
//!
//! * **Planning and composition share one mapping.** `plan_tile_at` derives the
//!   source-pixel bounding box by running `for_each_projected_pixel` — the
//!   *same* per-pixel transform `compose_tile` later resamples with — instead of
//!   sampling the tile boundary and hoping the interior stays inside. A sampled
//!   bbox that came out even one pixel short would show up as transparent holes
//!   and edge seams, silently: `compose_tile` skips a source pixel whose tile is
//!   not among the ones that were fetched. Costing a second transform pass buys
//!   consistency by construction.
//! * **Out-of-zone points are refused, not approximated.** Transverse Mercator
//!   returns a perfectly finite easting 40° off its central meridian, and a
//!   finite-but-meaningless easting is how a raster gets drawn in the wrong
//!   place. [`super::tmerc`] returns `None` past
//!   [`super::tmerc::TMERC_MAX_LON_OFFSET_DEG`], and those pixels stay
//!   transparent.
//!
//! # Resampling
//!
//! Nearest neighbour. It is the only filter that is correct for categorical
//! rasters (land cover, masks), it needs no source pixels outside the requested
//! window, and at the level selected above the source and destination
//! resolutions are within a factor of two, so the aliasing a box filter would
//! remove is slight.

use crate::cog::meta::{CogCrs, CogGeoTransform, CogMetadata};
use crate::cog::tmerc::TransverseMercator;
use crate::error::RenderError;
use crate::mercator::{EARTH_CIRCUMFERENCE_M, TILE_SIZE_PX, TileId};
use crate::renderer::DecodedTile;
use crate::source::ByteRange;

/// Edge length in pixels of the RGBA tiles this module composes.
pub const COG_OUTPUT_TILE_PX: u32 = TILE_SIZE_PX as u32;

/// Largest number of source tiles one map tile may be composed from.
///
/// Four is the norm once a level has been selected (a map tile straddles at
/// most two source tiles per axis); the cap only bites when a COG has no
/// overviews and the map is zoomed far out, where reading hundreds of
/// full-resolution tiles per screen tile is not something to do quietly.
pub const COG_MAX_SOURCE_TILES: usize = 64;

/// One source tile a map tile needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CogTileRef {
    /// Column in the level's tile grid.
    pub tile_x: u32,
    /// Row in the level's tile grid.
    pub tile_y: u32,
    /// Bytes to fetch, or `None` for a sparse (never-written) tile.
    pub range: Option<ByteRange>,
}

/// Everything needed to fetch and build one map tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CogTilePlan {
    /// The map tile being built.
    pub tile: TileId,
    /// Pyramid level the source tiles come from.
    pub level: usize,
    /// Source tiles covering the map tile, in row-major order.
    pub sources: Vec<CogTileRef>,
    /// Spacing, in source tiles, between consecutive planned tiles along
    /// each axis. `1` for an ordinary plan; greater than `1` when the level
    /// has no overview coarse enough for the zoom, so only every
    /// `stride`-th tile is read and [`CogMetadata::compose_tile`] leaves the
    /// gaps transparent — a coarse preview rather than
    /// [`RenderError::Unsupported`].
    pub stride: u32,
}

impl CogTilePlan {
    /// Byte ranges that actually have to be fetched (sparse tiles excluded).
    #[must_use]
    pub fn ranges(&self) -> Vec<ByteRange> {
        self.sources
            .iter()
            .filter_map(|source| source.range)
            .collect()
    }
}

/// A decoded source tile, ready to be resampled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CogSourceTile {
    /// Column in the level's tile grid.
    pub tile_x: u32,
    /// Row in the level's tile grid.
    pub tile_y: u32,
    /// RGBA8 pixels, `tile_width × tile_height × 4` bytes.
    pub rgba: Vec<u8>,
}

/// Samples per axis used to bound a projected image's Web Mercator footprint.
///
/// 17 × 17 over the image's pixel space: enough that the residual between the
/// sampled bounding box and the true one is far below the half-step padding
/// [`CogMetadata::utm_extent`] adds, while costing under three hundred inverse
/// projections.
const UTM_EXTENT_SAMPLES: u32 = 17;

/// [`Placement::of`] plus, for a projected CRS, its
/// [`CogMetadata::utm_extent`] footprint.
///
/// Built once by every public entry point ([`CogMetadata::select_level`],
/// [`CogMetadata::plan_tile`], [`CogMetadata::plan_tile_at`],
/// [`CogMetadata::world_pixel_size`]) and threaded through the private
/// `_with` helpers, rather than recomputed per pyramid level and again for
/// the tile plan: `utm_extent` depends only on the file's metadata, never on
/// the level or tile, so an unfused `plan_tile` (level count `L`) would
/// otherwise run its 17×17 projection grid `L + 1` times for one result.
struct PlacementContext {
    placement: Placement,
    /// `Some` exactly when `placement` is [`Placement::Projected`] — every
    /// reader below may assume the converse (an axis-pair placement) needs
    /// no footprint and must not fail just because this is `None`.
    extent: Option<UtmExtent>,
}

/// Normalised world easting of a longitude, without clamping.
fn world_x_of_lon(lon_deg: f64) -> f64 {
    (lon_deg + 180.0) / 360.0
}

/// Longitude of a normalised world easting, without clamping.
fn lon_of_world_x(world_x: f64) -> f64 {
    world_x * 360.0 - 180.0
}

/// Normalised world northing of a latitude, without clamping.
///
/// `LonLat::to_world` saturates at the projection cut-off; a COG extent that
/// reaches past it must still map monotonically, so the spherical Mercator
/// ordinate is taken directly here.
fn world_y_of_lat(lat_deg: f64) -> f64 {
    let lat = lat_deg.to_radians();
    0.5 * (1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / core::f64::consts::PI)
}

/// Latitude of a normalised world northing, without clamping — the inverse of
/// [`world_y_of_lat`].
fn lat_of_world_y(world_y: f64) -> f64 {
    (core::f64::consts::PI * (1.0 - 2.0 * world_y))
        .sinh()
        .atan()
        .to_degrees()
}

/// How a CRS's coordinates reach the normalised Web Mercator square.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Placement {
    /// EPSG:3857 metres: an affine map per axis.
    WebMercator,
    /// EPSG:4326 degrees: affine in easting, spherical Mercator in northing.
    Geographic,
    /// UTM metres: a full two-dimensional projection.
    Projected(TransverseMercator),
}

impl Placement {
    /// The placement `crs` needs.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] for a [`CogCrs::Utm`] whose zone
    /// number cannot be turned into a projection. [`CogCrs::from_epsg`] only
    /// produces zones `1..=60`, so this is unreachable in practice — it exists
    /// so the construction never has to be unwrapped.
    fn of(crs: CogCrs) -> Result<Self, RenderError> {
        match crs {
            CogCrs::WebMercator => Ok(Self::WebMercator),
            CogCrs::Geographic => Ok(Self::Geographic),
            CogCrs::Utm { zone, north } => TransverseMercator::wgs84_utm(zone, north)
                .map(Self::Projected)
                .ok_or_else(|| {
                    RenderError::Unsupported(format!(
                        "COG declares UTM zone {zone}, which is not a valid zone"
                    ))
                }),
        }
    }

    /// The per-axis pair, or `None` when a two-dimensional projection is needed.
    const fn axis_pair(self) -> Option<AxisPlacement> {
        match self {
            Self::WebMercator => Some(AxisPlacement::WebMercator),
            Self::Geographic => Some(AxisPlacement::Geographic),
            Self::Projected(_) => None,
        }
    }

    /// Normalised world position of a CRS coordinate, or `None` when the point
    /// cannot be placed (outside the projection's valid range).
    fn to_world(self, crs_x: f64, crs_y: f64) -> Option<(f64, f64)> {
        match self {
            Self::WebMercator | Self::Geographic => {
                let axis = self.axis_pair()?;
                Some((axis.world_x(crs_x), axis.world_y(crs_y)))
            }
            Self::Projected(projection) => {
                let (lon, lat) = projection.inverse(crs_x, crs_y)?;
                Some((world_x_of_lon(lon), world_y_of_lat(lat)))
            }
        }
    }
}

/// The separable half of [`Placement`], for the axis-at-a-time fast paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AxisPlacement {
    /// EPSG:3857 metres.
    WebMercator,
    /// EPSG:4326 degrees.
    Geographic,
}

impl AxisPlacement {
    /// Normalised world easting of a CRS easting.
    fn world_x(self, crs_x: f64) -> f64 {
        match self {
            Self::WebMercator => crs_x / EARTH_CIRCUMFERENCE_M + 0.5,
            Self::Geographic => world_x_of_lon(crs_x),
        }
    }

    /// Normalised world northing of a CRS northing.
    fn world_y(self, crs_y: f64) -> f64 {
        match self {
            Self::WebMercator => 0.5 - crs_y / EARTH_CIRCUMFERENCE_M,
            Self::Geographic => world_y_of_lat(crs_y),
        }
    }

    /// CRS easting of a normalised world easting.
    fn crs_x(self, world_x: f64) -> f64 {
        match self {
            Self::WebMercator => (world_x - 0.5) * EARTH_CIRCUMFERENCE_M,
            Self::Geographic => lon_of_world_x(world_x),
        }
    }

    /// CRS northing of a normalised world northing.
    fn crs_y(self, world_y: f64) -> f64 {
        match self {
            Self::WebMercator => (0.5 - world_y) * EARTH_CIRCUMFERENCE_M,
            Self::Geographic => lat_of_world_y(world_y),
        }
    }

    /// Full extent of the CRS along the easting axis, in CRS units — how far
    /// a coordinate travels before it names the same place again: 360° for
    /// geographic degrees, the Mercator circumference for EPSG:3857 metres.
    /// Normalises a pixel size into world units, and is the shift
    /// [`wrapped_pixel_span_x`] tries for a request near the antimeridian.
    const fn x_span(self) -> f64 {
        match self {
            Self::WebMercator => EARTH_CIRCUMFERENCE_M,
            Self::Geographic => 360.0,
        }
    }
}

/// The Web Mercator footprint of a projected image, plus the latitude its
/// ground resolution should be judged at.
#[derive(Debug, Clone, Copy, PartialEq)]
struct UtmExtent {
    /// Conservative normalised world bounding box, `(min_x, min_y, max_x, max_y)`.
    bounds: (f64, f64, f64, f64),
    /// Largest absolute latitude the image covers, in degrees.
    max_abs_lat_deg: f64,
}

/// Visits every output pixel of `tile` with the source-pixel position the
/// projected mapping puts it at.
///
/// `visit` receives `(out_x, out_y, source_pixel_x, source_pixel_y)` for the
/// pixel's **centre**; pixels the projection refuses (see
/// [`super::tmerc::TMERC_MAX_LON_OFFSET_DEG`]) are skipped entirely.
///
/// This is the single definition of the projected inverse mapping: both
/// [`CogMetadata::plan_tile_at`] (to bound the source pixels a tile can touch)
/// and [`CogMetadata::compose_tile`] (to resample them) go through it, so the
/// two cannot disagree.
///
/// The longitude/latitude step *is* separable — longitude depends only on
/// `world.x` and latitude only on `world.y` — so the 512 inverse-Mercator
/// evaluations are hoisted out of the double loop and only the Transverse
/// Mercator forward runs per pixel.
fn for_each_projected_pixel(
    projection: TransverseMercator,
    transform: CogGeoTransform,
    tile: TileId,
    mut visit: impl FnMut(usize, usize, f64, f64),
) {
    let edge = COG_OUTPUT_TILE_PX as usize;
    let north_west = tile.north_west();
    let south_east = tile.south_east();
    let span_x = south_east.x - north_west.x;
    let span_y = south_east.y - north_west.y;
    #[expect(
        clippy::cast_precision_loss,
        reason = "edge is 256; every index is exactly representable"
    )]
    let scale = 1.0 / edge as f64;
    #[expect(
        clippy::cast_precision_loss,
        reason = "edge is 256; every index is exactly representable"
    )]
    let longitudes: Vec<f64> = (0..edge)
        .map(|out_x| lon_of_world_x(north_west.x + (out_x as f64 + 0.5) * scale * span_x))
        .collect();
    #[expect(
        clippy::cast_precision_loss,
        reason = "edge is 256; every index is exactly representable"
    )]
    let latitudes: Vec<f64> = (0..edge)
        .map(|out_y| lat_of_world_y(north_west.y + (out_y as f64 + 0.5) * scale * span_y))
        .collect();

    projection.forward_grid(
        &longitudes,
        &latitudes,
        |out_x, out_y, easting, northing| {
            let (pixel_x, pixel_y) = transform.to_pixel(easting, northing);
            visit(out_x, out_y, pixel_x, pixel_y);
        },
    );
}

impl CogMetadata {
    /// Builds this file's [`PlacementContext`].
    fn placement_context(&self) -> Result<PlacementContext, RenderError> {
        let placement = Placement::of(self.crs()?)?;
        let extent = match placement {
            Placement::Projected(_) => Some(self.utm_extent(placement)?),
            Placement::WebMercator | Placement::Geographic => None,
        };
        Ok(PlacementContext { placement, extent })
    }

    /// Extent of the image in normalised world coordinates, as
    /// `(min_x, min_y, max_x, max_y)`, clamped to the `0..=1` unit square the
    /// tile grid can actually represent — the image's *drawable* footprint,
    /// not necessarily its full geographic extent. A global EPSG:4326
    /// raster's poles sit past the Web Mercator cut-off, where
    /// `world_y_of_lat` evaluates `tan`/`sec` near their asymptote — a huge
    /// but finite number, not the saturated 0/1 [`crate::mercator::LonLat::to_world`]
    /// would give — so without the clamp this would report a box roughly
    /// eleven times the unit square. The `Placement::Projected` branch is
    /// deliberately left unclamped: scoped to axis-separable CRSs, and a
    /// near-polar UTM zone is not a real-world case.
    ///
    /// # Errors
    ///
    /// Propagates [`CogMetadata::crs`] and [`CogMetadata::geo_transform`].
    pub fn world_bounds(&self) -> Result<(f64, f64, f64, f64), RenderError> {
        let placement = Placement::of(self.crs()?)?;
        let transform = self.geo_transform()?;
        let base = self.base_level()?;
        if let Some(axis) = placement.axis_pair() {
            let (left, top) = transform.to_crs(0.0, 0.0);
            let (right, bottom) = transform.to_crs(f64::from(base.width), f64::from(base.height));
            // Both mappings are monotone per axis, so the corners bound the extent.
            return Ok((
                axis.world_x(left).clamp(0.0, 1.0),
                axis.world_y(top).clamp(0.0, 1.0),
                axis.world_x(right).clamp(0.0, 1.0),
                axis.world_y(bottom).clamp(0.0, 1.0),
            ));
        }
        Ok(self.utm_extent(placement)?.bounds)
    }

    /// Web Mercator footprint and maximum latitude of a projected image.
    ///
    /// A UTM rectangle is *not* a Web Mercator rectangle: its edges bow, so the
    /// four corners under-report the footprint. Longitude and latitude are both
    /// smooth functions of easting/northing with no interior critical point, so
    /// their extremes over the image lie on its boundary — but sampling the
    /// boundary alone would leave the result sensitive to how the bow is cut, so
    /// a full [`UTM_EXTENT_SAMPLES`]² grid is taken and the bounding box is then
    /// padded by half a sample step. Over-reporting only costs a little extent
    /// culling; under-reporting would cull tiles that do overlap.
    ///
    /// # Errors
    ///
    /// * [`RenderError::Unsupported`] when no sample projects — a georeference
    ///   claiming a UTM zone but holding coordinates nowhere near it.
    /// * Propagates [`CogMetadata::geo_transform`] and
    ///   [`CogMetadata::base_level`].
    fn utm_extent(&self, placement: Placement) -> Result<UtmExtent, RenderError> {
        let transform = self.geo_transform()?;
        let base = self.base_level()?;
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut max_abs_lat_deg = 0.0_f64;
        let last = f64::from(UTM_EXTENT_SAMPLES - 1);
        for row in 0..UTM_EXTENT_SAMPLES {
            for column in 0..UTM_EXTENT_SAMPLES {
                let pixel_x = f64::from(base.width) * f64::from(column) / last;
                let pixel_y = f64::from(base.height) * f64::from(row) / last;
                let (crs_x, crs_y) = transform.to_crs(pixel_x, pixel_y);
                let Some((world_x, world_y)) = placement.to_world(crs_x, crs_y) else {
                    continue;
                };
                if !world_x.is_finite() || !world_y.is_finite() {
                    continue;
                }
                min_x = min_x.min(world_x);
                max_x = max_x.max(world_x);
                min_y = min_y.min(world_y);
                max_y = max_y.max(world_y);
                max_abs_lat_deg = max_abs_lat_deg.max(lat_of_world_y(world_y).abs());
            }
        }
        if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
            return Err(RenderError::Unsupported(
                "COG's georeference does not fall inside the UTM zone it declares".to_owned(),
            ));
        }
        // Half a sample step of slack in each direction, which dominates the
        // curvature the grid missed between neighbouring samples.
        let pad_x = (max_x - min_x) / last * 0.5;
        let pad_y = (max_y - min_y) / last * 0.5;
        Ok(UtmExtent {
            bounds: (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y),
            max_abs_lat_deg,
        })
    }

    /// Size of one pixel of `level` in normalised world units along easting.
    ///
    /// For a projected CRS a pixel is a fixed number of *ground* metres, and how
    /// much of the normalised world that spans depends on the latitude — by
    /// `1 / cos φ`, which is a factor of two by 60°, i.e. a whole zoom level. The
    /// image's largest absolute latitude is used, because that maximises the
    /// world-units-per-metre and so errs toward selecting a *finer* level:
    /// [`CogMetadata::select_level`] has no tile context to be more precise
    /// with, and reading more detail than needed is the safe direction.
    ///
    /// # Errors
    ///
    /// Propagates [`CogMetadata::crs`], [`CogMetadata::level_transform`] and
    /// `CogMetadata::utm_extent`.
    pub fn world_pixel_size(&self, level: usize) -> Result<f64, RenderError> {
        self.world_pixel_size_with(level, &self.placement_context()?)
    }

    /// [`CogMetadata::world_pixel_size`] against an already-built
    /// [`PlacementContext`].
    fn world_pixel_size_with(
        &self,
        level: usize,
        context: &PlacementContext,
    ) -> Result<f64, RenderError> {
        let transform = self.level_transform(level)?;
        if let Some(axis) = context.placement.axis_pair() {
            return Ok(transform.pixel_size_x / axis.x_span());
        }
        let Some(extent) = context.extent.as_ref() else {
            return Err(RenderError::Unsupported(
                "a projected CRS needs its footprint to size a pixel".to_owned(),
            ));
        };
        let cosine = extent.max_abs_lat_deg.to_radians().cos();
        if !cosine.is_finite() || cosine <= 0.0 {
            return Err(RenderError::Unsupported(
                "COG's extent reaches the pole, where its ground resolution has no Web Mercator \
                 equivalent"
                    .to_owned(),
            ));
        }
        Ok(transform.pixel_size_x / (EARTH_CIRCUMFERENCE_M * cosine))
    }

    /// Picks the pyramid level to serve `zoom` from.
    ///
    /// Returns the **coarsest** level that is still at least as detailed as the
    /// tile needs, so the fewest bytes are read for a given screen resolution;
    /// when even level 0 is coarser than the tile (the map is zoomed past the
    /// image's native resolution), level 0 is used and the pixels are magnified.
    ///
    /// # Errors
    ///
    /// Propagates [`CogMetadata::world_pixel_size`].
    pub fn select_level(&self, zoom: u8) -> Result<usize, RenderError> {
        self.select_level_with(zoom, &self.placement_context()?)
    }

    /// [`CogMetadata::select_level`] against an already-built
    /// [`PlacementContext`].
    fn select_level_with(
        &self,
        zoom: u8,
        context: &PlacementContext,
    ) -> Result<usize, RenderError> {
        let target = 1.0 / (TILE_SIZE_PX * f64::from(1u32 << zoom.min(30)));
        let mut chosen = 0usize;
        for level in 0..self.level_count() {
            let size = self.world_pixel_size_with(level, context)?;
            if size <= target {
                chosen = level;
            } else {
                break;
            }
        }
        Ok(chosen)
    }

    /// Lists the source tiles and byte ranges map tile `tile` is built from.
    ///
    /// `Ok(None)` means the tile does not overlap the image at all, which the
    /// caller should treat as "nothing to draw here" rather than as a failure.
    /// A level with no overview coarse enough for the zoom is not an error
    /// either: the plan is decimated instead — see [`CogTilePlan::stride`].
    ///
    /// # Errors
    ///
    /// * [`RenderError::Unsupported`] when the file's CRS or georeference
    ///   cannot be placed on the tile grid.
    /// * [`RenderError::InvalidRange`] for a malformed tile directory.
    pub fn plan_tile(&self, tile: TileId) -> Result<Option<CogTilePlan>, RenderError> {
        let context = self.placement_context()?;
        let level = self.select_level_with(tile.z, &context)?;
        self.plan_tile_at_with(tile, level, &context)
    }

    /// [`CogMetadata::plan_tile`] with the pyramid level chosen by the caller.
    ///
    /// # Errors
    ///
    /// As [`CogMetadata::plan_tile`], plus [`RenderError::Decode`] for a level
    /// index that does not exist.
    pub fn plan_tile_at(
        &self,
        tile: TileId,
        level: usize,
    ) -> Result<Option<CogTilePlan>, RenderError> {
        self.plan_tile_at_with(tile, level, &self.placement_context()?)
    }

    /// [`CogMetadata::plan_tile_at`] against an already-built
    /// [`PlacementContext`].
    fn plan_tile_at_with(
        &self,
        tile: TileId,
        level: usize,
        context: &PlacementContext,
    ) -> Result<Option<CogTilePlan>, RenderError> {
        let transform = self.level_transform(level)?;
        let source = self
            .level(level)
            .ok_or_else(|| RenderError::Decode(format!("COG has no level {level}")))?;

        let (first_x, last_x, first_y, last_y) = if let Some(axis) = context.placement.axis_pair() {
            let north_west = tile.north_west();
            let south_east = tile.south_east();
            let west_crs_x = axis.crs_x(north_west.x);
            let east_crs_x = axis.crs_x(south_east.x);
            let (_, top) = transform.to_pixel(west_crs_x, axis.crs_y(north_west.y));
            let (_, bottom) = transform.to_pixel(east_crs_x, axis.crs_y(south_east.y));
            let Some((first_x, last_x)) = wrapped_pixel_span_x(
                transform,
                axis.x_span(),
                west_crs_x,
                east_crs_x,
                source.width,
            ) else {
                return Ok(None);
            };
            let Some((first_y, last_y)) = pixel_span(top, bottom, source.height) else {
                return Ok(None);
            };
            (first_x, last_x, first_y, last_y)
        } else {
            let Placement::Projected(projection) = context.placement else {
                return Err(RenderError::Unsupported(
                    "COG CRS has neither an axis mapping nor a projection".to_owned(),
                ));
            };
            let Some(extent) = context.extent.as_ref() else {
                return Err(RenderError::Unsupported(
                    "a projected CRS needs its footprint to plan a tile".to_owned(),
                ));
            };
            // Cheap rejection first: a projected pass over 65 536 pixels is ~50×
            // the cost of the footprint sample, and most requested tiles miss.
            let (min_x, min_y, max_x, max_y) = extent.bounds;
            let north_west = tile.north_west();
            let south_east = tile.south_east();
            if south_east.x < min_x
                || north_west.x > max_x
                || south_east.y < min_y
                || north_west.y > max_y
            {
                return Ok(None);
            }
            let Some(span) = projected_pixel_span(projection, transform, tile, source) else {
                return Ok(None);
            };
            span
        };

        let first_tile_x = first_x / source.tile_width;
        let last_tile_x = last_x / source.tile_width;
        let first_tile_y = first_y / source.tile_height;
        let last_tile_y = last_y / source.tile_height;
        let needed = (u64::from(last_tile_x - first_tile_x) + 1)
            * (u64::from(last_tile_y - first_tile_y) + 1);
        let stride = decimation_stride(needed);

        let mut sources = Vec::new();
        'rows: for tile_y in (first_tile_y..=last_tile_y).step_by(stride as usize) {
            for tile_x in (first_tile_x..=last_tile_x).step_by(stride as usize) {
                if source.tile_index(tile_x, tile_y).is_none() {
                    continue;
                }
                sources.push(CogTileRef {
                    tile_x,
                    tile_y,
                    range: source.tile_range(tile_x, tile_y)?,
                });
                // Belt-and-suspenders: `decimation_stride` already keeps the
                // strided grid within budget (see its doc), but the plan
                // must stay bounded regardless of any float-precision slack
                // in that derivation.
                if sources.len() >= COG_MAX_SOURCE_TILES {
                    break 'rows;
                }
            }
        }
        if sources.is_empty() {
            return Ok(None);
        }
        Ok(Some(CogTilePlan {
            tile,
            level,
            sources,
            stride,
        }))
    }

    /// Resamples decoded source tiles into the map tile `plan` describes.
    ///
    /// Output pixels that fall outside the image, or inside a source tile that
    /// was not supplied (a sparse tile, or one whose fetch failed), are fully
    /// transparent — which is what lets a caller draw the result over a basemap.
    ///
    /// # Errors
    ///
    /// * [`RenderError::Decode`] for a level index that does not exist or a
    ///   source tile whose pixel buffer is the wrong size.
    /// * Propagates [`CogMetadata::crs`] and [`CogMetadata::level_transform`].
    /// * [`RenderError::InvalidTileImage`] from [`DecodedTile::new`].
    pub fn compose_tile(
        &self,
        plan: &CogTilePlan,
        sources: &[CogSourceTile],
    ) -> Result<DecodedTile, RenderError> {
        let placement = Placement::of(self.crs()?)?;
        let transform = self.level_transform(plan.level)?;
        let level = self
            .level(plan.level)
            .ok_or_else(|| RenderError::Decode(format!("COG has no level {}", plan.level)))?;
        let tile_width = level.tile_width as usize;
        let tile_height = level.tile_height as usize;
        let expected = tile_width
            .checked_mul(tile_height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| RenderError::Decode("COG tile geometry overflows".to_owned()))?;
        for source in sources {
            if source.rgba.len() != expected {
                return Err(RenderError::Decode(format!(
                    "COG source tile ({}, {}) holds {} bytes, expected {expected}",
                    source.tile_x,
                    source.tile_y,
                    source.rgba.len()
                )));
            }
        }
        let lookup = SourceLookup::new(sources);

        let edge = COG_OUTPUT_TILE_PX as usize;
        let mut rgba = vec![0u8; edge * edge * 4];

        if let Some(axis) = placement.axis_pair() {
            let north_west = plan.tile.north_west();
            let south_east = plan.tile.south_east();
            let span_x = south_east.x - north_west.x;
            let span_y = south_east.y - north_west.y;

            // Both axes are separable, so the source column of every output
            // column is computed once instead of once per pixel.
            let columns: Vec<Option<u32>> = (0..edge)
                .map(|out_x| {
                    let world = north_west.x + (out_x as f64 + 0.5) / edge as f64 * span_x;
                    let (pixel, _) = transform.to_pixel(axis.crs_x(world), transform.origin_y);
                    to_index(pixel, level.width)
                })
                .collect();

            for out_y in 0..edge {
                let world = north_west.y + (out_y as f64 + 0.5) / edge as f64 * span_y;
                let (_, pixel) = transform.to_pixel(transform.origin_x, axis.crs_y(world));
                let Some(source_y) = to_index(pixel, level.height) else {
                    continue;
                };
                for (out_x, column) in columns.iter().enumerate() {
                    let Some(source_x) = *column else {
                        continue;
                    };
                    blit_pixel(
                        &lookup,
                        tile_width,
                        (level.tile_width, level.tile_height),
                        (source_x, source_y),
                        out_y * edge + out_x,
                        &mut rgba,
                    );
                }
            }
        } else {
            let Placement::Projected(projection) = placement else {
                return Err(RenderError::Unsupported(
                    "COG CRS has neither an axis mapping nor a projection".to_owned(),
                ));
            };
            let (width, height) = (level.width, level.height);
            let geometry = (level.tile_width, level.tile_height);
            for_each_projected_pixel(
                projection,
                transform,
                plan.tile,
                |out_x, out_y, pixel_x, pixel_y| {
                    let (Some(source_x), Some(source_y)) =
                        (to_index(pixel_x, width), to_index(pixel_y, height))
                    else {
                        return;
                    };
                    blit_pixel(
                        &lookup,
                        tile_width,
                        geometry,
                        (source_x, source_y),
                        out_y * edge + out_x,
                        &mut rgba,
                    );
                },
            );
        }

        DecodedTile::new(COG_OUTPUT_TILE_PX, COG_OUTPUT_TILE_PX, rgba)
    }
}

/// Largest indexed lookup [`SourceLookup::new`] will build — comfortably
/// above the [`COG_MAX_SOURCE_TILES`] a plan this crate produces can ever
/// need, so only a `sources` slice assembled some other way can hit the
/// [`SourceLookup::Linear`] fallback.
const SOURCE_LOOKUP_MAX_SLOTS: u64 = 4 * COG_MAX_SOURCE_TILES as u64;

/// A `(tile_x, tile_y) → &CogSourceTile` lookup, built once per composed map
/// tile instead of scanning `sources` for every one of its
/// [`COG_OUTPUT_TILE_PX`]² output pixels.
enum SourceLookup<'a> {
    /// `sources`' tile coordinates span a small rectangle: direct-indexed.
    Grid {
        first_tile_x: u32,
        first_tile_y: u32,
        cols: usize,
        slots: Vec<Option<&'a CogSourceTile>>,
    },
    /// Still correct, just an `O(n)` scan: for an empty `sources`, or one
    /// spread too wide for [`SOURCE_LOOKUP_MAX_SLOTS`] to index directly.
    Linear(&'a [CogSourceTile]),
}

impl<'a> SourceLookup<'a> {
    fn new(sources: &'a [CogSourceTile]) -> Self {
        let Some(first) = sources.first() else {
            return Self::Linear(sources);
        };
        let (mut first_x, mut first_y) = (first.tile_x, first.tile_y);
        let (mut last_x, mut last_y) = (first.tile_x, first.tile_y);
        for source in sources {
            first_x = first_x.min(source.tile_x);
            first_y = first_y.min(source.tile_y);
            last_x = last_x.max(source.tile_x);
            last_y = last_y.max(source.tile_y);
        }
        let cols_u64 = u64::from(last_x - first_x) + 1;
        let rows_u64 = u64::from(last_y - first_y) + 1;
        let Some(count) = cols_u64
            .checked_mul(rows_u64)
            .filter(|&count| count <= SOURCE_LOOKUP_MAX_SLOTS)
        else {
            return Self::Linear(sources);
        };
        let (Ok(cols), Ok(slots_len)) = (usize::try_from(cols_u64), usize::try_from(count)) else {
            return Self::Linear(sources);
        };
        let mut slots = vec![None; slots_len];
        for source in sources {
            // `count <= SOURCE_LOOKUP_MAX_SLOTS` above bounds both factors,
            // so this index is always in range.
            let column = (source.tile_x - first_x) as usize;
            let row = (source.tile_y - first_y) as usize;
            slots[row * cols + column] = Some(source);
        }
        Self::Grid {
            first_tile_x: first_x,
            first_tile_y: first_y,
            cols,
            slots,
        }
    }

    fn get(&self, tile_x: u32, tile_y: u32) -> Option<&'a CogSourceTile> {
        match self {
            Self::Grid {
                first_tile_x,
                first_tile_y,
                cols,
                slots,
            } => {
                let column = tile_x.checked_sub(*first_tile_x)? as usize;
                let row = tile_y.checked_sub(*first_tile_y)? as usize;
                if column >= *cols {
                    return None;
                }
                // Checked: `row` is attacker-reachable, and `row * cols` can
                // overflow a 32-bit `usize` (wasm32).
                let index = row.checked_mul(*cols)?.checked_add(column)?;
                slots.get(index).copied().flatten()
            }
            Self::Linear(sources) => sources
                .iter()
                .find(|candidate| candidate.tile_x == tile_x && candidate.tile_y == tile_y),
        }
    }
}

/// Copies one source pixel into output pixel `out_index` of `rgba`.
///
/// Does nothing when `lookup` has no source tile at `(source_x, source_y)`'s
/// tile — a sparse tile, a fetch failure, or (for a decimated plan) simply
/// one that was never read — which leaves the output pixel at its initial
/// fully transparent value.
///
/// `stride` is the decoded buffer's row length in pixels (the level's tile
/// width) and `geometry` its `(tile_width, tile_height)` in `u32`, i.e. the
/// level's tile grid step.
fn blit_pixel(
    lookup: &SourceLookup<'_>,
    stride: usize,
    geometry: (u32, u32),
    source: (u32, u32),
    out_index: usize,
    rgba: &mut [u8],
) {
    let (source_x, source_y) = source;
    let (tile_width, tile_height) = geometry;
    let tile_x = source_x / tile_width;
    let tile_y = source_y / tile_height;
    let Some(found) = lookup.get(tile_x, tile_y) else {
        return;
    };
    let inner_x = (source_x % tile_width) as usize;
    let inner_y = (source_y % tile_height) as usize;
    let read = (inner_y * stride + inner_x) * 4;
    let Some(pixel) = found.rgba.get(read..read + 4) else {
        return;
    };
    let write = out_index * 4;
    if let Some(slot) = rgba.get_mut(write..write + 4) {
        slot.copy_from_slice(pixel);
    }
}

/// Source-pixel bounding box a projected map tile can read, as
/// `(first_x, last_x, first_y, last_y)`, or `None` when no output pixel of the
/// tile lands inside the image.
///
/// Derived from [`for_each_projected_pixel`], i.e. from exactly the pixels
/// [`CogMetadata::compose_tile`] will later ask for — see this module's header
/// for why that identity matters.
fn projected_pixel_span(
    projection: TransverseMercator,
    transform: CogGeoTransform,
    tile: TileId,
    level: &crate::cog::meta::CogLevel,
) -> Option<(u32, u32, u32, u32)> {
    let (width, height) = (level.width, level.height);
    let mut first_x = u32::MAX;
    let mut last_x = 0_u32;
    let mut first_y = u32::MAX;
    let mut last_y = 0_u32;
    let mut hit = false;
    for_each_projected_pixel(projection, transform, tile, |_, _, pixel_x, pixel_y| {
        let (Some(source_x), Some(source_y)) =
            (to_index(pixel_x, width), to_index(pixel_y, height))
        else {
            return;
        };
        hit = true;
        first_x = first_x.min(source_x);
        last_x = last_x.max(source_x);
        first_y = first_y.min(source_y);
        last_y = last_y.max(source_y);
    });
    if hit {
        Some((first_x, last_x, first_y, last_y))
    } else {
        None
    }
}

/// [`pixel_span`] for an axis-separable image's x axis, also trying the
/// tile's edges shifted by one [`AxisPlacement::x_span`] `period` so a
/// request near the antimeridian still matches an image whose georeferenced
/// x-extent runs past it — a legal, common way to store a Pacific-spanning
/// raster (the pixel grid keeps counting past 180° rather than the data
/// itself wrapping).
///
/// Unions whichever of the three attempts (unshifted, `+period`, `-period`)
/// land inside the image; the shifted ones cost a handful of comparisons
/// each and simply miss for the overwhelming majority of images, which do
/// not span a whole period.
fn wrapped_pixel_span_x(
    transform: CogGeoTransform,
    period: f64,
    west_crs_x: f64,
    east_crs_x: f64,
    extent: u32,
) -> Option<(u32, u32)> {
    let mut span: Option<(u32, u32)> = None;
    for shift in [0.0, period, -period] {
        let (left, _) = transform.to_pixel(west_crs_x + shift, transform.origin_y);
        let (right, _) = transform.to_pixel(east_crs_x + shift, transform.origin_y);
        let Some((first, last)) = pixel_span(left, right, extent) else {
            continue;
        };
        span = Some(
            span.map_or((first, last), |(existing_first, existing_last)| {
                (existing_first.min(first), existing_last.max(last))
            }),
        );
    }
    span
}

/// Spacing, in source tiles, [`CogMetadata::plan_tile_at`] should read at so
/// a map tile that would otherwise need more than [`COG_MAX_SOURCE_TILES`]
/// stays within budget. `1` (read every tile) when `needed` already is.
///
/// `s = ceil(sqrt(needed / cap))` keeps the *count* of tiles a stride-`s`
/// grid touches within `cap`: reading every `s`-th tile in each axis divides
/// the touched count by (at least) `s²`.
fn decimation_stride(needed: u64) -> u32 {
    if needed <= COG_MAX_SOURCE_TILES as u64 {
        return 1;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "a stride only needs to be right to a handful of significant figures"
    )]
    let ratio = needed as f64 / COG_MAX_SOURCE_TILES as f64;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "ratio > 1.0 here, so sqrt().ceil() is >= 2.0, and it saturates harmlessly for \
                   an absurd `needed`"
    )]
    let stride = ratio.sqrt().ceil() as u32;
    stride.max(2)
}

/// Clamps a floating-point pixel span onto `0..extent`, or `None` if it
/// misses.
///
/// [`CogMetadata::compose_tile`] only ever samples pixel *centres*, strictly
/// inside `[low, high)` — the raw tile edges — so a low edge exactly at
/// `extent`, or a high edge exactly at `0` or on a pixel boundary, must not
/// claim the pixel just past it: `high.floor()` alone still over-includes on
/// an exact integer, since `floor` and `ceil` agree there, but `ceil - 1`
/// does not.
fn pixel_span(low: f64, high: f64, extent: u32) -> Option<(u32, u32)> {
    if !low.is_finite() || !high.is_finite() || extent == 0 {
        return None;
    }
    let (low, high) = if low <= high {
        (low, high)
    } else {
        (high, low)
    };
    let last = extent - 1;
    if high <= 0.0 || low >= f64::from(extent) {
        return None;
    }
    let first = low.floor().max(0.0).min(f64::from(last));
    // `.max(first)`: a degenerate zero-width span (`low == high`, an exact
    // integer) collapses to one pixel rather than inverting the range, which
    // would underflow the `last_tile_x - first_tile_x` subtraction two
    // callers up.
    let end = (high.ceil() - 1.0).max(first).min(f64::from(last));
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to first..=extent-1 immediately above"
    )]
    let span = (first as u32, end as u32);
    Some(span)
}

/// Rounds a floating-point pixel position to an index inside `0..extent`.
fn to_index(position: f64, extent: u32) -> Option<u32> {
    if !position.is_finite() || position < 0.0 {
        return None;
    }
    let floored = position.floor();
    if floored >= f64::from(extent) {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "bounds-checked immediately above"
    )]
    let index = floored as u32;
    Some(index)
}

#[cfg(test)]
mod tests {
    use super::lat_of_world_y;
    use super::{COG_MAX_SOURCE_TILES, COG_OUTPUT_TILE_PX, CogSourceTile};
    use crate::cog::meta::{CogCrs, CogGeoTransform, CogLevel, CogMetadata};
    use crate::cog::tmerc::TransverseMercator;
    use crate::error::RenderError;
    use crate::mercator::{EARTH_CIRCUMFERENCE_M, LonLat, TileId, WorldCoord};

    fn level(width: u32, height: u32, tile: u32) -> CogLevel {
        let across = width.div_ceil(tile);
        let down = height.div_ceil(tile);
        let count = (across * down) as usize;
        CogLevel {
            width,
            height,
            tile_width: tile,
            tile_height: tile,
            bits_per_sample: 8,
            samples_per_pixel: 1,
            sample_format: 1,
            compression: 1,
            predictor: 1,
            photometric: 1,
            color_map: Vec::new(),
            tile_offsets: (0..count as u64)
                .map(|index| 4_096 + index * 1_024)
                .collect(),
            tile_byte_counts: vec![1_024; count],
        }
    }

    /// A geographic COG covering 10..20 °E, 40..50 °N at 0.01 °/px.
    fn geographic() -> CogMetadata {
        CogMetadata {
            little_endian: true,
            samples_per_pixel: 1,
            levels: vec![level(1_000, 1_000, 256), level(500, 500, 256)],
            epsg: Some(4326),
            geo: Some(
                CogGeoTransform::new((0.0, 0.0, 10.0, 50.0), 0.01, 0.01)
                    .expect("a well-formed transform"),
            ),
        }
    }

    /// A Web Mercator COG covering a 100 km square near the equator.
    fn mercator() -> CogMetadata {
        CogMetadata {
            little_endian: true,
            samples_per_pixel: 1,
            levels: vec![level(1_000, 1_000, 256)],
            epsg: Some(3857),
            geo: Some(
                CogGeoTransform::new((0.0, 0.0, 0.0, 100_000.0), 100.0, 100.0)
                    .expect("a well-formed transform"),
            ),
        }
    }

    /// A geographic COG whose raw georeferenced x-extent runs past 180°:
    /// origin 170°E, 400×200 px at 0.05°/px reaching CRS x = 190° (170°E
    /// across the antimeridian to 170°W) — a legal, common way to store a
    /// Pacific-spanning raster without wrapping the pixel data itself.
    fn antimeridian_crossing() -> CogMetadata {
        CogMetadata {
            little_endian: true,
            samples_per_pixel: 1,
            levels: vec![level(400, 200, 256)],
            epsg: Some(4326),
            geo: Some(
                CogGeoTransform::new((0.0, 0.0, 170.0, 10.0), 0.05, 0.05)
                    .expect("a well-formed transform"),
            ),
        }
    }

    /// A geographic COG spanning the whole world exactly: origin -180°E/90°N,
    /// 1440×720 px at 0.25°/px (360°/1440 and 180°/720 are both exact in
    /// binary floating point, so a tile edge lands *exactly* on a pixel/tile
    /// boundary) — chosen so both `pixel_span`'s boundary handling and the
    /// antimeridian shift's false-union risk (`high == 0.0` as much as
    /// `high == extent`) are exercised precisely rather than by chance.
    fn whole_world() -> CogMetadata {
        CogMetadata {
            little_endian: true,
            samples_per_pixel: 1,
            levels: vec![level(1_440, 720, 256)],
            epsg: Some(4326),
            geo: Some(
                CogGeoTransform::new((0.0, 0.0, -180.0, 90.0), 0.25, 0.25)
                    .expect("a well-formed transform"),
            ),
        }
    }

    /// A UTM zone 54N COG covering a 25.6 km square west of Tokyo at 100 m/px,
    /// with one half-resolution overview.
    ///
    /// 340 000–365 600 E / 3 924 400–3 950 000 N, i.e. roughly 139.23–139.52 °E
    /// by 35.45–35.68 °N.
    fn utm() -> CogMetadata {
        CogMetadata {
            little_endian: true,
            samples_per_pixel: 1,
            levels: vec![level(256, 256, 256), level(128, 128, 128)],
            epsg: Some(32_654),
            geo: Some(
                CogGeoTransform::new((0.0, 0.0, 340_000.0, 3_950_000.0), 100.0, 100.0)
                    .expect("a well-formed transform"),
            ),
        }
    }

    /// The projection [`utm`]'s COG declares.
    fn zone54() -> TransverseMercator {
        match TransverseMercator::wgs84_utm(54, true) {
            Some(projection) => projection,
            None => panic!("zone 54N must be constructible"),
        }
    }

    fn tile(z: u8, x: u32, y: u32) -> TileId {
        match TileId::new(z, x, y) {
            Ok(tile) => tile,
            Err(error) => panic!("tile {z}/{x}/{y} must be valid: {error}"),
        }
    }

    /// Recomputes, from the public API only, the set of output pixels of `tile`
    /// that the projected mapping puts inside the image — the pixels
    /// `compose_tile` is therefore obliged to paint opaquely once every planned
    /// source tile has been supplied.
    fn expected_covered_pixels(meta: &CogMetadata, target: TileId, level_index: usize) -> usize {
        let projection = zone54();
        let transform = meta
            .level_transform(level_index)
            .expect("the level transform");
        let level = meta.level(level_index).expect("the level");
        let edge = COG_OUTPUT_TILE_PX as usize;
        let north_west = target.north_west();
        let south_east = target.south_east();
        let span_x = south_east.x - north_west.x;
        let span_y = south_east.y - north_west.y;
        let mut covered = 0usize;
        for out_y in 0..edge {
            let world_y = north_west.y + (out_y as f64 + 0.5) / edge as f64 * span_y;
            let latitude = (core::f64::consts::PI * (1.0 - 2.0 * world_y))
                .sinh()
                .atan()
                .to_degrees();
            for out_x in 0..edge {
                let world_x = north_west.x + (out_x as f64 + 0.5) / edge as f64 * span_x;
                let longitude = world_x * 360.0 - 180.0;
                let Some((easting, northing)) = projection.forward(longitude, latitude) else {
                    continue;
                };
                let (pixel_x, pixel_y) = transform.to_pixel(easting, northing);
                if pixel_x >= 0.0
                    && pixel_y >= 0.0
                    && pixel_x.floor() < f64::from(level.width)
                    && pixel_y.floor() < f64::from(level.height)
                {
                    covered += 1;
                }
            }
        }
        covered
    }

    /// The map tile at `zoom` containing the image position `(fraction_x,
    /// fraction_y)` of `meta`'s extent, both in `0..=1`.
    fn utm_tile_at_pixel(meta: &CogMetadata, fraction_x: f64, fraction_y: f64, zoom: u8) -> TileId {
        let transform = meta.geo_transform().expect("a georeference");
        let base = meta.base_level().expect("level 0");
        let (easting, northing) = transform.to_crs(
            f64::from(base.width) * fraction_x,
            f64::from(base.height) * fraction_y,
        );
        let (lon, lat) = zone54()
            .inverse(easting, northing)
            .expect("the extent is inside zone 54");
        LonLat::new(lon, lat).tile(zoom).expect("a tile")
    }

    /// Every source tile a plan lists, filled with opaque pixels.
    fn opaque_sources(meta: &CogMetadata, plan: &super::CogTilePlan) -> Vec<CogSourceTile> {
        let level = meta.level(plan.level).expect("the planned level");
        let bytes = (level.tile_width * level.tile_height) as usize * 4;
        plan.sources
            .iter()
            .map(|source| CogSourceTile {
                tile_x: source.tile_x,
                tile_y: source.tile_y,
                rgba: vec![255u8; bytes],
            })
            .collect()
    }

    #[test]
    fn world_bounds_match_the_declared_extent() {
        let meta = geographic();
        let (min_x, min_y, max_x, max_y) = meta.world_bounds().expect("bounds");
        // 10 °E and 20 °E in normalised world coordinates.
        assert!((min_x - (190.0 / 360.0)).abs() < 1e-12);
        assert!((max_x - (200.0 / 360.0)).abs() < 1e-12);
        // y grows southwards, so the northern edge is the smaller value.
        assert!(min_y < max_y);
        assert!((min_y - LonLat::new(0.0, 50.0).to_world().y).abs() < 1e-12);
    }

    #[test]
    fn level_choice_follows_the_zoom() {
        let meta = geographic();
        // 0.01 deg/px is about 1.1 km/px: far coarser than a z=18 tile needs,
        // so the finest level is used.
        assert_eq!(meta.select_level(18).ok(), Some(0));
        // Zoomed out, the overview is enough.
        let coarse = meta.select_level(4).expect("a level");
        assert_eq!(coarse, 1);
        // The pixel size of level 1 is twice level 0's.
        let fine = meta.world_pixel_size(0).expect("a pixel size");
        let coarse_size = meta.world_pixel_size(1).expect("a pixel size");
        assert!((coarse_size / fine - 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_tile_outside_the_image_has_no_plan() {
        let meta = geographic();
        // The image is in Europe; this tile is in the Pacific.
        let plan = meta
            .plan_tile(tile(4, 1, 6))
            .expect("planning must not fail");
        assert!(plan.is_none());
    }

    #[test]
    fn a_covering_tile_plans_source_tiles_and_ranges() {
        let meta = geographic();
        // z=6 tile covering roughly 11..17 °E, 45..50 °N.
        let target = LonLat::new(15.0, 47.0).tile(6).expect("a tile");
        let plan = meta
            .plan_tile(target)
            .expect("planning must not fail")
            .expect("the tile overlaps the image");
        assert!(!plan.sources.is_empty());
        assert!(plan.sources.len() <= COG_MAX_SOURCE_TILES);
        assert_eq!(plan.ranges().len(), plan.sources.len());
        for source in &plan.sources {
            let level = meta.level(plan.level).expect("the planned level");
            assert!(source.tile_x < level.tiles_across());
            assert!(source.tile_y < level.tiles_down());
        }
    }

    #[test]
    fn a_sparse_tile_is_planned_without_a_range() {
        let mut meta = geographic();
        meta.levels[1].tile_byte_counts[0] = 0;
        let target = LonLat::new(10.5, 49.5).tile(4).expect("a tile");
        let plan = meta
            .plan_tile(target)
            .expect("planning must not fail")
            .expect("the tile overlaps the image");
        assert_eq!(plan.level, 1);
        assert!(plan.sources.iter().any(|source| source.range.is_none()));
        assert!(plan.ranges().len() < plan.sources.len());
    }

    #[test]
    fn too_many_source_tiles_are_decimated_rather_than_refused() {
        // A level-0-only file zoomed right out: one map tile would naively
        // touch every one of the ~24 649 tiles in a 157×157 grid… so shrink
        // the tile size until an un-decimated plan would blow the cap.
        let mut meta = mercator();
        meta.levels = vec![level(10_000, 10_000, 64)];
        let target = LonLat::new(0.5, 0.5).tile(2).expect("a tile");
        let plan = meta
            .plan_tile(target)
            .expect("a file with no coarser overview must decimate, not error")
            .expect("the tile overlaps the image");
        assert!(
            !plan.sources.is_empty(),
            "some coarse preview must be planned"
        );
        assert!(
            plan.sources.len() <= COG_MAX_SOURCE_TILES,
            "{} planned tiles must stay within the {COG_MAX_SOURCE_TILES} cap",
            plan.sources.len()
        );
        assert!(
            plan.stride > 1,
            "a decimated plan must record the stride it read at"
        );
        meta.compose_tile(&plan, &opaque_sources(&meta, &plan))
            .expect("composing a decimated plan must succeed");
    }

    #[test]
    fn an_ordinary_plan_is_never_decimated() {
        let meta = geographic();
        let target = LonLat::new(15.0, 47.0).tile(6).expect("a tile");
        let plan = meta
            .plan_tile(target)
            .expect("planning must not fail")
            .expect("the tile overlaps the image");
        assert_eq!(plan.stride, 1, "well within budget, so no decimation");
    }

    #[test]
    fn composition_places_source_pixels_and_leaves_gaps_transparent() {
        let mut meta = geographic();
        // One tile, one level, so the mapping is easy to reason about.
        meta.levels = vec![level(256, 256, 256)];
        let target = LonLat::new(10.5, 49.5).tile(10).expect("a tile");
        let plan = meta
            .plan_tile(target)
            .expect("planning must not fail")
            .expect("the tile overlaps the image");

        // A source tile whose red channel encodes the column.
        let level_meta = meta.level(0).expect("level 0");
        let mut rgba =
            Vec::with_capacity((level_meta.tile_width * level_meta.tile_height) as usize * 4);
        for y in 0..level_meta.tile_height {
            for x in 0..level_meta.tile_width {
                #[expect(clippy::cast_possible_truncation, reason = "a 256-wide fixture tile")]
                let (red, green) = (x as u8, y as u8);
                rgba.extend_from_slice(&[red, green, 0, 255]);
            }
        }
        let sources = vec![CogSourceTile {
            tile_x: plan.sources[0].tile_x,
            tile_y: plan.sources[0].tile_y,
            rgba,
        }];
        let composed = meta
            .compose_tile(&plan, &sources)
            .expect("composition must succeed");
        assert_eq!(composed.width(), COG_OUTPUT_TILE_PX);
        assert_eq!(composed.height(), COG_OUTPUT_TILE_PX);
        assert!(
            composed.rgba().chunks_exact(4).any(|pixel| pixel[3] == 255),
            "the tile overlaps the image, so some pixels must be opaque"
        );

        // With no source tiles supplied, every output pixel stays transparent.
        let empty = meta
            .compose_tile(&plan, &[])
            .expect("composition must succeed with no sources");
        assert!(empty.rgba().chunks_exact(4).all(|pixel| pixel[3] == 0));
    }

    #[test]
    fn composition_rejects_a_wrongly_sized_source_tile() {
        let meta = geographic();
        let target = LonLat::new(10.5, 49.5).tile(6).expect("a tile");
        let plan = meta
            .plan_tile(target)
            .expect("planning must not fail")
            .expect("the tile overlaps the image");
        let sources = vec![CogSourceTile {
            tile_x: plan.sources[0].tile_x,
            tile_y: plan.sources[0].tile_y,
            rgba: vec![0; 12],
        }];
        assert!(matches!(
            meta.compose_tile(&plan, &sources),
            Err(RenderError::Decode(_))
        ));
    }

    #[test]
    fn a_web_mercator_cog_plans_the_same_way() {
        let meta = mercator();
        assert_eq!(meta.crs().ok(), Some(CogCrs::WebMercator));
        let target = LonLat::new(0.45, 0.45).tile(10).expect("a tile");
        let plan = meta
            .plan_tile(target)
            .expect("planning must not fail")
            .expect("the tile overlaps the image");
        assert!(!plan.sources.is_empty());
        let composed = meta
            .compose_tile(&plan, &[])
            .expect("composition must succeed");
        assert_eq!(composed.width(), COG_OUTPUT_TILE_PX);
    }

    /// The positive antimeridian case: a raster whose georeferenced x-extent
    /// runs past 180° holds real data there, and a request for that side
    /// must find it rather than silently drawing nothing (finding 26).
    #[test]
    fn an_antimeridian_crossing_cog_finds_sources_on_both_sides() {
        let meta = antimeridian_crossing();

        // Near the image's western edge (170°E), well inside -180..180: no shift needed.
        let east_side = LonLat::new(175.0, 5.0).tile(6).expect("a tile");
        let east_plan = meta
            .plan_tile(east_side)
            .expect("planning must not fail")
            .expect("170..175°E must overlap the image directly");
        assert!(!east_plan.sources.is_empty());

        // Past the antimeridian: canonical -175° is raw CRS x 185° (170+15) — a +360° shift.
        let west_side = LonLat::new(-175.0, 5.0).tile(6).expect("a tile");
        let west_plan = meta
            .plan_tile(west_side)
            .expect("planning must not fail")
            .expect("the wrapped portion (raw CRS x 180..190°) must still be found");
        assert!(!west_plan.sources.is_empty());
        assert_ne!(
            east_plan.sources, west_plan.sources,
            "the two sides of the antimeridian must not collapse onto the same source tiles"
        );

        // And a tile nowhere near either side of the image is still culled.
        let far = LonLat::new(0.0, 5.0).tile(6).expect("a tile");
        assert!(
            meta.plan_tile(far)
                .expect("planning must not fail")
                .is_none(),
            "the antimeridian shift must not make an unrelated tile match"
        );
    }

    /// A raster spanning the whole world exactly, so its CRS x-extent touches
    /// both `0` and the image's own width/extent boundary — the specific
    /// values where an inclusive `pixel_span` bound or the antimeridian
    /// shift's union could pull in the entire image width for a single edge
    /// tile (see `pixel_span` and `wrapped_pixel_span_x`).
    #[test]
    fn a_whole_world_cog_plans_narrow_spans_at_both_edges() {
        let meta = whole_world();
        let level = meta.level(0).expect("level 0");
        let tiles_across = level.tiles_across();

        for target in [tile(6, 0, 32), tile(6, 63, 32)] {
            let plan = meta
                .plan_tile(target)
                .expect("planning must not fail")
                .expect("an edge tile of a whole-world raster must overlap it");
            let touched_columns: std::collections::BTreeSet<u32> =
                plan.sources.iter().map(|source| source.tile_x).collect();
            assert!(
                touched_columns.len() < tiles_across as usize,
                "tile {target:?} touched {} of {tiles_across} tile columns — an unguarded \
                 antimeridian shift, or an inclusive pixel_span boundary, unioning in the whole \
                 image width would touch all of them",
                touched_columns.len()
            );
        }

        // And the whole-world footprint is exactly the drawable unit square
        // (finding 25): a global raster's poles must not blow world_bounds
        // out to roughly eleven times its size.
        let (min_x, min_y, max_x, max_y) = meta.world_bounds().expect("bounds");
        for value in [min_x, min_y, max_x, max_y] {
            assert!(
                (0.0..=1.0).contains(&value),
                "{value} must lie inside the drawable unit square"
            );
        }
    }

    #[test]
    fn pixel_span_excludes_a_high_edge_on_a_tile_boundary() {
        use super::pixel_span;
        // The east edge lands exactly on a 256-px tile boundary: only pixel
        // 255 is ever sampled — `compose_tile`'s last centre is strictly
        // short of 256 — so the span must not reach into tile column 1.
        assert_eq!(pixel_span(0.0, 256.0, 512), Some((0, 255)));
        // A non-integer high must not round up past its actual pixel either.
        assert_eq!(pixel_span(10.3, 250.9, 512), Some((10, 250)));
        // A west edge exactly at the image's extent has no valid pixel right of it.
        assert_eq!(pixel_span(512.0, 600.0, 512), None);
        // A high edge exactly at 0 has no valid pixel either (mirrors the case above).
        assert_eq!(pixel_span(-20.0, 0.0, 512), None);
        // A degenerate zero-width span collapses to one pixel, not an inverted range.
        assert_eq!(pixel_span(100.0, 100.0, 512), Some((100, 100)));
        // Ordinary interior and non-finite inputs are unaffected.
        assert_eq!(pixel_span(1.5, 4.5, 512), Some((1, 4)));
        assert_eq!(pixel_span(f64::NAN, 1.0, 512), None);
        assert_eq!(pixel_span(0.0, 1.0, 0), None);
    }

    #[test]
    fn planning_needs_a_supported_crs_and_georeference() {
        let mut meta = geographic();
        // A national grid: a supported *kind* of projection is not the point,
        // the reader has no parameters for EPSG:27700.
        meta.epsg = Some(27_700);
        assert!(matches!(
            meta.plan_tile(tile(0, 0, 0)),
            Err(RenderError::Unsupported(_))
        ));
        meta.epsg = Some(4326);
        meta.geo = None;
        assert!(matches!(
            meta.world_bounds(),
            Err(RenderError::Unsupported(_))
        ));
        assert!(matches!(
            meta.plan_tile_at(tile(0, 0, 0), 0),
            Err(RenderError::Unsupported(_))
        ));
    }

    #[test]
    fn planning_rejects_a_level_that_does_not_exist() {
        let meta = geographic();
        assert!(meta.plan_tile_at(tile(4, 8, 5), 7).is_err());
    }

    #[test]
    fn a_utm_cog_is_classified_as_its_zone() {
        let meta = utm();
        assert_eq!(
            meta.crs().ok(),
            Some(CogCrs::Utm {
                zone: 54,
                north: true
            })
        );
        assert!(!meta.crs().expect("a CRS").is_axis_separable());
    }

    #[test]
    fn a_utm_footprint_bounds_every_edge_of_the_extent() {
        let meta = utm();
        let (min_x, min_y, max_x, max_y) = meta.world_bounds().expect("bounds");
        assert!(min_x < max_x && min_y < max_y);

        let projection = zone54();
        let transform = meta.geo_transform().expect("a transform");
        let base = meta.base_level().expect("level 0");
        // The whole boundary — not just the corners — must be inside the
        // reported footprint: a UTM rectangle's edges bow in Web Mercator.
        let steps = 64u32;
        let mut worst_corner_only_shortfall = 0.0_f64;
        let mut corner_min_x = f64::INFINITY;
        let mut corner_max_x = f64::NEG_INFINITY;
        for step in 0..=steps {
            let fraction = f64::from(step) / f64::from(steps);
            for (pixel_x, pixel_y) in [
                (f64::from(base.width) * fraction, 0.0),
                (f64::from(base.width) * fraction, f64::from(base.height)),
                (0.0, f64::from(base.height) * fraction),
                (f64::from(base.width), f64::from(base.height) * fraction),
            ] {
                let (easting, northing) = transform.to_crs(pixel_x, pixel_y);
                let (lon, lat) = projection
                    .inverse(easting, northing)
                    .expect("the extent is inside zone 54");
                let world = LonLat::new(lon, lat).to_world();
                assert!(
                    world.x >= min_x && world.x <= max_x,
                    "easting {} outside {min_x}..{max_x}",
                    world.x
                );
                assert!(
                    world.y >= min_y && world.y <= max_y,
                    "northing {} outside {min_y}..{max_y}",
                    world.y
                );
                if step == 0 || step == steps {
                    corner_min_x = corner_min_x.min(world.x);
                    corner_max_x = corner_max_x.max(world.x);
                } else {
                    worst_corner_only_shortfall = worst_corner_only_shortfall
                        .max((corner_min_x - world.x).max(world.x - corner_max_x));
                }
            }
        }
        // And the reason the corners alone are not enough: some boundary point
        // *is* outside the four-corner bounding box.
        assert!(
            worst_corner_only_shortfall > 0.0,
            "a bowed edge must overrun the corner bbox, else this test proves nothing"
        );
    }

    #[test]
    fn utm_pixel_size_is_scaled_by_the_latitude() {
        let meta = utm();
        // 100 m pixels at ~35.6 °N: a ground metre covers 1 / cos φ more of the
        // normalised Web Mercator world than it does at the equator.
        let size = meta.world_pixel_size(0).expect("a pixel size");
        let equator = 100.0 / EARTH_CIRCUMFERENCE_M;
        assert!(
            size > equator,
            "{size} must exceed the equatorial {equator}"
        );
        let latitude = 35.68_f64.to_radians();
        let expected = 100.0 / (EARTH_CIRCUMFERENCE_M * latitude.cos());
        assert!(
            (size / expected - 1.0).abs() < 5e-3,
            "{size} must be within 0.5% of {expected}"
        );
        // The overview's pixels are twice as wide, latitude notwithstanding.
        let coarse = meta.world_pixel_size(1).expect("a pixel size");
        assert!((coarse / size - 2.0).abs() < 1e-9);
        // …which is what makes level choice track the zoom for a UTM file too.
        assert_eq!(meta.select_level(20).ok(), Some(0));
        assert!(meta.select_level(4).expect("a level") >= 1);
    }

    #[test]
    fn a_utm_tile_far_from_the_image_is_culled_without_projecting() {
        let meta = utm();
        // Zone 54's central meridian is 141 °E; this tile is over the Atlantic,
        // well outside the range `tmerc` will project at all.
        assert!(
            meta.plan_tile(tile(6, 30, 24))
                .expect("planning must not fail")
                .is_none()
        );
        // Nearer, but still off the image: same zone, wrong latitude.
        let target = LonLat::new(139.6, 20.0).tile(12).expect("a tile");
        assert!(
            meta.plan_tile(target)
                .expect("planning must not fail")
                .is_none()
        );
    }

    #[test]
    fn a_utm_cog_plans_and_composes_without_holes() {
        let meta = utm();
        // An interior tile, a tile straddling the north-west corner and one
        // straddling the south-east corner: the three cases where planning and
        // composition could disagree.
        // Probes derived from the image itself rather than hard-coded, so they
        // stay meaningful if the fixture's georeference moves: the centre, the
        // two corners, and the centre again at a zoom where one map tile is
        // wider than the whole image.
        let targets = [
            utm_tile_at_pixel(&meta, 0.5, 0.5, 13),
            utm_tile_at_pixel(&meta, 0.0, 0.0, 13),
            utm_tile_at_pixel(&meta, 1.0, 1.0, 13),
            utm_tile_at_pixel(&meta, 0.5, 0.5, 11),
        ];
        let mut any_partial = false;
        for target in targets {
            let plan = meta
                .plan_tile(target)
                .expect("planning must not fail")
                .unwrap_or_else(|| panic!("tile {target:?} must overlap the image"));
            assert!(!plan.sources.is_empty());
            assert!(plan.sources.len() <= COG_MAX_SOURCE_TILES);
            let level = meta.level(plan.level).expect("the planned level");
            for source in &plan.sources {
                assert!(source.tile_x < level.tiles_across());
                assert!(source.tile_y < level.tiles_down());
            }

            let composed = meta
                .compose_tile(&plan, &opaque_sources(&meta, &plan))
                .expect("composition must succeed");
            let opaque = composed
                .rgba()
                .chunks_exact(4)
                .filter(|pixel| pixel[3] == 255)
                .count();
            let expected = expected_covered_pixels(&meta, target, plan.level);
            assert_eq!(
                opaque, expected,
                "tile {target:?}: every pixel the mapping puts inside the image must be painted, \
                 so planning and composition agree exactly"
            );
            assert!(expected > 0, "tile {target:?} must cover some image");
            let total = (COG_OUTPUT_TILE_PX * COG_OUTPUT_TILE_PX) as usize;
            if expected < total {
                any_partial = true;
            }
        }
        assert!(
            any_partial,
            "at least one probe must only partly overlap the image, else the edge case is untested"
        );
    }

    #[test]
    fn utm_composition_leaves_missing_sources_transparent() {
        let meta = utm();
        let target = utm_tile_at_pixel(&meta, 0.5, 0.5, 13);
        let plan = meta
            .plan_tile(target)
            .expect("planning must not fail")
            .expect("the tile overlaps the image");
        let empty = meta
            .compose_tile(&plan, &[])
            .expect("composition must succeed with no sources");
        assert!(empty.rgba().chunks_exact(4).all(|pixel| pixel[3] == 0));
    }

    /// The guards in [`crate::cog::tmerc`] are absolute-latitude and
    /// longitude-offset bounds, and a UTM product near the top of the grid's
    /// range is where they could plausibly clip something legitimate: Sentinel-2
    /// reaches 84 °N, and at that latitude a degree of longitude is only 12 km,
    /// so a granule's easting spread reads as a much larger longitude spread
    /// than it would at the equator.
    #[test]
    fn a_high_latitude_utm_cog_is_still_placed() {
        // Zone 60N (central meridian 177 °E), a 25.6 km square at ~80 °N.
        let meta = CogMetadata {
            little_endian: true,
            samples_per_pixel: 1,
            levels: vec![level(256, 256, 256)],
            epsg: Some(32_660),
            geo: Some(
                CogGeoTransform::new((0.0, 0.0, 450_000.0, 8_880_000.0), 100.0, 100.0)
                    .expect("a well-formed transform"),
            ),
        };
        let (min_x, min_y, max_x, max_y) = meta
            .world_bounds()
            .expect("an 80 °N extent must still have a footprint");
        assert!(max_x > min_x && max_y > min_y);
        let latitude = lat_of_world_y((min_y + max_y) / 2.0);
        assert!(
            (latitude - 80.0).abs() < 0.5,
            "the footprint must sit at ~80 °N, not {latitude}"
        );
        // Ground pixels are worth ~1 / cos 80° = 5.8× more world units here.
        let size = meta.world_pixel_size(0).expect("a pixel size");
        assert!(size > 5.0 * 100.0 / EARTH_CIRCUMFERENCE_M, "{size}");

        let target = WorldCoord::new((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
            .tile(13)
            .expect("a tile");
        let plan = meta
            .plan_tile_at(target, 0)
            .expect("planning must not fail")
            .expect("the centre tile must overlap the image");
        let composed = meta
            .compose_tile(&plan, &opaque_sources(&meta, &plan))
            .expect("composition must succeed");
        assert!(
            composed.rgba().chunks_exact(4).any(|pixel| pixel[3] == 255),
            "a tile at the centre of an 80 °N image must be painted"
        );
    }

    /// The southern zones (EPSG:327`zz`) are a different code path, not just
    /// different numbers: negative latitudes put `world_y` past 0.5, the
    /// `max_abs_lat_deg` in [`CogMetadata::utm_extent`] is only right because it
    /// takes an absolute value, and the northings run against a 10 000 km false
    /// northing where a sign or padding slip has room to hide.
    #[test]
    fn a_southern_utm_cog_is_placed_the_same_way() {
        // Zone 34S (central meridian 21 °E), a 25.6 km square over Cape Town:
        // 18.42 °E / 33.92 °S is 261 882 E / 6 243 182 N (see `tmerc`'s
        // reference points).
        let meta = CogMetadata {
            little_endian: true,
            samples_per_pixel: 1,
            levels: vec![level(256, 256, 256)],
            epsg: Some(32_734),
            geo: Some(
                CogGeoTransform::new((0.0, 0.0, 261_882.0, 6_243_182.0), 100.0, 100.0)
                    .expect("a well-formed transform"),
            ),
        };
        assert_eq!(
            meta.crs().ok(),
            Some(CogCrs::Utm {
                zone: 34,
                north: false
            })
        );
        let (min_x, min_y, max_x, max_y) = meta.world_bounds().expect("bounds");
        assert!(max_x > min_x && max_y > min_y);
        // Southern latitudes sit in the lower half of the world square.
        assert!(min_y > 0.5, "a southern extent must have world_y > 0.5");
        let latitude = lat_of_world_y((min_y + max_y) / 2.0);
        assert!(
            (latitude + 34.04).abs() < 0.3,
            "the footprint must sit at ~34 °S, not {latitude}"
        );
        // …and the `1 / cos φ` inflation applies to |φ|, not φ.
        let size = meta.world_pixel_size(0).expect("a pixel size");
        let equator = 100.0 / EARTH_CIRCUMFERENCE_M;
        assert!(
            size > equator,
            "{size} must exceed the equatorial {equator}"
        );
        assert!(
            (size / (equator / latitude.to_radians().cos()) - 1.0).abs() < 5e-3,
            "{size} must match the 1/cos|φ| prediction"
        );

        let target = WorldCoord::new((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
            .tile(13)
            .expect("a tile");
        let plan = meta
            .plan_tile_at(target, 0)
            .expect("planning must not fail")
            .expect("the centre tile must overlap the image");
        let composed = meta
            .compose_tile(&plan, &opaque_sources(&meta, &plan))
            .expect("composition must succeed");
        assert!(
            composed.rgba().chunks_exact(4).any(|pixel| pixel[3] == 255),
            "a tile at the centre of a southern-zone image must be painted"
        );
    }

    #[test]
    fn a_utm_georeference_outside_its_zone_is_reported() {
        let mut meta = utm();
        // Zone 54 spans roughly 138–144 °E; an easting 6000 km off the central
        // meridian is nowhere near it, and no sample projects.
        meta.geo = Some(
            CogGeoTransform::new((0.0, 0.0, 6_500_000.0, 3_950_000.0), 100.0, 100.0)
                .expect("a well-formed transform"),
        );
        assert!(matches!(
            meta.world_bounds(),
            Err(RenderError::Unsupported(_))
        ));
        assert!(matches!(
            meta.world_pixel_size(0),
            Err(RenderError::Unsupported(_))
        ));
    }

    /// Reports what the reprojecting path costs per map tile.
    ///
    /// Ignored because it is a measurement, not an assertion, and because a
    /// debug-build number would be 10–30× the figure that matters. Run with
    ///
    /// ```text
    /// cargo nextest run --release -p oxigis-render --run-ignored all \
    ///     -E 'test(the_projected_path_cost_per_tile)' --no-capture
    /// ```
    ///
    /// Measured on the development machine (release, 2026-07-30): 5.6 ms to
    /// plan and 5.6 ms to compose one 256×256 tile, against 0.2 µs / 198 µs for
    /// the axis-separable path. Two passes of 65 536 Transverse Mercator
    /// forwards dominate, and they are what
    /// [`TransverseMercator::forward_grid`]'s hoisting and series recurrences
    /// already cut by 3.1× (from 17.7 ms + 17.5 ms).
    ///
    /// The next thing to reach for, if 11 ms per tile on a background state
    /// machine with a tile cache ever mattered, would *not* be a fitted affine
    /// or bilinear approximation of the mapping: a map tile can span the whole
    /// world, where the inverse-Mercator step's curvature makes any fixed-order
    /// approximation's error unbounded. Deriving the plan's bounding box from
    /// the tile boundary alone (the mapping has no interior critical point, so
    /// its extremes are on the edges) is the sound version — at the price of
    /// planning and composition no longer sharing one code path, which is the
    /// invariant this module's header rests on.
    #[test]
    #[ignore = "measurement, not an assertion; needs --release to mean anything"]
    fn the_projected_path_cost_per_tile() {
        use std::time::Instant;

        let rounds = 200u32;
        for (name, meta, target) in [
            (
                "EPSG:4326 (axis-separable)",
                geographic(),
                LonLat::new(15.0, 47.0).tile(6).expect("a tile"),
            ),
            (
                "EPSG:32654 (reprojected)",
                utm(),
                utm_tile_at_pixel(&utm(), 0.5, 0.5, 13),
            ),
        ] {
            let started = Instant::now();
            let mut plan = None;
            for _ in 0..rounds {
                plan = meta.plan_tile(target).expect("planning must not fail");
            }
            let planning = started.elapsed() / rounds;
            let plan = plan.expect("the tile overlaps the image");
            let sources = opaque_sources(&meta, &plan);
            let started = Instant::now();
            for _ in 0..rounds {
                meta.compose_tile(&plan, &sources)
                    .expect("composition must succeed");
            }
            let composing = started.elapsed() / rounds;
            println!("{name}: plan {planning:?}/tile, compose {composing:?}/tile");
        }
    }

    #[test]
    fn the_utm_byte_fixture_parses_as_its_zone() {
        let source = crate::cog::reader::CogSource::new(crate::cog::reader::MemoryRangeFetch::new(
            crate::cog::sample_utm_cog_bytes(),
        ));
        let meta = futures::executor::block_on(source.open()).expect("the UTM fixture must open");
        assert_eq!(meta.epsg, Some(32_654));
        assert_eq!(
            meta.crs().ok(),
            Some(CogCrs::Utm {
                zone: 54,
                north: true
            })
        );
        let transform = meta.geo_transform().expect("a georeference");
        assert!((transform.origin_x - 380_000.0).abs() < 1e-9);
        assert!((transform.origin_y - 3_950_000.0).abs() < 1e-9);
        // 8 px of 10 m at ~35.7 °N: the footprint is a sliver, but a real one.
        let (min_x, min_y, max_x, max_y) = meta.world_bounds().expect("bounds");
        assert!(max_x > min_x && max_y > min_y);
        let north_west = LonLat::new(
            min_x * 360.0 - 180.0,
            (core::f64::consts::PI * (1.0 - 2.0 * min_y))
                .sinh()
                .atan()
                .to_degrees(),
        );
        assert!(
            (north_west.lon - 139.71).abs() < 0.05,
            "longitude {}",
            north_west.lon
        );
        assert!(
            (north_west.lat - 35.68).abs() < 0.05,
            "latitude {}",
            north_west.lat
        );
    }
}
