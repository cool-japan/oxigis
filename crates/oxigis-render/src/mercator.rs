//! Spherical Web Mercator (EPSG:3857) math and slippy-map tile addressing.
//!
//! Three coordinate spaces are modelled explicitly so that conversions cannot
//! be mixed up by accident:
//!
//! | Type | Space | Range |
//! |---|---|---|
//! | [`LonLat`] | WGS84 geographic degrees | lon `-180..=180`, lat `-85.0511..=85.0511` |
//! | [`WorldCoord`] | normalised Web Mercator | `0..=1` on both axes, `y` increasing southwards |
//! | [`MercatorPoint`] | EPSG:3857 metres | `±20 037 508.34` on both axes, `y` increasing northwards |
//!
//! [`TileId`] addresses a tile of the standard XYZ pyramid: at zoom `z` the map
//! is a `2^z * 2^z` grid, tile `(0, 0)` is the north-west corner, and one tile
//! is [`TILE_SIZE_PX`] pixels wide at its native zoom.
//!
//! The sphere (not the WGS84 ellipsoid) is used throughout, matching the
//! `EPSG:3857` "pseudo-mercator" definition used by every XYZ tile service.

use crate::error::RenderError;

/// Edge length in pixels of one tile at its native zoom level.
pub const TILE_SIZE_PX: f64 = 256.0;

/// Highest integer zoom level this renderer addresses.
///
/// `2^24` tiles per axis keeps `2^z` well inside `u32` and resolves roughly
/// 9 mm/pixel at the equator — past every basemap in circulation, and past the
/// z22 that vector-tile toolchains treat as their deepest addressable level.
///
/// Archives may legitimately declare more still (PMTiles allows up to 30).
/// Those levels are not unreachable: [`crate::renderer::MapRenderer`] clamps
/// its tile requests to a source's declared range and magnifies the deepest
/// available level, so a z30 archive shows detail-limited imagery rather than
/// nothing.
///
/// Raising this constant is *not* a local change. `oxigis-ui`'s MBTiles index
/// packs `z`, `x` and `y` into one `u64`, and a column wider than its
/// coordinate field would alias into the zoom bits — a lookup answering with a
/// different zoom's tile. That file now asserts `COORD_BITS >= MAX_ZOOM` at
/// compile time, so the coupling fails the build instead of the map; the two
/// asserts below cover the arithmetic here.
pub const MAX_ZOOM: u8 = 24;

/// `TileId::tiles_per_axis` shifts `1u32` left by the zoom, and
/// `WorldCoord::tile` needs `f64::from(2^z)` to stay exact.
const _: () = assert!(
    MAX_ZOOM < 31,
    "MAX_ZOOM must keep `1u32 << MAX_ZOOM` in range"
);

/// `TileId::sub_rect_in` multiplies a tile index by `2^-levels` in **f32**, and
/// `f32` represents every integer only up to `2^24`. One level deeper and the
/// index rounds, which lands a magnified tile's UV rectangle off its parent —
/// silently, since the arithmetic never overflows.
const _: () = assert!(
    MAX_ZOOM <= 24,
    "MAX_ZOOM must keep a tile index exact in f32 for `sub_rect_in`"
);

/// Radius of the sphere used by EPSG:3857, in metres.
pub const EARTH_RADIUS_M: f64 = 6_378_137.0;

/// Circumference of the EPSG:3857 sphere at the equator, in metres.
pub const EARTH_CIRCUMFERENCE_M: f64 = 2.0 * core::f64::consts::PI * EARTH_RADIUS_M;

/// Northern/southern cut-off latitude of the Web Mercator projection.
///
/// Equal to `atan(sinh(pi))` in degrees; beyond it the projection diverges and
/// the square world map would no longer be square.
pub const MAX_LATITUDE_DEG: f64 = 85.051_128_779_806_59;

/// A WGS84 geographic position in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LonLat {
    /// Longitude in degrees, positive eastwards.
    pub lon: f64,
    /// Latitude in degrees, positive northwards.
    pub lat: f64,
}

impl LonLat {
    /// Creates a position without normalising it.
    #[must_use]
    pub const fn new(lon: f64, lat: f64) -> Self {
        Self { lon, lat }
    }

    /// Returns the position with longitude wrapped into `-180..=180` and
    /// latitude clamped to `±`[`MAX_LATITUDE_DEG`].
    ///
    /// Non-finite components are replaced by `0.0`, so the result is always a
    /// projectable position.
    #[must_use]
    pub fn normalized(self) -> Self {
        let lon = if self.lon.is_finite() {
            let wrapped = (self.lon + 180.0).rem_euclid(360.0) - 180.0;
            // `rem_euclid` maps exactly +180 to -180; keep the eastern edge.
            if wrapped == -180.0 && self.lon > 0.0 {
                180.0
            } else {
                wrapped
            }
        } else {
            0.0
        };
        let lat = if self.lat.is_finite() {
            self.lat.clamp(-MAX_LATITUDE_DEG, MAX_LATITUDE_DEG)
        } else {
            0.0
        };
        Self { lon, lat }
    }

    /// Projects to normalised Web Mercator space (`0..=1`).
    ///
    /// The position is normalised first, so latitudes beyond the projection
    /// cut-off saturate at `y = 0` / `y = 1` instead of diverging.
    #[must_use]
    pub fn to_world(self) -> WorldCoord {
        let norm = self.normalized();
        let lat_rad = norm.lat.to_radians();
        // ln(tan(lat) + sec(lat)) is the Mercator ordinate of the unit sphere.
        let merc_y = (lat_rad.tan() + 1.0 / lat_rad.cos()).ln();
        WorldCoord {
            x: (norm.lon + 180.0) / 360.0,
            y: 0.5 * (1.0 - merc_y / core::f64::consts::PI),
        }
    }

    /// Projects to EPSG:3857 metres.
    #[must_use]
    pub fn to_mercator_meters(self) -> MercatorPoint {
        let norm = self.normalized();
        let lat_rad = norm.lat.to_radians();
        MercatorPoint {
            x: EARTH_RADIUS_M * norm.lon.to_radians(),
            y: EARTH_RADIUS_M * (core::f64::consts::FRAC_PI_4 + lat_rad / 2.0).tan().ln(),
        }
    }

    /// Returns the tile containing this position at `zoom`.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::ZoomOutOfRange`] if `zoom` exceeds [`MAX_ZOOM`].
    pub fn tile(self, zoom: u8) -> Result<TileId, RenderError> {
        self.to_world().tile(zoom)
    }
}

/// A position in normalised Web Mercator space.
///
/// `(0, 0)` is the north-west corner of the world map and `(1, 1)` the
/// south-east corner, i.e. `y` grows southwards like screen coordinates do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldCoord {
    /// Normalised easting, `0` at 180°W and `1` at 180°E.
    pub x: f64,
    /// Normalised northing, `0` at the northern cut-off latitude.
    pub y: f64,
}

impl WorldCoord {
    /// Creates a normalised Web Mercator position without clamping it.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Returns the position clamped into the `0..=1` unit square.
    #[must_use]
    pub fn clamped(self) -> Self {
        Self {
            x: if self.x.is_finite() {
                self.x.clamp(0.0, 1.0)
            } else {
                0.0
            },
            y: if self.y.is_finite() {
                self.y.clamp(0.0, 1.0)
            } else {
                0.0
            },
        }
    }

    /// Unprojects back to WGS84 degrees.
    #[must_use]
    pub fn to_lon_lat(self) -> LonLat {
        let clamped = self.clamped();
        let lat_rad = (core::f64::consts::PI * (1.0 - 2.0 * clamped.y))
            .sinh()
            .atan();
        LonLat {
            lon: clamped.x * 360.0 - 180.0,
            lat: lat_rad.to_degrees(),
        }
    }

    /// Converts to EPSG:3857 metres.
    #[must_use]
    pub fn to_mercator_meters(self) -> MercatorPoint {
        MercatorPoint {
            x: (self.x - 0.5) * EARTH_CIRCUMFERENCE_M,
            y: (0.5 - self.y) * EARTH_CIRCUMFERENCE_M,
        }
    }

    /// Converts from EPSG:3857 metres.
    #[must_use]
    pub fn from_mercator_meters(point: MercatorPoint) -> Self {
        Self {
            x: point.x / EARTH_CIRCUMFERENCE_M + 0.5,
            y: 0.5 - point.y / EARTH_CIRCUMFERENCE_M,
        }
    }

    /// Returns the tile containing this position at `zoom`.
    ///
    /// The position is clamped into the unit square first, so points outside
    /// the world map snap to the edge tiles rather than failing.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::ZoomOutOfRange`] if `zoom` exceeds [`MAX_ZOOM`].
    pub fn tile(self, zoom: u8) -> Result<TileId, RenderError> {
        if zoom > MAX_ZOOM {
            return Err(RenderError::ZoomOutOfRange {
                zoom: f64::from(zoom),
                max: MAX_ZOOM,
            });
        }
        let clamped = self.clamped();
        let n = f64::from(TileId::tiles_per_axis(zoom));
        let last = TileId::tiles_per_axis(zoom) - 1;
        let x = (clamped.x * n).floor().clamp(0.0, f64::from(last)) as u32;
        let y = (clamped.y * n).floor().clamp(0.0, f64::from(last)) as u32;
        TileId::new(zoom, x, y)
    }
}

/// A position in EPSG:3857 metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MercatorPoint {
    /// Easting in metres, `0` at the prime meridian.
    pub x: f64,
    /// Northing in metres, `0` at the equator and positive northwards.
    pub y: f64,
}

impl MercatorPoint {
    /// Creates a projected position.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Converts to normalised Web Mercator space.
    #[must_use]
    pub fn to_world(self) -> WorldCoord {
        WorldCoord::from_mercator_meters(self)
    }

    /// Unprojects back to WGS84 degrees.
    #[must_use]
    pub fn to_lon_lat(self) -> LonLat {
        let lat_rad = 2.0 * (self.y / EARTH_RADIUS_M).exp().atan() - core::f64::consts::FRAC_PI_2;
        LonLat {
            lon: (self.x / EARTH_RADIUS_M).to_degrees(),
            lat: lat_rad.to_degrees(),
        }
    }
}

/// An axis-aligned rectangle in EPSG:3857 metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MercatorBounds {
    /// Western edge in metres.
    pub min_x: f64,
    /// Southern edge in metres.
    pub min_y: f64,
    /// Eastern edge in metres.
    pub max_x: f64,
    /// Northern edge in metres.
    pub max_y: f64,
}

impl MercatorBounds {
    /// Creates bounds from its four edges without reordering them.
    #[must_use]
    pub const fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// East-west extent in metres.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// North-south extent in metres.
    #[must_use]
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// Centre of the rectangle.
    #[must_use]
    pub fn center(&self) -> MercatorPoint {
        MercatorPoint {
            x: 0.5 * (self.min_x + self.max_x),
            y: 0.5 * (self.min_y + self.max_y),
        }
    }
}

/// Address of one tile in the XYZ (slippy-map) pyramid.
///
/// The ordering derived here is `(z, x, y)` lexicographic; it is only used to
/// make tile lists deterministic, not to express spatial proximity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileId {
    /// Zoom level, `0..=`[`MAX_ZOOM`].
    pub z: u8,
    /// Column index, `0..2^z`, counting eastwards from 180°W.
    pub x: u32,
    /// Row index, `0..2^z`, counting southwards from the northern cut-off.
    pub y: u32,
}

impl TileId {
    /// Number of tiles per axis at `zoom` (`2^zoom`).
    ///
    /// `zoom` is clamped to [`MAX_ZOOM`] so the shift can never overflow.
    #[must_use]
    pub const fn tiles_per_axis(zoom: u8) -> u32 {
        let z = if zoom > MAX_ZOOM { MAX_ZOOM } else { zoom };
        1u32 << z
    }

    /// Creates a validated tile address.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::ZoomOutOfRange`] if `z` exceeds [`MAX_ZOOM`] and
    /// [`RenderError::InvalidTile`] if `x` or `y` is outside `0..2^z`.
    pub fn new(z: u8, x: u32, y: u32) -> Result<Self, RenderError> {
        if z > MAX_ZOOM {
            return Err(RenderError::ZoomOutOfRange {
                zoom: f64::from(z),
                max: MAX_ZOOM,
            });
        }
        let n = Self::tiles_per_axis(z);
        if x >= n {
            return Err(RenderError::InvalidTile {
                z,
                x,
                y,
                reason: "column index is not below 2^z",
            });
        }
        if y >= n {
            return Err(RenderError::InvalidTile {
                z,
                x,
                y,
                reason: "row index is not below 2^z",
            });
        }
        Ok(Self { z, x, y })
    }

    /// North-west corner of the tile in normalised Web Mercator space.
    #[must_use]
    pub fn north_west(&self) -> WorldCoord {
        let n = f64::from(Self::tiles_per_axis(self.z));
        WorldCoord {
            x: f64::from(self.x) / n,
            y: f64::from(self.y) / n,
        }
    }

    /// South-east corner of the tile in normalised Web Mercator space.
    #[must_use]
    pub fn south_east(&self) -> WorldCoord {
        let n = f64::from(Self::tiles_per_axis(self.z));
        WorldCoord {
            x: f64::from(self.x + 1) / n,
            y: f64::from(self.y + 1) / n,
        }
    }

    /// Centre of the tile in normalised Web Mercator space.
    #[must_use]
    pub fn center(&self) -> WorldCoord {
        let n = f64::from(Self::tiles_per_axis(self.z));
        WorldCoord {
            x: (f64::from(self.x) + 0.5) / n,
            y: (f64::from(self.y) + 0.5) / n,
        }
    }

    /// Geographic bounds of the tile as `(north_west, south_east)`.
    #[must_use]
    pub fn bounds_lon_lat(&self) -> (LonLat, LonLat) {
        (
            self.north_west().to_lon_lat(),
            self.south_east().to_lon_lat(),
        )
    }

    /// Bounds of the tile in EPSG:3857 metres.
    #[must_use]
    pub fn bounds_meters(&self) -> MercatorBounds {
        let nw = self.north_west().to_mercator_meters();
        let se = self.south_east().to_mercator_meters();
        MercatorBounds {
            min_x: nw.x,
            min_y: se.y,
            max_x: se.x,
            max_y: nw.y,
        }
    }

    /// Edge length of the tile in EPSG:3857 metres.
    #[must_use]
    pub fn size_meters(&self) -> f64 {
        EARTH_CIRCUMFERENCE_M / f64::from(Self::tiles_per_axis(self.z))
    }

    /// The tile one zoom level up that contains this one, or `None` at `z = 0`.
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        if self.z == 0 {
            return None;
        }
        Some(Self {
            z: self.z - 1,
            x: self.x / 2,
            y: self.y / 2,
        })
    }

    /// The tile `levels` zoom levels up that contains this one, or `None` if
    /// that would go above zoom 0.
    ///
    /// `levels == 0` returns the tile itself.
    #[must_use]
    pub fn ancestor(&self, levels: u8) -> Option<Self> {
        if levels > self.z {
            return None;
        }
        Some(Self {
            z: self.z - levels,
            x: self.x >> levels,
            y: self.y >> levels,
        })
    }

    /// The sub-rectangle this tile occupies inside `ancestor`, as
    /// `[u0, v0, width, height]` in `0..=1` texture coordinates.
    ///
    /// Returns `None` unless `ancestor` really is an ancestor of (or equal to)
    /// this tile, which is what makes the result safe to feed straight into a
    /// quad's UV rectangle.
    #[must_use]
    pub fn sub_rect_in(&self, ancestor: &Self) -> Option<[f32; 4]> {
        if ancestor.z > self.z {
            return None;
        }
        let levels = self.z - ancestor.z;
        if self.ancestor(levels)? != *ancestor {
            return None;
        }
        // `levels <= MAX_ZOOM`, so the span is at least 2^-24 and exact in f32,
        // and so is the index it scales — see the `MAX_ZOOM <= 24` assert.
        let span = 1.0f32 / ((1u32 << levels) as f32);
        let mask = (1u32 << levels) - 1;
        Some([
            (self.x & mask) as f32 * span,
            (self.y & mask) as f32 * span,
            span,
            span,
        ])
    }

    /// The TMS row of this tile: `2^z - 1 - y`.
    ///
    /// Total by construction — [`TileId::new`] already guarantees `y < 2^z` —
    /// so this is the form to prefer over [`crate::source::tms_row`] wherever a
    /// validated tile is at hand.
    #[must_use]
    pub const fn tms_row(&self) -> u32 {
        Self::tiles_per_axis(self.z) - 1 - self.y
    }

    /// The four tiles one zoom level down, or `None` at [`MAX_ZOOM`].
    ///
    /// Order is north-west, north-east, south-west, south-east.
    #[must_use]
    pub fn children(&self) -> Option<[Self; 4]> {
        if self.z >= MAX_ZOOM {
            return None;
        }
        let (z, x, y) = (self.z + 1, self.x * 2, self.y * 2);
        Some([
            Self { z, x, y },
            Self { z, x: x + 1, y },
            Self { z, x, y: y + 1 },
            Self {
                z,
                x: x + 1,
                y: y + 1,
            },
        ])
    }
}

/// Ground resolution in metres per pixel at `lat_deg` and (possibly
/// fractional) `zoom`, assuming [`TILE_SIZE_PX`]-wide tiles.
///
/// At the equator and zoom 0 this is the familiar `156543.034 m/px`; every
/// zoom level halves it and every degree away from the equator scales it by
/// `cos(lat)`. Non-finite inputs yield [`f64::NAN`].
#[must_use]
pub fn ground_resolution(lat_deg: f64, zoom: f64) -> f64 {
    let lat = lat_deg.clamp(-MAX_LATITUDE_DEG, MAX_LATITUDE_DEG);
    lat.to_radians().cos() * EARTH_CIRCUMFERENCE_M / (TILE_SIZE_PX * zoom.exp2())
}

#[cfg(test)]
mod tests {
    use super::{
        EARTH_CIRCUMFERENCE_M, LonLat, MAX_LATITUDE_DEG, MAX_ZOOM, MercatorPoint, TileId,
        WorldCoord, ground_resolution,
    };
    use crate::error::RenderError;
    use std::f64::consts::PI;

    /// Reference implementation of the slippy-map forward transform, written
    /// from the OSM wiki formulas and deliberately *not* sharing code with
    /// [`LonLat::to_world`]: it uses `asinh(tan(lat))` where the crate uses
    /// `ln(tan(lat) + sec(lat))`, and computes tile indices directly.
    fn reference_tile(lon_deg: f64, lat_deg: f64, zoom: u8) -> (u32, u32) {
        let n = 2f64.powi(i32::from(zoom));
        let lat_rad = lat_deg.to_radians();
        let x = (lon_deg + 180.0) / 360.0 * n;
        let y = (1.0 - lat_rad.tan().asinh() / PI) / 2.0 * n;
        (x.floor() as u32, y.floor() as u32)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what}: {actual} != {expected} (tolerance {tolerance})"
        );
    }

    #[test]
    fn constants_match_published_values() {
        // Values published for EPSG:3857 / the OSM wiki, independent of the
        // formulas implemented in this module.
        assert_close(
            EARTH_CIRCUMFERENCE_M / 2.0,
            20_037_508.342_789_244,
            1e-6,
            "half circumference",
        );
        assert_close(
            MAX_LATITUDE_DEG,
            PI.sinh().atan().to_degrees(),
            1e-12,
            "cut-off latitude",
        );
        assert_close(
            ground_resolution(0.0, 0.0),
            156_543.033_928_040_97,
            1e-6,
            "equatorial resolution at zoom 0",
        );
    }

    #[test]
    fn world_anchors() {
        let origin = LonLat::new(0.0, 0.0).to_world();
        assert_close(origin.x, 0.5, 1e-15, "null island x");
        assert_close(origin.y, 0.5, 1e-15, "null island y");

        let east = LonLat::new(180.0, 0.0).to_world();
        assert_close(east.x, 1.0, 1e-15, "antimeridian x");

        let top = LonLat::new(0.0, MAX_LATITUDE_DEG).to_world();
        assert_close(top.y, 0.0, 1e-12, "cut-off latitude y");
        let bottom = LonLat::new(0.0, -MAX_LATITUDE_DEG).to_world();
        assert_close(bottom.y, 1.0, 1e-12, "southern cut-off y");

        // Berlin, computed independently with the asinh form.
        let berlin = LonLat::new(13.377, 52.518).to_world();
        assert_close(berlin.x, 0.537_158_333_333_333_4, 1e-12, "berlin x");
        assert_close(berlin.y, 0.327_964_290_752_224_2, 1e-12, "berlin y");
    }

    #[test]
    fn mercator_meters_anchors() {
        let east = LonLat::new(180.0, 0.0).to_mercator_meters();
        assert_close(east.x, 20_037_508.342_789_244, 1e-6, "antimeridian metres");

        let london = LonLat::new(-0.1278, 51.5074).to_mercator_meters();
        assert_close(london.x, -14_226.630_923_380_362, 1e-6, "london easting");
        assert_close(london.y, 6_711_542.475_587_636, 1e-6, "london northing");

        let tokyo = LonLat::new(139.6917, 35.6895).to_mercator_meters();
        assert_close(tokyo.x, 15_550_408.912_046_732, 1e-6, "tokyo easting");
        assert_close(tokyo.y, 4_257_980.732_184_108, 1e-6, "tokyo northing");
    }

    #[test]
    fn both_mercator_paths_agree() {
        // `LonLat -> metres` and `LonLat -> world -> metres` are two different
        // code paths; they must produce the same projected coordinate.
        for &(lon, lat) in &[
            (0.0, 0.0),
            (13.377, 52.518),
            (-122.4194, 37.7749),
            (139.6917, -35.6895),
            (179.9, 84.0),
            (-179.9, -84.0),
        ] {
            let direct = LonLat::new(lon, lat).to_mercator_meters();
            let via_world = LonLat::new(lon, lat).to_world().to_mercator_meters();
            assert_close(via_world.x, direct.x, 1e-6, "easting agreement");
            assert_close(via_world.y, direct.y, 1e-6, "northing agreement");
        }
    }

    #[test]
    fn round_trips_are_stable() {
        let mut lon = -179.5;
        while lon <= 179.5 {
            let mut lat = -84.0;
            while lat <= 84.0 {
                let start = LonLat::new(lon, lat);

                let world_back = start.to_world().to_lon_lat();
                assert_close(world_back.lon, lon, 1e-9, "world round trip lon");
                assert_close(world_back.lat, lat, 1e-9, "world round trip lat");

                let meters_back = start.to_mercator_meters().to_lon_lat();
                assert_close(meters_back.lon, lon, 1e-9, "metres round trip lon");
                assert_close(meters_back.lat, lat, 1e-9, "metres round trip lat");

                let world_meters =
                    WorldCoord::from_mercator_meters(start.to_world().to_mercator_meters());
                assert_close(world_meters.x, start.to_world().x, 1e-12, "world x");
                assert_close(world_meters.y, start.to_world().y, 1e-12, "world y");

                lat += 21.0;
            }
            lon += 59.5;
        }
    }

    #[test]
    fn tile_indices_match_reference_formula() {
        for &(lon, lat, zoom) in &[
            (-0.1278, 51.5074, 16u8),
            (139.6917, 35.6895, 12),
            (-122.4194, 37.7749, 10),
            (13.377, 52.518, 18),
            (0.0, 0.0, 1),
            (-180.0, 85.0, 5),
        ] {
            let expected = reference_tile(lon, lat, zoom);
            let Ok(tile) = LonLat::new(lon, lat).tile(zoom) else {
                panic!("tile lookup failed for {lon},{lat} at z{zoom}");
            };
            assert_eq!((tile.x, tile.y), expected, "tile for {lon},{lat} z{zoom}");
        }
    }

    #[test]
    fn tile_indices_match_precomputed_anchors() {
        // Precomputed outside this crate with the standard slippy-map formula.
        let cases = [
            (-0.1278, 51.5074, 16u8, 32_744u32, 21_792u32),
            (139.6917, 35.6895, 12, 3_637, 1_612),
            (-122.4194, 37.7749, 10, 163, 395),
            (13.377, 52.518, 18, 140_812, 85_973),
        ];
        for (lon, lat, zoom, x, y) in cases {
            let Ok(tile) = LonLat::new(lon, lat).tile(zoom) else {
                panic!("tile lookup failed");
            };
            assert_eq!(tile, TileId { z: zoom, x, y });
        }
    }

    #[test]
    fn zoom_one_quadrants() {
        let cases = [
            (-90.0, 45.0, 0u32, 0u32),
            (90.0, 45.0, 1, 0),
            (-90.0, -45.0, 0, 1),
            (90.0, -45.0, 1, 1),
        ];
        for (lon, lat, x, y) in cases {
            let Ok(tile) = LonLat::new(lon, lat).tile(1) else {
                panic!("tile lookup failed");
            };
            assert_eq!((tile.x, tile.y), (x, y), "quadrant for {lon},{lat}");
        }
    }

    #[test]
    fn tile_validation() {
        assert!(TileId::new(0, 0, 0).is_ok());
        assert!(TileId::new(1, 1, 1).is_ok());
        assert!(matches!(
            TileId::new(1, 2, 0),
            Err(RenderError::InvalidTile { .. })
        ));
        assert!(matches!(
            TileId::new(1, 0, 2),
            Err(RenderError::InvalidTile { .. })
        ));
        assert!(matches!(
            TileId::new(MAX_ZOOM + 1, 0, 0),
            Err(RenderError::ZoomOutOfRange { .. })
        ));
        assert_eq!(TileId::tiles_per_axis(0), 1);
        assert_eq!(TileId::tiles_per_axis(10), 1024);
        assert_eq!(TileId::tiles_per_axis(22), 4_194_304);
        // Stated against the constant so raising it does not need this edited,
        // and so the shift is checked at whatever the limit currently is.
        assert_eq!(TileId::tiles_per_axis(MAX_ZOOM), 1u32 << MAX_ZOOM);
    }

    #[test]
    fn tile_bounds_in_meters() {
        let Ok(world) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        let bounds = world.bounds_meters();
        assert_close(bounds.min_x, -20_037_508.342_789_244, 1e-6, "world min x");
        assert_close(bounds.min_y, -20_037_508.342_789_244, 1e-6, "world min y");
        assert_close(bounds.max_x, 20_037_508.342_789_244, 1e-6, "world max x");
        assert_close(bounds.max_y, 20_037_508.342_789_244, 1e-6, "world max y");
        assert_close(bounds.center().x, 0.0, 1e-6, "world centre x");
        assert_close(bounds.center().y, 0.0, 1e-6, "world centre y");

        let Ok(north_east) = TileId::new(1, 1, 0) else {
            panic!("tile 1/1/0 is valid");
        };
        let bounds = north_east.bounds_meters();
        assert_close(bounds.min_x, 0.0, 1e-6, "quadrant min x");
        assert_close(bounds.min_y, 0.0, 1e-6, "quadrant min y");
        assert_close(bounds.max_x, 20_037_508.342_789_244, 1e-6, "quadrant max x");
        assert_close(bounds.max_y, 20_037_508.342_789_244, 1e-6, "quadrant max y");
        assert_close(bounds.width(), bounds.height(), 1e-9, "tiles are square");
        assert_close(
            north_east.size_meters(),
            EARTH_CIRCUMFERENCE_M / 2.0,
            1e-6,
            "tile edge at z1",
        );
    }

    #[test]
    fn tile_corners_agree_with_geographic_bounds() {
        let Ok(tile) = TileId::new(4, 8, 5) else {
            panic!("tile 4/8/5 is valid");
        };
        let (nw, se) = tile.bounds_lon_lat();
        assert!(nw.lon < se.lon, "west of east");
        assert!(nw.lat > se.lat, "north of south");
        assert_close(nw.lon, 0.0, 1e-9, "tile 4/8/5 western edge");

        // The tile that contains the centre of a tile is that tile.
        let Ok(again) = tile.center().tile(4) else {
            panic!("centre lookup failed");
        };
        assert_eq!(again, tile);
    }

    #[test]
    fn pyramid_navigation() {
        let Ok(tile) = TileId::new(3, 5, 2) else {
            panic!("tile 3/5/2 is valid");
        };
        assert_eq!(tile.parent(), Some(TileId { z: 2, x: 2, y: 1 }));
        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        assert_eq!(root.parent(), None);
        assert_eq!(
            root.children(),
            Some([
                TileId { z: 1, x: 0, y: 0 },
                TileId { z: 1, x: 1, y: 0 },
                TileId { z: 1, x: 0, y: 1 },
                TileId { z: 1, x: 1, y: 1 },
            ])
        );
        for child in root.children().into_iter().flatten() {
            assert_eq!(child.parent(), Some(root));
        }
    }

    #[test]
    fn ancestors_walk_the_pyramid_upwards() {
        let Ok(tile) = TileId::new(5, 21, 13) else {
            panic!("tile 5/21/13 is valid");
        };
        assert_eq!(tile.ancestor(0), Some(tile));
        assert_eq!(tile.ancestor(1), tile.parent());
        assert_eq!(tile.ancestor(2), Some(TileId { z: 3, x: 5, y: 3 }));
        assert_eq!(tile.ancestor(5), Some(TileId { z: 0, x: 0, y: 0 }));
        assert_eq!(tile.ancestor(6), None, "there is nothing above zoom 0");

        // Repeated `parent()` and one `ancestor()` must agree at every step.
        for levels in 0..=5u8 {
            let mut walked = tile;
            for _ in 0..levels {
                let Some(up) = walked.parent() else {
                    panic!("walked above zoom 0");
                };
                walked = up;
            }
            assert_eq!(tile.ancestor(levels), Some(walked), "{levels} levels up");
        }
    }

    #[test]
    fn a_tile_sub_rect_tiles_its_ancestor_exactly() {
        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        assert_eq!(root.sub_rect_in(&root), Some([0.0, 0.0, 1.0, 1.0]));

        // The four children of the root each take one quadrant.
        let Some(children) = root.children() else {
            panic!("the root has children");
        };
        let quadrants = [
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5, 0.5],
            [0.0, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ];
        for (child, expected) in children.into_iter().zip(quadrants) {
            assert_eq!(child.sub_rect_in(&root), Some(expected), "{child:?}");
        }

        // Two levels down: a 1/4 x 1/4 cell at the child's own offset.
        let Ok(deep) = TileId::new(4, 11, 6) else {
            panic!("tile 4/11/6 is valid");
        };
        let Some(grandparent) = deep.ancestor(2) else {
            panic!("tile 4/11/6 has a grandparent");
        };
        assert_eq!(grandparent, TileId { z: 2, x: 2, y: 1 });
        assert_eq!(
            deep.sub_rect_in(&grandparent),
            Some([0.75, 0.5, 0.25, 0.25])
        );

        // A tile that is not an ancestor — same zoom, different branch — and a
        // deeper "ancestor" are both refused rather than producing a rectangle.
        let Ok(stranger) = TileId::new(2, 0, 0) else {
            panic!("tile 2/0/0 is valid");
        };
        assert_eq!(deep.sub_rect_in(&stranger), None);
        assert_eq!(grandparent.sub_rect_in(&deep), None);
    }

    #[test]
    fn the_deepest_sub_rect_is_still_exact_in_f32() {
        // What the `MAX_ZOOM <= 24` assert buys: at the deepest level the tile
        // index reaches `2^MAX_ZOOM - 1`, the largest integer f32 still holds
        // exactly. One level deeper it would round, and the last column's UV
        // rectangle would land on its neighbour with no overflow to notice.
        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        let last = TileId::tiles_per_axis(MAX_ZOOM) - 1;
        let Ok(deep) = TileId::new(MAX_ZOOM, last, last) else {
            panic!("the last tile of the deepest level is valid");
        };
        let Some([u0, v0, width, height]) = deep.sub_rect_in(&root) else {
            panic!("every tile sits inside the root");
        };
        let span = 1.0f32 / (TileId::tiles_per_axis(MAX_ZOOM) as f32);
        // Exact equality on purpose: bit-exactness is the property under test.
        assert_eq!(width, span);
        assert_eq!(height, span);
        assert_eq!(u0, 1.0 - span, "the last column ends flush with the edge");
        assert_eq!(v0, 1.0 - span, "the last row ends flush with the edge");
        assert!(u0 < 1.0, "a rounded index would sit at or past the edge");
    }

    #[test]
    fn the_sub_rect_of_every_child_covers_the_parent_without_overlap() {
        let Ok(parent) = TileId::new(3, 5, 2) else {
            panic!("tile 3/5/2 is valid");
        };
        for levels in 1..=3u8 {
            let step = 1u32 << levels;
            let mut area = 0.0f32;
            for dy in 0..step {
                for dx in 0..step {
                    let Ok(child) = TileId::new(
                        parent.z + levels,
                        parent.x * step + dx,
                        parent.y * step + dy,
                    ) else {
                        panic!("child is valid");
                    };
                    let Some([u, v, du, dv]) = child.sub_rect_in(&parent) else {
                        panic!("{child:?} is a descendant of {parent:?}");
                    };
                    assert!((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v));
                    assert!((u + du) <= 1.0 + f32::EPSILON);
                    assert!((v + dv) <= 1.0 + f32::EPSILON);
                    area += du * dv;
                }
            }
            assert!((area - 1.0).abs() < 1e-5, "{levels} levels cover {area}");
        }
    }

    #[test]
    fn the_tms_row_is_the_mirror_of_the_xyz_row() {
        for z in 0..=6u8 {
            let side = TileId::tiles_per_axis(z);
            for y in [0, side / 2, side - 1] {
                let Ok(tile) = TileId::new(z, 0, y) else {
                    panic!("tile {z}/0/{y} is valid");
                };
                assert_eq!(tile.tms_row(), side - 1 - y);
                // Flipping twice is the identity, which is the property the
                // MBTiles reader depends on.
                let Ok(flipped) = TileId::new(z, 0, tile.tms_row()) else {
                    panic!("the flipped row is still in range");
                };
                assert_eq!(flipped.tms_row(), y);
            }
        }
    }

    #[test]
    fn ground_resolution_halves_per_zoom() {
        let equator_z0 = ground_resolution(0.0, 0.0);
        for z in 0..12 {
            let expected = equator_z0 / 2f64.powi(z);
            assert_close(
                ground_resolution(0.0, f64::from(z)),
                expected,
                1e-6,
                "resolution at zoom",
            );
        }
        // cos(60 deg) = 0.5 exactly.
        assert_close(
            ground_resolution(60.0, 10.0),
            76.437_028_285_176_27,
            1e-9,
            "resolution at 60N zoom 10",
        );
        assert_close(
            ground_resolution(60.0, 10.0) * 2.0,
            ground_resolution(0.0, 10.0),
            1e-9,
            "cosine scaling",
        );
    }

    #[test]
    fn normalization_clamps_and_wraps() {
        let wrapped = LonLat::new(200.0, 95.0).normalized();
        assert_close(wrapped.lon, -160.0, 1e-12, "wrapped longitude");
        assert_close(wrapped.lat, MAX_LATITUDE_DEG, 1e-12, "clamped latitude");

        let east_edge = LonLat::new(180.0, 0.0).normalized();
        assert_close(east_edge.lon, 180.0, 1e-12, "eastern edge kept");

        let broken = LonLat::new(f64::NAN, f64::INFINITY).normalized();
        assert_close(broken.lon, 0.0, 0.0, "non-finite longitude");
        assert_close(broken.lat, 0.0, 0.0, "non-finite latitude");

        let outside = WorldCoord::new(-3.0, 7.0).clamped();
        assert_close(outside.x, 0.0, 0.0, "clamped world x");
        assert_close(outside.y, 1.0, 0.0, "clamped world y");
    }

    #[test]
    fn mercator_point_round_trip() {
        let point = MercatorPoint::new(1_489_120.828_341_620_7, 6_894_333.918_619_941);
        let back = point.to_lon_lat();
        assert_close(back.lon, 13.377, 1e-9, "berlin lon");
        assert_close(back.lat, 52.518, 1e-9, "berlin lat");
        let world = point.to_world();
        assert_close(world.x, 0.537_158_333_333_333_4, 1e-12, "berlin world x");
    }
}
