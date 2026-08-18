//! Viewport state: what part of the world is on screen, and where each tile
//! lands in pixels.
//!
//! [`MapView`] is the single source of truth for the camera (centre, zoom,
//! surface size). From it two things are derived:
//!
//! * [`MapView::visible_tiles`] — the tiles of the pyramid that intersect the
//!   viewport at the current *integer* zoom, clamped to the valid grid,
//!   de-duplicated and sorted centre-outward so the middle of the screen is
//!   requested (and drawn) first.
//! * [`MapView::place_tile`] — where a tile's north-west corner lands, in
//!   physical pixels measured from the top-left of the surface, and how wide
//!   the tile is once the fractional part of the zoom is applied.
//!
//! Screen pixels use the usual top-left origin with `y` growing downwards,
//! which matches [`WorldCoord`] and therefore keeps the placement maths a pure
//! scale-and-offset with no axis flip.

use crate::error::RenderError;
use crate::mercator::{LonLat, MAX_ZOOM, TILE_SIZE_PX, TileId, WorldCoord};

/// Upper bound on the number of tiles [`MapView::visible_tiles`] will return.
///
/// The list is truncated *after* the centre-outward sort, so the tiles that
/// survive are the ones nearest the middle of the screen. A 4096-tile budget
/// covers a 16384x16384 physical-pixel surface at native tile scale.
pub const MAX_VISIBLE_TILES: usize = 4096;

/// Largest accepted surface edge, in physical pixels.
pub const MAX_VIEWPORT_PX: f32 = 65_536.0;

/// The map camera: where we are looking, how far in, and how big the surface is.
///
/// Constructed through [`MapView::new`], which normalises the centre and clamps
/// the zoom, so an existing `MapView` is always renderable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapView {
    center: LonLat,
    zoom: f64,
    size_px: [f32; 2],
}

impl MapView {
    /// Creates a viewport.
    ///
    /// `center` is normalised (longitude wrapped, latitude clamped to the
    /// Mercator cut-off) and `zoom` is clamped to `0..=`[`MAX_ZOOM`].
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidViewport`] if `zoom` is not finite or if
    /// either surface dimension is not finite, not positive, or larger than
    /// [`MAX_VIEWPORT_PX`].
    pub fn new(center: LonLat, zoom: f64, size_px: [f32; 2]) -> Result<Self, RenderError> {
        if !zoom.is_finite() {
            return Err(RenderError::InvalidViewport(format!(
                "zoom must be finite, got {zoom}"
            )));
        }
        Self::check_size(size_px)?;
        Ok(Self {
            center: center.normalized(),
            zoom: zoom.clamp(0.0, f64::from(MAX_ZOOM)),
            size_px,
        })
    }

    fn check_size(size_px: [f32; 2]) -> Result<(), RenderError> {
        for (axis, value) in ["width", "height"].into_iter().zip(size_px) {
            if !value.is_finite() || value <= 0.0 || value > MAX_VIEWPORT_PX {
                return Err(RenderError::InvalidViewport(format!(
                    "{axis} must be in (0, {MAX_VIEWPORT_PX}], got {value}"
                )));
            }
        }
        Ok(())
    }

    /// Geographic centre of the viewport.
    #[must_use]
    pub fn center(&self) -> LonLat {
        self.center
    }

    /// Fractional zoom level.
    #[must_use]
    pub fn zoom(&self) -> f64 {
        self.zoom
    }

    /// Surface size in physical pixels, `[width, height]`.
    #[must_use]
    pub fn size_px(&self) -> [f32; 2] {
        self.size_px
    }

    /// Returns a copy centred on `center` (normalised).
    #[must_use]
    pub fn with_center(self, center: LonLat) -> Self {
        Self {
            center: center.normalized(),
            ..self
        }
    }

    /// Returns a copy at `zoom`, clamped to `0..=`[`MAX_ZOOM`].
    ///
    /// A non-finite `zoom` leaves the view unchanged.
    #[must_use]
    pub fn with_zoom(self, zoom: f64) -> Self {
        if !zoom.is_finite() {
            return self;
        }
        Self {
            zoom: zoom.clamp(0.0, f64::from(MAX_ZOOM)),
            ..self
        }
    }

    /// Returns a copy with a new surface size.
    ///
    /// # Errors
    ///
    /// Same conditions as [`MapView::new`].
    pub fn with_size_px(self, size_px: [f32; 2]) -> Result<Self, RenderError> {
        Self::check_size(size_px)?;
        Ok(Self { size_px, ..self })
    }

    /// Integer zoom level whose tiles are drawn for this view.
    ///
    /// This is `floor(zoom)`: tiles are never magnified beyond their own level
    /// by more than a factor of two, which is the usual slippy-map behaviour.
    #[must_use]
    pub fn tile_zoom(&self) -> u8 {
        let z = self.zoom.floor().clamp(0.0, f64::from(MAX_ZOOM));
        z as u8
    }

    /// Width of the whole world in physical pixels at the current zoom.
    #[must_use]
    pub fn world_pixels(&self) -> f64 {
        TILE_SIZE_PX * self.zoom.exp2()
    }

    /// On-screen edge length of a tile at [`MapView::tile_zoom`], in pixels.
    ///
    /// Equal to `256 * 2^(zoom - floor(zoom))`, i.e. `256..512`.
    #[must_use]
    pub fn tile_size_px(&self) -> f32 {
        (self.world_pixels() / f64::from(TileId::tiles_per_axis(self.tile_zoom()))) as f32
    }

    /// Viewport corners in normalised Web Mercator space, as
    /// `(north_west, south_east)`.
    ///
    /// The corners are *not* clamped to the unit square: when the map is zoomed
    /// out far enough to show the whole world, they legitimately fall outside
    /// it, and clamping here would silently distort the placement maths.
    #[must_use]
    pub fn world_bounds(&self) -> (WorldCoord, WorldCoord) {
        let scale = self.world_pixels();
        let center = self.center.to_world();
        let half_w = f64::from(self.size_px[0]) / 2.0 / scale;
        let half_h = f64::from(self.size_px[1]) / 2.0 / scale;
        (
            WorldCoord::new(center.x - half_w, center.y - half_h),
            WorldCoord::new(center.x + half_w, center.y + half_h),
        )
    }

    /// Tiles covering the viewport at [`MapView::tile_zoom`].
    ///
    /// Indices are clamped to the `0..2^z` grid, de-duplicated, then sorted by
    /// squared distance from the viewport centre with a `(z, x, y)` tiebreak so
    /// the result is deterministic. At most [`MAX_VISIBLE_TILES`] entries are
    /// returned.
    ///
    /// Clamping, not wrapping: this list is one entry per *tile*, and a
    /// viewport straddling the antimeridian needs one entry per on-screen
    /// *copy* of a tile to be filled completely. That is
    /// [`MapView::visible_placements_into`]; the two live side by side because
    /// a caller pairing tiles with per-tile resources wants exactly one entry
    /// each.
    #[must_use]
    pub fn visible_tiles(&self) -> Vec<TileId> {
        let z = self.tile_zoom();
        let n = TileId::tiles_per_axis(z);
        let last = i64::from(n) - 1;
        let count = f64::from(n);
        let (north_west, south_east) = self.world_bounds();

        // `floor` for the inclusive lower edge, `ceil() - 1` for the upper one:
        // a viewport whose right edge lands exactly on a tile boundary must not
        // drag in the zero-width column beyond it.
        let axis_range = |low: f64, high: f64| -> (i64, i64) {
            let first = ((low * count).floor().clamp(0.0, count) as i64).clamp(0, last);
            let raw_last = (high * count).ceil().clamp(0.0, count) as i64 - 1;
            (first, raw_last.clamp(first, last))
        };
        let (x_first, x_last) = axis_range(north_west.x, south_east.x);
        let (y_first, y_last) = axis_range(north_west.y, south_east.y);

        let mut tiles = Vec::new();
        for y in y_first..=y_last {
            for x in x_first..=x_last {
                if let Ok(tile) = TileId::new(z, x as u32, y as u32) {
                    tiles.push(tile);
                }
            }
        }
        tiles.sort_unstable();
        tiles.dedup();

        let center = self.center.to_world();
        tiles.sort_by(|a, b| {
            let key = |tile: &TileId| {
                let c = tile.center();
                let dx = c.x - center.x;
                let dy = c.y - center.y;
                dx * dx + dy * dy
            };
            key(a).total_cmp(&key(b)).then_with(|| a.cmp(b))
        });
        tiles.truncate(MAX_VISIBLE_TILES);
        tiles
    }

    /// Where `tile` lands on screen, in physical pixels.
    ///
    /// Works for any zoom level, not just [`MapView::tile_zoom`], so an
    /// over-zoomed parent tile can be used as a placeholder while its children
    /// are still being fetched.
    #[must_use]
    pub fn place_tile(&self, tile: TileId) -> TilePlacement {
        self.place_tile_in_copy(tile, 0)
    }

    /// [`MapView::place_tile`] for the repeat of the world map `copy` copies
    /// east of the primary one; negative values are west.
    ///
    /// The offset is applied in normalised world space *before* the pixel
    /// scale, not to the resulting pixel position: at zoom 22 one world is over
    /// a billion pixels wide, far past what an `f32` can offset without losing
    /// whole tiles.
    #[must_use]
    pub fn place_tile_in_copy(&self, tile: TileId, copy: i32) -> TilePlacement {
        let scale = self.world_pixels();
        let center = self.center.to_world();
        let north_west = tile.north_west();
        let size = scale / f64::from(TileId::tiles_per_axis(tile.z));
        let world_x = north_west.x + f64::from(copy);
        TilePlacement {
            tile,
            x: ((world_x - center.x) * scale + f64::from(self.size_px[0]) / 2.0) as f32,
            y: ((north_west.y - center.y) * scale + f64::from(self.size_px[1]) / 2.0) as f32,
            size: size as f32,
        }
    }

    /// [`MapView::visible_tiles`], already placed on screen.
    ///
    /// One placement per tile, all in the primary world copy — the shape every
    /// caller that pairs a placement with a per-tile resource (a mesh, a label
    /// set) needs. Callers that want the map to continue across the
    /// antimeridian want [`MapView::visible_placements_into`].
    #[must_use]
    pub fn visible_placements(&self) -> Vec<TilePlacement> {
        self.visible_tiles()
            .into_iter()
            .map(|tile| self.place_tile(tile))
            .collect()
    }

    /// Fills `out` with the tiles covering the viewport at `zoom`, **one entry
    /// per on-screen copy of the world**, ordered centre-outward.
    ///
    /// Three things separate this from [`MapView::visible_placements`]:
    ///
    /// * `zoom` is explicit, so a caller whose source stops at zoom 14 can ask
    ///   for z14 tiles while the camera is at z18 and have them placed
    ///   magnified rather than showing nothing.
    /// * Columns **wrap** modulo `2^zoom` instead of being clamped, so a
    ///   viewport straddling 180° is filled on both sides. The [`TileId`] is
    ///   always the wrapped one — that is what a tile source is asked for — and
    ///   the world copy it belongs to is folded into the placement's `x`. A
    ///   viewport wider than the whole world therefore yields the same tile
    ///   more than once, at different `x`; callers building a fetch queue must
    ///   de-duplicate by [`TilePlacement::tile`].
    /// * `out` is cleared and refilled, so a caller driving one frame per
    ///   display refresh keeps a single allocation.
    ///
    /// Rows stay clamped to `0..2^zoom`: there is no world above the Mercator
    /// cut-off to wrap to. At most [`MAX_VISIBLE_TILES`] entries are produced.
    pub fn visible_placements_into(&self, zoom: u8, out: &mut Vec<TilePlacement>) {
        out.clear();
        let z = zoom.min(MAX_ZOOM);
        let n = TileId::tiles_per_axis(z);
        let columns = i64::from(n);
        let last = columns - 1;
        let count = f64::from(n);
        let (north_west, south_east) = self.world_bounds();

        // The x span is bounded by the surface width over the on-screen tile
        // size (at most `MAX_VIEWPORT_PX / 256` columns), so the unclamped
        // range cannot blow up even when the viewport shows many world copies.
        let span_limit = MAX_VISIBLE_TILES as i64;
        let to_index = |value: f64| -> i64 {
            if value.is_finite() {
                value.clamp(-1e15, 1e15) as i64
            } else {
                0
            }
        };
        let x_first = to_index((north_west.x * count).floor());
        let x_last = to_index((south_east.x * count).ceil() - 1.0)
            .max(x_first)
            .min(x_first.saturating_add(span_limit));
        let y_first = to_index((north_west.y * count).floor()).clamp(0, last);
        let y_last = to_index((south_east.y * count).ceil() - 1.0).clamp(y_first, last);

        // Candidates are truncated to `MAX_VISIBLE_TILES` after the sort, so
        // collecting a bounded multiple of it keeps the worst case cheap
        // without changing which tiles survive at any realistic surface size.
        let candidate_limit = MAX_VISIBLE_TILES.saturating_mul(4);
        'rows: for y in y_first..=y_last {
            let Ok(row) = u32::try_from(y) else {
                continue;
            };
            for unwrapped in x_first..=x_last {
                let wrapped = unwrapped.rem_euclid(columns);
                let Ok(column) = u32::try_from(wrapped) else {
                    continue;
                };
                let Ok(tile) = TileId::new(z, column, row) else {
                    continue;
                };
                // Exact: the difference is a multiple of `columns` and the span
                // is bounded well inside `i32`.
                let copy = i32::try_from((unwrapped - wrapped) / columns).unwrap_or(0);
                out.push(self.place_tile_in_copy(tile, copy));
                if out.len() >= candidate_limit {
                    break 'rows;
                }
            }
        }

        // Centre-outward, keyed on the *placed* position so that repeated world
        // copies sort by where they actually are. The key is computed from two
        // subtractions rather than from `TileId::center`, which would divide.
        let center_x = self.size_px[0] / 2.0;
        let center_y = self.size_px[1] / 2.0;
        let distance = |placement: &TilePlacement| {
            let half = placement.size / 2.0;
            let dx = placement.x + half - center_x;
            let dy = placement.y + half - center_y;
            dx.mul_add(dx, dy * dy)
        };
        out.sort_by(|a, b| {
            distance(a)
                .total_cmp(&distance(b))
                .then_with(|| a.tile.cmp(&b.tile))
                .then_with(|| a.x.total_cmp(&b.x))
        });
        out.truncate(MAX_VISIBLE_TILES);
    }

    /// Converts a physical-pixel position on the surface to a geographic one.
    #[must_use]
    pub fn screen_to_lon_lat(&self, px: [f32; 2]) -> LonLat {
        let scale = self.world_pixels();
        let center = self.center.to_world();
        WorldCoord::new(
            center.x + (f64::from(px[0]) - f64::from(self.size_px[0]) / 2.0) / scale,
            center.y + (f64::from(px[1]) - f64::from(self.size_px[1]) / 2.0) / scale,
        )
        .to_lon_lat()
    }

    /// Converts a geographic position to a physical-pixel position on the
    /// surface. The result may fall outside the surface rectangle.
    #[must_use]
    pub fn lon_lat_to_screen(&self, position: LonLat) -> [f32; 2] {
        let scale = self.world_pixels();
        let center = self.center.to_world();
        let world = position.to_world();
        [
            ((world.x - center.x) * scale + f64::from(self.size_px[0]) / 2.0) as f32,
            ((world.y - center.y) * scale + f64::from(self.size_px[1]) / 2.0) as f32,
        ]
    }
}

/// A tile positioned on the render surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TilePlacement {
    /// The tile this placement belongs to.
    pub tile: TileId,
    /// Physical-pixel x of the tile's north-west corner, from the left edge.
    pub x: f32,
    /// Physical-pixel y of the tile's north-west corner, from the top edge.
    pub y: f32,
    /// On-screen edge length of the tile in physical pixels.
    pub size: f32,
}

impl TilePlacement {
    /// Converts the placement to a normalised-device-coordinate rectangle
    /// `[x, y, width, height]` for the quad shader.
    ///
    /// `x`/`y` are the top-left corner in NDC (`-1..1`, `y` up) and
    /// `width`/`height` are positive NDC extents, so the vertex shader can walk
    /// the quad with `x + u * width`, `y - v * height`.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidViewport`] if either surface dimension is
    /// not finite and positive.
    pub fn to_ndc_rect(&self, view_size_px: [f32; 2]) -> Result<[f32; 4], RenderError> {
        let [width, height] = view_size_px;
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return Err(RenderError::InvalidViewport(format!(
                "surface size must be positive and finite, got {width}x{height}"
            )));
        }
        Ok([
            2.0 * self.x / width - 1.0,
            1.0 - 2.0 * self.y / height,
            2.0 * self.size / width,
            2.0 * self.size / height,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_VIEWPORT_PX, MAX_VISIBLE_TILES, MapView, TilePlacement};
    use crate::error::RenderError;
    use crate::mercator::{LonLat, MAX_ZOOM, TileId};

    fn view(zoom: f64, size: [f32; 2]) -> MapView {
        match MapView::new(LonLat::new(0.0, 0.0), zoom, size) {
            Ok(view) => view,
            Err(err) => panic!("view construction failed: {err}"),
        }
    }

    #[test]
    fn construction_validates_inputs() {
        assert!(MapView::new(LonLat::new(0.0, 0.0), 3.0, [800.0, 600.0]).is_ok());
        assert!(matches!(
            MapView::new(LonLat::new(0.0, 0.0), f64::NAN, [800.0, 600.0]),
            Err(RenderError::InvalidViewport(_))
        ));
        assert!(matches!(
            MapView::new(LonLat::new(0.0, 0.0), 3.0, [0.0, 600.0]),
            Err(RenderError::InvalidViewport(_))
        ));
        assert!(matches!(
            MapView::new(LonLat::new(0.0, 0.0), 3.0, [800.0, MAX_VIEWPORT_PX + 1.0]),
            Err(RenderError::InvalidViewport(_))
        ));
        assert!(matches!(
            MapView::new(LonLat::new(0.0, 0.0), 3.0, [f32::NAN, 600.0]),
            Err(RenderError::InvalidViewport(_))
        ));
    }

    #[test]
    fn zoom_is_clamped_and_floored() {
        let clamped = view(99.0, [256.0, 256.0]);
        assert!((clamped.zoom() - f64::from(MAX_ZOOM)).abs() < 1e-12);
        assert_eq!(clamped.tile_zoom(), MAX_ZOOM);

        let negative = view(-4.0, [256.0, 256.0]);
        assert!(negative.zoom().abs() < 1e-12);
        assert_eq!(negative.tile_zoom(), 0);

        let fractional = view(3.75, [256.0, 256.0]);
        assert_eq!(fractional.tile_zoom(), 3);
        assert!((f64::from(fractional.tile_size_px()) - 256.0 * 0.75_f64.exp2()).abs() < 1e-3);

        assert_eq!(view(2.0, [256.0, 256.0]).with_zoom(f64::NAN).zoom(), 2.0);
    }

    #[test]
    fn world_zoom_shows_exactly_one_tile() {
        let tiles = view(0.0, [256.0, 256.0]).visible_tiles();
        assert_eq!(tiles, vec![TileId { z: 0, x: 0, y: 0 }]);
    }

    #[test]
    fn zoom_one_shows_four_tiles() {
        let mut tiles = view(1.0, [512.0, 512.0]).visible_tiles();
        assert_eq!(tiles.len(), 4);
        tiles.sort_unstable();
        assert_eq!(
            tiles,
            vec![
                TileId { z: 1, x: 0, y: 0 },
                TileId { z: 1, x: 0, y: 1 },
                TileId { z: 1, x: 1, y: 0 },
                TileId { z: 1, x: 1, y: 1 },
            ]
        );
    }

    #[test]
    fn viewport_edge_on_tile_boundary_adds_no_column() {
        // Centre at the middle of tile 1/0/0, viewport exactly one tile wide:
        // the eastern edge lands on the 0.5 boundary and must not pull in x=1.
        let Ok(center) = TileId::new(1, 0, 0) else {
            panic!("tile 1/0/0 is valid");
        };
        let Ok(view) = MapView::new(center.center().to_lon_lat(), 1.0, [256.0, 256.0]) else {
            panic!("view construction failed");
        };
        assert_eq!(view.visible_tiles(), vec![center]);
    }

    #[test]
    fn tiles_are_unique_and_sorted_center_outward() {
        let Ok(view) = MapView::new(LonLat::new(2.0, 41.0), 6.0, [1024.0, 768.0]) else {
            panic!("view construction failed");
        };
        let tiles = view.visible_tiles();
        assert!(tiles.len() > 4, "expected a multi-tile viewport");

        let mut unique = tiles.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), tiles.len(), "tile list must be de-duplicated");

        let center = view.center().to_world();
        let distance = |tile: &TileId| {
            let c = tile.center();
            let dx = c.x - center.x;
            let dy = c.y - center.y;
            dx * dx + dy * dy
        };
        for pair in tiles.windows(2) {
            assert!(
                distance(&pair[0]) <= distance(&pair[1]),
                "tiles must be ordered centre-outward"
            );
        }
        assert!(tiles.len() <= MAX_VISIBLE_TILES);
    }

    #[test]
    fn tiles_are_clamped_to_the_grid() {
        // A viewport far wider than the world at zoom 1 clamps instead of
        // wrapping, and still yields every tile exactly once.
        let tiles = view(1.0, [4096.0, 4096.0]).visible_tiles();
        assert_eq!(tiles.len(), 4);
        for tile in &tiles {
            assert!(tile.x < 2 && tile.y < 2, "tile {tile:?} escaped the grid");
        }
    }

    #[test]
    fn placement_of_the_world_tile() {
        let view = view(0.0, [256.0, 256.0]);
        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        let placement = view.place_tile(root);
        assert!(placement.x.abs() < 1e-4, "x = {}", placement.x);
        assert!(placement.y.abs() < 1e-4, "y = {}", placement.y);
        assert!((placement.size - 256.0).abs() < 1e-4);
    }

    #[test]
    fn placement_tiles_the_screen_without_gaps() {
        let view = view(1.0, [512.0, 512.0]);
        let mut expected = std::collections::BTreeMap::new();
        for tile in view.visible_tiles() {
            let placement = view.place_tile(tile);
            expected.insert((tile.x, tile.y), (placement.x, placement.y));
            assert!((placement.size - 256.0).abs() < 1e-4);
        }
        assert_eq!(expected.get(&(0, 0)), Some(&(0.0, 0.0)));
        assert_eq!(expected.get(&(1, 0)), Some(&(256.0, 0.0)));
        assert_eq!(expected.get(&(0, 1)), Some(&(0.0, 256.0)));
        assert_eq!(expected.get(&(1, 1)), Some(&(256.0, 256.0)));
    }

    #[test]
    fn a_viewport_straddling_the_antimeridian_is_filled_on_both_sides() {
        // Centred exactly on 180°E at zoom 2: half the screen is the eastern
        // edge of the world (column 3), half is the western edge (column 0).
        let Ok(view) = MapView::new(LonLat::new(180.0, 0.0), 2.0, [512.0, 512.0]) else {
            panic!("view construction failed");
        };
        let mut placements = Vec::new();
        view.visible_placements_into(2, &mut placements);
        assert_eq!(placements.len(), 4, "two columns x two rows");

        let mut columns: Vec<u32> = placements.iter().map(|p| p.tile.x).collect();
        columns.sort_unstable();
        columns.dedup();
        assert_eq!(columns, vec![0, 3], "both sides of the cut are requested");

        // The wrapped column is placed in the *eastern* copy of the world, not
        // back at the far west where its unwrapped index would put it.
        for placement in &placements {
            let expected = if placement.tile.x == 3 { 0.0 } else { 256.0 };
            assert!(
                (placement.x - expected).abs() < 1e-3,
                "{placement:?} should sit at x={expected}"
            );
            assert!((placement.size - 256.0).abs() < 1e-3);
        }

        // The old, clamped list is deliberately untouched: it answers "which
        // tiles", not "which copies", and every other caller depends on that.
        let clamped = view.visible_placements();
        assert!(
            clamped.iter().all(|p| p.tile.x == 3),
            "visible_placements must stay in one world copy"
        );
    }

    #[test]
    fn a_viewport_wider_than_the_world_repeats_it() {
        // Zoom 0 on a 1024 px surface: the world is 256 px wide, so it repeats
        // five times to cover the surface and its two partly visible edges.
        let view = view(0.0, [1024.0, 256.0]);
        let mut placements = Vec::new();
        view.visible_placements_into(0, &mut placements);
        assert_eq!(placements.len(), 5);

        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        assert!(placements.iter().all(|p| p.tile == root));
        let mut xs: Vec<f32> = placements.iter().map(|p| p.x).collect();
        xs.sort_by(f32::total_cmp);
        assert_eq!(xs, vec![-128.0, 128.0, 384.0, 640.0, 896.0]);

        // Contiguous: every copy starts exactly where the previous one ends.
        for pair in xs.windows(2) {
            assert!((pair[1] - pair[0] - 256.0).abs() < 1e-3);
        }
    }

    #[test]
    fn an_explicit_zoom_magnifies_instead_of_showing_nothing() {
        // Camera at zoom 5, source only has zoom 2: the frame asks for z2 and
        // gets tiles eight times their native size rather than an empty list.
        let view = view(5.0, [1024.0, 1024.0]);
        let mut placements = Vec::new();
        view.visible_placements_into(2, &mut placements);
        assert!(!placements.is_empty());
        assert!(placements.iter().all(|p| p.tile.z == 2));
        for placement in &placements {
            assert!(
                (placement.size - 2048.0).abs() < 1e-2,
                "a z2 tile at zoom 5 is 256 * 2^3 px, got {}",
                placement.size
            );
        }

        // A zoom past the grid is clamped rather than shifting the world.
        let mut beyond = Vec::new();
        view.visible_placements_into(MAX_ZOOM + 4, &mut beyond);
        assert!(beyond.iter().all(|p| p.tile.z == MAX_ZOOM));
    }

    #[test]
    fn placements_are_refilled_center_outward_into_a_reused_buffer() {
        let Ok(view) = MapView::new(LonLat::new(2.0, 41.0), 6.0, [1024.0, 768.0]) else {
            panic!("view construction failed");
        };
        let Ok(root) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        let mut placements = vec![TilePlacement {
            tile: root,
            x: 1.0,
            y: 2.0,
            size: 3.0,
        }];
        view.visible_placements_into(6, &mut placements);
        assert!(placements.len() > 4, "expected a multi-tile viewport");
        assert!(
            placements.iter().all(|p| p.tile.z == 6),
            "the stale entry must be gone, not appended to"
        );

        let center = [view.size_px()[0] / 2.0, view.size_px()[1] / 2.0];
        let distance = |placement: &TilePlacement| {
            let half = placement.size / 2.0;
            let dx = placement.x + half - center[0];
            let dy = placement.y + half - center[1];
            dx * dx + dy * dy
        };
        for pair in placements.windows(2) {
            assert!(
                distance(&pair[0]) <= distance(&pair[1]),
                "placements must be ordered centre-outward"
            );
        }
        assert!(placements.len() <= MAX_VISIBLE_TILES);

        // Deterministic: the same view refills to exactly the same list.
        let mut again = Vec::new();
        view.visible_placements_into(6, &mut again);
        assert_eq!(again, placements);
    }

    #[test]
    fn a_world_copy_offset_survives_deep_zoom() {
        // At zoom 20 one world is 268 million pixels wide — far past what an
        // f32 can offset without losing whole tiles, which is why the copy is
        // folded in before the pixel scale rather than after.
        let Ok(view) = MapView::new(LonLat::new(180.0, 0.0), 20.0, [512.0, 512.0]) else {
            panic!("view construction failed");
        };
        let z = view.tile_zoom();
        let last = TileId::tiles_per_axis(z) - 1;
        let Ok(west) = TileId::new(z, last, TileId::tiles_per_axis(z) / 2) else {
            panic!("the westmost tile is valid");
        };
        let Ok(east) = TileId::new(z, 0, TileId::tiles_per_axis(z) / 2) else {
            panic!("the eastmost tile is valid");
        };
        let west_edge = view.place_tile(west);
        let east_edge = view.place_tile_in_copy(east, 1);
        // The two are neighbours across the cut: one tile width apart.
        assert!(
            (east_edge.x - west_edge.x - west_edge.size).abs() < 0.5,
            "west {west_edge:?} and east {east_edge:?} must abut"
        );
        assert!((west_edge.x - 0.0).abs() < 0.5, "{west_edge:?}");
    }

    #[test]
    fn screen_round_trip() {
        let Ok(view) = MapView::new(LonLat::new(139.7, 35.7), 9.25, [1280.0, 720.0]) else {
            panic!("view construction failed");
        };
        for px in [[0.0_f32, 0.0_f32], [640.0, 360.0], [1279.0, 719.0]] {
            let back = view.lon_lat_to_screen(view.screen_to_lon_lat(px));
            assert!((back[0] - px[0]).abs() < 1e-2, "x round trip: {back:?}");
            assert!((back[1] - px[1]).abs() < 1e-2, "y round trip: {back:?}");
        }
        let center_px = view.lon_lat_to_screen(view.center());
        assert!((center_px[0] - 640.0).abs() < 1e-2);
        assert!((center_px[1] - 360.0).abs() < 1e-2);
    }

    #[test]
    fn ndc_rect_maps_the_surface() {
        let Ok(tile) = TileId::new(0, 0, 0) else {
            panic!("root tile is valid");
        };
        let placement = TilePlacement {
            tile,
            x: 0.0,
            y: 0.0,
            size: 256.0,
        };
        let Ok(rect) = placement.to_ndc_rect([256.0, 256.0]) else {
            panic!("ndc conversion failed");
        };
        assert_eq!(rect, [-1.0, 1.0, 2.0, 2.0]);

        let half = TilePlacement {
            tile,
            x: 128.0,
            y: 64.0,
            size: 128.0,
        };
        let Ok(rect) = half.to_ndc_rect([256.0, 256.0]) else {
            panic!("ndc conversion failed");
        };
        assert_eq!(rect, [0.0, 0.5, 1.0, 1.0]);

        assert!(matches!(
            placement.to_ndc_rect([0.0, 256.0]),
            Err(RenderError::InvalidViewport(_))
        ));
    }

    #[test]
    fn world_bounds_span_the_surface() {
        let view = view(2.0, [1024.0, 512.0]);
        let (north_west, south_east) = view.world_bounds();
        let scale = view.world_pixels();
        assert!(((south_east.x - north_west.x) * scale - 1024.0).abs() < 1e-6);
        assert!(((south_east.y - north_west.y) * scale - 512.0).abs() < 1e-6);
    }
}
