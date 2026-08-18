//! `(z, x, y)` ⇄ PMTiles tile id, along the Hilbert curve.
//!
//! PMTiles addresses a tile with one `u64`:
//!
//! ```text
//! id = base(z) + hilbert_index(z, x, y)
//! base(z) = Σ_{t < z} 4^t = (4^z − 1)/3
//! ```
//!
//! The base term stacks the pyramids so that every zoom occupies a contiguous
//! block of ids; the Hilbert index orders the tiles *within* a zoom so that
//! ids close together are geographically close together, which is what lets a
//! directory's run-length encoding compress a whole ocean into one entry.
//!
//! The curve is the standard `xy2d`/`d2xy` pair with the `rotate` helper. The
//! known answers below were verified against real archives, and the round trip
//! to z15 with them: `z15 29126/12888 → 1 174 217 003`.
//!
//! No `as` truncation and no `unwrap` anywhere in here: every shift is guarded
//! by the zoom bound, every narrowing goes through `try_from`.

use crate::pmtiles::PmtilesError;

/// Highest zoom a PMTiles tile id can express.
///
/// `base(31) + 4^31 − 1 = 6 148 914 691 236 517 204`, comfortably inside
/// `u64`; zoom 32 would need `4^32`, which does not fit.
pub const MAX_TILE_ID_ZOOM: u8 = 31;

/// The largest id [`tile_id_to_zxy`] can decode: the last tile of zoom
/// [`MAX_TILE_ID_ZOOM`].
pub const MAX_TILE_ID: u64 = 6_148_914_691_236_517_204;

/// The first tile id at zoom `z`, i.e. `(4^z − 1)/3`.
///
/// Accepts `z` up to `MAX_TILE_ID_ZOOM + 1`, because `base(32)` is the
/// exclusive upper bound of zoom 31 and is needed to close the range.
///
/// # Errors
///
/// Returns [`PmtilesError::ZoomOutOfRange`] past that.
pub fn zoom_base(z: u8) -> Result<u64, PmtilesError> {
    if z > MAX_TILE_ID_ZOOM + 1 {
        return Err(PmtilesError::ZoomOutOfRange {
            z,
            limit: MAX_TILE_ID_ZOOM,
        });
    }
    // base(z + 1) = 4 * base(z) + 1, which keeps every intermediate inside u64
    // where `(4^z - 1) / 3` computed directly would overflow at z = 32.
    let mut base = 0u64;
    for _ in 0..z {
        base = base
            .checked_mul(4)
            .and_then(|value| value.checked_add(1))
            .ok_or(PmtilesError::ZoomOutOfRange {
                z,
                limit: MAX_TILE_ID_ZOOM,
            })?;
    }
    Ok(base)
}

/// The PMTiles tile id of `(z, x, y)`.
///
/// # Errors
///
/// * [`PmtilesError::ZoomOutOfRange`] if `z` is past [`MAX_TILE_ID_ZOOM`].
/// * [`PmtilesError::BadCoordinate`] if `x` or `y` is not below `2^z`.
pub fn zxy_to_tile_id(z: u8, x: u32, y: u32) -> Result<u64, PmtilesError> {
    if z > MAX_TILE_ID_ZOOM {
        return Err(PmtilesError::ZoomOutOfRange {
            z,
            limit: MAX_TILE_ID_ZOOM,
        });
    }
    let side = 1u64 << u32::from(z);
    if u64::from(x) >= side || u64::from(y) >= side {
        return Err(PmtilesError::BadCoordinate { z, x, y });
    }
    let base = zoom_base(z)?;
    let index = xy_to_hilbert(side, u64::from(x), u64::from(y));
    base.checked_add(index)
        .ok_or(PmtilesError::TileIdOutOfRange { id: index })
}

/// The `(z, x, y)` a PMTiles tile id addresses.
///
/// # Errors
///
/// Returns [`PmtilesError::TileIdOutOfRange`] for an id past the end of zoom
/// [`MAX_TILE_ID_ZOOM`].
pub fn tile_id_to_zxy(id: u64) -> Result<(u8, u32, u32), PmtilesError> {
    if id > MAX_TILE_ID {
        return Err(PmtilesError::TileIdOutOfRange { id });
    }
    let mut z = 0u8;
    let mut base = 0u64;
    loop {
        // 4^z, with 2 * z <= 62 guaranteed by the MAX_TILE_ID guard above.
        let count = 1u64 << (2 * u32::from(z));
        let next = base.saturating_add(count);
        if id < next {
            break;
        }
        base = next;
        z += 1;
        if z > MAX_TILE_ID_ZOOM {
            return Err(PmtilesError::TileIdOutOfRange { id });
        }
    }
    let side = 1u64 << u32::from(z);
    let (x, y) = hilbert_to_xy(side, id - base);
    let (Ok(x), Ok(y)) = (u32::try_from(x), u32::try_from(y)) else {
        return Err(PmtilesError::TileIdOutOfRange { id });
    };
    Ok((z, x, y))
}

/// Hilbert index of `(x, y)` on a `side × side` grid (`side` a power of two).
///
/// The rotation here reflects around **`side`**, not around the current step,
/// which is the one place this differs textually from the PMTiles reference
/// implementation. The two agree bit for bit: `side − 1 − x` and
/// `step − 1 − x` differ by `side − step`, a multiple of `step`, and every
/// later iteration only inspects bits below `step`. Reflecting around `side`
/// is what keeps the intermediates non-negative, so the whole function stays
/// in `u64` instead of needing the reference's signed arithmetic.
fn xy_to_hilbert(side: u64, mut x: u64, mut y: u64) -> u64 {
    let mut distance = 0u64;
    let mut step = side / 2;
    while step > 0 {
        let rx = u64::from((x & step) > 0);
        let ry = u64::from((y & step) > 0);
        // step * step <= 4^30 and the whole sum is at most 4^31 - 1, so this
        // stays inside u64 for every zoom this module accepts.
        distance += step * step * ((3 * rx) ^ ry);
        rotate(side, &mut x, &mut y, rx, ry);
        step /= 2;
    }
    distance
}

/// The `(x, y)` at Hilbert index `distance` on a `side × side` grid.
fn hilbert_to_xy(side: u64, distance: u64) -> (u64, u64) {
    let mut remaining = distance;
    let mut x = 0u64;
    let mut y = 0u64;
    let mut step = 1u64;
    while step < side {
        let rx = 1 & (remaining / 2);
        let ry = 1 & (remaining ^ rx);
        rotate(step, &mut x, &mut y, rx, ry);
        x += step * rx;
        y += step * ry;
        remaining /= 4;
        step *= 2;
    }
    (x, y)
}

/// Rotates/flips a quadrant so the curve stays continuous across it.
///
/// `extent` is the size to reflect around: the full side in
/// [`xy_to_hilbert`], the current step in [`hilbert_to_xy`] (where `x` and `y`
/// are always below it). Both callers keep `x, y < extent`, so the
/// subtractions never wrap; they are written saturating so a future caller
/// that broke that invariant would produce a wrong answer rather than a panic.
fn rotate(extent: u64, x: &mut u64, y: &mut u64, rx: u64, ry: u64) {
    if ry == 0 {
        if rx == 1 {
            *x = extent.saturating_sub(1).saturating_sub(*x);
            *y = extent.saturating_sub(1).saturating_sub(*y);
        }
        std::mem::swap(x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_TILE_ID, MAX_TILE_ID_ZOOM, tile_id_to_zxy, zoom_base, zxy_to_tile_id};
    use crate::pmtiles::PmtilesError;

    #[test]
    fn the_six_known_answers_hold() {
        // Decoded from real archives and confirmed against the v3 spec.
        assert_eq!(zxy_to_tile_id(0, 0, 0), Ok(0));
        assert_eq!(zxy_to_tile_id(1, 0, 0), Ok(1));
        assert_eq!(zxy_to_tile_id(1, 0, 1), Ok(2));
        assert_eq!(zxy_to_tile_id(1, 1, 1), Ok(3));
        assert_eq!(zxy_to_tile_id(1, 1, 0), Ok(4));
        assert_eq!(zxy_to_tile_id(2, 0, 0), Ok(5));
    }

    #[test]
    fn the_z15_spot_value_round_trips() {
        let id = zxy_to_tile_id(15, 29_126, 12_888).expect("a valid z15 address");
        assert_eq!(id, 1_174_217_003);
        assert_eq!(tile_id_to_zxy(id), Ok((15, 29_126, 12_888)));
    }

    #[test]
    fn zoom_bases_are_the_geometric_series() {
        for z in 0u8..=16 {
            let expected = (4u64.pow(u32::from(z)) - 1) / 3;
            assert_eq!(zoom_base(z), Ok(expected), "z {z}");
        }
        assert_eq!(zoom_base(31), Ok(1_537_228_672_809_129_301));
        assert_eq!(zoom_base(32), Ok(MAX_TILE_ID + 1));
    }

    #[test]
    fn every_tile_of_z0_to_z6_round_trips() {
        let mut seen = 0u64;
        for z in 0u8..=6 {
            let side = 1u32 << u32::from(z);
            for x in 0..side {
                for y in 0..side {
                    let id = zxy_to_tile_id(z, x, y).expect("a valid address");
                    assert_eq!(tile_id_to_zxy(id), Ok((z, x, y)), "z{z} {x}/{y}");
                    seen += 1;
                }
            }
        }
        // Σ 4^z for z in 0..=6.
        assert_eq!(seen, 5_461);
    }

    #[test]
    fn ids_within_a_zoom_are_a_permutation_of_that_zooms_block() {
        for z in 0u8..=5 {
            let side = 1u32 << u32::from(z);
            let base = zoom_base(z).expect("a small zoom");
            let mut ids: Vec<u64> = Vec::new();
            for x in 0..side {
                for y in 0..side {
                    ids.push(zxy_to_tile_id(z, x, y).expect("a valid address"));
                }
            }
            ids.sort_unstable();
            let expected: Vec<u64> = (0..u64::from(side) * u64::from(side))
                .map(|i| base + i)
                .collect();
            assert_eq!(ids, expected, "z {z}");
        }
    }

    #[test]
    fn a_zoom_past_thirty_one_is_refused() {
        assert_eq!(
            zxy_to_tile_id(32, 0, 0),
            Err(PmtilesError::ZoomOutOfRange {
                z: 32,
                limit: MAX_TILE_ID_ZOOM
            })
        );
        assert_eq!(
            zxy_to_tile_id(255, 0, 0),
            Err(PmtilesError::ZoomOutOfRange {
                z: 255,
                limit: MAX_TILE_ID_ZOOM
            })
        );
    }

    #[test]
    fn a_coordinate_at_or_past_two_to_the_z_is_refused() {
        assert_eq!(
            zxy_to_tile_id(0, 1, 0),
            Err(PmtilesError::BadCoordinate { z: 0, x: 1, y: 0 })
        );
        assert_eq!(
            zxy_to_tile_id(1, 0, 2),
            Err(PmtilesError::BadCoordinate { z: 1, x: 0, y: 2 })
        );
        assert_eq!(
            zxy_to_tile_id(10, u32::MAX, 0),
            Err(PmtilesError::BadCoordinate {
                z: 10,
                x: u32::MAX,
                y: 0
            })
        );
    }

    #[test]
    fn an_id_past_the_curve_is_refused() {
        assert_eq!(
            tile_id_to_zxy(u64::MAX),
            Err(PmtilesError::TileIdOutOfRange { id: u64::MAX })
        );
        assert_eq!(
            tile_id_to_zxy(MAX_TILE_ID + 1),
            Err(PmtilesError::TileIdOutOfRange {
                id: MAX_TILE_ID + 1
            })
        );
    }

    #[test]
    fn the_last_id_of_zoom_thirty_one_decodes() {
        let (z, x, y) = tile_id_to_zxy(MAX_TILE_ID).expect("the last addressable tile");
        assert_eq!(z, MAX_TILE_ID_ZOOM);
        assert_eq!(zxy_to_tile_id(z, x, y), Ok(MAX_TILE_ID));
    }

    #[test]
    fn zoom_boundaries_are_the_first_id_of_each_zoom() {
        for z in 0u8..=20 {
            let base = zoom_base(z).expect("a representable zoom");
            let (decoded_z, x, y) = tile_id_to_zxy(base).expect("a representable id");
            assert_eq!(decoded_z, z, "z {z}");
            assert_eq!((x, y), (0, 0), "z {z}");
        }
    }
}
