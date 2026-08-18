//! Transverse Mercator forward and inverse projection, in pure floating point.
//!
//! This is what lets [`super::plan`] place a COG that is *not* already in a
//! Mercator-family CRS on the Web Mercator tile grid. Almost every real
//! satellite / aerial product — Sentinel-2 L2A, NAIP, OpenAerialMap, Planet —
//! ships in a WGS 84 UTM zone, and UTM *is* Transverse Mercator with fixed
//! parameters, so one projection covers the whole class.
//!
//! # Why the maths lives here
//!
//! `oxigis-render` deliberately has no `oxigeo` dependency (blueprint §3: the
//! renderer must be reusable standalone), and `oxigeo-proj`'s ellipsoidal
//! kernel is only reachable through `Transformer`/`projections`, which would
//! drag `oxiproj` + `serde_json` + `once_cell` into the render crate to compute
//! two closed-form series. The series below is ~150 lines and has no
//! dependencies at all. See `TODO.md` §5.1 for the (small) upstream notes this
//! comparison produced.
//!
//! # Series
//!
//! Krüger's (1912) series in the third flattening `n`, truncated after the
//! fourth order — the formulation JHS 154 / Karney (2011) use, with the
//! conformal-latitude substitution done in closed form via `asinh`/`atanh`
//! rather than through a meridional-arc quadrature. Compared with the more
//! commonly transcribed Snyder/Redfearn series in `e²` it is both shorter and
//! better behaved a long way off the central meridian, which matters here: a
//! low-zoom map tile can span far more than one 6°-wide UTM zone, and the
//! projection is asked for points all over it.
//!
//! Measured accuracy (see this module's tests): 7 mm against the EPSG
//! Guidance Note 7-2 worked example (which is itself computed with the
//! truncated Redfearn series), exact agreement to the last few ULP with
//! `oxigeo-proj` 0.2.1 on four UTM reference points, and a forward/inverse
//! round trip that closes to < 1e-10° (< 1e-5 m) out to 20° off the central
//! meridian.

/// WGS 84 semi-major axis, in metres (EPSG:7030).
pub const WGS84_SEMI_MAJOR_M: f64 = 6_378_137.0;

/// WGS 84 inverse flattening, `1/f` (EPSG:7030).
pub const WGS84_INVERSE_FLATTENING: f64 = 298.257_223_563;

/// UTM scale factor at the central meridian.
pub const UTM_SCALE_FACTOR: f64 = 0.9996;

/// UTM false easting, in metres.
pub const UTM_FALSE_EASTING_M: f64 = 500_000.0;

/// UTM false northing for southern-hemisphere zones, in metres.
pub const UTM_FALSE_NORTHING_SOUTH_M: f64 = 10_000_000.0;

/// Largest longitude offset from the central meridian this module will project.
///
/// The series stays *numerically* well behaved much further out, but a point
/// 40° off the central meridian still produces a perfectly finite easting, and
/// a finite-but-meaningless easting is exactly how a raster ends up drawn in
/// the wrong place. 20° is over three times the width of a UTM zone, so it
/// keeps every legitimate zone overhang (Sentinel-2 granules routinely run
/// ~100 km past their zone edge) while rejecting the absurd.
pub const TMERC_MAX_LON_OFFSET_DEG: f64 = 20.0;

/// Largest absolute latitude this module will project.
///
/// `atanh(sin φ)` diverges at the poles; UTM is not defined there either.
pub const TMERC_MAX_LAT_DEG: f64 = 89.9;

/// Number of Krüger series terms retained (order 4 in the third flattening).
const TERMS: usize = 4;

/// Wraps a longitude into the canonical `-180..=180`.
fn wrap_lon_deg(lon_deg: f64) -> f64 {
    (lon_deg + 180.0).rem_euclid(360.0) - 180.0
}

/// Signed offset from `central_meridian_deg` to `lon_deg`, wrapped so the
/// antimeridian is not a discontinuity.
///
/// Plain subtraction breaks down when the central meridian sits within
/// [`TMERC_MAX_LON_OFFSET_DEG`] of ±180°: UTM zone 60's central meridian is
/// 177°E, and a point at 179°W is geometrically 4° away, but
/// `-179.0 - 177.0 == -356.0` unwrapped — nowhere near the guard it should
/// pass. Zone 1 (central meridian 177°W) has the mirror problem to the east.
fn wrapped_offset_deg(lon_deg: f64, central_meridian_deg: f64) -> f64 {
    wrap_lon_deg(lon_deg - central_meridian_deg)
}

/// A Transverse Mercator projection with a fixed parameter set.
///
/// Construct with [`TransverseMercator::new`] for an arbitrary ellipsoid and
/// origin, or [`TransverseMercator::wgs84_utm`] for a WGS 84 UTM zone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransverseMercator {
    /// Central meridian, in degrees.
    central_meridian_deg: f64,
    /// Central meridian, in radians.
    central_meridian: f64,
    /// Scale factor at the central meridian.
    scale_factor: f64,
    /// False easting, in projected units.
    false_easting: f64,
    /// False northing, in projected units.
    false_northing: f64,
    /// Rectifying radius `A`.
    rectifying_radius: f64,
    /// Third flattening `n = f / (2 - f)`.
    third_flattening: f64,
    /// `2 √n / (1 + n)`, the conformal-latitude coefficient.
    conformal_coefficient: f64,
    /// Rectifying latitude of the latitude of origin.
    origin_xi: f64,
    /// Forward series coefficients `α₁..α₄`.
    alpha: [f64; TERMS],
    /// Inverse series coefficients `β₁..β₄`.
    beta: [f64; TERMS],
    /// Conformal-to-geodetic latitude coefficients `δ₁..δ₄`.
    delta: [f64; TERMS],
}

impl TransverseMercator {
    /// Builds a projection from the seven EPSG method 9807 parameters.
    ///
    /// `semi_major_m` and `inverse_flattening` describe the ellipsoid;
    /// `inverse_flattening` is `1/f`, as ellipsoid definitions state it. A
    /// sphere is expressed as an infinite inverse flattening.
    ///
    /// Returns `None` for a parameter set the series cannot be built from: a
    /// non-positive or non-finite axis or scale factor, an inverse flattening
    /// below 1 (i.e. `f ≥ 1`), a non-finite false origin, or a latitude of
    /// origin beyond [`TMERC_MAX_LAT_DEG`].
    #[must_use]
    pub fn new(
        semi_major_m: f64,
        inverse_flattening: f64,
        scale_factor: f64,
        latitude_of_origin_deg: f64,
        central_meridian_deg: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Option<Self> {
        if !semi_major_m.is_finite() || semi_major_m <= 0.0 {
            return None;
        }
        if !scale_factor.is_finite() || scale_factor <= 0.0 {
            return None;
        }
        if !central_meridian_deg.is_finite()
            || !false_easting.is_finite()
            || !false_northing.is_finite()
        {
            return None;
        }
        if !latitude_of_origin_deg.is_finite() || latitude_of_origin_deg.abs() > TMERC_MAX_LAT_DEG {
            return None;
        }
        // A sphere is `1/f = ∞`; anything finite must flatten by less than 1.
        let flattening = if inverse_flattening.is_infinite() && inverse_flattening > 0.0 {
            0.0
        } else if inverse_flattening.is_finite() && inverse_flattening >= 1.0 {
            1.0 / inverse_flattening
        } else {
            return None;
        };

        let n = flattening / (2.0 - flattening);
        let (n2, n3) = (n * n, n * n * n);
        let n4 = n2 * n2;
        let rectifying_radius = semi_major_m / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0);
        if !rectifying_radius.is_finite() || rectifying_radius <= 0.0 {
            return None;
        }

        let alpha = [
            n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0 + 41.0 * n4 / 180.0,
            13.0 * n2 / 48.0 - 3.0 * n3 / 5.0 + 557.0 * n4 / 1_440.0,
            61.0 * n3 / 240.0 - 103.0 * n4 / 140.0,
            49_561.0 * n4 / 161_280.0,
        ];
        let beta = [
            n / 2.0 - 2.0 * n2 / 3.0 + 37.0 * n3 / 96.0 - n4 / 360.0,
            n2 / 48.0 + n3 / 15.0 - 437.0 * n4 / 1_440.0,
            17.0 * n3 / 480.0 - 37.0 * n4 / 840.0,
            4_397.0 * n4 / 161_280.0,
        ];
        let delta = [
            2.0 * n - 2.0 * n2 / 3.0 - 2.0 * n3 + 116.0 * n4 / 45.0,
            7.0 * n2 / 3.0 - 8.0 * n3 / 5.0 - 227.0 * n4 / 45.0,
            56.0 * n3 / 15.0 - 136.0 * n4 / 35.0,
            4_279.0 * n4 / 630.0,
        ];

        let mut projection = Self {
            central_meridian_deg,
            central_meridian: central_meridian_deg.to_radians(),
            scale_factor,
            false_easting,
            false_northing,
            rectifying_radius,
            third_flattening: n,
            conformal_coefficient: 2.0 * n.sqrt() / (1.0 + n),
            origin_xi: 0.0,
            alpha,
            beta,
            delta,
        };
        projection.origin_xi = projection.meridional_xi(latitude_of_origin_deg.to_radians());
        if !projection.origin_xi.is_finite() {
            return None;
        }
        Some(projection)
    }

    /// The WGS 84 UTM zone `zone` (`1..=60`) of the given hemisphere.
    ///
    /// These are EPSG:326`zz` (north) and EPSG:327`zz` (south).
    #[must_use]
    pub fn wgs84_utm(zone: u8, north: bool) -> Option<Self> {
        if zone == 0 || zone > 60 {
            return None;
        }
        Self::new(
            WGS84_SEMI_MAJOR_M,
            WGS84_INVERSE_FLATTENING,
            UTM_SCALE_FACTOR,
            0.0,
            utm_central_meridian_deg(zone)?,
            UTM_FALSE_EASTING_M,
            if north {
                0.0
            } else {
                UTM_FALSE_NORTHING_SOUTH_M
            },
        )
    }

    /// The projection's central meridian, in degrees.
    #[must_use]
    pub const fn central_meridian_deg(&self) -> f64 {
        self.central_meridian_deg
    }

    /// Projects WGS 84 degrees to projected units, or `None` when the point is
    /// outside the range this module is willing to project — see
    /// [`TMERC_MAX_LON_OFFSET_DEG`] and [`TMERC_MAX_LAT_DEG`].
    #[must_use]
    pub fn forward(&self, lon_deg: f64, lat_deg: f64) -> Option<(f64, f64)> {
        if !lon_deg.is_finite() || !lat_deg.is_finite() {
            return None;
        }
        if lat_deg.abs() > TMERC_MAX_LAT_DEG {
            return None;
        }
        if wrapped_offset_deg(lon_deg, self.central_meridian_deg).abs() > TMERC_MAX_LON_OFFSET_DEG {
            return None;
        }
        let offset = lon_deg.to_radians() - self.central_meridian;
        let tau = self.conformal_tau(lat_deg.to_radians());
        // Gauss–Schreiber (ξ′, η′): the projection of the point onto the
        // conformal sphere, before the series maps it onto the ellipsoid's TM.
        let xi_prime = tau.atan2(offset.cos());
        let eta_prime = (offset.sin() / (1.0 + tau * tau).sqrt()).atanh();
        let mut xi = xi_prime;
        let mut eta = eta_prime;
        for (index, coefficient) in self.alpha.iter().enumerate() {
            #[expect(
                clippy::cast_precision_loss,
                reason = "index is 0..4, exactly representable"
            )]
            let harmonic = 2.0 * (index as f64 + 1.0);
            xi += coefficient * (harmonic * xi_prime).sin() * (harmonic * eta_prime).cosh();
            eta += coefficient * (harmonic * xi_prime).cos() * (harmonic * eta_prime).sinh();
        }
        let easting = self.false_easting + self.scale_factor * self.rectifying_radius * eta;
        let northing = self.false_northing
            + self.scale_factor * self.rectifying_radius * (xi - self.origin_xi);
        if easting.is_finite() && northing.is_finite() {
            Some((easting, northing))
        } else {
            None
        }
    }

    /// Projects the outer product of `longitudes` × `latitudes`, calling
    /// `visit(column, row, easting, northing)` for each pair the projection
    /// accepts and skipping the rest.
    ///
    /// Equivalent to calling [`TransverseMercator::forward`] once per pair —
    /// same acceptances, same refusals, agreeing values — but roughly four
    /// times faster, which matters because [`super::plan`] projects 65 536
    /// points per map tile, twice. Three things move out of the inner loop:
    ///
    /// * the conformal latitude, which depends only on the latitude (per row);
    /// * `sin`/`cos` of the longitude offset, which depend only on the
    ///   longitude (per column);
    /// * the four series harmonics `sin/cos(2jξ′)` and `sinh/cosh(2jη′)`, which
    ///   are stepped from the `j = 1` pair by the angle-sum recurrences instead
    ///   of being evaluated term by term. That alone removes twelve
    ///   transcendental calls per point.
    ///
    /// Values may differ from [`TransverseMercator::forward`] in the last few
    /// ULP because the recurrences accumulate differently; the tests hold the
    /// two to within a micrometre.
    pub fn forward_grid(
        &self,
        longitudes: &[f64],
        latitudes: &[f64],
        mut visit: impl FnMut(usize, usize, f64, f64),
    ) {
        // Per column: the longitude offset's sine and cosine, or `None` for a
        // longitude `forward` would refuse.
        let columns: Vec<Option<(f64, f64)>> = longitudes
            .iter()
            .map(|lon_deg| {
                if !lon_deg.is_finite()
                    || wrapped_offset_deg(*lon_deg, self.central_meridian_deg).abs()
                        > TMERC_MAX_LON_OFFSET_DEG
                {
                    return None;
                }
                let offset = lon_deg.to_radians() - self.central_meridian;
                Some((offset.sin(), offset.cos()))
            })
            .collect();

        let scale = self.scale_factor * self.rectifying_radius;
        for (row, lat_deg) in latitudes.iter().enumerate() {
            if !lat_deg.is_finite() || lat_deg.abs() > TMERC_MAX_LAT_DEG {
                continue;
            }
            let tau = self.conformal_tau(lat_deg.to_radians());
            let hypot = (1.0 + tau * tau).sqrt();
            if !tau.is_finite() || !hypot.is_finite() {
                continue;
            }
            for (column, offset) in columns.iter().enumerate() {
                let Some((sine, cosine)) = *offset else {
                    continue;
                };
                let xi_prime = tau.atan2(cosine);
                let eta_prime = (sine / hypot).atanh();
                // Seeds for the `j = 1` harmonic; every further term follows by
                // the angle-sum recurrences below. `|η′| ≤ atanh(sin 20°)`
                // inside the accepted range, so `sinh`/`cosh` stay far from
                // overflow.
                let (mut sin_j, mut cos_j) = (2.0 * xi_prime).sin_cos();
                let mut sinh_j = (2.0 * eta_prime).sinh();
                let mut cosh_j = (2.0 * eta_prime).cosh();
                let (sin_1, cos_1, sinh_1, cosh_1) = (sin_j, cos_j, sinh_j, cosh_j);
                let mut xi = xi_prime;
                let mut eta = eta_prime;
                for coefficient in &self.alpha {
                    xi += coefficient * sin_j * cosh_j;
                    eta += coefficient * cos_j * sinh_j;
                    let next_sin = sin_j * cos_1 + cos_j * sin_1;
                    let next_cos = cos_j * cos_1 - sin_j * sin_1;
                    let next_sinh = sinh_j * cosh_1 + cosh_j * sinh_1;
                    let next_cosh = cosh_j * cosh_1 + sinh_j * sinh_1;
                    sin_j = next_sin;
                    cos_j = next_cos;
                    sinh_j = next_sinh;
                    cosh_j = next_cosh;
                }
                let easting = self.false_easting + scale * eta;
                let northing = self.false_northing + scale * (xi - self.origin_xi);
                if easting.is_finite() && northing.is_finite() {
                    visit(column, row, easting, northing);
                }
            }
        }
    }

    /// Unprojects projected units to WGS 84 degrees, or `None` when the result
    /// falls outside the range this module is willing to project — the same
    /// bounds [`TransverseMercator::forward`] enforces, so the pair round-trips.
    #[must_use]
    pub fn inverse(&self, easting: f64, northing: f64) -> Option<(f64, f64)> {
        if !easting.is_finite() || !northing.is_finite() {
            return None;
        }
        let scale = self.scale_factor * self.rectifying_radius;
        let xi = (northing - self.false_northing) / scale + self.origin_xi;
        let eta = (easting - self.false_easting) / scale;
        let mut xi_prime = xi;
        let mut eta_prime = eta;
        for (index, coefficient) in self.beta.iter().enumerate() {
            #[expect(
                clippy::cast_precision_loss,
                reason = "index is 0..4, exactly representable"
            )]
            let harmonic = 2.0 * (index as f64 + 1.0);
            xi_prime -= coefficient * (harmonic * xi).sin() * (harmonic * eta).cosh();
            eta_prime -= coefficient * (harmonic * xi).cos() * (harmonic * eta).sinh();
        }
        // Conformal latitude on the Gauss sphere, then the δ series back to
        // geodetic latitude.
        let sine = xi_prime.sin() / eta_prime.cosh();
        if !(-1.0..=1.0).contains(&sine) {
            return None;
        }
        let conformal = sine.asin();
        let mut lat = conformal;
        for (index, coefficient) in self.delta.iter().enumerate() {
            #[expect(
                clippy::cast_precision_loss,
                reason = "index is 0..4, exactly representable"
            )]
            let harmonic = 2.0 * (index as f64 + 1.0);
            lat += coefficient * (harmonic * conformal).sin();
        }
        let lon = self.central_meridian + eta_prime.sinh().atan2(xi_prime.cos());
        let (lon_deg, lat_deg) = (lon.to_degrees(), lat.to_degrees());
        if !lon_deg.is_finite() || !lat_deg.is_finite() {
            return None;
        }
        if lat_deg.abs() > TMERC_MAX_LAT_DEG
            || wrapped_offset_deg(lon_deg, self.central_meridian_deg).abs()
                > TMERC_MAX_LON_OFFSET_DEG
        {
            return None;
        }
        // The guard above is evaluated on the raw (possibly > 180° or <
        // -180°) `lon_deg` — wrapping is invariant under adding a multiple of
        // 360°, so it accepts exactly the same points either way — but the
        // *returned* longitude must be canonical: a zone-60 point east of the
        // antimeridian computes `lon_deg` past 180° here, and every caller in
        // `super::plan` assumes WGS 84 degrees, not that.
        Some((wrap_lon_deg(lon_deg), lat_deg))
    }

    /// `tan` of the conformal latitude, in closed form.
    ///
    /// `sinh(atanh(sin φ) − c · atanh(c · sin φ))` with `c = 2√n / (1 + n)`;
    /// this replaces the isometric-latitude quadrature the `e²` series needs.
    fn conformal_tau(&self, lat: f64) -> f64 {
        let sine = lat.sin();
        let coefficient = self.conformal_coefficient;
        (sine.atanh() - coefficient * (coefficient * sine).atanh()).sinh()
    }

    /// Rectifying latitude ξ on the central meridian, i.e. the meridional arc
    /// from the equator to `lat` divided by the rectifying radius.
    fn meridional_xi(&self, lat: f64) -> f64 {
        if self.third_flattening == 0.0 {
            // A sphere: the conformal latitude *is* the latitude.
            return lat;
        }
        let tau = self.conformal_tau(lat);
        let xi_prime = tau.atan();
        let mut xi = xi_prime;
        for (index, coefficient) in self.alpha.iter().enumerate() {
            #[expect(
                clippy::cast_precision_loss,
                reason = "index is 0..4, exactly representable"
            )]
            let harmonic = 2.0 * (index as f64 + 1.0);
            xi += coefficient * (harmonic * xi_prime).sin();
        }
        xi
    }
}

/// Central meridian of UTM zone `zone` (`1..=60`), in degrees.
///
/// Zone 1 is centred on 177°W and each following zone is 6° further east.
#[must_use]
pub fn utm_central_meridian_deg(zone: u8) -> Option<f64> {
    if zone == 0 || zone > 60 {
        return None;
    }
    Some(6.0 * f64::from(zone) - 183.0)
}

#[cfg(test)]
mod tests {
    use super::{
        TMERC_MAX_LAT_DEG, TMERC_MAX_LON_OFFSET_DEG, TransverseMercator, utm_central_meridian_deg,
    };

    /// The EPSG Guidance Note 7-2 worked example for method 9807
    /// (Transverse Mercator): OSGB36 / British National Grid, EPSG:27700.
    ///
    /// Airy 1830 (a = 6377563.396 m, b = 6356256.909 m), k₀ = 0.9996012717,
    /// origin 49°N 2°W, false origin (400000, −100000). The published result
    /// for 50°30′N 0°30′E is E = 577274.99 m, N = 69740.50 m.
    ///
    /// This is the load-bearing accuracy test: it is an independent published
    /// value on a *different* ellipsoid, so it validates the series itself
    /// rather than a transcription. The residual is ~7 mm because the EPSG
    /// example is computed with the truncated Redfearn series in `e²`, which
    /// is the less accurate of the two — hence a 2 cm tolerance rather than a
    /// millimetre one.
    fn british_national_grid() -> TransverseMercator {
        let semi_major = 6_377_563.396_f64;
        let semi_minor = 6_356_256.909_f64;
        let inverse_flattening = semi_major / (semi_major - semi_minor);
        TransverseMercator::new(
            semi_major,
            inverse_flattening,
            0.999_601_271_7,
            49.0,
            -2.0,
            400_000.0,
            -100_000.0,
        )
        .unwrap()
    }

    #[test]
    fn epsg_guidance_note_worked_example_matches() {
        let projection = british_national_grid();
        let (easting, northing) = projection.forward(0.5, 50.5).unwrap();
        assert!(
            (easting - 577_274.99).abs() < 2e-2,
            "easting {easting} must match the published 577274.99"
        );
        assert!(
            (northing - 69_740.50).abs() < 2e-2,
            "northing {northing} must match the published 69740.50"
        );
        // Unprojecting the *published* (centimetre-rounded) grid reference can
        // only be as good as its inputs: 1 cm of easting is ~1.4e-7°.
        let (lon, lat) = projection.inverse(577_274.99, 69_740.50).unwrap();
        assert!((lon - 0.5).abs() < 1e-6, "longitude {lon}");
        assert!((lat - 50.5).abs() < 1e-6, "latitude {lat}");
    }

    #[test]
    fn british_national_grid_false_origin_is_exact() {
        let projection = british_national_grid();
        let (easting, northing) = projection.forward(-2.0, 49.0).unwrap();
        assert!((easting - 400_000.0).abs() < 1e-9, "easting {easting}");
        assert!((northing + 100_000.0).abs() < 1e-6, "northing {northing}");
    }

    /// Reference eastings/northings captured from `oxigeo-proj` 0.2.1
    /// (`Transformer::from_epsg(4326, …)`), which uses the independent
    /// Snyder 6-term series in `e²`. Agreement to well under a millimetre
    /// across four zones and both hemispheres pins this implementation
    /// against a second one; it is a regression pin, not a validation (see
    /// `epsg_guidance_note_worked_example_matches` for that).
    #[test]
    fn utm_reference_points_agree_with_oxigeo_proj() {
        let cases: &[(&str, f64, f64, u8, bool, f64, f64)] = &[
            (
                "zone 33N central meridian",
                15.0,
                45.0,
                33,
                true,
                500_000.0,
                4_982_950.400_226_552,
            ),
            (
                "Tokyo, zone 54N",
                139.767_1,
                35.681_2,
                54,
                true,
                388_433.374_621_020_75,
                3_949_290.013_536_497,
            ),
            (
                "San Francisco, zone 10N",
                -122.419_4,
                37.774_9,
                10,
                true,
                551_130.768_481_284_1,
                4_180_998.881_499_062_3,
            ),
            (
                "Cape Town, zone 34S",
                18.424_1,
                -33.924_9,
                34,
                false,
                261_881.598_523_994_67,
                6_243_182.354_517_815,
            ),
        ];
        for &(name, lon, lat, zone, north, expect_e, expect_n) in cases {
            let projection = TransverseMercator::wgs84_utm(zone, north).unwrap();
            let (easting, northing) = projection.forward(lon, lat).unwrap();
            assert!(
                (easting - expect_e).abs() < 1e-3,
                "{name}: easting {easting} vs {expect_e}"
            );
            assert!(
                (northing - expect_n).abs() < 1e-3,
                "{name}: northing {northing} vs {expect_n}"
            );
            let (back_lon, back_lat) = projection.inverse(easting, northing).unwrap();
            assert!(
                (back_lon - lon).abs() < 1e-10,
                "{name}: longitude round trip {back_lon} vs {lon}"
            );
            assert!(
                (back_lat - lat).abs() < 1e-10,
                "{name}: latitude round trip {back_lat} vs {lat}"
            );
        }
    }

    #[test]
    fn the_round_trip_closes_across_a_whole_zone_and_beyond() {
        let projection = TransverseMercator::wgs84_utm(33, true).unwrap();
        let mut worst_lon = 0.0_f64;
        let mut worst_lat = 0.0_f64;
        let central = projection.central_meridian_deg();
        // Strictly *inside* the offset limit: the round trip drifts by ~1e-12°,
        // so probing exactly at the limit would have the inverse's own guard
        // reject the point it was just given.
        let limit = TMERC_MAX_LON_OFFSET_DEG - 1.0;
        let mut latitude = -80.0_f64;
        while latitude <= 84.0 {
            let mut offset = -limit;
            while offset <= limit {
                let (easting, northing) = projection.forward(central + offset, latitude).unwrap();
                let (lon, lat) = projection.inverse(easting, northing).unwrap();
                worst_lon = worst_lon.max((lon - (central + offset)).abs());
                worst_lat = worst_lat.max((lat - latitude).abs());
                offset += 2.5;
            }
            latitude += 4.0;
        }
        assert!(worst_lon < 1e-10, "worst longitude drift {worst_lon}°");
        assert!(worst_lat < 1e-10, "worst latitude drift {worst_lat}°");
    }

    #[test]
    fn southern_zones_carry_the_false_northing() {
        let north = TransverseMercator::wgs84_utm(34, true).unwrap();
        let south = TransverseMercator::wgs84_utm(34, false).unwrap();
        let (north_e, north_n) = north.forward(21.0, -33.9).unwrap();
        let (south_e, south_n) = south.forward(21.0, -33.9).unwrap();
        assert!((north_e - south_e).abs() < 1e-9);
        assert!((south_n - north_n - 10_000_000.0).abs() < 1e-6);
        assert!(south_n > 0.0, "a southern-zone northing must be positive");
    }

    #[test]
    fn central_meridians_follow_the_zone_numbering() {
        assert_eq!(utm_central_meridian_deg(1), Some(-177.0));
        assert_eq!(utm_central_meridian_deg(31), Some(3.0));
        assert_eq!(utm_central_meridian_deg(54), Some(141.0));
        assert_eq!(utm_central_meridian_deg(60), Some(177.0));
        assert_eq!(utm_central_meridian_deg(0), None);
        assert_eq!(utm_central_meridian_deg(61), None);
        assert!(TransverseMercator::wgs84_utm(0, true).is_none());
        assert!(TransverseMercator::wgs84_utm(61, true).is_none());
    }

    /// UTM zones 1 and 60 sit astride the antimeridian, so their ±20°
    /// overhang wraps through it — a plain `lon - central_meridian`
    /// subtraction breaks down there (see `wrapped_offset_deg`).
    #[test]
    fn zone_60_accepts_points_across_the_antimeridian_and_zone_1_mirrors_it() {
        // Zone 60N's central meridian is 177°E; the 20° overhang reaches
        // 157°E one way and, wrapping through 180°, 163°W the other.
        let zone60 = TransverseMercator::wgs84_utm(60, true).unwrap();
        assert!(zone60.forward(179.0, 10.0).is_some(), "2° east of 177°E");
        assert!(
            zone60.forward(-179.0, 10.0).is_some(),
            "4° east of 177°E via the antimeridian"
        );
        assert!(
            zone60.forward(-163.0, 10.0).is_some(),
            "exactly the 20° limit, wrapped"
        );
        assert!(
            zone60.forward(-162.0, 10.0).is_none(),
            "21° away, wrapped: outside the limit"
        );
        assert!(
            zone60.forward(150.0, 10.0).is_none(),
            "27° away the short way: outside the limit"
        );

        // Zone 1N's central meridian is 177°W: the mirror image.
        let zone1 = TransverseMercator::wgs84_utm(1, true).unwrap();
        assert!(zone1.forward(-179.0, 10.0).is_some(), "2° west of 177°W");
        assert!(
            zone1.forward(179.0, 10.0).is_some(),
            "4° west of 177°W via the antimeridian"
        );
        assert!(
            zone1.forward(163.0, 10.0).is_some(),
            "exactly the 20° limit, wrapped"
        );
        assert!(
            zone1.forward(162.0, 10.0).is_none(),
            "21° away, wrapped: outside the limit"
        );

        // `forward_grid` must accept and refuse exactly what `forward` does —
        // that is the guard this fix touches, in the batched path too.
        let longitudes = [179.0, -179.0, -163.0, -162.0, 150.0];
        let latitudes = [10.0];
        let mut grid_hits = [false; 5];
        zone60.forward_grid(&longitudes, &latitudes, |column, _row, _e, _n| {
            grid_hits[column] = true;
        });
        for (index, lon) in longitudes.iter().enumerate() {
            assert_eq!(
                grid_hits[index],
                zone60.forward(*lon, 10.0).is_some(),
                "forward_grid disagreed with forward at {lon}°"
            );
        }
    }

    /// `inverse` must return a canonical `-180..=180` longitude even when the
    /// point sits east of the antimeridian in the projection's own (raw,
    /// unwrapped) coordinate frame — `super::plan` feeds this straight into
    /// an affine `(lon + 180) / 360`, which silently produces `world_x > 1`
    /// for anything else.
    #[test]
    fn zone_60_inverse_returns_a_canonical_longitude_across_the_antimeridian() {
        let projection = TransverseMercator::wgs84_utm(60, true).unwrap();
        let (easting, northing) = projection.forward(-179.0, 10.0).unwrap();
        let (lon, lat) = projection.inverse(easting, northing).unwrap();
        assert!(
            (-180.0..=180.0).contains(&lon),
            "lon {lon} must be canonical"
        );
        assert!(
            (lon - (-179.0)).abs() < 1e-9,
            "lon {lon} must round-trip to -179°, not an unwrapped 181°"
        );
        assert!((lat - 10.0).abs() < 1e-9);
    }

    #[test]
    fn points_far_from_the_central_meridian_are_refused() {
        let projection = TransverseMercator::wgs84_utm(33, true).unwrap();
        let central = projection.central_meridian_deg();
        assert!(projection.forward(central + 19.0, 45.0).is_some());
        assert!(projection.forward(central + 21.0, 45.0).is_none());
        assert!(projection.forward(central - 21.0, 45.0).is_none());
        assert!(
            projection
                .forward(central, TMERC_MAX_LAT_DEG + 0.1)
                .is_none()
        );
        assert!(projection.forward(f64::NAN, 45.0).is_none());
        assert!(projection.forward(central, f64::INFINITY).is_none());
        // An easting 4000 km off the central meridian is >20° away at this
        // latitude, so the inverse refuses it too rather than inventing a
        // longitude.
        assert!(projection.inverse(4_500_000.0, 4_982_950.0).is_none());
        assert!(projection.inverse(f64::NAN, 0.0).is_none());
    }

    /// `forward_grid` is an optimisation of `forward`, so it has to be
    /// indistinguishable from it — including in what it *refuses*. A recurrence
    /// path that quietly projected a point `forward` rejects would be exactly
    /// how out-of-zone garbage gets back into the raster.
    ///
    /// The grid deliberately straddles both guards: longitudes reaching past
    /// ±[`TMERC_MAX_LON_OFFSET_DEG`] and latitudes past ±[`TMERC_MAX_LAT_DEG`],
    /// plus non-finite values, and it includes the equator at maximum offset,
    /// where `η′` — and so the `sinh`/`cosh` seeds — are largest.
    #[test]
    fn the_batched_forward_matches_the_scalar_one_point_for_point() {
        let projection = TransverseMercator::wgs84_utm(33, true).unwrap();
        let central = projection.central_meridian_deg();
        let mut longitudes = Vec::new();
        let mut offset = -TMERC_MAX_LON_OFFSET_DEG - 3.0;
        while offset <= TMERC_MAX_LON_OFFSET_DEG + 3.0 {
            longitudes.push(central + offset);
            offset += 0.7;
        }
        longitudes.push(f64::NAN);
        longitudes.push(f64::INFINITY);
        let mut latitudes = Vec::new();
        let mut latitude = -TMERC_MAX_LAT_DEG - 3.0;
        while latitude <= TMERC_MAX_LAT_DEG + 3.0 {
            latitudes.push(latitude);
            latitude += 1.3;
        }
        // The equator: maximum |η′|, and the row where `tan φ` is zero.
        latitudes.push(0.0);
        latitudes.push(f64::NAN);

        let mut batched = vec![None; longitudes.len() * latitudes.len()];
        projection.forward_grid(&longitudes, &latitudes, |column, row, easting, northing| {
            batched[row * longitudes.len() + column] = Some((easting, northing));
        });

        let mut compared = 0usize;
        let mut refused = 0usize;
        let mut worst = 0.0_f64;
        for (row, latitude) in latitudes.iter().enumerate() {
            for (column, longitude) in longitudes.iter().enumerate() {
                let scalar = projection.forward(*longitude, *latitude);
                let grid = batched[row * longitudes.len() + column];
                match (scalar, grid) {
                    (Some((expect_e, expect_n)), Some((easting, northing))) => {
                        worst = worst
                            .max((easting - expect_e).abs())
                            .max((northing - expect_n).abs());
                        compared += 1;
                    }
                    (None, None) => refused += 1,
                    (scalar, grid) => panic!(
                        "({longitude}, {latitude}): forward gave {scalar:?} but forward_grid gave \
                         {grid:?}"
                    ),
                }
            }
        }
        assert!(compared > 1_000, "only {compared} points were comparable");
        assert!(refused > 100, "only {refused} points exercised the guards");
        assert!(worst < 1e-6, "worst disagreement {worst} m");
    }

    #[test]
    fn a_sphere_is_accepted_and_degenerate_parameters_are_not() {
        let sphere =
            TransverseMercator::new(6_378_137.0, f64::INFINITY, 1.0, 0.0, 0.0, 0.0, 0.0).unwrap();
        // On a sphere with k₀ = 1 the central meridian is a great circle
        // scaled exactly by the radius.
        let (easting, northing) = sphere.forward(0.0, 45.0).unwrap();
        assert!(easting.abs() < 1e-9, "easting {easting}");
        let expected = 6_378_137.0 * 45.0_f64.to_radians();
        assert!((northing - expected).abs() < 1e-6, "northing {northing}");

        assert!(TransverseMercator::new(0.0, 298.0, 1.0, 0.0, 0.0, 0.0, 0.0).is_none());
        assert!(TransverseMercator::new(6e6, 298.0, 0.0, 0.0, 0.0, 0.0, 0.0).is_none());
        assert!(TransverseMercator::new(6e6, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0).is_none());
        assert!(TransverseMercator::new(6e6, f64::NAN, 1.0, 0.0, 0.0, 0.0, 0.0).is_none());
        assert!(TransverseMercator::new(6e6, 298.0, 1.0, 95.0, 0.0, 0.0, 0.0).is_none());
        assert!(TransverseMercator::new(6e6, 298.0, 1.0, 0.0, f64::NAN, 0.0, 0.0).is_none());
    }
}
