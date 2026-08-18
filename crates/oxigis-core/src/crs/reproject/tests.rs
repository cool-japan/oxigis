// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for the reprojection seam.
//!
//! Every numeric assertion here is anchored on something **outside** this
//! crate — a published worked example, a published national datum-shift
//! approximation, or a geodetic quantity with a textbook value — rather than
//! on what the code happens to produce. A table of projection parameters that
//! agrees with itself passes a round-trip test perfectly while placing data in
//! the wrong prefecture; the anchors below are what actually rule that out.

use super::*;

/// Degrees of latitude that are one metre, near 36°N. Used to state tolerances
/// in metres, which is the unit the accuracy claims are made in.
const DEG_PER_METRE_LAT: f64 = 1.0 / 111_000.0;

/// Asserts two lon/lat pairs are within `metres` of each other.
#[track_caller]
fn assert_near(actual: (f64, f64), expected: (f64, f64), metres: f64, what: &str) {
    let tolerance = metres * DEG_PER_METRE_LAT;
    let (dlon, dlat) = (actual.0 - expected.0, actual.1 - expected.1);
    assert!(
        dlon.abs() <= tolerance && dlat.abs() <= tolerance,
        "{what}: got {actual:?}, expected {expected:?} (±{metres} m = ±{tolerance} deg); \
         off by ({dlon}, {dlat}) deg",
    );
}

// ---------------------------------------------------------------------------
// The projection mathematics, anchored outside this crate
// ---------------------------------------------------------------------------

#[test]
fn the_transverse_mercator_reproduces_the_epsg_guidance_note_worked_example() {
    // EPSG Guidance Note 7-2, §3.5.4.1, the standard worked example for
    // Transverse Mercator (method 9807): OSGB 1936 / British National Grid,
    // Airy 1830, lat_0 = 49°N, lon_0 = 2°W, k0 = 0.9996012717,
    // FE = 400 000 m, FN = -100 000 m.
    //
    //   latitude 50°30'00"N, longitude 00°30'00"E
    //     -> Easting 577 274.99 m, Northing 69 740.50 m
    //
    // Exercised on the projection alone (no datum step), because the example
    // is stated in OSGB36 geodetic coordinates. This is what proves the
    // series is ELLIPSOIDAL: a spherical Transverse Mercator — which is what
    // `oxigeo_proj::transform::cylindrical::TransverseMercator` is, and why
    // this module uses `GaussKruger` instead — misses this by kilometres.
    let bng = TmercParams {
        latitude_of_origin_deg: 49.0,
        central_meridian_deg: -2.0,
        scale_factor: 0.999_601_271_7,
        false_easting: 400_000.0,
        false_northing: -100_000.0,
        semi_major_axis_m: 6_377_563.396,
        flattening: 1.0 / 299.324_96,
    };
    let projection = bng.to_gauss_kruger();

    let (easting, northing) = projection.forward(0.5, 50.5).expect("forward");
    assert!(
        (easting - 577_274.99).abs() < 0.05,
        "easting {easting}, published 577274.99",
    );
    assert!(
        (northing - 69_740.50).abs() < 0.05,
        "northing {northing}, published 69740.50",
    );

    let (lon, lat) = projection.inverse(577_274.99, 69_740.50).expect("inverse");
    assert!((lon - 0.5).abs() < 1e-6, "lon {lon}");
    assert!((lat - 50.5).abs() < 1e-6, "lat {lat}");
}

#[test]
fn a_degree_of_latitude_on_the_central_meridian_has_its_textbook_length() {
    // On the central meridian the easting is exactly zero and the northing is
    // k0 times the meridian arc from the latitude of origin. One degree of
    // latitude at 35–36°N on GRS 80 is 110.941 km (standard geodetic tables);
    // scaled by JGD2011 zone IX's k0 = 0.9999 that is 110 930 m.
    //
    // This is what validates the meridian-arc series and the scale factor
    // against a quantity nothing in this crate computes.
    let zone9 = Reprojector::for_crs(&Crs::from_epsg(6677)).expect("EPSG:6677");
    let Operation::TransverseMercator(params, None) = zone9.operation else {
        panic!("zone IX is a Transverse Mercator on a WGS 84-aligned datum");
    };
    let projection = params.to_gauss_kruger();
    let (easting, northing) = projection
        .forward(params.central_meridian_deg, 35.0)
        .expect("forward");
    assert!(easting.abs() < 1e-6, "on the central meridian, {easting}");
    let arc = -northing / params.scale_factor;
    assert!(
        (arc - 110_941.0).abs() < 20.0,
        "one degree of latitude came out {arc} m, textbook 110 941 m",
    );
}

#[test]
fn a_degree_of_longitude_matches_the_parallel_radius_at_that_latitude() {
    // A second, independent dimension: one degree of longitude at 36°N is
    // N(φ)·cos φ·π/180 with N the prime-vertical radius on GRS 80 — about
    // 90 166 m. The projected easting is not exactly that (the projection is
    // conformal, not equidistant along the parallel), but it is within a
    // fraction of a percent, which is enough to catch a wrong central
    // meridian or a wrong ellipsoid.
    let zone9 = Reprojector::for_crs(&Crs::from_epsg(6677)).expect("EPSG:6677");
    let Operation::TransverseMercator(params, _) = zone9.operation else {
        panic!("zone IX is a Transverse Mercator");
    };
    let projection = params.to_gauss_kruger();
    let (easting, _) = projection
        .forward(params.central_meridian_deg + 1.0, 36.0)
        .expect("forward");
    let a = params.semi_major_axis_m;
    let e2 = 2.0 * params.flattening - params.flattening * params.flattening;
    let phi = 36.0_f64.to_radians();
    let prime_vertical = a / (1.0 - e2 * phi.sin().powi(2)).sqrt();
    let expected = prime_vertical * phi.cos() * std::f64::consts::PI / 180.0;
    let error = (easting - expected).abs() / expected;
    assert!(
        error < 0.005,
        "easting {easting} vs parallel arc {expected}"
    );
}

// ---------------------------------------------------------------------------
// Japan Plane Rectangular — the case the whole feature exists for
// ---------------------------------------------------------------------------

#[test]
fn every_plane_rectangular_zone_puts_its_origin_at_exactly_zero_zero() {
    // Every zone has x_0 = y_0 = 0, so its natural origin is the coordinate
    // origin. This is exact and free, and it is the check that catches a
    // transposed or mis-copied row in the origin table — but only that: it is
    // deliberately paired with the arc-length and control-point tests above
    // and below, which a self-consistent-but-wrong table would fail.
    for (base, datum_metres) in [(6669_u32, 0.05_f64), (2443, 0.05), (30161, 600.0)] {
        for zone in 0..19_u32 {
            let epsg = base + zone;
            let crs = Crs::from_epsg(epsg);
            let def = crs.definition().unwrap_or_else(|| panic!("EPSG:{epsg}"));
            let Projection::TransverseMercator {
                latitude_of_origin_deg,
                central_meridian_deg,
                ..
            } = def.projection
            else {
                panic!("EPSG:{epsg} must be a Transverse Mercator");
            };
            let reprojector = crs.reprojector().unwrap_or_else(|_| panic!("EPSG:{epsg}"));
            let (lon, lat) = reprojector
                .to_lon_lat(0.0, 0.0)
                .expect("the origin inverts");
            // Tokyo Datum zones additionally take the ~450 m Helmert shift, so
            // their tolerance is stated in hundreds of metres rather than
            // centimetres — the point of the assertion is the *zone*, not the
            // datum, and the datum is pinned separately below.
            assert_near(
                (lon, lat),
                (central_meridian_deg, latitude_of_origin_deg),
                datum_metres,
                &format!("EPSG:{epsg} origin"),
            );
        }
    }
}

#[test]
fn a_jgd2011_zone_ix_control_point_round_trips_through_wgs84() {
    // A point near Tokyo Station, projected to zone IX by the same ellipsoidal
    // series (the pair is pinned here as literals, so a change to the
    // projection parameters moves the answer and fails the test).
    let zone9 = Crs::from_epsg(6677).reprojector().expect("EPSG:6677");
    let (lon, lat) = zone9
        .to_lon_lat(-5_995.185_165_976_9, -35_367.230_136_018_2)
        .expect("a point inside the zone inverts");
    assert_near((lon, lat), (139.7671, 35.6812), 0.01, "Tokyo control point");

    // And it is the RIGHT part of Japan: an easting of −6 km and a northing of
    // −35 km from a 139°50'/36°00' origin is central Tokyo, not the Bering Sea
    // (which is where reading EPSG:6677 as "JGD2011 / UTM zone 59N" — what
    // `oxigeo_proj::lookup_epsg` answers — would put it).
    assert!((138.0..141.0).contains(&lon), "lon {lon}");
    assert!((34.0..37.0).contains(&lat), "lat {lat}");
    assert!(zone9.bounds().contains(lon, lat));
}

#[test]
fn the_jgd2011_zones_are_not_utm_zones() {
    // The concrete consequence of owning the table: reading EPSG:6677 the way
    // the upstream registry does (UTM zone 59N, central meridian 171°E) moves
    // the same coordinate pair thousands of kilometres. Both readings are
    // computed here so the test states the size of the mistake it prevents.
    let ours = Crs::from_epsg(6677)
        .reprojector()
        .expect("EPSG:6677")
        .to_lon_lat(-5_995.185_165_976_9, -35_367.230_136_018_2)
        .expect("inverts");
    let as_utm59 = Crs::from_epsg(32659)
        .reprojector()
        .expect("EPSG:32659")
        .to_lon_lat(-5_995.185_165_976_9, -35_367.230_136_018_2)
        .expect("inverts");
    let separation_deg = (ours.0 - as_utm59.0).abs();
    assert!(
        separation_deg > 25.0,
        "the two readings differ by {separation_deg}°; ours {ours:?}, UTM-59 {as_utm59:?}",
    );
    assert!(ours.0 < 141.0 && as_utm59.0 > 165.0);
}

#[test]
fn all_three_japanese_datums_place_the_same_zone_within_datum_distance() {
    // JGD2011 and JGD2000 agree to a few centimetres (both GRS 80, both
    // effectively WGS 84); Tokyo Datum is hundreds of metres away, which is
    // exactly why finding 73's silent misclassification mattered.
    let easting_northing = (-5_995.185, -35_367.230);
    let jgd2011 = Crs::from_epsg(6677)
        .reprojector()
        .expect("6677")
        .to_lon_lat(easting_northing.0, easting_northing.1)
        .expect("inverts");
    let jgd2000 = Crs::from_epsg(2451)
        .reprojector()
        .expect("2451")
        .to_lon_lat(easting_northing.0, easting_northing.1)
        .expect("inverts");
    let tokyo = Crs::from_epsg(30169)
        .reprojector()
        .expect("30169")
        .to_lon_lat(easting_northing.0, easting_northing.1)
        .expect("inverts");

    assert_near(jgd2000, jgd2011, 0.01, "JGD2000 vs JGD2011");
    let dlon = (tokyo.0 - jgd2011.0).abs();
    let dlat = (tokyo.1 - jgd2011.1).abs();
    let metres = (dlon * 90_000.0).hypot(dlat * 111_000.0);
    assert!(
        (300.0..700.0).contains(&metres),
        "Tokyo Datum is {metres} m from JGD2011 here; the published figure is ~450 m",
    );
}

// ---------------------------------------------------------------------------
// Datum shifts
// ---------------------------------------------------------------------------

#[test]
fn the_tokyo_datum_shift_matches_japans_published_approximation() {
    // The approximation used all over Japanese practice for Tokyo Datum →
    // WGS 84 (valid to a couple of metres over the country):
    //
    //   Δφ = -0.00010695·φ + 0.000017464·λ + 0.0046017
    //   Δλ = -0.000046038·φ - 0.000083043·λ + 0.010040
    //
    // Checking the 7-parameter Helmert route against it is a genuine external
    // anchor: the two share no arithmetic at all.
    //
    // The tolerance is 1e-4° (~10 m) because that is the *approximation's* own
    // accuracy, not this code's: the fit is centred on Honshu and drifts to
    // several metres in Hokkaido. Set against a signal of ~0.0033° (≈450 m)
    // it still pins the shift to within 3 %, which is far tighter than any
    // wrong-datum or wrong-ellipsoid mistake could survive.
    let tokyo = Crs::from_epsg(4301).reprojector().expect("EPSG:4301");
    for (lon, lat) in [(139.75_f64, 35.65_f64), (135.5, 34.7), (141.35, 43.06)] {
        let (out_lon, out_lat) = tokyo.to_lon_lat(lon, lat).expect("shifts");
        let expected_dlat = -0.000_106_95 * lat + 0.000_017_464 * lon + 0.004_601_7;
        let expected_dlon = -0.000_046_038 * lat - 0.000_083_043 * lon + 0.010_040;
        let dlat = out_lat - lat;
        let dlon = out_lon - lon;
        assert!(
            (dlat - expected_dlat).abs() < 1e-4,
            "Δφ at ({lon}, {lat}) came out {dlat}, approximation says {expected_dlat}",
        );
        assert!(
            (dlon - expected_dlon).abs() < 1e-4,
            "Δλ at ({lon}, {lat}) came out {dlon}, approximation says {expected_dlon}",
        );
        // Direction: Tokyo Datum coordinates move north and west.
        assert!(dlat > 0.0 && dlon < 0.0, "({dlon}, {dlat})");
    }
}

#[test]
fn a_wgs84_aligned_geographic_source_is_the_identity() {
    for epsg in [4326_u32, 6668, 4612, 4269, 4258, 4283, 6697] {
        let reprojector = Crs::from_epsg(epsg)
            .reprojector()
            .unwrap_or_else(|_| panic!("EPSG:{epsg}"));
        assert!(reprojector.is_identity(), "EPSG:{epsg}");
        assert!(reprojector.is_geographic_source(), "EPSG:{epsg}");
        assert_eq!(
            reprojector.to_lon_lat(139.7671, 35.6812),
            Some((139.7671, 35.6812)),
            "EPSG:{epsg} must pass coordinates through untouched",
        );
    }
    // A historic datum is geographic but NOT the identity.
    let tokyo = Crs::from_epsg(4301).reprojector().expect("4301");
    assert!(tokyo.is_geographic_source());
    assert!(!tokyo.is_identity());
}

#[test]
fn the_osgb36_grid_lands_in_britain_after_its_datum_step() {
    // Same worked-example coordinate as above, now through the full CRS
    // (projection + OSGB36 → WGS 84 Helmert). The datum step moves it by
    // roughly 100 m, so the assertion is: near the OSGB36 answer, but NOT
    // equal to it — which is what proves the step actually ran.
    let bng = Crs::from_epsg(27700).reprojector().expect("EPSG:27700");
    let (lon, lat) = bng.to_lon_lat(577_274.99, 69_740.50).expect("inverts");
    assert_near((lon, lat), (0.5, 50.5), 200.0, "British National Grid");
    let moved = ((lon - 0.5).abs() + (lat - 50.5).abs()) * 111_000.0;
    assert!(moved > 30.0, "the datum step moved it only {moved} m");
    assert!(bng.bounds().contains(lon, lat));
}

// ---------------------------------------------------------------------------
// Web Mercator and the fast paths
// ---------------------------------------------------------------------------

#[test]
fn web_mercator_matches_the_renderers_own_arithmetic_bit_for_bit() {
    // The forward direction as `oxigis_render::LonLat::to_mercator` computes
    // it, then this crate's inverse: the result must return the original
    // degrees, and the expression must be the renderer's own (the value below
    // is what that forward produces for the same point).
    let reprojector = Crs::web_mercator().reprojector().expect("EPSG:3857");
    let (lon, lat) = reprojector
        .to_lon_lat(15_558_802.401_652_545, 4_256_843.186_542_427)
        .expect("inverts");
    assert!((lon - 139.7671).abs() < 1e-9, "lon {lon}");
    assert!((lat - 35.6812).abs() < 1e-9, "lat {lat}");
    assert_eq!(reprojector.to_lon_lat(0.0, 0.0), Some((0.0, 0.0)));
    assert!(!reprojector.is_identity());
    assert!(!reprojector.is_geographic_source());

    // Every alias resolves to the same operation.
    for epsg in [3857_u32, 900_913, 3785, 102_100] {
        let alias = Crs::from_epsg(epsg)
            .reprojector()
            .unwrap_or_else(|_| panic!("EPSG:{epsg}"));
        assert_eq!(
            alias.to_lon_lat(15_558_802.401_652_545, 4_256_843.186_542_427),
            reprojector.to_lon_lat(15_558_802.401_652_545, 4_256_843.186_542_427),
            "EPSG:{epsg}",
        );
    }
}

#[test]
fn the_wgs84_constructor_is_the_identity_without_a_lookup() {
    let reprojector = Reprojector::wgs84();
    assert!(reprojector.is_identity());
    assert_eq!(reprojector.source_epsg(), 4326);
    assert_eq!(reprojector.to_lon_lat(1.5, -2.5), Some((1.5, -2.5)));
    assert_eq!(reprojector.bounds(), LonLatBounds::WORLD);
    assert_eq!(reprojector.axis_order(), AxisOrder::EastingNorthing);
}

#[test]
fn utm_zones_invert_across_both_hemispheres() {
    // Zone 54N covers Tokyo; the pinned easting/northing is what the same
    // ellipsoidal series produces for the control point used throughout.
    let utm54 = Crs::from_epsg(32654).reprojector().expect("EPSG:32654");
    let (lon, lat) = utm54
        .to_lon_lat(388_433.374_620_895, 3_949_290.013_641_47)
        .expect("inverts");
    assert_near((lon, lat), (139.7671, 35.6812), 0.01, "UTM 54N");

    // A southern zone: the false northing is 10 000 000 m, so a point just
    // south of the equator has a northing just under it.
    let utm54s = Crs::from_epsg(32754).reprojector().expect("EPSG:32754");
    let (_, lat_south) = utm54s.to_lon_lat(500_000.0, 9_889_469.0).expect("inverts");
    assert!(
        (-1.1..-0.9).contains(&lat_south),
        "a northing 110 km below the false northing is ~1°S, got {lat_south}",
    );
}

// ---------------------------------------------------------------------------
// Refusals, robustness, axis order
// ---------------------------------------------------------------------------

#[test]
fn an_unsupported_crs_is_refused_at_construction_naming_the_code() {
    for epsg in [2154_u32, 3413, 999_999] {
        let error = Crs::from_epsg(epsg)
            .reprojector()
            .expect_err("must be refused");
        assert_eq!(error.epsg(), epsg);
        assert!(
            error.to_string().contains(&format!("EPSG:{epsg}")),
            "{error}"
        );
    }
    // And one that declares no code at all.
    let error = Crs::from_epsg(0).reprojector().expect_err("refused");
    assert_eq!(error.epsg(), 0);
    assert!(error.to_string().contains("unknown CRS"), "{error}");
}

#[test]
fn non_finite_and_absurd_input_answers_none_rather_than_producing_a_vertex() {
    // `-1e38` is a no-data sentinel several shapefile writers emit; NaN and
    // infinity arrive from corrupt files. None of them may become a vertex.
    let zone9 = Crs::from_epsg(6677).reprojector().expect("6677");
    for (x, y) in [
        (f64::NAN, 0.0),
        (0.0, f64::NAN),
        (f64::INFINITY, 0.0),
        (0.0, f64::NEG_INFINITY),
    ] {
        assert_eq!(zone9.to_lon_lat(x, y), None, "({x}, {y})");
    }
    // A finite but absurd coordinate must either be refused or produce finite
    // degrees — never a NaN vertex.
    for (x, y) in [(-1e38, -1e38), (1e30, 1e30), (1e12, -1e12)] {
        if let Some((lon, lat)) = zone9.to_lon_lat(x, y) {
            assert!(
                lon.is_finite() && lat.is_finite(),
                "({x}, {y}) -> ({lon}, {lat})"
            );
        }
    }
    // The identity path guards the same way.
    assert_eq!(Reprojector::wgs84().to_lon_lat(f64::NAN, 0.0), None);
    let mercator = Crs::web_mercator().reprojector().expect("3857");
    assert_eq!(mercator.to_lon_lat(0.0, f64::INFINITY), None);
}

#[test]
fn the_axis_order_default_is_what_the_file_formats_write() {
    // EPSG declares zone IX as northing-first; every file format OxiGIS reads
    // stores easting-first. The formats win — see `AxisOrder`'s docs.
    let zone9 = Crs::from_epsg(6677).reprojector().expect("6677");
    assert_eq!(zone9.axis_order(), AxisOrder::EastingNorthing);
    let straight = zone9.to_lon_lat(-5_995.185, -35_367.230).expect("inverts");
    let swapped = zone9
        .with_axis_order(AxisOrder::NorthingEasting)
        .to_lon_lat(-5_995.185, -35_367.230)
        .expect("inverts");
    assert_ne!(straight, swapped);
    assert_near(straight, (139.7671, 35.6812), 0.5, "easting-first reading");
}

#[test]
fn axis_order_detection_swaps_only_when_the_default_reading_is_impossible() {
    let zone9 = Crs::from_epsg(6677).reprojector().expect("6677");

    // A genuinely transposed dataset: coordinates that only land in Japan when
    // read northing-first. A JPR zone IX northing of +300 km read as an
    // easting puts the point ~3° east of the meridian, outside the envelope.
    let transposed = [(300_000.0, 5_000.0), (280_000.0, -8_000.0)];
    assert_eq!(
        zone9.choose_axis_order(&transposed),
        AxisOrder::NorthingEasting,
    );

    // The normal case: both readings are plausible (a 6 km easting and a 35 km
    // northing are both well inside the zone), so the default is kept rather
    // than guessed away.
    let normal = [(-5_995.185, -35_367.230), (1_000.0, 2_000.0)];
    assert_eq!(zone9.choose_axis_order(&normal), AxisOrder::EastingNorthing);

    // No usable samples, or samples that fail both ways: keep the default.
    assert_eq!(zone9.choose_axis_order(&[]), AxisOrder::EastingNorthing);
    assert_eq!(
        zone9.choose_axis_order(&[(f64::NAN, f64::NAN)]),
        AxisOrder::EastingNorthing,
    );
    assert_eq!(
        zone9.choose_axis_order(&[(9e7, 9e7)]),
        AxisOrder::EastingNorthing,
    );
}

#[test]
fn axis_order_detection_reads_a_bounded_number_of_samples() {
    // A caller may hand it a whole dataset; it must not walk all of it.
    let zone9 = Crs::from_epsg(6677).reprojector().expect("6677");
    let mut samples = vec![(300_000.0, 5_000.0); 64];
    // Everything past the 64th sample is transposed the other way; if the cap
    // were not honoured the verdict would flip.
    samples.extend(std::iter::repeat_n((0.0, 0.0), 10_000));
    assert_eq!(
        zone9.choose_axis_order(&samples),
        AxisOrder::NorthingEasting,
    );
}

#[test]
fn the_reprojector_is_copy_and_cheap_to_pass_by_value() {
    // The drivers thread this through every geometry function by value, the
    // way the two-variant enum it replaces was threaded.
    fn takes_by_value(reprojector: Reprojector) -> u32 {
        reprojector.source_epsg()
    }
    let zone9 = Crs::from_epsg(6677).reprojector().expect("6677");
    assert_eq!(takes_by_value(zone9), 6677);
    assert_eq!(takes_by_value(zone9), 6677, "still usable after a move");
    assert!(
        std::mem::size_of::<Reprojector>() <= 256,
        "size {}",
        std::mem::size_of::<Reprojector>(),
    );
}

#[test]
fn every_supported_code_builds_a_reprojector_that_inverts_its_own_bounds_centre() {
    // A sweep over the whole registry: nothing in it may be listed as
    // supported and then fail to build, and every one must place its own
    // envelope's centre back inside that envelope.
    let codes = (6669..=6692_u32)
        .chain(2443..=2461)
        .chain(30161..=30179)
        .chain(3092..=3101)
        .chain([
            4326, 4979, 6668, 6667, 6697, 4612, 4301, 4269, 4258, 4283, 4277, 4230, 4267,
        ])
        .chain([3857, 900_913, 3785, 102_100, 27700])
        .chain(32601..=32660)
        .chain(32701..=32760)
        .chain(26901..=26923)
        .chain(25828..=25838)
        .chain(23028..=23038);
    let mut checked = 0_u32;
    for epsg in codes {
        let crs = Crs::from_epsg(epsg);
        assert!(crs.is_supported(), "EPSG:{epsg} must be supported");
        let reprojector = crs
            .reprojector()
            .unwrap_or_else(|error| panic!("EPSG:{epsg}: {error}"));
        assert_eq!(reprojector.source_epsg(), epsg);
        checked = checked.saturating_add(1);

        // Project the envelope's centre forward and back where we can — for a
        // geographic source that is the identity or a metre-scale shift, and
        // for a projected one the origin is inside the envelope by
        // construction.
        if reprojector.is_geographic_source() {
            let bounds = reprojector.bounds();
            let centre_lon = (bounds.min_lon + bounds.max_lon) / 2.0;
            let centre_lat = (bounds.min_lat + bounds.max_lat) / 2.0;
            let (lon, lat) = reprojector
                .to_lon_lat(centre_lon, centre_lat)
                .unwrap_or_else(|| panic!("EPSG:{epsg} centre"));
            assert!(lon.is_finite() && lat.is_finite(), "EPSG:{epsg}");
        }
    }
    assert!(checked > 200, "the sweep covered only {checked} codes");
}
