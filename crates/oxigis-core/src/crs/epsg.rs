// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The EPSG registry this crate answers from: which codes OxiGIS knows, what
//! projection and datum each one names, and where on the globe it applies.
//!
//! # Why this table exists rather than `oxigeo_proj::lookup_epsg`
//!
//! `oxigeo-proj` 0.2.3 ships an embedded EPSG database, and it is **wrong for
//! exactly the codes this project's home market uses most**. Measured against
//! the crate as published:
//!
//! * `lookup_epsg(6677)` returns `"JGD2011 / UTM zone 59N"` with
//!   `+proj=utm +zone=59`. EPSG:6677 is *JGD2011 / Japan Plane Rectangular CS
//!   IX* — the CRS Tokyo's municipal data ships in. The whole block
//!   EPSG:6669–6678 is registered as `6618 + zone` UTM zones (a plausible
//!   arithmetic slip: JGD2011's UTM zones really are 6688–6692), and
//!   6679–6687 are absent entirely.
//! * `lookup_epsg(2443..=2461)` — JGD2000 / Japan Plane Rectangular CS I–XIX —
//!   returns `+proj=tmerc +lat_0=0 …`. Every Japan Plane Rectangular zone has
//!   a **non-zero latitude of origin** (20°, 26°, 33°, 36°, 40° or 44°
//!   depending on the zone). A `lat_0 = 0` tmerc puts the northing roughly
//!   4 000 km out.
//!
//! Both defects are silent: they produce coordinates, just not the right ones.
//! So OxiGIS owns the identification table and uses `oxigeo-proj` only for the
//! *mathematics* it is good at — see [`super::reproject`], which drives
//! `oxigeo_proj::transform::cylindrical::GaussKruger` (an ellipsoidal
//! Transverse Mercator, validated in this module's sibling tests against the
//! EPSG Guidance Note 7-2 worked example) with parameters taken from here.
//!
//! # Scope
//!
//! Every code listed here can be *loaded*; anything else is refused by name at
//! ingest rather than drawn in the wrong place. The set is deliberately
//! "what real files in circulation declare", not "all 7 500 EPSG codes":
//!
//! * Japan — JGD2011 / JGD2000 / Tokyo geographic, all 19 plane-rectangular
//!   zones of each, and their UTM zones.
//! * Global — WGS 84, Web Mercator, all 120 WGS 84 UTM zones.
//! * Regional — NAD83, ETRS89, GDA94, OSGB36 (incl. the British National
//!   Grid), ED50, NAD27, and the UTM/national grids on those datums.

use crate::crs::datum::Datum;

/// The map projection a CRS applies to geodetic coordinates.
///
/// Only the three shapes OxiGIS actually inverts. Anything else (Lambert
/// Conformal Conic, polar stereographic, equal-area products) is *not* in
/// [`definition`]'s table at all, so it is refused by EPSG code before a
/// projection is ever chosen — see [`super::reproject::Reprojector`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    /// Longitude/latitude in degrees on the CRS's own datum — no projection.
    Geographic,
    /// Spherical Mercator on a sphere of the WGS 84 semi-major axis
    /// (EPSG:3857 and its aliases). Metres.
    WebMercator,
    /// Transverse Mercator (EPSG method 9807) with the seven standard
    /// parameters, of which the ellipsoid comes from the CRS's [`Datum`].
    TransverseMercator {
        /// Latitude of natural origin, degrees.
        latitude_of_origin_deg: f64,
        /// Longitude of natural origin (central meridian), degrees.
        central_meridian_deg: f64,
        /// Scale factor at the natural origin.
        scale_factor: f64,
        /// False easting, metres.
        false_easting: f64,
        /// False northing, metres.
        false_northing: f64,
    },
}

/// A longitude/latitude rectangle, in degrees, that a CRS is defined over.
///
/// Deliberately generous — this is a *plausibility* envelope used to tell a
/// correctly ordered coordinate pair from a transposed one (see
/// [`super::reproject::Reprojector::choose_axis_order`]), never a hard refusal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LonLatBounds {
    /// Western edge, degrees.
    pub min_lon: f64,
    /// Southern edge, degrees.
    pub min_lat: f64,
    /// Eastern edge, degrees.
    pub max_lon: f64,
    /// Northern edge, degrees.
    pub max_lat: f64,
}

impl LonLatBounds {
    /// The whole globe — the envelope for a CRS with no meaningful bound.
    pub const WORLD: Self = Self {
        min_lon: -180.0,
        min_lat: -90.0,
        max_lon: 180.0,
        max_lat: 90.0,
    };

    /// Builds an envelope, ordering the corners so a caller cannot invert it.
    #[must_use]
    pub fn new(min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) -> Self {
        Self {
            min_lon: min_lon.min(max_lon),
            min_lat: min_lat.min(max_lat),
            max_lon: min_lon.max(max_lon),
            max_lat: min_lat.max(max_lat),
        }
    }

    /// Whether a lon/lat pair (degrees) falls inside the envelope.
    #[must_use]
    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        lon >= self.min_lon && lon <= self.max_lon && lat >= self.min_lat && lat <= self.max_lat
    }

    /// This envelope grown by `lon_pad`/`lat_pad` degrees on each side,
    /// clamped to the globe.
    #[must_use]
    pub fn padded(self, lon_pad: f64, lat_pad: f64) -> Self {
        Self {
            min_lon: (self.min_lon - lon_pad).max(-180.0),
            min_lat: (self.min_lat - lat_pad).max(-90.0),
            max_lon: (self.max_lon + lon_pad).min(180.0),
            max_lat: (self.max_lat + lat_pad).min(90.0),
        }
    }
}

/// Everything OxiGIS knows about one EPSG code.
///
/// Obtained from [`definition`]. The `name` is owned because most entries are
/// generated from a zone number (there are 120 WGS 84 UTM zones and 57 Japan
/// plane-rectangular zones across three datums; spelling each out as a
/// `&'static str` would be 200 lines of copy-paste to keep in step).
#[derive(Debug, Clone, PartialEq)]
pub struct CrsDef {
    /// The EPSG code itself.
    pub epsg: u32,
    /// The CRS's EPSG name, e.g. `"JGD2011 / Japan Plane Rectangular CS IX"`.
    pub name: String,
    /// Geodetic datum, which also fixes the ellipsoid and the shift to WGS 84.
    pub datum: Datum,
    /// How geodetic coordinates are mapped to the CRS's own axes.
    pub projection: Projection,
    /// Where on the globe the CRS applies (generous; see [`LonLatBounds`]).
    pub bounds: LonLatBounds,
}

impl CrsDef {
    /// Whether the CRS is geographic (its coordinates are lon/lat degrees).
    #[must_use]
    pub fn is_geographic(&self) -> bool {
        matches!(self.projection, Projection::Geographic)
    }

    /// Whether the CRS is projected (its coordinates are linear units).
    #[must_use]
    pub fn is_projected(&self) -> bool {
        !self.is_geographic()
    }

    /// `"<name> (EPSG:<code>)"` — what a refusal or a layer panel shows.
    #[must_use]
    pub fn label(&self) -> String {
        format!("{} (EPSG:{})", self.name, self.epsg)
    }
}

/// Roman numerals I–XIX, the way Japan's plane-rectangular zones are named.
const JPR_ZONE_NUMERALS: [&str; 19] = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV",
    "XVI", "XVII", "XVIII", "XIX",
];

/// `(latitude of origin, central meridian)` in degrees for Japan Plane
/// Rectangular CS zones I–XIX, as published by the Geospatial Information
/// Authority of Japan (Ministerial Ordinance No. 9 of 1949, restated in the
/// EPSG registry for every one of the three datums that carry these zones).
///
/// Every zone uses scale factor 0.9999 with a zero false easting/northing, so
/// the origin projects to exactly `(0, 0)` — which is what
/// [`super::reproject`]'s tests anchor on.
///
/// The central meridians here agree, minute for minute, with the (independent)
/// list `oxigeo-proj` carries for EPSG:2443–2461; the *latitudes* are the
/// column that crate sets to zero. See this module's docs.
const JPR_ORIGINS: [(f64, f64); 19] = [
    (33.0, 129.5),               // I     — Nagasaki, Kagoshima (west)
    (33.0, 131.0),               // II    — Fukuoka, Saga, Kumamoto, Oita, Miyazaki
    (36.0, 132.0 + 10.0 / 60.0), // III   — Yamaguchi, Shimane, Hiroshima
    (33.0, 133.5),               // IV    — Kagawa, Ehime, Tokushima, Kochi
    (36.0, 134.0 + 20.0 / 60.0), // V     — Hyogo, Tottori, Okayama
    (36.0, 136.0),               // VI    — Kyoto, Osaka, Fukui, Shiga, Mie, Nara, Wakayama
    (36.0, 137.0 + 10.0 / 60.0), // VII   — Ishikawa, Toyama, Gifu, Aichi
    (36.0, 138.5),               // VIII  — Niigata, Nagano, Yamanashi, Shizuoka
    (36.0, 139.0 + 50.0 / 60.0), // IX    — Tokyo (mainland), Kanto
    (40.0, 140.0 + 50.0 / 60.0), // X     — Aomori, Akita, Yamagata, Iwate, Miyagi
    (44.0, 140.0 + 15.0 / 60.0), // XI    — Hokkaido (west)
    (44.0, 142.0 + 15.0 / 60.0), // XII   — Hokkaido (central)
    (44.0, 144.0 + 15.0 / 60.0), // XIII  — Hokkaido (east)
    (26.0, 142.0),               // XIV   — Ogasawara
    (26.0, 127.5),               // XV    — Okinawa (main island)
    (26.0, 124.0),               // XVI   — Yaeyama, Miyako
    (26.0, 131.0),               // XVII  — Daito
    (20.0, 136.0),               // XVIII — Okinotorishima
    (26.0, 154.0),               // XIX   — Minamitorishima
];

/// Scale factor every Japan Plane Rectangular zone uses.
const JPR_SCALE_FACTOR: f64 = 0.9999;

/// Scale factor every UTM zone uses.
const UTM_SCALE_FACTOR: f64 = 0.9996;

/// False easting every UTM zone uses, metres.
const UTM_FALSE_EASTING: f64 = 500_000.0;

/// False northing a southern-hemisphere UTM zone uses, metres.
const UTM_FALSE_NORTHING_SOUTH: f64 = 10_000_000.0;

/// The central meridian of UTM zone `zone` (`1..=60`), degrees.
#[must_use]
pub fn utm_central_meridian_deg(zone: u32) -> Option<f64> {
    (1..=60)
        .contains(&zone)
        .then_some((zone as f64) * 6.0 - 183.0)
}

/// A Japan Plane Rectangular CS definition for `zone_index` (`0..19`).
fn jpr(epsg: u32, datum: Datum, datum_name: &str, zone_index: usize) -> Option<CrsDef> {
    let (latitude_of_origin_deg, central_meridian_deg) = *JPR_ORIGINS.get(zone_index)?;
    let numeral = JPR_ZONE_NUMERALS.get(zone_index)?;
    Some(CrsDef {
        epsg,
        name: format!("{datum_name} / Japan Plane Rectangular CS {numeral}"),
        datum,
        projection: Projection::TransverseMercator {
            latitude_of_origin_deg,
            central_meridian_deg,
            scale_factor: JPR_SCALE_FACTOR,
            false_easting: 0.0,
            false_northing: 0.0,
        },
        // Each zone is roughly ±1.5° of its meridian, but the published
        // extents are irregular (zone IX reaches from Izu to Tochigi) and this
        // envelope is only a plausibility hint, so it is padded generously.
        bounds: LonLatBounds::new(
            central_meridian_deg - 3.0,
            latitude_of_origin_deg - 6.0,
            central_meridian_deg + 3.0,
            latitude_of_origin_deg + 6.0,
        ),
    })
}

/// A UTM definition for `zone` on `datum`.
fn utm(epsg: u32, datum: Datum, datum_name: &str, zone: u32, north: bool) -> Option<CrsDef> {
    let central_meridian_deg = utm_central_meridian_deg(zone)?;
    let hemisphere = if north { 'N' } else { 'S' };
    Some(CrsDef {
        epsg,
        name: format!("{datum_name} / UTM zone {zone}{hemisphere}"),
        datum,
        projection: Projection::TransverseMercator {
            latitude_of_origin_deg: 0.0,
            central_meridian_deg,
            scale_factor: UTM_SCALE_FACTOR,
            false_easting: UTM_FALSE_EASTING,
            false_northing: if north { 0.0 } else { UTM_FALSE_NORTHING_SOUTH },
        },
        bounds: LonLatBounds::new(
            central_meridian_deg - 6.0,
            if north { -1.0 } else { -81.0 },
            central_meridian_deg + 6.0,
            if north { 85.0 } else { 1.0 },
        ),
    })
}

/// A geographic (lon/lat) definition.
fn geographic(epsg: u32, name: &str, datum: Datum, bounds: LonLatBounds) -> CrsDef {
    CrsDef {
        epsg,
        name: name.to_string(),
        datum,
        projection: Projection::Geographic,
        bounds,
    }
}

/// Japan's national extent, used as the plausibility envelope for every
/// Japanese *geographic* CRS. Includes the outlying islands the plane
/// rectangular zones XIV–XIX exist for.
const JAPAN_BOUNDS: LonLatBounds = LonLatBounds {
    min_lon: 122.0,
    min_lat: 17.0,
    max_lon: 156.0,
    max_lat: 46.5,
};

/// Everything OxiGIS knows about `epsg`, or [`None`] when the code is one this
/// build cannot place.
///
/// [`None`] is not "the code does not exist" — it is "OxiGIS will not guess
/// where this data belongs", which is the answer every ingest path turns into
/// a refusal naming the code. See the module docs for the covered set.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn definition(epsg: u32) -> Option<CrsDef> {
    // ---- Contiguous ranges first: one arithmetic branch each --------------
    match epsg {
        // JGD2011 / Japan Plane Rectangular CS I–XIX. THE block oxigeo-proj
        // mis-registers as UTM zones; see the module docs.
        6669..=6687 => return jpr(epsg, Datum::Jgd2011, "JGD2011", (epsg - 6669) as usize),
        // JGD2000 / Japan Plane Rectangular CS I–XIX.
        2443..=2461 => return jpr(epsg, Datum::Jgd2000, "JGD2000", (epsg - 2443) as usize),
        // Tokyo / Japan Plane Rectangular CS I–XIX (Bessel 1841).
        30161..=30179 => return jpr(epsg, Datum::Tokyo, "Tokyo", (epsg - 30161) as usize),
        // JGD2011 / UTM zones 51N–55N.
        6688..=6692 => return utm(epsg, Datum::Jgd2011, "JGD2011", epsg - 6637, true),
        // JGD2000 / UTM zones 51N–55N.
        3097..=3101 => return utm(epsg, Datum::Jgd2000, "JGD2000", epsg - 3046, true),
        // Tokyo / UTM zones 51N–55N.
        3092..=3096 => return utm(epsg, Datum::Tokyo, "Tokyo", epsg - 3041, true),
        // WGS 84 / UTM, both hemispheres.
        32601..=32660 => return utm(epsg, Datum::Wgs84, "WGS 84", epsg - 32600, true),
        32701..=32760 => return utm(epsg, Datum::Wgs84, "WGS 84", epsg - 32700, false),
        // NAD83 / UTM zones 1N–23N.
        26901..=26923 => return utm(epsg, Datum::Nad83, "NAD83", epsg - 26900, true),
        // ETRS89 / UTM zones 28N–38N.
        25828..=25838 => return utm(epsg, Datum::Etrs89, "ETRS89", epsg - 25800, true),
        // ED50 / UTM zones 28N–38N (International 1924).
        23028..=23038 => return utm(epsg, Datum::Ed50, "ED50", epsg - 23000, true),
        _ => {}
    }

    // ---- Individually named codes ----------------------------------------
    let def = match epsg {
        // Global geographic.
        4326 => geographic(4326, "WGS 84", Datum::Wgs84, LonLatBounds::WORLD),
        // WGS 84 (3D) and the 3D/compound realisations that carry the same
        // horizontal datum: the Z ordinate is dropped on ingest, so they place
        // identically to their 2-D siblings.
        4979 => geographic(4979, "WGS 84 (3D)", Datum::Wgs84, LonLatBounds::WORLD),

        // Japan geographic. 6697 is the compound (JGD2011 + height) CRS the
        // national 3-D city model ships in; 6668 is its horizontal half.
        6668 => geographic(6668, "JGD2011", Datum::Jgd2011, JAPAN_BOUNDS),
        6667 => geographic(6667, "JGD2011 (3D)", Datum::Jgd2011, JAPAN_BOUNDS),
        6697 => geographic(
            6697,
            "JGD2011 + JGD2011 (vertical) height",
            Datum::Jgd2011,
            JAPAN_BOUNDS,
        ),
        4612 => geographic(4612, "JGD2000", Datum::Jgd2000, JAPAN_BOUNDS),
        4301 => geographic(4301, "Tokyo", Datum::Tokyo, JAPAN_BOUNDS),

        // Regional geographic.
        4269 => geographic(
            4269,
            "NAD83",
            Datum::Nad83,
            LonLatBounds::new(-172.0, 14.0, -47.0, 87.0),
        ),
        4258 => geographic(
            4258,
            "ETRS89",
            Datum::Etrs89,
            LonLatBounds::new(-35.0, 32.0, 45.0, 85.0),
        ),
        4283 => geographic(
            4283,
            "GDA94",
            Datum::Gda94,
            LonLatBounds::new(93.0, -61.0, 174.0, -8.0),
        ),
        4277 => geographic(
            4277,
            "OSGB36",
            Datum::Osgb36,
            LonLatBounds::new(-9.0, 49.0, 2.0, 61.0),
        ),
        4230 => geographic(
            4230,
            "ED50",
            Datum::Ed50,
            LonLatBounds::new(-16.0, 25.0, 49.0, 85.0),
        ),
        4267 => geographic(
            4267,
            "NAD27",
            Datum::Nad27,
            LonLatBounds::new(-172.0, 7.0, -47.0, 84.0),
        ),

        // Web Mercator, and the three codes that are the same CRS under a
        // different (or deprecated, or vendor) number.
        3857 | 900_913 | 3785 | 102_100 => CrsDef {
            epsg,
            name: "WGS 84 / Pseudo-Mercator".to_string(),
            datum: Datum::Wgs84,
            projection: Projection::WebMercator,
            bounds: LonLatBounds::new(-180.0, -85.06, 180.0, 85.06),
        },

        // OSGB 1936 / British National Grid — the one non-UTM national grid
        // common enough to be worth naming, and the projection whose EPSG
        // worked example anchors this crate's Transverse Mercator tests.
        27700 => CrsDef {
            epsg,
            name: "OSGB36 / British National Grid".to_string(),
            datum: Datum::Osgb36,
            projection: Projection::TransverseMercator {
                latitude_of_origin_deg: 49.0,
                central_meridian_deg: -2.0,
                scale_factor: 0.999_601_271_7,
                false_easting: 400_000.0,
                false_northing: -100_000.0,
            },
            bounds: LonLatBounds::new(-9.0, 49.0, 2.0, 61.0),
        },

        _ => return None,
    };
    Some(def)
}

/// Whether OxiGIS can place data declared in `epsg`.
#[must_use]
pub fn is_supported(epsg: u32) -> bool {
    definition(epsg).is_some()
}

/// The code OxiGIS prefers for a CRS that has more than one.
///
/// Web Mercator is the only such CRS in circulation, and it has three extra
/// numbers: 900913 (Google's original, from the days before EPSG had a code),
/// 3785 (EPSG's own, deprecated in 2010) and ESRI:102100. All three name the
/// identical projection on the identical datum, so a resolver that keeps them
/// apart only forces every consumer to spell the same case four times.
///
/// Applied when a code is *read out of a file* — see
/// [`super::wkt::resolve_epsg`] — and not by [`super::Crs::from_epsg`], which
/// keeps whatever a caller explicitly asked for.
#[must_use]
pub const fn canonical(epsg: u32) -> u32 {
    match epsg {
        900_913 | 3785 | 102_100 => 3857,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jgd2011_plane_rectangular_ix_is_not_a_utm_zone() {
        // The exact regression oxigeo-proj 0.2.3 has: `lookup_epsg(6677)`
        // there answers "JGD2011 / UTM zone 59N" with a +proj=utm string,
        // which would place Tokyo municipal data in the Bering Sea.
        let def = definition(6677).expect("EPSG:6677 is known");
        assert_eq!(def.name, "JGD2011 / Japan Plane Rectangular CS IX");
        assert_eq!(def.datum, Datum::Jgd2011);
        match def.projection {
            Projection::TransverseMercator {
                latitude_of_origin_deg,
                central_meridian_deg,
                scale_factor,
                false_easting,
                false_northing,
            } => {
                assert!((latitude_of_origin_deg - 36.0).abs() < 1e-12);
                assert!((central_meridian_deg - (139.0 + 50.0 / 60.0)).abs() < 1e-12);
                assert!((scale_factor - 0.9999).abs() < 1e-12);
                assert_eq!((false_easting, false_northing), (0.0, 0.0));
            }
            other => panic!("expected a Transverse Mercator, got {other:?}"),
        }
    }

    #[test]
    fn every_plane_rectangular_zone_of_every_datum_resolves() {
        for (base, datum, datum_name) in [
            (6669_u32, Datum::Jgd2011, "JGD2011"),
            (2443, Datum::Jgd2000, "JGD2000"),
            (30161, Datum::Tokyo, "Tokyo"),
        ] {
            for index in 0..19_u32 {
                let epsg = base + index;
                let def = definition(epsg).unwrap_or_else(|| panic!("EPSG:{epsg} must be known"));
                assert_eq!(def.datum, datum);
                assert!(def.name.starts_with(datum_name), "{}", def.name);
                let numeral = JPR_ZONE_NUMERALS[index as usize];
                assert!(
                    def.name.ends_with(&format!("CS {numeral}")),
                    "zone {index} named {}",
                    def.name
                );
                let (expected_lat, expected_lon) = JPR_ORIGINS[index as usize];
                match def.projection {
                    Projection::TransverseMercator {
                        latitude_of_origin_deg,
                        central_meridian_deg,
                        scale_factor,
                        ..
                    } => {
                        assert!((latitude_of_origin_deg - expected_lat).abs() < 1e-12);
                        assert!((central_meridian_deg - expected_lon).abs() < 1e-12);
                        assert!((scale_factor - JPR_SCALE_FACTOR).abs() < 1e-12);
                        assert!(
                            latitude_of_origin_deg > 0.0,
                            "a zero latitude of origin is the oxigeo-proj bug this table exists \
                             to avoid"
                        );
                    }
                    other => panic!("expected a Transverse Mercator, got {other:?}"),
                }
            }
        }
        // One past the last zone must not resolve to a twentieth zone: the
        // two blocks that have nothing after them answer None, and the one
        // that does (JGD2011's UTM zones start at 6688) answers a UTM zone.
        assert!(definition(2462).is_none());
        assert!(definition(30180).is_none());
        assert_eq!(
            definition(6688).map(|def| def.name),
            Some("JGD2011 / UTM zone 51N".to_string()),
        );
    }

    #[test]
    fn the_jgd2011_utm_zones_are_6688_to_6692() {
        for (epsg, zone) in [
            (6688_u32, 51_u32),
            (6689, 52),
            (6690, 53),
            (6691, 54),
            (6692, 55),
        ] {
            let def = definition(epsg).unwrap_or_else(|| panic!("EPSG:{epsg} must be known"));
            assert_eq!(def.name, format!("JGD2011 / UTM zone {zone}N"));
            match def.projection {
                Projection::TransverseMercator {
                    central_meridian_deg,
                    scale_factor,
                    false_easting,
                    false_northing,
                    latitude_of_origin_deg,
                } => {
                    assert!(
                        (central_meridian_deg - ((zone as f64) * 6.0 - 183.0)).abs() < 1e-12,
                        "zone {zone} meridian"
                    );
                    assert!((scale_factor - UTM_SCALE_FACTOR).abs() < 1e-12);
                    assert!((false_easting - UTM_FALSE_EASTING).abs() < 1e-12);
                    assert_eq!((latitude_of_origin_deg, false_northing), (0.0, 0.0));
                }
                other => panic!("expected a Transverse Mercator, got {other:?}"),
            }
        }
    }

    #[test]
    fn all_120_wgs84_utm_zones_resolve_with_the_right_meridian() {
        for zone in 1..=60_u32 {
            let north = definition(32600 + zone).expect("northern zone");
            let south = definition(32700 + zone).expect("southern zone");
            assert_eq!(north.name, format!("WGS 84 / UTM zone {zone}N"));
            assert_eq!(south.name, format!("WGS 84 / UTM zone {zone}S"));
            let expected = (zone as f64) * 6.0 - 183.0;
            for (def, false_northing) in [(&north, 0.0), (&south, UTM_FALSE_NORTHING_SOUTH)] {
                match def.projection {
                    Projection::TransverseMercator {
                        central_meridian_deg,
                        false_northing: fnorth,
                        ..
                    } => {
                        assert!((central_meridian_deg - expected).abs() < 1e-12);
                        assert!((fnorth - false_northing).abs() < 1e-9);
                    }
                    other => panic!("expected a Transverse Mercator, got {other:?}"),
                }
            }
        }
        // Zone 54N covers Tokyo — the asymmetry finding 203 names (the COG
        // path already handled this while the vector path refused it).
        let tokyo_zone = definition(32654).expect("EPSG:32654");
        assert_eq!(tokyo_zone.name, "WGS 84 / UTM zone 54N");
    }

    #[test]
    fn web_mercator_aliases_all_land_on_the_same_projection() {
        for epsg in [3857_u32, 900_913, 3785, 102_100] {
            let def = definition(epsg).unwrap_or_else(|| panic!("EPSG:{epsg}"));
            assert_eq!(def.projection, Projection::WebMercator);
            assert_eq!(def.datum, Datum::Wgs84);
            assert_eq!(def.epsg, epsg, "the alias keeps its own code in its label");
        }
    }

    #[test]
    fn the_british_national_grid_carries_the_epsg_worked_example_parameters() {
        let def = definition(27700).expect("EPSG:27700");
        assert_eq!(def.datum, Datum::Osgb36);
        match def.projection {
            Projection::TransverseMercator {
                latitude_of_origin_deg,
                central_meridian_deg,
                scale_factor,
                false_easting,
                false_northing,
            } => {
                assert!((latitude_of_origin_deg - 49.0).abs() < 1e-12);
                assert!((central_meridian_deg + 2.0).abs() < 1e-12);
                assert!((scale_factor - 0.999_601_271_7).abs() < 1e-15);
                assert!((false_easting - 400_000.0).abs() < 1e-9);
                assert!((false_northing + 100_000.0).abs() < 1e-9);
            }
            other => panic!("expected a Transverse Mercator, got {other:?}"),
        }
    }

    #[test]
    fn unknown_codes_answer_none_rather_than_guessing() {
        // A real EPSG code for a projection family this build does not invert
        // (Lambert Conformal Conic), a polar one, and a nonsense one.
        for epsg in [2154_u32, 3413, 999_999, 0] {
            assert!(definition(epsg).is_none(), "EPSG:{epsg} must be refused");
            assert!(!is_supported(epsg));
        }
    }

    #[test]
    fn labels_name_the_code_so_a_refusal_is_actionable() {
        let def = definition(6677).expect("EPSG:6677");
        assert_eq!(
            def.label(),
            "JGD2011 / Japan Plane Rectangular CS IX (EPSG:6677)"
        );
        assert!(def.is_projected());
        assert!(!def.is_geographic());
        assert!(definition(6668).expect("EPSG:6668").is_geographic());
    }

    #[test]
    fn bounds_are_ordered_and_contain_their_origin() {
        for epsg in [6677_u32, 2451, 30169, 32654, 27700, 4326, 3857] {
            let def = definition(epsg).unwrap_or_else(|| panic!("EPSG:{epsg}"));
            assert!(def.bounds.min_lon < def.bounds.max_lon, "EPSG:{epsg}");
            assert!(def.bounds.min_lat < def.bounds.max_lat, "EPSG:{epsg}");
        }
        let inverted = LonLatBounds::new(10.0, 20.0, -10.0, -20.0);
        assert!(inverted.contains(0.0, 0.0));
        assert!(!inverted.contains(11.0, 0.0));
        assert!(LonLatBounds::WORLD.padded(10.0, 10.0).contains(180.0, 90.0));
    }

    #[test]
    fn utm_central_meridian_matches_the_standard_formula() {
        assert_eq!(utm_central_meridian_deg(1), Some(-177.0));
        assert_eq!(utm_central_meridian_deg(31), Some(3.0));
        assert_eq!(utm_central_meridian_deg(54), Some(141.0));
        assert_eq!(utm_central_meridian_deg(60), Some(177.0));
        assert_eq!(utm_central_meridian_deg(0), None);
        assert_eq!(utm_central_meridian_deg(61), None);
    }
}
