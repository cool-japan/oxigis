// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Geodetic datums: the ellipsoid a CRS measures on, and the shift that takes
//! its coordinates onto WGS 84.
//!
//! Split out of [`super::epsg`] because both the registry (which datum does
//! EPSG:6677 use?) and the reprojector (what ellipsoid do I hand the
//! Transverse Mercator series, and do I owe a Helmert step afterwards?) need
//! the same answers, and because the accuracy caveats belong in one place
//! rather than repeated at every call site.
//!
//! # The two classes
//!
//! * **GRS80-family** — WGS 84 itself, and the modern national realisations
//!   built on GRS 80: JGD2011, JGD2000, NAD83, ETRS89, GDA94. GRS 80 and
//!   WGS 84's ellipsoids differ only in the sixth decimal of the inverse
//!   flattening (298.257 222 101 vs 298.257 223 563 — about 0.1 mm at the
//!   pole), and the datums themselves agree to a metre or better within their
//!   areas of use. OxiGIS treats them as WGS 84 with **no datum shift**: at
//!   the scale a map is drawn, chasing the residual would cost a grid-shift
//!   file per country and buy nothing visible.
//! * **Historic datums** — Tokyo (Bessel 1841), OSGB36 (Airy 1830), ED50
//!   (International 1924), NAD27 (Clarke 1866). These are hundreds of metres
//!   from WGS 84 and MUST be shifted; each carries the published Helmert
//!   parameters below. This is exactly what finding 73 was about: a Tokyo
//!   datum file mistaken for WGS 84 draws ~450 m from where it belongs.
//!
//! The Helmert parameters are the country-wide averages EPSG publishes
//! (`Tokyo → WGS 84` = EPSG:1312, `OSGB36 → WGS 84` = EPSG:1314,
//! `ED50 → WGS 84` = EPSG:1134-class, `NAD27 → WGS 84` = EPSG:1173-class).
//! They are metre-class, not centimetre-class: the centimetre answer needs the
//! national grid-shift file (`TKY2JGD`, `OSTN15`, `NADCON5`), which OxiGIS
//! does not ship. Documented rather than hidden — see [`Datum::accuracy_note`].

use oxigeo::proj::datum_transform::{BursaWolfParams, Ellipsoid};

/// A reference ellipsoid, named rather than carried as raw numbers so the
/// registry reads as geodesy instead of as a table of magic constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EllipsoidKind {
    /// WGS 84 (a = 6 378 137, 1/f = 298.257 223 563).
    Wgs84,
    /// GRS 1980 (a = 6 378 137, 1/f = 298.257 222 101).
    Grs80,
    /// Bessel 1841 (a = 6 377 397.155, 1/f = 299.152 812 8) — Tokyo Datum.
    Bessel1841,
    /// Airy 1830 (a = 6 377 563.396, 1/f = 299.324 96) — OSGB36.
    Airy1830,
    /// International 1924 / Hayford (a = 6 378 388, 1/f = 297) — ED50.
    International1924,
    /// Clarke 1866 (a = 6 378 206.4, 1/f = 294.978 698 2) — NAD27.
    Clarke1866,
}

impl EllipsoidKind {
    /// Semi-major axis `a`, metres.
    #[must_use]
    pub const fn semi_major_axis_m(self) -> f64 {
        match self {
            Self::Wgs84 | Self::Grs80 => 6_378_137.0,
            Self::Bessel1841 => 6_377_397.155,
            Self::Airy1830 => 6_377_563.396,
            Self::International1924 => 6_378_388.0,
            Self::Clarke1866 => 6_378_206.4,
        }
    }

    /// Inverse flattening `1/f`, as ellipsoid definitions state it.
    #[must_use]
    pub const fn inverse_flattening(self) -> f64 {
        match self {
            Self::Wgs84 => 298.257_223_563,
            Self::Grs80 => 298.257_222_101,
            Self::Bessel1841 => 299.152_812_8,
            Self::Airy1830 => 299.324_96,
            Self::International1924 => 297.0,
            Self::Clarke1866 => 294.978_698_2,
        }
    }

    /// Flattening `f`.
    #[must_use]
    pub fn flattening(self) -> f64 {
        1.0 / self.inverse_flattening()
    }

    /// The same ellipsoid as `oxigeo-proj` states it, for the datum-shift and
    /// Transverse Mercator entry points this crate drives.
    #[must_use]
    pub fn to_oxigeo(self) -> Ellipsoid {
        Ellipsoid::new(self.semi_major_axis_m(), self.inverse_flattening())
    }
}

/// A geodetic datum — which ellipsoid, and how far from WGS 84.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Datum {
    /// WGS 84, the datum every OxiGIS coordinate ends up on.
    Wgs84,
    /// Japanese Geodetic Datum 2011 (GRS 80) — Japan's current national datum.
    Jgd2011,
    /// Japanese Geodetic Datum 2000 (GRS 80).
    Jgd2000,
    /// North American Datum 1983 (GRS 80).
    Nad83,
    /// European Terrestrial Reference System 1989 (GRS 80).
    Etrs89,
    /// Geocentric Datum of Australia 1994 (GRS 80).
    Gda94,
    /// Tokyo Datum (Bessel 1841) — Japan's pre-2002 datum, ~450 m from WGS 84.
    Tokyo,
    /// Ordnance Survey of Great Britain 1936 (Airy 1830).
    Osgb36,
    /// European Datum 1950 (International 1924).
    Ed50,
    /// North American Datum 1927 (Clarke 1866).
    Nad27,
}

impl Datum {
    /// The ellipsoid the datum measures on.
    #[must_use]
    pub const fn ellipsoid(self) -> EllipsoidKind {
        match self {
            Self::Wgs84 => EllipsoidKind::Wgs84,
            Self::Jgd2011 | Self::Jgd2000 | Self::Nad83 | Self::Etrs89 | Self::Gda94 => {
                EllipsoidKind::Grs80
            }
            Self::Tokyo => EllipsoidKind::Bessel1841,
            Self::Osgb36 => EllipsoidKind::Airy1830,
            Self::Ed50 => EllipsoidKind::International1924,
            Self::Nad27 => EllipsoidKind::Clarke1866,
        }
    }

    /// The datum's EPSG name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Wgs84 => "WGS 84",
            Self::Jgd2011 => "JGD2011",
            Self::Jgd2000 => "JGD2000",
            Self::Nad83 => "NAD83",
            Self::Etrs89 => "ETRS89",
            Self::Gda94 => "GDA94",
            Self::Tokyo => "Tokyo",
            Self::Osgb36 => "OSGB36",
            Self::Ed50 => "ED50",
            Self::Nad27 => "NAD27",
        }
    }

    /// Whether the datum is close enough to WGS 84 that OxiGIS applies no
    /// shift — see the module docs for the metre-level justification.
    #[must_use]
    pub const fn is_wgs84_aligned(self) -> bool {
        matches!(
            self,
            Self::Wgs84 | Self::Jgd2011 | Self::Jgd2000 | Self::Nad83 | Self::Etrs89 | Self::Gda94
        )
    }

    /// The published Helmert (Bursa-Wolf) parameters taking this datum onto
    /// WGS 84, or [`None`] when the datum is WGS 84-aligned and needs none.
    #[must_use]
    pub fn to_wgs84_helmert(self) -> Option<BursaWolfParams> {
        match self {
            _ if self.is_wgs84_aligned() => None,
            // EPSG:1312 — Tokyo to WGS 84 (1). The parameters GDAL, QGIS and
            // PROJ all emit in a `TOWGS84[…]` clause for a Tokyo datum `.prj`,
            // which is precisely the clause finding 73's marker scan tripped
            // over.
            Self::Tokyo => Some(BursaWolfParams::new(
                -146.414, 507.337, 680.507, 0.0, 0.0, 0.0, 0.0,
            )),
            // EPSG:1314 — OSGB36 to WGS 84 (6), the seven-parameter national
            // average published by Ordnance Survey.
            Self::Osgb36 => Some(BursaWolfParams::new(
                446.448, -125.157, 542.060, -0.1502, -0.2470, -0.8421, -20.4894,
            )),
            // ED50 to WGS 84, Europe-wide average.
            Self::Ed50 => Some(BursaWolfParams::new(
                -89.5, -93.8, -123.1, 0.0, 0.0, 0.156, -1.2,
            )),
            // NAD27 to WGS 84, CONUS average.
            Self::Nad27 => Some(BursaWolfParams::new(-8.0, 160.0, 176.0, 0.0, 0.0, 0.0, 0.0)),
            // Unreachable: every WGS 84-aligned datum took the first arm.
            Self::Wgs84
            | Self::Jgd2011
            | Self::Jgd2000
            | Self::Nad83
            | Self::Etrs89
            | Self::Gda94 => None,
        }
    }

    /// A one-line statement of how accurate this datum's route to WGS 84 is,
    /// for a status line or a layer panel. [`None`] when nothing needs saying.
    #[must_use]
    pub const fn accuracy_note(self) -> Option<&'static str> {
        match self {
            Self::Tokyo => Some(
                "Tokyo Datum is shifted to WGS 84 with the published national Helmert parameters \
                 (EPSG:1312); expect metre-level residuals, as the centimetre answer needs the \
                 TKY2JGD grid",
            ),
            Self::Osgb36 => Some(
                "OSGB36 is shifted to WGS 84 with the published national Helmert parameters \
                 (EPSG:1314); expect a few metres, as the centimetre answer needs the OSTN15 grid",
            ),
            Self::Ed50 => Some(
                "ED50 is shifted to WGS 84 with a Europe-wide Helmert average; accuracy varies by \
                 several metres across the continent",
            ),
            Self::Nad27 => Some(
                "NAD27 is shifted to WGS 84 with a CONUS-average Helmert; expect several metres, \
                 as the sub-metre answer needs the NADCON5 grids",
            ),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grs80_family_datums_take_no_shift() {
        for datum in [
            Datum::Wgs84,
            Datum::Jgd2011,
            Datum::Jgd2000,
            Datum::Nad83,
            Datum::Etrs89,
            Datum::Gda94,
        ] {
            assert!(datum.is_wgs84_aligned(), "{datum:?}");
            assert_eq!(datum.to_wgs84_helmert(), None, "{datum:?}");
            assert_eq!(datum.accuracy_note(), None, "{datum:?}");
        }
    }

    #[test]
    fn historic_datums_all_carry_a_shift_and_a_caveat() {
        for datum in [Datum::Tokyo, Datum::Osgb36, Datum::Ed50, Datum::Nad27] {
            assert!(!datum.is_wgs84_aligned(), "{datum:?}");
            let helmert = datum
                .to_wgs84_helmert()
                .unwrap_or_else(|| panic!("{datum:?} must carry Helmert parameters"));
            let magnitude =
                (helmert.tx * helmert.tx + helmert.ty * helmert.ty + helmert.tz * helmert.tz)
                    .sqrt();
            assert!(
                magnitude > 100.0,
                "{datum:?} is supposed to be hundreds of metres from WGS 84, got {magnitude}"
            );
            assert!(datum.accuracy_note().is_some(), "{datum:?}");
        }
    }

    #[test]
    fn tokyo_datum_uses_bessel_and_the_epsg_1312_parameters() {
        assert_eq!(Datum::Tokyo.ellipsoid(), EllipsoidKind::Bessel1841);
        let helmert = Datum::Tokyo.to_wgs84_helmert().expect("Tokyo shift");
        assert!((helmert.tx + 146.414).abs() < 1e-9);
        assert!((helmert.ty - 507.337).abs() < 1e-9);
        assert!((helmert.tz - 680.507).abs() < 1e-9);
        assert_eq!(
            (helmert.rx, helmert.ry, helmert.rz, helmert.ds),
            (0.0, 0.0, 0.0, 0.0),
            "EPSG:1312 is a pure three-parameter translation",
        );
    }

    #[test]
    fn ellipsoid_constants_match_their_published_definitions() {
        assert!((EllipsoidKind::Wgs84.semi_major_axis_m() - 6_378_137.0).abs() < 1e-9);
        assert!((EllipsoidKind::Grs80.inverse_flattening() - 298.257_222_101).abs() < 1e-9);
        assert!((EllipsoidKind::Bessel1841.semi_major_axis_m() - 6_377_397.155).abs() < 1e-6);
        assert!((EllipsoidKind::Airy1830.inverse_flattening() - 299.324_96).abs() < 1e-9);
        assert!((EllipsoidKind::International1924.inverse_flattening() - 297.0).abs() < 1e-12);
        assert!((EllipsoidKind::Clarke1866.semi_major_axis_m() - 6_378_206.4).abs() < 1e-6);
        // GRS 80 and WGS 84 differ by well under a millimetre at the pole —
        // the measurement behind "no shift needed" in the module docs. Stated
        // as a distance rather than as a bare flattening delta, because the
        // distance is the claim the docs actually make.
        let delta = (EllipsoidKind::Grs80.flattening() - EllipsoidKind::Wgs84.flattening()).abs();
        let polar_metres = delta * EllipsoidKind::Wgs84.semi_major_axis_m();
        assert!(
            polar_metres < 1e-3,
            "GRS 80 and WGS 84 differ by {polar_metres} m at the pole",
        );
        assert!(polar_metres > 0.0, "they are not the same ellipsoid");
    }

    #[test]
    fn oxigeo_ellipsoids_round_trip_the_same_numbers() {
        for kind in [
            EllipsoidKind::Wgs84,
            EllipsoidKind::Grs80,
            EllipsoidKind::Bessel1841,
            EllipsoidKind::Airy1830,
            EllipsoidKind::International1924,
            EllipsoidKind::Clarke1866,
        ] {
            let ellipsoid = kind.to_oxigeo();
            assert!(
                (ellipsoid.a - kind.semi_major_axis_m()).abs() < 1e-6,
                "{kind:?}"
            );
            assert!(
                (ellipsoid.f() - kind.flattening()).abs() < 1e-15,
                "{kind:?}"
            );
        }
    }

    #[test]
    fn every_datum_names_itself() {
        for datum in [
            Datum::Wgs84,
            Datum::Jgd2011,
            Datum::Jgd2000,
            Datum::Nad83,
            Datum::Etrs89,
            Datum::Gda94,
            Datum::Tokyo,
            Datum::Osgb36,
            Datum::Ed50,
            Datum::Nad27,
        ] {
            assert!(!datum.name().is_empty(), "{datum:?}");
        }
        assert_eq!(Datum::Jgd2011.name(), "JGD2011");
        assert_eq!(Datum::Tokyo.name(), "Tokyo");
    }
}
