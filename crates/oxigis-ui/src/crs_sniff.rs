// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The shell's CRS seam: turning whatever a local-vector format declares — a
//! `.prj`'s WKT, a GeoPackage `gpkg_spatial_ref_sys` row, a GeoParquet `crs`
//! object — into an [`oxigis_core::Crs`] and the
//! [`oxigis_core::Reprojector`] that places its coordinates.
//!
//! # What this module used to be
//!
//! A two-value classifier (`WktCrs::{Wgs84, WebMercator}`) that matched marker
//! substrings in upper-cased WKT and refused everything else. It had two
//! defects, and both are now structurally impossible rather than patched:
//!
//! 1. Its WGS 84 marker list contained the bare string `"WGS84"`, which is a
//!    substring of `TOWGS84` — the Helmert clause GDAL, ogr2ogr and QGIS emit
//!    for **every** datum that has one. Every Tokyo Datum, NAD27, ED50 and
//!    Pulkovo 1942 `.prj` therefore classified as WGS 84 and drew 100–800 m
//!    from where it belonged, silently. The replacement reads the root
//!    authority code first and strips `TOWGS84[…]` before any name is matched
//!    (`oxigis_core::crs::wkt`).
//! 2. It recognised exactly two CRSs, so a Japanese plane-rectangular
//!    shapefile — the most common real input in this project's home market —
//!    could not be opened at all. The replacement reprojects, through the
//!    projection engine `oxigis-core` had been compiling in and never calling.
//!
//! # What stays here
//!
//! Only the shell-side glue: the "an absent CRS means WGS 84" defaults (which
//! differ per format for different spec reasons) and the refusal *wording*,
//! which each format keeps its own of so a status line names the format the
//! user actually dropped. The classification itself, the EPSG registry and the
//! projection all live in `oxigis-core` — one rule serving ingest, the
//! renderer and the PDF exporter.

use oxigis_core::Crs;
use oxigis_core::crs::Reprojector;

use crate::local_vector::LocalVectorError;

/// The [`Crs`] a WKT string declares, or [`None`] when there is no string —
/// which every caller here reads as "WGS 84", for its own spec reason.
///
/// A leading UTF-8 BOM (common on `.prj` files written on Windows) and
/// surrounding whitespace are handled by [`Crs::from_wkt`]; an empty or
/// whitespace-only string is [`None`], not an unknown CRS.
pub(crate) fn crs_from_wkt(wkt: Option<&str>) -> Option<Crs> {
    let text = wkt
        .map(|text| text.trim_start_matches('\u{feff}').trim())
        .filter(|text| !text.is_empty())?;
    Some(Crs::from_wkt(text))
}

/// The [`Reprojector`] for `crs`, or a refusal naming the format, the CRS and
/// its EPSG code.
///
/// `format` is the word that goes in the message — `"shapefile"`,
/// `"GeoPackage"`, `"GeoParquet"` — so a user who dropped four files at once
/// can tell which one was refused.
///
/// # Errors
///
/// A [`LocalVectorError`] when `crs` is one this build will not place. Loading
/// it would put the data somewhere wrong rather than not at all, which is the
/// stance every driver here takes.
pub(crate) fn reprojector_or_refuse(
    crs: &Crs,
    format: &str,
) -> Result<Reprojector, LocalVectorError> {
    crs.reprojector()
        .map_err(|_| unsupported_crs(format, &crs.label()))
}

/// The refusal a driver shows for a CRS it cannot place.
///
/// Kept as one function so every format's wording differs only in the format
/// name, and so the advice stays honest: the old message suggested Web
/// Mercator as an alternative target, which made sense when those were the
/// only two CRSs that loaded and is now just noise.
pub(crate) fn unsupported_crs(format: &str, label: &str) -> LocalVectorError {
    LocalVectorError::new(format!(
        "unsupported {format} CRS \u{201c}{label}\u{201d}; OxiGIS cannot place this coordinate \
         system \u{2014} reproject the data to WGS 84 (EPSG:4326) first",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absent_or_empty_wkt_is_no_crs_at_all() {
        for absent in [None, Some(""), Some("   "), Some("\u{feff}"), Some("\n\t")] {
            assert_eq!(crs_from_wkt(absent), None, "{absent:?}");
        }
    }

    #[test]
    fn the_two_crss_the_old_classifier_knew_still_resolve() {
        // The old module's own assertions, kept so the rewrite cannot regress
        // what it did get right.
        let wgs84 = crs_from_wkt(Some(r#"GEOGCS["GCS_WGS_1984"]"#)).expect("a CRS");
        assert!(wgs84.is_wgs84());
        let mercator = crs_from_wkt(Some(r#"PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere"]"#))
            .expect("a CRS");
        assert_eq!(mercator.epsg(), 3857);
    }

    #[test]
    fn a_towgs84_clause_no_longer_reads_as_wgs84() {
        // Finding 73, at the seam: the GDAL WKT1 for EPSG:4301 carries
        // `TOWGS84[-146.414,…]`, whose text contains `WGS84`.
        let tokyo = r#"GEOGCS["Tokyo",DATUM["Tokyo",SPHEROID["Bessel 1841",6377397.155,299.1528128],TOWGS84[-146.414,507.337,680.507,0,0,0,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4301"]]"#;
        let crs = crs_from_wkt(Some(tokyo)).expect("a CRS");
        assert_eq!(crs.epsg(), 4301);
        assert!(
            !crs.is_wgs84(),
            "the datum-shift clause must not read as WGS 84"
        );
        // And it now LOADS, with the shift applied, rather than being drawn
        // 450 m out or refused.
        let reprojector = reprojector_or_refuse(&crs, "shapefile").expect("Tokyo Datum loads");
        let (lon, lat) = reprojector.to_lon_lat(139.75, 35.65).expect("shifts");
        assert!(lat > 35.65, "Tokyo Datum coordinates move north");
        assert!(lon < 139.75, "and west");
    }

    #[test]
    fn a_refusal_names_the_format_the_crs_and_the_code() {
        let crs = Crs::from_wkt(
            r#"PROJCS["RGF93 / Lambert-93",PROJECTION["Lambert_Conformal_Conic_2SP"],AUTHORITY["EPSG","2154"]]"#,
        );
        let error = reprojector_or_refuse(&crs, "shapefile").expect_err("must be refused");
        let message = error.message();
        assert!(message.contains("shapefile"), "{message}");
        assert!(message.contains("RGF93 / Lambert-93"), "{message}");
        assert!(message.contains("EPSG:2154"), "{message}");
        assert!(message.contains("EPSG:4326"), "{message}");
        // The old message suggested Web Mercator as a target; it no longer
        // does, because that advice stopped being true.
        assert!(!message.contains("3857"), "{message}");
    }

    #[test]
    fn an_unresolvable_crs_still_names_itself_from_its_wkt() {
        // `Crs::name()` falls back to the WKT's root name, which is what the
        // refusal quotes when the registry has nothing to say.
        let crs = crs_from_wkt(Some(r#"LOCAL_CS["Site grid"]"#)).expect("a CRS");
        assert_eq!(crs.name(), "Site grid");
        let error = reprojector_or_refuse(&crs, "shapefile").expect_err("refused");
        assert!(error.message().contains("Site grid"), "{}", error.message());
    }

    #[test]
    fn a_japanese_plane_rectangular_prj_now_resolves_and_places_tokyo_in_tokyo() {
        let esri = r#"PROJCS["JGD_2011_Japan_Zone_9",GEOGCS["GCS_JGD_2011",DATUM["D_JGD_2011",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",139.8333333333333],PARAMETER["Scale_Factor",0.9999],PARAMETER["Latitude_Of_Origin",36.0],UNIT["Meter",1.0]]"#;
        let crs = crs_from_wkt(Some(esri)).expect("a CRS");
        assert_eq!(crs.epsg(), 6677);
        let reprojector = reprojector_or_refuse(&crs, "shapefile").expect("zone IX loads");
        let (lon, lat) = reprojector
            .to_lon_lat(-5_995.185, -35_367.230)
            .expect("inverts");
        assert!((139.0..140.5).contains(&lon), "lon {lon}");
        assert!((35.0..36.5).contains(&lat), "lat {lat}");
    }
}
