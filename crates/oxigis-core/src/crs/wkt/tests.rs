// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for the WKT reader.
//!
//! The WKT strings here are the shapes real writers emit — GDAL/ogr2ogr WKT1,
//! ESRI `.prj` WKT1 (no `AUTHORITY` clause, `D_`-prefixed datum names) and
//! PROJ/GDAL WKT2 — rather than hand-minimised fragments, because the two
//! defects this module exists to fix (a nested authority code winning, and a
//! `TOWGS84[…]` clause reading as WGS 84) only appear at full length.

use super::*;

/// GDAL WKT1 for EPSG:4301 (Tokyo Datum), with the Helmert clause that made
/// finding 73 possible.
const TOKYO_4301_WKT1: &str = r#"GEOGCS["Tokyo",DATUM["Tokyo",SPHEROID["Bessel 1841",6377397.155,299.1528128,AUTHORITY["EPSG","7004"]],TOWGS84[-146.414,507.337,680.507,0,0,0,0],AUTHORITY["EPSG","6301"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4301"]]"#;

/// GDAL WKT1 for EPSG:4267 (NAD27).
const NAD27_4267_WKT1: &str = r#"GEOGCS["NAD27",DATUM["North_American_Datum_1927",SPHEROID["Clarke 1866",6378206.4,294.9786982138982,AUTHORITY["EPSG","7008"]],TOWGS84[-8,160,176,0,0,0,0],AUTHORITY["EPSG","6267"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4267"]]"#;

/// GDAL WKT1 for EPSG:4230 (ED50).
const ED50_4230_WKT1: &str = r#"GEOGCS["ED50",DATUM["European_Datum_1950",SPHEROID["International 1924",6378388,297,AUTHORITY["EPSG","7022"]],TOWGS84[-87,-98,-121,0,0,0,0],AUTHORITY["EPSG","6230"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4230"]]"#;

/// GDAL WKT1 for EPSG:6677 — JGD2011 / Japan Plane Rectangular CS IX. Five
/// authority clauses, only the last of which is the CRS's own.
const JGD2011_ZONE9_WKT1: &str = r#"PROJCS["JGD2011 / Japan Plane Rectangular CS IX",GEOGCS["JGD2011",DATUM["Japanese_Geodetic_Datum_2011",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","1128"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","6668"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",36],PARAMETER["central_meridian",139.833333333333],PARAMETER["scale_factor",0.9999],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Northing",NORTH],AXIS["Easting",EAST],AUTHORITY["EPSG","6677"]]"#;

/// PROJ WKT2 for the same CRS.
const JGD2011_ZONE9_WKT2: &str = r#"PROJCRS["JGD2011 / Japan Plane Rectangular CS IX",BASEGEOGCRS["JGD2011",DATUM["Japanese Geodetic Datum 2011",ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",6668]],CONVERSION["Japan Plane Rectangular CS zone IX",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",36,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",139.833333333333,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["Scale factor at natural origin",0.9999,SCALEUNIT["unity",1],ID["EPSG",8805]]],CS[Cartesian,2],AXIS["northing (X)",north,ORDER[1],LENGTHUNIT["metre",1]],AXIS["easting (Y)",east,ORDER[2],LENGTHUNIT["metre",1]],ID["EPSG",6677]]"#;

/// An ESRI `.prj` for the same zone: no `AUTHORITY` clause anywhere, and the
/// zone spelled the ESRI way. This is what a lot of Japanese municipal open
/// data actually ships.
const JGD2011_ZONE9_ESRI: &str = r#"PROJCS["JGD_2011_Japan_Zone_9",GEOGCS["GCS_JGD_2011",DATUM["D_JGD_2011",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",139.8333333333333],PARAMETER["Scale_Factor",0.9999],PARAMETER["Latitude_Of_Origin",36.0],UNIT["Meter",1.0]]"#;

/// The ESRI `.prj` for Tokyo Datum, which — unlike the GDAL form — has no
/// `TOWGS84` and was already refused by the classifier this replaces.
const TOKYO_ESRI: &str = r#"GEOGCS["GCS_Tokyo",DATUM["D_Tokyo",SPHEROID["Bessel_1841",6377397.155,299.1528128]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]"#;

#[test]
fn a_towgs84_clause_no_longer_makes_a_tokyo_datum_file_read_as_wgs84() {
    // THE regression of finding 73. `TOWGS84` contains `WGS84`; the classifier
    // this replaces accepted any geographic WKT with that substring anywhere,
    // so this exact string classified as EPSG:4326 and drew ~450 m out.
    assert!(
        TOKYO_4301_WKT1.contains("TOWGS84"),
        "the fixture has the clause"
    );
    assert_eq!(resolve_epsg(TOKYO_4301_WKT1), Some(4301));
    assert_ne!(resolve_epsg(TOKYO_4301_WKT1), Some(4326));

    let info = parse_wkt(TOKYO_4301_WKT1);
    assert!(info.has_towgs84, "the clause is recorded, not just ignored");
    assert_eq!(info.kind, WktKind::Geographic);
    assert_eq!(info.name.as_deref(), Some("Tokyo"));
    assert_eq!(info.datum_name.as_deref(), Some("Tokyo"));
    assert_eq!(info.authority_epsg, Some(4301));
}

#[test]
fn the_other_two_towgs84_datums_from_the_finding_also_resolve_correctly() {
    assert_eq!(resolve_epsg(NAD27_4267_WKT1), Some(4267));
    assert_eq!(resolve_epsg(ED50_4230_WKT1), Some(4230));
    for wkt in [NAD27_4267_WKT1, ED50_4230_WKT1] {
        assert!(parse_wkt(wkt).has_towgs84);
        assert_ne!(resolve_epsg(wkt), Some(4326), "{wkt}");
    }
}

#[test]
fn stripping_removes_the_helmert_clause_that_carried_the_false_positive() {
    let stripped = strip_shift_and_authority_clauses(TOKYO_4301_WKT1);
    assert!(!stripped.contains("TOWGS84"), "{stripped}");
    assert!(!stripped.contains("WGS84"), "{stripped}");
    assert!(!stripped.contains("AUTHORITY"), "{stripped}");
    // The names survive — they are what the fallback needs.
    assert!(stripped.contains("TOKYO"), "{stripped}");
    assert!(stripped.contains("BESSEL 1841"), "{stripped}");
    // And a genuine WGS 84 string still says so after stripping.
    let wgs84 = strip_shift_and_authority_clauses(
        r#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]]]"#,
    );
    assert!(wgs84.contains("WGS 84"), "{wgs84}");
}

#[test]
fn the_authority_code_is_read_from_the_root_not_from_the_first_match() {
    // Five EPSG authority clauses precede the CRS's own; a first-match grab
    // returns 7019 (the GRS 1980 spheroid) and a "last EPSG-looking number"
    // heuristic is one reordering away from breaking.
    let info = parse_wkt(JGD2011_ZONE9_WKT1);
    assert_eq!(info.authority_epsg, Some(6677));
    assert_eq!(info.authority_name.as_deref(), Some("EPSG"));
    assert_eq!(info.kind, WktKind::Projected);
    assert_eq!(
        info.name.as_deref(),
        Some("JGD2011 / Japan Plane Rectangular CS IX")
    );
    assert_eq!(
        info.datum_name.as_deref(),
        Some("Japanese_Geodetic_Datum_2011")
    );
    assert_eq!(resolve_epsg(JGD2011_ZONE9_WKT1), Some(6677));
    // 6668 is the *base* geographic CRS's code and appears in the same string;
    // reading it instead would treat metres as degrees.
    assert_ne!(resolve_epsg(JGD2011_ZONE9_WKT1), Some(6668));
}

#[test]
fn wkt2_id_clauses_resolve_the_same_way_with_an_unquoted_code() {
    let info = parse_wkt(JGD2011_ZONE9_WKT2);
    assert_eq!(info.kind, WktKind::Projected);
    assert_eq!(info.authority_name.as_deref(), Some("EPSG"));
    assert_eq!(info.authority_epsg, Some(6677));
    assert_eq!(resolve_epsg(JGD2011_ZONE9_WKT2), Some(6677));
    // METHOD["Transverse Mercator",ID["EPSG",9807]] is nested three deep and
    // must not win.
    assert_ne!(info.authority_epsg, Some(9807));
}

#[test]
fn an_esri_prj_with_no_authority_clause_resolves_by_zone_name() {
    let info = parse_wkt(JGD2011_ZONE9_ESRI);
    assert_eq!(info.authority_epsg, None, "ESRI writes no authority clause");
    assert_eq!(info.name.as_deref(), Some("JGD_2011_Japan_Zone_9"));
    assert_eq!(resolve_epsg(JGD2011_ZONE9_ESRI), Some(6677));
}

#[test]
fn esri_zone_names_map_across_all_three_japanese_datums() {
    for (name, expected) in [
        ("JGD_2011_Japan_Zone_1", 6669_u32),
        ("JGD_2011_Japan_Zone_9", 6677),
        ("JGD_2011_Japan_Zone_19", 6687),
        ("JGD_2000_Japan_Zone_1", 2443),
        ("JGD_2000_Japan_Zone_9", 2451),
        ("Tokyo_Japan_Zone_9", 30169),
    ] {
        let wkt = format!(
            r#"PROJCS["{name}",GEOGCS["GCS_X",DATUM["D_X",SPHEROID["S",6378137.0,298.257222101]]],PROJECTION["Transverse_Mercator"],UNIT["Meter",1.0]]"#
        );
        assert_eq!(resolve_epsg(&wkt), Some(expected), "{name}");
    }
}

#[test]
fn epsg_style_roman_zone_names_map_too() {
    for (numeral, expected) in [("I", 6669_u32), ("IX", 6677), ("XIX", 6687)] {
        let wkt = format!(
            r#"PROJCS["JGD2011 / Japan Plane Rectangular CS {numeral}",PROJECTION["Transverse_Mercator"],UNIT["metre",1]]"#
        );
        assert_eq!(resolve_epsg(&wkt), Some(expected), "{numeral}");
    }
    assert_eq!(roman_to_zone("XVIII"), Some(18));
    assert_eq!(roman_to_zone("XX"), None);
    assert_eq!(roman_to_zone(""), None);
}

#[test]
fn the_two_crss_the_old_classifier_knew_still_classify_the_same_way() {
    // Byte-for-byte the assertions the module this replaces made, so the
    // rewrite cannot regress what it did get right.
    assert_eq!(resolve_epsg(r#"GEOGCS["GCS_WGS_1984"]"#), Some(4326));
    assert_eq!(
        resolve_epsg(r#"PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere"]"#),
        Some(3857),
    );
}

#[test]
fn a_utm_shapefile_is_now_loadable_instead_of_refused() {
    // The old classifier asserted this must be refused. It is a real CRS with
    // a real inverse, so now it resolves — the asymmetry finding 203 names
    // (the COG path handled UTM 54N while the vector path refused it).
    assert_eq!(
        resolve_epsg(r#"PROJCS["WGS_1984_UTM_Zone_54N",AUTHORITY["EPSG","32654"]]"#),
        Some(32654),
    );
    // And with no authority clause at all — the ESRI `.prj` form.
    assert_eq!(
        resolve_epsg(r#"PROJCS["WGS_1984_UTM_Zone_54N",PROJECTION["Transverse_Mercator"]]"#),
        Some(32654),
    );
}

#[test]
fn utm_zone_names_map_across_hemispheres_and_datums() {
    for (name, expected) in [
        ("WGS_1984_UTM_Zone_54N", 32654_u32),
        ("WGS 84 / UTM zone 54N", 32654),
        ("WGS_1984_UTM_Zone_54S", 32754),
        ("WGS_1984_UTM_Zone_1N", 32601),
        ("WGS_1984_UTM_Zone_60S", 32760),
        ("NAD_1983_UTM_Zone_17N", 26917),
        ("ETRS_1989_UTM_Zone_32N", 25832),
        ("ED_1950_UTM_Zone_30N", 23030),
        ("JGD_2011_UTM_Zone_54N", 6691),
        ("JGD_2000_UTM_Zone_54N", 3100),
        ("Tokyo_UTM_Zone_54N", 3095),
    ] {
        let wkt =
            format!(r#"PROJCS["{name}",PROJECTION["Transverse_Mercator"],UNIT["Meter",1.0]]"#);
        assert_eq!(resolve_epsg(&wkt), Some(expected), "{name}");
    }
}

#[test]
fn a_utm_name_this_build_has_no_code_for_is_refused_rather_than_guessed() {
    // GDA94 / MGA is a real family this build does not carry; guessing a WGS 84
    // UTM code for it would apply the wrong datum silently. Rejected here by
    // the datum arm having no match and the zone being outside the fallback's
    // range check.
    for name in [
        "WGS_1984_UTM_Zone_61N",  // no such zone
        "WGS_1984_UTM_Zone_0N",   // no such zone
        "NAD_1983_UTM_Zone_40N",  // outside the block this build carries
        "ETRS_1989_UTM_Zone_10N", // likewise
        "Some_UTM_Zone_54X",      // not a hemisphere marker
    ] {
        let wkt =
            format!(r#"PROJCS["{name}",PROJECTION["Transverse_Mercator"],UNIT["Meter",1.0]]"#);
        assert_eq!(resolve_epsg(&wkt), None, "{name}");
    }
    assert_eq!(utm_zone_from_name("NOTHING HERE"), None);
    assert_eq!(
        utm_zone_from_name("WGS_1984_UTM_ZONE_54N"),
        Some((54, true))
    );
    assert_eq!(utm_zone_from_name("WGS_1984_UTM_ZONE_54"), Some((54, true)));
}

#[test]
fn an_unsupported_code_is_still_reported_so_a_refusal_can_name_it() {
    // RGF93 / Lambert-93: a real CRS, a projection family this build does not
    // invert. `resolve_epsg` reports the declared code; the caller refuses on
    // `is_supported`.
    let wkt = r#"PROJCS["RGF93 / Lambert-93",PROJECTION["Lambert_Conformal_Conic_2SP"],AUTHORITY["EPSG","2154"]]"#;
    assert_eq!(resolve_epsg(wkt), Some(2154));
    assert!(!epsg::is_supported(2154));
    assert_eq!(crs_label(wkt), "RGF93 / Lambert-93");
}

#[test]
fn an_esri_authority_namespace_is_not_read_as_epsg() {
    // ESRI:102008 (North America Albers) is a *different* namespace; treating
    // its number as an EPSG code would resolve to whatever EPSG:102008 is.
    let wkt = r#"PROJCS["North_America_Albers_Equal_Area_Conic",PROJECTION["Albers"],AUTHORITY["ESRI","102008"]]"#;
    assert!(!epsg::is_supported(102_008));
    assert_eq!(
        resolve_epsg(wkt),
        Some(102_008),
        "reported, so it can be named"
    );
    // But the one ESRI code that IS a well-known EPSG alias is translated.
    let mercator = r#"PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",PROJECTION["Mercator_Auxiliary_Sphere"],AUTHORITY["ESRI","102100"]]"#;
    assert_eq!(resolve_epsg(mercator), Some(3857));
}

#[test]
fn the_esri_tokyo_prj_is_still_refused_and_still_names_itself() {
    // No TOWGS84, no authority: only the datum name says Tokyo, and there is
    // no zone in the name, so nothing resolves. The refusal names it.
    assert_eq!(resolve_epsg(TOKYO_ESRI), None);
    assert_eq!(crs_label(TOKYO_ESRI), "GCS_Tokyo");
    let info = parse_wkt(TOKYO_ESRI);
    assert_eq!(info.datum_name.as_deref(), Some("D_Tokyo"));
    assert!(!info.has_towgs84);
}

#[test]
fn a_bom_and_surrounding_whitespace_do_not_shift_the_root_keyword() {
    let with_bom = format!(
        "\u{feff}  {}\n",
        r#"GEOGCS["WGS 84",AUTHORITY["EPSG","4326"]]"#
    );
    let info = parse_wkt(&with_bom);
    assert_eq!(info.kind, WktKind::Geographic);
    assert_eq!(resolve_epsg(&with_bom), Some(4326));
}

#[test]
fn parentheses_are_accepted_as_grouping_the_way_some_writers_emit_them() {
    let wkt = r#"GEOGCS("WGS 84",AUTHORITY("EPSG","4326"))"#;
    let info = parse_wkt(wkt);
    assert_eq!(info.authority_epsg, Some(4326));
    assert_eq!(info.kind, WktKind::Geographic);
}

#[test]
fn every_wkt_kind_keyword_classifies() {
    for (keyword, expected) in [
        ("GEOGCS", WktKind::Geographic),
        ("GEOGCRS", WktKind::Geographic),
        ("PROJCS", WktKind::Projected),
        ("PROJCRS", WktKind::Projected),
        ("COMPD_CS", WktKind::Compound),
        ("COMPOUNDCRS", WktKind::Compound),
        ("GEOCCS", WktKind::Geocentric),
        ("VERT_CS", WktKind::Vertical),
        ("LOCAL_CS", WktKind::Engineering),
        ("ENGCRS", WktKind::Engineering),
        ("NOTACRS", WktKind::Other),
    ] {
        let wkt = format!(r#"{keyword}["x"]"#);
        assert_eq!(parse_wkt(&wkt).kind, expected, "{keyword}");
    }
    assert!(WktKind::Geographic.is_geographic());
    assert!(!WktKind::Projected.is_geographic());
}

#[test]
fn garbage_input_parses_to_nothing_rather_than_panicking() {
    for text in [
        "",
        "   ",
        "\u{feff}",
        "not wkt at all",
        "[[[[[[",
        "]]]]]]",
        r#"GEOGCS["unterminated"#,
        r#"GEOGCS["\u{0}\u{1}",AUTHORITY["EPSG","notanumber"]]"#,
        "PROJCS[",
    ] {
        let info = parse_wkt(text);
        assert!(resolve_epsg(text).is_none() || info.authority_epsg.is_some());
        // `crs_label` always says something.
        let _ = crs_label(text);
    }
    assert_eq!(resolve_epsg("not wkt at all"), None);
}

#[test]
fn pathological_nesting_and_length_stay_bounded() {
    // Deeper than MAX_WKT_DEPTH and far longer than any real CRS: the scan is
    // linear and its per-node state stops growing at the cap.
    let deep = format!("PROJCS[{}\"x\"{}]", "A[".repeat(500), "]".repeat(500));
    let info = parse_wkt(&deep);
    assert_eq!(info.kind, WktKind::Projected);
    let huge = format!(
        r#"GEOGCS["WGS 84",{}AUTHORITY["EPSG","4326"]]"#,
        "X".repeat(200_000)
    );
    // Past MAX_WKT_SCAN_CHARS the authority clause is simply never reached —
    // the point is that this terminates and allocates nothing proportional to
    // the input.
    let info = parse_wkt(&huge);
    assert_eq!(info.kind, WktKind::Geographic);
    assert_eq!(info.name.as_deref(), Some("WGS 84"));
    let stripped = strip_shift_and_authority_clauses(&huge);
    assert!(stripped.len() <= MAX_WKT_SCAN_CHARS + 8);
}

#[test]
fn a_quoted_name_longer_than_the_cap_is_truncated_not_grown() {
    let long_name = "N".repeat(4096);
    let wkt = format!(r#"GEOGCS["{long_name}",AUTHORITY["EPSG","4326"]]"#);
    let info = parse_wkt(&wkt);
    let name = info.name.expect("a name");
    assert_eq!(name.len(), 256, "capped, not grown to 4096");
    assert_eq!(
        info.authority_epsg,
        Some(4326),
        "the scan continues past it"
    );
}

#[test]
fn crs_label_prefers_the_root_name_and_always_says_something() {
    assert_eq!(crs_label(r#"PROJCS["My CRS",UNIT["m",1]]"#), "My CRS");
    assert_eq!(crs_label("LOCAL_CS"), "LOCAL_CS");
    assert_eq!(crs_label(""), "");
    assert_eq!(
        crs_label(JGD2011_ZONE9_WKT1),
        "JGD2011 / Japan Plane Rectangular CS IX"
    );
    // A string with quotes but no parseable root still names the quoted part.
    assert_eq!(crs_label(r#"garbage "Quoted Name" more"#), "Quoted Name");
}

#[test]
fn a_geographic_wgs84_wkt2_with_an_ensemble_datum_resolves() {
    let wkt = r#"GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0],CS[ellipsoidal,2],AXIS["latitude",north],AXIS["longitude",east],ID["EPSG",4326]]"#;
    let info = parse_wkt(wkt);
    assert_eq!(info.kind, WktKind::Geographic);
    assert_eq!(info.authority_epsg, Some(4326));
    assert_eq!(
        info.datum_name.as_deref(),
        Some("World Geodetic System 1984 ensemble"),
    );
    assert_eq!(resolve_epsg(wkt), Some(4326));
}
