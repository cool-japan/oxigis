// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for the [`Crs`] model type — construction, naming, and the serde
//! contract the project format depends on.

use super::*;

#[test]
fn the_default_is_wgs84_and_an_empty_object_deserializes_to_it() {
    assert_eq!(Crs::default(), Crs::wgs84());
    assert_eq!(Crs::wgs84().epsg(), 4326);
    assert!(Crs::wgs84().is_wgs84());
    // THE additive rule: a `crs` object with no `epsg` key is WGS 84, so an
    // older writer that emits `{}` (or a hand-edit that drops the key) keeps
    // the meaning the model had before CRSs existed.
    let from_empty: Crs = serde_json::from_str("{}").expect("an empty object is a CRS");
    assert_eq!(from_empty, Crs::wgs84());
}

#[test]
fn a_crs_with_no_wkt_serializes_to_just_its_code() {
    let crs = Crs::from_epsg(6677);
    let json = serde_json::to_string(&crs).expect("serialize");
    assert_eq!(json, r#"{"epsg":6677}"#);
    let back: Crs = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, crs);
    assert_eq!(back.wkt(), None);
}

#[test]
fn a_crs_with_wkt_round_trips_both_fields_in_a_stable_order() {
    let crs = Crs::new(
        6677,
        Some(r#"PROJCS["JGD2011 / Japan Plane Rectangular CS IX"]"#),
    );
    let json = serde_json::to_string(&crs).expect("serialize");
    assert!(json.starts_with(r#"{"epsg":6677,"wkt":"#), "{json}");
    let back: Crs = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, crs);
    assert_eq!(
        back.wkt(),
        Some(r#"PROJCS["JGD2011 / Japan Plane Rectangular CS IX"]"#)
    );
    // Re-serializing is byte-stable, which is what the project format needs.
    assert_eq!(serde_json::to_string(&back).expect("re-serialize"), json);
}

#[test]
fn an_unknown_field_is_refused_rather_than_silently_dropped() {
    // The field is new, so its shape is still ours to fix; accepting unknown
    // keys now would make every future key ambiguous.
    let result = serde_json::from_str::<Crs>(r#"{"epsg":4326,"proj4":"+proj=longlat"}"#);
    assert!(result.is_err(), "{result:?}");
}

#[test]
fn an_oversized_wkt_is_dropped_and_the_drop_is_idempotent() {
    let huge = format!(r#"GEOGCS["{}"]"#, "X".repeat(Crs::MAX_WKT_BYTES));
    assert!(huge.len() > Crs::MAX_WKT_BYTES);
    let crs = Crs::new(4326, Some(&huge));
    assert_eq!(crs.wkt(), None, "not retained");
    assert_eq!(
        crs.epsg(),
        4326,
        "the code — the part decisions use — survives"
    );

    // A hand-edited file carrying one loads without it, and re-saves without
    // it: never truncated into something that no longer parses as WKT.
    let json = format!(
        r#"{{"epsg":4326,"wkt":{}}}"#,
        serde_json::to_string(&huge).expect("encode")
    );
    let loaded: Crs = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(loaded.wkt(), None);
    let again = serde_json::to_string(&loaded).expect("re-serialize");
    assert_eq!(again, r#"{"epsg":4326}"#);
    assert_eq!(
        serde_json::from_str::<Crs>(&again).expect("re-deserialize"),
        loaded,
        "dropping is idempotent",
    );
}

#[test]
fn an_empty_or_whitespace_wkt_is_treated_as_absent() {
    for text in ["", "   ", "\u{feff}", "\n\t "] {
        assert_eq!(Crs::new(4326, Some(text)).wkt(), None, "{text:?}");
    }
}

#[test]
fn from_wkt_resolves_the_code_and_keeps_the_text() {
    let wkt = r#"PROJCS["JGD2011 / Japan Plane Rectangular CS IX",AUTHORITY["EPSG","6677"]]"#;
    let crs = Crs::from_wkt(wkt);
    assert_eq!(crs.epsg(), 6677);
    assert_eq!(crs.wkt(), Some(wkt));
    assert!(crs.is_supported());
    assert_eq!(crs.name(), "JGD2011 / Japan Plane Rectangular CS IX");
    assert_eq!(
        crs.label(),
        "JGD2011 / Japan Plane Rectangular CS IX (EPSG:6677)"
    );
}

#[test]
fn a_wkt_that_names_nothing_becomes_the_unknown_code() {
    let crs = Crs::from_wkt(r#"LOCAL_CS["Engineering frame"]"#);
    assert_eq!(crs.epsg(), EPSG_UNKNOWN);
    assert!(!crs.is_supported());
    assert_eq!(crs.name(), "Engineering frame");
    assert_eq!(crs.label(), "Engineering frame", "no code to append");
    assert!(crs.unsupported_message().contains("Engineering frame"));
    assert!(crs.reprojector().is_err());
}

#[test]
fn an_unsupported_but_named_code_is_quoted_in_full_in_the_refusal() {
    let crs = Crs::from_wkt(
        r#"PROJCS["RGF93 / Lambert-93",PROJECTION["Lambert_Conformal_Conic_2SP"],AUTHORITY["EPSG","2154"]]"#,
    );
    assert_eq!(crs.epsg(), 2154);
    assert!(!crs.is_supported());
    let message = crs.unsupported_message();
    assert!(message.contains("RGF93 / Lambert-93"), "{message}");
    assert!(message.contains("EPSG:2154"), "{message}");
    match crs.reprojector() {
        Err(ReprojectError::UnsupportedCrs { epsg, message }) => {
            assert_eq!(epsg, 2154);
            assert!(message.contains("EPSG:2154"), "{message}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_bare_code_with_no_wkt_still_names_itself_from_the_registry() {
    assert_eq!(Crs::from_epsg(32654).name(), "WGS 84 / UTM zone 54N");
    assert_eq!(Crs::from_epsg(6668).name(), "JGD2011");
    assert_eq!(Crs::from_epsg(4301).name(), "Tokyo");
    // A code the registry does not know falls back to naming the number.
    assert_eq!(Crs::from_epsg(2154).name(), "EPSG:2154");
    assert_eq!(Crs::from_epsg(EPSG_UNKNOWN).name(), "unknown CRS");
}

#[test]
fn display_matches_label() {
    let crs = Crs::from_epsg(3857);
    assert_eq!(crs.to_string(), crs.label());
    assert_eq!(crs.to_string(), "WGS 84 / Pseudo-Mercator (EPSG:3857)");
    assert_eq!(Crs::web_mercator().epsg(), EPSG_WEB_MERCATOR);
}

#[test]
fn the_historic_datums_carry_an_accuracy_note_and_the_modern_ones_do_not() {
    assert!(
        Crs::from_epsg(4301).accuracy_note().is_some(),
        "Tokyo Datum"
    );
    assert!(
        Crs::from_epsg(30169).accuracy_note().is_some(),
        "Tokyo / JPR IX"
    );
    assert!(Crs::from_epsg(27700).accuracy_note().is_some(), "OSGB36");
    assert_eq!(
        Crs::from_epsg(6677).accuracy_note(),
        None,
        "JGD2011 / JPR IX"
    );
    assert_eq!(Crs::wgs84().accuracy_note(), None);
    assert_eq!(Crs::from_epsg(2154).accuracy_note(), None, "unknown code");
}

#[test]
fn a_definition_is_reachable_through_the_model_type() {
    let def = Crs::from_epsg(6677).definition().expect("EPSG:6677");
    assert_eq!(def.datum, Datum::Jgd2011);
    assert!(def.is_projected());
    assert_eq!(Crs::from_epsg(999_999).definition(), None);
}

#[test]
fn compact_drops_the_wkt_of_a_known_code_and_keeps_it_otherwise() {
    let known = Crs::from_wkt(
        r#"PROJCS["JGD2011 / Japan Plane Rectangular CS IX",AUTHORITY["EPSG","6677"]]"#,
    );
    assert!(known.wkt().is_some(), "the reader keeps the text");
    let compact = known.clone().compact();
    assert_eq!(compact.epsg(), 6677);
    assert_eq!(compact.wkt(), None);
    assert_eq!(
        compact.name(),
        known.name(),
        "the registry names it just as well without the text",
    );
    assert_eq!(
        serde_json::to_string(&compact).expect("serialize"),
        r#"{"epsg":6677}"#,
    );

    // A code the registry does not know keeps its WKT, because that is the
    // only thing that can name it in a refusal.
    let unknown = Crs::from_wkt(
        r#"PROJCS["RGF93 / Lambert-93",PROJECTION["Lambert_Conformal_Conic_2SP"],AUTHORITY["EPSG","2154"]]"#,
    );
    let compact = unknown.clone().compact();
    assert_eq!(compact, unknown);
    assert!(compact.wkt().is_some());
    assert!(compact.unsupported_message().contains("Lambert-93"));

    // Idempotent, and harmless on a CRS that never had WKT.
    assert_eq!(compact.clone().compact(), compact);
    assert_eq!(Crs::wgs84().compact(), Crs::wgs84());
}
