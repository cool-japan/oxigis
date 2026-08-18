// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`super::form`]: the typed drafts, the round trip that must
//! not change a value's JSON type, the parse refusals, the key guards and the
//! re-seed rule that keeps a dirty buffer.
//!
//! No egui context and no app: everything here is the buffer, not the widget.

use super::command::EditError;
use super::form::{FieldDraft, FieldKind, FormBuffer};
use crate::attribute_table::MAX_PROPERTY_COLUMNS;
use oxigeo::geojson::types::{Feature, FeatureCollection, Properties};
use oxigis_core::LayerId;
use serde_json::{Value, json};
use std::sync::Arc;

/// A collection of one null-geometry feature carrying `properties`.
fn collection(properties: Properties) -> Arc<FeatureCollection> {
    Arc::new(FeatureCollection::new(vec![Feature::new(
        None,
        Some(properties),
    )]))
}

/// The mixed-type map every round-trip test starts from.
fn mixed() -> Properties {
    let mut properties = Properties::new();
    properties.insert("name".to_string(), json!("Tokyo"));
    properties.insert("population".to_string(), json!(13_960_000_i64));
    properties.insert("area_km2".to_string(), json!(2194.07_f64));
    properties.insert("capital".to_string(), json!(true));
    properties.insert("nickname".to_string(), Value::Null);
    properties.insert("tags".to_string(), json!({ "iso": "JP-13", "rank": 1 }));
    properties
}

/// A buffer seeded from `properties`, bound to feature 0 of a fresh layer.
fn seeded(properties: Properties) -> (FormBuffer, LayerId, Arc<FeatureCollection>) {
    let layer = LayerId::new();
    let features = collection(properties);
    let mut form = FormBuffer::default();
    form.sync(Some((layer, 0)), Some(&features));
    (form, layer, features)
}

/// The index of the row named `key`.
fn row_index(form: &FormBuffer, key: &str) -> usize {
    form.rows()
        .iter()
        .position(|row| row.key == key)
        .unwrap_or_else(|| panic!("the buffer has no {key} row"))
}

#[test]
fn drafts_round_trip_a_mixed_type_properties_map_untouched() {
    let properties = mixed();
    let (form, _, _) = seeded(properties.clone());
    assert_eq!(form.rows().len(), properties.len());
    assert!(
        !form.is_dirty(),
        "seeding is not an edit; Apply must stay disabled until something is typed"
    );

    let built = form.build().expect("an untouched buffer always parses");
    assert_eq!(
        built, properties,
        "a value that went through the form unchanged must come back bit-identical"
    );

    // The kinds are derived from the live values, not guessed from the text.
    let kind_of = |key: &str| form.rows()[row_index(&form, key)].kind;
    assert_eq!(kind_of("name"), FieldKind::Text);
    assert_eq!(kind_of("population"), FieldKind::Integer);
    assert_eq!(kind_of("area_km2"), FieldKind::Float);
    assert_eq!(kind_of("capital"), FieldKind::Bool);
    assert_eq!(kind_of("nickname"), FieldKind::Null);
    assert_eq!(kind_of("tags"), FieldKind::Json);
}

#[test]
fn integer_stays_integer_and_float_stays_float_through_a_round_trip() {
    let mut properties = Properties::new();
    // The trap: 3 and 3.0 are the same number and different JSON types, and a
    // silent promotion changes the attribute table's column type for every
    // downstream consumer.
    properties.insert("whole".to_string(), json!(3_i64));
    properties.insert("fractional".to_string(), json!(3.0_f64));
    let (mut form, _, _) = seeded(properties);

    let built = form.build().expect("both parse");
    assert!(
        built["whole"].is_i64(),
        "an i64 3 must not become 3.0: {:?}",
        built["whole"]
    );
    assert!(
        built["fractional"].is_f64(),
        "an f64 3.0 must not become 3: {:?}",
        built["fractional"]
    );

    // Editing the value, without touching the kind, preserves the type too.
    let whole = row_index(&form, "whole");
    assert!(form.set_row_text(whole, "4"));
    let fractional = row_index(&form, "fractional");
    assert!(form.set_row_text(fractional, "4"));
    let built = form.build().expect("both still parse");
    assert_eq!(built["whole"], json!(4_i64));
    assert!(built["whole"].is_i64());
    assert!(
        built["fractional"].is_f64(),
        "a float row stays a float even when its text has no point: {:?}",
        built["fractional"]
    );

    // An explicit kind change is the *only* way to change the type, and it
    // carries the value across.
    assert!(form.set_row_kind(whole, FieldKind::Float));
    let built = form.build().expect("still parses");
    assert!(built["whole"].is_f64());
    assert_eq!(built["whole"].as_f64(), Some(4.0));
}

#[test]
fn malformed_number_and_malformed_json_each_name_the_offending_key() {
    let (mut form, _, _) = seeded(mixed());

    let population = row_index(&form, "population");
    assert!(form.set_row_text(population, "twelve"));
    let error = form.build().expect_err("a word is not a whole number");
    assert!(error.contains("population"), "{error}");

    assert!(form.set_row_text(population, "13960000"));
    let tags = row_index(&form, "tags");
    assert!(form.set_row_text(tags, "{not json"));
    let error = form.build().expect_err("that is not JSON");
    assert!(error.contains("tags"), "{error}");

    // A float that is not finite would be written as JSON `null`, silently
    // changing the value, so it is refused rather than stored.
    let area = row_index(&form, "area_km2");
    assert!(form.set_row_text(area, "13960000"));
    assert!(form.set_row_text(tags, "null"));
    assert!(form.build().is_ok());
    assert!(form.set_row_text(area, "inf"));
    let error = form.build().expect_err("a non-finite float is refused");
    assert!(error.contains("area_km2"), "{error}");

    // The row itself reports it, for the inline message the window draws.
    let draft = FieldDraft {
        key: "n".to_string(),
        kind: FieldKind::Integer,
        text: "1.5".to_string(),
        flag: false,
        error: None,
    };
    assert!(draft.value().is_err());
}

#[test]
fn duplicate_new_key_is_refused_and_key_removal_drops_it_from_the_result() {
    let (mut form, _, _) = seeded(mixed());

    assert_eq!(
        form.add_key("name", FieldKind::Text, 6),
        Err(EditError::DuplicateKey("name".to_string())),
        "overwriting a row the user cannot see is how attribute data is lost"
    );
    assert_eq!(
        form.add_key("   ", FieldKind::Text, 6),
        Err(EditError::DuplicateKey(String::new())),
        "a blank key names nothing"
    );

    form.add_key("region", FieldKind::Text, 6)
        .expect("a fresh key is accepted");
    assert!(form.is_dirty());
    let region = row_index(&form, "region");
    assert!(form.set_row_text(region, "Kanto"));
    let built = form.build().expect("everything parses");
    assert_eq!(built["region"], json!("Kanto"));

    let nickname = row_index(&form, "nickname");
    assert!(form.remove_row(nickname));
    let built = form.build().expect("everything parses");
    assert!(
        !built.contains_key("nickname"),
        "a removed key must be gone from the stored map, not stored as null"
    );
    assert!(!form.remove_row(form.rows().len()));
}

#[test]
fn adding_a_key_past_the_column_cap_is_refused() {
    let (mut form, _, _) = seeded(Properties::new());
    // The cap is measured against the *layer's* schema, not against this one
    // feature: a key the table could never show is worse than a refusal.
    assert_eq!(
        form.add_key("one_too_many", FieldKind::Text, MAX_PROPERTY_COLUMNS),
        Err(EditError::ColumnCapReached(MAX_PROPERTY_COLUMNS))
    );
    assert!(form.rows().is_empty());
    assert!(!form.is_dirty());

    // Just under the cap it is accepted.
    form.add_key("room_for_one", FieldKind::Text, MAX_PROPERTY_COLUMNS - 1)
        .expect("one column is still free");

    // And the buffer's own row count counts too, for a feature that already
    // carries a full map.
    let mut wide = Properties::new();
    for key in 0..MAX_PROPERTY_COLUMNS {
        wide.insert(format!("k{key:03}"), json!(key));
    }
    let (mut full, _, _) = seeded(wide);
    assert_eq!(full.rows().len(), MAX_PROPERTY_COLUMNS);
    assert_eq!(
        full.add_key("overflow", FieldKind::Text, 0),
        Err(EditError::ColumnCapReached(MAX_PROPERTY_COLUMNS))
    );
}

#[test]
fn sync_reseeds_on_an_arc_change_but_keeps_a_dirty_buffer() {
    let layer = LayerId::new();
    let first = collection(mixed());
    let mut form = FormBuffer::default();
    form.sync(Some((layer, 0)), Some(&first));
    assert_eq!(form.bound(), Some((layer, 0)));

    // A new `Arc` with different data re-seeds, because nothing was typed.
    let mut renamed = mixed();
    renamed.insert("name".to_string(), json!("Kyoto"));
    let second = collection(renamed);
    form.sync(Some((layer, 0)), Some(&second));
    let name = row_index(&form, "name");
    assert_eq!(form.rows()[name].text, "Kyoto");

    // Now type something. Neither a new `Arc` nor a different feature may throw
    // it away: the window shows a banner and offers Apply or Discard instead.
    assert!(form.set_row_text(name, "Osaka"));
    assert!(form.is_dirty());
    let third = collection(mixed());
    form.sync(Some((layer, 0)), Some(&third));
    assert_eq!(form.rows()[name].text, "Osaka", "a dirty buffer is kept");
    form.sync(Some((layer, 4)), Some(&third));
    assert_eq!(form.rows()[name].text, "Osaka");
    assert_eq!(
        form.bound(),
        Some((layer, 0)),
        "the banner has to be able to name the feature the rows belong to"
    );
    form.sync(None, None);
    assert!(form.is_dirty());

    // Discarding un-binds, and the next sync re-seeds from whatever is current.
    form.discard();
    assert!(!form.is_dirty());
    assert_eq!(form.bound(), None);
    form.sync(Some((layer, 0)), Some(&third));
    assert_eq!(form.rows()[row_index(&form, "name")].text, "Tokyo");

    // An out-of-range feature binds to an empty row set rather than panicking.
    form.sync(Some((layer, 99)), Some(&third));
    assert_eq!(form.bound(), Some((layer, 99)));
    assert!(form.rows().is_empty());
    assert_eq!(form.build().map(|built| built.len()), Ok(0));
}

#[test]
fn a_kind_change_carries_the_value_across_where_it_can() {
    let mut draft = FieldDraft::from_value("n", &json!(42_i64));
    assert_eq!(draft.kind, FieldKind::Integer);

    draft.retype(FieldKind::Text);
    assert_eq!(draft.value(), Ok(json!("42")));

    draft.retype(FieldKind::Float);
    assert_eq!(draft.value().map(|value| value.as_f64()), Ok(Some(42.0)));

    draft.retype(FieldKind::Bool);
    assert_eq!(draft.value(), Ok(json!(true)));

    draft.retype(FieldKind::Json);
    assert_eq!(draft.value(), Ok(json!(true)));

    draft.retype(FieldKind::Null);
    assert_eq!(draft.value(), Ok(Value::Null));

    // A conversion the value cannot survive falls back to the kind's neutral
    // value rather than refusing — nothing has been applied yet, so the cost of
    // being wrong is one keystroke.
    let mut text = FieldDraft::from_value("t", &json!("not a number"));
    text.retype(FieldKind::Integer);
    assert_eq!(text.value(), Ok(json!(0_i64)));
    // Retyping to the same kind is a no-op.
    text.retype(FieldKind::Integer);
    assert_eq!(text.value(), Ok(json!(0_i64)));

    assert_eq!(FieldKind::of(&json!([1, 2])), FieldKind::Json);
    assert_eq!(FieldKind::of(&json!(1.5)), FieldKind::Float);
    assert_eq!(FieldKind::ALL.len(), 6);
    assert_eq!(FieldKind::default(), FieldKind::Text);
}
