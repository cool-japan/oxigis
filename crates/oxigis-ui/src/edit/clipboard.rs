// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Feature clipboard format (editing v1.1 stage 3 — see
//! docs/plans/editing-v11.md): the pure encode/decode half.
//!
//! Copy writes a **bare GeoJSON `FeatureCollection`** — the one shape every
//! other tool reads (QGIS's "Copy features as GeoJSON", geojson.io,
//! ogr2ogr) — with every bbox scrubbed (a copied bbox describes another
//! document's geometry). Paste accepts a `FeatureCollection`, a bare
//! `Feature`, or a bare geometry, normalizing the last two into a
//! one-feature collection; anything else is refused by name. Both
//! directions carry an 8 MiB guard, checked **before** parsing on paste,
//! because a hostile clipboard must never reach serde.

use oxigeo::geojson::types::{Feature, FeatureCollection};

use super::command;

/// Largest clipboard payload either direction will carry, in bytes.
pub const MAX_CLIPBOARD_BYTES: usize = 8 * 1024 * 1024;

/// Encodes `features` (the selected subset of a collection, by index) as
/// bare GeoJSON FeatureCollection text.
///
/// # Errors
///
/// A human-readable refusal: nothing selected, an index past the
/// collection, a payload over [`MAX_CLIPBOARD_BYTES`], or a serializer
/// failure.
pub fn copy_text(features: &FeatureCollection, indices: &[usize]) -> Result<String, String> {
    if indices.is_empty() {
        return Err("Nothing is selected to copy.".to_string());
    }
    let selected: Option<Vec<Feature>> = indices
        .iter()
        .map(|&index| features.features.get(index).cloned())
        .collect();
    let Some(mut selected) = selected else {
        return Err("The selection no longer matches the layer.".to_string());
    };
    for feature in &mut selected {
        feature.bbox = None;
        if let Some(geometry) = feature.geometry.as_mut() {
            command::scrub_geometry_bbox(geometry);
        }
    }
    let collection = FeatureCollection::new(selected);
    let text = oxigeo::geojson::writer::to_string(&collection)
        .map_err(|error| format!("Could not serialize the selection: {error}"))?;
    if text.len() > MAX_CLIPBOARD_BYTES {
        return Err(format!(
            "The selection serializes to {} bytes — over the {} MiB clipboard limit.",
            text.len(),
            MAX_CLIPBOARD_BYTES / (1024 * 1024),
        ));
    }
    Ok(text)
}

/// Decodes pasted text into the features to insert: a FeatureCollection's
/// features verbatim, a bare Feature or bare geometry as a one-feature
/// collection. Bboxes are scrubbed on the way in for the same reason copy
/// scrubs them on the way out.
///
/// # Errors
///
/// A human-readable refusal naming what the text was, when it was not
/// pasteable GeoJSON.
pub fn paste_features(text: &str) -> Result<Vec<Feature>, String> {
    if text.len() > MAX_CLIPBOARD_BYTES {
        return Err(format!(
            "The pasted text is {} bytes — over the {} MiB clipboard limit.",
            text.len(),
            MAX_CLIPBOARD_BYTES / (1024 * 1024),
        ));
    }
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("The clipboard is empty.".to_string());
    }
    let document = oxigeo::geojson::reader::from_str(trimmed)
        .map_err(|error| format!("The pasted text is not GeoJSON: {error}"))?;
    let mut features = match document {
        oxigeo::geojson::reader::GeoJsonDocument::FeatureCollection(collection) => {
            collection.features
        }
        oxigeo::geojson::reader::GeoJsonDocument::Feature(feature) => vec![feature],
        oxigeo::geojson::reader::GeoJsonDocument::Geometry(geometry) => {
            vec![Feature::new(Some(geometry), None)]
        }
    };
    if features.is_empty() {
        return Err("The pasted FeatureCollection holds no features.".to_string());
    }
    for feature in &mut features {
        feature.bbox = None;
        if let Some(geometry) = feature.geometry.as_mut() {
            command::scrub_geometry_bbox(geometry);
        }
    }
    Ok(features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigeo::geojson::types::{Geometry, Point};

    fn point_features(count: usize) -> FeatureCollection {
        let features = (0..count)
            .map(|index| {
                Feature::new(
                    Some(Geometry::Point(
                        Point::new_2d(index as f64, index as f64).expect("a point"),
                    )),
                    None,
                )
            })
            .collect();
        FeatureCollection::new(features)
    }

    #[test]
    fn copy_then_paste_round_trips_the_selected_subset() {
        let features = point_features(3);
        let text = copy_text(&features, &[0, 2]).expect("copy succeeds");
        assert!(text.contains("\"FeatureCollection\""));
        let pasted = paste_features(&text).expect("paste succeeds");
        assert_eq!(pasted.len(), 2);
        // Geometry is the payload that must survive exactly; the properties
        // slot may normalize between `None` and an explicit `null`.
        assert_eq!(pasted[0].geometry, features.features[0].geometry);
        assert_eq!(pasted[1].geometry, features.features[2].geometry);
    }

    #[test]
    fn a_bare_feature_and_a_bare_geometry_paste_as_one_feature() {
        let feature = r#"{"type":"Feature","properties":{"name":"a"},
            "geometry":{"type":"Point","coordinates":[1.0,2.0]}}"#;
        assert_eq!(paste_features(feature).expect("a bare feature").len(), 1);
        let geometry = r#"{"type":"Point","coordinates":[1.0,2.0]}"#;
        let pasted = paste_features(geometry).expect("a bare geometry");
        assert_eq!(pasted.len(), 1);
        assert!(pasted[0].geometry.is_some());
    }

    #[test]
    fn refusals_are_named_and_nothing_panics() {
        assert!(copy_text(&point_features(1), &[]).is_err());
        assert!(copy_text(&point_features(1), &[5]).is_err());
        assert!(paste_features("").is_err());
        assert!(paste_features("not json at all").is_err());
        assert!(paste_features("{\"type\":\"Telemetry\"}").is_err());
        let over = "x".repeat(MAX_CLIPBOARD_BYTES + 1);
        assert!(paste_features(&over).is_err());
    }

    #[test]
    fn bboxes_are_scrubbed_both_ways() {
        let mut features = point_features(1);
        features.bbox = Some(vec![0.0, 0.0, 1.0, 1.0]);
        features.features[0].bbox = Some(vec![0.0, 0.0, 1.0, 1.0]);
        let text = copy_text(&features, &[0]).expect("copy succeeds");
        assert!(
            !text.contains("bbox"),
            "a copied bbox describes another document's geometry"
        );
        let with_bbox = r#"{"type":"Feature","bbox":[9.0,9.0,9.0,9.0],
            "geometry":{"type":"Point","coordinates":[1.0,2.0]},"properties":null}"#;
        let pasted = paste_features(with_bbox).expect("paste succeeds");
        assert!(pasted[0].bbox.is_none());
    }
}
