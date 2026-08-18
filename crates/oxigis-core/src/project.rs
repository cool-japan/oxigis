//! The project file format (`.oxigis.json`): [`Project`] bundles a
//! [`LayerStack`], per-layer [`LayerStyleSet`]s, and a saved [`View`] into a
//! single serde-JSON document.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::CoreError;
use crate::layer::{LayerId, LayerStack};
use crate::style::LayerStyleSet;

/// The current [`Project::format_version`]. Bump only on a breaking change
/// to the wire shape of [`Project`] or its fields; purely additive fields
/// don't need a bump, since loading tolerates unknown fields (see
/// [`Project::from_json_string`]).
pub const CURRENT_FORMAT_VERSION: u32 = 1;

/// Saved map view: where the camera is centered and how zoomed in it is.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct View {
    /// Center longitude, in degrees.
    pub center_lon: f64,
    /// Center latitude, in degrees.
    pub center_lat: f64,
    /// Zoom level (Web Mercator convention: `0` shows the whole world).
    pub zoom: f64,
}

impl Default for View {
    fn default() -> Self {
        Self {
            center_lon: 0.0,
            center_lat: 0.0,
            zoom: 2.0,
        }
    }
}

/// Saved basemap: the XYZ service the map draws under the layer stack, and
/// the credit line that service requires.
///
/// This mirrors the UI's runtime basemap configuration but lives here so a
/// `.oxigis.json` can restore the whole presentation. Additive field on
/// [`Project`] — files written before it exist load with `None`, which means
/// "the file doesn't say" (the app keeps whatever basemap is active), not
/// "no basemap".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectBasemap {
    /// `{z}/{x}/{y}` URL template.
    pub url_template: String,
    /// Hosts `{s}` rotates through; empty when the template has no `{s}`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subdomains: Vec<String>,
    /// Credit line the service requires. Empty hides it.
    #[serde(default)]
    pub attribution: String,
}

/// A saved OxiGIS project (`.oxigis.json`): the layer stack, per-layer
/// styles, and view state needed to fully restore a map session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    /// Project file format version. See [`CURRENT_FORMAT_VERSION`].
    pub format_version: u32,
    /// Project display name.
    pub name: String,
    /// The layer stack, back-to-front.
    pub layers: LayerStack,
    /// Per-layer style, keyed by [`LayerId`]. A layer without an entry here
    /// renders with a format-appropriate default style. Since tiles v1.3
    /// the value is a whole [`LayerStyleSet`] (base + per-family
    /// overrides); a set without overrides serializes byte-identically to
    /// the bare style it replaced, so no `format_version` bump.
    pub styles: BTreeMap<LayerId, LayerStyleSet>,
    /// Saved map view (center + zoom).
    pub view: View,
    /// The basemap the map was drawn over when the project was saved.
    /// `None` in files from builds that predate the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basemap: Option<ProjectBasemap>,
    /// The layer *promoted* to draw as the basemap, instead of the service
    /// [`Self::basemap`] names.
    ///
    /// A soft pointer, and resolution is **total**: a pointer that names a
    /// layer the project does not hold, a hidden layer, a layer that is not
    /// an XYZ raster, or one whose template is unusable is not an error — it
    /// simply does not resolve, and the service draws. "At most one promoted
    /// layer" is therefore structural: an [`Option`] cannot hold two.
    ///
    /// Sound because a [`LayerId`] is never reused (the counter is monotonic
    /// and deserializing reserves), so a pointer can never come to name a
    /// *different* layer than the one it was written for. `None` in files
    /// from builds that predate the field, and in every file where the
    /// service draws.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basemap_layer: Option<LayerId>,
}

impl Project {
    /// Creates an empty project with the given name, at the current file
    /// format version, default view, no layers, and no per-layer styles.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            format_version: CURRENT_FORMAT_VERSION,
            name: name.into(),
            layers: LayerStack::new(),
            styles: BTreeMap::new(),
            view: View::default(),
            basemap: None,
            basemap_layer: None,
        }
    }

    /// Serializes to pretty-printed JSON — the on-disk `.oxigis.json`
    /// shape.
    pub fn to_json_string(&self) -> Result<String, CoreError> {
        serde_json::to_string_pretty(self).map_err(CoreError::from)
    }

    /// Parses a project from JSON text.
    ///
    /// Tolerant of unknown/extra top-level and nested fields (this struct
    /// and its fields are not `#[serde(deny_unknown_fields)]`), and of any
    /// `format_version` at or below [`CURRENT_FORMAT_VERSION`] — the format
    /// only grows additively, so an older file loads exactly as a
    /// contemporary build wrote it. A `format_version` *above*
    /// [`CURRENT_FORMAT_VERSION`] names a wire shape this build was never
    /// taught to read; rather than silently misread it (and, since the app
    /// re-saves the whole document, write that damage back to disk), this
    /// refuses it with [`CoreError::UnsupportedFormatVersion`].
    ///
    /// Also refused: two layers sharing a [`LayerId`]
    /// ([`CoreError::DuplicateLayerId`]) — see
    /// [`crate::layer::LayerStack::validate_unique_ids`].
    pub fn from_json_string(json: &str) -> Result<Self, CoreError> {
        let project: Self = serde_json::from_str(json).map_err(CoreError::from)?;
        if project.format_version > CURRENT_FORMAT_VERSION {
            return Err(CoreError::UnsupportedFormatVersion {
                found: project.format_version,
                supported: CURRENT_FORMAT_VERSION,
            });
        }
        project.layers.validate_unique_ids()?;
        Ok(project)
    }

    /// The layer's style set, if one is stored.
    pub fn style(&self, id: LayerId) -> Option<&LayerStyleSet> {
        self.styles.get(&id)
    }

    /// Stores the layer's style — accepting a bare [`crate::style::LayerStyle`]
    /// (becomes a base-only set) or a whole [`LayerStyleSet`], which is what
    /// keeps most pre-v1.3 call sites compiling unchanged.
    ///
    /// Refuses with [`CoreError::StyleNotApplicable`] when `id` names a
    /// layer that is present in [`Self::layers`] but whose
    /// [`crate::layer::LayerKind::accepts_layer_style`] is `false` (a
    /// provider-drawn vector-tile layer, or any raster layer) — such a
    /// layer's renderer never consults [`Self::styles`], so the entry could
    /// only ever sit inert and be saved to disk unused. An `id` that names
    /// no layer at all is still accepted, matching [`Self::basemap_layer`]'s
    /// dangling-pointer tolerance: a style set slightly ahead of its layer
    /// (or after an undo removes the layer, pending the redo that restores
    /// it) is not corrupt.
    pub fn set_style(
        &mut self,
        id: LayerId,
        style: impl Into<LayerStyleSet>,
    ) -> Result<(), CoreError> {
        if let Some(layer) = self.layers.get(id)
            && !layer.kind.accepts_layer_style()
        {
            return Err(CoreError::StyleNotApplicable(id));
        }
        self.styles.insert(id, style.into());
        Ok(())
    }

    /// Style entries that can never be drawn: the id names a layer that is
    /// present but whose [`crate::layer::LayerKind`] does not consult
    /// [`Self::styles`] at all (see
    /// [`crate::layer::LayerKind::accepts_layer_style`]).
    ///
    /// Deliberately does **not** report an entry whose id names no layer at
    /// all — the same reasoning as [`Self::basemap_layer`]'s
    /// dangling-pointer tolerance applies: a style set for a layer that is
    /// momentarily absent (not yet added, or removed pending an undo that
    /// will bring it back) is not corrupt, and pruning it here would make
    /// that undo lose the layer's styling. Nothing calls this
    /// automatically; a host that wants to clean up an already-inert entry
    /// (e.g. one saved by an older build that predates
    /// [`Self::set_style`]'s refusal) calls it explicitly.
    #[must_use]
    pub fn prunable_styles(&self) -> Vec<LayerId> {
        self.styles
            .keys()
            .copied()
            .filter(|id| {
                self.layers
                    .get(*id)
                    .is_some_and(|layer| !layer.kind.accepts_layer_style())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{Layer, LayerKind, RasterSource};
    use crate::style::{Color, LayerStyle, SymbolStyle};

    #[test]
    fn new_project_has_current_format_version_and_default_view() {
        let project = Project::new("Untitled");
        assert_eq!(project.format_version, CURRENT_FORMAT_VERSION);
        assert!(project.layers.is_empty());
        assert!(project.styles.is_empty());
        assert_eq!(project.view, View::default());
    }

    #[test]
    fn json_round_trip_preserves_layers_styles_and_view() {
        let mut project = Project::new("My Map");
        let id = project.layers.add(Layer::new(
            "OSM",
            LayerKind::Raster(RasterSource::xyz(
                "https://tile.osm.example/{z}/{x}/{y}.png",
            )),
        ));
        project
            .styles
            .insert(id, LayerStyle::Symbol(SymbolStyle::new("name")).into());
        project.view = View {
            center_lon: 139.767,
            center_lat: 35.681,
            zoom: 10.0,
        };

        let json = project.to_json_string().expect("serialize");
        let restored = Project::from_json_string(&json).expect("deserialize");
        assert_eq!(restored, project);
    }

    #[test]
    fn from_json_string_tolerates_unknown_fields() {
        let json = r#"{
            "format_version": 1,
            "name": "Future Format",
            "unknown_top_level_field": {"whatever": true},
            "layers": [
                {
                    "id": 1,
                    "name": "basemap",
                    "visible": true,
                    "opacity": 1.0,
                    "kind": {
                        "kind": "raster",
                        "source": {"type": "xyz", "url_template": "https://x/{z}/{x}/{y}.png"}
                    },
                    "unknown_layer_field": 42
                }
            ],
            "styles": {},
            "view": {"center_lon": 0.0, "center_lat": 0.0, "zoom": 2.0, "unknown_view_field": "x"}
        }"#;

        let project = Project::from_json_string(json).expect("tolerant of unknown fields");
        assert_eq!(project.name, "Future Format");
        assert_eq!(project.layers.len(), 1);
    }

    #[test]
    fn a_saved_basemap_survives_the_json_round_trip() {
        let mut project = Project::new("With Basemap");
        project.basemap = Some(ProjectBasemap {
            url_template: "https://tiles.example/{z}/{x}/{y}.jpg".to_string(),
            subdomains: vec!["a".to_string(), "b".to_string()],
            attribution: "© Example".to_string(),
        });

        let json = project.to_json_string().expect("serialize");
        let restored = Project::from_json_string(&json).expect("deserialize");
        assert_eq!(restored.basemap, project.basemap);
    }

    #[test]
    fn a_file_without_a_basemap_field_loads_with_none() {
        // Every project written before the field existed — and the tolerant
        // fixture above — must keep loading; `None` means "the file doesn't
        // say", which the app treats as "leave the active basemap alone".
        let json = r#"{
            "format_version": 1,
            "name": "Pre-basemap file",
            "layers": [],
            "styles": {},
            "view": {"center_lon": 0.0, "center_lat": 0.0, "zoom": 2.0}
        }"#;

        let project = Project::from_json_string(json).expect("back-compat load");
        assert!(project.basemap.is_none());
    }

    #[test]
    fn a_project_without_a_basemap_serializes_without_the_field() {
        // `skip_serializing_if` keeps a never-touched project byte-compatible
        // with what older builds wrote (and re-loadable by them).
        let json = Project::new("Plain").to_json_string().expect("serialize");
        assert!(!json.contains("\"basemap\""));
    }

    #[test]
    fn a_promoted_basemap_layer_survives_the_json_round_trip() {
        let mut project = Project::new("With a promoted layer");
        let id = project.layers.add(Layer::new(
            "OSM",
            LayerKind::Raster(RasterSource::xyz("https://x/{z}/{x}/{y}.png")),
        ));
        project.basemap_layer = Some(id);

        let json = project.to_json_string().expect("serialize");
        let restored = Project::from_json_string(&json).expect("deserialize");
        assert_eq!(restored.basemap_layer, Some(id));
        assert_eq!(restored, project);
    }

    #[test]
    fn a_project_without_a_promoted_layer_serializes_without_the_field() {
        // `skip_serializing_if` keeps every project that never promoted a
        // layer byte-compatible with what older builds wrote.
        let json = Project::new("Plain").to_json_string().expect("serialize");
        assert!(!json.contains("basemap_layer"));
    }

    #[test]
    fn a_dangling_promoted_layer_pointer_loads_unchanged() {
        // Resolution is total, so a pointer at a layer the file does not hold
        // is NOT a load error — and it is not scrubbed either: an undo that
        // restores the layer must bring the promotion back with it.
        let json = r#"{
            "format_version": 1,
            "name": "Dangling pointer",
            "layers": [],
            "styles": {},
            "view": {"center_lon": 0.0, "center_lat": 0.0, "zoom": 2.0},
            "basemap_layer": 4242
        }"#;

        let project = Project::from_json_string(json).expect("a dangling pointer still loads");
        assert_eq!(project.basemap_layer, Some(LayerId::from_raw(4242)));
        // Deserializing the pointer reserves the id, exactly as a `Layer::id`
        // does, so a freshly minted layer can never collide with it.
        assert!(LayerId::new().get() > 4242);
    }

    #[test]
    fn from_json_string_rejects_malformed_json_without_panicking() {
        let result = Project::from_json_string("{ this is not json");
        assert!(matches!(result, Err(CoreError::Json { .. })));
    }

    #[test]
    fn from_json_string_rejects_a_format_version_newer_than_this_build_supports() {
        let json = format!(
            r#"{{
                "format_version": {},
                "name": "From The Future",
                "layers": [],
                "styles": {{}},
                "view": {{"center_lon": 0.0, "center_lat": 0.0, "zoom": 2.0}}
            }}"#,
            CURRENT_FORMAT_VERSION + 1
        );
        let err = Project::from_json_string(&json).expect_err("a newer format must be refused");
        assert_eq!(
            err,
            CoreError::UnsupportedFormatVersion {
                found: CURRENT_FORMAT_VERSION + 1,
                supported: CURRENT_FORMAT_VERSION,
            }
        );
    }

    #[test]
    fn from_json_string_still_loads_a_format_version_older_than_current() {
        // The format has only ever grown additively, so anything at or
        // below CURRENT_FORMAT_VERSION — including a version older than any
        // this build shipped — must keep loading.
        let json = r#"{
            "format_version": 0,
            "name": "From The Past",
            "layers": [],
            "styles": {},
            "view": {"center_lon": 0.0, "center_lat": 0.0, "zoom": 2.0}
        }"#;
        let project = Project::from_json_string(json).expect("an older format version still loads");
        assert_eq!(project.format_version, 0);
    }

    #[test]
    fn from_json_string_rejects_two_layers_sharing_a_layer_id() {
        let json = r#"{
            "format_version": 1,
            "name": "Duplicate ids",
            "layers": [
                {"id": 7, "name": "a", "visible": true, "opacity": 1.0,
                 "kind": {"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}},
                {"id": 7, "name": "b", "visible": true, "opacity": 1.0,
                 "kind": {"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}}
            ],
            "styles": {},
            "view": {"center_lon": 0.0, "center_lat": 0.0, "zoom": 2.0}
        }"#;
        let err = Project::from_json_string(json).expect_err("duplicate layer ids must be refused");
        assert_eq!(err, CoreError::DuplicateLayerId(LayerId::from_raw(7)));
    }

    #[test]
    fn from_json_string_rejects_a_layer_id_near_u64_max_rather_than_risking_a_collision() {
        // Reachable from any hand-edited or corrupt `.oxigis.json`: a
        // LayerId this large cannot be safely reseeded past (there is no
        // larger `u64` to reseed to), so it must be refused at load rather
        // than accepted and left to wrap the id counter into collision with
        // this project's own layer.
        let json = r#"{
            "format_version": 1,
            "name": "Near the ceiling",
            "layers": [
                {"id": 18446744073709551615, "name": "a", "visible": true, "opacity": 1.0,
                 "kind": {"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}}
            ],
            "styles": {},
            "view": {"center_lon": 0.0, "center_lat": 0.0, "zoom": 2.0}
        }"#;
        assert!(matches!(
            Project::from_json_string(json),
            Err(CoreError::Json { .. })
        ));
    }

    #[test]
    fn styles_map_keys_survive_the_layer_id_round_trip() {
        let mut project = Project::new("Styled");
        let id = project.layers.add(Layer::new(
            "Cities",
            LayerKind::Vector(crate::layer::VectorSource::InlineGeoJson {
                geojson: "{\"type\":\"FeatureCollection\",\"features\":[]}".to_string(),
            }),
        ));
        project.styles.insert(
            id,
            LayerStyle::Fill(crate::style::FillStyle::new(Color::from_rgb(10, 20, 30))).into(),
        );

        let json = project.to_json_string().expect("serialize");
        let restored = Project::from_json_string(&json).expect("deserialize");
        assert_eq!(restored.styles.get(&id), project.styles.get(&id));
    }

    #[test]
    fn set_style_refuses_a_provider_drawn_layer() {
        let mut project = Project::new("MVT");
        let id = project.layers.add(Layer::new(
            "Tiles",
            LayerKind::Vector(crate::layer::VectorSource::MvtTiles {
                url_template: "https://x/{z}/{x}/{y}.pbf".to_string(),
                paints: vec![],
            }),
        ));
        let err = project
            .set_style(id, LayerStyle::Symbol(SymbolStyle::new("name")))
            .expect_err("MVT tiles carry their own paint list, not a Project::styles entry");
        assert_eq!(err, CoreError::StyleNotApplicable(id));
        assert!(project.styles.is_empty());
    }

    #[test]
    fn set_style_refuses_a_raster_layer() {
        let mut project = Project::new("Raster");
        let id = project.layers.add(Layer::new(
            "Basemap",
            LayerKind::Raster(RasterSource::xyz("https://x/{z}/{x}/{y}.png")),
        ));
        assert_eq!(
            project.set_style(id, LayerStyle::Symbol(SymbolStyle::new("name"))),
            Err(CoreError::StyleNotApplicable(id))
        );
    }

    #[test]
    fn set_style_still_accepts_an_id_that_names_no_layer_yet() {
        // Mirrors `basemap_layer`'s dangling-pointer tolerance: a style set
        // for an id before its layer is added (or after an undo removes the
        // layer, pending a redo that restores it) is not an error.
        let mut project = Project::new("Not yet added");
        let ghost = LayerId::from_raw(999);
        assert!(
            project
                .set_style(ghost, LayerStyle::Symbol(SymbolStyle::new("name")))
                .is_ok()
        );
        assert!(project.styles.contains_key(&ghost));
    }

    #[test]
    fn set_style_accepts_a_file_backed_vector_layer() {
        let mut project = Project::new("Local vector");
        let id = project.layers.add(Layer::new(
            "Cities",
            LayerKind::Vector(crate::layer::VectorSource::LocalGeoJson {
                path: "cities.geojson".to_string(),
            }),
        ));
        assert!(
            project
                .set_style(id, LayerStyle::Symbol(SymbolStyle::new("name")))
                .is_ok()
        );
        assert!(project.styles.contains_key(&id));
    }

    #[test]
    fn prunable_styles_reports_only_present_layers_whose_kind_refuses_a_style() {
        let mut project = Project::new("Mixed");
        let vector_id = project.layers.add(Layer::new(
            "Cities",
            LayerKind::Vector(crate::layer::VectorSource::LocalGeoJson {
                path: "cities.geojson".to_string(),
            }),
        ));
        let raster_id = project.layers.add(Layer::new(
            "Basemap",
            LayerKind::Raster(RasterSource::xyz("https://x/{z}/{x}/{y}.png")),
        ));
        let ghost = LayerId::from_raw(999);

        // Bypasses `set_style`'s own refusal to simulate a file saved by an
        // older build, or one hand-edited to carry an inert entry — exactly
        // the state `prunable_styles` exists to detect.
        let style = || LayerStyle::Symbol(SymbolStyle::new("name")).into();
        project.styles.insert(vector_id, style());
        project.styles.insert(raster_id, style());
        project.styles.insert(ghost, style());

        assert_eq!(project.prunable_styles(), vec![raster_id]);
    }

    #[test]
    fn loading_a_project_avoids_layer_id_collisions_on_later_adds() {
        // Simulates a project file written by a prior process/session, whose
        // `LayerId` counter had already advanced past ids 1-3. This process's
        // own counter starts fresh at 1, so without reserving loaded ids on
        // deserialize, the next `Layer::new()` here would collide with id 1.
        let json = r#"{
            "format_version": 1,
            "name": "From Prior Session",
            "layers": [
                {"id": 1, "name": "a", "visible": true, "opacity": 1.0,
                 "kind": {"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}},
                {"id": 2, "name": "b", "visible": true, "opacity": 1.0,
                 "kind": {"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}},
                {"id": 3, "name": "c", "visible": true, "opacity": 1.0,
                 "kind": {"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}}
            ],
            "styles": {},
            "view": {"center_lon": 0.0, "center_lat": 0.0, "zoom": 2.0}
        }"#;

        let mut project = Project::from_json_string(json).expect("deserialize");
        let new_id = project.layers.add(Layer::new(
            "added-after-load",
            LayerKind::Raster(RasterSource::xyz("https://y/{z}/{x}/{y}.png")),
        ));

        let mut ids: Vec<u64> = project.layers.layers().iter().map(|l| l.id.get()).collect();
        let original_len = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            original_len,
            "LayerId collision after loading a project"
        );
        assert!(
            new_id.get() > 3,
            "new id {new_id} did not advance past loaded ids"
        );
    }
}
