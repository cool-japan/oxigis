//! The Processing tool registry: declarative descriptors that let a UI
//! layer auto-generate parameter forms for OxiGeo-backed tools, without
//! `oxigis-core` depending on any UI toolkit (blueprint §6 —
//! "Processing toolbox(bounds/count 程度) — OxiGeo 関数の GUI 自動マッピング").
//!
//! Only descriptors and the [`ToolExecutor`] contract live here; running a
//! tool against real data is a shell's job, by design — this crate holds no
//! OxiGeo types to run one against (see the crate root docs). `oxigis-ui`'s
//! `processing_exec` module is the reference implementation: it implements
//! `ToolExecutor` for every built-in tool id this registry declares and
//! dispatches [`ToolDescriptor::id`] to the matching executor from the
//! Processing panel.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// The kind of value a processing tool parameter accepts, and how a
/// generated form field should constrain it.
///
/// Adjacently tagged (`{"kind": "...", "value": ...}`) rather than
/// internally tagged: internal tagging requires every variant to serialize
/// as a JSON map, which [`ParamKind::Choice`]'s `Vec<String>` payload
/// cannot do (it's a bare sequence).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ParamKind {
    /// A numeric input, optionally bounded.
    Number {
        /// Minimum allowed value (inclusive), if any.
        min: Option<f64>,
        /// Maximum allowed value (inclusive), if any.
        max: Option<f64>,
    },
    /// A free-text input.
    Text,
    /// A boolean checkbox.
    Bool,
    /// A reference to one of the project's layers, by
    /// [`LayerId`](crate::layer::LayerId).
    LayerRef,
    /// A single choice from a fixed list of options.
    Choice(Vec<String>),
}

/// Declarative description of one tool parameter: enough to render a form
/// field and know whether a value is required, without depending on any UI
/// crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamSpec {
    /// Stable parameter name — the form field key, and the key
    /// [`ToolContext::params`] is expected to use for this parameter.
    pub name: String,
    /// What kind of value this parameter accepts.
    pub kind: ParamKind,
    /// Whether the tool can run without this parameter being set.
    pub required: bool,
    /// Default value, carried as JSON so it can hold any [`ParamKind`]'s
    /// shape (a number, string, bool, or choice) without a generic type
    /// parameter on `ParamSpec` itself.
    pub default: Option<serde_json::Value>,
}

/// Declarative description of one processing tool: enough for a UI to build
/// a parameter form and a toolbox entry without knowing the tool's Rust
/// type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDescriptor {
    /// Stable, unique tool id (e.g. `"bounds"`), used for registry lookup.
    pub id: String,
    /// Human-readable title shown in the Processing toolbox.
    pub title: String,
    /// Longer description shown as help text.
    pub description: String,
    /// Ordered list of parameters the tool's form should render.
    pub params: Vec<ParamSpec>,
}

/// Registry of [`ToolDescriptor`]s, keyed by [`ToolDescriptor::id`].
#[derive(Debug, Clone, Default)]
pub struct ProcessingRegistry {
    tools: BTreeMap<String, ToolDescriptor>,
}

impl ProcessingRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    /// Registers a tool descriptor, overwriting (and returning) any prior
    /// descriptor registered under the same id.
    pub fn register(&mut self, descriptor: ToolDescriptor) -> Option<ToolDescriptor> {
        self.tools.insert(descriptor.id.clone(), descriptor)
    }

    /// Looks up a tool descriptor by id.
    pub fn get(&self, id: &str) -> Result<&ToolDescriptor, CoreError> {
        self.tools
            .get(id)
            .ok_or_else(|| CoreError::UnknownTool(id.to_string()))
    }

    /// Iterates over all registered descriptors, ordered by id.
    pub fn iter(&self) -> impl Iterator<Item = &ToolDescriptor> {
        self.tools.values()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry has no tools registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// The default `simplify` tolerance, in degrees — a little over 100 m at the
/// equator, small enough to be visually lossless at city scale while still
/// dropping the densest vertex runs a hand-digitised coastline carries.
const DEFAULT_SIMPLIFY_TOLERANCE_DEG: f64 = 0.001;

/// A `LayerRef`-only parameter list shared by the built-in single-layer
/// tools (`bounds`, `feature_count`, `centroid`, `convex_hull`).
fn single_layer_param() -> Vec<ParamSpec> {
    vec![layer_param()]
}

/// The `layer` parameter every built-in tool starts with: the vector layer the
/// tool reads its features from.
fn layer_param() -> ParamSpec {
    ParamSpec {
        name: "layer".to_string(),
        kind: ParamKind::LayerRef,
        required: true,
        default: None,
    }
}

/// Builds the registry seeded with OxiGIS's built-in tools.
///
/// Phase 1 shipped `bounds` and `feature_count` (blueprint §6: "bounds/count
/// 程度"); Phase 3 adds the first three tools that *produce geometry* —
/// `centroid`, `simplify` and `convex_hull` — whose results the UI routes
/// straight into a new layer. `buffer` joined them here rather than staying a
/// UI-side registration: a descriptor that only appeared once the Processing
/// window had been opened was a tool that did not exist until then, and the
/// registry is the single place a shell asks what it can run. Each one costs a
/// descriptor here plus a match arm in the UI's executor table, which is the
/// near-zero marginal cost per tool the blueprint expects from
/// descriptor-driven form generation (§7).
pub fn builtin_registry() -> ProcessingRegistry {
    let mut registry = ProcessingRegistry::new();
    registry.register(ToolDescriptor {
        id: "bounds".to_string(),
        title: "Layer Bounds".to_string(),
        description: "Computes the bounding box of a vector layer's features.".to_string(),
        params: single_layer_param(),
    });
    registry.register(ToolDescriptor {
        id: "feature_count".to_string(),
        title: "Feature Count".to_string(),
        description: "Counts the features in a vector layer.".to_string(),
        params: single_layer_param(),
    });
    registry.register(ToolDescriptor {
        id: "centroid".to_string(),
        title: "Centroids".to_string(),
        description: "Replaces every feature with a point at its centroid, \
                      keeping the original properties and id. This is the \
                      vertex-mean centroid (a polygon uses the mean of its \
                      exterior-ring vertices), not the area-weighted centre \
                      of mass, so a polygon with unevenly spaced vertices \
                      leans toward the denser side. Features with no geometry \
                      — or whose centroid is not computable — are skipped."
            .to_string(),
        params: single_layer_param(),
    });
    registry.register(ToolDescriptor {
        id: "simplify".to_string(),
        title: "Simplify (Douglas-Peucker)".to_string(),
        description: "Thins the vertices of every line and polygon with the \
                      Douglas-Peucker algorithm, keeping properties and ids. \
                      The tolerance is measured in degrees of longitude and \
                      latitude, not metres, so its ground distance shrinks \
                      with the cosine of the latitude. Points and multipoints \
                      pass through untouched, as does any feature whose \
                      simplified geometry would no longer be valid."
            .to_string(),
        params: vec![
            layer_param(),
            ParamSpec {
                name: "tolerance_deg".to_string(),
                kind: ParamKind::Number {
                    min: Some(0.0),
                    max: None,
                },
                required: true,
                default: Some(serde_json::json!(DEFAULT_SIMPLIFY_TOLERANCE_DEG)),
            },
        ],
    });
    registry.register(ToolDescriptor {
        id: "convex_hull".to_string(),
        title: "Convex hull".to_string(),
        description: "Computes the smallest convex polygon containing every \
                      vertex of every feature in the layer, as a single \
                      polygon feature. Purely planar — longitude and latitude \
                      are treated as x and y, so a layer straddling the \
                      antimeridian hulls the long way round. A layer with \
                      fewer than three distinct vertices, or whose vertices \
                      are all collinear, bounds no area and is refused."
            .to_string(),
        params: single_layer_param(),
    });
    registry.register(ToolDescriptor {
        id: "buffer".to_string(),
        title: "Buffer (points and lines)".to_string(),
        // Both caveats are stated here rather than left to a README because
        // this string is what the Processing panel prints above the form —
        // the only place the user reads them BEFORE running the tool.
        description: "Draws a polygon at a fixed distance around every point and line \
                      feature. The distance is in DEGREES of longitude/latitude, not \
                      metres, so a buffer covers less ground the further it sits from \
                      the equator. Polygon and multipolygon features are skipped: \
                      buffering an area needs a planar union to dissolve the overlaps \
                      that offsetting its rings produces, which this build does not have."
            .to_string(),
        params: vec![
            layer_param(),
            ParamSpec {
                name: "distance_deg".to_string(),
                kind: ParamKind::Number {
                    min: Some(0.0),
                    max: None,
                },
                required: true,
                default: Some(serde_json::json!(0.001)),
            },
            ParamSpec {
                name: "quadrant_segments".to_string(),
                kind: ParamKind::Number {
                    min: Some(1.0),
                    // Bounded so the field draws as a slider and cannot ask
                    // for an unbounded vertex count. Matches
                    // `oxigis_ui::processing_exec`'s own ceiling; the executor
                    // clamps to it either way.
                    max: Some(MAX_BUFFER_QUADRANT_SEGMENTS),
                },
                required: false,
                default: Some(serde_json::json!(DEFAULT_BUFFER_QUADRANT_SEGMENTS)),
            },
        ],
    });
    registry
}

/// Default arc samples per quadrant for the `buffer` tool's round joins.
const DEFAULT_BUFFER_QUADRANT_SEGMENTS: f64 = 8.0;

/// Ceiling on the `buffer` tool's arc samples per quadrant.
const MAX_BUFFER_QUADRANT_SEGMENTS: f64 = 64.0;

/// Execution context passed to a [`ToolExecutor`].
///
/// Carries resolved parameter *values*, keyed by [`ParamSpec::name`]: a
/// [`ParamKind::Number`]/`Text`/`Bool`/`Choice` parameter's value lives
/// directly in [`Self::params`]. [`ParamKind::LayerRef`] is the one
/// exception — `oxigis-core` holds no [`crate::layer::LayerStack`] (or any
/// OxiGeo type) to resolve it against, so looking up the referenced layer,
/// and reading its features, is the host's job, done *before* a
/// [`ToolExecutor`] is even constructed. `oxigis-ui`'s `processing_exec`
/// module is the reference: each built-in tool there is a struct built with
/// its target layer's data already resolved, and reads [`Self::params`]
/// only for its non-layer parameters (e.g. `simplify`'s `tolerance_deg`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolContext {
    /// Resolved parameter values supplied by the form, keyed by
    /// [`ParamSpec::name`].
    pub params: BTreeMap<String, serde_json::Value>,
}

impl ToolContext {
    /// Creates an empty execution context.
    pub fn new() -> Self {
        Self {
            params: BTreeMap::new(),
        }
    }
}

/// Something that can execute a processing tool given its resolved
/// parameters.
///
/// `oxigis-core` defines only this contract; every built-in tool id
/// (`bounds`, `feature_count`, `centroid`, `simplify`, `convex_hull`) is
/// implemented against it in `oxigis-ui`'s `processing_exec` module, which
/// dispatches [`ToolDescriptor::id`] to the matching `ToolExecutor`.
pub trait ToolExecutor {
    /// Runs the tool against `context`, returning a JSON result value or a
    /// [`CoreError`] describing why it could not run.
    fn run(&self, context: &ToolContext) -> Result<serde_json::Value, CoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_registry_has_every_shipped_tool() {
        let registry = builtin_registry();
        assert_eq!(registry.len(), 6);
        for (id, title) in [
            ("bounds", "Layer Bounds"),
            ("feature_count", "Feature Count"),
            ("centroid", "Centroids"),
            ("simplify", "Simplify (Douglas-Peucker)"),
            ("convex_hull", "Convex hull"),
            // Moved here from `oxigis-ui`, where it was registered only once
            // the Processing window had been opened — so a tool the panel
            // could not see until then. The executor still lives in
            // `oxigis-ui`; only the vocabulary is core's.
            ("buffer", "Buffer (points and lines)"),
        ] {
            assert_eq!(registry.get(id).expect("registered").title, title);
        }
    }

    #[test]
    fn the_buffer_descriptor_states_both_of_its_limitations() {
        // This string is what the Processing panel prints above the form, and
        // it is the only place the user reads either caveat BEFORE running the
        // tool — which is why they are asserted rather than left to prose.
        let registry = builtin_registry();
        let buffer = registry.get("buffer").expect("registered");
        assert!(
            buffer.description.contains("DEGREES"),
            "the distance unit must be stated: {}",
            buffer.description
        );
        assert!(
            buffer.description.contains("planar union"),
            "refusing areas must be explained: {}",
            buffer.description
        );
        let names: Vec<&str> = buffer.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["layer", "distance_deg", "quadrant_segments"]);
    }

    #[test]
    fn every_single_layer_tool_takes_exactly_one_required_layer_ref() {
        let registry = builtin_registry();
        for id in ["bounds", "feature_count", "centroid", "convex_hull"] {
            let params = &registry.get(id).expect("registered").params;
            assert_eq!(params.len(), 1, "{id} must take exactly one parameter");
            assert_eq!(params[0].name, "layer");
            assert_eq!(params[0].kind, ParamKind::LayerRef);
            assert!(params[0].required);
            assert_eq!(params[0].default, None);
        }
    }

    #[test]
    fn simplify_takes_a_layer_and_a_lower_bounded_unbounded_above_tolerance() {
        let registry = builtin_registry();
        let params = &registry.get("simplify").expect("registered").params;
        assert_eq!(params.len(), 2);

        assert_eq!(params[0].name, "layer");
        assert_eq!(params[0].kind, ParamKind::LayerRef);

        assert_eq!(params[1].name, "tolerance_deg");
        // `max: None` is deliberate: it is what makes the panel draw a
        // `DragValue` rather than a `Slider`, since no upper bound is
        // meaningful for a degree tolerance.
        assert_eq!(
            params[1].kind,
            ParamKind::Number {
                min: Some(0.0),
                max: None,
            }
        );
        assert!(params[1].required);
        assert_eq!(params[1].default, Some(serde_json::json!(0.001)));
    }

    #[test]
    fn centroid_description_documents_the_vertex_mean_caveat() {
        let registry = builtin_registry();
        let description = &registry.get("centroid").expect("registered").description;
        assert!(
            description.contains("vertex-mean"),
            "the vertex-mean approximation must be documented: {description}"
        );
        assert!(
            description.contains("area-weighted"),
            "the description must say what it is *not*: {description}"
        );
    }

    #[test]
    fn get_reports_unknown_tool_without_panicking() {
        let registry = builtin_registry();
        let err = registry
            .get("does_not_exist")
            .expect_err("should be unknown");
        assert_eq!(err, CoreError::UnknownTool("does_not_exist".to_string()));
    }

    #[test]
    fn register_overwrites_and_returns_prior_descriptor() {
        let mut registry = ProcessingRegistry::new();
        let first = ToolDescriptor {
            id: "demo".to_string(),
            title: "Demo v1".to_string(),
            description: "first".to_string(),
            params: vec![],
        };
        let second = ToolDescriptor {
            id: "demo".to_string(),
            title: "Demo v2".to_string(),
            description: "second".to_string(),
            params: vec![],
        };
        assert_eq!(registry.register(first.clone()), None);
        let replaced = registry.register(second.clone());
        assert_eq!(replaced, Some(first));
        assert_eq!(registry.get("demo").expect("present").title, "Demo v2");
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn iter_yields_descriptors_ordered_by_id() {
        let registry = builtin_registry();
        let ids: Vec<&str> = registry.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "bounds",
                "buffer",
                "centroid",
                "convex_hull",
                "feature_count",
                "simplify"
            ]
        );
    }

    #[test]
    fn empty_registry_reports_empty() {
        let registry = ProcessingRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn param_kind_serde_tag_shape_round_trips() {
        let kind = ParamKind::Number {
            min: Some(0.0),
            max: Some(1.0),
        };
        let value = serde_json::to_value(&kind).expect("serialize");
        assert_eq!(value["kind"], "number");
        assert_eq!(value["value"]["min"], 0.0);
        assert_eq!(value["value"]["max"], 1.0);
        let back: ParamKind = serde_json::from_value(value).expect("deserialize");
        assert_eq!(back, kind);

        let choice = ParamKind::Choice(vec!["a".to_string(), "b".to_string()]);
        let value = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(value["kind"], "choice");
        assert_eq!(value["value"], serde_json::json!(["a", "b"]));
        let back: ParamKind = serde_json::from_value(value).expect("deserialize");
        assert_eq!(back, choice);

        // Unit variants carry no "value" key under adjacent tagging.
        let value = serde_json::to_value(&ParamKind::Text).expect("serialize");
        assert_eq!(value, serde_json::json!({"kind": "text"}));
        let back: ParamKind = serde_json::from_value(value).expect("deserialize");
        assert_eq!(back, ParamKind::Text);
    }

    #[test]
    fn tool_descriptor_json_round_trips() {
        let registry = builtin_registry();
        let bounds = registry.get("bounds").expect("registered");
        let json = serde_json::to_string(bounds).expect("serialize");
        let back: ToolDescriptor = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&back, bounds);
    }

    /// A trivial [`ToolExecutor`] used only to prove the trait is usable
    /// end to end (not a built-in tool).
    struct EchoLayerParam;

    impl ToolExecutor for EchoLayerParam {
        fn run(&self, context: &ToolContext) -> Result<serde_json::Value, CoreError> {
            context
                .params
                .get("layer")
                .cloned()
                .ok_or_else(|| CoreError::InvalidParameter {
                    name: "layer".to_string(),
                    reason: "missing required parameter".to_string(),
                })
        }
    }

    #[test]
    fn tool_executor_trait_is_usable() {
        let mut context = ToolContext::new();
        context
            .params
            .insert("layer".to_string(), serde_json::json!(1));
        let result = EchoLayerParam.run(&context).expect("layer param present");
        assert_eq!(result, serde_json::json!(1));

        let empty_context = ToolContext::new();
        assert!(EchoLayerParam.run(&empty_context).is_err());
    }
}
