// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The second command family (editing v1.1 stage 5 — see
//! docs/plans/editing-v11.md): reversible **project** operations — layer
//! add/remove, reorder, opacity and style — beside the feature-edit family
//! in [`super::command`].
//!
//! Pure model only: applying and recording live in the app glue, behind
//! one choke point, so recorder and applier are the same code and undo
//! symmetry is provable rather than a coincidence. Everything here is
//! egui-free and directly testable.
//!
//! # Why snapshots, not indices
//!
//! A removed layer is restored from a full [`LayerSnapshot`] — the
//! [`oxigis_core::Layer`] value (whose `kind` already carries the inline
//! GeoJSON, so a restore cannot fail on serialization), its stack slot, its
//! style entry and the exact features `Arc`. A reorder carries whole
//! before/after id orders rather than `(from, to)` index pairs — a
//! snapshot is idempotent and cannot corrupt an order it does not
//! describe.

use std::sync::Arc;

use oxigeo::geojson::types::FeatureCollection;
use oxigis_core::{Layer, LayerId, LayerStyleSet};

use super::command;
use super::stack::CoalesceKey;
use crate::tile_provider::BasemapConfig;

/// Both sides of a basemap-**service** change.
///
/// Boxed and behind ONE [`Option`] on [`ProjectOp::SetBasemap`], so "the
/// service changed on one side only" is UNREPRESENTABLE — a half-recorded
/// service is exactly the refusal editing v1.4 named (an undo that does not
/// restore what the redo set), and no runtime check can be trusted to hold an
/// invariant the type can hold for free.
///
/// The carried type is the **live** [`BasemapConfig`], not the serialized
/// [`oxigis_core::ProjectBasemap`] mirror: an op is a live-state record, never
/// persisted, so the applier converts once at the write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasemapServiceChange {
    /// The service the map was drawing before.
    pub before: BasemapConfig,
    /// The service the map draws after.
    pub after: BasemapConfig,
}

impl BasemapServiceChange {
    /// Roughly how much heap this change pins, both sides counted.
    ///
    /// It must actually be counted or the undo budget silently under-reports:
    /// a fat entry is two templates plus two credit lines plus their subdomain
    /// lists.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        fn side(config: &BasemapConfig) -> usize {
            config.url_template.len()
                + config.attribution.len()
                + config
                    .subdomains
                    .iter()
                    .map(|host| size_of::<String>() + host.len())
                    .sum::<usize>()
        }
        side(&self.before) + side(&self.after)
    }
}

/// Both sides of a layer **rename**.
///
/// Boxed behind [`ProjectOp::RenameLayer`]'s one field rather than inlined as
/// two `String`s, for the reason [`BasemapServiceChange`] is boxed: two inline
/// `String`s are 48 bytes, which would grow EVERY [`ProjectOp`] ever stored —
/// vertex drags included — to pay for the rarest gesture in the family. See
/// `the_op_did_not_grow`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerRename {
    /// The name the layer had before.
    pub before: String,
    /// The name it has after.
    pub after: String,
}

impl LayerRename {
    /// Roughly how much heap this change pins, both sides counted.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.before.len() + self.after.len()
    }
}

/// Everything needed to put a removed (or re-add an added) layer back
/// exactly: value, position, style, data.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSnapshot {
    /// The layer value — id, name, visibility, opacity and `kind` (which
    /// for an edited local layer already holds the inline GeoJSON).
    pub layer: Layer,
    /// The layer's slot in the stack's **storage** order (bottom-first).
    pub position: usize,
    /// The project's explicit style entry for it, if any.
    pub style: Option<LayerStyleSet>,
    /// The feature store's collection, for a local vector layer.
    pub features: Option<Arc<FeatureCollection>>,
    /// The geometry-derived style the layer was created with. Restored so
    /// that Style ▸ Remove after an undo still has a default to fall back to
    /// — `LocalInputState::forget` drops it with the layer, and without this
    /// a restored layer silently skipped the GPU restyle.
    pub default_style: Option<LayerStyleSet>,
}

impl LayerSnapshot {
    /// Roughly how much heap this snapshot pins. The features `Arc` is
    /// counted at full size deliberately: once the layer is removed the
    /// stack is its only owner, and budgeting for anything less would be
    /// unauditable.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        let kind_bytes = match &self.layer.kind {
            oxigis_core::LayerKind::Vector(oxigis_core::VectorSource::InlineGeoJson {
                geojson,
            }) => geojson.len(),
            _ => 0,
        };
        size_of::<Self>()
            + kind_bytes
            + self
                .features
                .as_ref()
                .map_or(0, |features| command::collection_bytes(features))
    }
}

/// One reversible project operation.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectOp {
    /// Layers were added — usually one; a multi-table drop (gpkg) records
    /// its whole gesture as ONE group, so one Ctrl+Z removes them all and a
    /// front-eviction can never bisect the gesture. Snapshots are stored
    /// **ascending by [`LayerSnapshot::position`]**: restoration walks them
    /// ascending (each lands on a stack one slot shorter than its own
    /// recorded slot needs), removal walks them descending — the feature
    /// family's descending-`Remove` rule. Undo removes them all, redo
    /// restores the snapshots.
    AddLayers(Vec<LayerSnapshot>),
    /// Layers were removed (undo restores the snapshots, redo removes them).
    /// The same group invariant as [`Self::AddLayers`].
    RemoveLayers(Vec<LayerSnapshot>),
    /// The stack order changed. Whole orders, storage order, so applying
    /// either side is idempotent and immune to positional staleness.
    Reorder {
        /// The order before.
        before: Vec<LayerId>,
        /// The order after.
        after: Vec<LayerId>,
    },
    /// A layer's opacity changed.
    SetOpacity {
        /// Which layer.
        layer: LayerId,
        /// The opacity before.
        before: f32,
        /// The opacity after.
        after: f32,
    },
    /// A layer's visibility checkbox changed.
    ///
    /// Both sides are recorded ABSOLUTELY rather than as "it flipped": an
    /// applier built on a flip is only correct while nothing else writes the
    /// flag, and a redo replayed after a project reload would then invert the
    /// wrong way. Absolute sides make applying either direction idempotent —
    /// the property [`Self::Reorder`] is built on for the same reason.
    SetVisibility {
        /// Which layer.
        layer: LayerId,
        /// Whether it was drawn before.
        before: bool,
        /// Whether it is drawn after.
        after: bool,
    },
    /// A layer was renamed. The names are boxed as ONE pair (see
    /// [`LayerRename`]) so this rare variant does not size every stored op.
    RenameLayer {
        /// Which layer.
        layer: LayerId,
        /// Both sides of the rename.
        names: Box<LayerRename>,
    },
    /// A layer's scale range — the zooms it draws between — changed.
    ///
    /// BOTH ends on BOTH sides, in one op, for the reason
    /// [`Self::SetBasemap`]'s service pair is one boxed field: a range whose
    /// halves could be recorded separately would let an undo restore one bound
    /// and not the other, which is an undo that does not restore what the redo
    /// set. Four `Option<f32>`s inline still fit inside the enum's existing
    /// size, so unlike a rename there is nothing to box.
    SetZoomRange {
        /// Which layer.
        layer: LayerId,
        /// The range before, `(min, max)`.
        before: (Option<f32>, Option<f32>),
        /// The range after, `(min, max)`.
        after: (Option<f32>, Option<f32>),
    },
    /// The layer *promoted* to draw as the basemap changed ([`None`] = the
    /// basemap service draws), and/or the **service** itself.
    ///
    /// Historical rationale, kept because it is the reason this variant was
    /// pointer-only for four releases: a service-carrying variant would be
    /// unsound while the basemap picker's own changes stay unrecorded (editing
    /// v1.1 decision 9): `promote → pick a preset (unrecorded) → Ctrl+Z` would
    /// write the pre-pick service back over the user's choice — an undo
    /// silently reverting something that was never on the stack, and
    /// unrecoverable by a second Ctrl+Z. With a pointer that is
    /// unrepresentable. Recording every service change through this same op is
    /// the forward path.
    ///
    /// **Superseded by editing v1.5; the forward condition is met** — see the
    /// writer census in `docs/plans/editing-v15.md` D9. Every writer of the
    /// service that a Ctrl+Z can jump over now records through this op, so the
    /// unsound sequence is unrepresentable *because the pick is on the stack*.
    /// A pick made while a layer is promoted demotes and swaps the service in
    /// one click, and a [`ProjectTransaction`] holds exactly one op, which is
    /// why the service rides here rather than in a sibling variant.
    SetBasemap {
        /// The layer that drew as the basemap before.
        before: Option<LayerId>,
        /// The layer that draws as the basemap after.
        after: Option<LayerId>,
        /// The service this step also changed, or [`None`] for a pure
        /// promote/demote/swap — which keeps this variant's original
        /// zero-heap property exactly.
        service: Option<Box<BasemapServiceChange>>,
    },
    /// A layer's explicit style entry changed (either side [`None`] =
    /// no entry, i.e. the format-default style). The sets are boxed so this
    /// rare variant does not size every [`ProjectOp`] ever stored — a
    /// [`LayerStyleSet`] carries a base style plus a family map.
    SetStyle {
        /// Which layer.
        layer: LayerId,
        /// The entry before.
        before: Option<Box<LayerStyleSet>>,
        /// The entry after.
        after: Option<Box<LayerStyleSet>>,
    },
}

impl ProjectOp {
    /// The operation that exactly undoes this one.
    ///
    /// For the grouped variants the SAME vector swaps constructors, so double
    /// inversion is syntactic identity for every group size.
    #[must_use]
    pub fn inverted(&self) -> Self {
        match self {
            Self::AddLayers(snapshots) => Self::RemoveLayers(snapshots.clone()),
            Self::RemoveLayers(snapshots) => Self::AddLayers(snapshots.clone()),
            Self::Reorder { before, after } => Self::Reorder {
                before: after.clone(),
                after: before.clone(),
            },
            Self::SetOpacity {
                layer,
                before,
                after,
            } => Self::SetOpacity {
                layer: *layer,
                before: *after,
                after: *before,
            },
            Self::SetVisibility {
                layer,
                before,
                after,
            } => Self::SetVisibility {
                layer: *layer,
                before: *after,
                after: *before,
            },
            Self::RenameLayer { layer, names } => Self::RenameLayer {
                layer: *layer,
                names: Box::new(LayerRename {
                    before: names.after.clone(),
                    after: names.before.clone(),
                }),
            },
            Self::SetZoomRange {
                layer,
                before,
                after,
            } => Self::SetZoomRange {
                layer: *layer,
                before: *after,
                after: *before,
            },
            Self::SetBasemap {
                before,
                after,
                service,
            } => Self::SetBasemap {
                before: *after,
                after: *before,
                // The `Option` is preserved and the inner pair swaps, so
                // double inversion is syntactic identity for every shape.
                service: service.as_ref().map(|change| {
                    Box::new(BasemapServiceChange {
                        before: change.after.clone(),
                        after: change.before.clone(),
                    })
                }),
            },
            Self::SetStyle {
                layer,
                before,
                after,
            } => Self::SetStyle {
                layer: *layer,
                before: after.clone(),
                after: before.clone(),
            },
        }
    }

    /// The layer this operation is about, when it names exactly one.
    #[must_use]
    pub fn layer(&self) -> Option<LayerId> {
        match self {
            Self::AddLayers(snapshots) | Self::RemoveLayers(snapshots) => {
                match snapshots.as_slice() {
                    [only] => Some(only.layer.id),
                    _ => None,
                }
            }
            Self::Reorder { .. } => None,
            // A promote or a demote is about the one layer it names; a swap
            // between two promoted layers is not about ONE of them, the same
            // rule the grouped variants follow. A service change is not about
            // a layer at all, so it does not enter this answer.
            Self::SetBasemap { before, after, .. } => match (before, after) {
                (Some(only), None) | (None, Some(only)) => Some(*only),
                _ => None,
            },
            Self::SetOpacity { layer, .. }
            | Self::SetStyle { layer, .. }
            | Self::SetVisibility { layer, .. }
            | Self::RenameLayer { layer, .. }
            | Self::SetZoomRange { layer, .. } => Some(*layer),
        }
    }

    /// Whether this operation names `layer` at all — what
    /// [`super::stack::EditStack::prune_layer`] asks. Load-bearing for the
    /// grouped variants: a hydrate of ONE member must drop the whole group
    /// entry, or a redo would splice a stale snapshot back over the re-read
    /// file.
    #[must_use]
    pub fn mentions_layer(&self, layer: LayerId) -> bool {
        match self {
            Self::AddLayers(snapshots) | Self::RemoveLayers(snapshots) => {
                snapshots.iter().any(|snapshot| snapshot.layer.id == layer)
            }
            Self::Reorder { .. } => false,
            // BOTH sides, and ONLY the pointer sides: a hydrate that replaces
            // the layer must drop an entry that would promote it as well as
            // one that demotes it, while a service-only entry
            // (`before: None, after: None, service: Some(_)`) must mention NO
            // layer — otherwise a hydrate would silently erase basemap
            // history.
            Self::SetBasemap { before, after, .. } => {
                *before == Some(layer) || *after == Some(layer)
            }
            Self::SetOpacity { layer: named, .. }
            | Self::SetStyle { layer: named, .. }
            | Self::SetVisibility { layer: named, .. }
            | Self::RenameLayer { layer: named, .. }
            | Self::SetZoomRange { layer: named, .. } => *named == layer,
        }
    }

    /// Roughly how much heap this operation pins.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        match self {
            Self::AddLayers(snapshots) | Self::RemoveLayers(snapshots) => snapshots
                .iter()
                .map(LayerSnapshot::estimated_bytes)
                .sum::<usize>(),
            Self::Reorder { before, after } => {
                size_of::<Self>() + (before.len() + after.len()) * size_of::<LayerId>()
            }
            // Its own arm: a pointer-only change still pins zero heap — two
            // `Option<LayerId>`s and nothing else — so this variant can never
            // grow a budget it does not spend, while a service-carrying one
            // pays for exactly the strings it holds.
            Self::SetBasemap { service, .. } => {
                size_of::<Self>()
                    + service
                        .as_deref()
                        .map_or(0, BasemapServiceChange::heap_bytes)
            }
            // Its own arm, like `SetBasemap`'s: a checkbox change is two
            // `bool`s and an id, so it pins no heap at all and must never be
            // charged a budget it does not spend — a hundred toggles are not
            // 25 KiB of history.
            // Two bools / four `Option<f32>`s and an id: no heap either way,
            // and neither may be charged a budget it does not spend.
            Self::SetVisibility { .. } | Self::SetZoomRange { .. } => size_of::<Self>(),
            // Exactly the two strings it holds, so a rename of a layer named
            // after a long path is budgeted for what it really costs.
            Self::RenameLayer { names, .. } => size_of::<Self>() + names.heap_bytes(),
            Self::SetOpacity { .. } | Self::SetStyle { .. } => size_of::<Self>() + 256,
        }
    }
}

/// One undo step of the project family.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectTransaction {
    /// Menu/status wording, e.g. `"Remove layer"`.
    pub label: &'static str,
    /// The operation.
    pub op: ProjectOp,
    /// Gestures that may fold into the previous entry (opacity/style
    /// drags) share a key; [`None`] never coalesces.
    pub coalesce: Option<CoalesceKey>,
}

impl ProjectTransaction {
    /// The transaction that exactly undoes this one.
    #[must_use]
    pub fn inverted(&self) -> Self {
        Self {
            label: self.label,
            op: self.op.inverted(),
            coalesce: None,
        }
    }

    /// Roughly how much heap this entry pins.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        size_of::<Self>() + self.op.estimated_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigis_core::{LayerKind, VectorSource};

    /// A service change with heap on BOTH sides — a template, a credit
    /// line and a subdomain list — so the byte accounting has something to
    /// count.
    fn service_change() -> BasemapServiceChange {
        BasemapServiceChange {
            before: BasemapConfig::openstreetmap(),
            after: BasemapConfig {
                url_template: "https://{s}.example.test/tiles/{z}/{x}/{y}.png".to_string(),
                subdomains: vec!["a".to_string(), "b".to_string()],
                attribution: "\u{a9} Example".to_string(),
            },
        }
    }

    fn snapshot() -> LayerSnapshot {
        LayerSnapshot {
            layer: Layer::new(
                "inline",
                LayerKind::Vector(VectorSource::InlineGeoJson {
                    geojson: "{\"type\":\"FeatureCollection\",\"features\":[]}".to_string(),
                }),
            ),
            position: 1,
            style: None,
            features: Some(Arc::new(FeatureCollection::new(Vec::new()))),
            default_style: None,
        }
    }

    #[test]
    fn every_op_inverts_to_its_exact_opposite_and_back() {
        let ops = [
            // The old singular contract is the N = 1 case of the group.
            ProjectOp::AddLayers(vec![snapshot()]),
            ProjectOp::RemoveLayers(vec![snapshot()]),
            // A multi-table drop's whole gesture: N = 3.
            ProjectOp::AddLayers(vec![snapshot(), snapshot(), snapshot()]),
            ProjectOp::RemoveLayers(vec![snapshot(), snapshot(), snapshot()]),
            ProjectOp::Reorder {
                before: Vec::new(),
                after: Vec::new(),
            },
            ProjectOp::SetOpacity {
                layer: snapshot().layer.id,
                before: 1.0,
                after: 0.5,
            },
            ProjectOp::SetStyle {
                layer: snapshot().layer.id,
                before: None,
                after: None,
            },
            // Both directions of a checkbox, and the degenerate equal shape a
            // gesture never records.
            ProjectOp::SetVisibility {
                layer: snapshot().layer.id,
                before: true,
                after: false,
            },
            ProjectOp::SetVisibility {
                layer: snapshot().layer.id,
                before: false,
                after: true,
            },
            ProjectOp::SetVisibility {
                layer: snapshot().layer.id,
                before: true,
                after: true,
            },
            ProjectOp::RenameLayer {
                layer: snapshot().layer.id,
                names: Box::new(LayerRename {
                    before: "Roads".to_string(),
                    after: "Roads (2024)".to_string(),
                }),
            },
            // An empty side on each end: a rename FROM and TO nothing must
            // still invert to syntactic identity.
            ProjectOp::RenameLayer {
                layer: snapshot().layer.id,
                names: Box::new(LayerRename {
                    before: String::new(),
                    after: "Named at last".to_string(),
                }),
            },
            // All four POINTER shapes of a basemap change: promote, demote,
            // swap and the degenerate no-op a gesture never records.
            ProjectOp::SetBasemap {
                before: None,
                after: Some(snapshot().layer.id),
                service: None,
            },
            ProjectOp::SetBasemap {
                before: Some(snapshot().layer.id),
                after: None,
                service: None,
            },
            ProjectOp::SetBasemap {
                before: Some(snapshot().layer.id),
                after: Some(snapshot().layer.id),
                service: None,
            },
            ProjectOp::SetBasemap {
                before: None,
                after: None,
                service: None,
            },
            // And the four SERVICE shapes editing v1.5 adds: service-only,
            // promote+service, demote+service, and the degenerate
            // equal-service shape a gesture never records either.
            ProjectOp::SetBasemap {
                before: None,
                after: None,
                service: Some(Box::new(service_change())),
            },
            ProjectOp::SetBasemap {
                before: None,
                after: Some(snapshot().layer.id),
                service: Some(Box::new(service_change())),
            },
            ProjectOp::SetBasemap {
                before: Some(snapshot().layer.id),
                after: None,
                service: Some(Box::new(service_change())),
            },
            ProjectOp::SetBasemap {
                before: None,
                after: None,
                service: Some(Box::new(BasemapServiceChange {
                    before: BasemapConfig::openstreetmap(),
                    after: BasemapConfig::openstreetmap(),
                })),
            },
        ];
        for op in ops {
            assert_eq!(op.inverted().inverted(), op);
        }
        // Add and Remove are each other's inverses.
        let add = ProjectOp::AddLayers(vec![snapshot()]);
        assert!(matches!(add.inverted(), ProjectOp::RemoveLayers(_)));
    }

    #[test]
    fn a_group_names_every_member_but_answers_layer_only_when_singular() {
        let first = snapshot();
        let second = snapshot();
        let first_id = first.layer.id;
        let second_id = second.layer.id;
        let single = ProjectOp::AddLayers(vec![first.clone()]);
        assert_eq!(single.layer(), Some(first_id));
        assert!(single.mentions_layer(first_id));
        assert!(!single.mentions_layer(second_id));

        let group = ProjectOp::AddLayers(vec![first, second]);
        assert_eq!(group.layer(), None, "a group is not about ONE layer");
        assert!(
            group.mentions_layer(first_id) && group.mentions_layer(second_id),
            "pruning any member must see the whole group",
        );
        // A group's bytes are the sum of its members'.
        assert!(group.estimated_bytes() >= 2 * snapshot().estimated_bytes());
    }

    #[test]
    fn a_basemap_change_names_both_its_sides_and_pins_no_heap() {
        let promoted = snapshot().layer.id;
        let other = snapshot().layer.id;
        let promote = ProjectOp::SetBasemap {
            before: None,
            after: Some(promoted),
            service: None,
        };
        assert_eq!(promote.layer(), Some(promoted));
        let swap = ProjectOp::SetBasemap {
            before: Some(promoted),
            after: Some(other),
            service: None,
        };
        assert_eq!(swap.layer(), None, "a swap is not about ONE layer");
        assert!(
            swap.mentions_layer(promoted) && swap.mentions_layer(other),
            "pruning either side must see the whole entry"
        );
        assert!(!promote.mentions_layer(other));
        assert_eq!(
            promote.estimated_bytes(),
            size_of::<ProjectOp>(),
            "two pointers, no heap"
        );
        // Nothing this small can re-trigger `large_enum_variant`: the payload
        // is smaller than `Reorder`'s two vectors.
        assert!(size_of::<Option<LayerId>>() * 2 <= size_of::<ProjectOp>());
    }

    #[test]
    fn a_service_carrying_basemap_change_counts_both_sides_and_names_no_layer() {
        let change = service_change();
        let op = ProjectOp::SetBasemap {
            before: None,
            after: None,
            service: Some(Box::new(change.clone())),
        };
        assert!(
            op.estimated_bytes() > size_of::<ProjectOp>(),
            "a service change pins heap and must be budgeted for it"
        );
        assert_eq!(
            op.estimated_bytes(),
            size_of::<ProjectOp>() + change.heap_bytes()
        );
        // Template, credit line and subdomain list, on BOTH sides.
        let counted = change.heap_bytes();
        assert!(counted > change.before.url_template.len() + change.after.url_template.len());
        assert!(counted > change.after.attribution.len());
        assert!(
            counted
                >= change
                    .after
                    .subdomains
                    .iter()
                    .map(String::len)
                    .sum::<usize>(),
            "the subdomain list is part of the service"
        );
        // A service-only entry is about NO layer, so a hydrate must not drop
        // it: `mentions_layer` reads the pointer sides and only those.
        let layer = snapshot().layer.id;
        assert_eq!(op.layer(), None);
        assert!(!op.mentions_layer(layer));
        // A fused entry still names the one layer it demoted.
        let fused = ProjectOp::SetBasemap {
            before: Some(layer),
            after: None,
            service: Some(Box::new(change)),
        };
        assert_eq!(fused.layer(), Some(layer));
        assert!(fused.mentions_layer(layer));
    }

    #[test]
    fn a_visibility_change_names_its_layer_and_pins_no_heap() {
        let layer = snapshot().layer.id;
        let other = snapshot().layer.id;
        let op = ProjectOp::SetVisibility {
            layer,
            before: true,
            after: false,
        };
        assert_eq!(op.layer(), Some(layer));
        assert!(op.mentions_layer(layer));
        assert!(!op.mentions_layer(other));
        assert_eq!(
            op.estimated_bytes(),
            size_of::<ProjectOp>(),
            "an id and two bools, no heap"
        );
        // Inversion is the checkbox's other position, and doing it twice is
        // syntactic identity — what makes redo exact.
        assert_eq!(
            op.inverted(),
            ProjectOp::SetVisibility {
                layer,
                before: false,
                after: true,
            }
        );
    }

    #[test]
    fn a_rename_names_its_layer_and_counts_both_of_its_strings() {
        let layer = snapshot().layer.id;
        let other = snapshot().layer.id;
        let names = LayerRename {
            before: "roads".to_string(),
            after: "Roads of Tokyo, 2024 revision".to_string(),
        };
        let op = ProjectOp::RenameLayer {
            layer,
            names: Box::new(names.clone()),
        };
        assert_eq!(op.layer(), Some(layer));
        assert!(op.mentions_layer(layer));
        assert!(!op.mentions_layer(other));
        assert_eq!(
            op.estimated_bytes(),
            size_of::<ProjectOp>() + names.heap_bytes()
        );
        assert!(
            names.heap_bytes() > names.after.len(),
            "BOTH sides are counted, or an undo budget under-reports a rename"
        );
        // The inverse really is the other direction, not a copy.
        match op.inverted() {
            ProjectOp::RenameLayer {
                layer: inverted,
                names,
            } => {
                assert_eq!(inverted, layer);
                assert_eq!(names.before, "Roads of Tokyo, 2024 revision");
                assert_eq!(names.after, "roads");
            }
            other => panic!("expected a rename, got {other:?}"),
        }
    }

    #[test]
    fn the_op_did_not_grow() {
        // The anti-regression on the boxing: an inline `Option<BasemapConfig>`
        // on both sides would size EVERY undo entry ever stored at ~3.7x this.
        // The same pin now also holds `RenameLayer`'s pair to one box — two
        // inline `String`s are 48 bytes on their own and would push this to 64.
        assert_eq!(
            size_of::<ProjectOp>(),
            48,
            "a service-carrying basemap op and a rename must cost the enum nothing"
        );
    }

    #[test]
    fn a_snapshot_counts_its_inline_text_and_its_pinned_collection() {
        let bytes = snapshot().estimated_bytes();
        assert!(
            bytes > "{\"type\":\"FeatureCollection\",\"features\":[]}".len(),
            "the inline text must be counted: {bytes}"
        );
    }
}
