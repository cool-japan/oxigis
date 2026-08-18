// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The style panel's app-side glue: the attribute view the renderer editor
//! reads, and the panel body itself.
//!
//! Split from `app/mod.rs` under the 2000-line rule, beside its siblings
//! `app::print_io` and `app::archive_io`. The editor itself is
//! [`crate::renderer_panel`] — pure, testable and shell-free; what lives here
//! is only what needs the *app*: which layer is selected, where its features
//! are, and what a tiled layer is told instead.
//!
//! # Why the field list is cached
//!
//! The renderer's field picker offers the layer's attribute keys, which come
//! from [`AttributeSchema::derive`] — a walk over every feature and every
//! property. The style panel is drawn EVERY FRAME, so deriving per frame would
//! put an `O(features × properties)` scan on the frame budget of any project
//! with the panel open. [`RendererFields`] therefore keys the derived list on
//! the collection's `Arc` identity, exactly as
//! [`crate::table_panel::AttributeTablePanel`] keys its own binding: the same
//! `Arc` is the same data, and every path that edits features publishes a new
//! `Arc`, so a stale list is unrepresentable rather than merely unlikely.
//!
//! # Why nothing here records an undo step
//!
//! Every renderer edit — a colour, a class, a whole re-classification —
//! mutates the [`oxigis_core::LayerStyleSet`] the caller already owns, which
//! `OxigisApp::sync_local_style_gated` observes as a before/after pair and
//! records through the existing `CoalesceField::Style` path. The ONE thing
//! this module reports upward is whether the change moved the class LIST, so
//! the caller can keep a structural change out of a colour drag's coalescing
//! window: one click, one undo step.

use std::sync::Arc;

use oxigeo::geojson::types::FeatureCollection;
use oxigis_core::{LayerId, LayerStyleSet};

use super::OxigisApp;
use crate::attribute_table::AttributeSchema;
use crate::renderer_panel::{LayerAttributes, RendererEvent};
use crate::style_panel::{self, StyleKind};

/// One layer's attribute keys, remembered against the collection they were
/// derived from.
///
/// The `Arc` is held, not merely compared: keeping it alive is what makes
/// [`Arc::ptr_eq`] a sound identity test — a dropped collection could otherwise
/// have its address reused by the next one, and the cache would then answer for
/// the wrong data.
#[derive(Debug)]
pub(super) struct RendererFields {
    /// The layer the keys belong to.
    layer: LayerId,
    /// The exact collection they were derived from.
    features: Arc<FeatureCollection>,
    /// The keys themselves, capped at
    /// [`crate::attribute_table::MAX_PROPERTY_COLUMNS`] by the schema.
    keys: Vec<String>,
}

impl RendererFields {
    /// Whether this list still describes `layer`'s current features.
    fn matches(&self, layer: LayerId, features: &Arc<FeatureCollection>) -> bool {
        self.layer == layer && Arc::ptr_eq(&self.features, features)
    }

    /// The keys the field picker is offering, and the layer they belong to.
    ///
    /// The cache is an optimisation with no product surface of its own, so its
    /// only reader is the test that pins BOTH of its obligations — one
    /// derivation per collection, and never the wrong layer's columns.
    #[cfg(test)]
    pub(super) fn offered(&self) -> (LayerId, &[String]) {
        (self.layer, &self.keys)
    }
}

impl OxigisApp {
    /// Makes sure the cached field list describes `layer`'s current features,
    /// deriving it only when the collection it was built from is gone.
    fn refresh_renderer_fields(&mut self, layer: LayerId, features: &Arc<FeatureCollection>) {
        if self
            .renderer_fields
            .as_ref()
            .is_some_and(|cached| cached.matches(layer, features))
        {
            return;
        }
        self.renderer_fields = Some(RendererFields {
            layer,
            features: Arc::clone(features),
            keys: AttributeSchema::derive(features).property_keys().to_vec(),
        });
    }

    /// Draws the style panel's body for the current selection, returning
    /// whether the renderer editor moved the CLASS LIST this frame.
    ///
    /// `styleable` is the caller's verdict on whether `project.styles` decides
    /// this layer's drawing at all — false for a tiled layer, which paints from
    /// its source's own rules.
    pub(super) fn style_panel_body(&mut self, ui: &mut egui::Ui, styleable: bool) -> bool {
        let selection = self.selection;
        if selection.is_some() && !styleable {
            return self.tiled_style_body(ui);
        }
        // Resolved BEFORE `&mut self.project.styles` is taken: `feature_set`
        // hands back a borrow of `self.local`, which would still be live across
        // the `get_mut` below. Cloning the `Arc` is a refcount bump.
        let features = selection.and_then(|id| self.local.feature_set(id)).cloned();
        if let (Some(id), Some(features)) = (selection, features.as_ref()) {
            self.refresh_renderer_fields(id, features);
        }
        let families = selection
            .map(|id| self.local.families(id))
            .unwrap_or_default();
        let mut state = self.style_panel_state;
        // Three DIRECT field projections, deliberately not three method calls:
        // the borrow checker splits `self.project`, `self.renderer_fields` and
        // `self.renderer_panel_state` into disjoint borrows only when they are
        // named as fields, and going through methods would force the key list
        // to be cloned once per frame instead.
        let keys: &[String] = self
            .renderer_fields
            .as_ref()
            // The cache survives a selection change, so it has to be checked
            // against the CURRENT layer: offering the previous layer's fields
            // would let a click write a field name the data has never carried.
            .filter(|cached| Some(cached.layer) == selection)
            .map_or(&[], |cached| cached.keys.as_slice());
        let renderer_state = &mut self.renderer_panel_state;
        let style = selection.and_then(|id| self.project.styles.get_mut(&id));
        let (action, event) = style_panel::ui_with_renderer(
            ui,
            selection,
            style,
            families,
            &mut state,
            Some((
                renderer_state,
                LayerAttributes {
                    keys,
                    features: features.as_deref(),
                    refusal: None,
                },
            )),
        );
        self.style_panel_state = state;
        if let Some(action) = action {
            self.apply_style_action(action);
        }
        event == Some(RendererEvent::Repartitioned)
    }

    /// The panel a TILED layer gets: the notice, then the renderer combo drawn
    /// disabled with the reason on hover.
    ///
    /// A named refusal rather than a missing control. Without the combo, "this
    /// build has no thematic renderer" and "this LAYER cannot use one, because
    /// an MVT paint never sees a feature's attributes" look identical, and only
    /// the second is true. The set handed in is a scratch value the refused
    /// path provably never touches (see `renderer_panel`'s own
    /// `a_refused_layer_shows_the_reason_and_no_combo_at_all`), because a tiled
    /// layer HAS no `LayerStyleSet` — its rules live on the source.
    fn tiled_style_body(&mut self, ui: &mut egui::Ui) -> bool {
        Self::tiled_style_notice(ui);
        let mut scratch = LayerStyleSet::new(StyleKind::Fill.default_style());
        let event = crate::renderer_panel::ui(
            ui,
            &mut scratch,
            LayerAttributes::refused(crate::vector_provider::TILED_RENDERER_REFUSAL),
            &mut self.renderer_panel_state,
        );
        debug_assert!(event.is_none(), "a refused layer reports no repartition");
        false
    }
}
