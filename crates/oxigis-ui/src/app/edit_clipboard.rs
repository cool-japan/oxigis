// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Clipboard half of the editing glue: Copy / Cut / Paste event handling
//! over [`crate::edit::clipboard`] — split from `app/edit_glue.rs` under
//! the 2000-line rule (a pure move, no behaviour change).

use super::OxigisApp;
use crate::edit::command::{self, EditError, EditTransaction};
use crate::edit::{self as edit, EditMode, EditNotice, EditSelection, selection};
use egui::Context;
use std::sync::Arc;

impl OxigisApp {
    /// Copy / Cut / Paste, driven by egui's clipboard **events** — never a
    /// `KeyboardShortcut`: egui-winit intercepts Ctrl+C/X/V and pushes
    /// `Event::Copy`/`Cut`/`Paste` without ever emitting a `Key` event, and
    /// the web runner does the same from the browser's own clipboard
    /// listeners (which is also what makes Safari's write-in-callback rule
    /// hold). Claimed only while editing is on and a local vector layer is
    /// the target — in `Browse` the events stay available to egui's label
    /// text selection.
    pub(super) fn edit_clipboard_events(&mut self, ctx: &Context) -> bool {
        if self.edit.mode() == EditMode::Off {
            return false;
        }
        let (copy, cut, pasted) = ctx.input(|input| {
            let mut copy = false;
            let mut cut = false;
            let mut pasted: Option<String> = None;
            for event in &input.events {
                match event {
                    egui::Event::Copy => copy = true,
                    egui::Event::Cut => cut = true,
                    egui::Event::Paste(text) => pasted = Some(text.clone()),
                    _ => {}
                }
            }
            (copy, cut, pasted)
        });
        if copy || cut {
            let copied = self.copy_selection(ctx);
            if copied && cut {
                self.delete_selected_feature();
            }
            return true;
        }
        if let Some(text) = pasted {
            self.paste_clipboard(&text);
            return true;
        }
        false
    }

    /// Copies the selected features to the system clipboard as bare GeoJSON.
    /// Returns whether anything was copied.
    pub(super) fn copy_selection(&mut self, ctx: &Context) -> bool {
        let Some(id) = self.selection else {
            self.status = Some("Select a layer before copying features.".to_string());
            return false;
        };
        let Some(multi) = self.edit.multi_selection().cloned() else {
            self.status = Some("Click a feature on the map before copying it.".to_string());
            return false;
        };
        let Some(features) = self.local.feature_set(id) else {
            self.status = Some(EditError::FeaturesNotLoaded(id).to_string());
            return false;
        };
        match edit::clipboard::copy_text(features, multi.features()) {
            Ok(text) => {
                ctx.copy_text(text);
                self.status = Some(match multi.len() {
                    1 => "1 feature copied as GeoJSON.".to_string(),
                    count => format!("{count} features copied as GeoJSON."),
                });
                true
            }
            Err(refusal) => {
                self.status = Some(refusal);
                false
            }
        }
    }

    /// Pastes clipboard GeoJSON into the active layer as ONE undoable
    /// transaction, selecting the pasted features.
    fn paste_clipboard(&mut self, text: &str) {
        let Some(id) = self.selection else {
            self.status = Some("Select a layer to paste features into.".to_string());
            return;
        };
        let Some(features) = self.local.feature_set(id).map(Arc::clone) else {
            self.status = Some(EditError::FeaturesNotLoaded(id).to_string());
            return;
        };
        let pasted = match edit::clipboard::paste_features(text) {
            Ok(pasted) => pasted,
            Err(refusal) => {
                self.status = Some(refusal);
                return;
            }
        };
        let start = features.features.len();
        let count = pasted.len();
        let ops = pasted
            .into_iter()
            .enumerate()
            .map(|(offset, feature)| command::FeatureOp::Add {
                index: start + offset,
                feature: Box::new(feature),
            })
            .collect();
        self.undo.close_coalescing();
        let landed = self.commit_edit(EditTransaction {
            layer: id,
            label: if count > 1 {
                "Paste features"
            } else {
                "Paste feature"
            },
            ops,
            selection_before: self.edit.selection(),
            selection_after: Some(EditSelection::feature(start)),
            coalesce: None,
        });
        if landed {
            // Select the whole pasted run, anchored on its first feature.
            //
            // Stopped at the FIRST refusal, not carried on to the end of the
            // run: `toggled` clones the whole set on every call, including the
            // calls it refuses, so a paste of 100 000 features used to pay 90
            // 000 full clones of a 10 000-element `Vec` for nothing.
            let mut multi = selection::FeatureSelection::single(start);
            let mut selected = 1;
            for feature in start + 1..start + count {
                let Some(next) = multi.toggled(feature) else {
                    break;
                };
                multi = next;
                selected += 1;
            }
            let multi = multi.with_anchor(start);
            self.edit.set_multi_selection(Some(multi));
            match (count, selected) {
                (1, _) => {
                    self.status =
                        Some(format!("1 feature pasted into the layer at index {start}."));
                }
                // A run past the cap is SAID, not silently trimmed: every
                // feature landed, but the selection — and therefore the next
                // Delete or drag — covers only its head.
                (count, selected) if selected < count => {
                    self.push_edit_notice(EditNotice::new(format!(
                        "{count} features pasted; the first {selected} are selected — a selection \
                         holds at most {} features.",
                        selection::MAX_MULTI_SELECT
                    )));
                }
                (count, _) => {
                    self.status = Some(format!("{count} features pasted — all selected."));
                }
            }
        }
    }
}
