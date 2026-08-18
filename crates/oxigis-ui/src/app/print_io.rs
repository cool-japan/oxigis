// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The PDF-export seam: the Export dialog, the snapshot it queues, and the
//! take-once hand-over a shell drains.
//!
//! Split from `app/mod.rs` under the 2000-line rule, beside its siblings
//! `app::data_io` (project files) and `app::archive_io` (tile archives) — the
//! same shape as both: `oxigis-ui` owns no filesystem and no download, so what
//! it can do is *decide what to export* and park it. Everything after that —
//! fetching the tiles, composing the page, writing or downloading the bytes —
//! is the shell's, through [`crate::print`]'s four pure entry points.
//!
//! Nothing here paints the page. [`crate::print`] is the whole layout and PDF
//! half and is wasm-clean; this module is only the app-side glue that reads the
//! project and the camera.

use super::{OxigisApp, VERTICAL_TITLE_HINT};
use crate::local_input;
use std::sync::Arc;

impl OxigisApp {
    /// A PDF export the user just asked for, if one is waiting — take-once,
    /// like the dropped-path seam. The shell fetches the tiles
    /// [`crate::print::required_tiles`] names, composes with
    /// [`crate::print::compose_map_rgb`], assembles with
    /// [`crate::print::pdf_document`], and delivers the bytes however its
    /// platform delivers files.
    pub fn take_pending_print(&mut self) -> Option<crate::print::PrintRequest> {
        self.pending_print.take()
    }
    /// Queues a PDF export of the current view — what File ▸ Export PDF asks.
    ///
    /// Captures a full snapshot (camera, basemap, COG, every visible local
    /// vector layer's collection and resolved style) so the export cannot
    /// change under the shell while it fetches tiles. Style resolution mirrors
    /// the map's own: the project's explicit style, else the layer's
    /// remembered default, else the collection-derived one.
    pub(super) fn request_print(&mut self) {
        let mut layers = Vec::new();
        for layer in self.project.layers.layers() {
            if !layer.visible || !local_input::is_local_layer(layer) {
                continue;
            }
            let Some(features) = self.local.feature_set(layer.id) else {
                continue;
            };
            let style = self
                .project
                .styles
                .get(&layer.id)
                .cloned()
                .or_else(|| self.local.default_style(layer.id).cloned())
                .unwrap_or_else(|| crate::local_vector::default_style_set_for(features));
            layers.push(crate::print::PrintLayer {
                // The legend names the layer the way the layer panel does:
                // the collection's own GeoJSON `name` member is a fallback,
                // not the primary, because a user who renamed a layer expects
                // the page to say what the panel says.
                name: layer.name.clone(),
                features: Arc::clone(features),
                style,
                families: self.local.families(layer.id),
                opacity: layer.opacity(),
            });
        }
        // The page and the screen read the same derivations, so what prints
        // is what is drawn — including after a remove, an undo, or a layer
        // promoted to the basemap. ONE call for the whole raster plan: the
        // basemap must not be read from a different source than the COG and
        // the archive it is composited with.
        let raster = self.desired_raster();
        self.pending_print = Some(crate::print::PrintRequest {
            title: self.project.name.clone(),
            attribution: self.credit_line(),
            view: self.map_panel.view(),
            basemap: raster.basemap,
            cog: raster.cog,
            archive: raster.archive,
            vector: self.desired_vector(),
            // The WHOLE drawn stack, not just the two top-most layers the
            // three fields above describe: a project holding an orthophoto
            // under a hillshade draws both on screen, so an export reading
            // only the three would print a different map from the one on
            // screen. `desired_tile_stack` is the very derivation the map
            // reconciles against, and `tile_layer_opacity` the very value it
            // tints with, so the page cannot disagree with the screen about
            // either.
            stack: self
                .desired_tile_stack()
                .entries
                .into_iter()
                .map(|entry| crate::print::PrintTileLayer {
                    opacity: self.tile_layer_opacity(entry.layer),
                    layer: entry.layer,
                    source: entry.source,
                })
                .collect(),
            layers,
            options: self.print_options,
        });
        self.status = Some(
            "Exporting PDF — fetching basemap tiles; the finished file is reported here."
                .to_string(),
        );
    }
    /// Draws the Export PDF options window while it is open: page size,
    /// orientation and raster resolution, then the Export button that queues
    /// the actual [`Self::request_print`] snapshot.
    pub(super) fn print_dialog_window(&mut self, ctx: &egui::Context) {
        if !self.print_dialog_open {
            return;
        }
        let mut open = true;
        let mut export = false;
        let mut cancel = false;
        egui::Window::new("Export PDF")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                egui::ComboBox::from_label("Page size")
                    .selected_text(self.print_options.page.label())
                    .show_ui(ui, |ui| {
                        for size in crate::print::PageSize::ALL {
                            ui.selectable_value(&mut self.print_options.page, size, size.label());
                        }
                    });
                egui::ComboBox::from_label("Orientation")
                    .selected_text(self.print_options.orientation.label())
                    .show_ui(ui, |ui| {
                        for orientation in crate::print::PageOrientation::ALL {
                            ui.selectable_value(
                                &mut self.print_options.orientation,
                                orientation,
                                orientation.label(),
                            );
                        }
                    });
                // The offered resolutions come from `print`'s own
                // `RASTER_PX_PER_PT_CHOICES` rather than from a list written
                // here: the dialog's choices and the export's ceiling
                // (`MAX_RASTER_PX_PER_PT`) then cannot drift apart, and
                // 300 dpi — the number a print shop asks for, exactly
                // `25.0 / 6.0` px/pt — is reachable at all, which a
                // hand-written `[2.0, 3.0, 4.0]` stopping at 288 dpi was not.
                // Both the button and the entries label through `dpi_label`,
                // which rounds, so that entry reads `300 dpi` instead of
                // `299.99998 dpi`.
                egui::ComboBox::from_label("Map image resolution")
                    .selected_text(crate::print::dpi_label(self.print_options.raster_px_per_pt))
                    .show_ui(ui, |ui| {
                        for px_per_pt in crate::print::RASTER_PX_PER_PT_CHOICES {
                            ui.selectable_value(
                                &mut self.print_options.raster_px_per_pt,
                                px_per_pt,
                                crate::print::dpi_label(px_per_pt),
                            );
                        }
                    });
                // print/text v1.4 (D-V1): the ONE surface the vertical
                // writing item exposes. Off by default; honoured only when
                // the title is CJK enough for the export's refusal ladder,
                // which otherwise prints it horizontally as before.
                ui.checkbox(
                    &mut self.print_options.vertical_title,
                    "Vertical title (CJK)",
                )
                .on_hover_text(VERTICAL_TITLE_HINT);
                // print v1.8: the size race is honest — `pdf_document` encodes
                // both ways and keeps the smaller — so this can never grow the
                // file, which is why it needs no quality slider beside it.
                ui.checkbox(
                    &mut self.print_options.photo_jpeg,
                    "Compress photographic basemap (JPEG)",
                )
                .on_hover_text(
                    "Embeds the map image as JPEG when doing so shrinks the file below the \
                     lossless path; a flat, line-art-like or screenshot-like map \
                     automatically stays lossless \u{2014} turning this on can never grow the PDF.",
                );
                // print v1.7 map furniture. Every one of these defaults to the
                // shipped behaviour, so a dialog the user never touches
                // produces byte-identical pages to the version before they
                // were reachable.
                ui.separator();
                ui.checkbox(&mut self.print_options.scale_bar, "Scale bar");
                egui::ComboBox::from_label("Scale units")
                    .selected_text(self.print_options.scale_units.label())
                    .show_ui(ui, |ui| {
                        for units in crate::print::ScaleUnits::ALL {
                            ui.selectable_value(
                                &mut self.print_options.scale_units,
                                units,
                                units.label(),
                            );
                        }
                    });
                ui.checkbox(
                    &mut self.print_options.representative_fraction,
                    "Representative fraction",
                );
                ui.checkbox(&mut self.print_options.north_arrow, "North arrow");
                ui.checkbox(&mut self.print_options.legend, "Legend");
                ui.checkbox(
                    &mut self.print_options.document_metadata,
                    "Document metadata",
                );
                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .button("Export")
                        .on_hover_text(
                            "The basemap as an image at the chosen resolution, local layers \
                             as real vector paths, plus title, attribution and a scale bar",
                        )
                        .clicked()
                    {
                        export = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });
        if export {
            self.request_print();
        }
        self.print_dialog_open = open && !export && !cancel;
    }
}
