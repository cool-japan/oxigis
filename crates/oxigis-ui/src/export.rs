// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Getting data back **out**: the take-once file seam every export crosses,
//! and the file-naming rule it carries.
//!
//! Until this module the File menu's only output was a PDF and the copy-JSON
//! modal, which made every other subsystem a dead end: a layer the user
//! digitized, a Processing result, a GeoPackage table reprojected on the way
//! in — all of it could leave the app only through `Ctrl+C` into another
//! program. A GIS that reads four vector formats and writes none is a viewer.
//!
//! # The seam
//!
//! `oxigis-ui` compiles to `wasm32` and owns no filesystem, so an export
//! crosses to the shell exactly the way a project save and a PDF do: this
//! crate serializes, parks the bytes in a **take-once** slot
//! ([`crate::app::OxigisApp::take_pending_export`]), and the shell writes them
//! wherever its platform writes files — a Save dialog on the desktop, a
//! download in the browser. The shell then reports back through exactly one of
//! [`crate::app::OxigisApp::confirm_export_written`],
//! [`crate::app::OxigisApp::report_export_failed`] or
//! [`crate::app::OxigisApp::cancel_pending_export`], which is what puts the
//! outcome on the status line.
//!
//! The one deliberate difference from [`crate::app::ProjectSaveRequest`] is
//! the payload: that one carries `path: Option<PathBuf>`, because a project
//! that has been saved once knows where it lives and a plain `Ctrl+S` must not
//! ask again. An export never has a remembered destination — it is a new file
//! every time — so it carries a **suggested file name** instead, which is what
//! a Save-As dialog needs and what a browser download must be given.
//!
//! The file is in two halves: the payload types and the naming rule first
//! (no `egui`, no app state, so a shell or a test can name and carry a file
//! without an `OxigisApp` in hand), then the `impl OxigisApp` that fills them
//! in — the File ▸ Export menu and the two request builders behind it.
//!
//! # What is exported
//!
//! * **Layer as GeoJSON** — the selected local vector layer's whole
//!   [`FeatureCollection`], compactly encoded (a data file, not a document to
//!   read: it is what every other GeoJSON writer emits, and it roughly halves
//!   the bytes). Geometry and properties are exactly what the map is drawing,
//!   because they are the same `Arc` the renderer was handed.
//! * **Attribute table as CSV** — the rows *currently shown*, in the order
//!   they are shown, through the same
//!   [`crate::attribute_table::FeatureRowSource::to_csv`] the panel's
//!   "Copy CSV" button uses. A filter that hides half the rows therefore
//!   exports half the rows, which is the only reading of the button that
//!   matches what the user is looking at.

use std::path::Path;

use oxigeo::geojson::types::FeatureCollection;
use oxigis_core::LayerId;

use crate::app::OxigisApp;

/// Longest file-name stem [`export_file_name`] will build, in characters.
///
/// A layer name is free text and can be a whole sentence; a file name that
/// long is refused outright by some filesystems and is unusable on all of
/// them. Truncation happens on a character boundary, so a Japanese layer name
/// survives it as text.
pub const MAX_EXPORT_STEM_CHARS: usize = 60;

/// What a layer whose name sanitizes away to nothing is called.
pub const FALLBACK_EXPORT_STEM: &str = "layer";

/// Which format an [`ExportRequest`] carries.
///
/// An enum rather than something left implicit in the file name, because the
/// shell needs it twice over: for its Save dialog's type filter, and to pick
/// the MIME type a browser download is served with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportKind {
    /// A GeoJSON `FeatureCollection` document.
    GeoJson,
    /// Comma-separated attribute rows, with a header line.
    Csv,
}

impl ExportKind {
    /// The file extension, without the dot.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::GeoJson => "geojson",
            Self::Csv => "csv",
        }
    }

    /// What a Save dialog's file-type filter should call this.
    #[must_use]
    pub fn filter_label(self) -> &'static str {
        match self {
            Self::GeoJson => "GeoJSON",
            Self::Csv => "CSV",
        }
    }

    /// The MIME type a browser download should be served with.
    #[must_use]
    pub fn mime_type(self) -> &'static str {
        match self {
            Self::GeoJson => "application/geo+json",
            Self::Csv => "text/csv",
        }
    }
}

/// A file the shell has been asked to write — the take-once seam
/// [`crate::app::OxigisApp::take_pending_export`] hands over.
///
/// Both formats are text, so the payload is a `String` rather than bytes;
/// [`Self::content_bytes`] is the UTF-8 encoding a writer actually wants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportRequest {
    /// What the file should be called if the user does not say otherwise —
    /// the layer's name, sanitized, with the format's extension.
    pub suggested_file_name: String,
    /// The document itself.
    pub content: String,
    /// Which format [`Self::content`] is in.
    pub kind: ExportKind,
}

impl ExportRequest {
    /// The document as the bytes a file writer takes.
    #[must_use]
    pub fn content_bytes(&self) -> Vec<u8> {
        self.content.clone().into_bytes()
    }

    /// How many bytes the file will be — what the status line quotes without
    /// having to encode the document first.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.content.len()
    }
}

/// Builds a file name for `layer_name` in `kind`'s format.
///
/// Everything that is not a letter, a digit, a dash, an underscore or a dot
/// becomes an underscore — path separators emphatically included, since a
/// layer called `../etc/passwd` must not be able to steer a shell's writer
/// anywhere. Runs of separators collapse, leading and trailing separators and
/// dots go (a name starting with a dot is hidden on Unix and refused on
/// Windows), the stem is capped at [`MAX_EXPORT_STEM_CHARS`] characters, and a
/// name that sanitizes away entirely becomes [`FALLBACK_EXPORT_STEM`].
#[must_use]
pub fn export_file_name(layer_name: &str, kind: ExportKind) -> String {
    let mut stem = String::new();
    let mut characters = 0;
    let mut last_was_separator = false;
    for character in layer_name.chars() {
        if characters == MAX_EXPORT_STEM_CHARS {
            break;
        }
        // `is_alphanumeric` rather than `is_ascii_alphanumeric`: a layer
        // called 東京の区 keeps its name, which is the point of a sanitizer
        // rather than an ASCII filter.
        let keep = character.is_alphanumeric() || matches!(character, '-' | '_' | '.');
        if !keep {
            if last_was_separator || stem.is_empty() {
                continue;
            }
            stem.push('_');
            characters += 1;
            last_was_separator = true;
            continue;
        }
        last_was_separator = character == '_';
        stem.push(character);
        characters += 1;
    }
    let trimmed = stem.trim_matches(|character| matches!(character, '_' | '.' | '-'));
    let stem = if trimmed.is_empty() {
        FALLBACK_EXPORT_STEM
    } else {
        trimmed
    };
    format!("{stem}.{}", kind.extension())
}

/// The document `features` would be exported as — the seam a caller with a
/// collection but no app (a test, a future batch exporter) reaches for, and
/// the exact call `request_layer_geojson_export` makes.
///
/// # Errors
///
/// The writer's own error, when the collection will not serialize.
pub fn feature_collection_geojson(
    features: &FeatureCollection,
) -> Result<String, oxigeo::geojson::GeoJsonError> {
    oxigeo::geojson::writer::to_string(features)
}

impl OxigisApp {
    /// An export the shell is being asked to write, if one is waiting —
    /// take-once, exactly like [`OxigisApp::take_pending_project_save`] and
    /// [`OxigisApp::take_pending_print`].
    ///
    /// The shell asks the user where the file goes (defaulting to
    /// [`ExportRequest::suggested_file_name`]), writes
    /// [`ExportRequest::content_bytes`], and reports the outcome through
    /// exactly one of [`Self::confirm_export_written`],
    /// [`Self::report_export_failed`] or [`Self::cancel_pending_export`]. Not
    /// reporting costs nothing but the status line: unlike a project save, an
    /// export carries no "saved" marker and parks no action behind itself.
    pub fn take_pending_export(&mut self) -> Option<ExportRequest> {
        self.pending_export.take()
    }

    /// Reports that the export's bytes are on disk at `path`.
    ///
    /// Borrowed rather than owned, unlike
    /// [`OxigisApp::confirm_project_saved`]: nothing here remembers the path,
    /// because an export is a one-way copy and the next one is a new file.
    pub fn confirm_export_written(&mut self, path: &Path) {
        self.status = Some(format!("Exported {}.", path.display()));
    }

    /// Reports that the export could not be written.
    pub fn report_export_failed(&mut self, error: &str) {
        self.status = Some(format!("Export failed: {error}"));
    }

    /// Reports that the user dismissed the shell's own file dialog — not an
    /// error, and deliberately not silent.
    pub fn cancel_pending_export(&mut self) {
        self.status = Some("Export cancelled.".to_string());
    }

    /// Whether an export is waiting for the shell to write it.
    #[must_use]
    pub fn export_pending(&self) -> bool {
        self.pending_export.is_some()
    }

    /// Parks `request` for the shell and says so on the status line.
    ///
    /// A second request while one is still waiting is refused rather than
    /// allowed to replace it — the same guard
    /// [`OxigisApp::request_project_save`] keeps, for the same reason: the
    /// first file was asked for and would otherwise silently never be
    /// written. It says so, because an export menu item that appears to do
    /// nothing is worse than one that explains itself.
    fn queue_export(&mut self, request: ExportRequest) -> bool {
        if self.pending_export.is_some() {
            self.status = Some(
                "An export is already waiting to be written; finish that one first.".to_string(),
            );
            return false;
        }
        self.status = Some(format!(
            "Exporting {} \u{2014} {} bytes of {}.",
            request.suggested_file_name,
            request.byte_len(),
            request.kind.filter_label()
        ));
        self.pending_export = Some(request);
        true
    }

    /// File ▸ Export ▸ Layer as GeoJSON…: serializes the selected local vector
    /// layer and hands it to the shell.
    ///
    /// Answers whether an export was queued. Every refusal explains itself on
    /// the status line: nothing selected, a selection whose drawing does not
    /// come from a `FeatureCollection` at all (a basemap, a COG, a vector-tile
    /// source), a layer whose features a shell has still to read, or a
    /// document that will not serialize.
    pub fn request_layer_geojson_export(&mut self) -> bool {
        let Some(id) = self.selection else {
            self.status = Some("Select a layer to export.".to_string());
            return false;
        };
        let Some(layer) = self.project.layers.get(id) else {
            self.status = Some("Select a layer to export.".to_string());
            return false;
        };
        let name = layer.name.clone();
        // Serialized while the borrow of the feature store is alive and not a
        // moment longer, so the status lines below can be written.
        let serialized = self
            .local
            .feature_set(id)
            .map(|features| crate::export::feature_collection_geojson(features));
        let Some(serialized) = serialized else {
            self.status = Some(format!(
                "\u{201c}{name}\u{201d} has no features to export \u{2014} a tiled layer draws \
                 from its own source, and a layer loaded by reference has still to be read."
            ));
            return false;
        };
        match serialized {
            Ok(content) => {
                let request = ExportRequest {
                    suggested_file_name: export_file_name(&name, ExportKind::GeoJson),
                    content,
                    kind: ExportKind::GeoJson,
                };
                self.queue_export(request)
            }
            Err(error) => {
                self.status = Some(format!(
                    "\u{201c}{name}\u{201d} could not be written: {error}"
                ));
                false
            }
        }
    }

    /// File ▸ Export ▸ Attribute table as CSV…: the rows the table is showing,
    /// in the order it is showing them.
    ///
    /// Answers whether an export was queued.
    pub fn request_table_csv_export(&mut self) -> bool {
        let Some(id) = self.table_panel.bound_layer() else {
            self.status = Some("The attribute table is not showing a layer to export.".to_string());
            return false;
        };
        let csv = self.table_panel.export_csv();
        self.queue_csv(Some(id), csv)
    }

    /// Queues `csv` as an export named after layer `id` (or the fallback stem
    /// when the binding is gone).
    fn queue_csv(&mut self, id: Option<LayerId>, csv: String) -> bool {
        let name = id.and_then(|id| self.project.layers.get(id)).map_or_else(
            || FALLBACK_EXPORT_STEM.to_string(),
            |layer| layer.name.clone(),
        );
        let request = ExportRequest {
            suggested_file_name: export_file_name(&name, ExportKind::Csv),
            content: csv,
            kind: ExportKind::Csv,
        };
        self.queue_export(request)
    }

    /// Takes whatever the attribute table's own "Export CSV" button asked for
    /// this frame.
    ///
    /// The panel captures the CSV **at click time** rather than handing back a
    /// flag for this to re-derive from, which is what keeps the promise its
    /// module docs make: what is exported is what was on screen when the
    /// button was pressed, not what the panel happens to show a frame later.
    pub(crate) fn drain_table_export(&mut self) {
        let Some(csv) = self.table_panel.take_export_request() else {
            return;
        };
        let id = self.table_panel.bound_layer();
        let _queued = self.queue_csv(id, csv);
    }

    /// The File ▸ Export ▸ submenu.
    ///
    /// The PDF export moved in here from the top level when it stopped being
    /// the only thing this application could write: three exports under one
    /// heading is a menu, three exports interleaved with New/Open/Save is a
    /// list.
    pub(crate) fn export_menu(&mut self, ui: &mut egui::Ui) {
        let can_export_layer = self
            .selection
            .is_some_and(|id| self.local.feature_set(id).is_some());
        let can_export_csv = self.table_panel.bound_layer().is_some();
        ui.menu_button("Export", |ui| {
            if ui
                .add_enabled(
                    can_export_layer,
                    egui::Button::new("Layer as GeoJSON\u{2026}"),
                )
                .on_hover_text("Writes the selected layer's features as a GeoJSON file")
                .on_disabled_hover_text("Select a loaded local vector layer to export it.")
                .clicked()
            {
                let _queued = self.request_layer_geojson_export();
                ui.close();
            }
            if ui
                .add_enabled(
                    can_export_csv,
                    egui::Button::new("Attribute table as CSV\u{2026}"),
                )
                .on_hover_text(
                    "Writes the rows the attribute table is showing \u{2014} the current filter \
                     and sort included",
                )
                .on_disabled_hover_text("Open the attribute table on a layer to export its rows.")
                .clicked()
            {
                let _queued = self.request_table_csv_export();
                ui.close();
            }
            ui.separator();
            if ui
                .button("PDF\u{2026}")
                .on_hover_text(
                    "One page \u{2014} size, orientation and resolution selectable: the basemap \
                     as an image, local layers as real vector paths, plus title, attribution and \
                     a scale bar",
                )
                .clicked()
            {
                self.print_dialog_open = true;
                ui.close();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_file_name_sanitizes_separators_and_keeps_the_extension() {
        assert_eq!(
            export_file_name("roads", ExportKind::GeoJson),
            "roads.geojson"
        );
        assert_eq!(
            export_file_name("City blocks 2026", ExportKind::Csv),
            "City_blocks_2026.csv"
        );
        // A name that could steer a writer somewhere must not survive as a
        // path.
        let escaped = export_file_name("../etc/passwd", ExportKind::GeoJson);
        assert!(!escaped.contains('/'), "{escaped}");
        assert!(!escaped.starts_with('.'), "{escaped}");
        assert_eq!(escaped, "etc_passwd.geojson");
        assert_eq!(
            export_file_name("C:\\data\\layer", ExportKind::Csv),
            "C_data_layer.csv"
        );
        // Runs collapse rather than producing `a___b`.
        assert_eq!(export_file_name("a   b", ExportKind::Csv), "a_b.csv");
    }

    #[test]
    fn export_file_name_survives_empty_unicode_and_overlong_names() {
        assert_eq!(export_file_name("", ExportKind::Csv), "layer.csv");
        assert_eq!(export_file_name("///", ExportKind::Csv), "layer.csv");
        assert_eq!(export_file_name("...", ExportKind::Csv), "layer.csv");
        // A non-ASCII name is a name, not noise.
        assert_eq!(
            export_file_name("\u{6771}\u{4EAC}\u{306E}\u{533A}", ExportKind::GeoJson),
            "\u{6771}\u{4EAC}\u{306E}\u{533A}.geojson"
        );
        let long = "a".repeat(500);
        let name = export_file_name(&long, ExportKind::GeoJson);
        assert_eq!(
            name.chars().count(),
            MAX_EXPORT_STEM_CHARS + ".geojson".len()
        );
        // Truncating a multi-byte name must not split a character.
        let kanji = "\u{6771}".repeat(500);
        let truncated = export_file_name(&kanji, ExportKind::Csv);
        assert_eq!(
            truncated.chars().count(),
            MAX_EXPORT_STEM_CHARS + ".csv".len()
        );
    }

    #[test]
    fn export_kind_describes_itself_for_a_shell() {
        assert_eq!(ExportKind::GeoJson.extension(), "geojson");
        assert_eq!(ExportKind::Csv.extension(), "csv");
        assert_eq!(ExportKind::GeoJson.mime_type(), "application/geo+json");
        assert_eq!(ExportKind::Csv.mime_type(), "text/csv");
        assert!(!ExportKind::Csv.filter_label().is_empty());
        assert!(!ExportKind::GeoJson.filter_label().is_empty());
    }

    #[test]
    fn export_request_bytes_are_the_content_and_the_length_agrees() {
        let request = ExportRequest {
            suggested_file_name: "roads.csv".to_string(),
            content: "#,geometry\n0,Point\n".to_string(),
            kind: ExportKind::Csv,
        };
        assert_eq!(request.content_bytes(), request.content.as_bytes());
        assert_eq!(request.byte_len(), request.content.len());
    }

    #[test]
    fn export_geojson_helper_round_trips_a_collection() {
        const POINTS: &str = r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"name":"Tokyo"},
             "geometry":{"type":"Point","coordinates":[139.76,35.68]}}]}"#;
        let Ok(features) = oxigeo::geojson::reader::feature_collection_from_str(POINTS) else {
            panic!("the fixture parses");
        };
        let Ok(document) = feature_collection_geojson(&features) else {
            panic!("the collection serializes");
        };
        let Ok(parsed) = oxigeo::geojson::reader::feature_collection_from_str(&document) else {
            panic!("the exported document parses");
        };
        assert_eq!(parsed.features.len(), 1);
        assert!(document.contains("139.76"), "{document}");
    }

    // ---- The app half: the take-once seam end to end. ----------------------

    /// Two points with properties — enough that a round trip has geometry,
    /// attributes and a feature count to disagree about.
    const POINTS: &str = r#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "Tokyo", "pop": 13},
             "geometry": {"type": "Point", "coordinates": [139.76, 35.68]}},
            {"type": "Feature", "properties": {"name": "Osaka", "pop": 2},
             "geometry": {"type": "Point", "coordinates": [135.5, 34.69]}}
        ]
    }"#;

    /// Drives one whole `ui` frame at a realistic window size, which is what
    /// binds the attribute table to the selected layer.
    fn run_one_frame(app: &mut OxigisApp) {
        let ctx = egui::Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            ..Default::default()
        };
        let _output = ctx.run_ui(raw_input, |ui| app.ui(ui));
    }

    #[test]
    fn export_layer_as_geojson_round_trips_every_feature() {
        let mut app = OxigisApp::new();
        let Some(_id) = app.add_geojson_layer_from_text("Kanto cities", POINTS, None) else {
            panic!("the fixture parses");
        };
        assert!(app.request_layer_geojson_export());
        let Some(request) = app.take_pending_export() else {
            panic!("the export is waiting for a shell");
        };
        assert_eq!(request.kind, ExportKind::GeoJson);
        assert_eq!(request.suggested_file_name, "Kanto_cities.geojson");
        assert_eq!(request.byte_len(), request.content.len());
        assert_eq!(request.content_bytes(), request.content.as_bytes());

        // The bytes are a GeoJSON document with both features and their
        // properties intact — parsed back rather than string-matched, so the
        // test fails on a malformed document instead of on a reordered key.
        let Ok(parsed) = oxigeo::geojson::reader::feature_collection_from_str(&request.content)
        else {
            panic!("the exported document parses");
        };
        assert_eq!(parsed.features.len(), 2);
        let names: Vec<String> = parsed
            .features
            .iter()
            .filter_map(|feature| feature.properties.as_ref())
            .filter_map(|properties| properties.get("name"))
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect();
        assert_eq!(names, vec!["Tokyo".to_string(), "Osaka".to_string()]);
        assert!(request.content.contains("139.76"), "{}", request.content);
    }

    #[test]
    fn export_is_take_once_and_refuses_to_queue_a_second_file_over_the_first() {
        let mut app = OxigisApp::new();
        let Some(_id) = app.add_geojson_layer_from_text("cities", POINTS, None) else {
            panic!("the fixture parses");
        };
        assert!(app.request_layer_geojson_export());
        assert!(app.export_pending());
        // The second request must not silently replace the first.
        assert!(!app.request_layer_geojson_export());
        assert!(
            app.status()
                .is_some_and(|status| status.contains("already waiting")),
            "{:?}",
            app.status()
        );
        assert!(app.take_pending_export().is_some());
        assert!(!app.export_pending());
        assert!(app.take_pending_export().is_none(), "take-once");
        // With the slot drained the next export goes through.
        assert!(app.request_layer_geojson_export());
    }

    #[test]
    fn export_refuses_a_selection_with_no_features_and_says_why() {
        let mut app = OxigisApp::new();
        assert!(!app.request_layer_geojson_export());
        assert!(!app.export_pending());
        assert!(
            app.status().is_some_and(|status| status.contains("Select")),
            "{:?}",
            app.status()
        );
    }

    #[test]
    fn export_reports_the_shells_outcome_on_the_status_line() {
        let mut app = OxigisApp::new();
        app.confirm_export_written(Path::new("/tmp/roads.geojson"));
        assert!(
            app.status()
                .is_some_and(|status| status.contains("roads.geojson")),
            "{:?}",
            app.status()
        );
        app.report_export_failed("read-only volume");
        assert!(
            app.status()
                .is_some_and(|status| status.contains("read-only volume")),
            "{:?}",
            app.status()
        );
        app.cancel_pending_export();
        assert!(
            app.status().is_some_and(|status| status.contains("ancel")),
            "{:?}",
            app.status()
        );
    }

    #[test]
    fn export_csv_carries_the_rows_the_table_is_showing() {
        let mut app = OxigisApp::new();
        let Some(_id) = app.add_geojson_layer_from_text("cities", POINTS, None) else {
            panic!("the fixture parses");
        };
        // The table binds during a frame, so run one.
        run_one_frame(&mut app);
        assert!(app.request_table_csv_export());
        let Some(request) = app.take_pending_export() else {
            panic!("a CSV is waiting");
        };
        assert_eq!(request.kind, ExportKind::Csv);
        assert_eq!(request.suggested_file_name, "cities.csv");
        let mut lines = request.content.lines();
        let header = lines.next().unwrap_or_default();
        assert!(header.starts_with("#,geometry"), "{header}");
        assert!(header.contains("name"), "{header}");
        assert_eq!(lines.count(), 2, "one line per feature");
        assert!(request.content.contains("Tokyo"), "{}", request.content);
    }

    #[test]
    fn export_csv_exports_exactly_the_filtered_rows() {
        // The promise the panel's docs make: what is exported is what is on
        // screen, filter included.
        let mut app = OxigisApp::new();
        let Some(_id) = app.add_geojson_layer_from_text("cities", POINTS, None) else {
            panic!("the fixture parses");
        };
        run_one_frame(&mut app);
        app.table_panel.set_filter_text("Osaka");
        assert!(app.request_table_csv_export());
        let Some(request) = app.take_pending_export() else {
            panic!("a CSV is waiting");
        };
        assert!(request.content.contains("Osaka"), "{}", request.content);
        assert!(
            !request.content.contains("Tokyo"),
            "a filtered-out row must not be exported: {}",
            request.content
        );
    }

    #[test]
    fn export_csv_refuses_when_the_table_shows_nothing() {
        let mut app = OxigisApp::new();
        assert!(!app.request_table_csv_export());
        assert!(!app.export_pending());
        assert!(
            app.status()
                .is_some_and(|status| status.contains("attribute table")),
            "{:?}",
            app.status()
        );
    }

    #[test]
    fn export_the_tables_own_button_is_drained_by_the_frame_loop() {
        // The panel captures the CSV at click time and parks it; the app takes
        // it on the same frame. Simulated here by parking it directly, which
        // is what the button does.
        let mut app = OxigisApp::new();
        let Some(_id) = app.add_geojson_layer_from_text("cities", POINTS, None) else {
            panic!("the fixture parses");
        };
        run_one_frame(&mut app);
        assert!(!app.table_panel.export_requested());
        let captured = app.table_panel.export_csv();
        app.table_panel.park_export_request(captured.clone());
        assert!(app.table_panel.export_requested());
        run_one_frame(&mut app);
        assert!(
            !app.table_panel.export_requested(),
            "the frame loop drains it"
        );
        let Some(request) = app.take_pending_export() else {
            panic!("the drained request became an export");
        };
        assert_eq!(request.kind, ExportKind::Csv);
        assert_eq!(
            request.content, captured,
            "the CSV captured at click time is the CSV exported"
        );
    }

    #[test]
    fn export_geojson_bytes_are_byte_identical_to_the_pure_writer_path() {
        // The menu item and the helper must not drift: the seam a shell writes
        // is exactly what `feature_collection_geojson` produces.
        let Ok(features) = oxigeo::geojson::reader::feature_collection_from_str(POINTS) else {
            panic!("the fixture parses");
        };
        let Ok(direct) = feature_collection_geojson(&features) else {
            panic!("the collection serializes");
        };
        let mut app = OxigisApp::new();
        let Some(_id) = app.add_geojson_layer_from_text("cities", POINTS, None) else {
            panic!("the fixture parses");
        };
        assert!(app.request_layer_geojson_export());
        let Some(request) = app.take_pending_export() else {
            panic!("waiting");
        };
        assert_eq!(direct, request.content);
    }
}
