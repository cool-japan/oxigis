//! Writing the two text-file seams `oxigis-ui` parks for a shell: a
//! Processing result the user asked to keep, and a layer/table export.
//!
//! Split out of `main.rs` for size (COOLJAPAN: files stay under 2000 lines),
//! the same reason [`crate::provider_watch`] and [`crate::cli`] are their own
//! modules.
//!
//! Both seams end in the same place — bytes, a name to suggest, and a
//! destination the user has to choose — so they share one parking slot and one
//! [`Ask`] dispatch. What differs is only how the outcome is REPORTED: an
//! export answers through `oxigis-ui`'s own
//! `confirm_export_written`/`report_export_failed`/`cancel_pending_export`
//! trio, while a Processing result has no report of its own (it is still in the
//! Processing window) and gets a status line instead. Keeping that the only
//! difference is what stops the two prompts disagreeing about what a cancelled
//! dialog means.

use crate::OxigisDesktopApp;
use crate::file_dialog::Ask;

/// A text document the app has asked this shell to write, waiting on a
/// destination.
///
/// Two seams, one parking slot and one `Ask` dispatch: the file they produce is
/// the same kind of file, and a shell that had two of everything would be two
/// places for the "the prompt is still up" case to be got wrong.
pub(crate) enum PendingFileWrite {
    /// Processing ▸ Output ▸ Save to file… — a tool result as GeoJSON.
    ///
    /// Reports through the status line only: unlike an export, this seam parks
    /// no action behind itself and the app keeps no record of the write.
    Processing(oxigis_ui::ProcessingFileRequest),
    /// A layer's GeoJSON export or the attribute table's CSV export, which
    /// report back through `confirm_export_written` / `report_export_failed` /
    /// `cancel_pending_export`.
    Export(oxigis_ui::ExportRequest),
}

impl PendingFileWrite {
    /// Which ask this write needs, and therefore which file-type filter the
    /// dialog offers.
    fn ask(&self) -> Ask {
        match self {
            Self::Processing(_) => Ask::SaveGeoJson,
            Self::Export(request) => match request.kind {
                oxigis_ui::ExportKind::GeoJson => Ask::SaveGeoJson,
                oxigis_ui::ExportKind::Csv => Ask::SaveCsv,
            },
        }
    }

    /// The file name to suggest.
    fn suggested_name(&self) -> String {
        match self {
            // The seam carries a basename without an extension, deliberately:
            // which one to use is the writing shell's decision, and here it is
            // always GeoJSON.
            Self::Processing(request) => format!("{}.geojson", request.name),
            Self::Export(request) => request.suggested_file_name.clone(),
        }
    }

    /// The bytes to write.
    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::Processing(request) => request.content.clone().into_bytes(),
            Self::Export(request) => request.content_bytes(),
        }
    }
}

impl OxigisDesktopApp {
    /// Settles one `take_pending_processing_save` or `take_pending_export`:
    /// asks where the file goes, and writes it when the answer is immediate.
    ///
    /// Modelled on [`Self::resolve_project_save`], and deliberately: an export
    /// has no known destination to write straight to (it is always a new file),
    /// but everything else — the "a prompt is already open" refusal, the
    /// deferred answer, the dismissal — has to behave identically or the two
    /// prompts would disagree about what a cancelled dialog means.
    pub(crate) fn resolve_file_write(&mut self, write: PendingFileWrite) {
        if self.asking_for_a_path() {
            self.report_file_write_cancelled(
                &write,
                "Finish the file prompt that is already open first.",
            );
            return;
        }
        let ask = write.ask();
        let suggested = write.suggested_name();
        match self.ask_for_path(ask, &suggested) {
            Some(path) => {
                let _written = self.write_data_file(write, path);
            }
            // The prompt is on screen; the bytes wait for its answer, which
            // arrives in a later frame through `poll_path_prompt`.
            None if self.path_prompt.is_some() => self.pending_file_write = Some(write),
            None => self.report_file_write_cancelled(&write, "Nothing was exported."),
        }
    }

    /// Reports a write that will not happen, in whichever vocabulary the seam
    /// that asked for it uses.
    pub(crate) fn report_file_write_cancelled(&mut self, write: &PendingFileWrite, reason: &str) {
        match write {
            // This seam has no cancel report of its own: the result is still in
            // the Processing window, so the status line is the whole message.
            PendingFileWrite::Processing(_) => self.inner.set_status(reason.to_string()),
            PendingFileWrite::Export(_) => {
                // Order matters, exactly as in `resolve_project_save`:
                // `cancel_pending_export` writes its OWN status line ("Export
                // cancelled."), so the reason has to be set after it, not
                // before — otherwise "a prompt is already open" reaches the
                // user as a bare cancellation with nothing to act on.
                self.inner.cancel_pending_export();
                self.inner.set_status(reason.to_string());
            }
        }
    }

    /// Writes one parked file and reports the outcome. Answers whether the
    /// bytes are on disk — `false` keeps the in-app prompt up with the reason
    /// on it, exactly as a failed project write does.
    pub(crate) fn write_data_file(
        &mut self,
        write: PendingFileWrite,
        path: std::path::PathBuf,
    ) -> bool {
        match std::fs::write(&path, write.bytes()) {
            Ok(()) => {
                tracing::info!(
                    path = %path.display(),
                    "OxiGIS desktop: data file written",
                );
                self.note_dialog_directory(&path);
                match &write {
                    PendingFileWrite::Processing(request) => {
                        let features = request.features;
                        let plural = if features == 1 { "feature" } else { "features" };
                        self.inner
                            .set_status(format!("Wrote {} ({features} {plural}).", path.display()));
                    }
                    PendingFileWrite::Export(_) => self.inner.confirm_export_written(&path),
                }
                true
            }
            Err(error) => {
                let reason = format!("could not write {}: {error}", path.display());
                match &write {
                    PendingFileWrite::Processing(_) => self.inner.set_status(reason),
                    PendingFileWrite::Export(_) => self.inner.report_export_failed(&reason),
                }
                // Re-parked so a corrected path can still write these exact
                // bytes — the same recovery a failed project write gets.
                self.pending_file_write = Some(write);
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PendingFileWrite;
    use crate::file_dialog::Ask;
    use crate::{OxigisDesktopApp, session};

    /// A directory under the OS temp dir, unique per call.
    fn scratch_dir(label: &str) -> std::path::PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let path = std::env::temp_dir().join(format!("oxigis-file-write-{label}-{stamp}"));
        std::fs::create_dir_all(&path).expect("the scratch directory is creatable");
        path
    }

    /// A shell with no startup paths and a blank session.
    fn shell() -> OxigisDesktopApp {
        OxigisDesktopApp::new(Vec::new(), session::SessionState::default())
    }

    /// A GeoJSON export request, as `OxigisApp::take_pending_export` yields one.
    fn export() -> oxigis_ui::ExportRequest {
        oxigis_ui::ExportRequest {
            suggested_file_name: "cities.geojson".to_string(),
            content: r#"{"type":"FeatureCollection","features":[]}"#.to_string(),
            kind: oxigis_ui::ExportKind::GeoJson,
        }
    }

    /// A Processing result, as `take_pending_processing_save` yields one.
    fn processing() -> oxigis_ui::ProcessingFileRequest {
        oxigis_ui::ProcessingFileRequest {
            name: "buffer_result".to_string(),
            content: r#"{"type":"FeatureCollection","features":[]}"#.to_string(),
            features: 3,
        }
    }

    #[test]
    fn the_ask_and_the_suggested_name_follow_the_seam_and_its_format() {
        // The dialog's file-type filter comes from the ask, so an export of
        // rows must not offer a GeoJSON filter — and a Processing result, whose
        // seam carries a basename with NO extension, has to gain one here.
        assert_eq!(
            PendingFileWrite::Export(export()).ask(),
            Ask::SaveGeoJson,
            "a GeoJSON export asks for a GeoJSON destination"
        );
        let csv = oxigis_ui::ExportRequest {
            suggested_file_name: "cities.csv".to_string(),
            kind: oxigis_ui::ExportKind::Csv,
            ..export()
        };
        assert_eq!(PendingFileWrite::Export(csv).ask(), Ask::SaveCsv);
        let write = PendingFileWrite::Processing(processing());
        assert_eq!(write.ask(), Ask::SaveGeoJson);
        assert_eq!(
            write.suggested_name(),
            "buffer_result.geojson",
            "the seam carries a basename; choosing the extension is the writer's job"
        );
    }

    #[test]
    fn an_export_is_written_and_reported_through_the_apps_own_confirmation() {
        let dir = scratch_dir("export");
        let mut app = shell();
        let path = dir.join("cities.geojson");
        assert!(app.write_data_file(PendingFileWrite::Export(export()), path.clone()));
        assert_eq!(
            std::fs::read_to_string(&path).expect("the file is on disk"),
            export().content,
            "the bytes the app parked are the bytes on disk"
        );
        let status = app.inner.status().unwrap_or_default().to_string();
        assert!(status.starts_with("Exported "), "{status}");
        assert!(status.contains("cities.geojson"), "{status}");
        assert!(
            app.pending_file_write.is_none(),
            "a settled write leaves nothing parked"
        );
        let _removed = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_processing_result_is_written_and_counts_its_features() {
        let dir = scratch_dir("processing");
        let mut app = shell();
        let path = dir.join("buffer_result.geojson");
        assert!(app.write_data_file(PendingFileWrite::Processing(processing()), path.clone()));
        assert_eq!(
            std::fs::read_to_string(&path).expect("the file is on disk"),
            processing().content
        );
        let status = app.inner.status().unwrap_or_default().to_string();
        assert!(status.starts_with("Wrote "), "{status}");
        assert!(
            status.contains("(3 features)"),
            "the seam's own count reaches the user: {status}"
        );
        let _removed = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_write_re_parks_the_bytes_so_a_corrected_path_still_saves_them() {
        let dir = scratch_dir("refused");
        let mut app = shell();
        // A directory is not a file: the write fails in a way the user can
        // correct by editing the path, which is exactly the case the in-app
        // prompt stays open for.
        let occupied = dir.join("occupied.geojson");
        std::fs::create_dir(&occupied).expect("the directory is creatable");
        assert!(
            !app.write_data_file(PendingFileWrite::Export(export()), occupied),
            "a failed write must not close the prompt"
        );
        let status = app.inner.status().unwrap_or_default().to_string();
        assert!(status.starts_with("Export failed: "), "{status}");
        assert!(
            app.pending_file_write.is_some(),
            "the bytes stay parked for a corrected path"
        );

        // The corrected path writes those very bytes.
        let good = dir.join("cities.geojson");
        let parked = app.pending_file_write.take().expect("just asserted");
        assert!(app.write_data_file(parked, good.clone()));
        assert_eq!(
            std::fs::read_to_string(&good).expect("the file is on disk"),
            export().content
        );
        let _removed = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_refused_export_says_why_and_not_merely_that_it_was_cancelled() {
        // `cancel_pending_export` writes its own "Export cancelled." line, so a
        // reason set BEFORE it would be overwritten and the user would be told
        // nothing actionable — the same ordering trap `resolve_project_save`
        // documents.
        let mut app = shell();
        app.report_file_write_cancelled(
            &PendingFileWrite::Export(export()),
            "Finish the file prompt that is already open first.",
        );
        let status = app.inner.status().unwrap_or_default().to_string();
        assert_eq!(status, "Finish the file prompt that is already open first.");
    }

    #[test]
    fn a_prompt_already_on_screen_refuses_the_write_rather_than_stacking_it() {
        let mut app = shell();
        app.path_prompt = Some(crate::file_dialog::PathPrompt::new(
            Ask::SaveProject,
            "project.oxigis.json",
            std::env::temp_dir(),
        ));
        app.resolve_file_write(PendingFileWrite::Export(export()));
        assert!(
            app.pending_file_write.is_none(),
            "the request is refused, not parked behind the other prompt"
        );
        let status = app.inner.status().unwrap_or_default().to_string();
        assert!(status.contains("already open"), "{status}");
    }
}
