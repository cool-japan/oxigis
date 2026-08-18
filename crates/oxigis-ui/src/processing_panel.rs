// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The Processing toolbox window: a form auto-generated from a
//! [`ToolDescriptor`]'s [`ParamSpec`] list, and the result/error area a run
//! reports into.
//!
//! Like [`crate::layer_panel`] and [`crate::style_panel`], [`draw`] never
//! mutates anything outside its own [`ProcessingPanelState`] argument — it
//! reports a user-requested run as a [`ProcessingAction`] for the caller
//! (`OxigisApp::apply_processing_action`) to execute, since running a tool
//! and routing its result both need `&mut OxigisApp` (the feature store, the
//! project, the layer this might add) that a stateless panel function has no
//! business reaching into directly.
//!
//! # The run in flight
//!
//! A run no longer happens inside the frame that asked for it. The panel
//! holds the [`ProcessingJob`] across frames, draws its progress and its
//! Cancel button, and the app's frame loop drives it through
//! [`ProcessingPanelState::poll_job`] — which is also where the browser
//! build's per-frame slice budget is tuned from the observed frame time. See
//! [`crate::processing_exec::ToolRun`] for the two ways a run is actually
//! driven.

use std::collections::BTreeMap;

use egui::{Color32, Ui};
use oxigis_core::{LayerId, ParamKind, ParamSpec, ProcessingRegistry, ToolDescriptor};

use crate::processing_exec::{ToolProgress, ToolRun, ToolRunState};

/// Color used for every inline validation/error hint in this window,
/// matching the red-hint precedent used by the File ▸ Open/Save modals.
const HINT_COLOR: Color32 = Color32::from_rgb(220, 80, 80);

/// Features a browser-driven run folds in on its first frame, before the
/// frame time has said anything about what this machine can afford.
///
/// Deliberately small: the cost of one feature spans four orders of magnitude
/// across the built-in tools (counting one, against Douglas-Peucker over a
/// coastline ring), so the opening budget is sized for the expensive end and
/// [`ProcessingPanelState::tune_slice`] grows it within a few frames when the
/// tool turns out to be cheap.
const INITIAL_SLICE: usize = 512;

/// Smallest per-frame slice. Below this the run makes so little progress per
/// frame that it would never finish on a large layer; a frame that overruns
/// at this size is a frame the tool simply costs.
const MIN_SLICE: usize = 64;

/// Largest per-frame slice, so a run that measures as cheap cannot grow its
/// budget until one slice freezes the frame after all.
const MAX_SLICE: usize = 65_536;

/// A frame slower than this (30 Hz) means the slice is too big.
const SLOW_FRAME_SECONDS: f32 = 1.0 / 30.0;

/// A frame faster than this (90 Hz) means there is headroom to spend.
const FAST_FRAME_SECONDS: f32 = 1.0 / 90.0;

/// One editable form field's live egui-side value, keyed by
/// [`ParamSpec::name`]. Mirrors [`ParamKind`] one-for-one; see
/// [`field_to_json`] for the JSON shape each variant encodes to when a tool
/// is run — the contract [`oxigis_core::ToolContext::params`] and any
/// [`oxigis_core::ToolExecutor`] decode against.
#[derive(Debug, Clone, PartialEq)]
enum ParamFieldState {
    /// A [`ParamKind::Number`] field's current value.
    Number(f64),
    /// A [`ParamKind::Text`] field's current buffer.
    Text(String),
    /// A [`ParamKind::Bool`] field's current value.
    Bool(bool),
    /// A [`ParamKind::LayerRef`] field's current selection, if any.
    LayerRef(Option<LayerId>),
    /// A [`ParamKind::Choice`] field's selected index into the spec's
    /// option list.
    Choice(usize),
}

/// What a run does with a result that is a **dataset**.
///
/// A scalar result (`bounds`, `feature_count`) ignores this entirely: there is
/// no layer to build, and the result area already shows the number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputDestination {
    /// Add the result to the project as a local layer.
    ///
    /// The default, and what every run did before this choice existed.
    #[default]
    Layer,
    /// Hand the GeoJSON document back for the user to copy out; the project
    /// keeps nothing.
    ///
    /// A tool result has no file to reference, so a result added as a layer is
    /// *embedded* in the project document
    /// ([`oxigis_core::VectorSource::InlineGeoJson`]) and is carried by every
    /// save from then on — a simplify over a large layer can double the file.
    /// This is the way out of that, and the only one this crate can offer on
    /// its own: `oxigis-ui` compiles to `wasm32` and owns no filesystem, so
    /// writing a file is a shell's job, exactly as File ▸ Save… is.
    GeoJsonText,
    /// Hand the document to the shell to **write to disk**, as a
    /// [`ProcessingFileRequest`].
    ///
    /// The same shape File ▸ Export PDF already uses (`take_pending_print`):
    /// this crate records what the user asked for and a shell that owns a
    /// filesystem (or, in a browser, a download) performs it. A build with
    /// neither drains nothing, which is why the request is *reported* rather
    /// than assumed — see `OxigisApp::take_pending_processing_save`.
    File,
}

/// Where a run's result should go, and what to call it — the Output group's
/// two answers, carried on the gesture rather than re-read afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OutputTarget {
    /// The name to give the result layer. **Empty** means "the caller names
    /// it", which is the historical `"{source} — {tool}"` rule; a blank field
    /// therefore behaves exactly as no field at all did.
    pub name: String,
    /// What to do with a dataset result.
    pub destination: OutputDestination,
}

/// What the app's feature selection holds right now, as much of it as the
/// Processing form needs to decide whether "selected features only" is
/// offerable.
///
/// Passed in rather than read here: the selection lives in the edit state
/// (`EditState::multi_selection`, mirrored into the attribute table), which a
/// stateless panel function has no business reaching into.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SelectionSummary {
    /// The layer the selection addresses, if anything is selected.
    pub layer: Option<LayerId>,
    /// How many of its features are selected.
    pub count: usize,
}

impl SelectionSummary {
    /// Whether this selection can restrict a run over `layer`.
    ///
    /// Both halves matter: an empty selection has nothing to restrict *to*,
    /// and a selection that addresses a **different** layer names feature
    /// indices that mean something else entirely in this one — filtering by
    /// them would silently run the tool over the wrong features rather than
    /// failing.
    #[must_use]
    pub fn covers(&self, layer: Option<LayerId>) -> bool {
        self.count > 0 && layer.is_some() && self.layer == layer
    }
}

/// A tool result the user asked to have written to a file, waiting for a
/// shell that owns a filesystem (or a browser download) to perform it.
///
/// Taken once through `OxigisApp::take_pending_processing_save`, exactly as
/// [`crate::print::PrintRequest`] is taken through `take_pending_print`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessingFileRequest {
    /// What the result was called — the basename a shell should suggest,
    /// without an extension.
    pub name: String,
    /// The pretty-printed GeoJSON document to write.
    pub content: String,
    /// How many features it holds, for the shell's own confirmation message.
    pub features: usize,
}

/// A Processing run in flight, with everything needed to route its result
/// when it lands.
///
/// The descriptor and output target are snapshotted here rather than re-read
/// from the form afterwards: a run takes seconds, and the user is free to
/// pick a different tool, rename the output or select another layer while it
/// works — none of which may retroactively change what the run they started
/// does.
#[derive(Debug)]
pub struct ProcessingJob {
    /// The tool that is running.
    descriptor: ToolDescriptor,
    /// The layer it read from, for naming the output.
    source: Option<LayerId>,
    /// Where its result goes.
    output: OutputTarget,
    /// The interruptible run itself.
    run: ToolRun,
}

impl ProcessingJob {
    /// Wraps a started run with the routing it will need.
    #[must_use]
    pub fn new(
        descriptor: ToolDescriptor,
        source: Option<LayerId>,
        output: OutputTarget,
        run: ToolRun,
    ) -> Self {
        Self {
            descriptor,
            source,
            output,
            run,
        }
    }

    /// How far the run has got.
    #[must_use]
    pub fn progress(&self) -> ToolProgress {
        self.run.progress()
    }

    /// The running tool's title, for a status line.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.descriptor.title
    }
}

/// A finished run, handed to the caller to route.
#[derive(Debug)]
pub struct FinishedRun {
    /// The tool that produced it.
    pub descriptor: ToolDescriptor,
    /// The layer it read from.
    pub source: Option<LayerId>,
    /// Where its result was asked to go.
    pub output: OutputTarget,
    /// The result, or the reason there is none.
    pub result: Result<serde_json::Value, String>,
}

/// What [`ProcessingPanelState::poll_job`] found this frame.
#[derive(Debug)]
pub enum ProcessingPoll {
    /// Nothing is running.
    Idle,
    /// A run is in flight; the caller should keep the frame loop awake.
    Running,
    /// A run ended and wants routing.
    Finished(Box<FinishedRun>),
    /// A run stopped because it was cancelled, naming the tool.
    Cancelled(String),
}

/// Persistent state of the Processing window across frames.
///
/// Ephemeral only — never serialized into [`oxigis_core::Project`], matching
/// that `OxigisApp::show_table` is likewise never saved: the window starts
/// fresh (nothing picked, no results) every run.
#[derive(Debug, Default)]
pub struct ProcessingPanelState {
    /// Which registered tool's form is showing, by id. [`None`] means
    /// nothing has been picked yet (an empty registry, or the window was
    /// just opened).
    selected_tool: Option<String>,
    /// The selected tool's field buffers, keyed by [`ParamSpec::name`].
    /// Rebuilt from scratch (cleared, then reseeded from
    /// [`ParamSpec::default`] where present) whenever [`Self::selected_tool`]
    /// changes — two tools' param shapes are unrelated, so nothing here is
    /// ever reused across a tool switch.
    fields: BTreeMap<String, ParamFieldState>,
    /// The last run's result, pretty-printed. Mutually exclusive with
    /// [`Self::last_error`]; both are cleared at the start of a fresh run.
    last_result: Option<String>,
    /// The last run's error message, if it failed.
    last_error: Option<String>,
    /// The Output group's name field. Empty is the meaningful default: it
    /// means "let the caller name it", so a user who never touches the field
    /// gets exactly the historical names.
    ///
    /// Deliberately NOT cleared on a tool switch, unlike [`Self::fields`]: a
    /// param buffer belongs to one tool's shape, while a name the user typed
    /// is their intent for the next run whichever tool that turns out to be.
    output_name: String,
    /// Where the next run's dataset result goes.
    output_destination: OutputDestination,
    /// Whether the next run should read only the selected features.
    ///
    /// Panel-level rather than a per-descriptor [`ParamSpec`], deliberately:
    /// every tool gets it for free and no descriptor has to declare it. Held
    /// even while it is not offerable (no selection, or one on another
    /// layer), so that re-selecting restores the user's own choice rather
    /// than silently resetting it — [`draw`] is what refuses to *act* on it,
    /// through [`SelectionSummary::covers`].
    selected_only: bool,
    /// The run in flight, if any.
    job: Option<ProcessingJob>,
    /// How many features a browser-driven run may fold in per frame; tuned
    /// from the observed frame time by [`Self::tune_slice`]. Zero until the
    /// first job starts, which [`Self::begin_job`] seeds.
    slice: usize,
    /// A result the user asked to have written to a file, waiting for a shell
    /// to take it.
    pending_file_save: Option<ProcessingFileRequest>,
}

impl ProcessingPanelState {
    /// Records a successful run's pretty-printed result, replacing any prior
    /// result or error — this crate keeps one slot, not a history (§4h).
    pub(crate) fn set_result(&mut self, text: String) {
        self.last_result = Some(text);
        self.last_error = None;
    }

    /// Records a failed run's error message, replacing any prior result.
    pub(crate) fn set_error(&mut self, message: String) {
        self.last_error = Some(message);
        self.last_result = None;
    }

    /// Adopts a freshly started run, replacing any prior result or error.
    ///
    /// Any run already in flight is cancelled first: the panel offers one
    /// Run button and one progress bar, so a second job would be work nothing
    /// can report on or stop.
    pub fn begin_job(&mut self, job: ProcessingJob) {
        if let Some(previous) = self.job.as_mut() {
            previous.run.cancel();
        }
        self.last_result = None;
        self.last_error = None;
        self.slice = INITIAL_SLICE;
        self.job = Some(job);
    }

    /// Whether a run is in flight.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.job.is_some()
    }

    /// The run in flight, if any.
    ///
    /// Exists so [`ProcessingJob`]'s own accessors are reachable from outside
    /// this module — the window draws the progress bar from here, and
    /// `OxigisApp::processing_progress` re-exports it for a shell that wants
    /// to mirror a long run somewhere else (a taskbar, a title bar).
    #[must_use]
    pub fn job(&self) -> Option<&ProcessingJob> {
        self.job.as_ref()
    }

    /// Drives (or observes) the run in flight, and reports what happened.
    ///
    /// `frame_seconds` is how long the previous frame took —
    /// `egui::InputState::stable_dt`. It is the whole basis of the browser
    /// build's slice budget: `wasm32` has no worker thread and no usable
    /// monotonic clock (`std::time::Instant::now` is not available on
    /// `wasm32-unknown-unknown`), so the frame time *is* the clock, read one
    /// frame late. A native run ignores the budget entirely — its worker
    /// thread sets its own pace.
    pub fn poll_job(&mut self, frame_seconds: f32) -> ProcessingPoll {
        if self.job.is_none() {
            return ProcessingPoll::Idle;
        }
        self.tune_slice(frame_seconds);
        let slice = self.slice;
        let Some(job) = self.job.as_mut() else {
            return ProcessingPoll::Idle;
        };
        match job.run.poll(slice) {
            ToolRunState::Running => ProcessingPoll::Running,
            ToolRunState::Finished(result) => match self.job.take() {
                Some(job) => ProcessingPoll::Finished(Box::new(FinishedRun {
                    descriptor: job.descriptor,
                    source: job.source,
                    output: job.output,
                    result,
                })),
                // Unreachable: the borrow above proved there is one.
                None => ProcessingPoll::Idle,
            },
            ToolRunState::Cancelled => {
                let title = job.descriptor.title.clone();
                self.job = None;
                ProcessingPoll::Cancelled(title)
            }
        }
    }

    /// Stops the run in flight, returning the tool's title for a status line.
    pub fn cancel_job(&mut self) -> Option<String> {
        let mut job = self.job.take()?;
        job.run.cancel();
        Some(job.descriptor.title)
    }

    /// Grows or shrinks the per-frame slice so a browser-driven run keeps the
    /// window repainting.
    ///
    /// One step per frame in either direction, halving on a frame slower than
    /// [`SLOW_FRAME_SECONDS`] and doubling on one faster than
    /// [`FAST_FRAME_SECONDS`], clamped to [`MIN_SLICE`]`..=`[`MAX_SLICE`]. The
    /// clamps are what keep the scheme honest at both ends: an unbounded
    /// shrink converges on a run that never finishes, and an unbounded growth
    /// converges on the frame-long freeze this whole mechanism exists to
    /// remove. A non-finite or zero `frame_seconds` — the very first frame,
    /// before egui has measured one — changes nothing.
    fn tune_slice(&mut self, frame_seconds: f32) {
        if self.slice == 0 {
            self.slice = INITIAL_SLICE;
        }
        if !frame_seconds.is_finite() || frame_seconds <= 0.0 {
            return;
        }
        if frame_seconds > SLOW_FRAME_SECONDS {
            self.slice = (self.slice / 2).max(MIN_SLICE);
        } else if frame_seconds < FAST_FRAME_SECONDS {
            self.slice = self.slice.saturating_mul(2).min(MAX_SLICE);
        }
    }

    /// Records a result the user asked to have written to a file.
    pub(crate) fn queue_file_save(&mut self, request: ProcessingFileRequest) {
        self.pending_file_save = Some(request);
    }

    /// Takes the pending file write, if any — the shell-side seam.
    pub(crate) fn take_file_save(&mut self) -> Option<ProcessingFileRequest> {
        self.pending_file_save.take()
    }

    /// Whether the user has asked for the next run to read only the selected
    /// features. Test-facing; [`draw`] is the only writer.
    #[cfg(test)]
    pub(crate) fn selected_only(&self) -> bool {
        self.selected_only
    }
}

/// A user-requested Processing gesture, reported by [`draw`] for the caller
/// to execute.
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingAction {
    /// Run `descriptor` with the given resolved parameter JSON, keyed by
    /// [`ParamSpec::name`] — ready for [`oxigis_core::ToolContext::params`].
    Run {
        /// The tool the user picked and asked to run.
        descriptor: ToolDescriptor,
        /// Every parameter's resolved value, encoded per each field's kind
        /// (a bare number/string/bool, a layer's raw id, or a `Choice`'s
        /// selected option string).
        params: BTreeMap<String, serde_json::Value>,
        /// What the Output group asked for: the result's name and where it
        /// goes. Snapshotted here rather than re-read from
        /// [`ProcessingPanelState`] afterwards, so the run is routed by what
        /// the user chose when they clicked Run.
        output: OutputTarget,
        /// Whether the run should read only the features selected on the
        /// input layer. Only ever `true` when the selection actually covers
        /// the picked layer ([`SelectionSummary::covers`]), so the caller
        /// never has to re-check that.
        selected_only: bool,
    },
    /// Stop the run in flight.
    Cancel,
}

/// Draws the Processing window's contents (tool picker, param form, Run
/// button, result/error area) into `ui`, returning a [`ProcessingAction`]
/// when the user clicked Run this frame.
///
/// `layer_options` is the caller's current
/// [`crate::app::OxigisApp::local_vector_layer_options`] (top-of-stack-first,
/// loaded local vector layers only) and `selection` its current selected
/// layer, if any — both read-only inputs this function uses to validate and
/// pre-seed `LayerRef` fields, never to reach back into the app itself.
/// `selected_features` is the app's current feature selection, which decides
/// whether the "selected features only" checkbox is offerable at all.
pub fn draw(
    ui: &mut Ui,
    registry: &ProcessingRegistry,
    state: &mut ProcessingPanelState,
    layer_options: &[(LayerId, &str)],
    selection: Option<LayerId>,
    selected_features: SelectionSummary,
) -> Option<ProcessingAction> {
    if registry.is_empty() {
        ui.weak("No processing tools are registered.");
        return None;
    }

    draw_tool_picker(ui, registry, state);

    let selected_id = state.selected_tool.clone()?;
    let Ok(descriptor) = registry.get(&selected_id) else {
        // The picked id no longer resolves — nothing in this build can
        // unregister a tool at runtime, but `ProcessingRegistry::register`
        // could overwrite one under a future dynamic-tool feature, so this
        // heals rather than draws a form for a descriptor that is gone.
        state.selected_tool = None;
        state.fields.clear();
        return None;
    };

    ui.separator();
    ui.label(&descriptor.description);
    ui.separator();

    reseed_layer_ref_fields(&mut state.fields, descriptor, layer_options, selection);

    let mut all_valid = true;
    for spec in &descriptor.params {
        // `.and_modify` heals a field left over from a *different*
        // `ParamKind` under this same parameter name — see `matches_kind`'s
        // docs for why that is reachable, not merely defensive.
        let field = state
            .fields
            .entry(spec.name.clone())
            .and_modify(|field| {
                if !matches_kind(&spec.kind, field) {
                    *field = field_for(spec);
                }
            })
            .or_insert_with(|| field_for(spec));
        draw_param_field(ui, &descriptor.id, spec, field, layer_options);
        all_valid &= field_is_valid(spec, field, layer_options);
    }

    let picked = picked_layer(&state.fields, descriptor);
    let restrict = draw_selection_row(ui, state, selected_features, picked);
    draw_output_group(ui, state);

    ui.separator();
    let mut action = None;
    let running = state.is_running();
    ui.horizontal(|ui| {
        if ui
            .add_enabled(all_valid && !running, egui::Button::new("Run"))
            .clicked()
        {
            let params = descriptor
                .params
                .iter()
                .map(|spec| {
                    let value = state
                        .fields
                        .get(&spec.name)
                        .map_or(serde_json::Value::Null, |field| {
                            field_to_json(&spec.kind, field)
                        });
                    (spec.name.clone(), value)
                })
                .collect();
            state.last_result = None;
            state.last_error = None;
            action = Some(ProcessingAction::Run {
                descriptor: descriptor.clone(),
                params,
                output: OutputTarget {
                    // Trimmed here, once, so the caller never has to decide
                    // whether a field holding only spaces counts as "named".
                    name: state.output_name.trim().to_string(),
                    destination: state.output_destination,
                },
                selected_only: restrict,
            });
        }
        if running && ui.button("Cancel").clicked() {
            action = Some(ProcessingAction::Cancel);
        }
    });
    draw_progress(ui, state.job.as_ref());

    if let Some(error) = &state.last_error {
        ui.colored_label(HINT_COLOR, error);
    }
    if let Some(result) = &state.last_result {
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Result:");
            if ui.button("Copy").clicked() {
                ui.ctx().copy_text(result.clone());
            }
        });
        // A `&mut &str` (not `&mut String`) is the documented egui idiom for
        // "selectable but not editable": `&str`'s `TextBuffer` impl refuses
        // every mutation, so the field keeps full cursor/selection support
        // without needing `.interactive(false)`, which disabled selection
        // along with editing and left a `bounds` result impossible to copy
        // out of the app.
        let mut display = result.as_str();
        ui.add(
            egui::TextEdit::multiline(&mut display)
                .desired_rows(10)
                .desired_width(360.0),
        );
    }

    action
}

/// The layer `descriptor`'s [`ParamKind::LayerRef`] field currently names, if
/// it names one.
///
/// The *first* such parameter, matching what
/// `OxigisApp::run_processing_tool` resolves — a tool with two layer
/// parameters is not a shape this build supports, and this deliberately does
/// not pretend otherwise.
fn picked_layer(
    fields: &BTreeMap<String, ParamFieldState>,
    descriptor: &ToolDescriptor,
) -> Option<LayerId> {
    let spec = descriptor
        .params
        .iter()
        .find(|spec| matches!(spec.kind, ParamKind::LayerRef))?;
    match fields.get(&spec.name) {
        Some(ParamFieldState::LayerRef(picked)) => *picked,
        _ => None,
    }
}

/// Draws the "selected features only" row and answers whether the next run
/// should actually be restricted.
///
/// The checkbox is only *live* when the app's selection covers the picked
/// input layer; otherwise it is drawn disabled with the reason beside it, and
/// the answer is `false` however the stored preference stands. That is the
/// whole safety property: an index set belonging to another layer must never
/// silently filter this one (it would run the tool over unrelated features and
/// report success), and the stored preference survives so that re-selecting
/// restores the user's own choice.
fn draw_selection_row(
    ui: &mut Ui,
    state: &mut ProcessingPanelState,
    selected: SelectionSummary,
    picked: Option<LayerId>,
) -> bool {
    let offerable = selected.covers(picked);
    ui.horizontal(|ui| {
        ui.add_enabled(
            offerable,
            egui::Checkbox::new(&mut state.selected_only, "Selected features only"),
        )
        .on_hover_text(
            "Run the tool over just the features selected on the input layer, \
             instead of the whole layer.",
        );
        if offerable {
            ui.weak(match selected.count {
                1 => "1 selected".to_string(),
                count => format!("{count} selected"),
            });
        } else if selected.count == 0 {
            ui.weak("nothing is selected");
        } else {
            ui.weak("the selection is on another layer");
        }
    });
    offerable && state.selected_only
}

/// Draws the progress bar and running-tool line for a run in flight.
///
/// Nothing at all when idle: an empty bar sitting under the Run button would
/// read as a run that is stuck rather than one that never started.
fn draw_progress(ui: &mut Ui, job: Option<&ProcessingJob>) {
    let Some(job) = job else { return };
    let progress = job.progress();
    ui.add(egui::ProgressBar::new(progress.fraction()).text(format!(
        "{} \u{2014} {}",
        job.title(),
        progress.label()
    )));
}

/// Draws the Output group: what the result will be called, and whether it
/// becomes a layer, a document to copy out, or a file on disk.
///
/// Drawn for every tool, including the scalar ones, and deliberately so: which
/// tools return a dataset is a fact about the *result*, not about the
/// descriptor (§1.5's routing rule keys off the value's `type`), so a form that
/// hid the group for some tools would be guessing. Both controls are simply
/// ignored by a scalar result.
fn draw_output_group(ui: &mut Ui, state: &mut ProcessingPanelState) {
    ui.separator();
    ui.label("Output");
    ui.horizontal(|ui| {
        ui.label("Name");
        ui.add(
            egui::TextEdit::singleline(&mut state.output_name)
                .hint_text("named after the source layer")
                .desired_width(200.0),
        )
        .on_hover_text(
            "What to call the result layer. Left blank, it is named after the \
             layer it came from and the tool that made it.",
        );
    });
    ui.horizontal(|ui| {
        ui.radio_value(
            &mut state.output_destination,
            OutputDestination::Layer,
            "Temporary layer",
        )
        .on_hover_text(
            "Add the result to the project as a layer with no file behind it. \
             Its GeoJSON is embedded in the project file, so a large result \
             makes every save large.",
        );
        ui.radio_value(
            &mut state.output_destination,
            OutputDestination::File,
            "Save to file\u{2026}",
        )
        .on_hover_text(
            "Hand the result to this build's shell to write to disk. \
             OxiGIS's UI owns no filesystem (it also runs in a browser), so a \
             build with no file writer attached reports that instead of \
             writing.",
        );
        ui.radio_value(
            &mut state.output_destination,
            OutputDestination::GeoJsonText,
            "GeoJSON to copy out",
        )
        .on_hover_text(
            "Show the result's GeoJSON to copy and save yourself, instead of \
             adding a layer; the project document keeps nothing.",
        );
    });
}

/// Draws the tool-picker `ComboBox`, clearing and reseeding
/// [`ProcessingPanelState::fields`] whenever the pick changes.
fn draw_tool_picker(ui: &mut Ui, registry: &ProcessingRegistry, state: &mut ProcessingPanelState) {
    let mut selected = state.selected_tool.clone();
    let label = selected
        .as_deref()
        .and_then(|id| registry.get(id).ok())
        .map_or("Select a tool", |descriptor| descriptor.title.as_str());
    egui::ComboBox::from_id_salt("oxigis_processing_tool_picker")
        .selected_text(label)
        .show_ui(ui, |ui| {
            for descriptor in registry.iter() {
                ui.selectable_value(
                    &mut selected,
                    Some(descriptor.id.clone()),
                    &descriptor.title,
                );
            }
        });
    if selected == state.selected_tool {
        return;
    }
    state.fields.clear();
    state.last_result = None;
    state.last_error = None;
    if let Some(descriptor) = selected.as_deref().and_then(|id| registry.get(id).ok()) {
        for spec in &descriptor.params {
            state.fields.insert(spec.name.clone(), field_for(spec));
        }
    }
    state.selected_tool = selected;
}

/// Per-frame stale-id / pre-seed pass for every `LayerRef` field of
/// `descriptor`, run before the fields are drawn. Two rules, in order:
///
/// 1. a field holding a layer id no longer in `layer_options` is reset to
///    [`None`] (the user removed that layer since the last frame);
/// 2. a field that is [`None`] is seeded from `selection`, if `selection`
///    names a layer currently in `layer_options` — so opening the toolbox
///    with a layer already selected pre-picks it, and a field rule 1 just
///    cleared heals itself in the same pass when possible.
///
/// A value the user (or a prior pre-seed) already set is never overwritten —
/// this only ever touches a field that is currently [`None`].
fn reseed_layer_ref_fields(
    fields: &mut BTreeMap<String, ParamFieldState>,
    descriptor: &ToolDescriptor,
    layer_options: &[(LayerId, &str)],
    selection: Option<LayerId>,
) {
    let eligible = |id: LayerId| layer_options.iter().any(|(option_id, _)| *option_id == id);
    for spec in &descriptor.params {
        if !matches!(spec.kind, ParamKind::LayerRef) {
            continue;
        }
        let Some(ParamFieldState::LayerRef(value)) = fields.get_mut(&spec.name) else {
            continue;
        };
        if let Some(id) = *value
            && !eligible(id)
        {
            *value = None;
        }
        if value.is_none()
            && let Some(id) = selection
            && eligible(id)
        {
            *value = Some(id);
        }
    }
}

/// How a [`ParamKind::Number`]'s bounds should be drawn — see
/// [`number_widget_kind`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum NumberWidgetKind {
    /// Both bounds are finite and ordered: an `egui::Slider` over `lo..=hi`.
    Slider {
        /// Inclusive lower bound.
        lo: f64,
        /// Inclusive upper bound.
        hi: f64,
    },
    /// At least one bound is missing, non-finite, or the two are inverted:
    /// an `egui::DragValue` over `lo..=hi` (already normalised so
    /// `lo <= hi`).
    DragValue {
        /// Inclusive lower bound, always `<= hi`.
        lo: f64,
        /// Inclusive upper bound, always `>= lo`.
        hi: f64,
    },
}

/// Chooses how a [`ParamKind::Number`]'s `min`/`max` should be drawn.
///
/// [`ParamSpec`]/[`ProcessingRegistry::register`] perform no validation on
/// `min`/`max` — nothing stops a descriptor shipping `min > max` or NaN — so
/// this always returns a safe widget spec: a bare `Slider::new(&mut v,
/// b..=a)` with `a > b` is undefined/panicky UI behaviour, which `DragValue`
/// with normalised bounds never risks.
fn number_widget_kind(min: Option<f64>, max: Option<f64>) -> NumberWidgetKind {
    if let (Some(lo), Some(hi)) = (min, max)
        && lo.is_finite()
        && hi.is_finite()
        && lo <= hi
    {
        return NumberWidgetKind::Slider { lo, hi };
    }
    let lo = min
        .filter(|value| value.is_finite())
        .unwrap_or(f64::NEG_INFINITY);
    let hi = max
        .filter(|value| value.is_finite())
        .unwrap_or(f64::INFINITY);
    if lo > hi {
        NumberWidgetKind::DragValue { lo: hi, hi: lo }
    } else {
        NumberWidgetKind::DragValue { lo, hi }
    }
}

/// Draws one parameter's widget, dispatching on its [`ParamKind`].
///
/// A field's variant is not supposed to drift from the `ParamKind` it was
/// built for ([`field_for`] matches them one-for-one), and [`draw`]'s
/// seeding loop enforces that every frame via [`matches_kind`] — so the
/// catch-all below should be unreachable in practice. It stays a visible red
/// hint rather than a silent no-op so a future gap in that enforcement is
/// obvious instead of shipping an empty space where a field belongs.
fn draw_param_field(
    ui: &mut Ui,
    tool_id: &str,
    spec: &ParamSpec,
    field: &mut ParamFieldState,
    layer_options: &[(LayerId, &str)],
) {
    match (&spec.kind, field) {
        (ParamKind::Number { min, max }, ParamFieldState::Number(value)) => {
            draw_number_field(ui, spec, *min, *max, value);
        }
        (ParamKind::Text, ParamFieldState::Text(text)) => {
            ui.horizontal(|ui| {
                ui.label(&spec.name);
                ui.text_edit_singleline(text);
            });
        }
        (ParamKind::Bool, ParamFieldState::Bool(value)) => {
            ui.checkbox(value, &spec.name);
        }
        (ParamKind::LayerRef, ParamFieldState::LayerRef(selected)) => {
            draw_layer_ref_field(ui, tool_id, spec, selected, layer_options);
        }
        (ParamKind::Choice(options), ParamFieldState::Choice(index)) => {
            draw_choice_field(ui, tool_id, spec, options, index);
        }
        _ => {
            ui.colored_label(
                HINT_COLOR,
                format!("{}: no matching field state (registry bug)", spec.name),
            );
        }
    }
}

/// Draws a [`ParamKind::Number`] field per [`number_widget_kind`]'s choice.
///
/// Only the `DragValue` arm needs a chosen [`drag_speed`]: `egui::Slider`
/// derives its own drag speed from the visible track's pixel width and its
/// bounded range (`current_gradient`, egui `widgets/slider.rs`), so it is
/// already scaled to whatever the field holds and has no `.speed()` builder
/// method to override it with.
fn draw_number_field(
    ui: &mut Ui,
    spec: &ParamSpec,
    min: Option<f64>,
    max: Option<f64>,
    value: &mut f64,
) {
    match number_widget_kind(min, max) {
        NumberWidgetKind::Slider { lo, hi } => {
            ui.add(egui::Slider::new(value, lo..=hi).text(&spec.name));
        }
        NumberWidgetKind::DragValue { lo, hi } => {
            ui.horizontal(|ui| {
                ui.label(&spec.name);
                ui.add(egui::DragValue::new(value).range(lo..=hi).speed(drag_speed(
                    spec.default.as_ref(),
                    lo,
                    hi,
                )));
            });
        }
    }
}

/// Chooses a [`egui::DragValue::speed`] scaled to the field's own magnitude,
/// rather than trusting egui's default of `1.0` per logical pixel of drag.
///
/// egui derives both the drag step size and the number of decimals shown
/// from `speed` (`auto_decimals = (aim_radius / speed).log10().ceil()`,
/// egui `widgets/drag_value.rs`), so a `speed` far larger than the values a
/// field actually holds both destroys them on the first drag *and* hides
/// the precision that would have shown the problem — a field parked at
/// `0.001` snaps to `0.1`-sized steps under the un-scaled default.
///
/// [`ParamKind::Number`] carries no scale of its own (only `min`/`max`, and
/// a `simplify`-shaped field leaves `max` unbounded, see
/// [`number_widget_kind`]), so the descriptor's own *default* value is the
/// best signal available here: a field defaulting to `0.001` is a field
/// whose meaningful steps are smaller still. `lo`/`hi` (by then already
/// normalised finite-or-infinite by [`number_widget_kind`]) are the fallback
/// when there is no usable default, and `1.0` — egui's own default — is what
/// is left once neither signal is usable.
fn drag_speed(default: Option<&serde_json::Value>, lo: f64, hi: f64) -> f64 {
    if let Some(default) = default.and_then(serde_json::Value::as_f64)
        && default.is_finite()
        && default != 0.0
    {
        return (default.abs() / 100.0).max(f64::MIN_POSITIVE);
    }
    if lo.is_finite() && hi.is_finite() && hi > lo {
        return ((hi - lo) / 100.0).max(f64::MIN_POSITIVE);
    }
    1.0
}

/// Draws a [`ParamKind::LayerRef`] field: a `ComboBox` over `layer_options`,
/// plus an inline hint when there is nothing to pick from.
fn draw_layer_ref_field(
    ui: &mut Ui,
    tool_id: &str,
    spec: &ParamSpec,
    selected: &mut Option<LayerId>,
    layer_options: &[(LayerId, &str)],
) {
    let label = selected
        .and_then(|id| layer_options.iter().find(|(option_id, _)| *option_id == id))
        .map_or("Select a layer", |(_, name)| *name);
    ui.horizontal(|ui| {
        ui.label(&spec.name);
        egui::ComboBox::from_id_salt(("oxigis_processing_param", tool_id, spec.name.as_str()))
            .selected_text(label)
            .show_ui(ui, |ui| {
                for (id, name) in layer_options {
                    ui.selectable_value(selected, Some(*id), *name);
                }
            });
    });
    if layer_options.is_empty() {
        ui.colored_label(HINT_COLOR, "no local vector layers are loaded");
    }
}

/// Draws a [`ParamKind::Choice`] field: a `ComboBox` over `options`, or a
/// disabled placeholder when the descriptor shipped no options at all (a
/// registry bug, not a user-fixable state).
fn draw_choice_field(
    ui: &mut Ui,
    tool_id: &str,
    spec: &ParamSpec,
    options: &[String],
    index: &mut usize,
) {
    if options.is_empty() {
        ui.add_enabled(false, egui::Button::new(&spec.name));
        ui.colored_label(
            HINT_COLOR,
            format!("no options configured for {}", spec.name),
        );
        return;
    }
    // A stored index surviving from a differently-shaped descriptor (or a
    // future edited-in-place `ParamSpec`) could be out of range; clamp so
    // `options[*index]` below never panics.
    *index = (*index).min(options.len() - 1);
    let label = options[*index].as_str();
    ui.horizontal(|ui| {
        ui.label(&spec.name);
        egui::ComboBox::from_id_salt(("oxigis_processing_param", tool_id, spec.name.as_str()))
            .selected_text(label)
            .show_ui(ui, |ui| {
                for (position, option) in options.iter().enumerate() {
                    ui.selectable_value(index, position, option);
                }
            });
    });
}

/// Whether `field`'s current value is acceptable for a Run, per each
/// [`ParamKind`]'s rule:
///
/// * `Number` is always valid — `Slider`/`DragValue` only ever produce a
///   finite value by construction.
/// * `Text` is valid unless `required` and the trimmed buffer is empty.
/// * `Bool` is always valid.
/// * `LayerRef` is valid iff a layer is picked and it is still in
///   `layer_options` — regardless of `required` (an *optional* `LayerRef`
///   isn't a case the registry has today; treated as always effectively
///   required until one exists).
/// * `Choice` is valid iff the spec has at least one option — a config
///   problem the user cannot fix by picking differently.
/// * anything else — a field whose variant does not match `spec.kind` —
///   is invalid. [`draw`]'s seeding loop should never let this combination
///   reach here (see [`matches_kind`]), so this fails *closed*: Run staying
///   disabled on a broken invariant is a bounded surprise, an invisible
///   field with `null` params and an enabled Run button is not.
fn field_is_valid(
    spec: &ParamSpec,
    field: &ParamFieldState,
    layer_options: &[(LayerId, &str)],
) -> bool {
    match (&spec.kind, field) {
        (ParamKind::Number { .. }, ParamFieldState::Number(_)) => true,
        (ParamKind::Text, ParamFieldState::Text(text)) => !spec.required || !text.trim().is_empty(),
        (ParamKind::Bool, ParamFieldState::Bool(_)) => true,
        (ParamKind::LayerRef, ParamFieldState::LayerRef(selected)) => {
            selected.is_some_and(|id| layer_options.iter().any(|(option_id, _)| *option_id == id))
        }
        (ParamKind::Choice(options), ParamFieldState::Choice(_)) => !options.is_empty(),
        _ => false,
    }
}

/// Whether `field`'s variant is the one [`field_for`] would build for a
/// parameter of kind `kind` — i.e. whether a field already stored under a
/// parameter's name still matches that parameter's current shape.
///
/// Exists for the moment [`ProcessingRegistry::register`] re-registers a
/// tool under an id already in the registry (its own docs call this
/// "overwriting any prior descriptor") with a parameter of the same *name*
/// but a different [`ParamKind`]: [`ProcessingPanelState::fields`] is keyed
/// on name alone, so the stale field would otherwise survive under the new
/// spec, and [`draw_param_field`]/[`field_is_valid`] would then have to cope
/// with a `(ParamKind, ParamFieldState)` pair [`field_for`] never builds
/// together — an invisible field and, before this existed, a silently
/// enabled Run button.
fn matches_kind(kind: &ParamKind, field: &ParamFieldState) -> bool {
    matches!(
        (kind, field),
        (ParamKind::Number { .. }, ParamFieldState::Number(_))
            | (ParamKind::Text, ParamFieldState::Text(_))
            | (ParamKind::Bool, ParamFieldState::Bool(_))
            | (ParamKind::LayerRef, ParamFieldState::LayerRef(_))
            | (ParamKind::Choice(_), ParamFieldState::Choice(_))
    )
}

/// The field state a [`ParamSpec`] starts with: seeded from
/// [`ParamSpec::default`] where present and shaped right, else a neutral
/// value for its [`ParamKind`].
fn field_for(spec: &ParamSpec) -> ParamFieldState {
    match &spec.kind {
        ParamKind::Number { .. } => {
            let value = spec
                .default
                .as_ref()
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            ParamFieldState::Number(value)
        }
        ParamKind::Text => {
            let value = spec
                .default
                .as_ref()
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            ParamFieldState::Text(value)
        }
        ParamKind::Bool => {
            let value = spec
                .default
                .as_ref()
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            ParamFieldState::Bool(value)
        }
        // A layer can never be a sensible compile-time default — there is no
        // "layer zero" — so this always starts unset; §4d's pre-seed pass
        // fills it from the current selection on the first draw instead.
        ParamKind::LayerRef => ParamFieldState::LayerRef(None),
        ParamKind::Choice(options) => {
            let index = spec
                .default
                .as_ref()
                .and_then(serde_json::Value::as_str)
                .and_then(|text| options.iter().position(|option| option == text))
                .unwrap_or(0);
            ParamFieldState::Choice(index)
        }
    }
}

/// Encodes a field's current value into the JSON shape
/// [`oxigis_core::ToolContext::params`] and any [`oxigis_core::ToolExecutor`]
/// expect it in:
///
/// * `Number(v)` → the number itself.
/// * `Text(s)` → the string itself.
/// * `Bool(b)` → the bool itself.
/// * `LayerRef(Some(id))` → `id.get()`, a bare `u64` — [`LayerId`]'s own
///   `#[serde(transparent)]` `Deserialize` reads this back unmodified.
///   `LayerRef(None)` encodes as `null`; unreachable through [`draw`] itself
///   (Run is disabled while any `LayerRef` field is unset), kept total here
///   so this function never panics on a field built by hand (as tests do).
/// * `Choice(index)` → the **option string** at that index, not the index —
///   an executor should never have to know the UI's internal ordering. An
///   out-of-range index (should not happen; [`draw_choice_field`] clamps it)
///   likewise encodes as `null` rather than panicking.
fn field_to_json(kind: &ParamKind, field: &ParamFieldState) -> serde_json::Value {
    match field {
        ParamFieldState::Number(value) => serde_json::json!(value),
        ParamFieldState::Text(text) => serde_json::json!(text),
        ParamFieldState::Bool(value) => serde_json::json!(value),
        ParamFieldState::LayerRef(Some(id)) => serde_json::json!(id.get()),
        ParamFieldState::LayerRef(None) => serde_json::Value::Null,
        ParamFieldState::Choice(index) => match kind {
            ParamKind::Choice(options) => options
                .get(*index)
                .cloned()
                .map_or(serde_json::Value::Null, serde_json::Value::String),
            _ => serde_json::Value::Null,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer_spec() -> ParamSpec {
        ParamSpec {
            name: "layer".to_string(),
            kind: ParamKind::LayerRef,
            required: true,
            default: None,
        }
    }

    // ---- JSON encoding: one test per `ParamKind` variant -----------------

    #[test]
    fn number_field_encodes_as_the_bare_number() {
        let kind = ParamKind::Number {
            min: None,
            max: None,
        };
        assert_eq!(
            field_to_json(&kind, &ParamFieldState::Number(2.5)),
            serde_json::json!(2.5)
        );
    }

    #[test]
    fn text_field_encodes_as_the_bare_string() {
        assert_eq!(
            field_to_json(&ParamKind::Text, &ParamFieldState::Text("hi".to_string())),
            serde_json::json!("hi")
        );
    }

    #[test]
    fn bool_field_encodes_as_the_bare_bool() {
        assert_eq!(
            field_to_json(&ParamKind::Bool, &ParamFieldState::Bool(true)),
            serde_json::json!(true)
        );
    }

    #[test]
    fn layer_ref_field_encodes_as_the_raw_layer_id() {
        let id = LayerId::from_raw(42);
        assert_eq!(
            field_to_json(&ParamKind::LayerRef, &ParamFieldState::LayerRef(Some(id))),
            serde_json::json!(42)
        );
        assert_eq!(
            field_to_json(&ParamKind::LayerRef, &ParamFieldState::LayerRef(None)),
            serde_json::Value::Null
        );
    }

    #[test]
    fn choice_field_encodes_as_the_option_string_not_the_index() {
        let kind = ParamKind::Choice(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(
            field_to_json(&kind, &ParamFieldState::Choice(1)),
            serde_json::json!("b")
        );
        assert_eq!(
            field_to_json(&kind, &ParamFieldState::Choice(99)),
            serde_json::Value::Null,
            "an out-of-range index must not panic"
        );
    }

    // ---- Number widget rule -------------------------------------------

    #[test]
    fn number_widget_kind_picks_slider_only_for_finite_ordered_bounds() {
        assert_eq!(
            number_widget_kind(Some(0.0), Some(10.0)),
            NumberWidgetKind::Slider { lo: 0.0, hi: 10.0 }
        );
        assert_eq!(
            number_widget_kind(None, Some(10.0)),
            NumberWidgetKind::DragValue {
                lo: f64::NEG_INFINITY,
                hi: 10.0
            }
        );
        assert_eq!(
            number_widget_kind(Some(0.0), None),
            NumberWidgetKind::DragValue {
                lo: 0.0,
                hi: f64::INFINITY
            }
        );
        assert_eq!(
            number_widget_kind(Some(10.0), Some(0.0)),
            NumberWidgetKind::DragValue { lo: 0.0, hi: 10.0 },
            "inverted bounds must be swapped, not fed to Slider as-is"
        );
        match number_widget_kind(Some(f64::NAN), Some(10.0)) {
            NumberWidgetKind::DragValue { lo, hi } => {
                assert_eq!(lo, f64::NEG_INFINITY);
                assert_eq!(hi, 10.0);
            }
            other => panic!("NaN must fall back to DragValue, got {other:?}"),
        }
    }

    // ---- DragValue speed ---------------------------------------------------

    #[test]
    fn drag_speed_scales_to_a_finite_nonzero_defaults_magnitude() {
        let tolerance_default = serde_json::json!(0.001);
        assert_eq!(
            drag_speed(Some(&tolerance_default), 0.0, f64::INFINITY),
            0.00001,
            "the shipped `tolerance_deg` default (0.001) must not snap to egui's 1.0-per-pixel default"
        );
        let negative_default = serde_json::json!(-250.0);
        assert_eq!(
            drag_speed(Some(&negative_default), 0.0, f64::INFINITY),
            2.5,
            "the speed is a magnitude — sign must not flip it negative"
        );
    }

    #[test]
    fn drag_speed_falls_back_to_the_bounds_span_without_a_usable_default() {
        assert_eq!(drag_speed(None, 0.0, 10.0), 0.1);
        let zero = serde_json::json!(0.0);
        assert_eq!(
            drag_speed(Some(&zero), 0.0, 10.0),
            0.1,
            "a zero default carries no scale of its own"
        );
        let not_a_number = serde_json::json!("nope");
        assert_eq!(
            drag_speed(Some(&not_a_number), 0.0, 10.0),
            0.1,
            "a non-numeric default must fall through as_f64, not panic"
        );
    }

    #[test]
    fn drag_speed_falls_back_to_eguis_own_default_with_no_usable_signal_at_all() {
        assert_eq!(drag_speed(None, f64::NEG_INFINITY, f64::INFINITY), 1.0);
    }

    // ---- Validation ------------------------------------------------------

    #[test]
    fn empty_choice_options_are_always_invalid() {
        let spec = ParamSpec {
            name: "mode".to_string(),
            kind: ParamKind::Choice(vec![]),
            required: false,
            default: None,
        };
        assert!(!field_is_valid(&spec, &ParamFieldState::Choice(0), &[]));
    }

    #[test]
    fn a_required_text_field_is_invalid_only_while_blank() {
        let spec = ParamSpec {
            name: "name".to_string(),
            kind: ParamKind::Text,
            required: true,
            default: None,
        };
        assert!(!field_is_valid(
            &spec,
            &ParamFieldState::Text("   ".to_string()),
            &[]
        ));
        assert!(field_is_valid(
            &spec,
            &ParamFieldState::Text("x".to_string()),
            &[]
        ));
    }

    #[test]
    fn layer_ref_is_valid_only_when_picked_and_still_eligible() {
        let spec = layer_spec();
        let id = LayerId::new();
        let options = [(id, "cities")];
        assert!(!field_is_valid(
            &spec,
            &ParamFieldState::LayerRef(None),
            &options
        ));
        assert!(field_is_valid(
            &spec,
            &ParamFieldState::LayerRef(Some(id)),
            &options
        ));
        assert!(!field_is_valid(
            &spec,
            &ParamFieldState::LayerRef(Some(LayerId::new())),
            &options
        ));
    }

    // ---- Stale-id / re-seed -----------------------------------------------

    #[test]
    fn a_stale_layer_ref_is_cleared_and_reseeded_from_the_selection() {
        let descriptor = ToolDescriptor {
            id: "bounds".to_string(),
            title: "Layer Bounds".to_string(),
            description: String::new(),
            params: vec![layer_spec()],
        };
        let removed = LayerId::new();
        let mut fields = BTreeMap::new();
        fields.insert(
            "layer".to_string(),
            ParamFieldState::LayerRef(Some(removed)),
        );

        // The stored id is gone from `layer_options` and there is no
        // eligible selection yet: the field must heal to `None`, not stay
        // pointed at a layer that no longer exists.
        reseed_layer_ref_fields(&mut fields, &descriptor, &[], None);
        assert_eq!(fields.get("layer"), Some(&ParamFieldState::LayerRef(None)));

        // Now the selection names a layer that *is* eligible: the same pass
        // re-seeds it in one step.
        let replacement = LayerId::new();
        let options = [(replacement, "new layer")];
        reseed_layer_ref_fields(&mut fields, &descriptor, &options, Some(replacement));
        assert_eq!(
            fields.get("layer"),
            Some(&ParamFieldState::LayerRef(Some(replacement)))
        );
    }

    #[test]
    fn reseed_never_overwrites_a_value_the_user_already_set() {
        let descriptor = ToolDescriptor {
            id: "bounds".to_string(),
            title: "Layer Bounds".to_string(),
            description: String::new(),
            params: vec![layer_spec()],
        };
        let chosen = LayerId::new();
        let other = LayerId::new();
        let mut fields = BTreeMap::new();
        fields.insert("layer".to_string(), ParamFieldState::LayerRef(Some(chosen)));
        let options = [(chosen, "a"), (other, "b")];
        // A different, still-eligible layer is selected on the map — the
        // user's own pick must not be silently swapped for it.
        reseed_layer_ref_fields(&mut fields, &descriptor, &options, Some(other));
        assert_eq!(
            fields.get("layer"),
            Some(&ParamFieldState::LayerRef(Some(chosen)))
        );
    }

    // ---- Defaults ----------------------------------------------------------

    #[test]
    fn field_for_seeds_from_the_param_specs_default_where_shaped_right() {
        let number = ParamSpec {
            name: "n".to_string(),
            kind: ParamKind::Number {
                min: None,
                max: None,
            },
            required: false,
            default: Some(serde_json::json!(3.5)),
        };
        assert_eq!(field_for(&number), ParamFieldState::Number(3.5));

        let choice = ParamSpec {
            name: "mode".to_string(),
            kind: ParamKind::Choice(vec!["a".to_string(), "b".to_string()]),
            required: false,
            default: Some(serde_json::json!("b")),
        };
        assert_eq!(field_for(&choice), ParamFieldState::Choice(1));

        // A default that doesn't match any option falls back to index 0
        // rather than panicking on an out-of-range lookup.
        let unmatched = ParamSpec {
            default: Some(serde_json::json!("nope")),
            ..choice
        };
        assert_eq!(field_for(&unmatched), ParamFieldState::Choice(0));

        assert_eq!(field_for(&layer_spec()), ParamFieldState::LayerRef(None));
    }

    #[test]
    fn drawing_the_window_does_not_panic_with_a_populated_registry() {
        let registry = oxigis_core::builtin_registry();
        let mut state = ProcessingPanelState::default();
        let id = LayerId::new();
        let options = [(id, "cities")];
        egui::__run_test_ui(|ui| {
            let action = draw(
                ui,
                &registry,
                &mut state,
                &options,
                Some(id),
                SelectionSummary::default(),
            );
            // No simulated click happened, so drawing must report nothing.
            assert_eq!(action, None);
        });
    }

    #[test]
    fn drawing_with_an_empty_registry_does_not_panic() {
        let registry = ProcessingRegistry::new();
        let mut state = ProcessingPanelState::default();
        egui::__run_test_ui(|ui| {
            let action = draw(
                ui,
                &registry,
                &mut state,
                &[],
                None,
                SelectionSummary::default(),
            );
            assert_eq!(action, None);
        });
    }

    // ---- The run in flight ------------------------------------------------

    /// A one-feature collection, enough to drive a real run.
    fn one_feature() -> std::sync::Arc<oxigeo::geojson::types::FeatureCollection> {
        let point = oxigeo::geojson::types::Point::new_2d(1.0, 2.0).expect("valid point");
        std::sync::Arc::new(oxigeo::geojson::types::FeatureCollection::new(vec![
            oxigeo::geojson::types::Feature::new(
                Some(oxigeo::geojson::types::Geometry::Point(point)),
                None,
            ),
        ]))
    }

    /// A started `feature_count` run over one feature.
    fn started_run() -> ToolRun {
        crate::processing_exec::start_builtin_run(
            "feature_count",
            one_feature(),
            &oxigis_core::ToolContext::new(),
        )
        .expect("feature_count is wired")
    }

    fn synthetic_descriptor() -> ToolDescriptor {
        ToolDescriptor {
            id: "feature_count".to_string(),
            title: "Feature Count".to_string(),
            description: String::new(),
            params: vec![layer_spec()],
        }
    }

    fn job() -> ProcessingJob {
        ProcessingJob::new(
            synthetic_descriptor(),
            None,
            OutputTarget::default(),
            started_run(),
        )
    }

    /// Polls until the run lands, exactly as the frame loop does — with a
    /// wait, because on native the work is on a worker thread that has to be
    /// scheduled at least once.
    fn poll_to_completion(state: &mut ProcessingPanelState) -> ProcessingPoll {
        for iteration in 0..100_000 {
            match state.poll_job(1.0 / 60.0) {
                ProcessingPoll::Running => {
                    if iteration < 100 {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    }
                }
                landed => return landed,
            }
        }
        panic!("the job never finished");
    }

    #[test]
    fn a_job_runs_to_a_result_and_then_reports_idle() {
        let mut state = ProcessingPanelState::default();
        assert!(matches!(state.poll_job(0.016), ProcessingPoll::Idle));

        state.begin_job(job());
        assert!(state.is_running());
        let landed = poll_to_completion(&mut state);
        let ProcessingPoll::Finished(finished) = landed else {
            panic!("expected a result, got {landed:?}");
        };
        assert_eq!(finished.result.expect("ok"), serde_json::json!(1));
        assert!(!state.is_running(), "the job must be cleared once it lands");
        assert!(
            matches!(state.poll_job(0.016), ProcessingPoll::Idle),
            "a finished job must not be reported twice"
        );
    }

    #[test]
    fn cancelling_a_job_clears_it_and_names_the_tool() {
        let mut state = ProcessingPanelState::default();
        state.begin_job(job());
        assert_eq!(state.cancel_job().as_deref(), Some("Feature Count"));
        assert!(!state.is_running());
        assert!(
            matches!(state.poll_job(0.016), ProcessingPoll::Idle),
            "a cancelled run must never deliver a result afterwards"
        );
        assert_eq!(state.cancel_job(), None, "there is nothing left to cancel");
    }

    #[test]
    fn beginning_a_job_replaces_the_one_in_flight_and_clears_the_last_result() {
        let mut state = ProcessingPanelState::default();
        state.set_result("stale".to_string());
        state.begin_job(job());
        state.begin_job(job());
        assert!(state.is_running(), "exactly one job, and it is the new one");
        let report = format!("{state:?}");
        assert!(
            report.contains("last_result: None") && report.contains("last_error: None"),
            "a fresh run must clear the previous run's report: {report}"
        );
        let landed = poll_to_completion(&mut state);
        assert!(matches!(landed, ProcessingPoll::Finished(_)));
    }

    #[test]
    fn the_slice_budget_tracks_the_frame_time_within_its_clamps() {
        let mut state = ProcessingPanelState {
            slice: INITIAL_SLICE,
            ..ProcessingPanelState::default()
        };
        state.tune_slice(1.0 / 120.0);
        assert_eq!(state.slice, INITIAL_SLICE * 2, "headroom must be spent");
        state.tune_slice(1.0 / 10.0);
        assert_eq!(state.slice, INITIAL_SLICE, "an overrun must be given back");

        for _ in 0..64 {
            state.tune_slice(1.0);
        }
        assert_eq!(state.slice, MIN_SLICE, "shrinking must bottom out");
        for _ in 0..64 {
            state.tune_slice(1.0 / 1000.0);
        }
        assert_eq!(state.slice, MAX_SLICE, "growing must top out");

        let before = state.slice;
        state.tune_slice(f32::NAN);
        state.tune_slice(0.0);
        state.tune_slice(-1.0);
        assert_eq!(
            state.slice, before,
            "a frame egui has not measured yet says nothing about the budget"
        );
    }

    #[test]
    fn a_zero_slice_is_never_handed_to_a_run() {
        // The pass makes no progress on a budget of zero, so a state that
        // somehow held one would poll forever without advancing.
        let mut state = ProcessingPanelState {
            slice: 0,
            ..ProcessingPanelState::default()
        };
        state.tune_slice(1.0);
        assert!(state.slice >= MIN_SLICE);
    }

    // ---- Selected features only ------------------------------------------

    #[test]
    fn a_selection_covers_only_the_layer_it_addresses() {
        let layer = LayerId::new();
        let other = LayerId::new();
        let selection = SelectionSummary {
            layer: Some(layer),
            count: 3,
        };
        assert!(selection.covers(Some(layer)));
        assert!(!selection.covers(Some(other)));
        assert!(!selection.covers(None));
        assert!(
            !SelectionSummary {
                layer: Some(layer),
                count: 0
            }
            .covers(Some(layer)),
            "an empty selection has nothing to restrict to"
        );
        assert!(!SelectionSummary::default().covers(Some(layer)));
    }

    #[test]
    fn the_selection_row_refuses_to_restrict_a_layer_the_selection_is_not_on() {
        let layer = LayerId::new();
        let mut state = ProcessingPanelState {
            selected_only: true,
            ..ProcessingPanelState::default()
        };
        egui::__run_test_ui(|ui| {
            let elsewhere = SelectionSummary {
                layer: Some(LayerId::new()),
                count: 3,
            };
            assert!(
                !draw_selection_row(ui, &mut state, elsewhere, Some(layer)),
                "a selection on another layer names indices that mean something \
                 else here — it must never filter this run"
            );
            assert!(
                state.selected_only(),
                "the user's own preference must survive an unoffered frame"
            );
            let here = SelectionSummary {
                layer: Some(layer),
                count: 3,
            };
            assert!(draw_selection_row(ui, &mut state, here, Some(layer)));
            // No layer picked yet: there is nothing for the selection to
            // cover, whatever it holds.
            assert!(!draw_selection_row(ui, &mut state, here, None));
        });
    }

    #[test]
    fn drawing_reports_progress_and_a_cancel_button_while_a_job_runs() {
        let registry = oxigis_core::builtin_registry();
        let mut state = ProcessingPanelState::default();
        state.begin_job(job());
        let id = LayerId::new();
        let options = [(id, "cities")];
        egui::__run_test_ui(|ui| {
            // The Run button is disabled and the progress bar drawn; nothing
            // here clicks, so no action is reported — the claim under test is
            // that a running job draws at all.
            let action = draw(
                ui,
                &registry,
                &mut state,
                &options,
                Some(id),
                SelectionSummary {
                    layer: Some(id),
                    count: 2,
                },
            );
            assert_eq!(action, None);
        });
        assert!(state.is_running(), "drawing must not disturb the run");
    }

    // ---- Output destination ------------------------------------------------

    #[test]
    fn a_queued_file_save_is_handed_over_exactly_once() {
        let mut state = ProcessingPanelState::default();
        assert_eq!(state.take_file_save(), None);
        let request = ProcessingFileRequest {
            name: "centroids".to_string(),
            content: "{}".to_string(),
            features: 3,
        };
        state.queue_file_save(request.clone());
        assert_eq!(state.take_file_save(), Some(request));
        assert_eq!(
            state.take_file_save(),
            None,
            "a taken request must not be handed out twice"
        );
    }

    #[test]
    fn the_output_destination_defaults_to_a_temporary_layer() {
        assert_eq!(OutputDestination::default(), OutputDestination::Layer);
        assert_eq!(
            OutputTarget::default().destination,
            OutputDestination::Layer
        );
        assert!(OutputTarget::default().name.is_empty());
    }
}
