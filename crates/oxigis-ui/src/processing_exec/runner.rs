// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Driving a [`super::job::ToolPass`] without freezing the window.
//!
//! A Processing run used to happen inside the egui frame that clicked Run:
//! `simplify` over a national dataset is seconds of Douglas-Peucker on the
//! frame thread, during which nothing repaints, no progress shows and there is
//! nothing to press to stop it — and on the browser build, no yielding to the
//! event loop at all, which is what puts the "page unresponsive" dialog on
//! screen.
//!
//! [`ToolRun`] is the fix, in the only two shapes the two builds allow:
//!
//! * **native** — the pass moves onto a `std::thread`. The frame loop polls a
//!   [`AtomicUsize`] counter for progress and an [`AtomicBool`] is the cancel
//!   token the worker checks between slices. Nothing is joined: a cancelled
//!   run's thread notices, drops its `Arc`s and exits on its own.
//! * **`wasm32`** — there is no thread to move it to (`std::thread::spawn`
//!   does not work on `wasm32-unknown-unknown`, and this crate has no worker
//!   plumbing), so the frame loop drives one bounded slice per frame and the
//!   browser repaints in between. Cancelling is simply not calling
//!   [`ToolRun::poll`] again.
//!
//! Both drivers consume the same [`super::job::FeatureSink`], so the answer
//! does not depend on which one ran.

use std::sync::Arc;

use oxigeo::geojson::types::FeatureCollection;

use super::job::{FeatureSink, ToolPass, ToolProgress};

/// How many features the native worker folds in between two checks of its
/// cancel flag.
///
/// Small enough that Cancel feels immediate (a few hundred features is well
/// under a millisecond for every built-in tool) and large enough that the
/// atomic load is lost in the noise.
#[cfg(not(target_arch = "wasm32"))]
const WORKER_SLICE: usize = 256;

/// What [`ToolRun::poll`] found.
#[derive(Debug)]
pub enum ToolRunState {
    /// Still working; poll again next frame.
    Running,
    /// Finished, with the tool's result or its refusal.
    Finished(Result<serde_json::Value, String>),
    /// [`ToolRun::cancel`] was called; no result will ever arrive.
    Cancelled,
}

/// A Processing run in flight.
///
/// Held by the Processing panel across frames and polled by the app's frame
/// loop; see the module docs for the two ways it is driven.
pub struct ToolRun {
    /// The build-specific driver.
    driver: Driver,
    /// How many features the run was handed — fixed for its whole life, so
    /// progress has a denominator even before the first slice.
    total: usize,
    /// Whether [`Self::cancel`] has been called. Kept here as well as in the
    /// worker's flag so [`Self::poll`] answers deterministically without
    /// waiting for a thread to notice.
    cancelled: bool,
}

impl std::fmt::Debug for ToolRun {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Box<dyn FeatureSink>` is not `Debug` and should not be — a sink can
        // hold a whole result collection. The progress pair is what a reader
        // of a panel dump actually needs.
        formatter
            .debug_struct("ToolRun")
            .field("progress", &self.progress())
            .field("cancelled", &self.cancelled)
            .finish()
    }
}

/// The build-specific half of a [`ToolRun`].
enum Driver {
    /// A worker thread owns the pass; this side only observes it.
    #[cfg(not(target_arch = "wasm32"))]
    Worker(Worker),
    /// The frame loop owns the pass and advances it a slice at a time.
    Sliced(SlicedDriver),
}

/// The `wasm32` (and spawn-failure) driver: the pass lives here and the frame
/// loop advances it.
struct SlicedDriver {
    /// The pass, or [`None`] once it has produced its result — or once
    /// [`ToolRun::cancel`] dropped it.
    pass: Option<ToolPass>,
    /// An outcome produced before any poll: the spawn-failure path's refusal.
    /// Handed out by the first [`Self::advance`] and never refilled, so a run
    /// that could not start reports once instead of polling forever.
    ready: Option<Result<serde_json::Value, String>>,
    /// Features folded in so far — the sliced twin of the worker's atomic.
    processed: usize,
}

impl SlicedDriver {
    /// A driver that has not started yet.
    fn new(pass: ToolPass) -> Self {
        Self {
            pass: Some(pass),
            ready: None,
            processed: 0,
        }
    }

    /// A driver that has nothing to run and answers `reason` on its first
    /// poll.
    #[cfg(not(target_arch = "wasm32"))]
    fn refused(reason: String) -> Self {
        Self {
            pass: None,
            ready: Some(Err(reason)),
            processed: 0,
        }
    }

    /// Folds at most `slice` more features in.
    fn advance(&mut self, slice: usize) -> Option<Result<serde_json::Value, String>> {
        if let Some(ready) = self.ready.take() {
            return Some(ready);
        }
        let pass = self.pass.as_mut()?;
        let outcome = pass.advance(slice.max(1));
        self.processed = pass.progress().processed;
        if outcome.is_some() {
            self.pass = None;
        }
        outcome.map(|result| result.map_err(|error| error.to_string()))
    }
}

/// The native driver: a detached worker thread plus the two atomics and the
/// slot it reports through.
#[cfg(not(target_arch = "wasm32"))]
struct Worker {
    /// Set by [`ToolRun::cancel`]; the worker checks it between slices.
    cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Features the worker has folded in, updated once per slice.
    processed: Arc<std::sync::atomic::AtomicUsize>,
    /// Where the worker leaves its result. `parking_lot::Mutex` has no
    /// poisoning, so no lock site here needs an `unwrap` (COOLJAPAN Policy
    /// \#3).
    outcome: Arc<parking_lot::Mutex<Option<Result<serde_json::Value, String>>>>,
}

impl ToolRun {
    /// Starts a run of `sink` over `features`.
    ///
    /// Never blocks: on native the pass is handed to a worker thread and this
    /// returns immediately; on `wasm32` nothing has run yet at all — the first
    /// [`Self::poll`] does the first slice.
    pub(super) fn start(features: Arc<FeatureCollection>, sink: Box<dyn FeatureSink>) -> Self {
        let pass = ToolPass::new(features, sink);
        let total = pass.progress().total;
        Self {
            driver: spawn_driver(pass),
            total,
            cancelled: false,
        }
    }

    /// A run driven from the **frame loop**, whichever build this is.
    ///
    /// The shape `wasm32` always gets, constructed explicitly so a native
    /// test run can exercise the browser's driver too: without this, native
    /// tests would only ever reach [`Driver::Worker`] and the sliced path
    /// would ship covered by nothing this workspace can execute.
    #[cfg(test)]
    pub(super) fn sliced(features: Arc<FeatureCollection>, sink: Box<dyn FeatureSink>) -> Self {
        let pass = ToolPass::new(features, sink);
        let total = pass.progress().total;
        Self {
            driver: Driver::Sliced(SlicedDriver::new(pass)),
            total,
            cancelled: false,
        }
    }

    /// Advances or observes the run.
    ///
    /// `slice` is how many features this call may fold in on a build that
    /// drives the pass from the frame loop; a native run ignores it (its
    /// worker uses its own fixed slice). Callers should scale it to the
    /// frame budget they have — see
    /// [`crate::processing_panel::ProcessingPanelState::poll_job`], which
    /// grows and shrinks it from the observed frame time.
    pub fn poll(&mut self, slice: usize) -> ToolRunState {
        if self.cancelled {
            return ToolRunState::Cancelled;
        }
        match &mut self.driver {
            #[cfg(not(target_arch = "wasm32"))]
            Driver::Worker(worker) => match worker.outcome.lock().take() {
                Some(result) => ToolRunState::Finished(result),
                None => ToolRunState::Running,
            },
            Driver::Sliced(driver) => match driver.advance(slice) {
                Some(result) => ToolRunState::Finished(result),
                None => ToolRunState::Running,
            },
        }
    }

    /// How far the run has got.
    #[must_use]
    pub fn progress(&self) -> ToolProgress {
        let processed = match &self.driver {
            #[cfg(not(target_arch = "wasm32"))]
            Driver::Worker(worker) => worker
                .processed
                .load(std::sync::atomic::Ordering::Relaxed)
                .min(self.total),
            Driver::Sliced(driver) => driver.processed,
        };
        ToolProgress {
            processed,
            total: self.total,
        }
    }

    /// Stops the run.
    ///
    /// Returns immediately in both builds: on native the worker is *told* to
    /// stop rather than waited for — it checks the flag between slices, then
    /// drops the features it was holding and exits. Nothing else observes its
    /// result, so there is nothing to join for, and blocking the frame thread
    /// to wait for a cancel would be the very freeze this type exists to
    /// remove.
    pub fn cancel(&mut self) {
        self.cancelled = true;
        match &mut self.driver {
            #[cfg(not(target_arch = "wasm32"))]
            Driver::Worker(worker) => worker
                .cancel
                .store(true, std::sync::atomic::Ordering::Relaxed),
            // Dropping the pass is the cancel: nothing else drives it.
            Driver::Sliced(driver) => {
                driver.pass = None;
                driver.ready = None;
            }
        }
    }

    /// Whether [`Self::cancel`] has been called.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

/// Starts the build's driver for `pass`.
///
/// On native this spawns the worker. A spawn the OS refuses (a thread limit,
/// or a sandbox with none) must not lose the run: the pass travels to the
/// worker inside a shared slot, so a failed spawn — whose closure is dropped
/// without ever running — leaves it recoverable, and the run falls back to
/// being driven from the frame loop exactly as the browser build drives it.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_driver(pass: ToolPass) -> Driver {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    let cancel = Arc::new(AtomicBool::new(false));
    let processed = Arc::new(AtomicUsize::new(0));
    let outcome = Arc::new(parking_lot::Mutex::new(None));
    let worker = Worker {
        cancel: Arc::clone(&cancel),
        processed: Arc::clone(&processed),
        outcome: Arc::clone(&outcome),
    };
    // The slot is what makes a failed spawn recoverable: `Builder::spawn`
    // consumes the closure, and the closure owns whatever it captured, so a
    // pass captured *by value* would be gone. Captured through a slot, it is
    // still here to hand to the fallback driver.
    let slot = Arc::new(parking_lot::Mutex::new(Some(pass)));
    let worker_slot = Arc::clone(&slot);
    let spawned = std::thread::Builder::new()
        .name("oxigis-processing".to_string())
        .spawn(move || {
            let Some(mut pass) = worker_slot.lock().take() else {
                return;
            };
            loop {
                if cancel.load(Ordering::Relaxed) {
                    // The user cancelled: drop everything without reporting.
                    return;
                }
                match pass.advance(WORKER_SLICE) {
                    Some(result) => {
                        processed.store(pass.progress().processed, Ordering::Relaxed);
                        *outcome.lock() = Some(result.map_err(|error| error.to_string()));
                        return;
                    }
                    None => processed.store(pass.progress().processed, Ordering::Relaxed),
                }
            }
        });
    match spawned {
        Ok(_handle) => Driver::Worker(worker),
        Err(error) => {
            tracing::warn!(
                %error,
                "oxigis-ui: could not start the Processing worker thread; \
                 running the tool from the frame loop instead",
            );
            match slot.lock().take() {
                Some(pass) => Driver::Sliced(SlicedDriver::new(pass)),
                // Unreachable: a failed spawn never ran the closure, so the
                // slot is still full. Reported rather than assumed, because
                // the alternative shape — a driver with nothing to drive —
                // would poll as "Running" forever.
                None => Driver::Sliced(SlicedDriver::refused(format!(
                    "the Processing worker thread could not be started: {error}"
                ))),
            }
        }
    }
}

/// The `wasm32` driver: no thread exists, so the frame loop owns the pass.
#[cfg(target_arch = "wasm32")]
fn spawn_driver(pass: ToolPass) -> Driver {
    Driver::Sliced(SlicedDriver::new(pass))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing_exec::fixtures::point_feature;
    use crate::processing_exec::{builtin_executor, start_builtin_run};
    use oxigis_core::ToolContext;

    fn collection(count: usize) -> Arc<FeatureCollection> {
        let features = (0..count)
            .map(|index| point_feature(index as f64, 0.0))
            .collect();
        Arc::new(FeatureCollection::new(features))
    }

    /// Polls `run` until it finishes, with a bounded number of frames so a
    /// driver that never completes fails the test instead of hanging it.
    fn drive(run: &mut ToolRun, slice: usize) -> Result<serde_json::Value, String> {
        for iteration in 0..100_000 {
            match run.poll(slice) {
                ToolRunState::Finished(result) => return result,
                ToolRunState::Cancelled => return Err("cancelled".to_string()),
                ToolRunState::Running => wait_a_moment(iteration),
            }
        }
        panic!("the run never finished");
    }

    /// Gives a native worker thread the CPU between polls.
    ///
    /// A plain spin here made these tests flaky: nextest runs test binaries in
    /// parallel, and on a loaded machine a hundred thousand un-yielded polls
    /// can all happen before the worker is scheduled even once. The frame
    /// loop this stands in for repaints at 60 Hz, so waiting is the honest
    /// simulation; `wasm32` has no worker to wait for at all (its driver
    /// advances inside `poll` itself) and no `std::thread` to wait with.
    fn wait_a_moment(iteration: usize) {
        #[cfg(not(target_arch = "wasm32"))]
        if iteration < 100 {
            std::thread::yield_now();
        } else {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        #[cfg(target_arch = "wasm32")]
        let _ = iteration;
    }

    #[test]
    fn a_sliced_run_produces_the_same_value_the_synchronous_executor_does() {
        let features = collection(50);
        let synchronous = builtin_executor("feature_count", Arc::clone(&features))
            .expect("wired")
            .run(&ToolContext::new())
            .expect("ok");
        let mut run =
            start_builtin_run("feature_count", features, &ToolContext::new()).expect("wired");
        assert_eq!(drive(&mut run, 7).expect("ok"), synchronous);
    }

    #[test]
    fn progress_reaches_the_total_and_never_exceeds_it() {
        let mut run =
            start_builtin_run("feature_count", collection(30), &ToolContext::new()).expect("wired");
        assert_eq!(run.progress().total, 30);
        let _value = drive(&mut run, 4).expect("ok");
        let progress = run.progress();
        assert_eq!(progress.processed, 30);
        assert!(progress.processed <= progress.total);
    }

    #[test]
    fn a_cancelled_run_reports_cancelled_and_never_a_result() {
        let mut run = start_builtin_run("feature_count", collection(10_000), &ToolContext::new())
            .expect("wired");
        run.cancel();
        assert!(run.is_cancelled());
        for _ in 0..10 {
            match run.poll(1) {
                ToolRunState::Cancelled => {}
                other => panic!("a cancelled run must stay cancelled, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_empty_layer_finishes_on_the_first_poll() {
        let mut run =
            start_builtin_run("feature_count", collection(0), &ToolContext::new()).expect("wired");
        assert_eq!(drive(&mut run, 1).expect("ok"), serde_json::json!(0));
    }

    /// A `feature_count` run over `count` features, driven the way the
    /// browser build drives every run.
    fn sliced_count_run(count: usize) -> ToolRun {
        let sink = super::super::builtin_sink("feature_count", &ToolContext::new())
            .expect("feature_count is wired")
            .expect("feature_count refuses no parameter");
        ToolRun::sliced(collection(count), sink)
    }

    #[test]
    fn the_frame_loop_driver_advances_exactly_one_slice_per_poll() {
        // The `wasm32` contract, asserted on a build that can actually run
        // it: each poll folds in at most `slice` features and nothing more,
        // which is what leaves the browser free to repaint in between.
        let mut run = sliced_count_run(5);
        for expected in 1..5 {
            assert!(matches!(run.poll(1), ToolRunState::Running));
            assert_eq!(
                run.progress().processed,
                expected,
                "one feature per poll, no more"
            );
        }
        let ToolRunState::Finished(result) = run.poll(1) else {
            panic!("the fifth slice must complete the run");
        };
        assert_eq!(result.expect("ok"), serde_json::json!(5));
        assert_eq!(run.progress().processed, 5);
    }

    #[test]
    fn cancelling_a_frame_loop_run_drops_its_work_immediately() {
        let mut run = sliced_count_run(1_000);
        assert!(matches!(run.poll(1), ToolRunState::Running));
        run.cancel();
        for _ in 0..4 {
            assert!(matches!(run.poll(1), ToolRunState::Cancelled));
        }
        assert!(
            run.progress().processed < 1_000,
            "a cancelled run must stop where it was, not run on"
        );
    }

    #[test]
    fn a_frame_loop_run_matches_the_synchronous_executor() {
        let features = collection(37);
        let synchronous = builtin_executor("feature_count", Arc::clone(&features))
            .expect("wired")
            .run(&ToolContext::new())
            .expect("ok");
        let sink = super::super::builtin_sink("feature_count", &ToolContext::new())
            .expect("wired")
            .expect("no parameter to refuse");
        let mut run = ToolRun::sliced(features, sink);
        assert_eq!(drive(&mut run, 3).expect("ok"), synchronous);
    }

    #[test]
    fn a_refusing_tool_reports_its_reason_through_the_run() {
        // `bounds` over geometry-less features refuses; the refusal must reach
        // the caller as the run's outcome, not as a panic or a silent stall.
        let features = Arc::new(FeatureCollection::new(vec![
            oxigeo::geojson::types::Feature::new(None, None),
        ]));
        let mut run = start_builtin_run("bounds", features, &ToolContext::new()).expect("wired");
        let error = drive(&mut run, 1).expect_err("must refuse");
        assert!(error.contains("no geometry to bound"), "{error}");
    }
}
