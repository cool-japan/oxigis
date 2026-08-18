// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The resumable half of every built-in tool: one per-feature accumulator
//! ([`FeatureSink`]) plus a cursor over the layer's features ([`ToolPass`]).
//!
//! Every built-in tool is a fold over `FeatureCollection::features` — that is
//! not a coincidence, it is what makes a Processing run interruptible at all.
//! Hoisting the loop body into a [`FeatureSink`] and the loop itself into
//! [`ToolPass`] buys three things a plain `for` loop inside
//! [`oxigis_core::ToolExecutor::run`] could not:
//!
//! * **progress** — [`ToolPass::progress`] is exact, because the cursor *is*
//!   the progress;
//! * **cancellation** — a caller that stops calling [`ToolPass::advance`] has
//!   cancelled the run, with no flag to check inside the geometry code;
//! * **a browser that still repaints** — `wasm32` has no worker thread to move
//!   the work to (see [`super::runner`]), so the frame loop drives one bounded
//!   slice per frame instead.
//!
//! The one-shot path has not gone anywhere: [`drain`] runs a pass to
//! completion in a single call, and every `ToolExecutor::run` in this module
//! tree is exactly that call. So the sliced driver and the synchronous one
//! execute the *same* accumulator over the *same* features and cannot drift
//! apart — the property that lets the existing tool tests keep asserting
//! against `run` while the app drives slices.

use std::sync::Arc;

use oxigeo::geojson::types::{Feature, FeatureCollection};
use oxigis_core::CoreError;

/// One tool's per-feature accumulator: the body of its loop, and the value it
/// answers with once every feature has been through it.
///
/// `Send` because [`super::runner`] moves a boxed sink onto a worker thread on
/// native builds; every implementation is plain owned data, so this costs
/// nothing.
pub(super) trait FeatureSink: Send {
    /// Folds one input feature in. Called exactly once per feature, in input
    /// order, and never after [`Self::finish`].
    fn absorb(&mut self, feature: &Feature);

    /// Produces the run's result value.
    ///
    /// Takes `Box<Self>` rather than `self` so the trait stays object-safe
    /// while the accumulator is still consumed by value — a sink that has
    /// built a `Vec<Feature>` must move it into the result, not clone it.
    ///
    /// # Errors
    ///
    /// Returns whatever the tool refuses on: an input that produced nothing
    /// usable ([`super::invalid_layer`]), or a result that will not serialize.
    fn finish(self: Box<Self>) -> Result<serde_json::Value, CoreError>;
}

/// How far a run has got — exact, because the pass's cursor *is* the count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ToolProgress {
    /// Features already folded into the accumulator.
    pub processed: usize,
    /// Features the run was handed. Zero for an empty layer, which is a
    /// legitimate (instantly complete) run, not an error.
    pub total: usize,
}

impl ToolProgress {
    /// The fraction done, in `0.0..=1.0`, for a progress bar.
    ///
    /// An empty run reports `1.0` rather than `0/0`: there is nothing left to
    /// do, and a bar that sits empty while the run is already over reads as a
    /// hang.
    #[must_use]
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            return 1.0;
        }
        // `as` on a `usize` this size is exact enough for a 0..1 ratio, and
        // the ratio is clamped by construction (`processed <= total`).
        (self.processed as f32 / self.total as f32).clamp(0.0, 1.0)
    }

    /// A short `"1 234 / 10 000 features"`-shaped label for the progress row.
    #[must_use]
    pub fn label(&self) -> String {
        format!("{} / {} features", self.processed, self.total)
    }
}

/// A tool run in progress: the features it reads, how far it has got, and the
/// accumulator it is folding them into.
///
/// `Send` (every field is), which is what lets [`super::runner`] hand a whole
/// pass to a worker thread on native builds.
pub(super) struct ToolPass {
    /// The features this run reads — shared with the app-side store, never
    /// cloned, so a run over a 200 MB layer costs one pointer.
    features: Arc<FeatureCollection>,
    /// How many of them have been folded in so far.
    cursor: usize,
    /// The accumulator, taken by [`Self::advance`] when the run finishes so a
    /// finished pass can answer at most once.
    sink: Option<Box<dyn FeatureSink>>,
}

impl ToolPass {
    /// A pass that has not started yet.
    pub(super) fn new(features: Arc<FeatureCollection>, sink: Box<dyn FeatureSink>) -> Self {
        Self {
            features,
            cursor: 0,
            sink: Some(sink),
        }
    }

    /// How far this run has got.
    pub(super) fn progress(&self) -> ToolProgress {
        ToolProgress {
            processed: self.cursor,
            total: self.features.features.len(),
        }
    }

    /// Folds at most `budget` more features in, returning the run's result
    /// once the last one has gone through — and [`None`] while there is more
    /// to do, or once the result has already been taken.
    ///
    /// A `budget` of zero still *finishes* a pass whose cursor has already
    /// reached the end (an empty layer completes on the first call whatever
    /// the budget), but never advances one that has not: a driver must pass
    /// at least one to make progress. See [`super::runner`] for the two
    /// drivers and the budgets they choose.
    pub(super) fn advance(
        &mut self,
        budget: usize,
    ) -> Option<Result<serde_json::Value, CoreError>> {
        let total = self.features.features.len();
        let sink = self.sink.as_mut()?;
        let end = self.cursor.saturating_add(budget).min(total);
        for feature in self.features.features.get(self.cursor..end).unwrap_or(&[]) {
            sink.absorb(feature);
        }
        self.cursor = end;
        if self.cursor < total {
            return None;
        }
        Some(self.sink.take()?.finish())
    }
}

/// Runs a whole pass in one call — the synchronous path every
/// [`oxigis_core::ToolExecutor::run`] in this module tree takes.
///
/// # Errors
///
/// Returns whatever the sink refuses on; see [`FeatureSink::finish`].
pub(super) fn drain(
    features: Arc<FeatureCollection>,
    sink: Box<dyn FeatureSink>,
) -> Result<serde_json::Value, CoreError> {
    let mut pass = ToolPass::new(features, sink);
    // Terminates on the first iteration — an unbounded budget always reaches
    // the end of the collection — and is written as a loop rather than a
    // single call plus an "unreachable" error so there is no arm to get wrong.
    loop {
        if let Some(result) = pass.advance(usize::MAX) {
            return result;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing_exec::fixtures::point_feature;

    /// A sink that records the ids it saw, so a test can prove the slicing
    /// visited every feature exactly once and in order.
    #[derive(Default)]
    struct CountingSink {
        /// One entry per absorbed feature, holding its longitude.
        seen: Vec<f64>,
    }

    impl FeatureSink for CountingSink {
        fn absorb(&mut self, feature: &Feature) {
            let lon = match feature.geometry.as_ref() {
                Some(oxigeo::geojson::types::Geometry::Point(point)) => point.coordinates[0],
                _ => f64::NAN,
            };
            self.seen.push(lon);
        }

        fn finish(self: Box<Self>) -> Result<serde_json::Value, CoreError> {
            Ok(serde_json::json!(self.seen))
        }
    }

    fn collection(count: usize) -> Arc<FeatureCollection> {
        let features = (0..count)
            .map(|index| point_feature(index as f64, 0.0))
            .collect();
        Arc::new(FeatureCollection::new(features))
    }

    #[test]
    fn a_sliced_pass_visits_every_feature_exactly_once_in_order() {
        let mut pass = ToolPass::new(collection(10), Box::new(CountingSink::default()));
        let mut result = None;
        let mut slices = 0;
        while result.is_none() {
            result = pass.advance(3);
            slices += 1;
            assert!(slices < 100, "the pass must terminate");
        }
        assert_eq!(slices, 4, "10 features in slices of 3 is four calls");
        let value = result
            .expect("the loop only exits with a result")
            .expect("ok");
        let seen: Vec<f64> = serde_json::from_value(value).expect("an array of longitudes");
        assert_eq!(seen, (0..10).map(f64::from).collect::<Vec<_>>());
    }

    #[test]
    fn progress_tracks_the_cursor_and_ends_full() {
        let mut pass = ToolPass::new(collection(4), Box::new(CountingSink::default()));
        assert_eq!(
            pass.progress(),
            ToolProgress {
                processed: 0,
                total: 4
            }
        );
        assert!(pass.advance(1).is_none());
        assert_eq!(pass.progress().processed, 1);
        assert!((pass.progress().fraction() - 0.25).abs() < f32::EPSILON);
        assert!(pass.advance(usize::MAX).is_some());
        assert_eq!(pass.progress().processed, 4);
        assert!((pass.progress().fraction() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn an_empty_pass_finishes_on_the_first_call_whatever_the_budget() {
        let mut pass = ToolPass::new(collection(0), Box::new(CountingSink::default()));
        assert_eq!(pass.progress().fraction(), 1.0, "nothing left to do");
        let result = pass
            .advance(0)
            .expect("an empty run is complete, not stalled");
        assert_eq!(result.expect("ok"), serde_json::json!([]));
        assert!(
            pass.advance(usize::MAX).is_none(),
            "a finished pass answers at most once"
        );
    }

    #[test]
    fn a_zero_budget_never_advances_an_unfinished_pass() {
        let mut pass = ToolPass::new(collection(3), Box::new(CountingSink::default()));
        assert!(pass.advance(0).is_none());
        assert_eq!(
            pass.progress().processed,
            0,
            "a driver that passes zero makes no progress — it must pass at least one"
        );
    }

    #[test]
    fn draining_matches_slicing() {
        let sliced = {
            let mut pass = ToolPass::new(collection(7), Box::new(CountingSink::default()));
            loop {
                if let Some(result) = pass.advance(2) {
                    break result.expect("ok");
                }
            }
        };
        let drained = drain(collection(7), Box::new(CountingSink::default())).expect("ok");
        assert_eq!(sliced, drained);
    }

    #[test]
    fn progress_label_reads_as_a_count() {
        let progress = ToolProgress {
            processed: 3,
            total: 12,
        };
        assert_eq!(progress.label(), "3 / 12 features");
    }
}
