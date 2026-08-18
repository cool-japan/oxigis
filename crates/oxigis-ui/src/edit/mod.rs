// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Feature editing: the mode machine, the reversible command model and the
//! undo stack.
//!
//! # Shape of the system
//!
//! * [`command`] holds the data model — [`command::FeatureOp`],
//!   [`command::EditTransaction`], the pure [`command::apply_ops`] and the
//!   geometry mutators. Nothing there knows about egui, the GPU, or
//!   [`crate::OxigisApp`]; it is total, all-or-nothing and directly testable.
//! * [`stack`] holds [`stack::EditStack`], a strict-LIFO log of those
//!   transactions with an entry cap, a byte budget, gesture-boundary
//!   coalescing and per-layer pruning.
//! * This module holds the per-frame state: which tool is active, what is
//!   selected, what is being sketched or dragged.
//!
//! # The two invariants worth stating up front
//!
//! **There is exactly one layer-selection concept.** The edit target is
//! [`crate::OxigisApp::selection`] — the same value the style panel and the
//! attribute table read — so the map, the table and the style editor can never
//! disagree about which layer they mean. [`EditSelection`] therefore carries no
//! [`LayerId`]: it addresses a feature *within* the selected layer.
//!
//! **All feature data changes through one function.** Every command, undo and
//! redo funnels through `OxigisApp::apply_feature_collection`, which serializes
//! first, rewrites the layer's source to
//! [`oxigis_core::VectorSource::InlineGeoJson`] and then replaces the shared
//! collection and the GPU copy together. Nothing in this module mutates a
//! layer's data directly.
//!
//! # What undo covers, and what it does not
//!
//! In: geometry edits, property edits, feature add, feature delete. Out (v1):
//! layer add/remove/reorder/rename, style edits, visibility and opacity, the
//! basemap. Those live in [`oxigis_core::Project`] rather than in the feature
//! data, and would need a second command family and a second choke point —
//! style edits already have their own coalesced
//! [`crate::local_input::LocalLayerOp::SetStyle`] queue.

pub mod clipboard;
pub mod command;
pub mod form;
pub mod hit;
pub mod overlay;
pub mod project_op;
pub mod selection;
pub mod sketch;
pub mod snap;
pub mod stack;
pub mod toolbar;
pub mod topology;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_bulk;
#[cfg(test)]
mod tests_form;
#[cfg(test)]
mod tests_hit;
#[cfg(test)]
mod tests_marks;
#[cfg(test)]
mod tests_sketch;
#[cfg(test)]
mod tests_snap;
#[cfg(test)]
mod tests_topology;
#[cfg(test)]
mod tests_translate;

use crate::local_input::{self, LocalInputState};
use crate::map_view::PanGate;
use command::{EditError, EditTransaction, FeatureOp};
use form::FormBuffer;
use hit::{HitTarget, PickCycle, WorldBboxIndex};
use oxigeo::geojson::types::{FeatureCollection, Geometry};
use oxigis_core::{LayerId, Project};
use oxigis_render::{LonLat, MapView, WorldCoord};
use snap::{SnapIndex, SnapResult, SnapSettings};
use std::collections::BTreeMap;
use std::sync::Arc;
use topology::FeatureIssue;

pub use sketch::Sketch;

/// Most notices kept in [`EditState`] at once; older ones are dropped.
///
/// A cap rather than a growing log: the notice list is a UI surface, not an
/// audit trail, and an unbounded one on a long editing session is a slow leak.
pub const MAX_NOTICES: usize = 200;

/// One short, user-facing message the edit system wants shown.
///
/// A newtype over [`String`] rather than an enum of cases: every notice this
/// system produces is a sentence assembled from live data (a layer name, a file
/// name, a count), so an enum would carry the same `String` in every variant and
/// buy nothing. The type exists so a notice cannot be confused with an error, a
/// label or a status line at a call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditNotice(String);

impl EditNotice {
    /// Wraps a message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// The message text.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Display for EditNotice {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// What the map's primary pointer button does this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EditMode {
    /// Editing off: clicks fall through unconsumed, nothing is painted over the
    /// map, and the map behaves bit-identically to a build without this module.
    #[default]
    Off,
    /// Click picks a feature (repeat clicks cycle). The picked feature shows
    /// vertex handles and midpoint ghosts immediately: drag a handle to move,
    /// drag a ghost to insert, `Delete` to remove. There is no sub-mode — the
    /// pan gate disambiguates a handle drag from a camera drag from the real
    /// hit test, so a second mode would be pure friction.
    Select,
    /// Each click commits one `Point` feature.
    DrawPoint,
    /// Clicks append vertices to a `LineString` sketch.
    DrawLine,
    /// Clicks append vertices to a `Polygon` sketch.
    DrawPolygon,
}

impl EditMode {
    /// Every mode, in toolbar order.
    pub const ALL: [Self; 5] = [
        Self::Off,
        Self::Select,
        Self::DrawPoint,
        Self::DrawLine,
        Self::DrawPolygon,
    ];

    /// Whether this mode digitizes new geometry.
    #[must_use]
    pub fn is_drawing(self) -> bool {
        matches!(self, Self::DrawPoint | Self::DrawLine | Self::DrawPolygon)
    }

    /// Short label for the toolbar button and the status line.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Browse",
            Self::Select => "Select",
            Self::DrawPoint => "Point",
            Self::DrawLine => "Line",
            Self::DrawPolygon => "Polygon",
        }
    }
}

/// Address of one coordinate inside a [`Geometry`].
///
/// `part` indexes a `Multi*` member (`0` otherwise), `ring` indexes a `Polygon`
/// ring (`0` = exterior), and `index` is the position within the **open**
/// sequence — a closed ring's duplicate final position is never addressable, so
/// "I dragged handle 0 and the ring came apart" is unreachable by construction.
///
/// A `MultiPoint` is addressed as a single path (`part` and `ring` both `0`)
/// whose `index` selects the member, which is what makes inserting and deleting
/// a member the same code path as inserting and deleting a line vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct VertexRef {
    /// Which `Multi*` member, or `0`.
    pub part: usize,
    /// Which polygon ring, `0` for the exterior or for a non-polygon.
    pub ring: usize,
    /// Position within the open coordinate sequence.
    pub index: usize,
}

impl VertexRef {
    /// The vertex at `index` of the first (and only) path — the common case for
    /// a `Point`, `LineString`, `MultiPoint` or a polygon's exterior ring.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self {
            part: 0,
            ring: 0,
            index,
        }
    }

    /// A fully addressed vertex.
    #[must_use]
    pub const fn at(part: usize, ring: usize, index: usize) -> Self {
        Self { part, ring, index }
    }
}

/// What is picked on the map.
///
/// Deliberately carries **no** [`LayerId`]: the edit target is always
/// [`crate::OxigisApp::selection`], so there is exactly one layer-selection
/// concept and it cannot desync from the table or the style panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditSelection {
    /// Index of the feature in the layer's **source** collection — the same
    /// number the attribute table reports and the drawn tile carries as its
    /// feature id.
    pub feature: usize,
    /// The picked vertex of that feature, when one is picked.
    pub vertex: Option<VertexRef>,
}

impl EditSelection {
    /// A whole-feature selection, with no vertex picked.
    #[must_use]
    pub const fn feature(feature: usize) -> Self {
        Self {
            feature,
            vertex: None,
        }
    }

    /// A selection of one vertex of one feature.
    #[must_use]
    pub const fn vertex(feature: usize, vertex: VertexRef) -> Self {
        Self {
            feature,
            vertex: Some(vertex),
        }
    }
}

/// A live vertex or midpoint drag.
///
/// Holds the pre-drag geometry so `Escape` restores it with no command at all,
/// and so the preview is rebuilt from ground truth every frame instead of
/// accumulating rounding error across the gesture.
#[derive(Debug, Clone, PartialEq)]
pub struct VertexDrag {
    /// Index of the feature being dragged, in the source collection.
    pub feature: usize,
    /// The vertex under the pointer when the gesture started.
    pub vertex: VertexRef,
    /// True when the gesture started on a midpoint ghost, so the commit inserts
    /// a position rather than moving one.
    pub inserting: bool,
    /// The feature's geometry as it stood before the drag.
    pub origin: Geometry,
    /// Where the dragged vertex currently sits (snapped, when snapping fired).
    pub current: LonLat,
    /// Set once the pointer actually moves: a press and release without motion
    /// is a *click* that selects the vertex, not a zero-length move that would
    /// otherwise push a no-op undo entry.
    pub moved: bool,
    /// Where the grabbed handle sat when the gesture started — the origin the
    /// set translation is measured from, stored so the delta is explicit and
    /// a long drag cannot accumulate error.
    pub start: LonLat,
    /// The marquee-marked vertices this gesture translates rigidly with the
    /// grabbed one — ascending and duplicate-free (straight from
    /// [`selection::FeatureSelection::vertex_set`], which sorts), always
    /// containing [`Self::vertex`] when non-empty. EMPTY for the plain
    /// single-vertex move and for every insert, which is exactly the v1.1
    /// gesture. Captured at press time, like [`Self::origin`]: the gesture
    /// owns what it grabbed, so nothing that happens to the selection
    /// mid-drag can change what a release commits.
    pub set: Vec<VertexRef>,
}

impl VertexDrag {
    /// A plain single-vertex (or insert) gesture — no marked set.
    #[must_use]
    pub fn single(
        feature: usize,
        vertex: VertexRef,
        inserting: bool,
        origin: Geometry,
        current: LonLat,
    ) -> Self {
        Self {
            feature,
            vertex,
            inserting,
            origin,
            start: current,
            current,
            moved: false,
            set: Vec::new(),
        }
    }

    /// Whether this gesture translates a marked set.
    ///
    /// `>= 2`, not "non-empty": a lone mark is a single-vertex move on every
    /// axis — the label (`drag_transaction` already commits it as
    /// `"Move vertex"`), the release status (no more "1 vertices moved"), the
    /// marks latch and the undo-side restoration threshold all agree on ONE
    /// definition of "this gesture moves a set".
    #[must_use]
    pub fn is_set_move(&self) -> bool {
        self.set.len() >= 2
    }

    /// This drag's translation in normalised Web Mercator world space.
    ///
    /// World space, not lon/lat degrees: Mercator is non-linear in latitude,
    /// so a constant Δlat is a *different* screen distance at each vertex —
    /// a marked cluster would visibly shear while dragged. A constant world
    /// delta is a constant screen delta at fixed zoom, i.e. the set moves
    /// rigidly under the pointer.
    #[must_use]
    pub fn translation(&self) -> [f64; 2] {
        let from = self.start.to_world();
        let to = self.current.to_world();
        [to.x - from.x, to.y - from.y]
    }

    /// Where `at` (whose committed position is `stored`) lands under this
    /// drag, or [`None`] when the drag does not move it.
    ///
    /// The grabbed vertex gets [`Self::current`] VERBATIM — a snap promised
    /// that exact stored position, and a world round trip would cost ~1 ulp.
    /// The other marked vertices translate through world space (one Mercator
    /// round trip, ~1e-15 relative, clamped at the world edge — only undo
    /// restores their coordinates bit-exact).
    #[must_use]
    pub fn target_of(&self, at: VertexRef, stored: LonLat) -> Option<LonLat> {
        if at == self.vertex && !self.inserting {
            return Some(self.current);
        }
        if self.set.binary_search(&at).is_err() {
            return None;
        }
        let delta = self.translation();
        let world = stored.to_world();
        Some(WorldCoord::new(world.x + delta[0], world.y + delta[1]).to_lon_lat())
    }

    /// The vertices this drag moves — the snap exclusion set.
    #[must_use]
    pub fn moving(&self) -> &[VertexRef] {
        if self.set.is_empty() {
            core::slice::from_ref(&self.vertex)
        } else {
            &self.set
        }
    }
}

/// A map gesture that owns the pointer button and addresses features by
/// index — the two things a change to the layer's collection makes stale.
/// What [`EditState::cancel_live_gestures`] reports it dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveGesture {
    /// A vertex or midpoint drag.
    VertexDrag,
    /// A Shift+drag box-select.
    BoxSelect,
}

/// A live Shift+drag box-select over the anchor feature's drawn handles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Marquee {
    /// Where the press landed, panel-local egui points.
    pub start: egui::Pos2,
    /// The pointer this frame, panel-local egui points.
    pub current: egui::Pos2,
}

/// What became of the selected feature's vertex handles this frame.
///
/// Recorded rather than recomputed at each use, because the *same* verdict has
/// to govern both painting and hit testing: a handle that was not drawn must
/// not be grabbable, and one that was drawn must be. Two independent decisions
/// would eventually disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Handles {
    /// Nothing is selected, the selection has no geometry, or the active tool
    /// does not show handles at all.
    #[default]
    None,
    /// Handles and midpoint ghosts are drawn, and therefore clickable.
    Active,
    /// The handle budget forced them off. The count is how many are in view,
    /// which is what the hint plate reports.
    Suppressed(usize),
}

impl Handles {
    /// Whether handles are live this frame.
    #[must_use]
    pub fn is_active(self) -> bool {
        self == Self::Active
    }

    /// How many handles were suppressed, when the budget suppressed them.
    #[must_use]
    pub fn suppressed_count(self) -> Option<usize> {
        match self {
            Self::Suppressed(count) => Some(count),
            Self::None | Self::Active => None,
        }
    }
}

/// Everything the edit system remembers between frames.
///
/// Stage 1 holds the mode machine, the selection, the in-progress sketch and
/// drag, and the notice log. The hit-test, snap, form and validation caches of
/// the full design arrive with their own modules; nothing here has to change to
/// receive them.
#[derive(Debug, Default)]
pub struct EditState {
    /// The active tool.
    mode: EditMode,
    /// Mirror of [`crate::OxigisApp::selection`], kept only so a change of
    /// layer can be detected.
    target: Option<LayerId>,
    /// What is picked inside the target layer — the full multi-feature set;
    /// [`Self::selection`] serves the anchor view every single-selection
    /// consumer reads.
    multi: Option<selection::FeatureSelection>,
    /// The geometry being digitized, if any.
    sketch: Sketch,
    /// The vertex gesture in progress, if any.
    drag: Option<VertexDrag>,
    /// The live Shift+drag box-select, if any.
    marquee: Option<Marquee>,
    /// Messages waiting to be shown, newest last, capped at [`MAX_NOTICES`].
    notices: Vec<EditNotice>,
    /// Topology issues per layer, oldest first, capped at [`MAX_NOTICES`] each.
    ///
    /// Keyed by layer rather than held for the active one alone, and therefore
    /// deliberately **not** cleared by [`Self::retarget`]: a validation run is
    /// work the user asked for, and losing it because they clicked another layer
    /// to look at it would make the Validate layer button useless for exactly
    /// the comparison it is most wanted for. Entries are dropped when the layer
    /// is removed, and by [`Self::reset`] on project load.
    ///
    /// A [`BTreeMap`] rather than a `HashMap`: the map is tiny, iteration order
    /// has to be deterministic for the tests, and [`LayerId`] is `Ord`.
    issues: BTreeMap<LayerId, Vec<FeatureIssue>>,
    /// Whether the Edit window is open.
    show_window: bool,
    /// Set when the `⚠` toolbar badge asks for the Edit window's Validation
    /// section; taken (once) by the window body, which force-expands the
    /// section that frame. A one-shot flag rather than persistent state: after
    /// the reveal the section is the user's to collapse again.
    reveal_validation: bool,
    /// Per-feature world bounding boxes of the target layer: the broad phase
    /// every hover frame and every click runs before any projection work.
    bbox_index: WorldBboxIndex,
    /// Repeat-click cycling through the features stacked under one spot.
    cycle: PickCycle,
    /// What the pointer was over when the pan gate ran this frame.
    ///
    /// Recorded there rather than re-derived after the fact because the gate is
    /// the one place that sees the pointer *before* the camera has moved — and
    /// on touch, where there is no hover, the frame the pointer first exists is
    /// already the press frame.
    hover_target: Option<HitTarget>,
    /// Whether the selected feature's handles are live this frame.
    handles: Handles,
    /// Set when `Escape` cancelled a drag whose button is still held.
    ///
    /// Without it, cancelling a vertex drag mid-gesture hands the still-pressed
    /// button straight back to the camera and the map lurches away under the
    /// pointer — the opposite of what "cancel" means.
    drag_cancelled: bool,
    /// Which snaps are live, and how close is close enough.
    snap: SnapSettings,
    /// The world-space snap grid over every visible local vector layer.
    snap_index: SnapIndex,
    /// The collections the index is asked to hold, rebuilt into the same buffer
    /// every frame so the staleness check costs no allocation.
    snap_sources: Vec<(LayerId, Arc<FeatureCollection>)>,
    /// What the pointer last snapped to, for the marker.
    hover_snap: Option<SnapResult>,
    /// The attribute form's deferred-apply buffer.
    ///
    /// Deliberately **not** cleared by [`Self::retarget`], unlike everything
    /// else that addresses a feature by index: the buffer records which layer
    /// and feature it was seeded from, and
    /// [`form::FormBuffer::sync`] keeps a dirty one across any binding change
    /// so typed data is never discarded without being offered back.
    form: FormBuffer,
    /// Screen-space buffers the overlay reuses between frames.
    scratch: overlay::OverlayScratch,
    /// Vertex marks to re-assert on the anchor once the next commit lands —
    /// how a vertex-set MOVE keeps its marks across a commit whose
    /// `selection_after` would otherwise collapse them. Sound because a
    /// translation renumbers nothing: every recorded [`VertexRef`] still
    /// names the same corner. ONE producer (the set arm of the finished
    /// drag), ONE consumer (`apply_transaction`, which takes it at the TOP so
    /// a refusal cannot leak it into the next commit); cleared by
    /// [`Self::retarget`] and [`Self::reset`].
    marks_after_commit: Option<Vec<VertexRef>>,
}

impl EditState {
    /// The active tool.
    #[must_use]
    pub fn mode(&self) -> EditMode {
        self.mode
    }

    /// Switches tool, cancelling anything the previous one had in progress.
    ///
    /// A half-drawn sketch does not survive a tool change: the vertices belong
    /// to the tool that was collecting them, and carrying them into another one
    /// produces geometry the user never asked for.
    pub fn set_mode(&mut self, mode: EditMode) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        self.sketch = Sketch::default();
        self.cancel_drag();
        self.handles = Handles::None;
        self.hover_snap = None;
    }

    /// Drops any live drag, and holds the pan gate shut until the button that
    /// was driving it comes back up.
    fn cancel_drag(&mut self) {
        if self.drag.take().is_some() {
            self.drag_cancelled = true;
        }
    }

    /// Drops whichever index-addressed gesture is live — a vertex drag or a
    /// box-select — command-free, exactly as Escape drops them, and holds the
    /// pan gate shut until the button that was driving it comes back up.
    /// Returns which one was live.
    ///
    /// This is how a data change that lands MID-GESTURE (an undo, a redo, a
    /// paste, a Delete, a shell's own commit, a file re-read) refuses to leave
    /// a stale gesture behind: a live drag holds an `origin` clone of data
    /// that is no longer on screen, and committing it would write coordinates
    /// derived from the pre-change geometry — for a set move of unchanged
    /// arity, `set_vertices` resolves every address and silently reverts the
    /// undo. Cancelling costs one re-drag; committing costs data.
    ///
    /// The sketch is deliberately left alone: it holds coordinates, not
    /// indices, so no commit can make it stale — digitized geometry must
    /// never be discarded by a change the user can absorb and continue over.
    ///
    /// Deliberately NOT `#[must_use]` — `retarget` calls it in statement
    /// position, the same reasoning `EvictionReport` follows.
    pub fn cancel_live_gestures(&mut self) -> Option<LiveGesture> {
        if self.drag.take().is_some() {
            self.drag_cancelled = true;
            self.hover_snap = None;
            return Some(LiveGesture::VertexDrag);
        }
        if self.marquee.take().is_some() {
            self.drag_cancelled = true;
            return Some(LiveGesture::BoxSelect);
        }
        None
    }

    /// What is picked inside the target layer — the **anchor** view. The
    /// full set is [`Self::multi_selection`].
    #[must_use]
    pub fn selection(&self) -> Option<EditSelection> {
        self.multi
            .as_ref()
            .map(selection::FeatureSelection::anchor_selection)
    }

    /// Replaces the selection outright with a single feature (the whole
    /// pre-v1.1 meaning): any wider set collapses to it.
    pub fn set_selection(&mut self, selection: Option<EditSelection>) {
        self.multi = selection.map(Into::into);
    }

    /// The full multi-feature selection, if anything is selected.
    #[must_use]
    pub fn multi_selection(&self) -> Option<&selection::FeatureSelection> {
        self.multi.as_ref()
    }

    /// Replaces the whole multi-feature selection.
    pub fn set_multi_selection(&mut self, selection: Option<selection::FeatureSelection>) {
        self.multi = selection;
    }

    /// Arms the one-shot marks latch: the next commit re-asserts these marks
    /// on the anchor instead of letting `selection_after` collapse them.
    /// The set-move release is the ONE producer.
    pub fn arm_marks_after_commit(&mut self, marks: Vec<VertexRef>) {
        self.marks_after_commit = Some(marks);
    }

    /// Takes the marks latch — `apply_transaction` is the ONE consumer, and
    /// it takes at the top so a refused commit cannot leak the latch into
    /// the next one.
    pub fn take_marks_after_commit(&mut self) -> Option<Vec<VertexRef>> {
        self.marks_after_commit.take()
    }

    /// Drops the marked vertex set — and any armed latch — while keeping the
    /// feature selection.
    ///
    /// For a collection replaced OUTSIDE the command choke point (the hydrate
    /// family): every [`VertexRef`] in the set addresses the data that was
    /// just thrown away, and unlike a live drag there is no gesture to cancel
    /// — the marks simply stop naming anything, and Delete would fire at
    /// whatever slid into their indices.
    pub fn clear_vertex_marks(&mut self) {
        if let Some(multi) = self.multi.as_ref()
            && !multi.vertex_set().is_empty()
        {
            self.multi = Some(multi.with_vertex_set(Vec::new()));
        }
        self.marks_after_commit = None;
    }

    /// Shift+click: toggles `feature` in the set (selects it alone when
    /// nothing was selected). Returns whether anything remains selected.
    pub fn toggle_feature(&mut self, feature: usize) -> bool {
        self.multi = match self.multi.take() {
            None => Some(selection::FeatureSelection::single(feature)),
            Some(multi) => multi.toggled(feature),
        };
        self.multi.is_some()
    }

    /// The layer being edited, as last adopted by [`Self::retarget`].
    #[must_use]
    pub fn target(&self) -> Option<LayerId> {
        self.target
    }

    /// The geometry being digitized, if any.
    #[must_use]
    pub fn sketch(&self) -> &Sketch {
        &self.sketch
    }

    /// The sketch, for the draw tools to append to and the rubber band to read.
    pub fn sketch_mut(&mut self) -> &mut Sketch {
        &mut self.sketch
    }

    /// The vertex gesture in progress, if any.
    #[must_use]
    pub fn drag(&self) -> Option<&VertexDrag> {
        self.drag.as_ref()
    }

    /// Starts, updates or ends the vertex gesture.
    pub fn set_drag(&mut self, drag: Option<VertexDrag>) {
        self.drag = drag;
    }

    /// The live box-select, if one is being dragged.
    #[must_use]
    pub fn marquee(&self) -> Option<Marquee> {
        self.marquee
    }

    /// Ends the live box-select, handing it to the caller to resolve.
    pub fn take_marquee(&mut self) -> Option<Marquee> {
        self.marquee.take()
    }

    /// Takes the vertex gesture, ending it — how a drag release reads what it
    /// has to commit.
    pub fn take_drag(&mut self) -> Option<VertexDrag> {
        self.drag.take()
    }

    /// Whether the Edit window is open.
    #[must_use]
    pub fn show_window(&self) -> bool {
        self.show_window
    }

    /// Opens or closes the Edit window.
    pub fn set_show_window(&mut self, show: bool) {
        self.show_window = show;
    }

    /// Asks the Edit window to expand its Validation section on its next draw
    /// — what the `⚠` toolbar badge promises.
    pub fn request_reveal_validation(&mut self) {
        self.reveal_validation = true;
    }

    /// Takes the one-shot Validation reveal, clearing it.
    #[must_use]
    pub fn take_reveal_validation(&mut self) -> bool {
        core::mem::take(&mut self.reveal_validation)
    }

    /// The notices waiting to be shown, oldest first.
    #[must_use]
    pub fn notices(&self) -> &[EditNotice] {
        &self.notices
    }

    /// Records a notice, dropping the oldest once [`MAX_NOTICES`] is reached.
    pub fn push_notice(&mut self, notice: EditNotice) {
        if self.notices.len() >= MAX_NOTICES {
            self.notices.remove(0);
        }
        self.notices.push(notice);
    }

    /// Drops every notice.
    pub fn clear_notices(&mut self) {
        self.notices.clear();
    }

    /// The topology issues recorded for `layer`, oldest first.
    #[must_use]
    pub fn issues(&self, layer: LayerId) -> &[FeatureIssue] {
        match self.issues.get(&layer) {
            Some(issues) => issues,
            None => &[],
        }
    }

    /// How many topology issues `layer` has — the toolbar's `⚠ N` counter.
    #[must_use]
    pub fn issue_count(&self, layer: LayerId) -> usize {
        self.issues.get(&layer).map_or(0, Vec::len)
    }

    /// Replaces `layer`'s issues wholesale: what one **Validate layer** run
    /// stores.
    ///
    /// Over [`MAX_NOTICES`] the **oldest** are dropped, matching
    /// [`Self::push_notice`]: the list is read newest-first, so trimming the
    /// other end would hide exactly the rows the user is looking at.
    pub fn set_issues(&mut self, layer: LayerId, mut issues: Vec<FeatureIssue>) {
        if issues.is_empty() {
            self.issues.remove(&layer);
            return;
        }
        if issues.len() > MAX_NOTICES {
            let excess = issues.len() - MAX_NOTICES;
            issues.drain(..excess);
        }
        self.issues.insert(layer, issues);
    }

    /// Replaces the issues recorded for the `touched` features with `fresh`,
    /// keeping every other feature's — what one commit's revalidation stores.
    ///
    /// Issues addressing a feature index the collection no longer has are
    /// dropped here, which is what keeps a delete from leaving rows that point
    /// past the end. The caller is expected to run
    /// [`Self::remap_issue_indices`] first when the transaction added or
    /// removed features, so `touched` (post-transaction indices) and the
    /// retained issues agree on what each index names.
    pub fn merge_issues(
        &mut self,
        layer: LayerId,
        touched: &[usize],
        feature_count: usize,
        fresh: Vec<FeatureIssue>,
    ) {
        let mut merged: Vec<FeatureIssue> = self
            .issues
            .remove(&layer)
            .unwrap_or_default()
            .into_iter()
            .filter(|issue| issue.feature < feature_count && !touched.contains(&issue.feature))
            .collect();
        merged.extend(fresh);
        self.set_issues(layer, merged);
    }

    /// Drops every topology issue recorded for `layer`.
    pub fn clear_issues(&mut self, layer: LayerId) {
        self.issues.remove(&layer);
    }

    /// Renumbers `layer`'s recorded issues through a transaction's operations,
    /// so an add or delete below a feature does not leave its issues — and
    /// their fly-to coordinates — attached to whatever slid into its old index.
    ///
    /// Issues whose feature one of `ops` removed are dropped outright; the
    /// caller's follow-up revalidation of the touched indices supplies the
    /// fresh truth for those slots. The geometry of a merely *shifted* feature
    /// did not change, so its issues — including their `at` coordinates — stay
    /// correct verbatim under the new index.
    pub fn remap_issue_indices(&mut self, layer: LayerId, ops: &[FeatureOp]) {
        if ops.iter().all(|op| matches!(op, FeatureOp::Replace { .. })) {
            return;
        }
        let Some(issues) = self.issues.get_mut(&layer) else {
            return;
        };
        issues.retain_mut(|issue| match command::remap_index(ops, issue.feature) {
            Some(index) => {
                issue.feature = index;
                true
            }
            None => false,
        });
        if issues.is_empty() {
            self.issues.remove(&layer);
        }
    }

    /// Adopts a new edit target.
    ///
    /// A *different* layer discards the selection, the sketch, the drag and the
    /// notices: every one of them addresses features by index inside one
    /// collection, so nothing may cross a layer boundary. Re-targeting the same
    /// layer is a no-op, which is what lets this be called every frame.
    ///
    /// Returns a notice when something the user could see was discarded.
    pub fn retarget(&mut self, target: Option<LayerId>) -> Option<EditNotice> {
        if self.target == target {
            return None;
        }
        let discarded_sketch = self.sketch.is_active();
        self.target = target;
        self.multi = None;
        self.marks_after_commit = None;
        self.sketch = Sketch::default();
        // Both index-addressed gestures, not just the drag: a live marquee's
        // anchor is equally meaningless on another layer.
        self.cancel_live_gestures();
        self.notices.clear();
        self.handles = Handles::None;
        self.hover_snap = None;
        // The snap index holds an `Arc` per source layer; the new target's
        // sources are rebuilt on the very next frame anyway.
        self.snap_index.clear();
        self.snap_sources.clear();
        // Every cached pick addresses features by index inside one collection,
        // so none of it may cross a layer boundary either. The bbox index also
        // holds an `Arc` the new target has no use for.
        self.cycle.clear();
        self.hover_target = None;
        self.bbox_index.clear();
        discarded_sketch
            .then(|| EditNotice::new("The unfinished sketch was discarded when the layer changed."))
    }

    /// Clamps the selection and any live drag against the layer's current
    /// features — cheap enough to run every frame, which is what makes
    /// "selection points at a feature that no longer exists" unrepresentable
    /// rather than merely unlikely.
    pub fn validate_selection(&mut self, features: Option<&FeatureCollection>) {
        let Some(features) = features else {
            self.multi = None;
            // Through the latch, not a bare `self.drag = None`: the button
            // may still be down, and dropping the drag without latching
            // `drag_cancelled` would hand the rest of the gesture to the
            // camera.
            self.cancel_drag();
            return;
        };
        let count = features.features.len();
        // Re-anchors rather than dropping the whole set: a stale anchor is
        // only one member of a marquee selection, and the set's other
        // members may still address real features. `clamped` also drops
        // every OTHER stale member — a set that survived only because its
        // anchor happened to stay in range is exactly what let one stray
        // index reject a whole multi-delete with `IndexOutOfRange`.
        self.multi = self.multi.take().and_then(|multi| multi.clamped(count));
        if let Some(multi) = self.multi.take() {
            let has_vertex_state =
                multi.picked_vertex().is_some() || !multi.vertex_set().is_empty();
            self.multi = Some(if has_vertex_state {
                let geometry = features
                    .features
                    .get(multi.anchor())
                    .and_then(|feature| feature.geometry.as_ref());
                match geometry {
                    // The anchor lost its geometry outright: a picked vertex
                    // or a marked set into nothing is stale even though the
                    // feature itself survived.
                    None => multi.with_vertex(None),
                    // The anchor kept its geometry, but a delete, an undo or
                    // a paste may have shrunk it under indices the picked
                    // vertex or the marked set still name — checked against
                    // the SAME `(part, ring, index)` addressing the mutators
                    // accept, so a mark that no longer resolves here could
                    // never resolve there either.
                    Some(geometry) => {
                        let paths = command::paths(geometry);
                        multi.clamped_vertices(|at| {
                            paths.iter().any(|path| {
                                path.part == at.part
                                    && path.ring == at.ring
                                    && at.index < path.positions.len()
                            })
                        })
                    }
                }
            } else {
                multi
            });
        }
        if self.drag.as_ref().is_some_and(|drag| drag.feature >= count) {
            self.cancel_drag();
        }
    }

    /// The Escape ladder: sketch → drag → vertex → feature → draw mode → `Off`.
    ///
    /// One press undoes exactly one layer of in-progress-ness, never more, so a
    /// user who mashes Escape lands in a known state without ever destroying
    /// committed data. Returns whether anything was consumed — a `false` lets
    /// the caller leave the key for someone else.
    pub fn escape(&mut self) -> bool {
        if self.sketch.is_active() {
            self.sketch = Sketch::default();
            return true;
        }
        if self.marquee.take().is_some() {
            // The button may still be down: the rest of the gesture must not
            // fall through to the camera — the same latch a cancelled drag
            // sets.
            self.drag_cancelled = true;
            return true;
        }
        if self.drag.is_some() {
            // The pre-drag geometry was never committed, so simply dropping the
            // gesture restores it: there is nothing to undo and nothing to say
            // to the undo stack.
            self.cancel_drag();
            self.hover_snap = None;
            return true;
        }
        // A marked vertex set is a layer of in-progress-ness of its own: one
        // press clears the marks and keeps the feature selection.
        if self
            .multi
            .as_ref()
            .is_some_and(|multi| !multi.vertex_set().is_empty())
        {
            self.multi = self
                .multi
                .take()
                .map(|multi| multi.with_vertex_set(Vec::new()));
            return true;
        }
        if let Some(selection) = self.selection()
            && selection.vertex.is_some()
        {
            self.multi = self.multi.take().map(|multi| multi.with_vertex(None));
            return true;
        }
        if self.multi.take().is_some() {
            return true;
        }
        if self.mode.is_drawing() {
            self.mode = EditMode::Select;
            return true;
        }
        if self.mode != EditMode::Off {
            self.mode = EditMode::Off;
            return true;
        }
        false
    }

    /// Whether the map panel may consume this frame's primary drag as a camera
    /// pan — and, on the way, what the pointer is over.
    ///
    /// Called from inside
    /// [`crate::map_view::MapPanelState::allocate_gated`], **after** the rect
    /// and the [`egui::Response`] exist and **before** a single pixel of pan is
    /// applied, so its verdict comes from this frame's real pointer rather than
    /// from last frame's hover. That distinction is not academic: on touch, the
    /// first frame the pointer exists at all is already the press frame, so any
    /// scheme that arms itself from a previous frame is wrong there by
    /// construction.
    ///
    /// The verdict is [`PanGate::Suppress`] exactly while a vertex gesture owns
    /// the button — from the frame the press lands on a handle to the frame it
    /// is released — and [`PanGate::Allow`] otherwise. [`EditMode::Off`] returns
    /// before any of it: with editing off the map must behave bit-identically to
    /// a build without this module, and that is invariant I11.
    #[expect(
        clippy::too_many_arguments,
        reason = "the pan gate is a per-frame context: bundling it into a struct \
                  would only move the same seven values one level down, and the \
                  call site is a single closure in OxigisApp::ui"
    )]
    pub fn gate_pan(
        &mut self,
        rect: egui::Rect,
        response: &egui::Response,
        ppp: f32,
        project: &Project,
        local: &LocalInputState,
        selection: Option<LayerId>,
        view: MapView,
    ) -> PanGate {
        // The cancelled-drag latch is serviced before the mode check, because a
        // drag can be cancelled *by* a mode switch — including one to `Off` —
        // and a latch that only ticks in the mode that set it would strand
        // itself, swallowing the next gesture whole once the mode came back.
        // Suppressing the tail of that gesture even in `Off` does not breach
        // invariant I11: the gesture being suppressed started as a vertex drag,
        // never as a pan, and handing its remainder to the camera is exactly
        // the lurch the latch exists to prevent.
        let primary_down = response.ctx.input(|input| input.pointer.primary_down());
        if self.drag_cancelled && !primary_down {
            self.drag_cancelled = false;
        }
        if self.mode == EditMode::Off {
            self.hover_target = None;
            self.handles = Handles::None;
            self.hover_snap = None;
            return if self.drag_cancelled {
                PanGate::Suppress
            } else {
                PanGate::Allow
            };
        }
        // `allocate_gated` has already resized the camera to this rect; the
        // value captured before the call has not seen that yet, so reproduce it
        // rather than hit-test against a stale viewport on a resize frame.
        let view = view
            .with_size_px([rect.width() * ppp, rect.height() * ppp])
            .unwrap_or(view);
        let features = selection.and_then(|id| local.feature_set(id));
        self.sync_indexes(selection, features);
        self.sync_snap_index(project, local, selection);
        let ctx = EditCtx {
            project,
            target: selection,
            features,
            view,
            rect,
            ppp,
        };
        // Touch has no hover, so the interaction position comes first: on touch
        // the first frame the pointer exists at all is already the press frame,
        // and anything that arms itself from a previous frame's hover is wrong
        // there by construction.
        let pointer = response
            .interact_pointer_pos()
            .or_else(|| response.hover_pos());
        // Ctrl suspends snapping for as long as it is held. A per-gesture
        // decision beats a mode to remember, and Ctrl has no other map meaning.
        let suspend_snap = response.ctx.input(|input| input.modifiers.ctrl);

        self.handles = plan_handles(&ctx, self.mode, self.selection());

        // A live gesture owns the button wherever the pointer goes, including
        // outside the panel — releasing over the layer panel must still end the
        // drag where the pointer last was, not wherever it was last inside.
        if self.drag.is_some() {
            self.update_drag(&ctx, pointer, suspend_snap, selection);
            self.hover_target = None;
            return PanGate::Suppress;
        }
        // A live box-select owns the button exactly like a vertex drag; its
        // resolution (on release) happens in the interact pass, which runs
        // after this and has the app state the resolution mutates.
        if let Some(marquee) = self.marquee.as_mut() {
            if let Some(pointer) = pointer {
                marquee.current = pointer;
            }
            self.hover_target = None;
            return PanGate::Suppress;
        }
        if self.drag_cancelled {
            return PanGate::Suppress;
        }

        let inside = pointer.filter(|pointer| rect.contains(*pointer));
        self.hover_target = if self.mode.is_drawing() {
            // A drawing tool cares about the map, not about what is under the
            // pointer; the crosshair says so.
            None
        } else {
            inside.and_then(|pointer| self.hit_test(&ctx, pointer, self.handles.is_active()))
        };
        // Outside a gesture the marker would only be noise: nothing is being
        // placed, so there is nothing for it to promise. Cleared here rather
        // than left over, so the only code that can put a marker on screen is
        // code that ran *this* frame — the digitizing tools set it from
        // `edit_interact`, which runs after this and before the overlay paints.
        self.hover_snap = None;

        // The press is hit-tested at the position the button actually went down
        // at, not at wherever the pointer has since travelled: egui only calls a
        // gesture a drag once it has left the click radius, so by this frame the
        // pointer has already moved off the handle the user grabbed.
        if self.mode == EditMode::Select && response.drag_started_by(egui::PointerButton::Primary) {
            let press = response
                .ctx
                .input(|input| input.pointer.press_origin())
                .or(inside)
                .filter(|press| rect.contains(*press));
            if let Some(press) = press {
                // Shift is tested BEFORE the handle hit test: a Shift+drag is
                // a box-select, never a handle drag. It arms only while
                // handles are drawn ("what you can mark is what you can
                // see"); with them suppressed the drag stays a pan.
                let shift = response.ctx.input(|input| input.modifiers.shift);
                if shift {
                    if self.handles.is_active() {
                        self.marquee = Some(Marquee {
                            start: press,
                            current: inside.unwrap_or(press),
                        });
                        return PanGate::Suppress;
                    }
                } else {
                    let target = self.hit_test(&ctx, press, self.handles.is_active());
                    if self.start_drag(&ctx, target) {
                        return PanGate::Suppress;
                    }
                }
            }
        }
        PanGate::Allow
    }

    /// Rebuilds the broad-phase index when the target layer's collection
    /// changed, and drops it when there is no target.
    ///
    /// Cheap enough to call every frame: [`hit::WorldBboxIndex`] compares the
    /// held [`Arc`] and returns immediately when nothing moved.
    pub fn sync_indexes(
        &mut self,
        target: Option<LayerId>,
        features: Option<&Arc<FeatureCollection>>,
    ) {
        match (target, features) {
            (Some(id), Some(features)) => self.bbox_index.rebuild_if_stale(id, features),
            _ => self.bbox_index.clear(),
        }
    }

    /// Rebuilds the snap index when — and only when — one of its source
    /// collections has actually been replaced.
    ///
    /// The candidate set is the active edit layer first (so it is the one kept
    /// if the budget degrades), then every other **visible** local vector layer
    /// whose features are loaded, in stack order: snapping to a reference layer
    /// is most of the reason snapping exists.
    pub fn sync_snap_index(
        &mut self,
        project: &Project,
        local: &LocalInputState,
        target: Option<LayerId>,
    ) {
        if !self.snap.enabled {
            self.snap_index.clear();
            self.snap_sources.clear();
            return;
        }
        self.snap_sources.clear();
        if let Some(id) = target
            && let Some(features) = local.feature_set(id)
        {
            self.snap_sources.push((id, Arc::clone(features)));
        }
        for layer in project.layers.layers() {
            if Some(layer.id) == target || !layer.visible || !local_input::is_local_layer(layer) {
                continue;
            }
            if let Some(features) = local.feature_set(layer.id) {
                self.snap_sources.push((layer.id, Arc::clone(features)));
            }
        }
        self.snap_index.rebuild_if_stale(&self.snap_sources);
    }

    /// Adopts `target` as a vertex gesture, if it is something draggable.
    /// Returns whether a gesture started.
    fn start_drag(&mut self, ctx: &EditCtx<'_>, target: Option<HitTarget>) -> bool {
        let (feature, vertex, inserting) = match target {
            Some(HitTarget::Vertex { feature, at }) => (feature, at, false),
            Some(HitTarget::Midpoint { feature, at }) => (feature, at, true),
            Some(HitTarget::Feature { .. }) | None => return false,
        };
        let Some(geometry) = ctx
            .features
            .and_then(|features| features.features.get(feature))
            .and_then(|feature| feature.geometry.as_ref())
        else {
            return false;
        };
        let Some(current) = hit::handle_position(geometry, vertex, inserting, ctx.view, ctx.ppp)
        else {
            return false;
        };
        // The pre-drag geometry, so Escape restores it with no command and
        // the preview is rebuilt from ground truth rather than accumulated.
        let mut drag = VertexDrag::single(feature, vertex, inserting, geometry.clone(), current);
        // A plain drag on a MARKED handle translates the whole marked set
        // (never an insert — an insert renumbers, and the marks would
        // corrupt). The anchor guard is structural — `hit::pick` only offers
        // the anchor's handles — but stated anyway.
        if !inserting
            && let Some(multi) = self.multi.as_ref()
            && multi.anchor() == feature
            && multi.vertex_set().binary_search(&vertex).is_ok()
        {
            drag.set = multi.vertex_set().to_vec();
        }
        self.drag = Some(drag);
        true
    }

    /// Moves the live gesture to this frame's pointer, snapped.
    ///
    /// Nothing here allocates an `Arc`, queues a GPU op or re-tessellates: a
    /// drag frame is pure arithmetic plus a grid lookup, and the committed data
    /// is not touched until the button comes up.
    fn update_drag(
        &mut self,
        ctx: &EditCtx<'_>,
        pointer: Option<egui::Pos2>,
        suspend_snap: bool,
        layer: Option<LayerId>,
    ) {
        let Some(pointer) = pointer else {
            return;
        };
        let Some(feature) = self.drag.as_ref().map(|drag| drag.feature) else {
            return;
        };
        let local = pointer - ctx.rect.min;
        let raw = ctx
            .view
            .screen_to_lon_lat([local.x * ctx.ppp, local.y * ctx.ppp]);
        // A moving vertex must not snap to itself, nor to either segment it is
        // an endpoint of — those follow the pointer, so they would attract it
        // from anywhere and the vertex could never be moved at all. The same
        // address does the same job for an insert: the vertex it names is the
        // neighbour a snap would collapse the new position onto, and the
        // segments it names include the one the ghost was pulled out of, which
        // would otherwise drag the new vertex straight back onto the line it
        // came from. A SET drag widens the exclusion to every marked vertex:
        // the others are stationary attractors at their pre-drag positions,
        // and without this a small nudge is yanked straight back.
        let snapped = if suspend_snap {
            None
        } else {
            // Split borrows: the exclusion slice borrows the drag while the
            // query borrows the index — disjoint fields of `self`.
            let Self {
                drag, snap_index, ..
            } = self;
            let moving = drag.as_ref().map(VertexDrag::moving);
            let exclude = layer
                .zip(moving)
                .map(|(id, vertices)| (id, feature, vertices));
            snap_index.query_excluding_set(
                ctx.view,
                ctx.rect.min,
                pointer,
                self.snap,
                exclude,
                ctx.ppp,
            )
        };
        let next = snapped.map_or(raw, |result| result.position);
        self.hover_snap = snapped;
        if let Some(drag) = self.drag.as_mut() {
            if next != drag.current {
                drag.moved = true;
            }
            drag.current = next;
        }
    }

    /// Re-plans the handle verdict from a fresh context — called at overlay
    /// time, after this frame's clicks have resolved.
    ///
    /// [`Self::gate_pan`] plans handles *before* `edit_interact` runs, so on
    /// the frame a click picks a new feature the gate's verdict still describes
    /// the previous selection and the freshly picked feature would paint with
    /// no handles until the next frame — an entire missed beat in a reactive
    /// event loop that renders nothing until the next input. The overlay
    /// re-plans here so painting always reflects this frame's selection; the
    /// gate's own hit tests keep using the value it planned, which is correct
    /// for them because a press can only land on handles that were *drawn*.
    pub fn refresh_handles(&mut self, ctx: &EditCtx<'_>) {
        self.handles = plan_handles(ctx, self.mode, self.selection());
    }

    /// Re-derives the live drag's position — and its snap marker — from a
    /// fresh context; called from `edit_interact`, after `allocate_gated` has
    /// applied this frame's wheel/pinch zoom.
    ///
    /// [`Self::gate_pan`] updates the drag *before* the zoom lands, so on a
    /// frame that both drags and zooms the committed position and the marker
    /// would be one zoom step's reprojection away from where the overlay
    /// (painted with the post-zoom camera) shows them. Re-running the update
    /// against the post-zoom view closes that gap; on the frames that did not
    /// zoom it recomputes the same position and moves nothing.
    pub fn refresh_drag(
        &mut self,
        ctx: &EditCtx<'_>,
        pointer: Option<egui::Pos2>,
        suspend_snap: bool,
    ) {
        if self.drag.is_some() {
            self.update_drag(ctx, pointer, suspend_snap, ctx.target);
        }
    }

    /// Where a digitizing tool's next vertex would land, snapped.
    ///
    /// The sketch's **own first vertex** is checked first and preferred: closing
    /// a ring is a decision the user makes, and an ordinary vertex snap to some
    /// other layer half a pixel nearer must not steal it. `exclude` is
    /// deliberately [`None`] — nothing is being dragged, so there is no vertex
    /// that would attract itself.
    ///
    /// Records what was snapped to, so the marker paints: this runs from
    /// `edit_interact`, which is *after* the pan gate cleared the marker and
    /// *before* the overlay reads it.
    pub fn snap_for_sketch(
        &mut self,
        ctx: &EditCtx<'_>,
        pointer: egui::Pos2,
        suspend_snap: bool,
    ) -> LonLat {
        let local = pointer - ctx.rect.min;
        let raw = ctx
            .view
            .screen_to_lon_lat([local.x * ctx.ppp, local.y * ctx.ppp]);
        if suspend_snap {
            self.hover_snap = None;
            return raw;
        }
        // Only from the second vertex on: offered at the first, the sketch's own
        // start would simply trap vertex two on top of vertex one.
        let start = (self.sketch.len() >= 2)
            .then(|| self.sketch.first())
            .flatten()
            .zip(ctx.target)
            .and_then(|(first, layer)| {
                snap::snap_to_sketch_start(
                    ctx.view,
                    ctx.rect.min,
                    pointer,
                    self.snap,
                    layer,
                    first,
                    ctx.ppp,
                )
            });
        let result = start.or_else(|| {
            self.snap_index
                .query(ctx.view, ctx.rect.min, pointer, self.snap, None, ctx.ppp)
        });
        self.hover_snap = result;
        result.map_or(raw, |snapped| snapped.position)
    }

    /// What is under `at_pt`, changing nothing.
    ///
    /// `handles_active` says whether vertex handles were drawn this frame; a
    /// target that is not drawn must not be clickable, so this is a parameter
    /// rather than an assumption.
    #[must_use]
    pub fn hit_test(
        &self,
        ctx: &EditCtx<'_>,
        at_pt: egui::Pos2,
        handles_active: bool,
    ) -> Option<HitTarget> {
        hit::pick(
            ctx,
            &self.bbox_index,
            self.selection().map(|selection| selection.feature),
            at_pt,
            handles_active,
        )
    }

    /// Resolves a `Select`-mode click at `at_pt`, advancing the repeat-click
    /// cycle.
    ///
    /// Returns the feature the click landed on, its **one-based** position in
    /// the stack of candidates under that spot, and how many candidates there
    /// were — which is what the status line reports as `feature 2 of 4 here`,
    /// the only way a user ever discovers that cycling exists. Deliberately does
    /// not touch [`Self::selection`]: the caller owns that, and owns the status
    /// line that has to agree with it.
    pub fn pick_click(
        &mut self,
        ctx: &EditCtx<'_>,
        at_pt: egui::Pos2,
    ) -> Option<(usize, usize, usize)> {
        let features = ctx.features?;
        let candidates = hit::pick_features(
            features,
            &self.bbox_index,
            ctx.view,
            ctx.rect.min,
            at_pt,
            ctx.ppp,
            // The SAME style `hit::pick` resolves with: the click cycle and the
            // hover target have to see one candidate list, or a marker could
            // highlight under the pointer and then not be selectable by a click
            // at the same pixel.
            hit::layer_style(ctx),
        );
        let feature = self.cycle.next(at_pt, &candidates)?;
        let (position, total) = self.cycle.position()?;
        Some((feature, position, total))
    }

    /// What the pointer was over when the pan gate last ran.
    #[must_use]
    pub fn hover_target(&self) -> Option<HitTarget> {
        self.hover_target
    }

    /// Whether the selected feature's handles are live this frame.
    #[must_use]
    pub fn handles(&self) -> Handles {
        self.handles
    }

    /// Which snaps are live, and how close is close enough.
    #[must_use]
    pub fn snap_settings(&self) -> SnapSettings {
        self.snap
    }

    /// Turns snapping on or off.
    pub fn set_snap_enabled(&mut self, enabled: bool) {
        self.snap.enabled = enabled;
        if !enabled {
            self.hover_snap = None;
            // An index nobody may query is a few megabytes of held collections
            // for nothing.
            self.snap_index.clear();
            self.snap_sources.clear();
        }
    }

    /// Replaces the whole snap configuration — what the Edit window's snap
    /// section writes back.
    ///
    /// Routed through [`Self::set_snap_enabled`] rather than assigning the
    /// field, so turning snapping off through the window releases the held
    /// collections exactly as the toolbar toggle does.
    pub fn set_snap_settings(&mut self, settings: SnapSettings) {
        self.snap = SnapSettings {
            enabled: self.snap.enabled,
            ..settings
        };
        self.set_snap_enabled(settings.enabled);
    }

    /// The attribute form's buffer.
    #[must_use]
    pub fn form(&self) -> &FormBuffer {
        &self.form
    }

    /// The attribute form's buffer, for [`form::FormBuffer::sync`] and the
    /// widget.
    pub fn form_mut(&mut self) -> &mut FormBuffer {
        &mut self.form
    }

    /// What the pointer last snapped to, for the marker.
    #[must_use]
    pub fn hover_snap(&self) -> Option<SnapResult> {
        self.hover_snap
    }

    /// Whether the snap index fell back to the active layer alone.
    #[must_use]
    pub fn snap_degraded(&self) -> bool {
        self.snap.enabled && self.snap_index.is_degraded()
    }

    /// The snap index, for tests and for callers that drive their own query.
    #[must_use]
    pub fn snap_index(&self) -> &SnapIndex {
        &self.snap_index
    }

    /// Where the repeat-click cycle stands, as `(one-based position, total)`.
    #[must_use]
    pub fn cycle_position(&self) -> Option<(usize, usize)> {
        self.cycle.position()
    }

    /// Forgets the repeat-click cycle, so the next click starts a fresh one.
    pub fn clear_cycle(&mut self) {
        self.cycle.clear();
    }

    /// The broad-phase index, for tests and for the overlay's culling.
    #[must_use]
    pub fn bbox_index(&self) -> &WorldBboxIndex {
        &self.bbox_index
    }

    /// Paints the selection outline for `geometry`, reusing this state's
    /// screen-space buffers.
    pub fn paint_selection(
        &mut self,
        painter: &egui::Painter,
        view: MapView,
        rect: egui::Rect,
        ppp: f32,
        geometry: &Geometry,
    ) {
        overlay::paint_selection(painter, view, rect, ppp, geometry, &mut self.scratch);
    }

    /// Paints the live vertex gesture, if there is one: the pre-edit ghost and
    /// the preview.
    pub fn paint_drag(
        &mut self,
        painter: &egui::Painter,
        view: MapView,
        rect: egui::Rect,
        ppp: f32,
    ) {
        if let Some(drag) = self.drag.as_ref() {
            overlay::paint_drag(painter, view, rect, ppp, drag, &mut self.scratch);
        }
    }

    /// Paints the in-progress sketch's rubber band, if there is one.
    pub fn paint_sketch(
        &mut self,
        painter: &egui::Painter,
        view: MapView,
        rect: egui::Rect,
        ppp: f32,
    ) {
        // Split borrows: the sketch is read while the scratch buffers are
        // written, and both live in `self`.
        let Self {
            sketch, scratch, ..
        } = self;
        overlay::paint_sketch(painter, view, rect, ppp, sketch, scratch);
    }

    /// Full reset: a project load or File ▸ New leaves nothing behind.
    pub fn reset(&mut self) {
        let held = self.drag.is_some() || self.marquee.is_some();
        *self = Self::default();
        // A gesture cancelled by a project load must not hand its still-held
        // button to the camera either — the pan latch survives the reset.
        self.drag_cancelled = held;
    }
}

/// Whether the selected feature's handles can be drawn — and therefore grabbed
/// — this frame.
///
/// Handles are culled to what a click inside the panel could reach first, and
/// only then measured against [`hit::HANDLE_BUDGET`]. Culling before budgeting
/// is what keeps a 200 000-vertex coastline editable when zoomed in on one bay;
/// budgeting the whole feature instead would make the geometry that most needs
/// editing the only geometry that cannot be edited.
#[must_use]
pub fn plan_handles(
    ctx: &EditCtx<'_>,
    mode: EditMode,
    selection: Option<EditSelection>,
) -> Handles {
    if mode != EditMode::Select {
        return Handles::None;
    }
    let Some(geometry) = selection
        .and_then(|selection| {
            ctx.features
                .and_then(|features| features.features.get(selection.feature))
        })
        .and_then(|feature| feature.geometry.as_ref())
    else {
        return Handles::None;
    };
    let count = hit::visible_handle_count(geometry, ctx.view, ctx.rect, ctx.ppp);
    if count == 0 {
        Handles::None
    } else if count <= hit::HANDLE_BUDGET {
        Handles::Active
    } else {
        Handles::Suppressed(count)
    }
}

/// The transaction one finished vertex gesture asks for.
///
/// Split out of the drag-release path so the whole commit shape — which mutator
/// runs, what the undo label reads, where the selection lands — is reachable
/// from a test with a synthetic [`VertexDrag`], without an egui frame, a GPU or
/// a simulated pointer.
///
/// # Errors
///
/// [`EditError::IndexOutOfRange`] when the feature has gone since the gesture
/// started, and whatever [`command::set_vertex`] or [`command::insert_vertex`]
/// refuses. Nothing is changed in either case: the caller holds a clone.
pub fn drag_transaction(
    layer: LayerId,
    features: &FeatureCollection,
    drag: &VertexDrag,
    selection_before: Option<EditSelection>,
) -> Result<EditTransaction, EditError> {
    let before = features
        .features
        .get(drag.feature)
        .ok_or(EditError::IndexOutOfRange {
            index: drag.feature,
            len: features.features.len(),
        })?
        .clone();
    let mut after = before.clone();
    let label = if drag.inserting {
        command::insert_vertex(&mut after, drag.vertex, drag.current)?;
        "Insert vertex"
    } else if drag.set.is_empty() {
        command::set_vertex(&mut after, drag.vertex, drag.current)?;
        "Move vertex"
    } else {
        // A set move: the whole marked set translates by the gesture's world
        // delta, as ONE `Replace`. A shortfall means a mark no longer
        // addresses a position in the pre-drag geometry (a Ctrl+Z mid-drag,
        // say) — refuse WHOLE rather than half-move the set.
        let moves = drag_translation(drag);
        if moves.len() != drag.set.len() {
            let missing = drag
                .set
                .iter()
                .find(|at| !moves.iter().any(|(moved, _)| moved == *at))
                .copied()
                .unwrap_or(drag.vertex);
            return Err(EditError::BadVertex(missing));
        }
        command::set_vertices(&mut after, &moves)?;
        if moves.len() > 1 {
            MOVE_VERTICES_LABEL
        } else {
            "Move vertex"
        }
    };
    Ok(EditTransaction {
        layer,
        label,
        ops: vec![FeatureOp::Replace {
            index: drag.feature,
            before: Box::new(before),
            after: Box::new(after),
        }],
        selection_before,
        // The moved — or newly inserted — vertex stays picked, so `Delete`
        // straight after a drag means the vertex the user was just holding.
        // (For a set move the marks latch re-asserts the set on top of this
        // at apply time.)
        selection_after: Some(EditSelection::vertex(drag.feature, drag.vertex)),
        // One gesture is one undo step: a drag that folded into its predecessor
        // would make Ctrl+Z undo an unpredictable number of moves.
        coalesce: None,
    })
}

/// The undo/menu label of a committed vertex-set move — the exact
/// discriminator gesture tests key on.
pub const MOVE_VERTICES_LABEL: &str = "Move vertices";

/// Where every vertex this gesture moves will land: the grabbed vertex at
/// [`VertexDrag::current`] verbatim, every other marked vertex translated by
/// the same world-space delta.
///
/// Marks that no longer address a position in [`VertexDrag::origin`] are
/// simply absent from the result — a painter must never fail;
/// [`drag_transaction`] turns the shortfall into a whole-gesture refusal.
#[must_use]
pub fn drag_translation(drag: &VertexDrag) -> Vec<(VertexRef, LonLat)> {
    let mut out = Vec::with_capacity(drag.set.len());
    for path in command::paths(&drag.origin) {
        for (index, position) in path.positions.iter().enumerate() {
            let reference = VertexRef::at(path.part, path.ring, index);
            if drag.set.binary_search(&reference).is_err() {
                continue;
            }
            let (Some(&lon), Some(&lat)) = (position.first(), position.get(1)) else {
                continue;
            };
            if !(lon.is_finite() && lat.is_finite()) {
                continue;
            }
            if let Some(target) = drag.target_of(reference, LonLat::new(lon, lat)) {
                out.push((reference, target));
            }
        }
    }
    out
}

/// The transaction a `Delete` on a picked vertex asks for.
///
/// # Errors
///
/// [`EditError::IndexOutOfRange`], or whatever [`command::remove_vertex`]
/// refuses — notably [`EditError::TooFewVertices`], which is how a delete that
/// would destroy the geometry is turned into a message rather than a loss.
pub fn remove_vertex_transaction(
    layer: LayerId,
    features: &FeatureCollection,
    feature: usize,
    at: VertexRef,
    selection_before: Option<EditSelection>,
) -> Result<EditTransaction, EditError> {
    let before = features
        .features
        .get(feature)
        .ok_or(EditError::IndexOutOfRange {
            index: feature,
            len: features.features.len(),
        })?
        .clone();
    let mut after = before.clone();
    command::remove_vertex(&mut after, at)?;
    Ok(EditTransaction {
        layer,
        label: "Delete vertex",
        ops: vec![FeatureOp::Replace {
            index: feature,
            before: Box::new(before),
            after: Box::new(after),
        }],
        selection_before,
        // The vertex is gone; leaving the selection pointing at it would leave
        // `Delete` armed against whatever slid into its index.
        selection_after: Some(EditSelection::feature(feature)),
        coalesce: None,
    })
}

/// Explicit per-frame context, so every edit entry point stays testable with no
/// egui context, no GPU and no [`crate::OxigisApp`].
pub struct EditCtx<'a> {
    /// The project the target layer belongs to.
    pub project: &'a Project,
    /// The layer being edited, when one is selected.
    pub target: Option<LayerId>,
    /// The target layer's features, when they are loaded.
    pub features: Option<&'a Arc<FeatureCollection>>,
    /// The camera this frame.
    pub view: MapView,
    /// The map panel's rect, in egui points.
    pub rect: egui::Rect,
    /// The context's `pixels_per_point`, for screen ↔ world conversions.
    pub ppp: f32,
}
