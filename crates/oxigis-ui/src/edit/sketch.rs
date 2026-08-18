// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Digitizing: the in-progress [`Sketch`], the rules that turn one into a
//! [`Geometry`], and the transaction that adds it to a layer.
//!
//! # What a sketch is, and what it is not
//!
//! A sketch is **pure interaction state**. Appending a vertex allocates nothing
//! beyond one [`LonLat`], queues no GPU work, re-quantizes nothing and pushes
//! nothing onto the undo stack: the rubber band is painted from these points and
//! the feature store is not touched until the sketch is *finished*. One drawn
//! feature is therefore exactly one undo step, however many clicks it took.
//!
//! # The two traps this module exists to close
//!
//! **A double-click's first press-release also reports `clicked()`.** egui
//! reports `clicked()` on *both* releases of a double click (`double_clicked_by`
//! is `CLICKED && button_double_clicked`), so a naive "append on click, finish on
//! double click" gets a duplicate final vertex on every double-click-finished
//! line and polygon — which then trips the repeated-vertex check as well. Two
//! independent defences: the caller handles the double click *first* and never
//! appends on that frame, and [`Sketch::finish_from_double_click`] drops a final
//! vertex that sits within [`DOUBLE_CLICK_DEDUPE_PT`] of its predecessor.
//!
//! **Ring closure is explicit in the stored model.** [`Sketch::finish`] appends
//! a clone of position `0` for a polygon, because the GeoJSON written to disk
//! must be closed — even though [`crate::edit::VertexRef`] addresses the *open*
//! sequence and the duplicate final position is never addressable. Every ring
//! this app produces is closed from the moment it is born.
//!
//! # Why the mode stays latched after a finish
//!
//! Drawing several features in a row is the common case, so finishing a sketch
//! clears the vertices and leaves the tool selected. The toolbar keeps the tool
//! visibly latched, which is what makes that state legible rather than
//! surprising.

use super::command::{EditTransaction, FeatureOp};
use super::hit::to_screen;
use super::{EditMode, EditSelection};
use egui::{Pos2, pos2};
use oxigeo::geojson::types::{Feature, Geometry, LineString, Point, Polygon, Position, Properties};
use oxigis_core::LayerId;
use oxigis_render::{LonLat, MapView};

/// How near, in egui points, a double click's second vertex has to sit to its
/// predecessor before it is treated as the same click and dropped.
///
/// Four points is roughly egui's own click radius: a genuine second vertex that
/// close would be a zero-length segment nobody wants, and the topology checks
/// would report it as a repeated vertex the moment the feature was committed.
pub const DOUBLE_CLICK_DEDUPE_PT: f32 = 4.0;

/// How many vertices a [`EditMode::DrawLine`] sketch needs before it is a line.
pub const MIN_LINE_VERTICES: usize = 2;

/// How many vertices a [`EditMode::DrawPolygon`] sketch needs before it is a
/// ring — counted on the **open** sequence, so the closed ring this yields holds
/// four positions.
pub const MIN_POLYGON_VERTICES: usize = 3;

/// The geometry currently being digitized, if any.
///
/// See the module docs: this is interaction state only, and nothing here has
/// touched the layer's data.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sketch {
    /// Which draw tool started this sketch; [`None`] when nothing is in
    /// progress.
    pub mode: Option<EditMode>,
    /// The vertices committed so far, in click order. A polygon's closing
    /// duplicate is **not** here: it is added by [`Self::finish`].
    pub points: Vec<LonLat>,
    /// Snapped-or-raw pointer position, for the rubber band's free end.
    pub cursor: Option<LonLat>,
}

impl Sketch {
    /// Whether anything has been digitized yet.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.mode.is_some() || !self.points.is_empty()
    }

    /// How many vertices have been placed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether no vertex has been placed yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The first vertex — the one a ring closes on to.
    #[must_use]
    pub fn first(&self) -> Option<LonLat> {
        self.points.first().copied()
    }

    /// The most recent vertex — the rubber band's fixed end.
    #[must_use]
    pub fn last(&self) -> Option<LonLat> {
        self.points.last().copied()
    }

    /// Appends `at` to the sketch.
    ///
    /// A sketch started by a *different* tool is discarded rather than extended:
    /// the vertices belong to the tool that was collecting them, and carrying
    /// them into another one produces geometry the user never asked for.
    pub fn append(&mut self, mode: EditMode, at: LonLat) {
        if self.mode != Some(mode) {
            self.points.clear();
            self.mode = Some(mode);
        }
        self.points.push(at);
        self.cursor = Some(at);
    }

    /// Drops the most recent vertex, returning it — `Backspace`.
    ///
    /// Emptying the sketch also forgets which tool owned it, so
    /// [`Self::is_active`] answers honestly and the next `Escape` climbs down to
    /// the next rung of the ladder rather than cancelling nothing.
    pub fn pop(&mut self) -> Option<LonLat> {
        let dropped = self.points.pop();
        if self.points.is_empty() {
            self.mode = None;
        }
        dropped
    }

    /// Discards the whole sketch — `Escape`, a tool change, a layer change.
    pub fn cancel(&mut self) {
        *self = Self::default();
    }

    /// Whether `pointer_pt` is within `tolerance_pt` of the sketch's first
    /// vertex, i.e. whether a click there closes the ring.
    ///
    /// Measured in screen points and deliberately independent of whether
    /// snapping is enabled: closing a ring is a gesture the user makes, not a
    /// coincidence the editor notices, and turning snapping off must not take
    /// the gesture away.
    #[must_use]
    pub fn closes_at(
        &self,
        view: MapView,
        rect_origin: Pos2,
        pointer_pt: Pos2,
        ppp: f32,
        tolerance_pt: f32,
    ) -> bool {
        let Some(first) = self.first() else {
            return false;
        };
        if !tolerance_pt.is_finite() || tolerance_pt <= 0.0 {
            return false;
        }
        let at = to_screen(view, rect_origin, ppp, first);
        (at - pointer_pt).length() <= tolerance_pt
    }

    /// Turns the sketch into a geometry, clearing it.
    ///
    /// [`None`] when there are too few vertices for `mode` — and the sketch is
    /// then left **exactly as it was**, so a premature `Enter` costs the user
    /// nothing but a status line.
    pub fn finish(&mut self, mode: EditMode) -> Option<Geometry> {
        let geometry = geometry_from(mode, &self.points)?;
        self.cancel();
        Some(geometry)
    }

    /// [`Self::finish`], with the double-click dedupe applied first.
    ///
    /// See the module docs: the last vertex is dropped when it sits within
    /// [`DOUBLE_CLICK_DEDUPE_PT`] of its predecessor on screen, because that is
    /// the signature of a double click whose first release already appended a
    /// vertex.
    pub fn finish_from_double_click(
        &mut self,
        mode: EditMode,
        view: MapView,
        ppp: f32,
    ) -> Option<Geometry> {
        self.drop_coincident_last(view, ppp);
        self.finish(mode)
    }

    /// Pops the last vertex when it is within [`DOUBLE_CLICK_DEDUPE_PT`] of the
    /// one before it, measured on screen.
    fn drop_coincident_last(&mut self, view: MapView, ppp: f32) {
        let Some(previous_index) = self.points.len().checked_sub(2) else {
            return;
        };
        let (Some(last), Some(previous)) = (
            self.points.last().copied(),
            self.points.get(previous_index).copied(),
        ) else {
            return;
        };
        let origin = pos2(0.0, 0.0);
        let a = to_screen(view, origin, ppp, last);
        let b = to_screen(view, origin, ppp, previous);
        if (a - b).length() <= DOUBLE_CLICK_DEDUPE_PT {
            self.points.pop();
        }
    }
}

/// The geometry `points` would make under `mode`, without consuming them.
///
/// A polygon's ring is closed here — a clone of position `0` is appended — so
/// every ring this app produces is closed from birth. Digitizing produces
/// single-part geometry only.
///
/// A non-finite coordinate yields [`None`]: `serde_json` writes one as `null`,
/// which would make the layer's stored text unreadable rather than merely wrong
/// — the same rule [`crate::edit::command::write_lon_lat`] enforces for a moved
/// vertex.
#[must_use]
pub fn geometry_from(mode: EditMode, points: &[LonLat]) -> Option<Geometry> {
    if !points.iter().copied().all(is_finite) {
        return None;
    }
    match mode {
        EditMode::DrawPoint => points.first().copied().and_then(point_geometry),
        EditMode::DrawLine => {
            if points.len() < MIN_LINE_VERTICES {
                return None;
            }
            LineString::new(points.iter().copied().map(position_of).collect())
                .ok()
                .map(Geometry::LineString)
        }
        EditMode::DrawPolygon => {
            if points.len() < MIN_POLYGON_VERTICES {
                return None;
            }
            let mut ring: Vec<Position> = points.iter().copied().map(position_of).collect();
            let first = ring.first()?.clone();
            ring.push(first);
            Polygon::from_exterior(ring).ok().map(Geometry::Polygon)
        }
        EditMode::Off | EditMode::Select => None,
    }
}

/// One `Point` geometry at `at`, or [`None`] for a non-finite coordinate.
#[must_use]
pub fn point_geometry(at: LonLat) -> Option<Geometry> {
    if !is_finite(at) {
        return None;
    }
    Point::new(position_of(at)).ok().map(Geometry::Point)
}

/// Whether a position can be written to GeoJSON at all.
fn is_finite(at: LonLat) -> bool {
    at.lon.is_finite() && at.lat.is_finite()
}

/// A stored position for `at`. Two elements: a digitized vertex has no altitude
/// to carry, and inventing one would be a guess written into the user's data.
#[must_use]
pub fn position_of(at: LonLat) -> Position {
    vec![at.lon, at.lat]
}

/// The undo label a finished sketch carries.
#[must_use]
pub fn draw_label(mode: EditMode) -> &'static str {
    match mode {
        EditMode::DrawPoint => "Draw point",
        EditMode::DrawLine => "Draw line",
        EditMode::DrawPolygon => "Draw polygon",
        EditMode::Off | EditMode::Select => "Draw feature",
    }
}

/// What to say when a sketch was finished before it was a geometry.
///
/// A refusal, never a silent no-op: the user pressed `Enter` and something has
/// to answer.
#[must_use]
pub fn too_few_message(mode: EditMode) -> String {
    match mode {
        EditMode::DrawLine => {
            format!(
                "A line needs at least {MIN_LINE_VERTICES} vertices — click the map to add one."
            )
        }
        EditMode::DrawPolygon => format!(
            "A polygon needs at least {MIN_POLYGON_VERTICES} vertices — click the map to add one."
        ),
        EditMode::DrawPoint | EditMode::Off | EditMode::Select => {
            "Click the map to place a point.".to_string()
        }
    }
}

/// The transaction one finished sketch asks for.
///
/// `index` is the collection's current length, so the new feature is appended
/// and every existing index keeps its meaning — which is what makes
/// `Add`/`Remove` exact inverses under strict LIFO. The new feature is selected,
/// so `Delete`, the attribute form and the vertex handles all address what was
/// just drawn.
///
/// Split out as a free function for the same reason as
/// [`crate::edit::drag_transaction`]: the whole commit shape is then reachable
/// from a test with no egui frame, no GPU and no simulated pointer.
#[must_use]
pub fn add_feature_transaction(
    layer: LayerId,
    index: usize,
    geometry: Geometry,
    label: &'static str,
    selection_before: Option<EditSelection>,
) -> EditTransaction {
    EditTransaction {
        layer,
        label,
        ops: vec![FeatureOp::Add {
            index,
            // An empty property map rather than `None`: a null-properties
            // feature reads as a blank row in the attribute table and gives the
            // attribute form nothing to add a key to.
            feature: Box::new(Feature::new(Some(geometry), Some(Properties::new()))),
        }],
        selection_before,
        selection_after: Some(EditSelection::feature(index)),
        // One drawn feature is one undo step; nothing folds into it.
        coalesce: None,
    }
}
