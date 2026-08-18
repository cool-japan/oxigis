// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Hit testing: what is under the pointer, and in what order.
//!
//! # The two orderings
//!
//! **Between kinds**, `Point` beats `Line` beats `Polygon`. Ranking by distance
//! alone would make a point inside a polygon unclickable — the polygon reports
//! distance `0` from every interior pixel — so the smaller target wins outright.
//! Within one kind the nearer candidate wins, and a tie goes to the **higher**
//! source index, because that is the one drawn last and therefore on top.
//!
//! **Between targets**, `Vertex` beats `Midpoint` beats `Feature`. Handles are
//! only ever considered for the selected feature, and only when they were
//! actually drawn this frame (`handles_active`): what you can click is always
//! what you can see.
//!
//! # Where the coordinates come from
//!
//! Everything here reads [`crate::local_input::LocalInputState`]'s feature
//! store, never the drawn [`oxigis_render::VectorTile`]. The tile is quantized
//! to [`crate::local_vector::LOCAL_EXTENT`], drops null-geometry features while
//! keeping the surviving indices, and is owned by the render side behind a lock
//! — three good reasons why "hit-test the tile instead, it is already
//! projected" is not an optimization but a correctness bug.
//!
//! Distances are measured in **egui points**, so a tolerance means the same
//! physical size at any `pixels_per_point`: geographic positions are projected
//! with [`oxigis_render::MapView::lon_lat_to_screen`] (physical pixels from the
//! map surface's top-left) and divided by `ppp`.

use super::{EditCtx, VertexRef};
use crate::edit::command::{PathKind, paths};
use crate::local_vector::{GeometryKind, geometry_kind};
use egui::{Pos2, Rect, pos2};
use oxigeo::geojson::types::{FeatureCollection, Geometry, Position};
use oxigis_core::{GeometryFamily, LayerId, LayerStyle, LayerStyleSet};
use oxigis_render::{LonLat, MapView};
use std::collections::HashMap;
use std::sync::Arc;

/// How near the pointer must come to a point feature to pick it, in egui points.
pub const PICK_POINT_PT: f32 = 9.0;
/// How near the pointer must come to a line, or to a polygon's boundary.
pub const PICK_LINE_PT: f32 = 6.0;
/// Radius a vertex handle is drawn at.
pub const HANDLE_DRAW_PT: f32 = 3.5;
/// Radius a vertex handle is *grabbed* at.
///
/// Deliberately larger than [`HANDLE_DRAW_PT`]: the target is always more
/// generous than it looks, which is what makes handle dragging feel reliable
/// rather than fussy.
pub const HANDLE_GRAB_PT: f32 = 12.0;
/// How far a repeat click may wander and still advance the cycle.
pub const CYCLE_SLOP_PT: f32 = 3.0;
/// Most handles drawn — and therefore hit-tested — at once.
pub const HANDLE_BUDGET: usize = 2_000;
/// Shortest on-screen segment that still earns a midpoint ghost.
///
/// On dense geometry the ghosts otherwise merge into a band of noise and steal
/// clicks from the real vertices between them.
pub const MIN_MIDPOINT_SEGMENT_PT: f32 = 24.0;

/// How deep a `GeometryCollection` is followed before the walk gives up.
///
/// GeoJSON permits unbounded nesting; a file is not trusted to be shallow, and
/// an unbounded recursion over hostile data is a stack overflow rather than a
/// slow pick.
const MAX_GEOMETRY_DEPTH: usize = 8;

/// One feature under the pointer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureHit {
    /// Index in the **source** collection — the same number the attribute table
    /// reports.
    pub feature: usize,
    /// Which family the geometry belongs to; the primary sort key.
    pub kind: GeometryKind,
    /// Distance from the pointer, in egui points; `0` inside a polygon.
    pub distance_pt: f32,
    /// True when the pointer is strictly inside a polygon, as opposed to within
    /// tolerance of its boundary.
    pub inside: bool,
}

/// What the pointer is over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitTarget {
    /// A draggable vertex handle of the selected feature.
    Vertex {
        /// The feature the handle belongs to.
        feature: usize,
        /// The addressed vertex.
        at: VertexRef,
    },
    /// The ghost between `at.index - 1` and `at.index`; dragging it inserts a
    /// new position **at** `at`.
    Midpoint {
        /// The feature the ghost belongs to.
        feature: usize,
        /// Where a drag would insert.
        at: VertexRef,
    },
    /// The feature's geometry itself.
    Feature {
        /// The feature under the pointer.
        feature: usize,
    },
}

impl HitTarget {
    /// The feature this target belongs to.
    #[must_use]
    pub fn feature(self) -> usize {
        match self {
            Self::Vertex { feature, .. }
            | Self::Midpoint { feature, .. }
            | Self::Feature { feature } => feature,
        }
    }
}

/// A `pixels_per_point` that can safely divide.
///
/// egui reports a sane value in every real frame, but a headless context or a
/// shell mid-resize can hand over `0` — and a division by it would put every
/// projected vertex at infinity, i.e. silently hit nothing at all.
fn safe_ppp(ppp: f32) -> f32 {
    if ppp.is_finite() && ppp > 0.0 {
        ppp
    } else {
        1.0
    }
}

/// The lon/lat a GeoJSON position addresses, if it has one.
///
/// A position shorter than two elements, or holding a non-finite coordinate, is
/// not projectable: it is skipped rather than picked at an arbitrary place.
#[must_use]
pub fn position_lon_lat(position: &Position) -> Option<LonLat> {
    let mut elements = position.iter();
    match (elements.next(), elements.next()) {
        (Some(&lon), Some(&lat)) if lon.is_finite() && lat.is_finite() => {
            Some(LonLat::new(lon, lat))
        }
        _ => None,
    }
}

/// Projects `position` into panel-local egui points.
#[must_use]
pub fn to_screen(view: MapView, rect_origin: Pos2, ppp: f32, position: LonLat) -> Pos2 {
    let ppp = safe_ppp(ppp);
    let px = view.lon_lat_to_screen(position);
    pos2(rect_origin.x + px[0] / ppp, rect_origin.y + px[1] / ppp)
}

/// Distance from `point` to the segment `a`–`b`, endpoints included.
fn segment_distance(point: Pos2, a: Pos2, b: Pos2) -> f32 {
    let along = b - a;
    let length_sq = along.length_sq();
    if length_sq <= f32::MIN_POSITIVE {
        return (point - a).length();
    }
    let t = ((point - a).dot(along) / length_sq).clamp(0.0, 1.0);
    (point - (a + along * t)).length()
}

/// Even-odd point-in-ring test, in screen space.
///
/// Screen space rather than lon/lat because that is what the renderer draws: a
/// Mercator segment is straight on screen, so an inside/outside answer computed
/// there is the answer the user sees.
fn point_in_ring(point: Pos2, ring: &[Pos2]) -> bool {
    let mut inside = false;
    let mut previous = match ring.last() {
        Some(last) => *last,
        None => return false,
    };
    for current in ring {
        let current = *current;
        if (current.y > point.y) != (previous.y > point.y) {
            let span = previous.y - current.y;
            if span != 0.0 {
                let crossing = current.x + (point.y - current.y) / span * (previous.x - current.x);
                if point.x < crossing {
                    inside = !inside;
                }
            }
        }
        previous = current;
    }
    inside
}

/// Sort rank of a geometry family: smaller targets first.
fn kind_rank(kind: GeometryKind) -> u8 {
    match kind {
        GeometryKind::Point => 0,
        GeometryKind::Line => 1,
        GeometryKind::Polygon => 2,
    }
}

/// How near the pointer must come to one feature's geometry to pick it, per
/// family, in egui points.
///
/// [`Default`] is the three shipped constants — the tolerance every layer used
/// before styles were consulted, and the one a caller with no style to offer
/// still gets. A style can only ever WIDEN a tolerance, never tighten one: a
/// hairline is still grabbable at [`PICK_LINE_PT`], because a target that is
/// hard to see is exactly the one that must not also be hard to hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PickTolerance {
    /// How near a point feature must be picked from.
    pub point: f32,
    /// How near a line must be picked from.
    pub line: f32,
    /// How near a polygon's BOUNDARY must be picked from. Its interior is a
    /// hit at any distance, so this only widens the grabbable edge.
    pub polygon: f32,
}

impl Default for PickTolerance {
    fn default() -> Self {
        Self {
            point: PICK_POINT_PT,
            line: PICK_LINE_PT,
            polygon: PICK_LINE_PT,
        }
    }
}

impl PickTolerance {
    /// The tolerance for one `kind`.
    #[must_use]
    pub fn of(self, kind: GeometryKind) -> f32 {
        match kind {
            GeometryKind::Point => self.point,
            GeometryKind::Line => self.line,
            GeometryKind::Polygon => self.polygon,
        }
    }

    /// The largest tolerance in this set — what the broad phase must reach
    /// with, since a feature culled there is never measured at all.
    #[must_use]
    pub fn widest(self) -> f32 {
        self.point.max(self.line).max(self.polygon)
    }

    /// Widens `self` by what `style` draws for `family`.
    #[must_use]
    pub fn widen(mut self, family: GeometryFamily, style: &LayerStyle) -> Self {
        match (family, style) {
            // The disc a user sees IS the target: a 20 pt marker that only
            // answers a click within 9 pt of its centre is the "what you can
            // click is what you can see" rule broken. Half the stroke joins it,
            // because a stroked marker's outer edge is drawn there.
            (GeometryFamily::Point, LayerStyle::Circle(circle)) => {
                self.point = self
                    .point
                    .max(circle.radius() + circle.stroke_width() * 0.5);
            }
            (GeometryFamily::Line, LayerStyle::Line(line)) => {
                self.line = self.line.max(line.width() * 0.5);
            }
            (GeometryFamily::Polygon, LayerStyle::Line(line)) => {
                self.polygon = self.polygon.max(line.width() * 0.5);
            }
            _ => {}
        }
        self
    }

    /// The tolerance `set` gives a feature carrying `attributes`.
    ///
    /// Resolved through [`LayerStyleSet::style_for`] — **the** rule the map's
    /// mesh partition, the PDF exporter and the legend all read, so a
    /// classified layer whose classes differ in marker size is picked with the
    /// size it is actually drawn at.
    ///
    /// GeoJSON's own property map implements `Attributes`, so this path needs
    /// no adapter; a caller holding an MVT property list instead reaches the
    /// identical answer through
    /// [`crate::local_vector::classify::MvtAttributes`].
    #[must_use]
    pub fn for_attributes<A>(set: &LayerStyleSet, attributes: &A) -> Self
    where
        A: oxigis_core::Attributes + ?Sized,
    {
        Self::for_class(set, set.renderer().class_of(attributes))
    }

    /// [`Self::for_attributes`] with the class already resolved.
    #[must_use]
    pub fn for_class(set: &LayerStyleSet, class: Option<usize>) -> Self {
        let mut tolerance = Self::default();
        for family in GeometryFamily::ALL {
            tolerance = tolerance.widen(family, &set.style_for_class(family, class));
        }
        tolerance
    }

    /// The widest tolerance ANY feature of `set` can earn — the broad phase's
    /// radius, computed once per pick.
    ///
    /// Every class is folded in, not just the one under the pointer: the broad
    /// phase runs before any feature is classified, so a radius fitted to the
    /// fallback would cull the very features a large class exists to make
    /// clickable, and the bug would only show on data big enough that the
    /// bounding boxes stop overlapping the pointer.
    #[must_use]
    pub fn widest_of(set: &LayerStyleSet) -> Self {
        let mut tolerance = Self::for_class(set, None);
        for class in 0..set.class_count() {
            let class = Self::for_class(set, Some(class));
            tolerance.point = tolerance.point.max(class.point);
            tolerance.line = tolerance.line.max(class.line);
            tolerance.polygon = tolerance.polygon.max(class.polygon);
        }
        tolerance
    }
}

/// One pick query, as the measuring functions read it: where the pointer is,
/// how to project a position to meet it, and how near it counts as a hit.
///
/// Grouped rather than passed as six parameters because every one of them is
/// constant for the whole query — bundling them makes that visible and keeps
/// the recursion's signature honest.
#[derive(Debug, Clone, Copy)]
struct Probe {
    /// The camera the geometry is projected with.
    view: MapView,
    /// The map panel's top-left corner, in egui points.
    rect_origin: Pos2,
    /// Where the pointer is, in egui points.
    at_pt: Pos2,
    /// The context's `pixels_per_point`.
    ppp: f32,
    /// How near this feature has to come, per family.
    tolerance: PickTolerance,
}

/// Screen-space measurement of one geometry against the probe's pointer.
///
/// Returns the family, the distance in egui points and whether the pointer fell
/// strictly inside a polygon — or [`None`] when nothing came within tolerance.
fn measure(
    geometry: &Geometry,
    probe: &Probe,
    depth: usize,
    scratch: &mut Vec<Pos2>,
) -> Option<(GeometryKind, f32, bool)> {
    match geometry {
        Geometry::GeometryCollection(collection) => {
            if depth >= MAX_GEOMETRY_DEPTH {
                return None;
            }
            let mut best: Option<(GeometryKind, f32, bool)> = None;
            for member in &collection.geometries {
                let Some(found) = measure(member, probe, depth + 1, scratch) else {
                    continue;
                };
                let better = best.is_none_or(|current| {
                    (kind_rank(found.0), found.1) < (kind_rank(current.0), current.1)
                });
                if better {
                    best = Some(found);
                }
            }
            best
        }
        Geometry::Polygon(polygon) => measure_polygon(&polygon.coordinates, probe, scratch),
        Geometry::MultiPolygon(polygons) => {
            let mut best: Option<(GeometryKind, f32, bool)> = None;
            for rings in &polygons.coordinates {
                let Some(found) = measure_polygon(rings, probe, scratch) else {
                    continue;
                };
                if best.is_none_or(|current| found.1 < current.1) {
                    best = Some(found);
                }
            }
            best
        }
        _ => {
            let kind = geometry_kind(geometry)?;
            let tolerance = probe.tolerance.of(kind);
            let mut best = f32::INFINITY;
            for path in paths(geometry) {
                project_path(
                    path.positions,
                    probe.view,
                    probe.rect_origin,
                    probe.ppp,
                    scratch,
                );
                match path.kind {
                    PathKind::Points => {
                        for screen in scratch.iter() {
                            best = best.min((probe.at_pt - *screen).length());
                        }
                    }
                    PathKind::Line | PathKind::Ring => {
                        for pair in scratch.windows(2) {
                            if let [a, b] = pair {
                                best = best.min(segment_distance(probe.at_pt, *a, *b));
                            }
                        }
                    }
                }
            }
            (best <= tolerance).then_some((kind, best, false))
        }
    }
}

/// [`measure`] for one polygon part: its exterior ring plus its holes.
///
/// The interior of a hole is a **miss** — that is what a hole means — but the
/// hole's boundary is still within tolerance, so a hole can be re-shaped by
/// grabbing its edge.
fn measure_polygon(
    rings: &[Vec<Position>],
    probe: &Probe,
    scratch: &mut Vec<Pos2>,
) -> Option<(GeometryKind, f32, bool)> {
    let mut boundary = f32::INFINITY;
    let mut in_exterior = false;
    let mut in_hole = false;
    for (index, ring) in rings.iter().enumerate() {
        project_path(ring, probe.view, probe.rect_origin, probe.ppp, scratch);
        for pair in scratch.windows(2) {
            if let [a, b] = pair {
                boundary = boundary.min(segment_distance(probe.at_pt, *a, *b));
            }
        }
        if index == 0 {
            in_exterior = point_in_ring(probe.at_pt, scratch);
        } else if point_in_ring(probe.at_pt, scratch) {
            in_hole = true;
        }
    }
    if in_exterior && !in_hole {
        return Some((GeometryKind::Polygon, 0.0, true));
    }
    (boundary <= probe.tolerance.polygon).then_some((GeometryKind::Polygon, boundary, false))
}

/// Projects `positions` into `out`, replacing whatever it held.
fn project_path(
    positions: &[Position],
    view: MapView,
    rect_origin: Pos2,
    ppp: f32,
    out: &mut Vec<Pos2>,
) {
    out.clear();
    out.extend(
        positions
            .iter()
            .filter_map(position_lon_lat)
            .map(|lon_lat| to_screen(view, rect_origin, ppp, lon_lat)),
    );
}

/// Every feature under `at_pt`, best first.
///
/// Ordering is `Point` before `Line` before `Polygon`, then nearer first, then
/// **higher source index** first. Null-geometry features are never candidates:
/// they have nothing to be under the pointer.
///
/// The broad phase visits only the grid cells the pick radius touches
/// (`WorldBboxIndex::candidate_features`) rather than every feature in the
/// collection — the difference between a click on a 1 000 000-feature layer
/// costing a handful of bounding-box tests and costing a full linear scan.
///
/// # The layer's style
///
/// `style` is the selected layer's own [`LayerStyleSet`], when the caller has
/// one. It only ever WIDENS the tolerances — a 20 pt marker is picked at 20 pt
/// rather than at [`PICK_POINT_PT`] — and it is read through
/// [`LayerStyleSet::style_for`], the same rule the map, the PDF exporter and
/// the legend resolve with, so a classified layer whose classes differ in
/// marker size is picked with the size each class is drawn at.
///
/// [`None`] is the pre-thematic behaviour exactly: the three constants, for
/// every feature.
#[must_use]
pub fn pick_features(
    features: &FeatureCollection,
    bbox_index: &WorldBboxIndex,
    view: MapView,
    rect_origin: Pos2,
    at_pt: Pos2,
    ppp: f32,
    style: Option<&LayerStyleSet>,
) -> Vec<FeatureHit> {
    let ppp = safe_ppp(ppp);
    let local = at_pt - rect_origin;
    let at_world = view
        .screen_to_lon_lat([local.x * ppp, local.y * ppp])
        .to_world();
    let at_world = [at_world.x, at_world.y];
    let world_pixels = view.world_pixels();
    // The BROAD phase reaches as far as the widest tolerance any class of this
    // layer can earn. Fitting it to the fallback instead would cull the very
    // features a large class exists to make clickable — and only on data whose
    // bounding boxes stop overlapping the pointer, i.e. never on a fixture.
    let widest = style.map_or_else(PickTolerance::default, PickTolerance::widest_of);
    let radius_world = if world_pixels > 0.0 {
        f64::from(widest.widest() * ppp) / world_pixels
    } else {
        f64::INFINITY
    };

    let mut candidates = Vec::new();
    bbox_index.candidate_features(
        features.features.len(),
        at_world,
        radius_world,
        &mut candidates,
    );

    // An unclassified layer resolves its tolerance ONCE: `style_for` would
    // otherwise compose a style per candidate to answer the same thing every
    // time. A classified one resolves per feature, because that is the point.
    let uniform = match style {
        Some(set) if !set.renderer().is_single() => None,
        Some(set) => Some(PickTolerance::for_class(set, None)),
        None => Some(PickTolerance::default()),
    };

    let mut scratch = Vec::new();
    let mut hits = Vec::new();
    for index in candidates {
        let Some(feature) = features.features.get(index) else {
            continue;
        };
        let Some(geometry) = feature.geometry.as_ref() else {
            continue;
        };
        let tolerance = match (uniform, style) {
            (Some(uniform), _) => uniform,
            (None, Some(set)) => match feature.properties.as_ref() {
                Some(properties) => PickTolerance::for_attributes(set, properties),
                None => PickTolerance::for_class(set, None),
            },
            // Unreachable: `uniform` is `Some` whenever `style` is `None`.
            (None, None) => PickTolerance::default(),
        };
        let probe = Probe {
            view,
            rect_origin,
            at_pt,
            ppp,
            tolerance,
        };
        if let Some((kind, distance_pt, inside)) = measure(geometry, &probe, 0, &mut scratch) {
            hits.push(FeatureHit {
                feature: index,
                kind,
                distance_pt,
                inside,
            });
        }
    }
    hits.sort_by(|a, b| {
        kind_rank(a.kind)
            .cmp(&kind_rank(b.kind))
            .then(a.distance_pt.total_cmp(&b.distance_pt))
            .then(b.feature.cmp(&a.feature))
    });
    hits
}

/// The full pick at `at_pt`: `Vertex` beats `Midpoint` beats `Feature`.
///
/// `selected` names the one feature whose handles are live, and
/// `handles_active` says whether handles were drawn at all this frame — the
/// handle budget and the draw modes both turn them off, and a target that is not
/// drawn must not be clickable.
#[must_use]
pub fn pick(
    ctx: &EditCtx<'_>,
    bbox_index: &WorldBboxIndex,
    selected: Option<usize>,
    at_pt: Pos2,
    handles_active: bool,
) -> Option<HitTarget> {
    let features = ctx.features?;
    let origin = ctx.rect.min;
    if handles_active
        && let Some(index) = selected
        && let Some(geometry) = features
            .features
            .get(index)
            .and_then(|feature| feature.geometry.as_ref())
        && within_handle_budget(geometry, ctx.view, ctx.rect, ctx.ppp)
    {
        let handles = visible_vertex_positions(geometry, ctx.view, ctx.rect, ctx.ppp);
        if let Some(at) = nearest_handle(&handles, ctx, origin, at_pt) {
            return Some(HitTarget::Vertex { feature: index, at });
        }
        let ghosts = visible_midpoint_positions(geometry, ctx.view, ctx.rect, ctx.ppp);
        if let Some(at) = nearest_handle(&ghosts, ctx, origin, at_pt) {
            return Some(HitTarget::Midpoint { feature: index, at });
        }
    }
    pick_features(
        features,
        bbox_index,
        ctx.view,
        origin,
        at_pt,
        ctx.ppp,
        layer_style(ctx),
    )
    .first()
    .map(|hit| HitTarget::Feature {
        feature: hit.feature,
    })
}

/// The style set the target layer draws with, when the project names one.
///
/// Read from `Project::styles` only — the map's own explicit-style slot, and
/// the ONE place a thematic renderer can live (the style panel writes there,
/// and a derived default set is never classified). A layer with no entry picks
/// with the shipped constants, exactly as it always has.
#[must_use]
pub fn layer_style<'a>(ctx: &EditCtx<'a>) -> Option<&'a LayerStyleSet> {
    ctx.project.styles.get(&ctx.target?)
}

/// The nearest of `handles` within [`HANDLE_GRAB_PT`] of `at_pt`.
fn nearest_handle(
    handles: &[(VertexRef, LonLat)],
    ctx: &EditCtx<'_>,
    origin: Pos2,
    at_pt: Pos2,
) -> Option<VertexRef> {
    let mut best: Option<(f32, VertexRef)> = None;
    for (at, position) in handles {
        let distance = (at_pt - to_screen(ctx.view, origin, ctx.ppp, *position)).length();
        if distance <= HANDLE_GRAB_PT && best.is_none_or(|(current, _)| distance < current) {
            best = Some((distance, *at));
        }
    }
    best.map(|(_, at)| at)
}

/// Every draggable handle of `geometry`, in `(part, ring, index)` order.
///
/// A closed ring yields `n` handles for its `n + 1` stored positions: the
/// duplicate closing position is not addressable, so dragging handle `0` moves
/// both ends of the ring at once and closure survives by construction.
/// A `GeometryCollection` yields none — its vertices are read-only in v1.
#[must_use]
pub fn vertex_positions(geometry: &Geometry) -> Vec<(VertexRef, LonLat)> {
    let mut handles = Vec::new();
    for path in paths(geometry) {
        for (index, position) in path.positions.iter().enumerate() {
            if let Some(lon_lat) = position_lon_lat(position) {
                handles.push((VertexRef::at(path.part, path.ring, index), lon_lat));
            }
        }
    }
    handles
}

/// The midpoint ghost of every segment, addressed by the vertex a drag would
/// insert **at** — so the ghost between vertices `i - 1` and `i` carries index
/// `i`, and a ring's wrap segment carries the append index.
///
/// Segments shorter than [`MIN_MIDPOINT_SEGMENT_PT`] on screen are omitted, so
/// the ghosts thin out as geometry gets dense instead of burying the real
/// vertices.
#[must_use]
pub fn midpoint_positions(
    geometry: &Geometry,
    view: MapView,
    ppp: f32,
) -> Vec<(VertexRef, LonLat)> {
    let ppp = safe_ppp(ppp);
    let mut ghosts = Vec::new();
    for path in paths(geometry) {
        // Separate points have no segments between them to bisect.
        if path.kind == PathKind::Points {
            continue;
        }
        let count = path.positions.len();
        if count < 2 {
            continue;
        }
        let closed = path.kind == PathKind::Ring;
        let segments = if closed { count } else { count - 1 };
        for segment in 0..segments {
            let next = (segment + 1) % count;
            let (Some(a), Some(b)) = (
                path.positions.get(segment).and_then(position_lon_lat),
                path.positions.get(next).and_then(position_lon_lat),
            ) else {
                continue;
            };
            let start = view.lon_lat_to_screen(a);
            let end = view.lon_lat_to_screen(b);
            let length_pt =
                ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt() / ppp;
            let long_enough = length_pt.is_finite() && length_pt >= MIN_MIDPOINT_SEGMENT_PT;
            if !long_enough {
                continue;
            }
            let middle =
                view.screen_to_lon_lat([(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5]);
            ghosts.push((VertexRef::at(path.part, path.ring, segment + 1), middle));
        }
    }
    ghosts
}

/// The region a handle must fall in to be worth drawing.
///
/// Exactly `rect` grown by [`HANDLE_GRAB_PT`], which is precisely the set of
/// positions a pointer *inside* `rect` can come within grab range of. Culling
/// for painting and culling for clicking are therefore the same operation, and
/// "what you can click is always what you can see" holds by construction rather
/// than by two functions agreeing.
#[must_use]
pub fn handle_bounds(rect: Rect) -> Rect {
    rect.expand(HANDLE_GRAB_PT)
}

/// [`vertex_positions`], culled to what a click inside `rect` could reach.
///
/// Culling before the budget is what keeps a 200 000-vertex coastline editable:
/// zoomed in on one bay, the handles in view are a few dozen, and suppressing
/// them because the *whole* feature is dense would make the only geometry that
/// really needs editing the only geometry that cannot be edited.
///
/// Streams the projection and the cull together rather than materializing
/// every vertex via [`vertex_positions`] first: a coastline that is only
/// culled *after* [`vertex_positions`] has built its full, unculled 200 000
/// entry `Vec` has already paid the memory and the projection twice over —
/// once to build it, once more inside a separate filter — for a result that
/// keeps a few dozen of them.
#[must_use]
pub fn visible_vertex_positions(
    geometry: &Geometry,
    view: MapView,
    rect: Rect,
    ppp: f32,
) -> Vec<(VertexRef, LonLat)> {
    let bounds = handle_bounds(rect);
    let origin = rect.min;
    let mut out = Vec::new();
    for path in paths(geometry) {
        for (index, position) in path.positions.iter().enumerate() {
            let Some(lon_lat) = position_lon_lat(position) else {
                continue;
            };
            if bounds.contains(to_screen(view, origin, ppp, lon_lat)) {
                out.push((VertexRef::at(path.part, path.ring, index), lon_lat));
            }
        }
    }
    out
}

/// [`midpoint_positions`], culled the same way.
///
/// Unlike [`visible_vertex_positions`], culling here also skips work rather
/// than merely reordering it: [`midpoint_positions`] returns each ghost's
/// position as `LonLat`, which needs an inverse Mercator projection
/// (`MapView::screen_to_lon_lat`) to derive from the segment's screen-space
/// midpoint — a projection worth paying for only once a ghost is known to
/// survive culling. So the bounds check runs on the screen-space midpoint
/// directly (already in hand from the segment-length test below it), and the
/// inverse projection runs only for ghosts that pass it, instead of for every
/// ghost the geometry has and then again, forward, for every ghost that
/// happened to be visible.
#[must_use]
pub fn visible_midpoint_positions(
    geometry: &Geometry,
    view: MapView,
    rect: Rect,
    ppp: f32,
) -> Vec<(VertexRef, LonLat)> {
    let ppp = safe_ppp(ppp);
    let bounds = handle_bounds(rect);
    let origin = rect.min;
    let mut ghosts = Vec::new();
    for path in paths(geometry) {
        // Separate points have no segments between them to bisect.
        if path.kind == PathKind::Points {
            continue;
        }
        let count = path.positions.len();
        if count < 2 {
            continue;
        }
        let closed = path.kind == PathKind::Ring;
        let segments = if closed { count } else { count - 1 };
        for segment in 0..segments {
            let next = (segment + 1) % count;
            let (Some(a), Some(b)) = (
                path.positions.get(segment).and_then(position_lon_lat),
                path.positions.get(next).and_then(position_lon_lat),
            ) else {
                continue;
            };
            let start = view.lon_lat_to_screen(a);
            let end = view.lon_lat_to_screen(b);
            let length_pt =
                ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt() / ppp;
            let long_enough = length_pt.is_finite() && length_pt >= MIN_MIDPOINT_SEGMENT_PT;
            if !long_enough {
                continue;
            }
            let mid_px = [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5];
            let screen = pos2(origin.x + mid_px[0] / ppp, origin.y + mid_px[1] / ppp);
            if !bounds.contains(screen) {
                continue;
            }
            let middle = view.screen_to_lon_lat(mid_px);
            ghosts.push((VertexRef::at(path.part, path.ring, segment + 1), middle));
        }
    }
    ghosts
}

/// How many of `geometry`'s handles are in view — the number the budget is
/// measured against, and the number the hint plate reports.
#[must_use]
pub fn visible_handle_count(geometry: &Geometry, view: MapView, rect: Rect, ppp: f32) -> usize {
    let bounds = handle_bounds(rect);
    let origin = rect.min;
    let mut count = 0;
    for path in paths(geometry) {
        for position in path.positions {
            if let Some(lon_lat) = position_lon_lat(position)
                && bounds.contains(to_screen(view, origin, ppp, lon_lat))
            {
                count += 1;
            }
        }
    }
    count
}

/// Whether `geometry` has few enough handles in view to draw — and therefore to
/// grab.
///
/// One verdict governs handles *and* midpoint ghosts: past the budget the
/// selection outline stays and everything draggable goes, because half a set of
/// targets is worse than none.
///
/// Exits as soon as the count passes [`HANDLE_BUDGET`] rather than finishing
/// the walk like [`visible_handle_count`] must to report an exact number: a
/// feature with millions of in-view vertices is exactly the hostile case a
/// budget gate exists to answer quickly, not the case that should force it to
/// keep counting past the point the answer is already "no".
#[must_use]
pub fn within_handle_budget(geometry: &Geometry, view: MapView, rect: Rect, ppp: f32) -> bool {
    let bounds = handle_bounds(rect);
    let origin = rect.min;
    let mut count = 0usize;
    for path in paths(geometry) {
        for position in path.positions {
            let Some(lon_lat) = position_lon_lat(position) else {
                continue;
            };
            if bounds.contains(to_screen(view, origin, ppp, lon_lat)) {
                count += 1;
                if count > HANDLE_BUDGET {
                    return false;
                }
            }
        }
    }
    true
}

/// Where the handle `at` sits: a vertex handle's own position, or a segment
/// ghost's midpoint.
///
/// Read back through the same two functions that produced the handle rather
/// than recomputed, so a drag starts exactly where the thing the user grabbed
/// was drawn.
#[must_use]
pub fn handle_position(
    geometry: &Geometry,
    at: VertexRef,
    midpoint: bool,
    view: MapView,
    ppp: f32,
) -> Option<LonLat> {
    let handles = if midpoint {
        midpoint_positions(geometry, view, ppp)
    } else {
        vertex_positions(geometry)
    };
    handles
        .into_iter()
        .find_map(|(reference, position)| (reference == at).then_some(position))
}

/// A world-space bounding box, in normalised `0..1` Web Mercator coordinates.
type WorldBbox = [f64; 4];

/// Width of one [`WorldBboxIndex`] grid cell, in normalised `0..1` world
/// units — the same order of magnitude as the snap index's own cell (see
/// `snap.rs`), chosen independently because that constant is private to its
/// module: fine enough to discriminate a layer of many small, localised
/// features (points, small polygons) — the case this grid exists for — while
/// [`MAX_QUERY_CELLS`] catches the opposite case (a viewport wide enough that
/// a query would touch an unreasonable number of cells) before it ever walks
/// them.
const BBOX_CELL_WORLD: f64 = 1.0 / 4096.0;

/// A single feature's box spanning more cells than this at index-build time
/// goes to the oversized fallback instead of being filed cell by cell — a
/// feature the size of a country must not blow up the grid's memory just
/// because most features in the same layer are the size of a building.
const MAX_BBOX_CELLS: i64 = 64;

/// A broad-phase query spanning more cells than this at pick time is not
/// walked cell by cell at all; it degrades to every index the grid covers,
/// the same "test everything" the index has always fallen back to. Bounds
/// the cost of a hostile or degenerate query (`radius_world` of
/// [`f64::INFINITY`] when [`MapView::world_pixels`] has collapsed to zero, or
/// simply a viewport zoomed out to the whole world) at a fixed multiple of
/// what a direct scan already costs, rather than letting it walk millions of
/// empty cells.
const MAX_QUERY_CELLS: i64 = 1024;

/// Per-feature world bounding boxes: the broad phase every click and every
/// hover frame runs first.
///
/// Rebuilt only when the collection's `Arc` actually changes, and it **holds**
/// that `Arc` rather than keying on its address — holding it makes the address
/// impossible to reuse, so staleness detection cannot suffer ABA against a
/// freed-and-reallocated collection. World space, not screen space, for the same
/// reason the snap index will be: a camera move must not invalidate it.
///
/// Beyond the flat per-feature boxes ([`Self::may_contain`]'s domain), the
/// index also files every box into a uniform grid so
/// `Self::candidate_features` can answer "which features are near here"
/// without visiting every feature in the collection — see
/// [`pick_features`].
#[derive(Debug, Default)]
pub struct WorldBboxIndex {
    /// Which layer the boxes belong to.
    layer: Option<LayerId>,
    /// The exact collection indexed, held for the lifetime of the index.
    source: Option<Arc<FeatureCollection>>,
    /// One entry per feature; [`None`] for a feature with no geometry.
    boxes: Vec<Option<WorldBbox>>,
    /// Grid cell to the feature indices whose box overlaps it.
    cells: HashMap<[i32; 2], Vec<u32>>,
    /// Boxes too large to file cell by cell ([`MAX_BBOX_CELLS`]); visited by
    /// every grid query that reaches the grid at all.
    oversized: Vec<u32>,
}

impl WorldBboxIndex {
    /// Rebuilds the boxes unless they already describe exactly this `Arc`.
    pub fn rebuild_if_stale(&mut self, layer: LayerId, features: &Arc<FeatureCollection>) {
        if self.layer == Some(layer)
            && self
                .source
                .as_ref()
                .is_some_and(|held| Arc::ptr_eq(held, features))
        {
            return;
        }
        self.boxes.clear();
        self.boxes.reserve(features.features.len());
        self.cells.clear();
        self.oversized.clear();
        for (index, feature) in features.features.iter().enumerate() {
            let bbox = feature
                .geometry
                .as_ref()
                .and_then(|geometry| world_bbox(geometry, 0));
            self.boxes.push(bbox);
            if let Some(bbox) = bbox {
                self.file_bbox(index as u32, bbox);
            }
        }
        self.layer = Some(layer);
        self.source = Some(Arc::clone(features));
    }

    /// Files `index`'s box into every grid cell it overlaps, or into
    /// [`Self::oversized`] when that would be more than [`MAX_BBOX_CELLS`].
    fn file_bbox(&mut self, index: u32, bbox: WorldBbox) {
        let Some((first_x, last_x, first_y, last_y)) = cell_span(bbox, MAX_BBOX_CELLS) else {
            self.oversized.push(index);
            return;
        };
        for x in first_x..=last_x {
            for y in first_y..=last_y {
                self.cells.entry([x, y]).or_default().push(index);
            }
        }
    }

    /// Drops everything, releasing the held collection.
    pub fn clear(&mut self) {
        self.layer = None;
        self.source = None;
        self.boxes.clear();
        self.cells.clear();
        self.oversized.clear();
    }

    /// The layer currently indexed, if any.
    #[must_use]
    pub fn layer(&self) -> Option<LayerId> {
        self.layer
    }

    /// How many features are indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.boxes.len()
    }

    /// Whether nothing is indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }

    /// Whether feature `index` can possibly be within `radius_world` of
    /// `at_world`.
    ///
    /// A feature the index has no entry for is **never** culled — an index that
    /// is empty or shorter than the collection must degrade into "test
    /// everything", never into "hit nothing". A feature with no geometry is
    /// always culled, which is where the null-geometry rule is enforced.
    #[must_use]
    pub fn may_contain(&self, index: usize, at_world: [f64; 2], radius_world: f64) -> bool {
        match self.boxes.get(index) {
            Some(Some(bbox)) => {
                at_world[0] >= bbox[0] - radius_world
                    && at_world[0] <= bbox[2] + radius_world
                    && at_world[1] >= bbox[1] - radius_world
                    && at_world[1] <= bbox[3] + radius_world
            }
            Some(None) => false,
            None => true,
        }
    }

    /// Fills `out` with every feature index [`pick_features`] must actually
    /// measure for a query at `at_world` within `radius_world`, against a
    /// collection of `features_len` features — deduplicated, in no particular
    /// order.
    ///
    /// Degrades to pushing a whole range of indices, rather than visiting the
    /// grid, in exactly the cases [`Self::may_contain`] already degrades for:
    /// indices beyond what the grid covers (`features_len` larger than what
    /// was last indexed) always go in, because an index that cannot speak for
    /// a feature must never silently exclude it. The query itself degrades
    /// the same way when it is not finite or would touch more than
    /// [`MAX_QUERY_CELLS`] — see that constant's docs — by pushing every
    /// index the grid *does* cover instead of walking cells for it.
    fn candidate_features(
        &self,
        features_len: usize,
        at_world: [f64; 2],
        radius_world: f64,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        let covered = self.boxes.len().min(features_len);
        if covered > 0 {
            let query = [
                at_world[0] - radius_world,
                at_world[1] - radius_world,
                at_world[0] + radius_world,
                at_world[1] + radius_world,
            ];
            match cell_span(query, MAX_QUERY_CELLS) {
                Some((first_x, last_x, first_y, last_y)) => {
                    for x in first_x..=last_x {
                        for y in first_y..=last_y {
                            let Some(hits) = self.cells.get(&[x, y]) else {
                                continue;
                            };
                            out.extend(
                                hits.iter()
                                    .map(|&index| index as usize)
                                    .filter(|&index| index < covered),
                            );
                        }
                    }
                    out.extend(
                        self.oversized
                            .iter()
                            .map(|&index| index as usize)
                            .filter(|&index| index < covered),
                    );
                    out.sort_unstable();
                    out.dedup();
                }
                None => {
                    // A non-finite query, or one broad enough that walking it
                    // cell by cell could cost more than it saves, degrades to
                    // every index the grid covers — the grid is a shortcut,
                    // never a filter of last resort.
                    out.extend(0..covered);
                }
            }
        }
        // Beyond what the grid covers, every index must degrade to "test
        // it", never to "skip it" — disjoint from the range above by
        // construction, so no dedup is needed across the two.
        out.extend(covered..features_len);
    }
}

/// The inclusive grid-cell range a `[min_x, min_y, max_x, max_y]` box
/// touches, or [`None`] when the range cannot be answered cell by cell: a
/// non-finite bound, an inverted range, or a span so large that visiting it
/// would touch more than `max_cells`.
fn cell_span(bounds: WorldBbox, max_cells: i64) -> Option<(i32, i32, i32, i32)> {
    if bounds.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let first_x = cell_of(bounds[0]);
    let last_x = cell_of(bounds[2]);
    let first_y = cell_of(bounds[1]);
    let last_y = cell_of(bounds[3]);
    let span_x = i64::from(last_x) - i64::from(first_x) + 1;
    let span_y = i64::from(last_y) - i64::from(first_y) + 1;
    if span_x <= 0 || span_y <= 0 || span_x.saturating_mul(span_y) > max_cells {
        return None;
    }
    Some((first_x, last_x, first_y, last_y))
}

/// Which [`BBOX_CELL_WORLD`]-wide grid cell `world` falls in, along one axis.
///
/// Saturates rather than panics on a non-finite or extreme input — Rust's
/// float-to-int cast has been saturating, not UB, since 1.45 — though every
/// caller here already filters non-finite bounds via [`cell_span`] first;
/// this is the same "degrade gracefully, never panic" margin the rest of the
/// module keeps against hostile input.
fn cell_of(world: f64) -> i32 {
    (world / BBOX_CELL_WORLD).floor() as i32
}

/// The world-space bounding box of `geometry`, or [`None`] when it holds no
/// projectable position.
fn world_bbox(geometry: &Geometry, depth: usize) -> Option<WorldBbox> {
    let mut bbox: Option<WorldBbox> = None;
    if let Geometry::GeometryCollection(collection) = geometry {
        if depth >= MAX_GEOMETRY_DEPTH {
            return None;
        }
        for member in &collection.geometries {
            if let Some(inner) = world_bbox(member, depth + 1) {
                bbox = Some(match bbox {
                    Some(current) => union_bbox(current, inner),
                    None => inner,
                });
            }
        }
        return bbox;
    }
    for path in paths(geometry) {
        for position in path.positions {
            if let Some(lon_lat) = position_lon_lat(position) {
                let world = lon_lat.to_world();
                let corner = [world.x, world.y, world.x, world.y];
                bbox = Some(match bbox {
                    Some(current) => union_bbox(current, corner),
                    None => corner,
                });
            }
        }
    }
    bbox
}

/// The smallest box holding both.
fn union_bbox(a: WorldBbox, b: WorldBbox) -> WorldBbox {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

/// Repeat-click cycling through the features stacked under one spot.
///
/// The cycle advances only when the click lands within [`CYCLE_SLOP_PT`] of the
/// previous one **and** resolves to the identical candidate list; anything else
/// restarts at the best candidate. Requiring the list to match is what keeps a
/// click that happens to land near the last one — but over different geometry —
/// from silently selecting the wrong feature.
#[derive(Debug, Default)]
pub struct PickCycle {
    /// Where the previous click landed.
    at: Option<Pos2>,
    /// The candidate features that click resolved to, best first.
    candidates: Vec<usize>,
    /// Which candidate the previous click landed on.
    cursor: usize,
}

impl PickCycle {
    /// Advances the cycle for a click at `at_pt` over `candidates`, and returns
    /// the feature it lands on.
    pub fn next(&mut self, at_pt: Pos2, candidates: &[FeatureHit]) -> Option<usize> {
        if candidates.is_empty() {
            self.clear();
            return None;
        }
        let ids: Vec<usize> = candidates.iter().map(|hit| hit.feature).collect();
        let repeat = self
            .at
            .is_some_and(|previous| (at_pt - previous).length() <= CYCLE_SLOP_PT)
            && self.candidates == ids;
        self.cursor = if repeat {
            (self.cursor + 1) % ids.len()
        } else {
            0
        };
        let picked = ids.get(self.cursor).copied();
        self.at = Some(at_pt);
        self.candidates = ids;
        picked
    }

    /// Forgets the previous click, so the next one starts a fresh cycle.
    pub fn clear(&mut self) {
        self.at = None;
        self.candidates.clear();
        self.cursor = 0;
    }

    /// The current position in the cycle as `(one-based position, total)`, or
    /// [`None`] when no click has been resolved yet.
    ///
    /// This is what the status line reports as `feature 2 of 4 here` — the only
    /// way a user ever learns that cycling exists.
    #[must_use]
    pub fn position(&self) -> Option<(usize, usize)> {
        (!self.candidates.is_empty()).then(|| (self.cursor + 1, self.candidates.len()))
    }
}
