// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Snapping: a uniform grid over normalised world space, and the query that
//! turns a pointer position into the exact coordinate the user meant.
//!
//! # Why world space and not screen space
//!
//! The index is built over [`oxigis_render::WorldCoord`] (`0..1` Web Mercator),
//! never over pixels, and it is rebuilt **only** when a source collection's
//! `Arc` actually changes. A screen-space index would have to be rebuilt on
//! every frame of every pan, every zoom step and every frame of inertia — and
//! editing is exactly when the user pans most. Panning here costs nothing at
//! all: the camera is a query parameter.
//!
//! # Why the index holds its `Arc`s
//!
//! Staleness is decided by [`Arc::ptr_eq`] against the collections the index
//! was built from, and those collections are **held** for as long as the index
//! lives. Holding them is what makes the comparison sound: a pointer-keyed
//! scheme (`*const`, or a hash of addresses) can be defeated by a collection
//! being dropped and a new one landing on the same allocation — the ABA hazard —
//! and the failure mode is snapping to coordinates that no longer exist.
//!
//! # What is bit-exact and what is not
//!
//! A [`SnapKind::Vertex`] result carries the **stored** [`oxigis_render::LonLat`]
//! of the vertex, copied straight out of the collection: never a screen
//! round-trip, never a value that has been through the renderer's
//! quantization. Snapping two features together and moving them again therefore
//! accumulates exactly zero drift. A [`SnapKind::Edge`] result is interpolated
//! along the segment **in world space**, which is the space the renderer draws
//! straight lines in, so the snapped point lies on the line the user sees.
//!
//! # Budget
//!
//! Above [`SNAP_MAX_SEGMENTS`] the index falls back to the active edit layer
//! alone and reports [`SnapIndex::is_degraded`], which the hint plate shows.
//! Snapping never goes silently partial: a user who cannot see why a reference
//! layer stopped attracting the pointer has no way to find out.

use super::VertexRef;
use crate::edit::command::{PathKind, paths};
use crate::edit::hit::{position_lon_lat, to_screen};
use egui::Pos2;
use oxigeo::geojson::types::{FeatureCollection, Geometry};
use oxigis_core::LayerId;
use oxigis_render::{LonLat, MapView, WorldCoord};
use std::collections::HashMap;
use std::sync::Arc;

/// How near the pointer must come, in egui points, before snapping fires.
pub const SNAP_TOLERANCE_PT: f32 = 12.0;
/// Most segments indexed before the index degrades to the active layer alone.
pub const SNAP_MAX_SEGMENTS: usize = 400_000;
/// Edge length of one grid cell, in normalised world coordinates.
///
/// `1/4096` is one tile's width at zoom 12: fine enough that a query at any
/// realistic editing zoom touches a handful of cells, coarse enough that a
/// city-block-sized segment lands in one or two of them.
const SNAP_CELL_WORLD: f64 = 1.0 / 4096.0;
/// Most cells one segment is filed under before it goes on the always-scanned
/// list instead.
///
/// A segment spanning a continent covers thousands of cells at this resolution,
/// and filing it in every one of them turns the index build into an O(cells)
/// blow-up on exactly the data — coarse world outlines — that is most likely to
/// be loaded as a reference layer. The design does not name this bound; without
/// it the grid is not robust against real data.
const MAX_SEGMENT_CELLS: usize = 64;

/// What a snap latched on to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapKind {
    /// An existing vertex: the stored position, bit for bit.
    Vertex,
    /// A point along a segment, interpolated in world space.
    Edge,
    /// The in-progress sketch's own first vertex — closing a ring.
    SketchStart,
}

/// One snap candidate that won.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapResult {
    /// What was snapped to.
    pub kind: SnapKind,
    /// Which layer holds it.
    pub layer: LayerId,
    /// Which feature of that layer, by source index.
    pub feature: usize,
    /// The exact handle, for [`SnapKind::Vertex`].
    pub vertex: Option<VertexRef>,
    /// The position to adopt. See the module docs on what is bit-exact.
    pub position: LonLat,
    /// Where that position lands on screen, in egui points — where the marker
    /// is drawn.
    pub screen_pt: Pos2,
    /// How far the pointer was from it, in egui points.
    pub distance_pt: f32,
}

/// Which snaps are live, and how close is close enough.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapSettings {
    /// Whether snapping runs at all.
    pub enabled: bool,
    /// Whether existing vertices attract the pointer.
    pub to_vertices: bool,
    /// Whether segments do.
    pub to_edges: bool,
    /// The catch radius, in egui points.
    pub tolerance_pt: f32,
}

impl Default for SnapSettings {
    /// Everything on at [`SNAP_TOLERANCE_PT`].
    ///
    /// On by default because the whole point of an editor snapping is that
    /// coincident geometry ends up actually coincident; a user who wants the
    /// raw pointer holds `Ctrl`, which is a per-gesture decision rather than a
    /// mode to remember.
    fn default() -> Self {
        Self {
            enabled: true,
            to_vertices: true,
            to_edges: true,
            tolerance_pt: SNAP_TOLERANCE_PT,
        }
    }
}

/// One indexed segment, in normalised world coordinates.
///
/// A `Point` contributes a zero-length segment (`from == to`), so one structure
/// serves points, lines and rings and the query has no per-kind branch at all.
#[derive(Debug, Clone, Copy)]
struct IndexedSegment {
    /// Index into [`SnapIndex::sources`].
    source: u32,
    /// Feature index inside that source collection.
    feature: u32,
    /// `Multi*` member.
    part: u32,
    /// Polygon ring.
    ring: u32,
    /// Index of the start vertex in the **open** sequence.
    from: u32,
    /// Index of the end vertex; equal to `from` for a point, and `0` on a
    /// ring's closing segment.
    to: u32,
    /// Start, in world coordinates.
    a: [f64; 2],
    /// End, in world coordinates.
    b: [f64; 2],
    /// Start, as stored.
    a_pos: LonLat,
    /// End, as stored.
    b_pos: LonLat,
}

impl IndexedSegment {
    /// The two endpoints as `(vertex index, world, stored position)`.
    fn endpoints(&self) -> [(u32, [f64; 2], LonLat); 2] {
        [
            (self.from, self.a, self.a_pos),
            (self.to, self.b, self.b_pos),
        ]
    }

    /// The address of the vertex at `index` of this segment's path.
    fn vertex_ref(&self, index: u32) -> VertexRef {
        VertexRef::at(self.part as usize, self.ring as usize, index as usize)
    }
}

/// A uniform grid over normalised world coordinates.
///
/// See the module docs for why world space, why the `Arc`s are held, and what
/// the budget does.
#[derive(Debug)]
pub struct SnapIndex {
    /// The exact collections indexed, held for the index's lifetime.
    sources: Vec<(LayerId, Arc<FeatureCollection>)>,
    /// Cell coordinate to the segments overlapping it.
    cells: HashMap<[i32; 2], Vec<u32>>,
    /// Segments too long to file in cells; scanned by every query.
    oversized: Vec<u32>,
    /// Every indexed segment.
    segments: Vec<IndexedSegment>,
    /// Whether the budget forced a fall back to the active layer alone.
    degraded: bool,
    /// Bumped on every real rebuild, so a test can prove a no-op was a no-op.
    generation: u64,
    /// The segment budget; a field rather than a constant so both sides of it
    /// are reachable in one test run, exactly as the undo stack's budgets are.
    max_segments: usize,
}

impl Default for SnapIndex {
    fn default() -> Self {
        Self::with_budget(SNAP_MAX_SEGMENTS)
    }
}

impl SnapIndex {
    /// An empty index with a custom segment budget.
    #[must_use]
    pub fn with_budget(max_segments: usize) -> Self {
        Self {
            sources: Vec::new(),
            cells: HashMap::new(),
            oversized: Vec::new(),
            segments: Vec::new(),
            degraded: false,
            generation: 0,
            max_segments,
        }
    }

    /// Rebuilds from `sources` unless they are, layer for layer, exactly what is
    /// already indexed.
    ///
    /// The first entry is the active edit layer: it is the one kept when the
    /// budget forces a degraded build.
    pub fn rebuild_if_stale(&mut self, sources: &[(LayerId, Arc<FeatureCollection>)]) {
        if self.matches(sources) {
            return;
        }
        self.rebuild(sources);
    }

    /// Whether `sources` is `Arc`-identical, layer for layer and in order, to
    /// what is indexed.
    fn matches(&self, sources: &[(LayerId, Arc<FeatureCollection>)]) -> bool {
        self.sources.len() == sources.len()
            && self
                .sources
                .iter()
                .zip(sources)
                .all(|((held_id, held), (id, features))| {
                    held_id == id && Arc::ptr_eq(held, features)
                })
    }

    /// Discards everything and indexes `sources` from scratch.
    fn rebuild(&mut self, sources: &[(LayerId, Arc<FeatureCollection>)]) {
        self.clear();
        self.generation = self.generation.wrapping_add(1);
        self.sources = sources.to_vec();

        // Count first, so the degraded decision is made before any work is
        // wasted rather than half way through it.
        let total: usize = sources
            .iter()
            .map(|(_, features)| count_segments(features))
            .sum();
        let keep = if total > self.max_segments {
            self.degraded = true;
            1.min(sources.len())
        } else {
            sources.len()
        };
        // One allocation for the whole rebuild instead of the repeated
        // doubling `Vec::push` would otherwise do — `total` is already
        // computed, and the budget is the hard ceiling either way.
        self.segments.reserve(total.min(self.max_segments));

        for (source, (_, features)) in sources.iter().enumerate().take(keep) {
            self.index_collection(source as u32, features);
        }
        self.file_segments();
    }

    /// Walks one collection into [`Self::segments`], stopping the moment the
    /// budget is spent — even the active layer alone can exceed it, and
    /// [`Self::index_geometry`] is what makes that a hard ceiling rather
    /// than a per-feature checkpoint a single oversized feature can sail
    /// past.
    fn index_collection(&mut self, source: u32, features: &FeatureCollection) {
        for (index, feature) in features.features.iter().enumerate() {
            let Some(geometry) = feature.geometry.as_ref() else {
                continue;
            };
            if !self.index_geometry(source, index as u32, geometry) {
                // `degraded` is already set: `index_geometry` is the one
                // place that discovers the budget ran out, mid-feature
                // included.
                return;
            }
        }
    }

    /// Walks one geometry's paths into [`Self::segments`], checking the
    /// budget before every push rather than only between features — a
    /// single oversized path (a dissolved coastline, a merged parcel layer,
    /// ordinary reference data at scale) would otherwise blow straight
    /// through [`Self::max_segments`] within one call.
    ///
    /// Returns whether budget remains; the caller stops walking the
    /// collection the moment this answers `false`.
    #[must_use]
    fn index_geometry(&mut self, source: u32, feature: u32, geometry: &Geometry) -> bool {
        let max_segments = self.max_segments;
        for path in paths(geometry) {
            let projected: Vec<Option<(LonLat, [f64; 2])>> = path
                .positions
                .iter()
                .map(|position| {
                    position_lon_lat(position).map(|lon_lat| {
                        let world = lon_lat.to_world();
                        (lon_lat, [world.x, world.y])
                    })
                })
                .collect();
            let count = projected.len();
            let push = |segments: &mut Vec<IndexedSegment>, from: usize, to: usize| -> bool {
                if segments.len() >= max_segments {
                    return false;
                }
                if let (Some(Some((a_pos, a))), Some(Some((b_pos, b)))) =
                    (projected.get(from), projected.get(to))
                {
                    segments.push(IndexedSegment {
                        source,
                        feature,
                        part: path.part as u32,
                        ring: path.ring as u32,
                        from: from as u32,
                        to: to as u32,
                        a: *a,
                        b: *b,
                        a_pos: *a_pos,
                        b_pos: *b_pos,
                    });
                }
                true
            };
            let in_budget = match path.kind {
                // A point is its own zero-length segment, so a vertex snap to a
                // point needs no special case anywhere in the query.
                PathKind::Points => (0..count).all(|index| push(&mut self.segments, index, index)),
                PathKind::Line => {
                    (1..count).all(|index| push(&mut self.segments, index - 1, index))
                }
                PathKind::Ring => {
                    (1..count).all(|index| push(&mut self.segments, index - 1, index))
                        && (count < 2 || push(&mut self.segments, count - 1, 0))
                }
            };
            if !in_budget {
                self.degraded = true;
                return false;
            }
        }
        true
    }

    /// Files every segment into the cells its bounding box covers.
    fn file_segments(&mut self) {
        for (index, segment) in self.segments.iter().enumerate() {
            let index = index as u32;
            let (min_x, max_x) = ordered(segment.a[0], segment.b[0]);
            let (min_y, max_y) = ordered(segment.a[1], segment.b[1]);
            let (first_x, last_x) = (cell_of(min_x), cell_of(max_x));
            let (first_y, last_y) = (cell_of(min_y), cell_of(max_y));
            let span_x = i64::from(last_x - first_x) + 1;
            let span_y = i64::from(last_y - first_y) + 1;
            if span_x.saturating_mul(span_y) > MAX_SEGMENT_CELLS as i64 {
                self.oversized.push(index);
                continue;
            }
            for x in first_x..=last_x {
                for y in first_y..=last_y {
                    self.cells.entry([x, y]).or_default().push(index);
                }
            }
        }
    }

    /// The nearest snap to `pointer_pt`, or [`None`] when nothing is in range.
    ///
    /// `exclude` names the vertex being dragged: it, and the two segments
    /// adjacent to it, are skipped — otherwise a dragged vertex snaps to itself
    /// and can never be moved at all.
    ///
    /// The catch radius is converted to world space once
    /// (`tolerance_pt * ppp / world_pixels`), and ranking is done on that same
    /// world distance. Screen scaling is uniform in both axes, so the world
    /// ordering *is* the screen ordering, and the reported `distance_pt` is the
    /// honest screen distance rather than an approximation of it.
    ///
    /// A vertex within tolerance beats an edge unconditionally, however much
    /// nearer the edge is: a vertex is a thing the user placed, an edge is
    /// merely the space between two of them.
    #[must_use]
    pub fn query(
        &self,
        view: MapView,
        rect_origin: Pos2,
        pointer_pt: Pos2,
        settings: SnapSettings,
        exclude: Option<(LayerId, usize, VertexRef)>,
        ppp: f32,
    ) -> Option<SnapResult> {
        // The single-vertex case is the one-element slice of the set case, so
        // the two predicates cannot drift.
        match exclude {
            Some((layer, feature, at)) => self.query_excluding_set(
                view,
                rect_origin,
                pointer_pt,
                settings,
                Some((layer, feature, core::slice::from_ref(&at))),
                ppp,
            ),
            None => self.query_excluding_set(view, rect_origin, pointer_pt, settings, None, ppp),
        }
    }

    /// [`Self::query`] for a rigid vertex-set translation: every listed
    /// vertex — and every segment incident to one of them — moves with the
    /// pointer, so all of it would otherwise attract the grabbed vertex back
    /// to where the set started. `vertices` must be sorted ascending (it
    /// comes from `FeatureSelection::vertex_set`, which sorts), so membership
    /// is a binary search.
    #[must_use]
    pub fn query_excluding_set(
        &self,
        view: MapView,
        rect_origin: Pos2,
        pointer_pt: Pos2,
        settings: SnapSettings,
        exclude: Option<(LayerId, usize, &[VertexRef])>,
        ppp: f32,
    ) -> Option<SnapResult> {
        if !settings.enabled || !(settings.to_vertices || settings.to_edges) {
            return None;
        }
        let scale = Scale::new(view, rect_origin, ppp, settings.tolerance_pt)?;
        let at = scale.pointer_world(pointer_pt);
        let radius = scale.radius_world;
        let radius_sq = radius * radius;

        let mut best_vertex: Option<(f64, SnapResult)> = None;
        let mut best_edge: Option<(f64, SnapResult)> = None;
        self.for_each_candidate(at, radius, |segment| {
            let layer = match self.sources.get(segment.source as usize) {
                Some((layer, _)) => *layer,
                None => return,
            };
            let excluded_here = exclude.filter(|(id, _, _)| *id == layer);

            if settings.to_vertices {
                for (index, world, position) in segment.endpoints() {
                    let reference = segment.vertex_ref(index);
                    if excluded_here.is_some_and(|(_, feature, vertices)| {
                        segment.feature as usize == feature
                            && vertices.binary_search(&reference).is_ok()
                    }) {
                        continue;
                    }
                    let distance_sq = distance_sq(at, world);
                    if distance_sq > radius_sq {
                        continue;
                    }
                    if best_vertex.is_none_or(|(current, _)| distance_sq < current) {
                        best_vertex = Some((
                            distance_sq,
                            SnapResult {
                                kind: SnapKind::Vertex,
                                layer,
                                feature: segment.feature as usize,
                                vertex: Some(reference),
                                position,
                                screen_pt: scale.screen_of(position),
                                distance_pt: scale.points_of(distance_sq.sqrt()),
                            },
                        ));
                    }
                }
            }

            if settings.to_edges && segment.a != segment.b {
                // Whether this segment belongs to an excluded path AND
                // touches it: the two segments a moved vertex is an
                // endpoint of, plus the one an *insert* would split — the
                // ring's closing segment, which runs from the final index
                // back to `0`, so an insert at the append index (one past
                // the final one) splits exactly it and no plain index
                // comparison can see that. Evaluated by binary search
                // against the sorted set instead of a linear scan:
                // `vertices` is guaranteed ascending (see this function's
                // doc), so membership is `O(log n)`, which matters here
                // because `vertices` is the whole moving set of a marquee
                // drag — up to `HANDLE_BUDGET` entries, re-tested every
                // candidate segment of every frame.
                if excluded_here.is_some_and(|(_, feature, vertices)| {
                    segment.feature as usize == feature
                        && (vertices
                            .binary_search(&segment.vertex_ref(segment.from))
                            .is_ok()
                            || vertices
                                .binary_search(&segment.vertex_ref(segment.to))
                                .is_ok()
                            || (segment.to < segment.from
                                && vertices
                                    .binary_search(&VertexRef::at(
                                        segment.part as usize,
                                        segment.ring as usize,
                                        segment.from as usize + 1,
                                    ))
                                    .is_ok()))
                }) {
                    return;
                }
                let (foot, distance_sq) = nearest_on_segment(at, segment.a, segment.b);
                if distance_sq > radius_sq {
                    return;
                }
                if best_edge.is_none_or(|(current, _)| distance_sq < current) {
                    let position = WorldCoord::new(foot[0], foot[1]).to_lon_lat();
                    best_edge = Some((
                        distance_sq,
                        SnapResult {
                            kind: SnapKind::Edge,
                            layer,
                            feature: segment.feature as usize,
                            vertex: None,
                            position,
                            screen_pt: scale.screen_of(position),
                            distance_pt: scale.points_of(distance_sq.sqrt()),
                        },
                    ));
                }
            }
        });

        best_vertex.or(best_edge).map(|(_, result)| result)
    }

    /// Calls `visit` for every segment that could be within `radius` of `at`.
    ///
    /// Walks the grid when the cell range is smaller than the segment list, and
    /// scans linearly when it is not — zoomed far enough out the catch radius
    /// covers a large fraction of the world, and iterating a quarter of a
    /// million empty cells to find nothing would be slower than looking at every
    /// segment once.
    fn for_each_candidate(
        &self,
        at: [f64; 2],
        radius: f64,
        mut visit: impl FnMut(&IndexedSegment),
    ) {
        let (first_x, last_x) = (cell_of(at[0] - radius), cell_of(at[0] + radius));
        let (first_y, last_y) = (cell_of(at[1] - radius), cell_of(at[1] + radius));
        let span_x = i64::from(last_x - first_x) + 1;
        let span_y = i64::from(last_y - first_y) + 1;
        if span_x.saturating_mul(span_y) >= self.segments.len() as i64 {
            for segment in &self.segments {
                visit(segment);
            }
            return;
        }
        for index in &self.oversized {
            if let Some(segment) = self.segments.get(*index as usize) {
                visit(segment);
            }
        }
        for x in first_x..=last_x {
            for y in first_y..=last_y {
                let Some(bucket) = self.cells.get(&[x, y]) else {
                    continue;
                };
                for index in bucket {
                    if let Some(segment) = self.segments.get(*index as usize) {
                        visit(segment);
                    }
                }
            }
        }
    }

    /// How many segments are indexed.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Whether the budget forced a fall back to the active layer alone.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.degraded
    }

    /// How many times the index has actually been rebuilt.
    ///
    /// Exists so "a camera change rebuilds nothing" is a *provable* claim rather
    /// than one inferred from a segment count that would not have moved anyway.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// How many layers are indexed.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Drops everything, releasing the held collections.
    pub fn clear(&mut self) {
        self.sources.clear();
        self.cells.clear();
        self.oversized.clear();
        self.segments.clear();
        self.degraded = false;
    }
}

/// The sketch's own first vertex, as a snap candidate.
///
/// Checked separately from the index because an in-progress sketch is not in any
/// collection yet — and deliberately its own [`SnapKind`], because closing a
/// ring is a decision the user makes, not a coincidence the editor notices. The
/// digitizing tools call this before [`SnapIndex::query`] and prefer its answer.
#[must_use]
pub fn snap_to_sketch_start(
    view: MapView,
    rect_origin: Pos2,
    pointer_pt: Pos2,
    settings: SnapSettings,
    layer: LayerId,
    first: LonLat,
    ppp: f32,
) -> Option<SnapResult> {
    if !settings.enabled {
        return None;
    }
    let scale = Scale::new(view, rect_origin, ppp, settings.tolerance_pt)?;
    let at = scale.pointer_world(pointer_pt);
    let world = first.to_world();
    let distance_sq = distance_sq(at, [world.x, world.y]);
    if distance_sq > scale.radius_world * scale.radius_world {
        return None;
    }
    Some(SnapResult {
        kind: SnapKind::SketchStart,
        layer,
        // A sketch has no feature index yet; the vertex it closes on to is
        // always its own first one.
        feature: 0,
        vertex: Some(VertexRef::new(0)),
        position: first,
        screen_pt: scale.screen_of(first),
        distance_pt: scale.points_of(distance_sq.sqrt()),
    })
}

/// The camera-derived constants a query needs, computed once.
#[derive(Debug, Clone, Copy)]
struct Scale {
    /// The view being queried against.
    view: MapView,
    /// The map panel's top-left corner, in egui points.
    origin: Pos2,
    /// A `pixels_per_point` that can safely divide.
    ppp: f32,
    /// How wide the whole world is, in physical pixels.
    world_pixels: f64,
    /// The catch radius, in world units.
    radius_world: f64,
    /// The viewport's north-west corner, in world units.
    north_west: WorldCoord,
}

impl Scale {
    /// [`None`] when the camera or the tolerance cannot produce a finite
    /// radius — a degenerate viewport must snap to nothing, never to everything.
    fn new(view: MapView, origin: Pos2, ppp: f32, tolerance_pt: f32) -> Option<Self> {
        let ppp = if ppp.is_finite() && ppp > 0.0 {
            ppp
        } else {
            1.0
        };
        let world_pixels = view.world_pixels();
        if !world_pixels.is_finite() || world_pixels <= 0.0 {
            return None;
        }
        if !tolerance_pt.is_finite() || tolerance_pt <= 0.0 {
            return None;
        }
        let (north_west, _) = view.world_bounds();
        Some(Self {
            view,
            origin,
            ppp,
            world_pixels,
            radius_world: f64::from(tolerance_pt * ppp) / world_pixels,
            north_west,
        })
    }

    /// Where `pointer_pt` lands in world space.
    ///
    /// Derived from the viewport's own corner rather than through
    /// [`MapView::screen_to_lon_lat`]: that path unprojects to degrees and
    /// clamps to the world square, and re-projecting the result would cost two
    /// transcendentals and lose the exactness the comparison below relies on.
    fn pointer_world(&self, pointer_pt: Pos2) -> [f64; 2] {
        let local = pointer_pt - self.origin;
        [
            self.north_west.x + f64::from(local.x * self.ppp) / self.world_pixels,
            self.north_west.y + f64::from(local.y * self.ppp) / self.world_pixels,
        ]
    }

    /// Where a geographic position lands on screen, in egui points.
    fn screen_of(self, position: LonLat) -> Pos2 {
        to_screen(self.view, self.origin, self.ppp, position)
    }

    /// A world distance, in egui points.
    fn points_of(self, distance_world: f64) -> f32 {
        (distance_world * self.world_pixels / f64::from(self.ppp)) as f32
    }
}

/// How many segments a collection would contribute.
fn count_segments(features: &FeatureCollection) -> usize {
    let mut total = 0;
    for feature in &features.features {
        let Some(geometry) = feature.geometry.as_ref() else {
            continue;
        };
        for path in paths(geometry) {
            let count = path.positions.len();
            total += match path.kind {
                PathKind::Points => count,
                PathKind::Line => count.saturating_sub(1),
                PathKind::Ring => {
                    if count >= 2 {
                        count
                    } else {
                        0
                    }
                }
            };
        }
    }
    total
}

/// The cell index a world coordinate falls in.
fn cell_of(world: f64) -> i32 {
    let cell = (world / SNAP_CELL_WORLD).floor();
    if cell.is_finite() {
        cell.clamp(f64::from(i32::MIN / 2), f64::from(i32::MAX / 2)) as i32
    } else {
        0
    }
}

/// `(min, max)` of two values, non-finite treated as `0`.
fn ordered(a: f64, b: f64) -> (f64, f64) {
    let a = if a.is_finite() { a } else { 0.0 };
    let b = if b.is_finite() { b } else { 0.0 };
    if a <= b { (a, b) } else { (b, a) }
}

/// Squared distance between two world positions.
fn distance_sq(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// The point on segment `a`–`b` nearest `point`, and its squared distance.
///
/// The parameter is clamped to `0..1`, so a pointer past an endpoint measures to
/// the endpoint rather than to the infinite line — without which every long
/// segment would attract the pointer from far beyond where it is drawn.
fn nearest_on_segment(point: [f64; 2], a: [f64; 2], b: [f64; 2]) -> ([f64; 2], f64) {
    let along = [b[0] - a[0], b[1] - a[1]];
    let length_sq = along[0] * along[0] + along[1] * along[1];
    if length_sq <= 0.0 {
        return (a, distance_sq(point, a));
    }
    let t =
        (((point[0] - a[0]) * along[0] + (point[1] - a[1]) * along[1]) / length_sq).clamp(0.0, 1.0);
    let foot = [a[0] + along[0] * t, a[1] + along[1] * t];
    (foot, distance_sq(point, foot))
}
