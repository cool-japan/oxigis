// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Everything the edit system paints over the map, with `egui`'s own painter.
//!
//! # Why the painter and not the GPU
//!
//! There is no per-feature render knob anywhere in the GPU path, and inventing
//! one would mean touching `oxigis-render`, both shells and the resident-mesh
//! key — for an outline the painter draws for free, on top of the map, with no
//! re-tessellation and no frame of latency. The overlay is therefore pure
//! screen-space work: nothing here allocates an `Arc`, queues a
//! [`crate::local_input::LocalLayerOp`] or touches the feature store.
//!
//! Paint order, later on top: GPU map → selection outline → drag ghost →
//! midpoint ghosts → vertex handles → snap marker → hint plate → attribution →
//! drop hint. The rubber band slots in beside the drag ghost when the digitizing
//! tools land.
//!
//! # What a drag draws instead of an outline
//!
//! While a vertex gesture is live the ordinary selection outline is **not**
//! drawn. It would be painted from the pre-edit geometry in exactly the same
//! accent as the preview, and the two would sit on top of each other with the
//! preview invisible against its own history. A drag draws the pre-edit geometry
//! at [`DRAG_GHOST_ALPHA`] instead — the GPU still shows its old tessellation
//! underneath, which this design names rather than fights — and the previewed
//! geometry at full strength on top of it.
//!
//! # Why the polygon tint is scanline-filled
//!
//! `epaint` fills a closed path with a triangle fan around vertex `0`
//! (`tessellator.rs`, `fill_closed_path`), which is correct only for a convex
//! ring and cannot represent a hole at all. A GIS polygon is routinely neither,
//! and a highlight that leaks across a concavity is worse than no highlight. The
//! even-odd scanline fill this module uses instead gets holes and concavities
//! right by construction. It builds an active-edge table once per part —
//! `O(V log V)` to sort — and then sweeps rows against only the edges
//! currently spanning them, so realistic geometry (a coastline's separate
//! bays and headlands, say) costs far less than a naive per-row walk of every
//! edge; a pathological shape whose every edge spans the full height is still
//! `O(rows × V)`, the bound the naive walk always had.

use super::hit::{HANDLE_DRAW_PT, position_lon_lat, to_screen};
use super::snap::{SnapKind, SnapResult};
use super::{EditMode, EditSelection, Handles, Sketch, VertexDrag, VertexRef};
use crate::edit::command::{PathKind, paths};
use egui::{Color32, FontId, Mesh, Painter, Pos2, Rect, Shape, Stroke, Vec2, pos2, vec2};
use oxigeo::geojson::types::{Geometry, Position};
use oxigis_render::{LonLat, MapView};

/// The edit system's accent colour: a warm amber that reads over both the
/// basemap's greys and the default layer palette's blues and greens.
pub const ACCENT: Color32 = Color32::from_rgb(0xFF, 0xB3, 0x2B);
/// The dark halo drawn under the accent, so the outline survives a light
/// basemap.
pub const HALO: Color32 = Color32::from_black_alpha(0xC8);
/// Width of the halo pass, in egui points.
pub const SELECTION_HALO_PT: f32 = 4.0;
/// Width of the accent pass, in egui points.
pub const SELECTION_STROKE_PT: f32 = 2.0;
/// Alpha of the polygon interior tint, out of 255 — the design's 12 %.
pub const SELECTION_FILL_ALPHA: u8 = 31;
/// Radius of the marker drawn around a selected point feature.
pub const POINT_MARKER_PT: f32 = 6.0;
/// Height of one scanline row of the polygon tint, in egui points.
const FILL_ROW_PT: f32 = 2.0;
/// How deep a `GeometryCollection` is followed before painting gives up.
const MAX_GEOMETRY_DEPTH: usize = 8;

/// Fill of an unpicked vertex handle: near-white, so it reads as a control
/// rather than as more geometry.
pub const HANDLE_FILL: Color32 = Color32::from_rgb(0xFF, 0xF6, 0xE6);
/// Width of the dark outline around a handle, in egui points.
pub const HANDLE_OUTLINE_PT: f32 = 1.0;
/// Half-edge of a midpoint ghost — smaller than [`HANDLE_DRAW_PT`], because a
/// ghost is an invitation and a handle is a thing.
pub const GHOST_DRAW_PT: f32 = 2.5;
/// Alpha of a midpoint ghost, out of 255.
pub const GHOST_ALPHA: u8 = 0x9A;
/// Alpha of the pre-edit geometry drawn under a live drag — the design's 20 %.
pub const DRAG_GHOST_ALPHA: u8 = 51;
/// Alpha of the rubber band's *pending* segments — the ones that follow the
/// pointer and are not vertices yet.
///
/// Fainter than a committed segment but far stronger than the drag ghost: it is
/// a live promise about the next click, not a record of what was.
pub const RUBBER_BAND_ALPHA: u8 = 0xB4;
/// Half-edge of the snap marker's square, in egui points.
pub const SNAP_MARKER_PT: f32 = 5.0;
/// Stroke width of the snap marker.
pub const SNAP_STROKE_PT: f32 = 1.5;

/// What the hint plate says when snapping fell back to the edited layer alone.
pub const SNAP_DEGRADED_HINT: &str =
    "snapping to this layer only \u{2014} too many segments in the others";

/// Font size of the hint plate.
pub const HINT_FONT_PT: f32 = 11.0;
/// Padding between the hint text and its plate.
pub const HINT_PAD_PT: f32 = 4.0;
/// Gap between the hint plate and the map panel's bottom-left corner.
pub const HINT_MARGIN_PT: f32 = 6.0;

/// Screen-space buffers kept across frames.
///
/// Painting a selected 10 000-vertex polygon allocates a `Vec<Pos2>` per ring
/// otherwise, on every frame, for as long as it stays selected. Nothing here is
/// state: it is all overwritten before it is read.
#[derive(Debug, Default)]
pub struct OverlayScratch {
    /// Rings of the polygon part being painted, reused across parts and frames.
    rings: Vec<Vec<Pos2>>,
    /// How many entries of [`Self::rings`] the current part is using.
    used: usize,
    /// Scanline crossings, sorted, for the interior tint.
    crossings: Vec<f32>,
    /// The interior tint's active-edge table: every non-horizontal segment of
    /// the part being filled, sorted by [`FillEdge::y_min`] once so
    /// [`fill_rings`] can admit them with a pointer that only moves forward
    /// instead of rescanning every edge on every row.
    fill_edges: Vec<FillEdge>,
    /// Indices into [`Self::fill_edges`] currently spanning the scanline.
    active_edges: Vec<usize>,
}

/// One polygon edge prepared for the interior tint's scanline sweep.
///
/// `x` at any `y` inside `[y_min, y_max)` is one multiply-add
/// (`x_at_min + (y - y_min) * dx_dy`) rather than a fresh division — the
/// benefit of building this table once per part instead of re-deriving it
/// from raw points on every row.
#[derive(Debug, Clone, Copy)]
struct FillEdge {
    /// The lower `y` of the segment's two endpoints.
    y_min: f32,
    /// The higher `y` of the segment's two endpoints.
    y_max: f32,
    /// `x` at `y_min`.
    x_at_min: f32,
    /// Change in `x` per unit `y`, from `y_min` toward `y_max`.
    dx_dy: f32,
}

impl OverlayScratch {
    /// Begins a fresh part, keeping the allocations of the previous one.
    fn restart(&mut self) {
        self.used = 0;
    }

    /// Hands over the next ring buffer, emptied.
    fn next_ring(&mut self) -> &mut Vec<Pos2> {
        if self.used == self.rings.len() {
            self.rings.push(Vec::new());
        }
        let index = self.used;
        self.used += 1;
        let ring = &mut self.rings[index];
        ring.clear();
        ring
    }

    /// The rings gathered for the current part.
    fn part(&self) -> &[Vec<Pos2>] {
        self.rings.get(..self.used).unwrap_or(&[])
    }
}

/// The hint plate's text for this frame, or [`None`] when nothing should be
/// drawn.
///
/// [`EditMode::Off`] deliberately yields [`None`]: with editing off the map must
/// look exactly as it did before this module existed, and a plate is something
/// the user did not ask for.
#[must_use]
pub fn hint_text(
    mode: EditMode,
    selection: Option<EditSelection>,
    cycle: Option<(usize, usize)>,
    sketch: &Sketch,
) -> Option<String> {
    match mode {
        EditMode::Off => None,
        EditMode::Select => Some(match (selection, cycle) {
            (Some(_), Some((position, total))) if total > 1 => {
                format!("Select — feature {position} of {total} here; click again to cycle")
            }
            (Some(selection), _) => {
                format!("Select — feature {} picked", selection.feature)
            }
            (None, _) => "Select — click a feature to pick it".to_string(),
        }),
        EditMode::DrawPoint => Some("Point — click the map to place a point".to_string()),
        EditMode::DrawLine | EditMode::DrawPolygon => Some(sketch_hint(mode, sketch)),
    }
}

/// The hint a digitizing tool shows, which is a function of how far the sketch
/// has got.
///
/// The empty sketch names the gesture that starts one; a sketch in progress
/// names its size and **every** way out of it, because `Backspace` and `Escape`
/// are exactly the keys a user needs when a click went wrong and exactly the
/// ones they have no way to guess.
fn sketch_hint(mode: EditMode, sketch: &Sketch) -> String {
    let count = sketch.len();
    if count == 0 {
        return match mode {
            EditMode::DrawPolygon => {
                "Polygon — click to add vertices, Enter or double-click to close".to_string()
            }
            _ => "Line — click to add vertices, Enter or double-click to finish".to_string(),
        };
    }
    let vertices = if count == 1 {
        "1 vertex".to_string()
    } else {
        format!("{count} vertices")
    };
    let close = if mode == EditMode::DrawPolygon && count >= 3 {
        ", or click the first vertex to close"
    } else {
        ""
    };
    format!(
        "{} — {vertices} \u{2014} Enter/double-click to finish{close}, Backspace to undo vertex, \
         Esc to cancel",
        mode.label()
    )
}

/// The plate line the handle budget forces, when it forces one.
///
/// A suppressed set of handles has to *say* it is suppressed. Silently drawing
/// only the outline on a dense feature reads as "this feature cannot be edited",
/// which is both wrong and unactionable; naming the count and the remedy turns
/// it into a one-gesture problem.
#[must_use]
pub fn handle_hint(handles: Handles) -> Option<String> {
    handles
        .suppressed_count()
        .map(|count| format!("{count} vertices in view \u{2014} zoom in to edit them"))
}

/// Joins the plate's lines into the one string it draws.
///
/// Composed rather than chosen so a budget warning and a degraded-snap warning
/// can both be true at once, which they routinely are on a big reference layer.
#[must_use]
pub fn plate_text(base: Option<String>, extra: &[Option<String>]) -> Option<String> {
    let mut lines: Vec<String> = base.into_iter().collect();
    lines.extend(extra.iter().flatten().cloned());
    (!lines.is_empty()).then(|| lines.join(" \u{b7} "))
}

/// Draws the hint plate in the map's bottom-left corner.
///
/// Bottom-**left** because the bottom-right belongs to the basemap attribution,
/// which is a licence condition rather than a design choice and must never be
/// covered.
pub fn paint_hint(painter: &Painter, rect: Rect, text: &str) {
    if text.is_empty() {
        return;
    }
    let galley = painter.layout_no_wrap(
        text.to_string(),
        FontId::proportional(HINT_FONT_PT),
        Color32::from_rgb(0xF0, 0xE4, 0xC8),
    );
    let plate_size = galley.size() + vec2(HINT_PAD_PT * 2.0, HINT_PAD_PT);
    let origin = pos2(
        rect.min.x + HINT_MARGIN_PT,
        rect.max.y - HINT_MARGIN_PT - plate_size.y,
    );
    let plate = Rect::from_min_size(origin, plate_size);
    painter.rect_filled(plate, 3.0, Color32::from_black_alpha(0xB0));
    painter.galley(
        plate.min + vec2(HINT_PAD_PT, HINT_PAD_PT * 0.5),
        galley,
        Color32::PLACEHOLDER,
    );
}

/// Draws the selected feature's outline: a dark halo, an accent stroke on top,
/// and a faint accent tint inside any polygon.
pub fn paint_selection(
    painter: &Painter,
    view: MapView,
    rect: Rect,
    ppp: f32,
    geometry: &Geometry,
    scratch: &mut OverlayScratch,
) {
    paint_geometry(painter, view, rect, ppp, geometry, 0, scratch);
}

/// [`paint_selection`]'s recursive body.
fn paint_geometry(
    painter: &Painter,
    view: MapView,
    rect: Rect,
    ppp: f32,
    geometry: &Geometry,
    depth: usize,
    scratch: &mut OverlayScratch,
) {
    match geometry {
        Geometry::GeometryCollection(collection) => {
            if depth >= MAX_GEOMETRY_DEPTH {
                return;
            }
            for member in &collection.geometries {
                paint_geometry(painter, view, rect, ppp, member, depth + 1, scratch);
            }
        }
        Geometry::Polygon(polygon) => {
            paint_polygon(painter, view, rect, ppp, &polygon.coordinates, scratch);
        }
        Geometry::MultiPolygon(polygons) => {
            for rings in &polygons.coordinates {
                paint_polygon(painter, view, rect, ppp, rings, scratch);
            }
        }
        _ => {
            for path in paths(geometry) {
                scratch.restart();
                let closed = path.kind == PathKind::Ring;
                {
                    let buffer = scratch.next_ring();
                    project_into(path.positions, view, rect.min, ppp, closed, buffer);
                }
                match path.kind {
                    PathKind::Points => {
                        if let Some(points) = scratch.part().first() {
                            paint_point_markers(painter, points);
                        }
                    }
                    PathKind::Line | PathKind::Ring => {
                        if let Some(points) = scratch.part().first() {
                            stroke_polyline(painter, points);
                        }
                    }
                }
            }
        }
    }
}

/// Paints one polygon part: the tint first, then every ring's outline.
fn paint_polygon(
    painter: &Painter,
    view: MapView,
    rect: Rect,
    ppp: f32,
    rings: &[Vec<Position>],
    scratch: &mut OverlayScratch,
) {
    scratch.restart();
    for ring in rings {
        let buffer = scratch.next_ring();
        project_into(ring, view, rect.min, ppp, true, buffer);
    }
    fill_rings(painter, rect, scratch);
    for index in 0..scratch.used {
        if let Some(points) = scratch.rings.get(index) {
            stroke_polyline(painter, points);
        }
    }
}

/// Projects `positions` into `out`, appending the first position again when
/// `closed` and the sequence does not already end on it.
fn project_into(
    positions: &[Position],
    view: MapView,
    origin: Pos2,
    ppp: f32,
    closed: bool,
    out: &mut Vec<Pos2>,
) {
    out.clear();
    out.extend(
        positions
            .iter()
            .filter_map(position_lon_lat)
            .map(|lon_lat| to_screen(view, origin, ppp, lon_lat)),
    );
    if closed
        && let (Some(first), Some(last)) = (out.first().copied(), out.last().copied())
        && first != last
    {
        out.push(first);
    }
}

/// One pass of a multi-pass stroke: width in egui points, and colour.
type StrokePass = (f32, Color32);

/// The selection's two passes: a dark halo so the outline survives a light
/// basemap, then the accent on top.
const SELECTION_PASSES: [StrokePass; 2] =
    [(SELECTION_HALO_PT, HALO), (SELECTION_STROKE_PT, ACCENT)];

/// The accent at an alpha.
fn accent_alpha(alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(ACCENT.r(), ACCENT.g(), ACCENT.b(), alpha)
}

/// Halo pass then accent pass along `points`.
fn stroke_polyline(painter: &Painter, points: &[Pos2]) {
    stroke_passes(painter, points, &SELECTION_PASSES);
}

/// Every pass of `passes` along `points`, in order.
///
/// One [`Shape::line`] per pass rather than one line-segment shape per
/// consecutive pair: `epaint` tessellates a path into a single mesh with
/// proper joins, so a 10 000-vertex ring costs the paint list two shapes
/// instead of 20 000 — and reads as one continuous outline instead of a chain
/// of butt-capped segments with a gap at every join.
fn stroke_passes(painter: &Painter, points: &[Pos2], passes: &[StrokePass]) {
    if points.len() < 2 {
        return;
    }
    for (width, color) in passes {
        painter.add(Shape::line(points.to_vec(), Stroke::new(*width, *color)));
    }
}

/// Halo-then-accent ring markers around every position of a point geometry.
fn paint_point_markers(painter: &Painter, points: &[Pos2]) {
    marker_passes(painter, points, &SELECTION_PASSES);
}

/// Ring markers around every position, one ring per pass.
///
/// Pass-major rather than point-major so [`Stroke::new`] runs once per pass
/// instead of once per point per pass.
fn marker_passes(painter: &Painter, points: &[Pos2], passes: &[StrokePass]) {
    for (width, color) in passes {
        let stroke = Stroke::new(*width, *color);
        for point in points {
            painter.circle_stroke(*point, POINT_MARKER_PT, stroke);
        }
    }
}

/// Draws every handle of `handles`, with `picked` filled in the accent.
///
/// Squares, deliberately: the selection outline and the point marker are both
/// round, and a control the user may drag has to be distinguishable from the
/// geometry it belongs to at a glance.
pub fn paint_handles(
    painter: &Painter,
    view: MapView,
    rect: Rect,
    ppp: f32,
    handles: &[(VertexRef, LonLat)],
    picked: Option<VertexRef>,
) {
    let origin = rect.min;
    let outline = Stroke::new(HANDLE_OUTLINE_PT, HALO);
    for (reference, position) in handles {
        let center = to_screen(view, origin, ppp, *position);
        let square = Rect::from_center_size(center, Vec2::splat(HANDLE_DRAW_PT * 2.0));
        let fill = if picked == Some(*reference) {
            ACCENT
        } else {
            HANDLE_FILL
        };
        painter.rect_filled(square, 0.0, fill);
        painter.rect_stroke(square, 0.0, outline, egui::StrokeKind::Middle);
    }
}

/// Draws every midpoint ghost: smaller and fainter than a handle, because it
/// stands for a vertex that does not exist yet.
pub fn paint_midpoint_ghosts(
    painter: &Painter,
    view: MapView,
    rect: Rect,
    ppp: f32,
    ghosts: &[(VertexRef, LonLat)],
) {
    let origin = rect.min;
    let fill = Color32::from_rgba_unmultiplied(
        HANDLE_FILL.r(),
        HANDLE_FILL.g(),
        HANDLE_FILL.b(),
        GHOST_ALPHA,
    );
    let outline = Stroke::new(HANDLE_OUTLINE_PT, Color32::from_black_alpha(0x80));
    for (_, position) in ghosts {
        let center = to_screen(view, origin, ppp, *position);
        let square = Rect::from_center_size(center, Vec2::splat(GHOST_DRAW_PT * 2.0));
        painter.rect_filled(square, 0.0, fill);
        painter.rect_stroke(square, 0.0, outline, egui::StrokeKind::Middle);
    }
}

/// Draws a live vertex gesture: the pre-edit geometry as a faint ghost, and the
/// previewed geometry at full strength on top of it.
///
/// The preview is rebuilt from [`VertexDrag::origin`] every frame rather than
/// accumulated, so a long drag cannot drift, and no feature, `Arc` or GPU op is
/// touched for any of it.
pub fn paint_drag(
    painter: &Painter,
    view: MapView,
    rect: Rect,
    ppp: f32,
    drag: &VertexDrag,
    scratch: &mut OverlayScratch,
) {
    let origin = rect.min;
    let ghost_passes = [(SELECTION_STROKE_PT, accent_alpha(DRAG_GHOST_ALPHA))];
    for path in paths(&drag.origin) {
        let closed = path.kind == PathKind::Ring;
        scratch.restart();
        {
            let buffer = scratch.next_ring();
            project_into(path.positions, view, origin, ppp, closed, buffer);
        }
        {
            let buffer = scratch.next_ring();
            project_preview(&path, drag, view, origin, ppp, closed, buffer);
        }
        let part = scratch.part();
        let (Some(before), Some(after)) = (part.first(), part.get(1)) else {
            continue;
        };
        match path.kind {
            PathKind::Points => {
                marker_passes(painter, before, &ghost_passes);
                marker_passes(painter, after, &SELECTION_PASSES);
            }
            PathKind::Line | PathKind::Ring => {
                stroke_passes(painter, before, &ghost_passes);
                stroke_passes(painter, after, &SELECTION_PASSES);
            }
        }
    }
}

/// Projects one path with the live drag applied to it.
///
/// The dragged vertex is *moved* on a move gesture and *inserted* on a midpoint
/// gesture, which is exactly what the commit will do — the preview and the
/// transaction are the same edit expressed twice, so what the user releases on
/// is what they get.
fn project_preview(
    path: &crate::edit::command::PathView<'_>,
    drag: &VertexDrag,
    view: MapView,
    origin: Pos2,
    ppp: f32,
    closed: bool,
    out: &mut Vec<Pos2>,
) {
    out.clear();
    out.extend(
        path.positions
            .iter()
            .enumerate()
            .filter_map(|(index, position)| {
                let stored = position_lon_lat(position)?;
                // A set move previews EVERY marked vertex at its translated
                // position — the preview and the commit are the same edit
                // expressed twice.
                let reference = VertexRef::at(path.part, path.ring, index);
                let live = drag.target_of(reference, stored).unwrap_or(stored);
                Some(to_screen(view, origin, ppp, live))
            }),
    );
    if drag.inserting && path.part == drag.vertex.part && path.ring == drag.vertex.ring {
        let moved = to_screen(view, origin, ppp, drag.current);
        if drag.vertex.index <= out.len() {
            out.insert(drag.vertex.index, moved);
        }
    }
    if closed
        && let (Some(first), Some(last)) = (out.first().copied(), out.last().copied())
        && first != last
    {
        out.push(first);
    }
}

/// Draws the in-progress sketch: the committed polyline, the pending segment to
/// the cursor, a polygon's closing segment, and a dot on every placed vertex.
///
/// The cursor end is drawn at the **snapped** position, never at the raw
/// pointer: the visible jump is the whole feedback that snapping fired, and a
/// band that ends under the crosshair while the vertex lands elsewhere is a lie
/// about where the click will go.
pub fn paint_sketch(
    painter: &Painter,
    view: MapView,
    rect: Rect,
    ppp: f32,
    sketch: &Sketch,
    scratch: &mut OverlayScratch,
) {
    if sketch.is_empty() {
        return;
    }
    let origin = rect.min;
    scratch.restart();
    {
        let buffer = scratch.next_ring();
        buffer.clear();
        buffer.extend(
            sketch
                .points
                .iter()
                .map(|point| to_screen(view, origin, ppp, *point)),
        );
    }
    let Some(points) = scratch.part().first() else {
        return;
    };
    stroke_polyline(painter, points);

    let rubber: [StrokePass; 1] = [(SELECTION_STROKE_PT, accent_alpha(RUBBER_BAND_ALPHA))];
    let cursor = sketch.cursor.map(|at| to_screen(view, origin, ppp, at));
    if let (Some(last), Some(free)) = (points.last().copied(), cursor) {
        stroke_passes(painter, &[last, free], &rubber);
    }
    // A ring shows what it would enclose: the closing segment runs from the
    // free end back to vertex zero, so the polygon on screen is always the
    // polygon a finish would produce.
    if sketch.mode == Some(EditMode::DrawPolygon)
        && points.len() >= 2
        && let (Some(first), Some(free)) = (
            points.first().copied(),
            cursor.or_else(|| points.last().copied()),
        )
    {
        stroke_passes(painter, &[free, first], &rubber);
    }
    paint_sketch_vertices(painter, points);
}

/// A dot on every placed sketch vertex.
///
/// Squares, like the editing handles, and for the same reason: a placed vertex
/// is a control the user can still take back with `Backspace`, and it must not
/// read as part of the geometry's outline.
fn paint_sketch_vertices(painter: &Painter, points: &[Pos2]) {
    let outline = Stroke::new(HANDLE_OUTLINE_PT, HALO);
    for point in points {
        let square = Rect::from_center_size(*point, Vec2::splat(HANDLE_DRAW_PT * 2.0));
        painter.rect_filled(square, 0.0, HANDLE_FILL);
        painter.rect_stroke(square, 0.0, outline, egui::StrokeKind::Middle);
    }
}

/// The handles to draw for a live gesture: the stored ones, with the dragged
/// vertex at its live position.
///
/// The preview *commits to the snap* — the handle is drawn where the vertex will
/// land, not where the pointer is — so the visible jump is the feedback that
/// snapping fired.
#[must_use]
pub fn drag_handles(
    handles: Vec<(VertexRef, LonLat)>,
    drag: &VertexDrag,
) -> Vec<(VertexRef, LonLat)> {
    let mut handles = handles;
    if drag.inserting {
        handles.push((drag.vertex, drag.current));
        return handles;
    }
    for entry in &mut handles {
        // A set move carries every marked handle with the pointer's delta.
        if let Some(live) = drag.target_of(entry.0, entry.1) {
            entry.1 = live;
        }
    }
    handles
}

/// Draws the snap marker: a hollow square for a vertex, a hollow diamond for an
/// edge, a heavier hollow circle for a sketch's own first vertex.
///
/// Hollow in every case, so the marker never hides the thing it is pointing at,
/// and drawn last of the geometry overlays because it is the only one that
/// answers "where exactly will this land".
pub fn paint_snap_marker(painter: &Painter, result: &SnapResult) {
    let stroke = Stroke::new(SNAP_STROKE_PT, ACCENT);
    let center = result.screen_pt;
    match result.kind {
        SnapKind::Vertex => {
            painter.rect_stroke(
                Rect::from_center_size(center, Vec2::splat(SNAP_MARKER_PT * 2.0)),
                0.0,
                stroke,
                egui::StrokeKind::Middle,
            );
        }
        SnapKind::Edge => {
            let reach = SNAP_MARKER_PT * 1.4;
            let corners = [
                center + vec2(0.0, -reach),
                center + vec2(reach, 0.0),
                center + vec2(0.0, reach),
                center + vec2(-reach, 0.0),
            ];
            for index in 0..corners.len() {
                let next = (index + 1) % corners.len();
                if let (Some(a), Some(b)) = (corners.get(index), corners.get(next)) {
                    painter.line_segment([*a, *b], stroke);
                }
            }
        }
        SnapKind::SketchStart => {
            painter.circle_stroke(
                center,
                SNAP_MARKER_PT * 1.4,
                Stroke::new(SNAP_STROKE_PT * 1.6, ACCENT),
            );
        }
    }
}

/// Even-odd scanline fill of the rings gathered in `scratch`, clipped to `clip`.
///
/// See the module docs for why this is not `Shape::Path { fill }`, and for the
/// active-edge-table sweep this function runs instead of a naive per-row edge
/// walk. One `Mesh` carries every row, so a filled polygon costs the paint
/// list exactly one shape however tall it is.
fn fill_rings(painter: &Painter, clip: Rect, scratch: &mut OverlayScratch) {
    let color =
        Color32::from_rgba_unmultiplied(ACCENT.r(), ACCENT.g(), ACCENT.b(), SELECTION_FILL_ALPHA);
    // Split borrows: the sweep below reads `fill_edges` while it mutates
    // `active_edges` and `crossings`, and all three live in `scratch`.
    let OverlayScratch {
        rings,
        used,
        crossings,
        fill_edges,
        active_edges,
    } = scratch;
    let part = rings.get(..*used).unwrap_or(&[]);

    let (mut top, mut bottom) = (f32::INFINITY, f32::NEG_INFINITY);
    for ring in part {
        for point in ring {
            top = top.min(point.y);
            bottom = bottom.max(point.y);
        }
    }
    let top = top.max(clip.min.y);
    let bottom = bottom.min(clip.max.y);
    if !(top.is_finite() && bottom.is_finite()) || bottom <= top {
        return;
    }

    // Build the edge table once for the whole part, sorted by its lower `y`
    // so the sweep below admits edges with a pointer that only moves forward
    // instead of rescanning every edge on every row.
    fill_edges.clear();
    for ring in part {
        for pair in ring.windows(2) {
            let [a, b] = pair else { continue };
            let (y_min, y_max, x_at_min, x_at_max) = if a.y <= b.y {
                (a.y, b.y, a.x, b.x)
            } else {
                (b.y, a.y, b.x, a.x)
            };
            // A horizontal segment never crosses a scanline; filing it would
            // only ever divide by zero for no benefit.
            if y_max <= y_min {
                continue;
            }
            fill_edges.push(FillEdge {
                y_min,
                y_max,
                x_at_min,
                dx_dy: (x_at_max - x_at_min) / (y_max - y_min),
            });
        }
    }
    fill_edges.sort_by(|a, b| a.y_min.total_cmp(&b.y_min));
    active_edges.clear();

    let mut mesh = Mesh::default();
    let mut row = top;
    let mut next_edge = 0usize;
    while row < bottom {
        let height = FILL_ROW_PT.min(bottom - row);
        let middle = row + height * 0.5;

        // Admit every edge whose span has begun by this row's middle — the
        // pointer only ever advances, so each edge is considered for
        // admission exactly once across the whole sweep.
        while let Some(edge) = fill_edges.get(next_edge) {
            if edge.y_min > middle {
                break;
            }
            active_edges.push(next_edge);
            next_edge += 1;
        }
        // Drop edges whose span has ended; what remains is exactly the set
        // spanning this row, never the part's full edge count.
        active_edges.retain(|&index| {
            fill_edges
                .get(index)
                .is_some_and(|edge| edge.y_max > middle)
        });

        crossings.clear();
        for &index in active_edges.iter() {
            let Some(edge) = fill_edges.get(index) else {
                continue;
            };
            crossings.push((middle - edge.y_min).mul_add(edge.dx_dy, edge.x_at_min));
        }
        crossings.sort_by(f32::total_cmp);
        for pair in crossings.chunks_exact(2) {
            let [left, right] = pair else { continue };
            let left = left.max(clip.min.x);
            let right = right.min(clip.max.x);
            if right > left {
                mesh.add_colored_rect(
                    Rect::from_min_max(pos2(left, row), pos2(right, row + height)),
                    color,
                );
            }
        }
        row += height;
    }
    if !mesh.is_empty() {
        painter.add(Shape::mesh(mesh));
    }
}
