// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The `"buffer"` built-in tool: a polygon at a fixed distance around each
//! **point** and **line** feature.
//!
//! # What this is, exactly
//!
//! A point becomes a regular n-gon inscribed in the circle of radius
//! `distance_deg` (n = `4 × quadrant_segments`, so the default 8 gives a
//! 32-gon). A line becomes the closed ring traced by walking one side's
//! offset forward, a round cap around the far end, the other side's offset
//! back, and a round cap around the start — round joins at every outside
//! corner, sampled at the same angular resolution as the caps, and a mitred
//! join at an inside corner where the two offset segments actually meet.
//!
//! # What this is not
//!
//! **Polygon inputs are refused, by design and not by omission.** Buffering
//! an area means offsetting its rings *and dissolving* the overlaps that
//! produces — a planar union (Greiner-Hormann / Vatti / a GEOS-class overlay
//! engine). Nothing in this workspace has one; `oxigeo` 0.2.3 does ship
//! `oxigeo_algorithms::vector::buffer_polygon`, but reaching it means adding
//! the `algorithms` feature to this crate's `oxigeo` dependency, which drags
//! `rayon`, `arrow-array`, `arrow-buffer`, `pest`, `rstar`, `delaunator`,
//! `geographiclib-rs` and `oxigeo-proj` into a crate that compiles to
//! `wasm32` — not a trade this crate can make on its own. So a polygon
//! feature is *counted and skipped*, and a run over a polygon-only layer
//! refuses with that reason rather than quietly producing an empty layer.
//!
//! The same missing union is why a **hairpin** turn tighter than the buffer
//! distance produces a ring that overlaps itself: the traced boundary is
//! correct, its self-overlap is simply not dissolved. Every buffer here is
//! also **planar in degrees** — `distance_deg` is a lon/lat distance, so a
//! buffer is narrower in ground metres the further from the equator it sits.
//! Both facts are in the descriptor's own description, where the user reads
//! them before running anything.

use std::f64::consts::{PI, TAU};
use std::sync::Arc;

use oxigeo::geojson::types::{
    Feature, FeatureCollection, Geometry, MultiPolygon, Polygon, Position,
};
use oxigis_core::{CoreError, ParamKind, ParamSpec, ToolContext, ToolDescriptor, ToolExecutor};

use super::job::FeatureSink;
use super::{LAYER_PARAM, invalid_layer};

/// The tool id this module answers to, in [`super::builtin_executor`] and in
/// the descriptor [`buffer_descriptor`] builds.
pub const BUFFER_TOOL_ID: &str = "buffer";

/// The buffer distance parameter's name, and the fallback name a foreign
/// descriptor might use for the same thing — see [`distance_from`].
const DISTANCE_PARAM: &str = "distance_deg";

/// The alternative spelling [`distance_from`] accepts, so that a descriptor
/// registered elsewhere (e.g. a later `oxigis_core::builtin_registry`
/// carrying its own `buffer`) still drives this executor instead of failing
/// on a name mismatch.
const DISTANCE_PARAM_ALT: &str = "distance";

/// The round-join/round-cap resolution parameter's name.
const SEGMENTS_PARAM: &str = "quadrant_segments";

/// The alternative spelling [`quadrant_segments_from`] accepts, for the same
/// reason as [`DISTANCE_PARAM_ALT`].
const SEGMENTS_PARAM_ALT: &str = "segments";

/// Default arc resolution: 8 segments per quadrant, i.e. a 32-gon circle —
/// the same default `oxigeo_algorithms::vector::BufferOptions` and JTS/GEOS
/// use, so a buffer produced here looks like one produced anywhere else.
const DEFAULT_QUADRANT_SEGMENTS: usize = 8;

/// Most segments per quadrant a run may ask for: 64 gives a 256-gon per
/// point, which is already beyond what any renderer resolves, and bounds the
/// output size of a run over a large point layer (a cap, not a preference).
const MAX_QUADRANT_SEGMENTS: usize = 64;

/// How far past the corner a mitred inside join may reach, as a multiple of
/// the buffer distance, before falling back to a bevel. Matches the JTS/GEOS
/// default miter limit of 5.
const MITER_LIMIT: f64 = 5.0;

/// The descriptor for this tool.
///
/// Lives here rather than in [`oxigis_core::builtin_registry`] because the
/// registry is in another crate: the app registers this one itself when the
/// The descriptor now LIVES in [`oxigis_core::builtin_registry`], which every
/// `OxigisApp` starts from, so nothing registers it at runtime any more. This
/// function is kept as the executor's own statement of the contract it reads —
/// parameter names, bounds and defaults — and an equality test in
/// `app::dispatch::tests` pins the two together so they cannot drift.
///
/// The description states both limitations the module docs explain, because
/// that string is what the panel shows above the form.
#[must_use]
pub fn buffer_descriptor() -> ToolDescriptor {
    ToolDescriptor {
        id: BUFFER_TOOL_ID.to_string(),
        title: "Buffer (points and lines)".to_string(),
        description: "Draws a polygon at a fixed distance around every point and line \
                      feature. The distance is in DEGREES of longitude/latitude, not \
                      metres, so a buffer covers less ground the further it sits from \
                      the equator. Polygon and multipolygon features are skipped: \
                      buffering an area needs a planar union to dissolve the overlaps \
                      that offsetting its rings produces, which this build does not have."
            .to_string(),
        params: vec![
            ParamSpec {
                name: LAYER_PARAM.to_string(),
                kind: ParamKind::LayerRef,
                required: true,
                default: None,
            },
            ParamSpec {
                name: DISTANCE_PARAM.to_string(),
                kind: ParamKind::Number {
                    min: Some(0.0),
                    max: None,
                },
                required: true,
                default: Some(serde_json::json!(0.001)),
            },
            ParamSpec {
                name: SEGMENTS_PARAM.to_string(),
                kind: ParamKind::Number {
                    min: Some(1.0),
                    // Bounded so the field draws as a slider and cannot ask
                    // for an unbounded vertex count.
                    max: Some(MAX_QUADRANT_SEGMENTS as f64),
                },
                required: false,
                default: Some(serde_json::json!(DEFAULT_QUADRANT_SEGMENTS as f64)),
            },
        ],
    }
}

/// The `"buffer"` built-in tool.
pub(super) struct BufferTool {
    /// The target layer's parsed features.
    pub(super) features: Arc<FeatureCollection>,
}

impl ToolExecutor for BufferTool {
    fn run(&self, context: &ToolContext) -> Result<serde_json::Value, CoreError> {
        super::job::drain(Arc::clone(&self.features), sink(context)?)
    }
}

/// [`BufferTool`]'s per-feature accumulator, with the run's distance and arc
/// resolution already validated out of `context`.
///
/// # Errors
///
/// Returns [`CoreError::InvalidParameter`] when the distance is missing, not
/// a number, or not a finite positive one, or when the segment count is
/// present but out of range.
pub(super) fn sink(context: &ToolContext) -> Result<Box<dyn FeatureSink>, CoreError> {
    Ok(Box::new(BufferSink {
        options: BufferOptions {
            distance: distance_from(context)?,
            quadrant_segments: quadrant_segments_from(context)?,
        },
        buffered: Vec::new(),
        skipped_areas: 0,
        skipped_empty: 0,
    }))
}

/// One run's settings.
#[derive(Debug, Clone, Copy)]
struct BufferOptions {
    /// The buffer distance, in degrees; always finite and `> 0`.
    distance: f64,
    /// Arc samples per quadrant; always in `1..=MAX_QUADRANT_SEGMENTS`.
    quadrant_segments: usize,
}

impl BufferOptions {
    /// How many samples to spend on an arc that sweeps `angle` radians, at
    /// least one so a join never degenerates into a missing edge.
    fn arc_steps(&self, angle: f64) -> usize {
        let per_quadrant = self.quadrant_segments as f64;
        let steps = (angle.abs() / (PI / 2.0) * per_quadrant).ceil();
        // `as usize` after a `ceil` on a value bounded by `4 * per_quadrant`
        // (a full turn) cannot overflow: `MAX_QUADRANT_SEGMENTS * 4` is 256.
        (steps as usize).clamp(1, MAX_QUADRANT_SEGMENTS * 4)
    }
}

/// The accumulator behind [`sink`].
#[derive(Debug)]
struct BufferSink {
    /// The run's settings.
    options: BufferOptions,
    /// One buffered feature per input feature that produced one.
    buffered: Vec<Feature>,
    /// Polygon/multipolygon features refused for want of a planar union.
    skipped_areas: usize,
    /// Features with no geometry, or none this tool could turn into an area.
    skipped_empty: usize,
}

impl FeatureSink for BufferSink {
    fn absorb(&mut self, feature: &Feature) {
        let Some(geometry) = feature.geometry.as_ref() else {
            self.skipped_empty += 1;
            return;
        };
        let mut rings = Vec::new();
        let refused_area = collect_buffer_rings(geometry, &self.options, &mut rings);
        if rings.is_empty() {
            if refused_area {
                self.skipped_areas += 1;
            } else {
                self.skipped_empty += 1;
            }
            return;
        }
        if refused_area {
            // A GeometryCollection with both a line and a polygon in it: the
            // line's buffer is real output, the polygon's refusal is still a
            // refusal, and both are counted.
            self.skipped_areas += 1;
        }
        let Some(geometry) = rings_to_geometry(rings) else {
            self.skipped_empty += 1;
            return;
        };
        let mut buffered = Feature::new(Some(geometry), feature.properties.clone());
        buffered.id = feature.id.clone();
        self.buffered.push(buffered);
    }

    fn finish(self: Box<Self>) -> Result<serde_json::Value, CoreError> {
        if self.buffered.is_empty() && self.skipped_areas > 0 {
            // Nothing at all came out and areas are why: say so where the
            // user is looking, rather than letting the app report the generic
            // "no feature produced a result".
            return Err(invalid_layer(
                "Buffer takes point and line features; this layer's areas were skipped \
                 because buffering one needs a planar union to dissolve the overlaps \
                 that offsetting its rings produces, which this build does not have",
            ));
        }
        if self.skipped_areas > 0 || self.skipped_empty > 0 {
            tracing::warn!(
                skipped_areas = self.skipped_areas,
                skipped_empty = self.skipped_empty,
                "oxigis-ui: buffer skipped features it cannot buffer",
            );
        }
        Ok(serde_json::to_value(FeatureCollection::new(self.buffered))?)
    }
}

/// Appends every buffer ring `geometry` produces to `out`, returning whether
/// an **area** member was refused along the way (which the caller counts
/// separately from "produced nothing").
fn collect_buffer_rings(
    geometry: &Geometry,
    options: &BufferOptions,
    out: &mut Vec<Vec<Position>>,
) -> bool {
    match geometry {
        Geometry::Point(point) => {
            if let Some(ring) = point_buffer_ring(&point.coordinates, options) {
                out.push(ring);
            }
            false
        }
        Geometry::MultiPoint(points) => {
            for position in &points.coordinates {
                if let Some(ring) = point_buffer_ring(position, options) {
                    out.push(ring);
                }
            }
            false
        }
        Geometry::LineString(line) => {
            if let Some(ring) = line_buffer_ring(&line.coordinates, options) {
                out.push(ring);
            }
            false
        }
        Geometry::MultiLineString(lines) => {
            for line in &lines.coordinates {
                if let Some(ring) = line_buffer_ring(line, options) {
                    out.push(ring);
                }
            }
            false
        }
        // The one honest answer this build has for an area — see the module
        // docs. Reported up, never silently dropped.
        Geometry::Polygon(_) | Geometry::MultiPolygon(_) => true,
        Geometry::GeometryCollection(collection) => {
            let mut refused = false;
            for member in &collection.geometries {
                refused |= collect_buffer_rings(member, options, out);
            }
            refused
        }
    }
}

/// Turns one feature's buffer rings into the geometry that carries them: a
/// `Polygon` for one ring, a `MultiPolygon` for several.
///
/// [`None`] when `oxigeo`'s own constructors refuse the rings — which they do
/// for a ring below four positions, a shape this module never builds (the
/// smallest circle it can emit is a 4-gon plus its closing position).
fn rings_to_geometry(mut rings: Vec<Vec<Position>>) -> Option<Geometry> {
    if rings.len() == 1 {
        let ring = rings.pop()?;
        return Polygon::from_exterior(ring).ok().map(Geometry::Polygon);
    }
    let members: Vec<Vec<Vec<Position>>> = rings.into_iter().map(|ring| vec![ring]).collect();
    MultiPolygon::new(members).ok().map(Geometry::MultiPolygon)
}

/// The closed ring of a regular n-gon around `position`, counter-clockwise
/// (RFC 7946 §3.1.6 asks an exterior ring to wind that way).
///
/// [`None`] for a position that is not a finite 2-D one — a malformed
/// coordinate has no centre to circle.
fn point_buffer_ring(position: &Position, options: &BufferOptions) -> Option<Vec<Position>> {
    let centre = finite_xy(position)?;
    let steps = options.quadrant_segments.saturating_mul(4).max(4);
    let mut ring = Vec::with_capacity(steps + 1);
    for step in 0..steps {
        let angle = TAU * (step as f64) / (steps as f64);
        ring.push(vec![
            centre[0] + options.distance * angle.cos(),
            centre[1] + options.distance * angle.sin(),
        ]);
    }
    close_ring(&mut ring);
    Some(ring)
}

/// The closed ring of a polyline's buffer: one side's offset forward, a round
/// cap, the other side's offset back, and a round cap home.
///
/// A line whose usable positions collapse to a single point buffers as that
/// point (what GEOS does for a degenerate line); one with no usable position
/// at all yields [`None`].
///
/// The two halves fit together without a seam by construction: the reversed
/// walk's first segment normal is exactly the negation of the forward walk's
/// last, so the far-end cap's closing vertex *is* the reversed walk's opening
/// vertex — which is why [`push_arc`] emits an arc's interior samples only.
fn line_buffer_ring(coords: &[Position], options: &BufferOptions) -> Option<Vec<Position>> {
    let points = distinct_finite_points(coords);
    match points.len() {
        0 => return None,
        1 => return point_buffer_ring(&vec![points[0][0], points[0][1]], options),
        _ => {}
    }
    let mut ring = Vec::new();
    offset_side(&points, options, &mut ring);
    let reversed: Vec<[f64; 2]> = points.iter().rev().copied().collect();
    offset_side(&reversed, options, &mut ring);
    close_ring(&mut ring);
    Some(ring)
}

/// Appends one side's offset chain — the **right** side of `points` walked
/// from first to last — plus the round cap that turns the far end around, so
/// that calling this twice (the second time on the reversed points) traces a
/// whole buffer.
///
/// The right side, not the left, is what makes the finished ring wind
/// counter-clockwise: walking a boundary with the interior on the left is
/// exactly RFC 7946 §3.1.6's exterior-ring convention, and the interior of a
/// line buffer is on the left of its right-hand offset.
///
/// `points` must already be [`distinct_finite_points`], so every consecutive
/// pair has a direction and [`right_normal`] never declines.
fn offset_side(points: &[[f64; 2]], options: &BufferOptions, out: &mut Vec<Position>) {
    let normals: Vec<[f64; 2]> = points
        .windows(2)
        .filter_map(|pair| right_normal(pair[0], pair[1]))
        .collect();
    let (Some(&first_normal), Some(&last_normal)) = (normals.first(), normals.last()) else {
        return;
    };
    let distance = options.distance;
    // The chain is written vertex by vertex, not segment by segment: an
    // inside corner REPLACES the two offset segment ends with one mitre
    // point, and a segment-by-segment walk that pushed both and then added
    // the mitre would leave a zero-width spike between them.
    out.push(offset_position(points[0], first_normal, distance));
    for index in 1..normals.len() {
        let corner = points[index];
        let (before, after) = (normals[index - 1], normals[index]);
        push_join(out, corner, before, after, options);
    }
    if let Some(&last) = points.last() {
        out.push(offset_position(last, last_normal, distance));
        // The round cap: a half turn from this side's normal to the other's.
        // Its closing vertex belongs to whatever walk comes next (the
        // reversed side, or `close_ring`), so only the arc's interior is
        // emitted here.
        push_arc(
            out,
            last,
            last_normal,
            [-last_normal[0], -last_normal[1]],
            options,
        );
    }
}

/// Appends the join between two consecutive offset segments that meet at
/// `corner`.
///
/// Outside corners (where the offset opens a gap) get both offset ends plus a
/// round arc between them, sampled at the run's resolution. Inside corners
/// (where the two offsets overlap) get the single mitre point where the offset
/// *lines* cross — unless that point sits further than [`MITER_LIMIT`]
/// buffer-distances from the corner, a nearly-doubled-back turn, in which case
/// both offset ends are emitted (a bevel) and the small overlap between them
/// is left undissolved. See the module docs on the missing planar union.
fn push_join(
    out: &mut Vec<Position>,
    corner: [f64; 2],
    before: [f64; 2],
    after: [f64; 2],
    options: &BufferOptions,
) {
    let distance = options.distance;
    // Rotation preserves the cross product, so the cross of the two NORMALS
    // has the sign of the turn between the two segment directions: positive
    // when the path turns left, which is exactly when the RIGHT side is the
    // outside of the corner and has a gap to fill.
    let cross = before[0] * after[1] - before[1] * after[0];
    if cross > 0.0 {
        out.push(offset_position(corner, before, distance));
        push_arc(out, corner, before, after, options);
        out.push(offset_position(corner, after, distance));
        return;
    }
    match miter_position(corner, before, after, distance) {
        Some(miter) => out.push(miter),
        None => {
            out.push(offset_position(corner, before, distance));
            out.push(offset_position(corner, after, distance));
        }
    }
}

/// Where the two offset lines meeting at an inside corner actually cross.
///
/// The bisector of the two unit normals points at it, and the distance along
/// that bisector is `distance / cos(half the turn)` — which the bisector's own
/// half-length is the cosine of. [`None`] when the two normals cancel (a line
/// doubling exactly back on itself, which has no crossing point at all) or
/// when the mitre would reach further than [`MITER_LIMIT`] buffer-distances.
fn miter_position(
    corner: [f64; 2],
    before: [f64; 2],
    after: [f64; 2],
    distance: f64,
) -> Option<Position> {
    let bisector = [before[0] + after[0], before[1] + after[1]];
    let length = (bisector[0] * bisector[0] + bisector[1] * bisector[1]).sqrt();
    if !length.is_finite() || length <= f64::EPSILON {
        return None;
    }
    let miter = distance / (length / 2.0);
    if miter > distance * MITER_LIMIT {
        return None;
    }
    Some(vec![
        corner[0] + bisector[0] / length * miter,
        corner[1] + bisector[1] / length * miter,
    ])
}

/// Appends the arc of radius `distance` around `centre` that sweeps
/// counter-clockwise from the `from` normal to the `to` normal, **excluding**
/// its two endpoints (which the offset segments either side of it carry).
///
/// Counter-clockwise always: on a right-hand offset chain, an outside corner
/// and an end cap both turn that way, and taking the sweep the other way
/// round would trace the arc through the buffer's interior.
fn push_arc(
    out: &mut Vec<Position>,
    centre: [f64; 2],
    from: [f64; 2],
    to: [f64; 2],
    options: &BufferOptions,
) {
    let start = from[1].atan2(from[0]);
    let end = to[1].atan2(to[0]);
    let mut sweep = end - start;
    while sweep <= 0.0 {
        sweep += TAU;
    }
    while sweep > TAU {
        sweep -= TAU;
    }
    let steps = options.arc_steps(sweep);
    for step in 1..steps {
        let angle = start + sweep * (step as f64) / (steps as f64);
        out.push(vec![
            centre[0] + options.distance * angle.cos(),
            centre[1] + options.distance * angle.sin(),
        ]);
    }
}

/// The unit normal pointing to the right of the direction `a → b`, or [`None`]
/// when the two coincide (there is no direction to be normal to).
fn right_normal(a: [f64; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let length = (dx * dx + dy * dy).sqrt();
    if !length.is_finite() || length <= 0.0 {
        return None;
    }
    Some([dy / length, -dx / length])
}

/// `point` moved `distance` along `normal`.
fn offset_position(point: [f64; 2], normal: [f64; 2], distance: f64) -> Position {
    vec![
        point[0] + normal[0] * distance,
        point[1] + normal[1] * distance,
    ]
}

/// Repeats a ring's first position as its last, per RFC 7946 §3.1.6.
fn close_ring(ring: &mut Vec<Position>) {
    if let Some(first) = ring.first().cloned()
        && ring.last() != Some(&first)
    {
        ring.push(first);
    }
}

/// A position's first two components, when both are finite.
fn finite_xy(position: &Position) -> Option<[f64; 2]> {
    let (&x, &y) = (position.first()?, position.get(1)?);
    (x.is_finite() && y.is_finite()).then_some([x, y])
}

/// `coords` reduced to finite 2-D points with consecutive duplicates
/// collapsed — the shape the offset walk needs, since a repeated position
/// carries no direction and would produce a normal of `NaN`.
fn distinct_finite_points(coords: &[Position]) -> Vec<[f64; 2]> {
    let mut points: Vec<[f64; 2]> = Vec::with_capacity(coords.len());
    for position in coords {
        let Some(point) = finite_xy(position) else {
            continue;
        };
        if points.last() != Some(&point) {
            points.push(point);
        }
    }
    points
}

/// Reads the buffer distance out of a run's parameters.
///
/// Accepts [`DISTANCE_PARAM`] or [`DISTANCE_PARAM_ALT`]: the descriptor this
/// module ships uses the first, but the executor is reachable from any
/// registry entry under [`BUFFER_TOOL_ID`] — including one a later
/// `oxigis_core::builtin_registry` might carry — and a name mismatch should
/// not turn into "the distance is missing" when the user plainly set one.
///
/// # Errors
///
/// Returns [`CoreError::InvalidParameter`] naming both spellings when neither
/// is present, and when the value is not a finite number greater than zero: a
/// zero or negative buffer bounds no area, and a shrink (a negative buffer of
/// a polygon) is not something a point/line tool can mean.
fn distance_from(context: &ToolContext) -> Result<f64, CoreError> {
    let value = context
        .params
        .get(DISTANCE_PARAM)
        .or_else(|| context.params.get(DISTANCE_PARAM_ALT))
        .ok_or_else(|| {
            invalid_distance(&format!(
                "missing required parameter (looked for {DISTANCE_PARAM} and \
                 {DISTANCE_PARAM_ALT})"
            ))
        })?;
    let distance = value
        .as_f64()
        .ok_or_else(|| invalid_distance("expected a number"))?;
    if !distance.is_finite() {
        return Err(invalid_distance("expected a finite number"));
    }
    if distance <= 0.0 {
        return Err(invalid_distance("must be greater than zero"));
    }
    Ok(distance)
}

/// Reads the arc resolution out of a run's parameters, defaulting to
/// [`DEFAULT_QUADRANT_SEGMENTS`] when absent — unlike the distance, a missing
/// resolution has an obviously right answer, and a foreign descriptor may not
/// carry the parameter at all.
///
/// # Errors
///
/// Returns [`CoreError::InvalidParameter`] when the value is present but is
/// not a number, or is not in `1..=MAX_QUADRANT_SEGMENTS`.
fn quadrant_segments_from(context: &ToolContext) -> Result<usize, CoreError> {
    let Some(value) = context
        .params
        .get(SEGMENTS_PARAM)
        .or_else(|| context.params.get(SEGMENTS_PARAM_ALT))
        .filter(|value| !value.is_null())
    else {
        return Ok(DEFAULT_QUADRANT_SEGMENTS);
    };
    let segments = value.as_f64().ok_or_else(|| CoreError::InvalidParameter {
        name: SEGMENTS_PARAM.to_string(),
        reason: "expected a number".to_string(),
    })?;
    if !segments.is_finite() || segments < 1.0 || segments > MAX_QUADRANT_SEGMENTS as f64 {
        return Err(CoreError::InvalidParameter {
            name: SEGMENTS_PARAM.to_string(),
            reason: format!("must be between 1 and {MAX_QUADRANT_SEGMENTS}"),
        });
    }
    // Truncation is the intent: 8.7 segments per quadrant is 8.
    Ok(segments as usize)
}

/// Builds a [`CoreError::InvalidParameter`] against the distance parameter.
fn invalid_distance(reason: &str) -> CoreError {
    CoreError::InvalidParameter {
        name: DISTANCE_PARAM.to_string(),
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing_exec::fixtures::{parse, run_to_collection, signed_ring_area};
    use oxigeo::geojson::types::FeatureId;

    /// A context with a distance and the default arc resolution.
    fn buffer_context(distance: f64) -> ToolContext {
        let mut context = ToolContext::new();
        context
            .params
            .insert(DISTANCE_PARAM.to_string(), serde_json::json!(distance));
        context
    }

    /// The exterior ring of a single-feature, single-polygon result.
    fn only_ring(result: &FeatureCollection) -> Vec<Position> {
        assert_eq!(result.features.len(), 1, "expected exactly one feature");
        let Some(Geometry::Polygon(polygon)) = result.features[0].geometry.as_ref() else {
            panic!("a buffer must be a Polygon, got {:?}", result.features[0]);
        };
        polygon.exterior().cloned().unwrap_or_default()
    }

    /// The distance from `[x, y]` to the origin-relative `centre`.
    fn distance(a: &Position, centre: [f64; 2]) -> f64 {
        ((a[0] - centre[0]).powi(2) + (a[1] - centre[1]).powi(2)).sqrt()
    }

    #[test]
    fn a_point_buffers_to_a_closed_ccw_ngon_at_the_asked_radius() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","id":7,"properties":{"name":"well"},
                 "geometry":{"type":"Point","coordinates":[10,20]}}
            ]}"#,
        );
        let tool = BufferTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &buffer_context(2.0));
        let ring = only_ring(&result);

        assert_eq!(
            ring.len(),
            DEFAULT_QUADRANT_SEGMENTS * 4 + 1,
            "8 segments per quadrant is a closed 32-gon"
        );
        assert_eq!(ring.first(), ring.last(), "the ring must be closed");
        assert!(
            signed_ring_area(&ring) > 0.0,
            "an exterior ring must wind counter-clockwise"
        );
        for position in &ring {
            assert!(
                (distance(position, [10.0, 20.0]) - 2.0).abs() < 1e-9,
                "every vertex must sit on the circle: {position:?}"
            );
        }
        assert_eq!(
            result.features[0].id,
            Some(FeatureId::Number(7)),
            "the source id must survive so the result joins back to the input"
        );
        assert_eq!(
            result.features[0].get_property("name"),
            Some(&serde_json::json!("well"))
        );
    }

    #[test]
    fn the_arc_resolution_is_honoured_and_bounded() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[0,0]}}
            ]}"#,
        );
        let tool = BufferTool {
            features: Arc::new(features),
        };
        let mut context = buffer_context(1.0);
        context
            .params
            .insert(SEGMENTS_PARAM.to_string(), serde_json::json!(2));
        let result = run_to_collection(&tool, &context);
        assert_eq!(only_ring(&result).len(), 9, "2 per quadrant is an octagon");

        context.params.insert(
            SEGMENTS_PARAM.to_string(),
            serde_json::json!(MAX_QUADRANT_SEGMENTS as f64 + 1.0),
        );
        let error = tool.run(&context).expect_err("beyond the cap");
        assert_eq!(
            error,
            CoreError::InvalidParameter {
                name: SEGMENTS_PARAM.to_string(),
                reason: format!("must be between 1 and {MAX_QUADRANT_SEGMENTS}"),
            }
        );
    }

    #[test]
    fn a_straight_line_buffers_to_a_stadium_of_the_right_extent() {
        // A horizontal 10-long segment buffered by 1: the result must reach
        // x = -1 .. 11 and y = -1 .. 1, and every vertex must sit within the
        // buffer distance of the line.
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"LineString",
                 "coordinates":[[0,0],[10,0]]}}
            ]}"#,
        );
        let tool = BufferTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &buffer_context(1.0));
        let ring = only_ring(&result);

        assert_eq!(ring.first(), ring.last(), "the ring must be closed");
        let min_x = ring.iter().fold(f64::INFINITY, |acc, p| acc.min(p[0]));
        let max_x = ring.iter().fold(f64::NEG_INFINITY, |acc, p| acc.max(p[0]));
        let min_y = ring.iter().fold(f64::INFINITY, |acc, p| acc.min(p[1]));
        let max_y = ring.iter().fold(f64::NEG_INFINITY, |acc, p| acc.max(p[1]));
        assert!((min_x + 1.0).abs() < 1e-9, "the start cap must reach -1");
        assert!((max_x - 11.0).abs() < 1e-9, "the end cap must reach 11");
        assert!((min_y + 1.0).abs() < 1e-9 && (max_y - 1.0).abs() < 1e-9);
        for position in &ring {
            let along = position[0].clamp(0.0, 10.0);
            assert!(
                distance(position, [along, 0.0]) <= 1.0 + 1e-9,
                "no vertex may sit outside the buffer distance: {position:?}"
            );
        }
        assert!(
            signed_ring_area(&ring) > 0.0,
            "the traced ring must wind counter-clockwise"
        );
        // A stadium's area is `2·d·length + π·d²`; the polygonal ring is
        // inscribed, so it is a little under that and well over the
        // rectangle alone.
        let area = signed_ring_area(&ring);
        assert!(
            area > 20.0 && area < 2.0 * 10.0 + PI + 1e-9,
            "a 10x1 stadium's area is bracketed by the rectangle and the exact \
             stadium, got {area}"
        );
    }

    #[test]
    fn a_right_angle_line_gets_a_round_join_outside_and_a_mitre_inside() {
        // An L: (0,0) → (10,0) → (10,10), buffered by 1. The outside of the
        // corner is at (10,0) offset by (+1,-1)-ish and must be traced by an
        // arc (several vertices near the corner at radius 1); the inside is
        // the single mitre point (9, 1).
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"LineString",
                 "coordinates":[[0,0],[10,0],[10,10]]}}
            ]}"#,
        );
        let tool = BufferTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &buffer_context(1.0));
        let ring = only_ring(&result);

        let on_corner_arc = ring
            .iter()
            .filter(|position| (distance(position, [10.0, 0.0]) - 1.0).abs() < 1e-9)
            .count();
        assert!(
            on_corner_arc >= 3,
            "the outside of a right-angle corner must be rounded, not cut: {ring:?}"
        );
        assert!(
            ring.iter().any(
                |position| (position[0] - 9.0).abs() < 1e-9 && (position[1] - 1.0).abs() < 1e-9
            ),
            "the inside corner must be mitred to (9, 1): {ring:?}"
        );
        for position in &ring {
            let to_first = distance(position, [position[0].clamp(0.0, 10.0), 0.0]);
            let to_second = distance(position, [10.0, position[1].clamp(0.0, 10.0)]);
            assert!(
                to_first.min(to_second) <= 1.0 + 1e-9,
                "no vertex may sit outside the buffer distance: {position:?}"
            );
        }
    }

    #[test]
    fn a_multi_geometry_buffers_to_one_multipolygon_per_member() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"MultiPoint",
                 "coordinates":[[0,0],[10,10]]}}
            ]}"#,
        );
        let tool = BufferTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &buffer_context(1.0));
        let Some(Geometry::MultiPolygon(polygons)) = result.features[0].geometry.as_ref() else {
            panic!("two members must produce a MultiPolygon");
        };
        assert_eq!(polygons.coordinates.len(), 2);
    }

    #[test]
    fn a_degenerate_line_buffers_as_the_point_it_collapses_to() {
        // Every position identical: there is no direction to offset along, so
        // the honest answer is the point's own circle, not nothing.
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"LineString",
                 "coordinates":[[3,4],[3,4],[3,4]]}}
            ]}"#,
        );
        let tool = BufferTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &buffer_context(1.0));
        let ring = only_ring(&result);
        assert_eq!(ring.len(), DEFAULT_QUADRANT_SEGMENTS * 4 + 1);
        for position in &ring {
            assert!((distance(position, [3.0, 4.0]) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn a_polygon_only_layer_is_refused_with_the_reason_named() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"Polygon",
                 "coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}}
            ]}"#,
        );
        let tool = BufferTool {
            features: Arc::new(features),
        };
        let error = tool
            .run(&buffer_context(1.0))
            .expect_err("areas need a planar union this build has not got");
        let CoreError::InvalidParameter { name, reason } = error else {
            panic!("expected a parameter refusal");
        };
        assert_eq!(name, LAYER_PARAM);
        assert!(
            reason.contains("planar union"),
            "the refusal must name the limitation: {reason}"
        );
    }

    #[test]
    fn a_mixed_layer_buffers_what_it_can_and_skips_the_areas() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":{"type":"Polygon",
                 "coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]},
                 "id":"area"},
                {"type":"Feature","properties":{},"geometry":{"type":"Point",
                 "coordinates":[5,5]},"id":"point"}
            ]}"#,
        );
        let tool = BufferTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &buffer_context(1.0));
        assert_eq!(
            result.features.len(),
            1,
            "the point buffers, the polygon is skipped rather than failing the run"
        );
        assert_eq!(
            result.features[0].id,
            Some(FeatureId::String("point".to_string()))
        );
    }

    #[test]
    fn a_null_geometry_feature_is_skipped_not_fatal() {
        let features = parse(
            r#"{"type":"FeatureCollection","features":[
                {"type":"Feature","properties":{},"geometry":null},
                {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[0,0]}}
            ]}"#,
        );
        let tool = BufferTool {
            features: Arc::new(features),
        };
        let result = run_to_collection(&tool, &buffer_context(1.0));
        assert_eq!(result.features.len(), 1);
    }

    #[test]
    fn the_distance_is_required_finite_and_positive() {
        let features = Arc::new(FeatureCollection::new(vec![]));
        let tool = BufferTool {
            features: Arc::clone(&features),
        };
        let error = tool
            .run(&ToolContext::new())
            .expect_err("the distance is required");
        let CoreError::InvalidParameter { name, reason } = error else {
            panic!("expected a parameter refusal");
        };
        assert_eq!(name, DISTANCE_PARAM);
        assert!(
            reason.contains(DISTANCE_PARAM_ALT),
            "the refusal must say what it looked for: {reason}"
        );

        for bad in [0.0, -1.0] {
            let error = tool
                .run(&buffer_context(bad))
                .expect_err("a non-positive buffer bounds no area");
            assert_eq!(
                error,
                CoreError::InvalidParameter {
                    name: DISTANCE_PARAM.to_string(),
                    reason: "must be greater than zero".to_string(),
                },
                "for {bad}"
            );
        }
    }

    #[test]
    fn the_alternative_parameter_spellings_are_accepted() {
        // A descriptor registered elsewhere under the same tool id must still
        // drive this executor — see `distance_from`'s docs.
        let mut context = ToolContext::new();
        context
            .params
            .insert(DISTANCE_PARAM_ALT.to_string(), serde_json::json!(1.0));
        context
            .params
            .insert(SEGMENTS_PARAM_ALT.to_string(), serde_json::json!(4));
        assert_eq!(distance_from(&context), Ok(1.0));
        assert_eq!(quadrant_segments_from(&context), Ok(4));
    }

    #[test]
    fn the_descriptor_names_both_limitations_the_module_documents() {
        let descriptor = buffer_descriptor();
        assert_eq!(descriptor.id, BUFFER_TOOL_ID);
        assert!(
            descriptor.description.contains("DEGREES"),
            "a planar degree buffer must say so: {}",
            descriptor.description
        );
        assert!(
            descriptor.description.contains("planar union"),
            "the polygon refusal must be visible before the run: {}",
            descriptor.description
        );
        // One LayerRef parameter is what `run_processing_tool` requires to
        // resolve a layer at all.
        assert_eq!(
            descriptor
                .params
                .iter()
                .filter(|spec| matches!(spec.kind, ParamKind::LayerRef))
                .count(),
            1
        );
    }
}
