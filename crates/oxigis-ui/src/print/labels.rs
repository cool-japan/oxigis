// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Symbol-layer labels on the printed page.
//!
//! The print twin of the on-screen label pass (`oxigis-render`'s
//! `label/place.rs`), sharing its **rules** rather than its code — the
//! screen pass works in tile space against a glyph atlas, while the page
//! works in points against the embedded fonts:
//!
//! * anchors: point → first position; line → arc-length midpoint of the
//!   longest part; polygon → shoelace centroid of the largest exterior ring
//!   (mean fallback for degenerate rings);
//! * priority: top-most layer first, then per layer by importance
//!   (|area| / length, descending) — the same features win collisions on
//!   paper as on screen;
//! * greedy collision on padded AABBs, measured with the *exact* `/W`
//!   widths of the embedded fonts (so the boxes cannot lie), fully inside
//!   the map box;
//! * orientation: a [`oxigis_core::LabelOrientation::Vertical`] style is set
//!   as a stacked column (print v1.6) whose box is one em wide and the summed
//!   cell pitch tall — the same rectangle the on-screen engine reserves — in
//!   the SAME collision pool as the horizontal labels, because on paper they
//!   share the page. A label the vertical ladder refuses falls back to the
//!   horizontal form per label, and the export says so once.
//!
//! Labels only exist on the page when an embedded-font plan is live: the
//! degraded Helvetica mode would draw `?` strings for CJK names, which is
//! worse than the v1 behavior of drawing nothing. Halos are the classic
//! two-pass PDF form — stroke-only text under fill-only text.

use std::sync::Arc;

use oxigeo::geojson::types::{Geometry, Position};
use oxigis_core::{Color, LabelWeight, LayerStyle, SymbolStyle, VectorTilePaint};
use oxigis_render::mvt::VectorTile as MvtVectorTile;
use oxigis_render::{MapView, TileId, feature_anchor, label_text};

use super::font::TextPlan;
use super::mvt::TileFrame;
use super::{MapBox, PrintRequest, project};
use crate::edit::command::{self, PathKind};

/// Padding around a label's box during collision testing, in points.
const COLLISION_PAD_PT: f32 = 2.0;

/// Smallest printed label size, in points — below this the text is noise.
const MIN_LABEL_PT: f32 = 4.0;

/// Where an UPRIGHT cell's em box sits above the glyph origin the emitter
/// steps from, as a fraction of the em — the conventional CJK ascent.
///
/// [`super::emit::show_vertical_line`] draws an upright cell on its
/// HORIZONTAL baseline and steps down by the pitch from there, so the cell's
/// em box hangs this far above the origin; a rotated run instead advances
/// straight down from the origin, so its box starts there. Turning the
/// collision box the anchor centred into the origin the emitter starts from
/// is [`column_top_offset`]'s whole job — without it a stacked label would
/// draw most of an em above the space it reserved.
const VERTICAL_CELL_ASCENT_EM: f32 = 0.88;

/// Hard cap on streamed-tile label candidates per PAINT RULE — bounds both
/// the font plan and the greedy collision loop on a dense basemap.
/// Deterministic: candidates are collected in (rule, tile, feature) order and
/// each rule's tail is dropped, with a log so the cut is visible.
///
/// Per rule rather than per export since v1.6: a single global cap let a
/// dense first rule (every building in a city tile) starve every later rule
/// of labels, which is a silently empty layer rather than a thinned one.
const MAX_MVT_LABELS_PER_RULE: usize = 512;

/// Hard cap on streamed-tile label candidates for the WHOLE export, so a
/// project with many Symbol rules cannot multiply the per-rule cap without
/// bound.
const MAX_MVT_LABELS: usize = 2048;

/// Hard cap on LOCAL Symbol-layer label candidates per LAYER — the local
/// twin of [`MAX_MVT_LABELS_PER_RULE`], and for the same reason: a single
/// dense layer (every building in a city) would otherwise eat the whole
/// export budget and leave every layer under it silently unlabelled, which
/// is worse than a thinned layer.
const MAX_LOCAL_LABELS_PER_LAYER: usize = 512;

/// Hard cap on LOCAL Symbol-layer label candidates for the WHOLE export.
///
/// A page holds a few hundred labels at most, so this is generous — its job
/// is to bound the font plan (which SHAPES every distinct string) and the
/// greedy collision loop against a 100 000-feature labelled layer. Candidates
/// are cut in placement-priority order — top-most layer first, then by
/// importance within the layer — so what the cap drops is what would have
/// lost its collision anyway.
const MAX_LOCAL_LABELS: usize = 2048;

/// Which alpha ExtGState a placed label draws under.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum LabelAlpha {
    /// A local layer: `GS{n}`, index into `PrintRequest::layers`.
    Layer(usize),
    /// A streamed vector-tile paint rule of one source: `GV{n}` for the
    /// legacy single slot, `GV{entry}s{n}` for a stack entry.
    VectorRule {
        /// Which stack entry the rule belongs to; [`None`] for
        /// `PrintRequest::vector`.
        entry: Option<usize>,
        /// The rule's index within that source's paint table.
        rule: usize,
    },
}

impl LabelAlpha {
    /// The ExtGState resource name — both families are already registered
    /// by `pdf_document`, so no new PDF objects are involved.
    pub(super) fn name(self) -> String {
        match self {
            Self::Layer(index) => super::alpha_name(index),
            Self::VectorRule { entry, rule } => super::mvt::rule_alpha_name(entry, rule),
        }
    }
}

/// One label that survived placement, ready to draw.
pub(super) struct PlacedLabel {
    /// Which alpha ExtGState the label draws under.
    pub alpha: LabelAlpha,
    /// The label text.
    pub text: String,
    /// Left edge of the baseline start, in page points.
    pub x: f32,
    /// Baseline y, in page points.
    pub y: f32,
    /// Font size, in points.
    pub size: f32,
    /// Fill color.
    pub color: Color,
    /// Halo (stroke) color and stroke width in points, when styled.
    pub halo: Option<(Color, f32)>,
    /// Which weight the label draws at (print/text v1.4) — the style's own
    /// [`SymbolStyle::weight`], carried through placement because the
    /// collision box has to be measured at the SAME weight the page renders.
    pub weight: LabelWeight,
    /// Whether the label draws as a stacked VERTICAL column (print v1.6).
    /// [`true`] only when the style asked for it AND the plan holds an
    /// accepted [`super::vertical::VerticalLine`] for this
    /// `(weight, text)` — a refusal falls back to the horizontal form, so
    /// `x`/`y` always mean what the matching emitter reads.
    pub vertical: bool,
}

/// One local Symbol-layer label candidate, already projected to the page.
struct LocalCandidate {
    /// Index into [`PrintRequest::layers`] — the `GS{n}` alpha state.
    layer: usize,
    /// The label text.
    text: String,
    /// Page-point anchor.
    anchor: (f32, f32),
    /// Whether the anchor is a point marker (the label offsets upward).
    point_anchor: bool,
    /// Page font size, in points.
    size: f32,
    /// Fill color.
    color: Color,
    /// Halo (stroke) color and width, when styled.
    halo: Option<(Color, f32)>,
    /// Which weight the label draws at.
    weight: LabelWeight,
    /// Whether the style asked for a vertical column.
    vertical: bool,
}

/// One string the font plan has to cover, with everything the plan needs to
/// decide HOW to cover it.
///
/// The plan is keyed by `(weight, text)` and — since v1.6 — also holds a
/// vertical column per `(weight, text)` that asked for one, so the candidate
/// builders have to hand both facts over together. A vertical string is
/// planned BOTH ways: the ladder can refuse it, and the fallback is the
/// ordinary horizontal line.
pub(super) struct PlanText {
    /// Which face chain draws the string.
    pub weight: LabelWeight,
    /// The string itself.
    pub text: String,
    /// Whether the style asked for a vertical column.
    pub vertical: bool,
}

/// The local Symbol-layer label candidates, in PLACEMENT order — top-most
/// layer first, then by importance within the layer — projected, filtered to
/// the map box and capped at [`MAX_LOCAL_LABELS_PER_LAYER`] per layer and
/// [`MAX_LOCAL_LABELS`] overall.
///
/// ONE builder feeds both [`texts`] (the font plan) and [`place`] (the boxes),
/// exactly as [`mvt_candidates`] already did for the streamed path, so a
/// placed label can never miss its CIDs.
///
/// **The map-box filter loses nothing.** A label's box is centred on its
/// anchor and [`try_place`] requires the WHOLE padded box inside the map box,
/// so an anchor outside the box could never have been placed — filtering here
/// merely stops the plan from shaping 100 000 strings to draw a hundred.
fn local_candidates(
    request: &PrintRequest,
    compose: &MapView,
    map_box: &MapBox,
) -> Vec<LocalCandidate> {
    let scale = page_scale(request, map_box);
    let mut out: Vec<LocalCandidate> = Vec::new();
    let mut dropped = 0_usize;
    let mut capped = false;
    // Top-most layer first: the same layers that draw over the others also
    // win the label collisions.
    for (layer_index, layer) in request.layers.iter().enumerate().rev() {
        // Labels are a layer-wide concern living in the BASE slot (a family
        // override is never Symbol through the panel).
        let LayerStyle::Symbol(symbol) = layer.style.base() else {
            continue;
        };
        let Some(field) = symbol.text_field.as_deref() else {
            continue;
        };
        let (size, halo) = symbol_geometry(symbol, scale);
        let mut ranked: Vec<(f64, LocalCandidate)> = Vec::new();
        for feature in &layer.features.features {
            let Some(text) = local_label_text(feature.properties.as_ref(), field) else {
                continue;
            };
            let Some(geometry) = feature.geometry.as_ref() else {
                continue;
            };
            let Some((anchor_position, importance, point_anchor)) = anchor(geometry) else {
                continue;
            };
            let Some(anchor) = project(compose, map_box, &anchor_position) else {
                continue;
            };
            if !inside_map_box(map_box, anchor) {
                dropped += 1;
                continue;
            }
            ranked.push((
                importance,
                LocalCandidate {
                    layer: layer_index,
                    text,
                    anchor,
                    point_anchor,
                    size,
                    color: symbol.text_color,
                    halo,
                    weight: symbol.weight(),
                    vertical: !symbol.orientation().is_horizontal(),
                },
            ));
        }
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        if ranked.len() > MAX_LOCAL_LABELS_PER_LAYER {
            capped = true;
            ranked.truncate(MAX_LOCAL_LABELS_PER_LAYER);
        }
        out.extend(ranked.into_iter().map(|(_, candidate)| candidate));
    }
    if out.len() > MAX_LOCAL_LABELS {
        capped = true;
        out.truncate(MAX_LOCAL_LABELS);
    }
    if capped {
        tracing::warn!(
            per_layer = MAX_LOCAL_LABELS_PER_LAYER,
            total = MAX_LOCAL_LABELS,
            "oxigis-ui print: local label candidates capped; each full layer's \
             lowest-priority tail is dropped",
        );
    }
    tracing::debug!(
        candidates = out.len(),
        dropped,
        "oxigis-ui print: local label candidates outside the map box were dropped before planning",
    );
    out
}

/// Points on the page per logical screen pixel: labels keep the same
/// proportion of the map they had on screen.
fn page_scale(request: &PrintRequest, map_box: &MapBox) -> f32 {
    let screen_w = request.view.size_px()[0];
    if screen_w > 0.0 {
        map_box.width / screen_w
    } else {
        0.5
    }
}

/// Whether a page-point anchor lies on the map box.
fn inside_map_box(map_box: &MapBox, anchor: (f32, f32)) -> bool {
    anchor.0 >= map_box.x
        && anchor.0 <= map_box.x + map_box.width
        && anchor.1 >= map_box.y
        && anchor.1 <= map_box.y + map_box.height
}

/// Every string the Symbol layers will draw — input for the font plan, so
/// each label's characters get CIDs before placement measures anything.
///
/// Derived from the SAME candidate list [`place`] walks (v1.6): before that,
/// every feature of every Symbol layer in the project was shaped into the
/// plan, viewport or no viewport, cap or no cap.
pub(super) fn texts(request: &PrintRequest, compose: &MapView, map_box: &MapBox) -> Vec<PlanText> {
    local_candidates(request, compose, map_box)
        .into_iter()
        .map(|candidate| PlanText {
            weight: candidate.weight,
            text: candidate.text,
            vertical: candidate.vertical,
        })
        .collect()
}

/// One streamed-tile label candidate, before page projection.
struct MvtCandidate {
    /// Index of the Symbol rule in `paints`, for the `GV{n}` alpha state.
    rule: usize,
    /// The label text, from the rule's `text_field`.
    text: String,
    /// Which tile the anchor lives in.
    tile: TileId,
    /// Tile-local anchor position (`y` down), from the render crate's
    /// [`feature_anchor`] — the SAME rule the screen places by.
    position: [f64; 2],
    /// The layer's extent grid.
    extent: u32,
    /// Whether the anchor is a point marker (label offsets upward).
    point_anchor: bool,
    /// Ranking key within the rule, larger first.
    priority: f64,
}

/// The streamed vector-tile label candidates, in deterministic
/// (rule, tile, feature) order, capped at [`MAX_MVT_LABELS`].
///
/// ONE builder feeds both [`mvt_texts`] (the font plan) and [`place`] (the
/// boxes), so a placed label can never miss its CIDs. The screen's seam rule
/// is applied verbatim: an anchor outside `0..=extent` belongs to the
/// neighbouring tile and is dropped — the primary cross-tile de-duplicator.
/// (A feature the TILER split carries one anchor per piece and can label
/// twice, exactly as on screen; no feature-id table on either side.)
fn mvt_candidates(
    paints: &[VectorTilePaint],
    tiles: &[(TileId, Arc<MvtVectorTile>)],
) -> Vec<MvtCandidate> {
    let mut out = Vec::new();
    let mut capped = false;
    'rules: for (rule, paint) in paints.iter().enumerate() {
        let LayerStyle::Symbol(symbol) = &paint.style else {
            continue;
        };
        let Some(field) = symbol.text_field.as_deref() else {
            continue;
        };
        let mut in_rule = 0_usize;
        for (tile, decoded) in tiles {
            let Some(layer) = decoded
                .layers
                .iter()
                .find(|layer| layer.name == paint.source_layer)
            else {
                continue;
            };
            for feature in &layer.features {
                let Some(text) = feature
                    .properties
                    .iter()
                    .find(|(key, _)| key == field)
                    .and_then(|(_, value)| label_text(value))
                else {
                    continue;
                };
                let Some(anchor) = feature_anchor(&feature.geometry) else {
                    continue;
                };
                let extent = f64::from(layer.extent.max(1));
                // Buffer-zone spill belongs to the neighbouring tile — the
                // screen's rule, verbatim.
                if anchor.position[0] < 0.0
                    || anchor.position[0] > extent
                    || anchor.position[1] < 0.0
                    || anchor.position[1] > extent
                {
                    continue;
                }
                if in_rule >= MAX_MVT_LABELS_PER_RULE {
                    // This RULE is full: its tail is dropped, the next rule
                    // still gets its own budget.
                    capped = true;
                    continue 'rules;
                }
                if out.len() >= MAX_MVT_LABELS {
                    capped = true;
                    break 'rules;
                }
                in_rule += 1;
                out.push(MvtCandidate {
                    rule,
                    text,
                    tile: *tile,
                    position: anchor.position,
                    extent: layer.extent,
                    point_anchor: matches!(anchor.kind, oxigis_render::AnchorKind::Point),
                    priority: anchor.priority,
                });
            }
        }
    }
    if capped {
        tracing::warn!(
            per_rule = MAX_MVT_LABELS_PER_RULE,
            total = MAX_MVT_LABELS,
            "oxigis-ui print: streamed-tile label candidates capped; each full rule's tail is \
             dropped",
        );
    }
    out
}

/// Every string the streamed vector-tile Symbol rules will draw — appended
/// to the font plan BEFORE placement, like the local labels.
pub(super) fn mvt_texts(
    paints: &[VectorTilePaint],
    tiles: &[(TileId, Arc<MvtVectorTile>)],
) -> Vec<PlanText> {
    mvt_candidates(paints, tiles)
        .into_iter()
        .map(|candidate| {
            let (weight, vertical) = match paints.get(candidate.rule).map(|paint| &paint.style) {
                Some(LayerStyle::Symbol(symbol)) => {
                    (symbol.weight(), !symbol.orientation().is_horizontal())
                }
                _ => (LabelWeight::Regular, false),
            };
            PlanText {
                weight,
                text: candidate.text,
                vertical,
            }
        })
        .collect()
}

/// One placement attempt, shared by the local and the streamed-tile passes
/// so both go through the SAME box rules.
struct Attempt {
    alpha: LabelAlpha,
    text: String,
    anchor: (f32, f32),
    point_anchor: bool,
    size: f32,
    color: Color,
    halo: Option<(Color, f32)>,
    weight: LabelWeight,
    /// Whether the plan holds an accepted vertical column for this label —
    /// resolved by the caller, so a refusal has already become a horizontal
    /// attempt by the time the box maths runs.
    vertical: bool,
}

/// The pen origin and the collision box one attempt occupies: `(x, y)` are
/// exactly what the matching emitter reads.
struct LabelBox {
    /// Emitter origin x: the baseline start horizontally, the em cell's LEFT
    /// edge vertically.
    x: f32,
    /// Emitter origin y: the baseline vertically, the FIRST cell's baseline
    /// for a column.
    y: f32,
    /// `[min_x, min_y, max_x, max_y]`, already padded.
    aabb: [f32; 4],
}

/// The horizontal box: exact `/W` width, centred on the anchor, ascending
/// from a baseline that clears a point marker.
fn horizontal_box(plan: &TextPlan, attempt: &Attempt) -> Option<LabelBox> {
    let width = plan.width_pt(attempt.weight, &attempt.text, attempt.size);
    if width <= 0.0 {
        return None;
    }
    let x = attempt.anchor.0 - width / 2.0;
    // Point markers get the label just above them; area/line labels sit
    // visually centered on the anchor.
    let y = if attempt.point_anchor {
        attempt.anchor.1 + 3.0
    } else {
        attempt.anchor.1 - attempt.size * 0.35
    };
    Some(LabelBox {
        x,
        y,
        aabb: [
            x - COLLISION_PAD_PT,
            y - attempt.size * 0.3 - COLLISION_PAD_PT,
            x + width + COLLISION_PAD_PT,
            y + attempt.size + COLLISION_PAD_PT,
        ],
    })
}

/// The vertical box (print v1.6): one em wide and the summed cell pitch
/// tall — the SAME rectangle the on-screen engine reserves for a column —
/// centred on the anchor, or hung above a point marker as the horizontal
/// twin is.
fn vertical_box(plan: &TextPlan, attempt: &Attempt) -> Option<LabelBox> {
    let line = plan.vertical_line(attempt.weight, &attempt.text)?;
    let [width, height] = line.box_pt(attempt.size);
    if width <= 0.0 || height <= 0.0 {
        return None;
    }
    let left = attempt.anchor.0 - width / 2.0;
    let top = if attempt.point_anchor {
        attempt.anchor.1 + 3.0 + height
    } else {
        attempt.anchor.1 + height / 2.0
    };
    Some(LabelBox {
        x: left,
        y: top - column_top_offset(line, attempt.size),
        aabb: [
            left - COLLISION_PAD_PT,
            top - height - COLLISION_PAD_PT,
            left + width + COLLISION_PAD_PT,
            top + COLLISION_PAD_PT,
        ],
    })
}

/// How far the column's em-box TOP sits above the origin
/// [`super::emit::show_vertical_line`] steps from, in points.
///
/// The two item kinds are positioned differently and the difference is a
/// whole em, so it cannot be papered over with one constant: an upright cell
/// is drawn on its horizontal baseline (box top one ascent ABOVE the origin),
/// while a rotated run's text matrix sends text `+x` to page `(0, −1)`, so it
/// advances straight down FROM the origin.
fn column_top_offset(line: &super::vertical::VerticalLine, size: f32) -> f32 {
    match line.items.first() {
        Some(super::vertical::VerticalItem::Upright(_)) => VERTICAL_CELL_ASCENT_EM * size,
        _ => 0.0,
    }
}

/// Greedy placement of one attempt against the shared collision pool:
/// exact measured box, fully inside the map box, first-come-wins. Horizontal
/// and vertical labels share ONE pool, because on paper they share the page.
fn try_place(
    plan: &TextPlan,
    map_box: &MapBox,
    boxes: &mut Vec<[f32; 4]>,
    placed: &mut Vec<PlacedLabel>,
    attempt: Attempt,
) {
    let measured = if attempt.vertical {
        vertical_box(plan, &attempt)
    } else {
        horizontal_box(plan, &attempt)
    };
    let Some(LabelBox { x, y, aabb }) = measured else {
        return;
    };
    let inside = aabb[0] >= map_box.x
        && aabb[1] >= map_box.y
        && aabb[2] <= map_box.x + map_box.width
        && aabb[3] <= map_box.y + map_box.height;
    if !inside {
        return;
    }
    let collides = boxes.iter().any(|other| {
        aabb[0] < other[2] && other[0] < aabb[2] && aabb[1] < other[3] && other[1] < aabb[3]
    });
    if collides {
        return;
    }
    boxes.push(aabb);
    placed.push(PlacedLabel {
        alpha: attempt.alpha,
        text: attempt.text,
        x,
        y,
        size: attempt.size,
        color: attempt.color,
        halo: attempt.halo,
        weight: attempt.weight,
        vertical: attempt.vertical,
    });
}

/// A Symbol style's page-scaled size and halo.
fn symbol_geometry(symbol: &SymbolStyle, scale: f32) -> (f32, Option<(Color, f32)>) {
    let size = (symbol.text_size() * scale).max(MIN_LABEL_PT);
    let halo = symbol.halo_color.and_then(|color| {
        let width = 2.0 * symbol.halo_width() * scale;
        (width > 0.05).then_some((color, width.max(0.4)))
    });
    (size, halo)
}

/// Places every label: the local Symbol layers first (they paint on top of
/// the streamed tiles, so they win the collisions too), then the streamed
/// vector-tile Symbol rules — one shared collision pool, priority order
/// within each source, greedy boxes, fully inside the map box.
pub(super) fn place(
    request: &PrintRequest,
    compose: &MapView,
    map_box: &MapBox,
    plan: &TextPlan,
    tiles: &super::PrintVectorTiles<'_>,
) -> Vec<PlacedLabel> {
    let scale = page_scale(request, map_box);
    let mut placed: Vec<PlacedLabel> = Vec::new();
    let mut boxes: Vec<[f32; 4]> = Vec::new();
    // Vertical styles the plan's refusal ladder turned down: they print
    // horizontally, and the export says so ONCE — the `refused_bidi` honesty
    // idiom. A label the ladder ACCEPTED is not counted; it really is a
    // column on the page.
    let mut refused_vertical = 0_usize;
    // The local layers first — they paint on top of the streamed tiles, so
    // they win the collisions too. One builder feeds the plan and this pass,
    // so a placed label can never miss its CIDs.
    for candidate in local_candidates(request, compose, map_box) {
        let vertical = resolve_vertical(
            plan,
            candidate.vertical,
            candidate.weight,
            &candidate.text,
            &mut refused_vertical,
        );
        try_place(
            plan,
            map_box,
            &mut boxes,
            &mut placed,
            Attempt {
                alpha: LabelAlpha::Layer(candidate.layer),
                text: candidate.text,
                anchor: candidate.anchor,
                point_anchor: candidate.point_anchor,
                size: candidate.size,
                color: candidate.color,
                halo: candidate.halo,
                weight: candidate.weight,
                vertical,
            },
        );
    }

    // The streamed vector-tile Symbol rules, after the locals — mirroring
    // the paint order (tiles draw UNDER the local layers). Priority sorts
    // within the whole candidate set; the units are tile-local and only
    // honestly comparable within one rule of one tile (the same caveat the
    // screen documents), which the deterministic collection order keeps
    // harmless.
    for (entry, config) in request.vector_sources() {
        let mut candidates = mvt_candidates(&config.paints, tiles.of(entry));
        candidates.sort_by(|a, b| {
            (a.rule.cmp(&b.rule)).then(
                b.priority
                    .partial_cmp(&a.priority)
                    .unwrap_or(core::cmp::Ordering::Equal),
            )
        });
        for candidate in candidates {
            let Some(LayerStyle::Symbol(symbol)) =
                config.paints.get(candidate.rule).map(|paint| &paint.style)
            else {
                continue;
            };
            let Some(frame) = TileFrame::for_tile(compose, map_box, candidate.tile) else {
                continue;
            };
            let anchor = frame.project_f64(candidate.position, candidate.extent);
            let (size, halo) = symbol_geometry(symbol, scale);
            let vertical = resolve_vertical(
                plan,
                !symbol.orientation().is_horizontal(),
                symbol.weight(),
                &candidate.text,
                &mut refused_vertical,
            );
            try_place(
                plan,
                map_box,
                &mut boxes,
                &mut placed,
                Attempt {
                    alpha: LabelAlpha::VectorRule {
                        entry,
                        rule: candidate.rule,
                    },
                    text: candidate.text,
                    anchor,
                    point_anchor: candidate.point_anchor,
                    size,
                    color: symbol.text_color,
                    halo,
                    weight: symbol.weight(),
                    vertical,
                },
            );
        }
    }
    if refused_vertical > 0 {
        tracing::warn!(
            labels = refused_vertical,
            "oxigis-ui print: labels ask for VERTICAL text the export's refusal ladder turned \
             down (the reason is logged once by the font plan); those labels print \
             horizontally, so the map and the page disagree for them",
        );
    }
    placed
}

/// Whether one label really draws as a column: the style asked for it AND
/// the plan holds an accepted line for this exact `(weight, text)`. A refusal
/// is counted for the one aggregated log and falls back to horizontal.
fn resolve_vertical(
    plan: &TextPlan,
    requested: bool,
    weight: LabelWeight,
    text: &str,
    refused: &mut usize,
) -> bool {
    if !requested {
        return false;
    }
    if plan.vertical_line(weight, text).is_some() {
        return true;
    }
    *refused += 1;
    false
}

/// The label text for one LOCAL feature: the field's string verbatim, or a
/// number/bool stringified; anything else (arrays, objects, null, a missing
/// field) draws nothing. (The streamed-tile twin is the render crate's
/// [`label_text`] over [`oxigis_render::mvt::MvtValue`].)
fn local_label_text(
    properties: Option<&serde_json::Map<String, serde_json::Value>>,
    field: &str,
) -> Option<String> {
    let value = properties?.get(field)?;
    let text = match value {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Number(number) => number.to_string(),
        serde_json::Value::Bool(flag) => flag.to_string(),
        _ => return None,
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// The label anchor for one geometry, with its collision priority and
/// whether it is a point marker (which offsets the label upward).
///
/// `GeometryCollection`s recurse and keep the most important member.
fn anchor(geometry: &Geometry) -> Option<(Position, f64, bool)> {
    if let Geometry::GeometryCollection(collection) = geometry {
        return collection
            .geometries
            .iter()
            .filter_map(anchor)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    }
    let paths = command::paths(geometry);
    // Largest exterior ring first: polygons label at their centroid.
    let mut best_ring: Option<(f64, Position)> = None;
    let mut best_line: Option<(f64, Position)> = None;
    let mut first_point: Option<Position> = None;
    for path in &paths {
        match path.kind {
            PathKind::Ring if path.ring == 0 => {
                if let Some((area, centroid)) = ring_centroid(path.positions)
                    && best_ring.as_ref().is_none_or(|(best, _)| area > *best)
                {
                    best_ring = Some((area, centroid));
                }
            }
            PathKind::Ring => {}
            PathKind::Line => {
                if let Some((length, midpoint)) = line_midpoint(path.positions)
                    && best_line.as_ref().is_none_or(|(best, _)| length > *best)
                {
                    best_line = Some((length, midpoint));
                }
            }
            PathKind::Points => {
                if first_point.is_none() {
                    first_point = path.positions.first().cloned();
                }
            }
        }
    }
    if let Some((area, centroid)) = best_ring {
        return Some((centroid, area, false));
    }
    if let Some((length, midpoint)) = best_line {
        return Some((midpoint, length, false));
    }
    first_point.map(|position| (position, 0.0, true))
}

/// Shoelace |area| and centroid of an **open** ring; mean fallback when the
/// area degenerates (collinear ring).
fn ring_centroid(positions: &[Position]) -> Option<(f64, Position)> {
    let points: Vec<(f64, f64)> = positions
        .iter()
        .filter_map(|position| Some((*position.first()?, *position.get(1)?)))
        .collect();
    if points.len() < 3 {
        return None;
    }
    let mut doubled_area = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for index in 0..points.len() {
        let (x0, y0) = points[index];
        let (x1, y1) = points[(index + 1) % points.len()];
        let cross = x0 * y1 - x1 * y0;
        doubled_area += cross;
        cx += (x0 + x1) * cross;
        cy += (y0 + y1) * cross;
    }
    let area = doubled_area / 2.0;
    if area.abs() < 1e-12 {
        let count = points.len() as f64;
        let (sx, sy) = points
            .iter()
            .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x, sy + y));
        return Some((0.0, vec![sx / count, sy / count]));
    }
    Some((area.abs(), vec![cx / (6.0 * area), cy / (6.0 * area)]))
}

/// Total length and arc-length midpoint of a line's positions.
fn line_midpoint(positions: &[Position]) -> Option<(f64, Position)> {
    let points: Vec<(f64, f64)> = positions
        .iter()
        .filter_map(|position| Some((*position.first()?, *position.get(1)?)))
        .collect();
    if points.len() < 2 {
        return None;
    }
    let mut lengths = Vec::with_capacity(points.len() - 1);
    let mut total = 0.0;
    for pair in points.windows(2) {
        let dx = pair[1].0 - pair[0].0;
        let dy = pair[1].1 - pair[0].1;
        let length = (dx * dx + dy * dy).sqrt();
        lengths.push(length);
        total += length;
    }
    if total <= 0.0 {
        return Some((0.0, vec![points[0].0, points[0].1]));
    }
    let mut remaining = total / 2.0;
    for (index, length) in lengths.iter().enumerate() {
        if remaining <= *length {
            let t = if *length > 0.0 {
                remaining / length
            } else {
                0.0
            };
            let (x0, y0) = points[index];
            let (x1, y1) = points[index + 1];
            return Some((total, vec![x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]));
        }
        remaining -= length;
    }
    Some((
        total,
        vec![points[points.len() - 1].0, points[points.len() - 1].1],
    ))
}

#[cfg(test)]
mod tests {
    use super::super::{PrintLayer, compose_view, map_box, raster_size_px};
    use super::*;
    use crate::tile_provider::BasemapConfig;
    use oxigeo::geojson::types::{Feature, FeatureCollection, LineString, Point, Polygon};
    use oxigis_core::SymbolStyle;
    use std::sync::Arc;

    fn labeled_points(entries: &[(f64, f64, &str)]) -> PrintLayer {
        let features = entries
            .iter()
            .map(|&(lon, lat, name)| {
                let mut properties = serde_json::Map::new();
                properties.insert("name".to_string(), serde_json::Value::String(name.into()));
                Feature::new(
                    Some(Geometry::Point(Point::new_2d(lon, lat).expect("a point"))),
                    Some(properties),
                )
            })
            .collect();
        let features = Arc::new(FeatureCollection::new(features));
        PrintLayer {
            name: String::new(),
            families: crate::local_vector::collection_families(&features),
            features,
            style: LayerStyle::Symbol(SymbolStyle::new("name")).into(),
            opacity: 1.0,
        }
    }

    fn request_with(layers: Vec<PrintLayer>) -> (PrintRequest, MapView, MapBox) {
        let view = MapView::new(oxigis_render::LonLat::new(0.0, 0.0), 2.0, [800.0, 600.0])
            .expect("a valid viewport");
        let options = super::super::PrintOptions::default();
        let map_box = map_box(&options);
        let out_px = raster_size_px(&map_box, &options);
        let compose = compose_view(view, out_px);
        let request = PrintRequest {
            // No tiled stack: these fixtures exercise the local-vector and
            // furniture halves of the page, which the snapshot does not reach.
            stack: Vec::new(),
            title: "labels".to_string(),
            attribution: String::new(),
            view,
            basemap: BasemapConfig::openstreetmap(),
            cog: None,
            archive: None,
            vector: None,
            layers,
            options,
        };
        (request, compose, map_box)
    }

    /// The label strings as `(weight, text)` pairs — what most assertions
    /// care about, now that the builder also carries orientation.
    fn weighted(
        request: &PrintRequest,
        compose: &MapView,
        map_box: &MapBox,
    ) -> Vec<(LabelWeight, String)> {
        texts(request, compose, map_box)
            .into_iter()
            .map(|planned| (planned.weight, planned.text))
            .collect()
    }

    fn latin_plan(request: &PrintRequest, compose: &MapView, map_box: &MapBox) -> TextPlan {
        let fonts =
            super::super::PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
        let label_texts = texts(request, compose, map_box);
        let mut plan_texts: Vec<(LabelWeight, &str)> =
            vec![(LabelWeight::Regular, request.title.as_str())];
        plan_texts.extend(
            label_texts
                .iter()
                .map(|planned| (planned.weight, planned.text.as_str())),
        );
        let vertical: Vec<(LabelWeight, &str)> = label_texts
            .iter()
            .filter(|planned| planned.vertical)
            .map(|planned| (planned.weight, planned.text.as_str()))
            .collect();
        super::super::font::plan_with_verticals(&fonts, &plan_texts, None, &vertical)
            .expect("a plan")
    }

    #[test]
    fn texts_collects_symbol_layer_fields_and_stringifies_numbers() {
        let mut layer = labeled_points(&[(0.0, 0.0, "Tokyo")]);
        // A numeric label value must survive as text.
        let mut properties = serde_json::Map::new();
        properties.insert(
            "name".to_string(),
            serde_json::Value::Number(serde_json::Number::from(42)),
        );
        let numbered = Feature::new(
            Some(Geometry::Point(Point::new_2d(1.0, 1.0).expect("a point"))),
            Some(properties),
        );
        let mut features = (*layer.features).clone();
        features.features.push(numbered);
        layer.features = Arc::new(features);
        let (request, compose, map_box) = request_with(vec![layer]);
        assert_eq!(
            weighted(&request, &compose, &map_box),
            vec![
                (LabelWeight::Regular, "Tokyo".to_string()),
                (LabelWeight::Regular, "42".to_string()),
            ],
        );
    }

    #[test]
    fn non_symbol_layers_produce_no_texts() {
        let mut layer = labeled_points(&[(0.0, 0.0, "Tokyo")]);
        layer.style = LayerStyle::Circle(oxigis_core::CircleStyle::new(3.0, Color::BLACK)).into();
        let (request, compose, map_box) = request_with(vec![layer]);
        assert!(texts(&request, &compose, &map_box).is_empty());
    }

    #[test]
    fn distant_labels_both_place_and_identical_anchors_collide() {
        let (request, compose, map_box) = request_with(vec![labeled_points(&[
            (-40.0, 0.0, "West"),
            (40.0, 0.0, "East"),
        ])]);
        let plan = latin_plan(&request, &compose, &map_box);
        let placed = place(
            &request,
            &compose,
            &map_box,
            &plan,
            &super::super::PrintVectorTiles::default(),
        );
        assert_eq!(placed.len(), 2, "distant labels must both land");

        let (request, compose, map_box) = request_with(vec![labeled_points(&[
            (0.0, 0.0, "Same"),
            (0.0, 0.0, "Same"),
        ])]);
        let plan = latin_plan(&request, &compose, &map_box);
        let placed = place(
            &request,
            &compose,
            &map_box,
            &plan,
            &super::super::PrintVectorTiles::default(),
        );
        assert_eq!(placed.len(), 1, "the second identical anchor must lose");
    }

    #[test]
    fn a_line_labels_at_its_arc_length_midpoint() {
        let line = LineString::new(vec![vec![-20.0, 10.0], vec![20.0, 10.0]]).expect("a line");
        let (_, _, midpoint_is_point) =
            anchor(&Geometry::LineString(line.clone())).expect("an anchor");
        assert!(!midpoint_is_point);
        let (anchor_position, _, _) = anchor(&Geometry::LineString(line)).expect("an anchor");
        assert!((anchor_position[0] - 0.0).abs() < 1e-9);
        assert!((anchor_position[1] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn a_polygon_labels_at_its_shoelace_centroid() {
        let ring = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![10.0, 10.0],
            vec![0.0, 10.0],
            vec![0.0, 0.0],
        ];
        let polygon = Polygon::from_exterior(ring).expect("a ring");
        let (position, importance, point_anchor) =
            anchor(&Geometry::Polygon(polygon)).expect("an anchor");
        assert!((position[0] - 5.0).abs() < 1e-9);
        assert!((position[1] - 5.0).abs() < 1e-9);
        assert!(importance > 0.0);
        assert!(!point_anchor);
    }

    #[test]
    fn labels_outside_the_map_box_are_dropped() {
        // Latitude 84 projects far above the default framing's map box.
        let (request, compose, map_box) =
            request_with(vec![labeled_points(&[(0.0, 84.0, "Arctic")])]);
        let plan = latin_plan(&request, &compose, &map_box);
        assert!(
            place(
                &request,
                &compose,
                &map_box,
                &plan,
                &super::super::PrintVectorTiles::default()
            )
            .is_empty()
        );
    }

    #[test]
    fn the_top_layer_wins_the_collision_against_the_bottom_layer() {
        let bottom = labeled_points(&[(0.0, 0.0, "Bottom")]);
        let top = labeled_points(&[(0.0, 0.0, "Top")]);
        // Stack order is bottom-first, so `top` is the later entry.
        let (request, compose, map_box) = request_with(vec![bottom, top]);
        let plan = latin_plan(&request, &compose, &map_box);
        let placed = place(
            &request,
            &compose,
            &map_box,
            &plan,
            &super::super::PrintVectorTiles::default(),
        );
        assert_eq!(placed.len(), 1);
        assert_eq!(placed[0].text, "Top");
        assert_eq!(placed[0].alpha, LabelAlpha::Layer(1));
    }

    /// One decoded tile whose `place` layer holds one named point at the
    /// tile-local `position`.
    fn labeled_tile(position: [i32; 2], name: &str) -> (TileId, Arc<MvtVectorTile>) {
        use oxigis_render::mvt::{MvtFeature, MvtGeometry, MvtLayer, MvtValue};
        (
            TileId { z: 0, x: 0, y: 0 },
            Arc::new(MvtVectorTile {
                layers: vec![MvtLayer {
                    name: "place".to_string(),
                    extent: 4096,
                    features: vec![MvtFeature {
                        id: None,
                        properties: vec![("name".to_string(), MvtValue::String(name.to_string()))],
                        geometry: MvtGeometry::Points(vec![position]),
                    }],
                }],
            }),
        )
    }

    /// A vector-tile config with one labelling Symbol rule over `place`.
    fn symbol_config() -> crate::vector_provider::VectorTileConfig {
        crate::vector_provider::VectorTileConfig {
            url_template: "https://example.invalid/{z}/{x}/{y}.pbf".to_string(),
            subdomains: Vec::new(),
            attribution: String::new(),
            paints: vec![VectorTilePaint::new(
                "place",
                LayerStyle::Symbol(SymbolStyle::new("name")),
            )],
            archive: None,
        }
    }

    #[test]
    fn a_streamed_tile_symbol_rule_places_a_label_under_its_rule_alpha() {
        let (mut request, compose, map_box) = request_with(Vec::new());
        request.vector = Some(symbol_config());
        let tiles = vec![labeled_tile([2048, 2048], "Tokyo")];
        // The same candidate builder feeds the plan and the placement.
        let config = request.vector.clone().expect("a config");
        let tile_texts = mvt_texts(&config.paints, &tiles);
        assert_eq!(
            tile_texts
                .iter()
                .map(|planned| (planned.weight, planned.text.clone()))
                .collect::<Vec<_>>(),
            vec![(LabelWeight::Regular, "Tokyo".to_string())],
        );
        let fonts =
            super::super::PrintFonts::new(vec![oxifont_bundled::NOTO_SANS_REGULAR.to_vec()]);
        let mut plan_texts: Vec<(LabelWeight, &str)> =
            vec![(LabelWeight::Regular, request.title.as_str())];
        plan_texts.extend(
            tile_texts
                .iter()
                .map(|planned| (planned.weight, planned.text.as_str())),
        );
        let plan = super::super::font::plan(&fonts, &plan_texts, None).expect("a plan");
        let placed = place(
            &request,
            &compose,
            &map_box,
            &plan,
            &super::super::PrintVectorTiles {
                single: &tiles,
                stack: &[],
            },
        );
        assert_eq!(placed.len(), 1, "the tile's point labels once");
        assert_eq!(placed[0].text, "Tokyo");
        assert_eq!(
            placed[0].alpha,
            LabelAlpha::VectorRule {
                entry: None,
                rule: 0
            },
            "the label draws under GV0, the rule's own alpha state",
        );
        assert_eq!(placed[0].alpha.name(), "GV0");
    }

    #[test]
    fn a_buffer_zone_anchor_belongs_to_the_neighbouring_tile() {
        // The screen's seam rule, verbatim: an anchor outside 0..=extent is
        // the neighbour's label, which is what keeps a feature carried in
        // two tiles' buffers from labelling twice.
        let config = symbol_config();
        for position in [[-100, 2048], [4200, 2048], [2048, -8], [2048, 4200]] {
            let tiles = vec![labeled_tile(position, "Spill")];
            assert!(
                mvt_texts(&config.paints, &tiles).is_empty(),
                "anchor at {position:?} must be dropped",
            );
        }
        // And an in-range anchor is kept.
        let tiles = vec![labeled_tile([1, 4095], "Edge")];
        assert_eq!(mvt_texts(&config.paints, &tiles).len(), 1);
    }

    #[test]
    fn a_local_label_wins_the_collision_against_a_streamed_one() {
        // A local Symbol layer and a streamed tile both label the SAME spot
        // (the view centre = the middle of the z0 tile). Locals place first
        // — they paint on top — so the streamed twin must lose.
        let (mut request, compose, map_box) =
            request_with(vec![labeled_points(&[(0.0, 0.0, "Local")])]);
        request.vector = Some(symbol_config());
        let tiles = vec![labeled_tile([2048, 2048], "Local")];
        let plan = latin_plan(&request, &compose, &map_box);
        let placed = place(
            &request,
            &compose,
            &map_box,
            &plan,
            &super::super::PrintVectorTiles {
                single: &tiles,
                stack: &[],
            },
        );
        assert_eq!(placed.len(), 1, "one spot, one label");
        assert_eq!(placed[0].alpha, LabelAlpha::Layer(0));
    }

    /// Turns a fixture layer's Symbol style vertical.
    fn set_vertical(layer: &mut PrintLayer) {
        use oxigis_core::LabelOrientation;
        let LayerStyle::Symbol(mut symbol) = layer.style.base().clone() else {
            panic!("a symbol base");
        };
        symbol.set_orientation(LabelOrientation::Vertical);
        layer.style = LayerStyle::Symbol(symbol).into();
    }

    /// print v1.6: a vertical style really does set a COLUMN on the page.
    ///
    /// Latin is `R` in UAX #50, so `Tokyo` becomes one sideways run — the
    /// rung the bundled (vmtx-less) Noto can actually reach — and the box
    /// stops being a wide horizontal strip: it is one em wide and as tall as
    /// the horizontal line was long.
    #[test]
    fn a_vertical_label_style_is_set_as_a_column() {
        let points = [(0.0, 0.0, "Tokyo")];
        let (flat_request, compose, map_box) = request_with(vec![labeled_points(&points)]);
        let flat_plan = latin_plan(&flat_request, &compose, &map_box);
        let flat = place(
            &flat_request,
            &compose,
            &map_box,
            &flat_plan,
            &super::super::PrintVectorTiles::default(),
        );
        assert_eq!(flat.len(), 1, "the fixture places something");
        assert!(!flat[0].vertical);

        let mut layer = labeled_points(&points);
        set_vertical(&mut layer);
        let (tall_request, compose, map_box) = request_with(vec![layer]);
        let tall_plan = latin_plan(&tall_request, &compose, &map_box);
        let tall = place(
            &tall_request,
            &compose,
            &map_box,
            &tall_plan,
            &super::super::PrintVectorTiles::default(),
        );
        assert_eq!(tall.len(), 1, "the column places too");
        assert!(tall[0].vertical, "the style asked for a column and got one");

        let line = tall_plan
            .vertical_line(tall[0].weight, &tall[0].text)
            .expect("an accepted column");
        let [width, height] = line.box_pt(tall[0].size);
        assert_eq!(width, tall[0].size, "one em wide");
        let flat_width = flat_plan.width_pt(flat[0].weight, &flat[0].text, flat[0].size);
        assert!(
            height > width,
            "a five-letter column is taller than it is wide: {height} vs {width}",
        );
        assert!(
            (height - flat_width).abs() < 0.01,
            "the column is exactly as tall as the line was long: {height} vs {flat_width}",
        );
        // A column hangs from the anchor rather than sitting on a baseline,
        // so the emitter origin is a different point.
        assert!((tall[0].y - flat[0].y).abs() > 0.5);
    }

    /// A refusal costs nothing: the label falls back to the HORIZONTAL box,
    /// byte for byte what the same style without the flag produces.
    #[test]
    fn a_refused_vertical_label_falls_back_to_the_horizontal_box() {
        use oxigis_core::LabelOrientation;

        // CJK with a Latin-only chain: no coverage, so the ladder refuses
        // before it ever asks the face for vertical metrics.
        let points = [
            (0.0, 0.0, "\u{6771}\u{4EAC}"),
            (2.0, 1.0, "\u{4EAC}\u{90FD}"),
        ];
        let (flat_request, compose, map_box) = request_with(vec![labeled_points(&points)]);
        let flat_plan = latin_plan(&flat_request, &compose, &map_box);
        let flat = place(
            &flat_request,
            &compose,
            &map_box,
            &flat_plan,
            &super::super::PrintVectorTiles::default(),
        );

        let mut layer = labeled_points(&points);
        set_vertical(&mut layer);
        let (tall_request, compose, map_box) = request_with(vec![layer]);
        let tall_plan = latin_plan(&tall_request, &compose, &map_box);
        let tall = place(
            &tall_request,
            &compose,
            &map_box,
            &tall_plan,
            &super::super::PrintVectorTiles::default(),
        );

        assert!(!flat.is_empty(), "the fixture places something");
        assert_eq!(tall.len(), flat.len(), "the same labels survive");
        for (tall, flat) in tall.iter().zip(&flat) {
            assert!(!tall.vertical, "the ladder refused, so it stays a line");
            assert_eq!(tall.text, flat.text);
            assert_eq!(tall.x, flat.x, "the same x, to the byte");
            assert_eq!(tall.y, flat.y, "and the same y");
            assert_eq!(tall.size, flat.size);
            assert_eq!(tall.weight, flat.weight);
        }
        // The style really did ask for vertical, so the equality above is not
        // vacuous.
        let LayerStyle::Symbol(symbol) = tall_request.layers[0].style.base() else {
            panic!("a symbol base");
        };
        assert_eq!(symbol.orientation(), LabelOrientation::Vertical);
    }

    /// A page's worth of layers, each dense enough to fill its own budget:
    /// the WHOLE-export cap is what stops the per-layer budgets multiplying
    /// without bound.
    #[test]
    fn many_full_layers_still_stop_at_the_whole_export_cap() {
        let per_layer = MAX_LOCAL_LABELS_PER_LAYER + 40;
        let layers: Vec<Vec<(f64, f64, String)>> = (0..6)
            .map(|layer| {
                (0..per_layer)
                    .map(|index| {
                        let lon = -60.0 + (index % 400) as f64 * 0.3;
                        (lon, 0.0, format!("L{layer}N{index}"))
                    })
                    .collect()
            })
            .collect();
        let built: Vec<PrintLayer> = layers
            .iter()
            .map(|entries| {
                let refs: Vec<(f64, f64, &str)> = entries
                    .iter()
                    .map(|(lon, lat, name)| (*lon, *lat, name.as_str()))
                    .collect();
                labeled_points(&refs)
            })
            .collect();
        let (request, compose, map_box) = request_with(built);
        const {
            assert!(
                6 * MAX_LOCAL_LABELS_PER_LAYER > MAX_LOCAL_LABELS,
                "the fixture has to overflow the whole-export budget to test it",
            );
        }
        assert_eq!(
            texts(&request, &compose, &map_box).len(),
            MAX_LOCAL_LABELS,
            "the per-layer budgets cannot multiply past the export's own cap",
        );
    }

    /// The v1.6 pre-filter: the plan is fed the SAME list the placer walks,
    /// so an off-page feature is never shaped — and the list is capped PER
    /// LAYER, so a dense bottom layer cannot shape 100 000 strings and a
    /// dense layer cannot starve the layers above or below it.
    #[test]
    fn the_plan_sees_only_capped_on_page_candidates() {
        let mut dense: Vec<(f64, f64, String)> = Vec::new();
        for index in 0..(MAX_LOCAL_LABELS_PER_LAYER * 3) {
            let lon = -60.0 + (index % 400) as f64 * 0.3;
            dense.push((lon, 0.0, format!("Dense{index}")));
        }
        // Latitude 84 projects far above the default framing's map box.
        for index in 0..64 {
            dense.push((0.0, 84.0, format!("Off{index}")));
        }
        let sparse: Vec<(f64, f64, String)> = (0..100)
            .map(|index| {
                (
                    -30.0 + f64::from(index) * 0.5,
                    5.0,
                    format!("Sparse{index}"),
                )
            })
            .collect();
        fn as_refs(entries: &[(f64, f64, String)]) -> Vec<(f64, f64, &str)> {
            entries
                .iter()
                .map(|(lon, lat, name)| (*lon, *lat, name.as_str()))
                .collect()
        }
        let dense_refs = as_refs(&dense);
        let sparse_refs = as_refs(&sparse);
        // Stack order is bottom-first, so the sparse layer is on TOP.
        let (request, compose, map_box) = request_with(vec![
            labeled_points(&dense_refs),
            labeled_points(&sparse_refs),
        ]);
        let planned = texts(&request, &compose, &map_box);
        assert_eq!(
            planned.len(),
            MAX_LOCAL_LABELS_PER_LAYER + sparse.len(),
            "each layer is capped on its own, before the plan shapes anything",
        );
        assert!(
            planned
                .iter()
                .all(|planned| !planned.text.starts_with("Off")),
            "an anchor outside the map box could never have been placed",
        );
        assert_eq!(
            planned
                .iter()
                .filter(|planned| planned.text.starts_with("Sparse"))
                .count(),
            sparse.len(),
            "a dense layer must not starve the layer above it",
        );
    }
}
