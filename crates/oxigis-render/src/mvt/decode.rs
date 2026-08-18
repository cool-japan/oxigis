//! `&[u8]` → typed vector tile: validation, tag resolution, geometry streams.
//!
//! [`decode_mvt`] is the default entry point. It parses the protobuf envelope
//! (the crate-private `proto` module), checks every layer against the
//! specification's invariants and turns each feature's command stream into an
//! [`MvtGeometry`] — dropping whatever *fails* those invariants instead of
//! losing the whole tile to it. [`decode_mvt_report`] does the same decode and
//! also hands back a [`DecodeReport`] of what was dropped; [`decode_mvt_strict`]
//! rejects the tile outright on the first such failure, for conformance tests
//! and a validation CLI that want a loud failure instead.
//!
//! # Coordinate space
//!
//! Output coordinates are tile-local integers on the layer's `extent` grid,
//! exactly as encoded. They are *not* clamped to `0..extent`: encoders emit a
//! buffer around the tile so that wide lines and labels join seamlessly across
//! tile seams, and dropping those vertices would create visible gaps. Consumers
//! divide by [`MvtLayer::extent`] to reach the unit tile square.
//!
//! # Documented decisions
//!
//! Where the specification permits more than one reasonable behaviour, this
//! decoder picks the lenient-but-lossless option and records it here:
//!
//! * **Rings are stored unclosed.** `ClosePath` is structural, not a vertex, so
//!   the first point is never repeated at the end of a ring. Consumers that need
//!   a closed sequence append `ring[0]` themselves.
//! * **Zero-area rings are dropped.** A ring whose shoelace area is `0`
//!   (fewer than three distinct points, or fully collinear) can be neither an
//!   exterior nor an interior ring under the specification's sign rule. It is
//!   discarded without touching the classification state, so a following
//!   interior ring still attaches to the last *retained* exterior ring.
//! * **Ring winding is relative, not absolute.** The specification's rule —
//!   positive shoelace area is exterior — only binds version 2 encoders;
//!   version 1 left winding unspecified, and a version 2 encoder can still get
//!   it backwards. Rather than rejecting those tiles, the sign of a geometry's
//!   *first* non-degenerate ring is taken as "exterior" for the rest of that
//!   geometry: a ring matching that sign starts a new polygon, any other sign
//!   becomes a hole in the most recently started one. This is lossless —
//!   `vector::tess` fills and strokes exterior and interior rings identically,
//!   so the grouping used here never changes what is drawn.
//! * **Duplicate property keys keep the first occurrence.** A feature whose tag
//!   list names the same key twice is a sloppy encoder, not a corrupt tile;
//!   later occurrences are ignored rather than rejecting the whole layer.
//! * **Malformed layers and features are isolated, not fatal.** Under
//!   [`decode_mvt`] and [`decode_mvt_report`], a layer or feature that fails
//!   any rule below — an empty geometry stream, an unsupported version, a bad
//!   tag index, and so on — is dropped rather than failing the tile; only an
//!   invalid protobuf envelope does that. [`decode_mvt_strict`] restores the
//!   fail-fast behaviour for every rule at once.
//! * **Empty geometry streams are dropped.** A feature with no geometry cannot
//!   be drawn, and silently producing an empty geometry would push the failure
//!   to a place with less context.
//! * **Empty layer names are dropped**, as is a tile's second layer of a name
//!   already seen (specification §4.1). `prost` cannot distinguish an absent
//!   proto2 `optional string` from a present empty one, so the conservative
//!   reading applies: a layer that cannot be named cannot be styled.

use prost::Message as _;

use crate::error::RenderError;
use crate::mvt::proto;
use crate::mvt::wire::{Command, CommandId, decode_command_integer, zigzag_decode};

/// A decoded Mapbox Vector Tile.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VectorTile {
    /// The tile's layers, in the order the encoder wrote them.
    pub layers: Vec<MvtLayer>,
}

/// One layer of a [`VectorTile`], with its features' properties already
/// resolved against the layer's key/value tables.
#[derive(Debug, Clone, PartialEq)]
pub struct MvtLayer {
    /// Layer name, unique within the tile — the key a style rule matches on.
    pub name: String,
    /// Width and height of the tile-local coordinate grid, `4096` by default.
    pub extent: u32,
    /// The layer's features, in the order the encoder wrote them.
    pub features: Vec<MvtFeature>,
}

/// One feature: an optional identifier, its properties and its geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct MvtFeature {
    /// Encoder-assigned identifier, unique within the layer when present.
    pub id: Option<u64>,
    /// Properties in tag order, first occurrence winning on duplicate keys.
    pub properties: Vec<(String, MvtValue)>,
    /// The feature's geometry in tile-local coordinates.
    pub geometry: MvtGeometry,
}

/// A property value. Mirrors the seven alternatives of `vector_tile.Value`,
/// collapsing its two 64-bit signed encodings onto one Rust type.
#[derive(Debug, Clone, PartialEq)]
pub enum MvtValue {
    /// `string_value`.
    String(String),
    /// `float_value`.
    F32(f32),
    /// `double_value`.
    F64(f64),
    /// `int_value` or `sint_value` — the encodings differ, the values do not.
    I64(i64),
    /// `uint_value`.
    U64(u64),
    /// `bool_value`.
    Bool(bool),
}

/// A feature's geometry. Every variant is inherently "multi": the
/// specification's `POINT` covers point and multipoint, and so on.
#[derive(Debug, Clone, PartialEq)]
pub enum MvtGeometry {
    /// One or more points.
    Points(Vec<[i32; 2]>),
    /// One or more line strings, each with at least two points.
    Lines(Vec<Vec<[i32; 2]>>),
    /// Zero or more polygons — zero only when every ring was degenerate.
    Polygons(Vec<MvtPolygon>),
}

/// A polygon: one exterior ring and the interior rings (holes) that follow it.
///
/// Rings are stored **unclosed** — the closing edge back to the first point is
/// implied by `ClosePath` and never materialised as a repeated vertex.
#[derive(Debug, Clone, PartialEq)]
pub struct MvtPolygon {
    /// The outer boundary; positive shoelace area in MVT's Y-down space.
    pub exterior: Vec<[i32; 2]>,
    /// Holes cut out of [`MvtPolygon::exterior`]; negative shoelace area.
    pub interiors: Vec<Vec<[i32; 2]>>,
}

/// A record of what a lenient decode had to drop, kept small enough to carry
/// around even for a badly malformed tile.
///
/// Every dropped layer or feature is counted; only the *first* reason is kept
/// as text. See [`decode_mvt_report`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DecodeReport {
    /// Layers dropped for failing header validation (unsupported version,
    /// empty or duplicate name, zero extent) — each one otherwise fatal to
    /// the whole tile.
    pub skipped_layers: u32,
    /// Features dropped for failing tag or geometry validation, summed
    /// across every layer that survived.
    pub skipped_features: u32,
    /// The reason the first layer or feature was dropped, if any. Later
    /// reasons are still counted above but not retained, so this field
    /// cannot grow with how malformed the tile is.
    pub first_issue: Option<String>,
}

impl DecodeReport {
    /// Whether every layer and feature in the tile decoded cleanly.
    #[must_use]
    pub const fn is_clean(&self) -> bool {
        self.skipped_layers == 0 && self.skipped_features == 0
    }

    /// Records one dropped layer, keeping only the first reason's text.
    fn skip_layer(&mut self, reason: impl FnOnce() -> String) {
        self.skipped_layers = self.skipped_layers.saturating_add(1);
        if self.first_issue.is_none() {
            self.first_issue = Some(reason());
        }
    }

    /// Records one dropped feature, keeping only the first reason's text.
    fn skip_feature(&mut self, reason: impl FnOnce() -> String) {
        self.skipped_features = self.skipped_features.saturating_add(1);
        if self.first_issue.is_none() {
            self.first_issue = Some(reason());
        }
    }
}

/// Decodes a raw MVT byte stream into a typed tile.
///
/// Malformed layers and features are isolated rather than failing the whole
/// tile — see the module documentation for exactly what that means, and
/// [`decode_mvt_report`] to also learn what, if anything, was dropped.
/// [`decode_mvt_strict`] is the all-or-nothing alternative.
///
/// # Errors
///
/// Returns [`RenderError::Mvt`] only when `bytes` is not a valid
/// `vector_tile.Tile` protobuf message. Every other rule this module defines
/// — an unsupported layer version, a zero extent, a missing or duplicate
/// layer name, an out-of-range tag index, an ambiguous `Value`, a malformed
/// geometry command stream — drops just the layer or feature it belongs to
/// instead of raising an error here.
pub fn decode_mvt(bytes: &[u8]) -> Result<VectorTile, RenderError> {
    decode_mvt_report(bytes).map(|(tile, _report)| tile)
}

/// Decodes a raw MVT byte stream like [`decode_mvt`], additionally reporting
/// what — if anything — was dropped to keep the rest of the tile.
///
/// # Errors
///
/// Returns [`RenderError::Mvt`] under the same condition as [`decode_mvt`]:
/// an invalid protobuf envelope.
pub fn decode_mvt_report(bytes: &[u8]) -> Result<(VectorTile, DecodeReport), RenderError> {
    let tile = proto::Tile::decode(bytes)
        .map_err(|err| RenderError::Mvt(format!("protobuf decode failed: {err}")))?;

    let mut layers: Vec<MvtLayer> = Vec::with_capacity(tile.layers.len());
    let mut report = DecodeReport::default();
    for layer in tile.layers {
        let Some(decoded) = decode_layer_lenient(layer, &mut report) else {
            continue;
        };
        if layers.iter().any(|seen| seen.name == decoded.name) {
            report.skip_layer(|| format!("duplicate layer name {:?}", decoded.name));
            continue;
        }
        layers.push(decoded);
    }
    Ok((VectorTile { layers }, report))
}

/// Decodes a raw MVT byte stream, rejecting the whole tile on the first
/// layer- or feature-level validation failure.
///
/// [`decode_mvt`] is the right choice for tiles from a network source, where
/// one malformed feature should not blank a whole basemap; this entry point
/// is for conformance tests and a validation CLI that want a loud failure on
/// the first problem instead.
///
/// # Errors
///
/// Returns [`RenderError::Mvt`] if the bytes are not a valid `vector_tile.Tile`
/// message, if a layer declares a version other than `1` or `2`, if a layer's
/// `extent` is `0`, if names are missing or duplicated, if a feature's tags do
/// not resolve against the layer's tables, if a `Value` has anything but one
/// field set, or if a geometry stream violates §4.3.4/§4.3.5 — including
/// truncation, an unexpected command order, a zero repeat count, or a cursor
/// that leaves `i32` range.
pub fn decode_mvt_strict(bytes: &[u8]) -> Result<VectorTile, RenderError> {
    let tile = proto::Tile::decode(bytes)
        .map_err(|err| RenderError::Mvt(format!("protobuf decode failed: {err}")))?;

    let mut layers: Vec<MvtLayer> = Vec::with_capacity(tile.layers.len());
    for layer in tile.layers {
        let decoded = decode_layer(layer)?;
        if layers.iter().any(|seen| seen.name == decoded.name) {
            return Err(RenderError::Mvt(format!(
                "duplicate layer name {:?}",
                decoded.name
            )));
        }
        layers.push(decoded);
    }
    Ok(VectorTile { layers })
}

/// The `extent` a layer inherits when the field is absent (proto default).
const DEFAULT_EXTENT: u32 = 4096;

/// Checks a layer's header — version, name, extent — against the
/// specification, shared by every decode mode.
///
/// Returns the effective extent: the default substituted for an absent one.
fn validate_layer_header(layer: &proto::Layer) -> Result<u32, RenderError> {
    if !matches!(layer.version, 1 | 2) {
        return Err(RenderError::Mvt(format!(
            "unsupported layer version {} (expected 1 or 2)",
            layer.version
        )));
    }
    if layer.name.is_empty() {
        return Err(RenderError::Mvt("layer has no name".to_owned()));
    }
    let extent = layer.extent.unwrap_or(DEFAULT_EXTENT);
    if extent == 0 {
        return Err(RenderError::Mvt(format!(
            "layer {:?} declares extent 0",
            layer.name
        )));
    }
    Ok(extent)
}

/// Validates a layer's header and decodes every feature in it, failing on
/// the first problem either raises. The lenient counterpart is
/// [`decode_layer_lenient`].
fn decode_layer(layer: proto::Layer) -> Result<MvtLayer, RenderError> {
    let extent = validate_layer_header(&layer)?;
    let mut value_cache = new_value_cache(&layer.values);
    let mut features = Vec::with_capacity(layer.features.len());
    for (index, feature) in layer.features.iter().enumerate() {
        features.push(
            decode_feature(feature, &layer.keys, &layer.values, &mut value_cache).map_err(
                |err| match err {
                    RenderError::Mvt(message) => RenderError::Mvt(format!(
                        "layer {:?} feature {index}: {message}",
                        layer.name
                    )),
                    other => other,
                },
            )?,
        );
    }

    Ok(MvtLayer {
        name: layer.name,
        extent,
        features,
    })
}

/// Validates a layer's header and decodes every feature in it, dropping a
/// malformed header (returning `None`) or a malformed feature (skipping just
/// that entry) instead of failing. Every drop is counted in `report`.
fn decode_layer_lenient(layer: proto::Layer, report: &mut DecodeReport) -> Option<MvtLayer> {
    let extent = match validate_layer_header(&layer) {
        Ok(extent) => extent,
        Err(err) => {
            report.skip_layer(|| format!("layer {:?}: {err}", layer.name));
            return None;
        }
    };

    let mut value_cache = new_value_cache(&layer.values);
    let mut features = Vec::with_capacity(layer.features.len());
    for (index, feature) in layer.features.iter().enumerate() {
        match decode_feature(feature, &layer.keys, &layer.values, &mut value_cache) {
            Ok(decoded) => features.push(decoded),
            Err(err) => {
                report.skip_feature(|| format!("layer {:?} feature {index}: {err}", layer.name));
            }
        }
    }

    Some(MvtLayer {
        name: layer.name,
        extent,
        features,
    })
}

/// Resolves one feature's tags and geometry.
fn decode_feature(
    feature: &proto::Feature,
    keys: &[String],
    values: &[proto::Value],
    value_cache: &mut [Option<Result<MvtValue, String>>],
) -> Result<MvtFeature, RenderError> {
    let properties = decode_tags(&feature.tags, keys, values, value_cache)?;
    let geometry_type = proto::GeomType::from_i32(feature.r#type).ok_or_else(|| {
        RenderError::Mvt(format!(
            "unknown geometry type {} (UNKNOWN and unassigned values are invalid)",
            feature.r#type
        ))
    })?;
    let geometry = decode_geometry(geometry_type, &feature.geometry)?;
    Ok(MvtFeature {
        id: feature.id,
        properties,
        geometry,
    })
}

/// Builds an empty per-layer cache for [`cached_value`], one slot per entry
/// of the layer's value table.
fn new_value_cache(values: &[proto::Value]) -> Vec<Option<Result<MvtValue, String>>> {
    vec![None; values.len()]
}

/// Resolves `values[index]` through `cache`, decoding and validating it only
/// on the first tag that references it.
///
/// A value referenced by many features — a typical enum-like property such
/// as `class` or `highway` — would otherwise re-run [`decode_value`]'s
/// ambiguity check once per occurrence; caching drops that to once per
/// layer. It does *not* avoid the allocation a string-valued occurrence still
/// needs: [`MvtFeature::properties`] owns its strings outright, so a cache
/// hit still clones one.
///
/// Kept lazy rather than pre-decoding the whole table up front: a value no
/// tag ever references is still never decoded, so an ambiguous or empty
/// `Value` the encoder never actually used still cannot fail a decode it
/// plays no part in.
fn cached_value(
    cache: &mut [Option<Result<MvtValue, String>>],
    index: usize,
    value: &proto::Value,
) -> Result<MvtValue, RenderError> {
    let Some(slot) = cache.get_mut(index) else {
        // Unreachable given callers size `cache` to the value table and
        // already resolved `value` at the same `index`; decode directly
        // rather than trust an invariant that turned out not to hold.
        return decode_value(value);
    };
    if let Some(outcome) = slot {
        return outcome.clone().map_err(RenderError::Mvt);
    }
    let outcome = decode_value(value).map_err(|error| error.to_string());
    *slot = Some(outcome.clone());
    outcome.map_err(RenderError::Mvt)
}

/// Turns a flat `[key_index, value_index, ...]` list into owned properties.
fn decode_tags(
    tags: &[u32],
    keys: &[String],
    values: &[proto::Value],
    value_cache: &mut [Option<Result<MvtValue, String>>],
) -> Result<Vec<(String, MvtValue)>, RenderError> {
    if !tags.len().is_multiple_of(2) {
        return Err(RenderError::Mvt(format!(
            "tag list has odd length {}",
            tags.len()
        )));
    }
    // De-duplication is keyed on the *key index*, not the key string: two
    // tags naming the same index are certainly the same key, and comparing
    // indices is O(1) instead of an O(key length) string comparison. This
    // keeps the loop below linear in `tags.len()` even for a tile crafted
    // with many distinct keys, where a string-keyed scan is quadratic.
    let mut key_seen = vec![false; keys.len()];
    let mut properties: Vec<(String, MvtValue)> = Vec::with_capacity(tags.len() / 2);
    for pair in tags.chunks_exact(2) {
        // `chunks_exact(2)` always yields two elements; index rather than
        // destructure so no fallible pattern is needed.
        let key_index = pair[0];
        let value_index = pair[1];
        let key_position = usize::try_from(key_index).unwrap_or(usize::MAX);
        let key = keys.get(key_position).ok_or_else(|| {
            RenderError::Mvt(format!(
                "tag key index {key_index} out of range (layer has {} keys)",
                keys.len()
            ))
        })?;
        let value_position = usize::try_from(value_index).unwrap_or(usize::MAX);
        let value = values.get(value_position).ok_or_else(|| {
            RenderError::Mvt(format!(
                "tag value index {value_index} out of range (layer has {} values)",
                values.len()
            ))
        })?;
        // `key_position` is in range: the successful `keys.get` above proved
        // it, and `key_seen` is sized to `keys.len()`.
        let Some(already_seen) = key_seen.get_mut(key_position) else {
            continue;
        };
        if *already_seen {
            // Documented policy: the first occurrence of a key wins.
            continue;
        }
        *already_seen = true;
        properties.push((
            key.clone(),
            cached_value(value_cache, value_position, value)?,
        ));
    }
    Ok(properties)
}

/// Collapses a `vector_tile.Value` onto [`MvtValue`], rejecting ambiguity.
fn decode_value(value: &proto::Value) -> Result<MvtValue, RenderError> {
    let mut decoded: Option<MvtValue> = None;
    let mut set = 0usize;

    if let Some(text) = value.string_value.as_ref() {
        set += 1;
        decoded = Some(MvtValue::String(text.clone()));
    }
    if let Some(number) = value.float_value {
        set += 1;
        decoded = Some(MvtValue::F32(number));
    }
    if let Some(number) = value.double_value {
        set += 1;
        decoded = Some(MvtValue::F64(number));
    }
    if let Some(number) = value.int_value {
        set += 1;
        decoded = Some(MvtValue::I64(number));
    }
    if let Some(number) = value.uint_value {
        set += 1;
        decoded = Some(MvtValue::U64(number));
    }
    if let Some(number) = value.sint_value {
        set += 1;
        decoded = Some(MvtValue::I64(number));
    }
    if let Some(flag) = value.bool_value {
        set += 1;
        decoded = Some(MvtValue::Bool(flag));
    }

    match (set, decoded) {
        (1, Some(value)) => Ok(value),
        (0, _) => Err(RenderError::Mvt(
            "value message has no field set".to_owned(),
        )),
        (count, _) => Err(RenderError::Mvt(format!(
            "value message has {count} fields set, expected exactly 1"
        ))),
    }
}

/// A cursor walking a feature's command/parameter stream.
///
/// The cursor position is cumulative across the whole stream — including across
/// `MoveTo` commands that start a new ring — and `ClosePath` never moves it.
struct GeometryCursor<'stream> {
    /// The command and parameter integers.
    stream: &'stream [u32],
    /// Read position within `stream`.
    offset: usize,
    /// Accumulated X, kept in `i64` so overflow is detectable, not silent.
    x: i64,
    /// Accumulated Y.
    y: i64,
}

impl<'stream> GeometryCursor<'stream> {
    /// Starts at the specification's origin `(0, 0)`.
    const fn new(stream: &'stream [u32]) -> Self {
        Self {
            stream,
            offset: 0,
            x: 0,
            y: 0,
        }
    }

    /// Whether every integer has been consumed.
    const fn is_exhausted(&self) -> bool {
        self.offset >= self.stream.len()
    }

    /// How many integers are left.
    const fn remaining(&self) -> usize {
        self.stream.len().saturating_sub(self.offset)
    }

    /// Reads the next command integer and checks that its parameters are all
    /// present *before* any capacity is reserved for them, so an inflated count
    /// cannot turn a short tile into a huge allocation.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mvt`] on an undefined command id, a zero repeat count, or
    /// a truncated parameter block.
    fn next_command(&mut self) -> Result<Command, RenderError> {
        let raw = *self.stream.get(self.offset).ok_or_else(|| {
            RenderError::Mvt("geometry stream ended where a command was expected".to_owned())
        })?;
        self.offset += 1;
        let command = decode_command_integer(raw)?;
        if command.count() == 0 {
            return Err(RenderError::Mvt(format!(
                "command {:?} has a repeat count of 0",
                command.id()
            )));
        }
        let needed = usize::try_from(command.parameter_count()).unwrap_or(usize::MAX);
        if needed > self.remaining() {
            return Err(RenderError::Mvt(format!(
                "command {:?} x{} needs {needed} parameters but only {} remain",
                command.id(),
                command.count(),
                self.remaining()
            )));
        }
        Ok(command)
    }

    /// Consumes one `(dx, dy)` parameter pair and returns the new absolute
    /// position.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mvt`] if the stream is truncated or if the accumulated
    /// cursor leaves the range of `i32`.
    fn next_point(&mut self) -> Result<[i32; 2], RenderError> {
        if self.remaining() < 2 {
            return Err(RenderError::Mvt(
                "geometry stream ended mid coordinate pair".to_owned(),
            ));
        }
        let raw_x = *self
            .stream
            .get(self.offset)
            .ok_or_else(|| RenderError::Mvt("geometry stream truncated".to_owned()))?;
        let raw_y = *self
            .stream
            .get(self.offset + 1)
            .ok_or_else(|| RenderError::Mvt("geometry stream truncated".to_owned()))?;
        self.offset += 2;

        // `i64` accumulation: a running sum of `i32`-range deltas cannot
        // overflow it within one tile's stream, so a hostile delta shows up in
        // the range check below instead of wrapping into a plausible coordinate.
        self.x += i64::from(zigzag_decode(raw_x));
        self.y += i64::from(zigzag_decode(raw_y));
        let x = i32::try_from(self.x).map_err(|_| {
            RenderError::Mvt(format!("geometry cursor x {} left i32 range", self.x))
        })?;
        let y = i32::try_from(self.y).map_err(|_| {
            RenderError::Mvt(format!("geometry cursor y {} left i32 range", self.y))
        })?;
        Ok([x, y])
    }
}

/// Dispatches on the geometry type and enforces its command grammar.
fn decode_geometry(
    geometry_type: proto::GeomType,
    stream: &[u32],
) -> Result<MvtGeometry, RenderError> {
    if stream.is_empty() {
        return Err(RenderError::Mvt(
            "feature has an empty geometry stream".to_owned(),
        ));
    }
    match geometry_type {
        proto::GeomType::Point => decode_points(stream).map(MvtGeometry::Points),
        proto::GeomType::LineString => decode_lines(stream).map(MvtGeometry::Lines),
        proto::GeomType::Polygon => decode_polygons(stream).map(MvtGeometry::Polygons),
    }
}

/// §4.3.4.2: a single `MoveTo` whose count is the number of points.
fn decode_points(stream: &[u32]) -> Result<Vec<[i32; 2]>, RenderError> {
    let mut cursor = GeometryCursor::new(stream);
    let command = cursor.next_command()?;
    if command.id() != CommandId::MoveTo {
        return Err(RenderError::Mvt(format!(
            "point geometry must start with MoveTo, found {:?}",
            command.id()
        )));
    }
    // `next_command` already proved the parameters are present, so this
    // capacity is bounded by the stream length, not by an untrusted count.
    let mut points = Vec::with_capacity(usize::try_from(command.count()).unwrap_or(0));
    for _ in 0..command.count() {
        points.push(cursor.next_point()?);
    }
    if !cursor.is_exhausted() {
        return Err(RenderError::Mvt(format!(
            "point geometry has {} trailing integers after its MoveTo",
            cursor.remaining()
        )));
    }
    Ok(points)
}

/// §4.3.4.3: repeated `(MoveTo count=1, LineTo count>=1)`.
fn decode_lines(stream: &[u32]) -> Result<Vec<Vec<[i32; 2]>>, RenderError> {
    let mut cursor = GeometryCursor::new(stream);
    let mut lines = Vec::new();
    while !cursor.is_exhausted() {
        lines.push(read_open_path(&mut cursor, "linestring")?);
    }
    if lines.is_empty() {
        return Err(RenderError::Mvt(
            "linestring geometry contains no line".to_owned(),
        ));
    }
    Ok(lines)
}

/// §4.3.4.4: repeated `(MoveTo count=1, LineTo count>=1, ClosePath count=1)`,
/// with rings classified into polygons by *relative* winding.
///
/// The specification's absolute rule (§4.3.3.3) — positive shoelace area is
/// exterior — only binds version 2 encoders; version 1 left winding
/// unspecified, and a version 2 encoder can still get it backwards. Rather
/// than rejecting such tiles, the sign of the geometry's first non-degenerate
/// ring is taken as "exterior" for the rest of the geometry: a ring matching
/// that sign starts a new polygon, any other sign becomes a hole in the
/// polygon started most recently. This is lossless — `vector::tess` fills and
/// strokes exterior and interior rings identically under
/// `FillRule::NonZero`, so this grouping never changes what is drawn, only
/// how the rings are labelled.
fn decode_polygons(stream: &[u32]) -> Result<Vec<MvtPolygon>, RenderError> {
    let mut cursor = GeometryCursor::new(stream);
    let mut polygons: Vec<MvtPolygon> = Vec::new();
    let mut rings = 0usize;
    // Area sign of the first non-degenerate ring seen so far, `None` until
    // then; whatever it turns out to be defines "exterior" for the rest of
    // this geometry.
    let mut exterior_is_positive: Option<bool> = None;

    while !cursor.is_exhausted() {
        let ring = read_open_path(&mut cursor, "polygon ring")?;
        let close = cursor.next_command()?;
        if close.id() != CommandId::ClosePath {
            return Err(RenderError::Mvt(format!(
                "polygon ring must end with ClosePath, found {:?}",
                close.id()
            )));
        }
        if close.count() != 1 {
            return Err(RenderError::Mvt(format!(
                "ClosePath must have a count of 1, found {}",
                close.count()
            )));
        }
        rings += 1;

        let area = doubled_signed_area(&ring);
        if area == 0 {
            // Documented policy: a degenerate ring is dropped without changing
            // the exterior/interior classification state.
            continue;
        }
        let positive = area > 0;
        let starts_new_polygon = match exterior_is_positive {
            Some(exterior) => exterior == positive,
            None => {
                exterior_is_positive = Some(positive);
                true
            }
        };
        if starts_new_polygon {
            polygons.push(MvtPolygon {
                exterior: ring,
                interiors: Vec::new(),
            });
        } else if let Some(polygon) = polygons.last_mut() {
            polygon.interiors.push(ring);
        } else {
            // Unreachable: `starts_new_polygon` is false only once
            // `exterior_is_positive` is `Some`, and the `None` arm above
            // never leaves that state without also pushing a polygon. Kept
            // as a fallback instead of a panic so a future refactor of this
            // match cannot turn a hole into a crash — it turns it into its
            // own (isolated) polygon instead.
            polygons.push(MvtPolygon {
                exterior: ring,
                interiors: Vec::new(),
            });
        }
    }

    if rings == 0 {
        return Err(RenderError::Mvt(
            "polygon geometry contains no ring".to_owned(),
        ));
    }
    Ok(polygons)
}

/// Reads one `MoveTo count=1` followed by one `LineTo count>=1`, returning the
/// resulting open point sequence.
fn read_open_path(
    cursor: &mut GeometryCursor<'_>,
    what: &str,
) -> Result<Vec<[i32; 2]>, RenderError> {
    let move_to = cursor.next_command()?;
    if move_to.id() != CommandId::MoveTo {
        return Err(RenderError::Mvt(format!(
            "{what} must start with MoveTo, found {:?}",
            move_to.id()
        )));
    }
    if move_to.count() != 1 {
        return Err(RenderError::Mvt(format!(
            "{what} MoveTo must have a count of 1, found {}",
            move_to.count()
        )));
    }
    let start = cursor.next_point()?;

    let line_to = cursor.next_command()?;
    if line_to.id() != CommandId::LineTo {
        return Err(RenderError::Mvt(format!(
            "{what} must continue with LineTo, found {:?}",
            line_to.id()
        )));
    }
    // Capacity is safe: `next_command` verified that `2 * count` parameters
    // really are in the stream.
    let mut path = Vec::with_capacity(usize::try_from(line_to.count()).unwrap_or(0) + 1);
    path.push(start);
    for _ in 0..line_to.count() {
        path.push(cursor.next_point()?);
    }
    Ok(path)
}

/// Twice the shoelace area of a closed ring given by its unclosed points.
///
/// Returns `2A = Σ (x_i · y_{i+1} − x_{i+1} · y_i)`; only the sign and the
/// zero-ness are used, so the factor of two is left in to keep it exact. The
/// accumulator is `i128` because a single term of two `i32` factors already
/// reaches `i64::MAX`.
fn doubled_signed_area(ring: &[[i32; 2]]) -> i128 {
    let successors = ring.iter().cycle().skip(1);
    ring.iter()
        .zip(successors)
        .map(|(current, next)| {
            i128::from(current[0]) * i128::from(next[1])
                - i128::from(next[0]) * i128::from(current[1])
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_EXTENT, DecodeReport, MvtGeometry, MvtPolygon, MvtValue, decode_geometry,
        decode_mvt, decode_mvt_report, decode_mvt_strict, doubled_signed_area,
    };
    use crate::error::RenderError;
    use crate::mvt::proto;
    use crate::mvt::wire::{Command, CommandId, encode_command_integer, zigzag_encode};
    use prost::Message as _;

    /// Wraps a geometry stream in a one-feature, one-layer tile.
    fn tile_bytes(geom_type: proto::GeomType, geometry: Vec<u32>) -> Vec<u8> {
        proto::Tile {
            layers: vec![proto::Layer {
                name: "spec".to_owned(),
                features: vec![proto::Feature {
                    id: None,
                    tags: Vec::new(),
                    r#type: geom_type as i32,
                    geometry,
                }],
                keys: Vec::new(),
                values: Vec::new(),
                extent: None,
                version: 2,
            }],
        }
        .encode_to_vec()
    }

    /// Decodes the single feature geometry of a single-layer tile.
    fn only_geometry(geom_type: proto::GeomType, geometry: Vec<u32>) -> MvtGeometry {
        let bytes = tile_bytes(geom_type, geometry);
        let tile = decode_mvt(&bytes).expect("tile decodes");
        assert_eq!(tile.layers.len(), 1);
        assert_eq!(tile.layers[0].extent, DEFAULT_EXTENT);
        assert_eq!(tile.layers[0].features.len(), 1);
        tile.layers[0].features[0].geometry.clone()
    }

    /// Asserts a decode failed with a [`RenderError::Mvt`] mentioning `needle`.
    fn assert_mvt_error<T: core::fmt::Debug>(result: Result<T, RenderError>, needle: &str) {
        match result {
            Err(RenderError::Mvt(message)) => assert!(
                message.contains(needle),
                "message {message:?} does not mention {needle:?}"
            ),
            other => panic!("expected an Mvt error mentioning {needle:?}, got {other:?}"),
        }
    }

    // ------------------------------------------------------- spec §4.3.5.1

    #[test]
    fn spec_example_point() {
        // Command 9 = MoveTo x1, parameters 50/34 = zigzag(+25)/zigzag(+17).
        assert_eq!(
            only_geometry(proto::GeomType::Point, vec![9, 50, 34]),
            MvtGeometry::Points(vec![[25, 17]])
        );
    }

    #[test]
    fn spec_example_multipoint() {
        // 17 = MoveTo x2: (+5,+7) then (-2,-5) -> (5,7) and (3,2).
        assert_eq!(
            only_geometry(proto::GeomType::Point, vec![17, 10, 14, 3, 9]),
            MvtGeometry::Points(vec![[5, 7], [3, 2]])
        );
    }

    #[test]
    fn spec_example_linestring() {
        // 9 = MoveTo x1 (+2,+2); 18 = LineTo x2 (0,+8) then (+8,0).
        assert_eq!(
            only_geometry(proto::GeomType::LineString, vec![9, 4, 4, 18, 0, 16, 16, 0]),
            MvtGeometry::Lines(vec![vec![[2, 2], [2, 10], [10, 10]]])
        );
    }

    #[test]
    fn spec_example_multilinestring() {
        // The second MoveTo is relative to the previous LineTo endpoint (10,10).
        assert_eq!(
            only_geometry(
                proto::GeomType::LineString,
                vec![9, 4, 4, 18, 0, 16, 16, 0, 9, 17, 17, 10, 4, 8],
            ),
            MvtGeometry::Lines(vec![vec![[2, 2], [2, 10], [10, 10]], vec![[1, 1], [3, 5]],])
        );
    }

    #[test]
    fn spec_example_polygon() {
        // (3,6),(8,12),(20,34); 2A = +38 > 0, so the ring is exterior.
        let ring = vec![[3, 6], [8, 12], [20, 34]];
        assert_eq!(doubled_signed_area(&ring), 38);
        assert_eq!(
            only_geometry(
                proto::GeomType::Polygon,
                vec![9, 6, 12, 18, 10, 12, 24, 44, 15]
            ),
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: ring,
                interiors: Vec::new(),
            }])
        );
    }

    #[test]
    fn spec_example_multipolygon() {
        // Three rings. Ring 2's MoveTo is relative to ring 1's *last LineTo*,
        // (0,10) — a ClosePath that moved the cursor would shift it by (0,-10).
        let stream = vec![
            9, 0, 0, 26, 20, 0, 0, 20, 19, 0, 15, // ring 1: 2A = +200
            9, 22, 2, 26, 18, 0, 0, 18, 17, 0, 15, // ring 2: 2A = +162
            9, 4, 13, 26, 0, 8, 8, 0, 0, 7, 15, // ring 3: 2A = -32 (hole)
        ];
        let outer_one = vec![[0, 0], [10, 0], [10, 10], [0, 10]];
        let outer_two = vec![[11, 11], [20, 11], [20, 20], [11, 20]];
        let hole = vec![[13, 13], [13, 17], [17, 17], [17, 13]];
        assert_eq!(doubled_signed_area(&outer_one), 200);
        assert_eq!(doubled_signed_area(&outer_two), 162);
        assert_eq!(doubled_signed_area(&hole), -32);

        assert_eq!(
            only_geometry(proto::GeomType::Polygon, stream),
            MvtGeometry::Polygons(vec![
                MvtPolygon {
                    exterior: outer_one,
                    interiors: Vec::new(),
                },
                MvtPolygon {
                    exterior: outer_two,
                    interiors: vec![hole],
                },
            ])
        );
    }

    // ------------------------------------------------------- ring winding

    /// Encodes one closed ring (`MoveTo`, `LineTo` xN, `ClosePath`) starting
    /// from `cursor`, i.e. where the previous ring's last `LineTo` left off.
    fn ring_commands(ring: &[[i32; 2]], cursor: [i32; 2]) -> Vec<u32> {
        let Some((first, rest)) = ring.split_first() else {
            panic!("empty ring");
        };
        let mut out = Vec::new();
        let move_to = Command::new(CommandId::MoveTo, 1).expect("MoveTo x1");
        out.push(encode_command_integer(move_to));
        out.push(zigzag_encode(first[0] - cursor[0]));
        out.push(zigzag_encode(first[1] - cursor[1]));

        let count = u32::try_from(rest.len()).expect("ring fits");
        let line_to = Command::new(CommandId::LineTo, count).expect("LineTo");
        out.push(encode_command_integer(line_to));
        let mut previous = *first;
        for point in rest {
            out.push(zigzag_encode(point[0] - previous[0]));
            out.push(zigzag_encode(point[1] - previous[1]));
            previous = *point;
        }

        let close = Command::new(CommandId::ClosePath, 1).expect("ClosePath x1");
        out.push(encode_command_integer(close));
        out
    }

    #[test]
    fn screen_clockwise_rings_are_exterior() {
        // Y is down, so this square is drawn right, down, left, up: clockwise
        // on screen. 2A = +200 > 0, hence exterior per §4.3.3.3.
        let clockwise = [[0, 0], [10, 0], [10, 10], [0, 10]];
        assert_eq!(doubled_signed_area(&clockwise), 200);
        // Reversing the order flips the sign, which is what marks a hole.
        let counter_clockwise = [[0, 10], [10, 10], [10, 0], [0, 0]];
        assert_eq!(doubled_signed_area(&counter_clockwise), -200);
        // Degenerate rings have no sign at all.
        assert_eq!(doubled_signed_area(&[[1, 1], [5, 5], [9, 9]]), 0);
        assert_eq!(doubled_signed_area(&[[4, 4], [4, 4]]), 0);
        assert_eq!(doubled_signed_area(&[[4, 4]]), 0);
        assert_eq!(doubled_signed_area(&[]), 0);
    }

    #[test]
    fn extreme_coordinates_do_not_overflow_the_area_accumulator() {
        // Half of the widest `i32` square. Each of the two non-zero shoelace
        // terms is just under `i64::MAX`, so their sum is about `2 * i64::MAX`:
        // an `i64` accumulator would panic in debug and wrap in release, which
        // could flip a ring's classification. Hence `i128`.
        let ring = [
            [i32::MAX, i32::MAX],
            [i32::MIN, i32::MAX],
            [i32::MIN, i32::MIN],
        ];
        let area = doubled_signed_area(&ring);
        // 2147483647 * 4294967295 + 2147483648 * 4294967295.
        assert_eq!(area, 18_446_744_065_119_617_025);
        assert!(area > i128::from(i64::MAX));
    }

    #[test]
    fn a_degenerate_ring_does_not_reparent_the_following_hole() {
        // Exterior square (2A = +3200), a collinear ring (2A = 0), then a hole
        // (2A = -200) that must still attach to the square.
        let exterior = [[0, 0], [40, 0], [40, 40], [0, 40]];
        let degenerate = [[1, 1], [2, 2], [3, 3]];
        let hole = [[10, 20], [10, 30], [20, 30], [20, 20]];
        assert_eq!(doubled_signed_area(&exterior), 3200);
        assert_eq!(doubled_signed_area(&degenerate), 0);
        assert_eq!(doubled_signed_area(&hole), -200);

        let mut stream = ring_commands(&exterior, [0, 0]);
        stream.extend(ring_commands(&degenerate, [0, 40]));
        stream.extend(ring_commands(&hole, [3, 3]));

        let MvtGeometry::Polygons(polygons) = only_geometry(proto::GeomType::Polygon, stream)
        else {
            panic!("expected polygons");
        };
        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].exterior, exterior.to_vec());
        assert_eq!(polygons[0].interiors, vec![hole.to_vec()]);
    }

    #[test]
    fn a_reversed_first_ring_becomes_its_own_exterior_instead_of_erroring() {
        // Counter-clockwise-on-screen square: 2A = -3200, and nothing precedes
        // it. Its sign becomes "exterior" for the rest of the geometry — there
        // is nothing else here — so it decodes as a single exterior ring
        // rather than being rejected for lacking one.
        let ring = [[0, 40], [40, 40], [40, 0], [0, 0]];
        assert_eq!(doubled_signed_area(&ring), -3200);
        let stream = ring_commands(&ring, [0, 0]);
        assert_eq!(
            only_geometry(proto::GeomType::Polygon, stream),
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: ring.to_vec(),
                interiors: Vec::new(),
            }])
        );
    }

    #[test]
    fn a_uniformly_reversed_multipolygon_still_pairs_exteriors_with_holes() {
        // A whole geometry wound backwards from the §4.3.3.3 convention: the
        // exterior is counter-clockwise-on-screen (2A < 0) and its hole is
        // clockwise-on-screen (2A > 0) — the opposite of the spec examples
        // above, but internally consistent, which is what a real encoder that
        // simply inverted its winding would produce.
        let exterior = [[0, 40], [40, 40], [40, 0], [0, 0]];
        let hole = [[10, 10], [20, 10], [20, 20], [10, 20]];
        assert_eq!(doubled_signed_area(&exterior), -3200);
        assert_eq!(doubled_signed_area(&hole), 200);

        // `exterior`'s last point is `[0, 0]`, which is where `ClosePath`
        // leaves the cursor for the next ring (see `GeometryCursor`'s doc).
        let mut stream = ring_commands(&exterior, [0, 0]);
        stream.extend(ring_commands(&hole, [0, 0]));

        assert_eq!(
            only_geometry(proto::GeomType::Polygon, stream),
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: exterior.to_vec(),
                interiors: vec![hole.to_vec()],
            }])
        );
    }

    #[test]
    fn a_polygon_of_only_degenerate_rings_decodes_to_nothing() {
        let stream = ring_commands(&[[0, 0], [5, 5], [9, 9]], [0, 0]);
        assert_eq!(
            only_geometry(proto::GeomType::Polygon, stream),
            MvtGeometry::Polygons(Vec::new())
        );
    }

    // --------------------------------------------------------- round trip

    #[test]
    fn a_hand_built_tile_round_trips_into_the_typed_model() {
        let tile = proto::Tile {
            layers: vec![
                proto::Layer {
                    name: "roads".to_owned(),
                    features: vec![
                        proto::Feature {
                            id: Some(11),
                            tags: vec![0, 0, 1, 1],
                            r#type: proto::GeomType::LineString as i32,
                            geometry: vec![9, 4, 4, 18, 0, 16, 16, 0],
                        },
                        proto::Feature {
                            id: None,
                            tags: vec![1, 2],
                            r#type: proto::GeomType::Point as i32,
                            geometry: vec![9, 50, 34],
                        },
                    ],
                    keys: vec!["class".to_owned(), "lanes".to_owned()],
                    values: vec![
                        proto::Value {
                            string_value: Some("primary".to_owned()),
                            ..proto::Value::default()
                        },
                        proto::Value {
                            uint_value: Some(4),
                            ..proto::Value::default()
                        },
                        proto::Value {
                            bool_value: Some(true),
                            ..proto::Value::default()
                        },
                    ],
                    extent: Some(512),
                    version: 2,
                },
                proto::Layer {
                    name: "water".to_owned(),
                    features: vec![proto::Feature {
                        id: Some(1),
                        tags: Vec::new(),
                        r#type: proto::GeomType::Polygon as i32,
                        geometry: vec![9, 6, 12, 18, 10, 12, 24, 44, 15],
                    }],
                    keys: Vec::new(),
                    values: Vec::new(),
                    extent: None,
                    version: 1,
                },
            ],
        };

        let decoded = decode_mvt(&tile.encode_to_vec()).expect("decodes");
        assert_eq!(decoded.layers.len(), 2);

        let roads = &decoded.layers[0];
        assert_eq!(roads.name, "roads");
        assert_eq!(roads.extent, 512);
        assert_eq!(roads.features.len(), 2);
        assert_eq!(roads.features[0].id, Some(11));
        assert_eq!(
            roads.features[0].properties,
            vec![
                ("class".to_owned(), MvtValue::String("primary".to_owned())),
                ("lanes".to_owned(), MvtValue::U64(4)),
            ]
        );
        assert_eq!(
            roads.features[0].geometry,
            MvtGeometry::Lines(vec![vec![[2, 2], [2, 10], [10, 10]]])
        );
        assert_eq!(roads.features[1].id, None);
        assert_eq!(
            roads.features[1].properties,
            vec![("lanes".to_owned(), MvtValue::Bool(true))]
        );
        assert_eq!(
            roads.features[1].geometry,
            MvtGeometry::Points(vec![[25, 17]])
        );

        let water = &decoded.layers[1];
        assert_eq!(water.name, "water");
        assert_eq!(water.extent, DEFAULT_EXTENT);
        assert_eq!(
            water.features[0].geometry,
            MvtGeometry::Polygons(vec![MvtPolygon {
                exterior: vec![[3, 6], [8, 12], [20, 34]],
                interiors: Vec::new(),
            }])
        );
    }

    #[test]
    fn every_value_alternative_decodes() {
        let values = vec![
            proto::Value {
                string_value: Some("s".to_owned()),
                ..proto::Value::default()
            },
            proto::Value {
                float_value: Some(1.5),
                ..proto::Value::default()
            },
            proto::Value {
                double_value: Some(-2.25),
                ..proto::Value::default()
            },
            proto::Value {
                int_value: Some(-7),
                ..proto::Value::default()
            },
            proto::Value {
                uint_value: Some(u64::MAX),
                ..proto::Value::default()
            },
            proto::Value {
                sint_value: Some(i64::MIN),
                ..proto::Value::default()
            },
            proto::Value {
                bool_value: Some(false),
                ..proto::Value::default()
            },
        ];
        let keys: Vec<String> = (0..values.len()).map(|index| format!("k{index}")).collect();
        let tags: Vec<u32> = (0..values.len())
            .flat_map(|index| {
                let index = u32::try_from(index).expect("small");
                [index, index]
            })
            .collect();
        let tile = proto::Tile {
            layers: vec![proto::Layer {
                name: "values".to_owned(),
                features: vec![proto::Feature {
                    id: None,
                    tags,
                    r#type: proto::GeomType::Point as i32,
                    geometry: vec![9, 0, 0],
                }],
                keys,
                values,
                extent: None,
                version: 2,
            }],
        };
        let decoded = decode_mvt(&tile.encode_to_vec()).expect("decodes");
        let properties = &decoded.layers[0].features[0].properties;
        assert_eq!(
            properties
                .iter()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>(),
            vec![
                MvtValue::String("s".to_owned()),
                MvtValue::F32(1.5),
                MvtValue::F64(-2.25),
                MvtValue::I64(-7),
                MvtValue::U64(u64::MAX),
                MvtValue::I64(i64::MIN),
                MvtValue::Bool(false),
            ]
        );
    }

    #[test]
    fn duplicate_keys_keep_the_first_value() {
        let tile = proto::Tile {
            layers: vec![proto::Layer {
                name: "dup".to_owned(),
                features: vec![proto::Feature {
                    id: None,
                    tags: vec![0, 0, 0, 1],
                    r#type: proto::GeomType::Point as i32,
                    geometry: vec![9, 0, 0],
                }],
                keys: vec!["k".to_owned()],
                values: vec![
                    proto::Value {
                        int_value: Some(1),
                        ..proto::Value::default()
                    },
                    proto::Value {
                        int_value: Some(2),
                        ..proto::Value::default()
                    },
                ],
                extent: None,
                version: 2,
            }],
        };
        let decoded = decode_mvt(&tile.encode_to_vec()).expect("decodes");
        assert_eq!(
            decoded.layers[0].features[0].properties,
            vec![("k".to_owned(), MvtValue::I64(1))]
        );
    }

    // ---------------------------------------------------------- validation

    /// A one-layer tile, encoded.
    fn layer_tile(layer: proto::Layer) -> Vec<u8> {
        proto::Tile {
            layers: vec![layer],
        }
        .encode_to_vec()
    }

    /// A minimal valid layer the validation tests perturb one field at a time.
    fn point_layer() -> proto::Layer {
        proto::Layer {
            name: "l".to_owned(),
            features: vec![proto::Feature {
                id: None,
                tags: Vec::new(),
                r#type: proto::GeomType::Point as i32,
                geometry: vec![9, 0, 0],
            }],
            keys: Vec::new(),
            values: Vec::new(),
            extent: None,
            version: 2,
        }
    }

    #[test]
    fn garbage_bytes_are_rejected() {
        // Field 3 (layers) declares a 200-byte payload inside a 4-byte buffer.
        assert_mvt_error(decode_mvt(&[0x1a, 200, 1, 2]), "protobuf decode failed");
    }

    #[test]
    fn an_empty_tile_decodes_to_no_layers() {
        let tile = decode_mvt(&[]).expect("an empty message is a valid empty tile");
        assert!(tile.layers.is_empty());
    }

    // These tests pin down the validation *rules* themselves, so they go
    // through `decode_mvt_strict`, which fails the whole tile on the first
    // one broken — the same all-or-nothing behaviour this module had before
    // it grew a lenient default. The "lenient decoding" section further down
    // proves what `decode_mvt`/`decode_mvt_report` do with these same rule
    // violations instead.

    #[test]
    fn unsupported_versions_are_rejected() {
        for version in [0u32, 3, 4, u32::MAX] {
            let bytes = layer_tile(proto::Layer {
                version,
                ..point_layer()
            });
            assert_mvt_error(decode_mvt_strict(&bytes), "unsupported layer version");
        }
        for version in [1u32, 2] {
            let bytes = layer_tile(proto::Layer {
                version,
                ..point_layer()
            });
            assert!(
                decode_mvt_strict(&bytes).is_ok(),
                "version {version} must decode"
            );
        }
    }

    #[test]
    fn a_zero_extent_is_rejected() {
        let bytes = layer_tile(proto::Layer {
            extent: Some(0),
            ..point_layer()
        });
        assert_mvt_error(decode_mvt_strict(&bytes), "extent 0");
        // A present, non-default extent survives.
        let bytes = layer_tile(proto::Layer {
            extent: Some(1),
            ..point_layer()
        });
        let tile = decode_mvt_strict(&bytes).expect("extent 1 is legal");
        assert_eq!(tile.layers[0].extent, 1);
    }

    #[test]
    fn nameless_and_duplicate_layers_are_rejected() {
        let bytes = layer_tile(proto::Layer {
            name: String::new(),
            ..point_layer()
        });
        assert_mvt_error(decode_mvt_strict(&bytes), "no name");

        let bytes = proto::Tile {
            layers: vec![point_layer(), point_layer()],
        }
        .encode_to_vec();
        assert_mvt_error(decode_mvt_strict(&bytes), "duplicate layer name");
    }

    #[test]
    fn odd_and_out_of_range_tag_lists_are_rejected() {
        let with_tags = |tags: Vec<u32>| {
            layer_tile(proto::Layer {
                features: vec![proto::Feature {
                    id: None,
                    tags,
                    r#type: proto::GeomType::Point as i32,
                    geometry: vec![9, 0, 0],
                }],
                keys: vec!["k".to_owned()],
                values: vec![proto::Value {
                    bool_value: Some(true),
                    ..proto::Value::default()
                }],
                ..point_layer()
            })
        };
        assert_mvt_error(decode_mvt_strict(&with_tags(vec![0])), "odd length 1");
        assert_mvt_error(decode_mvt_strict(&with_tags(vec![0, 0, 1])), "odd length 3");
        assert_mvt_error(decode_mvt_strict(&with_tags(vec![1, 0])), "tag key index 1");
        assert_mvt_error(
            decode_mvt_strict(&with_tags(vec![0, 9])),
            "tag value index 9",
        );
        assert_mvt_error(
            decode_mvt_strict(&with_tags(vec![0, u32::MAX])),
            "tag value index 4294967295",
        );
        // The failure is attributed to its layer and feature.
        assert_mvt_error(
            decode_mvt_strict(&with_tags(vec![0])),
            "layer \"l\" feature 0",
        );
        // The valid pair still decodes.
        let tile = decode_mvt_strict(&with_tags(vec![0, 0])).expect("valid tags decode");
        assert_eq!(
            tile.layers[0].features[0].properties,
            vec![("k".to_owned(), MvtValue::Bool(true))]
        );
    }

    #[test]
    fn ambiguous_and_empty_values_are_rejected() {
        let with_value = |value: proto::Value| {
            layer_tile(proto::Layer {
                features: vec![proto::Feature {
                    id: None,
                    tags: vec![0, 0],
                    r#type: proto::GeomType::Point as i32,
                    geometry: vec![9, 0, 0],
                }],
                keys: vec!["k".to_owned()],
                values: vec![value],
                ..point_layer()
            })
        };
        assert_mvt_error(
            decode_mvt_strict(&with_value(proto::Value::default())),
            "no field set",
        );
        assert_mvt_error(
            decode_mvt_strict(&with_value(proto::Value {
                int_value: Some(1),
                bool_value: Some(true),
                ..proto::Value::default()
            })),
            "2 fields set",
        );
        assert_mvt_error(
            decode_mvt_strict(&with_value(proto::Value {
                string_value: Some(String::new()),
                float_value: Some(0.0),
                double_value: Some(0.0),
                int_value: Some(0),
                uint_value: Some(0),
                sint_value: Some(0),
                bool_value: Some(false),
            })),
            "7 fields set",
        );
    }

    #[test]
    fn unknown_geometry_types_are_rejected() {
        for raw in [0i32, 4, -1, i32::MAX] {
            let bytes = layer_tile(proto::Layer {
                features: vec![proto::Feature {
                    id: None,
                    tags: Vec::new(),
                    r#type: raw,
                    geometry: vec![9, 0, 0],
                }],
                ..point_layer()
            });
            assert_mvt_error(decode_mvt_strict(&bytes), "unknown geometry type");
        }
    }

    #[test]
    fn malformed_geometry_streams_are_rejected() {
        let cases: Vec<(proto::GeomType, Vec<u32>, &str)> = vec![
            // Empty stream, whatever the type.
            (proto::GeomType::Point, vec![], "empty geometry stream"),
            (proto::GeomType::LineString, vec![], "empty geometry stream"),
            (proto::GeomType::Polygon, vec![], "empty geometry stream"),
            // MoveTo with a count of 0.
            (proto::GeomType::Point, vec![1], "repeat count of 0"),
            // Truncated parameter blocks.
            (proto::GeomType::Point, vec![9, 50], "needs 2 parameters"),
            (
                proto::GeomType::LineString,
                vec![9, 4, 4, 18, 0, 16, 16],
                "needs 4 parameters",
            ),
            // Undefined command id.
            (
                proto::GeomType::Point,
                vec![3, 0, 0],
                "undefined mvt geometry command id 3",
            ),
            // Wrong leading command.
            (
                proto::GeomType::Point,
                vec![10, 0, 0],
                "point geometry must start with MoveTo",
            ),
            (
                proto::GeomType::LineString,
                vec![10, 0, 0],
                "linestring must start with MoveTo",
            ),
            (
                proto::GeomType::Polygon,
                vec![15],
                "polygon ring must start with MoveTo",
            ),
            // Trailing integers after a POINT MoveTo.
            (
                proto::GeomType::Point,
                vec![9, 0, 0, 9, 0, 0],
                "trailing integers",
            ),
            // LineString: MoveTo count must be 1, and a LineTo must follow.
            (
                proto::GeomType::LineString,
                vec![17, 0, 0, 0, 0, 10, 2, 2],
                "MoveTo must have a count of 1",
            ),
            (
                proto::GeomType::LineString,
                vec![9, 0, 0],
                "ended where a command was expected",
            ),
            (
                proto::GeomType::LineString,
                vec![9, 0, 0, 15],
                "must continue with LineTo",
            ),
            // Polygon: ClosePath is required, exactly once.
            (
                proto::GeomType::Polygon,
                vec![9, 0, 0, 26, 20, 0, 0, 20, 19, 0],
                "ended where a command was expected",
            ),
            (
                proto::GeomType::Polygon,
                vec![9, 0, 0, 26, 20, 0, 0, 20, 19, 0, 9, 0, 0],
                "must end with ClosePath",
            ),
            (
                proto::GeomType::Polygon,
                vec![9, 0, 0, 26, 20, 0, 0, 20, 19, 0, 23],
                "ClosePath must have a count of 1",
            ),
        ];
        for (geom_type, geometry, needle) in cases {
            let bytes = tile_bytes(geom_type, geometry);
            assert_mvt_error(decode_mvt_strict(&bytes), needle);
        }
    }

    #[test]
    fn a_zero_count_line_to_is_rejected() {
        // 2 = LineTo x0, refused before any parameter is read.
        let bytes = tile_bytes(proto::GeomType::LineString, vec![9, 0, 0, 2]);
        assert_mvt_error(decode_mvt_strict(&bytes), "repeat count of 0");
    }

    #[test]
    fn a_huge_repeat_count_does_not_allocate() {
        // MoveTo with count 2^29-1 and nothing behind it: rejected on the
        // parameter check, before any capacity is reserved.
        let command = Command::new(CommandId::MoveTo, (1 << 29) - 1).expect("max count");
        let bytes = tile_bytes(
            proto::GeomType::Point,
            vec![encode_command_integer(command)],
        );
        assert_mvt_error(decode_mvt_strict(&bytes), "needs 1073741822 parameters");
    }

    #[test]
    fn a_cursor_that_leaves_i32_range_is_rejected() {
        let move_to = encode_command_integer(Command::new(CommandId::MoveTo, 1).expect("MoveTo"));
        let line_to = encode_command_integer(Command::new(CommandId::LineTo, 1).expect("LineTo"));

        // Jump to i32::MAX, then step one further in x.
        let stream = vec![
            move_to,
            zigzag_encode(i32::MAX),
            zigzag_encode(0),
            line_to,
            zigzag_encode(1),
            zigzag_encode(0),
        ];
        let bytes = tile_bytes(proto::GeomType::LineString, stream);
        assert_mvt_error(
            decode_mvt_strict(&bytes),
            "cursor x 2147483648 left i32 range",
        );

        // And the same in the negative direction, on y.
        let stream = vec![
            move_to,
            zigzag_encode(0),
            zigzag_encode(i32::MIN),
            line_to,
            zigzag_encode(0),
            zigzag_encode(-1),
        ];
        let bytes = tile_bytes(proto::GeomType::LineString, stream);
        assert_mvt_error(
            decode_mvt_strict(&bytes),
            "cursor y -2147483649 left i32 range",
        );
    }

    // ------------------------------------------------------ lenient decoding

    #[test]
    fn a_lenient_decode_skips_one_malformed_feature_and_keeps_its_siblings() {
        let good_a = proto::Feature {
            id: Some(1),
            tags: Vec::new(),
            r#type: proto::GeomType::Point as i32,
            geometry: vec![9, 0, 0],
        };
        // Key index 0 is valid but the layer has no values, so index 9 is not.
        let bad = proto::Feature {
            id: Some(2),
            tags: vec![0, 9],
            r#type: proto::GeomType::Point as i32,
            geometry: vec![9, 0, 0],
        };
        let good_b = proto::Feature {
            id: Some(3),
            tags: Vec::new(),
            r#type: proto::GeomType::Point as i32,
            geometry: vec![9, 4, 4],
        };
        let bytes = layer_tile(proto::Layer {
            features: vec![good_a, bad, good_b],
            keys: vec!["k".to_owned()],
            ..point_layer()
        });

        let (tile, report) = decode_mvt_report(&bytes).expect("envelope decodes");
        assert_eq!(tile.layers.len(), 1);
        let ids: Vec<Option<u64>> = tile.layers[0]
            .features
            .iter()
            .map(|feature| feature.id)
            .collect();
        assert_eq!(ids, vec![Some(1), Some(3)]);
        assert_eq!(report.skipped_features, 1);
        assert_eq!(report.skipped_layers, 0);
        assert!(!report.is_clean());
        let issue = report.first_issue.as_deref().unwrap_or_default();
        assert!(issue.contains("layer \"l\" feature 1"), "{issue:?}");

        // The default entry point applies the same policy, just without the
        // report.
        let lenient = decode_mvt(&bytes).expect("lenient decode still succeeds");
        assert_eq!(lenient.layers[0].features.len(), 2);

        // Strict mode rejects the exact same bytes outright.
        assert_mvt_error(decode_mvt_strict(&bytes), "tag value index 9");
    }

    #[test]
    fn a_lenient_decode_skips_one_malformed_layer_and_keeps_its_siblings() {
        let bad_version = proto::Layer {
            name: "bad-version".to_owned(),
            version: 99,
            ..point_layer()
        };
        let bad_extent = proto::Layer {
            name: "bad-extent".to_owned(),
            extent: Some(0),
            ..point_layer()
        };
        let bytes = proto::Tile {
            layers: vec![point_layer(), bad_version, bad_extent],
        }
        .encode_to_vec();

        let (tile, report) = decode_mvt_report(&bytes).expect("envelope decodes");
        let names: Vec<&str> = tile
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, vec!["l"]);
        assert_eq!(report.skipped_layers, 2);
        assert_eq!(report.skipped_features, 0);

        assert_mvt_error(decode_mvt_strict(&bytes), "unsupported layer version");
    }

    #[test]
    fn a_lenient_decode_keeps_the_first_of_two_same_named_layers() {
        let bytes = proto::Tile {
            layers: vec![point_layer(), point_layer()],
        }
        .encode_to_vec();

        let (tile, report) = decode_mvt_report(&bytes).expect("envelope decodes");
        assert_eq!(tile.layers.len(), 1);
        assert_eq!(report.skipped_layers, 1);
        let issue = report.first_issue.as_deref().unwrap_or_default();
        assert!(issue.contains("duplicate layer name"), "{issue:?}");

        assert_mvt_error(decode_mvt_strict(&bytes), "duplicate layer name");
    }

    #[test]
    fn a_layer_with_every_feature_malformed_survives_empty_rather_than_vanishing() {
        // Odd tag length: malformed regardless of which feature carries it.
        let bad = proto::Feature {
            id: None,
            tags: vec![0],
            r#type: proto::GeomType::Point as i32,
            geometry: vec![9, 0, 0],
        };
        let bytes = layer_tile(proto::Layer {
            features: vec![bad.clone(), bad],
            ..point_layer()
        });

        let (tile, report) = decode_mvt_report(&bytes).expect("envelope decodes");
        assert_eq!(tile.layers.len(), 1);
        assert_eq!(tile.layers[0].name, "l");
        assert!(tile.layers[0].features.is_empty());
        assert_eq!(report.skipped_features, 2);
        assert!(!report.is_clean());
        // Only the first reason is retained, and it names the first feature.
        let issue = report.first_issue.as_deref().unwrap_or_default();
        assert!(issue.contains("feature 0"), "{issue:?}");
        assert!(!issue.contains("feature 1"), "{issue:?}");
    }

    #[test]
    fn a_well_formed_tile_produces_a_clean_report() {
        let bytes = layer_tile(point_layer());
        let (tile, report) = decode_mvt_report(&bytes).expect("decodes");
        assert_eq!(tile.layers.len(), 1);
        assert!(report.is_clean());
        assert_eq!(report, DecodeReport::default());
    }

    #[test]
    fn buffer_coordinates_outside_the_extent_are_kept() {
        let geometry = only_geometry(
            proto::GeomType::Point,
            vec![
                encode_command_integer(Command::new(CommandId::MoveTo, 2).expect("MoveTo x2")),
                zigzag_encode(-64),
                zigzag_encode(4160),
                zigzag_encode(64),
                zigzag_encode(-4160),
            ],
        );
        assert_eq!(geometry, MvtGeometry::Points(vec![[-64, 4160], [0, 0]]));
    }

    #[test]
    fn geometry_dispatch_covers_every_type() {
        // Guards against a geometry type being wired to the wrong reader.
        assert!(matches!(
            decode_geometry(proto::GeomType::Point, &[9, 0, 0]),
            Ok(MvtGeometry::Points(_))
        ));
        assert!(matches!(
            decode_geometry(proto::GeomType::LineString, &[9, 0, 0, 10, 2, 2]),
            Ok(MvtGeometry::Lines(_))
        ));
        assert!(matches!(
            decode_geometry(
                proto::GeomType::Polygon,
                &[9, 0, 0, 26, 20, 0, 0, 20, 19, 0, 15]
            ),
            Ok(MvtGeometry::Polygons(_))
        ));
    }
}
