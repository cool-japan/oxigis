//! The style model: the blueprint's `fill` / `line` / `circle` / `symbol`
//! subset (see the blueprint `oxigis.md` §5.2, kept in the repository's
//! git history — MapLibre Style Spec compatibility is deliberately *not*
//! pursued in full) plus [`Color`], an RGBA8 color that serializes as a
//! compact hex string.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::CoreError;
use crate::renderer::{Attributes, Classification, Renderer};
use crate::util::{clamp_unit, deserialize_clamped_unit};

/// An RGBA color, 8 bits per channel.
///
/// Serializes as a lowercase `"rrggbbaa"` hex string (always 8 hex digits,
/// including alpha) rather than as a JSON object — compact, and matches the
/// convention most web map style formats already use for colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color {
    /// Red channel, 0-255.
    pub r: u8,
    /// Green channel, 0-255.
    pub g: u8,
    /// Blue channel, 0-255.
    pub b: u8,
    /// Alpha channel, 0-255 (`0` = fully transparent, `255` = fully opaque).
    pub a: u8,
}

impl Color {
    /// Fully opaque black.
    pub const BLACK: Color = Color {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    /// Fully opaque white.
    pub const WHITE: Color = Color {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    /// Fully transparent black — a sensible "no color" default.
    pub const TRANSPARENT: Color = Color {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };

    /// Builds an opaque color from RGB channels (alpha = 255).
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Builds a color from RGBA channels.
    pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Renders as a lowercase `"rrggbbaa"` hex string (no `#` prefix).
    pub fn to_hex(self) -> String {
        format!("{:02x}{:02x}{:02x}{:02x}", self.r, self.g, self.b, self.a)
    }

    /// Parses a `"rrggbb"` (alpha defaults to fully opaque) or `"rrggbbaa"`
    /// hex string, with an optional leading `#`.
    ///
    /// Never panics on malformed input — returns [`CoreError::InvalidColor`]
    /// instead, including for non-ASCII input (which, if sliced by byte
    /// offset without this check, could split a multi-byte UTF-8 character
    /// and panic).
    pub fn from_hex(input: &str) -> Result<Self, CoreError> {
        let body = input.strip_prefix('#').unwrap_or(input);
        if !body.is_ascii() {
            return Err(CoreError::InvalidColor {
                input: input.to_string(),
                reason: "expected ASCII hex digits".to_string(),
            });
        }
        // `u8::from_str_radix` accepts a leading `+` (`"+f"` -> `Ok(15)`),
        // which `body`'s fixed two-character channel slices would otherwise
        // let through as a valid-looking byte for a sign character instead
        // of a hex digit. Ruling out anything but hex digits up front closes
        // that hole before the per-channel parse below ever sees it.
        if !body.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(CoreError::InvalidColor {
                input: input.to_string(),
                reason: "expected only hex digits (0-9, a-f, A-F)".to_string(),
            });
        }

        let channel = |slice: &str| -> Result<u8, CoreError> {
            u8::from_str_radix(slice, 16).map_err(|_| CoreError::InvalidColor {
                input: input.to_string(),
                reason: format!("{slice:?} is not a valid hex byte"),
            })
        };

        match body.len() {
            6 => Ok(Self {
                r: channel(&body[0..2])?,
                g: channel(&body[2..4])?,
                b: channel(&body[4..6])?,
                a: 255,
            }),
            8 => Ok(Self {
                r: channel(&body[0..2])?,
                g: channel(&body[2..4])?,
                b: channel(&body[4..6])?,
                a: channel(&body[6..8])?,
            }),
            other => Err(CoreError::InvalidColor {
                input: input.to_string(),
                reason: format!("expected 6 or 8 hex digits, got {other}"),
            }),
        }
    }
}

impl Default for Color {
    fn default() -> Self {
        Color::BLACK
    }
}

impl Serialize for Color {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Color {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        Color::from_hex(&text).map_err(D::Error::custom)
    }
}

/// Fill (polygon) rendering style — the MapLibre `fill-*` subset.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FillStyle {
    /// Fill color.
    pub color: Color,
    /// Fill opacity. Always kept within `0.0..=1.0` — set via
    /// [`FillStyle::set_opacity`], which clamps.
    #[serde(deserialize_with = "deserialize_clamped_unit")]
    opacity: f32,
    /// Optional outline color; `None` draws no outline.
    pub outline_color: Option<Color>,
}

impl FillStyle {
    /// Creates a fill style with the given color, full opacity, no outline.
    pub fn new(color: Color) -> Self {
        Self {
            color,
            opacity: 1.0,
            outline_color: None,
        }
    }

    /// Current opacity, guaranteed to be within `0.0..=1.0`.
    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    /// Sets opacity, clamping the input into `0.0..=1.0`.
    pub fn set_opacity(&mut self, value: f32) {
        self.opacity = clamp_unit(value);
    }
}

/// Line (`LineString`) rendering style — the MapLibre `line-*` subset.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LineStyle {
    /// Stroke color.
    pub color: Color,
    /// Stroke width in logical pixels. Always kept non-negative.
    width: f32,
    /// Stroke opacity. Always kept within `0.0..=1.0` — set via
    /// [`LineStyle::set_opacity`], which clamps.
    #[serde(deserialize_with = "deserialize_clamped_unit")]
    opacity: f32,
}

impl LineStyle {
    /// Creates a line style with the given color and width, at full
    /// opacity. A negative `width` is clamped to `0.0`.
    pub fn new(color: Color, width: f32) -> Self {
        Self {
            color,
            width: width.max(0.0),
            opacity: 1.0,
        }
    }

    /// Stroke width in logical pixels, guaranteed non-negative.
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Sets the stroke width, clamping negative input to `0.0`.
    pub fn set_width(&mut self, value: f32) {
        self.width = value.max(0.0);
    }

    /// Current opacity, guaranteed to be within `0.0..=1.0`.
    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    /// Sets opacity, clamping the input into `0.0..=1.0`.
    pub fn set_opacity(&mut self, value: f32) {
        self.opacity = clamp_unit(value);
    }
}

/// Point-as-circle rendering style — the MapLibre `circle-*` subset.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CircleStyle {
    /// Circle radius in logical pixels. Always kept non-negative.
    radius: f32,
    /// Fill color.
    pub color: Color,
    /// Optional stroke color drawn around the circle.
    pub stroke_color: Option<Color>,
    /// Stroke width in logical pixels. Always kept non-negative.
    stroke_width: f32,
    /// Fill opacity. Always kept within `0.0..=1.0` — set via
    /// [`CircleStyle::set_opacity`], which clamps.
    #[serde(deserialize_with = "deserialize_clamped_unit")]
    opacity: f32,
}

impl CircleStyle {
    /// Creates a circle style with the given radius and fill color, no
    /// stroke, at full opacity. A negative `radius` is clamped to `0.0`.
    pub fn new(radius: f32, color: Color) -> Self {
        Self {
            radius: radius.max(0.0),
            color,
            stroke_color: None,
            stroke_width: 0.0,
            opacity: 1.0,
        }
    }

    /// Circle radius in logical pixels, guaranteed non-negative.
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Sets the radius, clamping negative input to `0.0`.
    pub fn set_radius(&mut self, value: f32) {
        self.radius = value.max(0.0);
    }

    /// Stroke width in logical pixels, guaranteed non-negative.
    pub fn stroke_width(&self) -> f32 {
        self.stroke_width
    }

    /// Sets the stroke width, clamping negative input to `0.0`.
    pub fn set_stroke_width(&mut self, value: f32) {
        self.stroke_width = value.max(0.0);
    }

    /// Current opacity, guaranteed to be within `0.0..=1.0`.
    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    /// Sets opacity, clamping the input into `0.0..=1.0`.
    pub fn set_opacity(&mut self, value: f32) {
        self.opacity = clamp_unit(value);
    }
}

/// The weight a label's text is drawn at — the MapLibre `text-font`
/// subset, reduced to the one axis a map style actually uses.
///
/// Deliberately an enum of two rather than a numeric weight: the renderer
/// and the PDF exporter both resolve it to a REAL bold face (never a
/// synthetic emboldening), and "any integer 1..1000" would promise a
/// gradation no font chain on a stock desktop can honour. A face that has
/// no bold twin falls back to [`LabelWeight::Regular`] with one log, so the
/// value is a request, not a guarantee.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum LabelWeight {
    /// The ordinary face — the default, and what every pre-v1.4 project
    /// means by saying nothing.
    #[default]
    Regular,
    /// The chain's bold face, when one exists.
    Bold,
}

impl LabelWeight {
    /// Both weights, in panel order.
    pub const ALL: [LabelWeight; 2] = [Self::Regular, Self::Bold];

    /// Whether this is the default weight — the serialization skip
    /// condition, which is what keeps a pre-v1.4 project's JSON
    /// byte-identical after a load/save round trip.
    pub fn is_regular(&self) -> bool {
        matches!(self, Self::Regular)
    }

    /// The panel label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Regular => "Regular",
            Self::Bold => "Bold",
        }
    }
}

/// Which way a label's glyphs run — the CSS `writing-mode` axis, reduced to
/// the one distinction a map style actually needs.
///
/// Like [`LabelWeight`] this is a request rather than a guarantee: the
/// renderer accepts a vertical label only when every character draws upright
/// under UAX #50 and the resolved face carries the vertical metrics to stack
/// it. Anything else draws HORIZONTALLY and says so once — never half a
/// column, never a rotated glyph.
///
/// The PDF exporter ignores this in v1.5 and prints such labels horizontally,
/// with one aggregated warning per export. That is the first *styled*
/// screen/page divergence OxiGIS ships, and it is named rather than hidden.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum LabelOrientation {
    /// Left to right on one baseline — the default, and what every pre-v1.5
    /// project means by saying nothing.
    #[default]
    Horizontal,
    /// Top to bottom, one upright cell per character: Japanese / Chinese /
    /// Korean vertical writing.
    Vertical,
}

impl LabelOrientation {
    /// Both orientations, in panel order.
    pub const ALL: [LabelOrientation; 2] = [Self::Horizontal, Self::Vertical];

    /// Whether this is the default orientation — the serialization skip
    /// condition, which is what keeps a pre-v1.5 project's JSON
    /// byte-identical after a load/save round trip.
    pub fn is_horizontal(&self) -> bool {
        matches!(self, Self::Horizontal)
    }

    /// The panel label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Horizontal => "Horizontal",
            Self::Vertical => "Vertical (CJK)",
        }
    }
}

/// Text-label rendering style for point/line symbol layers — the MapLibre
/// `text-*`/`symbol-*` subset.
///
/// Blueprint §5.3: label *placement* (CJK text shaping via `oxiui-text`,
/// greedy collision avoidance) is a rendering concern, not a core-model
/// concern — this struct only carries the styling knobs a style panel edits.
///
/// # Field order is byte-observable
///
/// [`LayerStyleSet`] `#[serde(flatten)]`s its base style, and a flattened
/// struct serializes its fields in declaration order into the parent map.
/// [`Self::weight`] is therefore declared **last** and skipped at its
/// default, so every project written before v1.4 round-trips to the same
/// bytes — and [`Self::orientation`], added in v1.5, is declared after it for
/// exactly that reason. Any new field must be appended for the same reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolStyle {
    /// Name of the feature attribute whose value is drawn as the label
    /// text, e.g. `"name"`. `None` draws no text.
    pub text_field: Option<String>,
    /// Text color.
    pub text_color: Color,
    /// Font size in logical pixels. Always kept non-negative.
    text_size: f32,
    /// Optional halo (outline) color, for legibility over busy basemaps.
    pub halo_color: Option<Color>,
    /// Halo width in logical pixels. Always kept non-negative.
    halo_width: f32,
    /// Which weight the label is drawn at (print/text v1.4). Declared LAST
    /// and skipped when [`LabelWeight::Regular`] — see the type docs.
    #[serde(default, skip_serializing_if = "LabelWeight::is_regular")]
    weight: LabelWeight,
    /// Which way the label's glyphs run (print/text v1.5). Declared LAST,
    /// after [`Self::weight`], and skipped when
    /// [`LabelOrientation::Horizontal`] — see the struct docs.
    #[serde(default, skip_serializing_if = "LabelOrientation::is_horizontal")]
    orientation: LabelOrientation,
}

impl SymbolStyle {
    /// Creates a symbol style labeling features with `text_field`, black
    /// 12px text on a 1px white halo.
    pub fn new(text_field: impl Into<String>) -> Self {
        Self {
            text_field: Some(text_field.into()),
            text_color: Color::BLACK,
            text_size: 12.0,
            halo_color: Some(Color::WHITE),
            halo_width: 1.0,
            weight: LabelWeight::Regular,
            orientation: LabelOrientation::Horizontal,
        }
    }

    /// Font size in logical pixels, guaranteed non-negative.
    pub fn text_size(&self) -> f32 {
        self.text_size
    }

    /// Sets the font size, clamping negative input to `0.0`.
    pub fn set_text_size(&mut self, value: f32) {
        self.text_size = value.max(0.0);
    }

    /// Halo width in logical pixels, guaranteed non-negative.
    pub fn halo_width(&self) -> f32 {
        self.halo_width
    }

    /// Sets the halo width, clamping negative input to `0.0`.
    pub fn set_halo_width(&mut self, value: f32) {
        self.halo_width = value.max(0.0);
    }

    /// The weight the label is drawn at.
    pub fn weight(&self) -> LabelWeight {
        self.weight
    }

    /// Sets the weight. A request only: a chain with no bold face draws
    /// [`LabelWeight::Regular`] and says so once — never a synthetic
    /// emboldening.
    pub fn set_weight(&mut self, value: LabelWeight) {
        self.weight = value;
    }

    /// Which way the label's glyphs run.
    pub fn orientation(&self) -> LabelOrientation {
        self.orientation
    }

    /// Sets the orientation. A request only: a label the renderer cannot
    /// stack — a rotated character, a right-to-left one, a face with no
    /// `vmtx` — draws horizontally and says so once.
    pub fn set_orientation(&mut self, value: LabelOrientation) {
        self.orientation = value;
    }
}

impl Default for SymbolStyle {
    fn default() -> Self {
        Self {
            text_field: None,
            text_color: Color::BLACK,
            text_size: 12.0,
            halo_color: None,
            halo_width: 0.0,
            weight: LabelWeight::Regular,
            orientation: LabelOrientation::Horizontal,
        }
    }
}

/// A layer's style, tagged by the kind of geometry it targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LayerStyle {
    /// Polygon fill styling.
    Fill(FillStyle),
    /// Line styling.
    Line(LineStyle),
    /// Point-as-circle styling.
    Circle(CircleStyle),
    /// Text label styling.
    Symbol(SymbolStyle),
}

/// The geometry family a per-family style override targets (tiles v1.3
/// item C: a mixed `GeometryCollection` splits into up to three families,
/// which used to share ONE style — so a mixed dataset drew about a third of
/// itself). The ONE family rule hit testing, styling and the tile path all
/// share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryFamily {
    /// Polygons and multi-polygons.
    Polygon,
    /// Lines and multi-lines.
    Line,
    /// Points and multi-points.
    Point,
}

impl GeometryFamily {
    /// Every family, painter's order (fills under strokes under markers).
    pub const ALL: [GeometryFamily; 3] = [Self::Polygon, Self::Line, Self::Point];

    /// The panel label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Polygon => "Polygons",
            Self::Line => "Lines",
            Self::Point => "Points",
        }
    }

    /// A stable 0/1/2 index (ExtGState naming, coalesce keys).
    pub fn index(self) -> usize {
        match self {
            Self::Polygon => 0,
            Self::Line => 1,
            Self::Point => 2,
        }
    }
}

/// Which slot of a [`LayerStyleSet`] an edit addresses — the base style or
/// one family's override. Never serialized (edits address it; files store
/// the whole set).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StyleSlot {
    /// The shared base style.
    #[default]
    Base,
    /// One family's override.
    Family(GeometryFamily),
}

/// A small set over [`GeometryFamily`] — which families a dataset actually
/// contains, or which carry overrides.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FamilySet {
    polygon: bool,
    line: bool,
    point: bool,
}

impl FamilySet {
    /// Whether `family` is in the set.
    pub fn contains(self, family: GeometryFamily) -> bool {
        match family {
            GeometryFamily::Polygon => self.polygon,
            GeometryFamily::Line => self.line,
            GeometryFamily::Point => self.point,
        }
    }

    /// Adds `family`.
    pub fn insert(&mut self, family: GeometryFamily) {
        match family {
            GeometryFamily::Polygon => self.polygon = true,
            GeometryFamily::Line => self.line = true,
            GeometryFamily::Point => self.point = true,
        }
    }

    /// The union of both sets.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            polygon: self.polygon || other.polygon,
            line: self.line || other.line,
            point: self.point || other.point,
        }
    }

    /// How many families are present.
    pub fn len(self) -> usize {
        usize::from(self.polygon) + usize::from(self.line) + usize::from(self.point)
    }

    /// Whether no family is present.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// More than one family — the only case the panel's family row appears.
    pub fn is_mixed(self) -> bool {
        self.len() > 1
    }

    /// The present families, painter's order.
    pub fn iter(self) -> impl Iterator<Item = GeometryFamily> {
        GeometryFamily::ALL
            .into_iter()
            .filter(move |family| self.contains(*family))
    }
}

/// Per-family style overrides. Every field absent = "use the layer's shared
/// base style", which is exactly the pre-v1.3 behaviour.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FamilyStyles {
    /// The polygon family's override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polygon: Option<LayerStyle>,
    /// The line family's override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<LayerStyle>,
    /// The point family's override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point: Option<LayerStyle>,
}

impl FamilyStyles {
    /// The override for `family`, if set.
    pub fn get(&self, family: GeometryFamily) -> Option<&LayerStyle> {
        match family {
            GeometryFamily::Polygon => self.polygon.as_ref(),
            GeometryFamily::Line => self.line.as_ref(),
            GeometryFamily::Point => self.point.as_ref(),
        }
    }

    /// Sets (or clears, with [`None`]) the override for `family`.
    pub fn set(&mut self, family: GeometryFamily, style: Option<LayerStyle>) {
        match family {
            GeometryFamily::Polygon => self.polygon = style,
            GeometryFamily::Line => self.line = style,
            GeometryFamily::Point => self.point = style,
        }
    }

    /// Whether no override is set — the serialization skip condition, so a
    /// set without overrides is byte-identical to a bare [`LayerStyle`].
    pub fn is_empty(&self) -> bool {
        self.polygon.is_none() && self.line.is_none() && self.point.is_none()
    }
}

/// A layer's whole style state: a shared base plus up to three per-family
/// overrides (tiles v1.3 item C).
///
/// ONE value type rather than a parallel map, so every place that carries a
/// style — the undo op's before/after, the layer snapshot, the GPU restyle
/// op, the print capture — carries the WHOLE state and cannot silently drop
/// half of it.
///
/// # Wire shape (serde_json only)
///
/// The base is `#[serde(flatten)]`ed and `families` is skipped when empty,
/// so a set without overrides serializes **byte-identically** to the bare
/// [`LayerStyle`] it replaces (probe-verified, pretty and compact), every
/// v1.2 project file loads unchanged, and a v1.2 build reading a v1.3 file
/// still gets the base. `flatten` buffers content and is NOT safe for
/// non-self-describing formats — `Project` is only ever JSON, and any
/// future binary project format must revisit this type first.
///
/// # Field order is byte-observable
///
/// The flattened base splices its own fields into the parent map in
/// declaration order, and every later field follows in declaration order too.
/// [`Self::renderer`] is therefore declared **last**, after
/// `Self::families`, and skipped at [`Renderer::Single`] — which is what
/// keeps every pre-v1.6 project file byte-identical through a load and re-save
/// (the same rule [`SymbolStyle`]'s trailing fields follow, and the same
/// battery of tests pins it). Any new field must be appended for that reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerStyleSet {
    /// The shared base style — what every family draws with unless
    /// overridden, and what a v1.2 reader sees.
    #[serde(flatten)]
    base: LayerStyle,
    /// The per-family overrides.
    #[serde(default, skip_serializing_if = "FamilyStyles::is_empty")]
    families: FamilyStyles,
    /// How one FEATURE's style is resolved (thematic v1.6). Declared LAST and
    /// skipped when [`Renderer::Single`] — see the struct docs.
    #[serde(default, skip_serializing_if = "Renderer::is_single")]
    renderer: Renderer,
}

impl LayerStyleSet {
    /// A base style with no overrides — exactly the pre-v1.3 state.
    pub fn new(base: LayerStyle) -> Self {
        Self {
            base,
            families: FamilyStyles::default(),
            renderer: Renderer::default(),
        }
    }

    /// The shared base style.
    pub fn base(&self) -> &LayerStyle {
        &self.base
    }

    /// The base style, for editing.
    pub fn base_mut(&mut self) -> &mut LayerStyle {
        &mut self.base
    }

    /// The override for `family`, if set.
    pub fn override_for(&self, family: GeometryFamily) -> Option<&LayerStyle> {
        self.families.get(family)
    }

    /// What `family` actually draws with: its override, else the base.
    pub fn effective(&self, family: GeometryFamily) -> &LayerStyle {
        self.families.get(family).unwrap_or(&self.base)
    }

    /// Which slot `family` resolves through right now.
    pub fn slot_of(&self, family: GeometryFamily) -> StyleSlot {
        if self.families.get(family).is_some() {
            StyleSlot::Family(family)
        } else {
            StyleSlot::Base
        }
    }

    /// The style stored in `slot`, if any ([`StyleSlot::Base`] always is).
    pub fn slot(&self, slot: StyleSlot) -> Option<&LayerStyle> {
        match slot {
            StyleSlot::Base => Some(&self.base),
            StyleSlot::Family(family) => self.families.get(family),
        }
    }

    /// The style stored in `slot`, for editing.
    pub fn slot_mut(&mut self, slot: StyleSlot) -> Option<&mut LayerStyle> {
        match slot {
            StyleSlot::Base => Some(&mut self.base),
            StyleSlot::Family(family) => match family {
                GeometryFamily::Polygon => self.families.polygon.as_mut(),
                GeometryFamily::Line => self.families.line.as_mut(),
                GeometryFamily::Point => self.families.point.as_mut(),
            },
        }
    }

    /// Sets `family`'s override.
    pub fn set_override(&mut self, family: GeometryFamily, style: LayerStyle) {
        self.families.set(family, Some(style));
    }

    /// Clears `family`'s override — back to the shared base.
    pub fn clear_override(&mut self, family: GeometryFamily) {
        self.families.set(family, None);
    }

    /// The families carrying overrides.
    pub fn overridden(&self) -> FamilySet {
        let mut set = FamilySet::default();
        for family in GeometryFamily::ALL {
            if self.families.get(family).is_some() {
                set.insert(family);
            }
        }
        set
    }

    /// Whether any override is set.
    pub fn has_overrides(&self) -> bool {
        !self.families.is_empty()
    }

    /// How this layer resolves one feature's style (thematic v1.6).
    pub fn renderer(&self) -> &Renderer {
        &self.renderer
    }

    /// The renderer, for editing.
    pub fn renderer_mut(&mut self) -> &mut Renderer {
        &mut self.renderer
    }

    /// Replaces the renderer.
    pub fn set_renderer(&mut self, renderer: Renderer) {
        self.renderer = renderer;
    }

    /// Whether every feature of this layer draws with the same style — the
    /// pre-v1.6 shape, and the case a renderer can skip partitioning for.
    pub fn is_single_symbol(&self) -> bool {
        self.renderer.is_single()
    }

    /// How the renderer partitions this layer's features — the comparison key
    /// that tells a cached mesh whether a restyle changed the buckets or only
    /// their colours. See [`Renderer::classification`].
    pub fn classification(&self) -> Classification {
        self.renderer.classification()
    }

    /// How many classes this layer draws, beyond the fallback.
    pub fn class_count(&self) -> usize {
        self.renderer.class_count()
    }

    /// What `family` draws with for a feature carrying `attributes` — **the**
    /// resolution rule.
    ///
    /// One function, called by everything that has to agree on the picture:
    /// the map's mesh partition, the PDF exporter, hit testing and the legend.
    /// It is deliberately not three similar functions.
    ///
    /// The order is renderer first, geometry family second, composed rather
    /// than replaced: see [`Self::style_for_class`] for the composition rule
    /// and for why a mixed-geometry layer needs one.
    pub fn style_for<A>(&self, family: GeometryFamily, attributes: &A) -> LayerStyle
    where
        A: Attributes + ?Sized,
    {
        self.style_for_class(family, self.renderer.class_of(attributes))
    }

    /// [`Self::style_for`] with the class already resolved — what a caller
    /// that classified once and now draws one bucket at a time uses.
    ///
    /// `None` is the fallback bucket, which resolves to exactly the pre-v1.6
    /// answer ([`Self::effective`]) whenever the renderer names no fallback
    /// style of its own.
    ///
    /// A class's style is composed OVER the family's, not substituted for it
    /// ([`crate::renderer::class_over_family`]): a class that names a `Fill`
    /// must not erase a mixed layer's points, which have no fill arm in the
    /// tessellator. For a single-family layer — the overwhelmingly common case
    /// — the two are the same kind, so the class's style is taken verbatim.
    pub fn style_for_class(&self, family: GeometryFamily, class: Option<usize>) -> LayerStyle {
        match self.renderer.class_style(class) {
            Some(style) => crate::renderer::class_over_family(style, self.effective(family)),
            None => self.effective(family).clone(),
        }
    }
}

impl From<LayerStyle> for LayerStyleSet {
    fn from(base: LayerStyle) -> Self {
        Self::new(base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip_with_and_without_alpha() {
        let opaque = Color::from_hex("ff8800").expect("6-digit hex");
        assert_eq!(
            opaque,
            Color {
                r: 0xff,
                g: 0x88,
                b: 0x00,
                a: 255
            }
        );
        assert_eq!(opaque.to_hex(), "ff8800ff");

        let translucent = Color::from_hex("#ff880080").expect("8-digit hex with #");
        assert_eq!(
            translucent,
            Color {
                r: 0xff,
                g: 0x88,
                b: 0x00,
                a: 0x80
            }
        );
        assert_eq!(translucent.to_hex(), "ff880080");
    }

    #[test]
    fn hex_parsing_is_case_insensitive() {
        let lower = Color::from_hex("aabbccdd").expect("lowercase");
        let upper = Color::from_hex("AABBCCDD").expect("uppercase");
        assert_eq!(lower, upper);
    }

    #[test]
    fn hex_rejects_bad_length_without_panicking() {
        assert!(Color::from_hex("").is_err());
        assert!(Color::from_hex("fff").is_err());
        assert!(Color::from_hex("fffffffff").is_err());
    }

    #[test]
    fn hex_rejects_non_hex_characters_without_panicking() {
        assert!(Color::from_hex("gggggg").is_err());
        assert!(Color::from_hex("zz00ff00").is_err());
    }

    #[test]
    fn hex_rejects_sign_characters_that_from_str_radix_would_otherwise_accept() {
        // `u8::from_str_radix` parses a leading `+` (`"+f"` -> `Ok(15)`), so
        // without an explicit hex-digit check these would read as a valid
        // (wrong) color instead of a malformed one.
        assert!(Color::from_hex("+1+2+3").is_err());
        assert!(Color::from_hex("#+f+f+f").is_err());
    }

    #[test]
    fn hex_rejects_multibyte_input_without_panicking() {
        // 8 *bytes* but 7 *characters*: 'é' is 2 UTF-8 bytes, straddling
        // the byte offset (2) a naive byte-slice implementation would cut
        // at. This must return an error, not panic on a non-char-boundary
        // slice.
        let input = "1é34567";
        assert_eq!(input.len(), 8);
        let result = Color::from_hex(input);
        assert!(result.is_err());
    }

    #[test]
    fn color_serde_round_trip() {
        let color = Color::from_rgba(0x11, 0x22, 0x33, 0x44);
        let json = serde_json::to_string(&color).expect("serialize");
        assert_eq!(json, "\"11223344\"");
        let back: Color = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, color);
    }

    #[test]
    fn color_deserialize_rejects_bad_input_without_panic() {
        let result: Result<Color, _> = serde_json::from_str("\"not-a-color\"");
        assert!(result.is_err());
    }

    #[test]
    fn fill_style_opacity_clamps_via_setter_and_deserialize() {
        let mut fill = FillStyle::new(Color::WHITE);
        fill.set_opacity(2.0);
        assert_eq!(fill.opacity(), 1.0);

        let json = r#"{"color":"ffffffff","opacity":-9.0,"outline_color":null}"#;
        let loaded: FillStyle = serde_json::from_str(json).expect("deserialize");
        assert_eq!(loaded.opacity(), 0.0);
    }

    #[test]
    fn line_style_width_and_opacity_stay_non_negative() {
        let mut line = LineStyle::new(Color::BLACK, -5.0);
        assert_eq!(line.width(), 0.0);
        line.set_width(-1.0);
        assert_eq!(line.width(), 0.0);
        line.set_width(3.5);
        assert_eq!(line.width(), 3.5);
        line.set_opacity(-1.0);
        assert_eq!(line.opacity(), 0.0);
    }

    #[test]
    fn circle_style_radius_and_stroke_width_stay_non_negative() {
        let mut circle = CircleStyle::new(-2.0, Color::BLACK);
        assert_eq!(circle.radius(), 0.0);
        circle.set_radius(4.0);
        assert_eq!(circle.radius(), 4.0);
        circle.set_stroke_width(-1.0);
        assert_eq!(circle.stroke_width(), 0.0);
    }

    #[test]
    fn symbol_style_sizes_stay_non_negative() {
        let mut symbol = SymbolStyle::new("name");
        symbol.set_text_size(-1.0);
        assert_eq!(symbol.text_size(), 0.0);
        symbol.set_halo_width(-1.0);
        assert_eq!(symbol.halo_width(), 0.0);
    }

    // --- LabelWeight (print/text v1.4 item 4, D-W1): the byte-identity
    // battery. The hazard is `LayerStyleSet`'s `#[serde(flatten)]`, which
    // splices SymbolStyle's fields into the parent map in DECLARATION
    // order — so a new field anywhere but last is byte-observable.

    /// A whole `.oxigis.json` exactly as a v1.3 build wrote it: two layers,
    /// a SYMBOL base style carrying two per-family overrides, and a plain
    /// fill style beside it. Written against the v1.3 field list, so it is
    /// an independent oracle rather than a snapshot of today's output.
    const V13_PROJECT_JSON: &str = r#"{
  "format_version": 1,
  "name": "Weight fixture",
  "layers": [
    {
      "id": 1,
      "name": "Labels",
      "visible": true,
      "opacity": 1.0,
      "kind": {
        "kind": "vector",
        "source": {
          "type": "local_geo_json",
          "path": "labels.geojson"
        }
      }
    },
    {
      "id": 2,
      "name": "Areas",
      "visible": true,
      "opacity": 1.0,
      "kind": {
        "kind": "vector",
        "source": {
          "type": "local_geo_json",
          "path": "areas.geojson"
        }
      }
    }
  ],
  "styles": {
    "1": {
      "type": "symbol",
      "text_field": "name",
      "text_color": "000000ff",
      "text_size": 12.0,
      "halo_color": "ffffffff",
      "halo_width": 1.0,
      "families": {
        "line": {
          "type": "line",
          "color": "000000ff",
          "width": 2.0,
          "opacity": 1.0
        },
        "point": {
          "type": "circle",
          "radius": 4.0,
          "color": "ffffffff",
          "stroke_color": null,
          "stroke_width": 0.0,
          "opacity": 1.0
        }
      }
    },
    "2": {
      "type": "fill",
      "color": "3377ddff",
      "opacity": 0.35,
      "outline_color": "1c4b8fff"
    }
  },
  "view": {
    "center_lon": 0.0,
    "center_lat": 0.0,
    "zoom": 2.0
  }
}"#;

    #[test]
    fn a_regular_weight_symbol_style_serializes_exactly_as_it_did_before_v14() {
        let symbol = SymbolStyle::new("name");
        assert_eq!(symbol.weight(), LabelWeight::Regular, "the default");
        let compact = serde_json::to_string(&symbol).expect("compact");
        assert_eq!(
            compact,
            r#"{"text_field":"name","text_color":"000000ff","text_size":12.0,"halo_color":"ffffffff","halo_width":1.0}"#,
            "the v1.3 bytes, character for character",
        );
        assert!(!compact.contains("weight"));
        assert!(
            !serde_json::to_string_pretty(&symbol)
                .expect("pretty")
                .contains("weight"),
            "pretty JSON skips it too",
        );
    }

    #[test]
    fn a_bold_weight_round_trips_through_a_snake_case_tag() {
        let mut symbol = SymbolStyle::new("name");
        symbol.set_weight(LabelWeight::Bold);
        let json = serde_json::to_string(&symbol).expect("compact");
        assert!(json.contains(r#""weight":"bold""#), "{json}");
        assert!(
            json.ends_with(r#""weight":"bold"}"#),
            "declared last, so it serializes last: {json}",
        );
        let back: SymbolStyle = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, symbol);
        assert_eq!(back.weight(), LabelWeight::Bold);
    }

    #[test]
    fn a_v13_symbol_document_with_no_weight_key_loads_as_regular() {
        let legacy = r#"{"text_field":"name","text_color":"000000ff","text_size":12.0,
            "halo_color":"ffffffff","halo_width":1.0}"#;
        let symbol: SymbolStyle = serde_json::from_str(legacy).expect("a v1.3 symbol style loads");
        assert_eq!(symbol.weight(), LabelWeight::Regular);
    }

    #[test]
    fn a_full_v13_project_with_overrides_re_saves_byte_identically() {
        // THE gate the plan names: the `#[serde(flatten)]` interaction is the
        // hazard, so it is measured on a WHOLE project document — a symbol
        // base style carrying per-family overrides, plus a second styled
        // layer — rather than on a bare style. Load-and-re-save is the
        // strongest form of it: a v1.3 file on a user's disk must come back
        // out of a v1.4 build with the same bytes. A field inserted anywhere
        // but LAST, or a `weight` key that stops being skipped at its
        // default, fails here.
        assert!(
            !V13_PROJECT_JSON.contains("weight"),
            "the fixture predates the field",
        );
        let loaded = crate::project::Project::from_json_string(V13_PROJECT_JSON)
            .expect("the v1.3 document parses");
        assert_eq!(
            loaded.to_json_string().expect("re-serialize"),
            V13_PROJECT_JSON,
            "adding LabelWeight must not move one byte of a no-bold project",
        );
        // Compact too — `to_string` and `to_string_pretty` take different
        // serde paths, and only the pretty one is the on-disk shape.
        let compact = serde_json::to_string(&loaded).expect("compact");
        assert!(!compact.contains("weight"), "{compact}");
        assert_eq!(
            serde_json::from_str::<crate::project::Project>(&compact).expect("compact parses"),
            loaded,
        );
        // The symbol base really is a symbol with the default weight (so the
        // byte identity above is not vacuous).
        let (_, set) = loaded.styles.iter().next().expect("a styled layer");
        let LayerStyle::Symbol(symbol) = set.base() else {
            panic!("the first style is the symbol one");
        };
        assert_eq!(symbol.weight(), LabelWeight::Regular);
        assert_eq!(set.overridden().len(), 2, "per-family overrides present");
    }

    #[test]
    fn a_bold_symbol_style_survives_the_whole_project_round_trip() {
        use crate::layer::{Layer, LayerKind, LayerStack, VectorSource};
        use crate::project::Project;

        let mut project = Project::new("Bold fixture");
        let mut stack = LayerStack::new();
        let id = stack.add(Layer::new(
            "Labels",
            LayerKind::Vector(VectorSource::LocalGeoJson {
                path: "labels.geojson".to_string(),
            }),
        ));
        project.layers = stack;
        let mut symbol = SymbolStyle::new("name");
        symbol.set_weight(LabelWeight::Bold);
        let mut set = LayerStyleSet::new(LayerStyle::Symbol(symbol));
        set.set_override(
            GeometryFamily::Line,
            LayerStyle::Line(LineStyle::new(Color::BLACK, 2.0)),
        );
        project
            .set_style(id, set)
            .expect("LocalGeoJson accepts a layer style");

        let json = project.to_json_string().expect("pretty");
        assert!(json.contains(r#""weight": "bold""#), "{json}");
        let back = Project::from_json_string(&json).expect("round trip");
        assert_eq!(back, project);
        let LayerStyle::Symbol(symbol) = back.style(id).expect("a style").base() else {
            panic!("a symbol base");
        };
        assert_eq!(symbol.weight(), LabelWeight::Bold);
    }

    #[test]
    fn an_unknown_weight_value_is_refused_rather_than_silently_regular() {
        let hostile = r#"{"text_field":"name","text_color":"000000ff","text_size":12.0,
            "halo_color":null,"halo_width":0.0,"weight":"ultra_black"}"#;
        assert!(
            serde_json::from_str::<SymbolStyle>(hostile).is_err(),
            "an unknown weight is a malformed document, not a default",
        );
    }

    #[test]
    fn label_weight_defaults_to_regular_and_labels_itself() {
        assert_eq!(LabelWeight::default(), LabelWeight::Regular);
        assert!(LabelWeight::Regular.is_regular());
        assert!(!LabelWeight::Bold.is_regular());
        assert_eq!(LabelWeight::ALL, [LabelWeight::Regular, LabelWeight::Bold]);
        assert_eq!(LabelWeight::Bold.label(), "Bold");
    }

    #[test]
    fn layer_style_tag_shape_round_trips() {
        let style = LayerStyle::Circle(CircleStyle::new(6.0, Color::from_rgb(255, 0, 0)));
        let value = serde_json::to_value(&style).expect("serialize to Value");
        assert_eq!(value["type"], "circle");
        let round_tripped: LayerStyle = serde_json::from_value(value).expect("deserialize");
        assert_eq!(round_tripped, style);
    }

    // --- LayerStyleSet (tiles v1.3 item C): the serde back-compat battery ---

    fn fill_set() -> LayerStyleSet {
        let mut fill = FillStyle::new(Color::from_hex("3377dd").expect("hex"));
        fill.set_opacity(0.35);
        fill.outline_color = Some(Color::from_hex("1c4b8f").expect("hex"));
        LayerStyleSet::new(LayerStyle::Fill(fill))
    }

    #[test]
    fn a_legacy_bare_style_document_loads_as_a_set_with_no_overrides() {
        let legacy =
            r#"{"type":"fill","color":"3377ddff","opacity":0.35,"outline_color":"1c4b8fff"}"#;
        let set: LayerStyleSet = serde_json::from_str(legacy).expect("a v1.2 style loads");
        assert!(!set.has_overrides());
        assert!(matches!(set.base(), LayerStyle::Fill(_)));
    }

    #[test]
    fn a_set_without_overrides_is_byte_identical_to_the_bare_style() {
        let set = fill_set();
        let bare = set.base().clone();
        assert_eq!(
            serde_json::to_string(&set).expect("compact"),
            serde_json::to_string(&bare).expect("compact"),
            "compact JSON is byte-identical"
        );
        assert_eq!(
            serde_json::to_string_pretty(&set).expect("pretty"),
            serde_json::to_string_pretty(&bare).expect("pretty"),
            "pretty JSON is byte-identical"
        );
        assert!(
            !serde_json::to_string(&set)
                .expect("compact")
                .contains("families")
        );
    }

    #[test]
    fn every_override_survives_the_json_round_trip() {
        let mut set = fill_set();
        set.set_override(
            GeometryFamily::Line,
            LayerStyle::Line(LineStyle::new(Color::BLACK, 2.0)),
        );
        set.set_override(
            GeometryFamily::Point,
            LayerStyle::Circle(CircleStyle::new(4.0, Color::WHITE)),
        );
        set.set_override(
            GeometryFamily::Polygon,
            LayerStyle::Fill(FillStyle::new(Color::from_rgb(1, 2, 3))),
        );
        let json = serde_json::to_string(&set).expect("serialize");
        let restored: LayerStyleSet = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, set);
        assert_eq!(restored.overridden().len(), 3);
    }

    #[test]
    fn a_v12_reader_still_gets_the_base_and_unknown_family_keys_are_ignored() {
        // Forward compat both ways: a v1.2 build (bare LayerStyle) reading a
        // v1.3 document gets the base; a v1.3 build tolerates unknown keys.
        let mut set = fill_set();
        set.set_override(
            GeometryFamily::Line,
            LayerStyle::Line(LineStyle::new(Color::BLACK, 2.0)),
        );
        let json = serde_json::to_string(&set).expect("serialize");
        let bare: LayerStyle = serde_json::from_str(&json).expect("a v1.2 reader parses");
        assert_eq!(&bare, set.base());

        let with_unknown = r#"{"type":"fill","color":"3377ddff","opacity":0.35,
            "outline_color":null,"families":{"hexagon":{"type":"fill","color":"00000000",
            "opacity":1.0,"outline_color":null}}}"#;
        assert!(
            serde_json::from_str::<LayerStyleSet>(with_unknown).is_ok(),
            "an unknown family key is tolerated"
        );
    }

    #[test]
    fn opacity_still_clamps_through_the_flattened_path() {
        let hostile = r#"{"type":"fill","color":"3377ddff","opacity":-9.0,"outline_color":null}"#;
        let set: LayerStyleSet = serde_json::from_str(hostile).expect("clamped, not refused");
        let LayerStyle::Fill(fill) = set.base() else {
            panic!("a fill");
        };
        assert_eq!(
            fill.opacity(),
            0.0,
            "deserialize_clamped_unit survives flatten"
        );
    }

    #[test]
    fn effective_falls_back_to_the_base_and_slot_of_reports_which() {
        let mut set = fill_set();
        for family in GeometryFamily::ALL {
            assert_eq!(set.effective(family), set.base());
            assert_eq!(set.slot_of(family), StyleSlot::Base);
        }
        set.set_override(
            GeometryFamily::Point,
            LayerStyle::Circle(CircleStyle::new(4.0, Color::WHITE)),
        );
        assert_eq!(
            set.slot_of(GeometryFamily::Point),
            StyleSlot::Family(GeometryFamily::Point)
        );
        assert!(matches!(
            set.effective(GeometryFamily::Point),
            LayerStyle::Circle(_)
        ));
        assert_eq!(set.slot_of(GeometryFamily::Line), StyleSlot::Base);
        set.clear_override(GeometryFamily::Point);
        assert!(!set.has_overrides());
    }

    #[test]
    fn family_set_reports_membership_len_and_painterly_order() {
        let mut set = FamilySet::default();
        assert!(set.is_empty());
        set.insert(GeometryFamily::Point);
        set.insert(GeometryFamily::Polygon);
        assert_eq!(set.len(), 2);
        assert!(set.is_mixed());
        assert_eq!(
            set.iter().collect::<Vec<_>>(),
            vec![GeometryFamily::Polygon, GeometryFamily::Point],
            "painter's order: polygon before point"
        );
        let other = {
            let mut other = FamilySet::default();
            other.insert(GeometryFamily::Line);
            other
        };
        assert_eq!(set.union(other).len(), 3);
    }

    // --- LabelOrientation (print/text v1.5, D-A8): the SAME battery, because
    // this is a SECOND `#[serde(flatten)]`-order hazard on the same struct.

    #[test]
    fn label_orientation_defaults_to_horizontal_and_names_itself() {
        assert_eq!(LabelOrientation::default(), LabelOrientation::Horizontal);
        assert!(LabelOrientation::default().is_horizontal());
        assert!(!LabelOrientation::Vertical.is_horizontal());
        assert_eq!(
            LabelOrientation::ALL,
            [LabelOrientation::Horizontal, LabelOrientation::Vertical],
        );
        assert_eq!(LabelOrientation::Horizontal.label(), "Horizontal");
        assert_eq!(LabelOrientation::Vertical.label(), "Vertical (CJK)");
        for orientation in LabelOrientation::ALL {
            assert!(!orientation.label().is_empty());
        }
    }

    #[test]
    fn a_horizontal_symbol_style_serializes_exactly_as_it_did_before_v15() {
        let symbol = SymbolStyle::new("name");
        assert_eq!(symbol.orientation(), LabelOrientation::Horizontal);
        let compact = serde_json::to_string(&symbol).expect("compact");
        assert_eq!(
            compact,
            r#"{"text_field":"name","text_color":"000000ff","text_size":12.0,"halo_color":"ffffffff","halo_width":1.0}"#,
            "the v1.3/v1.4 bytes, character for character",
        );
        assert!(!compact.contains("orientation"));
        assert!(
            !serde_json::to_string_pretty(&symbol)
                .expect("pretty")
                .contains("orientation"),
            "pretty JSON skips it too",
        );
        // And a BOLD style keeps `weight` last-but-one, `orientation` absent.
        let mut bold = SymbolStyle::new("name");
        bold.set_weight(LabelWeight::Bold);
        let json = serde_json::to_string(&bold).expect("compact");
        assert!(json.ends_with(r#""weight":"bold"}"#), "{json}");
    }

    #[test]
    fn a_vertical_orientation_round_trips_through_a_snake_case_tag() {
        let mut symbol = SymbolStyle::new("name");
        symbol.set_orientation(LabelOrientation::Vertical);
        let json = serde_json::to_string(&symbol).expect("compact");
        assert!(json.contains(r#""orientation":"vertical""#), "{json}");
        assert!(
            json.ends_with(r#""orientation":"vertical"}"#),
            "declared LAST, so it serializes last: {json}",
        );
        let back: SymbolStyle = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, symbol);
        assert_eq!(back.orientation(), LabelOrientation::Vertical);
        // With bold as well, the order is weight then orientation.
        symbol.set_weight(LabelWeight::Bold);
        let both = serde_json::to_string(&symbol).expect("compact");
        assert!(
            both.ends_with(r#""weight":"bold","orientation":"vertical"}"#),
            "declaration order is byte-observable: {both}",
        );
    }

    #[test]
    fn a_pre_v15_symbol_document_with_no_orientation_key_loads_as_horizontal() {
        let legacy = r#"{"text_field":"name","text_color":"000000ff","text_size":12.0,
            "halo_color":"ffffffff","halo_width":1.0,"weight":"bold"}"#;
        let symbol: SymbolStyle = serde_json::from_str(legacy).expect("a v1.4 symbol style loads");
        assert_eq!(symbol.orientation(), LabelOrientation::Horizontal);
        assert_eq!(symbol.weight(), LabelWeight::Bold);
    }

    #[test]
    fn a_full_v13_project_with_overrides_still_re_saves_byte_identically() {
        // THE gate, re-run for the second flattened field: the same whole
        // document, the same load-and-re-save, now with `orientation` in the
        // struct. A field inserted anywhere but LAST, or one that stops being
        // skipped at its default, fails here.
        assert!(
            !V13_PROJECT_JSON.contains("orientation"),
            "the fixture predates the field",
        );
        let loaded = crate::project::Project::from_json_string(V13_PROJECT_JSON)
            .expect("the v1.3 document parses");
        assert_eq!(
            loaded.to_json_string().expect("re-serialize"),
            V13_PROJECT_JSON,
            "adding LabelOrientation must not move one byte of a horizontal project",
        );
        let compact = serde_json::to_string(&loaded).expect("compact");
        assert!(!compact.contains("orientation"), "{compact}");
        assert_eq!(
            serde_json::from_str::<crate::project::Project>(&compact).expect("compact parses"),
            loaded,
        );
        let (_, set) = loaded.styles.iter().next().expect("a styled layer");
        let LayerStyle::Symbol(symbol) = set.base() else {
            panic!("the first style is the symbol one");
        };
        assert_eq!(symbol.orientation(), LabelOrientation::Horizontal);
        assert_eq!(set.overridden().len(), 2, "per-family overrides present");
    }

    #[test]
    fn a_vertical_symbol_style_survives_the_whole_project_round_trip() {
        use crate::layer::{Layer, LayerKind, LayerStack, VectorSource};
        use crate::project::Project;

        let mut project = Project::new("Vertical fixture");
        let mut stack = LayerStack::new();
        let id = stack.add(Layer::new(
            "Labels",
            LayerKind::Vector(VectorSource::LocalGeoJson {
                path: "labels.geojson".to_string(),
            }),
        ));
        project.layers = stack;
        let mut symbol = SymbolStyle::new("name");
        symbol.set_orientation(LabelOrientation::Vertical);
        symbol.set_weight(LabelWeight::Bold);
        let mut set = LayerStyleSet::new(LayerStyle::Symbol(symbol));
        set.set_override(
            GeometryFamily::Line,
            LayerStyle::Line(LineStyle::new(Color::BLACK, 2.0)),
        );
        project
            .set_style(id, set)
            .expect("LocalGeoJson accepts a layer style");

        let json = project.to_json_string().expect("pretty");
        assert!(json.contains(r#""orientation": "vertical""#), "{json}");
        let back = Project::from_json_string(&json).expect("round trip");
        assert_eq!(back, project);
        let LayerStyle::Symbol(symbol) = back.style(id).expect("a style").base() else {
            panic!("a symbol base");
        };
        assert_eq!(symbol.orientation(), LabelOrientation::Vertical);
        assert_eq!(symbol.weight(), LabelWeight::Bold);
    }

    #[test]
    fn an_unknown_orientation_value_is_refused_rather_than_silently_horizontal() {
        let hostile = r#"{"text_field":"name","text_color":"000000ff","text_size":12.0,
            "halo_color":null,"halo_width":0.0,"orientation":"sideways"}"#;
        assert!(
            serde_json::from_str::<SymbolStyle>(hostile).is_err(),
            "an unknown orientation must not load as Horizontal",
        );
    }

    // --- Renderer (thematic v1.6): the THIRD run of the same battery, because
    // `renderer` is a third field living beside a `#[serde(flatten)]`ed base.

    use crate::renderer::{AttrValue, CategoryClass, GraduatedClass, NoAttributes, Renderer};

    fn properties(json: &str) -> serde_json::Map<String, serde_json::Value> {
        match serde_json::from_str(json) {
            Ok(map) => map,
            Err(error) => panic!("the fixture must parse: {error}"),
        }
    }

    #[test]
    fn a_single_symbol_set_serializes_exactly_as_it_did_before_v16() {
        let set = fill_set();
        assert!(set.is_single_symbol());
        let bare = set.base().clone();
        assert_eq!(
            serde_json::to_string(&set).expect("compact"),
            serde_json::to_string(&bare).expect("compact"),
            "compact JSON is still byte-identical to the bare style",
        );
        assert_eq!(
            serde_json::to_string_pretty(&set).expect("pretty"),
            serde_json::to_string_pretty(&bare).expect("pretty"),
            "pretty JSON is still byte-identical to the bare style",
        );
        assert!(
            !serde_json::to_string(&set)
                .expect("compact")
                .contains("renderer")
        );
    }

    #[test]
    fn a_full_v13_project_with_overrides_re_saves_byte_identically_with_a_renderer_field() {
        // THE gate, third run. A field inserted anywhere but LAST, or one that
        // stops being skipped at its default, fails here — on a WHOLE project
        // document, load-and-re-save, exactly as the two earlier runs do.
        assert!(
            !V13_PROJECT_JSON.contains("renderer"),
            "the fixture predates the field",
        );
        let loaded = crate::project::Project::from_json_string(V13_PROJECT_JSON)
            .expect("the v1.3 document parses");
        assert_eq!(
            loaded.to_json_string().expect("re-serialize"),
            V13_PROJECT_JSON,
            "adding Renderer must not move one byte of a single-symbol project",
        );
        let compact = serde_json::to_string(&loaded).expect("compact");
        assert!(!compact.contains("renderer"), "{compact}");
        assert_eq!(
            serde_json::from_str::<crate::project::Project>(&compact).expect("compact parses"),
            loaded,
        );
        for (_, set) in loaded.styles.iter() {
            assert!(set.is_single_symbol(), "and every style loaded as Single");
            assert_eq!(set.class_count(), 0);
        }
    }

    #[test]
    fn a_renderer_survives_the_flattened_round_trip_in_both_json_shapes() {
        // The hazard `families` already proved once: `#[serde(flatten)]` is
        // greedy about unknown keys, so a sibling field has to be claimed by
        // the OUTER struct before the flattened `LayerStyle` sees it.
        let mut set = fill_set();
        set.set_override(
            GeometryFamily::Line,
            LayerStyle::Line(LineStyle::new(Color::BLACK, 2.0)),
        );
        set.set_renderer(Renderer::categorized(
            "pref",
            [
                CategoryClass::new(
                    AttrValue::text("Tokyo"),
                    LayerStyle::Fill(FillStyle::new(Color::from_rgb(1, 2, 3))),
                ),
                CategoryClass::new(
                    AttrValue::number(42.0).expect("finite"),
                    LayerStyle::Fill(FillStyle::new(Color::from_rgb(4, 5, 6))),
                ),
            ],
            Some(LayerStyle::Fill(FillStyle::new(Color::WHITE))),
        ));

        let compact = serde_json::to_string(&set).expect("compact");
        assert!(
            compact.contains(r#""renderer":{"kind":"categorized""#),
            "{compact}"
        );
        assert!(
            compact.ends_with(r#"}}"#) && compact.find("renderer") > compact.find("families"),
            "declared last, so it serializes after the families: {compact}",
        );
        assert_eq!(
            serde_json::from_str::<LayerStyleSet>(&compact).expect("compact parses"),
            set,
        );
        let pretty = serde_json::to_string_pretty(&set).expect("pretty");
        assert_eq!(
            serde_json::from_str::<LayerStyleSet>(&pretty).expect("pretty parses"),
            set,
        );
        // A pre-v1.6 reader still gets the base out of a v1.6 document.
        let bare: LayerStyle = serde_json::from_str(&compact).expect("a v1.5 reader parses");
        assert_eq!(&bare, set.base());
    }

    #[test]
    fn a_whole_project_carrying_a_graduated_renderer_round_trips() {
        use crate::layer::{Layer, LayerKind, LayerStack, VectorSource};
        use crate::project::Project;

        let mut project = Project::new("Choropleth fixture");
        let mut stack = LayerStack::new();
        let id = stack.add(Layer::new(
            "Prefectures",
            LayerKind::Vector(VectorSource::LocalGeoJson {
                path: "pref.geojson".to_string(),
            }),
        ));
        project.layers = stack;
        let mut set = fill_set();
        set.set_renderer(Renderer::graduated(
            "population",
            [
                GraduatedClass::new(
                    1_000_000.0,
                    LayerStyle::Fill(FillStyle::new(Color::from_rgb(0xed, 0xf8, 0xfb))),
                ),
                GraduatedClass::new(
                    5_000_000.0,
                    LayerStyle::Fill(FillStyle::new(Color::from_rgb(0x23, 0x6c, 0xa6))),
                ),
            ],
            None,
        ));
        project
            .set_style(id, set)
            .expect("a local layer is styleable");

        let json = project.to_json_string().expect("pretty");
        assert!(json.contains(r#""kind": "graduated""#), "{json}");
        let back = Project::from_json_string(&json).expect("round trip");
        assert_eq!(back, project);
        assert_eq!(
            back.to_json_string().expect("re-serialize"),
            json,
            "and a v1.6 document is itself stable across a save",
        );
        let stored = back.style(id).expect("a style");
        assert_eq!(stored.class_count(), 2);
        assert!(!stored.is_single_symbol());
    }

    #[test]
    fn style_for_resolves_the_renderer_first_and_the_family_second() {
        let mut set = fill_set();
        let point_override = LayerStyle::Circle(CircleStyle::new(4.0, Color::WHITE));
        set.set_override(GeometryFamily::Point, point_override.clone());

        // Single: every family answers exactly as `effective` does, for any
        // attributes at all — the pre-v1.6 picture, unchanged.
        for family in GeometryFamily::ALL {
            assert_eq!(&set.style_for(family, &NoAttributes), set.effective(family));
            assert_eq!(
                &set.style_for(family, &properties(r#"{"pref":"Tokyo"}"#)),
                set.effective(family),
            );
        }

        let tokyo = LayerStyle::Fill(FillStyle::new(Color::from_rgb(9, 9, 9)));
        set.set_renderer(Renderer::categorized(
            "pref",
            [CategoryClass::new(AttrValue::text("Tokyo"), tokyo.clone())],
            None,
        ));
        assert_eq!(
            set.style_for(GeometryFamily::Polygon, &properties(r#"{"pref":"Tokyo"}"#)),
            tokyo,
            "a matched class wins over the base",
        );
        assert_eq!(
            set.style_for(GeometryFamily::Point, &properties(r#"{"pref":"Kyoto"}"#)),
            point_override,
            "an unmatched feature falls through to the family override",
        );
        assert_eq!(
            &set.style_for(GeometryFamily::Line, &NoAttributes),
            set.base(),
            "and to the base where there is none",
        );

        // The pre-classified entry point answers identically.
        assert_eq!(set.style_for_class(GeometryFamily::Polygon, Some(0)), tokyo);
        assert_eq!(
            set.style_for_class(GeometryFamily::Point, None),
            point_override
        );
        assert_eq!(
            &set.style_for_class(GeometryFamily::Polygon, Some(99)),
            set.base(),
            "an out-of-range class reads as the fallback, never as a panic",
        );
    }

    #[test]
    fn a_class_of_the_wrong_kind_recolours_a_family_instead_of_erasing_it() {
        // THE hazard per-family overrides exist for, one level up: the
        // tessellator has no (Points, Fill) arm, so a class naming a Fill
        // handed verbatim to the point family would make every point on a
        // mixed layer VANISH the moment it was classified. The class's colour
        // must land on the family's own symbol instead.
        let mut set = fill_set();
        let circle = CircleStyle::new(4.0, Color::WHITE);
        set.set_override(GeometryFamily::Point, LayerStyle::Circle(circle));
        let red = Color::from_rgb(0xd0, 0x20, 0x20);
        set.set_renderer(Renderer::categorized(
            "pref",
            [CategoryClass::new(
                AttrValue::text("Tokyo"),
                LayerStyle::Fill(FillStyle::new(red)),
            )],
            None,
        ));

        let tokyo = properties(r#"{"pref":"Tokyo"}"#);
        let LayerStyle::Circle(point) = set.style_for(GeometryFamily::Point, &tokyo) else {
            panic!("the point family must stay a CIRCLE, or it draws nothing");
        };
        assert_eq!(point.color, red, "but it takes the class's colour");
        assert_eq!(point.radius(), 4.0, "and keeps its own radius");

        // The family the class's own kind matches takes it verbatim, opacity,
        // outline and all.
        let LayerStyle::Fill(polygon) = set.style_for(GeometryFamily::Polygon, &tokyo) else {
            panic!("a fill");
        };
        assert_eq!(polygon.color, red);
        assert_eq!(polygon.opacity(), 1.0, "the CLASS's opacity, verbatim");

        // A class naming a Line over a polygon family stays a fill too — the
        // rule is symmetric, not a special case for points.
        set.set_renderer(Renderer::categorized(
            "pref",
            [CategoryClass::new(
                AttrValue::text("Tokyo"),
                LayerStyle::Line(LineStyle::new(red, 3.0)),
            )],
            None,
        ));
        let LayerStyle::Fill(polygon) = set.style_for(GeometryFamily::Polygon, &tokyo) else {
            panic!("the polygon family must stay a FILL");
        };
        assert_eq!(polygon.color, red);
        assert_eq!(polygon.opacity(), 0.35, "the FAMILY's opacity, kept");
        assert_eq!(
            polygon.outline_color,
            Some(Color::from_hex("1c4b8f").unwrap_or(Color::BLACK)),
            "and its outline",
        );
    }

    #[test]
    fn a_renderer_fallback_style_overrides_the_family_one() {
        let mut set = fill_set();
        set.set_override(
            GeometryFamily::Line,
            LayerStyle::Line(LineStyle::new(Color::BLACK, 2.0)),
        );
        let grey = LayerStyle::Line(LineStyle::new(Color::from_rgb(0x88, 0x88, 0x88), 1.0));
        set.set_renderer(Renderer::categorized(
            "class",
            [CategoryClass::new(
                AttrValue::text("motorway"),
                LayerStyle::Line(LineStyle::new(Color::WHITE, 4.0)),
            )],
            Some(grey.clone()),
        ));
        assert_eq!(
            set.style_for(GeometryFamily::Line, &properties(r#"{"class":"track"}"#)),
            grey,
            "an explicit fallback is what 'everything else' means",
        );
        set.renderer_mut().set_field("kind");
        assert_eq!(set.renderer().field(), Some("kind"));
        assert_eq!(
            set.style_for(GeometryFamily::Line, &properties(r#"{"class":"motorway"}"#)),
            grey,
            "and the field really is the one that is read",
        );
    }

    #[test]
    fn the_classification_key_travels_with_the_set() {
        let mut set = fill_set();
        assert!(set.classification().is_single());
        set.set_renderer(Renderer::categorized(
            "pref",
            [CategoryClass::new(
                AttrValue::text("Tokyo"),
                LayerStyle::Fill(FillStyle::new(Color::BLACK)),
            )],
            None,
        ));
        assert_eq!(set.classification().class_count(), 1);
        // A colour edit inside a class must not move the partition key.
        let before = set.classification();
        match set.renderer_mut().class_style_mut(0) {
            Some(style) => *style = LayerStyle::Fill(FillStyle::new(Color::WHITE)),
            None => panic!("class 0 exists"),
        }
        assert_eq!(set.classification(), before);
    }
}
