//! The renderer model: what decides *which* [`LayerStyle`] a single feature
//! draws with (thematic v1.6).
//!
//! [`crate::style`] answers "what does this layer look like"; this module
//! answers "what does *this feature* look like". The two are deliberately
//! separate layers of the model:
//!
//! ```text
//! LayerStyleSet
//!   base:     LayerStyle            <- one constant paint (pre-v1.6, still the default)
//!   families: FamilyStyles          <- per geometry family override (v1.3)
//!   renderer: Renderer              <- per FEATURE resolution (v1.6, this module)
//! ```
//!
//! A [`Renderer::Single`] set resolves to exactly what it resolved to before
//! this module existed — `LayerStyleSet::effective(family)` — and serializes
//! to exactly the same bytes, because the field is skipped at its default.
//! That is the whole back-compatibility story: a project written by any
//! earlier build loads, re-saves byte-identically, and draws identically.
//!
//! # The one resolution rule
//!
//! [`Renderer::class_of`] maps a feature's attributes onto a **class index**
//! (or [`None`], the fallback). Everything that has to agree on how a feature
//! is painted — the map's mesh partition, the PDF exporter, hit testing —
//! calls that one function rather than re-deriving the rule, which is what
//! makes screen and page provably the same picture.
//!
//! # Reading attributes without owning a format
//!
//! `oxigis-core` holds no OxiGeo types (see the crate docs), and the renderer
//! must resolve against GeoJSON properties, MVT property lists and whatever a
//! future driver brings. The seam is [`Attributes`] — one method, borrowed
//! values, no allocation for the string case — with an implementation for
//! `serde_json`'s map (which *is* GeoJSON's `Properties`) provided here and
//! shells free to add their own.
//!
//! # Bounds
//!
//! A class list is capped at [`MAX_STYLE_CLASSES`] **at resolution time**, not
//! only where a panel builds one: a hand-edited (or hostile) project file
//! naming ten thousand categories must not turn into ten thousand meshes. The
//! classes past the cap are simply never matched, so their features draw with
//! the fallback.

use std::borrow::Cow;

use serde::{Deserialize, Deserializer, Serialize};

use crate::style::{Color, LayerStyle};

/// How many classes one renderer may resolve.
///
/// The cap is a *rendering* budget, not a modelling opinion: the local vector
/// path draws one mesh per class, so the class count is a multiplier on
/// tessellation work and on GPU draw state. 64 covers every thematic map a
/// person reads (a 47-prefecture choropleth, a 12-class land-use map) with
/// room to spare, and keeps the worst case bounded at three families ×
/// 65 buckets.
///
/// Classes stored past the cap are inert — see [`Renderer::class_of`] — never
/// truncated on load, so a file written by a future build that raises the cap
/// still round-trips through this one.
pub const MAX_STYLE_CLASSES: usize = 64;

/// One attribute value, as the renderer model stores it.
///
/// Three kinds, because those are the three JSON scalars a feature attribute
/// can carry that anything can be *compared* against; `null` is not among
/// them, since a null attribute and a missing one read the same to every
/// consumer (the same rule the GeoJSON → MVT property conversion already
/// applies).
///
/// # Wire shape
///
/// Serialized **untagged**, so a category value is written as the plain JSON
/// scalar it came from — `"Tokyo"`, `42.0`, `true` — rather than as a tagged
/// object. A project file's category list therefore reads like the data it
/// classifies.
///
/// # Numbers
///
/// One `f64` for every numeric attribute, so an `i64` from an MVT tile and a
/// JSON integer compare equal without a type dance. Integers beyond 2^53 lose
/// precision — but they lose it *identically* on both sides of the comparison,
/// so matching stays consistent (a feature either always matches its category
/// or never does).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttrValue {
    /// A boolean attribute.
    Bool(bool),
    /// A numeric attribute. Always finite when built through
    /// [`AttrValue::number`], which is the only path the panel and the
    /// classify helpers use.
    Number(f64),
    /// A textual attribute.
    Text(String),
}

impl AttrValue {
    /// A textual value.
    #[must_use]
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text(value.into())
    }

    /// A numeric value, or [`None`] for a non-finite one.
    ///
    /// NaN and the infinities are refused rather than stored: JSON has no
    /// literal for them, so a stored one would serialize as `null` and fail to
    /// load again — a silently corrupted project file. A NaN could never match
    /// anything anyway (`NaN != NaN`).
    #[must_use]
    pub fn number(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self::Number(value))
    }

    /// Whether this value can be written to (and read back from) JSON — false
    /// only for a non-finite number, which [`AttrValue::number`] refuses to
    /// build in the first place.
    #[must_use]
    pub fn is_serializable(&self) -> bool {
        match self {
            Self::Number(value) => value.is_finite(),
            Self::Bool(_) | Self::Text(_) => true,
        }
    }

    /// A borrowed view of this value.
    #[must_use]
    pub fn as_ref(&self) -> AttrRef<'_> {
        match self {
            Self::Bool(flag) => AttrRef::Bool(*flag),
            Self::Number(value) => AttrRef::Number(*value),
            Self::Text(text) => AttrRef::Text(Cow::Borrowed(text)),
        }
    }

    /// Whether a feature's value is the same value as this one — THE equality
    /// rule a categorized renderer matches by.
    ///
    /// Same-kind comparison only: a `"42"` in the data never matches a `42` in
    /// the style, because a renderer that silently coerces is a renderer whose
    /// picture cannot be reasoned about from the file. Numbers compare as
    /// `f64` (so an integer and a float of the same value do match, which is
    /// the case that actually occurs — MVT stores `int_value`, GeoJSON stores
    /// a JSON number, and both arrive here as `f64`).
    #[must_use]
    pub fn matches(&self, value: &AttrRef<'_>) -> bool {
        match (self, value) {
            (Self::Bool(want), AttrRef::Bool(got)) => want == got,
            (Self::Number(want), AttrRef::Number(got)) => want == got,
            (Self::Text(want), AttrRef::Text(got)) => want.as_str() == got.as_ref(),
            _ => false,
        }
    }

    /// A short human label for a legend row or a panel row.
    ///
    /// A whole number prints without a decimal tail (`42`, not `42.0`), which
    /// is what a category list built from integer codes should look like.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Bool(flag) => flag.to_string(),
            Self::Number(value) => format_number(*value),
            Self::Text(text) => text.clone(),
        }
    }
}

/// Formats a number the way a legend row reads it: whole values without a
/// decimal tail, everything else with the shortest round-tripping form.
#[must_use]
pub fn format_number(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{value:.0}")
    } else {
        format!("{value}")
    }
}

/// A borrowed attribute value read out of a feature.
///
/// The read side of [`AttrValue`]: an [`Attributes`] implementation hands one
/// of these back without allocating for the common string case, so classifying
/// a million-feature dataset costs no per-feature `String`.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrRef<'a> {
    /// A boolean attribute.
    Bool(bool),
    /// A numeric attribute.
    Number(f64),
    /// A textual attribute — borrowed from the feature when the source holds a
    /// string, owned when the source holds something that has to be rendered
    /// (a JSON array or object, which reads as its compact JSON text so that
    /// the GeoJSON and MVT paths classify identically).
    Text(Cow<'a, str>),
}

impl AttrRef<'_> {
    /// The value as a finite number, or [`None`].
    ///
    /// Only a genuinely numeric attribute qualifies: a textual `"42"` is NOT
    /// parsed (a graduated renderer over a text field is a modelling mistake,
    /// and silently rescuing it would hide the mistake), and a non-finite
    /// number is refused so that no comparison can accidentally succeed
    /// against NaN.
    #[must_use]
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(value) => value.is_finite().then_some(*value),
            Self::Bool(_) | Self::Text(_) => None,
        }
    }

    /// This value, owned — the shape a classify helper stores in a category
    /// list. [`None`] for a non-finite number (see [`AttrValue::number`]).
    #[must_use]
    pub fn to_value(&self) -> Option<AttrValue> {
        match self {
            Self::Bool(flag) => Some(AttrValue::Bool(*flag)),
            Self::Number(value) => AttrValue::number(*value),
            Self::Text(text) => Some(AttrValue::Text(text.clone().into_owned())),
        }
    }

    /// A short human label, matching [`AttrValue::label`].
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Bool(flag) => flag.to_string(),
            Self::Number(value) => format_number(*value),
            Self::Text(text) => text.clone().into_owned(),
        }
    }
}

/// Anything a renderer can read a feature attribute out of.
///
/// One method on purpose: a renderer asks for ONE named field per feature, so
/// an implementation is free to be a linear scan over a short property list
/// (which is what both GeoJSON and MVT actually are) rather than a map.
pub trait Attributes {
    /// The value stored under `key`, or [`None`] when the feature has no such
    /// attribute (a `null` attribute counts as absent).
    fn value(&self, key: &str) -> Option<AttrRef<'_>>;
}

impl<T> Attributes for &T
where
    T: Attributes + ?Sized,
{
    fn value(&self, key: &str) -> Option<AttrRef<'_>> {
        (**self).value(key)
    }
}

/// GeoJSON's `Properties` *is* this type, so a GeoJSON feature classifies with
/// no adapter at all.
///
/// `null` reads as absent; an array or an object reads as its compact JSON
/// text, which is exactly what the GeoJSON → MVT property conversion writes,
/// so the same feature classifies the same way on both paths.
impl Attributes for serde_json::Map<String, serde_json::Value> {
    fn value(&self, key: &str) -> Option<AttrRef<'_>> {
        match self.get(key)? {
            serde_json::Value::Null => None,
            serde_json::Value::Bool(flag) => Some(AttrRef::Bool(*flag)),
            serde_json::Value::Number(number) => number.as_f64().map(AttrRef::Number),
            serde_json::Value::String(text) => Some(AttrRef::Text(Cow::Borrowed(text))),
            other => Some(AttrRef::Text(Cow::Owned(other.to_string()))),
        }
    }
}

/// A feature with no attributes at all — what a caller that has none to offer
/// passes, so that resolution never needs a second, attribute-free entry
/// point.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoAttributes;

impl Attributes for NoAttributes {
    fn value(&self, _key: &str) -> Option<AttrRef<'_>> {
        None
    }
}

/// One exact-value class of a [`Renderer::Categorized`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoryClass {
    /// The attribute value this class matches.
    pub value: AttrValue,
    /// What a feature in this class draws with — for the geometry family this
    /// style can draw. Another family keeps its own style, RECOLOURED to this
    /// one's colour: see [`class_over_family`].
    pub style: LayerStyle,
}

impl CategoryClass {
    /// A class matching `value`, drawn with `style`.
    #[must_use]
    pub fn new(value: AttrValue, style: LayerStyle) -> Self {
        Self { value, style }
    }
}

/// One range class of a [`Renderer::Graduated`], named by its **upper** bound.
///
/// Class `i` covers `(upper[i-1], upper[i]]`, the first covers
/// `(-inf, upper[0]]`, and the LAST is open above — see [`Renderer::class_of`]
/// for why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraduatedClass {
    /// Inclusive upper bound. Always finite: kept private behind
    /// [`GraduatedClass::set_upper`], which sanitizes, because a NaN bound
    /// would serialize as JSON `null` and never load again.
    #[serde(deserialize_with = "deserialize_finite")]
    upper: f64,
    /// What a feature in this range draws with.
    pub style: LayerStyle,
}

impl GraduatedClass {
    /// A class whose range ends at `upper`, drawn with `style`. A non-finite
    /// bound is sanitized — see [`GraduatedClass::set_upper`].
    #[must_use]
    pub fn new(upper: f64, style: LayerStyle) -> Self {
        Self {
            upper: sanitize_bound(upper),
            style,
        }
    }

    /// The inclusive upper bound, guaranteed finite.
    #[must_use]
    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// Sets the upper bound. NaN becomes `0.0` and the infinities become the
    /// finite extremes, so the value stays ordered and serializable.
    pub fn set_upper(&mut self, upper: f64) {
        self.upper = sanitize_bound(upper);
    }
}

/// Maps a non-finite bound onto the nearest finite one it can stand for.
fn sanitize_bound(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else if value == f64::INFINITY {
        f64::MAX
    } else if value == f64::NEG_INFINITY {
        f64::MIN
    } else {
        0.0
    }
}

/// A `serde(deserialize_with = ...)` helper that sanitizes an incoming bound,
/// so a hand-edited `1e400` loads as a finite extreme instead of poisoning the
/// class list.
fn deserialize_finite<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let value = f64::deserialize(deserializer)?;
    Ok(sanitize_bound(value))
}

/// Which of the four style kinds a style is — the discriminant
/// [`class_over_family`] compares, without pulling in the panel's own enum.
#[must_use]
fn style_shape(style: &LayerStyle) -> u8 {
    match style {
        LayerStyle::Fill(_) => 0,
        LayerStyle::Line(_) => 1,
        LayerStyle::Circle(_) => 2,
        LayerStyle::Symbol(_) => 3,
    }
}

/// The main colour of `style` — a fill's fill, a line's stroke, a circle's
/// disc, a label's text.
#[must_use]
pub fn style_color(style: &LayerStyle) -> Color {
    match style {
        LayerStyle::Fill(fill) => fill.color,
        LayerStyle::Line(line) => line.color,
        LayerStyle::Circle(circle) => circle.color,
        LayerStyle::Symbol(symbol) => symbol.text_color,
    }
}

/// `style` with its main colour replaced, everything else kept.
///
/// A fill keeps its opacity and outline, a line its width, a circle its radius
/// and stroke, a symbol its text field and halo — so recolouring changes
/// exactly one thing about how a feature draws.
#[must_use]
pub fn recolor_style(style: &LayerStyle, color: Color) -> LayerStyle {
    match style {
        LayerStyle::Fill(fill) => {
            let mut fill = *fill;
            fill.color = color;
            LayerStyle::Fill(fill)
        }
        LayerStyle::Line(line) => {
            let mut line = *line;
            line.color = color;
            LayerStyle::Line(line)
        }
        LayerStyle::Circle(circle) => {
            let mut circle = *circle;
            circle.color = color;
            LayerStyle::Circle(circle)
        }
        LayerStyle::Symbol(symbol) => {
            let mut symbol = symbol.clone();
            symbol.text_color = color;
            LayerStyle::Symbol(symbol)
        }
    }
}

/// Composes one class's style over one family's style — THE rule that keeps a
/// classified mixed-geometry layer drawing all of itself.
///
/// A class names ONE style, but a layer can draw three geometry families, and
/// the tessellator's dispatch table has no `(Points, Fill)` arm: handing a
/// point family a class's `Fill` would silently erase every point on the map
/// — which is exactly the bug per-family overrides were introduced to fix in
/// v1.3, re-introduced one level up. So:
///
/// * **same kind** (the overwhelmingly common single-family case) — the
///   class's style is taken verbatim, including its width, radius and halo;
/// * **different kind** — the FAMILY's style is kept, recoloured to the
///   class's colour. The class still says "this feature is red"; the family
///   still says "points are 4 px circles".
///
/// Owned rather than borrowed on purpose: the recoloured case is a value that
/// exists nowhere in the model, and returning a `Cow` here would push the
/// same allocation onto every caller with a lifetime to reason about.
#[must_use]
pub fn class_over_family(class: &LayerStyle, family: &LayerStyle) -> LayerStyle {
    if style_shape(class) == style_shape(family) {
        class.clone()
    } else {
        recolor_style(family, style_color(class))
    }
}

/// Which kind of renderer a set uses — the panel's combo, and the one thing a
/// caller switching kinds names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RendererKind {
    /// One paint for every feature.
    #[default]
    Single,
    /// A paint per exact attribute value.
    Categorized,
    /// A paint per numeric range.
    Graduated,
}

impl RendererKind {
    /// Every kind, in panel order.
    pub const ALL: [RendererKind; 3] = [Self::Single, Self::Categorized, Self::Graduated];

    /// The panel label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Single => "Single symbol",
            Self::Categorized => "Categorized",
            Self::Graduated => "Graduated",
        }
    }
}

/// The body of a [`Renderer::Categorized`].
///
/// A named struct behind a [`Box`] rather than an inline struct variant: a
/// `LayerStyleSet` is carried by value through the undo stack, the layer
/// snapshot and the render-thread op queue, and an inline body would put ~100
/// bytes of classification into every one of them — including the
/// overwhelmingly common [`Renderer::Single`] case, which needs none of it.
/// Boxing keeps a whole [`Renderer`] at one pointer.
///
/// Serde-transparent: an internally tagged newtype variant splices its inner
/// struct's fields into the same map, so the wire shape is identical to the
/// inline variant it replaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategorizedSpec {
    /// Attribute the classes match against.
    pub field: String,
    /// The classes, in legend order. Only the first [`MAX_STYLE_CLASSES`] ever
    /// match.
    #[serde(default)]
    pub categories: Vec<CategoryClass>,
    /// What a feature matching no class draws with; [`None`] means the layer's
    /// own style.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<LayerStyle>,
}

/// The body of a [`Renderer::Graduated`] — the twin of [`CategorizedSpec`],
/// boxed for the same reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraduatedSpec {
    /// Numeric attribute the ranges are taken on.
    pub field: String,
    /// The classes, ascending by [`GraduatedClass::upper`]. Only the first
    /// [`MAX_STYLE_CLASSES`] ever match.
    #[serde(default)]
    pub classes: Vec<GraduatedClass>,
    /// What a feature with no usable value draws with; [`None`] means the
    /// layer's own style.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<LayerStyle>,
}

/// How a layer resolves one feature's style.
///
/// Stored on [`crate::style::LayerStyleSet`] and skipped when
/// [`Renderer::Single`], so every project written before v1.6 round-trips
/// byte-identically.
///
/// # Wire shape
///
/// Internally tagged on `kind` (`"single"` / `"categorized"` /
/// `"graduated"`), so the classification reads as one legible object:
///
/// ```json
/// { "kind": "categorized",
///   "field": "prefecture",
///   "categories": [ { "value": "Tokyo", "style": { "type": "fill", … } } ] }
/// ```
///
/// # Size
///
/// Both classified variants are [`Box`]ed, so the whole enum is one pointer
/// wide and [`Renderer::Single`] — what every unclassified layer is — costs
/// nothing but the discriminant. See [`CategorizedSpec`] for why that matters.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Renderer {
    /// Every feature draws with the layer's own style — the pre-v1.6 model,
    /// and what a set that says nothing means.
    #[default]
    Single,
    /// One class per exact attribute value.
    Categorized(Box<CategorizedSpec>),
    /// One class per numeric range, named by ascending upper bounds.
    Graduated(Box<GraduatedSpec>),
}

impl Renderer {
    /// A categorized renderer over `field`.
    ///
    /// Classes whose value is not serializable (a non-finite number) are
    /// dropped: they could never match, and they would break the project file.
    #[must_use]
    pub fn categorized(
        field: impl Into<String>,
        categories: impl IntoIterator<Item = CategoryClass>,
        fallback: Option<LayerStyle>,
    ) -> Self {
        Self::Categorized(Box::new(CategorizedSpec {
            field: field.into(),
            categories: categories
                .into_iter()
                .filter(|class| class.value.is_serializable())
                .collect(),
            fallback,
        }))
    }

    /// A graduated renderer over `field`, with the classes sorted ascending by
    /// upper bound (the order [`Renderer::class_of`] scans them in).
    #[must_use]
    pub fn graduated(
        field: impl Into<String>,
        classes: impl IntoIterator<Item = GraduatedClass>,
        fallback: Option<LayerStyle>,
    ) -> Self {
        let mut classes: Vec<GraduatedClass> = classes.into_iter().collect();
        // `total_cmp` rather than `partial_cmp`: every bound is finite by
        // construction, but a total order is what `sort_by` needs to be
        // panic-free by type rather than by argument.
        classes.sort_by(|left, right| left.upper.total_cmp(&right.upper));
        Self::Graduated(Box::new(GraduatedSpec {
            field: field.into(),
            classes,
            fallback,
        }))
    }

    /// Whether this is the default renderer — the serialization skip
    /// condition, and what keeps every pre-v1.6 project byte-identical.
    #[must_use]
    pub fn is_single(&self) -> bool {
        matches!(self, Self::Single)
    }

    /// Which kind this is.
    #[must_use]
    pub fn kind(&self) -> RendererKind {
        match self {
            Self::Single => RendererKind::Single,
            Self::Categorized(_) => RendererKind::Categorized,
            Self::Graduated(_) => RendererKind::Graduated,
        }
    }

    /// The attribute the classes are taken on, or [`None`] for
    /// [`Renderer::Single`].
    #[must_use]
    pub fn field(&self) -> Option<&str> {
        match self {
            Self::Single => None,
            Self::Categorized(spec) => Some(&spec.field),
            Self::Graduated(spec) => Some(&spec.field),
        }
    }

    /// Points the renderer at another attribute; a no-op for
    /// [`Renderer::Single`].
    pub fn set_field(&mut self, value: impl Into<String>) {
        match self {
            Self::Single => {}
            Self::Categorized(spec) => spec.field = value.into(),
            Self::Graduated(spec) => spec.field = value.into(),
        }
    }

    /// How many classes can actually match — the stored count, capped at
    /// [`MAX_STYLE_CLASSES`].
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.stored_class_count().min(MAX_STYLE_CLASSES)
    }

    /// How many classes the model holds, cap or no cap.
    #[must_use]
    pub fn stored_class_count(&self) -> usize {
        match self {
            Self::Single => 0,
            Self::Categorized(spec) => spec.categories.len(),
            Self::Graduated(spec) => spec.classes.len(),
        }
    }

    /// How many stored classes are past [`MAX_STYLE_CLASSES`] and therefore
    /// inert — what a panel turns into a notice.
    #[must_use]
    pub fn overflow_class_count(&self) -> usize {
        self.stored_class_count().saturating_sub(MAX_STYLE_CLASSES)
    }

    /// Which class `attributes` fall into: `Some(index)` for a matched class,
    /// [`None`] for the fallback.
    ///
    /// The rules, and every way a feature reaches the fallback:
    ///
    /// * [`Renderer::Single`] — always [`None`]; there are no classes.
    /// * the attribute is **missing** (or JSON `null`);
    /// * categorized: no class value **equals** the attribute (including
    ///   every cross-kind comparison — a number never matches a string);
    /// * graduated: the attribute is **not a finite number** (a text, a
    ///   boolean, a NaN);
    /// * either: the matching class is stored past [`MAX_STYLE_CLASSES`].
    ///
    /// A graduated feature ABOVE the last class's upper bound lands in that
    /// last class rather than in the fallback: the top bound of a
    /// classification is produced by arithmetic (an equal interval is
    /// `min + n·step`, which can round a hair below the true maximum), so a
    /// closed top would silently drop the single most interesting feature on
    /// the map. The last class is therefore documented, and drawn in the
    /// panel, as open above.
    #[must_use]
    pub fn class_of<A>(&self, attributes: &A) -> Option<usize>
    where
        A: Attributes + ?Sized,
    {
        match self {
            Self::Single => None,
            Self::Categorized(spec) => {
                let value = attributes.value(&spec.field)?;
                spec.categories
                    .iter()
                    .take(MAX_STYLE_CLASSES)
                    .position(|class| class.value.matches(&value))
            }
            Self::Graduated(spec) => {
                let value = attributes.value(&spec.field)?.as_number()?;
                let classes = &spec.classes;
                let capped = classes.len().min(MAX_STYLE_CLASSES);
                let last = capped.checked_sub(1)?;
                for (index, class) in classes.iter().take(capped).enumerate() {
                    if value <= class.upper || index == last {
                        return Some(index);
                    }
                }
                None
            }
        }
    }

    /// The style class `class` draws with: the class's own style, or the
    /// fallback for [`None`]. A [`None`] return means "the layer's own style"
    /// — [`crate::style::LayerStyleSet::style_for`] is what resolves that.
    ///
    /// A class index that names nothing — past the cap, past the end of the
    /// list, left over from a shortened class list — reads as the fallback
    /// rather than as a panic or an empty picture.
    #[must_use]
    pub fn class_style(&self, class: Option<usize>) -> Option<&LayerStyle> {
        let stored = match class {
            Some(index) if index < MAX_STYLE_CLASSES => match self {
                Self::Single => None,
                Self::Categorized(spec) => spec.categories.get(index).map(|class| &class.style),
                Self::Graduated(spec) => spec.classes.get(index).map(|class| &class.style),
            },
            _ => None,
        };
        stored.or_else(|| self.fallback())
    }

    /// One class's style, for editing.
    pub fn class_style_mut(&mut self, index: usize) -> Option<&mut LayerStyle> {
        match self {
            Self::Single => None,
            Self::Categorized(spec) => spec.categories.get_mut(index).map(|class| &mut class.style),
            Self::Graduated(spec) => spec.classes.get_mut(index).map(|class| &mut class.style),
        }
    }

    /// The style a feature matching no class draws with.
    #[must_use]
    pub fn fallback(&self) -> Option<&LayerStyle> {
        match self {
            Self::Single => None,
            Self::Categorized(spec) => spec.fallback.as_ref(),
            Self::Graduated(spec) => spec.fallback.as_ref(),
        }
    }

    /// The fallback style, for editing.
    pub fn fallback_mut(&mut self) -> Option<&mut Option<LayerStyle>> {
        match self {
            Self::Single => None,
            Self::Categorized(spec) => Some(&mut spec.fallback),
            Self::Graduated(spec) => Some(&mut spec.fallback),
        }
    }

    /// The class list a categorized renderer holds.
    #[must_use]
    pub fn categories(&self) -> &[CategoryClass] {
        match self {
            Self::Categorized(spec) => &spec.categories,
            Self::Single | Self::Graduated(_) => &[],
        }
    }

    /// The class list a graduated renderer holds.
    #[must_use]
    pub fn graduated_classes(&self) -> &[GraduatedClass] {
        match self {
            Self::Graduated(spec) => &spec.classes,
            Self::Single | Self::Categorized(_) => &[],
        }
    }

    /// Drops class `index`, keeping the rest in order. Returns whether
    /// anything was removed.
    pub fn remove_class(&mut self, index: usize) -> bool {
        match self {
            Self::Single => false,
            Self::Categorized(spec) => {
                if index < spec.categories.len() {
                    spec.categories.remove(index);
                    true
                } else {
                    false
                }
            }
            Self::Graduated(spec) => {
                if index < spec.classes.len() {
                    spec.classes.remove(index);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// The legend text for class `index` — the value for a categorized class,
    /// the range for a graduated one (with the last one shown open above,
    /// exactly as [`Renderer::class_of`] resolves it).
    #[must_use]
    pub fn class_label(&self, index: usize) -> String {
        match self {
            Self::Single => String::new(),
            Self::Categorized(spec) => spec
                .categories
                .get(index)
                .map(|class| class.value.label())
                .unwrap_or_default(),
            Self::Graduated(spec) => {
                let classes = &spec.classes;
                let Some(class) = classes.get(index) else {
                    return String::new();
                };
                let last = index + 1 == classes.len().min(MAX_STYLE_CLASSES);
                let lower = index
                    .checked_sub(1)
                    .and_then(|previous| classes.get(previous))
                    .map(GraduatedClass::upper);
                match (lower, last) {
                    (None, true) => "all values".to_string(),
                    (None, false) => format!("≤ {}", format_number(class.upper())),
                    (Some(lower), true) => format!("> {}", format_number(lower)),
                    (Some(lower), false) => {
                        format!(
                            "{} – {}",
                            format_number(lower),
                            format_number(class.upper())
                        )
                    }
                }
            }
        }
    }

    /// The part of this renderer a tessellated mesh depends on.
    ///
    /// Two renderers with the same [`Classification`] partition every dataset
    /// into the same buckets, so a restyle that leaves it unchanged — every
    /// colour edit, every width drag — needs no re-partition, only a repaint.
    #[must_use]
    pub fn classification(&self) -> Classification {
        match self {
            Self::Single => Classification::Single,
            Self::Categorized(spec) => Classification::Categorized {
                field: spec.field.clone(),
                values: spec
                    .categories
                    .iter()
                    .take(MAX_STYLE_CLASSES)
                    .map(|class| class.value.clone())
                    .collect(),
            },
            Self::Graduated(spec) => Classification::Graduated {
                field: spec.field.clone(),
                uppers: spec
                    .classes
                    .iter()
                    .take(MAX_STYLE_CLASSES)
                    .map(GraduatedClass::upper)
                    .collect(),
            },
        }
    }
}

/// The classification half of a [`Renderer`] — field and match keys, without
/// the styles.
///
/// A comparison key, not a second resolution path: it exists so that a caller
/// holding a tessellated mesh can ask "does the new style partition features
/// the same way?" in one `==` instead of re-classifying the dataset.
///
/// # Equality is IDENTITY, not arithmetic
///
/// [`PartialEq`] is written by hand rather than derived, and numbers compare
/// by BIT PATTERN. The question this type answers is "is this the same
/// partition?", so two NaN bounds at the same index are the same partition —
/// whereas IEEE inequality (`NaN != NaN`) would make a classification unequal
/// to *itself*, and a caller keying a cached mesh on it would re-partition the
/// whole dataset on every single frame of a colour drag.
///
/// A NaN cannot arrive through the constructors or through JSON
/// ([`AttrValue::number`] refuses it, [`GraduatedClass::new`] sanitizes it, and
/// JSON has no literal for it), but [`CategorizedSpec`]'s fields are public, so
/// the invariant is enforced here rather than assumed.
#[derive(Debug, Clone, Default)]
pub enum Classification {
    /// One bucket for everything.
    #[default]
    Single,
    /// Exact-value buckets on `field`, in order.
    Categorized {
        /// The attribute matched against.
        field: String,
        /// The values, in class order.
        values: Vec<AttrValue>,
    },
    /// Range buckets on `field`, ascending.
    Graduated {
        /// The attribute the ranges are taken on.
        field: String,
        /// The inclusive upper bounds, ascending; the last is open above.
        uppers: Vec<f64>,
    },
}

impl PartialEq for Classification {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Single, Self::Single) => true,
            (
                Self::Categorized { field, values },
                Self::Categorized {
                    field: other_field,
                    values: other_values,
                },
            ) => {
                field == other_field
                    && values.len() == other_values.len()
                    && values
                        .iter()
                        .zip(other_values)
                        .all(|(left, right)| same_value(left, right))
            }
            (
                Self::Graduated { field, uppers },
                Self::Graduated {
                    field: other_field,
                    uppers: other_uppers,
                },
            ) => {
                field == other_field
                    && uppers.len() == other_uppers.len()
                    && uppers
                        .iter()
                        .zip(other_uppers)
                        .all(|(left, right)| left.to_bits() == right.to_bits())
            }
            _ => false,
        }
    }
}

/// Whether two stored values are the SAME value — bit equality for numbers, so
/// a NaN is equal to itself. See [`Classification`]'s docs.
fn same_value(left: &AttrValue, right: &AttrValue) -> bool {
    match (left, right) {
        (AttrValue::Number(left), AttrValue::Number(right)) => left.to_bits() == right.to_bits(),
        _ => left == right,
    }
}

impl Classification {
    /// How many classes this partition has (never more than
    /// [`MAX_STYLE_CLASSES`], since a [`Renderer`] caps it on the way in).
    #[must_use]
    pub fn class_count(&self) -> usize {
        match self {
            Self::Single => 0,
            Self::Categorized { values, .. } => values.len(),
            Self::Graduated { uppers, .. } => uppers.len(),
        }
    }

    /// Whether every feature falls in one bucket — the case a caller can skip
    /// partitioning entirely for.
    #[must_use]
    pub fn is_single(&self) -> bool {
        matches!(self, Self::Single)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::{Color, FillStyle, LineStyle};

    fn fill(red: u8) -> LayerStyle {
        LayerStyle::Fill(FillStyle::new(Color::from_rgb(red, 0, 0)))
    }

    fn properties(json: &str) -> serde_json::Map<String, serde_json::Value> {
        match serde_json::from_str(json) {
            Ok(map) => map,
            Err(error) => panic!("the fixture must parse: {error}"),
        }
    }

    #[test]
    fn a_default_renderer_is_single_and_says_so() {
        let renderer = Renderer::default();
        assert!(renderer.is_single());
        assert_eq!(renderer.kind(), RendererKind::Single);
        assert_eq!(renderer.field(), None);
        assert_eq!(renderer.class_count(), 0);
        assert_eq!(renderer.class_of(&NoAttributes), None);
        assert_eq!(renderer.class_style(None), None);
        assert_eq!(renderer.classification(), Classification::Single);
        assert!(renderer.classification().is_single());
    }

    #[test]
    fn attr_values_serialize_as_the_plain_scalars_they_came_from() {
        let text = AttrValue::text("Tokyo");
        let number = AttrValue::number(42.0).expect("finite");
        let flag = AttrValue::Bool(true);
        assert_eq!(
            serde_json::to_string(&text).expect("serialize"),
            "\"Tokyo\""
        );
        assert_eq!(serde_json::to_string(&number).expect("serialize"), "42.0");
        assert_eq!(serde_json::to_string(&flag).expect("serialize"), "true");

        // And every one of them comes back as the same variant.
        assert_eq!(
            serde_json::from_str::<AttrValue>("\"Tokyo\"").expect("parse"),
            text
        );
        assert_eq!(
            serde_json::from_str::<AttrValue>("42").expect("an integer is a number"),
            number
        );
        assert_eq!(
            serde_json::from_str::<AttrValue>("true").expect("parse"),
            flag
        );
    }

    #[test]
    fn a_non_finite_number_is_refused_rather_than_stored_as_json_null() {
        assert!(AttrValue::number(f64::NAN).is_none());
        assert!(AttrValue::number(f64::INFINITY).is_none());
        assert!(AttrValue::number(f64::NEG_INFINITY).is_none());
        assert!(AttrValue::number(0.0).is_some());
        // The guard exists because serde_json writes a non-finite float as
        // `null`, which never loads back as an AttrValue.
        let poisoned = AttrValue::Number(f64::NAN);
        assert!(!poisoned.is_serializable());
        let json = serde_json::to_string(&poisoned).expect("serde_json writes null");
        assert_eq!(json, "null");
        assert!(serde_json::from_str::<AttrValue>(&json).is_err());
        // ... which is why the constructor filters it out.
        let renderer = Renderer::categorized(
            "type",
            [
                CategoryClass::new(poisoned, fill(1)),
                CategoryClass::new(AttrValue::text("ok"), fill(2)),
            ],
            None,
        );
        assert_eq!(renderer.class_count(), 1);
        assert!(serde_json::to_string(&renderer).is_ok());
    }

    #[test]
    fn matching_is_same_kind_only_and_numbers_unify_across_int_and_float() {
        let text = AttrValue::text("42");
        let number = AttrValue::number(42.0).expect("finite");
        assert!(text.matches(&AttrRef::Text(Cow::Borrowed("42"))));
        assert!(!text.matches(&AttrRef::Number(42.0)), "no silent coercion");
        assert!(number.matches(&AttrRef::Number(42.0)));
        assert!(!number.matches(&AttrRef::Text(Cow::Borrowed("42"))));
        assert!(!number.matches(&AttrRef::Bool(true)));
        assert!(AttrValue::Bool(false).matches(&AttrRef::Bool(false)));
        // NaN on the data side matches nothing, including a NaN category.
        assert!(!number.matches(&AttrRef::Number(f64::NAN)));
        assert!(!AttrValue::Number(f64::NAN).matches(&AttrRef::Number(f64::NAN)));
    }

    #[test]
    fn a_geojson_property_map_reads_every_json_scalar_and_treats_null_as_absent() {
        let map = properties(
            r#"{"name":"Tokyo","pop":13960000,"ratio":0.5,"capital":true,
                "empty":null,"tags":["a","b"],"nested":{"k":1}}"#,
        );
        assert_eq!(
            map.value("name"),
            Some(AttrRef::Text(Cow::Borrowed("Tokyo")))
        );
        assert_eq!(map.value("pop"), Some(AttrRef::Number(13_960_000.0)));
        assert_eq!(map.value("ratio"), Some(AttrRef::Number(0.5)));
        assert_eq!(map.value("capital"), Some(AttrRef::Bool(true)));
        assert_eq!(map.value("empty"), None, "null reads as absent");
        assert_eq!(map.value("missing"), None);
        // Arrays and objects read as their compact JSON text — the same thing
        // the GeoJSON -> MVT property conversion writes, so both paths agree.
        assert_eq!(
            map.value("tags").map(|value| value.label()),
            Some(r#"["a","b"]"#.to_string())
        );
        assert_eq!(
            map.value("nested").map(|value| value.label()),
            Some(r#"{"k":1}"#.to_string())
        );
    }

    #[test]
    fn a_categorized_renderer_resolves_by_exact_value_and_falls_back_otherwise() {
        let renderer = Renderer::categorized(
            "pref",
            [
                CategoryClass::new(AttrValue::text("Tokyo"), fill(1)),
                CategoryClass::new(AttrValue::text("Osaka"), fill(2)),
            ],
            Some(fill(3)),
        );
        assert_eq!(renderer.kind(), RendererKind::Categorized);
        assert_eq!(renderer.field(), Some("pref"));
        assert_eq!(renderer.class_count(), 2);
        assert_eq!(
            renderer.class_of(&properties(r#"{"pref":"Tokyo"}"#)),
            Some(0)
        );
        assert_eq!(
            renderer.class_of(&properties(r#"{"pref":"Osaka"}"#)),
            Some(1)
        );
        // Missing field, null field, wrong value, wrong kind: all fallback.
        assert_eq!(renderer.class_of(&properties(r#"{"other":1}"#)), None);
        assert_eq!(renderer.class_of(&properties(r#"{"pref":null}"#)), None);
        assert_eq!(renderer.class_of(&properties(r#"{"pref":"Kyoto"}"#)), None);
        assert_eq!(renderer.class_of(&properties(r#"{"pref":42}"#)), None);
        assert_eq!(renderer.class_style(None), Some(&fill(3)));
        assert_eq!(renderer.class_style(Some(0)), Some(&fill(1)));
        assert_eq!(
            renderer.class_style(Some(9)),
            Some(&fill(3)),
            "an out-of-range class reads as the fallback, never as a panic",
        );
    }

    #[test]
    fn a_categorized_renderer_without_a_fallback_defers_to_the_layer_style() {
        let renderer = Renderer::categorized(
            "pref",
            [CategoryClass::new(AttrValue::text("Tokyo"), fill(1))],
            None,
        );
        assert_eq!(renderer.class_style(None), None, "= the layer's own style");
        assert_eq!(renderer.fallback(), None);
    }

    #[test]
    fn a_graduated_renderer_sorts_its_breaks_and_resolves_by_upper_bound() {
        let renderer = Renderer::graduated(
            "pop",
            [
                GraduatedClass::new(300.0, fill(3)),
                GraduatedClass::new(100.0, fill(1)),
                GraduatedClass::new(200.0, fill(2)),
            ],
            Some(fill(9)),
        );
        let uppers: Vec<f64> = renderer
            .graduated_classes()
            .iter()
            .map(GraduatedClass::upper)
            .collect();
        assert_eq!(uppers, vec![100.0, 200.0, 300.0], "sorted on the way in");

        assert_eq!(renderer.class_of(&properties(r#"{"pop":0}"#)), Some(0));
        assert_eq!(renderer.class_of(&properties(r#"{"pop":100}"#)), Some(0));
        assert_eq!(
            renderer.class_of(&properties(r#"{"pop":100.0001}"#)),
            Some(1),
            "the bound is inclusive above, exclusive below",
        );
        assert_eq!(renderer.class_of(&properties(r#"{"pop":-9999}"#)), Some(0));
        assert_eq!(renderer.class_of(&properties(r#"{"pop":250}"#)), Some(2));
        assert_eq!(renderer.class_of(&properties(r#"{"pop":300}"#)), Some(2));
    }

    #[test]
    fn a_value_above_the_last_break_lands_in_the_last_class_not_the_fallback() {
        // THE rounding hazard: an equal interval's top bound is
        // `min + n*step`, which can sit a hair below the true maximum. A
        // closed top would silently drop the single most interesting feature
        // on the map, so the last class is open above.
        let renderer = Renderer::graduated(
            "pop",
            [
                GraduatedClass::new(10.0, fill(1)),
                GraduatedClass::new(20.0, fill(2)),
            ],
            Some(fill(9)),
        );
        assert_eq!(renderer.class_of(&properties(r#"{"pop":20.5}"#)), Some(1));
        assert_eq!(
            renderer.class_of(&properties(r#"{"pop":1e300}"#)),
            Some(1),
            "however far above",
        );
        assert_eq!(renderer.class_label(0), "≤ 10");
        assert_eq!(renderer.class_label(1), "> 10", "drawn as open above");
    }

    #[test]
    fn a_graduated_renderer_refuses_every_non_numeric_value() {
        let renderer =
            Renderer::graduated("pop", [GraduatedClass::new(10.0, fill(1))], Some(fill(9)));
        assert_eq!(renderer.class_of(&properties(r#"{"pop":"many"}"#)), None);
        assert_eq!(renderer.class_of(&properties(r#"{"pop":true}"#)), None);
        assert_eq!(renderer.class_of(&properties(r#"{"pop":null}"#)), None);
        assert_eq!(renderer.class_of(&properties(r#"{"other":5}"#)), None);
        assert_eq!(
            renderer.class_of(&properties(r#"{"pop":"5"}"#)),
            None,
            "a numeric-looking string is NOT parsed",
        );
        assert_eq!(renderer.class_style(None), Some(&fill(9)));
        // A NaN can only arrive through a hand-built AttrRef, and is refused
        // there too — otherwise the open-above last class would swallow it.
        assert_eq!(AttrRef::Number(f64::NAN).as_number(), None);
        assert_eq!(AttrRef::Number(f64::INFINITY).as_number(), None);
    }

    #[test]
    fn a_graduated_renderer_with_no_classes_at_all_resolves_to_the_fallback() {
        let renderer = Renderer::graduated("pop", [], Some(fill(9)));
        assert_eq!(renderer.class_of(&properties(r#"{"pop":5}"#)), None);
        assert_eq!(renderer.class_count(), 0);
        assert_eq!(renderer.class_label(0), "");
        // And one single class covers everything, open at both ends.
        let one = Renderer::graduated("pop", [GraduatedClass::new(1.0, fill(1))], None);
        assert_eq!(one.class_of(&properties(r#"{"pop":-1e9}"#)), Some(0));
        assert_eq!(one.class_of(&properties(r#"{"pop":1e9}"#)), Some(0));
        assert_eq!(one.class_label(0), "all values");
    }

    #[test]
    fn classes_past_the_cap_are_inert_rather_than_truncated() {
        let categories: Vec<CategoryClass> = (0..MAX_STYLE_CLASSES + 3)
            .map(|index| {
                CategoryClass::new(
                    AttrValue::number(index as f64).unwrap_or(AttrValue::Bool(false)),
                    fill(u8::try_from(index % 256).unwrap_or(0)),
                )
            })
            .collect();
        let renderer = Renderer::categorized("code", categories, None);
        assert_eq!(renderer.stored_class_count(), MAX_STYLE_CLASSES + 3);
        assert_eq!(renderer.class_count(), MAX_STYLE_CLASSES);
        assert_eq!(renderer.overflow_class_count(), 3);
        assert_eq!(
            renderer.class_of(&properties(r#"{"code":0}"#)),
            Some(0),
            "under the cap, matched",
        );
        let over = format!(r#"{{"code":{}}}"#, MAX_STYLE_CLASSES);
        assert_eq!(
            renderer.class_of(&properties(&over)),
            None,
            "past the cap, the fallback — never a 65th mesh",
        );
        // The classification key is capped too, so a partition can never be
        // asked for more buckets than the renderer can resolve.
        assert_eq!(renderer.classification().class_count(), MAX_STYLE_CLASSES);
    }

    #[test]
    fn a_graduated_bound_is_sanitized_on_the_way_in_and_on_the_way_back() {
        let mut class = GraduatedClass::new(f64::NAN, fill(1));
        assert_eq!(class.upper(), 0.0);
        class.set_upper(f64::INFINITY);
        assert_eq!(class.upper(), f64::MAX);
        class.set_upper(f64::NEG_INFINITY);
        assert_eq!(class.upper(), f64::MIN);
        class.set_upper(12.5);
        assert_eq!(class.upper(), 12.5);
        // `1e400` is what a hand-edited file spells infinity as. serde_json
        // refuses it outright ("number out of range"), so the belt-and-braces
        // `deserialize_with` sanitizer can only ever fire for a format that is
        // laxer than JSON — pinned here so that a future project format
        // inherits a finite bound rather than a poisoned list.
        let line = r#"{"type":"line","color":"000000ff","width":1.0,"opacity":1.0}"#;
        assert!(
            serde_json::from_str::<GraduatedClass>(&format!(r#"{{"upper":1e400,"style":{line}}}"#))
                .is_err(),
            "JSON has no literal for infinity",
        );
        let loaded: GraduatedClass =
            serde_json::from_str(&format!(r#"{{"upper":12.5,"style":{line}}}"#)).expect("parse");
        assert_eq!(loaded.upper(), 12.5);
        assert!(serde_json::to_string(&loaded).is_ok());
        // And every bound a renderer can hold survives a serialize/parse pair.
        let renderer = Renderer::graduated(
            "pop",
            [
                GraduatedClass::new(f64::MIN, fill(1)),
                GraduatedClass::new(f64::MAX, fill(2)),
            ],
            None,
        );
        let json = serde_json::to_string(&renderer).expect("serialize");
        assert_eq!(
            serde_json::from_str::<Renderer>(&json).expect("parse"),
            renderer
        );
    }

    #[test]
    fn a_renderer_round_trips_through_its_internally_tagged_wire_shape() {
        let renderer = Renderer::categorized(
            "pref",
            [CategoryClass::new(AttrValue::text("Tokyo"), fill(1))],
            Some(LayerStyle::Line(LineStyle::new(Color::BLACK, 1.0))),
        );
        let json = serde_json::to_string(&renderer).expect("serialize");
        assert!(
            json.starts_with(r#"{"kind":"categorized","field":"pref""#),
            "{json}"
        );
        assert!(json.contains(r#""value":"Tokyo""#), "{json}");
        let back: Renderer = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, renderer);

        let graduated = Renderer::graduated("pop", [GraduatedClass::new(10.0, fill(2))], None);
        let json = serde_json::to_string(&graduated).expect("serialize");
        assert!(
            json.starts_with(r#"{"kind":"graduated","field":"pop""#),
            "{json}"
        );
        assert!(!json.contains("fallback"), "an absent fallback is skipped");
        assert_eq!(
            serde_json::from_str::<Renderer>(&json).expect("deserialize"),
            graduated
        );

        // An unknown kind is a malformed document, not a silent Single.
        assert!(serde_json::from_str::<Renderer>(r#"{"kind":"rule_based"}"#).is_err());
        // A categorized renderer with no class list at all still loads.
        let bare: Renderer =
            serde_json::from_str(r#"{"kind":"categorized","field":"x"}"#).expect("parse");
        assert_eq!(bare.class_count(), 0);
        assert_eq!(bare.field(), Some("x"));
    }

    #[test]
    fn the_classification_key_ignores_the_styles_and_notices_the_breaks() {
        let red = Renderer::categorized(
            "pref",
            [CategoryClass::new(AttrValue::text("Tokyo"), fill(1))],
            None,
        );
        let blue = Renderer::categorized(
            "pref",
            [CategoryClass::new(AttrValue::text("Tokyo"), fill(200))],
            Some(fill(9)),
        );
        assert_eq!(
            red.classification(),
            blue.classification(),
            "a colour edit must not re-partition a million features",
        );
        let moved = Renderer::categorized(
            "pref",
            [CategoryClass::new(AttrValue::text("Osaka"), fill(1))],
            None,
        );
        assert_ne!(red.classification(), moved.classification());
        let elsewhere = Renderer::categorized(
            "region",
            [CategoryClass::new(AttrValue::text("Tokyo"), fill(1))],
            None,
        );
        assert_ne!(red.classification(), elsewhere.classification());

        let breaks = Renderer::graduated("pop", [GraduatedClass::new(1.0, fill(1))], None);
        let wider = Renderer::graduated("pop", [GraduatedClass::new(2.0, fill(1))], None);
        assert_ne!(breaks.classification(), wider.classification());
        assert_eq!(breaks.classification().class_count(), 1);
        assert!(!breaks.classification().is_single());
    }

    #[test]
    fn a_classification_is_equal_to_itself_even_with_a_nan_in_it() {
        // The self-equality hazard: `Classification` is a CACHE KEY, so a
        // partition that compares unequal to itself would re-partition the
        // whole dataset on every frame of a colour drag. IEEE says
        // `NaN != NaN`; identity says a partition is the partition it is.
        //
        // A NaN cannot arrive through the constructors, but `CategorizedSpec`
        // has public fields, so the invariant is enforced rather than assumed.
        let poisoned = Renderer::Categorized(Box::new(CategorizedSpec {
            field: "code".to_string(),
            categories: vec![CategoryClass::new(AttrValue::Number(f64::NAN), fill(1))],
            fallback: None,
        }));
        let key = poisoned.classification();
        assert_eq!(key, key.clone(), "a partition is equal to itself");
        assert_eq!(key, poisoned.classification(), "and stable across calls");

        // Two DIFFERENT partitions still compare unequal.
        let other = Renderer::categorized(
            "code",
            [CategoryClass::new(
                AttrValue::number(1.0).unwrap_or(AttrValue::Bool(false)),
                fill(1),
            )],
            None,
        );
        assert_ne!(key, other.classification());

        // The same guard on the graduated side, where a bound is a bare f64.
        let mut nan_break = GraduatedClass::new(0.0, fill(1));
        // `set_upper` sanitizes, so the NaN is written through the field's own
        // deserializer-equivalent path: build it, then confirm the guard holds
        // for the finite bounds a real classification carries.
        nan_break.set_upper(f64::NAN);
        assert_eq!(nan_break.upper(), 0.0, "the model refuses a NaN bound");
        let graduated = Renderer::graduated("pop", [GraduatedClass::new(1.5, fill(1))], None);
        let key = graduated.classification();
        assert_eq!(key, key.clone());
        assert_eq!(key, graduated.classification());
        assert_ne!(
            key,
            Renderer::graduated("pop", [GraduatedClass::new(1.6, fill(1))], None).classification(),
        );
        // And `-0.0` and `0.0` are the same partition, since they classify
        // every feature identically.
        let plus = Renderer::graduated("v", [GraduatedClass::new(0.0, fill(1))], None);
        let minus = Renderer::graduated("v", [GraduatedClass::new(-0.0, fill(1))], None);
        assert_eq!(
            plus.class_of(&properties(r#"{"v":0}"#)),
            minus.class_of(&properties(r#"{"v":0}"#)),
            "they resolve identically ...",
        );
        assert_ne!(
            plus.classification(),
            minus.classification(),
            "... but differ in bits, which is the conservative answer: a \
             needless re-partition is a stutter, a missed one is a wrong map",
        );
    }

    #[test]
    fn editing_helpers_reach_every_slot_and_refuse_the_ones_that_do_not_exist() {
        let mut renderer = Renderer::categorized(
            "pref",
            [
                CategoryClass::new(AttrValue::text("Tokyo"), fill(1)),
                CategoryClass::new(AttrValue::text("Osaka"), fill(2)),
            ],
            None,
        );
        match renderer.class_style_mut(1) {
            Some(style) => *style = fill(7),
            None => panic!("class 1 exists"),
        }
        assert_eq!(renderer.class_style(Some(1)), Some(&fill(7)));
        assert!(renderer.class_style_mut(9).is_none());

        match renderer.fallback_mut() {
            Some(slot) => *slot = Some(fill(8)),
            None => panic!("a categorized renderer has a fallback slot"),
        }
        assert_eq!(renderer.fallback(), Some(&fill(8)));

        assert!(renderer.remove_class(0));
        assert_eq!(renderer.class_count(), 1);
        assert_eq!(renderer.class_label(0), "Osaka");
        assert!(!renderer.remove_class(5));

        renderer.set_field("region");
        assert_eq!(renderer.field(), Some("region"));

        let mut single = Renderer::Single;
        single.set_field("ignored");
        assert_eq!(single.field(), None);
        assert!(single.fallback_mut().is_none());
        assert!(!single.remove_class(0));
        assert!(single.class_style_mut(0).is_none());
        assert_eq!(single.class_label(0), "");
        assert!(single.categories().is_empty());
        assert!(single.graduated_classes().is_empty());
    }

    #[test]
    fn a_class_style_composes_over_a_family_rather_than_replacing_it() {
        use crate::style::{CircleStyle, LineStyle, SymbolStyle};

        let red = Color::from_rgb(0xd0, 0x20, 0x20);
        let class = LayerStyle::Fill(FillStyle::new(red));

        // Same kind: verbatim, everything the class says included.
        let mut other = FillStyle::new(Color::WHITE);
        other.set_opacity(0.5);
        assert_eq!(
            class_over_family(&class, &LayerStyle::Fill(other)),
            class,
            "the class wins outright when the kinds agree",
        );

        // Different kind: the family's SHAPE survives, in the class's colour.
        let mut circle = CircleStyle::new(4.0, Color::WHITE);
        circle.stroke_color = Some(Color::BLACK);
        match class_over_family(&class, &LayerStyle::Circle(circle)) {
            LayerStyle::Circle(out) => {
                assert_eq!(out.color, red);
                assert_eq!(out.radius(), 4.0);
                assert_eq!(out.stroke_color, Some(Color::BLACK));
            }
            other => panic!("a point family must stay a circle, got {other:?}"),
        }
        let mut line = LineStyle::new(Color::WHITE, 3.0);
        line.set_opacity(0.25);
        match class_over_family(&class, &LayerStyle::Line(line)) {
            LayerStyle::Line(out) => {
                assert_eq!(out.color, red);
                assert_eq!(out.width(), 3.0);
                assert_eq!(out.opacity(), 0.25);
            }
            other => panic!("a line family must stay a line, got {other:?}"),
        }
        let symbol = SymbolStyle::new("name");
        match class_over_family(&class, &LayerStyle::Symbol(symbol)) {
            LayerStyle::Symbol(out) => {
                assert_eq!(out.text_color, red);
                assert_eq!(out.text_field.as_deref(), Some("name"));
                assert_eq!(out.halo_width(), 1.0, "the halo is the family's");
            }
            other => panic!("a labelling family must stay a symbol, got {other:?}"),
        }

        // And the colour reader agrees with what it composed.
        assert_eq!(style_color(&class), red);
        assert_eq!(
            style_color(&recolor_style(&class, Color::BLACK)),
            Color::BLACK
        );
    }

    #[test]
    fn a_renderer_kind_names_itself_for_the_panel() {
        assert_eq!(RendererKind::default(), RendererKind::Single);
        assert_eq!(RendererKind::ALL.len(), 3);
        for kind in RendererKind::ALL {
            assert!(!kind.label().is_empty());
        }
        assert_eq!(RendererKind::Graduated.label(), "Graduated");
    }

    #[test]
    fn numbers_print_the_way_a_legend_reads_them() {
        assert_eq!(format_number(42.0), "42");
        assert_eq!(format_number(-7.0), "-7");
        assert_eq!(format_number(0.5), "0.5");
        assert_eq!(format_number(1e20), "100000000000000000000");
        assert_eq!(AttrValue::Bool(false).label(), "false");
        assert_eq!(AttrValue::text("x").label(), "x");
        assert_eq!(AttrValue::Number(3.0).label(), "3");
    }

    #[test]
    fn a_borrowed_value_converts_to_the_stored_one_and_back() {
        let text = AttrRef::Text(Cow::Borrowed("Tokyo"));
        assert_eq!(text.to_value(), Some(AttrValue::text("Tokyo")));
        assert_eq!(AttrValue::text("Tokyo").as_ref(), text);
        assert_eq!(AttrRef::Bool(true).to_value(), Some(AttrValue::Bool(true)));
        assert_eq!(AttrValue::Bool(true).as_ref(), AttrRef::Bool(true));
        assert_eq!(AttrRef::Number(1.5).to_value(), AttrValue::number(1.5));
        assert_eq!(AttrValue::Number(1.5).as_ref(), AttrRef::Number(1.5));
        assert_eq!(AttrRef::Number(f64::NAN).to_value(), None);
        assert_eq!(AttrRef::Number(2.0).label(), "2");
    }

    #[test]
    fn a_reference_to_an_attribute_source_is_itself_one() {
        // The blanket `&T` impl: a caller holding a borrow classifies without
        // reborrowing dance at every call site.
        let map = properties(r#"{"pref":"Tokyo"}"#);
        let borrowed = &map;
        let renderer = Renderer::categorized(
            "pref",
            [CategoryClass::new(AttrValue::text("Tokyo"), fill(1))],
            None,
        );
        assert_eq!(renderer.class_of(&borrowed), Some(0));
        assert_eq!(NoAttributes.value("anything"), None);
        assert_eq!(renderer.class_of(&NoAttributes), None);
    }
}
