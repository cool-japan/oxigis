//! Turning a dataset's attributes into a [`Renderer`]: the scan helpers the
//! style panel's "Classify" button runs, and the MVT adapter that lets a
//! *drawn* feature resolve through the same rule its source feature did
//! (thematic v1.6).
//!
//! # Why the scans live here and not in the panel
//!
//! Nothing in this module draws. Reading unique values out of a field,
//! computing equal-interval or quantile breaks and building the class styles
//! are data operations with exact answers, so they are pure functions with
//! tests rather than closures inside an `egui` callback — the panel calls them
//! on a click and displays what comes back.
//!
//! # Cost, and its bound
//!
//! Every scan here is O(features) and runs on the UI thread, so each is bound
//! to run only on an explicit gesture (opening the field list, pressing
//! Classify) and never per frame. The collected data is bounded too:
//! [`unique_values`] stops at the class cap and reports the overflow, and
//! [`numeric_summary`] samples at most [`MAX_CLASSIFY_SAMPLE`] values — a
//! 10-million-feature dataset must not turn one button press into an 80 MB
//! allocation.

use std::collections::HashSet;

use oxigeo::geojson::types::FeatureCollection;
use oxigis_core::{
    AttrRef, AttrValue, Attributes, CategoryClass, CircleStyle, Color, FillStyle, GraduatedClass,
    LayerStyle, LineStyle, MAX_STYLE_CLASSES, Renderer, SymbolStyle,
};
use oxigis_render::MvtValue;

/// How many numeric values one [`numeric_summary`] keeps for the quantile
/// pass.
///
/// A quantile break needs the *distribution*, not just its extremes, so the
/// values have to be kept — but keeping every value of a ten-million-feature
/// dataset would be an 80 MB allocation for one button press. Past this cap
/// the scan switches to an even stride over the collection, which preserves
/// the shape of the distribution for any data that is not sorted by the very
/// field being classified, and says so through
/// [`NumericSummary::sampled`]. `min`/`max`/`count` are always exact — they
/// come from the full scan, not from the sample.
pub const MAX_CLASSIFY_SAMPLE: usize = 1 << 20;

/// The MVT property list of a *drawn* feature, as an attribute source.
///
/// The point of this adapter is agreement: the tile partition classifies a
/// GeoJSON feature through `serde_json`'s map, while anything working from the
/// drawn tile (hit testing, a future data-driven label) classifies the MVT
/// property list that came out of it. Both must land in the same class, so the
/// value mapping mirrors `local_vector::convert_properties` exactly — numbers
/// unify into `f64`, an array or object arrives as the same compact JSON text
/// on both sides — and a test pins the agreement on one feature.
#[derive(Debug, Clone, Copy)]
pub struct MvtAttributes<'a> {
    /// The feature's property list, in tile order.
    properties: &'a [(String, MvtValue)],
}

impl<'a> MvtAttributes<'a> {
    /// Reads `properties` as an attribute source.
    #[must_use]
    pub fn new(properties: &'a [(String, MvtValue)]) -> Self {
        Self { properties }
    }
}

impl Attributes for MvtAttributes<'_> {
    fn value(&self, key: &str) -> Option<AttrRef<'_>> {
        let (_, value) = self
            .properties
            .iter()
            .find(|(name, _)| name.as_str() == key)?;
        Some(match value {
            MvtValue::String(text) => AttrRef::Text(text.as_str().into()),
            MvtValue::F32(number) => AttrRef::Number(f64::from(*number)),
            MvtValue::F64(number) => AttrRef::Number(*number),
            // `as` on purpose, and matching `serde_json::Number::as_f64`: a
            // magnitude past 2^53 rounds, but it rounds the SAME way on both
            // sides of the comparison, so a feature either always matches its
            // category or never does.
            MvtValue::I64(number) => AttrRef::Number(*number as f64),
            MvtValue::U64(number) => AttrRef::Number(*number as f64),
            MvtValue::Bool(flag) => AttrRef::Bool(*flag),
        })
    }
}

/// The distinct values of one field, in first-encountered order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UniqueValues {
    /// The values that fit under the limit, first-encountered order — the same
    /// order the attribute table shows its columns in, so a classification
    /// reads like the data.
    pub values: Vec<AttrValue>,
    /// How many further distinct values the field had. Non-zero means the
    /// classification cannot cover the field and the rest draws with the
    /// fallback.
    pub overflow: usize,
    /// How many features carried no usable value at all (the field missing,
    /// `null`, or a non-finite number) — the count that will draw with the
    /// fallback even when `overflow` is zero.
    pub unclassified: usize,
}

/// The distinct values of `field` across `features`, capped at `limit`.
///
/// Dedup runs through hash sets rather than a scan of the growing list: the
/// inner loop runs once per feature, and a linear scan of up to
/// [`MAX_STYLE_CLASSES`] values inside it makes a 100k-feature dataset
/// quadratic in the class count (the same reasoning
/// [`crate::attribute_table::AttributeSchema::derive`] already applies to
/// column keys).
#[must_use]
pub fn unique_values(features: &FeatureCollection, field: &str, limit: usize) -> UniqueValues {
    let limit = limit.min(MAX_STYLE_CLASSES);
    let mut out = UniqueValues::default();
    let mut texts: HashSet<String> = HashSet::new();
    let mut numbers: HashSet<u64> = HashSet::new();
    let mut booleans = [false; 2];
    let mut distinct = 0_usize;
    for feature in &features.features {
        let Some(value) = feature
            .properties
            .as_ref()
            .and_then(|properties| properties.value(field))
        else {
            out.unclassified = out.unclassified.saturating_add(1);
            continue;
        };
        let fresh = match &value {
            AttrRef::Text(text) => texts.insert(text.clone().into_owned()),
            // Bit patterns, not values: `f64` is not `Hash`, and two NaNs are
            // never equal anyway (a non-finite value is refused below).
            AttrRef::Number(number) => {
                if !number.is_finite() {
                    out.unclassified = out.unclassified.saturating_add(1);
                    continue;
                }
                numbers.insert(number.to_bits())
            }
            AttrRef::Bool(flag) => {
                let slot = usize::from(*flag);
                let fresh = !booleans[slot];
                booleans[slot] = true;
                fresh
            }
        };
        if !fresh {
            continue;
        }
        distinct = distinct.saturating_add(1);
        if out.values.len() < limit
            && let Some(owned) = value.to_value()
        {
            out.values.push(owned);
        }
    }
    out.overflow = distinct.saturating_sub(out.values.len());
    out
}

/// What a numeric scan of one field found.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericSummary {
    /// How many features carried a finite number in the field.
    pub count: usize,
    /// The smallest value seen — exact, from the full scan.
    pub min: f64,
    /// The largest value seen — exact, from the full scan.
    pub max: f64,
    /// The values kept for the quantile pass, ascending. At most
    /// [`MAX_CLASSIFY_SAMPLE`] of them.
    pub values: Vec<f64>,
    /// Whether `values` is a stride sample rather than every value.
    pub sampled: bool,
}

/// Scans `field` for finite numbers, or [`None`] when it holds none at all
/// (the "this field cannot be graduated" answer the panel greys the button
/// out with).
#[must_use]
pub fn numeric_summary(features: &FeatureCollection, field: &str) -> Option<NumericSummary> {
    // The stride is chosen from the feature count up front, so the sample is
    // spread over the whole collection rather than being its first megabyte.
    let total = features.features.len();
    let stride = total.div_ceil(MAX_CLASSIFY_SAMPLE).max(1);
    let mut summary = NumericSummary {
        count: 0,
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
        values: Vec::new(),
        sampled: stride > 1,
    };
    for (index, feature) in features.features.iter().enumerate() {
        let Some(number) = feature
            .properties
            .as_ref()
            .and_then(|properties| properties.value(field))
            .and_then(|value| value.as_number())
        else {
            continue;
        };
        summary.count = summary.count.saturating_add(1);
        summary.min = summary.min.min(number);
        summary.max = summary.max.max(number);
        if index % stride == 0 && summary.values.len() < MAX_CLASSIFY_SAMPLE {
            summary.values.push(number);
        }
    }
    if summary.count == 0 {
        return None;
    }
    summary.values.sort_by(f64::total_cmp);
    Some(summary)
}

/// `count` equal-interval upper bounds over `min..=max`, ascending.
///
/// The LAST bound is set to `max` exactly rather than to `min + count·step`,
/// which floating-point arithmetic can leave a hair short of it. That is
/// belt-and-braces — a graduated renderer's last class is open above anyway
/// (see [`oxigis_core::Renderer::class_of`]) — but it keeps the legend honest:
/// the top row of a classification of populations should read "… – 13960000",
/// not "… – 13959999.999999998".
///
/// Empty for a `count` of zero or for non-finite input; a degenerate
/// `min == max` yields exactly one bound, which is one class covering
/// everything.
#[must_use]
pub fn equal_interval_breaks(min: f64, max: f64, count: usize) -> Vec<f64> {
    let count = count.min(MAX_STYLE_CLASSES);
    if count == 0 || !min.is_finite() || !max.is_finite() || max < min {
        return Vec::new();
    }
    if max == min {
        return vec![max];
    }
    let step = (max - min) / count as f64;
    let mut breaks = Vec::with_capacity(count);
    for index in 1..count {
        breaks.push(min + step * index as f64);
    }
    breaks.push(max);
    dedupe_ascending(breaks)
}

/// `count` quantile upper bounds over `values` (which must be ascending, as
/// [`numeric_summary`] leaves them), ascending.
///
/// Class `i` ends at the value at position `ceil((i+1)·n / count) - 1`, so
/// every class holds as close to `n / count` features as an integer split
/// allows. A heavily tied distribution can make two consecutive bounds equal —
/// the second class would then be empty — so equal bounds are collapsed, which
/// is why the result can be SHORTER than `count`. The panel reports the class
/// count it actually got rather than the one that was asked for.
#[must_use]
pub fn quantile_breaks(values: &[f64], count: usize) -> Vec<f64> {
    let count = count.min(MAX_STYLE_CLASSES);
    if count == 0 || values.is_empty() {
        return Vec::new();
    }
    let total = values.len();
    let mut breaks = Vec::with_capacity(count);
    for index in 1..=count {
        // `index * total` can only overflow for a `total` past 2^57 with the
        // class cap in force, which no in-memory collection reaches; the
        // checked form is here so the reasoning does not have to be repeated
        // by the next reader.
        let scaled = index.saturating_mul(total).div_ceil(count);
        let position = scaled.clamp(1, total) - 1;
        let Some(value) = values.get(position) else {
            break;
        };
        if value.is_finite() {
            breaks.push(*value);
        }
    }
    dedupe_ascending(breaks)
}

/// Drops non-ascending duplicates from an already-sorted bound list.
fn dedupe_ascending(mut breaks: Vec<f64>) -> Vec<f64> {
    breaks.sort_by(f64::total_cmp);
    breaks.dedup_by(|left, right| left == right);
    breaks
}

/// The low end of the default graduated ramp — a pale wash, so the lightest
/// class still reads as data rather than as background.
pub const RAMP_LOW: Color = Color {
    r: 0xed,
    g: 0xf4,
    b: 0xfb,
    a: 0xff,
};

/// The high end of the default graduated ramp.
pub const RAMP_HIGH: Color = Color {
    r: 0x08,
    g: 0x41,
    b: 0x81,
    a: 0xff,
};

/// A distinct colour for category `index`.
///
/// Hue-rotated by the golden ratio rather than taken from a fixed palette: a
/// table runs out (and then repeats adjacent colours), while this keeps
/// consecutive classes far apart in hue for any count up to the class cap, and
/// is deterministic — the same dataset classifies to the same colours in every
/// session, which is what makes a saved project's legend stable.
#[must_use]
pub fn category_color(index: usize) -> Color {
    const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_895;
    let hue = ((index as f64) * GOLDEN_RATIO_CONJUGATE).fract();
    // Alternating saturation/value on top of the hue rotation, so that even
    // neighbouring hues differ in a second axis.
    let saturation = if index.is_multiple_of(2) { 0.60 } else { 0.80 };
    let value = if index.is_multiple_of(3) { 0.94 } else { 0.76 };
    hsv_color(hue, saturation, value)
}

/// The ramp colour at `t` (clamped to `0..=1`), interpolated channel-wise.
#[must_use]
pub fn ramp_color(t: f64, low: Color, high: Color) -> Color {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mix = |from: u8, to: u8| -> u8 {
        let value = f64::from(from) + (f64::from(to) - f64::from(from)) * t;
        // `round` then clamp: the arithmetic cannot leave `0..=255` for two
        // `u8` endpoints and a clamped `t`, and the clamp makes that true by
        // construction rather than by argument.
        value.round().clamp(0.0, 255.0) as u8
    };
    Color::from_rgba(
        mix(low.r, high.r),
        mix(low.g, high.g),
        mix(low.b, high.b),
        mix(low.a, high.a),
    )
}

/// HSV → RGB, all inputs in `0..=1`, alpha fully opaque.
fn hsv_color(hue: f64, saturation: f64, value: f64) -> Color {
    let hue = if hue.is_finite() {
        hue.rem_euclid(1.0)
    } else {
        0.0
    };
    let saturation = saturation.clamp(0.0, 1.0);
    let value = value.clamp(0.0, 1.0);
    let sector = hue * 6.0;
    let index = sector.floor();
    let fraction = sector - index;
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * fraction);
    let t = value * (1.0 - saturation * (1.0 - fraction));
    let (red, green, blue) = match index as i32 % 6 {
        0 => (value, t, p),
        1 => (q, value, p),
        2 => (p, value, t),
        3 => (p, q, value),
        4 => (t, p, value),
        _ => (value, p, q),
    };
    let byte = |channel: f64| -> u8 { (channel * 255.0).round().clamp(0.0, 255.0) as u8 };
    Color::from_rgb(byte(red), byte(green), byte(blue))
}

/// `style` with its main colour replaced — the way a class style is built.
///
/// A class inherits everything else from the layer's own style (a fill's
/// opacity and outline, a line's width, a circle's radius and stroke, a
/// symbol's text field and halo), so classifying a layer changes exactly one
/// thing about how it draws: which colour each feature gets.
///
/// A re-export of the core rule rather than a second copy: the same function
/// is what [`oxigis_core::class_over_family`] applies when a class of one kind
/// lands on a family of another, so the panel's preview and the renderer's
/// resolution cannot drift.
pub use oxigis_core::recolor_style as recolored;

/// The colour a style is currently drawn in — what a class row's colour button
/// binds to, and what the panel seeds a hand-added class from.
pub use oxigis_core::style_color as color_of;

/// A categorized renderer over `field`, one class per value, each drawn as
/// `base` recoloured.
#[must_use]
pub fn categorized_renderer(
    base: &LayerStyle,
    field: &str,
    values: impl IntoIterator<Item = AttrValue>,
) -> Renderer {
    let categories = values
        .into_iter()
        .take(MAX_STYLE_CLASSES)
        .enumerate()
        .map(|(index, value)| CategoryClass::new(value, recolored(base, category_color(index))));
    Renderer::categorized(field, categories, Some(base.clone()))
}

/// A graduated renderer over `field`, one class per upper bound, each drawn as
/// `base` recoloured along the [`RAMP_LOW`] → [`RAMP_HIGH`] ramp.
#[must_use]
pub fn graduated_renderer(base: &LayerStyle, field: &str, breaks: &[f64]) -> Renderer {
    let count = breaks.len().min(MAX_STYLE_CLASSES);
    let last = count.saturating_sub(1);
    let classes = breaks.iter().take(count).enumerate().map(|(index, upper)| {
        let t = if last == 0 {
            1.0
        } else {
            index as f64 / last as f64
        };
        GraduatedClass::new(*upper, recolored(base, ramp_color(t, RAMP_LOW, RAMP_HIGH)))
    });
    Renderer::graduated(field, classes, Some(base.clone()))
}

/// A style of the same kind as `base`, in `color` — what the panel adds when
/// the user asks for one more class by hand.
#[must_use]
pub fn class_style_like(base: &LayerStyle, color: Color) -> LayerStyle {
    match base {
        LayerStyle::Fill(_) => recolored(&LayerStyle::Fill(FillStyle::new(color)), color),
        LayerStyle::Line(line) => LayerStyle::Line(LineStyle::new(color, line.width())),
        LayerStyle::Circle(circle) => LayerStyle::Circle(CircleStyle::new(circle.radius(), color)),
        LayerStyle::Symbol(symbol) => {
            let mut symbol = symbol.clone();
            symbol.text_color = color;
            LayerStyle::Symbol(symbol)
        }
    }
}

/// A blank symbol style, so `class_style_like` has something to fall back on
/// in a caller that has no base at all.
#[must_use]
pub fn blank_symbol_style() -> LayerStyle {
    LayerStyle::Symbol(SymbolStyle::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigis_core::{GeometryFamily, LayerStyleSet, NoAttributes};

    fn parse(text: &str) -> FeatureCollection {
        match oxigeo::geojson::reader::feature_collection_from_str(text) {
            Ok(features) => features,
            Err(error) => panic!("the fixture must parse: {error}"),
        }
    }

    /// Four points, one per prefecture, with a population and a mixed bag of
    /// attribute kinds.
    fn prefectures() -> FeatureCollection {
        parse(
            r#"{"type":"FeatureCollection","features":[
              {"type":"Feature","properties":{"pref":"Tokyo","pop":13960,"capital":true},
               "geometry":{"type":"Point","coordinates":[139.7,35.7]}},
              {"type":"Feature","properties":{"pref":"Osaka","pop":8839,"capital":false},
               "geometry":{"type":"Point","coordinates":[135.5,34.7]}},
              {"type":"Feature","properties":{"pref":"Kyoto","pop":2583,"capital":false},
               "geometry":{"type":"Point","coordinates":[135.8,35.0]}},
              {"type":"Feature","properties":{"pref":"Tokyo","pop":null},
               "geometry":{"type":"Point","coordinates":[139.8,35.6]}}]}"#,
        )
    }

    fn fill(color: Color) -> LayerStyle {
        let mut fill = FillStyle::new(color);
        fill.set_opacity(0.35);
        fill.outline_color = Some(Color::BLACK);
        LayerStyle::Fill(fill)
    }

    #[test]
    fn unique_values_are_first_encountered_and_deduplicated() {
        let unique = unique_values(&prefectures(), "pref", MAX_STYLE_CLASSES);
        assert_eq!(
            unique.values,
            vec![
                AttrValue::text("Tokyo"),
                AttrValue::text("Osaka"),
                AttrValue::text("Kyoto"),
            ],
            "document order, one entry per distinct value",
        );
        assert_eq!(unique.overflow, 0);
        assert_eq!(unique.unclassified, 0);
    }

    #[test]
    fn unique_values_count_what_it_could_not_classify() {
        let unique = unique_values(&prefectures(), "pop", MAX_STYLE_CLASSES);
        assert_eq!(unique.values.len(), 3, "the null one is not a value");
        assert_eq!(unique.unclassified, 1, "and is reported as unclassified");
        // A field no feature carries: every feature is unclassified.
        let missing = unique_values(&prefectures(), "nope", MAX_STYLE_CLASSES);
        assert!(missing.values.is_empty());
        assert_eq!(missing.unclassified, 4);
        assert_eq!(missing.overflow, 0);
    }

    #[test]
    fn unique_values_stop_at_the_limit_and_report_the_rest() {
        let unique = unique_values(&prefectures(), "pref", 2);
        assert_eq!(
            unique.values,
            vec![AttrValue::text("Tokyo"), AttrValue::text("Osaka")]
        );
        assert_eq!(unique.overflow, 1, "Kyoto did not fit");
        // The cap is the class cap even when a caller asks for more.
        let capped = unique_values(&prefectures(), "pref", usize::MAX);
        assert_eq!(capped.values.len(), 3);
    }

    #[test]
    fn unique_values_keep_kinds_apart() {
        let mixed = parse(
            r#"{"type":"FeatureCollection","features":[
              {"type":"Feature","properties":{"v":"1"},"geometry":{"type":"Point","coordinates":[0,0]}},
              {"type":"Feature","properties":{"v":1},"geometry":{"type":"Point","coordinates":[1,0]}},
              {"type":"Feature","properties":{"v":true},"geometry":{"type":"Point","coordinates":[2,0]}},
              {"type":"Feature","properties":{"v":true},"geometry":{"type":"Point","coordinates":[3,0]}},
              {"type":"Feature","properties":{"v":1.0},"geometry":{"type":"Point","coordinates":[4,0]}}]}"#,
        );
        let unique = unique_values(&mixed, "v", MAX_STYLE_CLASSES);
        assert_eq!(
            unique.values,
            vec![
                AttrValue::text("1"),
                AttrValue::Number(1.0),
                AttrValue::Bool(true),
            ],
            "a string 1, a number 1 and a boolean are three classes; 1 and 1.0 are one",
        );
    }

    #[test]
    fn a_numeric_summary_reports_the_exact_extremes_and_a_sorted_sample() {
        let summary = numeric_summary(&prefectures(), "pop").expect("a numeric field");
        assert_eq!(summary.count, 3, "the null is not a number");
        assert_eq!(summary.min, 2583.0);
        assert_eq!(summary.max, 13960.0);
        assert_eq!(summary.values, vec![2583.0, 8839.0, 13960.0], "ascending");
        assert!(!summary.sampled);
        // A text field has no numbers at all, and says so rather than
        // pretending to a range.
        assert!(numeric_summary(&prefectures(), "pref").is_none());
        assert!(numeric_summary(&prefectures(), "capital").is_none());
        assert!(numeric_summary(&prefectures(), "missing").is_none());
    }

    #[test]
    fn equal_intervals_split_the_range_and_end_exactly_on_the_maximum() {
        let breaks = equal_interval_breaks(0.0, 10.0, 5);
        assert_eq!(breaks, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        // The rounding hazard: `min + count*step` must not land below `max`.
        let breaks = equal_interval_breaks(0.1, 0.7, 3);
        assert_eq!(
            breaks.last().copied(),
            Some(0.7),
            "the top bound is the maximum itself, not the arithmetic's guess",
        );
        assert_eq!(breaks.len(), 3);
        // Degenerate inputs are answered, never panicked on.
        assert!(equal_interval_breaks(0.0, 10.0, 0).is_empty());
        assert!(equal_interval_breaks(f64::NAN, 10.0, 3).is_empty());
        assert!(equal_interval_breaks(10.0, 0.0, 3).is_empty(), "max < min");
        assert_eq!(equal_interval_breaks(5.0, 5.0, 4), vec![5.0], "one class");
        assert!(equal_interval_breaks(0.0, 1.0, usize::MAX).len() <= MAX_STYLE_CLASSES);
    }

    #[test]
    fn quantiles_put_the_same_number_of_features_in_every_class() {
        let values: Vec<f64> = (1..=100).map(f64::from).collect();
        let breaks = quantile_breaks(&values, 4);
        assert_eq!(breaks, vec![25.0, 50.0, 75.0, 100.0]);
        // Every class really does hold a quarter of the data.
        let renderer = graduated_renderer(&fill(Color::WHITE), "v", &breaks);
        let mut counts = [0_usize; 4];
        for value in &values {
            let map: serde_json::Map<String, serde_json::Value> =
                match serde_json::from_str(&format!(r#"{{"v":{value}}}"#)) {
                    Ok(map) => map,
                    Err(error) => panic!("the probe must parse: {error}"),
                };
            let class = renderer.class_of(&map).unwrap_or(usize::MAX);
            match counts.get_mut(class) {
                Some(slot) => *slot += 1,
                None => panic!("every value must land in a class, got {class}"),
            }
        }
        assert_eq!(counts, [25, 25, 25, 25]);
    }

    #[test]
    fn tied_quantiles_collapse_rather_than_leaving_an_empty_class() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 9.0];
        let breaks = quantile_breaks(&values, 4);
        assert_eq!(breaks, vec![1.0, 9.0], "two distinct bounds, not four");
        assert!(quantile_breaks(&[], 4).is_empty());
        assert!(quantile_breaks(&values, 0).is_empty());
        // A single class is legal and covers everything.
        assert_eq!(quantile_breaks(&values, 1), vec![9.0]);
    }

    #[test]
    fn a_classified_layer_covers_every_feature_it_scanned() {
        // The end-to-end property the two helpers exist for: classify a
        // dataset, then resolve every one of its features and land in a class
        // rather than in the fallback.
        let features = prefectures();
        let base = fill(Color::from_rgb(80, 140, 200));
        let unique = unique_values(&features, "pref", MAX_STYLE_CLASSES);
        let renderer = categorized_renderer(&base, "pref", unique.values.clone());
        assert_eq!(renderer.class_count(), 3);
        for feature in &features.features {
            let Some(properties) = feature.properties.as_ref() else {
                continue;
            };
            assert!(
                renderer.class_of(properties).is_some(),
                "every scanned value classifies: {properties:?}",
            );
        }
        // And the graduated twin, including the maximum itself.
        let summary = numeric_summary(&features, "pop").expect("numeric");
        let breaks = equal_interval_breaks(summary.min, summary.max, 3);
        let graduated = graduated_renderer(&base, "pop", &breaks);
        for value in &summary.values {
            let map: serde_json::Map<String, serde_json::Value> =
                match serde_json::from_str(&format!(r#"{{"pop":{value}}}"#)) {
                    Ok(map) => map,
                    Err(error) => panic!("the probe must parse: {error}"),
                };
            assert!(graduated.class_of(&map).is_some(), "{value} classifies");
        }
    }

    #[test]
    fn class_styles_inherit_everything_but_the_colour() {
        let base = fill(Color::from_rgb(1, 2, 3));
        let renderer = categorized_renderer(&base, "pref", [AttrValue::text("Tokyo")]);
        let LayerStyle::Fill(class) = renderer
            .class_style(Some(0))
            .cloned()
            .unwrap_or_else(blank_symbol_style)
        else {
            panic!("a fill class");
        };
        assert_eq!(class.opacity(), 0.35, "the base's opacity, kept");
        assert_eq!(class.outline_color, Some(Color::BLACK), "and its outline");
        assert_ne!(class.color, Color::from_rgb(1, 2, 3), "a new colour");
        // The fallback is the base itself, verbatim.
        assert_eq!(renderer.fallback(), Some(&base));
    }

    #[test]
    fn recoloring_touches_exactly_the_main_colour_of_every_style_kind() {
        let color = Color::from_rgb(9, 8, 7);
        let mut line = LineStyle::new(Color::BLACK, 3.5);
        line.set_opacity(0.5);
        match recolored(&LayerStyle::Line(line), color) {
            LayerStyle::Line(out) => {
                assert_eq!(out.color, color);
                assert_eq!(out.width(), 3.5);
                assert_eq!(out.opacity(), 0.5);
            }
            other => panic!("expected a line, got {other:?}"),
        }
        let mut circle = CircleStyle::new(6.0, Color::BLACK);
        circle.stroke_color = Some(Color::WHITE);
        match recolored(&LayerStyle::Circle(circle), color) {
            LayerStyle::Circle(out) => {
                assert_eq!(out.color, color);
                assert_eq!(out.radius(), 6.0);
                assert_eq!(out.stroke_color, Some(Color::WHITE));
            }
            other => panic!("expected a circle, got {other:?}"),
        }
        let symbol = SymbolStyle::new("name");
        match recolored(&LayerStyle::Symbol(symbol), color) {
            LayerStyle::Symbol(out) => {
                assert_eq!(out.text_color, color);
                assert_eq!(out.text_field.as_deref(), Some("name"));
            }
            other => panic!("expected a symbol, got {other:?}"),
        }
        assert_eq!(color_of(&recolored(&fill(Color::BLACK), color)), color);
    }

    #[test]
    fn category_colours_are_distinct_deterministic_and_opaque() {
        let colors: Vec<Color> = (0..MAX_STYLE_CLASSES).map(category_color).collect();
        for (index, color) in colors.iter().enumerate() {
            assert_eq!(color.a, 255, "class {index} must be opaque");
            assert_eq!(*color, category_color(index), "deterministic");
        }
        let distinct: HashSet<String> = colors.iter().map(|color| color.to_hex()).collect();
        assert_eq!(
            distinct.len(),
            colors.len(),
            "no two of the {MAX_STYLE_CLASSES} classes share a colour",
        );
        // Neighbours are far apart in hue, which is the point of the rotation.
        let far = colors.windows(2).all(|pair| match pair {
            [left, right] => {
                let delta = i32::from(left.r) - i32::from(right.r);
                let other = i32::from(left.b) - i32::from(right.b);
                delta.abs() + other.abs() > 20
            }
            _ => true,
        });
        assert!(far, "consecutive classes must not look alike");
    }

    #[test]
    fn the_ramp_runs_from_the_low_colour_to_the_high_one() {
        assert_eq!(ramp_color(0.0, RAMP_LOW, RAMP_HIGH), RAMP_LOW);
        assert_eq!(ramp_color(1.0, RAMP_LOW, RAMP_HIGH), RAMP_HIGH);
        let mid = ramp_color(0.5, RAMP_LOW, RAMP_HIGH);
        assert!(mid.r < RAMP_LOW.r && mid.r > RAMP_HIGH.r, "{mid:?}");
        // Out-of-range and non-finite input is answered, never panicked on.
        assert_eq!(ramp_color(-1.0, RAMP_LOW, RAMP_HIGH), RAMP_LOW);
        assert_eq!(ramp_color(9.0, RAMP_LOW, RAMP_HIGH), RAMP_HIGH);
        assert_eq!(ramp_color(f64::NAN, RAMP_LOW, RAMP_HIGH), RAMP_LOW);
        // A single-class ramp takes the high end rather than dividing by zero.
        let single = graduated_renderer(&fill(Color::WHITE), "v", &[1.0]);
        let LayerStyle::Fill(only) = single
            .class_style(Some(0))
            .cloned()
            .unwrap_or_else(blank_symbol_style)
        else {
            panic!("a fill class");
        };
        assert_eq!(only.color, RAMP_HIGH);
    }

    #[test]
    fn an_mvt_property_list_classifies_exactly_as_its_geojson_source_does() {
        // THE agreement the adapter exists for: the tile partition classifies
        // the GeoJSON map, anything working from the drawn tile classifies the
        // MVT list, and the two must land in the same class.
        let source = parse(
            r#"{"type":"FeatureCollection","features":[
              {"type":"Feature","properties":{"pref":"Tokyo","pop":13960,"ratio":0.5,
                "capital":true,"tags":["a","b"]},
               "geometry":{"type":"Point","coordinates":[139.7,35.7]}}]}"#,
        );
        let Some(feature) = source.features.first() else {
            panic!("one feature");
        };
        let Some(properties) = feature.properties.as_ref() else {
            panic!("with properties");
        };
        let converted = crate::local_vector::convert_properties(Some(properties));
        let mvt = MvtAttributes::new(&converted);

        for field in ["pref", "pop", "ratio", "capital", "tags", "missing"] {
            assert_eq!(
                properties.value(field).map(|value| value.label()),
                mvt.value(field).map(|value| value.label()),
                "the two paths must read `{field}` identically",
            );
        }

        let base = fill(Color::WHITE);
        let categorized = categorized_renderer(&base, "pref", [AttrValue::text("Tokyo")]);
        assert_eq!(categorized.class_of(properties), Some(0));
        assert_eq!(categorized.class_of(&mvt), Some(0), "and so does the tile");

        let graduated = graduated_renderer(&base, "pop", &[10000.0, 20000.0]);
        assert_eq!(graduated.class_of(properties), Some(1));
        assert_eq!(graduated.class_of(&mvt), Some(1));

        // A whole style set resolves the same way through either source.
        let mut set = LayerStyleSet::new(base);
        set.set_renderer(categorized);
        assert_eq!(
            set.style_for(GeometryFamily::Point, properties),
            set.style_for(GeometryFamily::Point, &mvt),
        );
        assert_eq!(
            &set.style_for(GeometryFamily::Point, &NoAttributes),
            set.base(),
            "and a feature with no attributes at all still draws",
        );
    }

    #[test]
    fn the_mvt_adapter_maps_every_value_kind() {
        let properties = vec![
            ("s".to_string(), MvtValue::String("x".to_string())),
            ("f32".to_string(), MvtValue::F32(1.5)),
            ("f64".to_string(), MvtValue::F64(2.5)),
            ("i".to_string(), MvtValue::I64(-3)),
            ("u".to_string(), MvtValue::U64(4)),
            ("b".to_string(), MvtValue::Bool(false)),
        ];
        let mvt = MvtAttributes::new(&properties);
        assert_eq!(
            mvt.value("s").and_then(|v| v.to_value()),
            Some(AttrValue::text("x"))
        );
        assert_eq!(mvt.value("f32").and_then(|v| v.as_number()), Some(1.5));
        assert_eq!(mvt.value("f64").and_then(|v| v.as_number()), Some(2.5));
        assert_eq!(mvt.value("i").and_then(|v| v.as_number()), Some(-3.0));
        assert_eq!(mvt.value("u").and_then(|v| v.as_number()), Some(4.0));
        assert_eq!(
            mvt.value("b").and_then(|v| v.to_value()),
            Some(AttrValue::Bool(false))
        );
        assert!(mvt.value("nope").is_none());
        assert!(MvtAttributes::new(&[]).value("s").is_none());
    }

    #[test]
    fn a_hand_added_class_takes_the_kind_of_the_layers_own_style() {
        let color = Color::from_rgb(7, 7, 7);
        assert!(matches!(
            class_style_like(&fill(Color::BLACK), color),
            LayerStyle::Fill(_)
        ));
        assert!(matches!(
            class_style_like(&LayerStyle::Line(LineStyle::new(Color::BLACK, 4.0)), color),
            LayerStyle::Line(_)
        ));
        match class_style_like(
            &LayerStyle::Circle(CircleStyle::new(9.0, Color::BLACK)),
            color,
        ) {
            LayerStyle::Circle(circle) => {
                assert_eq!(circle.radius(), 9.0, "the radius is inherited");
                assert_eq!(circle.color, color);
            }
            other => panic!("expected a circle, got {other:?}"),
        }
        assert!(matches!(blank_symbol_style(), LayerStyle::Symbol(_)));
    }
}
