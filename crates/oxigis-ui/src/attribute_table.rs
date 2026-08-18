//! The attribute table's *data model*: a [`FeatureCollection`] seen as rows and
//! columns, with no egui in sight.
//!
//! [`AttributeSchema`] derives the column set once per dataset and
//! [`FeatureRowSource`] turns feature `n`'s properties into
//! [`Cell`]s on demand. `table_panel` draws it; everything
//! here is testable without a UI.
//!
//! # Columns
//!
//! A GeoJSON `FeatureCollection` has no schema — every feature carries its own
//! property object and they need not agree. The column set is therefore the
//! *union* of every feature's property keys, in the order they are first
//! encountered while walking the collection in document order. (Within one
//! feature the keys arrive in `serde_json::Map`'s own order, which is
//! alphabetical unless its `preserve_order` feature is enabled somewhere in the
//! graph — it is not, today. Across features the union is genuinely
//! encounter-ordered, which is what makes a heterogeneous collection read
//! sensibly: the first feature's shape leads.)
//!
//! Two synthetic columns precede the property ones and are always present:
//!
//! * `#` — the feature's index in the source collection. This is not
//!   decoration: `local_vector::feature_collection_to_tile` uses exactly this
//!   index as the drawn feature's `MvtFeature::id`, so it is the handle a
//!   selected row will use to address a feature on the map.
//! * `geometry` — the geometry's type name, or empty for a null geometry.
//!
//! Property columns are capped at [`MAX_PROPERTY_COLUMNS`]; a dataset with more
//! keys than that reports the overflow through
//! [`AttributeSchema::omitted_columns`] rather than growing an unusable header.
//!
//! # Cell values
//!
//! JSON scalars map onto the matching [`Cell`] variant — which is what gives
//! the table numeric-aware sorting for free, since [`Cell::compare`] orders
//! `Int`/`Float` numerically and only falls back to text for text. Arrays and
//! objects are rendered as their compact JSON, as
//! [`crate::local_vector::convert_properties`] already does for labels. A
//! missing or `null` property is [`Cell::Empty`], not the string `"null"`: an
//! absent value and the text "null" must not sort or filter alike.
//!
//! # Sort keys
//!
//! `SortKey` mirrors [`Cell`]'s comparison semantics exactly but borrows
//! property text from the source JSON instead of allocating a `String` the
//! way [`Cell::Text`] must — sorting a text column would otherwise allocate
//! one `String` per row just to compare them once each.
//! `FeatureRowSource::cell_sort_key` is [`FeatureRowSource::cell_at`]'s
//! read-only twin; `table_panel::BoundLayer::sync_order` is the only caller.
//!
//! # Filtering and export
//!
//! `FeatureRowSource::row_contains` and [`FeatureRowSource::to_csv`] both
//! go through [`FeatureRowSource::cell_text`], the same text the panel draws,
//! so a filtered/exported row is never inconsistent with what is on screen.

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::Arc;

use oxigeo::geojson::types::{FeatureCollection, Properties};
use oxiui_table::{Cell, ColumnDef, RowSource};

/// Header of the leading row-number column.
pub const INDEX_COLUMN_NAME: &str = "#";

/// Header of the geometry-type column.
pub const GEOMETRY_COLUMN_NAME: &str = "geometry";

/// How many property columns are shown at most.
///
/// Beyond this the header stops being navigable long before it stops being
/// renderable, and every extra column costs a horizontal-scroll measurement per
/// frame. The overflow is reported, not silently dropped.
pub const MAX_PROPERTY_COLUMNS: usize = 64;

/// Number of synthetic columns (`#` and `geometry`) that precede the property
/// columns.
pub const SYNTHETIC_COLUMN_COUNT: usize = 2;

/// Default width, in logical pixels, of the `#` column.
const INDEX_COLUMN_WIDTH: f32 = 56.0;

/// Default width, in logical pixels, of the `geometry` column.
const GEOMETRY_COLUMN_WIDTH: f32 = 110.0;

/// Default width, in logical pixels, of a property column.
const PROPERTY_COLUMN_WIDTH: f32 = 130.0;

/// The column layout derived from one [`FeatureCollection`].
///
/// Derived once per dataset (deriving it walks every feature) and then held for
/// as long as the collection is displayed — see `table_panel`, which keys its
/// cache on the collection's [`Arc`] identity.
#[derive(Debug, Clone)]
pub struct AttributeSchema {
    /// Property keys, in first-encountered order, truncated to
    /// [`MAX_PROPERTY_COLUMNS`].
    property_keys: Vec<String>,
    /// How many further distinct keys the collection had beyond the cap.
    omitted_columns: usize,
    /// `#`, `geometry`, then one entry per element of `property_keys`.
    columns: Vec<ColumnDef>,
}

impl AttributeSchema {
    /// Derives the column layout of `features`.
    #[must_use]
    pub fn derive(features: &FeatureCollection) -> Self {
        let mut property_keys: Vec<String> = Vec::new();
        // Membership through a set, not a scan of `property_keys`: this loop
        // runs once per property of every feature, and a linear scan of up to
        // `MAX_PROPERTY_COLUMNS` strings inside it makes deriving the schema of
        // a 100k-feature dataset quadratic in the column count.
        let mut seen: HashSet<&str> = HashSet::new();
        for feature in &features.features {
            let Some(properties) = feature.properties.as_ref() else {
                continue;
            };
            for key in properties.keys() {
                if !seen.insert(key.as_str()) {
                    continue;
                }
                if property_keys.len() < MAX_PROPERTY_COLUMNS {
                    property_keys.push(key.clone());
                }
            }
        }
        // Distinct keys over the cap — not occurrences of them, which would
        // count a shared over-cap key once per feature that carries it.
        let omitted_columns = seen.len().saturating_sub(MAX_PROPERTY_COLUMNS);

        let mut columns = Vec::with_capacity(SYNTHETIC_COLUMN_COUNT + property_keys.len());
        columns.push(ColumnDef {
            name: INDEX_COLUMN_NAME.to_string(),
            width: INDEX_COLUMN_WIDTH,
            ..ColumnDef::default()
        });
        columns.push(ColumnDef {
            name: GEOMETRY_COLUMN_NAME.to_string(),
            width: GEOMETRY_COLUMN_WIDTH,
            ..ColumnDef::default()
        });
        for key in &property_keys {
            columns.push(ColumnDef {
                name: key.clone(),
                width: PROPERTY_COLUMN_WIDTH,
                ..ColumnDef::default()
            });
        }

        Self {
            property_keys,
            omitted_columns,
            columns,
        }
    }

    /// The property keys that became columns, in column order.
    #[must_use]
    pub fn property_keys(&self) -> &[String] {
        &self.property_keys
    }

    /// How many distinct property keys were dropped for exceeding
    /// [`MAX_PROPERTY_COLUMNS`]. Zero for every ordinary dataset.
    #[must_use]
    pub fn omitted_columns(&self) -> usize {
        self.omitted_columns
    }

    /// The full column list: `#`, `geometry`, then the property columns.
    #[must_use]
    pub fn columns(&self) -> &[ColumnDef] {
        &self.columns
    }
}

/// One [`FeatureCollection`] presented as table rows.
///
/// Rows are features in document order; the collection is shared, never copied
/// (`Arc`), and never mutated — the table is read-only, so [`RowSource`]'s
/// default `set_cell` (which reports [`oxiui_table::TableError::ReadOnly`]) is
/// exactly right and is deliberately not overridden.
#[derive(Debug, Clone)]
pub struct FeatureRowSource {
    /// The features being shown.
    features: Arc<FeatureCollection>,
    /// The column layout derived from them.
    schema: AttributeSchema,
}

impl FeatureRowSource {
    /// Builds a source over `features`, deriving its schema.
    #[must_use]
    pub fn new(features: Arc<FeatureCollection>) -> Self {
        let schema = AttributeSchema::derive(&features);
        Self { features, schema }
    }

    /// The shared collection behind this source.
    #[must_use]
    pub fn features(&self) -> &Arc<FeatureCollection> {
        &self.features
    }

    /// The derived column layout.
    #[must_use]
    pub fn schema(&self) -> &AttributeSchema {
        &self.schema
    }

    /// The value of one cell, without materialising the rest of its row.
    ///
    /// This is what keeps sorting and drawing off the `O(columns)` path
    /// [`RowSource::row`] necessarily takes: a sort touches one column of every
    /// row, and building a whole [`Vec<Cell>`] per row to read one of them is
    /// the difference between a responsive 100k-row table and an unusable one.
    ///
    /// Returns [`Cell::Empty`] for an out-of-range row or column, matching the
    /// treatment of an absent property.
    #[must_use]
    pub fn cell_at(&self, row: usize, column: usize) -> Cell {
        let Some(feature) = self.features.features.get(row) else {
            return Cell::Empty;
        };
        match column {
            0 => Cell::Int(i64::try_from(row).unwrap_or(i64::MAX)),
            1 => match feature.geometry.as_ref() {
                Some(geometry) => Cell::Text(geometry.geometry_type().as_str().to_string()),
                None => Cell::Empty,
            },
            _ => {
                let Some(key) = self
                    .schema
                    .property_keys
                    .get(column - SYNTHETIC_COLUMN_COUNT)
                else {
                    return Cell::Empty;
                };
                property_cell(feature.properties.as_ref(), key)
            }
        }
    }

    /// The display text of one cell — [`Cell`]'s own [`std::fmt::Display`], so
    /// the panel's rendering and any CSV/clipboard path agree by construction.
    #[must_use]
    pub fn cell_text(&self, row: usize, column: usize) -> String {
        self.cell_at(row, column).to_string()
    }

    /// [`Self::cell_at`]'s value as a [`SortKey`]: the same cell, but with
    /// property text borrowed from the source JSON instead of owned. See the
    /// module-level "Sort keys" section for why this exists as a separate
    /// method rather than a mode of `cell_at`.
    #[must_use]
    pub(crate) fn cell_sort_key(&self, row: usize, column: usize) -> SortKey<'_> {
        let Some(feature) = self.features.features.get(row) else {
            return SortKey::Empty;
        };
        match column {
            0 => SortKey::Int(i64::try_from(row).unwrap_or(i64::MAX)),
            1 => match feature.geometry.as_ref() {
                Some(geometry) => SortKey::Text(Cow::Borrowed(geometry.geometry_type().as_str())),
                None => SortKey::Empty,
            },
            _ => {
                let Some(key) = self
                    .schema
                    .property_keys
                    .get(column - SYNTHETIC_COLUMN_COUNT)
                else {
                    return SortKey::Empty;
                };
                property_sort_key(feature.properties.as_ref(), key)
            }
        }
    }

    /// Whether any cell of `row` contains `needle_lower` (already lower-cased
    /// by the caller once, not per cell) as a case-insensitive substring.
    /// Goes through [`Self::cell_text`], so a row that matches here is a row
    /// whose *drawn* text matches. Short-circuits on the first matching
    /// column.
    ///
    /// `O(columns)` in the worst case (no match) and allocates a `String` per
    /// cell it must examine — the same cost class as materialising a sort key
    /// for every row, which is likewise paid once per filter-text change and
    /// never per frame; see `table_panel::BoundLayer::sync_order`.
    #[must_use]
    pub(crate) fn row_contains(&self, row: usize, needle_lower: &str) -> bool {
        (0..self.schema.columns.len()).any(|column| {
            self.cell_text(row, column)
                .to_lowercase()
                .contains(needle_lower)
        })
    }

    /// Renders `rows` (typically a display order — sort and/or filter
    /// already applied) as RFC-4180-style CSV: a header from
    /// [`RowSource::column_defs`], then one line per row, cells rendered
    /// through [`Self::cell_text`] so a copied or exported row is always the
    /// row the panel is drawing. `\n` separates lines; a field containing the
    /// delimiter, a double quote, or a newline is quoted, with embedded
    /// quotes doubled.
    #[must_use]
    pub fn to_csv(&self, rows: &[usize]) -> String {
        let columns = self.column_defs();
        let mut out = String::new();
        for (index, column) in columns.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            push_csv_field(&mut out, &column.name);
        }
        out.push('\n');
        for &row in rows {
            for index in 0..columns.len() {
                if index > 0 {
                    out.push(',');
                }
                push_csv_field(&mut out, &self.cell_text(row, index));
            }
            out.push('\n');
        }
        out
    }

    /// [`Self::to_csv`], UTF-8 encoded — the seam a future file-save action
    /// can write to disk without this crate's UI layer performing any I/O of
    /// its own or knowing what a file is.
    #[must_use]
    pub fn to_csv_bytes(&self, rows: &[usize]) -> Vec<u8> {
        self.to_csv(rows).into_bytes()
    }
}

/// Appends `field` to `out` as one RFC-4180 CSV field: quoted, with embedded
/// quotes doubled, when it contains `,`, `"`, `\n` or `\r`; written straight
/// through otherwise. Shared by every field [`FeatureRowSource::to_csv`]
/// writes, header and data alike, so both escape identically.
fn push_csv_field(out: &mut String, field: &str) {
    let needs_quotes =
        field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r');
    if !needs_quotes {
        out.push_str(field);
        return;
    }
    out.push('"');
    for ch in field.chars() {
        if ch == '"' {
            out.push('"');
        }
        out.push(ch);
    }
    out.push('"');
}

/// Looks `key` up in `properties` and converts the JSON value to a [`Cell`].
///
/// Absent, null, or a value of a type with no scalar [`Cell`] counterpart
/// (arrays, objects) — the first two become [`Cell::Empty`], the last its
/// compact JSON text.
fn property_cell(properties: Option<&Properties>, key: &str) -> Cell {
    let Some(value) = properties.and_then(|properties| properties.get(key)) else {
        return Cell::Empty;
    };
    if value.is_null() {
        return Cell::Empty;
    }
    if let Some(text) = value.as_str() {
        return Cell::Text(text.to_string());
    }
    if let Some(flag) = value.as_bool() {
        return Cell::Bool(flag);
    }
    if let Some(number) = value.as_i64() {
        return Cell::Int(number);
    }
    if let Some(number) = value.as_f64() {
        return Cell::Float(number);
    }
    // Only arrays and objects reach here: `as_f64` already covers every JSON
    // number, `u64` above `i64::MAX` included, so this is their compact JSON,
    // not a numeric fallback.
    Cell::Text(value.to_string())
}

/// [`property_cell`]'s value as a [`SortKey`]: identical branching over the
/// same [`Properties`] lookup, so the two can never disagree about a cell's
/// type — only the text branch differs, borrowing instead of owning.
fn property_sort_key<'a>(properties: Option<&'a Properties>, key: &str) -> SortKey<'a> {
    let Some(value) = properties.and_then(|properties| properties.get(key)) else {
        return SortKey::Empty;
    };
    if value.is_null() {
        return SortKey::Empty;
    }
    if let Some(text) = value.as_str() {
        return SortKey::Text(Cow::Borrowed(text));
    }
    if let Some(flag) = value.as_bool() {
        return SortKey::Bool(flag);
    }
    if let Some(number) = value.as_i64() {
        return SortKey::Int(number);
    }
    if let Some(number) = value.as_f64() {
        return SortKey::Float(number);
    }
    SortKey::Text(Cow::Owned(value.to_string()))
}

/// A sort key mirroring [`Cell::compare`]'s semantics for exactly the
/// variants [`FeatureRowSource::cell_at`] ever produces (`Empty`/`Bool`/
/// `Int`/`Float`/`Text`) — never made to cover [`Cell`]'s other variants
/// (`Date`, `Currency`, `Link`, `Image`, `Custom`), since this row source
/// never emits them and a variant this enum cannot name is a variant its
/// `compare` cannot get wrong.
#[derive(Debug)]
pub(crate) enum SortKey<'a> {
    /// Absent or null — sorts first, matching [`Cell::Empty`].
    Empty,
    /// Matching [`Cell::Bool`]; `false` sorts before `true`.
    Bool(bool),
    /// Matching [`Cell::Int`].
    Int(i64),
    /// Matching [`Cell::Float`].
    Float(f64),
    /// Matching [`Cell::Text`] — borrowed from the source JSON when the
    /// value already is a string or (column 1) a geometry type name; owned
    /// only for the array/object fallback [`property_sort_key`] shares with
    /// [`property_cell`].
    Text(Cow<'a, str>),
}

impl SortKey<'_> {
    /// Stable rank for cross-variant comparison, identical to
    /// [`Cell`]'s private `type_rank` restricted to the variants this enum
    /// has — numeric types share a rank here exactly as they do there.
    fn type_rank(&self) -> u8 {
        match self {
            SortKey::Empty => 0,
            SortKey::Bool(_) => 1,
            SortKey::Int(_) | SortKey::Float(_) => 2,
            SortKey::Text(_) => 3,
        }
    }

    /// Total ordering reproducing [`Cell::compare`] exactly for the variants
    /// both types share: same-type cells compare naturally, mixed `Int`/
    /// `Float` promote to `f64` with a total order (`NaN` last), and any
    /// other cross-type pair falls back to [`Self::type_rank`]. Sorting row
    /// indices by this key therefore yields the identical order sorting by
    /// [`Cell::compare`] would, including tie order (both are used only with
    /// a stable sort).
    pub(crate) fn compare(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (SortKey::Int(a), SortKey::Int(b)) => a.cmp(b),
            (SortKey::Float(a), SortKey::Float(b)) => a.total_cmp(b),
            (SortKey::Int(a), SortKey::Float(b)) => (*a as f64).total_cmp(b),
            (SortKey::Float(a), SortKey::Int(b)) => a.total_cmp(&(*b as f64)),
            (SortKey::Text(a), SortKey::Text(b)) => a.cmp(b),
            (SortKey::Bool(a), SortKey::Bool(b)) => a.cmp(b),
            (SortKey::Empty, SortKey::Empty) => Ordering::Equal,
            _ => self.type_rank().cmp(&other.type_rank()),
        }
    }
}

impl RowSource for FeatureRowSource {
    fn row_count(&self) -> usize {
        self.features.features.len()
    }

    fn row(&self, index: usize) -> Vec<Cell> {
        (0..self.schema.columns.len())
            .map(|column| self.cell_at(index, column))
            .collect()
    }

    fn column_defs(&self) -> &[ColumnDef] {
        &self.schema.columns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigeo::geojson::reader::feature_collection_from_str;

    /// Two features with disjoint-but-overlapping property sets and different
    /// geometry types — the heterogeneous case the union rule exists for.
    const MIXED: &str = r#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [139.7, 35.7]},
             "properties": {"name": "Tokyo", "pop": 13960000, "area": 2194.07}},
            {"type": "Feature",
             "geometry": {"type": "LineString",
                          "coordinates": [[139.0, 35.0], [140.0, 36.0]]},
             "properties": {"name": "Line", "gauge": null, "tags": ["a", "b"],
                            "open": true}}
        ]
    }"#;

    fn mixed() -> Arc<FeatureCollection> {
        Arc::new(feature_collection_from_str(MIXED).expect("fixture parses"))
    }

    /// One property `"v"` covering every branch `property_cell` and
    /// `property_sort_key` share (text, int, float, bool, JSON null, an
    /// absent key, an array, an object, a `u64` above `i64::MAX`, and a
    /// negative int), plus a second property `"t"` that ties across three
    /// values — the fixture `cell_sort_key`'s cross-check tests replay both
    /// comparators over.
    const VARIED: &str = r#"{
        "type": "FeatureCollection",
        "features": [
            {"type":"Feature","geometry":null,"properties":{"v":"banana","t":0}},
            {"type":"Feature","geometry":null,"properties":{"v":42,"t":1}},
            {"type":"Feature","geometry":null,"properties":{"v":3.5,"t":2}},
            {"type":"Feature","geometry":null,"properties":{"v":true,"t":0}},
            {"type":"Feature","geometry":null,"properties":{"v":false,"t":1}},
            {"type":"Feature","geometry":null,"properties":{"v":null,"t":2}},
            {"type":"Feature","geometry":null,"properties":{"other":1,"t":0}},
            {"type":"Feature","geometry":null,"properties":{"v":[1,2,3],"t":1}},
            {"type":"Feature","geometry":null,"properties":{"v":{"a":1},"t":2}},
            {"type":"Feature","geometry":null,
             "properties":{"v":18446744073709551615,"t":0}},
            {"type":"Feature","geometry":null,"properties":{"v":"apple","t":1}},
            {"type":"Feature","geometry":null,"properties":{"v":-7,"t":2}}
        ]
    }"#;

    fn varied() -> FeatureRowSource {
        let collection = feature_collection_from_str(VARIED).expect("fixture parses");
        FeatureRowSource::new(Arc::new(collection))
    }

    /// A single plain-string property containing the CSV delimiter but no
    /// quotes — an unambiguous fixture for exercising quoting in `to_csv`
    /// without depending on `f64`'s exact `Display` text anywhere.
    const QUOTED: &str = r#"{
        "type": "FeatureCollection",
        "features": [
            {"type":"Feature","geometry":null,
             "properties":{"note":"hello, world"}}
        ]
    }"#;

    #[test]
    fn schema_unions_property_keys_and_prefixes_the_synthetic_columns() {
        let schema = AttributeSchema::derive(&mixed());
        let names: Vec<&str> = schema
            .columns()
            .iter()
            .map(|column| column.name.as_str())
            .collect();
        assert_eq!(names[0], INDEX_COLUMN_NAME);
        assert_eq!(names[1], GEOMETRY_COLUMN_NAME);
        // Every key of both features appears exactly once.
        for key in ["name", "pop", "area", "gauge", "tags", "open"] {
            assert_eq!(
                names.iter().filter(|name| **name == key).count(),
                1,
                "key {key} must contribute exactly one column"
            );
        }
        assert_eq!(schema.property_keys().len(), 6);
        assert_eq!(schema.omitted_columns(), 0);
    }

    #[test]
    fn first_features_keys_lead_the_union() {
        let schema = AttributeSchema::derive(&mixed());
        let keys = schema.property_keys();
        // "gauge"/"open"/"tags" are introduced by feature 1, so they follow
        // every key feature 0 introduced, whatever the within-feature order.
        let first_feature_keys = ["area", "name", "pop"];
        let last_of_first = first_feature_keys
            .iter()
            .filter_map(|key| keys.iter().position(|known| known == key))
            .max()
            .expect("feature 0's keys are columns");
        for later in ["gauge", "open", "tags"] {
            let position = keys
                .iter()
                .position(|known| known == later)
                .expect("feature 1's keys are columns");
            assert!(position > last_of_first, "{later} must follow feature 0");
        }
    }

    #[test]
    fn index_and_geometry_columns_hold_the_feature_index_and_type() {
        let source = FeatureRowSource::new(mixed());
        assert_eq!(source.cell_text(0, 0), "0");
        assert_eq!(source.cell_text(1, 0), "1");
        assert_eq!(source.cell_text(0, 1), "Point");
        assert_eq!(source.cell_text(1, 1), "LineString");
    }

    #[test]
    fn scalars_become_typed_cells_and_nested_values_become_compact_json() {
        let source = FeatureRowSource::new(mixed());
        let keys = source.schema().property_keys().to_vec();
        let column_of = |key: &str| {
            keys.iter()
                .position(|known| known == key)
                .map(|position| position + SYNTHETIC_COLUMN_COUNT)
                .expect("column exists")
        };
        assert!(matches!(
            source.cell_at(0, column_of("pop")),
            Cell::Int(13_960_000)
        ));
        assert!(matches!(
            source.cell_at(0, column_of("area")),
            Cell::Float(_)
        ));
        assert!(matches!(
            source.cell_at(1, column_of("open")),
            Cell::Bool(true)
        ));
        assert_eq!(source.cell_text(0, column_of("name")), "Tokyo");
        assert_eq!(source.cell_text(1, column_of("tags")), r#"["a","b"]"#);
    }

    #[test]
    fn absent_and_null_properties_are_empty_not_the_word_null() {
        let source = FeatureRowSource::new(mixed());
        let keys = source.schema().property_keys().to_vec();
        let column_of = |key: &str| {
            keys.iter()
                .position(|known| known == key)
                .map(|position| position + SYNTHETIC_COLUMN_COUNT)
                .expect("column exists")
        };
        // Feature 0 has no "open" key at all; feature 1's "gauge" is JSON null.
        assert!(source.cell_at(0, column_of("open")).is_empty());
        assert!(source.cell_at(1, column_of("gauge")).is_empty());
        assert_eq!(source.cell_text(0, column_of("open")), "");
    }

    #[test]
    fn out_of_range_row_or_column_is_empty_not_a_panic() {
        let source = FeatureRowSource::new(mixed());
        assert!(source.cell_at(999, 0).is_empty());
        assert!(source.cell_at(0, 999).is_empty());
    }

    #[test]
    fn row_materializes_every_column_in_order() {
        let source = FeatureRowSource::new(mixed());
        let row = source.row(0);
        assert_eq!(row.len(), source.column_defs().len());
        assert_eq!(row.len(), SYNTHETIC_COLUMN_COUNT + 6);
        assert_eq!(row[0].to_string(), "0");
        assert_eq!(row[1].to_string(), "Point");
    }

    #[test]
    fn property_columns_are_capped_and_the_overflow_is_reported() {
        let mut properties = String::new();
        for key in 0..MAX_PROPERTY_COLUMNS + 5 {
            if key > 0 {
                properties.push(',');
            }
            properties.push_str(&format!("\"k{key:03}\":{key}"));
        }
        let text = format!(
            r#"{{"type":"FeatureCollection","features":[{{"type":"Feature",
               "geometry":{{"type":"Point","coordinates":[0,0]}},
               "properties":{{{properties}}}}}]}}"#
        );
        let collection = feature_collection_from_str(&text).expect("fixture parses");
        let schema = AttributeSchema::derive(&collection);
        assert_eq!(schema.property_keys().len(), MAX_PROPERTY_COLUMNS);
        assert_eq!(schema.omitted_columns(), 5);
        assert_eq!(
            schema.columns().len(),
            SYNTHETIC_COLUMN_COUNT + MAX_PROPERTY_COLUMNS
        );
    }

    #[test]
    fn the_overflow_count_is_distinct_keys_not_occurrences_of_them() {
        // Two features carrying the *same* over-cap key. Counting occurrences
        // would report 2 columns not shown where there is only 1.
        let mut properties = String::new();
        for key in 0..MAX_PROPERTY_COLUMNS + 1 {
            if key > 0 {
                properties.push(',');
            }
            properties.push_str(&format!("\"k{key:03}\":{key}"));
        }
        let feature =
            format!(r#"{{"type":"Feature","geometry":null,"properties":{{{properties}}}}}"#);
        let text = format!(r#"{{"type":"FeatureCollection","features":[{feature},{feature}]}}"#);
        let collection = feature_collection_from_str(&text).expect("fixture parses");
        let schema = AttributeSchema::derive(&collection);
        assert_eq!(schema.property_keys().len(), MAX_PROPERTY_COLUMNS);
        assert_eq!(schema.omitted_columns(), 1);
    }

    #[test]
    fn a_feature_without_properties_contributes_no_columns_and_reads_empty() {
        let text = r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","geometry":null,"properties":null}]}"#;
        let collection = feature_collection_from_str(text).expect("fixture parses");
        let source = FeatureRowSource::new(Arc::new(collection));
        assert_eq!(source.schema().property_keys().len(), 0);
        assert_eq!(source.row_count(), 1);
        assert_eq!(source.cell_text(0, 0), "0");
        // Null geometry reads as empty rather than as a type name.
        assert!(source.cell_at(0, 1).is_empty());
    }

    #[test]
    fn source_shares_the_collection_rather_than_copying_it() {
        let features = mixed();
        let source = FeatureRowSource::new(Arc::clone(&features));
        assert!(Arc::ptr_eq(source.features(), &features));
    }

    #[test]
    fn cell_sort_key_reproduces_cell_compares_order_across_every_variant() {
        let source = varied();
        let v_column = source
            .schema()
            .property_keys()
            .iter()
            .position(|key| key == "v")
            .map(|position| position + SYNTHETIC_COLUMN_COUNT)
            .expect("v is a column");

        let mut by_cell: Vec<usize> = (0..source.row_count()).collect();
        by_cell.sort_by(|&a, &b| {
            source
                .cell_at(a, v_column)
                .compare(&source.cell_at(b, v_column))
        });

        let mut by_key: Vec<usize> = (0..source.row_count()).collect();
        by_key.sort_by(|&a, &b| {
            source
                .cell_sort_key(a, v_column)
                .compare(&source.cell_sort_key(b, v_column))
        });

        assert_eq!(
            by_cell, by_key,
            "SortKey must reorder the fixture identically to Cell::compare: \
             text, int, float, bool, null, an absent key, an array, an \
             object and a u64 above i64::MAX all appear in column \"v\""
        );
    }

    #[test]
    fn cell_sort_key_reproduces_cell_compares_stable_order_on_ties() {
        let source = varied();
        let t_column = source
            .schema()
            .property_keys()
            .iter()
            .position(|key| key == "t")
            .map(|position| position + SYNTHETIC_COLUMN_COUNT)
            .expect("t is a column");

        let mut by_cell: Vec<usize> = (0..source.row_count()).collect();
        by_cell.sort_by(|&a, &b| {
            source
                .cell_at(a, t_column)
                .compare(&source.cell_at(b, t_column))
        });

        let mut by_key: Vec<usize> = (0..source.row_count()).collect();
        by_key.sort_by(|&a, &b| {
            source
                .cell_sort_key(a, t_column)
                .compare(&source.cell_sort_key(b, t_column))
        });

        // "t" cycles over {0, 1, 2} across every row, so both sorts see
        // nothing but ties: this is the tie-breaking behaviour the previous
        // test's mostly-distinct "v" column does not exercise. A stable sort
        // over an identical total preorder must produce the identical
        // sequence, not just the identical grouping.
        assert_eq!(by_cell, by_key);
        assert_eq!(by_cell.len(), 12);
    }

    #[test]
    fn row_contains_matches_case_insensitively_through_cell_text() {
        let source = FeatureRowSource::new(mixed());
        // Upper-cased relative to the source property's "Tokyo".
        assert!(source.row_contains(0, "tokyo"));
        // The synthetic geometry column, not a property.
        assert!(source.row_contains(1, "linestring"));
        // A numeric property matched as its rendered text.
        assert!(source.row_contains(0, "13960000"));
        assert!(!source.row_contains(0, "no such substring"));
        assert!(!source.row_contains(1, "tokyo"));
    }

    #[test]
    fn row_contains_treats_an_empty_needle_as_matching_every_row() {
        let source = FeatureRowSource::new(mixed());
        assert!(source.row_contains(0, ""));
        assert!(source.row_contains(1, ""));
    }

    #[test]
    fn row_contains_is_out_of_range_safe() {
        let source = FeatureRowSource::new(mixed());
        assert!(!source.row_contains(999, "tokyo"));
    }

    #[test]
    fn push_csv_field_quotes_only_when_needed_and_doubles_embedded_quotes() {
        let mut plain = String::new();
        push_csv_field(&mut plain, "plain");
        assert_eq!(plain, "plain");

        let mut empty = String::new();
        push_csv_field(&mut empty, "");
        assert_eq!(empty, "");

        let mut comma = String::new();
        push_csv_field(&mut comma, "a,b");
        assert_eq!(comma, "\"a,b\"");

        let mut newline = String::new();
        push_csv_field(&mut newline, "line\nbreak");
        assert_eq!(newline, "\"line\nbreak\"");

        // A lone embedded quote: opening quote, the doubled embedded quote,
        // closing quote — four quote characters, nothing else.
        let mut one_quote = String::new();
        push_csv_field(&mut one_quote, "\"");
        assert_eq!(one_quote, "\"\"\"\"");

        // Two embedded quotes around plain text, built explicitly rather than
        // as one hand-counted literal: `say "hi"` is `say ` + `"` + `hi` +
        // `"`, and RFC-4180 doubles each embedded quote.
        let mut quoted = String::new();
        push_csv_field(&mut quoted, "say \"hi\"");
        let mut expected = String::from("\"say ");
        expected.push_str("\"\"");
        expected.push_str("hi");
        expected.push_str("\"\"");
        expected.push('"');
        assert_eq!(quoted, expected);
    }

    #[test]
    fn to_csv_of_no_rows_is_just_the_header() {
        let source = FeatureRowSource::new(mixed());
        // The property columns' relative order is `serde_json::Map`'s own
        // (alphabetical per feature, not JSON-text order — see the
        // module-level "Columns" section), so the expectation is derived
        // from `column_defs`, not hand-typed.
        let expected_header: String = source
            .column_defs()
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(source.to_csv(&[]), format!("{expected_header}\n"));
    }

    #[test]
    fn to_csv_uses_the_given_row_order_not_document_order() {
        let source = FeatureRowSource::new(mixed());
        let csv = source.to_csv(&[1, 0]);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3, "header plus exactly the 2 requested rows");

        // `#` and `geometry` are always columns 0 and 1 (pushed first by
        // `AttributeSchema::derive`, before any property key), unlike the
        // property columns that follow, whose relative order this test does
        // not assume.
        assert!(lines[1].starts_with("1,LineString,"));
        assert!(lines[2].starts_with("0,Point,"));
        // Feature 1 ("Line") first, feature 0 ("Tokyo") second: the caller's
        // order [1, 0], proving `to_csv` neither re-sorts nor re-selects its
        // input. Comma-bounded so "Line" cannot false-match inside
        // "LineString".
        assert!(lines[1].contains(",Line,"));
        assert!(lines[2].contains(",Tokyo,"));
    }

    #[test]
    fn to_csv_quotes_a_field_containing_the_delimiter() {
        let collection = feature_collection_from_str(QUOTED).expect("fixture parses");
        let source = FeatureRowSource::new(Arc::new(collection));
        let csv = source.to_csv(&[0]);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "#,geometry,note");
        assert_eq!(lines[1], "0,,\"hello, world\"");
    }

    #[test]
    fn to_csv_of_an_out_of_range_row_is_empty_fields_not_a_panic() {
        let source = FeatureRowSource::new(mixed());
        let csv = source.to_csv(&[999]);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1], ",,,,,,,");
    }

    #[test]
    fn to_csv_bytes_is_to_csv_as_utf8() {
        let source = FeatureRowSource::new(mixed());
        let rows = [0_usize, 1];
        assert_eq!(
            source.to_csv_bytes(&rows),
            source.to_csv(&rows).into_bytes()
        );
    }
}
