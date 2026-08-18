//! Hand-written `prost` messages mirroring the official `vector_tile.proto`.
//!
//! # Why hand-written
//!
//! The upstream schema is a 60-line proto2 file that has not changed since MVT
//! 2.1 was published. Generating it would mean a `build.rs` invoking `protoc`
//! (or `prost-build`'s vendored one), i.e. an external C++ binary in the build
//! graph — which the project's Pure Rust policy rules out, and which would also
//! break the crate's `wasm32-unknown-unknown` story. `#[derive(prost::Message)]`
//! produces exactly the same wire behaviour from attributes alone.
//!
//! # Fidelity notes
//!
//! * proto2 `optional` scalars become [`Option`], so "absent" stays
//!   distinguishable from "present and zero". That matters for
//!   [`Layer::extent`] (absent means 4096, present-and-zero is invalid) and for
//!   [`Value`], where the decoder has to count how many fields are set.
//! * proto2 `required uint32 version = 15` is modelled as a plain `u32`
//!   defaulting to `0`, an impossible version. `prost` has no notion of a
//!   required field, so the check is deferred to
//!   [`super::decode::decode_mvt`], which rejects anything but `1` or `2`.
//! * `Feature::type` stays a raw `i32` rather than a `prost` enumeration field:
//!   the decoder must reject `UNKNOWN` and unassigned values explicitly instead
//!   of having them silently collapse to a default.
//! * The `extensions 16 to …` ranges and the `extent`/`version` proto defaults
//!   have no runtime representation; unknown fields are skipped by `prost`, so
//!   extension-bearing tiles decode rather than fail.
//!
//! Kept `pub(crate)`: exposing `prost`-derived types would pin this crate's
//! public API to a `prost` major version.

/// Geometry kinds a feature can carry (`vector_tile.Tile.GeomType`).
///
/// Only used for its discriminants; the wire field is decoded as an `i32` and
/// converted through [`GeomType::from_i32`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GeomType {
    /// `POINT` — one or more points.
    Point = 1,
    /// `LINESTRING` — one or more line strings.
    LineString = 2,
    /// `POLYGON` — one or more polygons, each an exterior ring plus holes.
    Polygon = 3,
}

impl GeomType {
    /// Converts a wire value, mapping `UNKNOWN` (`0`) and unassigned numbers to
    /// `None` so callers must handle them.
    pub(crate) const fn from_i32(value: i32) -> Option<Self> {
        match value {
            1 => Some(Self::Point),
            2 => Some(Self::LineString),
            3 => Some(Self::Polygon),
            _ => None,
        }
    }
}

/// A tagged property value: exactly one of the fields must be set.
#[derive(Clone, PartialEq, prost::Message)]
pub(crate) struct Value {
    /// UTF-8 string value.
    #[prost(string, optional, tag = "1")]
    pub(crate) string_value: Option<String>,
    /// 32-bit float value.
    #[prost(float, optional, tag = "2")]
    pub(crate) float_value: Option<f32>,
    /// 64-bit float value.
    #[prost(double, optional, tag = "3")]
    pub(crate) double_value: Option<f64>,
    /// Signed 64-bit value, varint-encoded (not zigzag).
    #[prost(int64, optional, tag = "4")]
    pub(crate) int_value: Option<i64>,
    /// Unsigned 64-bit value.
    #[prost(uint64, optional, tag = "5")]
    pub(crate) uint_value: Option<u64>,
    /// Signed 64-bit value, zigzag-encoded.
    #[prost(sint64, optional, tag = "6")]
    pub(crate) sint_value: Option<i64>,
    /// Boolean value.
    #[prost(bool, optional, tag = "7")]
    pub(crate) bool_value: Option<bool>,
}

/// One feature: an optional identifier, a tag list and a geometry stream.
#[derive(Clone, PartialEq, prost::Message)]
pub(crate) struct Feature {
    /// Identifier, unique within the layer when present.
    #[prost(uint64, optional, tag = "1")]
    pub(crate) id: Option<u64>,
    /// Flat `[key_index, value_index, ...]` pairs into the layer's tables.
    #[prost(uint32, repeated, packed = "true", tag = "2")]
    pub(crate) tags: Vec<u32>,
    /// A [`GeomType`] discriminant; `0` (`UNKNOWN`) when absent.
    #[prost(int32, tag = "3")]
    pub(crate) r#type: i32,
    /// The command/parameter integer stream (see [`super::wire`]).
    #[prost(uint32, repeated, packed = "true", tag = "4")]
    pub(crate) geometry: Vec<u32>,
}

/// One layer: a name, a coordinate extent and de-duplicated key/value tables.
#[derive(Clone, PartialEq, prost::Message)]
pub(crate) struct Layer {
    /// Layer name, unique within the tile.
    #[prost(string, tag = "1")]
    pub(crate) name: String,
    /// The layer's features, in encoder-chosen order.
    #[prost(message, repeated, tag = "2")]
    pub(crate) features: Vec<Feature>,
    /// Property-name table indexed by even tag entries.
    #[prost(string, repeated, tag = "3")]
    pub(crate) keys: Vec<String>,
    /// Property-value table indexed by odd tag entries.
    #[prost(message, repeated, tag = "4")]
    pub(crate) values: Vec<Value>,
    /// Tile-local coordinate extent; absent means the proto default `4096`.
    #[prost(uint32, optional, tag = "5")]
    pub(crate) extent: Option<u32>,
    /// Specification version this layer was written against (`1` or `2`).
    ///
    /// `required` upstream, so a missing field leaves `0` here and is rejected
    /// by the decoder.
    #[prost(uint32, tag = "15")]
    pub(crate) version: u32,
}

/// The top-level tile message: nothing but a list of layers.
#[derive(Clone, PartialEq, prost::Message)]
pub(crate) struct Tile {
    /// The tile's layers, in encoder-chosen order.
    #[prost(message, repeated, tag = "3")]
    pub(crate) layers: Vec<Layer>,
}

#[cfg(test)]
mod tests {
    use super::{Feature, GeomType, Layer, Tile, Value};
    use prost::Message as _;

    #[test]
    fn geom_type_rejects_unknown_and_unassigned() {
        assert_eq!(GeomType::from_i32(1), Some(GeomType::Point));
        assert_eq!(GeomType::from_i32(2), Some(GeomType::LineString));
        assert_eq!(GeomType::from_i32(3), Some(GeomType::Polygon));
        for value in [-1i32, 0, 4, 99, i32::MAX, i32::MIN] {
            assert_eq!(GeomType::from_i32(value), None, "value {value}");
        }
    }

    #[test]
    fn tile_round_trips_through_prost() {
        let tile = Tile {
            layers: vec![Layer {
                name: "roads".to_owned(),
                features: vec![Feature {
                    id: Some(7),
                    tags: vec![0, 0],
                    r#type: GeomType::Point as i32,
                    geometry: vec![9, 50, 34],
                }],
                keys: vec!["class".to_owned()],
                values: vec![Value {
                    string_value: Some("primary".to_owned()),
                    ..Value::default()
                }],
                extent: Some(4096),
                version: 2,
            }],
        };
        let bytes = tile.encode_to_vec();
        let decoded = Tile::decode(bytes.as_slice()).expect("re-decode");
        assert_eq!(decoded, tile);
    }

    #[test]
    fn absent_optionals_stay_distinguishable_from_zero() {
        // `extent: None` must not be confused with `extent: Some(0)`.
        let absent = Layer {
            version: 2,
            ..Layer::default()
        };
        let zero = Layer {
            version: 2,
            extent: Some(0),
            ..Layer::default()
        };
        assert!(absent.encode_to_vec().len() < zero.encode_to_vec().len());
        let re_absent = Layer::decode(absent.encode_to_vec().as_slice()).expect("decode");
        let re_zero = Layer::decode(zero.encode_to_vec().as_slice()).expect("decode");
        assert_eq!(re_absent.extent, None);
        assert_eq!(re_zero.extent, Some(0));
    }

    #[test]
    fn unknown_fields_are_skipped() {
        // Tag 9 (varint) is not part of `Tile`; a decoder must ignore it.
        // Key = (9 << 3) | wire type 0 = 72.
        let mut bytes = Tile { layers: Vec::new() }.encode_to_vec();
        bytes.extend_from_slice(&[72, 42]);
        let decoded = Tile::decode(bytes.as_slice()).expect("unknown field tolerated");
        assert!(decoded.layers.is_empty());
    }

    #[test]
    fn repeated_scalars_decode_from_both_packed_and_unpacked_forms() {
        // Feature.tags, tag 2. Unpacked: repeated varints, key (2 << 3) | 0 = 16.
        let unpacked = [16u8, 3, 16, 4];
        let decoded = Feature::decode(unpacked.as_slice()).expect("unpacked decode");
        assert_eq!(decoded.tags, vec![3, 4]);
        // Packed: one length-delimited blob, key (2 << 3) | 2 = 18, length 2.
        let packed = [18u8, 2, 3, 4];
        let decoded = Feature::decode(packed.as_slice()).expect("packed decode");
        assert_eq!(decoded.tags, vec![3, 4]);
    }
}
