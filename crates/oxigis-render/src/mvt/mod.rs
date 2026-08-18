//! Mapbox Vector Tile 2.1 decoding: protobuf envelope, tag tables and geometry
//! command streams.
//!
//! An MVT tile is a protobuf message holding one or more *layers*. Each layer
//! carries a coordinate `extent` (the tile is a square `0..extent` grid), two
//! de-duplication tables (`keys`, `values`) and a list of *features*. A feature
//! is a geometry type, a flat `[key_index, value_index, ...]` tag list into
//! those tables, and a geometry expressed as a flat `Vec<u32>` of *command
//! integers* and zigzag-encoded coordinate deltas.
//!
//! # Layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`wire`] | zigzag varints and command integers — pure integer arithmetic |
//! | `proto` | hand-written `prost` messages mirroring `vector_tile.proto` (crate-private) |
//! | [`decode`] | `&[u8]` → [`VectorTile`]: validation, tags, geometry, rings |
//!
//! # What the decoder guarantees
//!
//! [`decode_mvt`] is total over hostile input: it never panics, never performs
//! an unbounded allocation, and the only way it returns [`RenderError::Mvt`]
//! is an invalid protobuf envelope. Everything else the specification
//! constrains — the layer version, `UNKNOWN` geometry types, odd tag lists,
//! out-of-range table indices, ambiguous `Value` messages, malformed command
//! sequences, coordinate cursors that leave `i32` range — is still checked,
//! but a violation isolates the layer or feature it belongs to instead of
//! failing the tile. [`decode::decode_mvt_report`] reports what, if anything,
//! was dropped; [`decode::decode_mvt_strict`] rejects such tiles outright.
//!
//! [`RenderError::Mvt`]: crate::error::RenderError::Mvt
//!
//! # What it deliberately does not do
//!
//! Coordinates stay in tile-local integer space; there is no reprojection and
//! no clipping to `0..extent`, because encoders legitimately emit buffer
//! coordinates outside that box. Turning the geometries into `lyon` paths
//! belongs to the tessellation step, which consumes this model.

pub mod decode;
pub(crate) mod proto;
pub mod wire;

pub use crate::mvt::decode::{
    DecodeReport, MvtFeature, MvtGeometry, MvtLayer, MvtPolygon, MvtValue, VectorTile, decode_mvt,
    decode_mvt_report, decode_mvt_strict,
};
pub use crate::mvt::wire::{
    Command, CommandId, MAX_COMMAND_COUNT, decode_command_integer, encode_command_integer,
    zigzag_decode, zigzag_encode,
};
