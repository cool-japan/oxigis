//! Small private helpers shared across modules.
//!
//! Not part of the public API — nothing here needs `missing_docs` coverage
//! at the crate boundary, but items are documented anyway for maintainers.

use serde::{Deserialize, Deserializer};

/// Clamps a value into the inclusive `0.0..=1.0` unit range.
///
/// `f32::clamp` never panics here because the bounds (`0.0`, `1.0`) are
/// fixed, non-NaN literals; a NaN `value` clamps to NaN (per IEEE 754,
/// matching `f32::clamp`'s documented behavior), which callers that also
/// clamp on every read will simply clamp again next time.
pub(crate) fn clamp_unit(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

/// A `serde(deserialize_with = ...)` helper that clamps an incoming `f32`
/// into `0.0..=1.0`, so opacity-like fields loaded from an out-of-range or
/// hand-edited project file are normalized rather than left invalid.
pub(crate) fn deserialize_clamped_unit<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let value = f32::deserialize(deserializer)?;
    Ok(clamp_unit(value))
}
