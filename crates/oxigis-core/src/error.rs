//! Crate-wide error type.
//!
//! Every fallible operation in `oxigis-core` returns [`CoreError`] instead
//! of panicking (COOLJAPAN Policy #3 — no `.unwrap()`/`.expect()` on
//! production paths).

use thiserror::Error;

use crate::layer::LayerId;

/// Errors produced by `oxigis-core` operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// No layer exists with the given [`LayerId`].
    #[error("layer not found: {0}")]
    LayerNotFound(LayerId),

    /// A hex color string failed to parse as `rrggbb`/`rrggbbaa`.
    #[error("invalid color {input:?}: {reason}")]
    InvalidColor {
        /// The string that failed to parse.
        input: String,
        /// Human-readable reason it was rejected.
        reason: String,
    },

    /// A project document failed to serialize or deserialize as JSON.
    #[error("project (de)serialization failed: {message}")]
    Json {
        /// The underlying `serde_json` error message.
        ///
        /// Stored as a `String` (rather than wrapping `serde_json::Error`
        /// directly) so `CoreError` stays `Clone + Eq`, which is convenient
        /// for equality assertions in tests and in callers that need to
        /// compare/log errors without re-running the fallible operation.
        message: String,
    },

    /// Lookup of a processing tool by id found nothing registered.
    #[error("unknown processing tool: {0:?}")]
    UnknownTool(String),

    /// A [`crate::processing::ToolExecutor`] rejected a parameter value.
    #[error("invalid parameter {name:?}: {reason}")]
    InvalidParameter {
        /// The offending parameter's name (matches
        /// [`crate::processing::ParamSpec::name`]).
        name: String,
        /// Human-readable reason it was rejected.
        reason: String,
    },

    /// A project file's `format_version` is newer than this build
    /// understands.
    ///
    /// Loading it anyway would silently misread (or drop) whatever changed
    /// in the newer wire shape, and — since the app re-saves the whole
    /// document — write that damage back over the user's file. See
    /// [`crate::project::CURRENT_FORMAT_VERSION`].
    #[error("project format version {found} is newer than this build supports (up to {supported})")]
    UnsupportedFormatVersion {
        /// `format_version` value read from the file.
        found: u32,
        /// The highest `format_version` this build understands.
        supported: u32,
    },

    /// Two layers in a project file share the same [`LayerId`].
    ///
    /// Every id-keyed lookup in this crate (`LayerStack::get`, `index_of`,
    /// `Project::styles`) resolves only the *first* match for a given id, so
    /// a duplicate would leave the second layer permanently unreachable and
    /// misdirect every mutation meant for it. Refused at load rather than
    /// left to corrupt those lookups silently.
    #[error("duplicate layer id in project file: {0}")]
    DuplicateLayerId(LayerId),

    /// [`crate::project::Project::set_style`] was asked to style a layer
    /// whose [`crate::layer::LayerKind`] never consults
    /// [`crate::project::Project::styles`] — a provider-drawn vector-tile
    /// layer (which carries its own paint list) or any raster layer.
    #[error("layer {0} does not accept a per-layer style (see LayerKind::accepts_layer_style)")]
    StyleNotApplicable(LayerId),
}

impl From<serde_json::Error> for CoreError {
    fn from(err: serde_json::Error) -> Self {
        CoreError::Json {
            message: err.to_string(),
        }
    }
}

/// Convenience alias for `Result<T, CoreError>`.
pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_not_found_displays_the_id() {
        let id = LayerId::from_raw(42);
        let err = CoreError::LayerNotFound(id);
        assert_eq!(err.to_string(), "layer not found: 42");
    }

    #[test]
    fn json_error_wraps_serde_json_message() {
        let parse_err = serde_json::from_str::<serde_json::Value>("{not valid").unwrap_err();
        let err: CoreError = parse_err.into();
        match &err {
            CoreError::Json { message } => assert!(!message.is_empty()),
            other => panic!("expected Json variant, got {other:?}"),
        }
    }

    #[test]
    fn core_error_is_clone_and_eq() {
        let a = CoreError::UnknownTool("bounds".to_string());
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn unsupported_format_version_displays_both_numbers() {
        let err = CoreError::UnsupportedFormatVersion {
            found: 2,
            supported: 1,
        };
        assert_eq!(
            err.to_string(),
            "project format version 2 is newer than this build supports (up to 1)"
        );
    }

    #[test]
    fn duplicate_layer_id_displays_the_id() {
        let err = CoreError::DuplicateLayerId(LayerId::from_raw(5));
        assert_eq!(err.to_string(), "duplicate layer id in project file: 5");
    }
}
