//! Error type shared by every module of the renderer.
//!
//! COOLJAPAN Policy #3 forbids `.unwrap()`/`.expect()` on production paths, so
//! every fallible operation in this crate returns [`RenderError`] instead of
//! panicking. The variants are intentionally coarse-grained: they carry enough
//! context for a shell to log or surface the failure, not enough to be matched
//! on exhaustively by callers.

use thiserror::Error;

/// Everything that can go wrong inside `oxigis-render`.
#[derive(Debug, Error)]
pub enum RenderError {
    /// A tile identifier was outside the valid slippy-map range for its zoom.
    #[error("invalid tile {z}/{x}/{y}: {reason}")]
    InvalidTile {
        /// Zoom level of the rejected tile.
        z: u8,
        /// Column of the rejected tile.
        x: u32,
        /// Row of the rejected tile.
        y: u32,
        /// Why the identifier was rejected.
        reason: &'static str,
    },

    /// A zoom level was not finite or fell outside `0..=`[`MAX_ZOOM`].
    ///
    /// [`MAX_ZOOM`]: crate::mercator::MAX_ZOOM
    #[error("zoom {zoom} is out of range 0..={max}")]
    ZoomOutOfRange {
        /// The rejected zoom level.
        zoom: f64,
        /// The highest supported integer zoom level.
        max: u8,
    },

    /// A viewport parameter (size, centre or zoom) was unusable.
    #[error("invalid viewport: {0}")]
    InvalidViewport(String),

    /// A cache was constructed with a capacity that cannot hold anything.
    #[error("invalid cache capacity: {0}")]
    InvalidCapacity(usize),

    /// An XYZ URL template was malformed or missing a required placeholder.
    #[error("invalid tile url template: {0}")]
    InvalidTemplate(String),

    /// A byte range was empty or inverted.
    #[error("invalid byte range: {0}")]
    InvalidRange(String),

    /// A shell-injected fetcher failed to deliver tile bytes.
    ///
    /// The renderer never performs I/O itself; this variant only ever carries a
    /// message produced by a [`TileFetch`]/[`RangeFetch`] implementation.
    ///
    /// [`TileFetch`]: crate::source::TileFetch
    /// [`RangeFetch`]: crate::source::RangeFetch
    #[error("tile fetch failed: {0}")]
    Fetch(String),

    /// Decoded tile pixels did not match the declared dimensions or format.
    #[error("invalid tile image: {0}")]
    InvalidTileImage(String),

    /// A GPU resource could not be created, sized or bound.
    #[error("gpu: {0}")]
    Gpu(String),

    /// [`MapRenderer::paint`] was called before [`MapRenderer::prepare`] had a
    /// chance to build the GPU pipeline.
    ///
    /// [`MapRenderer::paint`]: crate::renderer::MapRenderer::paint
    /// [`MapRenderer::prepare`]: crate::renderer::MapRenderer::prepare
    #[error("renderer has not been prepared: call prepare(device, queue) first")]
    NotPrepared,

    /// A Mapbox Vector Tile byte stream violated the encoding rules.
    #[error("invalid mvt data: {0}")]
    Mvt(String),

    /// A vector geometry could not be turned into triangles.
    ///
    /// Carries either a `lyon` tessellation failure or one of the limits
    /// [`crate::vector::tessellate_tile`] enforces itself (mesh size, layer
    /// extent). Degenerate-but-harmless input — an empty ring, a one-point
    /// line — is skipped silently instead of producing this.
    #[error("tessellation failed: {0}")]
    Tessellation(String),

    /// Text could not be shaped, rasterised or packed into the glyph atlas.
    ///
    /// Carries either an `oxitext` failure (an unparseable font, a shaper
    /// error) or one of the limits [`crate::label`] enforces itself (label size
    /// range, atlas dimensions). A *full* atlas is not this: it is a routine
    /// condition the engine answers by rebuilding.
    #[error("text: {0}")]
    Text(String),

    /// A deliberately deferred code path was reached.
    ///
    /// Used by skeletons whose full implementation is tracked in `TODO.md`
    /// rather than silently returning wrong data.
    #[error("{feature} is not implemented yet (tracked by {tracking})")]
    Unimplemented {
        /// Human-readable name of the missing capability.
        feature: &'static str,
        /// Where the work is tracked, e.g. `"TODO.md 5.1"`.
        tracking: &'static str,
    },

    /// A well-formed input asked for something this renderer does not support.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Raw tile bytes could not be sniffed or decoded into RGBA8 pixels.
    ///
    /// Produced by [`crate::decode::decode_tile`] (feature `decode`) — never
    /// by network or filesystem access, which this crate does not perform.
    #[error("tile image decode failed: {0}")]
    Decode(String),
}

#[cfg(test)]
mod tests {
    use super::RenderError;

    #[test]
    fn messages_are_descriptive() {
        let err = RenderError::InvalidTile {
            z: 3,
            x: 9,
            y: 0,
            reason: "x out of range",
        };
        assert_eq!(err.to_string(), "invalid tile 3/9/0: x out of range");

        let err = RenderError::Unimplemented {
            feature: "COG IFD parsing",
            tracking: "TODO.md 5.1",
        };
        assert_eq!(
            err.to_string(),
            "COG IFD parsing is not implemented yet (tracked by TODO.md 5.1)"
        );
    }

    #[test]
    fn error_is_std_error() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&RenderError::NotPrepared);
    }
}
