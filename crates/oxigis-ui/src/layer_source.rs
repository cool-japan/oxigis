// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! What one tiled layer draws — the source *identity* shared by the live map
//! and the printed page.
//!
//! # Why this is its own module
//!
//! [`TileLayerSource`] used to live in [`crate::app::providers`], which made
//! [`crate::print`] — whose `PrintTileLayer` names it — depend on `app`, while
//! `app` already depends on `print`. That cycle is legal in Rust and compiled
//! cleanly, but it contradicted `print`'s own contract: that module is
//! documented as *extractable* ("layout maths, pixel composition and PDF
//! assembly, with no I/O, no egui and no GPU"), and a module that reaches back
//! into the application shell is not extractable whatever its own contents look
//! like.
//!
//! The type itself belongs to neither side. It is a **union of four
//! configuration structs** — [`CogLayerConfig`], [`ArchiveLayerConfig`],
//! [`BasemapConfig`] and [`VectorTileConfig`] — none of which live in `app`
//! either, and it holds no state, no provider handle and no UI. Both consumers
//! read it for the same reason: to decide what to build tiles from. So it moved
//! down here rather than being mirrored, which would have created two types
//! that must be kept in step by hand.
//!
//! Nothing about the public API changed: `app::providers` re-exports it, so
//! `crate::app::providers::TileLayerSource` still resolves, and the crate root's
//! `oxigis_ui::TileLayerSource` — what both shells match on — is the same path
//! it always was.

use crate::archive::ArchiveLayerConfig;
use crate::cog_provider::CogLayerConfig;
use crate::tile_provider::BasemapConfig;
use crate::vector_provider::VectorTileConfig;

/// What one tiled layer draws — the source *identity* a shell builds a provider
/// from.
///
/// **Deliberately free of opacity, visibility and stack position.** Those move
/// while a slider is dragged; if they were part of this, the reconciliation
/// would offer a fresh install on every frame of the drag and the shell would
/// rebuild the provider, which blanks and re-fetches every visible tile. Two
/// entries comparing equal really are the same source, and the only thing an
/// opacity change has to reach is [`crate::map_gpu::set_tile_layer_opacity`].
#[derive(Debug, Clone, PartialEq)]
pub enum TileLayerSource {
    /// A Cloud-Optimized GeoTIFF over HTTP Range.
    Cog(CogLayerConfig),
    /// A single-file archive of *image* tiles (PMTiles / MBTiles).
    RasterArchive(ArchiveLayerConfig),
    /// An XYZ tile service drawn as an ordinary stack layer rather than as the
    /// basemap.
    Xyz(BasemapConfig),
    /// Streamed vector tiles — MVT over HTTP or a vector tile archive, which
    /// [`VectorTileConfig`] already unifies.
    Vector(VectorTileConfig),
}

impl TileLayerSource {
    /// Whether this entry draws through the raster pipeline.
    #[must_use]
    pub fn is_raster(&self) -> bool {
        matches!(self, Self::Cog(_) | Self::RasterArchive(_) | Self::Xyz(_))
    }

    /// Whether this entry draws through the vector-tile pipeline.
    #[must_use]
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vector(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_belongs_to_exactly_one_pipeline() {
        // The invariant both consumers rely on: the map installs a raster
        // renderer or a vector one per entry, and the printed page composites a
        // raster pass or paints MVT paths per entry. A variant answering both —
        // or neither — would silently be drawn twice or not at all.
        let archive = ArchiveLayerConfig::new(
            oxigis_core::ArchiveRef::Url {
                url: "https://example.test/tiles.pmtiles".to_string(),
            },
            oxigis_core::ArchiveFormat::PmTiles,
        );
        for source in [
            TileLayerSource::Cog(CogLayerConfig::new("https://example.test/a.tif")),
            TileLayerSource::RasterArchive(archive),
            TileLayerSource::Xyz(BasemapConfig::default()),
            TileLayerSource::Vector(VectorTileConfig::new(
                "https://example.test/{z}/{x}/{y}.pbf",
            )),
        ] {
            assert_ne!(
                source.is_raster(),
                source.is_vector(),
                "{source:?} must belong to exactly one pipeline",
            );
        }
    }
}
