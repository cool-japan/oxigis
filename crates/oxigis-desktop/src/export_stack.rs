// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Composing an N-layer tiled stack onto an exported page (compositing v1.6).
//!
//! `PrintRequest` carries three legacy single-slot fields — one COG *or* one
//! tile archive, and one vector tileset — because that is all the map itself
//! could draw before the stack existed. A project holding an orthophoto under a
//! hillshade under a cadastral tileset draws all three on screen, so an export
//! reading only the three would silently print a different map from the one
//! being exported.
//!
//! This module is the shell half of closing that: it builds one provider per
//! stack entry, waits for its tiles on the same budget the single-slot path
//! uses, and hands `oxigis-ui` the composed raster plus one decoded-tile list
//! per vector entry.
//!
//! Split from `main.rs` under the 2000-line rule, beside `file_write` — and it
//! belongs apart anyway: everything here is export-time work on a background
//! thread, with no window, no frame and no GPU.
//!
//! # What is composed where
//!
//! The RASTER entries are composited on the CPU into the one image the page
//! embeds, bottom-up, each at its own opacity
//! ([`oxigis_ui::print::overlay_map_rgb`]). The VECTOR entries stay vectors:
//! their decoded tiles travel to `oxigis-ui`, which paints them as real PDF
//! paths over that image, in stack order. The one arrangement the page still
//! flattens is a raster ABOVE a vector tileset — the page's own documented
//! z-order — which is stated in `print`'s `page_content_planned_with`.

use std::sync::Arc;

use oxigis_render::mvt::VectorTile;
use oxigis_render::{MapView, TileId};
use oxigis_ui::print::{PrintRequest, PrintTileLayer};
use oxigis_ui::{BoxedTileProvider, TileLayerSource};

/// Everything one composed export produces: the finished raster and the
/// vector tiles each stack entry contributed.
pub(crate) struct ComposedStack {
    /// The page's raster image — the basemap with every raster entry over it.
    pub rgb: Vec<u8>,
    /// One decoded-tile list per `PrintRequest::stack` entry, in stack order;
    /// a raster entry's list is empty, so a list index IS a stack index.
    pub vector: Vec<Vec<(TileId, Arc<VectorTile>)>>,
    /// Raster tiles that never arrived, summed over the basemap and every
    /// raster entry — reported, never silently printed as gray.
    pub missing_raster: usize,
    /// The same for vector tiles.
    pub missing_vector: usize,
}

/// Builds every provider the stack names, waits for its tiles, and composes
/// the page's raster.
///
/// The basemap is the bottom pass and each entry is built **base-less** — the
/// opposite of the single-slot path, which asks the COG/archive provider to
/// blend the basemap in. Here the basemap has already been pasted, so a second
/// copy under every layer would be blended twice and each entry's opacity would
/// fade the basemap along with the layer.
///
/// # Errors
///
/// Only the basemap is required: without it there is nothing to paste the stack
/// onto. An entry whose provider cannot be built is logged and skipped, exactly
/// as the live map does — one unreachable archive must not cost the user the
/// whole export.
pub(crate) fn compose_stack(
    request: &PrintRequest,
    ctx: &egui::Context,
    compose: &MapView,
    required: &[TileId],
) -> Result<ComposedStack, String> {
    let basemap = crate::build_tile_provider(&request.basemap, ctx)
        .ok_or_else(|| "no tile provider could be built for the export".to_string())?
        .provider;
    let mut missing_raster = await_raster(&basemap, required, "basemap");
    let mut rgb = oxigis_ui::print::compose_map_rgb(compose, &mut |tile| basemap.tile(tile));

    let mut vector = Vec::with_capacity(request.stack.len());
    let mut missing_vector = 0_usize;
    for entry in &request.stack {
        match &entry.source {
            TileLayerSource::Vector(config) => {
                let (tiles, missing) = fetch_vector(config, ctx, compose, required);
                missing_vector += missing;
                vector.push(tiles);
            }
            _ => {
                vector.push(Vec::new());
                let Some(provider) = raster_provider(entry, ctx) else {
                    continue;
                };
                missing_raster += await_raster(&provider, required, "stack layer");
                oxigis_ui::print::overlay_map_rgb(compose, &mut rgb, entry.opacity, &mut |tile| {
                    provider.tile(tile)
                });
            }
        }
    }
    Ok(ComposedStack {
        rgb,
        vector,
        missing_raster,
        missing_vector,
    })
}

/// The base-less raster provider one stack entry draws through, or [`None`]
/// when it cannot be built here.
///
/// The session's archive bytes and MBTiles readers are deliberately not
/// consulted: this runs on a background export thread that holds no reference
/// to the app, so an archive is opened from its own location exactly as the
/// single-slot export path already opens `PrintRequest::archive`.
fn raster_provider(entry: &PrintTileLayer, ctx: &egui::Context) -> Option<BoxedTileProvider> {
    let built = match &entry.source {
        TileLayerSource::Cog(config) => crate::build_cog_provider(config, None, ctx),
        TileLayerSource::RasterArchive(config) => {
            crate::build_archive_provider(config, None, ctx, None, None)
        }
        TileLayerSource::Xyz(config) => crate::build_tile_provider(config, ctx),
        TileLayerSource::Vector(_) => None,
    };
    if built.is_none() {
        tracing::warn!(
            "OxiGIS desktop: a tiled layer's provider could not be built for the export; \
             the page is printed without it",
        );
    }
    built.map(|built| built.provider)
}

/// Polls `provider` until every tile of `required` has arrived or the pass's
/// budget is spent, returning how many never came.
///
/// `tile()` is what enqueues each fetch, so the first pass primes the whole
/// set — the same loop shape the single-slot export uses, and the same budget,
/// so an N-layer export is N passes of a known cost rather than an open-ended
/// wait.
fn await_raster(provider: &BoxedTileProvider, required: &[TileId], what: &str) -> usize {
    let deadline = std::time::Instant::now() + crate::tile_budget(required.len());
    loop {
        let missing = required
            .iter()
            .filter(|tile| provider.tile(**tile).is_none())
            .count();
        if missing == 0 {
            return 0;
        }
        if std::time::Instant::now() >= deadline {
            tracing::warn!(
                missing,
                total = required.len(),
                layer = what,
                "OxiGIS desktop: PDF export proceeding with missing tiles",
            );
            return missing;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Builds one vector-tile source, waits for its tiles, and collects the
/// decoded ones the page will paint.
fn fetch_vector(
    config: &oxigis_ui::VectorTileConfig,
    ctx: &egui::Context,
    compose: &MapView,
    required: &[TileId],
) -> (Vec<(TileId, Arc<VectorTile>)>, usize) {
    let Some(installed) = crate::build_vector_source(config, ctx, None, None) else {
        tracing::warn!(
            "OxiGIS desktop: a vector tile layer's source could not be built for the export; \
             the page is printed without it",
        );
        return (Vec::new(), 0);
    };
    let source = installed.source;
    let deadline = std::time::Instant::now() + crate::tile_budget(required.len());
    let mut missing = 0;
    loop {
        let _ = source.begin_frame(*compose);
        let absent = required
            .iter()
            .filter(|tile| {
                let absent = source.decoded(**tile).is_none();
                if absent {
                    let _ = source.mesh(**tile);
                }
                absent
            })
            .count();
        if absent == 0 {
            break;
        }
        if std::time::Instant::now() >= deadline {
            tracing::warn!(
                missing = absent,
                total = required.len(),
                "OxiGIS desktop: PDF export proceeding with missing vector tiles",
            );
            missing = absent;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let tiles = required
        .iter()
        .filter_map(|tile| source.decoded(*tile).map(|decoded| (*tile, decoded)))
        .collect();
    (tiles, missing)
}
