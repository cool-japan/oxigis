// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GPU tile providers derived from the project — the reconciliation seam
//! that replaced the take-once `pending_*` one-shots in editing v1.3.
//!
//! What is drawn used to be a function of the *history of one-shots*
//! (`take_pending_basemap` / `take_pending_cog_layer` /
//! `take_pending_vector_layer`), which is why removing a COG or MVT layer —
//! or undoing its add, or loading a project that contains one — left the GPU
//! drawing something the layer panel no longer listed. It is now a function
//! of the **project**: [`OxigisApp::desired_raster`] and
//! [`OxigisApp::desired_vector`] derive what the map must draw, a shell
//! compares that against what it last confirmed, and every path that changes
//! the project (add, remove, reorder, undo, redo, project load, File ▸ New)
//! moves the map for free.
//!
//! # Shell protocol (replaces the deleted one-shots)
//!
//! Once per frame, **guard on the render state first**:
//!
//! ```text
//! if let Some(render_state) = frame.wgpu_render_state() {
//!     if let Some(work) = app.pending_raster_work() {
//!         let provider = match work.cog.as_ref() {
//!             Some(cog) => build_cog_provider(cog, &work.basemap, &ctx),
//!             None => build_tile_provider(&work.basemap, &ctx),
//!         };
//!         let outcome = ...; // replace_provider -> Ok(()) / Err(reason)
//!         app.settle_raster_work(work, outcome);
//!     }
//!     match app.pending_vector_work() {
//!         Some(VectorWork::Install(config)) => { /* build + replace_vector_source, then settle */ }
//!         Some(VectorWork::Detach) => { /* clear_vector_source, then settle Ok */ }
//!         None => {}
//!     }
//! }
//! ```
//!
//! The work **stays offered until it is settled**, so a frame with no render
//! state defers an install instead of losing it (the old seams' "was
//! discarded" failure). A settle with `Err` memoizes the refusal — the shell
//! is not asked to rebuild a provider it already failed to build, once per
//! frame, for ever — until the project implies a different plan.
//!
//! # Refusals are visible, and retryable
//!
//! The memo carries the shell's own reason ([`Refusal`]), and whether it is
//! *shown* is derived, never stored: [`OxigisApp::provider_refusal`] reports a
//! memoized reason only while its plan is still the one the project implies,
//! so a memo the desire has moved past can never badge a map that is drawing
//! fine. [`OxigisApp::retry_refused_installs`] drops both memos — a command,
//! not an edit: it writes no project state and never touches the undo stack.
//!
//! Migration for out-of-tree shells: `take_pending_basemap` +
//! `take_pending_cog_layer` → [`OxigisApp::pending_raster_work`] +
//! [`OxigisApp::settle_raster_work`]; `take_pending_vector_layer` →
//! [`OxigisApp::pending_vector_work`] + [`OxigisApp::settle_vector_work`];
//! `active_cog_layer` → `desired_raster().cog`; `active_vector_layer` →
//! `desired_vector()`.
//!
//! # The drawn stack (compositing v1.6): N layers, not one of each
//!
//! [`OxigisApp::desired_raster`] and [`OxigisApp::desired_vector`] each answer
//! with **one** layer — the top-most visible one of their kind — which is why a
//! project holding an orthophoto under a hillshade, or a DEM plus a cadastral
//! vector tileset, drew exactly one of the two while the other sat in the layer
//! tree with its visibility checkbox ticked. [`OxigisApp::desired_tile_stack`]
//! is the generalisation: **every** visible tiled layer, raster and vector-tile
//! interleaved, bottom-up, capped at [`MAX_DRAWN_TILE_LAYERS`] with the
//! remainder named in [`TileStack::undrawn`] rather than silently dropped.
//!
//! Two rules make it work, and both are load-bearing:
//!
//! * **The mirror is the map.** There is no `tile_stack_installed` field beside
//!   the project; what is installed is read back from the GPU state
//!   ([`crate::map_gpu::installed_tile_stack`]) and
//!   [`OxigisApp::tile_stack_work`] is a pure diff against it. A refusal is
//!   recorded *by the map*, on the entry, which is what stops a failed build
//!   being retried once per frame for ever.
//! * **Opacity is not part of a plan.** [`TileLayerSource`] carries source
//!   identity and nothing else, so a slider drag — which emits every frame —
//!   offers no work at all; the value reaches the GPU through
//!   [`crate::map_gpu::sync_tile_layer_opacities`] as an instance tint, which
//!   costs neither a texture nor a tessellation. That is what turns the layer
//!   panel's Opacity slider from a value written to disk and read by nothing
//!   into a control that means the same thing for a COG, an archive, an MVT
//!   source and a dropped GeoJSON.
//!
//! ```text
//! let installed = map_gpu::installed_tile_stack(render_state);
//! if let Some(work) = app.tile_stack_work(&installed) { /* one build, or a
//!     remove, or a reorder — see `map_gpu::installed_tile_stack` */ }
//! map_gpu::sync_tile_layer_opacities(render_state, |id| app.tile_layer_opacity(id));
//! ```
//!
//! The single-slot seams above are left exactly as they were, so a shell
//! migrates when it is ready; a shell that has migrated builds its stack
//! providers **without a basemap underneath** (`CogTileProvider::with_base` /
//! `ArchiveTileProvider::with_base` are skipped), because in the stack the
//! basemap is its own pass and CPU-blending it into every layer would fade the
//! basemap along with the layer.

use crate::archive::ArchiveLayerConfig;
use crate::cog_provider::CogLayerConfig;
use crate::tile_provider::BasemapConfig;
use crate::vector_provider::{self, VectorTileConfig};
use oxigis_core::{Layer, LayerId, LayerKind, RasterSource, VectorSource};

use super::OxigisApp;

/// The raster tile provider the project implies: always a basemap, with the
/// top-most raster *layer* composited over it when the project holds one.
///
/// `cog` and `archive` are the two kinds of raster layer, and **at most one is
/// ever set**: [`OxigisApp::desired_raster`] picks the top-most visible raster
/// layer of either kind, so "newest wins" holds across the kinds and not only
/// within one.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterWork {
    /// The basemap to build from.
    pub basemap: BasemapConfig,
    /// The COG composited over it, when the top-most raster layer is one.
    pub cog: Option<CogLayerConfig>,
    /// The tile archive composited over it, when the top-most raster layer is
    /// one.
    pub archive: Option<ArchiveLayerConfig>,
}

/// A plan a shell reported it could not build, together with the reason it
/// gave — the memo that stops a failed install from being retried once per
/// frame for ever.
///
/// Compared by `work` alone wherever the memo suppresses an offer, so
/// carrying the reason changes no reconciliation behaviour; the reason exists
/// only so the refusal can be *shown*, and shown in the shell's own words.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal<W> {
    /// The plan that was refused.
    pub work: W,
    /// What the shell said went wrong, ready to put in front of the user.
    pub reason: String,
}

/// What the vector-tile slot must become to match the project.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorWork {
    /// Build a source for this config and install it with
    /// [`crate::map_gpu::replace_vector_source`].
    Install(VectorTileConfig),
    /// Detach whatever is installed
    /// ([`crate::map_gpu::clear_vector_source`]).
    Detach,
}

/// The basemap an XYZ layer describes, or [`None`] for any other kind.
///
/// `subdomains` stays empty because a layer records no `{s}` host list: a
/// template that needs one therefore cannot resolve, and
/// [`BasemapConfig::template`] is what says so — one rule, not two.
fn xyz_basemap(layer: &Layer) -> Option<BasemapConfig> {
    match &layer.kind {
        LayerKind::Raster(RasterSource::Xyz {
            url_template,
            attribution,
        }) => Some(BasemapConfig {
            url_template: url_template.clone(),
            subdomains: Vec::new(),
            attribution: attribution.clone(),
        }),
        _ => None,
    }
}

/// Largest number of tiled layers the map composites over the basemap in one
/// frame.
///
/// Every entry is a whole renderer — its own tile cache, its own GPU pipeline,
/// its own share of the texture/mesh byte budget — so an unbounded stack is a
/// performance cliff and a VRAM one. Eight is well past what a real map layers
/// (orthophoto ▸ hillshade ▸ cadastre ▸ labels is four) and still divides the
/// budgets into usable shares. Visible tiled layers past the cap are reported
/// by [`TileStack::undrawn`] rather than silently dropped.
pub const MAX_DRAWN_TILE_LAYERS: usize = 8;

/// Re-exported from [`crate::layer_source`], where the type now lives.
///
/// It moved because [`crate::print`] names it too (a `PrintTileLayer` is a
/// tiled layer of the exported page), and `print` importing it from `app` made
/// the wasm-clean printing module depend on the application shell that depends
/// on it. The path here is kept so no caller — inside the crate or through
/// `oxigis_ui::TileLayerSource` — had to change.
pub use crate::layer_source::TileLayerSource;

/// One layer of the drawn stack: which project layer it is, and what it draws.
#[derive(Debug, Clone, PartialEq)]
pub struct TileLayerPlan {
    /// The project layer this entry draws — the key every
    /// `crate::map_gpu::*_tile_layer*` entry point takes.
    pub layer: LayerId,
    /// Where its tiles come from.
    pub source: TileLayerSource,
}

/// The tiled layers the map draws over the basemap, bottom-up, plus the ones
/// [`MAX_DRAWN_TILE_LAYERS`] left out.
///
/// Raster and vector-tile entries are **interleaved in one list**: a cadastral
/// vector tileset under a hillshade raster is a stack order the layer panel can
/// express, so it has to be one the map can draw.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TileStack {
    /// Bottom-up: index `0` paints first, directly over the basemap, and the
    /// last entry paints last. The same order the layer panel shows read from
    /// the bottom up.
    pub entries: Vec<TileLayerPlan>,
    /// Visible tiled layers the cap left undrawn, bottom-most first — the
    /// buried ones, since the cap keeps the top of the stack.
    pub undrawn: Vec<LayerId>,
}

impl TileStack {
    /// Whether `layer` is one of the entries the map draws.
    #[must_use]
    pub fn draws(&self, layer: LayerId) -> bool {
        self.entries.iter().any(|entry| entry.layer == layer)
    }

    /// Whether `layer` is visible and tiled but past the cap, so it is listed
    /// in the panel and draws nothing.
    #[must_use]
    pub fn hides(&self, layer: LayerId) -> bool {
        self.undrawn.contains(&layer)
    }

    /// The one sentence to put in front of the user when the cap is biting,
    /// or [`None`] when every visible tiled layer is drawn.
    ///
    /// Deliberately **not** folded into the map's `credit_line`: that line is
    /// the attribution, read by the painted credit and by the exported PDF
    /// page, and an apology about a draw budget is not a credit anybody is
    /// owed.
    #[must_use]
    pub fn notice(&self) -> Option<String> {
        let hidden = self.undrawn.len();
        if hidden == 0 {
            return None;
        }
        let layers = if hidden == 1 { "layer" } else { "layers" };
        Some(format!(
            "{hidden} more tiled {layers} {} visible but not drawn: the map composites at most \
             {MAX_DRAWN_TILE_LAYERS} tiled layers at once. Hide one of the upper layers to draw \
             a lower one.",
            if hidden == 1 { "is" } else { "are" }
        ))
    }
}

/// The change the drawn stack still needs to match the project — one unit at a
/// time, so a frame does at most one provider build.
///
/// Same **offer-until-settled** contract as [`VectorWork`]: what is installed
/// is read back from the GPU state
/// ([`crate::map_gpu::installed_tile_stack`]), so a frame without a render
/// state defers the work instead of losing it, and there is no second mirror
/// that could disagree with the map.
#[derive(Debug, Clone, PartialEq)]
pub enum TileStackWork {
    /// Build a provider for this layer and install it with
    /// [`crate::map_gpu::install_tile_layer`]. Also the *replace* case: a layer
    /// whose source changed is re-installed under the same id, in place.
    Install(TileLayerPlan),
    /// Drop these layers' providers
    /// ([`crate::map_gpu::remove_tile_layers`]) — they were removed, hidden,
    /// pushed past the cap, or the whole project was closed.
    ///
    /// Batched, unlike [`TileStackWork::Install`], because a removal builds
    /// nothing: emitting them one per frame would leave a closed project's
    /// layers on screen for as many frames as it had, vanishing one by one.
    Remove(Vec<LayerId>),
    /// Put the installed entries into this bottom-up order
    /// ([`crate::map_gpu::reorder_tile_layers`]). Rebuilds nothing: a drag in
    /// the layer panel must not re-fetch a single tile.
    Reorder(Vec<LayerId>),
}

/// The top-most visible raster layer, whichever kind it is.
///
/// Private, and exists only so the one scan in [`OxigisApp::desired_raster`]
/// can return either kind without two passes that could disagree about which
/// layer is on top.
enum TopRaster {
    /// A Cloud-Optimized GeoTIFF.
    Cog(CogLayerConfig),
    /// A single-file tile archive of image tiles.
    Archive(ArchiveLayerConfig),
}

/// The raster source `layer` draws through the tiled pipeline, or [`None`] for
/// any other kind — and for an archive this build cannot read, which is the
/// same rule the add seam consults, so a hand-edited project file cannot get an
/// MBTiles-over-HTTP layer past it either.
///
/// XYZ is deliberately **not** here: it is the basemap's kind, and
/// [`OxigisApp::desired_raster`] must keep answering "no composited layer" for
/// it. [`tile_layer_source`] adds it back for the stack, where a non-promoted
/// XYZ layer is an ordinary overlay.
fn raster_layer_source(layer: &Layer) -> Option<TileLayerSource> {
    match &layer.kind {
        LayerKind::Raster(RasterSource::Cog { url }) => {
            Some(TileLayerSource::Cog(CogLayerConfig::new(url.clone())))
        }
        LayerKind::Raster(RasterSource::TileArchive {
            archive,
            format,
            attribution,
        }) => {
            let config = ArchiveLayerConfig::new(archive.clone(), *format)
                .with_attribution(attribution.clone());
            config
                .refusal()
                .is_none()
                .then_some(TileLayerSource::RasterArchive(config))
        }
        _ => None,
    }
}

/// The vector-tile source `layer` draws, or [`None`] for any other kind.
///
/// The MVT arm goes through [`vector_provider::config_for`] — the ONE rule the
/// add seam also uses — so a loaded project credits MapLibre exactly as a fresh
/// add does.
fn vector_layer_source(layer: &Layer) -> Option<TileLayerSource> {
    match &layer.kind {
        LayerKind::Vector(VectorSource::MvtTiles {
            url_template,
            paints,
        }) => Some(TileLayerSource::Vector(vector_provider::config_for(
            url_template,
            paints.clone(),
        ))),
        LayerKind::Vector(VectorSource::TileArchive {
            archive,
            format,
            paints,
            attribution,
        }) => {
            let config = ArchiveLayerConfig::new(archive.clone(), *format)
                .with_attribution(attribution.clone());
            config.refusal().is_none().then(|| {
                TileLayerSource::Vector(VectorTileConfig::from_archive(config, paints.clone()))
            })
        }
        _ => None,
    }
}

/// Everything a stack entry can be — the classifier
/// [`OxigisApp::desired_tile_stack`] scans with.
///
/// Built from the very helpers [`OxigisApp::desired_raster`] and
/// [`OxigisApp::desired_vector`] scan with, so the stack and the two legacy
/// single-slot derivations cannot disagree about what a layer *is*; the
/// agreement is pinned by the tests in `app::tests_providers`.
///
/// The order of the arms is only an optimisation: the three are mutually
/// exclusive by [`LayerKind`].
fn tile_layer_source(layer: &Layer) -> Option<TileLayerSource> {
    raster_layer_source(layer)
        .or_else(|| vector_layer_source(layer))
        // An XYZ layer that is not drawing as the basemap is an ordinary
        // overlay — before the stack it was listed, ticked, and drew nothing at
        // all. A template needing a `{s}` host list cannot resolve (a layer
        // records none), so it never becomes an entry.
        .or_else(|| {
            xyz_basemap(layer)
                .filter(|config| config.template().is_ok())
                .map(TileLayerSource::Xyz)
        })
}

/// Whether `layer` draws through one of the tiled pipelines at all — the same
/// verdict as `tile_layer_source(layer).is_some()`, reached **without building
/// the config**.
///
/// Exists for the two callers that need the verdict and not the source: the
/// over-cap tail of [`OxigisApp::desired_tile_stack`] (which records ids, and
/// would otherwise allocate a whole paint table per buried MVT layer, once per
/// frame — the very waste `vector_credit` was written to avoid) and a layer
/// panel deciding whether a row's opacity slider means anything.
///
/// The archive arm reproduces [`crate::archive::ArchiveLayerConfig::refusal`]
/// through [`oxigis_core::archive_refusal`], exactly as `raster_credit` does;
/// the two are pinned to `tile_layer_source` by an agreement test in
/// `app::tests_providers`.
#[must_use]
pub fn draws_as_tile_layer(layer: &Layer) -> bool {
    match &layer.kind {
        LayerKind::Raster(RasterSource::Cog { .. })
        | LayerKind::Vector(VectorSource::MvtTiles { .. }) => true,
        LayerKind::Raster(RasterSource::TileArchive {
            archive, format, ..
        })
        | LayerKind::Vector(VectorSource::TileArchive {
            archive, format, ..
        }) => oxigis_core::archive_refusal(archive, *format).is_none(),
        LayerKind::Raster(RasterSource::Xyz { .. }) => {
            xyz_basemap(layer).is_some_and(|config| config.template().is_ok())
        }
        _ => false,
    }
}

impl OxigisApp {
    /// The camera zoom every "is this layer drawing?" question is asked at.
    ///
    /// [`Layer::visible_at`] is the ONE predicate the derivations cull with —
    /// it is the checkbox *and* the scale range together, so a call site that
    /// uses it cannot honour one and forget the other — and this is the single
    /// place the camera it needs is read. Resolving it once per derivation (not
    /// once per layer) matters: the scan below runs over the whole project,
    /// every frame.
    ///
    /// It makes the derivations camera-dependent, which is correct and is the
    /// point: a layer whose `max_zoom` the camera has passed must stop drawing,
    /// and the reconciliation already re-derives every frame, so honouring it
    /// costs nothing extra.
    fn drawn_zoom(&self) -> f64 {
        self.map_panel.view().zoom()
    }

    /// The raster provider the project implies right now — the basemap with
    /// the top-most **visible** COG layer composited over it (the same
    /// "newest wins" rule the composite provider has always drawn, plus the
    /// checkbox meaning what it says since v1.3). Pure: derived fresh from
    /// `(project, basemap)` on every call, never remembered.
    ///
    /// Consulting visibility means hiding and re-showing a COG rebuilds the
    /// provider and re-fetches its tiles — the honest cost of a checkbox
    /// that actually does something (it used to do nothing at all for
    /// provider layers).
    #[must_use]
    pub fn desired_raster(&self) -> RasterWork {
        // ONE scan for both kinds, so "the top-most visible raster layer wins"
        // is true across kinds: a COG added over an archive hides it, and vice
        // versa, exactly as the layer panel's order promises.
        let zoom = self.drawn_zoom();
        let top = self
            .project
            .layers
            .layers()
            .iter()
            .rev()
            .filter(|layer| layer.visible_at(zoom))
            .find_map(|layer| match raster_layer_source(layer) {
                Some(TileLayerSource::Cog(cog)) => Some(TopRaster::Cog(cog)),
                Some(TileLayerSource::RasterArchive(archive)) => Some(TopRaster::Archive(archive)),
                // `raster_layer_source` yields no other arm; a vector layer or
                // an unreadable archive is simply scanned past, exactly as
                // before this shared the stack's classifier.
                _ => None,
            });
        let (cog, archive) = match top {
            Some(TopRaster::Cog(cog)) => (Some(cog), None),
            Some(TopRaster::Archive(archive)) => (None, Some(archive)),
            None => (None, None),
        };
        RasterWork {
            // The promoted layer, when one resolves, else the service. The
            // scan above never matches `RasterSource::Xyz`, so a promoted
            // layer cannot also arrive as the composited raster.
            basemap: self.drawn_basemap(),
            cog,
            archive,
        }
    }

    /// The promoted layer itself, when the project's `basemap_layer` pointer
    /// resolves to something that can actually draw.
    ///
    /// **Total**: an absent id, a hidden layer, a non-XYZ kind and an unusable
    /// template all answer [`None`] rather than an error, and the service
    /// draws. Visibility is consulted here through [`Layer::visible_at`] — a
    /// promoted layer whose checkbox is off is not drawn, exactly as v1.3 made
    /// the checkbox mean, and one the camera has zoomed out of its scale range
    /// is not drawn either, which is what stops the promoted layer being the
    /// one place a range is ignored.
    ///
    /// Returns the borrow rather than a built [`BasemapConfig`] because the
    /// credit line runs every frame and needs only one `&str` of it; the
    /// config-shaped answer is [`Self::promoted_basemap`], one line below.
    fn promoted_basemap_layer(&self) -> Option<&Layer> {
        let id = self.project.basemap_layer?;
        let layer = self.project.layers.get(id)?;
        if !layer.visible_at(self.drawn_zoom()) || self.promotion_refusal(id).is_some() {
            return None;
        }
        matches!(layer.kind, LayerKind::Raster(RasterSource::Xyz { .. })).then_some(layer)
    }

    /// The promoted layer's basemap, when one resolves — the owned twin of
    /// [`Self::promoted_basemap_layer`], for the callers that need a whole
    /// config rather than a credit line.
    fn promoted_basemap(&self) -> Option<BasemapConfig> {
        self.promoted_basemap_layer().and_then(xyz_basemap)
    }

    /// The basemap the map actually draws: the promoted layer's, else the
    /// service's. The ONE resolution every consumer reads — the raster plan,
    /// the credit line and the exported page — so none of them can draw a
    /// different basemap from the others.
    #[must_use]
    pub fn drawn_basemap(&self) -> BasemapConfig {
        self.promoted_basemap()
            .unwrap_or_else(|| self.basemap.clone())
    }

    /// Whether `layer` is the one currently drawn as the basemap — what the
    /// panel's toggle reads, so the toggle reflects *resolution* and not just
    /// the stored pointer.
    #[must_use]
    pub fn draws_as_basemap(&self, layer: LayerId) -> bool {
        self.project.basemap_layer == Some(layer) && self.promoted_basemap().is_some()
    }

    /// Why `layer` cannot be promoted to the basemap, when it cannot.
    ///
    /// The ONE promotability rule: the gesture refuses by it and a project
    /// load reports by it, so a hand-edited file cannot install a pointer the
    /// gesture would have refused. Deliberately **not** about visibility — a
    /// hidden promoted layer is a legal, recorded state that merely does not
    /// resolve — and deliberately not consulted when the recorded op is
    /// applied, since promotability is derived state and an undo must not
    /// start refusing because a file names a COG.
    pub(super) fn promotion_refusal(&self, layer: LayerId) -> Option<String> {
        let Some(entry) = self.project.layers.get(layer) else {
            return Some(format!("layer {layer} is not in the project"));
        };
        let Some(config) = xyz_basemap(entry) else {
            return Some(format!(
                "\u{201c}{}\u{201d} is not an XYZ tile layer",
                entry.name
            ));
        };
        config.template().err().map(|error| error.to_string())
    }

    /// The streamed vector-tile source the project implies right now — the
    /// top-most **visible** MVT layer, its config rebuilt by
    /// [`vector_provider::config_for`] (the ONE rule the add seam also uses,
    /// so a loaded project credits MapLibre exactly as a fresh add does).
    #[must_use]
    pub fn desired_vector(&self) -> Option<VectorTileConfig> {
        let zoom = self.drawn_zoom();
        self.project
            .layers
            .layers()
            .iter()
            .rev()
            .filter(|layer| layer.visible_at(zoom))
            .find_map(|layer| match vector_layer_source(layer) {
                Some(TileLayerSource::Vector(config)) => Some(config),
                // `vector_layer_source` yields no other arm.
                _ => None,
            })
    }

    /// Every tiled layer the project says to draw, bottom-up — the derivation
    /// that replaces "only the top-most raster layer and the top-most
    /// vector-tile layer ever draw".
    ///
    /// Pure, like [`Self::desired_raster`]: derived fresh from the project on
    /// every call and never remembered, so an add, a remove, a hide, a reorder,
    /// an undo of any of them and a project load all move the map for free.
    ///
    /// **Opacity is not here.** The entries carry source identity only, which is
    /// what lets a slider drag be honoured without offering a single install —
    /// see [`TileLayerSource`] and [`Self::tile_layer_opacity`].
    ///
    /// The layer drawing as the basemap is skipped: it is already on screen
    /// under everything, and drawing it twice would be both wrong and slower.
    #[must_use]
    pub fn desired_tile_stack(&self) -> TileStack {
        // Resolved ONCE rather than per layer: `draws_as_basemap` clones a
        // whole config to answer, and this runs over the entire stack.
        let promoted = self.promoted_basemap_layer().map(|layer| layer.id);
        // Hoisted for the same reason `promoted` is: reading the camera per
        // layer would repeat a whole `MapView` build over the entire project,
        // every frame.
        let zoom = self.drawn_zoom();
        let mut entries: Vec<TileLayerPlan> = Vec::new();
        let mut undrawn: Vec<LayerId> = Vec::new();
        // Scanned TOP-DOWN so the cap keeps the top of the stack — the layers
        // the pre-stack "newest wins" rule drew — and reports the buried ones,
        // rather than dropping whatever happened to be added last.
        for layer in self.project.layers.layers().iter().rev() {
            if !layer.visible_at(zoom) || Some(layer.id) == promoted {
                continue;
            }
            if entries.len() >= MAX_DRAWN_TILE_LAYERS {
                // Past the cap only the id is wanted, so the config is never
                // built: a project with thirty buried MVT layers must not
                // allocate thirty paint tables per frame to say so.
                if draws_as_tile_layer(layer) {
                    undrawn.push(layer.id);
                }
                continue;
            }
            let Some(source) = tile_layer_source(layer) else {
                continue;
            };
            entries.push(TileLayerPlan {
                layer: layer.id,
                source,
            });
        }
        // Back to bottom-up, the order the passes run in.
        entries.reverse();
        undrawn.reverse();
        TileStack { entries, undrawn }
    }

    /// The one change the drawn stack still owes the project, given what the
    /// GPU currently holds (from [`crate::map_gpu::installed_tile_stack`]), or
    /// [`None`] when the two already agree.
    ///
    /// Pure and memo-free: `installed` IS the mirror, read back from the map
    /// itself, so there is no second copy of the truth to go stale. A build a
    /// shell could not do is recorded *by the map* as a refused entry (see
    /// [`crate::map_gpu::install_tile_layer`]), which occupies the slot and
    /// stops the plan being offered once per frame for ever —
    /// [`crate::map_gpu::retry_refused_tile_layers`] is the way back.
    ///
    /// One unit per call, in a deliberate order: **remove, then install, then
    /// reorder**. Layers that are gone stop drawing before anything is rebuilt
    /// (so a removed layer is never left on screen behind a slow fetch) — all
    /// of them at once, which is what makes File ▸ New and a project load clear
    /// the previous project's tiled layers in a single frame rather than one
    /// per frame. Reordering, which rebuilds nothing, is the last frame of the
    /// convergence.
    #[must_use]
    pub fn tile_stack_work(&self, installed: &[TileLayerPlan]) -> Option<TileStackWork> {
        let desired = self.desired_tile_stack();
        let stale: Vec<LayerId> = installed
            .iter()
            .filter(|entry| !desired.draws(entry.layer))
            .map(|entry| entry.layer)
            .collect();
        if !stale.is_empty() {
            return Some(TileStackWork::Remove(stale));
        }
        if let Some(plan) = desired.entries.iter().find(|want| {
            installed
                .iter()
                .find(|have| have.layer == want.layer)
                .map(|have| &have.source)
                != Some(&want.source)
        }) {
            return Some(TileStackWork::Install(plan.clone()));
        }
        let ordered = desired
            .entries
            .iter()
            .zip(installed)
            .all(|(want, have)| want.layer == have.layer);
        if ordered && desired.entries.len() == installed.len() {
            return None;
        }
        Some(TileStackWork::Reorder(
            desired.entries.iter().map(|entry| entry.layer).collect(),
        ))
    }

    /// The one sentence the stack's refused entries deserve, given what
    /// [`crate::map_gpu::tile_layer_refusals`] reports — layers *named*, so the
    /// user can find the row that is not drawing.
    ///
    /// Lives here rather than in the shell so the wording is decided once,
    /// beside [`Self::provider_refusal`] and with the same `credit_line`
    /// separator; it takes the refusals as an argument because they are GPU
    /// state, and this derivation reads only the project (for the names).
    /// Refusals naming a layer the project no longer holds are dropped: they
    /// are about a plan the desire has already moved past.
    #[must_use]
    pub fn tile_layer_refusal_line(&self, refusals: &[(LayerId, String)]) -> Option<String> {
        let joined = refusals
            .iter()
            .filter_map(|(layer, reason)| {
                let name = self.project.layers.get(*layer)?.name.as_str();
                Some(format!("\u{201c}{name}\u{201d} is not drawing: {reason}"))
            })
            .collect::<Vec<_>>()
            .join(" \u{b7} ");
        (!joined.is_empty()).then_some(joined)
    }

    /// The opacity `layer` is recorded with, or fully opaque when the id names
    /// nothing.
    ///
    /// The value the shell pushes to the GPU every frame through
    /// [`crate::map_gpu::sync_tile_layer_opacities`] — cheap enough to be
    /// unconditional, because a tint costs no texture and no tessellation.
    /// Reading it here, from the project, is what makes the panel's slider mean
    /// the same thing for a COG, an archive, an MVT source and a dropped
    /// GeoJSON.
    #[must_use]
    pub fn tile_layer_opacity(&self, layer: LayerId) -> f32 {
        self.project.layers.get(layer).map_or(1.0, Layer::opacity)
    }

    /// Declares that this shell reconciles through the N-layer tile stack
    /// ([`Self::tile_stack_work`]) rather than through the two legacy
    /// single-slot seams.
    ///
    /// It is a **capability**, in the same class as
    /// [`Self::set_native_project_io`], and it changes exactly two answers, so
    /// that the f4 compositing rules are *enforced* rather than merely
    /// documented for a shell to remember:
    ///
    /// * [`Self::pending_raster_work`] offers a **basemap-only** plan. The
    ///   stack owns every COG and archive now, and a plan that still carried
    ///   them would make the shell CPU-blend the basemap into each layer tile
    ///   — after which the per-layer opacity tint fades the basemap too.
    /// * [`Self::pending_vector_work`] offers nothing at all, so the legacy
    ///   vector slot stays empty and cannot draw a duplicate of the top-most
    ///   vector entry of the stack.
    ///
    /// It deliberately does **not** touch [`Self::desired_raster`] or
    /// [`Self::desired_vector`]: those are the derivations the print snapshot
    /// and the credit line read, and they stay the same function of the project
    /// for every shell.
    pub fn set_tile_stack_shell(&mut self, enabled: bool) {
        self.tile_stack_shell = enabled;
    }

    /// Whether this shell declared itself a stack shell — see
    /// [`Self::set_tile_stack_shell`].
    #[must_use]
    pub fn tile_stack_shell(&self) -> bool {
        self.tile_stack_shell
    }

    /// The raster plan the *seam* compares against, which is not always the
    /// one [`Self::desired_raster`] describes.
    ///
    /// For a stack shell it is the basemap alone. Without this the seam would
    /// still derive `cog`/`archive` from the project, so adding a COG layer —
    /// which the stack installs as its own entry — would ALSO make
    /// [`Self::pending_raster_work`] answer `Some`, and the shell would call
    /// `replace_provider` and blank + re-fetch every visible basemap tile for a
    /// provider that came out byte-identical.
    ///
    /// Defined once because [`Self::pending_raster_work`] and
    /// [`Self::raster_refusal`] must compare against the same thing: a memo
    /// judged against a different plan than the one that produced it is either
    /// never shown or never cleared.
    fn desired_raster_work(&self) -> RasterWork {
        if self.tile_stack_shell {
            return RasterWork {
                basemap: self.drawn_basemap(),
                cog: None,
                archive: None,
            };
        }
        self.desired_raster()
    }

    /// The raster provider a shell still has to build, or [`None`] when what
    /// it last confirmed already matches. Stays [`Some`] until
    /// [`Self::settle_raster_work`] answers, so a frame with no render state
    /// DEFERS the install instead of losing it.
    #[must_use]
    pub fn pending_raster_work(&self) -> Option<RasterWork> {
        let desired = self.desired_raster_work();
        if self.raster_installed.as_ref() == Some(&desired) {
            return None;
        }
        if self.raster_refused.as_ref().map(|memo| &memo.work) == Some(&desired) {
            return None;
        }
        Some(desired)
    }

    /// The vector-slot change the project implies, ignoring any memoized
    /// refusal — the desire [`Self::pending_vector_work`] filters and
    /// [`Self::vector_refusal`] compares its memo against, defined once so the
    /// two cannot disagree about what is currently wanted.
    fn desired_vector_work(&self) -> Option<VectorWork> {
        if self.tile_stack_shell {
            // The stack draws every vector-tile layer, including the one the
            // legacy slot would have taken. Offering an install here would draw
            // the top-most entry twice — once tinted by the stack's opacity and
            // once not — and offering a detach would ask the shell to clear a
            // slot it never filled.
            return None;
        }
        match (self.desired_vector(), self.vector_installed.as_ref()) {
            (Some(desired), Some(installed)) if desired == *installed => None,
            (Some(desired), _) => Some(VectorWork::Install(desired)),
            (None, Some(_)) => Some(VectorWork::Detach),
            (None, None) => None,
        }
    }

    /// The vector-slot change a shell still has to run, or [`None`] when the
    /// slot already matches the project. Same offer-until-settled contract
    /// as [`Self::pending_raster_work`].
    #[must_use]
    pub fn pending_vector_work(&self) -> Option<VectorWork> {
        let work = self.desired_vector_work()?;
        if self.vector_refused.as_ref().map(|memo| &memo.work) == Some(&work) {
            return None;
        }
        Some(work)
    }

    /// Records what became of the raster work: `Ok` marks it installed in
    /// the GPU-state mirror, `Err(reason)` memoizes the refusal (no
    /// per-frame rebuild spin) and says so on the status line.
    ///
    /// The mirror is **GPU state, never project state**: `load_project` and
    /// `new_project` must not reset it, or a project load whose basemap
    /// equals the active one would rebuild the provider and blank + re-fetch
    /// every visible tile for nothing.
    pub fn settle_raster_work(&mut self, work: RasterWork, outcome: Result<(), String>) {
        match outcome {
            Ok(()) => {
                self.raster_installed = Some(work);
                self.raster_refused = None;
            }
            Err(reason) => {
                self.status = Some(format!("The map's tile source was not installed: {reason}"));
                self.raster_refused = Some(Refusal { work, reason });
            }
        }
    }

    /// The vector twin of [`Self::settle_raster_work`]. A settled
    /// [`VectorWork::Detach`] records the slot as confirmed-empty.
    pub fn settle_vector_work(&mut self, work: VectorWork, outcome: Result<(), String>) {
        match outcome {
            Ok(()) => {
                self.vector_installed = match work {
                    VectorWork::Install(config) => Some(config),
                    VectorWork::Detach => None,
                };
                self.vector_refused = None;
            }
            Err(reason) => {
                self.status = Some(format!("The vector-tile layer was not installed: {reason}"));
                self.vector_refused = Some(Refusal { work, reason });
            }
        }
    }

    /// Why the raster slot is not what the project implies, when a shell
    /// refused the plan that is **still** outstanding.
    ///
    /// Derived, never stored: a memo whose plan the desire has moved past is
    /// stale by construction and reports nothing, so a banner can never badge
    /// a map that is drawing exactly what was asked for.
    #[must_use]
    pub fn raster_refusal(&self) -> Option<&str> {
        let memo = self.raster_refused.as_ref()?;
        (memo.work == self.desired_raster_work()).then_some(memo.reason.as_str())
    }

    /// The vector twin of [`Self::raster_refusal`], compared against the
    /// slot change the project currently implies for the same staleness
    /// reason.
    #[must_use]
    pub fn vector_refusal(&self) -> Option<&str> {
        let memo = self.vector_refused.as_ref()?;
        (self.desired_vector_work().as_ref() == Some(&memo.work)).then_some(memo.reason.as_str())
    }

    /// Why the GPU map itself is not attached, when an install was tried and
    /// failed.
    ///
    /// A latch, not a derivation — unlike a provider plan there is nothing to
    /// compare it against, because the *only* thing that can clear it is
    /// another attempt. [`Self::retry_refused_installs`] is that attempt's
    /// trigger.
    #[must_use]
    pub fn map_attach_refusal(&self) -> Option<&str> {
        self.map_gpu_failed.as_deref()
    }

    /// Reports what the drawn stack currently refuses
    /// ([`crate::map_gpu::tile_layer_refusals`]) so the banner can name it.
    ///
    /// Called once per frame by a stack shell, because those refusals are GPU
    /// state and only a shell holds the `RenderState` to read them from. This
    /// is a **report cache, not a plan mirror**: it is overwritten from the map
    /// every frame and nothing reconciles against it, so it cannot go stale in
    /// the way a second copy of "what is installed" would (see this module's
    /// "the mirror is the map" rule). Refusals naming a layer the project no
    /// longer holds are dropped by [`Self::tile_layer_refusal_line`] anyway.
    pub fn set_tile_layer_refusals(&mut self, refusals: Vec<(LayerId, String)>) {
        self.tile_layer_refusals = refusals;
    }

    /// Take-once: whether the banner's Retry was pressed since the shell last
    /// asked, so a stack shell knows to call
    /// [`crate::map_gpu::retry_refused_tile_layers`] as well.
    ///
    /// The app half of the same click runs immediately
    /// ([`Self::retry_refused_installs`]); only the GPU half needs a seam.
    pub fn take_tile_layer_retry(&mut self) -> bool {
        core::mem::take(&mut self.pending_tile_layer_retry)
    }

    /// The ONE banner builder — the GPU-map latch, both single slots' live
    /// refusals and the drawn stack's, joined with the `credit_line`
    /// separator, so no shell can word the same condition differently.
    /// [`None`] means the map is attached and every outstanding plan is either
    /// installed or still waiting for a shell.
    #[must_use]
    pub fn provider_refusal(&self) -> Option<String> {
        let attach = self
            .map_attach_refusal()
            .map(|reason| format!("The GPU map could not be attached: {reason}"));
        // The stack's half, joined with the same separator, so a refused stack
        // entry is worded exactly like a basemap that could not be built — and,
        // crucially, is visible at all: without this a refused entry is a layer
        // row with a ticked checkbox drawing nothing and no explanation
        // anywhere on screen.
        let stack = self.tile_layer_refusal_line(&self.tile_layer_refusals);
        let joined = [
            attach.as_deref(),
            self.raster_refusal(),
            self.vector_refusal(),
            stack.as_deref(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" \u{b7} ");
        (!joined.is_empty()).then_some(joined)
    }

    /// Drops both refusal memos so the next frame offers the refused plans
    /// again, and reports whether anything the user could see was cleared.
    ///
    /// A **command, not an edit**: it writes no project state (the memos are
    /// the `raster_installed` field class — GPU state, never project state),
    /// its inverse cannot be built from recorded data, and its effect is
    /// decided by the next settle rather than by the click — so it touches
    /// neither the undo stack nor the coalescing window, and an interrupted
    /// opacity drag still folds. The *installed* mirrors are deliberately
    /// left alone: clearing those would blank and re-fetch every visible tile.
    pub fn retry_refused_installs(&mut self) -> bool {
        let visible = self.provider_refusal().is_some();
        self.clear_refused_installs();
        visible
    }

    /// Forgets both refusal memos **and** the GPU-map attach latch. Called by
    /// [`Self::retry_refused_installs`] and by the GPU-map attach seam, where
    /// "the map is not attached" — the reason both shells settle with when
    /// `replace_provider` finds no map — is provably no longer true.
    ///
    /// Dropping the attach latch here is what makes Retry mean something for
    /// it: the next frame's [`Self::attach_gpu_map_with`] sees no latch and
    /// tries the install again. Without that, the banner would carry a button
    /// that could never dismiss it.
    pub(super) fn clear_refused_installs(&mut self) {
        self.raster_refused = None;
        self.vector_refused = None;
        self.map_gpu_failed = None;
    }

    /// The basemap credit the map actually owes, borrowed — the promoted
    /// layer's when one resolves, else the service's. The borrowed twin of
    /// `drawn_basemap().attribution`.
    fn drawn_credit(&self) -> &str {
        // The XYZ arm is the only one `promoted_basemap_layer` can answer; the
        // fallback is "no layer is promoted", not "a promoted layer of another
        // kind".
        match self.promoted_basemap_layer().map(|layer| &layer.kind) {
            Some(LayerKind::Raster(RasterSource::Xyz { attribution, .. })) => attribution.as_str(),
            _ => self.basemap.attribution.as_str(),
        }
    }

    /// The raster *layer* credit, borrowed — same scan and same "top-most
    /// visible wins across the kinds" rule as [`Self::desired_raster`], minus
    /// the configs it builds to answer a different question.
    ///
    /// A COG contributes nothing ([`CogLayerConfig::new`] carries no credit and
    /// nothing sets one), so the whole answer is an archive's own metadata
    /// credit — and only when the archive is one this build can read, which is
    /// exactly the layer `desired_raster` would have offered a shell.
    fn raster_credit(&self) -> &str {
        // The very predicate `desired_raster` culls with: the credit line and
        // the plan it summarises must agree about which layer is drawing, and a
        // scale range honoured by one and not the other is exactly how they
        // would drift (pinned by the credit-line agreement test in
        // `app::tests_providers`).
        let zoom = self.drawn_zoom();
        self.project
            .layers
            .layers()
            .iter()
            .rev()
            .filter(|layer| layer.visible_at(zoom))
            .find_map(|layer| match &layer.kind {
                LayerKind::Raster(RasterSource::Cog { .. }) => Some(""),
                LayerKind::Raster(RasterSource::TileArchive {
                    archive,
                    format,
                    attribution,
                }) => oxigis_core::archive_refusal(archive, *format)
                    .is_none()
                    .then_some(attribution.as_str()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// The vector-tile credit, borrowed — the twin of [`Self::raster_credit`]
    /// over [`Self::desired_vector`]'s scan.
    ///
    /// The MVT arm reproduces the one rule
    /// [`vector_provider::config_for`] applies (the keyless demo source credits
    /// MapLibre, anything else credits nobody) rather than building a whole
    /// [`VectorTileConfig`] to read one field off it — `config_for` allocates a
    /// five-entry demo paint table it immediately discards, twice a frame, for
    /// a string that is almost always empty. The two are pinned to each other
    /// by the credit-line agreement test in `app::tests_providers`.
    fn vector_credit(&self) -> &str {
        // Same cull as `desired_vector`, for the same agreement reason as
        // [`Self::raster_credit`].
        let zoom = self.drawn_zoom();
        self.project
            .layers
            .layers()
            .iter()
            .rev()
            .filter(|layer| layer.visible_at(zoom))
            .find_map(|layer| match &layer.kind {
                LayerKind::Vector(VectorSource::MvtTiles { url_template, .. }) => Some(
                    if url_template == vector_provider::MAPLIBRE_DEMO_URL_TEMPLATE {
                        vector_provider::MAPLIBRE_ATTRIBUTION
                    } else {
                        ""
                    },
                ),
                LayerKind::Vector(VectorSource::TileArchive {
                    archive,
                    format,
                    attribution,
                    ..
                }) => oxigis_core::archive_refusal(archive, *format)
                    .is_none()
                    .then_some(attribution.as_str()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Credit line the current raster *layer* contributes, if any — COG or
    /// tile archive, derived from the project, so it cannot outlive its layer.
    #[must_use]
    pub fn cog_attribution(&self) -> String {
        self.raster_credit().to_owned()
    }

    /// Credit line the current vector-tile layer contributes, if any —
    /// derived from the project, so it cannot outlive its layer.
    #[must_use]
    pub fn vector_attribution(&self) -> String {
        self.vector_credit().to_owned()
    }

    /// The one credit line the map owes — the ONE builder the painted
    /// attribution and the exported page both read, so they cannot drift.
    ///
    /// A promoted layer's credit **replaces** the service's, because the
    /// service is not on screen: crediting tiles that are not drawn is as
    /// wrong as omitting the credit for tiles that are.
    ///
    /// Runs once per frame under the attribution painter, so it derives from
    /// **borrows**: the config-building derivations it used to call
    /// (`drawn_basemap`, `desired_raster`, `desired_vector`) clone a basemap,
    /// a layer config and a whole paint table between them, all to read three
    /// strings.
    #[must_use]
    pub(super) fn credit_line(&self) -> String {
        [
            self.drawn_credit(),
            self.raster_credit(),
            self.vector_credit(),
        ]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" \u{b7} ")
    }
}
