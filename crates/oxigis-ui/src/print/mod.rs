// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Print layout and PDF export (blueprint §7 Phase 3): one page — A4
//! landscape by default, size/orientation/DPI selectable via
//! [`PrintOptions`] — with a raster basemap, a **true vector** overlay of the
//! local layers, a title, the attribution line, and (print v1.7) the
//! cartographic furniture a printed sheet is read with: a segmented scale bar
//! carrying its representative fraction, a north arrow and a legend.
//!
//! # Shape of the module
//!
//! Everything here is wasm-clean: layout maths, pixel composition and PDF
//! assembly, with no I/O, no egui and no GPU. It is pure too, with ONE
//! documented exception — `/CreationDate` reads the system clock on native
//! targets (never on wasm32, where `std` has none), and
//! [`PrintOptions::creation_epoch_secs`] overrides even that, which is what
//! makes an export reproducible. A shell drives the whole
//! export through four calls — [`compose_view`] reframes the camera snapshot
//! to the page's map box, [`required_tiles`] says which tiles to fetch,
//! [`compose_map_rgb`] pastes whatever arrived into one RGB buffer, and
//! [`pdf_document`] assembles the page. The shell owns the only impure parts:
//! building a tile provider, waiting for the fetches, and writing (desktop) or
//! downloading (web) the bytes.
//!
//! A project drawing MORE than the legacy single-slot fields describe — see
//! [`PrintRequest::stack_fits_legacy_slots`] — adds two calls to that sequence:
//! [`overlay_map_rgb`] composites each further raster layer onto the buffer at
//! its own opacity, and [`pdf_document_with`] takes one decoded-tile list per
//! vector-tile layer instead of one for `PrintRequest::vector`. Every other
//! project takes the four-call path unchanged, to the byte.
//!
//! # Why `pdf-writer` + `oxiarc-deflate`
//!
//! Every higher-level PDF crate on crates.io drags `flate2` or `miniz_oxide`
//! into the graph, which `deny.toml` bans (audited 2026-07-31; see the
//! workspace manifest). `pdf-writer` is a pure serializer with no compression
//! of its own — its streams take already-encoded bytes — so the zlib streams
//! `/FlateDecode` needs are produced by `oxiarc-deflate`, the same COOLJAPAN
//! inflater the tile pipeline already depends on.
//!
//! # Scope, stated once (v1.2)
//!
//! * The basemap (and any COG composite) is embedded as one **raster** image
//!   at [`RASTER_PX_PER_PT`]; tiles that never arrive print as neutral gray.
//! * Local vector layers, streamed MVT vector-tile layers and labels
//!   ([`LayerStyle::Symbol`] layers) are drawn as **real PDF paths/text** —
//!   sharp at any print zoom, z-ordered raster → MVT → local → labels. A
//!   Symbol style asking for VERTICAL text is set as a stacked column since
//!   v1.6, so the page and the map agree about orientation too.
//! * Text goes through embedded `/Type0` fonts (the `font` module),
//!   **shaped** since v1.2 (`shape`): kerning and ligatures land in `TJ`
//!   arrays, `/W` stays
//!   `hmtx`, `/ToUnicode` stays exact. RTL and n:m complex scripts keep the
//!   v1.1 per-character output; with no usable font chain the page degrades
//!   to Base-14 Helvetica + WinAnsi (`?` for CJK) — see [`win_ansi`].
//! * The furniture (v1.7) claims the map box's own corners — bar bottom-left
//!   (`scalebar`), legend bottom-right (`legend`), arrow top-right (`north`)
//!   — rather than a reserved band: a band would move [`map_box`], and with
//!   it the raster size and every existing page's framing. Each piece is
//!   switchable from [`PrintOptions`] and ON by default, and each yields its
//!   corner when the geometry does not fit it. The document's `/Info`
//!   metadata is the `meta` module.
//! * Still out of scope, deliberately: a MULTI-PAGE ATLAS (the page tree is
//!   one kid wide) and free layout templates — both need `PrintRequest` to
//!   carry a per-page view list, which is a separate design. So is a raster
//!   layer drawn ABOVE a vector-tile one: rasters are composited into the one
//!   embedded image and vectors are paths over it, so that arrangement still
//!   flattens to the page's documented order. Closing it means one image
//!   XObject per raster run, each with its own soft mask.

mod bidi;
mod cff;
mod document;
mod emit;
mod font;
mod instance;
mod labels;
mod legend;
mod meta;
mod mirror_table;
mod mvt;
mod north;
mod paint;
mod scalebar;
mod shape;
mod subset;
mod vertical;

use emit::{TextMark, WeightedText, show_line, show_line_marked, show_vertical_line};

pub use document::{pdf_document, pdf_document_with};
pub use emit::win_ansi;
pub use font::{FaceRole, PrintFace, PrintFonts, TextPlan};
pub use scalebar::{ScaleBar, ScaleUnits, scale_bar, scale_bar_with};

use crate::cog_provider::CogLayerConfig;
use crate::edit::command::PathKind;
use crate::tile_provider::BasemapConfig;
use crate::vector_provider::VectorTileConfig;
use oxigeo::geojson::types::{FeatureCollection, Position};
use oxigis_core::{Color, LayerStyle};
use oxigis_render::mvt::VectorTile as MvtVectorTile;
use oxigis_render::{DecodedTile, LonLat, MapView, TileId};
use pdf_writer::types::{LineJoinStyle, TextRenderingMode};
use pdf_writer::{Content, Name};
use std::sync::Arc;

/// A4 landscape, in PostScript points: `[width, height]` — the default page,
/// kept as a named constant because the layout tests reason in it.
pub const A4_LANDSCAPE_PT: [f32; 2] = [841.89, 595.28];

/// Selectable page sizes for the export dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PageSize {
    /// ISO A4: 210 × 297 mm.
    #[default]
    A4,
    /// ISO A3: 297 × 420 mm.
    A3,
    /// US Letter: 8.5 × 11 in.
    Letter,
}

impl PageSize {
    /// Every size, in dialog order.
    pub const ALL: [Self; 3] = [Self::A4, Self::A3, Self::Letter];

    /// Portrait dimensions in points, `[width, height]`.
    #[must_use]
    pub fn portrait_pt(self) -> [f32; 2] {
        match self {
            Self::A4 => [595.28, 841.89],
            Self::A3 => [841.89, 1190.55],
            Self::Letter => [612.0, 792.0],
        }
    }

    /// Dialog label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::A4 => "A4",
            Self::A3 => "A3",
            Self::Letter => "Letter",
        }
    }
}

/// Page orientation for the export dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PageOrientation {
    /// Wider than tall — the default, matching the usual map aspect.
    #[default]
    Landscape,
    /// Taller than wide.
    Portrait,
}

impl PageOrientation {
    /// Both orientations, in dialog order.
    pub const ALL: [Self; 2] = [Self::Landscape, Self::Portrait];

    /// Dialog label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Landscape => "Landscape",
            Self::Portrait => "Portrait",
        }
    }
}

/// What the Export-PDF dialog collects: page geometry and raster resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrintOptions {
    /// The paper size.
    pub page: PageSize,
    /// Which way the paper turns.
    pub orientation: PageOrientation,
    /// Raster resolution of the embedded map image, in physical pixels per
    /// point (DPI = 72 × this).
    pub raster_px_per_pt: f32,
    /// Set the page TITLE vertically, top to bottom (print/text v1.4,
    /// D-V1). Off by default, and honoured only when the title is CJK
    /// enough for the export's own refusal ladder to accept it — a
    /// refusal, or this being `false`, prints the title horizontally, byte
    /// for byte as every previous version did.
    ///
    /// Not serialized: [`PrintOptions`] is dialog state, never persisted
    /// (verified — the struct carries no serde derive and `Project` holds no
    /// print options), so there is no skip-if-default to write.
    pub vertical_title: bool,
    /// Draw the SCALE BAR in the map box's bottom-left corner (print v1.7).
    ///
    /// On by default — a map without a scale is not a map — but a sheet that
    /// carries its scale elsewhere (a series frame, a report caption) may
    /// turn it off, and every page then keeps the corner clear.
    pub scale_bar: bool,
    /// Which units the scale bar counts in.
    pub scale_units: ScaleUnits,
    /// Print the REPRESENTATIVE FRACTION — `1:25 000` — under the bar.
    pub representative_fraction: bool,
    /// Draw the NORTH ARROW in the map box's top-right corner.
    pub north_arrow: bool,
    /// Draw the LEGEND — one row per visible local vector layer — in the map
    /// box's bottom-right corner.
    pub legend: bool,
    /// Write the document's `/Info` dictionary (title, producer, creation
    /// date) and `/MarkInfo`.
    pub document_metadata: bool,
    /// The instant to stamp as `/CreationDate`, in Unix seconds — [`None`]
    /// reads the system clock.
    ///
    /// Two callers need it. `wasm32-unknown-unknown` has NO clock in `std`
    /// (`SystemTime::now()` panics there), so the web shell stamps
    /// `Date::now() / 1000` itself; and a test, or any caller that wants two
    /// runs of the same export to produce the same bytes, pins the second
    /// here rather than letting the wall clock into the output.
    pub creation_epoch_secs: Option<i64>,
    /// Allow the embedded map raster to encode as `/DCTDecode` (JPEG)
    /// instead of `/FlateDecode` (zlib) when doing so actually shrinks it
    /// (print v1.8).
    ///
    /// On by default: a photographic basemap tile — the common case — has
    /// enough pixel entropy that lossless zlib barely compresses it, while
    /// JPEG at [`Self::jpeg_quality`] shrinks it by a wide margin at a
    /// quality loss a printed page does not show. [`pdf_document`] always
    /// races the two encodings and keeps whichever comes out smaller, so
    /// turning this on can never grow a page — a flat, line-art-like or
    /// screenshot-like raster (few colours, hard edges) simply loses the
    /// race and the page comes out exactly as it would with this off. Turn
    /// it off to force `/FlateDecode` unconditionally: a lossless archival
    /// export, or a print pipeline downstream of this one that assumes
    /// every image stream is zlib.
    pub photo_jpeg: bool,
    /// JPEG quality for the [`Self::photo_jpeg`] path, `1..=100` (libjpeg
    /// convention: 1 worst, 100 best). Clamped to that range wherever it is
    /// used, so an out-of-range value from a corrupt settings file degrades
    /// to the nearest legal quality rather than panicking.
    pub jpeg_quality: u8,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            page: PageSize::A4,
            orientation: PageOrientation::Landscape,
            raster_px_per_pt: RASTER_PX_PER_PT,
            vertical_title: false,
            scale_bar: true,
            scale_units: ScaleUnits::Metric,
            representative_fraction: true,
            north_arrow: true,
            legend: true,
            document_metadata: true,
            creation_epoch_secs: None,
            photo_jpeg: true,
            jpeg_quality: DEFAULT_JPEG_QUALITY,
        }
    }
}

impl PrintOptions {
    /// Page dimensions in points, orientation applied: `[width, height]`.
    #[must_use]
    pub fn page_size_pt(&self) -> [f32; 2] {
        let [w, h] = self.page.portrait_pt();
        match self.orientation {
            PageOrientation::Portrait => [w, h],
            PageOrientation::Landscape => [h, w],
        }
    }

    /// The DPI the raster map is embedded at, as an exact number.
    ///
    /// The dialog labels through [`dpi_label`] instead, which ROUNDS: the
    /// 300 dpi choice is `25.0 / 6.0` px/pt and reads as `299.99998` here.
    /// This is the value to reason in — the layout tests do.
    #[must_use]
    pub fn dpi(&self) -> f32 {
        self.raster_px_per_pt * 72.0
    }
}

/// Page margin on every side, in points.
pub const PAGE_MARGIN_PT: f32 = 36.0;

/// Height reserved above the map box for the title, in points.
const TITLE_BAND_PT: f32 = 30.0;

/// Height reserved below the map box for the attribution line, in points.
const FOOTER_BAND_PT: f32 = 14.0;

/// Gap between the map box and each band, in points.
const BAND_GAP_PT: f32 = 6.0;

/// Default raster resolution of the embedded map image, in physical pixels
/// per point (`2.0` = 144 DPI — crisp enough for office printing, small
/// enough that the zlib-compressed image stays in single-digit megabytes).
/// [`PrintOptions::raster_px_per_pt`] can raise it.
pub const RASTER_PX_PER_PT: f32 = 2.0;

/// The raster resolutions the export dialog offers, in physical pixels per
/// point: 144, 216, 288 and **300 dpi**.
///
/// 300 dpi is the number a print shop asks for, and `25.0 / 6.0` is exactly
/// `300 / 72`; a dialog that stopped at 288 could not reach it at all. The
/// list is here rather than in the shell so the ceiling
/// ([`MAX_RASTER_PX_PER_PT`]) and the choices cannot drift apart.
pub const RASTER_PX_PER_PT_CHOICES: [f32; 4] = [2.0, 3.0, 4.0, 25.0 / 6.0];

/// One resolution's dialog label, rounded: `25.0 / 6.0` reads as `300 dpi`
/// rather than as the `299.99998` its f32 product prints.
#[must_use]
pub fn dpi_label(px_per_pt: f32) -> String {
    format!("{} dpi", (px_per_pt * 72.0).round())
}

/// Lowest legal [`PrintOptions::jpeg_quality`] (libjpeg convention: `1` is
/// the worst quality, not `0`).
pub const MIN_JPEG_QUALITY: u8 = 1;

/// Highest legal [`PrintOptions::jpeg_quality`].
pub const MAX_JPEG_QUALITY: u8 = 100;

/// Default [`PrintOptions::jpeg_quality`] — high enough that a printed
/// photographic basemap shows no visible blocking, low enough that the
/// encoded stream stays a fraction of the size the `/FlateDecode` path
/// would need for the same pixels.
pub const DEFAULT_JPEG_QUALITY: u8 = 85;

/// Title font size, in points.
const TITLE_FONT_PT: f32 = 16.0;

/// Attribution font size, in points.
const FOOTER_FONT_PT: f32 = 8.0;

/// Scale-bar label font size, in points.
const SCALE_BAR_FONT_PT: f32 = 8.0;

/// Scale-bar bar height, in points.
const SCALE_BAR_HEIGHT_PT: f32 = 4.0;

/// Inset of the scale bar from the map box's bottom-left corner, in points.
const SCALE_BAR_INSET_PT: f32 = 10.0;

/// The gray every pixel starts as, so a tile that never arrived prints as a
/// visibly neutral plate rather than as black.
pub const MISSING_TILE_GRAY: u8 = 0xE0;

/// One local vector layer to print, bottom-up in stack order.
pub struct PrintLayer {
    /// The layer's display name, for the legend row.
    ///
    /// The project's name for the layer — what the layer panel shows — which
    /// is the name the reader of the page recognises. Empty falls back to the
    /// collection's own GeoJSON `name` member and then to `Layer N`, so a
    /// fixture that carries no project name legends exactly as it always did.
    pub name: String,
    /// The layer's parsed features, shared with the app's store.
    pub features: Arc<FeatureCollection>,
    /// The style SET the map draws it with (tiles v1.3: base plus
    /// per-family overrides — the page paints exactly what the screen
    /// resolves).
    pub style: oxigis_core::LayerStyleSet,
    /// The geometry families the dataset actually draws. One `q…Q` block is
    /// emitted per PRESENT family, so a single-family layer emits exactly
    /// one — byte-identical operators to the pre-v1.3 output.
    pub families: oxigis_core::FamilySet,
    /// The layer's own opacity multiplier, `0.0..=1.0`.
    pub opacity: f32,
}

/// One tiled layer of the exported page: which project layer it is, where its
/// tiles come from, and how faded it is drawn.
///
/// The page's twin of `crate::app::providers::TileLayerPlan`, plus the opacity
/// — which the plan deliberately omits (it is an instance tint on screen, not
/// part of a source's identity) but which a page has to bake in, because a PDF
/// image stream carries no slider.
#[derive(Debug, Clone, PartialEq)]
pub struct PrintTileLayer {
    /// The project layer this entry draws.
    pub layer: oxigis_core::LayerId,
    /// Where its tiles come from — the same union the live map builds a
    /// provider from, so the page and the screen cannot disagree about what a
    /// layer *is*.
    ///
    /// Read from [`crate::layer_source`], which is neutral ground: naming it
    /// through `app::providers` (where it used to live) made this module depend
    /// on the application shell, contradicting the "no I/O, no egui and no GPU
    /// — extractable" contract stated at the top of this file.
    pub source: crate::layer_source::TileLayerSource,
    /// The alpha the map draws it at, `0.0..=1.0`.
    pub opacity: f32,
}

/// Everything one export needs, captured at the moment the user asked.
///
/// A snapshot rather than borrows of the app: the desktop shell fetches tiles
/// for seconds and the web shell exports inside an async task, and neither may
/// hold the UI's state hostage while they do.
pub struct PrintRequest {
    /// Page title — the project name.
    pub title: String,
    /// Combined credit line: basemap, plus COG and vector-tile credits if any.
    pub attribution: String,
    /// The camera exactly as the map panel showed it.
    pub view: MapView,
    /// The active basemap, for the shell to build a fresh provider from.
    pub basemap: BasemapConfig,
    /// The active COG layer, when one is composited over the basemap.
    pub cog: Option<CogLayerConfig>,
    /// The active raster tile archive, when one is composited over the
    /// basemap. Never [`Some`] at the same time as `cog`: the project's
    /// top-most visible raster layer is one or the other.
    pub archive: Option<crate::archive::ArchiveLayerConfig>,
    /// The active streamed vector-tile source, for the shell to fetch and
    /// decode the page's tiles from.
    pub vector: Option<VectorTileConfig>,
    /// EVERY tiled layer the map draws, bottom-up, with the alpha each is
    /// drawn at — the N-layer snapshot that generalises `cog` / `archive` /
    /// `vector` above.
    ///
    /// Those three fields describe at most one raster layer and one
    /// vector-tile layer, because that is all the map itself could draw before
    /// compositing v1.6. A project holding an orthophoto under a hillshade now
    /// draws both on screen, so an export that read only the three would
    /// silently print a different map from the one being exported.
    ///
    /// The three fields are kept, and are still what the shells compose from
    /// today: they are the same top-most layers this list ends with, so an
    /// export is unchanged for every project that has one raster and one
    /// vector-tile layer. Composing the whole list onto the page is the
    /// remaining half — see [`PrintTileLayer`].
    pub stack: Vec<PrintTileLayer>,
    /// Visible local vector layers, bottom-up in stack order.
    pub layers: Vec<PrintLayer>,
    /// Page geometry and raster resolution, as the export dialog set them.
    pub options: PrintOptions,
}

impl PrintRequest {
    /// Whether the legacy single-slot fields (`cog` / `archive` / `vector`)
    /// already describe everything [`Self::stack`] holds.
    ///
    /// True for every project the map could draw before compositing v1.6 — at
    /// most one raster layer and at most one vector-tile layer, neither of them
    /// an XYZ overlay, which the three fields cannot name at all. Those exports
    /// keep the exact path (and the exact bytes) they always had; only a stack
    /// the three fields would silently truncate takes the composed one.
    #[must_use]
    pub fn stack_fits_legacy_slots(&self) -> bool {
        let mut rasters = 0_usize;
        let mut vectors = 0_usize;
        for entry in &self.stack {
            match &entry.source {
                crate::layer_source::TileLayerSource::Cog(_)
                | crate::layer_source::TileLayerSource::RasterArchive(_) => rasters += 1,
                // An XYZ layer that is not the promoted basemap has no legacy
                // field at all: `cog` and `archive` cannot hold one, so the
                // single-slot path would print the map without it.
                crate::layer_source::TileLayerSource::Xyz(_) => return false,
                crate::layer_source::TileLayerSource::Vector(_) => vectors += 1,
            }
        }
        if rasters > 1 || vectors > 1 {
            return false;
        }
        // Counting is not enough: the legacy path prints what the FIELDS name,
        // so an entry the fields do not name is an entry the page would lose.
        // The two derivations are built by different scans — the stack's is
        // capped at `MAX_DRAWN_TILE_LAYERS` and the fields' picks a single
        // top-most layer — so requiring them to agree here is what makes "the
        // legacy path is lossless" true by construction rather than by
        // inspection of the app. A disagreement can only ever move a project
        // from the truncating path to the composing one.
        if vectors == 1 && self.vector.is_none() {
            return false;
        }
        if rasters == 1 && self.cog.is_none() && self.archive.is_none() {
            return false;
        }
        true
    }

    /// The raster entries of [`Self::stack`], bottom-up — the passes a shell
    /// composites over the basemap with [`overlay_map_rgb`].
    #[must_use]
    pub fn raster_stack(&self) -> Vec<&PrintTileLayer> {
        self.stack
            .iter()
            .filter(|entry| entry.source.is_raster())
            .collect()
    }

    /// The vector-tile configurations the page paints, bottom-up, each with the
    /// stack position that names its alpha states.
    ///
    /// On the legacy path that is `vector` alone, tagged [`None`] so its
    /// ExtGState names stay `GV0`, `GV1`, … exactly as every existing export
    /// wrote them. On the composed path it is every `Vector` entry of the
    /// stack, tagged with its position.
    #[must_use]
    fn vector_sources(&self) -> Vec<(Option<usize>, &VectorTileConfig)> {
        if self.stack_fits_legacy_slots() {
            return self
                .vector
                .as_ref()
                .map(|config| vec![(None, config)])
                .unwrap_or_default();
        }
        self.stack
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| match &entry.source {
                crate::layer_source::TileLayerSource::Vector(config) => Some((Some(index), config)),
                _ => None,
            })
            .collect()
    }
}

/// The decoded vector tiles one export fetched, per source.
///
/// Two lists rather than one because the two paths name their sources
/// differently: `single` is [`PrintRequest::vector`]'s tiles — the legacy slot,
/// unchanged — and `stack` holds one list per [`PrintRequest::stack`] entry, in
/// stack order, with a raster entry's list simply empty. A shell fills whichever
/// the request asks for; filling neither prints the page with no streamed
/// vector tiles at all, which is what a shell that could not fetch any does.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrintVectorTiles<'a> {
    /// Tiles for [`PrintRequest::vector`].
    pub single: &'a [(TileId, Arc<MvtVectorTile>)],
    /// Tiles per [`PrintRequest::stack`] entry, in stack order. A list shorter
    /// than the stack is legal — the entries past its end have no tiles.
    pub stack: &'a [Vec<(TileId, Arc<MvtVectorTile>)>],
}

impl<'a> PrintVectorTiles<'a> {
    /// The tiles belonging to the source `vector_sources` tagged `entry`.
    fn of(&self, entry: Option<usize>) -> &'a [(TileId, Arc<MvtVectorTile>)] {
        match entry {
            None => self.single,
            Some(index) => self.stack.get(index).map_or(&[][..], Vec::as_slice),
        }
    }
}

/// The page rectangle the map occupies, in PDF coordinates (origin at the
/// page's bottom-left corner, y up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapBox {
    /// Left edge, in points.
    pub x: f32,
    /// Bottom edge, in points.
    pub y: f32,
    /// Width, in points.
    pub width: f32,
    /// Height, in points.
    pub height: f32,
}

/// The four numbers the page layout subtracts from the paper — margins and
/// band heights — as one value the export dialog can edit.
///
/// [`Default`] is the shipped constants ([`PAGE_MARGIN_PT`] and friends), so
/// [`map_box`] is exactly `map_box_with(options, PageGeometry::default())` and
/// every existing export is unchanged to the byte. A production map export is
/// expected to let the user set margins — bleed, a binding edge, a plotter's
/// hardware margin — which is what this type exists for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageGeometry {
    /// Margin on every side, in points.
    pub margin_pt: f32,
    /// Height reserved above the map box for the title, in points.
    pub title_band_pt: f32,
    /// Height reserved below the map box for the attribution line, in points.
    pub footer_band_pt: f32,
    /// Gap between the map box and each band, in points.
    pub band_gap_pt: f32,
}

impl Default for PageGeometry {
    fn default() -> Self {
        Self {
            margin_pt: PAGE_MARGIN_PT,
            title_band_pt: TITLE_BAND_PT,
            footer_band_pt: FOOTER_BAND_PT,
            band_gap_pt: BAND_GAP_PT,
        }
    }
}

/// Smallest map box either axis may be reduced to, in points — an inch of
/// map. Whatever the geometry asks for, the page keeps a map on it.
const MIN_MAP_BOX_PT: f32 = 72.0;

/// The map box for the options' page: the page minus margins, title band and
/// footer.
#[must_use]
pub fn map_box(options: &PrintOptions) -> MapBox {
    map_box_with(options, PageGeometry::default())
}

/// [`map_box`] with an explicit [`PageGeometry`].
///
/// The geometry is sanitised here rather than at its edit site, because this
/// is the ONE call every consumer of the map rectangle goes through: a
/// non-finite or negative number falls back to the default, and a set of bands
/// that would leave less than `MIN_MAP_BOX_PT` on an axis is scaled down
/// proportionally instead of producing a negative rectangle.
#[must_use]
pub fn map_box_with(options: &PrintOptions, geometry: PageGeometry) -> MapBox {
    let [page_w, page_h] = options.page_size_pt();
    let default = PageGeometry::default();
    let sane = |value: f32, fallback: f32| {
        if value.is_finite() && value >= 0.0 {
            value
        } else {
            fallback
        }
    };
    let mut margin = sane(geometry.margin_pt, default.margin_pt);
    let mut title = sane(geometry.title_band_pt, default.title_band_pt);
    let mut footer = sane(geometry.footer_band_pt, default.footer_band_pt);
    let mut gap = sane(geometry.band_gap_pt, default.band_gap_pt);
    // Horizontal: only the margins eat width.
    let horizontal = 2.0 * margin;
    let horizontal_room = (page_w - MIN_MAP_BOX_PT).max(0.0);
    if horizontal > horizontal_room {
        margin *= horizontal_room / horizontal.max(f32::MIN_POSITIVE);
    }
    // Vertical: both margins, both bands and both gaps.
    let vertical = 2.0 * margin + title + footer + 2.0 * gap;
    let vertical_room = (page_h - MIN_MAP_BOX_PT).max(0.0);
    if vertical > vertical_room {
        let scale = vertical_room / vertical.max(f32::MIN_POSITIVE);
        margin *= scale;
        title *= scale;
        footer *= scale;
        gap *= scale;
    }
    let x = margin;
    let y = margin + footer + gap;
    MapBox {
        x,
        y,
        width: (page_w - 2.0 * margin).max(0.0),
        height: (page_h - margin - title - gap - y).max(0.0),
    }
}

/// The raster size of the embedded map image, in physical pixels.
///
/// **The export's one allocation gate.** Both shells reach the raster through
/// exactly this call — it sizes the buffer [`compose_map_rgb`] fills, the view
/// [`compose_view`] reframes and the length [`pdf_document`] validates — so
/// the two ceilings below are enforced here and nowhere else:
/// [`MAX_RASTER_PX_PER_PT`] on the resolution, and [`max_raster_pixels`] on
/// the product. [`PrintOptions`] is a public struct with public fields, so a
/// shell, a test harness or a future CLI can ask for any number; a clamped
/// page with a log beats a multi-gigabyte `Vec` and an out-of-memory abort.
#[must_use]
pub fn raster_size_px(map_box: &MapBox, options: &PrintOptions) -> [u32; 2] {
    let requested = if options.raster_px_per_pt.is_finite() && options.raster_px_per_pt > 0.0 {
        options.raster_px_per_pt
    } else {
        RASTER_PX_PER_PT
    };
    let mut px_per_pt = requested.min(MAX_RASTER_PX_PER_PT);
    let width_pt = f64::from(map_box.width.max(0.0));
    let height_pt = f64::from(map_box.height.max(0.0));
    // The area gate is applied to the CONTINUOUS size, so the scale factor is
    // exact rather than an iteration on rounded edges.
    let budget = max_raster_pixels() as f64;
    let pixels = width_pt * height_pt * f64::from(px_per_pt) * f64::from(px_per_pt);
    if pixels > budget && pixels.is_finite() {
        px_per_pt *= (budget / pixels).sqrt() as f32;
    }
    if px_per_pt < requested {
        tracing::warn!(
            requested,
            clamped = px_per_pt,
            "oxigis-ui print: the raster resolution exceeds the export's ceiling; clamped",
        );
    }
    [
        (width_pt * f64::from(px_per_pt)).round().max(1.0) as u32,
        (height_pt * f64::from(px_per_pt)).round().max(1.0) as u32,
    ]
}

/// Hard ceiling on [`PrintOptions::raster_px_per_pt`]: `8.0` = 576 dpi, twice
/// the finest setting the dialog offers and past what any printer resolves
/// from a raster basemap.
pub const MAX_RASTER_PX_PER_PT: f32 = 8.0;

/// Ceiling on the composed raster's PIXEL COUNT, whatever the page size and
/// resolution multiply out to.
///
/// Three large buffers live at once during an export — the RGB raster, its
/// deflate output and the assembled PDF — so the budget is stated in pixels
/// (× 3 bytes for the first). Native gets 32 Mpx (96 MB raw), which clears
/// A3 at 300 dpi with room to spare; wasm32 gets 8 Mpx, because the web
/// shell runs in a 32-bit address space under a browser heap ceiling. Print
/// v1.8's JPEG candidate (`PrintOptions::photo_jpeg`) is a fourth buffer
/// that briefly coexists with the other three, but a *compressed* one — at
/// worst comparable to the deflate output it races against, never to the
/// raw raster — so it does not move this budget.
#[must_use]
pub const fn max_raster_pixels() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        8_000_000
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        32_000_000
    }
}

/// Reframes the on-screen camera to the print raster.
///
/// The horizontal extent is preserved exactly: the print shows the same
/// west-to-east span the user was looking at, rendered at the raster's higher
/// pixel density by raising the zoom by `log2(out_width / screen_width)`. The
/// vertical extent then follows the map box's aspect ratio, which is the page
/// designer's choice rather than the window manager's.
#[must_use]
pub fn compose_view(view: MapView, out_px: [u32; 2]) -> MapView {
    let screen_w = f64::from(view.size_px()[0]);
    let out_w = f64::from(out_px[0]);
    let zoom = if screen_w > 0.0 && out_w > 0.0 {
        view.zoom() + (out_w / screen_w).log2()
    } else {
        view.zoom()
    };
    let reframed = view.with_zoom(zoom);
    match reframed.with_size_px([out_px[0] as f32, out_px[1] as f32]) {
        Ok(sized) => sized,
        // An out-of-range surface (never produced by `raster_size_px`) keeps
        // the zoom change but prints the on-screen framing.
        Err(_) => reframed,
    }
}

/// The tiles [`compose_map_rgb`] will ask for — what the shell has to fetch.
#[must_use]
pub fn required_tiles(view: &MapView) -> Vec<TileId> {
    view.visible_tiles()
}

/// Pastes every available tile into one RGB buffer of the view's size.
///
/// The same [`MapView::place_tile`] math the GPU renderer uses, sampled
/// **bilinearly** and composited over paper white (print v1.6); `lookup`
/// answers [`None`] for a tile that has not arrived (or never will), and those
/// pixels keep the neutral [`MISSING_TILE_GRAY`]. RGB rather than RGBA because
/// the page below the image is opaque white and `/DeviceRGB` with no soft mask
/// is the simplest legal image a PDF can hold — the tile's own alpha is
/// resolved HERE instead (see `paste_tile`).
#[must_use]
pub fn compose_map_rgb(
    view: &MapView,
    lookup: &mut dyn FnMut(TileId) -> Option<DecodedTile>,
) -> Vec<u8> {
    let width = view.size_px()[0].round().max(1.0) as usize;
    let height = view.size_px()[1].round().max(1.0) as usize;
    let mut out = vec![MISSING_TILE_GRAY; width * height * 3];
    for tile in view.visible_tiles() {
        let Some(pixels) = lookup(tile) else {
            continue;
        };
        let placement = view.place_tile(tile);
        paste_tile(&mut out, width, height, &placement, &pixels, Over::Paper);
    }
    out
}

/// Composites ONE MORE raster layer over an already-composed buffer, at
/// `opacity` (compositing v1.6's page half).
///
/// [`compose_map_rgb`] is the bottom pass — the basemap, resolved against paper
/// white. Every raster layer the map draws OVER it is one call of this, in
/// stack order, so a project holding an orthophoto under a half-faded hillshade
/// prints both, faded the same way, instead of printing whichever one the
/// single-slot snapshot happened to name.
///
/// `out` must be the `width × height × 3` buffer [`compose_map_rgb`] returned
/// for the same `view`; a buffer of any other size is left untouched rather
/// than written past.
pub fn overlay_map_rgb(
    view: &MapView,
    out: &mut [u8],
    opacity: f32,
    lookup: &mut dyn FnMut(TileId) -> Option<DecodedTile>,
) {
    let width = view.size_px()[0].round().max(1.0) as usize;
    let height = view.size_px()[1].round().max(1.0) as usize;
    if out.len() != width * height * 3 {
        tracing::warn!(
            got = out.len(),
            want = width * height * 3,
            "oxigis-ui print: the overlay buffer does not match the composed view; layer skipped",
        );
        return;
    }
    let opacity = if opacity.is_finite() {
        opacity.clamp(0.0, 1.0)
    } else {
        1.0
    };
    if opacity <= 0.0 {
        return;
    }
    for tile in view.visible_tiles() {
        let Some(pixels) = lookup(tile) else {
            continue;
        };
        let placement = view.place_tile(tile);
        paste_tile(
            out,
            width,
            height,
            &placement,
            &pixels,
            Over::Existing { opacity },
        );
    }
}

/// What a pasted tile's alpha resolves against.
#[derive(Debug, Clone, Copy)]
enum Over {
    /// The paper: the bottom pass, whose transparency is white rather than
    /// whatever the missing-tile plate happens to be.
    Paper,
    /// Whatever is already in the buffer, times this layer's own opacity —
    /// the overlay passes.
    Existing {
        /// The layer's alpha, `0.0..=1.0`.
        opacity: f32,
    },
}

/// The paper the raster is printed on: a tile's alpha resolves against WHITE,
/// never against [`MISSING_TILE_GRAY`] — the missing-tile plate is what a
/// pixel NO tile covered prints as, and letting it tint a transparent overlay
/// would put grey where the page is white.
const PAPER_WHITE: f32 = 255.0;

/// Pastes one decoded tile into the output buffer: bilinear sampling,
/// alpha-over-paper compositing.
///
/// Tiles land on the print raster magnified by a factor in `[1, 2)` — the
/// composed view raises the zoom by `log2(out_w / screen_w)` while the tile
/// zoom is its floor — so nearest sampling turned every source texel into a
/// hard-edged 1-2 px block at print resolution, harsher than what the GPU
/// shows on screen. The four neighbouring texels are interpolated instead,
/// **premultiplied** so a transparent texel cannot bleed its RGB into its
/// neighbours, and the result is composited over [`PAPER_WHITE`]: an opaque
/// tile (every standard XYZ basemap) reproduces its bytes exactly, while a
/// hillshade, a labels-only overlay or a partly covered archive prints on
/// white instead of printing its raw RGB — typically a black plate.
fn paste_tile(
    out: &mut [u8],
    out_w: usize,
    out_h: usize,
    placement: &oxigis_render::TilePlacement,
    pixels: &DecodedTile,
    over: Over,
) {
    if placement.size <= 0.0 {
        return;
    }
    let tile_w = pixels.width() as usize;
    let tile_h = pixels.height() as usize;
    let rgba = pixels.rgba();
    if tile_w == 0 || tile_h == 0 || rgba.len() < tile_w * tile_h * 4 {
        return;
    }
    let x0 = placement.x.floor().max(0.0) as usize;
    let y0 = placement.y.floor().max(0.0) as usize;
    let x1 = ((placement.x + placement.size).ceil() as isize).clamp(0, out_w as isize) as usize;
    let y1 = ((placement.y + placement.size).ceil() as isize).clamp(0, out_h as isize) as usize;
    for y in y0..y1 {
        // Half-pixel centred: a DESTINATION pixel's centre maps to the
        // SOURCE pixel's centre, so the raster keeps its registration
        // instead of shifting half a texel up and left.
        let source_y = (y as f32 + 0.5 - placement.y) / placement.size * tile_h as f32 - 0.5;
        let (v0, v1, ty) = neighbours(source_y, tile_h);
        for x in x0..x1 {
            let source_x = (x as f32 + 0.5 - placement.x) / placement.size * tile_w as f32 - 0.5;
            let (u0, u1, tx) = neighbours(source_x, tile_w);
            let weights = [
                (u0 + v0 * tile_w, (1.0 - tx) * (1.0 - ty)),
                (u1 + v0 * tile_w, tx * (1.0 - ty)),
                (u0 + v1 * tile_w, (1.0 - tx) * ty),
                (u1 + v1 * tile_w, tx * ty),
            ];
            // Premultiplied accumulation: `[r·a, g·a, b·a, a]`, all in
            // 0..=255 scale.
            let mut sum = [0.0_f32; 4];
            for (texel, weight) in weights {
                let Some(source) = rgba.get(texel * 4..texel * 4 + 4) else {
                    return;
                };
                let alpha = f32::from(source[3]) / 255.0;
                for channel in 0..3 {
                    sum[channel] += f32::from(source[channel]) * alpha * weight;
                }
                sum[3] += f32::from(source[3]) * weight;
            }
            let dst = (y * out_w + x) * 3;
            let Some(target) = out.get_mut(dst..dst + 3) else {
                return;
            };
            let coverage = sum[3] / 255.0;
            for channel in 0..3 {
                // `sum` is PREMULTIPLIED by the tile's own alpha, so scaling it
                // by the layer's opacity is exactly premultiplying by the
                // combined alpha — no divide-by-alpha round trip, and a fully
                // transparent texel stays black-free either way.
                let value = match over {
                    Over::Paper => sum[channel] + PAPER_WHITE * (1.0 - coverage),
                    Over::Existing { opacity } => {
                        sum[channel] * opacity
                            + f32::from(target[channel]) * (1.0 - coverage * opacity)
                    }
                };
                target[channel] = value.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
}

/// The two texel indices bracketing a continuous source coordinate and the
/// weight of the second, clamped at the tile's edges so the seam pixel of an
/// upscaled tile can neither read past the end nor wrap.
fn neighbours(coordinate: f32, extent: usize) -> (usize, usize, f32) {
    let last = extent.saturating_sub(1);
    let base = coordinate.floor();
    let fraction = (coordinate - base).clamp(0.0, 1.0);
    let low = (base.max(0.0) as usize).min(last);
    let high = if base < 0.0 { 0 } else { (low + 1).min(last) };
    (low, high, fraction)
}

/// Builds the page's raw content-stream operators — the test seam.
///
/// The degraded-mode variant of [`page_content_planned`]: text goes through
/// Base-14 Helvetica with the WinAnsi `?` fallback, exactly the v1 output.
#[must_use]
pub fn page_content(request: &PrintRequest, compose: &MapView, map_box: &MapBox) -> Vec<u8> {
    page_content_planned(request, compose, map_box, None, &[])
}

/// Builds the page's raw content-stream operators — the test seam.
///
/// [`pdf_document`] compresses this into the final file; tests read it
/// uncompressed to assert the image transform, the vector operators and the
/// text runs are all present. `plan` carries the embedded fonts the font
/// pass built from the same request, or [`None`] for the degraded
/// Helvetica text path; `vector_tiles` are the decoded MVT tiles the shell
/// fetched for [`PrintRequest::vector`] (empty when there is no streamed
/// vector layer).
#[must_use]
pub fn page_content_planned(
    request: &PrintRequest,
    compose: &MapView,
    map_box: &MapBox,
    plan: Option<&TextPlan>,
    vector_tiles: &[(TileId, Arc<MvtVectorTile>)],
) -> Vec<u8> {
    page_content_planned_with(
        request,
        compose,
        map_box,
        plan,
        &PrintVectorTiles {
            single: vector_tiles,
            stack: &[],
        },
    )
}

/// [`page_content_planned`] over the whole tiled stack (compositing v1.6).
///
/// The raster half is already in `map_rgb` — a shell composes the basemap and
/// then every raster entry with [`overlay_map_rgb`] — so what this adds is the
/// VECTOR half: one clipped group per vector-tile entry, painted bottom-up in
/// stack order rather than only for [`PrintRequest::vector`].
///
/// # The one z-order the page still flattens
///
/// Rasters are one embedded image and vectors are paths drawn over it, so a
/// raster entry that sits ABOVE a vector-tile entry on screen still prints
/// below it. That is the page's own documented order (raster → MVT → local →
/// labels) and predates the stack; changing it means one image XObject per
/// raster run with a soft mask each, which is a separate design. Every other
/// arrangement — any number of rasters, any number of vector tilesets — now
/// composes exactly as the screen does.
#[must_use]
pub fn page_content_planned_with(
    request: &PrintRequest,
    compose: &MapView,
    map_box: &MapBox,
    plan: Option<&TextPlan>,
    tiles: &PrintVectorTiles<'_>,
) -> Vec<u8> {
    let mut content = Content::new();

    // The raster map: unit image scaled to the map box.
    content.save_state();
    content.transform([
        map_box.width,
        0.0,
        0.0,
        map_box.height,
        map_box.x,
        map_box.y,
    ]);
    content.x_object(Name(b"Im0"));
    content.restore_state();

    // The vector overlay, clipped to the map box so a feature crossing the
    // page edge cannot draw over the margins.
    content.save_state();
    content.rect(map_box.x, map_box.y, map_box.width, map_box.height);
    content.clip_nonzero();
    content.end_path();
    // Streamed vector tiles draw over the raster and under the local
    // layers, matching the screen's paint order.
    {
        let screen_w = request.view.size_px()[0];
        let px_to_pt = if screen_w > 0.0 {
            map_box.width / screen_w
        } else {
            0.5
        };
        // Bottom-up, one group per source: a cadastral tileset under a labels
        // tileset prints in that order, which is the order the layer panel
        // shows and the map draws.
        for (entry, config) in request.vector_sources() {
            mvt::paint_vector_tiles(
                &mut content,
                &config.paints,
                tiles.of(entry),
                compose,
                map_box,
                px_to_pt,
                entry,
            );
        }
    }
    for (index, layer) in request.layers.iter().enumerate() {
        paint::paint_layer(&mut content, layer, index, compose, map_box);
    }
    // Labels draw over every vector layer, exactly as on screen — and only
    // when embedded fonts exist, because '?' place-holders for CJK names
    // would be worse than the v1 behavior of drawing nothing.
    if let Some(plan) = plan {
        let placed = labels::place(request, compose, map_box, plan, tiles);
        paint_labels(&mut content, &placed, plan);
    }
    content.restore_state();

    // Cartographic furniture (print v1.7). Each piece is drawn OUTSIDE the
    // map box's clip — it annotates the map rather than belonging to it —
    // and each claims one corner of the map box: the bar bottom-left, the
    // legend bottom-right, the arrow top-right. No band is reserved for
    // them, deliberately: reserving one would move `map_box`, and with it
    // the raster size and every page's framing.
    scalebar::paint(&mut content, &request.options, compose, map_box, plan);
    north::paint(&mut content, &request.options, map_box, plan);
    legend::paint(&mut content, request, compose, map_box, plan);

    // Title. Horizontally at the top left, exactly as it always has been —
    // or, when the export asked for a vertical title AND the refusal ladder
    // accepted the line, down the right-hand margin strip, which is the
    // conventional place for a Japanese vertical title and the only band of
    // the page that is clear of the map box for its whole height.
    let [page_w, page_h] = request.options.page_size_pt();
    let title_top = page_h - PAGE_MARGIN_PT - TITLE_FONT_PT + 4.0;
    content.set_fill_rgb(0.0, 0.0, 0.0);
    match plan.and_then(TextPlan::vertical_title).filter(|line| {
        // A title too long to hang inside the page prints horizontally
        // rather than running off the bottom edge.
        line.box_pt(TITLE_FONT_PT)[1] <= title_top - PAGE_MARGIN_PT
    }) {
        Some(line) => show_vertical_line(
            &mut content,
            line,
            page_w - PAGE_MARGIN_PT + (PAGE_MARGIN_PT - TITLE_FONT_PT) / 2.0,
            title_top,
            TITLE_FONT_PT,
            TextMark::Content,
        ),
        None => show_line(
            &mut content,
            plan,
            PAGE_MARGIN_PT,
            title_top,
            TITLE_FONT_PT,
            &elide_to_width(plan, &request.title, TITLE_FONT_PT, text_room_pt(page_w)),
        ),
    }

    // Attribution, bottom-left under the map box.
    if !request.attribution.is_empty() {
        content.set_fill_rgb(0.25, 0.25, 0.25);
        show_line(
            &mut content,
            plan,
            PAGE_MARGIN_PT,
            PAGE_MARGIN_PT,
            FOOTER_FONT_PT,
            &elide_to_width(
                plan,
                &request.attribution,
                FOOTER_FONT_PT,
                text_room_pt(page_w),
            ),
        );
    }

    content.finish().into_vec()
}

/// How much width a full-page line of page furniture has, in points: the
/// paper minus a margin on each side.
fn text_room_pt(page_w: f32) -> f32 {
    (page_w - 2.0 * PAGE_MARGIN_PT).max(0.0)
}

/// The width `text` advances at `size` pt, in points — the plan's exact `/W`
/// numbers when one is live, the degraded path's 0.6-em Helvetica estimate
/// otherwise (the same estimate the scale-bar plate has always used).
fn line_width_pt(plan: Option<&TextPlan>, text: &str, size: f32) -> f32 {
    match plan {
        Some(plan) => plan.width_pt(oxigis_core::LabelWeight::Regular, text, size),
        None => text.chars().count() as f32 * size * 0.6,
    }
}

/// The ellipsis a trimmed line ends with: the real character when the page
/// can draw it — degraded WinAnsi always can (0x85), an embedded plan only if
/// some face on the page covered it — and three periods otherwise.
fn ellipsis(plan: Option<&TextPlan>) -> &'static str {
    match plan {
        Some(plan) if plan.glyph(oxigis_core::LabelWeight::Regular, '…').is_none() => "...",
        _ => "…",
    }
}

/// `text` if it fits `max_width_pt`, otherwise the longest character prefix
/// that fits WITH a trailing ellipsis.
///
/// The title is `Project::name` verbatim and the attribution is a join of up
/// to three credits, at least one of them project-supplied, so an over-long
/// line is reachable without hostile intent — and the horizontal band has no
/// clip of its own (the map box's was restored before the furniture is
/// drawn), so an unchecked line simply runs off the media box. The vertical
/// title path has always measured itself; this is the horizontal twin.
///
/// Measurement equals rendering: a trimmed string is never a planned string,
/// so both this function and the emitter reach [`TextPlan`]'s synthetic
/// per-character path for it.
fn elide_to_width<'a>(
    plan: Option<&TextPlan>,
    text: &'a str,
    size: f32,
    max_width_pt: f32,
) -> std::borrow::Cow<'a, str> {
    if line_width_pt(plan, text, size) <= max_width_pt {
        return std::borrow::Cow::Borrowed(text);
    }
    let ellipsis = ellipsis(plan);
    let mut kept = String::new();
    let mut candidate = String::new();
    for ch in text.chars() {
        kept.push(ch);
        let mut probe = kept.clone();
        probe.push_str(ellipsis);
        if line_width_pt(plan, &probe, size) > max_width_pt {
            break;
        }
        candidate = probe;
    }
    std::borrow::Cow::Owned(candidate)
}

/// Projects one position to page points, or [`None`] for a malformed one.
fn project(compose: &MapView, map_box: &MapBox, position: &Position) -> Option<(f32, f32)> {
    let (&lon, &lat) = (position.first()?, position.get(1)?);
    if !lon.is_finite() || !lat.is_finite() {
        return None;
    }
    let px = compose.lon_lat_to_screen(LonLat::new(lon, lat));
    let scale_x = compose.size_px()[0] / map_box.width;
    let scale_y = compose.size_px()[1] / map_box.height;
    if scale_x <= 0.0 || scale_y <= 0.0 {
        return None;
    }
    Some((
        map_box.x + px[0] / scale_x,
        map_box.y + map_box.height - px[1] / scale_y,
    ))
}

/// Draws the placed labels: the classic two-pass halo (stroke-only text
/// under fill-only text), each under its layer's alpha ExtGState.
///
/// The stroke pass is wrapped as a marked-content **artifact** — it repeats
/// the same show-text operators as the fill pass, and without the marking a
/// conformant extractor reads every haloed label TWICE (a real v1.1 bug).
fn paint_labels(content: &mut Content, placed: &[labels::PlacedLabel], plan: &TextPlan) {
    for label in placed {
        // A vertical label draws through the column emitter and a horizontal
        // one through the line emitter; `x`/`y` were measured by whichever
        // box the placer used, so the two can never be swapped.
        let column = label
            .vertical
            .then(|| plan.vertical_line(label.weight, &label.text))
            .flatten();
        content.save_state();
        content.set_parameters(Name(label.alpha.name().as_bytes()));
        if let Some((halo_color, halo_width)) = label.halo {
            let rgb = to_rgb(halo_color);
            content.set_stroke_rgb(rgb[0], rgb[1], rgb[2]);
            content.set_line_width(halo_width);
            content.set_line_join(LineJoinStyle::RoundJoin);
            content.set_text_rendering_mode(TextRenderingMode::Stroke);
            content.begin_marked_content(Name(b"Artifact"));
            // Artifact mode: the stroke pass repeats the fill pass's
            // geometry and must NOT repeat its `/ActualText` spans.
            match column {
                Some(line) => show_vertical_line(
                    content,
                    line,
                    label.x,
                    label.y,
                    label.size,
                    TextMark::Artifact,
                ),
                None => show_line_marked(
                    content,
                    Some(plan),
                    label.x,
                    label.y,
                    label.size,
                    WeightedText {
                        text: &label.text,
                        weight: label.weight,
                    },
                    TextMark::Artifact,
                ),
            }
            content.end_marked_content();
        }
        content.set_text_rendering_mode(TextRenderingMode::Fill);
        let rgb = to_rgb(label.color);
        content.set_fill_rgb(rgb[0], rgb[1], rgb[2]);
        match column {
            Some(line) => show_vertical_line(
                content,
                line,
                label.x,
                label.y,
                label.size,
                TextMark::Content,
            ),
            None => show_line_marked(
                content,
                Some(plan),
                label.x,
                label.y,
                label.size,
                WeightedText {
                    text: &label.text,
                    weight: label.weight,
                },
                TextMark::Content,
            ),
        }
        content.restore_state();
    }
}

/// A layer's BASE ExtGState resource name, `GS0`, `GS1`, … — the labels'
/// alpha too, so the label pass keeps its exact pre-v1.3 meaning.
fn alpha_name(index: usize) -> String {
    format!("GS{index}")
}

/// The ExtGState name one slot of a layer's style set paints under: the
/// base keeps `GS{i}` (byte-identical resources for a no-override layer),
/// an override gets `GS{i}f{0|1|2}`.
fn slot_alpha_name(index: usize, slot: oxigis_core::StyleSlot) -> String {
    match slot {
        oxigis_core::StyleSlot::Base => alpha_name(index),
        oxigis_core::StyleSlot::Family(family) => format!("GS{index}f{}", family.index()),
    }
}

/// The ExtGState name one CLASS of a layer's slot paints under (thematic
/// v1.6): the fallback bucket keeps the slot's own name, a class appends
/// `c{class}`.
///
/// The fallback deliberately carries NO suffix. It is the bucket an
/// unclassified layer's every feature lands in, so a `Renderer::Single` layer
/// names exactly `GS{i}` / `GS{i}f{n}` — the pre-v1.6 resource set, byte for
/// byte, which is what keeps every existing export unchanged.
fn class_alpha_name(index: usize, slot: oxigis_core::StyleSlot, class: Option<usize>) -> String {
    match class {
        None => slot_alpha_name(index, slot),
        Some(class) => format!("{}c{class}", slot_alpha_name(index, slot)),
    }
}

/// The class buckets one style set paints, in the order they are painted:
/// the fallback first, then class `0`, class `1`, … .
///
/// The SAME order [`crate::local_vector::feature_collection_to_tile_with`]
/// emits the map's own buckets in, so a class that overlaps another draws on
/// top of it identically on screen and on the page.
fn class_buckets(set: &oxigis_core::LayerStyleSet) -> impl Iterator<Item = Option<usize>> + use<> {
    core::iter::once(None).chain((0..set.class_count()).map(Some))
}

/// Which geometry family one path kind belongs to — the shared partition
/// that makes a per-family pass a `PathKind` filter.
fn path_family(kind: PathKind) -> oxigis_core::GeometryFamily {
    match kind {
        PathKind::Ring => oxigis_core::GeometryFamily::Polygon,
        PathKind::Line => oxigis_core::GeometryFamily::Line,
        PathKind::Points => oxigis_core::GeometryFamily::Point,
    }
}

/// A [`Color`]'s RGB channels as unit floats (alpha travels separately,
/// through the layer's ExtGState).
fn to_rgb(color: Color) -> [f32; 3] {
    [
        f32::from(color.r) / 255.0,
        f32::from(color.g) / 255.0,
        f32::from(color.b) / 255.0,
    ]
}

/// A layer's effective BASE alpha: its own opacity times its base
/// style's — the `GS{i}` value, which the label pass also paints under.
fn layer_alpha(layer: &PrintLayer) -> f32 {
    (layer.opacity * style_opacity(layer.style.base())).clamp(0.0, 1.0)
}

/// One style's own constant alpha.
fn style_opacity(style: &LayerStyle) -> f32 {
    match style {
        LayerStyle::Fill(fill) => fill.opacity(),
        LayerStyle::Line(line) => line.opacity(),
        LayerStyle::Circle(circle) => circle.opacity(),
        LayerStyle::Symbol(_) => 1.0,
    }
}

/// The ExtGState slots one print layer actually names: the base always
/// (the label pass paints under it), plus each PRESENT family that carries
/// an override. The registration and the painter read this ONE list, so
/// they cannot disagree.
fn layer_alpha_slots(layer: &PrintLayer) -> Vec<(oxigis_core::StyleSlot, Option<usize>, f32)> {
    let mut slots = vec![(oxigis_core::StyleSlot::Base, None, layer_alpha(layer))];
    for family in layer.families.iter() {
        if let oxigis_core::StyleSlot::Family(_) = layer.style.slot_of(family) {
            let alpha =
                (layer.opacity * style_opacity(layer.style.effective(family))).clamp(0.0, 1.0);
            slots.push((oxigis_core::StyleSlot::Family(family), None, alpha));
        }
    }
    // One further state per (DISTINCT slot, class): a categorized layer whose
    // classes differ in opacity needs one ExtGState each, or the painter would
    // name a resource the page never registered — an invalid PDF rather than a
    // wrong colour. An unclassified layer adds none of these, so its resource
    // dictionary is unchanged.
    //
    // Distinct SLOTS, not families, because the painter's name is built from
    // the slot too: two families that both resolve through the base compose the
    // same class over the same style, so they are one state and one name, not
    // two identical dictionary keys.
    let mut named: Vec<(oxigis_core::StyleSlot, oxigis_core::GeometryFamily)> = Vec::new();
    for family in layer.families.iter() {
        let slot = layer.style.slot_of(family);
        if named.iter().any(|(known, _)| *known == slot) {
            continue;
        }
        named.push((slot, family));
    }
    for (slot, family) in named {
        for class in class_buckets(&layer.style).flatten() {
            let style = layer.style.style_for_class(family, Some(class));
            let alpha = (layer.opacity * style_opacity(&style)).clamp(0.0, 1.0);
            slots.push((slot, Some(class), alpha));
        }
    }
    slots
}

#[cfg(test)]
mod furniture_tests;
#[cfg(test)]
mod stack_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod thematic_tests;
