// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! [`ArchiveProbe`]: read the header first, decide what the layer *is* second.
//!
//! Adding an archive is the one add gesture in OxiGIS whose **layer kind is not
//! known when the user presses the button**. A `.pmtiles` URL may hold PNG
//! tiles or MVT ones, and those are drawn by two entirely different providers
//! and serialize as two different [`oxigis_core::LayerKind`] variants. So the
//! gesture starts a probe, and the layer is created only when the answer lands:
//! nothing half-decided is ever written into a project file, and
//! `record_layer_add` fires once, at creation, so one Ctrl+Z removes exactly
//! what one gesture added.
//!
//! # The polling idiom is the shell's existing one
//!
//! [`ArchiveProbe::take`] answers [`None`] until the header (and the metadata,
//! which may need a second read) has arrived. A shell polls it once per frame,
//! exactly as the desktop shell polls its background CJK font scan and the web
//! shell polls `font_fetch::take_pending_font`. No new idiom, no new thread, no
//! `async` in `oxigis-ui`.
//!
//! # What the probe does not do
//!
//! It does not hand the opened archive forward. The provider the answer selects
//! re-opens the archive itself, which costs one more 16 KiB range read and buys
//! a much smaller seam: the reconciliation in `app/providers.rs` derives
//! providers from the *project*, and threading a warm archive through that
//! derivation would make what is drawn depend on the history of probes again —
//! the exact coupling editing v1.3 removed.

use std::sync::Arc;

use oxigis_core::ArchiveFormat;
use oxigis_render::pmtiles::{PmtilesArchive, PmtilesOpen, TileType};
use parking_lot::Mutex;

use crate::archive::open::{ArchiveContent, OpenStep, advance_open, check_archive};
use crate::cog_provider::{RangeDelivery, RangeJob, RangeSink, RangeTransport};
use crate::mbtiles::paged::{PagedNeed, PagedOpen, PagedOpenStep};
use crate::tile_provider::TileError;

/// Everything about an archive a layer needs before it can be created.
///
/// Half comes from the 127-byte header (what the tiles are, the zoom range,
/// the bounding box) and half from the archive's own metadata JSON, which
/// `oxigis-render` deliberately hands over unparsed — see
/// [`oxigis_render::pmtiles::PmtilesInfo`] for why the parse belongs on this
/// side of the seam.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchiveInfo {
    /// Whether the archive holds raster tiles or vector ones.
    pub content: ArchiveContent,
    /// The tile bodies' codec, as the header declares it.
    pub codec: TileType,
    /// Lowest zoom the archive holds.
    pub min_zoom: u8,
    /// Highest zoom the archive holds.
    pub max_zoom: u8,
    /// `[min_lon, min_lat, max_lon, max_lat]` in degrees.
    pub bounds_deg: [f64; 4],
    /// Whether the bounding box was actually declared, or is all zeroes.
    pub has_bounds: bool,
    /// Suggested map centre as `(lon, lat)` in degrees.
    pub center_deg: (f64, f64),
    /// Suggested opening zoom.
    pub center_zoom: u8,
    /// The archive's own name, from its metadata; empty when it declares none.
    pub name: String,
    /// Credit line the archive's metadata asks for; empty when it asks for
    /// none. Flows into the layer's config and from there into the map's one
    /// derived credit line.
    pub attribution: String,
    /// Names of the layers *inside* the vector tiles, from the metadata's
    /// `vector_layers`. Empty for a raster archive, and for a vector archive
    /// whose writer declared none.
    pub layer_names: Vec<String>,
    /// On-screen size of one tile in pixels, when the metadata declares it.
    /// 512 is real: the pmtiles.io USGS sample says so.
    pub tile_size_px: Option<u32>,
}

impl ArchiveInfo {
    /// Reads an opened archive's header and metadata.
    ///
    /// # Errors
    ///
    /// Returns the named refusal `archive::open::check_archive` produces for an
    /// archive this
    /// build will not read (AVIF tiles, an undeclared tile type or codec, a
    /// codec `deny.toml` bans).
    pub fn from_pmtiles(archive: &PmtilesArchive) -> Result<Self, String> {
        let content = check_archive(archive, None)?;
        let facts = archive.info();
        let metadata = Metadata::parse(archive.metadata_json());
        Ok(Self {
            content,
            codec: facts.tile_type,
            min_zoom: facts.min_zoom,
            max_zoom: facts.max_zoom,
            bounds_deg: facts.bounds_deg,
            has_bounds: facts.has_bounds,
            center_deg: facts.center_deg,
            center_zoom: facts.center_zoom,
            name: metadata.name,
            attribution: metadata.attribution,
            layer_names: metadata.vector_layers,
            tile_size_px: metadata.tile_size_px,
        })
    }

    /// A one-line summary for the status bar.
    #[must_use]
    pub fn summary(&self) -> String {
        let kind = match self.content {
            ArchiveContent::Raster => format!("{} raster tiles", self.codec.name()),
            ArchiveContent::Vector => {
                if self.layer_names.is_empty() {
                    "vector tiles".to_owned()
                } else {
                    format!("vector tiles ({} layers)", self.layer_names.len())
                }
            }
        };
        format!("{kind}, zoom {}\u{2013}{}", self.min_zoom, self.max_zoom)
    }
}

/// The fields this crate reads out of an archive's metadata JSON.
///
/// Everything is optional and everything has a defined absence: a writer that
/// declares nothing produces a usable layer with no credit line and the neutral
/// default paints.
#[derive(Debug, Default)]
struct Metadata {
    /// `name`.
    name: String,
    /// `attribution`.
    attribution: String,
    /// The `id` of each entry of `vector_layers`, in declaration order.
    vector_layers: Vec<String>,
    /// `tileSize`, which real archives write as a *string*.
    tile_size_px: Option<u32>,
}

impl Metadata {
    /// Reads what it recognises out of `json`, ignoring the rest.
    ///
    /// Never fails: an archive with no metadata block at all (`""`), or with a
    /// block that is not JSON, is a usable archive with nothing declared —
    /// refusing it would make a perfectly readable tile pyramid unopenable over
    /// a field nothing depends on.
    fn parse(json: &str) -> Self {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(json) else {
            return Self::default();
        };
        let string = |key: &str| {
            value
                .get(key)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        let vector_layers = value
            .get("vector_layers")
            .and_then(serde_json::Value::as_array)
            .map(|layers| {
                layers
                    .iter()
                    .filter_map(|layer| layer.get("id").and_then(serde_json::Value::as_str))
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let tile_size_px = value.get("tileSize").and_then(|size| {
            size.as_u64()
                .or_else(|| size.as_str().and_then(|text| text.parse::<u64>().ok()))
                .and_then(|size| u32::try_from(size).ok())
                .filter(|size| *size > 0)
        });
        Self {
            name: string("name"),
            attribution: string("attribution"),
            vector_layers,
            tile_size_px,
        }
    }
}

/// An archive whose header has been read, and what it turned out to be.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenedArchive {
    /// What the probe was pointed at, verbatim — the string a layer records and
    /// a provider later re-opens.
    pub location: String,
    /// What the archive holds.
    pub info: ArchiveInfo,
}

/// The open a probe is driving, whichever container it was pointed at.
///
/// Learning the format here rather than at the call site is what keeps
/// probe-then-create true for **both** formats: a `.mbtiles` URL is surveyed
/// through the same one-round-trip gesture a `.pmtiles` URL is, so its refusals
/// — no index, no `images.tile_id` index, a non-`BINARY` collation, `WITHOUT
/// ROWID`, an untrustworthy header — land *before* a layer exists.
enum Reading {
    /// A PMTiles archive: header plus root directory.
    Pmtiles(Box<PmtilesOpen>),
    /// An MBTiles archive: the 16 KiB SQLite survey.
    Paged(Box<PagedOpen>),
}

/// How far along a probe is.
enum ProbeStage {
    /// The opening read (and possibly a little more) is still outstanding.
    Reading(Box<Reading>),
    /// The answer is waiting to be taken.
    Done(Box<Result<OpenedArchive, TileError>>),
    /// The answer has been taken; nothing more will happen.
    Taken,
}

/// Shared probe state.
struct ProbeInner {
    /// What the transport is pointed at.
    location: String,
    /// How far along the read is.
    stage: Mutex<ProbeStage>,
    /// Context to wake when the answer lands.
    ctx: egui::Context,
    /// The platform's range-read capability.
    transport: Box<dyn RangeTransport>,
}

impl ProbeInner {
    /// Drives the open one step and issues whatever read it asks for.
    fn advance(self: &Arc<Self>) {
        let request = {
            let mut stage = self.stage.lock();
            let ProbeStage::Reading(reading) = &mut *stage else {
                return;
            };
            match &mut **reading {
                Reading::Pmtiles(open) => match advance_open(open) {
                    OpenStep::Need(range) => {
                        Some((range, RangeJob::ArchiveHeader { start: range.start }))
                    }
                    OpenStep::Ready(archive) => {
                        *stage = ProbeStage::Done(Box::new(self.finish(&archive)));
                        None
                    }
                    OpenStep::Failed(message) => {
                        *stage = ProbeStage::Done(Box::new(Err(TileError::permanent(message))));
                        None
                    }
                },
                Reading::Paged(open) => match open.step() {
                    Ok(PagedOpenStep::Need(PagedNeed::Prefetch(range))) => {
                        Some((range, RangeJob::ArchiveSurvey { start: range.start }))
                    }
                    Ok(PagedOpenStep::Need(PagedNeed::Pages(run))) => match open.range_for(run) {
                        Ok(range) => Some((
                            range,
                            RangeJob::ArchivePage {
                                first: run.first,
                                count: run.count,
                            },
                        )),
                        Err(error) => {
                            *stage = ProbeStage::Done(Box::new(Err(TileError::permanent(
                                error.to_string(),
                            ))));
                            None
                        }
                    },
                    Ok(PagedOpenStep::Ready(archive)) => {
                        *stage = ProbeStage::Done(Box::new(Ok(OpenedArchive {
                            location: self.location.clone(),
                            info: archive.info(),
                        })));
                        None
                    }
                    Err(error) => {
                        *stage = ProbeStage::Done(Box::new(Err(TileError::permanent(
                            error.to_string(),
                        ))));
                        None
                    }
                },
            }
        };
        match request {
            Some((range, job)) => self.transport.request_range(
                self.location.clone(),
                range,
                job,
                RangeSink::from_delivery(Arc::clone(self) as Arc<dyn RangeDelivery>),
            ),
            None => self.ctx.request_repaint(),
        }
    }

    /// Turns an opened archive into the probe's answer.
    fn finish(&self, archive: &PmtilesArchive) -> Result<OpenedArchive, TileError> {
        match ArchiveInfo::from_pmtiles(archive) {
            Ok(info) => Ok(OpenedArchive {
                location: self.location.clone(),
                info,
            }),
            Err(message) => Err(TileError::permanent(message)),
        }
    }
}

impl RangeDelivery for ProbeInner {
    fn deliver_range(self: Arc<Self>, job: RangeJob, result: Result<Vec<u8>, TileError>) {
        {
            let mut stage = self.stage.lock();
            let ProbeStage::Reading(reading) = &mut *stage else {
                return;
            };
            let bytes = match result {
                Ok(bytes) => bytes,
                Err(error) => {
                    *stage = ProbeStage::Done(Box::new(Err(error)));
                    return;
                }
            };
            let refusal = match (&mut **reading, job) {
                (Reading::Pmtiles(open), RangeJob::ArchiveHeader { start }) => open
                    .supply(start, bytes)
                    .err()
                    .map(|error| error.to_string()),
                (Reading::Paged(open), RangeJob::ArchiveSurvey { .. }) => open
                    .supply_prefetch(&bytes)
                    .err()
                    .map(|error| error.to_string()),
                (Reading::Paged(open), RangeJob::ArchivePage { first, .. }) => {
                    open.supply_pages(first, &bytes);
                    None
                }
                _ => {
                    tracing::debug!("oxigis-ui: an unexpected range job reached the archive probe");
                    return;
                }
            };
            if let Some(message) = refusal {
                *stage = ProbeStage::Done(Box::new(Err(TileError::permanent(message))));
            }
        }
        self.advance();
    }
}

/// One archive being identified, before any layer exists for it.
///
/// Cheap to hold in [`crate::OxigisApp`] across frames; dropping it abandons
/// the probe (an in-flight range read simply finds nothing to report to).
pub struct ArchiveProbe {
    /// Shared probe state.
    inner: Arc<ProbeInner>,
}

impl core::fmt::Debug for ArchiveProbe {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ArchiveProbe")
            .field("location", &self.inner.location)
            .finish_non_exhaustive()
    }
}

impl ArchiveProbe {
    /// Starts reading the `format` archive at `location`, waking `ctx` when the
    /// answer lands.
    ///
    /// The first range read is issued immediately: unlike a provider, a probe
    /// exists *because* the user asked for the answer, so there is nothing to
    /// defer to a later frame.
    #[must_use]
    pub fn start(
        location: impl Into<String>,
        format: ArchiveFormat,
        ctx: &egui::Context,
        transport: Box<dyn RangeTransport>,
    ) -> Self {
        let reading = match format {
            ArchiveFormat::PmTiles => Reading::Pmtiles(Box::new(PmtilesOpen::new())),
            // No pinned length: the probe answers a gesture, not a transport, so
            // it has no `Content-Range` total to bound the page count with. A
            // page past the end is then caught by its short delivery.
            ArchiveFormat::MbTiles => Reading::Paged(Box::new(PagedOpen::new(None))),
        };
        let inner = Arc::new(ProbeInner {
            location: location.into(),
            stage: Mutex::new(ProbeStage::Reading(Box::new(reading))),
            ctx: ctx.clone(),
            transport,
        });
        inner.advance();
        Self { inner }
    }

    /// What the probe was pointed at.
    #[must_use]
    pub fn location(&self) -> &str {
        &self.inner.location
    }

    /// The answer, once and once only; [`None`] until it lands.
    ///
    /// Take-once, like every other one-shot hand-off in this crate: a caller
    /// that has already created the layer must not be handed the answer again
    /// on the next frame.
    #[must_use]
    pub fn take(&self) -> Option<Result<OpenedArchive, TileError>> {
        let mut stage = self.inner.stage.lock();
        if !matches!(*stage, ProbeStage::Done(_)) {
            return None;
        }
        let ProbeStage::Done(answer) = core::mem::replace(&mut *stage, ProbeStage::Taken) else {
            return None;
        };
        Some(*answer)
    }
}
