// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Coordinate reference systems: what CRS a layer's data is stored in, and how
//! to get it onto WGS 84.
//!
//! Before this module the model had no CRS at all — every coordinate in
//! `oxigis-core` was implicitly WGS 84 lon/lat, and the shell compensated with
//! a two-value string sniff that *refused* everything else at ingest. A
//! Japanese municipal shapefile in a plane-rectangular zone — the most common
//! real input in this project's home market — could not be opened.
//!
//! # The pieces
//!
//! * [`Crs`] — the model type. An EPSG code plus, when the source carried one,
//!   its WKT text. Lives on [`crate::layer::Layer`] as an **optional** field
//!   whose absence means WGS 84, so every `.oxigis.json` written before this
//!   existed round-trips byte-identically.
//! * [`wkt`] — a real WKT reader: depth-aware authority-code extraction for
//!   both WKT1 (`AUTHORITY["EPSG","6677"]`) and WKT2 (`ID["EPSG",6677]`), plus
//!   the name fallbacks ESRI `.prj` files need.
//! * [`epsg`] — which codes this build knows and what each one *is*. Owned
//!   here rather than delegated, because the obvious delegate is wrong for the
//!   Japanese codes; that module's docs carry the measurement.
//! * [`datum`] — ellipsoids and the published Helmert shifts to WGS 84.
//! * [`reproject`] — [`reproject::Reprojector`], the once-per-dataset object
//!   every ingest path maps its vertices through.
//!
//! # The seam
//!
//! Shells never touch OxiGeo's projection stack. They resolve a [`Crs`] from
//! whatever their format declares (`.prj` text, a GeoPackage `srs_id`, a
//! GeoParquet `crs` object), build one [`reproject::Reprojector`], and call
//! [`reproject::Reprojector::to_lon_lat`] per vertex. A CRS this build cannot
//! place is refused at construction, with the code and the name in the
//! message.

pub mod datum;
pub mod epsg;
pub mod reproject;
pub mod wkt;

pub use datum::{Datum, EllipsoidKind};
pub use epsg::{CrsDef, LonLatBounds, Projection, definition, is_supported};
pub use reproject::{AxisOrder, ReprojectError, Reprojector};
pub use wkt::{WktInfo, WktKind, crs_label, parse_wkt, resolve_epsg};

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// EPSG code for WGS 84 geographic — the CRS every OxiGIS coordinate ends up
/// in, and the one an absent [`Crs`] means.
pub const EPSG_WGS84: u32 = 4326;

/// EPSG code for Web Mercator ("WGS 84 / Pseudo-Mercator").
pub const EPSG_WEB_MERCATOR: u32 = 3857;

/// The code a [`Crs`] carries when its source declared none this reader could
/// name.
///
/// Zero is not an EPSG code, and the GeoPackage specification already uses
/// `srs_id = 0` for "undefined geographic", so it reads correctly in the one
/// format that has an opinion.
pub const EPSG_UNKNOWN: u32 = 0;

/// A coordinate reference system: an EPSG code, and the WKT text it came from
/// when there was one.
///
/// # Serde contract (additive, and it must stay that way)
///
/// * A missing `epsg` key deserializes to [`EPSG_WGS84`], so `{}` is WGS 84.
/// * `wkt` is skipped when absent, so a `Crs` with no WKT serializes to
///   `{"epsg":6677}` and nothing more.
/// * The whole field is [`Option`]al on [`crate::layer::Layer`] and skipped
///   when [`None`], so a project file written before CRSs existed re-saves
///   byte-identically. `oxigis-core` has tests that assert exactly that on a
///   whole v1.3 project document; they are load-bearing.
///
/// # Bounded
///
/// A WKT longer than [`Crs::MAX_WKT_BYTES`] is **not retained** — neither at
/// construction nor on load. The EPSG code is what every decision is made on;
/// keeping an unbounded blob of text on a model type that gets cloned per
/// layer and written into the project file buys nothing. Dropping is
/// idempotent: a file with an oversized WKT loads without it and re-saves
/// without it, rather than re-saving a truncated (and therefore invalid) one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Crs {
    epsg: u32,
    wkt: Option<String>,
}

impl Crs {
    /// The longest WKT text a `Crs` will retain, in bytes. A WKT2 compound CRS
    /// with a full usage block runs to a few kilobytes; this is comfortably
    /// past that and still a hard bound.
    pub const MAX_WKT_BYTES: usize = 16 * 1024;

    /// WGS 84 geographic — the default, and what an absent CRS means.
    #[must_use]
    pub fn wgs84() -> Self {
        Self {
            epsg: EPSG_WGS84,
            wkt: None,
        }
    }

    /// Web Mercator (EPSG:3857).
    #[must_use]
    pub fn web_mercator() -> Self {
        Self {
            epsg: EPSG_WEB_MERCATOR,
            wkt: None,
        }
    }

    /// A CRS named only by its EPSG code.
    #[must_use]
    pub fn from_epsg(epsg: u32) -> Self {
        Self { epsg, wkt: None }
    }

    /// A CRS read out of a WKT string.
    ///
    /// The code comes from [`wkt::resolve_epsg`] — the root authority clause
    /// first, then the name fallbacks — and is [`EPSG_UNKNOWN`] when the
    /// string named none. The text itself is kept (bounded; see the type
    /// docs), so a refusal and the layer panel can both quote the CRS's name.
    #[must_use]
    pub fn from_wkt(wkt: &str) -> Self {
        let epsg = wkt::resolve_epsg(wkt).unwrap_or(EPSG_UNKNOWN);
        Self {
            epsg,
            wkt: retained_wkt(wkt),
        }
    }

    /// A CRS with both parts already decided — for a format that states a code
    /// *and* carries WKT (a GeoPackage `gpkg_spatial_ref_sys` row, say).
    #[must_use]
    pub fn new(epsg: u32, wkt: Option<&str>) -> Self {
        Self {
            epsg,
            wkt: wkt.and_then(retained_wkt),
        }
    }

    /// The EPSG code, or [`EPSG_UNKNOWN`] when the source named none.
    #[must_use]
    pub fn epsg(&self) -> u32 {
        self.epsg
    }

    /// The WKT text the CRS came from, when one was carried.
    #[must_use]
    pub fn wkt(&self) -> Option<&str> {
        self.wkt.as_deref()
    }

    /// Whether this is WGS 84 geographic — the pass-through case.
    #[must_use]
    pub fn is_wgs84(&self) -> bool {
        self.epsg == EPSG_WGS84
    }

    /// Whether this build can place data in this CRS.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        epsg::is_supported(self.epsg)
    }

    /// What this build knows about the CRS, or [`None`] when it is one this
    /// build will not place.
    #[must_use]
    pub fn definition(&self) -> Option<CrsDef> {
        epsg::definition(self.epsg)
    }

    /// The CRS's name: the registry's, else the WKT's, else `"EPSG:<code>"`,
    /// else `"unknown CRS"`.
    #[must_use]
    pub fn name(&self) -> String {
        if let Some(def) = self.definition() {
            return def.name;
        }
        if let Some(text) = self.wkt.as_deref() {
            let label = wkt::crs_label(text);
            if !label.is_empty() {
                return label;
            }
        }
        if self.epsg == EPSG_UNKNOWN {
            return "unknown CRS".to_string();
        }
        format!("EPSG:{}", self.epsg)
    }

    /// `"<name> (EPSG:<code>)"`, or just the name when there is no code — what
    /// a status line or layer panel shows.
    #[must_use]
    pub fn label(&self) -> String {
        let name = self.name();
        // `name()` falls back to `"EPSG:<code>"` for a code nothing can name;
        // appending the code again there would read "EPSG:2154 (EPSG:2154)".
        if self.epsg == EPSG_UNKNOWN || name == format!("EPSG:{}", self.epsg) {
            name
        } else {
            format!("{name} (EPSG:{})", self.epsg)
        }
    }

    /// The sentence a driver shows when it refuses this CRS.
    ///
    /// Names the CRS and its code, because "unsupported CRS" on its own gives
    /// a user nothing to act on — they need to know *which* CRS to reproject
    /// away from.
    #[must_use]
    pub fn unsupported_message(&self) -> String {
        format!(
            "unsupported CRS \u{201c}{}\u{201d}; reproject the data to WGS 84 (EPSG:4326) first",
            self.label(),
        )
    }

    /// A [`Reprojector`] taking this CRS's coordinates onto WGS 84 lon/lat.
    ///
    /// # Errors
    ///
    /// [`ReprojectError::UnsupportedCrs`] for a CRS this build will not place;
    /// its message is [`Crs::unsupported_message`].
    pub fn reprojector(&self) -> Result<Reprojector, ReprojectError> {
        Reprojector::for_crs(self)
    }

    /// The same CRS with its WKT dropped when the EPSG code alone identifies
    /// it — what a [`crate::layer::Layer`] records.
    ///
    /// A code [`epsg::definition`] knows names the CRS completely: the
    /// registry supplies the name, the datum and every projection parameter,
    /// so the source's WKT adds nothing a reader would consult. Dropping it
    /// keeps a project file from carrying several hundred bytes of `PROJCS[…]`
    /// per layer, and it makes the two views a driver hands out — the layer's
    /// record and the reprojector's `source_epsg` — agree by construction.
    ///
    /// The WKT is kept for a code the registry does **not** know, because
    /// there it is the only thing that can name the CRS in a refusal.
    #[must_use]
    pub fn compact(self) -> Self {
        if self.is_supported() {
            Self {
                epsg: self.epsg,
                wkt: None,
            }
        } else {
            self
        }
    }

    /// A note about the accuracy of this CRS's route to WGS 84, when there is
    /// one worth showing (the historic datums carry metre-level residuals).
    #[must_use]
    pub fn accuracy_note(&self) -> Option<&'static str> {
        self.definition()?.datum.accuracy_note()
    }
}

impl Default for Crs {
    fn default() -> Self {
        Self::wgs84()
    }
}

impl std::fmt::Display for Crs {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.label())
    }
}

/// The WKT text to keep for a source string, applying [`Crs::MAX_WKT_BYTES`].
fn retained_wkt(wkt: &str) -> Option<String> {
    let trimmed = wkt.trim_start_matches('\u{feff}').trim();
    if trimmed.is_empty() || trimmed.len() > Crs::MAX_WKT_BYTES {
        return None;
    }
    Some(trimmed.to_string())
}

impl Serialize for Crs {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // One field when there is no WKT, so the common case is `{"epsg":6677}`.
        let fields = 1 + usize::from(self.wkt.is_some());
        let mut state = serializer.serialize_struct("Crs", fields)?;
        state.serialize_field("epsg", &self.epsg)?;
        if let Some(wkt) = self.wkt.as_deref() {
            state.serialize_field("wkt", wkt)?;
        }
        state.end()
    }
}

/// The wire shape, used only to drive [`Crs`]'s `Deserialize`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CrsWire {
    #[serde(default = "wgs84_code")]
    epsg: u32,
    #[serde(default)]
    wkt: Option<String>,
}

fn wgs84_code() -> u32 {
    EPSG_WGS84
}

impl<'de> Deserialize<'de> for Crs {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = CrsWire::deserialize(deserializer)?;
        Ok(Self {
            epsg: wire.epsg,
            wkt: wire.wkt.as_deref().and_then(retained_wkt),
        })
    }
}

#[cfg(test)]
mod tests;
