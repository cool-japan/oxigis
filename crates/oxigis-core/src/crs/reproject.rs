// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Reprojection to WGS 84 lon/lat — the one seam every ingest path in OxiGIS
//! runs its coordinates through.
//!
//! # The contract
//!
//! Build a [`Reprojector`] **once** per dataset from that dataset's [`Crs`],
//! then call [`Reprojector::to_lon_lat`] per vertex. Construction is where a
//! CRS this build cannot place is refused, by name and by code; the per-vertex
//! call is infallible-ish (it answers [`None`] only for a point the projection
//! genuinely cannot invert, e.g. a `-1e38` no-data sentinel) and allocates
//! nothing.
//!
//! The type is [`Copy`] and holds nothing but numbers, so the drivers that had
//! a `Copy` two-variant enum threaded through every geometry function keep
//! passing it by value.
//!
//! # What does the mathematics
//!
//! `oxigeo`'s `proj` feature — the one finding 6 observed was compiled into
//! every consumer and called by nobody. Specifically:
//!
//! * `oxigeo_proj::transform::cylindrical::GaussKruger`, an **ellipsoidal**
//!   Transverse Mercator (Snyder §8 series). Validated in this module's tests
//!   against the EPSG Guidance Note 7-2 worked example for the British
//!   National Grid, which it reproduces to under 2 cm.
//! * `oxigeo_proj::datum_transform::BursaWolfParams::transform_geodetic` for
//!   the historic datums, which routes geodetic → ECEF → 7-parameter Helmert →
//!   ECEF → geodetic.
//!
//! What OxiGeo is deliberately **not** asked for is *identification*: its
//! embedded EPSG table answers `lookup_epsg(6677)` with "JGD2011 / UTM zone
//! 59N" and gives every Japan Plane Rectangular zone a zero latitude of
//! origin. See [`super::epsg`]'s module docs. Parameters come from OxiGIS's
//! own table; only the series evaluation is delegated.
//!
//! Two projections are handled here rather than through OxiGeo:
//!
//! * geographic sources on a WGS 84-aligned datum, which are the identity;
//! * Web Mercator, whose closed-form inverse is reproduced arithmetic-for-
//!   arithmetic from `oxigis_render::MercatorPoint::to_lon_lat` so the values
//!   this crate produces are bit-identical to the ones the renderer has always
//!   produced for EPSG:3857 input.

use oxigeo::proj::datum_transform::{BursaWolfParams, Ellipsoid};
use oxigeo::proj::transform::cylindrical::GaussKruger;

use crate::crs::Crs;
use crate::crs::datum::Datum;
use crate::crs::epsg::{self, LonLatBounds, Projection};

/// Semi-major axis of the sphere Web Mercator is defined on, metres.
///
/// Identical to `oxigis_render::EARTH_RADIUS_M`; restated rather than imported
/// because `oxigis-core` deliberately does not depend on the render crate.
const WEB_MERCATOR_RADIUS_M: f64 = 6_378_137.0;

/// Why a [`Reprojector`] could not be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReprojectError {
    /// The source CRS is one this build will not place.
    ///
    /// Carries the code and whatever name the source declared, so the message
    /// a user sees names the CRS rather than saying "unsupported".
    #[error("{message}")]
    UnsupportedCrs {
        /// The EPSG code the source declared, or `0` when it declared none.
        epsg: u32,
        /// The full sentence to show, e.g. `unsupported CRS “RGF93 /
        /// Lambert-93” (EPSG:2154)`.
        message: String,
    },
}

impl ReprojectError {
    /// The EPSG code the refusal is about (`0` when the source named none).
    #[must_use]
    pub fn epsg(&self) -> u32 {
        match self {
            Self::UnsupportedCrs { epsg, .. } => *epsg,
        }
    }
}

/// Which ordinate of a projected coordinate pair comes first.
///
/// EPSG defines Japan's plane-rectangular CRSs with **northing first**
/// (`AXIS["Northing",NORTH],AXIS["Easting",EAST]` — the WKT1 fixture in
/// [`super::wkt`]'s tests says exactly that), while every file format OxiGIS
/// reads — the Shapefile `.shp` record, a GeoPackage geometry blob, a
/// GeoParquet WKB — stores an `(x, y)` pair whose `x` is the easting. The
/// formats win: [`AxisOrder::EastingNorthing`] is the default, because that is
/// what the bytes on disk actually hold.
///
/// [`Reprojector::choose_axis_order`] exists for the datasets where that is
/// not true, and is deliberately conservative — see its docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AxisOrder {
    /// `(x, y)` is `(easting, northing)` — what every file format writes.
    #[default]
    EastingNorthing,
    /// `(x, y)` is `(northing, easting)` — EPSG's declared order for the
    /// Japanese plane-rectangular systems, and for some WKT2-driven writers.
    NorthingEasting,
}

/// The Transverse Mercator parameters, kept as plain numbers so the whole
/// [`Reprojector`] stays [`Copy`].
///
/// Rebuilding OxiGeo's `GaussKruger` per call costs one struct literal and one
/// `e² = 2f − f²`; the projection does all of its series work inside
/// `inverse`, so there is nothing to cache and nothing to allocate.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TmercParams {
    latitude_of_origin_deg: f64,
    central_meridian_deg: f64,
    scale_factor: f64,
    false_easting: f64,
    false_northing: f64,
    semi_major_axis_m: f64,
    flattening: f64,
}

impl TmercParams {
    fn to_gauss_kruger(self) -> GaussKruger {
        GaussKruger::with_ellipsoid(
            self.central_meridian_deg,
            self.latitude_of_origin_deg,
            self.scale_factor,
            self.false_easting,
            self.false_northing,
            self.semi_major_axis_m,
            self.flattening,
        )
    }
}

/// The datum step, when the source datum is not WGS 84-aligned.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DatumStep {
    helmert: BursaWolfParams,
    source_ellipsoid: Ellipsoid,
}

impl DatumStep {
    /// Builds the step for `datum`, or [`None`] when it needs none.
    fn for_datum(datum: Datum) -> Option<Self> {
        Some(Self {
            helmert: datum.to_wgs84_helmert()?,
            source_ellipsoid: datum.ellipsoid().to_oxigeo(),
        })
    }

    /// Shifts one geodetic lon/lat (degrees) onto WGS 84.
    fn apply(self, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
        let (lat, lon, _height) = self.helmert.transform_geodetic(
            lat_deg.to_radians(),
            lon_deg.to_radians(),
            0.0,
            &self.source_ellipsoid,
            &Ellipsoid::WGS84,
        );
        (lon.to_degrees(), lat.to_degrees())
    }
}

/// What a [`Reprojector`] actually does per vertex.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Operation {
    /// Already WGS 84 lon/lat — the values pass through untouched.
    Identity,
    /// Geographic on a historic datum: a Helmert shift and nothing else.
    DatumShift(DatumStep),
    /// Web Mercator metres.
    WebMercator,
    /// Transverse Mercator metres, with an optional datum step afterwards.
    TransverseMercator(TmercParams, Option<DatumStep>),
}

/// Maps one dataset's coordinates onto WGS 84 longitude/latitude.
///
/// Build it with [`Reprojector::for_crs`]; see the module docs for the
/// once-per-dataset contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reprojector {
    epsg: u32,
    operation: Operation,
    axis: AxisOrder,
    bounds: LonLatBounds,
}

impl Reprojector {
    /// The identity — a dataset already in WGS 84 lon/lat.
    #[must_use]
    pub fn wgs84() -> Self {
        Self {
            epsg: 4326,
            operation: Operation::Identity,
            axis: AxisOrder::EastingNorthing,
            bounds: LonLatBounds::WORLD,
        }
    }

    /// Builds the reprojection for a source in `crs`.
    ///
    /// # Errors
    ///
    /// [`ReprojectError::UnsupportedCrs`] when the CRS is one this build will
    /// not place — carrying a message that names the CRS and its EPSG code, so
    /// the caller can show it verbatim.
    pub fn for_crs(crs: &Crs) -> Result<Self, ReprojectError> {
        let Some(def) = epsg::definition(crs.epsg()) else {
            return Err(ReprojectError::UnsupportedCrs {
                epsg: crs.epsg(),
                message: crs.unsupported_message(),
            });
        };
        let datum_step = DatumStep::for_datum(def.datum);
        let operation = match def.projection {
            Projection::Geographic => match datum_step {
                None => Operation::Identity,
                Some(step) => Operation::DatumShift(step),
            },
            Projection::WebMercator => Operation::WebMercator,
            Projection::TransverseMercator {
                latitude_of_origin_deg,
                central_meridian_deg,
                scale_factor,
                false_easting,
                false_northing,
            } => Operation::TransverseMercator(
                TmercParams {
                    latitude_of_origin_deg,
                    central_meridian_deg,
                    scale_factor,
                    false_easting,
                    false_northing,
                    semi_major_axis_m: def.datum.ellipsoid().semi_major_axis_m(),
                    flattening: def.datum.ellipsoid().flattening(),
                },
                datum_step,
            ),
        };
        Ok(Self {
            epsg: def.epsg,
            operation,
            axis: AxisOrder::EastingNorthing,
            bounds: def.bounds,
        })
    }

    /// The EPSG code this reprojection reads from.
    #[must_use]
    pub fn source_epsg(&self) -> u32 {
        self.epsg
    }

    /// Whether coordinates pass through unchanged — true only for a WGS 84
    /// geographic source, so a caller can keep a zero-cost fast path.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        matches!(self.operation, Operation::Identity)
    }

    /// Whether the source is geographic (its `(x, y)` are already degrees).
    #[must_use]
    pub fn is_geographic_source(&self) -> bool {
        matches!(
            self.operation,
            Operation::Identity | Operation::DatumShift(_)
        )
    }

    /// The axis order this reprojection reads its input in.
    #[must_use]
    pub fn axis_order(&self) -> AxisOrder {
        self.axis
    }

    /// The same reprojection reading `(x, y)` in the given order.
    #[must_use]
    pub fn with_axis_order(mut self, axis: AxisOrder) -> Self {
        self.axis = axis;
        self
    }

    /// The plausibility envelope of the source CRS, in WGS 84 degrees.
    #[must_use]
    pub fn bounds(&self) -> LonLatBounds {
        self.bounds
    }

    /// Maps one source coordinate pair to `(longitude, latitude)` in degrees.
    ///
    /// [`None`] for a pair the projection cannot invert — a non-finite input
    /// (some writers use `-1e38` as a no-data marker) or a point the series
    /// takes off the ellipsoid. Callers drop such vertices, which is what they
    /// already did for a non-finite result.
    #[must_use]
    pub fn to_lon_lat(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let (easting, northing) = match self.axis {
            AxisOrder::EastingNorthing => (x, y),
            AxisOrder::NorthingEasting => (y, x),
        };
        let (lon, lat) = match self.operation {
            // A geographic source is lon/lat already: the axis swap above is
            // the only reordering it can need, and `AxisOrder` for a
            // geographic CRS means (lat, lon) — handled by the same swap.
            Operation::Identity => (easting, northing),
            Operation::DatumShift(step) => step.apply(easting, northing),
            Operation::WebMercator => web_mercator_to_lon_lat(easting, northing),
            Operation::TransverseMercator(params, step) => {
                let (lon, lat) = params.to_gauss_kruger().inverse(easting, northing).ok()?;
                match step {
                    None => (lon, lat),
                    Some(step) => step.apply(lon, lat),
                }
            }
        };
        (lon.is_finite() && lat.is_finite()).then_some((lon, lat))
    }

    /// Picks the axis order that places `samples` inside the source CRS's
    /// declared area of use.
    ///
    /// Deliberately conservative, and deliberately not a refusal: it swaps
    /// **only** when every usable sample lands outside the envelope as read
    /// and inside it when transposed. A dataset whose two readings are both
    /// plausible — which is the normal case for a Japan Plane Rectangular
    /// zone, where a 6 km easting and a 35 km northing both sit well within
    /// the zone — keeps the default [`AxisOrder::EastingNorthing`], because
    /// that is what the file formats write (see [`AxisOrder`]).
    ///
    /// Reads at most 64 samples, so a caller may hand it a whole dataset.
    #[must_use]
    pub fn choose_axis_order(&self, samples: &[(f64, f64)]) -> AxisOrder {
        const MAX_SAMPLES: usize = 64;
        let straight = self.with_axis_order(AxisOrder::EastingNorthing);
        let swapped = self.with_axis_order(AxisOrder::NorthingEasting);
        let mut usable = 0_u32;
        let mut straight_inside = 0_u32;
        let mut swapped_inside = 0_u32;
        for (x, y) in samples.iter().take(MAX_SAMPLES) {
            let Some((lon_a, lat_a)) = straight.to_lon_lat(*x, *y) else {
                continue;
            };
            usable = usable.saturating_add(1);
            if self.bounds.contains(lon_a, lat_a) {
                straight_inside = straight_inside.saturating_add(1);
            }
            if let Some((lon_b, lat_b)) = swapped.to_lon_lat(*x, *y)
                && self.bounds.contains(lon_b, lat_b)
            {
                swapped_inside = swapped_inside.saturating_add(1);
            }
        }
        if usable > 0 && straight_inside == 0 && swapped_inside == usable {
            AxisOrder::NorthingEasting
        } else {
            AxisOrder::EastingNorthing
        }
    }
}

/// Web Mercator metres to lon/lat degrees.
///
/// Arithmetic-for-arithmetic the same expression as
/// `oxigis_render::MercatorPoint::to_lon_lat`, so a 3857 dataset loaded
/// through this crate produces the identical `f64`s the renderer's own
/// conversion always has.
fn web_mercator_to_lon_lat(x: f64, y: f64) -> (f64, f64) {
    let lat_rad = 2.0 * (y / WEB_MERCATOR_RADIUS_M).exp().atan() - core::f64::consts::FRAC_PI_2;
    (
        (x / WEB_MERCATOR_RADIUS_M).to_degrees(),
        lat_rad.to_degrees(),
    )
}

#[cfg(test)]
mod tests;
