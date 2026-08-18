// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The map's measuring tape, and the coordinate box that flies the camera
//! somewhere — the maths half, with no `egui` in it.
//!
//! Two gestures a GIS is opened for and this one did not have: *how long is
//! this?* / *how big is this?*, and *take me to 35.68, 139.76*. Both are
//! purely local — no service, no policy decision, no transport — which is why
//! they live in a crate that compiles to `wasm32` and owns nothing.
//!
//! The file is in two halves with a hard line between them. Everything above
//! [`MeasureSession`] is plain arithmetic over [`LonLat`] — no `egui`, no app
//! state — so the geodesy is tested by calling it rather than by driving a UI;
//! the `impl OxigisApp` below it is the thin shell that turns clicks into
//! vertices and paints the result.
//!
//! # What the numbers mean
//!
//! Everything here is measured on the **WGS 84 ellipsoid**, never on the Web
//! Mercator plane the map is drawn in. That distinction is the whole point:
//! Mercator over-states ground distance by `1/cos φ`, which is a factor of two
//! at 60°N and a factor of ten at 84°N, so a ruler that measured screen
//! distance would be wrong by more than it was right across most of the
//! inhabited north.
//!
//! * **Distance** — [`geodesic_distance_m`] is Vincenty's inverse formula on
//!   the ellipsoid, accurate to well under a millimetre over any distance a
//!   map click can span. Vincenty famously fails to converge for
//!   near-antipodal pairs; that case falls back to a great-circle distance on
//!   the authalic sphere, which is the right order of magnitude for a
//!   half-world measurement and is never a `NaN`.
//! * **Area** — [`ring_area_m2`] converts each vertex's geodetic latitude to
//!   its **authalic** latitude and applies the spherical-excess sum on the
//!   sphere of equal surface area ([`authalic_radius_m`]). That conversion is
//!   exact for the graticule cells it is easiest to check against (the sum
//!   telescopes into the closed form `a²·Δλ·(q(φ₂) − q(φ₁))/2`), and for a
//!   general ring it differs from a rigorous geodesic-edge computation only by
//!   `O(f²)` — about one part in 10⁵, i.e. a square metre in ten hectares.
//!
//! The on-screen scale bar is deliberately **not** here: it describes the
//! projection the map is drawn in rather than the ground truth a tape
//! measures, so it lives with the camera in [`crate::map_view`].
//!
//! # Who owns the map's clicks
//!
//! The measuring tape and the feature editor both want the map's primary
//! click, and a click that digitized a vertex *and* measured one is a gesture
//! nobody asked for. They are therefore mutually exclusive, enforced in both
//! directions: arming the tape switches the editor to
//! [`crate::edit::EditMode::Off`] ([`OxigisApp::set_measuring`]), and arming
//! any edit tool while the tape is out puts the tape away
//! (`OxigisApp::measure_tool`'s first lines). The tape then only ever reads
//! an [`egui::Response`] the editor has already declined, because
//! [`OxigisApp::ui`] calls it after `edit_interact` on a frame where the edit
//! mode is `Off` — which is precisely the frame the editor consumes nothing.
//!
//! # Coordinate entry
//!
//! [`parse_coordinate`] accepts `lat, lon` **and** `lon, lat` and works out
//! which was meant — see its documentation for the exact rule, which is stated
//! there because it is the contract [`GO_TO_HINT`] promises the user.

use std::fmt;

use egui::{Color32, Context, Pos2, Rect, Response, Stroke, Ui, Vec2};
use oxigis_core::crs::EllipsoidKind;
use oxigis_render::{LonLat, MAX_ZOOM};

use crate::app::OxigisApp;

/// The ellipsoid every reading on this map is taken against — the datum the
/// whole application works in (`oxigis-core`'s reprojector delivers WGS 84 and
/// the tile pyramid is Web Mercator over it).
const ELLIPSOID: EllipsoidKind = EllipsoidKind::Wgs84;

/// Semi-major axis `a`, metres.
const SEMI_MAJOR_M: f64 = ELLIPSOID.semi_major_axis_m();

/// Flattening `f`.
const FLATTENING: f64 = 1.0 / ELLIPSOID.inverse_flattening();

/// Semi-minor axis `b = a(1 − f)`, metres.
const SEMI_MINOR_M: f64 = SEMI_MAJOR_M * (1.0 - FLATTENING);

/// First eccentricity squared, `e² = f(2 − f)`.
const ECCENTRICITY_SQ: f64 = FLATTENING * (2.0 - FLATTENING);

/// How many refinement steps [`vincenty_inverse`] takes before it declares the
/// pair non-convergent and the caller falls back to a great circle.
///
/// Vincenty's own paper has the iteration settling in four or five steps for
/// ordinary pairs; the near-antipodal ones it cannot solve at all oscillate
/// forever, so the only job of this number is to bound that. 200 is far past
/// "slow but converging" and still microseconds.
const VINCENTY_MAX_ITERATIONS: usize = 200;

/// Convergence threshold on λ, in radians — Vincenty's own 1e-12, which is
/// about 0.06 mm of ground distance.
const VINCENTY_CONVERGENCE: f64 = 1.0e-12;

/// Upper bound on the vertices one measurement may collect.
///
/// A click-per-vertex tool cannot realistically reach this, which is exactly
/// why it is here: a stuck pointer, an auto-repeating synthetic click or a
/// scripted host must not be able to grow the session without limit. Reaching
/// it stops accepting vertices; nothing already measured is lost.
pub const MAX_MEASURE_VERTICES: usize = 4_096;

/// The hint under the Go-to-coordinate box — the contract
/// [`parse_coordinate`] keeps, in the words the user reads.
pub const GO_TO_HINT: &str = "35.68, 139.76 (lat, lon) or 139.76, 35.68 (lon, lat) \u{2014} whichever number cannot be a \
     latitude decides. N/S/E/W letters override the order.";

/// What the measure plate says before the first click.
pub const MEASURE_INSTRUCTION: &str = "Measure: click to add points, right-click or Esc to end";

/// A distance rendered for a human: metres under a kilometre, kilometres above
/// it, and never a trailing `.0`.
///
/// The switch is at exactly 1 000 m, which is where every slippy map's ruler
/// switches, so `999.4 m` is followed by `1 km` and never by `1000 m`.
#[must_use]
pub fn format_distance(metres: f64) -> String {
    if !metres.is_finite() || metres <= 0.0 {
        return "0 m".to_string();
    }
    if metres < 1_000.0 {
        // Sub-10 m readings keep two decimals, because a 0.44 m reading
        // rounded to one is a 10% statement about a measurement the user took
        // deliberately.
        let decimals = usize::from(metres < 10.0) + 1;
        return format!("{} m", trim_zeros(metres, decimals));
    }
    let km = metres / 1_000.0;
    let decimals = if km < 100.0 { 2 } else { 1 };
    format!("{} km", trim_zeros(km, decimals))
}

/// An area rendered for a human: square metres, hectares, or square
/// kilometres.
///
/// Hectares are not decoration — they are the unit every land registry states
/// a parcel in, and the band they cover (1 ha … 100 ha) is exactly the band a
/// parcel falls in.
#[must_use]
pub fn format_area(square_metres: f64) -> String {
    if !square_metres.is_finite() || square_metres <= 0.0 {
        return "0 m\u{b2}".to_string();
    }
    if square_metres < 10_000.0 {
        return format!("{} m\u{b2}", trim_zeros(square_metres, 1));
    }
    if square_metres < 1_000_000.0 {
        return format!("{} ha", trim_zeros(square_metres / 10_000.0, 2));
    }
    let km2 = square_metres / 1_000_000.0;
    let decimals = if km2 < 100.0 { 2 } else { 1 };
    format!("{} km\u{b2}", trim_zeros(km2, decimals))
}

/// `value` at `decimals` decimal places with a trailing `.0`/`.50` removed, so
/// a round number reads round.
fn trim_zeros(value: f64, decimals: usize) -> String {
    let text = format!("{value:.decimals$}");
    if !text.contains('.') {
        return text;
    }
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Distance between two positions across the WGS 84 ellipsoid, in metres.
///
/// Vincenty's inverse solution, falling back to a great circle on the authalic
/// sphere for the near-antipodal pairs it cannot solve (see the module docs).
/// Non-finite input measures zero rather than propagating a `NaN` into a
/// running total.
#[must_use]
pub fn geodesic_distance_m(from: LonLat, to: LonLat) -> f64 {
    if !is_finite(from) || !is_finite(to) {
        return 0.0;
    }
    match vincenty_inverse(from, to) {
        Some(metres) if metres.is_finite() => metres,
        _ => great_circle_m(from, to),
    }
}

/// Total length of the open polyline through `vertices`, in metres. Fewer than
/// two vertices measure zero.
#[must_use]
pub fn path_length_m(vertices: &[LonLat]) -> f64 {
    vertices
        .windows(2)
        .map(|pair| geodesic_distance_m(pair[0], pair[1]))
        .sum()
}

/// Area enclosed by the ring `vertices`, in square metres — the ring is closed
/// implicitly, so the first vertex need not be repeated at the end.
///
/// Authalic-latitude spherical excess (see the module docs). Always positive:
/// a ring's area does not depend on which way round it was drawn, and the sign
/// a signed-area formula produces is a winding fact rather than a measurement.
/// Fewer than three vertices, or any non-finite vertex, enclose nothing.
#[must_use]
pub fn ring_area_m2(vertices: &[LonLat]) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }
    if !vertices.iter().copied().all(is_finite) {
        return 0.0;
    }
    let pole = authalic_q(1.0);
    if !pole.is_finite() || pole == 0.0 {
        return 0.0;
    }
    let radius = authalic_radius_m();
    let mut excess = 0.0;
    for (index, &start) in vertices.iter().enumerate() {
        // The last edge is the implicit closing one, back to the first vertex.
        let end = vertices.get(index + 1).copied().unwrap_or(vertices[0]);
        // Each Δλ is wrapped into (−π, π] on its own, which is what makes a
        // ring that crosses the antimeridian measure the same as the ring it
        // is a translate of: the sum of the wrapped steps around a closed ring
        // is zero (or ±2π for one that encircles a pole), while the sum of the
        // raw differences is not.
        let delta_lon = wrap_pi((end.lon - start.lon).to_radians());
        excess += delta_lon * (sin_authalic(start.lat, pole) + sin_authalic(end.lat, pole));
    }
    let area = excess * radius * radius / 2.0;
    if area.is_finite() { area.abs() } else { 0.0 }
}

/// The initial bearing from `from` to `to`, in degrees clockwise from north
/// (`0` = north, `90` = east), or [`None`] when the two coincide.
///
/// The spherical formula rather than Vincenty's α₁: over the distances a
/// measuring tape spans the two agree to a small fraction of a degree, and
/// this one is total — it has no iteration that can fail to converge.
#[must_use]
pub fn initial_bearing_deg(from: LonLat, to: LonLat) -> Option<f64> {
    if !is_finite(from) || !is_finite(to) {
        return None;
    }
    let lat1 = from.lat.to_radians();
    let lat2 = to.lat.to_radians();
    let delta_lon = wrap_pi((to.lon - from.lon).to_radians());
    let y = delta_lon.sin() * lat2.cos();
    let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * delta_lon.cos();
    if y == 0.0 && x == 0.0 {
        return None;
    }
    let bearing = y.atan2(x).to_degrees().rem_euclid(360.0);
    bearing.is_finite().then_some(bearing)
}

/// Whether both components of `position` are finite.
fn is_finite(position: LonLat) -> bool {
    position.lon.is_finite() && position.lat.is_finite()
}

/// Wraps an angle in radians into `(−π, π]`.
fn wrap_pi(radians: f64) -> f64 {
    if !radians.is_finite() {
        return 0.0;
    }
    let wrapped = radians % std::f64::consts::TAU;
    if wrapped > std::f64::consts::PI {
        wrapped - std::f64::consts::TAU
    } else if wrapped <= -std::f64::consts::PI {
        wrapped + std::f64::consts::TAU
    } else {
        wrapped
    }
}

/// The authalic function `q(φ)` — twice the area of the ellipsoidal zone from
/// the equator to `φ`, divided by `a²`, taken at `sin φ`.
///
/// `q(1)` is the pole's value, and `2πa²·q(1)` is the ellipsoid's exact
/// surface area, which is what makes [`authalic_radius_m`] the radius of the
/// sphere with that same area.
fn authalic_q(sin_phi: f64) -> f64 {
    let e = ECCENTRICITY_SQ.sqrt();
    let e_sin = e * sin_phi;
    let denominator = 1.0 - e_sin * e_sin;
    if denominator <= 0.0 || e <= 0.0 {
        // A sphere (e = 0) has q = sin φ, and the guard also keeps the log
        // below out of its singularity for any input at all.
        return sin_phi;
    }
    (1.0 - ECCENTRICITY_SQ)
        * (sin_phi / denominator - (1.0 / (2.0 * e)) * ((1.0 - e_sin) / (1.0 + e_sin)).ln())
}

/// `sin β` for geodetic latitude `lat_deg`, where β is the authalic latitude —
/// exactly `q(sin φ)/q(1)`, so the `asin`/`sin` round trip β's definition
/// would need is skipped entirely.
fn sin_authalic(lat_deg: f64, pole: f64) -> f64 {
    (authalic_q(lat_deg.to_radians().sin()) / pole).clamp(-1.0, 1.0)
}

/// Radius of the sphere with the same surface area as the WGS 84 ellipsoid,
/// `R_q = a·√(q(1)/2)` — about 6 371 007.18 m.
#[must_use]
pub fn authalic_radius_m() -> f64 {
    SEMI_MAJOR_M * (authalic_q(1.0) / 2.0).sqrt()
}

/// Great-circle distance on the authalic sphere, in metres — the fallback for
/// the pairs [`vincenty_inverse`] cannot solve.
fn great_circle_m(from: LonLat, to: LonLat) -> f64 {
    let lat1 = from.lat.to_radians();
    let lat2 = to.lat.to_radians();
    let delta_lat = lat2 - lat1;
    let delta_lon = wrap_pi((to.lon - from.lon).to_radians());
    let haversine =
        (delta_lat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
    let central = 2.0 * haversine.clamp(0.0, 1.0).sqrt().asin();
    let metres = authalic_radius_m() * central;
    if metres.is_finite() { metres } else { 0.0 }
}

/// Vincenty's inverse solution: the ellipsoidal distance in metres, or [`None`]
/// when the iteration does not converge inside [`VINCENTY_MAX_ITERATIONS`] —
/// which happens only for near-antipodal pairs, and is the documented
/// limitation of the method rather than a defect here.
fn vincenty_inverse(from: LonLat, to: LonLat) -> Option<f64> {
    let lat1 = from.lat.to_radians();
    let lat2 = to.lat.to_radians();
    let l = wrap_pi((to.lon - from.lon).to_radians());
    // Reduced (parametric) latitudes: the ellipsoid's points projected onto
    // the auxiliary sphere the iteration below works on.
    let u1 = ((1.0 - FLATTENING) * lat1.tan()).atan();
    let u2 = ((1.0 - FLATTENING) * lat2.tan()).atan();
    let (sin_u1, cos_u1) = u1.sin_cos();
    let (sin_u2, cos_u2) = u2.sin_cos();

    let mut lambda = l;
    for _ in 0..VINCENTY_MAX_ITERATIONS {
        let (sin_lambda, cos_lambda) = lambda.sin_cos();
        let sin_sigma = ((cos_u2 * sin_lambda).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
        .sqrt();
        if sin_sigma == 0.0 {
            // Coincident points; the formula's other degenerate case is the
            // antipodal one, which simply never converges.
            return Some(0.0);
        }
        let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        let sigma = sin_sigma.atan2(cos_sigma);
        let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        let cos_sq_alpha = (1.0 - sin_alpha * sin_alpha).clamp(0.0, 1.0);
        // Zero on an equatorial line, where there is no `2σ_m` to speak of —
        // Vincenty's own convention.
        let cos_2sigma_m = if cos_sq_alpha == 0.0 {
            0.0
        } else {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
        };
        let c = FLATTENING / 16.0 * cos_sq_alpha * (4.0 + FLATTENING * (4.0 - 3.0 * cos_sq_alpha));
        let previous = lambda;
        lambda = l
            + (1.0 - c)
                * FLATTENING
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos_2sigma_m
                            + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));
        if !lambda.is_finite() {
            return None;
        }
        if (lambda - previous).abs() < VINCENTY_CONVERGENCE {
            let metres = vincenty_arc(sigma, sin_sigma, cos_sigma, cos_2sigma_m, cos_sq_alpha);
            return (metres.is_finite() && metres >= 0.0).then_some(metres);
        }
    }
    None
}

/// Vincenty's `s = b·A·(σ − Δσ)`, split out so the iteration above reads as
/// the iteration and this reads as the series it feeds.
fn vincenty_arc(
    sigma: f64,
    sin_sigma: f64,
    cos_sigma: f64,
    cos_2sigma_m: f64,
    cos_sq_alpha: f64,
) -> f64 {
    let u_sq = cos_sq_alpha * (SEMI_MAJOR_M * SEMI_MAJOR_M - SEMI_MINOR_M * SEMI_MINOR_M)
        / (SEMI_MINOR_M * SEMI_MINOR_M);
    let a_coefficient =
        1.0 + u_sq / 16_384.0 * (4_096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let b_coefficient = u_sq / 1_024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
    let delta_sigma = b_coefficient
        * sin_sigma
        * (cos_2sigma_m
            + b_coefficient / 4.0
                * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                    - b_coefficient / 6.0
                        * cos_2sigma_m
                        * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                        * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));
    SEMI_MINOR_M * a_coefficient * (sigma - delta_sigma)
}

/// Which axis a parsed number is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    /// A latitude, `-90..=90`.
    Lat,
    /// A longitude, `-180..=180`.
    Lon,
}

/// Why a coordinate string could not be read.
///
/// One variant per thing that can actually be wrong with it, because the
/// dialog shows this text under the box and "invalid input" tells a user
/// nothing about which half of what they typed to look at.
#[derive(Debug, Clone, PartialEq)]
pub enum CoordinateError {
    /// Nothing was typed.
    Empty,
    /// Something other than exactly two numbers was typed.
    NotTwoNumbers,
    /// A token that should have been a number was not.
    NotANumber(String),
    /// Both numbers named the same axis (`35N, 36N`).
    SameAxis,
    /// Both numbers are too large to be a latitude, so neither can be one.
    NoLatitude,
    /// A latitude outside `-90..=90`.
    LatitudeOutOfRange(f64),
    /// A longitude outside `-180..=180`.
    LongitudeOutOfRange(f64),
    /// A hemisphere letter contradicting the sign in front of its number
    /// (`-35 N`).
    SignedHemisphere(String),
}

impl fmt::Display for CoordinateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("Type a coordinate to go to."),
            Self::NotTwoNumbers => {
                formatter.write_str("Type two numbers \u{2014} a latitude and a longitude.")
            }
            Self::NotANumber(token) => {
                write!(formatter, "\u{201c}{token}\u{201d} is not a number.")
            }
            Self::SameAxis => formatter
                .write_str("Both numbers name the same axis; one must be N/S and the other E/W."),
            Self::NoLatitude => formatter.write_str(
                "Neither number can be a latitude: one of them has to be between -90 and 90.",
            ),
            Self::LatitudeOutOfRange(value) => {
                write!(formatter, "Latitude {value} is outside -90 to 90.")
            }
            Self::LongitudeOutOfRange(value) => {
                write!(formatter, "Longitude {value} is outside -180 to 180.")
            }
            Self::SignedHemisphere(token) => write!(
                formatter,
                "\u{201c}{token}\u{201d} carries both a sign and a hemisphere letter."
            ),
        }
    }
}

/// One number as it was typed, with whatever the text said about which axis it
/// is on.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Component {
    /// The signed value in degrees, hemisphere letter already applied.
    value: f64,
    /// The axis the text named, if it named one.
    axis: Option<Axis>,
}

/// Reads a `lat, lon` **or** `lon, lat` pair of decimal degrees.
///
/// Separators are a comma, a semicolon, a slash, a degree sign or plain
/// whitespace, so `35.68, 139.76`, `35.68 139.76` and `35.68N/139.76E` all
/// read the same.
///
/// # Which number is which
///
/// 1. **Hemisphere letters win.** `N`/`S` name a latitude and `E`/`W` a
///    longitude, on either side of their number and in either order, so
///    `139.76E, 35.68N` is unambiguous however it is written. Two letters
///    naming the same axis are refused rather than guessed at.
/// 2. **Otherwise, range decides.** A number outside `-90..=90` cannot be a
///    latitude, so the *other* one is: `139.76, 35.68` reads as `lon, lat`.
/// 3. **When both could be a latitude, the first is.** `35.68, 39.76` is
///    ambiguous in principle and `lat, lon` in practice — it is the order
///    every map site shows and every user pastes. [`GO_TO_HINT`] says so.
///
/// # Errors
///
/// [`CoordinateError`], one variant per thing that can be wrong — see there.
pub fn parse_coordinate(text: &str) -> Result<LonLat, CoordinateError> {
    let [first, second] = parse_components(text)?;
    let (lat, lon) = match (first.axis, second.axis) {
        (Some(Axis::Lat), Some(Axis::Lat)) | (Some(Axis::Lon), Some(Axis::Lon)) => {
            return Err(CoordinateError::SameAxis);
        }
        (Some(Axis::Lat), _) | (_, Some(Axis::Lon)) => (first.value, second.value),
        (Some(Axis::Lon), _) | (_, Some(Axis::Lat)) => (second.value, first.value),
        (None, None) => {
            match (first.value.abs() <= 90.0, second.value.abs() <= 90.0) {
                // Rule 3: an ambiguous pair reads as `lat, lon`.
                (true, _) => (first.value, second.value),
                (false, true) => (second.value, first.value),
                (false, false) => return Err(CoordinateError::NoLatitude),
            }
        }
    };
    if !(-90.0..=90.0).contains(&lat) {
        return Err(CoordinateError::LatitudeOutOfRange(lat));
    }
    if !(-180.0..=180.0).contains(&lon) {
        return Err(CoordinateError::LongitudeOutOfRange(lon));
    }
    Ok(LonLat::new(lon, lat))
}

/// Splits `text` into exactly two numbers, applying any hemisphere letters.
fn parse_components(text: &str) -> Result<[Component; 2], CoordinateError> {
    let cleaned: String = text
        .chars()
        .map(|character| match character {
            ',' | ';' | '/' | '\u{b0}' => ' ',
            other => other,
        })
        .collect();
    let tokens: Vec<&str> = cleaned.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(CoordinateError::Empty);
    }
    let mut components: Vec<Component> = Vec::with_capacity(2);
    for token in tokens {
        // A hemisphere letter standing on its own belongs to the number
        // before it: `35.68 N, 139.76 E`.
        if let Some(axis) = hemisphere(token) {
            match components.last_mut() {
                Some(component) if component.axis.is_none() => {
                    apply_hemisphere(component, axis, token)?;
                    continue;
                }
                _ => return Err(CoordinateError::NotANumber(token.to_string())),
            }
        }
        if components.len() == 2 {
            return Err(CoordinateError::NotTwoNumbers);
        }
        components.push(parse_component(token)?);
    }
    <[Component; 2]>::try_from(components).map_err(|_| CoordinateError::NotTwoNumbers)
}

/// The axis a lone `N`/`S`/`E`/`W` token names, if that is what it is.
fn hemisphere(token: &str) -> Option<Axis> {
    match token {
        "N" | "n" | "S" | "s" => Some(Axis::Lat),
        "E" | "e" | "W" | "w" => Some(Axis::Lon),
        _ => None,
    }
}

/// Whether a hemisphere letter points at the negative half of its axis.
fn is_negative_hemisphere(token: &str) -> bool {
    matches!(token, "S" | "s" | "W" | "w")
}

/// Records `axis` on `component`, refusing the contradiction of a signed
/// number that also carries a hemisphere letter.
fn apply_hemisphere(
    component: &mut Component,
    axis: Axis,
    token: &str,
) -> Result<(), CoordinateError> {
    if component.value < 0.0 {
        return Err(CoordinateError::SignedHemisphere(token.to_string()));
    }
    component.axis = Some(axis);
    if is_negative_hemisphere(token) {
        component.value = -component.value;
    }
    Ok(())
}

/// Reads one number, with an optional hemisphere letter stuck to either end.
///
/// A letter is only stripped when **what is left parses as a number**, which
/// is what keeps a word out of the coordinate grammar: `nowhere` begins with
/// `n` and ends with `e`, and a stripper that trusted its own letters would
/// report those two as contradicting hemispheres rather than saying the plain
/// truth, which is that `nowhere` is not a number. The candidates are
/// therefore tried in specificity order and the first one that leaves a number
/// behind wins; if none does, the whole token is reported as it was typed.
fn parse_component(token: &str) -> Result<Component, CoordinateError> {
    let first = token.chars().next();
    let last = token.chars().next_back();
    let leading = first.and_then(|character| {
        hemisphere(character.encode_utf8(&mut [0_u8; 4]))
            .map(|axis| (axis, character, &token[character.len_utf8()..]))
    });
    let trailing = last.and_then(|character| {
        // `token.len() > character.len_utf8()` excludes the one-character
        // token, whose leading and trailing letter are the same letter — a
        // lone `N` is a hemisphere marker for the *previous* number, handled
        // in `parse_components`, and never a number of its own.
        (token.len() > character.len_utf8()).then_some(())?;
        hemisphere(character.encode_utf8(&mut [0_u8; 4])).map(|axis| {
            (
                axis,
                character,
                &token[..token.len() - character.len_utf8()],
            )
        })
    });
    // Bare number first: `35.68` must never be read as a hemisphere-marked
    // anything, and `5E3` is scientific notation before it is "5 east".
    if let Some(value) = finite_number(token) {
        return Ok(Component { value, axis: None });
    }
    for (axis, letter, body) in [leading, trailing].into_iter().flatten() {
        let Some(value) = finite_number(body) else {
            continue;
        };
        if value < 0.0 {
            return Err(CoordinateError::SignedHemisphere(token.to_string()));
        }
        let negate = is_negative_hemisphere(letter.encode_utf8(&mut [0_u8; 4]));
        return Ok(Component {
            value: if negate { -value } else { value },
            axis: Some(axis),
        });
    }
    Err(CoordinateError::NotANumber(token.to_string()))
}

/// `text` as a finite `f64`, or [`None`] — the one place a number enters the
/// coordinate grammar, so `inf` and `NaN` are refused in exactly one place.
fn finite_number(text: &str) -> Option<f64> {
    let value = text.trim().parse::<f64>().ok()?;
    value.is_finite().then_some(value)
}

/// One measurement in progress (or just finished): the vertices the user
/// clicked, and whether the tape is still taking more.
///
/// Deliberately free of `egui`: everything the readout says is derived here,
/// which is what lets the geodesy be tested without a UI harness.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeasureSession {
    /// Whether the tool is armed at all — the View ▸ Measure toggle.
    active: bool,
    /// The clicked vertices, in click order.
    vertices: Vec<LonLat>,
    /// Whether the measurement is closed: `Esc`/right-click has ended it, so
    /// the numbers stay on screen but no further click extends them.
    finished: bool,
}

impl MeasureSession {
    /// Whether the tool is armed.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Arms the tool, discarding anything measured before.
    pub fn activate(&mut self) {
        self.active = true;
        self.vertices.clear();
        self.finished = false;
    }

    /// Puts the tool away and forgets the measurement.
    pub fn deactivate(&mut self) {
        self.active = false;
        self.vertices.clear();
        self.finished = false;
    }

    /// Whether the current measurement has been ended.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// The vertices measured so far.
    #[must_use]
    pub fn vertices(&self) -> &[LonLat] {
        &self.vertices
    }

    /// Adds a vertex, unless the measurement is closed, the position is not
    /// finite, or [`MAX_MEASURE_VERTICES`] has been reached. Answers whether
    /// the vertex was taken.
    pub fn push_vertex(&mut self, position: LonLat) -> bool {
        if self.finished || !is_finite(position) || self.vertices.len() >= MAX_MEASURE_VERTICES {
            return false;
        }
        self.vertices.push(position);
        true
    }

    /// Takes the most recent vertex back, re-opening a finished measurement so
    /// the correction can be continued. Answers whether there was one.
    pub fn undo_vertex(&mut self) -> bool {
        if self.vertices.pop().is_none() {
            return false;
        }
        self.finished = false;
        true
    }

    /// Ends the measurement, keeping the result on screen. Answers whether
    /// there was anything to end — a bare tool with no vertices is not a
    /// measurement, and must leave `Esc` for whoever else wants it.
    pub fn finish(&mut self) -> bool {
        if self.finished || self.vertices.is_empty() {
            return false;
        }
        self.finished = true;
        true
    }

    /// Throws the measurement away, keeping the tool armed. Answers whether
    /// there was one.
    pub fn clear(&mut self) -> bool {
        if self.vertices.is_empty() {
            return false;
        }
        self.vertices.clear();
        self.finished = false;
        true
    }

    /// Length of the measured polyline, in metres.
    #[must_use]
    pub fn length_m(&self) -> f64 {
        path_length_m(&self.vertices)
    }

    /// Area of the measured ring, in square metres — zero until three vertices
    /// enclose anything.
    #[must_use]
    pub fn area_m2(&self) -> f64 {
        ring_area_m2(&self.vertices)
    }

    /// The readout line: what the plate over the map says.
    ///
    /// `hover` is where the pointer is while the measurement is still open,
    /// which contributes a live segment the user has not committed yet — the
    /// rubber band's own length, so the number under the cursor is the number
    /// the next click will add.
    ///
    /// [`None`] means there is nothing to say; an armed tool that has not been
    /// clicked yet shows [`MEASURE_INSTRUCTION`] instead.
    #[must_use]
    pub fn readout(&self, hover: Option<LonLat>) -> Option<String> {
        if self.vertices.is_empty() {
            return None;
        }
        let live = if self.finished {
            None
        } else {
            hover.filter(|position| is_finite(*position))
        };
        let mut length = self.length_m();
        if let (Some(hover), Some(&last)) = (live, self.vertices.last()) {
            length += geodesic_distance_m(last, hover);
        }
        let mut line = format!("Length {}", format_distance(length));
        // The ring is only meaningful once three vertices enclose something;
        // the live vertex counts, so the area appears on the same frame the
        // rubber band first closes a triangle.
        let ring: Vec<LonLat> = match live {
            Some(hover) => self.vertices.iter().copied().chain([hover]).collect(),
            None => self.vertices.clone(),
        };
        if ring.len() >= 3 {
            line.push_str(&format!(
                "  \u{b7}  Area {}",
                format_area(ring_area_m2(&ring))
            ));
        }
        // The last leg's own length and bearing: the two numbers a surveyor
        // reads off a tape, and the only ones a running total hides.
        let leg = match (live, self.vertices.last()) {
            (Some(hover), Some(&last)) => Some((last, hover)),
            (None, _) if self.vertices.len() >= 2 => {
                let count = self.vertices.len();
                Some((self.vertices[count - 2], self.vertices[count - 1]))
            }
            _ => None,
        };
        if let Some((from, to)) = leg {
            line.push_str(&format!(
                "  \u{b7}  Segment {}",
                format_distance(geodesic_distance_m(from, to))
            ));
            if let Some(bearing) = initial_bearing_deg(from, to) {
                line.push_str(&format!(" at {bearing:.0}\u{b0}"));
            }
        }
        if self.finished {
            line.push_str("  \u{b7}  ended");
        }
        Some(line)
    }
}

/// The Go-to-coordinate dialog's cross-frame state: the box's text, the zoom
/// the jump lands at, and the last refusal to show under it.
///
/// Public fields, like [`crate::print::PrintOptions`]: this is a form's
/// contents, the window body binds `egui` widgets straight to them, and there
/// is no invariant between them for an accessor to protect.
#[derive(Debug, Clone, PartialEq)]
pub struct GoToDialog {
    /// Whether the window is on screen.
    pub open: bool,
    /// The coordinate box's contents.
    pub text: String,
    /// The zoom level the camera lands at.
    pub zoom: f64,
    /// The most recent parse refusal, shown under the box.
    pub error: Option<String>,
}

impl Default for GoToDialog {
    fn default() -> Self {
        Self {
            open: false,
            text: String::new(),
            // Replaced by the live camera's zoom every time the dialog opens;
            // this is only what an app that never opened it reports.
            zoom: 12.0,
            error: None,
        }
    }
}

/// Radius, in logical pixels, of the dot drawn at each measured vertex.
const VERTEX_RADIUS: f32 = 3.5;

/// Colour of the measuring tape — amber, so it reads over both a dark basemap
/// and a light one without being mistaken for the edit overlay's blue.
const MEASURE_COLOR: Color32 = Color32::from_rgb(0xFF, 0xC1, 0x07);

/// Padding between the readout text and its backing plate, in logical pixels
/// (the attribution plate's, so the two look like one family).
const READOUT_PAD: f32 = 4.0;

/// Gap between the readout plate and the top edge of the map panel.
const READOUT_MARGIN: f32 = 6.0;

/// Font size of the measurement readout, in points.
const READOUT_FONT: f32 = 12.0;

impl OxigisApp {
    /// The View menu's map-tool items: the measuring tape, the scale bar and
    /// the coordinate box.
    ///
    /// Drawn from here rather than inlined into `menu_bar` for the same reason
    /// every other panel body is: the frame loop in `app/mod.rs` stays the
    /// frame loop.
    pub(crate) fn map_tools_menu(&mut self, ui: &mut Ui) {
        let mut measuring = self.measure.is_active();
        if ui
            .checkbox(&mut measuring, "Measure")
            .on_hover_text(
                "Click the map to add points: the running geodesic length \u{2014} and, from \
                 three points on, the enclosed area \u{2014} is shown over the map. Right-click \
                 or Esc ends the measurement.",
            )
            .changed()
        {
            self.set_measuring(measuring);
        }
        let mut scale_bar = self.map_panel.scale_bar_visible();
        if ui
            .checkbox(&mut scale_bar, "Scale bar")
            .on_hover_text("A ground-distance bar in the map's bottom-left corner")
            .changed()
        {
            self.map_panel.set_scale_bar_visible(scale_bar);
        }
        let go_to_keys = ui.ctx().format_shortcut(&egui::KeyboardShortcut::new(
            egui::Modifiers::COMMAND,
            egui::Key::G,
        ));
        if ui
            .add(egui::Button::new("Go to coordinate\u{2026}").shortcut_text(go_to_keys))
            .on_hover_text("Recentres the map on a latitude and longitude you type")
            .clicked()
        {
            self.open_go_to_dialog();
            ui.close();
        }
    }

    /// Arms or puts away the measuring tape.
    ///
    /// Arming it switches the editor off — see the module docs on who owns the
    /// map's clicks.
    pub fn set_measuring(&mut self, measuring: bool) {
        if measuring {
            self.edit.set_mode(crate::edit::EditMode::Off);
            self.measure.activate();
            self.status = Some(MEASURE_INSTRUCTION.to_string());
        } else {
            self.measure.deactivate();
        }
    }

    /// Whether the measuring tape is armed.
    #[must_use]
    pub fn measuring(&self) -> bool {
        self.measure.is_active()
    }

    /// The measurement's readout line as the plate would show it with no
    /// pointer over the map, if there is one — the seam a test reads instead
    /// of a screenshot.
    #[must_use]
    pub fn measure_readout(&self) -> Option<String> {
        self.measure.readout(None)
    }

    /// Adds a measured vertex directly, bypassing the pointer.
    ///
    /// The `Ui`-free entry point a test (or a shell driving the tool from its
    /// own gesture recognizer) uses; the click path in `Self::measure_tool`
    /// is exactly this with the pointer position unprojected first. Answers
    /// whether the vertex was taken — see
    /// [`crate::measure::MeasureSession::push_vertex`] for the three ways it
    /// can be declined.
    pub fn measure_at(&mut self, position: LonLat) -> bool {
        if !self.measure.is_active() {
            return false;
        }
        if self.measure.is_finished() {
            let _cleared = self.measure.clear();
        }
        self.measure.push_vertex(position)
    }

    /// Ends the measurement in progress, keeping its numbers on screen.
    /// Answers whether there was one.
    pub fn finish_measurement(&mut self) -> bool {
        self.measure.finish()
    }

    /// `Esc` while measuring: ends the measurement, then puts the tape away.
    ///
    /// Called from the frame loop **before** the edit shortcuts, because the
    /// edit ladder consumes `Escape` for a retained selection even with its
    /// mode `Off`, and a measurement in progress is the more recent gesture.
    /// Peeked before it is consumed, exactly as `edit_escape` does: a key this
    /// tool has no use for stays available to whoever does.
    pub(crate) fn measure_escape(&mut self, ctx: &Context) {
        if !self.measure.is_active() {
            return;
        }
        if ctx.memory(|memory| memory.focused()).is_some() {
            return;
        }
        if !ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            return;
        }
        // One press ends the measurement, the next puts the tool away — the
        // same "one rung per press" ladder the editor's Escape climbs.
        if !self.measure.finish() {
            self.measure.deactivate();
            self.status = Some("Measure off.".to_string());
        }
        ctx.input_mut(|input| {
            let _consumed = input.consume_key(egui::Modifiers::NONE, egui::Key::Escape);
        });
    }

    /// `Ctrl/Cmd+G`: opens the Go-to-coordinate window.
    ///
    /// Guarded by the same focus check the File shortcuts keep, and consumed
    /// with `consume_shortcut` so the key a menu item also offers is taken
    /// exactly once.
    pub(crate) fn map_tool_shortcuts(&mut self, ctx: &Context) {
        if ctx.memory(|memory| memory.focused()).is_some() {
            return;
        }
        if ctx.input_mut(|input| {
            input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND,
                egui::Key::G,
            ))
        }) {
            self.open_go_to_dialog();
        }
    }

    /// This frame of the measuring tape: takes the click, then paints the tape
    /// and its readout over the map.
    ///
    /// A no-op while the tool is not armed, so the map behaves exactly as it
    /// did before this module existed.
    pub(crate) fn measure_tool(&mut self, ui: &Ui, rect: Rect, response: &Response) {
        if !self.measure.is_active() {
            return;
        }
        // The editor was armed from its own toolbar while the tape was out:
        // the two are mutually exclusive map gestures (see the module docs),
        // and the one the user just asked for wins.
        if self.edit.mode() != crate::edit::EditMode::Off {
            self.measure.deactivate();
            return;
        }
        let view = self.map_panel.view();
        let ppp = ui.ctx().pixels_per_point();
        let to_position = |pos: Pos2| -> LonLat {
            let local = pos - rect.min;
            view.screen_to_lon_lat([local.x * ppp, local.y * ppp])
        };
        if response.clicked()
            && let Some(pos) = response.interact_pointer_pos()
        {
            let position = to_position(pos);
            // A click after the measurement ended starts the next one, which
            // is what reaching for the tape again means.
            let _taken = self.measure_at(position);
        }
        if response.secondary_clicked() {
            let _ended = self.measure.finish();
        }
        let hover = response.hover_pos().map(to_position);
        self.paint_measure(ui, rect, hover);
    }

    /// Paints the tape: the segments, the vertex dots, the rubber band to the
    /// pointer, and the readout plate at the top of the map.
    fn paint_measure(&self, ui: &Ui, rect: Rect, hover: Option<LonLat>) {
        let view = self.map_panel.view();
        let ppp = ui.ctx().pixels_per_point();
        let painter = ui.painter_at(rect);
        let to_screen = |position: LonLat| -> Pos2 {
            let px = view.lon_lat_to_screen(position);
            rect.min + Vec2::new(px[0] / ppp, px[1] / ppp)
        };
        let points: Vec<Pos2> = self
            .measure
            .vertices()
            .iter()
            .copied()
            .map(to_screen)
            .collect();
        let stroke = Stroke::new(2.0, MEASURE_COLOR);
        for pair in points.windows(2) {
            painter.line_segment([pair[0], pair[1]], stroke);
        }
        // The rubber band, and — from two committed vertices on — the closing
        // edge of the ring whose area the readout is quoting, so the number
        // and the shape on screen describe the same shape.
        let live = if self.measure.is_finished() {
            None
        } else {
            hover
        };
        if let (Some(hover), Some(&last)) = (live, points.last()) {
            let hover_pos = to_screen(hover);
            painter.line_segment(
                [last, hover_pos],
                Stroke::new(1.5, MEASURE_COLOR.gamma_multiply(0.75)),
            );
            if points.len() >= 2
                && let Some(&first) = points.first()
            {
                painter.line_segment(
                    [hover_pos, first],
                    Stroke::new(1.0, MEASURE_COLOR.gamma_multiply(0.45)),
                );
            }
        } else if points.len() >= 3
            && let (Some(&first), Some(&last)) = (points.first(), points.last())
        {
            painter.line_segment(
                [last, first],
                Stroke::new(1.0, MEASURE_COLOR.gamma_multiply(0.45)),
            );
        }
        for point in &points {
            painter.circle_filled(*point, VERTEX_RADIUS, MEASURE_COLOR);
            painter.circle_stroke(
                *point,
                VERTEX_RADIUS,
                Stroke::new(1.0, Color32::from_black_alpha(0xC0)),
            );
        }
        let line = self
            .measure
            .readout(hover)
            .unwrap_or_else(|| MEASURE_INSTRUCTION.to_string());
        paint_readout_plate(&painter, rect, &line);
    }

    /// Opens the Go-to-coordinate dialog, seeded with the camera's own zoom so
    /// a jump that only changes position keeps the scale being worked at.
    pub(crate) fn open_go_to_dialog(&mut self) {
        self.go_to.open = true;
        self.go_to.zoom = self.map_panel.view().zoom();
        self.go_to.error = None;
    }

    /// Draws the Go-to-coordinate window while it is open.
    pub(crate) fn go_to_window(&mut self, ctx: &Context) {
        if !self.go_to.open {
            return;
        }
        let mut open = true;
        let mut go = false;
        let mut cancel = false;
        egui::Window::new("Go to coordinate")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Coordinate");
                    let entry = ui.add(
                        egui::TextEdit::singleline(&mut self.go_to.text)
                            .hint_text("35.68, 139.76")
                            .desired_width(200.0),
                    );
                    if entry.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                        go = true;
                    }
                });
                ui.weak(GO_TO_HINT);
                ui.horizontal(|ui| {
                    ui.label("Zoom");
                    ui.add(
                        egui::Slider::new(&mut self.go_to.zoom, 0.0..=f64::from(MAX_ZOOM))
                            .fixed_decimals(1),
                    );
                });
                if let Some(error) = &self.go_to.error {
                    ui.colored_label(ui.visuals().error_fg_color, error);
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Go").clicked() {
                        go = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });
        if go {
            let text = self.go_to.text.clone();
            let zoom = self.go_to.zoom;
            match self.go_to_coordinate(&text, Some(zoom)) {
                Ok(_landed) => {
                    self.go_to.error = None;
                    self.go_to.open = false;
                    return;
                }
                Err(error) => self.go_to.error = Some(error.to_string()),
            }
        }
        self.go_to.open = open && !cancel;
    }

    /// Recentres the map on the coordinate `text` names, at `zoom` when one is
    /// given.
    ///
    /// A camera move, so nothing is recorded: the undo log is for project
    /// state, and the view is not it — exactly as
    /// [`OxigisApp::zoom_to_selected_layer`] documents.
    ///
    /// The status line and the returned position report **where the camera
    /// actually went**, not what was typed: a latitude past the Web Mercator
    /// cut-off (±85.05°) lands at the cut-off, and saying otherwise would
    /// describe a place the map is not showing.
    ///
    /// # Errors
    ///
    /// [`CoordinateError`] when `text` is not a coordinate — see
    /// [`parse_coordinate`], which documents the `lat, lon` / `lon, lat` rule.
    pub fn go_to_coordinate(
        &mut self,
        text: &str,
        zoom: Option<f64>,
    ) -> Result<LonLat, CoordinateError> {
        let target = parse_coordinate(text)?;
        let view = self.map_panel.view();
        let moved = match zoom {
            Some(zoom) => view.with_center(target).with_zoom(zoom),
            None => view.with_center(target),
        };
        self.map_panel.set_view(moved);
        let landed = self.map_panel.view().center();
        self.status = Some(format!(
            "Moved to {:.5}, {:.5} at zoom {:.1}.",
            landed.lat,
            landed.lon,
            self.map_panel.view().zoom()
        ));
        Ok(landed)
    }
}

/// Draws one line of text on a dark plate at the top centre of `rect`, in the
/// same shape as the attribution plate in the opposite corner.
fn paint_readout_plate(painter: &egui::Painter, rect: Rect, line: &str) {
    let galley = painter.layout_no_wrap(
        line.to_string(),
        egui::FontId::proportional(READOUT_FONT),
        Color32::from_rgb(0xF4, 0xF6, 0xF9),
    );
    let plate_size = galley.size() + Vec2::new(READOUT_PAD * 2.0, READOUT_PAD * 1.5);
    let plate = Rect::from_min_size(
        Pos2::new(
            rect.center().x - plate_size.x / 2.0,
            rect.top() + READOUT_MARGIN,
        ),
        plate_size,
    );
    painter.rect_filled(plate, 3.0, Color32::from_black_alpha(0xB4));
    painter.galley(
        plate.min + Vec2::new(READOUT_PAD, READOUT_PAD * 0.75),
        galley,
        Color32::PLACEHOLDER,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One degree of longitude at the equator, in metres — `a·(π/180)`, which
    /// is what the equator being a geodesic of the ellipsoid means.
    fn equatorial_degree_m() -> f64 {
        SEMI_MAJOR_M * std::f64::consts::PI / 180.0
    }

    #[test]
    fn measure_geodesic_distance_along_the_equator_is_the_semi_major_arc() {
        // The equator IS a geodesic, so its length is exactly `a·Δλ` — the one
        // distance on the ellipsoid with a closed form, and therefore the one
        // that catches a wrong axis, a radians slip or a dropped term in the
        // λ iteration.
        let metres = geodesic_distance_m(LonLat::new(0.0, 0.0), LonLat::new(1.0, 0.0));
        assert!(
            (metres - equatorial_degree_m()).abs() < 0.01,
            "expected {} m, got {metres} m",
            equatorial_degree_m()
        );
        let ten = geodesic_distance_m(LonLat::new(-5.0, 0.0), LonLat::new(5.0, 0.0));
        assert!((ten - equatorial_degree_m() * 10.0).abs() < 0.1);
    }

    #[test]
    fn measure_geodesic_distance_along_a_meridian_matches_the_published_quarter_arc() {
        // Pole to equator on WGS 84 is 10 001 965.729 m — the metre's own
        // definition, restated on the modern ellipsoid.
        let metres = geodesic_distance_m(LonLat::new(0.0, 0.0), LonLat::new(0.0, 90.0));
        assert!(
            (metres - 10_001_965.729).abs() < 1.0,
            "expected the quarter meridian, got {metres} m"
        );
    }

    #[test]
    fn measure_geodesic_distance_beats_the_mercator_plane_at_high_latitude() {
        // The reason this is geodesic and not planar: one degree of longitude
        // at 60°N is half what it is at the equator, and a Web Mercator ruler
        // would report the same number at both.
        let equator = geodesic_distance_m(LonLat::new(0.0, 0.0), LonLat::new(1.0, 0.0));
        let north = geodesic_distance_m(LonLat::new(0.0, 60.0), LonLat::new(1.0, 60.0));
        let ratio = north / equator;
        assert!(
            (0.499..0.503).contains(&ratio),
            "1° at 60°N is {ratio} of 1° at the equator"
        );
    }

    #[test]
    fn measure_geodesic_distance_matches_vincentys_own_published_test_line() {
        // Vincenty's 1975 paper worked this line — Flinders Peak to Buninyong,
        // Victoria — and published s = 54 972.271 m. It is the standard
        // conformance vector for an inverse solution, and the one number that
        // pins the whole λ iteration AND the (A, B, Δσ) series at once: a
        // dropped term in either still passes the equator and meridian checks
        // above, because both of those are degenerate cases where the series
        // contributes almost nothing.
        let flinders = LonLat::new(144.424_867_888_888_9, -37.951_033_416_666_67);
        let buninyong = LonLat::new(143.926_495_527_777_8, -37.652_821_138_888_9);
        let metres = geodesic_distance_m(flinders, buninyong);
        assert!(
            (metres - 54_972.271).abs() < 0.001,
            "expected Vincenty's 54 972.271 m, got {metres} m"
        );
    }

    #[test]
    fn measure_geodesic_distance_is_symmetric_and_zero_for_a_point() {
        let tokyo = LonLat::new(139.7671, 35.6812);
        let paris = LonLat::new(2.3522, 48.8566);
        let there = geodesic_distance_m(tokyo, paris);
        let back = geodesic_distance_m(paris, tokyo);
        assert!((there - back).abs() < 1.0e-6, "{there} vs {back}");
        // ~9 739.5 km on the ellipsoid. The spherical haversine answer for the
        // same pair is ~9 715.9 km — 23 km short — so this range is also what
        // would catch a silent fall-through to the great-circle fallback.
        assert!(
            (9_739_000.0..9_740_000.0).contains(&there),
            "unexpected Tokyo-Paris distance {there} m"
        );
        assert_eq!(geodesic_distance_m(tokyo, tokyo), 0.0);
    }

    #[test]
    fn measure_geodesic_distance_survives_antipodes_and_non_finite_input() {
        // Vincenty cannot solve the antipodal case; the answer must still be a
        // finite half-circumference rather than a NaN.
        let metres = geodesic_distance_m(LonLat::new(0.0, 0.0), LonLat::new(180.0, 0.0));
        assert!(metres.is_finite(), "antipodal distance must be finite");
        assert!(
            (metres - std::f64::consts::PI * authalic_radius_m()).abs() < 50_000.0,
            "expected roughly half a circumference, got {metres} m"
        );
        assert_eq!(
            geodesic_distance_m(LonLat::new(f64::NAN, 0.0), LonLat::new(1.0, 0.0)),
            0.0
        );
        assert_eq!(
            geodesic_distance_m(LonLat::new(0.0, 0.0), LonLat::new(f64::INFINITY, 0.0)),
            0.0
        );
    }

    #[test]
    fn measure_path_length_sums_its_segments() {
        let path = [
            LonLat::new(0.0, 0.0),
            LonLat::new(1.0, 0.0),
            LonLat::new(2.0, 0.0),
        ];
        let total = path_length_m(&path);
        assert!((total - equatorial_degree_m() * 2.0).abs() < 0.05);
        assert_eq!(path_length_m(&path[..1]), 0.0);
        assert_eq!(path_length_m(&[]), 0.0);
    }

    #[test]
    fn measure_authalic_radius_reproduces_the_wgs84_surface_area() {
        // 4πR_q² must be the ellipsoid's own surface area, 5.100 656 2×10¹⁴ m²
        // — the check that the `q(1)` constant behind every area reading is
        // right, since a wrong one scales every polygon by the same factor.
        let radius = authalic_radius_m();
        assert!(
            (radius - 6_371_007.181).abs() < 0.5,
            "authalic radius {radius} m"
        );
        let surface = 4.0 * std::f64::consts::PI * radius * radius;
        assert!(
            (surface / 5.100_656_2e14 - 1.0).abs() < 1.0e-6,
            "surface area {surface} m\u{b2}"
        );
    }

    #[test]
    fn measure_ring_area_of_a_small_equatorial_box_matches_the_planar_product() {
        // Small enough that the sphere is flat to well under a part in 10³: a
        // 0.01° square at the equator, ~1.1 km on a side.
        let ring = [
            LonLat::new(0.0, 0.0),
            LonLat::new(0.01, 0.0),
            LonLat::new(0.01, 0.01),
            LonLat::new(0.0, 0.01),
        ];
        let area = ring_area_m2(&ring);
        let width = geodesic_distance_m(LonLat::new(0.0, 0.0), LonLat::new(0.01, 0.0));
        let height = geodesic_distance_m(LonLat::new(0.0, 0.0), LonLat::new(0.0, 0.01));
        let planar = width * height;
        assert!(
            (area / planar - 1.0).abs() < 1.0e-3,
            "ellipsoidal {area} m\u{b2} vs planar {planar} m\u{b2}"
        );
    }

    #[test]
    fn measure_ring_area_shrinks_with_the_cosine_of_latitude() {
        // A 1°×1° cell at 60°N covers about cos 60° = 0.5 of the same cell at
        // the equator: the check that the latitude conversion is a conversion
        // and not an identity.
        let equator = ring_area_m2(&[
            LonLat::new(0.0, 0.0),
            LonLat::new(1.0, 0.0),
            LonLat::new(1.0, 1.0),
            LonLat::new(0.0, 1.0),
        ]);
        let north = ring_area_m2(&[
            LonLat::new(0.0, 60.0),
            LonLat::new(1.0, 60.0),
            LonLat::new(1.0, 61.0),
            LonLat::new(0.0, 61.0),
        ]);
        let ratio = north / equator;
        assert!(
            (0.48..0.52).contains(&ratio),
            "a 1° cell at 60°N is {ratio} of the equatorial one"
        );
        // And the equatorial cell itself is the ~12 308 km² every gazetteer
        // quotes.
        assert!(
            (equator / 1.0e6 - 12_308.0).abs() < 30.0,
            "equatorial cell {equator} m\u{b2}"
        );
    }

    #[test]
    fn measure_ring_area_ignores_winding_and_degenerate_rings() {
        let ring = [
            LonLat::new(0.0, 0.0),
            LonLat::new(0.5, 0.0),
            LonLat::new(0.5, 0.5),
            LonLat::new(0.0, 0.5),
        ];
        let mut reversed = ring;
        reversed.reverse();
        let forward = ring_area_m2(&ring);
        let backward = ring_area_m2(&reversed);
        assert!(forward > 0.0);
        assert!((forward - backward).abs() < forward * 1.0e-9);
        assert_eq!(ring_area_m2(&ring[..2]), 0.0);
        assert_eq!(
            ring_area_m2(&[
                LonLat::new(0.0, 0.0),
                LonLat::new(f64::NAN, 0.0),
                LonLat::new(1.0, 1.0)
            ]),
            0.0
        );
    }

    #[test]
    fn measure_ring_area_is_unchanged_by_crossing_the_antimeridian() {
        // The same box, once around 0° and once around 180°: wrapping each Δλ
        // into (−π, π] is what makes those two the same measurement.
        let home = ring_area_m2(&[
            LonLat::new(-0.5, 10.0),
            LonLat::new(0.5, 10.0),
            LonLat::new(0.5, 11.0),
            LonLat::new(-0.5, 11.0),
        ]);
        let dateline = ring_area_m2(&[
            LonLat::new(179.5, 10.0),
            LonLat::new(-179.5, 10.0),
            LonLat::new(-179.5, 11.0),
            LonLat::new(179.5, 11.0),
        ]);
        assert!(
            (home - dateline).abs() < home * 1.0e-9,
            "{home} m\u{b2} vs {dateline} m\u{b2}"
        );
    }

    #[test]
    fn measure_bearing_points_the_right_way_round_the_compass() {
        let origin = LonLat::new(0.0, 0.0);
        let north = initial_bearing_deg(origin, LonLat::new(0.0, 1.0));
        let east = initial_bearing_deg(origin, LonLat::new(1.0, 0.0));
        let west = initial_bearing_deg(origin, LonLat::new(-1.0, 0.0));
        assert!(north.is_some_and(|value| value.abs() < 1.0e-6));
        assert!(east.is_some_and(|value| (value - 90.0).abs() < 1.0e-6));
        assert!(west.is_some_and(|value| (value - 270.0).abs() < 1.0e-6));
        assert_eq!(initial_bearing_deg(origin, origin), None);
    }

    #[test]
    fn measure_format_distance_switches_from_metres_to_kilometres_at_one_km() {
        assert_eq!(format_distance(0.0), "0 m");
        assert_eq!(format_distance(0.44), "0.44 m");
        assert_eq!(format_distance(845.0), "845 m");
        assert_eq!(format_distance(999.4), "999.4 m");
        assert_eq!(format_distance(1_000.0), "1 km");
        assert_eq!(format_distance(12_500.0), "12.5 km");
        assert_eq!(format_distance(1_234_500.0), "1234.5 km");
        assert_eq!(format_distance(f64::NAN), "0 m");
    }

    #[test]
    fn measure_format_area_uses_hectares_between_a_hectare_and_a_square_km() {
        assert_eq!(format_area(0.0), "0 m\u{b2}");
        assert_eq!(format_area(950.0), "950 m\u{b2}");
        assert_eq!(format_area(10_000.0), "1 ha");
        assert_eq!(format_area(123_400.0), "12.34 ha");
        assert_eq!(format_area(1_000_000.0), "1 km\u{b2}");
        assert_eq!(format_area(2_500_000.0), "2.5 km\u{b2}");
        assert_eq!(format_area(f64::INFINITY), "0 m\u{b2}");
    }

    #[test]
    fn measure_session_collects_finishes_and_clears() {
        let mut session = MeasureSession::default();
        assert!(!session.is_active());
        session.activate();
        assert!(session.is_active());
        assert!(session.readout(None).is_none());
        assert!(session.push_vertex(LonLat::new(0.0, 0.0)));
        assert!(session.push_vertex(LonLat::new(1.0, 0.0)));
        assert_eq!(session.vertices().len(), 2);
        assert!((session.length_m() - equatorial_degree_m()).abs() < 0.05);
        assert_eq!(session.area_m2(), 0.0, "two points enclose nothing");
        assert!(
            session
                .readout(None)
                .is_some_and(|line| line.contains("Length"))
        );
        assert!(session.finish());
        assert!(session.is_finished());
        // A finished measurement takes no more vertices.
        assert!(!session.push_vertex(LonLat::new(2.0, 0.0)));
        assert!(!session.finish());
        assert!(session.undo_vertex());
        assert!(
            !session.is_finished(),
            "correcting re-opens the measurement"
        );
        assert!(session.clear());
        assert!(session.vertices().is_empty());
        assert!(!session.clear());
        session.deactivate();
        assert!(!session.is_active());
    }

    #[test]
    fn measure_session_refuses_non_finite_vertices_and_stays_bounded() {
        let mut session = MeasureSession::default();
        session.activate();
        assert!(!session.push_vertex(LonLat::new(f64::NAN, 0.0)));
        assert!(session.vertices().is_empty());
        for index in 0..MAX_MEASURE_VERTICES {
            assert!(session.push_vertex(LonLat::new(0.001 * index as f64, 0.0)));
        }
        assert!(!session.push_vertex(LonLat::new(1.0, 1.0)));
        assert_eq!(session.vertices().len(), MAX_MEASURE_VERTICES);
    }

    #[test]
    fn measure_readout_reports_area_once_three_points_enclose_something() {
        let mut session = MeasureSession::default();
        session.activate();
        let _first = session.push_vertex(LonLat::new(0.0, 0.0));
        let _second = session.push_vertex(LonLat::new(0.1, 0.0));
        let two = session.readout(None).unwrap_or_default();
        assert!(two.contains("Length"), "{two}");
        assert!(!two.contains("Area"), "{two}");
        let _third = session.push_vertex(LonLat::new(0.1, 0.1));
        let three = session.readout(None).unwrap_or_default();
        assert!(three.contains("Area"), "{three}");
        // The live vertex under the pointer counts too, so the area appears on
        // the frame the rubber band first closes a triangle.
        let mut live = MeasureSession::default();
        live.activate();
        let _first = live.push_vertex(LonLat::new(0.0, 0.0));
        let _second = live.push_vertex(LonLat::new(0.1, 0.0));
        let hovered = live
            .readout(Some(LonLat::new(0.1, 0.1)))
            .unwrap_or_default();
        assert!(hovered.contains("Area"), "{hovered}");
        // A finished measurement ignores the hover entirely — the numbers must
        // stop moving once the tape is put down.
        let mut ended = live.clone();
        let _ended = ended.finish();
        assert_eq!(
            ended.readout(Some(LonLat::new(9.0, 9.0))),
            ended.readout(None)
        );
        assert!(
            ended
                .readout(None)
                .is_some_and(|line| line.contains("ended")),
            "an ended measurement says so"
        );
    }

    #[test]
    fn measure_readout_quotes_the_last_segment_and_its_bearing() {
        let mut session = MeasureSession::default();
        session.activate();
        let _first = session.push_vertex(LonLat::new(0.0, 0.0));
        let _second = session.push_vertex(LonLat::new(0.0, 0.1));
        let line = session.readout(None).unwrap_or_default();
        assert!(line.contains("Segment"), "{line}");
        // Due north.
        assert!(line.contains("at 0\u{b0}"), "{line}");
    }

    #[test]
    fn coord_parses_lat_lon_and_lon_lat_by_range() {
        // Rule 2: 139.76 cannot be a latitude, so the other number is.
        assert_eq!(
            parse_coordinate("139.76, 35.68"),
            Ok(LonLat::new(139.76, 35.68))
        );
        // Rule 3: both could be latitudes, so the first one is.
        assert_eq!(
            parse_coordinate("35.68, 39.76"),
            Ok(LonLat::new(39.76, 35.68))
        );
        // And the everyday paste, which is `lat, lon`.
        assert_eq!(
            parse_coordinate("35.68, 139.76"),
            Ok(LonLat::new(139.76, 35.68))
        );
    }

    #[test]
    fn coord_accepts_every_documented_separator_and_sign() {
        for text in [
            "35.68 139.76",
            "35.68;139.76",
            "35.68/139.76",
            "  35.68 ,  139.76  ",
            "35.68\u{b0} 139.76\u{b0}",
        ] {
            assert_eq!(
                parse_coordinate(text),
                Ok(LonLat::new(139.76, 35.68)),
                "failed on {text:?}"
            );
        }
        assert_eq!(
            parse_coordinate("-33.87, 151.21"),
            Ok(LonLat::new(151.21, -33.87))
        );
    }

    #[test]
    fn coord_hemisphere_letters_override_the_order() {
        assert_eq!(
            parse_coordinate("139.76E, 35.68N"),
            Ok(LonLat::new(139.76, 35.68))
        );
        assert_eq!(
            parse_coordinate("35.68 N 139.76 E"),
            Ok(LonLat::new(139.76, 35.68))
        );
        assert_eq!(
            parse_coordinate("33.87S, 151.21E"),
            Ok(LonLat::new(151.21, -33.87))
        );
        assert_eq!(
            parse_coordinate("W122.42, N37.77"),
            Ok(LonLat::new(-122.42, 37.77))
        );
        // Two of the same axis is a contradiction, not a guess.
        assert_eq!(parse_coordinate("35N, 36N"), Err(CoordinateError::SameAxis));
        assert!(matches!(
            parse_coordinate("-35 N, 139 E"),
            Err(CoordinateError::SignedHemisphere(_))
        ));
    }

    #[test]
    fn coord_refuses_what_is_not_a_coordinate() {
        assert_eq!(parse_coordinate(""), Err(CoordinateError::Empty));
        assert_eq!(parse_coordinate("   "), Err(CoordinateError::Empty));
        assert_eq!(
            parse_coordinate("35.68"),
            Err(CoordinateError::NotTwoNumbers)
        );
        assert_eq!(
            parse_coordinate("35.68, 139.76, 12"),
            Err(CoordinateError::NotTwoNumbers)
        );
        assert!(matches!(
            parse_coordinate("Tokyo, Japan"),
            Err(CoordinateError::NotANumber(_))
        ));
        // Neither number can be a latitude.
        assert_eq!(
            parse_coordinate("120.0, 130.0"),
            Err(CoordinateError::NoLatitude)
        );
        assert!(matches!(
            parse_coordinate("95N, 10E"),
            Err(CoordinateError::LatitudeOutOfRange(_))
        ));
        assert!(matches!(
            parse_coordinate("10N, 190E"),
            Err(CoordinateError::LongitudeOutOfRange(_))
        ));
        // Every refusal has something to say about which half to look at.
        assert!(!CoordinateError::Empty.to_string().is_empty());
        assert!(CoordinateError::NoLatitude.to_string().contains("-90"));
    }

    #[test]
    fn coord_a_hemisphere_letter_only_counts_when_a_number_is_left_behind() {
        // Regression: a stripper that trusted its own letters read `nowhere`
        // as N…E and reported two contradicting axes, and `-33.87` had to be
        // read as a number BEFORE `w`/`e` were considered at all.
        assert_eq!(
            parse_coordinate("nowhere, 10"),
            Err(CoordinateError::NotANumber("nowhere".to_string()))
        );
        assert_eq!(
            parse_coordinate("east, west"),
            Err(CoordinateError::NotANumber("east".to_string()))
        );
        // Scientific notation is a number first and a hemisphere never: `5E3`
        // is 5000, so the pair is (lat 10, lon 5000) and the LONGITUDE range
        // is what refuses it. Read as "5 east" it would have been a valid
        // (10, 5) and silently gone somewhere else entirely.
        assert_eq!(
            parse_coordinate("5E3, 10"),
            Err(CoordinateError::LongitudeOutOfRange(5_000.0))
        );
        assert_eq!(parse_coordinate("1e1, 2e1"), Ok(LonLat::new(20.0, 10.0)));
        // `inf`/`NaN` parse as `f64` and must still be refused.
        assert!(matches!(
            parse_coordinate("inf, 10"),
            Err(CoordinateError::NotANumber(_))
        ));
        assert!(matches!(
            parse_coordinate("NaN, 10"),
            Err(CoordinateError::NotANumber(_))
        ));
    }

    #[test]
    fn coord_error_display_names_the_offending_token() {
        let error = parse_coordinate("abc, 10");
        assert!(
            error
                .as_ref()
                .err()
                .is_some_and(|error| error.to_string().contains("abc")),
            "{error:?}"
        );
    }

    #[test]
    fn coord_go_to_dialog_starts_closed_at_a_usable_zoom() {
        let dialog = GoToDialog::default();
        assert!(!dialog.open);
        assert!(dialog.text.is_empty());
        assert!(dialog.error.is_none());
        assert!((0.0..=24.0).contains(&dialog.zoom));
    }

    // ---- The app half: the tape's arming rules and the camera jump. --------

    #[test]
    fn measure_arming_the_tape_switches_the_editor_off() {
        // The two tools want the same click, so they must not both be live —
        // see the module docs.
        let mut app = OxigisApp::new();
        assert!(!app.measuring());
        app.set_measuring(true);
        assert!(app.measuring());
        assert_eq!(app.edit.mode(), crate::edit::EditMode::Off);
        app.set_measuring(false);
        assert!(!app.measuring());
    }

    #[test]
    fn measure_at_collects_only_while_the_tape_is_out() {
        let mut app = OxigisApp::new();
        // A tape that was never taken out measures nothing.
        assert!(!app.measure_at(LonLat::new(0.0, 0.0)));
        assert_eq!(app.measure_readout(), None);

        app.set_measuring(true);
        assert!(app.measure_at(LonLat::new(0.0, 0.0)));
        assert!(app.measure_at(LonLat::new(1.0, 0.0)));
        let readout = app.measure_readout().unwrap_or_default();
        assert!(readout.contains("Length"), "{readout}");
        // 1° at the equator is ~111.3 km.
        assert!(readout.contains("111."), "{readout}");

        assert!(app.finish_measurement());
        assert!(
            app.measure_readout()
                .is_some_and(|line| line.contains("ended"))
        );
        // A click after the end starts a fresh measurement rather than
        // extending the finished one.
        assert!(app.measure_at(LonLat::new(5.0, 5.0)));
        assert_eq!(app.measure.vertices().len(), 1);

        // Putting the tape away forgets everything.
        app.set_measuring(false);
        assert_eq!(app.measure_readout(), None);
    }

    #[test]
    fn measure_go_to_coordinate_moves_the_camera_and_reports_where_it_landed() {
        let mut app = OxigisApp::new();
        let Ok(landed) = app.go_to_coordinate("35.68, 139.76", Some(11.0)) else {
            panic!("a plain lat, lon pair is a coordinate");
        };
        assert!((landed.lat - 35.68).abs() < 1e-9);
        assert!((landed.lon - 139.76).abs() < 1e-9);
        let camera = app.map_view();
        assert!((camera.center().lat - 35.68).abs() < 1e-9);
        assert!((camera.center().lon - 139.76).abs() < 1e-9);
        assert!((camera.zoom() - 11.0).abs() < 1e-9);
        assert!(
            app.status().is_some_and(|status| status.contains("35.68")),
            "{:?}",
            app.status()
        );

        // No zoom given keeps the one the camera is at.
        let Ok(_moved) = app.go_to_coordinate("-33.87, 151.21", None) else {
            panic!("a southern coordinate is a coordinate");
        };
        assert!((app.map_view().zoom() - 11.0).abs() < 1e-9);
    }

    #[test]
    fn measure_go_to_reports_the_cut_off_rather_than_what_was_typed() {
        // Web Mercator stops at ±85.05°, so a polar coordinate lands at the
        // cut-off — and the answer must say so rather than echo the input,
        // which would name a place the map is not showing.
        let mut app = OxigisApp::new();
        let Ok(landed) = app.go_to_coordinate("89.0, 10.0", Some(4.0)) else {
            panic!("89°N is a latitude");
        };
        assert!(landed.lat < 89.0, "landed at {}", landed.lat);
        assert!(
            (landed.lat - oxigis_render::MAX_LATITUDE_DEG).abs() < 1e-6,
            "landed at {}",
            landed.lat
        );
        assert_eq!(landed, app.map_view().center());
    }

    #[test]
    fn measure_go_to_refuses_a_non_coordinate_and_leaves_the_camera_alone() {
        let mut app = OxigisApp::new();
        let before = app.map_view();
        // `nowhere` begins with `n` and ends with `e`: a parser that trusted
        // its own hemisphere letters would report two contradicting axes here
        // instead of the plain truth.
        assert_eq!(
            app.go_to_coordinate("nowhere", None),
            Err(CoordinateError::NotANumber("nowhere".to_string()))
        );
        assert_eq!(
            app.go_to_coordinate("35.68", None),
            Err(CoordinateError::NotTwoNumbers)
        );
        assert_eq!(app.map_view().center(), before.center());
        assert!((app.map_view().zoom() - before.zoom()).abs() < 1e-12);
    }

    #[test]
    fn measure_a_full_frame_with_the_tape_out_paints_without_panicking() {
        // The overlay's real draw path — a rubber band, a closing edge and the
        // readout plate — against a real panel rect.
        let mut app = OxigisApp::new();
        app.set_measuring(true);
        let _first = app.measure_at(LonLat::new(0.0, 0.0));
        let _second = app.measure_at(LonLat::new(0.5, 0.0));
        let _third = app.measure_at(LonLat::new(0.5, 0.5));
        let ctx = egui::Context::default();
        for _frame in 0..2 {
            let raw_input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1024.0, 768.0),
                )),
                ..Default::default()
            };
            let _output = ctx.run_ui(raw_input.clone(), |ui| app.ui(ui));
        }
        assert!(app.measuring(), "the tape survives a frame");
        assert!(
            app.measure_readout()
                .is_some_and(|line| line.contains("Area")),
            "three vertices enclose an area"
        );
    }

    #[test]
    fn measure_escape_ends_the_measurement_then_puts_the_tape_away() {
        let mut app = OxigisApp::new();
        app.set_measuring(true);
        let _taken = app.measure_at(LonLat::new(0.0, 0.0));
        let ctx = egui::Context::default();
        let escape = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            events: vec![egui::Event::Key {
                key: egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        };
        let _output = ctx.run_ui(escape(), |ui| app.ui(ui));
        assert!(app.measure.is_finished(), "one press ends the measurement");
        assert!(app.measuring(), "and leaves the tool out");
        let _output = ctx.run_ui(escape(), |ui| app.ui(ui));
        assert!(!app.measuring(), "the next press puts the tape away");
    }
}
