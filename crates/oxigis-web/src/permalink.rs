//! The URL-hash permalink: `#map=<zoom>/<lat>/<lon>`.
//!
//! Standard slippy-map convention — the same order OpenStreetMap's own
//! permalink uses (`openstreetmap.org/#map=15/51.5074/-0.1278`) and the one
//! `leaflet-hash` and its descendants settled on, so a link copied out of
//! OxiGIS reads the same way a link copied out of any other slippy map does.
//!
//! # Basemap is deliberately not encoded here
//!
//! The mission this module implements allows it "if cheap"; it is not.
//! `oxigis_ui::BasemapConfig` is a URL template, a subdomain list and a free-
//! text attribution line (often a full sentence), not a small enum with a
//! compact id — encoding it would mean percent-escaping a string containing
//! the same `/` this format already uses as a field separator, and the
//! result would dwarf the three numbers around it for the *overwhelmingly*
//! common case (the default OpenStreetMap service). The format string this
//! module was asked to write, `z/lat/lon`, has no slot for it either. Camera
//! only.
//!
//! # Split from the glue
//!
//! This module is pure — parsing and formatting three numbers — and
//! deliberately **not** `#[cfg(target_arch = "wasm32")]`-gated, so
//! `cargo nextest run -p oxigis-web` exercises it on a host with no browser
//! at all. Reading `window.location.hash` and writing it back through
//! `history.replaceState` is the wasm-only half, and lives in the sibling
//! glue module instead (referenced here only in prose, never as an intra-doc
//! link — a link into wasm32-only code is exactly what breaks a native
//! `cargo doc` build; see this crate's release-gate fix for the two spots
//! that already got this wrong).

/// Decimal places kept for latitude and longitude.
///
/// About 11 cm at the equator — several orders of magnitude past a zoom-24
/// pixel — and, not incidentally, what `leaflet-hash` and OpenStreetMap's own
/// permalink both settled on, so a URL this shell writes has the shape
/// everyone already expects.
const LATLON_DECIMALS: i32 = 5;

/// Decimal places kept for zoom.
///
/// [`oxigis_render::viewport::MapView::zoom`] is fractional (pinch/scroll
/// land between integer levels); two places is finer than any input device
/// resolves and still short enough that the fragment stays readable.
const ZOOM_DECIMALS: i32 = 2;

/// A camera restored from, or destined for, a `#map=` fragment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PermalinkView {
    /// Fractional zoom level.
    pub zoom: f64,
    /// Center latitude, in degrees.
    pub lat: f64,
    /// Center longitude, in degrees.
    pub lon: f64,
}

/// Rounds `value` to `decimals` decimal places.
///
/// Non-finite input rounds to `0.0` rather than propagating — a permalink
/// fragment holds three plain numbers, never a NaN or an infinity, so this
/// is the one place that guarantee is enforced rather than trusted. A
/// rounded `-0.0` is normalised to `0.0` (`0.0 == -0.0` under IEEE-754, so
/// the comparison below costs nothing) so [`format_hash`] never prints a
/// spurious sign on a longitude that rounds down to exactly zero.
#[must_use]
pub fn round_to(value: f64, decimals: i32) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    let factor = 10f64.powi(decimals);
    let rounded = (value * factor).round() / factor;
    if rounded == 0.0 { 0.0 } else { rounded }
}

/// Rounds a `(zoom, lat, lon)` triple to permalink precision.
///
/// The exact rounding [`format_hash`] applies, exposed so a caller comparing
/// one frame's camera to the last (to decide whether anything worth writing
/// actually changed) compares like with like. Comparing a raw camera
/// reading against an already-rounded one is the bug this function exists to
/// make impossible: sub-precision float jitter would then read as "changed"
/// on every single frame, and a debounce timer that never sees two equal
/// readings in a row never fires.
#[must_use]
pub fn round_view(zoom: f64, lat: f64, lon: f64) -> (f64, f64, f64) {
    (
        round_to(zoom, ZOOM_DECIMALS),
        round_to(lat, LATLON_DECIMALS),
        round_to(lon, LATLON_DECIMALS),
    )
}

/// Formats `(zoom, lat, lon)` as a `#map=` fragment.
///
/// Rounds first ([`round_view`]), then relies on `f64`'s `Display` — which,
/// unlike `Debug`, prints the shortest string that round-trips and drops a
/// bare `.0` — to produce exactly the compact form OpenStreetMap's own
/// permalinks use (`15` rather than `15.00`, `-0.1278` rather than
/// `-0.12780`).
#[must_use]
pub fn format_hash(zoom: f64, lat: f64, lon: f64) -> String {
    let (zoom, lat, lon) = round_view(zoom, lat, lon);
    format!("#map={zoom}/{lat}/{lon}")
}

/// Parses a `#map=<zoom>/<lat>/<lon>` fragment.
///
/// Tolerant of a missing leading `#` (`window.location.hash` always includes
/// it when the fragment is non-empty, but a caller testing with a bare
/// string should not have to remember that) and of other `&`-joined
/// segments — only the one named `map` is read, so a page that appends its
/// own state after OxiGIS's does not confuse this parser and a future
/// OxiGIS release can add its own segments without breaking old links.
///
/// Returns [`None`] — meaning "no restore, start at the default view" —
/// for anything that is not exactly three finite numbers: a missing `map`
/// segment, a segment count other than three, unparsable text, or a `NaN`/
/// `inf` value (`f64::from_str` accepts both spellings, which
/// [`str::parse`] alone would silently let through as a camera that can
/// never render).
#[must_use]
pub fn parse_hash(hash: &str) -> Option<PermalinkView> {
    let hash = hash.trim().trim_start_matches('#');
    let map_value = hash
        .split('&')
        .find_map(|segment| segment.strip_prefix("map="))?;
    let mut parts = map_value.split('/');
    let zoom: f64 = parts.next()?.parse().ok()?;
    let lat: f64 = parts.next()?.parse().ok()?;
    let lon: f64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        // A fourth `/`-separated field: not a fragment this module wrote.
        return None;
    }
    if !zoom.is_finite() || !lat.is_finite() || !lon.is_finite() {
        return None;
    }
    Some(PermalinkView { zoom, lat, lon })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_documented_format() {
        let view = parse_hash("#map=15/51.5074/-0.1278").expect("a valid fragment");
        assert_eq!(view.zoom, 15.0);
        assert_eq!(view.lat, 51.5074);
        assert_eq!(view.lon, -0.1278);
    }

    #[test]
    fn tolerates_a_missing_leading_hash() {
        assert_eq!(
            parse_hash("map=3/10/20"),
            Some(PermalinkView {
                zoom: 3.0,
                lat: 10.0,
                lon: 20.0
            })
        );
    }

    #[test]
    fn reads_only_the_map_segment_among_others() {
        let view = parse_hash("#foo=bar&map=3/10/20&baz=1").expect("map segment present");
        assert_eq!(
            view,
            PermalinkView {
                zoom: 3.0,
                lat: 10.0,
                lon: 20.0
            }
        );
    }

    #[test]
    fn empty_hash_is_no_restore() {
        assert_eq!(parse_hash(""), None);
    }

    #[test]
    fn map_with_no_value_is_refused() {
        assert_eq!(parse_hash("#map="), None);
    }

    #[test]
    fn wrong_segment_counts_are_refused() {
        assert_eq!(parse_hash("#map=15/51.5"), None);
        assert_eq!(parse_hash("#map=15/51.5/1/2"), None);
    }

    #[test]
    fn non_numeric_fields_are_refused() {
        assert_eq!(parse_hash("#map=abc/51.5/1"), None);
    }

    #[test]
    fn nan_and_infinite_spellings_are_refused() {
        // `f64::from_str` accepts these spellings; a permalink must not.
        assert_eq!(parse_hash("#map=NaN/1/2"), None);
        assert_eq!(parse_hash("#map=15/inf/2"), None);
        assert_eq!(parse_hash("#map=15/-infinity/2"), None);
    }

    #[test]
    fn no_map_segment_at_all_is_no_restore() {
        assert_eq!(parse_hash("#foo=bar"), None);
    }

    #[test]
    fn format_matches_the_documented_shape() {
        assert_eq!(
            format_hash(15.0, 51.5074, -0.1278),
            "#map=15/51.5074/-0.1278"
        );
    }

    #[test]
    fn format_drops_a_bare_trailing_zero() {
        // `f64` `Display` (not `Debug`): whole numbers print with no decimal
        // point at all, matching OpenStreetMap's own permalink shape.
        assert_eq!(format_hash(2.0, 0.0, 0.0), "#map=2/0/0");
    }

    #[test]
    fn format_rounds_to_permalink_precision() {
        assert_eq!(
            format_hash(15.001, 51.507_400_001, -0.127_800_009),
            "#map=15/51.5074/-0.1278"
        );
    }

    #[test]
    fn format_normalises_a_rounded_negative_zero() {
        // A longitude that rounds down to exactly zero must not print "-0".
        assert_eq!(format_hash(2.0, 0.0, -0.000_001), "#map=2/0/0");
    }

    #[test]
    fn round_trips_through_parse_and_format() {
        let original = PermalinkView {
            zoom: 15.0,
            lat: 51.5074,
            lon: -0.1278,
        };
        let hash = format_hash(original.zoom, original.lat, original.lon);
        let parsed = parse_hash(&hash).expect("a fragment this module wrote must parse");
        assert_eq!(parsed, original);
        // And formatting the round-tripped view again is byte-identical —
        // the debounce in the wasm glue relies on this stability to decide
        // "nothing changed" without re-parsing.
        assert_eq!(format_hash(parsed.zoom, parsed.lat, parsed.lon), hash);
    }

    #[test]
    fn round_view_matches_format_hashs_rounding() {
        let (zoom, lat, lon) = round_view(15.001, 51.507_400_001, -0.127_800_009);
        assert_eq!(
            format!("#map={zoom}/{lat}/{lon}"),
            "#map=15/51.5074/-0.1278"
        );
    }

    #[test]
    fn round_to_normalises_negative_zero() {
        let rounded = round_to(-0.000_001, 5);
        assert_eq!(rounded, 0.0);
        assert!(rounded.is_sign_positive());
    }

    #[test]
    fn round_to_rejects_non_finite_input() {
        assert_eq!(round_to(f64::NAN, 5), 0.0);
        assert_eq!(round_to(f64::INFINITY, 5), 0.0);
        assert_eq!(round_to(f64::NEG_INFINITY, 5), 0.0);
    }
}
