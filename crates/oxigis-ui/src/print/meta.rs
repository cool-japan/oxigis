// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Document metadata for the exported PDF (print v1.7): the `/Info`
//! dictionary every viewer reads a file's *name* from.
//!
//! Before this module the export wrote no `/Info` at all, so a reader titled
//! the page by its file name, no producer was recorded, and nothing said when
//! the map was drawn — the three facts an archived map sheet is expected to
//! carry. What lands here is deliberately small and boring: `/Title` (the
//! project name), `/Creator` and `/Producer` (this build of OxiGIS) and
//! `/CreationDate`.
//!
//! # The clock, and why it is an option field
//!
//! `std::time::SystemTime::now()` **panics** on `wasm32-unknown-unknown` (std
//! has no clock there), and this crate is shared by the desktop and the web
//! shell, so the native read is `cfg`-gated and the web build simply has no
//! clock of its own. [`super::PrintOptions::creation_epoch_secs`] is the seam
//! that fills the gap from both directions: the web shell can stamp
//! `Date::now() / 1000`, and the tests pin an exact second so an assembled
//! document is byte-reproducible instead of varying with the wall clock.

use pdf_writer::Date;

use super::PrintOptions;

/// Seconds in one day.
const SECONDS_PER_DAY: i64 = 86_400;

/// The earliest instant a PDF date can name: `0000-01-01T00:00:00Z`.
///
/// `/CreationDate`'s year field is four digits, and [`Date`] clamps the year
/// into `0..=9999` — clamping the *input* instead keeps the whole conversion
/// inside the range its own arithmetic was reasoned about, so no epoch value,
/// [`i64::MIN`] included, can reach the civil-calendar maths with a day count
/// it cannot represent.
const MIN_EPOCH_SECS: i64 = -62_167_219_200;

/// The latest instant a PDF date can name: `9999-12-31T23:59:59Z`.
const MAX_EPOCH_SECS: i64 = 253_402_300_799;

/// The `/Producer` string: this build of OxiGIS.
///
/// `CARGO_PKG_VERSION` is the workspace version — `oxigis-ui` takes
/// `version.workspace = true`, so the number here is the release the whole
/// application ships under, not a per-crate number that could drift.
#[must_use]
pub(super) fn producer() -> String {
    format!("OxiGIS {}", env!("CARGO_PKG_VERSION"))
}

/// The `/Creator` string: the application that composed the page.
pub(super) const CREATOR: &str = "OxiGIS";

/// The instant to stamp as `/CreationDate`, or [`None`] when the platform
/// offers no clock and the shell supplied no timestamp.
///
/// An explicit [`PrintOptions::creation_epoch_secs`] always wins, which is
/// what makes an export reproducible: two runs with the same options produce
/// the same bytes.
#[must_use]
pub(super) fn creation_date(options: &PrintOptions) -> Option<Date> {
    options
        .creation_epoch_secs
        .or_else(system_epoch_secs)
        .map(pdf_date)
}

/// The system clock as Unix seconds — native only.
///
/// Returns [`None`] on wasm32, where `SystemTime::now()` panics rather than
/// answering, and on a native machine whose clock predates the epoch by more
/// than [`i64`] can hold (unreachable, handled anyway because this is the one
/// place a fallible conversion happens).
#[cfg(not(target_arch = "wasm32"))]
fn system_epoch_secs() -> Option<i64> {
    let now = std::time::SystemTime::now();
    match now.duration_since(std::time::UNIX_EPOCH) {
        Ok(since) => i64::try_from(since.as_secs()).ok(),
        // Before 1970: a clock set into the past still dates the sheet.
        Err(before) => i64::try_from(before.duration().as_secs())
            .ok()
            .map(|secs| -secs),
    }
}

/// wasm32 has no clock in `std`; the shell stamps
/// [`PrintOptions::creation_epoch_secs`] instead.
#[cfg(target_arch = "wasm32")]
fn system_epoch_secs() -> Option<i64> {
    None
}

/// One Unix instant as a PDF date, in UTC.
///
/// Total and panic-free for every [`i64`]: the input is clamped to the range
/// four-digit years can name before any arithmetic runs.
#[must_use]
pub(super) fn pdf_date(epoch_secs: i64) -> Date {
    let clamped = epoch_secs.clamp(MIN_EPOCH_SECS, MAX_EPOCH_SECS);
    let days = clamped.div_euclid(SECONDS_PER_DAY);
    let seconds = clamped.rem_euclid(SECONDS_PER_DAY);
    let (year, month, day) = civil_from_days(days);
    let hour = (seconds / 3_600) as u8;
    let minute = ((seconds % 3_600) / 60) as u8;
    let second = (seconds % 60) as u8;
    Date::new(year)
        .month(month)
        .day(day)
        .hour(hour)
        .minute(minute)
        .second(second)
        // `0` writes as the `Z` suffix rather than as `+00'00'`: the page is
        // stamped in UTC, and a reader that shows local time converts it.
        .utc_offset_hour(0)
        .utc_offset_minute(0)
}

/// Days since 1970-01-01 → `(year, month, day)` in the proleptic Gregorian
/// calendar (Howard Hinnant's `civil_from_days`, the standard branch-free
/// derivation the C++20 calendar is specified with).
///
/// `days` arrives from [`pdf_date`]'s clamped range, so the era arithmetic
/// stays far inside [`i64`] and the year always fits the `0..=9999` the PDF
/// date grammar allows.
fn civil_from_days(days: i64) -> (u16, u8, u8) {
    // Shift the epoch to 0000-03-01, which makes the leap day the LAST day of
    // the year and the month lengths a closed formula.
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097; // 0..=146096
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let march_month = (5 * day_of_year + 2) / 153; // 0..=11, 0 = March
    let day = day_of_year - (153 * march_month + 2) / 5 + 1;
    let month = if march_month < 10 {
        march_month + 3
    } else {
        march_month - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    (
        year.clamp(0, 9_999) as u16,
        month.clamp(1, 12) as u8,
        day.clamp(1, 31) as u8,
    )
}
