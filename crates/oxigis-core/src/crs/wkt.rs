// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! A real WKT CRS reader: what kind of CRS a WKT string declares, what it is
//! called, and — the part that actually decides everything downstream — which
//! EPSG code its own authority clause claims.
//!
//! This replaces a marker scan (`upper.contains("WGS84")`) that recognised two
//! CRSes and got one of them wrong. Two things make the difference:
//!
//! # 1. The authority code is read at the right nesting depth
//!
//! WKT carries an authority clause at *every* level. A single WKT1 `PROJCS`
//! for JGD2011 zone IX contains, in order: `AUTHORITY["EPSG","7019"]` (the
//! GRS 1980 spheroid), `AUTHORITY["EPSG","1128"]` (the JGD2011 datum),
//! `AUTHORITY["EPSG","6668"]` (the geographic CRS), `AUTHORITY["EPSG","9001"]`
//! (the metre) and finally `AUTHORITY["EPSG","6677"]` (the CRS itself). Only
//! the last of those is the answer, and "last in the text" is not a rule that
//! survives a reordered writer — the rule is **the authority clause that is a
//! direct child of the root node**. So [`parse_wkt`] tracks bracket depth and
//! reads the authority at depth 1 only. A first-match grab would hand back
//! 7019 (a spheroid), and in the 6668/6677 neighbourhood that is not a
//! theoretical confusion: 6668 IS a real CRS, just the wrong one — geographic
//! degrees where the file holds metres.
//!
//! WKT2 spells the same clause `ID["EPSG",6677]`, with an unquoted number and
//! optional `URI[…]`/`VERSION[…]` members; the depth rule is identical, and
//! the root of a WKT2 `PROJCRS` is likewise where the CRS's own `ID` lives.
//!
//! # 2. `TOWGS84[…]` is stripped before any name is matched
//!
//! `TOWGS84` contains `WGS84` as a substring. GDAL, ogr2ogr and QGIS emit a
//! `TOWGS84[…]` Helmert clause for **every datum that has one** — Tokyo
//! (EPSG:4301), NAD27, ED50, Pulkovo 1942, Beijing 1954. A scan that accepts
//! "WGS 84 appears anywhere in the text" therefore classified every one of
//! them as WGS 84 and drew them 100–800 m out with no notice (finding 73).
//! Here the name-marker fallback runs over a *stripped* copy from which every
//! `TOWGS84[…]`, `AUTHORITY[…]` and `ID[…]` group has been removed, and it
//! only ever looks at the root CRS name and the datum name — never at the
//! whole string.
//!
//! The fallback is a fallback: [`resolve_epsg`] tries the authority code
//! first, and only then the names.

use crate::crs::epsg;

/// The maximum number of characters [`parse_wkt`] will scan.
///
/// A WKT2 compound CRS with a full usage/scope/remark block runs to a few
/// kilobytes; anything past this is not a CRS definition, and the scan is
/// linear so the bound is what keeps a hostile 100 MB "`.prj`" from costing
/// 100 MB of work. Text past the limit is ignored, never truncated into the
/// result.
pub const MAX_WKT_SCAN_CHARS: usize = 64 * 1024;

/// The maximum nesting depth [`parse_wkt`] will track.
///
/// Real WKT nests about eight deep (`PROJCRS` → `BASEGEOGCRS` → `DATUM` →
/// `ELLIPSOID` → `LENGTHUNIT` → `ID`). A deeper input is malformed or
/// adversarial; the scanner stops descending rather than growing state.
const MAX_WKT_DEPTH: u32 = 64;

/// The broad class of CRS a WKT string declares.
///
/// What matters downstream is only "are these coordinates degrees or linear
/// units", but the distinction between "some other kind" and "not WKT at all"
/// is worth keeping so a refusal can say which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WktKind {
    /// `GEOGCS` (WKT1) / `GEOGCRS`, `BASEGEOGCRS` (WKT2) — lon/lat degrees.
    Geographic,
    /// `PROJCS` (WKT1) / `PROJCRS` (WKT2) — linear units.
    Projected,
    /// `COMPD_CS` / `COMPOUNDCRS` — a horizontal CRS plus a vertical one.
    Compound,
    /// `GEOCCS` / `GEODCRS` with a Cartesian axis set — earth-centred metres.
    Geocentric,
    /// `VERT_CS` / `VERTCRS` — a height system with no horizontal component.
    Vertical,
    /// `LOCAL_CS` / `ENGCRS` — an engineering/local frame with no datum.
    Engineering,
    /// Something that parsed as a bracketed keyword but is none of the above.
    Other,
}

impl WktKind {
    /// The keyword a WKT string opens with, mapped to a kind.
    fn from_keyword(keyword: &str) -> Self {
        match keyword {
            "GEOGCS" | "GEOGCRS" | "BASEGEOGCRS" | "GEOGRAPHICCRS" => Self::Geographic,
            "PROJCS" | "PROJCRS" | "BASEPROJCRS" => Self::Projected,
            "COMPD_CS" | "COMPOUNDCRS" => Self::Compound,
            "GEOCCS" | "GEODCRS" | "GEODETICCRS" => Self::Geocentric,
            "VERT_CS" | "VERTCS" | "VERTCRS" | "VERTICALCRS" => Self::Vertical,
            "LOCAL_CS" | "ENGCRS" | "ENGINEERINGCRS" => Self::Engineering,
            _ => Self::Other,
        }
    }

    /// Whether coordinates in this CRS are longitude/latitude degrees.
    #[must_use]
    pub fn is_geographic(self) -> bool {
        matches!(self, Self::Geographic)
    }
}

/// What [`parse_wkt`] read out of a WKT string.
#[derive(Debug, Clone, PartialEq)]
pub struct WktInfo {
    /// The root keyword's class.
    pub kind: WktKind,
    /// The CRS's own name — the first quoted string inside the root node.
    pub name: Option<String>,
    /// The datum's name, when the string declares one.
    pub datum_name: Option<String>,
    /// The EPSG code the root node's own `AUTHORITY`/`ID` clause claims.
    ///
    /// Only ever the *root's* code (see the module docs); a spheroid's or a
    /// datum's code is never reported here.
    pub authority_epsg: Option<u32>,
    /// The authority name from that same clause, upper-cased — `"EPSG"` for
    /// everything OxiGIS acts on, but `"ESRI"` and `"IGNF"` occur in the wild
    /// and are worth being able to name in a refusal.
    pub authority_name: Option<String>,
    /// Whether the string carries a `TOWGS84[…]` Helmert clause. Recorded
    /// because its mere presence is a strong signal that the datum is **not**
    /// WGS 84 — nobody writes a datum shift from WGS 84 to itself.
    pub has_towgs84: bool,
}

impl WktInfo {
    /// An empty result, for input that is not WKT at all.
    fn empty() -> Self {
        Self {
            kind: WktKind::Other,
            name: None,
            datum_name: None,
            authority_epsg: None,
            authority_name: None,
            has_towgs84: false,
        }
    }
}

/// One character of the scanner's view of the string.
enum Token {
    Open,
    Close,
    Quote,
    Other,
}

fn classify(character: char) -> Token {
    match character {
        '[' | '(' => Token::Open,
        ']' | ')' => Token::Close,
        '"' => Token::Quote,
        _ => Token::Other,
    }
}

/// Reads a WKT CRS string.
///
/// Never fails and never allocates unboundedly: a string that is not WKT
/// simply yields a [`WktInfo`] with [`WktKind::Other`] and no fields set. Only
/// the first [`MAX_WKT_SCAN_CHARS`] characters are examined.
///
/// The scan is a single pass that tracks bracket depth and quoting, so it is
/// robust against the two things a `contains`-based sniff cannot survive:
/// authority clauses nested inside the definition, and CRS names that happen
/// to contain a keyword.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn parse_wkt(wkt: &str) -> WktInfo {
    // A UTF-8 BOM is common on `.prj` files written on Windows and would push
    // the root keyword off by three bytes.
    let text = wkt.trim_start_matches('\u{feff}').trim();
    if text.is_empty() {
        return WktInfo::empty();
    }

    let mut info = WktInfo::empty();
    // The keyword currently being accumulated (letters/digits/underscore run
    // immediately before an opening bracket).
    let mut keyword = String::new();
    // The keyword that opened the node at each depth, innermost last.
    let mut open_keywords: Vec<String> = Vec::new();
    // The quoted string currently being accumulated, when inside quotes.
    let mut quoted = String::new();
    let mut in_quotes = false;
    // Non-quoted run since the last delimiter, used for WKT2's unquoted
    // `ID["EPSG",6677]` code.
    let mut bare = String::new();
    let mut depth: u32 = 0;
    // Which quoted string of the current node this is (0-based).
    let mut node_string_index: Vec<u32> = Vec::new();
    // Set while inside a node whose authority code should be captured.
    let mut root_name_taken = false;

    for character in text.chars().take(MAX_WKT_SCAN_CHARS) {
        if in_quotes {
            if character == '"' {
                in_quotes = false;
                let value = std::mem::take(&mut quoted);
                let index = node_string_index.last_mut().map_or(0, |slot| {
                    let current = *slot;
                    *slot = slot.saturating_add(1);
                    current
                });
                on_quoted_string(
                    &mut info,
                    &open_keywords,
                    depth,
                    index,
                    value,
                    &mut root_name_taken,
                );
            } else if quoted.len() < 256 {
                // A CRS name longer than this is not a name; keep the prefix
                // rather than growing the buffer for an adversarial input.
                quoted.push(character);
            }
            continue;
        }

        match classify(character) {
            Token::Quote => {
                on_bare_run(&mut info, &open_keywords, depth, &mut bare);
                in_quotes = true;
                quoted.clear();
            }
            Token::Open => {
                on_bare_run(&mut info, &open_keywords, depth, &mut bare);
                let word = std::mem::take(&mut keyword).to_ascii_uppercase();
                if depth == 0 {
                    info.kind = WktKind::from_keyword(&word);
                }
                if word == "TOWGS84" {
                    info.has_towgs84 = true;
                }
                depth = depth.saturating_add(1);
                if depth <= MAX_WKT_DEPTH {
                    open_keywords.push(word);
                    node_string_index.push(0);
                }
            }
            Token::Close => {
                on_bare_run(&mut info, &open_keywords, depth, &mut bare);
                keyword.clear();
                if depth <= MAX_WKT_DEPTH {
                    open_keywords.pop();
                    node_string_index.pop();
                }
                depth = depth.saturating_sub(1);
            }
            Token::Other => {
                if character == ',' {
                    on_bare_run(&mut info, &open_keywords, depth, &mut bare);
                    keyword.clear();
                } else if character.is_alphanumeric() || character == '_' {
                    if keyword.len() < 64 {
                        keyword.push(character);
                    }
                    if bare.len() < 64 {
                        bare.push(character);
                    }
                } else if character.is_whitespace() {
                    // Whitespace separates a keyword from its bracket in some
                    // writers; keep the keyword, end the bare run.
                    if !bare.is_empty() {
                        on_bare_run(&mut info, &open_keywords, depth, &mut bare);
                    }
                } else {
                    keyword.clear();
                    bare.clear();
                }
            }
        }
    }
    info
}

/// Handles one completed quoted string.
///
/// `depth` is the depth *inside* the node the string belongs to, and `index`
/// is its ordinal within that node — WKT puts the name first everywhere, and
/// `AUTHORITY["EPSG","6677"]` puts the authority name at 0 and the code at 1.
fn on_quoted_string(
    info: &mut WktInfo,
    open_keywords: &[String],
    depth: u32,
    index: u32,
    value: String,
    root_name_taken: &mut bool,
) {
    let Some(keyword) = open_keywords.last() else {
        return;
    };
    match keyword.as_str() {
        "AUTHORITY" | "ID" if depth == 2 => {
            // Depth 2 means this node opened at depth 1: a direct child of the
            // root, i.e. the CRS's own authority. See the module docs.
            if index == 0 {
                info.authority_name = Some(value.to_ascii_uppercase());
            } else if index == 1 {
                info.authority_epsg = value.trim().parse::<u32>().ok();
            }
        }
        "DATUM" | "GEODETICDATUM" | "GEODETICREFERENCEFRAME" | "ENSEMBLE"
            if index == 0 && info.datum_name.is_none() =>
        {
            info.datum_name = Some(value);
        }
        _ => {
            if depth == 1 && index == 0 && !*root_name_taken {
                info.name = Some(value);
                *root_name_taken = true;
            }
        }
    }
}

/// Handles one completed unquoted run — WKT2's `ID["EPSG",6677]` writes the
/// code as a bare number.
fn on_bare_run(info: &mut WktInfo, open_keywords: &[String], depth: u32, bare: &mut String) {
    let run = std::mem::take(bare);
    if run.is_empty() || info.authority_epsg.is_some() {
        return;
    }
    if depth != 2 {
        return;
    }
    let Some(keyword) = open_keywords.last() else {
        return;
    };
    if keyword != "ID" && keyword != "AUTHORITY" {
        return;
    }
    if let Ok(code) = run.parse::<u32>() {
        info.authority_epsg = Some(code);
    }
}

/// A copy of `wkt`, upper-cased, with every `TOWGS84[…]`, `AUTHORITY[…]` and
/// `ID[…]` group removed.
///
/// This is what the name-marker fallback matches against, and it is the whole
/// of finding 73's structural fix: `TOWGS84[-146.414,507.337,680.507,…]`
/// contains the literal `WGS84`, so any scan that sees it will call a Tokyo
/// datum file WGS 84 unless the clause is taken out first.
#[must_use]
pub fn strip_shift_and_authority_clauses(wkt: &str) -> String {
    let upper: String = wkt
        .trim_start_matches('\u{feff}')
        .chars()
        .take(MAX_WKT_SCAN_CHARS)
        .flat_map(char::to_uppercase)
        .collect();
    let mut out = String::with_capacity(upper.len());
    let mut keyword = String::new();
    // Depth of the group being skipped, or 0 when not skipping.
    let mut skip_depth: u32 = 0;
    let mut depth: u32 = 0;
    let mut in_quotes = false;

    for character in upper.chars() {
        if in_quotes {
            if character == '"' {
                in_quotes = false;
            }
            if skip_depth == 0 {
                out.push(character);
            }
            continue;
        }
        match classify(character) {
            Token::Quote => {
                in_quotes = true;
                if skip_depth == 0 {
                    out.push(character);
                }
                keyword.clear();
            }
            Token::Open => {
                depth = depth.saturating_add(1);
                let word = std::mem::take(&mut keyword);
                let strippable = matches!(word.as_str(), "TOWGS84" | "AUTHORITY" | "ID");
                if skip_depth == 0 && strippable {
                    // Drop the keyword we already emitted, then skip the group.
                    let cut = out.len().saturating_sub(word.len());
                    out.truncate(cut);
                    skip_depth = depth;
                } else if skip_depth == 0 {
                    out.push(character);
                }
            }
            Token::Close => {
                if skip_depth != 0 && depth == skip_depth {
                    skip_depth = 0;
                } else if skip_depth == 0 {
                    out.push(character);
                }
                depth = depth.saturating_sub(1);
                keyword.clear();
            }
            Token::Other => {
                if character.is_alphanumeric() || character == '_' {
                    if keyword.len() < 64 {
                        keyword.push(character);
                    }
                } else {
                    keyword.clear();
                }
                if skip_depth == 0 {
                    out.push(character);
                }
            }
        }
    }
    out
}

/// Markers whose presence in a *name* means Web Mercator.
const WEB_MERCATOR_MARKERS: [&str; 8] = [
    "3857",
    "900913",
    "PSEUDO-MERCATOR",
    "PSEUDO_MERCATOR",
    "PSEUDO MERCATOR",
    "WEB_MERCATOR",
    "WEB MERCATOR",
    "AUXILIARY_SPHERE",
];

/// Markers whose presence in a *CRS or datum name* means the WGS 84 datum.
///
/// Note what is missing compared to the classifier this replaces: the bare
/// string `"4326"`. A code is an authority answer, not a name answer, and
/// [`resolve_epsg`] has already tried the authority clause by the time these
/// run — matching a number inside a name is how a spheroid code leaks into a
/// CRS decision.
const WGS84_NAME_MARKERS: [&str; 4] = ["WGS_1984", "WGS 84", "WGS84", "WGS 1984"];

/// Name fragments of the Japanese CRSs common enough to recognise without an
/// authority code, paired with the EPSG code they resolve to.
///
/// ESRI-flavoured `.prj` files — what ArcGIS and a good deal of Japanese
/// municipal open data ship — routinely carry **no** `AUTHORITY` clause at
/// all, so for those the name is the only signal there is.
const JAPANESE_NAME_HINTS: [(&str, u32); 6] = [
    ("JGD_2011", 6668),
    ("JGD2011", 6668),
    ("JAPANESE GEODETIC DATUM 2011", 6668),
    ("JGD_2000", 4612),
    ("JGD2000", 4612),
    ("JAPANESE GEODETIC DATUM 2000", 4612),
];

/// Zone-name fragments for the Japan Plane Rectangular zones, indexed by zone.
///
/// Both spellings ESRI writes (`JGD_2011_Japan_Zone_9`) and the EPSG one
/// (`JGD2011 / Japan Plane Rectangular CS IX`) reduce to a zone number here.
fn japan_zone_number(name_upper: &str) -> Option<usize> {
    // ESRI: "…JAPAN_ZONE_9", "…JAPAN_ZONE_17".
    if let Some(rest) = name_upper.rsplit("JAPAN_ZONE_").next()
        && rest != name_upper
    {
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        if let Ok(zone) = digits.parse::<usize>()
            && (1..=19).contains(&zone)
        {
            return Some(zone);
        }
    }
    // EPSG: "… PLANE RECTANGULAR CS IX".
    let marker = "PLANE RECTANGULAR CS ";
    if let Some(position) = name_upper.find(marker) {
        let rest = name_upper[position + marker.len()..].trim_start();
        let numeral: String = rest
            .chars()
            .take_while(|character| matches!(character, 'I' | 'V' | 'X'))
            .collect();
        if !numeral.is_empty() {
            return roman_to_zone(&numeral);
        }
    }
    None
}

/// The UTM zone and hemisphere a projected CRS *name* announces.
///
/// ESRI `.prj` files spell it `WGS_1984_UTM_Zone_54N` and carry no authority
/// clause at all; the EPSG name is `WGS 84 / UTM zone 54N`. Both reduce to the
/// same `(zone, north)` here.
fn utm_zone_from_name(name_upper: &str) -> Option<(u32, bool)> {
    let marker = name_upper
        .find("UTM_ZONE_")
        .map(|at| at + "UTM_ZONE_".len())
        .or_else(|| {
            name_upper
                .find("UTM ZONE ")
                .map(|at| at + "UTM ZONE ".len())
        })?;
    let rest = &name_upper[marker..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    let zone = digits
        .parse::<u32>()
        .ok()
        .filter(|zone| (1..=60).contains(zone))?;
    // A trailing `S` means the southern hemisphere; `N`, or nothing at all,
    // means the northern one (the overwhelmingly common case, and what a name
    // that stops at the zone number means in every writer seen).
    let hemisphere = rest[digits.len()..].chars().next();
    match hemisphere {
        Some('S') => Some((zone, false)),
        Some('N') | None => Some((zone, true)),
        // `54_something` is not a hemisphere marker; refuse rather than guess.
        Some(other) if other.is_alphanumeric() => None,
        _ => Some((zone, true)),
    }
}

/// The EPSG code for `zone` on the datum a CRS name announces, when this build
/// carries that combination.
fn utm_epsg_from_name(name_upper: &str, datum_upper: &str, zone: u32, north: bool) -> Option<u32> {
    let names = |needle: &str| name_upper.contains(needle) || datum_upper.contains(needle);
    let code = if names("JGD_2011") || names("JGD2011") {
        (51..=55).contains(&zone).then(|| 6688 + zone - 51)?
    } else if names("JGD_2000") || names("JGD2000") {
        (51..=55).contains(&zone).then(|| 3097 + zone - 51)?
    } else if names("TOKYO") {
        (51..=55).contains(&zone).then(|| 3092 + zone - 51)?
    } else if names("NAD_1983") || names("NAD83") {
        (north && (1..=23).contains(&zone)).then(|| 26900 + zone)?
    } else if names("ETRS_1989") || names("ETRS89") {
        (north && (28..=38).contains(&zone)).then(|| 25800 + zone)?
    } else if names("ED_1950") || names("ED50") || names("EUROPEAN_DATUM_1950") {
        (north && (28..=38).contains(&zone)).then(|| 23000 + zone)?
    } else {
        // WGS 84 is both the explicit default and the fallback: a UTM zone
        // named with no datum at all is WGS 84 in every writer's convention.
        if north { 32600 + zone } else { 32700 + zone }
    };
    epsg::is_supported(code).then_some(code)
}

/// Roman numerals I–XIX to `1..=19`, or [`None`] for anything else.
fn roman_to_zone(numeral: &str) -> Option<usize> {
    const NUMERALS: [&str; 19] = [
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV",
        "XV", "XVI", "XVII", "XVIII", "XIX",
    ];
    NUMERALS
        .iter()
        .position(|candidate| *candidate == numeral)
        .map(|index| index + 1)
}

/// Resolves a WKT string to the EPSG code OxiGIS should treat it as.
///
/// Order, strongest signal first:
///
/// 1. the root node's own `AUTHORITY["EPSG",…]` / `ID["EPSG",…]` clause, when
///    it names a code this build knows;
/// 2. a Japan Plane Rectangular zone name plus a datum name (the shape ESRI
///    `.prj` files take, which carry no authority clause);
/// 3. a Web Mercator marker in a projected CRS's name;
/// 4. a WGS 84 marker in a geographic CRS's *name or datum name* — never
///    anywhere in the text, and never with `TOWGS84[…]` still in it.
///
/// When nothing in that list resolves to a CRS this build knows, the code the
/// string *declares* is returned anyway (if it declares one) so a refusal can
/// name it — callers gate on [`epsg::is_supported`], not on `Some`/`None`.
/// [`None`] means the string named no code at all.
#[must_use]
pub fn resolve_epsg(wkt: &str) -> Option<u32> {
    let info = parse_wkt(wkt);
    resolve_epsg_from(&info, wkt)
}

/// [`resolve_epsg`] for a [`WktInfo`] already parsed by the caller.
#[must_use]
pub fn resolve_epsg_from(info: &WktInfo, wkt: &str) -> Option<u32> {
    supported_epsg_from(info, wkt)
        .map(epsg::canonical)
        .or(info.authority_epsg)
}

/// The resolution rules proper — see [`resolve_epsg`]. Answers only with codes
/// [`epsg::is_supported`] accepts.
fn supported_epsg_from(info: &WktInfo, wkt: &str) -> Option<u32> {
    // 1. The authority clause. Only an EPSG (or absent — some writers omit the
    // authority name) code is acted on; an ESRI or IGNF code numbering is a
    // different namespace entirely.
    if let Some(code) = info.authority_epsg {
        let authority_is_epsg = info
            .authority_name
            .as_deref()
            .is_none_or(|name| name == "EPSG");
        if authority_is_epsg && epsg::is_supported(code) {
            return Some(code);
        }
        // ESRI writes Web Mercator as ESRI:102100. The number happens to be one
        // `epsg::canonical` folds, but it is reached here through the ESRI
        // namespace rather than EPSG's, so it needs its own arm.
        if info.authority_name.as_deref() == Some("ESRI") && code == 102_100 {
            return Some(3857);
        }
    }

    let stripped = strip_shift_and_authority_clauses(wkt);
    let root_name = info
        .name
        .as_deref()
        .map(str::to_ascii_uppercase)
        .unwrap_or_default();
    let datum_name = info
        .datum_name
        .as_deref()
        .map(str::to_ascii_uppercase)
        .unwrap_or_default();

    // 2. A Japan Plane Rectangular zone named in a projected CRS.
    if info.kind == WktKind::Projected
        && let Some(zone) = japan_zone_number(&root_name)
    {
        let base = if root_name.contains("2011") {
            6669
        } else if root_name.contains("2000") {
            2443
        } else if root_name.contains("TOKYO") || datum_name.contains("TOKYO") {
            30161
        } else {
            // A zone with no datum in its name is ambiguous by construction;
            // JGD2011 is the only one still being published, and it agrees
            // with JGD2000 to a few centimetres, so it is the safe reading.
            6669
        };
        let code = base + (zone as u32) - 1;
        if epsg::is_supported(code) {
            return Some(code);
        }
    }

    // 3. Web Mercator, by name.
    if info.kind == WktKind::Projected
        && WEB_MERCATOR_MARKERS
            .iter()
            .any(|marker| root_name.contains(marker))
    {
        return Some(3857);
    }

    // 3b. A UTM zone named in a projected CRS — `WGS_1984_UTM_Zone_54N` and
    // friends. Same reason as the Japanese zones: the ESRI `.prj` form carries
    // no authority clause, and the name is the only signal there is.
    if info.kind == WktKind::Projected
        && let Some((zone, north)) = utm_zone_from_name(&root_name)
        && let Some(code) = utm_epsg_from_name(&root_name, &datum_name, zone, north)
    {
        return Some(code);
    }

    // 4. WGS 84, by name — and ONLY by name. `stripped` is consulted for the
    // datum only when the parse found no datum node at all (a bare
    // `GEOGCS["WGS 84"]` with nothing else in it).
    if info.kind.is_geographic() {
        let wgs84_named = WGS84_NAME_MARKERS
            .iter()
            .any(|marker| root_name.contains(marker) || datum_name.contains(marker));
        if wgs84_named {
            return Some(4326);
        }
        for (hint, code) in JAPANESE_NAME_HINTS {
            if root_name.contains(hint) || datum_name.contains(hint) {
                return Some(code);
            }
        }
        // Bare `GEOGCS["WGS 84"]`-style input with no datum node: fall back to
        // the stripped text, which by construction can no longer contain the
        // `TOWGS84` false positive.
        if info.datum_name.is_none()
            && WGS84_NAME_MARKERS
                .iter()
                .any(|marker| stripped.contains(marker))
        {
            return Some(4326);
        }
    }

    None
}

/// The first quoted name in a WKT string — its CRS name — or a truncated copy
/// of the text when it has none, so a refusal always says *what* it refused.
#[must_use]
pub fn crs_label(wkt: &str) -> String {
    let info = parse_wkt(wkt);
    if let Some(name) = info
        .name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return name.to_string();
    }
    let mut quoted = wkt.split('"');
    let _before = quoted.next();
    if let Some(name) = quoted.next().map(str::trim).filter(|s| !s.is_empty()) {
        return name.to_string();
    }
    wkt.trim().chars().take(64).collect()
}

#[cfg(test)]
mod tests;
