// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Default paint rules for a vector tile archive, seeded from its own layer
//! names.
//!
//! # This file is taste, and it is isolated on purpose
//!
//! Everything else under `archive/` is decoded from bytes: an offset is right
//! or it is wrong. This is not. A PMTiles archive's metadata declares
//! `vector_layers` — `earth`, `water`, `roads`, `boundaries`, `places` in the
//! Protomaps basemap, entirely different names in a municipal export — and
//! something has to turn those names into colours before the user sees
//! anything at all. The alternative, a single hairline for every layer, draws a
//! grey scribble and teaches the user nothing about their data.
//!
//! So: a small case-insensitive substring table ([`RAMP`]) with a neutral
//! fallback, deliberately kept in one file with no other responsibility, so a
//! mismatched archive can be re-tuned here without any risk to the reader. The
//! rules are ordinary [`VectorTilePaint`]s — the same value the layer panel
//! edits and the project file stores — so the moment the user changes one, this
//! file stops mattering for that layer for ever.
//!
//! # Order is painter's order
//!
//! Rules come back grouped fills → lines → circles → symbols, regardless of the
//! order the archive declared its layers in, because that is the order that
//! makes a map legible: landmass under water under roads under labels. Inside a
//! group the archive's own order is preserved.

use oxigis_core::{
    CircleStyle, Color, FillStyle, LayerStyle, LineStyle, SymbolStyle, VectorTilePaint,
};

/// Landmass and land use: the layer everything else is drawn on top of.
pub const ARCHIVE_EARTH_FILL: Color = Color {
    r: 0xE9,
    g: 0xE4,
    b: 0xD8,
    a: 0xFF,
};

/// Water bodies.
pub const ARCHIVE_WATER_FILL: Color = Color {
    r: 0xAF,
    g: 0xC8,
    b: 0xDE,
    a: 0xFF,
};

/// Building footprints: a shade off the landmass, never a colour of its own.
pub const ARCHIVE_BUILDING_FILL: Color = Color {
    r: 0xD8,
    g: 0xD2,
    b: 0xC6,
    a: 0xFF,
};

/// Roads, railways and other transportation lines.
pub const ARCHIVE_ROAD_COLOR: Color = Color {
    r: 0x9A,
    g: 0x8F,
    b: 0x80,
    a: 0xFF,
};

/// Administrative boundaries: present but quiet.
pub const ARCHIVE_BOUNDARY_COLOR: Color = Color {
    r: 0x9B,
    g: 0x8C,
    b: 0x9F,
    a: 0xFF,
};

/// Point features that are not labelled.
pub const ARCHIVE_POINT_COLOR: Color = Color {
    r: 0x55,
    g: 0x5B,
    b: 0x66,
    a: 0xFF,
};

/// Anything the ramp does not recognise, drawn as a hairline so it is visible
/// without pretending to a meaning nobody claimed.
pub const ARCHIVE_NEUTRAL_COLOR: Color = Color {
    r: 0x7E,
    g: 0x84,
    b: 0x8E,
    a: 0xFF,
};

/// Stroke width of a road line, in pixels.
pub const ARCHIVE_ROAD_WIDTH_PX: f32 = 0.9;

/// Stroke width of a boundary line, in pixels.
pub const ARCHIVE_BOUNDARY_WIDTH_PX: f32 = 0.7;

/// Stroke width of an unrecognised layer's hairline, in pixels.
pub const ARCHIVE_NEUTRAL_WIDTH_PX: f32 = 0.5;

/// Radius of an unlabelled point feature, in pixels.
pub const ARCHIVE_POINT_RADIUS_PX: f32 = 1.8;

/// Label size for a place layer, in pixels.
pub const ARCHIVE_LABEL_SIZE_PX: f32 = 11.0;

/// The property a place layer's label text is read from.
///
/// Every vector basemap in circulation spells it `name`; a layer that does not
/// simply produces no labels, which is the correct failure.
pub const ARCHIVE_LABEL_FIELD: &str = "name";

/// Which built-in rule a source-layer name maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchivePaintKind {
    /// Landmass, land use, land cover, parks.
    Earth,
    /// Water bodies.
    Water,
    /// Building footprints.
    Building,
    /// Roads, railways, paths.
    Road,
    /// Administrative boundaries.
    Boundary,
    /// Named places, labelled rather than drawn.
    Place,
    /// Unrecognised: a neutral hairline.
    Neutral,
}

impl ArchivePaintKind {
    /// Painter's order: lower draws first, i.e. underneath.
    const fn rank(self) -> u8 {
        match self {
            Self::Earth => 0,
            Self::Water => 1,
            Self::Building => 2,
            Self::Road => 3,
            Self::Boundary => 4,
            Self::Neutral => 5,
            Self::Place => 6,
        }
    }

    /// The style this kind draws its layer with.
    fn style(self) -> LayerStyle {
        match self {
            Self::Earth => LayerStyle::Fill(FillStyle::new(ARCHIVE_EARTH_FILL)),
            Self::Water => LayerStyle::Fill(FillStyle::new(ARCHIVE_WATER_FILL)),
            Self::Building => LayerStyle::Fill(FillStyle::new(ARCHIVE_BUILDING_FILL)),
            Self::Road => {
                LayerStyle::Line(LineStyle::new(ARCHIVE_ROAD_COLOR, ARCHIVE_ROAD_WIDTH_PX))
            }
            Self::Boundary => LayerStyle::Line(LineStyle::new(
                ARCHIVE_BOUNDARY_COLOR,
                ARCHIVE_BOUNDARY_WIDTH_PX,
            )),
            Self::Place => LayerStyle::Circle(CircleStyle::new(
                ARCHIVE_POINT_RADIUS_PX,
                ARCHIVE_POINT_COLOR,
            )),
            Self::Neutral => LayerStyle::Line(LineStyle::new(
                ARCHIVE_NEUTRAL_COLOR,
                ARCHIVE_NEUTRAL_WIDTH_PX,
            )),
        }
    }
}

/// The name fragments each kind claims, matched case-insensitively as
/// substrings and in this order (first match wins).
///
/// Substrings rather than exact names because real archives spell the same
/// concept a dozen ways — `water`, `waterway`, `water_areas`, `ocean` — and
/// the price of a false positive is a wrong colour, not a wrong map. The
/// order matters where fragments overlap: `landuse` must be tried before the
/// bare `land` it contains, and `boundary` before nothing at all.
pub const RAMP: &[(&str, ArchivePaintKind)] = &[
    ("earth", ArchivePaintKind::Earth),
    ("landcover", ArchivePaintKind::Earth),
    ("landuse", ArchivePaintKind::Earth),
    ("landmass", ArchivePaintKind::Earth),
    ("park", ArchivePaintKind::Earth),
    ("forest", ArchivePaintKind::Earth),
    ("green", ArchivePaintKind::Earth),
    ("water", ArchivePaintKind::Water),
    ("ocean", ArchivePaintKind::Water),
    ("river", ArchivePaintKind::Water),
    ("lake", ArchivePaintKind::Water),
    ("sea", ArchivePaintKind::Water),
    ("building", ArchivePaintKind::Building),
    ("structure", ArchivePaintKind::Building),
    ("road", ArchivePaintKind::Road),
    ("transportation", ArchivePaintKind::Road),
    ("highway", ArchivePaintKind::Road),
    ("street", ArchivePaintKind::Road),
    ("rail", ArchivePaintKind::Road),
    ("path", ArchivePaintKind::Road),
    ("boundar", ArchivePaintKind::Boundary),
    ("border", ArchivePaintKind::Boundary),
    ("admin", ArchivePaintKind::Boundary),
    ("place", ArchivePaintKind::Place),
    ("poi", ArchivePaintKind::Place),
    ("label", ArchivePaintKind::Place),
    ("city", ArchivePaintKind::Place),
    // `land` last of the land family: `landuse` and `landcover` contain it.
    ("land", ArchivePaintKind::Earth),
];

/// Which built-in rule `layer_name` maps to.
///
/// Case-insensitive substring matching against [`RAMP`], falling back to
/// [`ArchivePaintKind::Neutral`].
#[must_use]
pub fn kind_for(layer_name: &str) -> ArchivePaintKind {
    let lower = layer_name.to_ascii_lowercase();
    RAMP.iter()
        .find(|(fragment, _)| lower.contains(fragment))
        .map_or(ArchivePaintKind::Neutral, |(_, kind)| *kind)
}

/// Paint rules for an archive whose vector tiles carry these source layers.
///
/// One geometry rule per layer, plus a [`LayerStyle::Symbol`] rule for each
/// layer the ramp recognises as a place layer, so a basemap archive comes up
/// labelled. Returns an empty list for an empty input, which is what an archive
/// declaring no `vector_layers` gets: the tessellator then draws nothing, and
/// the user styles the layers they can see in the attribute panel.
#[must_use]
pub fn archive_paints(layer_names: &[String]) -> Vec<VectorTilePaint> {
    let mut ranked: Vec<(u8, usize, VectorTilePaint)> = Vec::new();
    for (index, name) in layer_names.iter().enumerate() {
        if name.is_empty() {
            continue;
        }
        let kind = kind_for(name);
        ranked.push((
            kind.rank(),
            index,
            VectorTilePaint::new(name.clone(), kind.style()),
        ));
        if kind == ArchivePaintKind::Place {
            let mut labels = SymbolStyle::new(ARCHIVE_LABEL_FIELD);
            labels.set_text_size(ARCHIVE_LABEL_SIZE_PX);
            // Symbols last of all, so the label pass sees a finished map.
            ranked.push((
                u8::MAX,
                index,
                VectorTilePaint::new(name.clone(), LayerStyle::Symbol(labels)),
            ));
        }
    }
    ranked.sort_by_key(|(rank, index, _)| (*rank, *index));
    ranked.into_iter().map(|(_, _, paint)| paint).collect()
}

/// The rules a vector archive layer is created with.
///
/// A thin alias for [`archive_paints`] that names the *decision* rather than
/// the mechanism, so the add seam reads as "these are the defaults" and a
/// future preference can replace exactly this one call.
#[must_use]
pub fn default_archive_paints(layer_names: &[String]) -> Vec<VectorTilePaint> {
    archive_paints(layer_names)
}
