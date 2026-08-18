//! OxiGIS core — the platform-independent application model.
//!
//! This crate holds everything that is neither rendering nor windowing:
//! the layer model ([`layer`]), the style model ([`style`]) and the renderer
//! model above it ([`renderer`] — categorized and graduated thematic
//! classification), the coordinate reference systems ([`crs`]), the project
//! file format ([`project`], `.oxigis.json` via serde), and the Processing
//! registry ([`processing`]) that maps OxiGeo functions to auto-generated
//! parameter forms.
//!
//! See the OxiGIS blueprint (`oxigis.md`, kept in the repository's git
//! history) §3 and §6 for the workspace layout and the feature this crate
//! covers in Phase 1.
//!
//! Deliberately self-contained *in its model types*: none of the public model
//! types wrap OxiGeo types directly. OxiGeo's path-only I/O API (see the
//! project's OxiGeo survey notes) doesn't map cleanly onto "a layer's data
//! source" as a plain, serde-friendly, wasm32-safe value —
//! [`layer::RasterSource`] and [`layer::VectorSource`] instead describe
//! *where* data comes from (a URL template, a file path, inline text) as
//! plain strings. Reading that data, and executing a
//! [`processing::ToolExecutor`] against it, is a shell's job by design —
//! not a gap to be filled here. `oxigis-ui` is the reference shell for
//! both.
//!
//! The one place this crate *calls* OxiGeo is [`crs::reproject`], which
//! drives the `proj` feature's ellipsoidal Transverse Mercator and Helmert
//! datum shift so a dataset in a non-WGS 84 CRS can be loaded at all. Even
//! there the OxiGeo types stay inside the module: [`crs::Crs`] is a plain
//! `(EPSG code, optional WKT)` pair and [`crs::Reprojector`] is a `Copy`
//! struct of `f64`s, so no shell ever needs OxiGeo's projection stack by
//! name.

#![forbid(unsafe_code)]

pub mod crs;
pub mod error;
pub mod layer;
pub mod processing;
pub mod project;
pub mod renderer;
pub mod style;
mod util;

pub use crs::{
    AxisOrder, Crs, CrsDef, Datum, EPSG_UNKNOWN, EPSG_WEB_MERCATOR, EPSG_WGS84, Projection,
    ReprojectError, Reprojector,
};
pub use error::{CoreError, CoreResult};
pub use layer::{
    ArchiveFormat, ArchiveRef, Layer, LayerId, LayerKind, LayerStack, RasterSource, VectorSource,
    VectorTilePaint, archive_refusal,
};
pub use processing::{
    ParamKind, ParamSpec, ProcessingRegistry, ToolContext, ToolDescriptor, ToolExecutor,
    builtin_registry,
};
pub use project::{CURRENT_FORMAT_VERSION, Project, ProjectBasemap, View};
pub use renderer::{
    AttrRef, AttrValue, Attributes, CategoryClass, Classification, GraduatedClass,
    MAX_STYLE_CLASSES, NoAttributes, Renderer, RendererKind, class_over_family, recolor_style,
    style_color,
};
pub use style::{
    CircleStyle, Color, FamilySet, FamilyStyles, FillStyle, GeometryFamily, LabelOrientation,
    LabelWeight, LayerStyle, LayerStyleSet, LineStyle, StyleSlot, SymbolStyle,
};

/// Crate version, re-exported so shells can display it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adding_the_renderer_did_not_inflate_the_style_set() {
        // `LayerStyleSet` is carried BY VALUE through the undo stack, the
        // layer snapshot, the render-thread op queue and the print capture, so
        // its size is a cost every one of those pays — including for the
        // overwhelmingly common unclassified layer, which needs no
        // classification at all. Both classified variants are therefore boxed
        // (see `renderer::CategorizedSpec`), which keeps the whole enum at one
        // pointer plus its discriminant.
        //
        // The bound is deliberately loose: this pins the ORDER of magnitude
        // (a pointer, not an inline class list), not an exact layout, so a
        // field added to `LayerStyle` does not fail it spuriously.
        let renderer = size_of::<crate::renderer::Renderer>();
        assert!(
            renderer <= 2 * size_of::<usize>(),
            "a Renderer must stay pointer-sized, got {renderer} bytes",
        );
        let set = size_of::<LayerStyleSet>();
        let without = size_of::<LayerStyle>() + size_of::<crate::style::FamilyStyles>();
        assert!(
            set <= without + 2 * size_of::<usize>(),
            "the renderer field must cost a pointer, not a class list \
             ({set} bytes against {without} without it)",
        );
    }

    #[test]
    fn version_matches_cargo_toml() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }
}
