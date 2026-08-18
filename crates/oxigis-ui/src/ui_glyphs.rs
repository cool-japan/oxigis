// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Every non-ASCII codepoint the OxiGIS panels DRAW, in one table, with the
//! ONE pin that proves egui 0.35's default PROPORTIONAL chain actually
//! renders each of them.
//!
//! # Which chain, and why a monospace sighting proves nothing
//!
//! egui 0.35's proportional family is `[Ubuntu-Light, NotoEmoji-Regular,
//! emoji-icon-font]` (`epaint-0.35.0/src/text/fonts.rs:539-556`); `Hack` is in
//! the MONOSPACE family only, which is why a monospace-context sighting of a
//! glyph is no evidence at all for a proportional button label. Every constant
//! below is therefore measured in the proportional chain, at the three sizes
//! the shipped panels lay text out at.
//!
//! # A miss is a BOX, not a blank
//!
//! A codepoint no face owns is **not** drawn blank. The layout path
//! (`Font::glyph_info`, `epaint-0.35.0/src/text/font.rs:761-775`) substitutes
//! `PRIMARY_REPLACEMENT_CHAR`, which is `◻` U+25FB WHITE MEDIUM SQUARE
//! (`epaint-0.35.0/src/text/fonts.rs:643-646`), and lays it out at **that
//! glyph's real advance**. So every miss paints the SAME visible hollow box.
//! (The editing v1.4 record called this a "zero-width replacement"; that
//! sentence described `Fonts::glyph_width`, which resolves to the replacement
//! *face* and returns `0.0` — a real number, but not the one the layout path
//! uses.)
//!
//! # Why the pins compare atlas slots and not widths
//!
//! Because a box has a width. `glyph_width('\u{25fb}')` is `16.504` at
//! proportional 13.0 while U+25FB **is** the tofu, so an advance-width
//! predicate would applaud a substitution that picked the replacement box
//! itself. A codepoint no face owns is routed to the replacement's *atlas
//! slot*, so its `uv_rect` is bit-identical to a known-missing control's, and
//! two real glyphs never share a slot. `uv_rect.min` / `.max` are `[u16; 2]`
//! texel coordinates, so the comparison is exact with no float tolerance. See
//! `no_ui_glyph_paints_the_replacement_box` below.
//!
//! `Fonts::has_glyph` is **not** the predicate either: it is
//! `resolve_face(c) != replacement_face_key`
//! (`epaint-0.35.0/src/text/font.rs:719-723`) and the replacement face **is**
//! `NotoEmoji-Regular`, so it answers `false` for the shipped and visibly
//! working ✖ ⚠ 🌐 ☑. Never strengthen the pin to `has_glyph`.
//!
//! # House rule
//!
//! **An icon glyph must be spelled as a constant of this module, never as a
//! literal character.** Where the site builds a composite label
//! (`"▣ Browse"`) a constant cannot be used — `mode_glyph` returns
//! `&'static str` and `concat!` does not accept consts — so the glyph is
//! spelled as a `\u{…}` escape instead, and
//! `every_drawn_escape_is_in_the_table` proves every such escape in the seven
//! drawing modules is a member of [`ALL`]. A bare literal character is neither
//! gated nor greppable, which is why it is forbidden.
//!
//! # Carried risk
//!
//! These glyphs ride egui's bundled faces, which OxiGIS does not control. An
//! egui/epaint bump that drops or re-maps one of them silently reverts to
//! boxes; the pins below turn that into a loud test failure instead. The
//! residual risk is a future agent weakening a pin, or installing a custom
//! `FontDefinitions` and bypassing the pins' context instead of reflecting it
//! — no crate in this workspace calls `Context::set_fonts` today, and the pins
//! assume that.

/// "Move this layer up the stack" — the layer panel's reorder button — and the
/// attribute table's **ascending** sort indicator. One truth for "up", so the
/// two panels can never drift apart.
///
/// U+2B06 is owned by three of the four bundled faces, and its laid-out
/// advance is byte-identical to the replacement box's, so adopting it moved no
/// row by a pixel.
pub const MOVE_UP: &str = "\u{2b06}";

/// "Move this layer down the stack", and the **descending** sort indicator.
pub const MOVE_DOWN: &str = "\u{2b07}";

/// Remove/delete: the layer panel's row button, the attribute form's
/// property-row button and the edit toolbar's two Delete buttons.
pub const REMOVE: &str = "\u{2716}";

/// The glyph on an XYZ row's "draw this as the basemap" toggle.
///
/// U+1F310 GLOBE WITH MERIDIANS, chosen because it is in **two** of the three
/// faces of the proportional chain (`NotoEmoji-Regular` and
/// `emoji-icon-font`) — the same block the ✖ remove button already draws from.
pub const BASEMAP_TOGGLE: &str = "\u{1f310}";

/// Warning: the basemap refusal banner, the toolbar's issue badge and the Edit
/// window's issue list.
pub const WARNING: &str = "\u{26a0}";

/// The edit toolbar's Undo button.
pub const UNDO: &str = "\u{21a9}";

/// The edit toolbar's Redo button.
pub const REDO: &str = "\u{21aa}";

/// The toolbar's snapping toggle.
pub const SNAP: &str = "\u{2611}";

/// Tool mark: browse (editing off).
pub const MODE_BROWSE: &str = "\u{25a3}";

/// Tool mark: select.
pub const MODE_SELECT: &str = "\u{25b6}";

/// Tool mark: draw point.
pub const MODE_POINT: &str = "\u{2022}";

/// Tool mark: draw line.
///
/// An approximation, recorded as one: U+27CB, the diagonal the label spelled
/// before editing v1.5, lives in Miscellaneous Mathematical Symbols A, which
/// has **zero** coverage in the chain — as does the whole Box Drawing block.
/// The button carries the word "Line", so the mark is decoration rather than
/// the affordance.
pub const MODE_LINE: &str = "\u{2197}";

/// Tool mark: draw polygon. U+2B1F BLACK PENTAGON, the exact semantic twin of
/// the U+2B20 white pentagon the label spelled before editing v1.5 and the
/// chain does not own.
pub const MODE_POLYGON: &str = "\u{2b1f}";

/// The edit toolbar's Edit-window toggle.
pub const EDIT_WINDOW: &str = "\u{270f}";

/// Separator inside a compound source name, e.g. `atlas.gpkg › roads`.
pub const PATH_SEPARATOR: &str = "\u{203a}";

/// Prose: an elided continuation, e.g. `Open…`.
pub const ELLIPSIS: &str = "\u{2026}";

/// Prose: the clause break this codebase words its status lines with.
pub const EM_DASH: &str = "\u{2014}";

/// Prose: an inline list separator.
pub const MIDDLE_DOT: &str = "\u{b7}";

/// Prose: opening quotation mark around a layer name.
pub const LEFT_QUOTE: &str = "\u{201c}";

/// Prose: closing quotation mark around a layer name.
pub const RIGHT_QUOTE: &str = "\u{201d}";

/// Prose: the copyright sign that opens a tile attribution credit line.
pub const COPYRIGHT: &str = "\u{a9}";

/// `(glyph, the site that draws it)` — the pins' input.
///
/// Adding a drawn glyph without adding a row here is what the membership pin
/// forbids.
pub const ALL: &[(&str, &str)] = &[
    (MOVE_UP, "layer_panel Move up / table_panel sort ascending"),
    (
        MOVE_DOWN,
        "layer_panel Move down / table_panel sort descending",
    ),
    (
        REMOVE,
        "layer_panel Remove / form property row / toolbar Delete",
    ),
    (BASEMAP_TOGGLE, "layer_panel XYZ basemap toggle"),
    (WARNING, "layer_panel refusal banner / toolbar issue badge"),
    (UNDO, "edit toolbar Undo"),
    (REDO, "edit toolbar Redo"),
    (SNAP, "edit toolbar Snap"),
    (MODE_BROWSE, "edit toolbar Browse"),
    (MODE_SELECT, "edit toolbar Select"),
    (MODE_POINT, "edit toolbar Point"),
    (MODE_LINE, "edit toolbar Line"),
    (MODE_POLYGON, "edit toolbar Polygon"),
    (EDIT_WINDOW, "edit toolbar Edit window toggle"),
    (PATH_SEPARATOR, "edit_glue gpkg source_display_name"),
    (ELLIPSIS, "layer_panel Open… / toolbar Edit…"),
    (EM_DASH, "status-line clause break"),
    (MIDDLE_DOT, "edit_window issue list / table_panel selection"),
    (LEFT_QUOTE, "layer-name quoting in status lines"),
    (RIGHT_QUOTE, "layer-name quoting in status lines"),
    (COPYRIGHT, "tile attribution credit line"),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The proportional sizes the shipped panels lay text out at. 13.0 is
    /// egui's default `TextStyle::Button` (`egui-0.35.0/src/style.rs:1414`);
    /// 12.0 and 14.0 bracket it, because atlas packing is size-dependent and a
    /// glyph that resolves at one size is no proof about another.
    const SIZES: [f32; 3] = [12.0, 13.0, 14.0];

    /// A codepoint no bundled face owns, used as the known-missing control.
    const CONTROL: char = '\u{6f22}';

    /// One glyph's rectangle in the font atlas, as exact integer texel
    /// coordinates: `(top-left, bottom-right-exclusive)`. Two real glyphs
    /// never share one; every codepoint no face owns shares the replacement
    /// box's.
    type Slot = ([u16; 2], [u16; 2]);

    /// The atlas slot each `char` of `text` lands in, as exact integer texel
    /// coordinates.
    ///
    /// Every call must happen inside ONE pass of ONE context, with no
    /// `begin_pass` between: atlas packing is only stable within a pass, and
    /// that stability is what makes the comparison exact.
    fn slots(ctx: &egui::Context, size: f32, text: &str) -> Vec<Slot> {
        let font = egui::FontId::proportional(size);
        let galley = ctx
            .fonts_mut(|fonts| fonts.layout_no_wrap(text.to_owned(), font, egui::Color32::WHITE));
        galley
            .rows
            .iter()
            .flat_map(|row| row.glyphs.iter())
            .map(|glyph| (glyph.uv_rect.min, glyph.uv_rect.max))
            .collect()
    }

    /// A context whose first pass has run, so the font set is live.
    ///
    /// Built explicitly rather than through `egui::__run_test_ui`, which
    /// installs `FontDefinitions::empty()` to save CPU
    /// (`egui-0.35.0/src/lib.rs:682-688`) and would therefore answer about no
    /// fonts at all. These pins have to ask egui 0.35's REAL default fonts.
    fn measured_context() -> egui::Context {
        let ctx = egui::Context::default();
        let _output = ctx.run_ui(egui::RawInput::default(), |_ui| {});
        ctx
    }

    #[test]
    fn no_ui_glyph_paints_the_replacement_box() {
        let ctx = measured_context();
        for size in SIZES {
            let control = slots(&ctx, size, &CONTROL.to_string());
            assert_eq!(control.len(), 1, "the control is one glyph");
            for (glyph, site) in ALL {
                let measured = slots(&ctx, size, glyph);
                assert_eq!(
                    measured.len(),
                    glyph.chars().count(),
                    "{site}: one slot per char"
                );
                for (ch, slot) in glyph.chars().zip(measured) {
                    // The advance width is printed only as a diagnostic: it is
                    // a sound detector (every broken codepoint measures 0.000)
                    // but not an exact predicate, because the replacement BOX
                    // has a width of its own.
                    let advance = ctx.fonts_mut(|fonts| {
                        fonts.glyph_width(&egui::FontId::proportional(size), ch)
                    });
                    assert_ne!(
                        slot, control[0],
                        "{site}: U+{:04X} lands in the replacement-box slot at proportional \
                         {size} (advance {advance}), so it draws as a hollow box, not an \
                         affordance",
                        ch as u32
                    );
                }
            }
        }
    }

    #[test]
    fn no_two_ui_glyphs_paint_the_same_slot() {
        let ctx = measured_context();
        for size in SIZES {
            let measured: Vec<(&str, Slot)> = ALL
                .iter()
                .map(|(glyph, site)| {
                    let slot = slots(&ctx, size, glyph);
                    (*site, slot.first().copied().unwrap_or_default())
                })
                .collect();
            for (index, (site, slot)) in measured.iter().enumerate() {
                for (other_site, other_slot) in &measured[index + 1..] {
                    assert_ne!(
                        slot.0, other_slot.0,
                        "{site} and {other_site} share an atlas slot at proportional {size}, \
                         so the two affordances are visually identical"
                    );
                }
            }
        }
    }

    /// The modules that DRAW. A glyph reaching the screen from anywhere else
    /// is outside what these pins can promise, so a new drawing module belongs
    /// on this list the day it is written.
    ///
    /// `crate::print` is deliberately absent: its `\u{…}` tables are
    /// bidi/mirroring **data**, not labels, and it embeds its own fonts rather
    /// than using egui's.
    const DRAWING_MODULES: &[(&str, &str)] = &[
        ("layer_panel.rs", include_str!("layer_panel.rs")),
        ("table_panel.rs", include_str!("table_panel.rs")),
        ("edit/toolbar.rs", include_str!("edit/toolbar.rs")),
        ("edit/form.rs", include_str!("edit/form.rs")),
        ("app/edit_window.rs", include_str!("app/edit_window.rs")),
        ("app/project_edit.rs", include_str!("app/project_edit.rs")),
        ("app/edit_glue.rs", include_str!("app/edit_glue.rs")),
    ];

    /// Every codepoint spelled as a `\u{…}` escape in `source`, in order.
    ///
    /// Escapes are unambiguous to scan and — verified — never appear inside a
    /// comment in these files, so no comment stripping is needed. A malformed
    /// or non-scalar escape cannot exist here: the file compiles.
    fn escaped_codepoints(source: &str) -> Vec<char> {
        let mut found = Vec::new();
        let mut rest = source;
        while let Some(start) = rest.find("\\u{") {
            rest = &rest[start + 3..];
            let Some(end) = rest.find('}') else {
                break;
            };
            if let Ok(value) = u32::from_str_radix(&rest[..end], 16)
                && let Some(ch) = char::from_u32(value)
            {
                found.push(ch);
            }
            rest = &rest[end + 1..];
        }
        found
    }

    #[test]
    fn every_drawn_escape_is_in_the_table() {
        let known: Vec<char> = ALL.iter().flat_map(|(glyph, _)| glyph.chars()).collect();
        let mut checked = 0_usize;
        for (module, source) in DRAWING_MODULES {
            for ch in escaped_codepoints(source) {
                // Below U+00A0 an escape is spelling an ASCII or control
                // character for readability, not drawing a glyph.
                if (ch as u32) < 0xA0 {
                    continue;
                }
                assert!(
                    known.contains(&ch),
                    "{module} draws U+{:04X} but ui_glyphs::ALL does not list it, so nothing \
                     proves it is not a hollow replacement box",
                    ch as u32
                );
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "the scanner found no escapes at all, so it is proving nothing"
        );
    }
}
