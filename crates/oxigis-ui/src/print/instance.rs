// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! fvar named-instance selection for the PDF export (print v1.3 item C).
//!
//! ttf-parser 0.25.1 exposes the `fvar` AXES only — there is no
//! named-instance API — so the instance records are hand-parsed from the
//! raw table. Every read goes through `get()?`: this parses
//! attacker-supplied web-font bytes on the `oxigis-web` path, so a
//! truncated or hostile table must answer [`None`], never panic.
//!
//! Both observed record layouts are handled: `instanceSize = 4 +
//! axisCount·4` (no `postScriptNameID` — bahnschrift, SegUIVar) and
//! `6 + axisCount·4` (with it — NotoSansJP-VF).

use oxigis_core::LabelWeight;
use ttf_parser::{Face, Tag};

/// The instance a `PrintOnly` face is normalised to, per requested weight
/// (print/text v1.4, D-W6): what a reader expects body text and bold text to
/// be. Still not a knob — the style model offers exactly the two weights
/// [`LabelWeight`] names, and any number between them would be a guess.
pub(super) fn target_weight(weight: LabelWeight) -> f32 {
    match weight {
        LabelWeight::Regular => 400.0,
        LabelWeight::Bold => 700.0,
    }
}

/// The chosen fvar instance, or nothing when the face should embed at its
/// default (static faces, CFF faces, and — the byte-identity gate — a
/// variable face whose nearest instance IS the default).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ChosenInstance {
    /// Every axis in fvar order; non-`wght` axes keep their defaults.
    pub coordinates: Vec<([u8; 4], f32)>,
    /// The `wght` coordinate actually chosen — feeds `/FontWeight` and
    /// `StemV`.
    pub weight: f32,
    /// The fvar subfamily name (e.g. "Regular"), when resolvable.
    pub name: Option<String>,
}

/// One parsed fvar axis.
struct AxisRecord {
    tag: [u8; 4],
    min: f32,
    default: f32,
    max: f32,
}

/// One parsed fvar named instance.
struct InstanceRecord {
    subfamily_name_id: u16,
    coordinates: Vec<f32>,
}

fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*bytes.get(at)?, *bytes.get(at + 1)?]))
}

/// A 16.16 fixed-point value.
fn read_fixed(bytes: &[u8], at: usize) -> Option<f32> {
    let raw = i32::from_be_bytes([
        *bytes.get(at)?,
        *bytes.get(at + 1)?,
        *bytes.get(at + 2)?,
        *bytes.get(at + 3)?,
    ]);
    Some(raw as f32 / 65536.0)
}

/// Parses the raw `fvar` table into axes and named instances.
fn parse_fvar(table: &[u8]) -> Option<(Vec<AxisRecord>, Vec<InstanceRecord>)> {
    let axes_offset = usize::from(read_u16(table, 4)?);
    let axis_count = usize::from(read_u16(table, 8)?);
    let axis_size = usize::from(read_u16(table, 10)?);
    let instance_count = usize::from(read_u16(table, 12)?);
    let instance_size = usize::from(read_u16(table, 14)?);
    if axis_count == 0 || axis_size < 20 {
        return None;
    }
    // A crafted count cannot drive an unbounded allocation: every record
    // must fit inside the table.
    if axis_count.saturating_mul(axis_size) > table.len()
        || instance_count.saturating_mul(instance_size) > table.len()
    {
        return None;
    }
    let mut axes = Vec::with_capacity(axis_count);
    for index in 0..axis_count {
        let at = axes_offset.checked_add(index.checked_mul(axis_size)?)?;
        let tag = [
            *table.get(at)?,
            *table.get(at + 1)?,
            *table.get(at + 2)?,
            *table.get(at + 3)?,
        ];
        axes.push(AxisRecord {
            tag,
            min: read_fixed(table, at + 4)?,
            default: read_fixed(table, at + 8)?,
            max: read_fixed(table, at + 12)?,
        });
    }
    // Instances follow the axis array; `instanceSize` decides whether the
    // trailing `postScriptNameID` exists — either way the coordinates sit
    // at offset 4.
    let minimum = 4 + axis_count * 4;
    if instance_size < minimum {
        return None;
    }
    let instances_offset = axes_offset.checked_add(axis_count.checked_mul(axis_size)?)?;
    let mut instances = Vec::with_capacity(instance_count);
    for index in 0..instance_count {
        let at = instances_offset.checked_add(index.checked_mul(instance_size)?)?;
        let subfamily_name_id = read_u16(table, at)?;
        let mut coordinates = Vec::with_capacity(axis_count);
        for axis in 0..axis_count {
            coordinates.push(read_fixed(table, at + 4 + axis * 4)?);
        }
        instances.push(InstanceRecord {
            subfamily_name_id,
            coordinates,
        });
    }
    Some((axes, instances))
}

/// The face's English name for `name_id`, when the `name` table has a
/// Unicode-decodable record for it.
fn name_for(face: &Face<'_>, name_id: u16) -> Option<String> {
    face.names()
        .into_iter()
        .filter(|name| name.name_id == name_id)
        .find_map(|name| name.to_string())
}

/// Picks the instance a `PrintOnly` variable face embeds at.
///
/// Rules, in order (verbatim from v1.3, with `target` now supplied by the
/// caller): named instances first — minimise `|wght − target|`, ties broken
/// by the other axes' distance from their own defaults, then by the lowest
/// record index (deterministic) — else clamp `wght` into its range with every
/// other axis at its default. **If the chosen coordinates equal the fvar
/// defaults, answer [`None`]**: the untouched `subset_with_gid_set_at_face_mapped`
/// call then runs, which turns the measured default-location byte-identity
/// into a code path rather than a promise.
pub(super) fn choose_instance(
    face: &Face<'_>,
    raw_fvar: &[u8],
    target: f32,
) -> Option<ChosenInstance> {
    let (axes, instances) = parse_fvar(raw_fvar)?;
    let wght_index = axes.iter().position(|axis| axis.tag == *b"wght")?;
    let chosen: (Vec<f32>, Option<u16>) = if instances.is_empty() {
        let clamped: Vec<f32> = axes
            .iter()
            .enumerate()
            .map(|(index, axis)| {
                if index == wght_index {
                    target.clamp(axis.min, axis.max)
                } else {
                    axis.default
                }
            })
            .collect();
        (clamped, None)
    } else {
        let mut best: Option<(f32, f32, usize)> = None;
        for (index, instance) in instances.iter().enumerate() {
            let weight = *instance.coordinates.get(wght_index)?;
            let primary = (weight - target).abs();
            let secondary: f32 = instance
                .coordinates
                .iter()
                .zip(&axes)
                .enumerate()
                .filter(|(axis_index, _)| *axis_index != wght_index)
                .map(|(_, (value, axis))| (value - axis.default).abs())
                .sum();
            let better = match best {
                None => true,
                Some((p, s, _)) => primary < p || (primary == p && secondary < s),
            };
            if better {
                best = Some((primary, secondary, index));
            }
        }
        let (_, _, index) = best?;
        let instance = instances.get(index)?;
        (
            instance.coordinates.clone(),
            Some(instance.subfamily_name_id),
        )
    };
    let (coordinates, name_id) = chosen;
    // The byte-identity gate: the default location embeds through the
    // untouched `subset` call.
    let at_default = coordinates
        .iter()
        .zip(&axes)
        .all(|(value, axis)| (value - axis.default).abs() < f32::EPSILON);
    if at_default {
        return None;
    }
    let weight = *coordinates.get(wght_index)?;
    Some(ChosenInstance {
        coordinates: axes
            .iter()
            .zip(&coordinates)
            .map(|(axis, &value)| (axis.tag, value))
            .collect(),
        weight,
        name: name_id.and_then(|id| name_for(face, id)),
    })
}

/// The face's raw `fvar` bytes, when it has the table.
///
/// [`ttf_parser::RawFace::table`] **binary-searches** the table directory
/// while `Face::parse` walks it **linearly**, so on a face whose directory is
/// not sorted by tag — which the spec requires but real files occasionally
/// are not — `face.is_variable()` can be true while the binary search misses.
/// Today's failure is already graceful (a `WeightLadder::AxisCannotReachBold`
/// with the regular bytes, logged), so this is a widening rather than a fix:
/// when the search misses, fall back to a linear scan of the 16-byte records.
/// Output-neutral on every spec-conformant face.
pub(super) fn raw_fvar(bytes: &[u8]) -> Option<&[u8]> {
    let raw = ttf_parser::RawFace::parse(bytes, 0).ok()?;
    if let Some(table) = raw.table(Tag::from_bytes(b"fvar")) {
        return Some(table);
    }
    let want = Tag::from_bytes(b"fvar");
    let record = raw
        .table_records
        .into_iter()
        .find(|record| record.tag == want)?;
    let offset = record.offset as usize;
    let length = record.length as usize;
    bytes.get(offset..offset.checked_add(length)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-builds an fvar table: `axes` as (tag, min, def, max), then
    /// `instances` as (name_id, coords), with or without the trailing
    /// `postScriptNameID` slot.
    fn fvar(
        axes: &[([u8; 4], f32, f32, f32)],
        instances: &[(u16, &[f32])],
        with_ps_name: bool,
    ) -> Vec<u8> {
        let axis_size = 20_u16;
        let instance_size = 4 + 4 * axes.len() as u16 + if with_ps_name { 2 } else { 0 };
        let mut out = Vec::new();
        out.extend_from_slice(&1_u16.to_be_bytes()); // majorVersion
        out.extend_from_slice(&0_u16.to_be_bytes()); // minorVersion
        out.extend_from_slice(&16_u16.to_be_bytes()); // axesArrayOffset
        out.extend_from_slice(&2_u16.to_be_bytes()); // reserved
        out.extend_from_slice(&(axes.len() as u16).to_be_bytes());
        out.extend_from_slice(&axis_size.to_be_bytes());
        out.extend_from_slice(&(instances.len() as u16).to_be_bytes());
        out.extend_from_slice(&instance_size.to_be_bytes());
        let fixed = |value: f32| ((value * 65536.0) as i32).to_be_bytes();
        for (tag, min, def, max) in axes {
            out.extend_from_slice(tag);
            out.extend_from_slice(&fixed(*min));
            out.extend_from_slice(&fixed(*def));
            out.extend_from_slice(&fixed(*max));
            out.extend_from_slice(&0_u16.to_be_bytes()); // flags
            out.extend_from_slice(&256_u16.to_be_bytes()); // axisNameID
        }
        for (name_id, coords) in instances {
            out.extend_from_slice(&name_id.to_be_bytes());
            out.extend_from_slice(&0_u16.to_be_bytes()); // flags
            for value in *coords {
                out.extend_from_slice(&fixed(*value));
            }
            if with_ps_name {
                out.extend_from_slice(&257_u16.to_be_bytes());
            }
        }
        out
    }

    /// A face is only needed for the subfamily-name lookup; the bundled
    /// static Noto serves (the name simply resolves to `None`).
    fn any_face() -> Vec<u8> {
        oxifont_bundled::NOTO_SANS_REGULAR.to_vec()
    }

    #[test]
    fn named_instances_parse_from_both_record_layouts() {
        let bytes = any_face();
        let face = Face::parse(&bytes, 0).expect("bundled face");
        for with_ps in [false, true] {
            let table = fvar(
                &[
                    (*b"wght", 100.0, 100.0, 900.0),
                    (*b"wdth", 75.0, 100.0, 100.0),
                ],
                &[(17, &[100.0, 100.0][..]), (18, &[400.0, 100.0][..])],
                with_ps,
            );
            let chosen = choose_instance(&face, &table, target_weight(LabelWeight::Regular))
                .expect("a thin-default VF chooses Regular");
            assert_eq!(chosen.weight, 400.0);
            assert_eq!(chosen.coordinates[0], (*b"wght", 400.0));
            assert_eq!(chosen.coordinates[1], (*b"wdth", 100.0));
        }
    }

    #[test]
    fn an_instance_equal_to_the_fvar_default_selects_none() {
        // SegUIVar-shaped: default wght 400 — the nearest instance IS the
        // default, so the untouched `subset` call runs (byte-identity as a
        // code path).
        let bytes = any_face();
        let face = Face::parse(&bytes, 0).expect("bundled face");
        let table = fvar(
            &[(*b"wght", 300.0, 400.0, 700.0)],
            &[(17, &[400.0][..]), (18, &[700.0][..])],
            false,
        );
        assert_eq!(
            choose_instance(&face, &table, target_weight(LabelWeight::Regular)),
            None
        );
    }

    #[test]
    fn no_named_instances_clamps_the_weight_axis_only() {
        let bytes = any_face();
        let face = Face::parse(&bytes, 0).expect("bundled face");
        // Thin-default, no named instances: clamp to 400.
        let table = fvar(&[(*b"wght", 100.0, 100.0, 900.0)], &[], false);
        let chosen = choose_instance(&face, &table, target_weight(LabelWeight::Regular))
            .expect("clamp fallback");
        assert_eq!(chosen.weight, 400.0);
        // A face whose whole range is below the target clamps to its max.
        let table = fvar(&[(*b"wght", 100.0, 100.0, 300.0)], &[], false);
        let chosen = choose_instance(&face, &table, target_weight(LabelWeight::Regular))
            .expect("clamp fallback");
        assert_eq!(chosen.weight, 300.0);
    }

    #[test]
    fn truncated_or_garbage_tables_answer_none_without_panicking() {
        let bytes = any_face();
        let face = Face::parse(&bytes, 0).expect("bundled face");
        let table = fvar(
            &[(*b"wght", 100.0, 100.0, 900.0)],
            &[(17, &[400.0][..])],
            false,
        );
        for cut in [0, 4, 9, 15, 17, 21, table.len() - 1] {
            assert_eq!(
                choose_instance(&face, &table[..cut], target_weight(LabelWeight::Regular)),
                None,
                "truncation at {cut} must refuse"
            );
        }
        // A declared instance count that cannot fit refuses before any
        // allocation.
        let mut hostile = table.clone();
        hostile[12] = 0xFF;
        hostile[13] = 0xFF;
        assert_eq!(
            choose_instance(&face, &hostile, target_weight(LabelWeight::Regular)),
            None
        );
        // No `wght` axis at all.
        let table = fvar(&[(*b"wdth", 75.0, 100.0, 125.0)], &[], false);
        assert_eq!(
            choose_instance(&face, &table, target_weight(LabelWeight::Regular)),
            None
        );
    }

    #[test]
    fn a_static_face_has_no_fvar_and_selects_no_instance() {
        let bytes = any_face();
        assert!(raw_fvar(&bytes).is_none(), "the bundled Noto is static");
    }

    #[test]
    fn the_bold_target_selects_the_bold_instance_of_the_same_face() {
        // D-W6 L1: the SAME face, the SAME rules, a different target — the
        // only thing v1.4 changes about instance selection.
        let bytes = any_face();
        let face = Face::parse(&bytes, 0).expect("bundled face");
        let table = fvar(
            &[(*b"wght", 100.0, 400.0, 900.0)],
            &[(17, &[400.0][..]), (18, &[700.0][..]), (19, &[900.0][..])],
            false,
        );
        assert_eq!(target_weight(LabelWeight::Bold), 700.0);
        let chosen = choose_instance(&face, &table, target_weight(LabelWeight::Bold))
            .expect("Bold picks the 700 instance");
        assert_eq!(chosen.weight, 700.0);
        // And Regular still picks the default, i.e. None (byte-identity).
        assert_eq!(
            choose_instance(&face, &table, target_weight(LabelWeight::Regular)),
            None,
        );
    }

    #[test]
    fn a_bold_target_clamps_when_the_axis_stops_short() {
        // D-W6 L2's input: an axis whose maximum is below the bold target.
        // Selection still answers, and the CALLER decides that 500 is not
        // bold enough to be called bold.
        let bytes = any_face();
        let face = Face::parse(&bytes, 0).expect("bundled face");
        let table = fvar(&[(*b"wght", 300.0, 400.0, 500.0)], &[], false);
        let chosen = choose_instance(&face, &table, target_weight(LabelWeight::Bold))
            .expect("clamp fallback");
        assert_eq!(chosen.weight, 500.0);
    }

    // --- print/text v1.5, Part D cleanup 3: the unsorted table directory ---

    /// Builds a minimal sfnt whose table directory lists `tables` in the
    /// order given — NOT necessarily sorted by tag, which is what
    /// `RawFace::table`'s binary search assumes.
    fn sfnt(tables: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
        let count = tables.len();
        let header = 12 + count * 16;
        let mut directory = Vec::with_capacity(header);
        directory.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
        directory.extend_from_slice(&(count as u16).to_be_bytes());
        directory.extend_from_slice(&0_u16.to_be_bytes()); // searchRange
        directory.extend_from_slice(&0_u16.to_be_bytes()); // entrySelector
        directory.extend_from_slice(&0_u16.to_be_bytes()); // rangeShift
        let mut body = Vec::new();
        for (tag, data) in tables {
            let offset = header + body.len();
            directory.extend_from_slice(tag);
            directory.extend_from_slice(&0_u32.to_be_bytes()); // checkSum
            directory.extend_from_slice(&(offset as u32).to_be_bytes());
            directory.extend_from_slice(&(data.len() as u32).to_be_bytes());
            body.extend_from_slice(data);
            // 4-byte alignment, as every real file has.
            while body.len() % 4 != 0 {
                body.push(0);
            }
        }
        directory.extend_from_slice(&body);
        directory
    }

    /// One axis, one instance — enough for `parse_fvar` to answer.
    fn one_axis_fvar() -> Vec<u8> {
        fvar(&[(*b"wght", 100.0, 400.0, 900.0)], &[(2, &[700.0])], true)
    }

    #[test]
    fn raw_fvar_finds_the_table_in_a_sorted_directory() {
        // The spec-conformant case, which must keep going through the binary
        // search and must be byte-identical to what it always returned.
        let table = one_axis_fvar();
        let bytes = sfnt(&[
            (*b"cmap", vec![0_u8; 8]),
            (*b"fvar", table.clone()),
            (*b"name", vec![0_u8; 8]),
        ]);
        let found = raw_fvar(&bytes).expect("a sorted directory resolves");
        assert_eq!(found, table.as_slice());
    }

    #[test]
    fn raw_fvar_finds_the_table_in_an_unsorted_directory() {
        // `name` before `fvar` before `cmap`: the binary search misses and
        // the linear scan answers. Before this widening the face embedded at
        // its default with a logged `AxisCannotReachBold` — graceful, but
        // wrong for a face that really can reach bold.
        let table = one_axis_fvar();
        let bytes = sfnt(&[
            (*b"name", vec![0_u8; 8]),
            (*b"fvar", table.clone()),
            (*b"cmap", vec![0_u8; 8]),
        ]);
        let found = raw_fvar(&bytes).expect("an unsorted directory still resolves");
        assert_eq!(found, table.as_slice());
        // And the widening is not a shortcut past the parser: the recovered
        // bytes really do drive instance selection.
        let (axes, instances) = parse_fvar(found).expect("the recovered table parses");
        assert_eq!(axes.len(), 1);
        assert_eq!(instances.len(), 1);
    }

    #[test]
    fn raw_fvar_says_no_when_the_face_has_no_fvar_at_all() {
        let bytes = sfnt(&[(*b"cmap", vec![0_u8; 8]), (*b"name", vec![0_u8; 8])]);
        assert!(raw_fvar(&bytes).is_none());
        // And a face whose fvar record points past the end is refused, not
        // panicked — this parses attacker-supplied web-font bytes.
        let mut hostile = sfnt(&[(*b"fvar", one_axis_fvar())]);
        let length_at = 12 + 12;
        hostile[length_at..length_at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(raw_fvar(&hostile).is_none());
    }
}
