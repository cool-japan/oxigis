//! Wire-level MVT integer encodings: zigzag varint payloads and geometry
//! command integers.
//!
//! # Scope
//!
//! Only the two encodings that MVT geometry columns are built from live here,
//! because they are pure integer arithmetic that can be pinned down by tests
//! without any protobuf machinery:
//!
//! * **Zigzag** — MVT stores geometry deltas as signed values inside unsigned
//!   varints, mapping `0, -1, 1, -2, 2, ...` to `0, 1, 2, 3, 4, ...` so that
//!   small negative numbers stay short.
//! * **Command integers** — a geometry is a flat `Vec<u32>` in which a command
//!   integer `(id & 0x7) | (count << 3)` is followed by `count * parameters`
//!   zigzag-encoded values.
//!
//! The protobuf envelope that carries those columns lives in the crate-private
//! `proto` module, and [`super::decode`] turns the command streams into typed
//! geometries.

use crate::error::RenderError;

/// Largest repeat count a command integer can carry (`2^29 - 1`).
pub const MAX_COMMAND_COUNT: u32 = (1 << 29) - 1;

/// Maps a signed integer onto an unsigned one so that small magnitudes — of
/// either sign — stay small.
///
/// `0 -> 0`, `-1 -> 1`, `1 -> 2`, `-2 -> 3`, ...
#[must_use]
pub const fn zigzag_encode(value: i32) -> u32 {
    ((value as u32) << 1) ^ ((value >> 31) as u32)
}

/// Inverse of [`zigzag_encode`]. Total: every `u32` decodes to some `i32`.
#[must_use]
pub const fn zigzag_decode(value: u32) -> i32 {
    ((value >> 1) as i32) ^ -((value & 1) as i32)
}

/// The three geometry commands defined by the MVT specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandId {
    /// Start a new point/ring at the accumulated cursor position.
    MoveTo = 1,
    /// Extend the current path to the accumulated cursor position.
    LineTo = 2,
    /// Close the current ring. Takes no parameters.
    ClosePath = 7,
}

impl CommandId {
    /// The command's wire identifier.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// How many zigzag parameters each repetition of this command consumes.
    #[must_use]
    pub const fn parameters_per_repeat(self) -> u32 {
        match self {
            Self::MoveTo | Self::LineTo => 2,
            Self::ClosePath => 0,
        }
    }

    /// Parses a wire identifier.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Mvt`] for the identifiers the specification
    /// leaves undefined (`0`, `3..=6`).
    pub fn from_u32(id: u32) -> Result<Self, RenderError> {
        match id {
            1 => Ok(Self::MoveTo),
            2 => Ok(Self::LineTo),
            7 => Ok(Self::ClosePath),
            other => Err(RenderError::Mvt(format!(
                "undefined mvt geometry command id {other}"
            ))),
        }
    }
}

/// A command integer: an operation plus the number of times it repeats.
///
/// The fields are private so that the 29-bit range of the count is enforced by
/// [`Command::new`]; that in turn lets [`encode_command_integer`] be infallible
/// and exact rather than silently truncating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Command {
    id: CommandId,
    count: u32,
}

impl Command {
    /// Creates a command.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Mvt`] if `count` exceeds [`MAX_COMMAND_COUNT`],
    /// which is the widest value the 29-bit count field can hold.
    pub fn new(id: CommandId, count: u32) -> Result<Self, RenderError> {
        if count > MAX_COMMAND_COUNT {
            return Err(RenderError::Mvt(format!(
                "command count {count} exceeds the 29-bit maximum {MAX_COMMAND_COUNT}"
            )));
        }
        Ok(Self { id, count })
    }

    /// Which operation this command performs.
    #[must_use]
    pub const fn id(&self) -> CommandId {
        self.id
    }

    /// How many times the operation repeats before the next command integer.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// Total number of zigzag parameters that follow this command integer.
    #[must_use]
    pub const fn parameter_count(&self) -> u32 {
        // `count <= 2^29 - 1` and the multiplier is at most 2, so this cannot
        // overflow a `u32`.
        self.count * self.id.parameters_per_repeat()
    }
}

/// Packs a command into its wire representation `(id & 0x7) | (count << 3)`.
#[must_use]
pub fn encode_command_integer(command: Command) -> u32 {
    (command.id.as_u32() & 0x7) | (command.count << 3)
}

/// Unpacks a command integer.
///
/// # Errors
///
/// Returns [`RenderError::Mvt`] if the low three bits are not one of the
/// commands defined by the specification.
pub fn decode_command_integer(raw: u32) -> Result<Command, RenderError> {
    let id = CommandId::from_u32(raw & 0x7)?;
    Ok(Command {
        id,
        count: raw >> 3,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Command, CommandId, MAX_COMMAND_COUNT, decode_command_integer, encode_command_integer,
        zigzag_decode, zigzag_encode,
    };
    use crate::error::RenderError;

    #[test]
    fn zigzag_matches_the_specification_table() {
        // The mapping published in the MVT 2.1 specification.
        let table = [
            (0i32, 0u32),
            (-1, 1),
            (1, 2),
            (-2, 3),
            (2, 4),
            (-3, 5),
            (3, 6),
            (i32::MAX, 4_294_967_294),
            (i32::MIN, 4_294_967_295),
        ];
        for (signed, encoded) in table {
            assert_eq!(zigzag_encode(signed), encoded, "encode {signed}");
            assert_eq!(zigzag_decode(encoded), signed, "decode {encoded}");
        }
    }

    #[test]
    fn zigzag_round_trips() {
        let mut value = i32::MIN;
        loop {
            assert_eq!(zigzag_decode(zigzag_encode(value)), value);
            let Some(next) = value.checked_add(7_919_311) else {
                break;
            };
            value = next;
        }
        for value in [-1000i32, -1, 0, 1, 1000, i32::MAX, i32::MIN] {
            assert_eq!(zigzag_decode(zigzag_encode(value)), value);
        }
        // Decoding is total: every u32 maps back to some i32 and round trips.
        for raw in [0u32, 1, 2, 3, u32::MAX, u32::MAX - 1, 1 << 31] {
            assert_eq!(zigzag_encode(zigzag_decode(raw)), raw);
        }
    }

    #[test]
    fn command_integers_match_the_specification_examples() {
        // `9` = MoveTo repeated once, `26` = LineTo repeated three times,
        // `15` = ClosePath repeated once (spec section 4.3.3 examples).
        let cases = [
            (9u32, CommandId::MoveTo, 1u32),
            (26, CommandId::LineTo, 3),
            (15, CommandId::ClosePath, 1),
            (17, CommandId::MoveTo, 2),
        ];
        for (raw, id, count) in cases {
            let Ok(command) = decode_command_integer(raw) else {
                panic!("decoding {raw} failed");
            };
            assert_eq!(command.id(), id);
            assert_eq!(command.count(), count);
            assert_eq!(encode_command_integer(command), raw);
        }
    }

    #[test]
    fn command_round_trips_over_the_whole_count_range() {
        for id in [CommandId::MoveTo, CommandId::LineTo, CommandId::ClosePath] {
            for count in [0u32, 1, 2, 255, 65_535, MAX_COMMAND_COUNT] {
                let Ok(command) = Command::new(id, count) else {
                    panic!("command {id:?} x{count} rejected");
                };
                let raw = encode_command_integer(command);
                assert_eq!(raw & 0x7, id.as_u32());
                assert_eq!(raw >> 3, count);
                let Ok(decoded) = decode_command_integer(raw) else {
                    panic!("decoding {raw} failed");
                };
                assert_eq!(decoded, command);
            }
        }
    }

    #[test]
    fn parameter_counts_follow_the_command() {
        let Ok(move_to) = Command::new(CommandId::MoveTo, 3) else {
            panic!("command rejected");
        };
        assert_eq!(move_to.parameter_count(), 6);

        let Ok(line_to) = Command::new(CommandId::LineTo, 10) else {
            panic!("command rejected");
        };
        assert_eq!(line_to.parameter_count(), 20);

        let Ok(close) = Command::new(CommandId::ClosePath, 1) else {
            panic!("command rejected");
        };
        assert_eq!(close.parameter_count(), 0);
        assert_eq!(CommandId::ClosePath.parameters_per_repeat(), 0);
    }

    #[test]
    fn undefined_command_ids_are_rejected() {
        for id in [0u32, 3, 4, 5, 6] {
            assert!(
                matches!(CommandId::from_u32(id), Err(RenderError::Mvt(_))),
                "command id {id} should be undefined"
            );
            // Same identifier reached through a full command integer.
            let raw = id | (4 << 3);
            assert!(
                matches!(decode_command_integer(raw), Err(RenderError::Mvt(_))),
                "command integer {raw} should be rejected"
            );
        }
    }

    #[test]
    fn oversized_counts_are_rejected() {
        assert!(matches!(
            Command::new(CommandId::LineTo, MAX_COMMAND_COUNT + 1),
            Err(RenderError::Mvt(_))
        ));
        assert!(matches!(
            Command::new(CommandId::LineTo, u32::MAX),
            Err(RenderError::Mvt(_))
        ));
    }
}
