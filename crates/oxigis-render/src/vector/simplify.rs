//! Geometric level of detail: the point-dropping pass that runs before `lyon`.
//!
//! An MVT geometry is a polyline on an integer grid, so `lyon`'s own tolerance —
//! which governs how finely *curves* are flattened — never drops a single point
//! of it. At low zoom one tile unit covers a fraction of a pixel and most of a
//! ring's points land on the same pixel, which is what this module removes.
//!
//! Two passes, in order:
//!
//! 1. **Radial distance.** Drop every point closer than `tolerance` to the last
//!    kept one. `O(n)`, and it is what collapses the runs of near-duplicate
//!    points that dominate a low-zoom tile.
//! 2. **Douglas-Peucker.** Drop points whose perpendicular distance to the
//!    chord of the run they sit on stays below `tolerance`, which is what
//!    collapses a densely sampled *straight* run — the case radial distance
//!    cannot see, because consecutive points there are far apart.
//!
//! Douglas-Peucker is `O(n²)` on adversarial input (a spiral splits into two
//! spans of `n-1` points at every step), so the recursion is iterative and
//! spends at most [`WORK_PER_POINT`] distance evaluations per input point.
//! Exhausting that budget *keeps* the remaining points rather than dropping
//! them: the output is then simply less simplified, never coarser than asked.
//!
//! Fills and outlines call the same functions with the same tolerance, so a
//! polygon and its own outline stay geometrically identical — a fill simplified
//! differently from its outline leaks a hairline of colour along every corner
//! the two disagree on.

use lyon::math::{Point, point};

/// Distance evaluations Douglas-Peucker may spend per input point.
///
/// Balanced recursion costs about `n log n` evaluations (17 per point at
/// `n = 100_000`), so this is generous for real geometry and still linear for
/// hostile geometry.
const WORK_PER_POINT: usize = 32;

/// Reusable buffers for the simplification passes.
///
/// One instance per tessellation run: every ring of every feature is simplified
/// through the same three allocations instead of allocating per ring.
#[derive(Debug, Default)]
pub(crate) struct Simplifier {
    points: Vec<Point>,
    keep: Vec<bool>,
    stack: Vec<(usize, usize)>,
}

impl Simplifier {
    /// Creates an empty simplifier.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Simplifies an open polyline, always keeping its first and last points.
    ///
    /// The returned slice borrows the internal buffer and is valid until the
    /// next call.
    pub(crate) fn open(&mut self, points: &[[i32; 2]], tolerance: f32) -> &[Point] {
        self.radial(points, tolerance);
        if points.len() >= 2
            && let Some(last) = points.last()
        {
            let last = point(last[0] as f32, last[1] as f32);
            match self.points.last() {
                Some(previous) if *previous == last => {}
                _ => self.points.push(last),
            }
        }
        self.douglas_peucker(tolerance);
        &self.points
    }

    /// Simplifies a ring, whose closing edge is implicit: the last point is
    /// dropped when it collapses onto the first.
    ///
    /// The returned slice borrows the internal buffer and is valid until the
    /// next call.
    pub(crate) fn closed(&mut self, points: &[[i32; 2]], tolerance: f32) -> &[Point] {
        self.radial(points, tolerance);
        if self.points.len() >= 2 {
            let (Some(first), Some(last)) =
                (self.points.first().copied(), self.points.last().copied())
            else {
                return &self.points;
            };
            if (last - first).square_length() < tolerance * tolerance {
                self.points.pop();
            }
        }
        self.douglas_peucker(tolerance);
        &self.points
    }

    /// Keeps a point once it is at least `tolerance` away from the last kept
    /// one, starting from the first.
    fn radial(&mut self, points: &[[i32; 2]], tolerance: f32) {
        self.points.clear();
        self.points.reserve(points.len());
        let threshold = tolerance * tolerance;
        for position in points {
            let candidate = point(position[0] as f32, position[1] as f32);
            match self.points.last() {
                Some(previous) if (candidate - *previous).square_length() < threshold => {}
                _ => self.points.push(candidate),
            }
        }
    }

    /// Drops points that stay within `tolerance` of the chord spanning them,
    /// keeping the two endpoints. Work-bounded: see the module documentation.
    fn douglas_peucker(&mut self, tolerance: f32) {
        let len = self.points.len();
        if len < 3 || !tolerance.is_finite() || tolerance <= 0.0 {
            return;
        }
        self.keep.clear();
        self.keep.resize(len, false);
        self.keep[0] = true;
        self.keep[len - 1] = true;
        self.stack.clear();
        self.stack.push((0, len - 1));

        let threshold = tolerance * tolerance;
        let mut budget = len.saturating_mul(WORK_PER_POINT);
        while let Some((first, last)) = self.stack.pop() {
            if last <= first + 1 {
                continue;
            }
            let span = last - first - 1;
            if span > budget {
                // Out of budget: keeping the span is the conservative answer —
                // more vertices than necessary, never less detail than asked.
                for flag in &mut self.keep[first + 1..last] {
                    *flag = true;
                }
                continue;
            }
            budget -= span;

            let (from, to) = (self.points[first], self.points[last]);
            let mut worst = 0.0f32;
            let mut worst_index = first;
            for (offset, candidate) in self.points[first + 1..last].iter().enumerate() {
                let distance = squared_distance_to_segment(*candidate, from, to);
                if distance > worst {
                    worst = distance;
                    worst_index = first + 1 + offset;
                }
            }
            if worst > threshold && worst_index > first {
                self.keep[worst_index] = true;
                self.stack.push((first, worst_index));
                self.stack.push((worst_index, last));
            }
        }

        let mut write = 0usize;
        for read in 0..len {
            if self.keep[read] {
                self.points[write] = self.points[read];
                write += 1;
            }
        }
        self.points.truncate(write);
    }
}

/// Squared perpendicular distance from `position` to the infinite line through
/// `from` and `to`, or to the point itself when the two coincide.
///
/// Non-finite input (a coordinate large enough to overflow the cross product)
/// yields [`f32::INFINITY`], which keeps the point — the safe direction.
fn squared_distance_to_segment(position: Point, from: Point, to: Point) -> f32 {
    let along = to - from;
    let offset = position - from;
    let length_squared = along.square_length();
    if length_squared <= 0.0 || !length_squared.is_finite() {
        return offset.square_length();
    }
    let cross = along.x * offset.y - along.y * offset.x;
    cross * cross / length_squared
}

#[cfg(test)]
mod tests {
    use super::{Simplifier, squared_distance_to_segment};
    use lyon::math::point;

    #[test]
    fn a_dense_run_collapses_to_the_points_that_matter() {
        let mut line = Vec::new();
        for index in 0..100i32 {
            line.push([index, 0]);
        }
        let mut simplifier = Simplifier::new();
        // Radial distance alone would keep every second point at tolerance 2;
        // the run is straight, so Douglas-Peucker keeps only the ends.
        let simplified = simplifier.open(&line, 2.0).to_vec();
        assert_eq!(simplified, vec![point(0.0, 0.0), point(99.0, 0.0)]);
    }

    #[test]
    fn detail_above_the_tolerance_survives() {
        let line = vec![[0, 0], [50, 40], [100, 0]];
        let mut simplifier = Simplifier::new();
        assert_eq!(simplifier.open(&line, 2.0).len(), 3);
        // The spike is 40 units off the chord: a 64-unit budget swallows it.
        assert_eq!(simplifier.open(&line, 64.0).len(), 2);
    }

    #[test]
    fn endpoints_are_never_dropped_from_an_open_line() {
        let line = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
        let mut simplifier = Simplifier::new();
        let simplified = simplifier.open(&line, 100.0).to_vec();
        assert_eq!(simplified, vec![point(0.0, 0.0), point(3.0, 0.0)]);
    }

    #[test]
    fn a_ring_drops_its_repeated_closing_point() {
        // A square stored with the first point repeated at the end.
        let ring = vec![[0, 0], [0, 100], [100, 100], [100, 0], [0, 0]];
        let mut simplifier = Simplifier::new();
        let simplified = simplifier.closed(&ring, 1.0).to_vec();
        assert_eq!(simplified.len(), 4, "{simplified:?}");
        assert_eq!(simplified[0], point(0.0, 0.0));
        assert_eq!(simplified[3], point(100.0, 0.0));
    }

    #[test]
    fn a_ring_keeps_its_corners() {
        let mut ring = Vec::new();
        // Densely sampled square: 40 points per edge, all collinear.
        for index in 0..40i32 {
            ring.push([index * 10, 0]);
        }
        for index in 0..40i32 {
            ring.push([400, index * 10]);
        }
        for index in 0..40i32 {
            ring.push([400 - index * 10, 400]);
        }
        for index in 0..40i32 {
            ring.push([0, 400 - index * 10]);
        }
        let mut simplifier = Simplifier::new();
        let simplified = simplifier.closed(&ring, 1.0).to_vec();
        assert!(
            simplified.len() <= 5,
            "a square must not need more than its corners: {}",
            simplified.len()
        );
        for corner in [
            point(0.0, 0.0),
            point(400.0, 0.0),
            point(400.0, 400.0),
            point(0.0, 400.0),
        ] {
            assert!(simplified.contains(&corner), "{corner:?} was dropped");
        }
    }

    #[test]
    fn degenerate_input_is_answered_not_crashed() {
        let mut simplifier = Simplifier::new();
        assert!(simplifier.open(&[], 1.0).is_empty());
        assert_eq!(simplifier.open(&[[5, 5]], 1.0).len(), 1);
        assert_eq!(simplifier.closed(&[[5, 5], [5, 5]], 1.0).len(), 1);
        // A non-positive tolerance keeps everything the radial pass kept.
        assert_eq!(simplifier.open(&[[0, 0], [1, 1], [2, 2]], 0.0).len(), 3);
        assert_eq!(
            simplifier.open(&[[0, 0], [1, 1], [2, 2]], f32::NAN).len(),
            3
        );
    }

    #[test]
    fn the_work_budget_bounds_the_recursion() {
        // A staircase whose every point is a corner: the worst case for the
        // split heuristic, and the case the budget exists for.
        let mut line = Vec::new();
        for index in 0..2000i32 {
            line.push([index * 4, (index % 2) * 4]);
        }
        let mut simplifier = Simplifier::new();
        let simplified = simplifier.open(&line, 1.0).to_vec();
        // Every corner is above the tolerance, so nothing may be dropped — with
        // or without the budget running out part of the way through.
        assert_eq!(simplified.len(), line.len());

        // A budget that runs out keeps points; it never invents them.
        let coarse = simplifier.open(&line, 3.0).to_vec();
        assert!(coarse.len() >= 2, "{}", coarse.len());
        assert!(coarse.len() <= line.len());
    }

    #[test]
    fn distance_to_a_degenerate_chord_is_the_point_distance() {
        let same = point(3.0, 4.0);
        assert!((squared_distance_to_segment(point(0.0, 0.0), same, same) - 25.0).abs() < 1e-4);
        let distance =
            squared_distance_to_segment(point(0.0, 5.0), point(-10.0, 0.0), point(10.0, 0.0));
        assert!((distance - 25.0).abs() < 1e-4, "{distance}");
    }
}
