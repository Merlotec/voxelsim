use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::agent::MoveDir;
use crate::{ActionIntent, PlannerError};
use crate::{Coord, VoxelGrid, env::Cell};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct Node(Coord);

#[derive(Copy, Clone, Eq, PartialEq)]
struct QueueEntry {
    // Primary key: estimated total steps (g_steps + h_steps)
    f_steps: i32,
    // Secondary key: tie-breaker = sum of distances to line (scaled)
    f_dist: i64,
    // For stale-entry skipping
    g_steps: i32,
    g_dist: i64,
    node: Node,
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap behavior with BinaryHeap
        other
            .f_steps
            .cmp(&self.f_steps)
            .then_with(|| other.f_dist.cmp(&self.f_dist))
            .then_with(|| self.node.0.x.cmp(&other.node.0.x))
            .then_with(|| self.node.0.y.cmp(&other.node.0.y))
            .then_with(|| self.node.0.z.cmp(&other.node.0.z))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn manhattan(a: Coord, b: Coord) -> i32 {
    (a.x - b.x).abs() + (a.y - b.y).abs() + (a.z - b.z).abs()
}

fn is_obstacle_cell(cell: Cell) -> bool {
    cell.intersects(
        Cell::FILLED | Cell::GROUND | Cell::SPARSE | Cell::DYNAMIC_OCCUPIED | Cell::AGENT_OCCUPIED,
    )
}

fn is_blocked(world: &VoxelGrid, c: &Coord, padding: i32) -> bool {
    // If padding is 0, only check the cell itself
    if padding <= 0 {
        return world
            .cells()
            .get(c)
            .map(|v| is_obstacle_cell(*v))
            .unwrap_or(false);
    }

    // Check Chebyshev neighbourhood within padding for any obstacle
    for dx in -padding..=padding {
        for dy in -padding..=padding {
            for dz in -padding..=padding {
                let nb = *c + Coord::new(dx, dy, dz);
                if let Some(cell) = world.cells().get(&nb) {
                    if is_obstacle_cell(*cell) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn neighbours6(c: Coord) -> [Coord; 6] {
    [
        c + Coord::new(1, 0, 0),
        c + Coord::new(-1, 0, 0),
        c + Coord::new(0, 1, 0),
        c + Coord::new(0, -1, 0),
        c + Coord::new(0, 0, 1),
        c + Coord::new(0, 0, -1),
    ]
}

fn delta_to_dir(d: Coord) -> Option<MoveDir> {
    match (d.x, d.y, d.z) {
        (1, 0, 0) => Some(MoveDir::Forward),
        (-1, 0, 0) => Some(MoveDir::Back),
        (0, 1, 0) => Some(MoveDir::Right),
        (0, -1, 0) => Some(MoveDir::Left),
        (0, 0, 1) => Some(MoveDir::Down),
        (0, 0, -1) => Some(MoveDir::Up),
        _ => None,
    }
}

// Distance from point p to the line segment a-b, scaled by SCALE and rounded.
fn dist_point_to_segment_scaled(p: Coord, a: Coord, b: Coord, scale: f64) -> i64 {
    let ax = a.x as f64;
    let ay = a.y as f64;
    let az = a.z as f64;
    let bx = b.x as f64;
    let by = b.y as f64;
    let bz = b.z as f64;
    let px = p.x as f64;
    let py = p.y as f64;
    let pz = p.z as f64;

    let vx = bx - ax;
    let vy = by - ay;
    let vz = bz - az;
    let wx = px - ax;
    let wy = py - ay;
    let wz = pz - az;

    let vv = vx * vx + vy * vy + vz * vz;
    let t = if vv <= 1e-9 {
        0.0
    } else {
        ((wx * vx + wy * vy + wz * vz) / vv).clamp(0.0, 1.0)
    };
    let cx = ax + t * vx;
    let cy = ay + t * vy;
    let cz = az + t * vz;
    let dx = px - cx;
    let dy = py - cy;
    let dz = pz - cz;
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    (dist * scale).round() as i64
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct AStarActionPlanner {
    /// Padding radius (in blocks) around obstacles to treat as blocked.
    /// Padding applies in all directions (including corners/edges), i.e.
    /// any cell within Chebyshev distance <= padding from an obstacle is blocked.
    padding: i32,
}

impl AStarActionPlanner {
    pub fn new(padding: i32) -> Self {
        Self {
            padding: padding.max(0),
        }
    }

    pub fn find_nearest_target(start: Coord, world: &VoxelGrid) -> Option<Coord> {
        let mut best: Option<(i32, Coord)> = None;
        for kv in world.cells().iter() {
            if kv.intersects(Cell::TARGET) {
                let c = *kv.key();
                let dist = manhattan(start, c);
                if best.map(|(d, _)| dist < d).unwrap_or(true) {
                    best = Some((dist, c));
                }
            }
        }
        best.map(|(_, c)| c)
    }

    fn reconstruct_path(
        came_from: &HashMap<Coord, Coord>,
        mut current: Coord,
        start: Coord,
    ) -> Vec<MoveDir> {
        let mut steps: Vec<MoveDir> = Vec::new();
        while current != start {
            if let Some(prev) = came_from.get(&current) {
                let delta = current - *prev;
                if let Some(dir) = delta_to_dir(delta) {
                    steps.push(dir);
                }
                current = *prev;
            } else {
                break;
            }
        }
        steps.reverse();
        steps
    }
}

impl super::ActionPlanner for AStarActionPlanner {
    fn plan_action(
        &self,
        world: &VoxelGrid,
        origin: Coord,
        dst: Coord,
        urgency: f64,
        yaw: f64,
    ) -> Result<ActionIntent, PlannerError> {
        if origin == dst {
            return Err(PlannerError::InvalidParams);
        }

        if is_blocked(world, &origin, self.padding) {
            return Err(PlannerError::PathBlocked);
        }

        // A* with lexicographic tie-breaker:
        // 1) minimize number of steps
        // 2) among equal-length paths, minimize sum of distances of visited centroids to the
        //    straight segment (origin-dst)
        let mut open = BinaryHeap::<QueueEntry>::new();
        let mut came_from: HashMap<Coord, Coord> = HashMap::new();
        let mut g_score: HashMap<Coord, (i32, i64)> = HashMap::new(); // (g_steps, g_dist_scaled)

        // Scaling for distance sum to keep integer arithmetic stable
        const DIST_SCALE: f64 = 1000.0;

        g_score.insert(origin, (0, 0));
        open.push(QueueEntry {
            f_steps: manhattan(origin, dst),
            f_dist: 0,
            g_steps: 0,
            g_dist: 0,
            node: Node(origin),
        });

        // Limit to avoid infinite search in degenerate inputs
        let max_expansions = 200_000;
        let mut expansions = 0;

        while let Some(QueueEntry {
            f_steps: _,
            f_dist: _,
            g_steps,
            g_dist,
            node: Node(curr),
        }) = open.pop()
        {
            // Skip stale entries if we already have a better g_score recorded
            if let Some(&(gs_best, gd_best)) = g_score.get(&curr) {
                if (g_steps, g_dist) != (gs_best, gd_best) {
                    continue;
                }
            }

            if curr == dst {
                // Reconstruct path
                let move_sequence = Self::reconstruct_path(&came_from, dst, origin);
                return Ok(ActionIntent::new(urgency, yaw, move_sequence));
            }
            expansions += 1;
            if expansions > max_expansions {
                break;
            }

            for nb in neighbours6(curr) {
                if is_blocked(world, &nb, self.padding) {
                    continue;
                }
                let step_g_steps = g_steps + 1;
                let step_g_dist =
                    g_dist + dist_point_to_segment_scaled(nb, origin, dst, DIST_SCALE);

                let better = match g_score.get(&nb) {
                    None => true,
                    Some(&(gs_old, gd_old)) => {
                        // Lex compare: first by steps, then by distance
                        (step_g_steps < gs_old) || (step_g_steps == gs_old && step_g_dist < gd_old)
                    }
                };
                if better {
                    came_from.insert(nb, curr);
                    g_score.insert(nb, (step_g_steps, step_g_dist));
                    let f_steps = step_g_steps + manhattan(nb, dst);
                    let f_dist = step_g_dist; // admissible heuristic for dist is 0
                    open.push(QueueEntry {
                        f_steps,
                        f_dist,
                        g_steps: step_g_steps,
                        g_dist: step_g_dist,
                        node: Node(nb),
                    });
                }
            }
        }

        // If search fails, return empty (no path)

        Err(PlannerError::NoViablePath)
    }
}
