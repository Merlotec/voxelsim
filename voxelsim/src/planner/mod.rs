pub mod astar;

use crate::{ActionIntent, Coord, VoxelGrid};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PlannerError {
    PathBlocked,
    NoViablePath,
    DstFilled,
    InvalidParams,
}

impl std::fmt::Display for PlannerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PathBlocked => f.write_str("Path to the target is blocked."),
            Self::NoViablePath => f.write_str("No viable path to target."),
            Self::DstFilled => f.write_str("Destination voxel is filled."),
            Self::InvalidParams => f.write_str("The parameters specified were invalid."),
        }
    }
}

// Core planning trait implemented by different planners (e.g., A*)
pub trait ActionPlanner {
    /// Plan a route from `origin` to `dst` using 6-neighbour moves.
    fn plan_action(
        &self,
        world: &VoxelGrid,
        origin: Coord,
        dst: Coord,
        urgency: f64,
        yaw: f64,
    ) -> Result<ActionIntent, PlannerError>;
}
