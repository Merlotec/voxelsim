use std::time::Duration;

use voxelsim::{Agent, AgentStateUpdate, VoxelGrid};

pub mod manual;
pub mod python;
pub use python::PythonBackend;

pub struct ControlStep {
    pub update: Option<AgentStateUpdate>,
    /// The minimum time that will elapse before the next invocation of `update_action` occurs.
    pub min_sleep: Duration,
}

pub trait ControlBackend {
    fn update_action(&mut self, agent: &Agent, fw: &VoxelGrid) -> ControlStep;
}
