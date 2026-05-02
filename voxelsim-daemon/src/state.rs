use voxelsim::{
    ActionError, Agent, VoxelGrid,
    chase::{FixedLookaheadChaser, TrajectoryChaser},
};

use crate::backend::ControlBackend;

pub struct ControlState {
    agent: Agent,
    world: VoxelGrid,
    /// Used to define how action logic is followed.
    chaser: Box<dyn TrajectoryChaser>,
    control_backend: Option<Box<dyn ControlBackend>>,
}

impl ControlState {
    pub fn new(agent_id: usize, chaser: Box<dyn TrajectoryChaser>) -> Self {
        Self {
            // As there is only one agent in the system, always give id 0.
            agent: Agent::new(agent_id),
            world: VoxelGrid::new(),
            chaser,
            control_backend: None,
        }
    }

    pub fn set_backend(&mut self, backend: Option<Box<dyn ControlBackend>>) {
        self.control_backend = backend;
    }

    pub fn update(&mut self) -> Result<(), ActionError> {
        if let Some(cb) = &mut self.control_backend {
            let step = cb.update_action(&self.agent, &self.world);
            if let Some(update) = step.update {
                self.agent.update_state(update)?;
            }
        }
        Ok(())
    }
}

impl Default for ControlState {
    fn default() -> Self {
        Self::new(0, Box::new(FixedLookaheadChaser::default()))
    }
}
