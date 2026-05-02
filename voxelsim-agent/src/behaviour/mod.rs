use voxelsim::{Agent, AgentState, VoxelGrid};

pub struct BehaviourState {
    current: Option<Box<dyn AgentBehaviour>>,
}

pub trait AgentBehaviour {
    fn update_agent_state(&self, agent: &Agent, world: &VoxelGrid, world_time: f64) -> AgentState;
}
