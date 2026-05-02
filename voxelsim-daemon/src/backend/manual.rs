use std::time::Duration;

use voxelsim::{Action, AgentStateUpdate};

use crate::backend::{ControlBackend, ControlStep};

/// Abstracts over how the signal is received.
/// May want to be able to receive signal from both web sockets and mavlink radio signals.
pub trait ActionReceiver {
    fn try_recv_signal(&self) -> Option<AgentStateUpdate>;
}

pub struct ManualBackend<R: ActionReceiver> {
    receiver: R,
}

impl<R: ActionReceiver> ManualBackend<R> {
    pub fn new(receiver: R) -> Self {
        Self { receiver }
    }
}

impl<R: ActionReceiver> ControlBackend for ManualBackend<R> {
    fn update_action(
        &mut self,
        _agent: &voxelsim::Agent,
        _fw: &voxelsim::VoxelGrid,
    ) -> ControlStep {
        ControlStep {
            update: self.receiver.try_recv_signal(),
            min_sleep: Duration::from_millis(10),
        }
    }
}
