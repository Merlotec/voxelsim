use std::{collections::VecDeque, error::Error, fmt::Display};

use dashmap::DashSet;
use nalgebra::{Unit, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use crate::{
    Coord,
    trajectory::Trajectory,
    viewport::{CameraOrientation, CameraView},
};

pub mod chase;
pub mod trajectory;
pub mod uncertainty;
pub mod viewport;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Agent {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub thrust: Vector3<f64>,
    /// Angular velocity of rotation around own CoM.
    pub rate: Vector3<f64>,

    pub attitude: UnitQuaternion<f64>,

    pub id: usize,

    // Current action being processed by the agent.
    pub state: AgentState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentState {
    Action(Action),
    Hold { coord: Coord, yaw: f64 },
}

#[derive(
    Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, num_enum::TryFromPrimitive,
)]
#[repr(i32)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub enum MoveDir {
    #[default]
    None = 0,
    Forward = 1,
    Back = 2,
    Left = 3,
    Right = 4,
    Up = 5,
    Down = 6,
    Undecided = 7,
}

impl MoveDir {
    pub fn dir_vector(&self) -> Option<Coord> {
        match self {
            MoveDir::None => Some(Vector3::zeros()),
            MoveDir::Up => Some(Vector3::new(0, 0, -1)),
            MoveDir::Down => Some(Vector3::new(0, 0, 1)),
            MoveDir::Left => Some(Vector3::new(0, -1, 0)),
            MoveDir::Right => Some(Vector3::new(0, 1, 0)),
            MoveDir::Forward => Some(Vector3::new(1, 0, 0)),
            MoveDir::Back => Some(Vector3::new(-1, 0, 0)),
            MoveDir::Undecided => None,
        }
    }
}

pub type MoveSequence = Vec<MoveDir>;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ActionIntent {
    pub move_sequence: MoveSequence,
    pub urgency: f64,
    pub yaw: f64,
}

/// When we carry out an action we need to pyo3ensure that we remove subsequent steps when we enter the
/// relevant voxels.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Action {
    /// The queue of action intents to complete.
    pub intent_queue: VecDeque<ActionIntent>,
    /// The current trajectory that the chaser is following.
    pub trajectory: Trajectory,
    /// The current origin of the trajectory.
    pub origin: Vector3<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStateUpdate {
    Push(ActionIntent),
    Replace(VecDeque<ActionIntent>),
    Truncate(VecDeque<ActionIntent>),
    Hold {
        coord: Option<Coord>,
        yaw: Option<f64>,
    },
}

impl ActionIntent {
    pub fn new(urgency: f64, yaw: f64, move_sequence: MoveSequence) -> Self {
        assert!(urgency > 0.0);
        Self {
            urgency,
            yaw,
            move_sequence,
        }
    }

    // Gets the relative position of the centroid of the grid at position t (unscaled) from the
    // centroid of the starting cell.
    pub fn relative_centroid_pos(&self, t: usize) -> Option<Coord> {
        if self.move_sequence.len() >= t {
            Some(
                self.move_sequence
                    .iter()
                    .take(t)
                    .filter_map(|x| x.dir_vector())
                    .sum(),
            )
        } else {
            None
        }
    }

    pub fn relative_end_coord(&self) -> Coord {
        self.relative_centroid_pos(self.move_sequence.len())
            .unwrap()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ActionDesc {
    pub target: Coord,
    pub urgency: f64,
    pub yaw: f64,
}

impl ActionDesc {
    pub fn from_intent(intent: &ActionIntent) -> Self {
        Self {
            target: intent.relative_end_coord(),
            urgency: intent.urgency,
            yaw: intent.yaw,
        }
    }
}

impl Action {
    pub fn new_oneshot(intent: ActionIntent, origin: Coord) -> Result<Self, ActionError> {
        if intent.move_sequence.is_empty() {
            return Err(ActionError::NoCommands);
        }
        let intent_queue = VecDeque::from([intent]);
        let cells = Action::chained_centroids(&intent_queue, origin)?;
        let action = Action {
            intent_queue,
            origin,
            trajectory: Trajectory::new(origin, &cells),
        };
        Ok(action)
    }

    pub fn new(intent_queue: VecDeque<ActionIntent>, origin: Coord) -> Result<Self, ActionError> {
        for intent in intent_queue.iter() {
            if intent.move_sequence.is_empty() {
                return Err(ActionError::NoCommands);
            }
        }
        let cells = Action::chained_centroids(&intent_queue, origin)?;
        let action = Action {
            intent_queue,
            origin,
            trajectory: Trajectory::new(origin, &cells),
        };
        Ok(action)
    }

    pub fn appending_intent(mut self, intent: ActionIntent) -> Result<Self, ActionError> {
        self.push_back_intent(intent)?;
        Ok(self)
    }

    pub fn centroids<'a, I: IntoIterator<Item = &'a MoveDir>>(
        move_sequence: I,
        origin: Vector3<i32>,
    ) -> Result<Vec<Coord>, ActionError> {
        let mut total: Coord = origin;
        let mut centroids = Vec::new();
        let set: DashSet<Coord> = DashSet::new();
        // Make sure we dont duplicate origin either.
        set.insert(origin);
        for seq in move_sequence {
            if let Some(dir_vector) = seq.dir_vector() {
                let centroid = total + dir_vector;
                centroids.push(centroid);
                total = centroid;
                if !set.insert(centroid) {
                    return Err(ActionError::DuplicateCentroid);
                }
            }
        }

        Ok(centroids)
    }

    pub fn end_coord(&self) -> Coord {
        self.origin
            + self
                .intent_queue
                .iter()
                .map(|x| x.relative_end_coord())
                .sum::<Vector3<i32>>()
    }

    pub fn chained_centroids<'a, I: IntoIterator<Item = &'a ActionIntent>>(
        intents: I,
        origin: Coord,
    ) -> Result<Vec<(Coord, f64, f64)>, ActionError> {
        let mut origin = origin;
        let mut centroids = Vec::new();
        for intent in intents {
            let next_centroids = Self::centroids(&intent.move_sequence, origin)?;
            origin = (next_centroids.last().copied()).unwrap_or(origin);
            let mut action_centroids = next_centroids
                .into_iter()
                .map(|x| (x, intent.urgency, intent.yaw))
                .collect();

            centroids.append(&mut action_centroids);
        }

        Ok(centroids)
    }

    /// Gets the intent that is being executed if the progress is the given value.
    pub fn intent_for_progress(&self, progress: f64) -> Option<usize> {
        let mut move_buf = 0;
        for (i, intent) in self.intent_queue.iter().enumerate() {
            assert!(!intent.move_sequence.is_empty());
            move_buf += intent.move_sequence.len() - 1;
            let p = self.trajectory.progress_for_move_idx(move_buf)?;
            if progress < p {
                return Some(i);
            }
        }
        None
    }

    pub fn update_progress(&mut self, progress: f64, trim_tail: bool) -> Result<(), ActionError> {
        self.trajectory.progress = progress.clamp(0.0, self.trajectory.length());

        if trim_tail {
            // Get the intent that we are in.
            if let Some(i) = self.intent_for_progress(progress) {
                let to_remove = i.saturating_sub(1);
                for _ in 0..to_remove {
                    self.pop_front_intent()?;
                }
            }
        }

        Ok(())
    }

    pub fn pop_front_intent(&mut self) -> Result<ActionIntent, ActionError> {
        // We must update the trajectory...
        // We can do this by determining the change in length, and then subtracting that from the current progress.
        let progress = self.trajectory.progress;
        let old_m = self
            .trajectory
            .move_for_progress(progress)
            .ok_or(ActionError::InvalidProgress)?;
        let front = self.intent_queue.pop_front().ok_or(ActionError::NoIntent)?;
        // Change origin
        self.origin = self.origin + front.relative_end_coord();
        // We have removed something so we need to update the trajectory.
        self.trajectory = Trajectory::new(
            self.origin,
            &Self::chained_centroids(self.intent_queue.iter(), self.origin)?,
        );

        let new_m = (old_m - front.move_sequence.len() as f64).max(0.0);

        self.trajectory.progress = self
            .trajectory
            .progress_for_move(new_m)
            .ok_or(ActionError::InvalidProgress)?;

        Ok(front)
    }

    /// Removes all intents in front of the currently active intent.
    pub fn truncate_replace(
        &mut self,
        mut replace: VecDeque<ActionIntent>,
    ) -> Result<(), ActionError> {
        let progress = self.trajectory.progress;

        let curr_intent_idx = self
            .intent_for_progress(progress)
            .ok_or(ActionError::InvalidProgress)?;

        let t = self.intent_queue.len() - (curr_intent_idx + 1);
        self.intent_queue.truncate(t);

        self.intent_queue.append(&mut replace);

        // We have added something so we need to update the trajectory.
        self.trajectory = Trajectory::new(
            self.origin,
            &Self::chained_centroids(self.intent_queue.iter(), self.origin)?,
        );

        self.trajectory.progress = progress;

        Ok(())
    }

    pub fn push_back_intent(&mut self, intent: ActionIntent) -> Result<(), ActionError> {
        // We must update the trajectory...
        // We can do this by determining the change in length, and then subtracting that from the current progress.
        self.intent_queue.push_back(intent);
        let progress = self.trajectory.progress;
        // We have added something so we need to update the trajectory.
        self.trajectory = Trajectory::new(
            self.origin,
            &Self::chained_centroids(self.intent_queue.iter(), self.origin)?,
        );

        self.trajectory.progress = progress;

        Ok(())
    }
}

impl Agent {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            pos: Vector3::zeros(),
            vel: Vector3::zeros(),
            thrust: Vector3::zeros(),
            rate: Vector3::zeros(),
            state: AgentState::Hold {
                coord: Vector3::zeros(),
                yaw: 0.0,
            },
            attitude: UnitQuaternion::identity(),
        }
    }

    pub fn camera_view(&self, orientation: &CameraOrientation) -> CameraView {
        CameraView::from_pos_quat(self.pos, self.attitude * orientation.quat)
    }

    pub fn get_action(&self) -> Option<&Action> {
        if let AgentState::Action(action) = &self.state {
            Some(action)
        } else {
            None
        }
    }

    pub fn get_position(&self) -> Vector3<f64> {
        self.pos
    }

    pub fn get_coord(&self) -> Coord {
        self.pos.map(|e| e.round() as i32)
    }

    pub fn get_yaw(&self) -> f64 {
        let forward = MoveDir::Forward.dir_vector().unwrap().cast();
        let drone_forward = self.attitude * MoveDir::Forward.dir_vector().unwrap().cast();

        let up = MoveDir::Up.dir_vector().unwrap().cast();
        let proj = forward - (drone_forward.dot(&up)) * up;

        let yaw = forward.angle(&proj);
        yaw
    }

    pub fn set_hold(&mut self, coord: Coord, yaw: f64) {
        self.pos = coord.cast();
        self.state = AgentState::Hold { coord, yaw };
    }

    pub fn set_velocity(&mut self, x: f64, y: f64, z: f64) {
        self.vel = Vector3::new(x, y, z);
    }

    pub fn update_state(&mut self, update: AgentStateUpdate) -> Result<(), ActionError> {
        match update {
            AgentStateUpdate::Push(intent) => {
                if let AgentState::Action(a) = &mut self.state {
                    a.push_back_intent(intent)?;
                } else {
                    self.state = AgentState::Action(Action::new_oneshot(intent, self.get_coord())?);
                }
            }
            AgentStateUpdate::Replace(intent_queue) => {
                self.state = AgentState::Action(Action::new(intent_queue, self.get_coord())?);
            }
            AgentStateUpdate::Truncate(intent_queue) => {
                if let AgentState::Action(a) = &mut self.state {
                    a.truncate_replace(intent_queue.into())?;
                } else {
                    self.state = AgentState::Action(Action::new(intent_queue, self.get_coord())?);
                }
            }
            AgentStateUpdate::Hold { coord, yaw } => {
                let coord = coord.unwrap_or(self.get_coord());
                let yaw = yaw.unwrap_or(self.get_yaw());
                self.state = AgentState::Hold { coord, yaw }
            }
        }
        Ok(())
    }

    pub fn perform(&mut self, intent: ActionIntent) -> Result<(), ActionError> {
        let action = Action::new_oneshot(intent, self.get_coord())?;
        self.state = AgentState::Action(action);
        Ok(())
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ActionError {
    DuplicateCentroid,
    NoCommands,
    NoIntent,
    InvalidProgress,
}

impl Error for ActionError {}

impl Display for ActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateCentroid => f.write_str("Duplicate centroid"),
            Self::NoCommands => f.write_str("No commands"),
            Self::NoIntent => f.write_str("No intent"),
            Self::InvalidProgress => f.write_str("Invalid progress"),
        }
    }
}
