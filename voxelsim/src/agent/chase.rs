use nalgebra::Vector3;

use crate::{Agent, AgentState};

pub trait TrajectoryChaser {
    /// It is the chaser's job to remove/update the move sequence of the agent as it moves along its spline.
    fn step_chase(&self, agent: &Agent, delta: f64) -> ChaseTarget;
}

/// This is the data sent to ArduPilot.
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ChaseTarget {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub acc: Vector3<f64>,

    pub yaw: f64,

    pub progress: ActionProgress,
}

#[derive(Debug, Default, Clone)]
pub enum ActionProgress {
    /// Progress the current action to the specified progress.
    ProgressTo(f64, bool),
    /// End the current action and start the new state. Stores the next state that should be engaged.
    Complete(AgentState),
    /// Do not update progress (this can be used if there is no currently active action).
    #[default]
    Hold,
}

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct FixedLookaheadChaser {
    pub v_max_base: f64,
    pub s_lookahead_base: f64,
    pub min_step: f64,
}

impl Default for FixedLookaheadChaser {
    fn default() -> Self {
        Self {
            v_max_base: 20.0,
            s_lookahead_base: 2.0,
            min_step: 2.5,
        }
    }
}

impl TrajectoryChaser for FixedLookaheadChaser {
    fn step_chase(&self, agent: &Agent, delta: f64) -> ChaseTarget {
        // Get next chase.
        match &agent.state {
            AgentState::Action(action) => {
                // Guard against degenerate/zero-length trajectories
                if action.trajectory.length() <= std::f64::EPSILON {
                    return ChaseTarget {
                        pos: agent.pos,
                        vel: Vector3::zeros(),
                        acc: Vector3::zeros(),
                        yaw: 0.0,
                        progress: ActionProgress::Hold,
                    };
                }
                let x_act = agent.pos;

                let s_cur = action.trajectory.progress;
                let s_end = action.trajectory.length(); // normalized trajectory domain
                let urgency = action.trajectory.sample_urgencies(s_cur).unwrap_or(0.8);

                let v_max_cur = self.v_max_base * urgency;
                let ds_max = v_max_cur * delta;
                let ds_min = self.min_step * delta * urgency;
                let s_lookahead = self.s_lookahead_base * urgency;
                let s_star = action
                    .trajectory
                    .find_nearest_param_in_range(s_cur, s_end, &x_act)
                    .expect(&format!("No nearest param in range: {}, {}", s_cur, s_end));

                // Check if drone is close enough to the trajectory to advance progress
                let pos_at_s_star = action.trajectory.position(s_star);
                let distance_threshold = 1.0; // meters - adjust as needed

                let can_advance = if let Some(traj_pos) = pos_at_s_star {
                    (x_act - traj_pos).norm() < distance_threshold
                } else {
                    false
                };

                // Detect if we've overshot the end of the trajectory along its tangent
                let mut overshot_end = false;
                if (s_end - s_star).abs() < 1e-6 {
                    if let (Some(p_end), Some(v_end)) = (
                        action.trajectory.position(s_end),
                        action.trajectory.velocity(s_end),
                    ) {
                        let ahead_along_tangent =
                            v_end.norm() > 1e-6 && (x_act - p_end).dot(&v_end) > 0.0;
                        let very_close_to_end = (x_act - p_end).norm() < distance_threshold;
                        overshot_end = ahead_along_tangent || very_close_to_end;
                    }
                }

                let s_updated = if can_advance {
                    s_star.clamp(s_cur, s_cur + ds_max)
                } else {
                    // Don't advance if too far from trajectory
                    s_cur
                };

                let s_tgt = (s_updated + s_lookahead).min(s_end);
                if let (
                    Some(p_tgt),
                    Some(v_tgt_nominal),
                    Some(a_tgt_nominal),
                    Some(yaw),
                    Some(urgency),
                ) = (
                    action.trajectory.position(s_tgt),
                    action.trajectory.velocity(s_tgt),
                    action.trajectory.acceleration(s_tgt),
                    action.trajectory.sample_yaw(s_tgt),
                    action.trajectory.sample_urgencies(s_tgt),
                ) {
                    // 6) Scale velocity and accel targets by urgency uniformly
                    let v_tgt = v_tgt_nominal * urgency;
                    let a_tgt = a_tgt_nominal * 0.4 * urgency;
                    // Only enforce minimum step when we are allowed to advance
                    let s_next = if can_advance {
                        s_updated.max(s_cur + ds_min)
                    } else {
                        s_updated
                    };
                    let progress = if overshot_end || s_next >= s_end {
                        ActionProgress::Complete(AgentState::Hold {
                            coord: action.end_coord(),
                            yaw,
                        })
                    } else {
                        ActionProgress::ProgressTo(s_next, true)
                    };

                    return ChaseTarget {
                        pos: p_tgt,
                        vel: v_tgt,
                        acc: a_tgt,
                        yaw,
                        progress,
                    };
                } else {
                    return ChaseTarget {
                        progress: ActionProgress::Complete(AgentState::Hold {
                            coord: action.end_coord(),
                            yaw: 0.0,
                        }),
                        pos: action.end_coord().cast(),
                        vel: Vector3::zeros(),
                        acc: Vector3::zeros(),
                        ..Default::default()
                    };
                }
            }
            AgentState::Hold { coord, yaw } => ChaseTarget {
                pos: coord.cast(),
                vel: Vector3::zeros(),
                acc: Vector3::zeros(),
                yaw: *yaw,
                ..Default::default()
            },
        }
    }
}
