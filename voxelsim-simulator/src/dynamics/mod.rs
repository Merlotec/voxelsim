pub mod pid;
#[cfg(feature = "px4")]
pub mod px4;
pub mod quad;

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

use voxelsim::{Agent, chase::ChaseTarget};

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct EnvState {
    pub g: Vector3<f64>,
    pub wind: Vector3<f64>,
}

impl Default for EnvState {
    fn default() -> Self {
        Self {
            g: Vector3::new(0.0, 0.0, -9.8),
            wind: Vector3::zeros(),
        }
    }
}

// Completely controls how the agent's position is updated in the world through time.
pub trait AgentDynamics {
    // Responsible for updating agent physics and action state (i.e. removing actions when they have been completed).
    fn update_agent_dynamics(
        &mut self,
        agent: &mut Agent,
        env: &EnvState,
        chaser: ChaseTarget,
        delta: f64,
    );

    fn bounding_box(&self) -> Vector3<f64>;
}

// Determines external factors e.g. wind.
pub trait EnvDynamics {
    fn agent_env(&self, agent: &Agent) -> EnvState;
}

// World: ENU <-> NED (swap X/Y, flip Z)
#[inline]
fn enu_ned(v: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(v.y, v.x, -v.z)
}
// Body: FLU <-> FRD (keep X, flip Y & Z)
#[inline]
fn flu_frd(v: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(v.x, -v.y, -v.z)
}

#[inline]
fn c_ned_from_enu() -> Matrix3<f64> {
    Matrix3::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
}
#[inline]
fn c_frd_from_flu() -> Matrix3<f64> {
    Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0)
}
#[inline]
fn q_ned_frd_to_enu_flu(q_ned_frd: UnitQuaternion<f64>) -> UnitQuaternion<f64> {
    let r = q_ned_frd.to_rotation_matrix().into_inner();
    let r_conv = c_ned_from_enu() * r * c_frd_from_flu();
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_conv))
}
#[inline]
fn q_enu_flu_to_ned_frd(q_enu_flu: UnitQuaternion<f64>) -> UnitQuaternion<f64> {
    let r = q_enu_flu.to_rotation_matrix().into_inner();
    let r_conv = c_ned_from_enu() * r * c_frd_from_flu();
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_conv))
}

pub fn run_simulation_tick_rk4(
    agent: &mut Agent,                     // PX4-style FRD/NED
    quad: &mut peng_quad::Quadrotor,       // Peng FLU/ENU
    control_thrust_body_z_px4: f64,        // = att_sp.thrust_body[2] (negative)
    control_torque_body_frd: Vector3<f64>, // PX4 rate torque (body FRD)
    dt: f64,
) {
    // State to Peng
    quad.position = enu_ned(agent.pos).cast(); // NED -> ENU
    quad.velocity = enu_ned(agent.vel).cast(); // NED -> ENU
    quad.orientation = q_ned_frd_to_enu_flu(agent.attitude).cast(); // FRD/NED -> FLU/ENU
    quad.angular_velocity = flu_frd(agent.rate).cast(); // FRD -> FLU

    quad.time_step = dt as f32;

    // Controls to Peng
    let thrust_mag_peng = (-control_thrust_body_z_px4).max(0.0) as f32; // +|T| along +Z (FLU)
    let torque_flu = flu_frd(control_torque_body_frd).cast(); // FRD -> FLU

    quad.update_dynamics_with_controls_rk4(thrust_mag_peng, &torque_flu);

    // State back to PX4-style
    agent.pos = enu_ned(quad.position.cast()); // ENU -> NED
    agent.vel = enu_ned(quad.velocity.cast()); // ENU -> NED
    agent.attitude = q_enu_flu_to_ned_frd(quad.orientation.cast()); // FLU/ENU -> FRD/NED
    agent.rate = flu_frd(quad.angular_velocity.cast()); // FLU -> FRD
}
