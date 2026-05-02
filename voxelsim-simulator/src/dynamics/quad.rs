use nalgebra::{Matrix3, Rotation3, Vector3};
use peng_quad::Quadrotor;
use voxelsim::{
    Agent, AgentState,
    chase::{ActionProgress, ChaseTarget},
};

use crate::dynamics::{
    AgentDynamics,
    pid::{self, AttitudePParams, PosVelState, Px4PosVelParams, RatePIDParams, RatePIDState},
};

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct QuadParams {
    mass: f64,
    gravity: f64,
    drag_coefficient: f64,
    inertia_matrix: Matrix3<f64>,
    bounding_box: Vector3<f64>,
    // PX4-like pos/vel params (hover and moving)
    pos_params: Px4PosVelParams,
    moving_pos_params: Px4PosVelParams,
    rate_params: RatePIDParams,
    moving_rate_params: RatePIDParams,
    att_p: AttitudePParams,
    thrust_min: f64,
    thrust_max: f64,
    // Control axis sign mapping: multiply setpoints by this before control
    // Use (-1,-1,-1) to invert FB/LR/UD to match external input convention
    control_sign: Vector3<f64>,
}

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct QuadDynamics {
    quad: Quadrotor,
    params: QuadParams,
    pos_state: PosVelState,
    rate_state: RatePIDState,
    // Last commanded body torque (for slew limiting)
    last_torque: Vector3<f64>,
    // Counter for consecutive yaw saturation frames
    sat_z_count: u32,
    // Yaw setpoint generator state
    yaw_sp: f64,
    yaw_rate_cmd: f64,
    yaw_initialized: bool,
    // Blend factor between hover and moving rate params [0..1]
    rate_blend: f64,
    // Low-pass filter state for body-rate setpoint
    rate_sp_filt: Vector3<f64>,
    // Track idle state to reset integrators on transition
    last_idle: bool,
}

impl Default for QuadParams {
    fn default() -> Self {
        Self::new(
            1.0,
            9.81,
            0.1,
            Matrix3::new(0.0347563, 0.0, 0.0, 0.0, 0.0458929, 0.0, 0.0, 0.0, 0.0977),
            Vector3::new(0.5, 0.5, 0.1),
        )
    }
}

impl QuadDynamics {
    pub fn new(settings: QuadParams) -> Self {
        let quad = Quadrotor::new(
            0.01,
            settings.mass as f32,
            settings.gravity as f32,
            settings.drag_coefficient as f32,
            settings
                .inertia_matrix
                .cast::<f32>()
                .as_slice()
                .try_into()
                .unwrap(),
        )
        .unwrap();

        Self {
            quad,
            pos_state: Default::default(),
            rate_state: Default::default(),
            params: settings,
            last_torque: Vector3::zeros(),
            sat_z_count: 0,
            yaw_sp: 0.0,
            yaw_rate_cmd: 0.0,
            yaw_initialized: false,
            rate_blend: 0.0,
            rate_sp_filt: Vector3::zeros(),
            last_idle: false,
        }
    }
}

impl QuadParams {
    pub fn new(
        mass: f64,
        gravity: f64,
        drag_coefficient: f64,
        inertia_matrix: Matrix3<f64>,
        bounding_box: Vector3<f64>,
    ) -> Self {
        Self {
            mass,
            gravity,
            drag_coefficient,
            inertia_matrix,
            bounding_box: bounding_box,
            pos_params: Px4PosVelParams::default(),
            moving_pos_params: Px4PosVelParams::default_moving(),
            rate_params: Default::default(),
            moving_rate_params: RatePIDParams::default_moving(),
            att_p: AttitudePParams::default(),
            thrust_min: 0.05,
            thrust_max: 2.5,
            // Use ENU (X-east, Y-north, Z-up) throughout; no inversion by default
            control_sign: Vector3::new(1.0, 1.0, 1.0),
        }
    }
}

impl AgentDynamics for QuadDynamics {
    fn update_agent_dynamics(
        &mut self,
        agent: &mut Agent,
        _env: &super::EnvState,
        chaser: ChaseTarget,
        delta: f64,
    ) {
        let idle = !matches!(&chaser.progress, ActionProgress::ProgressTo(_, _));
        let (mut t_pos, mut t_vel, mut t_acc, pos_params_sel, _rate_params_sel, target_blend) =
            match chaser.progress {
                ActionProgress::ProgressTo(progress, trim_tail) => {
                    if let AgentState::Action(action) = &mut agent.state {
                        action.update_progress(progress, trim_tail);
                    }
                    // Debug print removed to avoid runtime jitter
                    (
                        chaser.pos,
                        chaser.vel,
                        chaser.acc,
                        self.params.moving_pos_params,
                        self.params.moving_rate_params,
                        1.0,
                    )
                }
                ActionProgress::Complete(next_state) => {
                    agent.state = next_state;
                    (
                        agent.pos,
                        Vector3::zeros(),
                        Vector3::zeros(),
                        self.params.pos_params,
                        self.params.rate_params,
                        0.0,
                    )
                }
                ActionProgress::Hold => (
                    agent.pos,
                    Vector3::zeros(),
                    Vector3::zeros(),
                    self.params.pos_params,
                    self.params.rate_params,
                    0.0,
                ),
            };
        // Smoothly blend rate controller params between hover and moving profiles to avoid step changes
        let tau_blend = 0.6_f64; // seconds
        let alpha = (-delta / tau_blend).exp();
        self.rate_blend = self.rate_blend * alpha + target_blend * (1.0 - alpha);
        let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
        let lerp_v3 = |a: Vector3<f64>, b: Vector3<f64>, t: f64| {
            Vector3::new(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t))
        };
        let rp_hover = self.params.rate_params;
        let rp_move = self.params.moving_rate_params;
        let rate_params = pid::RatePIDParams {
            kp: lerp_v3(rp_hover.kp, rp_move.kp, self.rate_blend),
            ki: lerp_v3(rp_hover.ki, rp_move.ki, self.rate_blend),
            kd: lerp_v3(rp_hover.kd, rp_move.kd, self.rate_blend),
            max_torque: lerp_v3(rp_hover.max_torque, rp_move.max_torque, self.rate_blend),
            max_integral: lerp_v3(rp_hover.max_integral, rp_move.max_integral, self.rate_blend),
            d_lpf_hz: lerp(rp_hover.d_lpf_hz, rp_move.d_lpf_hz, self.rate_blend),
        };
        // Apply control axis sign mapping to setpoints (PX4 uses NED; external may be ENU)
        // We flip relative directions by applying sign to deltas for position and directly to vel/acc.
        // let sign = self.params.control_sign;
        // t_pos = self.quad.position.cast::<f64>()
        //     + sign.component_mul(&(t_pos - self.quad.position.cast::<f64>()));
        // t_vel = sign.component_mul(&t_vel);
        // t_acc = sign.component_mul(&t_acc);
        self.quad.position = agent.pos.cast::<f32>();
        self.quad.velocity = agent.vel.cast::<f32>();

        // Determine idle (no active trajectory progress)

        // On transition to idle, reset integrators and filters to avoid latent windup
        if idle && !self.last_idle {
            self.pos_state.vel_integral = Vector3::zeros();
            self.rate_state.integral = Vector3::zeros();
            self.rate_state.d_filt = Vector3::zeros();
            self.rate_sp_filt = Vector3::zeros();
            self.last_torque = Vector3::zeros();
        }
        self.last_idle = idle;

        // Acceleration setpoint: use full PX4 cascade when moving; in idle, use simple velocity damping
        let (a_sp, _sat) = if idle {
            let v_act = self.quad.velocity.cast::<f64>();
            // Critically-damped-ish velocity brake; stronger on Z
            let k_v = Vector3::new(1.2, 1.2, 1.8);
            ((-k_v).component_mul(&v_act), false)
        } else {
            pid::px4_accel_sp(
                self.quad.position.cast::<f64>(),
                self.quad.velocity.cast::<f64>(),
                t_pos,
                t_vel,
                t_acc,
                &pos_params_sel,
                &mut self.pos_state,
                delta,
            )
        };
        // Total accel includes gravity compensation in ENU (Z-up)
        let a_total = a_sp + Vector3::new(0.0, 0.0, self.quad.gravity as f64);
        let mass = self.params.mass;
        let hover_thrust = mass * self.params.gravity;
        let thrust_ideal = mass * a_total.norm();
        let thrust = thrust_ideal.clamp(
            self.params.thrust_min * hover_thrust,
            self.params.thrust_max * hover_thrust,
        );

        let norm = a_total.norm();
        let z_b_cmd = if norm > 1e-6 {
            a_total / norm
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };

        // PX4 AttitudeControl equivalent: attitude error -> body rate setpoint
        let r_act = self.quad.orientation.to_rotation_matrix().cast::<f64>();
        // Compute current world yaw from body X axis.
        let x_b_world = r_act.matrix().column(0);
        let yaw_current = x_b_world[1].atan2(x_b_world[0]);
        // Initialize filtered yaw setpoint on first run
        if !self.yaw_initialized {
            self.yaw_sp = yaw_current;
            self.yaw_rate_cmd = 0.0;
            self.yaw_initialized = true;
        }
        // Yaw setpoint generator: smooth target into yaw_sp/yaw_rate_cmd
        let wrap = |mut a: f64| {
            while a > std::f64::consts::PI {
                a -= 2.0 * std::f64::consts::PI;
            }
            while a < -std::f64::consts::PI {
                a += 2.0 * std::f64::consts::PI;
            }
            a
        };
        // If idle (no active progress), follow current yaw to avoid fighting small drifts
        let target_yaw = chaser.yaw;

        let e_yaw = wrap(target_yaw - self.yaw_sp);
        let omega_z = self.quad.angular_velocity.z as f64;
        let e_abs = e_yaw.abs();
        // Taper gains and max rate as we approach target yaw to avoid overshoot-induced oscillation
        let small = 0.4; // rad where tapering begins
        let scale = ((e_abs / small).min(1.0)).powi(2); // quadratic taper to zero near target
        let kp_rate_base = 2.0; // base proportional mapping (rad -> rad/s)
        let kp_rate = kp_rate_base * scale;
        let kd_rate = 0.8; // damp with actual yaw rate (critically damp decel)
        let max_rate_base = 1.0; // rad/s
        let max_rate = max_rate_base * (0.5 + 0.5 * scale); // taper max rate near the goal
        let max_acc = 5.0; // rad/s^2, limit yaw rate change for stability

        let yaw_rate_des = (kp_rate * e_yaw - kd_rate * omega_z).clamp(-max_rate, max_rate);
        let delta_rate = yaw_rate_des - self.yaw_rate_cmd;
        let max_delta_rate = max_acc * delta;
        self.yaw_rate_cmd += delta_rate.clamp(-max_delta_rate, max_delta_rate);
        // Near target, snap to rest to avoid dithering
        if e_abs < 0.03 && omega_z.abs() < 0.05 {
            self.yaw_rate_cmd = 0.0;
            self.yaw_sp = yaw_current;
        } else {
            self.yaw_sp = wrap(self.yaw_sp + self.yaw_rate_cmd * delta);
        }

        // Build attitude setpoint using filtered yaw_sp (or hold current yaw when idle)
        let yaw_for_cmd = if idle { yaw_current } else { self.yaw_sp };
        let r_cmd =
            Rotation3::from_matrix_unchecked(pid::build_body_rotation(&z_b_cmd, yaw_for_cmd));
        // Yaw feedforward only when moving
        let yaw_rate_ff = if idle { 0.0 } else { self.yaw_rate_cmd };
        let mut rate_sp =
            pid::attitude_to_bodyrate(&r_cmd, &r_act, yaw_rate_ff, &self.params.att_p.p);
        // Clamp yaw body-rate to avoid torque saturation stealing authority
        let yaw_rate_limit = 1.0; // rad/s
        rate_sp.z = rate_sp.z.clamp(-yaw_rate_limit, yaw_rate_limit);
        // If idle, apply small deadband and soften rate demands to prevent self-excitation
        if idle {
            let dead = 0.2;
            if rate_sp.x.abs() < dead {
                rate_sp.x = 0.0;
            }
            if rate_sp.y.abs() < dead {
                rate_sp.y = 0.0;
            }
            if rate_sp.z.abs() < dead {
                rate_sp.z = 0.0;
            }
            // Soften remaining rates when idle
            rate_sp *= 0.8;
        }
        // Global body-rate magnitude clamp to bound combined roll/pitch/yaw demand
        let rate_norm_limit = if idle { 3.5 } else { 4.5 }; // rad/s overall magnitude limit
        let rate_norm = rate_sp.norm();
        if rate_norm > rate_norm_limit {
            let scale = rate_norm_limit / rate_norm;
            rate_sp *= scale;
        }
        // Low-pass filter the body-rate setpoint to avoid exciting high-frequency dynamics
        let fc_sp = 10.0_f64; // Hz
        let tau_sp = 1.0 / (2.0 * std::f64::consts::PI * fc_sp);
        let alpha_sp = tau_sp / (tau_sp + delta);
        self.rate_sp_filt = self.rate_sp_filt * alpha_sp + rate_sp * (1.0 - alpha_sp);
        let rate_sp = self.rate_sp_filt;

        // PX4 RateControl equivalent: PID on body rates
        // PX4-like RateControl with simple per-axis anti-windup (freeze on same-direction saturation)
        let rate_error = rate_sp - self.quad.angular_velocity.cast::<f64>();
        let p_term = Vector3::new(
            rate_params.kp.x * rate_error.x,
            rate_params.kp.y * rate_error.y,
            rate_params.kp.z * rate_error.z,
        );
        // Derivative with simple low-pass filtering to reduce oscillation (approx PX4 gyro filter)
        let raw_d = (rate_error - self.rate_state.last_error) / delta;
        let fc = rate_params.d_lpf_hz.max(0.1);
        let tau = 1.0 / (2.0 * std::f64::consts::PI * fc);
        let alpha = tau / (tau + delta);
        self.rate_state.d_filt = self.rate_state.d_filt * alpha + raw_d * (1.0 - alpha);
        let d_err = self.rate_state.d_filt;
        let d_term = Vector3::new(
            rate_params.kd.x * d_err.x,
            rate_params.kd.y * d_err.y,
            rate_params.kd.z * d_err.z,
        );
        let integral_candidate = self.rate_state.integral + rate_error * delta;
        let i_term_candidate = Vector3::new(
            rate_params.ki.x * integral_candidate.x,
            rate_params.ki.y * integral_candidate.y,
            rate_params.ki.z * integral_candidate.z,
        );
        let raw_candidate = p_term + i_term_candidate + d_term;
        let mt_active = rate_params.max_torque;
        // Anti-windup masks per axis
        let allow_x = !((raw_candidate.x > mt_active.x && rate_error.x > 0.0)
            || (raw_candidate.x < -mt_active.x && rate_error.x < 0.0));
        let allow_y = !((raw_candidate.y > mt_active.y && rate_error.y > 0.0)
            || (raw_candidate.y < -mt_active.y && rate_error.y < 0.0));
        let allow_z = !((raw_candidate.z > mt_active.z && rate_error.z > 0.0)
            || (raw_candidate.z < -mt_active.z && rate_error.z < 0.0));
        if allow_x {
            self.rate_state.integral.x = integral_candidate.x;
        }
        if allow_y {
            self.rate_state.integral.y = integral_candidate.y;
        }
        if allow_z {
            self.rate_state.integral.z = integral_candidate.z;
        }
        let mi_active = rate_params.max_integral;
        self.rate_state.integral.x = self.rate_state.integral.x.clamp(-mi_active.x, mi_active.x);
        self.rate_state.integral.y = self.rate_state.integral.y.clamp(-mi_active.y, mi_active.y);
        self.rate_state.integral.z = self.rate_state.integral.z.clamp(-mi_active.z, mi_active.z);
        let mut raw_torque = Vector3::new(
            p_term.x + rate_params.ki.x * self.rate_state.integral.x + d_term.x,
            p_term.y + rate_params.ki.y * self.rate_state.integral.y + d_term.y,
            p_term.z + rate_params.ki.z * self.rate_state.integral.z + d_term.z,
        );
        // Add passive viscous damping, especially useful in hover
        let ang_vel = self.quad.angular_velocity.cast::<f64>();
        let passive_damp = if idle {
            Vector3::new(0.02, 0.02, 0.03)
        } else {
            Vector3::new(0.05, 0.05, 0.07)
        };
        raw_torque -= passive_damp.component_mul(&ang_vel);
        // Priority clamp: roll/pitch first, allocate remaining authority to yaw
        let mut torque = Vector3::new(
            raw_torque.x.clamp(-mt_active.x, mt_active.x),
            raw_torque.y.clamp(-mt_active.y, mt_active.y),
            raw_torque.z.clamp(-mt_active.z, mt_active.z),
        );
        // Track yaw saturation frames; leak yaw integrator if sustained
        if torque.z.abs() >= mt_active.z * 0.999 {
            self.sat_z_count = self.sat_z_count.saturating_add(1);
        } else {
            self.sat_z_count = 0;
        }
        if self.sat_z_count >= 5 {
            self.rate_state.integral.z *= 0.98; // small leak
        }

        // If idle and nearly still, aggressively bleed integrators to prevent delayed wobble
        let idle_near_still = idle && ang_vel.norm() < 0.3;
        if idle_near_still {
            self.rate_state.integral *= 0.9;
            self.pos_state.vel_integral *= 0.9;
        }

        // Slew-limit torque command to avoid abrupt steps
        let torque_slew_rate = Vector3::new(6.0, 6.0, 5.0); // N·m per second (moderated)
        let max_delta = torque_slew_rate * delta;
        let delta_torque = torque - self.last_torque;
        let limited_delta = Vector3::new(
            delta_torque.x.clamp(-max_delta.x, max_delta.x),
            delta_torque.y.clamp(-max_delta.y, max_delta.y),
            delta_torque.z.clamp(-max_delta.z, max_delta.z),
        );
        torque = self.last_torque + limited_delta;
        self.last_torque = torque;

        // Idle hygiene: when not progressing, slowly decay integrators to prevent delayed oscillations
        if !idle {
            let decay = (-delta / 1.2).exp();
            self.rate_state.integral *= decay;
            self.rate_state.d_filt *= decay;
            self.pos_state.vel_integral *= decay;
            // bleed yaw rate command toward zero
            self.yaw_rate_cmd *= decay;
        }
        self.rate_state.last_error = rate_error;
        self.quad.time_step = delta as f32;

        self.quad
            .update_dynamics_with_controls_rk4(thrust as f32, &torque.cast::<f32>());

        agent.pos = self.quad.position.cast::<f64>();
        agent.vel = self.quad.velocity.cast::<f64>();
        agent.attitude = self.quad.orientation.cast::<f64>();
    }
    fn bounding_box(&self) -> Vector3<f64> {
        self.params.bounding_box
    }
}
