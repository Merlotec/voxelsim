use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

// PX4-like Position/Velocity controller parameters
// Maps conceptually to PX4 params:
// - pos_p:   MPC_XY_P (x,y), MPC_Z_P (z)
// - vel_p/i/d: MPC_XY_VEL_{P,I,D} and MPC_Z_VEL_{P,I,D}
// - acc limits: MPC_ACC_HOR_MAX, MPC_ACC_UP_MAX, MPC_ACC_DOWN_MAX
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Px4PosVelParams {
    pub pos_p: Vector3<f64>,
    pub vel_p: Vector3<f64>,
    pub vel_i: Vector3<f64>,
    pub vel_d: Vector3<f64>,
    pub acc_hor_max: f64,
    pub acc_up_max: f64,
    pub acc_down_max: f64,
    pub i_limit: Vector3<f64>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PosVelState {
    pub vel_integral: Vector3<f64>,
    pub last_vel_error: Vector3<f64>,
}

impl Default for Px4PosVelParams {
    fn default() -> Self {
        Self {
            // Typical PX4-ish defaults (tuned for sim):
            pos_p: Vector3::new(1.0, 1.0, 1.0),
            vel_p: Vector3::new(2.0, 2.0, 2.0),
            // Hover profile: very small I, higher D to avoid drift/wobble at rest
            vel_i: Vector3::new(0.05, 0.05, 0.10),
            vel_d: Vector3::new(0.40, 0.40, 0.30),
            // Conservative accel limits in hover to reduce excitation
            acc_hor_max: 4.0,
            acc_up_max: 3.5,
            acc_down_max: 3.5,
            // Tighter I clamp at hover
            i_limit: Vector3::new(0.5, 0.5, 0.5),
        }
    }
}

impl Px4PosVelParams {
    pub fn default_moving() -> Self {
        // Moving profile: more damping in XY, lower I to avoid sway
        Self {
            pos_p: Vector3::new(1.0, 1.0, 1.0),
            vel_p: Vector3::new(2.4, 2.4, 2.0),
            vel_i: Vector3::new(0.0, 0.0, 0.2),
            vel_d: Vector3::new(0.7, 0.7, 0.25),
            // Reduce vertical accel in motion to match horizontal responsiveness
            acc_hor_max: 6.0,
            acc_up_max: 4.0,
            acc_down_max: 4.0,
            i_limit: Vector3::new(1.5, 1.5, 2.0),
        }
    }
}

impl Default for PosVelState {
    fn default() -> Self {
        Self {
            vel_integral: Vector3::zeros(),
            last_vel_error: Vector3::zeros(),
        }
    }
}

// PX4 Position->Velocity->Acceleration cascade with accel limits and simple anti-windup.
// Equivalent flow:
// vel_sp = v_ff + MPC_*_P * (pos_sp - pos)
// a_sp = MPC_*_VEL_P * (vel_sp - vel) + I + MPC_*_VEL_D * d/dt(vel_err) + a_ff
// limit a_sp by XY/Z limits (tilt/collective priority)
pub fn px4_accel_sp(
    p_act: Vector3<f64>,
    v_act: Vector3<f64>,
    p_sp: Vector3<f64>,
    v_ff: Vector3<f64>,
    a_ff: Vector3<f64>,
    params: &Px4PosVelParams,
    state: &mut PosVelState,
    dt: f64,
) -> (Vector3<f64>, bool) {
    let pos_err = p_sp - p_act;
    let v_sp = v_ff + params.pos_p.component_mul(&pos_err);
    let mut vel_err = v_sp - v_act;

    // Apply small deadband on velocity error per axis to avoid self-excitation near hover
    let vel_db = Vector3::new(0.05, 0.05, 0.05);
    if vel_err.x.abs() < vel_db.x { vel_err.x = 0.0; }
    if vel_err.y.abs() < vel_db.y { vel_err.y = 0.0; }
    if vel_err.z.abs() < vel_db.z { vel_err.z = 0.0; }

    // PI-D on velocity error
    // Only integrate when error is outside deadband
    state.vel_integral += vel_err * dt;
    // clamp integrator
    state.vel_integral.x = state
        .vel_integral
        .x
        .clamp(-params.i_limit.x, params.i_limit.x);
    state.vel_integral.y = state
        .vel_integral
        .y
        .clamp(-params.i_limit.y, params.i_limit.y);
    state.vel_integral.z = state
        .vel_integral
        .z
        .clamp(-params.i_limit.z, params.i_limit.z);

    let d_err = if dt > 0.0 {
        (vel_err - state.last_vel_error) / dt
    } else {
        Vector3::zeros()
    };

    let p_term = params.vel_p.component_mul(&vel_err);
    let i_term = params.vel_i.component_mul(&state.vel_integral);
    let d_term = params.vel_d.component_mul(&d_err);
    let mut a_sp = p_term + i_term + d_term + a_ff;
    state.last_vel_error = vel_err;

    // Acceleration limits (PX4: MPC_ACC_HOR_MAX, MPC_ACC_UP/DOWN_MAX)
    let mut saturated = false;
    // Clamp vertical
    if a_sp.z > params.acc_up_max {
        a_sp.z = params.acc_up_max;
        saturated = true;
    }
    if a_sp.z < -params.acc_down_max {
        a_sp.z = -params.acc_down_max;
        saturated = true;
    }
    // Clamp horizontal magnitude
    let a_xy = Vector3::new(a_sp.x, a_sp.y, 0.0);
    let a_xy_norm = a_xy.norm();
    if a_xy_norm > params.acc_hor_max {
        let scale = params.acc_hor_max / a_xy_norm;
        a_sp.x *= scale;
        a_sp.y *= scale;
        saturated = true;
    }

    // Basic anti-windup: freeze integrator if saturated
    if saturated {
        state.vel_integral -= vel_err * dt; // revert last step
    }

    (a_sp, saturated)
}

/// PID parameters for position controller
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PosPIDParams {
    pub kp: Vector3<f64>,
    pub ki: Vector3<f64>,
    pub kd: Vector3<f64>,
}

/// Holds integrals and previous errors for PID
pub struct PosPIDState {
    pub pos_integral: Vector3<f64>,
    pub last_vel_error: Vector3<f64>,
}

/// PID parameters for rate controller
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RatePIDParams {
    pub kp: Vector3<f64>,
    pub ki: Vector3<f64>,
    pub kd: Vector3<f64>,
    pub max_torque: Vector3<f64>,
    pub max_integral: Vector3<f64>,
    // Derivative term low-pass cutoff (Hz), approximates PX4 gyro filtering influence
    pub d_lpf_hz: f64,
}

/// Holds integrals and previous errors for PID
pub struct RatePIDState {
    pub integral: Vector3<f64>,
    pub last_error: Vector3<f64>,
    pub d_filt: Vector3<f64>,
}

/// Attitude P gains to convert attitude error (axis-angle)
/// into body-rate setpoints, emulating PX4 AttitudeControl mapping.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AttitudePParams {
    pub p: Vector3<f64>,
}

impl Default for AttitudePParams {
    fn default() -> Self {
        // Approximate PX4 attitude P: roll/pitch around ~6 rad/s, yaw lower.
        Self {
            p: Vector3::new(6.0, 6.0, 3.0),
        }
    }
}

impl Default for PosPIDParams {
    fn default() -> Self {
        PosPIDParams {
            // Safer defaults to reduce aggressive reversal acceleration
            kp: Vector3::new(1.2, 1.2, 1.0),
            ki: Vector3::new(0.0, 0.0, 0.0),
            kd: Vector3::new(0.5, 0.5, 0.4),
        }
    }
}

impl PosPIDParams {
    pub fn default_moving() -> Self {
        PosPIDParams {
            // Sharper moving gains for tighter path tracking
            kp: Vector3::new(3.2, 3.2, 2.4),
            ki: Vector3::new(0.0, 0.0, 0.0),
            kd: Vector3::new(1.1, 1.1, 0.8),
        }
    }
}

impl Default for PosPIDState {
    fn default() -> Self {
        PosPIDState {
            pos_integral: Vector3::zeros(),
            last_vel_error: Vector3::zeros(),
        }
    }
}

impl Default for RatePIDParams {
    fn default() -> Self {
        RatePIDParams {
            // Hover profile: lower P, higher D, reduced I for stability at rest
            kp: Vector3::new(1.8, 1.8, 2.4),
            ki: Vector3::new(0.01, 0.01, 0.01),
            kd: Vector3::new(0.14, 0.14, 0.06),

            // Torques moderated in hover to reduce blow-ups
            max_torque: Vector3::new(0.20, 0.20, 0.35),

            // Prevent integral windup (tighter in hover)
            max_integral: Vector3::new(0.20, 0.20, 0.10),

            // Derivative filter cutoff (lower for more damping at rest)
            d_lpf_hz: 12.0,
        }
    }
}

impl RatePIDParams {
    pub fn default_moving() -> Self {
        RatePIDParams {
            // Moving gains: strong roll/pitch damping, very low I to prevent lateral sway
            kp: Vector3::new(2.6, 2.6, 4.2),
            ki: Vector3::new(0.015, 0.015, 0.03),
            kd: Vector3::new(0.16, 0.16, 0.045),

            // Slightly higher torque limits for moving, still conservative
            max_torque: Vector3::new(0.28, 0.28, 0.50),

            // Integral clamp for moving case
            max_integral: Vector3::new(0.36, 0.36, 0.18),

            d_lpf_hz: 15.0,
        }
    }
}

impl Default for RatePIDState {
    fn default() -> Self {
        RatePIDState {
            integral: Vector3::zeros(),
            last_error: Vector3::zeros(),
            d_filt: Vector3::zeros(),
        }
    }
}

/// Compute desired acceleration from position/velocity targets
/// a_cmd = Kp_pos * pos_error + Ki_pos * ∫pos_error dt + Kd_pos * vel_error + a_tgt (feedforward)
pub fn compute_accel_cmd(
    p_act: Vector3<f64>,
    v_act: Vector3<f64>,
    p_tgt: Vector3<f64>,
    v_tgt: Vector3<f64>,
    a_tgt: Vector3<f64>,
    pid_params: &PosPIDParams,
    pid_state: &mut PosPIDState,
    dt: f64,
) -> Vector3<f64> {
    // Position error
    let pos_error = p_tgt - p_act;
    // Integrate position error
    pid_state.pos_integral += pos_error * dt;
    // Simple anti-windup clamp on integral
    let i_lim = Vector3::new(1.0, 1.0, 0.8);
    pid_state.pos_integral.x = pid_state.pos_integral.x.clamp(-i_lim.x, i_lim.x);
    pid_state.pos_integral.y = pid_state.pos_integral.y.clamp(-i_lim.y, i_lim.y);
    pid_state.pos_integral.z = pid_state.pos_integral.z.clamp(-i_lim.z, i_lim.z);
    // Velocity error (desired minus actual)
    let vel_error = v_tgt - v_act;
    // PID terms
    let p_term = pid_params.kp.component_mul(&pos_error);
    let i_term = pid_params.ki.component_mul(&pid_state.pos_integral);
    let d_term = pid_params.kd.component_mul(&vel_error);
    // Save last velocity error
    pid_state.last_vel_error = vel_error;
    // Compute a_cmd
    p_term + i_term + d_term + a_tgt
}

/// Convert desired body orientation R_cmd and current orientation R_act
/// into body rate setpoint using small-angle approximation
pub fn attitude_to_bodyrate(
    r_cmd: &Rotation3<f64>,
    r_act: &Rotation3<f64>,
    yaw_rate_ff: f64,
    att_p: &Vector3<f64>,
) -> Vector3<f64> {
    // 1. Convert each rotation to a quaternion
    let q_cmd = UnitQuaternion::from_rotation_matrix(r_cmd);
    let q_act = UnitQuaternion::from_rotation_matrix(r_act);

    // 2. Compute the error quaternion as q_err = q_cmd * q_act^{-1}
    //    This matches the original convention used in this sim
    let q_err = q_cmd * q_act.inverse();

    // 3. scaled_axis() returns (axis * angle) as a Vector3
    let rot_vec = q_err.scaled_axis();

    // 4. Map attitude error to body rates with P gain and add yaw FF
    //    This emulates PX4 AttitudeControl: rate_sp = att_p * e_rot + yaw_ff
    att_p.component_mul(&rot_vec) + Vector3::new(0.0, 0.0, yaw_rate_ff)
}

/// Compute desired torque from rate error using PID
pub fn control_torque(
    rate_error: Vector3<f64>,
    pid_state: &mut RatePIDState,
    params: &RatePIDParams,
    dt: f64,
) -> Vector3<f64> {
    let p_term = params.kp.component_mul(&rate_error);
    pid_state.integral += rate_error * dt;
    let i_term = params.ki.component_mul(&pid_state.integral);
    let d_error = (rate_error - pid_state.last_error) / dt;
    let d_term = params.kd.component_mul(&d_error);
    pid_state.last_error = rate_error;
    p_term + i_term + d_term
}

pub fn build_body_rotation(z_b_cmd: &Vector3<f64>, yaw: f64) -> Matrix3<f64> {
    // 1) Ensure Z is unit length
    let z_b = z_b_cmd.normalize();

    // 2) Desired “course” X projected onto the horizontal plane
    let x_c = Vector3::new(yaw.cos(), yaw.sin(), 0.0);

    // 3) Compute body-Y = Z × Xc, then normalize
    let y_b = z_b.cross(&x_c).normalize();

    // 4) Body-X = Y × Z
    let x_b = y_b.cross(&z_b);

    // 5) Build rotation matrix: columns are Xb, Yb, Zb
    Matrix3::from_columns(&[x_b, y_b, z_b])
}

/// Build desired body rotation where yaw is applied relative to the current attitude,
/// rotating around the desired Z axis (z_b_cmd).
/// This interprets `yaw_delta` as an incremental rotation about z_b_cmd, starting from
/// the current body X axis projected onto the plane orthogonal to z_b_cmd.
pub fn build_body_rotation_with_yaw_delta(
    z_b_cmd: &Vector3<f64>,
    yaw_delta: f64,
    r_act: &Rotation3<f64>,
) -> Matrix3<f64> {
    use nalgebra::{Rotation3 as R3, Unit, Vector3 as V3};
    // Desired thrust axis as a unit vector
    let z_unit: Unit<V3<f64>> = Unit::new_normalize(*z_b_cmd);
    let z_b: V3<f64> = z_unit.as_ref().clone();

    // Start from current body X, projected onto plane orthogonal to z_b
    let x_act = r_act.matrix().column(0).into_owned();
    let x_proj = x_act - z_b * x_act.dot(&z_b);

    // If degenerate (near parallel), fall back to body Y projected
    let x_base = if x_proj.norm() > 1e-6 {
        x_proj.normalize()
    } else {
        let y_act = r_act.matrix().column(1).into_owned();
        let y_proj = y_act - z_b * y_act.dot(&z_b);
        if y_proj.norm() > 1e-6 {
            y_proj.normalize()
        } else {
            // Final fallback: construct any orthonormal X on the plane
            // pick a world axis not aligned with z_b
            let aux = if z_b.x.abs() < 0.9 { V3::x() } else { V3::y() };
            let aux_proj = (aux - z_b * aux.dot(&z_b)).normalize();
            aux_proj
        }
    };

    // Rotate x_base around z_b by yaw_delta
    let rot_about_z = R3::from_axis_angle(&z_unit, yaw_delta);
    let x_b = (rot_about_z * x_base).normalize();
    let y_b = z_b.cross(&x_b).normalize();
    Matrix3::from_columns(&[x_b, y_b, z_b])
}
