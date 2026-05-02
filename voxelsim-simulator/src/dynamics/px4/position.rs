use nalgebra::{Vector2, Vector3};

/// PX4 position controller, emulating the necessary parts of the PX4's `PositionControl` class.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PositionControl {
    /// Corresponds to _gain_pos_p.
    gain_pos_p: Vector3<f64>,
    /// Corresponds to _gain_vel_p.
    gain_vel_p: Vector3<f64>,
    gain_vel_i: Vector3<f64>,
    gain_vel_d: Vector3<f64>,
    /// Corresponds to _lim_vel_horizontal.
    lim_vel_horizontal: f64,
    hover_thrust: f64,
    /// Corresponsds to _constraints.
    constraints: CtrlConstraints,
    lim_thr_min: f64,
    lim_thr_max: f64,
}

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct CtrlConstraints {
    speed_up: f64,
    speed_down: f64,
    tilt: f64,
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PositionTarget {
    pos_sp: Vector3<f64>,
    vel_sp: Vector3<f64>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PIDState {
    pub thr_int: Vector3<f64>,
    pub vel_dot: Vector3<f64>,
    pub thr_sp: Vector3<f64>,
}

impl Default for CtrlConstraints {
    fn default() -> Self {
        // Typical PX4-ish vertical speeds and tilt limit (~35 deg)
        Self {
            speed_up: 3.0,
            speed_down: 2.0,
            tilt: 35_f64.to_radians(),
        }
    }
}

impl Default for PIDState {
    fn default() -> Self {
        Self {
            thr_int: Vector3::zeros(),
            vel_dot: Vector3::zeros(),
            thr_sp: Vector3::zeros(),
        }
    }
}

impl Default for PositionControl {
    fn default() -> Self {
        // Reasonable defaults approximating PX4 tuning for small quads
        Self {
            gain_pos_p: Vector3::new(1.0, 1.0, 1.0),
            gain_vel_p: Vector3::new(3.0, 3.0, 3.0),
            gain_vel_i: Vector3::new(0.6, 0.6, 1.2),
            gain_vel_d: Vector3::new(0.05, 0.05, 0.08),
            lim_vel_horizontal: 6.0,
            hover_thrust: 0.8,
            constraints: CtrlConstraints::default(),
            lim_thr_min: 0.1,
            lim_thr_max: 5.0,
        }
    }
}

impl PositionControl {
    pub fn position_controller(
        &self,
        pos: Vector3<f64>,
        pos_sp: Vector3<f64>,
        vel_sp: Vector3<f64>,
    ) -> Vector3<f64> {
        // const Vector3f vel_sp_position = (_pos_sp - _pos).emult(_gain_pos_p);
        let vel_sp_position = (pos_sp - pos).component_mul(&self.gain_pos_p);

        // _vel_sp = vel_sp_position + _vel_sp;
        let mut vel_sp = vel_sp_position + vel_sp;

        // const Vector2f vel_sp_xy = ControlMath::constrainXY(Vector2f(vel_sp_position), Vector2f(_vel_sp - vel_sp_position), _lim_vel_horizontal);
        let vel_sp_xy = constrain_xy(
            vel_sp.xy(),
            (vel_sp - vel_sp_position).xy(),
            self.lim_vel_horizontal,
        );

        // _vel_sp(0) = vel_sp_xy(0);
        // _vel_sp(1) = vel_sp_xy(1);
        vel_sp.x = vel_sp_xy.x;
        vel_sp.y = vel_sp_xy.y;

        // _vel_sp(2) = math::constrain(_vel_sp(2), -_constraints.speed_up, _constraints.speed_down);
        vel_sp.z = vel_sp
            .z
            .clamp(-self.constraints.speed_up, self.constraints.speed_down);

        vel_sp
    }

    /// `vel_dot` is taken from the actual IMU (so in this case is the previous acceleration).
    pub fn velocity_controller(
        &self,
        vel: Vector3<f64>,
        vel_sp: Vector3<f64>,
        mut pid: PIDState,
        dt: f64,
    ) -> PIDState {
        // FROM PX4 source code docs:
        // PID
        // u_des = P(vel_err) + D(vel_err_dot) + I(vel_integral)
        // Umin <= u_des <= Umax
        //
        // Anti-Windup:
        // u_des = _thr_sp; r = _vel_sp; y = _vel
        // u_des >= Umax and r - y >= 0 => Saturation = true
        // u_des >= Umax and r - y <= 0 => Saturation = false
        // u_des <= Umin and r - y <= 0 => Saturation = true
        // u_des <= Umin and r - y >= 0 => Saturation = false
        //
        //  Notes:
        // - PID implementation is in NED-frame
        // - control output in D-direction has priority over NE-direction
        // - the equilibrium point for the PID is at hover-thrust
        // - the maximum tilt cannot exceed 90 degrees. This means that it is
        //   not possible to have a desired thrust direction pointing in the positive
        //   D-direction (= downward)
        // - the desired thrust in D-direction is limited by the thrust limits
        // - the desired thrust in NE-direction is limited by the thrust excess after
        //   consideration of the desired thrust in D-direction. In addition, the thrust in
        //   NE-direction is also limited by the maximum tilt.

        let vel_err = vel_sp - vel;

        // In the z direction.
        let thrust_desired_d =
            self.gain_vel_p.z * vel_err.z + self.gain_vel_d.z * pid.vel_dot.z + pid.thr_int.z
                - self.hover_thrust;

        let mut u_max = -self.lim_thr_min;
        let u_min = -self.lim_thr_max;

        u_max = u_max.min(-10e-4);

        let stop_integral_d = (thrust_desired_d >= u_max && vel_err.z >= 0.0)
            || (thrust_desired_d <= u_min && vel_err.z <= 0.0);

        // Anti-Windup vertical
        if !stop_integral_d {
            pid.thr_int.z += vel_err.z * self.gain_vel_i.z * dt;
            pid.thr_int.z = pid.thr_int.z.abs().min(self.lim_thr_max) * pid.thr_int.z.signum();
        }

        pid.thr_sp.z = thrust_desired_d.clamp(u_min, u_max);

        if pid.thr_sp.x.is_finite() && pid.thr_sp.y.is_finite() {
            let thr_xy_max = pid.thr_sp.z * self.constraints.tilt.tan();
            pid.thr_sp.x *= thr_xy_max;
            pid.thr_sp.y *= thr_xy_max;
        } else {
            // Normal route, when thrust is not yet set.
            let mut thrust_desired_ne: Vector2<f64> = Vector2::default();
            thrust_desired_ne.x =
                self.gain_vel_p.x * vel_err.x + self.gain_vel_d.x * pid.vel_dot.x + pid.thr_int.x;

            thrust_desired_ne.y =
                self.gain_vel_p.y * vel_err.y + self.gain_vel_d.y * pid.vel_dot.y + pid.thr_int.y;

            // Maximum allowed thrust.
            let thrust_max_ne_tilt: f64 = pid.thr_sp.z * self.constraints.tilt.tan();
            let arg = (self.lim_thr_max * self.lim_thr_max) - (pid.thr_sp.z * pid.thr_sp.z);
            let mut thrust_max_ne = arg.max(0.0).sqrt();

            pid.thr_sp.x = thrust_desired_ne.x;
            pid.thr_sp.y = thrust_desired_ne.y;

            if thrust_desired_ne.norm_squared() > thrust_max_ne * thrust_max_ne {
                let mag = thrust_desired_ne.norm();
                pid.thr_sp.x = thrust_desired_ne.x / mag * thrust_max_ne;
                pid.thr_sp.y = thrust_desired_ne.y / mag * thrust_max_ne;
            }

            // Tracking anti-windup
            let arw_gain = 2.0 / self.gain_vel_p.x;

            let mut vel_err_lim: Vector2<f64> = Vector2::default();
            vel_err_lim.x = vel_err.x - (thrust_desired_ne.x - pid.thr_sp.x) * arw_gain;
            vel_err_lim.y = vel_err.y - (thrust_desired_ne.y - pid.thr_sp.y) * arw_gain;

            // Update thrust integral.
            pid.thr_int.x += self.gain_vel_i.x * vel_err_lim.x * dt;
            pid.thr_int.y += self.gain_vel_i.y * vel_err_lim.y * dt;
        }

        pid
    }
}

fn constrain_xy(v0: Vector2<f64>, v1: Vector2<f64>, max: f64) -> Vector2<f64> {
    const EPS: f64 = 1e-3;

    let sum = v0 + v1;
    if sum.norm() <= max {
        // vector does not exceed maximum magnitude
        return sum;
    }

    let v0_len = v0.norm();
    if v0_len >= max {
        // the magnitude along v0, which has priority, already exceeds maximum.
        return v0 * (max / v0_len);
    }

    if (v1 - v0).norm() < EPS {
        // the two vectors are equal
        return if v0_len > 0.0 {
            v0 * (max / v0_len)
        } else {
            Vector2::zeros()
        };
    }

    if v0_len < EPS {
        // the first vector is 0.
        let v1_len = v1.norm();
        return if v1_len > 0.0 {
            v1 * (max / v1_len)
        } else {
            Vector2::zeros()
        };
    }

    // General case:
    // vf = v0 + s * u1, with ||vf|| <= max and u1 = unit(v1)
    let v1_len = v1.norm();
    // v1 should be non-zero here due to earlier branches; still guard just in case.
    if v1_len < EPS {
        return v0 * (max / v0_len);
    }
    let u1 = v1 / v1_len;

    // s = -m + sqrt(m^2 - c), where:
    // m = u1 · v0, c = ||v0||^2 - max^2
    let m = u1.dot(&v0);
    let c = v0.dot(&v0) - max * max;
    let disc = (m * m - c).max(0.0); // clamp to avoid tiny negative due to numerics
    let s = -m + disc.sqrt();

    v0 + u1 * s
}
