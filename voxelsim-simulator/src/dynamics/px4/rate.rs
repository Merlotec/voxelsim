use nalgebra::Vector3;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Lpf1 {
    alpha: f64, // 0..1
    state: f64,
}
impl Lpf1 {
    pub fn new() -> Self {
        Self {
            alpha: 0.0,
            state: 0.0,
        }
    }
    pub fn set_cutoff(&mut self, loop_rate_hz: f64, cutoff_hz: f64) {
        // PX4-equivalent bilinear transform (~) for alpha
        // alpha = dt / (dt + 1/(2π fc))
        let dt = 1.0 / loop_rate_hz.max(1e-6);
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz.max(1e-3));
        self.alpha = (dt / (dt + rc)).clamp(0.0, 1.0);
    }
    pub fn reset(&mut self, x: f64) {
        self.state = x;
    }
    pub fn apply(&mut self, x: f64) -> f64 {
        self.state = self.state + self.alpha * (x - self.state);
        self.state
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RateState {
    pub lp_filters_d: [Lpf1; 3],
    pub torque: Vector3<f64>,
    rate_prev: Vector3<f64>,
    rate_prev_filtered: Vector3<f64>,
    rate_int: Vector3<f64>,
}

impl Default for RateState {
    fn default() -> Self {
        Self {
            lp_filters_d: [Lpf1::new(), Lpf1::new(), Lpf1::new()],
            torque: Vector3::zeros(),
            rate_prev: Vector3::zeros(),
            rate_prev_filtered: Vector3::zeros(),
            rate_int: Vector3::zeros(),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct SaturationFlags {
    pos: Vector3<bool>, // Roll, pitch yaw.
    neg: Vector3<bool>,
}

impl Default for SaturationFlags {
    fn default() -> Self {
        Self {
            pos: Vector3::new(false, false, false),
            neg: Vector3::new(false, false, false),
        }
    }
}

impl RateState {
    pub fn apply_filters(input: Vector3<f64>, filters: &mut [Lpf1; 3]) -> Vector3<f64> {
        Vector3::new(
            filters[0].apply(input.x),
            filters[1].apply(input.y),
            filters[2].apply(input.z),
        )
    }
}

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct RateControl {
    gain_p: Vector3<f64>,
    gain_i: Vector3<f64>,
    gain_d: Vector3<f64>,
    gain_ff: Vector3<f64>,
    lim_int: Vector3<f64>,
}

impl Default for RateControl {
    fn default() -> Self {
        // Reasonable small-quad defaults; tune as needed
        Self {
            gain_p: Vector3::new(0.15, 0.15, 0.10),
            gain_i: Vector3::new(0.05, 0.05, 0.02),
            gain_d: Vector3::new(0.003, 0.003, 0.001),
            gain_ff: Vector3::new(0.0, 0.0, 0.0),
            lim_int: Vector3::new(0.3, 0.3, 0.2),
        }
    }
}

impl RateControl {
    pub fn set_gains(&mut self, p: Vector3<f64>, i: Vector3<f64>, d: Vector3<f64>) {
        self.gain_p = p;
        self.gain_i = i;
        self.gain_d = d;
    }
    pub fn set_dterm_cutoff(&mut self, state: &mut RateState, loop_rate_hz: f64, cutoff_hz: f64) {
        for k in 0..3 {
            state.lp_filters_d[k].set_cutoff(loop_rate_hz, cutoff_hz);
        }
        // reset filtered state to avoid a spike
        state.rate_prev_filtered = state.rate_prev;
        for k in 0..3 {
            state.lp_filters_d[k].reset(state.rate_prev[k]);
        }
    }

    pub fn update(
        &self,
        rate: Vector3<f64>,
        rate_sp: Vector3<f64>,
        mut state: RateState,
        saturation_flags: SaturationFlags,
        dt: f64,
        landed: bool,
    ) -> (Vector3<f64>, RateState) {
        let rate_error = rate_sp - rate;

        // D-term on filtered rates
        let rate_filtered = RateState::apply_filters(rate, &mut state.lp_filters_d);
        let rate_d = if dt > f64::EPSILON {
            (rate_filtered - state.rate_prev_filtered) / dt
        } else {
            Vector3::zeros()
        };

        // PID + FF (PX4 sign convention)
        let torque = self.gain_p.component_mul(&rate_error) + state.rate_int
            - self.gain_d.component_mul(&rate_d)
            + self.gain_ff.component_mul(&rate_sp);

        state.rate_prev = rate;
        state.rate_prev_filtered = rate_filtered;

        if !landed {
            state = self.update_integral(state, rate_error, saturation_flags, dt);
        }
        (torque, state)
    }

    fn update_integral(
        &self,
        mut state: RateState,
        mut rate_error: Vector3<f64>,
        saturation_flags: SaturationFlags,
        dt: f64,
    ) -> RateState {
        const I_REDUCE_REF: f64 = (400f64).to_radians();
        for i in 0..3 {
            if saturation_flags.pos[i] {
                rate_error[i] = rate_error[i].min(0.0);
            }
            if saturation_flags.neg[i] {
                rate_error[i] = rate_error[i].max(0.0);
            }

            let mut i_factor = rate_error[i] / I_REDUCE_REF;
            i_factor = (1.0 - i_factor * i_factor).max(0.0);

            let rate_i = state.rate_int[i] + i_factor * self.gain_i[i] * rate_error[i] * dt;
            if rate_i.is_finite() {
                state.rate_int[i] = rate_i.clamp(-self.lim_int[i], self.lim_int[i]);
            }
        }
        state
    }
}
