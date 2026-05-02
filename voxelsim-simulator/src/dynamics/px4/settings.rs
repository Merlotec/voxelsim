#![cfg(all(feature = "python", feature = "px4"))]
use px4_mc::Px4McSettings;
use pyo3::prelude::*;

#[pyclass]
pub struct Px4SettingsPy {
    #[pyo3(get, set)]
    pub dt: f32,
    #[pyo3(get, set)]
    pub mass: f32,
    #[pyo3(get, set)]
    pub att_p: [f32; 3],
    #[pyo3(get, set)]
    pub att_yaw_weight: f32,
    #[pyo3(get, set)]
    pub att_rate_limit: [f32; 3],
    #[pyo3(get, set)]
    pub pos_p: [f32; 3],
    #[pyo3(get, set)]
    pub vel_p: [f32; 3],
    #[pyo3(get, set)]
    pub vel_i: [f32; 3],
    #[pyo3(get, set)]
    pub vel_d: [f32; 3],
    #[pyo3(get, set)]
    pub vel_lim_xy: f32,
    #[pyo3(get, set)]
    pub vel_up: f32,
    #[pyo3(get, set)]
    pub vel_down: f32,
    #[pyo3(get, set)]
    pub thr_min: f32,
    #[pyo3(get, set)]
    pub thr_max: f32,
    #[pyo3(get, set)]
    pub thr_xy_margin: f32,
    #[pyo3(get, set)]
    pub tilt_max_rad: f32,
    #[pyo3(get, set)]
    pub hover_thrust: f32,
    #[pyo3(get, set)]
    pub rate_p: [f32; 3],
    #[pyo3(get, set)]
    pub rate_i: [f32; 3],
    #[pyo3(get, set)]
    pub rate_d: [f32; 3],
    #[pyo3(get, set)]
    pub rate_int_lim: [f32; 3],
    #[pyo3(get, set)]
    pub decouple_horiz_vert_accel: u8,
    #[pyo3(get, set)]
    pub torque_scale_nm: [f32; 3],
}

#[pymethods]
impl Px4SettingsPy {
    #[new]
    pub fn new() -> Self {
        Self::from_native(Px4McSettings::default())
    }
}

impl Px4SettingsPy {
    pub fn from_native(s: Px4McSettings) -> Self {
        Self {
            dt: s.dt,
            mass: s.mass,
            att_p: s.att_p,
            att_yaw_weight: s.att_yaw_weight,
            att_rate_limit: s.att_rate_limit,
            pos_p: s.pos_p,
            vel_p: s.vel_p,
            vel_i: s.vel_i,
            vel_d: s.vel_d,
            vel_lim_xy: s.vel_lim_xy,
            vel_up: s.vel_up,
            vel_down: s.vel_down,
            thr_min: s.thr_min,
            thr_max: s.thr_max,
            thr_xy_margin: s.thr_xy_margin,
            tilt_max_rad: s.tilt_max_rad,
            hover_thrust: s.hover_thrust,
            rate_p: s.rate_p,
            rate_i: s.rate_i,
            rate_d: s.rate_d,
            rate_int_lim: s.rate_int_lim,
            decouple_horiz_vert_accel: s.decouple_horiz_vert_accel,
            torque_scale_nm: s.torque_scale_nm,
        }
    }

    pub fn to_native(&self) -> Px4McSettings {
        Px4McSettings {
            dt: self.dt,
            mass: self.mass,
            att_p: self.att_p,
            att_yaw_weight: self.att_yaw_weight,
            att_rate_limit: self.att_rate_limit,
            pos_p: self.pos_p,
            vel_p: self.vel_p,
            vel_i: self.vel_i,
            vel_d: self.vel_d,
            vel_lim_xy: self.vel_lim_xy,
            vel_up: self.vel_up,
            vel_down: self.vel_down,
            thr_min: self.thr_min,
            thr_max: self.thr_max,
            thr_xy_margin: self.thr_xy_margin,
            tilt_max_rad: self.tilt_max_rad,
            hover_thrust: self.hover_thrust,
            rate_p: self.rate_p,
            rate_i: self.rate_i,
            rate_d: self.rate_d,
            rate_int_lim: self.rate_int_lim,
            decouple_horiz_vert_accel: self.decouple_horiz_vert_accel,
            torque_scale_nm: self.torque_scale_nm,
        }
    }
}
