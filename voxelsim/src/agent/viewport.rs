use dashmap::DashMap;
use nalgebra::{Matrix4, Perspective3, Point3, Vector3};

use crate::Coord;

use super::*;

/// Camera orientation relative to the drone.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct CameraOrientation {
    pub quat: UnitQuaternion<f64>,
}

impl Default for CameraOrientation {
    fn default() -> Self {
        Self::vertical_tilt(-0.4)
    }
}

impl CameraOrientation {
    pub fn vertical_tilt(tilt: f64) -> Self {
        let fwd_axis = Unit::new_normalize(MoveDir::Forward.dir_vector().unwrap().cast::<f64>());
        let up_axis = Unit::new_normalize(MoveDir::Up.dir_vector().unwrap().cast::<f64>());
        let rot_axis = Unit::new_normalize(fwd_axis.cross(&up_axis));

        let quat = UnitQuaternion::from_axis_angle(&rot_axis, tilt);
        Self { quat }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all, set_all))]
pub struct CameraProjection {
    pub aspect: f64,
    pub fov_vertical: f64,
    pub max_distance: f64,
    pub near_distance: f64,
}

impl CameraProjection {
    pub fn projection_matrix(&self) -> Matrix4<f64> {
        Perspective3::new(
            self.aspect,        // width/height
            self.fov_vertical,  // vertical FOV (radians)
            self.near_distance, // near plane
            self.max_distance,  // far plane
        )
        .to_homogeneous()
    }
}

impl Default for CameraProjection {
    fn default() -> Self {
        Self {
            aspect: 1.4,
            fov_vertical: 0.8,
            max_distance: 50.0,
            near_distance: 0.5,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct CameraView {
    /// World‐space camera position, as (X, Y, Z) with Z = up
    pub camera_pos: Vector3<f64>,
    /// Forward direction, in the world X–Y plane (i.e. camera looks “along” +Y by default)
    pub camera_forward: Unit<Vector3<f64>>,
    /// Up direction → should now be +Z
    pub camera_up: Unit<Vector3<f64>>,
    /// Right direction → +X
    pub camera_right: Unit<Vector3<f64>>,
}

impl CameraView {
    pub fn new(
        camera_pos: Vector3<f64>,
        camera_forward: Unit<Vector3<f64>>,
        camera_up: Unit<Vector3<f64>>,
        camera_right: Unit<Vector3<f64>>,
    ) -> Self {
        Self {
            camera_pos,
            camera_forward,
            camera_up,
            camera_right,
        }
    }

    pub fn base_forward() -> Unit<Vector3<f64>> {
        Unit::new_normalize(MoveDir::Forward.dir_vector().unwrap().cast())
    }

    pub fn base_up() -> Unit<Vector3<f64>> {
        Unit::new_normalize(MoveDir::Up.dir_vector().unwrap().cast())
    }

    pub fn base_right() -> Unit<Vector3<f64>> {
        Unit::new_normalize(MoveDir::Right.dir_vector().unwrap().cast())
    }

    pub fn from_pos_quat(pos: Vector3<f64>, orient: UnitQuaternion<f64>) -> Self {
        let forward = orient * Self::base_forward();
        let up = orient * Self::base_up();
        let right = orient * Self::base_right();

        CameraView {
            camera_pos: pos,
            camera_forward: forward,
            camera_up: up,
            camera_right: right,
        }
    }

    pub fn view_matrix(&self) -> Matrix4<f64> {
        Matrix4::look_at_rh(
            &Point3::from(self.camera_pos), // eye
            &Point3::from(self.camera_pos + self.camera_forward.into_inner()), // target
            &self.camera_up,                // up = +Z
        )
    }

    /// Calculate distance from camera to a coordinate.
    /// We assume Coord is (x, y, z) but now treat z as the vertical axis.
    pub fn distance_to(&self, coord: Coord) -> f64 {
        // Swap Y↔Z so that coord.z is our “height”
        let pos = coord.cast::<f64>();
        (pos - self.camera_pos).norm()
    }
}

impl Default for CameraView {
    fn default() -> Self {
        // NED coord system
        Self {
            camera_pos: Vector3::new(0.0, 0.0, 0.0),
            camera_forward: Self::base_forward(),
            camera_up: Self::base_up(),
            camera_right: Self::base_right(),
        }
    }
}

/// We keep track of uncertainty by generating a linear field with control points.
/// This is updated if we are in a certain area.
/// This basically allows us to know whether to update an area when we look at it or not.
pub struct UncertaintyField<const X: usize, const Y: usize, const Z: usize> {
    pub dense_grid: DashMap<Coord, UncertaintyChunk<X, Y, Z>>,
}

pub struct UncertaintyChunk<const X: usize, const Y: usize, const Z: usize> {
    pub coord: Coord,
    pub control_points: [[[u8; Z]; Y]; X],
}
