use crate::AgentVisionRenderer;
use crate::FilterWorld;
use crate::WorldChangeset;
use crate::rasterizer::noise::NoiseParams;
use voxelsim::uncertainty::UncertaintyWorld;

use numpy::ndarray::Array2;
use numpy::ndarray::ShapeError;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::ops::Deref;
use voxelsim::Cell;
use voxelsim::RendererClient;
use voxelsim::env::{DenseSnapshot, VoxelGrid, VoxelSet};
use voxelsim::network::AsyncRendererClient;
use voxelsim::phase::PhaseGrid;
use voxelsim::viewport::CameraOrientation;
use voxelsim::viewport::{CameraProjection, CameraView};

#[pymodule]
pub fn voxelsim_compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FilterWorld>()?;
    m.add_class::<WorldChangeset>()?;
    m.add_class::<AgentVisionRenderer>()?;
    m.add_class::<NoiseParams>()?;
    Ok(())
}

#[pymethods]
impl FilterWorld {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_updating_py(&self, timestamp: f64) -> bool {
        self.is_updating(timestamp)
    }

    pub fn send_pov_py(
        &self,
        client: &mut RendererClient,
        stream_idx: usize,
        agent_id: usize,
        proj: CameraProjection,
        orientation: CameraOrientation,
        dynamic_world: &PhaseGrid,
    ) -> PyResult<()> {
        self.send_pov(
            client,
            stream_idx,
            agent_id,
            proj,
            orientation,
            dynamic_world,
        )
        .map_err(|e| PyException::new_err(format!("Could not send pov: {}", e)))
    }

    pub fn send_pov_async_py(
        &self,
        client: &AsyncRendererClient,
        stream_idx: usize,
        agent_id: usize,
        proj: CameraProjection,
        orientation: CameraOrientation,
        dynamic_world: PhaseGrid,
    ) {
        self.send_pov_async(
            client,
            stream_idx,
            agent_id,
            proj,
            orientation,
            dynamic_world,
        )
    }

    pub fn dense_snapshot_py(&self, centre: [i32; 3], half_dims: [i32; 3]) -> DenseSnapshot {
        self.dense_snapshot(centre.into(), half_dims.into())
    }

    pub fn as_numpy<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray1<f32>>)> {
        let world = self.world.lock().unwrap();
        let n = world.cells().len();
        let mut coords = Vec::<f32>::with_capacity(n * 3);
        let mut vals = Vec::<f32>::with_capacity(n);

        for e in world.cells().iter() {
            let (c, cell) = (*e.key(), *e.value());
            coords.extend_from_slice(&[c.x as f32, c.y as f32, c.z as f32]);
            vals.push(if cell.contains(Cell::FILLED) {
                1.0
            } else if cell.contains(Cell::SPARSE) {
                0.5
            } else {
                0.0
            });
        }

        // -------- coords: ndarray -> PyArray, then Bound -> Py -------------
        let coords_arr: Py<PyArray2<f32>> = Array2::from_shape_vec((n, 3), coords)
            .map_err(|e: ShapeError| PyException::new_err(e.to_string()))? // <-- map
            .into_pyarray(py)
            .into();
        // Py<PyArray2<…>>

        // -------- values: PyArray1 returned directly; just convert Bound -> Py
        let vals_arr: Py<PyArray1<f32>> = PyArray1::from_vec(py, vals).into();

        Ok((coords_arr, vals_arr))
    }

    pub fn timestamp_py(&self) -> Option<f64> {
        *self.timestamp.lock().unwrap()
    }

    pub fn clone_world_py(&self) -> VoxelGrid {
        self.clone_world()
    }
}

#[pymethods]
impl AgentVisionRenderer {
    #[new]
    pub fn init_py(world: &VoxelGrid, view_size: [u32; 2], noise: NoiseParams) -> Self {
        Self::init(world, view_size.into(), noise)
    }
    pub fn update_filter_world_py(
        &self,
        py: Python<'_>,
        camera: CameraView,
        proj: CameraProjection,
        filter_world: Py<FilterWorld>,
        dynamic_world: VoxelGrid,
        timestamp: f64,
        callback: PyObject,
    ) {
        let fw_clone = filter_world.borrow(py).deref().clone();
        py.allow_threads(move || {
            self.update_filter_world(
                camera.view_matrix(),
                proj.projection_matrix(),
                fw_clone,
                dynamic_world,
                timestamp,
                move |fw, ts| {
                    Python::with_gil(|py| {
                        let _ = callback.call(py, (filter_world.clone_ref(py), ts), None);
                    });
                },
            )
        });
    }

    pub fn update_filter_world_with_uncertainty_py(
        &self,
        py: Python<'_>,
        camera: CameraView,
        proj: CameraProjection,
        filter_world: Py<FilterWorld>,
        dynamic_world: VoxelGrid,
        uncertainty_world: UncertaintyWorld,
        timestamp: f64,
        callback: PyObject,
    ) {
        let fw_clone = filter_world.borrow(py).deref().clone();
        py.allow_threads(move || {
            self.update_filter_world_with_uncertainty(
                camera.view_matrix(),
                proj.projection_matrix(),
                fw_clone,
                dynamic_world,
                uncertainty_world,
                timestamp,
                move |fw, ts| {
                    Python::with_gil(|py| {
                        let _ = callback.call(py, (filter_world.clone_ref(py), ts), None);
                    });
                },
            )
        });
    }
    pub fn render_changeset_py(
        &self,
        py: Python<'_>,
        camera: CameraView,
        proj: CameraProjection,
        filter_world: Py<FilterWorld>,
        dynamic_world: VoxelGrid,
        timestamp: f64,
        callback: PyObject,
    ) {
        let fw_clone = filter_world.borrow(py).deref().clone();
        py.allow_threads(move || {
            self.render_changeset(
                camera.view_matrix(),
                proj.projection_matrix(),
                fw_clone,
                dynamic_world,
                timestamp,
                move |changeset| {
                    Python::with_gil(|py| {
                        let _ = callback.call1(py, (changeset,));
                    });
                },
            )
        });
    }
}

#[pymethods]
impl NoiseParams {
    #[staticmethod]
    pub fn default_with_seed_py(seed: [f32; 3]) -> Self {
        Self::default_with_seed(seed.into())
    }

    #[staticmethod]
    pub fn none_py() -> Self {
        Self::none()
    }
}

#[pymethods]
impl WorldChangeset {
    pub fn timestamp_py(&self) -> f64 {
        self.timestamp()
    }

    pub fn update_filter_world_py(&self, filter_world: &FilterWorld) {
        self.update_filter_world(filter_world)
    }

    pub fn update_filter_world_with_difference_py(&self, filter_world: &FilterWorld) -> VoxelSet {
        self.update_filter_world_with_difference(filter_world)
    }

    /// Python binding for gated updates with uncertainty.
    pub fn update_filter_world_uncertainty_py(
        &self,
        filter_world: &FilterWorld,
        unc_world: &UncertaintyWorld,
    ) {
        self.update_filter_world_with_uncertainty(filter_world, unc_world)
    }

    pub fn camera_pos_py(&self) -> [f64; 3] {
        [self.cam_pos.x, self.cam_pos.y, self.cam_pos.z]
    }

    pub fn camera_dir_py(&self) -> [f64; 3] {
        [self.cam_dir.x, self.cam_dir.y, self.cam_dir.z]
    }
}
