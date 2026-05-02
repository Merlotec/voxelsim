use nalgebra::Vector3;
use pyo3::prelude::*;
use voxelsim::chase::{ChaseTarget, FixedLookaheadChaser};
use voxelsim::py::PyCoord;
use voxelsim::{Agent, VoxelGrid};

use crate::dynamics::quad::{QuadDynamics, QuadParams};
use crate::dynamics::{AgentDynamics, EnvState};
use crate::terrain::{TerrainConfig, TerrainGenerator};

#[cfg(feature = "px4")]
use crate::dynamics::px4::Px4Dynamics;

#[pymodule]
pub fn voxelsim_simulator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QuadDynamics>()?;
    m.add_class::<QuadParams>()?;
    // PX4 nested module
    #[cfg(feature = "px4")]
    {
        use crate::dynamics::px4::settings::Px4SettingsPy;

        let px4_mod = PyModule::new(m.py(), "px4")?;
        px4_mod.add_class::<Px4Dynamics>()?;
        px4_mod.add_class::<Px4SettingsPy>()?;
        m.add_submodule(&px4_mod)?;
    }
    m.add_class::<TerrainGenerator>()?;
    m.add_class::<TerrainConfig>()?;
    m.add_class::<EnvState>()?;
    Ok(())
}

#[pymethods]
impl QuadParams {
    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
    }
}

#[pymethods]
impl QuadDynamics {
    #[new]
    pub fn new_py(params: QuadParams) -> Self {
        Self::new(params)
    }
    pub fn update_agent_dynamics_py(
        &mut self,
        agent: &mut Agent,
        env: &EnvState,
        chaser: ChaseTarget,
        delta: f64,
    ) {
        self.update_agent_dynamics(agent, env, chaser, delta);
    }
}

#[cfg(feature = "px4")]
#[pymethods]
impl Px4Dynamics {
    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
    }

    pub fn update_agent_dynamics_py(
        &mut self,
        agent: &mut Agent,
        env: &EnvState,
        chaser: ChaseTarget,
        delta: f64,
    ) {
        self.update_agent_dynamics(agent, env, chaser, delta);
    }

    pub fn apply_px4_settings_py(&mut self, s: &crate::dynamics::px4::settings::Px4SettingsPy) {
        let native = s.to_native();
        self.apply_px4_settings(&native);
    }

    pub fn update_agent_fixed_lookahead_py(
        &mut self,
        agent: &mut Agent,
        env: &EnvState,
        chaser: &FixedLookaheadChaser,
        delta: f64,
        n: usize,
    ) {
        for _ in 0..n {
            use voxelsim::chase::TrajectoryChaser;
            let chase_target = chaser.step_chase(agent, delta);
            self.update_agent_dynamics(agent, env, chase_target, delta);
        }
    }
}

#[pymethods]
impl EnvState {
    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
    }
}

#[pymethods]
impl TerrainConfig {
    #[new]
    pub fn new_py(seed: u32) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
    }
    pub fn set_seed_py(&mut self, seed: u32) {
        self.seed = seed;
    }

    pub fn get_seed_py(&self) -> u32 {
        self.seed
    }
    pub fn set_world_size_py(&mut self, sizev: i32) {
        let s = sizev.max(1); // clamp to ≥1
        self.size = Vector3::new(s, s, s);
    }

    pub fn set_world_dimensions_py(&mut self, x: i32, y: i32, z: i32) {
        self.size = Vector3::new(x.max(1), y.max(1), z.max(1));
    }
}

#[pymethods]
impl TerrainGenerator {
    #[new]
    pub fn new_py() -> Self {
        Self::new()
    }

    pub fn generate_terrain_py(&mut self, cfg: &TerrainConfig) {
        self.generate_terrain(cfg);
    }
    pub fn generate_world_py(&self) -> VoxelGrid {
        self.clone().generate_world()
    }
    // #[staticmethod]
    // pub fn generate_world48_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    //     // Allocate on GPU – uint8 is fine for class labels 0/1/2
    //     let mut vox = Tensor::zeros(&[48, 48, 48], (Kind::Uint8, Device::Cuda(0)));

    //     // ------------------------------------------------------------------
    //     // TODO: fill `vox` with your voxel classes as you wish.
    //     // Example (here: completely filled just for demo):
    //     // vox.fill_(2u8);
    //     // ------------------------------------------------------------------

    //     // Return *ownership* of the tensor to Python without any copy
    //     Ok(vox.into_py(py))
    // }
}
