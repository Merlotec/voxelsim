use dashmap::DashMap;
use nalgebra::Vector3;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyException;
use pyo3::prelude::*; // PyResult, Python, #[pyfunction], …
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;

use numpy::ndarray::Array2;
use numpy::ndarray::ShapeError;

use crate::{
    Cell, Coord,
    agent::{
        Action, ActionIntent, Agent, AgentState, AgentStateUpdate, MoveDir, MoveSequence,
        viewport::{CameraProjection, CameraView},
    },
    chase::{ChaseTarget, FixedLookaheadChaser, TrajectoryChaser},
    env::{DenseSnapshot, VoxelGrid, VoxelSet},
    network::{AsyncRendererClient, RendererClient},
    phase::{PhaseFlow, PhaseGrid, PhaseSolver, PrincipalComponent},
    planner::ActionPlanner,
    planner::astar::AStarActionPlanner,
    ptg::{PtgGraph, PtgObservationScope, PtgSolver},
    uncertainty::UncertaintyWorld,
    viewport::CameraOrientation,
};

#[pymodule]
pub fn voxelsim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VoxelGrid>()?;
    m.add_class::<VoxelSet>()?;
    m.add_class::<UncertaintyWorld>()?;
    m.add_class::<Cell>()?;
    m.add_class::<Agent>()?;
    m.add_class::<MoveDir>()?;
    m.add_class::<Action>()?;
    m.add_class::<ActionIntent>()?;
    m.add_class::<PyAgentStateUpdate>()?;
    m.add_class::<RendererClient>()?;
    m.add_class::<CameraProjection>()?;
    m.add_class::<FixedLookaheadChaser>()?;
    m.add_class::<ChaseTarget>()?;
    m.add_class::<CameraOrientation>()?;
    m.add_class::<DenseSnapshot>()?;
    m.add_class::<AStarActionPlanner>()?;
    m.add_class::<AsyncRendererClient>()?;
    m.add_class::<PhaseFlow>()?;
    m.add_class::<PhaseGrid>()?;
    m.add_class::<PrincipalComponent>()?;
    m.add_class::<PtgGraph>()?;
    m.add_class::<PtgSolver>()?;
    Ok(())
}

pub type PyCoord = [i32; 3];

#[pymethods]
impl VoxelGrid {
    #[new]
    pub fn new_py() -> Self {
        Self::new()
    }

    #[staticmethod]
    pub fn from_dict_py(dict: HashMap<PyCoord, Cell>) -> Self {
        let cells: DashMap<Coord, Cell> = DashMap::with_capacity(dict.len());

        dict.par_iter().for_each(|(coord, value)| {
            cells.insert((*coord).into(), *value);
        });

        Self::from_cells(cells)
    }

    pub fn to_dict_py_tolistpy(&self) -> Vec<((i32, i32, i32), Cell)> {
        let mut items = Vec::new();

        for entry in self.cells().iter() {
            let coord = entry.key();
            let cell = entry.value();

            // Return tuple of coordinates and Cell object
            items.push(((coord.x, coord.y, coord.z), *cell));
        }

        items
    }

    pub fn to_dict_py(&self) -> HashMap<PyCoord, Cell> {
        let mut cells = HashMap::with_capacity(self.cells().len());

        for entry in self.cells().iter() {
            let coord = *entry.key();
            let cell = entry.value();

            // Return tuple of coordinates and Cell object
            cells.insert(coord.into(), *cell);
        }

        cells
    }

    pub fn as_numpy<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray1<f32>>)> {
        let n = self.cells().len();
        let mut coords = Vec::<f32>::with_capacity(n * 3);
        let mut vals = Vec::<f32>::with_capacity(n);

        for e in self.cells().iter() {
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

    pub fn collisions_py(&self, centre: [f64; 3], dims: [f64; 3]) -> Vec<([i32; 3], Cell)> {
        self.collisions(centre.into(), dims.into())
            .into_iter()
            .map(|(coord, cell)| (coord.into(), cell))
            .collect()
    }

    pub fn dense_snapshot_py(&self, centre: [i32; 3], dims: [i32; 3]) -> DenseSnapshot {
        self.dense_snapshot(centre.into(), dims.into())
    }

    pub fn clone_py(&self) -> Self {
        self.clone()
    }

    pub fn add_target_cell(&mut self, coord: [i32; 3]) {
        self.cells().insert(coord.into(), Cell::TARGET);
    }

    pub fn difference_set_from_py(&self, other: &Self) -> VoxelSet {
        self.difference_set_from(other)
    }

    pub fn populate_with_agents_py(&mut self, agents: Vec<Agent>) {
        self.populate_with_agents(&agents);
    }
}

#[pymethods]
impl UncertaintyWorld {
    #[new]
    pub fn new_py(world_origin: [f64; 3], node_size: f64) -> Self {
        Self::new(world_origin.into(), node_size.into())
    }

    #[staticmethod]
    pub fn default_py() -> Self {
        Self::new(Vector3::zeros(), 10.0)
    }

    pub fn update_view_frustum_py(
        &mut self,
        cam_pos_world: [f64; 3],
        cam_orientation: CameraOrientation,
        cam_proj: CameraProjection,
        max_dist: f64,
    ) {
        self.update_view_frustum(
            cam_pos_world.into(),
            cam_orientation.quat,
            cam_proj.fov_vertical * 0.5,
            max_dist,
        )
    }
}

#[pymethods]
impl Cell {
    #[staticmethod]
    pub fn filled() -> Self {
        Self::FILLED
    }
    #[staticmethod]
    pub fn sparse() -> Self {
        Self::SPARSE
    }
    #[staticmethod]
    pub fn interest() -> Self {
        Self::INTEREST
    }
    pub fn is_filled_py(&self) -> bool {
        self.contains(Cell::FILLED)
    }
    pub fn is_sparse_py(&self) -> bool {
        self.contains(Cell::SPARSE)
    }
    pub fn is_interest_py(&self) -> bool {
        self.contains(Cell::INTEREST)
    }
    pub fn bits_py(&self) -> u32 {
        self.bits()
    }
}

#[pymethods]
impl Agent {
    #[new]
    pub fn new_py(id: usize) -> Self {
        Self::new(id)
    }

    /// Will overwrite any existing action and just perform this one intent.
    pub fn perform_oneshot_py(&mut self, intent: ActionIntent) -> PyResult<()> {
        self.perform(intent)
            .map_err(|e| PyException::new_err(format!("Could not perform intent: {}", e)))
    }

    pub fn push_back_intent_py(&mut self, intent: ActionIntent) -> PyResult<()> {
        if let AgentState::Action(action) = &mut self.state {
            action
                .push_back_intent(intent)
                .map_err(|e| PyException::new_err(format!("Could not push back intent: {}", e)))
        } else {
            self.state =
                AgentState::Action(Action::new_oneshot(intent, self.get_coord()).map_err(|e| {
                    PyException::new_err(format!("Could not perform intent: {}", e))
                })?);
            Ok(())
        }
    }

    pub fn update_state_py(&mut self, update: PyAgentStateUpdate) -> PyResult<()> {
        self.update_state(update.inner)
            .map_err(|e| PyException::new_err(format!("Could not update agent state: {}", e)))
    }

    pub fn camera_view_py(&self, orientation: &CameraOrientation) -> CameraView {
        self.camera_view(orientation)
    }

    pub fn get_action_py(&self) -> Option<Action> {
        self.get_action().cloned()
    }

    pub fn set_pos(&mut self, pos: [f64; 3]) {
        self.pos = Vector3::from(pos)
    }

    pub fn get_pos(&self) -> [f64; 3] {
        self.pos.into()
    }

    pub fn get_coord_py(&self) -> [i32; 3] {
        self.get_coord().into()
    }

    pub fn set_hold_py(&mut self, coord: PyCoord, yaw: f64) {
        self.set_hold(coord.into(), yaw);
    }
}

#[pymethods]
impl Action {
    /// Add a move command to the action sequence
    // pub fn add_command(&mut self, command: MoveCommand) -> bool {
    //     if self.cmd_sequence.len() < crate::agent::MAX_ACTIONS {
    //         self.cmd_sequence.push(command);
    //         true
    //     } else {
    //         false
    //     }
    // }

    /// Get the origin position as a tuple
    pub fn get_origin(&self) -> (i32, i32, i32) {
        (self.origin.x, self.origin.y, self.origin.z)
    }

    /// Set the origin position
    pub fn set_origin(&mut self, x: i32, y: i32, z: i32) {
        self.origin = Vector3::new(x, y, z);
    }

    pub fn intent_count(&self) -> usize {
        self.intent_queue.len()
    }

    pub fn get_intent_queue(&self) -> Vec<ActionIntent> {
        self.intent_queue.clone().into()
    }

    /// Clones the action under the hood.
    pub fn appending_intent_py(&self, intent: ActionIntent) -> PyResult<Self> {
        self.clone()
            .appending_intent(intent)
            .map_err(|e| PyException::new_err(format!("Could not append intent to action: {}", e)))
    }
}

#[pymethods]
impl ActionIntent {
    #[new]
    pub fn new_py(urgency: f64, yaw: f64, move_sequence: MoveSequence) -> Self {
        Self::new(urgency, yaw, move_sequence)
    }

    pub fn get_move_sequence(&self) -> MoveSequence {
        self.move_sequence.clone()
    }

    pub fn len(&self) -> usize {
        self.move_sequence.len()
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct PyAgentStateUpdate {
    pub inner: AgentStateUpdate,
}

impl Into<AgentStateUpdate> for PyAgentStateUpdate {
    fn into(self) -> AgentStateUpdate {
        self.inner
    }
}

#[pymethods]
impl PyAgentStateUpdate {
    #[staticmethod]
    pub fn push(intent: ActionIntent) -> Self {
        Self {
            inner: AgentStateUpdate::Push(intent),
        }
    }

    #[staticmethod]
    pub fn replace(intents: Vec<ActionIntent>) -> Self {
        Self {
            inner: AgentStateUpdate::Replace(intents.into()),
        }
    }

    #[staticmethod]
    pub fn truncate(intents: Vec<ActionIntent>) -> Self {
        Self {
            inner: AgentStateUpdate::Truncate(intents.into()),
        }
    }

    #[staticmethod]
    pub fn hold(coord: Option<[i32; 3]>, yaw: Option<f64>) -> Self {
        Self {
            inner: AgentStateUpdate::Hold {
                coord: coord.map(|x| x.into()),
                yaw,
            },
        }
    }
}

#[pymethods]
impl MoveDir {
    #[staticmethod]
    pub fn from_code_py(code: i32) -> PyResult<Self> {
        Self::try_from(code)
            .map_err(|e| PyException::new_err(format!("Failed to parse code: {}", e)))
    }

    /// Create a None/stationary direction
    #[classmethod]
    pub fn none(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::None
    }

    /// Create an Up direction
    #[classmethod]
    pub fn up(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Up
    }

    /// Create a Down direction
    #[classmethod]
    pub fn down(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Down
    }

    /// Create a Left direction
    #[classmethod]
    pub fn left(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Left
    }

    /// Create a Right direction
    #[classmethod]
    pub fn right(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Right
    }

    /// Create a Forward direction
    #[classmethod]
    pub fn forward(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Forward
    }

    /// Create a Back direction
    #[classmethod]
    pub fn back(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Back
    }

    /// Create an Undecided direction
    #[classmethod]
    pub fn undecided(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Undecided
    }

    /// Get the direction as a string representation
    pub fn to_string(&self) -> String {
        match self {
            MoveDir::None => "None".to_string(),
            MoveDir::Up => "Up".to_string(),
            MoveDir::Down => "Down".to_string(),
            MoveDir::Left => "Left".to_string(),
            MoveDir::Right => "Right".to_string(),
            MoveDir::Forward => "Forward".to_string(),
            MoveDir::Back => "Back".to_string(),
            MoveDir::Undecided => "Undecided".to_string(),
        }
    }

    /// Get the direction vector as a tuple (x, y, z) or None if undecided
    pub fn get_direction_vector(&self) -> Option<(i32, i32, i32)> {
        self.dir_vector().map(|v| (v.x, v.y, v.z))
    }

    /// Check if this direction represents movement
    pub fn is_movement(&self) -> bool {
        !matches!(self, MoveDir::None | MoveDir::Undecided)
    }

    /// Check if this is a vertical movement
    pub fn is_vertical(&self) -> bool {
        matches!(self, MoveDir::Up | MoveDir::Down)
    }

    /// Check if this is a horizontal movement
    pub fn is_horizontal(&self) -> bool {
        matches!(
            self,
            MoveDir::Left | MoveDir::Right | MoveDir::Forward | MoveDir::Back
        )
    }

    /// Get the opposite direction
    pub fn opposite(&self) -> Self {
        match self {
            MoveDir::Up => MoveDir::Down,
            MoveDir::Down => MoveDir::Up,
            MoveDir::Left => MoveDir::Right,
            MoveDir::Right => MoveDir::Left,
            MoveDir::Forward => MoveDir::Back,
            MoveDir::Back => MoveDir::Forward,
            MoveDir::None => MoveDir::None,
            MoveDir::Undecided => MoveDir::Undecided,
        }
    }
}

#[pymethods]
impl RendererClient {
    #[new]
    fn new_py(
        host: String,
        world_port: u16,
        agent_port: u16,
        pov_start_port: u16,
        pov_agent_start_port: u16,
        pov_count: u16,
    ) -> PyResult<Self> {
        Self::new(
            host.as_str(),
            world_port,
            agent_port,
            pov_start_port,
            pov_agent_start_port,
            pov_count,
        )
        .map_err(|e| PyException::new_err(format!("Failed to create render client: {}", e)))
    }

    #[staticmethod]
    pub fn default_localhost_py(pov_count: u16) -> PyResult<Self> {
        Self::new("127.0.0.1", 8080, 8081, 8090, 9090, pov_count)
            .map_err(|e| PyException::new_err(format!("Failed to create render client: {}", e)))
    }

    pub fn send_agents_py(&mut self, agents: HashMap<usize, Agent>) -> PyResult<()> {
        self.send_agents(&agents)
            .map_err(|e| PyException::new_err(format!("Failed to send agents: {}", e)))
    }
    pub fn send_world_py(&mut self, world: &VoxelGrid) -> PyResult<()> {
        self.send_world(world)
            .map_err(|e| PyException::new_err(format!("Failed to send world: {}", e)))
    }
}

#[pymethods]
impl AsyncRendererClient {
    #[new]
    fn new_py(
        host: String,
        world_port: u16,
        agent_port: u16,
        pov_start_port: u16,
        pov_agent_start_port: u16,
        pov_count: u16,
    ) -> PyResult<Self> {
        Ok(Self::new(RendererClient::new_py(
            host,
            world_port,
            agent_port,
            pov_start_port,
            pov_agent_start_port,
            pov_count,
        )?))
    }

    #[staticmethod]
    pub fn default_localhost_py(pov_count: u16) -> PyResult<Self> {
        Ok(Self::new(RendererClient::default_localhost_py(pov_count)?))
    }

    pub fn send_agents_py(&self, agents: HashMap<usize, Agent>) {
        self.send_agents(agents)
    }

    pub fn send_world_py(&self, world: VoxelGrid) {
        self.send_world(world)
    }
}

#[pymethods]
impl CameraProjection {
    #[new]
    pub fn new(aspect: f64, fov_vertical: f64, max_distance: f64, near_distance: f64) -> Self {
        Self {
            aspect,
            fov_vertical,
            max_distance,
            near_distance,
        }
    }
    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
    }
}

#[pymethods]
impl FixedLookaheadChaser {
    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
    }

    pub fn step_chase_py(&self, agent: &Agent, delta: f64) -> ChaseTarget {
        self.step_chase(agent, delta)
    }
}

#[pymethods]
impl CameraOrientation {
    #[staticmethod]
    pub fn vertical_tilt_py(tilt: f64) -> Self {
        Self::vertical_tilt(tilt)
    }
}

#[pymethods]
impl DenseSnapshot {
    pub fn data_py(&self) -> Vec<i32> {
        // Safety:
        // Since the underlying type of the Cell is 32 bit, should be fine.
        let data = self.data().to_vec();
        let transmuted_data = unsafe { std::mem::transmute_copy(&data) };
        std::mem::forget(data);
        transmuted_data
    }
}

#[pymethods]
impl AStarActionPlanner {
    #[new]
    pub fn new_py(padding: i32) -> Self {
        Self::new(padding)
    }

    pub fn plan_action_py(
        &self,
        world: &VoxelGrid,
        origin: PyCoord,
        dst: PyCoord,
        urgency: f64,
        yaw: f64,
    ) -> PyResult<ActionIntent> {
        self.plan_action(world, origin.into(), dst.into(), urgency, yaw)
            .map_err(|e| PyException::new_err(format!("Failed to create action intent: {}", e)))
    }
}

#[pymethods]
impl PhaseFlow {
    #[new]
    pub fn from_frames_py(frames: Vec<(VoxelSet, f64)>) -> Self {
        Self::from_frames(frames)
    }

    pub fn build_grid_py(&self, t: f64) -> PhaseGrid {
        self.build_grid(t)
    }
}

#[pymethods]
impl PhaseGrid {
    #[new]
    pub fn new_py() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PtgGraph {
    #[new]
    pub fn new_py() -> Self {
        Self::new()
    }

    pub fn add_observation_set_py(
        &mut self,
        obs: Vec<(f64, f64, f64)>,
        t: f64,
        min_t: f64,
        add_thresh: f64,
        sim_thresh: f64,
    ) {
        let obs: Vec<Vector3<f64>> = obs
            .into_iter()
            .map(|o| Vector3::new(o.0, o.1, o.2))
            .collect();

        self.add_observation_set(
            &obs,
            t,
            PtgObservationScope::MinTime(min_t),
            add_thresh,
            sim_thresh,
        );
    }

    pub fn generate_grid_py(
        &self,
        t_start: f64,
        t_end: f64,
        max_dist_thresh: f64,
        min_likelihood_thresh: f64,
    ) -> PhaseGrid {
        self.generate_grid(t_start..t_end, max_dist_thresh, min_likelihood_thresh)
    }
}

#[pymethods]
impl PtgSolver {
    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
    }

    pub fn latest_timestamp_py(&self) -> Option<f64> {
        self.latest_timestamp()
    }

    pub fn add_phase_frame_py(&mut self, agent: &Agent, virtual_world: &VoxelGrid, t: f64) {
        self.add_phase_frame(agent, virtual_world, t)
    }

    pub fn gen_phase_grid_py(&self, agent: &Agent, t_start: f64, t_end: f64) -> PhaseGrid {
        self.gen_phase_grid(agent, t_start..t_end)
    }
}
