use bitflags::bitflags;
use dashmap::{DashMap, DashSet};
use nalgebra::Vector3;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::Agent;

bitflags! {
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
    pub struct Cell: u32 {
        const DYNAMIC_OCCUPIED  = 0b0000_0001;
        const AGENT_OCCUPIED    = 0b0000_0010;
        const TRAJECTORY        = 0b0000_0100;
        const FILLED            = 0b0000_1000;
        const SPARSE            = 0b0001_0000;
        const GROUND            = 0b0010_0000;
        const TARGET            = 0b0100_0000;
        const INTEREST          = 0b1000_0000;
    }

}

// #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
// #[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
// pub struct Cell {
//     flags: CellFlags,
//     id_interest: u8,
//     id_dyn1: u8,
//     id_dyn2: u8,
// }

// impl Cell {
//     pub fn new(flags: CellFlags) -> Self {
//         Self {
//             flags,
//             id_interest: 0,
//             id_dyn1: 0,
//             id_dyn2: 0,
//         }
//     }
// }

pub type Coord = Vector3<i32>;

pub type GridShell = [Vector3<i32>; 26];

pub(crate) fn adjacent_coords(coord: Vector3<i32>) -> GridShell {
    let mut coords = [Vector3::zeros(); 26];
    let mut t = 0;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                if i != 1 || j != 1 || k != 1 {
                    coords[t] = coord + Vector3::new(i - 1, j - 1, k - 1);
                    t += 1;
                }
            }
        }
    }
    coords
}

/// Returns `true` when the axis-aligned box centred at `pos` with full
/// extents `dims` touches or overlaps the unit cube whose centre is the
/// integer coordinate `coord`.
///
/// Touching faces/edges counts as an intersection; change the comparisons
/// from `>=` to `>` if you need strictly positive volume overlap.
pub fn intersects(coord: Vector3<i32>, pos: Vector3<f64>, dims: Vector3<f64>) -> bool {
    // --- Unit-cube bounds (½-extent is 0.5 on every axis) ------------------
    let cube_min = Vector3::new(
        coord.x as f64 - 0.5,
        coord.y as f64 - 0.5,
        coord.z as f64 - 0.5,
    );
    let cube_max = Vector3::new(
        coord.x as f64 + 0.5,
        coord.y as f64 + 0.5,
        coord.z as f64 + 0.5,
    );

    // --- Object bounds -----------------------------------------------------
    let half = dims * 0.5; // component-wise scalar multiply
    let obj_min = pos - half;
    let obj_max = pos + half;

    // --- AABB overlap test -------------------------------------------------
    obj_max.x >= cube_min.x
        && cube_max.x >= obj_min.x
        && obj_max.y >= cube_min.y
        && cube_max.y >= obj_min.y
        && obj_max.z >= cube_min.z
        && cube_max.z >= obj_min.z
}

pub(crate) fn within_bounds<N: PartialOrd>(v: Vector3<N>, b: Vector3<N>) -> bool {
    v.iter().zip(b.iter()).all(|(vi, bi)| vi <= bi)
}

// Dynamic height voxel grid.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct VoxelGrid {
    cells: DashMap<Coord, Cell>,
}

/// Sparse voxel grid that does not store any cell.
/// Can be used to just reference a set of cells.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct VoxelSet {
    cells: DashSet<Coord>,
}

impl VoxelGrid {
    pub fn new() -> Self {
        Self {
            cells: DashMap::new(),
        }
    }

    pub fn from_cells(cells: DashMap<Coord, Cell>) -> Self {
        Self { cells }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cells: DashMap::with_capacity(capacity),
        }
    }

    pub fn cull(&mut self, centre: Vector3<i32>, bounds: Vector3<i32>) {
        self.cells
            .retain(|k, _| within_bounds(Vector3::from(*k) - centre, bounds));
    }

    pub fn cells(&self) -> &DashMap<Coord, Cell> {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut DashMap<Coord, Cell> {
        &mut self.cells
    }

    pub fn insert(&mut self, coord: Coord, cell: Cell) {
        self.cells.insert(coord, cell);
    }

    /// Adds the given bitflags onto the existing flags with a bitwise OR.
    pub fn set(&mut self, coord: Coord, cell: Cell) {
        if let Some(mut c) = self.cells.get_mut(&coord) {
            *c = *c | cell;
        } else {
            self.cells.insert(coord, cell);
        }
    }
    pub fn remove(&mut self, coord: &Coord) -> Option<Cell> {
        self.cells.remove(coord).map(|x| x.1)
    }

    pub fn populate_with_agents(&mut self, agents: &[Agent]) {
        for agent in agents {
            let coord_f = agent.pos.map(|v| v.round());
            if let Some(coord) = coord_f.try_cast() {
                self.set(coord, Cell::AGENT_OCCUPIED);
            }
        }
    }

    /// Returns a the list of cells if an object with the given centre coordinate and dimensions collides with any
    /// cells.
    pub fn collisions(&self, centre: Vector3<f64>, dims: Vector3<f64>) -> Vec<(Coord, Cell)> {
        let mut collisions = Vec::new();
        // We only need to check the cubes around.
        assert!(dims.x < 1.0 && dims.y < 1.0 && dims.z < 1.0);
        for cell_coord in adjacent_coords(centre.try_cast::<i32>().unwrap()) {
            if let Some(cell) = self.cells().get(&cell_coord) {
                if intersects(cell_coord, centre, dims) {
                    collisions.push((cell_coord, *cell));
                }
            }
        }
        collisions
    }

    pub fn dense_snapshot(&self, centre: Coord, half_dims: Vector3<i32>) -> DenseSnapshot {
        // We encode in the distance first approach.
        let min = centre - half_dims;
        let max = centre + half_dims;

        let dim = max - min;
        let len = dim.x * dim.y * dim.z;
        let mut dense = Vec::with_capacity(len as usize);
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    let c = self
                        .cells()
                        .get(&Vector3::new(x, y, z))
                        .map(|x| *x)
                        .unwrap_or(Cell::empty());
                    dense.push(c);
                }
            }
        }

        DenseSnapshot {
            data: dense,
            centre,
            half_dims,
        }
    }

    /// If we have a lot of cells, this is a faster approach.
    pub fn sparse_snapshot_lookup(&self, centre: Coord, half_dims: Vector3<i32>) -> VoxelGrid {
        let new_cells = DashMap::new();

        let min = centre - half_dims;
        let max = centre + half_dims;

        let dim = max - min;
        let _len = dim.x * dim.y * dim.z;
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    let coord = Vector3::new(x, y, z);
                    if let Some(c) = self.cells().get(&coord) {
                        new_cells.insert(coord, *c);
                    }
                }
            }
        }

        VoxelGrid::from_cells(new_cells)
    }

    pub fn sparse_snapshot_iterating(&self, centre: Coord, half_dims: Vector3<i32>) -> VoxelGrid {
        let new_cells = DashMap::new();

        self.cells().par_iter().for_each(|c| {
            let dist = c.key() - centre;
            if dist.x.abs() <= half_dims.x
                && dist.y.abs() <= half_dims.y
                && dist.z.abs() <= half_dims.z
            {
                new_cells.insert(*c.key(), *c.value());
            }
        });

        VoxelGrid::from_cells(new_cells)
    }

    /// Generates a difference map between self and other (i.e. what was added in self that was not in other).
    pub fn difference_set_from(&self, other: &Self) -> VoxelSet {
        let diff_map: DashSet<Coord> = DashSet::new();

        self.cells().par_iter().for_each(|c| {
            if !other.cells().contains_key(c.key()) {
                diff_map.insert(*c.key());
            }
        });

        VoxelSet { cells: diff_map }
    }
}

impl VoxelSet {
    pub fn new() -> Self {
        Self {
            cells: DashSet::new(),
        }
    }

    pub fn from_cells(cells: DashSet<Coord>) -> Self {
        Self { cells }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cells: DashSet::with_capacity(capacity),
        }
    }

    pub fn cull(&mut self, centre: Vector3<i32>, bounds: Vector3<i32>) {
        self.cells
            .retain(|k| within_bounds(Vector3::from(*k) - centre, bounds));
    }

    pub fn cells(&self) -> &DashSet<Coord> {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut DashSet<Coord> {
        &mut self.cells
    }

    pub fn set(&mut self, coord: Coord) {
        self.cells.insert(coord);
    }

    pub fn remove(&mut self, coord: &Coord) -> Option<Coord> {
        self.cells.remove(coord)
    }

    pub fn coords(&self) -> Vec<Coord> {
        self.cells().iter().map(|c| *c).collect()
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct DenseSnapshot {
    centre: Coord,
    half_dims: Vector3<i32>,

    data: Vec<Cell>,
}

impl DenseSnapshot {
    pub fn centre(&self) -> Coord {
        self.centre
    }

    pub fn half_dims(&self) -> Vector3<i32> {
        self.half_dims
    }

    pub fn data(&self) -> &[Cell] {
        &self.data
    }
}
