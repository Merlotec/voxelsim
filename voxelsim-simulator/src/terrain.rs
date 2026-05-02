use dashmap::DashMap;
use nalgebra::Vector3;
use noise::{NoiseFn, Perlin};
use rand::{Rng, SeedableRng, rngs::StdRng};
use voxelsim::{
    Coord,
    env::{Cell, VoxelGrid},
};

// Created structure for holding terrain configuration parameters.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct TerrainConfig {
    pub size: Vector3<i32>,
    pub height_scale: f64,
    pub flat_band: f64,
    pub max_terrain_height: i32,
    pub forest_scale: f64,
    pub max_forest_height: i32,
    pub seed: u32,
    pub base_thickness: i32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            size: Vector3::new(100, 30, 100),
            height_scale: 100.0,
            flat_band: 0.1,
            max_terrain_height: 32,
            forest_scale: 30.0,
            max_forest_height: 20,
            seed: 100,
            base_thickness: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TreeType {
    Oak,    // Round canopy, thick trunk, spreading branches
    Pine,   // Conical shape, straight trunk, layered branches
    Birch,  // Tall and thin, small round canopy
    Willow, // Drooping branches, wide canopy
}

#[derive(Debug, Clone)]
pub struct TreeParams {
    pub tree_type: TreeType,
    pub height: i32,
    pub trunk_radius: i32,
    pub canopy_radius: i32,
    pub canopy_height: i32,
    pub trunk_taper: f32,   // How much trunk narrows (0-1)
    pub branch_layers: i32, // Number of branch layers
    pub branch_length: i32, // Average branch length
    pub leaf_density: f32,  // Density of leaves (0-1)
}

impl TreeParams {
    fn for_type(tree_type: TreeType, base_height: i32, rng: &mut StdRng) -> Self {
        // Add more random variation for grove diversity
        let height_var: f32 = 0.6 + (rng.random::<f32>() * 0.8); // 0.6 to 1.4
        let base_height = base_height + 2; // Increase base height for taller trees
        let height = (base_height as f32 * height_var) as i32;

        let params = match tree_type {
            TreeType::Oak => TreeParams {
                tree_type,
                height,
                trunk_radius: 2,
                canopy_radius: (height as f32 * 0.3) as i32,
                canopy_height: (height as f32 * 0.7) as i32,
                trunk_taper: 0.3,
                branch_layers: height / 3,
                branch_length: (height as f32 * 0.4) as i32,
                leaf_density: 0.7,
            },
            TreeType::Pine => TreeParams {
                tree_type,
                height,
                trunk_radius: 1,
                canopy_radius: (height as f32 * 0.2) as i32,
                canopy_height: (height as f32 * 0.8) as i32,
                trunk_taper: 0.5,
                branch_layers: height / 2,
                branch_length: ((height as f32 * 0.3) as i32).max(2),
                leaf_density: 0.85,
            },
            TreeType::Birch => TreeParams {
                tree_type,
                height: (height as f32 * 1.2) as i32, // Birches are taller
                trunk_radius: 1,
                canopy_radius: (height as f32 * 0.1) as i32, // Narrower canopy
                canopy_height: (height as f32 * 0.3) as i32, // Slightly smaller canopy
                trunk_taper: 0.2,
                branch_layers: height / 4,
                branch_length: ((height as f32 * 0.3) as i32).max(3), // Shorter branches
                leaf_density: 0.6,
            },
            TreeType::Willow => TreeParams {
                tree_type,
                height: (height as f32 * 0.8) as i32, // Willows are shorter
                trunk_radius: 2,
                canopy_radius: (height as f32 * 0.4) as i32,
                canopy_height: (height as f32 * 0.6) as i32,
                trunk_taper: 0.2,
                branch_layers: height / 3,
                branch_length: (height as f32 * 0.5) as i32,
                leaf_density: 0.5,
            },
        };
        params
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct TerrainGenerator {
    cells: DashMap<Coord, Cell>,
}

impl TerrainGenerator {
    pub fn new() -> Self {
        Self {
            cells: DashMap::new(),
        }
    }
    pub fn cells_mut(&mut self) -> &mut DashMap<Coord, Cell> {
        &mut self.cells
    }

    pub fn cells(&mut self) -> &DashMap<Coord, Cell> {
        &self.cells
    }

    // Performs a coordinate switch to make z vertical (NED coords) which is very important!
    pub fn generate_world(self) -> VoxelGrid {
        VoxelGrid::from_cells(
            self.cells
                .into_iter()
                .map(|(k, v)| (Vector3::new(k.x, k.z, -k.y), v))
                .collect(),
        )
    }
}

impl TerrainGenerator {
    // ---------------------------------------------------------------------
    // 1. BUILD GROUND ------------------------------------------------------
    // ---------------------------------------------------------------------
    /// Fills all `GROUND|FILLED` voxels up to the height returned by `height_fn`.
    ///
    /// `height_fn` must return a normalized value in **[0,1]** for a given `(x,z)`.
    ///
    pub fn build_ground(
        &mut self,
        cfg: &TerrainConfig,
        height_fn: &mut dyn FnMut(i32, i32) -> f64,
    ) -> Vec<Vec<i32>> {
        // Return height map
        let [max_x, _max_y, max_z] = cfg.size.into();

        // --- 1. collect surface heights ---------------------------------
        let mut height_map = vec![vec![0i32; (max_z + 1) as usize]; (max_x + 1) as usize];
        for x in 0..=max_x {
            for z in 0..=max_z {
                let h = height_fn(x, z).clamp(0.0, 1.0);
                height_map[x as usize][z as usize] =
                    (h * cfg.max_terrain_height as f64).round() as i32;
            }
        }

        // --- 2. fill ground with proper connectivity ---------------------
        for x in 0..=max_x {
            for z in 0..=max_z {
                let h_here = height_map[x as usize][z as usize];

                // Calculate bottom_y based on current height, not max neighborhood height
                let bottom_y = (h_here - cfg.base_thickness + 1).max(0);

                // Fill current column up to its own height
                for y in bottom_y..=h_here {
                    let coord = Vector3::new(x, y, z);
                    if !within_bounds_arr(&coord, &cfg.size) {
                        continue;
                    }
                    let mut cell = self.cells_mut().entry(coord).or_insert(Cell::empty());
                    *cell |= Cell::GROUND | Cell::FILLED;
                }
            }
        }

        // --- 3. Second pass: fill gaps between different height levels ---
        for x in 0..=max_x {
            for z in 0..=max_z {
                let h_here = height_map[x as usize][z as usize];

                // Check all 8 neighbors and fill intermediate levels to prevent gaps
                for dx in -1..=1 {
                    for dz in -1..=1 {
                        if dx == 0 && dz == 0 {
                            continue;
                        }
                        let nx = x + dx;
                        let nz = z + dz;
                        if nx < 0 || nz < 0 || nx > max_x || nz > max_z {
                            continue;
                        }

                        let h_neighbor = height_map[nx as usize][nz as usize];

                        // Fill intermediate levels between current and neighbor heights
                        let min_height = h_here.min(h_neighbor);
                        let max_height = h_here.max(h_neighbor);

                        // Fill the gap between the two heights
                        for y in min_height..=max_height {
                            let coord = Vector3::new(x, y, z);
                            if !within_bounds_arr(&coord, &cfg.size) {
                                continue;
                            }
                            let mut cell = self.cells_mut().entry(coord).or_insert(Cell::empty());
                            *cell |= Cell::GROUND | Cell::FILLED;
                        }
                    }
                }
            }
        }

        height_map // Return the height map
    }

    // ------------------------------------------------------------------
    // 2. SCATTER TREES  --------------------------------------------------
    // ------------------------------------------------------------------
    /// Places voxel trees across the world.  The algorithm replicates the
    /// original behaviour: a Perlin‑based density field selects candidate
    /// (x,z) positions, then each tree is generated with a trunk and tapered
    /// canopy.  All numeric heuristics match the legacy code so visuals stay
    /// identical.

    pub fn scatter_trees(
        &mut self,
        cfg: &TerrainConfig,
        height_map: &Vec<Vec<i32>>,
        max_trees: usize,
    ) {
        let [max_x, max_y, max_z] = cfg.size.into();
        let min_distance_sq = 64; // Minimum squared distance between trees
        let tree_density_threshold = 0.7; // Threshold for placing trees in groves

        // Initialize Perlin noise for tree density
        let tree_noise = Perlin::new(cfg.seed.wrapping_add(2));
        let mut rng = StdRng::seed_from_u64(cfg.seed as u64);
        let mut placed_trees = Vec::new();

        // Possible tree types
        let tree_types = [
            (TreeType::Oak, "Oak"),
            (TreeType::Pine, "Pine"),
            (TreeType::Birch, "Birch"),
            (TreeType::Willow, "Willow"),
        ];

        // Iterate over the terrain grid
        for x in (0..=max_x).step_by(2) {
            // Step by 2 to reduce checks
            for z in (0..=max_z).step_by(2) {
                if placed_trees.len() >= max_trees {
                    break;
                }

                // Get Perlin noise value for tree density
                let density = tree_noise.get([
                    x as f64 / cfg.forest_scale,
                    z as f64 / cfg.forest_scale,
                    0.0,
                ]);
                let density = (density + 1.0) * 0.5; // Normalize to [0,1]
                if density < tree_density_threshold {
                    continue; // Skip if not in a dense grove area
                }

                // Get ground height
                let ground_y = height_map[x as usize][z as usize];

                // Check minimum distance to existing trees
                let too_close = placed_trees.iter().any(|&(px, pz, _, _)| {
                    let dx = x - px;
                    let dz = z - pz;
                    (dx * dx + dz * dz) < min_distance_sq
                });
                if too_close {
                    continue;
                }

                // Randomly select tree type
                let tree_idx = rng.random_range(0..tree_types.len());
                let (tree_type, _type_name) = tree_types[tree_idx];

                // Scale base height based on density for variation (0.8 to 1.2)
                let height_scale =
                    0.8 + (density - tree_density_threshold) / (1.0 - tree_density_threshold) * 0.4;
                let base_height =
                    (cfg.max_forest_height as f32 * height_scale as f32).round() as i32;

                // Generate tree parameters
                let params = TreeParams::for_type(tree_type, base_height, &mut rng);

                // Ensure canopy doesn't intersect ground or exceed bounds
                let canopy_bottom_y = ground_y + (params.height as f32 * 0.5) as i32;
                if canopy_bottom_y + params.canopy_height >= max_y || canopy_bottom_y <= ground_y {
                    continue;
                }

                placed_trees.push((x, z, tree_type, params));
            }
        }

        // Generate each tree
        let forbidden = Cell::AGENT_OCCUPIED | Cell::DYNAMIC_OCCUPIED | Cell::TARGET;

        for (_, &(x, z, _tree_type, ref params)) in placed_trees.iter().enumerate() {
            let ground_y = height_map[x as usize][z as usize];

            // Detailed debug output for each tree

            let mut rng = StdRng::seed_from_u64(cfg.seed as u64 ^ ((x as u64) << 32) ^ z as u64);
            self.generate_tree_structure(x, ground_y, z, params, cfg, &mut rng, forbidden);
        }
    }

    fn generate_tree_structure(
        &mut self,
        x: i32,
        ground_y: i32,
        z: i32,
        params: &TreeParams,
        cfg: &TerrainConfig,
        rng: &mut StdRng,
        forbidden: Cell,
    ) {
        // Generate trunk with taper and slight curve
        let trunk_curve = rng.random_range(-0.3..=0.3);
        let trunk_height = (params.height as f32 * 0.5) as i32; // Increased from 0.4 to 0.5 for longer trunk

        for y in 1..=trunk_height {
            let height_ratio = 1.0 - (y as f32 / trunk_height as f32) * params.trunk_taper;
            let current_radius = ((params.trunk_radius as f32 * height_ratio).max(0.5)) as i32;

            // Add slight curve to trunk
            let curve_offset = (trunk_curve * (y as f32 / trunk_height as f32).powi(2)) as i32;

            // Fill trunk voxels
            for dx in -current_radius..=current_radius {
                for dz in -current_radius..=current_radius {
                    if dx * dx + dz * dz > current_radius * current_radius {
                        continue;
                    }

                    let coord = Vector3::new(x + dx + curve_offset, ground_y + y, z + dz);
                    if !within_bounds_arr(&coord, &cfg.size) {
                        continue;
                    }

                    let mut entry = self.cells_mut().entry(coord).or_insert(Cell::empty());
                    if !entry.intersects(forbidden) {
                        *entry |= Cell::FILLED;
                    }
                }
            }
        }

        // Generate branches and canopy based on tree type
        match params.tree_type {
            TreeType::Pine => {
                self.generate_pine_canopy(x, ground_y, z, trunk_height, params, cfg, rng, forbidden)
            }
            TreeType::Oak => {
                self.generate_oak_canopy(x, ground_y, z, trunk_height, params, cfg, rng, forbidden)
            }
            TreeType::Birch => self.generate_birch_canopy(
                x,
                ground_y,
                z,
                trunk_height,
                params,
                cfg,
                rng,
                forbidden,
            ),
            TreeType::Willow => self.generate_willow_canopy(
                x,
                ground_y,
                z,
                trunk_height,
                params,
                cfg,
                rng,
                forbidden,
            ),
        }
    }

    fn generate_pine_canopy(
        &mut self,
        x: i32,
        ground_y: i32,
        z: i32,
        trunk_height: i32,
        params: &TreeParams,
        cfg: &TerrainConfig,
        rng: &mut StdRng,
        forbidden: Cell,
    ) {
        // Conical shape with horizontal branches
        let canopy_start = trunk_height / 2;

        for layer in 0..params.branch_layers {
            let y = ground_y + canopy_start + (layer * 2);
            if y > ground_y + params.height {
                break;
            }

            // Create horizontal branches in 4-6 directions
            let num_branches = rng.random_range(4..=6);
            let layer_progress = layer as f32 / params.branch_layers as f32;
            let branch_length =
                ((params.branch_length as f32) * (1.0 - layer_progress * 0.7)) as i32;

            for i in 0..num_branches {
                let angle = (i as f32 / num_branches as f32) * std::f32::consts::TAU;

                // Draw branch
                for r in 0..=branch_length {
                    let bx = x + (angle.cos() * r as f32) as i32;
                    let bz = z + (angle.sin() * r as f32) as i32;

                    let coord = Vector3::new(bx, y, bz);
                    if !within_bounds_arr(&coord, &cfg.size) {
                        continue;
                    }

                    {
                        let mut entry = self.cells_mut().entry(coord).or_insert(Cell::empty());
                        if entry.intersects(forbidden) {
                            continue;
                        }
                        *entry |= if rng.random::<f32>() < 0.7 {
                            Cell::SPARSE
                        } else {
                            Cell::FILLED
                        }; // Prefer SPARSE for branches
                    }
                    // Add needle clusters around branches
                    if r > branch_length / 3 {
                        for _ in 0..3 {
                            let dx = rng.random_range(-1..=1);
                            let dy = rng.random_range(-1..=0);
                            let dz = rng.random_range(-1..=1);

                            let needle_coord = Vector3::new(bx + dx, y + dy, bz + dz);
                            if !within_bounds_arr(&needle_coord, &cfg.size) {
                                continue;
                            }

                            if rng.random::<f32>() < params.leaf_density {
                                let mut entry = self
                                    .cells_mut()
                                    .entry(needle_coord)
                                    .or_insert(Cell::empty());
                                if !entry.intersects(forbidden) {
                                    *entry |= Cell::FILLED;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn generate_oak_canopy(
        &mut self,
        x: i32,
        ground_y: i32,
        z: i32,
        trunk_height: i32,
        params: &TreeParams,
        cfg: &TerrainConfig,
        rng: &mut StdRng,
        forbidden: Cell,
    ) {
        // Spherical canopy with thick branches
        let canopy_center_y = ground_y + trunk_height + params.canopy_height / 2;

        // Generate main branches
        let num_main_branches = rng.random_range(3..=5);
        for i in 0..num_main_branches {
            let angle = (i as f32 / num_main_branches as f32) * std::f32::consts::TAU
                + rng.random_range(-0.3..=0.3);
            let upward_angle = rng.random_range(0.3..=0.6); // 30-60 degrees

            // Main branch
            for t in 0..params.branch_length {
                let progress = t as f32 / params.branch_length as f32;
                let bx = x + (angle.cos() * t as f32) as i32;
                let by = ground_y + trunk_height + (upward_angle * t as f32) as i32;
                let bz = z + (angle.sin() * t as f32) as i32;

                // Branch thickness decreases with distance
                let thickness = (2.0 * (1.0 - progress * 0.7)) as i32;

                for dx in -thickness..=thickness {
                    for dy in -thickness..=thickness {
                        for dz in -thickness..=thickness {
                            if dx * dx + dy * dy + dz * dz > thickness * thickness {
                                continue;
                            }

                            let coord = Vector3::new(bx + dx, by + dy, bz + dz);
                            if !within_bounds_arr(&coord, &cfg.size) {
                                continue;
                            }

                            let mut entry = self.cells_mut().entry(coord).or_insert(Cell::empty());
                            if !entry.intersects(forbidden) {
                                *entry |= if rng.random::<f32>() < 0.7 {
                                    Cell::SPARSE
                                } else {
                                    Cell::FILLED
                                }; // Prefer SPARSE for branches
                            }
                        }
                    }
                }

                // Add sub-branches
                if t > params.branch_length / 3 && rng.random::<f32>() < 0.3 {
                    let sub_angle = angle + rng.random_range(-1.0..=1.0);
                    let sub_length = rng.random_range(2..=4);

                    for s in 0..sub_length {
                        let sbx = bx + (sub_angle.cos() * s as f32) as i32;
                        let sby = by + rng.random_range(-1..=1);
                        let sbz = bz + (sub_angle.sin() * s as f32) as i32;

                        let coord = Vector3::new(sbx, sby, sbz);
                        if !within_bounds_arr(&coord, &cfg.size) {
                            continue;
                        }

                        let mut entry = self.cells_mut().entry(coord).or_insert(Cell::empty());
                        if !entry.intersects(forbidden) {
                            *entry |= if rng.random::<f32>() < 0.7 {
                                Cell::SPARSE
                            } else {
                                Cell::FILLED
                            }; // Prefer SPARSE for sub-branches
                        }
                    }
                }
            }
        }

        // Fill spherical canopy with leaves
        for dy in -params.canopy_radius..=params.canopy_radius {
            for dx in -params.canopy_radius..=params.canopy_radius {
                for dz in -params.canopy_radius..=params.canopy_radius {
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    let radius_sq = params.canopy_radius * params.canopy_radius;

                    if dist_sq > radius_sq {
                        continue;
                    }

                    // Make canopy less dense at edges
                    let density_factor = 1.0 - (dist_sq as f32 / radius_sq as f32);
                    if rng.random::<f32>() > params.leaf_density * density_factor {
                        continue;
                    }

                    let coord = Vector3::new(x + dx, canopy_center_y + dy, z + dz);
                    if !within_bounds_arr(&coord, &cfg.size) {
                        continue;
                    }

                    // Check if near a branch for higher density
                    let mut entry = self.cells_mut().entry(coord).or_insert(Cell::empty());
                    if !entry.intersects(forbidden) {
                        *entry |= Cell::FILLED; // Always set FILLED for canopy leaves
                    }
                }
            }
        }
    }

    fn generate_birch_canopy(
        &mut self,
        x: i32,
        ground_y: i32,
        z: i32,
        trunk_height: i32,
        params: &TreeParams,
        cfg: &TerrainConfig,
        rng: &mut StdRng,
        forbidden: Cell,
    ) {
        // Tall, narrow canopy with delicate branches
        let canopy_start = trunk_height / 2; // Start branches higher to connect with long trunk

        // Branches for connectivity
        for layer in 0..params.branch_layers {
            let y = ground_y + canopy_start + ((layer as f32 * 1.0) as i32); // Standard step size
            if y > ground_y + trunk_height {
                break;
            } // Stop at trunk height

            let num_branches = rng.random_range(2..=4); // Fewer branches for narrower spread
            for i in 0..num_branches {
                let angle = (i as f32 / num_branches as f32) * std::f32::consts::TAU;
                let length = rng.random_range(2..=params.branch_length); // Shorter branches

                // Thin branches
                for r in 0..length {
                    let bx = x + (angle.cos() * r as f32) as i32;
                    let by = y + (r as f32 * 0.3) as i32; // Gentler upward curve
                    let bz = z + (angle.sin() * r as f32) as i32;

                    let coord = Vector3::new(bx, by, bz);
                    if !within_bounds_arr(&coord, &cfg.size) {
                        continue;
                    }
                    {
                        let mut entry = self.cells_mut().entry(coord).or_insert(Cell::empty());
                        if !entry.intersects(forbidden) {
                            *entry |= if rng.random::<f32>() < 0.7 {
                                Cell::SPARSE
                            } else {
                                Cell::FILLED
                            }; // Prefer SPARSE
                        }
                    }
                    // Leaf clusters for connectivity
                    if r > length / 2 || y < ground_y + trunk_height {
                        for _ in 0..3 {
                            // Reduced for narrower canopy
                            let leaf_coord = Vector3::new(
                                bx + rng.random_range(-1..=1),
                                by + rng.random_range(-1..=1),
                                bz + rng.random_range(-1..=1),
                            );
                            if !within_bounds_arr(&leaf_coord, &cfg.size) {
                                continue;
                            }

                            if rng.random::<f32>() < params.leaf_density * 1.2 {
                                // Reduced density
                                let mut entry =
                                    self.cells_mut().entry(leaf_coord).or_insert(Cell::empty());
                                if !entry.intersects(forbidden) {
                                    *entry |= if rng.random::<f32>() < 0.5 {
                                        Cell::SPARSE
                                    } else {
                                        Cell::FILLED
                                    }; // Mix SPARSE and FILLED
                                }
                            }
                        }
                    }
                }
            }
        }

        // Small oval canopy at top, aligned with trunk
        let canopy_top_y = ground_y + trunk_height; // Align with trunk top
        for dy in -params.canopy_height / 2..=params.canopy_height / 2 {
            let y_progress = (dy.abs() as f32) / (params.canopy_height as f32 / 2.0);
            let layer_radius = ((params.canopy_radius as f32) * (1.0 - y_progress * 0.3)) as i32;

            for dx in -layer_radius..=layer_radius {
                for dz in -layer_radius..=layer_radius {
                    if dx * dx + dz * dz > layer_radius * layer_radius {
                        continue;
                    }

                    let coord = Coord::new(x + dx, canopy_top_y + dy, z + dz);
                    if !within_bounds_arr(&coord, &cfg.size) {
                        continue;
                    }

                    if rng.random::<f32>() < params.leaf_density * (1.0 - y_progress * 0.2) {
                        // Slightly less dense
                        let mut entry = self.cells_mut().entry(coord).or_insert(Cell::empty());
                        if !entry.intersects(forbidden) {
                            *entry |= if rng.random::<f32>() < 0.3 {
                                Cell::SPARSE
                            } else {
                                Cell::FILLED
                            }; // Wispy look
                        }
                    }
                }
            }
        }
    }

    fn generate_willow_canopy(
        &mut self,
        x: i32,
        ground_y: i32,
        z: i32,
        trunk_height: i32,
        params: &TreeParams,
        cfg: &TerrainConfig,
        rng: &mut StdRng,
        forbidden: Cell,
    ) {
        // Drooping branches that curve downward
        let branch_start = trunk_height / 2;

        // More drooping branches
        let num_branches = rng.random_range(8..=12); // Increased from 6..=10 for more branches
        for i in 0..num_branches {
            let angle = (i as f32 / num_branches as f32) * std::f32::consts::TAU
                + rng.random_range(-0.2..=0.2);
            let start_y = ground_y + branch_start + rng.random_range(0..=trunk_height / 3);

            // Each branch curves downward
            for t in 0..params.branch_length {
                let progress = t as f32 / params.branch_length as f32;
                let droop = (progress * progress * 6.0) as i32; // Reduced from 8.0 to 6.0 for gentler droop

                let bx = x + (angle.cos() * t as f32) as i32;
                let by = start_y - droop;
                let bz = z + (angle.sin() * t as f32) as i32;

                if by <= ground_y {
                    break;
                } // Don't go below ground

                let coord = Vector3::new(bx, by, bz);
                if !within_bounds_arr(&coord, &cfg.size) {
                    continue;
                }
                {
                    let mut entry = self.cells_mut().entry(coord).or_insert(Cell::empty());
                    if !entry.intersects(forbidden) {
                        *entry |= if rng.random::<f32>() < 0.7 {
                            Cell::SPARSE
                        } else {
                            Cell::FILLED
                        }; // Prefer SPARSE for branches
                    }
                }
                // Hanging leaves along the branch
                if t > 2 {
                    for dy in -2..=0 {
                        let leaf_coord = Vector3::new(bx, by + dy, bz);
                        if !within_bounds_arr(&leaf_coord, &cfg.size) {
                            continue;
                        }

                        if rng.random::<f32>() < params.leaf_density {
                            let mut entry =
                                self.cells_mut().entry(leaf_coord).or_insert(Cell::empty());
                            if !entry.intersects(forbidden) {
                                *entry |= if rng.random::<f32>() < 0.5 {
                                    Cell::SPARSE
                                } else {
                                    Cell::FILLED
                                }; // Increased SPARSE probability from 0.3 to 0.5
                            }
                        }
                    }
                }
            }
        }

        // Dense canopy core with more SPARSE gaps
        let canopy_y = ground_y + trunk_height;
        for dy in 0..params.canopy_height / 2 {
            for dx in -params.canopy_radius / 2..=params.canopy_radius / 2 {
                for dz in -params.canopy_radius / 2..=params.canopy_radius / 2 {
                    if dx * dx + dz * dz > (params.canopy_radius / 2) * (params.canopy_radius / 2) {
                        continue;
                    }

                    let coord = Vector3::new(x + dx, canopy_y + dy, z + dz);
                    if !within_bounds_arr(&coord, &cfg.size) {
                        continue;
                    }

                    if rng.random::<f32>() < params.leaf_density * 1.0 {
                        // Reduced from 1.2 to 1.0 for more gaps
                        let mut entry = self.cells_mut().entry(coord).or_insert(Cell::empty());
                        if !entry.intersects(forbidden) {
                            *entry |= if rng.random::<f32>() < 0.5 {
                                Cell::SPARSE
                            } else {
                                Cell::FILLED
                            }; // Added SPARSE probability for canopy core
                        }
                    }
                }
            }
        }
    }
    // ------------------------------------------------------------------
    // 3. CARVE PASSAGES  -------------------------------------------------
    // ------------------------------------------------------------------
    /// Removes voxels along paths where `passage_fn` returns **true**.
    /// Typical usage: generate cave tunnels based on 3‑D noise or a
    /// randomized BFS. The closure is called for every coordinate and must
    /// be **pure** (no side‑effects) to keep the borrow checker happy.
    pub fn carve_passages(
        &mut self,
        cfg: &TerrainConfig,
        //                     ▼ trait-object, no generics
        passage_fn: &mut dyn FnMut(i32, i32, i32) -> bool,
    ) {
        let [max_x, max_y, max_z] = cfg.size.into();
        for x in 0..=max_x {
            for y in 0..=max_y {
                for z in 0..=max_z {
                    if !passage_fn(x, y, z) {
                        continue;
                    }
                    let coord = Vector3::new(x, y, z);
                    let mut to_remove = None;
                    if let Some(mut cell) = self.cells_mut().get_mut(&coord) {
                        if cell.contains(Cell::SPARSE | Cell::GROUND) {
                            *cell &= !Cell::FILLED;
                        } else {
                            to_remove = Some(coord);
                        }
                    }
                    if let Some(to_remove) = to_remove {
                        self.cells_mut().remove(&to_remove);
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    pub fn generate_terrain(&mut self, cfg: &TerrainConfig) {
        // -------------------------------------------------- ground height ----
        let flat_band = cfg.flat_band;
        let height_scale = cfg.height_scale;
        let ground_low = Perlin::new(cfg.seed);
        let ground_base = Perlin::new(cfg.seed.wrapping_add(1));

        let mut height_fn = |x: i32, z: i32| -> f64 {
            let a = ground_base.get([x as f64 / height_scale, z as f64 / height_scale, 0.0]);
            let b = ground_low.get([
                x as f64 / (height_scale * 4.0),
                z as f64 / (height_scale * 4.0),
                1.0,
            ]);
            let mut n = ((a * 0.8 + b * 0.2) + 1.0) * 0.5;
            if (0.5 - flat_band..=0.5 + flat_band).contains(&n) {
                n = 0.5;
            }
            n
        };

        // Get the height map from build_ground
        let height_map = self.build_ground(&cfg, &mut height_fn);

        // -------------------------------------------------- trees ------------
        // Pass the height map to scatter_trees
        self.scatter_trees(&cfg, &height_map, 50);

        // -------------------------------------------------- caves ------------
        // let cave_noise = Perlin::new(cfg.seed.wrapping_add(1234));
        // let cave_scale = 20.0;
        // let threshold = 0.8;

        // let passage_fn = move |x: i32, y: i32, z: i32| -> bool {
        //     if y <= 1 {
        //         return false;
        //     }
        //     let n = cave_noise.get([
        //         x as f64 / cave_scale,
        //         y as f64 / cave_scale,
        //         z as f64 / cave_scale,
        //     ]);
        //     (n + 1.0) * 0.5 > threshold
        // };

        // Flip coordinates.
        // self.carve_passages(&cfg, &mut passage_fn);
    }
}

fn within_bounds_arr(v: &Coord, max: &Vector3<i32>) -> bool {
    v.iter()
        .zip(max.iter())
        .all(|(vi, bi)| *vi >= 0 && *vi <= *bi)
}
