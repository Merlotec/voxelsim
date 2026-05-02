use dashmap::DashMap;
use nalgebra::{Unit, UnitQuaternion, UnitVector3, Vector3};

fn div_floor_i64(a: i64, b: i64) -> i64 {
    // Floor division for possibly negative integers
    let mut q = a / b;
    let r = a % b;
    if (r != 0) && ((r > 0) != (b > 0)) {
        q -= 1;
    }
    q
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct UncertaintyWorld {
    chunks: DashMap<Vector3<i32>, UncertaintyField>,
    // World-space position corresponding to node index (0,0,0)
    world_origin: Vector3<f64>,
    // Number of nodes per chunk along each axis
    chunk_size: Option<Vector3<usize>>, // inferred from first inserted field
    // Physical spacing between adjacent uncertainty nodes (world units per node)
    node_size: f64,
}

/// Dense grid of uncertainty.
/// Allows decoupling of uncertainty and block position.
#[derive(Debug, Clone)]
pub struct UncertaintyField {
    field_storage: Vec<[f64; 8]>,
    size: Vector3<usize>,
}

impl UncertaintyField {
    pub fn new(size: Vector3<usize>) -> Self {
        let len = size.x * size.y * size.z;

        Self {
            field_storage: vec![[0.0; 8]; len],
            size,
        }
    }
}

const SQRT_3: f64 = 1.732050807568877293527446341505872367_f64;

impl UncertaintyField {
    pub const fn dir(i: usize) -> UnitVector3<f64> {
        let x: f64 = if i < 4 { SQRT_3 } else { -SQRT_3 };

        let y: f64 = if i % 4 < 2 { SQRT_3 } else { -SQRT_3 };

        let z: f64 = if i % 2 == 0 { SQRT_3 } else { -SQRT_3 };

        UnitVector3::new_unchecked(Vector3::new(x, y, z))
    }

    pub const fn dirs() -> [UnitVector3<f64>; 8] {
        [
            Self::dir(0),
            Self::dir(1),
            Self::dir(2),
            Self::dir(3),
            Self::dir(4),
            Self::dir(5),
            Self::dir(6),
            Self::dir(7),
        ]
    }

    pub fn index(&self, coord: Vector3<usize>) -> usize {
        coord.x * (self.size.y * self.size.z) + coord.y * self.size.z + coord.z
    }
    pub fn get(&self, coord: Vector3<usize>) -> Option<&[f64; 8]> {
        self.field_storage.get(self.index(coord))
    }
}

impl UncertaintyField {
    pub fn sample_node(&self, coord: Vector3<usize>, in_dir: UnitVector3<f64>) -> Option<f64> {
        let values = self.get(coord)?;
        let dirs = Self::dirs().map(|d| d.dot(&in_dir));

        let to_collect: Vec<(usize, f64)> = dirs
            .into_iter()
            .enumerate()
            .filter(|(_, x)| *x > 0.0)
            .collect();

        let sum: f64 = to_collect.iter().map(|(_, x)| x).sum();

        Some(to_collect.iter().map(|(i, x)| (x / sum) * values[*i]).sum())
    }

    pub fn sample_field(&self, pos: Vector3<f64>, in_dir: UnitVector3<f64>) -> Option<f64> {
        // Require non-negative coordinates (we'll let `get()` enforce upper bounds).
        if pos.x < 0.0 || pos.y < 0.0 || pos.z < 0.0 {
            return None;
        }

        // XY cell indices and interpolation fractions.
        let i0 = pos.x.floor() as usize;
        let j0 = pos.y.floor() as usize;
        let i1 = i0.checked_add(1)?; // avoid overflow
        let j1 = j0.checked_add(1)?;

        let tx = pos.x - i0 as f64; // in [0, 1) if inside a cell
        let ty = pos.y - j0 as f64;

        // Use the nearest Z layer (so we only touch four nodes total).
        let k = pos.z.round();
        if k < 0.0 {
            return None;
        }
        let k = k as usize;

        // Sample the four corners via your node sampler.
        let v00 = self.sample_node(Vector3::new(i0, j0, k), in_dir)?;
        let v10 = self.sample_node(Vector3::new(i1, j0, k), in_dir)?;
        let v01 = self.sample_node(Vector3::new(i0, j1, k), in_dir)?;
        let v11 = self.sample_node(Vector3::new(i1, j1, k), in_dir)?;

        // Bilinear blend.
        let w00 = (1.0 - tx) * (1.0 - ty);
        let w10 = tx * (1.0 - ty);
        let w01 = (1.0 - tx) * ty;
        let w11 = tx * ty;

        Some(v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11)
    }
}

impl UncertaintyWorld {
    pub fn new(world_origin: Vector3<f64>, node_size: f64) -> Self {
        Self {
            chunks: DashMap::new(),
            world_origin,
            chunk_size: None,
            node_size,
        }
    }

    /// Set world-space origin corresponding to node index (0,0,0).
    pub fn set_world_origin(&mut self, origin: Vector3<f64>) {
        self.world_origin = origin;
    }

    pub fn set_chunk_size(&mut self, size: Vector3<usize>) {
        self.chunk_size = Some(size);
    }

    /// Set the physical spacing between nodes in world units (default 1.0).
    pub fn set_node_size(&mut self, node_size: f64) {
        self.node_size = node_size.max(1e-9);
    }

    pub fn insert_chunk(&mut self, key: Vector3<i32>, field: UncertaintyField) {
        if self.chunk_size.is_none() {
            self.chunk_size = Some(field.size);
        }
        self.chunks.insert(key, field);
    }

    fn node_from_global_idx(
        &self,
        gi: i64,
        gj: i64,
        gk: i64,
    ) -> Option<(Vector3<i32>, Vector3<usize>)> {
        let size = self.chunk_size?;
        let sx = size.x as i64;
        let sy = size.y as i64;
        let sz = size.z as i64;

        let cx = div_floor_i64(gi, sx);
        let cy = div_floor_i64(gj, sy);
        let cz = div_floor_i64(gk, sz);

        let lx = (gi - cx * sx) as usize;
        let ly = (gj - cy * sy) as usize;
        let lz = (gk - cz * sz) as usize;

        Some((
            Vector3::new(cx as i32, cy as i32, cz as i32),
            Vector3::new(lx, ly, lz),
        ))
    }

    fn sample_node_global(
        &self,
        gi: i64,
        gj: i64,
        gk: i64,
        in_dir: UnitVector3<f64>,
    ) -> Option<f64> {
        let (ckey, lidx) = self.node_from_global_idx(gi, gj, gk)?;
        let field = self.chunks.get(&ckey)?;
        field.sample_node(lidx, in_dir)
    }

    /// Sample the uncertainty field(s) at a global position in node coordinates,
    /// accounting for chunk boundaries by blending across nodes that may reside
    /// in neighboring chunks.
    /// Sample using world-space position; converts to node grid via `node_size`.
    pub fn sample_field(&self, pos_world: Vector3<f64>, in_dir: UnitVector3<f64>) -> Option<f64> {
        // Convert world-space to node-space
        let pos = (pos_world - self.world_origin) / self.node_size;
        // Convert to global node coordinates relative to origin (integer grid)
        // Here we assume pos already expressed in node units. If you want world-units,
        // do the conversion before calling.
        let gx = pos.x.floor();
        let gy = pos.y.floor();
        let gz = pos.z.round();

        // Reject negative global indices (before origin). Adjust if you maintain origin offset.
        if gx.is_nan() || gy.is_nan() || gz.is_nan() {
            return None;
        }

        let i0 = gx as i64;
        let j0 = gy as i64;
        let k = gz as i64;
        let i1 = i0 + 1;
        let j1 = j0 + 1;

        let tx = pos.x - gx;
        let ty = pos.y - gy;

        // Gather four corners possibly from different chunks
        let v00 = self.sample_node_global(i0, j0, k, in_dir);
        let v10 = self.sample_node_global(i1, j0, k, in_dir);
        let v01 = self.sample_node_global(i0, j1, k, in_dir);
        let v11 = self.sample_node_global(i1, j1, k, in_dir);

        // Bilinear weights
        let w00 = (1.0 - tx) * (1.0 - ty);
        let w10 = tx * (1.0 - ty);
        let w01 = (1.0 - tx) * ty;
        let w11 = tx * ty;

        // Accumulate only available samples and renormalize
        let mut acc = 0.0;
        let mut wsum = 0.0;
        if let Some(v) = v00 {
            acc += v * w00;
            wsum += w00;
        }
        if let Some(v) = v10 {
            acc += v * w10;
            wsum += w10;
        }
        if let Some(v) = v01 {
            acc += v * w01;
            wsum += w01;
        }
        if let Some(v) = v11 {
            acc += v * w11;
            wsum += w11;
        }

        if wsum > 0.0 { Some(acc / wsum) } else { None }
    }
}

impl UncertaintyWorld {
    /// Insert or update a single node at global indices (gi, gj, gk).
    /// Returns true if the node was updated in an existing chunk; false if the chunk was missing.
    pub fn insert_node(&mut self, gi: i64, gj: i64, gk: i64, value: [f64; 8]) -> bool {
        let Some((ckey, lidx)) = self.node_from_global_idx(gi, gj, gk) else {
            return false;
        };
        if let Some(mut field) = self.chunks.get_mut(&ckey) {
            let idx = field.index(lidx);
            if let Some(slot) = field.field_storage.get_mut(idx) {
                *slot = value;
                return true;
            }
        }
        false
    }

    /// Convenience: insert/update using a world-space position snapped to nearest node.
    pub fn insert_node_world(&mut self, pos_world: Vector3<f64>, value: [f64; 8]) -> bool {
        let pn = (pos_world - self.world_origin) / self.node_size;
        let gi = pn.x.round() as i64;
        let gj = pn.y.round() as i64;
        let gk = pn.z.round() as i64;
        self.insert_node(gi, gj, gk, value)
    }

    /// Update nodes within the camera view frustum up to `max_dist` with linear distance falloff.
    /// The frustum is approximated as a cone with half-angle `half_fov_rad` around the forward axis.
    /// For each affected node, uncertainty values are reduced by (1 - weight), where
    /// weight = max(0, 1 - dist/max_dist) and angle within the cone.
    pub fn update_view_frustum(
        &mut self,
        cam_pos_world: Vector3<f64>,
        cam_orient: UnitQuaternion<f64>,
        half_fov_rad: f64,
        max_dist: f64,
    ) {
        if self.chunk_size.is_none() {
            return;
        }

        // Compute forward direction from orientation
        let base_forward = Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0));
        let fwd_vec = (cam_orient * base_forward).into_inner();
        let fwd = Unit::new_normalize(fwd_vec);
        let cos_thresh = half_fov_rad.cos();

        // Iterate global node indices within a bounding cube around the camera
        // Compute search bounds in node indices using node_size spacing
        let ns = self.node_size;
        let min_i = (((cam_pos_world.x - max_dist) - self.world_origin.x) / ns).floor() as i64;
        let max_i = (((cam_pos_world.x + max_dist) - self.world_origin.x) / ns).ceil() as i64;
        let min_j = (((cam_pos_world.y - max_dist) - self.world_origin.y) / ns).floor() as i64;
        let max_j = (((cam_pos_world.y + max_dist) - self.world_origin.y) / ns).ceil() as i64;
        let min_k = (((cam_pos_world.z - max_dist) - self.world_origin.z) / ns).floor() as i64;
        let max_k = (((cam_pos_world.z + max_dist) - self.world_origin.z) / ns).ceil() as i64;

        for gi in min_i..=max_i {
            for gj in min_j..=max_j {
                for gk in min_k..=max_k {
                    // Node center in world coordinates
                    let p_world = Vector3::new(
                        gi as f64 * ns + self.world_origin.x,
                        gj as f64 * ns + self.world_origin.y,
                        gk as f64 * ns + self.world_origin.z,
                    );
                    let dvec = p_world - cam_pos_world;
                    let dist = dvec.norm();
                    if dist > max_dist || dist <= 1e-6 {
                        continue;
                    }
                    let dir = Unit::new_normalize(dvec);
                    let cosang = dir.dot(&fwd);
                    if cosang < cos_thresh {
                        continue;
                    }
                    // Linear falloff with distance
                    let weight = 1.0 - (dist / max_dist);

                    // Locate chunk and update node
                    if let Some((ckey, lidx)) = self.node_from_global_idx(gi, gj, gk) {
                        if let Some(mut field) = self.chunks.get_mut(&ckey) {
                            let idx = field.index(lidx);
                            if let Some(slot) = field.field_storage.get_mut(idx) {
                                // Reduce uncertainty proportionally across all directions
                                for v in slot.iter_mut() {
                                    *v *= 1.0 - weight;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
