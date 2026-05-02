use dashmap::DashMap;
use nalgebra::Vector3;
use nalgebra::{Matrix3, SymmetricEigen};
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tinyvec::{ArrayVec, array_vec};

use crate::{Agent, Coord, VoxelGrid, VoxelSet};

#[derive(Default)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PrincipalComponent {
    pub l: f64,
    pub u: Vector3<f64>,
    pub expl_var: f64,
}

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PhaseFlow {
    sources: DashMap<Coord, ArrayVec<[PrincipalComponent; 3]>>,
}

pub trait PhaseSolver {
    fn latest_timestamp(&self) -> Option<f64>;
    fn add_phase_frame(&mut self, agent: &Agent, virtual_world: &VoxelGrid, t: f64);
    fn gen_phase_grid(&self, agent: &Agent, time: std::ops::Range<f64>) -> PhaseGrid;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PhaseGrid {
    pub cells: DashMap<Coord, f64>,
}

impl PhaseGrid {
    pub fn new() -> Self {
        Self {
            cells: DashMap::new(),
        }
    }
    pub fn from_cells(cells: DashMap<Coord, f64>) -> Self {
        Self { cells }
    }

    /// Rasterize a line segment from `start` to `end` into the grid, accumulating `weight`
    /// into every voxel the line passes through.
    ///
    /// - `start`, `end`: world coordinates where voxel centres are at integer positions and
    ///   each voxel spans [i-0.5, i+0.5] on each axis.
    /// - `weight`: contribution in [0, 1]. Multiple calls accumulate using a bounded add
    ///   (approaches 1.0 without exceeding it).
    pub fn fill_segment(&mut self, start: Vector3<f64>, end: Vector3<f64>, weight: f64) {
        let w = weight.clamp(0.0, 1.0);
        if w == 0.0 || !start.iter().all(|v| v.is_finite()) || !end.iter().all(|v| v.is_finite()) {
            return;
        }

        // Helper to accumulate with bounded addition: (a + b) / (1 + a*b)
        let acc = |idx: Coord| {
            let mut e = self.cells.entry(idx).or_insert(0.0);
            let a = (*e).clamp(0.0, 1.0);
            *e = (a + w) / (1.0 + a * w);
        };

        // Degenerate: single point
        if (end - start).max() == 0.0 {
            let idx = start.map(|v| (v + 0.5).floor() as i32);
            acc(idx);
            return;
        }

        // Direction and stepping
        let d = end - start;

        let mut ix = (start.x + 0.5).floor() as i32;
        let mut iy = (start.y + 0.5).floor() as i32;
        let mut iz = (start.z + 0.5).floor() as i32;
        let tx = (end.x + 0.5).floor() as i32;
        let ty = (end.y + 0.5).floor() as i32;
        let tz = (end.z + 0.5).floor() as i32;

        let step_x: i32 = if d.x > 0.0 {
            1
        } else if d.x < 0.0 {
            -1
        } else {
            0
        };
        let step_y: i32 = if d.y > 0.0 {
            1
        } else if d.y < 0.0 {
            -1
        } else {
            0
        };
        let step_z: i32 = if d.z > 0.0 {
            1
        } else if d.z < 0.0 {
            -1
        } else {
            0
        };

        // Compute initial tMax and tDelta for Amanatides–Woo traversal.
        // t parameterizes the segment: p(t) = start + t*d, t in [0, 1].
        let (mut t_max_x, t_delta_x) = if step_x != 0 {
            let next_boundary = ix as f64 + 0.5 * step_x as f64;
            let t_max = (next_boundary - start.x) / d.x;
            (t_max, 1.0 / d.x.abs())
        } else {
            (f64::INFINITY, f64::INFINITY)
        };

        let (mut t_max_y, t_delta_y) = if step_y != 0 {
            let next_boundary = iy as f64 + 0.5 * step_y as f64;
            let t_max = (next_boundary - start.y) / d.y;
            (t_max, 1.0 / d.y.abs())
        } else {
            (f64::INFINITY, f64::INFINITY)
        };

        let (mut t_max_z, t_delta_z) = if step_z != 0 {
            let next_boundary = iz as f64 + 0.5 * step_z as f64;
            let t_max = (next_boundary - start.z) / d.z;
            (t_max, 1.0 / d.z.abs())
        } else {
            (f64::INFINITY, f64::INFINITY)
        };

        // Traverse, always including the end voxel.
        loop {
            acc(Vector3::new(ix, iy, iz));
            if ix == tx && iy == ty && iz == tz {
                break;
            }

            // Advance to the next voxel boundary (supporting ties).
            let m = t_max_x.min(t_max_y).min(t_max_z);
            if m.is_infinite() {
                // Direction is zero in all axes: nothing to march; avoid infinite loop
                break;
            }

            if m == t_max_x {
                ix += step_x;
                t_max_x += t_delta_x;
            }
            if m == t_max_y {
                iy += step_y;
                t_max_y += t_delta_y;
            }
            if m == t_max_z {
                iz += step_z;
                t_max_z += t_delta_z;
            }
        }
    }
}

impl PhaseFlow {
    pub fn from_frames(mut frames: Vec<(VoxelSet, f64)>) -> Self {
        let (present, t_end) = frames.pop().unwrap();

        let sources: DashMap<Coord, ArrayVec<[PrincipalComponent; 3]>> = DashMap::new();

        let threshold = 3.0;
        let lambda_threshold = 2.0;

        let (points, base_weights): (Vec<Vector3<f64>>, Vec<f64>) = frames
            .iter()
            .filter_map(|f| {
                let v = threshold - (t_end - f.1);
                if v > 0.0 {
                    let w = v / threshold;
                    Some(
                        f.0.cells()
                            .iter()
                            .map(|x| (x.cast::<f64>(), w))
                            .collect::<Vec<(Vector3<f64>, f64)>>(),
                    )
                } else {
                    None
                }
            })
            .flatten()
            .unzip();

        present.cells().par_iter().for_each(|src| {
            let s: Vector3<f64> = src.cast();
            let weights: Vec<f64> = base_weights
                .iter()
                .enumerate()
                .map(|(i, w)| w / points[i].metric_distance(&s))
                .collect();
            if let Some((m, lambda)) = weighted_pca3_with_centroid(&points, &weights, src.cast()) {
                let lsum: f64 = lambda.iter().sum();
                for (i, l) in lambda.iter().enumerate() {
                    if l > &lambda_threshold {
                        let expl_var = l / lsum;
                        let pc = PrincipalComponent {
                            l: *l,
                            u: m.column(i).into_owned(),
                            expl_var,
                        };
                        if let Some(mut pcs) = sources.get_mut(&src) {
                            pcs.push(pc);
                        } else {
                            sources.insert(*src, array_vec!([PrincipalComponent; 3] => pc));
                        }
                    }
                }
            }
        });

        Self { sources }
    }

    pub fn build_grid(&self, t: f64) -> PhaseGrid {
        // Accumulate contributions from each source along its principal components
        // into a scalar field over voxel coordinates.
        let cells: DashMap<Coord, f64> = DashMap::new();

        if t <= 0.0 {
            return PhaseGrid { cells };
        }

        // Tuning constants for cone marching
        let step: f64 = 0.5; // march step along the ray (in voxel units)
        let base_radius: f64 = 0.5; // initial radius near the source
        let radius_slope: f64 = 0.45; // how fast the cone opens with distance

        self.sources.par_iter().for_each(|arg| {
            let (coord, pcs) = (arg.key(), arg.value());
            let src = coord.cast::<f64>();

            for pc in pcs.iter() {
                if pc.l <= 0.0 {
                    continue;
                }

                // Length scales linearly with t and sqrt(l)
                let length = t * pc.l.sqrt();
                if length <= 0.0 {
                    continue;
                }

                let u = pc.u;
                let norm = u.norm();
                if norm == 0.0 {
                    continue;
                }
                let dir = u / norm; // unit direction

                // Strength scales with explained variance; clamp to [0,1]
                let base_strength = pc.expl_var.clamp(0.0, 1.0);

                // March forward along the ray, widening the cone
                let mut s = 0.0;
                while s <= length {
                    let center = src + dir * s;
                    let radius = base_radius + radius_slope * s;

                    // Compute bounds for integer voxels to consider around this slice
                    let min = (center - Vector3::new(radius, radius, radius))
                        .map(|v| v.floor() as i32 - 1);
                    let max = (center + Vector3::new(radius, radius, radius))
                        .map(|v| v.ceil() as i32 + 1);

                    for x in min.x..=max.x {
                        for y in min.y..=max.y {
                            for z in min.z..=max.z {
                                let c = Vector3::new(x, y, z);
                                let p = c.cast::<f64>();
                                // Radial distance from the cone axis at this slice
                                let dist = (p - center).norm();
                                if dist <= radius {
                                    // Radial falloff inside the cone (1 at axis → 0 at boundary)
                                    let radial = 1.0 - (dist / (radius + 1e-9));
                                    // Slightly sharpen towards axis
                                    let radial = radial.clamp(0.0, 1.0).powf(2.0);

                                    // Contribution at this voxel slice
                                    let delta = base_strength * radial;

                                    // Speed-of-light style accumulation: approach but never reach 1.0
                                    let mut entry = cells.entry(c).or_insert(0.0);
                                    *entry = *entry + (1.0 - *entry) * delta;
                                }
                            }
                        }
                    }

                    s += step;
                }
            }
        });

        PhaseGrid { cells }
    }
}

/// Weighted PCA in 3D *around a given centroid*.
///
/// Inputs:
/// - `points`: data points
/// - `weights`: point weights (>=0)
/// - `centroid`: the centroid to subtract (user provided)
///
/// Outputs:
/// - `components`: 3×3 matrix whose columns are PC1, PC2, PC3 (descending variance)
/// - `variances`: eigenvalues for those components
pub fn weighted_pca3_with_centroid(
    points: &[Vector3<f64>],
    weights: &[f64],
    centroid: Vector3<f64>,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    if points.len() != weights.len() || points.is_empty() {
        return None;
    }

    // 1) Weighted scatter matrix around the given centroid
    let mut sum_w = 0.0;
    let mut c = Matrix3::<f64>::zeros();

    for (p, &w) in points.iter().zip(weights) {
        if w <= 0.0 {
            continue;
        }
        let r = p - centroid;
        c += (r * r.transpose()) * w;
        sum_w += w;
    }

    if sum_w == 0.0 {
        return None;
    }

    c /= sum_w; // weighted covariance matrix (up to scale)

    // 2) Eigen decomposition (symmetric ⇒ real)
    let eig = SymmetricEigen::new(c);

    // nalgebra returns eigenvalues ascending; we want descending
    let idx = [2, 1, 0];

    let mut components = Matrix3::<f64>::zeros();
    let mut variances = Vector3::<f64>::zeros();

    for (j, &k) in idx.iter().enumerate() {
        let v = eig.eigenvectors.column(k);
        let vn = v.norm();
        if vn == 0.0 {
            return None;
        }
        let mut v = (v / vn).into_owned(); // normalized eigenvector

        // ---- choose direction so most weight lies "behind" (negative side) ----
        const EPS: f64 = 1e-12;
        let mut pos_w = 0.0;
        let mut neg_w = 0.0;

        for (p, &w) in points.iter().zip(weights) {
            if w <= 0.0 {
                continue;
            }
            let s = (p - centroid).dot(&v);
            if s > EPS {
                pos_w += w;
            }
            if s < -EPS {
                neg_w += w;
            }
            // points with |s| <= EPS don't affect the vote
        }

        // If more weight is on the positive side, flip so the majority ends up "behind"
        if pos_w > neg_w {
            v = -v;
        } else if (pos_w - neg_w).abs() <= EPS {
            // tie-breaker: make the largest-magnitude component positive for determinism
            let imax = (0..3)
                .max_by(|&a, &b| v[a].abs().partial_cmp(&v[b].abs()).unwrap())
                .unwrap();
            if v[imax] < 0.0 {
                v = -v;
            }
        }

        components.set_column(j, &v); // PC j with chosen direction
        variances[j] = eig.eigenvalues[k]; // λ_j
    }
    Some((components, variances))
}
