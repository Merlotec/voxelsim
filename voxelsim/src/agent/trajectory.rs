use bspline::BSpline;
use nalgebra::Vector3;

use crate::Coord;

fn project_point_onto_segment(
    p: &Vector3<f64>,
    a: &Vector3<f64>,
    b: &Vector3<f64>,
) -> Vector3<f64> {
    let ab = b - a;
    let ap = p - a;

    let ab_dot_ab = ab.dot(&ab);
    let ab_dot_ap = ab.dot(&ap);

    // Compute the projection scalar t
    let t = (ab_dot_ap / ab_dot_ab).clamp(0.0, 1.0);

    // Interpolate between a and b
    a + ab * t
}

fn first_surface_intersection(
    p0: Vector3<f64>,
    p1: Vector3<f64>,
    bmin: Vector3<f64>,
    bmax: Vector3<f64>,
) -> Option<Vector3<f64>> {
    let d = p1 - p0;

    // Use a practical tolerance for "parallel"
    let eps = 1e-12_f64;

    // Unclamped interval along the infinite line
    let mut t_enter = f64::NEG_INFINITY;
    let mut t_exit = f64::INFINITY;

    for i in 0..3 {
        let p0i = p0[i];
        let di = d[i];
        let bi_min = bmin[i];
        let bi_max = bmax[i];

        if di.abs() < eps {
            // Segment is (effectively) parallel to this axis slab: must be within the slab
            if p0i < bi_min || p0i > bi_max {
                return None;
            }
            // No change to t_enter/t_exit for this axis
        } else {
            let mut t0 = (bi_min - p0i) / di;
            let mut t1 = (bi_max - p0i) / di;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_enter = t_enter.max(t0);
            t_exit = t_exit.min(t1);
            if t_enter > t_exit {
                return None;
            }
        }
    }

    // Now choose the first surface hit along the *segment* t in [0,1]
    // Case 1: entering from outside (or touching): use t_enter if it’s within [0,1]
    if t_enter >= 0.0 && t_enter <= 1.0 {
        return Some(p0 + d * t_enter);
    }

    // Case 2: starting inside (t_enter < 0): first surface is at t_exit, if it’s within the segment
    if t_enter < 0.0 && t_exit >= 0.0 && t_exit <= 1.0 {
        return Some(p0 + d * t_exit);
    }

    // Otherwise, the infinite line intersects the box, but not within the segment bounds.
    None
}

pub fn solve_constrained_smoothing_pass(
    cells: &[Coord],
    cell_size: Vector3<f64>,
    path: &[Vector3<f64>],
) -> Vec<Vector3<f64>> {
    assert!(cells.len() == path.len());
    if path.len() < 2 {
        return path.to_vec();
    }
    let mut im_pts = Vec::with_capacity(path.len() - 2);
    for i in 1..path.len() - 1 {
        let a = path[i - 1];
        let b = path[i + 1];

        let p = path[i];

        let m = project_point_onto_segment(&p, &a, &b);

        let p_cnt = cells[i].cast::<f64>();

        let t = m * 0.5 + p * 0.5;

        let bmin = p_cnt - cell_size * 0.5;
        let bmax = p_cnt + cell_size * 0.5;

        // im_pts.push(t);

        if let Some(x) = first_surface_intersection(t, p, bmin, bmax) {
            // There is an intersection, so we need to pull back to the intersection.
            im_pts.push(x);
        } else {
            im_pts.push(t);
        }
    }

    let mut result = vec![path[0]];
    result.append(&mut im_pts);
    result.push(*path.last().unwrap());
    result
}

pub fn solve_constrained_smoothing(cells: &[Coord], passes: usize) -> Vec<Vector3<f64>> {
    let mut points: Vec<Vector3<f64>> = cells.iter().map(|c| c.cast::<f64>().into()).collect();
    for _ in 0..passes {
        points = solve_constrained_smoothing_pass(cells, Vector3::new(1.0, 1.0, 1.0), &points);
    }
    points
}

// pub fn path_length(path: &[Vector3<f64>]) -> f64 {
//     path.windows(2)
//         .map(|pair| (pair[1] - pair[0]).norm())
//         .sum::<f64>()
// }

pub fn control_waypoints(origin: Vector3<f64>, path: &[Vector3<f64>]) -> Vec<f64> {
    let mut sum = 0.0;
    let mut last = origin;
    let mut wps = Vec::new();
    for p in path {
        let delta = p - last;
        sum += delta.norm();
        wps.push(sum);
        last = *p;
    }

    wps
}

/// Signed distance to the unit cube centered at `coord`
/// Uses SDF for axis-aligned box of side length 1.
#[inline(always)]
pub fn sdf(coord: &Coord, p: &Vector3<f64>) -> f64 {
    // distance from p to centered box half-size = 0.5
    let c = coord.cast::<f64>();
    let d = (p - c).map(|x| x.abs() - 0.5);
    let outside_dist = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0));
    outside_dist.norm()
}
/// Build a clamped uniform knot vector for degree `p`
// fn uniform_knots(num_ctrl: usize, degree: usize) -> Vec<f64> {
//     let n = num_ctrl + degree + 1;
//     (0..n)
//         .map(|i| {
//             if i < degree {
//                 0.0
//             } else if i > num_ctrl {
//                 1.0
//             } else {
//                 (i - degree) as f64 / (num_ctrl - degree) as f64
//             }
//         })
//         .collect()
// }

fn uniform_knots(num_ctrl: usize, degree: usize) -> Vec<f64> {
    // open-clamped: (p+1) zeros, (n_ctrl-p-1) interior, (p+1) ones
    let mut knots = Vec::with_capacity(num_ctrl + degree + 1);
    // start clamp
    for _ in 0..=degree {
        knots.push(0.0);
    }
    // interior
    if num_ctrl > degree + 1 {
        let interior = num_ctrl - degree - 1;
        for i in 1..=interior {
            knots.push((i as f64) / ((interior + 1) as f64));
        }
    }
    // end clamp
    for _ in 0..=degree {
        knots.push(1.0);
    }
    knots
}

pub fn generate_path_fit_spline(control_points: Vec<Vector3<f64>>) -> BSpline<Vector3<f64>, f64> {
    // build the list of control points: the start + each cell’s centroid
    // pick degree ≤ 3, but no higher than (n_ctrl - 1); allow degree 0 for a single point
    let n_ctrl = control_points.len();
    let degree = std::cmp::min(3, n_ctrl.saturating_sub(1));

    // uniform, clamped knot vector
    let knots = uniform_knots(control_points.len(), degree);

    // build and return the spline
    BSpline::new(degree, control_points, knots)
}

/// Lightweight wrapper for runtime evaluation
#[derive(Debug, Clone)]
pub struct Trajectory {
    spline: BSpline<Vector3<f64>, f64>,
    waypoints: Vec<f64>,
    urgencies: Vec<(f64, f64)>,
    yaw_seq: Vec<(f64, f64)>,
    pub progress: f64,
    length: f64,
}

impl Default for Trajectory {
    fn default() -> Self {
        Self {
            spline: BSpline::new(0, Vec::new(), Vec::new()),
            waypoints: Vec::new(),
            urgencies: Vec::new(),
            yaw_seq: Vec::new(),
            progress: 0.0,
            length: 0.0,
        }
    }
}

impl Trajectory {
    /// Generates a spline through the grid path for target times specified in the tuple (coord, time).
    pub fn new(origin: Vector3<i32>, cells: &[(Coord, f64, f64)]) -> Self {
        let (xs, ys, zs) = (Vec::new(), Vec::new(), Vec::new());
        let (mut cells, times, yaw_seq): (Vec<Coord>, Vec<f64>, Vec<f64>) =
            cells
                .into_iter()
                .fold((xs, ys, zs), |(mut xs, mut ys, mut zs), (x, y, z)| {
                    xs.push(*x);
                    ys.push(*y);
                    zs.push(*z);
                    (xs, ys, zs)
                });
        cells.insert(0, origin);

        let path = solve_constrained_smoothing(&cells, 20);

        let waypoints = control_waypoints(origin.cast(), &path);
        let spline = generate_path_fit_spline(path);

        let length = *waypoints.last().unwrap_or(&0.0);
        Self {
            spline,
            urgencies: times
                .iter()
                .enumerate()
                .map(|(i, v)| (waypoints[i + 1], *v))
                .collect(),
            yaw_seq: yaw_seq
                .iter()
                .enumerate()
                .map(|(i, v)| (waypoints[i + 1], *v))
                .collect(),
            progress: 0.0,
            length,
            waypoints,
        }
    }

    pub fn inner(&self) -> &BSpline<Vector3<f64>, f64> {
        &self.spline
    }

    pub fn inner_mut(&mut self) -> &mut BSpline<Vector3<f64>, f64> {
        &mut self.spline
    }

    #[inline(always)]
    pub fn position(&self, t: f64) -> Option<Vector3<f64>> {
        Some(self.inner().point(self.domain_t(t)?))
    }

    #[inline(always)]
    pub fn velocity(&self, t: f64) -> Option<Vector3<f64>> {
        let h = 1e-2;
        let mut t_lo = (t - h).max(0.0);
        let mut t_hi = (t + h).min(1.0);

        t_lo = t_lo.min(t_hi - 2.0 * h);
        t_hi = t_hi.max(t_lo + 2.0 * h);
        let pdiff = self.position(t_hi)? - self.position(t_lo)?;
        if pdiff.norm() > std::f64::EPSILON {
            Some(pdiff.normalize())
        } else {
            Some(Vector3::zeros())
        }
    }

    #[inline(always)]
    pub fn acceleration(&self, t: f64) -> Option<Vector3<f64>> {
        let h = 1e-2;
        let mut t_lo = (t - h).max(0.0);
        let mut t_hi = (t + h).min(1.0);

        t_lo = t_lo.min(t_hi - 2.0 * h);
        t_hi = t_hi.max(t_lo + 2.0 * h);
        let vdiff = self.velocity(t_hi)? - self.velocity(t_lo)?;
        if vdiff.norm() > std::f64::EPSILON {
            Some(vdiff.normalize())
        } else {
            Some(Vector3::zeros())
        }
    }

    /// Returns (position, t) of the waypoint.
    #[inline(always)]
    pub fn waypoints(&self, splits: usize) -> Vec<(f64, Vector3<f64>)> {
        let (min, max) = self.inner().knot_domain();
        let dsplit = (max - min) / splits as f64;

        (0..splits)
            .map(|i| {
                let t = min + dsplit * i as f64;
                (t, self.inner().point(min + dsplit * i as f64))
            })
            .collect()
    }

    pub fn progress_for_move_idx(&self, i: usize) -> Option<f64> {
        self.waypoints.get(i).copied()
    }

    pub fn progress_for_move(&self, m: f64) -> Option<f64> {
        assert!(m >= 0.0);
        if m == 0.0 {
            return Some(0.0);
        }
        let i0 = m.floor() as usize;
        let i1 = m.ceil() as usize;

        if i1 >= self.waypoints.len() {
            return None;
        }

        let p0 = self.waypoints[i0];
        let p1 = self.waypoints[i1];

        let sub = (m % 1.0) * (p1 - p0);

        Some(p0 as f64 + sub)
    }

    pub fn move_for_progress(&self, progress: f64) -> Option<f64> {
        assert!(progress >= 0.0);
        if progress == 0.0 {
            return Some(0.0);
        }
        let (i1, p1) = self
            .waypoints
            .iter()
            .enumerate()
            .find(|(_i, p)| *p > &progress)?;

        assert!(i1 > 0);
        let i0 = i1 - 1;
        let p0 = self.waypoints[i0];

        let sub = (progress - p0) / (p1 - p0);

        Some(i0 as f64 + sub)
    }

    pub fn length(&self) -> f64 {
        self.length
    }

    pub fn domain_t(&self, t: f64) -> Option<f64> {
        let (min, max) = self.spline.knot_domain();
        // Guard against zero-length trajectories
        if self.length() <= std::f64::EPSILON {
            return None;
        }
        let mut dom_t = min + (max - min) * (t / self.length());
        // Many B-spline implementations expect t < max; clamp to an epsilon below max
        if dom_t >= max {
            let eps = 1e-9;
            dom_t = (max - eps).max(min);
        }
        if dom_t < min { None } else { Some(dom_t) }
    }

    pub fn sample_urgencies(&self, t: f64) -> Option<f64> {
        Self::sample_aux(&self.urgencies, t)
    }

    pub fn sample_yaw(&self, t: f64) -> Option<f64> {
        Self::sample_aux(&self.yaw_seq, t)
    }

    fn sample_aux(seq: &[(f64, f64)], t: f64) -> Option<f64> {
        if seq.is_empty() {
            return None;
        }

        // Indexing here is done starting at 1.
        let mut i1 = seq.len();

        for i in 0..seq.len() {
            if seq[i].0 >= t {
                i1 = i;
                break;
            }
        }

        if i1 == 0 {
            Some(seq[i1].1)
        } else if i1 < seq.len() {
            let i0 = i1 - 1;
            let u0 = seq[i0].1;
            let u1 = seq[i1].1;
            let w = (t - seq[i0].0) / (seq[i1].0 - seq[i0].0);

            // Linear interpolation between neighboring samples
            Some(u0 * (1.0 - w) + u1 * w)
        } else {
            None
        }
    }

    pub fn sample_spline(&self, t: f64) -> Option<Vector3<f64>> {
        let dom_t = self.domain_t(t)?;
        Some(self.spline.point(dom_t))
    }
    pub fn sample(&self, t: f64) -> Option<(Vector3<f64>, f64)> {
        let p = self.inner().point(t);
        let u = self.sample_urgencies(t)?;
        Some((p, u))
    }

    pub fn find_nearest_param_in_range(
        &self,
        start: f64,
        end: f64,
        point: &Vector3<f64>,
    ) -> Option<f64> {
        assert!(start >= 0.0);
        // Coarse uniform sampling to find initial guess
        let samples = 100;
        let mut best_t = start;
        let mut best_dist2 = f64::MAX;
        for i in 0..=samples {
            let t = start + (end - start) * (i as f64) / (samples as f64);
            if let Some(p) = self.position(t) {
                let d2 = (p - point).norm_squared();
                if d2 < best_dist2 {
                    best_dist2 = d2;
                    best_t = t;
                }
            }
        }

        // Refine with a ternary search around best_t
        let delta = (end - start) / (samples as f64);
        let mut left = (best_t - delta).max(start);
        let mut right = (best_t + delta).min(end);
        for _ in 0..10 {
            let t1 = left + (right - left) / 3.0;
            let t2 = right - (right - left) / 3.0;
            if let (Some(p1), Some(p2)) = (self.position(t1), self.position(t2)) {
                let d1 = (p1 - point).norm_squared();
                let d2 = (p2 - point).norm_squared();
                if d1 < d2 {
                    right = t2;
                } else {
                    left = t1;
                }
            }
        }

        // Return midpoint of final interval
        Some((left + right) / 2.0)
    }
}

use serde::{Deserialize, Deserializer, Serialize, Serializer};

// Internal plain‐old‐data for Serde round‐trips
#[derive(Serialize, Deserialize)]
struct SplineData {
    degree: usize,
    knots: Vec<f64>,
    ctrl_pts: Vec<[f64; 3]>,
    // In knot domain.
    urgencies: Vec<(f64, f64)>,
    yaw_seq: Vec<(f64, f64)>,
    waypoints: Vec<f64>,
    // In knot domain.
    progress: f64,
    length: f64,
}

impl Serialize for Trajectory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let spline = &self.inner();

        // Extract all data
        let ctrl_pts: Vec<[f64; 3]> = spline.control_points().map(|p| [p.x, p.y, p.z]).collect();
        let knots: Vec<f64> = spline.knots().map(|x| *x).collect();

        // Recover degree = knots.len() - ctrl_pts.len() - 1
        let n_knots = knots.len();
        let n_ctrl = ctrl_pts.len();
        let degree = n_knots.checked_sub(n_ctrl + 1).ok_or_else(|| {
            serde::ser::Error::custom("Invalid BSpline: knot/control-point count mismatch")
        })?;

        let data = SplineData {
            degree,
            knots,
            ctrl_pts,
            urgencies: self.urgencies.clone(),
            yaw_seq: self.yaw_seq.clone(),
            waypoints: self.waypoints.clone(),
            progress: self.progress,
            length: self.length,
        };
        data.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Trajectory {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = SplineData::deserialize(deserializer)?;
        // Rebuild control points
        let cps: Vec<_> = data
            .ctrl_pts
            .into_iter()
            .map(|[x, y, z]| Vector3::new(x, y, z))
            .collect();
        // Validate length: should be cps.len() + degree + 1 == knots.len()
        if data.knots.len() != cps.len() + data.degree + 1 {
            return Err(serde::de::Error::custom(
                "Invalid knot/control‐point lengths",
            ));
        }
        Ok(Trajectory {
            spline: BSpline::new(data.degree, cps, data.knots),
            urgencies: data.urgencies,
            yaw_seq: data.yaw_seq,
            progress: data.progress,
            waypoints: data.waypoints,
            length: data.length,
        })
    }
}
