use std::convert::identity;

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use crate::{
    Agent, VoxelGrid,
    phase::{PhaseGrid, PhaseSolver},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtgNode {
    pub point: Vector3<f64>,
    pub edges: Vec<PtgEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtgEdge {
    pub likelihood: f64,
    pub vel: Vector3<f64>,
    pub acc: Vector3<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PtgSolver {
    half_dims: Vector3<i32>,
    scope: PtgObservationScope,
    add_thresh: f64,
    sim_thresh: f64,

    max_dt_thresh: f64,
    max_dist_thresh: f64,
    min_likelihood_thresh: f64,

    old_snapshot: Option<(f64, VoxelGrid)>,
    graph: PtgGraph,
}

impl PtgSolver {
    pub fn new(
        half_dims: Vector3<i32>,
        scope: PtgObservationScope,
        add_thresh: f64,
        sim_thresh: f64,

        max_dt_thresh: f64,
        max_dist_thresh: f64,
        min_likelihood_thresh: f64,
    ) -> Self {
        Self {
            half_dims,
            scope,
            add_thresh,
            sim_thresh,
            max_dt_thresh,
            max_dist_thresh,
            min_likelihood_thresh,

            old_snapshot: None,
            graph: PtgGraph::new(),
        }
    }
}

impl Default for PtgSolver {
    fn default() -> Self {
        Self::new(
            Vector3::new(15, 15, 15),
            PtgObservationScope::MaxDeltaTime(2.0),
            0.4,
            0.4,
            2.0,
            5.0,
            0.1,
        )
    }
}

impl PhaseSolver for PtgSolver {
    fn latest_timestamp(&self) -> Option<f64> {
        self.graph.head_t()
    }
    fn add_phase_frame(&mut self, agent: &Agent, virtual_world: &VoxelGrid, t: f64) {
        let snapshot = virtual_world.sparse_snapshot_lookup(agent.get_coord(), self.half_dims);
        if let Some((_old_t, old_grid)) = &self.old_snapshot {
            let diff = snapshot.difference_set_from(old_grid);

            let obs: Vec<Vector3<f64>> = diff.coords().into_iter().map(|c| c.cast()).collect();

            self.graph
                .add_observation_set(&obs, t, self.scope, self.add_thresh, self.sim_thresh);
        }

        self.old_snapshot = Some((t, snapshot));
        self.graph.cull(self.max_dist_thresh, self.max_dt_thresh);
    }

    fn gen_phase_grid(&self, _agent: &Agent, time: std::ops::Range<f64>) -> PhaseGrid {
        self.graph
            .generate_grid(time, self.max_dist_thresh, self.min_likelihood_thresh)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PtgGraph {
    nodes: Vec<(f64, Vec<PtgNode>)>,
    head_t: Option<f64>,
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum PtgObservationScope {
    MaxDeltaTime(f64),
    MinTime(f64),
    MaxSteps(usize),
    All,
}

impl PtgGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            head_t: None,
        }
    }

    pub fn add_observation_set(
        &mut self,
        obs: &[Vector3<f64>],
        t: f64,
        scope: PtgObservationScope,
        add_thresh: f64,
        sim_thresh: f64,
    ) {
        if let Some(head_t) = self.head_t {
            assert!(t > head_t);
        }
        let mut new_nodes = Vec::new();
        for new in obs {
            let mut edges = Vec::new();
            for (i, (nt, nodes)) in self.nodes.iter().rev().enumerate() {
                match scope {
                    PtgObservationScope::MinTime(min_t) => {
                        if min_t > *nt {
                            continue;
                        }
                    }
                    PtgObservationScope::MaxDeltaTime(max_dt) => {
                        if t - nt > max_dt {
                            continue;
                        }
                    }
                    PtgObservationScope::MaxSteps(max_steps) => {
                        if i >= max_steps {
                            continue;
                        }
                    }
                    PtgObservationScope::All => {}
                }

                let dt = t - nt;
                for node in nodes {
                    // Now we check if the new point matches one of the direction of an existing edge in one of the nodes.
                    let obs_vel = (*new - node.point) / dt;

                    let mut continued_edges: Vec<Option<PtgEdge>> = Vec::new();

                    for edge in &node.edges {
                        // Calculate what the velocity should be if this edge holds perfectly.
                        let implied_vel = edge.vel + edge.acc * dt;

                        let vel_scale = 1.0 / implied_vel.norm().powf(0.5);
                        let local_likelihood =
                            vector_likelihood(&obs_vel, &implied_vel) * vel_scale;

                        let new_likelihood =
                            bounded_add(local_likelihood * edge.likelihood, local_likelihood);

                        if new_likelihood > add_thresh {
                            // Take arithmetic mean for velocity.
                            let new_vel = (implied_vel + obs_vel) / 2.0;
                            let new_acc = (obs_vel - edge.vel) / dt;
                            // Continue this edge!
                            continued_edges.push(Some(PtgEdge {
                                likelihood: new_likelihood,
                                vel: new_vel,
                                acc: new_acc,
                            }));
                        }
                    }

                    // Add the normal edge.
                    let new_edge = PtgEdge {
                        likelihood: 0.0,
                        vel: obs_vel,
                        acc: Vector3::zeros(),
                    };

                    // Iterate through every combination.
                    for i in 0..continued_edges.len() {
                        let (head, tail) = continued_edges.split_at_mut(i + 1);
                        let a = &mut head[i];
                        for b in tail.iter_mut() {
                            let mut remove_a = false;
                            let mut remove_b = false;
                            if let (Some(a), Some(b)) = (&a, &b) {
                                if vector_likelihood(&a.vel, &b.vel) > sim_thresh {
                                    // Remove the one with the lower likelihood.
                                    if a.likelihood < b.likelihood {
                                        remove_a = true;
                                    } else {
                                        remove_b = true;
                                    }
                                }
                            }
                            if remove_a {
                                *a = None;
                            }
                            if remove_b {
                                *b = None;
                            }
                        }
                    }

                    // Now add the edges.
                    let edges_iter = std::iter::once(new_edge)
                        .chain(continued_edges.into_iter().filter_map(identity));

                    edges.extend(edges_iter);
                }
            }
            new_nodes.push(PtgNode { point: *new, edges });
        }

        self.nodes.push((t, new_nodes));

        self.head_t = Some(t);
    }

    pub fn head_t(&self) -> Option<f64> {
        self.head_t
    }

    pub fn remove_nodes_before(&mut self, min_t: f64) {
        self.nodes.retain(|(t, _)| t > &min_t);
    }

    // Removes any node that would not contribute to a grid generation.
    pub fn cull(&mut self, _max_dist_thresh: f64, max_dt_thresh: f64) {
        if let Some(head_t) = self.head_t {
            self.nodes.retain_mut(|(t, nodes)| {
                let dt = head_t - *t;
                if dt > max_dt_thresh {
                    return false;
                } else if dt < std::f64::EPSILON {
                    return true;
                }
                // nodes.retain(|n| {
                //     !n.edges.iter().all(|edge| {
                //         let p = edge.vel.norm() * dt > max_dist_thresh;
                //         p
                //     })
                // });
                !nodes.is_empty()
            });
        }
    }

    pub fn generate_grid(
        &self,
        time: std::ops::Range<f64>,
        max_dist_thresh: f64,
        min_likelihood_thresh: f64,
    ) -> PhaseGrid {
        let mut grid = PhaseGrid::new();
        if let Some(head_t) = self.head_t {
            for (t, nodes) in &self.nodes {
                let dt = head_t - t;
                for node in nodes {
                    for edge in &node.edges {
                        if edge.likelihood > min_likelihood_thresh {
                            if edge.vel.norm() * dt < max_dist_thresh {
                                // We should consider this edge.
                                let p_min = node.point + edge.vel * (time.start - t);
                                let p_max = node.point + edge.vel * (time.end - t);

                                let weight = edge.likelihood;

                                // Rasterize the motion segment into the grid with given weight.
                                grid.fill_segment(p_min, p_max, weight);
                            }
                        }
                    }
                }
            }
        }
        grid
    }
}

fn vector_likelihood(u: &nalgebra::Vector3<f64>, v: &nalgebra::Vector3<f64>) -> f64 {
    let a = u.norm();
    let b = v.norm();
    if a == 0.0 || b == 0.0 {
        return 0.0;
    }
    let cos = (u.dot(v) / (a * b)).max(0.0); // remove .abs() if sign matters
    let mag = (2.0 * a * b) / (a * a + b * b);
    cos * mag
}

pub fn bounded_add(x: f64, y: f64) -> f64 {
    debug_assert!(x >= 0.0 && x < 1.0 && y >= 0.0 && y < 1.0);
    (x + y) / (1.0 + x * y)
}
