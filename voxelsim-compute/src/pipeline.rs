use std::ops::DerefMut;

use crate::buf::StagingBufferPool;
use crate::rasterizer::RasterizerState;
use crate::rasterizer::camera::CameraMatrix;
use crate::rasterizer::noise::NoiseParams;
use crate::rasterizer::{CellInstance, InstanceBuffer};
use bytemuck::{Pod, Zeroable};
use nalgebra::Vector2;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use voxelsim::uncertainty::UncertaintyWorld;
use voxelsim::{Cell, Coord, VoxelGrid, VoxelSet};

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Default, Pod, Zeroable)]
pub struct FilterCoord {
    coord: Coord,
    _p0: i32,
}

pub struct State {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub size: Vector2<u32>,
    pub rasterizer_state: RasterizerState,
    pub filter_buffer: InstanceBuffer,
    pub dynamic_buffer: InstanceBuffer,
    pub staging_pool: StagingBufferPool,
    // Reused scratch space for instance uploads to avoid per-frame allocations
    filter_instance_scratch: Vec<crate::rasterizer::CellInstance>,
    dynamic_instance_scratch: Vec<crate::rasterizer::CellInstance>,
}

impl State {
    const DYNAMIC_INSTANCE_SIZE: usize = 1000;
    // Creating some of the wgpu types requires async code
    pub async fn create(world: &VoxelGrid, view_size: Vector2<u32>, noise: NoiseParams) -> Self {
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU

        let (queue, device) = {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            // The adapter is a handle to a physical graphics card.
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .unwrap();

            // The device is the logical connection to the GPU, used to create resources.
            // The queue is used to submit command buffers to the GPU.
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::PUSH_CONSTANTS
                        | wgpu::Features::CLEAR_TEXTURE, // Add features you need here
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 80, // Size of mat4x4<f32>
                        ..wgpu::Limits::default()
                    },
                    label: None,
                    trace: wgpu::Trace::Off,
                    memory_hints: wgpu::MemoryHints::Performance,
                })
                .await
                .unwrap();

            (queue, device)
        };
        let rasterizer_state =
            RasterizerState::create(&device, world, [view_size.x, view_size.y], noise);

        let filter_buffer =
            CellInstance::create_instance_buffer_uninit(&device, world.cells().len());
        let dynamic_buffer =
            CellInstance::create_instance_buffer_uninit(&device, Self::DYNAMIC_INSTANCE_SIZE);

        let staging_pool = StagingBufferPool::new(device.clone());

        Self {
            device,
            queue,
            size: view_size,
            rasterizer_state,
            filter_buffer,
            dynamic_buffer,
            staging_pool,
            filter_instance_scratch: Vec::with_capacity(world.cells().len()),
            dynamic_instance_scratch: Vec::with_capacity(Self::DYNAMIC_INSTANCE_SIZE),
        }
    }

    pub async fn run(
        &mut self,
        camera_matrix: &CameraMatrix,
        filter_world: &VoxelGrid,
        dynamic_world: &VoxelGrid,
        timestamp: f64,
        cam_dir_world: nalgebra::Vector3<f64>,
    ) -> Result<WorldChangeset, wgpu::SurfaceError> {
        let profile_enabled = std::env::var("VXS_COMPUTE_PROFILE").is_ok();
        let total_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // Get the current texture to render to from the swap chain.
        //let output = self.surface.get_current_texture()?;
        //let view = output
        //    .texture
        //    .create_view(&wgpu::TextureViewDescriptor::default());

        // === PHASE 1: PARALLEL FRUSTUM CULLING ===
        let culling_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Clear filter buffers and prepare filter buffer FIRST (needed for filter culling)
        let _buffer_prep_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Compute filter instance count and ensure filter buffers are large enough before clearing
        let filter_buffer_len = filter_world.cells().len() as u32;
        self.rasterizer_state
            .ensure_filter_capacity(&self.device, filter_buffer_len as usize);

        // Start filter clear after any potential resize
        self.rasterizer_state
            .filter
            .clear_buffers(&self.device, &self.queue);

        let buffer_write_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // filter_buffer_len computed earlier
        // Ensure filter buffer can hold this frame's instances
        if (filter_buffer_len as usize) > (self.filter_buffer.len as usize) {
            let new_len = filter_buffer_len as usize;
            self.filter_buffer =
                CellInstance::create_instance_buffer_uninit(&self.device, new_len);
            if self.filter_instance_scratch.capacity() < new_len {
                self.filter_instance_scratch
                    .reserve(new_len - self.filter_instance_scratch.capacity());
            }
        }

        CellInstance::write_instance_buffer_with_scratch(
            &self.queue,
            &self.filter_buffer,
            filter_world,
            &mut self.filter_instance_scratch,
        );

        let dynamic_buffer_len = dynamic_world.cells().len() as u32;
        // Ensure dynamic buffer can hold this frame's dynamic instances
        if (dynamic_buffer_len as usize) > (self.dynamic_buffer.len as usize) {
            // Grow to the next power of two to avoid frequent reallocations
            let mut new_len = dynamic_buffer_len as usize;
            new_len = new_len.next_power_of_two();
            self.dynamic_buffer =
                CellInstance::create_instance_buffer_uninit(&self.device, new_len);
            if self.dynamic_instance_scratch.capacity() < new_len {
                self.dynamic_instance_scratch
                    .reserve(new_len - self.dynamic_instance_scratch.capacity());
            }
        }

        CellInstance::write_instance_buffer_with_scratch(
            &self.queue,
            &self.dynamic_buffer,
            dynamic_world,
            &mut self.dynamic_instance_scratch,
        );
        // Keep the dynamic buffer's logical len in sync with this frame's data
        self.dynamic_buffer.len = dynamic_buffer_len;
        if profile_enabled {
            println!("💾 Regular filter buffer upload from VirtualGrid");
        }

        if let Some(start) = buffer_write_start {
            let buffer_write_time = start.elapsed();
            println!(
                "💾 Filter Buffer Write: {:.2}ms",
                buffer_write_time.as_secs_f64() * 1000.0
            );
        }

        // Encode BOTH culling passes into a single encoder and submit once
        let mut cull_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Frustum Culling Encoder (Combined)"),
                });

        self.rasterizer_state
            .encode_frustum_culling(&mut cull_encoder, &self.queue, camera_matrix);

        self.rasterizer_state.encode_filter_frustum_culling(
            &mut cull_encoder,
            &self.queue,
            &self.device,
            camera_matrix,
            &self.filter_buffer.buf,
            filter_buffer_len,
        );

        let culling_submissions = self.queue.submit(std::iter::once(cull_encoder.finish()));

        if let Some(start) = culling_start {
            let culling_time = start.elapsed();
            println!(
                "✂️  Parallel Culling (Main + Filter): {:.2}ms",
                culling_time.as_secs_f64() * 1000.0
            );
        }

        // === PHASE 2: GPU Command Encoding ===

        if profile_enabled {
            println!("    ✅ Using optimized traditional culling with readbacks");
        }

        // Start GPU Command Encoding timer right before actual encoding begins
        let encoding_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // A command encoder builds a command buffer that we can send to the GPU.
        let main_encode_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Rasterizer Encoder"),
            });

        // Encode main rasterizer pass using culled instances with indirect draw
        self.rasterizer_state.encode_rasterizer(
            &mut encoder,
            &self.dynamic_buffer,
            camera_matrix,
            true,
            None,
        );

        let main_submit_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };
        self.queue.submit(std::iter::once(encoder.finish()));

        if let Some(start) = main_submit_start {
            let main_submit_time = start.elapsed();
            println!(
                "📤 Main Submission: {:.2}ms",
                main_submit_time.as_secs_f64() * 1000.0
            );
        }

        if let Some(start) = main_encode_start {
            let main_encode_time = start.elapsed();
            println!(
                "🎨 Main Render Encoding: {:.2}ms",
                main_encode_time.as_secs_f64() * 1000.0
            );
        }

        //output.present();

        // Create filter encoder with culling results
        let filter_encode_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let mut filter_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Filter Encoder"),
                });

        self.rasterizer_state.encode_filter(
            &mut filter_encoder,
            camera_matrix,
            &self.filter_buffer.buf,
            filter_buffer_len,
            true,
            None,
        );

        // Submit the filter rendering commands and capture the submission index
        let filter_submit_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let render_submission = self.queue.submit(std::iter::once(filter_encoder.finish()));

        if let Some(start) = filter_submit_start {
            let filter_submit_time = start.elapsed();
            println!(
                "📤 Filter Submission: {:.2}ms",
                filter_submit_time.as_secs_f64() * 1000.0
            );
        }

        if let Some(start) = filter_encode_start {
            let filter_encode_time = start.elapsed();
            println!(
                "🔍 Filter Render Encoding: {:.2}ms",
                filter_encode_time.as_secs_f64() * 1000.0
            );
        }

        if let Some(start) = encoding_start {
            let encoding_time = start.elapsed();
            println!(
                "📋 GPU Command Encoding: {:.2}ms",
                encoding_time.as_secs_f64() * 1000.0
            );
        }

        // === PHASE 2: Prepare Read Operations ===
        let prepare_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Prepare both operations without polling
        let texture_op = self
            .staging_pool
            .prepare_texture_extraction(&self.queue, &self.rasterizer_state.render_target.texture);

        let filter_op = self.staging_pool.prepare_buffer_read(
            &self.queue,
            &self.rasterizer_state.filter.output_buffer,
            true,
        );

        if let Some(start) = prepare_start {
            let prepare_time = start.elapsed();
            println!(
                "⚙️  Read Operations Setup: {:.2}ms",
                prepare_time.as_secs_f64() * 1000.0
            );
        }

        // Collect submission indices before moving the operations
        let texture_submission = texture_op.submission_index.clone();
        let filter_submission = filter_op.submission_index.clone();

        // === PHASE 3: GPU Polling ===
        let polling_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Wait for all specific submissions to complete
        let all_submissions = vec![
            culling_submissions,
            render_submission,
            texture_submission,
            filter_submission,
        ];

        self.staging_pool.wait_for_submissions(&all_submissions);

        if let Some(start) = polling_start {
            let polling_time = start.elapsed();
            println!(
                "⏳ GPU Submission Polling: {:.2}ms",
                polling_time.as_secs_f64() * 1000.0
            );
        }

        // === PHASE 4: Parallel Buffer Reads ===
        let parallel_read_start = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Execute both read operations in parallel
        let results = self
            .staging_pool
            .execute_batched_reads_parallel(vec![texture_op, filter_op])
            .await;

        if let Some(start) = parallel_read_start {
            let parallel_read_time = start.elapsed();
            println!(
                "🚀 Parallel Buffer Reads: {:.2}ms",
                parallel_read_time.as_secs_f64() * 1000.0
            );
        }

        // === PHASE 5: Data Processing ===
        let processing_start = std::time::Instant::now();

        // Extract results
        let to_insert_raw = results[0]
            .as_ref()
            .unwrap_or_else(|e| panic!("Texture extraction failed: {}", e))
            .clone();
        let filter_data = results[1]
            .as_ref()
            .unwrap_or_else(|e| panic!("Filter buffer read failed: {}", e))
            .clone();

        // Process the raw texture data
        let to_insert: Vec<(voxelsim::Coord, voxelsim::Cell)> = unsafe {
            let data_slice = std::slice::from_raw_parts::<(voxelsim::Coord, voxelsim::Cell)>(
                to_insert_raw.as_ptr() as *const (voxelsim::Coord, voxelsim::Cell),
                to_insert_raw.len() / std::mem::size_of::<(voxelsim::Coord, voxelsim::Cell)>(),
            );
            data_slice.to_vec()
        };

        // Process the filter data
        let count = u32::from_le_bytes(filter_data[0..4].try_into().unwrap()) as usize;
        // Due to std140 we must start offset at 16
        let base = 16usize;
        let output_data_slice: &[FilterCoord] = bytemuck::cast_slice(
            &filter_data[base..base + count * std::mem::size_of::<FilterCoord>()],
        );
        let to_remove = output_data_slice.to_vec();

        if profile_enabled {
            let processing_time = processing_start.elapsed();
            println!(
                "🔄 Data Processing: {:.2}ms",
                processing_time.as_secs_f64() * 1000.0
            );
        }

        if profile_enabled {
            if let Some(start) = total_start {
                let total_time = start.elapsed();
                println!(
                    "🏁 TOTAL PIPELINE: {:.2}ms",
                    total_time.as_secs_f64() * 1000.0
                );
                println!(
                    "📊 Items: {} to_insert, {} to_remove",
                    to_insert.len(),
                    to_remove.len()
                );
                println!("═══════════════════════════════════════");
            }
        }

        Ok(WorldChangeset {
            to_remove,
            to_insert,
            timestamp,
            cam_pos: nalgebra::Vector3::new(
                camera_matrix.pos[0] as f64,
                camera_matrix.pos[1] as f64,
                camera_matrix.pos[2] as f64,
            ),
            cam_dir: cam_dir_world,
        })
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct WorldChangeset {
    pub to_insert: Vec<(Coord, Cell)>,
    pub to_remove: Vec<FilterCoord>,
    pub timestamp: f64,
    pub cam_pos: nalgebra::Vector3<f64>,
    pub cam_dir: nalgebra::Vector3<f64>,
}

impl WorldChangeset {
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }
    pub fn update_world(&self, world: &mut VoxelGrid) {
        // Backward-compatible path: no uncertainty gating
        self.to_insert.par_iter().for_each(|(coord, cell)| {
            if !cell.is_empty() {
                world.cells().insert(*coord, *cell);
            }
        });
        self.to_remove.par_iter().for_each(|coord| {
            world.cells().remove(&coord.coord);
        });
    }
    pub fn update_world_with_difference(&self, world: &mut VoxelGrid) -> VoxelSet {
        let added = VoxelSet::new();
        // Backward-compatible path: no uncertainty gating
        self.to_insert.par_iter().for_each(|(coord, cell)| {
            if !cell.is_empty() {
                if world.cells().insert(*coord, *cell).is_none() {
                    added.cells().insert(*coord);
                }
            }
        });
        self.to_remove.par_iter().for_each(|coord| {
            world.cells().remove(&coord.coord);
        });

        added
    }

    /// Update world conditioned on uncertainty derived from camera pose: only apply a change
    /// if the observation uncertainty (based on distance and FOV) is below the current sampled
    /// field value at the cell position.
    pub fn update_world_with_uncertainty(
        &self,
        world: &mut VoxelGrid,
        unc_world: &UncertaintyWorld,
    ) {
        // Use camera pose from this changeset for sampling/observation uncertainty
        let cam_pos = self.cam_pos;
        let in_dir = nalgebra::Unit::new_normalize(self.cam_dir);
        let half_fov = 0.4_f64; // approx half vertical FOV in radians
        let cos_thresh = half_fov.cos();
        let max_dist = 40.0_f64; // limit range considered

        self.to_insert.par_iter().for_each(|(coord, cell)| {
            if cell.is_empty() {
                return;
            }
            let pos = nalgebra::Vector3::new(coord.x as f64, coord.y as f64, coord.z as f64);
            let dvec = pos - cam_pos;
            let dist = dvec.norm();
            if dist <= 1e-6 || dist > max_dist {
                return;
            }
            let dir = nalgebra::Unit::new_normalize(dvec);
            if dir.dot(&in_dir) < cos_thresh {
                return;
            }
            // Observation uncertainty increases with distance (0 near → 1 at max)
            let obs_uncertainty = dist / max_dist;
            if let Some(current) = unc_world.sample_field(pos, in_dir) {
                if obs_uncertainty < current {
                    world.cells().insert(*coord, *cell);
                }
            }
        });

        self.to_remove.par_iter().for_each(|coord| {
            let pos = nalgebra::Vector3::new(
                coord.coord.x as f64,
                coord.coord.y as f64,
                coord.coord.z as f64,
            );
            let dvec = pos - cam_pos;
            let dist = dvec.norm();
            if dist <= 1e-6 || dist > max_dist {
                return;
            }
            let dir = nalgebra::Unit::new_normalize(dvec);
            if dir.dot(&in_dir) < cos_thresh {
                return;
            }
            let obs_uncertainty = dist / max_dist;
            if let Some(current) = unc_world.sample_field(pos, in_dir) {
                if obs_uncertainty < current {
                    world.cells().remove(&coord.coord);
                }
            }
        });
    }

    pub fn update_filter_world(&self, filter_world: &crate::FilterWorld) {
        let mut vgrid = filter_world.world.lock().unwrap();
        self.update_world(vgrid.deref_mut());
        filter_world
            .frames
            .lock()
            .unwrap()
            .retain(|x| *x != self.timestamp);
        *filter_world.timestamp.lock().unwrap() = Some(self.timestamp);
    }

    pub fn update_filter_world_with_difference(
        &self,
        filter_world: &crate::FilterWorld,
    ) -> VoxelSet {
        let mut vgrid = filter_world.world.lock().unwrap();
        let added = self.update_world_with_difference(vgrid.deref_mut());
        filter_world
            .frames
            .lock()
            .unwrap()
            .retain(|x| *x != self.timestamp);
        *filter_world.timestamp.lock().unwrap() = Some(self.timestamp);
        added
    }

    /// Same as `update_filter_world`, but with uncertainty gating. Uses `uncertainty_level`
    /// as the candidate level to compare against the field's sampled level.
    pub fn update_filter_world_with_uncertainty(
        &self,
        filter_world: &crate::FilterWorld,
        unc_world: &UncertaintyWorld,
    ) {
        let mut vgrid = filter_world.world.lock().unwrap();
        self.update_world_with_uncertainty(vgrid.deref_mut(), unc_world);
        filter_world
            .frames
            .lock()
            .unwrap()
            .retain(|x| *x != self.timestamp);
        *filter_world.timestamp.lock().unwrap() = Some(self.timestamp);
    }
}
