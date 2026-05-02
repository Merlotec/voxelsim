use glsl_layout::Std140;
use glsl_layout::{Uniform, float, uint, vec3};
use nalgebra::Vector3;
use wgpu::util::DeviceExt;

#[derive(Copy, Clone, PartialEq, Uniform)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct NoiseParams {
    /// cycles / world unit for x,y,z
    pub spatial_freq: vec3,
    /// cycles / unit for sx,sy,sz
    pub seed_freq: vec3,
    /// your continuous seed vector
    pub seed_vec: vec3,
    pub lacunarity: float, // e.g. 2.0
    pub gain: float,       // e.g. 0.5
    pub octaves: uint,     // e.g. 4
    pub enabled: uint,     // still need one uint of padding
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self::default_with_seed([0.0, 0.0, 0.0].into())
    }
}

impl NoiseParams {
    pub fn default_with_seed(seed: Vector3<f32>) -> Self {
        Self {
            spatial_freq: vec3::from([0.5, 0.5, 0.5]),
            seed_freq: vec3::from([0.2, 0.2, 0.2]),
            seed_vec: vec3::from([seed.x, seed.y, seed.z]),
            lacunarity: 2.0,
            gain: 0.5,
            octaves: 4,
            enabled: 1,
        }
    }

    pub fn none() -> Self {
        Self {
            enabled: 0,
            ..Default::default()
        }
    }
}

pub struct NoiseBuffer {
    pub layout: wgpu::BindGroupLayout,
    pub group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}

impl NoiseBuffer {
    pub fn create_buffer(device: &wgpu::Device, noise: NoiseParams) -> Self {
        let noise_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Noise buffer"),
            contents: noise.std140().as_raw(),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // The atomic flag buffer, one u32 per instance.

        let layout = Self::layout(device);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: noise_buffer.as_entire_binding(),
            }],
        });
        Self {
            buffer: noise_buffer,
            layout,
            group,
        }
    }

    pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Rasterizer Bind Group Layout"),
            entries: &[
                // Output Buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
}
