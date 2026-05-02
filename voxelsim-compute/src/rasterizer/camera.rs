use nalgebra::{Matrix4, Vector3};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

// The CameraBinding struct has been removed entirely.

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraMatrix {
    pub view_proj: [[f32; 4]; 4],
    pub pos: [f32; 3],
    pub _padding: f32, // Align to 16 bytes for WGSL vec3<f32>
}

impl Default for CameraMatrix {
    fn default() -> Self {
        // Z-up coordinate system: position camera to see the voxel range
        let voxel_center = Vector3::<f32>::new(100.0, 100.0, 15.0); // Center of voxel data
        let camera_pos = Vector3::<f32>::new(80.0, 80.0, 50.0);     // Offset and above the voxels
        
        let v = nalgebra_glm::look_at_rh(
            &camera_pos,                             // Camera position
            &voxel_center,                          // Look at voxel center
            &Vector3::<f32>::new(0.0, 0.0, 1.0),    // Up vector is +Z
        );
        let p = nalgebra_glm::perspective(1.4, 0.8, 1.0, 200.0);
        
        Self::from_view_proj(v, p)
    }
}

impl CameraMatrix {
    pub fn from_view_proj(view: Matrix4<f32>, proj: Matrix4<f32>) -> Self {
        let pos = Self::camera_pos_fast(&view);
        Self {
            view_proj: (proj * view).into(),
            pos: pos.into(),
            _padding: 0.0,
        }
    }

    fn camera_pos_fast(view: &Matrix4<f32>) -> Vector3<f32> {
        // Grab the 3×3 rotation block
        let rot3 = view.fixed_view::<3, 3>(0, 0).into_owned();
        // Grab the translation column (3×1)
        let trans = view.fixed_view::<3, 1>(0, 3).into_owned();
        // p_cam = - Rᵀ * t
        -(rot3.transpose() * trans)
    }

    /// Extract frustum planes from the view-projection matrix
    /// Returns 6 planes as [left, right, bottom, top, near, far]
    /// Each plane is vec4(a, b, c, d) where ax + by + cz + d = 0
    pub fn extract_frustum_planes(&self) -> FrustumPlanes {
        let m = Matrix4::from(self.view_proj);
        
        // Only print debug info when environment variable is set
        if std::env::var("DEBUG_FRUSTUM").is_ok() {
            println!("🎥 Camera position: [{:.1}, {:.1}, {:.1}]", self.pos[0], self.pos[1], self.pos[2]);
        }
        
        // Extract frustum planes using standard method
        // For a row-major matrix, the planes are:
        let left = [
            m[(0, 3)] + m[(0, 0)],  // m41 + m11
            m[(1, 3)] + m[(1, 0)],  // m42 + m12
            m[(2, 3)] + m[(2, 0)],  // m43 + m13
            m[(3, 3)] + m[(3, 0)]   // m44 + m14
        ];
        
        let right = [
            m[(0, 3)] - m[(0, 0)],  // m41 - m11
            m[(1, 3)] - m[(1, 0)],  // m42 - m12
            m[(2, 3)] - m[(2, 0)],  // m43 - m13
            m[(3, 3)] - m[(3, 0)]   // m44 - m14
        ];
        
        let bottom = [
            m[(0, 3)] + m[(0, 1)],  // m41 + m21
            m[(1, 3)] + m[(1, 1)],  // m42 + m22
            m[(2, 3)] + m[(2, 1)],  // m43 + m23
            m[(3, 3)] + m[(3, 1)]   // m44 + m24
        ];
        
        let top = [
            m[(0, 3)] - m[(0, 1)],  // m41 - m21
            m[(1, 3)] - m[(1, 1)],  // m42 - m22
            m[(2, 3)] - m[(2, 1)],  // m43 - m23
            m[(3, 3)] - m[(3, 1)]   // m44 - m24
        ];
        
        let near = [
            m[(0, 3)] + m[(0, 2)],  // m41 + m31
            m[(1, 3)] + m[(1, 2)],  // m42 + m32
            m[(2, 3)] + m[(2, 2)],  // m43 + m33
            m[(3, 3)] + m[(3, 2)]   // m44 + m34
        ];
        
        let far = [
            m[(0, 3)] - m[(0, 2)],  // m41 - m31
            m[(1, 3)] - m[(1, 2)],  // m42 - m32
            m[(2, 3)] - m[(2, 2)],  // m43 - m33
            m[(3, 3)] - m[(3, 2)]   // m44 - m34
        ];
        
        // Normalize planes (optional but recommended for numerical stability)
        let normalize_plane = |plane: [f32; 4]| -> [f32; 4] {
            let normal_len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
            if normal_len > 0.0 {
                [plane[0] / normal_len, plane[1] / normal_len, plane[2] / normal_len, plane[3] / normal_len]
            } else {
                plane
            }
        };
        
        let planes = [
            normalize_plane(left),
            normalize_plane(right),
            normalize_plane(bottom),
            normalize_plane(top),
            normalize_plane(near),
            normalize_plane(far),
        ];
        
        // Debug: Print extracted planes
        if std::env::var("DEBUG_FRUSTUM").is_ok() {
            let plane_names = ["left", "right", "bottom", "top", "near", "far"];
            println!("Frustum Planes:");
            for (name, plane) in plane_names.iter().zip(planes.iter()) {
                println!("  {}: [{:.3}, {:.3}, {:.3}, {:.3}]", name, plane[0], plane[1], plane[2], plane[3]);
            }
        }
        
        FrustumPlanes { planes }
    }
    // The buffer creation and writing methods have been removed.
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrustumPlanes {
    pub planes: [[f32; 4]; 6], // 6 frustum planes: left, right, bottom, top, near, far
}
