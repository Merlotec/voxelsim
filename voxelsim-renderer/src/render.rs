use bevy::platform::collections::HashMap;
use bevy_panorbit_camera::PanOrbitCamera;
use crossbeam_channel::Receiver;

use voxelsim::{Agent, Cell, Coord, PovData, VoxelGrid};

use bevy::prelude::*;

use crate::pov::PovCamera;

#[derive(Component)]
pub struct CellComponent {
    pub coord: Coord,
    pub value: Cell,
}

#[derive(Component)]
pub struct VirtualCellComponent {
    pub coord: Coord,
    pub value: Cell,
}

#[derive(Component)]
pub struct DynamicCellComponent {
    pub coord: Coord,
    pub weight: f32,
}

#[derive(Component)]
pub struct AgentComponent {
    pub agent: Agent,
}

#[derive(Component)]
pub struct ActionCell {
    pub coord: Coord,
}

#[derive(Component)]
pub struct OriginCell {
    pub coord: Coord,
}

#[derive(Resource)]
pub struct QuitReceiver(pub Receiver<()>);

#[derive(Resource)]
pub struct WorldReceiver(pub Receiver<VoxelGrid>);

#[derive(Resource)]
pub struct AgentReceiver(pub Receiver<HashMap<usize, Agent>>);

#[derive(Resource)]
pub struct PovNum(pub u16);

#[derive(Resource)]
pub struct PovReceiver(pub Receiver<PovData>);

#[derive(Resource)]
pub struct FocusedAgent(pub usize);

#[derive(Resource)]
pub struct CellAssets {
    pub cube_mesh: Handle<Mesh>,
    pub drone_mesh: Handle<Mesh>,
    pub action_mesh: Handle<Mesh>,
    pub drone_body_mat: Handle<StandardMaterial>,
    pub sparse_mat: Handle<StandardMaterial>,
    pub ground_mat: Handle<StandardMaterial>,
    pub solid_mat: Handle<StandardMaterial>,
    pub target_mat: Handle<StandardMaterial>,
    pub interest_mat: Handle<StandardMaterial>,
    pub drone_mat: Handle<StandardMaterial>,
    pub action_mat: Handle<StandardMaterial>,
    pub action_origin_mat: Handle<StandardMaterial>,
    pub drone_scene: Option<Handle<Scene>>,
    pub dynamic_mats: Vec<Handle<StandardMaterial>>, // pre-binned by opacity
}

impl CellAssets {
    pub fn material_for_cell(&self, cell: &Cell) -> Handle<StandardMaterial> {
        if cell.contains(Cell::INTEREST) {
            return self.interest_mat.clone();
        }
        if cell.contains(Cell::TARGET) {
            return self.target_mat.clone();
        }
        if cell.contains(Cell::FILLED) {
            if cell.contains(Cell::GROUND) {
                self.ground_mat.clone()
            } else {
                self.solid_mat.clone()
            }
        } else if cell.contains(Cell::SPARSE) {
            self.sparse_mat.clone()
        } else {
            self.drone_mat.clone()
        }
    }

    pub fn material_for_weight(&self, weight: f64) -> Handle<StandardMaterial> {
        // weight in [0,1] mapped to bins
        let w = weight.clamp(0.0, 1.0);
        let bins = (self.dynamic_mats.len().saturating_sub(1)) as f64;
        let idx = (w * bins).round() as usize;
        self.dynamic_mats[idx].clone()
    }

    pub fn spawn_agent_with_model(
        &self,
        commands: &mut Commands,
        agent: Agent,
        transform: Transform,
    ) {
        if let Some(ref drone_scene) = self.drone_scene {
            // Spawn GLTF scene
            commands.spawn((
                AgentComponent { agent },
                SceneRoot(drone_scene.clone()),
                transform,
            ));
        } else {
            // Fallback to torus mesh
            commands.spawn((
                AgentComponent { agent },
                Mesh3d(self.drone_mesh.clone()),
                MeshMaterial3d(self.drone_body_mat.clone()),
                transform,
            ));
        }
    }
}

pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let drone_body_mat = materials.add(Color::srgb(1.0, 0.5, 0.5));
    let sparse_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.3, 0.8, 0.1, 0.8),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let ground_mat = materials.add(Color::srgb(0.3, 0.3, 0.3));
    let solid_mat = materials.add(Color::srgb(0.4, 0.3, 0.3));
    let target_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.0, 0.0, 0.8),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let interest_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.0, 0.0, 0.9),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let drone_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.3, 0.1, 0.8, 0.3),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    let action_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.7, 0.7, 0.9, 0.2),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    let action_origin_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.9, 0.9, 0.9, 0.1),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    let cube_mesh = meshes.add(Cuboid::default());
    let drone_mesh = meshes.add(Torus::default());

    // Try to load drone GLTF
    // In WASM, we can't check if file exists, so always try to load
    // The AssetServer will handle missing files gracefully
    #[cfg(not(target_arch = "wasm32"))]
    let drone_scene = if std::path::Path::new("assets/drone.gltf").exists() {
        Some(asset_server.load("drone.gltf#Scene0"))
    } else {
        None
    };

    #[cfg(target_arch = "wasm32")]
    let drone_scene = None; // For demo, just use fallback mesh

    // Precompute dynamic materials with varying opacity
    let mut dynamic_mats: Vec<Handle<StandardMaterial>> = Vec::new();
    let bins = 20usize; // 21 levels (0..=20)
    for i in 0..=bins {
        let w = i as f32 / bins as f32;
        // keep a consistent hue, varying alpha with weight
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgba(0.2, 0.6, 1.0, w.max(0.05)),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.6,
            metallic: 0.0,
            ..default()
        });
        dynamic_mats.push(mat);
    }

    commands.insert_resource(CellAssets {
        cube_mesh: cube_mesh.clone(),
        drone_mesh,
        action_mesh: cube_mesh.clone(),
        drone_body_mat,
        sparse_mat,
        ground_mat,
        solid_mat,
        drone_mat,
        target_mat,
        interest_mat,
        action_mat,
        action_origin_mat,
        drone_scene,
        dynamic_mats,
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 100.0,
            ..Default::default()
        },
        Transform::from_xyz(8.0, 16.0, 8.0),
    ));
    commands.insert_resource(AmbientLight {
        brightness: 1200.0,
        affects_lightmapped_meshes: true,
        ..Default::default()
    });

    // Set clear color to white
    commands.insert_resource(ClearColor(Color::BLACK));

    // Spawn PanOrbitCamera
    commands.spawn((
        PanOrbitCamera::default(),
        Transform::from_xyz(0.0, 7., 14.0).looking_at(Vec3::new(0., 0., 0.), Vec3::Y),
    ));

    // Spawn POV Camera (initially inactive)
    commands.spawn((
        PovCamera,
        Camera3d::default(),
        Camera {
            is_active: false,
            ..default()
        },
        Transform::from_xyz(0.0, 7., 14.0).looking_at(Vec3::new(0., 0., 0.), Vec3::Y),
    ));

    // Dedicated UI camera to always render HUD on top
    commands.spawn((
        Camera2d::default(),
        Camera {
            // Ensure UI draws over 3D cameras
            order: 100,
            ..default()
        },
    ));
}
