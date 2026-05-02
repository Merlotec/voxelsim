use std::ops::{Deref, DerefMut};

use bevy::platform::collections::HashMap;
use bevy::render::RenderPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use crossbeam_channel::{Receiver, Sender};
use nalgebra::Vector3;
use voxelsim::trajectory::Trajectory;
use voxelsim::viewport::{CameraOrientation, CameraProjection};

#[cfg(not(target_arch = "wasm32"))]
use crate::network::NetworkSubscriber;

#[cfg(target_arch = "wasm32")]
use crate::network_wasm::NetworkSubscriber;
use crate::render::{
    self, ActionCell, AgentComponent, AgentReceiver, CellAssets, CellComponent,
    DynamicCellComponent, FocusedAgent, OriginCell, PovNum, PovReceiver, QuitReceiver,
    VirtualCellComponent, WorldReceiver,
};
use voxelsim::{Action, Agent, MoveDir, PovData, VoxelGrid};

use bevy::app::AppExit;
use bevy::prelude::*;

use crate::convert::*;

#[derive(Resource)]
pub struct CameraMode {
    pub is_pov: bool,
}

#[derive(Component)]
pub struct PovCamera;

#[derive(Component)]
struct TelemetryText;

#[derive(Resource, Default, Debug, Clone, Copy)]
struct PovInfo {
    voxel_count: usize,
    dynamic_count: usize,
    agent_id: usize,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_pov_server(port_offset: u16) {
    let (mut pov_sub, pov_receiver) = NetworkSubscriber::<PovData>::new(
        std::env::var("VXS_POV_ADDR").unwrap_or("0.0.0.0".to_string()),
        std::env::var("VXS_POV_PORT")
            .ok()
            .and_then(|x| x.parse::<u16>().ok())
            .unwrap_or(8090)
            + port_offset,
    );

    let (mut agent_sub, agent_receiver) = NetworkSubscriber::<HashMap<usize, Agent>>::new(
        std::env::var("VXS_AGENT_POV_ADDR").unwrap_or("0.0.0.0".to_string()),
        std::env::var("VXS_AGENT_POV_PORT")
            .ok()
            .and_then(|x| x.parse::<u16>().ok())
            .unwrap_or(9090)
            + port_offset,
    );

    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            pov_sub.start().await;
            agent_sub.start().await;
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        });
    });

    let (gui_sender, gui_receiver) = crossbeam_channel::unbounded::<GuiCommand>();
    let (quit_sender, quit_receiver) = crossbeam_channel::unbounded::<()>();

    begin_render(
        port_offset,
        pov_receiver,
        agent_receiver,
        gui_sender,
        quit_receiver,
    );
}

pub enum GuiCommand {
    RegenerateWorld,
    MoveAgentA,
    MoveAgentB,
    MoveAgentC,
    MoveAgentD,
}

pub fn begin_render(
    num: u16,
    pov_receiver: Receiver<PovData>,
    agent_receiver: Receiver<HashMap<usize, Agent>>,
    gui_sender: Sender<GuiCommand>,
    quit_receiver: Receiver<()>,
) {
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: format!("Virtual POV View ({})", num),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .set(RenderPlugin {
                    render_creation: bevy::render::settings::RenderCreation::Automatic(
                        bevy::render::settings::WgpuSettings {
                            backends: Some(bevy::render::settings::Backends::all()),
                            ..Default::default()
                        },
                    ),
                    ..Default::default()
                }),
            PanOrbitCameraPlugin,
        ))
        .add_systems(Startup, (render::setup, setup_pov_ui))
        .add_systems(
            Update,
            (
                synchronise_world,
                centre_camera_system,
                update_pov_ui_text,
                quit_system,
            ),
        )
        .insert_resource(PovReceiver(pov_receiver))
        .insert_resource(PovNum(num))
        .insert_resource(AgentReceiver(agent_receiver))
        .insert_resource(FocusedAgent(0))
        .insert_resource(CameraMode { is_pov: false })
        .insert_resource(QuitReceiver(quit_receiver))
        .insert_resource(OrientationRes(CameraOrientation::default()))
        .insert_resource(PovInfo::default())
        .run();
}

fn quit_system(quit_rx: Res<QuitReceiver>, mut exit_ev: ResMut<Events<AppExit>>) {
    if quit_rx.0.try_recv().is_ok() {
        exit_ev.send(AppExit::Success);
    }
}

#[derive(Debug, Resource, Default)]
pub struct OrientationRes(CameraOrientation);

fn centre_camera_system(
    agent_query: Query<(&Transform, &AgentComponent), Without<Camera3d>>,
    mut agent_vis_query: Query<&mut Visibility, With<AgentComponent>>,
    mut pan_orbit_camera_query: Query<
        (&mut PanOrbitCamera, &mut Transform, &mut Camera),
        (With<Camera3d>, Without<PovCamera>, Without<AgentComponent>),
    >,
    mut pov_camera_query: Query<
        (&mut Transform, &mut Camera),
        (With<Camera3d>, With<PovCamera>, Without<AgentComponent>),
    >,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut focused: ResMut<FocusedAgent>,
    mut camera_mode: ResMut<CameraMode>,
    camera_orientation: Res<OrientationRes>,
    pov_info: Res<PovInfo>,
) {
    if keyboard_input.just_pressed(KeyCode::Tab) {
        camera_mode.is_pov = !camera_mode.is_pov;
        for (_, _, mut camera) in pan_orbit_camera_query.iter_mut() {
            camera.is_active = !camera_mode.is_pov;
        }
        for (_, mut camera) in pov_camera_query.iter_mut() {
            camera.is_active = camera_mode.is_pov;
        }

        if camera_mode.is_pov {
            for mut vis in agent_vis_query.iter_mut() {
                *vis = Visibility::Hidden;
            }
        } else {
            for mut vis in agent_vis_query.iter_mut() {
                *vis = Visibility::Visible;
            }
        }
    }

    let mut pov_cam = None;
    for (t, a) in agent_query.iter() {
        if a.agent.id == pov_info.agent_id {
            pov_cam = Some((t.translation, &a.agent));
        }
    }
    if let Some((pos, agent)) = pov_cam {
        if camera_mode.is_pov {
            if let Ok((mut pov_transform, _)) = pov_camera_query.single_mut() {
                update_pov_camera_transform(&mut pov_transform, &pos, agent, &camera_orientation.0);
            }
        } else if keyboard_input.just_pressed(KeyCode::KeyZ) {
            for (mut pan_orbit, _, _) in pan_orbit_camera_query.iter_mut() {
                pan_orbit.target_focus = pos;
            }
        }
    }

    // continuous POV update omitted for brevity
}

fn update_pov_camera_transform(
    camera_transform: &mut Transform,
    agent_pos: &Vec3,
    agent: &Agent,
    orientation: &CameraOrientation,
) {
    camera_transform.translation = *agent_pos;
    let camera_view = agent.camera_view(orientation);
    let forward_client = Vector3::new(
        camera_view.camera_forward.x,
        camera_view.camera_forward.y,
        camera_view.camera_forward.z,
    );
    let up_client = Vector3::new(
        camera_view.camera_up.x,
        camera_view.camera_up.y,
        camera_view.camera_up.z,
    );
    let right_client = Vector3::new(
        camera_view.camera_right.x,
        camera_view.camera_right.y,
        camera_view.camera_right.z,
    );
    let forward = client_to_bevy_f32(forward_client.cast::<f32>());
    let up = client_to_bevy_f32(up_client.cast::<f32>());
    let right = client_to_bevy_f32(right_client.cast::<f32>());
    camera_transform.rotation = Quat::from_mat3(&Mat3::from_cols(right, up, -forward));
}

#[allow(clippy::too_many_arguments)]
fn synchronise_world(
    mut commands: Commands,
    pov: Res<PovReceiver>,
    agents: Res<AgentReceiver>,
    assets: Res<CellAssets>,
    mut cell_sets: ParamSet<(
        Query<(
            Entity,
            &mut VirtualCellComponent,
            &mut MeshMaterial3d<StandardMaterial>,
        )>,
        Query<(
            Entity,
            &mut DynamicCellComponent,
            &mut MeshMaterial3d<StandardMaterial>,
        )>,
    )>,
    action_cell_query: Query<(Entity, &ActionCell)>,
    action_origin_cell_query: Query<(Entity, &OriginCell)>,
    mut agent_query: Query<(Entity, &mut AgentComponent, &mut Transform)>,
    mut camera_projection_query: Query<
        (&mut Projection),
        (With<Camera3d>, With<PovCamera>, Without<AgentComponent>),
    >,
    mut camera_orientation: ResMut<OrientationRes>,
    mut gizmos: Gizmos,
    mut pov_info: ResMut<PovInfo>,
) {
    if let Some(mut pov) = pov.0.try_iter().last() {
        // Snapshot voxel count before we mutate the grid while syncing
        pov_info.voxel_count = pov.virtual_world.cells().len();
        // Snapshot dynamic (phase world) block count as well
        pov_info.dynamic_count = pov.dynamic_world.cells.len();
        pov_info.agent_id = pov.agent_id;
        camera_orientation.0 = pov.orientation;

        // --- Dynamic world (PhaseGrid) ---
        {
            let mut dyn_cell_query = cell_sets.p1();
            for (entity, mut cell, mut material) in dyn_cell_query.iter_mut() {
                if let Some(w) = pov
                    .dynamic_world
                    .cells
                    .get(&cell.coord)
                    .map(|x| *x.deref() as f64)
                {
                    cell.weight = w as f32;
                    **material = assets.material_for_weight(w);
                    let _ = pov.dynamic_world.cells.remove(&cell.coord);
                } else {
                    commands.entity(entity).despawn();
                }
            }
        }
        for rm in pov.dynamic_world.cells.iter() {
            let (coord, w) = rm.pair();
            let client = Vector3::new(coord[0], coord[1], coord[2]);
            commands.spawn((
                DynamicCellComponent {
                    coord: *coord,
                    weight: *w as f32,
                },
                Mesh3d(assets.cube_mesh.clone()),
                MeshMaterial3d(assets.material_for_weight(*w)),
                Transform::from_translation(client_to_bevy_i32(client)),
            ));
        }

        {
            let mut cell_query = cell_sets.p0();
            for (entity, mut cell, mut material) in cell_query.iter_mut() {
                if let Some(v) = pov
                    .virtual_world
                    .cells()
                    .get(&cell.coord)
                    .map(|x| *x.deref())
                {
                    cell.value = v;
                    **material = assets.material_for_cell(&v);
                    pov.virtual_world.remove(&cell.coord);
                } else {
                    commands.entity(entity).despawn();
                }
            }
        }
        for rm in pov.virtual_world.cells().iter() {
            let (coord, cell) = rm.pair();
            let client = Vector3::new(coord[0], coord[1], coord[2]);
            commands.spawn((
                VirtualCellComponent {
                    coord: *coord,
                    value: *cell,
                },
                Mesh3d(assets.cube_mesh.clone()),
                MeshMaterial3d(assets.material_for_cell(&cell)),
                Transform::from_translation(client_to_bevy_i32(client)),
            ));
        }
        let proj = pov.proj;
        for mut camera_proj in camera_projection_query.iter_mut() {
            *camera_proj = Projection::Perspective(PerspectiveProjection {
                fov: proj.fov_vertical as f32,
                aspect_ratio: proj.aspect as f32,
                near: proj.near_distance as f32,
                far: proj.max_distance as f32,
                ..default()
            });
        }
    }

    if let Some(mut agents_map) = agents.0.try_iter().last() {
        let mut action_cells = Vec::new();
        let mut origin_cells = Vec::new();
        for (_id, agent) in agents_map.iter() {
            if let Some(action) = agent.get_action() {
                origin_cells.push(action.origin);
                let p = action.origin;
                if let Ok(centroids) = Action::chained_centroids(action.intent_queue.iter(), p) {
                    for centroid in centroids {
                        action_cells.push(centroid.0);
                    }
                }
            }
        }

        for (entity, cell) in action_cell_query.iter() {
            let mut contains = false;
            action_cells.retain(|x| {
                if x == &Vector3::from(cell.coord) {
                    contains = true;
                    false
                } else {
                    true
                }
            });
            if !contains {
                commands.entity(entity).despawn();
            }
        }

        for a in action_cells {
            commands.spawn((
                ActionCell { coord: a.into() },
                Mesh3d(assets.action_mesh.clone()),
                MeshMaterial3d(assets.action_mat.clone()),
                Transform::from_translation(client_to_bevy_i32(a)),
            ));
        }

        for (entity, cell) in action_origin_cell_query.iter() {
            let mut contains = false;
            origin_cells.retain(|x| {
                if x == &Vector3::from(cell.coord) {
                    contains = true;
                    false
                } else {
                    true
                }
            });
            if !contains {
                commands.entity(entity).despawn();
            }
        }

        for a in origin_cells {
            commands.spawn((
                OriginCell { coord: a.into() },
                Mesh3d(assets.action_mesh.clone()),
                MeshMaterial3d(assets.action_origin_mat.clone()),
                Transform::from_translation(client_to_bevy_i32(a)),
            ));
        }

        // update existing agents
        for (entity, mut agent_comp, mut transform) in agent_query.iter_mut() {
            let mut found = false;
            agents_map.retain(|_, net_a| {
                if agent_comp.agent.id == net_a.id {
                    let pos_f = net_a.pos.cast::<f32>();
                    let bevy_pos = client_to_bevy_f32(pos_f);
                    transform.translation = bevy_pos;
                    transform.rotation = client_to_bevy_quat(net_a.attitude);
                    agent_comp.agent = net_a.clone();

                    // forward‐vector gizmo
                    let fwd_client =
                        agent_comp.agent.attitude * MoveDir::Forward.dir_vector().unwrap().cast();
                    let fwd_bevy = client_to_bevy_f32(fwd_client.cast::<f32>());
                    gizmos.line(
                        bevy_pos,
                        bevy_pos + fwd_bevy * 5.0,
                        Color::Srgba(Srgba::RED),
                    );

                    // spline
                    if let Some(action) = &agent_comp.agent.get_action() {
                        draw_spline(&mut gizmos, &action.trajectory);
                    }

                    found = true;
                    false
                } else {
                    true
                }
            });
            if !found {
                commands.entity(entity).despawn();
            }
        }
        // spawn new agents
        for (_, net_a) in agents_map.iter() {
            let start_f = net_a.pos.cast::<f32>();
            let bevy_start = client_to_bevy_f32(start_f);
            let transform = Transform::from_translation(bevy_start);

            if let Some(ref drone_scene) = assets.drone_scene {
                // Spawn GLTF scene
                commands.spawn((
                    AgentComponent {
                        agent: net_a.clone(),
                    },
                    SceneRoot(drone_scene.clone()),
                    transform,
                    Visibility::Visible,
                ));
            } else {
                // Fallback to torus mesh
                commands.spawn((
                    AgentComponent {
                        agent: net_a.clone(),
                    },
                    Mesh3d(assets.drone_mesh.clone()),
                    MeshMaterial3d(assets.drone_body_mat.clone()),
                    transform,
                    Visibility::Visible,
                ));
            }

            // forward‐vector gizmo for new agents
            let fwd_client = net_a.attitude * Vector3::y_axis();
            let fwd_bevy = client_to_bevy_f32(fwd_client.cast::<f32>().into_inner());
            gizmos.line(
                bevy_start,
                bevy_start + fwd_bevy * 5.0,
                Color::Srgba(Srgba::RED),
            );
        }
    }
}

fn draw_spline(gizmos: &mut Gizmos, spline: &Trajectory) {
    let segments = 200;
    let pts: Vec<Vec3> = spline
        .waypoints(segments)
        .into_iter()
        .map(|(_t, p)| client_to_bevy_f32(p.cast::<f32>()))
        .collect();
    gizmos.linestrip(pts.into_iter(), Color::Srgba(Srgba::BLUE));
}

fn setup_pov_ui(mut commands: Commands) {
    // Root container (top-left overlay)
    commands
        .spawn((
            Node {
                position_type: bevy::ui::PositionType::Absolute,
                left: bevy::ui::Val::Px(10.0),
                top: bevy::ui::Val::Px(10.0),
                ..default()
            },
            BackgroundColor(Color::NONE),
        ))
        .with_children(|parent| {
            parent.spawn((Text::new(""), TextColor(Color::WHITE), TelemetryText));
        });
}

fn update_pov_ui_text(
    mut text_q: Query<&mut Text, With<TelemetryText>>,
    agents_q: Query<&AgentComponent>,
    pov_info: Res<PovInfo>,
) {
    let mut text = match text_q.single_mut() {
        Ok(t) => t,
        Err(_) => return,
    };

    // Find the focused agent
    let mut display = String::from("No agent");
    let target_id = pov_info.agent_id;
    for a in agents_q.iter() {
        // Prefer the POV stream's agent id; fallback to FocusedAgent for safety
        if a.agent.id == target_id {
            let p = a.agent.pos;
            let v = a.agent.vel;
            let (roll, pitch, yaw) = a.agent.attitude.euler_angles();
            let intents = a
                .agent
                .get_action()
                .map(|act| act.intent_queue.len())
                .unwrap_or(0);

            display = format!(
                "Agent {}\nPos:  ({:.2}, {:.2}, {:.2})\nVel:  ({:.2}, {:.2}, {:.2}) [{:.2}]\nRot:  R:{:.1}° P:{:.1}° Y:{:.1}°\nIntents: {}\nBlocks: {}\nPhase blocks: {}",
                a.agent.id,
                p.x,
                p.y,
                p.z,
                v.x,
                v.y,
                v.z,
                v.norm(),
                roll.to_degrees(),
                pitch.to_degrees(),
                yaw.to_degrees(),
                intents,
                pov_info.voxel_count,
                pov_info.dynamic_count,
            );
            break;
        }
    }

    *text = Text::new(display);
}
