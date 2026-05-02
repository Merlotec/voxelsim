// use std::ops::Deref;

use std::f64::consts::FRAC_PI_2;

use bevy::platform::collections::HashMap;
use bevy::render::RenderPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use crossbeam_channel::{Receiver, Sender};
use nalgebra::{Normed, UnitQuaternion, Vector3};
use voxelsim::trajectory::Trajectory;
use voxelsim::viewport::CameraOrientation;

#[cfg(not(target_arch = "wasm32"))]
use crate::network::NetworkSubscriber;

#[cfg(target_arch = "wasm32")]
use crate::network_wasm::NetworkSubscriber;
use crate::render::{
    self, ActionCell, AgentComponent, AgentReceiver, CellAssets, CellComponent, FocusedAgent,
    OriginCell, QuitReceiver, WorldReceiver,
};
use voxelsim::{Action, Agent, Cell, MoveDir, VoxelGrid};

use crate::convert::*;

use bevy::app::AppExit;
use bevy::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
pub fn run_world_server() {
    let (mut world_sub, world_receiver) = NetworkSubscriber::<VoxelGrid>::new(
        std::env::var("VXS_WORLD_PORT").unwrap_or("0.0.0.0".to_string()),
        std::env::var("VXS_WORLD_PORT")
            .ok()
            .and_then(|x| x.parse::<u16>().ok())
            .unwrap_or(8080),
    );

    let (mut agent_sub, agent_receiver) = NetworkSubscriber::<HashMap<usize, Agent>>::new(
        std::env::var("VXS_AGENT_PORT").unwrap_or("0.0.0.0".to_string()),
        std::env::var("VXS_AGENT_PORT")
            .ok()
            .and_then(|x| x.parse::<u16>().ok())
            .unwrap_or(8081),
    );

    // Start network listener in background thread with its own tokio runtime
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            world_sub.start().await;
            agent_sub.start().await;

            // Keep the async context alive
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        });
    });

    let (gui_sender, gui_receiver) = crossbeam_channel::unbounded::<GuiCommand>();
    let (quit_sender, quit_receiver) = crossbeam_channel::unbounded::<()>();

    begin_render(world_receiver, agent_receiver, gui_sender, quit_receiver);
}

pub enum GuiCommand {
    RegenerateWorld,
    MoveAgentA,
    MoveAgentB,
    MoveAgentC,
    MoveAgentD,
}

pub fn begin_render(
    world_receiver: Receiver<VoxelGrid>,
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
                        title: "World View".into(),
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
        .add_systems(Startup, render::setup)
        .add_systems(
            Update,
            (
                synchronise_world,
                input_system,
                centre_camera_system,
                quit_system,
            ),
        )
        .insert_resource(WorldReceiver(world_receiver))
        .insert_resource(AgentReceiver(agent_receiver))
        .insert_resource(GuiChannel(gui_sender))
        .insert_resource(FocusedAgent(0))
        .insert_resource(QuitReceiver(quit_receiver))
        .run();
}

#[derive(Resource)]
struct GuiChannel(Sender<GuiCommand>);

fn input_system(gui_channel: Res<GuiChannel>, keyboard: Res<ButtonInput<KeyCode>>) {
    for k in keyboard.get_just_pressed() {
        // we assign to `_` to drop the Result<(), SendError> so it's not unused
        let _ = match k {
            KeyCode::KeyA => gui_channel.0.send(GuiCommand::MoveAgentA),
            KeyCode::KeyB => gui_channel.0.send(GuiCommand::MoveAgentB),
            KeyCode::KeyC => gui_channel.0.send(GuiCommand::MoveAgentC),
            KeyCode::KeyD => gui_channel.0.send(GuiCommand::MoveAgentD),
            KeyCode::KeyR => gui_channel.0.send(GuiCommand::RegenerateWorld),

            _ => Ok(()),
        };
    }
}

fn quit_system(quit_rx: Res<QuitReceiver>, mut exit_ev: ResMut<Events<AppExit>>) {
    // try_recv() returns Ok(()) if someone sent a quit message
    if quit_rx.0.try_recv().is_ok() {
        exit_ev.send(AppExit::Success);
    }
}

fn centre_camera_system(
    agent_query: Query<(&Transform, &AgentComponent), Without<Camera3d>>,
    mut camera_query: Query<&mut PanOrbitCamera, Without<AgentComponent>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut focused: ResMut<FocusedAgent>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyZ) {
        let mut positions = Vec::new();
        for (t, a) in agent_query.iter() {
            positions.push((a.agent.id, t.translation));
        }
        positions.sort_by_key(|(k, _)| *k);
        let next_i = if let Some(i) = positions.iter().position(|(k, _)| *k == focused.0) {
            if (i + 1) < positions.len() { i + 1 } else { 0 }
        } else if !positions.is_empty() {
            0
        } else {
            return;
        };

        let pos = positions[next_i].1;
        for mut t in camera_query.iter_mut() {
            t.target_focus = pos;
        }
        focused.0 = positions[next_i].0;
    }
}

#[allow(clippy::too_many_arguments)]
fn synchronise_world(
    mut commands: Commands,
    world: Res<WorldReceiver>,
    agents: Res<AgentReceiver>,
    assets: Res<CellAssets>,
    mut cell_query: Query<(
        Entity,
        &mut CellComponent,
        &mut MeshMaterial3d<StandardMaterial>,
    )>,
    action_cell_query: Query<(Entity, &ActionCell)>,
    action_origin_cell_query: Query<(Entity, &OriginCell)>,
    mut agent_query: Query<(Entity, &mut AgentComponent, &mut Transform)>,
    mut gizmos: Gizmos,
) {
    // ── CELLS ───────────────────────────────────────────────────────────
    if let Some(mut world) = world.0.try_iter().last() {
        for (entity, mut cell, mut material) in cell_query.iter_mut() {
            if let Some(v) = world.cells().get(&cell.coord).map(|x| *x) {
                cell.value = v;
                **material = assets.material_for_cell(&v);
                world.remove(&cell.coord);
            } else {
                commands.entity(entity).despawn();
            }
        }

        // add new cells with Y-up swap
        for rm in world.cells().iter() {
            let (coord, cell_val) = rm.pair();
            commands.spawn((
                CellComponent {
                    coord: *coord,
                    value: *cell_val,
                },
                Mesh3d(assets.cube_mesh.clone()),
                MeshMaterial3d(assets.material_for_cell(cell_val)),
                Transform::from_translation(client_to_bevy_i32(*coord)),
            ));
        }
    }

    // ── ACTION CELLS & ORIGINS ───────────────────────────────────────────
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

        // ── AGENTS ────────────────────────────────────────────────────────
        for (entity, mut agent_comp, mut transform) in agent_query.iter_mut() {
            let mut found = false;
            agents_map.retain(|_, net_agent| {
                if agent_comp.agent.id == net_agent.id {
                    // swap incoming pos
                    let client_pos = net_agent.pos.cast::<f32>();
                    let bevy_pos = client_to_bevy_f32(client_pos);
                    transform.translation = bevy_pos;
                    transform.rotation = client_to_bevy_quat(net_agent.attitude);
                    agent_comp.agent = net_agent.clone();

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

        // spawn any new agents
        for (_, net_agent) in agents_map.iter() {
            let start = net_agent.pos.cast::<f32>();
            let bevy_start = client_to_bevy_f32(start);
            assets.spawn_agent_with_model(
                &mut commands,
                net_agent.clone(),
                Transform::from_translation(bevy_start),
            );
            let fwd_client = net_agent.attitude * Vector3::y_axis();
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

/// WASM mode: Connect to WebSocket proxy for live simulation data
///
/// This replaces the old static demo with real network connectivity.
/// The proxy server bridges TCP (Python) to WebSocket (browser).
#[cfg(target_arch = "wasm32")]
pub fn run_world_demo() {
    // Use default URLs for demo mode (external node proxy)
    run_world_with_urls("ws://localhost:9080".to_string(), "ws://localhost:9081".to_string());
}

/// Run the renderer with custom WebSocket URLs (for backend proxy mode)
#[cfg(target_arch = "wasm32")]
pub fn run_world_with_urls(world_url: String, agent_url: String) {
    web_sys::console::log_1(&"Starting VoxelSim Renderer (WASM WebSocket Mode)...".into());
    web_sys::console::log_1(&format!("World URL: {}", world_url).into());
    web_sys::console::log_1(&format!("Agent URL: {}", agent_url).into());

    // Create WebSocket subscribers with provided URLs
    let (world_sub, world_receiver) = NetworkSubscriber::<VoxelGrid>::new(world_url);
    let (agent_sub, agent_receiver) = NetworkSubscriber::<HashMap<usize, Agent>>::new(agent_url);

    // Start WebSocket connections
    world_sub.start();
    agent_sub.start();

    let (gui_sender, _gui_receiver) = crossbeam_channel::unbounded::<GuiCommand>();
    let (_quit_sender, quit_receiver) = crossbeam_channel::unbounded::<()>();

    web_sys::console::log_1(
        &"WebSocket connections initiated. Waiting for simulation data...".into(),
    );

    // Start Bevy with the same function as native version!
    begin_render(world_receiver, agent_receiver, gui_sender, quit_receiver);
}
