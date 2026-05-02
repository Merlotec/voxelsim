import voxelsim as vxs, time

WORLD_SIZE = 30

generator = vxs.TerrainGenerator()
config = vxs.TerrainConfig.default_py()
config.set_world_dimensions_py(WORLD_SIZE, 30, WORLD_SIZE)
generator.generate_terrain_py(config)
world = generator.generate_world_py()

agent = vxs.Agent(0)
cx, cy, cz = WORLD_SIZE // 2, WORLD_SIZE // 2, 30
agent.set_hold_py([cx, cy, -cz], 0.0)

fw = vxs.FilterWorld()
dw = vxs.VoxelGrid()
dynamics = vxs.px4.Px4Dynamics.default_py()
chaser = vxs.FixedLookaheadChaser.default_py()
planner = vxs.AStarActionPlanner(1)
proj = vxs.CameraProjection.default_py()
env = vxs.EnvState.default_py()
camera_orientation = vxs.CameraOrientation.vertical_tilt_py(-0.5)
noise = vxs.NoiseParams.default_with_seed_py([0.0, 0.0, 0.0])
renderer = vxs.AgentVisionRenderer(world, [200, 150], noise)

client = vxs.AsyncRendererClient.default_localhost_py(1)
client.send_world_py(world)
client.send_agents_py({0: agent})

target = [cx + 5, cy + 5, -cz]
astar_intent = planner.plan_action_py(world, agent.get_coord_py(), target, 0.9, 0.1)
agent.perform_oneshot_py(astar_intent)
print(f"A* action planned: {agent.get_coord_py()} -> {target}")

delta = 0.01
last_view_time = time.time()
FRAME_DELTA_MAX = 0.13
upd_start = 0.0

def world_update(world, timestamp):
    pass

while True:
    t0 = time.time()
    view_delta = t0 - last_view_time
    if fw.is_updating_py(last_view_time):
        if view_delta >= FRAME_DELTA_MAX:
            continue
    else:
        if view_delta >= FRAME_DELTA_MAX:
            fdw = vxs.PhaseGrid()
            fw.send_pov_async_py(client, 0, 0, proj, camera_orientation, fdw)
            upd_start = time.time()
            renderer.update_filter_world_py(agent.camera_view_py(camera_orientation), proj, fw, dw, t0, world_update)
            last_view_time = t0

    chase_target = chaser.step_chase_py(agent, delta)
    dynamics.update_agent_dynamics_py(agent, env, chase_target, delta)
    collisions = world.collisions_py(agent.get_pos(), [0.5, 0.5, 0.3])
    if len(collisions) > 0:
        print(f"{len(collisions)} collisions")
    client.send_agents_py({0: agent})
    d = time.time() - t0
    if d < delta:
        time.sleep(delta - d)
