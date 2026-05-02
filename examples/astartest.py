
from pynput import keyboard, mouse
import voxelsim as vxs, time
import math, json, sys
from pathlib import Path



def _load_world_from_json(path: Path):
    data = json.loads(path.read_text())
    cells = {}
    filled = vxs.Cell.filled()
    for item in data.get("cells", []):
        x, y, z = int(item[0]), int(item[1]), int(item[2])
        cells[(x, y, z)] = filled
    if hasattr(vxs.VoxelGrid, "from_dict_py"):
        vg = vxs.VoxelGrid.from_dict_py(cells)
    else:
        vg = vxs.VoxelGrid()
        for k, cell in cells.items():
            vg.set_cell_py(k, cell)
    dims = data.get("dims", {})
    nx = int(dims.get("x", 100))
    ny = int(dims.get("y", 100))
    nz = int(dims.get("z", 60))
    return vg, (nx, ny, nz)

# If a JSON world is provided as argv[1], load it; otherwise use generated terrain
world = None
nx = ny = 100
nz = 60
if len(sys.argv) > 1:
    in_path = Path(sys.argv[1])
    if in_path.exists():
        try:
            world, (nx, ny, nz) = _load_world_from_json(in_path)
            print(f"Loaded world from {in_path} with dims=({nx},{ny},{nz})")
        except Exception as e:
            print(f"Failed to load world from {in_path}: {e}. Falling back to generated terrain.")
if world is None:
    generator = vxs.TerrainGenerator()
    generator.generate_terrain_py(vxs.TerrainConfig.default_py())
    world = generator.generate_world_py()
    nx, ny, nz = 100, 100, 60

# dynamics = vxs.AgentDynamics.default_drone()
agent = vxs.Agent(0)
# Place agent near center of footprint and above ground
cx, cy = nx // 2, ny // 2
cz = max(5, nz // 3)
agent.set_hold_py([cx, cy, -cz], 0.0)


POS = [cx + 5, cy + 5, -cz]

fw = vxs.FilterWorld()
dynamics = vxs.px4.Px4Dynamics.default_py()

chaser = vxs.FixedLookaheadChaser.default_py()
planner = vxs.AStarActionPlanner(1)
proj = vxs.CameraProjection.default_py()
env = vxs.EnvState.default_py()

AGENT_CAMERA_TILT = -0.5
camera_orientation = vxs.CameraOrientation.vertical_tilt_py(AGENT_CAMERA_TILT)
# Renderer
noise = vxs.NoiseParams.default_with_seed_py([0.0, 0.0, 0.0])
renderer = vxs.AgentVisionRenderer(world, [200, 150], noise)

# Client

client = vxs.RendererClient.default_localhost_py(1)
# Specify the number of agent renderers we want to connect to.
print("Controls: WASD=move, Space=up, Shift=down, ESC=quit")

client.send_world_py(world)
client.send_agents_py({0: agent})

pressed = set()
just_pressed = set()

def on_press(key):
    # normalize all keys to a string
    try:
        key_str = key.char.lower()
    except AttributeError:
        if key == keyboard.Key.space:
            key_str = 'space'
        elif key == keyboard.Key.shift:
            key_str = 'shift'
        elif key == keyboard.Key.esc:
            # stop listener
            return False
        else:
            # ignore other non-char keys
            return

    # if this is the first time we've seen it down, mark as just_pressed
    if key_str not in pressed:
        just_pressed.add(key_str)
    # always record that it's down
    pressed.add(key_str)

def on_release(key):
    try:
        key_str = key.char.lower()
    except AttributeError:
        if key == keyboard.Key.space:
            key_str = 'space'
        elif key == keyboard.Key.shift:
            key_str = 'shift'
        else:
            return
    pressed.discard(key_str)

from threading import Thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

delta = 0.01

last_view_time = time.time()

FRAME_DELTA_MAX = 0.13

upd_start = 0.0
def world_update(world, timestamp):
    dtime = time.time() - upd_start
    # print(f"upd_time: {dtime}")

astar_intent = planner.plan_action_py(world, agent.get_coord_py(), POS, 0.9, 0.1)

agent.perform_oneshot_py(astar_intent)



while listener.running:
    t0 = time.time()
    view_delta = t0 - last_view_time
    # Rendering
    if fw.is_updating_py(last_view_time):
        if view_delta >= FRAME_DELTA_MAX:
            # Do not update the simulation because we want to await the view.
            continue;
    else:
        # Our next frame is ready, so wait for the frame delta time to hit then update the world
        # used for inference.
        # Here we just send the new world over to the renderer.
        if view_delta >= FRAME_DELTA_MAX:
            fw.send_pov_py(client, 0, 0, proj, camera_orientation)
            upd_start = time.time()
            renderer.update_filter_world_py(agent.camera_view_py(camera_orientation), proj, fw, t0, world_update)
            last_view_time = t0

    # The point in space that the drone should be chasing.
    chase_target = chaser.step_chase_py(agent, delta)
    dynamics.update_agent_dynamics_py(agent, env, chase_target, delta)
    collisions = world.collisions_py(agent.get_pos(), [0.5, 0.5, 0.3])
    if len(collisions) > 0:
        print (f"{len(collisions)}, collisions")
    # collisions = env.update_py(dynamics, delta)
    # if len(collisions) > 0:
    #     print("Collision!")
    # im = env.update_pov_py()
    
    client.send_agents_py({0: agent})
    # env.send_pov(client, 0, 0)
    d = time.time() - t0
    if d < delta:
        time.sleep(delta - d)

listener.join()
