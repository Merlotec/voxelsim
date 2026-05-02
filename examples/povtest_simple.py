from pynput import keyboard, mouse
import voxelsim as vxs, time
import math, json, sys
from pathlib import Path

# ============ WORLD SIZE CONFIGURATION ============
# Set this to control the world size (e.g., 10, 30, 100)
# Smaller values = faster performance for testing
# NOTE: Very small worlds (< 20) may cause GPU buffer errors
WORLD_SIZE = 120  # Change this value (20 = minimum, 30 = small, 100 = default)

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
nx = ny = nz = WORLD_SIZE
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
    config = vxs.TerrainConfig.default_py()
    config.set_world_size_py(WORLD_SIZE)
    generator.generate_terrain_py(config)
    world = generator.generate_world_py()
    print(f"Created world: {WORLD_SIZE}x{WORLD_SIZE}x{WORLD_SIZE}")

# Simple agent without dynamics
agent = vxs.Agent(0)
# Place agent near center of footprint and above ground
cx, cy = nx // 2, ny // 2
cz = max(5, nz // 3)
agent.set_hold_py([cx, cy, -cz], 0.0)

fw = vxs.FilterWorld()

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

YAW_STEP = math.pi * 0.125  # radians per key press (about 17°)

yaw_delta = 0.0

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

    action = agent.get_action_py()
    commands = []
    if action:
        commands = action.get_intent_queue()[0].get_move_sequence()
    commands_cl = list(commands)

    # Compute yaw delta from input this frame; Q = left (negative), E = right (positive)
    if 'q' in just_pressed:
        yaw_delta -= YAW_STEP
    if 'e' in just_pressed:
        yaw_delta += YAW_STEP

    # Apply yaw_delta to any move commands created this frame
    if 'w' in just_pressed: commands.append(vxs.MoveDir.Forward)
    if 's' in just_pressed: commands.append(vxs.MoveDir.Back)
    if 'a' in just_pressed: commands.append(vxs.MoveDir.Left)
    if 'd' in just_pressed: commands.append(vxs.MoveDir.Right)
    if 'space' in just_pressed: commands.append(vxs.MoveDir.Up)
    if 'shift' in just_pressed: commands.append(vxs.MoveDir.Down)
    if not action or commands != commands_cl:
        if len(commands) > 0:
            intent = vxs.ActionIntent(0.8, yaw_delta, commands, None)
            agent.perform_oneshot_py(intent)

    # The point in space that the drone should be chasing.
    chase_target = chaser.step_chase_py(agent, delta)
    # No dynamics update - agent position is controlled directly by actions
    just_pressed.clear()
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
