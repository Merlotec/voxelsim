from pynput import keyboard, mouse
import voxelsim as vxs, time
import math, json, sys
from pathlib import Path

# ============ WORLD SIZE CONFIGURATION ============
# Set this to control the world size (e.g., 10, 30, 100)
# Smaller values = faster performance for testing
# NOTE: Very small worlds (< 20) may cause GPU buffer errors
WORLD_SIZE = 100  # Change this value (20 = minimum, 30 = small, 100 = default)

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
nx = WORLD_SIZE
ny=WORLD_SIZE
nz = 30
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
    config.set_world_dimensions_py(WORLD_SIZE, 30 , WORLD_SIZE)  # (x, y/height, z)
    generator.generate_terrain_py(config)
    world = generator.generate_world_py()
    print(f"Created world: {WORLD_SIZE}x30x{WORLD_SIZE}")

class AgentInstance:
    def __init__(self, agent_id: int, start_pos):
        self.id = agent_id
        self.agent = vxs.Agent(agent_id)
        self.agent.set_hold_py(start_pos, 0.0)

        # Per-agent state
        self.fw = vxs.FilterWorld()
        self.phase_solver = vxs.PtgSolver.default_py()
        self.phase_world = vxs.PhaseGrid()

        # Motion control per agent
        self.chaser = vxs.FixedLookaheadChaser.default_py()
        self.dynamics = vxs.px4.Px4Dynamics.default_py()

        # View pacing per agent
        self.last_view_time = time.time()

    def render_changeset(self, changeset: vxs.WorldChangeset):
        # Update this agent's filter world and phase solver
        diff = changeset.update_filter_world_with_difference_py(self.fw)
        vw = self.fw.clone_world_py()
        self.phase_solver.add_phase_frame_py(self.agent, vw, changeset.timestamp_py())


# The dynamic world (with moving components in it) — shared across agents
dw = vxs.VoxelGrid()

# Shared planning/rendering helpers
planner = vxs.AStarActionPlanner(1)
proj = vxs.CameraProjection.default_py()
env = vxs.EnvState.default_py()

AGENT_CAMERA_TILT = -0.5
camera_orientation = vxs.CameraOrientation.vertical_tilt_py(AGENT_CAMERA_TILT)
# Renderer
noise = vxs.NoiseParams.default_with_seed_py([0.0, 0.0, 0.0])
renderer = vxs.AgentVisionRenderer(world, [200, 150], noise)

# Client
client = vxs.AsyncRendererClient.default_localhost_py(2) # We have 2 agents.
print("Controls: Agent0 WASD/Space/Shift + Q/E yaw | Agent1 Arrows and '/ for up/down. ESC quits.")

# Create two agents with their own worlds/solvers
cx, cy = nx // 2, ny // 2
cz = max(5, nz // 3)
agent0 = AgentInstance(0, [cx, cy, -cz])
# Slight offset for the second agent to avoid immediate collisions
agent1 = AgentInstance(1, [cx + 3, cy + 3, -cz])

client.send_world_py(world)
client.send_agents_py({0: agent0.agent, 1: agent1.agent})

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
        elif key == keyboard.Key.up:
            key_str = 'up'
        elif key == keyboard.Key.down:
            key_str = 'down'
        elif key == keyboard.Key.left:
            key_str = 'left'
        elif key == keyboard.Key.right:
            key_str = 'right'
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
        elif key == keyboard.Key.up:
            key_str = 'up'
        elif key == keyboard.Key.down:
            key_str = 'down'
        elif key == keyboard.Key.left:
            key_str = 'left'
        elif key == keyboard.Key.right:
            key_str = 'right'
        else:
            return
    pressed.discard(key_str)

from threading import Thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

delta = 0.01

FRAME_DELTA_MAX = 0.13

YAW_STEP = math.pi * 0.125  # radians per key press (about 17°)

while listener.running:
    t_loop = time.time()

    # Update viewing/rendering per agent
    dw = vxs.VoxelGrid()
    agents = [agent0.agent, agent1.agent]
    for ai in (agent0, agent1):
        t0 = time.time()
        view_delta = t0 - ai.last_view_time
        if ai.fw.is_updating_py(ai.last_view_time):
            # Skip sending a new POV until the previous one has updated or enough time has passed
            pass
        else:
            if view_delta >= FRAME_DELTA_MAX:
                lts = ai.phase_solver.latest_timestamp_py()
                if lts != None:
                    ai.phase_world = ai.phase_solver.gen_phase_grid_py(ai.agent, lts, lts + 1.0)
                ai.fw.send_pov_async_py(client, ai.id, ai.id, proj, camera_orientation, ai.phase_world)
                dw = vxs.VoxelGrid()
                to_populate = agents.copy()
                to_populate.remove(ai.agent)
                dw.populate_with_agents_py(to_populate)
                renderer.render_changeset_py(ai.agent.camera_view_py(camera_orientation), proj, ai.fw, dw, t0, ai.render_changeset)
                ai.last_view_time = t0

    # Controls and dynamics for Agent 0 (WASD, space/shift, yaw Q/E)
    action0 = agent0.agent.get_action_py()
    commands0 = []
    if action0:
        commands0 = action0.get_intent_queue()[0].get_move_sequence()
    commands0_cl = list(commands0)
    yaw_delta0 = 0.0
    if 'q' in just_pressed:
        yaw_delta0 -= YAW_STEP
    if 'e' in just_pressed:
        yaw_delta0 += YAW_STEP
    if 'w' in just_pressed: commands0.append(vxs.MoveDir.Forward)
    if 's' in just_pressed: commands0.append(vxs.MoveDir.Back)
    if 'a' in just_pressed: commands0.append(vxs.MoveDir.Left)
    if 'd' in just_pressed: commands0.append(vxs.MoveDir.Right)
    if 'space' in just_pressed: commands0.append(vxs.MoveDir.Up)
    if 'shift' in just_pressed: commands0.append(vxs.MoveDir.Down)
    if not action0 or commands0 != commands0_cl:
        if len(commands0) > 0:
            intent0 = vxs.ActionIntent(0.8, yaw_delta0, commands0)
            agent0.agent.perform_oneshot_py(intent0)

    # Controls and dynamics for Agent 1 (arrow keys, apostrophe up, slash down)
    action1 = agent1.agent.get_action_py()
    commands1 = []
    if action1:
        commands1 = action1.get_intent_queue()[0].get_move_sequence()
    commands1_cl = list(commands1)
    # No yaw for agent 1 unless specified; set 0.0
    yaw_delta1 = 0.0
    if 'up' in just_pressed: commands1.append(vxs.MoveDir.Forward)
    if 'down' in just_pressed: commands1.append(vxs.MoveDir.Back)
    if 'left' in just_pressed: commands1.append(vxs.MoveDir.Left)
    if 'right' in just_pressed: commands1.append(vxs.MoveDir.Right)
    if "'" in just_pressed: commands1.append(vxs.MoveDir.Up)
    if "/" in just_pressed: commands1.append(vxs.MoveDir.Down)
    if not action1 or commands1 != commands1_cl:
        if len(commands1) > 0:
            intent1 = vxs.ActionIntent(0.8, yaw_delta1, commands1)
            agent1.agent.perform_oneshot_py(intent1)

    # Dynamics updates
    chase0 = agent0.chaser.step_chase_py(agent0.agent, delta)
    agent0.dynamics.update_agent_dynamics_py(agent0.agent, env, chase0, delta)
    chase1 = agent1.chaser.step_chase_py(agent1.agent, delta)
    agent1.dynamics.update_agent_dynamics_py(agent1.agent, env, chase1, delta)

    # Clear edge-triggered inputs after processing both agents
    just_pressed.clear()

    # Collisions info per agent
    col0 = world.collisions_py(agent0.agent.get_pos(), [0.5, 0.5, 0.3])
    if len(col0) > 0:
        print(f"Agent 0: {len(col0)} collisions")
    col1 = world.collisions_py(agent1.agent.get_pos(), [0.5, 0.5, 0.3])
    if len(col1) > 0:
        print(f"Agent 1: {len(col1)} collisions")

    client.send_agents_py({0: agent0.agent, 1: agent1.agent})
    d = time.time() - t_loop
    if d < delta:
        time.sleep(delta - d)

listener.join()
