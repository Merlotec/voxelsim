# VoxelSim

VoxelSim is an ultra high performance simulation and perception environment built around a sparse voxel world. It is designed to enable rapid training of physical intelligence models and drone autonomy. This is great for two things:
- Rapidly training autonomous drones (e.g. using reinforcement learning).
- Benchmarking general (3D) physical intelligence models efficiently.

Because VoxelSim can do updating of the drone's PoV so quickly, allows users to quickly train/test ML models for physical intelligence. 

VoxelSim uses a highly optimised *voxel based renderer* to dramatically improve rendering performance of the drone's internal representation of the world. Any reasoning/control logic built on this platform can easily be converted to run on a real drone, since VoxelSim uses the exact PID loops from PX4.

<!-- Agents navigate a 3D grid, build an internal map from their GPU-rendered camera feed, plan smooth trajectories through the voxels, and predict future occupancy from motion history. The stack exposes everything through Python bindings (PyO3) for scripting and experimentation. -->

## Architecture

```
voxelsim/            Core — VoxelGrid, agents, A* planner, path execution, future occupancy prediction
voxelsim-compute/    GPU compute pipeline — rasterises the voxel world into the agent's FilterWorld
voxelsim-simulator/  Quad dynamics, terrain generator, optional PX4 controller (default feature)
voxelsim-renderer/   Bevy visualiser — world view + per-agent POV windows over TCP
voxelsim-py/         PyO3 shim that composes the three crates into a single `voxelsim` Python module
px4-mc/              Vendored PX4 attitude/position/rate controller C++ sources (no PX4 install needed)
examples/            Python scripts demonstrating the full loop
```

## Notable Features

### 1. GPU POV Rendering into FilterWorld

`AgentVisionRenderer` uses a wgpu compute pipeline to rasterise the voxel world from the drone's exact camera pose at each frame. The output is written into a `FilterWorld` — the agent's internal representation of what it has mapped so far. Because the rasterisation runs on the GPU, and renders actual voxel coordinates in the fragment shader,, a 100³-voxel scene renders in milliseconds, fast enough to keep up with the simulation loop. The `FilterWorld` accumulates evidence across frames so the agent progressively builds a complete internal map even as it moves. Artificial noise is then added as required, to mimic the uncertainty of the real mapping process.

### 2. Smooth Path Planning Through the Voxel Grid

`AStarActionPlanner` finds the shortest collision-free sequence of grid moves from origin to destination, padded by a configurable obstacle radius. Rather than executing this discrete path cell-by-cell, the path is lifted into a smooth trajectory: the centroid sequence is fit with a continuous spline so the drone follows a curved line through space. The drone is therefore not axis-locked — it can arc diagonally through open space while still respecting the topology of the grid search. The trajectory is stored per-agent and drawn live in the renderer as a blue spline.

### 3. Phase Trajectory Graph (PTG) - Future Cell Occupancy Prediction

`PtgSolver` maintains a rolling history of `PhaseGrid` snapshots — each recording observed occupancy across the agent's visible region at a point in time. Given a future time horizon, it projects those observations forward using a bounded-addition accumulator: cells that have been consistently occupied gain high phase values; cells that fluctuate are penalised. The resulting `PhaseGrid` is a probability field over future occupied voxels. This is sent alongside the standard POV data so the renderer can visualise predicted occupancy as a separate translucent layer.

---

## Quick Start

### 1. Build and install the Python module

```bash
pip install maturin pynput
maturin develop -m voxelsim-py/Cargo.toml --release
```

This builds all Rust crates (including the vendored PX4 controller) and installs the `voxelsim` Python module into the active virtualenv. No PX4 installation is required.

### 2. Start the renderer

The renderer is a separate native process. You need one instance for the world/agent view, plus one additional instance per agent POV stream, each passed a numeric port offset via `--virtual`.

**Single agent** (one POV stream):

```bash
# Terminal 1 — world + agent overview
cargo run -p voxelsim-renderer --release

# Terminal 2 — agent 0 POV
cargo run -p voxelsim-renderer --release -- --virtual 0
```

**Two agents** (e.g. `phasetest.py`):

```bash
# Terminal 1 — world + agent overview
cargo run -p voxelsim-renderer --release

# Terminal 2 — agent 0 POV
cargo run -p voxelsim-renderer --release -- --virtual 0

# Terminal 3 — agent 1 POV
cargo run -p voxelsim-renderer --release -- --virtual 1
```

The `--virtual N` offset selects the port pair for that stream: POV data arrives on `8090 + N` and agent data on `9090 + N`. The world renderer (no flag) listens for world updates on `8080` and agent positions on `8081`.

### 3. Run an example

```bash
python examples/povtest.py           # single agent, interactive WASD control
python examples/astartest.py         # single agent, A* autonomous path to target
python examples/phasetest.py         # two agents with PTG occupancy prediction
python examples/povtest_simple.py    # minimal single agent, no dynamics
```

---

## Examples

| Script | Renderer instances needed | Description |
|---|---|---|
| `povtest.py` | world + `--virtual 0` | Single agent, PX4 dynamics, manual WASD control |
| `astartest.py` | world + `--virtual 0` | Single agent autonomously follows an A*-planned spline to a fixed target |
| `phasetest.py` | world + `--virtual 0` + `--virtual 1` | Two agents, each with their own FilterWorld and PTG solver |
| `povtest_simple.py` | world + `--virtual 0` | Single agent without physics, position updates directly from action intent |
| `headless.py` | none | Legacy reference, runs without renderer |

### Controls (interactive examples)

| Key | Action |
|---|---|
| `W/S/A/D` | Forward / back / left / right |
| `Space` | Up |
| `Shift` | Down |
| `Q / E` | Yaw left / right |
| `Tab` | Toggle between orbit camera and agent POV in the virtual window |
| `Z` | Centre orbit camera on next agent |
| `ESC` | Quit |

In `phasetest.py`, agent 1 uses arrow keys for movement and `'` / `/` for up/down.

---

## Network Protocol

All messages are framed as a 4-byte little-endian length prefix followed by a bincode-serialised payload.

| Port | Stream |
|---|---|
| `8080` | VoxelGrid world updates |
| `8081` | Agent map (`HashMap<id, Agent>`) |
| `8090 + N` | POV data for virtual stream N (`PovData` — virtual world + phase grid + camera projection) |
| `9090 + N` | Agent positions for virtual stream N |

---

## Python API Reference

### World

```python
vxs.VoxelGrid.from_dict_py({(x,y,z): vxs.Cell.filled(), ...}) -> VoxelGrid
vxs.VoxelGrid.to_dict_py()           -> Dict[Tuple[int,int,int], Cell]
vxs.VoxelGrid.as_numpy()             -> (coords: ndarray[N,3], values: ndarray[N])
vxs.VoxelGrid.collisions_py(pos, dims) -> List[(coord, Cell)]
vxs.Cell.filled() / vxs.Cell.sparse()
```

### Agents and actions

```python
agent = vxs.Agent(id)
agent.set_hold_py(coord, yaw)
agent.get_pos() / agent.get_coord_py()
agent.camera_view_py(orientation) -> CameraView

intent = vxs.ActionIntent(urgency, yaw_delta, move_dirs)
agent.perform_oneshot_py(intent)   # replace current action
agent.push_back_intent_py(intent)  # queue behind current action
agent.get_action_py() -> Optional[Action]
action.get_intent_queue()          # List[ActionIntent]

vxs.MoveDir.Forward / Back / Left / Right / Up / Down
```

### Planning

```python
planner = vxs.AStarActionPlanner(padding)
intent = planner.plan_action_py(world, origin, destination, urgency, yaw)
```

### Compute pipeline (FilterWorld and POV)

```python
fw  = vxs.FilterWorld()
noise = vxs.NoiseParams.default_with_seed_py([sx, sy, sz])
renderer = vxs.AgentVisionRenderer(world, [width, height], noise)

# Render the current frame into fw (GPU, async callback on completion)
renderer.update_filter_world_py(camera_view, proj, fw, dyn_world, timestamp, callback)

# Render and return a WorldChangeset (for multi-agent use)
renderer.render_changeset_py(camera_view, proj, fw, dyn_world, timestamp, callback)

# Send POV to the renderer process
fw.send_pov_py(client, stream_idx, agent_id, proj, orientation, phase_grid)
fw.send_pov_async_py(client, stream_idx, agent_id, proj, orientation, phase_grid)
fw.is_updating_py(last_timestamp) -> bool
```

### Phase Trajectory Graph (PTG) occupancy prediction

```python
solver = vxs.PtgSolver.default_py()
phase_grid = vxs.PhaseGrid()

solver.add_phase_frame_py(agent, virtual_world, timestamp)
phase_grid = solver.gen_phase_grid_py(agent, t_start, t_end)
```

### Dynamics and simulation

```python
# PX4 cascaded position/attitude/rate controller (default feature)
dynamics = vxs.px4.Px4Dynamics.default_py()

# Simple PID quad model
dynamics = vxs.QuadDynamics(vxs.QuadParams.default_py())

chaser  = vxs.FixedLookaheadChaser.default_py()
env     = vxs.EnvState.default_py()

chase_target = chaser.step_chase_py(agent, dt)
dynamics.update_agent_dynamics_py(agent, env, chase_target, dt)
```

### Terrain generation

```python
gen = vxs.TerrainGenerator()
cfg = vxs.TerrainConfig.default_py()
cfg.set_world_dimensions_py(x, y, z)
gen.generate_terrain_py(cfg)
world = gen.generate_world_py()
```

### Network client

```python
# Blocking
client = vxs.RendererClient.default_localhost_py(pov_count)

# Non-blocking (sends in background threads)
client = vxs.AsyncRendererClient.default_localhost_py(pov_count)

client.send_world_py(world)
client.send_agents_py({id: agent, ...})
```

---

## Building from Source

```bash
# All Rust crates
cargo build --release

# Python extension only
maturin develop -m voxelsim-py/Cargo.toml --release

# Without PX4 dynamics (smaller binary)
maturin develop -m voxelsim-py/Cargo.toml --release --no-default-features
```

The PX4 controller sources are vendored in `px4-mc/cpp/px4_mc/vendor/` — a full PX4-Autopilot checkout is not required. If you have one and want to build against it instead, set `PX4_SRC_DIR=/path/to/PX4-Autopilot` before building.
