# px4-mc

Thin Rust bindings to PX4 multicopter controllers (position/attitude/rate), built as a CMake static library and exposed via a C FFI. Designed for use inside a Rust simulator with nalgebra types.

## Overview

- C++ static lib: `cpp/px4_mc` built via CMake
- Uses exact PX4 source files (you provide the PX4 checkout)
- Thin C wrapper exposes: create, update, destroy
- Rust crate converts nalgebra types to FFI arrays

## Setup

1. Acquire PX4 source
   - Clone PX4-Autopilot somewhere on disk.
   - Note the path, e.g. `/path/to/PX4-Autopilot`.

2. Point the build to PX4 sources
   - Option A: env var `PX4_SRC_DIR=/path/to/PX4-Autopilot`
   - Option B: pass `-DPX4_SRC_DIR=...` through your build system

3. List exact PX4 source files
   - Copy `cpp/px4_mc/px4_sources.cmake.example` to `cpp/px4_mc/px4_sources.cmake`
   - Edit it to include the exact `.cpp` files for:
     - mc_pos_control/PositionControl
     - mc_att_control/AttitudeControl
     - mc_rate_control/RateControl
     - Required libs: `src/lib/matrix`, `src/lib/mathlib` (and any others your PX4 version needs)

4. Build

```
cd px4-mc
PX4_SRC_DIR=/path/to/PX4-Autopilot cargo build
```

If `PX4_SRC_DIR` or `px4_sources.cmake` are not configured, the wrapper builds but returns zero torque/thrust.

## Rust API

```
use px4_mc::Px4McController;
use nalgebra::{Vector3, UnitQuaternion, Quaternion};

let mut ctrl = Px4McController::new(0.002, 1.2); // dt [s], mass [kg]
let pos = Vector3::new(0.0, 0.0, 0.0);
let vel = Vector3::new(0.0, 0.0, 0.0);
let att = UnitQuaternion::identity();
let rates = Vector3::new(0.0, 0.0, 0.0);
let target_pos = Vector3::new(0.0, 0.0, -1.0);
let target_vel = Vector3::new(0.0, 0.0, 0.0);

let (torque, thrust) = ctrl.update(&pos, &vel, &att, &rates, &target_pos, &target_vel);
```

## Notes

- This crate intentionally does not re-implement logic; it wires PX4 controllers together. Depending on PX4 version, you may need to include additional support sources for the library to link.
- The top-level orchestration inside `px4_mc_wrapper.cpp` is currently a stub; once PX4 sources are included, connect the Position/Attitude/Rate control calls using the upstream APIs.

