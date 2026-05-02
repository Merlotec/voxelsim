#pragma once
#include <stdint.h>

// Minimal uORB vehicle_attitude_setpoint message for standalone builds.
struct vehicle_attitude_setpoint_s {
    uint64_t timestamp;
    float q_d[4];
    float thrust_body[3];
    float yaw;      // optional, not used by wrapper
    float yawspeed; // used by AttitudeControl
    float yaw_sp_move_rate; // for PositionControl internals
};
