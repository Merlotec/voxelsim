#pragma once
#include <stdint.h>

// Minimal uORB trajectory_setpoint message definition for standalone builds.
struct trajectory_setpoint_s {
    uint64_t timestamp;
    float position[3];
    float velocity[3];
    float acceleration[3];
    float thrust[3];
    float yaw;
    float yawspeed;
};

