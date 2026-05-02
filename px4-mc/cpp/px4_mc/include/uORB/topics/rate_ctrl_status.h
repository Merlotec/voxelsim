#pragma once
#include <stdint.h>

// Minimal uORB rate_ctrl_status message for standalone builds.
struct rate_ctrl_status_s {
    uint64_t timestamp;
    float rollspeed;
    float pitchspeed;
    float yawspeed;
    float rollspeed_integ;
    float pitchspeed_integ;
    float yawspeed_integ;
};

