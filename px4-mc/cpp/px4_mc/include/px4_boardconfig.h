#pragma once

// Minimal POSIX board config for building PX4 control libraries
// in an external environment.

#ifndef __PX4_POSIX
#define __PX4_POSIX 1
#endif

// Default board root path for POSIX; not used by this binding.
#ifndef CONFIG_BOARD_ROOT_PATH
#define CONFIG_BOARD_ROOT_PATH "/tmp"
#endif

