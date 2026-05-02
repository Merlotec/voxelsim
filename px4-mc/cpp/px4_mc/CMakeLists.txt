cmake_minimum_required(VERSION 3.15)
project(px4_mc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Use PX4_SRC_DIR if provided, otherwise fall back to the bundled vendor sources
set(PX4_SRC_DIR "$ENV{PX4_SRC_DIR}" CACHE PATH "Path to PX4-Autopilot source tree")
if(NOT PX4_SRC_DIR OR PX4_SRC_DIR STREQUAL "")
    set(PX4_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/vendor")
    message(STATUS "PX4_SRC_DIR not set — using bundled vendor sources at ${PX4_SRC_DIR}")
else()
    message(STATUS "PX4_SRC_DIR=${PX4_SRC_DIR}")
endif()

add_library(px4_mc STATIC
    src/px4_mc_wrapper.cpp
)

target_include_directories(px4_mc
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${PX4_SRC_DIR}/src
        ${PX4_SRC_DIR}/src/include
        ${PX4_SRC_DIR}/src/lib
        ${PX4_SRC_DIR}/src/modules
        ${PX4_SRC_DIR}/platforms/common/include
        ${PX4_SRC_DIR}/src/modules/mc_pos_control/PositionControl
        ${PX4_SRC_DIR}/src/modules/mc_att_control/AttitudeControl
)

# Require source list
set(PX4_SOURCES_CMAKE ${CMAKE_CURRENT_LIST_DIR}/px4_sources.cmake)
if(NOT EXISTS ${PX4_SOURCES_CMAKE})
    message(FATAL_ERROR "${PX4_SOURCES_CMAKE} not found. Create it with the PX4 .cpp files to compile.")
endif()
message(STATUS "Including ${PX4_SOURCES_CMAKE} for PX4 sources")
include(${PX4_SOURCES_CMAKE})
target_sources(px4_mc PRIVATE ${PX4_MC_PX4_SOURCES})

if(APPLE)
    target_link_libraries(px4_mc PUBLIC c++)
elseif(UNIX)
    target_link_libraries(px4_mc PUBLIC stdc++)
endif()

install(TARGETS px4_mc ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
