cmake_minimum_required(VERSION 3.16)
project(wholebody_s0_offline LANGUAGES CXX)

if(NOT IS_DIRECTORY "${TELEOP_ROOT}/src/whole_body_teleop/test/replay")
  message(FATAL_ERROR "Set TELEOP_ROOT to the whole-body-teleop workspace")
endif()

add_subdirectory("${TELEOP_ROOT}/src/whole_body_teleop/test/replay" teleop-replay EXCLUDE_FROM_ALL)
add_executable(wholebody_s0_ik replay_ik.cpp)
target_compile_features(wholebody_s0_ik PRIVATE cxx_std_20)
target_link_libraries(wholebody_s0_ik PRIVATE replay_core)
