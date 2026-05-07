cmake_minimum_required(VERSION 3.14)
project(neon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library
add_library(neon INTERFACE)
target_include_directories(neon INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Examples
add_executable(xor_example examples/xor_example.cpp)
target_link_libraries(xor_example PRIVATE neon)

# Tests (optional, using simple assert-based tests)
add_executable(run_tests tests/test_tensor.cpp)
target_link_libraries(run_tests PRIVATE neon)

# Enable optimizations in Release
target_compile_options(xor_example PRIVATE
    $<$<CONFIG:Release>:-O2 -march=native>
)
target_compile_options(run_tests PRIVATE
    $<$<CONFIG:Release>:-O2 -march=native>
)
