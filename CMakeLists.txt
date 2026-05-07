cmake_minimum_required(VERSION 3.14)
project(neon VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Library (header-only) ──────────────────────────────────────
add_library(neon INTERFACE)
target_include_directories(neon INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ─── Third-party headers ────────────────────────────────────────
add_library(neon_third_party INTERFACE)
target_include_directories(neon_third_party INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
)

# ─── Examples ───────────────────────────────────────────────────
add_executable(xor_example examples/xor_example.cpp)
target_link_libraries(xor_example PRIVATE neon)

# ─── Tests ──────────────────────────────────────────────────────
add_executable(run_tests tests/test_tensor.cpp)
target_link_libraries(run_tests PRIVATE neon)

# ─── Web Server ─────────────────────────────────────────────────
add_executable(neon_server src/server_main.cpp)
target_link_libraries(neon_server PRIVATE neon neon_third_party)
target_compile_options(neon_server PRIVATE -pthread)
target_link_options(neon_server PRIVATE -pthread)

# ─── Optimizations ──────────────────────────────────────────────
target_compile_options(xor_example PRIVATE
    $<$<CONFIG:Release>:-O2 -march=native>
)
target_compile_options(run_tests PRIVATE
    $<$<CONFIG:Release>:-O2 -march=native>
)
target_compile_options(neon_server PRIVATE
    $<$<CONFIG:Release>:-O2 -march=native>
)

# ─── Install ────────────────────────────────────────────────────
install(TARGETS neon xor_example run_tests neon_server
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY web_gui/ DESTINATION share/neon/web_gui)
