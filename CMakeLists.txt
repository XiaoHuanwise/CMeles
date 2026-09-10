cmake_minimum_required(VERSION 3.15)
project(CMeles LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Build type ----
# Override with -DCMAKE_BUILD_TYPE=<Debug|Release|RelWithDebInfo|MinSizeRel>;
# defaults to Release when unspecified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)

# Profiling-friendly RelWithDebInfo (perf/flamegraph): Release-grade -O3
# codegen with frame pointers preserved for reliable stack unwinding.
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -fno-omit-frame-pointer")

# ---- Options ----
option(USE_FLOAT_PRECISION "Use float (32-bit) instead of double (64-bit) as the default floating-point type" OFF)
option(CMELES_ENABLE_LTO "Enable link-time optimization (-flto=auto)" OFF)
option(BUILD_SHARED_LIBS "Build cmeles_core as a shared library instead of a static one" OFF)
option(CMELES_ENABLE_PCH "Precompile the heavy third-party headers (occa/Eigen/toml++/exprtk) for cmeles_core" ON)

# -O3 lives in the base flags so every build type stays optimized (Debug
# becomes -O3 -g); -fopenmp must reach the link line too (libgomp), hence
# the paired add_link_options.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
add_compile_options(-march=native -fopenmp)
add_link_options(-fopenmp)
if(CMELES_ENABLE_LTO)
    add_compile_options(-flto=auto)
    add_link_options(-flto=auto)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/.occa_install")

# Refresh OKL kernel sources into the build tree on every build so kernel
# edits take effect with `cmake --build` alone. (A configure-time file(COPY)
# silently kept stale kernels around after .okl edits unless cmake was
# re-run; CONFIGURE_DEPENDS also picks up newly added kernel files.)
file(GLOB_RECURSE CMELES_OKL_KERNELS CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/blas/okl/*.okl"
     "${CMAKE_SOURCE_DIR}/src/math/okl/*.okl"
     "${CMAKE_SOURCE_DIR}/src/riemann/okl/*.okl"
     "${CMAKE_SOURCE_DIR}/src/dg/okl/*.okl"
     "${CMAKE_SOURCE_DIR}/src/time/okl/*.okl")
add_custom_target(okl_kernels ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMELES_OKL_KERNELS}
            ${CMAKE_BINARY_DIR}/occa/okl
    DEPENDS ${CMELES_OKL_KERNELS}
    COMMENT "Refreshing OKL kernels in the build tree")
# copy_if_different fails on a missing destination directory; create it at
# configure time so fresh build trees work without a manual mkdir.
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/occa/okl)
add_compile_definitions(OCCA_OKL_DIR="${CMAKE_BINARY_DIR}/occa/okl")

# ---- Core library: every solver module compiled once, linked by the main
# executable and all test executables (previously each test recompiled its
# module sources: 114 compile jobs for 32 distinct files). ----
add_library(cmeles_core)
add_subdirectory(src)
target_include_directories(cmeles_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

find_package(Eigen3 REQUIRED CONFIG)
target_link_libraries(cmeles_core PUBLIC Eigen3::Eigen)

find_package(OCCA REQUIRED CONFIG)
target_link_libraries(cmeles_core PUBLIC OCCA::libocca)

# Enable the in-process OpenMP thread-count override ([occa] threads);
# OCCA's own OpenMP support guarantees the runtime is available.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cmeles_core PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(cmeles_core PUBLIC ${CMAKE_SOURCE_DIR}/third_party/tomlplusplus)
target_include_directories(cmeles_core PUBLIC ${CMAKE_SOURCE_DIR}/third_party/exprtk)

if(USE_FLOAT_PRECISION)
    target_compile_definitions(cmeles_core PUBLIC USE_FLOAT_PRECISION)
endif()

# One precompiled header for the library's 18 TUs beats parsing occa.hpp /
# Eigen / exprtk (46k lines) in each of them. Test executables hold a single
# TU each, so a per-target PCH there would cost more than it saves.
if(CMELES_ENABLE_PCH)
    target_precompile_headers(cmeles_core PRIVATE
        <occa.hpp>
        <Eigen/Dense>
        <toml.hpp>
        <exprtk.hpp>)
endif()

# ---- Main executable ----
add_executable(CMeles CMeles.cpp)
target_link_libraries(CMeles PRIVATE cmeles_core)

find_package(HighFive REQUIRED CONFIG)
target_link_libraries(CMeles PRIVATE HighFive::HighFive)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(ctest)
endif()

# Every executable loads OKL kernels from the build tree at runtime: make
# the kernel refresh a prerequisite of each binary (the ALL keyword only
# covers default builds, not `cmake --build --target <name>`).
foreach(_tgt CMeles test_blas test_basis test_device_memory test_config
                test_dg_field test_dg_periodic test_math test_mesh_geometry
                test_mesh_device_memory test_riemann test_euler_vortex
                test_vortex_methods test_time_stepper)
    add_dependencies(${_tgt} okl_kernels)
endforeach()
