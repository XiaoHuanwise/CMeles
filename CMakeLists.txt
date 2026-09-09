cmake_minimum_required(VERSION 3.15)
project(CMeles LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo, Debug
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto -march=native -O3 -fopenmp")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Precision control ----
option(USE_FLOAT_PRECISION "Use float (32-bit) instead of double (64-bit) as the default floating-point type" OFF)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/.occa_install")

add_executable(CMeles CMeles.cpp)

if(USE_FLOAT_PRECISION)
    target_compile_definitions(CMeles PRIVATE USE_FLOAT_PRECISION)
endif()

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
add_compile_definitions(OCCA_OKL_DIR="${CMAKE_BINARY_DIR}/occa/okl")

add_subdirectory(src)

find_package(Eigen3 REQUIRED CONFIG)
target_link_libraries(CMeles PRIVATE Eigen3::Eigen)

find_package(OCCA REQUIRED CONFIG)
target_link_libraries(CMeles PRIVATE OCCA::libocca)

# Enable the in-process OpenMP thread-count override ([occa] threads);
# OCCA's own OpenMP support guarantees the runtime is available.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CMeles PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(CMeles PRIVATE ${CMAKE_SOURCE_DIR}/third_party/tomlplusplus)
target_include_directories(CMeles PRIVATE ${CMAKE_SOURCE_DIR}/third_party/exprtk)

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
