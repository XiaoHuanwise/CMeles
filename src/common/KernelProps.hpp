/// @file KernelProps.hpp
/// @brief Shared finalisation of the OCCA JIT kernel properties.
///
/// Every CMeles kernel build site (DgField, Blas, MathOps, the time
/// module, the Riemann test) applies the same two settings through
/// finaliseKernelProps:
///
/// 1. `serial/include_std` — the CPU backends (Serial/OpenMP) need the
///    injection of `<cmath>` for the math built-ins used in .okl sources
///    (sqrt in llf/estimate_dt); GPU parsers ignore the property.
/// 2. An explicit `-O3 -march=native` for the CPU backends: OCCA's OpenMP
///    mode appends `-fopenmp` to `compiler_flags` before delegating to
///    the Serial builder, which then sees a non-empty field and skips its own
///    `-O3` default — OpenMP kernels would compile unoptimised. GPU
///    backends must not receive `-march=native` (invalid for
///    clBuildProgram / nvcc), hence the mode check.
///
/// The OpenCL `-O3` append is governed by the `OpenclO3` gate below
/// ([occa] opencl_o3 in the TOML configuration).

#pragma once

#include <atomic>

#include <occa.hpp>

namespace cmeles {

/// @brief Default tile size for the `@tile(TILE_SIZE, @outer, @inner)`
///        batching of every CMeles OKL kernel; injected as the `TILE_SIZE`
///        JIT define at the kernel-build sites.
inline constexpr int DefaultTileSize = 256;

/// @brief Global gate for the OpenCL `-O3` kernel-build flag ([occa]
///        opencl_o3, set once at startup). The flag is an
///        implementation-defined `clBuildProgram` option: accepted by the
///        clang-based OpenCL stacks (AMD/Intel/POCL, where it mainly guards
///        against conservative driver defaults), but rejected outright by
///        NVIDIA's OpenCL implementation ("Don't understand command line
///        argument \"-O3\"") — on NVIDIA devices set opencl_o3 = false;
///        NVIDIA's compiler already optimises fully by default.
class OpenclO3 {
public:
    /// @brief Set the gate (see the class comment).
    static void setEnabled(bool enabled) noexcept {
        enabled_.store(enabled, std::memory_order_relaxed);
    }

    /// @brief Whether the OpenCL kernel build appends `-O3` (default true).
    static bool enabled() noexcept {
        return enabled_.load(std::memory_order_relaxed);
    }

private:
    inline static std::atomic<bool> enabled_{true};
};

/// @brief Apply the JIT settings shared by all CMeles device kernels.
/// @param props  Kernel properties under construction (modified in place).
/// @param device Target device (its mode selects the CPU/GPU handling).
inline void finaliseKernelProps(occa::json &props, occa::device &device) {
    props["serial/include_std"] = true;
    const std::string mode      = device.mode();
    if (mode == "Serial" || mode == "OpenMP") {
        props["compiler_flags"] += " -O3 -march=native";
    } else if (mode == "OpenCL" && OpenclO3::enabled()) {
        // Implementation-defined clBuildProgram option; gated by
        // [occa] opencl_o3 because NVIDIA's OpenCL rejects it — see
        // OpenclO3.
        props["compiler_flags"] += " -O3";
    }
}

} // namespace cmeles
