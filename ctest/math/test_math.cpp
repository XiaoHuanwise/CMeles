/// @file test_math.cpp
/// @brief Correctness tests for supplementary math kernels (MathOps).

#include "common/Types.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "math/MathOps.hpp"

#include <Eigen/Dense>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// Helper: attempt to create a device for a given mode string.
// ============================================================================
static occa::json tryMakeDevice(const std::string &mode)
{
    try
    {
        occa::json props;
        props["mode"] = mode;
        if (mode == "OpenCL")
        {
            props["platform_id"] = 0;
            props["device_id"]   = 0;
        }
        occa::device dev(props);
        if (dev.mode() != mode)
        {
            dev.free();
            return occa::json();
        }
        dev.free();
        return props;
    }
    catch (const std::exception &e)
    {
        std::cout << "  (skip " << mode << ": " << e.what() << ")\n";
        return occa::json();
    }
}

// ============================================================================
// Helper: read result from device memory accounting for memory model.
// ============================================================================
static void readResult(DeviceMemoryManager &mem, occa::memory &o_data,
                       Real *dst, occa::dim_t entries, const Real *hostAlias)
{
    if (mem.hasSeparateMemorySpace())
        mem.copyToHost(o_data, dst, entries);
    else
        std::memcpy(dst, hostAlias, static_cast<std::size_t>(entries) * sizeof(Real));
}

// ============================================================================
// Run all math tests on a single backend
// ============================================================================
static bool runMathTests(const std::string &mode, occa::json &props)
{
    std::cout << "\n===== Math tests [" << mode << "] =====" << std::endl;

    occa::device device(props);
    DeviceMemoryManager mem(device);
    MathOps math(device, mem);

    bool allPass = true;
    const Real tol = Real(1e3) * RealEpsilon;

    const std::vector<int> sizes = {1, 2, 255, 256, 257, 1000};

    // ========================================================================
    // vmul: z = x .* y  (Hadamard product)
    // ========================================================================
    std::cout << "  --- vmul ---" << std::endl;

    for (int n : sizes)
    {
        VectorXr h_x = VectorXr::Random(n);
        VectorXr h_y = VectorXr::Random(n);
        VectorXr h_z = VectorXr::Zero(n);
        VectorXr h_ref = h_x.array() * h_y.array();

        occa::memory o_x = mem.wrapOrMalloc(h_x.data(), n);
        occa::memory o_y = mem.wrapOrMalloc(h_y.data(), n);
        occa::memory o_z = mem.wrapOrMalloc(h_z.data(), n);

        math.vmul(n, o_x, o_y, o_z);
        device.finish();

        VectorXr h_r(n);
        readResult(mem, o_z, h_r.data(), n, h_z.data());
        if ((h_r - h_ref).template lpNorm<Eigen::Infinity>() >
            tol * h_ref.template lpNorm<Eigen::Infinity>())
        {
            std::cerr << "  FAIL vmul n=" << n << '\n';
            allPass = false;
        }
    }

    std::cout << "  Math [" << mode << "]: "
              << (allPass ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return allPass;
}

// ============================================================================
// main
// ============================================================================
int main()
{
    std::cout << "==================================================" << std::endl;
    std::cout << "  CMeles Math Kernel Test Suite" << std::endl;
    std::cout << "  Real = " << (sizeof(Real) == sizeof(float) ? "float" : "double")
              << ", RealEpsilon = " << RealEpsilon << std::endl;
    std::cout << "==================================================" << std::endl;

    int passed  = 0;
    int skipped = 0;
    int failed  = 0;

    {
        occa::json props = tryMakeDevice("Serial");
        if (props.isNull())
        {
            std::cerr << "FATAL: Serial backend not available.\n";
            return 1;
        }
        if (runMathTests("Serial", props)) ++passed; else ++failed;
    }

    {
        occa::json props = tryMakeDevice("OpenMP");
        if (!props.isNull())
        {
            if (runMathTests("OpenMP", props)) ++passed; else ++failed;
        }
        else
        {
            std::cout << "\n[OpenMP] skipped (not available)" << std::endl;
            ++skipped;
        }
    }

    {
        occa::json props = tryMakeDevice("OpenCL");
        if (!props.isNull())
        {
            if (runMathTests("OpenCL", props)) ++passed; else ++failed;
        }
        else
        {
            std::cout << "\n[OpenCL] skipped (not available)" << std::endl;
            ++skipped;
        }
    }

    std::cout << "\n==================================================" << std::endl;
    std::cout << "  Results: " << passed << " passed, " << skipped
              << " skipped, " << failed << " failed" << std::endl;
    std::cout << "==================================================" << std::endl;

    return failed > 0 ? 1 : 0;
}
