/// @file test_blas.cpp
/// @brief Correctness tests for the OCCA-backed BLAS 1-3 kernels.
///
/// Tests run on Serial (mandatory), OpenMP (if available), and OpenCL
/// (if available, separate memory space).  Reference results are computed
/// via Eigen on the host.

#include "blas/Blas.hpp"
#include "common/Types.hpp"
#include "core/DeviceMemoryManager.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
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
// On unified-memory backends, wrapOrMalloc aliases host memory and copyToHost
// is a no-op — the data is already in `hostAlias`.  On separate-memory
// backends, copyToHost performs an explicit device→host transfer.
// ============================================================================
static void readResult(DeviceMemoryManager &mem, occa::memory &o_data,
                       Real *dst, occa::dim_t entries, const Real *hostAlias)
{
    if (mem.hasSeparateMemorySpace())
    {
        mem.copyToHost(o_data, dst, entries);
    }
    else
    {
        std::memcpy(dst, hostAlias, static_cast<std::size_t>(entries) * sizeof(Real));
    }
}

// ============================================================================
// Tolerance check
// ============================================================================
static bool checkRelative(Real computed, Real reference, Real tol,
                          const std::string &label)
{
    const Real denom = std::max(std::abs(reference), RealEpsilon);
    const Real relErr = std::abs(computed - reference) / denom;
    if (relErr > tol)
    {
        std::cerr << "  FAIL " << label << ": computed=" << computed
                  << " ref=" << reference << " relErr=" << relErr
                  << " tol=" << tol << '\n';
        return false;
    }
    return true;
}

// ============================================================================
// Run all BLAS tests on a single backend
// ============================================================================
static bool runBlasTests(const std::string &mode, occa::json &props)
{
    std::cout << "\n===== BLAS tests [" << mode << "] =====" << std::endl;

    occa::device device(props);
    DeviceMemoryManager mem(device);
    Blas blas(device, mem);

    bool allPass = true;
    const Real tol = Real(1e3) * RealEpsilon;

    // ---- Test sizes (cover tile boundaries) ----
    const std::vector<int> sizes = {1, 2, 255, 256, 257, 1000, 4096};

    // ========================================================================
    // BLAS-1 element-wise: scal, axpy, copy
    // ========================================================================
    std::cout << "  --- BLAS-1 element-wise ---" << std::endl;

    for (int n : sizes)
    {
        const Real alpha = Real(2.5);

        // -- scal: x = alpha * x --
        {
            VectorXr h_x    = VectorXr::Random(n);
            VectorXr h_xRef = h_x;
            occa::memory o_x = mem.wrapOrMalloc(h_x.data(), n);
            blas.scal(n, alpha, o_x);
            h_xRef *= alpha;
            device.finish();
            VectorXr h_r(n);
            readResult(mem, o_x, h_r.data(), n, h_x.data());
            if ((h_r - h_xRef).template lpNorm<Eigen::Infinity>() >
                tol * h_xRef.template lpNorm<Eigen::Infinity>())
            { allPass = false; std::cerr << "  FAIL scal n=" << n << '\n'; }
        }

        // -- copy: y = x --
        {
            VectorXr h_x = VectorXr::Random(n);
            VectorXr h_y = VectorXr::Zero(n);
            VectorXr h_yRef = h_x;
            occa::memory o_x = mem.wrapOrMalloc(h_x.data(), n);
            occa::memory o_y = mem.wrapOrMalloc(h_y.data(), n);
            blas.copy(n, o_x, o_y);
            device.finish();
            VectorXr h_r(n);
            readResult(mem, o_y, h_r.data(), n, h_y.data());
            if ((h_r - h_yRef).template lpNorm<Eigen::Infinity>() > tol)
            { allPass = false; std::cerr << "  FAIL copy n=" << n << '\n'; }
        }

        // -- axpy: y = alpha * x + y --
        {
            VectorXr h_x    = VectorXr::Random(n);
            VectorXr h_y    = VectorXr::Random(n);
            VectorXr h_yRef = alpha * h_x + h_y;
            occa::memory o_x = mem.wrapOrMalloc(h_x.data(), n);
            occa::memory o_y = mem.wrapOrMalloc(h_y.data(), n);
            blas.axpy(n, alpha, o_x, o_y);
            device.finish();
            VectorXr h_r(n);
            readResult(mem, o_y, h_r.data(), n, h_y.data());
            if ((h_r - h_yRef).template lpNorm<Eigen::Infinity>() >
                tol * h_yRef.template lpNorm<Eigen::Infinity>())
            { allPass = false; std::cerr << "  FAIL axpy n=" << n << '\n'; }
        }

        // -- axpy(alpha=0): y unchanged --
        {
            VectorXr h_x    = VectorXr::Random(n);
            VectorXr h_y    = VectorXr::Random(n);
            VectorXr h_yRef = h_y;
            occa::memory o_x = mem.wrapOrMalloc(h_x.data(), n);
            occa::memory o_y = mem.wrapOrMalloc(h_y.data(), n);
            blas.axpy(n, Real(0), o_x, o_y);
            device.finish();
            VectorXr h_r(n);
            readResult(mem, o_y, h_r.data(), n, h_y.data());
            if ((h_r - h_yRef).template lpNorm<Eigen::Infinity>() > tol)
            { allPass = false; std::cerr << "  FAIL axpy(alpha=0) n=" << n << '\n'; }
        }
    }

    // ========================================================================
    // BLAS-1 reductions: dot, nrm2, asum
    // ========================================================================
    std::cout << "  --- BLAS-1 reductions ---" << std::endl;

    // Extra large size to exercise multi-level reduction (> 256*256)
    const std::vector<int> redSizes = {1, 255, 256, 257, 1000, 65537};

    for (int n : redSizes)
    {
        VectorXr h_x = VectorXr::Random(n);
        VectorXr h_y = VectorXr::Random(n);

        occa::memory o_x = mem.wrapOrMalloc(h_x.data(), n);
        occa::memory o_y = mem.wrapOrMalloc(h_y.data(), n);

        Real computed, ref;

        computed = blas.dot(n, o_x, o_y);
        ref      = h_x.dot(h_y);
        if (!checkRelative(computed, ref, tol, "dot n=" + std::to_string(n)))
            allPass = false;

        computed = blas.nrm2(n, o_x);
        ref      = h_x.norm();
        if (!checkRelative(computed, ref, tol, "nrm2 n=" + std::to_string(n)))
            allPass = false;

        computed = blas.asum(n, o_x);
        ref      = h_x.template lpNorm<1>();
        if (!checkRelative(computed, ref, tol, "asum n=" + std::to_string(n)))
            allPass = false;
    }

    // n=0 edge cases
    {
        occa::memory o_d0 = mem.wrapOrMalloc(occa::dim_t(0));
        if (blas.dot(0, o_d0, o_d0) != Real(0))
        { allPass = false; std::cerr << "  FAIL dot n=0\n"; }
        if (blas.nrm2(0, o_d0) != Real(0))
        { allPass = false; std::cerr << "  FAIL nrm2 n=0\n"; }
        if (blas.asum(0, o_d0) != Real(0))
        { allPass = false; std::cerr << "  FAIL asum n=0\n"; }
    }

    // ========================================================================
    // BLAS-2: gemv, ger
    // ========================================================================
    std::cout << "  --- BLAS-2 ---" << std::endl;

    {
        const int m = 64, n = 32;
        const Real alpha = Real(1.5), beta = Real(0.75);

        // gemv
        {
            MatrixXr h_A = MatrixXr::Random(m, n);
            VectorXr h_x = VectorXr::Random(n);
            VectorXr h_y = VectorXr::Random(m);
            VectorXr h_yRef = alpha * (h_A * h_x) + beta * h_y;

            occa::memory o_A = mem.wrapOrMalloc(h_A.data(), m * n);
            occa::memory o_x = mem.wrapOrMalloc(h_x.data(), n);
            occa::memory o_y = mem.wrapOrMalloc(h_y.data(), m);

            blas.gemv(m, n, alpha, o_A, o_x, beta, o_y);
            device.finish();
            VectorXr h_r(m);
            readResult(mem, o_y, h_r.data(), m, h_y.data());
            if ((h_r - h_yRef).template lpNorm<Eigen::Infinity>() >
                tol * h_yRef.template lpNorm<Eigen::Infinity>())
            { allPass = false; std::cerr << "  FAIL gemv\n"; }
        }

        // ger
        {
            MatrixXr h_A    = MatrixXr::Random(m, n);
            MatrixXr h_ARef = h_A;
            VectorXr h_x    = VectorXr::Random(m);
            VectorXr h_y    = VectorXr::Random(n);
            h_ARef += alpha * h_x * h_y.transpose();

            occa::memory o_A = mem.wrapOrMalloc(h_A.data(), m * n);
            occa::memory o_x = mem.wrapOrMalloc(h_x.data(), m);
            occa::memory o_y = mem.wrapOrMalloc(h_y.data(), n);

            blas.ger(m, n, alpha, o_x, o_y, o_A);
            device.finish();
            MatrixXr h_r(m, n);
            readResult(mem, o_A, h_r.data(), m * n, h_A.data());
            if ((h_r - h_ARef).template lpNorm<Eigen::Infinity>() >
                tol * h_ARef.template lpNorm<Eigen::Infinity>())
            { allPass = false; std::cerr << "  FAIL ger\n"; }
        }
    }

    // ========================================================================
    // BLAS-3: gemm
    // ========================================================================
    std::cout << "  --- BLAS-3 ---" << std::endl;

    {
        struct Dims { int m, n, k; };
        const std::vector<Dims> gemmShapes = {
            {2, 3, 4},
            {16, 16, 16},
            {32, 16, 24},
            {64, 64, 64},
        };

        for (const auto &d : gemmShapes)
        {
            const int m = d.m, n = d.n, k = d.k;
            const Real alpha = Real(1.5), beta = Real(0.5);

            MatrixXr h_A = MatrixXr::Random(m, k);
            MatrixXr h_B = MatrixXr::Random(k, n);
            MatrixXr h_C = MatrixXr::Random(m, n);
            MatrixXr h_CRef = alpha * (h_A * h_B) + beta * h_C;

            occa::memory o_A = mem.wrapOrMalloc(h_A.data(), m * k);
            occa::memory o_B = mem.wrapOrMalloc(h_B.data(), k * n);
            occa::memory o_C = mem.wrapOrMalloc(h_C.data(), m * n);

            blas.gemm(m, n, k, alpha, o_A, o_B, beta, o_C);
            device.finish();
            MatrixXr h_r(m, n);
            readResult(mem, o_C, h_r.data(), m * n, h_C.data());
            const Real gemmTol = k * tol;
            const Real refNorm = h_CRef.template lpNorm<Eigen::Infinity>();
            if ((h_r - h_CRef).template lpNorm<Eigen::Infinity>() > gemmTol * refNorm)
            {
                std::cerr << "  FAIL gemm (" << m << 'x' << n << 'x' << k << ")\n";
                allPass = false;
            }
        }
    }

    // ========================================================================
    // setTileSize
    // ========================================================================
    std::cout << "  --- setTileSize ---" << std::endl;
    {
        const int n = 512;
        const Real alpha = Real(3.0);

        VectorXr h_x = VectorXr::Random(n);
        occa::memory o_x = mem.wrapOrMalloc(h_x.data(), n);

        // Default tile
        {
            VectorXr h_y    = VectorXr::Random(n);
            VectorXr h_yRef = alpha * h_x + h_y;
            occa::memory o_y = mem.wrapOrMalloc(h_y.data(), n);
            blas.axpy(n, alpha, o_x, o_y);
            device.finish();
            VectorXr h_r(n);
            readResult(mem, o_y, h_r.data(), n, h_y.data());
            if ((h_r - h_yRef).template lpNorm<Eigen::Infinity>() >
                tol * h_yRef.template lpNorm<Eigen::Infinity>())
            { allPass = false; std::cerr << "  FAIL setTileSize default\n"; }
        }

        // 128 tile
        blas.setTileSize(128);
        {
            VectorXr h_y    = VectorXr::Random(n);
            VectorXr h_yRef = alpha * h_x + h_y;
            occa::memory o_y = mem.wrapOrMalloc(h_y.data(), n);
            blas.axpy(n, alpha, o_x, o_y);
            device.finish();
            VectorXr h_r(n);
            readResult(mem, o_y, h_r.data(), n, h_y.data());
            if ((h_r - h_yRef).template lpNorm<Eigen::Infinity>() >
                tol * h_yRef.template lpNorm<Eigen::Infinity>())
            { allPass = false; std::cerr << "  FAIL setTileSize(128)\n"; }
        }
        blas.setTileSize(256);
    }

    std::cout << "  BLAS [" << mode << "]: "
              << (allPass ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return allPass;
}

// ============================================================================
// main
// ============================================================================
int main()
{
    std::cout << "==================================================" << std::endl;
    std::cout << "  CMeles BLAS Kernel Test Suite" << std::endl;
    std::cout << "  Real = " << (sizeof(Real) == sizeof(float) ? "float" : "double")
              << ", RealEpsilon = " << RealEpsilon << std::endl;
    std::cout << "==================================================" << std::endl;

    int passed  = 0;
    int skipped = 0;
    int failed  = 0;

    // Serial — always available
    {
        occa::json props = tryMakeDevice("Serial");
        if (props.isNull())
        {
            std::cerr << "FATAL: Serial backend not available.\n";
            return 1;
        }
        if (runBlasTests("Serial", props)) ++passed; else ++failed;
    }

    // OpenMP — optional
    {
        occa::json props = tryMakeDevice("OpenMP");
        if (!props.isNull())
        {
            if (runBlasTests("OpenMP", props)) ++passed; else ++failed;
        }
        else
        {
            std::cout << "\n[OpenMP] skipped (not available)" << std::endl;
            ++skipped;
        }
    }

    // OpenCL — optional (separate memory space)
    {
        occa::json props = tryMakeDevice("OpenCL");
        if (!props.isNull())
        {
            if (runBlasTests("OpenCL", props)) ++passed; else ++failed;
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
