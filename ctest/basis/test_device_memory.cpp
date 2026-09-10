/// @file test_device_memory.cpp
/// @brief Verification tests for DeviceMemoryManager and OCCA device memory
///        allocation in BasisFunctions1D and BasisFunctions2D.

#include "basis/BasisFunctions1D.hpp"
#include "basis/BasisFunctions2D.hpp"
#include "common/Types.hpp"
#include "core/DeviceMemoryManager.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// ============================================================================
// Helper: attempt to create a device for a given mode string.
// Returns an empty json on failure so the caller can skip.
// ============================================================================
static occa::json tryMakeDevice(const std::string &mode) {
    try {
        occa::json props;
        props["mode"] = mode;
        if (mode == "OpenCL") {
            props["platform_id"] = 0;
            props["device_id"]   = 0;
        }
        occa::device dev(props);
        // OCCA may silently fall back to Serial when the mode is
        // compiled in but no hardware/driver is available.
        if (dev.mode() != mode) {
            dev.free();
            return occa::json();
        }
        dev.free();
        return props;
    } catch (const std::exception &e) {
        std::cout << "  (skip " << mode << ": " << e.what() << ")\n";
        return occa::json();
    }
}

// ============================================================================
// Test 1: DeviceMemoryManager construction and hasSeparateMemorySpace
// ============================================================================
static bool testMemoryManagerConstruction() {
    std::cout << "\n=== Test 1: DeviceMemoryManager construction ===\n";

    // Serial backend is always available with OCCA.
    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mgr(device);

    bool pass = true;

    // Serial is a unified-memory backend
    if (mgr.hasSeparateMemorySpace()) {
        std::cout << "  FAIL: Serial backend should NOT report separate memory "
                     "space\n";
        pass = false;
    } else {
        std::cout << "  Serial: hasSeparateMemorySpace = false  PASS\n";
    }

    if (&mgr.device() != &device) {
        std::cout << "  FAIL: device() returned wrong reference\n";
        pass = false;
    } else {
        std::cout << "  device() reference  PASS\n";
    }

    return pass;
}

// ============================================================================
// Test 2: wrapOrMalloc with host pointer (unified space — Serial)
// ============================================================================
static bool testWrapOrMallocUnified() {
    std::cout << "\n=== Test 2: wrapOrMalloc with host pointer (Serial) ===\n";

    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mgr(device);

    bool pass = true;

    constexpr int N = 10;
    std::vector<Real> host_data(N);
    for (int i = 0; i < N; ++i) {
        host_data[i] = static_cast<Real>(i + 1);
    }

    // For Serial (unified space), wrapOrMalloc should wrap the host pointer.
    occa::memory o_data = mgr.wrapOrMalloc(host_data.data(), N);

    if (!o_data.isInitialized()) {
        std::cout << "  FAIL: wrapOrMalloc returned uninitialised memory\n";
        pass = false;
    } else {
        std::cout << "  memory.isInitialized()  PASS\n";
    }

    if (o_data.size() != N) {
        std::cout << "  FAIL: size mismatch (" << o_data.size() << " vs " << N
                  << ")\n";
        pass = false;
    } else {
        std::cout << "  memory.size() == " << N << "  PASS\n";
    }

    // In unified space: o_data.ptr<Real>() should equal host_data.data()
    if (o_data.ptr<Real>() != host_data.data()) {
        std::cout << "  FAIL: wrapMemory should reuse host pointer "
                  << "(ptr: " << o_data.ptr<Real>()
                  << " vs host: " << static_cast<void *>(host_data.data())
                  << ")\n";
        pass = false;
    } else {
        std::cout << "  ptr equality (wrapMemory path)  PASS\n";
    }

    // Data integrity: values should match
    bool integrity = true;
    Real *dev_ptr  = o_data.ptr<Real>();
    for (int i = 0; i < N; ++i) {
        if (std::abs(dev_ptr[i] - host_data[i]) > RealEpsilon) {
            integrity = false;
            break;
        }
    }
    if (!integrity) {
        std::cout << "  FAIL: device data does not match host data\n";
        pass = false;
    } else {
        std::cout << "  data integrity  PASS\n";
    }

    return pass;
}

// ============================================================================
// Test 3: wrapOrMalloc zero-init (all backends)
// ============================================================================
static bool testWrapOrMallocZeroInit() {
    std::cout << "\n=== Test 3: wrapOrMalloc zero-initialised (Serial) ===\n";

    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mgr(device);

    bool pass = true;

    constexpr int N     = 20;
    occa::memory o_data = mgr.wrapOrMalloc(N);

    if (!o_data.isInitialized()) {
        std::cout << "  FAIL: zero-init memory is uninitialised\n";
        return false;
    }

    if (o_data.size() != N) {
        std::cout << "  FAIL: size mismatch\n";
        pass = false;
    }

    // Zero-init malloc should give zeroed memory
    Real *ptr   = o_data.ptr<Real>();
    Real maxVal = Real(0);
    for (int i = 0; i < N; ++i) {
        maxVal = std::max(maxVal, std::abs(ptr[i]));
    }

    if (maxVal > RealEpsilon) {
        std::cout << "  FAIL: zero-init memory contains non-zero values"
                  << " (max|v| = " << std::scientific << maxVal << ")\n";
        pass = false;
    } else {
        std::cout << "  all zero  PASS\n";
    }

    return pass;
}

// ============================================================================
// Test 4: copyToHost / copyFromHost (unified space — no-op)
// ============================================================================
static bool testCopyUnified() {
    std::cout
        << "\n=== Test 4: copyToHost / copyFromHost (unified space) ===\n";

    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mgr(device);

    bool pass = true;

    constexpr int N = 8;
    std::vector<Real> src(N), dst(N, Real(-999));
    for (int i = 0; i < N; ++i) {
        src[i] = static_cast<Real>(i * 3 + 1);
    }

    // Wrap host data
    occa::memory o_data = mgr.wrapOrMalloc(src.data(), N);

    // copyToHost should be no-op in unified space; but dst was -999
    // and o_data shares src memory, so dst should still be -999.
    mgr.copyToHost(o_data, dst.data(), N);

    // copyToHost in unified space is no-op → dst unchanged
    bool dstUnchanged = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(dst[i] - Real(-999)) > RealEpsilon) {
            dstUnchanged = false;
            break;
        }
    }

    if (!dstUnchanged) {
        std::cout << "  FAIL: copyToHost modified dst in unified space (no-op "
                     "expected)\n";
        pass = false;
    } else {
        std::cout << "  copyToHost no-op  PASS\n";
    }

    // copyFromHost: copy from a different buffer into o_data
    // In unified space this is also no-op — underlying pointer stays at src.
    std::vector<Real> new_data(N, Real(42));
    mgr.copyFromHost(o_data, new_data.data(), N);

    if (o_data.ptr<Real>() == src.data()) {
        std::cout << "  copyFromHost ptr unchanged (no-op)  PASS\n";
    } else {
        std::cout << "  FAIL: copyFromHost changed ptr in unified space\n";
        pass = false;
    }

    return pass;
}

// ============================================================================
// Test 5: BasisFunctions1D allocateDeviceMemory (Serial)
// ============================================================================
static bool testBasis1DDeviceMemory() {
    std::cout << "\n=== Test 5: BasisFunctions1D allocateDeviceMemory ===\n";

    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mgr(device);

    bool pass = true;

    constexpr int N  = 3;
    constexpr int Nq = 5;

    BasisFunctions1D basis(N, Nq);
    basis.allocateDeviceMemory(mgr);

    // Check every o_ handle is initialised
    auto checkMem = [&](occa::memory mem, const char *label,
                        int expectedSize) -> bool {
        if (!mem.isInitialized()) {
            std::cout << "  FAIL: o_" << label << " is not initialised\n";
            return false;
        }
        if (mem.size() != expectedSize) {
            std::cout << "  FAIL: o_" << label << " size " << mem.size()
                      << " != expected " << expectedSize << "\n";
            return false;
        }
        return true;
    };

    pass &= checkMem(basis.o_points(), "points", Nq);
    pass &= checkMem(basis.o_weights(), "weights", Nq);
    pass &= checkMem(basis.o_V(), "V", Nq * (N + 1));
    pass &= checkMem(basis.o_dV(), "dV", Nq * (N + 1));

    if (!pass) {
        return false;
    }

    // In Serial (unified space), device ptrs should equal host ptrs
    // (wrapMemory path)
    if (basis.o_points().ptr<Real>() != basis.points().data()) {
        std::cout
            << "  FAIL: o_points ptr != points ptr (wrapMemory expected)\n";
        pass = false;
    }
    if (basis.o_weights().ptr<Real>() != basis.weights().data()) {
        std::cout
            << "  FAIL: o_weights ptr != weights ptr (wrapMemory expected)\n";
        pass = false;
    }
    if (basis.o_V().ptr<Real>() != basis.vandermonde().data()) {
        std::cout << "  FAIL: o_V ptr != V ptr (wrapMemory expected)\n";
        pass = false;
    }
    if (basis.o_dV().ptr<Real>() != basis.vandermondeDerivative().data()) {
        std::cout << "  FAIL: o_dV ptr != dV ptr (wrapMemory expected)\n";
        pass = false;
    }
    std::cout << "  ptr equality (wrapMemory)  PASS\n";

    // Verify device-side data matches host-side via direct pointer read
    const Real *dev_V  = basis.o_V().ptr<Real>();
    const Real *host_V = basis.vandermonde().data();
    Real maxDiff       = Real(0);
    for (int k = 0; k < Nq * (N + 1); ++k) {
        maxDiff = std::max(maxDiff, std::abs(dev_V[k] - host_V[k]));
    }

    if (maxDiff > RealEpsilon) {
        std::cout << "  FAIL: V data mismatch (max diff = " << std::scientific
                  << maxDiff << ")\n";
        pass = false;
    } else {
        std::cout << "  V data match (max diff = " << maxDiff << ")  PASS\n";
    }

    // Same check for dV
    const Real *dev_dV  = basis.o_dV().ptr<Real>();
    const Real *host_dV = basis.vandermondeDerivative().data();
    maxDiff             = Real(0);
    for (int k = 0; k < Nq * (N + 1); ++k) {
        maxDiff = std::max(maxDiff, std::abs(dev_dV[k] - host_dV[k]));
    }
    if (maxDiff > RealEpsilon) {
        std::cout << "  FAIL: dV data mismatch (max diff = " << std::scientific
                  << maxDiff << ")\n";
        pass = false;
    } else {
        std::cout << "  dV data match (max diff = " << maxDiff << ")  PASS\n";
    }

    if (pass) {
        std::cout << "  BasisFunctions1D allocateDeviceMemory  PASS\n";
    }
    return pass;
}

// ============================================================================
// Test 6: BasisFunctions2D allocateDeviceMemory (Serial)
// ============================================================================
static bool testBasis2DDeviceMemory() {
    std::cout << "\n=== Test 6: BasisFunctions2D allocateDeviceMemory ===\n";

    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mgr(device);

    bool pass = true;

    constexpr int N  = 2;
    constexpr int Nq = 4;

    BasisFunctions1D basis1D(N, Nq);
    BasisFunctions2D basis2D(basis1D, N);
    basis2D.allocateDeviceMemory(mgr);

    int Nq2    = Nq * Nq;
    int N_base = (N + 1) * (N + 2) / 2;

    auto checkMem = [&](occa::memory mem, const char *label,
                        int expectedSize) -> bool {
        if (!mem.isInitialized()) {
            std::cout << "  FAIL: o_" << label << " is not initialised\n";
            return false;
        }
        if (mem.size() != expectedSize) {
            std::cout << "  FAIL: o_" << label << " size " << mem.size()
                      << " != expected " << expectedSize << "\n";
            return false;
        }
        return true;
    };

    pass &= checkMem(basis2D.o_quadraturePoints(), "quad_points", Nq2 * 2);
    pass &= checkMem(basis2D.o_quadratureWeights(), "quad_weights", Nq2);
    pass &= checkMem(basis2D.o_V2D(), "V2D", Nq2 * N_base);
    pass &= checkMem(basis2D.o_dV2D_r(), "dV2D_r", Nq2 * N_base);
    pass &= checkMem(basis2D.o_dV2D_s(), "dV2D_s", Nq2 * N_base);

    if (!pass) {
        return false;
    }

    // ptr equality (wrapMemory path in unified space)
    if (basis2D.o_quadraturePoints().ptr<Real>() !=
        basis2D.quadraturePoints().data()) {
        std::cout << "  FAIL: o_quad_points ptr mismatch\n";
        pass = false;
    }
    if (basis2D.o_quadratureWeights().ptr<Real>() !=
        basis2D.quadratureWeights().data()) {
        std::cout << "  FAIL: o_quad_weights ptr mismatch\n";
        pass = false;
    }
    if (basis2D.o_V2D().ptr<Real>() != basis2D.vandermonde().data()) {
        std::cout << "  FAIL: o_V2D ptr mismatch\n";
        pass = false;
    }
    std::cout << "  ptr equality (wrapMemory)  PASS\n";

    // Verify V2D data integrity
    const Real *dev_V2D  = basis2D.o_V2D().ptr<Real>();
    const Real *host_V2D = basis2D.vandermonde().data();
    Real maxDiff         = Real(0);
    for (int k = 0; k < Nq2 * N_base; ++k) {
        maxDiff = std::max(maxDiff, std::abs(dev_V2D[k] - host_V2D[k]));
    }
    if (maxDiff > RealEpsilon) {
        std::cout << "  FAIL: V2D data mismatch (max diff = " << std::scientific
                  << maxDiff << ")\n";
        pass = false;
    } else {
        std::cout << "  V2D data match (max diff = " << maxDiff << ")  PASS\n";
    }

    // Quick functional check: compute modalToNodal using host data,
    // then verify o_V2D contains same values
    VectorXr u_hat  = VectorXr::Random(N_base);
    VectorXr u_host = basis2D.modalToNodal(u_hat);

    // Recompute using o_V2D pointer interpreted as Eigen Map
    Eigen::Map<const MatrixXr> V2D_map(dev_V2D, Nq2, N_base);
    VectorXr u_dev = V2D_map * u_hat;

    Real fwdErr =
        (u_host - u_dev).norm() / std::max(RealEpsilon, u_host.norm());
    if (fwdErr > Real(1e3) * RealEpsilon) {
        std::cout << "  FAIL: o_V2D matrix-vector product mismatch"
                  << " (err = " << std::scientific << fwdErr << ")\n";
        pass = false;
    } else {
        std::cout << "  modalToNodal via o_V2D (err = " << fwdErr
                  << ")  PASS\n";
    }

    if (pass) {
        std::cout << "  BasisFunctions2D allocateDeviceMemory  PASS\n";
    }
    return pass;
}

// ============================================================================
// Test 7: OpenMP backend (if available)
// ============================================================================
static bool testOpenMPBackend() {
    std::cout << "\n=== Test 7: OpenMP backend ===\n";

    occa::json props = tryMakeDevice("OpenMP");
    if (props.isNull()) {
        return true; // not available → skip
    }

    bool pass = true;

    occa::device device(props);
    DeviceMemoryManager mgr(device);

    // OpenMP is also unified memory
    if (mgr.hasSeparateMemorySpace()) {
        std::cout << "  FAIL: OpenMP should NOT report separate memory space\n";
        pass = false;
    } else {
        std::cout << "  hasSeparateMemorySpace = false  PASS\n";
    }

    constexpr int N  = 2;
    constexpr int Nq = 3;
    BasisFunctions1D basis(N, Nq);
    basis.allocateDeviceMemory(mgr);

    if (basis.o_V().isInitialized() && basis.o_V().size() == Nq * (N + 1)) {
        // ptr equality should hold (wrapMemory in unified space)
        if (basis.o_V().ptr<Real>() == basis.vandermonde().data()) {
            std::cout << "  Basis1D ptr equality  PASS\n";
        } else {
            std::cout << "  FAIL: ptr mismatch for wrapMemory in OpenMP\n";
            pass = false;
        }
    } else {
        std::cout << "  FAIL: o_V invalid after allocateDeviceMemory\n";
        pass = false;
    }

    return pass;
}

// ============================================================================
// Test 8: OpenCL backend (mandatory)
// ============================================================================
static bool testOpenCLBackend() {
    std::cout << "\n=== Test 8: OpenCL backend ===\n";

    occa::json props = tryMakeDevice("OpenCL");
    if (props.isNull()) {
        std::cout << "  FAIL: OpenCL not available\n";
        return false;
    }

    bool pass = true;

    occa::device device(props);
    DeviceMemoryManager mgr(device);

    // OpenCL IS a separate memory space backend
    if (!mgr.hasSeparateMemorySpace()) {
        std::cout << "  FAIL: OpenCL SHOULD report separate memory space\n";
        pass = false;
    } else {
        std::cout << "  hasSeparateMemorySpace = true  PASS\n";
    }

    // Verify wrapOrMalloc uses malloc path in separate space
    constexpr int N = 5;
    std::vector<Real> host(N, Real(7));
    occa::memory o_data = mgr.wrapOrMalloc(host.data(), N);

    if (!o_data.isInitialized() || o_data.size() != N) {
        std::cout << "  FAIL: wrapOrMalloc in separate space failed\n";
        pass = false;
    } else {
        std::cout << "  wrapOrMalloc size correct  PASS\n";
    }

    // In separate space, device ptr must differ from host ptr
    if (o_data.ptr<Real>() == host.data()) {
        std::cout << "  FAIL: device ptr should differ from host"
                  << " in separate space\n";
        pass = false;
    } else {
        std::cout << "  device ptr != host ptr (separate space)  PASS\n";
    }

    // copyFromHost → copyToHost round-trip via explicit copies
    std::vector<Real> dst(N, Real(-1));
    mgr.copyFromHost(o_data, host.data(), N); // host → device
    mgr.copyToHost(o_data, dst.data(), N);    // device → host

    Real maxErr = Real(0);
    for (int i = 0; i < N; ++i) {
        maxErr = std::max(maxErr, std::abs(dst[i] - host[i]));
    }
    if (maxErr > Real(1e3) * RealEpsilon) {
        std::cout << "  FAIL: copyFromHost → copyToHost round-trip error"
                  << " (max = " << std::scientific << maxErr << ")\n";
        pass = false;
    } else {
        std::cout << "  copyToHost / copyFromHost round-trip  PASS\n";
    }

    // BasisFunctions1D via malloc path on OpenCL
    BasisFunctions1D basis1D(2, 4);
    basis1D.allocateDeviceMemory(mgr);

    auto checkO = [&](occa::memory mem, const char *label,
                      int expectedSize) -> bool {
        if (!mem.isInitialized()) {
            std::cout << "  FAIL: o_" << label << " not initialised\n";
            return false;
        }
        if (mem.size() != expectedSize) {
            std::cout << "  FAIL: o_" << label << " size " << mem.size()
                      << " != expected " << expectedSize << "\n";
            return false;
        }
        return true;
    };

    pass &= checkO(basis1D.o_points(), "points", 4);
    pass &= checkO(basis1D.o_weights(), "weights", 4);
    pass &= checkO(basis1D.o_V(), "V", 4 * 3);
    pass &= checkO(basis1D.o_dV(), "dV", 4 * 3);

    // In separate space, device ptr must differ from host ptr
    if (basis1D.o_V().ptr<Real>() == basis1D.vandermonde().data()) {
        std::cout << "  FAIL: o_V ptr equals host ptr (malloc expected)\n";
        pass = false;
    } else {
        std::cout << "  o_V device ptr != host ptr (malloc path)  PASS\n";
    }

    // But data must be recoverable via copyToHost
    std::vector<Real> dev_V(basis1D.vandermonde().size());
    occa::memory o_V = basis1D.o_V();
    mgr.copyToHost(o_V, dev_V.data(), basis1D.vandermonde().size());

    Real maxDiff       = Real(0);
    const Real *host_V = basis1D.vandermonde().data();
    for (Eigen::Index k = 0; k < basis1D.vandermonde().size(); ++k) {
        maxDiff = std::max(maxDiff, std::abs(dev_V[k] - host_V[k]));
    }
    if (maxDiff > Real(1e3) * RealEpsilon) {
        std::cout << "  FAIL: V data mismatch after copyToHost"
                  << " (max diff = " << std::scientific << maxDiff << ")\n";
        pass = false;
    } else {
        std::cout << "  V data round-trip via copyToHost"
                  << " (max diff = " << maxDiff << ")  PASS\n";
    }

    // BasisFunctions2D via malloc path on OpenCL
    BasisFunctions2D basis2D(basis1D, 1);
    basis2D.allocateDeviceMemory(mgr);

    int Nq2    = 4 * 4;
    int N_base = 3;
    pass &= checkO(basis2D.o_quadraturePoints(), "quad_points", Nq2 * 2);
    pass &= checkO(basis2D.o_quadratureWeights(), "quad_weights", Nq2);
    pass &= checkO(basis2D.o_V2D(), "V2D", Nq2 * N_base);
    pass &= checkO(basis2D.o_dV2D_r(), "dV2D_r", Nq2 * N_base);
    pass &= checkO(basis2D.o_dV2D_s(), "dV2D_s", Nq2 * N_base);

    if (basis2D.o_V2D().ptr<Real>() == basis2D.vandermonde().data()) {
        std::cout << "  FAIL: o_V2D ptr equals host ptr (malloc expected)\n";
        pass = false;
    } else {
        std::cout << "  o_V2D device ptr != host ptr (malloc path)  PASS\n";
    }

    if (pass) {
        std::cout << "  OpenCL backend  PASS\n";
    }
    return pass;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << std::setprecision(4) << std::scientific;
    std::cout << "Device Memory Manager & Basis OCCA Memory Test Suite\n";

    bool ok = true;
    ok &= testMemoryManagerConstruction();
    ok &= testWrapOrMallocUnified();
    ok &= testWrapOrMallocZeroInit();
    ok &= testCopyUnified();
    ok &= testBasis1DDeviceMemory();
    ok &= testBasis2DDeviceMemory();
    ok &= testOpenMPBackend();
    ok &= testOpenCLBackend();

    std::cout << "\n===========================\n";
    std::cout << (ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    std::cout << "===========================\n";

    return ok ? 0 : 1;
}
