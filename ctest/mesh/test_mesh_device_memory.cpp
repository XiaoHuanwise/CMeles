/// @file test_mesh_device_memory.cpp
/// @brief Device-memory tests for MeshGeometry::allocateDeviceMemory
///        (Serial / OpenMP / OpenCL backends), covering the unified vs
///        separate memory space paths and the int-typed
///        DeviceMemoryManager overloads.

#include <cstring>
#include <iostream>

#include <Eigen/Dense>
#include <occa.hpp>

#include "basis/BasisFunctions1D.hpp"
#include "basis/BasisFunctions2D.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/MeshGeometry.hpp"
#include "mesh/StructuredMeshGenerator.hpp"

namespace
{
const Real tol = Real(1e3) * RealEpsilon;

/// Probe whether a backend is available, returning its props if so.
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

/// Read a device array back to host, distinguishing the memory spaces.
static void readResult(DeviceMemoryManager &mem, occa::memory o_data, Real *dst,
                       occa::dim_t entries, const Real *hostAlias)
{
    if (mem.hasSeparateMemorySpace())
    {
        mem.copyToHost(o_data, dst, entries);
    }
    else
    {
        std::memcpy(dst, hostAlias,
                    static_cast<std::size_t>(entries) * sizeof(Real));
    }
}

/// Same for int arrays.
static void readResult(DeviceMemoryManager &mem, occa::memory o_data, int *dst,
                       occa::dim_t entries, const int *hostAlias)
{
    if (mem.hasSeparateMemorySpace())
    {
        mem.copyToHost(o_data, dst, entries);
    }
    else
    {
        std::memcpy(dst, hostAlias,
                    static_cast<std::size_t>(entries) * sizeof(int));
    }
}

/// Check a handle: initialised and of the expected size.
/// occa::memory::size() returns the number of elements.
bool checkHandle(occa::memory o, occa::dim_t expectedElements, const char *what)
{
    if (!o.isInitialized() || o.size() == 0)
    {
        std::cout << "  FAIL " << what << ": uninitialised\n";
        return false;
    }
    if (o.size() != expectedElements)
    {
        std::cout << "  FAIL " << what << ": size=" << o.size() << " (expect "
                  << expectedElements << " elements)\n";
        return false;
    }
    return true;
}

/// Build a small mixed mesh: 2x2 quads + one triangle-split row, so both
/// element kinds and interior/boundary faces appear.
struct TestFixture
{
    Mesh mesh;
    BasisFunctions2D basis;
    MeshGeometry geo;

    TestFixture()
        : mesh(
              StructuredMeshGenerator::generate(StructuredMeshGenerator::Params{
                  2, 2, Real(0), Real(0), Real(1), Real(1)})),
          basis(makeBasis()), geo(mesh, basis)
    {
    }

    static BasisFunctions2D makeBasis()
    {
        static BasisFunctions1D basis1D(2, 3);
        return BasisFunctions2D(basis1D, 2);
    }
};

bool runOnDevice(occa::json props, int &passed, int &skipped, int &failed)
{
    bool ok = true;
    if (props.isNull())
    {
        ++skipped;
        return ok;
    }

    occa::device device(props);
    DeviceMemoryManager mem(device);
    TestFixture fx;
    fx.geo.allocateDeviceMemory(mem);
    device.finish();

    const int N_elem = fx.geo.numElements();
    const int N_face = fx.geo.numFaces();
    const int Nq2    = fx.geo.numPointsPerElement();
    const int Nb     = fx.geo.numBases();

    // ---- All handles initialised with correct sizes ----
    ok &= checkHandle(fx.geo.o_vertices(), N_elem * 8, "o_vertices");
    ok &= checkHandle(fx.geo.o_absJ(), N_elem * Nq2, "o_absJ");
    ok &= checkHandle(fx.geo.o_Jinv11(), N_elem * Nq2, "o_Jinv11");
    ok &= checkHandle(fx.geo.o_Jinv12(), N_elem * Nq2, "o_Jinv12");
    ok &= checkHandle(fx.geo.o_Jinv21(), N_elem * Nq2, "o_Jinv21");
    ok &= checkHandle(fx.geo.o_Jinv22(), N_elem * Nq2, "o_Jinv22");
    ok &= checkHandle(fx.geo.o_lambdaWJ(), N_elem * Nq2, "o_lambdaWJ");
    ok &= checkHandle(fx.geo.o_minv(), N_elem * Nb * Nb, "o_minv");
    ok &= checkHandle(fx.geo.o_faceNormals(), N_face * 2, "o_faceNormals");
    ok &= checkHandle(fx.geo.o_faceJac(), N_face, "o_faceJac");
    ok &= checkHandle(fx.geo.o_faceTypes(), N_face, "o_faceTypes");
    ok &= checkHandle(fx.geo.o_faceKL(), N_face, "o_faceKL");
    ok &= checkHandle(fx.geo.o_faceFL(), N_face, "o_faceFL");
    ok &= checkHandle(fx.geo.o_faceKR(), N_face, "o_faceKR");
    ok &= checkHandle(fx.geo.o_faceFR(), N_face, "o_faceFR");
    ok &= checkHandle(fx.geo.o_elemFaces(), N_elem * 4, "o_elemFaces");

    // ---- Zero-copy check in unified memory space ----
    if (!mem.hasSeparateMemorySpace())
    {
        ok &= fx.geo.o_absJ().ptr<Real>() == fx.geo.absJacobian().data();
        ok &= fx.geo.o_faceKL().ptr<int>() != nullptr;
        ok &= fx.geo.o_lambdaWJ().ptr<Real>() == fx.geo.lambdaWJ().data();
    }
    else
    {
        ok &= fx.geo.o_absJ().ptr<Real>() != fx.geo.absJacobian().data();
    }

    // ---- Data roundtrip: Real array (absJ) ----
    {
        std::vector<Real> host(N_elem * Nq2);
        readResult(mem, fx.geo.o_absJ(), host.data(), N_elem * Nq2,
                   fx.geo.absJacobian().data());
        const Real err =
            (VectorXr::Map(host.data(), host.size()) - fx.geo.absJacobian())
                .template lpNorm<Eigen::Infinity>();
        ok &= err <=
              tol * fx.geo.absJacobian().template lpNorm<Eigen::Infinity>();
    }

    // ---- Data roundtrip: int array (faceKL, exercises int overloads) ----
    {
        // Row-major Eigen columns are non-contiguous; use a copy.
        Eigen::VectorXi kl = fx.mesh.faceElements().col(0);
        std::vector<int> hostKL(N_face);
        readResult(mem, fx.geo.o_faceKL(), hostKL.data(), N_face, kl.data());
        for (int F = 0; F < N_face; ++F)
        {
            ok &= hostKL[F] == fx.mesh.faceElements()(F, 0);
        }
    }

    if (!ok)
    {
        ++failed;
    }
    else
    {
        ++passed;
    }
    return ok;
}
} // namespace

int main()
{
    int passed = 0, skipped = 0, failed = 0;

    // Serial (unified memory space, mandatory).
    std::cout << "[Serial]\n";
    runOnDevice(tryMakeDevice("Serial"), passed, skipped, failed);

    // OpenMP (unified memory space, optional).
    std::cout << "[OpenMP]\n";
    runOnDevice(tryMakeDevice("OpenMP"), passed, skipped, failed);

    // OpenCL (separate memory space, optional: skipped if unavailable).
    std::cout << "[OpenCL]\n";
    {
        occa::json props = tryMakeDevice("OpenCL");
        if (props.isNull())
        {
            std::cout << "  (skip OpenCL: not available)\n";
            ++skipped;
        }
        else
        {
            runOnDevice(props, passed, skipped, failed);
        }
    }

    std::cout << "passed: " << passed << ", skipped: " << skipped
              << ", failed: " << failed << "\n";
    return failed > 0 ? 1 : 0;
}
