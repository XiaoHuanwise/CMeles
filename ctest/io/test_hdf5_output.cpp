/// @file test_hdf5_output.cpp
/// @brief Tests for the io module: HDF5Writer round-trip, [output] config
///        parsing, the strategy A/B writers and the offline converter,
///        plus an end-to-end solver run with output enabled.
///
/// Output is disabled by default, so every case here opts in through an
/// explicit [output] table; the final case also verifies that a default
/// configuration produces no files at all.

#include <highfive/H5File.hpp>
#include <occa.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "basis/BasisFunctions2D.hpp"
#include "config/Config.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "dg/DgField.hpp"
#include "io/CheckpointConvert.hpp"
#include "io/CheckpointWriter.hpp"
#include "io/HDF5Writer.hpp"
#include "io/SolutionWriter.hpp"
#include "io/VizWriter.hpp"
#include "solver/CompressibleFlowSolver.hpp"
#include "solver/ExprInitialCondition.hpp"

namespace fs = std::filesystem;

namespace {
const std::string kScratchDir  = "test_hdf5_scratch";
const std::string kScratchToml = "test_hdf5_scratch.toml";
const std::string kResultsDir  = kScratchDir + "/results";

bool check(bool cond, const char *what) {
    if (!cond) {
        std::cout << "  FAIL " << what << "\n";
    }
    return cond;
}

bool checkInt(int value, int ref, const char *what) {
    if (value != ref) {
        std::cout << "  FAIL " << what << ": value=" << value << " ref=" << ref
                  << "\n";
        return false;
    }
    return true;
}

bool checkReal(Real value, Real ref, const char *what) {
    const Real tol = Real(1e3) * RealEpsilon;
    if (std::abs(value - ref) > tol * std::max(std::abs(ref), Real(1))) {
        std::cout << "  FAIL " << what << ": value=" << value << " ref=" << ref
                  << "\n";
        return false;
    }
    return true;
}

/// @brief Equality of two comparable values with a roundoff-scaled
///        tolerance anchored at magnitude 1 (safe for zeros).
bool checkClose(Real a, Real b, const char *what) {
    const Real tol = Real(1e3) * RealEpsilon *
                     std::max(std::max(std::abs(a), std::abs(b)), Real(1));
    if (std::abs(a - b) > tol) {
        std::cout << "  FAIL " << what << ": a=" << a << " b=" << b << "\n";
        return false;
    }
    return true;
}

bool checkStr(const std::string &value, const std::string &ref,
              const char *what) {
    if (value != ref) {
        std::cout << "  FAIL " << what << ": value='" << value << "' ref='"
                  << ref << "'\n";
        return false;
    }
    return true;
}

/// @brief Write \p content to the scratch TOML file (next to the binary).
std::string writeToml(const std::string &content) {
    std::FILE *f = std::fopen(kScratchToml.c_str(), "w");
    if (!f) {
        throw std::runtime_error("cannot write " + kScratchToml);
    }
    std::fputs(content.c_str(), f);
    std::fclose(f);
    return kScratchToml;
}

template <typename T>
T readAttr(const HighFive::File &file, const std::string &name) {
    return file.getAttribute(name).read<T>();
}

std::vector<Real> readDs(const HighFive::File &file, const std::string &name) {
    // read_raw skips the shape adaptation so both 1-D and 2-D datasets
    // can be read into a flat buffer.
    std::vector<Real> data(file.getDataSet(name).getElementCount());
    file.getDataSet(name).read_raw(data.data());
    return data;
}

// ---------------------------------------------------------------------------
// 1. HDF5Writer round-trip
// ---------------------------------------------------------------------------

static bool testWriterRoundtrip() {
    std::cout << "Test 1: HDF5Writer round-trip\n";
    fs::create_directories(kScratchDir);
    const std::string path = (fs::path(kScratchDir) / "writer.h5").string();
    {
        HDF5Writer writer(path);
        writer.writeAttribute("a_real", Real(1.5));
        writer.writeAttribute("a_int", 7);
        writer.writeAttribute("a_str", std::string("quad"));
        const std::vector<Real> d1 = {Real(1), Real(2), Real(3)};
        // 2 x 4 row-major.
        const std::vector<Real> d2 = {Real(1), Real(2), Real(3), Real(4),
                                      Real(5), Real(6), Real(7), Real(8)};
        writer.writeDataset("d1", d1.data(), d1.size());
        writer.writeDataset2D("d2", d2.data(), 2, 4);
        writer.writeGroupDataset("mesh", "x", d1.data(), d1.size());
    }

    const HighFive::File file(path, HighFive::File::ReadOnly);
    bool ok = true;
    ok &= checkReal(readAttr<Real>(file, "a_real"), Real(1.5), "attr real");
    ok &= checkInt(readAttr<int>(file, "a_int"), 7, "attr int");
    ok &= checkStr(readAttr<std::string>(file, "a_str"), "quad", "attr str");

    const std::vector<Real> r1 = readDs(file, "d1");
    ok &= check(r1.size() == 3, "d1 size");
    ok &= check(r1.size() == 3 && r1[2] == Real(3), "d1 values");

    const auto dims = file.getDataSet("d2").getDimensions();
    ok &= check(dims.size() == 2 && dims[0] == 2 && dims[1] == 4, "d2 dims");
    const std::vector<Real> r2 = readDs(file, "d2");
    ok &= check(r2.size() == 8 && r2[7] == Real(8), "d2 values");

    std::vector<Real> rx;
    file.getGroup("mesh").getDataSet("x").read(rx);
    ok &= check(rx.size() == 3, "mesh/x size");
    return ok;
}

// ---------------------------------------------------------------------------
// 2. [output] config parsing and validation
// ---------------------------------------------------------------------------

static bool testConfigOutput() {
    std::cout << "Test 2: [output] config\n";
    bool ok = true;

    const Config def;
    ok &= check(!def.outputEnable(), "default enable is false");
    ok &= check(def.outputStrategy() == OutputStrategy::Direct,
                "default strategy is direct");
    ok &= check(def.outputFormat() == OutputFormat::Hdf5,
                "default format is hdf5");
    ok &= checkInt(def.outputInterval(), 100, "default interval");
    ok &= checkStr(def.outputDirectory(), "./results", "default directory");

    const Config cfg(writeToml("[output]\nenable = true\nstrategy = "
                               "\"checkpoint\"\nformat = \"all\"\ninterval = "
                               "7\ndirectory = \"./out_xyz\"\n"));
    ok &= check(cfg.outputEnable(), "enable parsed");
    ok &= check(cfg.outputStrategy() == OutputStrategy::Checkpoint,
                "strategy parsed");
    ok &= check(cfg.outputFormat() == OutputFormat::All, "format parsed");
    ok &= checkInt(cfg.outputInterval(), 7, "interval parsed");
    ok &= checkStr(cfg.outputDirectory(), "./out_xyz", "directory parsed");

    bool threw = false;
    try {
        const Config bad(
            writeToml("[output]\nenable = true\nstrategy = \"fancy\"\n"));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= check(threw, "unknown strategy throws");

    threw = false;
    try {
        const Config bad(
            writeToml("[output]\nenable = true\nformat = \"fancy\"\n"));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= check(threw, "unknown format throws");

    threw = false;
    try {
        const Config bad(writeToml("[output]\nenable = true\ninterval = 0\n"));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= check(threw, "interval 0 throws when enabled");

    threw = false;
    try {
        const Config okCfg(
            writeToml("[output]\nenable = false\ninterval = 0\n"));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= check(!threw, "interval 0 accepted when disabled");
    return ok;
}

// ---------------------------------------------------------------------------
// 3. SolutionWriter on a uniform free-stream field (exact values)
// ---------------------------------------------------------------------------

static bool testSolutionWriterUniform() {
    std::cout << "Test 3: SolutionWriter uniform free stream\n";
    // The strategy writers expect the output directory to exist (the
    // production path goes through FieldOutput, which creates it).
    fs::create_directories(kResultsDir);
    const Config cfg(writeToml("[output]\nenable = true\ndirectory = \"" +
                               kResultsDir + "\"\n"));
    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mem(device);
    DgField field(cfg, device, mem);
    field.setup();
    field.applyFreeStreamInitialCondition(field.o_u());
    device.finish();

    SolutionWriter writer(cfg, field);
    writer.write(3, Real(0.25));

    const std::string path =
        (fs::path(kResultsDir) / "test_hdf5_scratch_solution_0000003.h5")
            .string();
    const HighFive::File file(path, HighFive::File::ReadOnly);
    bool ok = true;
    ok &= checkReal(readAttr<Real>(file, "time"), Real(0.25), "time attr");
    ok &= checkInt(readAttr<int>(file, "step"), 3, "step attr");
    ok &= checkInt(readAttr<int>(file, "order"), cfg.polynomialOrder(),
                   "order attr");

    const size_t nTot =
        static_cast<size_t>(field.numElements()) * field.numQuadPoints();
    // Uniform free stream: rho = 1, u = v = 0, p = 1, gamma = 1.4, so
    // E = p / (gamma - 1) = 2.5 everywhere; the L2 projection of a
    // constant is exact at the quadrature points.
    const Real gamma    = cfg.gamma();
    const Real eRef     = cfg.flowP() / (gamma - Real(1));
    const char *names[] = {"rho", "rho_u", "rho_v", "E", "u", "v", "p"};
    const Real refs[]   = {cfg.flowRho(), Real(0),     Real(0),    eRef,
                           cfg.flowU(),   cfg.flowV(), cfg.flowP()};
    for (int i = 0; i < 7; ++i) {
        const std::vector<Real> data = readDs(file, names[i]);
        if (!check(data.size() == nTot, "dataset size")) {
            ok = false;
            continue;
        }
        for (size_t k = 0; k < nTot; ++k) {
            if (!checkClose(data[k], refs[i], names[i])) {
                ok = false;
                break;
            }
        }
    }

    // Coordinates: the default mesh is nx*dx = 8 squared, with the
    // quadrature points strictly inside their elements.
    std::vector<Real> gx, gy;
    file.getGroup("mesh").getDataSet("x").read(gx);
    file.getGroup("mesh").getDataSet("y").read(gy);
    ok &= check(gx.size() == nTot && gy.size() == nTot, "coord sizes");
    const Real xHi = cfg.meshX0() + Real(cfg.meshNx()) * cfg.meshDx();
    const Real yHi = cfg.meshY0() + Real(cfg.meshNy()) * cfg.meshDy();
    bool rangeOk   = true;
    for (size_t k = 0; k < nTot; ++k) {
        rangeOk &= gx[k] > cfg.meshX0() && gx[k] < xHi;
        rangeOk &= gy[k] > cfg.meshY0() && gy[k] < yHi;
    }
    ok &= check(rangeOk, "coords inside the mesh domain");
    return ok;
}

// ---------------------------------------------------------------------------
// 4. CheckpointWriter + convertCheckpoint vs SolutionWriter (expr IC)
// ---------------------------------------------------------------------------

static bool testCheckpointConvertRoundtrip() {
    std::cout << "Test 4: checkpoint -> convert round-trip\n";
    fs::create_directories(kResultsDir);
    // A non-constant initial condition makes the modal->nodal projection
    // comparison meaningful (both output paths must agree to roundoff).
    const Config cfg(
        writeToml("[output]\nenable = true\ndirectory = \"" + kResultsDir +
                  "\"\n\n[initial_condition]\ntype = \"expr\"\n"
                  "rho = \"1 + 0.2*sin(2*pi*x)*cos(2*pi*y)\"\nu = \"0.1\"\n"
                  "v = \"0\"\np = \"1\"\n"));
    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mem(device);
    DgField field(cfg, device, mem);
    field.setup();
    solver_detail::applyExprICFromConfig(field, cfg);
    device.finish();

    // Strategy A reference file.
    SolutionWriter solution(cfg, field);
    solution.write(5, Real(0.5));

    // Strategy B checkpoint + offline conversion.
    CheckpointWriter checkpoint(cfg, field);
    checkpoint.writeMesh();
    checkpoint.write(5, Real(0.5));
    const std::string meshPath =
        (fs::path(kResultsDir) / "test_hdf5_scratch_mesh.h5").string();
    const std::string ckptPath =
        (fs::path(kResultsDir) / "test_hdf5_scratch_checkpoint_0000005.h5")
            .string();
    const std::string convPath =
        (fs::path(kResultsDir) / "converted.h5").string();
    convertCheckpoint(meshPath, ckptPath, convPath);

    const HighFive::File direct(
        (fs::path(kResultsDir) / "test_hdf5_scratch_solution_0000005.h5")
            .string(),
        HighFive::File::ReadOnly);
    const HighFive::File converted(convPath, HighFive::File::ReadOnly);
    bool ok = true;
    ok &= checkClose(readAttr<Real>(direct, "time"),
                     readAttr<Real>(converted, "time"), "time attr match");
    ok &= checkInt(readAttr<int>(converted, "step"), 5, "converted step");
    ok &= checkInt(readAttr<int>(converted, "order"), cfg.polynomialOrder(),
                   "converted order");

    const char *names[] = {"rho", "rho_u", "rho_v", "E", "u", "v", "p"};
    for (const char *name : names) {
        const std::vector<Real> a = readDs(direct, name);
        const std::vector<Real> b = readDs(converted, name);
        if (!check(a.size() == b.size(), "sizes match") || a.empty()) {
            ok = false;
            continue;
        }
        bool same = true;
        for (size_t k = 0; k < a.size(); ++k) {
            same &= checkClose(a[k], b[k], name);
        }
        ok &= check(same, "per-point values match");
    }

    // mesh.h5 metadata.
    const HighFive::File mesh(meshPath, HighFive::File::ReadOnly);
    ok &= checkInt(readAttr<int>(mesh, "dim"), 2, "mesh dim");
    ok &= checkInt(readAttr<int>(mesh, "N_elem"), field.numElements(),
                   "mesh N_elem");
    ok &= checkInt(readAttr<int>(mesh, "N_q"), field.numPoints1D(), "mesh N_q");
    ok &= checkInt(readAttr<int>(mesh, "N_modes"), field.numModes(),
                   "mesh N_modes");
    ok &= check(readDs(mesh, "x").size() ==
                    static_cast<size_t>(field.numElements()) *
                        field.numQuadPoints(),
                "mesh x size");
    return ok;
}

// ---------------------------------------------------------------------------
// 5. End-to-end solver run: files on interval, none by default
// ---------------------------------------------------------------------------

static bool testSolverEndToEnd() {
    std::cout << "Test 5: solver end-to-end\n";
    // 3 fixed-dt Euler steps on a 4x4 mesh, output every step.
    const Config cfg(writeToml(
        "[mesh]\nnx = 4\nny = 4\ndx = 0.25\ndy = 0.25\n\n"
        "[time_marching]\nmethod = \"euler\"\ndt = 1e-3\nt_final = 3e-3\n\n"
        "[output]\nenable = true\ninterval = 1\ndirectory = \"" +
        kScratchDir + "/e2e\"\n"));
    bool ok = check(runCompressibleFlowSolver(cfg) == 0, "solver run");

    // Files: initial (0) + every step; the final step's interval output
    // already covers the final state, so exactly 4 files exist.
    const std::string outDir = kScratchDir + "/e2e";
    for (long step = 0; step <= 3; ++step) {
        char name[64];
        std::snprintf(name, sizeof(name), "test_hdf5_scratch_solution_%07ld.h5",
                      step);
        const std::string path = (fs::path(outDir) / name).string();
        ok &= check(fs::exists(path), name);
        if (!fs::exists(path)) {
            continue;
        }
        const HighFive::File file(path, HighFive::File::ReadOnly);
        ok &= checkClose(readAttr<Real>(file, "time"), Real(1e-3) * Real(step),
                         "time attr");
        ok &= checkInt(readAttr<int>(file, "step"), static_cast<int>(step),
                       "step attr");
        ok &= check(readDs(file, "rho").size() == static_cast<size_t>(16) * 16,
                    "rho dataset size");
    }

    // A default configuration (no [output] table) must not create the
    // default ./results directory.
    fs::remove_all("./results");
    const Config quiet(writeToml("[mesh]\nnx = 4\nny = 4\ndx = 0.25\ndy = "
                                 "0.25\n\n[time_marching]\nmethod = "
                                 "\"euler\"\ndt = 5e-4\nt_final = 1e-3\n"));
    ok &= check(runCompressibleFlowSolver(quiet) == 0, "quiet solver run");
    ok &=
        check(!fs::exists("./results"), "no ./results without [output] enable");
    return ok;
}

// ---------------------------------------------------------------------------
// 6. Visualization sub-grid connectivity (per-element-type adaptive)
// ---------------------------------------------------------------------------

static bool testVizConnectivity() {
    std::cout << "Test 6: viz connectivity (adaptive)\n";
    bool ok = true;

    // All-quad: N_q = 3 tensor grid -> 2x2 = 4 quad sub-cells/element.
    // Single-type mesh: plain ids (4 per cell), uniform QUADRILATERAL.
    {
        const viz::VizTopology t = viz::buildVizConnectivity(2, 3, {0, 0});
        ok &= checkInt(static_cast<int>(t.nNodes), 2 * 9, "quad nNodes");
        ok &= checkInt(static_cast<int>(t.nCells), 2 * 4, "quad nCells");
        ok &= check(!t.mixed && t.cellType == 5, "quad uniform type");
        ok &= checkInt(static_cast<int>(t.connect.size()), 2 * 4 * 4,
                       "quad connect size");
        // First sub-cell of element 0: corners (0,0),(1,0),(1,1),(0,1) of
        // the local Gauss grid (q = j*Nq + i) -> ids 0,1,4,3.
        const int expect[4] = {0, 1, 4, 3};
        bool first          = true;
        for (int i = 0; i < 4; ++i) {
            first &= t.connect[static_cast<size_t>(i)] == expect[i];
        }
        ok &= check(first, "first quad entry");
    }

    // Extended (boundary-inclusive) grid: n = 4 nodes per direction
    // (Nq = 2 Gauss points + the two endpoints) -> 3x3 = 9 quad
    // sub-cells/element; the first sub-cell spans (0,0)-(1,0)-(1,1)-(0,1)
    // of the local grid with stride 4 -> ids 0,1,5,4.
    {
        const viz::VizTopology t = viz::buildVizConnectivity(1, 4, {0});
        ok &= checkInt(static_cast<int>(t.nNodes), 16, "ext nNodes");
        ok &= checkInt(static_cast<int>(t.nCells), 9, "ext nCells");
        const int expect[4] = {0, 1, 5, 4};
        bool first          = true;
        for (int i = 0; i < 4; ++i) {
            first &= t.connect[static_cast<size_t>(i)] == expect[i];
        }
        ok &= check(first, "ext first quad entry");
    }

    // All-triangle: each tensor sub-cell splits into 2 triangles; plain
    // ids (3 per cell), uniform TRIANGLE.
    {
        const viz::VizTopology t = viz::buildVizConnectivity(1, 3, {1});
        ok &= checkInt(static_cast<int>(t.nCells), 2 * 4, "tri nCells");
        ok &= check(!t.mixed && t.cellType == 4, "tri uniform type");
        ok &= checkInt(static_cast<int>(t.connect.size()), 8 * 3,
                       "tri connect size");
    }

    // Mixed: quad element + triangle element side by side — the only
    // case using the [type, ids...] XDMF Mixed encoding.
    {
        const viz::VizTopology t = viz::buildVizConnectivity(2, 3, {0, 1});
        ok &= checkInt(static_cast<int>(t.nCells), 4 + 8, "mixed nCells");
        ok &= check(t.mixed, "mixed flag set");
        size_t i  = 0;
        int nQuad = 0, nTri = 0;
        while (i < t.connect.size()) {
            if (t.connect[i] == 5) {
                ++nQuad;
                i += 5;
            } else {
                ++nTri;
                i += 4;
            }
        }
        ok &= checkInt(nQuad, 4, "mixed quad count");
        ok &= checkInt(nTri, 8, "mixed triangle count");
    }

    bool threw = false;
    try {
        const viz::VizTopology t = viz::buildVizConnectivity(2, 3, {0});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= check(threw, "size mismatch throws");
    return ok;
}

// ---------------------------------------------------------------------------
// 7. XDMF + Tecplot PLT output of SolutionWriter (format = "all")
// ---------------------------------------------------------------------------

namespace {
/// Minimal reader of the classic PLT written by viz::writeTecplotPlt
/// (same layout, read back).
struct PltReader {
    std::vector<char> buf;
    size_t pos = 0;

    explicit PltReader(const std::string &path) {
        std::FILE *f = std::fopen(path.c_str(), "rb");
        if (!f) {
            throw std::runtime_error("cannot open " + path);
        }
        char c;
        while (std::fread(&c, 1, 1, f) == 1) {
            buf.push_back(c);
        }
        std::fclose(f);
    }
    void need(size_t n) const {
        if (pos + n > buf.size()) {
            throw std::runtime_error("plt truncated");
        }
    }
    int32_t int32() {
        need(4);
        int32_t v;
        std::memcpy(&v, buf.data() + pos, 4);
        pos += 4;
        return v;
    }
    float float32() {
        need(4);
        float v;
        std::memcpy(&v, buf.data() + pos, 4);
        pos += 4;
        return v;
    }
    Real real64() {
        need(sizeof(Real));
        Real v;
        std::memcpy(&v, buf.data() + pos, sizeof(Real));
        pos += sizeof(Real);
        return v;
    }
    std::string str() {
        std::string s;
        for (;;) {
            const int32_t c = int32();
            if (c == 0) {
                break;
            }
            s.push_back(static_cast<char>(c));
        }
        return s;
    }
    std::vector<Real> block(size_t n) {
        std::vector<Real> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = real64();
        }
        return v;
    }
};
} // namespace

static bool testVizOutputFormats() {
    std::cout << "Test 7: xdmf + tecplot output (format = all)\n";
    fs::create_directories(kResultsDir);
    const Config cfg(writeToml(
        "[output]\nenable = true\nformat = \"all\"\ndirectory = \"" +
        kResultsDir +
        "\"\n\n[initial_condition]\ntype = \"expr\"\n"
        "rho = \"1 + 0.2*sin(2*pi*x)*cos(2*pi*y)\"\nu = \"0.1\"\nv = \"0\"\n"
        "p = \"1\"\n"));
    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mem(device);
    DgField field(cfg, device, mem);
    field.setup();
    solver_detail::applyExprICFromConfig(field, cfg);
    device.finish();

    SolutionWriter writer(cfg, field);
    writer.write(9, Real(0.75));

    // format = "all": every dataset lives on the boundary-inclusive
    // extended grid, (nq + 2)^2 nodes per element.
    const int nq         = field.numPoints1D();
    const int nExt1      = nq + 2;
    const size_t nVizTot = static_cast<size_t>(field.numElements()) *
                           static_cast<size_t>(nExt1) *
                           static_cast<size_t>(nExt1);
    std::vector<int> isTriangle(static_cast<size_t>(field.numElements()), 0);
    const viz::VizTopology topo =
        viz::buildVizConnectivity(field.numElements(), nExt1, isTriangle);

    bool ok = true;

    // --- HDF5 side: connectivity + attributes.
    const HighFive::File file(
        (fs::path(kResultsDir) / "test_hdf5_scratch_solution_0000009.h5")
            .string(),
        HighFive::File::ReadOnly);
    std::vector<int> connect;
    file.getGroup("mesh").getDataSet("connect").read(connect);
    ok &= check(connect == topo.connect, "mesh/connect matches");
    // Interleaved [N][3] visualization coordinates (XYZ geometry).
    const auto xyzDims =
        file.getGroup("mesh").getDataSet("xyz").getDimensions();
    ok &= check(xyzDims.size() == 2 && xyzDims[0] == nVizTot && xyzDims[1] == 3,
                "mesh/xyz dims");
    ok &= checkInt(readAttr<int>(file, "viz_cells"),
                   static_cast<int>(topo.nCells), "viz_cells attr");
    ok &= checkInt(readAttr<int>(file, "viz_nodes"),
                   static_cast<int>(topo.nNodes), "viz_nodes attr");

    // --- XDMF sidecar: well-formed references.
    const std::string xmfPath =
        (fs::path(kResultsDir) / "test_hdf5_scratch_solution_0000009.xmf")
            .string();
    ok &= check(fs::exists(xmfPath), "xmf exists");
    std::string xmf;
    {
        std::FILE *f = std::fopen(xmfPath.c_str(), "r");
        char cbuf[4096];
        size_t n;
        while ((n = std::fread(cbuf, 1, sizeof(cbuf), f)) > 0) {
            xmf.append(cbuf, n);
        }
        std::fclose(f);
    }
    ok &= check(xmf.find("TopologyType=\"QUADRILATERAL\"") != std::string::npos,
                "xmf uniform quad topology");
    ok &= check(xmf.find("Mixed") == std::string::npos,
                "xmf no mixed topology for single-type mesh");
    ok &= check(xmf.find("DOCTYPE") == std::string::npos, "xmf no doctype");
    // XYZ geometry through the interleaved mesh/xyz dataset (the XY form
    // breaks ParaView 5.13's Xdmf3 readers - half the points, then
    // out-of-range connectivity crashes Apply/Render).
    ok &= check(xmf.find("GeometryType=\"XYZ\"") != std::string::npos,
                "xmf xyz geometry");
    ok &= check(xmf.find(".h5:/mesh/xyz") != std::string::npos,
                "xmf references mesh/xyz");
    ok &=
        check(xmf.find("test_hdf5_scratch_solution_0000009.h5:/mesh/connect") !=
                  std::string::npos,
              "xmf references connect");
    ok &= check(xmf.find("test_hdf5_scratch_solution_0000009.h5:/mesh/x") !=
                    std::string::npos,
                "xmf references x");
    ok &= check(xmf.find("test_hdf5_scratch_solution_0000009.h5:/rho") !=
                    std::string::npos,
                "xmf references rho");
    ok &= check(xmf.find("<Time Value=\"0.75\"/>") != std::string::npos,
                "xmf time");
    ok &= check(xmf.find("NumberOfElements=\"" + std::to_string(topo.nCells) +
                         "\"") != std::string::npos,
                "xmf cell count");

    // --- Temporal-collection index (incrementally maintained series). ---
    const std::string seriesPath =
        (fs::path(kResultsDir) / "test_hdf5_scratch_series.xmf").string();
    ok &= check(fs::exists(seriesPath), "series exists");
    {
        std::string series;
        std::FILE *s = std::fopen(seriesPath.c_str(), "r");
        char sbuf[8192];
        size_t n;
        while ((n = std::fread(sbuf, 1, sizeof(sbuf), s)) > 0) {
            series.append(sbuf, n);
        }
        std::fclose(s);
        ok &= check(series.find("CollectionType=\"Temporal\"") !=
                        std::string::npos,
                    "series temporal collection");
        ok &= check(series.find("test_hdf5_scratch_solution_0000009.h5:/rho") !=
                        std::string::npos,
                    "series references step data");
        ok &= check(series.find("<Time Value=\"0.75\"/>") != std::string::npos,
                    "series time value");
        // One write call so far -> exactly one uniform step grid.
        size_t nGrids = 0;
        for (size_t p = series.find("GridType=\"Uniform\"");
             p != std::string::npos;
             p = series.find("GridType=\"Uniform\"", p + 1)) {
            ++nGrids;
        }
        ok &= checkInt(static_cast<int>(nGrids), 1, "series step count");
    }

    // --- PLT side: parse the file back and cross-check with the HDF5.
    const std::string pltPath =
        (fs::path(kResultsDir) / "test_hdf5_scratch_solution_0000009.plt")
            .string();
    ok &= check(fs::exists(pltPath), "plt exists");
    try {
        PltReader plt(pltPath);
        ok &= check(plt.buf.size() >= 8 &&
                        std::string(plt.buf.data(), 8) == "#!TDV112",
                    "plt magic");
        plt.pos = 8;
        ok &= checkInt(plt.int32(), 1, "plt byte order");
        plt.str(); // Title.
        ok &= checkInt(plt.int32(), 9, "plt nVars");
        for (int v = 0; v < 9; ++v) {
            plt.str();
        }
        ok &= check(std::abs(plt.float32() - 299.0f) < 0.5f, "plt zone marker");
        plt.str(); // Zone name.
        ok &= checkInt(plt.int32(), -1, "plt parent zone");
        ok &= checkInt(plt.int32(), 1, "plt strand id");
        ok &= checkClose(plt.real64(), Real(0.75), "plt solution time");
        ok &= checkInt(plt.int32(), -1, "plt zone color");
        ok &= checkInt(plt.int32(), 3, "plt zone type FEQUADRILATERAL");
        // V112 has no data-packing field (readers assume block packing).
        ok &= checkInt(plt.int32(), 0, "plt var locations");
        ok &= checkInt(plt.int32(), 0, "plt face neighbours");
        ok &= checkInt(plt.int32(), 0, "plt user-defined neighbours");
        ok &= checkInt(plt.int32(), static_cast<int>(topo.nNodes),
                       "plt node count");
        ok &= checkInt(plt.int32(), static_cast<int>(topo.nCells),
                       "plt cell count");
        plt.int32();
        plt.int32();
        plt.int32(); // Cell dims (unused).
        ok &= checkInt(plt.int32(), 0, "plt aux end");
        ok &= check(std::abs(plt.float32() - 357.0f) < 0.5f, "plt EOH marker");
        ok &= check(std::abs(plt.float32() - 299.0f) < 0.5f, "plt data marker");
        for (int v = 0; v < 9; ++v) {
            ok &= checkInt(plt.int32(), (sizeof(Real) == 8) ? 2 : 1,
                           "plt var format");
        }
        ok &= checkInt(plt.int32(), 0, "plt passive none");
        ok &= checkInt(plt.int32(), 0, "plt sharing none");
        ok &= checkInt(plt.int32(), -1, "plt connectivity sharing");
        // Version >= 103 data section: per-variable min/max pairs. The
        // rho pair (variable 2) must match the dataset extrema.
        Real rhoMin = 0, rhoMax = 0;
        for (int v = 0; v < 9; ++v) {
            const Real lo = plt.real64();
            const Real hi = plt.real64();
            if (v == 2) {
                rhoMin = lo;
                rhoMax = hi;
            }
        }
        {
            const std::vector<Real> h5r = readDs(file, "rho");
            Real lo = h5r[0], hi = h5r[0];
            for (size_t k = 1; k < h5r.size(); ++k) {
                lo = h5r[k] < lo ? h5r[k] : lo;
                hi = h5r[k] > hi ? h5r[k] : hi;
            }
            ok &= checkClose(rhoMin, lo, "plt rho min/max pair") &&
                  checkClose(rhoMax, hi, "plt rho min/max pair");
        }
        // Nine data blocks; cross-check rho (block 2) with the HDF5 file.
        std::vector<Real> rho;
        for (int v = 0; v < 9; ++v) {
            const std::vector<Real> blk = plt.block(nVizTot);
            if (v == 2) {
                rho = blk;
            }
        }
        const std::vector<Real> h5rho = readDs(file, "rho");
        bool same                     = rho.size() == nVizTot;
        for (size_t k = 0; same && k < nVizTot; ++k) {
            same &= checkClose(rho[k], h5rho[k], "plt rho matches h5");
        }
        ok &= check(same, "plt rho block equals h5 dataset");
        // Connectivity: 1-based quads, matches the sub-grid. (One read
        // per id — a short-circuit comparison would consume two.)
        bool connOk = true;
        for (long long c = 0; c < topo.nCells && connOk; ++c) {
            for (int j = 0; j < 4; ++j) {
                const int32_t id = plt.int32();
                connOk &= id >= 1 && id <= topo.nNodes;
            }
        }
        ok &= check(connOk, "plt connectivity ids in range");
        ok &= check(plt.pos == plt.buf.size(), "plt fully consumed");
    } catch (const std::exception &e) {
        ok &= check(false, e.what());
    }

    // --- Superset invariant: the extended grid contains the quadrature
    //     values verbatim at tensor indices (i, j) in [1, nq]^2 ---
    {
        const int nElem  = field.numElements();
        const int nModes = field.numModes();
        std::vector<Real> modal(static_cast<size_t>(nElem) * 4 *
                                static_cast<size_t>(nModes));
        field.o_u().copyTo(modal.data(),
                           static_cast<occa::dim_t>(modal.size()));
        const MatrixXr &V             = field.basis().vandermonde();
        const std::vector<Real> rhoDs = readDs(file, "rho");
        bool same                     = rhoDs.size() == nVizTot;
        for (int e = 0; same && e < nElem; ++e) {
            const Eigen::Map<const VectorXr> uhat(
                modal.data() +
                    static_cast<size_t>(e) * 4 * static_cast<size_t>(nModes),
                nModes);
            const VectorXr gauss = V * uhat;
            for (int j = 0; same && j < nq; ++j) {
                for (int i = 0; same && i < nq; ++i) {
                    const size_t extIdx =
                        static_cast<size_t>(e) *
                            static_cast<size_t>(nExt1 * nExt1) +
                        static_cast<size_t>((j + 1) * nExt1 + (i + 1));
                    const size_t q = static_cast<size_t>(j * nq + i);
                    same &= checkClose(rhoDs[extIdx], gauss[q], "superset rho");
                }
            }
        }
        ok &= check(same, "extended rho superset of quadrature values");
    }

    // --- The extended grid closes the outer boundary: the coordinates
    //     reach the domain bounds exactly (no interior-only margin). ---
    {
        std::vector<Real> gx, gy;
        file.getGroup("mesh").getDataSet("x").read(gx);
        file.getGroup("mesh").getDataSet("y").read(gy);
        Real xmin = gx[0], xmax = gx[0], ymin = gy[0], ymax = gy[0];
        for (size_t k = 1; k < gx.size(); ++k) {
            xmin = std::min(xmin, gx[k]);
            xmax = std::max(xmax, gx[k]);
            ymin = std::min(ymin, gy[k]);
            ymax = std::max(ymax, gy[k]);
        }
        ok &= checkReal(xmin, cfg.meshX0(), "x reaches domain min");
        ok &= checkReal(xmax, cfg.meshX0() + Real(cfg.meshNx()) * cfg.meshDx(),
                        "x reaches domain max");
        ok &= checkReal(ymin, cfg.meshY0(), "y reaches domain min");
        ok &= checkReal(ymax, cfg.meshY0() + Real(cfg.meshNy()) * cfg.meshDy(),
                        "y reaches domain max");
    }

    // --- Pure-triangle mesh: native FETRIANGLE zone, 3 ids per cell. ---
    try {
        // Two triangle elements on the extended grid of Nq = 2: 4 nodes
        // per direction -> 16 nodes and 2*9 = 18 sub-cells (36 triangles)
        // per... per element: 9 sub-cells * 2 triangles.
        const viz::VizTopology topo = viz::buildVizConnectivity(2, 4, {1, 1});
        ok &= check(!topo.mixed && topo.cellType == 4, "tri topo uniform");
        ok &= checkInt(static_cast<int>(topo.nNodes), 2 * 16, "tri nNodes");
        ok &= checkInt(static_cast<int>(topo.nCells), 2 * 18, "tri nCells");
        std::vector<Real> x(static_cast<size_t>(topo.nNodes));
        std::vector<Real> y(static_cast<size_t>(topo.nNodes));
        for (size_t k = 0; k < x.size(); ++k) {
            x[k] = Real(k % 4) * Real(0.5);
            y[k] = Real(k / 4) * Real(0.5);
        }
        std::vector<Real> rho(topo.nNodes, Real(1));
        std::vector<Real> ru(topo.nNodes, Real(0));
        std::vector<Real> E(topo.nNodes, Real(2.5));
        const viz::Fields f{x.data(),  y.data(),  rho.data(),
                            ru.data(), ru.data(), E.data(),
                            ru.data(), ru.data(), rho.data()};
        const std::string triPlt = (fs::path(kResultsDir) / "tri.plt").string();
        viz::writeTecplotPlt(triPlt, f, topo, Real(0));
        PltReader plt(triPlt);
        ok &= check(std::string(plt.buf.data(), 8) == "#!TDV112",
                    "tri plt magic");
        plt.pos = 8;
        plt.int32(); // Byte order.
        plt.str();   // Title.
        plt.int32(); // nVars.
        for (int v = 0; v < 9; ++v) {
            plt.str(); // Variable names.
        }
        ok &= check(std::abs(plt.float32() - 299.0f) < 0.5f,
                    "tri plt zone marker");
        plt.str();   // Zone name.
        plt.int32(); // Parent zone.
        ok &= checkInt(plt.int32(), 1, "tri plt strand id");
        ok &= checkClose(plt.real64(), Real(0), "tri plt solution time");
        plt.int32(); // Zone color.
        ok &= checkInt(plt.int32(), 2, "tri plt zone type FETRIANGLE");
        plt.int32(); // Var locations.
        plt.int32(); // Face neighbours.
        plt.int32(); // User-defined neighbours.
        ok &= checkInt(plt.int32(), static_cast<int>(topo.nNodes),
                       "tri plt node count");
        ok &= checkInt(plt.int32(), static_cast<int>(topo.nCells),
                       "tri plt cell count");
        plt.int32();
        plt.int32();
        plt.int32();   // Cell dims.
        plt.int32();   // Aux end.
        plt.float32(); // EOH.
        plt.float32(); // Data marker.
        for (int v = 0; v < 9; ++v) {
            plt.int32(); // Format codes.
        }
        plt.int32(); // Passive variables.
        plt.int32(); // Sharing.
        plt.int32(); // Connectivity sharing.
        // Min/max pairs: rho (variable 2) is constant 1 on this mesh.
        Real triRhoMin = -1, triRhoMax = -1;
        for (int v = 0; v < 9; ++v) {
            const Real lo = plt.real64();
            const Real hi = plt.real64();
            if (v == 2) {
                triRhoMin = lo;
                triRhoMax = hi;
            }
        }
        ok &= checkClose(triRhoMin, Real(1), "tri plt rho min") &&
              checkClose(triRhoMax, Real(1), "tri plt rho max");
        for (int v = 0; v < 9; ++v) {
            plt.block(static_cast<size_t>(topo.nNodes)); // Data blocks.
        }
        // Element 0 sub-cell (i0,j0) of the 4x4 extended grid: corners
        // n1=(0,0), n2=(1,0), n3=(1,1), n4=(0,1) -> ids 0,1,5,4; the
        // diagonal split gives tri1 (0,1,5), tri2 (0,5,4), 1-based.
        ok &= checkInt(plt.int32(), 1, "tri conn 1");
        ok &= checkInt(plt.int32(), 2, "tri conn 2");
        ok &= checkInt(plt.int32(), 6, "tri conn 3");
        ok &= checkInt(plt.int32(), 1, "tri conn 4");
        ok &= checkInt(plt.int32(), 6, "tri conn 5");
        ok &= checkInt(plt.int32(), 5, "tri conn 6");
        // Sub-cell (i1,j0): corners 1,2,6,5 -> (2,3,7), (2,7,6).
        ok &= checkInt(plt.int32(), 2, "tri conn 7");
        ok &= checkInt(plt.int32(), 3, "tri conn 8");
        ok &= checkInt(plt.int32(), 7, "tri conn 9");
        ok &= checkInt(plt.int32(), 2, "tri conn 10");
        ok &= checkInt(plt.int32(), 7, "tri conn 11");
        ok &= checkInt(plt.int32(), 6, "tri conn 12");
        // Skip to element 1 (18 triangle cells per element; 4 already
        // consumed): its first sub-cell is (16,17,21), (16,21,20),
        // 1-based.
        for (int c = 4; c < 18; ++c) {
            for (int j = 0; j < 3; ++j) {
                plt.int32();
            }
        }
        ok &= checkInt(plt.int32(), 17, "tri conn e2 1");
        ok &= checkInt(plt.int32(), 18, "tri conn e2 2");
        ok &= checkInt(plt.int32(), 22, "tri conn e2 3");
        ok &= checkInt(plt.int32(), 17, "tri conn e2 4");
        ok &= checkInt(plt.int32(), 22, "tri conn e2 5");
        ok &= checkInt(plt.int32(), 21, "tri conn e2 6");
        // Remaining cells: ids in range (one read per id).
        bool connOk = true;
        for (long long c = 20; c < topo.nCells && connOk; ++c) {
            for (int j = 0; j < 3; ++j) {
                const int32_t id = plt.int32();
                connOk &= id >= 1 && id <= topo.nNodes;
            }
        }
        ok &= check(connOk, "tri remaining ids in range");
        ok &= check(plt.pos == plt.buf.size(), "tri plt fully consumed");
    } catch (const std::exception &e) {
        ok &= check(false, e.what());
    }

    // --- [output] series = false suppresses the index file. ---
    try {
        const std::string noSeriesDir = kResultsDir + "/noseries";
        fs::create_directories(noSeriesDir);
        const Config cfg2(
            writeToml("[output]\nenable = true\nformat = \"xdmf\"\nseries = "
                      "false\ndirectory = \"" +
                      noSeriesDir +
                      "\"\n\n[initial_condition]\ntype = \"expr\"\n"
                      "rho = \"1\"\nu = \"0\"\nv = \"0\"\np = \"1\"\n"));
        occa::device device2({{"mode", "Serial"}});
        DeviceMemoryManager mem2(device2);
        DgField field2(cfg2, device2, mem2);
        field2.setup();
        solver_detail::applyExprICFromConfig(field2, cfg2);
        device2.finish();
        SolutionWriter writer2(cfg2, field2);
        writer2.write(1, Real(0.1));
        ok &= check(
            fs::exists(noSeriesDir + "/test_hdf5_scratch_solution_0000001.xmf"),
            "noseries xmf written");
        ok &= check(!fs::exists(noSeriesDir + "/test_hdf5_scratch_series.xmf"),
                    "series suppressed");
    } catch (const std::exception &e) {
        ok &= check(false, e.what());
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 8. Checkpoint -> convert visualization dual path + format inheritance
// ---------------------------------------------------------------------------

static bool testConvertVizDualPath() {
    std::cout << "Test 8: checkpoint -> convert viz dual path\n";
    fs::create_directories(kResultsDir);
    const Config cfg(writeToml(
        "[output]\nenable = true\nstrategy = \"checkpoint\"\nformat = "
        "\"xdmf\"\ndirectory = \"" +
        kResultsDir +
        "\"\n\n[initial_condition]\ntype = \"expr\"\n"
        "rho = \"1 + 0.2*sin(2*pi*x)*cos(2*pi*y)\"\nu = \"0.1\"\nv = \"0\"\n"
        "p = \"1\"\n"));
    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mem(device);
    DgField field(cfg, device, mem);
    field.setup();
    solver_detail::applyExprICFromConfig(field, cfg);
    device.finish();

    // Direct-strategy reference with the same format.
    const Config directCfg(writeToml(
        "[output]\nenable = true\nformat = \"xdmf\"\ndirectory = \"" +
        kResultsDir +
        "\"\n\n[initial_condition]\ntype = \"expr\"\n"
        "rho = \"1 + 0.2*sin(2*pi*x)*cos(2*pi*y)\"\nu = \"0.1\"\nv = \"0\"\n"
        "p = \"1\"\n"));
    SolutionWriter direct(directCfg, field);
    direct.write(2, Real(0.5));

    CheckpointWriter checkpoint(cfg, field);
    checkpoint.writeMesh();
    checkpoint.write(2, Real(0.5));

    const std::string meshPath =
        (fs::path(kResultsDir) / "test_hdf5_scratch_mesh.h5").string();
    const std::string ckptPath =
        (fs::path(kResultsDir) / "test_hdf5_scratch_checkpoint_0000002.h5")
            .string();
    const std::string convH5 =
        (fs::path(kResultsDir) / "converted.h5").string();

    bool ok = true;

    // Format inheritance: mesh.h5 recorded xdmf; old-style files fall back.
    ok &= check(convertReadMeshFormat(meshPath) == OutputFormat::Xdmf,
                "mesh.h5 format inherited");
    convertCheckpoint(meshPath, ckptPath, convH5,
                      convertReadMeshFormat(meshPath));

    const std::string convXmf =
        (fs::path(kResultsDir) / "converted.xmf").string();
    ok &= check(fs::exists(convH5) && fs::exists(convXmf),
                "converted h5 + xmf exist");

    // Connectivity of both paths is identical (combinatorial rebuild).
    std::vector<int> cDirect, cConv;
    {
        const HighFive::File a(
            (fs::path(kResultsDir) / "test_hdf5_scratch_solution_0000002.h5")
                .string(),
            HighFive::File::ReadOnly);
        const HighFive::File b(convH5, HighFive::File::ReadOnly);
        a.getGroup("mesh").getDataSet("connect").read(cDirect);
        b.getGroup("mesh").getDataSet("connect").read(cConv);
    }
    ok &= check(cDirect == cConv, "connect identical across paths");

    // Values: both paths evaluate the same modal state on the same
    // boundary-inclusive extended grid; the converter recovers the node
    // coordinates through the exact bilinear fit (round-off only).
    {
        const HighFive::File a(
            (fs::path(kResultsDir) / "test_hdf5_scratch_solution_0000002.h5")
                .string(),
            HighFive::File::ReadOnly);
        const HighFive::File b(convH5, HighFive::File::ReadOnly);
        const std::vector<Real> ra = readDs(a, "rho");
        const std::vector<Real> rb = readDs(b, "rho");
        const size_t nExt = static_cast<size_t>(field.numElements()) *
                            static_cast<size_t>(field.numPoints1D() + 2) *
                            static_cast<size_t>(field.numPoints1D() + 2);
        ok &= check(ra.size() == nExt && rb.size() == nExt,
                    "dual-path extended node count");
        bool same = !ra.empty();
        for (size_t k = 0; same && k < ra.size(); ++k) {
            same &= checkClose(ra[k], rb[k], "dual-path rho");
        }
        ok &= check(same, "dual-path rho values match");
    }

    // The converted sidecar references its own file.
    std::string xmf;
    {
        std::FILE *f = std::fopen(convXmf.c_str(), "r");
        char cbuf[4096];
        size_t n;
        while ((n = std::fread(cbuf, 1, sizeof(cbuf), f)) > 0) {
            xmf.append(cbuf, n);
        }
        std::fclose(f);
    }
    ok &= check(xmf.find("converted.h5:/mesh/connect") != std::string::npos,
                "converted xmf references its own h5");

    // Old-style mesh.h5 (no format attribute) falls back to plain hdf5:
    // strip the attribute by rewriting a minimal mesh.h5.
    const std::string oldMesh =
        (fs::path(kResultsDir) / "old_mesh.h5").string();
    {
        const HighFive::File src(meshPath, HighFive::File::ReadOnly);
        const std::vector<Real> x = readDs(src, "x");
        const std::vector<Real> y = readDs(src, "y");
        HDF5Writer w(oldMesh);
        w.writeAttribute("dim", readAttr<int>(src, "dim"));
        w.writeAttribute("N_elem", readAttr<int>(src, "N_elem"));
        w.writeAttribute("order", readAttr<int>(src, "order"));
        w.writeAttribute("N_q", readAttr<int>(src, "N_q"));
        w.writeAttribute("N_modes", readAttr<int>(src, "N_modes"));
        w.writeAttribute("element_type",
                         readAttr<std::string>(src, "element_type"));
        w.writeDataset("x", x.data(), x.size());
        w.writeDataset("y", y.data(), y.size());
        w.flush();
    }
    ok &= check(convertReadMeshFormat(oldMesh) == OutputFormat::Hdf5,
                "old mesh.h5 falls back to hdf5");

    // Explicit format still selectable on an old mesh (CLI --format path).
    const std::string oldConv =
        (fs::path(kResultsDir) / "old_converted.h5").string();
    convertCheckpoint(oldMesh, ckptPath, oldConv, OutputFormat::Tecplot);
    ok &= check(
        fs::exists((fs::path(kResultsDir) / "old_converted.plt").string()),
        "old mesh + explicit format -> plt");
    return ok;
}

} // namespace

int main() {
    struct Case {
        const char *name;
        bool (*fn)();
    };
    const Case cases[] = {
        {"writer_roundtrip", testWriterRoundtrip},
        {"config_output", testConfigOutput},
        {"solution_writer_uniform", testSolutionWriterUniform},
        {"checkpoint_convert_roundtrip", testCheckpointConvertRoundtrip},
        {"solver_end_to_end", testSolverEndToEnd},
        {"viz_connectivity", testVizConnectivity},
        {"viz_output_formats", testVizOutputFormats},
        {"convert_viz_dual_path", testConvertVizDualPath},
    };

    bool allOk = true;
    for (const Case &c : cases) {
        fs::remove_all(kScratchDir);
        std::cout << "== " << c.name << " ==\n";
        bool ok = false;
        try {
            ok = c.fn();
        } catch (const std::exception &e) {
            std::cout << "  FAIL exception: " << e.what() << "\n";
        }
        std::cout << (ok ? "  PASS " : "  FAIL ") << c.name << "\n";
        allOk &= ok;
    }

    fs::remove_all(kScratchDir);
    fs::remove_all("./results");
    std::remove(kScratchToml.c_str());
    std::cout << (allOk ? "test_hdf5_output: all passed\n"
                        : "test_hdf5_output: FAILURES\n");
    return allOk ? 0 : 1;
}
