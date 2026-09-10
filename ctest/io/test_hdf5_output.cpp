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

#include "config/Config.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "dg/DgField.hpp"
#include "io/CheckpointConvert.hpp"
#include "io/CheckpointWriter.hpp"
#include "io/HDF5Writer.hpp"
#include "io/SolutionWriter.hpp"
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
    ok &= checkInt(def.outputInterval(), 100, "default interval");
    ok &= checkStr(def.outputDirectory(), "./results", "default directory");

    const Config cfg(writeToml("[output]\nenable = true\nstrategy = "
                               "\"checkpoint\"\ninterval = 7\ndirectory = "
                               "\"./out_xyz\"\n"));
    ok &= check(cfg.outputEnable(), "enable parsed");
    ok &= check(cfg.outputStrategy() == OutputStrategy::Checkpoint,
                "strategy parsed");
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
        (fs::path(kResultsDir) / "solution_0000003.h5").string();
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
    const std::string meshPath = (fs::path(kResultsDir) / "mesh.h5").string();
    const std::string ckptPath =
        (fs::path(kResultsDir) / "checkpoint_0000005.h5").string();
    const std::string convPath =
        (fs::path(kResultsDir) / "converted.h5").string();
    convertCheckpoint(meshPath, ckptPath, convPath);

    const HighFive::File direct(
        (fs::path(kResultsDir) / "solution_0000005.h5").string(),
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
        char name[32];
        std::snprintf(name, sizeof(name), "solution_%07ld.h5", step);
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
