/// @file CMelesConvert.cpp
/// @brief Offline converter tool: strategy-B checkpoints -> strategy-A
///        solution files (host-only, no OCCA device required).
///
/// Usage:
///   CMelesConvert <directory>
///       Batch mode: converts every checkpoint in the directory (paired
///       with its <prefix>_mesh.h5) into <prefix>_solution_<step>.h5 and,
///       for XDMF output, maintains <prefix>_series.xmf.
///   CMelesConvert <mesh.h5> <checkpoint.h5> [-o <out.h5>]
///       Single-file mode: converts one checkpoint; the output defaults to
///       the matching solution_<step>.h5 next to the checkpoint.

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "config/Config.hpp"
#include "io/CheckpointConvert.hpp"
#include "io/VizWriter.hpp"

namespace fs = std::filesystem;

namespace {

void printUsage() {
    std::cerr
        << "Usage:\n"
        << "  CMelesConvert [--format <hdf5|xdmf|tecplot|all>] <directory>\n"
        << "      Batch mode: <prefix>_mesh.h5 + <prefix>_checkpoint_*.h5\n"
        << "      -> <prefix>_solution_*.h5 (+ <prefix>_series.xmf for "
           "xdmf)\n"
        << "  CMelesConvert [--format <...>] <mesh.h5> <checkpoint.h5> "
           "[-o <out.h5>]\n"
        << "      Single-file mode: converts one checkpoint; the output "
           "defaults to\n"
        << "      the matching solution_<step>.h5 next to the checkpoint.\n"
        << "  The output format defaults to the one recorded in mesh.h5 at "
           "run time\n"
        << "  ([output] format; plain hdf5 for older files); the series "
           "index\n"
        << "  follows the recorded [output] series (default on).\n";
}

/// @brief <prefix>_checkpoint_<step>.h5 -> <prefix>_solution_<step>.h5
///        (legacy checkpoint_<step>.h5 -> solution_<step>.h5).
fs::path solutionPathFor(const fs::path &checkpoint) {
    const std::string name = checkpoint.filename().string();
    const std::string tag  = "_checkpoint_";
    const size_t pos       = name.rfind(tag);
    if (pos != std::string::npos) {
        return checkpoint.parent_path() / (name.substr(0, pos) + "_solution_" +
                                           name.substr(pos + tag.size()));
    }
    if (name.rfind("checkpoint_", 0) != 0) {
        throw std::runtime_error("not a checkpoint_<step>.h5 file: " + name);
    }
    return checkpoint.parent_path() /
           ("solution_" + name.substr(std::string("checkpoint_").size()));
}

/// @brief Resolve the output format: explicit flag, else the mesh.h5
///        record, else plain hdf5.
OutputFormat resolveFormat(const std::string &explicitFormat,
                           const fs::path &mesh) {
    if (!explicitFormat.empty()) {
        return parseOutputFormat(explicitFormat);
    }
    return convertReadMeshFormat(mesh.string());
}

/// @brief Read an int attribute from an h5 file.
int readMeshInt(const fs::path &mesh, const std::string &name, int fallback) {
    const HighFive::File f(mesh.string(), HighFive::File::ReadOnly);
    if (!f.hasAttribute(name)) {
        return fallback;
    }
    return f.getAttribute(name).read<int>();
}

/// @brief Read the element_type attribute ("quad"/"tri").
std::string readMeshElementType(const fs::path &mesh) {
    const HighFive::File f(mesh.string(), HighFive::File::ReadOnly);
    std::string t = "quad";
    if (f.hasAttribute("element_type")) {
        f.getAttribute("element_type").read(t);
    }
    return t;
}

/// @brief Locate the mesh file of a checkpoint directory: the newest
///        <prefix>_mesh.h5, falling back to the legacy mesh.h5.
fs::path findMesh(const fs::path &directory) {
    fs::path legacy = directory / "mesh.h5";
    if (fs::exists(legacy)) {
        return legacy;
    }
    for (const auto &entry : fs::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && name.size() >= 8 &&
            name.substr(name.size() - 8) == "_mesh.h5") {
            return entry.path();
        }
    }
    return legacy; // Non-existent: report below.
}

int runBatch(const fs::path &directory, const std::string &explicitFormat) {
    const fs::path mesh = findMesh(directory);
    if (!fs::exists(mesh)) {
        std::cerr << "CMelesConvert: no mesh.h5 / *_mesh.h5 in " << directory
                  << "\n";
        return 1;
    }
    // Output prefix: the mesh stem without the _mesh suffix, so the
    // converted files continue the run's <prefix>_ naming.
    std::string prefix = mesh.stem().string();
    if (prefix.size() > 5 && prefix.substr(prefix.size() - 5) == "_mesh") {
        prefix = prefix.substr(0, prefix.size() - 5);
    } else {
        prefix.clear(); // Legacy mesh.h5 -> legacy naming.
    }

    const OutputFormat format = resolveFormat(explicitFormat, mesh);
    const std::string ckptTag = prefix.empty() ? "checkpoint_" : "_checkpoint_";
    std::vector<fs::path> checkpoints;
    for (const auto &entry : fs::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (!entry.is_regular_file() || name.size() < 3 ||
            name.substr(name.size() - 3) != ".h5" ||
            name.rfind(ckptTag) == std::string::npos ||
            name.rfind("_mesh") != std::string::npos) {
            continue;
        }
        checkpoints.push_back(entry.path());
    }
    std::sort(checkpoints.begin(), checkpoints.end());
    if (checkpoints.empty()) {
        std::cerr << "CMelesConvert: no checkpoints in " << directory << "\n";
        return 1;
    }

    std::vector<viz::SeriesEntry> entries;
    for (const auto &ckpt : checkpoints) {
        const fs::path out = solutionPathFor(ckpt);
        const Real time    = convertCheckpoint(mesh.string(), ckpt.string(),
                                               out.string(), format);
        std::cout << "CMelesConvert: " << ckpt.filename().string() << " -> "
                  << out.filename().string() << "\n";
        if (formatWritesXdmf(format)) {
            entries.push_back({out.filename().string(), time});
        }
    }

    // Temporal-collection index for XDMF output: ParaView's Xdmf3 readers
    // do not lift a lone grid's <Time>, so the series file is the
    // supported time-series entry point.
    if (formatWritesXdmf(format) && convertReadMeshSeries(mesh.string())) {
        const int nElem            = readMeshInt(mesh, "N_elem", 0);
        const int nq               = readMeshInt(mesh, "N_q", 0);
        const std::string elemType = readMeshElementType(mesh);
        if (nElem > 0 && nq > 0) {
            const std::vector<int> isTriangle(static_cast<size_t>(nElem),
                                              elemType == "tri" ? 1 : 0);
            const viz::VizTopology topo =
                viz::buildVizConnectivity(nElem, nq + 2, isTriangle);
            const std::string seriesName =
                prefix.empty() ? "series.xmf" : prefix + "_series.xmf";
            viz::writeXdmfSeries((directory / seriesName).string(), entries,
                                 topo);
            std::cout << "CMelesConvert: " << entries.size()
                      << " steps indexed in " << seriesName << "\n";
        }
    }
    return 0;
}

int runSingle(const fs::path &mesh, const fs::path &checkpoint,
              const std::string &explicitOut,
              const std::string &explicitFormat) {
    const fs::path out = explicitOut.empty() ? solutionPathFor(checkpoint)
                                             : fs::path(explicitOut);
    const OutputFormat format = resolveFormat(explicitFormat, mesh);
    convertCheckpoint(mesh.string(), checkpoint.string(), out.string(), format);
    std::cout << "CMelesConvert: " << checkpoint.filename().string() << " -> "
              << out.string() << "\n";
    return 0;
}

} // namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }
    try {
        // Leading options: --format <name> (repeatable-tolerant: last wins).
        std::string explicitFormat;
        int first = 1;
        while (first + 1 < argc && std::string(argv[first]) == "--format") {
            explicitFormat = argv[first + 1];
            // Validate early for a clear error message.
            parseOutputFormat(explicitFormat);
            first += 2;
        }
        const fs::path firstPath(argv[first]);
        if (argc == first + 1) {
            if (!fs::is_directory(firstPath)) {
                printUsage();
                return 1;
            }
            return runBatch(firstPath, explicitFormat);
        }
        std::string explicitOut;
        for (int i = first + 2; i < argc; ++i) {
            if (std::string(argv[i]) == "-o" && i + 1 < argc) {
                explicitOut = argv[++i];
            } else {
                printUsage();
                return 1;
            }
        }
        return runSingle(argv[first], argv[first + 1], explicitOut,
                         explicitFormat);
    } catch (const std::exception &e) {
        std::cerr << "CMelesConvert: error: " << e.what() << "\n";
        return 1;
    }
}
