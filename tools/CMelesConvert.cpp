/// @file CMelesConvert.cpp
/// @brief Offline converter tool: strategy-B checkpoints -> strategy-A
///        solution files (host-only, no OCCA device required).
///
/// Usage:
///   CMelesConvert <directory>
///       Batch mode: converts every checkpoint_<step>.h5 in the directory
///       (paired with its mesh.h5) into solution_<step>.h5.
///   CMelesConvert <mesh.h5> <checkpoint.h5> [-o <out.h5>]
///       Single-file mode: converts one checkpoint; the output defaults to
///       solution_<step>.h5 next to the checkpoint.

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "config/Config.hpp"
#include "io/CheckpointConvert.hpp"

namespace fs = std::filesystem;

namespace {

void printUsage() {
    std::cerr
        << "Usage:\n"
        << "  CMelesConvert [--format <hdf5|xdmf|tecplot|all>] <directory>\n"
        << "      Batch mode: mesh.h5 + checkpoint_*.h5 -> solution_*.h5\n"
        << "  CMelesConvert [--format <...>] <mesh.h5> <checkpoint.h5> "
           "[-o <out.h5>]\n"
        << "      Single-file mode: converts one checkpoint; the output "
           "defaults to\n"
        << "      solution_<step>.h5 next to the checkpoint.\n"
        << "  The output format defaults to the one recorded in mesh.h5 at "
           "run time\n"
        << "  ([output] format; plain hdf5 for older files).\n";
}

/// @brief checkpoint_<step>.h5 -> solution_<step>.h5 (same directory).
fs::path solutionPathFor(const fs::path &checkpoint) {
    const std::string name = checkpoint.filename().string();
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

int runBatch(const fs::path &directory, const std::string &explicitFormat) {
    const fs::path mesh = directory / "mesh.h5";
    if (!fs::exists(mesh)) {
        std::cerr << "CMelesConvert: no mesh.h5 in " << directory << "\n";
        return 1;
    }
    const OutputFormat format = resolveFormat(explicitFormat, mesh);
    std::vector<fs::path> checkpoints;
    for (const auto &entry : fs::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (!entry.is_regular_file() || name.rfind("checkpoint_", 0) != 0 ||
            name.size() < 3 || name.substr(name.size() - 3) != ".h5") {
            continue;
        }
        checkpoints.push_back(entry.path());
    }
    std::sort(checkpoints.begin(), checkpoints.end());
    if (checkpoints.empty()) {
        std::cerr << "CMelesConvert: no checkpoint_*.h5 in " << directory
                  << "\n";
        return 1;
    }
    for (const auto &ckpt : checkpoints) {
        const fs::path out = solutionPathFor(ckpt);
        convertCheckpoint(mesh.string(), ckpt.string(), out.string(), format);
        std::cout << "CMelesConvert: " << ckpt.filename().string() << " -> "
                  << out.filename().string() << "\n";
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
