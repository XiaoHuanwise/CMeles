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

#include "io/CheckpointConvert.hpp"

namespace fs = std::filesystem;

namespace {

void printUsage() {
    std::cerr
        << "Usage:\n"
        << "  CMelesConvert <directory>\n"
        << "      Batch mode: mesh.h5 + checkpoint_*.h5 -> solution_*.h5\n"
        << "  CMelesConvert <mesh.h5> <checkpoint.h5> [-o <out.h5>]\n";
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

int runBatch(const fs::path &directory) {
    const fs::path mesh = directory / "mesh.h5";
    if (!fs::exists(mesh)) {
        std::cerr << "CMelesConvert: no mesh.h5 in " << directory << "\n";
        return 1;
    }
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
        convertCheckpoint(mesh.string(), ckpt.string(), out.string());
        std::cout << "CMelesConvert: " << ckpt.filename().string() << " -> "
                  << out.filename().string() << "\n";
    }
    return 0;
}

int runSingle(const fs::path &mesh, const fs::path &checkpoint,
              const std::string &explicitOut) {
    const fs::path out = explicitOut.empty() ? solutionPathFor(checkpoint)
                                             : fs::path(explicitOut);
    convertCheckpoint(mesh.string(), checkpoint.string(), out.string());
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
        const fs::path first(argv[1]);
        if (argc == 2) {
            if (!fs::is_directory(first)) {
                printUsage();
                return 1;
            }
            return runBatch(first);
        }
        std::string explicitOut;
        for (int i = 3; i < argc; ++i) {
            if (std::string(argv[i]) == "-o" && i + 1 < argc) {
                explicitOut = argv[++i];
            } else {
                printUsage();
                return 1;
            }
        }
        return runSingle(argv[1], argv[2], explicitOut);
    } catch (const std::exception &e) {
        std::cerr << "CMelesConvert: error: " << e.what() << "\n";
        return 1;
    }
}
