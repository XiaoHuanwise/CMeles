/// @file CMeles.cpp
/// @brief Program entry point: CMeles <config.toml>.
///
/// HDF5 output (and checkpointing) is controlled by the [output] section
/// of the configuration file (docs/tech_docs/io_module.md); the CLI takes
/// the configuration file path.

#include <iostream>
#include <string>

#include "config/Config.hpp"
#include "solver/CompressibleFlowSolver.hpp"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: CMeles <config.toml>\n";
        return 1;
    }
    try {
        const Config cfg(argv[1]);
        return runCompressibleFlowSolver(cfg);
    } catch (const std::exception &e) {
        std::cerr << "CMeles: error: " << e.what() << "\n";
        return 1;
    }
}
