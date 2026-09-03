/// @file CMeles.cpp
/// @brief Program entry point: CMeles <config.toml>.
///
/// Full argument parsing, HDF5 output and checkpointing belong to the I/O
/// module (docs/tech_docs/io_module.md) and are not implemented in this
/// stage; the minimal CLI takes the configuration file path.

#include <iostream>
#include <string>

#include "config/Config.hpp"
#include "solver/CompressibleFlowSolver.hpp"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: CMeles <config.toml>\n";
        return 1;
    }
    try
    {
        const Config cfg(argv[1]);
        return runCompressibleFlowSolver(cfg);
    }
    catch (const std::exception &e)
    {
        std::cerr << "CMeles: error: " << e.what() << "\n";
        return 1;
    }
}
