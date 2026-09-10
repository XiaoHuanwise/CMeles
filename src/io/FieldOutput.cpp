/// @file FieldOutput.cpp
/// @brief Implementation of the output facade.

#include "io/FieldOutput.hpp"

#include <filesystem>
#include <stdexcept>

#include "config/Config.hpp"
#include "io/CheckpointWriter.hpp"
#include "io/SolutionWriter.hpp"

FieldOutput::FieldOutput(const Config &cfg, const DgField &field) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(cfg.outputDirectory(), ec);
    if (ec) {
        throw std::runtime_error(
            "FieldOutput: cannot create output directory '" +
            cfg.outputDirectory() + "': " + ec.message());
    }
    switch (cfg.outputStrategy()) {
        case OutputStrategy::Direct:
            writer_ = std::make_unique<SolutionWriter>(cfg, field);
            break;
        case OutputStrategy::Checkpoint: {
            // The mesh companion file is written once up front; every
            // later write only dumps the modal coefficients.
            auto checkpoint = std::make_unique<CheckpointWriter>(cfg, field);
            checkpoint->writeMesh();
            writer_ = std::move(checkpoint);
            break;
        }
    }
}

FieldOutput::~FieldOutput() = default;

void FieldOutput::write(long step, Real time) {
    writer_->write(step, time);
}
