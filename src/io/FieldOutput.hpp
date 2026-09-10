/// @file FieldOutput.hpp
/// @brief Output facade owning the output directory and dispatching to
///        the strategy-specific writer selected in [output].

#pragma once

#include <memory>

#include "io/OutputWriterBase.hpp"

class Config;
class DgField;

/// @brief Creates the output directory and owns the configured output
///        strategy. Constructed only when [output] enable is true.
class FieldOutput {
public:
    /// @param cfg    Configuration ([output] strategy / directory).
    /// @param field  DG field (passed through to the strategy writer).
    /// @throws std::runtime_error if the directory cannot be created.
    FieldOutput(const Config &cfg, const DgField &field);

    ~FieldOutput();

    /// @brief Persist the current field state through the strategy writer.
    void write(long step, Real time);

private:
    std::unique_ptr<OutputWriterBase> writer_;
};
