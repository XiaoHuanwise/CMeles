/// @file OutputWriterBase.hpp
/// @brief Abstract output boundary shared by the output strategies.

#pragma once

#include "common/Types.hpp"

/// @brief Coarse-grained output boundary: one virtual call per output
///        event, negligible next to the device copy and HDF5 write it
///        triggers (the same boundary-granularity rule as StepperBase).
class OutputWriterBase {
public:
    virtual ~OutputWriterBase() = default;

    /// @brief Persist the current field state.
    /// @param step  Time-step index (zero-padded into the file name).
    /// @param time  Physical time (root attribute).
    virtual void write(long step, Real time) = 0;
};
