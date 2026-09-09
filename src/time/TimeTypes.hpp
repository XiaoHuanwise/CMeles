/// @file TimeTypes.hpp
/// @brief Shared callable types of the time-marching module.

#pragma once

#include <occa.hpp>

#include <functional>

/// @brief Right-hand side $\mathcal{R}(u) \to$ res, both device buffers.
///
/// Time is not an argument (matching the prototype); time-dependent
/// boundary conditions will be introduced through a closure later.
using RhsFunction = std::function<void(occa::memory, occa::memory)>;
