/// @file StepperFactory.hpp
/// @brief Config-driven stepper construction: the single runtime dispatch
///        point of the time-marching module.
///
/// All knowledge of which stepper a [time_marching] method maps to (and how
/// to configure it — Butcher tableau, tolerances, DITR variant) lives here,
/// so adding a time method touches the time module only.

#pragma once

#include <occa.hpp>

#include <functional>
#include <memory>

#include "core/DeviceMemoryManager.hpp"
#include "time/StepperBase.hpp"
#include "time/TimeTypes.hpp"

class Config;

/// @brief Stepper construction callback: (device, memory manager, rhs,
///        nDof) -> owned stepper.
using StepperFactory = std::function<std::unique_ptr<StepperBase>(
    occa::device &, DeviceMemoryManager &, const RhsFunction &, occa::dim_t)>;

/// @brief Map [time_marching] method to a factory constructing the matching
///        stepper (Euler / SSPRK3 / embedded RK by tableau / dual time over
///        a Backward-Euler or DITR residual).
/// @throws std::invalid_argument when an embedded-RK tableau is required
///         but the method (or pseudo_method) is not an embedded pair, or
///         the method is unknown.
StepperFactory makeStepperFactory(const Config &cfg);
