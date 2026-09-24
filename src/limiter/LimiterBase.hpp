/// @file LimiterBase.hpp
/// @brief Abstract interface of the discontinuity limiter subsystem.
///
/// A limiter modifies the modal coefficient vector in place on troubled
/// cells after every RK stage state update / implicit stage publish — the
/// hook is installed through StepperBase::setPostStageFilter (see the
/// filtering contract documented there and in
/// docs/tech_docs/mr_weno_limiter.md).

#pragma once

#include <occa.hpp>

/// @brief Abstract base for discontinuity limiters.
class LimiterBase {
public:
    virtual ~LimiterBase() = default;

    /// @brief Apply the limiter in place to the modal coefficient vector
    ///        \p o_u (layout [elem][var][mode]).
    ///
    /// Called once per stage state by the steppers; troubled-cell detection
    /// and the per-element skip happen inside the device kernels.
    virtual void apply(occa::memory o_u) = 0;
};
