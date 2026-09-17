/// @file Timer.hpp
/// @brief High-resolution wall-clock stopwatch for run-time reporting.

#pragma once

#include <chrono>

namespace cmeles {

/// @brief Monotonic wall-clock stopwatch.
///
/// Built on std::chrono::steady_clock (immune to wall-clock adjustments).
/// elapsed() reports seconds as double on purpose — wall-clock reporting
/// must not degrade to float when the project is built with
/// USE_FLOAT_PRECISION. Running spans accumulate in the clock's integer
/// tick count and are converted to double once per elapsed() call.
///
/// The stop()/resume() accumulation supports profiling: a section timer can
/// exclude nested sub-spans (e.g. march time minus output writes), and a
/// future named-section profiler can be layered on top unchanged.
/// Device-side kernel timing should bracket the section with
/// occa::device::finish() before reading elapsed().
class Timer {
public:
    /// @brief Restart: clear the accumulated time and start timing.
    void start() noexcept {
        accumulated_ = std::chrono::steady_clock::duration::zero();
        begin_       = std::chrono::steady_clock::now();
        running_     = true;
    }

    /// @brief Pause: add the running span to the accumulation. Idempotent.
    void stop() noexcept {
        if (running_) {
            accumulated_ += std::chrono::steady_clock::now() - begin_;
            running_ = false;
        }
    }

    /// @brief Continue accumulating after stop(). Idempotent while running.
    void resume() noexcept {
        if (!running_) {
            begin_   = std::chrono::steady_clock::now();
            running_ = true;
        }
    }

    /// @brief Clear the accumulated time and stop.
    void reset() noexcept {
        accumulated_ = std::chrono::steady_clock::duration::zero();
        running_     = false;
    }

    /// @brief Accumulated seconds, including the current running span.
    double elapsed() const noexcept {
        std::chrono::steady_clock::duration total = accumulated_;
        if (running_) {
            total += std::chrono::steady_clock::now() - begin_;
        }
        return std::chrono::duration<double>(total).count();
    }

    /// @brief Whether the watch is currently running.
    bool isRunning() const noexcept {
        return running_;
    }

private:
    std::chrono::steady_clock::duration accumulated_ =
        std::chrono::steady_clock::duration::zero();
    std::chrono::steady_clock::time_point begin_{};
    bool running_ = false;
};

} // namespace cmeles
