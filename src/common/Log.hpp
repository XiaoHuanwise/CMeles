/// @file Log.hpp
/// @brief Depth-gated console logging for run-time output.

#pragma once

#include <atomic>
#include <iostream>
#include <sstream>
#include <utility>

namespace cmeles {

/// @brief Log depth ladder ([log] level in the TOML configuration).
///
/// A message prints when its depth is <= the configured output level, so
/// larger values mean more verbose output. The ladder is open at the top:
/// deeper layers (per-pseudo-step detail, kernel counters, ...) can be
/// introduced by appending higher constants.
inline constexpr int kLogSilent = 0; ///< Output level: print nothing.
inline constexpr int kLogNormal = 1; ///< Physical-step layer: banner,
                                     ///< progress lines, aborts, summary.
inline constexpr int kLogDetail = 2; ///< Inner-iteration layer: dual-time
                                     ///< per-step pseudo statistics.

/// @brief Global output-level gate (set once at startup from [log] level).
class Log {
public:
    /// @brief Set the output level (see the depth constants).
    static void setLevel(int level) noexcept {
        currentLevel_.store(level, std::memory_order_relaxed);
    }

    /// @brief The configured output level.
    static int level() noexcept {
        return currentLevel_.load(std::memory_order_relaxed);
    }

    /// @brief Whether a message at \p depth prints: $depth \le level$.
    static bool enabled(int depth) noexcept {
        return depth <= level();
    }

private:
    inline static std::atomic<int> currentLevel_{kLogNormal};
};

/// @brief RAII log-line builder used through the CMES_LOG macros.
///
/// Streams into an internal buffer and writes the complete line (plus the
/// trailing newline) to stdout when the statement finishes, so one line
/// never interleaves with lines written by other OpenMP threads. The depth
/// is carried along for a future per-depth routing (e.g. a log file).
class LogLine {
public:
    explicit LogLine(int depth) : depth_(depth) {
    }

    LogLine(const LogLine &)            = delete;
    LogLine &operator=(const LogLine &) = delete;

    template <typename T> LogLine &operator<<(T &&value) {
        buffer_ << std::forward<T>(value);
        return *this;
    }

    ~LogLine() {
        buffer_ << '\n';
        std::cout << buffer_.str();
    }

    /// @brief The depth this line was requested at.
    int depth() const noexcept {
        return depth_;
    }

private:
    std::ostringstream buffer_;
    int depth_;
};

/// @brief Stream a log line at \p depth when the output level allows it;
///        the stream operands are not evaluated otherwise. Dangling-else
///        safe.
#define CMES_LOG(depth)                                                        \
    if (!cmeles::Log::enabled(depth)) {                                        \
    } else                                                                     \
        cmeles::LogLine(depth)

/// @brief Physical-step layer: banner, progress lines, aborts, summary.
#define CMES_LOG_NORMAL CMES_LOG(cmeles::kLogNormal)

/// @brief Inner-iteration layer: dual-time pseudo statistics.
#define CMES_LOG_DETAIL CMES_LOG(cmeles::kLogDetail)

} // namespace cmeles
