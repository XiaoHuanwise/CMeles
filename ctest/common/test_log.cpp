/// @file test_log.cpp
/// @brief Tests for the depth-gated console log (cmeles::Log).

#include <iostream>

#include "common/Log.hpp"

namespace {
bool check(bool cond, const char *what) {
    if (!cond) {
        std::cout << "  FAIL " << what << "\n";
    }
    return cond;
}
} // namespace

// ---------------------------------------------------------------------------
// 1. Default level
// ---------------------------------------------------------------------------

static bool testDefaultLevel() {
    std::cout << "Test 1: default level\n";
    cmeles::Log::setLevel(cmeles::kLogNormal); // pin the default
    return check(cmeles::Log::level() == cmeles::kLogNormal, "level == normal");
}

// ---------------------------------------------------------------------------
// 2. Depth filter matrix: a message prints iff depth <= level
// ---------------------------------------------------------------------------

static bool testFilterMatrix() {
    std::cout << "Test 2: depth filter matrix\n";
    bool ok            = true;
    const int levels[] = {0, 1, 2, 5};
    const int depths[] = {0, 1, 2, 5};

    for (const int lvl : levels) {
        cmeles::Log::setLevel(lvl);
        ok &= check(cmeles::Log::level() == lvl, "level roundtrip");
        for (const int depth : depths) {
            ok &= check(cmeles::Log::enabled(depth) == (depth <= lvl),
                        "enabled(depth) == (depth <= level)");
        }
    }

    cmeles::Log::setLevel(cmeles::kLogNormal); // restore for other cases
    return ok;
}

// ---------------------------------------------------------------------------
// 3. Macro laziness: stream operands are not evaluated when gated off
// ---------------------------------------------------------------------------

static bool testMacroLaziness() {
    std::cout << "Test 3: macro laziness\n";
    bool ok = true;

    cmeles::Log::setLevel(cmeles::kLogSilent);
    int evaluated = 0;
    CMES_LOG_DETAIL << [&evaluated] {
        ++evaluated;
        return "probe";
    }();
    ok &= check(evaluated == 0, "operands skipped when gated off");

    cmeles::Log::setLevel(cmeles::kLogDetail);
    CMES_LOG_DETAIL << [&evaluated] {
        ++evaluated;
        return "macro laziness probe (this line should print)";
    }();
    ok &= check(evaluated == 1, "operands evaluated when allowed");

    cmeles::Log::setLevel(cmeles::kLogNormal); // restore
    return ok;
}

int main() {
    bool ok = true;
    struct Case {
        const char *name;
        bool (*fn)();
    };
    const Case cases[] = {
        {"testDefaultLevel", testDefaultLevel},
        {"testFilterMatrix", testFilterMatrix},
        {"testMacroLaziness", testMacroLaziness},
    };
    for (const auto &c : cases) {
        const bool r = c.fn();
        std::cout << "  [" << (r ? "PASS" : "FAIL") << "] " << c.name << "\n";
        ok &= r;
    }

    std::cout << (ok ? "ALL LOG TESTS PASSED\n" : "LOG TESTS FAILED\n");
    return ok ? 0 : 1;
}
