/// @file test_timer.cpp
/// @brief Tests for the monotonic wall-clock stopwatch (cmeles::Timer).
///
/// Timing assertions use lower bounds only (sleep_for may oversleep under
/// load), plus exact-equality checks that hold deterministically while the
/// watch is stopped.

#include <chrono>
#include <iostream>
#include <thread>

#include "common/Timer.hpp"

namespace {
bool check(bool cond, const char *what) {
    if (!cond) {
        std::cout << "  FAIL " << what << "\n";
    }
    return cond;
}

void sleepMs(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
} // namespace

// ---------------------------------------------------------------------------
// 1. Fresh watch and running span
// ---------------------------------------------------------------------------

static bool testElapsed() {
    std::cout << "Test 1: elapsed while running\n";
    cmeles::Timer t;
    bool ok = true;

    ok &= check(!t.isRunning(), "fresh watch is stopped");
    ok &= check(t.elapsed() == 0.0, "fresh watch reads zero");

    t.start();
    ok &= check(t.isRunning(), "started watch is running");
    sleepMs(30);
    const double e = t.elapsed();
    ok &= check(e >= 0.03, "elapsed >= sleep duration");
    ok &= check(t.elapsed() >= e, "elapsed is non-decreasing");
    return ok;
}

// ---------------------------------------------------------------------------
// 2. stop() freezes the reading; paused time is not counted
// ---------------------------------------------------------------------------

static bool testStopFreezes() {
    std::cout << "Test 2: stop freezes the reading\n";
    cmeles::Timer t;
    bool ok = true;

    t.start();
    sleepMs(30);
    t.stop();
    ok &= check(!t.isRunning(), "stopped watch is not running");

    const double e1 = t.elapsed();
    ok &= check(e1 >= 0.03, "stopped reading keeps the span");
    sleepMs(20);
    const double e2 = t.elapsed();
    ok &= check(e1 == e2, "reading frozen while stopped (paused time free)");
    return ok;
}

// ---------------------------------------------------------------------------
// 3. resume() accumulates on top of the stopped reading
// ---------------------------------------------------------------------------

static bool testResumeAccumulates() {
    std::cout << "Test 3: resume accumulates\n";
    cmeles::Timer t;
    bool ok = true;

    t.start();
    sleepMs(20);
    t.stop();
    const double first = t.elapsed();

    t.resume();
    ok &= check(t.isRunning(), "resumed watch is running");
    sleepMs(20);
    ok &= check(t.elapsed() >= first + 0.02, "resumed span adds to the old");

    // Idempotence: double stop / double resume keep the reading sane.
    t.stop();
    t.stop();
    const double frozen = t.elapsed();
    ok &= check(frozen == t.elapsed(), "double stop is idempotent");
    t.resume();
    t.resume();
    ok &= check(t.isRunning(), "double resume is idempotent");
    return ok;
}

// ---------------------------------------------------------------------------
// 4. reset() clears the accumulation
// ---------------------------------------------------------------------------

static bool testReset() {
    std::cout << "Test 4: reset clears\n";
    cmeles::Timer t;
    bool ok = true;

    t.start();
    sleepMs(20);
    t.stop();
    ok &= check(t.elapsed() > 0.0, "accumulated something first");

    t.reset();
    ok &= check(!t.isRunning(), "reset watch is stopped");
    ok &= check(t.elapsed() == 0.0, "reset watch reads zero");
    return ok;
}

int main() {
    bool ok = true;
    struct Case {
        const char *name;
        bool (*fn)();
    };
    const Case cases[] = {
        {"testElapsed", testElapsed},
        {"testStopFreezes", testStopFreezes},
        {"testResumeAccumulates", testResumeAccumulates},
        {"testReset", testReset},
    };
    for (const auto &c : cases) {
        const bool r = c.fn();
        std::cout << "  [" << (r ? "PASS" : "FAIL") << "] " << c.name << "\n";
        ok &= r;
    }

    std::cout << (ok ? "ALL TIMER TESTS PASSED\n" : "TIMER TESTS FAILED\n");
    return ok ? 0 : 1;
}
