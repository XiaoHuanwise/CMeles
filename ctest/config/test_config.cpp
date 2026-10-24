/// @file test_config.cpp
/// @brief Tests for the Config class: default construction and TOML parsing.

#include <cstdio>
#include <iostream>
#include <string>

#include "config/Config.hpp"

namespace {
/// @brief Write \p content to a temporary file and return its path.
/// The file is created next to the test binary so ctest's working directory
/// (the build tree) always contains it.
std::string writeTempToml(const std::string &content) {
    static const char *kPath = "test_config_scratch.toml";
    std::FILE *f             = std::fopen(kPath, "w");
    if (!f) {
        return "";
    }
    std::fputs(content.c_str(), f);
    std::fclose(f);
    return kPath;
}

bool check(bool cond, const char *what) {
    if (!cond) {
        std::cout << "  FAIL " << what << "\n";
    }
    return cond;
}

bool checkInt(int value, int ref, const char *what) {
    if (value != ref) {
        std::cout << "  FAIL " << what << ": value=" << value << " ref=" << ref
                  << "\n";
        return false;
    }
    return true;
}

bool checkReal(Real value, Real ref, const char *what) {
    const Real tol = Real(1e3) * RealEpsilon;
    if (std::abs(value - ref) > tol * std::max(std::abs(ref), RealEpsilon)) {
        std::cout << "  FAIL " << what << ": value=" << value << " ref=" << ref
                  << "\n";
        return false;
    }
    return true;
}

bool checkStr(const std::string &value, const std::string &ref,
              const char *what) {
    if (value != ref) {
        std::cout << "  FAIL " << what << ": value='" << value << "' ref='"
                  << ref << "'\n";
        return false;
    }
    return true;
}
} // namespace

// ---------------------------------------------------------------------------
// 1. Default construction (no file)
// ---------------------------------------------------------------------------

static bool testDefaults() {
    std::cout << "Test 1: default construction\n";
    Config cfg;
    bool ok = true;

    ok &= checkInt(cfg.polynomialOrder(), 2, "polynomialOrder");
    ok &= checkInt(cfg.quadratureOrder(), 4, "quadratureOrder");
    ok &= checkInt(cfg.meshNx(), 8, "meshNx");
    ok &= checkInt(cfg.meshNy(), 8, "meshNy");
    ok &= checkReal(cfg.meshX0(), Real(0), "meshX0");
    ok &= checkReal(cfg.meshDx(), Real(1), "meshDx");
    ok &= checkReal(cfg.gamma(), Real(1.4), "gamma");
    ok &= checkReal(cfg.flowMach(), Real(0.1), "flowMach");
    ok &= checkStr(cfg.occaMode(), "Serial", "occaMode");
    ok &= checkInt(cfg.occaThreads(), 0, "occaThreads default");
    ok &= checkInt(cfg.occaPlatform(), 0, "occaPlatform default");
    ok &= checkInt(cfg.occaDevice(), 0, "occaDevice default");
    ok &= check(cfg.occaOpenclO3(), "occaOpenclO3 default");
    ok &= check(cfg.fluxType() == FluxType::Llf, "fluxType == Llf");
    ok &= checkInt(cfg.fluxTypeInt(), 0, "fluxTypeInt");
    ok &= checkInt(cfg.logLevel(), 1, "logLevel default");
    ok &= checkInt(cfg.logInterval(), 100, "logInterval default");
    return ok;
}

// ---------------------------------------------------------------------------
// 2. TOML parsing: full file
// ---------------------------------------------------------------------------

static bool testParseFull() {
    std::cout << "Test 2: full TOML file\n";
    const std::string content = R"(
[basis]
order = 3
nq = 5

[mesh]
nx = 4
ny = 6
x0 = -1.0
y0 = -2.0
dx = 0.5
dy = 0.25
shear = 0.125
split_triangles = true

[riemann]
flux = "llf"

[gas]
gamma = 1.3

[flow]
rho = 1.2
u = 0.3
v = -0.4
p = 2.5
mach = 0.8

[occa]
mode = "OpenMP"
threads = 8
platform = 1
device = 2
opencl_o3 = false

[log]
level = 2
interval = 25
)";
    const std::string path    = writeTempToml(content);
    if (path.empty()) {
        std::cout << "  FAIL could not write scratch toml\n";
        return false;
    }

    Config cfg(path);
    bool ok = true;

    ok &= checkInt(cfg.polynomialOrder(), 3, "polynomialOrder");
    ok &= checkInt(cfg.quadratureOrder(), 5, "quadratureOrder");
    ok &= checkInt(cfg.meshNx(), 4, "meshNx");
    ok &= checkInt(cfg.meshNy(), 6, "meshNy");
    ok &= checkReal(cfg.meshX0(), Real(-1), "meshX0");
    ok &= checkReal(cfg.meshY0(), Real(-2), "meshY0");
    ok &= checkReal(cfg.meshDx(), Real(0.5), "meshDx");
    ok &= checkReal(cfg.meshDy(), Real(0.25), "meshDy");
    ok &= checkReal(cfg.meshShear(), Real(0.125), "meshShear");
    ok &= check(cfg.meshSplitTriangles(), "meshSplitTriangles");
    ok &= check(cfg.fluxType() == FluxType::Llf, "fluxType");
    ok &= checkReal(cfg.gamma(), Real(1.3), "gamma");
    ok &= checkReal(cfg.flowRho(), Real(1.2), "flowRho");
    ok &= checkReal(cfg.flowU(), Real(0.3), "flowU");
    ok &= checkReal(cfg.flowV(), Real(-0.4), "flowV");
    ok &= checkReal(cfg.flowP(), Real(2.5), "flowP");
    ok &= checkReal(cfg.flowMach(), Real(0.8), "flowMach");
    ok &= checkStr(cfg.occaMode(), "OpenMP", "occaMode");
    ok &= checkInt(cfg.occaThreads(), 8, "occaThreads (parsed)");
    ok &= checkInt(cfg.occaPlatform(), 1, "occaPlatform (parsed)");
    ok &= checkInt(cfg.occaDevice(), 2, "occaDevice (parsed)");
    ok &= check(!cfg.occaOpenclO3(), "occaOpenclO3 (parsed)");
    ok &= checkInt(cfg.logLevel(), 2, "logLevel (parsed)");
    ok &= checkInt(cfg.logInterval(), 25, "logInterval (parsed)");

    std::remove(path.c_str());
    return ok;
}

// ---------------------------------------------------------------------------
// 3. Partial file: missing sections fall back to defaults
// ---------------------------------------------------------------------------

static bool testParsePartial() {
    std::cout << "Test 3: partial TOML file (fallback defaults)\n";
    const std::string content = R"(
[basis]
order = 1

[riemann]
flux = "llf"
)";
    const std::string path    = writeTempToml(content);
    if (path.empty()) {
        std::cout << "  FAIL could not write scratch toml\n";
        return false;
    }

    Config cfg(path);
    bool ok = true;

    ok &= checkInt(cfg.polynomialOrder(), 1, "polynomialOrder (parsed)");
    ok &= checkInt(cfg.quadratureOrder(), 4, "quadratureOrder (default)");
    ok &= checkInt(cfg.meshNx(), 8, "meshNx (default)");
    ok &= checkReal(cfg.gamma(), Real(1.4), "gamma (default)");

    std::remove(path.c_str());
    return ok;
}

// ---------------------------------------------------------------------------
// 4. [occa] threads validation
// ---------------------------------------------------------------------------

static bool testOcaThreadsValidation() {
    std::cout << "Test 4: occa threads validation\n";
    bool ok = true;

    // Negative count must throw; zero (env default) must be accepted.
    bool threw                = false;
    const std::string bad     = R"(
[time_marching]
t_final = 1.0

[occa]
threads = -2
)";
    const std::string badPath = writeTempToml(bad);
    try {
        Config cfg(badPath);
    } catch (const std::exception &) {
        threw = true;
    }
    ok &= check(threw, "negative threads throws");
    std::remove(badPath.c_str());

    const std::string good     = R"(
[time_marching]
t_final = 1.0

[occa]
mode = "OpenMP"
threads = 0
)";
    const std::string goodPath = writeTempToml(good);
    try {
        Config cfg(goodPath);
        ok &= checkInt(cfg.occaThreads(), 0, "threads = 0 accepted");
    } catch (const std::exception &e) {
        std::cout << "  FAIL threads = 0 threw: " << e.what() << "\n";
        ok = false;
    }
    std::remove(goodPath.c_str());
    return ok;
}

// ---------------------------------------------------------------------------
// 5. [time_marching] pseudo tolerance keys
// ---------------------------------------------------------------------------

static bool testPseudoToleranceKeys() {
    std::cout << "Test 5: pseudo tolerance keys\n";
    bool ok = true;

    // Unset keys keep the sentinel 0 (automatic heuristic); explicit values
    // parse through; negative values must throw.
    const std::string good     = R"(
[time_marching]
t_final = 1.0
pseudo_rtol = 1e-6
pseudo_atol = 2e-6
)";
    const std::string goodPath = writeTempToml(good);
    try {
        Config cfg(goodPath);
        ok &=
            checkReal(cfg.timePseudoRtol(), Real(1e-6), "pseudo_rtol (parsed)");
        ok &=
            checkReal(cfg.timePseudoAtol(), Real(2e-6), "pseudo_atol (parsed)");
    } catch (const std::exception &e) {
        std::cout << "  FAIL explicit pseudo tolerances threw: " << e.what()
                  << "\n";
        ok = false;
    }
    std::remove(goodPath.c_str());

    const std::string unset     = R"(
[time_marching]
t_final = 1.0
)";
    const std::string unsetPath = writeTempToml(unset);
    try {
        Config cfg(unsetPath);
        ok &= checkReal(cfg.timePseudoRtol(), Real(0),
                        "pseudo_rtol default (auto)");
        ok &= checkReal(cfg.timePseudoAtol(), Real(0),
                        "pseudo_atol default (auto)");
    } catch (const std::exception &e) {
        std::cout << "  FAIL unset pseudo tolerances threw: " << e.what()
                  << "\n";
        ok = false;
    }
    std::remove(unsetPath.c_str());

    const std::string bad     = R"(
[time_marching]
t_final = 1.0
pseudo_rtol = -1e-6
)";
    const std::string badPath = writeTempToml(bad);
    bool threw                = false;
    try {
        Config cfg(badPath);
    } catch (const std::exception &) {
        threw = true;
    }
    ok &= check(threw, "negative pseudo_rtol throws");
    std::remove(badPath.c_str());
    return ok;
}

// ---------------------------------------------------------------------------
// 6. FluxType name mapping
// ---------------------------------------------------------------------------

static bool testFluxTypeNames() {
    std::cout << "Test 6: flux type names\n";
    bool ok = true;
    ok &= check(parseFluxType("llf") == FluxType::Llf, "parse 'llf'");
    ok &= check(parseFluxType("LLF") == FluxType::Llf, "parse 'LLF' (upper)");
    ok &= check(parseFluxType("roe") == FluxType::Roe, "parse 'roe'");
    ok &= check(parseFluxType("roe-e") == FluxType::RoeE, "parse 'roe-e'");
    ok &= check(parseFluxType("steger-warming") == FluxType::StegerWarming,
                "parse 'steger-warming'");
    ok &= check(parseFluxType("van_leer") == FluxType::VanLeer,
                "parse 'van_leer'");
    ok &= checkStr(fluxTypeName(FluxType::Llf), "llf", "name 'llf'");
    ok &= checkStr(fluxTypeName(FluxType::RoeE), "roe-e", "name 'roe-e'");

    bool threw = false;
    try {
        (void)parseFluxType("hllc");
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= check(threw, "unknown name throws");
    return ok;
}

// ---------------------------------------------------------------------------
// 6b. BcType / RiemannTarget name mapping
// ---------------------------------------------------------------------------

static bool testBcTypeNames() {
    std::cout << "Test 6b: bc type names\n";
    bool ok = true;
    ok &=
        check(parseBcType("farfield") == BcType::Farfield, "parse 'farfield'");
    ok &=
        check(parseBcType("periodic") == BcType::Periodic, "parse 'periodic'");
    ok &= check(parseBcType("free") == BcType::Free, "parse 'free'");
    ok &= check(parseBcType("RIEMANN") == BcType::Riemann,
                "parse 'RIEMANN' (upper)");
    ok &= check(parseBcType("characteristic") == BcType::Riemann,
                "parse 'characteristic' (alias)");
    ok &= check(parseBcType("pressure-farfield") == BcType::Riemann,
                "parse 'pressure-farfield' (alias)");
    ok &= checkStr(bcTypeName(BcType::Riemann), "riemann", "name 'riemann'");
    ok &= check(parseBcType("slip") == BcType::Slip, "parse 'slip'");
    ok &= check(parseBcType("slip-wall") == BcType::Slip,
                "parse 'slip-wall' (alias)");
    ok &= check(parseBcType("wall") == BcType::Slip, "parse 'wall' (alias)");
    ok &= checkStr(bcTypeName(BcType::Slip), "slip", "name 'slip'");
    ok &= check(parseBcType("expr") == BcType::Expr, "parse 'expr'");
    ok &= check(parseBcType("dirichlet") == BcType::Expr,
                "parse 'dirichlet' (alias)");
    ok &= checkStr(bcTypeName(BcType::Expr), "expr", "name 'expr'");
    ok &= check(parseRiemannTarget("flow") == RiemannTarget::Flow,
                "target 'flow'");
    ok &= check(parseRiemannTarget("ic") == RiemannTarget::Ic, "target 'ic'");
    ok &= checkStr(riemannTargetName(RiemannTarget::Ic), "ic",
                   "target name 'ic'");

    bool threw = false;
    try {
        (void)parseBcType("no-slip");
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= check(threw, "unknown bc name throws");
    threw = false;
    try {
        (void)parseRiemannTarget("ghost");
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= check(threw, "unknown riemann_target throws");
    return ok;
}

// ---------------------------------------------------------------------------
// 6c. Per-side [bc] override tables + Dirichlet expressions
// ---------------------------------------------------------------------------

static bool testBcSideTables() {
    std::cout << "Test 6c: per-side [bc] override tables\n";
    bool ok = true;

    const std::string toml     = R"(
[bc]
type = "free"
rho = "1.4"
u = "0.0"
v = "0.0"
p = "1.0"

[bc.y_low]
type = "slip"
split_x = 0.16666666666666666
split_type = "expr"

[bc.x_low]
type = "expr"
)";
    const std::string tomlPath = writeTempToml(toml);
    try {
        const Config cfg(tomlPath);
        ok &= check(cfg.bcType() == BcType::Free, "global type free");

        const SideBc &yLow = cfg.bcSide(BcSide::YLow);
        ok &= check(yLow.hasType && yLow.type == BcType::Slip,
                    "y_low override slip");
        ok &= check(yLow.hasSplit, "y_low split present");
        ok &= checkReal(yLow.splitX, Real(1.0 / 6.0), "y_low split_x 1/6");
        ok &= check(yLow.splitType == BcType::Expr, "y_low split_type expr");

        const SideBc &xLow = cfg.bcSide(BcSide::XLow);
        ok &= check(xLow.hasType && xLow.type == BcType::Expr,
                    "x_low override expr");
        ok &= check(!xLow.hasSplit, "x_low no split");

        ok &= check(!cfg.bcSide(BcSide::XHigh).hasType &&
                        !cfg.bcSide(BcSide::YHigh).hasType,
                    "x_high/y_high fall back to the global type");

        ok &= checkStr(cfg.bcRho(), "1.4", "bc expression rho");
        ok &= checkStr(cfg.bcP(), "1.0", "bc expression p");
    } catch (const std::exception &e) {
        std::cout << "  FAIL side tables threw: " << e.what() << "\n";
        ok = false;
    }
    std::remove(tomlPath.c_str());

    // Per-side periodic is rejected (pairing is mesh-global).
    const std::string periodic     = R"(
[bc]
type = "free"

[bc.x_low]
type = "periodic"
)";
    const std::string periodicPath = writeTempToml(periodic);
    bool threw                     = false;
    try {
        const Config cfg(periodicPath);
        (void)cfg;
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= check(threw, "per-side periodic throws");
    std::remove(periodicPath.c_str());
    return ok;
}

// ---------------------------------------------------------------------------
// 7. [log] section validation
// ---------------------------------------------------------------------------

static bool testLogValidation() {
    std::cout << "Test 7: log section validation\n";
    bool ok = true;

    // level = 0 (silent) and large levels (deeper than any current emitter)
    // are both valid; interval = 0 must throw.
    const std::string silent     = R"(
[log]
level = 0
)";
    const std::string silentPath = writeTempToml(silent);
    try {
        Config cfg(silentPath);
        ok &= checkInt(cfg.logLevel(), 0, "level = 0 accepted");
    } catch (const std::exception &e) {
        std::cout << "  FAIL level = 0 threw: " << e.what() << "\n";
        ok = false;
    }
    std::remove(silentPath.c_str());

    const std::string deep     = R"(
[log]
level = 5
)";
    const std::string deepPath = writeTempToml(deep);
    try {
        Config cfg(deepPath);
        ok &= checkInt(cfg.logLevel(), 5, "level = 5 accepted (no cap)");
    } catch (const std::exception &e) {
        std::cout << "  FAIL level = 5 threw: " << e.what() << "\n";
        ok = false;
    }
    std::remove(deepPath.c_str());

    const std::string badLevel     = R"(
[log]
level = -1
)";
    const std::string badLevelPath = writeTempToml(badLevel);
    bool threw                     = false;
    try {
        Config cfg(badLevelPath);
    } catch (const std::exception &) {
        threw = true;
    }
    ok &= check(threw, "negative level throws");
    std::remove(badLevelPath.c_str());

    const std::string badInterval     = R"(
[log]
interval = 0
)";
    const std::string badIntervalPath = writeTempToml(badInterval);
    threw                             = false;
    try {
        Config cfg(badIntervalPath);
    } catch (const std::exception &) {
        threw = true;
    }
    ok &= check(threw, "zero interval throws");
    std::remove(badIntervalPath.c_str());
    return ok;
}

// ---------------------------------------------------------------------------
// 8. [time_marching] PI-controller keys (safety / min_factor / max_factor /
//    max_growth)
// ---------------------------------------------------------------------------

static bool testControllerKeys() {
    std::cout << "Test 8: PI-controller keys\n";
    bool ok = true;

    // Unset keys keep the defaults; explicit values parse through; invalid
    // combinations must throw.
    const std::string unset     = R"(
[time_marching]
t_final = 1.0
)";
    const std::string unsetPath = writeTempToml(unset);
    try {
        Config cfg(unsetPath);
        ok &= checkReal(cfg.timeSafety(), Real(0.9), "safety default");
        ok &= checkReal(cfg.timeMinFactor(), Real(0.2), "min_factor default");
        ok &= checkReal(cfg.timeMaxFactor(), Real(5), "max_factor default");
        ok &= checkReal(cfg.timeMaxGrowth(), Real(10), "max_growth default");
    } catch (const std::exception &e) {
        std::cout << "  FAIL unset controller keys threw: " << e.what() << "\n";
        ok = false;
    }
    std::remove(unsetPath.c_str());

    const std::string good     = R"(
[time_marching]
t_final = 1.0
safety = 0.8
min_factor = 0.05
max_factor = 50
max_growth = 20
)";
    const std::string goodPath = writeTempToml(good);
    try {
        Config cfg(goodPath);
        ok &= checkReal(cfg.timeSafety(), Real(0.8), "safety (parsed)");
        ok &= checkReal(cfg.timeMinFactor(), Real(0.05), "min_factor (parsed)");
        ok &= checkReal(cfg.timeMaxFactor(), Real(50), "max_factor (parsed)");
        ok &= checkReal(cfg.timeMaxGrowth(), Real(20), "max_growth (parsed)");
    } catch (const std::exception &e) {
        std::cout << "  FAIL explicit controller keys threw: " << e.what()
                  << "\n";
        ok = false;
    }
    std::remove(goodPath.c_str());

    const std::string badZero     = R"(
[time_marching]
t_final = 1.0
min_factor = 0.0
)";
    const std::string badZeroPath = writeTempToml(badZero);
    bool threw                    = false;
    try {
        Config cfg(badZeroPath);
    } catch (const std::exception &) {
        threw = true;
    }
    ok &= check(threw, "zero min_factor throws");
    std::remove(badZeroPath.c_str());

    const std::string badCross     = R"(
[time_marching]
t_final = 1.0
min_factor = 2.0
max_factor = 1.0
)";
    const std::string badCrossPath = writeTempToml(badCross);
    threw                          = false;
    try {
        Config cfg(badCrossPath);
    } catch (const std::exception &) {
        threw = true;
    }
    ok &= check(threw, "max_factor < min_factor throws");
    std::remove(badCrossPath.c_str());
    return ok;
}

// ---------------------------------------------------------------------------
// 9. [time_marching] pseudo_dt_mode (global / local adaptive pseudo dt)
// ---------------------------------------------------------------------------

static bool testPseudoDtModeKey() {
    std::cout << "Test 9: pseudo_dt_mode key\n";
    bool ok = true;

    // Unset keeps the global (scalar) default; "local" parses through;
    // unknown names must throw.
    const std::string unset     = R"(
[time_marching]
t_final = 1.0
)";
    const std::string unsetPath = writeTempToml(unset);
    try {
        Config cfg(unsetPath);
        ok &= check(cfg.timePseudoDtMode() == PseudoDtMode::Global,
                    "pseudo_dt_mode default (global)");
    } catch (const std::exception &e) {
        std::cout << "  FAIL unset pseudo_dt_mode threw: " << e.what() << "\n";
        ok = false;
    }
    std::remove(unsetPath.c_str());

    const std::string good     = R"(
[time_marching]
t_final = 1.0
pseudo_dt_mode = "local"
)";
    const std::string goodPath = writeTempToml(good);
    try {
        Config cfg(goodPath);
        ok &= check(cfg.timePseudoDtMode() == PseudoDtMode::Local,
                    "pseudo_dt_mode (parsed)");
    } catch (const std::exception &e) {
        std::cout << "  FAIL explicit pseudo_dt_mode threw: " << e.what()
                  << "\n";
        ok = false;
    }
    std::remove(goodPath.c_str());

    const std::string bad     = R"(
[time_marching]
t_final = 1.0
pseudo_dt_mode = "scalar"
)";
    const std::string badPath = writeTempToml(bad);
    bool threw                = false;
    try {
        Config cfg(badPath);
    } catch (const std::exception &) {
        threw = true;
    }
    ok &= check(threw, "unknown pseudo_dt_mode throws");
    std::remove(badPath.c_str());
    return ok;
}

int main() {
    bool ok = true;
    struct Case {
        const char *name;
        bool (*fn)();
    };
    const Case cases[] = {
        {"testDefaults", testDefaults},
        {"testParseFull", testParseFull},
        {"testParsePartial", testParsePartial},
        {"testOcaThreadsValidation", testOcaThreadsValidation},
        {"testPseudoToleranceKeys", testPseudoToleranceKeys},
        {"testFluxTypeNames", testFluxTypeNames},
        {"testBcTypeNames", testBcTypeNames},
        {"testBcSideTables", testBcSideTables},
        {"testLogValidation", testLogValidation},
        {"testControllerKeys", testControllerKeys},
        {"testPseudoDtModeKey", testPseudoDtModeKey},
    };
    for (const auto &c : cases) {
        const bool r = c.fn();
        std::cout << "  [" << (r ? "PASS" : "FAIL") << "] " << c.name << "\n";
        ok &= r;
    }

    std::cout << (ok ? "ALL CONFIG TESTS PASSED\n" : "CONFIG TESTS FAILED\n");
    return ok ? 0 : 1;
}
