/// @file test_config.cpp
/// @brief Tests for the Config class: default construction and TOML parsing.

#include <cstdio>
#include <iostream>
#include <string>

#include "config/Config.hpp"

namespace
{
/// @brief Write \p content to a temporary file and return its path.
/// The file is created next to the test binary so ctest's working directory
/// (the build tree) always contains it.
std::string writeTempToml(const std::string &content)
{
    static const char *kPath = "test_config_scratch.toml";
    std::FILE *f             = std::fopen(kPath, "w");
    if (!f)
    {
        return "";
    }
    std::fputs(content.c_str(), f);
    std::fclose(f);
    return kPath;
}

bool check(bool cond, const char *what)
{
    if (!cond)
    {
        std::cout << "  FAIL " << what << "\n";
    }
    return cond;
}

bool checkInt(int value, int ref, const char *what)
{
    if (value != ref)
    {
        std::cout << "  FAIL " << what << ": value=" << value << " ref=" << ref
                  << "\n";
        return false;
    }
    return true;
}

bool checkReal(Real value, Real ref, const char *what)
{
    const Real tol = Real(1e3) * RealEpsilon;
    if (std::abs(value - ref) > tol * std::max(std::abs(ref), RealEpsilon))
    {
        std::cout << "  FAIL " << what << ": value=" << value << " ref=" << ref
                  << "\n";
        return false;
    }
    return true;
}

bool checkStr(const std::string &value, const std::string &ref,
              const char *what)
{
    if (value != ref)
    {
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

static bool testDefaults()
{
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
    ok &= check(cfg.fluxType() == FluxType::Llf, "fluxType == Llf");
    ok &= checkInt(cfg.fluxTypeInt(), 0, "fluxTypeInt");
    return ok;
}

// ---------------------------------------------------------------------------
// 2. TOML parsing: full file
// ---------------------------------------------------------------------------

static bool testParseFull()
{
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
)";
    const std::string path    = writeTempToml(content);
    if (path.empty())
    {
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

    std::remove(path.c_str());
    return ok;
}

// ---------------------------------------------------------------------------
// 3. Partial file: missing sections fall back to defaults
// ---------------------------------------------------------------------------

static bool testParsePartial()
{
    std::cout << "Test 3: partial TOML file (fallback defaults)\n";
    const std::string content = R"(
[basis]
order = 1

[riemann]
flux = "llf"
)";
    const std::string path    = writeTempToml(content);
    if (path.empty())
    {
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

static bool testOcaThreadsValidation()
{
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
    try
    {
        Config cfg(badPath);
    }
    catch (const std::exception &)
    {
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
    try
    {
        Config cfg(goodPath);
        ok &= checkInt(cfg.occaThreads(), 0, "threads = 0 accepted");
    }
    catch (const std::exception &e)
    {
        std::cout << "  FAIL threads = 0 threw: " << e.what() << "\n";
        ok = false;
    }
    std::remove(goodPath.c_str());
    return ok;
}

// ---------------------------------------------------------------------------
// 5. FluxType name mapping
// ---------------------------------------------------------------------------

static bool testFluxTypeNames()
{
    std::cout << "Test 5: flux type names\n";
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
    try
    {
        (void)parseFluxType("hllc");
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    ok &= check(threw, "unknown name throws");
    return ok;
}

int main()
{
    bool ok = true;
    struct Case
    {
        const char *name;
        bool (*fn)();
    };
    const Case cases[] = {
        {"testDefaults", testDefaults},
        {"testParseFull", testParseFull},
        {"testParsePartial", testParsePartial},
        {"testOcaThreadsValidation", testOcaThreadsValidation},
        {"testFluxTypeNames", testFluxTypeNames},
    };
    for (const auto &c : cases)
    {
        const bool r = c.fn();
        std::cout << "  [" << (r ? "PASS" : "FAIL") << "] " << c.name << "\n";
        ok &= r;
    }

    std::cout << (ok ? "ALL CONFIG TESTS PASSED\n" : "CONFIG TESTS FAILED\n");
    return ok ? 0 : 1;
}
