/// @file Config.cpp
/// @brief Implementation of the TOML configuration parser.
///
/// Each section of the TOML schema (config/default.toml) is optional: a
/// missing key falls back to the built-in default. This keeps the file
/// minimal while remaining self-documenting through the example file.

#include "Config.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <toml.hpp>

namespace
{
/// @brief Lower-case a string (for case-insensitive flux name matching).
std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

/// @brief Read a TOML value as \c int with a fallback default.
int getInt(const toml::table &tbl, const char *key, int fallback)
{
    if (const auto it = tbl.find(key); it != tbl.end())
    {
        return it->second.value_or(fallback);
    }
    return fallback;
}

/// @brief Read a TOML value as \c Real (double underlying) with a fallback.
Real getReal(const toml::table &tbl, const char *key, Real fallback)
{
    if (const auto it = tbl.find(key); it != tbl.end())
    {
        return it->second.value_or<double>(static_cast<double>(fallback));
    }
    return fallback;
}

/// @brief Read a TOML value as \c bool with a fallback.
bool getBool(const toml::table &tbl, const char *key, bool fallback)
{
    if (const auto it = tbl.find(key); it != tbl.end())
    {
        return it->second.value_or(fallback);
    }
    return fallback;
}

/// @brief Read a TOML value as a string with a fallback.
std::string getString(const toml::table &tbl, const char *key,
                      const std::string &fallback)
{
    if (const auto it = tbl.find(key); it != tbl.end())
    {
        return it->second.value_or(fallback);
    }
    return fallback;
}
} // namespace

// ============================================================================
// FluxType name mapping
// ============================================================================

FluxType parseFluxType(const std::string &name)
{
    const std::string key = toLower(name);
    if (key == "llf")
    {
        return FluxType::Llf;
    }
    if (key == "roe")
    {
        return FluxType::Roe;
    }
    if (key == "roe-e" || key == "roe_e")
    {
        return FluxType::RoeE;
    }
    if (key == "steger-warming" || key == "steger_warming")
    {
        return FluxType::StegerWarming;
    }
    if (key == "van-leer" || key == "van_leer")
    {
        return FluxType::VanLeer;
    }
    throw std::invalid_argument("parseFluxType: unknown flux name '" + name +
                                "'");
}

const char *fluxTypeName(FluxType t) noexcept
{
    switch (t)
    {
        case FluxType::Llf:
            return "llf";
        case FluxType::Roe:
            return "roe";
        case FluxType::RoeE:
            return "roe-e";
        case FluxType::StegerWarming:
            return "steger-warming";
        case FluxType::VanLeer:
            return "van-leer";
        default:
            return "unknown";
    }
}

// ============================================================================
// Config
// ============================================================================

void Config::setDefaults()
{
    polynomial_order_ = 2;
    quadrature_order_ = 4;

    mesh_nx_              = 8;
    mesh_ny_              = 8;
    mesh_x0_              = Real(0);
    mesh_y0_              = Real(0);
    mesh_dx_              = Real(1);
    mesh_dy_              = Real(1);
    mesh_shear_           = Real(0);
    mesh_split_triangles_ = false;

    flux_type_ = FluxType::Llf;
    gamma_     = Real(1.4);

    flow_rho_  = Real(1);
    flow_u_    = Real(0);
    flow_v_    = Real(0);
    flow_p_    = Real(1);
    flow_mach_ = Real(0.1);

    occa_mode_ = "Serial";
}

Config::Config()
{
    setDefaults();
}

Config::Config(const std::string &path)
{
    setDefaults();

    const toml::table root = toml::parse_file(path);

    // [basis]
    if (const auto it = root.find("basis"); it != root.end())
    {
        const auto &tbl   = it->second.as_table();
        polynomial_order_ = getInt(*tbl, "order", polynomial_order_);
        quadrature_order_ = getInt(*tbl, "nq", quadrature_order_);
    }

    // [mesh]
    if (const auto it = root.find("mesh"); it != root.end())
    {
        const auto &tbl = it->second.as_table();
        mesh_nx_        = getInt(*tbl, "nx", mesh_nx_);
        mesh_ny_        = getInt(*tbl, "ny", mesh_ny_);
        mesh_x0_        = getReal(*tbl, "x0", mesh_x0_);
        mesh_y0_        = getReal(*tbl, "y0", mesh_y0_);
        mesh_dx_        = getReal(*tbl, "dx", mesh_dx_);
        mesh_dy_        = getReal(*tbl, "dy", mesh_dy_);
        mesh_shear_     = getReal(*tbl, "shear", mesh_shear_);
        mesh_split_triangles_ =
            getBool(*tbl, "split_triangles", mesh_split_triangles_);
    }

    // [riemann]
    if (const auto it = root.find("riemann"); it != root.end())
    {
        const auto &tbl = it->second.as_table();
        if (const auto fluxIt = tbl->find("flux"); fluxIt != tbl->end())
        {
            const std::string name =
                fluxIt->second.value_or(std::string("llf"));
            flux_type_ = parseFluxType(name);
        }
    }

    // [gas]
    if (const auto it = root.find("gas"); it != root.end())
    {
        const auto &tbl = it->second.as_table();
        gamma_          = getReal(*tbl, "gamma", gamma_);
    }

    // [flow]
    if (const auto it = root.find("flow"); it != root.end())
    {
        const auto &tbl = it->second.as_table();
        flow_rho_       = getReal(*tbl, "rho", flow_rho_);
        flow_u_         = getReal(*tbl, "u", flow_u_);
        flow_v_         = getReal(*tbl, "v", flow_v_);
        flow_p_         = getReal(*tbl, "p", flow_p_);
        flow_mach_      = getReal(*tbl, "mach", flow_mach_);
    }

    // [occa]
    if (const auto it = root.find("occa"); it != root.end())
    {
        const auto &tbl = it->second.as_table();
        occa_mode_      = getString(*tbl, "mode", occa_mode_);
    }
}
