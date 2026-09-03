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
// BcType / TimeMethod / IcType name mappings
// ============================================================================

BcType parseBcType(const std::string &name)
{
    const std::string key = toLower(name);
    if (key == "farfield" || key == "far-field" || key == "far_field")
    {
        return BcType::Farfield;
    }
    if (key == "periodic")
    {
        return BcType::Periodic;
    }
    throw std::invalid_argument("parseBcType: unknown bc name '" + name + "'");
}

const char *bcTypeName(BcType t) noexcept
{
    switch (t)
    {
        case BcType::Farfield:
            return "farfield";
        case BcType::Periodic:
            return "periodic";
        default:
            return "unknown";
    }
}

TimeMethod parseTimeMethod(const std::string &name)
{
    const std::string key = toLower(name);
    if (key == "euler")
    {
        return TimeMethod::Euler;
    }
    if (key == "ssprk3")
    {
        return TimeMethod::SspRk3;
    }
    if (key == "rk32")
    {
        return TimeMethod::Rk32;
    }
    if (key == "rk54")
    {
        return TimeMethod::Rk54;
    }
    if (key == "ssprk221")
    {
        return TimeMethod::SspRk221;
    }
    if (key == "ssprk321")
    {
        return TimeMethod::SspRk321;
    }
    if (key == "ssprk332")
    {
        return TimeMethod::SspRk332;
    }
    if (key == "ssprk432")
    {
        return TimeMethod::SspRk432;
    }
    if (key == "be" || key == "backward-euler" || key == "backward_euler")
    {
        return TimeMethod::BackwardEuler;
    }
    if (key == "ditr-u2r2" || key == "ditr_u2r2")
    {
        return TimeMethod::DitrU2R2;
    }
    if (key == "ditr-u2r1" || key == "ditr_u2r1")
    {
        return TimeMethod::DitrU2R1;
    }
    if (key == "ditr-u3r1" || key == "ditr_u3r1")
    {
        return TimeMethod::DitrU3R1;
    }
    throw std::invalid_argument("parseTimeMethod: unknown method name '" +
                                name + "'");
}

const char *timeMethodName(TimeMethod t) noexcept
{
    switch (t)
    {
        case TimeMethod::Euler:
            return "euler";
        case TimeMethod::SspRk3:
            return "ssprk3";
        case TimeMethod::Rk32:
            return "rk32";
        case TimeMethod::Rk54:
            return "rk54";
        case TimeMethod::SspRk221:
            return "ssprk221";
        case TimeMethod::SspRk321:
            return "ssprk321";
        case TimeMethod::SspRk332:
            return "ssprk332";
        case TimeMethod::SspRk432:
            return "ssprk432";
        case TimeMethod::BackwardEuler:
            return "be";
        case TimeMethod::DitrU2R2:
            return "ditr_u2r2";
        case TimeMethod::DitrU2R1:
            return "ditr_u2r1";
        case TimeMethod::DitrU3R1:
            return "ditr_u3r1";
        default:
            return "unknown";
    }
}

IcType parseIcType(const std::string &name)
{
    const std::string key = toLower(name);
    if (key == "uniform")
    {
        return IcType::Uniform;
    }
    if (key == "expr" || key == "expression")
    {
        return IcType::Expr;
    }
    throw std::invalid_argument("parseIcType: unknown ic name '" + name + "'");
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

    bc_type_ = BcType::Farfield;

    time_method_           = TimeMethod::SspRk3;
    time_cfl_              = Real(0.2);
    time_dt_               = Real(0);
    time_final_            = Real(1);
    time_rtol_             = Real(1e-6);
    time_atol_             = Real(1e-6);
    time_max_pseudo_steps_ = 100;
    time_pseudo_method_    = TimeMethod::SspRk332;
    time_pseudo_dt_        = Real(0);
    time_pseudo_single_dt_ = false;
    time_pseudo_reject_    = false;
    time_dual_decoupled_   = false;

    ic_type_ = IcType::Uniform;
    ic_rho_  = "1";
    ic_u_    = "0";
    ic_v_    = "0";
    ic_p_    = "1";
    ic_symbols_.clear();
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
        occa_threads_   = getInt(*tbl, "threads", occa_threads_);
        occa_platform_  = getInt(*tbl, "platform", occa_platform_);
        occa_device_    = getInt(*tbl, "device", occa_device_);
    }

    // [bc]
    if (const auto it = root.find("bc"); it != root.end())
    {
        const auto &tbl = it->second.as_table();
        if (const auto bcIt = tbl->find("type"); bcIt != tbl->end())
        {
            const std::string name =
                bcIt->second.value_or(std::string("farfield"));
            bc_type_ = parseBcType(name);
        }
    }

    // [time_marching]
    if (const auto it = root.find("time_marching"); it != root.end())
    {
        const auto &tbl = it->second.as_table();
        if (const auto mIt = tbl->find("method"); mIt != tbl->end())
        {
            const std::string name =
                mIt->second.value_or(std::string("ssprk3"));
            time_method_ = parseTimeMethod(name);
        }
        time_cfl_   = getReal(*tbl, "cfl", time_cfl_);
        time_dt_    = getReal(*tbl, "dt", time_dt_);
        time_final_ = getReal(*tbl, "t_final", time_final_);
        time_rtol_  = getReal(*tbl, "rtol", time_rtol_);
        time_atol_  = getReal(*tbl, "atol", time_atol_);
        time_max_pseudo_steps_ =
            getInt(*tbl, "max_pseudo_steps", time_max_pseudo_steps_);
        if (const auto pIt = tbl->find("pseudo_method"); pIt != tbl->end())
        {
            const std::string name =
                pIt->second.value_or(std::string("ssprk332"));
            time_pseudo_method_ = parseTimeMethod(name);
        }
        time_pseudo_dt_ = getReal(*tbl, "pseudo_dt", time_pseudo_dt_);
        time_pseudo_single_dt_ =
            getBool(*tbl, "pseudo_single_dt", time_pseudo_single_dt_);
        time_pseudo_reject_ =
            getBool(*tbl, "pseudo_reject", time_pseudo_reject_);
        time_dual_decoupled_ =
            getBool(*tbl, "dual_decoupled", time_dual_decoupled_);
    }

    // [initial_condition]
    if (const auto it = root.find("initial_condition"); it != root.end())
    {
        const auto &tbl = it->second.as_table();
        if (const auto tIt = tbl->find("type"); tIt != tbl->end())
        {
            const std::string name =
                tIt->second.value_or(std::string("uniform"));
            ic_type_ = parseIcType(name);
        }
        ic_rho_ = getString(*tbl, "rho", ic_rho_);
        ic_u_   = getString(*tbl, "u", ic_u_);
        ic_v_   = getString(*tbl, "v", ic_v_);
        ic_p_   = getString(*tbl, "p", ic_p_);
        // [initial_condition.symbols]: numeric constants available in the
        // expressions (e.g. beta = 5.0).
        if (const auto sIt = tbl->find("symbols"); sIt != tbl->end())
        {
            if (const auto symTbl = sIt->second.as_table())
            {
                for (const auto &[key, val] : *symTbl)
                {
                    if (const auto num = val.value<double>())
                    {
                        ic_symbols_[std::string(key.str())] = Real(*num);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Config: [initial_condition.symbols] entry '" +
                            std::string(key.str()) + "' must be a number");
                    }
                }
            }
        }
    }

    // Validation of the time-marching parameters.
    if (time_final_ <= Real(0))
    {
        throw std::invalid_argument("Config: t_final must be > 0");
    }
    if (time_cfl_ <= Real(0))
    {
        throw std::invalid_argument("Config: cfl must be > 0");
    }
    if (time_dt_ < Real(0))
    {
        throw std::invalid_argument("Config: dt must be >= 0");
    }
    if (time_rtol_ <= Real(0) || time_atol_ <= Real(0))
    {
        throw std::invalid_argument("Config: rtol and atol must be > 0");
    }
    if (time_max_pseudo_steps_ <= 0)
    {
        throw std::invalid_argument("Config: max_pseudo_steps must be > 0");
    }
    if (occa_threads_ < 0)
    {
        throw std::invalid_argument("Config: threads must be >= 0");
    }
    if (occa_platform_ < 0 || occa_device_ < 0)
    {
        throw std::invalid_argument("Config: platform and device must be >= 0");
    }
}
