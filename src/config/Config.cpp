/// @file Config.cpp
/// @brief Implementation of the TOML configuration parser.
///
/// Each section of the TOML schema (config/default.toml) is optional: a
/// missing key falls back to the built-in default. This keeps the file
/// minimal while remaining self-documenting through the example file.

#include "Config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <toml.hpp>

namespace {
/// @brief Lower-case a string (for case-insensitive flux name matching).
std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

/// @brief Read a TOML value as \c int with a fallback default.
int getInt(const toml::table &tbl, const char *key, int fallback) {
    if (const auto it = tbl.find(key); it != tbl.end()) {
        return it->second.value_or(fallback);
    }
    return fallback;
}

/// @brief Read a TOML value as \c Real (double underlying) with a fallback.
Real getReal(const toml::table &tbl, const char *key, Real fallback) {
    if (const auto it = tbl.find(key); it != tbl.end()) {
        return it->second.value_or<double>(static_cast<double>(fallback));
    }
    return fallback;
}

/// @brief Read a TOML value as \c bool with a fallback.
bool getBool(const toml::table &tbl, const char *key, bool fallback) {
    if (const auto it = tbl.find(key); it != tbl.end()) {
        return it->second.value_or(fallback);
    }
    return fallback;
}

/// @brief Read a TOML value as a string with a fallback.
std::string getString(const toml::table &tbl, const char *key,
                      const std::string &fallback) {
    if (const auto it = tbl.find(key); it != tbl.end()) {
        return it->second.value_or(fallback);
    }
    return fallback;
}
} // namespace

// ============================================================================
// FluxType name mapping
// ============================================================================

FluxType parseFluxType(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "llf") {
        return FluxType::Llf;
    }
    if (key == "roe") {
        return FluxType::Roe;
    }
    if (key == "roe-e" || key == "roe_e") {
        return FluxType::RoeE;
    }
    if (key == "steger-warming" || key == "steger_warming") {
        return FluxType::StegerWarming;
    }
    if (key == "van-leer" || key == "van_leer") {
        return FluxType::VanLeer;
    }
    throw std::invalid_argument("parseFluxType: unknown flux name '" + name +
                                "'");
}

const char *fluxTypeName(FluxType t) noexcept {
    switch (t) {
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

BcType parseBcType(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "farfield" || key == "far-field" || key == "far_field") {
        return BcType::Farfield;
    }
    if (key == "periodic") {
        return BcType::Periodic;
    }
    if (key == "free" || key == "zero-gradient" || key == "zero_gradient") {
        return BcType::Free;
    }
    if (key == "free-avg" || key == "free_avg" || key == "freeavg" ||
        key == "free-average") {
        return BcType::FreeAvg;
    }
    if (key == "riemann" || key == "characteristic" ||
        key == "pressure-farfield" || key == "pressure_farfield") {
        return BcType::Riemann;
    }
    if (key == "slip" || key == "slip-wall" || key == "slip_wall" ||
        key == "wall") {
        return BcType::Slip;
    }
    if (key == "expr" || key == "dirichlet") {
        return BcType::Expr;
    }
    throw std::invalid_argument("parseBcType: unknown bc name '" + name + "'");
}

const char *bcTypeName(BcType t) noexcept {
    switch (t) {
        case BcType::Farfield:
            return "farfield";
        case BcType::Periodic:
            return "periodic";
        case BcType::Free:
            return "free";
        case BcType::FreeAvg:
            return "free-avg";
        case BcType::Riemann:
            return "riemann";
        case BcType::Slip:
            return "slip";
        case BcType::Expr:
            return "expr";
        default:
            return "unknown";
    }
}

RiemannTarget parseRiemannTarget(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "flow") {
        return RiemannTarget::Flow;
    }
    if (key == "ic") {
        return RiemannTarget::Ic;
    }
    if (key == "expr" || key == "analytic") {
        return RiemannTarget::Expr;
    }
    throw std::invalid_argument("parseRiemannTarget: unknown target '" + name +
                                "' (expected \"flow\", \"ic\" or \"expr\")");
}

const char *riemannTargetName(RiemannTarget t) noexcept {
    switch (t) {
        case RiemannTarget::Flow:
            return "flow";
        case RiemannTarget::Ic:
            return "ic";
        case RiemannTarget::Expr:
            return "expr";
        default:
            return "unknown";
    }
}

TimeMethod parseTimeMethod(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "euler") {
        return TimeMethod::Euler;
    }
    if (key == "ssprk3") {
        return TimeMethod::SspRk3;
    }
    if (key == "rk32") {
        return TimeMethod::Rk32;
    }
    if (key == "rk54") {
        return TimeMethod::Rk54;
    }
    if (key == "ssprk221") {
        return TimeMethod::SspRk221;
    }
    if (key == "ssprk321") {
        return TimeMethod::SspRk321;
    }
    if (key == "ssprk332") {
        return TimeMethod::SspRk332;
    }
    if (key == "ssprk432") {
        return TimeMethod::SspRk432;
    }
    if (key == "be" || key == "backward-euler" || key == "backward_euler") {
        return TimeMethod::BackwardEuler;
    }
    if (key == "ditr-u2r2" || key == "ditr_u2r2") {
        return TimeMethod::DitrU2R2;
    }
    if (key == "ditr-u2r1" || key == "ditr_u2r1") {
        return TimeMethod::DitrU2R1;
    }
    if (key == "ditr-u3r1" || key == "ditr_u3r1") {
        return TimeMethod::DitrU3R1;
    }
    throw std::invalid_argument("parseTimeMethod: unknown method name '" +
                                name + "'");
}

const char *timeMethodName(TimeMethod t) noexcept {
    switch (t) {
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

PseudoDtMode parsePseudoDtMode(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "global") {
        return PseudoDtMode::Global;
    }
    if (key == "local") {
        return PseudoDtMode::Local;
    }
    throw std::invalid_argument("parsePseudoDtMode: unknown mode '" + name +
                                "' (expected \"global\" or \"local\")");
}

PseudoNorm parsePseudoNorm(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "inf" || key == "max" || key == "infinity") {
        return PseudoNorm::Inf;
    }
    if (key == "l2" || key == "rms") {
        return PseudoNorm::L2;
    }
    if (key == "mean" || key == "l1" || key == "avg" || key == "average") {
        return PseudoNorm::Mean;
    }
    throw std::invalid_argument("parsePseudoNorm: unknown norm '" + name +
                                "' (expected \"inf\", \"l2\" or \"mean\")");
}

LimiterType parseLimiterType(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "none") {
        return LimiterType::None;
    }
    if (key == "mr-weno" || key == "mr_weno" || key == "mrweno") {
        return LimiterType::MrWeno;
    }
    throw std::invalid_argument("parseLimiterType: unknown limiter name '" +
                                name + "'");
}

const char *limiterTypeName(LimiterType t) noexcept {
    switch (t) {
        case LimiterType::None:
            return "none";
        case LimiterType::MrWeno:
            return "mr-weno";
        default:
            return "unknown";
    }
}

ShockNorm parseShockNorm(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "min") {
        return ShockNorm::Min;
    }
    if (key == "max") {
        return ShockNorm::Max;
    }
    if (key == "mean") {
        return ShockNorm::Mean;
    }
    throw std::invalid_argument("parseShockNorm: unknown norm name '" + name +
                                "'");
}

const char *shockNormName(ShockNorm t) noexcept {
    switch (t) {
        case ShockNorm::Min:
            return "min";
        case ShockNorm::Max:
            return "max";
        case ShockNorm::Mean:
            return "mean";
        default:
            return "unknown";
    }
}

WenoDriveMode parseWenoDriveMode(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "auto") {
        return WenoDriveMode::Auto;
    }
    if (key == "list") {
        return WenoDriveMode::List;
    }
    if (key == "full") {
        return WenoDriveMode::Full;
    }
    throw std::invalid_argument("parseWenoDriveMode: unknown drive mode '" +
                                name + "'");
}

const char *wenoDriveModeName(WenoDriveMode t) noexcept {
    switch (t) {
        case WenoDriveMode::Auto:
            return "auto";
        case WenoDriveMode::List:
            return "list";
        case WenoDriveMode::Full:
            return "full";
        default:
            return "unknown";
    }
}

IcType parseIcType(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "uniform") {
        return IcType::Uniform;
    }
    if (key == "expr" || key == "expression") {
        return IcType::Expr;
    }
    throw std::invalid_argument("parseIcType: unknown ic name '" + name + "'");
}

OutputStrategy parseOutputStrategy(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "direct") {
        return OutputStrategy::Direct;
    }
    if (key == "checkpoint") {
        return OutputStrategy::Checkpoint;
    }
    throw std::invalid_argument("parseOutputStrategy: unknown strategy name '" +
                                name + "'");
}

const char *outputStrategyName(OutputStrategy t) noexcept {
    switch (t) {
        case OutputStrategy::Direct:
            return "direct";
        case OutputStrategy::Checkpoint:
            return "checkpoint";
        default:
            return "unknown";
    }
}

OutputFormat parseOutputFormat(const std::string &name) {
    const std::string key = toLower(name);
    if (key == "hdf5") {
        return OutputFormat::Hdf5;
    }
    if (key == "xdmf") {
        return OutputFormat::Xdmf;
    }
    if (key == "tecplot" || key == "plt") {
        return OutputFormat::Tecplot;
    }
    if (key == "vtu") {
        return OutputFormat::Vtu;
    }
    if (key == "all") {
        return OutputFormat::All;
    }
    throw std::invalid_argument("parseOutputFormat: unknown format name '" +
                                name + "'");
}

const char *outputFormatName(OutputFormat t) noexcept {
    switch (t) {
        case OutputFormat::Hdf5:
            return "hdf5";
        case OutputFormat::Xdmf:
            return "xdmf";
        case OutputFormat::Tecplot:
            return "tecplot";
        case OutputFormat::All:
            return "all";
        case OutputFormat::Vtu:
            return "vtu";
        default:
            return "unknown";
    }
}

// ============================================================================
// Config
// ============================================================================

void Config::setDefaults() {
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
    for (auto &side : bc_side_) {
        side = SideBc{};
    }
    bc_rho_.clear();
    bc_u_.clear();
    bc_v_.clear();
    bc_p_.clear();
    riemann_target_ = RiemannTarget::Flow;

    time_method_                = TimeMethod::SspRk3;
    time_cfl_                   = Real(0.2);
    time_dt_                    = Real(0);
    time_final_                 = Real(1);
    time_rtol_                  = Real(1e-6);
    time_atol_                  = Real(1e-6);
    time_safety_                = Real(0.9);
    time_min_factor_            = Real(0.2);
    time_max_factor_            = Real(5);
    time_max_growth_            = Real(10);
    time_max_pseudo_steps_      = 100;
    time_pseudo_method_         = TimeMethod::SspRk332;
    time_pseudo_dt_             = Real(0);
    time_pseudo_dt_mode_        = PseudoDtMode::Global;
    time_pseudo_norm_           = PseudoNorm::Inf;
    time_pseudo_stagnation_     = false;
    time_pseudo_stagnation_tol_ = Real(0.98);
    time_pseudo_reject_         = false;
    time_dual_decoupled_        = false;

    limiter_type_               = LimiterType::None;
    limiter_ckc_                = Real(1);
    limiter_shock_norm_         = ShockNorm::Min;
    limiter_gamma_low_          = Real(0.01);
    limiter_gamma_high_         = Real(0.99);
    limiter_epsilon_            = Real(1e-6);
    limiter_force_troubled_     = false;
    limiter_weno_drive_         = WenoDriveMode::Auto;
    limiter_positivity_         = false;
    limiter_positivity_epsilon_ = Real(1e-10);

    ic_type_ = IcType::Uniform;
    ic_rho_  = "1";
    ic_u_    = "0";
    ic_v_    = "0";
    ic_p_    = "1";
    ic_symbols_.clear();

    output_enable_   = false;
    output_strategy_ = OutputStrategy::Direct;
    output_format_   = OutputFormat::Hdf5;
    output_series_   = true;
    output_archive_  = true;
    output_prefix_.clear();
    output_interval_  = 100;
    output_directory_ = "./results";

    log_level_    = 1;
    log_interval_ = 100;
}

Config::Config() {
    setDefaults();
}

Config::Config(const std::string &path) {
    setDefaults();

    // Output file-name prefix: the configuration file stem, so several
    // configs can share one output directory
    // (<stem>_solution_<step>.<ext>, <stem>_mesh.h5, ...).
    output_prefix_ = std::filesystem::path(path).stem().string();

    const toml::table root = toml::parse_file(path);

    // [basis]
    if (const auto it = root.find("basis"); it != root.end()) {
        const auto &tbl   = it->second.as_table();
        polynomial_order_ = getInt(*tbl, "order", polynomial_order_);
        quadrature_order_ = getInt(*tbl, "nq", quadrature_order_);
    }

    // [mesh]
    if (const auto it = root.find("mesh"); it != root.end()) {
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
    if (const auto it = root.find("riemann"); it != root.end()) {
        const auto &tbl = it->second.as_table();
        if (const auto fluxIt = tbl->find("flux"); fluxIt != tbl->end()) {
            const std::string name =
                fluxIt->second.value_or(std::string("llf"));
            flux_type_ = parseFluxType(name);
        }
    }

    // [gas]
    if (const auto it = root.find("gas"); it != root.end()) {
        const auto &tbl = it->second.as_table();
        gamma_          = getReal(*tbl, "gamma", gamma_);
    }

    // [flow]
    if (const auto it = root.find("flow"); it != root.end()) {
        const auto &tbl = it->second.as_table();
        flow_rho_       = getReal(*tbl, "rho", flow_rho_);
        flow_u_         = getReal(*tbl, "u", flow_u_);
        flow_v_         = getReal(*tbl, "v", flow_v_);
        flow_p_         = getReal(*tbl, "p", flow_p_);
        flow_mach_      = getReal(*tbl, "mach", flow_mach_);
    }

    // [occa]
    if (const auto it = root.find("occa"); it != root.end()) {
        const auto &tbl = it->second.as_table();
        occa_mode_      = getString(*tbl, "mode", occa_mode_);
        occa_threads_   = getInt(*tbl, "threads", occa_threads_);
        occa_platform_  = getInt(*tbl, "platform", occa_platform_);
        occa_device_    = getInt(*tbl, "device", occa_device_);
        occa_opencl_o3_ = getBool(*tbl, "opencl_o3", occa_opencl_o3_);
    }

    // [bc]
    if (const auto it = root.find("bc"); it != root.end()) {
        const auto &tbl = it->second.as_table();
        if (const auto bcIt = tbl->find("type"); bcIt != tbl->end()) {
            const std::string name =
                bcIt->second.value_or(std::string("farfield"));
            bc_type_ = parseBcType(name);
        }
        // Spelling always validated; only consulted when type = "riemann".
        if (const auto tIt = tbl->find("riemann_target"); tIt != tbl->end()) {
            const std::string name = tIt->second.value_or(std::string("flow"));
            riemann_target_        = parseRiemannTarget(name);
        }
        // Per-side override tables. Periodic is rejected: periodic pairing
        // is global (mesh-level face gluing), not per-side.
        static const char *sideKeys[4] = {"x_low", "x_high", "y_low", "y_high"};
        for (int s = 0; s < 4; ++s) {
            const auto sIt = tbl->find(sideKeys[s]);
            if (sIt == tbl->end()) {
                continue;
            }
            const auto &st = sIt->second.as_table();
            SideBc &side   = bc_side_[s];
            if (const auto tIt2 = st->find("type"); tIt2 != st->end()) {
                side.type =
                    parseBcType(tIt2->second.value_or(std::string("free")));
                if (side.type == BcType::Periodic) {
                    throw std::invalid_argument(
                        std::string("Config [bc.") + sideKeys[s] +
                        "]: per-side 'periodic' is not supported (periodic "
                        "pairing is configured globally)");
                }
                side.hasType = true;
            }
            if (const auto xIt = st->find("split_x"); xIt != st->end()) {
                side.splitX   = xIt->second.value_or(Real(0));
                side.hasSplit = true;
                if (const auto dIt = st->find("split_type"); dIt != st->end()) {
                    side.splitType =
                        parseBcType(dIt->second.value_or(std::string("slip")));
                } else {
                    side.splitType = BcType::Slip;
                }
                if (side.splitType == BcType::Periodic) {
                    throw std::invalid_argument(
                        std::string("Config [bc.") + sideKeys[s] +
                        "]: split_type 'periodic' is not supported");
                }
            }
        }
        // Dirichlet expressions for "expr" faces (x, y, t available).
        bc_rho_ = getString(*tbl, "rho", bc_rho_);
        bc_u_   = getString(*tbl, "u", bc_u_);
        bc_v_   = getString(*tbl, "v", bc_v_);
        bc_p_   = getString(*tbl, "p", bc_p_);
    }

    // [time_marching]
    if (const auto it = root.find("time_marching"); it != root.end()) {
        const auto &tbl = it->second.as_table();
        if (const auto mIt = tbl->find("method"); mIt != tbl->end()) {
            const std::string name =
                mIt->second.value_or(std::string("ssprk3"));
            time_method_ = parseTimeMethod(name);
        }
        time_cfl_        = getReal(*tbl, "cfl", time_cfl_);
        time_dt_         = getReal(*tbl, "dt", time_dt_);
        time_final_      = getReal(*tbl, "t_final", time_final_);
        time_rtol_       = getReal(*tbl, "rtol", time_rtol_);
        time_atol_       = getReal(*tbl, "atol", time_atol_);
        time_safety_     = getReal(*tbl, "safety", time_safety_);
        time_min_factor_ = getReal(*tbl, "min_factor", time_min_factor_);
        time_max_factor_ = getReal(*tbl, "max_factor", time_max_factor_);
        time_max_growth_ = getReal(*tbl, "max_growth", time_max_growth_);
        time_max_pseudo_steps_ =
            getInt(*tbl, "max_pseudo_steps", time_max_pseudo_steps_);
        if (const auto pIt = tbl->find("pseudo_method"); pIt != tbl->end()) {
            const std::string name =
                pIt->second.value_or(std::string("ssprk332"));
            time_pseudo_method_ = parseTimeMethod(name);
        }
        time_pseudo_dt_   = getReal(*tbl, "pseudo_dt", time_pseudo_dt_);
        time_pseudo_rtol_ = getReal(*tbl, "pseudo_rtol", time_pseudo_rtol_);
        time_pseudo_atol_ = getReal(*tbl, "pseudo_atol", time_pseudo_atol_);
        time_pseudo_stagnation_ =
            getBool(*tbl, "pseudo_stagnation", time_pseudo_stagnation_);
        time_pseudo_stagnation_tol_ =
            getReal(*tbl, "pseudo_stagnation_tol", time_pseudo_stagnation_tol_);
        if (const auto nIt = tbl->find("pseudo_norm"); nIt != tbl->end()) {
            const std::string nname = nIt->second.value_or(std::string("inf"));
            time_pseudo_norm_       = parsePseudoNorm(nname);
        }
        if (const auto mIt = tbl->find("pseudo_dt_mode"); mIt != tbl->end()) {
            const std::string name =
                mIt->second.value_or(std::string("global"));
            time_pseudo_dt_mode_ = parsePseudoDtMode(name);
        }
        time_pseudo_reject_ =
            getBool(*tbl, "pseudo_reject", time_pseudo_reject_);
        time_dual_decoupled_ =
            getBool(*tbl, "dual_decoupled", time_dual_decoupled_);
    }

    // [limiter]
    if (const auto it = root.find("limiter"); it != root.end()) {
        const auto &tbl = it->second.as_table();
        if (const auto tIt = tbl->find("type"); tIt != tbl->end()) {
            const std::string name = tIt->second.value_or(std::string("none"));
            limiter_type_          = parseLimiterType(name);
        }
        limiter_ckc_ = getReal(*tbl, "ckc", limiter_ckc_);
        if (const auto nIt = tbl->find("shock_norm"); nIt != tbl->end()) {
            const std::string name = nIt->second.value_or(std::string("min"));
            limiter_shock_norm_    = parseShockNorm(name);
        }
        limiter_gamma_low_  = getReal(*tbl, "gamma_low", limiter_gamma_low_);
        limiter_gamma_high_ = getReal(*tbl, "gamma_high", limiter_gamma_high_);
        limiter_epsilon_    = getReal(*tbl, "epsilon", limiter_epsilon_);
        limiter_force_troubled_ =
            getBool(*tbl, "force_troubled", limiter_force_troubled_);
        if (const auto dIt = tbl->find("weno_drive_mode"); dIt != tbl->end()) {
            const std::string name = dIt->second.value_or(std::string("auto"));
            limiter_weno_drive_    = parseWenoDriveMode(name);
        }
        limiter_positivity_ = getBool(*tbl, "positivity", limiter_positivity_);
        limiter_positivity_epsilon_ =
            getReal(*tbl, "positivity_epsilon", limiter_positivity_epsilon_);
    }

    // [initial_condition]
    if (const auto it = root.find("initial_condition"); it != root.end()) {
        const auto &tbl = it->second.as_table();
        if (const auto tIt = tbl->find("type"); tIt != tbl->end()) {
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
        if (const auto sIt = tbl->find("symbols"); sIt != tbl->end()) {
            if (const auto symTbl = sIt->second.as_table()) {
                for (const auto &[key, val] : *symTbl) {
                    if (const auto num = val.value<double>()) {
                        ic_symbols_[std::string(key.str())] = Real(*num);
                    } else {
                        throw std::invalid_argument(
                            "Config: [initial_condition.symbols] entry '" +
                            std::string(key.str()) + "' must be a number");
                    }
                }
            }
        }
    }

    // [output]
    if (const auto it = root.find("output"); it != root.end()) {
        const auto &tbl = it->second.as_table();
        output_enable_  = getBool(*tbl, "enable", output_enable_);
        if (const auto sIt = tbl->find("strategy"); sIt != tbl->end()) {
            const std::string name =
                sIt->second.value_or(std::string("direct"));
            output_strategy_ = parseOutputStrategy(name);
        }
        if (const auto fIt = tbl->find("format"); fIt != tbl->end()) {
            const std::string name = fIt->second.value_or(std::string("hdf5"));
            output_format_         = parseOutputFormat(name);
        }
        output_series_    = getBool(*tbl, "series", output_series_);
        output_archive_   = getBool(*tbl, "archive", output_archive_);
        output_interval_  = getInt(*tbl, "interval", output_interval_);
        output_directory_ = getString(*tbl, "directory", output_directory_);
    }

    // [log]
    if (const auto it = root.find("log"); it != root.end()) {
        const auto &tbl = it->second.as_table();
        log_level_      = getInt(*tbl, "level", log_level_);
        log_interval_   = getInt(*tbl, "interval", log_interval_);
    }

    // Validation of the time-marching parameters.
    if (time_final_ <= Real(0)) {
        throw std::invalid_argument("Config: t_final must be > 0");
    }
    if (time_cfl_ <= Real(0)) {
        throw std::invalid_argument("Config: cfl must be > 0");
    }
    if (time_dt_ < Real(0)) {
        throw std::invalid_argument("Config: dt must be >= 0");
    }
    if (time_rtol_ <= Real(0) || time_atol_ <= Real(0)) {
        throw std::invalid_argument("Config: rtol and atol must be > 0");
    }
    if (time_safety_ <= Real(0)) {
        throw std::invalid_argument("Config: safety must be > 0");
    }
    if (time_min_factor_ <= Real(0) || time_max_factor_ < time_min_factor_) {
        throw std::invalid_argument(
            "Config: min_factor must be > 0 and max_factor >= min_factor");
    }
    if (time_max_growth_ <= Real(0)) {
        throw std::invalid_argument("Config: max_growth must be > 0");
    }
    if (time_max_pseudo_steps_ <= 0) {
        throw std::invalid_argument("Config: max_pseudo_steps must be > 0");
    }
    if (time_pseudo_rtol_ < Real(0) || time_pseudo_atol_ < Real(0)) {
        throw std::invalid_argument(
            "Config: pseudo_rtol and pseudo_atol must be >= 0 (0 = auto)");
    }
    if (limiter_ckc_ <= Real(0)) {
        throw std::invalid_argument("Config: limiter ckc must be > 0");
    }
    if (limiter_gamma_low_ <= Real(0) || limiter_gamma_high_ <= Real(0)) {
        throw std::invalid_argument(
            "Config: limiter gamma_low and gamma_high must be > 0");
    }
    if (std::abs(limiter_gamma_low_ + limiter_gamma_high_ - Real(1)) >
        Real(1e-12)) {
        throw std::invalid_argument(
            "Config: limiter gamma_low + gamma_high must equal 1 (the "
            "telescoping identity gamma_low*p_low + gamma_high*p_high = q "
            "requires it)");
    }
    if (limiter_epsilon_ <= Real(0)) {
        throw std::invalid_argument("Config: limiter epsilon must be > 0");
    }
    if (limiter_positivity_epsilon_ <= Real(0) ||
        limiter_positivity_epsilon_ >= Real(1)) {
        throw std::invalid_argument(
            "Config: limiter positivity_epsilon must be in (0, 1)");
    }
    if (occa_threads_ < 0) {
        throw std::invalid_argument("Config: threads must be >= 0");
    }
    if (occa_platform_ < 0 || occa_device_ < 0) {
        throw std::invalid_argument("Config: platform and device must be >= 0");
    }
    if (output_enable_ && output_interval_ < 1) {
        throw std::invalid_argument(
            "Config: output.interval must be >= 1 when output is enabled");
    }
    if (log_level_ < 0) {
        throw std::invalid_argument("Config: log.level must be >= 0");
    }
    if (log_interval_ < 1) {
        throw std::invalid_argument("Config: log.interval must be >= 1");
    }
}
