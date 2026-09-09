/// @file Config.hpp
/// @brief TOML configuration file parser and parameter store.
///
/// Reads a TOML file (see config/default.toml for the schema) and exposes
/// the solver parameters through typed accessors. A default constructor
/// (no file) provides the built-in defaults, which makes testing and
/// quick runs possible without a configuration file.
///
/// The Riemann flux family is central to the DG field module: the OCCA
/// kernels select the numerical flux at *compile time* via the JIT macro
/// `FLUX_TYPE` (see DgField::buildKernel), so this enum's integer values
/// are part of the public contract.

#pragma once

#include <map>
#include <string>

#include "common/Types.hpp"

/// @brief Numerical flux (Riemann solver) family.
///
/// Only LLF is implemented in this stage; the remaining entries are
/// reserved for later stages (Roe, Roe-E, Steger-Warming, van Leer).
enum class FluxType : int
{
    Llf           = 0, ///< Local Lax-Friedrichs / Rusanov (implemented).
    Roe           = 1, ///< Roe (reserved).
    RoeE          = 2, ///< Roe with entropy fix (reserved).
    StegerWarming = 3, ///< Steger-Warming FVS (reserved).
    VanLeer       = 4, ///< van Leer FVS (reserved).
    Count         = 5  ///< Number of flux families.
};

/// @brief Convert a \p FluxType to its integer value (for JIT defines).
constexpr int fluxTypeValue(FluxType t) noexcept
{
    return static_cast<int>(t);
}

/// @brief Parse a flux name string into a \p FluxType.
///
/// Accepted names (case-insensitive): "llf", "roe", "roe-e", "roe_e",
/// "steger-warming", "steger_warming", "van-leer", "van_leer".
/// Throws std::invalid_argument for unknown names.
FluxType parseFluxType(const std::string &name);

/// @brief Convert a \p FluxType back to its canonical string name.
const char *fluxTypeName(FluxType t) noexcept;

/// @brief Boundary-condition type.
enum class BcType : int
{
    Farfield = 0, ///< Far-field free-stream exterior state (implemented).
    Periodic = 1  ///< Translation-paired periodic boundaries (implemented).
};

/// @brief Parse a boundary-condition name string into a \p BcType.
///
/// Accepted names (case-insensitive): "farfield", "far-field", "far_field",
/// "periodic". Throws std::invalid_argument for unknown names.
BcType parseBcType(const std::string &name);

/// @brief Convert a \p BcType back to its canonical string name.
const char *bcTypeName(BcType t) noexcept;

/// @brief Time-marching method.
///
/// Simple explicit: Euler, SSPRK3. Embedded adaptive RK pairs: RK32, RK54,
/// SSPRK221/321/332/432 (Butcher tables in src/time/ButcherTable.hpp).
/// Implicit (advanced by dual time stepping): BE, DITR U2R2/U2R1/U3R1.
enum class TimeMethod : int
{
    Euler         = 0,  ///< Forward Euler (order 1).
    SspRk3        = 1,  ///< SSPRK3, Shu-Osher form (order 3).
    Rk32          = 2,  ///< Bogacki-Shampine 3(2) embedded pair.
    Rk54          = 3,  ///< Dormand-Prince 5(4) embedded pair.
    SspRk221      = 4,  ///< SSPRK2(2)1 embedded pair (Fekete et al. 2022).
    SspRk321      = 5,  ///< SSPRK3(2)1 embedded pair.
    SspRk332      = 6,  ///< SSPRK3(3)2 embedded pair (recommended).
    SspRk432      = 7,  ///< SSPRK4(3)2 embedded pair.
    BackwardEuler = 8,  ///< Backward Euler via dual time stepping.
    DitrU2R2      = 9,  ///< DITR U2R2 via dual time stepping.
    DitrU2R1      = 10, ///< DITR U2R1 via dual time stepping.
    DitrU3R1      = 11  ///< DITR U3R1 via dual time stepping.
};

/// @brief Parse a time-method name string into a \p TimeMethod.
///
/// Accepted names (case-insensitive): "euler", "ssprk3", "rk32", "rk54",
/// "ssprk221", "ssprk321", "ssprk332", "ssprk432", "be", "backward-euler",
/// "backward_euler", "ditr-u2r2", "ditr_u2r2", "ditr-u2r1", "ditr_u2r1",
/// "ditr-u3r1", "ditr_u3r1". Throws std::invalid_argument for unknown names.
TimeMethod parseTimeMethod(const std::string &name);

/// @brief Convert a \p TimeMethod back to its canonical string name.
const char *timeMethodName(TimeMethod t) noexcept;

/// @brief Whether a \p TimeMethod is advanced by dual time stepping.
constexpr bool isDualTimeMethod(TimeMethod t) noexcept
{
    return t == TimeMethod::BackwardEuler || t == TimeMethod::DitrU2R2 ||
           t == TimeMethod::DitrU2R1 || t == TimeMethod::DitrU3R1;
}

/// @brief Initial-condition type.
enum class IcType : int
{
    Uniform = 0, ///< Constant free stream from the [flow] section.
    Expr    = 1  ///< Primitive-variable expressions (exprtk) in x, y, t.
};

/// @brief Parse an initial-condition name string into an \p IcType.
///
/// Accepted names (case-insensitive): "uniform", "expr", "expression".
/// Throws std::invalid_argument for unknown names.
IcType parseIcType(const std::string &name);

/// @brief TOML configuration file parser and parameter store.
class Config
{
public:
    /// @brief Parse the TOML file at \p path.
    /// @throws toml::parse_error if the file is missing or malformed.
    explicit Config(const std::string &path);

    /// @brief Construct with built-in default parameters (no file).
    Config();

    // ---- Basis ----

    /// @brief Polynomial order $N$ of the DG approximation space.
    int polynomialOrder() const noexcept
    {
        return polynomial_order_;
    }

    /// @brief Number of Gauss-Legendre quadrature points per dimension $N_q$.
    int quadratureOrder() const noexcept
    {
        return quadrature_order_;
    }

    // ---- Mesh ----

    int meshNx() const noexcept
    {
        return mesh_nx_;
    }
    int meshNy() const noexcept
    {
        return mesh_ny_;
    }
    Real meshX0() const noexcept
    {
        return mesh_x0_;
    }
    Real meshY0() const noexcept
    {
        return mesh_y0_;
    }
    Real meshDx() const noexcept
    {
        return mesh_dx_;
    }
    Real meshDy() const noexcept
    {
        return mesh_dy_;
    }
    Real meshShear() const noexcept
    {
        return mesh_shear_;
    }
    bool meshSplitTriangles() const noexcept
    {
        return mesh_split_triangles_;
    }

    // ---- Riemann / numerical flux ----

    FluxType fluxType() const noexcept
    {
        return flux_type_;
    }
    int fluxTypeInt() const noexcept
    {
        return fluxTypeValue(flux_type_);
    }

    // ---- Gas ----

    Real gamma() const noexcept
    {
        return gamma_;
    }

    // ---- Flow (initial / reference state) ----

    Real flowRho() const noexcept
    {
        return flow_rho_;
    }
    Real flowU() const noexcept
    {
        return flow_u_;
    }
    Real flowV() const noexcept
    {
        return flow_v_;
    }
    Real flowP() const noexcept
    {
        return flow_p_;
    }
    Real flowMach() const noexcept
    {
        return flow_mach_;
    }

    // ---- OCCA device ----

    const std::string &occaMode() const noexcept
    {
        return occa_mode_;
    }

    /// @brief OpenMP thread count override ([occa] threads).
    /// @return 0 keeps the environment default (OMP_NUM_THREADS).
    int occaThreads() const noexcept
    {
        return occa_threads_;
    }

    /// @brief Device platform index ([occa] platform). Selects the OpenCL
    ///        / dpcpp platform on multi-platform systems.
    int occaPlatform() const noexcept
    {
        return occa_platform_;
    }

    /// @brief Device index within the platform ([occa] device); also used
    ///        as the CUDA / HIP device id. Ignored by Serial / OpenMP.
    int occaDevice() const noexcept
    {
        return occa_device_;
    }

    // ---- Boundary condition ----

    BcType bcType() const noexcept
    {
        return bc_type_;
    }

    // ---- Time marching ----

    /// @brief Time-integration method (see \p TimeMethod).
    TimeMethod timeMethod() const noexcept
    {
        return time_method_;
    }

    /// @brief CFL number for the dt estimate
    ///        $\Delta t = \mathrm{CFL} \min_K h_K / ((2N+1)\Lambda_{\max,K})$.
    Real timeCfl() const noexcept
    {
        return time_cfl_;
    }

    /// @brief Fixed time step; when $> 0$ it overrides the CFL estimate.
    Real timeDt() const noexcept
    {
        return time_dt_;
    }

    /// @brief Final physical time $T_{\mathrm{final}}$.
    Real timeFinal() const noexcept
    {
        return time_final_;
    }

    /// @brief Relative tolerance (adaptive RK error control / dual-time
    ///        convergence).
    Real timeRtol() const noexcept
    {
        return time_rtol_;
    }

    /// @brief Absolute tolerance (adaptive RK error control / dual-time
    ///        convergence).
    Real timeAtol() const noexcept
    {
        return time_atol_;
    }

    /// @brief Maximum pseudo-time steps per physical step (dual time).
    int timeMaxPseudoSteps() const noexcept
    {
        return time_max_pseudo_steps_;
    }

    /// @brief Pseudo stepper method for dual time stepping (embedded RK).
    TimeMethod timePseudoMethod() const noexcept
    {
        return time_pseudo_method_;
    }

    /// @brief Fixed pseudo time step; $\le 0$ means adaptive.
    Real timePseudoDt() const noexcept
    {
        return time_pseudo_dt_;
    }

    /// @brief Pseudo-stepper local error tolerance override (dual time,
    ///        adaptive pseudo mode); $\le 0$ keeps the automatic heuristic
    ///        ($10^{-3}$ double / $10^{-2}$ single precision).
    Real timePseudoRtol() const noexcept
    {
        return time_pseudo_rtol_;
    }

    /// @brief See timePseudoRtol().
    Real timePseudoAtol() const noexcept
    {
        return time_pseudo_atol_;
    }

    /// @brief Whether the pseudo stepper uses a single (scalar) dt with the
    ///        RMS-based controller (`step_single_dt` in the prototype).
    bool timePseudoSingleDt() const noexcept
    {
        return time_pseudo_single_dt_;
    }

    /// @brief Whether a rejected pseudo step may retry with a smaller dt
    ///        (`is_reject` in the prototype; default forced-accept).
    bool timePseudoReject() const noexcept
    {
        return time_pseudo_reject_;
    }

    /// @brief Whether the DITR stages are advanced by decoupled pseudo
    ///        steppers instead of a single stacked pseudo stepper.
    bool timeDualDecoupled() const noexcept
    {
        return time_dual_decoupled_;
    }

    // ---- Initial condition ----

    IcType icType() const noexcept
    {
        return ic_type_;
    }

    /// @brief Density expression (primitive variables, exprtk syntax).
    const std::string &icRho() const noexcept
    {
        return ic_rho_;
    }
    /// @brief X-velocity expression.
    const std::string &icU() const noexcept
    {
        return ic_u_;
    }
    /// @brief Y-velocity expression.
    const std::string &icV() const noexcept
    {
        return ic_v_;
    }
    /// @brief Pressure expression.
    const std::string &icP() const noexcept
    {
        return ic_p_;
    }

    /// @brief User numeric constants available in the initial-condition
    ///        expressions, e.g. `beta`, `x0`. The built-in symbols are
    ///        x, y, t and gamma.
    const std::map<std::string, Real> &icSymbols() const noexcept
    {
        return ic_symbols_;
    }

private:
    /// @brief Reset to built-in defaults (used by the default constructor).
    void setDefaults();

    int polynomial_order_ = 2;
    int quadrature_order_ = 4;

    int mesh_nx_               = 8;
    int mesh_ny_               = 8;
    Real mesh_x0_              = Real(0);
    Real mesh_y0_              = Real(0);
    Real mesh_dx_              = Real(1);
    Real mesh_dy_              = Real(1);
    Real mesh_shear_           = Real(0);
    bool mesh_split_triangles_ = false;

    FluxType flux_type_ = FluxType::Llf;
    Real gamma_         = Real(1.4);

    Real flow_rho_  = Real(1);
    Real flow_u_    = Real(0);
    Real flow_v_    = Real(0);
    Real flow_p_    = Real(1);
    Real flow_mach_ = Real(0.1);

    std::string occa_mode_ = "Serial";
    int occa_threads_      = 0;
    int occa_platform_     = 0;
    int occa_device_       = 0;

    BcType bc_type_ = BcType::Farfield;

    TimeMethod time_method_        = TimeMethod::SspRk3;
    Real time_cfl_                 = Real(0.2);
    Real time_dt_                  = Real(0);
    Real time_final_               = Real(1);
    Real time_rtol_                = Real(1e-6);
    Real time_atol_                = Real(1e-6);
    int time_max_pseudo_steps_     = 100;
    TimeMethod time_pseudo_method_ = TimeMethod::SspRk332;
    Real time_pseudo_dt_           = Real(0);
    Real time_pseudo_rtol_         = Real(0); ///< <= 0: automatic heuristic.
    Real time_pseudo_atol_         = Real(0); ///< <= 0: automatic heuristic.
    bool time_pseudo_single_dt_    = false;
    bool time_pseudo_reject_       = false;
    bool time_dual_decoupled_      = false;

    IcType ic_type_     = IcType::Uniform;
    std::string ic_rho_ = "1";
    std::string ic_u_   = "0";
    std::string ic_v_   = "0";
    std::string ic_p_   = "1";
    std::map<std::string, Real> ic_symbols_;
};
