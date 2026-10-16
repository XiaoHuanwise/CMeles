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
enum class FluxType : int {
    Llf           = 0, ///< Local Lax-Friedrichs / Rusanov (implemented).
    Roe           = 1, ///< Roe (reserved).
    RoeE          = 2, ///< Roe with entropy fix (reserved).
    StegerWarming = 3, ///< Steger-Warming FVS (reserved).
    VanLeer       = 4, ///< van Leer FVS (reserved).
    Count         = 5  ///< Number of flux families.
};

/// @brief Convert a \p FluxType to its integer value (for JIT defines).
constexpr int fluxTypeValue(FluxType t) noexcept {
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
enum class BcType : int {
    Farfield = 0, ///< Far-field free-stream exterior state (implemented).
    Periodic = 1, ///< Translation-paired periodic boundaries (implemented).
    Free     = 2, ///< Zero-gradient (free) boundary: the exterior state
                  ///< copies the interior trace; exactly neutral while the
                  ///< flow near the wall is constant (e.g. the 2-D Riemann
                  ///< tests before any wave reaches the boundary).
    Riemann = 3,  ///< Characteristic (Riemann-invariant / pressure
                  ///< far-field) non-reflecting boundary: the boundary
                  ///< state mixes the interior trace with a reference
                  ///< state through the normal acoustic invariants, so
                  ///< outgoing waves leave without reflection and the
                  ///< pressure is anchored to the reference value at
                  ///< subsonic outflow (see \p RiemannTarget).
    Slip = 4,     ///< Slip (reflection) wall: the exterior state mirrors
                  ///< the interior trace with the normal momentum negated,
                  ///< $\rho$ and $E$ unchanged. Any consistent numerical
                  ///< flux then produces exactly zero wall-normal mass
                  ///< flux — the inviscid (Euler) wall condition.
    Expr = 5,     ///< Dirichlet boundary from primitive-variable
                  ///< expressions ([bc] rho / u / v / p evaluated at
                  ///< x, y, t on every boundary quadrature point; the
                  ///< values are refreshed once per physical time step).
    FreeAvg = 6   ///< Zero-gradient from the CELL MEAN: the exterior state
                  ///< is the interior element's mean (the polynomial's
                  ///< constant mode u^(0); phi_0 = 1/2 so mean = u^(0)/2,
                  ///< exact on affine elements) instead of the pointwise
                  ///< trace -- the finite-volume edge-pad analogue. The
                  ///  pointwise free trace carries the polynomial's
                  ///  high-order oscillations straight into the wall flux
                  ///  when a shock crosses the wall cell; the constant
                  ///  mode averages them out and supplies only the mean.
};

/// @brief Per-side boundary-condition override ([bc.x_low] etc.). Faces are
///        assigned to a side by their outward normal; a side table missing
///        \p type falls back to the global [bc] type. The optional
///        \p split_x / \p split_type switch classifies boundary faces by
///        their centroid x (faces with centroid x < split_x take
///        \p splitType) — e.g. the Double-Mach floor (inflow segment ahead
///        of the shock foot, reflection wall behind it).
struct SideBc {
    bool hasType     = false;        ///< Side table with a type key present.
    BcType type      = BcType::Free; ///< Side type (valid when \p hasType).
    bool hasSplit    = false;        ///< split_x / split_type present.
    Real splitX      = Real(0);      ///< Split abscissa.
    BcType splitType = BcType::Slip; ///< Type for faces left of splitX.
};

/// @brief Boundary side index for the per-side [bc] override tables.
enum class BcSide : int {
    XLow  = 0, ///< Left boundary (x = x_min).
    XHigh = 1, ///< Right boundary (x = x_max).
    YLow  = 2, ///< Bottom boundary (y = y_min).
    YHigh = 3  ///< Top boundary (y = y_max).
};

/// @brief Reference-state source for the characteristic (Riemann)
///        boundary condition ([bc] riemann_target).
enum class RiemannTarget : int {
    Flow = 0, ///< Constant [flow] (rho, u, v, p): user-provided operating
              ///< condition — the standard far-field / pressure-outlet
              ///< interface (cf. Fluent pressure-outlet, SU2 far-field).
    Ic = 1,   ///< Trace of the initial field at each boundary quadrature
              ///< point: for benchmark problems whose ambient environment
              ///< varies along the wall (the four-quadrant 2-D Riemann
              ///< data), where a single constant target would anchor the
              ///< wrong pressure on part of the boundary.
    Expr = 2  ///< Analytic wall data: the [bc] rho/u/v/p expressions
              ///< evaluated at the boundary point (x, y, t) and refreshed
              ///< every time step (the same evaluator the Dirichlet
              ///< boundary uses). The reference then follows moving waves
              ///< instead of a frozen snapshot — e.g. the exact
              ///< self-similar 1-D Riemann wall states of the 2-D Riemann
              ///< problem — while the characteristic mixing still lets the
              ///< outgoing information leave freely.
};

/// @brief Parse a boundary-condition name string into a \p BcType.
///
/// Accepted names (case-insensitive): "farfield", "far-field", "far_field",
/// "periodic", "free", "zero-gradient", "zero_gradient", "riemann",
/// "characteristic", "pressure-farfield", "slip", "slip-wall", "slip_wall",
/// "wall", "expr", "dirichlet". Throws std::invalid_argument for unknown
/// names.
BcType parseBcType(const std::string &name);

/// @brief Convert a \p BcType back to its canonical string name.
const char *bcTypeName(BcType t) noexcept;

/// @brief Parse a Riemann-target name ("flow" | "ic" | "expr") into a
///        \p RiemannTarget. Throws std::invalid_argument otherwise.
RiemannTarget parseRiemannTarget(const std::string &name);

/// @brief Convert a \p RiemannTarget back to its canonical string name.
const char *riemannTargetName(RiemannTarget t) noexcept;

/// @brief Time-marching method.
///
/// Simple explicit: Euler, SSPRK3. Embedded adaptive RK pairs: RK32, RK54,
/// SSPRK221/321/332/432 (Butcher tables in src/time/ButcherTable.hpp).
/// Implicit (advanced by dual time stepping): BE, DITR U2R2/U2R1/U3R1.
enum class TimeMethod : int {
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
constexpr bool isDualTimeMethod(TimeMethod t) noexcept {
    return t == TimeMethod::BackwardEuler || t == TimeMethod::DitrU2R2 ||
           t == TimeMethod::DitrU2R1 || t == TimeMethod::DitrU3R1;
}

/// @brief Adaptive pseudo-dt granularity of the dual-time pseudo stepper.
enum class PseudoDtMode : int {
    Global = 0, ///< Single scalar dt, RMS-based controller (default).
    Local  = 1  ///< One pseudo dt per modal coefficient (per-DOF controller).
};

/// @brief Parse a pseudo-dt-mode name string into a \p PseudoDtMode.
///
/// Accepted names (case-insensitive): "global", "local". Throws
/// std::invalid_argument for unknown names.
PseudoDtMode parsePseudoDtMode(const std::string &name);

/// @brief Discontinuity limiter family ([limiter] section).
enum class LimiterType : int {
    None   = 0, ///< No limiter (the default; preserves unlimited behaviour).
    MrWeno = 1  ///< Multi-resolution WENO limiter (Zhu & Shu).
};

/// @brief Parse a limiter name string into a \p LimiterType.
///
/// Accepted names (case-insensitive): "none", "mr-weno", "mr_weno",
/// "mrweno". Throws std::invalid_argument for unknown names.
LimiterType parseLimiterType(const std::string &name);

/// @brief Convert a \p LimiterType back to its canonical string name.
const char *limiterTypeName(LimiterType t) noexcept;

/// @brief Normalisation of the KXRCF boundary-value scale ([limiter]
///        shock_norm): the trace of the indicator variable (density) along
///        the element boundary is reduced by min / max / mean over the
///        boundary quadrature points.
enum class ShockNorm : int {
    Min  = 0, ///< Minimum trace (Zhu & Shu; larger indicator, conservative).
    Max  = 1, ///< Maximum trace.
    Mean = 2  ///< Arithmetic mean of the trace.
};

/// @brief Parse a shock-norm name string into a \p ShockNorm.
///
/// Accepted names (case-insensitive): "min", "max", "mean". Throws
/// std::invalid_argument for unknown names.
ShockNorm parseShockNorm(const std::string &name);

/// @brief Convert a \p ShockNorm back to its canonical string name.
const char *shockNormName(ShockNorm t) noexcept;

/// @brief Initial-condition type.
enum class IcType : int {
    Uniform = 0, ///< Constant free stream from the [flow] section.
    Expr    = 1  ///< Primitive-variable expressions (exprtk) in x, y, t.
};

/// @brief Parse an initial-condition name string into an \p IcType.
///
/// Accepted names (case-insensitive): "uniform", "expr", "expression".
/// Throws std::invalid_argument for unknown names.
IcType parseIcType(const std::string &name);

/// @brief Solution output strategy ([output] section).
enum class OutputStrategy : int {
    Direct =
        0, ///< Strategy A: quadrature-point SoA files, ready for post-processing.
    Checkpoint =
        1 ///< Strategy B: modal-coefficient checkpoints + offline conversion.
};

/// @brief Parse an output-strategy name string into an \p OutputStrategy.
///
/// Accepted names (case-insensitive): "direct", "checkpoint". Throws
/// std::invalid_argument for unknown names.
OutputStrategy parseOutputStrategy(const std::string &name);

/// @brief Convert an \p OutputStrategy back to its canonical string name.
const char *outputStrategyName(OutputStrategy t) noexcept;

/// @brief Solution-file format ([output] format): the plain HDF5 archive
///        plus the opt-in visualization formats.
enum class OutputFormat : int {
    Hdf5 = 0,    ///< Plain solution_<step>.h5 (the previous behaviour).
    Xdmf = 1,    ///< HDF5 + self-contained XDMF sidecar (ParaView cloud
                 ///< maps; the pair solution_<step>.h5 / .xmf moves
                 ///< together).
    Tecplot = 2, ///< Classic binary PLT solution_<step>.plt (Tecplot 360).
    All     = 3, ///< HDF5 + XDMF + PLT + VTU.
    Vtu     = 4, ///< HDF5 + VTK XML UnstructuredGrid solution_<step>.vtu
                 ///< whose cells are the DG elements as high-order
                 ///< Lagrange cells (ParaView renders the actual
                 ///< polynomial with adaptive subdivision; the PVD
                 ///< <prefix>_series.pvd carries the time axis).
};

/// @brief Parse an output-format name string into an \p OutputFormat.
///
/// Accepted names (case-insensitive): "hdf5", "xdmf", "tecplot", "plt",
/// "vtu", "all". Throws std::invalid_argument for unknown names.
OutputFormat parseOutputFormat(const std::string &name);

/// @brief Convert an \p OutputFormat back to its canonical string name.
const char *outputFormatName(OutputFormat t) noexcept;

/// @brief Whether an \p OutputFormat requests the plain HDF5 file.
constexpr bool formatWritesHdf5(OutputFormat f) noexcept {
    return f == OutputFormat::Hdf5 || f == OutputFormat::Xdmf ||
           f == OutputFormat::All;
}

/// @brief Whether an \p OutputFormat requests the XDMF sidecar.
constexpr bool formatWritesXdmf(OutputFormat f) noexcept {
    return f == OutputFormat::Xdmf || f == OutputFormat::All;
}

/// @brief Whether an \p OutputFormat requests the Tecplot PLT file.
constexpr bool formatWritesTecplot(OutputFormat f) noexcept {
    return f == OutputFormat::Tecplot || f == OutputFormat::All;
}

/// @brief Whether an \p OutputFormat requests the high-order VTU file.
constexpr bool formatWritesVtu(OutputFormat f) noexcept {
    return f == OutputFormat::Vtu || f == OutputFormat::All;
}

/// @brief TOML configuration file parser and parameter store.
class Config {
public:
    /// @brief Parse the TOML file at \p path.
    /// @throws toml::parse_error if the file is missing or malformed.
    explicit Config(const std::string &path);

    /// @brief Construct with built-in default parameters (no file).
    Config();

    // ---- Basis ----

    /// @brief Polynomial order $N$ of the DG approximation space.
    int polynomialOrder() const noexcept {
        return polynomial_order_;
    }

    /// @brief Number of Gauss-Legendre quadrature points per dimension $N_q$.
    int quadratureOrder() const noexcept {
        return quadrature_order_;
    }

    // ---- Mesh ----

    int meshNx() const noexcept {
        return mesh_nx_;
    }
    int meshNy() const noexcept {
        return mesh_ny_;
    }
    Real meshX0() const noexcept {
        return mesh_x0_;
    }
    Real meshY0() const noexcept {
        return mesh_y0_;
    }
    Real meshDx() const noexcept {
        return mesh_dx_;
    }
    Real meshDy() const noexcept {
        return mesh_dy_;
    }
    Real meshShear() const noexcept {
        return mesh_shear_;
    }
    bool meshSplitTriangles() const noexcept {
        return mesh_split_triangles_;
    }

    // ---- Riemann / numerical flux ----

    FluxType fluxType() const noexcept {
        return flux_type_;
    }
    int fluxTypeInt() const noexcept {
        return fluxTypeValue(flux_type_);
    }

    // ---- Gas ----

    Real gamma() const noexcept {
        return gamma_;
    }

    // ---- Flow (initial / reference state) ----

    Real flowRho() const noexcept {
        return flow_rho_;
    }
    Real flowU() const noexcept {
        return flow_u_;
    }
    Real flowV() const noexcept {
        return flow_v_;
    }
    Real flowP() const noexcept {
        return flow_p_;
    }
    Real flowMach() const noexcept {
        return flow_mach_;
    }

    // ---- OCCA device ----

    const std::string &occaMode() const noexcept {
        return occa_mode_;
    }

    /// @brief OpenMP thread count override ([occa] threads).
    /// @return 0 keeps the environment default (OMP_NUM_THREADS).
    int occaThreads() const noexcept {
        return occa_threads_;
    }

    /// @brief Device platform index ([occa] platform). Selects the OpenCL
    ///        / dpcpp platform on multi-platform systems.
    int occaPlatform() const noexcept {
        return occa_platform_;
    }

    /// @brief Device index within the platform ([occa] device); also used
    ///        as the CUDA / HIP device id. Ignored by Serial / OpenMP.
    int occaDevice() const noexcept {
        return occa_device_;
    }

    // ---- Boundary condition ----

    BcType bcType() const noexcept {
        return bc_type_;
    }

    /// @brief Reference-state source for the characteristic (Riemann)
    ///        boundary ([bc] riemann_target, default "flow"). Only
    ///        consulted when \p bcType() is \p BcType::Riemann.
    RiemannTarget riemannTarget() const noexcept {
        return riemann_target_;
    }

    /// @brief Per-side [bc] override table for side \p side (see \p BcSide).
    ///        Sides without a table fall back to the global \p bcType();
    ///        periodic sides are not overridable (pairing is mesh-global).
    const SideBc &bcSide(BcSide side) const noexcept {
        return bc_side_[static_cast<int>(side)];
    }

    /// @brief Dirichlet density expression for \p BcType::Expr faces
    ///        ([bc] rho; empty unless configured).
    const std::string &bcRho() const noexcept {
        return bc_rho_;
    }

    /// @brief Dirichlet x-velocity expression for \p BcType::Expr faces
    ///        ([bc] u; empty unless configured).
    const std::string &bcU() const noexcept {
        return bc_u_;
    }

    /// @brief Dirichlet y-velocity expression for \p BcType::Expr faces
    ///        ([bc] v; empty unless configured).
    const std::string &bcV() const noexcept {
        return bc_v_;
    }

    /// @brief Dirichlet pressure expression for \p BcType::Expr faces
    ///        ([bc] p; empty unless configured).
    const std::string &bcP() const noexcept {
        return bc_p_;
    }

    // ---- Time marching ----

    /// @brief Time-integration method (see \p TimeMethod).
    TimeMethod timeMethod() const noexcept {
        return time_method_;
    }

    /// @brief CFL number for the dt estimate
    ///        $\Delta t = \mathrm{CFL} \min_K h_K / ((2N+1)\Lambda_{\max,K})$.
    Real timeCfl() const noexcept {
        return time_cfl_;
    }

    /// @brief Fixed time step; when $> 0$ it overrides the CFL estimate.
    Real timeDt() const noexcept {
        return time_dt_;
    }

    /// @brief Final physical time $T_{\mathrm{final}}$.
    Real timeFinal() const noexcept {
        return time_final_;
    }

    /// @brief Relative tolerance (adaptive RK error control / dual-time
    ///        convergence).
    Real timeRtol() const noexcept {
        return time_rtol_;
    }

    /// @brief Absolute tolerance (adaptive RK error control / dual-time
    ///        convergence).
    Real timeAtol() const noexcept {
        return time_atol_;
    }

    /// @brief PI-controller safety factor (adaptive RK step size; also
    ///        applied to the dual-time pseudo stepper).
    Real timeSafety() const noexcept {
        return time_safety_;
    }

    /// @brief Minimum step-size shrink factor applied on rejection.
    Real timeMinFactor() const noexcept {
        return time_min_factor_;
    }

    /// @brief Maximum step-size growth factor per accepted step.
    Real timeMaxFactor() const noexcept {
        return time_max_factor_;
    }

    /// @brief Step-size growth cap relative to the initial step.
    Real timeMaxGrowth() const noexcept {
        return time_max_growth_;
    }

    /// @brief Maximum pseudo-time steps per physical step (dual time).
    int timeMaxPseudoSteps() const noexcept {
        return time_max_pseudo_steps_;
    }

    /// @brief Pseudo stepper method for dual time stepping (embedded RK).
    TimeMethod timePseudoMethod() const noexcept {
        return time_pseudo_method_;
    }

    /// @brief Fixed pseudo time step; $\le 0$ means adaptive.
    Real timePseudoDt() const noexcept {
        return time_pseudo_dt_;
    }

    /// @brief Pseudo-stepper local error tolerance override (dual time,
    ///        adaptive pseudo mode); $\le 0$ keeps the automatic heuristic
    ///        ($10^{-3}$ double / $10^{-2}$ single precision).
    Real timePseudoRtol() const noexcept {
        return time_pseudo_rtol_;
    }

    /// @brief See timePseudoRtol().
    Real timePseudoAtol() const noexcept {
        return time_pseudo_atol_;
    }

    /// @brief Adaptive pseudo-dt granularity of the dual-time pseudo
    ///        stepper (`pseudo_dt_mode` in the prototype's terms:
    ///        `is_single_dt` false = Local). A fixed pseudo step
    ///        ([time_marching] pseudo_dt > 0) takes precedence over both
    ///        adaptive modes.
    PseudoDtMode timePseudoDtMode() const noexcept {
        return time_pseudo_dt_mode_;
    }

    /// @brief Whether a rejected pseudo step may retry with a smaller dt
    ///        (`is_reject` in the prototype; default forced-accept).
    bool timePseudoReject() const noexcept {
        return time_pseudo_reject_;
    }

    /// @brief Whether the DITR stages are advanced by decoupled pseudo
    ///        steppers instead of a single stacked pseudo stepper.
    bool timeDualDecoupled() const noexcept {
        return time_dual_decoupled_;
    }

    // ---- Limiter ----

    /// @brief Limiter family (see \p LimiterType). None by default.
    LimiterType limiterType() const noexcept {
        return limiter_type_;
    }

    /// @brief KXRCF troubled-cell threshold $C_k$.
    Real limiterCkc() const noexcept {
        return limiter_ckc_;
    }

    /// @brief KXRCF boundary-value normalisation (see \p ShockNorm).
    ShockNorm limiterShockNorm() const noexcept {
        return limiter_shock_norm_;
    }

    /// @brief MR-WENO linear weight of the low-order candidate,
    ///        $\gamma_{\ell_2-1,\ell_2}$.
    Real limiterGammaLow() const noexcept {
        return limiter_gamma_low_;
    }

    /// @brief MR-WENO linear weight of the high-order (detail) candidate,
    ///        $\gamma_{\ell_2,\ell_2}$.
    Real limiterGammaHigh() const noexcept {
        return limiter_gamma_high_;
    }

    /// @brief WENO-Z anti-zero epsilon $\varepsilon_w$.
    Real limiterEpsilon() const noexcept {
        return limiter_epsilon_;
    }

    /// @brief Force every element to be flagged as troubled (diagnostic
    ///        switch for the accuracy-acceptance tests: with the whole mesh
    ///        flagged, a smooth solution must be recovered to round-off on
    ///        affine elements). Never enable for production runs.
    bool limiterForceTroubled() const noexcept {
        return limiter_force_troubled_;
    }

    /// @brief Whether the positivity-preserving scaling limiter
    ///        ([limiter] positivity; Zhang & Shu) is enabled. Off by
    ///        default. Composable with the MR-WENO limiter (both run:
    ///        MR-WENO first, positivity last) and usable standalone
    ///        (type = "none").
    bool limiterPositivity() const noexcept {
        return limiter_positivity_;
    }

    /// @brief Relative positivity margin $\varepsilon$
    ///        ([limiter] positivity_epsilon): after limiting, every
    ///        quadrature / face point satisfies
    ///        $\rho \ge \varepsilon \bar{\rho}$ and
    ///        $e \ge \varepsilon \bar{e}$.
    Real limiterPositivityEpsilon() const noexcept {
        return limiter_positivity_epsilon_;
    }

    // ---- Initial condition ----

    IcType icType() const noexcept {
        return ic_type_;
    }

    /// @brief Density expression (primitive variables, exprtk syntax).
    const std::string &icRho() const noexcept {
        return ic_rho_;
    }
    /// @brief X-velocity expression.
    const std::string &icU() const noexcept {
        return ic_u_;
    }
    /// @brief Y-velocity expression.
    const std::string &icV() const noexcept {
        return ic_v_;
    }
    /// @brief Pressure expression.
    const std::string &icP() const noexcept {
        return ic_p_;
    }

    /// @brief User numeric constants available in the initial-condition
    ///        expressions, e.g. `beta`, `x0`. The built-in symbols are
    ///        x, y, t and gamma.
    const std::map<std::string, Real> &icSymbols() const noexcept {
        return ic_symbols_;
    }

    // ---- Output ----

    /// @brief Master switch of the [output] section. When false (the
    ///        default) the solver produces no files at all.
    bool outputEnable() const noexcept {
        return output_enable_;
    }

    /// @brief Output strategy (see \p OutputStrategy).
    OutputStrategy outputStrategy() const noexcept {
        return output_strategy_;
    }

    /// @brief Solution-file format (see \p OutputFormat).
    OutputFormat outputFormat() const noexcept {
        return output_format_;
    }

    /// @brief Whether the XDMF temporal-collection index
    ///        (<prefix>_series.xmf) is maintained ([output] series, default
    ///        true). Only effective while the format writes XDMF ("xdmf"
    ///        or "all") — ParaView's Xdmf3 readers do not lift a lone
    ///        uniform grid's <Time> into the time axis, so the index file
    ///        is the supported way to open a run as a time series.
    bool outputSeries() const noexcept {
        return output_series_;
    }

    /// @brief Whether the HDF5 archive is written ([output] archive,
    ///        default true). The archive is the storage backbone of the
    ///        "hdf5"/"xdmf" formats and the archival companion of "vtu";
    ///        with archive = false only the self-contained formats
    ///        (vtu / plt) are produced, and a format that cannot exist
    ///        without its heavy data ("hdf5", "xdmf") is ignored.
    bool outputArchive() const noexcept {
        return output_archive_;
    }

    /// @brief File-name prefix derived from the configuration file name
    ///        (stem, no extension): all output files of a run are named
    ///        <prefix>_solution_<step>.<ext>, <prefix>_checkpoint_<step>.h5,
    ///        <prefix>_mesh.h5 and <prefix>_series.xmf, so several configs
    ///        can share one output directory. Empty when the Config was
    ///        not built from a file (the legacy "solution_<step>" naming).
    const std::string &outputPrefix() const noexcept {
        return output_prefix_;
    }

    /// @brief Number of time steps between two outputs ($\ge 1$ when
    ///        output is enabled).
    int outputInterval() const noexcept {
        return output_interval_;
    }

    /// @brief Output directory (created if missing when output is enabled).
    const std::string &outputDirectory() const noexcept {
        return output_directory_;
    }

    // ---- Log ----

    /// @brief Console log output depth ([log] level). A message prints when
    ///        its depth is <= this level: 0 prints nothing, 1 (default) the
    ///        physical-step layer (banner, progress lines, aborts, summary),
    ///        2 the inner-iteration layer (dual-time pseudo statistics);
    ///        deeper layers may be added, so no upper bound is imposed.
    int logLevel() const noexcept {
        return log_level_;
    }

    /// @brief Number of time steps between two progress lines ($\ge 1$).
    int logInterval() const noexcept {
        return log_interval_;
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
    SideBc bc_side_[4];  ///< Per-side [bc] overrides (empty by default).
    std::string bc_rho_; ///< Expr-face Dirichlet density expression.
    std::string bc_u_;   ///< Expr-face Dirichlet x-velocity expression.
    std::string bc_v_;   ///< Expr-face y-velocity expression.
    std::string bc_p_;   ///< Expr-face Dirichlet pressure expression.
    RiemannTarget riemann_target_ = RiemannTarget::Flow;

    TimeMethod time_method_           = TimeMethod::SspRk3;
    Real time_cfl_                    = Real(0.2);
    Real time_dt_                     = Real(0);
    Real time_final_                  = Real(1);
    Real time_rtol_                   = Real(1e-6);
    Real time_atol_                   = Real(1e-6);
    Real time_safety_                 = Real(0.9);
    Real time_min_factor_             = Real(0.2);
    Real time_max_factor_             = Real(5);
    Real time_max_growth_             = Real(10);
    int time_max_pseudo_steps_        = 100;
    TimeMethod time_pseudo_method_    = TimeMethod::SspRk332;
    Real time_pseudo_dt_              = Real(0);
    Real time_pseudo_rtol_            = Real(0); ///< <= 0: automatic heuristic.
    Real time_pseudo_atol_            = Real(0); ///< <= 0: automatic heuristic.
    PseudoDtMode time_pseudo_dt_mode_ = PseudoDtMode::Global;
    bool time_pseudo_reject_          = false;
    bool time_dual_decoupled_         = false;

    LimiterType limiter_type_     = LimiterType::None;
    Real limiter_ckc_             = Real(1);
    ShockNorm limiter_shock_norm_ = ShockNorm::Min;
    Real limiter_gamma_low_       = Real(0.01);
    Real limiter_gamma_high_      = Real(0.99);
    Real limiter_epsilon_         = Real(1e-6);
    bool limiter_force_troubled_  = false;
    bool limiter_positivity_      = false;
    Real limiter_positivity_epsilon_ = Real(1e-10);

    IcType ic_type_     = IcType::Uniform;
    std::string ic_rho_ = "1";
    std::string ic_u_   = "0";
    std::string ic_v_   = "0";
    std::string ic_p_   = "1";
    std::map<std::string, Real> ic_symbols_;

    bool output_enable_             = false;
    OutputStrategy output_strategy_ = OutputStrategy::Direct;
    OutputFormat output_format_     = OutputFormat::Hdf5;
    bool output_series_             = true;
    bool output_archive_            = true;
    std::string
        output_prefix_; ///< Config-file stem; empty when not from a file.
    int output_interval_          = 100;
    std::string output_directory_ = "./results";

    int log_level_    = 1;
    int log_interval_ = 100;
};
