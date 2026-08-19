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
};
