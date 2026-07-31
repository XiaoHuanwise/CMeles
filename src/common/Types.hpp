/// @file Types.hpp
/// @brief Central floating-point type alias, enabling one-click switching between
///        double and float via the CMake option USE_FLOAT_PRECISION.
///
/// Usage:
///   cmake -B build -DUSE_FLOAT_PRECISION=ON  # compile with float
///   cmake -B build                           # compile with double (default)
///
/// All project code should use `Real` instead of `double`/`float` directly,
/// and use the Eigen convenience aliases (VectorXr, MatrixXr, MatrixX2r)
/// instead of the hard-coded Eigen typedefs (Eigen::VectorXd, etc.).

#pragma once

#include <Eigen/Dense>

#ifdef USE_FLOAT_PRECISION
using Real = float;
constexpr Real RealEpsilon = Real(1e-7);
#else
using Real = double;
constexpr Real RealEpsilon = Real(1e-15);
#endif

/// @name Eigen convenience aliases
/// @brief Typedefs that respect the active \p Real type, replacing the
///        hard-coded double Eigen typedefs (VectorXd, MatrixXd, etc.).
/// @{

/// @brief Dynamic-size column vector with scalar type Real.
using VectorXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

/// @brief Dynamic-size column-major matrix with scalar type Real.
/// Column-major layout matches Eigen's default and is efficient for
/// column-wise access patterns (e.g., sum-factorization).
using MatrixXrCol =
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

/// @brief Dynamic-size row-major matrix with scalar type Real.
/// Row-major layout matches OCCA's default memory order.
using MatrixXr =
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// @brief Dynamic-size matrix with 2 columns (row-major), scalar type Real.
using MatrixX2r = Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::RowMajor>;

/// @}
