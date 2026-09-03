/// @file ButcherTable.hpp
/// @brief Embedded Runge-Kutta Butcher tableaux for the adaptive stepper.
///
/// Table entries follow the prototype (.stepper.py) verbatim: C (stage
/// nodes), A (stage coefficients, row-major), B (solution weights of the
/// high-order formula), and E = b - b* (error weights, with an additional
/// FSAL entry for the f(u_new) row of K). The variants differ only by this
/// data — the RungeKuttaStepper code and its OKL kernels are shared, and
/// the coefficients are uploaded once and read at runtime.
///
/// Reference: docs/tech_docs/time_marching/explicit_adaptive_time_marching.md.

#pragma once

#include <array>

#include "common/Types.hpp"
#include "config/Config.hpp"

/// @brief Maximum number of RK stages across all tableaux.
inline constexpr int kMaxRkStages = 6;

/// @brief Butcher tableau of an embedded RK pair.
struct ButcherTable
{
    const char *name;        ///< Canonical method name.
    int nStages;             ///< Number of stages $s$.
    int order;               ///< Order of the solution weights B.
    int errorEstimatorOrder; ///< Order of the embedded error formula.

    std::array<Real, kMaxRkStages> C{}; ///< Stage nodes.
    std::array<std::array<Real, kMaxRkStages>, kMaxRkStages>
        A{};                                ///< Stage coefficients.
    std::array<Real, kMaxRkStages> B{};     ///< Solution weights.
    std::array<Real, kMaxRkStages + 1> E{}; ///< Error weights.
};

/// @brief Bogacki-Shampine 3(2) pair (scipy RK23).
inline constexpr ButcherTable kRk32{
    "rk32",
    3,
    3,
    2,
    {{Real(0), Real(1) / Real(2), Real(3) / Real(4)}},
    {{{Real(0), Real(0), Real(0)},
      {Real(1) / Real(2), Real(0), Real(0)},
      {Real(0), Real(3) / Real(4), Real(0)}}},
    {{Real(2) / Real(9), Real(1) / Real(3), Real(4) / Real(9)}},
    {{-Real(5) / Real(72), Real(1) / Real(12), Real(1) / Real(9),
      -Real(1) / Real(8)}}};

/// @brief Dormand-Prince 5(4) pair (scipy RK45).
inline constexpr ButcherTable kRk54{
    "rk54",
    6,
    5,
    4,
    {{Real(0), Real(1) / Real(5), Real(3) / Real(10), Real(4) / Real(5),
      Real(8) / Real(9), Real(1)}},
    {{{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)},
      {Real(1) / Real(5), Real(0), Real(0), Real(0), Real(0), Real(0)},
      {Real(3) / Real(40), Real(9) / Real(40), Real(0), Real(0), Real(0),
       Real(0)},
      {Real(44) / Real(45), -Real(56) / Real(15), Real(32) / Real(9), Real(0),
       Real(0), Real(0)},
      {Real(19372) / Real(6561), -Real(25360) / Real(2187),
       Real(64448) / Real(6561), -Real(212) / Real(729), Real(0), Real(0)},
      {Real(9017) / Real(3168), -Real(355) / Real(33), Real(46732) / Real(5247),
       Real(49) / Real(176), -Real(5103) / Real(18656), Real(0)}}},
    {{Real(35) / Real(384), Real(0), Real(500) / Real(1113),
      Real(125) / Real(192), -Real(2187) / Real(6784), Real(11) / Real(84)}},
    {{Real(71) / Real(57600), Real(0), -Real(71) / Real(16695),
      Real(71) / Real(1920), -Real(17253) / Real(339200), Real(22) / Real(525),
      -Real(1) / Real(40)}}};

/// @brief SSPRK2(2)1 embedded pair (Fekete et al. 2022).
inline constexpr ButcherTable kSspRk221{
    "ssprk221",
    2,
    2,
    1,
    {{Real(0), Real(1)}},
    {{{Real(0), Real(0)}, {Real(1), Real(0)}}},
    {{Real(0.5), Real(0.5)}},
    {{Real(0.5) - Real(0.694021459207626), Real(0.5) - Real(0.305978540792374),
      Real(0)}}};

/// @brief SSPRK3(2)1 embedded pair (Fekete et al. 2022).
inline constexpr ButcherTable kSspRk321{
    "ssprk321",
    3,
    2,
    1,
    {{Real(0), Real(0.5), Real(1)}},
    {{{Real(0), Real(0), Real(0)},
      {Real(0.5), Real(0), Real(0)},
      {Real(0.5), Real(0.5), Real(0)}}},
    {{Real(1) / Real(3), Real(1) / Real(3), Real(1) / Real(3)}},
    {{Real(1) / Real(3) - Real(0.635564950337195),
      Real(1) / Real(3) - Real(0.033488381714827),
      Real(1) / Real(3) - Real(0.330946667947978), Real(0)}}};

/// @brief SSPRK3(3)2 embedded pair — recommended default (Fekete et al.
///        2022).
inline constexpr ButcherTable kSspRk332{
    "ssprk332",
    3,
    3,
    2,
    {{Real(0), Real(1), Real(0.5)}},
    {{{Real(0), Real(0), Real(0)},
      {Real(1), Real(0), Real(0)},
      {Real(0.25), Real(0.25), Real(0)}}},
    {{Real(1) / Real(6), Real(1) / Real(6), Real(2) / Real(3)}},
    {{Real(1) / Real(6) - Real(0.291485418878409),
      Real(1) / Real(6) - Real(0.291485418878409),
      Real(2) / Real(3) - Real(0.417029162243181), Real(0)}}};

/// @brief SSPRK4(3)2 embedded pair (Fekete et al. 2022).
inline constexpr ButcherTable kSspRk432{
    "ssprk432",
    4,
    3,
    2,
    {{Real(0), Real(0.5), Real(1), Real(0.5)}},
    {{{Real(0), Real(0), Real(0), Real(0)},
      {Real(0.5), Real(0), Real(0), Real(0)},
      {Real(0.5), Real(0.5), Real(0), Real(0)},
      {Real(1) / Real(6), Real(1) / Real(6), Real(1) / Real(6), Real(0)}}},
    {{Real(1) / Real(6), Real(1) / Real(6), Real(1) / Real(6),
      Real(1) / Real(2)}},
    {{Real(1) / Real(6) - Real(0.138870252716866),
      Real(1) / Real(6) - Real(0.722259494566267),
      Real(1) / Real(6) - Real(0.138870252716866), Real(1) / Real(2),
      Real(0)}}};

/// @brief Map an embedded-RK \p TimeMethod to its tableau (nullptr for
///        non-embedded methods).
inline constexpr const ButcherTable *butcherTableForMethod(TimeMethod m)
{
    switch (m)
    {
        case TimeMethod::Rk32:
            return &kRk32;
        case TimeMethod::Rk54:
            return &kRk54;
        case TimeMethod::SspRk221:
            return &kSspRk221;
        case TimeMethod::SspRk321:
            return &kSspRk321;
        case TimeMethod::SspRk332:
            return &kSspRk332;
        case TimeMethod::SspRk432:
            return &kSspRk432;
        default:
            return nullptr;
    }
}
