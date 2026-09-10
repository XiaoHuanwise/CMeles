/// @file SolutionWriter.hpp
/// @brief Strategy A ("direct") output: quadrature-point SoA solution
///        files that post-processing tools can read immediately.

#pragma once

#include <string>
#include <vector>

#include "common/Types.hpp"
#include "io/OutputWriterBase.hpp"

class Config;
class DgField;

/// @brief Strategy A output writer: solution_<step>.h5.
///
/// At every write the modal coefficients are copied to the host and
/// projected onto the quadrature points through the Vandermonde matrix,
///
/// $$ u(\xi_q) = \sum_{m=0}^{N_{\text{modes}}-1} \hat u_m \, \phi_m(\xi_q), $$
///
/// the inverse of the L2 projection in DgField::setInitialConditionNodal.
/// The seven per-quantity arrays are written as SoA datasets flattened as
/// $k = e \cdot N_q^2 + q$, the same ordering as mesh/x and mesh/y.
class SolutionWriter : public OutputWriterBase {
public:
    /// @param cfg    Configuration (gamma, polynomial order, [output]).
    /// @param field  DG field supplying o_u(), sizes and the basis
    ///               (lifetime must exceed this object).
    SolutionWriter(const Config &cfg, const DgField &field);

    void write(long step, Real time) override;

private:
    const DgField &field_;

    Real gamma_;
    int polynomialOrder_;
    std::string directory_;

    int nElem_   = 0;
    int nModes_  = 0;
    int nq2_     = 0;
    size_t nTot_ = 0; ///< nElem_ * nq2_.

    /// Physical quadrature-point coordinates (contiguous per component).
    std::vector<Real> coordsX_, coordsY_;

    /// Reusable host buffers: modal coefficients, the projection scratch
    /// and the seven SoA fields.
    std::vector<Real> modal_;
    VectorXr nodal_;
    std::vector<Real> rho_, rhoU_, rhoV_, energy_, u_, v_, p_;
};
