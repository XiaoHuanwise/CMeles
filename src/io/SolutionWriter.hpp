/// @file SolutionWriter.hpp
/// @brief Strategy A ("direct") output writer for the opt-in output
///        formats: SoA solution files plus the visualization sidecars
///        ([output] format = xdmf / tecplot / all).

#pragma once

#include <string>
#include <vector>

#include "common/Types.hpp"
#include "config/Config.hpp"
#include "io/OutputWriterBase.hpp"
#include "io/VizWriter.hpp"

class DgField;

/// @brief Strategy A output writer: solution_<step>.h5 plus the opt-in
///        visualization files ([output] format = xdmf / tecplot / all).
///
/// At every write the modal coefficients are copied to the host and
/// projected onto the output node set through a Vandermonde matrix,
///
/// $$ u(\xi_k) = \sum_{m=0}^{N_{\text{modes}}-1} \hat u_m \, \phi_m(\xi_k), $$
///
/// the inverse of the L2 projection in DgField::setInitialConditionNodal.
/// The node set is selected ONCE at construction by the output format:
///
///  - format "hdf5" (the default): the $N_q^2$ interior Gauss points per
///    element — the archival layout, identical to the pre-visualization
///    behavior (dataset count $e \cdot N_q^2 + q$);
///  - formats "xdmf" / "tecplot" / "all": the boundary-inclusive extended
///    grid $\{-1, \xi_1, \dots, \xi_{N_q}, +1\}^2$ per element
///    ($(N_q+2)^2$ nodes, a superset of the Gauss points), so every
///    element's sub-grid reaches its interfaces. Values are evaluated
///    from the element's OWN modal coefficients — no cross-element
///    interpolation — and the DG discontinuity stays a color jump at the
///    interfaces.
///
/// The seven per-quantity arrays are written as SoA datasets flattened as
/// $k = e \cdot n^2 + (j \cdot n + i)$, the same ordering as mesh/x and
/// mesh/y. The XDMF / Tecplot outputs visualize the same nodal data
/// through the per-element-type adaptive sub-grid built once at
/// construction.
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
    OutputFormat format_ = OutputFormat::Hdf5;
    std::string prefix_;         ///< Config-file stem; empty -> legacy naming.
    bool seriesEnabled_ = false; ///< [output] series, effective for xdmf.

    /// Steps written so far (h5 basename + time), maintained
    /// incrementally to regenerate the <prefix>_series.xmf index.
    std::vector<viz::SeriesEntry> seriesSteps_;

    /// @brief Index-file name: <prefix>_series.xmf (series.xmf without a
    ///        prefix).
    std::string seriesFileName() const {
        return prefix_.empty() ? "series.xmf" : prefix_ + "_series.xmf";
    }

    int nElem_  = 0;
    int nModes_ = 0;
    int nOut1D_ = 0; ///< Output nodes per tensor direction ($N_q$ or $N_q+2$).
    int nOut2_  = 0; ///< $n_{\text{Out1D}}^2$ nodes per element.
    size_t nOutTot_ = 0; ///< nElem_ * nOut2_.

    /// Output-node Vandermonde ($n_{\text{Out2}} \times N_{\text{modes}}$):
    /// the quadrature table V2D for the archival layout, or the extended
    /// table basis.vandermondeAt(extended tensor points).
    MatrixXr Vout_;

    /// Visualization sub-grid (connectivity built once per run).
    viz::VizTopology topo_;

    /// Physical output-node coordinates (contiguous per component).
    std::vector<Real> coordsX_, coordsY_;

    /// Reusable host buffers: modal coefficients, the projection scratch
    /// and the seven SoA fields.
    std::vector<Real> modal_;
    VectorXr nodal_;
    std::vector<Real> rho_, rhoU_, rhoV_, energy_, u_, v_, p_;
};
