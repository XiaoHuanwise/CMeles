/// @file CheckpointWriter.hpp
/// @brief Strategy B ("checkpoint") output: raw modal-coefficient
///        checkpoints plus a one-time mesh file, minimising the overhead
///        inside the time loop. Conversion to post-processing files is
///        done offline by convertCheckpoint / the CMelesConvert tool.

#pragma once

#include <string>
#include <vector>

#include "common/Types.hpp"
#include "io/OutputWriterBase.hpp"

class Config;
class DgField;

/// @brief Strategy B output writer: mesh.h5 + checkpoint_<step>.h5.
class CheckpointWriter : public OutputWriterBase {
public:
    /// @param cfg    Configuration (gamma, polynomial order, [output]).
    /// @param field  DG field supplying o_u() and the sizes
    ///               (lifetime must exceed this object).
    CheckpointWriter(const Config &cfg, const DgField &field);

    /// @brief Write the one-time mesh.h5 companion file holding the
    ///        reconstruction metadata and the quadrature-point coordinates.
    void writeMesh();

    void write(long step, Real time) override;

private:
    const DgField &field_;

    Real gamma_;
    int polynomialOrder_;
    int numPoints1D_;
    std::string elementType_;
    std::string directory_;

    int nElem_   = 0;
    int nModes_  = 0;
    int nq2_     = 0;
    size_t nTot_ = 0; ///< nElem_ * nq2_.

    /// Physical quadrature-point coordinates (contiguous per component).
    std::vector<Real> coordsX_, coordsY_;

    /// Reusable host buffers: raw modal coefficients [e][var][mode] and
    /// the per-variable (N_e, N_modes) row-major staging buffer.
    std::vector<Real> modal_;
    std::vector<Real> perVar_;
};
