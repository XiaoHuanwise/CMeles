/// @file SolutionWriter.cpp
/// @brief Implementation of the strategy A output writer.

#include "io/SolutionWriter.hpp"

#include <cstdio>
#include <filesystem>

#include "basis/BasisFunctions1D.hpp"
#include "basis/BasisFunctions2D.hpp"
#include "common/Constants.hpp"
#include "config/Config.hpp"
#include "dg/DgField.hpp"
#include "dg/QuadratureCoords.hpp"
#include "io/HDF5Writer.hpp"
#include "io/VizSampling.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/MeshGeometry.hpp"

namespace {
/// @brief Zero-padded solution file name with the given extension:
///        <prefix>_solution_<step><ext> (legacy solution_<step> when the
///        Config was not built from a file and has no prefix).
std::string solutionFileName(const std::string &prefix, long step,
                             const char *ext) {
    char buf[128];
    const int n =
        prefix.empty()
            ? std::snprintf(buf, sizeof(buf), "solution_%07ld%s", step, ext)
            : std::snprintf(buf, sizeof(buf), "%s_solution_%07ld%s",
                            prefix.c_str(), step, ext);
    if (n < 0 || n >= static_cast<int>(sizeof(buf))) {
        throw std::runtime_error("solutionFileName: name truncated");
    }
    return std::string(buf);
}
} // namespace

SolutionWriter::SolutionWriter(const Config &cfg, const DgField &field)
    : field_(field), gamma_(cfg.gamma()),
      polynomialOrder_(cfg.polynomialOrder()),
      directory_(cfg.outputDirectory()), format_(cfg.outputFormat()),
      prefix_(cfg.outputPrefix()),
      seriesEnabled_(cfg.outputSeries() && formatWritesXdmf(format_)) {
    nElem_  = field.numElements();
    nModes_ = field.numModes();

    // Node-set selection by output format (see the class comment): the
    // visualization formats evaluate the DG polynomial on the
    // boundary-inclusive extended grid so each element's sub-grid closes
    // up to its interfaces; the plain archive keeps the interior Gauss
    // points.
    const bool extendedViz =
        formatWritesXdmf(format_) || formatWritesTecplot(format_);
    MatrixX2r outPts;
    if (extendedViz) {
        nOut1D_ = field.numPoints1D() + 2;
        outPts  = viz::tensorPoints(
            viz::extendedAbscissae(field.basis().basis1D().points()));
        Vout_ = field.basis().vandermondeAt(outPts);
    } else {
        nOut1D_ = field.numPoints1D();
        Vout_   = field.basis().vandermonde();
    }
    nOut2_   = nOut1D_ * nOut1D_;
    nOutTot_ = static_cast<size_t>(nElem_) * static_cast<size_t>(nOut2_);

    // Visualization sub-grid: per-element-type adaptive connectivity over
    // the output node grid (see VizWriter.hpp).
    const Mesh &mesh = field.mesh();
    std::vector<int> isTriangle(static_cast<size_t>(nElem_));
    for (int e = 0; e < nElem_; ++e) {
        isTriangle[static_cast<size_t>(e)] = mesh.isTriangle(e) ? 1 : 0;
    }
    if (extendedViz) {
        topo_ = viz::buildVizConnectivity(nElem_, nOut1D_, isTriangle);
    }

    modal_.assign(static_cast<size_t>(nElem_) * kNumVars2D *
                      static_cast<size_t>(nModes_),
                  Real(0));
    nodal_.resize(nOut2_);
    for (auto *buf : {&rho_, &rhoU_, &rhoV_, &energy_, &u_, &v_, &p_}) {
        buf->assign(nOutTot_, Real(0));
    }

    // Contiguous per-component coordinates (the MatrixX2r columns of the
    // map output are strided and not directly writable).
    const MatrixX2r xy =
        extendedViz
            ? bilinearPhysicalCoords(field.geometry().vertices(), outPts)
            : quadraturePhysicalCoords(field);
    coordsX_.resize(nOutTot_);
    coordsY_.resize(nOutTot_);
    for (size_t k = 0; k < nOutTot_; ++k) {
        coordsX_[k] = xy(static_cast<Eigen::Index>(k), 0);
        coordsY_[k] = xy(static_cast<Eigen::Index>(k), 1);
    }
}

void SolutionWriter::write(long step, Real time) {
    // Device -> host copy. Plain occa::memory::copyTo is used on purpose:
    // DeviceMemoryManager::copyToHost is a no-op on unified-memory
    // backends and would leave the buffer untouched there.
    field_.o_u().copyTo(modal_.data(), static_cast<occa::dim_t>(modal_.size()));

    // Modal -> output-node projection per element and variable:
    // nodal = V * u_hat with V the (nOut2_ x N_modes) output Vandermonde.
    // For the extended grid this evaluates the element's OWN polynomial
    // at the boundary-inclusive points (no cross-element interpolation).
    std::vector<Real> *conserved[kNumVars2D] = {&rho_, &rhoU_, &rhoV_,
                                                &energy_};
    for (int e = 0; e < nElem_; ++e) {
        for (int var = 0; var < kNumVars2D; ++var) {
            const Eigen::Map<const VectorXr> uhat(
                modal_.data() + (static_cast<size_t>(e) * kNumVars2D +
                                 static_cast<size_t>(var)) *
                                    static_cast<size_t>(nModes_),
                nModes_);
            nodal_.noalias() = Vout_ * uhat;
            Real *dst =
                conserved[var]->data() + static_cast<size_t>(e) * nOut2_;
            for (int q = 0; q < nOut2_; ++q) {
                dst[q] = nodal_[q];
            }
        }
    }

    // Primitive variables from the conserved ones:
    // $u = \rho u / \rho$, $v = \rho v / \rho$, $p = (\gamma - 1)(E - \tfrac{1}{2} \rho (u^2 + v^2))$.
    for (size_t k = 0; k < nOutTot_; ++k) {
        u_[k] = rhoU_[k] / rho_[k];
        v_[k] = rhoV_[k] / rho_[k];
        p_[k] = (gamma_ - Real(1)) *
                (energy_[k] -
                 Real(0.5) * rho_[k] * (u_[k] * u_[k] + v_[k] * v_[k]));
    }

    // --- HDF5 archive (plain or with the XDMF sidecar's heavy data). ---
    if (formatWritesHdf5(format_) || formatWritesXdmf(format_)) {
        const std::filesystem::path path =
            std::filesystem::path(directory_) /
            solutionFileName(prefix_, step, ".h5");
        HDF5Writer writer(path.string());
        writer.writeAttribute("time", time);
        writer.writeAttribute("step", static_cast<int>(step));
        writer.writeAttribute("order", polynomialOrder_);
        const std::vector<Real> *soa[] = {&rho_, &rhoU_, &rhoV_, &energy_,
                                          &u_,   &v_,    &p_};
        const char *const names[]      = {"rho", "rho_u", "rho_v", "E",
                                          "u",   "v",     "p"};
        for (int i = 0; i < 7; ++i) {
            writer.writeDataset(names[i], soa[i]->data(), nOutTot_);
        }
        writer.writeGroupDataset("mesh", "x", coordsX_.data(), nOutTot_);
        writer.writeGroupDataset("mesh", "y", coordsY_.data(), nOutTot_);
        if (formatWritesXdmf(format_)) {
            viz::writeVizToH5(writer, topo_, coordsX_.data(), coordsY_.data());
            writer.flush();
            viz::writeXdmf(path.filename().string(),
                           (std::filesystem::path(directory_) /
                            solutionFileName(prefix_, step, ".xmf"))
                               .string(),
                           topo_, time);
            // Incrementally maintain the temporal-collection index:
            // append this step and regenerate the (small) index file —
            // the supported way to open the run as a time series in
            // ParaView (see writeXdmfSeries).
            if (seriesEnabled_) {
                seriesSteps_.push_back({path.filename().string(), time});
                viz::writeXdmfSeries(
                    (std::filesystem::path(directory_) / seriesFileName())
                        .string(),
                    seriesSteps_, topo_);
            }
        }
        writer.flush();
    }

    // --- Classic Tecplot binary PLT. ---
    if (formatWritesTecplot(format_)) {
        viz::Fields f;
        f.x    = coordsX_.data();
        f.y    = coordsY_.data();
        f.rho  = rho_.data();
        f.rhoU = rhoU_.data();
        f.rhoV = rhoV_.data();
        f.E    = energy_.data();
        f.u    = u_.data();
        f.v    = v_.data();
        f.p    = p_.data();
        viz::writeTecplotPlt((std::filesystem::path(directory_) /
                              solutionFileName(prefix_, step, ".plt"))
                                 .string(),
                             f, topo_, time);
    }
}
