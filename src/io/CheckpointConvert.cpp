/// @file CheckpointConvert.cpp
/// @brief Implementation of the offline checkpoint conversion.

#include "io/CheckpointConvert.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "basis/BasisFunctions1D.hpp"
#include "basis/BasisFunctions2D.hpp"
#include "common/Types.hpp"
#include "io/HDF5Writer.hpp"
#include "io/VizSampling.hpp"
#include "io/VizWriter.hpp"

namespace {
/// Conserved-variable names in the canonical order [rho, rho u, rho v, E].
const char *const kVarNames[4] = {"rho", "rho_u", "rho_v", "E"};

/// @brief Read a scalar attribute of type \p T.
template <typename T>
T readAttribute(const HighFive::File &file, const std::string &name) {
    return file.getAttribute(name).read<T>();
}

/// @brief Read an optional scalar attribute of type \p T.
template <typename T>
bool readAttributeOpt(const HighFive::File &file, const std::string &name,
                      T &value) {
    if (!file.hasAttribute(name)) {
        return false;
    }
    value = file.getAttribute(name).read<T>();
    return true;
}

/// @brief Read a dataset into a flat row-major vector, checking its
///        element count against \p expectedCount. read_raw skips the
///        shape adaptation so 2-D datasets can land in the flat buffer.
std::vector<Real> readDataset(const HighFive::File &file,
                              const std::string &name, size_t expectedCount) {
    const HighFive::DataSet ds = file.getDataSet(name);
    const size_t count         = ds.getElementCount();
    if (count != expectedCount) {
        throw std::runtime_error("convertCheckpoint: dataset '" + name +
                                 "' has " + std::to_string(count) +
                                 " elements, expected " +
                                 std::to_string(expectedCount));
    }
    std::vector<Real> data(count);
    ds.read_raw(data.data());
    return data;
}
} // namespace

void convertCheckpoint(const std::string &meshPath,
                       const std::string &checkpointPath,
                       const std::string &outPath, OutputFormat format) {
    // Step 1: mesh metadata + quadrature-point coordinates.
    const HighFive::File mesh(meshPath, HighFive::File::ReadOnly);
    const int order      = readAttribute<int>(mesh, "order");
    const int nq         = readAttribute<int>(mesh, "N_q");
    const int nElem      = readAttribute<int>(mesh, "N_elem");
    const int nModes     = readAttribute<int>(mesh, "N_modes");
    std::string elemType = "quad";
    readAttributeOpt<std::string>(mesh, "element_type", elemType);
    const size_t nTot = static_cast<size_t>(nElem) * static_cast<size_t>(nq) *
                        static_cast<size_t>(nq);
    const std::vector<Real> coordsX = readDataset(mesh, "x", nTot);
    const std::vector<Real> coordsY = readDataset(mesh, "y", nTot);

    // Step 2: rebuild the basis and the Vandermonde matrix $V_{qm} =
    // \phi_m(\xi_q)$ from the polynomial order and quadrature order alone.
    const BasisFunctions1D basis1D(order, nq);
    const BasisFunctions2D basis2D(basis1D, order);
    if (basis2D.numBases() != nModes) {
        throw std::runtime_error(
            "convertCheckpoint: mesh.h5 N_modes does not match the "
            "reconstructed basis (order=" +
            std::to_string(order) + ")");
    }

    // Output node set, selected by the target format exactly as in
    // SolutionWriter: the visualization targets evaluate the DG polynomial
    // on the boundary-inclusive extended grid; the plain hdf5 target
    // keeps the interior Gauss points (archival layout).
    const bool extendedViz =
        formatWritesXdmf(format) || formatWritesTecplot(format);
    const MatrixX2r quadPts = basis2D.quadraturePoints();
    MatrixX2r outPts;
    MatrixXr Vout;
    int nOut1D = 0;
    if (extendedViz) {
        nOut1D = nq + 2;
        outPts = viz::tensorPoints(viz::extendedAbscissae(basis1D.points()));
        Vout   = basis2D.vandermondeAt(outPts);
    } else {
        nOut1D = nq;
        Vout   = basis2D.vandermonde();
    }
    const int nOut2 = nOut1D * nOut1D;
    const size_t nOutTot =
        static_cast<size_t>(nElem) * static_cast<size_t>(nOut2);

    // Output-node physical coordinates. mesh.h5 stores quadrature-point
    // coordinates but no vertices; the element map is bilinear and the
    // tensor Gauss points are unisolvent for $\{1, r, s, rs\}$, so a
    // least-squares fit recovers the map exactly (round-off only) and the
    // extended coordinates follow. Old checkpoints convert unchanged.
    MatrixX2r outXY(nOutTot, 2);
    if (extendedViz) {
        MatrixX2r quadXY(static_cast<Eigen::Index>(nTot), 2);
        for (size_t k = 0; k < nTot; ++k) {
            quadXY(static_cast<Eigen::Index>(k), 0) = coordsX[k];
            quadXY(static_cast<Eigen::Index>(k), 1) = coordsY[k];
        }
        outXY =
            viz::extendedCoordsFromQuadratureCoords(quadXY, quadPts, outPts);
    } else {
        for (size_t k = 0; k < nTot; ++k) {
            outXY(static_cast<Eigen::Index>(k), 0) = coordsX[k];
            outXY(static_cast<Eigen::Index>(k), 1) = coordsY[k];
        }
    }
    std::vector<Real> outX(nOutTot), outY(nOutTot);
    for (size_t k = 0; k < nOutTot; ++k) {
        outX[k] = outXY(static_cast<Eigen::Index>(k), 0);
        outY[k] = outXY(static_cast<Eigen::Index>(k), 1);
    }

    // Step 3: checkpoint modal coefficients, per variable (N_e, N_modes).
    const HighFive::File ckpt(checkpointPath, HighFive::File::ReadOnly);
    const Real time  = readAttribute<Real>(ckpt, "time");
    const int step   = readAttribute<int>(ckpt, "step");
    const Real gamma = readAttribute<Real>(ckpt, "gamma");
    const size_t nEvm =
        static_cast<size_t>(nElem) * static_cast<size_t>(nModes);
    std::vector<Real> modal[4];
    for (int var = 0; var < 4; ++var) {
        modal[var] = readDataset(ckpt, kVarNames[var], nEvm);
    }

    // Step 4: project every element and variable onto the output nodes
    // ($u = V \hat u$) and derive the primitive variables
    // $u = \rho u / \rho$, $v = \rho v / \rho$,
    // $p = (\gamma - 1)(E - \tfrac{1}{2} \rho (u^2 + v^2))$.
    // For the extended grid this evaluates each element's OWN polynomial
    // at its boundary-inclusive points (no cross-element interpolation).
    std::vector<Real> rho(nOutTot), rhoU(nOutTot), rhoV(nOutTot),
        energy(nOutTot);
    std::vector<Real> u(nOutTot), v(nOutTot), p(nOutTot);
    std::vector<Real> *soa[4] = {&rho, &rhoU, &rhoV, &energy};
    for (int e = 0; e < nElem; ++e) {
        for (int var = 0; var < 4; ++var) {
            const Eigen::Map<const VectorXr> uhat(
                modal[var].data() + static_cast<size_t>(e) * nModes, nModes);
            const VectorXr nodal = Vout * uhat;
            Real *dst = soa[var]->data() +
                        static_cast<size_t>(e) * static_cast<size_t>(nOut2);
            for (int q = 0; q < nOut2; ++q) {
                dst[q] = nodal[q];
            }
        }
    }
    for (size_t k = 0; k < nOutTot; ++k) {
        u[k] = rhoU[k] / rho[k];
        v[k] = rhoV[k] / rho[k];
        p[k] = (gamma - Real(1)) *
               (energy[k] - Real(0.5) * rho[k] * (u[k] * u[k] + v[k] * v[k]));
    }

    // Step 5: write the strategy-A-structured solution file plus the
    // requested visualization outputs (same layout as SolutionWriter).
    viz::VizTopology topo;
    if (extendedViz) {
        // The sub-grid connectivity is purely combinatorial: rebuild it
        // from N_elem and the element_type attribute (all-quad or
        // all-tri for the current structured generator; a future mixed
        // mesh input extends this with per-element flags).
        std::vector<int> isTriangle(static_cast<size_t>(nElem),
                                    elemType == "tri" ? 1 : 0);
        topo = viz::buildVizConnectivity(nElem, nOut1D, isTriangle);
    }

    if (formatWritesHdf5(format) || formatWritesXdmf(format)) {
        HDF5Writer writer(outPath);
        writer.writeAttribute("time", time);
        writer.writeAttribute("step", step);
        writer.writeAttribute("order", order);
        const std::vector<Real> *fields[] = {&rho, &rhoU, &rhoV, &energy,
                                             &u,   &v,    &p};
        const char *const names[]         = {"rho", "rho_u", "rho_v", "E",
                                             "u",   "v",     "p"};
        for (int i = 0; i < 7; ++i) {
            writer.writeDataset(names[i], fields[i]->data(), nOutTot);
        }
        writer.writeGroupDataset("mesh", "x", outX.data(), nOutTot);
        writer.writeGroupDataset("mesh", "y", outY.data(), nOutTot);
        if (formatWritesXdmf(format)) {
            viz::writeVizToH5(writer, topo, outX.data(), outY.data());
            writer.flush();
            const std::filesystem::path p(outPath);
            viz::writeXdmf(
                p.filename().string(),
                (p.parent_path() / (p.stem().string() + ".xmf")).string(), topo,
                time);
        }
        writer.flush();
    }

    if (formatWritesTecplot(format)) {
        viz::Fields f;
        f.x    = outX.data();
        f.y    = outY.data();
        f.rho  = rho.data();
        f.rhoU = rhoU.data();
        f.rhoV = rhoV.data();
        f.E    = energy.data();
        f.u    = u.data();
        f.v    = v.data();
        f.p    = p.data();
        const std::filesystem::path p(outPath);
        viz::writeTecplotPlt(
            (p.parent_path() / (p.stem().string() + ".plt")).string(), f, topo,
            time);
    }
}

OutputFormat convertReadMeshFormat(const std::string &meshPath) {
    const HighFive::File mesh(meshPath, HighFive::File::ReadOnly);
    std::string name;
    if (!readAttributeOpt<std::string>(mesh, "format", name)) {
        return OutputFormat::Hdf5;
    }
    return parseOutputFormat(name);
}
