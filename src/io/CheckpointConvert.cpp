/// @file CheckpointConvert.cpp
/// @brief Implementation of the offline checkpoint conversion.

#include "io/CheckpointConvert.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "basis/BasisFunctions1D.hpp"
#include "basis/BasisFunctions2D.hpp"
#include "common/Types.hpp"
#include "io/HDF5Writer.hpp"

namespace {
/// Conserved-variable names in the canonical order [rho, rho u, rho v, E].
const char *const kVarNames[4] = {"rho", "rho_u", "rho_v", "E"};

/// @brief Read a scalar attribute of type \p T.
template <typename T>
T readAttribute(const HighFive::File &file, const std::string &name) {
    return file.getAttribute(name).read<T>();
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
                       const std::string &outPath) {
    // Step 1: mesh metadata + quadrature-point coordinates.
    const HighFive::File mesh(meshPath, HighFive::File::ReadOnly);
    const int order   = readAttribute<int>(mesh, "order");
    const int nq      = readAttribute<int>(mesh, "N_q");
    const int nElem   = readAttribute<int>(mesh, "N_elem");
    const int nModes  = readAttribute<int>(mesh, "N_modes");
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
    const MatrixXr &V = basis2D.vandermonde();

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

    // Step 4: project every element and variable onto the quadrature
    // points ($u = V \hat u$) and derive the primitive variables
    // $u = \rho u / \rho$, $v = \rho v / \rho$,
    // $p = (\gamma - 1)(E - \tfrac{1}{2} \rho (u^2 + v^2))$.
    std::vector<Real> rho(nTot), rhoU(nTot), rhoV(nTot), energy(nTot);
    std::vector<Real> u(nTot), v(nTot), p(nTot);
    std::vector<Real> *soa[4] = {&rho, &rhoU, &rhoV, &energy};
    for (int e = 0; e < nElem; ++e) {
        for (int var = 0; var < 4; ++var) {
            const Eigen::Map<const VectorXr> uhat(
                modal[var].data() + static_cast<size_t>(e) * nModes, nModes);
            const VectorXr nodal = V * uhat;
            Real *dst = soa[var]->data() + static_cast<size_t>(e) *
                                               static_cast<size_t>(nq) *
                                               static_cast<size_t>(nq);
            for (int q = 0; q < nq * nq; ++q) {
                dst[q] = nodal[q];
            }
        }
    }
    for (size_t k = 0; k < nTot; ++k) {
        u[k] = rhoU[k] / rho[k];
        v[k] = rhoV[k] / rho[k];
        p[k] = (gamma - Real(1)) *
               (energy[k] - Real(0.5) * rho[k] * (u[k] * u[k] + v[k] * v[k]));
    }

    // Step 5: write the strategy-A-structured solution file.
    HDF5Writer writer(outPath);
    writer.writeAttribute("time", time);
    writer.writeAttribute("step", step);
    writer.writeAttribute("order", order);
    const std::vector<Real> *fields[] = {&rho, &rhoU, &rhoV, &energy,
                                         &u,   &v,    &p};
    const char *const names[] = {"rho", "rho_u", "rho_v", "E", "u", "v", "p"};
    for (int i = 0; i < 7; ++i) {
        writer.writeDataset(names[i], fields[i]->data(), nTot);
    }
    writer.writeGroupDataset("mesh", "x", coordsX.data(), nTot);
    writer.writeGroupDataset("mesh", "y", coordsY.data(), nTot);
    writer.flush();
}
