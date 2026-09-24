/// @file VizWriter.cpp
/// @brief Implementation of the visualization output helpers.

#include "io/VizWriter.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "io/HDF5Writer.hpp"

namespace viz {

// ============================================================================
// Connectivity
// ============================================================================

VizTopology buildVizConnectivity(int N_elem, int nPts1D,
                                 const std::vector<int> &isTriangle) {
    if (static_cast<int>(isTriangle.size()) != N_elem) {
        throw std::invalid_argument(
            "buildVizConnectivity: isTriangle size does not match N_elem");
    }

    VizTopology topo;
    topo.nNodes = static_cast<long long>(N_elem) * nPts1D * nPts1D;
    // Worst-case size (all triangles): 4 ints per entry.
    topo.connect.reserve(static_cast<size_t>(N_elem) * (nPts1D - 1) *
                         (nPts1D - 1) * 4);

    int nTriElem = 0;
    for (int e = 0; e < N_elem; ++e) {
        nTriElem += (isTriangle[static_cast<size_t>(e)] != 0) ? 1 : 0;
    }
    // Single-type meshes emit a PLAIN id table (uniform topology, the
    // most compatible reader path); only genuinely mixed meshes need the
    // per-cell type prefixes.
    topo.mixed    = (nTriElem != 0 && nTriElem != N_elem);
    topo.cellType = (nTriElem == 0) ? 5 : 4;

    const bool emitType = topo.mixed;
    for (int e = 0; e < N_elem; ++e) {
        const int base = e * nPts1D * nPts1D;
        for (int j = 0; j + 1 < nPts1D; ++j) {
            for (int i = 0; i + 1 < nPts1D; ++i) {
                // Tensor sub-cell corners in the local node grid:
                // n1 = (i, j), n2 = (i+1, j), n3 = (i+1, j+1), n4 = (i, j+1).
                const int n1 = base + j * nPts1D + i;
                const int n2 = base + j * nPts1D + i + 1;
                const int n3 = base + (j + 1) * nPts1D + i + 1;
                const int n4 = base + (j + 1) * nPts1D + i;
                if (isTriangle[static_cast<size_t>(e)] != 0) {
                    // Triangle element: split along the fixed diagonal
                    // n1-n3 (two triangles per sub-cell). The s = +1 node
                    // row collapses onto the apex, so the last row strip
                    // contains zero-area triangles; renderers skip them
                    // and the counts stay deterministic.
                    if (emitType) {
                        topo.connect.push_back(4);
                    }
                    topo.connect.push_back(n1);
                    topo.connect.push_back(n2);
                    topo.connect.push_back(n3);
                    if (emitType) {
                        topo.connect.push_back(4);
                    }
                    topo.connect.push_back(n1);
                    topo.connect.push_back(n3);
                    topo.connect.push_back(n4);
                    topo.nCells += 2;
                } else {
                    if (emitType) {
                        topo.connect.push_back(5);
                    }
                    topo.connect.push_back(n1);
                    topo.connect.push_back(n2);
                    topo.connect.push_back(n3);
                    topo.connect.push_back(n4);
                    topo.nCells += 1;
                }
            }
        }
    }
    return topo;
}

void writeVizToH5(HDF5Writer &writer, const VizTopology &topo, const Real *x,
                  const Real *y) {
    writer.writeGroupDatasetInt("mesh", "connect", topo.connect.data(),
                                topo.connect.size());
    // Interleaved [N][3] coordinates (z = 0) for the XYZ XDMF geometry.
    std::vector<Real> xyz(static_cast<size_t>(topo.nNodes) * 3);
    for (long long k = 0; k < topo.nNodes; ++k) {
        xyz[static_cast<size_t>(k) * 3 + 0] = x[static_cast<size_t>(k)];
        xyz[static_cast<size_t>(k) * 3 + 1] = y[static_cast<size_t>(k)];
        xyz[static_cast<size_t>(k) * 3 + 2] = Real(0);
    }
    writer.writeGroupDataset2D("mesh", "xyz", xyz.data(),
                               static_cast<size_t>(topo.nNodes), 3);
    writer.writeAttribute("viz_nodes", static_cast<int>(topo.nNodes));
    writer.writeAttribute("viz_cells", static_cast<int>(topo.nCells));
}

// ============================================================================
// XDMF sidecar
// ============================================================================

namespace {
/// @brief Real precision as the XDMF attribute digits (8/4).
constexpr int kPrecision = (sizeof(Real) == 8) ? 8 : 4;
} // namespace

void writeXdmf(const std::string &h5Basename, const std::string &xmfPath,
               const VizTopology &topo, Real time) {
    std::ofstream out(xmfPath);
    if (!out) {
        throw std::runtime_error("writeXdmf: cannot open " + xmfPath);
    }
    char tbuf[32];
    std::snprintf(tbuf, sizeof(tbuf), "%.12g", static_cast<double>(time));

    const char *const attrs[7] = {"rho", "rho_u", "rho_v", "E", "u", "v", "p"};
    // Uniform topology for single-type meshes (QUADRILATERAL / TRIANGLE)
    // — the most widely supported reader path; Mixed only for genuinely
    // mixed meshes. No DOCTYPE: some XML stacks try to resolve the
    // referenced DTD.
    const char *topoType =
        topo.mixed ? "Mixed"
                   : (topo.cellType == 5 ? "QUADRILATERAL" : "TRIANGLE");
    out << "<?xml version=\"1.0\" ?>\n"
        << "<Xdmf Version=\"2.0\">\n"
        << "  <Domain Name=\"CMeles\">\n"
        << "    <Grid Name=\"solution\" GridType=\"Uniform\">\n"
        << "      <Time Value=\"" << tbuf << "\"/>\n"
        << "      <Topology TopologyType=\"" << topoType
        << "\" NumberOfElements=\"" << topo.nCells << "\">\n"
        << "        <DataItem Dimensions=\"" << topo.connect.size()
        << "\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n"
        << "          " << h5Basename << ":/mesh/connect\n"
        << "        </DataItem>\n"
        << "      </Topology>\n"
        << "      <Geometry GeometryType=\"XYZ\">\n"
        << "        <DataItem Dimensions=\"" << topo.nNodes
        << " 3\" NumberType=\"Float\" Precision=\"" << kPrecision
        << "\" Format=\"HDF\">\n"
        << "          " << h5Basename << ":/mesh/xyz\n"
        << "        </DataItem>\n"
        << "      </Geometry>\n";
    for (int i = 0; i < 7; ++i) {
        out << "      <Attribute Name=\"" << attrs[i]
            << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
            << "        <DataItem Dimensions=\"" << topo.nNodes
            << "\" NumberType=\"Float\" Precision=\"" << kPrecision
            << "\" Format=\"HDF\">\n"
            << "          " << h5Basename << ":/" << attrs[i] << "\n"
            << "        </DataItem>\n"
            << "      </Attribute>\n";
    }
    out << "    </Grid>\n"
        << "  </Domain>\n"
        << "</Xdmf>\n";
}

// ============================================================================
// Classic Tecplot binary PLT (V101 layout)
// ============================================================================

namespace {
/// @brief Raw little-endian block writers (x86 host byte order — the same
///        assumption the HDF5 default makes on this platform).
void putInt32(std::ofstream &f, int32_t v) {
    f.write(reinterpret_cast<const char *>(&v), 4);
}
void putFloat32(std::ofstream &f, float v) {
    f.write(reinterpret_cast<const char *>(&v), 4);
}
void putBlock(std::ofstream &f, const Real *data, size_t n) {
    f.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(n) * sizeof(Real));
}

/// @brief Tecplot string convention: one INT4 per character + a 0.
void putPltString(std::ofstream &f, const std::string &s) {
    for (char c : s) {
        putInt32(f, static_cast<int32_t>(static_cast<unsigned char>(c)));
    }
    putInt32(f, 0);
}
} // namespace

void writeTecplotPlt(const std::string &pltPath, const Fields &f,
                     const VizTopology &topo, Real time) {
    std::ofstream out(pltPath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("writeTecplotPlt: cannot open " + pltPath);
    }

    const auto nNodes    = static_cast<size_t>(topo.nNodes);
    const int formatCode = (sizeof(Real) == 8) ? 2 : 1;

    // --- File header ---
    const char magic[8] = {'#', '!', 'T', 'D', 'V', '1', '0', '1'};
    out.write(magic, 8);
    putInt32(out, 1); // Byte-order check.
    putPltString(out, "CMeles solution");
    putInt32(out, 9); // Number of variables.
    const char *const varNames[9] = {"x", "y", "rho", "rho_u", "rho_v",
                                     "E", "u", "v",   "p"};
    for (const char *name : varNames) {
        putPltString(out, name);
    }

    // --- Zone header (single finite-element zone, block packing) ---
    // Field order for V101 as consumed by ParaView's "VisIt Tecplot Binary
    // Reader" (VisIt databases/Tecplot/TecplotFile.C) and legacy Tecplot:
    // marker, name, parent zone, zone type, variable locations, face
    // neighbours, FE counts. There is NO data-packing field for version
    // <= 102 — readers assume block packing — and version <= 102 readers
    // consume the parent-zone int as an "unused -1"; writing an extra
    // packing int shifts every later field (numPts would land on the
    // face-neighbour flag and the zone degenerates to zero nodes).
    // The classic FE zone admits a single cell type: all-quad and
    // all-triangle meshes get their native zone type; genuinely mixed
    // meshes use FEQUADRILATERAL with triangles folded to a repeated
    // node $(n_1, n_2, n_3, n_3)$.
    const int zoneType   = topo.mixed || topo.cellType == 5 ? 3 : 2;
    const int idsPerCell = (zoneType == 3) ? 4 : 3;
    char zname[64];
    std::snprintf(zname, sizeof(zname), "solution, t = %.12g",
                  static_cast<double>(time));
    putFloat32(out, 299.0f); // Zone marker.
    putPltString(out, zname);
    putInt32(out, -1); // Parent zone: none.
    putInt32(out, zoneType);
    putInt32(out, 0); // Variable locations: all nodal.
    putInt32(out, 0); // Face-neighbour connections: none.
    putInt32(out, static_cast<int32_t>(topo.nNodes));
    putInt32(out, static_cast<int32_t>(topo.nCells));
    putInt32(out, 0);        // iCellDim (unused).
    putInt32(out, 0);        // jCellDim (unused).
    putInt32(out, 0);        // kCellDim (unused).
    putInt32(out, 0);        // End of auxiliary name/value pairs.
    putFloat32(out, 357.0f); // End-of-header marker.

    // --- Data section ---
    putFloat32(out, 299.0f); // Zone data marker.
    for (int v = 0; v < 9; ++v) {
        putInt32(out, formatCode);
    }
    putInt32(out, 0);  // Variable sharing: none.
    putInt32(out, -1); // Connectivity sharing: none.

    const Real *blocks[9] = {f.x, f.y, f.rho, f.rhoU, f.rhoV,
                             f.E, f.u, f.v,   f.p};
    for (const Real *b : blocks) {
        if (b == nullptr) {
            throw std::runtime_error("writeTecplotPlt: missing field");
        }
        putBlock(out, b, nNodes);
    }

    // --- Connectivity (1-based INT4 cell node ids). Plain table for
    //     single-type meshes (4 ids per quad, 3 per triangle); [type,
    //     ids...] decoding for mixed meshes with triangles folded to a
    //     repeated node. ---
    if (!topo.mixed) {
        for (long long c = 0; c < topo.nCells; ++c) {
            const size_t base =
                static_cast<size_t>(c) * static_cast<size_t>(idsPerCell);
            for (int j = 0; j < idsPerCell; ++j) {
                putInt32(out, topo.connect[base + j] + 1);
            }
        }
    } else {
        size_t i = 0;
        while (i < topo.connect.size()) {
            const int type = topo.connect[i++];
            if (type == 5) {
                putInt32(out, topo.connect[i] + 1);
                putInt32(out, topo.connect[i + 1] + 1);
                putInt32(out, topo.connect[i + 2] + 1);
                putInt32(out, topo.connect[i + 3] + 1);
                i += 4;
            } else if (type == 4) {
                const int32_t a = topo.connect[i] + 1;
                const int32_t b = topo.connect[i + 1] + 1;
                const int32_t c = topo.connect[i + 2] + 1;
                putInt32(out, a);
                putInt32(out, b);
                putInt32(out, c);
                putInt32(out, c); // Repeated node: triangle as collapsed quad.
                i += 3;
            } else {
                throw std::runtime_error(
                    "writeTecplotPlt: unknown cell type in connectivity");
            }
        }
    }
    out.flush();
    if (!out) {
        throw std::runtime_error("writeTecplotPlt: write failure on " +
                                 pltPath);
    }
}

} // namespace viz
