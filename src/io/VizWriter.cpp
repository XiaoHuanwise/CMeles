/// @file VizWriter.cpp
/// @brief Implementation of the visualization output helpers.

#include "io/VizWriter.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
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

/// @brief The seven nodal attribute names (dataset names in the h5).
constexpr const char *const kAttrs[7] = {"rho", "rho_u", "rho_v", "E",
                                         "u",   "v",     "p"};

/// @brief Emit one uniform-grid step (Time + Topology + Geometry + the
///        seven nodal attributes referencing the paired h5).
///
/// Shared by the per-step sidecar and the temporal-collection index so
/// both stay byte-identical in their grid blocks.
void writeUniformGridBlock(std::ostream &out, const std::string &gridName,
                           const std::string &h5Basename,
                           const VizTopology &topo, Real time) {
    char tbuf[32];
    std::snprintf(tbuf, sizeof(tbuf), "%.12g", static_cast<double>(time));
    // Uniform topology for single-type meshes (QUADRILATERAL / TRIANGLE)
    // — the most widely supported reader path; Mixed only for genuinely
    // mixed meshes. No DOCTYPE: some XML stacks try to resolve the
    // referenced DTD.
    const char *topoType =
        topo.mixed ? "Mixed"
                   : (topo.cellType == 5 ? "QUADRILATERAL" : "TRIANGLE");
    out << "  <Grid Name=\"" << gridName << "\" GridType=\"Uniform\">\n"
        << "    <Time Value=\"" << tbuf << "\"/>\n"
        << "    <Topology TopologyType=\"" << topoType
        << "\" NumberOfElements=\"" << topo.nCells << "\">\n"
        << "      <DataItem Dimensions=\"" << topo.connect.size()
        << "\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n"
        << "        " << h5Basename << ":/mesh/connect\n"
        << "      </DataItem>\n"
        << "    </Topology>\n"
        << "    <Geometry GeometryType=\"XYZ\">\n"
        << "      <DataItem Dimensions=\"" << topo.nNodes
        << " 3\" NumberType=\"Float\" Precision=\"" << kPrecision
        << "\" Format=\"HDF\">\n"
        << "        " << h5Basename << ":/mesh/xyz\n"
        << "      </DataItem>\n"
        << "    </Geometry>\n";
    for (int i = 0; i < 7; ++i) {
        out << "    <Attribute Name=\"" << kAttrs[i]
            << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
            << "      <DataItem Dimensions=\"" << topo.nNodes
            << "\" NumberType=\"Float\" Precision=\"" << kPrecision
            << "\" Format=\"HDF\">\n"
            << "        " << h5Basename << ":/" << kAttrs[i] << "\n"
            << "      </DataItem>\n"
            << "    </Attribute>\n";
    }
    out << "  </Grid>\n";
}
} // namespace

void writeXdmf(const std::string &h5Basename, const std::string &xmfPath,
               const VizTopology &topo, Real time) {
    std::ofstream out(xmfPath);
    if (!out) {
        throw std::runtime_error("writeXdmf: cannot open " + xmfPath);
    }
    out << "<?xml version=\"1.0\" ?>\n"
        << "<Xdmf Version=\"2.0\">\n"
        << "  <Domain Name=\"CMeles\">\n";
    writeUniformGridBlock(out, "solution", h5Basename, topo, time);
    out << "  </Domain>\n"
        << "</Xdmf>\n";
}

void writeXdmfSeries(const std::string &seriesPath,
                     const std::vector<SeriesEntry> &entries,
                     const VizTopology &topo) {
    std::ofstream out(seriesPath);
    if (!out) {
        throw std::runtime_error("writeXdmfSeries: cannot open " + seriesPath);
    }
    out << "<?xml version=\"1.0\" ?>\n"
        << "<Xdmf Version=\"2.0\">\n"
        << "  <Domain Name=\"CMeles\">\n"
        << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" "
           "CollectionType=\"Temporal\">\n";
    for (const SeriesEntry &e : entries) {
        // Indent the shared grid block one level deeper inside the
        // collection wrapper.
        std::ostringstream block;
        writeUniformGridBlock(block, "step_" + e.h5Basename, e.h5Basename, topo,
                              e.time);
        std::istringstream lines(block.str());
        std::string line;
        while (std::getline(lines, line)) {
            out << "  " << line << "\n";
        }
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
void putFloat64(std::ofstream &f, double v) {
    f.write(reinterpret_cast<const char *>(&v), 8);
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
    const char magic[8] = {'#', '!', 'T', 'D', 'V', '1', '1', '2'};
    out.write(magic, 8);
    putInt32(out, 1); // Byte-order check.
    // V111+: Int32 FileType (0 = FULL) precedes the title. Omitting it
    // shifts the whole header stream: Tecplot 360 rejects the file with
    // "Invalid header in binary datafile" and the ParaView readers fail
    // to resync the variable names.
    putInt32(out, 0); // FileType: FULL.
    putPltString(out, "CMeles solution");
    putInt32(out, 9); // Number of variables.
    const char *const varNames[9] = {"x", "y", "rho", "rho_u", "rho_v",
                                     "E", "u", "v",   "p"};
    for (const char *name : varNames) {
        putPltString(out, name);
    }

    // --- Zone header (single finite-element zone, block packing) ---
    // V112 field order as consumed by ParaView's "VisIt Tecplot Binary
    // Reader" (VisIt databases/Tecplot/TecplotFile.C) and Tecplot 360:
    // marker, name, parent zone, StrandID, SolutionTime (FLOAT64), zone
    // color, zone type, variable locations, face neighbours,
    // user-defined neighbour connections, FE counts. V112 has NO
    // data-packing field (readers assume block packing). StrandID = 1
    // marks the zone transient, so a file series carries real time
    // metadata. The V101 form drops parent/strand/time/color and the
    // user-defined-connection field; readers consume the leading -1 as
    // an unused int (writing it twice, or adding a packing int, shifts
    // every later field and degenerates the zone). The classic FE zone
    // admits a single cell type: all-quad and all-triangle meshes get
    // their native zone type; genuinely mixed meshes use
    // FEQUADRILATERAL with triangles folded to a repeated node
    // $(n_1, n_2, n_3, n_3)$.
    const int zoneType   = topo.mixed || topo.cellType == 5 ? 3 : 2;
    const int idsPerCell = (zoneType == 3) ? 4 : 3;
    char zname[64];
    std::snprintf(zname, sizeof(zname), "solution, t = %.12g",
                  static_cast<double>(time));
    putFloat32(out, 299.0f); // Zone marker.
    putPltString(out, zname);
    putInt32(out, -1);                          // Parent zone: none.
    putInt32(out, 1);                           // StrandID: transient.
    putFloat64(out, static_cast<double>(time)); // Solution time.
    putInt32(out, -1);                          // Zone color: none.
    putInt32(out, zoneType);
    putInt32(out, 0); // Variable locations: all nodal.
    putInt32(out, 0); // Face-neighbour connections: none.
    putInt32(out, 0); // User-defined neighbour connections: none.
    putInt32(out, static_cast<int32_t>(topo.nNodes));
    putInt32(out, static_cast<int32_t>(topo.nCells));
    putInt32(out, 0);        // iCellDim (unused).
    putInt32(out, 0);        // jCellDim (unused).
    putInt32(out, 0);        // kCellDim (unused).
    putInt32(out, 0);        // End of auxiliary name/value pairs.
    putFloat32(out, 357.0f); // End-of-header marker.

    // --- Data section (version >= 103 form: the passive/sharing flags
    //     and per-variable min/max pairs precede the blocks) ---
    putFloat32(out, 299.0f); // Zone data marker.
    for (int v = 0; v < 9; ++v) {
        putInt32(out, formatCode);
    }
    putInt32(out, 0);  // Passive variables: none.
    putInt32(out, 0);  // Variable sharing: none.
    putInt32(out, -1); // Connectivity sharing: none.

    const Real *blocks[9] = {f.x, f.y, f.rho, f.rhoU, f.rhoV,
                             f.E, f.u, f.v,   f.p};
    for (const Real *b : blocks) {
        if (b == nullptr) {
            throw std::runtime_error("writeTecplotPlt: missing field");
        }
    }
    // Per-variable min/max pairs (metadata consumed by readers; the
    // values themselves are honest extrema of the blocks below).
    for (const Real *b : blocks) {
        Real lo = b[0];
        Real hi = b[0];
        for (size_t k = 1; k < nNodes; ++k) {
            lo = b[k] < lo ? b[k] : lo;
            hi = b[k] > hi ? b[k] : hi;
        }
        putFloat64(out, static_cast<double>(lo));
        putFloat64(out, static_cast<double>(hi));
    }

    for (const Real *b : blocks) {
        putBlock(out, b, nNodes);
    }

    // --- Connectivity (INT4 cell node ids, 0-based — the base the
    //     Tecplot Data Format Guide prescribes for V103+ files; V102 and
    //     earlier files were 1-based, and ASCII .dat stays 1-based at
    //     every version, so mixing the two conventions silently
    //     displaces every cell by one node in Tecplot 360 and ParaView).
    //     Plain table for single-type meshes (4 ids per quad, 3 per
    //     triangle); [type, ids...] decoding for mixed meshes with
    //     triangles folded to a repeated node. ---
    if (!topo.mixed) {
        for (long long c = 0; c < topo.nCells; ++c) {
            const size_t base =
                static_cast<size_t>(c) * static_cast<size_t>(idsPerCell);
            for (int j = 0; j < idsPerCell; ++j) {
                putInt32(out, topo.connect[base + j]);
            }
        }
    } else {
        size_t i = 0;
        while (i < topo.connect.size()) {
            const int type = topo.connect[i++];
            if (type == 5) {
                putInt32(out, topo.connect[i]);
                putInt32(out, topo.connect[i + 1]);
                putInt32(out, topo.connect[i + 2]);
                putInt32(out, topo.connect[i + 3]);
                i += 4;
            } else if (type == 4) {
                const int32_t a = topo.connect[i];
                const int32_t b = topo.connect[i + 1];
                const int32_t c = topo.connect[i + 2];
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

// ============================================================================
// VTK XML UnstructuredGrid (.vtu) with high-order Lagrange cells
// ============================================================================

std::vector<Real> vtkLagrangeParametricPoints(int order, bool triangle) {
    if (order < 1) {
        throw std::invalid_argument(
            "vtkLagrangeParametricPoints: order must be >= 1");
    }
    std::vector<Real> uv;
    if (!triangle) {
        // Tensor lattice, permuted into the VTK id order (corners, then
        // bottom/right/top/left edge interiors, then the row-major
        // interior block) — the inverse of the forward map
        // vtkHigherOrderQuadrilateral::PointIndexFromIJK, tabulated here
        // by evaluating that map over the whole lattice.
        const int n = order + 1;
        std::vector<int> id(static_cast<size_t>(n) * n, -1);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const bool ibdy = (i == 0 || i == order);
                const bool jbdy = (j == 0 || j == order);
                const int nbdy  = (ibdy ? 1 : 0) + (jbdy ? 1 : 0);
                int idx;
                if (nbdy == 2) {
                    idx = (i ? (j ? 2 : 1) : (j ? 3 : 0));
                } else if (nbdy == 1) {
                    if (!ibdy) {
                        // Bottom (j = 0) or top (j = order) edge, i increasing.
                        idx = 4 + (i - 1) + (j ? 2 * (order - 1) : 0);
                    } else {
                        // Right (i = order) or left (i = 0) edge, j increasing.
                        idx = 4 + (j - 1) + (i ? (order - 1) : 3 * (order - 1));
                    }
                } else {
                    idx = 4 * order + (i - 1) + (order - 1) * (j - 1);
                }
                id[static_cast<size_t>(idx)] = i + j * n;
            }
        }
        uv.resize(static_cast<size_t>(2) * n * n);
        for (int k = 0; k < n * n; ++k) {
            const int p                        = id[static_cast<size_t>(k)];
            uv[static_cast<size_t>(2) * k + 0] = Real(p % n) / Real(order);
            uv[static_cast<size_t>(2) * k + 1] = Real(p / n) / Real(order);
        }
        return uv;
    }

    // Triangle: the concentric-ring enumeration of
    // vtkHigherOrderTriangle::SetParametricCoords — per ring (outer to
    // inner): the 3 vertices, then the 3 edge interiors, each edge
    // walking from its (min,min,max) corner while the max component
    // decreases.
    const int nPts = (order + 1) * (order + 2) / 2;
    uv.reserve(static_cast<size_t>(nPts) * 2);
    int mx     = order;
    int mn     = 0;
    double ord = order;
    for (int ring = order; ring > 0; ring -= 3) {
        const Real mnO = Real(mn) / ord;
        const Real mxO = Real(mx) / ord;
        // Vertices: (mn,mn), (mx,mn), (mn,mx).
        uv.push_back(mnO);
        uv.push_back(mnO);
        uv.push_back(mxO);
        uv.push_back(mnO);
        uv.push_back(mnO);
        uv.push_back(mxO);
        if (ring > 1) {
            // Edge interiors: component `dim` grows from min/order while
            // component (dim+2)%3 shrinks from max/order.
            for (int dim = 0; dim < 3; ++dim) {
                Real p[3]        = {mnO, mnO, mnO};
                p[(dim + 2) % 3] = mxO;
                for (int s = 0; s < ring - 1; ++s) {
                    p[dim] += Real(1) / ord;
                    p[(dim + 2) % 3] -= Real(1) / ord;
                    uv.push_back(p[0]);
                    uv.push_back(p[1]);
                }
            }
        }
        mx -= 2;
        mn += 1;
    }
    if (static_cast<int>(ord) % 3 == 0 && order >= 3) {
        // Even ring count leaves one centroid point (ord hits 0 in the
        // loop above only when order % 3 == 0; SetParametricCoords then
        // appends (min/order, min/order)).
        const Real mnO = Real(mn) / ord;
        uv.push_back(mnO);
        uv.push_back(mnO);
    }
    if (static_cast<int>(uv.size()) != 2 * nPts) {
        throw std::logic_error(
            "vtkLagrangeParametricPoints: triangle point count mismatch (" +
            std::to_string(uv.size() / 2) + " vs " + std::to_string(nPts) +
            ")");
    }
    return uv;
}

void writePvdSeries(const std::string &pvdPath,
                    const std::vector<SeriesEntry> &entries) {
    std::ofstream out(pvdPath);
    if (!out) {
        throw std::runtime_error("writePvdSeries: cannot open " + pvdPath);
    }
    out << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"Collection\" version=\"0.1\">\n"
        << "  <Collection>\n";
    for (const SeriesEntry &e : entries) {
        char tbuf[32];
        std::snprintf(tbuf, sizeof(tbuf), "%.12g", static_cast<double>(e.time));
        out << "    <DataSet timestep=\"" << tbuf << "\" part=\"0\" file=\""
            << e.h5Basename << "\"/>\n";
    }
    out << "  </Collection>\n"
        << "</VTKFile>\n";
    out.flush();
    if (!out) {
        throw std::runtime_error("writePvdSeries: write failure on " + pvdPath);
    }
}

void writeVtu(const std::string &vtuPath, const Fields &f, int order, int nElem,
              const std::vector<int> &isTriangle) {
    if (order < 1) {
        throw std::invalid_argument("writeVtu: order must be >= 1");
    }
    if (static_cast<int>(isTriangle.size()) != nElem) {
        throw std::invalid_argument(
            "writeVtu: isTriangle size does not match nElem");
    }
    const Real *blocks[9] = {f.x, f.y, f.rho, f.rhoU, f.rhoV,
                             f.E, f.u, f.v,   f.p};
    for (const Real *b : blocks) {
        if (b == nullptr) {
            throw std::runtime_error("writeVtu: missing field");
        }
    }

    // Per-element node counts and the total point count (concatenated
    // element blocks in the vtkLagrangeParametricPoints order).
    const int nQuad = (order + 1) * (order + 1);
    const int nTri  = (order + 1) * (order + 2) / 2;
    std::vector<int> nPts(static_cast<size_t>(nElem));
    long long nPtsTot = 0;
    for (int e = 0; e < nElem; ++e) {
        const bool tri               = isTriangle[static_cast<size_t>(e)] != 0;
        nPts[static_cast<size_t>(e)] = tri ? nTri : nQuad;
        nPtsTot += tri ? nTri : nQuad;
    }

    // Interleaved xyz from the (x, y) SoA pair.
    std::vector<Real> xyz(static_cast<size_t>(nPtsTot) * 3);
    {
        size_t k = 0;
        for (int e = 0; e < nElem; ++e) {
            const int m = nPts[static_cast<size_t>(e)];
            for (int i = 0; i < m; ++i, ++k) {
                xyz[3 * k + 0] = f.x[k];
                xyz[3 * k + 1] = f.y[k];
                xyz[3 * k + 2] = Real(0);
            }
        }
    }

    // Cell arrays: connectivity (node ids are already sequential, one
    // block per element in the VTK lattice order), offsets, types.
    const int kTypeQuad = 70; // VTK_LAGRANGE_QUADRILATERAL
    const int kTypeTri  = 69; // VTK_LAGRANGE_TRIANGLE
    std::vector<int64_t> conn(static_cast<size_t>(nPtsTot));
    std::vector<int64_t> offs(static_cast<size_t>(nElem));
    std::vector<uint8_t> types(static_cast<size_t>(nElem));
    {
        int64_t acc = 0;
        for (int e = 0; e < nElem; ++e) {
            const bool tri = isTriangle[static_cast<size_t>(e)] != 0;
            for (int i = 0; i < nPts[static_cast<size_t>(e)]; ++i) {
                conn[static_cast<size_t>(acc + i)] = acc + i;
            }
            acc += nPts[static_cast<size_t>(e)];
            offs[static_cast<size_t>(e)] = acc;
            types[static_cast<size_t>(e)] =
                static_cast<uint8_t>(tri ? kTypeTri : kTypeQuad);
        }
    }

    // --- Appended raw layout: every block = [UInt32 byte count][bytes];
    //     the offset of each block (into the section after the '_'
    //     marker) is computable before writing. Block order: xyz, the
    //     seven point-data arrays, connectivity, offsets, types.
    const char *const names[7] = {"rho", "rho_u", "rho_v", "E", "u", "v", "p"};
    const Real *data[7]        = {f.rho, f.rhoU, f.rhoV, f.E, f.u, f.v, f.p};

    const auto blockBytes = [](long long n, int typeSize) {
        return 4 + static_cast<long long>(n) * typeSize;
    };
    long long off          = 0;
    const long long offXyz = off;
    off += blockBytes(3 * nPtsTot, 8);
    long long offData[7] = {0};
    for (int i = 0; i < 7; ++i) {
        offData[i] = off;
        off += blockBytes(nPtsTot, 8);
    }
    const long long offConn = off;
    off += blockBytes(nPtsTot, 8);
    const long long offOffs = off;
    off += blockBytes(nElem, 8);
    const long long offTyp = off;
    off += blockBytes(nElem, 1);

    std::ofstream out(vtuPath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("writeVtu: cannot open " + vtuPath);
    }
    const char *dtype = (sizeof(Real) == 8) ? "Float64" : "Float32";
    out << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
           "byte_order=\"LittleEndian\">\n"
        << "  <UnstructuredGrid>\n"
        << "    <Piece NumberOfPoints=\"" << nPtsTot << "\" NumberOfCells=\""
        << nElem << "\">\n"
        << "      <Points>\n"
        << "        <DataArray type=\"" << dtype
        << "\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << offXyz
        << "\"/>\n"
        << "      </Points>\n"
        << "      <PointData Scalars=\"rho\">\n";
    for (int i = 0; i < 7; ++i) {
        out << "        <DataArray type=\"" << dtype << "\" Name=\"" << names[i]
            << "\" format=\"appended\" offset=\"" << offData[i] << "\"/>\n";
    }
    out << "      </PointData>\n"
        << "      <Cells>\n"
        << "        <DataArray type=\"Int64\" Name=\"connectivity\" "
           "format=\"appended\" offset=\""
        << offConn << "\"/>\n"
        << "        <DataArray type=\"Int64\" Name=\"offsets\" "
           "format=\"appended\" offset=\""
        << offOffs << "\"/>\n"
        << "        <DataArray type=\"UInt8\" Name=\"types\" "
           "format=\"appended\" offset=\""
        << offTyp << "\"/>\n"
        << "      </Cells>\n"
        << "    </Piece>\n"
        << "  </UnstructuredGrid>\n"
        << "  <AppendedData encoding=\"raw\">\n   _";

    const auto putBlockHead = [&out](long long bytes) {
        const uint32_t n = static_cast<uint32_t>(bytes);
        out.write(reinterpret_cast<const char *>(&n), 4);
    };
    const auto putRealBlock = [&](const Real *p, long long n) {
        putBlockHead(static_cast<long long>(n) *
                     static_cast<long long>(sizeof(Real)));
        out.write(reinterpret_cast<const char *>(p),
                  static_cast<std::streamsize>(n) * sizeof(Real));
    };

    putRealBlock(xyz.data(), 3 * nPtsTot);
    for (int i = 0; i < 7; ++i) {
        putRealBlock(data[i], nPtsTot);
    }
    putBlockHead(static_cast<long long>(nPtsTot) * 8);
    out.write(reinterpret_cast<const char *>(conn.data()),
              static_cast<std::streamsize>(nPtsTot) * 8);
    putBlockHead(static_cast<long long>(nElem) * 8);
    out.write(reinterpret_cast<const char *>(offs.data()),
              static_cast<std::streamsize>(nElem) * 8);
    putBlockHead(static_cast<long long>(nElem));
    out.write(reinterpret_cast<const char *>(types.data()),
              static_cast<std::streamsize>(nElem));

    out << "\n  </AppendedData>\n</VTKFile>\n";
    out.flush();
    if (!out) {
        throw std::runtime_error("writeVtu: write failure on " + vtuPath);
    }
}

} // namespace viz
