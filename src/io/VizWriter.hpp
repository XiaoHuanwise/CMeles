/// @file VizWriter.hpp
/// @brief Visualization output helpers shared by the direct writer and the
///        offline checkpoint converter: sub-grid connectivity, the XDMF
///        sidecar and the classic Tecplot binary PLT writer.
///
/// Representation: the visualization nodes are a per-element tensor point
/// grid in the SoA ordering $k = e \cdot n^2 + (j \cdot n + i)$ with $n$
/// nodes per direction — the boundary-inclusive extended grid
/// $\{-1, \xi_1, \dots, \xi_{N_q}, +1\}$ of VizSampling for the
/// visualization formats, so each element's sub-grid reaches its own
/// interfaces (the DG polynomial is evaluated element-locally; NO values
/// are interpolated across elements). The nodes are connected by a
/// per-element sub-grid:
///   - quadrilateral elements contribute $(n-1)^2$ quadrilateral
///     sub-cells (NOT triangulated);
///   - triangle elements (degenerate quadrilaterals) split each tensor
///     sub-cell along the fixed diagonal into two triangles — sliver
///     quadrilaterals near the collapsed apex are avoided and thin
///     triangles render robustly;
///   - mixed meshes dispatch per element through the isTriangle flags.
///
/// The sub-grid is DISCONNECTED between elements: adjacent elements meet
/// exactly at their shared interface (each with its own boundary node
/// row), so the DG discontinuity renders as a color jump across the
/// interface with no background gap. For triangles the $s = +1$ node row
/// collapses onto the apex, producing zero-area sub-cells that renderers
/// skip (kept for deterministic counts).

#pragma once

#include <string>
#include <vector>

#include "common/Types.hpp"

class HDF5Writer;

namespace viz {

/// @brief Connectivity and counts of the visualization sub-grid.
struct VizTopology {
    long long nNodes = 0;     ///< N_elem * N_q^2 (quadrature points).
    long long nCells = 0;     ///< Total sub-cells (quad + triangle).
    bool mixed       = false; ///< True only when quad and triangle sub-cells
                              ///< coexist (mixed meshes).
    int cellType = 5;         ///< Single cell type when !mixed:
                              ///< 5 = quadrilateral, 4 = triangle.
    std::vector<int> connect; ///< !mixed: plain node ids (4 or 3 per
                              ///< cell). mixed: XDMF Mixed encoding
                              ///< [type, ids...] per cell.
};

/// @brief Build the per-element-type adaptive sub-grid connectivity.
///
/// Node ids follow the tensor SoA ordering $k = e\,n^2 + (j\,n + i)$ of
/// the visualization node grid (VizSampling::tensorPoints convention),
/// with \p nPts1D nodes per direction: pass $N_q$ for a grid over the
/// interior Gauss points or $N_q + 2$ for the boundary-inclusive extended
/// grid. Single-type meshes (all-quad / all-triangle) produce a plain id
/// table and the writers emit the maximally compatible UNIFORM topology
/// (QUADRILATERAL/TRIANGLE); only genuinely mixed meshes fall back to the
/// XDMF Mixed encoding. The construction is purely combinatorial (no
/// geometry), so the offline converter rebuilds it from the mesh.h5
/// attributes alone and old checkpoint directories convert without
/// regeneration.
VizTopology buildVizConnectivity(int N_elem, int nPts1D,
                                 const std::vector<int> &isTriangle);

/// @brief Write the visualization heavy data (mesh/connect and the
///        interleaved mesh/xyz coordinates, z = 0) into an open solution
///        HDF5 file.
///
/// Keeps the XDMF sidecar's heavy data self-contained in the same file
/// (the .h5/.xmf pair moves together). Shared by the direct writer and
/// the converter. The coordinates are stored interleaved as an
/// [N][3] dataset because the XDMF geometry is written as XYZ —
/// ParaView 5.13's Xdmf3 readers mis-parse GeometryType="XY" with two
/// 1-D items (they see only half the points), which sends the
/// connectivity out of range and crashes Apply/Render.
void writeVizToH5(HDF5Writer &writer, const VizTopology &topo, const Real *x,
                  const Real *y);

/// @brief Write the XDMF sidecar referencing the solution h5 datasets
///        (uniform or mixed topology, XYZ geometry via the interleaved
///        mesh/xyz dataset, seven nodal attributes, time). No DOCTYPE is
///        emitted (some XML stacks try to resolve the referenced DTD),
///        and single-type meshes use the uniform QUADRILATERAL/TRIANGLE
///        topology — the most widely supported reader path.
/// @param h5Basename File name (no directory) of the paired solution h5 —
///                   referenced relatively so the pair relocates cleanly.
/// @param xmfPath    Full path of the .xmf to write.
void writeXdmf(const std::string &h5Basename, const std::string &xmfPath,
               const VizTopology &topo, Real time);

/// @brief The nine nodal columns of a solution file.
struct Fields {
    const Real *x    = nullptr;
    const Real *y    = nullptr;
    const Real *rho  = nullptr;
    const Real *rhoU = nullptr;
    const Real *rhoV = nullptr;
    const Real *E    = nullptr;
    const Real *u    = nullptr;
    const Real *v    = nullptr;
    const Real *p    = nullptr;
};

/// @brief One time step of the XDMF temporal-collection index.
struct SeriesEntry {
    std::string h5Basename; ///< Solution h5 file name (no directory),
                            ///< referenced relatively from the series file.
    Real time = Real(0);    ///< Physical time of the step.
};

/// @brief (Re)write the XDMF temporal-collection index <prefix>_series.xmf.
///
/// ParaView's Xdmf3 readers do NOT lift a lone uniform grid's <Time> into
/// the time axis (opening the per-step .xmf files individually or as a
/// name-grouped series yields no time values); the temporal collection is
/// the supported XDMF time-series form (the analogue of ParaView's own
/// .pvd). The caller maintains the entry list incrementally — every
/// output step appends its (h5, time) pair and the whole index file is
/// regenerated, keeping it consistent by construction (a few KB of XML
/// referencing the existing heavy data; nothing is copied). Entries from
/// a previous run in the same directory are replaced by this run's.
void writeXdmfSeries(const std::string &seriesPath,
                     const std::vector<SeriesEntry> &entries,
                     const VizTopology &topo);

/// @brief Write a classic Tecplot binary PLT file (V112 layout, transient).
///
/// One finite-element zone per file with StrandID = 1 and the physical
/// SolutionTime, so a file series carries real time metadata in Tecplot
/// 360 and ParaView. Block (SoA-friendly) packing. The classic FE zone
/// admits a single cell type: all-quad and all-triangle meshes use their
/// native zone type (FEQUADRILATERAL / FETRIANGLE); only genuinely mixed
/// meshes fold the triangle sub-cells into quads with a repeated node
/// $(n_1, n_2, n_3, n_3)$ — the standard way mixed cell types enter a
/// classic Tecplot zone. The node ids are identical to the XDMF
/// connectivity, only the encoding is folded.
///
/// Byte layout (little-endian, matching the HDF5 default on x86), per
/// the reader used by ParaView's "VisIt Tecplot Binary Reader" and
/// Tecplot 360: magic "#!TDV112", INT4 1, title string, INT4 nVars,
/// variable-name strings (every string is one INT4 per character plus a
/// terminating INT4 0); zone header FLOAT32 299.0 marker, zone name,
/// INT4 parent zone (-1), INT4 StrandID (1), FLOAT64 SolutionTime, INT4
/// zone color (-1 = none), zone type (3 = FEQUADRILATERAL or 2 =
/// FETRIANGLE), INT4 variable locations (0 = all nodal), INT4 face
/// neighbours (0), INT4 user-defined neighbour connections (0), INT4
/// node/cell counts, 3 INT4 cell dims, INT4 0 auxiliary end; FLOAT32
/// 357.0 end-of-header. V112 has NO data-packing field (readers assume
/// block packing) and the version >= 103 data section carries
/// passive/sharing flags and per-variable min/max pairs: data section
/// FLOAT32 299.0 marker, per-variable INT4 format codes (2 = double,
/// 1 = float), INT4 passive 0, INT4 sharing 0, INT4 connectivity-sharing
/// -1, per-variable FLOAT64 min/max pairs, nine block data arrays,
/// 1-based INT4 cell connectivity.
void writeTecplotPlt(const std::string &pltPath, const Fields &f,
                     const VizTopology &topo, Real time);

} // namespace viz
