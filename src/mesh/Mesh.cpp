/// @file Mesh.cpp
/// @brief Implementation of Mesh topology and face construction.

#include "Mesh.hpp"

#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
/// @brief Directed edge record: one claim of an edge by an element.
struct FaceRecord {
    int elem; ///< Element index.
    int face; ///< Local face index in that element.
    int vA;   ///< First endpoint (direction of increasing face coordinate $t$).
    int vB;   ///< Second endpoint.
};

/// @brief Endpoint lookup table for directed faces.
///        Index by local face $f$: pairs $(P_j, P_k)$ in reference space.
constexpr int kFaceVertexPair[4][2] = {{3, 0}, {0, 1}, {1, 2}, {2, 3}};
} // namespace

Mesh::Mesh(const MatrixX2r &vertices,
           const Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>
               &elementVertices)
    : vertices_(vertices), elem_verts_(elementVertices) {
    N_vert_ = static_cast<int>(vertices_.rows());
    N_elem_ = static_cast<int>(elem_verts_.rows());

    elem_faces_.setConstant(N_elem_, 4, -1);
    face_elements_.resize(0, 4); // grown by buildFaces
    face_types_.resize(0);
    buildFaces();
}

std::pair<int, int> Mesh::faceVertices(int elem, int f) const {
    const int pj = kFaceVertexPair[f][0];
    const int pk = kFaceVertexPair[f][1];
    return {elem_verts_(elem, pj), elem_verts_(elem, pk)};
}

void Mesh::buildFaces() {
    // Key: unordered vertex pair (min, max). Value: first claim of the edge.
    // std::map is used (not unordered_map) so that the global face numbering
    // of boundary faces is deterministic across runs (ordered iteration).
    std::map<std::pair<int, int>, FaceRecord> edgeMap;

    // Phase 1: match shared edges between elements.
    for (int e = 0; e < N_elem_; ++e) {
        for (int f = 0; f < 4; ++f) {
            const auto [vA, vB] = faceVertices(e, f);
            if (vA == vB) {
                // Collapsed edge of a triangle: no physical face.
                elem_faces_(e, f) = -1;
                continue;
            }
            const auto key = std::make_pair(std::min(vA, vB), std::max(vA, vB));
            const auto it  = edgeMap.find(key);
            if (it == edgeMap.end()) {
                // First claimant becomes the left element K_L.
                edgeMap.emplace(key, FaceRecord{e, f, vA, vB});
            } else {
                const FaceRecord &rec = it->second;
                // In a consistently counter-clockwise mesh the second
                // claimant must traverse the edge in the opposite direction.
                if (vA != rec.vB || vB != rec.vA) {
                    throw std::runtime_error(
                        "Mesh::buildFaces: inconsistent edge orientation "
                        "between elements " +
                        std::to_string(rec.elem) + " and " + std::to_string(e));
                }
                const int F = N_face_++;
                face_elements_.conservativeResize(N_face_, Eigen::NoChange);
                face_elements_.row(F) << rec.elem, rec.face, e, f;
                elem_faces_(rec.elem, rec.face) = F;
                elem_faces_(e, f)               = F;
                face_types_.conservativeResize(N_face_);
                face_types_(F) = int(FaceType::Interior);
                edgeMap.erase(it);
            }
        }
    }

    // Phase 2: remaining unclaimed edges are boundary faces.
    // The single claimant is the left element; K_R = -1 marks the exterior.
    // Since the claimant traverses counter-clockwise, the face normal
    // (e_y, -e_x) points out of the domain (left element is the physical
    // element, per the "boundary element is always the left element"
    // convention).
    for (const auto &[key, rec] : edgeMap) {
        (void)key;
        const int F = N_face_++;
        face_elements_.conservativeResize(N_face_, Eigen::NoChange);
        face_elements_.row(F) << rec.elem, rec.face, -1, -1;
        elem_faces_(rec.elem, rec.face) = F;
        face_types_.conservativeResize(N_face_);
        face_types_(F) = int(FaceType::Boundary);
    }
}

void Mesh::applyPeriodicPairing(Real Lx, Real Ly) {
    // Candidate translations: +/-Lx in x, +/-Ly in y (0 disables a direction).
    std::vector<std::pair<Real, Real>> translations;
    if (Lx != Real(0)) {
        translations.emplace_back(Lx, Real(0));
        translations.emplace_back(-Lx, Real(0));
    }
    if (Ly != Real(0)) {
        translations.emplace_back(Real(0), Ly);
        translations.emplace_back(Real(0), -Ly);
    }
    if (translations.empty()) {
        throw std::runtime_error(
            "Mesh::applyPeriodicPairing: both periods are zero");
    }

    // Collect the unpaired boundary faces with their directed endpoints.
    struct BoundaryEdge {
        int face;
        int vA; ///< First endpoint (direction of increasing face coordinate).
        int vB; ///< Second endpoint.
    };
    std::vector<BoundaryEdge> boundary;
    for (int F = 0; F < N_face_; ++F) {
        if (face_elements_(F, 2) == -1) {
            const auto [vA, vB] =
                faceVertices(face_elements_(F, 0), face_elements_(F, 1));
            boundary.push_back(BoundaryEdge{F, vA, vB});
        }
    }

    // Coordinate tolerance scaled by the domain periods.
    const Real scale      = std::max({Real(1), std::abs(Lx), std::abs(Ly)});
    const Real tol        = Real(1e-9) * scale;
    const auto coincident = [&](const Real *p, const Real *q) {
        return std::abs(p[0] - q[0]) <= tol && std::abs(p[1] - q[1]) <= tol;
    };

    // Pair F1 (A1 -> B1) with F2 (A2 -> B2) when, for some translation T,
    // A1 + T = B2 and B1 + T = A2 (reversed traversal => t2 = -t1).
    std::vector<bool> paired(boundary.size(), false);
    int nPaired = 0;
    for (std::size_t i = 0; i < boundary.size(); ++i) {
        if (paired[i]) {
            continue;
        }
        for (std::size_t j = i + 1; j < boundary.size(); ++j) {
            if (paired[j]) {
                continue;
            }
            const Real *a1 = vertices_.row(boundary[i].vA).data();
            const Real *b1 = vertices_.row(boundary[i].vB).data();
            const Real *a2 = vertices_.row(boundary[j].vA).data();
            const Real *b2 = vertices_.row(boundary[j].vB).data();
            for (const auto &[tx, ty] : translations) {
                const Real targetA1[2] = {a1[0] + tx, a1[1] + ty};
                const Real targetB1[2] = {b1[0] + tx, b1[1] + ty};
                if (coincident(targetA1, b2) && coincident(targetB1, a2)) {
                    // Fill the K_R/f_R slots with the partner face's own left
                    // element and local face (pseudo-interior pairing).
                    const int F1          = boundary[i].face;
                    const int F2          = boundary[j].face;
                    face_elements_(F1, 2) = face_elements_(F2, 0);
                    face_elements_(F1, 3) = face_elements_(F2, 1);
                    face_elements_(F2, 2) = face_elements_(F1, 0);
                    face_elements_(F2, 3) = face_elements_(F1, 1);
                    face_types_(F1)       = int(FaceType::Periodic);
                    face_types_(F2)       = int(FaceType::Periodic);
                    paired[i] = paired[j] = true;
                    nPaired += 2;
                    break;
                }
            }
            if (paired[i]) {
                break;
            }
        }
    }

    if (nPaired != static_cast<int>(boundary.size())) {
        throw std::runtime_error(
            "Mesh::applyPeriodicPairing: " +
            std::to_string(boundary.size() - nPaired) +
            " boundary face(s) could not be paired under the given periods "
            "(non-matching boundary geometry, e.g. shear with y-periodicity)");
    }
}
