/// @file test_mesh_geometry.cpp
/// @brief Host-side tests for Mesh topology, face construction and
///        MeshGeometry pre-computation (quads, triangles, orthogonal
///        detection, mass matrix inversion).

#include <cmath>
#include <iostream>

#include <Eigen/Dense>

#include "basis/BasisFunctions1D.hpp"
#include "basis/BasisFunctions2D.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/MeshGeometry.hpp"
#include "mesh/StructuredMeshGenerator.hpp"

namespace {
const Real tol = Real(1e3) * RealEpsilon;

bool checkRelative(Real value, Real reference, const char *what) {
    const Real denom = std::max(std::abs(reference), RealEpsilon);
    if (std::abs(value - reference) > tol * denom) {
        std::cout << "  FAIL " << what << ": value=" << value
                  << " ref=" << reference << "\n";
        return false;
    }
    return true;
}

/// Relative error of vector \p v w.r.t. reference \p ref (max norm).
Real relErr(const VectorXr &v, const VectorXr &ref) {
    return (v - ref).template lpNorm<Eigen::Infinity>() /
           std::max(ref.template lpNorm<Eigen::Infinity>(), RealEpsilon);
}

bool checkVector(const VectorXr &v, const VectorXr &ref, const char *what) {
    if (relErr(v, ref) > tol) {
        std::cout << "  FAIL " << what << ": relErr=" << relErr(v, ref) << "\n";
        return false;
    }
    return true;
}

/// Build the 2D basis used throughout the tests.
BasisFunctions2D makeBasis(int order, int nq) {
    static BasisFunctions1D basis1D(order, nq); // shared 1D basis
    return BasisFunctions2D(basis1D, order);
}
} // namespace

// ---------------------------------------------------------------------------
// 1. Structured generator: rectangular domain
// ---------------------------------------------------------------------------

static bool testGeneratorRectangular() {
    std::cout << "Test 1: structured generator (rectangular)\n";
    bool ok = true;

    const int nx = 3, ny = 2;
    const Real dx = Real(2), dy = Real(3);
    auto mesh = StructuredMeshGenerator::generate(
        StructuredMeshGenerator::Params{nx, ny, Real(0), Real(0), dx, dy});

    ok &= mesh.numVertices() == (nx + 1) * (ny + 1);
    ok &= mesh.numElements() == nx * ny;
    // Interior faces: nx*(ny-1) vertical + (nx-1)*ny horizontal.
    const int nInterior = nx * (ny - 1) + (nx - 1) * ny;
    const int nBoundary = 2 * nx + 2 * ny;
    ok &= mesh.numFaces() == nInterior + nBoundary;
    if (!ok) {
        std::cout << "  FAIL sizes: verts=" << mesh.numVertices()
                  << " elems=" << mesh.numElements()
                  << " faces=" << mesh.numFaces() << "\n";
        return false;
    }

    // Vertex coordinates.
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            const int v = j * (nx + 1) + i;
            ok &= checkRelative(mesh.vertices()(v, 0), Real(i) * dx, "x");
            ok &= checkRelative(mesh.vertices()(v, 1), Real(j) * dy, "y");
        }
    }

    // Element vertex connectivity: (v00, v10, v11, v01).
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int e    = j * nx + i;
            const int v00  = j * (nx + 1) + i;
            const auto &ev = mesh.elementVertices();
            ok &= ev(e, 0) == v00;
            ok &= ev(e, 1) == v00 + 1;
            ok &= ev(e, 2) == v00 + (nx + 1) + 1;
            ok &= ev(e, 3) == v00 + (nx + 1);
        }
    }

    // Face type counts.
    int nInt = 0, nBnd = 0;
    for (int F = 0; F < mesh.numFaces(); ++F) {
        mesh.faceTypes()(F) == int(Mesh::FaceType::Interior) ? ++nInt : ++nBnd;
    }
    ok &= nInt == nInterior;
    ok &= nBnd == nBoundary;
    if (!ok) {
        std::cout << "  FAIL face types: interior=" << nInt << " (expect "
                  << nInterior << ") boundary=" << nBnd << " (expect "
                  << nBoundary << ")\n";
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 2. Face construction: bidirectional mapping and orientation
// ---------------------------------------------------------------------------

static bool testFaceConstruction() {
    std::cout << "Test 2: face construction (2x2 square grid)\n";
    bool ok = true;

    auto mesh =
        StructuredMeshGenerator::generate(StructuredMeshGenerator::Params{
            2, 2, Real(0), Real(0), Real(1), Real(1)});

    for (int F = 0; F < mesh.numFaces(); ++F) {
        const auto &fe = mesh.faceElements();
        const int KL = fe(F, 0), fL = fe(F, 1);
        const int KR = fe(F, 2), fR = fe(F, 3);

        // Bidirectional mapping: elemFaces(KL, fL) == F (and KR if interior).
        ok &= mesh.elementFaces()(KL, fL) == F;
        if (KR != -1) {
            ok &= mesh.elementFaces()(KR, fR) == F;
            // Orientation: the right element traverses the same edge in the
            // opposite direction.
            const auto [lA, lB] = mesh.faceVertices(KL, fL);
            const auto [rA, rB] = mesh.faceVertices(KR, fR);
            ok &= lA == rB && lB == rA;
        } else {
            ok &= mesh.faceTypes()(F) == int(Mesh::FaceType::Boundary);
        }
    }

    // A 2x2 grid has 2 vertical + 2 horizontal shared edges = 4 interior
    // faces; each is claimed by exactly two elements.
    int shared = 0;
    for (int F = 0; F < mesh.numFaces(); ++F) {
        if (mesh.faceElements()(F, 2) != -1) {
            ++shared;
        }
    }
    ok &= shared == 4;
    return ok;
}

// ---------------------------------------------------------------------------
// 3. Face normals: analytic values and direction
// ---------------------------------------------------------------------------

static bool testFaceNormals() {
    std::cout << "Test 3: face normals\n";
    bool ok = true;

    const Real h = Real(2);
    auto mesh    = StructuredMeshGenerator::generate(
        StructuredMeshGenerator::Params{2, 2, Real(0), Real(0), h, h});
    MeshGeometry geo(mesh, makeBasis(2, 3));

    // Boundary faces of element 0 (bottom-left quad): left face n=(-h,0),
    // bottom face n=(0,-h).
    for (int F = 0; F < mesh.numFaces(); ++F) {
        const int KL = mesh.faceElements()(F, 0);
        const int fL = mesh.faceElements()(F, 1);
        if (KL != 0) {
            continue;
        }
        if (mesh.faceElements()(F, 2) != -1) {
            continue; // interior face of element 0: checked below
        }
        // Boundary faces of element 0: fL == 0 (left) or fL == 1 (bottom).
        if (fL == 0) {
            ok &= checkRelative(geo.faceNormals()(F, 0), Real(-h), "n_x left");
            ok &= checkRelative(geo.faceNormals()(F, 1), Real(0), "n_y left");
        } else if (fL == 1) {
            ok &= checkRelative(geo.faceNormals()(F, 0), Real(0), "n_x bottom");
            ok &=
                checkRelative(geo.faceNormals()(F, 1), Real(-h), "n_y bottom");
        }
    }

    // Interior faces: n points from K_L to K_R (same direction as the
    // segment joining the two element centres).
    for (int F = 0; F < mesh.numFaces(); ++F) {
        const int KL = mesh.faceElements()(F, 0);
        const int KR = mesh.faceElements()(F, 2);
        if (KR == -1) {
            continue;
        }
        // Element centres via their vertex coordinates.
        auto centre = [&](int e) -> std::pair<Real, Real> {
            const auto &v = mesh.elementVertices();
            Real cx = Real(0), cy = Real(0);
            for (int k = 0; k < 4; ++k) {
                cx += mesh.vertices()(v(e, k), 0);
                cy += mesh.vertices()(v(e, k), 1);
            }
            return {cx / Real(4), cy / Real(4)};
        };
        const auto [cxL, cyL] = centre(KL);
        const auto [cxR, cyR] = centre(KR);
        const Real dx = cxR - cxL, dy = cyR - cyL;
        // n . (centre_K R - centre_K L) > 0.
        const Real dot =
            geo.faceNormals()(F, 0) * dx + geo.faceNormals()(F, 1) * dy;
        ok &= dot > Real(0);
    }

    // |n| == edge length h for all faces.
    for (int F = 0; F < mesh.numFaces(); ++F) {
        const Real nx = geo.faceNormals()(F, 0);
        const Real ny = geo.faceNormals()(F, 1);
        ok &= checkRelative(std::sqrt(nx * nx + ny * ny), h, "|n|");
    }

    // Face Jacobian |J_f| == h/2.
    for (int F = 0; F < mesh.numFaces(); ++F) {
        ok &= checkRelative(geo.faceJacobians()(F), h / Real(2), "|J_f|");
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 4. Square element geometry
// ---------------------------------------------------------------------------

static bool testSquareGeometry() {
    std::cout << "Test 4: square element geometry\n";
    bool ok = true;

    const Real h = Real(2);
    auto mesh    = StructuredMeshGenerator::generate(
        StructuredMeshGenerator::Params{1, 1, Real(0), Real(0), h, h});
    auto basis = makeBasis(2, 3);
    MeshGeometry geo(mesh, basis);
    const int Nq2 = geo.numPointsPerElement();

    // |J| = h^2/4 constant; J^-1 = (2/h) I; sum Lambda = h^2.
    for (int q = 0; q < Nq2; ++q) {
        ok &= checkRelative(geo.absJacobian()(q), h * h / Real(4), "|J|");
        ok &= checkRelative(geo.Jinv11()(q), Real(2) / h, "Jinv11");
        ok &= checkRelative(geo.Jinv12()(q), Real(0), "Jinv12");
        ok &= checkRelative(geo.Jinv21()(q), Real(0), "Jinv21");
        ok &= checkRelative(geo.Jinv22()(q), Real(2) / h, "Jinv22");
    }
    Real sumLambda = Real(0);
    for (int q = 0; q < Nq2; ++q) {
        sumLambda += geo.lambdaWJ()(q);
    }
    ok &= checkRelative(sumLambda, h * h, "sum Lambda_wJ");

    // Inverse mass matrix: M^-1 (V^T Lambda V) == I.
    const auto &V = basis.vandermonde();
    const int Nb  = geo.numBases();
    Eigen::Map<const MatrixXr> minv(geo.massMatrixInverse().data(), Nb, Nb);
    VectorXr lam  = geo.lambdaWJ().head(Nq2);
    MatrixXr M    = (lam.asDiagonal() * V).transpose() * V;
    MatrixXr prod = minv * M;
    MatrixXr eye  = MatrixXr::Identity(Nb, Nb);
    ok &= (prod - eye).template lpNorm<Eigen::Infinity>() <= tol;
    return ok;
}

// ---------------------------------------------------------------------------
// 5. Warped quadrilateral (hand-built vertices)
// ---------------------------------------------------------------------------

/// Trapezoid: P1=(0,0), P2=(2,0), P3=(1.5,2), P4=(0,1). Not a parallelogram.
static Mesh makeTrapezoidMesh() {
    MatrixX2r verts(4, 2);
    verts << Real(0), Real(0), Real(2), Real(0), Real(1.5), Real(2), Real(0),
        Real(1);
    Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> ev(1, 4);
    ev << 0, 1, 2, 3;
    return Mesh(verts, ev);
}

static bool testWarpedQuad() {
    std::cout << "Test 5: warped quadrilateral (trapezoid)\n";
    bool ok = true;

    auto mesh  = makeTrapezoidMesh();
    auto basis = makeBasis(2, 3);
    MeshGeometry geo(mesh, basis);
    const int Nq2  = geo.numPointsPerElement();
    const auto &qp = basis.quadraturePoints();

    // Jacobian elements vs bilinear map: x = a0 + a1 r + a2 s + a3 rs.
    const Real x1 = Real(0), x2 = Real(2), x3 = Real(1.5), x4 = Real(0);
    const Real y1 = Real(0), y2 = Real(0), y3 = Real(2), y4 = Real(1);
    const Real a1 = (-x1 + x2 + x3 - x4) / Real(4);
    const Real a2 = (-x1 - x2 + x3 + x4) / Real(4);
    const Real a3 = (x1 - x2 + x3 - x4) / Real(4);
    const Real b1 = (-y1 + y2 + y3 - y4) / Real(4);
    const Real b2 = (-y1 - y2 + y3 + y4) / Real(4);
    const Real b3 = (y1 - y2 + y3 - y4) / Real(4);

    for (int q = 0; q < Nq2; ++q) {
        const Real r = qp(q, 0), s = qp(q, 1);
        const Real dxdr = a1 + a3 * s, dxds = a2 + a3 * r;
        const Real dydr = b1 + b3 * s, dyds = b2 + b3 * r;
        const Real detJ = dxdr * dyds - dxds * dydr;
        ok &= checkRelative(geo.absJacobian()(q), detJ, "|J| analytic");

        // J^-1 J == I per point. Off-diagonal entries should vanish; use an
        // absolute tolerance for the zero references.
        const Real j11 = geo.Jinv11()(q), j12 = geo.Jinv12()(q);
        const Real j21 = geo.Jinv21()(q), j22 = geo.Jinv22()(q);
        const Real id11 = j11 * dxdr + j12 * dydr;
        const Real id12 = j11 * dxds + j12 * dyds;
        const Real id21 = j21 * dxdr + j22 * dydr;
        const Real id22 = j21 * dxds + j22 * dyds;
        ok &= checkRelative(id11, Real(1), "(Jinv J)_11");
        if (std::abs(id12) > tol) {
            std::cout << "  FAIL (Jinv J)_12: value=" << id12 << "\n";
            ok = false;
        }
        if (std::abs(id21) > tol) {
            std::cout << "  FAIL (Jinv J)_21: value=" << id21 << "\n";
            ok = false;
        }
        ok &= checkRelative(id22, Real(1), "(Jinv J)_22");
    }

    // M^-1 (V^T Lambda V) == I and projection roundtrip.
    const auto &V = basis.vandermonde();
    const int Nb  = geo.numBases();
    Eigen::Map<const MatrixXr> minv(geo.massMatrixInverse().data(), Nb, Nb);
    VectorXr lam  = geo.lambdaWJ().head(Nq2);
    MatrixXr M    = (lam.asDiagonal() * V).transpose() * V;
    MatrixXr prod = minv * M;
    ok &= (prod - MatrixXr::Identity(Nb, Nb))
              .template lpNorm<Eigen::Infinity>() <= tol;

    // Projection roundtrip: u_hat -> nodes -> modes.
    VectorXr uhat     = VectorXr::Random(Nb);
    VectorXr nodes    = V * uhat;
    VectorXr lamNodes = lam.asDiagonal() * nodes;
    VectorXr uhat2    = minv * (V.transpose() * lamNodes);
    ok &= relErr(uhat2, uhat) <= tol;

    // Area conservation: sum Lambda == shoelace area of trapezoid.
    // Shoelace: (0,0),(2,0),(1.5,2),(0,1) -> area = 2.75.
    const Real area = (Real(2) * Real(2) - Real(0) * Real(1.5)) / Real(2) +
                      (Real(1.5) * Real(1) - Real(0) * Real(2)) / Real(2);
    ok &= checkRelative(geo.lambdaWJ().sum(), area, "trapezoid area");
    return ok;
}

// ---------------------------------------------------------------------------
// 6. Triangle collapse (single degenerate quad)
// ---------------------------------------------------------------------------

static bool testTriangleCollapse() {
    std::cout << "Test 6: triangle collapse\n";
    bool ok = true;

    // Right triangle with legs h: P1=(0,0), P2=(h,0), P3=P4=(0,h).
    const Real h = Real(2);
    MatrixX2r verts(3, 2);
    verts << Real(0), Real(0), h, Real(0), Real(0), h;
    Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> ev(1, 4);
    ev << 0, 1, 2, 2; // P3 == P4 == vertex 2
    Mesh mesh(verts, ev);

    ok &= mesh.isTriangle(0);
    ok &= mesh.numFaces() == 3; // collapsed face excluded
    ok &= mesh.elementFaces()(0, 3) == -1;

    auto basis = makeBasis(2, 3);
    MeshGeometry geo(mesh, basis);
    const int Nq2  = geo.numPointsPerElement();
    const auto &qp = basis.quadraturePoints();

    // |J| analytic value. For the degenerate quad with P3=P4=(0,h):
    //   x(r,s) = h(1+r)(1-s)/4,  y(r,s) = h(1+s)/2
    //   dx/dr = h(1-s)/4,  dx/ds = -h(1+r)/4,  dy/dr = 0,  dy/ds = h/2
    //   |J| = dxdr*dyds - dxds*dydr = h^2(1-s)/8.
    // For h=2: |J| = (1-s)/2; integral over [-1,1]^2 gives area h^2/2.
    for (int q = 0; q < Nq2; ++q) {
        const Real s = qp(q, 1);
        ok &= checkRelative(geo.absJacobian()(q),
                            h * h * (Real(1) - s) / Real(8), "|J| triangle");
    }

    // Sum Lambda == triangle area h^2/2 (Nq=3 integrates the affine |J|
    // exactly).
    ok &= checkRelative(geo.lambdaWJ().sum(), h * h / Real(2), "triangle area");

    // M^-1 M == I still holds for the degenerate element.
    const auto &V = basis.vandermonde();
    const int Nb  = geo.numBases();
    Eigen::Map<const MatrixXr> minv(geo.massMatrixInverse().data(), Nb, Nb);
    VectorXr lam = geo.lambdaWJ().head(Nq2);
    MatrixXr M   = (lam.asDiagonal() * V).transpose() * V;
    ok &= (minv * M - MatrixXr::Identity(Nb, Nb))
              .template lpNorm<Eigen::Infinity>() <= tol;
    return ok;
}

// ---------------------------------------------------------------------------
// 7. Triangle-split structured mesh
// ---------------------------------------------------------------------------

static bool testTriangleSplit() {
    std::cout << "Test 7: triangle-split structured mesh\n";
    bool ok = true;

    const int nx = 2, ny = 2;
    const Real h = Real(2);
    auto mesh =
        StructuredMeshGenerator::generate(StructuredMeshGenerator::Params{
            nx, ny, Real(0), Real(0), h, h, Real(0), true});

    ok &= mesh.numElements() == 2 * nx * ny;
    // Faces: 3*nx*ny + nx + ny (each triangle contributes 3 edges).
    ok &= mesh.numFaces() == 3 * nx * ny + nx + ny;
    for (int e = 0; e < mesh.numElements(); ++e) {
        ok &= mesh.isTriangle(e);
        ok &= mesh.elementFaces()(e, 3) == -1;
    }

    // Total area conservation.
    MeshGeometry geo(mesh, makeBasis(2, 3));
    ok &= checkRelative(geo.lambdaWJ().sum(), Real(nx * h) * Real(ny * h),
                        "tri domain area");
    return ok;
}

// ---------------------------------------------------------------------------
// 8. Area conservation across mesh types
// ---------------------------------------------------------------------------

static bool testAreaConservation() {
    std::cout << "Test 8: area conservation\n";
    bool ok = true;

    // Rectangle.
    {
        auto mesh =
            StructuredMeshGenerator::generate(StructuredMeshGenerator::Params{
                3, 2, Real(0), Real(0), Real(2), Real(3)});
        MeshGeometry geo(mesh, makeBasis(2, 4));
        ok &=
            checkRelative(geo.lambdaWJ().sum(), Real(6) * Real(6), "rect area");
    }
    // Parallelogram (sheared).
    {
        auto mesh =
            StructuredMeshGenerator::generate(StructuredMeshGenerator::Params{
                2, 2, Real(0), Real(0), Real(2), Real(2), Real(1)});
        MeshGeometry geo(mesh, makeBasis(2, 4));
        // Area = nx*dx * ny*dy (shear preserves area).
        ok &= checkRelative(geo.lambdaWJ().sum(), Real(4) * Real(4),
                            "parallelogram area");
    }
    // Trapezoid (hand-built): shoelace area = 2.75.
    {
        auto mesh = makeTrapezoidMesh();
        MeshGeometry geo(mesh, makeBasis(2, 4));
        ok &= checkRelative(geo.lambdaWJ().sum(), Real(2.75), "trap area");
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 9. Orthogonal (constant-Jacobian) detection
// ---------------------------------------------------------------------------

static bool testOrthogonalDetection() {
    std::cout << "Test 9: orthogonal detection\n";
    bool ok = true;

    // Rectangular grid: orthogonal.
    {
        auto mesh =
            StructuredMeshGenerator::generate(StructuredMeshGenerator::Params{
                2, 2, Real(0), Real(0), Real(2), Real(3)});
        MeshGeometry geo(mesh, makeBasis(2, 3));
        ok &= geo.isOrthogonal();
        for (int e = 0; e < geo.numElements(); ++e) {
            ok &= checkRelative(geo.jacobianConstant()(e),
                                Real(2) * Real(3) / Real(4), "|J| const");
            ok &= checkRelative(geo.massMatrixInverseDiagonal()(e),
                                Real(4) / (Real(2) * Real(3)), "1/|J| diag");
        }
    }
    // Sheared parallelogram grid: still orthogonal.
    {
        auto mesh =
            StructuredMeshGenerator::generate(StructuredMeshGenerator::Params{
                2, 2, Real(0), Real(0), Real(2), Real(2), Real(1)});
        MeshGeometry geo(mesh, makeBasis(2, 3));
        ok &= geo.isOrthogonal();
        ok &= checkRelative(geo.jacobianConstant()(0),
                            Real(2) * Real(2) / Real(4), "|J| shear");
    }
    // Trapezoid: not orthogonal.
    {
        auto mesh = makeTrapezoidMesh();
        MeshGeometry geo(mesh, makeBasis(2, 3));
        ok &= !geo.isOrthogonal();
        // |J| varies across quadrature points.
        const int Nq2 = geo.numPointsPerElement();
        bool varies   = false;
        for (int q = 1; q < Nq2; ++q) {
            if (std::abs(geo.absJacobian()(q) - geo.absJacobian()(0)) > tol) {
                varies = true;
            }
        }
        ok &= varies;
    }
    // Triangle split grid: in the degenerate-quadrilateral framework the
    // triangle's |J| varies linearly with s (Section 2.4 of
    // mesh_and_geometry.md) — NOT constant — so isOrthogonal() is false
    // (the parallelogram condition P1+P3 == P2+P4 does not hold).
    {
        auto mesh =
            StructuredMeshGenerator::generate(StructuredMeshGenerator::Params{
                2, 2, Real(0), Real(0), Real(2), Real(2), Real(0), true});
        MeshGeometry geo(mesh, makeBasis(2, 3));
        ok &= !geo.isOrthogonal();
        // |J| varies with s across quadrature points.
        const int Nq2 = geo.numPointsPerElement();
        bool varies   = false;
        for (int q = 1; q < Nq2; ++q) {
            if (std::abs(geo.absJacobian()(q) - geo.absJacobian()(0)) > tol) {
                varies = true;
            }
        }
        ok &= varies;
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 10. Orientation regression: all generated meshes have |J| > 0
// ---------------------------------------------------------------------------

static bool testPositiveJacobian() {
    std::cout << "Test 10: positive Jacobian regression\n";
    bool ok = true;

    const std::vector<StructuredMeshGenerator::Params> cases = {
        {2, 3, Real(0), Real(0), Real(1), Real(1), Real(0), false},
        {3, 2, Real(-1), Real(-1), Real(0.5), Real(2), Real(0.5), false},
        {2, 2, Real(0), Real(0), Real(1), Real(1), Real(0), true},
        {4, 1, Real(0), Real(0), Real(2), Real(1), Real(1), true},
    };
    int idx = 0;
    for (const auto &p : cases) {
        auto mesh = StructuredMeshGenerator::generate(p);
        MeshGeometry geo(mesh, makeBasis(2, 3));
        if (geo.absJacobian().minCoeff() <= Real(0)) {
            std::cout << "  FAIL case " << idx << ": min |J| <= 0\n";
            ok = false;
        }
        ++idx;
    }
    return ok;
}

int main() {
    bool ok = true;
    struct Case {
        const char *name;
        bool (*fn)();
    };
    const Case cases[] = {
        {"testGeneratorRectangular", testGeneratorRectangular},
        {"testFaceConstruction", testFaceConstruction},
        {"testFaceNormals", testFaceNormals},
        {"testSquareGeometry", testSquareGeometry},
        {"testWarpedQuad", testWarpedQuad},
        {"testTriangleCollapse", testTriangleCollapse},
        {"testTriangleSplit", testTriangleSplit},
        {"testAreaConservation", testAreaConservation},
        {"testOrthogonalDetection", testOrthogonalDetection},
        {"testPositiveJacobian", testPositiveJacobian},
    };
    for (const auto &c : cases) {
        const bool r = c.fn();
        std::cout << "  [" << (r ? "PASS" : "FAIL") << "] " << c.name << "\n";
        ok &= r;
    }

    std::cout << (ok ? "ALL MESH GEOMETRY TESTS PASSED\n"
                     : "MESH GEOMETRY TESTS FAILED\n");
    return ok ? 0 : 1;
}
