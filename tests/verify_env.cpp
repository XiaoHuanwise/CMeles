#include <iostream>
#include <vector>
#include <cmath>

#include <Eigen/Dense>
#include <occa.hpp>

int main() {
    // ========== Eigen Test ==========
    std::cout << "===== Eigen Test =====" << std::endl;

    Eigen::Matrix3d A;
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    Eigen::Vector3d x(1.0, 2.0, 3.0);
    Eigen::Vector3d b = A * x;

    std::cout << "A =\n" << A << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;
    std::cout << "A * x = " << b.transpose() << std::endl;

    // Solve M * x = rhs
    Eigen::Matrix3d M;
    M << 2, 1, 0,
         1, 3, 1,
         0, 1, 2;
    Eigen::Vector3d rhs(1.0, 2.0, 3.0);
    Eigen::Vector3d sol = M.colPivHouseholderQr().solve(rhs);
    std::cout << "\nSolve M * x = rhs:\n"
              << "M =\n" << M << "\n"
              << "rhs = " << rhs.transpose() << "\n"
              << "x   = " << sol.transpose() << std::endl;

    // Verify
    Eigen::Vector3d residual = M * sol - rhs;
    std::cout << "residual ||M*x - rhs|| = " << residual.norm() << std::endl;

    // ========== OCCA Test ==========
    std::cout << "\n===== OCCA Test =====" << std::endl;

    const int entries = 8;
    std::vector<float> h_a(entries), h_b(entries), h_ab(entries);
    for (int i = 0; i < entries; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Create Serial device
    occa::device device({
        {"mode", "Serial"}
    });
    std::cout << "OCCA mode: " << device.mode() << std::endl;

    // Allocate device memory and copy data
    occa::memory o_a = device.malloc<float>(entries, h_a.data());
    occa::memory o_b = device.malloc<float>(entries, h_b.data());
    occa::memory o_ab = device.malloc<float>(entries);

    // Build kernel from string
    const std::string kernelSource = R"(
@kernel void addVectors(const int entries,
                        const float *a,
                        const float *b,
                        float *ab) {
  for (int i = 0; i < entries; ++i; @tile(4, @outer, @inner)) {
    ab[i] = a[i] + b[i];
  }
}
)";

    occa::kernel addVectors = device.buildKernelFromString(kernelSource, "addVectors");

    // Run kernel
    addVectors(entries, o_a, o_b, o_ab);

    // Copy results back to host
    o_ab.copyTo(h_ab.data());

    std::cout << "Vector addition result (a + b):" << std::endl;
    bool ok = true;
    for (int i = 0; i < entries; ++i) {
        float expected = h_a[i] + h_b[i];
        bool match = (std::fabs(h_ab[i] - expected) < 1e-6f);
        std::cout << "  " << h_a[i] << " + " << h_b[i]
                  << " = " << h_ab[i]
                  << (match ? " PASS" : " FAIL") << std::endl;
        if (!match) ok = false;
    }

    std::cout << "\nOCCA test: " << (ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Eigen test: PASSED" << std::endl;

    return ok ? 0 : 1;
}
