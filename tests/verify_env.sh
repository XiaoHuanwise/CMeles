#!/usr/bin/env bash
# CMeles environment verification script
# Usage: bash tests/verify_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
info() { echo -e "${YELLOW}[----]${NC} $1"; }

echo "========================================"
echo "  CMeles Environment Verification"
echo "========================================"
echo ""

ERRORS=0

# ------ 1. Check basic tools ------
info "Checking basic tools..."

for cmd in cmake g++; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1 | head -1)
        pass "$cmd  ($ver)"
    else
        fail "$cmd  not found"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""

# ------ 2. Check third-party dependencies ------
info "Checking third-party dependencies..."

OCCA_INSTALL="${SCRIPT_DIR}/../third_party/.occa_install"
if [ -f "${OCCA_INSTALL}/lib/cmake/OCCA/OCCAConfig.cmake" ]; then
    pass "OCCA installed"
else
    fail "OCCA not found (${OCCA_INSTALL})"
    ERRORS=$((ERRORS + 1))
fi

if cmake --find-package -DNAME=Eigen3 -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST &>/dev/null; then
    pass "Eigen3 installed"
else
    fail "Eigen3 not found"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ------ 3. CMake configure ------
info "CMake configure..."

mkdir -p "${BUILD_DIR}"
if cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" 2>&1 | tail -3; then
    pass "CMake configure succeeded"
else
    fail "CMake configure failed"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ------ 4. Build ------
info "Building..."

if cmake --build "${BUILD_DIR}" -j"$(nproc)" 2>&1 | tail -3; then
    pass "Build succeeded"
else
    fail "Build failed"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ------ 5. Run ------
info "Running verify_env..."
echo "----------------------------------------"

RUN_OK=0
if "${BUILD_DIR}/verify_env" 2>&1; then
    RUN_OK=1
fi

echo "----------------------------------------"
if [ "$RUN_OK" -eq 1 ]; then
    pass "Run succeeded"
else
    fail "Run failed"
    ERRORS=$((ERRORS + 1))
fi

# ------ Summary ------
echo ""
echo "========================================"
if [ "$ERRORS" -eq 0 ]; then
    echo -e "  ${GREEN}All checks passed ✓${NC}"
else
    echo -e "  ${RED}${ERRORS} check(s) failed ✗${NC}"
fi
echo "========================================"

exit "$ERRORS"
