#!/usr/bin/env bash
set -euo pipefail

readonly CASADI_REPOSITORY="https://github.com/casadi/casadi.git"
readonly CASADI_TAG="${CASADI_VERSION:-3.7.2}"
readonly CASADI_EXPECTED_COMMIT="${CASADI_COMMIT:-f959d3175a444d763e4eda4aece48f4c5f4a6f90}"
# CasADi 3.7.2 defaults to jgillis/alpaqa because that compatibility fork
# preserves the sparse Jacobian/Hessian signatures used by its plugin.
readonly ALPAQA_REPOSITORY="https://github.com/jgillis/alpaqa.git"
readonly ALPAQA_REVISION="${ALPAQA_COMMIT:-bf9f87d59640501ea72f94aa6e2d4e62b20c677b}"
readonly BUILD_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
readonly CASADI_CXX_ABI_FLAG="-D_GLIBCXX_USE_CXX11_ABI=0"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer is intended for Linux." >&2
  exit 1
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX must point to the active benchmark environment." >&2
  exit 1
fi

# The official CasADi wheel, RBDL-CasADi, and biorbd-casadi all use the
# pre-C++11 libstdc++ ABI. Exporting CXXFLAGS also propagates the same ABI to
# CasADi's Alpaqa ExternalProject, unlike a top-level CMAKE_CXX_FLAGS alone.
export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }$CASADI_CXX_ABI_FLAG"

build_root="$(mktemp -d)"
trap 'rm -rf "$build_root"' EXIT

git clone --quiet --branch "$CASADI_TAG" --depth 1 \
  "$CASADI_REPOSITORY" "$build_root/casadi"
if [[ "$(git -C "$build_root/casadi" rev-parse HEAD)" != "$CASADI_EXPECTED_COMMIT" ]]; then
  echo "CasADi tag $CASADI_TAG does not match the pinned commit." >&2
  exit 1
fi

cmake \
  -S "$build_root/casadi" \
  -B "$build_root/casadi-build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DWITH_PYTHON=ON \
  -DWITH_PYTHON3=ON \
  -DWITH_IPOPT=ON \
  -DWITH_ALPAQA=ON \
  -DWITH_BUILD_EIGEN3=ON \
  -DWITH_BUILD_ALPAQA=ON \
  -DBUILD_ALPAQA_GIT_REPO="$ALPAQA_REPOSITORY" \
  -DBUILD_ALPAQA_GIT_SHALLOW=OFF \
  -DBUILD_ALPAQA_VERSION="$ALPAQA_REVISION"
# CasADi 3.7.2 exposes Alpaqa as an imported target. Ninja does not propagate
# that target's ExternalProject dependency to the plugin reliably, so install
# the pinned Alpaqa build before linking the complete CasADi tree.
cmake --build "$build_root/casadi-build" \
  --target alpaqa-external \
  --parallel "$BUILD_JOBS"
cmake --build "$build_root/casadi-build" --parallel "$BUILD_JOBS"
cmake --install "$build_root/casadi-build"

python - <<'PY'
import casadi as cas

print(f"CasADi {cas.__version__}")
for solver in ("ipopt", "alpaqa"):
    available = cas.has_nlpsol(solver)
    print(f"{solver}: {available}")
    assert available, solver
PY
