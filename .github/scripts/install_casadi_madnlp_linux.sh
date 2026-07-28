#!/usr/bin/env bash
set -euo pipefail

readonly LIBMAD_INSTALL_DIR="${1:?libMad install directory is required}"
readonly CASADI_REPOSITORY="https://github.com/casadi/casadi.git"
readonly CASADI_EXPECTED_COMMIT="${CASADI_MADNLP_COMMIT:?CASADI_MADNLP_COMMIT is required}"
readonly BUILD_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
readonly CASADI_CXX_ABI="${CASADI_CXX_ABI:-1}"
readonly CASADI_CXX_ABI_FLAG="-D_GLIBCXX_USE_CXX11_ABI=${CASADI_CXX_ABI}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer is intended for Linux." >&2
  exit 1
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX must point to the active benchmark environment." >&2
  exit 1
fi

libmad_install_dir="$(cd "$LIBMAD_INSTALL_DIR" && pwd)"
test -f "$libmad_install_dir/lib/libMad.so"
test -f "$libmad_install_dir/include/libMad.h"

# Conda's IPOPT stack uses the modern libstdc++ ABI. Propagate the same ABI to
# CasADi and, later, to the biorbd-CasADi build.
export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }$CASADI_CXX_ABI_FLAG"

build_root="$(mktemp -d)"
trap 'rm -rf "$build_root"' EXIT

git clone --quiet "$CASADI_REPOSITORY" "$build_root/casadi"
git -C "$build_root/casadi" checkout --quiet "$CASADI_EXPECTED_COMMIT"
if [[ "$(git -C "$build_root/casadi" rev-parse HEAD)" != "$CASADI_EXPECTED_COMMIT" ]]; then
  echo "The CasADi checkout does not match CASADI_MADNLP_COMMIT." >&2
  exit 1
fi

# CasADi's current libMad integration consumes a release-style archive. Feed
# it the runtime already built and certified from the pinned libMad branch.
libmad_archive="$build_root/libMad-pardiso.tar.gz"
tar -C "$libmad_install_dir" -czf "$libmad_archive" .
libmad_hash="$(sha256sum "$libmad_archive" | awk '{print $1}')"

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
  -DWITH_THREAD=ON \
  -DWITH_IPOPT=ON \
  -DWITH_MADNLP=ON \
  -DWITH_BUILD_MADNLP=ON \
  -DLIBMAD_TAR_PATH="file://$libmad_archive" \
  -DLIBMAD_TAR_HASH="SHA256=$libmad_hash"
cmake --build "$build_root/casadi-build" --parallel "$BUILD_JOBS"
cmake --install "$build_root/casadi-build"

python - <<'PY'
import os

import casadi as cas

print(f"CasADi {cas.__version__}")
print(f"CasADi compiler flags: {cas.CasadiMeta.compiler_flags()}")
assert cas.__version__ == os.environ["CASADI_VERSION"]
assert "-DCASADI_WITH_THREAD" in cas.CasadiMeta.compiler_flags()
for solver in ("ipopt", "madnlp"):
    available = cas.has_nlpsol(solver)
    print(f"{solver}: {available}")
    assert available, solver
PY
