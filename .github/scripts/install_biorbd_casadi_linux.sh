#!/usr/bin/env bash
set -euo pipefail

readonly BIORBD_REPOSITORY="https://github.com/pyomeca/biorbd.git"
readonly BIORBD_TAG="Release_1.12.2"
readonly RBDL_REPOSITORY="https://github.com/pariterre/rbdl.git"
readonly RBDL_COMMIT="93475e2ea9bc87f37709a2312533ce3187f054b9"
readonly BUILD_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
readonly CASADI_CXX_ABI="${CASADI_CXX_ABI:-0}"
readonly CASADI_CXX_ABI_FLAG="-D_GLIBCXX_USE_CXX11_ABI=${CASADI_CXX_ABI}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer is intended for Linux." >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX must point to the active benchmark environment." >&2
  exit 1
fi

casadi_package_dir="$(
  python -c 'import casadi, pathlib; print(pathlib.Path(casadi.__file__).resolve().parent)'
)"
python_site_packages="$(
  python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)"

first_existing_path() {
  local candidate
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf 'None of the required CasADi paths exists:\n' >&2
  printf '  %s\n' "$@" >&2
  return 1
}

# The PyPI wheel keeps its CMake package, headers, and shared library inside
# site-packages/casadi. A source install (used for Alpaqa) places them under
# the active Conda prefix instead.
casadi_config="$(
  first_existing_path \
    "$casadi_package_dir/cmake/casadi-config.cmake" \
    "$CONDA_PREFIX/lib/cmake/casadi/casadi-config.cmake" \
    "$CONDA_PREFIX/share/casadi/cmake/casadi-config.cmake"
)"
casadi_cmake_dir="$(dirname "$casadi_config")"
casadi_header="$(
  first_existing_path \
    "$casadi_package_dir/include/casadi/casadi.hpp" \
    "$CONDA_PREFIX/include/casadi/casadi.hpp"
)"
casadi_include_dir="$(dirname "$casadi_header")"
casadi_library="$(
  first_existing_path \
    "$casadi_package_dir/libcasadi.so" \
    "$CONDA_PREFIX/lib/libcasadi.so"
)"

echo "Using CasADi CMake package: $casadi_cmake_dir"
echo "Using CasADi headers: $casadi_include_dir"
echo "Using CasADi library: $casadi_library"

build_root="$(mktemp -d)"
trap 'rm -rf "$build_root"' EXIT

echo "Building RBDL-CasADi with ${BUILD_JOBS} workers"
git clone --quiet "$RBDL_REPOSITORY" "$build_root/rbdl"
git -C "$build_root/rbdl" checkout --quiet "$RBDL_COMMIT"
cmake \
  -S "$build_root/rbdl" \
  -B "$build_root/rbdl-build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_CXX_FLAGS="$CASADI_CXX_ABI_FLAG -I$CONDA_PREFIX/include/eigen3" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -Dcasadi_DIR="$casadi_cmake_dir" \
  -DCasadi_DIR="$casadi_cmake_dir" \
  -DCasadi_INCLUDE_DIR="$casadi_include_dir" \
  -DCasadi_LIBRARY="$casadi_library" \
  -DRBDL_BUILD_CASADI=ON \
  -DRBDL_BUILD_EXECUTABLES=OFF \
  -DRBDL_BUILD_TESTS=OFF
cmake --build "$build_root/rbdl-build" --target install --parallel "$BUILD_JOBS"

# RBDL installs both math backends. Keeping the Eigen headers at
# $CONDA_PREFIX/include/rbdl makes biorbd find them before rbdl-casadi.
mv "$CONDA_PREFIX/include/rbdl" "$CONDA_PREFIX/include/rbdl-eigen-unused"

echo "Building biorbd ${BIORBD_TAG} with ${BUILD_JOBS} workers"
git clone --quiet --branch "$BIORBD_TAG" --depth 1 \
  "$BIORBD_REPOSITORY" "$build_root/biorbd"
cmake \
  -S "$build_root/biorbd" \
  -B "$build_root/biorbd-build" \
  -G Ninja \
  -DBINDER_PYTHON3=ON \
  -DBUILD_EXAMPLE=OFF \
  -DBUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_CXX_FLAGS="$CASADI_CXX_ABI_FLAG -I$CONDA_PREFIX/include/eigen3" \
  -DCasadi_DIR="$casadi_cmake_dir" \
  -DINSTALL_DEPENDENCIES_PREFIX="$CONDA_PREFIX" \
  -DMATH_LIBRARY_BACKEND=Casadi \
  -DMODULE_KALMAN=OFF \
  -DMODULE_STATIC_OPTIM=OFF \
  -DMODULE_VTP_FILES_READER=ON \
  -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DPython3_SITELIB_INSTALL="$python_site_packages"
cmake --build "$build_root/biorbd-build" --target install --parallel "$BUILD_JOBS"

python -c 'import biorbd_casadi as biorbd; print(f"biorbd {biorbd.__version__}")'
