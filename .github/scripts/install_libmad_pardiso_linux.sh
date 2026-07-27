#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:?libMad source directory is required}"
install_dir="${2:?libMad install directory is required}"
juliac_commit="${3:?JuliaC commit is required}"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "PardisoMKLSolver requires an x86-64 Linux runner." >&2
  exit 1
fi

source_dir="$(cd "$source_dir" && pwd)"
mkdir -p "$install_dir"
install_dir="$(cd "$install_dir" && pwd)"
build_dir="$source_dir/build-cocofest-pardiso"

julia --startup-file=no -e \
  "using Pkg; Pkg.add(url=\"https://github.com/apozharski/JuliaC.jl.git\", rev=\"$juliac_commit\"); Pkg.Apps.add(url=\"https://github.com/apozharski/JuliaC.jl.git\", rev=\"$juliac_commit\")"
export PATH="$HOME/.julia/bin:$PATH"

julia --startup-file=no --project="$source_dir" -e \
  'using Pkg; Pkg.instantiate()'

cmake \
  -S "$source_dir" \
  -B "$build_dir" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir"
cmake --build "$build_dir" --target install --config Release

test -f "$install_dir/lib/libMad.so"
export LD_LIBRARY_PATH="$install_dir/lib:$install_dir/share/julia/lib:${LD_LIBRARY_PATH:-}"
basic_output="$("$build_dir/basic_problem" 2>&1)"
echo "$basic_output"
grep -qi "running with pardiso-mkl" <<< "$basic_output"
if grep -qi "unknown type PardisoMKLSolver" <<< "$basic_output"; then
  echo "The built libMad runtime does not expose PardisoMKLSolver." >&2
  exit 1
fi
