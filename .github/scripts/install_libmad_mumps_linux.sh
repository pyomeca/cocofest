#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:?libMad source directory is required}"
install_dir="${2:?libMad install directory is required}"
juliac_commit="${3:?JuliaC commit is required}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$script_dir/check_libmad_host_linux.sh"

source_dir="$(cd "$source_dir" && pwd)"
mkdir -p "$install_dir"
install_dir="$(cd "$install_dir" && pwd)"
build_dir="$source_dir/build-cocofest-mumps"

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
smoke_output="$("$build_dir/no_hsl_example" 2>&1)"
echo "$smoke_output"
grep -qi "MUMPS" <<< "$smoke_output"
if grep -Fqi "libMAD WARNING: option linear_solver is of unknown type" <<< "$smoke_output"; then
  echo "The built libMad runtime rejected a requested linear solver type." >&2
  exit 1
fi
