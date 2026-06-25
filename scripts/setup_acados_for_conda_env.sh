#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <conda-env-name> <acados-source-dir>" >&2
  exit 2
fi

ENV_NAME="$1"
ACADOS_DIR="$(cd "$2" && pwd)"
BUILD_DIR="$ACADOS_DIR/build"
ENV_PYTHON="${CONDA_PREFIX:-}/bin/python"

if [[ ! -d "$ACADOS_DIR/interfaces/acados_template" ]]; then
  echo "Could not find interfaces/acados_template under $ACADOS_DIR" >&2
  exit 1
fi

if [[ ! -x "$ENV_PYTHON" ]]; then
  ENV_PYTHON="$HOME/miniconda3/envs/$ENV_NAME/bin/python"
fi
if [[ ! -x "$ENV_PYTHON" ]]; then
  ENV_PYTHON="$HOME/mambaforge/envs/$ENV_NAME/bin/python"
fi
if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "Could not find Python for conda env '$ENV_NAME'. Activate it first or install it under ~/miniconda3/envs." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
cmake -S "$ACADOS_DIR" -B "$BUILD_DIR" -DACADOS_WITH_QPOASES=ON -DBUILD_SHARED_LIBS=ON
cmake --build "$BUILD_DIR" --target install --parallel 4

"$ENV_PYTHON" -m pip install -e "$ACADOS_DIR/interfaces/acados_template"
printf 'y\n' | ACADOS_SOURCE_DIR="$ACADOS_DIR" "$ENV_PYTHON" -c 'from acados_template import get_tera; print(get_tera())'

cat <<EOF

ACADOS is built and acados_template is installed in conda env '$ENV_NAME'.
Add these exports before running the ACADOS cycling MHE:

export ACADOS_SOURCE_DIR="$ACADOS_DIR"
export DYLD_LIBRARY_PATH="$ACADOS_DIR/lib:\${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$ACADOS_DIR/lib:\${LD_LIBRARY_PATH:-}"

EOF
