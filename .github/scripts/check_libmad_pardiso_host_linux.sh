#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "MadNLP/PARDISO requires an x86-64 Linux runner." >&2
  exit 1
fi

# Julia 1.12's generated sysimage references a symbol version introduced in
# GCC 13. Check the library the linker will actually resolve, rather than the
# compiler's marketing version (which is unreliable when cc is Clang).
libgcc_path="$(cc -print-file-name=libgcc_s.so.1 2>/dev/null || true)"
if [[ -z "$libgcc_path" || "$libgcc_path" == "libgcc_s.so.1" || ! -f "$libgcc_path" ]]; then
  echo "Unable to resolve libgcc_s.so.1 through cc." >&2
  exit 1
fi
if ! strings "$libgcc_path" | grep -qx 'GCC_13.0.0'; then
  echo "$libgcc_path does not export GCC_13.0.0, required by the Julia 1.12 runtime." >&2
  echo "Use ubuntu-24.04 (the workflow default) or an equivalent x86-64 runner." >&2
  exit 1
fi

echo "Compatible MadNLP host runtime: $libgcc_path"
