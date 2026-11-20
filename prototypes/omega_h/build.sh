#!/usr/bin/env bash
# Build SCOREC Omega_h (serial: no MPI, no Kokkos) with its Python module
# PyOmega_h, with the patches in patches/ applied.
#
# Usage, from the repository root:
#
#   nix develop --command prototypes/omega_h/build.sh WORK_DIR [PYTHON]
#
#   WORK_DIR  directory for the source (WORK_DIR/src), the build tree
#             (WORK_DIR/build) and the install (WORK_DIR/install).
#   PYTHON    the Python to build the module for
#             (default: the project's .venv/bin/python).
#
# If WORK_DIR/src does not exist, the script clones the pinned commit and
# applies the patches. If it exists, the script builds it as it is (for
# work on the patches in a clone).
#
# The compiler comes from the devShell. cmake, ninja, pybind11 and zlib come
# from the flake's nixpkgs.
#
# Environment: OMEGA_H_REPO, OMEGA_H_COMMIT, OMEGA_H_JOBS (default 4),
# OMEGA_H_NO_PATCH=1 (a fresh clone without the patches, for a control).
set -euo pipefail

work=${1:?usage: build.sh WORK_DIR [PYTHON]}
repo_root=$(git rev-parse --show-toplevel)
python=${2:-$repo_root/.venv/bin/python}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
url=${OMEGA_H_REPO:-https://github.com/SCOREC/omega_h}
commit=${OMEGA_H_COMMIT:-7647c8a52547a737f06de91d3f47952643ca5b64}
jobs=${OMEGA_H_JOBS:-4}

mkdir -p "$work"
work=$(cd "$work" && pwd)
src=$work/src

if [[ ! -d $src ]]; then
  git clone --quiet "$url" "$src"
  git -C "$src" checkout --quiet "$commit"
  if [[ ${OMEGA_H_NO_PATCH:-0} != 1 ]]; then
    for p in "$here"/patches/*.patch; do
      git -C "$src" apply "$p"
      echo "applied $(basename "$p")"
    done
  fi
else
  echo "using the existing source: $src ($(git -C "$src" rev-parse --short HEAD))"
fi

tool() {
  nix build --inputs-from "$repo_root" --no-link --print-out-paths "nixpkgs#$1"
}
export PATH=$(tool cmake)/bin:$(tool ninja)/bin:$PATH
pybind11_dir=$(tool python312Packages.pybind11)/share/cmake/pybind11
zlib_dev=$(tool zlib.dev)
zlib_lib=$(tool zlib)/lib
if [[ $(uname) == Darwin ]]; then zlib_so=$zlib_lib/libz.dylib; else zlib_so=$zlib_lib/libz.so; fi

cmake -S "$src" -B "$work/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$work/install" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=ON \
  -DOmega_h_USE_MPI=OFF \
  -DOmega_h_USE_Kokkos=OFF \
  -DOmega_h_USE_OpenMP=OFF \
  -DOmega_h_THROW=ON \
  -DOmega_h_USE_ZLIB=ON \
  -DZLIB_INCLUDE_DIR="$zlib_dev/include" \
  -DZLIB_LIBRARY="$zlib_so" \
  -DOmega_h_USE_pybind11=ON \
  -Dpybind11_DIR="$pybind11_dir" \
  -DPython_EXECUTABLE="$python" \
  -DPYTHON_EXECUTABLE="$python" \
  -DPYBIND11_FINDPYTHON=ON
cmake --build "$work/build" -j "$jobs"
cmake --install "$work/build" >/dev/null
# -j1: two of Omega_h's own VTK tests write the same file.
(cd "$work/build" && ctest -j1 --output-on-failure)
echo "PyOmega_h: $work/build/src (put it on PYTHONPATH)"
