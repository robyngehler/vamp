#!/usr/bin/env bash

# Recreate the commonly used CMake build configurations for VAMP.
# Usage: scripts/build_vamp.sh [python|cpp|all|clean]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CMAKE_BIN=${CMAKE_BIN:-"$HOME/.local/bin/cmake"}

if ! command -v "$CMAKE_BIN" >/dev/null 2>&1; then
  CMAKE_BIN=$(command -v cmake)
fi

if ! command -v "$CMAKE_BIN" >/dev/null 2>&1; then
  echo "cmake not found. Install it or set CMAKE_BIN." >&2
  exit 1
fi

DEFAULT_EIGEN_PREFIX="/usr/local"
EIGEN_PREFIX=${EIGEN_PREFIX:-"$DEFAULT_EIGEN_PREFIX"}
PYTHON_BIN=${PYTHON_BIN:-"/usr/bin/python3"}

BUILD_PY_DIR=${BUILD_PY_DIR:-"build-py38"}
BUILD_CPP_DIR=${BUILD_CPP_DIR:-"build-release"}


build_python_targets() {
  python3 -m pip install --no-build-isolation -v --config-settings=cmake.define.CMAKE_PREFIX_PATH=/usr/local .
}

configure_cpp_build() {
  "$CMAKE_BIN" -S "$ROOT_DIR" -B "$ROOT_DIR/$BUILD_CPP_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$EIGEN_PREFIX" \
    -DVAMP_INSTALL_CPP_LIBRARY=ON \
    -DVAMP_BUILD_PYTHON_BINDINGS=OFF \
    -DVAMP_BUILD_CPP_DEMO=ON \
    -DVAMP_BUILD_OMPL_DEMO=OFF
}

build_cpp_targets() {
  "$CMAKE_BIN" --build "$ROOT_DIR/$BUILD_CPP_DIR"
}

install_cpp_targets() {
  "$CMAKE_BIN" --install "$ROOT_DIR/$BUILD_CPP_DIR"
}

clean_builds() {
  rm -rf "$ROOT_DIR/$BUILD_PY_DIR" "$ROOT_DIR/$BUILD_CPP_DIR"
}

target=${1:-all}

case "$target" in
  python)
    build_python_targets
    ;;
  cpp)
    configure_cpp_build
    build_cpp_targets
    install_cpp_targets
    ;;
  all)
    build_python_targets
    configure_cpp_build
    build_cpp_targets
    ;;
  clean)
    clean_builds
    ;;
  *)
    echo "Unknown target '$target'. Use python|cpp|all|clean." >&2
    exit 1
    ;;
esac
