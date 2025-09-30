# AGENTS.md

## Project Overview
- A customized package for SIMD-enabled vectorized robot collision checking and motion planning on CPU
## Codebase Overview
- The codebase has C++ implementation for many robot model (Forward Kinematic (FK) and Colliion Checking (CC) generated with the cricket tracing compiler), collision checking routine, and planner. 
- It also features a multi-robot collision checking extension
- Python bindings, example implementations, are also included.

## Build Commands
- check the scripts/build_vamp.sh for building python binding and C++

## Code Style Guidelines
- Make concise API
- Follow existing coding style, e.g. mostly a clean, minimum, header-only C++ implementation so that the API can be easily integrated and compiled