#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch Isaac Sim with the HTTP motion server (motion_server.cpp compatible API).
#
# Usage:
#   ./launch_isaac_motion_server.sh [port]
#
# Examples:
#   ./launch_isaac_motion_server.sh
#   ./launch_isaac_motion_server.sh 34568 --headless --test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_ROOT="${REPO_ROOT}/_build/linux-x86_64/release"

if [[ ! -x "${BUILD_ROOT}/python.sh" ]]; then
    echo "ERROR: Isaac Sim build not found at ${BUILD_ROOT}"
    echo "Run ./build.sh from the repo root first."
    exit 1
fi

PORT=34568
EXTRA_ARGS=()
if [[ $# -gt 0 && "${1}" =~ ^[0-9]+$ ]]; then
    PORT="${1}"
    EXTRA_ARGS=(--port "${PORT}")
    shift
fi

echo
echo "=== Isaac Sim motion server launcher ==="
echo "  BUILD_ROOT = ${BUILD_ROOT}"
echo "  PROJECT    = ${SCRIPT_DIR}"
echo "  PORT       = ${PORT}"
echo

cd "${BUILD_ROOT}"
./python.sh "${SCRIPT_DIR}/isaac_motion_server.py" "${EXTRA_ARGS[@]}" "$@"
