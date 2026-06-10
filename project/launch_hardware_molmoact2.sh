#!/usr/bin/env bash
# MolmoAct2 on real Franka: USB cameras + motion_server.cpp (no Isaac Sim).
#
# Prerequisites (three terminals):
#   1. ./motion_server <robot-ip>
#   2. ./start_molmoact2_3090.sh  (GPU host)
#   3. this script
#
# Usage:
#   ./launch_hardware_molmoact2.sh --instruction "Pick up the apple and place it on the plate"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python3 hardware_molmoact2_runner.py \
  --motion-url "${MOLMO_MOTION_URL:-http://127.0.0.1:34568}" \
  --molmoact2-url "${MOLMOACT2_URL:-http://127.0.0.1:8012}" \
  "$@"
