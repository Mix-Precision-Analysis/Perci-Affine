#!/bin/bash
# run_lorenz_rk4.sh  —  lorenz_rk4
# Usage: $0 <thres> <base_algo_dir>
#   thres        : e.g. 1e-4
#   base_algo_dir: absolute path to DD/ or PEDD/ directory
set -euo pipefail

THRES="${1:-1e-4}"
BASE_ALGO_DIR="${2}"
ALGO="dd"
TIMEOUT=20
PERCI="${PERCI_TUNER:-/home/lilei/workspace/perci-tuner-alpha}"
RUN="sh $PERCI/scripts/run.sh"

PROG_DIR="$BASE_ALGO_DIR/$THRES/G2_AccumulationDrift/micro/lorenz_rk4"
if [ ! -d "$PROG_DIR" ]; then
    echo "[SKIP] $PROG_DIR not found" >&2
    exit 0
fi

echo "======================================================="
echo "[START] lorenz_rk4  thres=$THRES  algo=$ALGO"
echo "  dir : $PROG_DIR"
echo "======================================================="

cd "$PROG_DIR"

$RUN search-optimize -alpha -verbose \
    lorenz_rk4.c \
    -thres="$THRES" -algorithm="$ALGO" \
    -timeout="$TIMEOUT" -history \
    -lm

echo "[DONE]  lorenz_rk4  thres=$THRES  algo=$ALGO"
