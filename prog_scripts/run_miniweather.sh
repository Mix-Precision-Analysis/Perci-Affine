#!/bin/bash
# run_miniweather.sh  —  miniweather
# Usage: $0 <thres> <base_algo_dir>
#   thres        : e.g. 1e-4
#   base_algo_dir: absolute path to DD/ or PEDD/ directory
set -euo pipefail

THRES="${1:-1e-4}"
BASE_ALGO_DIR="${2}"
ALGO="dd"
TIMEOUT=120
PERCI="${PERCI_TUNER:-/home/lilei/workspace/perci-tuner-alpha}"
RUN="sh $PERCI/scripts/run.sh"

PROG_DIR="$BASE_ALGO_DIR/$THRES/G2_AccumulationDrift/application/miniweather"
if [ ! -d "$PROG_DIR" ]; then
    echo "[SKIP] $PROG_DIR not found" >&2
    exit 0
fi

echo "======================================================="
echo "[START] miniweather  thres=$THRES  algo=$ALGO"
echo "  dir : $PROG_DIR"
echo "======================================================="

cd "$PROG_DIR"

$RUN search-optimize -alpha -verbose \
    miniweather_serial_c.c \
    -thres="$THRES" -algorithm="$ALGO" \
    -timeout="$TIMEOUT" -history \
    -arg=100 -arg=50 -arg=10.0 -arg=5.0 -arg=1 \
    -lm

echo "[DONE]  miniweather  thres=$THRES  algo=$ALGO"
