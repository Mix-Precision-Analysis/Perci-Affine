#!/bin/bash
# run_BarrierOption.sh  —  BarrierOption
# Usage: $0 <thres> <base_algo_dir>
#   thres        : e.g. 1e-4
#   base_algo_dir: absolute path to DD/ or PEDD/ directory
set -euo pipefail

THRES="${1:-1e-4}"
BASE_ALGO_DIR="${2}"
ALGO="dd"
TIMEOUT=60
PERCI="${PERCI_TUNER:-/home/lilei/workspace/perci-tuner-alpha}"
RUN="sh $PERCI/scripts/run.sh"

PROG_DIR="$BASE_ALGO_DIR/$THRES/G5_PSCF/kernel/BarrierOption"
if [ ! -d "$PROG_DIR" ]; then
    echo "[SKIP] $PROG_DIR not found" >&2
    exit 0
fi

echo "======================================================="
echo "[START] BarrierOption  thres=$THRES  algo=$ALGO"
echo "  dir : $PROG_DIR"
echo "======================================================="

cd "$PROG_DIR"

$RUN search-optimize -alpha -verbose \
    barrier_option_mc.c \
    -thres="$THRES" -algorithm="$ALGO" \
    -timeout="$TIMEOUT" -history \
    -arg=100.0 -arg=100.0 -arg=120.0 -arg=0.05 -arg=0.20 -arg=1.0 -arg=252 -arg=10000 -arg=3 -arg=0 -arg=0 -arg=1 \
    -lm

echo "[DONE]  BarrierOption  thres=$THRES  algo=$ALGO"
