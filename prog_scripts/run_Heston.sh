#!/bin/bash
# run_Heston.sh  —  Heston
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

PROG_DIR="$BASE_ALGO_DIR/$THRES/G3_SingularityApprox/application/Heston"
if [ ! -d "$PROG_DIR" ]; then
    echo "[SKIP] $PROG_DIR not found" >&2
    exit 0
fi

echo "======================================================="
echo "[START] Heston  thres=$THRES  algo=$ALGO"
echo "  dir : $PROG_DIR"
echo "======================================================="

cd "$PROG_DIR"

$RUN search-optimize -alpha -verbose \
    heston_mc.c \
    -thres="$THRES" -algorithm="$ALGO" \
    -timeout="$TIMEOUT" -history \
    -arg=100.0 -arg=100.0 -arg=0.03 -arg=0.04 -arg=2.0 -arg=0.04 -arg=0.50 -arg=-0.70 -arg=1.0 -arg=252 -arg=10000 -arg=3 -arg=0 -arg=1 \
    -lm

echo "[DONE]  Heston  thres=$THRES  algo=$ALGO"
