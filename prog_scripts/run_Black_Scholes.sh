#!/bin/bash
# run_Black_Scholes.sh  —  Black_Scholes
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

PROG_DIR="$BASE_ALGO_DIR/$THRES/G1_Cancellation/kernel/Black_Scholes"
if [ ! -d "$PROG_DIR" ]; then
    echo "[SKIP] $PROG_DIR not found" >&2
    exit 0
fi

echo "======================================================="
echo "[START] Black_Scholes  thres=$THRES  algo=$ALGO"
echo "  dir : $PROG_DIR"
echo "======================================================="

cd "$PROG_DIR"

$RUN search-optimize -alpha -verbose \
    Black-Scholes.c \
    -thres="$THRES" -algorithm="$ALGO" \
    -timeout="$TIMEOUT" -history \
    -arg=input_test.txt -arg=dummy_out -arg=100000 \
    -lm

echo "[DONE]  Black_Scholes  thres=$THRES  algo=$ALGO"
