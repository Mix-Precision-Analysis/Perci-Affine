#!/bin/bash
# run_bessel_j0.sh  —  bessel_j0
# Usage: $0 <thres> <base_algo_dir>
#   thres        : e.g. 1e-4
#   base_algo_dir: absolute path to DD/ or PEDD/ directory
set -euo pipefail

THRES="${1:-1e-4}"
BASE_ALGO_DIR="${2}"
ALGO="dd"
TIMEOUT=30
PERCI="${PERCI_TUNER:-/home/lilei/workspace/perci-tuner-alpha}"
RUN="sh $PERCI/scripts/run.sh"

PROG_DIR="$BASE_ALGO_DIR/$THRES/G1_Cancellation/kernel/bessel_j0"
if [ ! -d "$PROG_DIR" ]; then
    echo "[SKIP] $PROG_DIR not found" >&2
    exit 0
fi

echo "======================================================="
echo "[START] bessel_j0  thres=$THRES  algo=$ALGO"
echo "  dir : $PROG_DIR"
echo "======================================================="

cd "$PROG_DIR"

$RUN search-optimize -alpha -verbose \
    bessel.c -I./ -I/home/lilei/software/gsl/include\
    -thres="$THRES" -algorithm="$ALGO" \
    -timeout="$TIMEOUT" -history \
    -lm -L/home/lilei/software/gsl/lib -lgsl -lgslcblas

echo "[DONE]  bessel_j0  thres=$THRES  algo=$ALGO"
