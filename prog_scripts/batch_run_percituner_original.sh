#!/usr/bin/env bash
# Run all prog_scripts/run_*.sh with the original PerciTuner search order.
#
# Usage:
#   bash batch_run_percituner_original.sh [thres ...]
#
# Environment overrides:
#   AFFINE_TOOL_ROOT   default: parent directory of this script directory
#   ORIGINAL_BASE_DIR  default: $AFFINE_TOOL_ROOT/percituner
#   PERCI_TUNER        default: ../perci-tuner-alpha if present, else /home/lilei/workspace/perci-tuner-alpha

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFFINE_TOOL_ROOT="${AFFINE_TOOL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ORIGINAL_BASE_DIR="${ORIGINAL_BASE_DIR:-$AFFINE_TOOL_ROOT/percituner}"

if [ -z "${PERCI_TUNER:-}" ]; then
    if [ -d "$AFFINE_TOOL_ROOT/../perci-tuner-alpha" ]; then
        export PERCI_TUNER="$(cd "$AFFINE_TOOL_ROOT/../perci-tuner-alpha" && pwd)"
    else
        export PERCI_TUNER="/home/lilei/workspace/perci-tuner-alpha"
    fi
fi

if [ "$#" -gt 0 ]; then
    THRESHOLDS=("$@")
else
    THRESHOLDS=()
    while IFS= read -r dir; do
        THRESHOLDS+=("$(basename "$dir")")
    done < <(find "$ORIGINAL_BASE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [ "${#THRESHOLDS[@]}" -eq 0 ]; then
    echo "No threshold directories found under $ORIGINAL_BASE_DIR" >&2
    exit 1
fi

status=0
for thres in "${THRESHOLDS[@]}"; do
    log_dir="$ORIGINAL_BASE_DIR/$thres/_batch_logs"
    mkdir -p "$log_dir"

    for script in "$SCRIPT_DIR"/run_*.sh; do
        [ -f "$script" ] || continue
        prog="$(basename "$script" .sh)"
        prog="${prog#run_}"
        log="$log_dir/${prog}.log"

        echo "======================================================="
        echo "[BATCH ORIGINAL] thres=$thres prog=$prog"
        echo "  script: $script"
        echo "  log   : $log"
        echo "======================================================="

        if bash "$script" "$thres" "$ORIGINAL_BASE_DIR" >"$log" 2>&1; then
            echo "[OK]   $prog thres=$thres"
        else
            rc=$?
            echo "[FAIL] $prog thres=$thres rc=$rc log=$log" >&2
            status=1
        fi
    done
done

exit "$status"
