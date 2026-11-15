#!/usr/bin/env bash
# Run all prog_scripts/run_*.sh with affine variable_order.txt generated before
# each PerciTuner search, then pass -order=variable_order.txt to search.py.
#
# Usage:
#   bash batch_run_affine_reordered.sh [thres ...]
#
# Environment overrides:
#   AFFINE_TOOL_ROOT       default: parent directory of this script directory
#   AFFINE_BASE_DIR        default: $AFFINE_TOOL_ROOT/affine_percituner_test
#   REAL_PERCI_TUNER       default: ../perci-tuner-alpha if present, else /home/lilei/workspace/perci-tuner-alpha
#   AFFINE_FP_TOOL         default: $AFFINE_TOOL_ROOT/fp_affine_error.py
#   AFFINE_ORDER_DIRECTION default: desc

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFFINE_TOOL_ROOT="${AFFINE_TOOL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
AFFINE_BASE_DIR="${AFFINE_BASE_DIR:-$AFFINE_TOOL_ROOT/affine_percituner_test}"
AFFINE_FP_TOOL="${AFFINE_FP_TOOL:-$AFFINE_TOOL_ROOT/fp_affine_error.py}"
AFFINE_ORDER_DIRECTION="${AFFINE_ORDER_DIRECTION:-desc}"

if [ -z "${REAL_PERCI_TUNER:-}" ]; then
    if [ -d "$AFFINE_TOOL_ROOT/../perci-tuner-alpha" ]; then
        REAL_PERCI_TUNER="$(cd "$AFFINE_TOOL_ROOT/../perci-tuner-alpha" && pwd)"
    else
        REAL_PERCI_TUNER="/home/lilei/workspace/perci-tuner-alpha"
    fi
fi

if [ ! -f "$AFFINE_FP_TOOL" ]; then
    echo "Affine tool not found: $AFFINE_FP_TOOL" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    THRESHOLDS=("$@")
else
    THRESHOLDS=()
    while IFS= read -r dir; do
        THRESHOLDS+=("$(basename "$dir")")
    done < <(find "$AFFINE_BASE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [ "${#THRESHOLDS[@]}" -eq 0 ]; then
    echo "No threshold directories found under $AFFINE_BASE_DIR" >&2
    exit 1
fi

WRAPPER_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/affine-perci-wrapper.XXXXXX")"
mkdir -p "$WRAPPER_ROOT/scripts"
cat >"$WRAPPER_ROOT/scripts/run.sh" <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail

: "${REAL_PERCI_TUNER:?REAL_PERCI_TUNER is required}"
: "${AFFINE_FP_TOOL:?AFFINE_FP_TOOL is required}"

ORDER_FILE="${AFFINE_ORDER_FILE:-variable_order.txt}"
ORDER_DIRECTION="${AFFINE_ORDER_DIRECTION:-desc}"

src=""
affine_args=()
for arg in "$@"; do
    case "$arg" in
        *.c|*.cpp)
            if [ -z "$src" ]; then
                src="$arg"
            fi
            ;;
        -arg=*)
            affine_args+=("$arg")
            ;;
    esac
done

if [ -z "$src" ]; then
    echo "[AFFINE ORDER] No C/C++ source argument found; running without -order." >&2
    exec sh "$REAL_PERCI_TUNER/scripts/run.sh" "$@"
fi

if [ ! -f "$src" ]; then
    echo "[AFFINE ORDER] Source not found: $src" >&2
    exit 1
fi

echo "[AFFINE ORDER] Generate $ORDER_FILE from $src direction=$ORDER_DIRECTION"
python3 "$AFFINE_FP_TOOL" "$src" \
    --order-out "$ORDER_FILE" \
    --order-direction "$ORDER_DIRECTION" \
    "${affine_args[@]}" \
    > affine_order.log 2> affine_order.err

exec sh "$REAL_PERCI_TUNER/scripts/run.sh" "$@" "-order=$ORDER_FILE"
WRAP
chmod +x "$WRAPPER_ROOT/scripts/run.sh"

cleanup() {
    rm -rf "$WRAPPER_ROOT"
}
trap cleanup EXIT

status=0
for thres in "${THRESHOLDS[@]}"; do
    log_dir="$AFFINE_BASE_DIR/$thres/_batch_logs"
    mkdir -p "$log_dir"

    for script in "$SCRIPT_DIR"/run_*.sh; do
        [ -f "$script" ] || continue
        prog="$(basename "$script" .sh)"
        prog="${prog#run_}"
        log="$log_dir/${prog}.log"

        echo "======================================================="
        echo "[BATCH AFFINE] thres=$thres prog=$prog order_direction=$AFFINE_ORDER_DIRECTION"
        echo "  script: $script"
        echo "  log   : $log"
        echo "======================================================="

        if PERCI_TUNER="$WRAPPER_ROOT" \
           REAL_PERCI_TUNER="$REAL_PERCI_TUNER" \
           AFFINE_FP_TOOL="$AFFINE_FP_TOOL" \
           AFFINE_ORDER_DIRECTION="$AFFINE_ORDER_DIRECTION" \
           bash "$script" "$thres" "$AFFINE_BASE_DIR" >"$log" 2>&1; then
            echo "[OK]   $prog thres=$thres"
        else
            rc=$?
            echo "[FAIL] $prog thres=$thres rc=$rc log=$log" >&2
            status=1
        fi
    done
done

exit "$status"
