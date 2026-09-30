#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: bash scripts/run_sep14.sh bird|cardboard inference|executor H13|H69|H69_augbase|FT_mixed_augbase|H69_TA|H30|S13|S69|S69_TA|S30|FT_mixed|FT_multi|FT-mixed-30|FT_mixed_13|BASE STEP [--dry-run] [--gripper-hard-latch|--gripper-settle] [--gripper-open-thr VALUE]" >&2
    exit 2
fi
export ARM=$1
ROLE=$2
MODEL=$3
STEP=$4
case "$ARM" in bird|cardboard) ;; *) echo "Unknown arm: $ARM" >&2; exit 2 ;; esac
case "$ROLE" in inference|executor) ;; *) echo "Unknown role: $ROLE" >&2; exit 2 ;; esac
shift 4
DRY_RUN=false
HARD_LATCH=false
GRIPPER_SETTLE=false
GRIPPER_OPEN_THR=""
while [[ $# -gt 0 ]]; do
    OPTION=$1
    case "$OPTION" in
        --dry-run) DRY_RUN=true ;;
        --gripper-hard-latch) HARD_LATCH=true ;;
        --gripper-settle) GRIPPER_SETTLE=true ;;
        --gripper-open-thr)
            if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                echo "--gripper-open-thr requires a non-negative numeric value" >&2
                exit 2
            fi
            GRIPPER_OPEN_THR=$2
            shift
            ;;
        *) echo "Unknown option: $OPTION" >&2; exit 2 ;;
    esac
    shift
done
if [[ "$GRIPPER_SETTLE" == true && "$HARD_LATCH" == true ]]; then
    echo "--gripper-settle cannot be combined with --gripper-hard-latch" >&2
    exit 2
fi
if [[ "$HARD_LATCH" == true && "$ROLE" != executor ]]; then
    echo "--gripper-hard-latch is only valid for executor" >&2
    exit 2
fi
if [[ -n "$GRIPPER_OPEN_THR" && "$ROLE" != executor ]]; then
    echo "--gripper-open-thr is only valid for executor" >&2
    exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/sep14_python.sh"
if [[ "$ROLE" == inference ]]; then
    EXTRA=()
    if [[ "$DRY_RUN" == true ]]; then EXTRA=(--dry-run); fi
    if [[ "$GRIPPER_SETTLE" == true ]]; then EXTRA+=(--gripper-settle); fi
    exec bash "$SCRIPT_DIR/infer_campaign_sep14.sh" "$MODEL" "$STEP" "${EXTRA[@]}"
fi
bash "$SCRIPT_DIR/infer_campaign_sep14.sh" "$MODEL" "$STEP" --dry-run >/dev/null
SESSION="${ARM}_sep14_${MODEL}_s${STEP}_$(date +%m%d_%H%M%S_%N)"
if [[ "$ARM" == bird ]]; then
    ARM_OPTIONS=(--step-mm 4 --gripper-grasp-lock)
else
    ARM_OPTIONS=(--step-mm 3)
fi
NODE=ros2_nodes/xarm_codiff_servo_deploy_executor.py
EXECUTOR_OPTIONS=()
if [[ "$HARD_LATCH" == true ]]; then
    NODE=ros2_nodes/xarm_codiff_servo_deploy_executor_hardlatch.py
    # Opt into the existing gripper gate without also enabling chunk resampling.
    EXECUTOR_OPTIONS=(--gripper-hard-latch --gripper-min-open-steps 15 --max-chunk-jump-mm 0)
    SESSION="${SESSION}_hardlatch"
fi
if [[ "$GRIPPER_SETTLE" == true ]]; then
    EXECUTOR_OPTIONS=(--gripper-settle --request-topic /codiff/request_gripper_settle
        --actions-topic /codiff/actions_gripper_settle --request-retry 0 --request-timeout 5)
    SESSION="${SESSION}_gripper_settle"
fi
if [[ ! -f "$REPO/$NODE" ]]; then
    printf 'Missing executor: %s\n' "$REPO/$NODE" >&2
    exit 1
fi
OPEN_THR=${GRIPPER_OPEN_THR:-500}
COMMAND=("$PY" "$NODE"
    --replan-steps 20 --max-steps 900 --max-step-mm 150 "${ARM_OPTIONS[@]}"
    --gripper-hysteresis --gripper-latch --gripper-open-thr "$OPEN_THR" --align-chunk-start
    "${EXECUTOR_OPTIONS[@]}")
cd "$REPO"
if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi
LOGS_ROOT=${LOGS_ROOT:-"$REPO/public-validation/hardware-logs"}
mkdir -p "$LOGS_ROOT"
printf 'Executor log label: %s step %s; verify the running inference node matches.\n' "$MODEL" "$STEP"
"${COMMAND[@]}" 2>&1 | tee "$LOGS_ROOT/${SESSION}_executor.log"
