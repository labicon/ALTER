#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/infer_campaign_sep14.sh H13|H69|H69_TA|S13|S69|S69_TA|FT_mixed|FT_multi|H69_augbase|FT_mixed_augbase|H30|S30|FT-mixed-30|FT_mixed_13|BASE STEP [--dry-run] [--gripper-settle]" >&2
    exit 2
fi
MODEL=$1
STEP=$2
case "$MODEL" in H13|H69|H69_TA|S13|S69|S69_TA|FT_mixed|FT_multi|H69_augbase|FT_mixed_augbase|H30|S30|FT-mixed-30|FT_mixed_13|BASE) ;; *) echo "Unknown model: $MODEL" >&2; exit 2 ;; esac
if [[ ! "$STEP" =~ ^[1-9][0-9]{3,5}$ ]] || (( STEP % 5000 != 0 || STEP > 100000 )); then
    echo "STEP must be 5000, 10000, ..., 100000" >&2
    exit 2
fi
if [[ "$MODEL" == BASE && "$STEP" != 100000 ]]; then
    echo "BASE supports only step 100000: the frozen single-arm reference used by H69" >&2
    exit 2
fi
shift 2
DRY_RUN=false
GRIPPER_SETTLE=false
for OPTION in "$@"; do
    case "$OPTION" in
        --dry-run) DRY_RUN=true ;;
        --gripper-settle) GRIPPER_SETTLE=true ;;
        *) echo "Unknown option: $OPTION" >&2; exit 2 ;;
    esac
done
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ARM=${ARM:-bird}
case "$ARM" in bird|cardboard) ;; *) echo "ARM must be bird or cardboard" >&2; exit 2 ;; esac
source "$REPO/scripts/sep14_python.sh"
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the downloaded hardware checkpoint directory}
MODEL_DIR="$CHECKPOINT_ROOT/sep14_$MODEL"
BASE_DIR=${BASE_DIR:-"$CHECKPOINT_ROOT/joint_bird_cardboardfb_aug24fwd_aug16bwd_pruned_uw"}
case "$MODEL" in
    H69_augbase|FT_mixed_augbase) MODEL_DIR="$CHECKPOINT_ROOT/sep15_$MODEL" ;;
    H30|S30|FT-mixed-30|FT_mixed_13) MODEL_DIR="$CHECKPOINT_ROOT/sep16_$MODEL" ;;
esac
# Cardboard retains its September 14 installation beneath sep_14/.
# Prefer the explicit root layout whenever the selected folder exists.
if [[ ! -d "$MODEL_DIR" && -d "$CHECKPOINT_ROOT/sep_14/sep14_$MODEL" ]]; then
    MODEL_DIR="$CHECKPOINT_ROOT/sep_14/sep14_$MODEL"
fi
if [[ "$MODEL" == H69_augbase ]]; then
    BASE_DIR="$CHECKPOINT_ROOT/singlearm_expert_lightaug_sep15"
fi
IMAGE_TOPIC=${IMAGE_TOPIC:-/camera_top/camera_top/color/image_raw}
DEBUG_ROOT=${DEBUG_ROOT:-"$REPO/public-validation/hardware-dumps"}
SESSION="${ARM}_sep14_${MODEL}_s${STEP}_$(date +%m%d_%H%M%S_%N)"
if [[ "$MODEL" == BASE ]]; then
    MODEL_DIR="$BASE_DIR"
    PREFIX=singlearm_mixedfront_e2e_shoulder
    NODE=ros2_nodes/xarm_codiff_inference_node.py
    OPTIONS=(--checkpoint "$MODEL_DIR/${PREFIX}_step${STEP}.pt" --stats "$MODEL_DIR/${PREFIX}_stats.pkl")
elif [[ "$MODEL" == H* ]]; then
    PREFIX=mixed_coord_head_placewipe_hardware
    NODE=ros2_nodes/xarm_codiff_coordination_inference_node.py
    OPTIONS=(--coord-checkpoint "$MODEL_DIR/${PREFIX}_step${STEP}.pt" --coord-stats "$MODEL_DIR/${PREFIX}_stats.pkl"
        --base-checkpoint "$BASE_DIR/singlearm_mixedfront_e2e_shoulder_step100000.pt"
        --base-stats "$BASE_DIR/singlearm_mixedfront_e2e_shoulder_stats.pkl")
else
    PREFIX=twoarm_standalone_hardware
    NODE=ros2_nodes/xarm_codiff_inference_node.py
    OPTIONS=(--checkpoint "$MODEL_DIR/${PREFIX}_step${STEP}.pt" --stats "$MODEL_DIR/${PREFIX}_stats.pkl")
fi
REQUIRED_FILES=("$MODEL_DIR/${PREFIX}_step${STEP}.pt" "$MODEL_DIR/${PREFIX}_stats.pkl")
if [[ "$MODEL" != BASE ]]; then
    REQUIRED_FILES+=("$MODEL_DIR/DOWNLOAD_VERIFIED.json")
fi
if [[ "$MODEL" == H* ]]; then
    REQUIRED_FILES+=("$BASE_DIR/singlearm_mixedfront_e2e_shoulder_step100000.pt"
        "$BASE_DIR/singlearm_mixedfront_e2e_shoulder_stats.pkl")
fi
for REQUIRED_FILE in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$REQUIRED_FILE" ]]; then
        printf 'Missing required file: %s\nCheck CHECKPOINT_ROOT and BASE_DIR; see documentation/hardware/checkpoints.md.\n' "$REQUIRED_FILE" >&2
        exit 1
    fi
done
if ! command -v -- "$PY" >/dev/null 2>&1; then
    printf 'Python executable unavailable: %s\nSet PY to the existing local environment; see documentation/hardware/README.md.\n' "$PY" >&2
    exit 1
fi
if [[ "$GRIPPER_SETTLE" == true ]]; then
    OPTIONS+=(--fresh-frame-on-request --request-topic /codiff/request_gripper_settle
        --actions-topic /codiff/actions_gripper_settle)
    SESSION="${SESSION}_gripper_settle"
fi
# Preserve cardboard's original single-arm source conditioning.
# The six comparison models and bird keep the incoming two-arm profile.
if [[ "$MODEL" == BASE && "$ARM" == cardboard ]]; then
    INFERENCE_OPTIONS=(--on-request --anchor --anchor-gripper --prescale-hw 192 256 --cfg-w 1.2 --n-steps 20)
else
    INFERENCE_OPTIONS=(--on-request --anchor --publish-from-index 1 --prescale-hw 192 256 --cfg-w 1.2 --n-steps 50)
fi
COMMAND=("$PY" "$NODE" "${OPTIONS[@]}" --image-topic "$IMAGE_TOPIC"
    "${INFERENCE_OPTIONS[@]}" --dump-frames "$DEBUG_ROOT/$SESSION")
cd "$REPO"
if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi
LOGS_ROOT=${LOGS_ROOT:-"$REPO/public-validation/hardware-logs"}
mkdir -p "$LOGS_ROOT" "$DEBUG_ROOT"
"${COMMAND[@]}" 2>&1 | tee "$LOGS_ROOT/${SESSION}_inference.log"
