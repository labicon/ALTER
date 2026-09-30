#!/usr/bin/env bash
# Shared local environment for Co-Diff launchers.
#
# Machine-specific artifact paths belong in .codiff.local.env at the repo root.
# This file intentionally has no hardcoded clone-specific paths.

if [[ -z "${REPO_ROOT:-}" ]]; then
  _codiff_env_source="${BASH_SOURCE[0]}"
  while [[ -L "$_codiff_env_source" ]]; do
    _codiff_env_dir="$(cd -P "$(dirname "$_codiff_env_source")" >/dev/null 2>&1 && pwd)"
    _codiff_env_source="$(readlink "$_codiff_env_source")"
    [[ "$_codiff_env_source" != /* ]] && _codiff_env_source="$_codiff_env_dir/$_codiff_env_source"
  done
  _codiff_env_dir="$(cd -P "$(dirname "$_codiff_env_source")" >/dev/null 2>&1 && pwd)"
  export REPO_ROOT="$(cd "$_codiff_env_dir/.." >/dev/null 2>&1 && pwd)"
  unset _codiff_env_source _codiff_env_dir
fi

if [[ -f "$REPO_ROOT/.codiff.local.env" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.codiff.local.env"
fi

export CODIFF_DATA_ROOT="${CODIFF_DATA_ROOT:-${DATASET_ROOT:-$REPO_ROOT/release-artifacts}}"
export DATASET_ROOT="${DATASET_ROOT:-$CODIFF_DATA_ROOT}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${CKPT_ROOT:-$CODIFF_DATA_ROOT/checkpoints/arm}}"
export CKPT_ROOT="${CKPT_ROOT:-$CHECKPOINT_ROOT}"
export ROLLOUT_ROOT="${ROLLOUT_ROOT:-$CODIFF_DATA_ROOT/envs/arm/data_gen/rollouts}"
export RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/public-validation/results}"
export DEBUG_ROOT="${DEBUG_ROOT:-$REPO_ROOT/public-validation/debug}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/public-validation/wandb}"
export LOGS_ROOT="${LOGS_ROOT:-$REPO_ROOT/public-validation/logs}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="python"
  else
    export PYTHON_BIN="python3"
  fi
else
  export PYTHON_BIN
fi
