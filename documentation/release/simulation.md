# Simulation workflows

Use the [installation profile](installation.md) and [artifact guide](artifacts.md)
first. Entry-point names and model formats remain compatible with existing
Co-Diff checkpoints. Run every command from the repository root.

| Workflow | Entry point | Required inputs |
| --- | --- | --- |
| Base policy pretraining | `envs/arm/train/train_singlearm_mixedfront_e2e_shoulder.py` | Exact 400-demo selection; architecture/configuration from matching base statistics |
| Base-policy distilled replay | `scripts/distill_forwardonly_singlearm.py` | Selected base checkpoint and statistics; simulator |
| ALTER adaptation | `envs/arm/train/train_mixed_coordination_e2e_shoulder.py` | Exact adaptation manifest, two-arm demonstrations, distilled replay, frozen base pair |
| From scratch | `envs/arm/train/train_twoarm_fromscratch_e2e_shoulder.py` | Exact adaptation manifest, demonstrations/replay, selected capacity configuration |
| FT-mixed / FT-multi | `envs/arm/train/train_pure_full_policy_finetune.py` | Exact manifest, base pair, separately sealed immutable job specification |
| Capacity experiments | `scripts/queue_twoarm_forwardonly_capacity_ablation.py` | Exact 5/mode manifest, base pair and recorded size/optimizer settings |
| Coordination evaluation | `scripts/run_sealed_placewipe_full_eval.py` | Selected checkpoint/statistics and strict matching contract; base pair for ALTER |
| Source retention | `scripts/run_sealed_placewipe_source_retention.py` | Selected checkpoint/statistics; base pair for ALTER |

Every entry point above supports `--help`. Required files must be available
before running. For full-policy FT checkpoints, the evaluation route is named
`fs`/`fromscratch`; that route describes inference, not the training method.

The paper coordination panel has 200 episodes, four modes, replanning every 15
steps, and a 1,800-step cap. The forward-only source panel has 100 place-return
and 100 wipe episodes, equally weighted. Specify `--protocol
twoarm_native_source200` explicitly: the retained source evaluator also supports
other historical protocols and its default is not this paper panel.

For a complete coordination panel (substitute the mapped paths):

```bash
python scripts/run_sealed_placewipe_full_eval.py --method coord \
  --checkpoint-path /path/to/head.pt --stats-path /path/to/head_stats.pkl \
  --base-model-path /path/to/base.pt --base-stats-path /path/to/base_stats.pkl \
  --device cuda:0 --output-root "$PWD/public-validation/coordination"
python scripts/run_sealed_placewipe_source_retention.py --method coord \
  --checkpoint-path /path/to/head.pt --stats-path /path/to/head_stats.pkl \
  --base-model-path /path/to/base.pt --base-stats-path /path/to/base_stats.pkl \
  --protocol twoarm_native_source200 --device cuda:0 \
  --output-root "$PWD/public-validation/source"
```

Use `--method fs` for FS, FT-mixed and FT-multi, omitting the base arguments.
Use `--method base` in the source evaluator for the frozen base. Do not override
strict contract errors or use a different similarly named checkpoint.

These panels are full evaluations, not installation tests. Short functional
checks are `python -m pytest -q` and launcher `--help`. A two-step simulator
rollout is only a renderer/inference check; it cannot establish task success.
Release acceptance uses loading, inference, one-step training and short simulation
checks. Full paper retraining and full evaluation panels are not required for
this release; their commands are provided for optional research use.

The preserved checkpoint contracts contain original training commands. Prepare
one in a fresh directory using a verified materialized checkpoint contract:

```bash
python scripts/prepare_training_run.py --contract /path/to/model.pt.contract.json \
  --output "$PWD/public-validation/new-training" --device cuda:0
```

This prints a shell-quoted command and writes `command.json`; it never launches
training. For FT and capacity runs it derives a new job specification, validates
original input identities, reseals relocated hashes/commands, and invokes the
existing strict job validator. All 18 selected adaptation configurations passed
this preparation check. Review the printed command before running it. Recorded
hyperparameters, sampling, architecture and budgets remain intact. Historical
watcher commands are provenance only; this helper does not start queue watchers
or automate checkpoint selection. Use the evaluation entry points explicitly.

For artifact loading and a short CPU training check across every selected model:

```bash
python scripts/validate_release_models.py --inputs /path/to/materialized \
  --output "$PWD/public-validation/model-check.json" --training-smoke
```

This tests one update and save/reload, not complete training. The base's original
training contract is absent from the inspected archive, so its exact historical
training command remains an evidence gap despite verified demonstrations and
weights. The base trainer supports directory inputs, but its `--manifest-path`
option is for a different historical corpus and must not be used for this
400-demonstration release selection. Do not infer missing historical training
settings from the trainer's current defaults.

The FT-mixed archive records a fresh 20-episode selection panel and later-step
tie breaking; the general results description mentions a 40-episode panel.
That discrepancy is pending author review. Selected checkpoints and published
values are preserved. No new success rates are claimed by the local tests.
