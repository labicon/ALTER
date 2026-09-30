"""Validate seven hardware checkpoint/statistics pairs on saved frames, without ROS."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--code', type=Path, default=Path(__file__).resolve().parents[1], help='Public code checkout (default: this repository)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    catalog = root / 'catalog/selected-models.json'
    if not catalog.is_file():
        parser.error(f'Missing hardware model catalog: {catalog}; use a verified hardware artifact root')
    models = json.loads(catalog.read_text())
    frames = sorted((root / 'provenance/smoke-inputs').glob('*.npz'))
    if len(frames) != 2:
        parser.error(f'Expected two saved camera frames in {root / "provenance/smoke-inputs"}')
    for name, model in models.items():
        for field in ('weights', 'stats'):
            if not (root / model[field]).is_file():
                parser.error(f'Missing {name} {field}: {root / model[field]}')
        dependency = model['inference_base_dependency']
        if dependency is not None and dependency not in models:
            parser.error(f'Missing base model entry {dependency} required by {name}')
    if args.output and args.output.exists():
        parser.error(f'Choose a fresh output report: {args.output}')
    sys.path.insert(0, str(args.code.resolve()))
    from hardware_training.run_numpy_pickle_compat import enable_compatibility
    enable_compatibility()
    import numpy as np
    import torch
    from hardware_training.bench_failure_states import load_stack, preprocess, sample_chunk
    torch.set_num_threads(4)
    results = []
    for name, model in models.items():
        dependency = model['inference_base_dependency']
        base_info = models[dependency] if dependency else model
        options = SimpleNamespace(coord_stats=str(root / model['stats']),
            coord_checkpoint=str(root / model['weights']), base_only=not bool(dependency),
            base_stats=str(root / base_info['stats']), base_checkpoint=str(root / base_info['weights']), n_steps=50)
        device = torch.device('cpu')
        base, head, mean, std, horizon = load_stack(options, device)
        samples = []
        for frame in frames:
            with np.load(frame) as data:
                rgb, pose = data['rgb'], data['pose']
            anchor = np.zeros(7, dtype=np.float32)
            anchor[:6] = pose[:6]
            torch.manual_seed(0)
            chunk = sample_chunk(base, head, preprocess(rgb, device), (anchor - mean) / std, 1.2, device, horizon)
            if chunk.shape != (20, 7) or not np.isfinite(chunk).all():
                raise ValueError('Invalid sample: ' + name)
            samples.append(dict(frame=frame.relative_to(root).as_posix(), shape=list(chunk.shape), finite=True, sample_sha256=hashlib.sha256(chunk.tobytes()).hexdigest()))
        results.append(dict(model=name, passed=True, samples=samples, base_dependency=dependency))
        print('OFFLINE LOAD/SAMPLE PASSED:', name, flush=True)
        del base, head
        gc.collect()
    report = dict(device='cpu', ros_initialized=False, denoising_steps=50, cfg=1.2,
        note='Load/sample compatibility only; not hardware success or complete training reproduction.',
        python=sys.version, packages={p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'numpy')}, results=results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    else:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
