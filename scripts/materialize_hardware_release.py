#!/usr/bin/env python3
"""Relocate verified hardware metadata without changing model/data bytes.

Only load trusted release bundles: statistics use Python pickle. Large data stay
in the independent download directory. Unselected artifact references stay URIs.
Historical provenance hashes remain unchanged in export receipts and metadata.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_release import digest, safe_path


def materialize(bundle_root, output, bundles=('hardware-models',)):
    root = Path(bundle_root).absolute()
    output = Path(output).absolute()
    safe_path(root, 'release-manifest.json')
    safe_path(output, 'materialization-report.json')
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    manifest = json.loads((root / 'release-manifest.json').read_text())
    if manifest.get('schema') != 'alter.release.v1':
        raise ValueError('Unsupported release schema')
    if len(set(bundles)) != len(bundles) or 'hardware-models' not in bundles:
        raise ValueError('Select hardware-models once, plus any required data bundles')
    known = manifest['bundles']
    if any(name not in known or not name.startswith('hardware-') for name in bundles):
        raise ValueError('Unknown hardware bundle selection')
    receipt = safe_path(root, 'export-receipts.json')
    if digest(receipt) != manifest['export_receipts_sha256']:
        raise ValueError('Export receipt checksum mismatch')
    receipts = json.loads(receipt.read_text())
    by_path = {row['path']: row for row in receipts}
    files = {}
    for bundle, description in known.items():
        for row in description['files']:
            name = row['path']
            safe_path(root, name)
            if name in files:
                raise ValueError(f'Duplicate artifact path: {name}')
            files[name] = dict(row, bundle=bundle)
    if len(by_path) != len(receipts) or set(files) != set(by_path):
        raise ValueError('Receipt/file inventory mismatch')
    selected = {p: row for p, row in files.items() if row['bundle'] in bundles}
    for name, row in selected.items():
        original = by_path[name]
        if any(original.get(k) != row[k] for k in ('sha256', 'size', 'bundle')):
            raise ValueError(f'Receipt mismatch: {name}')
        if not re.fullmatch(r'[0-9a-f]{64}', original.get('original_sha256', '')):
            raise ValueError(f'Missing original identity: {name}')
        source = safe_path(root, name)
        if not source.is_file():
            raise FileNotFoundError(f'Missing {name}; download bundle {row["bundle"]}')
        if source.stat().st_size != row['size'] or digest(source) != row['sha256']:
            raise ValueError(f'Bundle content changed: {name}')
    unresolved = set()
    def destination(name):
        matches = [row for path, row in files.items() if path == name or path.startswith(name.rstrip('/') + '/')]
        if not matches:
            raise ValueError(f'Artifact reference absent from inventory: {name}')
        if any(row['bundle'] not in bundles for row in matches):
            unresolved.add('artifact://' + name)
            return 'artifact://' + name
        data = [row['bundle'].endswith('-data') for row in matches]
        if any(data) and not all(data):
            raise ValueError(f'Ambiguous data/metadata directory reference: {name}')
        return str(safe_path(root if all(data) else output, name))
    def relocate(obj):
        if isinstance(obj, dict):
            return {relocate(k): relocate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [relocate(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(relocate(v) for v in obj)
        if isinstance(obj, str) and obj.startswith('artifact://'):
            return destination(obj.removeprefix('artifact://'))
        return obj
    # Decode and validate metadata references before creating any output.
    metadata = {}
    for name, row in selected.items():
        if row['bundle'].endswith('-data'):
            continue
        source = root / name
        if source.suffix == '.json':
            metadata[name] = relocate(json.loads(source.read_text()))
        elif source.suffix == '.pkl':
            metadata[name] = relocate(pickle.loads(source.read_bytes()))
    output.mkdir(parents=True)
    report = {'manifest_sha256': digest(root / 'release-manifest.json'),
              'bundles': list(bundles), 'verified_files': len(selected),
              'unselected_artifact_references': sorted(unresolved), 'files': []}
    for name, row in selected.items():
        if row['bundle'].endswith('-data'):
            continue
        target = safe_path(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in metadata:
            if target.suffix == '.json':
                target.write_text(json.dumps(metadata[name], indent=2) + '\n')
            else:
                with target.open('xb') as stream:
                    pickle.dump(metadata[name], stream, protocol=4)
        else:
            shutil.copyfile(root / name, target)
        report['files'].append({'path': name, 'portable_sha256': row['sha256'],
                                'materialized_sha256': digest(target)})
    (output / 'materialization-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundles', nargs='+', default=['hardware-models'])
    args = parser.parse_args()
    try:
        report = materialize(args.bundle_root, args.output, args.bundles)
        print(f"Verified {report['verified_files']} files; materialized {len(report['files'])} metadata/model files")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
