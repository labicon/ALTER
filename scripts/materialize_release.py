#!/usr/bin/env python3
"""Materialize verified portable metadata and derive strictly checked local contracts.

Original hashes remain in export receipts and each derived contract. Scientific
fields and data-list ordering are retained. This does not certify missing records.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_release import digest, safe_path
from scripts.experiment_contract import contract_integrity_mismatches


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def materialize(bundle_root, output):
    bundle_root = Path(bundle_root).resolve()
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(f'Choose a fresh materialization directory: {output}')
    manifest = json.loads((bundle_root / 'release-manifest.json').read_text())
    receipt_path = bundle_root / 'export-receipts.json'
    if digest(receipt_path) != manifest.get('export_receipts_sha256'):
        raise ValueError('Export receipt checksum mismatch')
    receipts = json.loads(receipt_path.read_text())
    by_path = {r['path']: r for r in receipts}
    files = {r['path']: r for b in manifest['bundles'].values() for r in b['files']}
    if set(by_path) != set(files):
        raise ValueError('Receipt/file inventory mismatch')
    for name, row in files.items():
        source = safe_path(bundle_root, name)
        if row['sha256'] != by_path[name]['sha256'] or source.stat().st_size != row['size'] or digest(source) != row['sha256']:
            raise ValueError(f'Bundle content changed: {name}')
    def target(name):
        row = by_path.get(name)
        # Large demonstrations remain in the verified, independent download tree.
        root = bundle_root if row and row['bundle'].endswith('-data') else output
        return safe_path(root, name)
    def relocate(value):
        if isinstance(value, dict):
            return {relocate(k): relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, tuple):
            return tuple(relocate(v) for v in value)
        if isinstance(value, str):
            if value.startswith('artifact://'):
                name = value.removeprefix('artifact://')
                if name not in by_path:
                    # Directory roots can refer to either data or metadata.
                    data_children = [r for p, r in by_path.items() if p.startswith(name.rstrip('/') + '/')]
                    if data_children and all(r['bundle'].endswith('-data') for r in data_children):
                        return str(safe_path(bundle_root, name))
                return str(target(name))
            return value
        return value
    output.mkdir(parents=True)
    contracts = []
    report = {'bundle_root': str(bundle_root), 'bundle_manifest_sha256': digest(bundle_root / 'release-manifest.json'), 'files': [], 'derived_contracts': [], 'unresolved_metadata': []}
    for name, row in by_path.items():
        if row['bundle'].endswith('-data'):
            continue
        source = bundle_root / name
        dest = target(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.pt.contract.json'):
            contracts.append((name, json.loads(source.read_text())))
            continue
        if source.suffix == '.json':
            obj = json.loads(source.read_text())
            dest.write_text(json.dumps(relocate(obj), indent=2) + '\n')
        elif source.suffix == '.pkl':
            # Only use reviewed release bundles: pickle can execute Python code.
            obj = pickle.loads(source.read_bytes())
            with dest.open('xb') as stream:
                pickle.dump(relocate(obj), stream, protocol=4)
        else:
            shutil.copyfile(source, dest)
        report['files'].append({'path': name, 'source_sha256': row['sha256'], 'materialized_sha256': digest(dest)})
    for name, portable in contracts:
        row = by_path[name]
        if row.get('original_contract_integrity') != 'pass':
            raise ValueError(f'Original contract integrity not established: {name}')
        local = relocate(portable)
        originals = copy.deepcopy(portable.get('artifacts', {}))
        artifacts = local.get('artifacts', {})
        for path_key, hash_key in [('checkpoint_path', 'checkpoint_sha256'), ('stats_path', 'stats_sha256'), ('base_model_path', 'base_model_sha256'), ('base_stats_path', 'base_stats_sha256')]:
            uri = originals.get(path_key)
            expected = originals.get(hash_key)
            if not uri and not expected:
                continue
            if not uri or not expected or not uri.startswith('artifact://'):
                raise ValueError(f'Unresolved original artifact identity: {name}: {path_key}')
            artifact_name = uri.removeprefix('artifact://')
            receipt = by_path.get(artifact_name)
            if not receipt or receipt['original_sha256'] != expected:
                # An independently copied stats file may have the exact sealed bytes.
                matches = [r for r in receipts if r['original_sha256'] == expected]
                if not matches:
                    raise ValueError(f'Original artifact hash mismatch or unavailable: {name}: {path_key}')
                receipt = matches[0]
                artifact_name = receipt['path']
            artifacts[path_key] = str(target(artifact_name))
            artifacts[hash_key] = digest(target(artifact_name))
        local['release_derivation'] = {
            'kind': 'verified_path_and_metadata_relocation',
            'original_contract_sha256': row['original_sha256'],
            'portable_contract_sha256': row['sha256'],
            'original_artifact_hashes': {k: v for k, v in originals.items() if k.endswith('sha256')},
            'note': 'Historical provenance fields retain original hashes. Runtime artifact hashes and this contract identity are derived; no evaluation is relabeled.',
        }
        local.setdefault('contract', {}).pop('contract_id', None)
        local['contract'].pop('content_sha256', None)
        h = stable(local)
        local['contract'].update(content_sha256=h, contract_id=h[:16])
        context = {('adapter_stats_path' if k == 'stats_path' else k): v for k, v in artifacts.items() if k.endswith('_path')}
        errors = contract_integrity_mismatches(local, context)
        if errors:
            raise ValueError(f'Derived contract failed strict verification: {name}: {errors}')
        target(name).write_text(json.dumps(local, indent=2) + '\n')
        report['derived_contracts'].append({'path': name, 'original_sha256': row['original_sha256'], 'materialized_sha256': digest(target(name)), 'integrity_mismatches': errors})
    (output / 'materialization-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = materialize(args.bundle_root, args.output)
        print(f"Materialized {len(report['files'])} files and {len(report['derived_contracts'])} verified derived contracts")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'{error}\n')

if __name__ == '__main__':
    main()
