"""Verify RAM-cached arrays against the sealed corpus and relocate directories only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def array_digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def stage(root, cache, output):
    if output.exists():
        raise FileExistsError(output)
    original = root / 'config/manifest.json'
    manifest = json.loads(original.read_text())
    inventory = json.loads((root / 'metadata/TRANSFER_VERIFIED.json').read_text())
    file_hashes = {record['source']: record['sha256'] for record in inventory['files']}
    changed = []
    for record in manifest['records']:
        directory = cache / record['key']
        original_trajectory = '/' + str((Path(record['directory']) / 'trajectory.npz').relative_to(root / 'payload'))
        if hashlib.sha256((directory / 'trajectory.npz').read_bytes()).hexdigest() != file_hashes[original_trajectory]:
            raise ValueError('Trajectory file mismatch, including weights/indices: ' + record['key'])
        images = np.load(directory / 'images.npy', mmap_mode='r')
        with np.load(directory / 'trajectory.npz') as data:
            checks = dict(actions_sha256=array_digest(data['actions']),
                          timestamps_sha256=array_digest(data['timestamps_ns']),
                          camera_sha256=array_digest(images))
            if any(value != record[key] for key, value in checks.items()):
                raise ValueError('RAM cache hash mismatch: ' + record['key'])
            cached_record = json.loads((directory / 'record.json').read_text())
            if cached_record['actions_sha256'] != record['actions_sha256']:
                raise ValueError('Record mismatch: ' + record['key'])
        changed.append(dict(key=record['key'], source=record['directory'], cache=str(directory)))
        record['directory'] = str(directory)
    restored = json.loads(json.dumps(manifest))
    for record, change in zip(restored['records'], changed):
        record['directory'] = change['source']
    if restored != json.loads(original.read_text()):
        raise ValueError('Unexpected manifest change beyond directory relocation')
    output.write_text(json.dumps(manifest, indent=2))
    audit = dict(source_manifest=str(original), source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                 manifest=str(output), manifest_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                 records_verified=len(changed), changes=changed)
    output.with_suffix('.verified.json').write_text(json.dumps(audit, indent=2))
    print('Verified and relocated', len(changed), 'records', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    stage(args.root, args.cache, args.out)


if __name__ == '__main__':
    main()
