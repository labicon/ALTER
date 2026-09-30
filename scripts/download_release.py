#!/usr/bin/env python3
"""Download selected, revision-pinned release files without overwriting conflicts."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import quote
from urllib.request import urlopen


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '\\' in relative or str(p) != relative:
        raise ValueError(f'Unsafe release path: {relative!r}')
    root = Path(root).absolute()
    target = root.joinpath(*p.parts)
    for item in [root, *root.parents, target, *target.parents]:
        if item.is_symlink():
            raise ValueError(f'Symlink destination is not allowed: {item}')
    return target


def download(manifest, bundles, destination, local_source=None):
    if manifest.get('schema') != 'alter.release.v1':
        raise ValueError('Unsupported release manifest schema')
    selected = []
    known = manifest['bundles']
    for name in bundles:
        if name not in known:
            raise ValueError(f'Unknown bundle {name!r}; available: {", ".join(known)}')
        bundle = known[name]
        if not bundle.get('files'):
            raise ValueError(f'Bundle {name!r} is unavailable; see release status')
        if local_source is None:
            if not re.fullmatch(r'[0-9a-f]{40}', bundle.get('revision', '')):
                raise ValueError(f'Bundle {name!r} needs a pinned 40-character Hub revision')
            if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', bundle.get('repo_id', '')):
                raise ValueError('Invalid Hub repository identifier')
            if bundle.get('repo_type') not in ('model', 'dataset'):
                raise ValueError('repo_type must be model or dataset')
        for row in bundle['files']:
            if not re.fullmatch(r'[0-9a-f]{64}', row['sha256']) or not isinstance(row['size'], int) or row['size'] < 0:
                raise ValueError('Invalid checksum or size')
            target = safe_path(destination, row['path'])
            safe_path(destination, row.get('remote_path', row['path']))
            selected.append((bundle, row, target))
    for control in manifest.get('control_files', []):
        owner = known[control['bundle']]
        if local_source is None and (not re.fullmatch(r'[0-9a-f]{40}', owner.get('revision', '')) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', owner.get('repo_id', '')) or owner.get('repo_type') not in ('model', 'dataset')):
            raise ValueError('Control-file repository requires a pinned revision and valid identity')
        if not re.fullmatch(r'[0-9a-f]{64}', control['sha256']) or not isinstance(control['size'], int) or control['size'] < 0:
            raise ValueError('Invalid control-file checksum or size')
        safe_path(destination, control.get('remote_path', control['path']))
        selected.append((owner, control, safe_path(destination, control['path'])))
    if manifest.get('control_files'):
        payload = (json.dumps(manifest, indent=2) + '\n').encode()
        row = {'path': 'release-manifest.json', 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), '_payload': payload}
        selected.append(({}, row, safe_path(destination, row['path'])))
    if len({p for _, _, p in selected}) != len(selected):
        raise ValueError('Duplicate destination paths in requested bundles')
    # Check every conflict before any download or installation.
    for _, row, target in selected:
        if target.exists() and (not target.is_file() or target.stat().st_size != row['size'] or digest(target) != row['sha256']):
            raise FileExistsError(f'Conflicting file; choose a fresh destination: {target}')
    counts = {'downloaded': 0, 'reused': 0}
    for bundle, row, target in selected:
        if target.exists():
            counts['reused'] += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.alter-download-', dir=target.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(fd, 'wb') as output:
                if '_payload' in row:
                    source = io.BytesIO(row['_payload'])
                elif local_source is not None:
                    source = safe_path(local_source, row['path']).open('rb')
                else:
                    prefix = 'datasets/' if bundle['repo_type'] == 'dataset' else ''
                    url = f"https://huggingface.co/{prefix}{bundle['repo_id']}/resolve/{bundle['revision']}/{quote(row.get('remote_path', row['path']), safe='/')}"
                    source = urlopen(url, timeout=60)
                with source:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != row['size'] or digest(temporary) != row['sha256']:
                raise ValueError(f'Checksum/size mismatch: {row["path"]}')
            # Atomic no-clobber installation; temporary bytes are independently copied.
            os.link(temporary, target)
            counts['downloaded'] += 1
        finally:
            temporary.unlink(missing_ok=True)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--bundles', nargs='+', required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--local-source', type=Path, help='Validate an approved offline bundle instead of contacting the Hub')
    args = p.parse_args()
    try:
        print(json.dumps(download(json.loads(args.manifest.read_text()), args.bundles, args.destination, args.local_source), indent=2))
    except (OSError, ValueError, KeyError) as error:
        p.exit(2, f'{error}\n')

if __name__ == '__main__':
    main()
