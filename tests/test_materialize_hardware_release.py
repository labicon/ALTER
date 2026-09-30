import hashlib
import json
import pickle

import numpy as np
import pytest

from scripts.materialize_hardware_release import materialize


def bundle(tmp_path):
    root = tmp_path / 'download'
    root.mkdir()
    payloads = {
        'hardware-models': {
            'hardware/stats.pkl': pickle.dumps({'action_mean': np.arange(7, dtype=np.float32),
                'directory': 'artifact://hardware/prepared/cache',
                'data_manifest_sha256': 'a' * 64, 'history': 'provenance://sha256/' + 'b' * 64}),
            'hardware/model.pt': b'unchanged model',
        },
        'hardware-training-data': {'hardware/prepared/cache/images.npy': b'unchanged images'},
    }
    manifest = {'schema': 'alter.release.v1', 'bundles': {}}
    receipts = []
    for name, contents in payloads.items():
        rows = []
        for path, data in contents.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            row = {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
            rows.append(row)
            receipts.append(dict(row, bundle=name, original_sha256=row['sha256']))
        manifest['bundles'][name] = {'files': rows}
    receipt = json.dumps(receipts).encode()
    (root / 'export-receipts.json').write_bytes(receipt)
    manifest['export_receipts_sha256'] = hashlib.sha256(receipt).hexdigest()
    (root / 'release-manifest.json').write_text(json.dumps(manifest))
    return root


def test_relocation_preserves_scientific_metadata_and_independent_model(tmp_path):
    root = bundle(tmp_path)
    output = tmp_path / 'materialized'
    materialize(root, output, ['hardware-models', 'hardware-training-data'])
    stats = pickle.loads((output / 'hardware/stats.pkl').read_bytes())
    np.testing.assert_array_equal(stats['action_mean'], np.arange(7, dtype=np.float32))
    assert stats['action_mean'].dtype == np.float32
    assert stats['directory'] == str(root / 'hardware/prepared/cache')
    assert stats['data_manifest_sha256'] == 'a' * 64
    assert stats['history'] == 'provenance://sha256/' + 'b' * 64
    original, copied = root / 'hardware/model.pt', output / 'hardware/model.pt'
    assert copied.read_bytes() == original.read_bytes()
    assert copied.stat().st_ino != original.stat().st_ino
    with pytest.raises(FileExistsError):
        materialize(root, output)


def test_inference_subset_keeps_unselected_data_reference(tmp_path):
    root = bundle(tmp_path)
    (root / 'hardware/prepared/cache/images.npy').unlink()
    output = tmp_path / 'model_only'
    report = materialize(root, output)
    stats = pickle.loads((output / 'hardware/stats.pkl').read_bytes())
    assert stats['directory'] == 'artifact://hardware/prepared/cache'
    assert report['unselected_artifact_references'] == ['artifact://hardware/prepared/cache']
    with pytest.raises(FileNotFoundError, match='download bundle hardware-training-data'):
        materialize(root, tmp_path / 'training', ['hardware-models', 'hardware-training-data'])
    assert not (tmp_path / 'training').exists()


def test_corruption_and_receipt_tampering_fail_before_output(tmp_path):
    root = bundle(tmp_path)
    (root / 'hardware/model.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Bundle content changed'):
        materialize(root, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
    (root / 'export-receipts.json').write_text('[]')
    with pytest.raises(ValueError, match='receipt checksum'):
        materialize(root, tmp_path / 'bad')


def test_symlink_source_rejected(tmp_path):
    root = bundle(tmp_path)
    target = root / 'hardware/model.pt'
    contents = target.read_bytes()
    target.unlink()
    outside = tmp_path / 'outside'
    outside.write_bytes(contents)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match='Symlink'):
        materialize(root, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
