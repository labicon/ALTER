import json

import numpy as np
import pytest
import torch

from hardware_training.stage_robustness_manifest import array_digest
from hardware_training.train_twoarm_standalone import make_model, prepare_manifest


def corpus(root):
    records = []
    for arm in (0, 1):
        for episode in range(39):
            directory = root / f'{arm}_{episode}'
            directory.mkdir()
            actions = np.arange(210, dtype=np.float32).reshape(30, 7) + arm
            images = np.zeros((30, 8, 8, 3), dtype=np.uint8)
            timestamps = np.arange(30, dtype=np.int64)
            np.save(directory / 'images.npy', images)
            np.savez(directory / 'trajectory.npz', actions=actions, timestamps_ns=timestamps)
            records.append(dict(task='twoarm', arm=arm, key=f'ta_arm{arm}_{episode}', directory=str(directory),
                                actions_sha256=array_digest(actions), camera_sha256=array_digest(images),
                                timestamps_sha256=array_digest(timestamps)))
    records.append(dict(task='bird', directory='/must/not/be/read'))
    path = root / 'source.json'
    path.write_text(json.dumps(dict(records=records, history_seconds=[1., .5, 0.])))
    return path


def test_excludes_singlearm_and_computes_twoarm_only_stats(tmp_path):
    source = corpus(tmp_path)
    destination = tmp_path / 'selected.json'
    mean, std, sigma = prepare_manifest(source, destination)
    selected = json.loads(destination.read_text())
    assert len(selected['records']) == 78
    assert all(record['task'] == 'twoarm' for record in selected['records'])
    assert selected['singlearm_task_mass'] == {}
    expected = np.concatenate([np.arange(210, dtype=np.float32).reshape(30, 7) + arm for arm in (0, 1) for _ in range(39)])
    np.testing.assert_array_equal(mean, expected.mean(axis=0))
    np.testing.assert_array_equal(std, expected.std(axis=0))
    assert sigma == pytest.approx(1., abs=1e-5)


def test_corrupt_images_rejected(tmp_path):
    source = corpus(tmp_path)
    np.save(tmp_path / '0_0/images.npy', np.ones((30, 8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match='Changed training data'):
        prepare_manifest(source, tmp_path / 'selected.json')


def test_missing_arm_partner_rejected(tmp_path):
    source = corpus(tmp_path)
    data = json.loads(source.read_text())
    data['records'][0]['key'] = 'ta_arm0_99'
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='membership mismatch'):
        prepare_manifest(source, tmp_path / 'selected.json')


def test_optional_singlearm_is_verified_and_retained(tmp_path):
    source = corpus(tmp_path)
    manifest = json.loads(source.read_text())
    manifest['records'][-1] = dict(manifest['records'][0], key='sa_bird_01', task='bird', arm=None)
    manifest['singlearm_task_mass'] = {'bird': 1}
    source.write_text(json.dumps(manifest))
    destination = tmp_path / 'mixed.json'
    prepare_manifest(source, destination, include_singlearm=True)
    selected = json.loads(destination.read_text())
    assert len(selected['records']) == 79
    assert selected['singlearm_task_mass'] == {'bird': 1}
    manifest['records'][-1]['camera_sha256'] = 'incorrect'
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Changed training data'):
        prepare_manifest(source, tmp_path / 'bad.json', include_singlearm=True)


def test_configurable_episode_budget(tmp_path):
    source = corpus(tmp_path)
    manifest = json.loads(source.read_text())
    manifest['records'] = [record for record in manifest['records']
                           if record['task'] == 'twoarm' and record['key'].endswith('_0')]
    source.write_text(json.dumps(manifest))
    prepare_manifest(source, tmp_path / 'one_pair.json', expected_episodes=1)
    with pytest.raises(ValueError, match='Expected'):
        prepare_manifest(source, tmp_path / 'incorrect_budget.json', expected_episodes=30)


def test_standalone_model_trains_and_checkpoint_roundtrips(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    stats = dict(sigma_data=1., d_model=32, n_heads=4, depth=1, dim_feedforward=64)
    model = make_model(stats, 'cpu', 2e-4)
    assert all(parameter.requires_grad for parameter in model.F.parameters())
    assert not any(parameter.requires_grad for parameter in model.F_ema.parameters())
    before = {name: parameter.detach().clone() for name, parameter in model.F.named_parameters()}
    for _ in range(8):
        loss, norm = model.update(torch.randn(2, 20, 7), None, torch.rand(2, 3, 128, 128))
        assert np.isfinite([loss, norm]).all()
    changed = [name for name, parameter in model.F.named_parameters() if not torch.equal(before[name], parameter)]
    assert any('conv1.weight' in name for name in changed)
    path = tmp_path / 'model.pt'
    model.save(path)
    loaded = make_model(stats, 'cpu', 2e-4)
    assert loaded.load(path)
    for key, tensor in model.F_ema.state_dict().items():
        assert torch.equal(tensor, loaded.F_ema.state_dict()[key])
