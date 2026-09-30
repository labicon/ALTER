import hashlib
import pytest
from scripts.download_release import download


def manifest(payload=b'weights'):
    return {'schema': 'alter.release.v1', 'bundles': {'models': {'files': [
        {'path': 'models/a.pt', 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    ]}}}


def test_download_reuse_conflict_and_independent_bytes(tmp_path):
    source = tmp_path / 'source'
    (source / 'models').mkdir(parents=True)
    original = source / 'models/a.pt'
    original.write_bytes(b'weights')
    dest = tmp_path / 'dest'
    assert download(manifest(), ['models'], dest, source) == {'downloaded': 1, 'reused': 0}
    assert (dest / 'models/a.pt').stat().st_ino != original.stat().st_ino
    assert download(manifest(), ['models'], dest, source) == {'downloaded': 0, 'reused': 1}
    (dest / 'models/a.pt').write_bytes(b'changed')
    with pytest.raises(FileExistsError):
        download(manifest(), ['models'], dest, source)
    assert original.read_bytes() == b'weights'


def test_corrupt_source_leaves_no_final_file(tmp_path):
    (tmp_path / 'source/models').mkdir(parents=True)
    (tmp_path / 'source/models/a.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Checksum'):
        download(manifest(), ['models'], tmp_path / 'dest', tmp_path / 'source')
    assert not (tmp_path / 'dest/models/a.pt').exists()
    assert not list((tmp_path / 'dest').rglob('.alter-download-*'))


@pytest.mark.parametrize('path', ['../escape', '/absolute', 'a/../../escape', 'a\\b', 'a//b'])
def test_reject_unsafe_paths(tmp_path, path):
    m = manifest(); m['bundles']['models']['files'][0]['path'] = path
    with pytest.raises(ValueError, match='Unsafe'):
        download(m, ['models'], tmp_path, tmp_path)


def test_reject_symlink_and_unpinned_revision(tmp_path):
    (tmp_path / 'link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        download(manifest(), ['models'], tmp_path / 'link', tmp_path)
    with pytest.raises(ValueError, match='pinned'):
        download(manifest(), ['models'], tmp_path)
    with pytest.raises(ValueError, match='Unknown'):
        download(manifest(), ['missing'], tmp_path, tmp_path)
