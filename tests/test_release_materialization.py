import hashlib
import json
import pytest
from scripts.materialize_release import materialize
from scripts.download_release import digest
from scripts.experiment_contract import contract_integrity_mismatches


def bundle(tmp_path, bad_original=False):
    root=tmp_path/'bundle'; root.mkdir()
    model=root/'model.pt'; model.write_bytes(b'test checkpoint')
    receipt={'path':'model.pt','bundle':'simulation-models','size':model.stat().st_size,'sha256':digest(model),'original_sha256':digest(model)}
    contract={'artifacts':{'checkpoint_path':'artifact://model.pt','checkpoint_sha256':'0'*64 if bad_original else digest(model)}, 'model':{'task_family':'placewipe'},'contract':{}}
    p=root/'model.pt.contract.json';p.write_text(json.dumps(contract))
    receipts=[receipt,{'path':p.name,'bundle':'simulation-provenance','size':p.stat().st_size,'sha256':digest(p),'original_sha256':'1'*64,'original_contract_integrity':'pass'}]
    r=root/'export-receipts.json';r.write_text(json.dumps(receipts))
    m={'schema':'alter.release.v1','export_receipts_sha256':digest(r),'bundles':{'all':{'files':[{k:x[k] for k in ['path','size','sha256']} for x in receipts]}}}
    (root/'release-manifest.json').write_text(json.dumps(m));return root


def test_derived_contract_preserves_science_and_strict_hash_validation(tmp_path):
    root=bundle(tmp_path);output=tmp_path/'materialized'
    report=materialize(root,output)
    contract=json.loads((output/'model.pt.contract.json').read_text())
    assert contract['model']=={'task_family':'placewipe'}
    assert contract['release_derivation']['original_contract_sha256']=='1'*64
    context={'checkpoint_path':str(output/'model.pt')}
    assert contract_integrity_mismatches(contract,context)==[]
    (output/'model.pt').write_bytes(b'corrupt')
    assert 'checkpoint_sha256' in contract_integrity_mismatches(contract,context)
    with pytest.raises(FileExistsError):materialize(root,output)


def test_original_identity_mismatch_is_not_resealed(tmp_path):
    root=bundle(tmp_path,bad_original=True)
    with pytest.raises(ValueError,match='Original artifact hash mismatch'):
        materialize(root,tmp_path/'out')


def test_corrupt_bundle_rejected_before_materialization(tmp_path):
    root=bundle(tmp_path);(root/'model.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Bundle content changed'):
        materialize(root,tmp_path/'out')
    assert not (tmp_path/'out').exists()
