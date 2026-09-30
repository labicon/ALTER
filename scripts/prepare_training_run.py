#!/usr/bin/env python3
"""Prepare a fresh run from a materialized checkpoint's recorded training command.

Only interpreter/device/output locations change. This prints the command and
never launches training. Inspect the new job and command before executing it.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_release import digest
from scripts.experiment_contract import _stable_hash, load_pure_finetune_job_contract, load_capacity_ablation_job_contract


def prepare(contract_path, output, device, python):
    contract_path = Path(contract_path).resolve()
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(f'Refusing existing output: {output}')
    contract = json.loads(contract_path.read_text())
    if contract.get('release_derivation', {}).get('kind') != 'verified_path_and_metadata_relocation':
        raise ValueError('Use a verified materialized checkpoint contract')
    from scripts.experiment_contract import contract_integrity_mismatches
    artifacts = contract['artifacts']
    context = {('adapter_stats_path' if k == 'stats_path' else k): v for k, v in artifacts.items() if k.endswith('_path')}
    errors = contract_integrity_mismatches(contract, context)
    if errors:
        raise ValueError(f'Input contract failed integrity: {errors}')
    argv = list(contract['contract']['argv'])
    if not argv or not Path(argv[0]).is_file():
        raise ValueError('Run from the release repository root; recorded trainer is unavailable')
    def get(flag):
        return argv[argv.index(flag) + 1]
    def set_value(flag, value):
        if flag in argv:
            argv[argv.index(flag) + 1] = str(value)
        else:
            argv.extend([flag, str(value)])
    set_value('--checkpoint-dir', output / 'checkpoints')
    if '--stats-path' in argv:
        set_value('--stats-path', output / 'checkpoints' / Path(get('--stats-path')).name)
    set_value('--device', device)
    job = None
    flag = next((f for f in ['--pure-finetune-job-contract', '--capacity-job-contract'] if f in argv), None)
    if flag:
        original_job = Path(get(flag))
        if not original_job.is_file():
            raise FileNotFoundError(f'Missing sealed training job: {original_job}; recover its exact identity before preparing this run')
        report_root = next((p for p in contract_path.parents if (p/'materialization-report.json').is_file()), None)
        if report_root is None:
            raise ValueError('Missing materialization report')
        report = json.loads((report_root/'materialization-report.json').read_text())
        bundle_root = Path(report['bundle_root'])
        if digest(bundle_root/'release-manifest.json') != report['bundle_manifest_sha256']:
            raise ValueError('Source bundle manifest changed after materialization')
        file_records = {r['path']: r for r in report['files']}
        job_relative = str(original_job.relative_to(report_root))
        if digest(original_job) != file_records[job_relative]['materialized_sha256']:
            raise ValueError('Materialized job changed after verification')
        receipts = {r['path']: r for r in json.loads((bundle_root/'export-receipts.json').read_text())}
        job = json.loads(original_job.read_text())
        original_hash = digest(original_job)
        set_value(flag, output / 'job.json')
        job['output_dir'] = str(output / 'checkpoints')
        job['training_argv'] = argv
        job['command_sha256'] = _stable_hash(argv)
        if 'watcher_argv' in job:
            job['watcher_command_sha256'] = _stable_hash(job['watcher_argv'])
        job['release_derivation'] = {'materialized_job_sha256': original_hash, 'checkpoint_contract_sha256': digest(contract_path), 'changes': ['output locations', 'device', 'local input hashes', 'command seal']}
        # Input lists, grouping, hyperparameters and original regimes stay intact.
        for block_name, pairs in [('manifest', [('path','sha256')]), ('base_policy',[('model_path','model_sha256'),('stats_path','stats_sha256')]), ('frozen_base',[('model_path','base_model_sha256'),('stats_path','base_stats_sha256')])]:
            block = job.get(block_name)
            if not block:
                continue
            for path_key, hash_key in pairs:
                if path_key in block:
                    local_input = Path(block[path_key])
                    relative = str(local_input.relative_to(report_root))
                    if block.get(hash_key) != receipts[relative]['original_sha256']:
                        raise ValueError(f'Job input original identity differs: {block_name}/{path_key}')
                    if digest(local_input) != file_records[relative]['materialized_sha256']:
                        raise ValueError(f'Job input changed after materialization: {relative}')
                    block[hash_key] = digest(local_input)
    output.mkdir(parents=True)
    if job is not None:
        (output/'job.json').write_text(json.dumps(job, indent=2)+'\n')
        old = sys.argv
        try:
            sys.argv = argv
            loader = load_pure_finetune_job_contract if flag == '--pure-finetune-job-contract' else load_capacity_ablation_job_contract
            if flag == '--capacity-job-contract':
                loader(str(output/'job.json'), method=job['method'])
            else:
                loader(str(output/'job.json'))
        finally:
            sys.argv = old
    command = [python, *argv]
    (output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    (output/'derivation.json').write_text(json.dumps({'source_contract_sha256':digest(contract_path),'command':command,'source_argv':contract['contract']['argv']},indent=2)+'\n')
    return command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--python',default=sys.executable)
    args=p.parse_args()
    try:
        print(shlex.join(prepare(args.contract,args.output,args.device,args.python)))
    except (OSError,ValueError,KeyError) as error:
        p.exit(2,f'{error}\n')

if __name__=='__main__':main()
