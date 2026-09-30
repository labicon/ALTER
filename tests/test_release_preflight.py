import subprocess
import sys
from pathlib import Path


def test_missing_artifacts_fail_before_simulator_or_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'output'
    proc = subprocess.run([sys.executable, str(root/'scripts/run_eval.py'), '--mode', 'coord-shoulderonly', '--checkpoint-path', str(tmp_path/'missing.pt'), '--adapter-stats-path', str(tmp_path/'missing.pkl'), '--output-dir', str(output)], cwd=root, text=True, capture_output=True)
    assert proc.returncode == 2
    assert 'Download the matching release bundle' in proc.stderr
    assert 'robosuite' not in proc.stderr
    assert not output.exists()
