"""Regression checks for hardware archives serialized with NumPy 2 module names."""
import importlib
import pickle
import sys

import numpy as np
import pytest

from hardware_training import run_numpy_pickle_compat as compat


def test_numpy2_array_payload_preserves_values_and_dtype(monkeypatch):
    name = "numpy._core.numeric"
    monkeypatch.delitem(sys.modules, name, raising=False)
    expected = np.array([[1.25, -2.5], [0.0, 4096.125]], dtype=np.float32)
    args = pickle.dumps((expected.tobytes(), expected.dtype, expected.shape, "C"), protocol=2)
    payload = b"\x80\x02cnumpy._core.numeric\n_frombuffer\n" + args[2:-1] + b"R."
    compat.enable_compatibility()
    actual = pickle.loads(payload)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_unrelated_import_failure_is_not_hidden(monkeypatch):
    real_import = importlib.import_module
    def broken(name):
        if name == "numpy._core.numeric":
            raise ModuleNotFoundError("missing another dependency", name="unrelated_dependency")
        return real_import(name)
    monkeypatch.setattr(compat.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError) as error:
        compat.enable_compatibility()
    assert error.value.name == "unrelated_dependency"
