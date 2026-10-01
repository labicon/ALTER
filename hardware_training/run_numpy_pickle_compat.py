"""Opt-in process-local NumPy2 pickle compatibility for the NumPy1.26 runtime.

Usage: python run_numpy_pickle_compat.py MODULE_OR_SCRIPT [arguments...]
Dataset bytes and the installed NumPy package are not modified.
"""

import importlib
import runpy
import sys


def enable_compatibility():
    name = 'numpy._core.numeric'
    try:
        importlib.import_module(name)
        return False
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
    numeric = importlib.import_module('numpy.core.numeric')
    if not hasattr(numeric, '_frombuffer'):
        raise RuntimeError('NumPy numeric module lacks the required pickle reconstruction function')
    sys.modules[name] = numeric
    return True


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: run_numpy_pickle_compat.py MODULE_OR_SCRIPT [arguments...]')
    if enable_compatibility():
        print('Enabled process-local numpy._core.numeric pickle compatibility', file=sys.stderr, flush=True)
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    if target.endswith('.py'):
        runpy.run_path(target, run_name='__main__')
    else:
        runpy.run_module(target, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
