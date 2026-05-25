#!/usr/bin/env python3
import importlib
import runpy
import sys
import types
from pathlib import Path

REPO = Path('/home/wzzz/LPRNet')
EXTRA_PATHS = [
    REPO,
    REPO / 'src',
    REPO / 'src' / 'evaluation',
    REPO / 'src' / 'training',
    REPO / 'src' / 'utils',
    REPO / 'src' / 'manifest',
]
for p in reversed([str(x) for x in EXTRA_PATHS]):
    if p not in sys.path:
        sys.path.insert(0, p)

# Backward-compat alias: historical code imports from model.LPRNet,
# but current repo keeps single-head and multihead builders in src/LPRNet.py
# and src/LPRNet_multihead.py.
if 'model.LPRNet' not in sys.modules:
    single_mod = importlib.import_module('LPRNet')
    multi_mod = importlib.import_module('LPRNet_multihead')
    shim = types.ModuleType('model.LPRNet')
    for name in dir(single_mod):
        if not name.startswith('__'):
            setattr(shim, name, getattr(single_mod, name))
    for name in ['build_lprnet_multihead', 'FAMILY_HEADS']:
        setattr(shim, name, getattr(multi_mod, name))
    model_pkg = sys.modules.get('model')
    if model_pkg is None:
        model_pkg = types.ModuleType('model')
        model_pkg.__path__ = []
        sys.modules['model'] = model_pkg
    model_pkg.LPRNet = shim
    sys.modules['model.LPRNet'] = shim

if len(sys.argv) < 2:
    raise SystemExit('usage: run_lpr_python_entry.py <script.py> [args ...]')

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name='__main__')
