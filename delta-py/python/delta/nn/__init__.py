import sys

from .. import _delta as _C

_mod = _C.nn

from .module import Module

for name in dir(_mod):
    if not name.startswith("_") and name not in globals():
        globals()[name] = getattr(_mod, name)

sys.modules["delta.nn.functional"] = _mod.functional

from .linear import Linear

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
