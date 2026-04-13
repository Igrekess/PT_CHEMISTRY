"""Root conftest.py — Python 3.9 compatibility fix for the PTC test suite.

ptc/__init__.py uses PEP 604 union-type syntax (``X | Y``) which requires
Python 3.10+.  This conftest pre-registers a stub for the ``ptc`` namespace
so that individual sub-module tests can import ``ptc.constants``,
``ptc.shell_polygon``, etc. directly without executing the broken
``ptc/__init__.py``.
"""
from __future__ import annotations

import sys
import types


def pytest_configure(config: object) -> None:  # noqa: ARG001
    """Pre-register a minimal ptc stub before any test module is collected."""
    if "ptc" not in sys.modules:
        stub = types.ModuleType("ptc")
        stub.__path__ = ["ptc"]  # type: ignore[assignment]
        stub.__package__ = "ptc"
        sys.modules["ptc"] = stub
