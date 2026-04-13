"""ie_geo.py — Ionization Energy from polygon geometry.

Derives IE from the geometric shell operator (ShellPolygon / AtomicShell)
without calling the full atom.py engine.  Uses PT screening_action for
the effective charge, then modulates by the ejection amplitude of the
active polygon.

Zero adjustable parameters.

March 2026 — Persistence Theory
"""
from __future__ import annotations

import math

from ptc.constants import RY
from ptc.periodic import period
from ptc.shell_polygon import build_atomic_shell


def IE_geo_eV(Z: int) -> float:
    """Ionization energy (eV) from polygon geometry.

    IE = Ry × (Z_eff / per)² × ejection

    The ejection_amplitude now contains the unified PT formula
    (insight #30: I_Fisher - I_GFT), absorbing the former 2-loop
    self-energy correction.  No separate 2-loop term needed.

    Parameters
    ----------
    Z:
        Atomic number (1–118).

    Returns
    -------
    float
        Estimated first ionization energy in eV.
    """
    from ptc.atom import screening_action  # lazy import for Python 3.9 compat

    shell = build_atomic_shell(Z)
    per = period(Z)
    S = screening_action(Z)
    Z_eff = Z * math.exp(-S)
    ie_base = RY * (Z_eff / per) ** 2

    # Ejection from active polygon — unified (DC + pairing + I_Fisher - I_GFT).
    ej = shell.active_polygon.ejection_amplitude()

    return ie_base * ej


def benchmark_ie_geo() -> dict:
    """Benchmark IE_geo_eV against NIST first ionization energies.

    Returns
    -------
    dict with keys:
        count       : int — number of elements benchmarked
        mae_percent : float — mean absolute error in percent
        by_block    : dict[str, float] — MAE per block (s, p, d, f)
        rows        : list[dict] — per-element details
    """
    from ptc.data.experimental import IE_NIST
    from ptc.periodic import block_of

    rows = []
    block_errors: dict[str, list[float]] = {}

    for Z, ie_ref in sorted(IE_NIST.items()):
        if ie_ref <= 0:
            continue
        ie_calc = IE_geo_eV(Z)
        err_pct = abs(ie_calc - ie_ref) / ie_ref * 100.0
        blk = block_of(Z)
        block_errors.setdefault(blk, []).append(err_pct)
        rows.append({
            "Z": Z,
            "block": blk,
            "ie_ref": ie_ref,
            "ie_calc": ie_calc,
            "err_pct": err_pct,
        })

    mae = sum(r["err_pct"] for r in rows) / len(rows) if rows else 0.0
    by_block = {blk: sum(errs) / len(errs) for blk, errs in block_errors.items()}

    return {
        "count": len(rows),
        "mae_percent": mae,
        "by_block": by_block,
        "rows": rows,
    }
