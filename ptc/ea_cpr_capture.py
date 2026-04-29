"""Exploratory EA capture-side CPR corrections.

This module tests whether the contraction-penetration resonance (CPR)
that improves IE also has a dual signature in electron affinity.

The canonical IE correction acts on the occupied vertex ``n``.  EA is a
capture observable, so the natural dual hypothesis is to evaluate the
same geometric field on the receiver vertex ``n + 1`` and apply it to the
capture amplitude, not to the IE screening action.

Status: research only.  Nothing here is used by ``ptc.atom.EA_eV``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

from ptc.constants import AEM, C3, D3, D5, D7, P2, S3, S_HALF
from ptc.data.experimental import EA_NIST, SYMBOLS
from ptc.ea_geo import EA_geo_eV
from ptc.periodic import block_of, l_of, n_fill, ns_config, period


EDGE_GAIN = 1.0 + S_HALF
BOUNDARY_GAIN = float(P2)


_CAPACITY_BY_L = {0: 2, 1: 6, 2: 10, 3: 14}


@dataclass(frozen=True)
class EACPRRow:
    """One benchmark row for an EA CPR variant."""

    Z: int
    symbol: str
    block: str
    period: int
    n: int
    ea_ref: float
    ea_base: float
    ea_calc: float
    lambda_cap: float
    base_err_percent: float
    signed_err_percent: float
    err_percent: float


def vertex_projector(n: int, N: int, targets: tuple[int, ...]) -> float:
    """Exact real idempotent projector on selected vertices of Z/NZ."""
    total = 0.0
    for a in targets:
        val = 1.0
        for k in range(1, N // 2 + 1):
            coeff = 1.0 if (N % 2 == 0 and k == N // 2) else 2.0
            val += coeff * math.cos(2.0 * math.pi * k * (n - a) / N)
        total += val / N
    return 0.0 if abs(total) < 1e-12 else total


def _cpr_geometric_action_at(Z: int, n_eval: int) -> float:
    """IE-CPR geometric field evaluated at an arbitrary channel vertex."""
    u_z = (Z * AEM) ** 2
    per = period(Z)
    l = l_of(Z)
    ns = ns_config(Z)
    S = 0.0

    # Heavy s penetration after compact d cores.
    if l == 0 and per in (5, 6):
        S -= (
            u_z * S3 * D3 * C3 / float(per - 2)
            * vertex_projector(n_eval, 2, (1, 2))
        )

    # Pre-spinor p resonance in the period-6 hexagon.
    if l == 1 and per == 6:
        S += (
            u_z * S3 * D3 * D5
            * (
                vertex_projector(n_eval, 6, (3,))
                - S_HALF * vertex_projector(n_eval, 6, (4,))
                + S_HALF * vertex_projector(n_eval, 6, (6,))
            )
        )

    # Lanthanide-contracted d receiver with ns2 support.
    if l == 2 and per == 6 and ns == 2:
        S += (
            u_z * S3 * D5 * D7
            * (
                vertex_projector(n_eval, 10, (1,))
                - vertex_projector(n_eval, 10, (2,))
                - S_HALF * vertex_projector(n_eval, 10, (4, 5))
                + S_HALF * vertex_projector(n_eval, 10, (7,))
            )
        )

    # Lanthanide f resonance on the heptagonal double-spin channel.
    if l == 3 and per == 6:
        S += (
            u_z * S3 * D7 * D3 * S_HALF
            * (
                -vertex_projector(n_eval, 14, (2, 5, 12, 13))
                + vertex_projector(n_eval, 14, (4, 6))
            )
        )

    return S


def _capture_capacity(Z: int) -> int:
    """Capacity of the active capture channel."""
    return _CAPACITY_BY_L.get(l_of(Z), 2)


def _vacancy_pressure(Z: int) -> float:
    """Residual vacancy pressure after capture.

    EA is a boundary observable: it is stronger when a capture still sees a
    significant open edge and weaker when the receiver is already almost a
    closed point.  The pressure is normalized by the active channel capacity.
    """
    n = n_fill(Z)
    N = _capture_capacity(Z)
    vacancies_after = max(0, N - (n + 1))
    return 1.0 + vacancies_after / float(N)


def _virtual_capture_support(Z: int) -> float:
    """Virtual channel support available to the captured electron."""
    l = l_of(Z)
    if l == 0:
        return S3
    if l == 1:
        return D5
    if l == 2:
        return D7
    return D7 * S_HALF


def _hund_capture_orientation(Z: int) -> float:
    """Hund/Pauli orientation of the receiver edge.

    Before and at half-fill, capture must preserve same-spin ordering and
    the CPR field is read with opposite orientation.  After half-fill,
    capture contributes to pairing/closure and the orientation is direct.
    """
    n_after = n_fill(Z) + 1
    half = _capture_capacity(Z) / 2.0
    return -1.0 if n_after <= half else 1.0


def surface_transmission_l(l: int) -> float:
    """Continuous surface transmission of orbital channel ``l``.

    With ``q = 2l + 1``, the exponent is ``(q - 1)(q - 3)/4 = l(l - 1)``.
    The degenerate s diameter (q=1) and the p triangle (q=3) are surface
    channels, while d/f polygons are progressively internalized.
    """
    if l < 0:
        raise ValueError("l must be non-negative")
    return math.exp(-float(l * (l - 1)))


def _surface_transmission(Z: int) -> float:
    """Continuous surface transmission of the active polygonal channel."""
    l = l_of(Z)
    return surface_transmission_l(l)


def _boundary_capture_field(Z: int) -> float:
    """PT boundary field for capture-side CPR.

    The field combines:

    - receiver vertex ``n+1``;
    - half-weighted boundary flux ``CPR(n+1)-CPR(n)``;
    - Hund/Pauli orientation;
    - virtual channel support;
    - residual vacancy pressure.
    """
    n = n_fill(Z)
    receiver = _cpr_geometric_action_at(Z, n + 1)
    occupied = _cpr_geometric_action_at(Z, n)
    edge = receiver - occupied
    return (
        _hund_capture_orientation(Z)
        * (receiver + S_HALF * edge)
        * _vacancy_pressure(Z)
        * (1.0 + _virtual_capture_support(Z))
    )


def ea_cpr_capture_lambda(
    Z: int,
    *,
    mode: str = "receiver",
    strength: float = EDGE_GAIN,
) -> float:
    """Return the exploratory EA capture-side CPR amplitude.

    Modes
    -----
    receiver:
        Evaluate the CPR field on the receiver vertex ``n+1``.
    occupied:
        Evaluate the same field on the occupied vertex ``n``.
    edge:
        Use the boundary flux ``CPR(n+1) - CPR(n)``.
    receiver_vacancy:
        Receiver field weighted by the residual vacancy fraction.
    boundary:
        Capture boundary field: receiver + half-edge flux, oriented by
        Hund/Pauli phase and weighted by virtual support plus vacancy
        pressure.
    boundary_continuum:
        Same boundary field, multiplied by the continuous polygonal
        surface transmission ``exp[-l(l-1)]``.

    ``strength`` is explicit because this is still a research channel.
    The default ``1+s`` is the minimal edge lift tested here.
    """
    n = n_fill(Z)
    receiver = _cpr_geometric_action_at(Z, n + 1)
    occupied = _cpr_geometric_action_at(Z, n)

    if mode == "receiver":
        raw = receiver
    elif mode == "occupied":
        raw = occupied
    elif mode == "edge":
        raw = receiver - occupied
    elif mode == "receiver_vacancy":
        raw = receiver * _vacancy_pressure(Z)
    elif mode == "boundary":
        raw = _boundary_capture_field(Z)
    elif mode == "boundary_continuum":
        raw = _surface_transmission(Z) * _boundary_capture_field(Z)
    else:
        raise ValueError(
            "mode must be one of: receiver, occupied, edge, "
            "receiver_vacancy, boundary, boundary_continuum"
        )

    return strength * raw


def EA_cpr_capture_eV(
    Z: int,
    *,
    mode: str = "receiver",
    strength: float = EDGE_GAIN,
) -> float:
    """Electron affinity with exploratory capture-side CPR applied."""
    base = EA_geo_eV(Z)
    if base <= 0.0:
        return 0.0
    lam = ea_cpr_capture_lambda(Z, mode=mode, strength=strength)
    return max(0.0, base * math.exp(lam))


def _mae(rows: Iterable[EACPRRow]) -> float:
    rows = list(rows)
    return sum(row.err_percent for row in rows) / len(rows) if rows else 0.0


def benchmark_ea_cpr_capture(
    *,
    mode: str = "receiver",
    strength: float = EDGE_GAIN,
) -> dict[str, object]:
    """Benchmark one EA CPR variant against positive NIST EA entries."""
    rows: list[EACPRRow] = []
    by_block_raw: dict[str, list[float]] = {}

    for Z, ea_ref in sorted(EA_NIST.items()):
        if ea_ref <= 0.0:
            continue
        ea_base = EA_geo_eV(Z)
        ea_calc = EA_cpr_capture_eV(Z, mode=mode, strength=strength)
        base_err = abs(ea_base - ea_ref) / ea_ref * 100.0
        signed_err = (ea_calc - ea_ref) / ea_ref * 100.0
        err = abs(ea_calc - ea_ref) / ea_ref * 100.0
        blk = block_of(Z)
        by_block_raw.setdefault(blk, []).append(err)
        rows.append(
            EACPRRow(
                Z=Z,
                symbol=SYMBOLS.get(Z, f"Z{Z}"),
                block=blk,
                period=period(Z),
                n=n_fill(Z),
                ea_ref=ea_ref,
                ea_base=ea_base,
                ea_calc=ea_calc,
                lambda_cap=ea_cpr_capture_lambda(
                    Z, mode=mode, strength=strength
                ),
                base_err_percent=base_err,
                signed_err_percent=signed_err,
                err_percent=err,
            )
        )

    return {
        "mode": mode,
        "strength": strength,
        "count": len(rows),
        "mae_percent": _mae(rows),
        "max_percent": max((row.err_percent for row in rows), default=0.0),
        "by_block": {
            blk: sum(values) / len(values)
            for blk, values in sorted(by_block_raw.items())
        },
        "rows": rows,
    }


def compare_ea_cpr_variants(
    modes: tuple[str, ...] = (
        "boundary_continuum",
        "boundary",
        "receiver",
        "occupied",
        "edge",
        "receiver_vacancy",
    ),
    strengths: tuple[float, ...] = (S_HALF, 1.0, EDGE_GAIN, 2.0, BOUNDARY_GAIN),
) -> dict[str, object]:
    """Run a small grid of research variants and return ranked results."""
    baseline_rows = []
    for Z, ea_ref in sorted(EA_NIST.items()):
        if ea_ref <= 0.0:
            continue
        ea_base = EA_geo_eV(Z)
        baseline_rows.append(abs(ea_base - ea_ref) / ea_ref * 100.0)
    baseline = {
        "mode": "baseline",
        "strength": 0.0,
        "count": len(baseline_rows),
        "mae_percent": (
            sum(baseline_rows) / len(baseline_rows) if baseline_rows else 0.0
        ),
        "max_percent": max(baseline_rows) if baseline_rows else 0.0,
    }

    variants = []
    for mode in modes:
        for strength in strengths:
            report = benchmark_ea_cpr_capture(mode=mode, strength=strength)
            variants.append({
                "mode": mode,
                "strength": strength,
                "count": report["count"],
                "mae_percent": report["mae_percent"],
                "max_percent": report["max_percent"],
                "by_block": report["by_block"],
            })

    variants.sort(key=lambda item: item["mae_percent"])
    return {"baseline": baseline, "variants": variants}


def signed_residual_correlation(
    *,
    mode: str = "receiver",
    strength: float = EDGE_GAIN,
) -> float:
    """Correlation between baseline signed residuals and CPR amplitudes.

    A useful capture correction should tend to have the opposite sign of
    the baseline residual.  Values near zero mean the projected field is not
    yet aligned with the remaining EA error structure.
    """
    signed = []
    lambdas = []
    for Z, ea_ref in sorted(EA_NIST.items()):
        if ea_ref <= 0.0:
            continue
        ea_base = EA_geo_eV(Z)
        signed.append((ea_base - ea_ref) / ea_ref * 100.0)
        lambdas.append(ea_cpr_capture_lambda(Z, mode=mode, strength=strength))

    if len(signed) < 2:
        return 0.0
    mean_signed = sum(signed) / len(signed)
    mean_lam = sum(lambdas) / len(lambdas)
    var_signed = sum((x - mean_signed) ** 2 for x in signed)
    var_lam = sum((x - mean_lam) ** 2 for x in lambdas)
    if var_signed <= 0.0 or var_lam <= 0.0:
        return 0.0
    cov = sum(
        (x - mean_signed) * (y - mean_lam)
        for x, y in zip(signed, lambdas)
    )
    return cov / math.sqrt(var_signed * var_lam)


if __name__ == "__main__":
    report = compare_ea_cpr_variants()
    base = report["baseline"]
    print(
        f"baseline  MAE={base['mae_percent']:.6f}% "
        f"max={base['max_percent']:.6f}%"
    )
    for row in report["variants"][:12]:
        print(
            f"{row['mode']:16s} strength={row['strength']:.6g} "
            f"MAE={row['mae_percent']:.6f}% max={row['max_percent']:.6f}%"
        )
    print(
        "signed residual correlation "
        f"{signed_residual_correlation():+.6f}"
    )
