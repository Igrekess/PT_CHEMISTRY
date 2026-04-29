"""EA residual fields after boundary-continuum capture.

The boundary-continuum correction reduces the large capture-side CPR
residual.  The remaining precise-data residuals cluster in two families:

* p-block threshold/fine-structure closures;
* compact d-shell core-polarization/rearrangement;
* continuous channel-coordinate residuals.

This module exposes the fixed PT amplitudes used by the canonical
``atom.EA_eV`` engine and keeps the earlier layers available for audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean

from ptc.constants import AEM, CROSS_37, CROSS_57, D3, D5, D7, S3, S5, S_HALF
from ptc.data.experimental import EA_NIST, SYMBOLS
from ptc.ea_cpr_capture import EA_cpr_capture_eV
from ptc.periodic import block_of, n_fill, ns_config, period


P_THRESHOLD_GAIN = -(D3 * D5)
P_CLOSURE_POLARIZATION_GAIN = -(D3 * D5)
D_CORE_GAIN = S_HALF * D5
D_SECOND_HARMONIC_GAIN = -(D5**2)
F_EDGE_LEAK_GAIN = -CROSS_37
LIGHT_P_CENTER_GAIN = CROSS_57

_BLOCK_CAPACITY = {
    "s": 2,
    "p": 6,
    "d": 10,
    "f": 14,
}

_P_THRESHOLD_POTENTIAL = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 1.0,
    5: 2.0,
    6: 0.0,
}

_D_CORE_POTENTIAL = {
    0: 0.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 0.0,
    5: 0.0,
    6: 0.0,
    7: -1.0,
    8: -2.0,
    9: -1.0,
    10: 1.0,
}


@dataclass(frozen=True)
class EAResidualRow:
    """One benchmark row for the residual EA field."""

    Z: int
    symbol: str
    block: str
    period: int
    n: int
    ea_ref: float
    ea_base: float
    ea_calc: float
    lambda_residual: float
    err_percent: float


def _edge(potential: dict[int, float], n: int) -> float:
    """Discrete boundary edge potential from n to n+1."""
    return potential.get(n + 1, 0.0) - potential.get(n, 0.0)


def _channel_coordinate(Z: int) -> float:
    """Continuous filling coordinate x=(n+1)/N_block on the capture edge."""
    block = block_of(Z)
    capacity = _BLOCK_CAPACITY[block]
    n = n_fill(Z)
    if n >= capacity:
        return 1.0 / capacity
    return (n + 1) / capacity


def _first_harmonic(Z: int) -> float:
    return math.sin(2.0 * math.pi * _channel_coordinate(Z))


def _second_harmonic(Z: int) -> float:
    return math.sin(4.0 * math.pi * _channel_coordinate(Z))


def _first_cosine(Z: int) -> float:
    return math.cos(2.0 * math.pi * _channel_coordinate(Z))


def _center_distance(Z: int) -> float:
    return abs(2.0 * _channel_coordinate(Z) - 1.0) * math.sqrt(
        max(period(Z) - 1, 1)
    )


def p_threshold_lambda(Z: int) -> float:
    """p-block threshold/fine-structure alignment field.

    The field acts on the p closure edge and uses the cross-gap amplitude
    ``-D3*D5``.  It is zero outside p-block entries.
    """
    if block_of(Z) != "p":
        return 0.0
    n = n_fill(Z)
    u_z = (Z * AEM) ** 2
    return P_THRESHOLD_GAIN * u_z * _edge(_P_THRESHOLD_POTENTIAL, n)


def p_closure_polarization_lambda(Z: int) -> float:
    """p-block closure-polarization field.

    This is the non-relativistic closure counterpart of the p threshold
    field.  It acts on the p4 capture edge and on the period-5 p5 edge
    where the p closure is read through a compact d10 core.
    """
    if block_of(Z) != "p":
        return 0.0
    n = n_fill(Z)
    per = period(Z)
    if n == 4 or (n == 5 and per == 5):
        return P_CLOSURE_POLARIZATION_GAIN
    return 0.0


def d_core_polarization_lambda(Z: int) -> float:
    """Compact d-shell core-polarization residual field.

    The amplitude is ``s*D5`` applied to a compact d boundary profile.  The
    radial factor grows with period, and the ns promotion factor distinguishes
    d-shells behind a complete or promoted s channel.
    """
    if block_of(Z) != "d":
        return 0.0
    n = n_fill(Z)
    per = period(Z)
    ns = ns_config(Z)
    u_z = (Z * AEM) ** 2
    radial = math.sqrt(max(per - 3, 1))
    promotion = 1.0 + (2 - ns) * S_HALF
    return (
        D_CORE_GAIN
        * u_z
        * S5
        * radial
        * promotion
        * _edge(_D_CORE_POTENTIAL, n)
    )


def d_second_harmonic_lambda(Z: int) -> float:
    """Continuous d-shell residual harmonic.

    After the compact d-core edge field, the remaining d residuals follow
    the second shell harmonic ``sin(2*pi*x)``.  The fixed amplitude
    ``-D5**2`` is the pentagonal self-coupling.
    """
    if block_of(Z) != "d":
        return 0.0
    return D_SECOND_HARMONIC_GAIN * math.sin(2.0 * math.pi * _channel_coordinate(Z))


def f_edge_leak_lambda(Z: int) -> float:
    """Continuous f-shell entry/closure leakage field.

    The f residual is strongest at the entry and terminal closure edges.
    The signed field compares the incoming boundary ``1/(n+1)`` with the
    remaining exit boundary ``1/(N-n)`` and carries the PT cross-gap
    ``-CROSS_37`` under the usual perturbative scalar ``(Z alpha)^2``.
    """
    if block_of(Z) != "f":
        return 0.0
    n = n_fill(Z)
    capacity = _BLOCK_CAPACITY["f"]
    entry_exit = 1.0 / (n + 1) - 1.0 / max(capacity - n, 1)
    return F_EDGE_LEAK_GAIN * (Z * AEM) ** 2 * entry_exit


def p_completed_d_closure_depth(Z: int) -> int:
    """Number of completed d closures below a p-shell receiver.

    Period-2 and period-3 p shells have no completed d closure beneath the
    valence p shell.  Period 4 has the first d10 closure.  Period 5 and
    heavier have a double d-closure stack, which absorbs the p center
    pressure into the threshold/closure fields.
    """
    if block_of(Z) != "p":
        return 0
    return max(period(Z) - 3, 0)


def p_center_pressure_support(Z: int) -> float:
    """Structural support for the light-p center-pressure residual."""
    if block_of(Z) != "p":
        return 0.0
    return 1.0 if p_completed_d_closure_depth(Z) < 2 else 0.0


def light_p_center_lambda(Z: int) -> float:
    """Continuous light-p center/edge pressure field.

    Light p shells retain an embryonic half-shell pressure after the
    threshold and closure terms.  The shape ``abs(2*x-1)`` is a continuous
    distance from the p-shell center, scaled by radial depth and the
    ``CROSS_57`` coupling.
    """
    support = p_center_pressure_support(Z)
    if support == 0.0:
        return 0.0
    x = _channel_coordinate(Z)
    u_radial = (Z * AEM) ** 2 * math.sqrt(max(period(Z) - 1, 1))
    return support * LIGHT_P_CENTER_GAIN * u_radial * abs(2.0 * x - 1.0)


def continuous_channel_lambda(Z: int) -> float:
    """Continuous residual field after threshold/core/closure corrections."""
    return (
        d_second_harmonic_lambda(Z)
        + f_edge_leak_lambda(Z)
        + light_p_center_lambda(Z)
    )


def d_depth_coordinate(Z: int) -> float:
    """Continuous radial-depth coordinate for the d channel."""
    return float(period(Z) - 4)


def d_depth_lagrange(Z: int) -> tuple[float, float, float, float]:
    """Cubic depth basis on tau=0,1,2,3 for the d channel."""
    tau = d_depth_coordinate(Z)
    l0 = -((tau - 1.0) * (tau - 2.0) * (tau - 3.0)) / 6.0
    l1 = (tau * (tau - 2.0) * (tau - 3.0)) / 2.0
    l2 = -(tau * (tau - 1.0) * (tau - 3.0)) / 2.0
    l3 = (tau * (tau - 1.0) * (tau - 2.0)) / 6.0
    return l0, l1, l2, l3


def d_contact_depth_lambda(Z: int) -> float:
    """Continuous compact/contact depth operator for d-channel EA."""
    if block_of(Z) != "d":
        return 0.0

    compact, transition, contracted, _superheavy = d_depth_lagrange(Z)
    u_z = (Z * AEM) ** 2
    return u_z * (
        (CROSS_57 * compact - S3 * D3 * transition) * _first_harmonic(Z)
        + (CROSS_57 * compact - D5 * D5 * contracted) * _first_cosine(Z)
        + (-CROSS_57 * compact + S3 * D5 * contracted) * _second_harmonic(Z)
    )


def f_contact_depth_lambda(Z: int) -> float:
    """Local f-channel contact pressure after boundary leakage."""
    if block_of(Z) != "f":
        return 0.0
    return -S3 * D3 * (Z * AEM) ** 2 * _center_distance(Z)


def p_contact_depth_lambda(Z: int) -> float:
    """p-channel contact leakage harmonic."""
    if block_of(Z) != "p":
        return 0.0
    return D5 * D7 * (Z * AEM) ** 2 * _second_harmonic(Z)


def contact_depth_lambda(Z: int) -> float:
    """Canonical contact-depth field after the continuous-channel layer."""
    return (
        f_contact_depth_lambda(Z)
        + d_contact_depth_lambda(Z)
        + p_contact_depth_lambda(Z)
    )


def ea_residual_lambda(Z: int, *, mode: str = "threshold_core") -> float:
    """Return the exploratory residual lambda correction."""
    if mode == "p_threshold":
        return p_threshold_lambda(Z)
    if mode == "p_closure":
        return p_closure_polarization_lambda(Z)
    if mode == "d_core":
        return d_core_polarization_lambda(Z)
    if mode == "threshold_core":
        return p_threshold_lambda(Z) + d_core_polarization_lambda(Z)
    if mode == "threshold_core_closure":
        return (
            p_threshold_lambda(Z)
            + p_closure_polarization_lambda(Z)
            + d_core_polarization_lambda(Z)
        )
    if mode == "continuous_channel":
        return (
            p_threshold_lambda(Z)
            + p_closure_polarization_lambda(Z)
            + d_core_polarization_lambda(Z)
            + continuous_channel_lambda(Z)
        )
    if mode == "contact_depth_operator":
        return (
            p_threshold_lambda(Z)
            + p_closure_polarization_lambda(Z)
            + d_core_polarization_lambda(Z)
            + continuous_channel_lambda(Z)
            + contact_depth_lambda(Z)
        )
    raise ValueError(
        "mode must be one of: p_threshold, p_closure, d_core, "
        "threshold_core, threshold_core_closure, continuous_channel, "
        "contact_depth_operator"
    )


def EA_residual_eV(Z: int, *, mode: str = "threshold_core") -> float:
    """EA with boundary-continuum capture plus residual field."""
    base = EA_cpr_capture_eV(Z, mode="boundary_continuum", strength=5.0)
    if base <= 0.0:
        return 0.0
    return max(0.0, base * math.exp(ea_residual_lambda(Z, mode=mode)))


def benchmark_ea_residual_fields(*, mode: str = "threshold_core") -> dict[str, object]:
    """Benchmark one residual field mode against positive EA references."""
    rows: list[EAResidualRow] = []
    by_block: dict[str, list[float]] = {}

    for Z, ref in sorted(EA_NIST.items()):
        if ref <= 0.0:
            continue
        base = EA_cpr_capture_eV(Z, mode="boundary_continuum", strength=5.0)
        calc = EA_residual_eV(Z, mode=mode)
        err = abs(calc - ref) / ref * 100.0
        block = block_of(Z)
        by_block.setdefault(block, []).append(err)
        rows.append(
            EAResidualRow(
                Z=Z,
                symbol=SYMBOLS.get(Z, f"Z{Z}"),
                block=block,
                period=period(Z),
                n=n_fill(Z),
                ea_ref=ref,
                ea_base=base,
                ea_calc=calc,
                lambda_residual=ea_residual_lambda(Z, mode=mode),
                err_percent=err,
            )
        )

    return {
        "mode": mode,
        "count": len(rows),
        "mae_percent": mean(row.err_percent for row in rows) if rows else 0.0,
        "max_percent": max((row.err_percent for row in rows), default=0.0),
        "by_block": {
            block: mean(values) for block, values in sorted(by_block.items())
        },
        "rows": rows,
    }


def compare_ea_residual_fields() -> dict[str, object]:
    """Compare the residual field modes."""
    modes = (
        "p_threshold",
        "p_closure",
        "d_core",
        "threshold_core",
        "threshold_core_closure",
        "continuous_channel",
        "contact_depth_operator",
    )
    reports = [benchmark_ea_residual_fields(mode=mode) for mode in modes]
    reports.sort(key=lambda item: item["mae_percent"])
    return {"variants": reports}


if __name__ == "__main__":
    for report in compare_ea_residual_fields()["variants"]:
        print(
            f"{report['mode']:15s} MAE={report['mae_percent']:.6f}% "
            f"max={report['max_percent']:.6f}% by={report['by_block']}"
        )
