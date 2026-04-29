"""Atomic IE/EA residual diagnostics for the next PT precision layer.

This module is deliberately diagnostic.  It does not change the canonical
``IE_eV`` or ``EA_eV`` engines.  Its role is to expose the remaining
atomic residuals in PT coordinates so that future correction layers can be
derived before they are promoted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
from typing import Callable, Literal

from ptc.atom import EA_eV, IE_eV
from ptc.constants import AEM, CROSS_57, D3, D5, D7, S3
from ptc.data.experimental import EA_NIST, IE_NIST, SYMBOLS
from ptc.periodic import block_of, capacity, l_of, n_fill, ns_config, period


Observable = Literal["IE", "EA"]


@dataclass(frozen=True)
class AtomicResidualRow:
    """Signed residual of one atomic observable in PT shell coordinates."""

    observable: Observable
    Z: int
    symbol: str
    block: str
    period: int
    l: int
    n: int
    ns: int
    capacity: int
    reference_eV: float
    calculated_eV: float
    signed_error_eV: float
    signed_error_percent: float
    abs_error_eV: float
    abs_error_percent: float

    @property
    def u_z(self) -> float:
        """Perturbative relativistic envelope ``(Z alpha)^2``."""
        return (self.Z * AEM) ** 2

    @property
    def capture_coordinate(self) -> float:
        """Receiver coordinate on the active channel.

        Closed shells are read as virtual entry edges rather than as
        ``(N+1)/N``.  This keeps the diagnostic coordinate inside the
        channel circle.
        """
        if self.n >= self.capacity:
            return 1.0 / self.capacity
        return (self.n + 1) / self.capacity

    @property
    def target_lambda(self) -> float:
        """Action correction that would exactly remove the signed residual."""
        if self.calculated_eV <= 0.0 or self.reference_eV <= 0.0:
            return 0.0
        return math.log(self.reference_eV / self.calculated_eV)


@dataclass(frozen=True)
class FrontierCandidate:
    """One non-canonical residual mechanism worth deriving next."""

    name: str
    observable: Observable
    status: str
    pt_reading: str
    lambda_feature: Callable[[AtomicResidualRow], float]


def residual_rows(observable: Observable) -> list[AtomicResidualRow]:
    """Return signed residual rows for the current canonical engines."""
    if observable == "IE":
        table = IE_NIST
        calculator = IE_eV
    elif observable == "EA":
        table = {Z: v for Z, v in EA_NIST.items() if v > 0.0}
        calculator = EA_eV
    else:
        raise ValueError("observable must be 'IE' or 'EA'")

    rows: list[AtomicResidualRow] = []
    for Z, ref in sorted(table.items()):
        if ref <= 0.0:
            continue
        calc = calculator(Z)
        signed_eV = calc - ref
        signed_pct = signed_eV / ref * 100.0
        rows.append(
            AtomicResidualRow(
                observable=observable,
                Z=Z,
                symbol=SYMBOLS.get(Z, f"Z{Z}"),
                block=block_of(Z),
                period=period(Z),
                l=l_of(Z),
                n=n_fill(Z),
                ns=ns_config(Z),
                capacity=capacity(Z),
                reference_eV=ref,
                calculated_eV=calc,
                signed_error_eV=signed_eV,
                signed_error_percent=signed_pct,
                abs_error_eV=abs(signed_eV),
                abs_error_percent=abs(signed_pct),
            )
        )
    return rows


def benchmark_residuals(observable: Observable) -> dict[str, object]:
    """Aggregate current residuals by observable and block."""
    rows = residual_rows(observable)
    by_block: dict[str, list[AtomicResidualRow]] = {}
    for row in rows:
        by_block.setdefault(row.block, []).append(row)
    return {
        "observable": observable,
        "count": len(rows),
        "mae_percent": mean(row.abs_error_percent for row in rows),
        "mae_eV": mean(row.abs_error_eV for row in rows),
        "bias_percent": mean(row.signed_error_percent for row in rows),
        "max_percent": max(row.abs_error_percent for row in rows),
        "by_block": {
            block: mean(row.abs_error_percent for row in block_rows)
            for block, block_rows in sorted(by_block.items())
        },
        "rows": rows,
    }


def top_residuals(
    observable: Observable,
    *,
    limit: int = 12,
) -> list[AtomicResidualRow]:
    """Largest absolute residuals for an observable."""
    return sorted(
        residual_rows(observable),
        key=lambda row: row.abs_error_percent,
        reverse=True,
    )[:limit]


def _first_harmonic(row: AtomicResidualRow) -> float:
    return math.sin(2.0 * math.pi * row.capture_coordinate)


def _second_harmonic(row: AtomicResidualRow) -> float:
    return math.sin(4.0 * math.pi * row.capture_coordinate)


def _first_cosine(row: AtomicResidualRow) -> float:
    return math.cos(2.0 * math.pi * row.capture_coordinate)


def _center_distance(row: AtomicResidualRow) -> float:
    return abs(2.0 * row.capture_coordinate - 1.0) * math.sqrt(
        max(row.period - 1, 1)
    )


def _d_depth_coordinate(row: AtomicResidualRow) -> float:
    """Continuous depth coordinate for the d channel.

    The four observed d depths are period 4, 5, 6, and 7.  Reading them as
    ``tau = 0, 1, 2, 3`` allows a continuous depth kernel whose samples are
    the usual compact, transition, lanthanide-contracted, and superheavy
    d regimes.
    """
    return float(row.period - 4)


def _d_depth_lagrange(row: AtomicResidualRow) -> tuple[float, float, float, float]:
    """Cubic depth basis on tau=0,1,2,3 for the d channel."""
    tau = _d_depth_coordinate(row)
    l0 = -((tau - 1.0) * (tau - 2.0) * (tau - 3.0)) / 6.0
    l1 = (tau * (tau - 2.0) * (tau - 3.0)) / 2.0
    l2 = -(tau * (tau - 1.0) * (tau - 3.0)) / 2.0
    l3 = (tau * (tau - 1.0) * (tau - 2.0)) / 6.0
    return l0, l1, l2, l3


def _d_contact_depth_kernel(row: AtomicResidualRow) -> float:
    """Continuous compact/contact kernel for the d-channel residual."""
    if row.block != "d":
        return 0.0

    compact, transition, contracted, _superheavy = _d_depth_lagrange(row)
    h1 = _first_harmonic(row)
    h2 = _second_harmonic(row)
    c1 = _first_cosine(row)
    return row.u_z * (
        (CROSS_57 * compact - S3 * D3 * transition) * h1
        + (CROSS_57 * compact - D5 * D5 * contracted) * c1
        + (-CROSS_57 * compact + S3 * D5 * contracted) * h2
    )


def _ea_contact_projector_stack(row: AtomicResidualRow) -> float:
    """Fixed-constant diagnostic stack for the EA contact frontier.

    This is intentionally not canonical.  It collects the smallest fixed
    PT projections that push the EA residual below one percent while keeping
    the worst residual below the current canonical worst case.
    """
    if row.observable != "EA":
        return 0.0

    lam = 0.0
    if row.block == "f":
        lam += -S3 * D3 * row.u_z * _center_distance(row)
    if row.block == "d" and row.period == 6:
        lam += S3 * D5 * row.u_z * _second_harmonic(row)
        lam += -(D5 * D5) * row.u_z * _first_cosine(row)
    if row.block == "d" and row.period == 5:
        lam += -S3 * D3 * row.u_z * _first_harmonic(row)
    if row.block == "d" and row.period == 4:
        compact_kernel = (
            _first_harmonic(row)
            + _first_cosine(row)
            - _second_harmonic(row)
        )
        lam += CROSS_57 * row.u_z * compact_kernel
    if row.block == "p":
        lam += D5 * D7 * row.u_z * _second_harmonic(row)
    return lam


def _ea_contact_depth_operator(row: AtomicResidualRow) -> float:
    """Single fixed-constant contact operator using continuous depth."""
    if row.observable != "EA":
        return 0.0

    lam = 0.0
    if row.block == "f":
        lam += -S3 * D3 * row.u_z * _center_distance(row)
    if row.block == "d":
        lam += _d_contact_depth_kernel(row)
    if row.block == "p":
        lam += D5 * D7 * row.u_z * _second_harmonic(row)
    return lam


def depth_operator_minus_stack(row: AtomicResidualRow) -> float:
    """Difference between the depth operator and its sampled projector stack."""
    return _ea_contact_depth_operator(row) - _ea_contact_projector_stack(row)


def frontier_candidates() -> tuple[FrontierCandidate, ...]:
    """Return the current non-canonical precision-frontier hypotheses."""
    return (
        FrontierCandidate(
            name="f_contact_center",
            observable="EA",
            status="DIAG",
            pt_reading=(
                "f-shell contact/QED-like residual: local capture pressure "
                "around the heptagonal center, not yet a canonical term"
            ),
            lambda_feature=lambda row: (
                -S3 * D3 * row.u_z * _center_distance(row)
                if row.observable == "EA" and row.block == "f"
                else 0.0
            ),
        ),
        FrontierCandidate(
            name="d6_second_harmonic_depth",
            observable="EA",
            status="DIAG",
            pt_reading=(
                "period-6 d-shell second harmonic, likely the same "
                "contraction/penetration phase seen in IE"
            ),
            lambda_feature=lambda row: (
                S3 * D5 * row.u_z * _second_harmonic(row)
                if row.observable == "EA"
                and row.block == "d"
                and row.period == 6
                else 0.0
            ),
        ),
        FrontierCandidate(
            name="d4_compact_core_phase",
            observable="EA",
            status="DIAG",
            pt_reading=(
                "compact first d-shell phase; numerically visible before "
                "lanthanide contraction dominates"
            ),
            lambda_feature=lambda row: (
                -(D5 * D5) * row.u_z * _second_harmonic(row)
                if row.observable == "EA"
                and row.block == "d"
                and row.period == 4
                else 0.0
            ),
        ),
        FrontierCandidate(
            name="ea_contact_projector_stack",
            observable="EA",
            status="DIAG",
            pt_reading=(
                "combined fixed-constant contact projector stack; it crosses "
                "the 1 percent EA frontier but still needs a single-operator "
                "derivation before any canonical promotion"
            ),
            lambda_feature=_ea_contact_projector_stack,
        ),
        FrontierCandidate(
            name="ea_contact_depth_operator",
            observable="EA",
            status="DER-PHYS",
            pt_reading=(
                "same fixed PT contact field rewritten as a continuous "
                "d-depth operator; still bridge-level for EA capture, but no "
                "free coefficient or hard period gate remains inside d"
            ),
            lambda_feature=_ea_contact_depth_operator,
        ),
    )


def apply_candidate(
    observable: Observable,
    candidate: FrontierCandidate,
) -> dict[str, object]:
    """Evaluate one diagnostic candidate without modifying canonical engines."""
    rows = residual_rows(observable)
    corrected = []
    for row in rows:
        calc = row.calculated_eV * math.exp(candidate.lambda_feature(row))
        err = abs(calc - row.reference_eV) / row.reference_eV * 100.0
        corrected.append(err)
    return {
        "observable": observable,
        "candidate": candidate.name,
        "status": candidate.status,
        "mae_percent": mean(corrected),
        "max_percent": max(corrected),
    }


def candidate_report() -> dict[str, object]:
    """Compact report of the current atomic precision frontier."""
    return {
        "IE": benchmark_residuals("IE"),
        "EA": benchmark_residuals("EA"),
        "candidates": [
            apply_candidate(candidate.observable, candidate)
            for candidate in frontier_candidates()
        ],
    }


if __name__ == "__main__":
    for observable in ("IE", "EA"):
        bench = benchmark_residuals(observable)  # type: ignore[arg-type]
        print(
            f"{observable} N={bench['count']} "
            f"MAE={bench['mae_percent']:.6f}% "
            f"MAE_eV={bench['mae_eV']:.6f} "
            f"max={bench['max_percent']:.6f}%"
        )
        for row in top_residuals(observable, limit=8):  # type: ignore[arg-type]
            print(
                f"  {row.Z:3d} {row.symbol:>2} {row.block}{row.n:<2d} "
                f"p{row.period} err={row.signed_error_percent:+.4f}% "
                f"dE={row.signed_error_eV:+.5f} eV"
            )
    print("Candidates")
    for item in candidate_report()["candidates"]:
        print(
            f"  {item['candidate']:24s} "
            f"MAE={item['mae_percent']:.6f}% "
            f"max={item['max_percent']:.6f}%"
        )
