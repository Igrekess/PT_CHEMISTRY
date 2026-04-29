"""Precision audit helpers for the canonical PT electron affinity engine."""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean

from ptc.atom import EA_eV
from ptc.data.experimental import EA_NIST, SYMBOLS
from ptc.ea_geo import EA_geo_eV
from ptc.ea_cpr_capture import EA_cpr_capture_eV
from ptc.ea_residual_fields import EA_residual_eV
from ptc.periodic import block_of, n_fill, period


@dataclass(frozen=True)
class EAPrecisionRow:
    """One row in the EA precision audit."""

    Z: int
    symbol: str
    block: str
    period: int
    n: int
    ea_ref: float
    ea_calc: float
    signed_error_eV: float
    signed_error_percent: float
    abs_error_eV: float
    abs_error_percent: float


AB_INITIO_PRECISION_LADDER = (
    {
        "layer": "correlation",
        "classical_name": "multi-reference correlation / high-order CC",
        "pt_reading": "polygon capture/ejection plus threshold/core fields",
    },
    {
        "layer": "finite_mass",
        "classical_name": "finite nuclear mass / recoil / mass polarization",
        "pt_reading": "mass-weighted boundary motion of the capture edge",
    },
    {
        "layer": "relativistic_scalar",
        "classical_name": "mass-velocity + Darwin scalar relativistic terms",
        "pt_reading": "CPR surface transmission and (Z alpha)^2 residual fields",
    },
    {
        "layer": "fine_structure",
        "classical_name": "spin-orbit / Breit-Pauli threshold splitting",
        "pt_reading": "p-threshold and p-closure alignment fields",
    },
    {
        "layer": "qed",
        "classical_name": "radiative QED and Araki-Sucher-like contact terms",
        "pt_reading": "sub-residual local contact fields, not yet canonical",
    },
)


def _row_for(Z: int, calc: float, ref: float) -> EAPrecisionRow:
    signed_eV = calc - ref
    signed_pct = signed_eV / ref * 100.0
    return EAPrecisionRow(
        Z=Z,
        symbol=SYMBOLS.get(Z, f"Z{Z}"),
        block=block_of(Z),
        period=period(Z),
        n=n_fill(Z),
        ea_ref=ref,
        ea_calc=calc,
        signed_error_eV=signed_eV,
        signed_error_percent=signed_pct,
        abs_error_eV=abs(signed_eV),
        abs_error_percent=abs(signed_pct),
    )


def benchmark_ea_precision(model: str = "canonical") -> dict[str, object]:
    """Benchmark EA models on positive embedded EA references."""
    calculators = {
        "geo": EA_geo_eV,
        "boundary_continuum": lambda Z: EA_cpr_capture_eV(
            Z, mode="boundary_continuum", strength=5.0
        ),
        "threshold_core_closure": lambda Z: EA_residual_eV(
            Z, mode="threshold_core_closure"
        ),
        "canonical": EA_eV,
        "continuous_channel": lambda Z: EA_residual_eV(
            Z, mode="continuous_channel"
        ),
        "contact_depth_operator": lambda Z: EA_residual_eV(
            Z, mode="contact_depth_operator"
        ),
    }
    try:
        calculator = calculators[model]
    except KeyError as exc:
        allowed = ", ".join(sorted(calculators))
        raise ValueError(f"Unknown EA precision model '{model}'. Allowed: {allowed}") from exc

    rows = [
        _row_for(Z, calculator(Z), ref)
        for Z, ref in sorted(EA_NIST.items())
        if ref > 0.0
    ]
    by_block: dict[str, list[float]] = {}
    for row in rows:
        by_block.setdefault(row.block, []).append(row.abs_error_percent)

    return {
        "model": model,
        "count": len(rows),
        "mae_percent": mean(row.abs_error_percent for row in rows),
        "rmse_percent": math.sqrt(mean(row.abs_error_percent**2 for row in rows)),
        "max_percent": max(row.abs_error_percent for row in rows),
        "mae_eV": mean(row.abs_error_eV for row in rows),
        "bias_percent": mean(row.signed_error_percent for row in rows),
        "by_block": {
            block: mean(errors) for block, errors in sorted(by_block.items())
        },
        "rows": rows,
    }


def small_atom_precision_rows(max_Z: int = 18) -> list[EAPrecisionRow]:
    """Canonical EA residuals for first- and second-row positive references."""
    return [
        _row_for(Z, EA_eV(Z), ref)
        for Z, ref in sorted(EA_NIST.items())
        if ref > 0.0 and Z <= max_Z
    ]


def precision_ladder() -> tuple[dict[str, str], ...]:
    """Return the ab-initio precision ladder and the corresponding PT reading."""
    return AB_INITIO_PRECISION_LADDER


if __name__ == "__main__":
    for model in ("geo", "boundary_continuum", "threshold_core_closure", "canonical"):
        report = benchmark_ea_precision(model)
        print(
            f"{model:24s} N={report['count']} "
            f"MAE={report['mae_percent']:.6f}% "
            f"MAE_eV={report['mae_eV']:.6f} "
            f"max={report['max_percent']:.6f}%"
        )
