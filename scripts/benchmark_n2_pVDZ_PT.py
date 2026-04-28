"""Phase 6.B.8 — Quick N₂ σ_p^HF benchmark for pVDZ-PT contracted basis.

Skips MP2 (too slow) and reports HF-level paramagnetic shielding +
overlap-matrix conditioning, the two key Chantier 3 success criteria:
    1. cond(S)  < 1e5
    2. |σ_p^HF| > 220 ppm  (50 % of cc-pVDZ -440 reference)

Compares against existing DZP and TZP basis sets so the gain from the
contracted polarisation can be read off directly.
"""
from __future__ import annotations

import time

import numpy as np

from ptc.lcao.cluster import build_explicit_cluster
from ptc.lcao.fock import density_matrix_PT_scf
from ptc.lcao.giao import paramagnetic_shielding_iso


def benchmark(basis_type: str,
              use_becke: bool = False,
              zeta_method: str = "pt",
              include_core: bool = False) -> dict:
    Z = [7, 7]
    coords = np.array([[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]])
    bonds = [(0, 1, 3.0)]
    basis, topology = build_explicit_cluster(
        Z_list=Z, coords=coords, bonds=bonds, basis_type=basis_type,
        zeta_method=zeta_method, include_core=include_core,
    )

    if use_becke:
        scf_grid = dict(use_becke=True, n_radial=20, lebedev_order=26)
        op_grid = dict(n_radial=20, n_theta=10, n_phi=12,
                          use_becke=True, lebedev_order=26)
    else:
        scf_grid = dict(n_radial=12, n_theta=8, n_phi=10)
        op_grid = dict(n_radial=12, n_theta=8, n_phi=10,
                          use_becke=False, lebedev_order=14)

    t0 = time.time()
    rho, S, eigvals, c, conv, _ = density_matrix_PT_scf(
        topology, basis=basis, mode="hf",
        max_iter=20, tol=1e-4, **scf_grid,
    )
    t_scf = time.time() - t0
    n_e = int(round(basis.total_occ))

    eigS = np.linalg.eigvalsh(S)
    cond_S = float(eigS.max() / max(eigS.min(), 1e-30))

    probe = np.array([0.0, 0.0, 0.1])
    t0 = time.time()
    sigma_p = paramagnetic_shielding_iso(
        basis, eigvals, c, n_e, probe, use_giao=True, **op_grid,
    )
    t_para = time.time() - t0

    return {
        "basis_type": basis_type,
        "zeta_method": zeta_method,
        "include_core": include_core,
        "n_orb": basis.n_orbitals,
        "cond_S": cond_S,
        "lambda_min": float(eigS.min()),
        "sigma_p_HF": sigma_p,
        "t_scf": t_scf,
        "t_para": t_para,
        "scf_converged": bool(conv),
    }


def main():
    print("=" * 80)
    print(" Phase 6.B.8 — Chantier 3 quick benchmark on N₂ σ_p^HF (HF only)")
    print(" reference Stanton-Gauss 1996 cc-pVDZ HF: -440 ppm")
    print("=" * 80)
    rows = []
    configs = [
        ("DZP",      "pt",            False, "DZP / pt"),
        ("TZ2P",     "pt",            False, "TZ2P / pt"),
        ("pVDZ-PT",  "pt",            False, "pVDZ-PT / pt (Phase 6.B.8c)"),
        ("pVDZ-PT",  "pt-shielding",  True,  "pVDZ-PT / pt-shielding+core (6.B.8d)"),
        ("pVTZ-PT",  "pt",            False, "pVTZ-PT / pt (Phase 6.B.8e)"),
        ("pVTZ-PT",  "pt-shielding",  True,  "pVTZ-PT / pt-shielding+core (6.B.8d+e)"),
    ]
    for bt, zm, ic, lab in configs:
        print(f"\n  Running N₂ {lab} ...")
        try:
            r = benchmark(bt, use_becke=False, zeta_method=zm, include_core=ic)
            rows.append((lab, r))
            print(f"    n_orb={r['n_orb']}  cond(S)={r['cond_S']:.2e}  "
                  f"σ_p^HF={r['sigma_p_HF']:+.2f} ppm  "
                  f"t_scf={r['t_scf']:.1f}s")
        except Exception as e:
            print(f"    {lab} FAILED : {type(e).__name__} {e}")

    print()
    print(f"{'Configuration':<42} {'n_orb':>5} {'cond(S)':>10} "
          f"{'σ_p^HF':>10} {'% -440':>8}")
    for lab, r in rows:
        pct = (r["sigma_p_HF"] / -440.0) * 100.0
        print(f"{lab:<42} {r['n_orb']:>5} "
              f"{r['cond_S']:>10.2e} "
              f"{r['sigma_p_HF']:>+10.2f} "
              f"{pct:>+8.2f}")


if __name__ == "__main__":
    main()
