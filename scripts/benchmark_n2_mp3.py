"""Phase 6.B.10 — N₂ σ_p^MP3-GIAO benchmark with the pVDZ-PT basis.

Stacks the third-order amplitude correction on the same Stanton-Gauss
machinery used in Phase 6.B.9. Tests whether the additional
particle-particle + hole-hole ladder closure moves Δ_total toward the
S-G 1996 +18% target.

Reference (Stanton-Gauss 1996, J. Chem. Phys. 104):
    σ_p^HF / cc-pVDZ        = -440 ppm
    σ_p^MP2 / cc-pVDZ       = -360 ppm
    Δ_MP2_total            = +18 % (correction REDUCES |σ_p|)
"""
from __future__ import annotations

import time

import numpy as np

from ptc.lcao.cluster import build_explicit_cluster
from ptc.lcao.fock import density_matrix_PT_scf
from ptc.lcao.mp3 import mp3_paramagnetic_shielding_coupled


def benchmark_mp3(basis_type: str,
                    zeta_method: str,
                    include_core: bool,
                    label: str,
                    use_becke: bool = True) -> dict:
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

    print(f"  [{label}] HF SCF on {basis.n_orbitals} orbitals ...")
    t0 = time.time()
    rho, S, eigvals, c, conv, _ = density_matrix_PT_scf(
        topology, basis=basis, mode="hf",
        max_iter=20, tol=1e-4, **scf_grid,
    )
    t_scf = time.time() - t0
    n_occ = int(round(basis.total_occ)) // 2
    print(f"  [{label}] HF done in {t_scf:.1f}s n_occ={n_occ}")

    probe = np.array([0.0, 0.0, 0.1])
    print(f"  [{label}] MP2 + MP3 + Stanton-Gauss CPHF ...")
    t0 = time.time()
    out = mp3_paramagnetic_shielding_coupled(
        basis, topology, eigvals, c, n_occ, probe,
        mp2_kwargs=op_grid,
        lagrangian_kwargs=op_grid,
        mp3_grid_kwargs=op_grid,
        z_vector_kwargs=dict(max_iter=15, tol=1e-5,
                                n_radial_grid=op_grid["n_radial"],
                                n_theta_grid=op_grid["n_theta"],
                                n_phi_grid=op_grid["n_phi"],
                                use_becke=use_becke,
                                lebedev_order=op_grid.get("lebedev_order", 14)),
        relax_kwargs=dict(n_radial=op_grid["n_radial"],
                            n_theta=op_grid["n_theta"],
                            n_phi=op_grid["n_phi"],
                            use_becke=use_becke,
                            lebedev_order=op_grid.get("lebedev_order", 14)),
        cphf_kwargs=dict(max_iter=12, tol=1e-4,
                            n_radial=op_grid["n_radial"],
                            n_theta=op_grid["n_theta"],
                            n_phi=op_grid["n_phi"]),
    )
    t_pipe = time.time() - t0
    print(f"  [{label}] MP3 done in {t_pipe:.1f}s")

    return {
        "label": label,
        "n_orb": basis.n_orbitals,
        "t_scf": t_scf,
        "t_pipeline": t_pipe,
        "sigma_p_HF": out["sigma_p_HF"],
        "sigma_p_MP2_full": out["sigma_p_MP2_full"],
        "sigma_p_MP3_full": out["sigma_p_MP3_full"],
        "delta_MP2_pct": out["delta_MP2_pct"],
        "delta_MP3_pct": out["delta_MP3_pct"],
        "e_corr_MP2": out["e_corr_MP2"],
        "e_corr_MP3": out["e_corr_MP3"],
    }


def main():
    print("=" * 80)
    print(" Phase 6.B.10 — N₂ σ_p^MP3-GIAO with pVDZ-PT contracted basis")
    print(" reference Stanton-Gauss 1996: σ^HF=-440, σ^MP2=-360, Δ=+18%")
    print("=" * 80)
    rows = []
    configs = [
        ("DZP",      "pt",            False, "DZP / pt"),
        ("pVDZ-PT",  "pt",            False, "pVDZ-PT / pt (6.B.8c)"),
        ("pVDZ-PT",  "pt-shielding",  True,  "pVDZ-PT / pt-shielding+core (6.B.8d)"),
    ]
    for bt, zm, ic, lab in configs:
        print()
        try:
            r = benchmark_mp3(bt, zm, ic, lab, use_becke=True)
            rows.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {lab} FAILED : {type(e).__name__} {e}")

    print()
    print(f"{'Config':<42} {'n_orb':>5} {'σ^HF':>8} {'σ^MP2':>8} "
          f"{'σ^MP3':>8} {'ΔMP2%':>7} {'ΔMP3%':>7}")
    for r in rows:
        print(f"{r['label']:<42} {r['n_orb']:>5} "
              f"{r['sigma_p_HF']:>+8.2f} {r['sigma_p_MP2_full']:>+8.2f} "
              f"{r['sigma_p_MP3_full']:>+8.2f} "
              f"{r['delta_MP2_pct']:>+7.2f} {r['delta_MP3_pct']:>+7.2f}")
    print()
    print("Stanton-Gauss 1996 / cc-pVDZ : σ^HF=-440, σ^MP2=-360, Δ=+18%")


if __name__ == "__main__":
    main()
