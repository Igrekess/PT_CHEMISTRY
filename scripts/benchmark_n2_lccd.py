"""Phase 6.B.11a — N₂ LCCD energy benchmark.

Compares correlation energies along the cumulative-PT-order ladder:
    HF        → MP2 → MP3 (with ring) → LCCD (converged)
on the SZ basis (fast) and DZP basis (more virtuals, larger correction).
"""
from __future__ import annotations

import time

import numpy as np

from ptc.lcao.ccd import lccd_iterate
from ptc.lcao.cluster import build_explicit_cluster
from ptc.lcao.fock import density_matrix_PT_scf
from ptc.lcao.mp2 import mp2_at_hf
from ptc.lcao.mp3 import mp3_at_hf


def benchmark(basis_type: str,
                use_becke: bool = False,
                damp: float = 0.3,
                max_iter_lccd: int = 80) -> dict:
    Z = [7, 7]
    coords = np.array([[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]])
    bonds = [(0, 1, 3.0)]
    basis, topology = build_explicit_cluster(
        Z_list=Z, coords=coords, bonds=bonds, basis_type=basis_type,
    )

    if use_becke:
        scf_grid = dict(use_becke=True, n_radial=20, lebedev_order=26)
        op_grid = dict(n_radial=20, n_theta=10, n_phi=12,
                          use_becke=True, lebedev_order=26)
    else:
        scf_grid = dict(n_radial=12, n_theta=8, n_phi=10)
        op_grid = dict(n_radial=12, n_theta=8, n_phi=10,
                          use_becke=False, lebedev_order=14)

    print(f"\n=== N₂ {basis_type} ===")
    t0 = time.time()
    rho, S, eigvals, c, conv, _ = density_matrix_PT_scf(
        topology, basis=basis, mode="hf",
        max_iter=20, tol=1e-4, **scf_grid,
    )
    t_scf = time.time() - t0
    n_occ = int(round(basis.total_occ)) // 2
    print(f"  HF SCF: n_orb={basis.n_orbitals}, n_occ={n_occ}, "
          f"n_virt={basis.n_orbitals - n_occ}, t={t_scf:.1f}s")

    t0 = time.time()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **op_grid)
    t_mp2 = time.time() - t0

    t0 = time.time()
    mp3 = mp3_at_hf(
        basis, eigvals, c, n_occ, mp2_result=mp2,
        include_ring=True, **op_grid,
    )
    t_mp3 = time.time() - t0

    t0 = time.time()
    lccd = lccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=max_iter_lccd, tol=1e-7, damp=damp, verbose=True,
        **op_grid,
    )
    t_lccd = time.time() - t0

    return {
        "basis": basis_type,
        "n_orb": basis.n_orbitals,
        "e_mp2": mp2.e_corr,
        "e_mp3": mp3.e_corr,
        "e_lccd": lccd.e_corr,
        "lccd_n_iter": lccd.n_iter,
        "t_mp2": t_mp2,
        "t_mp3": t_mp3,
        "t_lccd": t_lccd,
    }


def main():
    print("=" * 76)
    print(" Phase 6.B.11a — N₂ LCCD energy benchmark (canonical HF refs)")
    print("=" * 76)
    rows = []
    for bt in ("SZ", "DZP"):
        r = benchmark(bt, use_becke=False, damp=0.3, max_iter_lccd=80)
        rows.append(r)

    print()
    print(f"{'Basis':<6} {'n_orb':>5} {'E_MP2':>9} {'E_MP3':>9} "
          f"{'E_LCCD':>9} {'iters':>5} "
          f"{'ΔLCCD/MP2':>11} {'t_LCCD':>8}")
    for r in rows:
        ratio = (r["e_lccd"] - r["e_mp2"]) / r["e_mp2"] * 100.0
        print(f"{r['basis']:<6} {r['n_orb']:>5} "
              f"{r['e_mp2']:>+9.4f} {r['e_mp3']:>+9.4f} "
              f"{r['e_lccd']:>+9.4f} {r['lccd_n_iter']:>5} "
              f"{ratio:>+10.1f}% {r['t_lccd']:>7.1f}s")
    print()
    print("LCCD = MP3 with infinite-order pp+hh+ring resummation.")
    print("ΔLCCD/MP2 typically 20-50%% on covalent systems.")


if __name__ == "__main__":
    main()
