"""Phase 6.B.5 — chemical shifts from σ_total^MP2 with Stanton-Gauss closure.

Computes σ_total = σ_d + σ_p^coupled at the spectroscopically interesting
nucleus for several reference molecules at DZP Becke, then derives
chemical shifts δ = σ_ref - σ_sample. Comparing relative shifts to
experimental references is more robust than absolute σ_p magnitudes
because systematic basis-set errors largely cancel.

Run::

    PYTHONPATH=. python scripts/benchmark_chemical_shifts.py
"""
from __future__ import annotations

import time

import numpy as np

from ptc.lcao.cluster import build_explicit_cluster
from ptc.lcao.fock import density_matrix_PT_scf
from ptc.lcao.giao import shielding_diamagnetic_iso
from ptc.lcao.mp2 import (
    mp2_at_hf,
    mp2_density_relaxed_AO,
    mp2_lagrangian,
    mp2_paramagnetic_shielding_coupled,
    solve_z_vector,
)


def _benzene_xyz():
    r_CC, r_CH = 1.395, 1.090
    coords_C = [[r_CC * np.cos(2 * np.pi * k / 6),
                  r_CC * np.sin(2 * np.pi * k / 6), 0.0] for k in range(6)]
    coords_H = [[(r_CC + r_CH) * np.cos(2 * np.pi * k / 6),
                  (r_CC + r_CH) * np.sin(2 * np.pi * k / 6), 0.0] for k in range(6)]
    coords = np.array(coords_C + coords_H)
    Z = [6] * 6 + [1] * 6
    bonds = []
    for k in range(6):
        bonds.append((k, (k + 1) % 6, 1.5))
        bonds.append((k, 6 + k, 1.0))
    return Z, coords, bonds


SYSTEMS = {
    # Reference: CH4 ¹³C (δ = -2.3 ppm vs TMS, treated as our ¹³C reference)
    "CH4": dict(
        Z=[6, 1, 1, 1, 1],
        coords=np.array([
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [0.629, -0.629, -0.629],
            [-0.629, 0.629, -0.629],
            [-0.629, -0.629, 0.629],
        ]),
        bonds=[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)],
        nucleus_idx=0,
        probe_offset=np.array([0.0, 0.0, 0.05]),    # 0.05 Å above C
        nucleus_type="13C",
        delta_exp=-2.3,                               # vs TMS
    ),
    "CO": dict(
        Z=[6, 8],
        coords=np.array([[0.0, 0.0, 0.0], [1.128, 0.0, 0.0]]),
        bonds=[(0, 1, 3.0)],
        nucleus_idx=0,
        probe_offset=np.array([0.0, 0.0, 0.05]),    # near C
        nucleus_type="13C",
        delta_exp=+184.0,                             # vs TMS
    ),
    "Benzene": dict(
        Z=_benzene_xyz()[0],
        coords=_benzene_xyz()[1],
        bonds=_benzene_xyz()[2],
        nucleus_idx=0,
        probe_offset=np.array([0.05, 0.0, 0.0]),     # very close to first C
        nucleus_type="13C",
        delta_exp=+128.5,                             # vs TMS
    ),
    "H2O": dict(
        Z=[8, 1, 1],
        coords=np.array([
            [0.0, 0.0, 0.117],
            [0.757, 0.0, -0.470],
            [-0.757, 0.0, -0.470],
        ]),
        bonds=[(0, 1, 1.0), (0, 2, 1.0)],
        nucleus_idx=0,
        probe_offset=np.array([0.0, 0.05, 0.0]),     # near O
        nucleus_type="17O",
        delta_exp=0.0,                                # H2O is the ¹⁷O ref
    ),
    "N2": dict(
        Z=[7, 7],
        coords=np.array([[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]]),
        bonds=[(0, 1, 3.0)],
        nucleus_idx=0,
        probe_offset=np.array([0.0, 0.0, 0.05]),
        nucleus_type="15N",
        delta_exp=+302.0,                             # vs CH3NO2 standard
    ),
}


def benchmark(name: str, basis_type: str = "DZP",
                 use_becke: bool = True,
                 n_radial: int = 18, lebedev_order: int = 26) -> dict:
    """σ_d + σ_p^MP2_full at one nucleus. Both via MP2-relaxed density."""
    sys_def = SYSTEMS[name]
    basis, topology = build_explicit_cluster(
        Z_list=sys_def["Z"], coords=sys_def["coords"],
        bonds=sys_def["bonds"], basis_type=basis_type,
    )

    op_grid = dict(n_radial=n_radial, n_theta=10, n_phi=12,
                      use_becke=use_becke,
                      lebedev_order=lebedev_order if use_becke else 14)

    print(f"  --- {name} ({sys_def['nucleus_type']}, basis={basis_type}) ---")
    t0 = time.time()
    rho, S, eigvals, c, _, _ = density_matrix_PT_scf(
        topology, basis=basis, mode="hf",
        max_iter=20, tol=1e-4,
        **(dict(use_becke=True, n_radial=n_radial,
                  lebedev_order=lebedev_order) if use_becke
              else dict(n_radial=12, n_theta=8, n_phi=10)),
    )
    n_occ = int(round(basis.total_occ)) // 2

    nucleus_pos = sys_def["coords"][sys_def["nucleus_idx"]]
    probe = nucleus_pos + sys_def["probe_offset"]

    # MP2 + Lagrangian + Z-vector
    mp2_result = mp2_at_hf(basis, eigvals, c, n_occ, **op_grid)
    L = mp2_lagrangian(basis, c, n_occ, mp2_result, **op_grid)
    z = solve_z_vector(
        basis, eigvals, c, n_occ, L,
        max_iter=15, tol=1e-5,
        n_radial_grid=op_grid["n_radial"],
        n_theta_grid=op_grid["n_theta"],
        n_phi_grid=op_grid["n_phi"],
        use_becke=use_becke,
        lebedev_order=op_grid.get("lebedev_order", 14),
    )

    # σ_d at probe with relaxed MP2 density
    rho_HF = 2.0 * c[:, :n_occ] @ c[:, :n_occ].T
    rho_corr = mp2_density_relaxed_AO(c, n_occ, mp2_result, z)
    rho_MP2 = rho_HF + rho_corr
    sigma_d_HF = shielding_diamagnetic_iso(rho_HF, basis, probe,
                                              n_radial=op_grid["n_radial"],
                                              n_theta=op_grid["n_theta"],
                                              n_phi=op_grid["n_phi"])
    sigma_d_MP2 = shielding_diamagnetic_iso(rho_MP2, basis, probe,
                                                n_radial=op_grid["n_radial"],
                                                n_theta=op_grid["n_theta"],
                                                n_phi=op_grid["n_phi"])

    # σ_p at probe with full Stanton-Gauss closure
    out = mp2_paramagnetic_shielding_coupled(
        basis, topology, eigvals, c, n_occ, probe,
        mp2_result=mp2_result, z_vector=z,
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
    sigma_p_HF = out["sigma_p_HF"]
    sigma_p_MP2 = out["sigma_p_MP2_full"]

    sigma_total_HF = sigma_d_HF + sigma_p_HF
    sigma_total_MP2 = sigma_d_MP2 + sigma_p_MP2
    t_total = time.time() - t0

    print(f"  σ_d^HF   = {sigma_d_HF:+9.3f}    σ_d^MP2  = {sigma_d_MP2:+9.3f} ppm")
    print(f"  σ_p^HF   = {sigma_p_HF:+9.3f}    σ_p^MP2  = {sigma_p_MP2:+9.3f} ppm")
    print(f"  σ_tot^HF = {sigma_total_HF:+9.3f}    σ_tot^MP2= {sigma_total_MP2:+9.3f} ppm")
    print(f"  ({t_total:.1f}s)")
    return {
        "name": name,
        "n_orb": basis.n_orbitals,
        "nucleus_type": sys_def["nucleus_type"],
        "probe": probe.tolist(),
        "sigma_d_HF": sigma_d_HF,
        "sigma_d_MP2": sigma_d_MP2,
        "sigma_p_HF": sigma_p_HF,
        "sigma_p_MP2": sigma_p_MP2,
        "sigma_total_HF": sigma_total_HF,
        "sigma_total_MP2": sigma_total_MP2,
        "delta_exp": sys_def["delta_exp"],
        "t_total": t_total,
    }


def main():
    print("=" * 78)
    print(" Chemical shifts δ from σ_total^MP2 (Phase 6.B.5)")
    print("=" * 78)
    print()
    print(" Strategy: compute σ_total = σ_d + σ_p^MP2_full at each nucleus,")
    print(" then derive δ = σ_ref - σ_sample (vs CH₄ for ¹³C).")
    print()

    # Run benchmarks (CH4 first, so we can use it as ¹³C reference)
    rows = {}
    for name in ("CH4", "CO", "Benzene", "H2O", "N2"):
        rows[name] = benchmark(name)
        print()

    # Chemical shifts
    print("=" * 78)
    print("Chemical shifts (PT-pure DZP Becke, σ_total^MP2)")
    print("=" * 78)
    sigma_ref_C = rows["CH4"]["sigma_total_MP2"]
    print()
    print(f"{'System':<10} {'Nuc':<5} {'σ_total^HF':>12} {'σ_total^MP2':>12} "
            f"{'δ_calc':>9} {'δ_exp':>9} {'Δ':>9}")
    print("-" * 78)
    for name, r in rows.items():
        if r["nucleus_type"] == "13C":
            delta_calc = sigma_ref_C - r["sigma_total_MP2"]
            delta_exp = r["delta_exp"]
            err = delta_calc - delta_exp
            print(f"{name:<10} {r['nucleus_type']:<5} "
                    f"{r['sigma_total_HF']:>+12.3f} "
                    f"{r['sigma_total_MP2']:>+12.3f} "
                    f"{delta_calc:>+9.2f} {delta_exp:>+9.2f} {err:>+9.2f}")
        else:
            print(f"{name:<10} {r['nucleus_type']:<5} "
                    f"{r['sigma_total_HF']:>+12.3f} "
                    f"{r['sigma_total_MP2']:>+12.3f} "
                    f"{'(no ¹³C ref)':>9} {r['delta_exp']:>+9.2f} {'—':>9}")
    print()
    print(f"Reference: σ_C(CH₄)^MP2 = {sigma_ref_C:+.3f} ppm")
    print(f"  Literature: σ_C(TMS)^exp ≈ 192 ppm absolute, δ_C(CH₄ vs TMS) ≈ -2.3 ppm")


if __name__ == "__main__":
    main()
