"""Phase 6.B.4e — quantitative MP2-GIAO validation across small molecules.

Reproduces the σ_p^MP2-GIAO Stanton-Gauss correction on a battery of
canonical small molecules (CO, H₂O, CH₄, benzene) at DZP basis with
Becke + Lebedev quadrature, comparing HF / MP2-LO / MP2-full to the
literature MP2/cc-pVDZ values.

Probe convention: 0.1 Å above the spectroscopically interesting nucleus
(or NICS(1) for benzene). σ_p magnitudes scale as 1/r³ near nuclei,
so the offset matters but the sign and ratio of MP2 corrections do not.

Run::

    python scripts/benchmark_mp2_giao_validation.py

Memory cost ~2 GB at peak ; CPU ~20 min total on a modern laptop.
"""
from __future__ import annotations

import time

import numpy as np

from ptc.lcao.cluster import build_explicit_cluster
from ptc.lcao.fock import density_matrix_PT_scf
from ptc.lcao.mp2 import mp2_paramagnetic_shielding_coupled


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
    "CO": dict(
        Z=[6, 8],
        coords=np.array([[0.0, 0.0, 0.0], [1.128, 0.0, 0.0]]),
        bonds=[(0, 1, 3.0)],
        probe=np.array([0.0, 0.0, 0.1]),       # 0.1 Å above C
        ref_label="σ_iso(C)/cc-pVDZ:  HF=-25 ppm, MP2=+5 ppm  (lit. C-13 NMR)",
    ),
    "H2O": dict(
        Z=[8, 1, 1],
        coords=np.array([
            [0.0, 0.0, 0.117],
            [0.757, 0.0, -0.470],
            [-0.757, 0.0, -0.470],
        ]),
        bonds=[(0, 1, 1.0), (0, 2, 1.0)],
        probe=np.array([0.0, 0.0, 0.217]),     # 0.1 Å above O
        ref_label="σ_iso(O)/cc-pVDZ:  HF=326, MP2=337 ppm  (lit. H₂O Helgaker)",
    ),
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
        probe=np.array([0.0, 0.0, 0.1]),       # 0.1 Å above C
        ref_label="σ_iso(C)/cc-pVDZ:  HF=192, MP2=200 ppm  (lit. CH₄ Helgaker)",
    ),
    "Benzene": dict(
        Z=_benzene_xyz()[0],
        coords=_benzene_xyz()[1],
        bonds=_benzene_xyz()[2],
        probe=np.array([0.0, 0.0, 1.0]),       # NICS(1)
        ref_label="NICS(1)^MP2/cc-pVDZ ≈ -10 ppm   (Wilson 1996, Cremer 2007)",
    ),
}


def benchmark(name: str, basis_type: str = "DZP", use_becke: bool = True,
                 n_radial: int = 18, lebedev_order: int = 26) -> dict:
    """Run the full σ_p^MP2-GIAO pipeline on the named system."""
    sys_def = SYSTEMS[name]
    basis, topology = build_explicit_cluster(
        Z_list=sys_def["Z"], coords=sys_def["coords"],
        bonds=sys_def["bonds"], basis_type=basis_type,
    )

    if use_becke:
        scf_grid = dict(use_becke=True, n_radial=n_radial,
                          lebedev_order=lebedev_order)
        op_grid = dict(n_radial=n_radial, n_theta=10, n_phi=12,
                          use_becke=True, lebedev_order=lebedev_order)
    else:
        scf_grid = dict(n_radial=12, n_theta=8, n_phi=10)
        op_grid = dict(n_radial=12, n_theta=8, n_phi=10,
                          use_becke=False, lebedev_order=14)

    print(f"  --- {name} {basis_type} {'Becke' if use_becke else 'spherical'} ---")
    t_total = time.time()

    rho, S, eigvals, c, conv, _ = density_matrix_PT_scf(
        topology, basis=basis, mode="hf",
        max_iter=20, tol=1e-4, **scf_grid,
    )
    n_occ = int(round(basis.total_occ)) // 2
    print(f"  HF SCF: {conv} iter, {time.time()-t_total:.1f}s, "
            f"n_orb={basis.n_orbitals} n_occ={n_occ}")

    t_pipe = time.time()
    out = mp2_paramagnetic_shielding_coupled(
        basis, topology, eigvals, c, n_occ, sys_def["probe"],
        mp2_kwargs=op_grid, lagrangian_kwargs=op_grid,
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
    t_pipe = time.time() - t_pipe
    t_total = time.time() - t_total

    sigma_HF = out["sigma_p_HF"]
    sigma_full = out["sigma_p_MP2_full"]
    delta_pct = (sigma_full - sigma_HF) / sigma_HF * 100.0 if sigma_HF != 0 else 0.0

    print(f"  pipeline: {t_pipe:.1f}s   total: {t_total:.1f}s")
    print(f"  σ_p^HF       = {sigma_HF:+9.3f} ppm")
    print(f"  σ_p^MP2_LO   = {out['sigma_p_MP2_LO']:+9.3f} ppm")
    print(f"  σ_p^MP2_full = {sigma_full:+9.3f} ppm")
    print(f"  Δ_LO−HF       = {out['delta_LO_minus_HF']:+9.4f} ppm")
    print(f"  Δ_full−LO     = {out['delta_full_minus_LO']:+9.4f} ppm  (Z-vector)")
    print(f"  Δ_total / σ^HF = {delta_pct:+6.2f}%")
    print(f"  |Z|_max       = {float(np.abs(out['z_vector']).max()):.4e}")
    print(f"  Lit ref      : {sys_def['ref_label']}")
    print()
    return {
        "name": name, "basis_type": basis_type,
        "n_orb": basis.n_orbitals, "n_occ": n_occ,
        "t_total": t_total, "t_pipeline": t_pipe,
        "sigma_p_HF": sigma_HF,
        "sigma_p_MP2_LO": out["sigma_p_MP2_LO"],
        "sigma_p_MP2_full": sigma_full,
        "delta_LO": out["delta_LO_minus_HF"],
        "delta_Z": out["delta_full_minus_LO"],
        "delta_pct": delta_pct,
        "Z_max": float(np.abs(out["z_vector"]).max()),
    }


def main():
    print("=" * 78)
    print(" σ_p^MP2-GIAO validation suite — Phase 6.B.4e (DZP Becke)")
    print("=" * 78)
    print()
    results = []
    for name in ("CO", "H2O", "CH4", "Benzene"):
        results.append(benchmark(name, basis_type="DZP", use_becke=True))

    print()
    print("=" * 78)
    print("Summary table")
    print("=" * 78)
    print(f"{'System':<10} {'n_orb':>5} {'σ^HF':>10} {'σ^MP2_LO':>10} "
            f"{'σ^MP2_full':>11} {'Δ%':>7} {'|Z|_max':>10} {'time(s)':>8}")
    for r in results:
        print(f"{r['name']:<10} {r['n_orb']:>5} "
                f"{r['sigma_p_HF']:>+10.3f} {r['sigma_p_MP2_LO']:>+10.3f} "
                f"{r['sigma_p_MP2_full']:>+11.3f} "
                f"{r['delta_pct']:>+6.2f}% "
                f"{r['Z_max']:>10.3e} {r['t_total']:>8.1f}")


if __name__ == "__main__":
    main()
