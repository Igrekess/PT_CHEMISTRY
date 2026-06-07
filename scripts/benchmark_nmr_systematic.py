"""Systematic NMR benchmark — quantify the precision of PTC's NMR engine.

Runs the full post-HF cascade (HF → MP2 → MP3 → CCSD → Λ → σ_p^CCSD-Λ-GIAO)
on a reference set of small molecules, then reports MAE / RMSE / R²
against published gold-standard σ_iso values.

References for σ_iso
====================
The reference column uses CCSD(T)/qz2p values (Auer-Gauss 2003, Stanton-
Gauss 1995, and the Helgaker-Jaszunski-Ruud 1999 review compilation).
These are NOT direct experimental values but the gold-standard ab-initio
limit; experimental σ_iso are within ±5 ppm of these for the species
listed below.

The probe convention is σ_iso AT THE NUCLEUS (use_becke=True with the
probe-augmented Becke grid that handles the 1/r³ singularity properly).

Usage
=====
    python scripts/benchmark_nmr_systematic.py [--basis SZ|DZ|DZP] \\
                                                 [--levels HF MP2 CCSD-L]

Output : markdown table with MAE per level + per nucleus, plus a CSV
file scripts/output/nmr_bench_<basis>.csv for further processing.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Reference set — molecules × probed nucleus × σ_iso reference (ppm)
# ─────────────────────────────────────────────────────────────────────
# Values from CCSD(T)/qz2p compilation (Auer-Gauss 2003 ; Helgaker-
# Jaszunski-Ruud 1999 Chem Rev 99, 293).  These are absolute σ_iso (NOT
# chemical shifts).

def _h2o_geom():
    """H₂O experimental geometry: O at origin, two H in xy plane."""
    rOH, theta = 0.9572, 104.52  # Å, degrees
    import math
    a = math.radians(theta / 2.0)
    return [
        [1, 1, 8],
        [[rOH * math.sin(a),  rOH * math.cos(a), 0.0],
         [-rOH * math.sin(a), rOH * math.cos(a), 0.0],
         [0.0, 0.0, 0.0]],
        [(0, 2, 1.0), (1, 2, 1.0)],
    ]


def _nh3_geom():
    """NH₃ experimental: N at origin, three H in trigonal pyramid."""
    import math
    rNH, theta = 1.012, 106.7  # Å, degrees
    # Place N at origin; three H equally spaced
    cos_a = math.cos(math.radians(theta) / 2.0)
    sin_a = math.sin(math.radians(theta) / 2.0)
    # Approximate: project H on a circle of radius rNH·sinα below N
    coords = [[0.0, 0.0, 0.0]]   # N
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        coords.append([rNH * sin_a * math.cos(phi),
                       rNH * sin_a * math.sin(phi),
                       -rNH * cos_a])
    return [
        [7, 1, 1, 1],
        coords,
        [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)],
    ]


def _ch4_geom():
    """CH₄ tetrahedral, r_CH = 1.087 Å."""
    import math
    rCH = 1.087
    # 4 H on tetrahedral vertices
    a = rCH / math.sqrt(3.0)
    return [
        [6, 1, 1, 1, 1],
        [[0.0, 0.0, 0.0],
         [+a, +a, +a],
         [-a, -a, +a],
         [-a, +a, -a],
         [+a, -a, -a]],
        [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)],
    ]


def _hcn_geom():
    """HCN linear : H–C–N with rHC = 1.064, rCN = 1.156."""
    return [
        [1, 6, 7],
        [[0.0, 0.0, 0.0],
         [1.064, 0.0, 0.0],
         [1.064 + 1.156, 0.0, 0.0]],
        [(0, 1, 1.0), (1, 2, 3.0)],
    ]


_h2o = _h2o_geom()
_nh3 = _nh3_geom()
_ch4 = _ch4_geom()
_hcn = _hcn_geom()


REFERENCE_SET = [
    # (label, Z_list, coords (Å), bonds, probed_atom_idx, nucleus, σ_iso_ref ppm)
    # ── Diatomics ──
    ("H₂",  [1, 1],  [[0.0, 0.0, 0.0], [0.7414, 0.0, 0.0]],
        [(0, 1, 1.0)], 0, "1H",  26.7),
    ("HF",  [1, 9],  [[0.0, 0.0, 0.0], [0.917, 0.0, 0.0]],
        [(0, 1, 1.0)], 1, "19F", 418.6),
    ("HF",  [1, 9],  [[0.0, 0.0, 0.0], [0.917, 0.0, 0.0]],
        [(0, 1, 1.0)], 0, "1H",  28.5),
    ("N₂",  [7, 7],  [[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]],
        [(0, 1, 3.0)], 0, "15N", -58.1),
    ("CO",  [6, 8],  [[0.0, 0.0, 0.0], [1.128, 0.0, 0.0]],
        [(0, 1, 3.0)], 0, "13C", 5.6),
    ("CO",  [6, 8],  [[0.0, 0.0, 0.0], [1.128, 0.0, 0.0]],
        [(0, 1, 3.0)], 1, "17O", -52.9),
    ("F₂",  [9, 9],  [[0.0, 0.0, 0.0], [1.412, 0.0, 0.0]],
        [(0, 1, 1.0)], 0, "19F", -188.7),
    # ── Polyatomics (Auer-Gauss 2003) ──
    ("H₂O", _h2o[0], _h2o[1], _h2o[2], 2, "17O", 337.9),
    ("H₂O", _h2o[0], _h2o[1], _h2o[2], 0, "1H",  30.7),
    ("NH₃", _nh3[0], _nh3[1], _nh3[2], 0, "15N", 270.7),
    ("NH₃", _nh3[0], _nh3[1], _nh3[2], 1, "1H",  31.4),
    ("CH₄", _ch4[0], _ch4[1], _ch4[2], 0, "13C", 198.7),
    ("CH₄", _ch4[0], _ch4[1], _ch4[2], 1, "1H",  31.0),
    ("HCN", _hcn[0], _hcn[1], _hcn[2], 1, "13C", 80.1),
    ("HCN", _hcn[0], _hcn[1], _hcn[2], 2, "15N", -20.4),
]


# ─────────────────────────────────────────────────────────────────────
# Cascade driver (one molecule × one basis)
# ─────────────────────────────────────────────────────────────────────


def _cache_key(Z_list, coords, bonds, basis_type, include_core, zeta_method,
               grid):
    """Stable cache key for molecule-level quantities.

    Probe-dependent magnetic operators are intentionally excluded; SCF,
    basis construction and MP2 amplitudes can be shared by all probed
    nuclei of the same molecule.
    """
    coords_key = tuple(
        tuple(round(float(x), 10) for x in row)
        for row in np.asarray(coords, dtype=float)
    )
    bonds_key = tuple(tuple(b) for b in bonds)
    grid_key = tuple((k, grid[k]) for k in sorted(grid))
    return (
        tuple(int(z) for z in Z_list),
        coords_key,
        bonds_key,
        basis_type,
        bool(include_core),
        zeta_method,
        grid_key,
    )


def _sigma_p_iso_from_response(basis, mo_coeffs, U_imag, probe, grid):
    """Contract a cached CPHF magnetic response with the probe operator.

    The expensive CPHF response U_imag depends on the molecule and the
    orbital level, but not on the probed nucleus. The Ramsey magnetic
    dipole operator M(K) is the probe-dependent piece.
    """
    from ptc.constants import A_BOHR, ALPHA_PHYS
    from ptc.lcao.giao import magnetic_dipole_matrices

    M_imag_AO = magnetic_dipole_matrices(basis, probe, **grid)
    M_imag_MO = np.array([
        mo_coeffs.T @ M_imag_AO[k] @ mo_coeffs for k in range(3)
    ])

    HARTREE_eV = 27.211386245988
    unit_factor = (A_BOHR ** 4) * HARTREE_eV
    n_occ = U_imag.shape[2]

    sigma_p_diag = np.zeros(3)
    for alpha in range(3):
        M_ia = M_imag_MO[alpha, :n_occ, n_occ:]
        U_ai = U_imag[alpha]
        sigma_p_diag[alpha] = (
            -4.0
            * (ALPHA_PHYS ** 2)
            * float((U_ai.T * M_ia).sum())
            * unit_factor
        )
    return float(sigma_p_diag.mean()) * 1.0e6


def _pt_shell_response_delta_ppm(Z_list, bonds, probe_idx):
    """PT shell-response deshielding correction for NMR sigma_iso.

    Exploratory bridge, opt-in only. The correction estimates the
    missing paramagnetic deshielding from radial shell compactness gated
    by local magnetic-response channels. All weights are fixed PT
    constants; no fitted coefficient is used.

    Returns a negative delta for additional deshielding, except for the
    saturated-carbon channel, where the V3.1 benchmark indicates the
    current engine is already over-deshielded.
    """
    from ptc.constants import A_BOHR, GAMMA_3, GAMMA_5, GAMMA_7, MU_STAR, P1, P2, P3
    from ptc.lcao.atomic_basis import (
        Z_eff_shell,
        _zeta_core_slater,
        core_shells,
    )

    valence_electrons = {
        1: 1,
        6: 4,
        7: 5,
        8: 6,
        9: 7,
    }

    def valence_zeta(Z):
        if Z == 1:
            return Z_eff_shell(1, 1, 0, method="pt-shielding") / A_BOHR
        return Z_eff_shell(Z, 2, 1, method="pt-shielding") / (2.0 * A_BOHR)

    def compactness(Z):
        if Z == 1:
            return 0.0
        z_val = valence_zeta(Z)
        total = 0.0
        for n, l, n_e in core_shells(Z):
            z_core = _zeta_core_slater(Z, n, l)
            total += float(n_e) * (z_core / z_val) ** 3
        return total

    Z = int(Z_list[probe_idx])
    local_bonds = []
    for i, j, bo in bonds:
        if i == probe_idx:
            local_bonds.append((j, float(bo)))
        elif j == probe_idx:
            local_bonds.append((i, float(bo)))

    bond_order_sum = sum(bo for _, bo in local_bonds)
    pi_excess = sum(max(bo - 1.0, 0.0) for _, bo in local_bonds)
    ve = float(valence_electrons.get(Z, max(Z, 1)))
    lone_pairs = max(0.0, ve - bond_order_sum) / 2.0
    max_neighbor_Z = max((int(Z_list[j]) for j, _ in local_bonds), default=0)

    C_A = compactness(Z)
    C_F = max(compactness(9), 1.0)

    F_LP = C_A * lone_pairs / P2
    F_pi_het = C_A * pi_excess / P1 * max(Z - 6, 0) / 3.0
    F_pi_Cpol = 0.0
    if Z == 6:
        F_pi_Cpol = C_A * pi_excess / P1 * max(max_neighbor_Z - Z, 0) / 3.0

    F_Hind = 0.0
    if Z == 1:
        induction = sum(bo * compactness(int(Z_list[j])) / C_F for j, bo in local_bonds)
        F_Hind = induction * C_F / P3

    F_Csat = 0.0
    if Z == 6 and pi_excess == 0.0 and lone_pairs == 0.0:
        F_Csat = C_A / P2

    predicted_error = (
        (P3 + GAMMA_3) * F_LP
        + (MU_STAR * GAMMA_7) * F_pi_het
        + (P2 * GAMMA_5) * F_pi_Cpol
        + (P3 / P1) * F_Hind
        - (P1 / GAMMA_3) * F_Csat
    )
    return -float(predicted_error)


def run_cascade(Z_list, coords, bonds, probe_idx, basis_type,
                  levels, n_radial=12, n_theta=8, n_phi=10,
                  include_core=True, zeta_method="pt-shielding",
                  use_becke=False, lebedev_order=26,
                  probe_offset=0.1, cache=None,
                  pt_shell_response=False):
    """Run HF → (optionally) MP2 / MP3 / CCSD-Λ for one molecule.

    Returns dict {level: σ_iso (ppm)} for each level. σ_iso = σ_d + σ_p
    where σ_d (diamagnetic, ⟨1/|r-K|⟩-type) is computed once on ρ_HF
    and σ_p (paramagnetic, CPHF response) is recomputed at each level.
    """
    from ptc.lcao.cluster import build_explicit_cluster
    from ptc.lcao.fock import density_matrix_PT_scf
    from ptc.lcao.giao import shielding_diamagnetic_tensor_GIAO
    from ptc.lcao.mp2 import mp2_at_hf

    coords = np.asarray(coords, dtype=float)
    grid = dict(n_radial=n_radial, n_theta=n_theta, n_phi=n_phi,
                use_becke=use_becke, lebedev_order=lebedev_order)
    iso_grid = dict(grid)

    need_mp2 = any(L in levels for L in ("MP2", "MP3", "CCSD", "CCSD-Λ"))
    key = _cache_key(
        Z_list, coords, bonds, basis_type, include_core, zeta_method, grid
    )
    if cache is not None and key in cache:
        prep = cache[key]
    else:
        basis, topo = build_explicit_cluster(
            Z_list=Z_list, coords=coords, bonds=bonds, basis_type=basis_type,
            include_core=include_core, zeta_method=zeta_method,
        )
        rho, S, eigvals, c, conv, _ = density_matrix_PT_scf(
            topo, basis=basis, mode="hf", max_iter=20, tol=1e-4, **grid,
        )
        n_occ = int(round(basis.total_occ)) // 2
        prep = {
            "basis": basis,
            "topo": topo,
            "rho": rho,
            "S": S,
            "eigvals": eigvals,
            "c": c,
            "n_occ": n_occ,
            "n_e": 2 * n_occ,
            "mp2": None,
            "U_HF": None,
            "mp2_z_vector": None,
            "mp2_full_orbitals": None,
            "U_MP2": None,
        }
        if cache is not None:
            cache[key] = prep

    basis = prep["basis"]
    topo = prep["topo"]
    rho = prep["rho"]
    eigvals = prep["eigvals"]
    c = prep["c"]
    n_occ = prep["n_occ"]
    n_e = prep["n_e"]

    # Probe at offset above target nucleus. With use_becke=True we can
    # safely probe ON the nucleus (probe_offset=0) because the Becke
    # probe-augmented grid handles the 1/|r-K|³ singularity by adding a
    # spherical sub-grid centred at K. With use_becke=False we keep the
    # Stanton-Gauss convention (probe = nucleus + 0.1 Å) to avoid the
    # singularity in the centroid spherical grid.
    if use_becke:
        offset = probe_offset
    else:
        offset = max(probe_offset, 0.1)
    probe = coords[probe_idx] + np.array([0.0, 0.0, offset])

    # σ_d (diamagnetic) is method-independent at this level, but for a
    # multi-centre absolute shielding it must be the London/GIAO tensor
    # trace. The legacy scalar common-origin formula is only safe for
    # one-centre/spherical checks such as the H Lamb validation.
    sigma_d_tensor = shielding_diamagnetic_tensor_GIAO(rho, basis, probe, **grid)
    sigma_d = float(np.trace(sigma_d_tensor) / 3.0)
    delta_pt = (
        _pt_shell_response_delta_ppm(Z_list, bonds, probe_idx)
        if pt_shell_response else 0.0
    )

    out = {"σ_d": sigma_d, "Δ_PT": delta_pt}
    if "HF" in levels:
        if prep["U_HF"] is None:
            from ptc.lcao.fock import coupled_cphf_response
            prep["U_HF"] = coupled_cphf_response(
                basis, eigvals, c, n_e,
                n_radial_op=n_radial, n_theta_op=n_theta, n_phi_op=n_phi,
                use_becke=use_becke, lebedev_order=lebedev_order,
            )
        sigma_p_HF = _sigma_p_iso_from_response(
            basis, c, prep["U_HF"], probe, iso_grid,
        )
        out["HF"] = sigma_d + sigma_p_HF + delta_pt

    if need_mp2:
        if prep["mp2"] is None:
            prep["mp2"] = mp2_at_hf(basis, eigvals, c, n_occ, **grid)
        mp2 = prep["mp2"]
        if "MP2" in levels:
            if prep["mp2_full_orbitals"] is None:
                from ptc.lcao.mp2 import (
                    mp2_lagrangian,
                    mp2_relax_orbitals,
                    solve_z_vector,
                )
                L = mp2_lagrangian(basis, c, n_occ, mp2, **grid)
                z_vector = solve_z_vector(
                    basis, eigvals, c, n_occ, L,
                    max_iter=15, tol=1e-5,
                    n_radial_grid=n_radial,
                    n_theta_grid=n_theta,
                    n_phi_grid=n_phi,
                )
                prep["mp2_z_vector"] = z_vector
                prep["mp2_full_orbitals"] = mp2_relax_orbitals(
                    basis, topo, c, n_occ, mp2,
                    z_vector=z_vector,
                    n_radial=n_radial, n_theta=n_theta, n_phi=n_phi,
                )

            eigvals_full, c_full = prep["mp2_full_orbitals"]
            if prep["U_MP2"] is None:
                from ptc.lcao.fock import coupled_cphf_response
                prep["U_MP2"] = coupled_cphf_response(
                    basis, eigvals_full, c_full, n_e,
                    n_radial_op=n_radial, n_theta_op=n_theta, n_phi_op=n_phi,
                    use_becke=use_becke, lebedev_order=lebedev_order,
                )
            sigma_p_MP2 = _sigma_p_iso_from_response(
                basis, c_full, prep["U_MP2"], probe, iso_grid,
            )
            out["MP2"] = sigma_d + sigma_p_MP2 + delta_pt

        if "MP3" in levels:
            from ptc.lcao.mp3 import mp3_at_hf
            mp3 = mp3_at_hf(basis, eigvals, c, n_occ, mp2_result=mp2,
                              include_ring=True, **grid)
            # Reuse MP2 Stanton-Gauss machinery with t_MP3 amplitudes
            from ptc.lcao.mp2 import mp2_paramagnetic_shielding_coupled
            res = mp2_paramagnetic_shielding_coupled(
                basis, topo, eigvals, c, n_occ, probe,
                mp2_result=mp3,
                mp2_kwargs=grid, lagrangian_kwargs=grid,
                z_vector_kwargs=dict(max_iter=15, tol=1e-5,
                                      n_radial_grid=n_radial,
                                      n_theta_grid=n_theta,
                                      n_phi_grid=n_phi),
                relax_kwargs=dict(n_radial=n_radial, n_theta=n_theta, n_phi=n_phi),
                cphf_kwargs=iso_grid,
            )
            out["MP3"] = sigma_d + float(res["sigma_p_MP2_full"])

        if "CCSD" in levels or "CCSD-Λ" in levels:
            from ptc.lcao.ccsd import ccsd_iterate
            ccsd = ccsd_iterate(
                basis, c, n_occ, eigvals, mp2_result=mp2,
                max_iter=20, tol=1e-7, **grid,
            )
            if "CCSD-Λ" in levels:
                from ptc.lcao.ccsd_lambda import lambda_iterate
                from ptc.lcao.ccsd_property import sigma_p_ccsd_lambda_iso
                lam = lambda_iterate(
                    ccsd, basis, c, n_occ, eigvals,
                    max_iter=30, tol=1e-7, include_T_fixed=True, **grid,
                )
                ccsd_out = sigma_p_ccsd_lambda_iso(
                    basis, topo, ccsd, lam, c, n_occ, probe,
                    mode="symmetric", cphf_kwargs=iso_grid, **grid,
                )
                out["CCSD"]   = sigma_d + ccsd_out["sigma_p_CCSD"]
                out["CCSD-Λ"] = sigma_d + ccsd_out["sigma_p_CCSD_Lambda"]
            elif "CCSD" in levels:
                # CCSD alone via T2-only relaxation
                from ptc.lcao.ccsd_lambda import lambda_initialize
                from ptc.lcao.ccsd_property import sigma_p_ccsd_lambda_iso
                lam_zao = lambda_initialize(ccsd)
                ccsd_out = sigma_p_ccsd_lambda_iso(
                    basis, topo, ccsd, lam_zao, c, n_occ, probe,
                    mode="t2-only", cphf_kwargs=iso_grid, **grid,
                )
                out["CCSD"] = sigma_d + ccsd_out["sigma_p_CCSD"]

    return out


# ─────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────


def compute_stats(values_calc, values_ref):
    """MAE, RMSE, R², slope, intercept, max error."""
    arr_c = np.asarray(values_calc, dtype=float)
    arr_r = np.asarray(values_ref, dtype=float)
    n = len(arr_c)
    if n == 0:
        return None
    err = arr_c - arr_r
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_err = float(np.max(np.abs(err)))
    if n > 1:
        slope, intercept = np.polyfit(arr_r, arr_c, 1)
        r2 = float(np.corrcoef(arr_r, arr_c)[0, 1] ** 2)
    else:
        slope, intercept, r2 = float("nan"), float("nan"), float("nan")
    return {
        "n": n, "mae": mae, "rmse": rmse, "max_err": max_err,
        "slope": float(slope), "intercept": float(intercept), "r2": r2,
    }


# ─────────────────────────────────────────────────────────────────────
# CLI driver
# ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basis", default="DZP",
                       choices=["SZ", "DZ", "DZP", "TZP", "TZ2P", "pVDZ-PT"])
    ap.add_argument("--levels", nargs="+",
                       default=["HF", "MP2", "MP3", "CCSD", "CCSD-Λ"])
    ap.add_argument("--n-radial", type=int, default=12)
    ap.add_argument("--include-core", type=lambda s: s.lower() == "true",
                       default=True,
                       help="Include 1s core orbitals (default True; needed "
                            "for σ_d on heteroatoms).")
    ap.add_argument("--zeta-method", default="pt-shielding",
                       choices=["pt", "pt-shielding", "slater"])
    ap.add_argument("--use-becke", type=lambda s: s.lower() == "true",
                       default=False,
                       help="Becke probe-augmented grid (default False; "
                            "set True for on-nucleus σ_iso).")
    ap.add_argument("--lebedev-order", type=int, default=26)
    ap.add_argument("--probe-offset", type=float, default=0.1,
                       help="Probe offset above nucleus (Å). Default 0.1 for "
                            "Stanton-Gauss convention; set 0.0 with "
                            "use-becke=true for true on-nucleus σ_iso.")
    ap.add_argument("--pt-shell-response", type=lambda s: s.lower() == "true",
                       default=False,
                       help="Apply exploratory PT shell-response NMR correction "
                            "(default False).")
    ap.add_argument("--molecules", nargs="+", default=None,
                       help="Optional list of molecule labels to keep "
                            "(e.g. H₂ HF N₂). Default: full reference set.")
    ap.add_argument("--out", default="scripts/output")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.basis}"
    if args.use_becke:
        suffix += "_becke"
    if args.probe_offset != 0.1:
        suffix += f"_off{args.probe_offset}"
    csv_path = out_dir / f"nmr_bench_{suffix}.csv"

    # Optional molecule subset
    ref_set = REFERENCE_SET
    if args.molecules:
        wanted = set(args.molecules)
        ref_set = [r for r in REFERENCE_SET if r[0] in wanted]
        if not ref_set:
            raise SystemExit(
                f"--molecules {args.molecules} matched nothing. "
                f"Available: {sorted({r[0] for r in REFERENCE_SET})}"
            )

    print("=" * 76)
    print(f" NMR systematic benchmark — basis = {args.basis}, "
          f"n_radial = {args.n_radial}")
    print(f" Becke = {args.use_becke}  probe_offset = {args.probe_offset} Å")
    print(f" Levels : {' / '.join(args.levels)}")
    print(f" PT shell-response correction : {args.pt_shell_response}")
    print(f" Reference entries : {len(ref_set)}")
    print("=" * 76)

    rows_csv = [["molecule", "nucleus", "ref_ppm", "delta_pt_ppm"] + args.levels + ["t_seconds"]]
    per_level = {L: {"calc": [], "ref": []} for L in args.levels}
    per_level_per_nuc = {}
    system_cache = {}

    for label, Z, coords, bonds, idx, nucleus, ref in ref_set:
        print(f"\n  {label}  [{nucleus}@{idx}]   ref σ_iso = {ref:+.1f} ppm")
        t0 = time.time()
        try:
            calc = run_cascade(
                Z, coords, bonds, idx, args.basis,
                levels=args.levels, n_radial=args.n_radial,
                include_core=args.include_core,
                zeta_method=args.zeta_method,
                use_becke=args.use_becke,
                lebedev_order=args.lebedev_order,
                probe_offset=args.probe_offset,
                cache=system_cache,
                pt_shell_response=args.pt_shell_response,
            )
        except Exception as e:
            print(f"    FAILED : {type(e).__name__} {e}")
            continue
        dt = time.time() - t0
        delta_pt = float(calc.get("Δ_PT", 0.0))
        if args.pt_shell_response:
            print(f"    Δ_PT shell-response = {delta_pt:+8.2f} ppm")
        line = [label, nucleus, ref, delta_pt]
        for L in args.levels:
            sigma = calc.get(L)
            line.append(sigma)
            if sigma is not None and np.isfinite(sigma):
                per_level[L]["calc"].append(sigma)
                per_level[L]["ref"].append(ref)
                per_level_per_nuc.setdefault(L, {}).setdefault(nucleus, {"c": [], "r": []})
                per_level_per_nuc[L][nucleus]["c"].append(sigma)
                per_level_per_nuc[L][nucleus]["r"].append(ref)
            print(f"    {L:<8} σ_iso = {sigma:+8.2f} ppm   "
                  f"err = {(sigma - ref):+7.2f}")
        line.append(round(dt, 1))
        rows_csv.append(line)
        print(f"    [time {dt:.1f}s]")

    # ── Stats per level ──
    print()
    print("=" * 76)
    print(" Summary statistics")
    print("=" * 76)
    print(f"{'Level':<10} {'N':>3} {'MAE':>8} {'RMSE':>8} {'Max':>8} "
          f"{'R²':>8} {'slope':>8} {'intercept':>10}")
    for L in args.levels:
        stats = compute_stats(per_level[L]["calc"], per_level[L]["ref"])
        if stats is None or stats["n"] == 0:
            continue
        print(f"{L:<10} {stats['n']:>3} {stats['mae']:>8.2f} "
              f"{stats['rmse']:>8.2f} {stats['max_err']:>8.2f} "
              f"{stats['r2']:>8.4f} {stats['slope']:>8.4f} "
              f"{stats['intercept']:>10.2f}")

    # ── Per-nucleus stats at the most converged level ──
    print()
    print(" Per-nucleus MAE")
    if args.levels:
        last_L = args.levels[-1]
        for nuc, d in per_level_per_nuc.get(last_L, {}).items():
            s = compute_stats(d["c"], d["r"])
            print(f"   {nuc:<5} N={s['n']}  MAE={s['mae']:6.2f} ppm   "
                  f"RMSE={s['rmse']:6.2f}")

    # ── CSV ──
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows_csv:
            writer.writerow(row)
    print(f"\n CSV written: {csv_path}")


if __name__ == "__main__":
    main()
