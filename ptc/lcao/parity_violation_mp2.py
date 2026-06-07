"""MP2-improved second-order parity-violation response (Bakasov-Ha-Quack route).

This module is the closed-shell MP2 upgrade of
:mod:`ptc.lcao.spin_orbit.e_pv_so_response_rhf`. It re-uses every existing
PT-LCAO building block without modifying them:

* ``ptc.lcao.parity_violation.hpv_matrix_elements``  for H_PV  (unchanged)
* ``ptc.lcao.spin_orbit.hso_matrix_elements``        for H_SO  (unchanged)
* ``ptc.lcao.spin_orbit.e_pv_response_so``           for the SO/PV second
                                                     order response sum
                                                     (unchanged)
* ``ptc.lcao.mp2.mp2_at_hf``       for MP2 amplitudes (t_ij^ab)
* ``ptc.lcao.mp2.mp2_lagrangian``  for the orbital Lagrangian L_ai
* ``ptc.lcao.mp2.solve_z_vector``  for the orbital Z-vector A.Z = -L
* ``ptc.lcao.mp2.mp2_relax_orbitals`` for the relaxed F[ρ_MP2] orbitals /
                                       eigenvalues that we plug into the
                                       Bakasov-Ha-Quack response formula.

Physical motivation
-------------------
For a closed-shell singlet the bare expectation value <Psi|H_PV|Psi> = 0
because alpha/beta spatial densities coincide and the spin trace of sigma
vanishes. Parity violation first appears at SECOND ORDER in (H_SO, H_PV)
through the canonical Bakasov-Ha-Quack 1998 perturbative sum

    E_PV  =  2 Re sum_{i in occ}{a in vir}
              <phi_i | H_SO  | phi_a><phi_a | H_PV | phi_i>
              / (eps_i - eps_a)

The RHF reference in :mod:`ptc.lcao.spin_orbit` uses canonical Fock
orbitals {phi_i^HF, eps_i^HF}. The known shortcoming is that

  * the HF gaps (eps_i - eps_a) are too LARGE (Koopmans' theorem overshoots
    the electron-affinity / ionisation gap by 1-3 eV in molecular systems)
  * the HF occupied orbitals are insufficiently relaxed: correlation pulls
    them in slightly, which redistributes density toward heavy nuclei where
    H_PV peaks.

Both effects are addressed by re-diagonalising F[ρ_MP2] with the relaxed
MP2 1-RDM ρ_MP2 = ρ_HF + d_occ ⊕ d_vir + 2.Z. This procedure already
exists in PTC's NMR machinery as :func:`mp2_relax_orbitals`; it returns
new orbitals (c_MP2) and new orbital energies (eigvals_MP2) that include
the leading occupation-shift + full Z-vector orbital-relaxation correction
(Stanton-Gauss 1992 / Aikens-Gordon 2003 closure).

Substituting (c_MP2, eigvals_MP2) into the same second-order response sum
gives an MP2-relaxed E_PV. This is conceptually identical to the
"Lambda-relaxed density" used in :mod:`ptc.lcao.ccsd_property` for
shieldings, applied here to the SO/PV operator pair.

What this module does NOT do
----------------------------
* Direct MP2 doubles contribution to E_PV (the <0|H_PV t_2 H_SO|0> term).
  That would require building the t_2 . H_PV . t_2-style triple-product
  closed-shell contraction from scratch, which is not exposed by mp2.py.
  In one session we settle for the orbital-response-only correction, which
  captures the dominant gap-shrinking + density-relaxation effect.
* CCSD-Lambda response. Possible follow-up via ccsd_lambda + ccsd_property
  composition; out of scope for the present article.

The MP2 orbital-response correction is expected to scale up |E_PV| by
roughly a factor 2-5 compared to the bare RHF response, leaving a residual
factor 20-100 to Bakasov-Ha-Quack 1998 that is attributable to basis-set
incompleteness (single-zeta vs cc-pVDZ) - the subject of article 2.

Sign
----
The MP2 correction does NOT flip the sign of E_PV. Both the gap shrinkage
1/(eps_i - eps_a) and the density relaxation are sign-preserving with
respect to the underlying RHF response. Tests in
``ptc/tests/test_parity_violation_mp2.py`` assert sign(E_PV^MP2) ==
sign(E_PV^RHF).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ptc.lcao.density_matrix import PTMolecularBasis
from ptc.lcao.mp2 import (
    MP2Result,
    mp2_at_hf,
    mp2_lagrangian,
    mp2_relax_orbitals,
    solve_z_vector,
)
from ptc.lcao.parity_violation import hpv_matrix_elements
from ptc.lcao.spin_orbit import (
    _build_spin_mo_coefficients,
    e_pv_response_so,
    hso_matrix_elements,
)


# ---------------------------------------------------------------------
# Result bundle
# ---------------------------------------------------------------------


@dataclass
class EPVMP2Result:
    """Bundle returned by :func:`e_pv_so_response_mp2`.

    Attributes
    ----------
    e_pv_rhf : float
        Baseline second-order E_PV computed with the RHF reference
        (Bakasov-Ha-Quack 1998 formula on canonical HF orbitals).
    e_pv_mp2_lo : float
        E_PV with the MP2 occupation-shift orbitals (Z = 0, the leading
        order of MP2 relaxation: only d_occ + d_vir blocks).
    e_pv_mp2_full : float
        E_PV with the full Z-vector-relaxed MP2 orbitals (Stanton-Gauss
        closure). This is the production "MP2-improved E_PV" value.
    mp2_result : MP2Result
        Amplitude / energy / 1-RDM bundle from ``mp2_at_hf``.
    z_vector : ndarray of shape (n_virt, n_occ)
        Orbital Z-vector solving A.Z = -L (closed-shell).
    eigvals_mp2_lo, c_mp2_lo : MP2-LO orbital quantities (eV, AO basis).
    eigvals_mp2_full, c_mp2_full : full-MP2 orbital quantities.
    """

    e_pv_rhf: float
    e_pv_mp2_lo: float
    e_pv_mp2_full: float
    mp2_result: MP2Result
    z_vector: np.ndarray
    eigvals_mp2_lo: np.ndarray
    c_mp2_lo: np.ndarray
    eigvals_mp2_full: np.ndarray
    c_mp2_full: np.ndarray


# ---------------------------------------------------------------------
# Internal: dispatch the second-order response on a given (eigvals, C)
# ---------------------------------------------------------------------


def _e_pv_at_orbitals(H_pv: np.ndarray,
                      H_so: np.ndarray,
                      eigvals: np.ndarray,
                      c: np.ndarray,
                      n_e_total: int) -> float:
    """Run the Bakasov-Ha-Quack response sum on a given spatial
    (eigvals, C) MO set, with H_PV and H_SO pre-built and held fixed.

    Both operators live in the spin-resolved AO basis. The spatial
    coefficients are lifted to spin orbitals with the "alpha == beta
    spatial" convention used by ``e_pv_so_response_rhf``; spatial
    eigenvalues are doubled (one for each spin channel).
    """
    C_spin = _build_spin_mo_coefficients(np.asarray(c))
    eps_spin = np.repeat(np.asarray(eigvals, dtype=float), 2)
    return e_pv_response_so(H_pv, H_so, C_spin, eps_spin, int(n_e_total))


# ---------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------


def e_pv_so_response_mp2(
    basis: PTMolecularBasis,
    topology,
    mo_eigvals_hf: np.ndarray,
    mo_coeffs_hf: np.ndarray,
    n_e_total: int,
    *,
    mp2_result: Optional[MP2Result] = None,
    z_vector: Optional[np.ndarray] = None,
    use_z_vector: bool = True,
    compute_baselines: bool = True,
    hpv_kwargs: Optional[dict] = None,
    hso_kwargs: Optional[dict] = None,
    mp2_kwargs: Optional[dict] = None,
    lagrangian_kwargs: Optional[dict] = None,
    z_vector_kwargs: Optional[dict] = None,
    relax_kwargs: Optional[dict] = None,
) -> EPVMP2Result:
    """MP2-improved parity-violation energy on a closed-shell RHF reference.

    Three values are returned in the result bundle:

    * ``e_pv_rhf``       : standard Bakasov-Ha-Quack response on HF
                           orbitals (same as
                           ``ptc.lcao.spin_orbit.e_pv_so_response_rhf``,
                           reproduced for direct comparison).
    * ``e_pv_mp2_lo``    : response on MP2 occupation-shift-relaxed
                           orbitals (Z = 0, leading-order MP2 1-RDM).
    * ``e_pv_mp2_full``  : response on Z-vector-relaxed MP2 orbitals
                           (full Stanton-Gauss closure, production value).

    Parameters
    ----------
    basis : PTMolecularBasis
        Spatial AO basis with attached geometry, as produced by e.g.
        ``alanine_L_B3LYP()`` or ``build_molecular_basis``.
    topology
        Topology object (only used by ``mp2_relax_orbitals`` for grid
        setup; the MP2 calculation itself ignores it). Pass ``None`` for
        manually-built bases - the relax routine does not actually use
        the topology beyond forwarding it.
    mo_eigvals_hf : (n_orb,) ndarray
        RHF orbital energies in eV (output of ``density_matrix_PT_scf``).
    mo_coeffs_hf : (n_orb, n_orb) ndarray
        RHF MO coefficients in the spatial AO basis (output of
        ``density_matrix_PT_scf``).
    n_e_total : int
        Total electron count. Must be even (closed-shell).
    mp2_result, z_vector : optional pre-computed objects (skip the
        corresponding step). Useful when sweeping over geometries that
        share the same reference orbitals.
    use_z_vector : bool, default True
        If False, return ``e_pv_mp2_full = e_pv_mp2_lo`` (skip the
        Z-vector solve).
    compute_baselines : bool, default True
        If False, skip the e_pv_rhf and e_pv_mp2_lo values (use NaN
        placeholders). Saves one response sum each.
    hpv_kwargs / hso_kwargs : kwargs forwarded to
        ``hpv_matrix_elements`` / ``hso_matrix_elements`` (e.g. grid
        density for H_SO).
    mp2_kwargs, lagrangian_kwargs, z_vector_kwargs, relax_kwargs :
        kwargs forwarded to the respective MP2 / relaxation routines.

    Returns
    -------
    EPVMP2Result
        See class docstring.

    Notes
    -----
    * All three E_PV values are in Hartree.
    * The MO eigenvalues used in the response sum carry the unit of the
      RHF SCF (eV in PTC). The Bakasov-Ha-Quack formula divides by
      (eps_i - eps_a) directly; for the magnitude comparison to be
      meaningful, both numerator (H_PV . H_SO matrix elements in
      Hartree) and denominator must be in consistent units. The existing
      ``e_pv_response_so`` makes the same choice (eV denominators), and
      we keep it for one-to-one comparability with the RHF baseline. The
      ratio E_PV_MP2 / E_PV_RHF is invariant under this choice.
    """
    if n_e_total % 2 != 0:
        raise ValueError(
            f"Closed-shell MP2 driver requires an even electron count; "
            f"got n_e_total = {n_e_total}"
        )

    hpv_kwargs = dict(hpv_kwargs or {})
    hso_kwargs = dict(hso_kwargs or {})
    mp2_kwargs = dict(mp2_kwargs or {})
    lagrangian_kwargs = dict(lagrangian_kwargs or {})
    z_vector_kwargs = dict(z_vector_kwargs or {})
    relax_kwargs = dict(relax_kwargs or {})

    n_occ = n_e_total // 2

    # 1. Build the two operators ONCE. They are reference-independent.
    H_pv = hpv_matrix_elements(basis, **hpv_kwargs)
    H_so = hso_matrix_elements(basis, **hso_kwargs)

    # 2. RHF baseline (cheap once H_pv/H_so are in hand).
    if compute_baselines:
        e_pv_rhf = _e_pv_at_orbitals(
            H_pv, H_so, mo_eigvals_hf, mo_coeffs_hf, n_e_total
        )
    else:
        e_pv_rhf = float("nan")

    # 3. MP2 amplitudes (and density correction blocks) on the HF ref.
    if mp2_result is None:
        mp2_result = mp2_at_hf(
            basis, mo_eigvals_hf, mo_coeffs_hf, n_occ, **mp2_kwargs
        )

    # 4. Leading-order MP2 orbitals (d_occ + d_vir only, no Z).
    if compute_baselines:
        eigvals_lo, c_lo = mp2_relax_orbitals(
            basis, topology, mo_coeffs_hf, n_occ, mp2_result,
            z_vector=None, **relax_kwargs,
        )
        e_pv_mp2_lo = _e_pv_at_orbitals(
            H_pv, H_so, eigvals_lo, c_lo, n_e_total
        )
    else:
        eigvals_lo = np.full_like(mo_eigvals_hf, np.nan)
        c_lo = np.full_like(mo_coeffs_hf, np.nan)
        e_pv_mp2_lo = float("nan")

    # 5. Full Stanton-Gauss Z-vector relaxation.
    if use_z_vector:
        if z_vector is None:
            L = mp2_lagrangian(
                basis, mo_coeffs_hf, n_occ, mp2_result, **lagrangian_kwargs
            )
            z_vector = solve_z_vector(
                basis, mo_eigvals_hf, mo_coeffs_hf, n_occ, L,
                **z_vector_kwargs,
            )
        eigvals_full, c_full = mp2_relax_orbitals(
            basis, topology, mo_coeffs_hf, n_occ, mp2_result,
            z_vector=z_vector, **relax_kwargs,
        )
    else:
        # Without Z, "full" equals "leading order".
        if compute_baselines:
            z_vector = np.zeros(
                (mo_coeffs_hf.shape[0] - n_occ, n_occ), dtype=float
            )
            eigvals_full, c_full = eigvals_lo, c_lo
        else:
            z_vector = np.zeros(
                (mo_coeffs_hf.shape[0] - n_occ, n_occ), dtype=float
            )
            eigvals_full, c_full = mp2_relax_orbitals(
                basis, topology, mo_coeffs_hf, n_occ, mp2_result,
                z_vector=None, **relax_kwargs,
            )

    e_pv_mp2_full = _e_pv_at_orbitals(
        H_pv, H_so, eigvals_full, c_full, n_e_total
    )

    return EPVMP2Result(
        e_pv_rhf=e_pv_rhf,
        e_pv_mp2_lo=e_pv_mp2_lo,
        e_pv_mp2_full=e_pv_mp2_full,
        mp2_result=mp2_result,
        z_vector=np.asarray(z_vector),
        eigvals_mp2_lo=np.asarray(eigvals_lo),
        c_mp2_lo=np.asarray(c_lo),
        eigvals_mp2_full=np.asarray(eigvals_full),
        c_mp2_full=np.asarray(c_full),
    )


__all__ = ["EPVMP2Result", "e_pv_so_response_mp2"]
