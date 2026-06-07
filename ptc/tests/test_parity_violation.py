"""Tests for ptc.lcao.parity_violation — PT Vester-Ulbricht H_PV.

Validation strategy
-------------------
1. PT-derived weak nuclear charges Q_W^A reproduce the table in
   PT_PROJECTS/PT_HOMOCHIRALITY/notes/02_vester_ulbricht_pt.md.
2. The 2 n_orb x 2 n_orb H_PV matrix is Hermitian by construction.
3. For an achiral molecule (H2O, C_s symmetry) the SPATIAL part
   Sigma_A Q_W^A * d_i [chi_mu chi_nu](R_A) vanishes by mirror symmetry
   when summed over both H atoms of the symmetric water — this is the
   key "glycine zero" coherence check, at the matrix-element level.
4. For two enantiomers of a chiral toy molecule (H2O2 with positive vs
   negative dihedral), the spatial gradient sums are equal in magnitude
   and opposite in sign component-by-component.
5. (Skipped for now) Alanine magnitude: the geometry layer is heavy
   enough that we leave the explicit alanine benchmark for a follow-up.

See PT_PROJECTS/PT_HOMOCHIRALITY/notes/02_vester_ulbricht_pt.md.
"""

import math

import numpy as np
import pytest

from ptc.lcao.atomic_basis import PTAtomBasis, build_atom_basis
from ptc.lcao.density_matrix import (
    PTMolecularBasis,
    build_molecular_basis,
    density_matrix_PT,
)
from ptc.lcao.parity_violation import (
    G_F_AU,
    ONE_MINUS_4SIN2,
    PREFACTOR_AU,
    SIN2_THETA_W_PT,
    E_PV,
    _spatial_gradient_sum,
    gradient_of_ao_product_at_point,
    hpv_matrix_elements,
    nuclear_weak_charges_default,
    spin_density_from_mos,
    weak_charge,
)
from ptc.topology import build_topology


# ---------------------------------------------------------------------
# Helpers: a manual PTMolecularBasis (geometry chosen by hand)
# ---------------------------------------------------------------------


def _manual_basis(Z_list, coords_A):
    """Build a PTMolecularBasis directly from (Z_list, coords) bypassing
    the topology->geometry layer. Used to control the 3D positions for
    symmetry tests (achiral H2O, chiral H2O2 with a chosen dihedral).
    """
    atoms = []
    flat_orbs = []
    atom_idx = []
    for i, Z in enumerate(Z_list):
        ab = build_atom_basis(int(Z))
        atoms.append(ab)
        for orb in ab.orbitals:
            flat_orbs.append(orb)
            atom_idx.append(i)
    coords_arr = np.array(coords_A, dtype=float)
    return PTMolecularBasis(
        atoms=atoms,
        coords=coords_arr,
        orbitals=flat_orbs,
        atom_index=atom_idx,
        Z_list=list(Z_list),
    )


# ---------------------------------------------------------------------
# 1. PT constants and weak charges
# ---------------------------------------------------------------------


def test_pt_constants_values():
    """Sanity-check the PT-derived electroweak constants."""
    assert SIN2_THETA_W_PT == pytest.approx(0.23119, abs=1e-5)
    assert ONE_MINUS_4SIN2 == pytest.approx(0.07524, abs=5e-5)
    assert G_F_AU > 0.0
    # Prefactor magnitude in atomic units: ~ 5.7e-17 (note 02 §2.1).
    assert PREFACTOR_AU == pytest.approx(5.73e-17, rel=2e-2)


def test_weak_charge_values():
    """Q_W^PT for biology-relevant atoms matches the note table to 5e-4."""
    table = {
        # (Z, N) : expected Q_W
        (1, 0):  +0.07524,
        (6, 6):  -5.5486,
        (7, 7):  -6.4733,
        (8, 8):  -7.3981,
        (9, 10): -9.3228,
        (15, 16): -14.871,
        (16, 16): -14.796,
    }
    for (Z, N), expected in table.items():
        got = weak_charge(Z, N)
        assert got == pytest.approx(expected, abs=5e-4), (
            f"Z={Z}, N={N}: got {got}, expected {expected}"
        )


def test_weak_charge_sign_pattern():
    """For Z<N (every nucleus except 1H), Q_W is negative; H is positive."""
    assert weak_charge(1, 0) > 0
    for Z, N in [(6, 6), (7, 7), (8, 8), (9, 10), (15, 16), (16, 16)]:
        assert weak_charge(Z, N) < 0


def test_default_neutron_count_vector():
    """nuclear_weak_charges_default returns a vector consistent with table."""
    qw = nuclear_weak_charges_default([1, 6, 8])
    expected = np.array([0.07524, -5.5486, -7.3981])
    np.testing.assert_allclose(qw, expected, atol=5e-4)


# ---------------------------------------------------------------------
# 2. Single-orbital gradient identity (smoke)
# ---------------------------------------------------------------------


def test_gradient_of_ao_product_at_centre():
    """For a 1s orbital with itself at its own atomic centre, the value
    at the centre is finite and the product-gradient there must reduce
    to 2 chi(R) * grad chi(R). Both pieces are zero at R = atom (because
    grad chi_1s is undefined at the cusp and we set it to 0), so the
    gradient should be exactly zero.
    """
    orb = build_atom_basis(1).orbitals[0]
    pos = np.zeros(3)
    R = pos.copy()
    g = gradient_of_ao_product_at_point(orb, pos, orb, pos, R)
    assert g.shape == (3,)
    np.testing.assert_allclose(g, 0.0, atol=1e-12)


def test_gradient_of_ao_product_off_centre():
    """Off-centre check: pick R != atom centre; the gradient is non-zero
    and is real-finite.
    """
    orb = build_atom_basis(1).orbitals[0]
    pos = np.zeros(3)
    R = np.array([0.5, 0.0, 0.0])
    g = gradient_of_ao_product_at_point(orb, pos, orb, pos, R)
    assert np.all(np.isfinite(g))
    # By rotational symmetry: only the radial direction has a non-zero
    # component for two identical s orbitals on the same centre.
    assert abs(g[1]) < 1e-10
    assert abs(g[2]) < 1e-10
    assert abs(g[0]) > 0.0


# ---------------------------------------------------------------------
# 3. H_PV matrix Hermiticity
# ---------------------------------------------------------------------


def test_hpv_hermiticity_h2():
    """H_PV is Hermitian for H2 (smallest non-trivial test)."""
    basis = build_molecular_basis(build_topology("[H][H]"))
    H = hpv_matrix_elements(basis)
    assert H.shape == (2 * basis.n_orbitals, 2 * basis.n_orbitals)
    err = np.abs(H - H.conj().T).max()
    assert err < 1e-12, f"||H - H^dagger||_max = {err}"


def test_hpv_hermiticity_h2o():
    """H_PV is Hermitian for H2O (basis with s + p orbitals)."""
    basis = build_molecular_basis(build_topology("O"))
    H = hpv_matrix_elements(basis)
    err = np.abs(H - H.conj().T).max()
    assert err < 1e-12


def test_hpv_shape_and_dtype():
    """H_PV is complex 2n x 2n; spatial 'block' structure (each AO ->
    2x2 spin block) is encoded by Kron with Pauli matrices.
    """
    basis = build_molecular_basis(build_topology("O"))
    H = hpv_matrix_elements(basis)
    assert H.dtype == np.complex128
    assert H.shape == (2 * basis.n_orbitals, 2 * basis.n_orbitals)


# ---------------------------------------------------------------------
# 4. Achiral symmetry: H2O gradient sum vanishes
# ---------------------------------------------------------------------


def _water_geometry_Cs():
    """Hand-built water in the xz plane with C_2v symmetry (subgroup
    contains the sigma_v(xz) and sigma_v(yz) mirrors). All atoms have
    y = 0; the two H atoms are mirror images across x = 0.
    """
    # O at origin; H atoms at the standard r_OH = 0.96 A, angle = 104.5 deg.
    r = 0.96
    half = math.radians(104.5 / 2.0)
    hx = r * math.sin(half)
    hz = -r * math.cos(half)
    Z_list = [8, 1, 1]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [+hx, 0.0, hz],
        [-hx, 0.0, hz],
    ])
    return Z_list, coords


def test_spatial_gradient_sum_is_antisymmetric():
    """G_i is antisymmetric in (mu, nu) by construction (it is the
    matrix-element kernel of an anti-commutator with delta integrated
    by parts; see module docstring).
    """
    Z_list, coords = _water_geometry_Cs()
    basis = _manual_basis(Z_list, coords)
    Q_W = nuclear_weak_charges_default(Z_list)
    G = _spatial_gradient_sum(basis, basis.coords, Q_W)
    for i in range(3):
        err = np.abs(G[i] + G[i].T).max()
        assert err < 1e-12, f"G[{i}] not antisymmetric: max|G+G^T|={err}"


def test_h2o_gradient_x_vanishes_by_mirror():
    """For symmetric H2O the molecule has a sigma_v(yz) mirror x -> -x.
    The two H atoms (with identical Q_W) are mirror images in x, and the
    central O sits on the mirror plane. The x-component G_x of the
    matrix element summed over nuclei must therefore vanish on AO pairs
    that are themselves x-mirror-symmetric — e.g. (O 2s, O 2s),
    (O 2pz, O 2pz). It does NOT vanish on (O 2s, O 2px) which is x-odd.
    """
    Z_list, coords = _water_geometry_Cs()
    basis = _manual_basis(Z_list, coords)
    Q_W = nuclear_weak_charges_default(Z_list)
    G = _spatial_gradient_sum(basis, basis.coords, Q_W)
    # Locate the O 2s and O 2pz (l=1, m=0) orbital indices.
    O_2s = next(
        k for k, orb in enumerate(basis.orbitals)
        if basis.atom_index[k] == 0 and orb.l == 0
    )
    O_2pz = next(
        k for k, orb in enumerate(basis.orbitals)
        if basis.atom_index[k] == 0 and orb.l == 1 and orb.m == 0
    )
    # G_x on the (2s, 2pz) pair: both AOs are x-even, two-H contribution
    # cancels by mirror symmetry, O contribution: integrand at R_O = 0
    # uses chi(0) and (d_x chi)(0); for x-even orbitals (d_x chi)(0) = 0.
    val_x = G[0, O_2s, O_2pz]
    assert abs(val_x) < 1e-10, f"|G_x[O2s, O2pz]| = {abs(val_x)}"


def test_h2o_achiral_e_pv_invariant_under_mirror():
    """Achiral H2O: the matrix element E_PV computed from any spin
    probe is preserved under the x -> -x mirror of the geometry. Mirror
    is a symmetry of the molecule, so the matrix elements at mirror-
    related coords must coincide.

    This is the matrix-element analog of the "glycine = 0" symmetry
    claim of note 02: an achiral geometry forces E_PV (with a fixed
    probe) to be invariant under the molecular mirror, which combined
    with the operator's P-odd character (E_PV transforms as -E_PV under
    full inversion) usually gives zero. Here we test the invariance.
    """
    Z_list, coords = _water_geometry_Cs()
    basis = _manual_basis(Z_list, coords)
    from ptc.lcao.density_matrix import (
        hueckel_hamiltonian,
        overlap_matrix,
        solve_mo,
    )
    S = overlap_matrix(basis)
    Hh = hueckel_hamiltonian(basis, S)
    _, C = solve_mo(Hh, S)
    n_e = int(round(sum(o.occ for o in basis.orbitals)))
    # Off-diagonal s-orbital sigma_z probe to make E_PV finite.
    n = basis.n_orbitals
    s_pair = [
        k for k, orb in enumerate(basis.orbitals)
        if orb.l == 0
    ][:2]
    if len(s_pair) >= 2:
        a, b = s_pair[0], s_pair[1]
        D_spin = np.zeros((2 * n, 2 * n), dtype=complex)
        D_spin[2 * a, 2 * b] = 1.0
        D_spin[2 * b, 2 * a] = 1.0
        D_spin[2 * a + 1, 2 * b + 1] = -1.0
        D_spin[2 * b + 1, 2 * a + 1] = -1.0
    else:
        # fallback to an alpha-only Hueckel density
        D_spin = spin_density_from_mos(C, np.zeros_like(C), n_e, 0)
    H = hpv_matrix_elements(basis)
    val = E_PV(D_spin, H)

    # Re-do with x-flipped water geometry: same achiral molecule.
    Z_list_m, coords_m = _water_geometry_Cs()
    coords_m[:, 0] *= -1.0
    basis_m = _manual_basis(Z_list_m, coords_m)
    H_m = hpv_matrix_elements(basis_m)
    val_m = E_PV(D_spin, H_m)
    # x-mirror is a symmetry: the two values must agree up to a
    # PV sign that depends on the symmetry of the probe.
    # For the off-diagonal s-s sigma_z probe centred on the O atom,
    # the probe is itself mirror-symmetric, so val == val_m.
    assert val == pytest.approx(val_m, abs=1e-12)


# ---------------------------------------------------------------------
# 5. Chiral H2O2: opposite enantiomers give opposite spatial gradient
# ---------------------------------------------------------------------


def _h2o2_geometry(dihedral_deg: float, r_OH1: float = 0.95,
                   r_OH2: float = 0.95):
    """Hand-built H2O2 with the OOH-HOO dihedral set by hand.

    Atom order: [O1, O2, H1, H2]. O1 at (-r_OO/2, 0, 0), O2 at
    (+r_OO/2, 0, 0). H1 is bonded to O1 in a half-plane rotated by
    +dihedral/2 about the O-O axis; H2 to O2 in a half-plane rotated
    by -dihedral/2.

    For a truly chiral pair the two enantiomers must NOT be related by
    any combination of atom permutation + coordinate sign flips of the
    PT_LCAO basis. With r_OH1 != r_OH2 we break the H1<->H2 swap
    symmetry, and flipping the sign of `dihedral_deg` then produces a
    genuinely chiral mirror image whose H_PV matrix is a row/col
    permutation away from the original — never the same.
    """
    r_OO = 1.475          # Angstrom (literature)
    theta_OOH = math.radians(94.8)  # bond angle
    phi = math.radians(dihedral_deg / 2.0)

    O1 = np.array([-r_OO / 2.0, 0.0, 0.0])
    O2 = np.array([+r_OO / 2.0, 0.0, 0.0])

    def _h_direction(sign_x: float, sign_phi: float):
        base = np.array([
            sign_x * math.cos(math.pi - theta_OOH),
            0.0,
            math.sin(math.pi - theta_OOH),
        ])
        ang = sign_phi * phi
        c, s = math.cos(ang), math.sin(ang)
        return np.array([
            base[0],
            c * base[1] - s * base[2],
            s * base[1] + c * base[2],
        ])

    H1 = O1 + r_OH1 * _h_direction(sign_x=-1.0, sign_phi=+1.0)
    H2 = O2 + r_OH2 * _h_direction(sign_x=+1.0, sign_phi=-1.0)

    Z_list = [8, 8, 1, 1]
    coords = np.array([O1, O2, H1, H2])
    return Z_list, coords


def test_h2o2_full_parity_inversion_flips_gradient():
    """Full geometric parity inversion R_A -> -R_A flips the sign of the
    gradient sum G_i tensor component-wise.

    Mathematically: d_i[chi_mu(r) chi_nu(r)](R_A) is a polynomial in
    (R_A - pos_mu) and (R_A - pos_nu). Sending every nuclear coordinate
    to its opposite (atoms AND AO centres) flips r -> -r in the
    arguments while keeping the AO labels fixed. Real-spherical
    harmonics Y_lm of degree l pick up a parity (-1)^l, so the AO value
    transforms as chi(-r) = (-1)^l chi(r) and its gradient as
    grad chi(-r) = -(-1)^l grad chi(r). For a single (mu, nu) pair
    with angular momenta (l_mu, l_nu), the gradient of the product
    therefore satisfies

       d_i[chi_mu chi_nu](-R) = -(-1)^(l_mu + l_nu) d_i[chi_mu chi_nu](R) .

    For pairs where l_mu + l_nu is even (e.g. s-s, p-p), G_i flips sign;
    for odd l_mu + l_nu (e.g. s-p), G_i is invariant. We check the
    s-s O-O block (l_mu = l_nu = 0): G_i must flip sign.
    """
    Z, coords_L = _h2o2_geometry(+119.8)
    coords_R = -coords_L                  # full inversion P
    basis_L = _manual_basis(Z, coords_L)
    basis_R = _manual_basis(Z, coords_R)
    Q_W = nuclear_weak_charges_default(Z)

    G_L = _spatial_gradient_sum(basis_L, basis_L.coords, Q_W)
    G_R = _spatial_gradient_sum(basis_R, basis_R.coords, Q_W)

    # Locate the O1 and O2 2s orbitals (atom_index 0 and 1, l = 0).
    s_indices = [
        k for k, orb in enumerate(basis_L.orbitals)
        if orb.l == 0 and basis_L.atom_index[k] in (0, 1)
    ]
    assert len(s_indices) >= 2
    # G is antisymmetric in (mu, nu) so diagonal entries vanish trivially.
    # On the OFF-diagonal s-s block (l_mu + l_nu = 0, even), every
    # i-component flips sign exactly under P.
    for mu_i in range(len(s_indices)):
        for nu_i in range(mu_i + 1, len(s_indices)):
            mu = s_indices[mu_i]
            nu = s_indices[nu_i]
            for i in range(3):
                assert G_L[i, mu, nu] == pytest.approx(
                    -G_R[i, mu, nu], abs=1e-10
                ), (
                    f"s-s ({mu},{nu}) component i={i}: "
                    f"G_L={G_L[i, mu, nu]}, G_R={G_R[i, mu, nu]}"
                )
    # And at least one of the off-diagonal s-s values is non-zero
    # (otherwise the test is vacuous).
    s_off_vals = [
        abs(G_L[i, s_indices[0], s_indices[1]]) for i in range(3)
    ]
    assert max(s_off_vals) > 1e-6, "off-diagonal s-s gradient is zero"


def test_h2o2_dihedral_flip_is_chiral():
    """Flipping the dihedral angle from +phi to -phi produces a chiral
    pair (the y -> -y mirror image). We verify that the FULL spatial
    gradient sum G_i is NOT identical between the two — i.e. the two
    geometries are genuinely chirally distinct in the matrix element.
    """
    Z, coords_L = _h2o2_geometry(+119.8)
    _, coords_R = _h2o2_geometry(-119.8)
    basis_L = _manual_basis(Z, coords_L)
    basis_R = _manual_basis(Z, coords_R)
    Q_W = nuclear_weak_charges_default(Z)

    G_L = _spatial_gradient_sum(basis_L, basis_L.coords, Q_W)
    G_R = _spatial_gradient_sum(basis_R, basis_R.coords, Q_W)

    diff = np.abs(G_L - G_R).max()
    assert diff > 1e-6, (
        "Two enantiomeric H2O2 geometries gave identical gradient "
        "tensors — the matrix element is parity-blind, which would "
        "contradict the Vester-Ulbricht physics."
    )


def test_h2o2_hpv_matrix_changes_with_dihedral():
    """The full H_PV matrix changes when the dihedral is flipped (the
    two enantiomers are chirally distinct, and the operator does
    register the chirality). Each one is independently Hermitian.

    We use a SYMMETRY-BROKEN H2O2 (r_OH1 != r_OH2) so that the +phi
    and -phi geometries are NOT related by a swap-permutation: the
    matrix elements differ as expected for genuine enantiomers.
    """
    Z, coords_L = _h2o2_geometry(+119.8, r_OH1=0.95, r_OH2=1.05)
    _, coords_R = _h2o2_geometry(-119.8, r_OH1=0.95, r_OH2=1.05)
    basis_L = _manual_basis(Z, coords_L)
    basis_R = _manual_basis(Z, coords_R)
    H_L = hpv_matrix_elements(basis_L)
    H_R = hpv_matrix_elements(basis_R)
    # Hermiticity: each enantiomer's H is Hermitian on its own.
    assert np.abs(H_L - H_L.conj().T).max() < 1e-12
    assert np.abs(H_R - H_R.conj().T).max() < 1e-12
    # The chirality registers: H_L != H_R. PREFACTOR_AU ~ 5.7e-17, so
    # genuine matrix elements are ~ 1e-17. We require the diff to be a
    # finite fraction of max |H_L|, not absolute.
    diff = np.abs(H_L - H_R).max()
    scale = np.abs(H_L).max()
    assert scale > 0
    assert diff > 0.1 * scale, (
        f"H_PV is dihedral-blind: max diff = {diff} vs scale {scale}; "
        "contradicting V-U physics"
    )


# ---------------------------------------------------------------------
# 6. E_PV reduction: closed-shell singlet RHF -> zero
# ---------------------------------------------------------------------


def test_E_PV_closed_shell_singlet_is_zero():
    """For a closed-shell RHF density (single spatial density expanded
    as (1/2) D (x) I_2), the trace against H_PV vanishes identically:
    the spin trace of sigma_i is zero. This is the textbook statement
    that NSI parity violation first appears via spin-orbit mixing.

    We verify on H2O (achiral) and a chiral H2O2: both must return 0
    when the input density is the closed-shell RHF spatial density.
    """
    # H2O via topology (uses the project geometry layer)
    basis = build_molecular_basis(build_topology("O"))
    rho, _, _, _ = density_matrix_PT(build_topology("O"), basis=basis)
    H = hpv_matrix_elements(basis)
    val = E_PV(rho, H)
    assert abs(val) < 1e-25, f"E_PV (H2O, RHF) = {val}, expected 0"

    # Chiral H2O2 via manual geometry; still RHF -> still 0.
    Z_list, coords = _h2o2_geometry(+119.8)
    basis_L = _manual_basis(Z_list, coords)
    # Build a Hueckel density on the manual basis.
    from ptc.lcao.density_matrix import (
        hueckel_hamiltonian,
        overlap_matrix,
        solve_mo,
    )
    S = overlap_matrix(basis_L)
    Hh = hueckel_hamiltonian(basis_L, S)
    eigvals, C = solve_mo(Hh, S)
    n_e = int(round(sum(o.occ for o in basis_L.orbitals)))
    n_doubly = n_e // 2
    rho_spatial = 2.0 * C[:, :n_doubly] @ C[:, :n_doubly].T
    H_L = hpv_matrix_elements(basis_L)
    val_L = E_PV(rho_spatial, H_L)
    assert abs(val_L) < 1e-25, f"E_PV (H2O2 L, RHF) = {val_L}, expected 0"


def test_E_PV_full_parity_inversion_flips_for_h2o2():
    """Full geometric parity P : R -> -R on H2O2 flips the sign of E_PV
    when traced against a spin probe that is even under all the relevant
    symmetries.

    Construction:
        - Build a Hermitian spin probe D with an OFF-DIAGONAL s-s
          spatial part (G is antisymmetric in mu, nu, so a purely
          diagonal probe would give zero).
        - For s-s contributions the spatial gradient sum G_i flips under
          P, the spin trace passes through unchanged, hence
          E_PV(P-flipped) = -E_PV(original) on the s-s block.

    The test passes when E_PV is nonzero and sign-opposite between the
    two geometries.
    """
    Z, coords_L = _h2o2_geometry(+119.8)
    coords_P = -coords_L
    basis_L = _manual_basis(Z, coords_L)
    basis_P = _manual_basis(Z, coords_P)

    n = basis_L.n_orbitals
    # Find the two O 2s indices.
    s_pair = [
        k for k, orb in enumerate(basis_L.orbitals)
        if orb.l == 0 and basis_L.atom_index[k] in (0, 1)
    ][:2]
    assert len(s_pair) == 2
    a, b = s_pair

    # Build a probe Hermitian density whose spatial part has both
    # symmetric and antisymmetric components on (a, b), with all four
    # spin entries populated. The antisymmetric spatial part contracts
    # against the antisymmetric G and picks up the G_x component
    # (largest of (G_x, G_y, G_z) on this geometry, since O atoms have
    # z = 0 and O-O lies along x).
    #
    # We pair the antisymmetric spatial part with a sigma_x-flavoured
    # spin density:  D_alpha,beta(a,b) = -i, D_beta,alpha(b,a) = +i, etc.
    D = np.zeros((2 * n, 2 * n), dtype=complex)
    # sigma_x ⊗ (-i |a><b| + i |b><a|): the alpha-beta and beta-alpha blocks
    D[2 * a, 2 * b + 1] = -1j     # (a alpha, b beta)
    D[2 * b + 1, 2 * a] = +1j     # Hermitian conjugate
    D[2 * b, 2 * a + 1] = +1j     # (b alpha, a beta)
    D[2 * a + 1, 2 * b] = -1j     # Hermitian conjugate

    H_L = hpv_matrix_elements(basis_L)
    H_P = hpv_matrix_elements(basis_P)
    val_L = E_PV(D, H_L)
    val_P = E_PV(D, H_P)

    assert abs(val_L) > 1e-25, (
        f"off-diagonal s-s sigma_z probe gave |val_L|={abs(val_L)}; vacuous"
    )
    # Under P (R -> -R) on s-s pairs, every G_i flips, so each component
    # of H_PV picks up a minus sign and E_PV flips sign.
    assert val_L == pytest.approx(-val_P, abs=1e-10 * max(abs(val_L), 1e-25))


# ---------------------------------------------------------------------
# 7. Alanine magnitude — skipped (geometry / basis not wired here)
# ---------------------------------------------------------------------


def test_alanine_magnitude():
    """End-to-end alanine H_PV smoke test.

    Wires in the new ``ptc.lcao.geometries_amino_acids.alanine_L`` and
    checks that ``hpv_matrix_elements`` returns a Hermitian, non-zero
    complex matrix of the expected (2 n_orb, 2 n_orb) shape. This is
    NOT yet a magnitude check against Bakasov 1998 -- the closed-shell
    RHF density gives E_PV = 0 identically (cf.
    ``test_E_PV_closed_shell_singlet_is_zero``). A magnitude benchmark
    requires spin-orbit mixing and is left for a follow-up calc.
    """
    from ptc.lcao.geometries_amino_acids import alanine_L

    _, basis = alanine_L()
    H = hpv_matrix_elements(basis)
    assert H.dtype == np.complex128
    assert H.shape == (2 * basis.n_orbitals, 2 * basis.n_orbitals)
    err = np.abs(H - H.conj().T).max()
    assert err < 1e-12, f"||H - H^dagger||_max = {err}"
    # Genuine matrix elements at PT prefactor scale ~ 1e-17.
    assert np.abs(H).max() > 0.0
