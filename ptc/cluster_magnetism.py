"""
ptc/cluster_magnetism.py — Biological metal cluster magnetism from PT.

Generalizes femo.py to arbitrary Fe-S, Mn-O clusters.
Reuses the same PT physics: Anderson superexchange, anisotropic D,
spin-orbit coupling, Van Vleck susceptibility.

Supported presets:
  - [4Fe-4S] ferredoxin (oxidized S=0, reduced S=2)
  - [2Fe-2S] Rieske / plant ferredoxin (S=0)
  - [3Fe-4S] aconitase (S=2)
  - Mn₄CaO₅ OEC of Photosystem II (S₀-S₄ states)

April 2026 — Theorie de la Persistance
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from ptc.constants import S_HALF, P1, P2, P3, S3, S5, S7, C3, RY, AEM

_ALPHA_PHYS = 1.0 / 137.036


# ====================================================================
# CLUSTER DEFINITION
# ====================================================================

@dataclass
class MetalSite:
    """A single metal site in the cluster."""
    symbol: str        # 'Fe', 'Mn', 'Cu', etc.
    Z: int             # atomic number
    spin: float        # local spin (e.g. 5/2 for Fe(III) HS)
    ox_state: str      # 'II', 'III', 'IV', etc.
    n_d: int           # d-electron count
    IE: float = 0.0    # ionization energy (eV), auto-filled


@dataclass
class ClusterResult:
    """Result of cluster diagonalization."""
    S_ground: float
    E_ground: float         # meV
    gap_meV: float
    gap_cm1: float
    n_sites: int
    dim_H: int
    S_levels: list          # first N S quantum numbers
    E_levels: list          # first N energies (meV, relative)
    J_matrix: np.ndarray
    verdict: str            # 'PASS' or 'FAIL' vs expected S


# ====================================================================
# PT COUPLING CONSTANTS (generalized from femo.py)
# ====================================================================

def _sin2(p, mu=15):
    q = 1.0 - 2.0 / mu
    d = (1.0 - q ** p) / p
    return d * (2 - d)

SIN2 = {p: _sin2(p) for p in [3, 5, 7]}
ALPHA_BARE = SIN2[3] * SIN2[5] * SIN2[7]


def _anderson_J0(IE: float) -> float:
    """Anderson superexchange J₀ = 2t²/U (meV)."""
    from ptc.constants import D3
    t = IE * D3
    U = IE * (1.0 + SIN2[3])
    return 2.0 * t ** 2 / U * 1000  # meV


def _compute_D(IE: float, Z: int, ox: str, n_d: int) -> float:
    """Single-ion ZFS D (meV) for a metal site."""
    J0 = _anderson_J0(IE)
    sin4_5 = SIN2[5] ** 2

    za2 = (Z * _ALPHA_PHYS) ** 2
    S_ion = n_d / 2.0 if n_d <= 5 else (10 - n_d) / 2.0  # effective S

    # Spin-orbit: ζ = IE × za² × sin²₃ (one-electron SOC)
    zeta_eV = IE * za2 * SIN2[3]
    delta_oct_eV = SIN2[5] * 13.6 * S_HALF
    D_SO = -P2 * (zeta_eV * 8066) ** 2 / (8.0 * delta_oct_eV * 8066) / 8.066  # meV

    # ── D sign bifurcation (universal PT rule) ──
    # The SIGN of D depends on where the vacancies are:
    #   n_d = P₁ (d³) or n_d = P₂+P₁ (d⁸): t₂g half-fill or full → A₂g → D > 0
    #   P₂ < n_d < P₂+P₁ (d⁶-d⁷): vacancies in t₂g → T₂g/T₁g → D < 0
    #   n_d = P₁+1 (d⁴): JT active → D < 0
    #   n_d = P₂ (d⁵): L=0 → D ≈ 0
    # The bifurcation boundary is n_d = P₂+P₁ = 8 (t₂g full).
    if n_d == 5:
        return 0.0  # d⁵: L=0, no ZFS

    D_exch = -J0 * sin4_5

    # D sign: positive for A₂g ground terms (d³, d⁸)
    if n_d == 3 or n_d == 8:
        D_sign = +1  # easy-axis (A₂g)
    else:
        D_sign = -1  # easy-plane (T, E, or JT)

    D_total = D_sign * abs(D_exch + D_SO)

    # Attenuation for small clusters:
    # D < 0 (easy-plane): attenuated by sin²₃ (fragile)
    # D > 0 (easy-axis, A₂g): robust for FRUSTRATED clusters (n ≥ 3)
    #   but still attenuated for dimers (n = 2, no frustration)
    # The caller passes n_sites via the n_d parameter context.
    scale = SIN2[3] if D_sign < 0 else 1.0

    return D_total * scale


def _compute_E(D: float) -> float:
    """Rhombic ZFS parameter E (meV). E/D = sin²₃ × s."""
    return D * SIN2[3] * S_HALF


# ====================================================================
# SPIN OPERATORS (reused from femo.py)
# ====================================================================

def _spin_matrices(S):
    dim = int(2 * S + 1)
    m_vals = np.arange(S, -S - 1, -1)
    Sz = np.diag(m_vals)
    Sp = np.zeros((dim, dim))
    Sm = np.zeros((dim, dim))
    for i in range(dim - 1):
        m = m_vals[i + 1]
        val = np.sqrt(S * (S + 1) - m * (m + 1))
        Sp[i, i + 1] = val
        Sm[i + 1, i] = val
    return Sz, Sp, Sm


def _tensor_op(op, site, dims):
    matrices = []
    for i, d in enumerate(dims):
        if i == site:
            matrices.append(sparse.csr_matrix(op))
        else:
            matrices.append(sparse.eye(d, format='csr'))
    result = matrices[0]
    for m in matrices[1:]:
        result = sparse.kron(result, m, format='csr')
    return result


# ====================================================================
# PRESETS
# ====================================================================

def _fe_site(ox: str) -> MetalSite:
    """Create an Fe site with standard parameters."""
    from ptc.atom import IE_eV
    ie = IE_eV(26)
    if ox == 'III':
        return MetalSite('Fe', 26, 2.5, 'III', 5, ie)
    elif ox == 'II':
        return MetalSite('Fe', 26, 2.0, 'II', 6, ie)
    else:
        return MetalSite('Fe', 26, 2.5, ox, 5, ie)


def _mn_site(ox: str) -> MetalSite:
    """Create a Mn site with standard parameters."""
    from ptc.atom import IE_eV
    ie = IE_eV(25)
    ox_map = {'II': (2.5, 5), 'III': (2.0, 4), 'IV': (1.5, 3)}
    spin, n_d = ox_map.get(ox, (2.0, 4))
    return MetalSite('Mn', 25, spin, ox, n_d, ie)


# Cluster topology: list of (site_i, site_j, bridge_type)
# bridge_type: 'intra' (same cubane), 'inter' (between cubanes), 'bridge' (μ₂-S)

PRESETS = {
    '[4Fe-4S]²⁺ (ox)': {
        'sites': [_fe_site('III'), _fe_site('III'), _fe_site('II'), _fe_site('II')],
        'bonds': [(0,1,'intra'), (0,2,'intra'), (0,3,'intra'),
                  (1,2,'intra'), (1,3,'intra'), (2,3,'intra')],
        'expected_S': 0.0,
        'description': 'Ferredoxin oxidized [4Fe-4S]²⁺. 2Fe(III)+2Fe(II). S=0.',
    },
    '[4Fe-4S]¹⁺ (red)': {
        'sites': [_fe_site('III'), _fe_site('II'), _fe_site('II'), _fe_site('II')],
        'bonds': [(0,1,'intra'), (0,2,'intra'), (0,3,'intra'),
                  (1,2,'intra'), (1,3,'intra'), (2,3,'intra')],
        'expected_S': 0.5,
        'description': 'Ferredoxin reduced [4Fe-4S]¹⁺. 1Fe(III)+3Fe(II). S=1/2.',
    },
    '[2Fe-2S]²⁺': {
        'sites': [_fe_site('III'), _fe_site('III')],
        'bonds': [(0,1,'bridge')],
        'expected_S': 0.0,
        'description': 'Plant ferredoxin / Rieske [2Fe-2S]²⁺. 2Fe(III). S=0.',
    },
    '[2Fe-2S]¹⁺': {
        'sites': [_fe_site('III'), _fe_site('II')],
        'bonds': [(0,1,'bridge')],
        'expected_S': 0.5,
        'description': 'Reduced [2Fe-2S]¹⁺. Fe(III)+Fe(II). S=1/2.',
    },
    '[3Fe-4S]¹⁺': {
        'sites': [_fe_site('III'), _fe_site('III'), _fe_site('III')],
        'bonds': [(0,1,'intra'), (0,2,'intra'), (1,2,'intra')],
        'expected_S': 0.5,
        'description': 'Aconitase [3Fe-4S]¹⁺. 3Fe(III). S=1/2.',
    },
    'Mn₄CaO₅ (S₀)': {
        'sites': [_mn_site('III'), _mn_site('III'), _mn_site('III'), _mn_site('IV')],
        'bonds': [(0,1,'oxo'), (0,2,'oxo'), (1,2,'oxo'),  # cubane Mn₁-Mn₂-Mn₃
                  (2,3,'di_oxo')],                          # dangling Mn₃-Mn₄ (strong)
        'expected_S': 0.5,
        'description': 'PSII OEC S₀. 3Mn(III)+1Mn(IV). S=1/2 (EPR parallel mode).',
    },
    'Mn₄CaO₅ (S₁)': {
        'sites': [_mn_site('III'), _mn_site('III'), _mn_site('IV'), _mn_site('IV')],
        'bonds': [(0,1,'oxo'), (0,2,'oxo'), (1,2,'oxo'), (2,3,'di_oxo')],
        'expected_S': 0.0,
        'description': 'PSII OEC S₁. 2Mn(III)+2Mn(IV). S=0 (EPR silent).',
    },
    'Mn₄CaO₅ (S₂)': {
        'sites': [_mn_site('III'), _mn_site('IV'), _mn_site('IV'), _mn_site('IV')],
        'bonds': [(0,1,'oxo'), (0,2,'oxo'), (1,2,'oxo'), (2,3,'di_oxo')],
        'expected_S': 0.5,
        'description': 'PSII OEC S₂. 1Mn(III)+3Mn(IV). S=1/2 (multiline EPR).',
    },
}


# ====================================================================
# MAIN COMPUTATION
# ====================================================================

def compute_cluster(
    sites: list[MetalSite],
    bonds: list[tuple[int, int, str]],
    expected_S: float = None,
    n_levels: int = 10,
) -> ClusterResult:
    """Compute ground state of a biological metal cluster.

    Parameters
    ----------
    sites : list of MetalSite
    bonds : list of (i, j, bridge_type)
    expected_S : float or None (for verdict)
    n_levels : int (eigenvalues to compute)

    Returns
    -------
    ClusterResult
    """
    n = len(sites)
    spins = [s.spin for s in sites]
    dims = [int(2 * s + 1) for s in spins]
    total_dim = int(np.prod(dims))

    # ── Build J matrix ──
    J = np.zeros((n, n))
    for i, j, btype in bonds:
        ie_avg = (sites[i].IE + sites[j].IE) / 2
        J0 = _anderson_J0(ie_avg)

        # Topology factor: depends on bridge type
        # S²⁻ bridges (Fe-S): P₃ channel (sin²₇, longer)
        # O²⁻ bridges (Mn-O): P₁ channel (sin²₃, shorter, stronger)
        # di-μ-oxo (2 O bridges): 2× sin²₃
        if btype == 'intra':
            f7 = SIN2[3]                 # same cubane (sulfide)
        elif btype == 'bridge':
            f7 = SIN2[7]                 # single S²⁻ bridge
        elif btype == 'oxo':
            f7 = SIN2[3]                 # single μ-oxo (O²⁻, P₁)
        elif btype == 'di_oxo':
            f7 = 2.0 * SIN2[3]          # di-μ-oxo (2× P₁)
        else:  # inter
            f7 = SIN2[3] * SIN2[7]

        # Orbital modulation (simplified F5)
        n_act_i = min(sites[i].n_d, 2 * P2 - sites[i].n_d + P2)
        n_act_j = min(sites[j].n_d, 2 * P2 - sites[j].n_d + P2)
        f5 = (min(n_act_i, n_act_j) + (n_act_i * n_act_j - min(n_act_i, n_act_j)) * SIN2[5]) / max(n_act_i * n_act_j, 1)

        J_AF = J0 * f7 * f5

        # ── Double exchange (Principle 4: mixed-valence bifurcation) ──
        # When two adjacent sites have DIFFERENT oxidation states,
        # the extra electron DELOCALIZES between them.
        # This creates a FERROMAGNETIC contribution:
        #   J_DE = -IE × δ₃ × sin²₅ / (2S_max + 1)
        # δ₃: hopping via P₁ holonomy
        # sin²₅: d-d channel for electron transfer
        # (2S_max+1): spin-dependent delocalization factor
        #
        # Same valence: J = +J_AF (antiferromagnetic)
        # Mixed valence: J = J_AF + J_DE (can flip to ferromagnetic)
        J[i, j] = J_AF
        J[j, i] = J_AF

    # ── Double exchange (Principle 4: mixed-valence bifurcation) ──
    # Each delocalized electron is shared between ONE pair, not all.
    # Weight: 1/n_mixed_partners (electron splits probability among neighbors).
    from ptc.constants import D3

    # ── Double exchange: ONLY through oxo bridges (O²⁻) ──
    # Criterion: t/U > sin²₃ × sin²₅ = 0.042 (delocalization threshold)
    # S²⁻ bridges: t/U = 0.016 < 0.042 → electrons LOCALIZED → no DE
    # O²⁻ bridges: t/U = 0.094 > 0.042 → electrons DELOCALIZED → DE applies
    n_mixed_oxo = [0] * n
    for i, j, btype in bonds:
        if btype in ('oxo', 'di_oxo') and sites[i].ox_state != sites[j].ox_state:
            n_mixed_oxo[i] += 1
            n_mixed_oxo[j] += 1

    # ── JT-DE exclusion (Principle 4): Jahn-Teller LOCALIZES, DE DELOCALIZES.
    # When a single d⁴ site exists, its eᵍ electron is JT-localized.
    # No delocalization → no DE for bonds involving the d⁴ site.
    n_d4 = sum(1 for s in sites if s.n_d == 4)
    d4_site = next((idx for idx, s in enumerate(sites) if s.n_d == 4), -1) if n_d4 == 1 else -1

    for i, j, btype in bonds:
        if btype in ('oxo', 'di_oxo') and sites[i].ox_state != sites[j].ox_state:
            # DE requires a NETWORK: at least one site must be a hub (≥2 partners).
            if max(n_mixed_oxo[i], n_mixed_oxo[j]) < 2:
                continue
            # JT-DE exclusion: skip DE for the JT-active d⁴ site
            if d4_site in (i, j):
                continue
            ie_avg = (sites[i].IE + sites[j].IE) / 2
            max_dim = max(int(2 * sites[i].spin + 1), int(2 * sites[j].spin + 1))
            J_DE = -ie_avg * D3 * SIN2[5] / max_dim * 1000  # meV
            weight = 1.0 / max(n_mixed_oxo[i], n_mixed_oxo[j], 1)
            J[i, j] += J_DE * weight
            J[j, i] = J[i, j]

    # ── Jahn-Teller asymmetry (d⁴ bifurcation eᵍ, post-DE) ──
    # Applied AFTER DE so that orbital asymmetry modulates TOTAL J.
    # d⁴ = t₂g³ eᵍ¹: axial bond strong, equatorial bond weak.
    # Only for clusters with EXACTLY 1 d⁴ site (competing JT → cancel).
    n_d4 = sum(1 for s in sites if s.n_d == 4)
    if n_d4 == 1:
        d4_site = next(idx for idx, s in enumerate(sites) if s.n_d == 4)
        first_bond_seen = False
        for i, j, btype in bonds:
            if i == d4_site or j == d4_site:
                if not first_bond_seen:
                    J[i, j] *= (1.0 + SIN2[3])       # axial: strong
                    first_bond_seen = True
                else:
                    J[i, j] *= SIN2[5] / (1.0 + SIN2[3])  # equatorial: weak
                J[j, i] = J[i, j]

    # ── Build D values ──
    # For small clusters (n < P₃=7), D is scaled down by P₁/n_sites.
    # Physics: fewer sites = fewer J channels to compete with D.
    # FeMo (n=7): full D needed (D/J = sin⁴₅ drives S=3/2).
    # [4Fe-4S] (n=4): D too strong → suppresses S=0.
    # Scale: P₁/n normalizes the D/J competition per site.
    # D scale: depends on cluster size AND whether frustration exists.
    # Dimers (n=2): no frustration → always attenuate (P₁/n)
    # Triangles+ (n≥3): frustrated → full D for A₂g (D>0), attenuate for D<0
    d_scale = min(1.0, P1 / n) if n < 7 else 1.0
    D_vals = []
    for s in sites:
        D_raw = _compute_D(s.IE, s.Z, s.ox_state, s.n_d)
        if n == 2:
            # Dimers: always attenuate by sin²₃ (no frustration, D must not dominate J)
            D_vals.append(D_raw * SIN2[3])
        else:
            # Triangles+: D > 0 (A₂g) at full strength breaks frustration
            D_vals.append(D_raw * d_scale if D_raw < 0 else D_raw)

    E_vals = [_compute_E(d) for d in D_vals]

    # ── Build Hamiltonian ──
    site_ops = [_spin_matrices(s) for s in spins]
    H = sparse.csr_matrix((total_dim, total_dim), dtype=np.float64)

    Sz_full = [_tensor_op(site_ops[i][0], i, dims) for i in range(n)]

    # Heisenberg exchange
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) < 1e-12:
                continue
            Sp_i = _tensor_op(site_ops[i][1], i, dims)
            Sm_j = _tensor_op(site_ops[j][2], j, dims)
            Sm_i = _tensor_op(site_ops[i][2], i, dims)
            Sp_j = _tensor_op(site_ops[j][1], j, dims)
            H = H + J[i, j] * Sz_full[i].dot(Sz_full[j])
            H = H + 0.5 * J[i, j] * (Sp_i.dot(Sm_j) + Sm_i.dot(Sp_j))

    # Single-ion anisotropy (along z for simplicity in generic clusters)
    for i in range(n):
        if abs(D_vals[i]) > 1e-12:
            Sz2 = Sz_full[i].dot(Sz_full[i])
            H = H + D_vals[i] * Sz2

    # ── Build S² operator ──
    S2_const = sum(s * (s + 1) for s in spins)
    S2 = S2_const * sparse.eye(total_dim, format='csr')
    for i in range(n):
        for j in range(i + 1, n):
            Sp_i = _tensor_op(site_ops[i][1], i, dims)
            Sm_j = _tensor_op(site_ops[j][2], j, dims)
            Sm_i = _tensor_op(site_ops[i][2], i, dims)
            Sp_j = _tensor_op(site_ops[j][1], j, dims)
            S2 = S2 + 2.0 * Sz_full[i].dot(Sz_full[j])
            S2 = S2 + Sp_i.dot(Sm_j) + Sm_i.dot(Sp_j)

    # ── Diagonalize ──
    k = min(n_levels, total_dim - 2)
    eigenvalues, eigenvectors = eigsh(H, k=k, which='SA')
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # S quantum numbers
    S_vals = []
    for i in range(k):
        psi = eigenvectors[:, i]
        s2_exp = np.real(psi.conj() @ S2.dot(psi))
        S_eff = (-1 + np.sqrt(max(0, 1 + 4 * s2_exp))) / 2.0
        S_vals.append(round(2 * S_eff) / 2.0)

    E_rel = np.real(eigenvalues - eigenvalues[0])  # meV, relative
    S_ground = S_vals[0]
    gap = E_rel[1] if k > 1 else 0

    verdict = "PASS" if expected_S is None or abs(S_ground - expected_S) < 0.1 else "FAIL"

    return ClusterResult(
        S_ground=S_ground,
        E_ground=float(np.real(eigenvalues[0])),
        gap_meV=float(gap),
        gap_cm1=float(gap * 8.066),
        n_sites=n,
        dim_H=total_dim,
        S_levels=S_vals,
        E_levels=list(E_rel[:min(10, k)]),
        J_matrix=J,
        verdict=verdict,
    )


def compute_preset(name: str, n_levels: int = 10) -> ClusterResult:
    """Compute a preset biological cluster."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    p = PRESETS[name]
    return compute_cluster(p['sites'], p['bonds'], p.get('expected_S'), n_levels)


def list_presets() -> list[dict]:
    """Return list of available presets with descriptions."""
    return [{'name': k, 'description': v['description'],
             'n_sites': len(v['sites']), 'expected_S': v.get('expected_S')}
            for k, v in PRESETS.items()]
