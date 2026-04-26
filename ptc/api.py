"""
api.py -- Unified API for PTC universal chemistry simulator.

Usage:
    from ptc.api import Molecule

    mol = Molecule("CCO")          # ethanol from SMILES
    print(mol.D_at)                # atomization energy (eV)
    print(mol.geometry)            # 3D coordinates
    print(mol.frequencies)         # vibrational modes (cm^-1)
    print(mol.thermo(T=298))       # H, S, G at temperature T
    mol.visualize()                # 3D interactive (if available)

0 adjustable parameters. All from s = 1/2.

April 2026 -- Theorie de la Persistance
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ptc.topology import Topology, build_topology
from ptc.bond import BondResult, omega_e as _omega_e
from ptc.constants import S_HALF, RY
from ptc.data.experimental import SYMBOLS, MASS, IE_NIST
from ptc.geometry import (
    Geometry3D,
    compute_geometry_3d,
    bond_angle_pt,
    bond_length_pt,
    classify_geometry,
    period_of,
    to_xyz,
    to_mol_block,
)
from ptc.frequencies import (
    FrequencyResult as _HessianFreqResult,
    compute_frequencies as _compute_frequencies_hessian,
)


# ====================================================================
# THERMOCHEMISTRY
# ====================================================================

_KB_EV = 8.617333e-5     # Boltzmann constant (eV/K)
_HBAR_EV_S = 6.582e-16   # hbar (eV*s)
_C_CM_S = 2.998e10        # speed of light (cm/s)


@dataclass
class ThermoResult:
    """Thermochemical properties at temperature T."""
    T: float          # temperature (K)
    ZPE: float        # zero-point energy (eV)
    H_vib: float      # vibrational enthalpy (eV)
    S_vib: float      # vibrational entropy (eV/K)
    G_vib: float      # vibrational Gibbs energy (eV)
    H_total: float    # total enthalpy (eV)
    S_total: float    # total entropy (eV/K)
    G_total: float    # total Gibbs energy (eV)


def _vib_thermo(freqs_cm: List[float], T: float) -> ThermoResult:
    """Vibrational thermochemistry from harmonic frequencies.

    ZPE = sum(h*nu/2)
    H_vib = ZPE + sum(h*nu / (exp(h*nu/kT) - 1))
    S_vib = sum(x/(exp(x)-1) - ln(1-exp(-x))) * kB
    where x = h*nu / kT.
    """
    kT = _KB_EV * max(T, 1.0)
    zpe = 0.0
    h_vib = 0.0
    s_vib = 0.0

    for freq in freqs_cm:
        if freq <= 0:
            continue
        # h*nu in eV: freq(cm^-1) / 8065.54 (cm^-1/eV)
        hnu = freq / 8065.54
        zpe += hnu / 2.0

        x = hnu / kT
        if x > 100:
            # high-freq limit
            h_vib += hnu / 2.0
            continue
        exp_x = math.exp(x)
        h_vib += hnu / 2.0 + hnu / (exp_x - 1.0)
        if exp_x > 1.0 + 1e-15:
            s_vib += _KB_EV * (x / (exp_x - 1.0) - math.log(1.0 - math.exp(-x)))

    # Translational + rotational (classical, 3/2 kT each for nonlinear)
    h_trans_rot = 3.0 * kT  # 3/2 kT trans + 3/2 kT rot (nonlinear)
    s_trans_rot = 0.0  # would need mass and moment of inertia for exact value

    return ThermoResult(
        T=T,
        ZPE=zpe,
        H_vib=h_vib,
        S_vib=s_vib,
        G_vib=h_vib - T * s_vib,
        H_total=h_vib + h_trans_rot,
        S_total=s_vib + s_trans_rot,
        G_total=(h_vib + h_trans_rot) - T * (s_vib + s_trans_rot),
    )


# ====================================================================
# FREQUENCY CALCULATOR (Module 2: full Hessian diagonalization)
# ====================================================================

# Re-export FrequencyResult from the Hessian module for backward compat
FrequencyResult = _HessianFreqResult


def _compute_frequencies(topo: Topology, bond_results: List[BondResult],
                         geometry: 'Geometry3D' = None) -> FrequencyResult:
    """Compute vibrational frequencies via full Hessian diagonalization.

    Module 2: uses ptc.frequencies for Morse + bend + torsion potential,
    numerical Hessian, mass-weighted diagonalization.

    Falls back to per-bond estimate if Hessian computation fails.
    """
    # Need 3D coordinates for the Hessian
    if geometry is None:
        geometry = compute_geometry_3d(topo, bond_results)

    try:
        return _compute_frequencies_hessian(topo, geometry.coords, bond_results)
    except Exception:
        # Fallback: simple per-bond estimate (legacy)
        return _compute_frequencies_legacy(topo, bond_results)


def _compute_frequencies_legacy(topo: Topology, bond_results: List[BondResult]) -> FrequencyResult:
    """Legacy per-bond frequency estimate (fallback only)."""
    n = topo.n_atoms
    is_linear = all(topo.z_count[i] <= 2 for i in range(n) if topo.z_count[i] > 0)
    n_modes = 3 * n - (5 if is_linear else 6)
    n_modes = max(n_modes, 0)

    freqs = []
    for bi, br in enumerate(bond_results):
        if br is not None and br.omega_e > 0:
            freqs.append(br.omega_e)

    for center in range(n):
        z = topo.z_count[center]
        lp = topo.lp[center]
        if z < 2:
            continue
        Z_c = topo.Z_list[center]
        ie_c = IE_NIST.get(Z_c, RY)
        n_bends = z * (z - 1) // 2
        avg_stretch = sum(freqs) / max(len(freqs), 1) if freqs else 1000.0
        f_bend = 0.45 * (1.0 + 0.1 * lp) * math.sqrt(ie_c / RY)
        for _ in range(n_bends):
            if len(freqs) < n_modes:
                freqs.append(avg_stretch * f_bend)

    for bi, (a, b, bo) in enumerate(topo.bonds):
        if topo.z_count[a] >= 2 and topo.z_count[b] >= 2:
            if len(freqs) < n_modes:
                avg_stretch = sum(freqs) / max(len(freqs), 1) if freqs else 500.0
                freqs.append(avg_stretch * 0.15)

    freqs.sort(reverse=True)
    freqs = freqs[:n_modes]
    zpe = sum(f / 8065.54 / 2.0 for f in freqs if f > 0)

    return FrequencyResult(
        frequencies=sorted(freqs),
        modes=None,
        n_modes=n_modes,
        ZPE=zpe,
    )


# ====================================================================
# REACTION
# ====================================================================

@dataclass
class ReactionResult:
    """Reaction thermochemistry."""
    delta_H: float       # reaction enthalpy (eV)
    delta_H_kJ: float    # in kJ/mol
    reactants: List[str]
    products: List[str]


# ====================================================================
# MOLECULE CLASS
# ====================================================================

class Molecule:
    """Universal PT chemistry simulator.

    All properties computed from s = 1/2. Zero adjustable parameters.

    Usage:
        mol = Molecule("CCO")         # ethanol
        mol = Molecule("O")           # water (SMILES for H2O)
        mol = Molecule("[H]O[H]")     # water (explicit SMILES)

        # Properties
        mol.D_at                # atomization energy (eV)
        mol.formula             # molecular formula
        mol.geometry            # Geometry3D object
        mol.xyz                 # XYZ format string
        mol.frequencies         # FrequencyResult
        mol.thermo(T=298)       # ThermoResult
        mol.visualize()         # 3D view (requires plotly or py3Dmol)
    """

    def __init__(self, smiles_or_formula: str):
        self._input = smiles_or_formula
        self._topo = build_topology(smiles_or_formula)
        self._cascade_result = None
        self._bond_results = None
        self._geometry = None
        self._freq_result = None

    # ── Core: D_at from cascade v5 ──

    def _ensure_cascade(self):
        if self._cascade_result is not None:
            return
        from ptc.cascade import compute_D_at_cascade
        self._cascade_result = compute_D_at_cascade(self._topo)

    def _ensure_bonds(self):
        """Build per-bond BondResult list."""
        if self._bond_results is not None:
            return
        self._ensure_cascade()
        results = []
        for bi, (a, b, bo) in enumerate(self._topo.bonds):
            Z_a, Z_b = self._topo.Z_list[a], self._topo.Z_list[b]
            m_a = MASS.get(Z_a, 2.0 * Z_a)
            m_b = MASS.get(Z_b, 2.0 * Z_b)
            # Approximate per-bond D0 from total D_at
            D0_approx = self._cascade_result.D_at / max(len(self._topo.bonds), 1)
            lp_a = self._topo.lp[a] if a < len(self._topo.lp) else 0
            lp_b = self._topo.lp[b] if b < len(self._topo.lp) else 0
            r_e = bond_length_pt(Z_a, Z_b, bo, lp_A=lp_a, lp_B=lp_b)
            w_e = _omega_e(D0_approx, r_e, m_a, m_b)
            results.append(BondResult(
                D0=D0_approx,
                v_sigma=D0_approx * 0.7,
                v_pi=D0_approx * 0.3 if bo > 1 else 0.0,
                v_ionic=0.0,
                r_e=r_e,
                omega_e=w_e,
            ))
        self._bond_results = results

    @property
    def topology(self) -> Topology:
        return self._topo

    @property
    def D_at(self) -> float:
        """Atomization energy (eV)."""
        self._ensure_cascade()
        return self._cascade_result.D_at

    @property
    def formula(self) -> str:
        """Molecular formula (Hill notation)."""
        return self._topo.formula

    @property
    def bonds(self) -> List[BondResult]:
        """Per-bond results."""
        self._ensure_bonds()
        return self._bond_results

    # ── Geometry ──

    @property
    def geometry(self) -> Geometry3D:
        """3D molecular geometry."""
        if self._geometry is None:
            self._ensure_bonds()
            self._geometry = compute_geometry_3d(self._topo, self._bond_results)
        return self._geometry

    @property
    def xyz(self) -> str:
        """XYZ format string."""
        g = self.geometry
        return to_xyz(g.coords, g.Z_list,
                      f"PTC | {self.formula} | D_at={self.D_at:.3f} eV | 0 params")

    @property
    def mol_block(self) -> str:
        """MOL V2000 format string."""
        g = self.geometry
        return to_mol_block(g.coords, g.Z_list, self._topo.bonds)

    # ── Frequencies ──

    @property
    def frequencies(self) -> FrequencyResult:
        """Vibrational frequencies (cm^-1) from full Hessian diagonalization."""
        if self._freq_result is None:
            self._ensure_bonds()
            self._freq_result = _compute_frequencies(
                self._topo, self._bond_results, self.geometry
            )
        return self._freq_result

    # ── Thermochemistry ──

    def thermo(self, T: float = 298.15) -> ThermoResult:
        """Thermochemical properties at temperature T (K)."""
        return _vib_thermo(self.frequencies.frequencies, T)

    # ── Visualization ──

    def visualize(self, backend: str = "auto"):
        """3D interactive visualization.

        backend: "plotly", "py3dmol", "matplotlib", or "auto" (tries in order).
        Returns the figure/view object.
        """
        from ptc.viz import plot_molecule_3d
        return plot_molecule_3d(self, backend=backend)

    # ── Export ──

    def save_xyz(self, path: str):
        """Save geometry as XYZ file."""
        with open(path, 'w') as f:
            f.write(self.xyz)

    def save_mol(self, path: str):
        """Save geometry as MOL file."""
        with open(path, 'w') as f:
            f.write(self.mol_block)

    # ── Aromaticity (NICS, σ/π split, full signature) ──

    def aromaticity(self, ring_index: int = 0):
        """Aromaticity diagnosis for a ring (default: first detected).

        Returns the NICSResult dataclass:
            .NICS_0 / .NICS_1   : ppm at z=0 / z=1 Å
            .n_aromatic         : signed σ + π electron count
            .R                  : ring radius (Å)
            .f_coh              : T³ composition coherence ∈ [0,1]
            .aromatic           : bool (NICS(0) < −5 ppm + diamagnetic)

        PT-pure Pauling-London formula with signed Hückel rule
        per channel — diamagnetic for 4n+2, paramagnetic for 4n.

        Returns None if the molecule has no ring.
        """
        from ptc.nics import nics_for_ring
        if not self._topo.rings:
            return None
        if ring_index >= len(self._topo.rings):
            raise IndexError(
                f"ring_index {ring_index} out of range "
                f"({len(self._topo.rings)} rings)"
            )
        return nics_for_ring(self._topo, self._topo.rings[ring_index])

    def nics(self, z: float = 0.0, ring_index: int = 0):
        """NICS at probe distance z (Å) above the ring plane.

        z=0 is the ring centroid (NICS(0)).
        z=1 is 1 Å above the ring plane (NICS(1)).
        """
        from ptc.nics import nics_for_ring, _NICS_K, _huckel_sign
        if not self._topo.rings:
            return None
        ring = self._topo.rings[ring_index]
        # Direct formula: avoid building a separate NICSResult per z
        r = nics_for_ring(self._topo, ring, z_probe=z)
        if z == 0.0:
            return r.NICS_0
        return r.NICS_1

    def nics_profile(self, zs=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
                     ring_index: int = 0):
        """NICS(z) profile across multiple probe distances (Å).

        Returns list of (z, NICS_ppm) tuples.
        """
        if not self._topo.rings:
            return []
        from ptc.nics import _NICS_K, _huckel_sign, _f_coh_T3, _ring_radius
        from ptc.signature import _separate_sigma_pi
        ring = self._topo.rings[ring_index]
        Zs = [self._topo.Z_list[a] for a in ring]
        R = _ring_radius(self._topo, ring)
        n_sigma, n_pi = _separate_sigma_pi(self._topo, ring)
        f_coh = _f_coh_T3(Zs)
        n_eff = (_huckel_sign(n_sigma) * n_sigma
                 + _huckel_sign(n_pi) * n_pi)
        out = []
        for z in zs:
            if R > 0 and n_eff != 0:
                v = -_NICS_K * n_eff * f_coh * R * R \
                     / (R * R + z * z) ** 1.5
            else:
                v = 0.0
            out.append((float(z), v))
        return out

    def signature(self, cap_idx: int = None):
        """Full predicted experimental signature (FullSignature).

        Aggregates D_at, geometry, fragmentation channels, σ/π split,
        NICS(z) profile, vibrational modes (Morse-calibrated), IE/EA
        estimates. Useful for generating experimental datasheets.

        Args:
            cap_idx: index of cap atom for capped clusters (e.g. 0
                     for "[U][S]1[S][S]1"). Auto-detected as first
                     non-ring atom if None.
        """
        from ptc.signature import predict_full_signature
        if cap_idx is None and self._topo.rings:
            ring_atoms = set(a for r in self._topo.rings for a in r)
            non_ring = [k for k in range(self._topo.n_atoms)
                        if k not in ring_atoms]
            if non_ring and self._topo.n_atoms > len(ring_atoms):
                cap_idx = non_ring[0]
        return predict_full_signature(self._input, cap_idx=cap_idx)

    # ── String ──

    def __repr__(self) -> str:
        try:
            d = self.D_at
            return f"Molecule('{self._input}', D_at={d:.3f} eV, formula={self.formula})"
        except Exception:
            return f"Molecule('{self._input}', formula={self.formula})"


# ====================================================================
# REACTION (simple enthalpy balance)
# ====================================================================

def reaction(equation: str) -> ReactionResult:
    """Compute reaction enthalpy from SMILES equation.

    Format: "SMILES1 + SMILES2 >> SMILES3 + SMILES4"

    delta_H = sum(D_at_reactants) - sum(D_at_products)
    """
    if ">>" not in equation:
        raise ValueError("Equation must contain '>>' separator")

    lhs, rhs = equation.split(">>")
    reactant_smiles = [s.strip() for s in lhs.split("+")]
    product_smiles = [s.strip() for s in rhs.split("+")]

    D_react = sum(Molecule(s).D_at for s in reactant_smiles)
    D_prod = sum(Molecule(s).D_at for s in product_smiles)

    delta_H = D_react - D_prod  # eV (exothermic if negative)
    delta_H_kJ = delta_H * 96.485  # eV -> kJ/mol

    return ReactionResult(
        delta_H=delta_H,
        delta_H_kJ=delta_H_kJ,
        reactants=reactant_smiles,
        products=product_smiles,
    )
