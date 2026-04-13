"""
spectroscopy.py -- UV-Vis / IR / Raman / Fluorescence from PT.

All spectroscopic observables derived from s = 1/2.
Zero adjustable parameters.

UV-Vis:
  HOMO/LUMO gap from atomic IE/EA weighted by composition,
  filtered through PT sin^2 channels.

IR:
  Dipole-active modes: intensity ~ (delta_chi * D_e / r_e)^2.
  Homonuclear bonds are IR-silent (zero dipole derivative).
  Bending modes scaled by S3 * sqrt(S_HALF) from stretch.

Raman:
  Polarizability-active modes: intensity ~ (n_val * a_B / r_e)^2.
  ALL bonds Raman-active (polarizability always changes).

Fluorescence:
  Stokes-shifted emission from UV gap:
  gap_fluor = gap_UV * (1 - S3 * S_HALF).

April 2026 -- Theorie de la Persistance
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from ptc.constants import S_HALF, S3, S5, C3, A_BOHR, RY, P1
from ptc.atom import IE_eV, EA_eV
from ptc.topology import valence_electrons
from ptc.geometry import period_of


# ====================================================================
# RESULT DATACLASSES
# ====================================================================

@dataclass
class UVResult:
    """UV-Vis absorption or fluorescence result.

    Attributes:
        formula:       molecular formula
        homo:          HOMO energy estimate (eV, negative)
        lumo:          LUMO energy estimate (eV, negative or near-zero)
        gap_eV:        HOMO-LUMO gap (eV, positive)
        wavelength_nm: peak absorption/emission wavelength (nm)
    """
    formula: str
    homo: float
    lumo: float
    gap_eV: float
    wavelength_nm: float


@dataclass
class SpectrumPeak:
    """Single peak in an IR or Raman spectrum.

    Attributes:
        frequency:   peak position (cm^-1)
        intensity:   normalized intensity in [0, 1]
        mode:        vibrational mode type ('stretch' or 'bend')
        assignment:  human-readable assignment (e.g. 'C-H stretch')
    """
    frequency: float
    intensity: float
    mode: str
    assignment: str


@dataclass
class SpectrumResult:
    """Collection of spectral peaks for one molecule.

    Attributes:
        formula: molecular formula
        peaks:   list of SpectrumPeak objects, sorted by frequency
        type:    spectrum type ('IR', 'Raman', or 'UV')
    """
    formula: str
    peaks: List[SpectrumPeak]
    type: str


# ====================================================================
# INTERNAL HELPERS
# ====================================================================

def _chi_mulliken(Z: int) -> float:
    """Mulliken electronegativity: (IE + EA) / 2  (eV).

    PT-derived from atom.IE_eV and atom.EA_eV, both from s = 1/2.
    """
    ie = IE_eV(Z)
    ea = EA_eV(Z)
    return (ie + ea) / 2.0


def _symbol(Z: int) -> str:
    """Atomic symbol from Z, with fallback."""
    from ptc.data.experimental import SYMBOLS
    return SYMBOLS.get(Z, f"Z{Z}")


def _build_mol(smiles: str):
    """Build a Molecule and return it (lazy import to avoid circular deps)."""
    from ptc.api import Molecule
    return Molecule(smiles)


# ====================================================================
# UV-VIS ABSORPTION
# ====================================================================

def _classify_transition(topo) -> str:
    """Classify the dominant electronic transition type from topology.

    PT logic: the transition type depends on which CRT channel
    carries the frontier orbitals.

    Priority (lowest energy transition dominates absorption):
      0. spin_forbidden : homonuclear diatomic with triplet ground state (O2, S2)
      1. n -> pi*       : LP + pi bonds (carbonyl, imine) — lowest gap
      2. pi -> pi*      : conjugated/aromatic systems — medium gap
      3. n -> sigma*    : single LP on heteroatom, no pi bonds (NH3)
      4. sigma          : saturated, no LP on heteroatoms

    Note: molecules like N2 (triple bond + LP) and alkanes
    have their lowest transition in the sigma channel.

    Returns: 'spin_forbidden', 'n_pi', 'pi', 'n_sigma', or 'sigma'.
    """
    bonds = topo.bonds
    Z_list = topo.Z_list

    # ── FIX 3: Spin-forbidden (triplet ground state homonuclear diatomics) ──
    # O2 and S2 have triplet ground states; their lowest absorption is
    # a spin-forbidden transition at very long wavelength.
    if topo.n_atoms == 2 and Z_list[0] == Z_list[1] and Z_list[0] in (8, 16):
        return 'spin_forbidden'

    # Detect TRUE pi bonds (bo >= 2, not aromatic delocalized)
    has_double = any(bo >= 1.9 and bo <= 2.1 for _, _, bo in bonds)
    has_aromatic = any(abs(bo - 1.5) < 0.1 for _, _, bo in bonds)
    has_triple = any(bo >= 2.9 for _, _, bo in bonds)

    # LP on heteroatoms (N, O, S)
    has_lp = False
    has_single_lp = False  # at least one heteroatom with exactly 1 LP pair
    if hasattr(topo, 'lp'):
        has_lp = any(topo.lp[i] > 0 for i in range(len(topo.lp))
                     if Z_list[i] in (7, 8, 16))
        has_single_lp = any(topo.lp[i] == 1 for i in range(len(topo.lp))
                            if Z_list[i] in (7, 8, 16))

    # n -> pi* requires BOTH LP and a double bond (not triple — those are sigma-like)
    if has_lp and has_double and not has_triple:
        return 'n_pi'
    elif has_aromatic:
        return 'pi'
    elif has_double:
        return 'pi'
    # ── FIX 1: n -> sigma* for LP-bearing molecules without pi bonds ──
    # Requires a heteroatom with a single LP pair (lp=1, e.g. NH3).
    # Atoms with lp >= 2 (H2O) have LP too deeply bound — stays sigma.
    # Also exclude triple bonds (N2): those are sigma-like despite LP.
    elif has_single_lp and not has_triple:
        return 'n_sigma'
    else:
        # Everything else: sigma (including N2 triple, alkanes)
        return 'sigma'


def uv_spectrum(smiles: str) -> UVResult:
    """Estimate UV-Vis absorption from the HOMO-LUMO gap.

    PT channel-dependent gap (cascade sequentielle):

      spin_forbidden  : gap = Ry * S3 * S5 * P1           [triplet O2/S2]
      sigma -> sigma* : gap = (IE - EA) * cos^2_3          [P1 channel]
      pi -> pi*       : gap = (IE - EA) * f_pi(n_pi)       [P2 channel, pi-filling]
      n -> pi*        : gap = (IE - EA) * (S3 * S5)^(1/4)  [cross P1xP2]
      n -> sigma*     : gap = (IE - EA) * sqrt(S3 * C3)    [LP + no pi]

    The transition type is classified from the molecular topology
    (bond orders, lone pairs, aromaticity).

    Parameters:
        smiles: SMILES string for the molecule.

    Returns:
        UVResult with HOMO, LUMO, gap, and wavelength.
    """
    mol = _build_mol(smiles)
    topo = mol.topology
    Z_list = topo.Z_list
    n_atoms = topo.n_atoms

    # Composition-weighted average IE
    ie_sum = 0.0
    for i in range(n_atoms):
        ie_sum += IE_eV(Z_list[i])
    ie_avg = ie_sum / max(n_atoms, 1)

    # Most electronegative atom's EA
    ea_max = max(EA_eV(Z_list[i]) for i in range(n_atoms))

    homo = -ie_avg

    # Classify transition type from topology
    trans_type = _classify_transition(topo)

    # ── Channel-dependent gap (PT cascade) ──
    #
    # The key insight: the gap factor depends on whether the transition
    # is INTRA-channel or CROSS-channel.
    #
    # Intra-channel (sigma -> sigma*):
    #   Both states live on P1. No cross-channel filter.
    #   gap = (IE - EA) * cos^2_3
    #   cos^2_3 = fraction TRANSMITTED through P1 (complement of absorbed).
    #
    # Intra-channel (pi -> pi*):
    #   Both states live on P2. No cross-channel filter.
    #   gap = (IE - EA) * sqrt(sin^2_5)
    #   sqrt(S5) because pi orbitals are WEAKER than sigma (P2 < P1).
    #
    # Cross-channel (n -> pi*):
    #   LP on P1, pi* on P2. Must cross channels.
    #   gap = (IE - EA) * (S3 * S5)^(1/4)
    #   Geometric mean of the two channel filters.
    #
    ie_ea_diff = ie_avg - max(ea_max, 0.0)

    # ── For LP-based transitions (n→σ*, n→π*): use heteroatom EA ──
    # The LUMO involves the heteroatom, not H. Using max(EA) would pick
    # EA(H)=0.75 which is irrelevant for the n→σ* antibonding orbital.
    if trans_type in ('n_sigma', 'n_pi'):
        hetero_ea = 0.0
        for i in range(n_atoms):
            if Z_list[i] in (7, 8, 16):
                hetero_ea = max(hetero_ea, EA_eV(Z_list[i]))
        ie_ea_diff = ie_avg - hetero_ea

    if trans_type == 'spin_forbidden':
        # ── FIX 3: Spin-forbidden gap (triplet ground state) ──
        # O2/S2: gap = Ry × S3 × S5 × P1 (very small gap, long wavelength)
        gap = RY * S3 * S5 * P1
    elif trans_type == 'sigma':
        # Intra-P1: cos²₃ = transmitted fraction through P₁ valence shell.
        # LP screening (Principle 4): lone pairs add an EXTRA screening pass.
        # σ→σ* with LP = round-trip through valence + LP shell:
        # gap = (IE-EA) × C₃ × (1 - n_LP × sin²₃ / per_max)
        # When n_LP/per = 1 (H₂O, N₂): reduces to C₃²
        # When n_LP = 0 (methane): stays at C₃
        n_lp_mol = 0
        if hasattr(topo, 'lp'):
            n_lp_mol = sum(topo.lp[i] for i in range(n_atoms)
                          if Z_list[i] in (7, 8, 16))
        per_max = max(period_of(Z_list[i]) for i in range(n_atoms))
        f_lp = S3 * n_lp_mol / max(per_max, 1)
        gap = ie_ea_diff * C3 * (1.0 - min(f_lp, S3))
    elif trans_type == 'pi':
        # ── π→π* gap: bifurcation isolated vs conjugated (Principle 4) ──
        # Count pi electrons from bond topology
        n_pi = 0
        for _, _, bo in topo.bonds:
            if bo >= 1.5:
                n_pi += int(round(2 * (bo - 1)))
        n_pi = max(n_pi, 2)

        from ptc.constants import P0, C5
        if n_pi <= P0:
            # ── Isolated π bond (n_π = P₀ = 2): no conjugation ──
            # Both P₁ and P₂ channels screen independently:
            # gap = (IE-EA) × C₃ × C₅ = complement of BOTH channels.
            # Physical: σ framework transmits (C₃), π system transmits (C₅).
            f_pi = C3 * C5
        else:
            # ── Conjugated/aromatic (n_π > P₀): filling-dependent ──
            # f_pi scales from sqrt(S5) for full conjugation (benzene, n_pi=2P1)
            # to C₃×C₅ for minimal conjugation (n_pi just above P₀).
            f_pi = math.sqrt(S5) * (2 * P1 / n_pi) ** (S_HALF * S3)
            f_pi = min(f_pi, math.sqrt(C3))

        gap = ie_ea_diff * f_pi
    elif trans_type == 'n_pi':
        # Cross P1×P2: inclusion-exclusion probability
        # P(at least one channel) = 1 - P(miss P₁)×P(miss P₂) = 1 - C₃×C₅
        # = S₃ + S₅ - S₃×S₅ = 0.371
        # Physical: the n→π* transition excites through EITHER the LP channel
        # (P₁) OR the π channel (P₂), with double-counting subtracted.
        gap = ie_ea_diff * (S3 + S5 - S3 * S5)
    elif trans_type == 'n_sigma':
        # ── FIX 1: n -> sigma* gap ──
        # LP on heteroatom but no pi bonds: cross P1 channel attenuated
        gap = ie_ea_diff * math.sqrt(S3 * C3)
    else:
        gap = ie_ea_diff * C3

    # ── FIX 4: Rydberg detection (Principle 4, bifurcation) ──
    # When IE - gap_valence > Ry × sin²₃, the lowest transition is Rydberg
    # (promotion to diffuse n=P₁ orbital), not valence.
    # Rydberg gap = IE - Ry / (n - delta)²
    # delta (quantum defect) = per_max × sin²₃ for s-type Rydberg states
    _RYDBERG_THRESHOLD = RY * S3
    if ie_avg - gap > _RYDBERG_THRESHOLD and trans_type == 'sigma':
        per_max = max(period_of(Z_list[i]) for i in range(n_atoms))
        delta_qd = per_max * S3  # quantum defect from P₁ screening
        n_ryd = P1                # first Rydberg series (n=3)
        gap_rydberg = ie_avg - RY / (n_ryd - delta_qd) ** 2
        if gap_rydberg > 0 and gap_rydberg < gap:
            gap = gap_rydberg

    lumo = homo + gap

    gap = max(gap, 0.01)
    wavelength = 1240.0 / gap

    return UVResult(
        formula=mol.formula,
        homo=homo,
        lumo=lumo,
        gap_eV=gap,
        wavelength_nm=wavelength,
    )


# ====================================================================
# IR SPECTRUM
# ====================================================================

def ir_spectrum(smiles: str) -> SpectrumResult:
    """Compute IR absorption spectrum from bond dipole derivatives.

    For each bond (i, j, bo):
      - Stretch frequency from Molecule.bonds[k].omega_e
      - IR intensity = S3 * (delta_chi * D_e / r_e)^2
      - Homonuclear bonds (Z_i == Z_j): IR inactive (intensity = 0)

    For polyatomic centers (coordination >= 2):
      - Bending modes at nu_bend = nu_stretch * S3 * sqrt(S_HALF)
      - Bending intensity = stretch intensity * S3

    All intensities normalized to [0, 1].

    Parameters:
        smiles: SMILES string for the molecule.

    Returns:
        SpectrumResult with type='IR'.
    """
    mol = _build_mol(smiles)
    topo = mol.topology
    bonds_data = topo.bonds          # list of (i, j, bo)
    bond_results = mol.bonds         # list of BondResult
    Z_list = topo.Z_list

    raw_peaks: List[dict] = []

    # Use Hessian frequencies when available (more accurate than per-bond omega_e)
    hessian_freqs = []
    try:
        freq_result = mol.frequencies
        if freq_result and freq_result.frequencies:
            hessian_freqs = sorted([f for f in freq_result.frequencies if f > 50], reverse=True)
    except Exception:
        pass

    for k, ((i, j, bo), br) in enumerate(zip(bonds_data, bond_results)):
        Z_i, Z_j = Z_list[i], Z_list[j]
        sym_i, sym_j = _symbol(Z_i), _symbol(Z_j)
        # Prefer Hessian frequency for this bond (highest remaining)
        if hessian_freqs:
            freq = hessian_freqs.pop(0)
        else:
            freq = br.omega_e
        r_e = br.r_e
        D_e = br.D0

        if freq <= 0 or r_e <= 0:
            continue

        # Electronegativity difference
        delta_chi = abs(_chi_mulliken(Z_i) - _chi_mulliken(Z_j))

        # IR intensity: proportional to (d mu / d r)^2
        # Homonuclear bonds: zero dipole derivative -> IR silent
        if Z_i == Z_j:
            intensity = 0.0
        else:
            intensity = S3 * (delta_chi * D_e / r_e) ** 2

        bo_label = {1: '', 2: '=', 3: '#'}.get(int(round(bo)), '')
        assignment = f"{sym_i}{bo_label}{sym_j} stretch"
        raw_peaks.append({
            'frequency': freq,
            'intensity': intensity,
            'mode': 'stretch',
            'assignment': assignment,
        })

        # Bending modes for polyatomic centers
        z_i = topo.z_count[i] if i < len(topo.z_count) else 1
        z_j = topo.z_count[j] if j < len(topo.z_count) else 1
        if z_i >= 2 or z_j >= 2:
            nu_bend = freq * S3 * math.sqrt(S_HALF)
            bend_intensity = intensity * S3
            center_Z = Z_i if z_i >= 2 else Z_j
            assignment_bend = f"{_symbol(center_Z)} bend ({sym_i}-{sym_j})"
            raw_peaks.append({
                'frequency': nu_bend,
                'intensity': bend_intensity,
                'mode': 'bend',
                'assignment': assignment_bend,
            })

    # Normalize intensities to [0, 1]
    peaks = _normalize_and_build(raw_peaks)

    return SpectrumResult(
        formula=mol.formula,
        peaks=peaks,
        type='IR',
    )


# ====================================================================
# RAMAN SPECTRUM
# ====================================================================

def raman_spectrum(smiles: str) -> SpectrumResult:
    """Compute Raman scattering spectrum from Hessian normal modes.

    Two-step PT approach:
      1. Frequencies from Hessian diagonalization (all normal modes)
      2. Raman intensity from polarizability: ALL modes Raman-active

    Raman intensity per mode (PT, Principe 3 — CRT orthogonal):
      I_Raman = S5 × (n_val × A_BOHR / r_avg)² × f_sym
      f_sym = 1 + sin²₃ for symmetric modes (bo > 1.5 or homonuclear)
      f_sym = 1 for asymmetric modes

    All intensities normalized to [0, 1].

    Parameters:
        smiles: SMILES string for the molecule.

    Returns:
        SpectrumResult with type='Raman'.
    """
    mol = _build_mol(smiles)
    topo = mol.topology
    bonds_data = topo.bonds
    bond_results = mol.bonds
    Z_list = topo.Z_list
    n_atoms = topo.n_atoms

    raw_peaks: List[dict] = []

    # ── Average molecular parameters for Raman intensity ──
    n_val_total = sum(valence_electrons(Z_list[i]) for i in range(n_atoms))
    r_avg = sum(br.r_e for br in bond_results) / max(len(bond_results), 1)
    base_intensity = S5 * (n_val_total * A_BOHR / max(r_avg, 0.5)) ** 2

    # ── Hessian normal mode frequencies ──
    hessian_freqs = []
    try:
        freq_result = mol.frequencies
        if freq_result and freq_result.frequencies:
            hessian_freqs = sorted([f for f in freq_result.frequencies if f > 50])
    except Exception:
        pass

    if hessian_freqs:
        # Use ALL Hessian modes as Raman peaks.
        # Assign intensity based on frequency range (PT heuristic):
        #   > 2500 cm⁻¹: X-H stretch (high polarizability derivative)
        #   1000-2500: heavy-atom stretch (medium)
        #   < 1000: bends/torsions (low)
        for freq in hessian_freqs:
            if freq > 2500:
                intensity = base_intensity * 1.0
                assignment = "X-H stretch"
            elif freq > 1000:
                intensity = base_intensity * (1.0 + S3)  # symmetric stretches
                assignment = "stretch"
            else:
                intensity = base_intensity * S3
                assignment = "bend/torsion"

            raw_peaks.append({
                'frequency': freq,
                'intensity': intensity,
                'mode': 'stretch' if freq > 1000 else 'bend',
                'assignment': assignment,
            })
    else:
        # Fallback: per-bond Morse frequencies
        for k, ((i, j, bo), br) in enumerate(zip(bonds_data, bond_results)):
            Z_i, Z_j = Z_list[i], Z_list[j]
            sym_i, sym_j = _symbol(Z_i), _symbol(Z_j)
            freq = br.omega_e
            r_e = br.r_e
            if freq <= 0 or r_e <= 0:
                continue

            n_val = valence_electrons(Z_i) + valence_electrons(Z_j)
            intensity = S5 * (n_val * A_BOHR / r_e) ** 2

            bo_label = {1: '', 2: '=', 3: '#'}.get(int(round(bo)), '')
            assignment = f"{sym_i}{bo_label}{sym_j} stretch"
            raw_peaks.append({
                'frequency': freq,
                'intensity': intensity,
                'mode': 'stretch',
                'assignment': assignment,
            })

    peaks = _normalize_and_build(raw_peaks)

    return SpectrumResult(
        formula=mol.formula,
        peaks=peaks,
        type='Raman',
    )


# ====================================================================
# FLUORESCENCE
# ====================================================================

def fluorescence(smiles: str) -> UVResult:
    """Estimate fluorescence emission from Stokes-shifted UV gap.

    The Stokes shift arises from vibrational relaxation in the excited
    state before emission. In PT, this relaxation is governed by the
    sin^2_3 channel coupling to spin-1/2 thermal bath:

        gap_fluor = gap_UV * (1 - S3 * S_HALF)

    The emitted photon has lower energy (longer wavelength) than
    the absorbed photon.

    Parameters:
        smiles: SMILES string for the molecule.

    Returns:
        UVResult with Stokes-shifted gap and emission wavelength.
    """
    uv = uv_spectrum(smiles)

    # Stokes shift: excited-state vibrational relaxation
    stokes_factor = 1.0 - S3 * S_HALF
    gap_fluor = uv.gap_eV * stokes_factor
    gap_fluor = max(gap_fluor, 0.01)

    wavelength_fluor = 1240.0 / gap_fluor

    return UVResult(
        formula=uv.formula,
        homo=uv.homo,
        lumo=uv.lumo,
        gap_eV=gap_fluor,
        wavelength_nm=wavelength_fluor,
    )


# ====================================================================
# NORMALIZATION HELPER
# ====================================================================

def _normalize_and_build(raw_peaks: List[dict]) -> List[SpectrumPeak]:
    """Normalize raw intensity values to [0, 1] and build SpectrumPeak list.

    Peaks are sorted by ascending frequency.
    """
    if not raw_peaks:
        return []

    max_intensity = max(p['intensity'] for p in raw_peaks)
    if max_intensity <= 0:
        max_intensity = 1.0  # avoid division by zero (all inactive)

    peaks = []
    for p in raw_peaks:
        norm_intensity = p['intensity'] / max_intensity
        peaks.append(SpectrumPeak(
            frequency=p['frequency'],
            intensity=norm_intensity,
            mode=p['mode'],
            assignment=p['assignment'],
        ))

    peaks.sort(key=lambda pk: pk.frequency)
    return peaks
