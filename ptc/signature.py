"""
signature.py — Full predicted experimental signature for a candidate
aromatic ring or capped cluster.

Aggregates all PT-derivable observables for one molecule and returns
them in a structured form ready for an experimental datasheet:

  - Geometry        : bond lengths, ring radius, cap height, symmetry
  - Energetics      : D_at, per-atom, cap binding, fragmentation channels
  - Aromaticity     : NICS profile NICS(z), σ vs π split, f_coh, n_aromatic
  - Vibrational     : ring breathing, cap stretch, Morse-calibrated ω
  - Electronic      : IE estimate (Koopmans + Hückel), EA estimate
  - NMR shielding   : ³³S / ¹H / ¹³C δ(ring atoms)

Many quantities require approximations:
  - ω_e from PT k=2D/r² is harmonic (under-estimates exp by Morse factor
    ~3-5×). We compute a Morse calibration factor automatically from
    the homonuclear dimer of the most-represented ring atom and apply
    it to ring-mode frequencies, with the raw value also reported.
  - HOMO/LUMO from the σ + π Hückel cycle modes (analytical), not from
    the engine's spectrum_P1 (which mixes scales).

This is the template fiche for U@S₃ and other novel candidates.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from ptc.topology import build_topology, Topology
from ptc.transfer_matrix import compute_D_at_transfer, _dimer_D_cached
from ptc.atom import IE_eV, EA_eV
from ptc.bond import omega_e
from ptc.constants import ALPHA_PHYS, A_BOHR, P1, P2, P3, S3, S5, S7
from ptc.periodic import period, l_of, ns_config, _np_of
from ptc.smiles_parser import _SYM
from ptc.data.experimental import MASS
from ptc.nics import nics_for_ring, _aromatic_electron_count, _f_coh_T3, _ring_radius

_TWO_PI = 2.0 * math.pi


# ── Experimental ω_e for select dimers (NIST), used for Morse cal ──
_OMEGA_E_EXP_DIMER = {
    (16, 16): 725.65,    # S₂   ground state ³Σg⁻
    (15, 15): 780.77,    # P₂
    (14, 14): 511.0,     # Si₂  (theoretical, no stable diatomic)
    (8, 8):   1580.19,   # O₂
    (7, 7):   2358.57,   # N₂
    (6, 6):   1854.71,   # C₂
    (32, 32): 286.0,     # Ge₂
    (33, 33): 429.55,    # As₂
    (34, 34): 385.30,    # Se₂
    (51, 51): 269.9,     # Sb₂
    (52, 52): 247.06,    # Te₂
    (83, 83): 172.7,     # Bi₂
    (29, 29): 264.55,    # Cu₂
    (47, 47): 192.0,     # Ag₂
    (79, 79): 191.0,     # Au₂
    (3, 3):   351.43,    # Li₂
    (11, 11): 159.13,    # Na₂
    (13, 13): 350.0,     # Al₂ (theoretical)
}


@dataclass
class FragmentationChannel:
    products: List[str]                # SMILES of each fragment
    delta_E: float                     # ΔE (eV), positive = endothermic
    is_lowest: bool = False
    note: str = ""


@dataclass
class VibSignature:
    omega_raw: float                   # PT raw ω (cm⁻¹)
    omega_calibrated: float            # × Morse factor (cm⁻¹)
    morse_factor: float                # exp/PT ratio for reference dimer
    label: str
    bond_indices: List[int] = field(default_factory=list)


@dataclass
class FullSignature:
    smiles: str
    formula: str
    n_atoms: int
    Z_list: List[int]

    # Geometry
    bond_lengths: List[Tuple[str, float]] = field(default_factory=list)  # (label, r_e)
    ring_radius: float = 0.0
    cap_height: float = 0.0       # for capped systems, cap atom height above ring plane

    # Energetics
    D_at: float = 0.0
    D_per_atom: float = 0.0
    cap_binding: float = 0.0          # capped only: ΔE for M + ring → M·ring
    fragmentation: List[FragmentationChannel] = field(default_factory=list)
    lowest_decomposition_eV: float = 0.0

    # Aromaticity
    n_aromatic_total: int = 0
    n_aromatic_sigma: int = 0
    n_aromatic_pi: int = 0
    f_coh: float = 0.0
    NICS_profile: List[Tuple[float, float]] = field(default_factory=list)  # (z, NICS) Å→ppm
    aromatic_class: str = ""

    # Vibrational
    vib_modes: List[VibSignature] = field(default_factory=list)
    ir_breathing_cm1: float = 0.0     # calibrated ring breathing mode

    # Electronic
    IE_PT_eV: float = 0.0
    EA_PT_eV: float = 0.0
    HL_gap_eV: float = 0.0

    # NMR
    delta_ring_ppm: Dict[str, float] = field(default_factory=dict)

    # Metadata
    notes: List[str] = field(default_factory=list)


# ─── Helpers ────────────────────────────────────────────────────────

def _formula_of(Z_list: List[int]) -> str:
    cnt = {}
    for z in Z_list:
        cnt[z] = cnt.get(z, 0) + 1
    parts = []
    for z in sorted(cnt.keys()):
        n = cnt[z]
        parts.append(f"{_SYM[z]}{n if n > 1 else ''}")
    return "".join(parts)


def _atomic_mass(Z: int) -> float:
    return MASS.get(Z, 2 * Z)


def _morse_factor(Z: int) -> float:
    """Calibration factor exp/PT for the homonuclear dimer of Z.

    Returns 1.0 if no exp data; otherwise ω_exp / ω_PT(dimer).
    """
    if (Z, Z) not in _OMEGA_E_EXP_DIMER:
        return 1.0
    omega_exp = _OMEGA_E_EXP_DIMER[(Z, Z)]
    sym = _SYM[Z]
    try:
        topo = build_topology(f"[{sym}][{sym}]")
        res = compute_D_at_transfer(topo)
        if not res.bonds or res.bonds[0].omega_e <= 0:
            return 1.0
        omega_pt = res.bonds[0].omega_e
        return omega_exp / omega_pt
    except Exception:
        return 1.0


def _separate_sigma_pi(topology: Topology, ring: List[int]) -> Tuple[int, int]:
    """Split aromatic-electron count into σ (in-plane) and π (perpendicular).

    By group:
      G1, G11 (s¹)      → σ:1   π:0  per atom
      G13 (np=1)         → σ:0   π:1
      G14 (np=2, s² LP)  → σ:2   π:0
      G15 (np=3)         → σ:2   π:1
      G16 (np=4)         → σ:2   π:2
      SMILES-aromatic    → σ:0   π:1 (Hückel π)
    """
    n_sigma = 0
    n_pi = 0
    ring_set = set(ring)
    has_smiles_aro = any(abs(bo - 1.5) < 0.01
                         for i, j, bo in topology.bonds
                         if i in ring_set and j in ring_set)
    has_kekule = any(abs(bo - 2.0) < 0.01
                     for i, j, bo in topology.bonds
                     if i in ring_set and j in ring_set)
    if has_kekule:
        for i, j, bo in topology.bonds:
            if i in ring_set and j in ring_set and abs(bo - 2.0) < 0.01:
                n_pi += 2
    for a in ring:
        Z = topology.Z_list[a]
        ns = ns_config(Z)
        np_v = _np_of(Z)
        if has_smiles_aro:
            n_pi += 1
            continue
        if has_kekule and Z == 6:
            continue
        if ns == 1 and np_v == 0:
            n_sigma += 1
            continue
        if l_of(Z) == 1:
            if np_v == 1:
                n_pi += 1
            elif np_v == 2:
                n_sigma += 2
            elif np_v == 3:
                n_sigma += 2
                n_pi += 1
            elif np_v == 4:
                n_sigma += 2
                n_pi += 2
    return n_sigma, n_pi


def _classify_aromatic(n_sigma: int, n_pi: int) -> str:
    """Hückel rule classification for σ and π channels separately."""
    flags = []
    for n, lbl in [(n_sigma, 'σ'), (n_pi, 'π')]:
        if n == 0:
            continue
        if n % 4 == 2:
            flags.append(f"{lbl}-aromatic ({n}e)")
        elif n % 4 == 0:
            flags.append(f"{lbl}-antiaromatic ({n}e)")
        else:
            flags.append(f"{lbl}-radical ({n}e)")
    if not flags:
        return "non-aromatic"
    if len(flags) == 2 and 'aromatic' in flags[0] and 'aromatic' in flags[1] \
            and 'anti' not in flags[0] and 'anti' not in flags[1]:
        return "double aromatic (σ ⊕ π)"
    return ", ".join(flags)


# ─── Fragmentation channel computation ──────────────────────────────

def _compute_fragmentation(topo: Topology, smiles: str, D_total: float,
                           cap_idx: Optional[int] = None
                           ) -> Tuple[List[FragmentationChannel], float, float]:
    """Try a set of physically reasonable fragmentations.

    Returns (channels, lowest_dE, cap_binding_E).
    """
    out: List[FragmentationChannel] = []
    Z_list = topo.Z_list
    sym = [_SYM[z] for z in Z_list]
    cap_binding = 0.0

    def _D(s: str) -> float:
        try:
            t = build_topology(s)
            r = compute_D_at_transfer(t)
            return float(r.D_at)
        except Exception:
            return 0.0

    # Channel 1: total atomization
    out.append(FragmentationChannel(
        products=[f"[{s}]" for s in sym],
        delta_E=D_total, note="total atomization (4 atoms)"))

    # If capped (cap_idx given): cap removal channel
    if cap_idx is not None:
        cap_sym = sym[cap_idx]
        ring_smiles = build_smiles_ring([Z_list[k] for k in range(len(Z_list)) if k != cap_idx])
        D_ring = _D(ring_smiles) if ring_smiles else 0.0
        cap_binding = D_total - D_ring   # since D(cap atom) = 0
        out.append(FragmentationChannel(
            products=[f"[{cap_sym}]", ring_smiles],
            delta_E=cap_binding,
            note="cap removal: M @ X_n → M + X_n"))

    # Channel: lose 1 ring atom
    # (only sensible for homonuclear or near-homonuclear rings)
    ring_atoms = [k for k in range(len(Z_list)) if k != cap_idx] if cap_idx is not None \
                 else list(range(len(Z_list)))
    if len(ring_atoms) >= 3:
        z_ring = [Z_list[k] for k in ring_atoms]
        # most-common ring element
        from collections import Counter
        z_dom = Counter(z_ring).most_common(1)[0][0]
        # Build M-X_{n-1} + X
        if cap_idx is not None:
            cap_sym = sym[cap_idx]
            sub_zs = [z for z in z_ring if z == z_dom][:-1]   # one fewer
            other_zs = [z for z in z_ring if z != z_dom]
            sub_zs += other_zs
            if len(sub_zs) >= 2:
                sub_smiles = build_smiles_ring(sub_zs)
                if sub_smiles:
                    s_capped = f"[{cap_sym}]" + sub_smiles[1:] if len(sub_smiles) > 0 else ""
                    # Simpler: build fragment by removing ring via chain
                    chain = f"[{cap_sym}]" + "".join(f"[{_SYM[z]}]" for z in sub_zs)
                    D_sub = _D(chain)
                    if D_sub > 0:
                        dE = D_total - D_sub - 0.0  # D(atom)=0
                        out.append(FragmentationChannel(
                            products=[chain, f"[{_SYM[z_dom]}]"],
                            delta_E=dE,
                            note=f"lose 1 ring atom: M·X_{len(z_ring)} → M·X_{len(z_ring)-1} + X"))

    # Channel: dimer + atom (for ring of 3)
    if len(ring_atoms) == 3:
        z_ring = [Z_list[k] for k in ring_atoms]
        D_dim = _dimer_D_cached(z_ring[0], z_ring[1])
        dE = D_total - D_dim - 0.0  # leaves dimer + atom (+ optional cap)
        if cap_idx is not None:
            cap_z = Z_list[cap_idx]
            # M + X₂ + X
            dE = D_total - D_dim - 0.0 - 0.0
            out.append(FragmentationChannel(
                products=[f"[{_SYM[cap_z]}]", f"[{_SYM[z_ring[0]]}][{_SYM[z_ring[1]]}]",
                          f"[{_SYM[z_ring[2]]}]"],
                delta_E=dE,
                note="full ring fragmentation: M + X₂ + X"))
        else:
            out.append(FragmentationChannel(
                products=[f"[{_SYM[z_ring[0]]}][{_SYM[z_ring[1]]}]",
                          f"[{_SYM[z_ring[2]]}]"],
                delta_E=dE,
                note="ring → dimer + atom"))

    # Find lowest-energy channel (excluding total atomization)
    feasible = [c for c in out if c.delta_E > 0 and "total" not in c.note]
    if feasible:
        lowest = min(feasible, key=lambda c: c.delta_E)
        lowest.is_lowest = True
        lowest_dE = lowest.delta_E
    else:
        lowest_dE = D_total

    return out, lowest_dE, cap_binding


def build_smiles_ring(Zs: List[int]) -> str:
    if len(Zs) < 3:
        return ""
    syms = [_SYM[z] for z in Zs]
    return f"[{syms[0]}]1" + "".join(f"[{s}]" for s in syms[1:]) + "1"


# ─── Hückel HOMO/LUMO + IE/EA estimate ─────────────────────────────

def _huckel_HOMO_LUMO(N: int, n_e: int, IE_atom: float, beta: float
                      ) -> Tuple[float, float]:
    """Cycle-Hückel HOMO and LUMO orbital energies (eV).

    ε_k = α + 2β cos(2πk/N), with α = -IE_atom (most bonding modes
    sit lowest in energy when β > 0). Aufbau-fill pairs from largest
    cos. HOMO = highest filled ε, LUMO = lowest unfilled ε.
    """
    if N < 2 or beta <= 0:
        return -IE_atom, -IE_atom
    # cosines and energies (ε = α + 2β cos)
    pairs = sorted([(math.cos(_TWO_PI * k / N), k) for k in range(N)],
                   key=lambda p: -p[0])  # descending cos = most bonding first
    energies = [(-IE_atom + 2.0 * beta * c, k) for c, k in pairs]
    # Aufbau in pairs
    rem = n_e
    occupied = []
    unoccupied = []
    for e, k in energies:
        if rem >= 2:
            occupied.append(e); rem -= 2
        elif rem == 1:
            occupied.append(e); rem -= 1
        else:
            unoccupied.append(e)
    if not occupied:
        return -IE_atom, -IE_atom
    HOMO = max(occupied) if all(c > 0 for c, _ in pairs) else occupied[-1]
    # In our sort, occupied are filled first (most bonding) — HOMO is the
    # last one we filled (which has the smallest cos among filled)
    HOMO = occupied[-1]
    LUMO = unoccupied[0] if unoccupied else HOMO
    return HOMO, LUMO


# ─── Main signature predictor ───────────────────────────────────────

def predict_full_signature(smiles: str,
                           cap_idx: Optional[int] = None,
                           η3: bool = False) -> FullSignature:
    """Compute full predicted experimental signature for a SMILES.

    Args:
        smiles  : SMILES string of the candidate
        cap_idx : index (in topology.Z_list) of the cap atom, or None
        η3      : if True for capped systems, augment SMILES topology
                  by adding cap → ring bonds for all ring atoms (η³)

    Returns:
        FullSignature dataclass
    """
    warnings.filterwarnings("ignore")

    # Build topology — use SMILES as-given (η¹ for capped systems);
    # η³ multi-bond model over-estimates cap binding because PTC treats
    # each cap-ring bond as an independent 2c-2e bond. C9b's R₅₇
    # back-donation already accounts for multi-center 5f→σ effect.
    topo = build_topology(smiles)

    res = compute_D_at_transfer(topo)
    sig = FullSignature(
        smiles=smiles,
        formula=_formula_of(topo.Z_list),
        n_atoms=topo.n_atoms,
        Z_list=list(topo.Z_list),
        D_at=float(res.D_at),
        D_per_atom=float(res.D_at) / max(topo.n_atoms, 1),
    )

    # Geometry — bond lengths
    for bi, (i, j, bo) in enumerate(topo.bonds):
        Zi, Zj = topo.Z_list[i], topo.Z_list[j]
        if bi < len(res.bonds) and res.bonds[bi].r_e > 0:
            label = f"{_SYM[Zi]}-{_SYM[Zj]}"
            sig.bond_lengths.append((label, res.bonds[bi].r_e))

    # Fragmentation
    sig.fragmentation, sig.lowest_decomposition_eV, sig.cap_binding = \
        _compute_fragmentation(topo, smiles, sig.D_at, cap_idx)

    # Aromaticity / NICS — pick the ring NOT containing the cap atom
    pure_ring = None
    if topo.rings:
        for r in topo.rings:
            if cap_idx is None or cap_idx not in r:
                pure_ring = r
                break
        if pure_ring is None:
            # fallback: smallest ring (least cap-contamination)
            pure_ring = min(topo.rings, key=len)

    if pure_ring is not None:
        ring = pure_ring
        # Use full ring atoms (cap excluded in ring detection naturally)
        sig.ring_radius = _ring_radius(topo, ring)
        sig.n_aromatic_total = _aromatic_electron_count(topo, ring)
        sig.n_aromatic_sigma, sig.n_aromatic_pi = _separate_sigma_pi(topo, ring)
        Zs_ring = [topo.Z_list[a] for a in ring]
        sig.f_coh = _f_coh_T3(Zs_ring)
        sig.aromatic_class = _classify_aromatic(sig.n_aromatic_sigma, sig.n_aromatic_pi)
        # NICS profile — use signed Hückel sum per channel (handles
        # antiaromatic and mixed σ⊕π cases)
        from ptc.nics import _NICS_K, _huckel_sign
        R = sig.ring_radius
        n_eff = (_huckel_sign(sig.n_aromatic_sigma) * sig.n_aromatic_sigma
                 + _huckel_sign(sig.n_aromatic_pi) * sig.n_aromatic_pi)
        for z in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            if R > 0 and n_eff != 0:
                val = -_NICS_K * n_eff * sig.f_coh * R * R \
                       / (R * R + z * z) ** 1.5
            else:
                val = 0.0
            sig.NICS_profile.append((z, val))

    # Vibrational signatures (Morse-calibrated)
    for bi, (i, j, bo) in enumerate(topo.bonds):
        if bi >= len(res.bonds):
            continue
        b = res.bonds[bi]
        if b.omega_e <= 0:
            continue
        Zi, Zj = topo.Z_list[i], topo.Z_list[j]
        # Morse calibration: average factors of i and j (pairwise dimer ref)
        f_i = _morse_factor(Zi)
        f_j = _morse_factor(Zj)
        morse_f = 0.5 * (f_i + f_j) if (f_i > 0 and f_j > 0) else max(f_i, f_j, 1.0)
        sig.vib_modes.append(VibSignature(
            omega_raw=b.omega_e,
            omega_calibrated=b.omega_e * morse_f,
            morse_factor=morse_f,
            label=f"{_SYM[Zi]}-{_SYM[Zj]}",
            bond_indices=[bi]))

    # Identify ring-breathing mode: average of in-ring symmetric bonds
    if pure_ring is not None:
        ring = pure_ring
        ring_set = set(ring)
        ring_bonds = [vm for vm, (bi, (i, j, _)) in zip(sig.vib_modes, enumerate(topo.bonds))
                      if i in ring_set and j in ring_set]
        if ring_bonds:
            sig.ir_breathing_cm1 = sum(v.omega_calibrated for v in ring_bonds) / len(ring_bonds)

    # Electronic: Hückel HOMO/LUMO + IE/EA estimate
    if pure_ring is not None:
        ring = pure_ring
        ring_set = set(ring)
        Zs_ring = [topo.Z_list[a] for a in ring]
        # Use most common ring element for atomic IE
        from collections import Counter
        z_dom = Counter(Zs_ring).most_common(1)[0][0]
        IE_dom = IE_eV(z_dom)
        EA_dom = EA_eV(z_dom)
        # β from S-S (in-ring) bond's σ component if available
        beta = 0.0
        for bi, (i, j, bo) in enumerate(topo.bonds):
            if i in ring_set and j in ring_set and bi < len(res.bonds):
                beta = max(beta, res.bonds[bi].v_sigma / 2.0)
        if beta > 0 and sig.n_aromatic_sigma > 0:
            HOMO_σ, LUMO_σ = _huckel_HOMO_LUMO(len(ring), sig.n_aromatic_sigma,
                                               IE_dom, beta)
        else:
            HOMO_σ, LUMO_σ = -IE_dom, -IE_dom
        # The cluster IE is approximately -HOMO of σ system (or π if higher)
        IE_PT = -HOMO_σ
        # For capped systems with f-block: cap atom IE often dominant
        if cap_idx is not None:
            cap_Z = topo.Z_list[cap_idx]
            IE_cap = IE_eV(cap_Z)
            IE_PT = min(IE_PT, IE_cap)   # whichever is lower (easier to ionize)
        sig.IE_PT_eV = IE_PT
        sig.HL_gap_eV = max(0.0, LUMO_σ - HOMO_σ)
        # EA estimate: aromatic ring → high LUMO, weak EA. Take the
        # atomic EA as upper bound; multi-center delocalization shifts
        # it to ~half the atomic value (Wade-Mingos type estimate).
        EA_atom_max = EA_dom
        if cap_idx is not None:
            EA_atom_max = max(EA_atom_max, EA_eV(topo.Z_list[cap_idx]))
        sig.EA_PT_eV = max(0.0, 0.5 * EA_atom_max)

    # Cap geometry: estimate cap height from U-S length and ring radius
    if cap_idx is not None and topo.rings:
        ring = topo.rings[0]
        # find a cap-ring bond
        cap_ring_bonds = []
        for bi, (i, j, bo) in enumerate(topo.bonds):
            if (i == cap_idx and j in ring) or (j == cap_idx and i in ring):
                if bi < len(res.bonds) and res.bonds[bi].r_e > 0:
                    cap_ring_bonds.append(res.bonds[bi].r_e)
        if cap_ring_bonds and sig.ring_radius > 0:
            r_MX = sum(cap_ring_bonds) / len(cap_ring_bonds)
            R = sig.ring_radius
            if r_MX > R:
                sig.cap_height = math.sqrt(r_MX * r_MX - R * R)
            else:
                sig.cap_height = 0.0

    return sig


def format_datasheet(sig: FullSignature, name: str = "") -> str:
    """Render a FullSignature as a human-readable markdown datasheet."""
    L = []
    title = name or sig.formula
    L.append(f"# Predicted experimental signature — {title}")
    L.append("")
    L.append(f"**SMILES**: `{sig.smiles}`   **Formula**: {sig.formula}   "
             f"**Atoms**: {sig.n_atoms}")
    L.append("")
    L.append("## Geometry")
    for label, r in sig.bond_lengths:
        L.append(f"- {label}: r_e = **{r:.3f} Å**")
    if sig.ring_radius > 0:
        L.append(f"- Ring radius (centroid → vertex): **{sig.ring_radius:.3f} Å**")
    if sig.cap_height > 0:
        L.append(f"- Cap height above ring plane: **{sig.cap_height:.3f} Å**")
    L.append("")
    L.append("## Energetics")
    L.append(f"- Total atomization D_at = **{sig.D_at:.3f} eV**  "
             f"({sig.D_per_atom:.3f} eV/atom)")
    if sig.cap_binding > 0:
        L.append(f"- Cap binding energy: **{sig.cap_binding:.3f} eV**")
    L.append(f"- Lowest fragmentation barrier: **{sig.lowest_decomposition_eV:.3f} eV**")
    L.append("")
    L.append("**Fragmentation channels (ΔE in eV; lowest = activation energy):**")
    for c in sig.fragmentation:
        marker = " ← *lowest*" if c.is_lowest else ""
        L.append(f"  - ΔE = **{c.delta_E:+.2f}** eV  →  {' + '.join(c.products)}  "
                 f"({c.note}){marker}")
    L.append("")
    L.append("## Aromaticity")
    L.append(f"- Class: **{sig.aromatic_class}**")
    def _huckel_status(n: int) -> str:
        if n == 0:
            return "n/a"
        if n % 4 == 2:
            return f"4n+2 with n={(n-2)//4} ✓ aromatic"
        if n % 4 == 0:
            return f"4n with n={n//4} (anti-aromatic)"
        return f"odd-electron (radical)"
    L.append(f"- σ-aromatic electrons: **{sig.n_aromatic_sigma}** "
             f"({_huckel_status(sig.n_aromatic_sigma)})")
    L.append(f"- π-aromatic electrons: **{sig.n_aromatic_pi}** "
             f"({_huckel_status(sig.n_aromatic_pi)})")
    L.append(f"- Total delocalized: **{sig.n_aromatic_total}**")
    L.append(f"- T³ composition coherence f_coh: **{sig.f_coh:.3f}**")
    L.append("")
    L.append("**NICS(z) profile (PT Pauling-London, in ppm):**")
    L.append("")
    L.append("| z (Å) | NICS (ppm) |")
    L.append("|------:|-----------:|")
    for z, v in sig.NICS_profile:
        L.append(f"| {z:.1f}  | **{v:+.2f}** |")
    L.append("")
    L.append("## Vibrational signature")
    L.append(f"- Raw PT ω uses k = 2D/r² (harmonic). Calibration via Morse "
             f"factor exp/PT on reference dimer.")
    L.append("")
    L.append("| bond | ω_PT raw (cm⁻¹) | Morse factor | ω calibrated (cm⁻¹) |")
    L.append("|------|----------------:|-------------:|--------------------:|")
    for v in sig.vib_modes:
        L.append(f"| {v.label} | {v.omega_raw:.0f} | {v.morse_factor:.2f} | "
                 f"**{v.omega_calibrated:.0f}** |")
    if sig.ir_breathing_cm1 > 0:
        L.append("")
        L.append(f"**Ring-breathing mode (totally symmetric, IR-active in C_nv): "
                 f"≈ {sig.ir_breathing_cm1:.0f} cm⁻¹**")
    L.append("")
    L.append("## Electronic structure (Koopmans + cycle-Hückel)")
    L.append(f"- IE (vertical): **{sig.IE_PT_eV:.2f} eV**")
    L.append(f"- EA (vertical): **{sig.EA_PT_eV:.2f} eV**")
    L.append(f"- HOMO–LUMO gap (cycle-Hückel σ): **{sig.HL_gap_eV:.2f} eV**")
    L.append("")
    if sig.notes:
        L.append("## Notes")
        for n in sig.notes:
            L.append(f"- {n}")
    return "\n".join(L)
