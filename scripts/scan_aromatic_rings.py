"""
scan_aromatic_rings.py — predictive PT scan with signed Hückel NICS
index for aromatic ring discovery.

- Uses predict_full_signature for full per-channel σ/π split
- Scores only diamagnetic candidates (max(0, -NICS))
- Antiaromatic and mixed σ-aro/π-anti get score 0 → drop from top
- Candidate classes:
    * Homonuclear triangles and squares
    * Heteronuclear B-A-B bridges
    * Actinide / lanthanide-capped X₃
    * Group 13 / Group 14 cap on chalcogenide ring

Output: ranked list of diamagnetic candidates, with class
breakdown by σ⊕π topology (single, double, capped, etc).
"""

import warnings; warnings.filterwarnings("ignore")
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from ptc.signature import predict_full_signature
from ptc.smiles_parser import _SYM


# Element groups
S1 = [3, 11, 19, 29, 47, 79]                          # Li Na K Cu Ag Au
G13 = [5, 13, 31, 49, 81]                             # B Al Ga In Tl
G14 = [6, 14, 32, 50, 82]                             # C Si Ge Sn Pb
G15 = [15, 33, 51, 83]                                # P As Sb Bi
G16 = [16, 34, 52, 84]                                # S Se Te Po
F_BLOCK = [57, 58, 59, 60, 90, 91, 92, 93, 94, 95]   # La Ce Pr Nd Th Pa U Np Pu Am


@dataclass
class Cand:
    name: str
    smiles: str
    n_atoms: int
    n_sigma: int = 0
    n_pi: int = 0
    NICS_0: float = 0.0
    NICS_1: float = 0.0
    R: float = 0.0
    D_per_atom: float = 0.0
    f_coh: float = 0.0
    omega_breath: float = 0.0
    IE: float = 0.0
    aromatic_class: str = ""
    A_signed: float = 0.0    # only diamagnetic counts
    error: Optional[str] = None


def _ring_smiles(Zs: Tuple[int, ...]) -> str:
    if len(Zs) < 3:
        return ""
    syms = [_SYM[z] for z in Zs]
    return f"[{syms[0]}]1" + "".join(f"[{s}]" for s in syms[1:]) + "1"


def _capped_smiles(Z_cap: int, Zs_ring: Tuple[int, ...]) -> str:
    cap = _SYM[Z_cap]
    base = _ring_smiles(Zs_ring)
    if not base:
        return ""
    return f"[{cap}]" + base


def evaluate(name: str, smi: str, cap_idx: Optional[int] = None) -> Cand:
    c = Cand(name=name, smiles=smi, n_atoms=0)
    try:
        sig = predict_full_signature(smi, cap_idx=cap_idx)
        c.n_atoms = sig.n_atoms
        c.n_sigma = sig.n_aromatic_sigma
        c.n_pi = sig.n_aromatic_pi
        c.NICS_0 = sig.NICS_profile[0][1] if sig.NICS_profile else 0.0
        c.NICS_1 = sig.NICS_profile[2][1] if len(sig.NICS_profile) > 2 else 0.0
        c.R = sig.ring_radius
        c.D_per_atom = sig.D_per_atom
        c.f_coh = sig.f_coh
        c.omega_breath = sig.ir_breathing_cm1
        c.IE = sig.IE_PT_eV
        c.aromatic_class = sig.aromatic_class
        # Signed aromatic index — only diamagnetic counts
        c.A_signed = max(0.0, -c.NICS_0) * c.f_coh * c.D_per_atom
    except Exception as e:
        c.error = f"{type(e).__name__}: {str(e)[:60]}"
    return c


def scan():
    cands: List[Cand] = []

    # ── Class 1: homonuclear triangles (re-checked with signed NICS) ──
    for Z in S1 + G13 + G14 + G15 + G16:
        comp = (Z, Z, Z)
        smi = _ring_smiles(comp)
        cands.append(evaluate(f"{_SYM[Z]}3", smi))

    # ── Class 2: homonuclear squares ──
    for Z in S1 + G13 + G14 + G15 + G16:
        comp = (Z,) * 4
        smi = _ring_smiles(comp)
        cands.append(evaluate(f"{_SYM[Z]}4", smi))

    # ── Class 3: B-A-B triangles (heteronuclear bridge) ──
    p_block = G13 + G14 + G15 + G16
    for ZA in p_block:
        for ZB in p_block:
            if ZA == ZB:
                continue
            # B-A-B pattern
            comp = (ZB, ZA, ZB)
            smi = _ring_smiles(comp)
            cands.append(evaluate(f"{_SYM[ZB]}{_SYM[ZA]}{_SYM[ZB]}", smi))

    # ── Class 4: M @ X3 with f-block cap (actinide/lanthanide) ──
    for X in G14 + G15 + G16:
        for M in F_BLOCK:
            comp = (X, X, X)
            smi = _capped_smiles(M, comp)
            if smi:
                cands.append(evaluate(f"{_SYM[M]}@{_SYM[X]}3", smi, cap_idx=0))

    # ── Class 5: Group 14 cap on Group 16 ring ──
    # cap (Group 14, σ=2) + ring (Group 16, σ=4 + π=4)
    for cap_z in G14:
        for ring_z in G16:
            comp = (ring_z, ring_z, ring_z)
            smi = _capped_smiles(cap_z, comp)
            if smi:
                cands.append(evaluate(f"{_SYM[cap_z]}@{_SYM[ring_z]}3", smi, cap_idx=0))

    # ── Class 6: Group 13 cap on Group 16 ring ──
    for cap_z in G13:
        for ring_z in G16:
            comp = (ring_z, ring_z, ring_z)
            smi = _capped_smiles(cap_z, comp)
            if smi:
                cands.append(evaluate(f"{_SYM[cap_z]}@{_SYM[ring_z]}3", smi, cap_idx=0))

    # filter errors and zero-aromatic
    cands = [c for c in cands if c.error is None]
    return cands


def format_top(label: str, items: List[Cand], n: int = 15) -> None:
    print(f"\n── {label} ──")
    print(f"{'name':<12}{'D/at':>6}{'σ':>3}{'π':>3}{'NICS0':>7}{'NICS1':>7}"
          f"{'ratio':>6}{'ω':>5}{'IE':>5}{'A':>7}  class")
    for c in items[:n]:
        ratio = c.NICS_0 / c.NICS_1 if c.NICS_1 != 0 else 0
        print(f"{c.name:<12}{c.D_per_atom:>6.2f}{c.n_sigma:>3d}{c.n_pi:>3d}"
              f"{c.NICS_0:>+7.1f}{c.NICS_1:>+7.1f}{ratio:>6.2f}"
              f"{c.omega_breath:>5.0f}{c.IE:>5.1f}{c.A_signed:>7.2f}  {c.aromatic_class}")


def main():
    print("PT predictive scan — signed Hückel aromatic index")
    print("=" * 80)
    cands = scan()
    print(f"\nTotal candidates evaluated: {len(cands)}")

    # filter to truly diamagnetic (NICS_0 < -3 ppm threshold)
    diamag = [c for c in cands if c.NICS_0 < -3 and c.A_signed > 0]
    print(f"Strongly diamagnetic (NICS(0) < −3 ppm): {len(diamag)}")
    diamag.sort(key=lambda c: c.A_signed, reverse=True)

    # Top by aromatic index
    format_top("TOP-15 BY SIGNED AROMATIC INDEX (all classes)", diamag, 15)

    # Per-class breakdowns
    homo3 = [c for c in diamag if c.n_atoms == 3 and len(set(c.smiles)) <= 5]
    homo3.sort(key=lambda c: c.A_signed, reverse=True)
    format_top("homonuclear triangles X₃", homo3, 8)

    capped = [c for c in diamag if "@" in c.name]
    capped.sort(key=lambda c: c.A_signed, reverse=True)
    format_top("capped triangles M@X₃ (any cap)", capped, 12)

    # Filter for double aromatic (both 4n+2)
    double = [c for c in diamag
              if (c.n_sigma % 4 == 2 or c.n_sigma == 0)
              and (c.n_pi % 4 == 2 or c.n_pi == 0)
              and (c.n_sigma + c.n_pi) > 0]
    double.sort(key=lambda c: c.A_signed, reverse=True)
    format_top("double aromatic (σ AND π both 4n+2 or empty)", double, 15)

    # G13/G14 cap on chalcogenide ring
    new_capped = [c for c in diamag
                  if "@" in c.name and c.name.split("@")[0] in
                  [_SYM[z] for z in G13 + G14]]
    new_capped.sort(key=lambda c: c.A_signed, reverse=True)
    format_top("G13/G14 cap on chalcogenide ring", new_capped, 10)

    # Save top diamag list to a json/markdown summary
    print("\n" + "=" * 80)
    print("SUMMARY: candidates with strongest signed aromatic signature.")
    print(f"All NICS values from PT signed-Hückel formula. n_σ ≡ in-plane,")
    print(f"n_π ≡ perpendicular. 4n+2 in BOTH = double aromatic (best).")


if __name__ == "__main__":
    main()
