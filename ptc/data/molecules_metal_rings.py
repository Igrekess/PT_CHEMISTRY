"""
molecules_metal_rings.py — All-metal aromatic rings mini-bench.

Homonuclear / quasi-homonuclear metal rings where aromaticity (σ or π) is
experimentally documented or computationally established. These test the
σ-aromaticity mechanism added to transfer_matrix.py (C9b block) and the
actinide-capping back-donation (5f→σ via R_57).

D_at values are ATOMIZATION energies in eV for the NEUTRAL ring (unless
noted). Many all-metal aromatic clusters are only stable as anions —
gas-phase neutrals are higher-energy references, so comparison should
account for ~1-3 eV shift from the charged species literature data.

Sources used (D_at values):
  - Al₃:   CCSD(T) + KEMS (Sunwoo & Gole 1989, Fischer et al. 1997)
  - Al₄:   KEMS + theoretical (Sunwoo 1989, extrapolation)
  - Si₃/Si₄: CCSD(T) Raghavachari 1986; atomization from JANAF-type refs
  - Ge₃/Ge₄: Lewerenz et al. 1988
  - P₄:    NIST ATcT (tetrahedron); our ring is square (topology differ)
  - Sb₄:   CRC Handbook (tetrahedron value); our ring is square
  - Bi₃:   Smoes, Drowart 1974 (mass spec dissociation ΔH)
  - Hg₃:   Miyoshi et al. 1988 / very weak ~0.5-1 eV
  - S₃/S₄: NIST (S₃ 6.00 eV atomization; S₄ ~8.5 eV)
  - Cu₃:   Knudsen effusion (Morse 1986, D_at ~2.4 eV)
  - Ag₃:   Knudsen 1979, D_at ≈ 2.5 eV

Values left None = no well-documented reference (predictive only).

April 2026 — Théorie de la Persistance
"""

METAL_RINGS = {
    # ── σ-aromatic triangles (N=3, p-block homonuclear) ──
    'Al3':  {'smiles': '[Al]1[Al][Al]1',  'D_at': 4.90, 'source': 'Sunwoo 1989 / Fischer 1997 CCSD(T)'},
    'Ga3':  {'smiles': '[Ga]1[Ga][Ga]1',  'D_at': 3.60, 'source': 'Balasubramanian 1989 theor.'},
    'Si3':  {'smiles': '[Si]1[Si][Si]1',  'D_at': 7.60, 'source': 'Raghavachari 1986 CCSD(T)'},
    'Ge3':  {'smiles': '[Ge]1[Ge][Ge]1',  'D_at': 6.90, 'source': 'Lewerenz 1988'},
    'P3':   {'smiles': '[P]1[P][P]1',     'D_at': None, 'source': 'predictive (P₃ triangular unstable vs P₂+P)'},
    'As3':  {'smiles': '[As]1[As][As]1',  'D_at': None, 'source': 'predictive'},
    'Sb3':  {'smiles': '[Sb]1[Sb][Sb]1',  'D_at': None, 'source': 'predictive'},
    'Bi3':  {'smiles': '[Bi]1[Bi][Bi]1',  'D_at': 4.80, 'source': 'Smoes, Drowart 1974 mass spec'},
    'S3':   {'smiles': '[S]1[S][S]1',     'D_at': 6.00, 'source': 'NIST ATcT'},
    'Se3':  {'smiles': '[Se]1[Se][Se]1',  'D_at': None, 'source': 'predictive'},
    'Cu3':  {'smiles': '[Cu]1[Cu][Cu]1',  'D_at': 2.40, 'source': 'Morse 1986 / Knudsen'},
    'Ag3':  {'smiles': '[Ag]1[Ag][Ag]1',  'D_at': 2.50, 'source': 'Knudsen 1979'},
    'Au3':  {'smiles': '[Au]1[Au][Au]1',  'D_at': 6.10, 'source': 'Bishea, Morse 1989'},

    # ── π-aromatic squares (N=4, p-block homonuclear) ──
    'Al4':  {'smiles': '[Al]1[Al][Al][Al]1',  'D_at': 6.20, 'source': 'Sunwoo 1989 / Li 2001 anion derived'},
    'Ga4':  {'smiles': '[Ga]1[Ga][Ga][Ga]1',  'D_at': None, 'source': 'predictive'},
    'Hg4':  {'smiles': '[Hg]1[Hg][Hg][Hg]1',  'D_at': None, 'source': 'predictive (weak vdW)'},
    'Si4':  {'smiles': '[Si]1[Si][Si][Si]1',  'D_at': 11.5, 'source': 'Raghavachari 1988 (planar isomer)'},
    'Sb4':  {'smiles': '[Sb]1[Sb][Sb][Sb]1',  'D_at': None, 'source': 'predictive (Sb₄ tetra preferred)'},
    'Bi4':  {'smiles': '[Bi]1[Bi][Bi][Bi]1',  'D_at': None, 'source': 'predictive'},
    'Cu4':  {'smiles': '[Cu]1[Cu][Cu][Cu]1',  'D_at': 4.50, 'source': 'Morse 1986 extrapolation'},

    # ── N=5 and N=6 homonuclear metallic ──
    'P5':   {'smiles': '[P]1[P][P][P][P]1',    'D_at': None, 'source': 'predictive'},
    'As5':  {'smiles': '[As]1[As][As][As][As]1',  'D_at': None, 'source': 'predictive'},
    'Si6':  {'smiles': '[Si]1[Si][Si][Si][Si][Si]1', 'D_at': 18.5, 'source': 'Raghavachari (Si₆ hexagonal)'},
    'Ge6':  {'smiles': '[Ge]1[Ge][Ge][Ge][Ge][Ge]1', 'D_at': None, 'source': 'predictive'},

    # ── Actinide / lanthanide-capped Bi₃ (task 2 mechanism) ──
    'U-Bi3':   {'smiles': '[U][Bi]1[Bi][Bi]1',   'D_at': None, 'source': 'novel — the 2026 paper motif'},
    'Th-Bi3':  {'smiles': '[Th][Bi]1[Bi][Bi]1',  'D_at': None, 'source': 'predictive analog'},
    'Np-Bi3':  {'smiles': '[Np][Bi]1[Bi][Bi]1',  'D_at': None, 'source': 'predictive analog'},
}
