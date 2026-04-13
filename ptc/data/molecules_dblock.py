"""
molecules_dblock.py — Transition metal molecular benchmark.

D_at = atomization energy at 0K (eV).
Computed from NIST-JANAF / CRC / NIST WebBook formation enthalpies:
  D_at = sum(DeltaHf_atoms_gas) - DeltaHf_molecule_gas

Atomic DeltaHf at 298K (kJ/mol) from NIST:
  Sc=377.8 Ti=473.0 V=514.2 Cr=397.5 Mn=283.3
  Fe=416.3 Co=426.7 Ni=430.1 Cu=337.6 Zn=130.4
  H=218.0 C=716.7 N=472.7 O=249.2
  F=79.38 Cl=121.3 Br=117.9

Conversion: 1 eV = 96.485 kJ/mol

Sources:
  JANAF = NIST-JANAF Thermochemical Tables (4th ed., Chase 1998)
  NIST  = NIST Chemistry WebBook (webbook.nist.gov), verified 2026-04
  CRC   = CRC Handbook of Chemistry and Physics (97th ed.)
  HH    = Huber & Herzberg, Constants of Diatomic Molecules (1979)
  3dMLBE = Xu et al. JCTC 2015; Cheng et al. JCTC 2017 (revised expt)
  LIT   = Research literature (see inline references)

Diatomic D0 values: from spectroscopic measurements compiled in
Huber-Herzberg, CRC 97th ed., and the 3dMLBE20 benchmark dataset.
Uncertainty typically ±5-15 kJ/mol for diatomics, ±3-8 kJ/mol for
JANAF polyatomics.

Polyatomic D_at: computed as D_at = Sigma(DeltaHf_atoms) - DeltaHf_mol.
All molecular DeltaHf(g,298K) verified against NIST WebBook where available.
"""

MOLECULES_DBLOCK = {
    # ══════════════════════════════════════════════════
    # TIER 1: DIATOMICS — spectroscopic D0 data
    # ══════════════════════════════════════════════════
    # D0 values from Huber-Herzberg (HH), CRC 97th ed,
    # and 3dMLBE20 benchmark (revised experimental).
    # For diatomics, D_at = D0 (single bond to break).

    # ── Scandium ──
    'ScO':   {'smiles': '[Sc]=O',    'D_at': 7.01, 'source': 'CRC'},    # D0=676.2±6.3 kJ/mol
    'ScF':   {'smiles': '[Sc]F',     'D_at': 5.99, 'source': 'CRC'},    # D0=578.0±13 kJ/mol
    'ScCl':  {'smiles': '[Sc]Cl',    'D_at': 4.62, 'source': 'CRC'},    # D0=445.8 kJ/mol
    'ScH':   {'smiles': '[Sc][H]',   'D_at': 2.07, 'source': '3dMLBE'}, # D0=200±7 kJ/mol (3dMLBE20)

    # ── Titanium ──
    'TiO':   {'smiles': '[Ti]=O',    'D_at': 6.92, 'source': 'JANAF'},  # D0=667.8±5.4 kJ/mol
    'TiF':   {'smiles': '[Ti]F',     'D_at': 5.84, 'source': 'JANAF'},  # D0=563.6±8 kJ/mol
    'TiCl':  {'smiles': '[Ti]Cl',    'D_at': 4.44, 'source': 'JANAF'},  # D0=428.4±8 kJ/mol

    # ── Vanadium ──
    'VO':    {'smiles': '[V]=O',     'D_at': 6.44, 'source': 'JANAF'},  # D0=621.3±9 kJ/mol
    'VF':    {'smiles': '[V]F',      'D_at': 5.45, 'source': 'CRC'},    # D0=526±8 kJ/mol
    'VCl':   {'smiles': '[V]Cl',     'D_at': 4.49, 'source': 'CRC'},    # D0=433.0 kJ/mol
    'VH':    {'smiles': '[V][H]',    'D_at': 2.31, 'source': '3dMLBE'}, # D0=223±7 kJ/mol (revised)

    # ── Chromium ──
    'CrH':   {'smiles': '[Cr][H]',   'D_at': 2.11, 'source': '3dMLBE'}, # D0=204±7 kJ/mol (revised)
    'CrF':   {'smiles': '[Cr]F',     'D_at': 4.56, 'source': 'CRC'},    # D0=440±6 kJ/mol
    'CrCl':  {'smiles': '[Cr]Cl',    'D_at': 3.78, 'source': 'CRC'},    # D0=364.8 kJ/mol
    'CrO':   {'smiles': '[Cr]=O',    'D_at': 4.65, 'source': 'CRC'},    # D0=448.5 kJ/mol

    # ── Manganese ──
    'MnH':   {'smiles': '[Mn][H]',   'D_at': 2.51, 'source': 'CRC'},   # D0=242±13 kJ/mol
    'MnF':   {'smiles': '[Mn]F',     'D_at': 4.28, 'source': 'CRC'},    # D0=413±13 kJ/mol
    'MnCl':  {'smiles': '[Mn]Cl',    'D_at': 3.53, 'source': 'CRC'},    # D0=340.6 kJ/mol
    'MnO':   {'smiles': '[Mn]=O',    'D_at': 3.83, 'source': 'CRC'},    # D0=369.6 kJ/mol

    # ── Iron ──
    'FeH':   {'smiles': '[Fe][H]',   'D_at': 1.98, 'source': '3dMLBE'}, # D0=191.2±9 kJ/mol
    'FeF':   {'smiles': '[Fe]F',     'D_at': 4.51, 'source': 'CRC'},    # D0=435.1 kJ/mol
    'FeCl':  {'smiles': '[Fe]Cl',    'D_at': 3.52, 'source': 'CRC'},    # D0=339.7 kJ/mol
    'FeO':   {'smiles': '[Fe]=O',    'D_at': 4.20, 'source': 'CRC'},    # D0=405.2±6 kJ/mol

    # ── Cobalt ──
    'CoH':   {'smiles': '[Co][H]',   'D_at': 2.20, 'source': '3dMLBE'}, # D0=212.3±9 kJ/mol
    'CoF':   {'smiles': '[Co]F',     'D_at': 4.45, 'source': 'CRC'},    # D0=429.3 kJ/mol
    'CoCl':  {'smiles': '[Co]Cl',    'D_at': 3.51, 'source': 'CRC'},    # D0=338.5 kJ/mol
    'CoO':   {'smiles': '[Co]=O',    'D_at': 3.94, 'source': 'CRC'},    # D0=380.3±6 kJ/mol

    # ── Nickel ──
    'NiH':   {'smiles': '[Ni][H]',   'D_at': 2.50, 'source': '3dMLBE'}, # D0=240.9±8 kJ/mol
    'NiF':   {'smiles': '[Ni]F',     'D_at': 4.36, 'source': 'CRC'},    # D0=420.7 kJ/mol
    'NiCl':  {'smiles': '[Ni]Cl',    'D_at': 3.63, 'source': 'CRC'},    # D0=350.2 kJ/mol
    'NiO':   {'smiles': '[Ni]=O',    'D_at': 3.66, 'source': 'CRC'},    # D0=353±5 kJ/mol

    # ── Copper ──
    'CuH':   {'smiles': '[Cu][H]',   'D_at': 2.85, 'source': 'HH'},    # D0=275.2±13 kJ/mol
    'CuF':   {'smiles': '[Cu]F',     'D_at': 4.41, 'source': 'HH'},    # D0=425.7 kJ/mol
    'CuCl':  {'smiles': 'Cl[Cu]',    'D_at': 3.75, 'source': 'HH'},    # D0=361.7±10 kJ/mol
    'CuBr':  {'smiles': 'Br[Cu]',    'D_at': 3.36, 'source': 'CRC'},    # D0=324.3 kJ/mol
    'CuO':   {'smiles': '[Cu]=O',    'D_at': 2.87, 'source': 'CRC'},    # D0=277±10 kJ/mol

    # ── Zinc ──
    'ZnH':   {'smiles': '[Zn][H]',   'D_at': 0.88, 'source': 'CRC'},   # D0=85.0 kJ/mol
    'ZnF':   {'smiles': '[Zn]F',     'D_at': 3.70, 'source': 'CRC'},    # D0=357.0 kJ/mol
    'ZnCl':  {'smiles': '[Zn]Cl',    'D_at': 2.29, 'source': 'CRC'},    # D0=221.0 kJ/mol
    'ZnO':   {'smiles': '[Zn]=O',    'D_at': 1.65, 'source': 'CRC'},    # D0=159.4±4 kJ/mol

    # ══════════════════════════════════════════════════
    # TIER 2: POLYATOMICS — JANAF/NIST formation enthalpies
    # D_at = Sigma(DeltaHf_atoms) - DeltaHf_mol(g)
    # All DeltaHf verified against NIST WebBook 2026-04
    # ══════════════════════════════════════════════════

    # ── Titanium compounds ──
    # TiCl4: DeltaHf(g)=-763.2 kJ/mol [NIST confirmed]
    # D_at = 473.0 + 4*121.3 + 763.2 = 1721.4 kJ/mol = 17.84 eV
    'TiCl4': {'smiles': '[Ti](Cl)(Cl)(Cl)Cl', 'D_at': 17.84, 'source': 'JANAF'},
    # TiF4: DeltaHf(g)=-1551.4 kJ/mol [NIST confirmed]
    # D_at = 473.0 + 4*79.38 + 1551.4 = 2341.9 kJ/mol = 24.27 eV
    'TiF4':  {'smiles': '[Ti](F)(F)(F)F',     'D_at': 24.27, 'source': 'JANAF'},
    # TiCl2: DeltaHf(g)=-237.2 kJ/mol [JANAF]
    # D_at = 473.0 + 2*121.3 + 237.2 = 952.8 kJ/mol = 9.88 eV
    'TiCl2': {'smiles': 'Cl[Ti]Cl',           'D_at':  9.88, 'source': 'JANAF'},
    # TiO2: DeltaHf(g)=-305.4 kJ/mol [NIST confirmed]
    # D_at = 473.0 + 2*249.2 + 305.4 = 1276.8 kJ/mol = 13.23 eV
    'TiO2':  {'smiles': 'O=[Ti]=O',           'D_at': 13.23, 'source': 'JANAF'},

    # ── Vanadium compounds ──
    # VF5: DeltaHf(g)=-1432.6 kJ/mol [JANAF]
    # D_at = 514.2 + 5*79.38 + 1432.6 = 2343.7 kJ/mol = 24.29 eV
    'VF5':   {'smiles': '[V](F)(F)(F)(F)F',   'D_at': 24.29, 'source': 'JANAF'},
    # VCl4: DeltaHf(g)=-525.5 kJ/mol [JANAF]
    # D_at = 514.2 + 4*121.3 + 525.5 = 1525.0 kJ/mol = 15.80 eV
    'VCl4':  {'smiles': '[V](Cl)(Cl)(Cl)Cl',  'D_at': 15.80, 'source': 'JANAF'},

    # ── Chromium compounds ──
    # CrO3: DeltaHf(g)=-292.9 kJ/mol [NIST confirmed, Chase 1998]
    # D_at = 397.5 + 3*249.2 + 292.9 = 1437.9 kJ/mol = 14.90 eV
    'CrO3':  {'smiles': 'O=[Cr](=O)=O',       'D_at': 14.90, 'source': 'JANAF'},
    # CrCl2: DeltaHf(g)=-266.1 kJ/mol [JANAF]
    # D_at = 397.5 + 2*121.3 + 266.1 = 906.2 kJ/mol = 9.39 eV
    'CrCl2': {'smiles': 'Cl[Cr]Cl',           'D_at':  9.39, 'source': 'JANAF'},
    # CrF2: DeltaHf(g)=-540.0 kJ/mol [JANAF]
    # D_at = 397.5 + 2*79.38 + 540.0 = 1096.3 kJ/mol = 11.36 eV
    'CrF2':  {'smiles': 'F[Cr]F',             'D_at': 11.36, 'source': 'JANAF'},

    # ── Manganese compounds ──
    # MnCl2: DeltaHf(g)=-310.9 kJ/mol [JANAF]
    # D_at = 283.3 + 2*121.3 + 310.9 = 836.8 kJ/mol = 8.67 eV
    'MnCl2': {'smiles': 'Cl[Mn]Cl',           'D_at':  8.67, 'source': 'JANAF'},
    # MnF2: DeltaHf(g)=-630.0 kJ/mol [JANAF]
    # D_at = 283.3 + 2*79.38 + 630.0 = 1072.1 kJ/mol = 11.11 eV
    'MnF2':  {'smiles': 'F[Mn]F',             'D_at': 11.11, 'source': 'JANAF'},

    # ── Iron compounds ──
    # FeCl2: DeltaHf(g)=-141.0 kJ/mol [NIST confirmed, Chase 1998]
    # D_at = 416.3 + 2*121.3 + 141.0 = 799.9 kJ/mol = 8.29 eV
    'FeCl2': {'smiles': 'Cl[Fe]Cl',           'D_at':  8.29, 'source': 'JANAF'},
    # FeCl3: DeltaHf(g)=-253.1 kJ/mol [NIST confirmed, Chase 1998]
    # D_at = 416.3 + 3*121.3 + 253.1 = 1033.3 kJ/mol = 10.71 eV
    'FeCl3': {'smiles': 'Cl[Fe](Cl)Cl',       'D_at': 10.71, 'source': 'JANAF'},
    # FeF2: DeltaHf(g)=-389.5 kJ/mol [JANAF]
    # D_at = 416.3 + 2*79.38 + 389.5 = 964.6 kJ/mol = 10.00 eV
    'FeF2':  {'smiles': 'F[Fe]F',             'D_at': 10.00, 'source': 'JANAF'},

    # ── Cobalt compounds ──
    # CoCl2: DeltaHf(g)=-93.7 kJ/mol [NIST confirmed, Chase 1998]
    # D_at = 426.7 + 2*121.3 + 93.7 = 763.0 kJ/mol = 7.91 eV
    'CoCl2': {'smiles': 'Cl[Co]Cl',           'D_at':  7.91, 'source': 'JANAF'},

    # ── Nickel compounds ──
    # NiCl2: DeltaHf(g)=-73.9 kJ/mol [NIST confirmed, Chase 1998]
    # D_at = 430.1 + 2*121.3 + 73.9 = 746.6 kJ/mol = 7.74 eV
    'NiCl2': {'smiles': 'Cl[Ni]Cl',           'D_at':  7.74, 'source': 'JANAF'},
    # NiF2: DeltaHf(g)=-335.6 kJ/mol [JANAF]
    # D_at = 430.1 + 2*79.38 + 335.6 = 924.5 kJ/mol = 9.58 eV
    'NiF2':  {'smiles': 'F[Ni]F',             'D_at':  9.58, 'source': 'JANAF'},

    # ── Copper compounds ──
    # CuCl2: DeltaHf(g)=-43.5 kJ/mol [JANAF]
    # D_at = 337.6 + 2*121.3 + 43.5 = 623.7 kJ/mol = 6.46 eV
    'CuCl2': {'smiles': 'Cl[Cu]Cl',           'D_at':  6.46, 'source': 'JANAF'},
    # CuF2: DeltaHf(g)=-266.9 kJ/mol [NIST confirmed, Chase 1998]
    # D_at = 337.6 + 2*79.38 + 266.9 = 763.3 kJ/mol = 7.91 eV
    'CuF2':  {'smiles': 'F[Cu]F',             'D_at':  7.91, 'source': 'JANAF'},

    # ── Zinc compounds ──
    # ZnCl2: DeltaHf(g)=-265.7 kJ/mol [JANAF]
    # D_at = 130.4 + 2*121.3 + 265.7 = 638.7 kJ/mol = 6.62 eV
    'ZnCl2': {'smiles': 'Cl[Zn]Cl',           'D_at':  6.62, 'source': 'JANAF'},
    # ZnF2: DeltaHf(g)=-533.9 kJ/mol [JANAF]
    # D_at = 130.4 + 2*79.38 + 533.9 = 823.1 kJ/mol = 8.53 eV
    'ZnF2':  {'smiles': 'F[Zn]F',             'D_at':  8.53, 'source': 'JANAF'},

    # ══════════════════════════════════════════════════
    # TIER 3: ORGANOMETALLICS — NIST formation enthalpies
    # Larger uncertainties (~10-80 kJ/mol)
    # ══════════════════════════════════════════════════

    # Ni(CO)4: DeltaHf(g)=-601.6 kJ/mol [NIST confirmed, Chase 1998]
    # D_at = 430.1 + 4*(716.7+249.2) + 601.6 = 4895.3 kJ/mol = 50.74 eV
    'Ni(CO)4': {'smiles': '[Ni](=C=O)(=C=O)(=C=O)=C=O', 'D_at': 50.74, 'source': 'NIST'},
    # Fe(CO)5: DeltaHf(g)=-727.9 kJ/mol [NIST confirmed, Chase 1998]
    # D_at = 416.3 + 5*(716.7+249.2) + 727.9 = 5973.7 kJ/mol = 61.91 eV
    'Fe(CO)5': {'smiles': '[Fe](=C=O)(=C=O)(=C=O)(=C=O)=C=O', 'D_at': 61.91, 'source': 'NIST'},
    # Cr(CO)6: DeltaHf(g)=-910±80 kJ/mol [NIST, large uncertainty]
    # D_at = 397.5 + 6*(716.7+249.2) + 910.0 = 7102.9 kJ/mol = 73.62 eV
    'Cr(CO)6': {'smiles': '[Cr](=C=O)(=C=O)(=C=O)(=C=O)(=C=O)=C=O', 'D_at': 73.62, 'source': 'NIST'},  # ±0.8 eV
}
