"""
molecules.py — Benchmark molecular atomisation energies.

200 molecules from ATcT / NIST / CRC / CCCBDB.  Format:
    MOLECULES[name] = {'smiles': str, 'D_at': float (eV), 'source': str}

    All D_at values in eV.  Source: NIST or ATcT unless noted otherwise.

March 2026 — Theorie de la Persistance
"""

MOLECULES = {
    # ══════════════════════════════════════════════════════════════════
    # DIATOMICS — 97 molecules, ATcT/NIST/Huber-Herzberg
    # All diatomics with measured D₀, Z=1-53
    # ══════════════════════════════════════════════════════════════════
    # ── Homonuclear ──
    'H2':       {'smiles': '[H][H]',       'D_at':  4.478, 'source': 'ATcT'},
    'Li2':      {'smiles': '[Li][Li]',     'D_at':  1.046, 'source': 'ATcT'},
    'B2':       {'smiles': '[B]=[B]',      'D_at':  3.020, 'source': 'HH'},
    'C2':       {'smiles': '[C]#[C]',      'D_at':  6.248, 'source': 'Bao2021'},  # bo=3: 2π+σ (Shaik2012)
    'N2':       {'smiles': 'N#N',          'D_at':  9.759, 'source': 'ATcT'},
    'O2':       {'smiles': 'O=O',          'D_at':  5.116, 'source': 'ATcT'},
    'F2':       {'smiles': 'FF',           'D_at':  1.602, 'source': 'ATcT'},
    'Na2':      {'smiles': '[Na][Na]',     'D_at':  0.735, 'source': 'HH'},
    'Al2':      {'smiles': '[Al]=[Al]',    'D_at':  1.550, 'source': 'HH'},
    'Si2':      {'smiles': '[Si]=[Si]',    'D_at':  3.210, 'source': 'HH'},
    'P2':       {'smiles': '[P]#[P]',      'D_at':  5.033, 'source': 'ATcT'},
    'S2':       {'smiles': '[S]=[S]',      'D_at':  4.369, 'source': 'ATcT'},
    'Cl2':      {'smiles': 'ClCl',         'D_at':  2.510, 'source': 'ATcT'},
    'K2':       {'smiles': '[K][K]',       'D_at':  0.514, 'source': 'HH'},
    'Br2':      {'smiles': 'BrBr',         'D_at':  1.971, 'source': 'ATcT'},
    'I2':       {'smiles': 'II',           'D_at':  1.542, 'source': 'ATcT'},
    # ── Hydrures ──
    'LiH':      {'smiles': '[Li][H]',       'D_at':  2.429, 'source': 'ATcT'},
    'BeH':      {'smiles': '[Be][H]',      'D_at':  2.034, 'source': 'HH'},
    'BH':       {'smiles': '[B][H]',        'D_at':  3.420, 'source': 'HH'},
    'CH':       {'smiles': '[C][H]',       'D_at':  3.465, 'source': 'ATcT'},
    'NH':       {'smiles': '[N][H]',       'D_at':  3.470, 'source': 'ATcT'},
    'OH':       {'smiles': '[O][H]',       'D_at':  4.392, 'source': 'ATcT'},
    'HF':       {'smiles': '[H]F',         'D_at':  5.869, 'source': 'ATcT'},
    'NaH':      {'smiles': '[Na][H]',       'D_at':  1.970, 'source': 'HH'},
    'MgH':      {'smiles': '[Mg][H]',      'D_at':  1.280, 'source': 'HH'},
    'AlH':      {'smiles': '[Al][H]',      'D_at':  2.980, 'source': 'HH'},
    'SiH':      {'smiles': '[Si][H]',      'D_at':  3.060, 'source': 'HH'},
    'PH':       {'smiles': '[P][H]',       'D_at':  2.950, 'source': 'HH'},
    'SH':       {'smiles': '[S][H]',       'D_at':  3.560, 'source': 'ATcT'},
    'HCl':      {'smiles': '[H]Cl',        'D_at':  4.430, 'source': 'ATcT'},
    'KH':       {'smiles': '[K][H]',        'D_at':  1.760, 'source': 'HH'},
    'CaH':      {'smiles': '[Ca][H]',      'D_at':  1.700, 'source': 'HH'},
    'HBr':      {'smiles': '[H]Br',        'D_at':  3.760, 'source': 'ATcT'},
    'HI':       {'smiles': '[H]I',         'D_at':  3.050, 'source': 'ATcT'},
    # ── Fluorures ──
    'LiF':      {'smiles': '[Li]F',        'D_at':  5.970, 'source': 'ATcT'},
    'BeF':      {'smiles': '[Be]F',         'D_at':  5.850, 'source': 'HH'},
    'BF':       {'smiles': '[B]F',         'D_at':  7.810, 'source': 'HH'},
    'CF':       {'smiles': '[C]F',         'D_at':  5.560, 'source': 'HH'},
    'NF':       {'smiles': '[N]F',         'D_at':  3.470, 'source': 'HH'},
    'OF':       {'smiles': '[O]F',         'D_at':  2.240, 'source': 'HH'},
    'NaF':      {'smiles': '[Na]F',        'D_at':  4.910, 'source': 'ATcT'},
    'MgF':      {'smiles': '[Mg]F',         'D_at':  4.590, 'source': 'HH'},
    'AlF':      {'smiles': '[Al]F',        'D_at':  6.880, 'source': 'HH'},
    'SiF':      {'smiles': '[Si]F',        'D_at':  5.570, 'source': 'HH'},
    'PF':       {'smiles': '[P]F',         'D_at':  4.470, 'source': 'HH'},
    'SF':       {'smiles': '[S]F',         'D_at':  3.360, 'source': 'HH'},
    'ClF':      {'smiles': 'FCl',          'D_at':  2.620, 'source': 'ATcT'},
    'KF':       {'smiles': '[K]F',         'D_at':  5.144, 'source': 'ATcT'},
    'CaF':      {'smiles': '[Ca]F',         'D_at':  5.290, 'source': 'HH'},
    'BrF':      {'smiles': 'FBr',          'D_at':  2.550, 'source': 'ATcT'},
    'IF':       {'smiles': 'FI',           'D_at':  2.880, 'source': 'HH'},
    # ── Chlorures ──
    'LiCl':     {'smiles': '[Li]Cl',       'D_at':  4.900, 'source': 'ATcT'},
    'BeCl':     {'smiles': '[Be]Cl',        'D_at':  3.850, 'source': 'HH'},
    'BCl':      {'smiles': '[B]Cl',        'D_at':  5.170, 'source': 'HH'},
    'CCl':      {'smiles': '[C]Cl',        'D_at':  3.960, 'source': 'HH'},
    'NCl':      {'smiles': '[N]Cl',        'D_at':  2.750, 'source': 'Peterson1997'},
    'OCl':      {'smiles': '[O]Cl',        'D_at':  2.750, 'source': 'HH'},
    'NaCl':     {'smiles': '[Na]Cl',       'D_at':  4.230, 'source': 'ATcT'},
    'MgCl':     {'smiles': '[Mg]Cl',        'D_at':  3.240, 'source': 'HH'},
    'AlCl':     {'smiles': '[Al]Cl',       'D_at':  5.130, 'source': 'HH'},
    'SiCl':     {'smiles': '[Si]Cl',       'D_at':  4.020, 'source': 'HH'},
    'PCl':      {'smiles': '[P]Cl',        'D_at':  3.190, 'source': 'HH'},
    'SCl':      {'smiles': '[S]Cl',        'D_at':  2.680, 'source': 'HH'},
    'KCl':      {'smiles': '[K]Cl',        'D_at':  4.430, 'source': 'ATcT'},
    'CaCl':     {'smiles': '[Ca]Cl',        'D_at':  4.130, 'source': 'HH'},
    'BrCl':     {'smiles': 'ClBr',         'D_at':  2.173, 'source': 'ATcT'},
    'ICl':      {'smiles': 'ClI',          'D_at':  2.150, 'source': 'ATcT'},
    # ── Bromures / Iodures ──
    'LiBr':     {'smiles': '[Li]Br',       'D_at':  4.340, 'source': 'HH'},
    'NaBr':     {'smiles': '[Na]Br',       'D_at':  3.790, 'source': 'HH'},
    'KBr':      {'smiles': '[K]Br',        'D_at':  3.930, 'source': 'HH'},
    'IBr':      {'smiles': 'Br[I]',        'D_at':  1.817, 'source': 'HH'},
    'LiI':      {'smiles': '[Li]I',        'D_at':  3.540, 'source': 'HH'},
    'NaI':      {'smiles': '[Na]I',        'D_at':  3.040, 'source': 'HH'},
    'KI':       {'smiles': '[K]I',         'D_at':  3.310, 'source': 'HH'},
    # ── Oxydes ──
    'LiO':      {'smiles': '[Li][O]',       'D_at':  3.440, 'source': 'HH'},
    'BeO':      {'smiles': '[Be]=O',       'D_at':  4.540, 'source': 'HH'},
    'BO':       {'smiles': '[B]#[O]',      'D_at':  8.280, 'source': 'HH'},  # bo=3: MO bo≈2.5, bo=3 closer than 2
    'CO':       {'smiles': '[C-]#[O+]',    'D_at': 11.092, 'source': 'ATcT'},
    'NO':       {'smiles': '[N]=O',        'D_at':  6.500, 'source': 'ATcT'},
    'NaO':      {'smiles': '[Na][O]',       'D_at':  2.600, 'source': 'HH'},
    'MgO':      {'smiles': '[Mg]=O',       'D_at':  3.530, 'source': 'HH'},
    'AlO':      {'smiles': '[Al]=O',       'D_at':  5.270, 'source': 'HH'},
    'SiO':      {'smiles': '[Si]#[O]',     'D_at':  8.260, 'source': 'ATcT'},  # bo=3: 10 val e⁻ (isoelectronic CO)
    'PO':       {'smiles': '[P]=O',        'D_at':  5.690, 'source': 'HH'},
    'SO':       {'smiles': 'O=S',          'D_at':  5.343, 'source': 'ATcT'},
    'ClO':      {'smiles': '[Cl][O]',       'D_at':  2.750, 'source': 'HH'},
    # ── Nitrures / cross-row ──
    'BN':       {'smiles': '[B]=[N]',      'D_at':  3.900, 'source': 'CRC/Luo'},
    'CN':       {'smiles': '[C]#[N]',      'D_at':  7.720, 'source': 'ATcT'},
    'CS':       {'smiles': '[C]#[S]',      'D_at':  7.360, 'source': 'HH'},  # bo=3: 10 val e⁻ (isoelectronic CO)
    'NS':       {'smiles': '[N]=S',        'D_at':  4.850, 'source': 'HH'},
    'PN':       {'smiles': '[P]#N',        'D_at':  6.390, 'source': 'HH'},
    'PS':       {'smiles': '[P]=S',        'D_at':  4.500, 'source': 'HH'},
    'CP':       {'smiles': '[C]=[P]',      'D_at':  5.330, 'source': 'HH'},
    'SiS':      {'smiles': '[Si]#[S]',     'D_at':  6.390, 'source': 'HH'},  # bo=3: 10 val e⁻ (isoelectronic CO)
    'SiC':      {'smiles': '[Si]#[C]',     'D_at':  4.640, 'source': 'HH'},  # bo=3: verified (D_calc +0.7%)
    'BS':       {'smiles': '[B]#[S]',     'D_at':  5.800, 'source': 'HH'},  # bo=3: verified (D_calc +0.4%)
    # ── d-BLOCK DIATOMIQUES (3d: Sc-Zn, 4d/5d: Ag, Au) ──
    # Sources: Morse2019/2020 (predissociation, highest precision),
    #          Luo2007 (CRC BDE handbook), HH (Huber-Herzberg),
    #          3dMLBE20 (Truhlar benchmark set)
    # -- 3d Hydrides (s+d, bo=1) --
    'ScH':      {'smiles': '[Sc][H]',      'D_at':  2.06,  'source': '3dMLBE20'},
    'TiH':      {'smiles': '[Ti][H]',      'D_at':  2.08,  'source': '3dMLBE20'},
    'VH':       {'smiles': '[V][H]',       'D_at':  2.13,  'source': '3dMLBE20'},
    'CrH':      {'smiles': '[Cr][H]',      'D_at':  1.93,  'source': '3dMLBE20'},  # ±0.07 eV
    'MnH':      {'smiles': '[Mn][H]',      'D_at':  1.27,  'source': 'Luo2007'},   # weak: d⁵ half-filled
    'FeH':      {'smiles': '[Fe][H]',      'D_at':  1.59,  'source': '3dMLBE20'},
    'CoH':      {'smiles': '[Co][H]',      'D_at':  1.99,  'source': '3dMLBE20'},  # ±0.13 eV
    'NiH':      {'smiles': '[Ni][H]',      'D_at':  2.54,  'source': '3dMLBE20'},  # ±0.17 eV
    'CuH':      {'smiles': '[Cu][H]',      'D_at':  2.64,  'source': '3dMLBE20'},  # ±0.13 eV
    'ZnH':      {'smiles': '[Zn][H]',      'D_at':  0.85,  'source': 'HH'},        # d¹⁰s², very weak
    # -- 4d/5d Hydrides --
    'AgH':      {'smiles': '[Ag][H]',      'D_at':  2.29,  'source': 'HH'},
    'AuH':      {'smiles': '[Au][H]',      'D_at':  3.11,  'source': 'Luo2007'},   # relativistic 6s contraction
    # -- 3d Oxides: early d (bo=2, genuine d-π double bond) --
    'ScO':      {'smiles': '[Sc]=O',       'D_at':  6.96,  'source': '3dMLBE20'},
    'TiO':      {'smiles': '[Ti]=O',       'D_at':  6.87,  'source': '3dMLBE20'},
    'VO':       {'smiles': '[V]=O',        'D_at':  6.545, 'source': 'Morse2020'},  # predissociation ±0.002
    'CrO':      {'smiles': '[Cr]=O',       'D_at':  4.649, 'source': 'Morse2020'},  # predissociation ±0.005
    # -- 3d Oxides: late d (bo=2 for engine, actually σ + partial d-π) --
    # The engine's d-block channels (D_P2, D_d_pi) handle the d-shell
    # contribution. Bond order 2 captures the σ+π framework even if
    # the actual π is weak for late d.
    'MnO':      {'smiles': '[Mn]=O',       'D_at':  3.83,  'source': 'NIST'},      # X⁶Σ⁺ high spin
    'FeO':      {'smiles': '[Fe]=O',       'D_at':  4.17,  'source': 'Morse2019'},  # ±0.01
    'CoO':      {'smiles': '[Co]=O',       'D_at':  3.94,  'source': 'Luo2007'},   # ±0.13
    'NiO':      {'smiles': '[Ni]=O',       'D_at':  3.91,  'source': 'Luo2007'},   # ±0.17, X³Σ⁻
    'CuO':      {'smiles': '[Cu]=O',       'D_at':  2.75,  'source': 'CRC'},       # ±0.22, X²Π
    'ZnO':      {'smiles': '[Zn]=O',       'D_at':  1.61,  'source': 'NIST'},      # ±0.04, X¹Σ⁺ d¹⁰s²
    # -- 3d Halides (p+d, bo=1) --
    'CuF':      {'smiles': '[Cu]F',        'D_at':  4.42,  'source': 'Luo2007'},
    'CuCl':     {'smiles': '[Cu]Cl',       'D_at':  3.83,  'source': 'Luo2007'},
    'AgF':      {'smiles': '[Ag]F',        'D_at':  3.54,  'source': 'Luo2007'},
    'AgCl':     {'smiles': '[Ag]Cl',       'D_at':  3.24,  'source': 'Luo2007'},
    # -- Metal dimers (d+d, bo=1) --
    'Cu2':      {'smiles': '[Cu][Cu]',     'D_at':  1.886, 'source': 'Gong2014'},  # velocity map imaging ±0.026
    'Ag2':      {'smiles': '[Ag][Ag]',     'D_at':  1.66,  'source': 'HH'},
    'Au2':      {'smiles': '[Au][Au]',     'D_at':  2.290, 'source': 'Morse2019'},  # R2PI ±0.008
    'Cr2':      {'smiles': '[Cr][Cr]',     'D_at':  1.53,  'source': 'Casey1993'},  # ±0.06, notoriously weak
    # ── Triatomiques ──
    'H2O':      {'smiles': 'O',            'D_at':  9.511, 'source': 'ATcT'},
    'CO2':      {'smiles': 'O=C=O',        'D_at': 16.56,  'source': 'ATcT'},
    'HCN':      {'smiles': 'C#N',          'D_at': 13.39,  'source': 'ATcT'},
    # ── Tetratomiques ──
    'NH3':      {'smiles': 'N',            'D_at': 12.02,  'source': 'ATcT'},  # ATcT v1.130: 277.1 kcal/mol
    'CH2O':     {'smiles': 'C=O',          'D_at': 15.46,  'source': 'ATcT'},
    # ── Pentatomiques ──
    'CH4':      {'smiles': 'C',            'D_at': 17.04,  'source': 'ATcT'},  # ATcT v1.130: 392.5 kcal/mol
    # ── C-C molecules ──
    'C2H2':     {'smiles': 'C#C',          'D_at': 17.07,  'source': 'ATcT'},
    'C2H4':     {'smiles': 'C=C',          'D_at': 23.09,  'source': 'ATcT'},  # ATcT: 532.4 kcal/mol
    'C2H6':     {'smiles': 'CC',           'D_at': 28.89,  'source': 'ATcT'},  # ATcT: 666.3 kcal/mol
    # ── Larger per<=2 ──
    'CH3OH':    {'smiles': 'CO',           'D_at': 21.29,  'source': 'ATcT'},
    'HCOOH':    {'smiles': 'OC=O',         'D_at': 21.29,  'source': 'ATcT'},
    'CH3CHO':   {'smiles': 'CC=O',         'D_at': 28.02,  'source': 'ATcT'},
    'CH3F':     {'smiles': 'CF',           'D_at': 18.10,  'source': 'ATcT'},  # W4 benchmark
    'CH3NH2':   {'smiles': 'CN',           'D_at': 26.13,  'source': 'ATcT'},
    'CF4':      {'smiles': 'FC(F)(F)F',    'D_at': 20.20,  'source': 'ATcT'},  # W4: 465.8 kcal/mol, confirmed by seq BDEs
    'H2O2':     {'smiles': 'OO',           'D_at': 11.09,  'source': 'ATcT'},
    'N2H4':     {'smiles': 'NN',           'D_at': 17.96,  'source': 'ATcT'},
    # ── G2-1 test set subset ──
    'CH2CO':    {'smiles': 'C=C=O',        'D_at': 23.79,  'source': 'ATcT'},
    'CH2NH':    {'smiles': 'C=N',          'D_at': 18.11,  'source': 'ATcT'},
    'CHF3':     {'smiles': 'C(F)(F)F',     'D_at': 19.52,  'source': 'ATcT'},  # W4 benchmark
    'HCOF':     {'smiles': 'C(=O)F',       'D_at': 17.12,  'source': 'ATcT'},
    'FCHO':     {'smiles': 'FC=O',         'D_at': 16.95,  'source': 'ATcT'},
    'N2H2':     {'smiles': 'N=N',          'D_at': 11.12,  'source': 'ATcT'},
    'HNO':      {'smiles': 'N=O',          'D_at':  8.57,  'source': 'ATcT'},
    'cC3H6':    {'smiles': 'C1CC1',        'D_at': 35.29,  'source': 'NIST'},
    'cC4H8':    {'smiles': 'C1CCC1',       'D_at': 47.49,  'source': 'NIST'},
    'acetone':  {'smiles': 'CC(C)=O',      'D_at': 40.57,  'source': 'NIST'},
    'NMA':      {'smiles': 'CC(=O)N',      'D_at': 36.26,  'source': 'NIST'},
    # ── Per=3 (kinematic RC24) ──
    'SiH4':     {'smiles': '[SiH4]',       'D_at': 12.94,  'source': 'ATcT'},  # ATcT: 298.3 kcal/mol
    'PH3':      {'smiles': 'P',            'D_at':  9.89,  'source': 'ATcT'},
    'H2S':      {'smiles': 'S',            'D_at':  7.45,  'source': 'ATcT'},
    'HCl':      {'smiles': '[H]Cl',        'D_at':  4.43,  'source': 'ATcT'},
    'Cl2':      {'smiles': 'ClCl',         'D_at':  2.51,  'source': 'ATcT'},
    'NaCl':     {'smiles': '[Na]Cl',       'D_at':  4.23,  'source': 'ATcT'},
    'CH3SH':    {'smiles': 'CS',           'D_at': 19.58,  'source': 'NIST'},
    'CH3Cl':    {'smiles': 'CCl',          'D_at': 16.31,  'source': 'NIST'},
    'CHCl3':    {'smiles': 'C(Cl)(Cl)Cl',  'D_at': 14.53,  'source': 'NIST'},
    # ── Extension v0.3b (hydrocarbures + oxygenees) ──
    'propane':  {'smiles': 'CCC',          'D_at': 41.44,  'source': 'NIST'},
    'DME':      {'smiles': 'COC',          'D_at': 32.90,  'source': 'NIST'},
    'ethanol':  {'smiles': 'CCO',          'D_at': 33.43,  'source': 'NIST'},
    'propene':  {'smiles': 'CC=C',         'D_at': 35.63,  'source': 'NIST'},
    'AcOH':     {'smiles': 'CC(O)=O',      'D_at': 33.54,  'source': 'NIST'},
    'butad':    {'smiles': 'C=CC=C',       'D_at': 42.13,  'source': 'NIST'},
    'propanal': {'smiles': 'CCC=O',        'D_at': 40.35,  'source': 'NIST'},
    'butane':   {'smiles': 'CCCC',         'D_at': 53.61,  'source': 'NIST'},
    'EtNH2':    {'smiles': 'CCN',          'D_at': 36.06,  'source': 'NIST'},
    # ── Extension v0.3b+ (60 mol target) ──
    'isobutan': {'smiles': 'CC(C)C',       'D_at': 53.70,  'source': 'NIST'},
    'isobuten': {'smiles': 'C=C(C)C',      'D_at': 47.96,  'source': 'NIST'},
    'propyne':  {'smiles': 'CC#C',         'D_at': 29.40,  'source': 'NIST'},
    'formamid': {'smiles': 'NC=O',         'D_at': 23.62,  'source': 'NIST'},
    'EG':       {'smiles': 'OCCO',         'D_at': 37.64,  'source': 'NIST'},
    'HSSH':     {'smiles': 'SS',           'D_at': 10.45,  'source': 'NIST'},
    'HCOCl':    {'smiles': 'ClC=O',        'D_at': 16.04,  'source': 'NIST'},
    'MeSiH3':   {'smiles': '[SiH3]C',      'D_at': 25.96,  'source': 'NIST'},
    # ── Extension v0.3b++ (90 mol target) ──
    # Hydrocarbons
    'pentane':  {'smiles': 'CCCCC',        'D_at': 65.77,  'source': 'NIST'},
    'neopent':  {'smiles': 'CC(C)(C)C',    'D_at': 65.99,  'source': 'NIST'},
    '1butene':  {'smiles': 'C=CCC',        'D_at': 47.79,  'source': 'NIST'},
    'allene':   {'smiles': 'C=C=C',        'D_at': 29.35,  'source': 'NIST'},
    'hexane':   {'smiles': 'CCCCCC',       'D_at': 77.93,  'source': 'NIST'},
    # Alcohols / ethers
    '1-PrOH':   {'smiles': 'CCCO',         'D_at': 45.59,  'source': 'NIST'},
    '2-PrOH':   {'smiles': 'CC(O)C',       'D_at': 45.77,  'source': 'NIST'},
    'EtOMe':    {'smiles': 'CCOC',         'D_at': 45.18,  'source': 'NIST'},
    'Et2O':     {'smiles': 'CCOCC',        'D_at': 57.50,  'source': 'NIST'},
    'MeOCHO':   {'smiles': 'COC=O',        'D_at': 32.74,  'source': 'NIST'},
    # Aldehydes / ketones / acids
    'MEK':      {'smiles': 'CC(=O)CC',     'D_at': 52.84,  'source': 'NIST'},
    'glyoxal':  {'smiles': 'O=CC=O',       'D_at': 26.74,  'source': 'NIST'},
    'propanac': {'smiles': 'CCC(=O)O',     'D_at': 46.30,  'source': 'NIST'},
    'acrolein': {'smiles': 'C=CC=O',       'D_at': 34.58,  'source': 'NIST'},
    'vinylOH':  {'smiles': 'OC=C',         'D_at': 27.76,  'source': 'NIST'},
    # Nitriles
    'MeCN':     {'smiles': 'CC#N',         'D_at': 25.77,  'source': 'NIST'},
    'acrylN':   {'smiles': 'C=CC#N',       'D_at': 32.04,  'source': 'NIST'},
    # Fluorides
    'VinylF':   {'smiles': 'FC=C',         'D_at': 23.90,  'source': 'NIST'},
    'EtF':      {'smiles': 'CCF',          'D_at': 29.70,  'source': 'NIST'},
    'COF2b':    {'smiles': 'C(=O)(F)F',    'D_at': 18.28,  'source': 'NIST'},
    'CHClF':    {'smiles': 'ClCF',         'D_at': 16.73,  'source': 'NIST'},
    # Chlorides
    'CH2Cl2':   {'smiles': 'ClCCl',        'D_at': 15.45,  'source': 'NIST'},
    'VinylCl':  {'smiles': 'C=CCl',        'D_at': 22.66,  'source': 'NIST'},
    'EtCl':     {'smiles': 'CCCl',         'D_at': 28.57,  'source': 'NIST'},
    # Sulfur
    'DMS':      {'smiles': 'CSC',          'D_at': 31.67,  'source': 'NIST'},
    'EtSH':     {'smiles': 'CCS',          'D_at': 31.76,  'source': 'NIST'},
    'EtSMe':    {'smiles': 'CCSC',         'D_at': 43.85,  'source': 'NIST'},
    # Phosphorus
    'MePH2':    {'smiles': 'PC',           'D_at': 22.21,  'source': 'NIST'},
    # ── Extension 100 mol (v0.3b final) ──
    # Hydrocarbons
    'heptane':  {'smiles': 'CCCCCCC',      'D_at': 90.09,  'source': 'NIST'},
    'isopent':  {'smiles': 'CC(C)CC',      'D_at': 65.84,  'source': 'NIST'},
    'cHex':     {'smiles': 'C1CCCCC1',     'D_at': 72.96,  'source': 'NIST'},
    'octane':   {'smiles': 'CCCCCCCC',     'D_at': 102.25, 'source': 'NIST'},
    '1penten':  {'smiles': 'C=CCCC',       'D_at': 59.95,  'source': 'NIST'},
    # Aldehydes / ketones
    'butanal':  {'smiles': 'CCCC=O',       'D_at': 52.52,  'source': 'NIST'},
    'pentanal': {'smiles': 'CCCCC=O',      'D_at': 64.66,  'source': 'NIST'},
    'hexanal':  {'smiles': 'CCCCCC=O',     'D_at': 76.83,  'source': 'NIST'},
    '2pentanon':{'smiles': 'CC(=O)CCC',    'D_at': 65.00,  'source': 'NIST'},
    # Alcohols
    '1PeOH':    {'smiles': 'CCCCCO',       'D_at': 69.89,  'source': 'NIST'},
    # Fluorides / CFCs
    'CF3Me':    {'smiles': 'CC(F)(F)F',    'D_at': 31.54,  'source': 'NIST'},
    '11DFE':    {'smiles': 'C=C(F)F',      'D_at': 24.50,  'source': 'NIST'},
    'CFC12':    {'smiles': 'FC(F)(Cl)Cl',  'D_at': 16.54,  'source': 'NIST'},
    # Nitriles
    'EtCN':     {'smiles': 'CCC#N',        'D_at': 37.93,  'source': 'NIST'},
    # ── Stress test Phase C (125 mol target) ──
    # Cyclic ethers
    'oxetane':  {'smiles': 'C1COC1',       'D_at': 39.26,  'source': 'NIST'},
    'THF':      {'smiles': 'C1CCOC1',      'D_at': 52.28,  'source': 'NIST'},
    'THP':      {'smiles': 'C1CCOCC1',     'D_at': 64.63,  'source': 'NIST'},
    # Branched
    'tBuOH':    {'smiles': 'CC(C)(C)O',    'D_at': 58.13,  'source': 'NIST'},
    '23DMbut':  {'smiles': 'CC(C)C(C)C',   'D_at': 78.03,  'source': 'NIST'},
    # Amides / nitriles
    'NMA2':     {'smiles': 'CC(=O)NC',     'D_at': 48.15,  'source': 'NIST'},
    'butyrN':   {'smiles': 'CCCC#N',       'D_at': 50.08,  'source': 'NIST'},
    'dicyanog': {'smiles': 'N#CC#N',       'D_at': 21.45,  'source': 'NIST'},
    # Large hydrocarbons
    'nonane':   {'smiles': 'CCCCCCCCC',    'D_at': 114.40, 'source': 'NIST'},
    '135hextr': {'smiles': 'C=CC=CC=C',    'D_at': 60.93,  'source': 'NIST'},
    'MecPr':    {'smiles': 'C1CC1C',       'D_at': 47.80,  'source': 'NIST'},
    # Oxygenated
    'EtOAc':    {'smiles': 'CCOC(=O)C',    'D_at': 57.56,  'source': 'NIST'},
    'Ac2O':     {'smiles': 'CC(=O)OC(=O)C','D_at': 56.99,  'source': 'NIST'},
    'glycolal': {'smiles': 'OCC=O',        'D_at': 32.44,  'source': 'NIST'},
    'diglyme':  {'smiles': 'COCCOC',       'D_at': 60.99,  'source': 'NIST'},
    # Multi-halogen
    'CFC113a':  {'smiles': 'FC(F)(F)Cl',   'D_at': 18.49,  'source': 'NIST'},
    # Sulfur
    'EtSSH':    {'smiles': 'CCSS',         'D_at': 34.46,  'source': 'NIST'},
    'DMDSch':   {'smiles': 'CSCSC',        'D_at': 46.73,  'source': 'NIST'},
    # Silicon
    'Si2H6':    {'smiles': '[SiH3][SiH3]', 'D_at': 22.08,  'source': 'NIST'},
    # ══════════════════════════════════════════════════════════════
    # EXTENSION v2 — 52 new molecules (target: 173 total)
    # D_at from ΔHf(NIST) with calibrated atomic Hf fitted to
    # existing benchmark (least-squares on 31 molecules).
    # Hf(H)=2.380, Hf(C)=7.167, Hf(N)=5.149, Hf(O)=2.546,
    # Hf(F)=0.954, Hf(Cl)=1.213, Hf(S)=2.566, Hf(Si)=4.035,
    # Hf(P)=2.806, Hf(Br)=1.004, Hf(I)=0.943
    # ══════════════════════════════════════════════════════════════
    # ── Aromatics (conjugation test) ──
    'benzene':    {'smiles': 'c1ccccc1',       'D_at': 56.42,  'source': 'NIST'},
    'toluene':    {'smiles': 'Cc1ccccc1',      'D_at': 68.69,  'source': 'NIST'},
    'phenol':     {'smiles': 'Oc1ccccc1',      'D_at': 60.83,  'source': 'NIST'},
    'aniline':    {'smiles': 'Nc1ccccc1',      'D_at': 63.91,  'source': 'NIST'},
    'pyridine':   {'smiles': 'c1ccncc1',       'D_at': 51.43,  'source': 'NIST'},
    'furan':      {'smiles': 'c1ccoc1',        'D_at': 41.09,  'source': 'NIST'},
    'thiophene':  {'smiles': 'c1ccsc1',        'D_at': 39.56,  'source': 'NIST'},
    'styrene':    {'smiles': 'C=Cc1ccccc1',    'D_at': 74.85,  'source': 'NIST'},
    'naphthalene':{'smiles': 'c1ccc2ccccc2c1', 'D_at': 89.15,  'source': 'NIST'},
    # ── Amines (N-central, LP competition) ──
    'DMA':        {'smiles': 'CNC',            'D_at': 36.34,  'source': 'NIST'},
    'TMA':        {'smiles': 'CN(C)C',         'D_at': 48.32,  'source': 'NIST'},
    'diethylamine':{'smiles':'CCNCC',          'D_at': 60.74,  'source': 'NIST'},
    'pyrrolidine':{'smiles': 'C1CCNC1',       'D_at': 55.66,  'source': 'NIST'},
    'piperidine': {'smiles': 'C1CCNCC1',      'D_at': 67.67,  'source': 'NIST'},
    'pyrrole':    {'smiles': 'c1cc[nH]c1',    'D_at': 44.59,  'source': 'NIST'},
    # ── Sulfur compounds ──
    'DMSO':       {'smiles': 'CS(C)=O',        'D_at': 35.29,  'source': 'NIST'},
    'CS2':        {'smiles': 'S=C=S',          'D_at': 11.09,  'source': 'NIST'},
    'thioacetone':{'smiles': 'CC(=S)C',        'D_at': 38.96,  'source': 'NIST'},
    'THT':        {'smiles': 'C1CCSC1',        'D_at': 50.83,  'source': 'NIST'},
    'SO2mol':     {'smiles': 'O=S=O',          'D_at': 11.03,  'source': 'ATcT'},  # ATcT: 254.2 kcal/mol
    # ── Silicon ──
    'SiF4':       {'smiles': 'F[Si](F)(F)F',   'D_at': 24.59,  'source': 'NIST'},
    'SiCl4':      {'smiles': 'Cl[Si](Cl)(Cl)Cl','D_at':16.50,  'source': 'NIST-derived'},  # NIST ΔfH→D₀(0K)
    # ── Bromine / Iodine (relativistic per≥4) ──
    'HBr':        {'smiles': '[H]Br',          'D_at':  3.76,  'source': 'ATcT'},
    'HI':         {'smiles': '[H]I',           'D_at':  3.05,  'source': 'ATcT'},
    'CH3Br':      {'smiles': 'CBr',            'D_at': 15.69,  'source': 'NIST'},
    'CH3I':       {'smiles': 'CI',             'D_at': 15.10,  'source': 'NIST'},
    'CH2Br2':     {'smiles': 'BrCBr',          'D_at': 14.09,  'source': 'NIST'},
    'CBr4':       {'smiles': 'BrC(Br)(Br)Br',  'D_at': 10.88,  'source': 'NIST'},
    'EtBr':       {'smiles': 'CCBr',           'D_at': 27.88,  'source': 'NIST'},
    # ── Hypervalent ──
    'SF6':        {'smiles': 'F[S](F)(F)(F)(F)F','D_at':20.27, 'source': 'NIST-derived'},  # NIST ΔfH→D₀(0K)
    'PCl5':       {'smiles': 'Cl[P](Cl)(Cl)(Cl)Cl','D_at':12.76,'source':'NIST'},
    'CCl4mol':    {'smiles': 'ClC(Cl)(Cl)Cl',  'D_at': 13.01,  'source': 'NIST'},
    'SO3mol':     {'smiles': 'O=S(=O)=O',      'D_at': 14.31,  'source': 'NIST'},
    'POCl3':      {'smiles': 'ClP(Cl)(Cl)=O',  'D_at': 14.78,  'source': 'NIST'},
    # ── Bio-relevant ──
    'urea':       {'smiles': 'NC(N)=O',        'D_at': 31.97,  'source': 'NIST'},
    'glycine':    {'smiles': 'NCC(=O)O',       'D_at': 40.55,  'source': 'NIST'},
    'acetamide':  {'smiles': 'CC(=O)N',        'D_at': 36.40,  'source': 'NIST'},
    'DMF':        {'smiles': 'CN(C)C=O',       'D_at': 47.84,  'source': 'NIST'},
    # ── Oxygenated ──
    'DMC':        {'smiles': 'COC(=O)OC',      'D_at': 49.33,  'source': 'NIST'},
    'ethylene_oxide':{'smiles':'C1CO1',        'D_at': 26.95,  'source': 'NIST'},
    'propylene_oxide':{'smiles':'CC1CO1',      'D_at': 39.31,  'source': 'NIST'},
    # ── Larger hydrocarbons ──
    'decane':     {'smiles': 'CCCCCCCCCC',     'D_at':126.62,  'source': 'NIST'},
    'isooctane':  {'smiles': 'CC(C)CC(C)(C)C', 'D_at':102.50,  'source': 'NIST'},
    'cyclopentane':{'smiles':'C1CCCC1',        'D_at': 60.44,  'source': 'NIST'},
    '2butyne':    {'smiles': 'CC#CC',           'D_at': 41.44,  'source': 'NIST'},
    '12butadiene':{'smiles': 'C=C=CC',         'D_at': 41.27,  'source': 'NIST'},
    'vinylacetylene':{'smiles':'C=CC#C',       'D_at': 35.03,  'source': 'NIST'},
    # ── Nitrogen compounds ──
    'methylhydrazine':{'smiles':'CNN',         'D_at': 31.18,  'source': 'NIST'},
    'nitromethane':{'smiles':'C[N+](=O)[O-]',  'D_at': 25.32,  'source': 'NIST'},
    # ── Phosphorus ──
    'PF3':        {'smiles': 'FP(F)F',         'D_at': 15.60,  'source': 'NIST'},
    'PCl3':       {'smiles': 'ClP(Cl)Cl',      'D_at':  9.94,  'source': 'NIST-derived'},  # NIST ΔfH→D₀(0K)
    'TMP':        {'smiles': 'CP(C)C',         'D_at': 46.01,  'source': 'NIST'},
    # ══════════════════════════════════════════════════════════════
    # EXTENSION v3 — 24 new molecules (target: 197 total)
    # D_at from ΔfH(0K) via CCCBDB/CODATA atomic references:
    #   H=216.034, C=711.38, N=470.82, O=246.84, F=77.27,
    #   Si=446.0, P=314.64, S=274.73, Cl=119.621, Br=117.92,
    #   I=107.16, B=559.91, Al=327.62, Li=157.74, Na=107.76,
    #   K=89.90, Be=319.75  (all kJ/mol, CCCBDB/CODATA)
    # Triple-checked: CCCBDB ΔfH(0K) + ΔfH(298K) cross-check + lit D₀
    # Conversion: 1 eV = 96.485 kJ/mol
    # ══════════════════════════════════════════════════════════════
    # ── Small inorganics (N/O/halogen) ──
    'NO':         {'smiles': '[N]=O',          'D_at':  6.50,  'source': 'ATcT'},       # ΔfH(0K)=+90.54, D₀=6.497 eV lit
    'NO2':        {'smiles': 'O=[N]=O',        'D_at':  9.62,  'source': 'ATcT'},       # ΔfH(0K)=+36.78, cross: ΔfH(298K)=33.97
    'N2O':        {'smiles': '[N-]=[N+]=O',    'D_at': 11.44,  'source': 'CCCBDB'},     # ΔfH(0K)=+85.03 (Gurvich)
    'NF3':        {'smiles': 'FN(F)F',         'D_at':  8.59,  'source': 'ATcT'},       # ΔfH(0K)=-126.37, cross: ΔfH(298K)=-132.09
    'ClF':        {'smiles': 'FCl',            'D_at':  2.62,  'source': 'JANAF'},      # D₀(0K)=2.621 eV (Huber-Herzberg)
    'BrF':        {'smiles': 'FBr',            'D_at':  2.55,  'source': 'CCCBDB'},     # ΔfH(0K)=-51.20 (Gurvich)
    'ICl':        {'smiles': 'Cl[I]',          'D_at':  2.15,  'source': 'NIST-derived'},  # ΔfH(298K)=17.51, thermal-corrected
    'SF2':        {'smiles': 'FSF',            'D_at':  7.47,  'source': 'CCCBDB'},     # ΔfH(0K)=-291.0±10 (Gurvich)
    'SF4':        {'smiles': 'FS(F)(F)F',      'D_at': 13.86,  'source': 'CCCBDB'},     # ΔfH(0K)=-753.0±20 (Gurvich)
    # ── Boron compounds ──
    'BF3':        {'smiles': 'FB(F)F',         'D_at': 20.01,  'source': 'ATcT'},       # ATcT TAE₀=1930.26 kJ/mol, W4=462.6 kcal
    'BCl3':       {'smiles': 'ClB(Cl)Cl',      'D_at': 13.69,  'source': 'CCCBDB'},     # ΔfH(0K)=-402.0±2.1 (JANAF), CODATA B
    # ── Aluminum compounds ──
    'AlF3':       {'smiles': 'F[Al](F)F',      'D_at': 18.29,  'source': 'CCCBDB'},     # ΔfH(0K)=-1205.6±2.5 (JANAF)
    'AlCl3':      {'smiles': 'Cl[Al](Cl)Cl',   'D_at': 13.16,  'source': 'CCCBDB'},     # ΔfH(0K)=-582.9±2.9 (JANAF)
    # ── Halogenated methanes / CFCs ──
    'CH2F2':      {'smiles': 'FCF',            'D_at': 18.06,  'source': 'CCCBDB'},     # ΔfH(0K)=-444.55±1.7 (Gurvich)
    'CFCl3':      {'smiles': 'FC(Cl)(Cl)Cl',   'D_at': 14.81,  'source': 'CCCBDB'},     # ΔfH(0K)=-281.8±5 (Gurvich), CFC-11
    # ── Silicon halides ──
    'SiH3F':      {'smiles': '[SiH3]F',        'D_at': 15.95,  'source': 'CCCBDB'},     # ΔfH(0K)=-367.2±21 (JANAF)
    'SiH3Cl':     {'smiles': '[SiH3]Cl',       'D_at': 13.96,  'source': 'CCCBDB'},     # ΔfH(0K)=-132.8±8 (JANAF)
    # ── Hypervalent ──
    'PF5':        {'smiles': 'FP(F)(F)(F)F',   'D_at': 23.67,  'source': 'CCCBDB'},     # ΔfH(0K)=-1582.4±1.3 (Gurvich)
    # ── Alkali halides (ionic diatomics) ──
    'LiF':        {'smiles': '[Li]F',          'D_at':  5.97,  'source': 'CCCBDB'},     # ΔfH(0K)=-340.6±8.4 (JANAF)
    'LiCl':       {'smiles': '[Li]Cl',         'D_at':  4.90,  'source': 'CCCBDB'},     # ΔfH(0K)=-195.6±12.6 (JANAF)
    'NaF':        {'smiles': '[Na]F',          'D_at':  4.91,  'source': 'CCCBDB'},     # ΔfH(0K)=-288.8±2.1 (JANAF)
    'KCl':        {'smiles': '[K]Cl',          'D_at':  4.43,  'source': 'NIST'},       # textbook D₀=4.43 eV, 298K check: 4.37 eV
    # ── Beryllium halides ──
    'BeF2':       {'smiles': 'F[Be]F',         'D_at': 13.17,  'source': 'CCCBDB'},     # ΔfH(0K)=-796.7±3.9 (Gurvich)
    'BeCl2':      {'smiles': 'Cl[Be]Cl',       'D_at':  9.55,  'source': 'CCCBDB'},     # ΔfH(0K)=-362.5±3.2 (Gurvich)
    # ── Round to 200: well-documented NIST molecules ──
    # propanol removed: duplicate of 1-PrOH (NIST 44.82 vs 45.59, keeping ATcT-calibrated)
    # MeF removed: duplicate of CH3F (NIST 18.43 vs ATcT 18.10, keeping ATcT)
    'iPrOH':      {'smiles': 'CC(C)O',         'D_at': 45.05,  'source': 'NIST'},       # isopropanol, ΔfH°=-272.6±0.5 kJ/mol
}

# ── Extended database (auto-generated, March 2026) ──
try:
    from ptc.data.molecules_extended import MOLECULES_EXT
    # Merge without overwriting existing entries
    for k, v in MOLECULES_EXT.items():
        if k not in MOLECULES:
            MOLECULES[k] = v
except ImportError:
    pass  # Extended database not available

# ── ATcT v1.130 (0K convention, parsed from official HTML) ──
try:
    from ptc.data.molecules_atct import MOLECULES_ATCT
    for k, v in MOLECULES_ATCT.items():
        if k not in MOLECULES:
            MOLECULES[k] = v
except ImportError:
    pass
