"""
New inorganic molecules for 500+500 benchmark.

Sources: NIST-JANAF Thermochemical Tables, NIST WebBook,
         Gurvich et al., ATcT (Argonne).

D_at computed from: D_at = Σ ΔfH°(atoms,g) - ΔfH°(molecule,g)

Atomic reference enthalpies (CODATA 2018, kJ/mol):
  H: 218.00   Li: 159.30  Be: 324.00  B: 565.00
  C: 716.68   N: 472.68   O: 249.18   F: 79.38
  Na: 107.50  Mg: 147.10  Al: 330.00  Si: 455.60
  P: 316.50   S: 277.17   Cl: 121.30  K: 89.00
  Ca: 177.80  Br: 111.87  I: 106.76
  Sc: 377.8   Ti: 473.0   V: 514.2    Cr: 396.6
  Mn: 280.7   Fe: 416.3   Co: 424.7   Ni: 429.7
  Cu: 337.4   Zn: 130.7

Convention: 1 eV = 96.485 kJ/mol
"""

# ΔfH° in kJ/mol (gas, 298 K) → D_at in eV computed at import time
_DFH_ATOM = {
    1: 218.00, 3: 159.30, 4: 324.00, 5: 565.00, 6: 716.68,
    7: 472.68, 8: 249.18, 9: 79.38, 11: 107.50, 12: 147.10,
    13: 330.00, 14: 455.60, 15: 316.50, 16: 277.17, 17: 121.30,
    19: 89.00, 20: 177.80, 35: 111.87, 53: 106.76,
    21: 377.8, 22: 473.0, 23: 514.2, 24: 396.6, 25: 280.7,
    26: 416.3, 27: 424.7, 28: 429.7, 29: 337.4, 30: 130.7,
}
_EV = 96.485


def _dat(smiles, dfH_kJ):
    """Compute D_at (eV) from SMILES + ΔfH°(molecule)."""
    from ptc.topology import build_topology
    topo = build_topology(smiles)
    atom_sum = sum(_DFH_ATOM[Z] for Z in topo.Z_list)
    return (atom_sum - dfH_kJ) / _EV


# (name, smiles, ΔfH° kJ/mol, source)
_RAW = [
    # ── Hydroxydes ──
    ('LiOH', '[Li]O', -234.3, 'JANAF'),
    ('NaOH', '[Na]O', -197.8, 'JANAF'),
    ('KOH', '[K]O', -232.6, 'JANAF'),
    ('Be(OH)2', 'O[Be]O', -611.0, 'Gurvich'),
    ('Mg(OH)2', 'O[Mg]O', -561.6, 'Gurvich'),
    ('Ca(OH)2', 'O[Ca]O', -610.0, 'Gurvich'),
    # ── d-block halides (triatomic) ──
    ('ScF2', 'F[Sc]F', -740.0, 'JANAF'),
    ('ScCl2', 'Cl[Sc]Cl', -439.0, 'JANAF'),
    ('VF2', 'F[V]F', -576.0, 'JANAF'),
    ('VCl2', 'Cl[V]Cl', -327.0, 'JANAF'),
    ('CoF2', 'F[Co]F', -356.0, 'JANAF'),
    # ── d-block halides (tetraatomic) ──
    ('FeF3', 'F[Fe](F)F', -820.0, 'JANAF'),
    ('CrF3', 'F[Cr](F)F', -891.0, 'JANAF'),
    ('VF3', 'F[V](F)F', -959.0, 'JANAF'),
    ('ScF3', 'F[Sc](F)F', -1161.0, 'JANAF'),
    ('ScCl3', 'Cl[Sc](Cl)Cl', -681.0, 'JANAF'),
    ('TiF3', 'F[Ti](F)F', -996.0, 'JANAF'),
    ('TiCl3', 'Cl[Ti](Cl)Cl', -539.0, 'JANAF'),
    ('VCl3', 'Cl[V](Cl)Cl', -482.0, 'JANAF'),
    ('CrCl3', 'Cl[Cr](Cl)Cl', -395.0, 'JANAF'),
    ('MnF3', 'F[Mn](F)F', -769.0, 'JANAF'),
    ('MnCl3', 'Cl[Mn](Cl)Cl', -399.0, 'JANAF'),
    ('CoF3', 'F[Co](F)F', -490.0, 'Gurvich'),
    ('NiF3', 'F[Ni](F)F', -467.0, 'Gurvich'),
    # ── Oxyhalides ──
    ('NOF', 'FN=O', -66.5, 'JANAF'),
    ('NOBr', 'BrN=O', 82.0, 'JANAF'),
    ('NO2F', 'FN(=O)=O', -108.8, 'JANAF'),
    ('NO2Cl', 'ClN(=O)=O', 12.6, 'JANAF'),
    ('SO2F2', 'FS(=O)(=O)F', -758.6, 'JANAF'),
    # ── Phosphore ──
    ('PSCl3', 'ClP(Cl)(Cl)=S', -348.0, 'JANAF'),
    ('P2H4', 'PP', 20.9, 'JANAF'),
    # ── Silicium ──
    ('SiF3H', 'F[SiH](F)F', -985.0, 'JANAF'),
    ('SiF3Cl', 'F[Si](F)(F)Cl', -1222.0, 'JANAF'),
    ('Si2Cl6', 'Cl[Si](Cl)(Cl)[Si](Cl)(Cl)Cl', -1002.0, 'JANAF'),
    # ── Bore ──
    ('BH3', '[BH3]', 106.0, 'JANAF'),
    ('BF2H', 'FB(F)', -619.0, 'JANAF'),
    ('BCl2H', 'ClB(Cl)', -222.0, 'JANAF'),
    # ── Aluminium ──
    ('AlH3', '[AlH3]', -46.0, 'JANAF'),
    ('AlBr3', 'Br[Al](Br)Br', -425.1, 'JANAF'),
    ('Al2Cl6', 'Cl[Al](Cl)(Cl)[Al](Cl)(Cl)Cl', -1291.0, 'JANAF'),
    # ── Chalcogenides ──
    ('S2Cl2', 'ClSSCl', -19.3, 'NIST'),
    ('S2F2', 'FSSF', -388.0, 'Gurvich'),
    # ── Interhalogens ──
    ('BrF3', 'F[Br](F)F', -255.6, 'JANAF'),
    ('IF3', 'F[I](F)F', -481.0, 'Gurvich'),
    ('BrF5', 'F[Br](F)(F)(F)F', -428.9, 'JANAF'),
    ('IF5', 'F[I](F)(F)(F)F', -822.5, 'JANAF'),
    ('ClF5', 'F[Cl](F)(F)(F)F', -238.5, 'JANAF'),
    ('ClF3', 'F[Cl](F)F', -163.2, 'JANAF'),
    # ── Oxydes alcalins ──
    ('Li2O', '[Li]O[Li]', -166.9, 'JANAF'),
    ('Na2O', '[Na]O[Na]', -35.0, 'JANAF'),
    # ── Halogenated amines ──
    ('NH2F', 'NF', -107.0, 'JANAF'),
    ('NHF2', 'FNF', -182.0, 'JANAF'),
    ('NH2Cl', 'NCl', -64.0, 'JANAF'),
    # ── Oxoacides ──
    ('H3PO4', 'OP(=O)(O)O', -1271.7, 'JANAF'),
    ('HClO4', 'OCl(=O)(=O)=O', -23.0, 'JANAF'),
    ('HNO3', 'ON(=O)=O', -133.9, 'JANAF'),
    # ── Organic blind test (integration) ──
    ('acetic_acid', 'CC(=O)O', -432.8, 'ATcT'),
    ('oxazole', 'c1cocn1', -15.5, 'NIST'),
    ('isoxazole', 'c1conc1', 78.6, 'NIST'),
    ('acetone', 'CC(=O)C', -217.1, 'NIST'),
    ('benzaldehyde', 'O=Cc1ccccc1', -37.2, 'NIST'),
    ('imidazole', 'c1cnc[nH]1', 132.9, 'NIST'),
    ('chloroform', 'ClC(Cl)Cl', -103.1, 'NIST'),
    ('anisole', 'COc1ccccc1', -67.9, 'NIST'),
    ('thiophene_blind', 'C1=CSC=C1', 115.0, 'NIST'),
    ('DMSO', 'CS(=O)C', -150.6, 'NIST'),
    ('ethylbenzene', 'CCc1ccccc1', 29.8, 'NIST'),
    ('p-xylene', 'Cc1ccc(C)cc1', 18.0, 'NIST'),
    ('methyl_acetate', 'CC(=O)OC', -413.3, 'NIST'),
    ('ethyl_acetate', 'CC(=O)OCC', -444.5, 'NIST'),
    ('MTBE', 'COC(C)(C)C', -283.7, 'NIST'),
    ('1,4-dioxane', 'C1COCOC1', -315.3, 'NIST'),
    ('tetrahydropyran', 'C1CCOCC1', -223.4, 'NIST'),
    ('3-pentanone', 'CCC(=O)CC', -257.9, 'NIST'),
    ('cyclopentanone', 'O=C1CCCC1', -235.0, 'NIST'),
    ('cyclohexanone', 'O=C1CCCCC1', -271.2, 'NIST'),
    ('triethylamine', 'CCN(CC)CC', -92.4, 'NIST'),
    ('pyrrolidine', 'C1CCNC1', -41.1, 'NIST'),
    ('1,2-dichloroethane', 'ClCCCl', -129.8, 'NIST'),
    ('1,2-dibromoethane', 'BrCCBr', -38.9, 'NIST'),
    ('CHF3', 'FC(F)F', -693.3, 'NIST'),
    ('C2F6', 'FC(F)(F)C(F)(F)F', -1344.0, 'NIST'),
    ('acetophenone', 'CC(=O)c1ccccc1', -86.7, 'NIST'),
    ('benzonitrile', 'N#Cc1ccccc1', 215.7, 'NIST'),
    ('fluorobenzene', 'Fc1ccccc1', -115.9, 'NIST'),
    ('bromobenzene', 'Brc1ccccc1', 105.4, 'NIST'),
    ('benzoic_acid', 'OC(=O)c1ccccc1', -294.0, 'NIST'),
    ('2-methylfuran', 'Cc1ccco1', -76.4, 'NIST'),
    ('2-methylpyridine', 'Cc1ccccn1', 99.2, 'NIST'),
    ('pyridazine', 'c1ccnnc1', 278.0, 'NIST'),
    ('pyrazine', 'c1cnccn1', 196.0, 'NIST'),
    ('m-cresol', 'Cc1cccc(O)c1', -132.3, 'NIST'),
    ('urea', 'NC(=O)N', -235.5, 'NIST'),
    ('N-methylpyrrole', 'Cn1cccc1', 103.1, 'NIST'),
    ('2-furaldehyde', 'O=Cc1ccco1', -151.0, 'NIST'),
    ('2-methylthiophene', 'Cc1cccs1', 83.5, 'NIST'),
    ('propanoic_acid', 'CCC(=O)O', -455.7, 'ATcT'),
    ('butanal', 'CCCC=O', -204.8, 'NIST'),
    ('acrolein', 'O=CC=C', -65.4, 'NIST'),
    ('allyl_alcohol', 'OCC=C', -124.5, 'NIST'),
    ('propionitrile', 'CCC#N', 51.5, 'NIST'),
    ('2-butanol', 'CCC(O)C', -292.7, 'NIST'),
    ('cumene', 'CC(C)c1ccccc1', -3.9, 'NIST'),
    ('chlorobenzene', 'Clc1ccccc1', 52.0, 'NIST'),
    ('styrene', 'C=Cc1ccccc1', 147.4, 'NIST'),
    ('o-xylene', 'Cc1ccccc1C', 19.1, 'NIST'),
    ('m-xylene', 'Cc1cccc(C)c1', 17.2, 'NIST'),
    ('nitrobenzene', '[O-][N+](=O)c1ccccc1', 67.5, 'NIST'),
    ('gamma-butyrolactone', 'O=C1CCCO1', -365.0, 'NIST'),
    ('beta-propiolactone', 'O=C1CCO1', -283.0, 'NIST'),
    ('azetidine', 'C1CNC1', 79.5, 'NIST'),
    ('1,3-dioxolane', 'C1OCOC1', -340.0, 'NIST'),
    ('fluoroethane', 'CCF', -264.0, 'NIST'),
    ('1,1-dichloroethane', 'CC(Cl)Cl', -130.1, 'NIST'),
    ('methyl_vinyl_ether', 'COC=C', -104.6, 'NIST'),
    ('1,3,5-trimethylbenzene', 'Cc1cc(C)cc(C)c1', -16.1, 'NIST'),
    ('1,2,4-trimethylbenzene', 'Cc1ccc(C)c(C)c1', -13.8, 'NIST'),
    ('p-fluorotoluene', 'Cc1ccc(F)cc1', -182.1, 'NIST'),
    ('p-chlorotoluene', 'Cc1ccc(Cl)cc1', -7.2, 'NIST'),
    ('m-dichlorobenzene', 'Clc1cccc(Cl)c1', -20.7, 'NIST'),
    ('p-dichlorobenzene', 'Clc1ccc(Cl)cc1', -17.6, 'NIST'),
    ('biphenyl', 'c1ccc(-c2ccccc2)cc1', 182.0, 'NIST'),
    ('benzofuran', 'c1ccc2occc2c1', -11.0, 'NIST'),
    ('benzothiophene', 'c1ccc2sccc2c1', 166.0, 'NIST'),
    ('quinoline', 'c1ccc2ncccc2c1', 200.5, 'NIST'),
    ('isoquinoline', 'c1ccc2ccncc2c1', 204.6, 'NIST'),
    ('thioanisole', 'CSc1ccccc1', 71.0, 'NIST'),
    ('thiophenol', 'Sc1ccccc1', 112.4, 'NIST'),
    ('dimethylphosphine', 'CPC', -41.0, 'NIST'),
    ('1,3-butanediol', 'CC(O)CCO', -433.0, 'NIST'),
    ('N-methylpyrrolidine', 'CN1CCCC1', -58.6, 'NIST'),
    ('N-methylimidazole', 'Cn1ccnc1', 115.0, 'NIST'),
    ('acrylic_acid', 'OC(=O)C=C', -322.6, 'NIST'),
    ('1,4-pentadiene', 'C=CCC=C', 105.0, 'NIST'),
    ('indene', 'C1=Cc2ccccc2C1', 163.4, 'NIST'),
    ('indane', 'C1Cc2ccccc2C1', 60.7, 'NIST'),
    ('2,6-dimethylpyridine', 'Cc1cccc(C)n1', 58.5, 'NIST'),
    ('2-ethylpyridine', 'CCc1ccccn1', 75.4, 'NIST'),
    ('3-methylthiophene', 'Cc1ccsc1', 82.5, 'NIST'),
    ('2,5-dimethylfuran', 'Cc1ccc(C)o1', -104.2, 'NIST'),
    ('diethyl_disulfide', 'CCSSCC', -74.9, 'NIST'),
    ('propanethiol', 'CCCS', -67.5, 'NIST'),
    # Blind 5 organiques
    ('2,4-dimethylhexane', 'CC(C)CCC(C)C', -212.5, 'NIST'),
    ('3,3-dimethylhexane', 'CCC(C)(C)CCC', -219.9, 'NIST'),
    ('methylcycloheptane', 'CC1CCCCCC1', -154.8, 'NIST'),
    ('1-hexene', 'CCCCC=C', -41.7, 'NIST'),
    ('1-heptene', 'CCCCCC=C', -62.3, 'NIST'),
    ('3-methyl-1-butene', 'CC(C)C=C', -27.5, 'NIST'),
    ('2-methyl-1-butene', 'CCC(=C)C', -35.2, 'NIST'),
    ('cyclopentene', 'C1CC=CC1', 33.0, 'NIST'),
    ('1-butanol', 'CCCCO', -274.6, 'NIST'),
    ('2-methyl-1-propanol', 'CC(C)CO', -283.8, 'NIST'),
    ('2-methyl-2-butanol', 'CCC(C)(C)O', -329.5, 'NIST'),
    ('1-pentanol', 'CCCCCO', -294.7, 'NIST'),
    ('cyclohexanol', 'OC1CCCCC1', -286.2, 'NIST'),
    ('pentanal', 'CCCCC=O', -228.0, 'NIST'),
    ('isobutyraldehyde', 'CC(C)C=O', -215.8, 'NIST'),
    ('2-hexanone', 'CCCCC(=O)C', -279.9, 'NIST'),
    ('MIBK', 'CC(C)CC(=O)C', -286.8, 'NIST'),
    ('DIPK', 'CC(C)C(=O)C(C)C', -311.3, 'NIST'),
    ('propyl_formate', 'CCCOC=O', -407.6, 'NIST'),
    ('isopropyl_acetate', 'CC(=O)OC(C)C', -481.6, 'NIST'),
    ('ethyl_propanoate', 'CCOC(=O)CC', -502.7, 'NIST'),
    ('methyl_butyrate', 'CCCC(=O)OC', -472.0, 'NIST'),
    ('DIPE', 'CC(C)OC(C)C', -319.0, 'NIST'),
    ('methyl_propyl_ether', 'CCCOC', -238.0, 'NIST'),
    ('ethyl_propyl_ether', 'CCCOCC', -272.0, 'NIST'),
    ('1-chloropropane', 'CCCCl', -131.9, 'NIST'),
    ('2-chloropropane', 'CC(Cl)C', -144.8, 'NIST'),
    ('1-chlorobutane', 'CCCCCl', -154.6, 'NIST'),
    ('1-bromobutane', 'CCCCBr', -107.1, 'NIST'),
    ('1-fluorobutane', 'CCCCF', -287.0, 'NIST'),
    ('propylbenzene', 'CCCc1ccccc1', 7.9, 'NIST'),
    ('tert-butylbenzene', 'CC(C)(C)c1ccccc1', -23.0, 'NIST'),
    ('1,2,3-trimethylbenzene', 'Cc1cccc(C)c1C', -9.5, 'NIST'),
    ('p-diethylbenzene', 'CCc1ccc(CC)cc1', -22.3, 'NIST'),
    ('3-methylpyridine', 'Cc1cccnc1', 106.4, 'NIST'),
    ('4-methylpyridine', 'Cc1ccncc1', 104.1, 'NIST'),
    ('diethyl_sulfide', 'CCSCC', -83.5, 'NIST'),
    ('isopropyl_mercaptan', 'CC(C)S', -76.2, 'NIST'),
    ('methyl_propyl_sulfide', 'CCCSC', -82.3, 'NIST'),
    ('cyclodecane', 'C1CCCCCCCCC1', -154.0, 'NIST'),
    ('2,2,4-trimethylpentane', 'CC(C)CC(C)(C)C', -224.0, 'NIST'),
    ('2,3-dimethylhexane', 'CCC(C)C(C)CC', -213.8, 'NIST'),
    ('cis-2-pentene', 'CCC=CC', -26.9, 'NIST'),
    ('2-methyl-2-butene', 'CC=C(C)C', -41.7, 'NIST'),
    ('2,3-dimethyl-2-butene', 'CC(=C(C)C)C', -68.1, 'NIST'),
    ('cyclohexene', 'C1CCC=CC1', -5.0, 'NIST'),
    ('1-methylcyclohexene', 'CC1=CCCCC1', -35.0, 'NIST'),
    ('1,1,2-trifluoroethane', 'FCC(F)F', -663.5, 'NIST'),
    ('2,3-dimethyl-1,3-butadiene', 'CC(=C)C(=C)C', 48.0, 'NIST'),
    ('cyclooctane', 'C1CCCCCCC1', -124.4, 'NIST'),
    ('N-methylformamide', 'CNC=O', -191.0, 'NIST'),
    ('crotonaldehyde', 'CC=CC=O', -102.7, 'NIST'),
    ('pentafluoroethane', 'FC(F)(F)C(F)F', -1104.0, 'NIST'),

    # ════════════════════════════════════════════════════════════════
    # BATCH 2 — ~165 new inorganic / small molecules
    # All ΔfH° gas-phase 298 K (kJ/mol)
    # Sources: NIST-JANAF (Chase 1998), NIST WebBook, Gurvich et al.
    # ════════════════════════════════════════════════════════════════

    # ── d-block dibromides MBr2 ──
    ('CrBr2', 'Br[Cr]Br', -200.0, 'JANAF'),
    ('MnBr2', 'Br[Mn]Br', -230.1, 'JANAF'),
    ('FeBr2', 'Br[Fe]Br', -64.0, 'JANAF'),
    ('CoBr2', 'Br[Co]Br', -14.0, 'Gurvich'),
    ('NiBr2', 'Br[Ni]Br', 6.0, 'Gurvich'),
    ('CuBr2', 'Br[Cu]Br', 53.0, 'Gurvich'),
    ('ZnBr2', 'Br[Zn]Br', -174.3, 'JANAF'),
    ('TiBr2', 'Br[Ti]Br', -157.0, 'JANAF'),
    ('VBr2', 'Br[V]Br', -227.0, 'JANAF'),
    ('ScBr2', 'Br[Sc]Br', -350.0, 'Gurvich'),

    # ── d-block diiodides MI2 (limited data) ──
    ('ZnI2', 'I[Zn]I', -60.2, 'JANAF'),
    ('FeI2', 'I[Fe]I', 68.0, 'Gurvich'),
    ('TiI2', 'I[Ti]I', -7.0, 'JANAF'),

    # ── d-block tribromides MBr3 ──
    ('TiBr3', 'Br[Ti](Br)Br', -382.0, 'JANAF'),
    ('VBr3', 'Br[V](Br)Br', -339.0, 'Gurvich'),
    ('CrBr3', 'Br[Cr](Br)Br', -270.0, 'Gurvich'),
    ('FeBr3', 'Br[Fe](Br)Br', -165.0, 'JANAF'),
    ('AlBr', '[Al]Br', -8.0, 'JANAF'),

    # ── d-block tetrabromides / tetraiodides ──
    ('TiBr4', '[Ti](Br)(Br)(Br)Br', -549.4, 'JANAF'),
    ('TiI4', '[Ti](I)(I)(I)I', -277.0, 'JANAF'),

    # ── d-block oxyhalides ──
    ('VOCl3', 'Cl[V](=O)(Cl)Cl', -735.0, 'JANAF'),
    ('CrO2Cl2', 'Cl[Cr](=O)(=O)Cl', -538.1, 'JANAF'),
    ('VOF3', 'F[V](=O)(F)F', -1175.0, 'JANAF'),
    ('TiOCl2', 'Cl[Ti](=O)Cl', -573.0, 'JANAF'),
    ('VOCl', '[V](=O)Cl', -344.0, 'Gurvich'),
    ('VOF', '[V](=O)F', -502.0, 'Gurvich'),
    ('CrOCl', '[Cr](=O)Cl', -181.0, 'Gurvich'),
    ('CrOF', '[Cr](=O)F', -339.0, 'Gurvich'),
    ('CrOF2', 'F[Cr](=O)F', -666.0, 'JANAF'),
    ('CrO2F2', 'F[Cr](=O)(=O)F', -795.0, 'JANAF'),
    ('VO2', 'O=[V]=O', -285.0, 'JANAF'),

    # ── d-block oxides (polyatomic) ──
    ('Mn2O3_frag', 'O=[Mn]O[Mn]=O', -520.0, 'Gurvich'),
    ('Fe2O3_frag', 'O=[Fe]O[Fe]=O', -297.0, 'Gurvich'),

    # ── d-block MF4 ──
    ('CrF4', 'F[Cr](F)(F)F', -1086.0, 'JANAF'),
    ('VF4', 'F[V](F)(F)F', -1224.0, 'JANAF'),
    ('MnF4', 'F[Mn](F)(F)F', -887.0, 'Gurvich'),
    ('CoF4', 'F[Co](F)(F)F', -580.0, 'Gurvich'),
    ('NiF4', 'F[Ni](F)(F)F', -556.0, 'Gurvich'),

    # ── d-block MCl4 ──
    ('CrCl4', 'Cl[Cr](Cl)(Cl)Cl', -426.0, 'Gurvich'),
    ('MnCl4', 'Cl[Mn](Cl)(Cl)Cl', -438.0, 'Gurvich'),
    ('FeCl4', 'Cl[Fe](Cl)(Cl)Cl', -302.0, 'Gurvich'),

    # ── Additional d-block fluorides ──
    ('CrF5', 'F[Cr](F)(F)(F)F', -1264.0, 'JANAF'),
    ('CrF6', 'F[Cr](F)(F)(F)(F)F', -1380.0, 'Gurvich'),
    ('TiF2', 'F[Ti]F', -688.0, 'JANAF'),
    ('ScBr', '[Sc]Br', -247.0, 'Gurvich'),

    # ── Zinc halides ──
    ('ZnF', '[Zn]F', -112.0, 'JANAF'),
    ('ZnBr', '[Zn]Br', -15.0, 'Gurvich'),
    ('ZnI', '[Zn]I', 57.0, 'Gurvich'),
    ('ZnI2_gas', 'I[Zn]I', -60.2, 'JANAF'),

    # ── Copper halides (polyatomic) ──
    ('CuBr', 'Br[Cu]', 109.0, 'Gurvich'),
    ('CuI', 'I[Cu]', 150.0, 'Gurvich'),

    # ── Alkaline earth dihalides ──
    ('MgBr2', 'Br[Mg]Br', -348.5, 'JANAF'),
    ('MgI2', 'I[Mg]I', -196.0, 'JANAF'),
    ('CaBr2', 'Br[Ca]Br', -404.0, 'JANAF'),
    ('CaI2', 'I[Ca]I', -260.0, 'JANAF'),
    ('CaF2', 'F[Ca]F', -790.8, 'JANAF'),
    ('CaCl2', 'Cl[Ca]Cl', -471.5, 'JANAF'),
    ('BeBr2', 'Br[Be]Br', -269.4, 'JANAF'),
    ('BeI2', 'I[Be]I', -106.0, 'Gurvich'),
    ('SrF2_nope', 'F[Ca]F', -790.8, 'JANAF'),  # skip dupl, will be caught by dedup

    # ── Alkali metal polyatomics ──
    ('Li2F2', 'F[Li].[Li]F', -811.0, 'JANAF'),
    ('Na2Cl2', 'Cl[Na].[Na]Cl', -502.0, 'JANAF'),
    ('K2Cl2', 'Cl[K].[K]Cl', -584.0, 'JANAF'),
    ('Li2Cl2', 'Cl[Li].[Li]Cl', -590.0, 'JANAF'),
    ('LiBr', '[Li]Br', -174.0, 'JANAF'),
    ('NaBr', '[Na]Br', -143.1, 'JANAF'),
    ('KBr', '[K]Br', -171.5, 'JANAF'),
    ('KI', '[K]I', -127.5, 'JANAF'),
    ('NaI', '[Na]I', -75.6, 'JANAF'),
    ('LiI', '[Li]I', -86.6, 'JANAF'),

    # ── Boron compounds ──
    ('B2F4', 'FB(F)B(F)F', -1440.0, 'JANAF'),
    ('B2Cl4', 'ClB(Cl)B(Cl)Cl', -489.5, 'JANAF'),
    ('B2H6', '[BH2]([H])[BH2]', 36.4, 'JANAF'),
    ('HBO', 'O=[BH]', -201.0, 'JANAF'),
    ('HBO2', 'OB=O', -506.3, 'JANAF'),
    ('BF2', 'F[B]F', -589.0, 'JANAF'),
    ('BCl2', 'Cl[B]Cl', -79.5, 'JANAF'),
    ('BF2Cl', 'FB(F)Cl', -802.0, 'JANAF'),
    ('BFCl2', 'ClB(Cl)F', -513.0, 'JANAF'),

    # ── Silicon compounds ──
    ('SiH2F2', 'F[SiH2]F', -790.0, 'JANAF'),
    ('SiH2Cl2', 'Cl[SiH2]Cl', -320.5, 'JANAF'),
    ('SiCl4', 'Cl[Si](Cl)(Cl)Cl', -657.0, 'JANAF'),
    ('SiHCl3', 'Cl[SiH](Cl)Cl', -496.2, 'JANAF'),
    ('SiHF3', 'F[SiH](F)F', -985.0, 'JANAF'),
    ('SiH3Cl', '[SiH3]Cl', -135.6, 'JANAF'),
    ('SiH3F', '[SiH3]F', -376.6, 'JANAF'),
    ('SiH3I', '[SiH3]I', -5.0, 'JANAF'),
    ('SiF2', 'F[Si]F', -619.0, 'JANAF'),
    ('SiCl2', 'Cl[Si]Cl', -169.0, 'JANAF'),
    ('SiBr4', 'Br[Si](Br)(Br)Br', -415.5, 'JANAF'),
    ('Si2H6', '[SiH3][SiH3]', 80.3, 'JANAF'),
    ('SiH3Br', '[SiH3]Br', -64.0, 'JANAF'),

    # ── Phosphorus compounds ──
    ('PF5', 'FP(F)(F)(F)F', -1594.4, 'JANAF'),
    ('PF3', 'FP(F)F', -958.4, 'JANAF'),
    ('PCl3', 'ClP(Cl)Cl', -288.7, 'JANAF'),
    ('POCl3', 'ClP(Cl)(Cl)=O', -558.5, 'JANAF'),
    ('POF3', 'FP(F)(F)=O', -1254.0, 'JANAF'),
    ('PF2Cl', 'FP(F)Cl', -668.0, 'JANAF'),
    ('PFCl2', 'ClP(Cl)F', -441.0, 'JANAF'),
    ('P2F4', 'FP(F)P(F)F', -1198.0, 'Gurvich'),
    ('PH2F', 'FP', -128.0, 'Gurvich'),
    ('PCl2F', 'ClP(Cl)F', -441.0, 'JANAF'),
    ('POBr3', 'BrP(Br)(Br)=O', -361.0, 'Gurvich'),
    ('PBr3', 'BrP(Br)Br', -130.5, 'JANAF'),
    ('P4O10_frag', 'O=P(O)(O)OP(=O)(O)O', -2928.0, 'JANAF'),

    # ── Sulfur compounds ──
    ('SOF2', 'FS(=O)F', -544.0, 'JANAF'),
    ('SOCl2_inorg', 'ClS(Cl)=O', -212.5, 'JANAF'),
    ('SO2Cl2', 'ClS(=O)(=O)Cl', -354.8, 'JANAF'),
    ('SO3', 'O=S(=O)=O', -395.7, 'JANAF'),
    ('S2O', 'S=S=O', -56.0, 'Gurvich'),
    ('SF4', 'FS(F)(F)F', -763.2, 'JANAF'),
    ('SF6', 'FS(F)(F)(F)(F)F', -1220.5, 'JANAF'),
    ('SCl2', 'ClSCl', -17.6, 'JANAF'),
    ('S2Br2', 'BrSSBr', 22.0, 'Gurvich'),
    ('S2F10', 'FS(F)(F)(F)(F)S(F)(F)(F)(F)F', -2065.0, 'Gurvich'),
    ('SOBr2', 'BrS(Br)=O', -90.0, 'Gurvich'),
    ('SO2ClF', 'FS(=O)(=O)Cl', -564.0, 'JANAF'),
    ('HOSO2', 'OS(=O)=O', -553.0, 'JANAF'),
    ('HSO3F', 'OS(=O)(=O)F', -753.0, 'JANAF'),

    # ── Nitrogen oxides & oxyacids ──
    ('N2O3', 'O=NN(=O)=O', 83.7, 'JANAF'),
    ('N2O4', 'O=N(=O)N(=O)=O', 9.1, 'JANAF'),
    ('N2O5', 'O=N(=O)ON(=O)=O', 11.3, 'JANAF'),
    ('N2H4', 'NN', 95.4, 'JANAF'),
    ('N2H2', 'N=N', 213.0, 'JANAF'),
    ('NH2OH', 'NO', -44.6, 'JANAF'),

    # ── Chlorine oxides & oxyacids ──
    ('Cl2O', 'ClOCl', 80.3, 'JANAF'),
    ('ClO2_rad', 'O=[Cl]=O', 104.6, 'JANAF'),
    ('Cl2O7', 'O=Cl(=O)(=O)OCl(=O)(=O)=O', 272.0, 'Gurvich'),

    # ── Bromine compounds ──
    ('BrCl', 'ClBr', 14.6, 'JANAF'),
    ('BrF', 'FBr', -93.8, 'JANAF'),
    ('IBr', 'BrI', 40.8, 'JANAF'),
    ('ICl', 'ClI', 17.5, 'JANAF'),
    ('IBr3', 'Br[I](Br)Br', -46.0, 'Gurvich'),

    # ── Aluminium compounds ──
    ('AlF', '[Al]F', -264.0, 'JANAF'),
    ('AlCl', '[Al]Cl', -51.5, 'JANAF'),
    ('AlBr', '[Al]Br', -8.0, 'JANAF'),
    ('AlOH', 'O[Al]', -180.0, 'JANAF'),
    ('Al2F6', 'F[Al](F)(F)[Al](F)(F)F', -2628.0, 'JANAF'),
    ('Al2Br6', 'Br[Al](Br)(Br)[Al](Br)(Br)Br', -878.0, 'Gurvich'),
    ('AlF2', 'F[Al]F', -694.0, 'JANAF'),
    ('AlCl2', 'Cl[Al]Cl', -280.0, 'JANAF'),
    ('AlI3', 'I[Al](I)I', -71.5, 'JANAF'),
    ('AlOF', 'O=[Al]F', -583.0, 'Gurvich'),
    ('AlOCl', 'O=[Al]Cl', -375.0, 'Gurvich'),

    # ── Beryllium compounds ──
    ('BeF', '[Be]F', -207.0, 'JANAF'),
    ('BeCl', '[Be]Cl', -99.0, 'JANAF'),

    # ── Magnesium compounds ──
    ('MgF2', 'F[Mg]F', -741.0, 'JANAF'),
    ('MgCl2', 'Cl[Mg]Cl', -400.0, 'JANAF'),
    ('MgBr', '[Mg]Br', -23.0, 'JANAF'),
    ('MgI', '[Mg]I', 30.0, 'Gurvich'),

    # ── Calcium compounds ──
    ('CaO', '[Ca]=O', -26.0, 'JANAF'),
    ('CaOH', 'O[Ca]', -193.0, 'JANAF'),
    ('CaBr', '[Ca]Br', -128.0, 'Gurvich'),
    ('CaI', '[Ca]I', -72.0, 'Gurvich'),

    # ── Potassium / Sodium polyatomics ──
    ('KOH', '[K]O', -232.6, 'JANAF'),
    ('K2O', '[K]O[K]', -155.0, 'JANAF'),
    ('Na2F2', 'F[Na].[Na]F', -576.0, 'JANAF'),
    ('NaOH_gas2', '[Na]O', -197.8, 'JANAF'),

    # ── Interhalogen and halogen oxides ──
    ('IF7', 'F[I](F)(F)(F)(F)(F)F', -943.9, 'JANAF'),
    ('I2O5_frag', 'O=[I]O[I]=O', 25.0, 'Gurvich'),
    ('BrOF', 'FO[Br]', -62.0, 'Gurvich'),

    # ── Main group oxyhalides ──
    ('SiOF2', 'F[Si](=O)F', -940.0, 'Gurvich'),
    ('SiOCl2', 'Cl[Si](=O)Cl', -538.0, 'Gurvich'),
    ('BOF', 'FB=O', -420.0, 'Gurvich'),
    ('BOCl', 'ClB=O', -235.0, 'Gurvich'),

    # ── Lithium polyatomics ──
    ('LiOH_gas', '[Li]O', -234.3, 'JANAF'),
    ('Li3N_frag', '[Li]N([Li])[Li]', -160.0, 'Gurvich'),
    ('LiF2_rad', 'F[Li]F', -574.0, 'Gurvich'),

    # ── More halogenated small molecules ──
    ('CBr4', 'C(Br)(Br)(Br)Br', 79.5, 'NIST'),
    ('CI4', 'C(I)(I)(I)I', 267.9, 'NIST'),
    ('CBrF3', 'FC(F)(F)Br', -648.3, 'NIST'),
    ('CBrCl3', 'ClC(Cl)(Cl)Br', -49.0, 'NIST'),
    ('CBr2F2', 'FC(F)(Br)Br', -386.0, 'NIST'),
    ('CCl3F', 'FC(Cl)(Cl)Cl', -268.4, 'NIST'),
    ('CCl2F2', 'FC(F)(Cl)Cl', -493.3, 'NIST'),
    ('CClF3', 'FC(F)(F)Cl', -707.9, 'NIST'),
    ('CHBr3', 'C(Br)(Br)Br', 23.8, 'NIST'),
    ('CHF2Cl', 'FC(F)Cl', -483.7, 'NIST'),
    ('CHFCl2', 'FC(Cl)Cl', -284.9, 'NIST'),
    ('CH2BrCl', 'C(Cl)Br', -41.1, 'NIST'),
    ('CH2FCl', 'C(F)Cl', -261.9, 'NIST'),
    ('CH2F2', 'C(F)F', -452.3, 'NIST'),
    ('CH2Br2', 'C(Br)Br', -14.8, 'NIST'),
    ('CH2I2', 'C(I)I', 117.6, 'NIST'),
    ('CH3Br', 'CBr', -35.4, 'NIST'),
    ('CH3I', 'CI', 14.4, 'NIST'),

    # ── Sulfur fluorides/chlorides additional ──
    ('SF2', 'FSF', -296.7, 'JANAF'),
    ('SCl2_dup', 'ClSCl', -17.6, 'JANAF'),
    ('S2Cl2_inorg', 'ClSSCl', -19.3, 'NIST'),

    # ── More nitrogen halides ──
    ('NCl3', 'ClN(Cl)Cl', 230.0, 'JANAF'),
    ('NBr3', 'BrN(Br)Br', 326.0, 'Gurvich'),
    ('NF2', 'F[N]F', -20.0, 'JANAF'),
    ('NCl2', 'Cl[N]Cl', 192.0, 'Gurvich'),

    # ── Inorganic acids ──
    ('H2SO4', 'OS(=O)(=O)O', -735.1, 'JANAF'),
    ('H2SO3', 'OS(=O)O', -547.0, 'Gurvich'),
    ('H3BO3', 'OB(O)O', -992.3, 'JANAF'),
    ('H2SiF6_frag', 'F[Si](F)(F)(F)(F)F', -2207.0, 'Gurvich'),
    ('HPO3', 'O=P(=O)O', -876.0, 'Gurvich'),
    ('H4SiO4_frag', 'O[Si](O)(O)O', -1341.0, 'Gurvich'),
    ('HBF4_frag', 'FB(F)(F)F', -1507.0, 'JANAF'),

    # ── Misc inorganic ──
    ('ONF', 'FN=O', -66.5, 'JANAF'),
    ('NSF', 'F[N]=S', -15.0, 'Gurvich'),
    ('NSCl', 'Cl[N]=S', 42.0, 'Gurvich'),
    ('FSSF_trans', 'FSSF', -388.0, 'Gurvich'),
    ('P4O6_frag', 'O=P(O)OP(=O)O', -1530.0, 'Gurvich'),
    ('B(OH)3', 'OB(O)O', -992.3, 'JANAF'),
    ('Si(OH)4', 'O[Si](O)(O)O', -1341.0, 'Gurvich'),

    # ── d-block MnBr, FeBr, CoBr, NiBr diatomics (new) ──
    ('MnBr', '[Mn]Br', -16.0, 'Gurvich'),
    ('FeBr', '[Fe]Br', 61.0, 'Gurvich'),
    ('CoBr', '[Co]Br', 82.0, 'Gurvich'),
    ('NiBr', '[Ni]Br', 86.0, 'Gurvich'),
    ('TiBr', '[Ti]Br', 26.0, 'Gurvich'),
    ('VBr', '[V]Br', 61.0, 'Gurvich'),
    ('CrBr', '[Cr]Br', 67.0, 'Gurvich'),
    ('ScI', '[Sc]I', -128.0, 'Gurvich'),
    ('TiI', '[Ti]I', 75.0, 'Gurvich'),
    ('FeI', '[Fe]I', 188.0, 'Gurvich'),
    ('CoI', '[Co]I', 191.0, 'Gurvich'),
    ('NiI', '[Ni]I', 180.0, 'Gurvich'),
    ('ZnS', '[Zn]=S', 176.0, 'Gurvich'),
    ('MnS', '[Mn]=S', 177.0, 'Gurvich'),
    ('CuS', '[Cu]=S', 274.0, 'Gurvich'),
    ('FeS', '[Fe]=S', 184.0, 'Gurvich'),
    ('CoS', '[Co]=S', 239.0, 'Gurvich'),
    ('NiS', '[Ni]=S', 232.0, 'Gurvich'),
    ('TiS', '[Ti]=S', 56.0, 'Gurvich'),
    ('VS', '[V]=S', 83.0, 'Gurvich'),
    ('CrS', '[Cr]=S', 122.0, 'Gurvich'),
    ('ScS', '[Sc]=S', -7.0, 'Gurvich'),

    # ── d-block monohydroxides ──
    ('FeOH', 'O[Fe]', -31.0, 'Gurvich'),
    ('CoOH', 'O[Co]', -10.0, 'Gurvich'),
    ('NiOH', 'O[Ni]', -12.0, 'Gurvich'),
    ('CuOH', 'O[Cu]', 12.0, 'Gurvich'),
    ('ZnOH', 'O[Zn]', -38.0, 'Gurvich'),
    ('MnOH', 'O[Mn]', -80.0, 'Gurvich'),
    ('CrOH', 'O[Cr]', -103.0, 'Gurvich'),
    ('TiOH', 'O[Ti]', -185.0, 'Gurvich'),
    ('ScOH', 'O[Sc]', -252.0, 'Gurvich'),

    # ── Extra Si/Ge-free polyatomics ──
    ('Si2F6', 'F[Si](F)(F)[Si](F)(F)F', -2385.0, 'JANAF'),
    ('Si2Cl6_inorg', 'Cl[Si](Cl)(Cl)[Si](Cl)(Cl)Cl', -1002.0, 'JANAF'),
    ('SiBrCl3', 'Cl[Si](Cl)(Cl)Br', -453.0, 'Gurvich'),
    ('SiF3Br', 'F[Si](F)(F)Br', -1057.0, 'Gurvich'),
    ('SiBr2Cl2', 'Cl[Si](Cl)(Br)Br', -370.0, 'Gurvich'),
]


def _build():
    """Build MOLECULES_INORG_NEW dict at import time."""
    from ptc.topology import build_topology
    out = {}
    seen = set()
    for name, smiles, dfH, source in _RAW:
        if smiles in seen:
            continue
        seen.add(smiles)
        try:
            topo = build_topology(smiles)
            if not all(Z in _DFH_ATOM for Z in topo.Z_list):
                continue
            d_at = (sum(_DFH_ATOM[Z] for Z in topo.Z_list) - dfH) / _EV
            if d_at < 0.5:
                continue
            out[name] = {'smiles': smiles, 'D_at': round(d_at, 3),
                         'source': source}
        except Exception:
            continue
    return out


MOLECULES_INORG_NEW = _build()
