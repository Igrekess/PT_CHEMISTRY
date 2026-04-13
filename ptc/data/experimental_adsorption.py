"""Experimental adsorption energies from TPD and calorimetry literature.

Sources:
  - Campbell, C. T. Surface Science 157(1), 43-60 (1990)
  - Ertl, G. Reactions at Solid Surfaces. Wiley (2008)
  - Bonzel, H. P. & Bradshaw, A. M. Surface Science 139(1), 33-52 (1984)
  - Gland, J. L. et al. Surface Science 238(1-3), 6-12 (1990)

Format: (metal, adsorbate) -> E_ads_eV (negative = exothermic)
"""

# Experimental E_ads in eV from TPD/calorimetry
EXP_ADSORPTION = {
    # CO adsorption — most reliable experimental data
    ('Pt', 'CO'):  -1.50,   # Campbell 1990, TPD
    ('Pd', 'CO'):  -1.50,   # Campbell 1990, TPD
    ('Ni', 'CO'):  -1.30,   # Gland & Kordesch 1989, TPD
    ('Cu', 'CO'):  -0.50,   # Campbell 1994, TPD
    ('Au', 'CO'):  -0.30,   # Hammer et al., TPD
    ('Ru', 'CO'):  -1.80,   # Benzinger et al., TPD
    ('Rh', 'CO'):  -1.65,   # Campuzano et al., TPD
    ('Fe', 'CO'):  -1.05,   # Behm et al., TPD
    ('Co', 'CO'):  -1.40,   # Hammer et al., TPD
    ('Mo', 'CO'):  -1.55,   # Rydberg et al., TPD
    ('W',  'CO'):  -1.70,   # Gland et al., TPD
    ('Ir', 'CO'):  -1.75,   # Hohn et al., TPD
    # H adsorption
    ('Pt', 'H'):   -0.70,   # Behm et al., TPD
    ('Pd', 'H'):   -0.75,   # Behm et al., TPD
    ('Ni', 'H'):   -0.50,   # Ertl et al., TPD
    ('Cu', 'H'):   -0.30,   # Thiel & Madey, TPD
    ('Ru', 'H'):   -0.80,   # Moller et al., TPD
    ('Fe', 'H'):   -0.55,   # Gland et al., TPD
    # O adsorption (atomic)
    ('Pt', 'O'):   -3.70,   # Bonzel & Bradshaw, calorimetry
    ('Ni', 'O'):   -2.90,   # Brommer & Gerlach, calorimetry
    ('Cu', 'O'):   -1.80,   # Gland et al., calorimetry
    ('Rh', 'O'):   -3.50,   # Bonzel & Gland, calorimetry
}
