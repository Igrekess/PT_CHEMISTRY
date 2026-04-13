"""
ptc/solvation_2body.py — Crible a 2 corps pour la solvation (STUB).

Developpement futur : depasser le Born (champ moyen 1 corps) par un
crible a 2 corps ion + solvant.

Architecture prevue
===================

Le Born-PT (solvation.py) est la limite de champ moyen : l'ion seul
dans un continuum dielectrique epsilon = P1*P2*P3*P1/(P1+1).
Precision: MAE 15% (Marcus 1991).

Le crible a 2 corps traite l'ion et chaque molecule de solvant comme
deux sous-systemes CRT couples. La cle: D_KL(ion+H2O || ion) mesure
l'information de liaison ion-solvant, qui n'est PAS capturee par Born.

3 couches du crible a 2 corps
------------------------------

Couche 1 — Coordination (canal P1, p=3)
  - N_coord = P1 + 1 = 4 molecules H2O en contact direct (tetraedre)
  - Interaction: D_KL(ion, H2O_k) pour chaque k = 1..4
  - Le D_KL depend de:
      * Distance ion-O (variable, optimisee)
      * Orientation du dipole H2O (projection sin^2_3)
      * Transfert de charge ion → H2O (canal vertex/q_stat)
  - Energie: dG_1 = -sum_k COULOMB * sin^2_3 / r_k
    avec r_k = distance ion-O du k-eme voisin

Couche 2 — Dipolaire (canal P2, p=5)
  - N_dipole ~ 2*P2 = 10 molecules dans la 2eme coquille
  - Interaction: dipole-charge (pas contact direct)
  - Orientation moyenne: sin^2_5 (canal P2, plus faible)
  - Rayon: r_2 = r_1 + a_B * P1 (geometrie du crible)
  - Energie: dG_2 = -N_dip * COULOMB * sin^2_5 * mu_H2O / r_2^2
    ou mu_H2O = dipole de l'eau (derive PT)

Couche 3 — Continuum Born (canal P3, p=7)
  - Au-dela de r_3 = r_2 + a_B * P2: continuum dielectrique
  - Facteur Born classique: (1 - 1/eps) = 311/315
  - Energie: dG_3 = -q^2 * COULOMB * sin^2_7 * (1-1/eps) / (2*r_3)

Total: dG = dG_1 + dG_2 + dG_3

Le gain attendu par rapport au Born 1-corps est la decomposition
de la 1ere couche en N_coord contributions INDIVIDUELLES, chacune
avec son propre D_KL. Cela capture:
  - La variation de CN avec la taille de l'ion (Li+ CN=4, Cs+ CN=8)
  - Le transfert de charge partiel (Principle 4: bifurcation)
  - L'asymetrie cation/anion naturellement (sans formules separees)
  - La saturation dielectrique (automatique: couche 1 ≠ couche 3)

Resultats Phase 1 (avril 2026)
================================

Phase 1a (ion-dipole discrete + Born bulk) testee:
  - Interaction ion-dipole: -CN * q * mu_H2O * cos^2_3 / r^2
  - Ecrantage local: eps_local = 1 + (CN-1) * sin^2_3
  - Born continuum au-dela de la 1ere couche
  - Resultat: MAE 22-33% → PIRE que Born 1-corps (11%)
  - Cause: le dipole de H2O en PT (2.74 D) surestime de 48%,
    et l'ecrantage local ne compense pas assez

Phase 1b (Born multi-couche coeur+valence) testee:
  - Chaque couche electronique (coeur, valence) a son rayon
  - Moyenne harmonique ponderee par le nombre d'electrons
  - Resultat: MAE 483% → CATASTROPHIQUE
  - Cause: le coeur atomique (r~0.3A) domine la moyenne harmonique,
    mais le solvant ne penetre PAS jusqu'au coeur (ecantage complet)

CONCLUSION FONDAMENTALE:
  Le Born-PT avec rayons de valence (r_cov-based) EST la solution
  optimale au niveau champ moyen. Le solvant ne voit que la couche
  EXTERNE de l'ion — les couches internes sont invisibles.
  MAE 11% (0 param) = limite du modele 1-corps.

  Le gain au-dela de 11% necessite un probleme a N-CORPS:
  N molecules d'eau dans la 1ere couche interagissent ENTRE ELLES
  (effet cooperatif), ce qui n'est pas capture par Born ni par
  ion-dipole simple.

Plan revise
===========

Phase 2 (v0.3): Interaction N-corps cooperative
  - Les N_coord H2O de la couche 1 forment un reseau H-bond
  - Chaque H2O est stabilisee par ses voisines (cooperativite)
  - En PT: resommation geometrique (Principle 6) sur le reseau
  - Le facteur cooperatif ~ G_Fisher / (CN * sin^2_3)
  - Gain attendu: 11% → ~5%

Phase 3 (v0.4): Transfert de charge (acides forts)
  - D_KL(ion+H2O) ≠ D_KL(ion) + D_KL(H2O) quand transfert complet
  - Bifurcation (Principle 4): pKa << 0 = regime de transfert
  - Cela resoudrait le probleme HCl/HBr/HI

Precision actuelle: Born-PT MAE 11% (champ moyen optimal).
Objectif v0.3: MAE < 5% (N-corps cooperatif).
Tout depuis s = 1/2. 0 parametre ajuste.
"""
from __future__ import annotations

from ptc.constants import (
    S_HALF, P1, P2, P3,
    S3, S5, S7,
    COULOMB_EV_A, A_BOHR,
)


# ── Coordination numbers PT-derives ──────────────────────────────────

def coordination_number(Z: int, charge: int) -> int:
    """Coordination number in water, derived from period and charge.

    CN = P1 + 1 = 4 for most small ions (tetrahedral)
    CN increases with period (larger ions accommodate more H2O).

    TODO: derive from D_KL minimization in Phase 1.
    """
    from ptc.geometry import period_of
    per = period_of(Z)
    q = abs(charge)

    if per <= 2:
        return P1 + 1  # 4 (tetrahedral: Li+, F-, Na+ sometimes)
    elif per <= 3:
        return 2 * P1  # 6 (octahedral: Na+, Mg2+, Cl-)
    elif per <= 4:
        return 2 * P1 + 2  # 8 (K+, Ca2+, Br-)
    else:
        return 2 * P2  # 10 (Cs+, I-, Ba2+)


# ── Stub: 2-body solvation (not yet implemented) ────────────────────

def solvation_energy_2body(Z: int, charge: int,
                           solvent: str = 'water') -> dict:
    """2-body solvation energy (STUB — falls back to Born-PT).

    This function will implement the 3-layer discrete + continuum
    solvation model when Phase 1 is complete.

    Currently delegates to the 1-body Born-PT model in solvation.py.
    """
    from ptc.solvation import solvation_energy
    result = solvation_energy(Z, charge, solvent)
    result['model'] = '1-body Born-PT (2-body not yet implemented)'
    result['CN'] = coordination_number(Z, charge)
    return result


# ── Ion-water D_KL (Phase 1 building block) ─────────────────────────

def dkl_ion_water(Z: int, charge: int, r_OW: float) -> float:
    """D_KL between an ion and a single water molecule at distance r_OW.

    This is the fundamental building block of the 2-body model:
    the information gained by adding one H2O to the ion's solvation
    shell at distance r_OW (ion-oxygen distance, Angstrom).

    D_KL = q^2 * COULOMB * sin^2_3 / (2 * r_OW * Ry)

    Normalized to bits by dividing by Ry (the PT energy unit).

    TODO (Phase 1): add orientation dependence via sin^2_5.
    TODO (Phase 2): add charge transfer via bifurcation threshold.
    """
    from ptc.constants import RY
    if r_OW <= 0:
        return 0.0
    return charge ** 2 * COULOMB_EV_A * S3 / (2.0 * r_OW * RY)
