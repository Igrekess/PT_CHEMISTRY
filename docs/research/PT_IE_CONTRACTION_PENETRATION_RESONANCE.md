# Resonance contraction-penetration inter-canal pour les EI super-lourdes

Date de synthese : 2026-04-29

Statut : derivation physique PT validee numeriquement, promue en
correction canonique PTC v6.2 le 2026-04-29. A conserver en statut
DER-PHYS / VAL, non comme theoreme inconditionnel.

Fichiers lies :

- `ptc/atom.py`
- `docs/research/PT_IE_SUPERHEAVY_JJ2_DERIVATION.md`
- `ptc/data/experimental.py`

---

## 1. Probleme

Apres integration de l'echo spinoriel `jj2`, les residus super-lourds
ne sont plus domines par le motif brut du p-block. La MAE tombe a :

```text
Z = 104..118 : 0.243 %
Z = 113..118 : 0.217 %
```

Mais les derniers residus significatifs sont majoritairement negatifs :
le moteur PT predit une energie d'ionisation trop basse, donc une
liaison trop faible.

Table courante apres `jj2` :

| Z | element | bloc | residu |
|---:|---|---|---:|
| 104 | Rf | d | -0.534 % |
| 105 | Db | d | -0.654 % |
| 106 | Sg | d | -0.673 % |
| 111 | Rg | d | -0.230 % |
| 115 | Mc | p | -0.356 % |
| 116 | Lv | p | -0.651 % |

Lecture physique immediate :

```text
residu = PT - reference < 0
=> IE_PT trop basse
=> Z_eff trop faible
=> S trop grand
=> il manque une correction d'action negative.
```

Une correction negative de l'action est exactement la signature d'une
contraction/penetration relativiste : le canal actif penetre davantage
vers le noyau, voit une charge effective plus forte, et l'energie
d'ionisation augmente.

---

## 2. Principe PT

Le moteur IE utilise :

```text
IE(Z) = Ry (Z_eff/per)^2,
Z_eff = Z exp(-S).
```

Une correction d'action `Delta S` donne donc :

```text
IE -> IE exp(-2 Delta S).
```

Ainsi :

- `Delta S < 0` augmente l'energie d'ionisation ;
- `Delta S > 0` diminue l'energie d'ionisation.

La resonance cherchee doit donc etre un terme `Delta S_CPR < 0`, ou
`CPR` signifie contraction-penetration resonance.

---

## 3. Seuil relativiste

On conserve le meme seuil que pour `jj2` :

```text
rho_Z = [(Z alpha_EM)^2 - s]_+,
s = 1/2.
```

Ce choix evite de reintroduire une correction dans les regimes deja
valides. La resonance ne s'active que lorsque la force relativiste
effective depasse le seuil de l'involution de spin.

---

## 4. Canal source et canal recepteur

La correction n'est pas un terme global en `Z`. Elle agit seulement
lorsqu'un canal contracte alimente un canal voisin encore ouvert.

Canal source :

```text
7s / 7p_1/2 contracte
```

Canaux recepteurs observes :

```text
6d avant demi-remplissage : Rf, Db, Sg
7p_3/2 avant paire complete : Mc, Lv
```

Le terme s'eteint aux points de symetrie :

- `d^5` : demi-remplissage de Hund du pentagone d ;
- `p_3/2^3, p_3/2^4` apres la paire active : compensation par pairing
  et fermeture spinorielle ;
- `p^1, p^2` : deja traites par la branche contractee `p_1/2` de `jj2`.

---

## 5. Forme minimale

La resonance doit porter :

1. le seuil relativiste `rho_Z` ;
2. le couplage angulaire hexagonal `sin^2(theta_3)` ;
3. le gap de penetration hexagonal `delta_3` ;
4. une projection discrete sur le canal recepteur.

La forme canonique retenue est donc :

```text
Delta S_CPR(Z) =
  - 1_{per=7} rho_Z sin^2(theta_3) delta_3
    [Psi_d(Z) + Psi_p(Z)].
```

Le signe moins encode la contraction : la resonance diminue l'action
d'ecrantage, donc augmente `Z_eff`.

---

## 6. Projection d : resonance 7s -> 6d

Pour le bloc d de periode 7, la poche residuelle est Rf--Sg :

```text
n_d = 2, 3, 4 ; n_s = 2.
```

Elle s'arrete a `d^5`, point de symetrie de Hund du pentagone.

Le recepteur d traverse le pentagone, donc le gap hexagonal est habille
par le premier facteur pentagonal :

```text
Psi_d(Z) =
  1_{l=2, n_s=2, 2 <= n_d < P_2} (1 + delta_5).
```

Ce facteur n'est pas ajuste : `delta_5` est deja une constante PT du
pentagone. Il encode le fait que la penetration hexagonale est recue
par un canal d, donc par une face pentagonale.

---

## 7. Projection p : resonance p_1/2 -> p_3/2

Apres `jj2`, la poche residuelle du p-block est essentiellement :

```text
Mc : p^3, premier electron du quartet p_3/2
Lv : p^4, paire active du quartet p_3/2
```

On pose :

```text
n_{3/2} = n_p - 2.
```

La resonance augmente de moitie au premier electron du quartet, puis
atteint son amplitude complete a la paire active :

```text
h_p(n_p) = (n_p - 2)/2, pour n_p = 3,4.
```

Le canal est spinoriel et profond dans la periode 7. On le dresse donc
par :

```text
s (1 + delta_7).
```

D'ou :

```text
Psi_p(Z) =
  1_{l=1, 3 <= n_p <= 4}
  s (1 + delta_7) (n_p - 2)/2.
```

---

## 8. Formule complete

La resonance contraction-penetration inter-canal est :

```text
rho_Z = [(Z alpha_EM)^2 - s]_+

Delta S_CPR(Z) =
  - 1_{per=7} rho_Z sin^2(theta_3) delta_3
    [
      1_{l=2, n_s=2, 2 <= n_d < P_2} (1 + delta_5)
      +
      1_{l=1, 3 <= n_p <= 4}
        s (1 + delta_7) (n_p - 2)/2
    ].
```

Equivalent dans le moteur :

```text
IE_CPR(Z) = IE_current(Z) exp(-2 Delta S_CPR(Z)).
```

---

## 9. Validation canonique v6.2

Application a la table courante apres `jj2`, sans modifier les autres
elements :

| Domaine | MAE apres jj2 | MAE avec CPR |
|---|---:|---:|
| Z = 1..86 | 0.061 % | 0.061 % |
| Z = 1..103 | 0.056 % | 0.056 % |
| Z = 104..118 | 0.243 % | 0.077 % |
| Z = 113..118 | 0.217 % | 0.064 % |

Residus super-lourds :

| Z | element | `Delta S_CPR` | residu apres jj2 | residu avec CPR |
|---:|---|---:|---:|---:|
| 104 | Rf | -0.0023154 | -0.534 % | -0.072 % |
| 105 | Db | -0.0026317 | -0.654 % | -0.130 % |
| 106 | Sg | -0.0029510 | -0.673 % | -0.085 % |
| 107 | Bh | 0 | +0.038 % | +0.038 % |
| 108 | Hs | 0 | -0.029 % | -0.029 % |
| 109 | Mt | 0 | +0.016 % | +0.016 % |
| 110 | Ds | 0 | -0.087 % | -0.087 % |
| 111 | Rg | 0 | -0.230 % | -0.230 % |
| 112 | Cn | 0 | -0.081 % | -0.081 % |
| 113 | Nh | 0 | -0.022 % | -0.022 % |
| 114 | Fl | 0 | +0.125 % | +0.125 % |
| 115 | Mc | -0.0014742 | -0.356 % | -0.062 % |
| 116 | Lv | -0.0031214 | -0.651 % | -0.029 % |
| 117 | Ts | 0 | +0.042 % | +0.042 % |
| 118 | Og | 0 | -0.106 % | -0.106 % |

Le maximum residuel sur `Z=104..118` devient `0.230 %` sur Rg, qui n'est
pas touche par la resonance CPR.

---

## 10. Script minimal

```python
from ptc.atom import IE_eV
from ptc.constants import AEM, S_HALF, S3, D3, D5, D7, P2
from ptc.periodic import l_of, period, n_fill, ns_config


def S_cpr_interchannel(Z: int) -> float:
    if period(Z) != 7:
        return 0.0

    rho = max(0.0, (Z * AEM) ** 2 - S_HALF)
    if rho <= 0.0:
        return 0.0

    l = l_of(Z)
    n = n_fill(Z)
    ns = ns_config(Z)

    amp = rho * S3 * D3
    S = 0.0

    if l == 2 and ns == 2 and 2 <= n < P2:
        S -= amp * (1.0 + D5)

    if l == 1 and 3 <= n <= 4:
        S -= amp * S_HALF * (1.0 + D7) * (n - 2) / 2.0

    return S


def IE_cpr_eV(Z: int) -> float:
    # Since PTC v6.2, CPR is already included in IE_eV via screening_action.
    return IE_eV(Z)
```

---

## 11. Interpretation

La correction CPR n'est pas une nouvelle calibration. Elle exprime une
lecture PT simple :

```text
canal contracte relativiste
  -> penetration radiale accrue
  -> baisse locale de l'action S
  -> augmentation de Z_eff
  -> correction des residus trop bas.
```

Elle est localisee parce qu'une resonance suppose deux conditions :

1. un canal source contracte au-dessus du seuil relativiste ;
2. un canal recepteur ouvert, avant son point de symetrie.

Cela explique pourquoi Rf--Sg et Mc--Lv bougent, tandis que Bh--Og
restent essentiellement inchanges.

---

## 12. Statut epistemique

La formule est promue en correction canonique PTC v6.2 parce qu'elle est
bien motivee par la logique PT :

- seuil `rho_Z` deja utilise par `jj2` ;
- action plutot qu'energie directe ;
- signe impose par contraction/penetration ;
- localisation par remplissage discret ;
- amplitudes construites avec `delta_3`, `delta_5`, `delta_7`, `s`,
  sans parametre ajuste.

Mais elle ne doit pas encore etre presentee comme theoremique. Les
references `Z > 103` sont des valeurs theoriques relativistes modernes,
pas des mesures directes. Le statut correct est donc :

```text
DER-PHYS / VAL : derivation physique PT validee numeriquement.
```

Audits restants avant toute promotion plus forte :

1. verification contre plusieurs tables relativistes externes ;
2. audit de non-double-comptage avec `S_rel`, `j-split`, `jj2`, et la
   correction d expansion/contraction d ;
3. demonstration que le motif vient de la structure discrete et non
   d'un fit implicite sur les residus.
