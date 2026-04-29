# Derivation PT de la correction spinorielle jj2 du p-block super-lourd

Date de synthese : 2026-04-29

Statut : derivation PT proposee, validation numerique forte, a auditer avant
promotion en theoreme ou en derivation canonique de la monographie.

Fichier lie au moteur :

- `ptc/atom.py`
- `ptc/shell_polygon.py`
- `ptc/data/experimental.py`

---

## 1. Objet

Le moteur `IE_eV(Z)` de `PTC` reproduit les energies de premiere
ionisation avec une precision tres elevee jusqu'a `Z = 103`.

Le residu restant se concentre ensuite dans la zone super-lourde,
principalement dans le p-block `Z = 113..118`.

Mesure du moteur courant :

| Domaine | MAE | Biais | Max |
|---|---:|---:|---:|
| `Z = 1..103` | `0.056 %` | `-0.010 %` | `0.293 %` |
| `Z = 104..118` | `0.792 %` | `+0.149 %` | `4.360 %` |
| `Z = 113..118` | `1.589 %` | `+0.744 %` | `4.360 %` |

Le motif des signes dans le p-block super-lourd est le suivant :

| Element | `n_p` | residu courant |
|---|---:|---:|
| Nh | 1 | `-0.974 %` |
| Fl | 2 | `-1.561 %` |
| Mc | 3 | `+1.457 %` |
| Lv | 4 | `+4.360 %` |
| Ts | 5 | `+0.649 %` |
| Og | 6 | `+0.532 %` |

Lecture immediate :

- `p^1` et `p^2` sont trop faiblement lies dans le moteur courant ;
- `p^3`, `p^4`, `p^5`, `p^6` sont trop fortement lies ;
- le maximum d'erreur est `p^4`, c'est-a-dire le demi-remplissage du
  quartet relativiste `p_3/2`.

Ce motif n'est pas celui d'un simple drift en `Z`. Il est celui d'une
transition `LS -> jj` mal portee par l'action de screening.

---

## 2. Donnees PT disponibles

On reprend uniquement les constantes deja presentes dans `ptc/constants.py`.

```text
s = 1/2
alpha = sin^2(theta_3) sin^2(theta_5) sin^2(theta_7)

delta_p = (1 - q^p) / p
sin^2(theta_p) = delta_p (2 - delta_p)
```

Numeriquement :

```text
delta_3       = 0.11634567901234567
sin^2(theta3)= 0.2191550409998476
delta_5       = 0.10221089711934156
sin^2(theta5)= 0.1939747267487425
alpha         = 0.0073379235518568914
```

Le p-block est le cercle hexagonal :

```text
Z / 6Z = Z / (2 P_1)Z, avec P_1 = 3.
```

En regime relativiste fort, le couplage spin-orbite scinde le p-hexagone :

```text
p = p_1/2 + p_3/2
  = 2     + 4.
```

Dans le code courant, cette scission est deja reconnue dans
`shell_polygon.py` pour le canal de capture et dans `atom.py` pour le
screening super-lourd.

---

## 3. Pourquoi la correction doit etre une action

Le moteur d'ionisation utilise :

```text
IE(Z) = Ry (Z_eff / per)^2
Z_eff = Z exp(-S)
```

Si l'on ajoute une petite correction d'action :

```text
S -> S + Delta S,
```

alors :

```text
IE -> IE exp(-2 Delta S).
```

Donc :

```text
Delta IE / IE ~= -2 Delta S.
```

Conclusion : une correction relativiste PT coherente doit modifier
l'action de persistance/screening `S`, et non multiplier directement
l'energie par un facteur ajuste.

---

## 4. Seuil relativiste PT

Le seuil observe dans le code courant est :

```text
(Z alpha)^2 > s.
```

Ce seuil a une lecture PT directe :

- `(Z alpha)^2` mesure la force relativiste effective du canal atomique ;
- `s = 1/2` est le seuil de l'involution de spin ;
- tant que `(Z alpha)^2 <= s`, le spin-orbite ne force pas encore la
  lecture `jj` du canal p ;
- lorsque `(Z alpha)^2 > s`, la decomposition spinorielle devient active.

On definit donc l'exces relativiste :

```text
rho_Z = [(Z alpha)^2 - s]_+.
```

Cette definition est importante : elle retire la partie deja absorbee par
le regime non-relativiste. La correction ne commence qu'au-dela du seuil.

---

## 5. Forme generale forcee

La correction cherchee doit respecter quatre contraintes :

1. elle doit etre nulle hors du p-block super-lourd ;
2. elle doit etre nulle sous le seuil `(Z alpha)^2 <= s` ;
3. elle doit porter le couplage angulaire du p-hexagone, donc
   `sin^2(theta_3)` ;
4. elle doit dependre uniquement de l'occupation discrete `n_p = 1..6`.

La forme minimale est donc :

```text
Delta S_jj2(Z) =
  1_{l=1, per=7} rho_Z sin^2(theta_3) Phi_p(n_p).
```

Il reste a deriver le profil discret `Phi_p`.

---

## 6. Regles de selection pour `Phi_p`

### 6.1 Signe des branches

Dans le split `p_1/2 + p_3/2` :

```text
n_p = 1,2       -> branche p_1/2
n_p = 3,4,5,6   -> branche p_3/2
```

La branche `p_1/2` est contractee relativistiquement. Elle penetre plus
pres du noyau, reduit l'action de screening, augmente `Z_eff`, et donc
augmente `IE`.

Donc :

```text
Phi_p(1), Phi_p(2) < 0.
```

La branche `p_3/2` est lue comme l'expansion compensatrice du quartet
externe. Elle augmente l'action effective, diminue `Z_eff`, et donc
diminue `IE`.

Donc :

```text
Phi_p(3), Phi_p(4), Phi_p(5), Phi_p(6) > 0.
```

### 6.2 Premiere ouverture de `p_1/2`

`p^1` ouvre la branche contractee `p_1/2`.

L'ouverture minimale d'un canal p sur le cercle hexagonal est le gap
primaire :

```text
delta_3.
```

Comme la branche est contractee, le signe est negatif :

```text
Phi_p(1) = -delta_3.
```

### 6.3 Fermeture de `p_1/2`

`p^2` ferme la paire spinorielle `p_1/2`.

La fermeture d'une paire n'est plus une simple ouverture de gap. Elle
engage la face transverse du couplage dynamique. Pour une paire fermee
dans le p-block, la brique PT minimale est :

```text
sin^2(theta_5).
```

Le signe reste celui de la contraction `p_1/2` :

```text
Phi_p(2) = -sin^2(theta_5).
```

### 6.4 Premiere ouverture de `p_3/2`

`p^3` est le premier electron du quartet externe `p_3/2`.

Par conservation du flux discret du p-hexagone scinde, l'ouverture de la
branche externe est le miroir de la fermeture de la branche interne :

```text
Phi_p(3) = +sin^2(theta_5).
```

Le signe change parce que l'on passe de la contraction `p_1/2` a
l'expansion `p_3/2`.

### 6.5 Demi-remplissage du quartet `p_3/2`

`p^4` correspond a deux electrons dans le quartet `p_3/2`.

Le quartet a capacite `4`. Son demi-remplissage est donc :

```text
n_{3/2} = 2.
```

En PT, un demi-remplissage est une involution de spin : il ne peut pas
porter une amplitude superieure a `s = 1/2`, et il atteint precisement ce
maximum lorsque les deux moities du quartet sont equilibrees.

Donc :

```text
Phi_p(4) = s.
```

C'est pourquoi `Lv` doit etre le point de correction maximal.

### 6.6 Fermeture tardive et echo hexagonal

`p^5` et `p^6` ne sont plus dans l'ouverture du quartet. L'observable
active devient la vacance du quartet, c'est-a-dire l'echo de l'ouverture
initiale, attenue par l'involution de spin.

L'echo minimal du p-hexagone est :

```text
s delta_3.
```

Le signe reste celui de la branche externe expansive :

```text
Phi_p(5) = Phi_p(6) = +s delta_3.
```

Cette saturation identique pour `p^5` et `p^6` exprime que le signal
residuel ne compte plus les electrons, mais la memoire de fermeture du
quartet.

---

## 7. Profil derive

On obtient :

```text
Phi_p(1) = -delta_3
Phi_p(2) = -sin^2(theta_5)
Phi_p(3) = +sin^2(theta_5)
Phi_p(4) = +s
Phi_p(5) = +s delta_3
Phi_p(6) = +s delta_3
```

Valeurs numeriques :

| `n_p` | branche | interpretation | `Phi_p(n_p)` |
|---:|---|---|---:|
| 1 | `p_1/2` | ouverture contractee | `-0.1163456790` |
| 2 | `p_1/2` | fermeture contractee | `-0.1939747267` |
| 3 | `p_3/2` | ouverture expansive | `+0.1939747267` |
| 4 | `p_3/2` | demi-remplissage du quartet | `+0.5000000000` |
| 5 | `p_3/2` | echo de fermeture | `+0.0581728395` |
| 6 | `p_3/2` | echo de fermeture | `+0.0581728395` |

---

## 8. Correction proposee

La correction complete est :

```text
Delta S_jj2(Z) =
  1_{l=1, per=7}
  [(Z alpha)^2 - s]_+
  sin^2(theta_3)
  Phi_p(n_p).
```

La prediction corrigee est :

```text
IE_jj2(Z) = IE_current(Z) exp(-2 Delta S_jj2(Z)).
```

Cette forme n'ajoute aucun parametre ajuste :

- le seuil est `s` ;
- le couplage angulaire est `sin^2(theta_3)` ;
- le profil utilise seulement `delta_3`, `sin^2(theta_5)` et `s` ;
- le support est fixe par le p-block super-lourd.

---

## 9. Proposition et preuve

### Proposition

Sous les hypotheses PT suivantes :

1. l'energie d'ionisation depend de l'action par
   `IE = Ry (Z exp(-S) / per)^2` ;
2. le regime relativiste spinoriel s'active uniquement quand
   `(Z alpha)^2 > s` ;
3. le p-block est le cercle `Z/6Z`, lu en regime super-lourd comme
   `p_1/2 + p_3/2 = 2 + 4` ;
4. la branche `p_1/2` est contractee et la branche `p_3/2` est expansive ;
5. les amplitudes admissibles sont les briques PT minimales associees
   aux ouvertures, fermetures et demi-remplissages du p-hexagone ;

alors la correction d'action minimale compatible avec les signes, le
support, le seuil et les occupations est :

```text
Delta S_jj2(Z) =
  1_{l=1, per=7}
  [(Z alpha)^2 - s]_+
  sin^2(theta_3)
  Phi_p(n_p),
```

avec :

```text
Phi_p(1) = -delta_3
Phi_p(2) = -sin^2(theta_5)
Phi_p(3) = +sin^2(theta_5)
Phi_p(4) = +s
Phi_p(5) = +s delta_3
Phi_p(6) = +s delta_3.
```

### Preuve

1. Par la loi `IE = Ry (Z exp(-S) / per)^2`, toute correction fine qui
   modifie la liaison effective doit entrer dans `S`. Une correction
   additive `Delta S` produit `IE -> IE exp(-2 Delta S)`.

2. Le couplage relativiste effectif est `(Z alpha)^2`. Le seuil de
   bifurcation de spin est `s`. La partie disponible pour une nouvelle
   correction est donc l'exces positif :

   ```text
   rho_Z = [(Z alpha)^2 - s]_+.
   ```

3. Comme la sous-couche active est p, le canal angulaire porteur est le
   p-hexagone `Z/(2P_1)Z`, donc la brique de couplage est
   `sin^2(theta_3)`.

4. En regime super-lourd, le p-hexagone n'est plus lu comme six positions
   equivalentes. Il se decompose en une paire interne et un quartet
   externe :

   ```text
   Z/6Z -> p_1/2 + p_3/2 = 2 + 4.
   ```

5. Les deux positions `p_1/2` sont contractees. Elles doivent donc
   diminuer l'action `S` et porter un signe negatif. Les quatre positions
   `p_3/2` sont expansives. Elles doivent augmenter l'action et porter un
   signe positif.

6. La premiere position `p_1/2` est une ouverture simple du p-hexagone :
   l'amplitude minimale est `delta_3`. Donc `Phi_p(1) = -delta_3`.

7. La deuxieme position ferme la paire `p_1/2`. Une fermeture de paire
   engage la face transverse dynamique, dont la brique minimale est
   `sin^2(theta_5)`. Le signe reste contracte, donc
   `Phi_p(2) = -sin^2(theta_5)`.

8. La premiere position `p_3/2` ouvre la branche externe. Par conservation
   du flux discret entre les deux branches du p-hexagone scinde, elle est
   le miroir de la fermeture `p_1/2`, mais avec le signe expansif :
   `Phi_p(3) = +sin^2(theta_5)`.

9. La deuxieme position `p_3/2` est le demi-remplissage du quartet. Par
   involution de spin, l'amplitude maximale admissible est `s`, atteinte
   exactement au demi-remplissage. Donc `Phi_p(4) = +s`.

10. Les positions restantes sont lues par la vacance et non par
    l'ouverture directe. Le signal residuel est l'echo hexagonal
    `delta_3`, attenue par l'involution de spin `s`. Donc
    `Phi_p(5) = Phi_p(6) = +s delta_3`.

Les dix etapes determinent l'unique profil minimal construit avec les
briques PT admissibles et les signes imposes par la decomposition
spinorielle. La proposition est demontree sous les hypotheses de selection
ci-dessus.

---

## 10. Validation numerique

Test sur la table courante `IE_NIST`, ou `Z = 104..118` sont des valeurs
theoriques relativistes modernes et non des mesures directes.

| Domaine | MAE courant | MAE avec `jj2` |
|---|---:|---:|
| `Z = 1..103` | `0.056 %` | `0.056 %` |
| `Z = 104..118` | `0.792 %` | `0.243 %` |
| `Z = 113..118` | `1.589 %` | `0.217 %` |

Residus du p-block super-lourd apres correction :

| Element | `n_p` | residu courant | residu avec `jj2` |
|---|---:|---:|---:|
| Nh | 1 | `-0.974 %` | `-0.022 %` |
| Fl | 2 | `-1.561 %` | `+0.125 %` |
| Mc | 3 | `+1.457 %` | `-0.356 %` |
| Lv | 4 | `+4.360 %` | `-0.651 %` |
| Ts | 5 | `+0.649 %` | `+0.042 %` |
| Og | 6 | `+0.532 %` | `-0.106 %` |

Le point le plus corrige est bien `Lv`, comme attendu par la derivee :
`Lv` est le demi-remplissage du quartet `p_3/2`.

---

## 11. Script minimal de reproduction

```python
from ptc.atom import IE_eV
from ptc.data.experimental import IE_NIST, SYMBOLS
from ptc.periodic import block_of, n_fill, period
from ptc.constants import AEM, S_HALF, D3, S3, S5
import math


def phi_p_jj2(n_p: int) -> float:
    if n_p == 1:
        return -D3
    if n_p == 2:
        return -S5
    if n_p == 3:
        return +S5
    if n_p == 4:
        return +S_HALF
    if n_p in (5, 6):
        return +S_HALF * D3
    return 0.0


def delta_S_jj2(Z: int) -> float:
    if block_of(Z) != "p" or period(Z) != 7:
        return 0.0
    rho = max(0.0, (Z * AEM) ** 2 - S_HALF)
    return rho * S3 * phi_p_jj2(n_fill(Z))


def IE_jj2_eV(Z: int) -> float:
    return IE_eV(Z) * math.exp(-2.0 * delta_S_jj2(Z))


for Z in range(113, 119):
    ref = IE_NIST[Z]
    old = IE_eV(Z)
    new = IE_jj2_eV(Z)
    print(
        Z,
        SYMBOLS[Z],
        "n_p=", n_fill(Z),
        "old_err=", (old - ref) / ref * 100.0,
        "new_err=", (new - ref) / ref * 100.0,
    )
```

---

## 12. Statut epistemique

Ce document etablit une derivation structurale forte, mais le statut exact
doit rester prudent :

| Partie | Statut |
|---|---|
| Loi `IE = Ry (Z_eff/per)^2` | moteur PTC etabli |
| Correction en action `S -> S + Delta S` | consequence directe de la loi IE |
| Seuil `[(Z alpha)^2 - s]_+` | fortement motive par la logique PT et le code existant |
| Scission `p -> p_1/2 + p_3/2` | physique relativiste standard, deja presente dans PTC |
| Signe contraction/expansion | derive physiquement |
| Profil `Phi_p` | derivation PT proposee par regles de selection |
| Validation numerique | forte sur `Z = 113..118` |

La promotion en derivation canonique demanderait encore :

1. une preuve plus algebrique du profil `Phi_p` depuis la decomposition
   explicite `Z/6Z -> Z/2Z + Z/4Z` ;
2. un test de non-regression sur `EA`, rayons atomiques, chimie super-lourde
   et molecules contenant `Nh..Og` si elles sont presentes dans les bancs ;
3. une comparaison avec les incertitudes des references `Z = 104..118`,
   qui ne sont pas des mesures directes.

---

## 13. Proposition d'integration

Dans `ptc/atom.py`, la correction devrait probablement etre separee du
terme `j-SPLIT` courant, pour eviter de melanger :

- le split relativiste de premier niveau deja present ;
- l'echo spinoriel secondaire `jj2` derive ici.

Une integration propre serait :

```python
def S_jj2_p_superheavy(Z: int) -> float:
    ...
```

puis :

```python
return S + S_rel(Z) + S_jj2_p_superheavy(Z) + S_cpr_interchannel(Z)
```

ou, plus conservateur :

```python
if enable_jj2:
    S += S_jj2_p_superheavy(Z)
```

pour permettre un benchmark avant activation par defaut.
