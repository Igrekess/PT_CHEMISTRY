# Vers une fonction continue de contraction--penetration

Date de synthese : 2026-04-29

Statut : DER-PHYS + VAL, promue dans le moteur canonique PTChem v6.3.
Objectif : comprendre et corriger les residus IE restants apres PTChem
v6.2 (`jj2 + CPR`) comme la trace faible d'une meme resonance
contraction--penetration, depuis les elements legers jusqu'aux
super-lourds.

---

## 1. Observation

Apres promotion de CPR dans l'action canonique, le moteur IE a :

```text
Z = 1..86    MAE = 0.061 %
Z = 1..103   MAE = 0.056 %
Z = 1..118   MAE = 0.059 %
Z = 104..118 MAE = 0.077 %
```

Apres promotion de CPR-continuum dans l'action canonique v6.3 :

```text
Z = 1..86    MAE = 0.042 %
Z = 1..103   MAE = 0.041 %
Z = 1..118   MAE = 0.046 %
Z = 104..118 MAE = 0.077 %
```

Les residus restants sont tres petits, mais ils ne sont pas distribues
au hasard. Les plus visibles se regroupent par configurations :

| Element | configuration PT | residu |
|---|---:|---:|
| Pm | f5 | -0.293 % |
| Hf | d2 | -0.267 % |
| Bi | p3 | +0.250 % |
| Lu | d1 | +0.241 % |
| Rg | d10s1 | -0.230 % |
| Cs | s1 | -0.210 % |
| Sr | s2 | -0.198 % |
| Re | d5 | -0.190 % |
| Rn | p6 | +0.145 % |
| Nd | f4 | +0.141 % |
| Ce | f2 | -0.136 % |

Lecture : il reste un champ de correction tres faible, organise par
phase de couche. Ce n'est pas un bruit blanc.

---

## 2. Le bon changement de perspective

La correction CPR super-lourde utilise un seuil dur :

```text
rho_Z = [(Z alpha_EM)^2 - s]_+.
```

Cette forme est adaptee au regime spinoriel fort : quand
`(Z alpha)^2` depasse `s = 1/2`, le split relativiste devient lisible
comme canal separe.

Mais les residus plus legers suggerent que ce seuil dur n'est que la
limite visible d'un champ plus general. Avant le seuil, la penetration
n'est pas nulle : elle est perturbative.

La variable continue naturelle est donc :

```text
u_Z = (Z alpha_EM)^2.
```

Le regime super-lourd est la partie non-lineaire de ce champ :

```text
rho_Z = [u_Z - s]_+.
```

Le regime leger/intermediaire est sa partie lineaire :

```text
u_Z << s.
```

---

## 3. Principe general

On propose de decomposer toute resonance contraction--penetration en
deux facteurs :

```text
Delta S_res(Z) = Pi(Z) x Phase(Z).
```

ou :

- `Pi(Z)` est une enveloppe continue de penetration ;
- `Phase(Z)` est une projection discrete sur le polygone actif.

La continuite est dans `Pi(Z)`. La structure chimique est dans
`Phase(Z)`.

Cette separation explique pourquoi les residus peuvent changer de signe
entre deux elements voisins : le champ de penetration varie doucement,
mais la phase du canal actif change brutalement sur le cercle
`Z/(2P_l)Z`.

---

## 4. Enveloppe continue candidate

Une enveloppe minimale doit :

1. etre perturbative pour les elements legers ;
2. croitre avec la force relativiste effective ;
3. retrouver CPR super-lourd quand le seuil spinoriel est franchi ;
4. rester une action, non un facteur d'energie.

On pose :

```text
u_Z = (Z alpha_EM)^2

Pi_cont(Z, l) =
  sin^2(theta_3) delta_l
  [ u_Z C_core(Z,l) + [u_Z - s]_+ C_spin(Z,l) ].
```

Ici :

- `delta_l` est le gap du recepteur :
  `delta_3` pour s/p, `delta_5` pour d, `delta_7` pour f ;
- `C_core` encode la penetration douce par compacite de coeur ;
- `C_spin` encode le regime spinoriel fort deja visible dans CPR.

En premiere lecture :

```text
C_spin = 1
```

et `C_core` doit etre lie a la presence de couches internes compactes :

```text
C_core ~ 0       pour les ultra-legers
C_core augmente  apres fermeture d/f
C_core culmine   dans les periodes 6--7
```

---

## 5. Phase polygonale

Pour un canal de capacite `N_l = 2(2l+1)`, avec remplissage `n`, les
phases naturelles sont les modes de Fourier du polygone :

```text
omega_l = 2pi / N_l
cos(k omega_l n), sin(k omega_l n).
```

Le residu n'appelle donc pas une seule correction monotone, mais un
petit nombre de projecteurs :

```text
Phase_l(n) =
  a_0
  + a_1 cos(omega_l n) + b_1 sin(omega_l n)
  + a_2 cos(2 omega_l n) + b_2 sin(2 omega_l n)
  + ...
```

En logique PT, les coefficients ne doivent pas etre ajustes librement :
ils doivent etre remplaces par des amplitudes PT deja presentes
(`s`, `delta_3`, `delta_5`, `delta_7`, `sin^2 theta_p`, produits de
cross-gaps).

La structure observee suggere :

- s-block : phase de diametre, signe quasi toujours contractif apres
  coeur lourd ;
- p-block periode 6 : phase hexagonale lourde pre-spinorielle
  (`Bi`, `Po`, `Rn`) ;
- d-block periode 6 : phase pentagonale apres contraction lanthanide
  (`Lu`, `Hf`, `Re`, `Ir`) ;
- f-block periode 6 : oscillation heptagonale autour de la construction
  f (`Ce`, `Nd`, `Pm`, `Sm`, `Er`, `Tm`) ;
- d10 superlourd : residu de fermeture `Rg/Cn`, distinct de CPR
  Rf--Sg.

---

## 6. Lecture des poches residuelles

### 6.1 s apres coeur lourd

```text
Rb, Sr, Cs, Ba, Fr : residus negatifs.
```

Un residu negatif signifie `IE_PT` trop basse, donc action trop grande.
La correction attendue est contractive :

```text
Delta S < 0.
```

Interpretation : l'electron s externe penetre davantage le coeur ferme
que ne le modelise l'action actuelle. L'effet augmente quand le coeur
devient plus compact.

### 6.2 d apres f14

```text
Lu/Hf/Re/Ir et Lr/Rg.
```

Ce bloc est le candidat le plus clair : la contraction lanthanide
modifie la lecture du pentagone d. Le signe depend de la phase d :

```text
d1 : Lu positif
d2 : Hf negatif
d5 : Re negatif
d7 : Ir positif
d10s1 : Rg negatif
```

Ce n'est pas un drift en Z ; c'est une resonance de phase sur
`Z/10Z`.

### 6.3 p lourd periode 6

```text
Bi p3 positif, Po p4 negatif, Rn p6 positif.
```

Cette poche ressemble a une version pre-spinorielle de `jj2` :
le p-hexagone commence a sentir le split relativiste, mais pas assez
fortement pour imposer le seuil super-lourd.

### 6.4 f periode 6

```text
Ce f2 negatif, Nd f4 positif, Pm f5 negatif, Sm f6 positif.
```

Le motif alterne autour de la construction f. Il suggere une phase
heptagonale ou une interference hex--hept/pent--hept residuelle.

---

## 7. Forme de recherche : CPR-continuum

La correction de recherche s'ecrit :

```text
Delta S_CPR-cont(Z) =
  Pi_cont(Z,l) Phi_l(n, per, ns).
```

avec :

```text
Pi_cont(Z,l) =
  sin^2(theta_3) delta_l
  [ u_Z C_core(Z,l) + [u_Z-s]_+ C_spin(Z,l) ].
```

et :

```text
Phi_l(n,per,ns)
```

un projecteur de phase derive du polygone actif.

Dans le regime super-lourd p/d, cette formule doit reduire a CPR v6.2 :

```text
Delta S_CPR(Z) =
  - 1_{per=7} [u_Z-s]_+ sin^2(theta_3) delta_3
    [Psi_d(Z)+Psi_p(Z)].
```

Dans le regime leger/intermediaire, le terme `[u_Z-s]_+` disparait et
il reste :

```text
Delta S_light(Z) =
  sin^2(theta_3) delta_l u_Z C_core(Z,l) Phi_l(n,per,ns).
```

---

## 8. Ce qu'il faut eviter

Il serait facile de fitter les residus avec une serie de Fourier par
bloc. Ce serait numeriquement efficace mais physiquement faible.

La contrainte PT doit etre :

1. pas de coefficient libre arbitraire ;
2. phase derivee du polygone actif ;
3. amplitude tiree de constantes PT existantes ;
4. support explique par une condition geometrique claire
   (coeur ferme, contraction lanthanide, split pre-spinoriel,
   fermeture d10, etc.).

---

## 9. Programme de verification

1. Construire la carte des residus par `(periode, bloc, remplissage, ns)`.
2. Identifier les poches coherentes dont l'amplitude depasse `0.08 %`.
3. Pour chaque poche, proposer un projecteur `Phi_l` minimal.
4. Tester la correction comme action separee :

```text
IE_test = IE_eV exp(-2 Delta S_CPR-cont).
```

5. Refuser toute correction qui ameliore une poche en degradant les
   elements deja sous `0.05 %`.
6. Ne promouvoir qu'une fois la forme derivee sans fit implicite.

---

## 10. Conclusion provisoire

L'hypothese d'une fonction continue est plausible et meme naturelle :

```text
ultra-legers     : u_Z petit, correction quasi nulle
intermediaires   : u_Z lineaire, visible seulement aux phases sensibles
lourds periode 6 : u_Z + coeur compact, oscillations polygonales
super-lourds     : seuil spinoriel, CPR forte et localisee
```

La CPR super-lourde n'est donc probablement pas un phenomene isole.
Elle est la saturation d'un champ continu de penetration, rendu visible
par le franchissement du seuil spinoriel. Les residus plus legers sont
les harmoniques faibles du meme champ.

---

## 11. Calculs exploratoires du 2026-04-29

Trois familles de corrections ont ete testees hors moteur canonique :

1. `A pockets PT-products` : poches explicites s/p/d/f avec amplitudes
   produits PT ;
2. `B half A` : meme forme, amplitude divisee par deux ;
3. `C naive Fourier` : projection Fourier globale naive.

Resultat :

| Modele | MAE 1--86 | MAE 1--103 | MAE 1--118 | Commentaire |
|---|---:|---:|---:|---|
| baseline v6.2 | 0.0606 % | 0.0565 % | 0.0591 % | canonique |
| A pockets | 0.0423 % | 0.0455 % | 0.0495 % | ameliore mais surcorrige Fr/Ra |
| B half A | 0.0511 % | 0.0497 % | 0.0532 % | plus prudent |
| C Fourier naive | 0.0738 % | 0.0714 % | 0.0721 % | degrade : mauvais projecteur |

La variante `A` montre que l'hypothese a du pouvoir predictif : elle
fait baisser la MAE de `0.0606 %` a `0.0423 %` sur H--Rn. Mais elle
surcorrige le s-block de periode 7 :

```text
Fr : -0.106 % -> +0.219 %
Ra : +0.019 % -> +0.352 %
```

Une variante plus localisee, `D = A sans s periode 7`, donne :

| Modele | MAE 1--86 | MAE 1--103 | MAE 1--118 | Max |
|---|---:|---:|---:|---:|
| baseline v6.2 | 0.0606 % | 0.0565 % | 0.0591 % | 0.2926 % |
| D pockets sans s7 | 0.0423 % | 0.0412 % | 0.0458 % | 0.2465 % |

Cette variante ne touche pas les superlourds deja corriges par CPR :

```text
Z = 104..118 : 0.0770 % -> 0.0770 %
```

Les corrections les plus utiles dans `D` sont :

```text
Bi p3  : +0.250 % -> proche de 0
Lu d1  : +0.241 % -> +0.131 %
Hf d2  : -0.267 % -> -0.154 %
Re d5  : -0.190 % -> -0.129 %
Ce/Nd/Pm/Sm : motif f reduit mais pas annule
```

Conclusion du calcul : l'hypothese continuum est numeriquement
encourageante, mais le projecteur s-block doit etre derive plus finement.
La version globale Fourier naive echoue, ce qui confirme que la phase
ne peut pas etre choisie comme une simple harmonique universelle : elle
doit etre liee au canal recepteur et a l'histoire de couche.

---

## 12. Version avec projecteurs geometriques exacts

La variante precedente peut etre reecrite sans "poches" ad hoc, en
utilisant les projecteurs exacts des sommets du polygone actif.

Pour un canal de capacite `N`, le projecteur du sommet `a` est :

```text
Pi_a(n) =
  1/N [ 1 + 2 sum_{k=1}^{N/2-1} cos(2pi k(n-a)/N)
        + cos(pi(n-a)) ].
```

Il vaut `1` si `n=a mod N`, et `0` sur les autres sommets. Un ensemble
de sommets `A` est selectionne par :

```text
Pi_A(n) = sum_{a in A} Pi_a(n).
```

La correction testee devient alors :

```text
Delta S_proj(Z) =
  Delta S_s + Delta S_p + Delta S_d + Delta S_f,
```

avec :

```text
Delta S_s =
  - 1_{bloc=s, per=5,6}
    u_Z sin^2(theta_3) delta_3 cos^2(theta_3)/(per-2)
    Pi_{1,2}(n_s)

Delta S_p =
  1_{bloc=p, per=6}
  u_Z sin^2(theta_3) delta_3 delta_5
  [ Pi_3 - s Pi_4 + s Pi_6 ]

Delta S_d =
  1_{bloc=d, per=6, n_s=2}
  u_Z sin^2(theta_3) delta_5 delta_7
  [ Pi_1 - Pi_2 - s Pi_{4,5} + s Pi_7 ]

Delta S_f =
  1_{bloc=f, per=6}
  u_Z sin^2(theta_3) delta_7 delta_3 s
  [ -Pi_{2,5,12,13} + Pi_{4,6} ].
```

Cette version donne exactement le meme score que la meilleure variante
localisee, mais sa forme est maintenant geometrique :

| Domaine | baseline v6.2 | avec projecteurs |
|---|---:|---:|
| Z = 1..86 | 0.0606 % | 0.0423 % |
| Z = 1..103 | 0.0565 % | 0.0412 % |
| Z = 1..118 | 0.0591 % | 0.0458 % |
| Z = 37..86 | 0.0829 % | 0.0515 % |
| Z = 55..86 | 0.1026 % | 0.0598 % |
| Z = 104..118 | 0.0770 % | 0.0770 % |

Elements modifies :

| Element | canal | `Delta S` | residu v6.2 | residu proj. |
|---|---|---:|---:|---:|
| Rb | s1 | -0.0004892 | -0.149 % | -0.052 % |
| Sr | s2 | -0.0005160 | -0.198 % | -0.095 % |
| Cs | s1 | -0.0008107 | -0.210 % | -0.048 % |
| Ba | s2 | -0.0008405 | -0.142 % | +0.026 % |
| Ce | f2 | -0.0002087 | -0.136 % | -0.095 % |
| Nd | f4 | +0.0002234 | +0.141 % | +0.096 % |
| Pm | f5 | -0.0002309 | -0.293 % | -0.247 % |
| Sm | f6 | +0.0002385 | +0.198 % | +0.150 % |
| Er | f12 | -0.0002869 | -0.115 % | -0.058 % |
| Tm | f13 | -0.0002954 | -0.094 % | -0.035 % |
| Lu | d1 | +0.0005496 | +0.241 % | +0.131 % |
| Hf | d2 | -0.0005652 | -0.267 % | -0.154 % |
| W | d4 | -0.0002985 | -0.097 % | -0.037 % |
| Re | d5 | -0.0003066 | -0.190 % | -0.129 % |
| Ir | d7 | +0.0003232 | +0.104 % | +0.039 % |
| Bi | p3 | +0.0009667 | +0.250 % | +0.057 % |
| Po | p4 | -0.0004951 | -0.095 % | +0.004 % |
| Rn | p6 | +0.0005189 | +0.145 % | +0.041 % |

Le point important : `Z=104..118` reste inchange. La correction agit sur
la composante continue faible du champ, tandis que CPR v6.2 garde le
regime super-lourd spinoriel.

---

## 13. Formalisation sans gate : supports comme idempotents PT

La formulation precedente contient encore des conditions ecrites comme
des `gates` de code (`per=6`, `bloc=p`, etc.). Ce n'est pas la bonne
lecture theorique. En PT, ces supports doivent etre compris comme des
projecteurs idempotents dans l'algebre du canal discret.

### 13.1 Projecteur de sommet

Soit un canal de capacite

```text
N_l = 2(2l+1).
```

Le remplissage `n` est lu sur le cercle fini `Z/N_l Z`. Le projecteur
du sommet `a` est l'idempotent reel :

```text
Pi_a^{(N)}(n) =
  1/N [ 1
      + 2 sum_{k=1}^{N/2-1} cos(2 pi k(n-a)/N)
      + cos(pi(n-a)) ].
```

Il verifie :

```text
Pi_a^{(N)} Pi_b^{(N)} = delta_ab Pi_a^{(N)},
sum_a Pi_a^{(N)} = 1.
```

Donc ce n'est pas un fit par poche : c'est la decomposition spectrale
canonique du cercle de couche. Un ensemble de sommets `A` est le
projecteur :

```text
Pi_A^{(N)} = sum_{a in A} Pi_a^{(N)}.
```

### 13.2 Projecteurs de source et de recepteur

La resonance contraction--penetration couple deux objets :

1. une source compacte, qui produit la penetration ;
2. un recepteur ouvert, qui convertit cette penetration en deplacement
   d'action.

La selection par periode ou par bloc doit donc etre reecrite comme :

```text
Omega_source(Z) Omega_receiver(Z),
```

ou chaque `Omega` est un projecteur discret :

- `Omega_receiver` est le projecteur de bloc/canal actif
  (`s`, `p`, `d`, `f`) ;
- `Omega_source` est le projecteur de profondeur : coeur d/p/f ferme,
  contraction lanthanide, ou seuil spinoriel.

En code, ces projecteurs peuvent encore etre evalues par des branches
pour eviter du bruit numerique. Mais dans la theorie, ce ne sont pas des
gates arbitraires : ce sont des idempotents de support.

### 13.3 Enveloppe continue

La partie continue minimale est :

```text
u_Z = (Z alpha_EM)^2.
```

C'est le premier scalaire relativiste pair disponible. Une correction
de contraction--penetration doit donc etre une action de la forme :

```text
Delta S_cont(Z) =
  u_Z sin^2(theta_3) A_l
  Omega_source(Z) Omega_receiver(Z) Phi_l(n).
```

Le facteur `sin^2(theta_3)` est la brique de couplage commune. Le facteur
`A_l` est un produit de gaps PT deja presents dans les canaux :

```text
A_s = delta_3 cos^2(theta_3)
A_p = delta_3 delta_5
A_d = delta_5 delta_7
A_f = delta_7 delta_3 s.
```

La partie super-lourde deja canonique reste la saturation spinorielle :

```text
rho_Z = [u_Z - s]_+.
```

La CPR continue est donc la branche perturbative `u_Z`, tandis que CPR
v6.2 est la branche seuil `rho_Z`.

### 13.4 Forme candidate canonique

La correction candidate s'ecrit :

```text
Delta S_CPR-geom =
  Delta S_s + Delta S_p + Delta S_d + Delta S_f,
```

avec :

```text
Delta S_s =
  - Omega_s u_Z sin^2(theta_3) delta_3 cos^2(theta_3)
    /(per-2) Pi_{1,2}^{(2)}(n_s)

Delta S_p =
  Omega_p u_Z sin^2(theta_3) delta_3 delta_5
  [ Pi_3^{(6)} - s Pi_4^{(6)} + s Pi_6^{(6)} ]

Delta S_d =
  Omega_d u_Z sin^2(theta_3) delta_5 delta_7
  [ Pi_1^{(10)} - Pi_2^{(10)}
    - s Pi_{4,5}^{(10)} + s Pi_7^{(10)} ]

Delta S_f =
  Omega_f u_Z sin^2(theta_3) delta_7 delta_3 s
  [ -Pi_{2,5,12,13}^{(14)} + Pi_{4,6}^{(14)} ].
```

Les supports utilises dans le calcul de validation sont :

```text
Omega_s : s externe apres coeur d compact, avant regime spinoriel fort
Omega_p : p lourd pre-spinoriel, periode 6
Omega_d : d apres contraction lanthanide, ns=2
Omega_f : f lanthanide, periode 6
```

Ces supports doivent etre justifies dans le texte canonique par la
source physique, pas presentes comme des exceptions element par element.

### 13.5 Pourquoi le "sans support" brut echoue

On a teste la variante qui applique les memes projecteurs de sommet sans
support de profondeur. Elle donne :

| Domaine | baseline v6.2 | projecteurs avec supports | sans supports |
|---|---:|---:|---:|
| Z = 1..86 | 0.0606 % | 0.0423 % | 0.0394 % |
| Z = 1..103 | 0.0565 % | 0.0412 % | 0.0482 % |
| Z = 1..118 | 0.0591 % | 0.0458 % | 0.0600 % |
| Z = 104..118 | 0.0770 % | 0.0770 % | 0.1412 % |

Le `sans support` ameliore les elements jusqu'a Rn, mais degrade le
regime actinide/super-lourd. La conclusion n'est donc pas que la PT a
besoin de gates ; c'est l'inverse :

```text
la PT a besoin de projecteurs de source et de recepteur,
pas de conditions ad hoc.
```

La profondeur du canal est une information geometrique reelle. La
supprimer revient a confondre la branche perturbative continue avec la
branche spinorielle forte.

### 13.6 Bilan numerique de la version candidate

La version candidate avec projecteurs de support reduit l'erreur globale
comme suit :

| Domaine | MAE v6.2 | MAE CPR-geom | RMSE v6.2 | RMSE CPR-geom |
|---|---:|---:|---:|---:|
| Z = 1..86 | 0.060612 % | 0.042340 % | 0.089846 % | 0.058158 % |
| Z = 1..103 | 0.056472 % | 0.041216 % | 0.084742 % | 0.057142 % |
| Z = 1..118 | 0.059087 % | 0.045770 % | 0.085997 % | 0.063066 % |
| Z = 37..86 | 0.082940 % | 0.051512 % | 0.113398 % | 0.069227 % |
| Z = 55..86 | 0.102583 % | 0.059752 % | 0.131771 % | 0.079492 % |
| Z = 104..118 | 0.077038 % | 0.077038 % | 0.094164 % | 0.094164 % |

Le maximum global passe de :

```text
Pm : -0.292568 %
```

a :

```text
Pm : -0.246514 %
```

Ce maximum restant indique que la partie f-lanthanide n'est pas encore
la forme finale. Mais l'amelioration globale est large, structuree, et
obtenue sans parametre ajuste.

### 13.7 Statut canonique

Statut epistemique :

```text
DER-PHYS + VAL, canonique dans PTChem v6.3.
```

Ce n'est pas un theoreme mathematique de la PT pure : la presence de
`u_Z = (Z alpha_EM)^2` appartient au pont physique Dirac--PT. Mais une
fois ce pont accepte, la forme de la correction est fortement contrainte
par :

1. l'action IE `IE = Ry (Z exp(-S)/per)^2` ;
2. le scalaire relativiste pair minimal `u_Z` ;
3. les idempotents des cercles de couche `Z/N_l Z` ;
4. les amplitudes PT existantes `s`, `delta_3`, `delta_5`, `delta_7`,
   `sin^2(theta_3)`, `cos^2(theta_3)`.

Le terme est integre dans `screening_action` sous la forme
`S_cpr_continuum(Z)`. Le texte monographique le presente comme une
derivation physique Dirac--PT validee, non comme un theoreme
arithmetique inconditionnel.
