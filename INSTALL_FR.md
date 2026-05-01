# PTC — Guide rapide d'installation et de lancement

Installer PTC en local et lancer son interface Streamlit en cinq minutes.

## Prérequis

- **Python 3.10 ou plus récent** (`python3 --version` pour vérifier)
- **Git** (`git --version` pour vérifier)
- ~1 Go d'espace disque libre (essentiellement pour `pyscf` et ses dépendances)

## 1. Cloner le dépôt

```bash
git clone https://github.com/Igrekess/PT_CHEMISTRY.git
cd PT_CHEMISTRY
```

## 2. Créer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Sur Windows, utiliser `.venv\Scripts\activate` au lieu de `source .venv/bin/activate`.

Le prompt du terminal doit maintenant commencer par `(.venv)`.

## 3. Installer les dépendances

Le `requirements.txt` du dépôt est minimal (juste `numpy` et `scipy`). Pour l'UI complète et toutes les observables, il en faut un peu plus :

```bash
pip install --upgrade pip
pip install numpy>=1.24 scipy>=1.10 streamlit rdkit pyscf matplotlib plotly
```

Sur Apple Silicon (M1/M2/M3), la compilation de `pyscf` peut prendre 1 à 2 minutes. En cas d'échec, essayer :

```bash
pip install --no-cache-dir pyscf
```

ou via conda :

```bash
conda install -c conda-forge pyscf
```

## 4. Lancer PTC

```bash
python launch_ptc.py
```

Le lanceur :

- configure correctement le `PYTHONPATH`
- trouve un port libre
- démarre `streamlit run ptc_app/app.py` en sous-processus
- ouvre automatiquement le navigateur sur `http://localhost:<port>`

Le terminal affiche :

```
Starting PTC — Persistence Theory Chemistry
  App: ptc_app/app.py
  Port: 51234
  Opening browser at http://localhost:51234
PTC is running. Close this window to stop the server.
```

L'interface a 15 onglets : tableau périodique, atome, liaison, molécule (par SMILES), benchmark, RMN, aromaticité, et d'autres.

## 5. Arrêter le serveur

`Ctrl+C` dans le terminal, ou fermer la fenêtre. Le sous-processus Streamlit s'arrête avec le lanceur.

## 6. Lancements suivants

Une fois le venv créé, plus besoin de réinstaller :

```bash
cd PT_CHEMISTRY
source .venv/bin/activate
python launch_ptc.py
```

Pour récupérer la dernière version depuis GitHub avant de lancer :

```bash
cd PT_CHEMISTRY
git pull
source .venv/bin/activate
python launch_ptc.py
```

## Figer l'environnement pour la reproductibilité

Pour geler les versions exactes installées :

```bash
pip freeze > requirements-pinned.txt
```

On peut ensuite recréer le même environnement sur une autre machine avec `pip install -r requirements-pinned.txt`.

## Dépannage

| Symptôme | Cause / Solution |
|---|---|
| `ModuleNotFoundError: streamlit` | Le venv n'est pas activé — refaire `source .venv/bin/activate` (le prompt doit afficher `(.venv)`). |
| Erreur d'import RDKit sur Apple Silicon | Utiliser le wheel moderne : `pip uninstall rdkit-pypi rdkit; pip install rdkit`. |
| Échec de compilation PySCF | Installer via conda (`conda install -c conda-forge pyscf`), ou s'en passer si les onglets LCAO / RMN ne sont pas nécessaires. |
| Streamlit s'ouvre mais la page est blanche | Recharger en force (Cmd/Ctrl + Shift + R). Le premier chargement peut prendre quelques secondes. |
| Port déjà utilisé | Le lanceur choisit automatiquement un port libre ; si ça persiste, redémarrer le terminal. |

## Exécuter la suite de tests (optionnel)

Pour vérifier que l'installation est propre :

```bash
python -m pytest ptc/tests/ -x -q
```

Une installation propre passe ~700+ tests en moins d'une minute.

## Lancer un benchmark en CLI sans l'UI

Pour calculer D_at sur les 1000 molécules ATcT sans démarrer l'interface :

```bash
python -c "
import json, sys, time
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
from ptc.api import Molecule
with open('ptc/data/benchmark_1000.json') as f:
    bench = json.load(f)
t0 = time.time()
errs = []
for entry in bench:
    m = Molecule(entry['smiles'])
    errs.append(abs(m.D_at - entry['D_exp']) / entry['D_exp'] * 100)
print(f'{len(errs)} mol en {time.time()-t0:.1f}s — MAE {sum(errs)/len(errs):.2f}%')
"
```

Sortie attendue : `1000 mol en 3.0s — MAE 3.20%` (jeu complet ; ≈1.81 % sur les molécules main-group seules).
