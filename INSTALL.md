# PTC — Quick install & launch guide

Install PTC locally and launch its Streamlit UI in five minutes.

## Prerequisites

- **Python 3.10 or later** (`python3 --version` to check)
- **Git** (`git --version` to check)
- ~1 GB free disk space (mostly for `pyscf` and its dependencies)

## 1. Clone the repository

```bash
git clone https://github.com/Igrekess/PT_CHEMISTRY.git
cd PT_CHEMISTRY
```

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows use `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

The shell prompt should now start with `(.venv)`.

## 3. Install dependencies

The repo's `requirements.txt` is minimal (just `numpy` and `scipy`). For the full UI and all observables you need a bit more:

```bash
pip install --upgrade pip
pip install numpy>=1.24 scipy>=1.10 streamlit rdkit pyscf matplotlib plotly
```

On Apple Silicon (M1/M2/M3), `pyscf` may take 1–2 minutes to compile. If it fails, try:

```bash
pip install --no-cache-dir pyscf
```

or via conda:

```bash
conda install -c conda-forge pyscf
```

## 4. Launch PTC

```bash
python launch_ptc.py
```

The launcher:

- sets `PYTHONPATH` correctly
- finds a free port
- starts `streamlit run ptc_app/app.py` as a subprocess
- automatically opens your browser at `http://localhost:<port>`

You should see in the terminal:

```
Starting PTC — Persistence Theory Chemistry
  App: ptc_app/app.py
  Port: 51234
  Opening browser at http://localhost:51234
PTC is running. Close this window to stop the server.
```

The UI has 15 tabs: periodic table, atom, bond, molecule (by SMILES), benchmark, NMR, aromaticity, and more.

## 5. Stop the server

`Ctrl+C` in the terminal, or close the window. The Streamlit subprocess stops with the launcher.

## 6. Subsequent runs

Once the venv exists, no reinstall needed:

```bash
cd PT_CHEMISTRY
source .venv/bin/activate
python launch_ptc.py
```

To pull the latest version from GitHub before launching:

```bash
cd PT_CHEMISTRY
git pull
source .venv/bin/activate
python launch_ptc.py
```

## Pinning the environment for reproducibility

To freeze the exact versions you installed:

```bash
pip freeze > requirements-pinned.txt
```

You can then recreate the same environment on another machine with `pip install -r requirements-pinned.txt`.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ModuleNotFoundError: streamlit` | Virtualenv not activated — re-run `source .venv/bin/activate` (prompt must show `(.venv)`). |
| RDKit import error on Apple Silicon | Use the modern wheel: `pip uninstall rdkit-pypi rdkit; pip install rdkit`. |
| PySCF compile failure | Install via conda (`conda install -c conda-forge pyscf`), or skip if you don't need the LCAO / NMR tabs. |
| Streamlit opens but page is blank | Hard-refresh the browser (Cmd/Ctrl + Shift + R). The first load can take a few seconds. |
| Port already in use | The launcher picks a free port automatically; if it still fails, restart your terminal. |

## Run the test suite (optional)

To verify the install ran clean:

```bash
python -m pytest ptc/tests/ -x -q
```

A clean install passes ~700+ tests in under a minute.

## Run a single benchmark from the CLI

To compute D_at on the 1000-molecule ATcT benchmark without launching the UI:

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
print(f'{len(errs)} mol in {time.time()-t0:.1f}s — MAE {sum(errs)/len(errs):.2f}%')
"
```

Expected output: `1000 mol in 3.0s — MAE 3.20%` (full set; ≈1.81 % on main-group only).
