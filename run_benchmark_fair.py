#!/usr/bin/env python3
"""
Fair benchmark: PTC vs B3LYP/6-31G* on the SAME 1000 molecules.

Reads the fixed benchmark_1000.json (same list as Benchmark tab).
Runs B3LYP on each, compares head-to-head. Checkpoints incrementally.
"""
import sys, time, json, warnings, os, statistics
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyscf import gto, dft
from rdkit import Chem
from rdkit.Chem import AllChem

HA_TO_EV = 27.211386245988
BASIS = '6-31g*'

ATOM_SPINS = {
    1:1, 3:1, 4:0, 5:1, 6:2, 7:3, 8:2, 9:1, 10:0,
    11:1, 12:0, 13:1, 14:2, 15:3, 16:2, 17:1, 18:0,
    19:1, 20:0, 26:4, 29:1, 30:0, 35:1, 53:1,
}

_atom_cache = {}

def atom_energy(sym, Z):
    if sym in _atom_cache:
        return _atom_cache[sym]
    spin = ATOM_SPINS.get(Z, 0)
    try:
        a = gto.M(atom=f'{sym} 0 0 0', basis=BASIS, spin=spin, verbose=0)
        mf = dft.UKS(a) if spin else dft.RKS(a)
        mf.xc = 'b3lyp'
        E = mf.kernel()
        _atom_cache[sym] = E
        return E
    except Exception:
        return None

def b3lyp_atomization(smiles):
    mol_rd = Chem.MolFromSmiles(smiles)
    if mol_rd is None: return None, None
    mol_rd = Chem.AddHs(mol_rd)
    ok = AllChem.EmbedMolecule(mol_rd, AllChem.ETKDGv3())
    if ok != 0: AllChem.EmbedMolecule(mol_rd, randomSeed=42)
    if mol_rd.GetNumConformers() == 0: return None, None
    try: AllChem.MMFFOptimizeMolecule(mol_rd, maxIters=200)
    except: pass
    conf = mol_rd.GetConformer()
    atoms_str, atom_counts = [], {}
    for i in range(mol_rd.GetNumAtoms()):
        a = mol_rd.GetAtomWithIdx(i)
        sym, Z = a.GetSymbol(), a.GetAtomicNum()
        pos = conf.GetAtomPosition(i)
        atoms_str.append(f'{sym} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}')
        if sym not in atom_counts: atom_counts[sym] = (0, Z)
        atom_counts[sym] = (atom_counts[sym][0]+1, Z)
    t0 = time.time()
    try:
        mol_pyscf = gto.M(atom='; '.join(atoms_str), basis=BASIS, verbose=0)
        mf = dft.UKS(mol_pyscf) if mol_pyscf.nelectron % 2 else dft.RKS(mol_pyscf)
        mf.xc = 'b3lyp'; E_mol = mf.kernel()
        if not mf.converged: return None, None
    except: return None, None
    E_atoms = 0.0
    for sym, (count, Z) in atom_counts.items():
        Ea = atom_energy(sym, Z)
        if Ea is None: return None, None
        E_atoms += Ea * count
    D_at = (E_atoms - E_mol) * HA_TO_EV
    dt = time.time() - t0
    return (D_at, dt) if D_at > 0 else (None, None)

# ── Load fixed 1000 ──
with open('ptc/data/benchmark_1000.json') as f:
    mols = json.load(f)
print(f"Benchmark: {len(mols)} molecules (fixed list)")

# ── PTC on all 1000 ──
from ptc.topology import build_topology
from ptc.cascade import compute_D_at_cascade

t0_ptc = time.time()
for m in mols:
    try:
        topo = build_topology(m['smiles'])
        res = compute_D_at_cascade(topo)
        m['D_ptc'] = res.D_at
        m['err_ptc'] = (res.D_at - m['D_exp']) / m['D_exp'] * 100
    except:
        m['D_ptc'] = m['err_ptc'] = None
dt_ptc = time.time() - t0_ptc
ptc_ok = [m for m in mols if m.get('err_ptc') is not None]
print(f"PTC: {len(ptc_ok)}/1000 in {dt_ptc:.1f}s, MAE {statistics.mean([abs(m['err_ptc']) for m in ptc_ok]):.2f}%")

# ── B3LYP checkpoint ──
CHECKPOINT = 'benchmark_fair_checkpoint.json'
done = {}
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        done = json.load(f)
    print(f"Resuming B3LYP: {len(done)} already done")

# ── B3LYP on 1000 ──
print(f"Running B3LYP/{BASIS}...")
n_fail = 0
t_start = time.time()
for idx, m in enumerate(mols):
    name = m['name']
    if name in done:
        m['D_b3lyp'] = done[name].get('D_b3lyp')
        m['err_b3lyp'] = done[name].get('err_b3lyp')
        m['time_b3lyp'] = done[name].get('time_b3lyp')
        continue
    D_b3, dt_b3 = b3lyp_atomization(m['smiles'])
    if D_b3 is not None:
        m['D_b3lyp'] = D_b3
        m['err_b3lyp'] = (D_b3 - m['D_exp']) / m['D_exp'] * 100
        m['time_b3lyp'] = dt_b3
    else:
        m['D_b3lyp'] = m['err_b3lyp'] = m['time_b3lyp'] = None
        n_fail += 1
    done[name] = m
    n_done = len(done)
    if n_done % 20 == 0:
        both = [d for d in done.values() if d.get('err_b3lyp') is not None and d.get('err_ptc') is not None]
        if both:
            mae_b = statistics.mean([abs(d['err_b3lyp']) for d in both])
            mae_p = statistics.mean([abs(d['err_ptc']) for d in both])
            wins = sum(1 for d in both if abs(d['err_ptc']) < abs(d['err_b3lyp']))
            eta = (time.time()-t_start)/max(n_done-len([1 for d in done.values() if 'time_b3lyp' not in d or d.get('time_b3lyp') is None]),1)*(1000-n_done)/60
            print(f"  [{n_done:4d}/1000] B3LYP={mae_b:.1f}% PTC={mae_p:.1f}% wins={wins}/{len(both)} fail={n_fail}")
        with open(CHECKPOINT, 'w') as f:
            json.dump(done, f)
with open(CHECKPOINT, 'w') as f:
    json.dump(done, f)

# ── Save for DFT panel ──
both_ok = [m for m in mols if m.get('err_b3lyp') is not None and m.get('err_ptc') is not None]
b3_times = [m['time_b3lyp'] for m in both_ok if m.get('time_b3lyp')]
output = {
    'all_ptc': mols,
    'n_total': 1000,
    'n_with_b3lyp': len(both_ok),
    'n_fail_b3lyp': n_fail,
    'ptc_mae_all': statistics.mean([abs(m['err_ptc']) for m in ptc_ok]),
    'ptc_median_all': statistics.median([abs(m['err_ptc']) for m in ptc_ok]),
    'ptc_mae_overlap': statistics.mean([abs(m['err_ptc']) for m in both_ok]) if both_ok else 0,
    'b3lyp_mae_overlap': statistics.mean([abs(m['err_b3lyp']) for m in both_ok]) if both_ok else 0,
    'b3lyp_median_overlap': statistics.median([abs(m['err_b3lyp']) for m in both_ok]) if both_ok else 0,
    'ptc_time_ms': dt_ptc / len(mols) * 1000,
    'b3lyp_avg_time_s': statistics.mean(b3_times) if b3_times else 30,
    'basis': BASIS,
    'dataset': 'benchmark_1000.json (fixed, same as Benchmark tab)',
}
with open('ptc_app/dft_comparison_data.json', 'w') as f:
    json.dump(output, f, indent=2)
total_t = time.time() - t_start
print(f"\n{'='*70}")
print(f"PTC vs B3LYP/{BASIS} on 1000 molecules (same benchmark)")
print(f"{'='*70}")
if both_ok:
    wins = sum(1 for m in both_ok if abs(m['err_ptc']) < abs(m['err_b3lyp']))
    print(f"  B3LYP OK: {len(both_ok)}/1000 | failed: {n_fail}")
    print(f"  PTC MAE:   {output['ptc_mae_overlap']:.2f}%")
    print(f"  B3LYP MAE: {output['b3lyp_mae_overlap']:.2f}%")
    print(f"  PTC wins: {wins}/{len(both_ok)} ({wins/len(both_ok)*100:.0f}%)")
    print(f"  Speedup: {output['b3lyp_avg_time_s']/(output['ptc_time_ms']/1000):,.0f}x")
print(f"  Time: {total_t/60:.0f} min")
