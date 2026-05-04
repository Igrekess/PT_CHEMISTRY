#!/usr/bin/env python3
"""
B3LYP/def2-TZVP single-point benchmark on small ATcT molecules (n_atoms <= 4).

Self-contained: requires ONLY pyscf and numpy. No RDKit, no internet.

Methodology:
  * Geometries are FROZEN (read from input_n4.json) — identical to the ones
    used in the B3LYP/6-31G* run, so the comparison isolates the basis-set
    effect.
  * Functional: B3LYP (Becke 1993, Lee-Yang-Parr).
  * Basis    : def2-TZVP (Weigend & Ahlrichs 2005), supports H-Rn natively.
  * SCF      : tight (1e-9), max 200 cycles, level shift, Newton fallback.
  * ZPE      : recycled from B3LYP/6-31G* (Scott-Radom 0.9806 already
               applied) — see ZPE_NOTE in input_n4.json. Differences
               between basis sets are typically <2% on ZPE, negligible
               next to the D_at error budget.
  * Output   : results_def2tzvp_n4.json (resumable via --resume).

Usage:
  python run_def2tzvp.py
  python run_def2tzvp.py --resume          # continue an interrupted run
  python run_def2tzvp.py --limit 10        # debug: first 10 molecules
  python run_def2tzvp.py --workers 4       # NOT implemented; run single-process

Expected runtime:
  ~3-8 hours on a modern desktop (varies with how many large 4-atom systems
  the SCF struggles with). All outputs are checkpointed after every
  molecule.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

HA_TO_EV = 27.211386245988
BASIS = "def2-tzvp"
FUNCTIONAL = "b3lyp"
RUN_CONFIG_VERSION = 1

# ---------------------------------------------------------------------------
# Atomic spin multiplicities (PySCF convention: spin = N_alpha - N_beta)
# Mirrors run_b3lyp_atct.py so isolated-atom energies are identical setup.
# ---------------------------------------------------------------------------
ATOM_SPINS = {
    1: 1,   3: 1,   4: 0,   5: 1,   6: 2,   7: 3,   8: 2,   9: 1,  10: 0,
    11: 1, 12: 0,  13: 1,  14: 2,  15: 3,  16: 2,  17: 1,  18: 0,
    19: 1, 20: 0,
    21: 1, 22: 2,  23: 3,  24: 6,  25: 5,  26: 4,  27: 3,  28: 2,
    29: 1, 30: 0,
    35: 1, 53: 1,
    33: 3, 34: 2,
}

ELEMENT_Z = {
    "H": 1,  "Li": 3, "Be": 4, "B": 5,  "C": 6,  "N": 7,  "O": 8,
    "F": 9,  "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27,
    "Ni": 28, "Cu": 29, "Zn": 30,
    "As": 33, "Se": 34, "Br": 35, "I": 53,
}

ATOM_CACHE: dict[str, float] = {}


def real_number(value, *, label=None, out=None, imag_tol=1e-6):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, complex):
        imag = float(value.imag)
        if abs(imag) > imag_tol and label and out is not None:
            out[f"{label}_imag"] = imag
        value = value.real
    return float(value)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_results(out_path: Path, payload: dict) -> None:
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(json_ready(payload), f, indent=1)
        f.write("\n")
    os.replace(tmp_path, out_path)


def load_previous_results(out_path: Path) -> list[dict]:
    if not out_path.exists():
        return []
    try:
        with open(out_path) as f:
            return json.load(f).get("results", [])
    except Exception as e:
        print(f"warn: couldn't parse existing {out_path} ({e})")
        return []


def atom_energy(symbol: str) -> float:
    """B3LYP/def2-TZVP energy for a single isolated atom (eV).

    Cached on first call. Uses UKS if spin > 0.
    """
    from pyscf import gto, dft
    if symbol in ATOM_CACHE:
        return ATOM_CACHE[symbol]
    Z = ELEMENT_Z.get(symbol)
    if Z is None:
        raise ValueError(f"Unknown element symbol: {symbol}")
    spin = ATOM_SPINS.get(Z, 0)
    mol = gto.M(
        atom=f"{symbol} 0 0 0",
        basis=BASIS,
        spin=spin,
        charge=0,
        verbose=0,
        unit="Bohr",
    )
    mf = dft.UKS(mol)
    mf.xc = FUNCTIONAL
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.level_shift = 0.2
    e_ha = mf.kernel()
    if not getattr(mf, "converged", False):
        # try second-order SCF
        mf2 = mf.newton()
        e_ha = mf2.kernel()
        if not getattr(mf2, "converged", False):
            raise RuntimeError(f"atom {symbol} SCF did not converge in def2-TZVP")
    e_eV = real_number(e_ha * HA_TO_EV)
    ATOM_CACHE[symbol] = e_eV
    return e_eV


def molecule_energy(atoms, coords, charge=0, spin=0):
    """B3LYP/def2-TZVP single-point electronic energy (eV) on frozen geometry."""
    from pyscf import gto, dft
    atom_str = "\n".join(
        f"{a} {x:.6f} {y:.6f} {z:.6f}" for a, (x, y, z) in zip(atoms, coords)
    )
    mol = gto.M(
        atom=atom_str,
        basis=BASIS,
        spin=spin,
        charge=charge,
        verbose=0,
        unit="Angstrom",
    )
    if spin > 0:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = FUNCTIONAL
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.level_shift = 0.1
    e_ha = mf.kernel()
    if not getattr(mf, "converged", False):
        mf2 = mf.newton()
        e_ha = mf2.kernel()
        if not getattr(mf2, "converged", False):
            raise RuntimeError("molecular SCF did not converge in def2-TZVP")
    return real_number(e_ha * HA_TO_EV)


def compute_one(entry: dict) -> dict:
    name = entry["name"]
    t0 = time.time()
    out = {
        "name": name,
        "smiles": entry["smiles"],
        "category": entry.get("category", "?"),
        "n_atoms": entry["n_atoms"],
        "D_exp": entry["D_exp"],
        "atoms": entry["atoms"],
        "charge": entry["charge"],
        "spin": entry["spin"],
        "geometry_source": entry["geometry_source"],
    }
    try:
        e_mol_eV = molecule_energy(
            entry["atoms"], entry["coords_angstrom"],
            charge=entry["charge"], spin=entry["spin"],
        )
        e_atoms = sum(atom_energy(sym) for sym in entry["atoms"])
        D_elec_eV = e_atoms - e_mol_eV
        zpe_eV = float(entry.get("zpe_eV_recycled") or 0.0)
        D_at_eV = D_elec_eV - zpe_eV
        err_eV = D_at_eV - entry["D_exp"]
        out.update({
            "E_mol_eV":    e_mol_eV,
            "E_atoms_eV":  e_atoms,
            "ZPE_eV":      zpe_eV,
            "ZPE_source":  "recycled_from_b3lyp_631gs",
            "D_elec_eV":   D_elec_eV,
            "D_def2tzvp_eV": D_at_eV,
            "err_eV":      err_eV,
            "rel_err_pct": abs(err_eV) / entry["D_exp"] * 100.0
                          if entry["D_exp"] > 0 else None,
            "status":      "ok",
        })
    except Exception as e:
        out["status"] = "fail"
        out["error_msg"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
    finally:
        out["time_s"] = round(time.time() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="input_n4.json",
                    help="Frozen-geometry input JSON (default: input_n4.json)")
    ap.add_argument("--out", default="results_def2tzvp_n4.json",
                    help="Output JSON (resumable)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip molecules already done in --out")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N molecules (debug)")
    ap.add_argument("--checkpoint", type=int, default=1,
                    help="Save every N completed molecules (default 1)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    in_path = (here / args.input) if not Path(args.input).is_absolute() else Path(args.input)
    out_path = (here / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    if not in_path.exists():
        sys.exit(f"missing input: {in_path}")
    with open(in_path) as f:
        data = json.load(f)
    mols = data["molecules"]
    if args.limit:
        mols = mols[:args.limit]
    print(f"Loaded {len(mols)} molecules from {in_path}")

    # Resume: pre-fill results for already-done molecules
    done_keys: set[str] = set()
    results: list[dict] = []
    if args.resume:
        prev = load_previous_results(out_path)
        for r in prev:
            if r.get("status") in {"ok", "fail"}:
                done_keys.add(f"{r['name']}|{r.get('smiles','')}")
                results.append(r)
        print(f"Resuming: {len(results)} previous rows kept")

    t_start = time.time()

    def payload():
        return {
            "method": f"{FUNCTIONAL.upper()}/{BASIS}",
            "run_config_version": RUN_CONFIG_VERSION,
            "zpe_source": "recycled_from_b3lyp_631gs (Scott-Radom 0.9806 applied)",
            "geometry_source": "frozen from B3LYP/6-31G* run (input_n4.json)",
            "elapsed_s": time.time() - t_start,
            "n_done": len(results),
            "results": results,
        }

    if args.resume and results:
        save_results(out_path, payload())

    n_new = 0
    try:
        for i, entry in enumerate(mols, 1):
            key = f"{entry['name']}|{entry['smiles']}"
            if key in done_keys:
                continue
            print(f"[{i:4d}/{len(mols)}] {entry['name'][:30]:30}  ", end="", flush=True)
            r = compute_one(entry)
            results.append(r)
            n_new += 1
            if r["status"] == "ok":
                print(f"D={r['D_def2tzvp_eV']:7.3f}  err={r['err_eV']:+.3f}  "
                      f"({r['time_s']:5.1f}s)")
            else:
                print(f"FAIL: {r.get('error_msg','?')[:60]}")
            if n_new % max(1, args.checkpoint) == 0:
                save_results(out_path, payload())
    except KeyboardInterrupt:
        save_results(out_path, payload())
        print(f"\nInterrupted; saved {len(results)} rows to {out_path}")
        raise SystemExit(130)

    save_results(out_path, payload())

    ok = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] == "fail"]
    print(f"\n=== Done in {time.time()-t_start:.1f}s ===")
    print(f"  ok      : {len(ok)}")
    print(f"  failed  : {len(fail)}")
    if ok:
        rels = [r["rel_err_pct"] for r in ok if r["rel_err_pct"] is not None]
        errs = [abs(r["err_eV"]) for r in ok]
        mae = sum(errs)/len(errs)
        print(f"  MAE     : {mae:.4f} eV ({mae*23.0605:.2f} kcal/mol)")
        if rels:
            print(f"  MAE %   : {sum(rels)/len(rels):.3f} %")


if __name__ == "__main__":
    main()
