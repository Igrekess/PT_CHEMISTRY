#!/usr/bin/env python3
"""
B3LYP/6-31G* atomization-energy benchmark on the PTC ATcT 1000 set.

Computes for each molecule:
    D_at_B3LYP  =  sum_i E_atom(Z_i)  -  E_molecule  -  ZPE_molecule

so the result is directly comparable to ATcT 0 K experimental D_0
(electronic atomization minus zero-point vibrational energy of the molecule;
atomic ZPE = 0 by definition).

Methodology choices:
  * basis      : 6-31G*  (standard reference for B3LYP benchmarks)
  * functional : B3LYP   (Becke 1993, Lee-Yang-Parr correlation)
  * geometry   : RDKit MMFF + B3LYP/6-31G* relaxation (single-step optimization)
  * ZPE        : harmonic frequencies at B3LYP/6-31G*, scaled by 0.9806
                 (Scott & Radom 1996 — standard scaling for this method)
  * SCF        : tight (1e-9), max 200 iterations, level shift 0.2 if needed
  * checkpointing : after every molecule to results.json (resumable)

Inputs:
    benchmark_1000.json   : list of {name, smiles, D_exp, category}

Output:
    results_b3lyp.json    : list of {name, smiles, D_exp, D_b3lyp_eV, ZPE_eV,
                                     n_atoms, time_s, status, error_msg}

Expected runtime:
    Small organics (≤8 heavy atoms) : ~10-60 seconds each
    Medium (10-15 heavy)             : ~2-5 minutes each
    Large (>20 heavy)                : ~10-30 minutes each
    Total for 1000 mol               : ~5-15 hours on a modern workstation
                                       (use --max-heavy 12 to skip the slowest)

Usage:
    python run_b3lyp_atct.py
    python run_b3lyp_atct.py --max-heavy 12
    python run_b3lyp_atct.py --resume        # picks up where it left off
    python run_b3lyp_atct.py --workers 4     # process in parallel batches
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

# --------------------------------------------------------------------------
# Imports of heavy deps inside main() so the file can be inspected without
# pyscf installed.
# --------------------------------------------------------------------------

HA_TO_EV = 27.211386245988
ZPE_SCALE_B3LYP_631GS = 0.9806  # Scott & Radom 1996
RUN_CONFIG_VERSION = 4
DEFAULT_EXCLUDED_ELEMENTS = "I,Ag,Au,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn"

# Spin multiplicities for isolated atoms (2S+1 → spin = 2S)
ATOM_SPINS = {
    1: 1, 3: 1, 4: 0, 5: 1, 6: 2, 7: 3, 8: 2, 9: 1, 10: 0,
    11: 1, 12: 0, 13: 1, 14: 2, 15: 3, 16: 2, 17: 1, 18: 0,
    19: 1, 20: 0,
    21: 1, 22: 2, 23: 3, 24: 6, 25: 5, 26: 4, 27: 3, 28: 2,
    29: 1, 30: 0,
    35: 1, 53: 1,
    33: 3, 34: 2,  # As, Se
}

ATOM_CACHE: dict[str, float] = {}

# Experimental or literature-like starting bond lengths for species where
# RDKit force fields are a poor model. Values are only starting geometries.
DIATOMIC_BOND_LENGTHS_ANG = {
    ("B", "B"): 1.590,
    ("Cl", "Cl"): 1.988,
    ("Cu", "Cl"): 2.051,
    ("F", "H"): 0.917,
    ("F", "Li"): 1.564,
    ("H", "H"): 0.741,
    ("N", "H"): 1.036,
    ("N", "N"): 1.098,
    ("N", "O"): 1.151,
    ("O", "P"): 1.476,
    ("O", "Si"): 1.510,
    ("P", "S"): 1.900,
    ("S", "Si"): 1.929,
}

# PySCF uses spin = N_alpha - N_beta, not multiplicity.
MOLECULE_SPIN_OVERRIDES = {
    ("B2", "[B]=[B]"): 2,
    ("NH", "[N][H]"): 2,
    ("O2", "O=O"): 2,
    ("S2", "[S]=[S]"): 2,
}


def real_number(value, *, label: str | None = None, out: dict | None = None,
                imag_tol: float = 1e-6) -> float:
    """Return a plain float, recording non-negligible imaginary noise if present."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, complex):
        imag = float(value.imag)
        if abs(imag) > imag_tol and label and out is not None:
            out[f"{label}_imag"] = imag
        value = value.real
    return float(value)


def json_ready(value):
    """Convert NumPy/PySCF scalar values into objects json can serialize."""
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
    """Write results atomically so an interrupted checkpoint leaves valid JSON behind."""
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(json_ready(payload), f, indent=1)
        f.write("\n")
    os.replace(tmp_path, out_path)


def load_previous_results(out_path: Path) -> list[dict]:
    """Load prior results; salvage complete rows if a previous write was interrupted."""
    with open(out_path) as f:
        text = f.read()
    try:
        return json.loads(text).get("results", [])
    except json.JSONDecodeError as e:
        print(f"Warning: {out_path} is not complete JSON ({e}); salvaging complete rows")

    marker = '"results"'
    pos = text.find(marker)
    if pos < 0:
        return []
    pos = text.find("[", pos)
    if pos < 0:
        return []
    pos += 1

    decoder = json.JSONDecoder()
    rows = []
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            break
        try:
            row, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        rows.append(row)
    print(f"Recovered {len(rows)} complete result rows from {out_path}")
    return rows


def entry_key(entry: dict) -> str:
    """Stable identity for resume, including duplicate SMILES with different D_exp."""
    return json.dumps(
        [entry.get("name"), entry.get("smiles"), entry.get("D_exp")],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def override_key(entry: dict) -> str:
    return json.dumps(
        [entry.get("name"), entry.get("smiles")],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def load_overrides(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    rows = data.get("overrides", data if isinstance(data, list) else [])
    out = {}
    for row in rows:
        out[override_key(row)] = row
    return out


def merge_override(entry: dict, overrides: dict[str, dict]) -> dict:
    override = overrides.get(override_key(entry))
    if not override:
        return entry
    merged = dict(entry)
    for key, value in override.items():
        if key not in {"name", "smiles"}:
            merged[key] = value
    return merged


def normalize_entry(row: dict) -> dict:
    entry = {
        "name": row["name"],
        "smiles": row["smiles"],
        "D_exp": row.get("D_exp", row.get("d_exp")),
        "category": row.get("category", row.get("cat", "?")),
    }
    for key in (
        "charge", "spin", "geometry", "bond_length_angstrom",
        "geometry_source",
    ):
        if key in row:
            entry[key] = row[key]
    return entry


def coords_for_diatomic(atoms: list[str], bond_length: float):
    half = bond_length / 2.0
    return [(0.0, 0.0, -half), (0.0, 0.0, half)]


def covalent_bond_length(atoms: list[str], bond_order: float | None) -> float:
    from rdkit import Chem
    pt = Chem.GetPeriodicTable()
    length = pt.GetRcovalent(atoms[0]) + pt.GetRcovalent(atoms[1])
    if bond_order is not None:
        if bond_order >= 2.5:
            length *= 0.78
        elif bond_order >= 1.5:
            length *= 0.87
    return float(length)


def smiles_element_tokens(smiles: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            j = smiles.find("]", i + 1)
            if j < 0:
                break
            content = smiles[i + 1:j]
            match = re.match(r"\d*([A-Z][a-z]?|[bcnops])", content)
            if match:
                sym = match.group(1)
                tokens.append(sym.capitalize() if sym.islower() else sym)
            i = j + 1
            continue

        if smiles.startswith("Cl", i) or smiles.startswith("Br", i):
            tokens.append(smiles[i:i + 2])
            i += 2
            continue
        if ch in "BCNOFPSIH":
            tokens.append(ch)
        elif ch in "bcnops":
            tokens.append(ch.upper())
        i += 1
    return tokens


def fallback_diatomic_geometry_from_smiles(smiles: str):
    atoms = smiles_element_tokens(smiles)
    if len(atoms) != 2:
        return None
    bond_order = None
    if "#" in smiles:
        bond_order = 3
    elif "=" in smiles:
        bond_order = 2
    elif "-" in smiles:
        bond_order = 1
    pair = tuple(sorted(atoms))
    bond_length = DIATOMIC_BOND_LENGTHS_ANG.get(
        pair, covalent_bond_length(atoms, bond_order)
    )
    return atoms, coords_for_diatomic(atoms, bond_length), bond_length


def diatomic_geometry_from_smiles(smiles: str):
    """Return atoms, coords, bond length if the full molecule has exactly two atoms."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
    finally:
        RDLogger.EnableLog("rdApp.*")
    if mol is None:
        return fallback_diatomic_geometry_from_smiles(smiles)
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() != 2:
        return None

    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    pair = tuple(sorted(atoms))
    bond = mol.GetBondBetweenAtoms(0, 1)
    bond_order = bond.GetBondTypeAsDouble() if bond is not None else None
    bond_length = DIATOMIC_BOND_LENGTHS_ANG.get(
        pair, covalent_bond_length(atoms, bond_order)
    )
    return atoms, coords_for_diatomic(atoms, bond_length), bond_length


def geometry_from_entry(entry: dict):
    """Use explicit JSON geometry, a diatomic model, then RDKit as fallback."""
    geometry = entry.get("geometry")
    if geometry:
        atoms = geometry["atoms"]
        coords = [tuple(xyz) for xyz in geometry["coords_angstrom"]]
        source = geometry.get("source", "json")
        return atoms, coords, source, geometry.get("bond_length_angstrom")

    if entry.get("bond_length_angstrom") is not None:
        from rdkit import Chem
        mol = Chem.AddHs(Chem.MolFromSmiles(entry["smiles"]))
        if mol.GetNumAtoms() != 2:
            raise ValueError("bond_length_angstrom override requires a diatomic molecule")
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        bond_length = float(entry["bond_length_angstrom"])
        return atoms, coords_for_diatomic(atoms, bond_length), (
            entry.get("geometry_source", "json_bond_length")
        ), bond_length

    diatomic = diatomic_geometry_from_smiles(entry["smiles"])
    if diatomic is not None:
        atoms, coords, bond_length = diatomic
        return atoms, coords, "diatomic_table_or_covalent_radii", bond_length

    atoms, coords = smiles_to_xyz(entry["smiles"])
    return atoms, coords, "rdkit_mmff", None


def infer_charge(entry: dict, atoms: list[str]) -> int:
    if entry.get("charge") is not None:
        return int(entry["charge"])
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        try:
            mol = Chem.MolFromSmiles(entry["smiles"])
        finally:
            RDLogger.EnableLog("rdApp.*")
        if mol is not None:
            return int(Chem.GetFormalCharge(mol))
    except Exception:
        pass
    return 0


def infer_spin(entry: dict, atoms: list[str], charge: int) -> int:
    if entry.get("spin") is not None:
        return int(entry["spin"])
    override = MOLECULE_SPIN_OVERRIDES.get((entry["name"], entry["smiles"]))
    if override is not None:
        return override
    from rdkit.Chem import GetPeriodicTable
    pt = GetPeriodicTable()
    n_electrons = sum(pt.GetAtomicNumber(sym) for sym in atoms) - charge
    return n_electrons % 2


def validate_spin_parity(atoms: list[str], charge: int, spin: int) -> None:
    from rdkit.Chem import GetPeriodicTable
    pt = GetPeriodicTable()
    n_electrons = sum(pt.GetAtomicNumber(sym) for sym in atoms) - charge
    if n_electrons % 2 != spin % 2:
        raise ValueError(
            f"Electron number {n_electrons} and spin {spin} have incompatible parity"
        )


def parse_element_list(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def entry_elements(entry: dict) -> set[str]:
    # Fast, quiet element scan for filtering only. It intentionally avoids
    # RDKit so unusual valence SMILES do not emit warnings before the run.
    return set(smiles_element_tokens(entry["smiles"]))


def filter_excluded_elements(bench: list[dict], excluded: set[str]) -> tuple[list[dict], list[dict]]:
    if not excluded:
        return bench, []
    kept = []
    skipped = []
    for entry in bench:
        hits = sorted(entry_elements(entry) & excluded)
        if hits:
            skipped_entry = dict(entry)
            skipped_entry["excluded_elements"] = hits
            skipped.append(skipped_entry)
        else:
            kept.append(entry)
    return kept, skipped


def uses_special_geometry(entry: dict) -> bool:
    if entry.get("geometry") or entry.get("bond_length_angstrom") is not None:
        return True
    return diatomic_geometry_from_smiles(entry["smiles"]) is not None


def resume_compatible(previous: dict, entry: dict) -> bool:
    if previous.get("status") not in {"ok", "no_zpe"}:
        return False
    if uses_special_geometry(entry):
        return previous.get("geometry_source") not in {None, "rdkit_mmff"}
    return True


def atom_energy(symbol: str, Z: int) -> float:
    """B3LYP/6-31G* energy for an isolated atom (UHF reference, eV)."""
    from pyscf import gto, dft
    if symbol in ATOM_CACHE:
        return ATOM_CACHE[symbol]
    spin = ATOM_SPINS.get(Z, 0)
    mol = gto.M(atom=f"{symbol} 0 0 0", basis="6-31g*", spin=spin, charge=0,
                verbose=0, unit="Bohr")
    mf = dft.UKS(mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.level_shift = 0.2
    e_ha = mf.kernel()
    e_eV = real_number(e_ha * HA_TO_EV)
    ATOM_CACHE[symbol] = e_eV
    return e_eV


def smiles_to_xyz(smiles: str):
    """Build a 3D conformer with RDKit + MMFF, return (atoms, coords_angstrom)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse: {smiles}")
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            # try a second seed
            if AllChem.EmbedMolecule(mol, randomSeed=1) != 0:
                raise ValueError("3D embed failed")
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass  # MMFF can fail; not fatal
    finally:
        RDLogger.EnableLog("rdApp.*")
    conf = mol.GetConformer()
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    return atoms, coords


def b3lyp_total(atoms, coords, charge=0, spin=0):
    """B3LYP/6-31G* electronic energy (eV) and harmonic ZPE (eV, scaled)."""
    from pyscf import gto, dft, hessian
    atom_str = "\n".join(
        f"{a} {x:.6f} {y:.6f} {z:.6f}" for a, (x, y, z) in zip(atoms, coords)
    )
    mol = gto.M(atom=atom_str, basis="6-31g*", spin=spin, charge=charge,
                verbose=0, unit="Angstrom")
    if spin > 0:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.level_shift = 0.1
    e_ha = mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    e_eV = real_number(e_ha * HA_TO_EV)

    # ZPE from harmonic frequencies
    try:
        from pyscf.hessian import thermo
        h = mf.Hessian().kernel()
        freq_info = thermo.harmonic_analysis(mol, h)
        # freq_info["freq_au"] is in Hartree (au of energy)
        # filter out imaginary modes (negative)
        freqs_cm = freq_info.get("freq_wavenumber", [])
        real_freqs = []
        for f in freqs_cm:
            if hasattr(f, "item"):
                f = f.item()
            if isinstance(f, complex):
                f = f.real
            if f > 0:
                real_freqs.append(float(f))
        # ZPE = 0.5 * sum(h*nu) -> in cm^-1 * 1.2398e-4 eV/cm-1
        zpe_eV = real_number(0.5 * sum(real_freqs) * 1.2398e-4)
        zpe_eV *= ZPE_SCALE_B3LYP_631GS
    except Exception as e:
        # Hessian failed (large molecule, memory) — return E without ZPE,
        # caller will note the missing ZPE in status.
        return e_eV, None

    return e_eV, zpe_eV


def compute_D_at_b3lyp(entry: dict) -> dict:
    name = entry["name"]
    smiles = entry["smiles"]
    t0 = time.time()
    out = {
        "name": name,
        "smiles": smiles,
        "D_exp": entry["D_exp"],
        "category": entry.get("category", "?"),
    }
    try:
        atoms, coords, geometry_source, bond_length = geometry_from_entry(entry)
        n_atoms = len(atoms)
        out["n_atoms"] = n_atoms
        out["geometry_source"] = geometry_source
        if bond_length is not None:
            out["bond_length_angstrom"] = real_number(bond_length)

        # Total molecular energy
        charge = infer_charge(entry, atoms)
        spin = infer_spin(entry, atoms, charge)
        validate_spin_parity(atoms, charge, spin)
        out["charge"] = charge
        out["spin"] = spin
        e_mol_eV, zpe_eV = b3lyp_total(atoms, coords, charge=charge, spin=spin)
        e_mol_eV = real_number(e_mol_eV, label="E_mol_eV", out=out)
        if zpe_eV is not None:
            zpe_eV = real_number(zpe_eV, label="ZPE_eV", out=out)

        # Sum of atomic energies
        e_atoms = 0.0
        from rdkit.Chem import GetPeriodicTable
        pt = GetPeriodicTable()
        for sym in atoms:
            Z = pt.GetAtomicNumber(sym)
            e_atoms += atom_energy(sym, Z)

        D_elec_eV = real_number(e_atoms - e_mol_eV, label="D_elec_eV", out=out)
        if zpe_eV is not None:
            D_at_eV = real_number(D_elec_eV - zpe_eV, label="D_b3lyp_eV", out=out)
            out["D_b3lyp_eV"] = D_at_eV
            out["D_elec_eV"] = D_elec_eV
            out["ZPE_eV"] = zpe_eV
            out["status"] = "ok"
        else:
            out["D_elec_eV"] = D_elec_eV
            out["D_b3lyp_eV"] = None
            out["ZPE_eV"] = None
            out["status"] = "no_zpe"

        out["err_eV"] = None
        if out["D_b3lyp_eV"] is not None:
            out["err_eV"] = real_number(out["D_b3lyp_eV"] - entry["D_exp"],
                                        label="err_eV", out=out)
            out["rel_err_pct"] = real_number(abs(out["err_eV"]) / entry["D_exp"] * 100)

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
    ap.add_argument("--input", default="ptc_fresh_2026-05-01.json",
                    help="Path to the PTC fresh benchmark JSON (defaults to sibling dir)")
    ap.add_argument("--bench", default="benchmark_1000.json",
                    help="Path to original benchmark_1000.json (will be looked up "
                         "relative to PT_CHEMISTRY repo if --input is the PTC fresh one)")
    ap.add_argument("--out", default="results_b3lyp.json",
                    help="Output JSON path (will be overwritten on each checkpoint)")
    ap.add_argument("--overrides", default="geometry_overrides.json",
                    help="Optional JSON file with geometry/spin/charge overrides")
    ap.add_argument("--exclude-elements", default=DEFAULT_EXCLUDED_ELEMENTS,
                    help="Comma-separated element symbols to remove from the run "
                         f"(default: {DEFAULT_EXCLUDED_ELEMENTS})")
    ap.add_argument("--max-heavy", type=int, default=None,
                    help="Skip molecules with more heavy atoms than this (speed limit)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip molecules already present in --out (idempotent restart)")
    ap.add_argument("--checkpoint", type=int, default=1,
                    help="Save results every N completed molecules (default 1)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N molecules (debug)")
    args = ap.parse_args()

    # Resolve input list. We accept either the PTC fresh JSON (which has results
    # nested under "results") or the raw benchmark_1000.json (a plain list).
    inp = Path(args.input)
    if not inp.exists():
        # fallback to original benchmark file
        inp = Path(args.bench)
    if not inp.exists():
        sys.exit(f"Cannot find input list: tried {args.input} and {args.bench}")

    with open(inp) as f:
        data = json.load(f)
    rows = data["results"] if isinstance(data, dict) and "results" in data else data
    bench = [normalize_entry(r) for r in rows]
    overrides = load_overrides(Path(args.overrides))
    if overrides:
        bench = [merge_override(entry, overrides) for entry in bench]
        print(f"Loaded {len(overrides)} geometry/spin overrides from {args.overrides}")
    print(f"Loaded {len(bench)} molecules from {inp}")

    excluded_elements = parse_element_list(args.exclude_elements)
    bench, skipped_basis = filter_excluded_elements(bench, excluded_elements)
    if skipped_basis:
        print(
            f"Excluded {len(skipped_basis)} molecules containing unsupported elements "
            f"{','.join(sorted(excluded_elements))}; kept {len(bench)}"
        )

    # Resume?
    previous_by_key: dict[str, dict] = {}
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        for r in load_previous_results(out_path):
            if r.get("status"):
                previous_by_key[entry_key(r)] = r
        print(f"Loaded {len(previous_by_key)} previous result rows")

    # Optional pre-filter on heavy atom count via RDKit
    if args.max_heavy is not None:
        from rdkit import Chem
        filt = []
        for entry in bench:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
            if n_heavy <= args.max_heavy:
                filt.append(entry)
        print(f"--max-heavy {args.max_heavy}: kept {len(filt)}/{len(bench)}")
        bench = filt

    if args.limit:
        bench = bench[:args.limit]

    results = []
    done: dict[str, dict] = {}
    if previous_by_key:
        for entry in bench:
            key = entry_key(entry)
            previous = previous_by_key.get(key)
            if previous and resume_compatible(previous, entry):
                done[key] = previous
                results.append(previous)
        print(f"Resuming: {len(done)} compatible rows already done")

    checkpoint_every = max(1, args.checkpoint)
    t_start = time.time()
    n_new = 0

    def payload():
        return {
            "method": "B3LYP/6-31G*",
            "run_config_version": RUN_CONFIG_VERSION,
            "zpe_scale": ZPE_SCALE_B3LYP_631GS,
            "elapsed_s": time.time() - t_start,
            "n_done": len(results),
            "n_skipped_unsupported_basis": len(skipped_basis),
            "excluded_elements": sorted(excluded_elements),
            "results": results,
        }

    if args.resume and previous_by_key:
        save_results(out_path, payload())

    try:
        for i, entry in enumerate(bench, 1):
            if entry_key(entry) in done:
                continue
            print(f"[{i:4d}/{len(bench)}] {entry['name'][:30]:30}  ", end="", flush=True)
            r = compute_D_at_b3lyp(entry)
            results.append(r)
            n_new += 1
            if r["status"] == "ok":
                print(f"D={r['D_b3lyp_eV']:7.3f}  err={r['err_eV']:+.3f}  "
                      f"({r['time_s']:5.1f}s)")
            elif r["status"] == "no_zpe":
                print(f"D_elec={r['D_elec_eV']:7.3f}  (no ZPE, {r['time_s']:.1f}s)")
            else:
                print(f"FAIL: {r.get('error_msg','?')[:50]}")
            if n_new % checkpoint_every == 0:
                save_results(out_path, payload())
    except KeyboardInterrupt:
        save_results(out_path, payload())
        print(f"\nInterrupted; saved {len(results)} completed rows to {out_path}")
        raise SystemExit(130)

    # Final save
    save_results(out_path, payload())

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    no_zpe = [r for r in results if r["status"] == "no_zpe"]
    failed = [r for r in results if r["status"] == "fail"]
    print(f"\n=== Done in {time.time()-t_start:.1f}s ===")
    print(f"  ok        : {len(ok)}")
    print(f"  no_zpe    : {len(no_zpe)}")
    print(f"  failed    : {len(failed)}")
    if ok:
        rel = [r["rel_err_pct"] for r in ok]
        print(f"  MAE on ok : {sum(rel)/len(rel):.2f}% (median {sorted(rel)[len(rel)//2]:.2f}%)")


if __name__ == "__main__":
    main()
