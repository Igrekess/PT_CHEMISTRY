"""
reactions.py -- Module 3: Reactions and Thermochemistry.

Parse balanced equations, compute reaction enthalpy from D_at balance,
estimate activation energy (Evans-Polanyi PT) and rate constants
(Arrhenius-TST).  Equilibrium from Gibbs free energy.

   ΔH  = Σ D_at(reactants) − Σ D_at(products)    [from cascade]
   Ea  = Evans-Polanyi with α = s = 1/2           [T0: universal mixing]
   E0  = <D_bond(broken)> × sin²_3                [P1 gate barrier height]
   k(T)= (kT/h) × exp(−Ea/kT)                    [TST pre-exponential]
   Keq = exp(−ΔG/kT)                              [ΔG = ΔH − TΔS]

0 adjustable parameters.  All from s = 1/2.

April 2026 -- Theorie de la Persistance
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ptc.constants import (
    S_HALF, S3, RY, P1,
    HBAR_EV_S, EV_J,
)

# ── Physical constants ──
_EV_PER_KJ_MOL = 96.485        # 1 eV = 96.485 kJ/mol
_KB_EV = 8.617333e-5            # Boltzmann (eV/K)
_KB_J = 1.380649e-23            # Boltzmann (J/K)
_H_PLANCK_EV_S = 2 * math.pi * HBAR_EV_S   # h = 2π ℏ  (eV·s)
_R_GAS_KJ = 8.314462e-3        # gas constant (kJ/(mol·K))


# ════════════════════════════════════════════════════════════════════
#  REACTION SPECIFICATION
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ReactionSpecies:
    """One species in a reaction: coefficient + SMILES."""
    smiles: str
    coeff: int


@dataclass(frozen=True)
class ReactionSpec:
    """Parsed reaction equation."""
    reactants: List[ReactionSpecies]
    products: List[ReactionSpecies]
    equation: str          # original string

    @property
    def n_react(self) -> int:
        return sum(s.coeff for s in self.reactants)

    @property
    def n_prod(self) -> int:
        return sum(s.coeff for s in self.products)

    @property
    def delta_n(self) -> int:
        """Change in number of molecules (products - reactants)."""
        return self.n_prod - self.n_react


def parse_reaction(equation: str) -> ReactionSpec:
    """Parse a reaction equation.

    Format: "A + B >> C + D"   or   "2 A + B >> 3 C"
    Stoichiometric coefficients: integer before the SMILES, separated by space.
    The '>>' separator is required.

    Examples:
        "2 [H][H] + O=O >> 2 [H]O[H]"
        "[H][H] + Cl-Cl >> 2 [H]Cl"
        "C + 2 O=O >> O=C=O + 2 [H]O[H]"
    """
    if ">>" not in equation:
        raise ValueError(f"Equation must contain '>>' separator: {equation!r}")

    lhs, rhs = equation.split(">>", 1)

    def _parse_side(text: str) -> List[ReactionSpecies]:
        species = []
        # Split on ' + ' (with spaces) to avoid splitting SMILES like [Cu+2]
        parts = re.split(r'\s*\+\s+', text.strip())
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Check for leading coefficient: "2 [H][H]" or "3 O=O"
            m = re.match(r'^(\d+)\s+(.+)$', part)
            if m:
                coeff = int(m.group(1))
                smiles = m.group(2).strip()
            else:
                coeff = 1
                smiles = part
            species.append(ReactionSpecies(smiles=smiles, coeff=coeff))
        return species

    return ReactionSpec(
        reactants=_parse_side(lhs),
        products=_parse_side(rhs),
        equation=equation.strip(),
    )


# ════════════════════════════════════════════════════════════════════
#  REACTION RESULT
# ════════════════════════════════════════════════════════════════════

@dataclass
class ReactionResult:
    """Full reaction thermochemistry and kinetics."""
    # Identity
    equation: str
    spec: ReactionSpec

    # Per-species D_at (eV)
    D_at_reactants: Dict[str, float]    # SMILES -> D_at
    D_at_products: Dict[str, float]     # SMILES -> D_at

    # Thermochemistry
    delta_H_eV: float       # reaction enthalpy (eV)
    delta_H_kJ: float       # reaction enthalpy (kJ/mol)
    delta_S_approx: float   # approximate entropy change (eV/K)
    delta_G_eV: float       # Gibbs free energy at T (eV)
    delta_G_kJ: float       # Gibbs free energy at T (kJ/mol)

    # Activation energy (Evans-Polanyi PT)
    Ea_eV: float             # activation energy (eV)
    Ea_kJ: float             # activation energy (kJ/mol)
    E0_eV: float             # intrinsic barrier (eV)

    # Kinetics
    T: float                 # temperature (K)
    P: float                 # pressure (atm)
    delta_n: int             # change in moles of gas (products - reactants)
    k_rate: float            # rate constant k(T) (s^-1)
    A_prefactor: float       # pre-exponential factor (s^-1)
    K_eq: float              # equilibrium constant K_eq(T)


# ════════════════════════════════════════════════════════════════════
#  AVERAGE BROKEN-BOND ENERGY (for Evans-Polanyi E0)
# ════════════════════════════════════════════════════════════════════

def _avg_bond_energy(smiles_list: List[ReactionSpecies]) -> float:
    """Average per-bond energy across reactant molecules.

    E0_barrier = <D_bond> * sin^2_3 (P1 gate projection).
    """
    from ptc.api import Molecule

    total_D = 0.0
    total_bonds = 0
    for sp in smiles_list:
        mol = Molecule(sp.smiles)
        nb = len(mol.topology.bonds)
        if nb > 0:
            total_D += sp.coeff * mol.D_at
            total_bonds += sp.coeff * nb

    if total_bonds == 0:
        return 0.0
    return total_D / total_bonds


# ════════════════════════════════════════════════════════════════════
#  ENTROPY ESTIMATION
# ════════════════════════════════════════════════════════════════════

def _delta_S_approx(spec: ReactionSpec, T: float) -> float:
    """Approximate entropy change from molecule count.

    ΔS ≈ Δn × R × ln(kT/P°V°)

    For gas-phase reactions, Δn changes in translational entropy
    dominate.  The Sackur-Tetrode leading term gives ~ kB per
    gained/lost molecule at 298 K, 1 atm.

    PT derivation: each new translational degree of freedom
    contributes s × kB × ln(μ*) from the P1 gate,
    but for simplicity we use the standard physical-chemistry
    estimate: ΔS ≈ Δn × 14.7 cal/(mol·K) ≈ Δn × 6.36e-4 eV/K.
    """
    # Δn = n_prod - n_react (molecular count change)
    delta_n = spec.delta_n

    # Empirical: ~14.7 cal/(mol·K) per molecule of gas gained/lost
    # = 61.5 J/(mol·K) = 6.38e-4 eV/K
    dS_per_mol = 6.38e-4  # eV/K per molecule

    return delta_n * dS_per_mol


# ════════════════════════════════════════════════════════════════════
#  ACTIVATION ENERGY FROM D_KL (PT-derived, April 2026)
# ════════════════════════════════════════════════════════════════════
#
# Hammond's postulate is a CONSEQUENCE of the sieve.
#
# Principe 2 (cap Shannon) : D_KL ≤ 1 bit par canal.
# Activation barrier = D_KL between reactant and TS.
# Elle sature au cap du canal P₁ : Ry × sin²₃ = 2.98 eV.
#
# Formule D_KL :
#   Exothermique (ΔH ≤ 0) :
#     Ea = E_cap × (1 - exp(-|ΔH|/E_cap))
#     E_cap = Ry × sin²₃ (cap du canal P₁)
#
#   Endothermique (ΔH > 0) :
#     Ea = ΔH + E_cap × (1 - exp(-ΔH/E_cap))
#     Thermodynamic floor + reverse barrier.
#
# Properties:
#   - |ΔH| → 0: Ea → |ΔH| (linear, all energy = barrier)
#   - |ΔH| → ∞: Ea → E_cap (saturation = Hammond, early TS)
#   - ΔH = 0: Ea = 0 (thermoneutral, no D_KL barrier)
#   - Perfect continuity between exo and endo
#
# Note: for multi-step reactions, Ea applies to EACH elementary
# step. The overall Ea is the maximum barrier
# along the reaction path.

# Cap du canal P₁ : Ry × sin²₃
_E_CAP_P1 = RY * S3   # 2.98 eV = 287 kJ/mol


def _ea_dkl(delta_H: float, E0: float) -> float:
    """Activation energy from hybrid Evans-Polanyi + D_KL barrier (eV).

    Two PT-derived lower bounds, take the maximum:

    1. Evans-Polanyi (T0: s = 1/2 mixing):
       Ea_EP = max(0, s×ΔH + E0)          [exothermic]
       Ea_EP = ΔH + max(0, E0 - s×ΔH)    [endothermic]

    2. D_KL exponential decay (Principe 2, cap Shannon):
       Ea_DKL = E0 × exp(-|ΔH| / E_cap)
       E_cap = Ry × sin²₃ = 2.98 eV (P₁ channel cap)

    The D_KL term ensures a NON-ZERO barrier for exothermic reactions
    (Hammond: TS is early but NOT at the reactant configuration).
    The exponential decay rate 1/E_cap is the inverse cap Shannon.
    """
    if E0 < 1e-10:
        return max(0.0, delta_H)

    abs_dH = abs(delta_H)

    if delta_H <= 0:
        # Exothermic: two lower bounds
        Ea_EP = max(0.0, S_HALF * delta_H + E0)
        Ea_DKL = E0 * math.exp(-abs_dH / _E_CAP_P1)
        Ea = max(Ea_EP, Ea_DKL)
    else:
        # Endothermic: thermodynamic floor + reverse barrier
        reverse_EP = max(0.0, E0 - S_HALF * delta_H)
        reverse_DKL = E0 * math.exp(-delta_H / _E_CAP_P1)
        Ea = delta_H + max(reverse_EP, reverse_DKL)

    return max(Ea, 0.0)


# Legacy wrapper
def _evans_polanyi(delta_H: float, E0: float) -> float:
    """Redirects to D_KL barrier (backward compatibility)."""
    return _ea_dkl(delta_H, E0)


# ════════════════════════════════════════════════════════════════════
#  MAIN COMPUTATION
# ════════════════════════════════════════════════════════════════════

def compute_reaction(equation: str, T: float = 298.15, P: float = 1.0) -> ReactionResult:
    """Compute full reaction thermochemistry and kinetics.

    Parameters
    ----------
    equation : str
        Reaction in "A + B >> C + D" format with optional coefficients.
    T : float
        Temperature in Kelvin (default 298.15 K).
    P : float
        Pressure in atm (default 1.0 atm).
        For gas-phase reactions with Δn ≠ 0, the Gibbs free energy is
        corrected: ΔG = ΔG° + kT × Δn × ln(P).

    Returns
    -------
    ReactionResult
        Complete thermochemistry: ΔH, ΔG, Ea, k(T), K_eq.

    Examples
    --------
    >>> r = compute_reaction("[H][H] + Cl-Cl >> 2 [H]Cl")
    >>> print(f"ΔH = {r.delta_H_kJ:.1f} kJ/mol")
    """
    from ptc.api import Molecule

    spec = parse_reaction(equation)

    # ── Compute D_at for each species ──
    D_at_react = {}
    D_at_prod = {}
    mol_cache = {}

    for sp in spec.reactants:
        if sp.smiles not in mol_cache:
            mol_cache[sp.smiles] = Molecule(sp.smiles)
        D_at_react[sp.smiles] = mol_cache[sp.smiles].D_at

    for sp in spec.products:
        if sp.smiles not in mol_cache:
            mol_cache[sp.smiles] = Molecule(sp.smiles)
        D_at_prod[sp.smiles] = mol_cache[sp.smiles].D_at

    # ── Reaction enthalpy ──
    # ΔH = Σ (coeff × D_at)_reactants − Σ (coeff × D_at)_products
    # Negative ΔH = exothermic (products more bound)
    sum_D_react = sum(sp.coeff * D_at_react[sp.smiles] for sp in spec.reactants)
    sum_D_prod = sum(sp.coeff * D_at_prod[sp.smiles] for sp in spec.products)
    delta_H = sum_D_react - sum_D_prod  # eV

    # ── Entropy approximation ──
    delta_S = _delta_S_approx(spec, T)

    # ── Gibbs free energy ──
    # Standard Gibbs: ΔG° = ΔH - TΔS
    delta_G_std = delta_H - T * delta_S
    # Pressure correction for gas-phase: ΔG = ΔG° + kT × Δn × ln(P)
    delta_n = spec.delta_n
    kT = _KB_EV * max(T, 1.0)
    delta_G = delta_G_std + kT * delta_n * math.log(max(P, 1e-10))

    # ── Activation energy (Evans-Polanyi PT) ──
    # Intrinsic barrier: <D_bond> × sin²₃
    avg_D_bond = _avg_bond_energy(spec.reactants)
    E0 = avg_D_bond * S3   # barrier from P1 gate
    Ea = _evans_polanyi(delta_H, E0)

    # ── Rate constant (TST) ──
    # k(T) = (kT/h) × exp(-Ea/kT)
    A_tst = kT / _H_PLANCK_EV_S   # s^-1 (transition state theory pre-exponential)

    if Ea / kT > 500:
        k_rate = 0.0
    else:
        k_rate = A_tst * math.exp(-Ea / kT)

    # ── Equilibrium constant ──
    # K_eq = exp(-ΔG / kT)
    dG_over_kT = delta_G / kT
    if abs(dG_over_kT) > 500:
        K_eq = math.inf if dG_over_kT < 0 else 0.0
    else:
        K_eq = math.exp(-dG_over_kT)

    return ReactionResult(
        equation=equation.strip(),
        spec=spec,
        D_at_reactants=D_at_react,
        D_at_products=D_at_prod,
        delta_H_eV=delta_H,
        delta_H_kJ=delta_H * _EV_PER_KJ_MOL,
        delta_S_approx=delta_S,
        delta_G_eV=delta_G,
        delta_G_kJ=delta_G * _EV_PER_KJ_MOL,
        Ea_eV=Ea,
        Ea_kJ=Ea * _EV_PER_KJ_MOL,
        E0_eV=E0,
        T=T,
        P=P,
        delta_n=delta_n,
        k_rate=k_rate,
        A_prefactor=A_tst,
        K_eq=K_eq,
    )


# ════════════════════════════════════════════════════════════════════
#  CONVENIENCE: multi-temperature sweep
# ════════════════════════════════════════════════════════════════════

def reaction_vs_T(equation: str,
                  T_range: Tuple[float, ...] = (200, 298.15, 400, 600, 800, 1000),
                  P: float = 1.0,
                  ) -> List[ReactionResult]:
    """Compute reaction at multiple temperatures."""
    return [compute_reaction(equation, T=T, P=P) for T in T_range]


# ════════════════════════════════════════════════════════════════════
#  PRETTY PRINTER
# ════════════════════════════════════════════════════════════════════

def _fmt_species(species_list: List[ReactionSpecies]) -> str:
    parts = []
    for sp in species_list:
        if sp.coeff == 1:
            parts.append(sp.smiles)
        else:
            parts.append(f"{sp.coeff} {sp.smiles}")
    return " + ".join(parts)


def print_reaction(r: ReactionResult, exp_dH_kJ: Optional[float] = None):
    """Pretty-print a ReactionResult with optional experimental comparison."""
    print("=" * 70)
    print(f"  {_fmt_species(r.spec.reactants)}  -->  {_fmt_species(r.spec.products)}")
    print("=" * 70)

    print(f"\n  D_at (reactants):")
    for sp in r.spec.reactants:
        d = r.D_at_reactants[sp.smiles]
        label = f"  {sp.coeff} x " if sp.coeff > 1 else "      "
        print(f"    {label}{sp.smiles:15s} D_at = {d:8.4f} eV   (x{sp.coeff} = {sp.coeff*d:8.4f})")

    print(f"\n  D_at (products):")
    for sp in r.spec.products:
        d = r.D_at_products[sp.smiles]
        label = f"  {sp.coeff} x " if sp.coeff > 1 else "      "
        print(f"    {label}{sp.smiles:15s} D_at = {d:8.4f} eV   (x{sp.coeff} = {sp.coeff*d:8.4f})")

    print(f"\n  Thermochemistry (T = {r.T:.1f} K, P = {r.P:.2f} atm, Δn = {r.delta_n:+d}):")
    print(f"    delta_H  = {r.delta_H_eV:+.4f} eV  = {r.delta_H_kJ:+.1f} kJ/mol")
    if exp_dH_kJ is not None:
        err = r.delta_H_kJ - exp_dH_kJ
        pct = 100.0 * err / abs(exp_dH_kJ) if exp_dH_kJ != 0 else float('inf')
        print(f"    exp      =                  {exp_dH_kJ:+.1f} kJ/mol")
        print(f"    error    =                  {err:+.1f} kJ/mol  ({pct:+.1f}%)")

    print(f"    delta_S  ~ {r.delta_S_approx:+.2e} eV/K")
    print(f"    delta_G  = {r.delta_G_eV:+.4f} eV  = {r.delta_G_kJ:+.1f} kJ/mol")

    print(f"\n  Activation energy (Evans-Polanyi PT, alpha = s = 1/2):")
    print(f"    E0       = {r.E0_eV:.4f} eV  = {r.E0_eV * _EV_PER_KJ_MOL:.1f} kJ/mol  (<D_bond> x sin^2_3)")
    print(f"    Ea       = {r.Ea_eV:.4f} eV  = {r.Ea_kJ:.1f} kJ/mol")

    print(f"\n  Kinetics (TST, T = {r.T:.1f} K):")
    print(f"    A        = {r.A_prefactor:.3e} s^-1")
    print(f"    k(T)     = {r.k_rate:.3e} s^-1")
    print(f"    K_eq     = {r.K_eq:.3e}")

    exo = "EXOTHERMIC" if r.delta_H_eV < 0 else "ENDOTHERMIC"
    print(f"\n  Classification: {exo}")
    print()


# ════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ════════════════════════════════════════════════════════════════════

def _selftest():
    """Run the 3 benchmark reactions and print results."""
    benchmarks = [
        ("[H][H] + Cl-Cl >> 2 [H]Cl",        -184.6, "H2 + Cl2 -> 2 HCl"),
        ("C + 2 O=O >> O=C=O + 2 [H]O[H]",   -890.4, "CH4 + 2 O2 -> CO2 + 2 H2O"),
        ("N#N + 3 [H][H] >> 2 N",             -91.8,  "N2 + 3 H2 -> 2 NH3"),
    ]

    print()
    print("=" * 70)
    print("  PTC Module 3 — Reactions and Thermochemistry")
    print("  0 adjustable parameters. All from s = 1/2.")
    print("=" * 70)
    print()

    results = []
    for eq, exp_kJ, label in benchmarks:
        print(f"--- {label} ---")
        r = compute_reaction(eq, T=298.15)
        print_reaction(r, exp_dH_kJ=exp_kJ)
        results.append((label, r, exp_kJ))

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Reaction':<35s} {'PTC kJ/mol':>12s} {'Exp kJ/mol':>12s} {'Error':>10s} {'Err%':>8s}")
    print("-" * 70)
    for label, r, exp_kJ in results:
        err = r.delta_H_kJ - exp_kJ
        pct = 100.0 * err / abs(exp_kJ) if exp_kJ != 0 else 0.0
        print(f"  {label:<35s} {r.delta_H_kJ:>+12.1f} {exp_kJ:>+12.1f} {err:>+10.1f} {pct:>+7.1f}%")
    print("-" * 70)
    avg_abs_pct = sum(abs(100.0 * (r.delta_H_kJ - exp) / abs(exp))
                      for _, r, exp in results) / len(results)
    print(f"  Average |error|: {avg_abs_pct:.1f}%")
    print()

    return results


if __name__ == "__main__":
    _selftest()
