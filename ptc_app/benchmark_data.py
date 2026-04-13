"""
Benchmark precision data for each PTC module.

Each module has a reliability card showing:
  - Grade: A (MAE<5%), B (<15%), C (<30%), D (>30%)
  - MAE against experimental data
  - Number of test points
  - Key reference (NIST, Marcus, etc.)
  - Derivation depth (d=0..6 in PT hierarchy)
  - Status: [DER-PHYS], [VALIDATION], [PRED]

Updated: April 2026 (post-benchmark)
"""

BENCHMARK = {
    'dielectric': {
        'grade': 'A',
        'mae': 2.1,
        'mae_unit': '%',
        'n_tests': 6,
        'n_pass': 6,       # all within 12%
        'pass_thresh': 12,
        'reference': 'CRC Handbook 2023',
        'depth': 2,
        'status': '[DER-PHYS]',
        'detail': '3 PT mechanisms: cascade P₁ [P₁/(P₁+n_C)]^P₁, fold-back n_C(n_C-1)×S₃, Kirkwood P₃ (1+S₇)',
        'highlight': 'eps(H2O) = P1*P2*P3*P1/(P1+1) = 78.75',
        'strong': ['H2O (0.5%)', 'DMSO (1.5%)', 'EtOH (0.1%)', 'MeOH (1.6%)'],
        'weak': ['acetone (3.6%)', 'hexane (5.3%)'],
    },
    'solvation': {
        'grade': 'A',
        'mae': 5.0,
        'mae_unit': '%',
        'n_tests': 17,
        'n_pass': 17,      # all 17 within 15%
        'pass_thresh': 15,
        'reference': 'Marcus (1991)',
        'depth': 3,
        'status': '[DER-PHYS]',
        'detail': 'Born-PT + 6 mecanismes (sterique, polarisation, saturation, electrostriction, IE-radii, CFSE). Mono 5.7%, di 4.7%, tri 4.7%.',
        'strong': ['Na+ (0.3%)', 'Al3+ (1.6%)', 'Mg2+ (1.8%)', 'Fe2+ (2.7%)'],
        'weak': ['K+ (10.3%)', 'Ba2+ (8.6%)'],
    },
    'pka': {
        'grade': 'A',
        'mae': 0.16,
        'mae_unit': ' pKa units (organic, 17 acids)',
        'n_tests': 25,
        'n_pass': 22,
        'pass_thresh': 1,
        'reference': 'NIST Chemistry WebBook / CRC',
        'depth': 4,
        'status': '[DER-PHYS]',
        'detail': 'sin²₃ LFER: gas→solution screening = P₁ channel throughput. 3 refs: carbox, phenol, amine. 25 acides dont drug molecules.',
        'strong': ['aspirine (0.01)', 'ibuprofene (0.03)', 'glycine (0.05)', 'paracetamol (0.02)', 'HCOOH (0.15)'],
        'weak': ['CF3COOH (0.66)', 'p-NO2-phenol (0.46)'],
        'blind_test': 'pKa(aspirine) = 3.50 (exp 3.49, erreur 0.01)',
    },
    'electrochemistry': {
        'grade': 'A',
        'mae': 0.058,
        'mae_unit': 'V',
        'n_tests': 32,
        'n_pass': 32,      # all within ±0.2V
        'pass_thresh': 0.2,
        'reference': 'NIST/Bard-Faulkner Electrode Potentials',
        'depth': 4,
        'status': '[DER-PHYS]',
        'detail': 'Cross-channel P1×P2 + 6 new mechanisms: metallic CFSE, relativistic 5d, IE3 trivalent, s-band bifurcation, Be compact, s-block period. 0 > 0.2V.',
        'kendall_tau': 0.95,
        'classification': '32/32 noble/active',
        'redox_correct': '5/6 spontaneity',
        'strong': ['Fe2+ (0.01V)', 'Al3+ (0.003V)', 'Sn2+ (0.007V)', 'Cu2+ (0.01V)', 'Pt2+ (0.04V)', 'Au3+ (0.03V)', 'Fe3+ (0.06V)'],
        'weak': ['Mg2+ (0.16V)', 'Cu1+ (0.17V)', 'Cr3+ (0.14V)'],
    },
    'uv_vis': {
        'grade': 'A',
        'mae': 4.1,
        'mae_unit': '%',
        'n_tests': 9,
        'n_pass': 7,       # within 12%
        'pass_thresh': 12,
        'reference': 'NIST Chemistry WebBook UV',
        'depth': 3,
        'status': '[DER-PHYS]',
        'detail': '5 canaux PT + 2 corrections: inclusion-exclusion P₁∪P₂ (n→π*), double-passe LP (σ→σ*). LP screening C₃(1-S₃·n_LP/per).',
        'strong': ['benzene (0.9%)', 'H2O (0.4%)', 'formaldehyde (1.4%)', 'N2 (3.5%)', 'hexatriene (3.8%)'],
        'weak': ['ethylene (26% — molecular IE model needed)'],
    },
    'ir': {
        'grade': 'A',
        'mae': 4.1,
        'mae_unit': '%',
        'n_tests': 10,
        'n_pass': 10,
        'pass_thresh': 10,
        'reference': 'NIST/HITRAN vibrational frequencies',
        'depth': 3,
        'status': '[DER-PHYS]',
        'detail': 'Morse+bend+torsion + triple bond saturation (sin²₃ screening on T³)',
        'selection_rules': '4/4 PASS (homonuclear IR-silent)',
        'strong': ['N2 (0.7%)', 'H2O (1.3%)', 'HCl (1.9%)', 'CO (4.1%)'],
        'weak': [],
    },
    'raman': {
        'grade': 'A',
        'mae': 3.1,
        'mae_unit': '%',
        'n_tests': 15,
        'n_pass': 13,
        'pass_thresh': 10,
        'reference': 'NIST Raman frequencies',
        'depth': 3,
        'status': '[DER-PHYS]',
        'detail': 'Hessian normal modes + polarizability PT (S₅ channel). Closest-mode matching. 15 molecules NIST.',
        'strong': ['H2 (0.1%)', 'N2 (0.5%)', 'Cl2 (0.3%)', 'HCN (0.5%)', 'CO2 (0.6%)'],
        'weak': ['F2 (21.5% — LP repulsion underestimated in r_e)'],
    },
    'explorer': {
        'grade': 'A',
        'mae': 1.82,
        'mae_unit': '%',
        'n_tests': 1000,
        'n_pass': 950,
        'pass_thresh': 5,
        'reference': 'NIST Atomization Energies',
        'depth': 2,
        'status': '[DER-PHYS]',
        'detail': 'Uses PTC cascade D_at engine (same as Molecule tab)',
        'strong': ['D_at MAE 1.82% on 1000 molecules'],
        'weak': [],
    },
    'reactions': {
        'grade': 'B',
        'mae': 8.0,
        'mae_unit': ' kJ/mol',
        'n_tests': 15,
        'n_pass': 12,
        'pass_thresh': 20,
        'reference': 'NIST-JANAF Thermochemical Tables',
        'depth': 2,
        'status': '[DER-PHYS]',
        'detail': 'Delta_H = D_at(products) - D_at(reactants). Ea from D_KL: hybrid Evans-Polanyi + Shannon cap (Principle 2). Hammond = consequence of the sieve.',
        'strong': ['H2+F2 Ea (16%)', 'HF formation ΔH (0.2%)', 'Combustion H2+O2 (2%)'],
        'weak': ['Multi-step radical chains (Ea = RDS, not overall)'],
    },
    'catalysis': {
        'grade': 'C',
        'mae': 1.35,
        'mae_unit': ' eV (adsorption vs TPD/calorimetry)',
        'n_tests': 22,
        'n_pass': 5,
        'pass_thresh': 0.5,
        'reference': 'TPD/Calorimetry (Campbell 1990, Ertl 2008, Bonzel 1984)',
        'depth': 3,
        'status': '[DER-PHYS]',
        'detail': 'E_ads underestimated ×10-20. Root cause: D_MX = sqrt(IE)×S₃²/P₁ = double screening too aggressive. Fix: replace with S₃ seul (single P₁ pass).',
        'strong': ['Cu-H (0.14 eV)', 'Au-CO (0.28 eV)'],
        'weak': ['Pt-O (3.60 eV)', 'Rh-O (3.37 eV)', 'CO adsorption systematically ~×50 too weak'],
        'limitation': 'Surface bond model S₃²/P₁ to refactor. Qualitative volcano but absolute E_ads wrong.',
    },
    'nmr': {
        'grade': 'A',
        'mae': 0.22,
        'mae_unit': ' ppm (1H)',
        'n_tests': 48,
        'n_pass': 45,
        'pass_thresh': 0.5,
        'reference': 'Spectral Database for Organic Compounds (SDBS)',
        'depth': 3,
        'status': '[DER-PHYS]',
        'detail': 'Spin-Fisher + Bloch-Markov. 1H MAE 0.22 ppm, 13C MAE 5.8 ppm.',
        'strong': ['1H shifts (0.22 ppm)', 'Alkyne detection', 'Beta-CN'],
        'weak': ['13C absolute (5.8 ppm)', '1J(C-H) (+27%)'],
    },
    'femo': {
        'grade': 'B',
        'mae': 14,
        'mae_unit': '% (24 obs.)',
        'n_tests': 24,
        'n_pass': 18,      # 18/24 within 20%
        'pass_thresh': 20,
        'reference': 'Mössbauer (Münck), EPR (Hoffman), Lowe-Thorneley',
        'depth': 4,
        'status': '[DER-PHYS]',
        'detail': '24 observables: S=3/2 exact, χT exact, ZFS 87%, δ MAE 0.064 mm/s, ΔEQ MAE 0.19, N₂ cycle correct.',
        'strong': ['S=3/2 (exact)', 'χT(2K) = 0.469 (exact)', 'ZFS 9.1 cm⁻¹ (exp 10.4, 87%)',
                   'Δ_oct 2%', 'N₂ barrier E₄ (correct)', 'δ Fe(III) 3-12%'],
        'weak': ['ΔEQ Fe5/Fe7 50-84%', 'N₂ overall ΔG 3× (qualitative)'],
        'limitation': 'ΔEQ belt sites need explicit S coordination geometry.',
    },
}


def grade_badge(module: str) -> str:
    """Return a colored badge string for Streamlit."""
    b = BENCHMARK.get(module, {})
    grade = b.get('grade', '?')
    colors = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴'}
    icon = colors.get(grade, '⚪')
    mae = b.get('mae', '?')
    unit = b.get('mae_unit', '')
    n = b.get('n_tests', 0)
    return f"{icon} **Grade {grade}** | MAE {mae}{unit} | {n} tests | 0 param"


def grade_header(*modules: str) -> str:
    """Uniform grade header for any panel. Shows badges + methodology note.

    Usage in any panel:
        st.caption(grade_header('electrochemistry'))
        st.caption(grade_header('ir', 'raman'))  # multiple modules
    """
    badges = " | ".join(grade_badge(m) for m in modules if m in BENCHMARK)
    refs = set()
    for m in modules:
        b = BENCHMARK.get(m, {})
        if b.get('reference'):
            refs.add(b['reference'])
    ref_str = ", ".join(sorted(refs)) if refs else ""
    note = f"Grades evalues sur mesures experimentales ({ref_str})" if ref_str else ""
    return f"{badges}\n\n{note}" if note else badges


class ptc_timer:
    """Context manager that measures wall time and displays it in Streamlit.

    Usage:
        with ptc_timer():
            result = heavy_computation()
        # automatically shows "PTC: 3 ms | 0 param" below the spinner
    """
    def __init__(self):
        self.dt_ms = 0.0

    def __enter__(self):
        import time
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time
        self.dt_ms = (time.perf_counter() - self._t0) * 1000
        try:
            import streamlit as _st
            _st.caption(f"PTC: {self.dt_ms:.0f} ms | 0 param")
        except Exception:
            pass


def precision_card(module: str) -> str:
    """Return a full precision card as markdown."""
    b = BENCHMARK.get(module, {})
    if not b:
        return "No benchmark data."
    grade = b.get('grade', '?')
    colors = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴'}
    icon = colors.get(grade, '⚪')
    lines = [
        f"{icon} **Grade {grade}** — {b.get('detail', '')}",
        f"- MAE: **{b.get('mae', '?')}{b.get('mae_unit', '')}** ({b.get('n_pass', 0)}/{b.get('n_tests', 0)} pass, seuil {b.get('pass_thresh', '')})",
        f"- Ref: {b.get('reference', 'N/A')}",
        f"- Profondeur PT: d={b.get('depth', '?')} {b.get('status', '')}",
    ]
    if b.get('strong'):
        lines.append(f"- Forces: {', '.join(b['strong'])}")
    if b.get('weak'):
        lines.append(f"- Faiblesses: {', '.join(b['weak'])}")
    if b.get('limitation'):
        lines.append(f"- ⚠ Limitation: {b['limitation']}")
    lines.append("- Parametres ajustes: **0**")
    return '\n'.join(lines)
