"""
generate_datasheets.py — produce experimental datasheets for the
top PT-predicted aromatic candidates and write them to
docs/research/<name>_DATASHEET_<date>.md.
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import date
from ptc.signature import predict_full_signature, format_datasheet


CANDIDATES = [
    # (display name, SMILES, cap_idx, descriptor, file_slug)
    ("U@S₃",   "[U][S]1[S][S]1",   0,    "Uranium-capped sulfur triangle — chalcogenide analog of Bi₃@U", "U_S3"),
    ("Th@S₃",  "[Th][S]1[S][S]1",  0,    "Thorium-capped sulfur triangle (recommended first synthesis target)", "Th_S3"),
    ("Ce@S₃",  "[Ce][S]1[S][S]1",  0,    "Cerium-capped sulfur triangle (lanthanide model, less radioactive)", "Ce_S3"),
    ("GaSGa",  "[Ga]1[S][Ga]1",    None, "Gallium-bridged sulfide triangle — heteronuclear", "GaSGa"),
    ("AlSeAl", "[Al]1[Se][Al]1",   None, "Aluminum-bridged selenide triangle — heteronuclear", "AlSeAl"),
    ("P₃",     "[P]1[P][P]1",      None, "Cyclic triphosphorus — homonuclear σ-aromatic gas-phase candidate", "P3"),
]

OUT_DIR = Path("docs/research")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATE = date.today().isoformat().replace("-", "_")


def _experimental_section(name: str, sig, descriptor: str) -> str:
    """Tailored experimental protocol for each candidate."""
    L = []
    L.append("\n## Experimental protocol — tailored to this candidate")
    L.append("")
    cap = (sig.cap_binding > 0)
    L.append(f"_{descriptor}_")
    L.append("")

    # Protocol selection by candidate features
    if cap and any(z >= 89 for z in sig.Z_list):  # actinide-capped
        L.append("### Synthesis (actinide cap)")
        L.append("")
        L.append("- **Solvent:** liquid NH₃ at −78 °C, or DME/THF at −80 °C")
        L.append("- **Reductant:** alkali metal (Na, K) in 2:1 stoichiometry vs cap precursor")
        L.append("- **Cap precursor:** tris-Cp* actinide chloride [Cp*ₓMCl_y] with M = the cap atom")
        L.append("- **Ring source:** Na₂S, K₂Sₓ, or in-situ-generated S²⁻")
        L.append("- **Counter-ion:** [K(crypt-2.2.2)]⁺ for crystal isolation (Goicoechea protocol)")
        L.append(f"- **Predicted cap binding:** {sig.cap_binding:.2f} eV — synthesis margin OK at < 200 °C")
    elif cap and any(57 <= z <= 71 for z in sig.Z_list):  # lanthanide-capped
        L.append("### Synthesis (lanthanide cap)")
        L.append("")
        L.append("- **Easier than actinide** — no radioactive handling required")
        L.append("- Use the same NH₃(l) protocol as Goicoechea, replacing actinide source")
        L.append("- **Recommended cap source:** [Cp*ₓLnCl] tetrahydrofuran adduct")
        L.append(f"- **Predicted cap binding:** {sig.cap_binding:.2f} eV")
    elif sig.f_coh < 1.0:  # heteronuclear
        L.append("### Synthesis (heteronuclear ring)")
        L.append("")
        L.append("- **Direct route:** pulsed laser ablation of a binary precursor (e.g. Ga₂S₃ for GaSGa)")
        L.append("- **Source:** 532 nm laser, 10-20 mJ/pulse, supersonic He expansion at 5 bar")
        L.append("- **Detection:** TOF-MS (positive + negative modes)")
        L.append(f"- **Predicted (vertical) IE:** {sig.IE_PT_eV:.2f} eV — within accessible photoionization range")
    else:  # homonuclear neutral
        L.append("### Synthesis (homonuclear neutral cluster)")
        L.append("")
        L.append("- **Source:** Knudsen-cell mass spec from elemental precursor")
        L.append("- **Temperature:** 800-1200 K (cluster sublimation)")
        L.append("- **Detection:** TOF-MS, photoionization threshold")
        L.append(f"- **Lowest fragmentation barrier:** {sig.lowest_decomposition_eV:.2f} eV")

    L.append("")
    L.append("### Spectroscopic targets (PT predictions)")
    L.append("")
    L.append("| technique | predicted observable | notes |")
    L.append("|-----------|---------------------|-------|")
    if sig.ir_breathing_cm1 > 0:
        L.append(f"| IR (matrix Ar 4 K) | breathing mode at **{sig.ir_breathing_cm1:.0f} cm⁻¹** | totally symmetric, A₁ in C_nv |")
    nics0 = sig.NICS_profile[0][1] if sig.NICS_profile else 0
    nics1 = sig.NICS_profile[2][1] if len(sig.NICS_profile) > 2 else 0
    L.append(f"| ¹H NMR / ³³S NMR | ring current shift, NICS(0) = **{nics0:+.1f} ppm** | post-DFT GIAO comparison |")
    L.append(f"| Anion PES | EA = **{sig.EA_PT_eV:.2f} ± 0.3 eV** (Wang/Boldyrev) | Franck-Condon to ω vibration |")
    L.append(f"| Photoionization | IE = **{sig.IE_PT_eV:.2f} ± 0.2 eV** | VUV synchrotron source |")
    if cap:
        L.append("| X-ray monocrystal | r(M-X), cap height, plane symmetry | low T (< 100 K) |")
        L.append("| Mössbauer (¹⁵¹Eu, ²³³U etc.) | isomer shift signs cap oxidation state | distinguishes M(III) from M(IV) |")
    L.append("")

    # Falsifiability
    L.append("### Falsifiability criteria (PT predictions to test)")
    L.append("")
    if nics0 < 0:
        L.append(f"1. NICS(0) is **diamagnetic** (negative). If post-DFT GIAO gives positive NICS, ring is antiaromatic — PT prediction wrong.")
    else:
        L.append(f"1. NICS(0) is **paramagnetic** (positive). If post-DFT GIAO gives negative NICS, the σ-aromatic cancellation is incomplete.")
    if sig.NICS_profile and len(sig.NICS_profile) > 2:
        ratio = sig.NICS_profile[0][1] / sig.NICS_profile[2][1] if sig.NICS_profile[2][1] != 0 else 0
        L.append(f"2. NICS(0) / NICS(1) ratio = {ratio:.2f}. Ratio < 1.5 indicates π-only (no σ aromaticity); ratio > 1.5 confirms σ-dominant or double-aromatic.")
    if cap:
        L.append(f"3. Cap binding ≥ {0.7*sig.cap_binding:.1f} eV by TGA-MS or thermal decomposition. Below this, the f-block back-donation channel R₅₇ is overestimated.")
    if sig.ir_breathing_cm1 > 0:
        L.append(f"4. Ring breathing IR pic at {sig.ir_breathing_cm1:.0f} ± 50 cm⁻¹. Significant deviation indicates ring r_e or D₀ wrong.")
    L.append("")
    return "\n".join(L)


def main():
    summary_rows = []
    for name, smi, cap_idx, descriptor, slug in CANDIDATES:
        print(f"Generating {name} ({smi})…")
        sig = predict_full_signature(smi, cap_idx=cap_idx)

        # Core datasheet
        body = format_datasheet(sig, name=name)
        # Append experimental section
        body += "\n" + _experimental_section(name, sig, descriptor)

        # Front matter / metadata
        header = (f"<!-- PT predictive datasheet, {DATE.replace('_', '-')} -->\n"
                  f"<!-- engine: ptc/transfer_matrix.py post-C9c, ptc/nics.py, ptc/signature.py -->\n\n")
        path = OUT_DIR / f"{slug}_DATASHEET_{DATE}.md"
        path.write_text(header + body)
        print(f"  → {path}")

        # Summary row
        nics0 = sig.NICS_profile[0][1] if sig.NICS_profile else 0
        ratio = (sig.NICS_profile[0][1] / sig.NICS_profile[2][1]
                 if sig.NICS_profile and len(sig.NICS_profile) > 2
                 and sig.NICS_profile[2][1] != 0 else 0)
        summary_rows.append({
            'name': name,
            'D_per_atom': sig.D_per_atom,
            'cap_E': sig.cap_binding,
            'lowest_frag': sig.lowest_decomposition_eV,
            'class': sig.aromatic_class,
            'n_sigma': sig.n_aromatic_sigma,
            'n_pi': sig.n_aromatic_pi,
            'NICS0': nics0,
            'NICS1': sig.NICS_profile[2][1] if len(sig.NICS_profile) > 2 else 0,
            'ratio': ratio,
            'omega_breath': sig.ir_breathing_cm1,
            'IE': sig.IE_PT_eV,
            'EA': sig.EA_PT_eV,
            'f_coh': sig.f_coh,
        })

    # Comparison table
    print("\n" + "="*90)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*90)
    print(f"{'name':<10}{'D/at':>6}{'cap':>5}{'frag':>5}{'σ':>4}{'π':>4}"
          f"{'NICS0':>8}{'NICS1':>8}{'ratio':>7}{'ω':>6}{'IE':>6}{'EA':>5}{'f_coh':>7}")
    for r in summary_rows:
        print(f"{r['name']:<10}{r['D_per_atom']:>6.2f}{r['cap_E']:>5.1f}"
              f"{r['lowest_frag']:>5.1f}{r['n_sigma']:>4d}{r['n_pi']:>4d}"
              f"{r['NICS0']:>+8.1f}{r['NICS1']:>+8.1f}{r['ratio']:>7.2f}"
              f"{r['omega_breath']:>6.0f}{r['IE']:>6.2f}{r['EA']:>5.2f}{r['f_coh']:>7.3f}")
    print("="*90)
    print("Class summary:")
    for r in summary_rows:
        print(f"  {r['name']}: {r['class']}")


if __name__ == "__main__":
    main()
