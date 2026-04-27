"""PT-LCAO + GIAO mini quantum-chemistry package.

PT-pure linear combination of atomic orbitals (LCAO) built from the
existing PT atomic-shell screening machinery (atom.py + periodic.py),
plus inter-atomic T3 coupling (transfer_matrix.py) and London/GIAO
magnetic-perturbation operators.

Roadmap (PROMPT_PT_LCAO_GIAO.md):
  Phase A  atomic_basis    Per-atom STO basis with PT Z_eff
  Phase B  density_matrix  Molecular rho_PT via Hueckel-T3 diagonalisation
  Phase C  giao            London phase factors + Zeeman perturbation
  Phase D  shielding       sigma_alpha,beta(P) tensor at arbitrary 3D point

This Phase A delivery is READ-ONLY w.r.t. the existing PTC engine:
nothing here may impact compute_D_at_transfer or nics output.
"""

from .atomic_basis import (
    PTAtomicOrbital,
    PTAtomBasis,
    build_atom_basis,
    Z_eff_shell,
    occupied_shells,
    overlap_atomic,
)
from .sto_overlap import (
    slater_koster_axial,
    overlap_sp_general,
)
from .density_matrix import (
    PTMolecularBasis,
    build_molecular_basis,
    overlap_matrix,
    hueckel_hamiltonian,
    solve_mo,
    density_matrix_PT,
    mulliken_populations,
)
from .giao import (
    evaluate_sto,
    evaluate_sto_gradient,
    nuclear_attraction_matrix,
    shielding_diamagnetic_iso,
    shielding_diamagnetic_tensor,
    shielding_at_point,
    H_atom_lamb_ppm,
    angular_momentum_matrices,
    angular_momentum_matrices_GIAO,
    momentum_matrix,
    magnetic_dipole_matrices,
    paramagnetic_shielding_iso,
    shielding_total_iso,
)
from .shielding import (
    GIAOShieldingTensor,
    paramagnetic_shielding_tensor,
    shielding_tensor_at_point,
    nics_iso_giao,
    nics_zz_giao,
)
from .fock import (
    coulomb_J_matrix,
    exchange_K_matrix,
    fock_matrix,
    density_matrix_PT_scf,
    DIIS,
    coupled_cphf_response,
    paramagnetic_shielding_iso_coupled,
    paramagnetic_shielding_tensor_coupled,
)
from .mp2 import (
    MP2Result,
    mp2_amplitudes,
    mp2_energy,
    mp2_density_correction,
    mp2_density_correction_AO,
    mp2_at_hf,
    mp2_relax_orbitals,
    mo_eri_iajb,
)

__all__ = [
    "PTAtomicOrbital",
    "PTAtomBasis",
    "build_atom_basis",
    "Z_eff_shell",
    "occupied_shells",
    "overlap_atomic",
    "slater_koster_axial",
    "overlap_sp_general",
    "PTMolecularBasis",
    "build_molecular_basis",
    "overlap_matrix",
    "hueckel_hamiltonian",
    "solve_mo",
    "density_matrix_PT",
    "mulliken_populations",
    "evaluate_sto",
    "nuclear_attraction_matrix",
    "shielding_diamagnetic_iso",
    "shielding_diamagnetic_tensor",
    "shielding_at_point",
    "H_atom_lamb_ppm",
    "evaluate_sto_gradient",
    "angular_momentum_matrices",
    "angular_momentum_matrices_GIAO",
    "magnetic_dipole_matrices",
    "momentum_matrix",
    "paramagnetic_shielding_iso",
    "shielding_total_iso",
    "GIAOShieldingTensor",
    "paramagnetic_shielding_tensor",
    "shielding_tensor_at_point",
    "nics_iso_giao",
    "nics_zz_giao",
    "coulomb_J_matrix",
    "exchange_K_matrix",
    "fock_matrix",
    "density_matrix_PT_scf",
    "DIIS",
    "coupled_cphf_response",
    "paramagnetic_shielding_iso_coupled",
    "paramagnetic_shielding_tensor_coupled",
    "MP2Result",
    "mp2_amplitudes",
    "mp2_energy",
    "mp2_density_correction",
    "mp2_density_correction_AO",
    "mp2_at_hf",
    "mp2_relax_orbitals",
    "mo_eri_iajb",
]
