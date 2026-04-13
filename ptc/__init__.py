"""PTC phase-1 foundation package."""

from ptc.atom import (
    Atom,
    EA_eV,
    IE_eV,
    benchmark_atom_ea_models_against_nist,
    compare_ea_channels,
    effective_charge,
    screening_action,
)
from ptc.base import AtomProvider, BaseCalculator, Result
from ptc.ea_operator import (
    EA_operator_eV,
    AtomicCaptureState,
    HierarchicalEAOperator,
    atomic_capture_state,
    build_atomic_hierarchical_ea_operator,
    build_hierarchical_ea_operator,
    operator_capture_amplitude,
)
from ptc.smiles_parser import is_smiles, parse_smiles

__all__ = [
    "Atom",
    "AtomicCaptureState",
    "AtomProvider",
    "BaseCalculator",
    "EA_eV",
    "EA_operator_eV",
    "HierarchicalEAOperator",
    "IE_eV",
    "Result",
    "atomic_capture_state",
    "benchmark_atom_ea_models_against_nist",
    "build_atomic_hierarchical_ea_operator",
    "build_hierarchical_ea_operator",
    "compare_ea_channels",
    "effective_charge",
    "is_smiles",
    "operator_capture_amplitude",
    "parse_smiles",
    "screening_action",
]
