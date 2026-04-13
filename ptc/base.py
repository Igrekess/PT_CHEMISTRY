"""Base infrastructure for the PTC package.

`AtomProvider` is the single entry point for selecting atomic data sources.
It lets the rest of the engine stay unchanged while switching between:

- ``full_pt``: all atomic inputs derived from PT
- ``hybrid``: PT by default, with controlled NIST injection per property/atom
- ``full_nist``: use official measurements whenever available
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_SOURCE_ALIASES = {
    "pt": "full_pt",
    "full_pt": "full_pt",
    "hybrid": "hybrid",
    "exp": "full_nist",
    "experimental": "full_nist",
    "nist": "full_nist",
    "full_nist": "full_nist",
}

_PROPERTY_ALIASES = {
    "IE": "IE",
    "EA": "EA",
    "MASS": "MASS",
    "M": "MASS",
}


def _normalize_source(source: str) -> str:
    """Normalize legacy and canonical source names."""
    try:
        return _SOURCE_ALIASES[source.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_SOURCE_ALIASES.values())))
        raise ValueError(f"Unknown source '{source}'. Allowed canonical values: {allowed}") from exc


def _normalize_property(prop: str) -> str:
    """Normalize property names used by the provider."""
    key = prop.upper()
    try:
        return _PROPERTY_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_PROPERTY_ALIASES))
        raise ValueError(f"Unknown property '{prop}'. Allowed values: {allowed}") from exc


@dataclass
class Result:
    """A computed result plus lightweight metadata."""

    value: float
    unit: str = "eV"
    label: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class AtomProvider:
    """Resolve atomic inputs from PT, NIST, or a hybrid combination.

    Resolution order for each property:
    1. Direct per-atom override, e.g. ``overrides={6: {"IE": 11.5}}``
    2. Per-property source, e.g. ``ie_source="full_nist"``
    3. Global source, e.g. ``source="full_pt"`` / ``"hybrid"`` / ``"full_nist"``

    Notes
    -----
    ``hybrid`` is PT-native by default. It only switches to NIST when a
    per-property source or a local override explicitly requests it.
    """

    def __init__(self, source: str = "full_pt", overrides: dict[int, dict[str, Any]] | None = None, **kwargs: Any):
        self.source = _normalize_source(source)
        self.overrides = overrides or {}
        self._prop_sources: dict[str, str] = {}

        for key in ("ie_source", "ea_source", "mass_source"):
            if key in kwargs:
                prop = _normalize_property(key.replace("_source", ""))
                self._prop_sources[prop] = _normalize_source(kwargs[key])

    def property_source(self, prop: str) -> str:
        """Return the effective canonical source for a given property."""
        prop_name = _normalize_property(prop)
        if prop_name in self._prop_sources:
            return self._prop_sources[prop_name]
        if self.source == "full_nist":
            return "full_nist"
        return "full_pt"

    def resolve(self, prop: str, Z: int, pt_value: float, exp_value: float) -> float:
        """Resolve a property value using overrides, per-property, then global mode."""
        prop_name = _normalize_property(prop)

        if Z in self.overrides and prop_name in self.overrides[Z]:
            return self.overrides[Z][prop_name]

        if self.property_source(prop_name) == "full_nist":
            return exp_value
        return pt_value


class BaseCalculator:
    """Base class for PTC calculators sharing a single AtomProvider."""

    def __init__(self, source: str = "full_pt", overrides: dict[int, dict[str, Any]] | None = None, **kwargs: Any):
        self.atom_provider = AtomProvider(source=source, overrides=overrides, **kwargs)

    def compute(self, **kwargs: Any) -> Result:
        raise NotImplementedError

    def benchmark(self) -> dict[str, Any]:
        raise NotImplementedError
