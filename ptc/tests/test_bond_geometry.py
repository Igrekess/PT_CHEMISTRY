"""Tests for BondGeometry 3D."""
import pytest
from ptc.bond import BondGeometry
from ptc.molecule import Molecule


class TestBondGeometryDataclass:
    def test_creation(self):
        bg = BondGeometry(r_e=1.54, theta_A=109.5, theta_B=109.5,
                         sin2_half_A=0.667, sin2_half_B=0.667,
                         lp_proj_A=0.0, lp_proj_B=0.0)
        assert bg.r_e == pytest.approx(1.54)
        assert bg.sin2_half_A == pytest.approx(0.667)

    def test_frozen(self):
        bg = BondGeometry(r_e=1.0, theta_A=90.0, theta_B=90.0,
                         sin2_half_A=0.5, sin2_half_B=0.5,
                         lp_proj_A=0.0, lp_proj_B=0.0)
        with pytest.raises(Exception):
            bg.r_e = 2.0


class TestBondGeometryValues:
    def test_ch4_geometry(self):
        mol = Molecule("C")
        for br in mol.bonds:
            assert br.geometry is not None
            assert 0.5 < br.geometry.r_e < 2.0
            assert br.geometry.lp_proj_A >= 0.0

    def test_h2o_geometry(self):
        mol = Molecule("O")
        for br in mol.bonds:
            assert br.geometry is not None
            assert 0.5 < br.geometry.r_e < 1.5

    def test_geometry_count(self):
        mol = Molecule("CCO")
        for br in mol.bonds:
            assert br.geometry is not None
