import pytest
import numpy as np
from src.thermal_model import ThermalBESSModel

def test_arrhenius_resistance_scaling():
    thermal = ThermalBESSModel()
    r_ref = 0.02
    r_hot = thermal.arrhenius_resistance(r_ref, t_cell_k=318.15)
    assert r_hot < r_ref
    r_cold = thermal.arrhenius_resistance(r_ref, t_cell_k=278.15)
    assert r_cold > r_ref

def test_thermal_step_joule_heating():
    thermal = ThermalBESSModel()
    t_initial = 298.15
    t_next = thermal.step_thermal(i_app=10.0, r0_act=0.02, r1_act=0.015, t_cell_k=t_initial, dt=1.0)
    assert t_next > t_initial
