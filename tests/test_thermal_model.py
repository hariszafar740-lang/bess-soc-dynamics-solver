import math

import pytest

from src.thermal_model import (
    EA_J_MOL,
    R_GAS,
    T_MAX_K,
    T_MIN_K,
    T_REF_K,
    ThermalBESSModel,
    arrhenius_factor,
)


# ---------------------------------------------------------------------------
# Original behaviour
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Arrhenius correction
# ---------------------------------------------------------------------------
def test_arrhenius_factor_is_unity_at_reference_and_monotone():
    assert arrhenius_factor(T_REF_K) == pytest.approx(1.0, rel=1e-15)
    temps = [253.15 + 5.0 * k for k in range(17)]
    factors = [arrhenius_factor(t) for t in temps]
    assert all(b < a for a, b in zip(factors, factors[1:]))  # strictly decreasing in T


def test_arrhenius_factor_closed_form_at_10c():
    expected = math.exp((EA_J_MOL / R_GAS) * (1.0 / 283.15 - 1.0 / 298.15))
    assert arrhenius_factor(283.15) == pytest.approx(expected, rel=1e-14)
    assert 1.5 < expected < 1.6  # ~1.53 for Ea = 20 kJ/mol


def test_arrhenius_resistance_is_pure_function_of_reference_value():
    thermal = ThermalBESSModel()
    r_ref = 0.02
    first = thermal.arrhenius_resistance(r_ref, 283.15)
    for _ in range(100):
        assert thermal.arrhenius_resistance(r_ref, 283.15) == first
    assert r_ref == 0.02


@pytest.mark.parametrize("t_k", [0.0, -10.0, float("nan"), float("inf")])
def test_arrhenius_factor_rejects_non_physical_temperature(t_k):
    with pytest.raises(ValueError):
        arrhenius_factor(t_k)


def test_arrhenius_resistance_rejects_negative_reference():
    with pytest.raises(ValueError):
        ThermalBESSModel().arrhenius_resistance(-0.01, 298.15)


# ---------------------------------------------------------------------------
# Thermal balance: closed-form and steady-state validation
# ---------------------------------------------------------------------------
def test_single_step_matches_closed_form_exact_solution():
    thermal = ThermalBESSModel(m_kg=0.045, cp_j_kgk=900.0, h_a_w_k=0.25, t_amb_k=298.15)
    i_app, r0, r1 = 2.5, 0.02, 0.015
    t0, dt = 305.0, 400.0  # dt larger than m*Cp/hA = 162 s
    q = i_app ** 2 * (r0 + r1)
    t_eq = 298.15 + q / 0.25
    expected = t_eq + (t0 - t_eq) * math.exp(-0.25 * dt / (0.045 * 900.0))
    assert thermal.step_thermal(i_app, r0, r1, t0, dt) == pytest.approx(expected, abs=1e-12)


def test_large_time_step_is_stable_and_monotone_toward_equilibrium():
    """Regression: explicit Euler oscillated/diverged for dt > m*Cp/hA (162 s)."""
    thermal = ThermalBESSModel()
    i_app, r0, r1 = 10.0, 0.02, 0.015
    t_eq = thermal.t_amb + i_app ** 2 * (r0 + r1) / thermal.h_a
    t = 298.15
    previous_gap = abs(t_eq - t)
    for _ in range(20):
        t = thermal.step_thermal(i_app, r0, r1, t, dt=1000.0)
        assert math.isfinite(t)
        gap = abs(t_eq - t)
        assert gap <= previous_gap + 1e-12  # never overshoots the equilibrium
        previous_gap = gap
    assert t == pytest.approx(t_eq, abs=1e-6)


def test_constant_current_converges_to_analytic_steady_state():
    thermal = ThermalBESSModel()
    i_app, r0, r1 = 2.5, 0.02, 0.015
    t = thermal.t_amb
    for _ in range(2000):
        t = thermal.step_thermal(i_app, r0, r1, t, dt=10.0)
    expected = thermal.t_amb + i_app ** 2 * (r0 + r1) / thermal.h_a
    assert t == pytest.approx(expected, abs=1e-9)


def test_no_current_cools_toward_ambient():
    thermal = ThermalBESSModel(t_amb_k=283.15)
    t = 313.15
    t_next = thermal.step_thermal(0.0, 0.02, 0.015, t, dt=10.0)
    assert 283.15 < t_next < t


def test_joule_heating_is_sign_independent():
    thermal = ThermalBESSModel()
    charge = thermal.step_thermal(-5.0, 0.02, 0.015, 298.15, dt=5.0)
    discharge = thermal.step_thermal(5.0, 0.02, 0.015, 298.15, dt=5.0)
    assert charge == pytest.approx(discharge, rel=1e-15)


def test_adiabatic_cell_heats_linearly_with_energy():
    thermal = ThermalBESSModel(h_a_w_k=0.0)
    q = 3.0 ** 2 * (0.02 + 0.015)
    t_next = thermal.step_thermal(3.0, 0.02, 0.015, 300.0, dt=2.0)
    assert t_next == pytest.approx(300.0 + q * 2.0 / (0.045 * 900.0), rel=1e-14)
    assert math.isinf(ThermalBESSModel(h_a_w_k=0.0).thermal_time_constant_s)


def test_thermal_time_constant_matches_mcp_over_ha():
    assert ThermalBESSModel().thermal_time_constant_s == pytest.approx(162.0)


# ---------------------------------------------------------------------------
# Constructor and input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"m_kg": 0.0},
    {"cp_j_kgk": -1.0},
    {"h_a_w_k": -0.1},
    {"t_amb_k": 0.0},
    {"t_amb_k": 10.0},  # Celsius passed as Kelvin
])
def test_invalid_thermal_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ThermalBESSModel(**kwargs)


@pytest.mark.parametrize("t_k", [T_MIN_K - 1.0, T_MAX_K + 1.0, float("nan")])
def test_step_rejects_temperature_outside_envelope(t_k):
    with pytest.raises(ValueError):
        ThermalBESSModel().step_thermal(1.0, 0.02, 0.015, t_k, dt=1.0)


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan")])
def test_step_rejects_invalid_time_step(dt):
    with pytest.raises(ValueError):
        ThermalBESSModel().step_thermal(1.0, 0.02, 0.015, 298.15, dt=dt)


@pytest.mark.parametrize("args", [
    (float("nan"), 0.02, 0.015),
    (1.0, float("inf"), 0.015),
    (1.0, 0.02, -0.001),
])
def test_step_rejects_non_finite_or_negative_inputs(args):
    with pytest.raises(ValueError):
        ThermalBESSModel().step_thermal(*args, t_cell_k=298.15, dt=1.0)
