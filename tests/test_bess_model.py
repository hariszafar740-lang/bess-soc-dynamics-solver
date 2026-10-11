import math
import warnings

import numpy as np
import pytest

from src.bess_model import LiIonBatteryModel
from src.thermal_model import T_MAX_K, T_MIN_K, T_REF_K, arrhenius_factor


# ---------------------------------------------------------------------------
# Original behaviour
# ---------------------------------------------------------------------------
def test_ocv_voltage_bounds():
    battery = LiIonBatteryModel()
    assert pytest.approx(battery.get_ocv(0.0), abs=1e-2) == 3.20
    assert pytest.approx(battery.get_ocv(1.0), abs=1e-2) == 4.10


def test_coulomb_counting_discharge():
    battery = LiIonBatteryModel(capacity_ah=2.5)
    soc, v_rc = 1.0, 0.0

    # Discharge at 2.5A (1C rate) for 1800 seconds (0.5 hour)
    for _ in range(1800):
        soc, v_rc, v_term = battery.step(i_app=2.5, dt=1.0, soc=soc, v_rc=v_rc)

    # Remaining SoC should be exactly 50%
    assert pytest.approx(soc, abs=1e-2) == 0.50
    assert v_term < battery.get_ocv(soc)  # Terminal voltage under load must be lower than OCV


# ---------------------------------------------------------------------------
# Reference parameters, temperature correction, and derived quantities
# ---------------------------------------------------------------------------
def test_reference_parameters_are_not_mutated_by_temperature_changes():
    battery = LiIonBatteryModel(r0=0.02, r1=0.015, c1=2000.0)
    for t_k in (283.15, 253.15, 333.15, 298.15, 268.15, 283.15):
        battery.set_temperature(t_k)
    assert battery.r0_ref == 0.02
    assert battery.r1_ref == 0.015
    assert battery.c1_ref == 2000.0


def test_active_resistance_is_arrhenius_of_reference_not_of_previous_value():
    battery = LiIonBatteryModel(r0=0.02, r1=0.015)
    battery.set_temperature(283.15)
    battery.set_temperature(283.15)  # Repeated call at same temperature must not compound
    expected_factor = arrhenius_factor(283.15)
    assert battery.r0 == pytest.approx(0.02 * expected_factor, rel=1e-14)
    assert battery.r1 == pytest.approx(0.015 * expected_factor, rel=1e-14)


def test_forward_and_reverse_temperature_sequences_give_identical_parameters():
    temps = [253.15, 263.15, 283.15, 298.15, 318.15, 333.15]
    expected = {t_k: LiIonBatteryModel().set_temperature(t_k) for t_k in temps}

    # Same instance, forward order, then reverse order: each point must equal a fresh model.
    for order in (temps, list(reversed(temps))):
        battery = LiIonBatteryModel()
        for t_k in order:
            battery.set_temperature(t_k)
            ref = expected[t_k]
            assert battery.r0 == ref.r0
            assert battery.r1 == ref.r1
            assert battery.tau1 == ref.tau1


def test_arrhenius_direction_and_reference_value():
    battery = LiIonBatteryModel(r0=0.02, r1=0.015)
    battery.set_temperature(T_REF_K)
    assert battery.r0 == pytest.approx(0.02, rel=1e-14)
    battery.set_temperature(283.15)  # 10 degC: cold -> higher resistance
    assert battery.r0 > 0.02
    battery.set_temperature(318.15)  # 45 degC: hot -> lower resistance
    assert battery.r0 < 0.02


def test_rc_time_constant_tracks_resistance_changes():
    """Regression: tau1 was computed once in __init__ and went stale when R1 changed."""
    battery = LiIonBatteryModel(r1=0.015, c1=2000.0)
    assert battery.tau1 == pytest.approx(30.0)
    battery.set_temperature(283.15)
    assert battery.tau1 == pytest.approx(battery.r1 * battery.c1, rel=1e-14)
    assert battery.tau1 > 30.0
    battery.set_temperature(T_REF_K)
    assert battery.tau1 == pytest.approx(30.0, rel=1e-14)


def test_direct_assignment_of_active_parameters_is_rejected():
    """Active parameters are derived; silent overwrites that could leave tau1 stale are disallowed."""
    battery = LiIonBatteryModel()
    with pytest.raises(AttributeError):
        battery.r1 = 0.03
    with pytest.raises(AttributeError):
        battery.tau1 = 1.0


def test_reset_restores_reference_state():
    battery = LiIonBatteryModel(r0=0.02, r1=0.015)
    battery.set_temperature(253.15)
    battery.reset()
    assert battery.t_cell_k == T_REF_K
    assert battery.r0 == pytest.approx(0.02, rel=1e-14)
    assert battery.r1 == pytest.approx(0.015, rel=1e-14)


def test_temperature_envelope_limits_are_accepted_and_outside_rejected():
    battery = LiIonBatteryModel()
    battery.set_temperature(T_MIN_K)
    battery.set_temperature(T_MAX_K)
    with pytest.raises(ValueError):
        battery.set_temperature(T_MIN_K - 1.0)
    with pytest.raises(ValueError):
        battery.set_temperature(T_MAX_K + 1.0)
    with pytest.raises(ValueError):
        battery.set_temperature(float("nan"))
    with pytest.raises(ValueError):
        # A Celsius value passed where Kelvin is required is caught by the envelope check
        battery.set_temperature(10.0)


# ---------------------------------------------------------------------------
# Constructor validation (documented parameter limits)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"capacity_ah": 0.0},
    {"capacity_ah": -1.0},
    {"r0": -0.001},
    {"r1": 0.0},
    {"c1": 0.0},
    {"c1": float("inf")},
    {"r0": float("nan")},
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LiIonBatteryModel(**kwargs)


def test_parameter_limit_r0_zero_is_allowed():
    battery = LiIonBatteryModel(r0=0.0)
    soc, v_rc, v_term = battery.step(i_app=2.0, dt=1.0, soc=0.5, v_rc=0.0)
    assert v_term == pytest.approx(battery.get_ocv(soc) - v_rc, abs=1e-15)


# ---------------------------------------------------------------------------
# Battery operation: analytic validation
# ---------------------------------------------------------------------------
def test_rc_polarization_matches_closed_form_for_constant_current():
    battery = LiIonBatteryModel(capacity_ah=2.5, r0=0.02, r1=0.015, c1=2000.0)
    current = 2.5
    soc, v_rc = 1.0, 0.0
    for k in range(1, 1801):
        soc, v_rc, _ = battery.step(current, 1.0, soc, v_rc)
        expected = current * battery.r1 * (1.0 - math.exp(-k / battery.tau1))
        assert v_rc == pytest.approx(expected, abs=1e-12)


def test_terminal_voltage_equation_under_load():
    battery = LiIonBatteryModel()
    soc, v_rc, v_term = battery.step(i_app=2.5, dt=1.0, soc=0.8, v_rc=0.01)
    assert v_term == pytest.approx(battery.get_ocv(soc) - 2.5 * battery.r0 - v_rc, abs=1e-12)


def test_charge_raises_soc_and_terminal_voltage_above_ocv():
    battery = LiIonBatteryModel()
    soc, v_rc, v_term = battery.step(i_app=-2.5, dt=1.0, soc=0.5, v_rc=0.0)
    assert soc > 0.5
    assert v_term > battery.get_ocv(soc)


def test_idle_operation_relaxes_polarization_exponentially():
    battery = LiIonBatteryModel()
    soc, v_rc = 0.6, 0.05
    v0 = v_rc
    for k in range(1, 301):
        soc, v_rc, v_term = battery.step(0.0, 1.0, soc, v_rc)
        assert v_rc == pytest.approx(v0 * math.exp(-k / battery.tau1), abs=1e-12)
    assert soc == 0.6
    assert v_term == pytest.approx(battery.get_ocv(0.6) - v_rc, abs=1e-12)


def test_full_idle_from_rest_gives_ocv():
    battery = LiIonBatteryModel()
    soc, v_rc, v_term = battery.step(0.0, 1.0, 0.7, 0.0)
    assert v_rc == 0.0
    assert v_term == pytest.approx(battery.get_ocv(0.7), abs=1e-15)


def test_rc_step_is_stable_when_dt_exceeds_twice_tau():
    """Regression: explicit Euler diverged (sign-alternating, exponentially growing) for dt > 2*tau."""
    battery = LiIonBatteryModel(r1=0.001, c1=1e-3)  # tau1 = 1e-6 s
    soc, v_rc = 0.9, 0.0
    for _ in range(50):
        soc, v_rc, v_term = battery.step(1.0, 1.0, soc, v_rc)
        assert math.isfinite(v_term)
        assert 0.0 <= v_rc <= 1.0 * battery.r1 + 1e-15  # bounded by I*R1 for constant discharge
    assert v_rc == pytest.approx(battery.r1, rel=1e-9)


def test_soc_boundary_clipping_emits_runtime_warning():
    battery = LiIonBatteryModel(capacity_ah=2.5)
    with pytest.warns(RuntimeWarning, match="SoC limit reached"):
        soc, _, _ = battery.step(i_app=100.0, dt=1000.0, soc=0.1, v_rc=0.0)
    assert soc == 0.0


def test_mid_range_charge_emits_no_warning():
    battery = LiIonBatteryModel()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        soc, _, _ = battery.step(i_app=-2.5, dt=1.0, soc=0.5, v_rc=0.0)
    assert 0.5 < soc < 1.0


def test_charging_a_full_cell_warns_and_clips_to_one():
    battery = LiIonBatteryModel()
    with pytest.warns(RuntimeWarning, match="SoC limit reached"):
        soc, _, _ = battery.step(i_app=-0.0001, dt=1.0, soc=1.0, v_rc=0.0)
    assert soc == 1.0


# ---------------------------------------------------------------------------
# Invalid inputs (documented policy: raise ValueError / FloatingPointError)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("soc", [-0.1, 1.2, float("nan"), float("inf")])
def test_invalid_initial_soc_is_rejected(soc):
    battery = LiIonBatteryModel()
    with pytest.raises(ValueError):
        battery.step(1.0, 1.0, soc, 0.0)


@pytest.mark.parametrize("soc", [-0.5, 1.5])
def test_get_ocv_rejects_out_of_domain_soc(soc):
    with pytest.raises(ValueError):
        LiIonBatteryModel().get_ocv(soc)


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_time_step_is_rejected(dt):
    with pytest.raises(ValueError):
        LiIonBatteryModel().step(1.0, dt, 0.5, 0.0)


@pytest.mark.parametrize("i_app", [float("nan"), float("inf")])
def test_non_finite_current_is_rejected(i_app):
    with pytest.raises(ValueError):
        LiIonBatteryModel().step(i_app, 1.0, 0.5, 0.0)


def test_non_finite_polarization_state_is_rejected():
    with pytest.raises(ValueError):
        LiIonBatteryModel().step(1.0, 1.0, 0.5, float("nan"))


def test_ocv_is_finite_and_monotone_over_domain():
    battery = LiIonBatteryModel()
    socs = np.linspace(0.0, 1.0, 201)
    ocv = np.array([battery.get_ocv(s) for s in socs])
    assert np.all(np.isfinite(ocv))
    assert np.all(np.diff(ocv) > 0.0)
    assert ocv[0] == pytest.approx(3.2)
    assert ocv[-1] == pytest.approx(4.1)
