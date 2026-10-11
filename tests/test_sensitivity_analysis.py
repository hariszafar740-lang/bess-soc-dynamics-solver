import math

import pytest

from src.bess_model import LiIonBatteryModel
from src.sensitivity_analysis import SimulationError, run_thermal_sensitivity_sweep
from src.thermal_model import ThermalBESSModel, arrhenius_factor


def test_sensitivity_sweep_output_structure():
    results = run_thermal_sensitivity_sweep(amb_temps_c=[25], c_rates=[1.0, 2.0])
    assert len(results) == 2
    assert 'min_v_term' in results[0]
    assert 'peak_temp_c' in results[0]
    assert results[1]['peak_temp_c'] > results[0]['peak_temp_c']


def test_regression_10c_1c_sweep_is_finite_and_physical():
    """
    Regression for the reported defect: at 10 degC / 1C the previous code returned
    min V_term ~ -3.08e18 V and peak T ~ 3.32e17 degC because R0/R1 were rescaled
    from already-scaled values at every step (~1.53x per step, compounding).
    """
    (res,) = run_thermal_sensitivity_sweep(amb_temps_c=[10], c_rates=[1.0])
    assert math.isfinite(res['min_v_term']) and math.isfinite(res['peak_temp_c'])
    # Terminal voltage must stay within the OCV envelope of the cell
    assert 3.0 < res['min_v_term'] < LiIonBatteryModel().get_ocv(1.0)
    # Cell must not heat by more than the analytic steady-state bound (see below)
    assert 10.0 < res['peak_temp_c'] < 60.0


def test_peak_temperature_respects_analytic_upper_bound_at_10c_1c():
    """
    Upper bound: with R(T) <= R(T_amb) for T >= T_amb (Arrhenius is decreasing in T),
    the cell heat source never exceeds I^2*(R0+R1)(T_amb), so the lumped energy balance
    gives  T_peak <= T_amb + I^2*(R0+R1)(T_amb)/h_A.
    """
    i_pulse = 2.5
    bound_rise = i_pulse ** 2 * (0.02 + 0.015) * arrhenius_factor(283.15) / 0.25
    (res,) = run_thermal_sensitivity_sweep(amb_temps_c=[10], c_rates=[1.0])
    assert res['peak_temp_c'] <= 10.0 + bound_rise + 1e-9
    assert res['peak_temp_c'] > 10.0  # heating does occur


def test_cold_cell_has_lower_minimum_voltage_than_warm_cell():
    cold = run_thermal_sensitivity_sweep(amb_temps_c=[10], c_rates=[1.0])[0]
    warm = run_thermal_sensitivity_sweep(amb_temps_c=[25], c_rates=[1.0])[0]
    assert cold['min_v_term'] < warm['min_v_term']


def test_trends_with_c_rate_are_monotone():
    results = run_thermal_sensitivity_sweep(amb_temps_c=[10, 50], c_rates=[1.0, 2.0, 3.0, 4.0])
    for amb in (10, 50):
        rows = [r for r in results if r['amb_temp_c'] == amb]
        vmin = [r['min_v_term'] for r in rows]
        tpk = [r['peak_temp_c'] for r in rows]
        assert all(b < a for a, b in zip(vmin, vmin[1:]))
        assert all(b > a for a, b in zip(tpk, tpk[1:]))


def test_forward_and_reverse_sweep_orders_give_identical_results():
    forward = run_thermal_sensitivity_sweep(amb_temps_c=[10, 25, 50], c_rates=[1.0, 4.0])
    reverse = run_thermal_sensitivity_sweep(amb_temps_c=[50, 25, 10], c_rates=[4.0, 1.0])
    lookup = {(r['amb_temp_c'], r['c_rate']): r for r in reverse}
    for r in forward:
        other = lookup[(r['amb_temp_c'], r['c_rate'])]
        assert r['min_v_term'] == other['min_v_term']
        assert r['peak_temp_c'] == other['peak_temp_c']


def test_repeated_sweeps_are_identical():
    first = run_thermal_sensitivity_sweep(amb_temps_c=[10], c_rates=[1.0, 2.0])
    second = run_thermal_sensitivity_sweep(amb_temps_c=[10], c_rates=[1.0, 2.0])
    assert first == second


def test_default_sweep_all_cases_finite_and_bounded():
    results = run_thermal_sensitivity_sweep()
    assert len(results) == 16
    for r in results:
        assert math.isfinite(r['min_v_term']) and math.isfinite(r['peak_temp_c'])
        assert 2.5 < r['min_v_term'] < 4.1
        assert r['amb_temp_c'] <= r['peak_temp_c'] < 60.0


def test_runaway_case_raises_truthful_diagnostic_instead_of_corrupt_results():
    """An extreme current drives the cell outside the supported envelope: must raise, not return numbers."""
    with pytest.raises(SimulationError, match="supported envelope|non-finite"):
        run_thermal_sensitivity_sweep(amb_temps_c=[50], c_rates=[40.0])


def test_same_model_instance_reused_across_temperatures_matches_fresh_instances():
    """Repeated temperature changes on one ECM must not depend on history."""
    shared = LiIonBatteryModel()
    for t_k in (283.15, 323.15, 258.15, 298.15, 283.15):
        shared.set_temperature(t_k)
        fresh = LiIonBatteryModel().set_temperature(t_k)
        assert shared.r0 == fresh.r0 and shared.r1 == fresh.r1 and shared.tau1 == fresh.tau1
        # Independent cross-check of the thermal resistance helper on the same state
        assert shared.r0 == pytest.approx(ThermalBESSModel().arrhenius_resistance(0.02, t_k), rel=1e-14)
