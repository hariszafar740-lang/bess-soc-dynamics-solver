import pytest
from src.sensitivity_analysis import run_thermal_sensitivity_sweep

def test_sensitivity_sweep_output_structure():
    results = run_thermal_sensitivity_sweep(amb_temps_c=[25], c_rates=[1.0, 2.0])
    assert len(results) == 2
    assert 'min_v_term' in results[0]
    assert 'peak_temp_c' in results[0]
    assert results[1]['peak_temp_c'] > results[0]['peak_temp_c']
