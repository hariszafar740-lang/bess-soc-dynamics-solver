import numpy as np
from src.profile_generator import LoadProfileGenerator

def test_pulse_profile_shape_and_bounds():
    profile = LoadProfileGenerator.pulse_discharge(total_time=600, i_pulse=5.0, dt=1.0)
    assert len(profile) == 600
    assert np.max(profile) == 5.0
    assert np.min(profile) == 0.0

def test_grid_profile_bounds():
    profile = LoadProfileGenerator.dynamic_grid_step(total_time=3600, dt=1.0)
    assert len(profile) == 3600
    assert np.max(profile) <= 5.0
    assert np.min(profile) >= -3.0


def test_grid_profile_is_deterministic_and_does_not_reseed_global_rng():
    np.random.seed(123)
    expected_next = np.random.rand()
    np.random.seed(123)
    first = LoadProfileGenerator.dynamic_grid_step(total_time=3600, dt=1.0)
    observed_next = np.random.rand()
    assert observed_next == expected_next  # global RNG state not consumed/reset by the generator
    second = LoadProfileGenerator.dynamic_grid_step(total_time=3600, dt=1.0)
    np.testing.assert_array_equal(first, second)


def test_pulse_profile_duty_cycle():
    profile = LoadProfileGenerator.pulse_discharge(total_time=1800, i_pulse=2.5, pulse_dur=300, rest_dur=300, dt=1.0)
    assert np.count_nonzero(profile) == 900
    assert np.all(profile[:300] == 2.5) and np.all(profile[300:600] == 0.0)
