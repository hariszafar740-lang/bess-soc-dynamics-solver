import pytest
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
