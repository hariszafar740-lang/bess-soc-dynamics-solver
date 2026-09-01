import pytest
from src.bess_model import LiIonBatteryModel

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
