import numpy as np
from src.bess_model import LiIonBatteryModel
from src.profile_generator import LoadProfileGenerator

def main():
    battery = LiIonBatteryModel(capacity_ah=2.5, r0=0.02, r1=0.015, c1=2000.0)
    dt = 1.0
    profile = LoadProfileGenerator.pulse_discharge(total_time=1800, i_pulse=5.0, dt=dt)
    
    soc, v_rc = 1.0, 0.0
    v_term_history = []
    soc_history = []
    
    for i_app in profile:
        soc, v_rc, v_term = battery.step(i_app, dt, soc, v_rc)
        v_term_history.append(v_term)
        soc_history.append(soc)
        
    print("=" * 70)
    print("DAY 12: BESS DYNAMIC LOAD PROFILE SIMULATION EVALUATION")
    print("=" * 70)
    print(f"Initial State-of-Charge (SoC):     100.0 %")
    print(f"Final State-of-Charge (SoC):       {soc_history[-1]*100:.2f} %")
    print(f"Minimum Terminal Voltage (V_term): {min(v_term_history):.3f} V")
    print(f"Maximum Terminal Voltage (V_term): {max(v_term_history):.3f} V")
    print("=" * 70)

if __name__ == "__main__":
    main()
