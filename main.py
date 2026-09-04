import numpy as np
from src.bess_model import LiIonBatteryModel
from src.profile_generator import LoadProfileGenerator
from src.visualizer import plot_bess_response

def main():
    battery = LiIonBatteryModel(capacity_ah=2.5, r0=0.02, r1=0.015, c1=2000.0)
    dt = 1.0
    total_time = 1800
    time_array = np.arange(0, total_time, dt)
    
    profile = LoadProfileGenerator.pulse_discharge(total_time=total_time, i_pulse=5.0, pulse_dur=300, rest_dur=300, dt=dt)
    
    soc, v_rc = 1.0, 0.0
    v_term_hist, v_rc_hist, soc_hist = [], [], []
    
    for i_app in profile:
        soc, v_rc, v_term = battery.step(i_app, dt, soc, v_rc)
        v_term_hist.append(v_term)
        v_rc_hist.append(v_rc)
        soc_hist.append(soc)
        
    plot_bess_response(time_array, profile, v_term_hist, v_rc_hist, soc_hist, save_path='results/bess_dynamic_response.png')
    
    print("=" * 70)
    print("DAY 13: BESS DYNAMIC RESPONSE VISUALIZATION & EVALUATION")
    print("=" * 70)
    print(f"Simulation Duration:               {total_time} s")
    print(f"Initial State-of-Charge (SoC):     100.0 %")
    print(f"Final State-of-Charge (SoC):       {soc_hist[-1]*100:.2f} %")
    print(f"Minimum Terminal Voltage (V_term): {min(v_term_hist):.3f} V")
    print(f"Maximum Terminal Voltage (V_term): {max(v_term_hist):.3f} V")
    print(f"Maximum RC Polarization Drop:      {max(v_rc_hist):.3f} V")
    print("=" * 70)

if __name__ == "__main__":
    main()
