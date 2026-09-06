import numpy as np
from src.bess_model import LiIonBatteryModel
from src.profile_generator import LoadProfileGenerator
from src.thermal_model import ThermalBESSModel
from src.visualizer import plot_bess_response

def main():
    battery = LiIonBatteryModel(capacity_ah=2.5, r0=0.02, r1=0.015, c1=2000.0)
    thermal = ThermalBESSModel()
    
    dt = 1.0
    total_time = 1800
    time_array = np.arange(0, total_time, dt)
    
    profile = LoadProfileGenerator.pulse_discharge(total_time=total_time, i_pulse=7.5, pulse_dur=300, rest_dur=300, dt=dt)
    
    soc, v_rc = 1.0, 0.0
    t_cell = 298.15  # 25.0 °C in Kelvin
    
    v_term_hist, v_rc_hist, soc_hist, temp_c_hist = [], [], [], []
    
    for i_app in profile:
        r0_act = thermal.arrhenius_resistance(battery.r0, t_cell)
        r1_act = thermal.arrhenius_resistance(battery.r1, t_cell)
        
        battery.r0 = r0_act
        battery.r1 = r1_act
        
        soc, v_rc, v_term = battery.step(i_app, dt, soc, v_rc)
        t_cell = thermal.step_thermal(i_app, r0_act, r1_act, t_cell, dt)
        
        v_term_hist.append(v_term)
        v_rc_hist.append(v_rc)
        soc_hist.append(soc)
        temp_c_hist.append(t_cell - 273.15)
        
    plot_bess_response(time_array, profile, v_term_hist, v_rc_hist, soc_hist, save_path='results/bess_dynamic_response.png')
    
    print("=" * 70)
    print("DAY 14: BESS ELECTRO-THERMAL DYNAMIC SIMULATION EVALUATION")
    print("=" * 70)
    print(f"Simulation Duration:               {total_time} s")
    print(f"Initial State-of-Charge (SoC):     100.0 %")
    print(f"Final State-of-Charge (SoC):       {soc_hist[-1]*100:.2f} %")
    print(f"Minimum Terminal Voltage (V_term): {min(v_term_hist):.3f} V")
    print(f"Initial Cell Temperature:          25.00 °C")
    print(f"Peak Cell Temperature:             {max(temp_c_hist):.2f} °C")
    print("=" * 70)

if __name__ == "__main__":
    main()
