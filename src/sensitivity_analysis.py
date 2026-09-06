import numpy as np
from src.bess_model import LiIonBatteryModel
from src.thermal_model import ThermalBESSModel
from src.profile_generator import LoadProfileGenerator

def run_thermal_sensitivity_sweep(amb_temps_c=[10, 25, 40, 50], c_rates=[1.0, 2.0, 3.0, 4.0]):
    """
    Executes multi-parameter sweep across ambient temperatures and discharge C-rates
    to evaluate peak cell temperature and minimum terminal voltage boundaries.
    """
    results = []
    nominal_cap_ah = 2.5
    
    for t_amb_c in amb_temps_c:
        for c_rate in c_rates:
            i_pulse = c_rate * nominal_cap_ah
            battery = LiIonBatteryModel(capacity_ah=nominal_cap_ah)
            thermal = ThermalBESSModel(t_amb_k=t_amb_c + 273.15)
            
            dt = 1.0
            total_time = 1800
            profile = LoadProfileGenerator.pulse_discharge(total_time=total_time, i_pulse=i_pulse, dt=dt)
            
            soc, v_rc = 1.0, 0.0
            t_cell_k = t_amb_c + 273.15
            v_term_list, temp_c_list = [], []
            
            for i_app in profile:
                r0_act = thermal.arrhenius_resistance(battery.r0, t_cell_k)
                r1_act = thermal.arrhenius_resistance(battery.r1, t_cell_k)
                battery.r0, battery.r1 = r0_act, r1_act
                
                soc, v_rc, v_term = battery.step(i_app, dt, soc, v_rc)
                t_cell_k = thermal.step_thermal(i_app, r0_act, r1_act, t_cell_k, dt)
                
                v_term_list.append(v_term)
                temp_c_list.append(t_cell_k - 273.15)
                
            results.append({
                'amb_temp_c': t_amb_c,
                'c_rate': c_rate,
                'min_v_term': min(v_term_list),
                'peak_temp_c': max(temp_c_list)
            })
    return results
