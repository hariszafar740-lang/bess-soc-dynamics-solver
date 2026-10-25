import math

from .bess_model import LiIonBatteryModel
from .profile_generator import LoadProfileGenerator
from .thermal_model import T_MAX_K, T_MIN_K, ThermalBESSModel


class SimulationError(RuntimeError):
    """Raised when a sweep case produces non-finite or out-of-envelope states."""


def run_thermal_sensitivity_sweep(amb_temps_c=(10, 25, 40, 50), c_rates=(1.0, 2.0, 3.0, 4.0),
                                  nominal_cap_ah=2.5, total_time=1800, dt=1.0):
    """
    Executes a multi-parameter sweep across ambient temperatures and discharge C-rates
    to evaluate peak cell temperature and minimum terminal voltage.

    Each case uses a fresh ECM and thermal model. At every step the ECM's active
    R0/R1 are recomputed from the reference values at the current cell temperature
    (no accumulation across steps). Any non-finite state, or a cell temperature
    outside the documented envelope, raises SimulationError rather than returning
    corrupted results.
    """
    results = []

    for t_amb_c in amb_temps_c:
        for c_rate in c_rates:
            i_pulse = c_rate * nominal_cap_ah
            label = f"ambient={t_amb_c} degC, C-rate={c_rate}"
            battery = LiIonBatteryModel(capacity_ah=nominal_cap_ah)
            thermal = ThermalBESSModel(t_amb_k=t_amb_c + 273.15)

            profile = LoadProfileGenerator.pulse_discharge(total_time=total_time, i_pulse=i_pulse, dt=dt)

            soc, v_rc = 1.0, 0.0
            t_cell_k = t_amb_c + 273.15
            v_term_list, temp_c_list = [], []

            for k, i_app in enumerate(profile):
                try:
                    battery.set_temperature(t_cell_k)
                    soc, v_rc, v_term = battery.step(i_app, dt, soc, v_rc)
                    t_cell_k = thermal.step_thermal(i_app, battery.r0, battery.r1, t_cell_k, dt)
                except (ValueError, FloatingPointError) as exc:
                    raise SimulationError(f"{label}, step {k}: {exc}") from exc

                if not (math.isfinite(v_term) and math.isfinite(t_cell_k)):
                    raise SimulationError(f"{label}, step {k}: non-finite state")
                if not (T_MIN_K <= t_cell_k <= T_MAX_K):
                    raise SimulationError(
                        f"{label}, step {k}: cell temperature {t_cell_k - 273.15:.2f} degC left the "
                        f"supported envelope [{T_MIN_K - 273.15:.0f}, {T_MAX_K - 273.15:.0f}] degC"
                    )

                v_term_list.append(v_term)
                temp_c_list.append(t_cell_k - 273.15)

            results.append({
                'amb_temp_c': t_amb_c,
                'c_rate': c_rate,
                'min_v_term': float(min(v_term_list)),
                'peak_temp_c': float(max(temp_c_list)),
            })
    return results
