import numpy as np

from src.bess_model import LiIonBatteryModel
from src.ecm_2rc import run_chemistry_comparison_benchmark
from src.profile_generator import LoadProfileGenerator
from src.sensitivity_analysis import run_thermal_sensitivity_sweep
from src.thermal_model import ThermalBESSModel
from src.visualizer import plot_bess_response, plot_chemistry_2rc_comparison


def generate_reference_dynamic_figure(save_path: str = "results/bess_dynamic_response.png") -> dict[str, float]:
    """Run the documented 3C (7.5 A) pulsed electro-thermal trajectory and export Figure 1."""
    dt = 1.0
    total_time = 1800
    t_amb_k = 298.15  # 25.0 degC

    battery = LiIonBatteryModel(capacity_ah=2.5, r0=0.02, r1=0.015, c1=2000.0)
    thermal = ThermalBESSModel(t_amb_k=t_amb_k)
    i_profile = LoadProfileGenerator.pulse_discharge(
        total_time=total_time, i_pulse=7.5, pulse_dur=300, rest_dur=300, dt=dt
    )
    time_array = np.arange(len(i_profile), dtype=float) * dt

    soc, v_rc = 1.0, 0.0
    t_cell_k = t_amb_k
    v_term_hist, v_rc_hist, soc_hist, ocv_hist, temp_c_hist = [], [], [], [], []

    for i_app in i_profile:
        battery.set_temperature(t_cell_k)
        soc, v_rc, v_term = battery.step(float(i_app), dt, soc, v_rc)
        t_cell_k = thermal.step_thermal(float(i_app), battery.r0, battery.r1, t_cell_k, dt)
        v_term_hist.append(v_term)
        v_rc_hist.append(v_rc)
        soc_hist.append(soc)
        ocv_hist.append(battery.get_ocv(soc))
        temp_c_hist.append(t_cell_k - 273.15)

    plot_bess_response(
        time_array,
        i_profile,
        v_term_hist,
        v_rc_hist,
        soc_hist,
        save_path=save_path,
        ocv=ocv_hist,
        temp_c=temp_c_hist,
    )
    return {
        "min_v_term": float(min(v_term_hist)),
        "peak_temp_c": float(max(temp_c_hist)),
        "final_soc": float(soc_hist[-1]),
    }


def generate_chemistry_comparison_figure(
    save_path: str = "results/chemistry_2rc_comparison.png",
) -> dict[str, float]:
    """Run the 2RC NMC811 vs. LFP electro-thermal & EIS benchmark and export Figure 2."""
    comp = run_chemistry_comparison_benchmark(dt=1.0)
    plot_chemistry_2rc_comparison(comp, save_path=save_path)
    eis_10 = comp.eis_nmc_by_temp_c[10]
    eis_25 = comp.eis_nmc_by_temp_c[25]
    return {
        "nmc_hppc_min_v": float(np.min(comp.nmc_hppc.v_term_v)),
        "lfp_hppc_min_v": float(np.min(comp.lfp_hppc.v_term_v)),
        "nmc_1c_min_q_rev_w": float(np.min(comp.nmc_discharge_1c.q_rev_w)),
        "nmc_1c_max_q_rev_w": float(np.max(comp.nmc_discharge_1c.q_rev_w)),
        "nmc_rdc_25c_mohm": float((eis_25.r0_ohm + eis_25.r1_ohm + eis_25.r2_ohm) * 1e3),
        "nmc_rdc_10c_mohm": float((eis_10.r0_ohm + eis_10.r1_ohm + eis_10.r2_ohm) * 1e3),
    }


def main() -> None:
    ref_metrics = generate_reference_dynamic_figure()
    chem_metrics = generate_chemistry_comparison_figure()

    print("=" * 75)
    print("BESS ELECTRO-THERMAL PARAMETER SENSITIVITY & VOLTAGE SAG BENCHMARK")
    print("=" * 75)
    print(
        f"Reference 3C Pulsed Trajectory (25 °C): "
        f"Min V_term = {ref_metrics['min_v_term']:.3f} V | "
        f"Peak T_cell = {ref_metrics['peak_temp_c']:.2f} °C | "
        f"Final SoC = {ref_metrics['final_soc'] * 100.0:.1f}%"
    )
    print(
        f"2RC Chemistry Benchmark (NMC811 vs LFP): "
        f"NMC R_dc(25/10 °C) = {chem_metrics['nmc_rdc_25c_mohm']:.1f}/"
        f"{chem_metrics['nmc_rdc_10c_mohm']:.1f} mΩ | "
        f"1C Q_rev = [{chem_metrics['nmc_1c_min_q_rev_w']:.2f}, "
        f"+{chem_metrics['nmc_1c_max_q_rev_w']:.2f}] W"
    )
    print("-" * 75)
    print(f"{'Ambient (°C)':<15}{'C-Rate':<12}{'Min V_term (V)':<18}{'Peak Temp (°C)':<18}")
    print("-" * 75)

    results = run_thermal_sensitivity_sweep()
    for res in results:
        print(
            f"{res['amb_temp_c']:<15}{res['c_rate']:<12.1f}"
            f"{res['min_v_term']:<18.3f}{res['peak_temp_c']:<18.2f}"
        )
    print("=" * 75)


if __name__ == "__main__":
    main()
