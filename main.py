import numpy as np

from src.bess_model import LiIonBatteryModel
from src.bms_thermal_sop import run_thermal_sop_benchmark
from src.degradation_model import run_degradation_benchmark
from src.ecm_2rc import run_chemistry_comparison_benchmark
from src.profile_generator import LoadProfileGenerator
from src.sensitivity_analysis import run_thermal_sensitivity_sweep
from src.soc_estimators import run_soc_estimation_benchmark
from src.thermal_model import ThermalBESSModel
from src.visualizer import (
    plot_bess_response,
    plot_chemistry_2rc_comparison,
    plot_degradation_benchmark,
    plot_soc_estimation_benchmark,
    plot_thermal_sop_derating,
)


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


def generate_soc_estimation_figure(
    save_path: str = "results/soc_ekf_ukf_benchmark.png",
) -> dict[str, float]:
    """Run the closed-loop EKF & UKF vs. Coulomb counting SoC benchmark and export Figure 3."""
    bench = run_soc_estimation_benchmark(dt=1.0)
    plot_soc_estimation_benchmark(bench, save_path=save_path)
    return {
        "nmc_rmse_cc_pct": bench.nmc.rmse_cc_full * 100.0,
        "nmc_rmse_ekf_pct": bench.nmc.rmse_ekf_full * 100.0,
        "nmc_rmse_ukf_pct": bench.nmc.rmse_ukf_full * 100.0,
        "lfp_rmse_cc_pct": bench.lfp.rmse_cc_full * 100.0,
        "lfp_rmse_ekf_pct": bench.lfp.rmse_ekf_full * 100.0,
        "lfp_rmse_ukf_pct": bench.lfp.rmse_ukf_full * 100.0,
    }


def generate_thermal_sop_figure(
    save_path: str = "results/thermal_sop_derating.png",
) -> dict[str, float]:
    """Run the 2-node radial thermal, CC-CV taper & SOP BMS benchmark and export Figure 4."""
    bench = run_thermal_sop_benchmark()
    plot_thermal_sop_derating(bench, save_path=save_path)
    return {
        "peak_core_surf_delta_c": bench.pulse_2node.peak_core_surf_delta_c,
        "nmc_cc_to_cv_min": bench.nmc_cccv.cc_to_cv_time_s / 60.0,
        "lfp_cc_to_cv_min": bench.lfp_cccv.cc_to_cv_time_s / 60.0,
        "nmc_overshoot_v": bench.nmc_cccv.max_v_overshoot_v,
        "lfp_overshoot_v": bench.lfp_cccv.max_v_overshoot_v,
        "sop_30s_peak_dis_w": float(np.max(bench.p_dis_by_horizon_w[30.0])),
    }


def generate_degradation_figure(
    save_path: str = "results/degradation_rainflow_aging.png",
) -> dict[str, float]:
    """Run the ASTM E1049-85 rainflow & SEI/cycle/plating aging benchmark and export Figure 5."""
    bench = run_degradation_benchmark()
    plot_degradation_benchmark(bench, save_path=save_path)
    return {
        "nmc_arb_eol_yr": bench.nmc_arb.eol_years,
        "nmc_arb_eol_efc": bench.nmc_arb.eol_efc,
        "nmc_regd_12yr_soh_pct": float(bench.nmc_regd.soh_pct[-1]),
        "lfp_arb_eol_yr": bench.lfp_arb.eol_years,
        "lfp_arb_eol_efc": bench.lfp_arb.eol_efc,
        "lfp_regd_12yr_soh_pct": float(bench.lfp_regd.soh_pct[-1]),
        "optimal_temp_nmc_c": bench.optimal_temp_nmc_c,
    }


def main() -> None:
    ref_metrics = generate_reference_dynamic_figure()
    chem_metrics = generate_chemistry_comparison_figure()
    est_metrics = generate_soc_estimation_figure()
    sop_metrics = generate_thermal_sop_figure()
    deg_metrics = generate_degradation_figure()

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
    print(
        f"Closed-Loop SoC RMSE (-23% Init, +0.05C Bias): "
        f"NMC CC/EKF/UKF = {est_metrics['nmc_rmse_cc_pct']:.1f}%/"
        f"{est_metrics['nmc_rmse_ekf_pct']:.2f}%/{est_metrics['nmc_rmse_ukf_pct']:.2f}% | "
        f"LFP EKF/UKF = {est_metrics['lfp_rmse_ekf_pct']:.2f}%/{est_metrics['lfp_rmse_ukf_pct']:.2f}%"
    )
    print(
        f"2-Node Thermal & SOP BMS: "
        f"Peak ΔT_core-surf = {sop_metrics['peak_core_surf_delta_c']:.2f} °C | "
        f"2C CC→CV (NMC/LFP) = {sop_metrics['nmc_cc_to_cv_min']:.1f}/"
        f"{sop_metrics['lfp_cc_to_cv_min']:.1f} min | "
        f"30s Peak P_dis = {sop_metrics['sop_30s_peak_dis_w']:.1f} W"
    )
    print(
        f"Rainflow & SEI Aging (12 yr): "
        f"Arb EoL (NMC/LFP) = {deg_metrics['nmc_arb_eol_yr']:.1f}/{deg_metrics['lfp_arb_eol_yr']:.1f} yr | "
        f"RegD 12yr SoH (NMC/LFP) = {deg_metrics['nmc_regd_12yr_soh_pct']:.1f}%/"
        f"{deg_metrics['lfp_regd_12yr_soh_pct']:.1f}%"
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
