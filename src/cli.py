"""Command-line interface (`bess-sim` / `python -m bess_solver`) for the BESS solver suite."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence

import numpy as np

from .bess_model import LiIonBatteryModel
from .bms_thermal_sop import (
    CoupledTwoNodeBESS,
    CoupledTwoNodeState,
    compute_sop_limits,
    run_thermal_sop_benchmark,
)
from .cell_chemistry import get_chemistry_spec, list_chemistry_specs
from .degradation_model import (
    get_degradation_spec,
    run_degradation_benchmark,
    simulate_grid_duty_aging,
)
from .ecm_2rc import (
    TwoRCBatteryModel,
    compute_eis_nyquist,
    run_chemistry_comparison_benchmark,
)
from .profile_generator import LoadProfileGenerator
from .sensitivity_analysis import run_thermal_sensitivity_sweep
from .soc_estimators import run_single_chemistry_estimation, run_soc_estimation_benchmark
from .thermal_model import ThermalBESSModel
from .visualizer import (
    plot_bess_response,
    plot_chemistry_2rc_comparison,
    plot_degradation_benchmark,
    plot_soc_estimation_benchmark,
    plot_thermal_sop_derating,
)

PACKAGE_NAME = "bess-soc-dynamics-solver"
NAMESPACE_NAME = "bess_solver"
CLI_VERSION = "0.7.0"


def _unit_soc(raw: str) -> float:
    val = float(raw)
    if not (0.0 <= val <= 1.0):
        raise argparse.ArgumentTypeError(f"SoC must lie in [0.0, 1.0], got {raw!r}")
    return val


def _positive_float(raw: str) -> float:
    val = float(raw)
    if not (math.isfinite(val) and val > 0.0):
        raise argparse.ArgumentTypeError(f"Value must be finite and > 0, got {raw!r}")
    return val


def _emit_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, allow_nan=False))


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected at least one numeric value")
    return tuple(float(p) for p in parts)


def _cmd_info(args: argparse.Namespace) -> int:
    chemistries = []
    for spec in list_chemistry_specs():
        deg = get_degradation_spec(spec)
        rdc_mohm = (spec.r0_ref_ohm + spec.r1_ref_ohm + spec.r2_ref_ohm) * 1e3
        chemistries.append(
            {
                "key": spec.key,
                "name": spec.name,
                "chemistry": spec.chemistry,
                "nominal_capacity_ah": spec.nominal_capacity_ah,
                "v_window_v": [spec.v_min_v, spec.v_max_v],
                "r_dc_25c_mohm": round(rdc_mohm, 3),
                "thermal_mass_j_per_k": round(spec.mass_kg * spec.cp_j_kgk, 3),
                "aging_z_cal": deg.z_cal,
                "aging_beta_dod": deg.beta_dod,
                "eol_soh_threshold": deg.eol_soh_threshold,
            }
        )
    payload = {
        "package": PACKAGE_NAME,
        "namespace": NAMESPACE_NAME,
        "version": CLI_VERSION,
        "chemistries": chemistries,
    }
    if args.json:
        _emit_json(payload)
    else:
        print(f"{PACKAGE_NAME} ({NAMESPACE_NAME}) v{CLI_VERSION}")
        for c in chemistries:
            v_lo, v_hi = c["v_window_v"]
            print(
                f"  - {c['key']} ({c['chemistry']}): Q={c['nominal_capacity_ah']} Ah, "
                f"V=[{v_lo:.2f}, {v_hi:.2f}] V, R_dc(25 °C)={c['r_dc_25c_mohm']:.2f} mOhm, "
                f"mc_p={c['thermal_mass_j_per_k']:.1f} J/K, beta_DoD={c['aging_beta_dod']:.2f}"
            )
    return 0


def _cmd_pulse(args: argparse.Namespace) -> int:
    t_amb_k = float(args.temp_c) + 273.15
    i_pulse = float(args.c_rate) * float(args.capacity_ah)
    battery = LiIonBatteryModel(capacity_ah=args.capacity_ah, r0=0.02, r1=0.015, c1=2000.0)
    thermal = ThermalBESSModel(t_amb_k=t_amb_k)
    i_profile = LoadProfileGenerator.pulse_discharge(
        total_time=int(round(args.total_time_s)),
        i_pulse=i_pulse,
        pulse_dur=int(round(args.pulse_dur_s)),
        rest_dur=int(round(args.rest_dur_s)),
        dt=args.dt,
    )
    time_array = np.arange(len(i_profile), dtype=float) * args.dt

    soc, v_rc = 1.0, 0.0
    t_cell_k = t_amb_k
    v_term_hist, v_rc_hist, soc_hist, ocv_hist, temp_c_hist = [], [], [], [], []
    for i_app in i_profile:
        battery.set_temperature(t_cell_k)
        soc, v_rc, v_term = battery.step(float(i_app), args.dt, soc, v_rc)
        t_cell_k = thermal.step_thermal(float(i_app), battery.r0, battery.r1, t_cell_k, args.dt)
        v_term_hist.append(v_term)
        v_rc_hist.append(v_rc)
        soc_hist.append(soc)
        ocv_hist.append(battery.get_ocv(soc))
        temp_c_hist.append(t_cell_k - 273.15)

    if args.plot:
        plot_bess_response(
            time_array,
            i_profile,
            v_term_hist,
            v_rc_hist,
            soc_hist,
            save_path=args.plot,
            ocv=ocv_hist,
            temp_c=temp_c_hist,
        )

    payload = {
        "capacity_ah": float(args.capacity_ah),
        "c_rate": float(args.c_rate),
        "ambient_temp_c": float(args.temp_c),
        "min_v_term_v": round(float(min(v_term_hist)), 6),
        "max_v_rc_v": round(float(max(v_rc_hist)), 6),
        "peak_temp_c": round(float(max(temp_c_hist)), 4),
        "final_soc": round(float(soc_hist[-1]), 6),
        "plot_path": args.plot,
    }
    if args.json:
        _emit_json(payload)
    else:
        print(
            f"1RC Pulse ({payload['c_rate']:.1f}C @ {payload['ambient_temp_c']:.1f} °C): "
            f"Min V_term={payload['min_v_term_v']:.3f} V | "
            f"Peak T_cell={payload['peak_temp_c']:.2f} °C | "
            f"Final SoC={payload['final_soc'] * 100.0:.1f}%"
        )
    return 0


def _cmd_ecm(args: argparse.Namespace) -> int:
    keys = ("nmc811", "lfp") if args.chemistry == "both" else (args.chemistry,)
    t_k = float(args.temp_c) + 273.15
    items = []
    for key in keys:
        spec = get_chemistry_spec(key)
        model = TwoRCBatteryModel(spec, t_amb_k=t_k)
        r0, r1, c1, r2, c2 = model.get_impedances(args.soc, t_k)
        eis = compute_eis_nyquist(model, soc=args.soc, t_cell_k=t_k)
        items.append(
            {
                "chemistry": spec.key,
                "soc": float(args.soc),
                "temp_c": float(args.temp_c),
                "ocv_v": round(float(model.get_ocv(args.soc, t_cell_k=t_k)), 6),
                "dudt_mv_per_k": round(float(model.get_entropic_dudt(args.soc)) * 1e3, 4),
                "r0_mohm": round(r0 * 1e3, 4),
                "r1_mohm": round(r1 * 1e3, 4),
                "r2_mohm": round(r2 * 1e3, 4),
                "r_dc_mohm": round((r0 + r1 + r2) * 1e3, 4),
                "tau1_s": round(r1 * c1, 3),
                "tau2_s": round(r2 * c2, 3),
                "eis_max_neg_imag_mohm": round(float(np.max(eis.neg_z_imag_ohm)) * 1e3, 4),
            }
        )

    if args.plot:
        comp = run_chemistry_comparison_benchmark(dt=args.dt)
        plot_chemistry_2rc_comparison(comp, save_path=args.plot)

    payload = {"results": items, "plot_path": args.plot}
    if args.json:
        _emit_json(payload)
    else:
        for row in items:
            print(
                f"2RC ECM [{row['chemistry']}] (SoC={row['soc'] * 100.0:.0f}%, T={row['temp_c']:.1f} °C): "
                f"OCV={row['ocv_v']:.3f} V | R_dc={row['r_dc_mohm']:.2f} mOhm "
                f"(tau1={row['tau1_s']:.1f}s, tau2={row['tau2_s']:.1f}s)"
            )
    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    keys = ("nmc811", "lfp") if args.chemistry == "both" else (args.chemistry,)
    runs = []
    for key in keys:
        spec = get_chemistry_spec(key)
        run = run_single_chemistry_estimation(
            spec,
            total_time_s=args.total_time_s,
            dt=args.dt,
            true_initial_soc=args.true_soc,
            est_initial_soc=args.init_soc,
            current_bias_c_rate=args.bias_c,
            seed=args.seed,
        )
        runs.append(
            {
                "chemistry": spec.key,
                "rmse_cc_pct": round(run.rmse_cc_full * 100.0, 4),
                "rmse_ekf_pct": round(run.rmse_ekf_full * 100.0, 4),
                "rmse_ukf_pct": round(run.rmse_ukf_full * 100.0, 4),
                "rmse_ekf_post_300s_pct": (
                    None if not math.isfinite(run.rmse_ekf_post_300s) else round(run.rmse_ekf_post_300s * 100.0, 4)
                ),
                "rmse_ukf_post_300s_pct": (
                    None if not math.isfinite(run.rmse_ukf_post_300s) else round(run.rmse_ukf_post_300s * 100.0, 4)
                ),
                "convergence_time_ekf_s": (
                    None if math.isinf(run.convergence_time_ekf_s) else round(run.convergence_time_ekf_s, 2)
                ),
                "convergence_time_ukf_s": (
                    None if math.isinf(run.convergence_time_ukf_s) else round(run.convergence_time_ukf_s, 2)
                ),
            }
        )

    if args.plot:
        bench = run_soc_estimation_benchmark(dt=args.dt)
        plot_soc_estimation_benchmark(bench, save_path=args.plot)

    payload = {"results": runs, "plot_path": args.plot}
    if args.json:
        _emit_json(payload)
    else:
        for r in runs:
            print(
                f"SoC Estimators [{r['chemistry']}]: "
                f"CC={r['rmse_cc_pct']:.2f}% | EKF={r['rmse_ekf_pct']:.2f}% | UKF={r['rmse_ukf_pct']:.2f}%"
            )
    return 0


def _cmd_sop(args: argparse.Namespace) -> int:
    spec = get_chemistry_spec(args.chemistry)
    t_k = float(args.temp_c) + 273.15
    horizons = _parse_float_csv(args.horizons)
    bess = CoupledTwoNodeBESS(spec, t_amb_k=t_k)
    state = CoupledTwoNodeState(soc=args.soc, t_core_k=t_k, t_surf_k=t_k)

    rows = []
    for h_s in horizons:
        res = compute_sop_limits(bess, state, horizon_s=h_s)
        rows.append(
            {
                "horizon_s": float(h_s),
                "i_dis_max_a": round(res.i_dis_max_a, 4),
                "p_dis_max_w": round(res.p_dis_max_w, 4),
                "dis_bottleneck": res.dis_bottleneck,
                "i_chg_max_a": round(res.i_chg_max_a, 4),
                "p_chg_max_w": round(res.p_chg_max_w, 4),
                "chg_bottleneck": res.chg_bottleneck,
            }
        )

    if args.plot:
        bench = run_thermal_sop_benchmark()
        plot_thermal_sop_derating(bench, save_path=args.plot)

    payload = {
        "chemistry": spec.key,
        "soc": float(args.soc),
        "temp_c": float(args.temp_c),
        "horizons": rows,
        "plot_path": args.plot,
    }
    if args.json:
        _emit_json(payload)
    else:
        for row in rows:
            print(
                f"SOP [{spec.key}] ({row['horizon_s']:.0f}s @ SoC={args.soc * 100.0:.0f}%, T={args.temp_c:.1f} °C): "
                f"P_dis={row['p_dis_max_w']:.2f} W ({row['dis_bottleneck']}) | "
                f"P_chg={row['p_chg_max_w']:.2f} W ({row['chg_bottleneck']})"
            )
    return 0


def _cmd_degrade(args: argparse.Namespace) -> int:
    proj = simulate_grid_duty_aging(
        args.chemistry,
        args.mode,
        total_years=args.years,
        step_days=args.step_days,
        blocks_per_day=args.blocks_per_day,
        t_amb_c=args.temp_c,
    )
    if args.plot:
        bench = run_degradation_benchmark()
        plot_degradation_benchmark(bench, save_path=args.plot)

    eol_yr = None if math.isinf(proj.eol_years) else round(proj.eol_years, 4)
    eol_efc = None if math.isinf(proj.eol_efc) else round(proj.eol_efc, 2)
    payload = {
        "chemistry": proj.chemistry.key,
        "mode": proj.mode,
        "years_simulated": float(args.years),
        "ambient_temp_c": float(args.temp_c),
        "total_efc": round(float(proj.efc[-1]), 2),
        "final_soh_pct": round(float(proj.soh_pct[-1]), 3),
        "q_loss_cal_pct": round(float(proj.q_loss_cal_pct[-1]), 3),
        "q_loss_cyc_pct": round(float(proj.q_loss_cyc_pct[-1]), 3),
        "final_delta_r_pct": round(float(proj.delta_r_pct[-1]), 3),
        "eol_years": eol_yr,
        "eol_efc": eol_efc,
        "plot_path": args.plot,
    }
    if args.json:
        _emit_json(payload)
    else:
        eol_txt = f"{eol_yr:.2f} yr ({eol_efc:.0f} EFC)" if eol_yr is not None else f">{args.years:.1f} yr"
        print(
            f"Degradation [{payload['chemistry']} / {payload['mode']}]: "
            f"EoL={eol_txt} | Final SoH={payload['final_soh_pct']:.2f}% | "
            f"Delta R={payload['final_delta_r_pct']:.1f}%"
        )
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    temps = tuple(int(round(v)) for v in _parse_float_csv(args.temps_c))
    crates = _parse_float_csv(args.c_rates)
    rows = run_thermal_sensitivity_sweep(
        amb_temps_c=temps,
        c_rates=crates,
        nominal_cap_ah=args.capacity_ah,
    )
    if args.json:
        _emit_json({"cases": rows})
    else:
        print(f"{'Ambient (°C)':<15}{'C-Rate':<10}{'Min V_term (V)':<18}{'Peak Temp (°C)':<16}")
        for r in rows:
            print(
                f"{r['amb_temp_c']:<15}{r['c_rate']:<10.1f}"
                f"{r['min_v_term']:<18.3f}{r['peak_temp_c']:<16.2f}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the `bess-sim` top-level CLI parser and 7 subcommand parsers."""
    parser = argparse.ArgumentParser(
        prog="bess-sim",
        description="Physics-grounded Li-ion BESS 1RC/2RC, EKF/UKF, 2-node SOP & rainflow aging CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {CLI_VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Display package version and registered cell/aging parameters.")
    p_info.add_argument("--json", action="store_true", help="Emit machine-readable RFC-8259 JSON.")
    p_info.set_defaults(func=_cmd_info)

    p_pulse = sub.add_parser("pulse", help="Run 1RC electro-thermal pulsed discharge simulation.")
    p_pulse.add_argument("--capacity-ah", type=_positive_float, default=2.5)
    p_pulse.add_argument("--c-rate", type=float, default=3.0)
    p_pulse.add_argument("--temp-c", type=float, default=25.0)
    p_pulse.add_argument("--total-time-s", type=_positive_float, default=1800.0)
    p_pulse.add_argument("--pulse-dur-s", type=float, default=300.0)
    p_pulse.add_argument("--rest-dur-s", type=float, default=300.0)
    p_pulse.add_argument("--dt", type=_positive_float, default=1.0)
    p_pulse.add_argument("--plot", type=str, default=None, help="Optional PNG path for Figure 1.")
    p_pulse.add_argument("--json", action="store_true")
    p_pulse.set_defaults(func=_cmd_pulse)

    p_ecm = sub.add_parser("ecm", help="Evaluate 2RC NMC811/LFP impedances and EIS Nyquist metrics.")
    p_ecm.add_argument("--chemistry", choices=("nmc811", "lfp", "both"), default="both")
    p_ecm.add_argument("--soc", type=_unit_soc, default=0.50)
    p_ecm.add_argument("--temp-c", type=float, default=25.0)
    p_ecm.add_argument("--dt", type=_positive_float, default=1.0)
    p_ecm.add_argument("--plot", type=str, default=None, help="Optional PNG path for Figure 2.")
    p_ecm.add_argument("--json", action="store_true")
    p_ecm.set_defaults(func=_cmd_ecm)

    p_est = sub.add_parser("estimate", help="Run closed-loop CC, EKF & UKF SoC estimation.")
    p_est.add_argument("--chemistry", choices=("nmc811", "lfp", "both"), default="both")
    p_est.add_argument("--true-soc", type=_unit_soc, default=0.88)
    p_est.add_argument("--init-soc", type=_unit_soc, default=0.65)
    p_est.add_argument("--bias-c", type=float, default=0.05)
    p_est.add_argument("--total-time-s", type=_positive_float, default=3600.0)
    p_est.add_argument("--dt", type=_positive_float, default=1.0)
    p_est.add_argument("--seed", type=int, default=42)
    p_est.add_argument("--plot", type=str, default=None, help="Optional PNG path for Figure 3.")
    p_est.add_argument("--json", action="store_true")
    p_est.set_defaults(func=_cmd_estimate)

    p_sop = sub.add_parser("sop", help="Compute 2-node thermal & receding-horizon BMS SOP limits.")
    p_sop.add_argument("--chemistry", choices=("nmc811", "lfp"), default="nmc811")
    p_sop.add_argument("--soc", type=_unit_soc, default=0.50)
    p_sop.add_argument("--temp-c", type=float, default=25.0)
    p_sop.add_argument("--horizons", type=str, default="2,10,30", help="Comma-separated horizons [s].")
    p_sop.add_argument("--plot", type=str, default=None, help="Optional PNG path for Figure 4.")
    p_sop.add_argument("--json", action="store_true")
    p_sop.set_defaults(func=_cmd_sop)

    p_deg = sub.add_parser("degrade", help="Project ASTM E1049-85 rainflow & SEI/plating lifetime aging.")
    p_deg.add_argument("--chemistry", choices=("nmc811", "lfp"), default="nmc811")
    p_deg.add_argument("--mode", choices=("deep_arbitrage", "regd_freq_reg"), default="deep_arbitrage")
    p_deg.add_argument("--years", type=_positive_float, default=12.0)
    p_deg.add_argument("--step-days", type=_positive_float, default=60.0)
    p_deg.add_argument("--blocks-per-day", type=float, default=1.05)
    p_deg.add_argument("--temp-c", type=float, default=25.0)
    p_deg.add_argument("--plot", type=str, default=None, help="Optional PNG path for Figure 5.")
    p_deg.add_argument("--json", action="store_true")
    p_deg.set_defaults(func=_cmd_degrade)

    p_swp = sub.add_parser("sweep", help="Run ambient-temperature x C-rate electro-thermal sweep.")
    p_swp.add_argument("--temps-c", type=str, default="10,25,40,50")
    p_swp.add_argument("--c-rates", type=str, default="1,2,3,4")
    p_swp.add_argument("--capacity-ah", type=_positive_float, default=2.5)
    p_swp.add_argument("--json", action="store_true")
    p_swp.set_defaults(func=_cmd_sweep)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `bess-sim` and `python -m bess_solver`."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
