from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_bess_response(
    time_array,
    i_profile,
    v_term,
    v_rc,
    soc,
    save_path="results/bess_dynamic_response.png",
    ocv=None,
    temp_c=None,
):
    """
    Renders 4-panel dynamic response curves for 1RC Lithium-Ion Battery Model:
    1. Applied Load Current I_app(t)
    2. Terminal Voltage V_term(t) (with optional open-circuit voltage V_ocv(SoC) reference)
    3. Polarization RC Voltage Drop V_rc(t) (with optional coupled cell temperature T_cell(t))
    4. Coulomb-Counted State-of-Charge SoC(t)
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    t_arr = np.asarray(time_array, dtype=float)
    i_arr = np.asarray(i_profile, dtype=float)
    vt_arr = np.asarray(v_term, dtype=float)
    vrc_arr = np.asarray(v_rc, dtype=float)
    soc_pct = np.asarray(soc, dtype=float) * 100.0

    fig, axs = plt.subplots(4, 1, figsize=(11.8, 9.4), sharex=True)

    # 1. Applied Current Profile
    axs[0].plot(t_arr, i_arr, color="#1f77b4", linewidth=1.9, label="Applied Current I(t)")
    axs[0].fill_between(t_arr, 0.0, i_arr, color="#1f77b4", alpha=0.12)
    axs[0].set_ylabel("Current (A)", fontsize=9.5, fontweight="bold")
    axs[0].set_title(
        "Coupled 1RC Electro-Thermal BESS Dynamic Response (3C Pulsed Discharge, T_amb = 25 °C)",
        fontsize=11.2,
        fontweight="bold",
    )
    axs[0].grid(True, ls="--", alpha=0.45)
    axs[0].legend(loc="upper right", fontsize=8.8, framealpha=0.94)

    # 2. Terminal Voltage & Open-Circuit Voltage Envelope
    if ocv is not None:
        ocv_arr = np.asarray(ocv, dtype=float)
        axs[1].plot(
            t_arr,
            ocv_arr,
            color="#475569",
            linestyle="--",
            linewidth=1.6,
            label="Open-Circuit Voltage V_ocv(SoC)",
        )
    axs[1].plot(t_arr, vt_arr, color="#d62728", linewidth=1.9, label="Terminal Voltage V_term(t)")
    axs[1].set_ylabel("Voltage (V)", fontsize=9.5, fontweight="bold")
    axs[1].grid(True, ls="--", alpha=0.45)
    axs[1].legend(loc="upper right", fontsize=8.8, framealpha=0.94)

    # 3. Polarization RC Overpotential (+ optional Cell Temperature twin axis)
    (l_vrc,) = axs[2].plot(
        t_arr,
        vrc_arr * 1e3,
        color="#ea580c",
        linewidth=1.9,
        label="Polarization Overpotential V_rc(t) (mV)",
    )
    vrc_max_mv = max(1.0, float(np.max(np.abs(vrc_arr)) * 1e3))
    axs[2].set_ylim(min(-3.0, float(np.min(vrc_arr) * 1e3) * 1.15), vrc_max_mv * 1.34)
    axs[2].set_ylabel("V_rc (mV)", fontsize=9.5, fontweight="bold", color="#c2410c")
    axs[2].grid(True, ls="--", alpha=0.45)
    if temp_c is not None:
        tc_arr = np.asarray(temp_c, dtype=float)
        ax2_t = axs[2].twinx()
        (l_tc,) = ax2_t.plot(
            t_arr,
            tc_arr,
            color="#7e22ce",
            linestyle="-.",
            linewidth=1.8,
            label=f"Cell Temperature T_cell(t) (Peak {tc_arr.max():.2f} °C)",
        )
        ax2_t.set_ylabel("T_cell (°C)", fontsize=9.5, fontweight="bold", color="#6b21a8")
        t_span = max(1.0, float(tc_arr.max() - tc_arr.min()))
        ax2_t.set_ylim(float(tc_arr.min()) - 0.12 * t_span, float(tc_arr.max()) + 0.45 * t_span)
        axs[2].legend(
            handles=[l_vrc, l_tc], loc="upper right", fontsize=8.5, framealpha=0.94
        )
    else:
        axs[2].legend(loc="upper right", fontsize=8.8, framealpha=0.94)

    # 4. State of Charge Dynamics
    axs[3].plot(t_arr, soc_pct, color="#16a34a", linewidth=2.0, label="Coulomb-Counted SoC(t)")
    axs[3].set_ylim(max(0.0, float(soc_pct.min()) - 6.0), min(105.0, float(soc_pct.max()) + 6.0))
    axs[3].set_xlabel("Simulation Time t (Seconds)", fontsize=10, fontweight="bold")
    axs[3].set_ylabel("SoC (%)", fontsize=9.5, fontweight="bold")
    axs[3].grid(True, ls="--", alpha=0.45)
    axs[3].legend(loc="upper right", fontsize=8.8, framealpha=0.94)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_chemistry_2rc_comparison(
    comp,
    save_path: str = "results/chemistry_2rc_comparison.png",
) -> None:
    """Render 4-panel multi-chemistry 2RC, hysteresis, Bernardi heat & EIS figure."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(13.8, 9.6))
    ax_ocv, ax_hppc = axs[0, 0], axs[0, 1]
    ax_heat, ax_eis = axs[1, 0], axs[1, 1]

    soc_pct = comp.soc_curve * 100.0

    # Panel (a): OCV(SoC) + LFP Hysteresis Band + NMC811 Entropic dU/dT
    (l_nmc,) = ax_ocv.plot(
        soc_pct,
        comp.nmc_ocv_v,
        color="#1d4ed8",
        lw=2.1,
        label="NMC811 OCV (LG M50, O'Regan 2022)",
    )
    (l_lfp,) = ax_ocv.plot(
        soc_pct,
        comp.lfp_ocv_v,
        color="#059669",
        lw=2.1,
        label="LFP OCV (A123 26650, Prada 2013)",
    )
    hyst_fill = ax_ocv.fill_between(
        soc_pct,
        comp.lfp_hyst_lower_v * 1.0,
        comp.lfp_hyst_upper_v * 1.0,
        color="#10b981",
        alpha=0.22,
        label="LFP ±18 mV Hysteresis Envelope",
    )
    ax_ocv.set_title("(a) Equilibrium OCV & Entropic Coefficient", fontsize=10.8, fontweight="bold")
    ax_ocv.set_xlabel("State of Charge SoC (%)", fontsize=9.5, fontweight="bold")
    ax_ocv.set_ylabel("Equilibrium Potential U_ocv (V)", fontsize=9.5, fontweight="bold")
    ax_ocv.set_xlim(0.0, 100.0)
    ax_ocv.set_ylim(1.90, 4.55)
    ax_ocv.grid(True, ls="--", alpha=0.42)

    ax_dudt = ax_ocv.twinx()
    ax_dudt.axhline(0.0, color="#94a3b8", lw=0.9, ls=":")
    (l_dudt,) = ax_dudt.plot(
        soc_pct,
        comp.nmc_dudt_mv_k,
        color="#dc2626",
        lw=1.7,
        ls="--",
        label="NMC811 dU/dT (mV/K)",
    )
    ax_dudt.set_ylabel("dU_ocv / dT (mV/K)", fontsize=9.5, fontweight="bold", color="#b91c1c")
    ax_dudt.set_ylim(-0.52, 0.30)
    ax_ocv.legend(
        handles=[l_nmc, l_lfp, hyst_fill, l_dudt],
        loc="lower right",
        fontsize=8.0,
        framealpha=0.94,
    )

    # Panel (b): Bidirectional HPPC Overpotential Decomposition (V_term - U_ocv)
    t_min = comp.nmc_hppc.time_s / 60.0
    dv_nmc_mv = (comp.nmc_hppc.v_term_v - comp.nmc_hppc.u_ocv_v) * 1e3
    dv_lfp_mv = (comp.lfp_hppc.v_term_v - comp.lfp_hppc.u_ocv_v) * 1e3
    ax_hppc.axhline(0.0, color="#64748b", lw=0.9, ls=":")
    ax_hppc.plot(t_min, dv_nmc_mv, color="#1d4ed8", lw=2.0, label="NMC811 ΔV = V_term - U_ocv")
    ax_hppc.plot(t_min, dv_lfp_mv, color="#059669", lw=1.9, label="LFP ΔV (with Hysteresis Memory)")
    ax_hppc.plot(
        t_min,
        -comp.nmc_hppc.v_rc1_v * 1e3,
        color="#ea580c",
        lw=1.5,
        ls="--",
        label="NMC811 -V_rc1 (Fast τ1 ≈ 18 s)",
    )
    ax_hppc.plot(
        t_min,
        -comp.nmc_hppc.v_rc2_v * 1e3,
        color="#7c3aed",
        lw=1.6,
        ls="-.",
        label="NMC811 -V_rc2 (Diffusion τ2 ≈ 240 s)",
    )
    ax_hppc.set_title(
        "(b) Bidirectional HPPC Pulse-Relaxation Overpotential (80% SoC)",
        fontsize=10.8,
        fontweight="bold",
    )
    ax_hppc.set_xlabel("Time (minutes)", fontsize=9.5, fontweight="bold")
    ax_hppc.set_ylabel("Overpotential V_term - U_ocv (mV)", fontsize=9.5, fontweight="bold")
    ax_hppc.set_xlim(0.0, 30.0)
    ax_hppc.set_ylim(-265.0, 215.0)
    ax_hppc.grid(True, ls="--", alpha=0.42)
    ax_hppc.legend(loc="lower right", fontsize=8.0, framealpha=0.94)

    # Panel (c): Bernardi Reversible vs Irreversible Heat during 1C NMC811 Discharge
    soc_dis_pct = comp.nmc_discharge_1c.soc * 100.0
    q_irr = comp.nmc_discharge_1c.q_irrev_w
    q_rev = comp.nmc_discharge_1c.q_rev_w
    q_tot = comp.nmc_discharge_1c.q_total_w
    ax_heat.axhline(0.0, color="#64748b", lw=0.9, ls=":")
    ax_heat.plot(soc_dis_pct, q_irr, color="#ea580c", lw=1.9, label="Irreversible Q_irrev (Ohmic + 2RC)")
    ax_heat.plot(soc_dis_pct, q_rev, color="#0d9488", lw=1.9, ls="--", label="Reversible Entropic Q_rev = -I·T·dU/dT")
    ax_heat.plot(soc_dis_pct, q_tot, color="#7c3aed", lw=2.1, label="Net Bernardi Heat Q_total")
    ax_heat.fill_between(
        soc_dis_pct,
        0.0,
        q_rev,
        where=(q_rev < 0.0),
        color="#14b8a6",
        alpha=0.18,
        label="Endothermic Entropic Cooling (Q_rev < 0)",
    )
    ax_heat.fill_between(
        soc_dis_pct,
        0.0,
        q_rev,
        where=(q_rev >= 0.0),
        color="#f97316",
        alpha=0.18,
        label="Exothermic Entropic Heating (Q_rev > 0)",
    )
    ax_heat.set_xlim(98.0, 4.0)  # Left-to-right follows 1C discharge progression
    ax_heat.set_ylim(-0.35, 2.25)
    ax_heat.set_title(
        "(c) Bernardi Heat Decomposition (NMC811 1C Discharge, 98% → 4% SoC)",
        fontsize=10.8,
        fontweight="bold",
    )
    ax_heat.set_xlabel("State of Charge during 1C Discharge (%)", fontsize=9.5, fontweight="bold")
    ax_heat.set_ylabel("Heat Generation Rate (W)", fontsize=9.5, fontweight="bold")
    ax_heat.grid(True, ls="--", alpha=0.42)
    ax_heat.legend(loc="upper left", fontsize=7.9, framealpha=0.94)

    # Panel (d): Multi-Temperature EIS Nyquist Spectrum (NMC811 @ 50% SoC)
    palette = {10: "#0284c7", 25: "#16a34a", 45: "#dc2626"}
    for temp_c in (10, 25, 45):
        eis = comp.eis_nmc_by_temp_c[temp_c]
        z_re_mohm = eis.z_real_ohm * 1e3
        z_im_mohm = eis.neg_z_imag_ohm * 1e3
        ax_eis.plot(
            z_re_mohm,
            z_im_mohm,
            color=palette[temp_c],
            lw=2.0,
            label=(
                f"T = {temp_c} °C (R_dc = {(eis.r0_ohm + eis.r1_ohm + eis.r2_ohm) * 1e3:.1f} mΩ)"
            ),
        )
        # Mark characteristic RC1 and RC2 semicircle peak frequencies
        for f_pk in (eis.f_peak_rc1_hz, eis.f_peak_rc2_hz):
            idx_pk = int(np.argmin(np.abs(eis.freq_hz - f_pk)))
            ax_eis.scatter(
                [z_re_mohm[idx_pk]],
                [z_im_mohm[idx_pk]],
                color=palette[temp_c],
                s=22,
                zorder=4,
            )

    eis_25 = comp.eis_nmc_by_temp_c[25]
    idx1 = int(np.argmin(np.abs(eis.freq_hz - eis_25.f_peak_rc1_hz)))
    idx2 = int(np.argmin(np.abs(eis.freq_hz - eis_25.f_peak_rc2_hz)))
    ax_eis.annotate(
        f"f_rc1 = {eis_25.f_peak_rc1_hz * 1e3:.1f} mHz",
        xy=(eis_25.z_real_ohm[idx1] * 1e3, eis_25.neg_z_imag_ohm[idx1] * 1e3),
        xytext=(11.2, 13.8),
        fontsize=8.2,
        color="#15803d",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#15803d", "lw": 1.0},
    )
    ax_eis.annotate(
        f"f_rc2 = {eis_25.f_peak_rc2_hz * 1e3:.2f} mHz",
        xy=(eis_25.z_real_ohm[idx2] * 1e3, eis_25.neg_z_imag_ohm[idx2] * 1e3),
        xytext=(24.0, 20.5),
        fontsize=8.2,
        color="#15803d",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#15803d", "lw": 1.0},
    )
    ax_eis.set_title(
        "(d) 2RC EIS Nyquist Impedance Spectrum (NMC811 @ 50% SoC, 0.01 mHz–1 kHz)",
        fontsize=10.8,
        fontweight="bold",
    )
    ax_eis.set_xlabel("Re{Z(jω)} (mΩ)", fontsize=9.5, fontweight="bold")
    ax_eis.set_ylabel("-Im{Z(jω)} (mΩ)", fontsize=9.5, fontweight="bold")
    ax_eis.set_xlim(10.0, 92.0)
    ax_eis.set_ylim(0.0, 26.0)
    ax_eis.grid(True, ls="--", alpha=0.42)
    ax_eis.legend(loc="upper right", fontsize=8.1, framealpha=0.94)

    fig.suptitle(
        "Chemistry-Parameterized 2RC Dual-Polarization ECM: NMC811 vs. LFP, Hysteresis, Bernardi Heat & EIS",
        fontsize=12.2,
        fontweight="bold",
        y=0.992,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.972))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_soc_estimation_benchmark(
    bench,
    save_path: str = "results/soc_ekf_ukf_benchmark.png",
) -> None:
    """Render 4-panel closed-loop EKF & UKF vs Coulomb Counting SoC benchmark figure."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(13.8, 9.6))
    ax_nmc, ax_lfp = axs[0, 0], axs[0, 1]
    ax_err, ax_obs = axs[1, 0], axs[1, 1]

    t_min = bench.nmc.time_s / 60.0

    # Panel (a): NMC811 Closed-Loop SoC Tracking (-23% Initial Offset + +0.05C Bias)
    r_nmc = bench.nmc
    ax_nmc.plot(t_min, r_nmc.soc_true * 100.0, color="#0f172a", lw=2.2, label="True Plant SoC(t)")
    ax_nmc.plot(
        t_min,
        r_nmc.soc_cc * 100.0,
        color="#64748b",
        lw=1.8,
        ls="--",
        label=f"Coulomb Counting (RMSE = {r_nmc.rmse_cc_full * 100.0:.1f}%)",
    )
    ax_nmc.plot(
        t_min,
        r_nmc.soc_ekf * 100.0,
        color="#1d4ed8",
        lw=1.9,
        label=f"EKF (RMSE = {r_nmc.rmse_ekf_full * 100.0:.2f}%)",
    )
    ax_nmc.plot(
        t_min,
        r_nmc.soc_ukf * 100.0,
        color="#dc2626",
        lw=1.7,
        ls="-.",
        label=f"UKF (RMSE = {r_nmc.rmse_ukf_full * 100.0:.2f}%)",
    )
    ax_nmc.fill_between(
        t_min,
        (r_nmc.soc_ekf - 2.0 * r_nmc.soc_ekf_std) * 100.0,
        (r_nmc.soc_ekf + 2.0 * r_nmc.soc_ekf_std) * 100.0,
        color="#3b82f6",
        alpha=0.18,
        label="EKF ±2σ Confidence Band",
    )
    ax_nmc.set_title(
        "(a) NMC811 SoC Estimation (-23% Init Error, +0.05C Bias, 8 mV Noise)",
        fontsize=10.6,
        fontweight="bold",
    )
    ax_nmc.set_xlabel("Time (minutes)", fontsize=9.5, fontweight="bold")
    ax_nmc.set_ylabel("State of Charge SoC (%)", fontsize=9.5, fontweight="bold")
    ax_nmc.set_xlim(0.0, 60.0)
    ax_nmc.set_ylim(0.0, 98.0)
    ax_nmc.grid(True, ls="--", alpha=0.42)
    ax_nmc.legend(loc="lower left", fontsize=8.0, framealpha=0.94)

    # Panel (b): LFP Flat-Plateau SoC Tracking & Nonlinear Knee Convergence
    r_lfp = bench.lfp
    ax_lfp.plot(t_min, r_lfp.soc_true * 100.0, color="#0f172a", lw=2.2, label="True Plant SoC(t)")
    ax_lfp.plot(
        t_min,
        r_lfp.soc_cc * 100.0,
        color="#64748b",
        lw=1.8,
        ls="--",
        label=f"Coulomb Counting (RMSE = {r_lfp.rmse_cc_full * 100.0:.1f}%)",
    )
    ax_lfp.plot(
        t_min,
        r_lfp.soc_ekf * 100.0,
        color="#059669",
        lw=1.9,
        label=f"EKF (RMSE = {r_lfp.rmse_ekf_full * 100.0:.2f}%)",
    )
    ax_lfp.plot(
        t_min,
        r_lfp.soc_ukf * 100.0,
        color="#7c3aed",
        lw=1.8,
        ls="-.",
        label=f"UKF (RMSE = {r_lfp.rmse_ukf_full * 100.0:.2f}%)",
    )
    ax_lfp.fill_between(
        t_min,
        (r_lfp.soc_ekf - 2.0 * r_lfp.soc_ekf_std) * 100.0,
        (r_lfp.soc_ekf + 2.0 * r_lfp.soc_ekf_std) * 100.0,
        color="#10b981",
        alpha=0.18,
        label="EKF ±2σ Confidence Band",
    )
    ax_lfp.set_title(
        "(b) LFP Two-Phase Plateau SoC Estimation (Flat OCV Observability Test)",
        fontsize=10.6,
        fontweight="bold",
    )
    ax_lfp.set_xlabel("Time (minutes)", fontsize=9.5, fontweight="bold")
    ax_lfp.set_ylabel("State of Charge SoC (%)", fontsize=9.5, fontweight="bold")
    ax_lfp.set_xlim(0.0, 60.0)
    ax_lfp.set_ylim(0.0, 98.0)
    ax_lfp.grid(True, ls="--", alpha=0.42)
    ax_lfp.legend(loc="lower left", fontsize=8.0, framealpha=0.94)

    # Panel (c): Absolute SoC Estimation Error Trajectories |SoC_hat - SoC_true| (%)
    ax_err.plot(
        t_min,
        np.abs(r_nmc.soc_cc - r_nmc.soc_true) * 100.0,
        color="#64748b",
        lw=1.8,
        ls="--",
        label="Open-Loop Coulomb Counting Drift",
    )
    ax_err.plot(
        t_min,
        np.abs(r_nmc.soc_ekf - r_nmc.soc_true) * 100.0,
        color="#1d4ed8",
        lw=1.9,
        label="NMC811 EKF |Error|",
    )
    ax_err.plot(
        t_min,
        np.abs(r_nmc.soc_ukf - r_nmc.soc_true) * 100.0,
        color="#dc2626",
        lw=1.6,
        ls="-.",
        label="NMC811 UKF |Error|",
    )
    ax_err.plot(
        t_min,
        np.abs(r_lfp.soc_ekf - r_lfp.soc_true) * 100.0,
        color="#059669",
        lw=1.8,
        label="LFP EKF |Error| (Plateau → Knee Lock)",
    )
    ax_err.plot(
        t_min,
        np.abs(r_lfp.soc_ukf - r_lfp.soc_true) * 100.0,
        color="#7c3aed",
        lw=1.7,
        ls=":",
        label="LFP UKF |Error| (Sigma-Point Knee Capture)",
    )
    ax_err.axhline(2.0, color="#94a3b8", lw=1.0, ls=":", label="2.0% SoC Accuracy Band")
    ax_err.set_title(
        "(c) Absolute SoC Estimation Error |SoC_hat - SoC_true| (%)",
        fontsize=10.6,
        fontweight="bold",
    )
    ax_err.set_xlabel("Time (minutes)", fontsize=9.5, fontweight="bold")
    ax_err.set_ylabel("Absolute SoC Error (%)", fontsize=9.5, fontweight="bold")
    ax_err.set_xlim(0.0, 60.0)
    ax_err.set_ylim(0.0, 38.5)
    ax_err.grid(True, ls="--", alpha=0.42)
    ax_err.legend(loc="upper left", ncol=2, fontsize=7.7, framealpha=0.95)

    # Panel (d): Electrochemical Voltage Observability Slope dU_ocv/dSoC (log scale)
    soc_obs_pct = bench.soc_grid_obs * 100.0
    ax_obs.semilogy(
        soc_obs_pct,
        np.maximum(1e-3, bench.nmc_docv_dsoc_v * 1e3),
        color="#1d4ed8",
        lw=2.1,
        label="NMC811 dU_ocv/dSoC (mV / unit SoC)",
    )
    ax_obs.semilogy(
        soc_obs_pct,
        np.maximum(1e-3, bench.lfp_docv_dsoc_v * 1e3),
        color="#059669",
        lw=2.1,
        label="LFP dU_ocv/dSoC (mV / unit SoC)",
    )
    ax_obs.axvspan(
        40.0,
        70.0,
        color="#10b981",
        alpha=0.14,
        label="LFP Two-Phase Plateau (16–40 mV/SoC Weak Observability)",
    )
    ax_obs.axhline(
        8.0,
        color="#dc2626",
        lw=1.2,
        ls="--",
        label="1σ Voltage Sensor Noise Floor (8 mV)",
    )
    ax_obs.set_title(
        "(d) Local OCV Observability Jacobian dU_ocv/dSoC: NMC811 vs. LFP",
        fontsize=10.6,
        fontweight="bold",
    )
    ax_obs.set_xlabel("State of Charge SoC (%)", fontsize=9.5, fontweight="bold")
    ax_obs.set_ylabel("OCV Slope dU_ocv / dSoC (mV / unit SoC)", fontsize=9.5, fontweight="bold")
    ax_obs.set_xlim(2.0, 98.0)
    ax_obs.set_ylim(5.0, 1.2e5)
    ax_obs.grid(True, which="both", ls="--", alpha=0.38)
    ax_obs.legend(loc="upper right", fontsize=7.9, framealpha=0.95)

    fig.suptitle(
        "Closed-Loop Nonlinear BESS SoC Estimation: EKF & UKF vs. Drifting Coulomb Counting",
        fontsize=12.2,
        fontweight="bold",
        y=0.992,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.972))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
