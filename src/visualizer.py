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
