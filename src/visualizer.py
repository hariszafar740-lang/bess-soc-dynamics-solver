import matplotlib.pyplot as plt

def plot_bess_response(time_array, i_profile, v_term, v_rc, soc, save_path='results/bess_dynamic_response.png'):
    """
    Renders 4-panel dynamic response curves for 1RC Lithium-Ion Battery Model:
    1. Applied Load Current I_app(t)
    2. Terminal Voltage V_term(t)
    3. Polarization RC Voltage Drop V_rc(t)
    4. Coulomb Counting State-of-Charge SoC(t)
    """
    fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    # 1. Applied Current Profile
    axs[0].plot(time_array, i_profile, color='#1f77b4', linewidth=1.5, label='Applied Current I(t)')
    axs[0].set_ylabel('Current (A)', fontsize=9, fontweight='bold')
    axs[0].set_title('BESS 1RC Equivalent Circuit Model Dynamic Response', fontsize=11, fontweight='bold')
    axs[0].grid(True, ls='--', alpha=0.5)
    axs[0].legend(loc='upper right')

    # 2. Terminal Voltage Response
    axs[1].plot(time_array, v_term, color='#d62728', linewidth=1.5, label='Terminal Voltage V_term(t)')
    axs[1].set_ylabel('Voltage (V)', fontsize=9, fontweight='bold')
    axs[1].grid(True, ls='--', alpha=0.5)
    axs[1].legend(loc='upper right')

    # 3. Polarization RC Overpotential
    axs[2].plot(time_array, v_rc, color='#ff7f0e', linewidth=1.5, label='Polarization Voltage V_rc(t)')
    axs[2].set_ylabel('Overpotential (V)', fontsize=9, fontweight='bold')
    axs[2].grid(True, ls='--', alpha=0.5)
    axs[2].legend(loc='upper right')

    # 4. State of Charge Dynamics
    axs[3].plot(time_array, [s * 100.0 for s in soc], color='#2ca02c', linewidth=1.8, label='State of Charge SoC(t)')
    axs[3].set_xlabel('Time (Seconds)', fontsize=10, fontweight='bold')
    axs[3].set_ylabel('SoC (%)', fontsize=9, fontweight='bold')
    axs[3].grid(True, ls='--', alpha=0.5)
    axs[3].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
