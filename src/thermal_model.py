import numpy as np

class ThermalBESSModel:
    """
    Lumped-parameter electro-thermal model for Li-ion battery cells.
    Calculates transient cell temperature evolution T_cell(t) based on
    Joule heating (I^2 * R) and convective heat dissipation to ambient,
    with Arrhenius temperature-dependent internal resistance scaling.
    """
    def __init__(self, m_kg=0.045, cp_j_kgk=900.0, h_a_w_k=0.25, t_amb_k=298.15, ea_j_mol=20000.0):
        self.m = m_kg           # Cell mass (kg)
        self.cp = cp_j_kgk       # Specific heat capacity (J/(kg*K))
        self.h_a = h_a_w_k       # Convective heat transfer coefficient x Area (W/K)
        self.t_amb = t_amb_k     # Ambient temperature (K)
        self.ea = ea_j_mol       # Activation energy (J/mol)
        self.r_gas = 8.314       # Universal gas constant (J/(mol*K))

    def arrhenius_resistance(self, r_ref, t_cell_k, t_ref_k=298.15):
        """Scales internal resistance based on cell temperature using Arrhenius kinetics."""
        return r_ref * np.exp((self.ea / self.r_gas) * ((1.0 / t_cell_k) - (1.0 / t_ref_k)))

    def step_thermal(self, i_app, r0_act, r1_act, t_cell_k, dt=1.0):
        """
        Updates cell temperature using 1st-order forward Euler heat balance:
        dT_cell/dt = (Q_gen - Q_conv) / (m * Cp)
        """
        q_gen = (i_app ** 2) * (r0_act + r1_act)
        q_conv = self.h_a * (t_cell_k - self.t_amb)
        dt_dt = (q_gen - q_conv) / (self.m * self.cp)
        return t_cell_k + dt_dt * dt
