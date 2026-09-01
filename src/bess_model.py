import numpy as np

class LiIonBatteryModel:
    """
    1RC Thevenin Equivalent Circuit Model (ECM) for Lithium-Ion Cells.
    Simulates Open-Circuit Voltage (OCV), Ohmic drop, transient RC polarization,
    and Coulomb Counting State-of-Charge (SoC).
    """
    def __init__(self, capacity_ah=2.5, r0=0.02, r1=0.015, c1=2000.0):
        self.q_nominal = capacity_ah * 3600.0  # Capacity converted to Coulombs (A*s)
        self.r0 = r0                            # Internal Ohmic resistance (Ohms)
        self.r1 = r1                            # Polarization resistance (Ohms)
        self.c1 = c1                            # Polarization capacitance (Farads)
        self.tau1 = r1 * c1                     # Time constant (seconds)

    def get_ocv(self, soc):
        """
        Computes cell Open-Circuit Voltage (OCV) using empirical polynomial function.
        SoC range: [0.0, 1.0]
        """
        soc_c = np.clip(soc, 0.0, 1.0)
        return 3.2 + 0.9 * soc_c - 0.5 * (soc_c**2) + 0.8 * (soc_c**3) - 0.3 * (soc_c**4)

    def step(self, i_app, dt, soc, v_rc):
        """
        Discrete-time state integration step using Euler method.
        Positive current (I > 0) = Discharge | Negative current (I < 0) = Charge
        """
        # 1. Coulomb Counting SoC Update
        d_soc = -(i_app * dt) / self.q_nominal
        new_soc = np.clip(soc + d_soc, 0.0, 1.0)

        # 2. RC Polarization Voltage Dynamics Update
        dv_rc = (i_app / self.c1 - v_rc / self.tau1) * dt
        new_v_rc = v_rc + dv_rc

        # 3. Terminal Voltage Calculation
        v_ocv = self.get_ocv(new_soc)
        v_term = v_ocv - (i_app * self.r0) - new_v_rc

        return new_soc, new_v_rc, v_term
