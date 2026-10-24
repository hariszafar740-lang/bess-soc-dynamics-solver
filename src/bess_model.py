import math
import warnings

from src.thermal_model import EA_J_MOL, T_REF_K, arrhenius_factor, check_temperature_k

SOC_TOL = 1e-12  # Numerical tolerance on the SoC domain check


def _finite(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


class LiIonBatteryModel:
    """
    1RC Thevenin Equivalent Circuit Model (ECM) for Lithium-Ion Cells.

    Sign convention: I > 0 is discharge, I < 0 is charge.

    Equations (all SI units):
        V_term = OCV(SoC) - I*R0(T) - V_rc
        dV_rc/dt = I/C1 - V_rc/(R1(T)*C1)          (exactly integrated, see `step`)
        SoC(t+dt) = SoC(t) - I*dt/Q_nominal,  SoC clipped to [0, 1]

    Reference parameters (`r0_ref`, `r1_ref`, `c1_ref`) are the values at
    `t_ref_k` and are never modified. The active values R0, R1, C1 and the
    time constant tau1 = R1*C1 are derived from them by `set_temperature`, so
    repeated temperature changes never compound and tau1 is always consistent
    with R1. C1 is held constant with temperature (no data available).

    The OCV curve is an empirical 4th-order polynomial fitted to the nominal
    shape of a generic Li-ion cell (3.2 V at SoC=0, 4.1 V at SoC=1). It is NOT
    fitted to any specific cell chemistry and has no temperature dependence.
    """

    def __init__(self, capacity_ah=2.5, r0=0.02, r1=0.015, c1=2000.0,
                 t_ref_k=T_REF_K, ea_j_mol=EA_J_MOL):
        self.capacity_ah = _finite("Capacity (Ah)", capacity_ah)
        if self.capacity_ah <= 0.0:
            raise ValueError(f"Capacity must be > 0 Ah, got {capacity_ah!r}")
        self.q_nominal = self.capacity_ah * 3600.0  # Capacity in Coulombs (A*s)

        self.r0_ref = _finite("R0 (ohm)", r0)
        self.r1_ref = _finite("R1 (ohm)", r1)
        self.c1_ref = _finite("C1 (F)", c1)
        if self.r0_ref < 0.0:
            raise ValueError(f"R0 must be >= 0 ohm, got {r0!r}")
        if self.r1_ref <= 0.0:
            raise ValueError(f"R1 must be > 0 ohm (tau1 = R1*C1 must be positive), got {r1!r}")
        if self.c1_ref <= 0.0:
            raise ValueError(f"C1 must be > 0 F, got {c1!r}")

        self.t_ref_k = check_temperature_k(t_ref_k)
        self.ea_j_mol = _finite("Activation energy (J/mol)", ea_j_mol)
        self.reset()

    # ------------------------------------------------------------------
    # Temperature-dependent active parameters (derived, never stored as inputs)
    # ------------------------------------------------------------------
    @property
    def t_cell_k(self):
        """Temperature (K) for which the active parameters were last computed."""
        return self._t_cell_k

    @property
    def r0(self):
        """Active ohmic resistance R0(T) in ohms."""
        return self._r0

    @property
    def r1(self):
        """Active polarization resistance R1(T) in ohms."""
        return self._r1

    @property
    def c1(self):
        """Polarization capacitance C1 in farads (temperature-independent)."""
        return self._c1

    @property
    def tau1(self):
        """RC time constant R1(T)*C1 in seconds, always consistent with the active R1."""
        return self._r1 * self._c1

    def set_temperature(self, t_cell_k):
        """
        Computes active R0 and R1 at cell temperature t_cell_k (K) from the
        reference values. Calling this repeatedly (in any order) yields the same
        result for the same temperature. Raises ValueError outside the supported envelope.
        """
        t_cell_k = check_temperature_k(t_cell_k)
        factor = arrhenius_factor(t_cell_k, self.ea_j_mol, self.t_ref_k)
        self._r0 = self.r0_ref * factor
        self._r1 = self.r1_ref * factor
        self._c1 = self.c1_ref
        self._t_cell_k = t_cell_k
        return self

    def reset(self, t_cell_k=None):
        """Restores active parameters to the reference temperature (or to t_cell_k if given)."""
        self._t_cell_k = None
        self.set_temperature(self.t_ref_k if t_cell_k is None else t_cell_k)
        return self

    # ------------------------------------------------------------------
    # Model equations
    # ------------------------------------------------------------------
    def get_ocv(self, soc):
        """
        Open-circuit voltage (V) from the empirical polynomial OCV(SoC).
        Domain: SoC in [0.0, 1.0]; values outside raise ValueError.
        """
        soc = _finite("SoC", soc)
        if soc < -SOC_TOL or soc > 1.0 + SOC_TOL:
            raise ValueError(f"SoC must lie in [0, 1], got {soc!r}")
        soc_c = min(max(soc, 0.0), 1.0)
        return 3.2 + 0.9 * soc_c - 0.5 * (soc_c ** 2) + 0.8 * (soc_c ** 3) - 0.3 * (soc_c ** 4)

    def step(self, i_app, dt, soc, v_rc):
        """
        Advances the ECM by one step of length dt (s) with current i_app (A) held constant.

        RC polarization uses the exact solution for constant input over the step:
            V_rc(t+dt) = V_rc(t)*exp(-dt/tau1) + I*R1*(1 - exp(-dt/tau1))
        which is unconditionally stable (the previous explicit Euler update grew
        without bound when dt > 2*tau1).

        SoC is integrated by Coulomb counting and clipped to [0, 1]. If clipping
        occurs (cell fully charged or depleted), a RuntimeWarning is emitted so the
        lost charge is not silent.

        Returns (new_soc, new_v_rc, v_term).
        """
        i_app = _finite("Applied current (A)", i_app)
        dt = _finite("Time step dt (s)", dt)
        if dt <= 0.0:
            raise ValueError(f"Time step dt must be > 0 s, got {dt!r}")
        soc = _finite("SoC", soc)
        if soc < -SOC_TOL or soc > 1.0 + SOC_TOL:
            raise ValueError(f"Initial/current SoC must lie in [0, 1], got {soc!r}")
        v_rc = _finite("V_rc (V)", v_rc)

        # 1. Coulomb counting SoC update (discharge I > 0 reduces SoC)
        raw_soc = soc - (i_app * dt) / self.q_nominal
        new_soc = min(max(raw_soc, 0.0), 1.0)
        if raw_soc < -SOC_TOL or raw_soc > 1.0 + SOC_TOL:
            warnings.warn(
                f"SoC limit reached (raw SoC={raw_soc:.6f}); clipped to {new_soc:.1f}. "
                "Requested charge/discharge could not be fully stored or delivered.",
                RuntimeWarning,
                stacklevel=2,
            )

        # 2. RC polarization voltage, exact exponential integration
        decay = math.exp(-dt / self.tau1)
        new_v_rc = v_rc * decay + i_app * self._r1 * (1.0 - decay)

        # 3. Terminal voltage
        v_term = self.get_ocv(new_soc) - i_app * self._r0 - new_v_rc

        if not all(math.isfinite(x) for x in (new_soc, new_v_rc, v_term)):
            raise FloatingPointError("ECM step produced a non-finite state")
        return new_soc, new_v_rc, v_term
