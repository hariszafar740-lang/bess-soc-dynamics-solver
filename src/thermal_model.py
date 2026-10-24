import math

# ---------------------------------------------------------------------------
# Shared constants (all temperatures in Kelvin unless noted)
# ---------------------------------------------------------------------------
T_REF_K = 298.15           # Reference temperature (25 degC) at which R_ref, R1_ref, C1_ref are specified
EA_J_MOL = 20000.0         # Assumed Arrhenius activation energy for ohmic/polarization resistance (J/mol)
R_GAS = 8.314              # Universal gas constant (J/(mol*K))

# Documented validity envelope of the Arrhenius correction and of the lumped thermal
# model (-20 degC to +60 degC). Values outside this envelope raise ValueError.
T_MIN_K = 253.15
T_MAX_K = 333.15


def _check_finite_positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return value


def check_temperature_k(t_cell_k):
    """Validates a cell temperature (K) against the documented envelope and returns it as float."""
    t_cell_k = float(t_cell_k)
    if not math.isfinite(t_cell_k):
        raise ValueError(f"Cell temperature must be finite, got {t_cell_k!r} K")
    if not (T_MIN_K <= t_cell_k <= T_MAX_K):
        raise ValueError(
            f"Cell temperature {t_cell_k:.2f} K is outside the supported envelope "
            f"[{T_MIN_K:.2f}, {T_MAX_K:.2f}] K (-20 to +60 degC)"
        )
    return t_cell_k


def arrhenius_factor(t_cell_k, ea_j_mol=EA_J_MOL, t_ref_k=T_REF_K):
    """
    Dimensionless Arrhenius scaling factor:

        f(T) = exp( Ea / R * (1/T - 1/T_ref) )

    f(T_ref) == 1. f > 1 for T < T_ref (cold cells have higher resistance).
    This is a pure function of its arguments: it never mutates any stored state.
    """
    t_cell_k = _check_finite_positive("Cell temperature (K)", t_cell_k)
    t_ref_k = _check_finite_positive("Reference temperature (K)", t_ref_k)
    exponent = (ea_j_mol / R_GAS) * ((1.0 / t_cell_k) - (1.0 / t_ref_k))
    factor = math.exp(exponent)
    if not math.isfinite(factor):
        raise ValueError(f"Arrhenius factor is not finite at T={t_cell_k} K")
    return factor


class ThermalBESSModel:
    """
    Lumped-parameter electro-thermal model for Li-ion battery cells.

    Cell energy balance (lumped, uniform temperature):

        m * Cp * dT/dt = I^2 * (R0(T) + R1(T)) - h_A * (T - T_amb)

    Joule heat uses the steady-state DC dissipation I^2*(R0+R1). Because the RC
    voltage satisfies |V_rc| <= |I|*R1 for constant-sign current, the transient
    dissipation V_rc^2/R1 is never larger than I^2*R1, so this term is a
    conservative (upper-bound) estimate of polarization heat. Entropic heat is not
    modelled (no entropy data is available in this repository).

    Temperature dependence of R0 and R1 uses Arrhenius scaling (see
    `arrhenius_factor`). The reference resistances must be supplied by the caller;
    this class never stores scaled resistances.
    """

    def __init__(self, m_kg=0.045, cp_j_kgk=900.0, h_a_w_k=0.25, t_amb_k=298.15, ea_j_mol=EA_J_MOL):
        self.m = _check_finite_positive("Cell mass m (kg)", m_kg)
        self.cp = _check_finite_positive("Specific heat Cp (J/(kg*K))", cp_j_kgk)
        h_a = float(h_a_w_k)
        if not math.isfinite(h_a) or h_a < 0.0:
            raise ValueError(f"Convective coefficient h_A (W/K) must be finite and >= 0, got {h_a!r}")
        self.h_a = h_a                                  # 0 means adiabatic cell
        self.t_amb = check_temperature_k(t_amb_k)       # Ambient temperature (K)
        self.ea = float(ea_j_mol)                       # Activation energy (J/mol)
        self.r_gas = R_GAS                              # Universal gas constant (J/(mol*K))

    @property
    def thermal_time_constant_s(self):
        """Thermal time constant m*Cp/h_A (s); infinite for an adiabatic cell."""
        return math.inf if self.h_a == 0.0 else self.m * self.cp / self.h_a

    def arrhenius_resistance(self, r_ref, t_cell_k, t_ref_k=T_REF_K):
        """
        Returns r_ref * f(T_cell) with f from `arrhenius_factor`.
        `r_ref` MUST be the reference-temperature value; passing an already
        temperature-scaled value would compound the correction.
        """
        r_ref = float(r_ref)
        if not math.isfinite(r_ref) or r_ref < 0.0:
            raise ValueError(f"Reference resistance must be finite and >= 0, got {r_ref!r}")
        return r_ref * arrhenius_factor(t_cell_k, self.ea, t_ref_k)

    def step_thermal(self, i_app, r0_act, r1_act, t_cell_k, dt=1.0):
        """
        Advances the cell temperature by one time step dt (s).

        With the Joule source held constant over the step (piecewise-constant
        input, evaluated at the start-of-step temperature), the linear ODE
            dT/dt = (Q - h_A (T - T_amb)) / (m Cp)
        has the exact solution
            T(t+dt) = T_eq + (T(t) - T_eq) * exp(-h_A dt / (m Cp)),  T_eq = T_amb + Q / h_A.
        This is unconditionally stable for any dt > 0 (the previous explicit Euler
        update oscillated/diverged when dt > m*Cp/h_A).
        """
        i_app = float(i_app)
        r0_act = float(r0_act)
        r1_act = float(r1_act)
        t_cell_k = check_temperature_k(t_cell_k)
        dt = _check_finite_positive("Time step dt (s)", dt)
        for name, val in (("Applied current", i_app), ("R0", r0_act), ("R1", r1_act)):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val!r}")
        if r0_act < 0.0 or r1_act < 0.0:
            raise ValueError("Resistances must be >= 0")

        q_gen = (i_app ** 2) * (r0_act + r1_act)     # W
        c_th = self.m * self.cp                       # J/K

        if self.h_a == 0.0:
            t_next = t_cell_k + q_gen * dt / c_th
        else:
            t_eq = self.t_amb + q_gen / self.h_a
            t_next = t_eq + (t_cell_k - t_eq) * math.exp(-self.h_a * dt / c_th)

        if not math.isfinite(t_next):
            raise FloatingPointError(f"Thermal step produced non-finite temperature from T={t_cell_k} K")
        return float(t_next)
