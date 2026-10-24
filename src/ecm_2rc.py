"""Chemistry-parameterized 2RC dual-polarization ECM with hysteresis & Bernardi heat.

Governing continuous-time electro-thermal system (discharge-positive I > 0):

    dSoC/dt   = -I / (3600 * Q_nom)
    dV_rc1/dt = -V_rc1 / (R_1 * C_1) + I / C_1
    dV_rc2/dt = -V_rc2 / (R_2 * C_2) + I / C_2
    dh/dt     = -(|gamma * I| / (3600 * Q_nom)) * (h + sgn(I)),   I != 0 (0 at rest)
    V_term    = U_ocv(SoC, T_cell) + M_h * h - I * R_0 - V_rc1 - V_rc2

with temperature-corrected equilibrium open-circuit potential:

    U_ocv(SoC, T_cell) = U_ocv,ref(SoC) + (T_cell - T_ref) * (dU_ocv/dT)(SoC)

and coupled First-Law Bernardi electro-thermal energy balance (Bernardi, Pawlikowski
& Newman, J. Electrochem. Soc. 132(1), 5-12, 1985):

    m * c_p * dT_cell/dt = Q_irrev + Q_rev - h_A * (T_cell - T_amb)
    Q_rev                = -I * T_cell * (dU_ocv/dT)(SoC)

Over each time step [t, t + dt] with piecewise-constant current I, V_rc1, V_rc2, h,
and the RC resistor Joule dissipation integral int_0^dt (V_rc,k(s)^2 / R_k) ds are
integrated in exact closed form using expm1-stabilized expressions so First-Law
capacitor energy conservation holds to machine precision (~1e-15 J) at arbitrary dt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.bess_model import SOC_TOL
from src.cell_chemistry import (
    LFP_A123_ANR26650_2P3AH,
    NMC811_LG_M50_5AH,
    ChemistrySpec,
    MonotonePCHIP,
    get_chemistry_spec,
)
from src.thermal_model import arrhenius_factor, check_temperature_k

__all__ = [
    "ChemistryComparisonResult",
    "EISResult",
    "HPPCBenchmarkResult",
    "TwoRCBatteryModel",
    "TwoRCState",
    "TwoRCStepResult",
    "compute_eis_nyquist",
    "run_chemistry_comparison_benchmark",
]


def _phi1(x: float) -> float:
    """Compute (1 - exp(-x)) / x without catastrophic cancellation as x -> 0+."""
    if x < 1e-8:
        return 1.0 - 0.5 * x + (x * x) / 6.0
    return -math.expm1(-x) / x


@dataclass(frozen=True)
class TwoRCState:
    """Complete state vector of the coupled 2RC + hysteresis + thermal cell model."""

    soc: float = 1.0
    v_rc1_v: float = 0.0
    v_rc2_v: float = 0.0
    hysteresis: float = 0.0
    t_cell_k: float = 298.15

    def __post_init__(self) -> None:
        if not math.isfinite(self.soc) or not (-SOC_TOL <= self.soc <= 1.0 + SOC_TOL):
            raise ValueError(f"soc must lie in [0, 1], got {self.soc!r}")
        for name, val in (
            ("v_rc1_v", self.v_rc1_v),
            ("v_rc2_v", self.v_rc2_v),
            ("hysteresis", self.hysteresis),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val!r}")
        if abs(self.hysteresis) > 1.0 + 1e-9:
            raise ValueError(f"hysteresis state h must lie in [-1, 1], got {self.hysteresis!r}")
        check_temperature_k(self.t_cell_k)


@dataclass(frozen=True)
class TwoRCStepResult:
    """Single-step diagnostic output of :meth:`TwoRCBatteryModel.step`."""

    state: TwoRCState
    v_term_v: float
    u_ocv_v: float
    v_hyst_v: float
    q_irrev_w: float
    q_rev_w: float
    q_total_w: float
    r0_ohm: float
    r1_ohm: float
    r2_ohm: float


@dataclass(frozen=True)
class EISResult:
    """Complex-plane Electrochemical Impedance Spectroscopy (EIS) evaluation."""

    freq_hz: np.ndarray
    omega_rad_s: np.ndarray
    z_real_ohm: np.ndarray
    neg_z_imag_ohm: np.ndarray
    z_mag_ohm: np.ndarray
    phase_deg: np.ndarray
    r0_ohm: float
    r1_ohm: float
    r2_ohm: float
    f_peak_rc1_hz: float
    f_peak_rc2_hz: float


@dataclass(frozen=True)
class HPPCBenchmarkResult:
    """Trajectory of a dynamic pulse-relaxation benchmark on a single chemistry."""

    spec: ChemistrySpec
    time_s: np.ndarray
    current_a: np.ndarray
    soc: np.ndarray
    v_term_v: np.ndarray
    u_ocv_v: np.ndarray
    v_rc1_v: np.ndarray
    v_rc2_v: np.ndarray
    v_hyst_v: np.ndarray
    t_cell_c: np.ndarray
    q_irrev_w: np.ndarray
    q_rev_w: np.ndarray
    q_total_w: np.ndarray


@dataclass(frozen=True)
class ChemistryComparisonResult:
    """Combined multi-chemistry 2RC, Bernardi heat, and EIS benchmark bundle."""

    nmc_hppc: HPPCBenchmarkResult
    lfp_hppc: HPPCBenchmarkResult
    nmc_discharge_1c: HPPCBenchmarkResult
    eis_nmc_by_temp_c: dict[int, EISResult]
    soc_curve: np.ndarray
    nmc_ocv_v: np.ndarray
    nmc_dudt_mv_k: np.ndarray
    lfp_ocv_v: np.ndarray
    lfp_hyst_upper_v: np.ndarray
    lfp_hyst_lower_v: np.ndarray


class TwoRCBatteryModel:
    """Chemistry-parameterized 2RC Equivalent Circuit Model with Bernardi thermal coupling."""

    def __init__(
        self,
        chemistry: ChemistrySpec | str = NMC811_LG_M50_5AH,
        *,
        t_amb_k: float | None = None,
        h_a_w_k: float | None = None,
        enable_soc_impedance: bool = True,
        enable_entropic_heat: bool = True,
        enable_hysteresis: bool = True,
        polarization_heat_mode: Literal["resistor_exact", "overpotential"] = "resistor_exact",
    ) -> None:
        spec = get_chemistry_spec(chemistry) if isinstance(chemistry, str) else chemistry
        amb_k = spec.t_ref_k if t_amb_k is None else float(t_amb_k)
        check_temperature_k(amb_k)
        ha = spec.h_a_w_k if h_a_w_k is None else float(h_a_w_k)
        if not math.isfinite(ha) or ha < 0.0:
            raise ValueError(f"h_a_w_k must be finite and >= 0 (0 = adiabatic), got {ha!r}")
        if polarization_heat_mode not in ("resistor_exact", "overpotential"):
            raise ValueError(
                f"polarization_heat_mode must be 'resistor_exact' or 'overpotential', got {polarization_heat_mode!r}"
            )

        self._spec = spec
        self._t_amb_k = amb_k
        self._h_a_w_k = ha
        self._enable_soc_impedance = bool(enable_soc_impedance)
        self._enable_entropic_heat = bool(enable_entropic_heat)
        self._enable_hysteresis = bool(enable_hysteresis)
        self._polarization_heat_mode = polarization_heat_mode

        self._ocv_spline = MonotonePCHIP(spec.soc_grid, spec.ocv_v_grid)
        self._soc_grid_np = np.asarray(spec.soc_grid, dtype=float)
        self._dudt_grid_np = np.asarray(spec.dudt_v_per_k_grid, dtype=float)

    @property
    def spec(self) -> ChemistrySpec:
        return self._spec

    @property
    def t_amb_k(self) -> float:
        return self._t_amb_k

    @property
    def h_a_w_k(self) -> float:
        return self._h_a_w_k

    def get_ocv_ref(self, soc: float | np.ndarray) -> float | np.ndarray:
        """Equilibrium OCV [V] at T_ref evaluated via monotone PCHIP spline."""
        soc_clipped = np.clip(soc, 0.0, 1.0)
        return self._ocv_spline(soc_clipped)

    def get_docv_dsoc(self, soc: float | np.ndarray) -> float | np.ndarray:
        """Analytical slope dU_ocv/dSoC [V] from the monotone PCHIP spline."""
        soc_clipped = np.clip(soc, 0.0, 1.0)
        return self._ocv_spline.derivative(soc_clipped)

    def get_entropic_dudt(self, soc: float | np.ndarray) -> float | np.ndarray:
        """Entropic potential derivative dU_ocv/dT [V/K] interpolated over SoC."""
        scalar = np.ndim(soc) == 0
        soc_clipped = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        val = np.interp(soc_clipped, self._soc_grid_np, self._dudt_grid_np)
        return float(val) if scalar else val

    def get_ocv(self, soc: float | np.ndarray, t_cell_k: float | None = None) -> float | np.ndarray:
        """Temperature-corrected equilibrium OCV: U_ref(SoC) + (T - T_ref) * dU/dT(SoC)."""
        u_ref = self.get_ocv_ref(soc)
        if t_cell_k is None or not self._enable_entropic_heat:
            return u_ref
        check_temperature_k(t_cell_k)
        return u_ref + (t_cell_k - self._spec.t_ref_k) * self.get_entropic_dudt(soc)

    def get_impedances(self, soc: float, t_cell_k: float) -> tuple[float, float, float, float, float]:
        """Return (R0, R1, C1, R2, C2) at (soc, t_cell_k) in SI units (Ohm, F)."""
        check_temperature_k(t_cell_k)
        s = float(np.clip(soc, 0.0, 1.0))
        soc_scale = (
            1.0 + self._spec.r_low_soc_gain * math.exp(-self._spec.r_low_soc_decay * s)
            if self._enable_soc_impedance
            else 1.0
        )
        t_ref = self._spec.t_ref_k
        r0 = self._spec.r0_ref_ohm * soc_scale * arrhenius_factor(
            t_cell_k, ea_j_mol=self._spec.ea_r0_j_mol, t_ref_k=t_ref
        )
        r1 = self._spec.r1_ref_ohm * soc_scale * arrhenius_factor(
            t_cell_k, ea_j_mol=self._spec.ea_r1_j_mol, t_ref_k=t_ref
        )
        r2 = self._spec.r2_ref_ohm * soc_scale * arrhenius_factor(
            t_cell_k, ea_j_mol=self._spec.ea_r2_j_mol, t_ref_k=t_ref
        )
        return r0, r1, self._spec.c1_ref_f, r2, self._spec.c2_ref_f

    def terminal_voltage(self, state: TwoRCState, i_app: float) -> float:
        """Evaluate instantaneous terminal voltage V_term [V] at ``state`` and ``i_app`` [A]."""
        if not math.isfinite(i_app):
            raise ValueError(f"i_app must be finite, got {i_app!r}")
        r0, _, _, _, _ = self.get_impedances(state.soc, state.t_cell_k)
        u_ocv = float(self.get_ocv(state.soc, state.t_cell_k))
        v_h = (self._spec.hysteresis_max_v * state.hysteresis) if self._enable_hysteresis else 0.0
        return u_ocv + v_h - i_app * r0 - state.v_rc1_v - state.v_rc2_v

    @staticmethod
    def _exact_rc_branch_step(
        v_rc0: float,
        i_app: float,
        r_ohm: float,
        c_f: float,
        dt: float,
    ) -> tuple[float, float, float]:
        """Advance one RC branch exactly over dt and return (v_new, p_resistor_w, p_overpot_w).

        Returns:
            v_new: Exact end-of-step capacitor voltage V_rc(t + dt) [V].
            p_resistor_w: Exact step-averaged resistor Joule power (1/dt) int V_rc(s)^2/R ds [W].
            p_overpot_w: Exact step-averaged overpotential power (1/dt) int I * V_rc(s) ds [W].
        """
        tau = r_ohm * c_f
        x = dt / tau
        v_inf = i_app * r_ohm
        delta0 = v_rc0 - v_inf
        v_new = v_rc0 + delta0 * math.expm1(-x)

        phi1_x = _phi1(x)
        phi1_2x = _phi1(2.0 * x)
        p_resistor_w = (v_inf * v_inf + 2.0 * v_inf * delta0 * phi1_x + (delta0 * delta0) * phi1_2x) / r_ohm
        p_overpot_w = i_app * (v_inf + delta0 * phi1_x)
        return v_new, max(0.0, p_resistor_w), p_overpot_w

    def step(self, state: TwoRCState, i_app: float, dt: float) -> TwoRCStepResult:
        """Advance coupled 2RC + hysteresis + thermal state over ``dt`` [s]."""
        if not math.isfinite(i_app):
            raise ValueError(f"i_app must be finite, got {i_app!r}")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt!r}")

        q_coulomb = self._spec.nominal_capacity_ah * 3600.0
        new_soc = float(np.clip(state.soc - (i_app * dt) / q_coulomb, 0.0, 1.0))
        soc_mid = 0.5 * (state.soc + new_soc)

        r0, r1, c1, r2, c2 = self.get_impedances(soc_mid, state.t_cell_k)

        v_rc1_new, p_r1_w, p_eta1_w = self._exact_rc_branch_step(state.v_rc1_v, i_app, r1, c1, dt)
        v_rc2_new, p_r2_w, p_eta2_w = self._exact_rc_branch_step(state.v_rc2_v, i_app, r2, c2, dt)

        # Plett one-state hysteresis exact update (discharge I > 0 drives h -> -1)
        if self._enable_hysteresis and self._spec.hysteresis_max_v > 0.0 and abs(i_app) > 0.0:
            rate = (self._spec.hysteresis_rate_gamma * abs(i_app)) / q_coulomb
            xh = rate * dt
            target = -1.0 if i_app > 0.0 else 1.0
            h_new = state.hysteresis + (state.hysteresis - target) * math.expm1(-xh)
            h_avg = target + (state.hysteresis - target) * _phi1(xh)
            h_new = float(np.clip(h_new, -1.0, 1.0))
            p_hyst_w = max(0.0, -i_app * self._spec.hysteresis_max_v * h_avg)
        else:
            h_new = state.hysteresis if self._enable_hysteresis else 0.0
            p_hyst_w = 0.0

        p_ohmic_w = (i_app * i_app) * r0
        if self._polarization_heat_mode == "resistor_exact":
            q_irrev_w = p_ohmic_w + p_r1_w + p_r2_w + p_hyst_w
        else:
            q_irrev_w = p_ohmic_w + p_eta1_w + p_eta2_w + p_hyst_w

        dudt_mid = float(self.get_entropic_dudt(soc_mid)) if self._enable_entropic_heat else 0.0
        q_rev_w = -i_app * state.t_cell_k * dudt_mid
        q_total_w = q_irrev_w + q_rev_w

        mcp = self._spec.mass_kg * self._spec.cp_j_kgk
        if self._h_a_w_k == 0.0:
            t_new_k = state.t_cell_k + (q_total_w * dt) / mcp
        else:
            tau_th = mcp / self._h_a_w_k
            t_ss_k = self._t_amb_k + q_total_w / self._h_a_w_k
            t_new_k = t_ss_k + (state.t_cell_k - t_ss_k) * math.exp(-dt / tau_th)
        check_temperature_k(t_new_k)

        new_state = TwoRCState(
            soc=new_soc,
            v_rc1_v=v_rc1_new,
            v_rc2_v=v_rc2_new,
            hysteresis=h_new,
            t_cell_k=t_new_k,
        )
        u_ocv_new = float(self.get_ocv(new_soc, t_new_k))
        v_hyst_new = (self._spec.hysteresis_max_v * h_new) if self._enable_hysteresis else 0.0
        r0_end, _, _, _, _ = self.get_impedances(new_soc, t_new_k)
        v_term_new = u_ocv_new + v_hyst_new - i_app * r0_end - v_rc1_new - v_rc2_new

        return TwoRCStepResult(
            state=new_state,
            v_term_v=v_term_new,
            u_ocv_v=u_ocv_new,
            v_hyst_v=v_hyst_new,
            q_irrev_w=q_irrev_w,
            q_rev_w=q_rev_w,
            q_total_w=q_total_w,
            r0_ohm=r0_end,
            r1_ohm=r1,
            r2_ohm=r2,
        )

    def simulate(
        self,
        current_profile_a: np.ndarray,
        dt: float,
        initial_state: TwoRCState | None = None,
    ) -> HPPCBenchmarkResult:
        """Simulate a discrete-time current profile I[k] of length N with step dt [s]."""
        i_arr = np.asarray(current_profile_a, dtype=float)
        if i_arr.ndim != 1 or len(i_arr) == 0 or not np.all(np.isfinite(i_arr)):
            raise ValueError("current_profile_a must be a non-empty finite 1D array")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt!r}")

        state = TwoRCState(soc=1.0, t_cell_k=self._t_amb_k) if initial_state is None else initial_state
        n = len(i_arr)
        time_s = np.arange(n, dtype=float) * dt
        soc = np.empty(n, dtype=float)
        v_term = np.empty(n, dtype=float)
        u_ocv = np.empty(n, dtype=float)
        v_rc1 = np.empty(n, dtype=float)
        v_rc2 = np.empty(n, dtype=float)
        v_hyst = np.empty(n, dtype=float)
        t_cell_c = np.empty(n, dtype=float)
        q_irrev = np.empty(n, dtype=float)
        q_rev = np.empty(n, dtype=float)
        q_total = np.empty(n, dtype=float)

        for k in range(n):
            res = self.step(state, float(i_arr[k]), dt)
            state = res.state
            soc[k] = state.soc
            v_term[k] = res.v_term_v
            u_ocv[k] = res.u_ocv_v
            v_rc1[k] = state.v_rc1_v
            v_rc2[k] = state.v_rc2_v
            v_hyst[k] = res.v_hyst_v
            t_cell_c[k] = state.t_cell_k - 273.15
            q_irrev[k] = res.q_irrev_w
            q_rev[k] = res.q_rev_w
            q_total[k] = res.q_total_w

        return HPPCBenchmarkResult(
            spec=self._spec,
            time_s=time_s,
            current_a=i_arr.copy(),
            soc=soc,
            v_term_v=v_term,
            u_ocv_v=u_ocv,
            v_rc1_v=v_rc1,
            v_rc2_v=v_rc2,
            v_hyst_v=v_hyst,
            t_cell_c=t_cell_c,
            q_irrev_w=q_irrev,
            q_rev_w=q_rev,
            q_total_w=q_total,
        )


def compute_eis_nyquist(
    model: TwoRCBatteryModel,
    *,
    soc: float = 0.50,
    t_cell_k: float = 298.15,
    freq_min_hz: float = 1e-4,
    freq_max_hz: float = 1e3,
    n_points: int = 200,
) -> EISResult:
    """Evaluate exact small-signal AC impedance Z(jw) of the 2RC network."""
    if freq_min_hz <= 0.0 or freq_max_hz <= freq_min_hz or n_points < 8:
        raise ValueError("Require 0 < freq_min_hz < freq_max_hz and n_points >= 8")
    r0, r1, c1, r2, c2 = model.get_impedances(soc, t_cell_k)
    freq_hz = np.geomspace(freq_min_hz, freq_max_hz, n_points)
    omega = 2.0 * math.pi * freq_hz
    jw = 1j * omega
    z_complex = r0 + r1 / (1.0 + jw * r1 * c1) + r2 / (1.0 + jw * r2 * c2)
    return EISResult(
        freq_hz=freq_hz,
        omega_rad_s=omega,
        z_real_ohm=np.real(z_complex),
        neg_z_imag_ohm=-np.imag(z_complex),
        z_mag_ohm=np.abs(z_complex),
        phase_deg=np.degrees(np.angle(z_complex)),
        r0_ohm=r0,
        r1_ohm=r1,
        r2_ohm=r2,
        f_peak_rc1_hz=1.0 / (2.0 * math.pi * r1 * c1),
        f_peak_rc2_hz=1.0 / (2.0 * math.pi * r2 * c2),
    )


def run_chemistry_comparison_benchmark(dt: float = 1.0) -> ChemistryComparisonResult:
    """Run reproducible NMC811 vs LFP HPPC pulse-relaxation, 1C Bernardi heat & EIS suite."""
    nmc_model = TwoRCBatteryModel(NMC811_LG_M50_5AH)
    lfp_model = TwoRCBatteryModel(LFP_A123_ANR26650_2P3AH)

    # 1) 1800 s HPPC bidirectional pulse-relaxation sequence starting at 80% SoC:
    #    120 s rest -> 300 s 1C discharge -> 600 s rest -> 300 s 0.75C charge -> 480 s rest
    n_hppc = int(round(1800.0 / dt))
    t_hppc = np.arange(n_hppc, dtype=float) * dt

    def make_hppc_profile(cap_ah: float) -> np.ndarray:
        prof = np.zeros(n_hppc, dtype=float)
        prof[(t_hppc >= 120.0) & (t_hppc < 420.0)] = 1.0 * cap_ah
        prof[(t_hppc >= 1020.0) & (t_hppc < 1320.0)] = -0.75 * cap_ah
        return prof

    nmc_hppc = nmc_model.simulate(
        make_hppc_profile(NMC811_LG_M50_5AH.nominal_capacity_ah),
        dt=dt,
        initial_state=TwoRCState(soc=0.80, t_cell_k=298.15),
    )
    lfp_hppc = lfp_model.simulate(
        make_hppc_profile(LFP_A123_ANR26650_2P3AH.nominal_capacity_ah),
        dt=dt,
        initial_state=TwoRCState(soc=0.80, t_cell_k=298.15),
    )

    # 2) Full 1C galvanostatic discharge on NMC811 (SoC 0.98 -> 0.04, 3384 s) showing
    #    Bernardi reversible entropic cooling (high SoC) -> heating (low SoC) crossover
    n_1c = int(round(3384.0 / dt))
    i_1c = np.full(n_1c, NMC811_LG_M50_5AH.nominal_capacity_ah, dtype=float)
    nmc_discharge_1c = nmc_model.simulate(
        i_1c,
        dt=dt,
        initial_state=TwoRCState(soc=0.98, t_cell_k=298.15),
    )

    # 3) Multi-temperature EIS Nyquist spectra on NMC811 at 50% SoC (10 C, 25 C, 45 C)
    eis_by_temp: dict[int, EISResult] = {}
    for temp_c in (10, 25, 45):
        eis_by_temp[temp_c] = compute_eis_nyquist(
            nmc_model,
            soc=0.50,
            t_cell_k=273.15 + float(temp_c),
            freq_min_hz=1e-5,
            freq_max_hz=1e3,
            n_points=260,
        )

    soc_curve = np.linspace(0.0, 1.0, 201)
    nmc_ocv_v = np.asarray(nmc_model.get_ocv_ref(soc_curve), dtype=float)
    nmc_dudt_mv_k = np.asarray(nmc_model.get_entropic_dudt(soc_curve), dtype=float) * 1e3
    lfp_ocv_v = np.asarray(lfp_model.get_ocv_ref(soc_curve), dtype=float)
    mh = LFP_A123_ANR26650_2P3AH.hysteresis_max_v
    return ChemistryComparisonResult(
        nmc_hppc=nmc_hppc,
        lfp_hppc=lfp_hppc,
        nmc_discharge_1c=nmc_discharge_1c,
        eis_nmc_by_temp_c=eis_by_temp,
        soc_curve=soc_curve,
        nmc_ocv_v=nmc_ocv_v,
        nmc_dudt_mv_k=nmc_dudt_mv_k,
        lfp_ocv_v=lfp_ocv_v,
        lfp_hyst_upper_v=lfp_ocv_v + mh,
        lfp_hyst_lower_v=lfp_ocv_v - mh,
    )
