"""Coupled 2-node radial thermal network, CC-CV taper & receding-horizon SOP BMS engine.

Governing 2-node radial core-surface thermal state-space equations (Forgez et al.,
J. Power Sources 195(9), 2961-2968, 2010; Lin et al., J. Power Sources 257, 1-11, 2014):

    C_core * dT_core/dt = Q_gen - (T_core - T_surf) / R_cs
    C_surf * dT_surf/dt = (T_core - T_surf) / R_cs - h_sa * (T_surf - T_amb)

Writing theta = [T_core - T_amb, T_surf - T_amb]^T gives d(theta)/dt = A * theta + b * Q_gen,
whose 2x2 dissipative generator A always possesses real distinct eigenvalues
lambda_1 < lambda_2 <= 0 (with lambda_2 = 0 in the adiabatic limit h_sa = 0). Over each
step dt, theta(t + dt) is evaluated in exact closed form via the Putzer matrix exponential
Phi(dt) = exp(A * dt) and forced integral Psi(dt) = int_0^dt exp(A * s) ds, guaranteeing
unconditional stability for arbitrary dt > 0 and machine-precision First-Law thermal
energy conservation (~1e-14 J).

Coupled with :class:`~src.ecm_2rc.TwoRCBatteryModel` (impedances and Bernardi reversible
heat evaluated at electrochemically active jelly-roll temperature T_core), this module
provides:
- :class:`TwoNodeThermalNetwork` and :class:`CoupledTwoNodeBESS`
- :func:`simulate_cc_cv_charge` — zero-overshoot implicit CC-CV fast-charging with
  active T_core ceiling derating
- :func:`compute_sop_limits` — multi-constraint receding-horizon State-of-Power (SOP)
  BMS power-envelope solver (Plett, BMS Vol. II, Ch. 6) identifying the active physical
  bottleneck ("current", "soc", "voltage", or "thermal_core")
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
    get_chemistry_spec,
)
from src.ecm_2rc import TwoRCBatteryModel, TwoRCState, _phi1
from src.thermal_model import T_MAX_K, check_temperature_k

__all__ = [
    "CCCVChargeResult",
    "CoupledTwoNodeBESS",
    "CoupledTwoNodeState",
    "CoupledTwoNodeStepResult",
    "SOPBottleneck",
    "SOPResult",
    "ThermalPulseRun",
    "ThermalSOPBenchmarkResult",
    "TwoNodeThermalNetwork",
    "TwoNodeThermalState",
    "TwoNodeThermalStepResult",
    "compute_sop_limits",
    "run_thermal_sop_benchmark",
    "simulate_cc_cv_charge",
]

SOPBottleneck = Literal["current", "soc", "voltage", "thermal_core"]


@dataclass(frozen=True)
class TwoNodeThermalState:
    """Radial core and casing-surface temperatures [K] of a cylindrical Li-ion cell."""

    t_core_k: float = 298.15
    t_surf_k: float = 298.15

    def __post_init__(self) -> None:
        check_temperature_k(self.t_core_k)
        check_temperature_k(self.t_surf_k)

    @property
    def t_core_c(self) -> float:
        return self.t_core_k - 273.15

    @property
    def t_surf_c(self) -> float:
        return self.t_surf_k - 273.15

    @property
    def delta_t_core_surf_k(self) -> float:
        return self.t_core_k - self.t_surf_k


@dataclass(frozen=True)
class TwoNodeThermalStepResult:
    """Single-step output of :meth:`TwoNodeThermalNetwork.step`."""

    state: TwoNodeThermalState
    q_core_to_surf_w: float
    q_surf_to_amb_w: float


class TwoNodeThermalNetwork:
    """Exact matrix-exponential 2-node radial (core-surface) thermal network."""

    def __init__(
        self,
        *,
        c_core_j_k: float = 46.0,
        c_surf_j_k: float = 10.1,
        r_cs_k_w: float = 1.95,
        h_sa_w_k: float = 0.35,
        t_amb_k: float = 298.15,
    ) -> None:
        for name, val in (
            ("c_core_j_k", c_core_j_k),
            ("c_surf_j_k", c_surf_j_k),
            ("r_cs_k_w", r_cs_k_w),
        ):
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {val!r}")
        if not math.isfinite(h_sa_w_k) or h_sa_w_k < 0.0:
            raise ValueError(f"h_sa_w_k must be finite and >= 0 (0 = adiabatic), got {h_sa_w_k!r}")
        check_temperature_k(t_amb_k)

        self._c_core = float(c_core_j_k)
        self._c_surf = float(c_surf_j_k)
        self._r_cs = float(r_cs_k_w)
        self._k_cs = 1.0 / self._r_cs
        self._h_sa = float(h_sa_w_k)
        self._t_amb_k = float(t_amb_k)

        a11 = -self._k_cs / self._c_core
        a12 = self._k_cs / self._c_core
        a21 = self._k_cs / self._c_surf
        a22 = -(self._k_cs + self._h_sa) / self._c_surf
        self._A = np.array([[a11, a12], [a21, a22]], dtype=float)

        # Cancellation-free real negative eigenvalues: lambda_1 < lambda_2 <= 0
        trace = a11 + a22
        det = (self._k_cs * self._h_sa) / (self._c_core * self._c_surf)
        half_diff = 0.5 * (a11 - a22)
        disc_root = math.sqrt(half_diff * half_diff + a12 * a21)
        lam1 = 0.5 * trace - disc_root
        lam2 = det / lam1  # Exact even as h_sa -> 0+ (lam2 == 0.0 when h_sa == 0)
        self._lam1 = lam1
        self._lam2 = lam2

    @classmethod
    def from_chemistry(
        cls,
        chemistry: ChemistrySpec | str = NMC811_LG_M50_5AH,
        *,
        t_amb_k: float | None = None,
        h_sa_w_k: float | None = None,
        r_cs_k_w: float | None = None,
    ) -> TwoNodeThermalNetwork:
        spec = get_chemistry_spec(chemistry) if isinstance(chemistry, str) else chemistry
        c_tot = spec.mass_kg * spec.cp_j_kgk
        f_core = spec.core_heat_capacity_fraction
        return cls(
            c_core_j_k=f_core * c_tot,
            c_surf_j_k=(1.0 - f_core) * c_tot,
            r_cs_k_w=spec.r_core_surf_k_w if r_cs_k_w is None else float(r_cs_k_w),
            h_sa_w_k=spec.h_a_w_k if h_sa_w_k is None else float(h_sa_w_k),
            t_amb_k=spec.t_ref_k if t_amb_k is None else float(t_amb_k),
        )

    @property
    def c_core_j_k(self) -> float:
        return self._c_core

    @property
    def c_surf_j_k(self) -> float:
        return self._c_surf

    @property
    def r_cs_k_w(self) -> float:
        return self._r_cs

    @property
    def h_sa_w_k(self) -> float:
        return self._h_sa

    @property
    def t_amb_k(self) -> float:
        return self._t_amb_k

    @property
    def eigenvalues_per_s(self) -> tuple[float, float]:
        return (self._lam1, self._lam2)

    @property
    def modal_time_constants_s(self) -> tuple[float, float]:
        tau_fast = -1.0 / self._lam1
        tau_slow = math.inf if self._lam2 == 0.0 else -1.0 / self._lam2
        return (tau_fast, tau_slow)

    def steady_state_temperatures(self, q_gen_w: float) -> tuple[float, float]:
        """Analytical steady-state (T_core_inf, T_surf_inf) [K] for constant heat Q_gen."""
        if not math.isfinite(q_gen_w):
            raise ValueError("q_gen_w must be finite")
        if self._h_sa == 0.0:
            if abs(q_gen_w) < 1e-15:
                return (self._t_amb_k, self._t_amb_k)
            sign_inf = math.inf if q_gen_w > 0.0 else -math.inf
            return (sign_inf, sign_inf)
        t_surf_inf = self._t_amb_k + q_gen_w / self._h_sa
        t_core_inf = t_surf_inf + q_gen_w * self._r_cs
        return (t_core_inf, t_surf_inf)

    def _propagate_raw(self, t_core_k: float, t_surf_k: float, q_gen_w: float, dt: float) -> tuple[float, float]:
        """Unchecked exact matrix-exponential step used inside fast horizon rollouts."""
        th0_0 = t_core_k - self._t_amb_k
        th0_1 = t_surf_k - self._t_amb_k

        e2 = math.exp(self._lam2 * dt)
        gap = self._lam2 - self._lam1  # Strictly > 0

        # Phi(dt) = e2 * I + (e2 * (1 - exp(-gap * dt)) / gap) * (A - lam2 * I)
        c_phi = e2 * (-math.expm1(-gap * dt)) / gap
        a00_m = self._A[0, 0] - self._lam2
        a11_m = self._A[1, 1] - self._lam2
        phi00 = e2 + c_phi * a00_m
        phi01 = c_phi * self._A[0, 1]
        phi10 = c_phi * self._A[1, 0]
        phi11 = e2 + c_phi * a11_m

        # Psi(dt) = j2 * I + ((j2 - j1) / gap) * (A - lam2 * I), j_i = dt * phi1(-lam_i * dt)
        j1 = dt * _phi1(-self._lam1 * dt)
        j2 = dt * _phi1(-self._lam2 * dt)
        c_psi = (j2 - j1) / gap
        psi00 = j2 + c_psi * a00_m
        psi10 = c_psi * self._A[1, 0]

        u0 = q_gen_w / self._c_core
        th1_0 = phi00 * th0_0 + phi01 * th0_1 + psi00 * u0
        th1_1 = phi10 * th0_0 + phi11 * th0_1 + psi10 * u0
        return (self._t_amb_k + th1_0, self._t_amb_k + th1_1)

    def step(
        self,
        state: TwoNodeThermalState,
        q_gen_w: float,
        dt: float,
    ) -> TwoNodeThermalStepResult:
        """Advance (T_core, T_surf) over dt [s] under constant core heat rate q_gen_w [W]."""
        if not math.isfinite(q_gen_w):
            raise ValueError(f"q_gen_w must be finite, got {q_gen_w!r}")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt!r}")

        tc_new, ts_new = self._propagate_raw(state.t_core_k, state.t_surf_k, float(q_gen_w), float(dt))
        next_state = TwoNodeThermalState(t_core_k=tc_new, t_surf_k=ts_new)

        # Exact step-averaged heat fluxes from First-Law control-volume balances
        q_cs_avg = float(q_gen_w) - self._c_core * (tc_new - state.t_core_k) / dt
        q_sa_avg = 0.0 if self._h_sa == 0.0 else (q_cs_avg - self._c_surf * (ts_new - state.t_surf_k) / dt)
        return TwoNodeThermalStepResult(
            state=next_state,
            q_core_to_surf_w=q_cs_avg,
            q_surf_to_amb_w=q_sa_avg,
        )


@dataclass(frozen=True)
class CoupledTwoNodeState:
    """State vector of the coupled 2RC electrical + 2-node radial thermal cell model."""

    soc: float = 1.0
    v_rc1_v: float = 0.0
    v_rc2_v: float = 0.0
    hysteresis: float = 0.0
    t_core_k: float = 298.15
    t_surf_k: float = 298.15

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
            raise ValueError(f"hysteresis must lie in [-1, 1], got {self.hysteresis!r}")
        check_temperature_k(self.t_core_k)
        check_temperature_k(self.t_surf_k)

    @property
    def t_core_c(self) -> float:
        return self.t_core_k - 273.15

    @property
    def t_surf_c(self) -> float:
        return self.t_surf_k - 273.15


@dataclass(frozen=True)
class CoupledTwoNodeStepResult:
    """Single-step diagnostic output of :meth:`CoupledTwoNodeBESS.step`."""

    state: CoupledTwoNodeState
    v_term_v: float
    u_ocv_v: float
    v_hyst_v: float
    q_irrev_w: float
    q_rev_w: float
    q_total_w: float
    q_core_to_surf_w: float
    q_surf_to_amb_w: float
    r0_ohm: float
    r1_ohm: float
    r2_ohm: float


class CoupledTwoNodeBESS:
    """2RC multi-chemistry ECM coupled to a 2-node radial core-surface thermal network."""

    def __init__(
        self,
        chemistry: ChemistrySpec | str = NMC811_LG_M50_5AH,
        *,
        t_amb_k: float | None = None,
        h_sa_w_k: float | None = None,
        r_cs_k_w: float | None = None,
        enable_soc_impedance: bool = True,
        enable_entropic_heat: bool = True,
        enable_hysteresis: bool = True,
    ) -> None:
        spec = get_chemistry_spec(chemistry) if isinstance(chemistry, str) else chemistry
        amb_k = spec.t_ref_k if t_amb_k is None else check_temperature_k(t_amb_k)
        # Set internal 1-node h_a_w_k=0 on electrical helper; 2-node network owns all heat transfer
        self._ecm = TwoRCBatteryModel(
            spec,
            t_amb_k=amb_k,
            h_a_w_k=0.0,
            enable_soc_impedance=enable_soc_impedance,
            enable_entropic_heat=enable_entropic_heat,
            enable_hysteresis=enable_hysteresis,
        )
        self._thermal = TwoNodeThermalNetwork.from_chemistry(
            spec,
            t_amb_k=amb_k,
            h_sa_w_k=h_sa_w_k,
            r_cs_k_w=r_cs_k_w,
        )

    @property
    def spec(self) -> ChemistrySpec:
        return self._ecm.spec

    @property
    def ecm(self) -> TwoRCBatteryModel:
        return self._ecm

    @property
    def thermal(self) -> TwoNodeThermalNetwork:
        return self._thermal

    def terminal_voltage(self, state: CoupledTwoNodeState, i_app: float) -> float:
        ecm_state = TwoRCState(
            soc=state.soc,
            v_rc1_v=state.v_rc1_v,
            v_rc2_v=state.v_rc2_v,
            hysteresis=state.hysteresis,
            t_cell_k=state.t_core_k,
        )
        return self._ecm.terminal_voltage(ecm_state, i_app)

    def step(self, state: CoupledTwoNodeState, i_app: float, dt: float) -> CoupledTwoNodeStepResult:
        """Advance coupled 2RC electrical states (at T_core) and 2-node temperatures over dt."""
        ecm_in = TwoRCState(
            soc=state.soc,
            v_rc1_v=state.v_rc1_v,
            v_rc2_v=state.v_rc2_v,
            hysteresis=state.hysteresis,
            t_cell_k=state.t_core_k,
        )
        # Evaluate electrical impedances, polarization work, and Bernardi heat at T_core
        # Note: pass a clamped shadow temperature to ecm_in check while letting TwoNodeThermalNetwork
        # validate actual core/surface bounds after cooling is included.
        r0, r1, c1, r2, c2 = self._ecm.get_impedances(ecm_in.soc, ecm_in.t_cell_k)
        if not math.isfinite(i_app):
            raise ValueError(f"i_app must be finite, got {i_app!r}")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt!r}")

        q_coulomb = self.spec.nominal_capacity_ah * 3600.0
        soc0 = float(np.clip(state.soc, 0.0, 1.0))
        soc_new = float(np.clip(soc0 - (i_app * dt) / q_coulomb, 0.0, 1.0))
        soc_mid = 0.5 * (soc0 + soc_new)
        r0, r1, c1, r2, c2 = self._ecm.get_impedances(soc_mid, state.t_core_k)

        # Exact 2RC polarization step + exact resistor Joule dissipation
        x1 = dt / (r1 * c1)
        v1_inf = i_app * r1
        dv1 = state.v_rc1_v - v1_inf
        v1_new = state.v_rc1_v + dv1 * math.expm1(-x1)
        e_r1_j = (
            (v1_inf * v1_inf) * dt
            + 2.0 * v1_inf * dv1 * dt * _phi1(x1)
            + (dv1 * dv1) * dt * _phi1(2.0 * x1)
        ) / r1

        x2 = dt / (r2 * c2)
        v2_inf = i_app * r2
        dv2 = state.v_rc2_v - v2_inf
        v2_new = state.v_rc2_v + dv2 * math.expm1(-x2)
        e_r2_j = (
            (v2_inf * v2_inf) * dt
            + 2.0 * v2_inf * dv2 * dt * _phi1(x2)
            + (dv2 * dv2) * dt * _phi1(2.0 * x2)
        ) / r2

        if self._ecm._enable_hysteresis and self.spec.hysteresis_max_v > 0.0 and abs(i_app) > 0.0:
            xh = ((self.spec.hysteresis_rate_gamma * abs(i_app)) / q_coulomb) * dt
            target = -1.0 if i_app > 0.0 else 1.0
            h_new = float(np.clip(state.hysteresis + (state.hysteresis - target) * math.expm1(-xh), -1.0, 1.0))
        else:
            h_new = state.hysteresis if self._ecm._enable_hysteresis else 0.0

        q_irrev_w = (i_app * i_app) * r0 + (e_r1_j + e_r2_j) / dt
        dudt_mid = float(self._ecm.get_entropic_dudt(soc_mid)) if self._ecm._enable_entropic_heat else 0.0
        q_rev_w = -i_app * state.t_core_k * dudt_mid
        q_total_w = q_irrev_w + q_rev_w

        th_step = self._thermal.step(
            TwoNodeThermalState(t_core_k=state.t_core_k, t_surf_k=state.t_surf_k),
            q_total_w,
            dt,
        )

        next_state = CoupledTwoNodeState(
            soc=soc_new,
            v_rc1_v=v1_new,
            v_rc2_v=v2_new,
            hysteresis=h_new,
            t_core_k=th_step.state.t_core_k,
            t_surf_k=th_step.state.t_surf_k,
        )
        r0_end, _, _, _, _ = self._ecm.get_impedances(soc_new, next_state.t_core_k)
        u_ocv_end = float(self._ecm.get_ocv(soc_new, next_state.t_core_k))
        v_hyst_end = (self.spec.hysteresis_max_v * h_new) if self._ecm._enable_hysteresis else 0.0
        v_term_end = u_ocv_end + v_hyst_end - i_app * r0_end - v1_new - v2_new

        return CoupledTwoNodeStepResult(
            state=next_state,
            v_term_v=v_term_end,
            u_ocv_v=u_ocv_end,
            v_hyst_v=v_hyst_end,
            q_irrev_w=q_irrev_w,
            q_rev_w=q_rev_w,
            q_total_w=q_total_w,
            q_core_to_surf_w=th_step.q_core_to_surf_w,
            q_surf_to_amb_w=th_step.q_surf_to_amb_w,
            r0_ohm=r0_end,
            r1_ohm=r1,
            r2_ohm=r2,
        )


@dataclass(frozen=True)
class CCCVChargeResult:
    """Complete trajectory & summary metrics of a closed-loop CC-CV fast-charge cycle."""

    spec: ChemistrySpec
    time_s: np.ndarray
    current_a: np.ndarray
    soc: np.ndarray
    v_term_v: np.ndarray
    u_ocv_v: np.ndarray
    t_core_c: np.ndarray
    t_surf_c: np.ndarray
    q_total_w: np.ndarray
    phase_code: np.ndarray  # 0 = CC, 1 = THERMAL_DERATE, 2 = CV
    cc_to_cv_time_s: float
    total_charge_time_s: float
    final_soc: float
    peak_t_core_c: float
    peak_t_surf_c: float
    peak_core_surf_delta_c: float
    max_v_overshoot_v: float


def simulate_cc_cv_charge(
    bess: CoupledTwoNodeBESS,
    *,
    initial_soc: float = 0.10,
    cc_c_rate: float = 2.0,
    v_cv_v: float | None = None,
    cutoff_c_rate: float = 0.05,
    t_core_max_c: float | None = 46.0,
    initial_temp_c: float | None = None,
    dt: float = 2.0,
    max_time_s: float = 7200.0,
) -> CCCVChargeResult:
    """Simulate closed-loop CC-CV charging with exact zero-overshoot CV & T_core ceiling."""
    if not (0.0 <= initial_soc < 1.0):
        raise ValueError("initial_soc must lie in [0, 1)")
    if cc_c_rate <= 0.0 or cutoff_c_rate <= 0.0 or cutoff_c_rate >= cc_c_rate:
        raise ValueError("Require 0 < cutoff_c_rate < cc_c_rate")
    if dt <= 0.0 or max_time_s <= 0.0:
        raise ValueError("dt and max_time_s must be > 0")

    spec = bess.spec
    v_target = spec.v_max_v if v_cv_v is None else float(v_cv_v)
    t0_k = bess.thermal.t_amb_k if initial_temp_c is None else check_temperature_k(initial_temp_c + 273.15)
    t_ceil_k = None if t_core_max_c is None else check_temperature_k(t_core_max_c + 273.15)

    cap_ah = spec.nominal_capacity_ah
    i_cc_mag = cc_c_rate * cap_ah
    i_cut_mag = cutoff_c_rate * cap_ah

    state = CoupledTwoNodeState(soc=float(initial_soc), t_core_k=t0_k, t_surf_k=t0_k)
    max_steps = int(math.ceil(max_time_s / dt))

    t_list, i_list, soc_list, vt_list, ocv_list = [], [], [], [], []
    tc_list, ts_list, q_list, phase_list = [], [], [], []
    in_cv = False
    cc_to_cv_s = math.nan

    for k in range(max_steps):
        i_mag = i_cc_mag
        phase = 0  # 0 = CC, 1 = THERMAL_DERATE, 2 = CV

        # 1. Enforce core temperature ceiling including post-step 2RC relaxation heat tail
        if t_ceil_k is not None:
            def _peak_tc_with_relax(i_trial_mag: float) -> float:
                st_step = bess.step(state, -i_trial_mag, dt).state
                _, _, max_tc_relax, _ = _rollout_horizon_extremes(bess, st_step, 0.0, 3.0 * dt, 3)
                return max(st_step.t_core_k, max_tc_relax)

            if state.t_core_k >= t_ceil_k or _peak_tc_with_relax(0.0) >= t_ceil_k:
                i_mag = 0.0
                phase = 1
            elif _peak_tc_with_relax(i_mag) > t_ceil_k:
                lo, hi = 0.0, i_mag
                for _ in range(28):
                    mid = 0.5 * (lo + hi)
                    if _peak_tc_with_relax(mid) <= t_ceil_k:
                        lo = mid
                    else:
                        hi = mid
                i_mag = lo
                phase = 1

        # 2. Enforce upper voltage ceiling V_term(t + dt; -i_mag) <= v_target
        cand = bess.step(state, -i_mag, dt)
        if cand.v_term_v > v_target or in_cv:
            rest_step = bess.step(state, 0.0, dt)
            if rest_step.v_term_v >= v_target:
                i_mag = 0.0
                cand = rest_step
            elif cand.v_term_v > v_target:
                lo, hi = 0.0, i_mag
                for _ in range(44):
                    mid = 0.5 * (lo + hi)
                    if bess.step(state, -mid, dt).v_term_v <= v_target:
                        lo = mid
                    else:
                        hi = mid
                i_mag = lo
                cand = bess.step(state, -i_mag, dt)
            if cand.v_term_v >= v_target - 1e-4 or in_cv:
                if not in_cv:
                    in_cv = True
                    cc_to_cv_s = (k + 1) * dt
                phase = 2

        state = cand.state
        t_now = (k + 1) * dt
        t_list.append(t_now)
        i_list.append(-i_mag)
        soc_list.append(state.soc)
        vt_list.append(cand.v_term_v)
        ocv_list.append(cand.u_ocv_v)
        tc_list.append(state.t_core_c)
        ts_list.append(state.t_surf_c)
        q_list.append(cand.q_total_w)
        phase_list.append(phase)

        if in_cv and i_mag <= i_cut_mag:
            break
        if state.soc >= 0.9995:
            break

    tc_arr = np.asarray(tc_list, dtype=float)
    ts_arr = np.asarray(ts_list, dtype=float)
    vt_arr = np.asarray(vt_list, dtype=float)
    return CCCVChargeResult(
        spec=spec,
        time_s=np.asarray(t_list, dtype=float),
        current_a=np.asarray(i_list, dtype=float),
        soc=np.asarray(soc_list, dtype=float),
        v_term_v=vt_arr,
        u_ocv_v=np.asarray(ocv_list, dtype=float),
        t_core_c=tc_arr,
        t_surf_c=ts_arr,
        q_total_w=np.asarray(q_list, dtype=float),
        phase_code=np.asarray(phase_list, dtype=int),
        cc_to_cv_time_s=float(t_list[-1] if math.isnan(cc_to_cv_s) else cc_to_cv_s),
        total_charge_time_s=float(t_list[-1]),
        final_soc=float(soc_list[-1]),
        peak_t_core_c=float(np.max(tc_arr)),
        peak_t_surf_c=float(np.max(ts_arr)),
        peak_core_surf_delta_c=float(np.max(tc_arr - ts_arr)),
        max_v_overshoot_v=float(max(0.0, float(np.max(vt_arr)) - v_target)),
    )


@dataclass(frozen=True)
class SOPResult:
    """Receding-horizon discharge & charge State-of-Power (SOP) BMS envelope."""

    horizon_s: float
    i_dis_max_a: float
    i_chg_max_a: float
    p_dis_max_w: float
    p_chg_max_w: float
    v_dis_end_v: float
    v_chg_end_v: float
    t_core_dis_end_c: float
    t_core_chg_end_c: float
    dis_bottleneck: SOPBottleneck
    chg_bottleneck: SOPBottleneck
    dis_limits_by_constraint_a: dict[SOPBottleneck, float]
    chg_limits_by_constraint_a: dict[SOPBottleneck, float]


def _rollout_horizon_extremes(
    bess: CoupledTwoNodeBESS,
    state0: CoupledTwoNodeState,
    i_const_a: float,
    horizon_s: float,
    n_steps: int,
) -> tuple[float, float, float, float]:
    """Roll out constant current over horizon_s; return (min_V, max_V, max_T_core_K, V_end)."""
    dt = horizon_s / n_steps
    spec = bess.spec
    ecm = bess.ecm
    th = bess.thermal
    q_coulomb = spec.nominal_capacity_ah * 3600.0
    use_hyst = ecm._enable_hysteresis and spec.hysteresis_max_v > 0.0
    use_ent = ecm._enable_entropic_heat

    soc = min(max(state0.soc, 0.0), 1.0)
    v1 = state0.v_rc1_v
    v2 = state0.v_rc2_v
    h_val = state0.hysteresis
    tc_k = state0.t_core_k
    ts_k = state0.t_surf_k

    min_v = math.inf
    max_v = -math.inf
    max_tc = tc_k
    v_end = bess.terminal_voltage(state0, i_const_a)
    for _ in range(n_steps):
        soc_new = min(max(soc - (i_const_a * dt) / q_coulomb, 0.0), 1.0)
        soc_mid = 0.5 * (soc + soc_new)
        tc_eval = min(max(tc_k, 253.15), T_MAX_K)
        r0, r1, c1, r2, c2 = ecm.get_impedances(soc_mid, tc_eval)

        x1 = dt / (r1 * c1)
        v1_inf = i_const_a * r1
        dv1 = v1 - v1_inf
        v1_new = v1 + dv1 * math.expm1(-x1)
        e_r1_j = (
            (v1_inf * v1_inf) * dt
            + 2.0 * v1_inf * dv1 * dt * _phi1(x1)
            + (dv1 * dv1) * dt * _phi1(2.0 * x1)
        ) / r1

        x2 = dt / (r2 * c2)
        v2_inf = i_const_a * r2
        dv2 = v2 - v2_inf
        v2_new = v2 + dv2 * math.expm1(-x2)
        e_r2_j = (
            (v2_inf * v2_inf) * dt
            + 2.0 * v2_inf * dv2 * dt * _phi1(x2)
            + (dv2 * dv2) * dt * _phi1(2.0 * x2)
        ) / r2

        if use_hyst and abs(i_const_a) > 0.0:
            xh = ((spec.hysteresis_rate_gamma * abs(i_const_a)) / q_coulomb) * dt
            target = -1.0 if i_const_a > 0.0 else 1.0
            h_new = min(max(h_val + (h_val - target) * math.expm1(-xh), -1.0), 1.0)
        else:
            h_new = h_val if ecm._enable_hysteresis else 0.0

        q_irrev = (i_const_a * i_const_a) * r0 + (e_r1_j + e_r2_j) / dt
        dudt_mid = float(ecm.get_entropic_dudt(soc_mid)) if use_ent else 0.0
        q_tot = q_irrev - i_const_a * tc_eval * dudt_mid

        tc_new, ts_new = th._propagate_raw(tc_k, ts_k, q_tot, dt)
        tc_clamped = min(max(tc_new, 253.15), T_MAX_K)
        r0_end, _, _, _, _ = ecm.get_impedances(soc_new, tc_clamped)
        u_end = float(ecm.get_ocv(soc_new, tc_clamped))
        vh_end = (spec.hysteresis_max_v * h_new) if ecm._enable_hysteresis else 0.0
        v_end = u_end + vh_end - i_const_a * r0_end - v1_new - v2_new

        if v_end < min_v:
            min_v = v_end
        if v_end > max_v:
            max_v = v_end
        if tc_new > max_tc:
            max_tc = tc_new

        soc = soc_new
        v1 = v1_new
        v2 = v2_new
        h_val = h_new
        tc_k = tc_clamped
        ts_k = min(max(ts_new, 253.15), T_MAX_K)
    return (min_v, max_v, max_tc, v_end)


def compute_sop_limits(
    bess: CoupledTwoNodeBESS,
    state: CoupledTwoNodeState,
    *,
    horizon_s: float = 30.0,
    dt_sub: float = 2.0,
    soc_min: float = 0.05,
    soc_max: float = 0.98,
    v_min_v: float | None = None,
    v_max_v: float | None = None,
    t_core_max_c: float = 48.0,
    t_charge_min_c: float = 0.0,
    t_charge_taper_c: float = 15.0,
    max_discharge_c_rate: float = 4.0,
    max_charge_c_rate: float = 2.5,
) -> SOPResult:
    """Compute horizon-sustained BMS discharge & charge State-of-Power (SOP) bounds."""
    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("horizon_s must be finite and > 0")
    if not (0.0 <= soc_min < soc_max <= 1.0):
        raise ValueError("Require 0 <= soc_min < soc_max <= 1")
    if max_discharge_c_rate <= 0.0 or max_charge_c_rate <= 0.0:
        raise ValueError("C-rate limits must be > 0")

    spec = bess.spec
    vmin = spec.v_min_v if v_min_v is None else float(v_min_v)
    vmax = spec.v_max_v if v_max_v is None else float(v_max_v)
    if vmin >= vmax:
        raise ValueError("v_min_v must be < v_max_v")
    t_max_k = check_temperature_k(t_core_max_c + 273.15)

    n_steps = max(1, int(math.ceil(horizon_s / max(0.25, dt_sub))))
    cap_ah = spec.nominal_capacity_ah
    q_coulomb = cap_ah * 3600.0

    # --- 1. DISCHARGE SOP (I >= 0) ---
    i_dis_hw = max_discharge_c_rate * cap_ah
    i_dis_soc = max(0.0, (state.soc - soc_min) * q_coulomb / horizon_s)

    # Voltage limit I_dis_v in [0, i_dis_hw * 1.5]
    search_top = i_dis_hw * 1.25
    if _rollout_horizon_extremes(bess, state, 0.0, horizon_s, n_steps)[0] < vmin:
        i_dis_v = 0.0
    elif _rollout_horizon_extremes(bess, state, search_top, horizon_s, n_steps)[0] >= vmin:
        i_dis_v = search_top
    else:
        lo, hi = 0.0, search_top
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if _rollout_horizon_extremes(bess, state, mid, horizon_s, n_steps)[0] >= vmin:
                lo = mid
            else:
                hi = mid
        i_dis_v = lo

    # Core thermal limit I_dis_th
    if state.t_core_k >= t_max_k or _rollout_horizon_extremes(bess, state, 0.0, horizon_s, n_steps)[2] > t_max_k:
        i_dis_th = 0.0
    elif _rollout_horizon_extremes(bess, state, search_top, horizon_s, n_steps)[2] <= t_max_k:
        i_dis_th = search_top
    else:
        lo, hi = 0.0, search_top
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if _rollout_horizon_extremes(bess, state, mid, horizon_s, n_steps)[2] <= t_max_k:
                lo = mid
            else:
                hi = mid
        i_dis_th = lo

    dis_map: dict[SOPBottleneck, float] = {
        "current": i_dis_hw,
        "soc": i_dis_soc,
        "voltage": i_dis_v,
        "thermal_core": i_dis_th,
    }
    dis_bottleneck: SOPBottleneck = min(dis_map, key=lambda k: dis_map[k])
    i_dis_lim = dis_map[dis_bottleneck]
    _, _, tc_dis_end_k, v_dis_end = _rollout_horizon_extremes(bess, state, i_dis_lim, horizon_s, n_steps)
    p_dis_max_w = max(0.0, i_dis_lim * v_dis_end)

    # --- 2. CHARGE SOP (I = -I_mag <= 0) ---
    tc_c = state.t_core_c
    if tc_c <= t_charge_min_c:
        i_chg_hw = 0.0
    elif tc_c < t_charge_taper_c:
        frac = (tc_c - t_charge_min_c) / max(1e-6, t_charge_taper_c - t_charge_min_c)
        i_chg_hw = cap_ah * (0.15 + frac * (max_charge_c_rate - 0.15))
    else:
        i_chg_hw = max_charge_c_rate * cap_ah

    i_chg_soc = max(0.0, (soc_max - state.soc) * q_coulomb / horizon_s)
    chg_top = max(i_chg_hw * 1.25, 0.5 * cap_ah)

    if _rollout_horizon_extremes(bess, state, 0.0, horizon_s, n_steps)[1] > vmax:
        i_chg_v = 0.0
    elif _rollout_horizon_extremes(bess, state, -chg_top, horizon_s, n_steps)[1] <= vmax:
        i_chg_v = chg_top
    else:
        lo, hi = 0.0, chg_top
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if _rollout_horizon_extremes(bess, state, -mid, horizon_s, n_steps)[1] <= vmax:
                lo = mid
            else:
                hi = mid
        i_chg_v = lo

    if state.t_core_k >= t_max_k or _rollout_horizon_extremes(bess, state, 0.0, horizon_s, n_steps)[2] > t_max_k:
        i_chg_th = 0.0
    elif _rollout_horizon_extremes(bess, state, -chg_top, horizon_s, n_steps)[2] <= t_max_k:
        i_chg_th = chg_top
    else:
        lo, hi = 0.0, chg_top
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if _rollout_horizon_extremes(bess, state, -mid, horizon_s, n_steps)[2] <= t_max_k:
                lo = mid
            else:
                hi = mid
        i_chg_th = lo

    chg_map: dict[SOPBottleneck, float] = {
        "current": i_chg_hw,
        "soc": i_chg_soc,
        "voltage": i_chg_v,
        "thermal_core": i_chg_th,
    }
    chg_bottleneck: SOPBottleneck = min(chg_map, key=lambda k: chg_map[k])
    i_chg_lim = chg_map[chg_bottleneck]
    _, _, tc_chg_end_k, v_chg_end = _rollout_horizon_extremes(bess, state, -i_chg_lim, horizon_s, n_steps)
    p_chg_max_w = max(0.0, i_chg_lim * v_chg_end)

    return SOPResult(
        horizon_s=horizon_s,
        i_dis_max_a=i_dis_lim,
        i_chg_max_a=i_chg_lim,
        p_dis_max_w=p_dis_max_w,
        p_chg_max_w=p_chg_max_w,
        v_dis_end_v=v_dis_end,
        v_chg_end_v=v_chg_end,
        t_core_dis_end_c=tc_dis_end_k - 273.15,
        t_core_chg_end_c=tc_chg_end_k - 273.15,
        dis_bottleneck=dis_bottleneck,
        chg_bottleneck=chg_bottleneck,
        dis_limits_by_constraint_a=dis_map,
        chg_limits_by_constraint_a=chg_map,
    )


@dataclass(frozen=True)
class ThermalPulseRun:
    """High-rate pulsed dispatch comparing 2-node (T_core, T_surf) vs lumped T_cell."""

    time_s: np.ndarray
    current_a: np.ndarray
    soc: np.ndarray
    t_core_c: np.ndarray
    t_surf_c: np.ndarray
    t_lumped_c: np.ndarray
    q_total_w: np.ndarray
    peak_core_surf_delta_c: float
    peak_core_vs_lumped_delta_c: float


@dataclass(frozen=True)
class ThermalSOPBenchmarkResult:
    """Complete Module 4 benchmark bundle for Figure 4 and quantitative reporting."""

    pulse_2node: ThermalPulseRun
    nmc_cccv: CCCVChargeResult
    lfp_cccv: CCCVChargeResult
    soc_grid: np.ndarray
    horizons_s: tuple[float, ...]
    p_dis_by_horizon_w: dict[float, np.ndarray]
    p_chg_by_horizon_w: dict[float, np.ndarray]
    temp_grid_c: np.ndarray
    p_dis_vs_temp_w: np.ndarray
    p_chg_vs_temp_w: np.ndarray
    dis_bottleneck_vs_temp: tuple[SOPBottleneck, ...]


def run_thermal_sop_benchmark() -> ThermalSOPBenchmarkResult:
    """Execute reproducible 2-node thermal gradient, CC-CV taper & SOP BMS benchmark."""
    # 1. High-rate bidirectional 3C pulse train (NMC811, active liquid/forced-air h_sa = 1.1 W/K)
    # demonstrating strong radial T_core - T_surf divergence vs a single lumped node.
    dt = 2.0
    n_pulse = 900  # 1800 s
    time_s = np.arange(n_pulse, dtype=float) * dt
    bess_pulse = CoupledTwoNodeBESS(NMC811_LG_M50_5AH, t_amb_k=298.15, h_sa_w_k=1.10)
    lumped_ecm = TwoRCBatteryModel(NMC811_LG_M50_5AH, t_amb_k=298.15, h_a_w_k=1.10)

    st_2n = CoupledTwoNodeState(soc=0.85, t_core_k=298.15, t_surf_k=298.15)
    st_1n = TwoRCState(soc=0.85, t_cell_k=298.15)
    cap = NMC811_LG_M50_5AH.nominal_capacity_ah

    i_arr = np.empty(n_pulse, dtype=float)
    soc_arr = np.empty(n_pulse, dtype=float)
    tc_arr = np.empty(n_pulse, dtype=float)
    ts_arr = np.empty(n_pulse, dtype=float)
    tl_arr = np.empty(n_pulse, dtype=float)
    q_arr = np.empty(n_pulse, dtype=float)

    for k, t in enumerate(time_s):
        phase = int(t // 180.0) % 3
        if phase == 0:
            i_app = 3.0 * cap  # 3C discharge (15 A)
        elif phase == 1:
            i_app = -2.2 * cap  # 2.2C regen charge (-11 A)
        else:
            i_app = 0.0  # 180 s relaxation
        r2 = bess_pulse.step(st_2n, i_app, dt)
        r1 = lumped_ecm.step(st_1n, i_app, dt)
        st_2n = r2.state
        st_1n = r1.state
        i_arr[k] = i_app
        soc_arr[k] = st_2n.soc
        tc_arr[k] = st_2n.t_core_c
        ts_arr[k] = st_2n.t_surf_c
        tl_arr[k] = st_1n.t_cell_k - 273.15
        q_arr[k] = r2.q_total_w

    pulse_run = ThermalPulseRun(
        time_s=time_s,
        current_a=i_arr,
        soc=soc_arr,
        t_core_c=tc_arr,
        t_surf_c=ts_arr,
        t_lumped_c=tl_arr,
        q_total_w=q_arr,
        peak_core_surf_delta_c=float(np.max(tc_arr - ts_arr)),
        peak_core_vs_lumped_delta_c=float(np.max(tc_arr - tl_arr)),
    )

    # 2. 2C CC-CV fast-charging comparison on NMC811 vs LFP (T_amb = 25 degC, h_sa = 0.85 W/K)
    nmc_charger = CoupledTwoNodeBESS(NMC811_LG_M50_5AH, t_amb_k=298.15, h_sa_w_k=0.85)
    lfp_charger = CoupledTwoNodeBESS(LFP_A123_ANR26650_2P3AH, t_amb_k=298.15, h_sa_w_k=0.85)
    nmc_cccv = simulate_cc_cv_charge(
        nmc_charger,
        initial_soc=0.10,
        cc_c_rate=2.0,
        cutoff_c_rate=0.05,
        t_core_max_c=45.0,
        dt=4.0,
    )
    lfp_cccv = simulate_cc_cv_charge(
        lfp_charger,
        initial_soc=0.10,
        cc_c_rate=2.0,
        cutoff_c_rate=0.05,
        t_core_max_c=45.0,
        dt=4.0,
    )

    # 3. Receding-horizon SOP envelope across SoC (10 s, 30 s, 120 s) on NMC811 at 25 degC
    bess_sop = CoupledTwoNodeBESS(NMC811_LG_M50_5AH, t_amb_k=298.15, h_sa_w_k=0.85)
    soc_grid = np.linspace(0.05, 0.98, 32)
    horizons = (10.0, 30.0, 120.0)
    p_dis_by_h: dict[float, np.ndarray] = {}
    p_chg_by_h: dict[float, np.ndarray] = {}
    for h_s in horizons:
        p_d = np.empty_like(soc_grid)
        p_c = np.empty_like(soc_grid)
        for idx, s_val in enumerate(soc_grid):
            st_s = CoupledTwoNodeState(soc=float(s_val), t_core_k=298.15, t_surf_k=298.15)
            sop = compute_sop_limits(bess_sop, st_s, horizon_s=h_s, dt_sub=2.0)
            p_d[idx] = sop.p_dis_max_w
            p_c[idx] = sop.p_chg_max_w
        p_dis_by_h[h_s] = p_d
        p_chg_by_h[h_s] = p_c

    # 4. Thermal derating curve across T_0 in [-18, +49] degC (35% SoC, 60 s horizon)
    # Captures all 3 physical discharge regimes: cold Arrhenius voltage sag -> current ceiling -> T_core limit!
    temp_grid_c = np.linspace(-18.0, 49.0, 35)
    p_dis_t = np.empty_like(temp_grid_c)
    p_chg_t = np.empty_like(temp_grid_c)
    bottlenecks: list[SOPBottleneck] = []
    for idx, tc_val in enumerate(temp_grid_c):
        tk = float(tc_val) + 273.15
        bess_t = CoupledTwoNodeBESS(NMC811_LG_M50_5AH, t_amb_k=tk, h_sa_w_k=0.85)
        st_t = CoupledTwoNodeState(soc=0.35, t_core_k=tk, t_surf_k=tk)
        sop_t = compute_sop_limits(bess_t, st_t, horizon_s=60.0, dt_sub=3.0, t_core_max_c=48.0)
        p_dis_t[idx] = sop_t.p_dis_max_w
        p_chg_t[idx] = sop_t.p_chg_max_w
        bottlenecks.append(sop_t.dis_bottleneck)

    return ThermalSOPBenchmarkResult(
        pulse_2node=pulse_run,
        nmc_cccv=nmc_cccv,
        lfp_cccv=lfp_cccv,
        soc_grid=soc_grid,
        horizons_s=horizons,
        p_dis_by_horizon_w=p_dis_by_h,
        p_chg_by_horizon_w=p_chg_by_h,
        temp_grid_c=temp_grid_c,
        p_dis_vs_temp_w=p_dis_t,
        p_chg_vs_temp_w=p_chg_t,
        dis_bottleneck_vs_temp=tuple(bottlenecks),
    )
