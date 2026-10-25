"""ASTM E1049-85 rainflow cycle counting & semi-empirical SEI + cycle + plating aging engine.

Implements two coupled components for multi-year BESS lifetime & grid-duty assessment:

1. **Exact ASTM E1049-85 Four-Point Rainflow Cycle Extractor** (:func:`extract_turning_points`,
   :func:`extract_rainflow_cycles`):
   Decomposes arbitrary non-monotonic State-of-Charge trajectories ``soc(t)`` (such as PJM
   ``RegD`` frequency-regulation micro-cycles superimposed on daily arbitrage) into closed
   full cycles (``count = 1.0``) and residual half-cycles (``count = 0.5``), preserving the
   exact topological throughput identity ``sum(2 * count_i * dod_i) == sum(|diff(soc)|)`` to
   machine precision when ``min_reversal_dod = 0``.

2. **Path-Consistent Virtual-Age Semi-Empirical Degradation Engine** (:class:`DegradationSpec`,
   :class:`SemiEmpiricalAgingModel`, :func:`simulate_grid_duty_aging`):
   Couples diffusion-limited SEI calendar aging (``t^z_cal``, Schmalstieg et al., J. Power
   Sources 257, 325-334, 2014), Wöhler power-law DoD cycle fatigue (Xu et al., IEEE Trans.
   Power Syst. 33(2), 2248-2259, 2018; Naumann et al., J. Energy Storage 27, 101153, 2020),
   and sub-15 degC anodic lithium-plating acceleration (Waldmann et al., J. Power Sources 262,
   129-135, 2014) with 2-node core temperature ``T_core`` and resistance-growth Joule feedback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from src.bms_thermal_sop import CoupledTwoNodeBESS, CoupledTwoNodeState
from src.cell_chemistry import (
    LFP_A123_ANR26650_2P3AH,
    NMC811_LG_M50_5AH,
    ChemistrySpec,
    get_chemistry_spec,
)
from src.thermal_model import R_GAS, T_REF_K, check_temperature_k

__all__ = [
    "AgingState",
    "AgingStepResult",
    "DegradationBenchmarkResult",
    "DegradationSpec",
    "DutyAgingProjection",
    "DutyCycleSample",
    "GridDutyMode",
    "LFP_AGING_SPEC",
    "NMC811_AGING_SPEC",
    "RainflowCycle",
    "SemiEmpiricalAgingModel",
    "extract_rainflow_cycles",
    "extract_turning_points",
    "generate_grid_duty_current",
    "get_degradation_spec",
    "run_degradation_benchmark",
    "simulate_duty_cycle_block",
    "simulate_grid_duty_aging",
]

GridDutyMode = Literal["regd_freq_reg", "deep_arbitrage"]


@dataclass(frozen=True)
class RainflowCycle:
    """Single closed full cycle (count=1.0) or residual half-cycle (count=0.5)."""

    dod: float
    mean_soc: float
    soc_min: float
    soc_max: float
    count: float
    t_start_s: float
    t_end_s: float
    duration_s: float
    avg_c_rate: float


def extract_turning_points(
    soc: np.ndarray,
    time_s: np.ndarray | None = None,
    *,
    min_reversal_dod: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract alternating local extrema (reversals) from a 1D SoC series."""
    soc_arr = np.asarray(soc, dtype=float)
    if soc_arr.ndim != 1 or len(soc_arr) < 2:
        raise ValueError("soc must be a 1D array with at least 2 samples")
    if not np.all(np.isfinite(soc_arr)):
        raise ValueError("soc samples must be finite")
    if min_reversal_dod < 0.0 or not math.isfinite(min_reversal_dod):
        raise ValueError("min_reversal_dod must be finite and >= 0")

    if time_s is None:
        t_arr = np.arange(len(soc_arr), dtype=float)
    else:
        t_arr = np.asarray(time_s, dtype=float)
        if t_arr.shape != soc_arr.shape or not np.all(np.isfinite(t_arr)):
            raise ValueError("time_s must match soc shape and be finite")
        if np.any(np.diff(t_arr) < 0.0):
            raise ValueError("time_s must be non-decreasing")

    # 1. Collapse flat plateaus while keeping first timestamp of each distinct level
    vals: list[float] = [float(soc_arr[0])]
    times: list[float] = [float(t_arr[0])]
    for k in range(1, len(soc_arr)):
        v = float(soc_arr[k])
        if abs(v - vals[-1]) > 0.0:
            vals.append(v)
            times.append(float(t_arr[k]))
    if len(vals) == 1:
        return np.asarray(vals, dtype=float), np.asarray(times, dtype=float)

    # 2. Keep strict alternating local extrema
    tp_v: list[float] = [vals[0]]
    tp_t: list[float] = [times[0]]
    for k in range(1, len(vals) - 1):
        d_prev = vals[k] - tp_v[-1]
        d_next = vals[k + 1] - vals[k]
        if d_prev * d_next < 0.0:
            tp_v.append(vals[k])
            tp_t.append(times[k])
    tp_v.append(vals[-1])
    tp_t.append(times[-1])

    # 3. Optional deadband hysteresis filter for micro-noise reversals
    if min_reversal_dod > 0.0 and len(tp_v) > 2:
        filt_v: list[float] = []
        filt_t: list[float] = []
        for v, t in zip(tp_v, tp_t, strict=True):
            filt_v.append(v)
            filt_t.append(t)
            while len(filt_v) >= 3 and abs(filt_v[-1] - filt_v[-2]) < min_reversal_dod:
                filt_v.pop(-2)
                filt_t.pop(-2)
                if len(filt_v) >= 3:
                    d1 = filt_v[-2] - filt_v[-3]
                    d2 = filt_v[-1] - filt_v[-2]
                    if d1 * d2 >= 0.0:
                        filt_v.pop(-2)
                        filt_t.pop(-2)
        tp_v, tp_t = filt_v, filt_t

    return np.asarray(tp_v, dtype=float), np.asarray(tp_t, dtype=float)


def extract_rainflow_cycles(
    soc: np.ndarray,
    time_s: np.ndarray | None = None,
    *,
    min_reversal_dod: float = 0.0,
) -> list[RainflowCycle]:
    """Decompose an SoC trajectory into ASTM E1049-85 full (1.0) and half (0.5) cycles."""
    tp_soc, tp_t = extract_turning_points(soc, time_s, min_reversal_dod=min_reversal_dod)
    if len(tp_soc) < 2:
        return []

    stack: list[tuple[float, float]] = []
    closed_cycles: list[RainflowCycle] = []

    def _make_cycle(s_a: float, t_a: float, s_b: float, t_b: float, count: float) -> RainflowCycle | None:
        dod = abs(s_b - s_a)
        if dod <= 0.0:
            return None
        s_min = min(s_a, s_b)
        s_max = max(s_a, s_b)
        t0 = min(t_a, t_b)
        t1 = max(t_a, t_b)
        dur = max(0.0, t1 - t0)
        c_rate = (dod * 3600.0 / dur) if dur > 1e-9 else 0.0
        return RainflowCycle(
            dod=float(dod),
            mean_soc=float(0.5 * (s_a + s_b)),
            soc_min=float(s_min),
            soc_max=float(s_max),
            count=float(count),
            t_start_s=float(t0),
            t_end_s=float(t1),
            duration_s=float(dur),
            avg_c_rate=float(c_rate),
        )

    for s_val, t_val in zip(tp_soc, tp_t, strict=True):
        stack.append((float(s_val), float(t_val)))
        while len(stack) >= 4:
            s0, _ = stack[-4]
            s1, t1 = stack[-3]
            s2, t2 = stack[-2]
            s3, _ = stack[-1]
            r1 = abs(s1 - s0)
            r2 = abs(s2 - s1)
            r3 = abs(s3 - s2)
            if r2 <= r1 and r2 <= r3:
                cyc = _make_cycle(s1, t1, s2, t2, 1.0)
                if cyc is not None:
                    closed_cycles.append(cyc)
                del stack[-3:-1]
            else:
                break

    residual_cycles: list[RainflowCycle] = []
    for idx in range(len(stack) - 1):
        s_a, t_a = stack[idx]
        s_b, t_b = stack[idx + 1]
        cyc = _make_cycle(s_a, t_a, s_b, t_b, 0.5)
        if cyc is not None:
            residual_cycles.append(cyc)

    return closed_cycles + residual_cycles


@dataclass(frozen=True)
class DegradationSpec:
    """Chemistry-specific semi-empirical calendar, Wöhler cycle & plating aging parameters."""

    key: str
    name: str
    k_cal_ref_per_day_z: float
    z_cal: float
    ea_cal_j_mol: float
    alpha_soc_cal: float
    soc_ref: float
    k_cyc_ref_per_efc_z: float
    z_cyc: float
    beta_dod: float
    dod_ref: float
    ea_cyc_j_mol: float
    gamma_crate: float
    crate_ref: float
    k_plat_ref_per_efc_z: float
    ea_plat_j_mol: float
    t_plat_ref_k: float
    gamma_plat_crate: float
    alpha_plat_soc: float
    r_cal_gain: float
    r_cyc_gain: float
    r_plat_gain: float
    eol_soh_threshold: float = 0.80

    def __post_init__(self) -> None:
        if self.k_cal_ref_per_day_z <= 0.0 or not (0.2 <= self.z_cal <= 1.0):
            raise ValueError("Invalid calendar aging parameters")
        if self.k_cyc_ref_per_efc_z <= 0.0 or not (0.3 <= self.z_cyc <= 1.0):
            raise ValueError("Invalid cycle aging parameters")
        if self.beta_dod < 1.0:
            raise ValueError("beta_dod must be >= 1.0 for convex/linear Wöhler fatigue")
        if not (0.50 <= self.eol_soh_threshold < 1.0):
            raise ValueError("eol_soh_threshold must lie in [0.50, 1.0)")


NMC811_AGING_SPEC = DegradationSpec(
    key=NMC811_LG_M50_5AH.key,
    name="LG Chem INR21700-M50 NMC811/SiOx-Graphite Aging Spec",
    k_cal_ref_per_day_z=2.66e-3,
    z_cal=0.50,
    ea_cal_j_mol=32_000.0,
    alpha_soc_cal=1.35,
    soc_ref=0.50,
    k_cyc_ref_per_efc_z=1.94e-4,
    z_cyc=0.82,
    beta_dod=1.68,
    dod_ref=0.80,
    ea_cyc_j_mol=24_500.0,
    gamma_crate=0.38,
    crate_ref=0.50,
    k_plat_ref_per_efc_z=5.15e-5,
    ea_plat_j_mol=62_000.0,
    t_plat_ref_k=288.15,  # 15.0 degC crossover anchor
    gamma_plat_crate=1.45,
    alpha_plat_soc=1.20,
    r_cal_gain=1.90,
    r_cyc_gain=2.75,
    r_plat_gain=3.60,
    eol_soh_threshold=0.80,
)

LFP_AGING_SPEC = DegradationSpec(
    key=LFP_A123_ANR26650_2P3AH.key,
    name="A123 ANR26650M1-B Olivine LFP/Graphite Aging Spec",
    k_cal_ref_per_day_z=1.52e-3,
    z_cal=0.50,
    ea_cal_j_mol=27_500.0,
    alpha_soc_cal=0.75,
    soc_ref=0.50,
    k_cyc_ref_per_efc_z=9.92e-5,
    z_cyc=0.84,
    beta_dod=1.36,
    dod_ref=0.80,
    ea_cyc_j_mol=20_000.0,
    gamma_crate=0.20,
    crate_ref=0.50,
    k_plat_ref_per_efc_z=1.60e-5,
    ea_plat_j_mol=50_000.0,
    t_plat_ref_k=283.15,  # 10.0 degC crossover anchor
    gamma_plat_crate=1.25,
    alpha_plat_soc=0.90,
    r_cal_gain=1.10,
    r_cyc_gain=1.45,
    r_plat_gain=2.10,
    eol_soh_threshold=0.80,
)

_AGING_REGISTRY: dict[str, DegradationSpec] = {
    NMC811_LG_M50_5AH.key: NMC811_AGING_SPEC,
    "nmc811": NMC811_AGING_SPEC,
    LFP_A123_ANR26650_2P3AH.key: LFP_AGING_SPEC,
    "lfp": LFP_AGING_SPEC,
}


def get_degradation_spec(chemistry: ChemistrySpec | str) -> DegradationSpec:
    """Look up the :class:`DegradationSpec` for a registered chemistry key or spec."""
    key = chemistry.strip().lower() if isinstance(chemistry, str) else chemistry.key
    if key not in _AGING_REGISTRY:
        valid = ", ".join(sorted(_AGING_REGISTRY))
        raise ValueError(f"Unknown degradation spec key {key!r}. Available: {valid}")
    return _AGING_REGISTRY[key]


@dataclass(frozen=True)
class AgingState:
    """Path-consistent virtual-age semigroup state variables."""

    elapsed_days: float = 0.0
    equivalent_full_cycles: float = 0.0
    u_cal: float = 0.0
    u_cyc: float = 0.0
    u_plat: float = 0.0
    q_loss_cal: float = 0.0
    q_loss_cyc: float = 0.0
    q_loss_plating_cum: float = 0.0

    @property
    def q_loss_total(self) -> float:
        return min(0.80, self.q_loss_cal + self.q_loss_cyc)

    @property
    def soh_capacity(self) -> float:
        return max(0.20, 1.0 - self.q_loss_total)


@dataclass(frozen=True)
class AgingStepResult:
    """Result of advancing :class:`SemiEmpiricalAgingModel` over a macro-block."""

    state: AgingState
    soh_capacity: float
    delta_r_rel: float
    resistance_scale: float


class SemiEmpiricalAgingModel:
    """Virtual-age SEI calendar + rainflow Wöhler cycle + Waldmann plating aging solver."""

    def __init__(self, spec: DegradationSpec | ChemistrySpec | str = NMC811_AGING_SPEC) -> None:
        if isinstance(spec, DegradationSpec):
            self._spec = spec
        else:
            self._spec = get_degradation_spec(spec)

    @property
    def spec(self) -> DegradationSpec:
        return self._spec

    def calendar_rate_factor(self, mean_soc: float, t_core_k: float) -> float:
        """Compute SEI calendar rate factor k_cal(mean_soc, T_core) [day^-z_cal]."""
        tk = check_temperature_k(t_core_k)
        s_clamped = float(np.clip(mean_soc, 0.0, 1.0))
        sp = self._spec
        f_soc = math.exp(sp.alpha_soc_cal * (s_clamped - sp.soc_ref))
        f_arr = math.exp(-(sp.ea_cal_j_mol / R_GAS) * (1.0 / tk - 1.0 / T_REF_K))
        return sp.k_cal_ref_per_day_z * f_soc * f_arr

    def cycle_stress_factors(self, cycle: RainflowCycle, t_core_k: float) -> tuple[float, float]:
        """Return (s_sei, s_plating) per-EFC severity factors for a single rainflow cycle."""
        tk = check_temperature_k(t_core_k)
        sp = self._spec
        dod_eff = max(1e-6, min(1.0, cycle.dod))
        c_eff = max(0.05, cycle.avg_c_rate)

        f_dod = (dod_eff / sp.dod_ref) ** (sp.beta_dod - 1.0)
        f_crate = math.exp(sp.gamma_crate * (c_eff - sp.crate_ref))
        f_arr_cyc = math.exp(-(sp.ea_cyc_j_mol / R_GAS) * (1.0 / tk - 1.0 / T_REF_K))
        s_sei = sp.k_cyc_ref_per_efc_z * f_dod * f_crate * f_arr_cyc

        f_plat_c = (c_eff / sp.crate_ref) ** sp.gamma_plat_crate
        f_plat_soc = math.exp(sp.alpha_plat_soc * (cycle.soc_max - 0.50))
        f_plat_t = math.exp(+(sp.ea_plat_j_mol / R_GAS) * (1.0 / tk - 1.0 / sp.t_plat_ref_k))
        s_plat = sp.k_plat_ref_per_efc_z * f_plat_c * f_plat_soc * f_plat_t
        return s_sei, s_plat

    def delta_r_relative(self, state: AgingState) -> float:
        """Relative resistance growth Delta R / R_BOL from SEI film & plating damage."""
        sp = self._spec
        q_cyc_non_plat = max(0.0, state.q_loss_cyc - state.q_loss_plating_cum)
        return (
            sp.r_cal_gain * state.q_loss_cal
            + sp.r_cyc_gain * q_cyc_non_plat
            + sp.r_plat_gain * state.q_loss_plating_cum
        )

    def resistance_scale(self, state: AgingState) -> float:
        """Multiplicative impedance factor R(t) / R_BOL >= 1.0."""
        return 1.0 + self.delta_r_relative(state)

    def step_block(
        self,
        state: AgingState,
        *,
        duration_days: float,
        mean_soc: float,
        t_core_k: float,
        cycles: tuple[RainflowCycle, ...] | list[RainflowCycle] = (),
        repeat_cycles: float = 1.0,
    ) -> AgingStepResult:
        """Advance virtual-age calendar, cycle, and plating states over a macro-step."""
        if duration_days < 0.0 or not math.isfinite(duration_days):
            raise ValueError("duration_days must be finite and >= 0")
        if repeat_cycles < 0.0 or not math.isfinite(repeat_cycles):
            raise ValueError("repeat_cycles must be finite and >= 0")
        tk = check_temperature_k(t_core_k)
        sp = self._spec

        # 1. Calendar virtual age update: exact semigroup for t^z_cal
        u_cal_new = state.u_cal
        if duration_days > 0.0:
            k_cal = self.calendar_rate_factor(mean_soc, tk)
            u_cal_new += (k_cal ** (1.0 / sp.z_cal)) * duration_days
        q_cal_new = u_cal_new**sp.z_cal if u_cal_new > 0.0 else 0.0

        # 2. Cycle + plating virtual age update: exact semigroup for EFC^z_cyc
        u_cyc_new = state.u_cyc
        u_plat_new = state.u_plat
        d_efc_total = 0.0
        if repeat_cycles > 0.0 and len(cycles) > 0:
            inv_z = 1.0 / sp.z_cyc
            for cyc in cycles:
                d_efc = repeat_cycles * cyc.count * cyc.dod
                if d_efc <= 0.0:
                    continue
                d_efc_total += d_efc
                s_sei, s_plat = self.cycle_stress_factors(cyc, tk)
                u_cyc_new += ((s_sei + s_plat) ** inv_z) * d_efc
                u_plat_new += (s_plat**inv_z) * d_efc

        q_cyc_new = u_cyc_new**sp.z_cyc if u_cyc_new > 0.0 else 0.0
        q_plat_new = min(q_cyc_new, u_plat_new**sp.z_cyc if u_plat_new > 0.0 else 0.0)

        next_state = AgingState(
            elapsed_days=state.elapsed_days + duration_days,
            equivalent_full_cycles=state.equivalent_full_cycles + d_efc_total,
            u_cal=u_cal_new,
            u_cyc=u_cyc_new,
            u_plat=u_plat_new,
            q_loss_cal=q_cal_new,
            q_loss_cyc=q_cyc_new,
            q_loss_plating_cum=q_plat_new,
        )
        dr_rel = self.delta_r_relative(next_state)
        return AgingStepResult(
            state=next_state,
            soh_capacity=next_state.soh_capacity,
            delta_r_rel=dr_rel,
            resistance_scale=1.0 + dr_rel,
        )


@dataclass(frozen=True)
class DutyCycleSample:
    """4-hour coupled 2-node electro-thermal & rainflow dispatch sample."""

    mode: GridDutyMode
    time_s: np.ndarray
    current_a: np.ndarray
    soc: np.ndarray
    t_core_c: np.ndarray
    turning_soc: np.ndarray
    turning_time_s: np.ndarray
    cycles: tuple[RainflowCycle, ...]
    mean_soc: float
    rms_t_core_k: float
    block_efc: float


def generate_grid_duty_current(
    mode: GridDutyMode,
    *,
    capacity_ah: float,
    duration_s: float = 14_400.0,
    dt: float = 10.0,
    seed: int = 202605,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize deterministic 4-hour PJM RegD AGC or Deep Energy Arbitrage current [A].

    Uses an isolated ``np.random.default_rng(seed)`` instance so global NumPy RNG state is
    never mutated. Positive current = discharge, negative current = charge.
    """
    if capacity_ah <= 0.0 or duration_s <= 0.0 or dt <= 0.0:
        raise ValueError("capacity_ah, duration_s, and dt must be positive")
    n = int(round(duration_s / dt)) + 1
    t = np.arange(n, dtype=float) * dt

    if mode == "deep_arbitrage":
        i_c = np.zeros_like(t)
        c_amp = 0.923 * capacity_ah
        m_dis = (t >= 360.0) & (t < 3480.0)
        m_chg = (t >= 7380.0) & (t < 10500.0)
        i_c[m_dis] = c_amp
        i_c[m_chg] = -c_amp
        return t, i_c

    if mode == "regd_freq_reg":
        rng = np.random.default_rng(seed)
        w1 = 2.0 * math.pi / 3600.0
        w2 = 2.0 * math.pi / 680.0
        w3 = 2.0 * math.pi / 210.0
        agc_c = (
            0.42 * np.sin(w1 * t)
            + 0.58 * np.sin(w2 * t + 0.4)
            + 0.34 * np.cos(w3 * t - 0.7)
            + 0.08 * rng.standard_normal(n)
        )
        agc_c -= np.mean(agc_c)
        return t, agc_c * capacity_ah

    raise ValueError(f"Unknown GridDutyMode {mode!r}")


def simulate_duty_cycle_block(
    chemistry: ChemistrySpec | str,
    mode: GridDutyMode,
    *,
    t_amb_c: float = 25.0,
    soh_capacity: float = 1.0,
    resistance_scale: float = 1.0,
    duration_s: float = 14_400.0,
    dt: float = 10.0,
) -> DutyCycleSample:
    """Simulate a 4-hour grid duty block on :class:`CoupledTwoNodeBESS` with aged Q & R."""
    base_spec = get_chemistry_spec(chemistry) if isinstance(chemistry, str) else chemistry
    aged_spec = replace(
        base_spec,
        nominal_capacity_ah=base_spec.nominal_capacity_ah * max(0.20, soh_capacity),
        r0_ref_ohm=base_spec.r0_ref_ohm * max(1.0, resistance_scale),
        r1_ref_ohm=base_spec.r1_ref_ohm * max(1.0, resistance_scale),
        r2_ref_ohm=base_spec.r2_ref_ohm * max(1.0, resistance_scale),
    )
    t_amb_k = t_amb_c + 273.15
    bess = CoupledTwoNodeBESS(aged_spec, t_amb_k=t_amb_k)
    t_arr, i_raw = generate_grid_duty_current(
        mode,
        capacity_ah=aged_spec.nominal_capacity_ah,
        duration_s=duration_s,
        dt=dt,
    )

    soc0 = 0.88 if mode == "deep_arbitrage" else 0.52
    st = CoupledTwoNodeState(soc=soc0, t_core_k=t_amb_k, t_surf_k=t_amb_k)
    q_coul = aged_spec.nominal_capacity_ah * 3600.0

    i_applied = np.empty_like(t_arr)
    soc_hist = np.empty_like(t_arr)
    tc_hist = np.empty_like(t_arr)

    for k, i_req in enumerate(i_raw):
        if k == 0:
            i_applied[0] = 0.0
            soc_hist[0] = st.soc
            tc_hist[0] = st.t_core_c
            continue
        i_max_dis = max(0.0, (st.soc - 0.08) * q_coul / dt)
        i_max_chg = max(0.0, (0.90 - st.soc) * q_coul / dt)
        i_act = float(np.clip(i_req, -i_max_chg, i_max_dis))
        res = bess.step(st, i_act, dt)
        st = res.state
        i_applied[k] = i_act
        soc_hist[k] = st.soc
        tc_hist[k] = st.t_core_c

    tp_soc, tp_t = extract_turning_points(soc_hist, t_arr, min_reversal_dod=1e-4)
    cycles = extract_rainflow_cycles(soc_hist, t_arr, min_reversal_dod=1e-4)
    efc = sum(c.count * c.dod for c in cycles)
    rms_tc_k = float(np.mean(tc_hist) + 273.15)

    return DutyCycleSample(
        mode=mode,
        time_s=t_arr,
        current_a=i_applied,
        soc=soc_hist,
        t_core_c=tc_hist,
        turning_soc=tp_soc,
        turning_time_s=tp_t,
        cycles=tuple(cycles),
        mean_soc=float(np.mean(soc_hist)),
        rms_t_core_k=rms_tc_k,
        block_efc=float(efc),
    )


@dataclass(frozen=True)
class DutyAgingProjection:
    """Multi-year State-of-Health & resistance-growth trajectory for a grid duty mode."""

    chemistry: ChemistrySpec
    mode: GridDutyMode
    years: np.ndarray
    efc: np.ndarray
    soh_pct: np.ndarray
    q_loss_cal_pct: np.ndarray
    q_loss_cyc_pct: np.ndarray
    delta_r_pct: np.ndarray
    mean_t_core_c: np.ndarray
    eol_years: float
    eol_efc: float


def simulate_grid_duty_aging(
    chemistry: ChemistrySpec | str,
    mode: GridDutyMode,
    *,
    total_years: float = 8.0,
    step_days: float = 60.0,
    blocks_per_day: float = 2.0,
    t_amb_c: float = 25.0,
) -> DutyAgingProjection:
    """Project multi-year capacity fade & R0 growth with electro-thermal feedback."""
    spec = get_chemistry_spec(chemistry) if isinstance(chemistry, str) else chemistry
    aging = SemiEmpiricalAgingModel(spec)

    n_macro = int(math.ceil((total_years * 365.0) / step_days))
    state = AgingState()

    yr_list = [0.0]
    efc_list = [0.0]
    soh_list = [100.0]
    qcal_list = [0.0]
    qcyc_list = [0.0]
    dr_list = [0.0]

    sample0 = simulate_duty_cycle_block(
        spec, mode, t_amb_c=t_amb_c, soh_capacity=1.0, resistance_scale=1.0
    )
    tc_list = [sample0.rms_t_core_k - 273.15]

    eol_yr = math.inf
    eol_efc = math.inf
    eol_pct = aging.spec.eol_soh_threshold * 100.0
    blk = sample0

    for m in range(n_macro):
        if m > 0 and m % 10 == 0:
            blk = simulate_duty_cycle_block(
                spec,
                mode,
                t_amb_c=t_amb_c,
                soh_capacity=state.soh_capacity,
                resistance_scale=aging.resistance_scale(state),
            )

        prev_soh = state.soh_capacity * 100.0
        prev_yr = state.elapsed_days / 365.0
        prev_efc = state.equivalent_full_cycles

        repeats = blocks_per_day * step_days
        step_res = aging.step_block(
            state,
            duration_days=step_days,
            mean_soc=blk.mean_soc,
            t_core_k=blk.rms_t_core_k,
            cycles=blk.cycles,
            repeat_cycles=repeats,
        )
        state = step_res.state

        cur_yr = state.elapsed_days / 365.0
        cur_soh = step_res.soh_capacity * 100.0
        cur_efc = state.equivalent_full_cycles

        if math.isinf(eol_yr) and cur_soh <= eol_pct and prev_soh > cur_soh:
            frac = (prev_soh - eol_pct) / (prev_soh - cur_soh)
            eol_yr = prev_yr + frac * (cur_yr - prev_yr)
            eol_efc = prev_efc + frac * (cur_efc - prev_efc)

        yr_list.append(cur_yr)
        efc_list.append(cur_efc)
        soh_list.append(cur_soh)
        qcal_list.append(state.q_loss_cal * 100.0)
        qcyc_list.append(state.q_loss_cyc * 100.0)
        dr_list.append(step_res.delta_r_rel * 100.0)
        tc_list.append(blk.rms_t_core_k - 273.15)

    return DutyAgingProjection(
        chemistry=spec,
        mode=mode,
        years=np.asarray(yr_list, dtype=float),
        efc=np.asarray(efc_list, dtype=float),
        soh_pct=np.asarray(soh_list, dtype=float),
        q_loss_cal_pct=np.asarray(qcal_list, dtype=float),
        q_loss_cyc_pct=np.asarray(qcyc_list, dtype=float),
        delta_r_pct=np.asarray(dr_list, dtype=float),
        mean_t_core_c=np.asarray(tc_list, dtype=float),
        eol_years=float(eol_yr),
        eol_efc=float(eol_efc),
    )


@dataclass(frozen=True)
class DegradationBenchmarkResult:
    """Complete Module 5 benchmark bundle for Figure 5 and quantitative reporting."""

    regd_sample_nmc: DutyCycleSample
    arb_sample_nmc: DutyCycleSample
    nmc_regd: DutyAgingProjection
    nmc_arb: DutyAgingProjection
    lfp_regd: DutyAgingProjection
    lfp_arb: DutyAgingProjection
    waldmann_temps_c: np.ndarray
    nmc_fade_500efc_pct: np.ndarray
    nmc_plating_500efc_pct: np.ndarray
    nmc_sei_500efc_pct: np.ndarray
    lfp_fade_500efc_pct: np.ndarray
    optimal_temp_nmc_c: float


def run_degradation_benchmark() -> DegradationBenchmarkResult:
    """Execute the canonical 12-year matched-throughput degradation & Waldmann V-curve suite."""
    regd_sample = simulate_duty_cycle_block(
        NMC811_LG_M50_5AH, "regd_freq_reg", duration_s=14_400.0, dt=10.0
    )
    arb_sample = simulate_duty_cycle_block(
        NMC811_LG_M50_5AH, "deep_arbitrage", duration_s=14_400.0, dt=10.0
    )

    arb_bpd = 1.05
    regd_bpd = arb_bpd * (arb_sample.block_efc / max(1e-6, regd_sample.block_efc))

    nmc_arb = simulate_grid_duty_aging(
        NMC811_LG_M50_5AH,
        "deep_arbitrage",
        total_years=12.0,
        step_days=60.0,
        blocks_per_day=arb_bpd,
    )
    nmc_regd = simulate_grid_duty_aging(
        NMC811_LG_M50_5AH,
        "regd_freq_reg",
        total_years=12.0,
        step_days=60.0,
        blocks_per_day=regd_bpd,
    )
    lfp_arb = simulate_grid_duty_aging(
        LFP_A123_ANR26650_2P3AH,
        "deep_arbitrage",
        total_years=12.0,
        step_days=60.0,
        blocks_per_day=arb_bpd,
    )
    lfp_regd = simulate_grid_duty_aging(
        LFP_A123_ANR26650_2P3AH,
        "regd_freq_reg",
        total_years=12.0,
        step_days=60.0,
        blocks_per_day=regd_bpd,
    )

    temps_c = np.linspace(-5.0, 50.0, 45)
    nmc_tot = np.empty_like(temps_c)
    nmc_plat = np.empty_like(temps_c)
    nmc_sei = np.empty_like(temps_c)
    lfp_tot = np.empty_like(temps_c)

    ref_cycle = RainflowCycle(
        dod=0.75,
        mean_soc=0.52,
        soc_min=0.145,
        soc_max=0.895,
        count=1.0,
        t_start_s=0.0,
        t_end_s=2700.0,
        duration_s=2700.0,
        avg_c_rate=1.0,
    )
    repeats_500 = 500.0 / 0.75
    nmc_m = SemiEmpiricalAgingModel(NMC811_AGING_SPEC)
    lfp_m = SemiEmpiricalAgingModel(LFP_AGING_SPEC)

    for idx, tc in enumerate(temps_c):
        tk = float(tc) + 273.15
        r_n = nmc_m.step_block(
            AgingState(),
            duration_days=120.0,
            mean_soc=0.52,
            t_core_k=tk,
            cycles=(ref_cycle,),
            repeat_cycles=repeats_500,
        )
        r_l = lfp_m.step_block(
            AgingState(),
            duration_days=120.0,
            mean_soc=0.52,
            t_core_k=tk,
            cycles=(ref_cycle,),
            repeat_cycles=repeats_500,
        )
        tot_n = r_n.state.q_loss_total * 100.0
        plat_n = r_n.state.q_loss_plating_cum * 100.0
        nmc_tot[idx] = tot_n
        nmc_plat[idx] = plat_n
        nmc_sei[idx] = tot_n - plat_n
        lfp_tot[idx] = r_l.state.q_loss_total * 100.0

    opt_t = float(temps_c[int(np.argmin(nmc_tot))])
    return DegradationBenchmarkResult(
        regd_sample_nmc=regd_sample,
        arb_sample_nmc=arb_sample,
        nmc_regd=nmc_regd,
        nmc_arb=nmc_arb,
        lfp_regd=lfp_regd,
        lfp_arb=lfp_arb,
        waldmann_temps_c=temps_c,
        nmc_fade_500efc_pct=nmc_tot,
        nmc_plating_500efc_pct=nmc_plat,
        nmc_sei_500efc_pct=nmc_sei,
        lfp_fade_500efc_pct=lfp_tot,
        optimal_temp_nmc_c=opt_t,
    )
