"""Closed-loop nonlinear State-of-Charge (SoC) estimators for 2RC BESS cells.

Implements three state estimators operating on noisy, bias-corrupted current and
terminal-voltage measurements of a :class:`~src.ecm_2rc.TwoRCBatteryModel`:

1. :class:`CoulombCountingEstimator`, Open-loop ampere-hour integration baseline
   (demonstrates unbounded linear SoC drift under persistent Hall-sensor current bias
   and inability to recover from boot-up SoC initialization offsets).
2. :class:`ExtendedKalmanFilterEstimator` (EKF), Analytical discrete-time Jacobian
   linearization (including off-diagonal SoC-impedance dV_rc/dSoC chain-rule terms and
   exact PCHIP dU_ocv/dSoC measurement slopes) with Bucy-Joseph positive-definite
   covariance updates and Normalized Innovation Squared (NIS) outlier gating.
3. :class:`UnscentedKalmanFilterEstimator` (UKF), Scaled Unscented Transform (Wan &
   van der Merwe, 2000) propagating 2n + 1 = 7 deterministic sigma points through the
   full nonlinear 2RC electro-thermal + PCHIP OCV equations without linearization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cell_chemistry import LFP_A123_ANR26650_2P3AH, NMC811_LG_M50_5AH, ChemistrySpec
from .ecm_2rc import TwoRCBatteryModel, TwoRCState
from .thermal_model import check_temperature_k

__all__ = [
    "ChemistryEstimationRun",
    "CoulombCountingEstimator",
    "EstimatorStepResult",
    "ExtendedKalmanFilterEstimator",
    "SoCEstimationBenchmarkResult",
    "UnscentedKalmanFilterEstimator",
    "run_single_chemistry_estimation",
    "run_soc_estimation_benchmark",
]

_STATE_DIM = 3
_PSD_FLOOR = 1e-12


def _validate_cov_3x3(name: str, mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=float)
    if arr.shape != (_STATE_DIM, _STATE_DIM) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite (3, 3) matrix")
    if not np.allclose(arr, arr.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigvals = np.linalg.eigvalsh(arr)
    if np.min(eigvals) <= 0.0:
        raise ValueError(f"{name} must be strictly positive-definite (min eig={np.min(eigvals):.3e})")
    return 0.5 * (arr + arr.T)


def _enforce_spd_3x3(mat: np.ndarray) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if np.min(eigvals) < _PSD_FLOOR:
        eigvals = np.maximum(eigvals, _PSD_FLOOR)
        sym = (eigvecs * eigvals) @ eigvecs.T
        sym = 0.5 * (sym + sym.T)
    return sym


@dataclass(frozen=True)
class EstimatorStepResult:
    """Diagnostic output for a single recursive estimation step."""

    soc_hat: float
    v_rc1_hat_v: float
    v_rc2_hat_v: float
    hysteresis_hat: float
    t_cell_k: float
    v_pred_v: float
    innovation_v: float
    innovation_var_v2: float
    nis: float
    kalman_gain_soc: float
    soc_std: float
    docv_dsoc_v: float
    covariance: np.ndarray
    innovation_gated: bool = False


@dataclass(frozen=True)
class ChemistryEstimationRun:
    """Complete truth + noisy sensor + 3-estimator comparison on one chemistry."""

    spec: ChemistrySpec
    time_s: np.ndarray
    i_true_a: np.ndarray
    i_meas_a: np.ndarray
    v_true_v: np.ndarray
    v_meas_v: np.ndarray
    t_true_k: np.ndarray
    soc_true: np.ndarray
    soc_cc: np.ndarray
    soc_ekf: np.ndarray
    soc_ukf: np.ndarray
    soc_ekf_std: np.ndarray
    soc_ukf_std: np.ndarray
    ekf_gain_soc: np.ndarray
    ekf_nis: np.ndarray
    docv_dsoc_true_v: np.ndarray
    rmse_cc_full: float
    rmse_ekf_full: float
    rmse_ukf_full: float
    rmse_cc_post_300s: float
    rmse_ekf_post_300s: float
    rmse_ukf_post_300s: float
    convergence_time_ekf_s: float
    convergence_time_ukf_s: float


@dataclass(frozen=True)
class SoCEstimationBenchmarkResult:
    """NMC811 vs LFP closed-loop SoC estimation & observability benchmark bundle."""

    nmc: ChemistryEstimationRun
    lfp: ChemistryEstimationRun
    soc_grid_obs: np.ndarray
    nmc_docv_dsoc_v: np.ndarray
    lfp_docv_dsoc_v: np.ndarray


class CoulombCountingEstimator:
    """Open-loop ampere-hour integration estimator."""

    def __init__(
        self,
        model: TwoRCBatteryModel,
        *,
        initial_soc: float = 0.50,
        initial_t_cell_k: float | None = None,
    ) -> None:
        if not math.isfinite(initial_soc) or not (0.0 <= initial_soc <= 1.0):
            raise ValueError(f"initial_soc must lie in [0, 1], got {initial_soc!r}")
        t0 = model.t_amb_k if initial_t_cell_k is None else check_temperature_k(initial_t_cell_k)
        self._model = model
        self._state = TwoRCState(soc=float(initial_soc), t_cell_k=t0)

    @property
    def soc(self) -> float:
        return self._state.soc

    @property
    def state(self) -> TwoRCState:
        return self._state

    def step(
        self,
        i_meas_a: float,
        v_meas_v: float,
        dt: float,
        *,
        t_cell_k: float | None = None,
    ) -> EstimatorStepResult:
        if not (math.isfinite(i_meas_a) and math.isfinite(v_meas_v)):
            raise ValueError("i_meas_a and v_meas_v must be finite")
        state_in = self._state if t_cell_k is None else TwoRCState(
            soc=self._state.soc,
            v_rc1_v=self._state.v_rc1_v,
            v_rc2_v=self._state.v_rc2_v,
            hysteresis=self._state.hysteresis,
            t_cell_k=check_temperature_k(t_cell_k),
        )
        sim_step = self._model.step(state_in, i_meas_a, dt)
        self._state = sim_step.state if t_cell_k is None else TwoRCState(
            soc=sim_step.state.soc,
            v_rc1_v=sim_step.state.v_rc1_v,
            v_rc2_v=sim_step.state.v_rc2_v,
            hysteresis=sim_step.state.hysteresis,
            t_cell_k=state_in.t_cell_k,
        )
        v_pred = self._model.terminal_voltage(self._state, i_meas_a)
        innov = float(v_meas_v) - v_pred
        slope = float(self._model.get_docv_dsoc(self._state.soc))
        return EstimatorStepResult(
            soc_hat=self._state.soc,
            v_rc1_hat_v=self._state.v_rc1_v,
            v_rc2_hat_v=self._state.v_rc2_v,
            hysteresis_hat=self._state.hysteresis,
            t_cell_k=self._state.t_cell_k,
            v_pred_v=v_pred,
            innovation_v=innov,
            innovation_var_v2=math.nan,
            nis=math.nan,
            kalman_gain_soc=0.0,
            soc_std=math.nan,
            docv_dsoc_v=slope,
            covariance=np.full((_STATE_DIM, _STATE_DIM), math.nan),
            innovation_gated=False,
        )


class _BaseKalmanSoCEstimator:
    """Shared state, hysteresis propagation, and exact discrete 2RC helper logic."""

    def __init__(
        self,
        model: TwoRCBatteryModel,
        *,
        initial_soc: float = 0.50,
        initial_v_rc1_v: float = 0.0,
        initial_v_rc2_v: float = 0.0,
        initial_hysteresis: float = 0.0,
        initial_t_cell_k: float | None = None,
        p0: np.ndarray | None = None,
        q_process: np.ndarray | None = None,
        r_meas_v2: float = 6.4e-5,
        innovation_gate_nis: float | None = None,
    ) -> None:
        if not math.isfinite(initial_soc) or not (0.0 <= initial_soc <= 1.0):
            raise ValueError(f"initial_soc must lie in [0, 1], got {initial_soc!r}")
        if not math.isfinite(r_meas_v2) or r_meas_v2 <= 0.0:
            raise ValueError(f"r_meas_v2 must be finite and > 0, got {r_meas_v2!r}")
        if innovation_gate_nis is not None and (
            not math.isfinite(innovation_gate_nis) or innovation_gate_nis <= 0.0
        ):
            raise ValueError("innovation_gate_nis must be finite and > 0 when provided")

        t0 = model.t_amb_k if initial_t_cell_k is None else check_temperature_k(initial_t_cell_k)
        init_state = TwoRCState(
            soc=float(initial_soc),
            v_rc1_v=float(initial_v_rc1_v),
            v_rc2_v=float(initial_v_rc2_v),
            hysteresis=float(initial_hysteresis),
            t_cell_k=t0,
        )
        default_p0 = np.diag([0.04, 1e-4, 1e-4])
        default_q = np.diag([2.5e-7, 1e-6, 1e-6])

        self._model = model
        self._x = np.array([init_state.soc, init_state.v_rc1_v, init_state.v_rc2_v], dtype=float)
        self._h = init_state.hysteresis
        self._t_k = init_state.t_cell_k
        self._P = _validate_cov_3x3("p0", default_p0 if p0 is None else p0)
        self._Q = _validate_cov_3x3("q_process", default_q if q_process is None else q_process)
        self._R = float(r_meas_v2)
        self._gate_nis = innovation_gate_nis

    @property
    def soc(self) -> float:
        return float(self._x[0])

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def state_vector(self) -> np.ndarray:
        return self._x.copy()

    def _propagate_x(self, x_vec: np.ndarray, i_app: float, dt: float, t_k: float) -> np.ndarray:
        """Pure discrete-time transition for x = [soc, v_rc1, v_rc2]."""
        q_coulomb = self._model.spec.nominal_capacity_ah * 3600.0
        soc0 = float(np.clip(x_vec[0], 0.0, 1.0))
        soc_new = float(np.clip(soc0 - (i_app * dt) / q_coulomb, 0.0, 1.0))
        soc_mid = 0.5 * (soc0 + soc_new)
        _, r1, c1, r2, c2 = self._model.get_impedances(soc_mid, t_k)

        x1 = dt / (r1 * c1)
        v1_inf = i_app * r1
        v1_new = float(x_vec[1]) + (float(x_vec[1]) - v1_inf) * math.expm1(-x1)

        x2 = dt / (r2 * c2)
        v2_inf = i_app * r2
        v2_new = float(x_vec[2]) + (float(x_vec[2]) - v2_inf) * math.expm1(-x2)
        return np.array([soc_new, v1_new, v2_new], dtype=float)

    def _advance_hysteresis_and_temp(
        self,
        i_app: float,
        dt: float,
        t_meas_k: float | None,
    ) -> tuple[float, float]:
        spec = self._model.spec
        q_coulomb = spec.nominal_capacity_ah * 3600.0
        if self._model._enable_hysteresis and spec.hysteresis_max_v > 0.0 and abs(i_app) > 0.0:
            xh = ((spec.hysteresis_rate_gamma * abs(i_app)) / q_coulomb) * dt
            target = -1.0 if i_app > 0.0 else 1.0
            h_new = float(np.clip(self._h + (self._h - target) * math.expm1(-xh), -1.0, 1.0))
        else:
            h_new = self._h if self._model._enable_hysteresis else 0.0

        if t_meas_k is not None:
            return h_new, check_temperature_k(t_meas_k)

        state_tmp = TwoRCState(
            soc=float(np.clip(self._x[0], 0.0, 1.0)),
            v_rc1_v=float(self._x[1]),
            v_rc2_v=float(self._x[2]),
            hysteresis=self._h,
            t_cell_k=self._t_k,
        )
        t_new = self._model.step(state_tmp, i_app, dt).state.t_cell_k
        return h_new, t_new

    def _predict_voltage(self, x_vec: np.ndarray, h_val: float, t_k: float, i_app: float) -> float:
        st = TwoRCState(
            soc=float(np.clip(x_vec[0], 0.0, 1.0)),
            v_rc1_v=float(x_vec[1]),
            v_rc2_v=float(x_vec[2]),
            hysteresis=h_val,
            t_cell_k=t_k,
        )
        return self._model.terminal_voltage(st, i_app)


class ExtendedKalmanFilterEstimator(_BaseKalmanSoCEstimator):
    """Extended Kalman Filter with analytical 2RC/PCHIP Jacobians and Joseph update."""

    def compute_state_jacobian(self, x_vec: np.ndarray, i_app: float, dt: float, t_k: float) -> np.ndarray:
        """Exact 3x3 Jacobian F_k = df/dx of the discrete-time 2RC state transition."""
        spec = self._model.spec
        q_coulomb = spec.nominal_capacity_ah * 3600.0
        soc0 = float(np.clip(x_vec[0], 0.0, 1.0))
        soc_unclipped = soc0 - (i_app * dt) / q_coulomb
        unclipped = 0.0 < soc_unclipped < 1.0
        soc_new = float(np.clip(soc_unclipped, 0.0, 1.0))
        soc_mid = 0.5 * (soc0 + soc_new)
        dsoc_mid_dsoc0 = 0.5 * (1.0 + (1.0 if unclipped else 0.0))

        _, r1, c1, r2, c2 = self._model.get_impedances(soc_mid, t_k)
        x1 = dt / (r1 * c1)
        x2 = dt / (r2 * c2)
        e1 = math.exp(-x1)
        e2 = math.exp(-x2)

        f_mat = np.zeros((_STATE_DIM, _STATE_DIM), dtype=float)
        f_mat[0, 0] = 1.0 if unclipped else 0.0
        f_mat[1, 1] = e1
        f_mat[2, 2] = e2

        if self._model._enable_soc_impedance and spec.r_low_soc_gain > 0.0:
            exp_term = spec.r_low_soc_gain * math.exp(-spec.r_low_soc_decay * soc_mid)
            dscale_over_scale = (-spec.r_low_soc_decay * exp_term) / (1.0 + exp_term)
            dr1_dsoc0 = r1 * dscale_over_scale * dsoc_mid_dsoc0
            dr2_dsoc0 = r2 * dscale_over_scale * dsoc_mid_dsoc0

            dv1_dr1 = i_app * (-math.expm1(-x1)) + (float(x_vec[1]) - i_app * r1) * e1 * (x1 / r1)
            dv2_dr2 = i_app * (-math.expm1(-x2)) + (float(x_vec[2]) - i_app * r2) * e2 * (x2 / r2)
            f_mat[1, 0] = dv1_dr1 * dr1_dsoc0
            f_mat[2, 0] = dv2_dr2 * dr2_dsoc0

        return f_mat

    def compute_measurement_jacobian(self, x_vec: np.ndarray, i_app: float, t_k: float) -> np.ndarray:
        """Exact 1x3 Jacobian H_k = dV_term/dx using analytical PCHIP and impedance slopes."""
        spec = self._model.spec
        soc = float(np.clip(x_vec[0], 0.0, 1.0))
        du_ref_dsoc = float(self._model.get_docv_dsoc(soc))

        # Entropic temperature correction slope (T - T_ref) * d/dSoC(dU/dT)
        dt_k = t_k - spec.t_ref_k
        if self._model._enable_entropic_heat and abs(dt_k) > 1e-12:
            eps = 1e-5
            s_hi = min(1.0, soc + eps)
            s_lo = max(0.0, soc - eps)
            ddudt_dsoc = (
                float(self._model.get_entropic_dudt(s_hi)) - float(self._model.get_entropic_dudt(s_lo))
            ) / max(1e-12, s_hi - s_lo)
        else:
            ddudt_dsoc = 0.0

        r0, _, _, _, _ = self._model.get_impedances(soc, t_k)
        if self._model._enable_soc_impedance and spec.r_low_soc_gain > 0.0:
            exp_term = spec.r_low_soc_gain * math.exp(-spec.r_low_soc_decay * soc)
            dr0_dsoc = r0 * (-spec.r_low_soc_decay * exp_term) / (1.0 + exp_term)
        else:
            dr0_dsoc = 0.0

        h0 = du_ref_dsoc + dt_k * ddudt_dsoc - i_app * dr0_dsoc
        return np.array([h0, -1.0, -1.0], dtype=float)

    def step(
        self,
        i_meas_a: float,
        v_meas_v: float,
        dt: float,
        *,
        t_cell_k: float | None = None,
    ) -> EstimatorStepResult:
        if not (math.isfinite(i_meas_a) and math.isfinite(v_meas_v)):
            raise ValueError("i_meas_a and v_meas_v must be finite")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt!r}")

        f_mat = self.compute_state_jacobian(self._x, i_meas_a, dt, self._t_k)
        x_pred = self._propagate_x(self._x, i_meas_a, dt, self._t_k)
        h_pred, t_pred = self._advance_hysteresis_and_temp(i_meas_a, dt, t_cell_k)
        p_pred = _enforce_spd_3x3(f_mat @ self._P @ f_mat.T + self._Q)

        v_pred = self._predict_voltage(x_pred, h_pred, t_pred, i_meas_a)
        h_vec = self.compute_measurement_jacobian(x_pred, i_meas_a, t_pred)
        innov = float(v_meas_v) - v_pred
        s_var = float(h_vec @ p_pred @ h_vec + self._R)
        nis = (innov * innov) / s_var

        if self._gate_nis is not None and nis > self._gate_nis:
            self._x = x_pred
            self._h = h_pred
            self._t_k = t_pred
            self._P = p_pred
            k_soc = 0.0
            gated = True
        else:
            k_gain = (p_pred @ h_vec) / s_var
            x_upd = x_pred + k_gain * innov
            x_upd[0] = float(np.clip(x_upd[0], 0.0, 1.0))

            # Symmetrized Bucy-Joseph covariance update
            i_kh = np.eye(_STATE_DIM) - np.outer(k_gain, h_vec)
            p_upd = i_kh @ p_pred @ i_kh.T + np.outer(k_gain, k_gain) * self._R
            self._x = x_upd
            self._h = h_pred
            self._t_k = t_pred
            self._P = _enforce_spd_3x3(p_upd)
            k_soc = float(k_gain[0])
            gated = False

        return EstimatorStepResult(
            soc_hat=float(self._x[0]),
            v_rc1_hat_v=float(self._x[1]),
            v_rc2_hat_v=float(self._x[2]),
            hysteresis_hat=self._h,
            t_cell_k=self._t_k,
            v_pred_v=v_pred,
            innovation_v=innov,
            innovation_var_v2=s_var,
            nis=nis,
            kalman_gain_soc=k_soc,
            soc_std=math.sqrt(max(0.0, float(self._P[0, 0]))),
            docv_dsoc_v=float(self._model.get_docv_dsoc(self._x[0])),
            covariance=self._P.copy(),
            innovation_gated=gated,
        )


class UnscentedKalmanFilterEstimator(_BaseKalmanSoCEstimator):
    """Scaled Unscented Kalman Filter (Wan & van der Merwe, 2000) for 2RC SoC estimation."""

    def __init__(
        self,
        model: TwoRCBatteryModel,
        *,
        initial_soc: float = 0.50,
        initial_v_rc1_v: float = 0.0,
        initial_v_rc2_v: float = 0.0,
        initial_hysteresis: float = 0.0,
        initial_t_cell_k: float | None = None,
        p0: np.ndarray | None = None,
        q_process: np.ndarray | None = None,
        r_meas_v2: float = 6.4e-5,
        alpha: float = 0.8,
        beta: float = 2.0,
        kappa: float = 1.0,
        innovation_gate_nis: float | None = None,
    ) -> None:
        super().__init__(
            model,
            initial_soc=initial_soc,
            initial_v_rc1_v=initial_v_rc1_v,
            initial_v_rc2_v=initial_v_rc2_v,
            initial_hysteresis=initial_hysteresis,
            initial_t_cell_k=initial_t_cell_k,
            p0=p0,
            q_process=q_process,
            r_meas_v2=r_meas_v2,
            innovation_gate_nis=innovation_gate_nis,
        )
        if not (math.isfinite(alpha) and alpha > 0.0):
            raise ValueError("alpha must be finite and > 0")
        if not (math.isfinite(beta) and beta >= 0.0):
            raise ValueError("beta must be finite and >= 0")
        if not math.isfinite(kappa) or (_STATE_DIM + kappa) <= 0.0:
            raise ValueError("kappa must satisfy n + kappa > 0")

        lam = (alpha * alpha) * (_STATE_DIM + kappa) - _STATE_DIM
        c_scale = _STATE_DIM + lam
        if c_scale <= 0.0:
            raise ValueError("Unscented scaling n + lambda must be > 0")

        wm = np.full(2 * _STATE_DIM + 1, 0.5 / c_scale, dtype=float)
        wc = np.full(2 * _STATE_DIM + 1, 0.5 / c_scale, dtype=float)
        wm[0] = lam / c_scale
        wc[0] = (lam / c_scale) + (1.0 - alpha * alpha + beta)

        self._gamma = math.sqrt(c_scale)
        self._wm = wm
        self._wc = wc

    def _sigma_points(self, x_mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        spd = _enforce_spd_3x3(cov)
        chol = np.linalg.cholesky(spd)
        scaled = self._gamma * chol
        sigmas = np.empty((2 * _STATE_DIM + 1, _STATE_DIM), dtype=float)
        sigmas[0] = x_mean
        for i in range(_STATE_DIM):
            col = scaled[:, i]
            sigmas[1 + i] = x_mean + col
            sigmas[1 + _STATE_DIM + i] = x_mean - col
        sigmas[:, 0] = np.clip(sigmas[:, 0], 0.0, 1.0)
        return sigmas

    def step(
        self,
        i_meas_a: float,
        v_meas_v: float,
        dt: float,
        *,
        t_cell_k: float | None = None,
    ) -> EstimatorStepResult:
        if not (math.isfinite(i_meas_a) and math.isfinite(v_meas_v)):
            raise ValueError("i_meas_a and v_meas_v must be finite")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt!r}")

        sigmas = self._sigma_points(self._x, self._P)
        prop_sigmas = np.array(
            [self._propagate_x(s, i_meas_a, dt, self._t_k) for s in sigmas],
            dtype=float,
        )
        x_pred = np.sum(self._wm[:, None] * prop_sigmas, axis=0)
        x_pred[0] = float(np.clip(x_pred[0], 0.0, 1.0))

        dx = prop_sigmas - x_pred[None, :]
        p_pred_raw = self._Q.copy()
        for j in range(len(self._wc)):
            p_pred_raw = p_pred_raw + self._wc[j] * np.outer(dx[j], dx[j])
        p_pred = _enforce_spd_3x3(p_pred_raw)

        h_pred, t_pred = self._advance_hysteresis_and_temp(i_meas_a, dt, t_cell_k)

        # Redraw sigma points around predicted covariance for nonlinear measurement UT
        meas_sigmas = self._sigma_points(x_pred, p_pred)
        y_sigmas = np.array(
            [self._predict_voltage(s, h_pred, t_pred, i_meas_a) for s in meas_sigmas],
            dtype=float,
        )
        v_pred = float(np.dot(self._wm, y_sigmas))
        dy = y_sigmas - v_pred
        dx_m = meas_sigmas - x_pred[None, :]

        s_var = float(np.dot(self._wc, dy * dy) + self._R)
        s_var = max(s_var, self._R * 0.1)
        p_xy = np.sum((self._wc * dy)[:, None] * dx_m, axis=0)

        innov = float(v_meas_v) - v_pred
        nis = (innov * innov) / s_var

        if self._gate_nis is not None and nis > self._gate_nis:
            self._x = x_pred
            self._h = h_pred
            self._t_k = t_pred
            self._P = p_pred
            k_soc = 0.0
            gated = True
        else:
            k_gain = p_xy / s_var
            x_upd = x_pred + k_gain * innov
            x_upd[0] = float(np.clip(x_upd[0], 0.0, 1.0))
            p_upd = p_pred - np.outer(k_gain, k_gain) * s_var
            self._x = x_upd
            self._h = h_pred
            self._t_k = t_pred
            self._P = _enforce_spd_3x3(p_upd)
            k_soc = float(k_gain[0])
            gated = False

        return EstimatorStepResult(
            soc_hat=float(self._x[0]),
            v_rc1_hat_v=float(self._x[1]),
            v_rc2_hat_v=float(self._x[2]),
            hysteresis_hat=self._h,
            t_cell_k=self._t_k,
            v_pred_v=v_pred,
            innovation_v=innov,
            innovation_var_v2=s_var,
            nis=nis,
            kalman_gain_soc=k_soc,
            soc_std=math.sqrt(max(0.0, float(self._P[0, 0]))),
            docv_dsoc_v=float(self._model.get_docv_dsoc(self._x[0])),
            covariance=self._P.copy(),
            innovation_gated=gated,
        )


def _dynamic_bess_crate_schedule(time_s: np.ndarray) -> np.ndarray:
    """Deterministic multi-regime grid storage C-rate schedule (88% -> 29% SoC swing)."""
    crates = np.zeros_like(time_s, dtype=float)
    for k, t in enumerate(time_s):
        window = int(t // 300.0) % 4
        if window == 1:
            crates[k] = 0.0
        elif window == 3:
            crates[k] = -0.35 + 0.15 * math.sin(2.0 * math.pi * t / 180.0)
        else:
            crates[k] = 1.35 + 0.35 * math.sin(2.0 * math.pi * t / 240.0)
    return crates


def _first_sustained_convergence_s(
    time_s: np.ndarray,
    abs_err: np.ndarray,
    threshold: float = 0.02,
    hold_steps: int = 30,
) -> float:
    """Return earliest time [s] where |error| <= threshold continuously for hold_steps."""
    within = abs_err <= threshold
    n = len(within)
    for idx in range(0, max(0, n - hold_steps + 1)):
        if bool(np.all(within[idx : idx + hold_steps])):
            return float(time_s[idx])
    return float(time_s[-1])


def run_single_chemistry_estimation(
    spec: ChemistrySpec,
    *,
    total_time_s: float = 3600.0,
    dt: float = 1.0,
    true_initial_soc: float = 0.88,
    est_initial_soc: float = 0.65,
    current_bias_c_rate: float = 0.05,
    current_noise_std_c_rate: float = 0.01,
    voltage_noise_std_v: float = 0.008,
    seed: int = 42,
) -> ChemistryEstimationRun:
    """Run truth plant + corrupted sensors + CC / EKF / UKF head-to-head on ``spec``."""
    rng = np.random.default_rng(seed)
    n = int(round(total_time_s / dt))
    time_s = np.arange(n, dtype=float) * dt
    cap_ah = spec.nominal_capacity_ah

    crates = _dynamic_bess_crate_schedule(time_s)
    i_true = crates * cap_ah
    i_bias_a = current_bias_c_rate * cap_ah
    i_noise_std_a = current_noise_std_c_rate * cap_ah

    plant = TwoRCBatteryModel(spec)
    truth_state = TwoRCState(soc=true_initial_soc, t_cell_k=298.15)

    i_meas = np.empty(n, dtype=float)
    v_true = np.empty(n, dtype=float)
    v_meas = np.empty(n, dtype=float)
    t_true_k = np.empty(n, dtype=float)
    soc_true = np.empty(n, dtype=float)
    docv_true = np.empty(n, dtype=float)

    for k in range(n):
        step_res = plant.step(truth_state, float(i_true[k]), dt)
        truth_state = step_res.state
        soc_true[k] = truth_state.soc
        v_true[k] = step_res.v_term_v
        t_true_k[k] = truth_state.t_cell_k
        docv_true[k] = float(plant.get_docv_dsoc(truth_state.soc))
        i_meas[k] = i_true[k] + i_bias_a + rng.normal(0.0, i_noise_std_a)
        v_meas[k] = v_true[k] + rng.normal(0.0, voltage_noise_std_v)

    r_var = max(1e-8, voltage_noise_std_v * voltage_noise_std_v)
    sigma_i_eff = abs(i_bias_a) + i_noise_std_a
    q_soc = 4.0 * ((sigma_i_eff * dt) / (cap_ah * 3600.0)) ** 2 + 5e-7
    q_rc1 = ((sigma_i_eff * dt) / spec.c1_ref_f) ** 2 + 1e-8
    q_rc2 = ((sigma_i_eff * dt) / spec.c2_ref_f) ** 2 + 2e-9
    q_mat = np.diag([q_soc, q_rc1, q_rc2])
    p0_mat = np.diag([(true_initial_soc - est_initial_soc) ** 2 + 5e-3, 1e-5, 1e-5])

    cc = CoulombCountingEstimator(plant, initial_soc=est_initial_soc)
    ekf = ExtendedKalmanFilterEstimator(
        plant,
        initial_soc=est_initial_soc,
        p0=p0_mat,
        q_process=q_mat,
        r_meas_v2=r_var,
    )
    ukf = UnscentedKalmanFilterEstimator(
        plant,
        initial_soc=est_initial_soc,
        p0=p0_mat,
        q_process=q_mat,
        r_meas_v2=r_var,
    )

    soc_cc = np.empty(n, dtype=float)
    soc_ekf = np.empty(n, dtype=float)
    soc_ukf = np.empty(n, dtype=float)
    soc_ekf_std = np.empty(n, dtype=float)
    soc_ukf_std = np.empty(n, dtype=float)
    ekf_gain = np.empty(n, dtype=float)
    ekf_nis = np.empty(n, dtype=float)

    for k in range(n):
        im = float(i_meas[k])
        vm = float(v_meas[k])
        tk = float(t_true_k[k])

        r_cc = cc.step(im, vm, dt, t_cell_k=tk)
        r_ekf = ekf.step(im, vm, dt, t_cell_k=tk)
        r_ukf = ukf.step(im, vm, dt, t_cell_k=tk)

        soc_cc[k] = r_cc.soc_hat
        soc_ekf[k] = r_ekf.soc_hat
        soc_ukf[k] = r_ukf.soc_hat
        soc_ekf_std[k] = r_ekf.soc_std
        soc_ukf_std[k] = r_ukf.soc_std
        ekf_gain[k] = r_ekf.kalman_gain_soc
        ekf_nis[k] = r_ekf.nis

    post_mask = time_s >= 300.0
    has_post = bool(np.any(post_mask))
    err_cc = soc_cc - soc_true
    err_ekf = soc_ekf - soc_true
    err_ukf = soc_ukf - soc_true

    return ChemistryEstimationRun(
        spec=spec,
        time_s=time_s,
        i_true_a=i_true,
        i_meas_a=i_meas,
        v_true_v=v_true,
        v_meas_v=v_meas,
        t_true_k=t_true_k,
        soc_true=soc_true,
        soc_cc=soc_cc,
        soc_ekf=soc_ekf,
        soc_ukf=soc_ukf,
        soc_ekf_std=soc_ekf_std,
        soc_ukf_std=soc_ukf_std,
        ekf_gain_soc=ekf_gain,
        ekf_nis=ekf_nis,
        docv_dsoc_true_v=docv_true,
        rmse_cc_full=float(np.sqrt(np.mean(err_cc * err_cc))),
        rmse_ekf_full=float(np.sqrt(np.mean(err_ekf * err_ekf))),
        rmse_ukf_full=float(np.sqrt(np.mean(err_ukf * err_ukf))),
        rmse_cc_post_300s=float(np.sqrt(np.mean(err_cc[post_mask] ** 2))) if has_post else float("nan"),
        rmse_ekf_post_300s=float(np.sqrt(np.mean(err_ekf[post_mask] ** 2))) if has_post else float("nan"),
        rmse_ukf_post_300s=float(np.sqrt(np.mean(err_ukf[post_mask] ** 2))) if has_post else float("nan"),
        convergence_time_ekf_s=_first_sustained_convergence_s(time_s, np.abs(err_ekf)),
        convergence_time_ukf_s=_first_sustained_convergence_s(time_s, np.abs(err_ukf)),
    )


def run_soc_estimation_benchmark(dt: float = 1.0) -> SoCEstimationBenchmarkResult:
    """Run paired NMC811 vs LFP closed-loop SoC estimation & observability benchmark."""
    nmc_run = run_single_chemistry_estimation(NMC811_LG_M50_5AH, dt=dt, seed=42)
    lfp_run = run_single_chemistry_estimation(LFP_A123_ANR26650_2P3AH, dt=dt, seed=42)

    soc_obs = np.linspace(0.02, 0.98, 193)
    nmc_slope = np.asarray(TwoRCBatteryModel(NMC811_LG_M50_5AH).get_docv_dsoc(soc_obs), dtype=float)
    lfp_slope = np.asarray(TwoRCBatteryModel(LFP_A123_ANR26650_2P3AH).get_docv_dsoc(soc_obs), dtype=float)
    return SoCEstimationBenchmarkResult(
        nmc=nmc_run,
        lfp=lfp_run,
        soc_grid_obs=soc_obs,
        nmc_docv_dsoc_v=nmc_slope,
        lfp_docv_dsoc_v=lfp_slope,
    )
