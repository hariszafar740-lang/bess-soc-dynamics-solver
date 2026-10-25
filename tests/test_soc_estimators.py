"""Analytical Jacobian, covariance SPD, convergence, and outlier-gating tests (Module 3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.cell_chemistry import LFP_A123_ANR26650_2P3AH, NMC811_LG_M50_5AH
from src.ecm_2rc import TwoRCBatteryModel, TwoRCState
from src.soc_estimators import (
    CoulombCountingEstimator,
    ExtendedKalmanFilterEstimator,
    UnscentedKalmanFilterEstimator,
    run_single_chemistry_estimation,
    run_soc_estimation_benchmark,
)


class TestJacobiansAndUnscentedMoments:
    @pytest.mark.parametrize("soc", [0.125, 0.375, 0.625, 0.875])
    @pytest.mark.parametrize("i_app", [-4.0, 0.0, 6.5])
    def test_ekf_state_jacobian_matches_central_finite_differences_all_9_entries(self, soc, i_app):
        model = TwoRCBatteryModel(NMC811_LG_M50_5AH)
        ekf = ExtendedKalmanFilterEstimator(model, initial_soc=soc)
        x0 = np.array([soc, 0.025, -0.018], dtype=float)
        dt = 1.5
        t_k = 303.15

        f_analytic = ekf.compute_state_jacobian(x0, i_app, dt, t_k)
        f_numeric = np.zeros((3, 3), dtype=float)
        eps = 1e-6
        for j in range(3):
            dx = np.zeros(3, dtype=float)
            dx[j] = eps
            f_plus = ekf._propagate_x(x0 + dx, i_app, dt, t_k)
            f_minus = ekf._propagate_x(x0 - dx, i_app, dt, t_k)
            f_numeric[:, j] = (f_plus - f_minus) / (2.0 * eps)

        np.testing.assert_allclose(f_analytic, f_numeric, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("soc", [0.125, 0.455, 0.785])
    @pytest.mark.parametrize("t_k", [285.15, 298.15, 313.15])
    def test_ekf_measurement_jacobian_matches_central_finite_differences(self, soc, t_k):
        model = TwoRCBatteryModel(NMC811_LG_M50_5AH)
        ekf = ExtendedKalmanFilterEstimator(model, initial_soc=soc)
        x0 = np.array([soc, 0.03, 0.02], dtype=float)
        i_app = 5.0
        h_val = -0.4

        h_analytic = ekf.compute_measurement_jacobian(x0, i_app, t_k)
        h_numeric = np.zeros(3, dtype=float)
        eps = 1e-6
        for j in range(3):
            dx = np.zeros(3, dtype=float)
            dx[j] = eps
            vp = ekf._predict_voltage(x0 + dx, h_val, t_k, i_app)
            vm = ekf._predict_voltage(x0 - dx, h_val, t_k, i_app)
            h_numeric[j] = (vp - vm) / (2.0 * eps)

        np.testing.assert_allclose(h_analytic, h_numeric, rtol=1e-5, atol=1e-6)

    def test_ukf_sigma_points_reproduce_exact_mean_and_covariance(self):
        model = TwoRCBatteryModel(NMC811_LG_M50_5AH)
        ukf = UnscentedKalmanFilterEstimator(model, initial_soc=0.55)
        mean = np.array([0.55, 0.012, -0.008], dtype=float)
        cov = np.array(
            [
                [1.6e-3, 1.2e-5, -0.8e-5],
                [1.2e-5, 4.0e-5, 0.5e-5],
                [-0.8e-5, 0.5e-5, 3.0e-5],
            ],
            dtype=float,
        )
        sigmas = ukf._sigma_points(mean, cov)
        assert sigmas.shape == (7, 3)

        rec_mean = np.sum(ukf._wm[:, None] * sigmas, axis=0)
        diff = sigmas - rec_mean[None, :]
        rec_cov = sum(ukf._wc[i] * np.outer(diff[i], diff[i]) for i in range(7))

        np.testing.assert_allclose(rec_mean, mean, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(rec_cov, cov, rtol=1e-12, atol=1e-15)


class TestClosedLoopConvergenceAndInvariants:
    def test_coulomb_counting_exact_without_bias_and_linear_drift_under_bias(self):
        model = TwoRCBatteryModel(NMC811_LG_M50_5AH)
        st = TwoRCState(soc=0.80, t_cell_k=298.15)
        cc_clean = CoulombCountingEstimator(model, initial_soc=0.80)
        cc_biased = CoulombCountingEstimator(model, initial_soc=0.80)

        i_true = 5.0
        i_bias = 0.50  # +0.5 A persistent bias over 1000 s -> -0.5*1000/(5*3600) = -1/36 SoC drift
        for _ in range(1000):
            sr = model.step(st, i_true, 1.0)
            st = sr.state
            cc_clean.step(i_true, sr.v_term_v, 1.0, t_cell_k=st.t_cell_k)
            cc_biased.step(i_true + i_bias, sr.v_term_v, 1.0, t_cell_k=st.t_cell_k)

        assert cc_clean.soc == pytest.approx(st.soc, abs=1e-14)
        expected_drift = -(i_bias * 1000.0) / (5.0 * 3600.0)
        assert (cc_biased.soc - st.soc) == pytest.approx(expected_drift, abs=1e-13)

    def test_nmc811_ekf_and_ukf_recover_from_23pct_offset_and_preserve_spd_covariance(self):
        run = run_single_chemistry_estimation(NMC811_LG_M50_5AH, total_time_s=1800.0, dt=1.0, seed=7)
        assert run.rmse_cc_full > 0.22
        assert run.rmse_ekf_full < 0.02
        assert run.rmse_ukf_full < 0.02
        assert abs(run.soc_ekf[-1] - run.soc_true[-1]) < 0.02
        assert abs(run.soc_ukf[-1] - run.soc_true[-1]) < 0.02
        assert np.all(run.soc_ekf_std > 0.0)
        assert np.all(run.soc_ukf_std > 0.0)

    def test_lfp_ukf_outperforms_linearized_ekf_across_initial_phase_knee(self):
        run_lfp = run_single_chemistry_estimation(LFP_A123_ANR26650_2P3AH, total_time_s=3600.0, dt=1.0, seed=42)
        # Over the initial 300 s knee-straddling transient (est=65% plateau vs true=88% upper knee),
        # UKF sigma points avoid EKF's local-slope overshoot.
        early_rmse_ekf = float(np.sqrt(np.mean((run_lfp.soc_ekf[:300] - run_lfp.soc_true[:300]) ** 2)))
        early_rmse_ukf = float(np.sqrt(np.mean((run_lfp.soc_ukf[:300] - run_lfp.soc_true[:300]) ** 2)))
        assert early_rmse_ukf < early_rmse_ekf
        # And once LFP exits the flat plateau below 40% SoC, both lock within 2.5% of truth!
        assert abs(run_lfp.soc_ekf[-1] - run_lfp.soc_true[-1]) < 0.025
        assert abs(run_lfp.soc_ukf[-1] - run_lfp.soc_true[-1]) < 0.025

    def test_nis_innovation_gate_rejects_corrupt_voltage_sensor_spike(self):
        model = TwoRCBatteryModel(NMC811_LG_M50_5AH)
        st = TwoRCState(soc=0.70, t_cell_k=298.15)
        ekf_ungated = ExtendedKalmanFilterEstimator(model, initial_soc=0.70)
        ekf_gated = ExtendedKalmanFilterEstimator(model, initial_soc=0.70, innovation_gate_nis=25.0)
        ukf_gated = UnscentedKalmanFilterEstimator(model, initial_soc=0.70, innovation_gate_nis=25.0)

        for k in range(60):
            sr = model.step(st, 5.0, 1.0)
            st = sr.state
            v_meas = sr.v_term_v + (0.90 if k == 30 else 0.0)  # +900 mV single-sample ADC glitch
            r_un = ekf_ungated.step(5.0, v_meas, 1.0, t_cell_k=st.t_cell_k)
            r_ekf = ekf_gated.step(5.0, v_meas, 1.0, t_cell_k=st.t_cell_k)
            r_ukf = ukf_gated.step(5.0, v_meas, 1.0, t_cell_k=st.t_cell_k)
            if k == 30:
                assert r_ekf.innovation_gated is True
                assert r_ukf.innovation_gated is True
                assert r_un.innovation_gated is False
                assert abs(r_ekf.soc_hat - st.soc) < 1e-4
                assert abs(r_ukf.soc_hat - st.soc) < 1e-4
                assert abs(r_un.soc_hat - st.soc) > 0.03

    def test_benchmark_does_not_mutate_global_numpy_rng_and_validates_inputs(self):
        np.random.seed(12345)
        expected = np.random.rand(4)
        np.random.seed(12345)
        _ = run_soc_estimation_benchmark(dt=5.0)
        actual = np.random.rand(4)
        np.testing.assert_array_equal(actual, expected)

        model = TwoRCBatteryModel(NMC811_LG_M50_5AH)
        with pytest.raises(ValueError, match="initial_soc"):
            ExtendedKalmanFilterEstimator(model, initial_soc=1.2)
        with pytest.raises(ValueError, match="positive-definite"):
            ExtendedKalmanFilterEstimator(model, p0=np.diag([0.01, -1e-4, 1e-4]))
        with pytest.raises(ValueError, match="symmetric"):
            ExtendedKalmanFilterEstimator(
                model, p0=np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            )
        with pytest.raises(ValueError, match="r_meas_v2"):
            UnscentedKalmanFilterEstimator(model, r_meas_v2=0.0)
