"""Unit, analytical matrix-exponential, CC-CV, and SOP BMS tests for Module 4."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.bms_thermal_sop import (
    CoupledTwoNodeBESS,
    CoupledTwoNodeState,
    TwoNodeThermalNetwork,
    TwoNodeThermalState,
    compute_sop_limits,
    simulate_cc_cv_charge,
)
from src.cell_chemistry import LFP_A123_ANR26650_2P3AH, NMC811_LG_M50_5AH


class TestTwoNodeThermalMatrixExponential:
    @pytest.mark.parametrize("dt", [0.05, 1.0, 25.0, 400.0])
    def test_step_matches_eigenmode_ode_and_semigroup_composition(self, dt):
        net = TwoNodeThermalNetwork(
            c_core_j_k=45.0,
            c_surf_j_k=11.0,
            r_cs_k_w=2.0,
            h_sa_w_k=0.50,
            t_amb_k=298.15,
        )
        s0 = TwoNodeThermalState(t_core_k=303.15, t_surf_k=299.65)
        q_w = 3.2

        # Exact semigroup property: 1 step of dt == 10 sub-steps of dt/10 to machine precision
        one_big = net.step(s0, q_w, dt).state
        sub = s0
        for _ in range(10):
            sub = net.step(sub, q_w, dt / 10.0).state

        assert one_big.t_core_k == pytest.approx(sub.t_core_k, abs=1e-12)
        assert one_big.t_surf_k == pytest.approx(sub.t_surf_k, abs=1e-12)

        # Compare against 4th-order Runge-Kutta on fine grid
        th = np.array([s0.t_core_k - 298.15, s0.t_surf_k - 298.15], dtype=float)
        u = np.array([q_w / 45.0, 0.0], dtype=float)
        n_rk = 2000
        h = dt / n_rk
        a_mat = net._A
        for _ in range(n_rk):
            k1 = a_mat @ th + u
            k2 = a_mat @ (th + 0.5 * h * k1) + u
            k3 = a_mat @ (th + 0.5 * h * k2) + u
            k4 = a_mat @ (th + h * k3) + u
            th = th + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        assert one_big.t_core_k == pytest.approx(298.15 + th[0], abs=1e-11)
        assert one_big.t_surf_k == pytest.approx(298.15 + th[1], abs=1e-11)

    @pytest.mark.parametrize("dt", [1e-3, 1.0, 60.0, 900.0])
    def test_adiabatic_limit_conserves_first_law_energy_to_machine_precision(self, dt):
        net = TwoNodeThermalNetwork(
            c_core_j_k=46.0,
            c_surf_j_k=10.0,
            r_cs_k_w=1.95,
            h_sa_w_k=0.0,  # Adiabatic!
            t_amb_k=298.15,
        )
        assert net.eigenvalues_per_s[1] == 0.0
        assert math.isinf(net.modal_time_constants_s[1])

        s0 = TwoNodeThermalState(t_core_k=301.15, t_surf_k=296.15)
        q_w = 0.85
        res = net.step(s0, q_w, dt)
        s1 = res.state

        delta_u_j = net.c_core_j_k * (s1.t_core_k - s0.t_core_k) + net.c_surf_j_k * (
            s1.t_surf_k - s0.t_surf_k
        )
        assert delta_u_j == pytest.approx(q_w * dt, rel=1e-12, abs=5e-12)
        assert res.q_surf_to_amb_w == pytest.approx(0.0, abs=1e-13)

    def test_adiabatic_zero_heat_equilibrates_to_weighted_mean_temperature(self):
        net = TwoNodeThermalNetwork(
            c_core_j_k=40.0,
            c_surf_j_k=10.0,
            r_cs_k_w=2.0,
            h_sa_w_k=0.0,
            t_amb_k=298.15,
        )
        s0 = TwoNodeThermalState(t_core_k=318.15, t_surf_k=298.15)
        s_eq = net.step(s0, 0.0, 5000.0).state
        expected_k = (40.0 * 318.15 + 10.0 * 298.15) / 50.0
        assert s_eq.t_core_k == pytest.approx(expected_k, abs=1e-11)
        assert s_eq.t_surf_k == pytest.approx(expected_k, abs=1e-11)

    def test_convective_steady_state_matches_closed_form_thermal_resistances(self):
        net = TwoNodeThermalNetwork.from_chemistry(NMC811_LG_M50_5AH, t_amb_k=298.15, h_sa_w_k=0.80)
        q_w = 3.5
        tc_inf, ts_inf = net.steady_state_temperatures(q_w)
        assert ts_inf == pytest.approx(298.15 + q_w / 0.80, abs=1e-13)
        assert tc_inf == pytest.approx(ts_inf + q_w * NMC811_LG_M50_5AH.r_core_surf_k_w, abs=1e-13)

        st = net.step(TwoNodeThermalState(298.15, 298.15), q_w, 25000.0).state
        assert st.t_core_k == pytest.approx(tc_inf, abs=1e-9)
        assert st.t_surf_k == pytest.approx(ts_inf, abs=1e-9)


class TestCCCVFastChargingAndThermalDerating:
    @pytest.mark.parametrize("spec", [NMC811_LG_M50_5AH, LFP_A123_ANR26650_2P3AH])
    def test_cc_cv_zero_voltage_overshoot_and_monotone_soc_and_cv_taper(self, spec):
        bess = CoupledTwoNodeBESS(spec, t_amb_k=298.15, h_sa_w_k=0.90)
        res = simulate_cc_cv_charge(
            bess,
            initial_soc=0.20,
            cc_c_rate=2.0,
            cutoff_c_rate=0.05,
            t_core_max_c=48.0,
            dt=4.0,
        )
        assert res.max_v_overshoot_v <= 1e-10
        assert np.all(res.v_term_v <= spec.v_max_v + 1e-10)
        assert np.all(np.diff(res.soc) >= -1e-15)
        assert res.final_soc > 0.96
        assert np.any(res.phase_code == 0)  # CC phase occurred
        assert np.any(res.phase_code == 2)  # CV phase occurred
        assert abs(res.current_a[-1]) <= 0.05 * spec.nominal_capacity_ah + 1e-9

    def test_cc_cv_active_core_thermal_derating_enforces_ceiling(self):
        # Warm 38 degC enclosure + tight 42.0 degC T_core limit forces active Phase 1 derating
        bess = CoupledTwoNodeBESS(NMC811_LG_M50_5AH, t_amb_k=311.15, h_sa_w_k=0.45)
        res = simulate_cc_cv_charge(
            bess,
            initial_soc=0.15,
            cc_c_rate=2.2,
            cutoff_c_rate=0.05,
            t_core_max_c=42.0,
            initial_temp_c=38.0,
            dt=4.0,
        )
        assert np.any(res.phase_code == 1)
        assert res.peak_t_core_c <= 42.0 + 1e-6
        assert res.max_v_overshoot_v <= 1e-10


class TestRecedingHorizonSOPEnvelope:
    def test_sop_monotone_decreasing_with_horizon_and_respects_all_rollout_bounds(self):
        bess = CoupledTwoNodeBESS(NMC811_LG_M50_5AH, t_amb_k=298.15, h_sa_w_k=0.85)
        st = CoupledTwoNodeState(soc=0.22, t_core_k=298.15, t_surf_k=298.15)

        sop_10 = compute_sop_limits(bess, st, horizon_s=10.0, dt_sub=1.0)
        sop_30 = compute_sop_limits(bess, st, horizon_s=30.0, dt_sub=1.0)
        sop_120 = compute_sop_limits(bess, st, horizon_s=120.0, dt_sub=2.0)

        assert sop_10.i_dis_max_a >= sop_30.i_dis_max_a - 1e-6
        assert sop_30.i_dis_max_a >= sop_120.i_dis_max_a - 1e-6

        # Explicitly simulate 30 steps at i_dis_max_a and verify every safety invariant holds
        cur = st
        for _ in range(30):
            sr = bess.step(cur, sop_30.i_dis_max_a, 1.0)
            cur = sr.state
            assert sr.v_term_v >= NMC811_LG_M50_5AH.v_min_v - 1e-4
            assert cur.soc >= 0.05 - 1e-6
            assert cur.t_core_c <= 48.0 + 1e-4

    def test_sop_identifies_all_four_physical_bottleneck_regimes_and_plating_lockout(self):
        bess = CoupledTwoNodeBESS(NMC811_LG_M50_5AH, t_amb_k=298.15, h_sa_w_k=0.85)

        # 1. Near-empty SoC -> "soc" discharge bottleneck
        sop_low_soc = compute_sop_limits(
            bess, CoupledTwoNodeState(soc=0.055, t_core_k=298.15, t_surf_k=298.15), horizon_s=60.0
        )
        assert sop_low_soc.dis_bottleneck == "soc"

        # 2. Mid-SoC room temperature -> "current" hardware ceiling
        sop_mid = compute_sop_limits(
            bess, CoupledTwoNodeState(soc=0.60, t_core_k=298.15, t_surf_k=298.15), horizon_s=20.0
        )
        assert sop_mid.dis_bottleneck == "current"

        # 3. Cold -16 degC -> "voltage" discharge bottleneck & 0 W Li-plating charge lockout
        bess_cold = CoupledTwoNodeBESS(NMC811_LG_M50_5AH, t_amb_k=257.15, h_sa_w_k=0.85)
        sop_cold = compute_sop_limits(
            bess_cold, CoupledTwoNodeState(soc=0.35, t_core_k=257.15, t_surf_k=257.15), horizon_s=60.0
        )
        assert sop_cold.dis_bottleneck == "voltage"
        assert sop_cold.i_chg_max_a == 0.0
        assert sop_cold.p_chg_max_w == 0.0

        # 4. Hot 46.5 degC -> "thermal_core" bottleneck
        bess_hot = CoupledTwoNodeBESS(NMC811_LG_M50_5AH, t_amb_k=319.65, h_sa_w_k=0.85)
        sop_hot = compute_sop_limits(
            bess_hot,
            CoupledTwoNodeState(soc=0.55, t_core_k=319.65, t_surf_k=319.65),
            horizon_s=60.0,
            t_core_max_c=48.0,
        )
        assert sop_hot.dis_bottleneck == "thermal_core"

    def test_validation_rejects_invalid_radial_and_sop_parameters(self):
        with pytest.raises(ValueError, match="core_heat_capacity_fraction"):
            replace(NMC811_LG_M50_5AH, core_heat_capacity_fraction=1.0)
        with pytest.raises(ValueError, match="r_core_surf_k_w"):
            replace(NMC811_LG_M50_5AH, r_core_surf_k_w=0.0)
        with pytest.raises(ValueError, match="h_sa_w_k"):
            TwoNodeThermalNetwork(h_sa_w_k=-0.1)

        bess = CoupledTwoNodeBESS(NMC811_LG_M50_5AH)
        st = CoupledTwoNodeState(soc=0.5)
        with pytest.raises(ValueError, match="horizon_s"):
            compute_sop_limits(bess, st, horizon_s=0.0)
        with pytest.raises(ValueError, match="soc_min"):
            compute_sop_limits(bess, st, soc_min=0.8, soc_max=0.5)
