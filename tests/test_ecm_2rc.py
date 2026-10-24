"""Unit, analytical, thermodynamic conservation, and EIS tests for Module 2 (2RC ECM)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from src.cell_chemistry import (
    LFP_A123_ANR26650_2P3AH,
    NMC811_LG_M50_5AH,
    MonotonePCHIP,
    get_chemistry_spec,
    list_chemistry_specs,
)
from src.ecm_2rc import (
    TwoRCBatteryModel,
    TwoRCState,
    compute_eis_nyquist,
    run_chemistry_comparison_benchmark,
)


class TestMonotonePCHIPAndChemistrySpecs:
    @pytest.mark.parametrize("spec", [NMC811_LG_M50_5AH, LFP_A123_ANR26650_2P3AH])
    def test_table_sha256_provenance_integrity(self, spec):
        payload = json.dumps(
            {
                "soc": list(spec.soc_grid),
                "ocv": list(spec.ocv_v_grid),
                "dudt": list(spec.dudt_v_per_k_grid),
            },
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert digest == spec.table_sha256

    @pytest.mark.parametrize("spec", [NMC811_LG_M50_5AH, LFP_A123_ANR26650_2P3AH])
    def test_pchip_exact_knots_and_strict_monotonicity(self, spec):
        spline = MonotonePCHIP(spec.soc_grid, spec.ocv_v_grid)
        knot_vals = spline(np.asarray(spec.soc_grid))
        np.testing.assert_allclose(knot_vals, np.asarray(spec.ocv_v_grid), rtol=0.0, atol=1e-14)

        dense_soc = np.linspace(0.0, 1.0, 5001)
        dense_ocv = spline(dense_soc)
        assert np.all(np.diff(dense_ocv) >= -1e-15)
        dense_slope = spline.derivative(dense_soc)
        assert np.all(dense_slope >= -1e-14)

    def test_pchip_analytical_derivative_matches_central_difference(self):
        spline = MonotonePCHIP(NMC811_LG_M50_5AH.soc_grid, NMC811_LG_M50_5AH.ocv_v_grid)
        probe = np.linspace(0.025, 0.975, 20)  # Interior to 1%-SoC spline intervals
        eps = 1e-6
        fd = (spline(probe + eps) - spline(probe - eps)) / (2.0 * eps)
        analytic = spline.derivative(probe)
        np.testing.assert_allclose(analytic, fd, rtol=1e-6, atol=1e-7)

    def test_pchip_handles_flat_plateau_without_overshoot(self):
        x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = np.array([2.5, 3.2, 3.2, 3.2, 3.3, 3.6])
        spline = MonotonePCHIP(x, y)
        mid = spline(np.linspace(0.2, 0.6, 50))
        np.testing.assert_allclose(mid, 3.2, atol=1e-15)

    def test_pchip_rejects_invalid_knots(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            MonotonePCHIP([0.0, 0.5, 0.5, 1.0], [3.0, 3.5, 3.6, 4.0])
        with pytest.raises(ValueError, match="equal length >= 3"):
            MonotonePCHIP([0.0, 1.0], [3.0, 4.0])

    def test_chemistry_registry_lookup_and_validation(self):
        assert get_chemistry_spec("NMC811") is NMC811_LG_M50_5AH
        assert get_chemistry_spec("lfp") is LFP_A123_ANR26650_2P3AH
        assert len(list_chemistry_specs()) == 2
        with pytest.raises(ValueError, match="Unknown chemistry key"):
            get_chemistry_spec("sodium_ion_unknown")
        with pytest.raises(ValueError, match="v_min_v"):
            replace(NMC811_LG_M50_5AH, v_min_v=4.3, v_max_v=4.2)


class TestTwoRCAnalyticalDynamicsAndConservation:
    def test_bi_exponential_step_and_rest_matches_closed_form_solution(self):
        iso_spec = replace(NMC811_LG_M50_5AH, ea_r0_j_mol=0.0, ea_r1_j_mol=0.0, ea_r2_j_mol=0.0)
        model = TwoRCBatteryModel(
            iso_spec,
            h_a_w_k=1e6,
            enable_soc_impedance=False,
            enable_entropic_heat=False,
            enable_hysteresis=False,
        )
        r0, r1, c1, r2, c2 = model.get_impedances(0.8, 298.15)
        tau1, tau2 = r1 * c1, r2 * c2
        i_step = 5.0
        dt = 2.0

        state = TwoRCState(soc=0.80, t_cell_k=298.15)
        for step_idx in range(1, 151):
            res = model.step(state, i_step, dt)
            state = res.state
            t = step_idx * dt
            v1_exact = i_step * r1 * (1.0 - math.exp(-t / tau1))
            v2_exact = i_step * r2 * (1.0 - math.exp(-t / tau2))
            assert state.v_rc1_v == pytest.approx(v1_exact, abs=1e-13)
            assert state.v_rc2_v == pytest.approx(v2_exact, abs=1e-13)
            v_term_exact = model.get_ocv_ref(state.soc) - i_step * r0 - v1_exact - v2_exact
            assert res.v_term_v == pytest.approx(v_term_exact, abs=1e-9)

        # Now switch to zero-current rest and verify pure bi-exponential decay
        v1_0, v2_0 = state.v_rc1_v, state.v_rc2_v
        for rest_idx in range(1, 121):
            res = model.step(state, 0.0, dt)
            state = res.state
            t_rest = rest_idx * dt
            assert state.v_rc1_v == pytest.approx(v1_0 * math.exp(-t_rest / tau1), abs=1e-13)
            assert state.v_rc2_v == pytest.approx(v2_0 * math.exp(-t_rest / tau2), abs=1e-13)

    @pytest.mark.parametrize("dt", [1e-5, 0.01, 1.0, 25.0, 400.0])
    def test_first_law_rc_branch_work_energy_identity_exact_all_dt(self, dt):
        """Verify int I*V_rc dt - int (V_rc^2/R) dt == 0.5*C*(V_new^2 - V_old^2) to ~1e-15 J."""
        r_ohm, c_f = 0.015, 2400.0
        v_old = 0.042
        i_app = -6.5
        v_new, p_res_w, p_eta_w = TwoRCBatteryModel._exact_rc_branch_step(v_old, i_app, r_ohm, c_f, dt)
        delta_u_cap_j = 0.5 * c_f * (v_new * v_new - v_old * v_old)
        net_work_minus_heat_j = (p_eta_w - p_res_w) * dt
        assert net_work_minus_heat_j == pytest.approx(delta_u_cap_j, rel=1e-11, abs=5e-15)

    def test_adiabatic_capacitor_relaxation_conserves_stored_electrostatic_energy(self):
        """At I=0 in an adiabatic cell (h_A=0), total temperature rise equals stored 0.5*C*V^2."""
        spec = NMC811_LG_M50_5AH
        model = TwoRCBatteryModel(
            spec,
            h_a_w_k=0.0,  # Adiabatic calorimeter
            enable_soc_impedance=False,
            enable_entropic_heat=False,
            enable_hysteresis=False,
        )
        v1_init, v2_init = 0.14, 0.09
        stored_energy_j = 0.5 * spec.c1_ref_f * (v1_init**2) + 0.5 * spec.c2_ref_f * (v2_init**2)

        state = TwoRCState(soc=0.50, v_rc1_v=v1_init, v_rc2_v=v2_init, t_cell_k=298.15)
        # Coarse 50 s steps over 4000 s (> 16 * tau_2) to prove dt-independent exact conservation
        for _ in range(80):
            state = model.step(state, 0.0, 50.0).state

        thermal_gain_j = spec.mass_kg * spec.cp_j_kgk * (state.t_cell_k - 298.15)
        assert thermal_gain_j == pytest.approx(stored_energy_j, rel=1e-10, abs=1e-12)

    def test_plett_hysteresis_envelope_and_rest_memory_on_lfp(self):
        model = TwoRCBatteryModel(LFP_A123_ANR26650_2P3AH, enable_entropic_heat=False)
        mh = LFP_A123_ANR26650_2P3AH.hysteresis_max_v

        # Discharge from 90% -> 50% SoC drives h -> -1, then rest preserves h
        state = TwoRCState(soc=0.90, t_cell_k=298.15)
        for _ in range(1440):
            state = model.step(state, 2.3, 1.0).state
        assert state.hysteresis == pytest.approx(-1.0, abs=1e-4)

        h_before_rest = state.hysteresis
        for _ in range(600):
            res_dis_rest = model.step(state, 0.0, 1.0)
            state = res_dis_rest.state
        assert state.hysteresis == pytest.approx(h_before_rest, abs=1e-15)

        # Charge from 10% -> 50% SoC drives h -> +1; rested terminal voltage differs by ~2*M_h
        state_chg = TwoRCState(soc=0.10, t_cell_k=298.15)
        for _ in range(1440):
            state_chg = model.step(state_chg, -2.3, 1.0).state
        for _ in range(1800):
            res_chg_rest = model.step(state_chg, 0.0, 1.0)
            state_chg = res_chg_rest.state

        # Relax residual slow RC2 voltage and compare rested terminal potentials at ~50% SoC
        for _ in range(1800):
            res_dis_rest = model.step(state, 0.0, 1.0)
            state = res_dis_rest.state
        assert (res_chg_rest.v_term_v - res_dis_rest.v_term_v) == pytest.approx(2.0 * mh, abs=6e-4)

    def test_bernardi_reversible_heat_endothermic_vs_exothermic_regimes(self):
        model = TwoRCBatteryModel(NMC811_LG_M50_5AH)
        # At SoC = 0.75, NMC811 dU/dT > 0 -> 1C discharge has Q_rev < 0 (endothermic)
        res_high = model.step(TwoRCState(soc=0.75, t_cell_k=298.15), 5.0, 1.0)
        assert res_high.q_rev_w < -0.05
        assert res_high.q_total_w < res_high.q_irrev_w

        # At SoC = 0.08, NMC811 dU/dT < 0 -> 1C discharge has Q_rev > 0 (exothermic)
        res_low = model.step(TwoRCState(soc=0.08, t_cell_k=298.15), 5.0, 1.0)
        assert res_low.q_rev_w > 0.30
        assert res_low.q_total_w > res_low.q_irrev_w

        # Reversing current sign (charging) flips reversible entropic heat sign
        res_chg_high = model.step(TwoRCState(soc=0.75, t_cell_k=298.15), -5.0, 1.0)
        assert res_chg_high.q_rev_w == pytest.approx(-res_high.q_rev_w, rel=2e-3)


class TestEISNyquistAndInputValidation:
    def test_eis_high_and_low_frequency_asymptotes_and_arrhenius_ordering(self):
        model = TwoRCBatteryModel(NMC811_LG_M50_5AH)
        eis_10 = compute_eis_nyquist(model, soc=0.50, t_cell_k=283.15, freq_min_hz=1e-7, freq_max_hz=1e6)
        eis_25 = compute_eis_nyquist(model, soc=0.50, t_cell_k=298.15, freq_min_hz=1e-7, freq_max_hz=1e6)
        eis_45 = compute_eis_nyquist(model, soc=0.50, t_cell_k=318.15, freq_min_hz=1e-7, freq_max_hz=1e6)

        assert eis_25.z_real_ohm[-1] == pytest.approx(eis_25.r0_ohm, rel=1e-5)
        assert eis_25.z_real_ohm[0] == pytest.approx(
            eis_25.r0_ohm + eis_25.r1_ohm + eis_25.r2_ohm, rel=1e-5
        )
        assert np.all(eis_25.neg_z_imag_ohm > 0.0)
        assert eis_10.z_real_ohm[0] > eis_25.z_real_ohm[0] > eis_45.z_real_ohm[0]
        assert eis_25.f_peak_rc1_hz > eis_25.f_peak_rc2_hz

    def test_state_and_model_reject_invalid_inputs(self):
        model = TwoRCBatteryModel(NMC811_LG_M50_5AH)
        with pytest.raises(ValueError, match="soc"):
            TwoRCState(soc=1.2)
        with pytest.raises(ValueError, match="hysteresis"):
            TwoRCState(hysteresis=1.5)
        with pytest.raises(ValueError, match="dt"):
            model.step(TwoRCState(), 1.0, 0.0)
        with pytest.raises(ValueError, match="i_app"):
            model.step(TwoRCState(), float("nan"), 1.0)
        with pytest.raises(ValueError, match="h_a_w_k"):
            TwoRCBatteryModel(NMC811_LG_M50_5AH, h_a_w_k=-0.1)

    def test_chemistry_comparison_benchmark_shapes_and_bounds(self):
        comp = run_chemistry_comparison_benchmark(dt=2.0)
        assert len(comp.nmc_hppc.time_s) == 900
        assert len(comp.lfp_hppc.time_s) == 900
        assert np.all(np.isfinite(comp.nmc_hppc.v_term_v))
        assert np.all(np.isfinite(comp.lfp_hppc.v_term_v))
        assert set(comp.eis_nmc_by_temp_c.keys()) == {10, 25, 45}
