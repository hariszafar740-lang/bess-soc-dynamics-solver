"""ASTM E1049-85 rainflow conservation, virtual-age semigroup & SEI/plating aging tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.cell_chemistry import LFP_A123_ANR26650_2P3AH, NMC811_LG_M50_5AH
from src.degradation_model import (
    LFP_AGING_SPEC,
    NMC811_AGING_SPEC,
    AgingState,
    RainflowCycle,
    SemiEmpiricalAgingModel,
    extract_rainflow_cycles,
    extract_turning_points,
    generate_grid_duty_current,
    get_degradation_spec,
    run_degradation_benchmark,
    simulate_grid_duty_aging,
)


class TestASTME1049RainflowCounting:
    def test_astm_e1049_textbook_9point_nested_history_exact_cycles(self):
        soc = np.array([0.0, 0.8, 0.2, 0.5, 0.1, 0.9, 0.3, 0.7, 0.0], dtype=float)
        t_s = np.arange(len(soc), dtype=float) * 600.0
        cycles = extract_rainflow_cycles(soc, t_s, min_reversal_dod=0.0)

        full_dods = sorted(round(c.dod, 6) for c in cycles if c.count == 1.0)
        half_dods = sorted(round(c.dod, 6) for c in cycles if c.count == 0.5)
        assert full_dods == [0.3, 0.4, 0.7]
        assert half_dods == [0.9, 0.9]

        rainflow_travel = sum(2.0 * c.count * c.dod for c in cycles)
        time_domain_travel = float(np.sum(np.abs(np.diff(soc))))
        assert rainflow_travel == pytest.approx(time_domain_travel, abs=1e-14)

    @pytest.mark.parametrize("seed", [11, 42, 2026])
    def test_rainflow_exact_topological_throughput_identity_on_arbitrary_signals(self, seed):
        rng = np.random.default_rng(seed)
        t_s = np.arange(600, dtype=float) * 5.0
        raw = (
            0.50
            + 0.25 * np.sin(t_s / 380.0)
            + 0.09 * np.sin(t_s / 47.0)
            + np.cumsum(0.008 * rng.standard_normal(600))
        )
        soc = np.clip(raw, 0.05, 0.95)
        soc[120:135] = soc[120]

        cycles = extract_rainflow_cycles(soc, t_s, min_reversal_dod=0.0)
        rainflow_travel = sum(2.0 * c.count * c.dod for c in cycles)
        exact_travel = float(np.sum(np.abs(np.diff(soc))))
        assert rainflow_travel == pytest.approx(exact_travel, rel=1e-13, abs=1e-13)
        assert all(0.0 < c.dod <= 1.0 for c in cycles)
        assert all(0.0 <= c.mean_soc <= 1.0 for c in cycles)
        assert all(c.count in (0.5, 1.0) for c in cycles)

    def test_constant_soc_returns_zero_cycles_and_input_validation(self):
        flat = np.full(50, 0.60, dtype=float)
        assert extract_rainflow_cycles(flat) == []

        with pytest.raises(ValueError, match="at least 2"):
            extract_turning_points(np.array([0.5]))
        with pytest.raises(ValueError, match="non-decreasing"):
            extract_turning_points(np.array([0.2, 0.6, 0.3]), np.array([0.0, 10.0, 5.0]))


class TestVirtualAgeSemigroupAndAgingPhysics:
    @pytest.mark.parametrize("spec", [NMC811_AGING_SPEC, LFP_AGING_SPEC])
    def test_calendar_virtual_age_exact_semigroup_and_closed_form(self, spec):
        model = SemiEmpiricalAgingModel(spec)
        t_core_k = 308.15
        mean_soc = 0.72
        total_days = 1000.0

        one_shot = model.step_block(
            AgingState(), duration_days=total_days, mean_soc=mean_soc, t_core_k=t_core_k
        ).state
        st = AgingState()
        for _ in range(200):
            st = model.step_block(st, duration_days=5.0, mean_soc=mean_soc, t_core_k=t_core_k).state

        expected_q_cal = model.calendar_rate_factor(mean_soc, t_core_k) * (total_days**spec.z_cal)
        assert one_shot.q_loss_cal == pytest.approx(expected_q_cal, rel=1e-13, abs=1e-14)
        assert st.q_loss_cal == pytest.approx(expected_q_cal, rel=1e-13, abs=1e-14)

    @pytest.mark.parametrize("spec", [NMC811_AGING_SPEC, LFP_AGING_SPEC])
    def test_cycle_virtual_age_exact_semigroup_invariance(self, spec):
        model = SemiEmpiricalAgingModel(spec)
        cyc = RainflowCycle(0.70, 0.50, 0.15, 0.85, 1.0, 0.0, 2520.0, 2520.0, 1.0)
        one_batch = model.step_block(
            AgingState(),
            duration_days=0.0,
            mean_soc=0.50,
            t_core_k=298.15,
            cycles=(cyc,),
            repeat_cycles=600.0,
        ).state
        st = AgingState()
        for _ in range(60):
            st = model.step_block(
                st,
                duration_days=0.0,
                mean_soc=0.50,
                t_core_k=298.15,
                cycles=(cyc,),
                repeat_cycles=10.0,
            ).state

        assert st.equivalent_full_cycles == pytest.approx(420.0, abs=1e-12)
        assert st.q_loss_cyc == pytest.approx(one_batch.q_loss_cyc, rel=1e-13, abs=1e-14)
        assert st.q_loss_plating_cum == pytest.approx(one_batch.q_loss_plating_cum, rel=1e-13, abs=1e-14)

    def test_arrhenius_soc_wohler_dod_and_waldmann_v_curve_ordering(self):
        model = SemiEmpiricalAgingModel(NMC811_AGING_SPEC)
        q_10c = model.step_block(AgingState(), duration_days=365.0, mean_soc=0.5, t_core_k=283.15).state.q_loss_cal
        q_25c = model.step_block(AgingState(), duration_days=365.0, mean_soc=0.5, t_core_k=298.15).state.q_loss_cal
        q_45c = model.step_block(AgingState(), duration_days=365.0, mean_soc=0.5, t_core_k=318.15).state.q_loss_cal
        assert q_10c < q_25c < q_45c

        q_soc20 = model.step_block(AgingState(), duration_days=365.0, mean_soc=0.2, t_core_k=298.15).state.q_loss_cal
        q_soc90 = model.step_block(AgingState(), duration_days=365.0, mean_soc=0.9, t_core_k=298.15).state.q_loss_cal
        assert q_soc20 < q_25c < q_soc90

        deep = RainflowCycle(0.80, 0.50, 0.10, 0.90, 1.0, 0.0, 2880.0, 2880.0, 1.0)
        shallow = RainflowCycle(0.08, 0.50, 0.46, 0.54, 1.0, 0.0, 288.0, 288.0, 1.0)
        fade_deep = model.step_block(
            AgingState(), duration_days=100.0, mean_soc=0.50, t_core_k=298.15, cycles=(deep,), repeat_cycles=500.0
        ).state.q_loss_cyc
        fade_shallow = model.step_block(
            AgingState(), duration_days=100.0, mean_soc=0.50, t_core_k=298.15, cycles=(shallow,), repeat_cycles=5000.0
        ).state.q_loss_cyc
        assert fade_deep > 2.5 * fade_shallow

        fade_m5 = model.step_block(
            AgingState(), duration_days=100.0, mean_soc=0.50, t_core_k=268.15, cycles=(deep,), repeat_cycles=500.0
        ).state.q_loss_total
        fade_20 = model.step_block(
            AgingState(), duration_days=100.0, mean_soc=0.50, t_core_k=293.15, cycles=(deep,), repeat_cycles=500.0
        ).state.q_loss_total
        fade_50 = model.step_block(
            AgingState(), duration_days=100.0, mean_soc=0.50, t_core_k=323.15, cycles=(deep,), repeat_cycles=500.0
        ).state.q_loss_total
        assert fade_20 < fade_m5
        assert fade_20 < fade_50

    def test_lfp_outlives_nmc811_and_regd_outlives_deep_arbitrage(self):
        nmc_arb = simulate_grid_duty_aging(NMC811_LG_M50_5AH, "deep_arbitrage", total_years=12.0, step_days=90.0)
        lfp_arb = simulate_grid_duty_aging(LFP_A123_ANR26650_2P3AH, "deep_arbitrage", total_years=12.0, step_days=90.0)
        assert lfp_arb.eol_efc > 2.2 * nmc_arb.eol_efc
        assert lfp_arb.soh_pct[-1] > nmc_arb.soh_pct[-1] + 20.0

        bench = run_degradation_benchmark()
        assert bench.lfp_arb.eol_years > bench.nmc_arb.eol_years + 5.0
        assert bench.nmc_regd.soh_pct[-1] > bench.nmc_arb.soh_pct[-1] + 20.0
        assert 15.0 <= bench.optimal_temp_nmc_c <= 26.0

    def test_rng_isolation_and_parameter_validation(self):
        np.random.seed(777)
        before = np.random.random()
        np.random.seed(777)
        _ = generate_grid_duty_current("regd_freq_reg", capacity_ah=5.0, duration_s=1200.0, dt=5.0)
        after = np.random.random()
        assert before == after

        with pytest.raises(ValueError, match="Unknown degradation"):
            get_degradation_spec("sodium_ion")
        with pytest.raises(ValueError, match="beta_dod"):
            replace(NMC811_AGING_SPEC, beta_dod=0.8)
