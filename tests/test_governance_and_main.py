"""Governance, public API, and reproducible figure-provenance tests."""

import re
from pathlib import Path

import pytest

import main as main_mod
import src


def test_public_package_exports_and_version():
    assert src.__version__ == "0.3.0"
    for symbol in (
        "LiIonBatteryModel",
        "ThermalBESSModel",
        "LoadProfileGenerator",
        "SimulationError",
        "arrhenius_factor",
        "check_temperature_k",
        "run_thermal_sensitivity_sweep",
        "plot_bess_response",
        "ChemistrySpec",
        "NMC811_LG_M50_5AH",
        "LFP_A123_ANR26650_2P3AH",
        "MonotonePCHIP",
        "get_chemistry_spec",
        "list_chemistry_specs",
        "TwoRCState",
        "TwoRCStepResult",
        "TwoRCBatteryModel",
        "EISResult",
        "compute_eis_nyquist",
        "HPPCBenchmarkResult",
        "ChemistryComparisonResult",
        "run_chemistry_comparison_benchmark",
        "plot_chemistry_2rc_comparison",
    ):
        assert hasattr(src, symbol)


def test_generate_reference_dynamic_figure_reproducible(tmp_path):
    fig_path = tmp_path / "bess_dynamic_response.png"
    metrics = main_mod.generate_reference_dynamic_figure(save_path=str(fig_path))
    assert fig_path.is_file() and fig_path.stat().st_size > 30_000
    assert metrics["min_v_term"] == pytest.approx(3.181, abs=5e-3)
    assert metrics["peak_temp_c"] == pytest.approx(30.97, abs=5e-2)
    assert metrics["final_soc"] == pytest.approx(0.25, abs=1e-6)


def test_generate_chemistry_comparison_figure_reproducible(tmp_path):
    fig_path = tmp_path / "chemistry_2rc_comparison.png"
    metrics = main_mod.generate_chemistry_comparison_figure(save_path=str(fig_path))
    assert fig_path.is_file() and fig_path.stat().st_size > 50_000
    assert metrics["nmc_rdc_25c_mohm"] == pytest.approx(50.0, abs=0.2)
    assert metrics["nmc_rdc_10c_mohm"] == pytest.approx(83.0, abs=0.5)
    assert metrics["nmc_1c_min_q_rev_w"] < -0.10
    assert metrics["nmc_1c_max_q_rev_w"] > 0.50


def test_main_entrypoint_prints_sweep_without_runaway_overclaim(capsys, tmp_path, monkeypatch):
    real_ref = main_mod.generate_reference_dynamic_figure
    real_chem = main_mod.generate_chemistry_comparison_figure
    monkeypatch.setattr(
        main_mod,
        "generate_reference_dynamic_figure",
        lambda: real_ref(save_path=str(tmp_path / "fig1.png")),
    )
    monkeypatch.setattr(
        main_mod,
        "generate_chemistry_comparison_figure",
        lambda: real_chem(save_path=str(tmp_path / "fig2.png")),
    )
    main_mod.main()
    out = capsys.readouterr().out
    assert "BESS ELECTRO-THERMAL PARAMETER SENSITIVITY" in out
    assert "2RC Chemistry Benchmark (NMC811 vs LFP)" in out
    assert "DAY 14" not in out
    assert "THERMAL RUNAWAY RISK" not in out


def test_governance_files_and_readme_asset_integrity():
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "LICENSE",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        ".gitignore",
        ".github/workflows/ci.yml",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        "requirements-validation.txt",
    ):
        assert (root / rel).is_file(), f"Missing governance file: {rel}"

    readme = (root / "README.md").read_text(encoding="utf-8")
    local_imgs = [p for p in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme) if not p.startswith("http")]
    assert "results/bess_dynamic_response.png" in local_imgs
    assert "results/chemistry_2rc_comparison.png" in local_imgs
    for img_rel in local_imgs:
        img_file = root / img_rel
        assert img_file.is_file() and img_file.stat().st_size > 30_000
