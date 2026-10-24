"""Governance, public API, and reproducible figure-provenance tests (Module 1)."""

import re
from pathlib import Path

import pytest

import main as main_mod
import src


def test_public_package_exports_and_version():
    assert src.__version__ == "0.2.0"
    for symbol in (
        "LiIonBatteryModel",
        "ThermalBESSModel",
        "LoadProfileGenerator",
        "SimulationError",
        "arrhenius_factor",
        "check_temperature_k",
        "run_thermal_sensitivity_sweep",
        "plot_bess_response",
    ):
        assert hasattr(src, symbol)


def test_generate_reference_dynamic_figure_reproducible(tmp_path):
    fig_path = tmp_path / "bess_dynamic_response.png"
    metrics = main_mod.generate_reference_dynamic_figure(save_path=str(fig_path))
    assert fig_path.is_file() and fig_path.stat().st_size > 30_000
    assert metrics["min_v_term"] == pytest.approx(3.181, abs=5e-3)
    assert metrics["peak_temp_c"] == pytest.approx(30.97, abs=5e-2)
    assert metrics["final_soc"] == pytest.approx(0.25, abs=1e-6)


def test_main_entrypoint_prints_sweep_without_runaway_overclaim(capsys, tmp_path, monkeypatch):
    real_fn = main_mod.generate_reference_dynamic_figure
    monkeypatch.setattr(
        main_mod,
        "generate_reference_dynamic_figure",
        lambda: real_fn(save_path=str(tmp_path / "fig.png")),
    )
    main_mod.main()
    out = capsys.readouterr().out
    assert "BESS ELECTRO-THERMAL PARAMETER SENSITIVITY" in out
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
    ):
        assert (root / rel).is_file(), f"Missing governance file: {rel}"

    readme = (root / "README.md").read_text(encoding="utf-8")
    local_imgs = [p for p in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme) if not p.startswith("http")]
    assert "results/bess_dynamic_response.png" in local_imgs
    for img_rel in local_imgs:
        img_file = root / img_rel
        assert img_file.is_file() and img_file.stat().st_size > 30_000
