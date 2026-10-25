"""PEP 621 wheel layout, PEP 561 py.typed, namespace hygiene & `bess-sim` CLI tests."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import bess_solver
import src
from bess_solver.cli import CLI_VERSION, main


class TestPackageNamespaceAndWheelHygiene:
    def test_public_namespace_version_and_py_typed_marker(self):
        assert bess_solver.__version__ == "0.7.0"
        assert src.__version__ == "0.7.0"
        assert CLI_VERSION == bess_solver.__version__
        assert set(bess_solver.__all__) == set(src.__all__)

        pkg_dir = Path(bess_solver.__file__).resolve().parent
        assert (pkg_dir / "py.typed").is_file()

    def test_no_hardcoded_src_or_main_imports_inside_package_modules(self):
        src_dir = Path(__file__).resolve().parents[1] / "src"
        py_files = sorted(src_dir.glob("*.py"))
        assert len(py_files) >= 10
        forbidden_src = re.compile(r"^\s*(?:from|import)\s+src\b", re.M)
        forbidden_main = re.compile(r"^\s*(?:from|import)\s+main\b", re.M)
        for path in py_files:
            text = path.read_text(encoding="utf-8")
            assert not forbidden_src.search(text), f"Hardcoded `src` import in {path.name}"
            assert not forbidden_main.search(text), f"Top-level `main` import in {path.name}"

    def test_built_wheel_maps_exclusively_to_bess_solver_and_runs_standalone(self, tmp_path):
        import os

        from setuptools import build_meta

        root = Path(__file__).resolve().parents[1]
        whl_dir = tmp_path / "dist"
        whl_dir.mkdir()

        prev_cwd = Path.cwd()
        try:
            os.chdir(root)
            whl_name = build_meta.build_wheel(str(whl_dir))
        finally:
            os.chdir(prev_cwd)

        whl_path = whl_dir / whl_name
        assert whl_path.is_file()

        with zipfile.ZipFile(whl_path) as zf:
            names = zf.namelist()

        assert "bess_solver/__init__.py" in names
        assert "bess_solver/cli.py" in names
        assert "bess_solver/__main__.py" in names
        assert "bess_solver/py.typed" in names
        assert not any(n.startswith("src/") for n in names), f"Wheel leaked src/: {names}"

        run_dir = tmp_path / "isolated_cwd"
        run_dir.mkdir()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(whl_path)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "bess_solver",
                "degrade",
                "--chemistry",
                "lfp",
                "--mode",
                "regd_freq_reg",
                "--years",
                "4",
                "--step-days",
                "120",
                "--json",
            ],
            cwd=str(run_dir),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["chemistry"] == "lfp_a123_26650"
        assert data["eol_years"] is None
        assert data["final_soh_pct"] > 90.0


class TestBessSimCLISubcommands:
    def test_cli_info_json_and_human_output(self, capsys):
        assert main(["info", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["package"] == "bess-soc-dynamics-solver"
        assert payload["namespace"] == "bess_solver"
        assert payload["version"] == "0.7.0"
        assert len(payload["chemistries"]) == 2

        assert main(["info"]) == 0
        human = capsys.readouterr().out
        assert "nmc811_lg_m50" in human
        assert "lfp_a123_26650" in human

    def test_cli_pulse_json_and_plot_export(self, capsys, tmp_path):
        fig = tmp_path / "pulse.png"
        rc = main(
            [
                "pulse",
                "--c-rate",
                "3.0",
                "--temp-c",
                "25.0",
                "--plot",
                str(fig),
                "--json",
            ]
        )
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["min_v_term_v"] == pytest.approx(3.181, abs=5e-3)
        assert data["peak_temp_c"] == pytest.approx(30.97, abs=5e-2)
        assert data["final_soc"] == pytest.approx(0.25, abs=1e-6)
        assert fig.is_file() and fig.stat().st_size > 30_000

    def test_cli_ecm_and_estimate_json(self, capsys):
        assert main(["ecm", "--chemistry", "both", "--soc", "0.5", "--temp-c", "25", "--json"]) == 0
        ecm = json.loads(capsys.readouterr().out)
        assert len(ecm["results"]) == 2
        assert ecm["results"][0]["r_dc_mohm"] == pytest.approx(50.05, abs=0.2)

        rc = main(
            [
                "estimate",
                "--chemistry",
                "nmc811",
                "--total-time-s",
                "900",
                "--json",
            ]
        )
        assert rc == 0
        est = json.loads(capsys.readouterr().out)
        row = est["results"][0]
        assert row["rmse_ekf_pct"] < row["rmse_cc_pct"]
        assert row["rmse_ukf_pct"] < row["rmse_cc_pct"]

        assert main(["estimate", "--chemistry", "nmc811", "--total-time-s", "120", "--json"]) == 0
        est_short = json.loads(capsys.readouterr().out)
        assert est_short["results"][0]["rmse_ekf_post_300s_pct"] is None

    def test_cli_sop_room_temp_and_subzero_charge_lockout(self, capsys):
        assert main(["sop", "--chemistry", "nmc811", "--soc", "0.5", "--temp-c", "25", "--json"]) == 0
        warm = json.loads(capsys.readouterr().out)
        p_dis = [h["p_dis_max_w"] for h in warm["horizons"]]
        assert p_dis[0] >= p_dis[1] >= p_dis[2] > 50.0

        assert main(["sop", "--chemistry", "nmc811", "--soc", "0.5", "--temp-c", "-5", "--json"]) == 0
        cold = json.loads(capsys.readouterr().out)
        for h in cold["horizons"]:
            assert h["i_chg_max_a"] == 0.0
            assert h["p_chg_max_w"] == 0.0
            assert h["p_dis_max_w"] > 40.0

    def test_cli_degrade_and_sweep_strict_rfc8259_json(self, capsys):
        rc = main(
            [
                "degrade",
                "--chemistry",
                "nmc811",
                "--mode",
                "deep_arbitrage",
                "--years",
                "12",
                "--step-days",
                "90",
                "--json",
            ]
        )
        assert rc == 0
        deg = json.loads(capsys.readouterr().out)
        assert 3.0 < deg["eol_years"] < 5.5
        assert deg["final_delta_r_pct"] > 50.0

        assert main(["sweep", "--temps-c", "10,25", "--c-rates", "1,3", "--json"]) == 0
        swp = json.loads(capsys.readouterr().out)
        assert len(swp["cases"]) == 4

    def test_cli_human_readable_modes_and_argument_validation(self, capsys):
        for argv in [
            ["pulse", "--c-rate", "2.0", "--total-time-s", "300", "--dt", "5.0"],
            ["ecm", "--chemistry", "nmc811", "--soc", "0.5", "--temp-c", "25"],
            ["estimate", "--chemistry", "nmc811", "--total-time-s", "300", "--dt", "1.0"],
            ["sop", "--chemistry", "lfp", "--soc", "0.5", "--temp-c", "25"],
            ["degrade", "--chemistry", "lfp", "--mode", "regd_freq_reg", "--years", "2.0", "--step-days", "120"],
            ["sweep", "--temps-c", "10,25", "--c-rates", "1,2"],
        ]:
            assert main(argv) == 0
            out = capsys.readouterr().out
            assert len(out.strip()) > 20

        with pytest.raises(SystemExit) as exc_soc:
            main(["sop", "--soc", "1.5"])
        assert exc_soc.value.code == 2

        with pytest.raises(SystemExit) as exc_dt:
            main(["pulse", "--dt", "-0.5"])
        assert exc_dt.value.code == 2
