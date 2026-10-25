"""Tests for the zero-CDN static HTML5/Canvas BESS workbench & `bess-sim web` CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path

from bess_solver import (
    WORKBENCH_VERSION,
    build_static_web_workbench,
    build_web_presets_payload,
)
from bess_solver.cli import main


class TestStaticWebWorkbench:
    def test_presets_payload_matches_python_solver_invariants_and_strict_json(self):
        payload = build_web_presets_payload()
        assert payload["version"] == "1.0.0"
        assert WORKBENCH_VERSION == "1.0.0"
        raw = json.dumps(payload, allow_nan=False)
        assert "NaN" not in raw and "Infinity" not in raw

        nmc = payload["chemistries"]["nmc811"]
        lfp = payload["chemistries"]["lfp"]
        assert len(nmc["soc_grid"]) == 101
        assert len(lfp["ocv_v_grid"]) == 101
        assert nmc["verified_baselines"]["sop_10s_minus5c_p_chg_w"] == 0.0
        assert nmc["verified_baselines"]["sop_10s_minus5c_p_dis_w"] > 45.0
        assert 3.5 < nmc["verified_baselines"]["arb_eol_years"] < 5.0
        assert lfp["verified_baselines"]["arb_eol_years"] > nmc["verified_baselines"]["arb_eol_years"] + 4.0

    def test_build_static_bundle_zero_cdn_and_dom_id_integrity(self, tmp_path):
        summary = build_static_web_workbench(output_dir=tmp_path / "site")
        site_dir = Path(summary["output_dir"])
        for fname in (".nojekyll", "index.html", "styles.css", "app.js", "bess_presets.json"):
            assert (site_dir / fname).is_file(), f"Missing {fname}"
        assert len(summary["copied_figures"]) == 5
        for png in summary["copied_figures"]:
            assert (site_dir / "results" / png).stat().st_size > 30_000

        html = (site_dir / "index.html").read_text(encoding="utf-8")
        css = (site_dir / "styles.css").read_text(encoding="utf-8")
        js = (site_dir / "app.js").read_text(encoding="utf-8")

        assert not re.search(r'<script[^>]+src=["\']https?://', html, re.I)
        assert not re.search(r'<link[^>]+href=["\']https?://', html, re.I)
        assert "https://" not in css and "http://" not in css and "@import" not in css
        assert "fetch(" not in js

        html_ids = set(re.findall(r'\bid="([^"]+)"', html))
        js_ids = set(re.findall(r'getElementById\("([^"]+)"\)', js))
        assert len(js_ids) >= 20
        missing = js_ids - html_ids
        assert not missing, f"Missing DOM IDs in index.html: {missing}"

    def test_cli_web_subcommand_json_and_human_modes(self, capsys, tmp_path):
        out_json = tmp_path / "web_json"
        assert main(["web", "--output-dir", str(out_json), "--json"]) == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["version"] == "1.0.0"
        assert (out_json / "index.html").is_file()

        out_txt = tmp_path / "web_txt"
        assert main(["web", "--output-dir", str(out_txt)]) == 0
        msg = capsys.readouterr().out
        assert str(out_txt) in msg
        assert (out_txt / "bess_presets.json").is_file()
