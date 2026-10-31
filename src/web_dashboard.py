"""Zero-CDN static HTML5/Canvas interactive BESS, EKF/UKF, 2-node SOP & aging workbench."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .bms_thermal_sop import CoupledTwoNodeBESS, CoupledTwoNodeState, compute_sop_limits
from .cell_chemistry import list_chemistry_specs
from .degradation_model import get_degradation_spec, run_degradation_benchmark
from .ecm_2rc import TwoRCBatteryModel, compute_eis_nyquist
from .soc_estimators import run_single_chemistry_estimation

WORKBENCH_VERSION = "1.0.0"

BENCHMARK_FIGURES: tuple[dict[str, str], ...] = (
    {
        "file": "bess_dynamic_response.png",
        "title": "Figure 1, Coupled 1RC Electro-Thermal Pulsed Discharge (3C @ 25 °C)",
        "caption": (
            "Min V_term = 3.181 V | Peak T_cell = 30.97 °C | Final SoC = 25.0% "
            "(exact ZOH RC + Joule-convective step)."
        ),
    },
    {
        "file": "chemistry_2rc_comparison.png",
        "title": "Figure 2, 2RC Chemistry Comparison: NMC811 vs. LFP, Bernardi Heat & EIS Nyquist",
        "caption": (
            "PyBaMM 26.10.1.1 OCV/entropic PCHIP splines, Plett hysteresis, "
            "reversible cooling/heating, and 10/25/45 °C EIS arcs."
        ),
    },
    {
        "file": "soc_ekf_ukf_benchmark.png",
        "title": "Figure 3, Closed-Loop Nonlinear SoC Estimation: EKF & UKF vs. Biased Coulomb Counting",
        "caption": (
            "Recovers -23% initial SoC offset under +0.05C drift: "
            "NMC811 EKF/UKF = 1.26%/1.28% RMSE; LFP EKF/UKF = 7.19%/6.69%."
        ),
    },
    {
        "file": "thermal_sop_derating.png",
        "title": "Figure 4, 2-Node Radial Core–Surface Thermal Divergence, CC-CV Taper & Horizon SOP",
        "caption": (
            "8.67 °C peak radial gradient, zero-overshoot 2C CC-CV (15.6/26.7 min), "
            "and sub-zero 0 °C Li-plating charge lockout."
        ),
    },
    {
        "file": "degradation_rainflow_aging.png",
        "title": "Figure 5, ASTM E1049-85 Rainflow Decomposition, 12-Year SoH & Waldmann V-Curve",
        "caption": (
            "Deep arbitrage EoL = 4.1 yr (NMC811) vs. 11.0 yr (LFP); "
            "PJM RegD 12-yr SoH = 80.9% / 87.7%; 20 °C fade minimum."
        ),
    },
)


def build_web_presets_payload() -> dict[str, Any]:
    """Export solver-verified chemistry tables, EIS curves, SOP envelopes, and aging presets."""
    deg_bench = run_degradation_benchmark()
    chemistries: dict[str, Any] = {}
    for spec in list_chemistry_specs():
        deg = get_degradation_spec(spec)
        bess_25 = CoupledTwoNodeBESS(spec, t_amb_k=298.15)
        eis_by_temp: dict[str, Any] = {}
        for temp_c in (10, 25, 45):
            t_k = float(temp_c) + 273.15
            ecm_t = TwoRCBatteryModel(spec, t_amb_k=t_k)
            eis = compute_eis_nyquist(ecm_t, soc=0.50, t_cell_k=t_k, n_points=55)
            eis_by_temp[str(temp_c)] = {
                "temp_c": temp_c,
                "r_dc_mohm": round((eis.r0_ohm + eis.r1_ohm + eis.r2_ohm) * 1e3, 3),
                "re_mohm": [round(float(v) * 1e3, 4) for v in eis.z_real_ohm],
                "neg_im_mohm": [round(float(v) * 1e3, 4) for v in eis.neg_z_imag_ohm],
            }

        est = run_single_chemistry_estimation(spec, total_time_s=900.0, dt=2.0, seed=42)
        sop_25 = compute_sop_limits(
            bess_25,
            CoupledTwoNodeState(soc=0.50, t_core_k=298.15, t_surf_k=298.15),
            horizon_s=10.0,
        )
        bess_m5 = CoupledTwoNodeBESS(spec, t_amb_k=268.15)
        sop_m5 = compute_sop_limits(
            bess_m5,
            CoupledTwoNodeState(soc=0.50, t_core_k=268.15, t_surf_k=268.15),
            horizon_s=10.0,
        )
        short_key = "nmc811" if "nmc811" in spec.key else "lfp"
        arb = deg_bench.nmc_arb if short_key == "nmc811" else deg_bench.lfp_arb
        regd = deg_bench.nmc_regd if short_key == "nmc811" else deg_bench.lfp_regd
        chemistries[short_key] = {
            "key": spec.key,
            "short_key": short_key,
            "name": spec.name,
            "chemistry": spec.chemistry,
            "citation": spec.source_citation,
            "capacity_ah": spec.nominal_capacity_ah,
            "v_min_v": spec.v_min_v,
            "v_max_v": spec.v_max_v,
            "r0_ref_ohm": spec.r0_ref_ohm,
            "r1_ref_ohm": spec.r1_ref_ohm,
            "c1_ref_f": spec.c1_ref_f,
            "r2_ref_ohm": spec.r2_ref_ohm,
            "c2_ref_f": spec.c2_ref_f,
            "ea_r0_j_mol": spec.ea_r0_j_mol,
            "ea_r1_j_mol": spec.ea_r1_j_mol,
            "ea_r2_j_mol": spec.ea_r2_j_mol,
            "r_low_soc_gain": spec.r_low_soc_gain,
            "r_low_soc_decay": spec.r_low_soc_decay,
            "hysteresis_max_v": spec.hysteresis_max_v,
            "hysteresis_rate_gamma": spec.hysteresis_rate_gamma,
            "c_core_j_k": round(bess_25.thermal.c_core_j_k, 4),
            "c_surf_j_k": round(bess_25.thermal.c_surf_j_k, 4),
            "r_cs_k_w": spec.r_core_surf_k_w,
            "h_sa_w_k": spec.h_a_w_k,
            "t_ref_k": spec.t_ref_k,
            "soc_grid": [round(float(x), 4) for x in spec.soc_grid],
            "ocv_v_grid": [round(float(x), 6) for x in spec.ocv_v_grid],
            "dudt_v_per_k_grid": [round(float(x), 7) for x in spec.dudt_v_per_k_grid],
            "aging": {
                "k_cal_ref_per_day_z": deg.k_cal_ref_per_day_z,
                "z_cal": deg.z_cal,
                "ea_cal_j_mol": deg.ea_cal_j_mol,
                "alpha_soc_cal": deg.alpha_soc_cal,
                "soc_ref": deg.soc_ref,
                "k_cyc_ref_per_efc_z": deg.k_cyc_ref_per_efc_z,
                "z_cyc": deg.z_cyc,
                "beta_dod": deg.beta_dod,
                "dod_ref": deg.dod_ref,
                "ea_cyc_j_mol": deg.ea_cyc_j_mol,
                "gamma_crate": deg.gamma_crate,
                "crate_ref": deg.crate_ref,
                "k_plat_ref_per_efc_z": deg.k_plat_ref_per_efc_z,
                "ea_plat_j_mol": deg.ea_plat_j_mol,
                "t_plat_ref_k": deg.t_plat_ref_k,
                "gamma_plat_crate": deg.gamma_plat_crate,
                "alpha_plat_soc": deg.alpha_plat_soc,
                "r_cal_gain": deg.r_cal_gain,
                "r_cyc_gain": deg.r_cyc_gain,
                "r_plat_gain": deg.r_plat_gain,
                "eol_soh_threshold": deg.eol_soh_threshold,
            },
            "eis_50soc": eis_by_temp,
            "verified_baselines": {
                "est_900s_ekf_rmse_pct": round(est.rmse_ekf_full * 100.0, 3),
                "est_900s_cc_rmse_pct": round(est.rmse_cc_full * 100.0, 3),
                "sop_10s_25c_p_dis_w": round(sop_25.p_dis_max_w, 3),
                "sop_10s_25c_p_chg_w": round(sop_25.p_chg_max_w, 3),
                "sop_10s_minus5c_p_chg_w": round(sop_m5.p_chg_max_w, 3),
                "sop_10s_minus5c_p_dis_w": round(sop_m5.p_dis_max_w, 3),
                "arb_eol_years": None if arb.eol_years == float("inf") else round(arb.eol_years, 3),
                "arb_12yr_soh_pct": round(float(arb.soh_pct[-1]), 3),
                "regd_12yr_soh_pct": round(float(regd.soh_pct[-1]), 3),
            },
        }

    return {
        "package": "bess-soc-dynamics-solver",
        "namespace": "bess_solver",
        "version": WORKBENCH_VERSION,
        "chemistries": chemistries,
        "figures": list(BENCHMARK_FIGURES),
    }


STYLES_CSS = """/* Zero-CDN BESS Electro-Thermal, EKF, SOP & Degradation Workbench */
:root {
  --bg: #0b1020; --panel: #121a2f; --panel-alt: #17223d; --border: #26355c;
  --text: #e8eefc; --muted: #9fb0d6; --accent: #4ea8ff; --accent-2: #2dd4bf;
}
* { box-sizing: border-box; }
body {
  margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  background: radial-gradient(circle at 15% 0%, #152142 0%, var(--bg) 55%);
  color: var(--text); line-height: 1.5;
}
.container { max-width: 1380px; margin: 0 auto; padding: 24px 20px 56px; }
header.hero {
  display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start;
  gap: 16px; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 22px;
}
.hero h1 { margin: 0 0 6px; font-size: 1.62rem; letter-spacing: -0.02em; }
.hero p { margin: 0; color: var(--muted); max-width: 840px; font-size: 0.94rem; }
.badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.badge {
  font-size: 0.76rem; font-weight: 600; padding: 4px 10px; border-radius: 999px;
  background: var(--panel-alt); border: 1px solid var(--border); color: var(--accent-2);
}
.btn-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
button, a.btn-link {
  cursor: pointer; font-size: 0.84rem; font-weight: 600; padding: 8px 14px;
  border-radius: 8px; border: 1px solid var(--border); background: var(--panel-alt);
  color: var(--text); text-decoration: none;
}
button:hover, a.btn-link:hover { border-color: var(--accent); background: #1d2c50; }
button.primary { background: #1d4ed8; border-color: #3b82f6; }
.controls-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
  gap: 14px; background: var(--panel); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin-bottom: 18px;
}
.control-item label {
  display: flex; justify-content: space-between; font-size: 0.82rem;
  color: var(--muted); margin-bottom: 6px; font-weight: 600;
}
.control-item label span.val { color: var(--accent); font-variant-numeric: tabular-nums; }
.control-item select, .control-item input[type="range"] { width: 100%; }
.control-item select {
  background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 7px 9px; font-size: 0.86rem;
}
.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
  gap: 12px; margin-bottom: 20px;
}
.kpi-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;
}
.kpi-card .kpi-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.kpi-card .kpi-value { font-size: 1.24rem; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
.kpi-card .kpi-sub { font-size: 0.78rem; color: var(--muted); margin-top: 3px; }
.charts-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-bottom: 26px; }
@media (max-width: 960px) { .charts-grid { grid-template-columns: 1fr; } }
.chart-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
.chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; gap: 8px; }
.chart-header h2 { margin: 0; font-size: 0.94rem; font-weight: 600; }
.chart-header .legend-inline { font-size: 0.76rem; color: var(--muted); }
canvas.plot-canvas {
  width: 100%; height: 268px; display: block; border-radius: 8px; background: #090e1c; border: 1px solid #1c2848;
}
.sop-table-wrap {
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px; margin-bottom: 28px; overflow-x: auto;
}
.sop-table-wrap h2 { margin: 0 0 10px; font-size: 1.02rem; }
table.data-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; font-variant-numeric: tabular-nums; }
table.data-table th, table.data-table td {
  text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border);
}
table.data-table th { color: var(--muted); font-weight: 600; }
.pill {
  display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem;
  background: var(--panel-alt); border: 1px solid var(--border);
}
.gallery-section h2 { font-size: 1.18rem; margin: 0 0 6px; }
.gallery-section p.lead { color: var(--muted); margin: 0 0 16px; font-size: 0.9rem; }
.gallery-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
@media (max-width: 560px) { .gallery-grid { grid-template-columns: 1fr; } }
.gallery-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
.gallery-card h3 { margin: 0 0 8px; font-size: 0.94rem; }
.gallery-card img { width: 100%; height: auto; border-radius: 8px; background: #fff; display: block; }
.gallery-card p { margin: 8px 0 0; font-size: 0.82rem; color: var(--muted); }
footer {
  margin-top: 34px; padding-top: 16px; border-top: 1px solid var(--border);
  font-size: 0.82rem; color: var(--muted); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
}
"""

APP_JS = r"""/* Interactive Physics Engine & HiDPI Canvas Renderer, bess_solver v1.0.0 */
(function () {
  "use strict";

  const R_GAS = 8.314462618;
  const PRESETS = JSON.parse(document.getElementById("bess-presets-data").textContent);

  const els = {
    chemistry: document.getElementById("ctrl-chemistry"),
    profile: document.getElementById("ctrl-profile"),
    tempC: document.getElementById("ctrl-temp"),
    cRate: document.getElementById("ctrl-crate"),
    initSoc: document.getElementById("ctrl-soc"),
    estOffset: document.getElementById("ctrl-offset"),
    hSa: document.getElementById("ctrl-hsa"),
    valTemp: document.getElementById("val-temp"),
    valCrate: document.getElementById("val-crate"),
    valSoc: document.getElementById("val-soc"),
    valOffset: document.getElementById("val-offset"),
    valHsa: document.getElementById("val-hsa"),
    btnPanel4: document.getElementById("btn-toggle-panel4"),
    btnExport: document.getElementById("btn-export-json"),
    btnReset: document.getElementById("btn-reset"),
    kpiVmin: document.getElementById("kpi-vmin"),
    kpiVminSub: document.getElementById("kpi-vmin-sub"),
    kpiRmse: document.getElementById("kpi-rmse"),
    kpiRmseSub: document.getElementById("kpi-rmse-sub"),
    kpiTcore: document.getElementById("kpi-tcore"),
    kpiTcoreSub: document.getElementById("kpi-tcore-sub"),
    kpiSop: document.getElementById("kpi-sop"),
    kpiSopSub: document.getElementById("kpi-sop-sub"),
    kpiRdc: document.getElementById("kpi-rdc"),
    kpiRdcSub: document.getElementById("kpi-rdc-sub"),
    kpiEol: document.getElementById("kpi-eol"),
    kpiEolSub: document.getElementById("kpi-eol-sub"),
    panel4Title: document.getElementById("panel4-title"),
    sopBody: document.getElementById("sop-table-body"),
    cVolt: document.getElementById("canvas-voltage"),
    cSoc: document.getElementById("canvas-soc"),
    cTherm: document.getElementById("canvas-thermal"),
    cAging: document.getElementById("canvas-aging"),
  };

  let showNyquist = false;
  let lastTelemetry = null;

  function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }

  function interpGrid(gridX, gridY, x) {
    const xc = clamp(x, gridX[0], gridX[gridX.length - 1]);
    const scaled = xc * (gridX.length - 1);
    const idx = Math.min(Math.floor(scaled), gridX.length - 2);
    const frac = scaled - idx;
    return gridY[idx] * (1.0 - frac) + gridY[idx + 1] * frac;
  }

  function getOcv(spec, soc, tCoreK) {
    return interpGrid(spec.soc_grid, spec.ocv_v_grid, soc) +
      interpGrid(spec.soc_grid, spec.dudt_v_per_k_grid, soc) * (tCoreK - spec.t_ref_k);
  }

  function getDocvDsoc(spec, soc, tCoreK) {
    const sHi = clamp(soc + 0.002, 0.0, 1.0);
    const sLo = clamp(soc - 0.002, 0.0, 1.0);
    return (getOcv(spec, sHi, tCoreK) - getOcv(spec, sLo, tCoreK)) / Math.max(sHi - sLo, 1e-6);
  }

  function getImpedances(spec, soc, tCoreK) {
    const invDiff = 1.0 / tCoreK - 1.0 / spec.t_ref_k;
    const lowSoc = 1.0 + spec.r_low_soc_gain * Math.exp(-spec.r_low_soc_decay * clamp(soc, 0.0, 1.0));
    return {
      r0: spec.r0_ref_ohm * lowSoc * Math.exp((spec.ea_r0_j_mol / R_GAS) * invDiff),
      r1: spec.r1_ref_ohm * lowSoc * Math.exp((spec.ea_r1_j_mol / R_GAS) * invDiff),
      c1: spec.c1_ref_f,
      r2: spec.r2_ref_ohm * lowSoc * Math.exp((spec.ea_r2_j_mol / R_GAS) * invDiff),
      c2: spec.c2_ref_f,
    };
  }

  function buildCurrentProfile(mode, cRate, capAh, nSteps, dt) {
    const iPeak = cRate * capAh;
    const arr = new Float64Array(nSteps);
    for (let k = 0; k < nSteps; k++) {
      const t = k * dt;
      if (mode === "hppc") {
        const cyc = t % 600.0;
        arr[k] = cyc < 180.0 ? iPeak : cyc < 360.0 ? 0.0 : cyc < 480.0 ? -0.65 * iPeak : 0.0;
      } else if (mode === "regd") {
        arr[k] = iPeak * (
          0.58 * Math.sin((2.0 * Math.PI * t) / 220.0) +
          0.32 * Math.sin((2.0 * Math.PI * t) / 53.0 + 0.6) +
          0.14 * Math.cos((2.0 * Math.PI * t) / 19.0)
        );
      } else {
        arr[k] = (t % 450.0) < 300.0 ? iPeak : -0.35 * iPeak;
      }
    }
    return arr;
  }

  function stepPlantAndEkf(spec, params) {
    const dt = 2.0, nSteps = 900;
    const tAmbK = params.tempC + 273.15;
    const qCoul = spec.capacity_ah * 3600.0;
    const iTrue = buildCurrentProfile(params.profile, params.cRate, spec.capacity_ah, nSteps, dt);
    const iBias = 0.05 * spec.capacity_ah;

    let soc = params.initSoc, v1 = 0.0, v2 = 0.0, hState = 0.0;
    let tCoreK = tAmbK, tSurfK = tAmbK;
    let socCc = clamp(params.initSoc + params.estOffset, 0.02, 0.98);
    let x0 = socCc, x1 = 0.0, x2 = 0.0;
    let p00 = 0.06, p01 = 0.0, p02 = 0.0, p11 = 1e-3, p12 = 0.0, p22 = 1e-3;
    const q0 = 2.5e-7, q1 = 1e-6, q2 = 1e-6, rMeas = 6.4e-5;

    const timeMin = new Float64Array(nSteps);
    const vTermArr = new Float64Array(nSteps);
    const uOcvArr = new Float64Array(nSteps);
    const socTrueArr = new Float64Array(nSteps);
    const socCcArr = new Float64Array(nSteps);
    const socEkfArr = new Float64Array(nSteps);
    const tCoreArr = new Float64Array(nSteps);
    const tSurfArr = new Float64Array(nSteps);

    let seed = 20261025;
    function nextNoise() {
      seed = (1664525 * seed + 1013904223) >>> 0;
      const u1 = Math.max((seed + 1) / 4294967297.0, 1e-7);
      seed = (1664525 * seed + 1013904223) >>> 0;
      const u2 = (seed + 1) / 4294967297.0;
      return 0.008 * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    }

    let sumSqEkf = 0.0, sumSqCc = 0.0, minV = 99.0, maxTcoreC = -999.0, maxDeltaTCs = 0.0;

    for (let k = 0; k < nSteps; k++) {
      let iApp = iTrue[k];
      if ((soc <= 0.03 && iApp > 0.0) || (soc >= 0.98 && iApp < 0.0) || (tCoreK <= 273.15 && iApp < 0.0)) {
        iApp = 0.0;
      }
      const imp = getImpedances(spec, soc, tCoreK);
      const uOcv = getOcv(spec, soc, tCoreK);
      const dudt = interpGrid(spec.soc_grid, spec.dudt_v_per_k_grid, soc);
      const vTerm = uOcv + spec.hysteresis_max_v * hState - iApp * imp.r0 - v1 - v2;

      timeMin[k] = (k * dt) / 60.0;
      vTermArr[k] = vTerm;
      uOcvArr[k] = uOcv;
      socTrueArr[k] = soc * 100.0;
      socCcArr[k] = socCc * 100.0;
      socEkfArr[k] = x0 * 100.0;
      tCoreArr[k] = tCoreK - 273.15;
      tSurfArr[k] = tSurfK - 273.15;

      if (vTerm < minV) minV = vTerm;
      if (tCoreArr[k] > maxTcoreC) maxTcoreC = tCoreArr[k];
      if (tCoreK - tSurfK > maxDeltaTCs) maxDeltaTCs = tCoreK - tSurfK;

      sumSqEkf += (x0 - soc) * (x0 - soc);
      sumSqCc += (socCc - soc) * (socCc - soc);

      const iMeas = iApp + iBias;
      const vMeas = vTerm + nextNoise();
      const impE = getImpedances(spec, x0, tCoreK);
      const a1 = Math.exp(-dt / (impE.r1 * impE.c1));
      const a2 = Math.exp(-dt / (impE.r2 * impE.c2));
      const x0Pred = clamp(x0 - (iMeas * dt) / qCoul, 0.005, 0.995);
      const x1Pred = a1 * x1 + impE.r1 * (1.0 - a1) * iMeas;
      const x2Pred = a2 * x2 + impE.r2 * (1.0 - a2) * iMeas;

      p00 += q0; p01 *= a1; p02 *= a2;
      p11 = a1 * a1 * p11 + q1; p12 *= a1 * a2; p22 = a2 * a2 * p22 + q2;

      const impP = getImpedances(spec, x0Pred, tCoreK);
      const h0 = getDocvDsoc(spec, x0Pred, tCoreK);
      const innov = vMeas - (getOcv(spec, x0Pred, tCoreK) - iMeas * impP.r0 - x1Pred - x2Pred);
      const ph0 = h0 * p00 - p01 - p02;
      const ph1 = h0 * p01 - p11 - p12;
      const ph2 = h0 * p02 - p12 - p22;
      const sCov = Math.max(h0 * ph0 - ph1 - ph2 + rMeas, 1e-7);

      if ((innov * innov) / sCov < 25.0) {
        const k0 = ph0 / sCov, k1 = ph1 / sCov, k2 = ph2 / sCov;
        x0 = clamp(x0Pred + k0 * innov, 0.005, 0.995);
        x1 = x1Pred + k1 * innov;
        x2 = x2Pred + k2 * innov;
        p00 = Math.max(p00 - k0 * ph0, 1e-8); p01 -= k0 * ph1; p02 -= k0 * ph2;
        p11 = Math.max(p11 - k1 * ph1, 1e-8); p12 -= k1 * ph2; p22 = Math.max(p22 - k2 * ph2, 1e-8);
      } else {
        x0 = x0Pred; x1 = x1Pred; x2 = x2Pred;
      }
      socCc = clamp(socCc - (iMeas * dt) / qCoul, 0.0, 1.0);

      const e1 = Math.exp(-dt / (imp.r1 * imp.c1));
      const e2 = Math.exp(-dt / (imp.r2 * imp.c2));
      const v1Next = e1 * v1 + imp.r1 * (1.0 - e1) * iApp;
      const v2Next = e2 * v2 + imp.r2 * (1.0 - e2) * iApp;
      if (Math.abs(iApp) > 1e-9 && spec.hysteresis_max_v > 0.0) {
        const eh = Math.exp(-(Math.abs(iApp) * spec.hysteresis_rate_gamma * dt) / qCoul);
        hState = eh * hState + (1.0 - eh) * (iApp > 0.0 ? -1.0 : 1.0);
      }

      const qGen = iApp * iApp * imp.r0 + (v1 * v1) / imp.r1 + (v2 * v2) / imp.r2 - iApp * tCoreK * dudt;
      const dtSub = dt / 4.0;
      for (let m = 0; m < 4; m++) {
        const qCs = (tCoreK - tSurfK) / spec.r_cs_k_w;
        const qSa = params.hSa * (tSurfK - tAmbK);
        tCoreK += (dtSub / spec.c_core_j_k) * (qGen - qCs);
        tSurfK += (dtSub / spec.c_surf_j_k) * (qCs - qSa);
      }
      soc = clamp(soc - (iApp * dt) / qCoul, 0.0, 1.0);
      v1 = v1Next; v2 = v2Next;
    }

    return {
      timeMin: timeMin, vTermArr: vTermArr, uOcvArr: uOcvArr,
      socTrueArr: socTrueArr, socCcArr: socCcArr, socEkfArr: socEkfArr,
      tCoreArr: tCoreArr, tSurfArr: tSurfArr, minV: minV,
      maxTcoreC: maxTcoreC, maxDeltaTCs: maxDeltaTCs,
      rmseEkfPct: Math.sqrt(sumSqEkf / nSteps) * 100.0,
      rmseCcPct: Math.sqrt(sumSqCc / nSteps) * 100.0,
    };
  }

  function evaluateHorizonSop(spec, soc, tempC, hSa) {
    const tK = tempC + 273.15;
    const qCoul = spec.capacity_ah * 3600.0;
    const imp = getImpedances(spec, soc, tK);
    const uOcv = getOcv(spec, soc, tK);
    const iDisHw = 4.0 * spec.capacity_ah;
    const chgScale = tempC <= 0.0 ? 0.0 : tempC >= 15.0 ? 1.0 : tempC / 15.0;
    const iChgHw = 2.0 * spec.capacity_ah * chgScale;

    return [2, 10, 30].map(function (hSec) {
      const rEff = imp.r0 + imp.r1 * (1.0 - Math.exp(-hSec / (imp.r1 * imp.c1))) +
        imp.r2 * (1.0 - Math.exp(-hSec / (imp.r2 * imp.c2)));
      const iDisVolt = Math.max((uOcv - spec.v_min_v) / rEff, 0.0);
      const iDisSoc = Math.max(((soc - 0.05) * qCoul) / hSec, 0.0);
      const dTMax = Math.max(48.0 - tempC, 0.0);
      const iTherm = Math.sqrt(Math.max((dTMax * (spec.c_core_j_k / hSec + 0.35 * hSa)) / rEff, 0.0));

      let iDis = iDisHw, disBot = "current";
      if (iDisVolt < iDis) { iDis = iDisVolt; disBot = "voltage"; }
      if (iDisSoc < iDis) { iDis = iDisSoc; disBot = "soc"; }
      if (iTherm < iDis) { iDis = iTherm; disBot = "thermal_core"; }

      let iChg = iChgHw, chgBot = tempC <= 0.0 ? "li_plating_lockout" : "current";
      if (iChgHw > 0.0) {
        const iChgVolt = Math.max((spec.v_max_v - uOcv) / rEff, 0.0);
        const iChgSoc = Math.max(((0.98 - soc) * qCoul) / hSec, 0.0);
        if (iChgVolt < iChg) { iChg = iChgVolt; chgBot = "voltage"; }
        if (iChgSoc < iChg) { iChg = iChgSoc; chgBot = "soc"; }
        if (iTherm < iChg) { iChg = iTherm; chgBot = "thermal_core"; }
      }
      return {
        horizon_s: hSec,
        i_dis_a: iDis,
        p_dis_w: iDis * Math.max(uOcv - iDis * rEff, spec.v_min_v),
        dis_bottleneck: disBot,
        i_chg_a: iChg,
        p_chg_w: iChg * Math.min(uOcv + iChg * rEff, spec.v_max_v),
        chg_bottleneck: chgBot,
      };
    });
  }

  function project12YrAging(spec, tempC, cRate) {
    const ag = spec.aging, stepDays = 30.0;
    const nPts = Math.floor((12 * 365.25) / stepDays) + 1;
    const yrAxis = new Float64Array(nPts);

    function runMode(dod, meanSoc, chgScale) {
      const out = new Float64Array(nPts);
      let uCal = 0.0, uCyc = 0.0, eolYr = null;
      const tCoreK = tempC + 273.15 + 1.8 * cRate;
      const efcPerDay = 307.0 / 365.25;
      const kCal = ag.k_cal_ref_per_day_z *
        Math.exp(-(ag.ea_cal_j_mol / R_GAS) * (1.0 / tCoreK - 1.0 / 298.15)) *
        Math.exp(ag.alpha_soc_cal * (meanSoc - ag.soc_ref));
      const cEff = Math.max(0.35 * cRate * chgScale, 0.2);
      const sSei = ag.k_cyc_ref_per_efc_z * Math.pow(dod / ag.dod_ref, ag.beta_dod) *
        Math.exp(-(ag.ea_cyc_j_mol / R_GAS) * (1.0 / tCoreK - 1.0 / 298.15)) *
        Math.exp(ag.gamma_crate * (cEff - ag.crate_ref));
      const sPlat = ag.k_plat_ref_per_efc_z *
        Math.exp((ag.ea_plat_j_mol / R_GAS) * (1.0 / tCoreK - 1.0 / ag.t_plat_ref_k)) *
        Math.exp(ag.gamma_plat_crate * Math.max(cEff - 0.5, 0.0)) *
        Math.exp(ag.alpha_plat_soc * Math.max(meanSoc + 0.5 * dod - 0.5, 0.0));
      const rCal = Math.pow(kCal, 1.0 / ag.z_cal);
      const rCyc = Math.pow(sSei + sPlat, 1.0 / ag.z_cyc);

      for (let i = 0; i < nPts; i++) {
        const yr = (i * stepDays) / 365.25;
        yrAxis[i] = yr;
        const soh = Math.max(1.0 - (Math.pow(uCal, ag.z_cal) + Math.pow(uCyc, ag.z_cyc)), 0.35) * 100.0;
        out[i] = soh;
        if (eolYr === null && soh <= ag.eol_soh_threshold * 100.0 && i > 0) {
          const prev = out[i - 1], target = ag.eol_soh_threshold * 100.0;
          eolYr = yrAxis[i - 1] + ((prev - target) / Math.max(prev - soh, 1e-6)) * (yr - yrAxis[i - 1]);
        }
        uCal += rCal * stepDays;
        uCyc += rCyc * (efcPerDay * stepDays);
      }
      return { soh: out, eolYr: eolYr };
    }

    const arb = runMode(0.80, 0.48, 1.0);
    const regd = runMode(0.11, 0.50, 0.75);
    return {
      yrAxis: yrAxis, sohArb: arb.soh, sohRegd: regd.soh,
      arbEolYr: arb.eolYr, regdFinalSoh: regd.soh[nPts - 1],
    };
  }

  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(Math.round(rect.width), 360);
    const h = Math.max(Math.round(rect.height), 240);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }

  function drawLinePlot(canvas, cfg) {
    const surf = setupCanvas(canvas);
    const ctx = surf.ctx;
    const pad = { l: 50, r: 16, t: 18, b: 34 };
    const pw = surf.w - pad.l - pad.r, ph = surf.h - pad.t - pad.b;
    let xMin = cfg.xMin, xMax = cfg.xMax, yMin = cfg.yMin, yMax = cfg.yMax;
    if (yMin === undefined || yMax === undefined) {
      yMin = Infinity; yMax = -Infinity;
      cfg.series.forEach(function (s) {
        for (let i = 0; i < s.y.length; i++) {
          if (s.y[i] < yMin) yMin = s.y[i];
          if (s.y[i] > yMax) yMax = s.y[i];
        }
      });
      const span = Math.max(yMax - yMin, 1e-3);
      yMin -= 0.08 * span; yMax += 0.10 * span;
    }
    function sx(v) { return pad.l + ((v - xMin) / Math.max(xMax - xMin, 1e-9)) * pw; }
    function sy(v) { return pad.t + ph - ((v - yMin) / Math.max(yMax - yMin, 1e-9)) * ph; }

    ctx.strokeStyle = "#1c2a4d"; ctx.lineWidth = 1;
    ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = "#8ea3cf";
    for (let g = 0; g <= 4; g++) {
      const yv = yMin + (g / 4.0) * (yMax - yMin), py = sy(yv);
      ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(pad.l + pw, py); ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(yv.toFixed(cfg.yDecimals !== undefined ? cfg.yDecimals : 2), pad.l - 6, py + 4);
    }
    for (let gx = 0; gx <= 5; gx++) {
      const xv = xMin + (gx / 5.0) * (xMax - xMin), px = sx(xv);
      ctx.beginPath(); ctx.moveTo(px, pad.t); ctx.lineTo(px, pad.t + ph); ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(xv.toFixed(cfg.xDecimals !== undefined ? cfg.xDecimals : 0), px, pad.t + ph + 15);
    }
    if (cfg.hLines) {
      cfg.hLines.forEach(function (hl) {
        if (hl.y >= yMin && hl.y <= yMax) {
          ctx.save(); ctx.strokeStyle = hl.color; ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(pad.l, sy(hl.y)); ctx.lineTo(pad.l + pw, sy(hl.y)); ctx.stroke();
          ctx.restore();
        }
      });
    }
    cfg.series.forEach(function (s) {
      ctx.save(); ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2.0;
      if (s.dash) ctx.setLineDash(s.dash);
      ctx.beginPath();
      for (let i = 0; i < s.x.length; i++) {
        const px = sx(s.x[i]), py = sy(s.y[i]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke(); ctx.restore();
    });
    ctx.fillStyle = "#9fb0d6"; ctx.textAlign = "center";
    ctx.fillText(cfg.xLabel, pad.l + pw / 2, surf.h - 4);
  }

  function updateWorkbench() {
    const spec = PRESETS.chemistries[els.chemistry.value];
    const params = {
      profile: els.profile.value,
      tempC: parseFloat(els.tempC.value),
      cRate: parseFloat(els.cRate.value),
      initSoc: parseFloat(els.initSoc.value) / 100.0,
      estOffset: parseFloat(els.estOffset.value) / 100.0,
      hSa: parseFloat(els.hSa.value),
    };

    els.valTemp.textContent = params.tempC.toFixed(0) + " °C";
    els.valCrate.textContent = params.cRate.toFixed(1) + " C";
    els.valSoc.textContent = Math.round(params.initSoc * 100) + "%";
    els.valOffset.textContent = (params.estOffset >= 0 ? "+" : "") + Math.round(params.estOffset * 100) + "%";
    els.valHsa.textContent = params.hSa.toFixed(1) + " W/K";

    const sim = stepPlantAndEkf(spec, params);
    const sopRows = evaluateHorizonSop(spec, params.initSoc, params.tempC, params.hSa);
    const aging = project12YrAging(spec, params.tempC, params.cRate);
    const impNow = getImpedances(spec, params.initSoc, params.tempC + 273.15);
    const rDcMohm = (impNow.r0 + impNow.r1 + impNow.r2) * 1000.0;

    els.kpiVmin.textContent = sim.minV.toFixed(3) + " V";
    els.kpiVminSub.textContent = "Window [" + spec.v_min_v.toFixed(2) + ", " + spec.v_max_v.toFixed(2) + "] V";
    els.kpiRmse.textContent = sim.rmseEkfPct.toFixed(2) + "%";
    els.kpiRmseSub.textContent = "Biased CC RMSE: " + sim.rmseCcPct.toFixed(1) + "% (+0.05C drift)";
    els.kpiTcore.textContent = sim.maxTcoreC.toFixed(1) + " °C";
    els.kpiTcoreSub.textContent = "Peak ΔT_core-surf: " + sim.maxDeltaTCs.toFixed(2) + " °C";

    const sop10 = sopRows[1];
    els.kpiSop.textContent = sop10.p_dis_w.toFixed(1) + " / " + sop10.p_chg_w.toFixed(1) + " W";
    els.kpiSopSub.textContent = "10s Dis (" + sop10.dis_bottleneck + ") | Chg (" + sop10.chg_bottleneck + ")";
    els.kpiRdc.textContent = rDcMohm.toFixed(1) + " mΩ";
    els.kpiRdcSub.textContent = spec.chemistry + " @ " + params.tempC.toFixed(0) + " °C";
    els.kpiEol.textContent = aging.arbEolYr !== null ? aging.arbEolYr.toFixed(1) + " yr" : ">12.0 yr";
    els.kpiEolSub.textContent = "RegD 12-yr SoH: " + aging.regdFinalSoh.toFixed(1) + "%";

    els.sopBody.innerHTML = sopRows.map(function (r) {
      return "<tr><td>" + r.horizon_s + " s</td><td>" + r.i_dis_a.toFixed(2) + " A</td><td>" +
        r.p_dis_w.toFixed(2) + " W</td><td><span class='pill'>" + r.dis_bottleneck + "</span></td><td>" +
        r.i_chg_a.toFixed(2) + " A</td><td>" + r.p_chg_w.toFixed(2) + " W</td><td><span class='pill'>" +
        r.chg_bottleneck + "</span></td></tr>";
    }).join("");

    drawLinePlot(els.cVolt, {
      xMin: 0, xMax: 30, yDecimals: 2, xDecimals: 0, xLabel: "Simulation Time (minutes)",
      hLines: [{ y: spec.v_min_v, color: "#f87171" }, { y: spec.v_max_v, color: "#f59e0b" }],
      series: [
        { x: sim.timeMin, y: sim.uOcvArr, color: "#94a3b8", width: 1.6, dash: [5, 4] },
        { x: sim.timeMin, y: sim.vTermArr, color: "#4ea8ff", width: 2.2 },
      ],
    });
    drawLinePlot(els.cSoc, {
      xMin: 0, xMax: 30, yMin: 0, yMax: 100, yDecimals: 0, xDecimals: 0, xLabel: "Simulation Time (minutes)",
      series: [
        { x: sim.timeMin, y: sim.socTrueArr, color: "#e2e8f0", width: 2.2 },
        { x: sim.timeMin, y: sim.socCcArr, color: "#f87171", width: 1.7, dash: [6, 4] },
        { x: sim.timeMin, y: sim.socEkfArr, color: "#2dd4bf", width: 2.1 },
      ],
    });
    drawLinePlot(els.cTherm, {
      xMin: 0, xMax: 30, yDecimals: 1, xDecimals: 0, xLabel: "Simulation Time (minutes)",
      hLines: [{ y: 48.0, color: "#f87171" }],
      series: [
        { x: sim.timeMin, y: sim.tCoreArr, color: "#fb923c", width: 2.3 },
        { x: sim.timeMin, y: sim.tSurfArr, color: "#38bdf8", width: 1.9, dash: [5, 3] },
      ],
    });

    if (!showNyquist) {
      els.panel4Title.textContent = "4. 12-Year Rainflow & Virtual-Age SoH Projection (%)";
      drawLinePlot(els.cAging, {
        xMin: 0, xMax: 12, yMin: 55, yMax: 101, yDecimals: 0, xDecimals: 0,
        xLabel: "Operating Time (years @ ~307 EFC/yr)",
        hLines: [{ y: 80.0, color: "#f87171" }],
        series: [
          { x: aging.yrAxis, y: aging.sohRegd, color: "#2dd4bf", width: 2.3 },
          { x: aging.yrAxis, y: aging.sohArb, color: "#f59e0b", width: 2.3, dash: [6, 4] },
        ],
      });
    } else {
      els.panel4Title.textContent = "4. 2RC EIS Nyquist Spectrum @ 50% SoC: -Im(Z) vs Re(Z) (mΩ)";
      const eis10 = spec.eis_50soc["10"], eis25 = spec.eis_50soc["25"], eis45 = spec.eis_50soc["45"];
      drawLinePlot(els.cAging, {
        xMin: 0, xMax: Math.max.apply(null, eis10.re_mohm) * 1.05, yDecimals: 1, xDecimals: 0,
        xLabel: "Re{Z(jω)} (mΩ), 10 °C (blue), 25 °C (teal), 45 °C (amber)",
        series: [
          { x: eis10.re_mohm, y: eis10.neg_im_mohm, color: "#60a5fa", width: 2.2 },
          { x: eis25.re_mohm, y: eis25.neg_im_mohm, color: "#2dd4bf", width: 2.2 },
          { x: eis45.re_mohm, y: eis45.neg_im_mohm, color: "#f59e0b", width: 2.2 },
        ],
      });
    }

    lastTelemetry = {
      solver_version: PRESETS.version,
      chemistry: spec.key,
      parameters: params,
      kpis: {
        min_v_term_v: Number(sim.minV.toFixed(4)),
        ekf_rmse_pct: Number(sim.rmseEkfPct.toFixed(3)),
        cc_rmse_pct: Number(sim.rmseCcPct.toFixed(3)),
        peak_t_core_c: Number(sim.maxTcoreC.toFixed(3)),
        peak_delta_t_cs_c: Number(sim.maxDeltaTCs.toFixed(3)),
        r_dc_mohm: Number(rDcMohm.toFixed(3)),
        arb_eol_years: aging.arbEolYr === null ? null : Number(aging.arbEolYr.toFixed(3)),
        regd_12yr_soh_pct: Number(aging.regdFinalSoh.toFixed(3)),
      },
      sop_horizons: sopRows,
    };
  }

  ["change", "input"].forEach(function (ev) {
    [els.chemistry, els.profile, els.tempC, els.cRate, els.initSoc, els.estOffset, els.hSa].forEach(function (el) {
      el.addEventListener(ev, updateWorkbench);
    });
  });

  els.btnPanel4.addEventListener("click", function () {
    showNyquist = !showNyquist;
    els.btnPanel4.textContent = showNyquist ? "Show 12-Yr SoH Aging" : "Show EIS Nyquist";
    updateWorkbench();
  });

  els.btnReset.addEventListener("click", function () {
    els.chemistry.value = "nmc811"; els.profile.value = "hppc";
    els.tempC.value = "25"; els.cRate.value = "2.0";
    els.initSoc.value = "80"; els.estOffset.value = "-20"; els.hSa.value = "1.1";
    showNyquist = false; els.btnPanel4.textContent = "Show EIS Nyquist";
    updateWorkbench();
  });

  els.btnExport.addEventListener("click", function () {
    if (!lastTelemetry) return;
    const blob = new Blob([JSON.stringify(lastTelemetry, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "bess_workbench_telemetry.json";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  window.addEventListener("resize", updateWorkbench);
  updateWorkbench();
})();
"""


def _render_index_html(presets: dict[str, Any], has_figures: bool) -> str:
    presets_json = json.dumps(presets, separators=(",", ":"), allow_nan=False)
    gallery_cards = []
    for fig in presets["figures"]:
        img_tag = (
            f'<img src="results/{fig["file"]}" alt="{fig["title"]}" loading="lazy">'
            if has_figures
            else ""
        )
        gallery_cards.append(
            f'<article class="gallery-card"><h3>{fig["title"]}</h3>{img_tag}<p>{fig["caption"]}</p></article>'
        )
    gallery_html = "\n".join(gallery_cards)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BESS Electro-Thermal, EKF/UKF, SOP &amp; Degradation Workbench (v{WORKBENCH_VERSION})</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="container">
    <header class="hero">
      <div>
        <h1>Li-Ion BESS Electro-Thermal, EKF, 2-Node SOP &amp; Rainflow Aging Workbench</h1>
        <p>
          Zero-CDN interactive physics explorer powered by <code>bess_solver v{WORKBENCH_VERSION}</code>:
          PyBaMM-parameterized 2RC dual-polarization ECMs (<code>NMC811</code> &amp; <code>LFP</code>),
          closed-loop <code>EKF</code> drift rejection, 2-node radial core–surface thermal &amp; receding-horizon
          <code>SOP</code> derating, and 12-year virtual-age SEI/Wöhler/plating projections.
        </p>
        <div class="badge-row">
          <span class="badge">bess_solver v{WORKBENCH_VERSION}</span>
          <span class="badge">PyBaMM 26.10.1.1 OCV/dUdT Verified</span>
          <span class="badge">Zero External CDNs</span>
          <span class="badge">PEP 561 Typed</span>
        </div>
      </div>
      <div class="btn-row">
        <button id="btn-reset" type="button">Reset Defaults</button>
        <button id="btn-export-json" class="primary" type="button">Export JSON Telemetry</button>
        <a class="btn-link" href="https://github.com/hariszafar740-lang/bess-soc-dynamics-solver"
           target="_blank" rel="noopener">GitHub Repository</a>
      </div>
    </header>

    <section class="controls-grid" aria-label="Simulation Controls">
      <div class="control-item">
        <label for="ctrl-chemistry">Cell Chemistry</label>
        <select id="ctrl-chemistry">
          <option value="nmc811">NMC811, LG INR21700-M50 (5.0 Ah)</option>
          <option value="lfp">LFP, A123 ANR26650M1B (2.3 Ah)</option>
        </select>
      </div>
      <div class="control-item">
        <label for="ctrl-profile">Dispatch Duty Profile</label>
        <select id="ctrl-profile">
          <option value="hppc">Bidirectional HPPC Pulses</option>
          <option value="regd">PJM RegD Frequency Regulation</option>
          <option value="cycling">Sustained Charge/Discharge Cycling</option>
        </select>
      </div>
      <div class="control-item">
        <label for="ctrl-temp">Ambient Temp T_amb <span id="val-temp" class="val">25 °C</span></label>
        <input id="ctrl-temp" type="range" min="-10" max="50" step="1" value="25">
      </div>
      <div class="control-item">
        <label for="ctrl-crate">Peak Current Rate <span id="val-crate" class="val">2.0 C</span></label>
        <input id="ctrl-crate" type="range" min="0.5" max="4.0" step="0.1" value="2.0">
      </div>
      <div class="control-item">
        <label for="ctrl-soc">Initial True SoC <span id="val-soc" class="val">80%</span></label>
        <input id="ctrl-soc" type="range" min="20" max="95" step="1" value="80">
      </div>
      <div class="control-item">
        <label for="ctrl-offset">EKF Init SoC Error <span id="val-offset" class="val">-20%</span></label>
        <input id="ctrl-offset" type="range" min="-30" max="30" step="1" value="-20">
      </div>
      <div class="control-item">
        <label for="ctrl-hsa">Convective Cooling h_sa <span id="val-hsa" class="val">1.1 W/K</span></label>
        <input id="ctrl-hsa" type="range" min="0.3" max="3.0" step="0.1" value="1.1">
      </div>
    </section>

    <section class="kpi-grid" aria-label="Key Performance Indicators">
      <div class="kpi-card">
        <div class="kpi-label">Min Terminal Voltage</div>
        <div id="kpi-vmin" class="kpi-value">--</div>
        <div id="kpi-vmin-sub" class="kpi-sub">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Closed-Loop EKF SoC RMSE</div>
        <div id="kpi-rmse" class="kpi-value">--</div>
        <div id="kpi-rmse-sub" class="kpi-sub">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Peak Jelly-Roll Core T_core</div>
        <div id="kpi-tcore" class="kpi-value">--</div>
        <div id="kpi-tcore-sub" class="kpi-sub">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">10s BMS SOP (Dis / Chg)</div>
        <div id="kpi-sop" class="kpi-value">--</div>
        <div id="kpi-sop-sub" class="kpi-sub">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Total DC Resistance R_dc</div>
        <div id="kpi-rdc" class="kpi-value">--</div>
        <div id="kpi-rdc-sub" class="kpi-sub">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Deep Arbitrage 80% SoH EoL</div>
        <div id="kpi-eol" class="kpi-value">--</div>
        <div id="kpi-eol-sub" class="kpi-sub">--</div>
      </div>
    </section>

    <section class="charts-grid" aria-label="Interactive Simulation Plots">
      <div class="chart-card">
        <div class="chart-header">
          <h2>1. Terminal Voltage V_term(t) vs Equilibrium U_ocv(t)</h2>
          <span class="legend-inline">Solid: V_term | Dashed: U_ocv &amp; V_min/V_max</span>
        </div>
        <canvas id="canvas-voltage" class="plot-canvas"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-header">
          <h2>2. Closed-Loop EKF vs Biased Coulomb Counting SoC (%)</h2>
          <span class="legend-inline">White: True | Teal: EKF | Red: Biased CC</span>
        </div>
        <canvas id="canvas-soc" class="plot-canvas"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-header">
          <h2>3. 2-Node Radial Core &amp; Surface Temperature (°C)</h2>
          <span class="legend-inline">Orange: T_core | Blue: T_surf | Red: 48 °C Limit</span>
        </div>
        <canvas id="canvas-thermal" class="plot-canvas"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-header">
          <h2 id="panel4-title">4. 12-Year Rainflow &amp; Virtual-Age SoH Projection (%)</h2>
          <button id="btn-toggle-panel4" type="button">Show EIS Nyquist</button>
        </div>
        <canvas id="canvas-aging" class="plot-canvas"></canvas>
      </div>
    </section>

    <section class="sop-table-wrap" aria-label="Receding-Horizon State of Power Table">
      <h2>Receding-Horizon BMS State-of-Power (SOP) Envelope &amp; Active Bottlenecks</h2>
      <table class="data-table">
        <thead>
          <tr>
            <th>Horizon</th><th>Max I_dis</th><th>Max P_dis</th><th>Discharge Bottleneck</th>
            <th>Max I_chg</th><th>Max P_chg</th><th>Charge Bottleneck</th>
          </tr>
        </thead>
        <tbody id="sop-table-body"></tbody>
      </table>
    </section>

    <section class="gallery-section" aria-label="Validated Benchmark Gallery">
      <h2>Reproducible Python Solver Benchmark Gallery (main.py)</h2>
      <p class="lead">
        All 5 publication figures are deterministically generated by <code>python main.py</code> and verified in CI
        across Python 3.10–3.13 against analytical invariants and <code>PyBaMM 26.10.1.1</code>.
      </p>
      <div class="gallery-grid">{gallery_html}</div>
    </section>

    <footer>
      <span>MIT License, Haris Zafar (<code>hariszafar740-lang/bess-soc-dynamics-solver</code>)</span>
      <span>Single-cell electro-thermal, BMS SOP &amp; semi-empirical degradation solver (v{WORKBENCH_VERSION})</span>
    </footer>
  </div>

  <script id="bess-presets-data" type="application/json">{presets_json}</script>
  <script src="app.js"></script>
</body>
</html>
"""


def build_static_web_workbench(
    output_dir: str | Path = "site",
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate the standalone zero-CDN GitHub Pages workbench into ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    results_src = root / "results"
    copied_figures: list[str] = []
    if results_src.is_dir():
        results_dst = out / "results"
        results_dst.mkdir(parents=True, exist_ok=True)
        for item in BENCHMARK_FIGURES:
            src_png = results_src / item["file"]
            if src_png.is_file():
                shutil.copy2(src_png, results_dst / item["file"])
                copied_figures.append(item["file"])

    presets = build_web_presets_payload()
    (out / ".nojekyll").write_text("", encoding="utf-8")
    (out / "bess_presets.json").write_text(
        json.dumps(presets, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (out / "app.js").write_text(APP_JS, encoding="utf-8")
    (out / "index.html").write_text(
        _render_index_html(presets, has_figures=len(copied_figures) == len(BENCHMARK_FIGURES)),
        encoding="utf-8",
    )

    return {
        "output_dir": str(out),
        "version": WORKBENCH_VERSION,
        "files_written": [".nojekyll", "index.html", "styles.css", "app.js", "bess_presets.json"],
        "copied_figures": copied_figures,
        "chemistries": sorted(presets["chemistries"].keys()),
    }
