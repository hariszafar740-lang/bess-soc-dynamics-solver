"""Literature-pinned lithium-ion cell chemistry records & monotone PCHIP OCV splines.

Provides two experimentally derived open-circuit potential and entropic datasets
extracted at 1% SoC resolution from PyBaMM 26.10.1.1 stoichiometric balances:

1. ``NMC811_LG_M50_5AH`` — LG Chem INR21700-M50 (5.0 Ah, NMC811 / Graphite-SiOx),
   OCV and entropic dU/dT from O'Regan et al., Electrochimica Acta 425, 140700 (2022)
   (PyBaMM parameter set ``"ORegan2022"``).
2. ``LFP_A123_ANR26650_2P3AH`` — A123 Systems ANR26650M1-B (2.3 Ah, LiFePO4 / Graphite),
   equilibrium OCV from Prada et al., J. Electrochem. Soc. 160(4), A616 (2013)
   (PyBaMM parameter set ``"Prada2013"``), paired with Plett one-state hysteresis
   parameters representative of olivine LiFePO4 two-phase plateaus.

Interpolation uses a self-contained Fritsch-Carlson monotone piecewise cubic Hermite
(PCHIP) spline (SIAM J. Numer. Anal. 17(2), 238-246, 1980), guaranteeing C1 continuity,
exact knot reproduction, zero Runge oscillation on flat LFP plateaus, and exact
analytical dOCV/dSoC derivatives for closed-loop Kalman filter Jacobians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.thermal_model import EA_J_MOL, T_REF_K, check_temperature_k

__all__ = [
    "ChemistrySpec",
    "LFP_A123_ANR26650_2P3AH",
    "MonotonePCHIP",
    "NMC811_LG_M50_5AH",
    "get_chemistry_spec",
    "list_chemistry_specs",
]

_SOC_GRID_101: tuple[float, ...] = (
    0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
    0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
    0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
    0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49,
    0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
    0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
    0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
    0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
    0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
    1.0,
)

_NMC811_OREGAN2022_OCV_V: tuple[float, ...] = (
    2.5, 2.70248, 2.843075, 2.94186, 3.012835, 3.066015, 3.10885, 3.147029,
    3.184554, 3.223089, 3.261432, 3.296498, 3.325673, 3.348343, 3.365562, 3.378955,
    3.389992, 3.399763, 3.40901, 3.418225, 3.42773, 3.43773, 3.448329, 3.459536,
    3.471268, 3.483352, 3.495553, 3.507612, 3.519289, 3.530399, 3.540831, 3.55055,
    3.559581, 3.567988, 3.575859, 3.583285, 3.590356, 3.597153, 3.603745, 3.610196,
    3.616557, 3.622875, 3.629196, 3.63556, 3.642012, 3.648598, 3.655369, 3.662384,
    3.669708, 3.677411, 3.685572, 3.694272, 3.703593, 3.713619, 3.72444, 3.736167,
    3.748946, 3.762954, 3.778311, 3.794858, 3.811937, 3.828497, 3.843622, 3.856952,
    3.868639, 3.879062, 3.888606, 3.897586, 3.90624, 3.914746, 3.923241, 3.931837,
    3.940624, 3.94968, 3.959068, 3.968835, 3.97901, 3.989596, 4.000565, 4.011843,
    4.023307, 4.034769, 4.045981, 4.056636, 4.066394, 4.07492, 4.081951, 4.087372,
    4.091302, 4.094144, 4.096595, 4.099573, 4.104079, 4.111016, 4.12102, 4.134344,
    4.150842, 4.170029, 4.1912, 4.21356, 4.236342,
)

_NMC811_OREGAN2022_DUDT_V_PER_K: tuple[float, ...] = (
    -0.00023967, -0.00028435, -0.00032748, -0.00036513, -0.00039347, -0.00040943, -0.0004113,
    -0.00039902, -0.00037422, -0.00033987, -0.00029975, -0.00025777, -0.00021738, -0.00018117,
    -0.00015066, -0.00012638, -0.00010802, -9.479e-05, -8.562e-05, -7.943e-05, -7.526e-05,
    -7.233e-05, -7.007e-05, -6.81e-05, -6.617e-05, -6.412e-05, -6.189e-05, -5.943e-05,
    -5.674e-05, -5.384e-05, -5.073e-05, -4.744e-05, -4.399e-05, -4.042e-05, -3.675e-05,
    -3.301e-05, -2.922e-05, -2.542e-05, -2.162e-05, -1.786e-05, -1.415e-05, -1.051e-05,
    -6.96e-06, -3.51e-06, -1.7e-07, 3.05e-06, 6.13e-06, 9.08e-06, 1.19e-05,
    1.459e-05, 1.714e-05, 1.955e-05, 2.185e-05, 2.402e-05, 2.608e-05, 2.803e-05,
    2.989e-05, 3.165e-05, 3.333e-05, 3.493e-05, 3.646e-05, 3.793e-05, 3.934e-05,
    4.071e-05, 4.203e-05, 4.331e-05, 4.457e-05, 4.579e-05, 4.699e-05, 4.817e-05,
    4.933e-05, 5.047e-05, 5.161e-05, 5.273e-05, 5.385e-05, 5.495e-05, 5.606e-05,
    5.715e-05, 5.825e-05, 5.934e-05, 6.042e-05, 6.151e-05, 6.26e-05, 6.369e-05,
    6.48e-05, 6.596e-05, 6.727e-05, 6.888e-05, 7.109e-05, 7.431e-05, 7.9e-05,
    8.543e-05, 9.343e-05, 0.00010216, 0.00011014, 0.0001157, 0.0001176, 0.00011562,
    0.00011062, 0.00010416, 9.787e-05,
)

_LFP_PRADA2013_OCV_V: tuple[float, ...] = (
    2.0, 2.26547, 2.460021, 2.60276, 2.707735, 2.785319, 2.843246, 2.887373,
    2.922251, 2.951517, 2.978092, 3.004143, 3.030822, 3.057994, 3.084331, 3.108029,
    3.127731, 3.143032, 3.154356, 3.162544, 3.168489, 3.172959, 3.176543, 3.179671,
    3.182651, 3.185703, 3.188982, 3.192593, 3.196598, 3.201015, 3.205821, 3.210951,
    3.216305, 3.221752, 3.227151, 3.232364, 3.237269, 3.241776, 3.245828, 3.2494,
    3.252499, 3.255149, 3.257391, 3.259272, 3.260841, 3.262144, 3.263226, 3.264125,
    3.264874, 3.265501, 3.26603, 3.26648, 3.266867, 3.267203, 3.2675, 3.267765,
    3.268005, 3.268226, 3.268432, 3.26863, 3.268822, 3.269013, 3.269211, 3.269424,
    3.269666, 3.269957, 3.270328, 3.270829, 3.271534, 3.272551, 3.274028, 3.276146,
    3.27908, 3.282913, 3.287534, 3.292575, 3.297501, 3.301826, 3.305286, 3.307863,
    3.309691, 3.310954, 3.311821, 3.312425, 3.312857, 3.31318, 3.313435, 3.313647,
    3.313833, 3.314003, 3.314167, 3.314333, 3.314523, 3.314788, 3.315272, 3.316387,
    3.319306, 3.327379, 3.350173, 3.415021, 3.6,
)


class MonotonePCHIP:
    """Shape-preserving Fritsch-Carlson piecewise cubic Hermite interpolator."""

    def __init__(self, x: np.ndarray | tuple[float, ...], y: np.ndarray | tuple[float, ...]) -> None:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim != 1 or y_arr.ndim != 1 or len(x_arr) != len(y_arr) or len(x_arr) < 3:
            raise ValueError("x and y must be 1D arrays of equal length >= 3")
        if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
            raise ValueError("x and y knots must be finite")
        h = np.diff(x_arr)
        if np.any(h <= 0.0):
            raise ValueError("x knots must be strictly increasing")

        delta = np.diff(y_arr) / h
        n = len(x_arr)
        d = np.zeros(n, dtype=float)

        # Interior slopes: weighted harmonic mean where adjacent secants share sign
        same_sign = (delta[:-1] * delta[1:]) > 0.0
        w1 = 2.0 * h[1:] + h[:-1]
        w2 = h[1:] + 2.0 * h[:-1]
        idx = np.where(same_sign)[0]
        d[idx + 1] = (w1[idx] + w2[idx]) / (w1[idx] / delta[idx] + w2[idx] / delta[idx + 1])

        # Non-centered shape-preserving endpoint slopes (Fritsch-Carlson / Moler)
        d[0] = self._endpoint_slope(h[0], h[1], delta[0], delta[1])
        d[-1] = self._endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])

        self._x = x_arr
        self._y = y_arr
        self._h = h
        self._d = d

    @staticmethod
    def _endpoint_slope(h0: float, h1: float, del0: float, del1: float) -> float:
        d = ((2.0 * h0 + h1) * del0 - h0 * del1) / (h0 + h1)
        if d * del0 <= 0.0:
            return 0.0
        if (del0 * del1 <= 0.0) and (abs(d) > abs(3.0 * del0)):
            return 3.0 * del0
        return float(d)

    def __call__(self, xq: float | np.ndarray) -> float | np.ndarray:
        scalar = np.ndim(xq) == 0
        xq_arr = np.asarray(xq, dtype=float)
        idx = np.clip(np.searchsorted(self._x, xq_arr, side="right") - 1, 0, len(self._h) - 1)
        hk = self._h[idx]
        t = (xq_arr - self._x[idx]) / hk
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        val = (
            h00 * self._y[idx]
            + h10 * hk * self._d[idx]
            + h01 * self._y[idx + 1]
            + h11 * hk * self._d[idx + 1]
        )
        return float(val) if scalar else val

    def derivative(self, xq: float | np.ndarray) -> float | np.ndarray:
        """Exact analytical first derivative dy/dx of the cubic Hermite spline."""
        scalar = np.ndim(xq) == 0
        xq_arr = np.asarray(xq, dtype=float)
        idx = np.clip(np.searchsorted(self._x, xq_arr, side="right") - 1, 0, len(self._h) - 1)
        hk = self._h[idx]
        t = (xq_arr - self._x[idx]) / hk
        t2 = t * t
        dh00 = 6.0 * t2 - 6.0 * t
        dh10 = 3.0 * t2 - 4.0 * t + 1.0
        dh01 = -6.0 * t2 + 6.0 * t
        dh11 = 3.0 * t2 - 2.0 * t
        deriv = (
            (dh00 * self._y[idx] + dh01 * self._y[idx + 1]) / hk
            + dh10 * self._d[idx]
            + dh11 * self._d[idx + 1]
        )
        return float(deriv) if scalar else deriv


@dataclass(frozen=True)
class ChemistrySpec:
    """Immutable electro-thermal and 2RC parameter specification for a Li-ion cell."""

    key: str
    name: str
    chemistry: str
    format_factor: str
    source_citation: str
    pybamm_parameter_set: str
    table_sha256: str
    nominal_capacity_ah: float
    v_min_v: float
    v_max_v: float
    r0_ref_ohm: float
    r1_ref_ohm: float
    c1_ref_f: float
    r2_ref_ohm: float
    c2_ref_f: float
    ea_r0_j_mol: float = EA_J_MOL
    ea_r1_j_mol: float = 24000.0
    ea_r2_j_mol: float = 28000.0
    r_low_soc_gain: float = 0.35
    r_low_soc_decay: float = 12.0
    hysteresis_max_v: float = 0.0
    hysteresis_rate_gamma: float = 0.0
    mass_kg: float = 0.068
    cp_j_kgk: float = 825.0
    h_a_w_k: float = 0.35
    t_ref_k: float = T_REF_K
    soc_grid: tuple[float, ...] = _SOC_GRID_101
    ocv_v_grid: tuple[float, ...] = _NMC811_OREGAN2022_OCV_V
    dudt_v_per_k_grid: tuple[float, ...] = _NMC811_OREGAN2022_DUDT_V_PER_K

    def __post_init__(self) -> None:
        check_temperature_k(self.t_ref_k)
        if self.nominal_capacity_ah <= 0.0 or not math.isfinite(self.nominal_capacity_ah):
            raise ValueError("nominal_capacity_ah must be finite and > 0")
        if self.v_min_v >= self.v_max_v:
            raise ValueError("v_min_v must be strictly less than v_max_v")
        for label, val in (
            ("r0_ref_ohm", self.r0_ref_ohm),
            ("r1_ref_ohm", self.r1_ref_ohm),
            ("c1_ref_f", self.c1_ref_f),
            ("r2_ref_ohm", self.r2_ref_ohm),
            ("c2_ref_f", self.c2_ref_f),
            ("mass_kg", self.mass_kg),
            ("cp_j_kgk", self.cp_j_kgk),
        ):
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{label} must be finite and > 0, got {val!r}")
        if self.hysteresis_max_v < 0.0 or self.hysteresis_rate_gamma < 0.0:
            raise ValueError("hysteresis parameters must be >= 0")
        if len(self.soc_grid) != len(self.ocv_v_grid) or len(self.soc_grid) != len(self.dudt_v_per_k_grid):
            raise ValueError("soc_grid, ocv_v_grid, and dudt_v_per_k_grid must have identical length")
        if np.any(np.diff(np.asarray(self.ocv_v_grid, dtype=float)) < 0.0):
            raise ValueError("ocv_v_grid must be non-decreasing in SoC")


NMC811_LG_M50_5AH = ChemistrySpec(
    key="nmc811_lg_m50",
    name="LG Chem INR21700-M50 (NMC811 / Graphite-SiOx, 5.0 Ah)",
    chemistry="NMC811/Graphite-SiOx",
    format_factor="21700 cylindrical",
    source_citation=(
        "OCV & dU/dT: O'Regan et al., Electrochimica Acta 425, 140700 (2022) "
        "[PyBaMM 'ORegan2022']; 2RC HPPC baseline scales representative of 21700 NMC811."
    ),
    pybamm_parameter_set="ORegan2022",
    table_sha256="0f1c4984836396bb638fa06900196e4b22f82aa0e8d605bd3b4e53756b52e3eb",
    nominal_capacity_ah=5.0,
    v_min_v=2.50,
    v_max_v=4.20,
    r0_ref_ohm=0.018,
    r1_ref_ohm=0.012,
    c1_ref_f=1500.0,
    r2_ref_ohm=0.020,
    c2_ref_f=12000.0,
    ea_r0_j_mol=18000.0,
    ea_r1_j_mol=24000.0,
    ea_r2_j_mol=28000.0,
    r_low_soc_gain=0.40,
    r_low_soc_decay=12.0,
    hysteresis_max_v=0.002,
    hysteresis_rate_gamma=18.0,
    mass_kg=0.068,
    cp_j_kgk=825.0,
    h_a_w_k=0.35,
    soc_grid=_SOC_GRID_101,
    ocv_v_grid=_NMC811_OREGAN2022_OCV_V,
    dudt_v_per_k_grid=_NMC811_OREGAN2022_DUDT_V_PER_K,
)

LFP_A123_ANR26650_2P3AH = ChemistrySpec(
    key="lfp_a123_26650",
    name="A123 Systems ANR26650M1-B (LiFePO4 / Graphite, 2.3 Ah)",
    chemistry="LFP/Graphite",
    format_factor="26650 cylindrical",
    source_citation=(
        "Equilibrium OCV: Prada et al., J. Electrochem. Soc. 160(4), A616 (2013) "
        "[PyBaMM 'Prada2013']; Plett one-state hysteresis & 2RC HPPC scales for 26650 LFP."
    ),
    pybamm_parameter_set="Prada2013",
    table_sha256="7fcb8a99f1bf13d27eb80c6ca78b30e50e3e4f4b0f76bf0e05bef2edb43938f7",
    nominal_capacity_ah=2.3,
    v_min_v=2.00,
    v_max_v=3.60,
    r0_ref_ohm=0.014,
    r1_ref_ohm=0.010,
    c1_ref_f=1200.0,
    r2_ref_ohm=0.015,
    c2_ref_f=12000.0,
    ea_r0_j_mol=16000.0,
    ea_r1_j_mol=22000.0,
    ea_r2_j_mol=26000.0,
    r_low_soc_gain=0.45,
    r_low_soc_decay=14.0,
    hysteresis_max_v=0.018,
    hysteresis_rate_gamma=25.0,
    mass_kg=0.076,
    cp_j_kgk=1100.0,
    h_a_w_k=0.38,
    soc_grid=_SOC_GRID_101,
    ocv_v_grid=_LFP_PRADA2013_OCV_V,
    dudt_v_per_k_grid=tuple(0.0 for _ in _SOC_GRID_101),
)

_REGISTRY: dict[str, ChemistrySpec] = {
    NMC811_LG_M50_5AH.key: NMC811_LG_M50_5AH,
    "nmc811": NMC811_LG_M50_5AH,
    LFP_A123_ANR26650_2P3AH.key: LFP_A123_ANR26650_2P3AH,
    "lfp": LFP_A123_ANR26650_2P3AH,
}


def get_chemistry_spec(key: str) -> ChemistrySpec:
    """Return a registered :class:`ChemistrySpec` by key (case-insensitive)."""
    norm = key.strip().lower()
    if norm not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown chemistry key {key!r}. Available keys: {valid}")
    return _REGISTRY[norm]


def list_chemistry_specs() -> tuple[ChemistrySpec, ...]:
    """Return the canonical built-in cell chemistry specifications."""
    return (NMC811_LG_M50_5AH, LFP_A123_ANR26650_2P3AH)
