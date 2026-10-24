"""Battery Energy Storage System (BESS) Electro-Thermal & SoC Dynamics Solver."""

from src.bess_model import SOC_TOL, LiIonBatteryModel
from src.cell_chemistry import (
    LFP_A123_ANR26650_2P3AH,
    NMC811_LG_M50_5AH,
    ChemistrySpec,
    MonotonePCHIP,
    get_chemistry_spec,
    list_chemistry_specs,
)
from src.ecm_2rc import (
    ChemistryComparisonResult,
    EISResult,
    HPPCBenchmarkResult,
    TwoRCBatteryModel,
    TwoRCState,
    TwoRCStepResult,
    compute_eis_nyquist,
    run_chemistry_comparison_benchmark,
)
from src.profile_generator import LoadProfileGenerator
from src.sensitivity_analysis import SimulationError, run_thermal_sensitivity_sweep
from src.thermal_model import (
    EA_J_MOL,
    R_GAS,
    T_MAX_K,
    T_MIN_K,
    T_REF_K,
    ThermalBESSModel,
    arrhenius_factor,
    check_temperature_k,
)
from src.visualizer import plot_bess_response, plot_chemistry_2rc_comparison

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "ChemistryComparisonResult",
    "ChemistrySpec",
    "EA_J_MOL",
    "EISResult",
    "HPPCBenchmarkResult",
    "LFP_A123_ANR26650_2P3AH",
    "LiIonBatteryModel",
    "LoadProfileGenerator",
    "MonotonePCHIP",
    "NMC811_LG_M50_5AH",
    "R_GAS",
    "SOC_TOL",
    "SimulationError",
    "T_MAX_K",
    "T_MIN_K",
    "T_REF_K",
    "ThermalBESSModel",
    "TwoRCBatteryModel",
    "TwoRCState",
    "TwoRCStepResult",
    "arrhenius_factor",
    "check_temperature_k",
    "compute_eis_nyquist",
    "get_chemistry_spec",
    "list_chemistry_specs",
    "plot_bess_response",
    "plot_chemistry_2rc_comparison",
    "run_chemistry_comparison_benchmark",
    "run_thermal_sensitivity_sweep",
]
