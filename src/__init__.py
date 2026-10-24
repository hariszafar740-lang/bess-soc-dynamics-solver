"""Battery Energy Storage System (BESS) Electro-Thermal & SoC Dynamics Solver."""

from src.bess_model import SOC_TOL, LiIonBatteryModel
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
from src.visualizer import plot_bess_response

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "EA_J_MOL",
    "LiIonBatteryModel",
    "LoadProfileGenerator",
    "R_GAS",
    "SOC_TOL",
    "SimulationError",
    "T_MAX_K",
    "T_MIN_K",
    "T_REF_K",
    "ThermalBESSModel",
    "arrhenius_factor",
    "check_temperature_k",
    "plot_bess_response",
    "run_thermal_sensitivity_sweep",
]
