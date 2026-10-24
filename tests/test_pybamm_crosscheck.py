"""Independent cross-validation against PyBaMM 26.10.1.1 (ORegan2022, Prada2013, 2RC Thevenin)."""

from __future__ import annotations

import numpy as np
import pytest

pybamm = pytest.importorskip("pybamm")

from src.cell_chemistry import (  # noqa: E402
    LFP_A123_ANR26650_2P3AH,
    NMC811_LG_M50_5AH,
    ChemistrySpec,
)
from src.ecm_2rc import TwoRCBatteryModel, TwoRCState  # noqa: E402


@pytest.mark.parametrize("spec", [NMC811_LG_M50_5AH, LFP_A123_ANR26650_2P3AH])
def test_stoichiometric_ocv_and_entropic_tables_match_pybamm_parameter_sets(spec: ChemistrySpec):
    """Verify pinned 101-point OCV and dU/dT arrays match PyBaMM electrode balances."""
    pv = pybamm.ParameterValues(spec.pybamm_parameter_set)
    ocv_ref, dudt_ref = [], []
    for s in spec.soc_grid:
        x, y = pybamm.lithium_ion.get_initial_stoichiometries(float(s), pv)
        u_p = float(np.squeeze(pv.evaluate(pv["Positive electrode OCP [V]"](pybamm.Scalar(y)))))
        u_n = float(np.squeeze(pv.evaluate(pv["Negative electrode OCP [V]"](pybamm.Scalar(x)))))
        ocv_ref.append(u_p - u_n)

        dp = pv["Positive electrode OCP entropic change [V.K-1]"]
        dn = pv["Negative electrode OCP entropic change [V.K-1]"]
        val_dp = float(np.squeeze(pv.evaluate(dp(pybamm.Scalar(y))))) if callable(dp) else float(dp)
        val_dn = float(np.squeeze(pv.evaluate(dn(pybamm.Scalar(x))))) if callable(dn) else float(dn)
        dudt_ref.append(val_dp - val_dn)

    np.testing.assert_allclose(np.asarray(spec.ocv_v_grid), np.asarray(ocv_ref), rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(np.asarray(spec.dudt_v_per_k_grid), np.asarray(dudt_ref), rtol=0.0, atol=5e-8)


def test_dynamic_2rc_electro_thermal_trajectory_matches_pybamm_thevenin():
    """Cross-check TwoRCBatteryModel against pybamm.equivalent_circuit.Thevenin(2RC)."""
    # Isothermal R0/R1/R2 (ea=0) with constant entropic coefficient to compare ODE trajectories directly
    dudt_const = -2.5e-4  # V/K
    spec_iso = ChemistrySpec(
        key="nmc_iso_test",
        name="Isothermal-Impedance NMC811 Verification Cell",
        chemistry="NMC811",
        format_factor="21700",
        source_citation="PyBaMM Thevenin 2RC cross-check",
        pybamm_parameter_set="ORegan2022",
        table_sha256="crosscheck",
        nominal_capacity_ah=5.0,
        v_min_v=2.5,
        v_max_v=4.3,
        r0_ref_ohm=0.018,
        r1_ref_ohm=0.012,
        c1_ref_f=1500.0,
        r2_ref_ohm=0.020,
        c2_ref_f=12000.0,
        ea_r0_j_mol=0.0,
        ea_r1_j_mol=0.0,
        ea_r2_j_mol=0.0,
        r_low_soc_gain=0.0,
        hysteresis_max_v=0.0,
        mass_kg=0.068,
        cp_j_kgk=825.0,
        h_a_w_k=0.35,
        soc_grid=NMC811_LG_M50_5AH.soc_grid,
        ocv_v_grid=NMC811_LG_M50_5AH.ocv_v_grid,
        dudt_v_per_k_grid=tuple(dudt_const for _ in NMC811_LG_M50_5AH.soc_grid),
    )
    our_model = TwoRCBatteryModel(
        spec_iso,
        enable_soc_impedance=False,
        enable_entropic_heat=True,
        enable_hysteresis=False,
        polarization_heat_mode="overpotential",
    )

    # Note: PyBaMM's Thevenin model evaluates Open-circuit voltage [V] as U_ref(SoC)
    # without adding (T - T_ref)*dU/dT to terminal voltage, while including -I*T*dU/dT in thermal RHS.
    m_pb = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})
    pv_pb = m_pb.default_parameter_values.copy()
    soc_np = np.asarray(spec_iso.soc_grid, dtype=float)
    ocv_np = np.asarray(spec_iso.ocv_v_grid, dtype=float)

    def ocv_pb(sto):
        return pybamm.Interpolant(soc_np, ocv_np, sto, "ocv_pb")

    i_discharge = 5.0
    mcp = spec_iso.mass_kg * spec_iso.cp_j_kgk
    pv_pb.update(
        {
            "Initial SoC": 0.90,
            "Initial temperature [K]": 298.15,
            "Ambient temperature [K]": 298.15,
            "Cell capacity [A.h]": 5.0,
            "Nominal cell capacity [A.h]": 5.0,
            "Current function [A]": i_discharge,
            "Upper voltage cut-off [V]": 4.4,
            "Lower voltage cut-off [V]": 2.4,
            "Open-circuit voltage [V]": ocv_pb,
            "R0 [Ohm]": spec_iso.r0_ref_ohm,
            "R1 [Ohm]": spec_iso.r1_ref_ohm,
            "C1 [F]": spec_iso.c1_ref_f,
            "R2 [Ohm]": spec_iso.r2_ref_ohm,
            "C2 [F]": spec_iso.c2_ref_f,
            "Element-1 initial overpotential [V]": 0.0,
            "Element-2 initial overpotential [V]": 0.0,
            "Entropic change [V/K]": dudt_const,
            "Cell thermal mass [J/K]": mcp,
            "Cell-jig heat transfer coefficient [W/K]": spec_iso.h_a_w_k,
            "Jig thermal mass [J/K]": 1e12,
            "Jig-air heat transfer coefficient [W/K]": 1e9,
        },
        check_already_exists=False,
    )

    dt = 0.5
    n_steps = 1200  # 600 s 1C discharge
    t_eval = np.arange(n_steps + 1, dtype=float) * dt
    sol = pybamm.Simulation(m_pb, parameter_values=pv_pb).solve(t_eval)

    t_query = t_eval[1:]
    pb_soc = np.asarray(sol["SoC"](t_query), dtype=float)
    pb_v1 = -np.asarray(sol["Element-1 overpotential [V]"](t_query), dtype=float)
    pb_v2 = -np.asarray(sol["Element-2 overpotential [V]"](t_query), dtype=float)
    pb_t_k = np.asarray(sol["Cell temperature [K]"](t_query), dtype=float)
    pb_v_term = np.asarray(sol["Voltage [V]"](t_query), dtype=float)

    traj = our_model.simulate(
        np.full(n_steps, i_discharge),
        dt=dt,
        initial_state=TwoRCState(soc=0.90, t_cell_k=298.15),
    )
    # Subtract (T - T_ref)*dU/dT to compare directly against PyBaMM's isothermal-OCV convention
    our_v_iso_ocv = traj.v_term_v - ((traj.t_cell_c + 273.15) - 298.15) * dudt_const

    np.testing.assert_allclose(traj.soc, pb_soc, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(traj.v_rc1_v, pb_v1, rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(traj.v_rc2_v, pb_v2, rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(our_v_iso_ocv, pb_v_term, rtol=0.0, atol=4e-4)
    np.testing.assert_allclose(traj.t_cell_c + 273.15, pb_t_k, rtol=0.0, atol=2e-3)
