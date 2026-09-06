from src.sensitivity_analysis import run_thermal_sensitivity_sweep

def main():
    print("=" * 75)
    print("DAY 14: BESS PARAMETER SENSITIVITY & THERMAL RUNAWAY RISK ANALYSIS")
    print("=" * 75)
    print(f"{'Ambient (°C)':<15}{'C-Rate':<12}{'Min V_term (V)':<18}{'Peak Temp (°C)':<18}")
    print("-" * 75)
    
    results = run_thermal_sensitivity_sweep()
    for res in results:
        print(f"{res['amb_temp_c']:<15}{res['c_rate']:<12.1f}{res['min_v_term']:<18.3f}{res['peak_temp_c']:<18.2f}")
    print("=" * 75)

if __name__ == "__main__":
    main()
