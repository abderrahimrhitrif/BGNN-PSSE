import time
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from tqdm import tqdm

def setup_observable_measurements(net):
    """
    Ensures the system is observable by adding V, P, and Q measurements.
    This allows 'Full WLS' to converge.
    """
    # Clear existing measurements
    net.measurement = net.measurement.iloc[0:0]
    
    # Add voltage measurements to all buses (Standard Deviation 0.01 pu)
    for bus in net.bus.index:
        pp.create_measurement(net, "v", "bus", net.res_bus.at[bus, "vm_pu"], 0.01, bus)
    
    # Add active and reactive power measurements to all lines (Standard Deviation 1.0 MW/Mvar)
    for line in net.line.index:
        pp.create_measurement(net, "p", "line", net.res_line.at[line, "p_from_mw"], 1.0, line, side="from")
        pp.create_measurement(net, "q", "line", net.res_line.at[line, "q_from_mvar"], 1.0, line, side="from")

def run_benchmark(case_name, num_samples=50):
    try:
        if case_name == "case300":
            net = pn.case300()
        elif case_name == "case1354":
            net = pn.case1354pegase()
        else:
            print(f"Unknown case: {case_name}")
            return
    except Exception as e:
        print(f"Could not load {case_name}: {e}")
        return

    print(f"\n--- Benchmarking {case_name.upper()} ({len(net.bus)} Buses) ---")
    
    # 1. Base Power Flow (to get true values)
    pp.runpp(net)
    setup_observable_measurements(net)
    
    # 2. Benchmark AC Power Flow (Newton-Raphson)
    pp_times = []
    for _ in range(num_samples):
        start = time.perf_counter()
        pp.runpp(net, numba=False)
        pp_times.append(time.perf_counter() - start)
    
    # 3. Benchmark Full WLS State Estimation
    # This is 'Full' non-linear WLS using Gauss-Newton.
    se_times = []
    base_meas = net.measurement.value.values.copy()
    for _ in tqdm(range(num_samples), desc=f"WLS {case_name}"):
        # Add small measurement noise
        net.measurement.value = base_meas + np.random.normal(0, 0.001, size=len(base_meas))
        start = time.perf_counter()
        try:
            pp.runse(net, calculate_covariance=False)
            se_times.append(time.perf_counter() - start)
        except:
            continue

    print(f"AC Power Flow Avg: {np.mean(pp_times):.6f}s")
    if se_times:
        print(f"Full WLS Avg:      {np.mean(se_times):.6f}s")
    else:
        print("WLS failed to converge.")

if __name__ == "__main__":
    # We use case1354 as the large system substitute if case2000 is unavailable
    run_benchmark("case300", 20)
    run_benchmark("case1354", 10)
