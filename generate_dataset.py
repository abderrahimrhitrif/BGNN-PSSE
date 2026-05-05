import os
import copy
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────────────
TOTAL_SAMPLES  = 10000
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.15
TEST_FRAC      = 0.15
OUTPUT_DIR     = "./IEEE300"
RANDOM_SEED    = 42

LOAD_NOISE_STD    = 0.05
GEN_NOISE_STD     = 0.03
LINE_P_NOISE_STD  = 0.01
PINJ_NOISE_STD    = 0.01
ANGLE_NOISE_STD   = 0.005


# ── Network ────────────────────────────────────────────────────────────────────
def build_base_network():
    net = pn.case300()
    pp.runpp(net, numba=False, verbose=False)
    assert net.converged
    return net


def perturb_and_solve(net_base, rng):
    # Create a deep copy of the network
    net = copy.deepcopy(net_base)

    # Add random noise to the generator loads (50% to 150%)
    scale = np.clip(rng.normal(1.0, LOAD_NOISE_STD, len(net.load)), 0.5, 1.5)
    net.load["p_mw"] *= scale # Active power
    net.load["q_mvar"] *= scale # Reactive power

    if len(net.gen) > 0: # Check for generator presence and add noise
        sg = np.clip(rng.normal(1.0, GEN_NOISE_STD, len(net.gen)), 0.5, 1.5)
        net.gen["p_mw"] *= sg # Active power

    try:
        pp.runpp(net, numba=False, verbose=False, max_iteration=50) # Run the SE solver
    except:
        return None

    return net if net.converged else None


# This function gets the state vector x aka the ground truth
def state_vector(net):
    vm = net.res_bus["vm_pu"].values # Magnitude
    va = np.deg2rad(net.res_bus["va_degree"].values) # Angle/Phase
    return np.concatenate([vm * np.cos(va), vm * np.sin(va)]) # This actually returns Vreal and Vimg



def _ybus_and_voltage(net):
    ppc = net._ppc
    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    Ybus = Ybus.toarray()
    Yf   = Yf.toarray()

    vm = net.res_bus["vm_pu"].values
    va = np.deg2rad(net.res_bus["va_degree"].values)
    V  = vm * np.exp(1j * va)

    return Ybus, Yf, V, ppc["baseMVA"], ppc 


# This is still confusing to me, I need to get through it again
def true_measurement_vector(net, slack, n_buses, n_branches):
    Ybus, Yf, V, baseMVA, ppc = _ybus_and_voltage(net)

    If = Yf @ V
    fb = ppc["branch"][:n_branches, 0].astype(int)
    Sf = V[fb] * np.conj(If[:n_branches])
    z_lp = Sf.real * baseMVA

    Sbus = V * np.conj(Ybus @ V) * baseMVA
    z_pi = Sbus.real

    va = np.deg2rad(net.res_bus["va_degree"].values)
    non_slack = [i for i in range(n_buses) if i != slack]
    z_th = va[non_slack]

    return np.concatenate([z_lp, z_pi, z_th])


def variance_vector(z, n_line, n_pinj, n_theta):
    eps = 1e-4
    var = np.zeros_like(z)

    var[:n_line] = (LINE_P_NOISE_STD * np.abs(z[:n_line]) + eps) ** 2
    var[n_line:n_line+n_pinj] = (PINJ_NOISE_STD * np.abs(z[n_line:n_line+n_pinj]) + eps) ** 2
    var[n_line+n_pinj:] = (ANGLE_NOISE_STD + eps) ** 2

    return var


# ── Topology-based covariance ──────────────────────────────────────────────────
def build_topology_pairs(ppc, n_buses, n_branches, slack,
                         n_line, n_pinj, n_theta):

    pairs = []

    line_off  = 0
    pinj_off  = n_line
    theta_off = n_line + n_pinj

    branches = ppc["branch"][:n_branches]
    fb = branches[:, 0].astype(int)
    tb = branches[:, 1].astype(int)

    neighbors = {i: set() for i in range(n_buses)}
    for i, j in zip(fb, tb):
        neighbors[i].add(j)
        neighbors[j].add(i)

    non_slack = [i for i in range(n_buses) if i != slack]
    theta_map = {b: k for k, b in enumerate(non_slack)}

    # Line ↔ Line
    for i in range(n_branches):
        for j in range(n_branches):
            if i != j and (fb[j] == fb[i] or fb[j] in neighbors[fb[i]]):
                pairs.append((line_off+i, line_off+j))

    # Injection ↔ Injection
    for i in range(n_buses):
        for j in neighbors[i]:
            pairs.append((pinj_off+i, pinj_off+j))

    # Injection ↔ Angle
    for i in non_slack:
        pairs.append((pinj_off+i, theta_off+theta_map[i]))

    # Angle ↔ Angle
    for i in non_slack:
        for j in neighbors[i]:
            if j in theta_map:
                pairs.append((theta_off+theta_map[i],
                              theta_off+theta_map[j]))

    return pairs


def covariance_vector(var, pairs, M,
                      n_line, n_pinj, n_theta):

    cov = np.zeros(M//2)

    rho_line  = 0.3
    rho_pinj  = 0.25
    rho_angle = 0.2
    rho_cross = 0.35

    for k, (i, j) in enumerate(pairs[:M//2]):

        if i < n_line and j < n_line:
            rho = rho_line
        elif n_line <= i < n_line+n_pinj and n_line <= j < n_line+n_pinj:
            rho = rho_pinj
        elif i >= n_line+n_pinj and j >= n_line+n_pinj:
            rho = rho_angle
        else:
            rho = rho_cross

        cov[k] = rho * np.sqrt(var[i] * var[j])

    return cov


# ── Dataset ────────────────────────────────────────────────────────────────────
def generate_samples(N, seed):
    rng = np.random.default_rng(seed)
    base = build_base_network()

    n_buses = len(base.bus)
    n_branches = len(base.line) + len(base.trafo)
    V = 2*n_buses

    slack = int(base._pd2ppc_lookups["bus"][base.ext_grid["bus"].values[0]])

    n_line = n_branches
    n_pinj = n_buses
    n_theta = n_buses - 1
    M = n_line + n_pinj + n_theta

    _, _, _, _, ppc = _ybus_and_voltage(base)
    topo_pairs = build_topology_pairs(ppc, n_buses, n_branches, slack,
                                      n_line, n_pinj, n_theta)

    X = np.zeros((N, V))
    Z = np.zeros((N, M))
    VAR = np.zeros((N, M))
    COV = np.zeros((N, M//2))

    k = 0
    pbar = tqdm(total=N, desc="Generating Power Grid Samples")
    while k < N:
        net = perturb_and_solve(base, rng)
        if net is None:
            continue

        X[k] = state_vector(net)

        z = true_measurement_vector(net, slack, n_buses, n_branches)
        var = variance_vector(z, n_line, n_pinj, n_theta)

        Z[k] = z + rng.normal(0, np.sqrt(var))
        VAR[k] = var
        COV[k] = covariance_vector(var, topo_pairs, M,
                                   n_line, n_pinj, n_theta)

        k += 1
        pbar.update(1)
    
    pbar.close()

    return X, Z, VAR, COV, M, n_buses


# ── Save ───────────────────────────────────────────────────────────────────────
def save(prefix, out, X, Z, VAR, COV, M, B):
    os.makedirs(out, exist_ok=True)

    pd.DataFrame(X).to_csv(f"{out}/{prefix}_Estimate.csv", index=False, header=False)
    pd.DataFrame(Z).to_csv(f"{out}/{prefix}_Measurement.csv", index=False, header=False)
    pd.DataFrame(VAR).to_csv(f"{out}/{prefix}_Variance.csv", index=False, header=False)
    pd.DataFrame(COV).to_csv(f"{out}/{prefix}_Covariance.csv", index=False, header=False)

    with open(f"{out}/{prefix}_NumMeasurements.txt", "w") as f:
        f.write(f"{M}\n{B}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    X, Z, VAR, COV, M, B = generate_samples(TOTAL_SAMPLES, RANDOM_SEED)

    n_train = int(TRAIN_FRAC * TOTAL_SAMPLES)
    n_val   = int(VAL_FRAC * TOTAL_SAMPLES)

    save("Training", OUTPUT_DIR,
         X[:n_train], Z[:n_train], VAR[:n_train], COV[:n_train], M, B)

    save("Validation", OUTPUT_DIR,
         X[n_train:n_train+n_val],
         Z[n_train:n_train+n_val],
         VAR[n_train:n_train+n_val],
         COV[n_train:n_train+n_val], M, B)

    save("Test", OUTPUT_DIR,
         X[n_train+n_val:],
         Z[n_train+n_val:],
         VAR[n_train+n_val:],
         COV[n_train+n_val:], M, B)


if __name__ == "__main__":
    main()