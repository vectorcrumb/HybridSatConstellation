import numpy as np
from model import *
from step_model import step_model
import matplotlib.pyplot as plt
from tqdm import tqdm

dt = 60
est_t_samples = t_horizon / dt
est_j_samples = N * t_horizon / T_sat
n = int((est_t_samples + est_j_samples) * 1.25)

def main():
    # Declare containers for x, t, j and x0
    x = np.zeros((n, S*N), dtype=float)
    t = np.zeros(n, dtype=float)
    j = np.zeros(n, dtype=int)
    x0 = gen_x0()
    # Prepare first samples of x, t and j
    x[0,:] = x0.T
    t[0] = 0.0
    j[0] = 0
    # Loops uses xk-1 as previous state and xk as calculated state. 
    # Decisions are made based on xk-1.
    for idx in tqdm(range(1,n), desc='Simulation'):
        if not idx % 10000:
            print(f"Time {t[idx-1]} @ idx {idx}")
        if idx >= 4100:
            break
        # Step 0: auxiliary variables
        tk = t[idx-1]
        jk = j[idx-1]
        # Step 1: Get xk-1 (previous sample) and determine belonging to set C or D
        xk_1 = x[idx-1,:].T
        c_test = c_set(xk_1)
        d_test = d_set(xk_1)
        # Step 2: If xk_1 belongs to set D, jump to xk
        if d_test:
            xk_g = g_map(xk_1)
            # Set t (same) and j (add 1)
            j[idx] = jk + 1
            t[idx] = tk
            # Set xk
            x[idx,:] = xk_g
        # Step 3: If xk_1 belongs to set D, flow to xk. Calculate control signals
        elif c_test:
            # Step 2: Observe previous state
            yk_1 = state_observer(xk_1)
            # Step 3: Calculate constellation feedback control signal
            uk_1 = constellation_law(yk_1, dt)
            # Step 4: Calculate internal feedback control signal
            tauk_1 = fb_law(tk, xk_1, uk_1, ref)
            # Simulate system for a timestep
            _, x_aux = step_model(constellation_model, tauk_1, tk, dt, xk_1)
            # Generate xk
            xk_f = (x_aux.T)[-1,:]
            # Set t (+dt) and j (same)
            j[idx] = jk
            t[idx] = tk + dt
            # Set xk
            x[idx,:] = xk_f
        # Step : If xk_1 belongs to neiter set C nor set D, end simulation
        else:
            print("Error, simulation ended prematurely")
            break
    
    print("Finished simulating!")
    np.save(f"data/x.npy", x)
    np.save(f"data/t.npy", t)
    np.save(f"data/j.npy", j)
    

if __name__ == '__main__':
    main()