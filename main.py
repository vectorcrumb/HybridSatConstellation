import numpy as np
from scipy.integrate import odeint, solve_ivp, BDF
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import *


dt = 5 * 60
est_t_samples = t_horizon / dt
est_j_samples = N * t_horizon / T_sat
n = int((est_t_samples + est_j_samples) * 1.25)


def step_model(model, u, t0, step_size, x0, sample_size=10):
    tf = t0 + step_size
    tX = np.linspace(t0, tf, sample_size)
    # print(f"Integrating from {t0} to {tf}")
    x = solve_ivp(model, (t0, tf), y0=x0, t_eval=tX, args=(u,))
    return x.t, x.y

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
        try:
            # Step 0: auxiliary variables
            xk_1 = x[idx-1,:].T
            tk = t[idx-1]
            jk = j[idx-1]
            # Step 1: Determine belonging to set C or D
            c_test = c_set(xk_1)
            d_test = d_set(xk_1)
            # Step 2: If xk_1 belongs to set D, jump to xk
            if d_test:
                xk_g = g_map(xk_1)
                # Set t (same) and j (add 1)
                j[idx] = jk + 1
                t[idx] = tk
                # Set xk
                x[idx,:] = xk_g.ravel()
            # Step 3: If xk_1 belongs to set D, flow to xk. Calculate control signals
            elif c_test:
                # Step 2: Observe previous state
                # yk_1 = state_observer(xk_1)
                # Step 3: Calculate constellation feedback control signal
                uk_1 = constellation_law(xk_1)
                # Step 4: Calculate internal feedback control signal
                # tauk_1 = fb_law(tk, xk_1, uk_1, ref)
                # Simulate system for a timestep
                _, x_aux = step_model(f_map, uk_1, tk, dt, xk_1)
                # Generate xk
                xk_f = (x_aux.T)[-1,:]
                # Set t (+dt) and j (same)
                j[idx] = jk
                t[idx] = tk + dt
                # Set xk
                x[idx,:] = xk_f.ravel()
            # Step : If xk_1 belongs to neiter set C nor set D, end simulation
            else:
                print("Error, simulation ended prematurely")
                break
        except KeyboardInterrupt:
            pass
    
    print("Finished simulating!")
    np.save(f"data/xh.npy", x)
    np.save(f"data/th.npy", t)
    np.save(f"data/jh.npy", j)
    

if __name__ == '__main__':
    main()