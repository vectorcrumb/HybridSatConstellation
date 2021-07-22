import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, BDF
from tqdm import tqdm
from model_cont import *

dt = 10 * 60
n = int(1.25 * t_horizon / dt)

def step_model(model, u, t0, step_size, x0, sample_size=5):
    tf = t0 + step_size
    tX = np.linspace(t0, tf, sample_size)
    # print(f"Integrating from {t0} to {tf}")
    x = solve_ivp(model, (t0, tf), y0=x0, t_eval=tX, args=(u,), method="BDF")
    return x.t, x.y

def main():
    x = np.zeros((n, S*N), dtype=float)
    t = np.zeros(n, dtype=float)
    x0 = gen_x0()
    # print(x[0,:].shape)
    x[0,:] = x0.ravel()
    t[0] = 0.0

    try:
        for idx in tqdm(range(1, n)):
            tk_1 = t[idx-1]
            xk_1 = x[idx-1, :].T

            uk_1 = constellation_law(xk_1)

            _, x_aux = step_model(sat_cont_model, uk_1, tk_1, dt, xk_1)
            xk = (x_aux.T)[-1,:]
            t[idx] = tk_1 + dt
            x[idx, :] = xk
    except KeyboardInterrupt:
        pass
    
    print("Finished simulating!")
    np.save(f"data/xc.npy", x)
    np.save(f"data/tc.npy", t)

if __name__ == "__main__":
    main()