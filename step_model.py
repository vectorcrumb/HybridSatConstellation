from scipy.integrate import odeint, solve_ivp, BDF
import numpy as np

def step_model(model, u, t0, step_size, x0, sample_size=10):
    tf = t0 + step_size
    tX = np.linspace(t0, tf, sample_size)
    print(f"Integrating from {t0} to {tf}")
    x = solve_ivp(model, (t0, tf), y0=x0, t_eval=tX, args=(u,step_size,))
    return x.t, x.y