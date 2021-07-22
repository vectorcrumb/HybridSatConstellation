import numpy as np
from scipy import integrate

np.random.seed(2021)

N = 10
M = N - 1 
S = 4
mu = 4.282837e13
t_horizon = 355 * 1.0275 * 24 * 60 * 60
# t_horizon = 3 * 30 * 24 * 60 * 60
t_f = t_horizon
m_sat = 100
mass = m_sat * np.ones((N, 1))
rd = 20428.2 * 1000
ref = np.array([rd, 0, np.sqrt(mu / (rd ** 3))])
theta_rel_d = 2 * np.pi / N
h_L = lambda theta_rel: theta_rel - theta_rel_d
D = np.eye(N,M) - np.eye(N,M,-1)

kr = 1e-5
kv = 1e-4
kw = 1e4
kc_sup = 1e11
kc_inf  = 1e9
kc_c = 30

def gen_x0():
    r0 = ref[0] + 0.2 * np.random.rand(N,1) - 0.1
    theta0 = 1e-3 * (10 * np.random.rand(N,1) - 5)
    v0 = 1e-8 * (2 * np.random.rand(N,1) - 1)
    w0 = ref[2] + 1e-5 * (0.02 * np.random.rand(N,1) - 0.01)

    x0 = np.zeros((S*N, 1))
    x0[0::S] = r0
    x0[1::S] = theta0
    x0[2::S] = v0
    x0[3::S] = w0

    return x0

def kc_gain(t):
    kc = (kc_sup - kc_inf) * np.exp(-(kc_c/t_f)*t) + kc_inf
    return kc

def sat_cont_model(t, x, u):
    r = x[0::S]
    v = x[2::S]
    w = x[3::S]

    rd = (np.ones((N, 1)) * ref[0]).ravel()
    vd = (np.ones((N, 1)) * ref[1]).ravel()
    wd = (np.ones((N, 1)) * ref[2]).ravel()

    xdot = np.zeros(x.shape)
    xdot[0::S] = v
    xdot[1::S] = w
    xdot[2::S] = -kv * (v - vd) - kr * (r - rd)
    xdot[3::S] = -kw * (w - wd) / r + u / kc_gain(t)

    return xdot

def constellation_law(x):
    theta = x[1::S]
    e = D.T @ theta
    y = h_L(e)
    u = -D @ y
    return u