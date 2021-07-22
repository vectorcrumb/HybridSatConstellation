import numpy as np
from scipy import integrate

np.random.seed(2021)

N = 10
M = N - 1
S = 7
mu = 4.282837e13
t_horizon = 355 * 1.0275 * 24 * 60 * 60
t_f = t_horizon
# t_horizon = 7 * 24 * 60 * 60
T_sat = 60 * 60 # 1 hour
m_sat = 100
mass = m_sat * np.ones((N, 1))
rd = 20428.2 * 1000
ref = np.array([rd, 0, np.sqrt(mu / (rd ** 3))])
theta_rel_d = 2 * np.pi / N

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
    timer0 = T_sat * np.random.rand(N,1) + (10*60) # 10 min + 1 hour random

    x0 = np.zeros((S*N, 1))
    x0[0::S] = r0
    x0[1::S] = theta0
    x0[2::S] = v0
    x0[3::S] = w0
    x0[4::S] = timer0
    x0[5::S] = np.append(0, theta0[:-1])[:, np.newaxis]
    x0[6::S] = np.append(theta0[1:], 0)[:, np.newaxis]
    return x0


def kc_gain(t):
    kc = (kc_sup - kc_inf) * np.exp(-(kc_c/t_f)*t) + kc_inf
    return kc


def f_map(t, x, u):
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
    xdot[4::S] = -1
    xdot[5::S] = 0
    xdot[6::S] = 0

    return xdot


def c_set(x):
    r = x[0::S]
    belongs = np.all(r > 0)
    return belongs


def g_map(x):

    xplus = np.copy(x)
    # Retrieve relevant variables
    theta = x[1::S]
    timer = x[4::S]
    # Find expired timers
    exp_timers = np.where(timer <= 0)
    # Grab only the first expired timer
    idx = exp_timers[0][0]
    # print(idx, exp_timers)
    # Reset expired timer
    xplus[4::S][idx] = T_sat
    # Check if the expired timer is the first agent
    if idx == 0:
        # Set theta_n for second agent to the first agents theta value
        xplus[5::S][1] = theta[idx]
    # Check if the expired timer is the last agent
    elif idx == N - 1:
        # Set theta_p for the second to last agent to the last agents theta value
        xplus[6::S][-2] = theta[idx]
    else:
        # Set theta_n for the next agent to the agents theta value
        xplus[5::S][idx + 1] = theta[idx]
        # Set theta_p for the previous agent to the agents theta value
        xplus[6::S][idx - 1] = theta[idx]
    
    return xplus


def d_set(x):
    belongs_to_c = c_set(x)
    # Check for expired timers
    timer = x[4::S]
    timer_test = np.any(timer <= 0)

    belongs = belongs_to_c and timer_test

    return belongs


# def fb_law(t, x, u, ref):
#     r = x[0::S]
#     v = x[2::S]
#     w = x[3::S]

#     rd, vd, wd = ref
#     rd = rd * np.ones((N, 1))
#     vd = vd * np.ones((N, 1))
#     wd = wd * np.ones((N, 1))

#     taur = mass * (-r * (w ** 2) + mu / (r ** 2)) - kv * (v - vd) - kr * (r - rd)
#     taut = mass * (2 * (v * w) - kw * (w - wd) + (r * u) / kc_gain(t))

#     tau = np.concatenate((taur, taut), axis=1)

#     return tau

# def state_observer(x):
#     theta = x[1::S]
#     theta_n = x[5::S]
#     theta_p = x[6::S]

#     y = np.concatenate((theta, theta_n, theta_p), axis=0)

#     return y

def constellation_law(x):
    theta = x[1::S]
    theta_n = x[5::S]
    theta_p = x[6::S]

    u = np.zeros((N, 1))
    # Set first element of u
    u[0] = theta_p[0] - theta[0] + theta_rel_d
    # Set last element of u
    u[-1] = theta_n[-1] - theta[-1] - theta_rel_d
    # Set the rest of u
    u_inner = theta_n[1:-1] + theta_p[1:-1] - 2 * theta[1:-1]
    u[1:-1] = u_inner[:, np.newaxis]
    u = u.ravel()
    return u


# def constellation_model(t, x, u):
#     return f_map(t, x, u)
