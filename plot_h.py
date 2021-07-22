import numpy as np
import matplotlib.pyplot as plt


x = np.load("data/xh.npy")
t = np.load("data/th.npy")
j = np.load("data/jh.npy")
S = 7
r = x[:,::S]
theta = x[:,1::S]
v = x[:,2::S]
w = x[:,3::S]
tau = x[:,4::S]
e = (theta[:,:-1] - theta[:,1:]) * 180 / np.pi
t_days = t / (24 * 60 * 60)

plt.figure(1)

plt.subplot(221)
plt.plot(t_days, r)
plt.title("$r$")
# plt.xlabel("t")
plt.ylabel("m")
plt.subplot(222)
plt.plot(t_days, e)
plt.title("$\\theta^{rel}$")
plt.axhline(y=36, color='r', linestyle='-')
# plt.xlabel("t")
plt.ylabel("deg")
plt.subplot(223)
plt.plot(t_days, v)
plt.title("$v$")
plt.xlabel("t")
plt.ylabel("m/s")
plt.subplot(224)
plt.plot(t_days, w)
plt.title("$\\omega$")
plt.xlabel("t")
plt.ylabel("rad/s")


plt.show()