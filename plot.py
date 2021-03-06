import numpy as np
import matplotlib.pyplot as plt

# Load data in data folder
x = np.load("data/xc.npy")
t = np.load("data/tc.npy")

r = x[:,::4]
theta = x[:,1::4]
v = x[:,2::4]
w = x[:,3::4]
t_days = t / (24 * 60 * 60)
e = (theta[:,:-1] - theta[:,1:]) * 180 / np.pi

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

