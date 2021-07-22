import numpy as np
import matplotlib.pyplot as plt

# Load data in data folder
x = np.load("data/x_378350.npy")
t = np.load("data/t_378350.npy")
j = np.load("data/j_378350.npy")

# Clip x, t and j to length time_window
time_window = 4099
t = t[:time_window]
j = j[:time_window]
x = x[:time_window,:]

plt.figure(1)
# Display t vector in plot
plt.subplot(211)
plt.plot(t)
plt.title("t")
plt.xlabel("k")
plt.ylabel("t")
# Display j vector in second subplot
plt.subplot(212)
plt.plot(j)
plt.title("j")
plt.xlabel("k")
plt.ylabel("j")

# Create new figure and display r vector contained in every 6th element of x starting from 0
plt.figure(2)
plt.plot(t, x[:,0::7])
plt.title("r")
plt.xlabel("t")
plt.ylabel("r")


plt.show()