import numpy as np
import matplotlib.pyplot as plt


D = 100 # diffusivity
Lx = 300 # model domain


dx = 0.5 # grid spacing
x = np.arange(start = 0, stop = Lx, step = dx)
nx = len(x) # number of grid nodes


z = np.zeros_like(x) #array of 0 of type np (float)
z_left = 500 # left side is at 500 (elevation)
z_right = 0 # right side is at 0
z[x <= Lx / 2] = z_left # cliff z for values x that are <= half domain size set to z_left
z[x >= Lx / 2] = z_right # cliff z for values x that are >= half domain size set to z_right


plt.figure()
plt.plot(x,z,"r")
plt.xlabel("x")
plt.ylabel("z")
plt.title("initial hillslope profile")


nt = 5000 # number of time steps
dt = 0.5 * dx ** 2 / D # time step double star is exponental 


for t in range(0, nt):
	z[1:-1] += D * dt / dx ** 2 * (z[:-2] - 2*z[1:-1] + z[2:])


plt.figure()
plt.plot(x,z,"b")
plt.xlabel("x")
plt.ylabel("z")
plt.title("Final hillslope profile")







