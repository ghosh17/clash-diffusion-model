import numpy as np
import matplotlib.pyplot as plt


def plot_profile(grid, elevation, color="r", title=None, output=None):
    plt.figure()
    plt.plot(grid, elevation, color)
    plt.xlabel("grid (x)")
    plt.ylabel("elevation (z)")
    if title is not None:
        plt.title(title)
    if output is not None:
        plt.savefig(output)


def calculate_stable_time_step(grid_spacing, diffusivity):
    return 0.5 * grid_spacing**2 / diffusivity  # time step double star is exponental


D = 100  # diffusivity
Lx = 300  # model domain

dx = 0.5  # grid spacing
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x)  # number of grid nodes

z = np.zeros_like(x)  # array of 0 of type np (float)
z_left = 500  # left side is at 500 (elevation)
z_right = 0  # right side is at 0
z[x <= Lx / 2] = (
    z_left  # cliff z for values x that are <= half domain size set to z_left
)
z[x >= Lx / 2] = (
    z_right  # cliff z for values x that are >= half domain size set to z_right
)

plot_profile(
    x, z, color="r", title="Initial hillslope profile", output="initial-profile.png"
)

nt = 5000  # number of time steps
dt = calculate_stable_time_step(dx, D)

for t in range(0, nt):
    z[1:-1] += D * dt / dx**2 * (z[:-2] - 2 * z[1:-1] + z[2:])

plot_profile(
    x, z, color="b", title="Final hillslope profile", output="final-profile.png"
)
