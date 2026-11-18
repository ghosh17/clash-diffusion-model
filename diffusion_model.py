"""A simple one-dimensional hillslope diffusion model."""

import numpy as np


def diffusion_step(profile, diffusivity, dx, dt):
    """Advance a hillslope elevation profile by one explicit-diffusion step.

    The model uses a centered finite difference approximation of
    ``dh/dt = diffusivity * d2h/dx2``.  Zero-gradient (no-flux) boundary
    conditions are imposed at both ends.
    """
    profile = np.asarray(profile, dtype=float)

    if profile.ndim != 1:
        raise ValueError("profile must be one-dimensional")
    if profile.size < 3:
        raise ValueError("profile must contain at least three values")
    if not np.isfinite(profile).all():
        raise ValueError("profile values must be finite")
    if dx <= 0:
        raise ValueError("dx must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    diffusivity = np.asarray(diffusivity, dtype=float)
    if np.any(diffusivity < 0) or not np.isfinite(diffusivity).all():
        raise ValueError("diffusivity must be finite and non-negative")

    courant = diffusivity * dt / (dx * dx)
    if np.any(courant > 0.5):
        raise ValueError("diffusion is unstable: diffusivity * dt / dx**2 must be <= 0.5")

    next_profile = profile.copy()
    if diffusivity.ndim == 0:
        coefficient = float(courant)
        next_profile[1:-1] = profile[1:-1] + coefficient * (
            profile[:-2] - 2 * profile[1:-1] + profile[2:]
        )
    else:
        if diffusivity.shape != profile.shape:
            raise ValueError("diffusivity must be a scalar or match the profile shape")
        coefficient = diffusivity * dt / (dx * dx)
        next_profile[1:-1] = profile[1:-1] + coefficient[1:-1] * (
            profile[:-2] - 2 * profile[1:-1] + profile[2:]
        )

    # Mirror the boundary nodes to represent zero-gradient boundaries.
    next_profile[0] = next_profile[1]
    next_profile[-1] = next_profile[-2]
    return next_profile


def run_diffusion(profile, diffusivity, dx, dt, num_steps):
    """Run a hillslope diffusion model for a number of time steps."""
    if not isinstance(num_steps, (int, np.integer)) or num_steps < 0:
        raise ValueError("num_steps must be non-negative")

    elevation = np.asarray(profile, dtype=float).copy()
    for _ in range(int(num_steps)):
        elevation = diffusion_step(elevation, diffusivity, dx, dt)
    return elevation
