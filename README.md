# clash-diffusion-model
CSDMS/CLaSH roadshow workshop for simple diffusion model.

## Python example

The 1-D finite-difference hillslope diffusion model is available in
`diffusion_model.py`. It solves

`dh/dt = diffusivity * d²h/dx²`

with no-flux boundary conditions. For example:

```python
import numpy as np
from diffusion_model import run_diffusion

x = np.linspace(0, 1000, 101)
profile = 100 - 0.02 * (x - 500) ** 2

elevation = run_diffusion(
    profile,
    diffusivity=0.01,
    dx=x[1] - x[0],
    dt=1,
    num_steps=100,
)
```

Run the interactive example in `simple-diffusion-model-hill-slope.ipynb`.

Install the notebook dependencies with:

```bash
python -m pip install -r requirements.txt
```
