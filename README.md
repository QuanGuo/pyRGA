# GWSolver

A Python package for groundwater flow simulation and hydraulic tomography using the Reformulated Geostatistical Approach (RGA).

## Features

- 2D steady-state and transient groundwater flow simulation
- Hydraulic tomography analysis
- Reformulated Geostatistical Approach (RGA) for parameter estimation
- Parallel computation support
- Visualization tools for hydraulic head and conductivity fields

## Installation

```bash
pip install gwsolver
```

## Quick Start

### Basic Usage

```python
import numpy as np
from gwsolver import hydraulic_tomography, RGA

# Define domain parameters
nx, ny = 64, 64
K = np.exp(np.random.randn(nx * ny) * 0.1 - 2)

# Define well locations
well_locs = [(0.25, 0.25), (0.75, 0.75)]
well_nodes = [int(x*nx) + int(y*ny)*nx for x, y in well_locs]

# Solve hydraulic tomography
Q = -0.1  # Pumping rate
heads = hydraulic_tomography(K, well_nodes, Q)

```

### Complete Optimization Example

```python
import numpy as np
import time
from gwsolver import hydraulic_tomography, RGA
from gwsolver.RGA import generate_random_field, observation_operator, forward_model, gauss_newton_dynamic_lambda

# Define numerical domain parameters
Lox = Loy = 100  # Domain size
nx = ny = 64     # Grid resolution
numel = nx * ny
numnodx = numnody = nx + 1
numnod = numnodx * numnody

# Random field parameters
sigma = 1.0      # Standard deviation
lx, ly = 0.15, 0.2  # Correlation lengths
NR = 400         # Number of random fields for covariance estimation
mu = -4          # Mean of the random field

# Generate random field using PCA
ucr = generate_random_field(
    NR,
    Nx=nx,
    Ny=ny,
    cov_type='gaussian',
    variance=sigma**2,
    lx=lx,
    ly=ly,
    mu=mu
).reshape(NR, -1)

beta = np.mean(ucr)  # Estimated mean

# Perform SVD for dimension reduction
U, S, Vt = np.linalg.svd(ucr - beta, full_matrices=False)
k = 50  # Number of retained principal components
V = np.expand_dims(np.sqrt(S[:k]), axis=1) * Vt[:k]

# Generate synthetic random field
alpha = np.random.randn(k)
logK = mu + (V.T @ alpha[:, np.newaxis]).flatten()
K = np.exp(logK)  # Conductivity field in m/s

# Define physical domain and pumping parameters
Lox = Loy = 320  # Domain length in meters
dx = dy = Lox / nx  # Grid size in meters
q_original = -0.02 * (64/nx)**2  # Original pumping rate in m³/s
Q = q_original/dx/dy*3600  # Applied force: -2.88 m/hr

# Define well locations
well_relative_locs = []
horizontal_locs = vertical_locs = [0.25, 0.375, 0.5, 0.625, 0.75]
for h in horizontal_locs:
    for v in vertical_locs:
        well_relative_locs.append((h, v))

well_nodes = [int(x*numnodx) + int(y*numnody)*numnodx 
             for x, y in well_relative_locs]

# Solve hydraulic tomography
t0 = time.time()
hydraulic_heads = hydraulic_tomography(K, well_nodes, Q)
print(f"Elapsed time for solving HT: {time.time() - t0:.2f} seconds")

# Prepare observations
y0 = observation_operator(hydraulic_heads, well_nodes)

# Add measurement noise
Nobv = len(y0)
rng = np.random.default_rng()
noise_level = 0.05
obv_error = np.abs(y0) * rng.normal(0, noise_level, Nobv)
y = y0 + obv_error

# Initialize optimization history
initial_history = {
    'true_alpha': alpha,
    'true_mu': mu,
    'true_y': y0,
    'alpha': [],
    'mu': [],
    'loss': [],
    'lambda': [],
    'step_norm': [],
    'time': [],
    'yp': []
}

# Run optimization
b, opt_history = gauss_newton_dynamic_lambda(
    lambda b: forward_model(b, V.T, Q, well_nodes),
    b0=np.concatenate((np.zeros(k), np.array([beta]))),
    y_obs=y,
    lam_init=1e-3,
    max_iter=10,
    tol=1e-5,
    history=initial_history,
    adaptive_lambda=False,
    anneal_lambda=True,
    min_lambda=1e-5
)
```

## Documentation

For detailed documentation, please visit [Read the Docs](https://gwsolver.readthedocs.io/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{gwsolver2024,
  author = {Quan Guo},
  title = {GWSolver: A Python package for groundwater flow simulation and hydraulic tomography},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/QuanGuo/GWSolver}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
