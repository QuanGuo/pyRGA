# pyRGA

A Python package for groundwater flow simulation and hydraulic tomography using the Reformulated Geostatistical Approach (RGA).

## Features

- 2D steady-state and transient groundwater flow simulation
- Hydraulic tomography analysis
- Reformulated Geostatistical Approach (RGA) for parameter estimation
- Parallel computation support
- Visualization tools for hydraulic head and conductivity fields

## Installation

```bash
pip install pyrga
```

## Quick Start

### Basic Usage

```python
import numpy as np
from pyrga import hydraulic_tomography
from pyrga.RGA import prepare_physical_domain

# Define domain parameters
nx, ny = 64, 64
K = np.exp(np.random.randn(nx * ny) * 0.1 - 2)

# Prepare well configuration
well_nodes, Q, _, _ = prepare_physical_domain(nx, ny)

# Solve hydraulic tomography
heads = hydraulic_tomography(K, well_nodes, Q)
```

### Complete Optimization Example

```python
import numpy as np
import time
from pyrga import hydraulic_tomography
from pyrga.RGA import (
    generate_synthetic_field,
    prepare_physical_domain,
    observation_operator,
    forward_model,
    gauss_newton_dynamic_lambda,
    add_noise
)

# Define numerical domain parameters
nx = ny = 64     # Grid resolution

# Generate synthetic conductivity field
K, alpha, V, beta = generate_synthetic_field(
    nx, ny,
    k=50,                    # Number of retained components
    mu=-4,                   # Mean of the random field
    NR=400,                  # Number of random fields
    cov_type='gaussian',
    variance=1.0,            # Standard deviation squared
    lx=0.15,                # Correlation length in x
    ly=0.2                  # Correlation length in y
)

# Prepare physical domain and well configuration
well_nodes, Q, dx, dy = prepare_physical_domain(
    nx, ny,
    Lox=320,                # Domain length in x direction (m)
    Loy=320,                # Domain length in y direction (m)
    q_original=-0.02,       # Original pumping rate in m³/s
    well_relative_locs=None # Use default 5x5 grid of wells
)

# Solve hydraulic tomography
t0 = time.time()
hydraulic_heads = hydraulic_tomography(K, well_nodes, Q)
print(f"Elapsed time for solving HT: {time.time() - t0:.2f} seconds")

# Prepare observations
y0 = observation_operator(hydraulic_heads, well_nodes)

# Add measurement noise
y, obv_error = add_noise(y0, noise_level=0.05)

# Initialize optimization history
initial_history = {
    'true_alpha': alpha,
    'true_mu': -4,
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
    b0=np.concatenate((np.zeros(50), np.array([beta]))),
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

For detailed documentation, please visit [Read the Docs](https://pyrga.readthedocs.io/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{zhao2020reformulation,
  author = {Zhao, Yue and Luo, Jian},
  title = {Reformulation of Bayesian Geostatistical Approach on Principal Components},
  journal = {Water Resources Research},
  volume = {56},
  number = {4},
  pages = {e2019WR026732},
  year = {2020},
  doi = {https://doi.org/10.1029/2019WR026732},
  url = {https://doi.org/10.1029/2019WR026732},
  keywords = {geostatistical approach, inverse modeling, principal component}
}
```

Or in APA format:

Zhao, Y., & Luo, J. (2020). Reformulation of Bayesian Geostatistical Approach on Principal Components. *Water Resources Research*, *56*(4), e2019WR026732. https://doi.org/10.1029/2019WR026732

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
