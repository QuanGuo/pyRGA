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

# Use RGA for parameter estimation
rga = RGA()
estimated_K = rga.estimate_conductivity(heads, well_nodes)
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
