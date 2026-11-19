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
pip install pyrga-ht
```

For the development workflows described below, install this checkout with
Python 3.10 or newer:

```bash
python -m pip install -e '.[mcp]'
```

Optional plotting dependencies are available with `.[plot]`, or install both
extras with `.[mcp,plot]`. The distribution is named `pyrga-ht`; Python imports
use the lowercase name `pyrga`.

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
well_nodes, _, _, _ = prepare_physical_domain(nx, ny)
Q = -0.02  # SI nodal pumping rate, m^3/s, for a one-metre aquifer thickness

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
well_nodes, _, dx, dy = prepare_physical_domain(
    nx, ny,
    Lox=320,                # Domain length in x direction (m)
    Loy=320,                # Domain length in y direction (m)
    q_original=-0.02,       # Original pumping rate in m³/s
    well_relative_locs=None # Use default 5x5 grid of wells
)
Q = -0.02  # Apply the SI nodal rate directly to the steady-state solver

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
    'y_obs': y,
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

# Reconstruct the estimated conductivity field
estimated_K = np.exp(V.T @ b[:-1] + b[-1])
```

`forward_model` uses a basis of shape `(nx * ny, k)`, so pass `V.T` from
`generate_synthetic_field`. Its output orders observations by pumping well,
then by every other well in `well_nodes`; the pumping well's own head is
excluded. Real measurements must follow the same order. The current optimizer
serializes `true_alpha`, `true_mu`, and `y_obs` from the supplied history; the
example above supplies those fields for a synthetic experiment.

`prepare_physical_domain` also returns a legacy rate scaled by cell area and
3600. The steady-state examples above discard that returned rate and use SI
nodal rates directly, matching the validated forward workflow below.

## Documentation

For detailed documentation, please visit [Read the Docs](https://pyrga.readthedocs.io/).

## Validated forward workflow (development)

Version `0.2` introduces a canonical, validated dataset shared by Python,
MCP tools, and the research portal. The first vertical slice supports CSV or
JSON input, steady-state forward simulation, and HTML/SVG result reports.
This slice requires SI units (length and head in `m`, conductivity in `m/s`,
and pumping rate in `m^3/s`), equal grid counts in x and y, and square physical
cells. The two-dimensional forward model assumes a one-metre aquifer thickness.
Unsupported units and grid geometry are rejected rather than converted.
Numerical input fields must be finite, and conductivity must be positive.
Well IDs must be unique, nodes must lie within the grid, and observations must
reference existing wells.

Node indices start at zero, with x varying fastest:
`node = row * (nx + 1) + column`. The left and right boundaries have fixed
zero head; nonzero pumping or injection rates on those boundaries are rejected.
Observation-only wells may lie there. The top and bottom boundaries have
zero normal flux. Negative pumping rates represent extraction; positive rates
represent injection.

```python
from pyrga.models import DarcyDataset, GridSpec, Well
from pyrga.workflows import run_forward
from pyrga.rendering import write_forward_report

dataset = DarcyDataset(
    name="Small aquifer",
    grid=GridSpec(nx=2, ny=2, length_x=20, length_y=20),
    conductivity=[1e-4] * 4,
    wells=[Well(id="PW-1", node=4, pumping_rate=-0.001)],
)

result = run_forward(dataset)
write_forward_report(result, "output/small-aquifer")
```

This runs each nonzero-rate well as an independent pumping test. It does not
apply all wells simultaneously or fit the dataset's observations. The report
plots the first pumping test; its embedded JSON contains all simulated fields.

## Using the MCP tools

Install the development checkout with `python -m pip install -e '.[mcp]'`.
The server uses stdio transport. An MCP client launches it as a subprocess;
it is not an HTTP endpoint. To launch it manually:

```bash
pyrga-mcp
```

Equivalently, use `python -m pyrga.mcp_server`. Configure your MCP client's
server command to use the Python interpreter from the environment where pyRGA
is installed, with arguments `-m`, `pyrga.mcp_server`. Use an absolute interpreter
path if your client does not inherit that environment. The installed package
can be launched from any working directory.

### Tool arguments and execution order

| Tool | Arguments | Result |
| --- | --- | --- |
| `validate_dataset` | `payload`: canonical dataset object | `valid` plus counts, or structured `errors` |
| `run_forward_model` | `payload`: canonical dataset object | `ForwardResult`: grid, well IDs, head fields, summary, solver version |
| `render_forward_result` | `result`: full forward-result object; `output_directory`: directory path | Paths to `index.html` and `hydraulic-head.svg` |

Call `validate_dataset` first and correct any reported errors before computing.
It checks dataset structure, units, numerical values, and well references;
`run_forward_model` additionally requires square grid counts, square physical
cells, and at least one nonzero-rate well. Pass the full returned forward
object to `render_forward_result`, rather than a text summary. Relative output
paths resolve against the server process's working directory; absolute paths
make artifact locations explicit. Reports are written on the machine running
the server, and files with the same names in that directory are overwritten.

### Runnable MCP client example

Save this as `mcp_forward.py` and run `python mcp_forward.py` in the installed
Python environment. It launches the server, discovers its tools, validates a
small dataset, solves it, and writes a report under `output/mcp-forward/`.

```python
import asyncio
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    payload = {
        "name": "Small aquifer",
        "grid": {"nx": 2, "ny": 2, "length_x": 20, "length_y": 20},
        "conductivity": [1e-4] * 4,
        "wells": [{"id": "PW-1", "node": 4, "pumping_rate": -0.001}],
    }
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "pyrga.mcp_server"]
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Available tools:", [tool.name for tool in tools.tools])

            async def call(name, arguments):
                response = await session.call_tool(name, arguments=arguments)
                if response.isError or response.structuredContent is None:
                    raise RuntimeError(response.content)
                return response.structuredContent

            assessment = await call("validate_dataset", {"payload": payload})
            if not assessment["valid"]:
                raise ValueError(assessment["errors"])
            result = await call("run_forward_model", {"payload": payload})
            print("Head summary:", result["summary"])
            artifacts = await call("render_forward_result", {
                "result": result,
                "output_directory": str(Path("output/mcp-forward").resolve()),
            })
            print("Report:", artifacts["report"])


if __name__ == "__main__":
    asyncio.run(main())
```

For an agent connected to this MCP server, an example request is:

> Validate the attached canonical dataset with `validate_dataset`. If it is
> valid and meets the forward solver's grid requirements, run
> `run_forward_model`, then pass the full result to `render_forward_result`
> using an absolute output directory. Return the head summary and report paths.

The tools accept parsed dataset objects, not CSV or JSON file paths. For local
files, use `pyrga.io.load_dataset_json` or `pyrga.io.dataset_from_csv` in Python,
then pass `dataset.model_dump(mode="json")` as `payload`.

### Inverse optimization and MCP

The experiment tools support seeded truth generation, explicit pumping-test
measurements, and truth-free inverse optimization:

| Tool | Arguments | Result |
| --- | --- | --- |
| `experiment_schema` | None | Shared JSON Schema bundle |
| `generate_field` | `payload`: `GenerateFieldRequest` | `GeneratedField` |
| `run_pumping_test` | `payload`: `PumpingTestRequest` | Immutable `PumpingTestResult` with seeded noise |
| `prepare_inverse_request` | `experiment`: `ExperimentExport` | `InverseRequest` for the selected runs, excluding truth and clean heads |
| `run_inverse_model` | `payload`: `InverseRequest` | Estimated conductivity, predictions, objective history and convergence status |

These tools execute directly. For longer computations, use
`submit_experiment_job(kind, payload)` instead, where `kind` is
`generate_field`, `run_pumping_test`, or `run_inverse_model`. Poll with
`get_experiment_job(job_id)`, retrieve a successful result with
`get_experiment_job_result(job_id)`, and cancel queued or running work with
`cancel_experiment_job(job_id)`. Each MCP server process owns its own job queue;
HTTP job IDs and MCP job IDs are not interchangeable.

The inverse workflow builds its PCA basis independently from truth, permits a
different structural prior, and minimizes uncertainty-weighted head residuals
plus a prior penalty on PCA coefficients. The mean log-conductivity is fitted
without that penalty. Truth is available for comparison, not to the optimizer.
The earlier complete optimization example remains the legacy Python interface.
Transient simulation is still available through Python only.

See the [experiment API contract](docs/experiment-api.md) for full payloads,
defaults, noise conventions, progress, cancellation, and export/import rules.

## Local portal backend

Install and start the HTTP service from this repository:

```bash
python -m pip install -e '.[web,mcp]'
pyrga-api
```

It listens at `http://127.0.0.1:8000`. Open `/docs` for the interactive API,
`/openapi.json` for HTTP schemas, or `/api/v1/schema` for the shared experiment
models. Use `pyrga-api --port 8001` to select another port. The optional `web`
extra is not required for ordinary Python or MCP use.

Submit jobs to `POST /api/v1/jobs`, poll `GET /api/v1/jobs/{job_id}`, retrieve
`GET /api/v1/jobs/{job_id}/result`, or cancel with `DELETE /api/v1/jobs/{job_id}`.
The local service runs one isolated compute process at a time and stores jobs
in memory; export experiments before restarting it. It is a local, single-process
development service, not a hosted multi-user deployment.

The frontend can build against the [checked-in schema bundle](docs/experiment-api.schema.json)
without starting Python. Regenerate it with:

```bash
python -m pyrga.api_contract > docs/experiment-api.schema.json
```

Python callers use the same workflows directly:

```python
from pyrga import (
    ExperimentExport, GenerateFieldRequest, InstalledWell,
    PumpingTestRequest, generate_field, run_pumping_test, run_inverse,
)

field = generate_field(GenerateFieldRequest(seed=42))
wells = [InstalledWell(id="PW-1", node=544),
         InstalledWell(id="MW-1", node=272),
         InstalledWell(id="MW-2", node=816)]  # Default 32 x 32 grid
test = run_pumping_test(PumpingTestRequest(
    field=field, wells=wells, pump_well_id="PW-1", test_id="test-1",
))
experiment = ExperimentExport(
    name="Seeded aquifer", field=field, wells=wells,
    tests=[test], selected_test_ids=[test.test_id],
    inverse_prior=field.request.structure,
)
estimate = run_inverse(experiment.inverse_request())
saved = ExperimentExport.model_validate({
    **experiment.model_dump(), "inverse_result": estimate.model_dump(),
})
export_json = saved.model_dump_json(indent=2)
reloaded = ExperimentExport.model_validate_json(export_json)
```

Generation defaults to a 32 x 32 grid over 320 x 320 m. Correlation lengths
are in metres; variance is for natural-log conductivity. Each pumping test
uses one pump, selected monitors, a recorded noise seed, and independent
Gaussian errors with standard deviation
`sqrt((relative * abs(clean_head))**2 + absolute_floor**2)`.
Defaults are 5% relative noise and a `0.001 m` absolute floor. Set both to zero
for noiseless measurements. Repeating a test with a new `test_id` preserves
its own measurements; change the noise seed to obtain a new noise realization.

An `ExperimentExport` includes materialized truth, seeds, well snapshots, all
collected tests, selected test IDs, inverse settings, and an optional inverse
result. Export it as JSON in the frontend, or write `export_json` to a file in
Python. Import validates field identity, references, and stale results. Existing
test snapshots survive later edits to the current installed-well list.

## Frontend repository

The portal source is maintained in the separate `prRGA-portal` repository
(a sibling directory at `../prRGA-portal/`). This repository contains the Python
library, numerical solvers, canonical models, and MCP server. The portal has its
own Git history, Node dependencies, tests, and deployment configuration.

The portal currently supports editing/loading dataset JSON and browser-side
structure checks. It is not yet connected to the backend solver or MCP server;
authoritative validation and numerical execution use the Python/MCP interfaces
above.

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
