# Local experiment API contract — version 1.0

This is the integration contract for the sibling `prRGA-portal` frontend. Python,
HTTP, and MCP share the same experiment models and numerical workflows. The first
release supports independent seeded conductivity fields, single-pump steady-state
tests, noisy monitoring measurements, and PCA-based inverse estimation.

## Start the local service

From the `pyRGA` checkout, with Python 3.10 or newer:

```bash
python -m pip install -e '.[web,mcp]'
pyrga-api
```

The API listens on `http://127.0.0.1:8000`. Use `pyrga-api --port 8001` to change
the port. The CLI binds to loopback. The frontend is a separate process and
repository; no frontend files are served by this API.

Browser origins `http://localhost:3000` and `http://127.0.0.1:3000` are allowed.
Set `PYRGA_PORTAL_ORIGIN` before starting the service to allow one additional
exact origin, for example `http://localhost:5173`. Requests do not use cookies
or authentication. This is a local development service, not a public hosted API.

Run one service process. Do not configure multiple Uvicorn workers: job IDs,
queues, and results live in process memory and are not shared between workers.

## Machine-readable schemas

The authoritative schemas come from the Pydantic models, including defaults,
field bounds, and rejection of unknown properties:

- `GET /api/v1/schema`: versioned bundle containing request, result, export, and
  job schemas under `schemas`, keyed by model name.
- `GET /openapi.json`: HTTP operations and their request/response definitions.
- `GET /docs`: interactive HTTP documentation.
- [`experiment-api.schema.json`](experiment-api.schema.json): checked-in schema
  bundle for frontend development without a running Python service.

Regenerate the checked-in bundle after changing the models:

```bash
python -m pyrga.api_contract > docs/experiment-api.schema.json
```

The bundle has the shape `{"schema_version":"1.0","schemas":{...}}`. Each model
schema includes its own `$defs`; resolve references relative to that schema.
Some cross-field rules, such as unique well nodes and matching test IDs, are
enforced by Python validators in addition to the generated JSON Schema.

## Scientific and coordinate conventions

All numerical values must be finite. Lengths and hydraulic heads are metres,
conductivity is `m/s`, and pumping rate is `m^3/s`. The two-dimensional model
assumes a one-metre aquifer thickness. Negative rates extract water; positive
rates inject it. Rates are applied directly as nodal loads, without an hourly
or cell-area conversion.

Grids require `nx == ny`, with integer resolution from 2 through 64 inclusive,
and equal physical x/y lengths. The portal may offer 16, 32, and 64 as presets;
smaller supported grids are useful for tests. Domain lengths must be positive.

The origin is bottom left, x increases rightward, and y increases upward. Array
indexing is row-major, with x varying fastest:

- Element conductivity index: `row * nx + column`, with `nx * ny` values.
- Well/head node index: `row * (nx + 1) + column`, with
  `(nx + 1) * (ny + 1)` head values.
- Node coordinates: `x = column * length_x / nx`,
  `y = row * length_y / ny`.
- Element centres: `x = (column + 0.5) * length_x / nx`, and similarly for y.

Snap clicked well positions to nodes, not element centres. The left/right
boundaries have fixed zero head; pumping there is rejected, but monitoring is
allowed. The top/bottom boundaries have zero normal flux. These boundary
conditions are fixed for this release.

Field generation models natural logarithm of conductivity, `ln(K)`. Correlation
lengths are physical metres, not fractions of the domain. Fields use a periodic
FFT grid. For `r² = (length_x_corr * kx)² + (length_y_corr * ky)²`, the discrete
spectral shapes are:

- Gaussian: `exp(-r² / 2)`.
- Matérn: `(1 + r² / (2 * nu))^(-(nu + 1))` in two dimensions.
- Exponential: the Matérn shape with `nu = 0.5`.

The normalized spectrum sets the ensemble marginal variance of `ln(K)` to
`log_variance`. A single realization's spatial mean or variance need not equal
the configured ensemble values. True fields are sampled independently of the
inverse PCA ensemble, even when the structural priors match.
Truth and basis use different random-stream namespaces, so matching their
numeric seeds does not reuse the same realizations.

## Routes and job lifecycle

| Method and route | Success | Meaning |
| --- | --- | --- |
| `GET /api/v1/health` | `200` | `status`, `schema_version`, and `solver_version` |
| `GET /api/v1/schema` | `200` | JSON Schema bundle |
| `POST /api/v1/jobs` | `202` | Validate and enqueue `{kind, payload}`; return `JobStatus` |
| `GET /api/v1/jobs/{job_id}` | `200` | Current `JobStatus`, without result arrays |
| `GET /api/v1/jobs/{job_id}/result` | `200` | Full successful result for that job's kind |
| `DELETE /api/v1/jobs/{job_id}` | `200` | Cancel queued/running work; return terminal/current status |
| `POST /api/v1/experiments/validate` | `200` | Validate and return a canonical `ExperimentExport` |
| `POST /api/v1/experiments/inverse-request` | `200` | Convert selected saved tests into a truth-free `InverseRequest` |

`POST /api/v1/jobs` accepts exactly these kinds:

| `kind` | `payload` model | Successful result model |
| --- | --- | --- |
| `generate_field` | `GenerateFieldRequest` | `GeneratedField` |
| `run_pumping_test` | `PumpingTestRequest` | `PumpingTestResult` |
| `run_inverse_model` | `InverseRequest` | `InverseResult` |

Validation happens before enqueueing. Invalid requests return `422`; a full
queue returns `429` with code `queue_full`. Unknown or expired job IDs return
`404` with code `job_not_found`. A result request before success, including a
failed or cancelled job, returns `409` with code `result_unavailable`.

Error responses for these operations have an `error` object containing `code`
and `message`. Validation errors also include `details` entries with `type`,
`loc`, and `msg`. A valid export without a usable inverse selection returns
`422` from the inverse-request endpoint with code `invalid_inverse_selection`.

Job states are `queued`, `running`, `succeeded`, `failed`, and `cancelled`.
The last three are terminal. A status response includes:

```json
{
  "job_id": "<opaque job identifier>",
  "kind": "run_inverse_model",
  "status": "running",
  "progress": 0.1,
  "stage": "optimizing",
  "message": "<human-readable progress>",
  "detail": {},
  "created_at": "<UTC timestamp>",
  "started_at": "<UTC timestamp or null>",
  "finished_at": null,
  "error": null
}
```

This is a shape illustration, not a request to submit. Progress is a monotonic
fraction from 0 to 1, not a time estimate. `stage` and `message` are display
information; branch on `status` for lifecycle decisions. During inversion,
`detail` can contain `iteration`, `objective`, and `accepted`.

Only one compute process runs at a time; later jobs wait in FIFO order. There
may be up to 16 queued jobs. The service retains the most recent 64 terminal
job records/results, then expires older ones. Fetch completed results promptly
and keep them in the frontend experiment state.

Cancellation removes a queued job or terminates its running worker. It does
not return partial numerical results. Cancelling a terminal job is idempotent
and preserves its existing status; a cancellation that arrives after success
does not undo the result. Poll after cancellation to obtain the authoritative
terminal state. Service shutdown cancels unfinished work. All job state is
lost on restart, so an old job ID can then return `404`.

## Field generation

This is a complete JSON request for `POST /api/v1/jobs`:

```json
{
  "kind": "generate_field",
  "payload": {
    "grid": {"nx": 32, "ny": 32, "length_x": 320, "length_y": 320},
    "structure": {
      "covariance_type": "gaussian",
      "mean_log_conductivity": -9.210340371976184,
      "log_variance": 1,
      "correlation_length_x": 48,
      "correlation_length_y": 64,
      "matern_nu": 0.5
    },
    "seed": 42
  }
}
```

These are also the generation defaults, so `{ "kind": "generate_field",
"payload": {} }` requests the same configuration. `mean_log_conductivity`
must be between -30 and 10, `log_variance` must be positive and at most 10,
correlation lengths must be positive, and Matérn `nu` must be positive and at
most 10. Seeds are integers from 0 through `2^32 - 1`.

`GeneratedField` contains `field_id`, the fully populated generation `request`,
the materialized `conductivity` array, and `solver_version`. Keep the complete
returned object for subsequent pumping-test requests and exports. Treat
`field_id` as an opaque identifier; preserve it rather than inventing or
recomputing it in the browser.

The same generation settings, seed, and numerical version reproduce the field.
Exported materialized arrays preserve the actual experiment even if a later
software version changes random or numerical implementation details.

## Pumping tests and measurements

`PumpingTestRequest` contains:

| Field | Meaning |
| --- | --- |
| `field` | Complete `GeneratedField` returned by generation/import |
| `wells` | 2–25 installed `{id, node}` objects; IDs and nodes must be unique |
| `test_id` | Unique run label; an omitted value gets a UUID |
| `pump_well_id` | Installed well active as pump for this run |
| `pumping_rate` | Nonzero finite rate; defaults to `-0.001` |
| `monitor_well_ids` | Optional ordered subset; defaults to all other wells in installed order |
| `noise` | `{relative, absolute_floor, seed}`; defaults to `{0.05, 0.001, 44}` |

The active pump cannot also monitor in the same run. At least one monitor is
required. Roles belong to each run: a monitoring well can pump in a later run.
Each request solves one pump independently; multiple tests are not simultaneous
loads. Keep repeated runs as distinct test IDs.

For each selected monitor, the service samples its clean nodal head and adds
independent zero-mean Gaussian noise. Its standard deviation is
`hypot(relative * abs(clean_head), absolute_floor)`. Relative noise is a fraction
between 0 and 1; the floor is a nonnegative number of metres. Setting both to
zero produces noiseless observations.

The same noise seed with the same ordered monitors and clean heads reproduces
the noise. Change the seed to obtain an independent repeated measurement;
merely changing `test_id` does not change noise. Truth-generation, basis, and
noise RNGs do not mutate global NumPy random state.

`PumpingTestResult` contains `test_id`, `field_id`, `grid`, the installed `wells`
snapshot, `pump_well_id`, `pumping_rate`, `noise`, complete nodal `heads`, ordered
`measurements`, and `solver_version`. Each measurement is:

```json
{
  "test_id": "<this run's identifier>",
  "well_id": "<selected monitoring well>",
  "node": 0,
  "clean_head": 0.0,
  "head": 0.0,
  "standard_deviation": 0.0
}
```

The numbers illustrate types only. Use returned values; `head` is the noisy
measurement used by inversion, while `clean_head` supports synthetic comparison.
Measurements retain the requested monitoring order. Save each returned test
unchanged: editing the current well layout must not rewrite earlier snapshots.

## Inversion

`InverseRequest` contains `grid`, structural `prior`, `settings`, and 1–64
selected `tests`. It deliberately has no true conductivity, clean-head array,
or true PCA coefficients. Unknown properties are rejected.

Each inverse test contains `test_id`, `pump_node`, `pumping_rate`, and ordered
`observations`. An observation contains only `well_id`, `node`, measured `head`,
and `standard_deviation`. IDs must be unique per test; duplicate monitoring
nodes, the active pump as monitor, and duplicate test IDs are rejected. At least
one monitor across the selected tests must be away from fixed-head boundaries.

Prefer `POST /api/v1/experiments/inverse-request` to constructing inverse rows
by hand. It follows `selected_test_ids` order, looks up each immutable test
snapshot, then preserves that test's measurement order. Thus current well edits
do not change historical pump/monitor coordinates.

The prior has the same structure as generation settings. It may match the
truth-generation structure or deliberately differ. Its ensemble is independently
seeded; PCA uses sample-covariance scaling and canonical SVD signs. The default
inverse settings are:

```json
{
  "components": 20,
  "ensemble_size": 200,
  "seed": 43,
  "max_iterations": 10,
  "tolerance": 0.00001,
  "regularization": 1,
  "initial_damping": 0.001
}
```

Components must be 1–50 and cannot exceed either the element count or
`ensemble_size - 1`. The ensemble size is 2–400, iteration limit is 1–50,
tolerance is positive and at most 1, and regularization/damping are positive.
For tiny test grids, explicitly reduce `components` below its default of 20.

The optimizer starts with zero PCA coefficients and the prior mean. It fits
the weighted MAP objective

```text
0.5 * sum(((predicted_head - measured_head) / effective_sigma)^2)
  + 0.5 * regularization * sum(PCA_coefficients^2)
```

The mean log-conductivity is unpenalized. Positive supplied uncertainties are
used unchanged; zero uncertainties use a numerical floor of
`max(1e-8 m, 1e-6 * max(abs(selected measured heads)))`. This only prevents
division by zero; it does not resample or alter measurements. LM damping is a
numerical step control, distinct from measurement uncertainty and regularization.
PCA coefficients are bounded to [-8, 8], the mean to [-30, 10], and reconstructed
log-conductivity is clipped to [-30, 10] for numerical stability.

`InverseResult` contains:

- The complete truth-free `request`, `estimated_conductivity` with `nx * ny`
  values, fitted `coefficients`, and `mean_log_conductivity`.
- `initial_objective`, `final_objective`, `converged`, and `termination_reason`
  (`converged`, `max_iterations`, or `stalled`). A successful job need not have
  converged; display the termination reason.
- `iterations`, starting at iteration zero. Each entry contains objective,
  `data_misfit`, `prior_penalty`, `step_norm`, `damping`, and `accepted`. A
  rejected step retains the previous accepted state/objective.
- Ordered `predictions`, each with `test_id`, `well_id`, `observed_head`,
  `predicted_head`, effective `standard_deviation`, and residual
  `predicted_head - observed_head`.
- `solver_version`.

The true field remains available in the experiment for frontend comparisons,
but is never passed to the inverse workflow. Compare `ln(K)` arrays or apply a
shared display transform/colour scale; exact recovery of independent truth is
not guaranteed by successful optimization.

## Experiment export/import and stale results

An experiment is a single JSON document with these top-level fields:

```text
schema_version: "1.0"
name: experiment label
field: complete GeneratedField
wells: current installed well layout
tests: complete immutable PumpingTestResult objects
selected_test_ids: ordered subset to use for inversion
inverse_prior: StructureParameters
inverse_settings: InverseSettings
inverse_result: complete InverseResult or null
```

Only `name` and `field` are required initially; other fields have defaults.
If `inverse_prior` is omitted, it copies the generated field's structural
parameters; supply it explicitly to investigate a mismatched prior.
The export supports at most 64 collected tests and 25 current installed wells.
Tests must share the current field ID and grid, and test IDs must be unique.
Selected IDs must be a unique subset of collected tests. Import validation
also checks snapshot references, array shapes, positive conductivity, and
alignment of saved inverse predictions/history with their inverse request.

If `inverse_result` is present, its request must exactly match the request
derived from the current selected test snapshots, inverse prior, and settings.
Clear `inverse_result` before changing those inputs. A stale result is rejected
rather than silently attached to a different experiment.

The frontend owns export/download and file import. These API routes perform
validation/conversion only: they do not save files or maintain an experiment
database. Validate imported JSON through `/experiments/validate`, use the
canonical response as state, and pass it to `/experiments/inverse-request`
before submitting an inverse job.

Frontend invalidation rules:

- Regenerating/replacing truth or changing the grid starts new downstream
  state: clear tests, selected IDs, and inverse results.
- Editing the current installed layout affects future tests only; preserve old
  snapshots. Removing a well from the current layout does not remove its past
  measurements.
- Changing noise/pumping controls affects future runs only.
- Changing selected runs, prior, settings, or measured values clears the inverse
  result. Cancelling a job leaves existing successful experiment data intact.
- Associate each submitted job with its input revision and only apply a result
  if that revision is still current. The server does not know browser state.

## Browser integration example

This JavaScript uses actual prior responses rather than invented field IDs or
head arrays. It generates a field, installs three wells, collects one test,
converts the selected snapshot into inverse input, and saves the validated export.
Call `runExample()` from an allowed local origin. Use `cancel(jobId)` for an
explicit user cancellation; aborting a browser fetch alone does not cancel
server compute.

```javascript
const base = "http://127.0.0.1:8000/api/v1";

async function api(path, options = {}) {
  const response = await fetch(`${base}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const value = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(value.error ?? value));
  return value;
}

async function job(kind, payload, onProgress = console.log) {
  let status = await api("/jobs", {
    method: "POST", body: JSON.stringify({ kind, payload }),
  });
  const id = status.job_id; // Store in UI state so its Cancel button can use it.
  while (true) {
    onProgress(status);
    if (status.status === "succeeded") return api(`/jobs/${id}/result`);
    if (status.status === "failed") throw new Error(status.error?.message);
    if (status.status === "cancelled") throw new Error("Computation cancelled");
    await new Promise(resolve => setTimeout(resolve, 500));
    status = await api(`/jobs/${id}`);
  }
}

const cancel = id => api(`/jobs/${id}`, { method: "DELETE" });

async function runExample() {
  const field = await job("generate_field", {});
  const { nx } = field.request.grid;
  const node = (row, column) => row * (nx + 1) + column;
  const wells = [
    { id: "PW-1", node: node(nx / 2, nx / 2) },
    { id: "MW-1", node: node(nx / 4, nx / 4) },
    { id: "MW-2", node: node(3 * nx / 4, 3 * nx / 4) },
  ];
  const test = await job("run_pumping_test", {
    field, wells, test_id: crypto.randomUUID(), pump_well_id: "PW-1",
    pumping_rate: -0.001, monitor_well_ids: ["MW-1", "MW-2"],
    noise: { relative: 0.05, absolute_floor: 0.001, seed: 44 },
  });
  let experiment = await api("/experiments/validate", {
    method: "POST", body: JSON.stringify({
      schema_version: "1.0", name: "First experiment", field, wells,
      tests: [test], selected_test_ids: [test.test_id],
      inverse_prior: field.request.structure,
    }),
  });
  const inverseRequest = await api("/experiments/inverse-request", {
    method: "POST", body: JSON.stringify(experiment),
  });
  const inverse = await job("run_inverse_model", inverseRequest);
  experiment = await api("/experiments/validate", {
    method: "POST", body: JSON.stringify({ ...experiment, inverse_result: inverse }),
  });
  const url = URL.createObjectURL(new Blob([JSON.stringify(experiment, null, 2)], {
    type: "application/json",
  }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "pyrga-experiment.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return experiment;
}
```

## MCP and Python equivalents

Start stdio MCP with `pyrga-mcp` or `python -m pyrga.mcp_server`. The same request
objects are accepted by the synchronous tools:

| MCP tool | Arguments | Result |
| --- | --- | --- |
| `generate_field` | `{payload: GenerateFieldRequest}` | `GeneratedField` |
| `run_pumping_test` | `{payload: PumpingTestRequest}` | `PumpingTestResult` |
| `run_inverse_model` | `{payload: InverseRequest}` | `InverseResult` |
| `experiment_schema` | `{}` | Versioned schema bundle |
| `prepare_inverse_request` | `{experiment: ExperimentExport}` | `InverseRequest` |

Synchronous tools wait for the computation to finish. For long computations
requiring polling or cancellation, use these MCP job tools:

| MCP tool | Arguments | Result |
| --- | --- | --- |
| `submit_experiment_job` | `{kind, payload}` | `JobStatus` |
| `get_experiment_job` | `{job_id}` | `JobStatus` |
| `get_experiment_job_result` | `{job_id}` | Successful result |
| `cancel_experiment_job` | `{job_id}` | Current/terminal `JobStatus` |

Each MCP server instance owns its own in-memory job manager. An HTTP job ID is
not usable in a separate MCP server, and vice versa. Tool errors use the MCP
error mechanism, not HTTP status codes. Synchronous computations do not occupy
the job queue. The legacy `validate_dataset`, `run_forward_model`, and
`render_forward_result` tools remain available with their existing contracts;
they are separate from the experiment models in this document.

Python callers can directly use `generate_field`, `run_pumping_test`, and
`run_inverse` with the corresponding validated models. Each accepts optional
`progress(dict)` and `cancelled() -> bool` callbacks. The direct workflows return
Pydantic result models and do not write files. The legacy report-rendering tool
is the explicit file-writing interface.
