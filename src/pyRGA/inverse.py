"""Seeded, measurement-driven inverse workflows without truth-field inputs.

The prior penalizes PCA coefficients while leaving the mean log conductivity
free. Measurement uncertainty and numerical damping have distinct roles.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .experiment_models import (
    InverseIteration, InverseRequest, InverseResult, PredictedMeasurement,
)


def _fit_map(
    forward: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    observed: np.ndarray,
    uncertainty: np.ndarray,
    *,
    max_iterations: int,
    regularization: float,
    damping: float,
    tolerance: float,
    notify: Callable[[dict], None],
    check_cancelled: Callable[[], None],
) -> tuple[np.ndarray, list[dict], str]:
    """Fit whitened measurements with bounded, damped Gauss--Newton steps."""
    parameters = initial.copy()
    penalty = np.ones(parameters.size)
    penalty[-1] = 0.0

    def objective(candidate, predictions):
        residual = (predictions - observed) / uncertainty
        return float(0.5 * residual @ residual +
                     0.5 * regularization * np.dot(candidate[:-1], candidate[:-1]))

    check_cancelled()
    predictions = forward(parameters)
    value = objective(parameters, predictions)
    if not np.isfinite(value):
        raise ValueError("initial inverse objective is not finite; check measurement scales")
    history = []

    def record(iteration, accepted, step_norm):
        entry = {
            "iteration": iteration,
            "objective": value,
            "data_misfit": float(0.5 * np.sum(((predictions - observed) / uncertainty) ** 2)),
            "regularization": float(0.5 * regularization * np.sum(parameters[:-1] ** 2)),
            "damping": damping,
            "accepted": accepted,
            "step_norm": step_norm,
            "mean_log_conductivity": float(parameters[-1]),
            "coefficients": parameters[:-1].tolist(),
        }
        history.append(entry)
        notify(entry)

    record(0, True, 0.0)
    status = "max_iterations"
    for iteration in range(1, max_iterations + 1):
        check_cancelled()
        jacobian = np.empty((observed.size, parameters.size), dtype=float)
        for index in range(parameters.size):
            check_cancelled()
            epsilon = 1e-4 * max(1.0, abs(parameters[index]))
            perturbed = parameters.copy()
            # Use an inward finite difference at the upper bound.
            if parameters[index] + epsilon > (10.0 if index == parameters.size - 1 else 8.0):
                epsilon = -epsilon
            perturbed[index] += epsilon
            jacobian[:, index] = (forward(perturbed) - predictions) / (epsilon * uncertainty)

        residual = (predictions - observed) / uncertainty
        hessian = jacobian.T @ jacobian + regularization * np.diag(penalty)
        gradient = jacobian.T @ residual + regularization * penalty * parameters
        if not np.isfinite(hessian).all() or not np.isfinite(gradient).all():
            raise ValueError("inverse derivatives are not finite; check prior and measurement scales")
        if np.linalg.norm(gradient, ord=np.inf) <= tolerance:
            status = "converged"
            break

        previous_value = value
        accepted = False
        actual_step = np.zeros_like(parameters)
        # A rejected trial leaves parameters/predictions/objective unchanged.
        # Retry with stronger damping before taking another Jacobian.
        for _ in range(8):
            check_cancelled()
            scaling = np.maximum(np.diag(hessian), 1.0)
            try:
                step = np.linalg.solve(hessian + damping * np.diag(scaling), -gradient)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = parameters + step
            candidate[:-1] = np.clip(candidate[:-1], -8.0, 8.0)
            candidate[-1] = np.clip(candidate[-1], -30.0, 10.0)
            try:
                trial_predictions = forward(candidate)
            except ValueError:
                # A high-contrast trial can fail the forward solver's residual
                # check even though the current accepted field is well posed.
                # Reject that trial and shorten the step; cancellation uses a
                # distinct RuntimeError subclass and must propagate unchanged.
                damping *= 10.0
                continue
            trial_value = objective(candidate, trial_predictions)
            if np.isfinite(trial_value) and trial_value < value:
                actual_step = candidate - parameters
                parameters = candidate
                predictions = trial_predictions
                value = trial_value
                damping = max(damping / 3.0, 1e-12)
                accepted = True
                break
            damping *= 10.0

        step_norm = float(np.linalg.norm(actual_step))
        record(iteration, accepted, step_norm)
        if accepted and (step_norm <= tolerance * (1.0 + np.linalg.norm(parameters)) or
                         previous_value - value <= tolerance * max(1.0, previous_value)):
            status = "converged"
            break
        if not accepted:
            status = "stalled"
            break

    return parameters, history, status


def run_inverse(
    request: InverseRequest,
    *,
    progress: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> InverseResult:
    """Estimate conductivity from explicit pumping/monitor measurement rows.

    Neither the true field nor clean synthetic observations are accepted by
    ``InverseRequest``. All uncertainty-weighted residuals are based solely on
    the measurements selected by the caller. Zero-uncertainty observations use
    a numerical standard-deviation floor of max(1e-8 m, 1e-6 max(abs(head))).
    """
    from . import __version__
    from .experiments import WorkflowCancelled, sample_log_fields, solve_heads

    request = InverseRequest.model_validate(request.model_dump())
    settings = request.settings

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise WorkflowCancelled("inverse computation cancelled")

    def emit(event):
        if progress is not None:
            progress(event)

    check_cancelled()
    emit({"progress": 0.0, "stage": "basis", "message": "Generating independent prior ensemble"})
    ensemble = sample_log_fields(
        request.grid, request.prior, settings.seed, count=settings.ensemble_size, stream=1
    )
    check_cancelled()
    # Singular values are sqrt((N - 1) * covariance eigenvalues). Dividing by
    # sqrt(N - 1), rather than taking sqrt(singular_values), gives a covariance
    # square root whose coefficients have independent unit-normal priors.
    centred = ensemble - ensemble.mean(axis=0, keepdims=True)
    _, singular_values, vectors = np.linalg.svd(centred, full_matrices=False)
    basis = vectors[:settings.components].T * (
        singular_values[:settings.components] / np.sqrt(settings.ensemble_size - 1)
    )
    # SVD signs are arbitrary; canonical signs stabilize seeded coefficient
    # interpretation across repeat calculations with the same numerical stack.
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    check_cancelled()

    observed = np.array([
        observation.head for test in request.tests for observation in test.observations
    ], dtype=float)
    uncertainty = np.array([
        observation.standard_deviation
        for test in request.tests for observation in test.observations
    ], dtype=float)
    numerical_floor = max(1e-8, 1e-6 * float(np.max(np.abs(observed))))
    uncertainty = np.where(uncertainty > 0, uncertainty, numerical_floor)

    def conductivity(parameters):
        log_field = basis @ parameters[:-1] + parameters[-1]
        return np.exp(np.clip(log_field, -30.0, 10.0))

    def predict(parameters):
        check_cancelled()
        head_fields = solve_heads(
            request.grid, conductivity(parameters),
            [test.pump_node for test in request.tests],
            [test.pumping_rate for test in request.tests],
            cancelled=cancelled,
        )
        check_cancelled()
        predictions = np.array([
            head_fields[test_index, observation.node]
            for test_index, test in enumerate(request.tests)
            for observation in test.observations
        ], dtype=float)
        if not np.isfinite(predictions).all():
            raise ValueError("forward predictions became non-finite during inversion")
        return predictions

    def notify(entry):
        emit({
            "progress": min(0.95, 0.1 + 0.85 * entry["iteration"] / settings.max_iterations),
            "stage": "optimizing",
            "message": f"Inverse iteration {entry['iteration']}",
            "iteration": entry["iteration"], "objective": entry["objective"],
            "accepted": entry["accepted"],
        })

    initial = np.zeros(settings.components + 1)
    initial[-1] = request.prior.mean_log_conductivity
    parameters, history, status = _fit_map(
        predict, initial, observed, uncertainty,
        max_iterations=settings.max_iterations,
        regularization=settings.regularization,
        damping=settings.initial_damping,
        tolerance=settings.tolerance,
        notify=notify, check_cancelled=check_cancelled,
    )
    check_cancelled()
    final_predictions = predict(parameters)
    predictions = []
    row = 0
    for test in request.tests:
        for observation in test.observations:
            predictions.append(PredictedMeasurement(
                test_id=test.test_id, well_id=observation.well_id,
                observed_head=observation.head,
                predicted_head=float(final_predictions[row]),
                standard_deviation=float(uncertainty[row]),
                residual=float(final_predictions[row] - observation.head),
            ))
            row += 1
    result = InverseResult(
        request=request,
        estimated_conductivity=conductivity(parameters).tolist(),
        coefficients=parameters[:-1].tolist(),
        mean_log_conductivity=float(parameters[-1]),
        initial_objective=history[0]["objective"],
        final_objective=history[-1]["objective"],
        converged=status == "converged", termination_reason=status,
        iterations=[InverseIteration(
            iteration=entry["iteration"], objective=entry["objective"],
            data_misfit=entry["data_misfit"], prior_penalty=entry["regularization"],
            step_norm=entry["step_norm"], damping=entry["damping"],
            accepted=entry["accepted"],
        ) for entry in history],
        predictions=predictions, solver_version=__version__,
    )
    emit({"progress": 1.0, "stage": "complete", "message": f"Inverse fit {status}"})
    return result
