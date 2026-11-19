"""Numerical regressions for measurement-driven inverse experiments."""

import unittest
import importlib
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from pyrga.experiment_models import (
    InverseRequest, InverseSettings, InverseTest, ObservedHead, StructureParameters,
)
from pyrga.gwsolver import groundwater_solver
from pyrga.inverse import _fit_map, run_inverse
from pyrga.models import GridSpec


class MapOptimizerTests(unittest.TestCase):
    def test_linear_map_solution_leaves_mean_unpenalized(self):
        def forward(parameters):
            alpha, mean = parameters
            return np.array([alpha + mean, mean])

        parameters, history, status = _fit_map(
            forward, np.zeros(2), np.array([2.0, 1.0]), np.ones(2),
            max_iterations=10, regularization=1.0, damping=1e-3,
            tolerance=1e-9, notify=lambda entry: None, check_cancelled=lambda: None,
        )
        np.testing.assert_allclose(parameters, [1 / 3, 4 / 3], atol=1e-6)
        self.assertEqual(status, "converged")
        self.assertLess(history[-1]["objective"], history[0]["objective"])
        for entry in history:
            candidate = np.array(entry["coefficients"] + [entry["mean_log_conductivity"]])
            expected = 0.5 * np.sum((forward(candidate) - [2, 1]) ** 2) + 0.5 * candidate[0] ** 2
            self.assertAlmostEqual(entry["objective"], expected)

    def test_failed_trial_increases_damping_and_retries(self):
        rejected = []

        def forward(parameters):
            if parameters[-1] > 3:
                rejected.append(parameters.copy())
                raise ValueError("trial conductivity is ill conditioned")
            return np.array([np.exp(parameters[-1])])

        parameters, history, status = _fit_map(
            forward, np.zeros(2), np.array([np.exp(2.5)]), np.ones(1),
            max_iterations=20, regularization=1, damping=1e-3,
            tolerance=1e-8, notify=lambda entry: None, check_cancelled=lambda: None,
        )
        self.assertTrue(rejected)
        self.assertEqual(status, "converged")
        self.assertAlmostEqual(parameters[-1], 2.5, places=5)
        self.assertLess(history[-1]["objective"], history[0]["objective"])


class InverseWorkflowTests(unittest.TestCase):
    def make_request(self, sigma=0.01):
        grid = GridSpec(nx=4, ny=4, length_x=40, length_y=40)
        conductivity = np.exp(-9.2 + np.linspace(-0.2, 0.2, 16))
        tests = []
        # Monitoring subsets and their order differ deliberately by test.
        for test_id, pump, rate, monitors in (
            ("run-2", 6, -0.001, [("C", 17), ("A", 7)]),
            ("run-1", 18, -0.0007, [("B", 12), ("A", 7)]),
        ):
            heads = groundwater_solver(conductivity, pump, rate)
            tests.append(InverseTest(
                test_id=test_id, pump_node=pump, pumping_rate=rate,
                observations=[ObservedHead(well_id=well_id, node=node,
                                           head=float(heads[node]),
                                           standard_deviation=sigma)
                              for well_id, node in monitors],
            ))
        return InverseRequest(
            grid=grid, prior=StructureParameters(
                mean_log_conductivity=-8.8, correlation_length_x=8,
                correlation_length_y=10,
            ),
            settings=InverseSettings(components=3, ensemble_size=12,
                                     max_iterations=6, seed=7),
            tests=tests,
        )

    def test_explicit_measurement_order_and_objective_reduction(self):
        result = run_inverse(self.make_request())
        self.assertLess(result.final_objective, 0.1 * result.initial_objective)
        self.assertTrue(np.isfinite(result.estimated_conductivity).all())
        self.assertEqual(len(result.estimated_conductivity), 16)
        self.assertEqual([(item.test_id, item.well_id) for item in result.predictions],
                         [("run-2", "C"), ("run-2", "A"),
                          ("run-1", "B"), ("run-1", "A")])
        self.assertTrue(all(current.objective <= previous.objective
                            for previous, current in zip(result.iterations, result.iterations[1:])))
        # Reported predictions must correspond to the returned field, not a
        # rejected trial or a different final parameter vector.
        index = 0
        for test in result.request.tests:
            predicted = groundwater_solver(np.array(result.estimated_conductivity),
                                           test.pump_node, test.pumping_rate)
            for observed in test.observations:
                self.assertAlmostEqual(result.predictions[index].predicted_head,
                                       predicted[observed.node], places=4)
                index += 1

    def test_seeded_repeated_fit_is_identical(self):
        request = self.make_request()
        first = run_inverse(request)
        second = run_inverse(request)
        self.assertEqual(first.model_dump(), second.model_dump())

    def test_basis_uses_independent_stream_even_with_same_numeric_seed(self):
        experiments = importlib.import_module("pyrga.experiments")

        request = self.make_request()
        with patch.object(experiments, "sample_log_fields", wraps=experiments.sample_log_fields) as sample:
            run_inverse(request)
        self.assertEqual(sample.call_args.kwargs["stream"], 1)
        truth_sample = experiments.sample_log_fields(request.grid, request.prior,
                                                    request.settings.seed, count=1, stream=0)
        inverse_sample = experiments.sample_log_fields(request.grid, request.prior,
                                                      request.settings.seed, count=1, stream=1)
        self.assertFalse(np.array_equal(truth_sample, inverse_sample))

    def test_zero_noise_uses_finite_numerical_uncertainty(self):
        result = run_inverse(self.make_request(sigma=0))
        self.assertTrue(np.isfinite(result.final_objective))
        self.assertLess(result.final_objective, result.initial_objective)
        self.assertTrue(all(item.standard_deviation > 0 for item in result.predictions))

    def test_noisy_observations_and_mismatched_prior_fit_without_truth(self):
        payload = self.make_request(sigma=0.02).model_dump()
        payload["prior"]["correlation_length_x"] = 25
        payload["prior"]["correlation_length_y"] = 4
        random = np.random.default_rng(21)
        for test in payload["tests"]:
            for observation in test["observations"]:
                observation["head"] += float(random.normal(scale=0.02))
        result = run_inverse(InverseRequest.model_validate(payload))
        self.assertLess(result.final_objective, result.initial_objective)
        self.assertTrue(np.isfinite(result.estimated_conductivity).all())

    def test_ill_conditioned_trial_is_rejected_without_aborting_fit(self):
        from pyrga.experiment_models import GenerateFieldRequest
        from pyrga.experiments import generate_field, solve_heads

        grid = GridSpec(nx=8, ny=8, length_x=80, length_y=80)
        truth_structure = StructureParameters(
            mean_log_conductivity=-15, log_variance=5,
            correlation_length_x=10, correlation_length_y=10,
        )
        field = generate_field(GenerateFieldRequest(grid=grid,
                                                    structure=truth_structure, seed=42))
        tests = []
        for pump in (30, 50):
            head = solve_heads(grid, field.conductivity, [pump], [-0.001])[0]
            tests.append(InverseTest(
                test_id=str(pump), pump_node=pump, pumping_rate=-0.001,
                observations=[ObservedHead(well_id=str(node), node=node,
                                           head=float(head[node]), standard_deviation=0.01)
                              for node in (20, 40, 60)],
            ))
        prior = truth_structure.model_copy(update={"mean_log_conductivity": -8})
        result = run_inverse(InverseRequest(
            grid=grid, prior=prior, tests=tests,
            settings=InverseSettings(components=10, ensemble_size=30, max_iterations=5),
        ))
        self.assertTrue(np.isfinite(result.final_objective))
        self.assertLessEqual(result.final_objective, result.initial_objective)
        self.assertTrue(np.isfinite(result.estimated_conductivity).all())

    def test_inverse_input_rejects_truth(self):
        payload = self.make_request().model_dump()
        payload["true_conductivity"] = [1e-4] * 16
        with self.assertRaises(ValidationError):
            InverseRequest.model_validate(payload)

    def test_cancelled_fit_does_not_return_a_result(self):
        from pyrga.experiments import WorkflowCancelled

        events = []
        with self.assertRaises(WorkflowCancelled):
            run_inverse(self.make_request(), progress=events.append,
                        cancelled=lambda: bool(events and events[-1].get("iteration") == 1))
        self.assertTrue(any(item.get("iteration") == 1 for item in events))


if __name__ == "__main__":
    unittest.main()
