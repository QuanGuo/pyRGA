"""Regression tests for reproducible synthetic tomography experiments."""

import json
import unittest

import numpy as np
from pydantic import ValidationError

from pyrga.experiment_models import (
    ExperimentExport,
    GenerateFieldRequest,
    GeneratedField,
    InstalledWell,
    InverseSettings,
    InverseResult,
    NoiseSettings,
    PumpingTestRequest,
    StructureParameters,
)
from pyrga.experiments import generate_field, run_pumping_test, sample_log_fields, solve_heads
from pyrga.models import GridSpec


def field_request(seed=42):
    return GenerateFieldRequest(
        grid=GridSpec(nx=4, ny=4, length_x=40, length_y=40),
        structure=StructureParameters(correlation_length_x=8, correlation_length_y=12),
        seed=seed,
    )


def uniform_field():
    request = GenerateFieldRequest(grid=GridSpec(nx=2, ny=2, length_x=20, length_y=20))
    return GeneratedField(
        field_id=GeneratedField.content_id(request, [1e-4] * 4, "test"),
        request=request,
        conductivity=[1e-4] * 4,
        solver_version="test",
    )


def pumping_request(**changes):
    payload = {
        "field": uniform_field(),
        "test_id": "test-a",
        "wells": [
            InstalledWell(id="pump", node=4),
            InstalledWell(id="bottom", node=1),
            InstalledWell(id="top", node=7),
            InstalledWell(id="boundary", node=3),
        ],
        "pump_well_id": "pump",
        "pumping_rate": -0.001,
        "noise": NoiseSettings(relative=0, absolute_floor=0),
    }
    payload.update(changes)
    return PumpingTestRequest(**payload)


class FieldGenerationTests(unittest.TestCase):
    def test_legacy_generator_accepts_an_isolated_random_generator(self):
        from pyrga.RGA import generate_synthetic_field
        first = generate_synthetic_field(4, 4, k=3, NR=8, rng=np.random.default_rng(123))
        second = generate_synthetic_field(4, 4, k=3, NR=8, rng=np.random.default_rng(123))
        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual, expected)

    def test_materialized_truth_cannot_change_without_a_new_identity(self):
        payload = generate_field(field_request()).model_dump()
        payload["conductivity"][0] *= 2
        with self.assertRaisesRegex(ValidationError, "field_id"):
            GeneratedField.model_validate(payload)

    def test_generation_is_reproducible_and_seed_changes_the_field(self):
        first = generate_field(field_request())
        replay = generate_field(field_request())
        changed = generate_field(field_request(seed=43))
        self.assertEqual(first, replay)
        self.assertNotEqual(first.field_id, changed.field_id)
        self.assertFalse(np.array_equal(first.conductivity, changed.conductivity))
        self.assertEqual(len(first.conductivity), 16)
        self.assertTrue(np.isfinite(first.conductivity).all())
        self.assertTrue((np.asarray(first.conductivity) > 0).all())

    def test_generation_does_not_change_numpy_global_random_state(self):
        saved = np.random.get_state()
        try:
            np.random.seed(184)
            expected = np.random.normal(size=8)
            np.random.seed(184)
            generate_field(field_request())
            actual = np.random.normal(size=8)
            np.testing.assert_array_equal(actual, expected)
        finally:
            np.random.set_state(saved)

    def test_covariance_variants_produce_valid_seeded_fields(self):
        for covariance_type in ("gaussian", "exponential", "matern"):
            with self.subTest(covariance_type=covariance_type):
                payload = field_request().model_dump()
                payload["structure"]["covariance_type"] = covariance_type
                request = GenerateFieldRequest.model_validate(payload)
                first = generate_field(request)
                self.assertEqual(first, generate_field(request))
                self.assertTrue(np.isfinite(first.conductivity).all())

    def test_spectral_samples_have_requested_marginal_mean_and_variance(self):
        request = field_request()
        # Check the ensemble at one location, not a realization's spatial
        # variance: correlated fields need not have zero spatial mean.
        samples = sample_log_fields(request.grid, request.structure, 112, count=8192)
        location = samples[:, 0]
        self.assertAlmostEqual(location.mean(), request.structure.mean_log_conductivity, delta=0.04)
        self.assertAlmostEqual(location.var(), request.structure.log_variance, delta=0.06)


class PumpingMeasurementTests(unittest.TestCase):
    def test_known_si_solution_and_default_monitors(self):
        result = run_pumping_test(pumping_request())
        np.testing.assert_allclose(
            result.heads, [0, -1, 0, 0, -4, 0, 0, -1, 0], atol=1e-7
        )
        self.assertEqual([m.well_id for m in result.measurements], ["bottom", "top", "boundary"])
        self.assertEqual([m.test_id for m in result.measurements], ["test-a"] * 3)
        np.testing.assert_allclose([m.head for m in result.measurements], [-1, -1, 0], atol=1e-7)
        self.assertTrue(all(m.head == m.clean_head for m in result.measurements))
        self.assertTrue(all(m.standard_deviation == 0 for m in result.measurements))

    def test_monitor_subset_preserves_explicit_order(self):
        result = run_pumping_test(pumping_request(monitor_well_ids=["boundary", "top"]))
        self.assertEqual([m.well_id for m in result.measurements], ["boundary", "top"])
        self.assertEqual([m.node for m in result.measurements], [3, 7])

    def test_noise_matches_relative_plus_absolute_floor_formula(self):
        noise = NoiseSettings(relative=0.05, absolute_floor=0.02, seed=999)
        request = pumping_request(noise=noise)
        result = run_pumping_test(request)
        clean = np.asarray([m.clean_head for m in result.measurements])
        sigma = np.sqrt((noise.relative * np.abs(clean)) ** 2 + noise.absolute_floor**2)
        np.testing.assert_allclose([m.standard_deviation for m in result.measurements], sigma)
        expected = clean + np.random.default_rng(noise.seed).normal(size=clean.size) * sigma
        np.testing.assert_allclose([m.head for m in result.measurements], expected)
        self.assertGreater(result.measurements[-1].standard_deviation, 0)
        self.assertEqual(result, run_pumping_test(request))

    def test_noise_seed_changes_only_measurement_noise(self):
        first = run_pumping_test(pumping_request(noise=NoiseSettings(seed=8)))
        second = run_pumping_test(pumping_request(noise=NoiseSettings(seed=9)))
        self.assertEqual(first.heads, second.heads)
        self.assertEqual([m.clean_head for m in first.measurements], [m.clean_head for m in second.measurements])
        self.assertNotEqual([m.head for m in first.measurements], [m.head for m in second.measurements])

    def test_opposite_rate_reverses_head_and_repeated_tests_keep_ids(self):
        extraction = run_pumping_test(pumping_request())
        injection = run_pumping_test(pumping_request(test_id="test-b", pumping_rate=0.001))
        np.testing.assert_allclose(injection.heads, -np.asarray(extraction.heads), atol=1e-7)
        self.assertNotEqual(extraction.test_id, injection.test_id)

    def test_multiple_pump_right_hand_sides_match_independent_solves(self):
        grid = uniform_field().request.grid
        conductivity = np.asarray(uniform_field().conductivity)
        both = solve_heads(grid, conductivity, [4, 1], [-0.001, -0.002])
        self.assertEqual(both.shape, (2, 9))
        for i, (node, rate) in enumerate(((4, -0.001), (1, -0.002))):
            single = solve_heads(grid, conductivity, [node], [rate])
            np.testing.assert_allclose(both[i], single[0])

    def test_conductivity_scaling_has_inverse_effect_on_head(self):
        request = field_request()
        field = generate_field(request)
        base = solve_heads(request.grid, field.conductivity, [12], [-0.001])
        scaled = solve_heads(request.grid, np.asarray(field.conductivity) * 10, [12], [-0.001])
        np.testing.assert_allclose(scaled, base / 10, rtol=1e-10, atol=1e-12)

    def test_invalid_wells_and_monitor_roles_are_rejected(self):
        valid = pumping_request().model_dump()
        candidates = []
        duplicate_node = json.loads(json.dumps(valid))
        duplicate_node["wells"][1]["node"] = 4
        candidates.append(duplicate_node)
        boundary_pump = json.loads(json.dumps(valid))
        boundary_pump["pump_well_id"] = "boundary"
        boundary_pump["monitor_well_ids"] = ["top"]
        candidates.append(boundary_pump)
        for monitor_ids in ([], ["missing"], ["pump"], ["top", "top"]):
            candidates.append({**valid, "monitor_well_ids": monitor_ids})
        for payload in candidates:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                PumpingTestRequest.model_validate(payload)


class ExperimentExportTests(unittest.TestCase):
    def test_omitted_inverse_prior_matches_generated_structure(self):
        request = field_request()
        field = generate_field(request)
        experiment = ExperimentExport(name="Custom structure", field=field)
        self.assertEqual(experiment.inverse_prior, request.structure)
        self.assertIsNot(experiment.inverse_prior, request.structure)
        payload = experiment.model_dump()
        payload.pop("inverse_prior")
        self.assertEqual(ExperimentExport.model_validate(payload).inverse_prior, request.structure)

    def test_import_rejects_measurements_detached_from_their_saved_field(self):
        payload = self.experiment().model_dump()
        payload["tests"][0]["measurements"][0]["clean_head"] += 1
        with self.assertRaisesRegex(ValidationError, "clean measurement"):
            ExperimentExport.model_validate(payload)

    def experiment(self):
        request = pumping_request()
        result = run_pumping_test(request)
        return ExperimentExport(
            name="Saved aquifer",
            field=request.field,
            wells=request.wells,
            tests=[result],
            selected_test_ids=[result.test_id],
            inverse_settings=InverseSettings(components=2, ensemble_size=8),
        )

    def test_export_round_trip_and_truth_free_inverse_request(self):
        experiment = self.experiment()
        self.assertEqual(ExperimentExport.model_validate_json(experiment.model_dump_json()), experiment)
        request = experiment.inverse_request()
        payload = request.model_dump()
        self.assertNotIn("field", payload)
        self.assertNotIn("conductivity", payload)
        self.assertNotIn("clean_head", payload["tests"][0]["observations"][0])
        self.assertEqual(request.tests[0].pump_node, 4)
        self.assertEqual([o.well_id for o in request.tests[0].observations], ["bottom", "top", "boundary"])

    def test_installed_well_edits_do_not_rewrite_past_test_snapshots(self):
        payload = self.experiment().model_dump()
        payload["wells"] = [{"id": "new", "node": 1}]
        reloaded = ExperimentExport.model_validate(payload)
        self.assertEqual(reloaded.wells[0].id, "new")
        self.assertEqual(reloaded.tests[0].pump_well_id, "pump")
        self.assertEqual(reloaded.inverse_request().tests[0].pump_node, 4)

    def test_stale_field_and_duplicate_or_unknown_selection_are_rejected(self):
        valid = self.experiment().model_dump()
        stale_field = json.loads(json.dumps(valid))
        stale_field["tests"][0]["field_id"] = "other-field"
        duplicate_test = json.loads(json.dumps(valid))
        duplicate_test["tests"] *= 2
        for payload in (
            stale_field,
            duplicate_test,
            {**valid, "selected_test_ids": ["missing"]},
            {**valid, "selected_test_ids": ["test-a", "test-a"]},
            {**valid, "schema_version": "2.0"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                ExperimentExport.model_validate(payload)

    def test_inverse_result_cannot_survive_selected_data_or_prior_changes(self):
        experiment = self.experiment()
        request = experiment.inverse_request()
        predictions = [
            {
                "test_id": test.test_id,
                "well_id": observation.well_id,
                "observed_head": observation.head,
                "predicted_head": observation.head,
                "standard_deviation": observation.standard_deviation,
                "residual": 0,
            }
            for test in request.tests for observation in test.observations
        ]
        result = InverseResult(
            request=request,
            estimated_conductivity=[1e-4] * 4,
            coefficients=[0, 0],
            mean_log_conductivity=float(np.log(1e-4)),
            initial_objective=1,
            final_objective=0,
            converged=True,
            termination_reason="test fixture",
            iterations=[
                dict(iteration=0, objective=1, data_misfit=1, prior_penalty=0,
                     step_norm=0, damping=0.001, accepted=True),
                dict(iteration=1, objective=0, data_misfit=0, prior_penalty=0,
                     step_norm=1, damping=0.001, accepted=True),
            ],
            predictions=predictions,
            solver_version="test",
        )
        payload = experiment.model_dump()
        payload["inverse_result"] = result.model_dump()
        valid = ExperimentExport.model_validate(payload)
        self.assertEqual(valid.inverse_result, result)
        changed_prior = json.loads(json.dumps(payload))
        changed_prior["inverse_prior"]["log_variance"] = 2
        changed_measurement = json.loads(json.dumps(payload))
        changed_measurement["tests"][0]["measurements"][0]["head"] += 1
        for changed in (changed_prior, changed_measurement, {**payload, "selected_test_ids": []}):
            with self.subTest(changed=changed), self.assertRaises(ValidationError):
                ExperimentExport.model_validate(changed)


if __name__ == "__main__":
    unittest.main()
