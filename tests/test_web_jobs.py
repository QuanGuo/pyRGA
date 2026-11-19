"""HTTP contract and spawned-worker lifecycle regression tests."""

import importlib.util
import time
import unittest
from unittest.mock import patch

from pyrga.jobs import JobManager


def _waiting_worker(kind, payload, connection, cancelled):
    while not cancelled.is_set():
        time.sleep(0.01)
    connection.close()


def _failing_worker(kind, payload, connection, cancelled):
    connection.send(("error", {"code": "computation_failed", "message": "Test numerical failure"}))
    connection.close()


def _request():
    return {"grid": {"nx": 4, "ny": 4, "length_x": 40, "length_y": 40}, "seed": 42}


def _wait_for(manager, job_id, states, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status(job_id)
        if status.status in states:
            return status
        time.sleep(0.02)
    raise AssertionError(f"Job did not reach {states}: {manager.status(job_id)}")


class JobManagerTests(unittest.TestCase):
    def test_real_spawned_generation_and_result(self):
        manager = JobManager()
        try:
            job = manager.submit("generate_field", _request())
            final = _wait_for(manager, job.job_id, {"succeeded", "failed"})
            self.assertEqual(final.status, "succeeded", final.error)
            self.assertEqual(final.progress, 1)
            self.assertEqual(len(manager.result(job.job_id)["conductivity"]), 16)
            self.assertEqual(manager.cancel(job.job_id).status, "succeeded")
        finally:
            manager.close()

    def test_running_and_queued_cancellation(self):
        with patch("pyrga.jobs._compute", _waiting_worker):
            manager = JobManager()
            try:
                first = manager.submit("generate_field", _request())
                _wait_for(manager, first.job_id, {"running"})
                second = manager.submit("generate_field", _request())
                self.assertEqual(manager.cancel(second.job_id).status, "cancelled")
                self.assertEqual(manager.cancel(first.job_id).status, "cancelled")
                self.assertEqual(manager.cancel(first.job_id).status, "cancelled")
                with self.assertRaises(ValueError):
                    manager.result(first.job_id)
                # A worker exit must never overwrite a cancelled terminal state.
                time.sleep(0.1)
                self.assertEqual(manager.status(first.job_id).status, "cancelled")
            finally:
                manager.close()

    def test_failure_releases_the_worker(self):
        with patch("pyrga.jobs._compute", _failing_worker):
            manager = JobManager()
            try:
                first = manager.submit("generate_field", _request())
                second = manager.submit("generate_field", _request())
                for job in (first, second):
                    status = _wait_for(manager, job.job_id, {"failed"})
                    self.assertEqual(status.error.code, "computation_failed")
                    with self.assertRaises(ValueError):
                        manager.result(job.job_id)
            finally:
                manager.close()

    def test_shutdown_cancels_work_and_unknown_ids_fail(self):
        with patch("pyrga.jobs._compute", _waiting_worker):
            manager = JobManager()
            job = manager.submit("generate_field", _request())
            manager.close()
            self.assertEqual(manager.status(job.job_id).status, "cancelled")
            with self.assertRaises(KeyError):
                manager.status("missing")
            with self.assertRaises(RuntimeError):
                manager.submit("generate_field", _request())

    def test_finished_job_retention_is_bounded(self):
        with patch("pyrga.jobs._compute", _failing_worker):
            manager = JobManager(retained_limit=1)
            try:
                first = manager.submit("generate_field", _request())
                _wait_for(manager, first.job_id, {"failed"})
                second = manager.submit("generate_field", _request())
                _wait_for(manager, second.job_id, {"failed"})
                with self.assertRaises(KeyError):
                    manager.status(first.job_id)
                self.assertEqual(manager.status(second.job_id).status, "failed")
            finally:
                manager.close()


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "requires the web extra")
class HttpTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from pyrga.web_server import create_app

        self.manager = JobManager()
        self.client = TestClient(create_app(manager=self.manager))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_health_schema_openapi_and_cors(self):
        response = self.client.get("/api/v1/health", headers={"Origin": "http://localhost:3000"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")
        schema = self.client.get("/api/v1/schema").json()
        self.assertEqual(schema["schema_version"], "1.0")
        self.assertIn("ExperimentExport", schema["schemas"])
        openapi_response = self.client.get("/openapi.json")
        self.assertEqual(openapi_response.status_code, 200)
        openapi = openapi_response.json()
        submission = openapi["paths"]["/api/v1/jobs"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        self.assertEqual(submission["discriminator"]["propertyName"], "kind")
        result = openapi["paths"]["/api/v1/jobs/{job_id}/result"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        self.assertEqual(len(result["anyOf"]), 3)
        denied = self.client.get("/api/v1/health", headers={"Origin": "https://untrusted.example"})
        self.assertNotIn("access-control-allow-origin", denied.headers)

    def test_validation_happens_before_queueing(self):
        for submission in (
            {"kind": "missing", "payload": {}},
            {"kind": "generate_field", "payload": {"seed": -1}},
            {"kind": "generate_field", "payload": {"grid": {"nx": 100, "ny": 100, "length_x": 40, "length_y": 40}}},
        ):
            with self.subTest(submission=submission):
                response = self.client.post("/api/v1/jobs", json=submission)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["error"]["code"], "validation_error")
        self.assertFalse(self.manager._jobs)

    def test_submit_poll_result_and_missing_job(self):
        response = self.client.post("/api/v1/jobs", json={"kind": "generate_field", "payload": _request()})
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        pending = self.client.get(f"/api/v1/jobs/{job_id}/result")
        self.assertEqual(pending.status_code, 409)
        final = _wait_for(self.manager, job_id, {"succeeded", "failed"})
        self.assertEqual(final.status, "succeeded", final.error)
        self.assertEqual(self.client.get(f"/api/v1/jobs/{job_id}").json()["status"], "succeeded")
        result = self.client.get(f"/api/v1/jobs/{job_id}/result")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(result.json()["conductivity"]), 16)
        for method, path in (("get", "/api/v1/jobs/missing"), ("get", "/api/v1/jobs/missing/result"), ("delete", "/api/v1/jobs/missing")):
            self.assertEqual(getattr(self.client, method)(path).status_code, 404)

    def test_experiment_import_and_inverse_request(self):
        from pyrga.experiment_models import GenerateFieldRequest
        from pyrga.experiments import generate_field

        field = generate_field(GenerateFieldRequest.model_validate(_request()))
        experiment = {"name": "HTTP roundtrip", "field": field.model_dump(mode="json")}
        response = self.client.post("/api/v1/experiments/validate", json=experiment)
        self.assertEqual(response.status_code, 200)
        canonical = response.json()
        self.assertEqual(canonical["schema_version"], "1.0")
        self.assertEqual(self.client.post("/api/v1/experiments/validate", json=canonical).json(), canonical)
        self.assertEqual(self.client.post("/api/v1/experiments/inverse-request", json=canonical).status_code, 422)
        canonical["schema_version"] = "future"
        self.assertEqual(self.client.post("/api/v1/experiments/validate", json=canonical).status_code, 422)

    def test_queue_full_returns_429(self):
        self.manager.queue_limit = 0
        response = self.client.post("/api/v1/jobs", json={"kind": "generate_field", "payload": _request()})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"]["code"], "queue_full")

    def test_complete_http_experiment_with_selected_pumping_runs(self):
        def compute(kind, payload):
            response = self.client.post("/api/v1/jobs", json={"kind": kind, "payload": payload})
            self.assertEqual(response.status_code, 202, response.text)
            job_id = response.json()["job_id"]
            status = _wait_for(self.manager, job_id, {"succeeded", "failed"})
            self.assertEqual(status.status, "succeeded", status.error)
            response = self.client.get(f"/api/v1/jobs/{job_id}/result")
            self.assertEqual(response.status_code, 200, response.text)
            return response.json()

        field_request = _request()
        field_request["structure"] = {"correlation_length_x": 8, "correlation_length_y": 10}
        field = compute("generate_field", field_request)
        wells = [{"id": "A", "node": 6}, {"id": "B", "node": 18},
                 {"id": "C", "node": 12}, {"id": "D", "node": 7}]
        runs = []
        for test_id, pump, monitors, seed in (
            ("first", "A", ["B", "C"], 41),
            ("second", "B", ["A", "D"], 42),
        ):
            runs.append(compute("run_pumping_test", {
                "field": field, "wells": wells, "test_id": test_id,
                "pump_well_id": pump, "monitor_well_ids": monitors,
                "pumping_rate": -0.001,
                "noise": {"relative": 0, "absolute_floor": 0, "seed": seed},
            }))
        experiment = {
            "name": "HTTP end-to-end", "field": field, "wells": wells,
            "tests": runs, "selected_test_ids": ["second", "first"],
            "inverse_prior": {"mean_log_conductivity": -8.8,
                              "correlation_length_x": 8, "correlation_length_y": 10},
            "inverse_settings": {"components": 3, "ensemble_size": 12, "max_iterations": 4},
        }
        response = self.client.post("/api/v1/experiments/inverse-request", json=experiment)
        self.assertEqual(response.status_code, 200, response.text)
        inverse = response.json()
        self.assertEqual([item["test_id"] for item in inverse["tests"]], ["second", "first"])
        self.assertEqual([item["well_id"] for item in inverse["tests"][0]["observations"]], ["A", "D"])
        self.assertNotIn("field", inverse)
        self.assertNotIn("conductivity", inverse)
        estimate = compute("run_inverse_model", inverse)
        self.assertLess(estimate["final_objective"], estimate["initial_objective"])
        self.assertEqual(len(estimate["estimated_conductivity"]), 16)
        experiment["inverse_result"] = estimate
        exported = self.client.post("/api/v1/experiments/validate", json=experiment)
        self.assertEqual(exported.status_code, 200, exported.text)
        reloaded = self.client.post("/api/v1/experiments/validate", json=exported.json())
        self.assertEqual(reloaded.json(), exported.json())


if __name__ == "__main__":
    unittest.main()
