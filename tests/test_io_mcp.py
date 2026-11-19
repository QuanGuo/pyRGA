import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from pyrga.io import dataset_from_csv, load_dataset_json, save_dataset_json
from pyrga.mcp_server import create_server, render_forward_payload, run_forward_payload, validate_dataset_payload
from pyrga.models import GridSpec


def invalid_dataset_payloads():
    valid = {
        "name": "validation test",
        "grid": {"nx": 2, "ny": 2, "length_x": 20, "length_y": 20},
        "conductivity": [1e-4] * 4,
        "wells": [{"id": "PW-1", "node": 4, "pumping_rate": -0.001}],
    }
    yield {**valid, "conductivity": [-1] * 4}
    yield {**valid, "wells": valid["wells"] * 2}
    yield {**valid, "wells": [{"id": "PW-1", "node": 3, "pumping_rate": -0.001}]}


class IoAndMcpTests(unittest.TestCase):
    def test_csv_json_and_mcp_vertical_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conductivity_path = root / "conductivity.csv"
            wells_path = root / "wells.csv"
            with conductivity_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["conductivity"])
                writer.writeheader()
                writer.writerows({"conductivity": 1e-4} for _ in range(4))
            with wells_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["id", "node", "pumping_rate"]
                )
                writer.writeheader()
                writer.writerow({"id": "PW-1", "node": 4, "pumping_rate": -0.001})

            dataset = dataset_from_csv(
                conductivity_path,
                wells_path,
                name="CSV aquifer",
                grid=GridSpec(nx=2, ny=2, length_x=20, length_y=20),
            )
            json_path = save_dataset_json(dataset, root / "dataset.json")
            loaded = load_dataset_json(json_path)
            payload = loaded.model_dump(mode="json")

            validation = validate_dataset_payload(payload)
            result = run_forward_payload(payload)
            artifacts = render_forward_payload(result, str(root / "report"))

            self.assertTrue(validation["valid"])
            self.assertEqual(result["well_ids"], ["PW-1"])
            self.assertEqual(len(result["heads"][0]), 9)
            self.assertEqual(len(loaded.provenance), 1)
            self.assertTrue(Path(artifacts["report"]).exists())
            self.assertIn("Hydraulic head", Path(artifacts["report"]).read_text())

    def test_invalid_mcp_payload_has_structured_errors(self):
        result = validate_dataset_payload({"name": "missing fields"})
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)

    def test_model_validator_errors_are_json_serializable(self):
        for payload in invalid_dataset_payloads():
            with self.subTest(payload=payload):
                result = validate_dataset_payload(payload)
                self.assertFalse(result["valid"])
                self.assertTrue(result["errors"])
                decoded = json.loads(json.dumps(result))
                self.assertFalse(decoded["valid"])
                self.assertEqual(decoded["errors"][0]["msg"], result["errors"][0]["msg"])


@unittest.skipUnless(importlib.util.find_spec("mcp"), "requires the MCP extra")
class McpTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_validator_errors_survive_tool_serialization(self):
        server = create_server()
        for payload in invalid_dataset_payloads():
            with self.subTest(payload=payload):
                content, structured = await server.call_tool(
                    "validate_dataset", {"payload": payload}
                )
                self.assertFalse(structured["valid"])
                self.assertTrue(structured["errors"])
                self.assertFalse(json.loads(content[0].text)["valid"])


if __name__ == "__main__":
    unittest.main()
