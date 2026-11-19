"""Shared numerical results and job lifecycle through the MCP adapter."""

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec("mcp"), "requires the MCP extra")
class ExperimentMcpTests(unittest.IsolatedAsyncioTestCase):
    async def test_typed_tools_share_python_workflows(self):
        from pyrga.mcp_server import create_server
        from pyrga.experiment_models import GenerateFieldRequest
        from pyrga.experiments import generate_field
        server = create_server()

        async def call(name, arguments):
            _, result = await server.call_tool(name, arguments)
            return result

        payload = {"grid": {"nx": 4, "ny": 4, "length_x": 40, "length_y": 40}, "seed": 99}
        field = await call("generate_field", {"payload": payload})
        self.assertEqual(field, generate_field(GenerateFieldRequest.model_validate(payload)).model_dump(mode="json"))
        test = await call("run_pumping_test", {"payload": {
            "field": field, "test_id": "mcp-test",
            "wells": [{"id": "pump", "node": 12}, {"id": "monitor", "node": 7}],
            "pump_well_id": "pump", "noise": {"relative": 0, "absolute_floor": 0},
        }})
        experiment = {"name": "MCP experiment", "field": field, "tests": [test],
                      "selected_test_ids": [test["test_id"]],
                      "inverse_settings": {"components": 2, "ensemble_size": 8, "max_iterations": 2}}
        request = await call("prepare_inverse_request", {"experiment": experiment})
        self.assertNotIn("field", request)
        self.assertNotIn("clean_head", request["tests"][0]["observations"][0])
        result = await call("run_inverse_model", {"payload": request})
        self.assertLessEqual(result["final_objective"], result["initial_objective"])
        self.assertEqual(len(result["estimated_conductivity"]), 16)
        tools = await server.list_tools()
        generator = next(tool for tool in tools if tool.name == "generate_field")
        self.assertIn("$defs", generator.inputSchema)

    async def test_real_stdio_job_submission_polling_result_and_terminal_cancel(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        server = StdioServerParameters(command=sys.executable, args=["-m", "pyrga.mcp_server"])
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments=arguments)
                    self.assertFalse(response.isError, response.content)
                    return response.structuredContent

                job = await call("submit_experiment_job", {
                    "kind": "generate_field",
                    "payload": {"grid": {"nx": 4, "ny": 4, "length_x": 40, "length_y": 40}},
                })
                for _ in range(200):
                    status = await call("get_experiment_job", {"job_id": job["job_id"]})
                    if status["status"] in ("succeeded", "failed", "cancelled"):
                        break
                    await asyncio.sleep(0.05)
                self.assertEqual(status["status"], "succeeded", status)
                result = await call("get_experiment_job_result", {"job_id": job["job_id"]})
                self.assertEqual(len(result["conductivity"]), 16)
                terminal = await call("cancel_experiment_job", {"job_id": job["job_id"]})
                self.assertEqual(terminal["status"], "succeeded")


class SchemaSnapshotTests(unittest.TestCase):
    def test_checked_in_contract_matches_python_models(self):
        from pyrga.api_contract import experiment_schemas
        snapshot = Path(__file__).resolve().parents[1] / "docs/experiment-api.schema.json"
        self.assertEqual(json.loads(snapshot.read_text()), experiment_schemas())
