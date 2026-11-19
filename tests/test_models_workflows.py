import unittest

import numpy as np
from pydantic import ValidationError

from pyrga.models import DarcyDataset, GridSpec, Well
from pyrga.mcp_server import validate_dataset_payload
from pyrga.workflows import run_forward


class ModelAndWorkflowTests(unittest.TestCase):
    def make_dataset(self):
        return DarcyDataset(
            name="tiny aquifer",
            grid=GridSpec(nx=2, ny=2, length_x=20, length_y=20),
            conductivity=[1e-4] * 4,
            wells=[Well(id="PW-1", node=4, pumping_rate=-0.001)],
        )

    def test_conductivity_shape_is_validated(self):
        with self.assertRaises(ValidationError):
            DarcyDataset(
                name="invalid",
                grid=GridSpec(nx=2, ny=2, length_x=20, length_y=20),
                conductivity=[1e-4],
            )

    def test_forward_result_is_json_serializable(self):
        result = run_forward(self.make_dataset())
        payload = result.model_dump(mode="json")
        self.assertEqual(payload["well_ids"], ["PW-1"])
        self.assertEqual(len(payload["heads"][0]), 9)
        self.assertEqual(payload["summary"]["pumping_well_count"], 1)

    def test_forward_matches_known_si_solution(self):
        result = run_forward(self.make_dataset())
        # Symmetry gives h_top = h_bottom = h_center / 4. For the centre
        # equation, (8/3 - 1/3) * K * h_center = Q, hence h_center = -4 m.
        np.testing.assert_allclose(
            result.heads[0], [0, -1, 0, 0, -4, 0, 0, -1, 0], atol=1e-7
        )

    def test_square_domain_rescaling_preserves_steady_heads(self):
        payload = self.make_dataset().model_dump()
        payload["grid"].update(length_x=40, length_y=40)
        np.testing.assert_allclose(
            run_forward(self.make_dataset()).heads,
            run_forward(DarcyDataset.model_validate(payload)).heads,
        )

    def test_rectangular_cells_are_rejected(self):
        payload = self.make_dataset().model_dump()
        payload["grid"].update(length_x=40, length_y=10)
        with self.assertRaisesRegex(ValueError, "square cells"):
            run_forward(DarcyDataset.model_validate(payload))

    def test_fixed_head_boundary_pumps_are_rejected(self):
        for node in (0, 2, 3, 5, 6, 8):
            for rate in (-0.001, 0.001):
                with self.subTest(node=node, rate=rate):
                    payload = self.make_dataset().model_dump()
                    payload["wells"][0].update(node=node, pumping_rate=rate)
                    with self.assertRaisesRegex(ValidationError, "fixed-head boundaries"):
                        DarcyDataset.model_validate(payload)

    def test_boundary_observation_wells_are_allowed(self):
        payload = self.make_dataset().model_dump()
        payload["wells"].extend(
            {"id": f"OW-{node}", "node": node}
            for node in (0, 2, 3, 5, 6, 8)
        )
        payload["observations"] = [{"well_id": "OW-3", "head": 0}]
        result = run_forward(DarcyDataset.model_validate(payload))
        self.assertEqual(result.well_ids, ["PW-1"])
        self.assertAlmostEqual(result.heads[0][4], -4, places=6)

    def test_top_and_bottom_boundary_pumps_are_allowed(self):
        for node in (1, 7):
            with self.subTest(node=node):
                payload = self.make_dataset().model_dump()
                payload["wells"][0]["node"] = node
                result = run_forward(DarcyDataset.model_validate(payload))
                self.assertLess(result.heads[0][node], 0)

    def test_unsupported_units_are_rejected(self):
        for field, unit in (("length_unit", "ft"),
                            ("conductivity_unit", "m/h"),
                            ("pumping_rate_unit", "m^3/h"),
                            ("head_unit", "ft")):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                payload = self.make_dataset().model_dump()
                if field == "length_unit":
                    payload["grid"][field] = unit
                elif field == "pumping_rate_unit":
                    payload["wells"][0][field] = unit
                elif field == "head_unit":
                    payload["observations"] = [{"well_id": "PW-1", "head": 1, field: unit}]
                else:
                    payload[field] = unit
                DarcyDataset.model_validate(payload)

    def test_non_finite_inputs_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for field in ("conductivity", "length_x", "pumping_rate", "x", "head"):
                with self.subTest(value=value, field=field):
                    payload = self.make_dataset().model_dump()
                    if field == "conductivity":
                        payload[field][0] = value
                    elif field == "length_x":
                        payload["grid"][field] = value
                    elif field == "head":
                        payload["observations"] = [{"well_id": "PW-1", "head": value}]
                    else:
                        payload["wells"][0][field] = value
                    self.assertFalse(validate_dataset_payload(payload)["valid"])


if __name__ == "__main__":
    unittest.main()
