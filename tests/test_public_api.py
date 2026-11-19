import unittest

import numpy as np

import pyrga


class PublicApiTests(unittest.TestCase):
    def test_public_api_imports(self):
        self.assertTrue(callable(pyrga.groundwater_solver))
        self.assertTrue(callable(pyrga.hydraulic_tomography))
        self.assertEqual(pyrga.__version__, "0.2.0.dev0")

    def test_small_steady_state_problem(self):
        conductivity = np.ones(4, dtype=float)
        heads = pyrga.groundwater_solver(conductivity, well_node=4, Q=-1.0)
        self.assertEqual(heads.shape, (9,))
        self.assertTrue(np.isfinite(heads).all())

    def test_domain_preparation_returns_scalar_rate(self):
        wells, pumping_rate, dx, dy = pyrga.prepare_physical_domain(8, 8)
        self.assertEqual(len(wells), 25)
        self.assertIsInstance(pumping_rate, float)
        self.assertGreater(dx, 0)
        self.assertGreater(dy, 0)


if __name__ == "__main__":
    unittest.main()
