import unittest

import numpy as np

from diffusion_model import diffusion_step, run_diffusion


class DiffusionModelTest(unittest.TestCase):
    def test_flat_profile_remains_flat(self):
        profile = np.full(5, 10.0)
        result = diffusion_step(profile, diffusivity=0.1, dx=1.0, dt=1.0)
        np.testing.assert_allclose(result, profile)

    def test_diffusion_smooths_peak(self):
        profile = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = diffusion_step(profile, diffusivity=0.25, dx=1.0, dt=1.0)
        self.assertLess(result[2], profile[2])
        self.assertGreater(result[1], profile[1])

    def test_run_diffusion_preserves_length_and_is_stable(self):
        profile = np.linspace(0.0, 10.0, 20)
        result = run_diffusion(profile, diffusivity=0.1, dx=1.0, dt=1.0, num_steps=10)
        self.assertEqual(result.shape, profile.shape)
        np.testing.assert_allclose(result[0], result[1])
        np.testing.assert_allclose(result[-1], result[-2])

    def test_unstable_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            diffusion_step(np.arange(5.0), diffusivity=1.0, dx=1.0, dt=1.0)


if __name__ == "__main__":
    unittest.main()
