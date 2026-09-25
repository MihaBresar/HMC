"""Checks that QQ plots use raw, independent chain averages."""

import unittest

import numpy as np
from scipy.stats import norm

from .plots import normal_qq_coordinates


class PlotTests(unittest.TestCase):
    def test_original_scale_without_iteration_normalization(self):
        values = np.array([0.35, 0.1, 0.6, 0.25])
        theoretical, empirical = normal_qq_coordinates(values)
        np.testing.assert_array_equal(empirical, [0.1, 0.25, 0.35, 0.6])
        expected = values.mean() + values.std(ddof=1) * norm.ppf([0.125, 0.375, 0.625, 0.875])
        np.testing.assert_allclose(theoretical, expected)

    def test_positive_affine_transformation_preserves_qq_shape(self):
        values = np.array([-0.5, 0.1, 0.8, 1.6, 4.7])
        original = normal_qq_coordinates(values)
        transformed = normal_qq_coordinates(3.0 * values + 7.0)
        for before, after in zip(original, transformed):
            np.testing.assert_allclose(after, 3.0 * before + 7.0)

    def test_degenerate_sample(self):
        theoretical, empirical = normal_qq_coordinates(np.full(8, 0.5))
        np.testing.assert_array_equal(theoretical, np.full(8, 0.5))
        np.testing.assert_array_equal(empirical, np.full(8, 0.5))

    def test_invalid_samples(self):
        for values in ([], [1], [[1, 2]], [1, np.nan], [1, np.inf], 1.0):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normal_qq_coordinates(values)


if __name__ == "__main__":
    unittest.main()
