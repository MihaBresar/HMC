"""Numerical correctness checks; run with python -m unittest clt_qq.test_nuts."""

import unittest

import numpy as np
from scipy.special import stdtr

from clt_qq.nuts import _leapfrog, nuts_step


class NutsTests(unittest.TestCase):
    def test_leapfrog_reversibility(self):
        rng = np.random.default_rng(721)
        for df in (1.0, 1.5, 3.0):
            for epsilon in (0.01, 0.4, 1.0):
                for _ in range(20):
                    x, p = rng.normal(size=2)
                    original = np.array([x, p])
                    for _ in range(30):
                        x, p = _leapfrog(x, p, epsilon, df)
                    for _ in range(30):
                        x, p = _leapfrog(x, p, -epsilon, df)
                    np.testing.assert_allclose([x, p], original, rtol=1e-10, atol=1e-10)

    def test_independent_stationary_transitions(self):
        # Each trial starts from a fresh exact Student-t draw. Thus the outputs
        # are independent, and no effective-sample-size assumption is needed.
        # Probability-integral transforms should be Uniform(0, 1) for ALL df,
        # including Cauchy, whose mean and variance cannot be used for testing.
        n = 6000
        for df in (1.0, 1.5, 3.0):
            for epsilon, max_depth in ((0.4, 6), (2.0, 3)):
                with self.subTest(df=df, epsilon=epsilon, max_depth=max_depth):
                    rng = np.random.default_rng(49200 + int(10 * df + epsilon))
                    initial = rng.standard_t(df, size=n)
                    draws = np.array([
                        nuts_step(float(x), df, epsilon, max_depth, rng)[0] for x in initial
                    ])
                    uniform = np.sort(stdtr(df, draws))
                    d_plus = np.max(np.arange(1, n + 1) / n - uniform)
                    d_minus = np.max(uniform - np.arange(n) / n)
                    # DKW bound per setting: P(D > .035) <= 8.3e-7.
                    self.assertLess(max(d_plus, d_minus), 0.035)

    def test_depth_cap_and_work_count(self):
        for depth in (1, 3, 6):
            result = nuts_step(100000.0, 1.0, 0.1, depth, np.random.default_rng(8))
            self.assertEqual(result[1], 2**depth - 1)
            self.assertTrue(result[2])
            self.assertFalse(result[3])

    def test_divergent_subtree_does_not_supply_a_proposal(self):
        result = nuts_step(1.0, 3.0, 100.0, 6, np.random.default_rng(1))
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 1)
        self.assertFalse(result[2])
        self.assertTrue(result[3])

    def test_u_turn_at_maximum_depth_is_not_a_cap_hit(self):
        result = nuts_step(0.0, 3.0, 2.0, 1, np.random.default_rng(1))
        self.assertEqual(result[1], 1)
        self.assertFalse(result[2])
        self.assertFalse(result[3])

    def test_nonfinite_trajectory_is_rejected(self):
        result = nuts_step(1.0, 3.0, 1e308, 6, np.random.default_rng(1))
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 1)
        self.assertFalse(result[2])
        self.assertTrue(result[3])

    def test_invalid_parameters(self):
        for df, epsilon, depth in ((0.0, 0.1, 3), (3.0, -0.1, 3), (3.0, 0.1, 0), (3.0, 0.1, 1.5)):
            with self.subTest(df=df, epsilon=epsilon, depth=depth):
                with self.assertRaises(ValueError):
                    nuts_step(0.0, df, epsilon, depth, np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
