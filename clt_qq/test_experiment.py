"""Checks of the numerical experiment, independent of the plotting layout."""

import unittest

import numpy as np
from scipy.special import stdtr

from .experiment import CASES, Method, leapfrog, parse_args, rng_for, run_nuts, run_vectorized, transition


class ExperimentTests(unittest.TestCase):
    def test_exact_expectations(self):
        self.assertAlmostEqual(CASES[0].truth(2), 2*np.sqrt(3)/np.pi)
        self.assertAlmostEqual(CASES[1].truth(2), 1-2*np.arctan(2)/np.pi)

    def test_variable_length_leapfrog_reversibility(self):
        rng = np.random.default_rng(815)
        x, p = rng.normal(size=(2, 80))
        counts = rng.integers(1, 21, size=80)
        q, momentum = leapfrog(x, p, 1.5, 0.45, counts)
        x_back, p_back = leapfrog(q, momentum, 1.5, -0.45, counts)
        np.testing.assert_allclose(x_back, x, atol=1e-12)
        np.testing.assert_allclose(p_back, p, atol=1e-12)

    def test_ula_is_one_unadjusted_leapfrog_position(self):
        x = np.linspace(-5, 5, 30)
        ula = Method("ula", "ULA", "ula")
        uhmc = Method("uhmc_1", "UHMC", "fixed", 1)
        a = transition(x, CASES[0], ula, 0.3, np.random.default_rng(72))[0]
        b = transition(x, CASES[0], uhmc, 0.3, np.random.default_rng(72))[0]
        np.testing.assert_allclose(a, b, atol=1e-14)

    def test_ula_rejects_nonfinite_positions(self):
        method = Method("ula", "ULA", "ula")
        with self.assertRaises(FloatingPointError):
            transition(np.array([1.0]), CASES[0], method, 1e308, np.random.default_rng(1))

    def test_adjusted_hmc_preserves_student_target(self):
        # Independent exact starts allow a distributional check without using
        # a CLT for a single correlated chain (the very property under study).
        size = 30000
        for case in CASES:
            for kind in ("fixed", "random"):
                with self.subTest(case=case.key, kind=kind):
                    rng = rng_for(473, case.key, kind)
                    x = rng.standard_t(case.df, size=size)
                    method = Method(kind, kind, kind, 10, True)
                    for _ in range(5):
                        x, _, accepted = transition(x, case, method, 1.3, rng)
                    self.assertGreater(accepted, 0)
                    self.assertLess(accepted, size)  # Metropolis correction exercised.
                    uniform = np.sort(stdtr(case.df, x))
                    distance = max(np.max(np.arange(1, size+1)/size-uniform),
                                   np.max(uniform-np.arange(size)/size))
                    self.assertLess(distance, 0.02)

    def test_iid_tail_averages_have_bernoulli_variance(self):
        args = parse_args(["--replicates", "5000", "--lengths", "50", "--seed", "73"])
        values, _ = run_vectorized(CASES[1], Method("iid", "iid", "iid"), args)
        probability = CASES[1].truth(args.tail_threshold)
        expected_var = probability*(1-probability)/50
        self.assertLess(abs(np.mean(values)-probability), 5*np.sqrt(expected_var/5000))
        self.assertLess(abs(np.var(values, ddof=1)/expected_var-1), 0.08)

    def test_checkpoint_prefixes_are_reproducible(self):
        short = parse_args(["--replicates", "16", "--lengths", "10", "--unadjusted-burn-in", "5"])
        long = parse_args(["--replicates", "16", "--lengths", "10", "20", "--unadjusted-burn-in", "5"])
        for method in (Method("ula", "ULA", "ula"), Method("hmc_random", "HMC", "random", 5, True)):
            a, _ = run_vectorized(CASES[0], method, short)
            b, _ = run_vectorized(CASES[0], method, long)
            np.testing.assert_array_equal(a[0], b[0])

    def test_nuts_workers_do_not_change_results(self):
        args = parse_args(["--replicates", "8", "--lengths", "8", "20"])
        serial, _ = run_nuts(CASES[0], args)
        args.workers = 2
        parallel, _ = run_nuts(CASES[0], args)
        np.testing.assert_array_equal(serial, parallel)


if __name__ == "__main__":
    unittest.main()
