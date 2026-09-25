"""Checks of the numerical experiment, independent of the plotting layout."""

from concurrent.futures import ProcessPoolExecutor
import os
import unittest

import numpy as np
from scipy.special import stdtr

from .experiment import (CASES, Method, leapfrog, methods_for, parse_args,
                         rng_for, run_ensemble, run_vectorized, transition)
from .nuts import nuts_step


class ExperimentTests(unittest.TestCase):
    def test_exact_expectations(self):
        self.assertAlmostEqual(CASES[0].truth(2), 2*np.sqrt(3)/np.pi)
        self.assertAlmostEqual(CASES[1].truth(2), 0.5-np.arctan(2)/np.pi)
        self.assertAlmostEqual(CASES[2].truth(2), 1-stdtr(1.5, 2))

    def test_tail_observable_is_one_sided_and_includes_threshold(self):
        for case in CASES[1:]:
            np.testing.assert_array_equal(case.values(np.array([-3, -2, 0, 2, 3]), 2),
                                          [False, False, False, True, True])

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
        args = parse_args(["--chains", "5000", "--iterations", "75", "--burn-in", "25",
                           "--seed", "73", "--workers", "1"])
        values, _ = run_vectorized(CASES[1], Method("iid", "iid", "iid"), args)
        probability = CASES[1].truth(args.tail_threshold)
        expected_var = probability*(1-probability)/50
        self.assertLess(abs(np.mean(values)-probability), 5*np.sqrt(expected_var/5000))
        self.assertLess(abs(np.var(values, ddof=1)/expected_var-1), 0.08)

    def test_one_raw_postburn_mean_per_numbered_chain(self):
        # Reproduce complete paths and average their retained observations.
        # The runner instead accumulates sums online, across multiple batches.
        args = parse_args(["--chains", "19", "--iterations", "29", "--batch-size", "8",
                           "--workers", "1", "--fixed-steps", "2", "3",
                           "--random-max", "3", "--max-depth", "3"])
        self.assertEqual(args.burn_in, 9)
        case = CASES[0]
        for method in methods_for(args):
            with self.subTest(method=method.key):
                actual, info = run_ensemble(case, method, args)
                expected = []
                if method.kind == "nuts":
                    for chain_id in range(args.chains):
                        rng = rng_for(args.seed, case.key, method.key, chain_id)
                        x, path = 0.0, []
                        for _ in range(args.iterations):
                            x, _, _, _ = nuts_step(x, case.df, args.step_size, args.max_depth, rng)
                            path.append(abs(x))
                        expected.append(np.mean(path[args.burn_in:]))
                else:
                    for batch_id, start in enumerate(range(0, args.chains, args.batch_size)):
                        count = min(args.batch_size, args.chains-start)
                        rng = rng_for(args.seed, case.key, method.key, batch_id)
                        x, path = np.zeros(count), []
                        for _ in range(args.iterations):
                            if method.kind == "iid":
                                x = rng.standard_t(case.df, size=count)
                            else:
                                x, _, _ = transition(x, case, method, args.step_size, rng)
                            path.append(np.abs(x))
                        expected.extend(np.mean(path[args.burn_in:], axis=0))
                self.assertEqual(actual.shape, (args.chains,))
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
                self.assertEqual(len(np.unique(actual)), args.chains)
                self.assertEqual(info["chains_completed"], args.chains)
                self.assertEqual(info["batches"], 3)
                self.assertEqual(info["retained_per_chain"], 20)

    def test_workers_preserve_all_sampler_results_and_chain_order(self):
        args = parse_args(["--chains", "19", "--iterations", "40", "--burn-in", "7",
                           "--batch-size", "8", "--workers", "1", "--fixed-steps", "2", "3",
                           "--random-max", "3", "--max-depth", "3"])
        with ProcessPoolExecutor(max_workers=2) as pool:
            for method in methods_for(args):
                with self.subTest(method=method.key):
                    serial, serial_info = run_ensemble(CASES[0], method, args)
                    parallel, parallel_info = run_ensemble(CASES[0], method, args, pool=pool)
                    np.testing.assert_array_equal(serial, parallel)
                    # Scheduling changes process identities, never estimates or diagnostics.
                    for key in serial_info.keys() - {"worker_pids", "worker_processes_used"}:
                        self.assertEqual(serial_info[key], parallel_info[key])
                    self.assertEqual(serial_info["worker_pids"], [os.getpid()])
                    self.assertNotIn(os.getpid(), parallel_info["worker_pids"])

    def test_ensemble_uses_multiple_worker_processes(self):
        # Enough separate batches and work to exercise both process workers,
        # rather than merely requesting parallel execution in the configuration.
        args = parse_args(["--chains", "128", "--iterations", "3000",
                           "--batch-size", "8", "--workers", "2"])
        values, info = run_ensemble(CASES[1], Method("ula", "ULA", "ula"), args)
        self.assertEqual(values.shape, (128,))
        self.assertEqual(info["worker_processes_used"], 2)
        self.assertEqual(len(set(info["worker_pids"])), 2)
        self.assertNotIn(os.getpid(), info["worker_pids"])
        self.assertEqual(info["batches"], 16)
        self.assertEqual(info["burn_in_per_chain"], 1000)
        self.assertEqual(info["retained_per_chain"], 2000)
        self.assertEqual(info["force_evals_per_retained_draw"], 1)


if __name__ == "__main__":
    unittest.main()
