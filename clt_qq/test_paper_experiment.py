"""Checks for the extra length laws and reproducible paper experiment."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.special import stdtr

from .experiment import leapfrog, potential, run_ensemble, transition
from .paper_experiment import (CASE, BASELINE_KEYS, ROOT, load_saved, main, method_settings,
                               paper_methods, parse_args, save_results, selected_case,
                               selected_methods, signature)


class PaperExperimentTests(unittest.TestCase):
    def test_defaults_retain_80000_and_select_each_figures_methods(self):
        first = parse_args(["--case", "t3_abs"])
        self.assertEqual((first.chains, first.iterations, first.burn_in), (2000, 120000, 40000))
        self.assertEqual(first.iterations-first.burn_in, 80000)
        self.assertEqual(first.output, ROOT / "paper/data_80k/t3_abs")
        self.assertEqual([method.key for method in selected_methods(first)],
                         ["ula", "uhmc_10", "uhmc_uniform_19"])
        case = selected_case(first)
        self.assertEqual((case.df, case.observable), (3.0, "abs"))
        self.assertAlmostEqual(case.truth(2), 2*np.sqrt(3)/np.pi)
        tail = parse_args([])
        self.assertEqual(selected_case(tail).key, "t1_tail")
        self.assertEqual(tail.output, ROOT / "paper/data_80k/t1_tail")
        self.assertEqual(len(selected_methods(tail)), 9)
        self.assertTrue(all(method.adjusted for method in selected_methods(tail)))

    def test_requested_subset_and_nuts_partition_do_not_invalidate_signature(self):
        nuts = parse_args(["--methods", "nuts", "--nuts-batch-size", "4"])
        full = parse_args(["--nuts-batch-size", "16"])
        self.assertEqual([method.key for method in selected_methods(nuts)], ["nuts"])
        self.assertEqual(signature(nuts), signature(full))
        first = parse_args(["--case", "t3_abs"])
        self.assertNotEqual(signature(nuts), signature(first))

    def test_80000_protocol_does_not_reuse_historical_baseline(self):
        args = parse_args([])
        self.assertEqual(load_saved(ROOT / "examples/simple", args, baseline=True), ({}, {}, {}))

    def test_random_laws_match_mean_length_but_not_second_moment(self):
        methods = {method.key: method for method in paper_methods()}
        for mean, upper in ((5, 9), (20, 39)):
            second_moments = []
            for key in (f"hmc_{mean}", f"hmc_uniform_{upper}", f"hmc_two_point_{upper}"):
                setting = method_settings(methods[key])
                law = setting["leapfrog_law"]
                self.assertAlmostEqual(sum(law["probabilities"]), 1)
                self.assertAlmostEqual(np.dot(law["values"], law["probabilities"]), mean)
                self.assertEqual(setting["expected_leapfrog_steps"], mean)
                second_moments.append(setting["expected_squared_leapfrog_steps"])
            self.assertLess(second_moments[0], second_moments[1])
            self.assertLess(second_moments[1], second_moments[2])

    def test_two_point_transition_uses_exact_mixture_and_metropolis(self):
        method = next(m for m in paper_methods() if m.key == "hmc_two_point_39")
        seed, size, epsilon = 173, 400, 1.8
        x = np.linspace(-8, 8, size)
        reference_rng = np.random.default_rng(seed)
        counts = np.where(reference_rng.integers(0, 2, size=size), 39, 1)
        self.assertEqual(set(counts), {1, 39})
        momenta = reference_rng.normal(size=size)
        proposals, new_momenta = leapfrog(x, momenta, CASE.df, epsilon, counts)
        ratio = potential(x, CASE.df) + momenta**2/2 - potential(proposals, CASE.df) - new_momenta**2/2
        accept = -reference_rng.exponential(size=size) < np.minimum(0, ratio)
        actual, cost, accepted = transition(x, CASE, method, epsilon, np.random.default_rng(seed))
        np.testing.assert_array_equal(actual, np.where(accept, proposals, x))
        self.assertEqual(cost, np.sum(counts+1))
        self.assertEqual(accepted, np.sum(accept))
        self.assertGreater(accepted, 0)
        self.assertLess(accepted, size)

    def test_two_point_hmc_preserves_cauchy_target(self):
        # Exact independent target starts test stationarity, without assuming
        # the Markov-chain CLT which these figures investigate.
        size = 40000
        rng = np.random.default_rng(9753)
        x = rng.standard_t(CASE.df, size=size)
        method = next(m for m in paper_methods() if m.key == "hmc_two_point_39")
        for _ in range(4):
            x, _, accepted = transition(x, CASE, method, 1.3, rng)
        self.assertGreater(accepted, 0)
        self.assertLess(accepted, size)
        uniform = np.sort(stdtr(CASE.df, x))
        distance = max(np.max(np.arange(1, size+1)/size-uniform),
                       np.max(uniform-np.arange(size)/size))
        self.assertLess(distance, 0.015)

    def test_new_methods_are_identical_across_worker_schedules(self):
        args = parse_args(["--chains", "19", "--iterations", "61", "--burn-in", "11",
                           "--workers", "1", "--batch-size", "8"])
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            for method in paper_methods():
                if method.key in BASELINE_KEYS:
                    continue
                with self.subTest(method=method.key):
                    serial, serial_info = run_ensemble(CASE, method, args)
                    parallel, parallel_info = run_ensemble(CASE, method, args, pool)
                    np.testing.assert_array_equal(serial, parallel)
                    for key in serial_info.keys()-{"worker_pids", "worker_processes_used"}:
                        self.assertEqual(serial_info[key], parallel_info[key])

    def test_baseline_reuse_is_exact_and_rejects_changed_protocol(self):
        args = parse_args(["--iterations", "30000", "--burn-in", "10000"])
        means, diagnostics, _ = load_saved(ROOT / "examples/simple", args, baseline=True)
        self.assertEqual(set(means), BASELINE_KEYS)
        with np.load(ROOT / "examples/simple/means_t1_tail.npz") as original:
            for key, values in means.items():
                np.testing.assert_array_equal(values, original[key])
                self.assertEqual(diagnostics[f"{CASE.key}/{key}"]["provenance"]["kind"], "reused_baseline")
        args.seed += 1
        self.assertEqual(load_saved(ROOT / "examples/simple", args, baseline=True), ({}, {}, {}))

    def test_partial_results_resume_with_original_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--output", directory, "--chains", "8", "--iterations", "12"])
            means = {"hmc_two_point_39": np.arange(8)/20}
            diagnostics = {f"{CASE.key}/hmc_two_point_39": {"worker_processes_used": 2,
                                                            "provenance": {"kind": "paper_experiment"}}}
            config = {"experiment_signature": signature(args)}
            save_results(args, means, diagnostics, config)
            loaded, loaded_info, _ = load_saved(Path(directory), args)
            np.testing.assert_array_equal(loaded["hmc_two_point_39"], means["hmc_two_point_39"])
            self.assertEqual(loaded_info, diagnostics)
            args.step_size = 0.36
            self.assertEqual(load_saved(Path(directory), args), ({}, {}, {}))

    def test_method_subset_resume_preserves_actual_previous_chains(self):
        with tempfile.TemporaryDirectory() as directory:
            common = ["--case", "t3_abs", "--output", directory, "--chains", "8",
                      "--iterations", "15", "--workers", "1"]
            main([*common, "--methods", "ula"])
            with np.load(Path(directory) / "means_t3_abs.npz") as saved:
                original = saved["ula"].copy()
                self.assertAlmostEqual(float(saved["target_mean"]), 2*np.sqrt(3)/np.pi)
            main([*common, "--methods", "uhmc_10"])
            args = parse_args(common)
            means, diagnostics, config = load_saved(Path(directory), args)
            self.assertEqual(set(means), {"ula", "uhmc_10"})
            np.testing.assert_array_equal(means["ula"], original)
            self.assertEqual(set(diagnostics), {"t3_abs/ula", "t3_abs/uhmc_10"})
            self.assertEqual(config["observable"], "abs")
            self.assertNotIn("nuts", config)

    def test_saved_case_and_embedded_metadata_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--case", "t3_abs", "--output", directory,
                               "--chains", "8", "--iterations", "12"])
            diagnostics = {"t3_abs/ula": {"worker_processes_used": 1}}
            config = {"experiment_signature": signature(args)}
            save_results(args, {"ula": np.ones(8)}, diagnostics, config)
            other = parse_args(["--case", "t1_tail", "--output", directory,
                               "--chains", "8", "--iterations", "12"])
            # Even a wrongly named copy cannot make another case's data reusable.
            path = Path(directory) / "means_t3_abs.npz"
            (Path(directory) / "means_t1_tail.npz").write_bytes(path.read_bytes())
            self.assertEqual(load_saved(Path(directory), other), ({}, {}, {}))
            with np.load(path) as saved:
                entries = {key: saved[key].copy() for key in saved.files}
            entries["target_mean"] = np.array(0.5)
            np.savez_compressed(path, **entries)
            with self.assertRaisesRegex(ValueError, "Stored metadata"):
                load_saved(Path(directory), args)


if __name__ == "__main__":
    unittest.main()
