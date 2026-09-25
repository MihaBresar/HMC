"""Checks that the experiment uses genuine BlackJAX transitions and raw means."""

import hashlib
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from clt_qq.nuts import _backend, _compiled_runner, backend_metadata, run_nuts_batch


class NutsTests(unittest.TestCase):
    def arguments(self, **overrides):
        args = dict(df=3.0, observable="abs", threshold=2.0, seed=721,
                    chain_start=0, count=4, iterations=31, burn_in=11,
                    step_size=0.35)
        args.update(overrides)
        return args

    def test_library_defaults_are_not_overridden(self):
        blackjax, _, _ = _backend()
        metadata = backend_metadata()
        self.assertEqual(metadata["backend"], "blackjax.nuts")
        self.assertEqual(metadata["blackjax_version"], blackjax.__version__)
        self.assertEqual(metadata["default_max_num_doublings"], inspect.signature(
            blackjax.mcmc.nuts.as_top_level_api).parameters["max_num_doublings"].default)
        _compiled_runner.cache_clear()
        with patch.object(blackjax, "nuts", wraps=blackjax.nuts) as constructor:
            run_nuts_batch(**self.arguments(count=1, iterations=3, burn_in=1))
        constructor.assert_called_once()
        # Only log density, step size, and unit inverse mass are supplied.
        self.assertEqual(len(constructor.call_args.args), 3)
        self.assertEqual(constructor.call_args.kwargs, {})
        np.testing.assert_array_equal(constructor.call_args.args[2], [1.0])

    def test_raw_postburn_means_and_diagnostics_match_blackjax_paths(self):
        blackjax, jax, jnp = _backend()
        for df, observable in ((3.0, "abs"), (1.0, "tail"), (1.5, "tail")):
            with self.subTest(df=df, observable=observable):
                args = self.arguments(df=df, observable=observable, count=2)
                actual, stats = run_nuts_batch(**args)
                # Independently construct the public library algorithm, keep
                # complete paths, and average them after discarding burn-in.
                def logdensity(position):
                    return -0.5 * (df + 1) * jnp.sum(jnp.log1p(position**2 / df))

                algorithm = blackjax.nuts(logdensity, args["step_size"], jnp.ones(1))
                step = jax.jit(algorithm.step)
                identity = f"blackjax.nuts/{float(df).hex()}/{observable}".encode()
                words = np.frombuffer(hashlib.sha256(identity).digest()[:16], dtype="<u4")
                seeds = np.random.SeedSequence([args["seed"], *map(int, words)]).generate_state(2)
                base_key = jax.random.fold_in(jax.random.key(seeds[0]), seeds[1])
                expected, costs, accepts, caps, divergences = [], 0, 0.0, 0, 0
                default_depth = backend_metadata()["default_max_num_doublings"]
                for chain_id in range(args["count"]):
                    key = jax.random.fold_in(base_key, chain_id)
                    state, path = algorithm.init(jnp.zeros(1)), []
                    for iteration in range(args["iterations"]):
                        key, step_key = jax.random.split(key)
                        state, info = step(step_key, state)
                        x = float(state.position[0])
                        path.append(abs(x) if observable == "abs" else float(x >= args["threshold"]))
                        if iteration >= args["burn_in"]:
                            costs += int(info.num_integration_steps)
                            accepts += float(info.acceptance_rate)
                            caps += int(info.num_trajectory_expansions >= default_depth
                                        and not info.is_turning and not info.is_divergent)
                            divergences += int(info.is_divergent)
                    expected.append(np.mean(path[args["burn_in"]:]))
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                self.assertEqual(stats["force_evals"], costs)
                self.assertAlmostEqual(stats["accepted"], accepts, places=11)
                self.assertEqual(stats["depth_caps"], caps)
                self.assertEqual(stats["divergences"], divergences)

    def test_batch_partition_preserves_chain_streams(self):
        args = self.arguments(chain_start=5, count=7)
        whole, whole_stats = run_nuts_batch(**args)
        left, left_stats = run_nuts_batch(**{**args, "count": 3})
        right, right_stats = run_nuts_batch(**{**args, "chain_start": 8, "count": 4})
        np.testing.assert_allclose(whole, np.concatenate([left, right]), rtol=1e-12, atol=1e-12)
        self.assertEqual(len(np.unique(whole)), args["count"])
        self.assertTrue(np.isfinite(whole).all())
        for key in whole_stats:
            self.assertAlmostEqual(whole_stats[key], left_stats[key] + right_stats[key], places=11)

    def test_finite_results_and_retained_diagnostics(self):
        args = self.arguments(df=1.0, observable="tail", count=4)
        means, stats = run_nuts_batch(**args)
        draws = args["count"] * (args["iterations"] - args["burn_in"])
        self.assertTrue(np.isfinite(means).all())
        self.assertTrue(((0 <= means) & (means <= 1)).all())
        self.assertGreaterEqual(stats["force_evals"], draws)
        self.assertGreaterEqual(stats["accepted"], 0)
        self.assertLessEqual(stats["accepted"], draws)
        for key in ("depth_caps", "divergences"):
            self.assertGreaterEqual(stats[key], 0)
            self.assertLessEqual(stats[key], draws)

    def test_invalid_parameters(self):
        for override in ({"df": 0}, {"df": np.inf}, {"step_size": -0.1},
                         {"threshold": np.nan}, {"observable": "unknown"},
                         {"seed": -1}, {"chain_start": -1}, {"count": 0},
                         {"count": True}, {"iterations": 3, "burn_in": 3},
                         {"chain_start": 2**32 - 1, "count": 2}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                run_nuts_batch(**self.arguments(**override))


if __name__ == "__main__":
    unittest.main()
