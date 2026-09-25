"""Independent Student-t chains using BlackJAX's standard NUTS kernel.

The library's trajectory limit, divergence threshold, proposal selection and
turning criterion are left at their defaults. Step size and unit mass are fixed
to match the other samplers in this experiment; discarded iterations are burn-in,
not parameter adaptation. JAX imports are lazy so the parent process can launch
workers without first initializing a multithreaded JAX runtime.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import inspect
import math

import numpy as np


@lru_cache(maxsize=1)
def _backend():
    import blackjax
    import jax
    import jax.numpy as jnp

    # Tail excursions and long sums benefit from double precision. Configure it
    # before creating any arrays or compiling the sampler in this worker.
    jax.config.update("jax_enable_x64", True)
    return blackjax, jax, jnp


def backend_metadata() -> dict:
    """Report the installed implementation and its actual default trajectory cap."""
    blackjax, jax, _ = _backend()
    default = inspect.signature(blackjax.mcmc.nuts.as_top_level_api).parameters[
        "max_num_doublings"
    ].default
    return {
        "backend": "blackjax.nuts",
        "blackjax_version": blackjax.__version__,
        "jax_version": jax.__version__,
        "default_max_num_doublings": int(default),
    }


@lru_cache(maxsize=32)
def _compiled_runner(df, observable, threshold, iterations, burn_in, step_size):
    """Cache a compiled, vectorized batch kernel for each experiment setting."""
    blackjax, jax, jnp = _backend()
    default_doublings = backend_metadata()["default_max_num_doublings"]

    def logdensity(position):
        return -0.5 * (df + 1.0) * jnp.sum(jnp.log1p(position * position / df))

    # Deliberately omit max_num_doublings: this is the standard library kernel,
    # with its built-in default rather than an experiment-specific depth setting.
    algorithm = blackjax.nuts(logdensity, step_size, jnp.ones(1))

    def single_chain(base_key, chain_id):
        key = jax.random.fold_in(base_key, chain_id)
        state = algorithm.init(jnp.zeros(1))
        zero_float = jnp.asarray(0.0, dtype=jnp.float64)
        zero_int = jnp.asarray(0, dtype=jnp.int64)

        def body(iteration, carry):
            key, state, total, force_evals, accepted, depth_caps, divergences = carry
            key, step_key = jax.random.split(key)
            state, info = algorithm.step(step_key, state)
            value = (jnp.abs(state.position[0]) if observable == "abs"
                     else (state.position[0] >= threshold).astype(jnp.float64))
            retain = iteration >= burn_in
            hit_cap = ((info.num_trajectory_expansions >= default_doublings)
                       & ~info.is_turning & ~info.is_divergent)
            return (
                key,
                state,
                total + jnp.where(retain, value, 0.0),
                force_evals + jnp.where(retain, info.num_integration_steps, 0),
                accepted + jnp.where(retain, info.acceptance_rate, 0.0),
                depth_caps + (retain & hit_cap).astype(jnp.int64),
                divergences + (retain & info.is_divergent).astype(jnp.int64),
            )

        result = jax.lax.fori_loop(
            0, iterations, body,
            (key, state, zero_float, zero_int, zero_float, zero_int, zero_int),
        )
        return (result[2] / (iterations - burn_in), *result[3:])

    # Only the current states and running sums are kept. Neither trajectories
    # nor an iterations-by-chains array of random keys is materialized.
    return jax.jit(jax.vmap(single_chain, in_axes=(None, 0)))


def run_nuts_batch(
    df: float,
    observable: str,
    threshold: float,
    seed: int,
    chain_start: int,
    count: int,
    iterations: int,
    burn_in: int,
    step_size: float,
) -> tuple[np.ndarray, dict]:
    """Run independent zero-started chains and return one raw mean per chain.

    Random streams are indexed by target, observable and global chain ID, so
    changing the worker count or batch partition does not change a chain's draws.
    All diagnostics concern retained transitions only. ``accepted`` is the sum
    of BlackJAX's mean integration acceptance probabilities, not a count of
    accepted proposals. ``force_evals`` counts integration steps: BlackJAX's
    default velocity Verlet integrator caches the current gradient and evaluates
    one new gradient per step. The initial gradient and burn-in are excluded.
    """
    for name, value in (("df", df), ("step_size", step_size)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if observable not in ("abs", "tail"):
        raise ValueError("observable must be 'abs' or 'tail'")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    for name, value in (("seed", seed), ("chain_start", chain_start),
                        ("count", count), ("iterations", iterations),
                        ("burn_in", burn_in)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if count == 0 or iterations <= burn_in:
        raise ValueError("count and the number of retained iterations must be positive")
    if chain_start + count > 2**32:
        raise ValueError("chain IDs must fit in an unsigned 32-bit integer")

    _, jax, jnp = _backend()
    identity = f"blackjax.nuts/{float(df).hex()}/{observable}".encode()
    words = np.frombuffer(hashlib.sha256(identity).digest()[:16], dtype="<u4")
    seed_words = np.random.SeedSequence([int(seed), *map(int, words)]).generate_state(2)
    base_key = jax.random.fold_in(jax.random.key(seed_words[0]), seed_words[1])
    chain_ids = jnp.arange(chain_start, chain_start + count, dtype=jnp.uint32)
    runner = _compiled_runner(float(df), observable, float(threshold),
                              int(iterations), int(burn_in), float(step_size))
    means, force_evals, accepted, depth_caps, divergences = jax.device_get(
        runner(base_key, chain_ids)
    )
    stats = {
        "force_evals": int(np.sum(force_evals)),
        "accepted": float(np.sum(accepted)),
        "depth_caps": int(np.sum(depth_caps)),
        "divergences": int(np.sum(divergences)),
    }
    return np.asarray(means, dtype=float), stats
