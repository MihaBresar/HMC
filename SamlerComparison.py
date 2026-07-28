"""
Compare NUTS and random-walk Metropolis on a heavy-tailed skew-t target.

The script:
  1. generates exact i.i.d. reference draws from the target;
  2. runs several independent RWM and NUTS chains;
  3. reports Monte Carlo estimates, standard errors, and RMSEs;
  4. saves Q-Q, boxplot, RMSE, and trace figures.

Required packages:
    pip install numpy scipy matplotlib jax blackjax

The settings below are deliberately modest so that the file is useful as a
standalone example. Increase NUM_CHAINS, NUM_SAMPLES, and NUM_REFERENCE for a
more accurate benchmark.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import gammaln as jax_gammaln
from jax.scipy.stats import norm as jax_norm
from scipy.special import gammaln, log_ndtr

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Experiment settings
# =============================================================================

DIMENSION = 100
DEGREES_OF_FREEDOM = 3.0

SKEWNESS = np.zeros(DIMENSION)
SKEWNESS[0] = 20.0
SKEWNESS[1] = -30.0

NUM_CHAINS = 8
NUM_WARMUP = 1_000
NUM_SAMPLES = 2_000
NUM_REFERENCE = 500_000

TAIL_THRESHOLD = 50.0
RWM_TARGET_ACCEPTANCE = 0.234
SEED = 2026

QQ_PROBABILITIES = np.array(
    [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.98, 0.99]
)
OUTPUT_DIRECTORY = Path("sampler_comparison_output")


# =============================================================================
# Heavy-tailed skew-t target
# =============================================================================

NU = DEGREES_OF_FREEDOM
D = DIMENSION
ALPHA_JAX = jnp.asarray(SKEWNESS)

LOG_T_NORMALIZER = (
    gammaln((NU + D) / 2.0)
    - gammaln(NU / 2.0)
    - (D / 2.0) * np.log(NU * np.pi)
)


def log_density_numpy(x):
    """Log-density of the d-dimensional skew-t target."""
    x = np.asarray(x, dtype=np.float64)
    squared_radius = np.dot(x, x)
    log_student_t = (
        LOG_T_NORMALIZER
        - ((NU + D) / 2.0) * np.log1p(squared_radius / NU)
    )
    skew_argument = (
        np.dot(SKEWNESS, x)
        * np.sqrt((NU + D) / (NU + squared_radius))
    )
    return float(np.log(2.0) + log_student_t + log_ndtr(skew_argument))


@jax.jit
def log_density_jax(x):
    """JAX version of the same log-density, used by NUTS."""
    squared_radius = jnp.dot(x, x)
    log_normalizer = (
        jax_gammaln((NU + D) / 2.0)
        - jax_gammaln(NU / 2.0)
        - (D / 2.0) * jnp.log(NU * jnp.pi)
    )
    log_student_t = (
        log_normalizer
        - ((NU + D) / 2.0) * jnp.log1p(squared_radius / NU)
    )
    skew_argument = (
        jnp.dot(ALPHA_JAX, x)
        * jnp.sqrt((NU + D) / (NU + squared_radius))
    )
    return jnp.log(2.0) + log_student_t + jax_norm.logcdf(skew_argument)


def exact_skew_t_reference(num_samples, rng, batch_size=25_000):
    """
    Generate exact skew-t draws in batches.

    If X and Y are independent standard normal variables, changing X to -X
    according to Y < alpha'X produces the required skew-normal draw. Dividing
    by an independent square-root chi-square scale gives the skew-t draw.
    """
    # Only four coordinates and the norm are needed downstream. Keeping this
    # reduced reference avoids storing a potentially very large n-by-d array.
    coordinates = np.empty((num_samples, 4), dtype=np.float64)
    norms = np.empty(num_samples, dtype=np.float64)

    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        size = stop - start

        x = rng.standard_normal((size, D))
        y = rng.standard_normal(size)
        signs = np.where(y < x @ SKEWNESS, 1.0, -1.0)
        skew_normal = signs[:, None] * x

        chi_square = rng.chisquare(NU, size=size)
        samples = skew_normal / np.sqrt(chi_square / NU)[:, None]
        coordinates[start:stop] = samples[:, :4]
        norms[start:stop] = np.linalg.norm(samples, axis=1)

    return {"coordinates": coordinates, "norms": norms}


# =============================================================================
# Samplers
# =============================================================================

def run_rwm(seed):
    """Gaussian random-walk Metropolis with warmup step-size adaptation."""
    rng = np.random.default_rng(seed)
    position = rng.standard_normal(D)
    log_probability = log_density_numpy(position)
    log_step_size = np.log(2.38 / np.sqrt(D))

    warmup_acceptances = 0
    for iteration in range(1, NUM_WARMUP + 1):
        proposal = position + np.exp(log_step_size) * rng.standard_normal(D)
        proposal_log_probability = log_density_numpy(proposal)
        log_acceptance_ratio = proposal_log_probability - log_probability
        accepted = np.log(rng.random()) < log_acceptance_ratio

        if accepted:
            position = proposal
            log_probability = proposal_log_probability
            warmup_acceptances += 1

        # Robbins-Monro adaptation; adaptation stops before retained sampling.
        gain = min(0.05, iteration ** -0.6)
        acceptance_probability = min(1.0, np.exp(min(0.0, log_acceptance_ratio)))
        log_step_size += gain * (
            acceptance_probability - RWM_TARGET_ACCEPTANCE
        )

    samples = np.empty((NUM_SAMPLES, D), dtype=np.float64)
    accepted = 0
    step_size = np.exp(log_step_size)

    for iteration in range(NUM_SAMPLES):
        proposal = position + step_size * rng.standard_normal(D)
        proposal_log_probability = log_density_numpy(proposal)

        if np.log(rng.random()) < proposal_log_probability - log_probability:
            position = proposal
            log_probability = proposal_log_probability
            accepted += 1

        samples[iteration] = position

    return {
        "samples": samples,
        "acceptance_rate": accepted / NUM_SAMPLES,
        "warmup_acceptance_rate": warmup_acceptances / NUM_WARMUP,
        "step_size": step_size,
    }


def run_nuts(seed):
    """NUTS with BlackJAX window adaptation."""
    key = jax.random.PRNGKey(seed)
    initial_key, warmup_key, sampling_key = jax.random.split(key, 3)
    initial_position = jax.random.normal(initial_key, (D,))

    adaptation = blackjax.window_adaptation(blackjax.nuts, log_density_jax)
    (state, parameters), _ = adaptation.run(
        warmup_key,
        initial_position,
        num_steps=NUM_WARMUP,
    )

    kernel = blackjax.nuts(
        log_density_jax,
        **parameters,
        max_num_doublings=7,
    ).step

    @jax.jit
    def one_step(current_state, step_key):
        new_state, info = kernel(step_key, current_state)
        return new_state, (new_state.position, info.acceptance_rate)

    sampling_keys = jax.random.split(sampling_key, NUM_SAMPLES)
    state, (positions, acceptance_rates) = jax.lax.scan(
        one_step, state, sampling_keys
    )
    positions.block_until_ready()

    return {
        "samples": np.asarray(positions),
        "acceptance_rate": float(np.mean(np.asarray(acceptance_rates))),
        "step_size": float(np.asarray(parameters["step_size"])),
    }


# =============================================================================
# Error summaries
# =============================================================================

def chain_functionals(samples):
    """Two estimands chosen to expose heavy-tail exploration."""
    norms = np.linalg.norm(samples, axis=1)
    return {
        "tail probability": np.mean(norms >= TAIL_THRESHOLD),
        "mean norm": np.mean(norms),
    }


def reference_functionals(reference):
    norms = reference["norms"]
    return {
        "tail probability": np.mean(norms >= TAIL_THRESHOLD),
        "mean norm": np.mean(norms),
    }


def summarize(method_results, exact_values):
    estimates = {
        name: np.array(
            [chain_functionals(result["samples"])[name] for result in method_results]
        )
        for name in exact_values
    }

    summary = {}
    for name, values in estimates.items():
        summary[name] = {
            "estimates": values,
            "mean": np.mean(values),
            "standard_error": np.std(values, ddof=1) / np.sqrt(len(values)),
            "rmse": np.sqrt(np.mean((values - exact_values[name]) ** 2)),
        }
    return summary


def print_summary(results, summaries, exact_values):
    print("\nMonte Carlo comparison")
    print(
        f"Target: skew-t(d={D}, nu={NU:g}); "
        f"{NUM_CHAINS} chains x {NUM_SAMPLES} retained draws"
    )
    print("-" * 105)
    print(
        f"{'Method':<8} {'Estimand':<20} {'Reference':>12} "
        f"{'Estimate':>12} {'SE':>12} {'RMSE':>12} {'Accept':>10}"
    )
    print("-" * 105)

    for method in ("RWM", "NUTS"):
        acceptance = np.mean(
            [result["acceptance_rate"] for result in results[method]]
        )
        for estimand in ("tail probability", "mean norm"):
            values = summaries[method][estimand]
            print(
                f"{method:<8} {estimand:<20} "
                f"{exact_values[estimand]:>12.6g} "
                f"{values['mean']:>12.6g} "
                f"{values['standard_error']:>12.3g} "
                f"{values['rmse']:>12.3g} "
                f"{acceptance:>10.1%}"
            )
    print("-" * 105)


# =============================================================================
# Plots
# =============================================================================

METHODS = ("RWM", "NUTS")
COLORS = {"RWM": "#2878B5", "NUTS": "#E07A1F"}


def save_qq_plot(results, reference):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for dimension, axis in enumerate(axes.flat):
        reference_quantiles = np.quantile(
            reference["coordinates"][:, dimension], QQ_PROBABILITIES
        )
        lower = np.min(reference_quantiles)
        upper = np.max(reference_quantiles)
        axis.plot([lower, upper], [lower, upper], "k--", label="45-degree line")

        for method in METHODS:
            chain_quantiles = np.array(
                [
                    np.quantile(
                        result["samples"][:, dimension], QQ_PROBABILITIES
                    )
                    for result in results[method]
                ]
            )
            axis.plot(
                reference_quantiles,
                np.median(chain_quantiles, axis=0),
                "o-",
                color=COLORS[method],
                label=method,
            )
            axis.fill_between(
                reference_quantiles,
                np.percentile(chain_quantiles, 10, axis=0),
                np.percentile(chain_quantiles, 90, axis=0),
                color=COLORS[method],
                alpha=0.18,
            )

        axis.set_title(f"Coordinate {dimension + 1}")
        axis.set_xlabel("Exact quantiles")
        axis.set_ylabel("Estimated quantiles")
        axis.grid(alpha=0.25)

    axes.flat[0].legend()
    fig.suptitle("Q-Q plots (bands show the 10th-90th chain percentiles)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "qq_plots.png", dpi=180)
    plt.close(fig)


def save_estimator_boxplots(summaries, exact_values):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for axis, estimand in zip(axes, ("tail probability", "mean norm")):
        values = [
            summaries[method][estimand]["estimates"] for method in METHODS
        ]
        boxes = axis.boxplot(values, tick_labels=METHODS, patch_artist=True)
        for patch, method in zip(boxes["boxes"], METHODS):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.7)
        axis.axhline(
            exact_values[estimand],
            color="black",
            linestyle="--",
            label="Exact reference",
        )
        axis.set_title(estimand.capitalize())
        axis.set_ylabel("Chain estimate")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    fig.suptitle("Variation of estimates across independent chains")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "estimator_boxplots.png", dpi=180)
    plt.close(fig)


def save_rmse_plot(summaries):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for axis, estimand in zip(axes, ("tail probability", "mean norm")):
        rmse = [summaries[method][estimand]["rmse"] for method in METHODS]
        bars = axis.bar(
            METHODS,
            rmse,
            color=[COLORS[method] for method in METHODS],
            edgecolor="black",
        )
        axis.bar_label(bars, fmt="%.3g", padding=3)
        axis.set_title(estimand.capitalize())
        axis.set_ylabel("RMSE")
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Root mean squared error across chains")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "rmse.png", dpi=180)
    plt.close(fig)


def save_trace_plot(results):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for axis, method in zip(axes, METHODS):
        samples = results[method][0]["samples"]
        axis.plot(samples[:, 0], linewidth=0.7, color=COLORS[method])
        axis.set_title(f"{method}: first coordinate, first chain")
        axis.set_ylabel(r"$x_1$")
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Retained iteration")
    fig.suptitle("Representative trace plots")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "trace_plots.png", dpi=180)
    plt.close(fig)


# =============================================================================
# Main experiment
# =============================================================================

def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Generating {NUM_REFERENCE:,} exact reference draws...")
    reference = exact_skew_t_reference(NUM_REFERENCE, rng)
    exact_values = reference_functionals(reference)

    results = {"RWM": [], "NUTS": []}
    for chain in range(NUM_CHAINS):
        print(f"RWM chain {chain + 1}/{NUM_CHAINS}")
        results["RWM"].append(run_rwm(SEED + 100 + chain))

    for chain in range(NUM_CHAINS):
        print(f"NUTS chain {chain + 1}/{NUM_CHAINS}")
        results["NUTS"].append(run_nuts(SEED + 1_000 + chain))

    summaries = {
        method: summarize(results[method], exact_values) for method in METHODS
    }
    print_summary(results, summaries, exact_values)

    save_qq_plot(results, reference)
    save_estimator_boxplots(summaries, exact_values)
    save_rmse_plot(summaries)
    save_trace_plot(results)

    print(f"\nPlots saved in: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
