"""QQ plots of independent Monte Carlo averages, not of raw target draws.

Run: python -m clt_qq.experiment
Only NumPy, SciPy and Matplotlib are required; see README.md for interpretation.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm, t

from .nuts import nuts_step


@dataclass(frozen=True)
class Case:
    key: str
    df: float
    observable: str

    def values(self, x, threshold):
        return np.abs(x) if self.observable == "abs" else (np.abs(x) > threshold)

    def truth(self, threshold):
        if self.observable == "tail":
            return float(2 * t.sf(threshold, self.df))
        # E|T_nu| exists only for nu > 1.
        return float(np.exp(np.log(2 * np.sqrt(self.df) / np.sqrt(np.pi))
                            + gammaln((self.df + 1) / 2)
                            - gammaln(self.df / 2) - np.log(self.df - 1)))

    def title(self, threshold):
        observable = "g(x) = |x|" if self.observable == "abs" else f"g(x) = 1{{|x| > {threshold:g}}}"
        return f"Student t({self.df:g}), {observable}"


CASES = (Case("t3_abs", 3.0, "abs"), Case("t1_tail", 1.0, "tail"),
         Case("t1p5_tail", 1.5, "tail"))


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    kind: str
    steps: int = 0
    adjusted: bool = False


def methods_for(args):
    methods = [Method("iid", "i.i.d. reference", "iid"), Method("ula", "ULA", "ula")]
    for adjusted, prefix, label in ((False, "uhmc", "Unadjusted HMC"),
                                    (True, "hmc", "MH-HMC")):
        for steps in args.fixed_steps:
            methods.append(Method(f"{prefix}_{steps}", f"{label}, L={steps}", "fixed", steps, adjusted))
        methods.append(Method(f"{prefix}_random", f"{label}, L~Unif[1,{args.random_max}]",
                              "random", args.random_max, adjusted))
    methods.append(Method("nuts", f"NUTS, max depth={args.max_depth}", "nuts", adjusted=True))
    if args.methods:
        unknown = set(args.methods) - {m.key for m in methods}
        if unknown:
            raise ValueError(f"Unknown method(s): {', '.join(sorted(unknown))}")
        methods = [m for m in methods if m.key in args.methods]
    return methods


def rng_for(seed, case_key, method_key, replicate=0):
    """Stable streams: selecting fewer cases/methods does not change their output."""
    digest = hashlib.sha256(f"{case_key}/{method_key}".encode()).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").astype(int).tolist()
    return np.random.default_rng(np.random.SeedSequence([seed, replicate, *words]))


def potential(x, df):
    return 0.5 * (df + 1) * np.log1p(x * x / df)


def gradient(x, df):
    return (df + 1) * x / (df + x * x)


def leapfrog(x, p, df, epsilon, counts):
    """Vectorized independent trajectories, each with its own positive L."""
    q, momentum = x.copy(), p.copy()
    momentum -= 0.5 * epsilon * gradient(q, df)
    for step in range(int(np.max(counts))):
        active = counts > step
        q[active] += epsilon * momentum[active]
        # Fuse the adjacent half-kicks at interior positions.
        kick = np.where(counts[active] == step + 1, 0.5, 1.0)
        momentum[active] -= epsilon * kick * gradient(q[active], df)
    return q, momentum


def transition(x, case, method, epsilon, rng):
    if method.kind == "ula":
        # h = epsilon^2/2: x' = x - h grad U(x) + sqrt(2h) Z.
        with np.errstate(over="ignore", invalid="ignore"):
            q = x - (0.5 * epsilon) * epsilon * gradient(x, case.df) + epsilon * rng.normal(size=x.size)
        if not np.isfinite(q).all():
            raise FloatingPointError("Nonfinite ULA position; reduce the step size.")
        return q, x.size, 0
    counts = (rng.integers(1, method.steps + 1, size=x.size) if method.kind == "random"
              else np.full(x.size, method.steps))
    p = rng.normal(size=x.size)
    q, p_new = leapfrog(x, p, case.df, epsilon, counts)
    accepted = 0
    if method.adjusted:
        log_ratio = potential(x, case.df) + p*p/2 - potential(q, case.df) - p_new*p_new/2
        accept = -rng.exponential(size=x.size) < np.minimum(0.0, log_ratio)
        q = np.where(accept, q, x)
        accepted = int(np.sum(accept))
    if not np.isfinite(q).all():
        raise FloatingPointError("Nonfinite position; reduce the step size.")
    # The implementation uses L+1 force evaluations, with no inter-step cache.
    return q, int(np.sum(counts + 1)), accepted


def run_vectorized(case, method, args):
    """Each vector coordinate is an independent replicate; save nested prefixes."""
    rng = rng_for(args.seed, case.key, method.key)
    x = rng.standard_t(case.df, size=args.replicates)
    means = np.empty((len(args.lengths), args.replicates))
    sums = np.zeros(args.replicates)
    force_evals, accepts = 0, 0
    burn = args.unadjusted_burn_in if not method.adjusted and method.kind != "iid" else 0
    for _ in range(burn):
        x, _, _ = transition(x, case, method, args.step_size, rng)
    checkpoint = 0
    for iteration in range(1, args.lengths[-1] + 1):
        if method.kind == "iid":
            x = rng.standard_t(case.df, size=args.replicates)
        else:
            x, cost, accepted = transition(x, case, method, args.step_size, rng)
            force_evals += cost
            accepts += accepted
        sums += case.values(x, args.tail_threshold)
        if iteration == args.lengths[checkpoint]:
            means[checkpoint] = sums / iteration
            checkpoint += 1
    transitions = args.replicates * args.lengths[-1]
    return means, {"force_evals_per_draw": force_evals / transitions,
                   "acceptance_rate": accepts / transitions if method.adjusted else None,
                   "depth_cap_fraction": None, "divergence_fraction": None,
                   "burn_in": burn}


def _nuts_replicate(task):
    case, args, replicate = task
    rng = rng_for(args.seed, case.key, "nuts", replicate)
    # Exact target initialization eliminates burn-in for this invariant kernel.
    x = float(rng.standard_t(case.df))
    sums = 0.0
    means = np.empty(len(args.lengths))
    checkpoint, leapfrogs, caps, divergences = 0, 0, 0, 0
    for iteration in range(1, args.lengths[-1] + 1):
        x, cost, cap, divergent = nuts_step(x, case.df, args.step_size, args.max_depth, rng)
        leapfrogs += cost
        caps += cap
        divergences += divergent
        sums += float(case.values(x, args.tail_threshold))
        if iteration == args.lengths[checkpoint]:
            means[checkpoint] = sums / iteration
            checkpoint += 1
    return means, leapfrogs, caps, divergences


def run_nuts(case, args):
    tasks = [(case, args, r) for r in range(args.replicates)]
    if args.workers == 1:
        results = list(map(_nuts_replicate, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_nuts_replicate, tasks))
    means = np.stack([r[0] for r in results], axis=1)
    transitions = args.replicates * args.lengths[-1]
    counts = np.sum([r[1:] for r in results], axis=0) / transitions
    # nuts.py computes two gradients per leapfrog, with no gradient cache.
    return means, {"force_evals_per_draw": float(2 * counts[0]), "acceptance_rate": None,
                   "depth_cap_fraction": float(counts[1]), "divergence_fraction": float(counts[2]),
                   "burn_in": 0}


def robust_scale(values):
    """Normal-equivalent IQR, without assuming a finite empirical variance limit."""
    return float(np.subtract(*np.quantile(values, [0.75, 0.25])) / (2 * norm.ppf(0.75)))


def save_plots(case, methods, means, args):
    # Defer plotting import so NUTS workers do not initialize Matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = case.truth(args.tail_threshold)
    columns = min(3, len(methods))
    rows = math.ceil(len(methods) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.3*columns, 3.3*rows), squeeze=False)
    normal_q = norm.ppf((np.arange(args.replicates) + 0.5) / args.replicates)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.lengths)))
    for ax, method in zip(axes.flat, methods):
        for k, (length, color) in enumerate(zip(args.lengths, colors)):
            errors = np.sqrt(length) * (means[method.key][k] - truth)
            q25, q75 = np.quantile(errors, [0.25, 0.75])
            slope = (q75 - q25) / (2 * norm.ppf(0.75))
            ax.scatter(normal_q, np.sort(errors), s=8, alpha=0.65, color=color, label=f"n={length:,}")
            ax.plot(normal_q, (q25+q75)/2 + slope*normal_q, color=color, ls="--", lw=1)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_title(method.label, fontsize=10)
        ax.set_xlabel("Standard normal quantiles", fontsize=9)
        ax.set_ylabel(r"Quantiles of $\sqrt{n}(\bar{g}_n-\pi(g))$", fontsize=9)
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{case.title(args.tail_threshold)}\n{args.replicates} independent replicates; target mean = {truth:.6g}", fontsize=13)
    fig.text(0.5, 0.012, "Dashed: normal line through each sample's quartiles. Axes vary by sampler.\n"
             "ULA / unadjusted HMC may be shifted by discretization bias. Finite-run QQ plots do not prove CLT failure.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"qq_{case.key}.{suffix}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    has_zero_scale, has_positive_scale = False, False
    for method in methods:
        errors = (means[method.key] - truth) * np.sqrt(args.lengths)[:, None]
        scales = np.array([robust_scale(row) for row in errors])
        positive = scales > 0
        has_positive_scale |= bool(np.any(positive))
        line, = axes[0].plot(args.lengths, np.where(positive, scales, np.nan), "o-", ms=4, label=method.label)
        if not positive.all():
            has_zero_scale = True
            axes[0].plot(np.array(args.lengths)[~positive], np.full(np.sum(~positive), 0.025),
                         "v", color=line.get_color(), transform=axes[0].get_xaxis_transform())
        axes[1].plot(args.lengths, np.mean(means[method.key], axis=1) - truth, "o-", ms=4)
    axes[0].set(xscale="log", yscale="log", xlabel="Retained draws per replicate, n",
                ylabel=r"IQR of $\sqrt{n}(\bar{g}_n-\pi(g))$ / 1.349",
                title="Does the fluctuation width stabilize?")
    if not has_positive_scale:
        axes[0].set_ylim(0.5, 2)
    if has_zero_scale:
        axes[0].text(0.02, 0.98, "Bottom triangles: IQR = 0 (degenerate quartiles)",
                     transform=axes[0].transAxes, va="top", fontsize=8)
    axes[1].set(xscale="log", xlabel="Retained draws per replicate, n", ylabel="Grand mean minus target mean",
                title="Target error (includes discretization bias)")
    axes[1].axhline(0, color="0.4", ls="--", lw=1)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xticks(args.lengths, labels=[f"{n:,}" for n in args.lengths])
        ax.minorticks_off()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(case.title(args.tail_threshold))
    fig.tight_layout(rect=(0, 0.2, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"scaling_{case.key}.{suffix}", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--replicates", type=int, default=128, help="Independent chains / QQ points")
    parser.add_argument("--lengths", type=int, nargs="+", default=[250, 1000, 4000], help="Nested sample sizes")
    parser.add_argument("--step-size", type=float, default=0.35, help="Leapfrog epsilon; ULA h=epsilon^2/2")
    parser.add_argument("--fixed-steps", type=int, nargs="+", default=[5, 10], help="Fixed HMC leapfrog counts")
    parser.add_argument("--random-max", type=int, default=10, help="Random L is uniform on 1,...,this value")
    parser.add_argument("--max-depth", type=int, default=7, help="NUTS tree depth cap; at most 2^depth-1 leapfrogs")
    parser.add_argument("--unadjusted-burn-in", type=int, default=1000, help="Discarded ULA/UHMC transitions")
    parser.add_argument("--tail-threshold", type=float, default=2.0, help="Two-sided event |x| > threshold")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--workers", type=int, default=1, help="Parallel NUTS workers; independent seeds")
    parser.add_argument("--cases", nargs="+", choices=[c.key for c in CASES], default=[c.key for c in CASES])
    parser.add_argument("--methods", nargs="+", help="Subset, e.g. iid ula hmc_5 hmc_random nuts")
    parser.add_argument("--output", type=Path, default=Path("clt_qq/output"))
    args = parser.parse_args(argv)
    if args.replicates < 8:
        parser.error("--replicates must be at least 8 (use 128 or more for meaningful QQ plots)")
    if any(x <= 0 for x in args.lengths + args.fixed_steps) or args.random_max < 1:
        parser.error("Lengths and leapfrog counts must be positive")
    if not math.isfinite(args.step_size) or args.step_size <= 0:
        parser.error("--step-size must be finite and positive")
    if not math.isfinite(args.tail_threshold) or args.tail_threshold <= 0:
        parser.error("--tail-threshold must be finite and positive")
    if args.workers < 1 or args.max_depth < 1 or args.max_depth > 15 or args.unadjusted_burn_in < 0 or args.seed < 0:
        parser.error("Require workers >= 1, depth in [1,15], burn-in >= 0, seed >= 0")
    args.lengths = sorted(set(args.lengths))
    args.fixed_steps = sorted(set(args.fixed_steps))
    try:
        methods_for(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    methods = methods_for(args)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["python_version"] = sys.version.split()[0]
    config["numpy_version"] = np.__version__
    import scipy
    import matplotlib
    config["scipy_version"] = scipy.__version__
    config["matplotlib_version"] = matplotlib.__version__
    config["initialization"] = "Independent exact target draws; burn-in for ULA/UHMC only"
    config["cost_note"] = "Force evaluations exclude burn-in; comparisons use equal retained iterations, not equal work"
    config["ula_note"] = "ULA/UHMC target errors include bias; their invariant laws are not exactly Student t"
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    summaries, diagnostics = [], {}
    for case in CASES:
        if case.key not in args.cases:
            continue
        truth, means = case.truth(args.tail_threshold), {}
        for method in methods:
            print(f"{case.key:12s} {method.key:14s} ...", end=" ", flush=True)
            start = time.perf_counter()
            values, info = run_nuts(case, args) if method.kind == "nuts" else run_vectorized(case, method, args)
            info["seconds"] = time.perf_counter() - start
            means[method.key] = values
            diagnostics[f"{case.key}/{method.key}"] = info
            print(f"{info['seconds']:.1f}s; {info['force_evals_per_draw']:.1f} force evals/draw", flush=True)
            if method.kind == "nuts":
                print(f"  NUTS depth-cap fraction: {info['depth_cap_fraction']:.3%}; "
                      f"divergence fraction: {info['divergence_fraction']:.3%}", flush=True)
            for k, length in enumerate(args.lengths):
                errors = np.sqrt(length) * (values[k] - truth)
                summaries.append({"case": case.key, "method": method.key, "n": length, "target_mean": truth,
                                  "grand_mean": float(np.mean(values[k])),
                                  "target_error": float(np.mean(values[k]) - truth),
                                  "root_n_iqr_scale": robust_scale(errors),
                                  "root_n_sd": float(np.std(errors, ddof=1))})
            # Save each completed method, so interrupted long runs retain results.
            np.savez_compressed(args.output / f"means_{case.key}.npz", lengths=args.lengths,
                                target_mean=truth, **means)
            (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        save_plots(case, methods, means, args)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Plots and replicate averages saved in {args.output.resolve()}")


if __name__ == "__main__":
    main()
