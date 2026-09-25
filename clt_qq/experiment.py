"""QQ plots of independent Monte Carlo averages, not of raw target draws.

Run: python -m clt_qq.experiment
Only NumPy, SciPy and Matplotlib are required; see README.md for interpretation.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln
from scipy.stats import t

from .nuts import nuts_step


@dataclass(frozen=True)
class Case:
    key: str
    df: float
    observable: str

    def values(self, x, threshold):
        return np.abs(x) if self.observable == "abs" else (np.asarray(x) >= threshold)

    def truth(self, threshold):
        if self.observable == "tail":
            return float(t.sf(threshold, self.df))
        # E|T_nu| exists only for nu > 1.
        return float(np.exp(np.log(2 * np.sqrt(self.df) / np.sqrt(np.pi))
                            + gammaln((self.df + 1) / 2)
                            - gammaln(self.df / 2) - np.log(self.df - 1)))

    def title(self, threshold):
        observable = "g(x) = |x|" if self.observable == "abs" else f"g(x) = 1{{x >= {threshold:g}}}"
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


def run_vectorized(case, method, args, batch_id=0, count=None):
    """Independent chains in a worker's vector; each coordinate has its own sum."""
    count = args.chains if count is None else count
    rng = rng_for(args.seed, case.key, method.key, batch_id)
    x = np.zeros(count)
    sums = np.zeros(count)
    force_evals, accepts = 0, 0
    for iteration in range(args.iterations):
        if method.kind == "iid":
            x = rng.standard_t(case.df, size=count)
            cost, accepted = 0, 0
        else:
            x, cost, accepted = transition(x, case, method, args.step_size, rng)
        if iteration >= args.burn_in:
            sums += case.values(x, args.tail_threshold)
            force_evals += cost
            accepts += accepted
    retained = args.iterations - args.burn_in
    return sums / retained, {"force_evals": force_evals, "accepted": accepts,
                             "depth_caps": 0, "divergences": 0}


def _nuts_replicate(case, args, chain_id):
    rng = rng_for(args.seed, case.key, "nuts", chain_id)
    x = 0.0
    total = 0.0
    leapfrogs, caps, divergences = 0, 0, 0
    for iteration in range(args.iterations):
        x, cost, cap, divergent = nuts_step(x, case.df, args.step_size, args.max_depth, rng)
        if iteration >= args.burn_in:
            total += float(case.values(x, args.tail_threshold))
            leapfrogs += cost
            caps += cap
            divergences += divergent
    return total / (args.iterations - args.burn_in), leapfrogs, caps, divergences


def _run_batch(task):
    case, method, args, start, count, batch_id = task
    if method.kind == "nuts":
        results = [_nuts_replicate(case, args, i) for i in range(start, start + count)]
        means = np.array([r[0] for r in results])
        stats = {"force_evals": 2 * sum(r[1] for r in results), "accepted": 0,
                 "depth_caps": sum(r[2] for r in results), "divergences": sum(r[3] for r in results)}
    else:
        means, stats = run_vectorized(case, method, args, batch_id, count)
    return start, means, stats, os.getpid()


def run_ensemble(case, method, args, pool=None, progress=False):
    """Process-parallel batches; collect one mean for each numbered chain.

    Batch boundaries and seeds do not depend on worker scheduling or count.
    A vector batch draws independent random increments for every coordinate.
    NUTS has a separate generator for every chain. No chain interacts with
    another, and no chain is split into pseudo-independent subsamples.
    """
    batch_size = min(args.batch_size, 16) if method.kind == "nuts" else args.batch_size
    tasks = [(case, method, args, start, min(batch_size, args.chains-start), batch_id)
             for batch_id, start in enumerate(range(0, args.chains, batch_size))]
    if pool is None and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as own_pool:
            return run_ensemble(case, method, args, own_pool, progress)
    if pool is None:
        results = map(_run_batch, tasks)
    else:
        futures = [pool.submit(_run_batch, task) for task in tasks]
        results = (future.result() for future in as_completed(futures))
    means = np.empty(args.chains)
    totals = {"force_evals": 0, "accepted": 0, "depth_caps": 0, "divergences": 0}
    pids = set()
    completed, reported = 0, time.monotonic()
    for start, values, stats, pid in results:
        means[start:start+len(values)] = values
        for key in totals:
            totals[key] += stats[key]
        pids.add(pid)
        completed += len(values)
        if progress and time.monotonic() - reported > 20:
            print(f"  {completed:,}/{args.chains:,} chains complete", flush=True)
            reported = time.monotonic()
    draws = args.chains * (args.iterations - args.burn_in)
    info = {"chains_completed": completed, "worker_pids": sorted(pids),
            "worker_processes_used": len(pids), "batches": len(tasks),
            "iterations_per_chain": args.iterations, "burn_in_per_chain": args.burn_in,
            "retained_per_chain": args.iterations-args.burn_in,
            "force_evals_per_retained_draw": totals["force_evals"] / draws,
            "acceptance_rate": totals["accepted"] / draws if method.adjusted and method.kind != "nuts" else None,
            "depth_cap_fraction": totals["depth_caps"] / draws if method.kind == "nuts" else None,
            "divergence_fraction": totals["divergences"] / draws if method.kind == "nuts" else None}
    return means, info


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--chains", "--replicates", dest="chains", type=int, default=2000,
                        help="Independent chains; one ergodic average / QQ point per chain")
    parser.add_argument("--iterations", type=int, default=30000, help="Total transitions per chain, including burn-in")
    parser.add_argument("--burn-in", type=int, default=None, help="Discarded transitions; default floor(iterations/3)")
    parser.add_argument("--step-size", type=float, default=0.35, help="Leapfrog epsilon; ULA h=epsilon^2/2")
    parser.add_argument("--fixed-steps", type=int, nargs="+", default=[5, 10], help="Fixed HMC leapfrog counts")
    parser.add_argument("--random-max", type=int, default=10, help="Random L is uniform on 1,...,this value")
    parser.add_argument("--max-depth", type=int, default=7, help="NUTS tree depth cap")
    parser.add_argument("--tail-threshold", type=float, default=2.0, help="One-sided event x >= threshold (paper Figure 1)")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1), help="Worker processes for ALL samplers")
    parser.add_argument("--batch-size", type=int, default=128, help="Chains per vectorized worker batch (NUTS capped at 16)")
    parser.add_argument("--cases", nargs="+", choices=[c.key for c in CASES], default=[c.key for c in CASES])
    parser.add_argument("--methods", nargs="+", help="Subset, e.g. iid ula hmc_5 hmc_random nuts")
    parser.add_argument("--output", type=Path, default=Path("clt_qq/output/simple"))
    args = parser.parse_args(argv)
    if args.burn_in is None:
        args.burn_in = args.iterations // 3
    if args.chains < 8:
        parser.error("--chains must be at least 8; use 2,000 or more for smoother QQ plots")
    if args.iterations < 1 or not 0 <= args.burn_in < args.iterations:
        parser.error("Require 0 <= burn-in < iterations")
    if any(x <= 0 for x in args.fixed_steps) or args.random_max < 1:
        parser.error("Leapfrog counts must be positive")
    if not math.isfinite(args.step_size) or args.step_size <= 0:
        parser.error("--step-size must be finite and positive")
    if not math.isfinite(args.tail_threshold) or args.tail_threshold <= 0:
        parser.error("--tail-threshold must be finite and positive")
    if args.workers < 1 or not 1 <= args.max_depth <= 15 or args.seed < 0 or args.batch_size < 1:
        parser.error("Require workers >= 1, depth in [1,15], seed >= 0, batch-size >= 1")
    args.fixed_steps = sorted(set(args.fixed_steps))
    try:
        methods_for(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    from .plots import save_plots
    import scipy
    import matplotlib

    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    methods = methods_for(args)
    config = vars(args).copy()
    config.update(output=str(args.output), python_version=sys.version.split()[0],
                  numpy_version=np.__version__, scipy_version=scipy.__version__,
                  matplotlib_version=matplotlib.__version__, initialization="Every chain starts at zero",
                  retained_per_chain=args.iterations-args.burn_in,
                  qq="Raw chain means versus Normal(mean(chain_means), var(chain_means, ddof=1)); identity line; no sqrt(n)",
                  cost_note="Force evaluations exclude burn-in; equal iterations, not equal work",
                  rng_note="Independent streams by case/method/batch; per-chain streams for NUTS; worker count does not change results")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"{args.chains:,} chains x {args.iterations:,} iterations; "
          f"discard first {args.burn_in:,}; {args.workers} worker processes", flush=True)
    summaries, diagnostics = [], {}
    context = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else nullcontext(None)
    with context as pool:
        for case in CASES:
            if case.key not in args.cases:
                continue
            truth, means = case.truth(args.tail_threshold), {}
            for method in methods:
                print(f"{case.key:12s} {method.key:14s} ...", flush=True)
                start = time.perf_counter()
                values, info = run_ensemble(case, method, args, pool, progress=True)
                info["seconds"] = time.perf_counter() - start
                means[method.key] = values
                diagnostics[f"{case.key}/{method.key}"] = info
                print(f"  {info['chains_completed']:,} chains completed in {info['seconds']:.1f}s; "
                      f"{info['worker_processes_used']} worker processes used", flush=True)
                summaries.append({"case": case.key, "method": method.key, "chains": args.chains,
                                  "iterations": args.iterations, "burn_in": args.burn_in,
                                  "retained": args.iterations-args.burn_in, "target_mean": truth,
                                  "mean_of_chain_means": float(np.mean(values)),
                                  "sd_of_chain_means": float(np.std(values, ddof=1))})
                # Save completed means before beginning the next expensive method.
                np.savez_compressed(args.output / f"means_{case.key}.npz", chain_ids=np.arange(args.chains),
                                    iterations=args.iterations, burn_in=args.burn_in, target_mean=truth, **means)
                (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
                with (args.output / "summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                    writer.writeheader()
                    writer.writerows(summaries)
            with (args.output / f"chain_means_{case.key}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["chain_id", *means])
                writer.writerows([i, *(means[key][i] for key in means)] for i in range(args.chains))
            save_plots(case, methods, means, args)
    print(f"Simple QQ plots and chain averages saved in {args.output.resolve()}")


if __name__ == "__main__":
    main()
