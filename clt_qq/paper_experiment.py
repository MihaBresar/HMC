"""Reproduce the Cauchy chain averages used by the three paper figures.

Run ``python -m clt_qq.paper_experiment``. Compatible, previously generated
averages are reused exactly, with their original diagnostics and file hashes.
Every missing method runs genuine independent chains in worker processes.
``--recompute`` reruns every method, including the baseline samplers.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy

from .experiment import CASES, Method, run_ensemble


ROOT = Path(__file__).resolve().parent
CASE = next(case for case in CASES if case.key == "t1_tail")
BASELINE_KEYS = {"ula", "uhmc_10", "hmc_5", "hmc_10", "nuts"}
SETTING_KEYS = ("chains", "iterations", "burn_in", "step_size", "tail_threshold",
                "seed", "batch_size")


def paper_methods():
    return [
        Method("ula", "ULA", "ula"),
        Method("uhmc_10", "Unadjusted HMC, L=10", "fixed", 10),
        Method("uhmc_uniform_19", "Unadjusted HMC, L~Uniform{1,...,19}", "random", 19),
        Method("hmc_5", "HMC, L=5", "fixed", 5, True),
        Method("hmc_10", "HMC, L=10", "fixed", 10, True),
        Method("hmc_20", "HMC, L=20", "fixed", 20, True),
        Method("hmc_uniform_9", "HMC, L~Uniform{1,...,9}", "random", 9, True),
        Method("hmc_uniform_19", "HMC, L~Uniform{1,...,19}", "random", 19, True),
        Method("hmc_uniform_39", "HMC, L~Uniform{1,...,39}", "random", 39, True),
        Method("hmc_two_point_9", "HMC, P(L=1)=P(L=9)=1/2", "two_point", 9, True),
        Method("hmc_two_point_39", "HMC, P(L=1)=P(L=39)=1/2", "two_point", 39, True),
        Method("nuts", "NUTS", "nuts", adjusted=True),
    ]


def method_settings(method):
    settings = {"kind": method.kind, "metropolis_adjusted": method.adjusted,
                "momentum": "Independent N(0,1) refresh at each transition"}
    if method.kind == "fixed":
        settings.update(leapfrog_law={"values": [method.steps], "probabilities": [1.0]},
                        expected_leapfrog_steps=float(method.steps),
                        expected_squared_leapfrog_steps=float(method.steps**2))
    elif method.kind in {"random", "two_point"}:
        values = (list(range(1, method.steps+1)) if method.kind == "random"
                  else [1, method.steps])
        settings.update(leapfrog_law={"values": values,
                                      "probabilities": [1/len(values)]*len(values)},
                        expected_leapfrog_steps=float(np.mean(values)),
                        expected_squared_leapfrog_steps=float(np.mean(np.square(values))),
                        length_draw="Independent of state and momentum; redrawn each transition")
    elif method.kind == "ula":
        settings.update(update="x' = x - epsilon^2/2 * grad U(x) + epsilon * Z")
    else:
        settings.update(backend="blackjax.nuts", inverse_mass_matrix=[1.0],
                        adaptation=False, precision="float64", library_defaults=True)
    return settings


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(args):
    return {**{key: getattr(args, key) for key in SETTING_KEYS},
            "case": CASE.key, "df": CASE.df, "observable": CASE.observable,
            "initial_position": 0.0,
            "methods": {method.key: method_settings(method) for method in paper_methods()}}


def load_saved(directory, args, baseline=False):
    """Only reuse compatible complete arrays, together with their diagnostics."""
    config_path = directory / "config.json"
    means_path = directory / f"means_{CASE.key}.npz"
    diagnostics_path = directory / "diagnostics.json"
    if not all(path.is_file() for path in (config_path, means_path, diagnostics_path)):
        return {}, {}, {}
    config = json.loads(config_path.read_text())
    if baseline:
        compatible = all(config.get(key) == getattr(args, key) for key in SETTING_KEYS)
        compatible &= config.get("initialization") == "Every chain starts at zero"
        compatible &= config.get("nuts", {}).get("backend") == "blackjax.nuts"
    else:
        compatible = config.get("experiment_signature") == signature(args)
    if not compatible:
        return {}, {}, {}
    diagnostics = json.loads(diagnostics_path.read_text())
    means, info = {}, {}
    with np.load(means_path, allow_pickle=False) as saved:
        if not (np.array_equal(saved["chain_ids"], np.arange(args.chains))
                and int(saved["iterations"]) == args.iterations
                and int(saved["burn_in"]) == args.burn_in
                and float(saved["target_mean"]) == CASE.truth(args.tail_threshold)):
            raise ValueError(f"Stored metadata does not match its configuration: {means_path}")
        for method in paper_methods():
            key = method.key
            if key not in saved or (baseline and key not in BASELINE_KEYS):
                continue
            values = saved[key].copy()
            diag_key = f"{CASE.key}/{key}"
            if values.shape != (args.chains,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid chain averages: {means_path}:{key}")
            if diag_key not in diagnostics:
                raise ValueError(f"Missing diagnostics: {diagnostics_path}:{diag_key}")
            means[key], info[diag_key] = values, dict(diagnostics[diag_key])
            if baseline:
                info[diag_key]["provenance"] = {
                    "kind": "reused_baseline",
                    "path": "clt_qq/examples/simple" if directory == ROOT / "examples/simple" else str(directory),
                    "means_sha256": sha256(means_path),
                    "config_sha256": sha256(config_path),
                    "diagnostics_sha256": sha256(diagnostics_path),
                    "note": "Original chain means and process diagnostics, copied without resampling",
                }
    return means, info, config


def save_results(args, means, diagnostics, config):
    ordered = {method.key: means[method.key] for method in paper_methods() if method.key in means}
    truth = CASE.truth(args.tail_threshold)
    np.savez_compressed(args.output / f"means_{CASE.key}.npz", chain_ids=np.arange(args.chains),
                        iterations=args.iterations, burn_in=args.burn_in, target_mean=truth, **ordered)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    with (args.output / f"chain_means_{CASE.key}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["chain_id", *ordered])
        writer.writerows([i, *(values[i] for values in ordered.values())] for i in range(args.chains))
    rows = [{"case": CASE.key, "method": key, "chains": args.chains,
             "iterations": args.iterations, "burn_in": args.burn_in,
             "retained": args.iterations-args.burn_in, "target_mean": truth,
             "mean_of_chain_means": float(np.mean(values)),
             "sd_of_chain_means": float(np.std(values, ddof=1))}
            for key, values in ordered.items()]
    if rows:
        with (args.output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/data")
    parser.add_argument("--baseline", type=Path, default=ROOT / "examples/simple")
    parser.add_argument("--recompute", action="store_true", help="Rerun all methods, including baseline samplers")
    parser.add_argument("--chains", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=0.35)
    parser.add_argument("--tail-threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.burn_in is None:
        args.burn_in = args.iterations // 3
    if args.chains < 8 or args.iterations <= 0 or not 0 <= args.burn_in < args.iterations:
        parser.error("Require chains >= 8 and 0 <= burn-in < iterations")
    if any(not math.isfinite(value) or value <= 0 for value in (args.step_size, args.tail_threshold)):
        parser.error("Step size and threshold must be finite and positive")
    if args.seed < 0 or args.workers < 1 or args.batch_size < 1:
        parser.error("Require seed >= 0, workers >= 1, batch-size >= 1")
    args.output, args.baseline = args.output.resolve(), args.baseline.resolve()
    if args.output == args.baseline:
        parser.error("Output must differ from the baseline directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    means, diagnostics, previous_config = ({}, {}, {}) if args.recompute else load_saved(args.output, args)
    baseline, baseline_info, baseline_config = ({}, {}, {}) if args.recompute else load_saved(args.baseline, args, True)
    for key, values in baseline.items():
        if key not in means:
            means[key] = values
            diagnostics[f"{CASE.key}/{key}"] = baseline_info[f"{CASE.key}/{key}"]
    config = {
        **{key: getattr(args, key) for key in SETTING_KEYS},
        "experiment_signature": signature(args), "workers_requested": args.workers,
        "retained_per_chain": args.iterations-args.burn_in,
        "initialization": "Every chain starts at zero",
        "qq": "Raw chain means against fitted Gaussian quantiles; identity line; no sqrt(n) scaling",
        "comparison_note": "Equal iterations, not equal work; random laws match E[L], not E[L^2]",
        "rng_note": "Stable case/method/batch streams; independent increments per chain; scheduling does not affect results",
        "python_version": sys.version.split()[0], "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "baseline": "clt_qq/examples/simple" if args.baseline == ROOT / "examples/simple" else str(args.baseline),
    }
    if "nuts" in means:
        config["nuts"] = (previous_config if "nuts" in previous_config else baseline_config)["nuts"]
    else:
        from .nuts import backend_metadata
        config["nuts"] = {**backend_metadata(), "step_size": args.step_size,
                          "inverse_mass_matrix": [1.0], "adaptation": False, "precision": "float64"}
    save_results(args, means, diagnostics, config)
    missing = [method for method in paper_methods() if method.key not in means]
    print(f"Reuse {len(means)} methods; simulate {len(missing)} methods: "
          f"{args.chains:,} chains x {args.iterations:,} transitions; "
          f"{args.burn_in:,} burn-in; {args.workers} workers", flush=True)
    context = (ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
               if args.workers > 1 and missing else nullcontext(None))
    with context as pool:
        for method in missing:
            print(f"{CASE.key}/{method.key} ...", flush=True)
            start = time.perf_counter()
            values, info = run_ensemble(CASE, method, args, pool, progress=True)
            info["seconds"] = time.perf_counter()-start
            info["provenance"] = {"kind": "paper_experiment", "module": "clt_qq.paper_experiment"}
            means[method.key], diagnostics[f"{CASE.key}/{method.key}"] = values, info
            save_results(args, means, diagnostics, config)
            print(f"  {info['chains_completed']:,} chains in {info['seconds']:.1f}s; "
                  f"{info['worker_processes_used']} worker processes", flush=True)
    print(f"Saved {len(means)} methods to {args.output}", flush=True)


if __name__ == "__main__":
    main()
