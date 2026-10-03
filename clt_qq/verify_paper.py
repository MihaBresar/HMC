"""Validate the published 80k paper data and figure metadata without sampling.

Run ``python -m clt_qq.verify_paper`` from the repository checkout. This checks
internal consistency and saved provenance; it does not establish a CLT or
prove independence from the stored averages alone.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import BadZipFile

import numpy as np

from .paper_experiment import CASE_METHOD_KEYS, selected_case, signature
from .plots import normal_qq_coordinates


PAPER_DIR = Path(__file__).resolve().parent / "paper"
PROTOCOL = dict(chains=2000, iterations=120000, burn_in=40000,
                step_size=.35, tail_threshold=2., seed=20260925, batch_size=128)
RETAINED = PROTOCOL["iterations"] - PROTOCOL["burn_in"]
FIGURE_METHODS = {
    "01_ula_fixed_random": ("t3_abs", ("ula", "uhmc_10", "uhmc_uniform_19")),
    "02_randomisation_and_length": ("t1_tail", (
        "hmc_5", "hmc_uniform_9", "hmc_two_point_9",
        "hmc_20", "hmc_uniform_39", "hmc_two_point_39")),
    "03_hmc_nuts": ("t1_tail", ("hmc_10", "hmc_uniform_19", "nuts")),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected, message):
    # Permit roundoff across NumPy/SciPy versions, but never missing/nonfinite
    # values or changes large enough to affect the published numerical data.
    require(np.allclose(actual, expected, rtol=1e-12, atol=1e-14), message)


def padded_limits(values):
    low, high = float(values[0]), float(values[-1])
    span = high - low
    padding = .05 * span if span > 0 else .05 * max(abs(low), 1.)
    return [low - padding, high + padding]


def verify_case(paper_dir, case_key):
    directory = paper_dir / "data_80k" / case_key
    config = read_json(directory / "config.json")
    args = SimpleNamespace(**PROTOCOL, case=case_key)
    case = selected_case(args)
    methods = CASE_METHOD_KEYS[case_key]
    for key, expected in {**PROTOCOL, "case": case_key, "df": case.df,
                          "observable": case.observable,
                          "retained_per_chain": RETAINED}.items():
        require(config[key] == expected, f"{case_key}: incorrect config {key}")
    require(config["experiment_signature"] == signature(args),
            f"{case_key}: experiment signature or method law differs")
    require(config["initialization"] == "Every chain starts at zero",
            f"{case_key}: incorrect initialization")
    if "nuts" in methods:
        for key, expected in dict(backend="blackjax.nuts", blackjax_version="1.6.2",
                                  jax_version="0.11.2", default_max_num_doublings=10,
                                  step_size=.35, inverse_mass_matrix=[1.],
                                  adaptation=False, precision="float64").items():
            require(config["nuts"][key] == expected,
                    f"{case_key}: incorrect NUTS setting {key}")

    source = directory / f"means_{case_key}.npz"
    with np.load(source, allow_pickle=False) as archive:
        means = {key: archive[key].copy() for key in archive.files}
    metadata = {"chain_ids", "iterations", "burn_in", "target_mean"}
    require(set(means) == set(methods) | metadata,
            f"{case_key}: missing or unexpected NPZ arrays")
    require(np.array_equal(means["chain_ids"], np.arange(PROTOCOL["chains"])),
            f"{case_key}: chain IDs must be ordered, unique and complete")
    for key in ("iterations", "burn_in"):
        require(means[key].shape == () and means[key].item() == PROTOCOL[key],
                f"{case_key}: incorrect NPZ {key}")
    require(means["target_mean"].shape == (), f"{case_key}: nonscalar target mean")
    close(means["target_mean"], case.truth(PROTOCOL["tail_threshold"]),
          f"{case_key}: incorrect target mean")
    for method in methods:
        values = means[method]
        require(values.shape == (PROTOCOL["chains"],) and np.isfinite(values).all(),
                f"{case_key}/{method}: invalid chain-average shape or nonfinite values")
        require(np.all(values >= 0) and (case.observable != "tail" or np.all(values <= 1)),
                f"{case_key}/{method}: chain averages outside observable range")

    with (directory / f"chain_means_{case_key}.csv").open(newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader) == ["chain_id", *methods], f"{case_key}: incorrect CSV columns")
        rows = list(reader)
    require(len(rows) == PROTOCOL["chains"] and all(len(row) == len(methods)+1 for row in rows),
            f"{case_key}: incorrect CSV dimensions")
    require([int(row[0]) for row in rows] == list(range(PROTOCOL["chains"])),
            f"{case_key}: CSV chain IDs are missing, duplicated or out of order")
    csv_values = np.asarray([row[1:] for row in rows], dtype=float)
    require(np.array_equal(csv_values, np.column_stack([means[key] for key in methods])),
            f"{case_key}: CSV chain averages differ from NPZ")

    with (directory / "summary.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    require([row["method"] for row in summaries] == list(methods),
            f"{case_key}: missing, duplicated or reordered summary methods")
    for row in summaries:
        method = row["method"]
        require(row["case"] == case_key, f"{case_key}/{method}: incorrect summary case")
        for key in ("chains", "iterations", "burn_in", "retained"):
            expected = RETAINED if key == "retained" else PROTOCOL[key]
            require(int(row[key]) == expected, f"{case_key}/{method}: incorrect summary {key}")
        for key, expected in dict(target_mean=means["target_mean"],
                                  mean_of_chain_means=np.mean(means[method]),
                                  sd_of_chain_means=np.std(means[method], ddof=1)).items():
            close(float(row[key]), expected, f"{case_key}/{method}: inconsistent summary {key}")

    diagnostics = read_json(directory / "diagnostics.json")
    require(set(diagnostics) == {f"{case_key}/{method}" for method in methods},
            f"{case_key}: missing or unexpected method diagnostics")
    for method in methods:
        info = diagnostics[f"{case_key}/{method}"]
        for key, expected in dict(chains_completed=PROTOCOL["chains"],
                                  iterations_per_chain=PROTOCOL["iterations"],
                                  burn_in_per_chain=PROTOCOL["burn_in"],
                                  retained_per_chain=RETAINED).items():
            require(info[key] == expected, f"{case_key}/{method}: incorrect diagnostics {key}")
        require(info["worker_processes_used"] == len(set(info["worker_pids"])) >= 1,
                f"{case_key}/{method}: inconsistent recorded worker count")
    return means, source, case


def verify_paper(paper_dir=PAPER_DIR):
    """Raise ValueError for inconsistent data; return a concise success report."""
    paper_dir = Path(paper_dir)
    datasets = {key: verify_case(paper_dir, key) for key in CASE_METHOD_KEYS}
    manifest = read_json(paper_dir / "figures/manifest.json")
    for key in ("chains", "iterations", "burn_in", "retained"):
        expected = RETAINED if key == "retained" else PROTOCOL[key]
        require(manifest[key] == expected, f"Manifest: incorrect {key}")
    require(set(manifest["sources"]) == set(datasets), "Manifest: incorrect source cases")
    for key, (_, source, case) in datasets.items():
        recorded = manifest["sources"][key]
        expected_suffix = Path("data_80k") / key / source.name
        require(Path(recorded["path"]).parts[-3:] == expected_suffix.parts,
                f"Manifest: incorrect source path for {key}")
        require(recorded["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest(),
                f"Manifest: source hash mismatch for {key}")
        require(recorded["df"] == case.df and recorded["observable"] == case.observable,
                f"Manifest: incorrect target for {key}")
    require(set(manifest["figures"]) == set(FIGURE_METHODS), "Manifest: incorrect figure names")
    for name, (case_key, methods) in FIGURE_METHODS.items():
        panels = manifest["figures"][name]
        require(len(panels) == len(methods), f"{name}: incorrect panel count")
        for index, (panel, method) in enumerate(zip(panels, methods)):
            require((panel["panel"], panel["case"], panel["method"], panel["points"]) ==
                    (chr(ord("a")+index), case_key, method, PROTOCOL["chains"]),
                    f"{name}: incorrect panel mapping or point count")
            theoretical, empirical = normal_qq_coordinates(datasets[case_key][0][method])
            for axis, values in (("xlim", theoretical), ("ylim", empirical)):
                require(np.asarray(panel[axis]).shape == (2,), f"{name}/{method}: invalid {axis}")
                close(panel[axis], padded_limits(values), f"{name}/{method}: inconsistent QQ {axis}")
        for suffix, header in (("pdf", b"%PDF-"), ("png", b"\x89PNG\r\n\x1a\n")):
            path = paper_dir / "figures" / f"{name}.{suffix}"
            require(path.is_file() and path.stat().st_size > len(header), f"Missing/empty artifact: {path}")
            with path.open("rb") as stream:
                require(stream.read(len(header)) == header, f"Invalid {suffix.upper()} header: {path}")
        caption = paper_dir / "latex" / f"{name}.tex"
        require(caption.is_file() and caption.stat().st_size > 0, f"Missing/empty caption: {caption}")
        require(f"figures/{name}.pdf" in caption.read_text(encoding="utf-8"),
                f"Caption references the wrong figure: {caption}")
    return ("Verified 2 targets, 12 methods, 24,000 chain averages and 3 paper figures.\n"
            "Each setting: 2,000 chains, 40,000 burn-in + 80,000 retained iterations.\n"
            "Configs, diagnostics, CSV/NPZ values, summaries, source hashes, QQ limits "
            "and artifact files agree. No simulations were run.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR,
                        help="Directory containing data_80k/, figures/ and latex/")
    args = parser.parse_args(argv)
    try:
        report = verify_paper(args.paper_dir)
    except (OSError, ValueError, KeyError, TypeError, StopIteration, csv.Error, BadZipFile) as error:
        print(f"Paper verification failed: {error}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
