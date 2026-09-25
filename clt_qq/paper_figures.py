"""Build the three manuscript figures from saved independent chain averages.

Run the simulations with ``python -m clt_qq.paper_experiment`` first, then:
    python -m clt_qq.paper_figures
No sampler runs, outlier removal or rescaling occur in this plotting command.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .plots import normal_qq_coordinates


# The order and settings are specified before looking at the new simulations.
FIGURES = (
    ("01_ula_fixed_random", 1, (
        ("ula", "ULA\n$T=1$"),
        ("uhmc_10", "Unadjusted HMC\n$T=10$"),
        ("uhmc_uniform_19", "Unadjusted HMC\n$T\\sim\\mathrm{Unif}\\{1,\\ldots,19\\}$"),
    )),
    ("02_randomisation_and_length", 2, (
        ("hmc_5", "Fixed\n$T=5$"),
        ("hmc_uniform_9", "Uniform\n$T\\sim\\mathrm{Unif}\\{1,\\ldots,9\\}$"),
        ("hmc_two_point_9", "Two-point\n$T\\in\\{1,9\\}$, equally likely"),
        ("hmc_20", "Fixed\n$T=20$"),
        ("hmc_uniform_39", "Uniform\n$T\\sim\\mathrm{Unif}\\{1,\\ldots,39\\}$"),
        ("hmc_two_point_39", "Two-point\n$T\\in\\{1,39\\}$, equally likely"),
    )),
    ("03_hmc_nuts", 1, (
        ("hmc_10", "Metropolis HMC\n$T=10$"),
        ("hmc_uniform_19", "Metropolis HMC\n$T\\sim\\mathrm{Unif}\\{1,\\ldots,19\\}$"),
        ("nuts", "NUTS\n"),
    )),
)


def _limits(values):
    low, high = float(values[0]), float(values[-1])
    span = high - low
    padding = .05 * span if span > 0 else .05 * max(abs(low), 1.)
    return low - padding, high + padding


def draw_panel(ax, values, title, letter):
    theoretical, empirical = normal_qq_coordinates(values)
    xlim, ylim = _limits(theoretical), _limits(empirical)
    ax.plot(xlim, xlim, color="#b23a36", linewidth=.85, zorder=1)
    ax.scatter(theoretical, empirical, s=3.0, color="#205c90", alpha=.75,
               edgecolors="none", zorder=2)
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_box_aspect(1)
    ax.set_title(f"({letter}) {title}", fontsize=9, pad=8, linespacing=1.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.tick_params(labelsize=8, length=3, width=.65, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(.65)
    return {"xlim": list(xlim), "ylim": list(ylim), "points": len(empirical)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("clt_qq/paper/data"))
    parser.add_argument("--output", type=Path, default=Path("clt_qq/paper/figures"))
    args = parser.parse_args(argv)
    config = json.loads((args.data / "config.json").read_text())
    signature = config["experiment_signature"]
    if (signature["df"] != 1.0 or signature["observable"] != "tail"
            or config["tail_threshold"] != 2.0 or config["step_size"] != .35):
        raise ValueError("These manuscript captions require the Cauchy tail at 2 and step size 0.35")
    source = args.data / "means_t1_tail.npz"
    with np.load(source) as archive:
        means = {key: archive[key].copy() for key in archive.files}
    count = len(means["chain_ids"])
    iterations, burn_in = int(means["iterations"]), int(means["burn_in"])
    if (count != config["chains"] or iterations != config["iterations"]
            or burn_in != config["burn_in"]):
        raise ValueError("Saved chain metadata disagrees with the experiment configuration")
    for _, _, panels in FIGURES:
        for key, _ in panels:
            values = means[key]
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid chain averages for {key}")
            if np.any((values < 0) | (values > 1)):
                raise ValueError(f"Tail indicator average outside [0, 1]: {key}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "target": "Student t(1), standard Cauchy",
        "observable": "1{x >= 2}",
        "chains": count, "iterations": iterations, "burn_in": burn_in,
        "retained": iterations - burn_in,
        "qq": "Raw ordered means versus fitted Gaussian; identity line; all points retained",
        "axes": "Independent x and y limits in each square panel",
        "figures": {},
    }
    style = {
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": True,
    }
    with plt.rc_context(style):
        for name, rows, panels in FIGURES:
            fig, axes = plt.subplots(rows, 3, figsize=(7., 2.9 if rows == 1 else 5.25),
                                     squeeze=False)
            fig.subplots_adjust(left=.09, right=.985, bottom=.24 if rows == 1 else .14,
                                top=.77 if rows == 1 else .85, wspace=.42, hspace=.72)
            details = []
            for i, (ax, (key, title)) in enumerate(zip(axes.flat, panels)):
                detail = draw_panel(ax, means[key], title, chr(ord("a") + i))
                details.append({"panel": chr(ord("a") + i), "method": key, **detail})
            heading = r"Cauchy target, $g(x)=\mathbf{1}_{\{x\geq 2\}}$"
            if rows == 2:
                heading += " - Metropolis HMC"
            fig.suptitle(heading, y=.99, fontsize=10)
            fig.supxlabel("Fitted normal quantiles", fontsize=9, y=.09 if rows == 1 else .055)
            fig.supylabel("Ergodic averages", fontsize=9, x=.012, y=.51)
            fig.text(.54, .023 if rows == 1 else .016,
                     f"{count:,} independent chains; {iterations - burn_in:,} retained iterations per chain",
                     ha="center", fontsize=7.5)
            for suffix in ("pdf", "png"):
                options = {"dpi": 400} if suffix == "png" else {
                    "metadata": {"Title": name.replace("_", " "),
                                 "Creator": "clt_qq.paper_figures", "CreationDate": None,
                                 "ModDate": None}}
                fig.savefig(args.output / f"{name}.{suffix}", **options)
            plt.close(fig)
            manifest["figures"][name] = details
            print(f"Saved {name}: {len(panels)} panels, {count:,} points per panel")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
