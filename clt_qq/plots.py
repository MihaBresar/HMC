"""Simple normal QQ plots: one unscaled ergodic average per chain."""

import math
from pathlib import Path

import numpy as np
from scipy.stats import norm


def normal_qq_coordinates(values):
    """Compare raw averages with a normal fitted to their mean and sample SD.

    Both coordinates stay in the observable's original units. In particular,
    this function neither centers at the target expectation nor multiplies by
    the square root of the number of iterations.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("QQ plots require a one-dimensional array of at least two averages.")
    if not np.isfinite(values).all():
        raise ValueError("QQ plots require finite averages.")
    probabilities = (np.arange(values.size) + 0.5) / values.size
    theoretical = values.mean() + values.std(ddof=1) * norm.ppf(probabilities)
    return theoretical, np.sort(values)


def _draw_qq(ax, values, title):
    from matplotlib.ticker import MaxNLocator

    theoretical, empirical = normal_qq_coordinates(values)
    def padded_limits(values):
        low, high = values[0], values[-1]
        width = high - low
        padding = 0.05 * width if width > 0 else 0.05 * max(abs(low), 1.0)
        return low - padding, high + padding

    # Each axis follows its own quantiles. A large observed average should
    # expand the vertical axis without adding empty horizontal space.
    x_limits = padded_limits(theoretical)
    y_limits = padded_limits(empirical)
    ax.plot(x_limits, x_limits, color="#b83b3b", linewidth=1.2, zorder=1)
    ax.scatter(theoretical, empirical, s=9, color="#2463a6", alpha=0.7,
               edgecolors="none", zorder=2)
    ax.set(xlim=x_limits, ylim=y_limits, title=title,
           xlabel="Fitted normal quantiles", ylabel="Ergodic averages")
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def _save_figure(fig, path):
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=180)


def save_plots(case, methods, means, args):
    """Save a sampler comparison and one standalone QQ plot for each sampler."""
    # Workers that simulate chains need not import or initialize Matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not methods:
        raise ValueError("At least one method is required for plotting.")
    output = Path(args.output)
    individual = output / "individual"
    individual.mkdir(parents=True, exist_ok=True)
    case_title = case.title(args.tail_threshold)
    run_title = (f"{args.chains:,} chains · {args.iterations:,} iterations · "
                 f"{args.burn_in:,} burn-in")

    columns = min(3, len(methods))
    rows = math.ceil(len(methods) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 4.1 * rows),
                             squeeze=False)
    for ax, method in zip(axes.flat, methods):
        _draw_qq(ax, means[method.key], method.label)
    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    fig.suptitle(f"{case_title}\n{run_title}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / (4.1 * rows)))
    _save_figure(fig, output / f"qq_{case.key}")
    plt.close(fig)

    for method in methods:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        _draw_qq(ax, means[method.key], method.label)
        fig.suptitle(f"{case_title}\n{run_title}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _save_figure(fig, individual / f"qq_{case.key}_{method.key}")
        plt.close(fig)
