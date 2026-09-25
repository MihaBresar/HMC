"""Small, one-dimensional slice NUTS implementation for a Student-t target.

This is Algorithm 3 of Hoffman and Gelman (2014), with a fixed step size and
a maximum tree depth. There is no warm-up adaptation or multinomial sampling.
Reference: https://jmlr.org/papers/v15/hoffman14a.html
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


_MAX_SLICE_ENERGY_ERROR = 1000.0


def _potential(x: float, df: float) -> float:
    """Student-t negative log density, omitting its normalizing constant."""
    return 0.5 * (df + 1.0) * math.log1p(x * x / df)


def _gradient(x: float, df: float) -> float:
    return (df + 1.0) * x / (df + x * x)


def _leapfrog(x: float, p: float, epsilon: float, df: float) -> tuple[float, float]:
    """One reversible leapfrog step; negative epsilon integrates backwards."""
    p_half = p - 0.5 * epsilon * _gradient(x, df)
    x_new = x + epsilon * p_half
    p_new = p_half - 0.5 * epsilon * _gradient(x_new, df)
    return x_new, p_new


def _no_u_turn(x_left: float, p_left: float, x_right: float, p_right: float) -> bool:
    displacement = x_right - x_left
    return displacement * p_left >= 0.0 and displacement * p_right >= 0.0


class _Tree(NamedTuple):
    x_left: float
    p_left: float
    x_right: float
    p_right: float
    proposal: float
    n_valid: int
    keep_growing: bool
    n_steps: int
    divergent: bool


def _build_tree(
    x: float,
    p: float,
    log_slice: float,
    direction: int,
    depth: int,
    epsilon: float,
    df: float,
    rng: np.random.Generator,
) -> _Tree:
    if depth == 0:
        x_new, p_new = _leapfrog(x, p, direction * epsilon, df)
        log_joint = -_potential(x_new, df) - 0.5 * p_new * p_new
        finite = math.isfinite(x_new) and math.isfinite(p_new) and math.isfinite(log_joint)
        n_valid = int(finite and log_slice <= log_joint)
        safe = finite and log_joint > log_slice - _MAX_SLICE_ENERGY_ERROR
        return _Tree(x_new, p_new, x_new, p_new, x_new, n_valid, safe, 1, not safe)

    first = _build_tree(x, p, log_slice, direction, depth - 1, epsilon, df, rng)
    if not first.keep_growing:
        return first

    if direction == -1:
        second = _build_tree(
            first.x_left, first.p_left, log_slice, direction, depth - 1, epsilon, df, rng
        )
        x_left, p_left = second.x_left, second.p_left
        x_right, p_right = first.x_right, first.p_right
    else:
        second = _build_tree(
            first.x_right, first.p_right, log_slice, direction, depth - 1, epsilon, df, rng
        )
        x_left, p_left = first.x_left, first.p_left
        x_right, p_right = second.x_right, second.p_right

    # Sample uniformly among the slice-valid points in the combined subtree.
    n_valid = first.n_valid + second.n_valid
    proposal = first.proposal
    if n_valid and rng.random() < second.n_valid / n_valid:
        proposal = second.proposal
    keep_growing = second.keep_growing and _no_u_turn(x_left, p_left, x_right, p_right)
    return _Tree(
        x_left, p_left, x_right, p_right, proposal, n_valid, keep_growing,
        first.n_steps + second.n_steps, first.divergent or second.divergent,
    )


def nuts_step(
    x: float,
    df: float,
    step_size: float,
    max_depth: int,
    rng: np.random.Generator,
) -> tuple[float, int, bool, bool]:
    """Take one NUTS transition with fully refreshed standard-normal momentum.

    Return ``(position, leapfrog_steps, hit_depth_cap, divergent)``. At most
    ``2**max_depth - 1`` leapfrog steps are used. ``hit_depth_cap`` means the
    tree was still eligible to grow when the cap was reached; reaching that
    depth simultaneously with a U-turn is not counted as hitting the cap.

    ``divergent`` flags any nonfinite integration result or violation of the
    original paper's slice safety check: H_new >= -log(slice) + 1000. This is
    a one-sided error relative to the slice, not an absolute energy error.
    A subtree that stopped internally cannot supply the returned proposal.

    The target is the standard (location zero, scale one) Student-t law with
    positive ``df``. A finite depth cap limits travel even far into the tails.
    """
    if not math.isfinite(x):
        raise ValueError("x must be finite")
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError("df must be finite and positive")
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    if isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer)) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")

    x = float(x)
    p = float(rng.normal())
    log_joint = -_potential(x, df) - 0.5 * p * p
    if not math.isfinite(log_joint):
        raise ValueError("initial state has a nonfinite Hamiltonian")
    # If E ~ Exponential(1), exp(-E) is Uniform(0, 1).
    log_slice = log_joint - float(rng.exponential())
    x_left = x_right = proposal = x
    p_left = p_right = p
    n_valid = 1  # The initial state is always in the slice.
    keep_growing = True
    n_steps = 0
    divergent = False
    depth = 0

    while keep_growing and depth < max_depth:
        direction = -1 if rng.random() < 0.5 else 1
        if direction == -1:
            tree = _build_tree(x_left, p_left, log_slice, direction, depth, step_size, df, rng)
            x_left, p_left = tree.x_left, tree.p_left
        else:
            tree = _build_tree(x_right, p_right, log_slice, direction, depth, step_size, df, rng)
            x_right, p_right = tree.x_right, tree.p_right

        # Algorithm 3 uses n_new / n_old HERE, not n_new / (n_old + n_new).
        # The latter ratio is used only inside _build_tree above.
        if tree.keep_growing and rng.random() < min(1.0, tree.n_valid / n_valid):
            proposal = tree.proposal
        n_valid += tree.n_valid
        keep_growing = tree.keep_growing and _no_u_turn(x_left, p_left, x_right, p_right)
        n_steps += tree.n_steps
        divergent = divergent or tree.divergent
        depth += 1

    hit_depth_cap = bool(keep_growing and depth == max_depth)
    return proposal, n_steps, hit_depth_cap, divergent
