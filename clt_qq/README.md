# Simple QQ plots for heavy-tailed Monte Carlo averages

Compare ULA, HMC with fixed or independently randomized leapfrog counts, and
NUTS on three one-dimensional Student-t examples. Both Metropolis-adjusted HMC
and unadjusted HMC are included. An exact i.i.d. reference shows what happens
without serial dependence. The original `SamlerComparison.py` is separate.

From the repository root, with Python 3.10 or newer:

```bash
python -m pip install -r clt_qq/requirements.txt
python -m clt_qq.experiment --workers 3
```

The default uses 128 independent replicates, each with 4,000 retained draws,
and also records its first 250 and 1,000 draws. Output goes to `clt_qq/output/`.
Use `--workers 1` to run NUTS without multiprocessing. No JAX, BlackJAX, or
probabilistic programming framework is required.

Generated examples with these settings (seed 20260925) are included:

- [t(3), absolute value](examples/qq_t3_abs.png)
- [t(1), tail probability](examples/qq_t1_tail.png)
- [t(1.5), tail probability](examples/qq_t1p5_tail.png)

The [examples directory](examples/) also contains the scaling figures, raw
replicate averages, and settings. Reproduce it with
`python -m clt_qq.experiment --workers 3 --output clt_qq/examples`.

## Cases

All targets have location zero and scale one. The tail event is **two-sided**.

| Case key | Target | Observable | Exact target expectation |
| --- | --- | --- | --- |
| `t3_abs` | Student t with 3 degrees of freedom | `abs(x)` | `2*sqrt(3)/pi` |
| `t1_tail` | Student t with 1 degree of freedom (Cauchy) | `abs(x) > 2` | `1 - 2*atan(2)/pi` |
| `t1p5_tail` | Student t with 1.5 degrees of freedom | `abs(x) > 2` | `2*scipy.stats.t.sf(2, 1.5)` |

The Cauchy distribution has no finite first absolute moment. The t(1.5)
distribution has a finite first absolute moment and infinite variance. We
estimate a bounded tail indicator for both, so its expectation always exists.
For t(3), `abs(X)` has finite variance `3 - 12/pi**2`. Thus **all three i.i.d.
reference averages have ordinary CLTs**. These examples investigate failure
caused by serial dependence, rather than an infinite observable variance.

Long excursions into the tails can give unusually large contributions to an
ergodic sum. For `abs(x)`, both excursion height and duration matter; for the
tail indicator, the duration matters even though the observable is bounded.
These are motivated by the CLT-failure examples and excursion analysis in
[Brešar, Mijatović and Roberts](https://arxiv.org/abs/2512.18255).

The plots are finite-run diagnostics, **not a proof of CLT failure for every
sampler shown**. In particular, independently randomized HMC theory does not
automatically cover NUTS's state-dependent stopping and proposal selection.
No universal failure claim for all bounded observables is intended. A nearly
straight QQ plot at one sample size also does not establish a CLT.

## What is plotted

Each QQ point comes from a separate independent chain. For each sample size
`n`, the ordinate is an empirical quantile across replicates of

```text
sqrt(n) * (mean(g(X_1), ..., g(X_n)) - exact_target_expectation).
```

The abscissa is a standard-normal quantile. Dashed lines pass through each
sample's first and third quartiles: they allow an unknown location and scale.
Curvature or extreme departures from these lines indicate non-Gaussian shape.
All points are shown, including outliers; each sampler panel has its own y-axis.
No estimated autocorrelation time or chainwise standard error is used to
studentize the averages. The saved raw averages permit other normalizations.

Colors compare nested prefix lengths within each chain. Replicates are
independent; the different lengths for one replicate are correlated.

The companion scaling plot shows the IQR of the square-root-n errors divided
by `1.349`, plus the grand-mean error against the target. Under a nondegenerate
Gaussian CLT, this IQR scale should eventually stabilize. Apparent widening
is evidence to investigate at larger n; neither this diagnostic nor an
unstable sample variance alone proves failure. With only 128 replicates,
rare excursions can noticeably change the figures across seeds.

**ULA and unadjusted HMC have discretization bias.** Their invariant laws
generally differ from the Student-t target. A vertical shift or increasing
target-centered error can be caused by that bias. Their CLT, if any, must be
centered at their own invariant mean, which this script does not claim to know.
Fitting the QQ reference line's location and using the translation-invariant
IQR help distinguish shape and width from this shift.

## Algorithms and tuning

The potential is `U(x) = (nu+1)/2 * log(1 + x*x/nu)`; the kinetic energy is
`p*p/2`. Every HMC/NUTS iteration refreshes an independent standard-normal
momentum. There is no parameter adaptation during sampling.

- **ULA:** `x <- x - epsilon**2/2 * U'(x) + epsilon*Z`. Its Langevin time step
  is `h = epsilon**2/2`; this matches the position update of one unadjusted
  leapfrog step.
- **Fixed HMC:** `L=5` and `L=10` by default, at `epsilon=0.35`. The unadjusted
  version always takes the endpoint; the MH version uses the Hamiltonian
  Metropolis correction. `L=1` adjusted HMC is MALA.
- **Random HMC:** draw `L` uniformly from the integers 1 through 10 on every
  iteration, independently of position, momentum and previous iterations.
  The leapfrog **count**, not the integrator step size, is randomized.
- **NUTS:** original efficient slice NUTS, Algorithm 3 of
  [Hoffman and Gelman (2014)](https://jmlr.org/papers/v15/hoffman14a.html), with
  fixed epsilon and depth capped at 7 (at most 127 leapfrog steps). This is
  not the newer multinomial variant used by some libraries. The cap changes
  how far NUTS can travel into the tails. Inspect its reported cap-hit rate
  and repeat with larger depth caps. Divergence reporting uses the original
  paper's one-sided slice safety threshold of 1000, or nonfinite integration.

Replicates start from independent exact Student-t draws. MH-HMC and NUTS
therefore start in stationarity. ULA and unadjusted HMC additionally discard
1,000 transitions; this is a configurable burn-in, **not a guarantee of
stationarity for their biased invariant laws**. For them, repeat with a longer
burn-in and a smaller epsilon if location/transient effects matter.

Sample sizes count retained iterations, not force evaluations. The diagnostics
record actual gradient calls per retained draw (1 for ULA, L+1 for HMC, 2 per
leapfrog for this NUTS implementation), excluding burn-in. These are not
comparisons at equal computational cost. Random `L~Uniform{1,...,10}` has mean
5.5 and is not cost-matched to both fixed choices.

## Change the experiment

A very small smoke run:

```bash
python -m clt_qq.experiment --replicates 16 --lengths 30 100 --unadjusted-burn-in 100
```

Only the requested basic methods and one target:

```bash
python -m clt_qq.experiment --cases t3_abs --methods iid ula hmc_5 hmc_10 hmc_random nuts
```

A more substantial run (NUTS can take considerably longer):

```bash
python -m clt_qq.experiment --replicates 1000 --lengths 1000 10000 100000 --workers 4 --output clt_qq/output/long
```

Change fixed counts and random range, add MALA, and change the tail threshold:

```bash
python -m clt_qq.experiment --fixed-steps 1 5 20 --random-max 20 --tail-threshold 3 --step-size 0.25 --max-depth 9
```

Use `--seed` to repeat with new random streams, `--unadjusted-burn-in` to
change burn-in, and `--help` for all options. Subsetting cases/methods or
changing worker count preserves each selected method's random stream.
Memory use scales with replicates and checkpoints, not total chain length.
The vectorized non-NUTS streams depend on the number of replicates.

## Output files and checks

- `qq_<case>.png` and `.pdf`: QQ comparisons at the requested sample sizes.
- `scaling_<case>.png` and `.pdf`: fluctuation width and target error versus n.
- `means_<case>.npz`: arrays of shape `(number_of_lengths, replicates)` keyed
  by method, plus `lengths` and `target_mean`.
- `summary.csv`: target expectations, grand means, errors, and empirical
  square-root-n IQR scales and standard deviations.
- `diagnostics.json`: retained-phase acceptance, cost, NUTS cap hits and
  divergences, burn-in counts, and run times.
- `config.json`: settings and environment metadata.

Each completed method's raw averages are saved before the next method starts.
Use a new output directory for a distinct run; there is no automatic resume.

Run numerical checks with:

```bash
python -m unittest discover -s clt_qq -t . -v
```

These test leapfrog reversibility, the ULA/one-step identity, exact
expectations, independent stationary transitions for MH-HMC and NUTS,
i.i.d. Bernoulli average variance, prefix reproducibility, NUTS depth and
divergence behavior, and agreement across NUTS worker counts.
