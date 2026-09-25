# Simple QQ plots of independent ergodic averages

ULA, fixed/random-step HMC (adjusted and unadjusted), and NUTS, with an i.i.d.
reference. **One independent chain produces one ergodic average and one QQ
point.** The plots use the original average scale, with no multiplication or
division by `sqrt(n)`.

```bash
python -m pip install -r clt_qq/requirements.txt
python -m clt_qq.experiment --workers 6
```

Defaults: **2,000 chains per sampler and target, 30,000 iterations per chain,
10,000 discarded, 20,000 retained**. Output goes to `clt_qq/output/simple/`.
Python 3.12+ is required by the pinned JAX version. Dependencies include NumPy,
SciPy, Matplotlib, JAX and BlackJAX.

## View the new results

For the manuscript, see the [three comparison figures](paper/README.md), with
80,000 retained iterations per chain: t(3) absolute moments first, then Cauchy
tail probabilities. Vector PDFs, PNGs, raw averages and LaTeX captions are included.

- [Student t(3), absolute value](examples/simple/qq_t3_abs.png)
- [Student t(1), tail probability](examples/simple/qq_t1_tail.png)
- [Student t(1.5), tail probability](examples/simple/qq_t1p5_tail.png)
- [Individual sampler plots](examples/simple/individual/)
- [Raw averages, settings and diagnostics](examples/simple/)

Reproduce these examples:

```bash
python -m clt_qq.experiment --workers 6 --output clt_qq/examples/simple
```

The earlier 128-chain, square-root-n figures in the parent `examples/`
directory are historical results, not the current defaults.

## Simulation design

Following the design in [Appendix B of Brešar, Mijatović and Roberts](https://arxiv.org/html/2512.18255v1#A2):

1. Start every chain at `X_0 = 0`, with independent random inputs.
2. Simulate `n` total transitions.
3. Discard `b = floor(n/3)` transitions from every chain.
4. Compute one average per chain, `A_i = sum(g(X_k), k=b+1,...,n)/(n-b)`.
5. Plot the ordered averages against a Gaussian fitted to all these averages.

For `N` chains, the exact QQ coordinates are

```text
y_i = sorted(A)[i]
x_i = mean(A) + std(A, ddof=1) * NormalQuantile((i + 0.5) / N)
```

Both axes are in the observable's original units; the reference is `y=x`.
The axes use independent ranges so extreme observed averages do not compress
the normal quantiles horizontally. Consequently, `y=x` need not look like a
45-degree line on the page.
There is one cloud of points per panel, no sample-size overlays, and no target
centering. All points are kept, including extremes. Each sampler also gets a
standalone PNG and PDF.

**Normalization detail:** `std(A)` already measures the spread of the chain
averages. It must not be divided by `sqrt(n)` again. Appendix B's displayed
sample-variance definition and subsequent extra division by `n` appear
inconsistent; this implementation follows its stated comparison with a
Gaussian fitted to the averages on their original scale.

This is the paper's simulation design at a smaller computational scale. Its
Figure 1 uses 10,000 chains of 200 million iterations for the Cauchy example;
its t(3) absolute-value example uses 50,000 chains of 100 million iterations.
The bundled run is not a reproduction of those full computational budgets.

## Cases

| Target | Observable | Exact expectation |
| --- | --- | --- |
| Student t(3) | `abs(x)` | `2*sqrt(3)/pi` |
| Student t(1), Cauchy | `x >= 2` | `0.5 - atan(2)/pi` |
| Student t(1.5) | `x >= 2` | `scipy.stats.t.sf(2, 1.5)` |

The tail event is now **one-sided**, as in the paper's Figure 1. Cauchy has no
finite first absolute moment; t(1.5) has a finite first absolute moment but
infinite variance. The indicators are bounded, and `abs(X)` under t(3) has
finite variance. All three i.i.d. reference averages therefore satisfy a CLT.

These QQ plots assess the Gaussian shape of finite-run averages. They alone
do not establish the asymptotic CLT or its failure, including for NUTS. Fitting
the Gaussian mean also removes location differences, including discretization
bias in ULA/unadjusted HMC; it does not show that a sampler has the correct
invariant mean.

## Are the chains really parallel and independent?

Yes. With `--workers 6`, **all samplers** run across up to six Python worker
processes. ULA and HMC split the chains into fixed batches of 128. Inside each
batch, a NumPy vector has a separate position and accumulator for every
chain, with independent noise, momenta and random leapfrog counts for its
coordinates. Batches use separate random streams. NUTS uses an independent JAX
random key for every numbered chain and runs compiled batches of up to 16 chains
per worker. These small batches limit the waiting caused by different adaptive
trajectory lengths.

Chains never share their positions or averages, and one long chain is never
split into artificial replicates. Worker scheduling does not affect the
results. Changing worker count leaves the raw averages unchanged; changing
NumPy vector batch size can change the random streams. NUTS streams are stable
across batch sizes too. With fewer chains than the
batch size, a vectorized sampler uses only one batch/worker.

`diagnostics.json` records the actual worker process IDs, completed chain
counts, and total/burn-in/retained iterations. `chain_means_<case>.csv` has
one row per numbered chain, making the averages directly inspectable.

## Samplers and options

All use fixed leapfrog epsilon `0.35` and unit-mass Gaussian momentum.

- ULA: `x <- x - epsilon**2/2 * U'(x) + epsilon*Z`.
- Fixed HMC: `L=5` or `L=10`, with or without the Metropolis correction.
- Random HMC: independently draw integer `L` uniformly from 1 through 10 on
  every transition, with or without the Metropolis correction.
- NUTS: the standard [BlackJAX NUTS kernel](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/nuts/index.html),
  using iterative multinomial sampling and the library's default trajectory
  limit, turning criterion, integrator and divergence threshold. The constructor
  is simply `blackjax.nuts(logdensity, step_size, inverse_mass_matrix)`.
  No maximum-depth argument is passed or exposed by the experiment. The installed
  versions and actual library default are recorded in `config.json` for
  reproducibility; the plot label is just **NUTS**. Cap-hit and divergence rates
  remain in the diagnostics.

Step size and unit mass stay fixed throughout each chain, including NUTS, so
the independent averages refer to the same kernel. Discarding the first third
is burn-in, not parameter adaptation. This is BlackJAX 1.6.2 with JAX 0.11.2 in
double precision; the earlier handwritten slice sampler has been replaced.

Only basic methods and one target:

```bash
python -m clt_qq.experiment --cases t3_abs --methods iid ula hmc_5 hmc_10 hmc_random nuts
```

More chains and longer paths:

```bash
python -m clt_qq.experiment --chains 10000 --iterations 300000 --workers 6 --output clt_qq/output/larger
```

A quick smoke run:

```bash
python -m clt_qq.experiment --chains 64 --iterations 300 --batch-size 16 --workers 2
```

Use `--burn-in` to override the first-third rule, `--fixed-steps 1 5 20` to
include MALA and other fixed lengths, `--random-max`, `--step-size`,
`--tail-threshold`, and `--seed` to change the experiment.
`--replicates` is an alias for `--chains`; the former `--lengths` interface has
been replaced by one `--iterations` value to keep the QQ plots simple.

The comparisons use equal iterations, not equal gradient work. Retained-phase
gradient cost is reported: 1 per ULA step, L+1 for HMC, and 1 new gradient per
BlackJAX leapfrog step, whose default integrator caches the current gradient.
Initial gradients and burn-in are excluded. NUTS reports the mean integration
acceptance probability separately from HMC's accepted-proposal fraction.

## Data and checks

The output contains comparison and individual plots, one CSV and compressed
NPZ of chain means per target, `summary.csv` with the fitted means and standard
deviations, `config.json`, and `diagnostics.json`. Raw averages are saved after
every completed method. Use a fresh directory for a new run; automatic resume
is not implemented. Old scaling figures are not produced by this version.

```bash
python -m unittest discover -s clt_qq -t . -v
```

Checks cover sampler correctness, the burn-in/average calculation, fitted-normal
QQ coordinates, and reproducibility across serial and process-parallel runs.
