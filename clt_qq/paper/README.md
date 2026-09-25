# Three figures for the HMC paper

The first figure uses Student t(3) and the absolute-moment observable `g(x)=abs(x)`.
The second and third use the standard Cauchy target and `g(x)=1{x >= 2}`.
Each QQ point is one average from an independent chain.
Every chain starts at zero, runs for 120,000 transitions, discards the first
40,000, and averages the remaining **80,000 observations**. There are 2,000 chains
per sampler setting, with six worker processes used in the saved experiments.

| Figure | Comparison | PNG | Vector PDF | LaTeX caption |
| --- | --- | --- | --- | --- |
| 1 | t(3) absolute moment: ULA, fixed unadjusted HMC, randomised unadjusted HMC | [Image](figures/01_ula_fixed_random.png) | [PDF](figures/01_ula_fixed_random.pdf) | [TeX](latex/01_ula_fixed_random.tex) |
| 2 | Three step-count laws at two mean lengths, all Metropolis-adjusted HMC | [Image](figures/02_randomisation_and_length.png) | [PDF](figures/02_randomisation_and_length.pdf) | [TeX](latex/02_randomisation_and_length.tex) |
| 3 | Fixed Metropolis HMC, randomised Metropolis HMC, standard NUTS | [Image](figures/03_hmc_nuts.png) | [PDF](figures/03_hmc_nuts.pdf) | [TeX](latex/03_hmc_nuts.tex) |

Figure 1 follows the Langevin-to-HMC comparison without Metropolis correction:
ULA is the one-leapfrog position update. The two HMC panels use `T=10` and
discrete uniform `T` on `1,...,19`, respectively. Figure 3 uses the corresponding
Metropolis-adjusted HMC kernels alongside BlackJAX NUTS, so all three have the
same intended invariant distribution.

The first example has `E|X|=2*sqrt(3)/pi` and `E[X^2]=3` under the Student t(3)
target, hence the observable has finite variance. Its Gaussian QQ comparison
illustrates dependent moment averages in a setting where independent target
draws have the usual finite-variance CLT. The unadjusted kernels can have
discretisation bias; the reference Gaussian is fitted to their actual averages.

Figure 2 separates changing the mean length from changing its distribution:

| Row | Fixed | Discrete uniform | Equiprobable two-point |
| --- | --- | --- | --- |
| Mean T = 5 | T = 5 | T in 1,...,9 | T = 1 or 9 |
| Mean T = 20 | T = 20 | T in 1,...,39 | T = 1 or 39 |

The expected number of leapfrog steps is matched within each row. Its second
moment differs between the laws: matching expected computational work does not
match the leading tail-drift coefficient. All random step counts are redrawn
independently of position and momentum at every transition. The choices above
were specified before running the additional simulations.

All settings use leapfrog step size `h=0.35` and unit-mass Gaussian momentum.
In ULA's conventional notation, the time step is `h^2/2 = 0.06125`. NUTS uses
BlackJAX 1.6.2 / JAX 0.11.2, its default trajectory rules and limit, fixed tuning,
and double precision. No maximum-depth override or adaptation is performed.

## Reading the QQ plots

For the vector `A` of raw chain averages, each plot uses

```text
x_i = mean(A) + std(A, ddof=1) * NormalQuantile((i + 0.5) / 2000)
y_i = sorted(A)[i]
```

The red reference is `y=x`. There is no square-root-n scaling, target centring,
clipping, trimming, or outlier removal. Each axis is fitted independently to
its own quantiles, so the identity line need not appear at 45 degrees. The
same number of transitions is used for every chain, rather than the same
amount of gradient work; actual retained-phase costs are recorded.

The figures illustrate the finite-run distribution of the averages. They do
not measure polynomial convergence exponents or prove CLT failure. In particular,
Figure 3 is an empirical comparison accompanying a NUTS discussion, not a proof
that the independent-randomisation theorem applies to NUTS. Fitting the normal
location also removes invariant-mean differences, including discretisation bias
in the unadjusted methods.

## Reproduce and insert

From the repository root:

```bash
python -m pip install -r clt_qq/requirements.txt
python -m clt_qq.paper_experiment --case t3_abs
python -m clt_qq.paper_experiment --case t1_tail --nuts-batch-size 4
python -m clt_qq.paper_figures
```

The simulation commands re-use only matching case/protocol results and simulate
missing settings; the old 20,000-retained results are incompatible with this run.
Use `--recompute` to rerun the selected case. The NUTS batch size changes only
parallel scheduling, with independent random streams indexed by chain ID.
The plotting command only reads saved averages. Raw NPZ/CSV data, settings,
per-method provenance, worker IDs and diagnostics are in
[t(3) data](data_80k/t3_abs/) and [Cauchy data](data_80k/t1_tail/).
[The figure manifest](figures/manifest.json) records panel membership, axis
limits and SHA-256 hashes of both source NPZ files. The earlier all-Cauchy,
20,000-retained data remain in [data](data/) as historical results.

Copy the `figures` directory into the manuscript directory and input the three
LaTeX snippets at the relevant discussion points: after the initial CLT
discussion, after the examples of admissible independent randomisation, and
after the NUTS conjecture. The local placement note supplies exact anchors for
the current manuscript. The figures use `T` to match its leapfrog-count notation.

PDFs have vector points/lines and embedded fonts; PNGs are exported at 400 dpi.
