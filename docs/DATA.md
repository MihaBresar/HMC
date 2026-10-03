# Paper data and interpretation

The reference dataset is [`clt_qq/paper/data_80k/`](../clt_qq/paper/data_80k/).
It contains 2,000 independent chain averages for each of 12 target/method
settings. Each chain starts at zero and runs 120,000 transitions, discarding
40,000 and retaining 80,000. The seed is `20260925`, the leapfrog step size is
`0.35`, and the mass is one. The saved averages are the input to the three
[`paper figures`](../clt_qq/paper/figures/).

## Targets and methods

The Student target has potential
`U(x) = (nu + 1)/2 * log(1 + x*x/nu)`, up to an additive constant.
`T` below is the integer number of leapfrog steps, so integration time is
`0.35*T`. Momentum is refreshed independently from a standard normal at each
transition. Random trajectory counts are independently redrawn at every
transition, independently of the current state and momentum.

| Case | Target | Observable | Target expectation |
| --- | --- | --- | --- |
| `t3_abs` | Student t, 3 degrees of freedom | `abs(x)` | `2*sqrt(3)/pi`, approximately 1.1026577908435842 |
| `t1_tail` | Standard Cauchy, 1 degree of freedom | `1{x >= 2}` | `0.5 - atan(2)/pi`, approximately 0.1475836176504332 |

| Case | Saved method key | Kernel / leapfrog law |
| --- | --- | --- |
| `t3_abs` | `ula` | ULA |
| `t3_abs` | `uhmc_10` | Unadjusted HMC, `T=10` |
| `t3_abs` | `uhmc_uniform_19` | Unadjusted HMC, uniform on integers 1 through 19 |
| `t1_tail` | `hmc_5`, `hmc_10`, `hmc_20` | Metropolis-adjusted HMC, `T=5`, `10`, `20` respectively |
| `t1_tail` | `hmc_uniform_9`, `hmc_uniform_19`, `hmc_uniform_39` | Metropolis-adjusted HMC, uniform on integers 1 through 9, 19, 39 respectively |
| `t1_tail` | `hmc_two_point_9`, `hmc_two_point_39` | Metropolis-adjusted HMC, equally likely counts `{1,9}` or `{1,39}` |
| `t1_tail` | `nuts` | Standard multinomial BlackJAX NUTS |

ULA uses `x' = x - 0.35**2/2 * U'(x) + 0.35*Z`, with independent standard
normal `Z`. Its conventional Langevin step is therefore `0.06125`; it is also
the position update of one unadjusted leapfrog step with refreshed momentum.
ULA and unadjusted HMC do not generally preserve the exact Student target.

NUTS uses BlackJAX 1.6.2 and JAX 0.11.2 in float64, with fixed step size and
unit mass and no adaptation. The constructor supplies no trajectory-depth
override. The recorded library default is `max_num_doublings=10`.

## Files and fields

Each case directory contains:

| File | Contents |
| --- | --- |
| `means_<case>.npz` | Compressed NumPy arrays: `chain_ids`, scalar `iterations`, scalar `burn_in`, scalar `target_mean`, and one length-2,000 array per method key |
| `chain_means_<case>.csv` | Same means as the NPZ, one row per `chain_id` from 0 to 1999, one column per method |
| `summary.csv` | One row per method: protocol counts, target expectation, mean of chain means, and sample standard deviation (`ddof=1`) of chain means |
| `config.json` | Target, protocol, seed, random-stream design, execution/library metadata, and an `experiment_signature` including each method's law and its first two leapfrog-count moments |
| `diagnostics.json` | Entries keyed by `<case>/<method>`: completed chains, actual worker process IDs/counts, batches, iteration counts, retained-phase costs and sampler diagnostics, elapsed time and provenance |

The common `chain_id` column identifies an independent replicate within each
method; it does not indicate common random numbers or paired simulations across
different methods. No full trajectories are archived. Each saved number is the
raw average of the observable over the retained portion of one chain.

The reference diagnostics record six actual worker processes for every method.
The NumPy settings used 16 batches per method (the last batch has 80 chains);
NUTS used 500 batches of four. Process IDs and wall times are historical
execution metadata, not inputs required to regenerate the chain streams.
Continued runs may include an `execution_history` in the configuration, with
method-specific execution metadata preserved in their provenance.

[`figures/manifest.json`](../clt_qq/paper/figures/manifest.json) records SHA-256
hashes of the source NPZ files, source targets, chain/iteration counts, and
each figure's case/method mapping, point count and axis bounds. It links the
figures to exact saved averages; it does not contain image-file hashes.
Run `python -m clt_qq.verify_paper` to check the archived data and manifest.

## QQ construction

For `N=2000` chain averages `A`, the plotted coordinates are:

```text
y[i] = sorted(A)[i]
x[i] = mean(A) + std(A, ddof=1) * NormalQuantile((i + 0.5)/N)
```

There is one point per independent chain. The Gaussian mean and variance are
fitted to all chain averages in that panel. No target centering, square-root-n
scaling, clipping or outlier removal is applied. The reference line is `y=x`.
Each square panel has independent horizontal and vertical limits; the reference
line consequently need not appear at 45 degrees. Different panels also have
different scales, so compare the numeric axes when assessing spread.

## Costs and scientific scope

All cost and acceptance diagnostics concern retained transitions. The reported
`force_evals_per_retained_draw` is one for ULA, `T+1` for the implemented HMC
integrator, and the integration-step count for BlackJAX NUTS, whose default
integrator caches the current gradient and evaluates one new gradient per step.
NUTS initialization and burn-in costs are excluded. These counters describe
the implementations here, not a hardware-independent wall-time comparison.

For adjusted fixed/random HMC, `acceptance_rate` is the accepted-proposal
fraction. For NUTS, `mean_integration_acceptance_probability` is BlackJAX's
mean integration acceptance statistic, not an accepted-proposal fraction.
`depth_cap_fraction` counts retained transitions that reach the library's
default expansion limit without a turning or divergence stop;
`divergence_fraction` is the fraction flagged divergent by BlackJAX.
Inapplicable fields are JSON `null`.

Comparisons use equal retained iteration counts. In the second figure, laws
within each row share `E[T]` (5 or 20) but have different `E[T^2]`. NUTS has
adaptive trajectory lengths and a different gradient cost. These are not
comparisons at equal gradient budgets.

The Student t(3) absolute-value observable has finite variance under its target;
the Cauchy tail indicator is bounded. Thus infinite marginal variance of the
observable is not the explanation for non-Gaussian chain averages in these
examples. A fitted-normal QQ plot assesses shape at this finite run length.
It does not establish an asymptotic CLT, CLT failure, or a polynomial convergence
rate, including for NUTS. Fitting the mean also hides location bias and does not
verify that an unadjusted kernel has the correct invariant mean. Burn-in from
zero does not guarantee stationarity.

## Historical outputs

- `clt_qq/paper/data/`: the earlier all-Cauchy paper experiment with 20,000
  retained iterations per chain.
- `clt_qq/examples/simple/`: earlier 2,000-chain comparisons with 20,000 retained
  iterations, including the Student t(1.5) tail indicator and an i.i.d. reference.
- Files directly in `clt_qq/examples/`: older 128-chain experiments with
  square-root-n scaling.

These folders are retained for provenance and additional examples. They are
not the sources of the current 80,000-retained-iteration paper figures. The
general runner still supports t(1.5), which has a finite first absolute moment
but infinite variance; the Cauchy target has no finite first absolute moment.
