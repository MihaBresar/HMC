# Simple QQ results

Every panel contains **2,000 independent chain averages**. Each chain starts
at zero, runs for 30,000 iterations, discards its first 10,000, and averages the
remaining 20,000 observations. Six worker processes are used for every sampler.

NUTS results use **BlackJAX 1.6.2 / JAX 0.11.2**, with the standard kernel and
library defaults for trajectory construction. Step size is 0.35 and mass is 1,
with no parameter adaptation. These replace the earlier handwritten NUTS results.

The QQ axes are raw ergodic averages and quantiles of the Gaussian fitted to
their empirical mean and standard deviation. There is no square-root-n scaling.

- [Student t(3), absolute value](qq_t3_abs.png)
- [Student t(1), probability of X >= 2](qq_t1_tail.png)
- [Student t(1.5), probability of X >= 2](qq_t1p5_tail.png)
- [One plot per sampler](individual/README.md)

Inspect the numbered chain averages in [t(3) CSV](chain_means_t3_abs.csv),
[t(1) CSV](chain_means_t1_tail.csv), and [t(1.5) CSV](chain_means_t1p5_tail.csv).
The NPZ files contain the same means plus run metadata. [Diagnostics](diagnostics.json)
record completed chains, worker PIDs, acceptance, NUTS cap hits, and gradient
cost; [config.json](config.json) contains all settings.

From the repository root:

```bash
python -m clt_qq.experiment --workers 6 --output clt_qq/examples/simple
```

See the [experiment README](../../README.md) for the paper reference and
normalization convention. This follows the paper's design at a smaller scale.
