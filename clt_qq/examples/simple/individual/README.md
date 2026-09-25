# Individual simple QQ plots

Each plot shows 2,000 raw chain averages and a fitted-normal identity line.

| Sampler | t(3), absolute value | t(1), tail | t(1.5), tail |
| --- | --- | --- | --- |
| i.i.d. reference | [View](qq_t3_abs_iid.png) | [View](qq_t1_tail_iid.png) | [View](qq_t1p5_tail_iid.png) |
| ULA | [View](qq_t3_abs_ula.png) | [View](qq_t1_tail_ula.png) | [View](qq_t1p5_tail_ula.png) |
| Unadjusted HMC, L=5 | [View](qq_t3_abs_uhmc_5.png) | [View](qq_t1_tail_uhmc_5.png) | [View](qq_t1p5_tail_uhmc_5.png) |
| Unadjusted HMC, L=10 | [View](qq_t3_abs_uhmc_10.png) | [View](qq_t1_tail_uhmc_10.png) | [View](qq_t1p5_tail_uhmc_10.png) |
| Unadjusted HMC, L~Unif[1,10] | [View](qq_t3_abs_uhmc_random.png) | [View](qq_t1_tail_uhmc_random.png) | [View](qq_t1p5_tail_uhmc_random.png) |
| MH-HMC, L=5 | [View](qq_t3_abs_hmc_5.png) | [View](qq_t1_tail_hmc_5.png) | [View](qq_t1p5_tail_hmc_5.png) |
| MH-HMC, L=10 | [View](qq_t3_abs_hmc_10.png) | [View](qq_t1_tail_hmc_10.png) | [View](qq_t1p5_tail_hmc_10.png) |
| MH-HMC, L~Unif[1,10] | [View](qq_t3_abs_hmc_random.png) | [View](qq_t1_tail_hmc_random.png) | [View](qq_t1p5_tail_hmc_random.png) |
| NUTS, max depth=7 | [View](qq_t3_abs_nuts.png) | [View](qq_t1_tail_nuts.png) | [View](qq_t1p5_tail_nuts.png) |
