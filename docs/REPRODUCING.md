# Reproducing the paper figures

Run every command below from the repository root. The three manuscript figures
use the archived averages in `clt_qq/paper/data_80k/`; generating the figures
does not require running the simulations again.

## Environment

Use Python 3.12 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. The requirements
pin NumPy 2.4.0, SciPy 1.16.3, Matplotlib 3.10.8, JAX/JAXlib 0.11.2,
BlackJAX 1.6.2 and Optax 0.2.8. The archived simulation configurations record
Python 3.14.2. The reference runs used CPU execution and double precision;
the NUTS backend explicitly enables JAX float64. To select CPU execution when
other JAX devices are installed, set `JAX_PLATFORMS=cpu` before running Python.
No external LaTeX installation is needed to draw the figures.

## Regenerate figures from archived averages

```bash
python -m clt_qq.verify_paper
python -m clt_qq.paper_figures --output clt_qq/output/paper_figures
```

The first command checks the bundled paper data and figure manifest. The second
writes three PDFs, three PNGs and a manifest to a separate output directory:

| Basename | Comparison |
| --- | --- |
| `01_ula_fixed_random` | Student t(3), absolute moment: ULA and unadjusted HMC with fixed or uniform trajectory length |
| `02_randomisation_and_length` | Cauchy tail indicator: adjusted HMC with fixed, uniform and two-point lengths at two mean lengths |
| `03_hmc_nuts` | Cauchy tail indicator: adjusted fixed/random HMC and BlackJAX NUTS |

To build only one figure, append, for example,
`--figures 03_hmc_nuts`. LaTeX figure environments and captions are supplied in
[`clt_qq/paper/latex/`](../clt_qq/paper/latex/).

The plotter reads saved averages only. It neither reruns a sampler nor removes
extreme observations. PDF metadata and rendering can vary across environments;
the underlying averages and QQ coordinates are the scientific reference.

## Run all paper simulations afresh

These commands write to new output folders, leaving the archived paper data
intact. They use 2,000 independent chains per method, each with 120,000
transitions: 40,000 discarded and 80,000 retained.

```bash
python -m clt_qq.paper_experiment --case t3_abs \
  --chains 2000 --iterations 120000 --burn-in 40000 \
  --step-size 0.35 --tail-threshold 2 --seed 20260925 \
  --workers 6 --batch-size 128 --nuts-batch-size 4 \
  --output clt_qq/output/paper_reproduction/t3_abs

python -m clt_qq.paper_experiment --case t1_tail \
  --chains 2000 --iterations 120000 --burn-in 40000 \
  --step-size 0.35 --tail-threshold 2 --seed 20260925 \
  --workers 6 --batch-size 128 --nuts-batch-size 4 \
  --output clt_qq/output/paper_reproduction/t1_tail

python -m clt_qq.paper_figures \
  --data clt_qq/output/paper_reproduction \
  --output clt_qq/output/paper_reproduction/figures
```

The first case runs three methods; the second runs nine. See
[`DATA.md`](DATA.md) for the exact method keys and trajectory laws. Every chain
starts at zero. Mass is one, and the step size stays fixed throughout each
chain; burn-in does not adapt parameters. NUTS uses the standard BlackJAX kernel
with its library defaults for trajectory construction, turning, selection and
divergence handling. No maximum-depth override is passed.

Six worker processes handle separate batches of independent chains. NumPy
samplers use batches of 128; NUTS uses batches of four. A chain has its own
state and accumulated average. The published Cauchy data were generated in
separate six-worker pools for NUTS and the other HMC settings; the sequential
commands above use the same per-method random streams. Runtime depends on
hardware, process overhead and JAX compilation.

Repeating a command reuses compatible completed methods from its output folder.
If all requested methods are already present, it makes no file changes and
preserves the original execution metadata. When adding methods, their new
environment is recorded in `execution_history` and method provenance.
Results are saved after each complete method, not after individual batches.
Use `--methods hmc_10 nuts`, for example, to run a subset and later add the
remaining methods with the same settings. `--recompute` discards the selected
case's saved results in that output directory and runs the requested methods
again. Existing incompatible or incomplete result folders are refused unless
`--recompute` is supplied explicitly. Use a different output directory when
changing the experimental design to preserve earlier results.

Worker scheduling and worker count do not change the chain streams. Keep
`--batch-size 128` when reproducing NumPy results: changing that partition
changes their streams. NUTS streams are also stable across NUTS batch sizes.
Pinned packages and settings improve repeatability, but bit-for-bit simulation
agreement across CPU architectures and numerical libraries is not guaranteed.

The verifier checks a complete paper bundle with `data_80k/<case>/`, `figures/`
and `latex/` subdirectories. To check a copied bundle at another location, use:

```bash
python -m clt_qq.verify_paper --paper-dir path/to/copied/paper
```

The fresh simulation output layout above contains only datasets and regenerated
figures and is not itself a complete paper bundle.

## Small runs and tests

For a small end-to-end run that generates simple QQ plots:

```bash
python -m clt_qq.experiment --cases t3_abs t1_tail \
  --methods iid ula hmc_10 nuts --chains 16 --iterations 300 --burn-in 100 \
  --workers 2 --batch-size 4 --output clt_qq/output/smoke
```

This is an execution check, not a statistically informative experiment. The
general experiment runner makes its own plots and has different default
budgets and method keys from the paper runner. The paper plotter deliberately
rejects small datasets because its captions specify the full paper protocol.

Run the automated checks separately:

```bash
python -m unittest discover -s clt_qq -t . -v
python -m clt_qq.verify_paper
```

Tests cover sampler updates, averaging and burn-in, QQ coordinates, process
reproducibility, and safe reuse of compatible paper data. Neither these checks
nor a finite-run QQ plot prove an asymptotic central limit theorem or its failure.
