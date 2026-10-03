# Exact placement and captions for the three QQ figures

All panels use 2,000 independent chains, each started at zero and run for
120,000 transitions. The first 40,000 are discarded and each plotted point is
one average over the remaining **80,000 observations**. The first figure is
now the **first absolute moment, $g(x)=|x|$, for Student $t_3$**. Figures 2 and
3 retain the **standard Cauchy tail probability, $g(x)=\mathbf{1}_{\{x\geq2\}}$**.

The insertion points refer to the manuscript introduction reviewed on
25 September 2026. Section names and quoted anchors are more reliable than
line numbers if the manuscript has since changed. The alternative anchors
refer to the earlier upload labelled (3). The manuscript source is maintained
separately from this companion repository.

To insert the figures, copy the three PDFs from `clt_qq/paper/figures/` into
the manuscript's `figures/` directory, and copy the three `.tex` snippets from
`clt_qq/paper/latex/` into a manuscript directory named `figure_captions/`.
Ensure the preamble loads `graphicx`, `amsmath`, and `amssymb`:

```latex
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
```

Compile the manuscript twice to resolve figure references. At each position below, insert the
linking sentence and `\input` command. The complete figure environments and
captions are reproduced below as an alternative to using `\input`.
LaTeX may float the figures; these positions place their first references in
the intended discussion.

## Figure 1: absolute-moment averages for Student $t_3$

**Current revision:** immediately after
line 180 of the reviewed introduction,
the explanatory paragraph following Theorem `theorem clt`. Its last words
are “are included for completeness.” The next line is
`\subsection{Randomised HMC}`. Insert the figure before that subsection.

**Upload (3) equivalent:** after line 176, the paragraph ending “which
correspond to the $T=1$ HMC algorithm with and without metropolis correction.”,
and before `\subsection{Randomised HMC}` at line 177.

This puts the finite-variance moment example beside the CLT discussion and
introduces the randomised kernel considered in the next subsection. In this
example $|X|$ has a finite second moment under Student $t_3$; that statement
concerns the target law, not exact stationarity of the unadjusted kernels.

Paste at that position:

```latex
Figure~\ref{fig:ula-fixed-random} illustrates the distribution of first
absolute-moment averages for a Student $t$ potential with three degrees of
freedom, comparing ULA with unadjusted HMC using fixed and independently
randomised trajectory lengths. Although the observable has a finite second
moment under the target law, that condition alone does not guarantee a
central limit theorem for its ergodic averages.
\input{figure_captions/01_ula_fixed_random}
```

## Figure 2: randomisation law and trajectory length

**Current revision and upload (3):** immediately after line 259, the
`\end{rem}` closing the remark beginning “Canonical examples covered by
Assumption”. Its final sentence ends with
`\cite{mackenzie1989improved,neal2011mcmc}.` Insert before the next subsection,
`\subsection{Related literature and discussion}`, at line 262.

This places the fixed, uniform and two-point examples directly after the
paper's admissible randomisation schemes. The target changes here from
Student $t_3$ to Cauchy, so the linking sentence and caption both say so.

Paste at that position:

```latex
For the standard Cauchy target and the tail observable
$g(x)=\mathbf{1}_{\{x\geq2\}}$, Figure~\ref{fig:randomisation-length}
compares three trajectory-length laws at two mean lengths. The expected
number of leapfrog steps is equal within each row, while the second moment
of the integration time differs.
\input{figure_captions/02_randomisation_and_length}
```

## Figure 3: HMC and NUTS

**Current revision:** immediately after
line 269 of the reviewed introduction,
the NUTS paragraph ending “requires analysis of those stopping and selection
rules.” Insert before
`\subsection{Structure of the paper and notation}` at line 271.

**Upload (3) equivalent:** after line 298, the `\end{rem}` closing the NUTS
remark whose final words are “index-selection rules.” Insert before
`\subsection{Structure of the paper and notation}` at line 302.

This keeps the empirical comparison next to the NUTS conjecture. It does not
turn the conjecture into a claim proved by either the QQ plots or the theorem
for independent randomisation.

Paste at that position:

```latex
Figure~\ref{fig:hmc-nuts} compares fixed-length and independently randomised
HMC with standard NUTS for the same Cauchy tail observable. This finite-run
comparison does not determine the asymptotic fluctuation scaling or the
polynomial convergence exponent of NUTS.
\input{figure_captions/03_hmc_nuts}
```

## Complete figure environments and captions

The following blocks exactly reproduce the [caption files](../clt_qq/paper/latex/). Use either these
blocks or the `\input` commands above, not both.

### Figure 1

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/01_ula_fixed_random.pdf}
    \caption{Gaussian QQ plots of first absolute-moment averages for ULA,
    unadjusted HMC with $T=10$ leapfrog steps, and unadjusted HMC with an
    independent $T\sim\mathrm{Unif}\{1,\ldots,19\}$ at each iteration.
    The potential is $U(x)=2\log(1+x^2/3)$, corresponding to the Student
    $t$ distribution with three degrees of freedom, and $g(x)=|x|$.
    This observable has a finite second moment under the target law.
    Each panel contains 2,000 independent chains started at zero, run for
    120,000 iterations with the first 40,000 discarded; each point is an
    average over the remaining 80,000 observations. Step size is $h=0.35$,
    mass is one, and tuning is fixed.
    Vertical coordinates are the ordered raw chain averages; horizontal
    coordinates are Gaussian quantiles fitted using their sample mean and
    standard deviation. There is no $\sqrt n$ scaling. The line is $y=x$;
    axis limits are chosen separately and every average is shown.
    All three kernels are unadjusted and may have invariant-distribution
    bias. These finite-run plots illustrate distributional shape rather
    than establish a CLT or its failure.}
    \label{fig:ula-fixed-random}
\end{figure}
```

### Figure 2

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/02_randomisation_and_length.pdf}
    \caption{Trajectory length and independent randomisation for
    Metropolis-adjusted HMC. Here the target is standard Cauchy,
    $U(x)=\log(1+x^2)$, and the observable is
    $g(x)=\mathbf{1}_{\{x\geq2\}}$.
    Rows have mean leapfrog count $m=5$ and $m=20$.
    Columns use $T=m$, $T\sim\mathrm{Unif}\{1,\ldots,2m-1\}$, and
    $T\in\{1,2m-1\}$ with equal probabilities, respectively; random counts
    are redrawn independently at each iteration. Expected leapfrog work is
    equal within each row, while $\mathbb{E}[T^2]$ differs.
    Each panel uses 2,000 independent chains and 80,000 retained observations
    per chain. Initialisation, burn-in, step size, unit mass and the raw
    Gaussian QQ construction are as in
    Figure~\ref{fig:ula-fixed-random}. These are finite-run distributional
    comparisons.}
    \label{fig:randomisation-length}
\end{figure}
```

### Figure 3

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/03_hmc_nuts.pdf}
    \caption{Cauchy tail-probability averages under Metropolis-adjusted HMC
    with $T=10$, Metropolis-adjusted HMC with independent
    $T\sim\mathrm{Unif}\{1,\ldots,19\}$ at each iteration, and NUTS.
    The target is standard Cauchy, $U(x)=\log(1+x^2)$, with
    $g(x)=\mathbf{1}_{\{x\geq2\}}$.
    NUTS uses BlackJAX 1.6.2 with JAX 0.11.2, standard multinomial
    trajectory construction and the library's default tree limit.
    Each panel uses 2,000 independent chains and 80,000 retained observations
    per chain. All methods use fixed step size $h=0.35$, unit mass and no
    adaptation, with initialisation, burn-in and the raw Gaussian QQ
    construction as in Figure~\ref{fig:ula-fixed-random}.
    Iteration counts are equal; gradient costs differ.
    The plots compare finite-run distributional shape and do not establish
    polynomial convergence rates or CLT failure for NUTS.}
    \label{fig:hmc-nuts}
\end{figure}
```

## Interpretation details

- Figure 1 uses unadjusted $Q$ kernels. The potential corresponds to Student
  $t_3$, but their invariant distributions can have discretisation bias.
- Figure 2 uses adjusted $P$ kernels. It matches $\mathbb{E}[T]$ within each
  row and therefore expected leapfrog work at the common step size, while
  $\mathbb{E}[T^2]$ differs. Each random count is sampled independently at
  every transition.
- Figure 3 uses adjusted HMC and standard BlackJAX NUTS with fixed step size
  and unit mass. The library's default tree limit is left unchanged; no
  adaptation is run. Equal transition counts do not imply equal gradient
  work.
- Every chain average is retained in every QQ plot. The Gaussian location
  and scale are fitted separately to each panel, with no $\sqrt n$ scaling
  or target centring. Thus the figures examine finite-run distributional
  shape rather than establish a CLT, its failure, or a convergence rate.
