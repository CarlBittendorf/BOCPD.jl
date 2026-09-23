# Algorithm

BOCPD maintains competing hypotheses about how long the current segment has lasted. For the sequence `10.1, 9.8, 10.2, 10.0, 15.1, 14.9`, the fifth value may be an outlier or evidence for a new regime. The filter keeps both possibilities until later data resolve some uncertainty.

## Run length

The run length is

\[
r_t = \text{number of observations since the most recent changepoint}.
\]

BOCPD.jl uses `r_t = 0` when `x_t` is the first observation of a new segment. Each active run length has its own posterior state and predictive distribution.

## Recursion

Let `H(r)` be the hazard, the prior probability that the next observation starts a new segment. For an existing run-length hypothesis, the growth branch is

\[
p(r_t=r+1, x_t) = p(r_{t-1}=r) [1-H(r)] p(x_t \mid r).
\]

The reset branch collects all possible previous run lengths:

\[
p(r_t=0, x_t) = \sum_r p(r_{t-1}=r) H(r) p(x_t \mid r=0).
\]

The reset predictive uses a fresh prior state. Growth uses the predictive density of the existing posterior state and then updates that state. After both branches are constructed, weights are normalized.

For a constant hazard `h`, `ConstantHazard(lambda)` uses `h = 1 / lambda`. This is a geometric duration prior with expected duration `lambda`, not a deterministic interval.

## Numerical stability

Weights are stored as logarithms. Sums over competing hypotheses use log-sum-exp, preventing underflow on long sequences or highly surprising observations.

## Missing observations

A fully missing observation has log predictive contribution zero. Time still advances and the hazard transition still occurs:

\[
\log p(x_t=\text{missing}\mid state)=0.
\]

Growth carries the posterior state unchanged and reset creates a fresh prior state. It is therefore not equivalent to skipping the time point or imputing a value.

## Pruning

Exact filtering grows with the number of observations. `MaxRunLength`, `ProbabilityPruning`, and `TopKPruning` bound the active hypotheses. Retained weights are renormalized and discarded mass is recorded. These strategies are approximations.

## Fixed-lag smoothing

Later observations can revise whether an earlier observation began a segment. `changepoint_probability(result, t, FixedLag(L))` computes a marginal over latent transition histories conditioned on data through `t + L`. It is not generally the same as the terminal event `r_(t+L) == L`, which excludes paths containing later changepoints.
