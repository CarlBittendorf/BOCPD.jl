# Getting started

Changepoint detection asks a very simple question: does the latest observation look like the continuation of the current regime, or is it the start of a new one? BOCPD.jl answers that question by maintaining a probability distribution over the current run length and the probability that the most recent observation began a new segment.

The key quantity is the posterior changepoint probability. A value near 1 means that the most recent observation is likely to be the first point of a new regime. A value near 0 means the current state likely continued. This is more informative than a hard threshold: it keeps uncertainty explicit instead of forcing a binary decision.

BOCPD.jl can process scalar or vector observations, supports custom probability models and hazards, and handles `missing` values without pretending they were observed.

## Installation

BOCPD.jl is not yet registered in the Julia General registry, so install it directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/CarlBittendorf/BOCPD.jl")
```

If you are working in the package environment, the package can also be added via the Julia package manager:

```julia
] add BOCPD
```

## A first synthetic example

Consider a sequence that is stable around 0 for a while and then shifts to around 4:

```julia
using BOCPD

observations = vcat(fill(0.0, 30), fill(4.0, 30))
```

The first 30 observations come from one regime with mean near 0, and the later 30 observations come from a second regime with mean near 4. The detector should infer that the transition happens around the first post-shift point, while allowing some uncertainty because the noise is nonzero.

## Choose an observation model

For a scalar Gaussian signal with known measurement noise, `GaussianMeanModel` is a good default. It treats the segment mean as unknown and keeps the observation variance fixed:

```julia
model = GaussianMeanModel(
    prior_mean = 0.0,
    prior_variance = 1.0,
    observation_variance = 1.0,
)
```

The prior mean is the starting estimate for the segment center. A large `prior_variance` makes the model uncertain at the start and lets the data drive early updates. A small `prior_variance` makes the model believe the segment is tightly centered near `prior_mean`.

The `observation_variance` is the noise level of each measurement. Larger values make changes in the underlying mean less surprising, while smaller values make the filter react more strongly to shifts.

For a scalar Gaussian where both the mean and variance are unknown, use `NormalInverseGammaModel`:

```julia
model = NormalInverseGammaModel(
    prior_mean = 0.0,
    prior_strength = 1.0,
    shape = 2.0,
    scale = 1.0,
)
```

This is a conjugate normal-inverse-gamma prior. The `prior_strength` controls how much the initial mean estimate is trusted. The `shape` and `scale` parameters describe the prior uncertainty in the segment variance. A larger `shape` and larger `scale` make the variance prior more concentrated or more spread out depending on the values chosen.

## Multivariate data shape

For multivariate observations, the detector expects a collection of vectors, where each vector contains one observation for all coordinates at a given time step. If your data are stored as separate coordinate vectors, reshape them before fitting:

```julia
x = [0.0, 0.2, 4.0]
y = [1.0, 0.9, 5.0]
observations = collect.(zip(x, y))
# observations == [[0.0, 1.0], [0.2, 0.9], [4.0, 5.0]]
```

This gives the correct input shape for a 2D model: a vector with one length-2 observation per time point.

## Choose a hazard function

A hazard is the prior probability that the next observation begins a new segment. In BOCPD.jl, hazards are functions or callable objects. A simple choice is a constant hazard with an expected segment length of 100 observations:

```julia
hazard = ConstantHazard(100)
```

For a geometric hazard, this means the probability of a changepoint on the next transition is approximately $1 / 100$. This does not force a changepoint every 100 observations; it only encodes a prior belief that segment lengths are typically of that order. A shorter expected segment length gives a stronger preference for frequent changes. A longer one makes the detector more conservative.

## Run batch inference

The smallest useful batch fit is:

```julia
result = fit(
    model,
    observations;
    hazard = ConstantHazard(100),
    history = FullHistory(),
)
```

The returned `BOCPDResult` captures the batch run. Its display is deliberately compact, but the plain-text summary gives the main metadata for the sequence, the latest estimate, the retained history, and any pruning approximation.

## Inspect changepoint probabilities

The core output is the vector of online changepoint probabilities:

```julia
probabilities = changepoint_probabilities(result)
```

This vector has one entry per observation time. A large value means that the corresponding observation is likely to be the first point of a new regime. A small value means the current regime likely continued. The values are probabilities, not hard labels, and both the observation model and the hazard contribute to them.

To find the most likely changepoint time in the batch:

```julia
candidate = argmax(probabilities)
```

The result is indexed with Julia's one-based convention, so `candidate == 31` would mean the first observation after the change is the most probable changepoint candidate. The package's changepoint convention is that `r_t = 0` means the observation at time `t` starts a new segment.

## Understand run length

The run length is the number of observations since the current regime began. In BOCPD.jl, the current posterior over run lengths is available directly:

```julia
most_likely = result.most_likely_runlengths[end]
current = changepoint_probabilities(result)[end]
```

The batch result stores the most likely current run length and the latest online changepoint probability at each time step. The special value `r_t = 0` means the latest observation is the first point in the current segment. This is the same transition convention used throughout the package.

## Use delayed probabilities

A delayed changepoint probability uses future observations to revise earlier estimates. This is useful when a later observation resolves uncertainty about whether a past value started a new segment:

```julia
delayed = changepoint_probabilities(result; delay = 5)
```

A delay of 0 returns the online probability, while a positive delay asks for a fixed-lag estimate conditioned on observations up to `t + L`. Near the end of the sequence, fewer future observations may be available, and the maximum possible delay depends on the retained history. The result must have retained enough history for the requested delay, or the query raises an informative error.

For a small example, compare the online and delayed values around the simulated transition:

```julia
online = changepoint_probabilities(result)
delayed = changepoint_probabilities(result; delay = 5)

online[25:35]
delayed[25:35]
```

Delayed probabilities are a true marginal over latent changepoint histories and are not simply the later run-length probability at a terminal time.

## Process observations online

Batch inference is convenient for a complete dataset, but BOCPD.jl also supports streaming updates with a persistent detector:

```julia
detector = Detector(model, ConstantHazard(100))

for observation in observations
    update!(detector, observation)
end

current_changepoint_probability(detector)
```

This maintains the posterior state across observations without recomputing the entire fit. The online detector exposes the same run-length and probability queries, and it is often useful for monitoring a live process or for comparing batch and streaming results on the same data.

## Handle missing observations

Missing data are part of the model contract. A missing observation still advances time and applies the hazard, but it contributes no likelihood evidence. In Julia, use `missing` for an unavailable measurement:

```julia
stream = Union{Missing,Float64}[0.1, 0.2, missing, 0.0, 4.8, 5.1]

result = fit(
    GaussianMeanModel(),
    stream;
    hazard = ConstantHazard(50),
    history = FixedLagHistory(5),
)

observation_status(result, 3)
```

The missing point still counts as a time index. The hazard is evaluated for the transition, but no observation-model sufficient statistics are updated from that value. Later observations can still revise delayed changepoint probabilities for the missing time point. `NaN` is not treated as `missing` unless you explicitly choose `TreatNaNAsMissing()`.

## Common configuration choices

The workflow usually changes in a few predictable ways:

- Increase `prior_strength` or make the prior mean more specific when you already know the segment center.
- Shorten the expected segment length to make the detector more responsive to shifts.
- Increase `max_run_length` or choose a milder pruning policy when rare but large run lengths matter.
- Use `FixedLagHistory(L)` when you want delayed changepoint estimates without retaining the full run-length history.
- Pick the observation model that matches your noise assumptions: known variance, unknown variance, scalar or multivariate data.

## Common mistakes

A few patterns usually lead to confusion:

- Treating the hazard as a detection threshold instead of a prior probability on the next transition.
- Choosing a model with unrealistic noise assumptions relative to your signal.
- Reading every local maximum as a definite changepoint without considering uncertainty.
- Requesting a delay that is larger than the retained history.
- Confusing `missing` with skipping a time step.
- Passing a multivariate observation without matching its dimension to the model.
- Reading private result fields directly instead of using the documented query functions.

## Next steps

The rest of the documentation expands on the same ideas with a bit more detail:

- [Algorithm](resources/algorithm.md)
- [Missing data](resources/missing_data.md)
- [Extending BOCPD](resources/extending.md)
- [API reference](resources/api.md)

The package also exposes run-length queries, fixed-lag smoothing, custom hazards, and pruning controls for production workflows. Use these pages as the next stop when you are ready for more detail or custom extensions.
