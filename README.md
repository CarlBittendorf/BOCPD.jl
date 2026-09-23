# BOCPD.jl

[![CI](https://github.com/CarlBittendorf/BOCPD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/CarlBittendorf/BOCPD.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

BOCPD.jl implements Bayesian online changepoint detection by tracking a posterior distribution over the current run length and the probability that the latest observation started a new segment.

## Features

- Online changepoint detection for streaming data
- Univariate and multivariate conjugate models
- Missing observations and partial multivariate updates
- Callable hazards depending on run length, time, or both
- Fixed-lag changepoint probabilities and pruning controls
- Small, multiple-dispatch-based extension API

## Installation

This package is not yet registered in the Julia General registry. Install it directly from GitHub:

```julia
using Pkg

Pkg.add(url = "https://github.com/CarlBittendorf/BOCPD.jl")
```

## Quick start

```julia
using BOCPD

observations = vcat(fill(0.0, 30), fill(4.0, 30))

model = NormalInverseGammaModel(
    prior_mean = 0.0,
    prior_strength = 1.0,
    shape = 2.0,
    scale = 1.0,
)

result = fit(
    model,
    observations;
    hazard = ConstantHazard(50),
    pruning = MaxRunLength(100),
    history = FullHistory(),
)

changepoint_probabilities(result)
```

The package uses the convention `r_t = 0` when `x_t` is the first observation of a new segment. A changepoint probability at time `t` therefore means that `x_t` started the segment.

## Online filtering

```julia
model = NormalInverseGammaModel()
detector = BOCPDDetector(model, ConstantHazard(50))

for x in observations
    update!(detector, x)
    current_changepoint_probability(detector)
end
```

## Multivariate example

```julia
using LinearAlgebra

multivariate_data = [[0.0, 1.0], [0.2, 0.9], [4.0, 5.0]]

model = MultivariateGaussianMeanModel(
    [0.0, 0.0],
    Matrix{Float64}(I, 2, 2),
    Matrix{Float64}(I, 2, 2),
)

fit(model, multivariate_data; hazard = ConstantHazard(50))
```

## Missing data

```julia
observations = Union{Missing, Float64}[0.1, 0.2, missing, 0.0, 4.8, 5.1]

result = fit(
    GaussianMeanModel(),
    observations;
    hazard = ConstantHazard(100),
    history = FixedLagHistory(5),
)

observation_status(result, 3)
```

A fully missing observation advances time and applies the hazard without updating posterior sufficient statistics. Use `TreatNaNAsMissing()` when your pipeline encodes missing values as `NaN`.

## Fixed-lag smoothing

```julia
result = fit(
    model,
    observations;
    hazard = ConstantHazard(50),
    history = FixedLagHistory(5),
)

changepoint_probabilities(result, FixedLag(5))
```

This yields delayed changepoint probabilities conditioned on later observations. It is a marginal over latent changepoint histories and is not equivalent to the later terminal run-length event.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – GRK2739/2 – Project Nr. 447089431 – Research Training Group: KD²School – Designing Biosignal-Adaptive Systems for Decision-Making Processes