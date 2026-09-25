# BOCPD.jl

[![CI](https://github.com/CarlBittendorf/BOCPD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/CarlBittendorf/BOCPD.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://carlbittendorf.github.io/BOCPD.jl/)

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

## Choosing a model

| Model | Observation type | Unknown quantities | Predictive distribution | Partial vectors |
| --- | --- | --- | --- | --- |
| `GaussianMeanModel` | Scalar continuous | Mean; variance is known | Gaussian | Not applicable |
| `NormalInverseGammaModel` | Scalar continuous | Mean and variance | Student-t | Not applicable |
| `MultivariateGaussianMeanModel` | Vector continuous | Mean; covariance is known | Multivariate Gaussian | Exact marginal update |
| `MultivariateGaussianMeanCovarianceModel` | Vector continuous | Mean and full covariance | Multivariate Student-t | Not supported; fully missing vectors are supported |
| `BernoulliModel` | Boolean | Success probability | Beta-Bernoulli predictive | Not applicable |
| `PoissonModel` | Nonnegative integer count | Rate | Negative-binomial predictive | Not applicable |

Use `GaussianMeanModel` when the measurement noise variance is known and stable. Use `NormalInverseGammaModel` when both the segment mean and its scalar variance may change. For vector-valued measurements with known measurement covariance, use `MultivariateGaussianMeanModel`; it also supports vectors with some coordinates missing. For vector-valued measurements where both the mean and full covariance are unknown, use `MultivariateGaussianMeanCovarianceModel`. Its complete observations are conjugate, but partial vectors are rejected because exact conjugate updates with an unknown full covariance are not implemented.

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

When your multivariate data are stored as separate coordinate vectors, reshape them so each element is one observation at a single time point:

```julia
x = [0.0, 0.2, 4.0]
y = [1.0, 0.9, 5.0]
observations = collect.(zip(x, y))
# observations == [[0.0, 1.0], [0.2, 0.9], [4.0, 5.0]]
```

The detector expects a vector of observations, where each observation is itself a vector of length equal to the measurement dimension.

## Hazards

`ConstantHazard(100)` means a geometric expected segment length of 100. `GeometricHazard(0.01)` expresses the same constant probability directly. Ordinary functions are also accepted:

```julia
increasing = r -> clamp(0.001 + 0.0001r, 0.0, 1.0)
result = fit(model, multivariate_data; hazard=increasing)
```

For hazards depending on elapsed time, use `CustomHazard`:

```julia
time_hazard = CustomHazard(t -> t > 100 ? 0.2 : 0.01; depends_on=:time)

joint_hazard = CustomHazard(
    (r, t) -> clamp(0.01 + 0.001r + 0.0001t, 0, 1);
    depends_on=:run_time
)
```

The `depends_on` value describes the callable signature; hazard probabilities are still validated in `[0, 1]` at every transition.

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
    GaussianMeanModel(),
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