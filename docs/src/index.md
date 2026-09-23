```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "BOCPD.jl"
  text:
  tagline: Flexible Bayesian Online Change Point Detection
  actions:
    - theme: brand
      text: Getting started
      link: /getting_started
    - theme: alt
      text: View on Github
      link: https://github.com/CarlBittendorf/BOCPD.jl
---
```

# Welcome to BOCPD!

BOCPD.jl provides a flexible and efficient implementation of Bayesian Online Change Point Detection for univariate and multivariate data. It supports multiple observation models, unknown means and variances, custom hazard functions, missing observations, and delayed changepoint probabilities. Its idiomatic Julia API makes the package easy to use, extend, and integrate into real-time data-processing workflows.

## Installation

BOCPD.jl is not yet registered in the Julia General registry. Install it directly from GitHub:

```julia
using Pkg

Pkg.add(url = "https://github.com/CarlBittendorf/BOCPD.jl")
```

## Quick start

```julia
using BOCPD

observations = vcat(fill(0.0, 30), fill(4.0, 30))

model = NormalInverseGammaModel()

result = fit(model, observations; hazard = ConstantHazard(50))

changepoint_probabilities(result)
```

## Acknowledgements

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – GRK2739/2 – Project Nr. 447089431 – Research Training Group: KD²School – Designing Biosignal-Adaptive Systems for Decision-Making Processes

```@meta
CurrentModule = BOCPD
```