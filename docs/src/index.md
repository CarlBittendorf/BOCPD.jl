# BOCPD.jl

BOCPD.jl detects changes in streaming data by maintaining a posterior distribution over run length.

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

```@contents
Pages = ["getting_started.md", "algorithm.md", "missing_data.md", "extending.md", "api.md"]
Depth = 2
```

```@meta
CurrentModule = BOCPD
```
