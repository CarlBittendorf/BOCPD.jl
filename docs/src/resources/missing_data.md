# Missing data

Use Julia's `missing` value when a time point occurred but its measurement is unavailable:

```julia
observations = Union{Missing,Float64}[0.1, 0.2, missing, 4.8]

result = fit(
    GaussianMeanModel(),
    observations;
    hazard=ConstantHazard(50),
    history=FixedLagHistory(5)
)
```

A missing point advances the run-length process and applies the hazard. It contributes no likelihood evidence and does not update posterior sufficient statistics. `observation_status(result, t)` reports `:observed`, `:missing`, or `:partially_observed`.

`NaN` is rejected by `RejectNaN()`, the default invalid-data policy. If NaN is your explicit missing-value encoding, opt in with `TreatNaNAsMissing()`. Infinite values remain invalid.

For multivariate Gaussian means with known covariance, partial vectors are marginalized to their observed coordinates:

```julia
update!(detector, Union{Missing,Float64}[1.2, missing, 3.7])
```

Normal-Inverse-Wishart partial updates are rejected because exact conjugate updates for an unknown full covariance are not implemented. Completely missing observations remain supported for every built-in model.

To represent an elapsed gap, repeat missing transitions explicitly or use `update!(detector, missing; delta_t=3)`. This preserves the intermediate hazard opportunities.