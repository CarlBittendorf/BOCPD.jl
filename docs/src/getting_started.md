# Getting started

## Batch inference

```julia
using BOCPD

observations = vcat(fill(0.0, 20), fill(4.0, 20))
model = GaussianMeanModel(observation_variance=1.0)
result = fit(model, observations;
    hazard=ConstantHazard(40),
    history=FullHistory())

probabilities = changepoint_probabilities(result)
```

The returned vector is indexed by observation time. A changepoint at time `t` means that `observations[t]` starts a new segment.

## Streaming inference

```julia
detector = Detector(model, ConstantHazard(40))
for observation in observations
    update!(detector, observation)
end
current_changepoint_probability(detector)
```

Use `runlength_probs(detector)` for the current posterior and `most_likely_runlength(detector)` for its mode.

## Configuration policies

History, pruning, invalid data, and fixed-lag queries are represented by values:

```julia
result = fit(model, observations;
    pruning=MaxRunLength(100),
    history=FixedLagHistory(5),
    invalid_data=RejectNaN())

delayed = changepoint_probabilities(result, FixedLag(5))
```
