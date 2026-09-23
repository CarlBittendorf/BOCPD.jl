# Extending BOCPD.jl

## Observation models

A custom model is a plain Julia type. Implement the small generic interface:

```julia
struct MyModel <: AbstractObservationModel
    prior_mean::Float64
end

struct MyState <: AbstractPosteriorState
    mean::Float64
end

prior_state(model::MyModel) = MyState(model.prior_mean)
logpredictive(model::MyModel, state::MyState, x::Real) = ...
update(model::MyModel, state::MyState, x::Real) = ...
```

`reset(model, x)` defaults to `update(model, prior_state(model), x)`. Fully missing observations are handled by the detector and do not require model-specific methods.

## Hazards

Hazards are callable objects or ordinary functions:

```julia
hazard = r -> clamp(0.001 + 0.0001r, 0, 1)
```

Use `CustomHazard` when a function depends on time or both run length and time:

```julia
hazard = CustomHazard(
    (r, t) -> clamp(0.01 + 0.001r + 0.0001t, 0, 1);
    depends_on=:run_time
)
```

## Pruning

Built-in pruning strategies are concrete values. A custom strategy can implement the internal selection contract, but should be accompanied by normalization and discarded-mass tests before becoming public API.