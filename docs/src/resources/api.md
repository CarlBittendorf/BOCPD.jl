# API

```@meta
CurrentModule = BOCPD
```

## Core filtering

```@docs
BOCPDDetector
Detector
update!
fit
runlength_probs
runlength_probabilities
log_runlength_probs
most_likely_runlength
current_changepoint_probability
time_index
predictive_distribution
```

## Results and smoothing

```@docs
BOCPDResult
changepoint_probabilities
changepoint_probability
delayed_changepoint_probability
observation_status
ismissingobservation
FixedLag
```

## Observation models

```@docs
AbstractObservationModel
AbstractPosteriorState
prior_state
logpredictive
update
reset
GaussianMeanModel
NormalInverseGammaModel
MultivariateGaussianMeanModel
MultivariateGaussianMeanCovarianceModel
BernoulliModel
PoissonModel
ObservedSubset
supports_partial_observations
```

## Hazards and policies

```@docs
ConstantHazard
GeometricHazard
CustomHazard
NoHistory
FullHistory
FixedLagHistory
RejectNaN
TreatNaNAsMissing
NoPruning
MaxRunLength
ProbabilityPruning
TopKPruning
AbstractHistoryPolicy
AbstractInvalidDataPolicy
AbstractPruning
```