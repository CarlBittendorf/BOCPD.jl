module BOCPD

using Distributions
using LinearAlgebra
using Random
using Statistics
using LogExpFunctions: logsumexp

export AbstractObservationModel, AbstractPosteriorState,
       prior_state, logpredictive, update, reset, predictive_distribution,
       ObservedSubset, supports_partial_observations,
       GaussianMeanModel, NormalInverseGammaModel,
       MultivariateGaussianMeanModel, MultivariateGaussianMeanCovarianceModel,
       BernoulliModel, PoissonModel,
         ConstantHazard, GeometricHazard, NegativeBinomialHazard, CustomHazard,
       AbstractPruning, NoPruning, MaxRunLength, ProbabilityPruning, TopKPruning,
       AbstractHistoryPolicy, NoHistory, FullHistory, FixedLagHistory,
       AbstractInvalidDataPolicy, RejectNaN, TreatNaNAsMissing, FixedLag,
       BOCPDDetector, Detector, update!, fit,
       runlength_probs, runlength_probabilities, log_runlength_probs, most_likely_runlength,
       current_changepoint_probability, changepoint_probabilities,
       predictive_distribution, time_index, delayed_changepoint_probability,
       changepoint_probability, BOCPDResult, observation_status,
       ismissingobservation

include("models.jl")
include("hazards.jl")
include("pruning.jl")
include("policies.jl")
include("filter.jl")
include("results.jl")
include("smoothing.jl")

end