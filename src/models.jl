"""Supertype for Bayesian observation models used by the run-length filter."""
abstract type AbstractObservationModel end

"""Supertype for immutable sufficient-statistic or posterior states."""
abstract type AbstractPosteriorState end

"""Observed coordinates of a partially observed vector and its full dimension."""
struct ObservedSubset{V<:AbstractVector,I<:AbstractVector{Int}}
    values::V
    indices::I
    dimension::Int

    function ObservedSubset(
        values::V, indices::I, dimension::Int
    ) where {V<:AbstractVector,I<:AbstractVector{Int}}
        length(values) == length(indices) || throw(DimensionMismatch("Observed values and indices must have equal length"))
        all(1 <= index <= dimension for index in indices) || throw(ArgumentError("Observed index is outside the model dimension"))
        issorted(indices) && length(unique(indices)) == length(indices) || throw(ArgumentError("Observed indices must be sorted and unique"))

        return new{V,I}(values, indices, dimension)
    end
end

"""Return whether `model` implements exact partial-vector inference."""
supports_partial_observations(::AbstractObservationModel) = false

function logpredictive(model::AbstractObservationModel, state::AbstractPosteriorState, ::Missing)
    0.0
end

update(model::AbstractObservationModel, state::AbstractPosteriorState, ::Missing) = state

reset(model::AbstractObservationModel, ::Missing) = prior_state(model)

"""Return a fresh prior posterior state for `model`."""
prior_state(model::AbstractObservationModel) = throw(MethodError(prior_state, (model,)))

"""Evaluate the log predictive density for one state and observation."""
function logpredictive(model::AbstractObservationModel, state::AbstractPosteriorState, x)
    throw(MethodError(logpredictive, (model, state, x)))
end

"""Update a posterior state with one observed value."""
function update(model::AbstractObservationModel, state::AbstractPosteriorState, x)
    throw(MethodError(update, (model, state, x)))
end

"""Create the posterior state for a new segment after observing `x`."""
reset(model::AbstractObservationModel, x) = update(model, prior_state(model), x)

function predictive_distribution(model::AbstractObservationModel, state::AbstractPosteriorState)
    throw(ArgumentError("predictive_distribution is not implemented for $(typeof(model))"))
end

"""Univariate Gaussian observations with known variance and unknown mean."""
struct GaussianMeanModel{T<:Real} <: AbstractObservationModel
    prior_mean::T
    prior_variance::T
    observation_variance::T

    function GaussianMeanModel(
        prior_mean::T, prior_variance::T, observation_variance::T
    ) where {T<:Real}
        prior_variance > 0 && observation_variance > 0 ||
            throw(ArgumentError("Variances must be positive"))

        return new{T}(prior_mean, prior_variance, observation_variance)
    end
end

function GaussianMeanModel(; prior_mean=0.0, prior_variance=1.0, observation_variance=1.0)
    GaussianMeanModel(promote(prior_mean, prior_variance, observation_variance)...)
end

struct GaussianMeanState{T<:Real} <: AbstractPosteriorState
    mean::T
    variance::T
end

prior_state(m::GaussianMeanModel) = GaussianMeanState(m.prior_mean, m.prior_variance)

function logpredictive(m::GaussianMeanModel, s::GaussianMeanState, x::Real)
    logpdf(Normal(s.mean, sqrt(s.variance + m.observation_variance)), x)
end

function update(m::GaussianMeanModel, s::GaussianMeanState, x::Real)
    v = inv(inv(s.variance) + inv(m.observation_variance))

    return GaussianMeanState(v * (s.mean / s.variance + x / m.observation_variance), v)
end

function predictive_distribution(m::GaussianMeanModel, s::GaussianMeanState)
    Normal(s.mean, sqrt(s.variance + m.observation_variance))
end

"""Univariate Gaussian observations with a Normal-Inverse-Gamma prior."""
struct NormalInverseGammaModel{T<:Real} <: AbstractObservationModel
    prior_mean::T
    prior_strength::T
    shape::T
    scale::T

    function NormalInverseGammaModel(m::T, k::T, a::T, b::T) where {T<:Real}
        k > 0 && a > 0 && b > 0 || throw(ArgumentError("prior_strength, shape, and scale must be positive"))

        return new{T}(m, k, a, b)
    end
end

function NormalInverseGammaModel(; prior_mean=0.0, prior_strength=1.0, shape=2.0, scale=1.0)
    NormalInverseGammaModel(promote(prior_mean, prior_strength, shape, scale)...)
end

struct NormalInverseGammaState{T<:Real} <: AbstractPosteriorState
    mean::T
    strength::T
    shape::T
    scale::T
end

function prior_state(m::NormalInverseGammaModel)
    NormalInverseGammaState(m.prior_mean, m.prior_strength, m.shape, m.scale)
end

function predictive_distribution(::NormalInverseGammaModel, s::NormalInverseGammaState)
    ν = 2s.shape
    scale = sqrt(s.scale * (s.strength + 1) / (s.shape * s.strength))

    return s.mean + scale * TDist(ν)
end

function logpredictive(m::NormalInverseGammaModel, s::NormalInverseGammaState, x::Real)
    logpdf(predictive_distribution(m, s), x)
end

function update(::NormalInverseGammaModel, s::NormalInverseGammaState, x::Real)
    k = s.strength + one(s.strength)
    δ = x - s.mean
    m = s.mean + δ / k
    a = s.shape + one(s.shape) / 2
    b = s.scale + s.strength * δ^2 / (2k)

    return NormalInverseGammaState(m, k, a, b)
end

"""Multivariate Gaussian observations with known covariance and unknown mean."""
struct MultivariateGaussianMeanModel{
    T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T},C} <:
    AbstractObservationModel
    prior_mean::V
    prior_covariance::M
    observation_covariance::C
end

supports_partial_observations(::MultivariateGaussianMeanModel) = true

function MultivariateGaussianMeanModel(
    prior_mean::AbstractVector{T},
    prior_covariance::AbstractMatrix{T},
    observation_covariance::AbstractMatrix{T}
) where {T<:Real}
    length(prior_mean) == size(prior_covariance, 1) == size(prior_covariance, 2) ==
        size(observation_covariance, 1) == size(observation_covariance, 2) ||
        throw(DimensionMismatch("Mean and covariance dimensions must agree"))

    cholesky(Symmetric(prior_covariance))
    cholesky(Symmetric(observation_covariance))

    pm, pc, oc = copy(prior_mean), copy(prior_covariance), copy(observation_covariance)

    return MultivariateGaussianMeanModel{T,typeof(pm),typeof(pc),typeof(oc)}(pm, pc, oc)
end

struct MultivariateGaussianMeanState{T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: AbstractPosteriorState
    mean::V
    covariance::M
end

function prior_state(m::MultivariateGaussianMeanModel)
    MultivariateGaussianMeanState(copy(m.prior_mean), copy(m.prior_covariance))
end

function predictive_distribution(m::MultivariateGaussianMeanModel, s::MultivariateGaussianMeanState)
    MvNormal(s.mean, Symmetric(s.covariance + m.observation_covariance))
end

function logpredictive(m::MultivariateGaussianMeanModel, s::MultivariateGaussianMeanState, x::AbstractVector)
    length(x) == length(s.mean) || throw(DimensionMismatch("Observation dimension does not match model"))

    return logpdf(predictive_distribution(m, s), x)
end

function update(m::MultivariateGaussianMeanModel, s::MultivariateGaussianMeanState, x::AbstractVector)
    length(x) == length(s.mean) || throw(DimensionMismatch("Observation dimension does not match model"))

    prior_precision = s.covariance \ Matrix{eltype(s.covariance)}(I, size(s.covariance))
    observation_precision = m.observation_covariance \ Matrix{eltype(m.observation_covariance)}(I, size(m.observation_covariance))
    posterior_precision = cholesky(Symmetric(prior_precision + observation_precision))
    covariance = posterior_precision \ Matrix{eltype(s.covariance)}(I, size(s.covariance))

    return MultivariateGaussianMeanState(covariance * (prior_precision * s.mean + observation_precision * x), covariance)
end

function logpredictive(m::MultivariateGaussianMeanModel, s::MultivariateGaussianMeanState, x::ObservedSubset)
    x.dimension == length(s.mean) || throw(DimensionMismatch("Observation dimension does not match model"))

    isempty(x.indices) && return 0.0

    covariance = s.covariance[x.indices, x.indices] + m.observation_covariance[x.indices, x.indices]

    return logpdf(MvNormal(s.mean[x.indices], Symmetric(covariance)), x.values)
end

function update(m::MultivariateGaussianMeanModel, s::MultivariateGaussianMeanState, x::ObservedSubset)
    x.dimension == length(s.mean) || throw(DimensionMismatch("Observation dimension does not match model"))

    isempty(x.indices) && return s

    prior_cross = s.covariance[:, x.indices]
    observed_covariance = s.covariance[x.indices, x.indices] + m.observation_covariance[x.indices, x.indices]
    factor = cholesky(Symmetric(observed_covariance))
    residual = x.values - s.mean[x.indices]
    posterior_mean = s.mean + prior_cross * (factor \ residual)
    posterior_covariance = s.covariance - prior_cross * (factor \ prior_cross')
    posterior_covariance = Matrix(Symmetric((posterior_covariance + posterior_covariance') / 2))

    return MultivariateGaussianMeanState(posterior_mean, posterior_covariance)
end

"""Multivariate Gaussian observations with a Normal-Inverse-Wishart prior."""
struct NormalInverseWishartModel{
    T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: AbstractObservationModel
    prior_mean::V
    prior_strength::T
    degrees_of_freedom::T
    scale_matrix::M
end

function NormalInverseWishartModel(;
    prior_mean,
    prior_strength=1.0,
    degrees_of_freedom=length(prior_mean)+2.0,
    scale_matrix=Matrix{typeof(prior_strength)}(I, length(prior_mean), length(prior_mean))
)
    NormalInverseWishartModel(prior_mean, prior_strength, degrees_of_freedom, scale_matrix)
end

struct NormalInverseWishartState{
    T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: AbstractPosteriorState
    mean::V
    strength::T
    degrees_of_freedom::T
    scale_matrix::M
end

function prior_state(m::NormalInverseWishartModel)
    NormalInverseWishartState(copy(m.prior_mean), m.prior_strength, m.degrees_of_freedom, copy(m.scale_matrix))
end

function predictive_distribution(::NormalInverseWishartModel, s::NormalInverseWishartState)
    d = length(s.mean)
    ν = s.degrees_of_freedom - d + 1
    scale = ((s.strength + 1) / (s.strength * ν)) * s.scale_matrix

    return MvTDist(ν, s.mean, Matrix(Symmetric(scale)))
end

function logpredictive(m::NormalInverseWishartModel, s::NormalInverseWishartState, x::AbstractVector)
    length(x) == length(s.mean) || throw(DimensionMismatch("Observation dimension does not match model"))

    return logpdf(predictive_distribution(m, s), x)
end

function update(::NormalInverseWishartModel, s::NormalInverseWishartState, x::AbstractVector)
    length(x) == length(s.mean) || throw(DimensionMismatch("Observation dimension does not match model"))

    k = s.strength + 1
    δ = x - s.mean

    return NormalInverseWishartState(s.mean + δ / k, k, s.degrees_of_freedom + 1, s.scale_matrix + (s.strength / k) * (δ * δ'))
end

"""Bernoulli observations with a Beta prior on the success probability."""
struct BernoulliModel{T<:Real} <: AbstractObservationModel
    alpha::T
    beta::T

    function BernoulliModel(a::T, b::T) where {T<:Real}
        a > 0 && b > 0 || throw(ArgumentError("Beta parameters must be positive"))

        return new{T}(a, b)
    end
end

BernoulliModel(; alpha=1.0, beta=1.0) = BernoulliModel(promote(alpha, beta)...)

struct BernoulliState{T<:Real} <: AbstractPosteriorState
    alpha::T
    beta::T
end

prior_state(m::BernoulliModel) = BernoulliState(m.alpha, m.beta)

function logpredictive(::BernoulliModel, s::BernoulliState, x::Bool)
    x ? log(s.alpha / (s.alpha + s.beta)) : log(s.beta / (s.alpha + s.beta))
end

function update(::BernoulliModel, s::BernoulliState, x::Bool)
    x ? BernoulliState(s.alpha + 1, s.beta) : BernoulliState(s.alpha, s.beta + 1)
end

function predictive_distribution(::BernoulliModel, s::BernoulliState)
    Bernoulli(s.alpha / (s.alpha + s.beta))
end

"""Count observations with a Gamma prior on the Poisson rate."""
struct PoissonModel{T<:Real} <: AbstractObservationModel
    shape::T
    rate::T

    function PoissonModel(a::T, b::T) where {T<:Real}
        a > 0 && b > 0 || throw(ArgumentError("Gamma parameters must be positive"))

        return new{T}(a, b)
    end
end

PoissonModel(; shape=1.0, rate=1.0) = PoissonModel(promote(shape, rate)...)

struct PoissonState{T<:Real} <: AbstractPosteriorState
    shape::T
    rate::T
end

prior_state(m::PoissonModel) = PoissonState(m.shape, m.rate)

function logpredictive(::PoissonModel, s::PoissonState, x::Integer)
    x >= 0 || throw(DomainError(x, "Poisson observations must be nonnegative"))

    return logpdf(NegativeBinomial(s.shape, s.rate / (s.rate + 1)), x)
end

function update(::PoissonModel, s::PoissonState, x::Integer)
    (x >= 0 || throw(DomainError(x)); PoissonState(s.shape + x, s.rate + 1))
end

function predictive_distribution(::PoissonModel, s::PoissonState)
    NegativeBinomial(s.shape, s.rate / (s.rate + 1))
end