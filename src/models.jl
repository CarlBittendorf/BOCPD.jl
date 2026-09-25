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

"""
    supports_partial_observations(model::AbstractObservationModel) -> Bool

Indicate whether `model` implements exact inference for partially observed
vectors.
"""
supports_partial_observations(::AbstractObservationModel) = false

"""
    logpredictive(model, state, observation) -> Real

Evaluate the log predictive probability or density of `observation` under a
posterior `state`.
"""
function logpredictive(model::AbstractObservationModel, state::AbstractPosteriorState, ::Missing)
    0.0
end

"""
    update(model, state, ::Missing) -> AbstractPosteriorState

Preserve `state` when an observation is fully missing.
"""
update(model::AbstractObservationModel, state::AbstractPosteriorState, ::Missing) = state

"""
    reset(model, observation) -> AbstractPosteriorState

Create the posterior state for a new segment after observing `observation`.
"""
reset(model::AbstractObservationModel, ::Missing) = prior_state(model)

"""
    prior_state(model::AbstractObservationModel) -> AbstractPosteriorState

Create a fresh prior posterior state for `model`.
"""
prior_state(model::AbstractObservationModel) = throw(MethodError(prior_state, (model,)))

"""
    logpredictive(model, state, observation) -> Real

Evaluate the log predictive probability or density of `observation` under a
posterior `state`.
"""
function logpredictive(model::AbstractObservationModel, state::AbstractPosteriorState, x)
    throw(MethodError(logpredictive, (model, state, x)))
end

"""
    update(model, state, observation) -> AbstractPosteriorState

Update a posterior state with one observed value.
"""
function update(model::AbstractObservationModel, state::AbstractPosteriorState, x)
    throw(MethodError(update, (model, state, x)))
end

"""
    reset(model, observation) -> AbstractPosteriorState

Create the posterior state for a new segment by updating a fresh prior state
with `observation`.
"""
reset(model::AbstractObservationModel, x) = update(model, prior_state(model), x)

"""
    predictive_distribution(model, state) -> Distribution

Return the predictive distribution associated with a posterior `state`.
Observation models should implement this method when a natural distribution
object exists.
"""
function predictive_distribution(model::AbstractObservationModel, state::AbstractPosteriorState)
    throw(ArgumentError("predictive_distribution is not implemented for $(typeof(model))"))
end

"""
    GaussianMeanModel(; prior_mean=0.0, prior_variance=1.0,
                      observation_variance=1.0)

Model univariate Gaussian observations with known observation variance and an
unknown mean.
"""
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

"""
    NormalInverseGammaModel(; prior_mean=0.0, prior_strength=1.0, shape=2.0,
                            scale=1.0)

Model univariate Gaussian observations with a Normal-Inverse-Gamma prior for
the unknown mean and variance.
"""
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

"""
    MultivariateGaussianMeanModel(prior_mean, prior_covariance,
                                  observation_covariance)

Model multivariate Gaussian observations with known covariance and an unknown
mean. Partially observed vectors are supported.
"""
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

"""
    MultivariateGaussianMeanCovarianceModel(dimension::Integer; T=Float64)
    MultivariateGaussianMeanCovarianceModel(; dimension::Integer, T=Float64)
    MultivariateGaussianMeanCovarianceModel(prior_mean::AbstractVector;
        prior_strength=1.0,
        degrees_of_freedom=length(prior_mean) + 2.0,
        scale=I)

A multivariate Gaussian observation model with an unknown mean and unknown full
covariance matrix. The model uses a Normal-Inverse-Wishart prior, which yields a
multivariate Student-t posterior predictive distribution. Partially observed
vectors are not supported, but fully missing vectors remain valid.

The default constructor `MultivariateGaussianMeanCovarianceModel(dimension)` creates
an isotropic zero-centered prior with identity-scale uncertainty, where the prior
mean is `zeros(dimension)`, the prior strength is `1`, the degrees of freedom are
`dimension + 2`, and the scale matrix is the identity. This gives a proper,
weakly informative prior for a multivariate Gaussian with an unknown mean and an
unknown covariance. For stronger or more informative priors, supply the full
hyperparameters explicitly.

The model is the multivariate analogue of `GaussianMeanModel` with unknown mean
and known variance, and the multivariate analogue of `NormalInverseGammaModel`
for scalar unknown variance. The predictive distribution is a multivariate
Student-t distribution, and the posterior updates preserve the same conjugate
Normal-Inverse-Wishart family.
"""
struct MultivariateGaussianMeanCovarianceModel{
    T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: AbstractObservationModel
    prior_mean::V
    prior_strength::T
    degrees_of_freedom::T
    scale_matrix::M

    function MultivariateGaussianMeanCovarianceModel(
        prior_mean::AbstractVector{T},
        prior_strength::Real,
        degrees_of_freedom::Real,
        scale_matrix::AbstractMatrix{U},
    ) where {T<:Real,U<:Real}
        d = length(prior_mean)
        d > 0 || throw(ArgumentError("prior_mean must be non-empty"))
        all(isfinite, prior_mean) || throw(ArgumentError("prior_mean entries must be finite"))

        strength = float(prior_strength)
        df = float(degrees_of_freedom)
        strength > 0 || throw(ArgumentError("prior_strength must be positive"))
        df > d - 1 || throw(ArgumentError("degrees_of_freedom must exceed dimension - 1"))

        size(scale_matrix, 1) == d && size(scale_matrix, 2) == d || throw(DimensionMismatch(
            "scale_matrix must have the same dimension as prior_mean"
        ))
        all(isfinite, scale_matrix) || throw(ArgumentError("scale_matrix entries must be finite"))
        isapprox(scale_matrix, scale_matrix', atol=1e-8, rtol=1e-7) || throw(ArgumentError("scale_matrix must be symmetric"))

        try
            cholesky(Symmetric(scale_matrix))
        catch
            throw(PosDefException("scale_matrix must be positive definite"))
        end

        promoted = promote_type(T, typeof(strength), typeof(df), U)
        prior_mean_t = Vector{promoted}(prior_mean)
        scale_t = Matrix{promoted}(scale_matrix)
        return new{promoted,typeof(prior_mean_t),typeof(scale_t)}(
            prior_mean_t,
            convert(promoted, strength),
            convert(promoted, df),
            scale_t,
        )
    end
end

function Base.:(==)(a::MultivariateGaussianMeanCovarianceModel, b::MultivariateGaussianMeanCovarianceModel)
    return a.prior_mean == b.prior_mean &&
           a.prior_strength == b.prior_strength &&
           a.degrees_of_freedom == b.degrees_of_freedom &&
           a.scale_matrix == b.scale_matrix
end

function MultivariateGaussianMeanCovarianceModel(
    dimension::Integer;
    T::Type{<:Real}=Float64,
)
    dimension > 0 || throw(ArgumentError("dimension must be positive"))
    prior_mean = zeros(T, Int(dimension))
    prior_strength = one(T)
    degrees_of_freedom = T(dimension + 2)
    scale_matrix = Matrix{T}(I, Int(dimension), Int(dimension))

    return MultivariateGaussianMeanCovarianceModel(prior_mean, prior_strength, degrees_of_freedom, scale_matrix)
end

function MultivariateGaussianMeanCovarianceModel(;
    dimension::Union{Integer,Nothing}=nothing,
    prior_mean::Union{AbstractVector,Nothing}=nothing,
    T::Type{<:Real}=Float64,
    prior_strength::Real=1.0,
    degrees_of_freedom::Union{Real,Nothing}=nothing,
    scale::Union{AbstractMatrix,Nothing}=nothing,
    scale_matrix::Union{AbstractMatrix,Nothing}=nothing,
)
    if prior_mean === nothing
        dimension === nothing && throw(ArgumentError("Specify either dimension or prior_mean"))
        dimension > 0 || throw(ArgumentError("dimension must be positive"))
        prior_mean = zeros(T, Int(dimension))
    else
        if dimension !== nothing && length(prior_mean) != Int(dimension)
            throw(DimensionMismatch("dimension and prior_mean length must agree"))
        end
        dimension = length(prior_mean)
    end

    if scale_matrix !== nothing
        scale === nothing || throw(ArgumentError("Specify only one of scale or scale_matrix"))
        scale = scale_matrix
    end
    if scale === nothing
        scale = Matrix{T}(I, Int(dimension), Int(dimension))
    end
    if degrees_of_freedom === nothing
        degrees_of_freedom = T(dimension + 2)
    end

    return MultivariateGaussianMeanCovarianceModel(prior_mean, prior_strength, degrees_of_freedom, scale)
end

function MultivariateGaussianMeanCovarianceModel(
    prior_mean::AbstractVector{T};
    prior_strength::Real=one(T),
    degrees_of_freedom::Real=length(prior_mean) + 2,
    scale::AbstractMatrix{<:Real}=Matrix{T}(I, length(prior_mean), length(prior_mean)),
) where {T<:Real}
    return MultivariateGaussianMeanCovarianceModel(prior_mean, prior_strength, degrees_of_freedom, scale)
end

const NormalInverseWishartModel = MultivariateGaussianMeanCovarianceModel
Base.@deprecate NormalInverseWishartModel(args...; kwargs...) MultivariateGaussianMeanCovarianceModel(args...; kwargs...)

struct NormalInverseWishartState{
    T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: AbstractPosteriorState
    mean::V
    strength::T
    degrees_of_freedom::T
    scale_matrix::M
end

function prior_state(m::MultivariateGaussianMeanCovarianceModel)
    NormalInverseWishartState(copy(m.prior_mean), m.prior_strength, m.degrees_of_freedom, copy(m.scale_matrix))
end

function predictive_distribution(::MultivariateGaussianMeanCovarianceModel, s::NormalInverseWishartState)
    d = length(s.mean)
    ν = s.degrees_of_freedom - d + 1
    scale = ((s.strength + 1) / (s.strength * ν)) * s.scale_matrix

    return MvTDist(ν, s.mean, Matrix(Symmetric(scale)))
end

function logpredictive(m::MultivariateGaussianMeanCovarianceModel, s::NormalInverseWishartState, x::AbstractVector)
    length(x) == length(s.mean) || throw(DimensionMismatch("Observation dimension does not match model"))

    return logpdf(predictive_distribution(m, s), x)
end

function update(::MultivariateGaussianMeanCovarianceModel, s::NormalInverseWishartState, x::AbstractVector)
    length(x) == length(s.mean) || throw(DimensionMismatch("Observation dimension does not match model"))

    k = s.strength + 1
    δ = x - s.mean

    return NormalInverseWishartState(s.mean + δ / k, k, s.degrees_of_freedom + 1, s.scale_matrix + (s.strength / k) * (δ * δ'))
end

"""
    BernoulliModel(; alpha=1.0, beta=1.0)

Model Boolean observations with a Beta prior on the success probability.
"""
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

"""
    PoissonModel(; shape=1.0, rate=1.0)

Model nonnegative integer count observations with a Gamma prior on the rate.
"""
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