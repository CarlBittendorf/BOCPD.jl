"""Supertype for strategies that remove run-length hypotheses."""
abstract type AbstractPruning end

"""Retain every hypothesis."""
struct NoPruning <: AbstractPruning end

"""Retain only run lengths no greater than `maximum`."""
struct MaxRunLength <: AbstractPruning
    maximum::Int
end

"""Remove hypotheses whose normalized probability is below `threshold`."""
struct ProbabilityPruning <: AbstractPruning
    threshold::Float64
end

"""Retain at most the `k` most probable hypotheses."""
struct TopKPruning <: AbstractPruning
    k::Int
end

function MaxRunLength(n::Integer)
    n >= 0 ? MaxRunLength(Int(n)) :
    throw(ArgumentError("Maximum run length must be nonnegative"))
end

function ProbabilityPruning(x::Real)
    0 <= x <= 1 ? ProbabilityPruning(Float64(x)) :
    throw(ArgumentError("Pruning threshold must lie in [0, 1]"))
end

function TopKPruning(k::Integer)
    k >= 1 ? TopKPruning(Int(k)) : throw(ArgumentError("Top-k must be positive"))
end

_keep_indices(::NoPruning, runs, raw, total, max_run_length) = collect(eachindex(runs))

function _keep_indices(strategy::MaxRunLength, runs, raw, total, max_run_length)
    [i for i in eachindex(runs) if runs[i] <= strategy.maximum]
end

function _keep_indices(strategy::ProbabilityPruning, runs, raw, total, max_run_length)
    [i for i in eachindex(runs) if exp(raw[i] - total) >= strategy.threshold]
end

function _keep_indices(strategy::TopKPruning, runs, raw, total, max_run_length)
    n = min(strategy.k, length(runs))

    sortperm(raw; rev=true)[1:n]
end