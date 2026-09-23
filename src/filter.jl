struct FilterSnapshot{S,T}
    runs::Vector{Int}
    logprobs::Vector{T}
    states::Vector{S}
    observation::Any
end

"""
    BOCPDDetector(model, hazard=ConstantHazard(100); <keyword arguments>)

Create an online BOCPD detector. The convention is `r_t == 0` when the
observation at time `t` starts a new segment. Keyword arguments include
`hazard`, `max_run_length`, `prune_threshold`, `pruning`, `history`, `lag`,
and `invalid_data`. Use `FixedLagHistory(L)` to retain data for fixed-lag
smoothing, or `FullHistory()` for the complete filtering history.
"""
mutable struct BOCPDDetector{M,H,P,S,T}
    model::M
    hazard::H
    pruning::P
    max_run_length::Int
    prune_threshold::T
    invalid_data::AbstractInvalidDataPolicy
    runs::Vector{Int}
    logprobs::Vector{T}
    states::Vector{S}
    time::Int
    changepoint_probability::T
    discarded_mass::T
    lag::Int
    snapshots::Vector{FilterSnapshot{S,T}}
    statuses::Vector{Symbol}
    fully_missing_count::Int
    partially_missing_count::Int
    rejected_invalid_count::Int
end

function BOCPDDetector(
    model::M, hazard=ConstantHazard(100);
    max_run_length=typemax(Int),
    prune_threshold=0.0,
    pruning=nothing,
    history::AbstractHistoryPolicy=NoHistory(),
    invalid_data::AbstractInvalidDataPolicy=RejectNaN()
) where {M<:AbstractObservationModel}
    max_run_length >= 0 || throw(ArgumentError("max_run_length must be nonnegative"))
    0 <= prune_threshold <= 1 || throw(ArgumentError("prune_threshold must lie in [0, 1]"))

    T = typeof(float(prune_threshold))
    S = typeof(prior_state(model))
    p = pruning === nothing ? (prune_threshold > 0 ? ProbabilityPruning(prune_threshold) : NoPruning()) : pruning
    effective_lag = history_lag(history)

    BOCPDDetector{M,typeof(hazard),typeof(p),S,T}(
        model, hazard, p, Int(max_run_length), T(prune_threshold), invalid_data, Int[], T[],
        S[], 0, zero(T), zero(T), effective_lag, FilterSnapshot{S,T}[], Symbol[], 0, 0, 0)
end

"""Convenience alias for `BOCPDDetector`."""
const Detector = BOCPDDetector

function _finish_step!(d, runs, raw, states, total, status, observation)
    keep = _keep_indices(d.pruning, runs, raw, total, d.max_run_length)

    isempty(keep) && throw(ArgumentError("Pruning removed every run-length hypothesis"))

    retained = logsumexp(raw[keep])
    d.discarded_mass = max(zero(eltype(d.logprobs)), one(eltype(d.logprobs)) -
        exp(retained - total))
    d.runs = runs[keep]
    d.logprobs = raw[keep] .- retained
    d.states = states[keep]
    d.time += 1

    push!(d.statuses, status)

    status === :missing && (d.fully_missing_count += 1)
    status === :partially_observed && (d.partially_missing_count += 1)

    if d.lag > 0
        push!(d.snapshots, FilterSnapshot(copy(d.runs), copy(d.logprobs), deepcopy(d.states), observation))

        d.lag < typemax(Int) && length(d.snapshots) > d.lag + 1 && popfirst!(d.snapshots)
    end

    return d
end

function _update_missing_one!(d::BOCPDDetector)
    T = eltype(d.logprobs)

    if d.time == 0
        runs = [0]
        raw = [zero(T)]
        states = [prior_state(d.model)]
        total = zero(T)
    else
        reset_terms = similar(d.logprobs)
        growth_runs = Int[]
        growth_raw = T[]
        growth_states = Vector{typeof(d.states[1])}()

        for j in eachindex(d.runs)
            reset_terms[j] = d.logprobs[j] + _hazard_log(d.hazard, d.runs[j], d.time)
            survival = _survival_log(d.hazard, d.runs[j], d.time)

            if survival > -Inf && d.runs[j] + 1 <= d.max_run_length
                push!(growth_runs, d.runs[j] + 1)
                push!(growth_raw, d.logprobs[j] + survival)
                push!(growth_states, d.states[j])
            end
        end

        runs = [0; growth_runs]
        raw = [logsumexp(reset_terms); growth_raw]
        states = [prior_state(d.model); growth_states]
        total = logsumexp(raw)
    end

    d.changepoint_probability = exp(raw[1] - total)

    _finish_step!(d, runs, raw, states, total, :missing, missing)
end

"""
    update!(detector::BOCPDDetector, ::Missing; delta_t=1)

Advance time without observation evidence or posterior-state updates.
"""
function update!(d::BOCPDDetector, ::Missing; delta_t::Integer=1)
    delta_t >= 1 || throw(ArgumentError("delta_t must be at least one"))

    for _ in 1:delta_t
        _update_missing_one!(d)
    end

    return d
end

function _update_observed_one!(d::BOCPDDetector, x, status::Symbol)
    T = eltype(d.logprobs)

    if d.time == 0
        raw = [T(logpredictive(d.model, prior_state(d.model), x))]
        runs = [0]
        states = [reset(d.model, x)]
        total = raw[1]
    else
        reset_terms = similar(d.logprobs)
        growth_runs = Int[]
        growth_raw = T[]
        growth_states = Vector{typeof(d.states[1])}()
        reset_lp = T(logpredictive(d.model, prior_state(d.model), x))

        for j in eachindex(d.runs)
            lp = logpredictive(d.model, d.states[j], x)
            reset_terms[j] = d.logprobs[j] + _hazard_log(d.hazard, d.runs[j], d.time) + reset_lp
            survival = _survival_log(d.hazard, d.runs[j], d.time)

            if survival > -Inf && d.runs[j] + 1 <= d.max_run_length
                push!(growth_runs, d.runs[j] + 1)
                push!(growth_raw, d.logprobs[j] + survival + lp)
                push!(growth_states, update(d.model, d.states[j], x))
            end
        end

        runs = [0; growth_runs]
        raw = [logsumexp(reset_terms); growth_raw]
        states = [reset(d.model, x); growth_states]
        total = logsumexp(raw)
    end

    d.changepoint_probability = exp(raw[1] - total)

    _finish_step!(d, runs, raw, states, total, status, x)
end

function _validate_scalar!(d::BOCPDDetector, x)
    if x isa Real && !isfinite(x)
        d.rejected_invalid_count += 1

        throw(DomainError(x, "non-finite observations are not supported; use missing for unavailable data"))
    end

    return x
end

"""
    update!(detector::BOCPDDetector, observation::Real; delta_t=1)

Process a scalar observation, optionally inserting `delta_t - 1` missing steps
first.
"""
function update!(d::BOCPDDetector, x::Real; delta_t::Integer=1)
    delta_t >= 1 || throw(ArgumentError("delta_t must be at least one"))

    if isnan(x) && d.invalid_data isa TreatNaNAsMissing
        return update!(d, missing; delta_t)
    end

    _validate_scalar!(d, x)

    for _ in 1:(delta_t-1)
        _update_missing_one!(d)
    end

    _update_observed_one!(d, x, :observed)
end

function _prepare_vector(d::BOCPDDetector, x::AbstractVector)
    values = x

    if d.invalid_data isa TreatNaNAsMissing && any(value -> value isa Real && isnan(value), x)
        values = Vector{Union{Missing,eltype(x)}}(undef, length(x))

        for i in eachindex(x)
            value = x[i]
            values[i] = value isa Real && isnan(value) ? missing : value
        end
    end

    for value in values
        value isa Real && !isfinite(value) &&
            (d.rejected_invalid_count += 1; throw(DomainError(value, "Non-finite observations are not supported")))
    end

    return values
end

"""
    update!(detector::BOCPDDetector, observation::AbstractVector; delta_t=1)

Process a vector observation, including fully or partially missing vectors.
"""
function update!(d::BOCPDDetector, x::AbstractVector; delta_t::Integer=1)
    delta_t >= 1 || throw(ArgumentError("delta_t must be at least one"))

    values = _prepare_vector(d, x)
    missing_indices = findall(ismissing, values)

    if isempty(missing_indices)
        for _ in 1:(delta_t-1)
            _update_missing_one!(d)
        end

        return _update_observed_one!(d, values, :observed)

    elseif length(missing_indices) == length(values)
        return update!(d, missing; delta_t)
    end

    supports_partial_observations(d.model) || throw(ArgumentError("$(typeof(d.model)) does not support partially observed vectors"))

    indices = collect(Int, setdiff(collect(eachindex(values)), missing_indices))
    observed_values = collect(skipmissing(values))
    subset = ObservedSubset(observed_values, indices, length(values))

    for _ in 1:(delta_t-1)
        _update_missing_one!(d)
    end

    _update_observed_one!(d, subset, :partially_observed)
end

function update!(d::BOCPDDetector, x; delta_t::Integer=1)
    delta_t >= 1 || throw(ArgumentError("delta_t must be at least one"))

    _validate_scalar!(d, x)

    for _ in 1:(delta_t-1)
        _update_missing_one!(d)
    end

    _update_observed_one!(d, x, :observed)
end

"""
    runlength_probs(detector::BOCPDDetector) -> Vector

Return current posterior probabilities indexed by explicit run length.
"""
runlength_probs(d::BOCPDDetector) = exp.(d.logprobs)

"""
    runlength_probabilities(detector::BOCPDDetector) -> Vector

Return current posterior probabilities indexed by explicit run length.

See also [`runlength_probs`](@ref).
"""
runlength_probabilities(d::BOCPDDetector) = runlength_probs(d)

"""
    log_runlength_probs(detector::BOCPDDetector) -> Vector

Return the current log run-length posterior.
"""
log_runlength_probs(d::BOCPDDetector) = copy(d.logprobs)

"""
    most_likely_runlength(detector::BOCPDDetector) -> Int

Return the run length with greatest current posterior probability.
"""
most_likely_runlength(d::BOCPDDetector) = d.runs[argmax(d.logprobs)]

"""
    current_changepoint_probability(detector::BOCPDDetector) -> Real

Return the online probability that the current observation starts a segment.
"""
current_changepoint_probability(d::BOCPDDetector) = d.changepoint_probability

"""
    time_index(detector::BOCPDDetector) -> Int

Return the number of processed time points.
"""
time_index(d::BOCPDDetector) = d.time

"""
    predictive_distribution(detector::BOCPDDetector) -> Distribution

Return the predictive distribution for the most likely current state.
"""
function predictive_distribution(d::BOCPDDetector)
    predictive_distribution(d.model, d.states[argmax(d.logprobs)])
end

"""
    observation_status(detector::BOCPDDetector, t::Integer) -> Symbol

Return `:observed`, `:missing`, or `:partially_observed` for time `t`.
"""
observation_status(d::BOCPDDetector, t::Integer) = d.statuses[t]

"""
    ismissingobservation(detector::BOCPDDetector, t::Integer) -> Bool

Return whether time `t` was fully missing.
"""
ismissingobservation(d::BOCPDDetector, t::Integer) = d.statuses[t] === :missing