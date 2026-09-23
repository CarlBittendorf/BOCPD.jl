"""Immutable batch BOCPD output, including filtering, status, and pruning diagnostics."""
struct BOCPDResult{M,H,T,S}
    model::M
    hazard::H
    changepoint_probabilities::Vector{T}
    most_likely_runlengths::Vector{Int}
    runlength_history::Union{Nothing,Vector{Vector{T}}}
    runlength_values::Union{Nothing,Vector{Vector{Int}}}
    predictive_log_scores::Union{Nothing,Vector{T}}
    discarded_mass::Vector{T}
    observations::Vector{Any}
    snapshots::Vector{FilterSnapshot{S,T}}
    statuses::Vector{Symbol}
    fully_missing_count::Int
    partially_missing_count::Int
    rejected_invalid_count::Int
end

"""
    fit(model, observations; kwargs...)

Process an iterable of scalar, vector, or `missing` observations. The returned
`BOCPDResult` stores changepoint probabilities and diagnostics. Full run-length
history is stored only with `FullHistory()`; fixed-lag smoothing requires
`FixedLagHistory(L)` or `FullHistory()`.
"""
function fit(
    model::AbstractObservationModel, observations;
    hazard=ConstantHazard(100),
    max_run_length=typemax(Int), prune_threshold=0.0, pruning=nothing,
    history::AbstractHistoryPolicy=NoHistory(),
    store_predictive_log_scores=false,
    invalid_data::AbstractInvalidDataPolicy=RejectNaN()
)
    detector = BOCPDDetector(model, hazard; max_run_length, prune_threshold, pruning, history, invalid_data)
    cps = Float64[]
    modes = Int[]
    lost = Float64[]
    scores = Float64[]
    hist = Vector{Vector{Float64}}()
    vals = Vector{Vector{Int}}()
    obs = Any[]

    for x in observations
        push!(obs, x)
        update!(detector, x)
        push!(cps, detector.changepoint_probability)
        push!(modes, most_likely_runlength(detector))
        push!(lost, detector.discarded_mass)

        if store_predictive_log_scores
            score_state = detector.states[argmax(detector.logprobs)]

            push!(scores, x isa Missing ? 0.0 : logpredictive(model, score_state, x))
        end

        history isa FullHistory && (push!(hist, runlength_probs(detector)); push!(vals, copy(detector.runs)))
    end

    BOCPDResult(model, hazard, cps, modes, history isa FullHistory ? hist : nothing,
        history isa FullHistory ? vals : nothing,
        store_predictive_log_scores ? scores : nothing, lost, obs,
        copy(detector.snapshots), copy(detector.statuses), detector.fully_missing_count,
        detector.partially_missing_count, detector.rejected_invalid_count)
end

"""
    fit(detector::BOCPDDetector, observations; store_predictive_log_scores=false)

Continue `detector` over an iterable of observations and return a
`BOCPDResult`.
"""
function fit(detector::BOCPDDetector, observations; store_predictive_log_scores=false)
    cps = Float64[]
    modes = Int[]
    lost = Float64[]
    scores = Float64[]
    hist = Vector{Vector{Float64}}()
    vals = Vector{Vector{Int}}()
    obs = Any[]

    for x in observations
        push!(obs, x)
        update!(detector, x)
        push!(cps, detector.changepoint_probability)
        push!(modes, most_likely_runlength(detector))
        push!(lost, detector.discarded_mass)

        if store_predictive_log_scores
            state = detector.states[argmax(detector.logprobs)]

            push!(scores, x isa Missing ? 0.0 : logpredictive(detector.model, state, x))
        end

        detector.lag == typemax(Int) && (push!(hist, runlength_probs(detector)); push!(vals, copy(detector.runs)))
    end

    full = detector.lag == typemax(Int)

    BOCPDResult(detector.model, detector.hazard, cps, modes, full ? hist : nothing,
        full ? vals : nothing, store_predictive_log_scores ? scores : nothing, lost, obs,
        copy(detector.snapshots), copy(detector.statuses), detector.fully_missing_count,
        detector.partially_missing_count, detector.rejected_invalid_count)
end

"""
    changepoint_probabilities(result::BOCPDResult; delay=0)

Return online or delayed changepoint probabilities for every retained time
point.
"""
function changepoint_probabilities(r::BOCPDResult; delay=0)
    [changepoint_probability(r, t; delay) for t in eachindex(r.changepoint_probabilities)]
end

"""
    runlength_probs(result::BOCPDResult, t::Integer)

Return the retained run-length posterior at batch time `t`.
"""
function runlength_probs(r::BOCPDResult, t::Integer)
    r.runlength_history === nothing ?
    throw(ArgumentError("run-length history was not retained")) : r.runlength_history[t]
end

"""Return the retained run-length posterior at batch time `t`."""
runlength_probabilities(r::BOCPDResult, t::Integer) = runlength_probs(r, t)

"""Return the number of observations processed in a batch result."""
time_index(r::BOCPDResult) = length(r.changepoint_probabilities)

"""Return the observation status at time `t`."""
observation_status(r::BOCPDResult, t::Integer) = r.statuses[t]

"""Return whether time `t` was fully missing."""
ismissingobservation(r::BOCPDResult, t::Integer) = r.statuses[t] === :missing