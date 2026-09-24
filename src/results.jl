"""
    BOCPDResult

Immutable batch output from a BOCPD fit or detector continuation. It stores the
online changepoint probabilities, retained run-length summaries, observation
status metadata, and pruning diagnostics for the processed sequence.

The default `show` output is intentionally brief; `show(io, MIME("text/plain"), result)`
provides a readable summary of the main batch-level metadata.
"""
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

function _result_history_label(r::BOCPDResult)
    if r.runlength_history !== nothing
        return "FullHistory()"
    elseif !isempty(r.snapshots)
        return "FixedLagHistory($(max(0, length(r.snapshots) - 1)))"
    else
        return "not retained"
    end
end

function _result_max_delay(r::BOCPDResult)
    if r.runlength_history !== nothing
        return "full history"
    elseif !isempty(r.snapshots)
        return max(0, length(r.snapshots) - 1)
    else
        return "unavailable"
    end
end

function _result_pruning_label(r::BOCPDResult)
    isempty(r.discarded_mass) && return "none"
    all(iszero, r.discarded_mass) && return "none"
    return "approximate"
end

function Base.show(io::IO, result::BOCPDResult)
    times = time_index(result)
    history = _result_history_label(result)
    max_delay = _result_max_delay(result)
    has_missing = result.fully_missing_count > 0
    pruning = _result_pruning_label(result)

    print(io, "BOCPDResult(times=")
    print(io, times)
    print(io, ", history=")
    print(io, history)
    if max_delay isa Integer
        print(io, ", max delay=")
        print(io, max_delay)
    end
    if has_missing
        print(io, ", missing=")
        print(io, result.fully_missing_count)
    end
    if pruning != "none"
        print(io, ", pruning=")
        print(io, pruning)
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::BOCPDResult)
    println(io, "BOCPDResult")
    println(io, "  time points:            ", time_index(result))
    if !isempty(result.changepoint_probabilities)
        println(io, "  latest changepoint prob.: ", result.changepoint_probabilities[end])
    end
    if !isempty(result.most_likely_runlengths)
        println(io, "  latest MAP run length:  ", result.most_likely_runlengths[end])
    end
    println(io, "  retained history:       ", _result_history_label(result))
    delay = _result_max_delay(result)
    println(io, "  maximum delay:          ", delay)
    println(io, "  missing observations:   ", result.fully_missing_count)
    println(io, "  pruning:               ", _result_pruning_label(result))
end

"""
    fit(model, observations; kwargs...)

Process an iterable of scalar, vector, or `missing` observations. The returned
`BOCPDResult` stores changepoint probabilities and diagnostics. By default, full
run-length history is retained so delayed changepoint probabilities and fixed-lag
smoothing are available without extra configuration. Use `NoHistory()` or
`FixedLagHistory(L)` to limit retention.
"""
function fit(
    model::AbstractObservationModel, observations;
    hazard=ConstantHazard(100),
    max_run_length=typemax(Int), prune_threshold=0.0, pruning=nothing,
    history::AbstractHistoryPolicy=FullHistory(),
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

Return the online changepoint probabilities, or delayed probabilities for each
observation time. With `delay = 0`, the function returns the in-sample online
probabilities `P(c_t = true | x_1:t)`. With a positive delay `L`, it returns
`P(c_t = true | x_1:min(T, t + L))`, where `c_t` means the observation at time
`t` starts a new segment. This requires retained fixed-lag or full history, and
it is not the same as the later terminal event `P(r_(t+L) = L | x_1:(t+L))`.

The return value is a fresh vector of the same length as the batch result. If the
requested delay exceeds the retained history, an `ArgumentError` is thrown with
both the requested delay and the maximum available delay.
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