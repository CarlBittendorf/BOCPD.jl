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
        return "not retained"
    end
end

function _result_pruning_label(r::BOCPDResult)
    isempty(r.discarded_mass) && return "none"
    all(iszero, r.discarded_mass) && return "none"
    return "approximate"
end

function _format_float(value)
    buffer = IOBuffer()
    show(IOContext(buffer, :compact => true), value)
    return String(take!(buffer))
end

function _observation_dimension(result::BOCPDResult)
    for value in result.observations
        value === missing && continue
        value isa AbstractVector && return length(value)
        return 1
    end
    return nothing
end

function Base.show(io::IO, result::BOCPDResult)
    n = time_index(result)
    history = _result_history_label(result)
    max_delay = _result_max_delay(result)
    has_missing = result.fully_missing_count > 0
    pruning = _result_pruning_label(result)
    cp = isempty(result.changepoint_probabilities) ? nothing : result.changepoint_probabilities[end]
    mode = isempty(result.most_likely_runlengths) ? nothing : result.most_likely_runlengths[end]

    print(io, "BOCPDResult(")
    print(io, n)
    print(io, " ")
    print(io, n == 1 ? "time point" : "time points")
    if history != "not retained"
        print(io, ", history=")
        print(io, history)
    end
    if max_delay isa Integer
        print(io, ", max_delay=")
        print(io, max_delay)
    end
    if cp !== nothing
        print(io, ", latest changepoint probability=")
        print(io, _format_float(cp))
    end
    if mode !== nothing
        print(io, ", MAP run length=")
        print(io, mode)
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
    println(io, "  sequence:")
    println(io, "    time points:                  ", time_index(result))
    if result.fully_missing_count > 0
        println(io, "    missing observations:         ", result.fully_missing_count)
    end
    dimension = _observation_dimension(result)
    if dimension !== nothing
        println(io, "    observation dimension:         ", dimension)
    end

    if !isempty(result.changepoint_probabilities)
        println(io, "  latest estimate:")
        println(io, "    changepoint probability:      ", _format_float(result.changepoint_probabilities[end]))
        if !isempty(result.most_likely_runlengths)
            println(io, "    MAP run length:               ", result.most_likely_runlengths[end])
        end
    end

    println(io, "  history:")
    println(io, "    retention:                   ", _result_history_label(result))
    println(io, "    maximum delay:               ", _result_max_delay(result))

    pruning = _result_pruning_label(result)
    if pruning != "none" && !isempty(result.discarded_mass)
        println(io, "  approximation:")
        println(io, "    pruning:                     ", pruning)
        println(io, "    discarded mass:              ", _format_float(last(result.discarded_mass)))
    else
        println(io, "  approximation:")
        println(io, "    pruning:                     ", pruning)
    end
end

"""
    fit(model, observations; kwargs...)

Process an iterable of scalar, vector, or `missing` observations. The returned
`BOCPDResult` stores changepoint probabilities and diagnostics. By default, full
run-length history is retained so delayed changepoint probabilities and fixed-lag
smoothing are available without extra configuration. Use `NoHistory()` or
`FixedLagHistory(L)` to limit retention. Set `store_predictive_log_scores=true`
to retain the prequential log evidence `log p(x_t | x_1:t-1)`, marginalized over
the previous run-length posterior. Fully missing observations have score zero.
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
        if store_predictive_log_scores
            push!(scores, _prequential_log_score(detector, x))
        end

        update!(detector, x)
        push!(cps, detector.changepoint_probability)
        push!(modes, most_likely_runlength(detector))
        push!(lost, detector.discarded_mass)

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
`BOCPDResult`. When `store_predictive_log_scores=true`, scores are prequential
log evidence marginalized over the pre-update run-length posterior.
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
        if store_predictive_log_scores
            push!(scores, _prequential_log_score(detector, x))
        end

        update!(detector, x)
        push!(cps, detector.changepoint_probability)
        push!(modes, most_likely_runlength(detector))
        push!(lost, detector.discarded_mass)

        detector.lag == typemax(Int) && (push!(hist, runlength_probs(detector)); push!(vals, copy(detector.runs)))
    end

    full = detector.lag == typemax(Int)

    BOCPDResult(detector.model, detector.hazard, cps, modes, full ? hist : nothing,
        full ? vals : nothing, store_predictive_log_scores ? scores : nothing, lost, obs,
        copy(detector.snapshots), copy(detector.statuses), detector.fully_missing_count,
        detector.partially_missing_count, detector.rejected_invalid_count)
end

function _prequential_log_score(d::BOCPDDetector, ::Missing)
    0.0
end

function _prequential_log_score(d::BOCPDDetector, x::Real)
    isnan(x) && d.invalid_data isa TreatNaNAsMissing && return 0.0
    return _log_predictive_evidence(d, x)
end

function _prequential_log_score(d::BOCPDDetector, x::AbstractVector)
    values = _prepare_vector(d, x)
    missing_indices = findall(ismissing, values)

    if isempty(missing_indices)
        return _log_predictive_evidence(d, values)
    elseif length(missing_indices) == length(values)
        return 0.0
    end

    supports_partial_observations(d.model) ||
        throw(ArgumentError("$(typeof(d.model)) does not support partially observed vectors"))

    indices = collect(Int, setdiff(collect(eachindex(values)), missing_indices))
    observation = ObservedSubset(collect(skipmissing(values)), indices, length(values))
    return _log_predictive_evidence(d, observation)
end

_prequential_log_score(d::BOCPDDetector, x) = _log_predictive_evidence(d, x)

function _log_predictive_evidence(d::BOCPDDetector, observation)
    if d.time == 0
        return logpredictive(d.model, prior_state(d.model), observation)
    end

    terms = similar(d.logprobs)
    for index in eachindex(d.logprobs)
        terms[index] = d.logprobs[index] + logpredictive(d.model, d.states[index], observation)
    end

    return logsumexp(terms)
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