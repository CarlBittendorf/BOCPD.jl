function _snapshot_at(r::BOCPDResult, t::Int)
    t >= 1 && t <= length(r.observations) || throw(BoundsError(r.observations, t))

    first_time = length(r.observations) - length(r.snapshots) + 1
    i = t - first_time + 1

    1 <= i <= length(r.snapshots) || throw(ArgumentError("History for time $(t) was not retained"))

    return r.snapshots[i]
end

function _backward_beta(r::BOCPDResult, t::Int, finish::Int)
    next_snapshot = _snapshot_at(r, finish)
    beta = zeros(Float64, length(next_snapshot.runs))

    for u in finish:-1:(t+1)
        previous = _snapshot_at(r, u - 1)
        reset_index = findfirst(==(0), next_snapshot.runs)
        parent_beta = fill(-Inf, length(previous.runs))
        reset_lp = logpredictive(r.model, prior_state(r.model), r.observations[u])

        for j in eachindex(previous.runs)
            lp = logpredictive(r.model, previous.states[j], r.observations[u])
            reset_log = _hazard_log(r.hazard, previous.runs[j], u - 1) + reset_lp
            growth_log = _survival_log(r.hazard, previous.runs[j], u - 1) + lp
            growth_index = findfirst(==(previous.runs[j] + 1), next_snapshot.runs)
            terms = Float64[]
            reset_index !== nothing && push!(terms, reset_log + beta[reset_index])
            growth_index !== nothing && push!(terms, growth_log + beta[growth_index])
            parent_beta[j] = logsumexp(terms)
        end

        beta = parent_beta
        next_snapshot = previous
    end

    return beta
end

"""
    changepoint_probability(result, t; delay=0)

Return `P(c_t=true | x_1:min(T,t+delay))`, where `c_t` means that the
observation at time `t` starts a new segment. A positive delay requires
retained lag or full history and uses a forward-backward transition pass.
"""
function changepoint_probability(r::BOCPDResult, t::Integer; delay=0)
    1 <= t <= length(r.changepoint_probabilities) || throw(BoundsError(r.changepoint_probabilities, t))
    delay >= 0 || throw(ArgumentError("delay must be nonnegative"))

    delay == 0 && return r.changepoint_probabilities[t]

    if r.runlength_history !== nothing
        max_delay = typemax(Int)
    elseif isempty(r.snapshots)
        max_delay = 0
    else
        max_delay = max(0, length(r.snapshots) - 1)
    end

    if delay > max_delay
        msg = "Requested delay $(delay) exceeds the maximum available delay $(max_delay); " *
              "retain more history with fit(...; history=FixedLagHistory($(delay))) or history=FullHistory()."
        throw(ArgumentError(msg))
    end

    finish = min(length(r.observations), t + delay)

    finish > t || return r.changepoint_probabilities[t]

    beta = _backward_beta(r, t, finish)
    after_t = _snapshot_at(r, t)
    reset_index = findfirst(==(0), after_t.runs)

    if t == 1
        return 1.0
    end

    previous = _snapshot_at(r, t - 1)
    reset_terms = Float64[]
    all_terms = Float64[]
    reset_lp = logpredictive(r.model, prior_state(r.model), r.observations[t])

    for j in eachindex(previous.runs)
        lp = logpredictive(r.model, previous.states[j], r.observations[t])
        reset = previous.logprobs[j] + _hazard_log(r.hazard, previous.runs[j], t - 1) + reset_lp + beta[reset_index]
        growth = previous.logprobs[j] + _survival_log(r.hazard, previous.runs[j], t - 1) + lp
        growth_index = findfirst(==(previous.runs[j] + 1), _snapshot_at(r, t).runs)

        push!(reset_terms, reset)
        push!(all_terms, reset)

        growth_index !== nothing && push!(all_terms, growth + beta[growth_index])
    end

    exp(logsumexp(reset_terms) - logsumexp(all_terms))
end

"""
    changepoint_probabilities(result::BOCPDResult, query::FixedLag)

Return changepoint probabilities using the fixed-lag query in `query`.
"""
function changepoint_probabilities(r::BOCPDResult, query::FixedLag)
    changepoint_probabilities(r; delay=query.lag)
end

"""
    changepoint_probability(result::BOCPDResult, t::Integer, query::FixedLag)

Return the fixed-lag changepoint probability for time `t`.
"""
function changepoint_probability(r::BOCPDResult, t::Integer, query::FixedLag)
    changepoint_probability(r, t; delay=query.lag)
end

"""
    delayed_changepoint_probability(detector::BOCPDDetector, delay::Integer)

Return the delayed changepoint probability for `detector`. Positive delays
require batch history and therefore raise an `ArgumentError` for an online
detector.
"""
function delayed_changepoint_probability(d::BOCPDDetector, delay::Integer)
    delay >= 0 || throw(ArgumentError("delay must be nonnegative"))

    delay == 0 && return current_changepoint_probability(d)

    throw(ArgumentError("Online delayed smoothing requires batch history; use fit(...; history=FixedLagHistory($(delay)))"))
end