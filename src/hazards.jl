"""
    CustomHazard(f; depends_on=:run)

Wrap a user-defined hazard probability function. `f` must return a finite
number in `[0, 1]`. Set `depends_on` to `:run` for `f(run)`, `:time` for
`f(time)`, or `:run_time` for `f(run, time)`. The time argument is the number
of observations already processed, so the first update uses `time == 0`.

Examples:

```julia
CustomHazard(r -> clamp(0.001 + 0.0001r, 0, 1))
CustomHazard(t -> t > 100 ? 0.2 : 0.01; depends_on=:time)
CustomHazard((r, t) -> clamp(0.01 + 0.001r + 0.0001t, 0, 1); depends_on=:run_time)
```
"""
struct CustomHazard{F}
    f::F
    depends_on::Symbol

    function CustomHazard(f::F, depends_on::Symbol) where {F}
        depends_on in (:run, :time, :run_time) ||
            throw(ArgumentError("depends_on must be :run, :time, or :run_time"))

        return new{F}(f, depends_on)
    end
end

CustomHazard(f::F; depends_on=:run) where {F} = CustomHazard(f, depends_on)

function (h::CustomHazard)(run::Integer, time::Integer)
    h.depends_on === :run ? h.f(run) : h.depends_on === :time ? h.f(time) : h.f(run, time)
end

"""Constant hazard represented by an expected segment length."""
struct ConstantHazard
    probability::Float64
end

"""Construct `H(r) = 1 / expected_segment_length`."""
function ConstantHazard(expected_segment_length::Integer)
    expected_segment_length > 0 ||
        throw(ArgumentError("expected segment length must be positive"))

    return ConstantHazard(1 / Float64(expected_segment_length))
end

(h::CustomHazard)(run::Integer) = h(run, 0)
(h::ConstantHazard)(::Integer) = h.probability

"""A geometric-segment hazard with the supplied changepoint probability."""
struct GeometricHazard{T<:Real}
    probability::T

    function GeometricHazard(p::T) where {T<:Real}
        isfinite(p) && 0 <= p <= 1 ||
            throw(ArgumentError("hazard probability must lie in [0, 1]"))

        return new{T}(p)
    end
end

"""Evaluate the constant geometric hazard at any run length."""
(h::GeometricHazard)(::Integer) = h.probability

function _hazard_probability(hazard, run::Int, time::Int)
    hazard isa CustomHazard ? hazard(run, time) : hazard(run)
end

function _hazard_log(hazard, run::Int, time::Int=0)
    p = _hazard_probability(hazard, run, time)

    isfinite(p) && 0 <= p <= 1 ||
        throw(ArgumentError("hazard($(run)) must be finite and lie in [0, 1], got $(p)"))

    return p == 0 ? -Inf : log(p)
end

function _survival_log(hazard, run::Int, time::Int=0)
    p = _hazard_probability(hazard, run, time)

    isfinite(p) && 0 <= p <= 1 ||
        throw(ArgumentError("hazard($(run)) must be finite and lie in [0, 1], got $(p)"))

    return p == 1 ? -Inf : log1p(-p)
end