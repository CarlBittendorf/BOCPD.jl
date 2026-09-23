"""Supertype for retained filtering-history policies."""
abstract type AbstractHistoryPolicy end

"""Do not retain filtering snapshots."""
struct NoHistory <: AbstractHistoryPolicy end

"""Retain every filtering snapshot."""
struct FullHistory <: AbstractHistoryPolicy end

"""Retain the snapshots needed to smooth a fixed lag."""
struct FixedLagHistory{I<:Integer} <: AbstractHistoryPolicy
    lag::I

    function FixedLagHistory(lag::I) where {I<:Integer}
        lag >= 0 || throw(ArgumentError("Fixed lag must be nonnegative"))

        return new{I}(lag)
    end
end

"""Describe how invalid floating-point observations are handled."""
abstract type AbstractInvalidDataPolicy end

"""Reject NaN and infinite observations."""
struct RejectNaN <: AbstractInvalidDataPolicy end

"""Interpret NaN as Julia's `missing` value."""
struct TreatNaNAsMissing <: AbstractInvalidDataPolicy end

"""Request a fixed-lag changepoint probability."""
struct FixedLag{I<:Integer}
    lag::I

    function FixedLag(lag::I) where {I<:Integer}
        lag >= 0 || throw(ArgumentError("Fixed lag must be nonnegative"))

        return new{I}(lag)
    end
end

history_lag(::NoHistory) = 0
history_lag(::FullHistory) = typemax(Int)
history_lag(policy::FixedLagHistory) = policy.lag