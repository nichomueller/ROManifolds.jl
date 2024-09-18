abstract type TimeCombination{O} <: Function end

struct ThetaMethodCombination <: TimeCombination{2}
  θ::Float64
end

(c::ThetaMethodCombination)(α::AbstractArray,αprev::AbstractArray) = c.θ*α+(1-c.θ)*αprev
