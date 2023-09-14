abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

struct PSpace
  domain::Vector{Vector{Float}}
  sampling_style::SamplingStyle
  function PSpace(d::Vector{Vector{Float}},s::SamplingStyle)
    new([Float.(di) for di = d],s)
  end
end

function realization(d::Vector{<:Number},::UniformSampling)
  rand(Uniform(first(d),last(d)))
end

function realization(d::Vector{<:Number},::NormalSampling)
  rand(Normal(first(d),last(d)))
end

function realization(pspace::PSpace)
  [realization(d,pspace.sampling_style) for d = pspace.domain]
end

realization(pspace::PSpace,n::Int) = Table([realization(pspace) for _ = 1:n])
