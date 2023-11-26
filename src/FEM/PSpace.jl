abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

struct PSpace
  domain::Table
  sampling_style::SamplingStyle
end

function PSpace(d::Vector{Vector{Float}},sampling::SamplingStyle=UniformSampling())
  new(Table(d),sampling)
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
