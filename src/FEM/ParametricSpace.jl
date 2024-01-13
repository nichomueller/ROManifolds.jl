abstract type Realization end

get_parameters(r::Realization) = r.params

struct PRealization{P} <: Realization
  params::P
end

struct TransientPRealization{P,T} <: Realization
  params::P
  times::T
end

get_times(r::TransientPRealization) = r.times

function change_times!(
  r::TransientPRealization{P,T} where P,
  times::T
  ) where T

  r.times .= times
end

abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

abstract type AbstractParametricSpace end

struct ParametricSpace <: AbstractParametricSpace
  parametric_domain::AbstractVector
  sampling_style::SamplingStyle
  function ParametricSpace(
    parametric_domain::AbstractVector{<:AbstractVector},
    sampling=UniformSampling())

    new(parametric_domain,sampling)
  end
end

function generate_parameter(p::ParametricSpace)
  _value(d,::UniformSampling) = rand(Uniform(first(d),last(d)))
  _value(d,::UniformSampling) = rand(Normal(first(d),last(d)))
  [_value(d,p.sampling_style) for d = p.parametric_domain]
end

function realization(p::ParametricSpace;nparams=1)
  PRealization([generate_parameter(p) for i = 1:nparams])
end

struct TransientParametricSpace <: AbstractParametricSpace
  pspace::ParametricSpace
  temporal_domain::AbstractVector
end

function TransientParametricSpace(
  parametric_domain::AbstractVector{<:AbstractVector},
  temporal_domain::AbstractVector{<:Number},
  args...)

  pspace = ParametricSpace(parametric_domain,args...)
  TransientParametricSpace(pspace,temporal_domain)
end

function shift_temporal_domain!(p::TransientParametricSpace,δ::Number)
  p.times .+= δ
end

function realization(
  p::TransientParametricSpace;
  nparams=1,time_locations=eachindex(p.temporal_domain)
  )

  params = [generate_parameter(p) for i = 1:nparams]
  times = p.temporal_domain[time_locations]
  TransientPRealization(params,times)
end
