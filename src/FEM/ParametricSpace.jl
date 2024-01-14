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
get_initial_time(r::TransientPRealization) = first(get_times(r))
get_midpoint_time(r::TransientPRealization) = (get_initial_time(r) + get_final_time(r)) / 2
get_final_time(r::TransientPRealization) = last(get_times(r))
get_delta_time(r::TransientPRealization) = r.times[2] - r.times[1]

function get_at_time(r::TransientPRealization,time=:initial)
  params = get_parameters(r)
  if time == :initial
    TransientPRealization(params,get_initial_time(r))
  elseif time == :midpoint
    TransientPRealization(params,get_midpoint_time(r))
  elseif time == :final
    TransientPRealization(params,get_final_time(r))
  else
    @notimplemented
  end
end

function change_time!(
  r::TransientPRealization{P,T} where P,
  time::T
  ) where T

  r.times .= time
end

abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

abstract type AbstractParametricSpace <: AbstractSet{Realization} end

struct ParametricSpace <: AbstractParametricSpace
  parametric_domain::AbstractVector
  sampling_style::SamplingStyle
  function ParametricSpace(
    parametric_domain::AbstractVector{<:AbstractVector},
    sampling=UniformSampling())

    new(parametric_domain,sampling)
  end
end

function Base.show(io::IO,::MIME"text/plain",p::ParametricSpace)
  msg = "Set of parameters in $(p.parametric_domain), sampled with $(p.sampling_style)"
  println(io,msg)
end

function generate_parameter(p::ParametricSpace)
  _value(d,::UniformSampling) = rand(Uniform(first(d),last(d)))
  _value(d,::NormalSampling) = rand(Normal(first(d),last(d)))
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

function Base.show(io::IO,::MIME"text/plain",p::TransientParametricSpace)
  msg = "Set of tuples (p,t) in $(p.pspace.parametric_domain) × $(p.temporal_domain)"
  println(io,msg)
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
