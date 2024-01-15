abstract type Realization end

struct PRealization{P} <: Realization
  params::Base.RefValue{P}
  function PRealization(params::P) where P
    new{P}(Ref(params))
  end
end

struct TransientPRealization{P,T} <: Realization
  params::Base.RefValue{P}
  times::Base.RefValue{T}
  function TransientPRealization(params::P,times::T) where {P,T}
    new{P,T}(Ref(params),Ref(times))
  end
end

get_parameters(r::Realization) = r.params[]
num_parameters(r::Realization) = length(get_parameters(r))
get_times(r::TransientPRealization) = r.times[]
num_times(r::TransientPRealization) = length(get_times(r))
function get_initial_time(r::TransientPRealization{P,T} where P) where T<:AbstractVector
  first(get_times(r))
end
function get_final_time(r::TransientPRealization{P,T} where P) where T<:AbstractVector
  last(get_times(r))
end
function get_midpoint_time(r::TransientPRealization{P,T} where P) where T<:AbstractVector
  (get_final_time(r) + get_initial_time(r)) / 2
end
function get_delta_time(r::TransientPRealization{P,T} where P) where T<:AbstractVector
  (get_final_time(r) - get_initial_time(r)) / num_times(r)
end

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

  r.times[] = time
end

abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

abstract type AbstractParametricSpace{R} <: AbstractSet{R} end

struct ParametricSpace <: AbstractParametricSpace{PRealization}
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

struct TransientParametricSpace <: AbstractParametricSpace{TransientPRealization}
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

  params = [generate_parameter(p.pspace) for i = 1:nparams]
  times = p.temporal_domain[time_locations]
  TransientPRealization(params,times)
end
