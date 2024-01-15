abstract type Realization end
abstract type PRealization{P} <: Realization end
abstract type TransientPRealization{P,T} <: Realization end

get_parameters(r::Realization) = r.params
num_parameters(r::Realization) = length(get_parameters(r))
Base.length(r::PRealization) = num_parameters(r)
get_times(r::TransientPRealization) = r.times[]
num_times(r::TransientPRealization) = length(get_times(r))
Base.length(r::TransientPRealization) = num_parameters(r)*num_times(r)

struct GenericPRealization{P} <: PRealization{P}
  params::P
end

struct GenericTransientPRealization{P,T} <: TransientPRealization{P,T}
  params::P
  times::Base.RefValue{T}
  function GenericTransientPRealization(params::P,times::T) where {P,T}
    new{P,T}(params,Ref(times))
  end
end

function get_initial_time(r::GenericTransientPRealization{P,T} where P) where T<:AbstractVector
  first(get_times(r))
end
function get_final_time(r::GenericTransientPRealization{P,T} where P) where T<:AbstractVector
  last(get_times(r))
end
function get_midpoint_time(r::GenericTransientPRealization{P,T} where P) where T<:AbstractVector
  (get_final_time(r) + get_initial_time(r)) / 2
end
function get_delta_time(r::GenericTransientPRealization{P,T} where P) where T<:AbstractVector
  (get_final_time(r) - get_initial_time(r)) / num_times(r)
end

function get_at_time(r::GenericTransientPRealization,time=:initial)
  params = get_parameters(r)
  if time == :initial
    GenericTransientPRealization(params,get_initial_time(r))
  elseif time == :midpoint
    GenericTransientPRealization(params,get_midpoint_time(r))
  elseif time == :final
    GenericTransientPRealization(params,get_final_time(r))
  else
    @notimplemented
  end
end

function change_time!(
  r::GenericTransientPRealization{P,T} where P,
  time::T
  ) where T

  r.times[] = time
end

struct GenericTrivialPRealization <: PRealization{<:AbstractVector{<:Number}}
  params::AbstractVector
end

GenericPRealization(p::AbstractVector{<:Number}) = GenericTrivialPRealization(p)

struct TrivialTransientPRealization <: TransientPRealization{<:AbstractVector{<:Number},<:Number}
  params::AbstractVector
  times::Number
end

function GenericTransientPRealization(p::TrivialTransientPRealization,t::Number)
  TrivialTransientPRealization(p,t)
end

num_parameters(r::GenericTrivialPRealization) = 1
get_times(r::TrivialTransientPRealization) = r.times
num_times(r::TrivialTransientPRealization) = 1

abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

abstract type AbstractParametricSpace{R} <: AbstractSet{R} end

struct ParametricSpace <: AbstractParametricSpace{GenericPRealization}
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
  GenericPRealization([generate_parameter(p) for i = 1:nparams])
end

struct TransientParametricSpace <: AbstractParametricSpace{GenericTransientPRealization}
  parametric_domain::AbstractVector
  temporal_domain::AbstractVector
  sampling_style::SamplingStyle
  function TransientParametricSpace(
    parametric_domain::AbstractVector{<:AbstractVector},
    temporal_domain::AbstractVector{<:Number},
    sampling=UniformSampling())

    new(parametric_domain,temporal_domain,sampling)
  end
end

function Base.show(io::IO,::MIME"text/plain",p::TransientParametricSpace)
  msg = "Set of tuples (p,t) in $(p.parametric_domain) × $(p.temporal_domain)"
  println(io,msg)
end

function realization(
  p::TransientParametricSpace;
  nparams=1,time_locations=eachindex(p.temporal_domain)
  )

  pspace = ParametricSpace(p.parametric_domain,p.realization)
  params = [generate_parameter(pspace) for i = 1:nparams]
  times = p.temporal_domain[time_locations]
  GenericTransientPRealization(params,times)
end

function shift_temporal_domain!(p::TransientParametricSpace,δ::Number)
  p.times .+= δ
end

abstract type AbstractPFunction{P} <: Function end

struct PFunction{P} <: AbstractPFunction{P}
  f::Function
  params::P
end

const 𝑓ₚ = PFunction

PFunction(f,p::AbstractVector{<:Number}) = f(p)
PFunction(f,r::GenericTrivialPRealization) = f(get_parameters(r))

struct TransientPFunction{P,T} <: AbstractPFunction{P}
  fun::Function
  params::P
  times::T
end

const 𝑓ₚₜ = TransientPFunction

TransientPFunction(f,p::AbstractVector{<:Number},t::Number) = f(p,t)
TransientPFunction(f,t::Number,p::AbstractVector{<:Number}) = f(p,t)
TransientPFunction(f,r::GenericTransientPRealization) = f(get_parameters(r),get_times(r))

get_parameters(f::AbstractPFunction) = f.params
num_parameters(f::AbstractPFunction) = num_parameters(get_parameters(f))
get_times(f::TransientPFunction) = f.times
num_times(f::TransientPFunction) = num_times(get_times(f))

function get_fields(f::PFunction)
  fields = GenericField[]
  params = get_parameters(f)
  for p = params
    push!(fields,GenericField(f.f(p)))
  end
  fields
end

function get_fields(f::TransientPFunction)
  fields = GenericField[]
  params = get_parameters(f)
  times = get_times(f)
  for (t,p) = Iterators.product(times,params)
    push!(fields,GenericField(f.f(p,t)))
  end
  fields
end
