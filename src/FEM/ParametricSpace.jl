struct PRealization{P<:AbstractVector}
  params::P
end

const TrivialPRealization = PRealization{AbstractVector{<:Number}}

get_parameters(r::PRealization) = r
num_parameters(r::PRealization) = length(r.params)
num_parameters(r::TrivialPRealization) = 1
Base.length(r::PRealization) = num_parameters(r)
Base.size(r::PRealization) = (length(r),)
Base.iterate(r::PRealization,iter...) = iterate(r.params,iter...)

function Base.convert(::Type{<:PRealization},p::TrivialPRealization)
  PRealization([p.params])
end

struct TransientPRealization{P<:PRealization,T}
  params::P
  times::Base.RefValue{T}
end

function TransientPRealization(params::PRealization,times::Union{Number,AbstractVector})
  TransientPRealization(params,Ref(times))
end

const TrivialTransientPRealization = TransientPRealization{TrivialPRealization,Number}

get_parameters(r::TransientPRealization) = get_parameters(r.params)
num_parameters(r::TransientPRealization) = num_parameters(r.params)
get_times(r::TransientPRealization) = r.times[]
num_times(r::TransientPRealization) = length(get_times(r))
Base.length(r::TransientPRealization) = num_parameters(r)*num_times(r)
Base.size(r::TransientPRealization) = (length(r),)
Base.iterate(r::TransientPRealization,iter...) = iterate(Iterators.product(r.times,r.params),iter...)

function Base.convert(
  ::Type{<:TransientPRealization},
  p::TrivialTransientPRealization)

  params = convert(PRealization,p.params)
  TransientPRealization(params,p.times)
end

function get_initial_time(r::TransientPRealization)
  first(get_times(r))
end

function get_final_time(r::TransientPRealization)
  last(get_times(r))
end

function get_midpoint_time(r::TransientPRealization)
  (get_final_time(r) + get_initial_time(r)) / 2
end

function get_delta_time(r::TransientPRealization)
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

function change_time!(r::TransientPRealization{P,T} where P,time::T) where T
  r.times[] = time
end

abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

struct ParametricSpace <: AbstractSet{PRealization}
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

struct TransientParametricSpace <: AbstractSet{TransientPRealization}
  parametric_space::ParametricSpace
  temporal_domain::AbstractVector
  function TransientParametricSpace(
    parametric_domain::AbstractVector{<:AbstractVector},
    temporal_domain::AbstractVector{<:Number},
    sampling=UniformSampling())

    parametric_space = ParametricSpace(parametric_domain,sampling)
    new(parametric_space,temporal_domain)
  end
end

function Base.show(io::IO,::MIME"text/plain",p::TransientParametricSpace)
  msg = "Set of tuples (p,t) in $(p.parametric_space.parametric_domain) × $(p.temporal_domain)"
  println(io,msg)
end

function realization(
  p::TransientParametricSpace;
  nparams=1,time_locations=eachindex(p.temporal_domain)
  )

  params = realization(p.parametric_space;nparams)
  times = p.temporal_domain[time_locations]
  TransientPRealization(params,times)
end

function shift_temporal_domain!(p::TransientParametricSpace,δ::Number)
  p.temporal_domain .+= δ
end

abstract type AbstractPFunction{P<:PRealization} <: Function end

struct PFunction{P} <: AbstractPFunction{P}
  f::Function
  params::P
end

const 𝑓ₚ = PFunction

function PFunction(f::Function,p::AbstractArray)
  @notimplemented "Use a PRealization as a parameter input"
end

function PFunction(f::Function,r::TrivialPRealization)
  f(r.params)
end

struct TransientPFunction{P,T} <: AbstractPFunction{P}
  fun::Function
  params::P
  times::T
end

const 𝑓ₚₜ = TransientPFunction

function TransientPFunction(f::Function,p::AbstractArray,t)
  @notimplemented "Use a PRealization as a parameter input"
end

function TransientPFunction(f::Function,p::TrivialPRealization,t::Number)
  f(p.params,t)
end

function TransientPFunction(f::Function,p::TrivialPRealization,t)
  TransientPFunction(f,convert(PRealization,p),t)
end

function get_fields(f::PFunction)
  fields = GenericField[]
  for p = f.params.params
    push!(fields,GenericField(f.f(p)))
  end
  fields
end

function get_fields(f::TransientPFunction)
  fields = GenericField[]
  for (t,p) = Iterators.product(f.times,f.params.params)
    push!(fields,GenericField(f.f(p,t)))
  end
  fields
end
