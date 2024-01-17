struct PRealization{P<:AbstractVector}
  params::P
end

const TrivialPRealization = PRealization{<:AbstractVector{<:Number}}

get_parameters(r::PRealization) = r # we only want to deal with a PRealization type
_get_parameters(r::PRealization) = r.params # this function should stay local
num_parameters(r::PRealization) = length(_get_parameters(r))
num_parameters(r::TrivialPRealization) = 1
Base.length(r::PRealization) = num_parameters(r)
Base.size(r::PRealization) = (length(r),)
Base.iterate(r::PRealization,iter...) = iterate(_get_parameters(r),iter...)

struct TransientPRealization{P<:PRealization,T}
  params::P
  times::Base.RefValue{T}
end

function TransientPRealization(params::PRealization,times::Union{Number,AbstractVector})
  TransientPRealization(params,Ref(times))
end

const TrivialTransientPRealization = TransientPRealization{<:TrivialPRealization,<:Number}

get_parameters(r::TransientPRealization) = get_parameters(r.params)
_get_parameters(r::TransientPRealization) = _get_parameters(r.params)
num_parameters(r::TransientPRealization) = num_parameters(r.params)
get_times(r::TransientPRealization) = r.times[]
num_times(r::TransientPRealization) = length(get_times(r))
Base.length(r::TransientPRealization) = num_parameters(r)*num_times(r)
Base.size(r::TransientPRealization) = (length(r),)

function Base.iterate(r::TransientPRealization,iter...)
  iterate(Iterators.product(get_times(r),_get_parameters(r)),iter...)
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

_get_parameters(f::AbstractPFunction) = _get_parameters(f.params)

struct PFunction{P} <: AbstractPFunction{P}
  f::Function
  params::P
end

const 𝑓ₚ = PFunction

function PFunction(f::Function,p::AbstractArray)
  @notimplemented "Use a PRealization as a parameter input"
end

function PFunction(f::Function,r::TrivialPRealization)
  f(_get_parameters(r))
end

struct TransientPFunction{P,T} <: AbstractPFunction{P}
  fun::Function
  params::P
  times::T
end

get_times(f::TransientPFunction) = f.times

const 𝑓ₚₜ = TransientPFunction

function TransientPFunction(f::Function,p::AbstractArray,t)
  @notimplemented "Use a PRealization as a parameter input"
end

function TransientPFunction(f::Function,r::TrivialPRealization,t::Number)
  f(_get_parameters(r),t)
end

function TransientPFunction(f::Function,r::TrivialPRealization,t)
  p = PRealization([_get_parameters(r)])
  TransientPFunction(f,p,t)
end

function get_fields(f::PFunction)
  fields = GenericField[]
  for p = _get_parameters(f)
    push!(fields,GenericField(f.fun(p)))
  end
  fields
end

function get_fields(f::TransientPFunction)
  fields = GenericField[]
  for (t,p) = Iterators.product(get_times(f),_get_parameters(f))
    push!(fields,GenericField(f.fun(p,t)))
  end
  fields
end

function Arrays.evaluate!(cache,f::AbstractPFunction,x...)
  map(g->g(x...),get_fields(f))
end

(f::AbstractPFunction)(x...) = evaluate(f,x...)

function test_parametric_space()
  α = PRealization(rand(10))
  β = PRealization([rand(10)])
  @check isa(α,TrivialPRealization)
  @check isa(α,PRealization{Vector{Float64}})
  @check isa(β,PRealization{Vector{Vector{Float64}}})
  γ = TransientPRealization(α,1)
  δ = TransientPRealization(α,1:10)
  ϵ = TransientPRealization(β,1:10)
  @check isa(γ,TrivialTransientPRealization)
  @check isa(δ,TransientPRealization{<:TrivialPRealization,UnitRange{Int}})
  @check isa(ϵ,TransientPRealization{PRealization{Vector{Vector{Float64}}},UnitRange{Int}})
  @check length(γ) == 1 && length(δ) == 10 && length(ϵ) == 10
  change_time!(ϵ,11:20)
  @check get_times(get_at_time(ϵ,:final)) == 20
  parametric_domain = [[1,10],[11,20]]
  p = ParametricSpace(parametric_domain)
  t = 1:10
  pt = TransientParametricSpace(parametric_domain,t)
  μ = realization(p)
  μt = realization(pt)
  @check isa(μ,PRealization) && isa(μt,TransientPRealization)
  a(x,μ,t) = sum(x)*sum(μ)*t
  a(μ,t) = x -> a(x,μ,t)
  da = ∂ₚt(a)
  aμt = 𝑓ₚₜ(a,get_parameters(μt),get_times(μt))
  daμt = ∂ₚt(aμt)
  @check isa(𝑓ₚₜ(a,α,t),Function)
  @check isa(aμt,AbstractPFunction)
  @check isa(daμt,AbstractPFunction)
  x = Point(1,2)
  Aμt = aμt(x)
  DAμt = daμt(x)
  for (i,(t,μ)) in enumerate(μt)
    @check Aμt[i] == a(μ,t)(x)
    @check DAμt[i] == da(μ,t)(x)
  end
end
