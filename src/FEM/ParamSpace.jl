struct ParamRealization{P<:AbstractVector}
  params::P
end

const TrivialParamRealization = ParamRealization{<:AbstractVector{<:Number}}

get_parameters(r::ParamRealization) = r # we only want to deal with a ParamRealization type
_get_parameters(r::ParamRealization) = r.params # this function should stay local
num_parameters(r::ParamRealization) = length(_get_parameters(r))
num_parameters(r::TrivialParamRealization) = 1
Base.length(r::ParamRealization) = num_parameters(r)
Base.size(r::ParamRealization) = (length(r),)
Arrays.testitem(r::ParamRealization) = testitem(_get_parameters(r))

# when iterating over a ParamRealization{P}, we return return eltype(P) ∀ index i
function Base.iterate(r::TrivialParamRealization)
  state = 1
  rstate = _get_parameters(r)
  return rstate, state
end

function Base.iterate(r::ParamRealization)
  state = 1
  rstate = _get_parameters(r)[state]
  return rstate, state
end

function Base.iterate(r::ParamRealization,state)
  state += 1
  if state > length(r)
    return nothing
  end
  rstate = _get_parameters(r)[state]
  return rstate, state
end

struct TransientParamRealization{P<:ParamRealization,T}
  params::P
  times::Base.RefValue{T}
end

function TransientParamRealization(params::ParamRealization,times::Union{Number,AbstractVector})
  TransientParamRealization(params,Ref(times))
end

const TrivialTransientParamRealization = TransientParamRealization{<:TrivialParamRealization,<:Number}

get_parameters(r::TransientParamRealization) = get_parameters(r.params)
_get_parameters(r::TransientParamRealization) = _get_parameters(r.params)
num_parameters(r::TransientParamRealization) = num_parameters(r.params)
get_times(r::TransientParamRealization) = r.times[]
num_times(r::TransientParamRealization) = length(get_times(r))
Base.length(r::TransientParamRealization) = num_parameters(r)*num_times(r)
Base.size(r::TransientParamRealization) = (length(r),)
Arrays.testitem(r::TransientParamRealization) = testitem(get_parameters(r)),testitem(get_times(r))

function Base.iterate(r::TransientParamRealization)
  iterator = Iterators.product(get_times(r),get_parameters(r))
  iternext = iterate(iterator)
  if isnothing(iternext)
    return nothing
  end
  (tstate,pstate),itstate = iternext
  state = (iterator,itstate)
  (pstate,tstate),state
end

function Base.iterate(r::TransientParamRealization,state)
  iterator,itstate = state
  iternext = iterate(iterator,itstate)
  if isnothing(iternext)
    return nothing
  end
  (tstate,pstate),itstate = iternext
  state = (iterator,itstate)
  (pstate,tstate),state
end

function get_initial_time(r::TransientParamRealization)
  first(get_times(r))
end

function get_final_time(r::TransientParamRealization)
  last(get_times(r))
end

function get_midpoint_time(r::TransientParamRealization)
  (get_final_time(r) + get_initial_time(r)) / 2
end

function get_delta_time(r::TransientParamRealization)
  (get_final_time(r) - get_initial_time(r)) / num_times(r)
end

function get_at_time(r::TransientParamRealization,time=:initial)
  params = get_parameters(r)
  if time == :initial
    TransientParamRealization(params,get_initial_time(r))
  elseif time == :midpoint
    TransientParamRealization(params,get_midpoint_time(r))
  elseif time == :final
    TransientParamRealization(params,get_final_time(r))
  else
    @notimplemented
  end
end

function change_time!(r::TransientParamRealization{P,T} where P,time::T) where T
  r.times[] = time
end

abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

struct ParamSpace <: AbstractSet{ParamRealization}
  parametric_domain::AbstractVector
  sampling_style::SamplingStyle
  function ParamSpace(
    parametric_domain::AbstractVector{<:AbstractVector},
    sampling=UniformSampling())

    new(parametric_domain,sampling)
  end
end

function Base.show(io::IO,::MIME"text/plain",p::ParamSpace)
  msg = "Set of parameters in $(p.parametric_domain), sampled with $(p.sampling_style)"
  println(io,msg)
end

function generate_parameter(p::ParamSpace)
  _value(d,::UniformSampling) = rand(Uniform(first(d),last(d)))
  _value(d,::NormalSampling) = rand(Normal(first(d),last(d)))
  [_value(d,p.sampling_style) for d = p.parametric_domain]
end

function realization(p::ParamSpace;nparams=1)
  ParamRealization([generate_parameter(p) for i = 1:nparams])
end

struct TransientParamSpace <: AbstractSet{TransientParamRealization}
  parametric_space::ParamSpace
  temporal_domain::AbstractVector
  function TransientParamSpace(
    parametric_domain::AbstractVector{<:AbstractVector},
    temporal_domain::AbstractVector{<:Number},
    sampling=UniformSampling())

    parametric_space = ParamSpace(parametric_domain,sampling)
    new(parametric_space,temporal_domain)
  end
end

function Base.show(io::IO,::MIME"text/plain",p::TransientParamSpace)
  msg = "Set of tuples (p,t) in $(p.parametric_space.parametric_domain) × $(p.temporal_domain)"
  println(io,msg)
end

function realization(
  p::TransientParamSpace;
  nparams=1,time_locations=eachindex(p.temporal_domain)
  )

  params = realization(p.parametric_space;nparams)
  times = p.temporal_domain[time_locations]
  TransientParamRealization(params,times)
end

function shift_temporal_domain!(p::TransientParamSpace,δ::Number)
  p.temporal_domain .+= δ
end

abstract type AbstractParamFunction{P<:ParamRealization} <: Function end

struct ParamFunction{P} <: AbstractParamFunction{P}
  fun::Function
  params::P
end

get_parameters(f::ParamFunction) = get_parameters(f.params)
_get_parameters(f::ParamFunction) = _get_parameters(f.params)
num_parameters(f::ParamFunction) = length(_get_parameters(f))
Base.length(f::ParamFunction) = num_parameters(f)
Base.size(f::ParamFunction) = (length(f),)
Arrays.testitem(f::ParamFunction) = f.fun(testitem(f.params))

function Fields.gradient(f::ParamFunction)
  function _gradient(x,μ)
    gradient(f.fun(μ))(x)
  end
  _gradient(μ) = x -> _gradient(x,μ)
  ParamFunction(_gradient,f.params)
end

# when iterating over a ParamFunction{P}, we return return f(eltype(P)) ∀ index i
function Base.iterate(f::ParamFunction,state...)
  riter = iterate(get_parameters(f),state...)
  if isnothing(riter)
    return nothing
  end
  rstate,statenext = riter
  return f.fun(rstate),statenext
end

const 𝑓ₚ = ParamFunction

function ParamFunction(f::Function,p::AbstractArray)
  @notimplemented "Use a ParamRealization as a parameter input"
end

function ParamFunction(f::Function,r::TrivialParamRealization)
  f(_get_parameters(r))
end

struct TransientParamFunction{P,T} <: AbstractParamFunction{P}
  fun::Function
  params::P
  times::T
end

get_parameters(f::TransientParamFunction) = get_parameters(f.params)
_get_parameters(f::TransientParamFunction) = _get_parameters(f.params)
num_parameters(f::TransientParamFunction) = length(_get_parameters(f))
get_times(f::TransientParamFunction) = f.times
num_times(f::TransientParamFunction) = length(get_times(f))
Base.length(f::TransientParamFunction) = num_parameters(f)*num_times(f)
Base.size(f::TransientParamFunction) = (length(f),)
Arrays.testitem(f::TransientParamFunction) = f.fun(testitem(f.params),testitem(f.times))

function Fields.gradient(f::TransientParamFunction)
  function _gradient(x,μ,t)
    gradient(f.fun(μ,t))(x)
  end
  _gradient(μ,t) = x -> _gradient(x,μ,t)
  TransientParamFunction(_gradient,f.params,f.times)
end

function Base.iterate(f::TransientParamFunction)
  iterator = Iterators.product(get_times(f),get_parameters(f))
  (tstate,pstate),state = iterate(iterator)
  iterstatenext = iterator,state
  f.fun(pstate,tstate),iterstatenext
end

function Base.iterate(f::TransientParamFunction,iterstate)
  iterator,state = iterstate
  statenext = iterate(iterator,state)
  if isnothing(statenext)
    return nothing
  end
  (tstate,pstate),state = statenext
  iterstatenext = iterator,state
  f.fun(pstate,tstate),iterstatenext
end

const 𝑓ₚₜ = TransientParamFunction

function TransientParamFunction(f::Function,p::AbstractArray,t)
  @notimplemented "Use a ParamRealization as a parameter input"
end

function TransientParamFunction(f::Function,r::TrivialParamRealization,t::Number)
  f(_get_parameters(r),t)
end

function TransientParamFunction(f::Function,r::TrivialParamRealization,t)
  p = ParamRealization([_get_parameters(r)])
  TransientParamFunction(f,p,t)
end

function get_fields(f::AbstractParamFunction,type=:GenericField;N=1)
  if type == :GenericField
    map(f) do fi
      GenericField(fi)
    end
  elseif type == :ZeroField
    map(f) do fi
      ZeroField(fi)
    end
  elseif type == :FieldGradient
    map(f) do fi
      FieldGradient{N}(fi)
    end
  else
    @notimplemented
  end
end

function Arrays.evaluate!(cache,f::AbstractParamFunction,x...)
  map(g->g(x...),f)
end

(f::AbstractParamFunction)(x...) = evaluate(f,x...)

function test_parametric_space()
  α = ParamRealization(rand(10))
  β = ParamRealization([rand(10)])
  @test isa(α,TrivialParamRealization)
  @test isa(α,ParamRealization{Vector{Float64}})
  @test isa(β,ParamRealization{Vector{Vector{Float64}}})
  γ = TransientParamRealization(α,1)
  δ = TransientParamRealization(α,1:10)
  ϵ = TransientParamRealization(β,1:10)
  @test isa(γ,TrivialTransientParamRealization)
  @test isa(δ,TransientParamRealization{<:TrivialParamRealization,UnitRange{Int}})
  @test isa(ϵ,TransientParamRealization{ParamRealization{Vector{Vector{Float64}}},UnitRange{Int}})
  @test length(γ) == 1 && length(δ) == 10 && length(ϵ) == 10
  change_time!(ϵ,11:20)
  @test get_times(get_at_time(ϵ,:final)) == 20
  parametric_domain = [[1,10],[11,20]]
  p = ParamSpace(parametric_domain)
  t = 1:10
  pt = TransientParamSpace(parametric_domain,t)
  μ = realization(p)
  μt = realization(pt)
  @test isa(μ,ParamRealization) && isa(μt,TransientParamRealization)
  a(x,t) = sum(x)*t^2*sin(t)
  a(t) = x -> a(x,t)
  da = ∂t(a)
  aμ(x,μ,t) = sum(μ)*a(x,t)
  aμ(μ,t) = x -> aμ(x,μ,t)
  aμt = 𝑓ₚₜ(aμ,get_parameters(μt),get_times(μt))
  daμt = ∂t(aμt)
  @test isa(𝑓ₚₜ(a,α,t),Function)
  @test isa(aμt,AbstractParamFunction)
  @test isa(daμt,AbstractParamFunction)
  x = Point(1,2)
  aμtx = aμt(x)
  daμtx = daμt(x)
  for (i,(μ,t)) in enumerate(μt)
    @test aμtx[i] == a(t)(x)*sum(μ)
    @test daμtx[i] == da(t)(x)*sum(μ)
  end
  b(x,μ) = sum(x)*sum(μ)
  b(μ) = x -> b(x,μ)
  bμ = 𝑓ₚ(b,get_parameters(μ))
  bμx = bμ(x)
  for (i,bμi) in enumerate(bμ)
    @test bμi(x) == bμx[i]
  end
  for (i,aμti) in enumerate(aμt)
    @test aμti(x) == aμtx[i]
  end
  for (i,daμti) in enumerate(daμt)
    @test daμti(x) == daμtx[i]
  end
end
