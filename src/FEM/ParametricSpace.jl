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
Arrays.testitem(r::PRealization) = testitem(_get_parameters(r))

# when iterating over a PRealization{P}, we return return eltype(P) ∀ index i
function Base.iterate(r::TrivialPRealization)
  state = 1
  rstate = _get_parameters(r)
  return rstate, state
end

function Base.iterate(r::PRealization)
  state = 1
  rstate = _get_parameters(r)[state]
  return rstate, state
end

function Base.iterate(r::PRealization,state)
  state += 1
  if state > length(r)
    return nothing
  end
  rstate = _get_parameters(r)[state]
  return rstate, state
end

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
Arrays.testitem(r::TransientPRealization) = testitem(get_parameters(r)),testitem(get_times(r))

function Base.iterate(r::TransientPRealization)
  iterator = Iterators.product(get_times(r),get_parameters(r))
  iternext = iterate(iterator)
  if isnothing(iternext)
    return nothing
  end
  (tstate,pstate),itstate = iternext
  state = (iterator,itstate)
  (pstate,tstate),state
end

function Base.iterate(r::TransientPRealization,state)
  iterator,itstate = state
  iternext = iterate(iterator,itstate)
  if isnothing(iternext)
    return nothing
  end
  (tstate,pstate),itstate = iternext
  state = (iterator,itstate)
  (pstate,tstate),state
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
  fun::Function
  params::P
end

get_parameters(f::PFunction) = get_parameters(f.params)
_get_parameters(f::PFunction) = _get_parameters(f.params)
num_parameters(f::PFunction) = length(_get_parameters(f))
Base.length(f::PFunction) = num_parameters(f)
Base.size(f::PFunction) = (length(f),)
Arrays.testitem(f::PFunction) = f.fun(testitem(f.params))

function Fields.gradient(f::PFunction)
  function _gradient(x,μ)
    gradient(f.fun(μ))(x)
  end
  _gradient(μ) = x -> _gradient(x,μ)
  PFunction(_gradient,f.params)
end

# when iterating over a PFunction{P}, we return return f(eltype(P)) ∀ index i
function Base.iterate(f::PFunction,state...)
  riter = iterate(get_parameters(f),state...)
  if isnothing(riter)
    return nothing
  end
  rstate,statenext = riter
  return f.fun(rstate),statenext
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

get_parameters(f::TransientPFunction) = get_parameters(f.params)
_get_parameters(f::TransientPFunction) = _get_parameters(f.params)
num_parameters(f::TransientPFunction) = length(_get_parameters(f))
get_times(f::TransientPFunction) = f.times
num_times(f::TransientPFunction) = length(get_times(f))
Base.length(f::TransientPFunction) = num_parameters(f)*num_times(f)
Base.size(f::TransientPFunction) = (length(f),)
Arrays.testitem(f::TransientPFunction) = f.fun(testitem(f.params),testitem(f.times))

function Fields.gradient(f::TransientPFunction)
  function _gradient(x,μ,t)
    gradient(f.fun(μ,t))(x)
  end
  _gradient(μ,t) = x -> _gradient(x,μ,t)
  TransientPFunction(_gradient,f.params,f.times)
end

function Base.iterate(f::TransientPFunction)
  iterator = Iterators.product(get_times(f),get_parameters(f))
  (tstate,pstate),state = iterate(iterator)
  iterstatenext = iterator,state
  f.fun(pstate,tstate),iterstatenext
end

function Base.iterate(f::TransientPFunction,iterstate)
  iterator,state = iterstate
  statenext = iterate(iterator,state)
  if isnothing(statenext)
    return nothing
  end
  (tstate,pstate),state = statenext
  iterstatenext = iterator,state
  f.fun(pstate,tstate),iterstatenext
end

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

function get_fields(f::AbstractPFunction,type=:GenericField;N=1)
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

function Arrays.evaluate!(cache,f::AbstractPFunction,x...)
  map(g->g(x...),f)
end

(f::AbstractPFunction)(x...) = evaluate(f,x...)

function test_parametric_space()
  α = PRealization(rand(10))
  β = PRealization([rand(10)])
  @test isa(α,TrivialPRealization)
  @test isa(α,PRealization{Vector{Float64}})
  @test isa(β,PRealization{Vector{Vector{Float64}}})
  γ = TransientPRealization(α,1)
  δ = TransientPRealization(α,1:10)
  ϵ = TransientPRealization(β,1:10)
  @test isa(γ,TrivialTransientPRealization)
  @test isa(δ,TransientPRealization{<:TrivialPRealization,UnitRange{Int}})
  @test isa(ϵ,TransientPRealization{PRealization{Vector{Vector{Float64}}},UnitRange{Int}})
  @test length(γ) == 1 && length(δ) == 10 && length(ϵ) == 10
  change_time!(ϵ,11:20)
  @test get_times(get_at_time(ϵ,:final)) == 20
  parametric_domain = [[1,10],[11,20]]
  p = ParametricSpace(parametric_domain)
  t = 1:10
  pt = TransientParametricSpace(parametric_domain,t)
  μ = realization(p)
  μt = realization(pt)
  @test isa(μ,PRealization) && isa(μt,TransientPRealization)
  a(x,t) = sum(x)*t^2*sin(t)
  a(t) = x -> a(x,t)
  da = ∂t(a)
  aμ(x,μ,t) = sum(μ)*a(x,t)
  aμ(μ,t) = x -> aμ(x,μ,t)
  aμt = 𝑓ₚₜ(aμ,get_parameters(μt),get_times(μt))
  daμt = ∂t(aμt)
  @test isa(𝑓ₚₜ(a,α,t),Function)
  @test isa(aμt,AbstractPFunction)
  @test isa(daμt,AbstractPFunction)
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
